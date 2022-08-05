/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/operators/fused/cublasLt_helper.h"
#include "paddle/fluid/platform/float16.h"

#define MULTI_STREAM

namespace paddle {
namespace operators {

static inline __device__ int8_t float_to_int8_rn(float x)
{
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;"
               : "=r"(dst)
               : "f"(x));
  return reinterpret_cast<const int8_t &>(dst);
}

// transpose matrix & transfrom row-major to COL32 & quantize
// input matrix is (m, n) row-major
// output matrix is (n, m) COL32, using char4 to write out
// m should be a mutiple of 32
// grid((m+31)/32, (n+31)/32)
// block(8, 32)
// But have confusion about how the scale is used.
template <typename T>
__global__ void row_major_to_col32_quantize_kernel(const T* input,
                                                 char4* output,
                                                 int m,
                                                 int n)
{
    // const float scale = __ldg(scale_ptr);
    
    int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    int m_id = blockIdx.y * blockDim.y + threadIdx.y;

    bool check = ((m_id < m) && (n_id < n));
    if (check) {
        char4 tmp;
        tmp.x = __float2int_rn(static_cast<float>(input[m_id * n + n_id]) * 1.0);
        tmp.y = __float2int_rn(static_cast<float>(input[m_id * n + n_id+1]) * 1.0);
        tmp.z = __float2int_rn(static_cast<float>(input[m_id * n + n_id+2]) * 1.0);
        tmp.w = __float2int_rn(static_cast<float>(input[m_id * n + n_id+3]) * 1.0);
        // COL32_col = n_id >> 5 ; COL32_row = (m_id << 5) + (n_id & 31)
        // COL32_idx = (COL32_col << 5) * m + COL32_row = (n_id & 0xffffffe0)*m + (m_id << 5) + (n_id & 31)
        output[((n_id & 0xffffffe0) * m + (m_id << 5) + (n_id & 31)) >> 2] = tmp;
    }
    
}

template <typename T>
void row_major_to_col32_quantize_kernelLauncher(const T* input,
                                                int8_t* output,
                                                // T* scale,
                                                const int m,
                                                const int n,
                                                cudaStream_t stream) {
  dim3 grid((m + 31) / 32, (n + 31) / 32);
  dim3 block(32, 32);

  row_major_to_col32_quantize_kernel<<<grid, block, 0, stream>>>(
      input,
      (char4*)output,
      m,
      n);
}

// convert COL32 to row-major 
// and dequantize using weight scales and input scales
template <typename T>
__global__ void col32_to_row_major_dequantize_kernel(T* output,
                                                  const int32_t* input,
                                                  const int m,  // hidden
                                                  const int n,  // batch size
                                                  const float max_range) 
{
  int m_id = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int n_id = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    // int tmp = m_id * n + n_id;
    // printf("%d, %d, %d, %d\n", m_id, m, n_id, n);
    output[n_id * m + m_id] =
        static_cast<T>(static_cast<float>(input[(m_id & 0xffffffe0) * n + (n_id << 5) + (m_id & 31)]) *
            1.0 / max_range * 1.0 / max_range);
  }
}

template <typename T>
void col32_to_row_major_dequantize_kernelLauncher(const int32_t* input,
                                                  T* output,
                                                  const int batch_size, // m
                                                  const int hidden_units,  // n
                                                  cudaStream_t stream) {
  dim3 grid((hidden_units + 31) / 32, (batch_size + 31) / 32);
  dim3 block(32, 32);

  col32_to_row_major_dequantize_kernel<<<grid, block, 0, stream>>>(
      output, input, hidden_units, batch_size, 127.0f);
}


__global__ void reduce(int32_t* data, const int m, const int n, const int num_streams) {
    int num_threads = m * n * num_streams;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int num_items = (m * n);
    int stream = tid / num_items;
    if (stream == 0) return;
    int item = tid % num_items;

    for (; tid < num_threads; tid += stride) {
        atomicAdd(&data[item], data[tid]);
    }
}

template <typename T>
class AttnMatmulINT8 {
public:
    AttnMatmulINT8(
            const platform::CUDADeviceContext& dev_ctx,
             int m,
             int n,
             int k,
             bool compute_bias,
             const std::vector<const framework::Tensor*> weight_cts,
             const platform::Place& place,
             bool is_ffn2=false)
        :dev_ctx_(dev_ctx),
        m_(m),
        n_(n),
        k_(k),
        compute_bias_(compute_bias) {
#ifdef MULTI_STREAM
        if (k_ == 4 * n_ && k_ == 16384) {
#else
        if (false) {
#endif
            // Use multi stream calculation
            for (int i = 0; i < 4; ++i) {
                auto helper = std::make_shared<CublasLtHelper>(m, k / 4, n);
                helpers_.emplace_back(helper);
            }
        } else {
            auto helper = std::make_shared<CublasLtHelper>(m, k, n);
            helpers_.emplace_back(helper);
        }

        //quantize and transpose weight
        // for (const framework::Tensor* weight_ct : weight_cts) {
        //     framework::Tensor weight_tmp;
        //     framework::Tensor* weight_t = const_cast<framework::Tensor*>(weight_ct);
        //     framework::TensorCopy(*weight_t, place, &weight_tmp);

        //     weight_t->Resize({k*n});
        //     weight_t->mutable_data<int8_t>(place);
            
        //     VLOG(1) << "[DEBUG] TransformB";
        //     helper_->TransformB(weight_t, place, dev_ctx_.stream());
        // }
    }
    ~AttnMatmulINT8(){}
    void ComputeForward(const framework::Tensor* weight, // [int8] which has been transformed in pass
                        const framework::Tensor* input, // [fp16/32] 
                        framework::Tensor* input_tmp, // [int8]  workspace
                        const framework::Tensor* bias, //[fp16/32] 
                        framework::Tensor* output, // [fp16/32] has been dequantized/detranspose/detranbsform
                        framework::Tensor* output_tmp, //[int32]  workspace
                        framework::Tensor* bias_out,
                        cudaStream_t* streams,
                        cudaEvent_t* stream_events
                        ){
        int m = m_, k = k_, n = n_;
        //quant transpose A
        float scale = 1.0f;
        VLOG(1) << "[DEBUG] row_major_to_col32_quantize_kernelLauncher";
        row_major_to_col32_quantize_kernelLauncher<T>(input->data<T>(), 
                                                      input_tmp->data<int8_t>(), 
                                                        m_, k_,                     
                                                      dev_ctx_.stream());

        VLOG(1) << "[DEBUG] GEMM";
        VLOG(1) << "input_tmp " << input_tmp->numel() << " dtype " << input_tmp->dtype();
        VLOG(1) << "weight_tmp " << weight->numel() << " dtype " << weight->dtype();
        VLOG(1) << "output_tmp " << output_tmp->numel() << " dtype " << output_tmp->dtype();
#ifdef MULTI_STREAM
        if (k_ == 4 * n_ && k_ == 16384) {
#else
        if (false) {
#endif
            // Synchronize stream
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
           
            // Use multi stream calculation
            for (int i = 0; i < 4; ++i) {
                helpers_[i] -> GEMM(input_tmp->data<int8_t>(), weight->data<int8_t>(), output_tmp->data<int32_t>(), streams[i]);
            }
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

        } else {
            helpers_[0] -> GEMM(input_tmp->data<int8_t>(), weight->data<int8_t>(), output_tmp->data<int32_t>(), dev_ctx_.stream());
        }

        

        //dequant C
        VLOG(1) << "[DEBUG] col32_to_row_major_dequantize_kernelLauncher";
        col32_to_row_major_dequantize_kernelLauncher<T>(output_tmp->data<int32_t>(), 
                                                        output->data<T>(), 
                                                        m_, n_, 
                                                        dev_ctx_.stream());

        if (compute_bias_) {
            // bias_out = output + bias
            VLOG(1) << "[DEBUG] compute_bias_";
            std::vector<const Tensor*> ins = {output, bias};
            std::vector<Tensor*> outs = {bias_out};
            phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
            dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
            PADDLE_ENFORCE_EQ(cudaGetLastError(), cudaSuccess, platform::errors::Fatal("Add"));
        }
    }

    void ComputeForwardWoQDQ(const framework::Tensor* weight, // [int8] which has been transformed in pass
                        framework::Tensor* input, // [int8]  workspace
                        const framework::Tensor* bias, //[fp16/32] 
                        framework::Tensor* output, //[int32]  workspace
                        framework::Tensor* bias_out,
                        cudaStream_t* streams,
                        cudaEvent_t* stream_events){
        int m = m_, k = k_, n = n_;

        VLOG(1) << "[DEBUG] GEMM";
        VLOG(1) << "input " << input->numel() << " dtype " << input->dtype();
        VLOG(1) << "weight " << weight->numel() << " dtype " << weight->dtype();
        VLOG(1) << "output " << output->numel() << " dtype " << output->dtype();
#ifdef MULTI_STREAM
        if (k_ == 4 * n_ && k_ == 16384) {
#else
        if (false) {
#endif
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
           
            // Use multi stream calculation
            for (int i = 0; i < 4; ++i) {
                helpers_[i] -> GEMM(input->data<int8_t>(), weight->data<int8_t>(), output->data<int32_t>(), streams[i]);
            }
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

        } else {
            helpers_[0] -> GEMM(input->data<int8_t>(), weight->data<int8_t>(), output->data<int32_t>(), dev_ctx_.stream());
        }

        if (compute_bias_) {
            // bias_out = output + bias
            VLOG(1) << "[DEBUG] compute_bias_";
            std::vector<const Tensor*> ins = {output, bias};
            std::vector<Tensor*> outs = {bias_out};
            phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
            dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
            PADDLE_ENFORCE_EQ(cudaGetLastError(), cudaSuccess, platform::errors::Fatal("Add"));
        }
    }

    void ComputeForwardWoQ(const framework::Tensor* weight, // [int8] which has been transformed in pass
                        framework::Tensor* input, // [int8] 
                        const framework::Tensor* bias, //[fp16/32] 
                        framework::Tensor* output, // [fp16/32] has been dequantized/detranspose/detranbsform
                        framework::Tensor* output_tmp, //[int32]  workspace
                        framework::Tensor* bias_out,
                        cudaStream_t* streams,
                        cudaEvent_t* stream_events){
        int m = m_, k = k_, n = n_;

        VLOG(1) << "[DEBUG] GEMM";
        VLOG(1) << "input_tmp " << input->numel() << " dtype " << input->dtype();
        VLOG(1) << "weight_tmp " << weight->numel() << " dtype " << weight->dtype();
        VLOG(1) << "output_tmp " << output_tmp->numel() << " dtype " << output_tmp->dtype();
#ifdef MULTI_STREAM
        if (k_ == 4 * n_ && k_ == 16384) {
#else
        if (false) {
#endif
            // Synchronize stream
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
           
            // Use multi stream calculation
            for (int i = 0; i < 4; ++i) {
                helpers_[i] -> GEMM(input->data<int8_t>(), weight->data<int8_t>(), output_tmp->data<int32_t>(), streams[i]);
            }
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

        } else {
            helpers_[0] -> GEMM(input->data<int8_t>(), weight->data<int8_t>(), output_tmp->data<int32_t>(), dev_ctx_.stream());
        }

        

        //dequant C
        VLOG(1) << "[DEBUG] col32_to_row_major_dequantize_kernelLauncher";
        col32_to_row_major_dequantize_kernelLauncher<T>(output_tmp->data<int32_t>(), 
                                                        output->data<T>(), 
                                                        m_, n_, 
                                                        dev_ctx_.stream());

        if (compute_bias_) {
            // bias_out = output + bias
            VLOG(1) << "[DEBUG] compute_bias_";
            std::vector<const Tensor*> ins = {output, bias};
            std::vector<Tensor*> outs = {bias_out};
            phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
            dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
            PADDLE_ENFORCE_EQ(cudaGetLastError(), cudaSuccess, platform::errors::Fatal("Add"));
        }
    }

    void ComputeForwardWoDQ(const framework::Tensor* weight, // [int8] which has been transformed in pass
                        const framework::Tensor* input, // [fp16/32] 
                        framework::Tensor* input_tmp, // [int8]  workspace
                        const framework::Tensor* bias, //[fp16/32] 
                        framework::Tensor* output, // [int32] 
                        framework::Tensor* bias_out,
                        cudaStream_t* streams,
                        cudaEvent_t* stream_events
                        ){
        int m = m_, k = k_, n = n_;
        //quant transpose A
        float scale = 1.0f;
        VLOG(1) << "[DEBUG] row_major_to_col32_quantize_kernelLauncher";
        row_major_to_col32_quantize_kernelLauncher<T>(input->data<T>(), 
                                                      input_tmp->data<int8_t>(), 
                                                        m_, k_, 
                                                      dev_ctx_.stream());

        VLOG(1) << "[DEBUG] GEMM";
        VLOG(1) << "input_tmp " << input_tmp->numel() << " dtype " << input_tmp->dtype();
        VLOG(1) << "weight_tmp " << weight->numel() << " dtype " << weight->dtype();
        VLOG(1) << "output_tmp " << output->numel() << " dtype " << output->dtype();
#ifdef MULTI_STREAM
        if (k_ == 4 * n_ && k_ == 16384) {
#else
        if (false) {
#endif
            // Synchronize stream
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
           
            // Use multi stream calculation
            for (int i = 0; i < 4; ++i) {
                helpers_[i] -> GEMM(input_tmp->data<int8_t>(), weight->data<int8_t>(), output->data<int32_t>(), streams[i]);
            }
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

        } else {
            helpers_[0] -> GEMM(input_tmp->data<int8_t>(), weight->data<int8_t>(), output->data<int32_t>(), dev_ctx_.stream());
        }

        //TODO: add bias in in col32-int32 format
        if (compute_bias_) {
            // bias_out = output + bias
            VLOG(1) << "[DEBUG] compute_bias_";
            std::vector<const Tensor*> ins = {output, bias};
            std::vector<Tensor*> outs = {bias_out};
            phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
            dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
            PADDLE_ENFORCE_EQ(cudaGetLastError(), cudaSuccess, platform::errors::Fatal("Add"));
        }
    }

private:
    const platform::CUDADeviceContext& dev_ctx_;

    int m_; // m
    int n_; // n
    int k_; // k

    int compute_bias_;
    std::vector<std::shared_ptr<CublasLtHelper>> helpers_;

};


}  // namespace operators
}  // namespace paddle
