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
#include "paddle/fluid/operators/fused/cublas_gemm_ex.h"
#include "paddle/fluid/platform/float16.h"

// #define TRANSPOSE_GEMM
// #define MULTI_STREAM

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
                                                 int n,
                                                 const int num_streams)
{
    // const float scale = __ldg(scale_ptr);
    int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    int m_id = blockIdx.y * blockDim.y + threadIdx.y;
    int tmp = n / 4;
    int stream_id = n_id / tmp; // 0,1,2,..., num_streams-1

    int out_id_tmp = (m * stream_id + m_id - stream_id) * tmp + n_id; // split into num_streams submatrix
    int stream_offset = m * stream_id * tmp;                    // offset of the current submatrix
    int stream_m_id = (out_id_tmp - stream_offset) / tmp;           // m_id in the submatrix
    int stream_n_id = (out_id_tmp - stream_offset) % tmp;           // n_id in the submatrix
    int output_id = stream_offset + ((stream_n_id & 0xffffffe0) * m + (stream_m_id << 5) + (stream_n_id & 31)) >> 2;

    bool check = ((m_id < m) && (n_id < n));
    if (check) {
        char4 tmp;
        tmp.x = __float2int_rn(static_cast<float>(input[m_id * n + n_id]) * 1.0);
        tmp.y = __float2int_rn(static_cast<float>(input[m_id * n + n_id+1]) * 1.0);
        tmp.z = __float2int_rn(static_cast<float>(input[m_id * n + n_id+2]) * 1.0);
        tmp.w = __float2int_rn(static_cast<float>(input[m_id * n + n_id+3]) * 1.0);
        // COL32_col = n_id >> 5 ; COL32_row = (m_id << 5) + (n_id & 31)
        // COL32_idx = (COL32_col << 5) * m + COL32_row = (n_id & 0xffffffe0)*m + (m_id << 5) + (n_id & 31)
        output[output_id] = tmp;
    }
}

template <typename T>
void row_major_to_col32_quantize_kernelLauncher(const T* input,
                                                int8_t* output,
                                                // T* scale,
                                                const int m,
                                                const int n,
                                                cudaStream_t stream,
                                                const int num_streams) {
//   std::cout << "row-major-to-col32: m: " << m << " n: " << n << std::endl;

    dim3 grid((m + 31) / 32, (n + 31) / 32);
    dim3 block(32, 32);

    row_major_to_col32_quantize_kernel<<<grid, block, 0, stream>>>(
    input,
    (char4*)output,
    m,
    n,
    num_streams);
}

// convert COL32 to row-major 
// and dequantize using weight scales and input scales
template <typename T>
__global__ void col32_to_row_major_dequantize_kernel(T* output,
                                                  int32_t* input,
                                                  const int m,  // hidden
                                                  const int n,  // batch size
                                                  const float max_range,
                                                  const int num_streams) 
{

  int m_id = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int n_id = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    int input_index = (m_id & 0xffffffe0) * n + (n_id << 5) + (m_id & 31);
    for (int s=1; s<num_streams; s++){
        input[input_index] = input[input_index] + input[input_index+s*m*n];
    }

    output[n_id * m + m_id] =
        static_cast<T>(static_cast<float>(input[input_index]) *
            1.0 / max_range * 1.0 / max_range);
  }
}

template <typename T>
void col32_to_row_major_dequantize_kernelLauncher(int32_t* input,
                                                  T* output,
                                                  const int batch_size, // m
                                                  const int hidden_units,  // n
                                                  cudaStream_t stream,
                                                  const int num_streams) {
                                                      
  dim3 grid((hidden_units + 31) / 32, (batch_size + 31) / 32);
  dim3 block(32, 32);

  // std::cout << "col32-to-row-major: m: " << batch_size << " n: " << hidden_units << std::endl;

//   dim3 grid(1, 512);
//   dim3 block(32, 4);

//   int repeat = batch_size * hidden_units / 65536;

  col32_to_row_major_dequantize_kernel<<<grid, block, 0, stream>>>(
      output, input, hidden_units, batch_size, 127.0f, num_streams);
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
#ifdef TRANSPOSE_GEMM
        if (k_ == 4 * m_ && k_ == 16384) {
#else
        if (k_ == 4 * n_ && k_ == 16384) {
#endif
#else
        if (false) {
#endif
            // Use multi stream calculation
            for (int i = 0; i < 4; ++i) {
                auto helper = std::make_shared<CublasHelper>(m, k / 4, n);
                helpers_.emplace_back(helper);
            }
        } else {
            auto helper = std::make_shared<CublasHelper>(m, k, n);
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
                        cudaEvent_t* stream_events){
        int m = m_, k = k_, n = n_;
        //quant transpose A
        float scale = 1.0f;
        int num_streams = 1;
#ifdef MULTI_STREAM
        num_streams = 4;
#endif        
        VLOG(1) << "[DEBUG] row_major_to_col32_quantize_kernelLauncher";
        row_major_to_col32_quantize_kernelLauncher<T>(input->data<T>(), 
                                                      input_tmp->data<int8_t>(), 
#ifdef TRANSPOSE_GEMM
                                                        n_, k_,
#else
                                                        m_, k_,
#endif
                                                      
                                                      dev_ctx_.stream(), num_streams);

        VLOG(1) << "[DEBUG] GEMM";
        VLOG(1) << "input_tmp " << input_tmp->numel() << " dtype " << input_tmp->dtype();
        VLOG(1) << "weight_tmp " << weight->numel() << " dtype " << weight->dtype();
        VLOG(1) << "output_tmp " << output_tmp->numel() << " dtype " << output_tmp->dtype();
        
#ifdef MULTI_STREAM
#ifdef TRANSPOSE_GEMM
        if (k_ == 4 * m_ && k_ == 16384) {
#else
        if (k_ == 4 * n_ && k_ == 16384) {
#endif
#else
        if (false) {
#endif
            // Synchronize stream
            // PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(stream_events[0], dev_ctx_.stream()));
            // PADDLE_ENFORCE_GPU_SUCCESS(
            //         cudaStreamWaitEvent(streams[0], stream_events[0], 0));
            // PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(stream_events[1], streams[0]));
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
           // for (int i = 1; i < 4; ++i) {
           //      PADDLE_ENFORCE_GPU_SUCCESS(
           //          cudaStreamWaitEvent(streams[i], stream_events[1], 0));
           // }
            // Use multi stream calculation
            for (int i = 0; i < 4; ++i) {
                helpers_[i] -> GEMM(input_tmp->data<int8_t>(), weight->data<int8_t>(), output_tmp->data<int32_t>(), streams[i]);
                // helpers_[i] -> GEMM(input_tmp->data<int8_t>(), weight->data<int8_t>(), output_tmp->data<int32_t>(), dev_ctx_.stream());
                // PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(stream_events[i+1], streams[i]));
            }
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
            // for (int i = 0; i < 4; ++i)
            //     PADDLE_ENFORCE_GPU_SUCCESS(
            //         cudaStreamWaitEvent(dev_ctx_.stream(), stream_events[i+1], 0));
            // Reduce
            // ...
            // VLOG(1) << "REDUCE";
            // int block = 1024;
            // int grid = (m_ * n_ * 4 + block - 1) / block;
            // reduce<<<grid, block, 0, dev_ctx_.stream()>>>(output_tmp->data<int32_t>(), m_, n_, 4);

        } else {
            helpers_[0] -> GEMM(input_tmp->data<int8_t>(), weight->data<int8_t>(), output_tmp->data<int32_t>(), dev_ctx_.stream());
        }

        

        //dequant C
        VLOG(1) << "[DEBUG] col32_to_row_major_dequantize_kernelLauncher";
        col32_to_row_major_dequantize_kernelLauncher<T>(output_tmp->data<int32_t>(), 
                                                        output->data<T>(), 
#ifdef TRANSPOSE_GEMM
                                                        n_, m_,
#else
                                                        m_, n_, 
#endif
                                                        dev_ctx_.stream(), num_streams);

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
    // std::vector<std::shared_ptr<CublasLtHelper>> helpers_;
    std::vector<std::shared_ptr<CublasHelper>> helpers_;

};


}  // namespace operators
}  // namespace paddle
