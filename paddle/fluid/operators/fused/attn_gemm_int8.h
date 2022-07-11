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

#include "paddle/fluid/operators/fused/cublasLt_helper.h"
#include "paddle/fluid/platform/float16.h"


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
        tmp.x = __float2int_rn(static_cast<float>(__ldg(input + m_id * n + n_id) * 1.0));
        tmp.y = __float2int_rn(static_cast<float>(__ldg(input + m_id * n + n_id+1) * 1.0));
        tmp.z = __float2int_rn(static_cast<float>(__ldg(input + m_id * n + n_id+2) * 1.0));
        tmp.w = __float2int_rn(static_cast<float>(__ldg(input + m_id * n + n_id+3) * 1.0));
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
        ((T)(input[(m_id & 0xffffffe0) * n + (n_id << 5) + (m_id & 31)]) *
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

template <typename T>
class AttnMatmulINT8 {
public:
    AttnMatmulINT8(
            const platform::CUDADeviceContext& dev_ctx,
             int bsz_seq,
             int output_size,
             int input_size,
             bool compute_bias,
             const std::vector<const framework::Tensor*> weight_cts,
             const platform::Place& place)
        :dev_ctx_(dev_ctx),
        bsz_seq_(bsz_seq),
        output_size_(output_size),
        input_size_(input_size),
        compute_bias_(compute_bias) {
            
        helper_ = std::make_unique<CublasLtHelper>(bsz_seq, input_size, output_size);

        //quantize and transpose weight
        // for (const framework::Tensor* weight_ct : weight_cts) {
        //     framework::Tensor weight_tmp;
        //     framework::Tensor* weight_t = const_cast<framework::Tensor*>(weight_ct);
        //     framework::TensorCopy(*weight_t, place, &weight_tmp);

        //     weight_t->Resize({input_size*output_size});
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
                        framework::Tensor* bias_out){
        //quant transpose A
        float scale = 1.0f;
        VLOG(1) << "[DEBUG] row_major_to_col32_quantize_kernelLauncher";
        row_major_to_col32_quantize_kernelLauncher<T>(input->data<T>(), 
                                                      input_tmp->data<int8_t>(), 
                                                      bsz_seq_, input_size_, 
                                                      dev_ctx_.stream());

        VLOG(1) << "[DEBUG] GEMM";
        VLOG(1) << "input_tmp " << input_tmp->numel();
        VLOG(1) << "weight_tmp " << weight->numel();
        VLOG(1) << "output_tmp " << output_tmp->numel();

        helper_ -> GEMM(input_tmp->data<int8_t>(), weight->data<int8_t>(), output_tmp->data<int32_t>(), dev_ctx_.stream());

        //dequant C
        VLOG(1) << "[DEBUG] col32_to_row_major_dequantize_kernelLauncher";
        col32_to_row_major_dequantize_kernelLauncher<T>(output_tmp->data<int32_t>(), 
                                                        output->data<T>(), 
                                                        bsz_seq_, 
                                                        output_size_, 
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

private:
    const platform::CUDADeviceContext& dev_ctx_;

    int bsz_seq_; // m
    int output_size_; // n
    int input_size_; // k

    int compute_bias_;
    std::unique_ptr<CublasLtHelper> helper_;

};


}  // namespace operators
}  // namespace paddle
