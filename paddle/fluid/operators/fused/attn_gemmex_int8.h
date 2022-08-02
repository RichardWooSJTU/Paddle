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
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/fluid/platform/dynload/cublas.h"

namespace paddle {
namespace operators {

// static inline __device__ int8_t float_to_int8_rn(float x)
// {
//   uint32_t dst;
//   asm volatile("cvt.rni.sat.s8.f32 %0, %1;"
//                : "=r"(dst)
//                : "f"(x));
//   return reinterpret_cast<const int8_t &>(dst);
// }

// template <typename T>
// __global__ void row_major_to_col32_quantize_kernel(const T* input,
//                                                  char4* output,
//                                                  int m,
//                                                  int n) {

// }

// template <typename T>
// void row_major_to_col32_quantize_kernelLauncher(const T* input,
//                                                 int8_t* output,
//                                                 // T* scale,
//                                                 const int m,
//                                                 const int n,
//                                                 cudaStream_t stream) {
// //   std::cout << "row-major-to-col32: m: " << m << " n: " << n << std::endl;

//     dim3 grid((m + 31) / 32, (n + 31) / 32);
//     dim3 block(32, 32);

//     row_major_to_col32_quantize_kernel<<<grid, block, 0, stream>>>(
//     input,
//     (char4*)output,
//     m,
//     n);
// }

// template <typename T>
// __global__ void col32_to_row_major_dequantize_kernel(T* output,
//                                                   int32_t* input,
//                                                   const int m,  // hidden
//                                                   const int n,  // batch size
//                                                   const float max_range,
//                                                   int repeat) 
// {
// }

// template <typename T>
// void col32_to_row_major_dequantize_kernelLauncher(int32_t* input,
//                                                   T* output,
//                                                   const int batch_size, // m
//                                                   const int hidden_units,  // n
//                                                   cudaStream_t stream) {
                                                      
// //   dim3 grid((hidden_units + 31) / 32, (batch_size + 31) / 32);
// //   dim3 block(32, 32);

//   dim3 grid(1, 512);
//   dim3 block(32, 4);

//   int repeat = batch_size * hidden_units / 65536;

//   col32_to_row_major_dequantize_kernel<<<grid, block, 0, stream>>>(
//       output, input, hidden_units, batch_size, 127.0f, repeat);
// }



namespace dyl = paddle::platform::dynload;

template <typename T>
class AttnMatmulINT8Ex {
public:
    AttnMatmulINT8Ex(
            const platform::CUDADeviceContext& dev_ctx,
             int m,
             int n,
             int k,
             bool compute_bias)
        :dev_ctx_(dev_ctx),
        m_(m),
        n_(n),
        k_(k),
        compute_bias_(compute_bias) {}
    ~AttnMatmulINT8Ex(){}
    void ComputeForward(const framework::Tensor* weight, // [int8] which has been transformed in pass
                        const framework::Tensor* input, // [fp16/32] 
                        framework::Tensor* input_tmp, // [int8]  workspace
                        const framework::Tensor* bias, //[fp16/32] 
                        framework::Tensor* output, // [fp16/32] has been dequantized/detranspose/detranbsform
                        framework::Tensor* output_tmp, //[int32]  workspace
                        framework::Tensor* bias_out,
                        cudaStream_t* streams,
                        cudaEvent_t* stream_events,
                        cublasHandle_t main_handle,
                        cublasHandle_t* sub_handles){
        int m = m_, k = k_, n = n_;
        VLOG(1) << "m=" << m_ << "k=" << k_ << "n=" << n_;
        // CBLAS_TRANSPOSE transA = CblasNoTrans;
        // CBLAS_TRANSPOSE transB = CblasTrans;
        int8_t alpha = static_cast<int8_t>(1);
        int8_t beta = static_cast<int8_t>(0);
        // row_major_to_col32_quantize_kernelLauncher<T>(input->data<T>(), 
        //                                               input_tmp->data<int8_t>(), 
        //                                                 m_, k_,
        //                                               dev_ctx_.stream());

        // auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, int8_t>(dev_ctx_);
        // blas.GEMMINT8(transA,
        //         transB,
        //         m_,
        //         n_,
        //         k_,
        //         alpha,
        //         input_tmp->data<int8_t>(),
        //         weight->data<int8_t>(),
        //         beta,
        //         output_tmp->data<int32_t>());
        cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        cublasStatus_t status;
        if (k_ == 4 * n_ && k_ == 16384) {
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
            for (int i = 0; i < 4; ++i) {
                status = dyl::cublasGemmEx(sub_handles[i],
                    transa,
                    transb,
                    m,
                    // n,
                    n,
                    // m,
                    k/4,
                    &alpha,
                    input_tmp->data<int8_t>(),
                    CUDA_R_8I,
                    k/4,
                    weight->data<int8_t>(),
                    CUDA_R_8I,
                    k/4,
                    &beta,
                    output_tmp->data<int32_t>(),
                    CUDA_R_32I,
                    m,
                    CUBLAS_COMPUTE_32I,
                    algo);
                PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasGemmEx"));
            }
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
        } else {
            status = dyl::cublasGemmEx(main_handle,
                transa,
                transb,
                m,
                // n,
                n,
                // m,
                k,
                &alpha,
                input_tmp->data<int8_t>(),
                CUDA_R_8I,
                k,
                weight->data<int8_t>(),
                CUDA_R_8I,
                k,
                &beta,
                output_tmp->data<int32_t>(),
                CUDA_R_32I,
                m,
                CUBLAS_COMPUTE_32I,
                algo);
            PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasGemmEx"));
        }
        // col32_to_row_major_dequantize_kernelLauncher<T>(output_tmp->data<int32_t>(), 
        //                                                 output->data<T>(), 
        //                                                 m_, n_, 
        //                                                 dev_ctx_.stream());
        // PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

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
};


}  // namespace operators
}  // namespace paddle