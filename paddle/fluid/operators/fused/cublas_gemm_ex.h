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

#include "paddle/fluid/platform/dynload/cublas.h"


namespace dyl = paddle::platform::dynload;

namespace paddle {
namespace operators {

class CublasHelper {
public:
    CublasHelper(int m, int k, int n):alpha_(1), beta_(0), m_(m), k_(k), n_(n) {
        
    }
   
    void GEMM(
        int8_t* A,
        const int8_t* B,
        int32_t* C,
        cublasHandle_t handle) {
        // PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
        cublasStatus_t status;
        VLOG(1) << "m=" << m_ << "k=" << k_ << "n=" << n_;

        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

        status = dyl::cublasGemmEx(handle,
            transa,
            transb,
            m_,
            n_,
            k_,
            &alpha_,
            A,
            CUDA_R_8I,
            k_,
            B,
            CUDA_R_8I,
            k_,
            &beta_,
            C,
            CUDA_R_32I,
            m_,
            CUBLAS_COMPUTE_32I,
            algo);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasGemmEx"));

        // PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
        VLOG(1) << "gemm finsh";


    }

private:
    
    int32_t alpha_;
    int32_t beta_;

    int m_;
    int k_;
    int n_;
};

}  // namespace operators
}  // namespace paddle