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

#include "paddle/fluid/platform/dynload/cublasLt.h"
#include "paddle/fluid/framework/tensor.h"

namespace dyl = paddle::platform::dynload;

namespace paddle {
namespace operators {

class CublasLtHelper {
public:
    CublasLtHelper(int m, int k, int n):alpha_(1), beta_(0), m_(m), k_(k), n_(n) {
        cublasStatus_t status;
        status = dyl::cublasLtCreate(&handle_);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));

        status = dyl::cublasLtMatmulDescCreate(&matmul_desc_, CUBLAS_COMPUTE_32I, CUDA_R_32I);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatmulDescCreate"));
        cublasOperation_t op_transpose = CUBLAS_OP_T;
        status = dyl::cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &op_transpose, sizeof(op_transpose));
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatmulDescSetAttribute"));

        cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
        cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
        int ldatransform = 32 * m;
        ldatransform_ = ldatransform;
        int ldbtransform = 32 * ((n + 8 - 1) / 8) * 8;
        ldbtransform_ = ldbtransform;
        int ldctransform = 32 * m;
        ldctransform_ = ldctransform;

        status = dyl::cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_8I, k, n, k);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
        
        status = dyl::cublasLtMatrixLayoutCreate(&B_transform_desc_, CUDA_R_8I, n, k, ldbtransform);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
        VLOG(1) << "B_transform_desc_ " << (k + 32 - 1) / 32 * ldbtransform;
        status = dyl::cublasLtMatrixLayoutSetAttribute(B_transform_desc_, 
                                                                    CUBLASLT_MATRIX_LAYOUT_ORDER, 
                                                                    &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C));
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));

        
        status = dyl::cublasLtMatrixLayoutCreate(&A_transfrom_desc_, CUDA_R_8I, m, k, ldatransform);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
        VLOG(1) << "A_transfrom_desc_" << (k + 32 - 1) / 32 * ldatransform;
        status = dyl::cublasLtMatrixLayoutSetAttribute(A_transfrom_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutSetAttribute"));

        
        status = dyl::cublasLtMatrixLayoutCreate(&C_transform_desc_, CUDA_R_32I, m, n, ldctransform);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
        VLOG(1) << "C_transform_desc_" << (n + 32 - 1) / 32 * ldctransform;
        status = dyl::cublasLtMatrixLayoutSetAttribute(C_transform_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutSetAttribute"));

        status = dyl::cublasLtMatrixTransformDescCreate(&transform_desc_, CUDA_R_32F);
        // op_transpose = CUBLAS_OP_T;
        status = dyl::cublasLtMatrixTransformDescSetAttribute(transform_desc_, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op_transpose, sizeof(op_transpose)); 
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixTransformDescSetAttribute"));
    }
    ~CublasLtHelper() {
        
    }

    void TransformB(framework::Tensor* B_t/*int8_t*/, const platform::Place& place, cudaStream_t stream) {
        cublasStatus_t status;
        framework::Tensor B_tmp;
        framework::TensorCopy(*B_t, place, &B_tmp);


        B_t -> Resize({(k_ + 32 - 1) / 32 * ldbtransform_});
        int8_t* B_transform_data_dev = B_t->mutable_data<int8_t>(place);
        float transform_alpha = 1.0f, transform_beta = 0.0f;

        status = dyl::cublasLtMatrixTransform(handle_, transform_desc_, (void*)&transform_alpha, B_tmp.data<int8_t>(), B_desc_, 
                (void*)&transform_beta, nullptr, nullptr, B_transform_data_dev, B_transform_desc_, stream);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixTransform"));
    }

    void GEMM(
        int8_t* A_transform_data_dev,
        int8_t* B_transform_data_dev,
        int32_t* C_transform_data_dev,
        cudaStream_t stream) {
        cublasStatus_t status;

        // PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&A_transform_data_dev, sizeof(int8_t) * (k_ + 32 - 1) / 32 * ldatransform_));
        // PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&B_transform_data_dev, sizeof(int8_t) * (k_ + 32 - 1) / 32 * ldbtransform_));
        // PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&C_transform_data_dev, sizeof(int32_t) * (n_ + 32 - 1) / 32 * ldctransform_));


        status = dyl::cublasLtMatmul(handle_,
                                    matmul_desc_,
                                    &alpha_,
                                    A_transform_data_dev,
                                    A_transfrom_desc_,
                                    B_transform_data_dev,
                                    B_transform_desc_,
                                    &beta_,
                                    C_transform_data_dev,
                                    C_transform_desc_,
                                    C_transform_data_dev,
                                    C_transform_desc_,
                                    nullptr,
                                    nullptr,
                                    0,
                                    stream);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatmul"));
    }

private:
    cublasLtHandle_t handle_;
    cublasLtMatmulDesc_t matmul_desc_;
    cublasLtMatrixLayout_t A_transfrom_desc_;
    cublasLtMatrixLayout_t B_desc_;
    cublasLtMatrixLayout_t B_transform_desc_;
    cublasLtMatrixLayout_t C_transform_desc_;

    cublasLtMatrixTransformDesc_t transform_desc_; // For Transform weights
    int32_t alpha_;
    int32_t beta_;

    int m_;
    int k_;
    int n_;
    int ldatransform_;
    int ldbtransform_;
    int ldctransform_;

};

}  // namespace operators
}  // namespace paddle