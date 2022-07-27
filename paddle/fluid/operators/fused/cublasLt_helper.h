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

#include <sstream>
#include <string>
#include <unordered_map>
#include "paddle/fluid/platform/dynload/cublasLt.h"
#include "paddle/fluid/framework/tensor.h"

namespace dyl = paddle::platform::dynload;

namespace paddle {
namespace operators {

struct AlgoParams {
    AlgoParams(int algoId, int swizzle, int customOption, int tile, int splitK_val, int reductionScheme, int stages): algoId(algoId), swizzle(swizzle), customOption(customOption), tile(tile), splitK_val(splitK_val),
       reductionScheme(reductionScheme), stages(stages) {}
    int algoId;
    int swizzle;
    int customOption;
    int tile;
    int splitK_val;
    int reductionScheme;
    int stages;
};

const std::unordered_map<std::string, AlgoParams> AlgoMap {
   {"1: 4096: 12288", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"1: 4096: 16384", AlgoParams( 7, 1, 0, 23, 0, 0, 15)},
    {"1: 4096: 4096", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"1: 16384: 4096", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"1: 256: 768", AlgoParams( 7, 0, 0, 20, 0, 0, 15)},
    {"1: 256: 1024", AlgoParams( 7, 0, 0, 20, 0, 0, 15)},
    {"1: 1024: 256", AlgoParams( 7, 0, 0, 20, 0, 0, 15)},
    {"1: 256: 256", AlgoParams( 7, 0, 0, 20, 0, 0, 15)},
    {"128: 4096: 12288", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"128: 4096: 16384", AlgoParams( 7, 1, 0, 23, 0, 0, 15)},
    {"128: 4096: 4096", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"128: 16384: 4096", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"128: 256: 768", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"128: 256: 1024", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"128: 1024: 256", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"128: 256: 256", AlgoParams( 7, 0, 0, 20, 0, 0, 15)},
    {"12288: 4096: 1", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"16384: 4096: 1", AlgoParams( 7, 1, 0, 24, 0, 0, 15)},
    {"4096: 4096: 1", AlgoParams( 7, 0, 0, 20, 0, 0, 15)},
    {"4096: 16384: 1", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"768: 256: 1", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"1024: 256: 1", AlgoParams( 7, 0, 0, 20, 0, 0, 15)},
    {"256: 1024: 1", AlgoParams( 7, 0, 0, 20, 0, 0, 15)},
    {"256: 256: 1", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"12288: 4096: 128", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"16384: 4096: 128", AlgoParams( 7, 1, 0, 24, 0, 0, 15)},
    {"4096: 4096: 128", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"4096: 16384: 128", AlgoParams( 7, 0, 0, 20, 0, 0, 15)},
    {"768: 256: 128", AlgoParams( 7, 1, 0, 20, 0, 0, 15)},
    {"1024: 256: 128", AlgoParams( 7, 0, 0, 20, 0, 0, 15)},
    {"256: 1024: 128", AlgoParams( 7, 0, 0, 20, 0, 0, 15)},
    {"256: 256: 128", AlgoParams( 7, 1, 0, 20, 0, 0, 15)}
};

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
        cublasLtOrder_t order_matrixB;
#if CUDA_VERSION >= 11000
        use_4r4_ = true;
#elif 
        use_4r4_ = false;
#endif
        if (use_4r4_) {
            order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
        } else {
            order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
        }


        int ldatransform = 32 * m;
        ldatransform_ = ldatransform;
        int ldbtransform;
        if (use_4r4_) {
            ldbtransform = 32 * ((n + 32 - 1) / 32) * 32;
        } else {
            ldbtransform = 32 * ((n + 8 - 1) / 8) * 8;
        }
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
                                                                    &order_matrixB, sizeof(order_matrixB));
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));

        
        status = dyl::cublasLtMatrixLayoutCreate(&A_transform_desc_, CUDA_R_8I, m, k, ldatransform);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
        VLOG(1) << "A_transform_desc_" << (k + 32 - 1) / 32 * ldatransform;
        status = dyl::cublasLtMatrixLayoutSetAttribute(A_transform_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
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
        if (handle_) dyl::cublasLtDestroy(handle_);
        if (matmul_desc_) dyl::cublasLtMatmulDescDestroy(matmul_desc_);
        if (transform_desc_) dyl::cublasLtMatrixTransformDescDestroy(transform_desc_);
        if (A_transform_desc_) dyl::cublasLtMatrixLayoutDestroy(A_transform_desc_);
        if (B_desc_) dyl::cublasLtMatrixLayoutDestroy(B_desc_);
        if (B_transform_desc_) dyl::cublasLtMatrixLayoutDestroy(B_transform_desc_);
        if (C_transform_desc_) dyl::cublasLtMatrixLayoutDestroy(C_transform_desc_);

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
        const int8_t* B_transform_data_dev,
        int32_t* C_transform_data_dev,
        cudaStream_t stream) {
        cublasStatus_t status;
        VLOG(1) << "m=" << m_ << "k=" << k_ << "n=" << n_;

        // PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&A_transform_data_dev, sizeof(int8_t) * (k_ + 32 - 1) / 32 * ldatransform_));
        // PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&B_transform_data_dev, sizeof(int8_t) * (k_ + 32 - 1) / 32 * ldbtransform_));
        // PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&C_transform_data_dev, sizeof(int32_t) * (n_ + 32 - 1) / 32 * ldctransform_));

        //test using specific algo

        cublasLtMatmulAlgo_t algo;
        int algoId;
        if (use_4r4_) {
            algoId = 7;
        } else {
            algoId = 6;
        }
        int swizzle = 1;
        int customOption = 0;
        int tile = 20;
        int splitK_val = 0;
        int reductionScheme = 0;
#if CUDA_VERSION >= 11000
        int stages;
        if (use_4r4_) {
            stages = 15;
        } else {
            stages = 13;
        }
#endif
        std::stringstream ss;
        ss << m_ << ": " << k_ << ": " << n_;
        std::string key(ss.str());
        if (AlgoMap.count(key) != 0) {
            AlgoParams params = AlgoMap.at(key);
            algoId = params.algoId;
            swizzle = params.swizzle;
            customOption = params.customOption;
            tile = params.tile;
            splitK_val = params.splitK_val;
            reductionScheme = params.reductionScheme;
            stages = params.stages;
            VLOG(1) << key << " has map tile = " << tile;
        } else {
            VLOG(1) << key << " has no map";
        }


        dyl::cublasLtMatmulAlgoInit(
            handle_, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, algoId, &algo);
        dyl::cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption));
        dyl::cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
        dyl::cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val));
        dyl::cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
        dyl::cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(int));
#if CUDA_VERSION >= 11000
        dyl::cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
#endif


        status = dyl::cublasLtMatmul(handle_,
                                    matmul_desc_,
                                    &alpha_,
                                    A_transform_data_dev,
                                    A_transform_desc_,
                                    B_transform_data_dev,
                                    B_transform_desc_,
                                    &beta_,
                                    C_transform_data_dev,
                                    C_transform_desc_,
                                    C_transform_data_dev,
                                    C_transform_desc_,
                                    &algo,
                                    nullptr,
                                    0,
                                    stream);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatmul"));
        VLOG(1) << "gemm finsh";
        // PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
    }

private:
    cublasLtHandle_t handle_;
    cublasLtMatmulDesc_t matmul_desc_;
    cublasLtMatrixLayout_t A_transform_desc_;
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

    bool use_4r4_;

};

}  // namespace operators
}  // namespace paddle