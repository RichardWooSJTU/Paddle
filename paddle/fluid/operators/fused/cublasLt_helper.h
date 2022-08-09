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

enum  CublasDataLayout {
    COL32 = 0,
    COL32_2R_4R4 = 1,
    COL4_4R2_8C = 2,
};

class CublasLtHelper {
public:
    CublasLtHelper(int m, int k, int n):alpha_(1), beta_(0), m_(m), k_(k), n_(n) {
        cublasStatus_t status;
        // handle and matmul desc
        status = dyl::cublasLtCreate(&handle_);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));

        status = dyl::cublasLtMatmulDescCreate(&matmul_desc_, CUBLAS_COMPUTE_32I, CUDA_R_32I);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatmulDescCreate"));
        cublasOperation_t op_transpose = CUBLAS_OP_T;
        status = dyl::cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &op_transpose, sizeof(op_transpose));
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatmulDescSetAttribute"));

        // matrix desc
        status = dyl::cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_8I, k, n, k);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));

        status = dyl::cublasLtMatrixLayoutCreate(&A_desc_, CUDA_R_8I, k, m, k);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));

        status = dyl::cublasLtMatrixLayoutCreate(&C_desc_, CUDA_R_32I, n, m, n);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
    }
    ~CublasLtHelper() {
        if (handle_) dyl::cublasLtDestroy(handle_);
        if (matmul_desc_) dyl::cublasLtMatmulDescDestroy(matmul_desc_);
        if (A_desc_) dyl::cublasLtMatrixLayoutDestroy(A_desc_);
        if (B_desc_) dyl::cublasLtMatrixLayoutDestroy(B_desc_);
        if (C_desc_) dyl::cublasLtMatrixLayoutDestroy(C_desc_);

    }

    void GEMM(
        int8_t* A_dev,
        const int8_t* B_dev,
        int32_t* C_dev,
        cudaStream_t stream) {


        // PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

        cublasStatus_t status;
        VLOG(1) << "m=" << m_ << "k=" << k_ << "n=" << n_;

        cublasLtMatmulAlgo_t algo;
        int algoId = 21;
        int swizzle = 0;
        int customOption = 0;
        int tile = 15;
        int splitK_val = 0;
        int reductionScheme = 0;
#if CUDA_VERSION >= 11000
        int stages = 23;
#endif
        // std::stringstream ss;
        // ss << n_ << ": " << k_ << ": " << m_;
        // std::string key(ss.str());
        // if (AlgoMap.count(key) != 0) {
        //     AlgoParams params = AlgoMap.at(key);
        //     algoId = params.algoId;
        //     swizzle = params.swizzle;
        //     customOption = params.customOption;
        //     tile = params.tile;
        //     splitK_val = params.splitK_val;
        //     reductionScheme = params.reductionScheme;
        //     stages = params.stages;
        //     VLOG(1) << key << " has map tile = " << tile;
        // } else {
        //     VLOG(1) << key << " has no map";
        // }


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
                                    B_dev,
                                    B_desc_,
                                    A_dev,
                                    A_desc_,
                                    &beta_,
                                    C_dev,
                                    C_desc_,
                                    C_dev,
                                    C_desc_,
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
    cublasLtMatrixLayout_t A_desc_;
    cublasLtMatrixLayout_t B_desc_;
    cublasLtMatrixLayout_t C_desc_;
    int32_t alpha_;
    int32_t beta_;

    int m_;
    int k_;
    int n_;

};
// class CublasLtTransformHelper {
// public:
//     // col_major: true: m/n is set correctly for col-major matrix
//     CublasLtTransformHelper(int m, int n, CublasDataLayout layout, bool col_major){
//         cublasStatus_t status;
//         status = dyl::cublasLtCreate(&handle_);
//         PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
        
//         cublasOperation_t op_transpose = CUBLAS_OP_N;

//         status = dyl::cublasLtMatrixTransformDescCreate(&transform_desc_, CUDA_R_32F);
//         status = dyl::cublasLtMatrixTransformDescSetAttribute(transform_desc_, CUBLASLT_MATRIX_DESC_TRANSA, &op_transpose, sizeof(op_transpose)); 
//         PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixTransformDescSetAttribute"));

//         if (col_major) {
//             status = dyl::cublasLtMatrixLayoutCreate(&mat_desc_, CUDA_R_8I, m, n, m);    
//         } else {
//             status = dyl::cublasLtMatrixLayoutCreate(&mat_desc_, CUDA_R_8I, n, m, n);
//         }
        
//         PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
        
//         int ld;
//         cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
//         cublasLtOrder_t order_COL32_2R_4R4 = CUBLASLT_ORDER_COL32_2R_4R4;

//         switch (layout) {
//         case COL32:
//             ld = col_major ? 32 * m : 32 * n;
//             status = dyl::cublasLtMatrixLayoutCreate(&mat_desc_, CUDA_R_8I, n, m, ld);
//             status = dyl::cublasLtMatrixLayoutSetAttribute(mat_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
//             break;
//         case COL32_2R_4R4:
//             ld = col_major ?  32 * ((m + 32 - 1) / 32) * 32 :  32 * ((n + 32 - 1) / 32) * 32;
//             status = dyl::cublasLtMatrixLayoutCreate(&mat_desc_, CUDA_R_8I, n, m, ld);
//             status = dyl::cublasLtMatrixLayoutSetAttribute(mat_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32_2R_4R4, sizeof(order_COL32_2R_4R4));
//             break;
//         case COL4_4R2_8C:
//         default:
//             // not support
//              PADDLE_THROW(platform::errors::Unimplemented(
//                     "This layout in cublasLt is not supported."));
//         }
//         PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutSetAttribute"));
//     }

//     ~CublasLtTransformHelper() {
//         if (handle_) dyl::cublasLtDestroy(handle_);
//         if (transform_desc_) dyl::cublasLtMatrixTransformDescDestroy(transform_desc_);
//         if (mat_desc_) dyl::cublasLtMatrixLayoutDestroy(mat_desc_);
//         if (mat_desc_) dyl::cublasLtMatrixLayoutDestroy(mat_desc_);
//     }

//     void Transform(const int8_t *src, int8_t *dst) {
//         cublasStatus_t status;
//         float alpha = 1.0f, beta = 0.0f;

//         cudaStream_t stream = 0;
//         PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
//         status = dyl::cublasLtMatrixTransform(handle_, 
//                                         transform_desc_, 
//                                         (void*)&alpha, 
//                                         src, 
//                                         mat_desc_, 
//                                         (void*)&beta, 
//                                         nullptr, 
//                                         nullptr, 
//                                         dst, 
//                                         mat_desc_, stream);
//         PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixTransform"));
//         PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
//     }


// private:
//     cublasLtHandle_t handle_;
//     cublasLtMatrixTransformDesc_t transform_desc_;

//     cublasLtMatrixLayout_t mat_desc_;

// };

}  // namespace operators
}  // namespace paddle