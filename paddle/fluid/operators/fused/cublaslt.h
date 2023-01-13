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

#pragma once

#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/dynload/cublasLt.h"

namespace dyl = paddle::platform::dynload;

namespace paddle {
namespace operators {

#define PADDLE_CUBLASLT_STATUS_CHECK(name)                                    \
  PADDLE_ENFORCE_EQ(                                                          \
      status,                                                                 \
      CUBLAS_STATUS_SUCCESS,                                                  \
      platform::errors::External(                                             \
          #name                                                               \
          "execution error"                                                   \
          "refer https://docs.nvidia.com/cuda/cublas/index.html to get more " \
          "information"))

const int split_k_candidates[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};

struct CublasLtAlgoParam {
  int algo_id;
  int swizzle;
  int custom_option;
  int tile;
  int split_k_val;
  int reduction_scheme;
  int stages;
  size_t workspace_size;
};

struct CublasLtAlgoSelectorParam {
  cublasLtMatmulAlgo_t algo;
  int m;
  int n;
  int k;
  int algo_id;
  int swizzle;
  int custom_option;
  int tile;
  int split_k_val;
  int reduction_scheme;
  int stages;
  size_t workspace_size;
  float time;
};

static std::map<std::tuple<int /* m */, int /* k */, int /* n */>,
                CublasLtAlgoParam>
    AlgoParamCache{};

inline bool compare_algo_time(const CublasLtAlgoSelectorParam& param_a,
                              const CublasLtAlgoSelectorParam& params_b) {
  return (param_a.time < params_b.time);
}

template <typename InT, typename OutT>
void TestMatmulRun(cublasLtHandle_t handle,
                   cublasLtMatmulDesc_t matmul_desc,
                   cublasLtMatrixLayout_t A_desc,
                   cublasLtMatrixLayout_t B_desc,
                   cublasLtMatrixLayout_t C_desc,
                   const InT* A,
                   const InT* B,
                   OutT* C,
                   CublasLtAlgoSelectorParam& param,  // NOLINT
                   cudaEvent_t& start_event,          // NOLINT
                   cudaEvent_t& stop_event,           // NOLINT
                   cudaStream_t stream) {
  cublasStatus_t status;
  cublasLtMatmulHeuristicResult_t heuristic_result;
  status = dyl::cublasLtMatmulAlgoCheck(handle,
                                        matmul_desc,
                                        A_desc,
                                        B_desc,
                                        C_desc,
                                        C_desc,
                                        &param.algo,
                                        &heuristic_result);
  PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCheck);

  param.workspace_size = heuristic_result.workspaceSize;

  if (status == CUBLAS_STATUS_SUCCESS) {
    cudaError_t err;
    OutT alpha = 1, beta = 0;
    void* work_space;

    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMallocAsync(&work_space, param.workspace_size, stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(start_event, stream));
    int repeats = 100;
    for (int loop = 0; loop < repeats; loop++) {
      status = dyl::cublasLtMatmul(handle,
                                   matmul_desc,
                                   &alpha,
                                   A,
                                   A_desc,
                                   B,
                                   B_desc,
                                   &beta,
                                   C,
                                   C_desc,
                                   C,
                                   C_desc,
                                   &param.algo,
                                   work_space,
                                   param.workspace_size,
                                   stream);
      if (status != CUBLAS_STATUS_SUCCESS) {
        break;
      }
    }

    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(stop_event, stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

    float time;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventElapsedTime(&time, start_event, stop_event));

    if (status == CUBLAS_STATUS_SUCCESS) {
      param.time = time / repeats;
    } else {
      param.time = std::numeric_limits<float>::max();
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(work_space));
  }
  return;
}

template <typename InT, typename OutT>
void CublasLtAlgoSelect(cublasLtHandle_t handle,
                        int m,
                        int n,
                        int k,
                        const InT* A,
                        const InT* B,
                        OutT* C,
                        cublasLtMatmulDesc_t matmul_desc,
                        cublasLtMatrixLayout_t A_desc,
                        cublasLtMatrixLayout_t B_desc,
                        cublasLtMatrixLayout_t C_desc,
                        cublasComputeType_t computeType,
                        cudaDataType_t scaleType,
                        cudaDataType_t Atype,
                        cudaDataType_t Btype,
                        cudaDataType_t Ctype,
                        cudaStream_t stream) {
  // Get Ids
  // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoGetIds
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  int algo_ids[100];
  int num_algo_ids;
  status = dyl::cublasLtMatmulAlgoGetIds(handle,
                                         computeType,
                                         scaleType,
                                         Atype,
                                         Btype,
                                         Ctype,
                                         Ctype,
                                         100,
                                         algo_ids,
                                         &num_algo_ids);
  PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoGetIds);

  // Traverse all posssible algo combinations
  int step = 0;
  int limit = 20000;
  std::vector<CublasLtAlgoSelectorParam> params;

  for (int idx = 0; idx < num_algo_ids; idx++) {
    cublasLtMatmulAlgo_t algo;

    /* Initialize algo structure with given Algp ID */
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoInit
    status = dyl::cublasLtMatmulAlgoInit(handle,
                                         computeType,
                                         scaleType,
                                         Atype,
                                         Btype,
                                         Ctype,
                                         Ctype,
                                         algo_ids[idx],
                                         &algo);
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoInit);

    // Query the tiles enums supported by that algo which is used to alloc
    // enough space to store it
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoCapGetAttribute
    size_t attr_size = 0;
    status = dyl::cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &attr_size);
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCapGetAttribute);

    int num_tiles = static_cast<int>(attr_size / sizeof(int));
    std::vector<int> tiles(num_tiles == 0 ? 1 : num_tiles);
    if (num_tiles == 0) {
      tiles[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
      num_tiles = 1;
    } else {
      status =
          dyl::cublasLtMatmulAlgoCapGetAttribute(&algo,
                                                 CUBLASLT_ALGO_CAP_TILE_IDS,
                                                 tiles.data(),
                                                 sizeof(int) * num_tiles,
                                                 &attr_size);
      PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCapGetAttribute);
    }

    // Query the stages enums supported by that algo (cuda must >= 11.0)
    status = dyl::cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &attr_size);
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCapGetAttribute);
    int num_stages = static_cast<int>(attr_size / sizeof(int));
    std::vector<int> stages(num_stages == 0 ? 1 : num_stages);
    if (num_stages == 0) {
      stages[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
      num_stages = 1;
    } else {
      status =
          dyl::cublasLtMatmulAlgoCapGetAttribute(&algo,
                                                 CUBLASLT_ALGO_CAP_STAGES_IDS,
                                                 stages.data(),
                                                 sizeof(int) * num_stages,
                                                 &attr_size);
      PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCapGetAttribute);
    }

    // Retrieve Other Algo Capabilities attributes
    int splitk_support, red_mask, swizzling_max, custom_option_max;
    status =
        dyl::cublasLtMatmulAlgoCapGetAttribute(&algo,
                                               CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
                                               &splitk_support,
                                               sizeof(splitk_support),
                                               &attr_size);
    status = dyl::cublasLtMatmulAlgoCapGetAttribute(
        &algo,
        CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,
        &red_mask,
        sizeof(red_mask),
        &attr_size);
    status = dyl::cublasLtMatmulAlgoCapGetAttribute(
        &algo,
        CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,
        &swizzling_max,
        sizeof(swizzling_max),
        &attr_size);
    status = dyl::cublasLtMatmulAlgoCapGetAttribute(
        &algo,
        CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,
        &custom_option_max,
        sizeof(custom_option_max),
        &attr_size);
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoCapGetAttribute);

    /* Loop over the different tiles */
    for (int tile_id = 0; tile_id < num_tiles && step < limit; tile_id++) {
      /* Loop over different stages count */
      for (int stage_id = 0; stage_id < num_stages && step < limit;
           stage_id++) {
        /* Loop over the different custom option if any */
        for (int custom_option = 0;
             custom_option <= custom_option_max && step < limit;
             custom_option++) {
          /* Loop over the CTAs swizzling support */
          for (int k = 0; k <= swizzling_max && step < limit; k++) {
            int splir_k_trial = 0;
            if (splitk_support) {
              splir_k_trial +=
                  sizeof(split_k_candidates) / sizeof(split_k_candidates[0]);
            }

            for (int l = 0; (l < (1 + splir_k_trial)) && (step < limit); l++) {
              status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_TILE_ID,
                  &tiles[tile_id],
                  sizeof(tiles[tile_id]));
              status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_STAGES_ID,
                  &stages[stage_id],
                  sizeof(stages[stage_id]));
              status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                  &custom_option,
                  sizeof(custom_option));
              status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
              int split_k_val = 0;
              int reduction_scheme = CUBLASLT_REDUCTION_SCHEME_NONE;
              status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                  &split_k_val,
                  sizeof(split_k_val));
              status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                  &reduction_scheme,
                  sizeof(int));
              if (l > 0) {  // Split-K case
                split_k_val = split_k_candidates[l - 1];
                status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
                    &algo,
                    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                    &split_k_candidates[l - 1],
                    sizeof(split_k_candidates[l - 1]));
                for (reduction_scheme = 1;
                     reduction_scheme <
                         static_cast<int>(CUBLASLT_REDUCTION_SCHEME_MASK) &&
                     (step < limit);
                     reduction_scheme = reduction_scheme << 1) {
                  if (reduction_scheme & red_mask) {
                    status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
                        &algo,
                        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                        &reduction_scheme,
                        sizeof(reduction_scheme));
                    PADDLE_CUBLASLT_STATUS_CHECK(
                        cublasLtMatmulAlgoConfigSetAttribute);

                    cublasLtMatmulHeuristicResult_t heurResult;
                    status = dyl::cublasLtMatmulAlgoCheck(handle,
                                                          matmul_desc,
                                                          A_desc,
                                                          B_desc,
                                                          C_desc,
                                                          C_desc,
                                                          &algo,
                                                          &heurResult);
                    if (status == CUBLAS_STATUS_SUCCESS) {
                      CublasLtAlgoSelectorParam algo_select_params;
                      algo_select_params.algo = algo;
                      algo_select_params.m = m;
                      algo_select_params.n = n;
                      algo_select_params.k = k;
                      algo_select_params.algo_id = algo_ids[idx];
                      algo_select_params.tile = tiles[tile_id];
                      algo_select_params.swizzle = k;
                      algo_select_params.custom_option = custom_option;
                      algo_select_params.split_k_val = split_k_val;
                      algo_select_params.reduction_scheme = reduction_scheme;
                      algo_select_params.stages = stages[stage_id];
                      params.emplace_back(algo_select_params);
                      step++;
                    }
                  }  // end if
                }
              } else {
                // Prepare algos
                cublasLtMatmulHeuristicResult_t heurResult;
                // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoCheck
                status = dyl::cublasLtMatmulAlgoCheck(handle,
                                                      matmul_desc,
                                                      A_desc,
                                                      B_desc,
                                                      C_desc,
                                                      C_desc,
                                                      &algo,
                                                      &heurResult);
                if (status == CUBLAS_STATUS_SUCCESS) {
                  CublasLtAlgoSelectorParam algo_select_params;
                  algo_select_params.algo = algo;
                  algo_select_params.m = m;
                  algo_select_params.n = n;
                  algo_select_params.k = k;
                  algo_select_params.algo_id = algo_ids[idx];
                  algo_select_params.tile = tiles[tile_id];
                  algo_select_params.swizzle = k;
                  algo_select_params.custom_option = custom_option;
                  algo_select_params.split_k_val = split_k_val;
                  algo_select_params.reduction_scheme = reduction_scheme;
                  algo_select_params.stages = stages[stage_id];
                  params.emplace_back(algo_select_params);
                  step++;
                }
              }
            }
          }
        }
      }
    }
  }
  cudaEvent_t start_event;
  cudaEvent_t stop_event;

  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&start_event));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&stop_event));

  for (int i = 0; i < step; i++) {
    TestMatmulRun(handle,
                  matmul_desc,
                  A_desc,
                  B_desc,
                  C_desc,
                  A,
                  B,
                  C,
                  params[i],
                  start_event,
                  stop_event,
                  stream);
  }
  std::sort(params.begin(), params.end(), compare_algo_time);

  int res_id = 0;
  while (params[res_id].time == 0) res_id++;

  CublasLtAlgoParam res;
  res.algo_id = params[res_id].algo_id;
  res.swizzle = params[res_id].swizzle;
  res.tile = params[res_id].tile;
  res.split_k_val = params[res_id].split_k_val;
  res.reduction_scheme = params[res_id].reduction_scheme;
  res.stages = params[res_id].stages;

  AlgoParamCache[{m, k, n}] = std::move(res);
}

class CublasLtHelper {
 public:
  CublasLtHelper(int m, int k, int n)
      : alpha_(1), beta_(0), m_(m), k_(k), n_(n) {
    cublasStatus_t status;
    // handle and matmul desc
    status = dyl::cublasLtCreate(&handle_);
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtCreate);

#if CUBLAS_VER_MAJOR < 11
    cudaDataType_t cudaComputeType = CUDA_R_32I;
#else
    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32I;
#endif

#if CUBLAS_VER_MAJOR < 11
    status = dyl::cublasLtMatmulDescCreate(&matmul_desc_, cudaComputeType);
#else
    status = dyl::cublasLtMatmulDescCreate(
        &matmul_desc_, cudaComputeType, CUDA_R_32I);
#endif

    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescCreate);

    cublasOperation_t op_transpose = CUBLAS_OP_T;
    status = dyl::cublasLtMatmulDescSetAttribute(matmul_desc_,
                                                 CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &op_transpose,
                                                 sizeof(op_transpose));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulDescSetAttribute);

    // matrix desc
    status = dyl::cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_8I, k, n, k);
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatrixLayoutCreate);

    status = dyl::cublasLtMatrixLayoutCreate(&A_desc_, CUDA_R_8I, k, m, k);
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatrixLayoutCreate);

    status = dyl::cublasLtMatrixLayoutCreate(&C_desc_, CUDA_R_32I, n, m, n);
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatrixLayoutCreate);
  }
  ~CublasLtHelper() {}

  void GEMM(int8_t* A_dev,
            const int8_t* B_dev,
            int32_t* C_dev,
            cudaStream_t stream,
            void* work_space = nullptr) {
    cublasStatus_t status;
    size_t workspace_size = 0;

#if CUDA_VERSION >= 11020
    int algo_id = 21;
    int swizzle = 0;
    int custom_option = 0;
    int tile = 15;
    int split_k_val = 0;
    int reduction_scheme = 0;
    int stages = 23;
    if (m_ >= 128) {
      tile = 20;
      stages = 17;
    }

    std::tuple<int, int, int> key(m_, k_, n_);
    if (AlgoParamCache.count(key) == 0) {
      CublasLtAlgoSelect(handle_,
                         m_,
                         n_,
                         k_,
                         A_dev,
                         B_dev,
                         C_dev,
                         matmul_desc_,
                         A_desc_,
                         B_desc_,
                         C_desc_,
                         CUBLAS_COMPUTE_32I,
                         CUDA_R_32I,
                         CUDA_R_8I,
                         CUDA_R_8I,
                         CUDA_R_32I,
                         stream);
    }
    auto value = AlgoParamCache.at(key);
    algo_id = value.algo_id;
    swizzle = value.swizzle;
    custom_option = value.custom_option;
    tile = value.tile;
    split_k_val = value.split_k_val;
    reduction_scheme = value.reduction_scheme;
    stages = value.stages;
    workspace_size = value.workspace_size;

    status = dyl::cublasLtMatmulAlgoInit(handle_,
                                         CUBLAS_COMPUTE_32I,
                                         CUDA_R_32I,
                                         CUDA_R_8I,
                                         CUDA_R_8I,
                                         CUDA_R_32I,
                                         CUDA_R_32I,
                                         algo_id,
                                         &algo_);
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoInit);

    status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo_,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        &(custom_option),
        sizeof(custom_option));
    status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo_, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo_,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        &(split_k_val),
        sizeof(split_k_val));
    status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo_,
        CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        &(swizzle),
        sizeof(swizzle));
    status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo_,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(reduction_scheme),
        sizeof(int));
    status = dyl::cublasLtMatmulAlgoConfigSetAttribute(
        &algo_, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
    PADDLE_CUBLASLT_STATUS_CHECK(cublasLtMatmulAlgoConfigSetAttribute);
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
#if CUDA_VERSION >= 11020
                                 &algo_,
                                 work_space,
                                 workspace_size,
#else
                                 nullptr,
                                 nullptr,
                                 0,
#endif
                                 stream);
    PADDLE_ENFORCE_EQ(
        status,
        CUBLAS_STATUS_SUCCESS,
        platform::errors::External(
            "cublasLtMatmul execution error"
            "refer https://docs.nvidia.com/cuda/cublas/index.html to get more "
            "information"));
  }

 private:
  cublasLtHandle_t handle_;
  cublasLtMatmulDesc_t matmul_desc_;
  cublasLtMatrixLayout_t A_desc_;
  cublasLtMatrixLayout_t B_desc_;
  cublasLtMatrixLayout_t C_desc_;

  cublasLtMatmulAlgo_t algo_;

  int32_t alpha_;
  int32_t beta_;

  int m_;
  int k_;
  int n_;
};

}  // namespace operators
}  // namespace paddle
