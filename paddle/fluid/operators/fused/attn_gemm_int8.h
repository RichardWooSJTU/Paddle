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

#include <iostream>
#include <vector>
#include "paddle/fluid/operators/fused/cublaslt.h"
#include "paddle/fluid/operators/fused/quant_dequant_kernel.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace paddle {
namespace operators {

#define DEBUG_PRINT
const std::unordered_set<int> debug_layers{0,1,2,3,4,5};
const std::unordered_set<int> debug_steps{0, 1};
static int step = 0;
static int layer = 0;

using Tensor = framework::Tensor;

#ifdef DEBUG_PRINT
template <typename T>
static void PrintMatrix(const T* mat_d, int num, std::string name) {
  if (debug_layers.count(layer) == 0) return;
  if (debug_steps.count(step) == 0) return;

    std::vector<T> tmp(num);
    cudaMemcpy(tmp.data(), mat_d, sizeof(T) * num, cudaMemcpyDeviceToHost);
    int sum_i8 = 0;
    T sum = static_cast<T>(0);

    std::ofstream outfile;
    outfile.open(name+"_" + std::to_string(step) + ".txt", std::ios::out);
    std::stringstream ss;

    for (int i = 0; i < num; ++i) {
      if(std::is_same<T, int8_t>::value) {
        ss << static_cast<int>(tmp[i]) << std::endl;
        // sum_i8 += static_cast<int>(tmp[i*n+j]);
      } else {
        ss << std::setprecision(8) << tmp[i] << std::endl;
        // sum += tmp[i*n+j];
      }
    }
    outfile << ss.str();
    // if(std::is_same<T, int8_t>::value) {
    //   std::cout << "sum = " << sum_i8 << std::endl;
    // } else {
    //   std::cout << "sum = " << sum << std::endl;
    // }
    outfile.close();
}
#else
template <typename T>
static void PrintMatrix(const T* mat_d, int num, std::string name) {
}
#endif

template <typename T>
class AttnMatmulINT8 {
 public:
  AttnMatmulINT8(
      const phi::GPUContext& dev_ctx, int m, int n, int k, bool compute_bias)
      : dev_ctx_(dev_ctx), m_(m), n_(n), k_(k), compute_bias_(compute_bias) {
    auto helper = std::make_shared<CublasLtHelper>(m, k, n);
    helpers_.emplace_back(helper);
  }
  ~AttnMatmulINT8() {}

  // This function is used to execute GEMM, with input and output's types are
  // both T.
  void ComputeForward(const framework::Tensor* weight,
                      const framework::Tensor* input,
                      framework::Tensor* input_tmp,
                      const framework::Tensor* bias,
                      framework::Tensor* output,
                      framework::Tensor* output_tmp,
                      framework::Tensor* bias_out,
                      const float quant_in_scale,
                      const framework::Tensor* dequant_out_scale,
                      const int quant_out_scale_offset,
                      const int quant_round_type = 1,
                      const float quant_max_bound = 127.0,
                      const float quant_min_bound = -127.0,
                      const std::string name = "") {
    PrintMatrix(input->data<T>(),  input->numel(), "inference_" + name + "_input_" + std::to_string(layer) + "_float");
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale,
                                m_,
                                k_,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx_.stream());
    PrintMatrix(input_tmp->data<int8_t>(),  input->numel(), "inference_" + name + "_input_" + std::to_string(layer) + "_int8");

    helpers_[0]->GEMM(input_tmp->data<int8_t>(),
                      weight->data<int8_t>(),
                      output_tmp->data<int32_t>(),
                      dev_ctx_.stream());
    PrintMatrix(output_tmp->data<int32_t>(),  output->numel(), "inference_" + name + "_output_" + std::to_string(layer) + "_int32");

    dequantize_kernel_launcher<T>(output_tmp->data<int32_t>(),
                                  output->data<T>(),
                                  m_,
                                  n_,
                                  dev_ctx_.stream(),
                                  quant_in_scale,
                                  dequant_out_scale->data<float>(),
                                  quant_out_scale_offset);
    PrintMatrix(output->data<T>(),  output->numel(), "inference_" + name + "_output_" + std::to_string(layer) + "_float");

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const framework::Tensor*> ins = {output, bias};
      std::vector<framework::Tensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      // PADDLE_ENFORCE_EQ(cudaGetLastError(),
      //                   cudaSuccess,
      //                   platform::errors::Fatal(
      //                       "cuda error occured after computing bias. "
      //                       "But it does not mean this error is caused by "
      //                       "bias computing"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are
  // both INT8.
  void ComputeForwardINT8ToINT8(const framework::Tensor* weight,
                                framework::Tensor* input,
                                const framework::Tensor* bias,
                                framework::Tensor* output,
                                framework::Tensor* bias_out) {
    helpers_[0]->GEMM(input->data<int8_t>(),
                      weight->data<int8_t>(),
                      output->data<int32_t>(),
                      dev_ctx_.stream());
  }

  // This function is used to execute GEMM, with input and output's types are
  // INT8 and T.
  void ComputeForwardINT8ToT(const framework::Tensor* weight,
                             const float quant_in_scale,
                             framework::Tensor* input,
                             const framework::Tensor* bias,
                             framework::Tensor* output,
                             framework::Tensor* output_tmp,
                             framework::Tensor* bias_out,
                             const framework::Tensor* dequant_out_scale,
                             const int quant_out_scale_offset) {
    helpers_[0]->GEMM(input->data<int8_t>(),
                      weight->data<int8_t>(),
                      output_tmp->data<int32_t>(),
                      dev_ctx_.stream());
    
    // PrintMatrix(output_tmp->data<int32_t>(),  output->numel(), "infer_qkv_out_" + std::to_string(layer) + "_int8", layer, is_encoder);

    dequantize_kernel_launcher<T>(output_tmp->data<int32_t>(),
                                  output->data<T>(),
                                  m_,
                                  n_,
                                  dev_ctx_.stream(),
                                  quant_in_scale,
                                  dequant_out_scale->data<float>(),
                                  quant_out_scale_offset);
    // PrintMatrix(output->data<T>(), output->numel(), "infer_qkv_out_" + std::to_string(layer) + "_float", layer, is_encoder);

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const framework::Tensor*> ins = {output, bias};
      std::vector<framework::Tensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      // PADDLE_ENFORCE_EQ(cudaGetLastError(),
      //                   cudaSuccess,
      //                   platform::errors::Fatal(
      //                       "cuda error occured after computing bias. "
      //                       "But it does not mean this error is caused by "
      //                       "bias computing"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are T
  // and INT8.
  void ComputeForwardTToINT8(const framework::Tensor* weight,
                             const float quant_in_scale,
                             const framework::Tensor* input,
                             framework::Tensor* input_tmp,
                             const framework::Tensor* bias,
                             framework::Tensor* output,
                             framework::Tensor* bias_out,
                             const int quant_round_type = 1,
                             const float quant_max_bound = 127.0,
                             const float quant_min_bound = -127.0) {
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale,
                                m_,
                                k_,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx_.stream());

    // PrintMatrix(input_tmp->data<int8_t>(),  input->numel(), "infer_out_linear_in_" + std::to_string(layer) + "_int", layer, is_encoder);
    helpers_[0]->GEMM(input_tmp->data<int8_t>(),
                      weight->data<int8_t>(),
                      output->data<int32_t>(),
                      dev_ctx_.stream());
    // PrintMatrix(output->data<int32_t>(),  input->numel(), "infer_out_linear_out_" + std::to_string(layer) + "_int", layer, is_encoder);
  }

 private:
  const phi::GPUContext& dev_ctx_;

  int m_;  // m
  int n_;  // n
  int k_;  // k

  int compute_bias_;
  std::vector<std::shared_ptr<CublasLtHelper>> helpers_;
};

}  // namespace operators
}  // namespace paddle
