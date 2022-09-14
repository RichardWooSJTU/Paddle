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

using Tensor = framework::Tensor;

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
                      const float quant_in_scale_data,
                      const framework::Tensor* quant_out_scale,
                      const int quant_out_scale_offset) {
    int m = m_, k = k_, n = n_;

    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale_data,
                                m_,
                                k_,
                                dev_ctx_.stream());

    helpers_[0]->GEMM(input_tmp->data<int8_t>(),
                      weight->data<int8_t>(),
                      output_tmp->data<int32_t>(),
                      dev_ctx_.stream());

    dequantize_kernel_launcher<T>(output_tmp->data<int32_t>(),
                                  output->data<T>(),
                                  m_,
                                  n_,
                                  dev_ctx_.stream(),
                                  quant_in_scale_data,
                                  quant_out_scale->data<float>(),
                                  quant_out_scale_offset);

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const framework::Tensor*> ins = {output, bias};
      std::vector<framework::Tensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      PADDLE_ENFORCE_EQ(
          cudaGetLastError(), cudaSuccess, platform::errors::Fatal("Add"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are
  // both INT8.
  void ComputeForwardINT8ToINT8(const framework::Tensor* weight,
                                framework::Tensor* input,
                                const framework::Tensor* bias,
                                framework::Tensor* output,
                                framework::Tensor* bias_out) {
    int m = m_, k = k_, n = n_;

    helpers_[0]->GEMM(input->data<int8_t>(),
                      weight->data<int8_t>(),
                      output->data<int32_t>(),
                      dev_ctx_.stream());
  }

  // This function is used to execute GEMM, with input and output's types are
  // INT8 and T.
  void ComputeForwardINT8ToT(const framework::Tensor* weight,
                             const float quant_in_scale_data,
                             framework::Tensor* input,
                             const framework::Tensor* bias,
                             framework::Tensor* output,
                             framework::Tensor* output_tmp,
                             framework::Tensor* bias_out,
                             const framework::Tensor* quant_out_scale,
                             const int quant_out_scale_offset) {
    int m = m_, k = k_, n = n_;

    helpers_[0]->GEMM(input->data<int8_t>(),
                      weight->data<int8_t>(),
                      output_tmp->data<int32_t>(),
                      dev_ctx_.stream());

    dequantize_kernel_launcher<T>(output_tmp->data<int32_t>(),
                                  output->data<T>(),
                                  m_,
                                  n_,
                                  dev_ctx_.stream(),
                                  quant_in_scale_data,
                                  quant_out_scale->data<float>(),
                                  quant_out_scale_offset);

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const framework::Tensor*> ins = {output, bias};
      std::vector<framework::Tensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      PADDLE_ENFORCE_EQ(
          cudaGetLastError(), cudaSuccess, platform::errors::Fatal("Add"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are T
  // and INT8.
  void ComputeForwardTToINT8(const framework::Tensor* weight,
                             const float quant_in_scale_data,
                             const framework::Tensor* input,
                             framework::Tensor* input_tmp,
                             const framework::Tensor* bias,
                             framework::Tensor* output,
                             framework::Tensor* bias_out) {
    int m = m_, k = k_, n = n_;
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale_data,
                                m_,
                                k_,
                                dev_ctx_.stream());

    helpers_[0]->GEMM(input_tmp->data<int8_t>(),
                      weight->data<int8_t>(),
                      output->data<int32_t>(),
                      dev_ctx_.stream());
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
