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
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"

DECLARE_int64(cublaslt_exhaustive_search_times);

namespace paddle {
namespace operators {

template <typename T>
static void PrintMatrix(const T* mat_d, int num, std::string name) {
  if (FLAGS_cublaslt_exhaustive_search_times == 0) return;

    std::vector<T> tmp(num);
    cudaMemcpy(tmp.data(), mat_d, sizeof(T) * num, cudaMemcpyDeviceToHost);

    std::ofstream outfile;
    outfile.open(name+".txt", std::ios::out);
    std::stringstream ss;

    for (int i = 0; i < num; ++i) {
      if(std::is_same<T, int8_t>::value) {
        ss << static_cast<int>(tmp[i]) << std::endl;
      } else {
        ss << std::setprecision(8) << tmp[i] << std::endl;
      }
    }
    outfile << ss.str();
    outfile.close();
}

using phi::backends::gpu::GpuLaunchConfig;

template <typename T>
class AttnMatmulINT8 {
 public:
  AttnMatmulINT8(
      const phi::GPUContext& dev_ctx, int m, int n, int k, bool compute_bias)
      : dev_ctx_(dev_ctx), m_(m), n_(n), k_(k), compute_bias_(compute_bias) {
    auto helper = std::make_shared<CublasLtHelper>(m, k, n, dev_ctx.cublaslt_handle());
    helpers_.emplace_back(helper);
    gpu_config_ = std::make_unique<GpuLaunchConfig>(
        phi::backends::gpu::GetGpuLaunchConfig1D(
            dev_ctx, m * n, DequantKernelVecSize));
  }
  ~AttnMatmulINT8() {}

  void ComputeForwardDyquant(const phi::DenseTensor* weight,
                      const phi::DenseTensor* input,
                      phi::DenseTensor* input_tmp,
                      const phi::DenseTensor* bias,
                      phi::DenseTensor* output,
                      phi::DenseTensor* output_tmp,
                      phi::DenseTensor* bias_out,
                      const phi::DenseTensor* dequant_out_scale,
                      std::string name,
                      const int quant_round_type = 1,
                      const float quant_max_bound = 127.0,
                      const float quant_min_bound = -127.0) {
    // PrintMatrix(weight->data<int8_t>(), weight->numel(), name + "_weight");      
    PrintMatrix(input->data<T>(), input->numel(), name + "_in" + "_device_" + std::to_string(dev_ctx_.GetPlace().GetDeviceId())); 
        
    // PrintMatrix(dequant_out_scale->data<float>(), n_, name + "_out_scale");

    phi::DenseTensor quant_in_scale;
    quant_in_scale.Resize({m_});
    dev_ctx_.Alloc<T>(&quant_in_scale, m_ * sizeof(T));
                
    {
      VLOG(1) << "enter in max_kernel_launcher";
      phi::DenseTensor tmp;
      tmp.Resize(input->dims());
      dev_ctx_.Alloc<T>(&tmp, input->numel() * sizeof(T));
      phi::AbsKernel<T>(dev_ctx_, *input, &tmp);

      // max_kernel_launcher(dev_ctx_,
      //                     tmp.data<T>(),
      //                     quant_in_scale.data<T>(),
      //                     m_ * k_);  // max range of input
      std::vector<int64_t> dims{-1};
      phi::MaxRawKernel<T>(dev_ctx_, tmp, dims, false, false, &quant_in_scale);

      VLOG(1) << "end max_kernel_launcher";
    }
    PrintMatrix(quant_in_scale.data<T>(), quant_in_scale.numel(), name + "_in_scale" + "_device_" + std::to_string(dev_ctx_.GetPlace().GetDeviceId()));  
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                0,
                                quant_in_scale.data<T>(),
                                m_,
                                k_,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx_.stream());
    // PrintMatrix(input_tmp->data<int8_t>(), input->numel(), name + "_in_int8");  
    VLOG(1) << "end quantize_kernel_launcher";

    helpers_[0]->GEMM(input_tmp->data<int8_t>(),
                      weight->data<int8_t>(),
                      output_tmp->data<int32_t>(),
                      dev_ctx_.stream());

    dequantize_kernel_launcher<T>(output_tmp->data<int32_t>(),
                                  output->data<T>(),
                                  m_,
                                  n_,
                                  dev_ctx_.stream(),
                                  gpu_config_.get(),
                                  0,
                                  quant_in_scale.data<T>(),
                                  dequant_out_scale->data<float>());
    // PrintMatrix(output->data<T>(), output->numel(), name + "_out");  
    VLOG(1) << "end dequantize_kernel_launcher " << compute_bias_;  

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const phi::DenseTensor*> ins = {output, bias};
      std::vector<phi::DenseTensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      PADDLE_ENFORCE_EQ(cudaGetLastError(),
                        cudaSuccess,
                        platform::errors::Fatal(
                            "cuda error occured after computing bias. "
                            "But it does not mean this error is caused by "
                            "bias computing"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are
  // both T.
  void ComputeForward(const phi::DenseTensor* weight,
                      const phi::DenseTensor* input,
                      phi::DenseTensor* input_tmp,
                      const phi::DenseTensor* bias,
                      phi::DenseTensor* output,
                      phi::DenseTensor* output_tmp,
                      phi::DenseTensor* bias_out,
                      const float quant_in_scale,
                      const phi::DenseTensor* dequant_out_scale,
                      std::string name,
                      const int quant_round_type = 1,
                      const float quant_max_bound = 127.0,
                      const float quant_min_bound = -127.0) {
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale,
                                nullptr,
                                m_,
                                k_,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx_.stream());

    PrintMatrix(input_tmp->data<int8_t>(), input->numel(), name + "_in_int" + "_device_" + std::to_string(dev_ctx_.GetPlace().GetDeviceId()));                            

    helpers_[0]->GEMM(input_tmp->data<int8_t>(),
                      weight->data<int8_t>(),
                      output_tmp->data<int32_t>(),
                      dev_ctx_.stream());
    PrintMatrix(output_tmp->data<int32_t>(), output->numel(), name + "_out_int" + "_device_" + std::to_string(dev_ctx_.GetPlace().GetDeviceId())); 

    dequantize_kernel_launcher<T>(output_tmp->data<int32_t>(),
                                  output->data<T>(),
                                  m_,
                                  n_,
                                  dev_ctx_.stream(),
                                  gpu_config_.get(),
                                  quant_in_scale,
                                  nullptr,
                                  dequant_out_scale->data<float>());

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const phi::DenseTensor*> ins = {output, bias};
      std::vector<phi::DenseTensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      PADDLE_ENFORCE_EQ(cudaGetLastError(),
                        cudaSuccess,
                        platform::errors::Fatal(
                            "cuda error occured after computing bias. "
                            "But it does not mean this error is caused by "
                            "bias computing"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are
  // both INT8.
  void ComputeForwardINT8ToINT8(const phi::DenseTensor* weight,
                                phi::DenseTensor* input,
                                const phi::DenseTensor* bias,
                                phi::DenseTensor* output,
                                phi::DenseTensor* bias_out,
                                void* workspace = nullptr) {
    helpers_[0]->GEMM(input->data<int8_t>(),
                      weight->data<int8_t>(),
                      output->data<int32_t>(),
                      dev_ctx_.stream(),
                      workspace);
  }

  // This function is used to execute GEMM, with input and output's types are
  // INT8 and T.
  void ComputeForwardINT8ToT(const phi::DenseTensor* weight,
                             const float quant_in_scale,
                             phi::DenseTensor* input,
                             const phi::DenseTensor* bias,
                             phi::DenseTensor* output,
                             phi::DenseTensor* output_tmp,
                             phi::DenseTensor* bias_out,
                             const phi::DenseTensor* dequant_out_scale,
                             std::string name="") {
    PrintMatrix(input->data<int8_t>(), m_ * k_, name + "_in_int" + "_device_" + std::to_string(dev_ctx_.GetPlace().GetDeviceId())); 
    helpers_[0]->GEMM(input->data<int8_t>(),
                      weight->data<int8_t>(),
                      output_tmp->data<int32_t>(),
                      dev_ctx_.stream());

    PrintMatrix(output_tmp->data<int32_t>(), m_ * n_, name + "_out_int" + "_device_" + std::to_string(dev_ctx_.GetPlace().GetDeviceId()));                   

    dequantize_kernel_launcher<T>(output_tmp->data<int32_t>(),
                                  output->data<T>(),
                                  m_,
                                  n_,
                                  dev_ctx_.stream(),
                                  gpu_config_.get(),
                                  quant_in_scale,
                                  nullptr,
                                  dequant_out_scale->data<float>());
    PrintMatrix(output->data<T>(),m_ * n_, name + "_out_fp" + "_device_" + std::to_string(dev_ctx_.GetPlace().GetDeviceId())); 

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const phi::DenseTensor*> ins = {output, bias};
      std::vector<phi::DenseTensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      PADDLE_ENFORCE_EQ(cudaGetLastError(),
                        cudaSuccess,
                        platform::errors::Fatal(
                            "cuda error occured after computing bias. "
                            "But it does not mean this error is caused by "
                            "bias computing"));
      
      // PrintMatrix(output->data<T>(),m_ * n_, name + "_out_fp_with_bias" + "_device_" + std::to_string(dev_ctx_.GetPlace().GetDeviceId()));                       
    }
  }

  // This function is used to execute GEMM, with input and output's types are T
  // and INT8.
  void ComputeForwardTToINT8(const phi::DenseTensor* weight,
                             const float quant_in_scale,
                             const phi::DenseTensor* input,
                             phi::DenseTensor* input_tmp,
                             const phi::DenseTensor* bias,
                             phi::DenseTensor* output,
                             phi::DenseTensor* bias_out,
                             const int quant_round_type = 1,
                             const float quant_max_bound = 127.0,
                             const float quant_min_bound = -127.0) {
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale,
                                nullptr,
                                m_,
                                k_,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
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
  std::unique_ptr<GpuLaunchConfig> gpu_config_;
};

}  // namespace operators
}  // namespace paddle
