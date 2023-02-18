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

constexpr int DequantKernelVecSize = 4;

template <typename T>
void LLMGemm(const phi::GPUContext& dev_ctx, 
             const phi::DenseTensor* weight,
             const phi::DenseTensor* input,
             const phi::DenseTensor* weight_range,
             phi::DenseTensor* output,
             phi::DenseTensor* int8_space,
             phi::DenseTensor* int32_space,
             std::string name,
             int m, int k, int n,
             const int quant_round_type = 1,
             const float quant_max_bound = 127.0,
             const float quant_min_bound = -127.0,
             const float threshold = 6.0f {
  // 1. raw/col range
  phi::DenseTensor tmp;
  tmp.Resize(input->dims());
  dev_ctx.Alloc<T>(&tmp, input->numel() * sizeof(T));
  phi::AbsKernel<T>(dev_ctx, *input, &tmp);

  if (input->dims().size() == 3) {
    tmp.Resize({input->dims()[0] * input->dims()[1], input->dims()[2]});
  }

  phi::DenseTensor raw_range;
  raw_range.Resize({m});
  dev_ctx.Alloc<T>(&raw_range, m * sizeof(T));

  std::vector<int64_t> raw_dims{-1};
  phi::MaxRawKernel<T>(dev_ctx, tmp, raw_dims, false, false, &raw_range);

  phi::DenseTensor col_range;
  col_range.Resize({k});
  dev_ctx.Alloc<T>(&col_range, k * sizeof(T));


  std::vector<int64_t> col_dims{0};
  phi::MaxRawKernel<T>(dev_ctx, tmp, col_dims, false, false, &col_range);


  // 2. fetch col_ids and slice

  std::vector<T> col_range_vec(k);
  std::vector<int64_t> fp16_index_vec;
  std::vector<int64_t> int8_index_vec;

  cudaMemcpy(col_range_vec.data(), col_range.data<T>(), k * sizeof(T), cudaMemcpyDeviceToHost);
  for (int64_t i = 0; i < k; ++i) {
    if (col_range_vec >= threshold) {
      fp16_index_vec.push_back(i);
    } else {
      int8_index_vec.push_back(i);
    }
  }

  int k_fp16 = fp16_index_vec.size();
  int k_int8 = k - k_fp16;

  phi::DenseTensor fp16_index;
  fp16_index.Resize({k_fp16});
  dev_ctx.Alloc<int64_t>(&fp16_index, k_fp16 * sizeof(int64_t));
  cudaMemcpy(fp16_index.data<int64_t>(), fp16_index_vec.data(), k_fp16 * sizeof(int64_t));
  phi::DenseTensor int8_index;
  int8_index.Resize({k_int8});
  dev_ctx.Alloc<int64_t>(&int8_index, k_int8 * sizeof(int64_t));
  cudaMemcpy(int8_index.data<int64_t>(), int8_index_vec.data(), k_int8 * sizeof(int64_t));

  phi::DenseTensor fp16_in;
  phi::DenseTensor int8_in;
  fp16_in.Resize({m, k_fp16});
  // dev_ctx.Alloc<T>(&fp16_in, m * k_fp16 * sizeof(T));
  int8_in.Resize({m, k_int8});
  // dev_ctx.Alloc<T>(&int8_in, m * k_int8 * sizeof(T));


  phi::DenseTensor fp16_weight;
  phi::DenseTensor fp16_weight_dequant;
  phi::DenseTensor int8_weight;
  fp16_weight.Resize({n, k_fp16});
  // dev_ctx.Alloc<int8_t>(&fp16_weight, n * k_fp16 * sizeof(int8_t));
  int8_weight.Resize({n, k_int8});
  // dev_ctx.Alloc<int8_t>(&int8_weight, n * k_int8 * sizeof(int8_t));
  fp16_weight_dequant.Resize({n, k_fp16});
  dev_ctx.Alloc<T>(&fp16_weight_dequant, n * k_fp16 * sizeof(T));
  
  phi::IndexSelectKernel<T>(dev_ctx, tmp, fp16_index, 1, &fp16_in);
  phi::IndexSelectKernel<T>(dev_ctx, tmp, int8_index, 1, &int8_in);
  phi::IndexSelectKernel<int8_t>(dev_ctx, weight, fp16_index, 1, &fp16_weight);
  phi::IndexSelectKernel<int8_t>(dev_ctx, weight, int8_index, 1, &int8_weight);

  // 3. fp16 matmul
  // 3.1 dequant weight
  gpu_config_weight = std::make_unique<GpuLaunchConfig>(
        phi::backends::gpu::GetGpuLaunchConfig1D(
            dev_ctx, k_fp16 * n, DequantKernelVecSize));
  dequant_weight_kernel<<<gpu_config_weight->block_per_grid, gpu_config_weight->thread_per_block, 0,  dev_ctx.stream()>>>(
    fp16_weight_dequant.data<T>(),
    fp16_weight.data<T>(),
    k, n, 
    weight_range->data<float>()
  );
  // 3.2 matmul
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasTrans;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);

    // (m, n, k) = bsz_seq, output_size, input_size, (input, weight, out)
    phi::DenseTensor fp16_out;
    fp16_out.Resize(output->dims());
    dev_ctx.Alloc<T>(&fp16_out, fp16_out.numel() * sizeof(T));
    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
    blas.GEMM(transA,
              transB,
              m,
              n,
              k_fp16,
              alpha,
              fp16_in.data<T>(),
              fp16_weight_dequant.data<T>(),
              beta,
              fp16_out.data<T>());

  // 4. int8 matmul
  phi::DenseTensor int8_in_tmp;
  int8_in_tmp.Resize(int8_in.dims());
  dev_ctx.Alloc<int8_t>(&int8_in_tmp, int8_in_tmp.numel() * sizeof(int8_t));
  phi::DenseTensor int8_out;
  int8_out.Resize(output->dims());
  dev_ctx.Alloc<int32_t>(&int8_out, int8_out.numel() * sizeof(int32_t));
  quantize_kernel_launcher<T>(int8_in.data<T>(),
                              int8_in_tmp.data<int8_t>(),
                                0,
                                raw_range.data<T>(),
                                m,
                                k_int8,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx.stream());
  auto helper = std::make_shared<CublasLtHelper>(m, k_int8, n, dev_ctx.cublaslt_handle());
  helper->GEMM(int8_in_tmp.data<int8_t>(),
               int8_weight.data<int8_t>(),
                int8_out.data<int32_t>(),
                dev_ctx_.stream());
  gpu_config = std::make_unique<GpuLaunchConfig>(
        phi::backends::gpu::GetGpuLaunchConfig1D(
            dev_ctx, m * n, DequantKernelVecSize));             
  dequantize_kernel_launcher<T>(int8_out.data<int32_t>(),
                                  output->data<T>(),
                                  m,
                                  n,
                                  dev_ctx.stream(),
                                  gpu_config.get(),
                                  0,
                                  raw_range.data<T>(),
                                  weight_range->data<float>());
  // 5. Add

      
}

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
                      output_tmp->data<int32_t>(),
                      dev_ctx_.stream());

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
                             const phi::DenseTensor* dequant_out_scale) {
    helpers_[0]->GEMM(input->data<int8_t>(),
                      weight->data<int8_t>(),
                      output_tmp->data<int32_t>(),
                      dev_ctx_.stream());

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
