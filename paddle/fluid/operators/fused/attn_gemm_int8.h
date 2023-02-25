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
#include "paddle/phi/kernels/index_select_kernel.h"

DECLARE_int64(cublaslt_exhaustive_search_times);

namespace paddle {
namespace operators {

template <typename T>
static void PrintMatrix(const T* mat_d, int num, std::string name) {
  if (FLAGS_cublaslt_exhaustive_search_times != 114514) return;

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
void LLMGemm(const phi::GPUContext& dev_ctx, 
             const phi::DenseTensor* weight,
             const phi::DenseTensor* input,
             const phi::DenseTensor* weight_range,
             phi::DenseTensor* output,
             std::string name,
             int m, int k, int n,
             phi::DenseTensor* workspace,
             const int quant_round_type = 1,
             const float quant_max_bound = 127.0,
             const float quant_min_bound = -127.0,
             const float threshold = 6.0f) {
  auto weight_dims = weight->dims();
  auto input_dims = input->dims();
  VLOG(2) << "weight_dims " << weight_dims;
  VLOG(2) << "input_dims " << input_dims;
  phi::DenseTensor weight_tmp;
  weight_tmp.Resize({n, k});
  dev_ctx.Alloc<int8_t>(&weight_tmp);

  phi::DenseTensor input_tmp;
  input_tmp.Resize({m, k});
  dev_ctx.Alloc<T>(&input_tmp);

  cudaMemcpy(weight_tmp.data<int8_t>(), weight->data<int8_t>(), n * k * sizeof(int8_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(input_tmp.data<T>(), input->data<T>(), m * k * sizeof(T), cudaMemcpyDeviceToDevice);

 
  // 1. raw/col range
  phi::DenseTensor abs_input;
  abs_input.Resize(input->dims());
  dev_ctx.Alloc<T>(&abs_input);
  phi::AbsKernel<T>(dev_ctx, *input, &abs_input);

  if (input->dims().size() == 3) {
    abs_input.Resize({input->dims()[0] * input->dims()[1], input->dims()[2]});
  }

  // phi::DenseTensor raw_range;
  // raw_range.Resize({m});
  // dev_ctx.Alloc<T>(&raw_range);

  // std::vector<int64_t> raw_dims{-1};
  // phi::MaxRawKernel<T>(dev_ctx, tmp, raw_dims, false, false, &raw_range);

  phi::DenseTensor col_range;
  col_range.Resize({k});
  dev_ctx.Alloc<T>(&col_range);


  std::vector<int64_t> col_dims{0};
  phi::MaxRawKernel<T>(dev_ctx, abs_input, col_dims, false, false, &col_range);
  VLOG(2) << "Finish max raw kernel";

  // 2. fetch col_ids and slice

  std::vector<T> col_range_vec(k);
  std::vector<int64_t> fp16_index_vec;
  std::vector<int64_t> int8_index_vec;

  cudaMemcpy(col_range_vec.data(), col_range.data<T>(), k * sizeof(T), cudaMemcpyDeviceToHost);
  for (int64_t i = 0; i < k; ++i) {
    if (static_cast<float>(col_range_vec[i]) >= threshold) {
      fp16_index_vec.push_back(i);
    } else {
      int8_index_vec.push_back(i);
    }
  }

  int k_fp16 = fp16_index_vec.size();
  int k_int8 = k - k_fp16;

  int k_int8_supp = k_int8 % 4;
  if (k_int8_supp != 0) {
    for (int i = 0; i < k_int8_supp; ++i) {
      fp16_index_vec.push_back(int8_index_vec[k_int8-1]);
      int8_index_vec.pop_back();
      k_int8--;
      k_fp16++;
    }
  }

  VLOG(1) << "k_fp16 " << k_fp16;
  VLOG(2) << "k_int8 " << k_int8;


  phi::DenseTensor int8_out_dequant;
  phi::DenseTensor fp16_out;

  if (k_fp16 != 0) {
    phi::DenseTensor fp16_index;
    fp16_index.Resize({k_fp16});
    dev_ctx.Alloc<int64_t>(&fp16_index);
    cudaMemcpy(fp16_index.data<int64_t>(), fp16_index_vec.data(), k_fp16 * sizeof(int64_t), cudaMemcpyHostToDevice);

    phi::DenseTensor fp16_in;
    fp16_in.Resize({m, k_fp16});
    phi::DenseTensor fp16_weight;
    phi::DenseTensor fp16_weight_dequant;
    fp16_weight.Resize({n, k_fp16});
    fp16_weight_dequant.Resize({n, k_fp16});
    dev_ctx.Alloc<T>(&fp16_weight_dequant);

    phi::IndexSelectKernel<T>(dev_ctx, input_tmp, fp16_index, 1, &fp16_in);
    phi::IndexSelectKernel<int8_t>(dev_ctx, weight_tmp, fp16_index, 1, &fp16_weight);

    // 3. fp16 matmul
    // 3.1 dequant weight
    auto gpu_config_weight = std::make_unique<GpuLaunchConfig>(
          phi::backends::gpu::GetGpuLaunchConfig1D(
              dev_ctx, k_fp16 * n));
    dequant_weight_kernel<T><<<gpu_config_weight->block_per_grid, gpu_config_weight->thread_per_block, 0,  dev_ctx.stream()>>>(
      fp16_weight_dequant.data<T>(),
      fp16_weight.data<int8_t>(),
      k_fp16, n, 
      weight_range->data<float>()
    );
    VLOG(2) << "dequant weight";
    // 3.2 matmul
      CBLAS_TRANSPOSE transA = CblasNoTrans;
      CBLAS_TRANSPOSE transB = CblasTrans;
      T alpha = static_cast<T>(1.0);
      T beta = static_cast<T>(0.0);

      // (m, n, k) = bsz_seq, output_size, input_size, (input, weight, out)
      fp16_out.Resize(output->dims());
      dev_ctx.Alloc<T>(&fp16_out);
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
    VLOG(2) << "fp16 matmul";
  }

  if (k_int8 != 0) {
    phi::DenseTensor int8_index;
    int8_index.Resize({k_int8});
    dev_ctx.Alloc<int64_t>(&int8_index);
    cudaMemcpy(int8_index.data<int64_t>(), int8_index_vec.data(), k_int8 * sizeof(int64_t), cudaMemcpyHostToDevice);

    phi::DenseTensor int8_in;
    int8_in.Resize({m, k_int8});
    // dev_ctx.Alloc<T>(&int8_in, m * k_int8 * sizeof(T));


    phi::DenseTensor int8_weight;
    // dev_ctx.Alloc<int8_t>(&fp16_weight, n * k_fp16 * sizeof(int8_t));
    int8_weight.Resize({n, k_int8});
    // dev_ctx.Alloc<int8_t>(&int8_weight, n * k_int8 * sizeof(int8_t));
    
    phi::IndexSelectKernel<T>(dev_ctx, input_tmp, int8_index, 1, &int8_in);
    phi::IndexSelectKernel<int8_t>(dev_ctx, weight_tmp, int8_index, 1, &int8_weight);

    VLOG(2) << "select int8";

    // 4. int8 matmul
    phi::DenseTensor int8_in_tmp;
    int8_in_tmp.Resize(int8_in.dims());
    dev_ctx.Alloc<int8_t>(&int8_in_tmp);
    phi::DenseTensor int8_out;
    int8_out.Resize(output->dims());
    dev_ctx.Alloc<int32_t>(&int8_out);
    int8_out_dequant.Resize(output->dims());
    dev_ctx.Alloc<T>(&int8_out_dequant);


    phi::DenseTensor raw_range;
    raw_range.Resize({m});
    dev_ctx.Alloc<T>(&raw_range);

    std::vector<int64_t> raw_dims{-1};
    phi::MaxRawKernel<T>(dev_ctx, int8_in, raw_dims, false, false, &raw_range);

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
    VLOG(2) << "quant";                              
    auto helper = std::make_unique<CublasLtHelper<int32_t>>(m, k_int8, n, dev_ctx.cublaslt_handle());
    VLOG(2) << "int8_in_tmp.dims() " << int8_in_tmp.dims();
    VLOG(2) << "int8_weight.dims() " << int8_weight.dims();
    VLOG(2) << "int8_out.dims() " << int8_out.dims();
    helper->GEMM(int8_in_tmp.data<int8_t>(),
                int8_weight.data<int8_t>(),
                  int8_out.data<int32_t>(),
                  dev_ctx.stream(),
                  (void*)workspace->data<int8_t>(),
                  workspace->numel());
    VLOG(2) << "int8 GEMM";              
    auto gpu_config = std::make_unique<GpuLaunchConfig>(
          phi::backends::gpu::GetGpuLaunchConfig1D(
              dev_ctx, m * n, DequantKernelVecSize));             
    dequant_out_kernel<T, DequantKernelVecSize><<<gpu_config->block_per_grid, gpu_config->thread_per_block, 0,  dev_ctx.stream()>>>(
      int8_out_dequant.data<T>(),
      int8_out.data<int32_t>(),
      m, n, 
      raw_range.data<T>(),
      weight_range->data<float>()
    );
    VLOG(2) << "dequant";            
  }

  if (k_fp16 > 0 && k_int8 > 0) {
    // 5. Add
    std::vector<const phi::DenseTensor*> ins = {&int8_out_dequant, &fp16_out};
    std::vector<phi::DenseTensor*> outs = {output};
    phi::funcs::ElementwiseKernel<T>(
        dev_ctx, ins, &outs, phi::funcs::AddFunctor<T>());
  } else if (k_fp16 > 0 && k_int8 <= 0) {
    PADDLE_THROW(platform::errors::Fatal("Unbelievable"));
  } else if (k_int8 > 0 && k_fp16 <= 0) {
    cudaMemcpy(output->data<T>(), int8_out_dequant.data<T>(), output->numel() * sizeof(T), cudaMemcpyDeviceToDevice);
  }

}

template <typename T>
class AttnMatmulINT8 {
 public:
  AttnMatmulINT8(
      const phi::GPUContext& dev_ctx, int m, int n, int k, bool compute_bias)
      : dev_ctx_(dev_ctx), m_(m), n_(n), k_(k), compute_bias_(compute_bias) {
    helper_ = std::make_unique<CublasLtHelper<int32_t>>(m, k, n, dev_ctx.cublaslt_handle());
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
      VLOG(2) << "enter in max_kernel_launcher";
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

      VLOG(2) << "end max_kernel_launcher";
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
    VLOG(2) << "end quantize_kernel_launcher";

    helper_->GEMM(input_tmp->data<int8_t>(),
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
    VLOG(2) << "end dequantize_kernel_launcher " << compute_bias_;  

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

    helper_->GEMM(input_tmp->data<int8_t>(),
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
    helper_->GEMM(input->data<int8_t>(),
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
    helper_->GEMM(input->data<int8_t>(),
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

    helper_->GEMM(input_tmp->data<int8_t>(),
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
  std::unique_ptr<CublasLtHelper<int32_t>> helper_;
  std::unique_ptr<GpuLaunchConfig> gpu_config_;
};

}  // namespace operators
}  // namespace paddle
