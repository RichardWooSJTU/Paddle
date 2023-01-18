// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/ernie_for_inference_kernel.h"

#include <algorithm>
#include <vector>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/embedding_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"
#include "paddle/phi/kernels/impl/fused_multi_transformer_kernel_impl.h"
#include "paddle/phi/kernels/impl/topk_sampling_kernel_impl.h"
#include "paddle/phi/kernels/layer_norm_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/kernels/softmax_kernel.h"
#include "paddle/phi/kernels/stack_kernel.h"

namespace phi {

template <typename T>
__global__ void LogAdd(T* data, const T* add_data, const int num) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < num; tid += stride) {
    auto tmp = log(static_cast<float>(data[tid]));
    data[tid] = static_cast<T>(tmp) + add_data[tid];
  }
#endif
}

template <typename T>
inline __device__ T gelu(const T x) {
  // actual gelu with approximation = false
  float tmp = static_cast<float>(x);
  return static_cast<T>(tmp * normcdf(tmp));
}

template <typename T>
__global__ void BiasGelu(T* data, const T* bias, const int num, const int k) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  for (; tid < num; tid += stride) {
    int bias_idx = tid % k;
    const T bias_value = bias[bias_idx];
    const T in_value = data[tid];
    const T tmp = in_value + bias_value;
    data[tid] = gelu(tmp);
  }
#endif
}

template <typename T>
struct AddThreeFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b, const T c) const {
    return a + b + c;
  }
};

inline int GetDesiredBlockDim(int block_dim) {
  if (block_dim <= 64) {
    block_dim = 64;
  } else if (block_dim <= 128) {
    block_dim = 128;
  } else if (block_dim <= 256) {
    block_dim = 256;
  } else if (block_dim <= 512) {
    block_dim = 512;
  } else if (block_dim <= 1024) {
    block_dim = 1024;
  } else {
    if (block_dim % 1024 > 512) {
      block_dim = 1024;
    } else if (block_dim % 512 >= 256) {
      block_dim = 512;
    } else if (block_dim % 256 >= 128) {
      block_dim = 256;
    } else {
      block_dim = 128;
    }
  }
  return block_dim;
}

template <typename T, typename Context>
void ErnieEmbedding(const Context& ctx,
                    const DenseTensor& src_ids,
                    const DenseTensor& pos_ids,
                    const DenseTensor& pos_ids_extra,
                    const std::vector<const DenseTensor*>& embedding_weight,
                    DenseTensor* emb_out) {
  auto input_dims = src_ids.dims();
  int bsz = input_dims[0];
  int input_seq_len = input_dims[1];
  int dim_embed = embedding_weight[0]->dims()[1];

  DenseTensor word_emb_out, pos_embed_out, pos_extra_embed_out;
  word_emb_out.Resize({{bsz, input_seq_len, dim_embed}});
  pos_embed_out.Resize({{bsz, input_seq_len, dim_embed}});
  pos_extra_embed_out.Resize({{bsz, input_seq_len, dim_embed}});

  ctx.template Alloc<T>(emb_out);

  EmbeddingKernel<T, Context>(
      ctx, src_ids, *embedding_weight[0], -1, &word_emb_out);
  EmbeddingKernel<T, Context>(
      ctx, pos_ids, *embedding_weight[1], -1, &pos_embed_out);
  EmbeddingKernel<T, Context>(
      ctx, pos_ids_extra, *embedding_weight[2], -1, &pos_extra_embed_out);

  std::vector<const DenseTensor*> embed_add_ins = {
      &word_emb_out, &pos_embed_out, &pos_extra_embed_out};
  std::vector<DenseTensor*> embed_add_outs = {emb_out};
  phi::funcs::BroadcastKernel<phi::ElementwiseType::kTernary, T, T>(
      ctx, embed_add_ins, &embed_add_outs, -1, AddThreeFunctor<T>());
}

template <typename T, typename Context>
void ErnieForInferenceKernel(
    const Context& ctx,
    const DenseTensor& src_ids,
    const DenseTensor& pos_ids,
    const DenseTensor& input_mask,
    const DenseTensor& pos_ids_extra,
    const DenseTensor& tgt_ids,
    const DenseTensor& tgt_pos,
    const DenseTensor& tgt_pos_extra,
    const DenseTensor& init_score,
    const DenseTensor& tgt_mask,
    const std::vector<const DenseTensor*>&
        embedding_weight,  // word_embed/pos_embed/pos_exra_embed
    const std::vector<const DenseTensor*>& qkv_weight,
    const std::vector<const DenseTensor*>& qkv_bias,
    const std::vector<const DenseTensor*>& out_linear_weight,
    const std::vector<const DenseTensor*>& out_linear_bias,
    const std::vector<const DenseTensor*>& ffn1_weight,
    const std::vector<const DenseTensor*>& ffn1_bias,
    const std::vector<const DenseTensor*>& ffn2_weight,
    const std::vector<const DenseTensor*>& ffn2_bias,
    const std::vector<const DenseTensor*>& attn_ln_scale,
    const std::vector<const DenseTensor*>& attn_ln_bias,
    const std::vector<const DenseTensor*>& ffn_ln_scale,
    const std::vector<const DenseTensor*>& ffn_ln_bias,
    const DenseTensor& post_encoder_ln_scale,
    const DenseTensor& post_encoder_ln_bias,
    const DenseTensor& mask_lm_trans_ln_scale,
    const DenseTensor& mask_lm_trans_ln_bias,
    const DenseTensor& mask_lm_trans_fc_weight,
    const DenseTensor& mask_lm_trans_fc_bias,
    const DenseTensor& mask_lm_out_fc_weight,
    const DenseTensor& mask_lm_out_fc_bias,
    const std::string& decoding_strategy,
    int64_t end_idx,
    int max_dec_len,
    int min_dec_len,
    int topk,
    float topp,
    DenseTensor* scores,
    DenseTensor* indices) {
  auto input_dims = src_ids.dims();
  int bsz = input_dims[0];
  int input_seq_len = input_dims[1];
  int vocab_size = embedding_weight[0]->dims()[0];
  int dim_embed = embedding_weight[0]->dims()[1];
  int num_layers = qkv_weight.size();
  int num_head = qkv_weight[0]->dims()[1];
  int dim_head = qkv_weight[0]->dims()[2];

  // Embedding
  VLOG(1) << "Embedding";

  DenseTensor emb_out;
  emb_out.Resize({{bsz, input_seq_len, dim_embed}});

  ErnieEmbedding<T, Context>(
      ctx, src_ids, pos_ids, pos_ids_extra, embedding_weight, &emb_out);

  // Process Mask
  VLOG(1) << "Process Mask";
  VLOG(1) << "input_mask.dims() " << input_mask.dims();
  DenseTensor scale_mask, attn_mask;
  scale_mask.Resize(input_mask.dims());
  attn_mask.Resize({{bsz, num_head, input_seq_len, input_seq_len}});

  VLOG(1) << "Scale kernel";
  ScaleKernel<T, Context>(ctx, input_mask, 1e4, -1, false, &scale_mask);

  VLOG(1) << "StackKernel";
  std::vector<const DenseTensor*> stack_ins(num_head, &scale_mask);
  StackKernel<T, Context>(ctx, stack_ins, 1, &attn_mask);
  // PreProcess (Layernorm)

  // Fuse MT
  std::vector<DenseTensor> cache_kv_tensors(num_layers);
  std::vector<DenseTensor*> cache_kvs(num_layers);
  DenseTensor enc_out;
  enc_out.Resize(emb_out.dims());

  VLOG(1) << "Prepare cache kv";

  for (int i = 0; i < num_layers; ++i) {
    cache_kvs[i] = &cache_kv_tensors[i];
    cache_kvs[i]->Resize(
        {{2, bsz, num_head, input_seq_len + max_dec_len, dim_head}});
    ctx.template Alloc<T>(cache_kvs[i]);
  }

  VLOG(1) << "Entre fuse mt";

  VLOG(1) << *ffn1_weight[0];
  VLOG(1) << *ffn2_weight[0];
  FusedMultiTransformerKernel<T, Context>(ctx,
                                          emb_out,
                                          attn_mask,
                                          qkv_weight,
                                          qkv_bias,
                                          out_linear_weight,
                                          out_linear_bias,
                                          ffn1_weight,
                                          ffn1_bias,
                                          ffn2_weight,
                                          ffn2_bias,
                                          attn_ln_scale,
                                          attn_ln_bias,
                                          ffn_ln_scale,
                                          ffn_ln_bias,
                                          -1,
                                          true,
                                          1e-9,
                                          "gelu",
                                          &enc_out,
                                          cache_kvs);

  // Loop vars
  VLOG(1) << "Prepare Loop vars";

  DenseTensor concated_mask(tgt_mask);
  DenseTensor append_mask;
  FullKernel<T, Context>(ctx, {bsz, 1, 1}, 1, tgt_mask.dtype(), &append_mask);

  DenseTensor pos_extra_out;
  pos_extra_out.Resize(tgt_pos_extra.dims());
  ctx.template Alloc<int64_t>(&pos_extra_out);

  DenseTensor out_scores, out_indices;
  DenseTensor pre_scores(init_score);

  DenseTensor ln_out, ln_mean, ln_var, fc_out1, fc_out2;
  ln_out.Resize({{bsz, 1, dim_embed}});
  ln_mean.Resize({{bsz}});
  ln_var.Resize({{bsz}});
  fc_out1.Resize({{bsz, 1, dim_embed}});
  ctx.template Alloc<T>(&fc_out1);
  fc_out2.Resize({{bsz, 1, vocab_size}});
  ctx.template Alloc<T>(&fc_out2);

  DenseTensor lm_out;
  lm_out.Resize({{bsz, 1, vocab_size}});
  ctx.template Alloc<T>(&lm_out);

  DenseTensor softmax_out;
  softmax_out.Resize({{bsz, 1, vocab_size}});

  DenseTensor topk_scores, topk_indices;
  topk_scores.Resize({{bsz, 1}});
  topk_indices.Resize({{bsz, 1}});

  std::vector<int64_t> h_idx(bsz);

  // Reused vars resize
  emb_out.Resize({{bsz, 1, dim_embed}});

  for (int step = 0; step < max_dec_len; ++step) {
    VLOG(1) << "step " << step;
    // Process pos_ids_extra
    DenseTensor pos_extra_bias;
    FullKernel<int64_t, Context>(
        ctx, {bsz, 1}, step, tgt_pos_extra.dtype(), &pos_extra_bias);

    VLOG(1) << pos_extra_bias.dtype() << " " << tgt_pos_extra.dtype() << " "
            << pos_extra_out.dtype();
    std::vector<const DenseTensor*> pos_extra_ins = {&pos_extra_bias,
                                                     &tgt_pos_extra};
    std::vector<DenseTensor*> pos_extra_outs = {&pos_extra_out};
    phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary,
                                int64_t,
                                int64_t>(ctx,
                                         pos_extra_ins,
                                         &pos_extra_outs,
                                         -1,
                                         phi::funcs::AddFunctor<int64_t>());

    // Embedding
    VLOG(1) << "generation Embedding";
    ErnieEmbedding<T, Context>(
        ctx, tgt_ids, tgt_pos, pos_extra_out, embedding_weight, &emb_out);
    // Process mask (concat + scale + stack)
    VLOG(1) << "generation Process mask";
    DenseTensor tmp_mask(concated_mask);
    std::vector<const DenseTensor*> concat_ins{&tmp_mask, &append_mask};
    ConcatKernel<T, Context>(ctx, concat_ins, 2, &concated_mask);

    scale_mask.Resize(concated_mask.dims());
    attn_mask.Resize({{bsz, num_head, 1, concated_mask.dims()[2]}});

    ScaleKernel<T, Context>(ctx, concated_mask, 1e4, -1, false, &scale_mask);

    std::vector<const DenseTensor*> stack_ins(num_head, &scale_mask);
    StackKernel<T, Context>(ctx, stack_ins, 1, &attn_mask);

    // Fuse MT
    VLOG(1) << "generation Fuse MT";
    FusedMultiTransformerKernel<T, Context>(ctx,
                                            emb_out,
                                            attn_mask,
                                            qkv_weight,
                                            qkv_bias,
                                            out_linear_weight,
                                            out_linear_bias,
                                            ffn1_weight,
                                            ffn1_bias,
                                            ffn2_weight,
                                            ffn2_bias,
                                            attn_ln_scale,
                                            attn_ln_bias,
                                            ffn_ln_scale,
                                            ffn_ln_bias,
                                            input_seq_len + step + 1,
                                            true,
                                            1e-9,
                                            "gelu",
                                            &enc_out,
                                            cache_kvs);

    // Calculate Logits
    // layernorm
    VLOG(1) << "generation LayerNormKernel";

    LayerNormKernel<T, Context>(ctx,
                                enc_out,
                                post_encoder_ln_scale,
                                post_encoder_ln_bias,
                                1e-9,
                                2,
                                &ln_out,
                                &ln_mean,
                                &ln_var);
    // fc
    VLOG(1) << "generation fc1";

    auto blas = funcs::GetBlas<Context, T>(ctx);
    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              bsz,
              dim_embed,
              dim_embed,
              static_cast<T>(1.0),
              ln_out.data<T>(),
              mask_lm_trans_fc_weight.data<T>(),
              static_cast<T>(0.0),
              fc_out1.data<T>());

    auto block_num = GetDesiredBlockDim(vocab_size);
    BiasGelu<<<bsz, block_num, 0, ctx.stream()>>>(
        fc_out1.data<T>(),
        mask_lm_trans_fc_bias.data<T>(),
        fc_out1.numel(),
        bsz);
    // layernorm
    VLOG(1) << "generation ln2";
    LayerNormKernel<T, Context>(ctx,
                                fc_out1,
                                mask_lm_trans_ln_scale,
                                mask_lm_trans_ln_bias,
                                1e-9,
                                2,
                                &ln_out,
                                &ln_mean,
                                &ln_var);

    // fc
    VLOG(1) << "generation fc2";

    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              bsz,
              vocab_size,
              dim_embed,
              static_cast<T>(1.0),
              ln_out.data<T>(),
              mask_lm_out_fc_weight.data<T>(),
              static_cast<T>(0.0),
              fc_out2.data<T>());

    std::vector<const DenseTensor*> lm_out_ins = {&fc_out2,
                                                  &mask_lm_out_fc_bias};
    std::vector<DenseTensor*> lm_out_outs = {&lm_out};
    phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
        ctx, lm_out_ins, &lm_out_outs, -1, phi::funcs::AddFunctor<T>());
    // softmax
    VLOG(1) << "generation SoftmaxKernel";
    // SoftmaxKernel<T, Context>(ctx, lm_out, -1, &softmax_out);
    ctx.template Alloc<T>(&softmax_out);
    SoftmaxForwardCUDAKernelDriver<T>(ctx, lm_out, -1, &softmax_out);
    // sampling

    VLOG(1) << "generation TopKSamplingKernel";
    TopKSamplingKernel<T, Context>(
        ctx, softmax_out, topk, 0, &topk_scores, &topk_indices);
    // log and add pre scores
    LogAdd<T><<<bsz, 1, 0, ctx.stream()>>>(
        topk_scores.data<T>(), pre_scores.data<T>(), topk_scores.numel());

    pre_scores = topk_scores;

    // Concat
    VLOG(1) << "generation concat scores";
    if (step == 0) {
      out_scores = topk_scores;
      out_indices = topk_indices;
    } else {
      DenseTensor tmp_scores, tmp_indices;
      std::vector<const DenseTensor*> concat_ins{&out_scores, &topk_scores};
      ConcatKernel<T, Context>(ctx, concat_ins, 1, &tmp_scores);
      out_scores = tmp_scores;
      concat_ins = std::vector<const DenseTensor*>{&out_indices, &topk_indices};
      ConcatKernel<int64_t, Context>(ctx, concat_ins, 1, &tmp_indices);
      out_indices = tmp_indices;
    }

    // Whether to stop
    VLOG(1) << "sync and copy";
    cudaMemcpy(h_idx.data(),
               topk_indices.data<int64_t>(),
               topk_indices.numel() * sizeof(int64_t),
               cudaMemcpyDeviceToHost);

    bool finish_flag = true;
    for (int b = 0; b < bsz; ++b) {
      finish_flag &= (h_idx[b] == end_idx);
    }

    if (finish_flag) {
      break;
    }
  }
  scores = &out_scores;
  indices = &out_indices;
}
}  // namespace phi

PD_REGISTER_KERNEL(ernie_for_inference,
                   GPU,
                   ALL_LAYOUT,
                   phi::ErnieForInferenceKernel,
                   float,
                   phi::dtype::float16) {}
