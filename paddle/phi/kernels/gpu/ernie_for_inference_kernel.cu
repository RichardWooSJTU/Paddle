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
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/fused_multi_transformer_kernel.h"
#include "paddle/phi/kernels/layer_norm_kernel.h"
#include "paddle/phi/kernels/softmax_kernel.h"

namespace phi {

template <typename T>
__global__ void LogAdd(T* data, const T* add_data, const int num) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < num; tid += stride) {
    auto tmp = log(static_cast<float>(data[tid]));
    data[tid] = tmp + add_data[tid];
  }
#endif
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
  DenseTensor word_emb_out, pos_embed_out, pos_extra_embed_out;
  emb_out->Resize({{bsz, input_seq_len, dim_embed}});
  ctx.template Alloc<T>(emb_out);

  EmbeddingKernel<T, Context>(
      ctx, src_ids, embedding_weight[0], -1, word_emb_out);
  EmbeddingKernel<T, Context>(
      ctx, pos_ids, embedding_weight[1], -1, &pos_embed_out);
  EmbeddingKernel<T, Context>(
      ctx, pos_ids_extra, embedding_weight[2], -1, pos_extra_embed_out);

  std::vector<const DenseTensor*> embed_add_ins = {
      word_emb_out, pos_embed_out, pos_extra_embed_out};
  std::vector<DenseTensor*> embed_add_outs = {emb_out};
  phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
      ctx, embed_add_ins, &embed_add_outs, -1, phi::funcs::AddFunctor<T>());
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
    const DenseTensor& max_dec_len,
    const DenseTensor& min_dec_len,
    const DenseTensor& topk,
    const DenseTensor& topp,
    const DenseTensor& topk,
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
    bool decoding_strategy,
    int64_t end_idx,
    DenseTensor* scores,
    DenseTensor* indices) {
  auto input_dims = src_ids.dims();
  int bsz = input_dims[0];
  int input_seq_len = input_dims[1];
  int vocab_size = embedding_weight[0].dims()[0];
  int dim_embed = embedding_weight[0].dims()[1];
  int num_layers = qkv_weight.size();
  int num_head = qkv_weight[0].dims()[1];
  int dim_head = qkv_weight[0].dims()[2];

  // Embedding

  DenseTensor emb_out;

  ErnieEmbedding(
      ctx, src_ids, pos_ids, pos_ids_extra, embedding_weight, &emb_out);

  // Process Mask

  DenseTensor scale_mask, attn_mask;

  ScaleKernel<T, Context>(ctx, input_mask, 1e4, -1, false, &scale_mask);

  std::vector<const DenseTensor*> stack_ins(num_head, &scale_mask);
  StackKernel<T, Context>(ctx, stack_ins, 1, attn_mask);
  // PreProcess (Layernorm)

  // Fuse MT
  std::vector<DenseTensor*> cache_kvs(num_layers);
  DenseTensor enc_out;

  for (int i = 0; i < num_layers; ++i) {
    cache_kvs[i]->Resize(
        {{2, bsz, num_head, input_seq_len + max_dec_len, dim_head}});
    ctx.template Alloc<T>(cache_kvs[i]);
  }

  FusedMultiTransformerKernel(ctx,
                              emb_out,
                              *attn_mask,
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
                              &enc_out,
                              cache_kvs);

  // Loop vars
  DenseTensor concated_mask, pos_extra_out;

  DenseTensor out_scores, out_indices;

  DenseTensor append_mask;
  FullKernel(ctx, {bsz, 1, 1}, tgt_mask.dtype(), 1, &append_mask);

  DenseTensor pre_scores(init_score);

  std::vector<int64_t> h_idx(bsz);

  for (int step = 0; step < max_dec_len; ++step) {
    // Process pos_ids_extra
    DenseTensor pos_extra_bias;
    FullKernel(ctx, {bsz, 1}, tgt_mask.dtype(), step, &pos_extra_bias);
    std::vector<const DenseTensor*> pos_extra_ins = {&pos_extra_bias,
                                                     &tgt_pos_extra};
    std::vector<DenseTensor*> pos_extra_outs = {&pos_extra_out};
    phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
        ctx, pos_extra_ins, &pos_extra_outs, -1, phi::funcs::AddFunctor<T>());

    // Embedding
    ErnieEmbedding(
        ctx, tgt_ids, tgt_pos, pos_extra_out, embedding_weight, &emb_out);
    // Process mask (concat + scale + stack)

  std:
    vector<const DenseTensor*> concat_ins{&tgt_mask, &append_mask};
    ConcatKernel<T, Context>(ctx, concat_ins, 2, &concated_mask);

    ScaleKernel<T, Context>(ctx, concated_mask, 1e4, -1, false, &scale_mask);

    std::vector<const DenseTensor*> stack_ins(num_head, &scale_mask);
    StackKernel<T, Context>(ctx, stack_ins, 1, attn_mask);

    // Fuse MT
    FusedMultiTransformerKernel(ctx,
                                emb_out,
                                *attn_mask,
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
                                input_seq_len + i + 1,
                                true,
                                1e-9,
                                enc_out,
                                cache_kvs);

    // Calculate Logits
    // layernorm
    DenseTensor* ln_out;

    LayerNormKernel<T, Context>(ctx,
                                *enc_out,
                                post_encoder_ln_scale,
                                post_encoder_ln_bias,
                                1e-9,
                                2,
                                ln_out,
                                nullptr,
                                nullptr);
    // fc
    DenseTensor fc_out;
    fc_out.Resize({{bsz, 1, dim_embed}});
    ctx.template Alloc<T>(&fc_out);

    auto blas = funcs::GetBlas<Context, T>(ctx);
    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              bsz,
              dim_embed,
              dim_embed,
              1.0,
              ln_out->data<T>(),
              mask_lm_trans_fc_weight.data<T>(),
              0.0,
              fc_out.data<T>());

    auto block_num = GetDesiredBlockDim(vocab_size);
    BiasGelu<<<bsz, block_num, 0, ctx.stream()>>>(
        fc_out.data<T>(),
        mask_lm_trans_fc_bias.data<T>(),
        fc_out.numel(),
        batch_size);
    // layernorm
    LayerNormKernel<T, Context>(ctx,
                                fc_out,
                                mask_lm_trans_ln_scale,
                                mask_lm_trans_ln_bias,
                                1e-9,
                                2,
                                ln_out,
                                nullptr,
                                nullptr);

    // fc
    DenseTensor lm_out;
    fc_out.Resize({{bsz, 1, vocab_size}});
    ctx.template Alloc<T>(&fc_out);
    lm_out.Resize({{bsz, 1, vocab_size}});
    ctx.template Alloc<T>(&lm_out);

    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              bsz,
              vocab_size,
              dim_embed,
              1.0,
              ln_out->data<T>(),
              mask_lm_out_fc_weight.data<T>(),
              0.0,
              fc_out.data<T>());

    std::vector<const DenseTensor*> lm_out_ins = {&fc_out,
                                                  &mask_lm_out_fc_bias};
    std::vector<DenseTensor*> lm_out_outs = {&lm_out};
    phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
        ctx, lm_out_ins, &lm_out_outs, -1, phi::funcs::AddFunctor<T>());
    // softmax
    DenseTensor softmax_out;
    SoftmaxKernel<T, Context>(ctx, lm_out, -1, softmax_out);
    // sampling
    DenseTensor topk_scores, topk_indices;
    topk_scores.Resize({{bsz, 1}});
    ctx.template Alloc<T>(&topk_scores);
    topk_indices.Resize({{bsz, 1}});
    ctx.template Alloc<int64_t>(&topk_indices);

    TopKSamplingKernel<T, Context>(
        ctx, softmax_out, topk, 0, topk_scores, topk_indices);
    // log and add pre scores
    LogAdd<<<batch_size, 1, 0, stream>>>(
        topk_scores.data<T>, pre_scores.data<T>(), topk_scores.numel());

    // Concat
    if (step == 0) {
      out_scores = topk_scores;
      out_indices = topk_indices;
    } else {
      DenseTensor tmp_scores, tmp_indices;
    std:
      vector<const DenseTensor*> concat_ins{&out_scores, &topk_scores};
      ConcatKernel<T, Context>(ctx, concat_ins, 1, &tmp_scores);
      out_scores = tmp_scores;
      concat_ins = std::vector<const DenseTensor*>{&out_indices, &topk_indices};
      ConcatKernel<T, Context>(ctx, concat_ins, 1, &tmp_indices);
      out_indices = tmp_indices;
    }

    // Whether to stop
    cudaMemcpy(h_idx.data(), topk_indices.data<T>(), cudaMemcpyDeviceToHost);

    bool finish_flag = true;
    for (int b = 0; b < bsz; ++b) {
      finish_flag &= (h_idx[b] == end_idx);
    }

    if (finish_flag) {
      break;
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(ernie_for_inference,
                   GPU,
                   ALL_LAYOUT,
                   phi::ErnieForInferenceKernel,
                   float,
                   phi::dtype::float16) {}
