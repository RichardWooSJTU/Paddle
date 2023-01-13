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

#include "paddle/phi/kernels/embedding_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace phi {

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
    const std::vector<DenseTensor>&
        embedding_weight,  // word_embed/pos_embed/pos_exra_embed
    const std::vector<DenseTensor>& qkv_weight,
    const std::vector<DenseTensor>& qkv_bias,
    const std::vector<DenseTensor>& out_linear_weight,
    const std::vector<DenseTensor>& out_linear_bias,
    const std::vector<DenseTensor>& ffn1_weight,
    const std::vector<DenseTensor>& ffn1_bias,
    const std::vector<DenseTensor>& ffn2_weight,
    const std::vector<DenseTensor>& ffn2_bias,
    const std::vector<DenseTensor>& attn_ln_scale,
    const std::vector<DenseTensor>& attn_ln_bias,
    const std::vector<DenseTensor>& ffn_ln_scale,
    const std::vector<DenseTensor>& ffn_ln_bias,
    bool decoding_strategy,
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

  DenseTensor word_emb_out, pos_embed_out, pos_extra_embed_out, emb_out;
  emb_out.Resize({{bsz, input_seq_len, dim_embed}});
  ctx.template Alloc<T>(&emb_out);
  pos_embed_out.Resize({{bsz, input_seq_len, dim_embed}});
  ctx.template Alloc<T>(&pos_embed_out);
  pos_extra_embed_out.Resize({{bsz, input_seq_len, dim_embed}});
  ctx.template Alloc<T>(&pos_extra_embed_out);
  emb_out.Resize({{bsz, input_seq_len, dim_embed}});
  ctx.template Alloc<T>(&emb_out);

  EmbeddingKernel<T, Context>(
      ctx, src_ids, embedding_weight[0], -1, &word_emb_out);
  EmbeddingKernel<T, Context>(
      ctx, pos_ids, embedding_weight[1], -1, &pos_embed_out);
  EmbeddingKernel<T, Context>(
      ctx, pos_ids_extra, embedding_weight[2], -1, &pos_extra_embed_out);

  std::vector<const phi::DenseTensor*> embed_add_ins = {
      word_emb_out, pos_embed_out, pos_extra_embed_out};
  std::vector<phi::DenseTensor*> embed_add_outs = {emb_out};
  phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
      ctx, embed_add_ins, &embed_add_outs, -1, phi::funcs::AddFunctor<T>());

  // Process Mask

  DenseTensor scale_mask, attn_mask;
  scale_mask.Resize({{bsz, input_seq_len, dim_embed}});
  ctx.template Alloc<T>(&scale_mask);
  attn_mask.Resize({{bsz, num_head, input_seq_len, dim_embed}});
  ctx.template Alloc<T>(&attn_mask);

  ScaleKernel<T, Context>(ctx, input_mask, 1e4, -1, false, &scale_mask);

  std::vector<const DenseTensor*> stack_ins(num_head, &scale_mask);
  StackKernel<T, Context>(ctx, stack_ins, 1, attn_mask);
  // PreProcess (Layernorm)

  // Fuse MT
  std::vector<DenseTensor> cache_kvs(num_layers);
  for (int i = 0; i < num_layers; ++i) {
    cache_kvs[i].Resize(
        {{2, bsz, num_head, input_seq_len + max_dec_len, dim_head}});
    ctx.template Alloc<T>(&cache_kvs[i]);
  }

  for (int step = 0; step < max_dec_len; ++step) {
    bool finish_flag = false;

    if (finish_flag) {
      break;
    }
  }

  ctx.template Alloc<T>(scores);
  ctx.template Alloc<T>(indices);
}
}  // namespace phi

PD_REGISTER_KERNEL(ernie_for_inference,
                   GPU,
                   ALL_LAYOUT,
                   phi::ErnieForInferenceKernel,
                   float,
                   phi::dtype::float16) {}
