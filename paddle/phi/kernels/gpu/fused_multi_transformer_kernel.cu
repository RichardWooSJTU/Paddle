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

#include "paddle/phi/kernels/fused_multi_transformer_kernel.h"
#include "paddle/fluid/operators/fused/fused_multi_transformer_op.cu.h"

#include <vector>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void FusedMultiTransformerKernel(
    const Context &ctx,
    const DenseTensor &x,
    const DenseTensor &src_mask,
    const std::vector<const DenseTensor *> &qkv_weights,
    const std::vector<const DenseTensor *> &qkv_biases,
    const std::vector<const DenseTensor *> &out_linear_weighta,
    const std::vector<const DenseTensor *> &out_linear_biases,
    const std::vector<const DenseTensor *> &ffn1_weights,
    const std::vector<const DenseTensor *> &ffn1_biases,
    const std::vector<const DenseTensor *> &ffn2_weights,
    const std::vector<const DenseTensor *> &ffn2_biases,
    const std::vector<const DenseTensor *> &ln_scales,
    const std::vector<const DenseTensor *> &ln_biass,
    const std::vector<const DenseTensor *> &ffn_ln_scales,
    const std::vector<const DenseTensor *> &ffn_ln_biases,
    int time_step,
    bool pre_layer_norm,
    float epsilon,
    DenseTensor *out,
    std::vector<DenseTensor *> cache_kvs) {
  using U = LayerNormParamType<T>;
  // 0. input/out
  const auto input_x_dims = x.dims();
  int bsz = input_x_dims[0];
  int seq_len = input_x_dims[1];
  int dim_embed = input_x_dims[2];
  int bsz_seq = bsz * seq_len;
  const std::string act_method = ctx.Attr<std::string>("act_method");

  out->Resize({{bsz, seq_len, dim_embed}});
  ctx.template Alloc<T>(out);

  // 1. layer norm
  auto ln_compute =
      paddle::operators::AttnLayerNorm<T>(ctx, epsilon, bsz_seq, dim_embed);
  DenseTensor ln_mean, ln_var;
  ln_mean.Resize({{bsz_seq}});
  auto *ln_mean_data = ctx.Alloc<U>(&ln_mean, ln_mean.numel() * sizeof(U));
  ln_var.Resize({{bsz_seq}});
  auto *ln_var_data = ctx.Alloc<U>(&ln_var, ln_var.numel() * sizeof(U));

  // 2. qkv
  // x: qkv's input [batch_size, seq_len, dim_embed]
  // y: qkv's weight: [3, num_head, dim_head, dim_embed]
  const bool trans_qkvw = true;
  const auto qkv_w_dims = qkv_weights[0].dims();
  int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
  int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
  int hidden_size = num_head * dim_head;
  int output_size = 3 * hidden_size;
  int input_size = dim_embed;

  bool compute_bias = qkv_biases.size() > 0 && time_step == nullptr;
  // (transA, transB, compute_bias) = (false, trans_qkvw, false)
  // Since we fused QKVBias into QKVBiasAddTransposeSplit kernel, here we
  // set compute_bias as false.
  auto qkv_compute = paddle::operators::AttnMatMul<T>(ctx,
                                                      false,
                                                      trans_qkvw,
                                                      bsz_seq,
                                                      output_size,
                                                      input_size,
                                                      /*compute_bias=*/false);

  DenseTensor qkv_out;
  qkv_out.Resize({{bsz, seq_len, 3, num_head, dim_head}});

  auto *qkv_out_data = ctx.Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

  // 3. fmha
  paddle::operators::AttnDropoutParam attn_param(
      true, "upscale_in_train", 0.0, true, true, 0, nullptr);
  auto fmha_compute = paddle::operators::FMHARef<T>(
      ctx, bsz, seq_len, num_head, dim_head, attn_param);
  auto cache_kv_outs = cache_kvs;
  // auto pre_caches = ctx.MultiInput<DenseTensor>("PreCaches");
  std::vector<DenseTensor> pre_caches;
  int cache_offset = 0;
  if (pre_caches.size() > 0) {
    cache_offset = pre_caches[0]->dims()[3];
  }

  auto out_seq_len = seq_len;
  if (time_step != -1) {
    PADDLE_ENFORCE_GT(
        time_step,
        0,
        platform::errors::PreconditionNotMet(
            "The value of time_step must > 0, but now is %d", time_step));
    PADDLE_ENFORCE_EQ(
        seq_len,
        1,
        platform::errors::PreconditionNotMet(
            "In decode stage, the seq_len of input must be 1, but now is %d",
            seq_len));
    out_seq_len += time_step;
  } else {
    out_seq_len += cache_offset;
  }

  DenseTensor q_transpose_out, kv_transpose_out, qk_out;
  q_transpose_out.Resize({{bsz, num_head, seq_len, dim_head}});
  auto *q_transpose_out_data =
      ctx.Alloc<T>(&q_transpose_out, q_transpose_out.numel() * sizeof(T));

  kv_transpose_out.Resize({{2, bsz, num_head, seq_len, dim_head}});
  auto *kv_transpose_out_data =
      ctx.Alloc<T>(&kv_transpose_out, kv_transpose_out.numel() * sizeof(T));

  qk_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
  auto *qk_out_data = ctx.Alloc<T>(&qk_out, qk_out.numel() * sizeof(T));

  DenseTensor src_mask_out;
  if (cache_offset > 0) {
    src_mask_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
    auto *src_mask_out_data =
        ctx.Alloc<T>(&src_mask_out, src_mask_out.numel() * sizeof(T));
  }

  // [2, bs, num_head, cache_seq_len + seq_len, head_dim]
  DenseTensor pre_cache_kv_out;
  if (cache_offset > 0) {
    pre_cache_kv_out.Resize(
        {{2, bsz, num_head, seq_len + cache_offset, dim_head}});
    auto *pre_cache_kv_out_data =
        ctx.Alloc<T>(&pre_cache_kv_out, pre_cache_kv_out.numel() * sizeof(T));
  }

  DenseTensor softmax_out;
  DenseTensor attn_dropout_mask_out, attn_dropout_out;
  DenseTensor qktv_out, fmha_out;
  softmax_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
  auto *softmax_out_data =
      ctx.Alloc<T>(&softmax_out, softmax_out.numel() * sizeof(T));

  attn_dropout_mask_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
  auto *attn_dropout_mask_out_data = ctx.Alloc<T>(
      &attn_dropout_mask_out, attn_dropout_mask_out.numel() * sizeof(T));
  attn_dropout_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
  auto *attn_dropout_data_data =
      ctx.Alloc<T>(&attn_dropout_out, attn_dropout_out.numel() * sizeof(T));

  qktv_out.Resize({{bsz, num_head, seq_len, dim_head}});
  auto *qktv_out_data = ctx.Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
  fmha_out.Resize({{bsz, seq_len, num_head, dim_head}});
  auto *fmha_out_data = ctx.Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));

  // 4. out_linear
  int ring_id = 0;  // TODO(@wufeisheng): Need as input
  // (transA, transB, compute_bias) = (false, false, false)
  auto out_linear_compute = paddle::operators::AttnMatMul<T>(
      ctx, false, false, bsz_seq, dim_embed, hidden_size, false);

  // 5. ln(residual + bias)
  paddle::operators::DropoutParam dropout_param2(
      true, 0, true, true, 0.0, nullptr, 0);
  paddle::operators::FusedDropoutLayerNormHelper<T, uint8_t>
      fused_dropout_layernorm_helper(
          ctx, bsz_seq, dim_embed, dropout_param2, epsilon);
  DenseTensor bias_dropout_residual_out, dropout_mask_out;
  T *bias_dropout_residual_out_data = nullptr;
  if (pre_layer_norm) {
    bias_dropout_residual_out.Resize({{bsz, seq_len, dim_embed}});
    bias_dropout_residual_out_data =
        ctx.Alloc<T>(&bias_dropout_residual_out,
                     bias_dropout_residual_out.numel() * sizeof(T));
  }
  dropout_mask_out.Resize({{bsz, seq_len, dim_embed}});
  auto *dropout_mask_out_data = ctx.Alloc<uint8_t>(
      &dropout_mask_out, dropout_mask_out.numel() * sizeof(uint8_t));

  // 6. ffn matmul1
  auto ffn1_weight_dim = ffn1_weights[0].dims();

  int dim_ffn = ffn1_weight_dim[1];
  auto ffn1_linear_compute = paddle::operators::AttnMatMul<T>(
      ctx, false, false, bsz_seq, dim_ffn, dim_embed, false);
  DenseTensor ffn1_out;
  ffn1_out.Resize({{bsz_seq, dim_ffn}});
  auto *ffn1_out_data = ctx.Alloc<T>(&ffn1_out, ffn1_out.numel() * sizeof(T));

  // 7. ffn act + bias
  paddle::operators::DropoutParam ffn1_dropout_param(
      true, 0, true, true, 0.0, nullptr, 0);
  paddle::operators::FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
      ctx, bsz_seq, dim_ffn, ffn1_dropout_param);
  DenseTensor ffn1_dropout_out, ffn1_dropout_mask;
  ffn1_dropout_out.Resize({{bsz_seq, dim_ffn}});
  auto *ffn1_dropout_out_data =
      ctx.Alloc<T>(&ffn1_dropout_out, ffn1_dropout_out.numel() * sizeof(T));
  ffn1_dropout_mask.Resize({{bsz_seq, dim_ffn}});
  auto *ffn1_dropout_mask_data = ctx.Alloc<uint8_t>(
      &ffn1_dropout_mask, ffn1_dropout_mask.numel() * sizeof(uint8_t));

  // 8. ffn2 matmul
  auto ffn2_linear_compute = paddle::operators::AttnMatMul<T>(
      ctx, false, false, bsz_seq, dim_embed, dim_ffn, false);

  // 9. ffn2 residual bias
  paddle::operators::DropoutParam ffn2_dropout_param(
      true, 0, true, true, 0.0, nullptr, 0);
  paddle::operators::FusedDropoutLayerNormHelper<T, uint8_t>
      ffn2_fused_dropout_helper(
          ctx, bsz_seq, dim_embed, ffn2_dropout_param, epsilon);

  // calc
  auto *from_data = ctx.Alloc<T>(out, out->numel() * sizeof(T));
  DenseTensor *from_tensor = out;
  DenseTensor tmp_out;
  tmp_out.Resize({{bsz, seq_len, dim_embed}});
  auto *tmp_out_data = ctx.Alloc<T>(&tmp_out, tmp_out.numel() * sizeof(T));

  auto *x_data = x.data<T>();
  DenseTensor *buf0 = nullptr;
  DenseTensor *buf1 = nullptr;

  // step0:  x   --> buf1
  // step1: buf1 --> buf0
  // step2: buf0 --> buf1
  int layers = qkv_weights.size();
  if (pre_layer_norm) {
    if (layers & 1) {
      // odd, set buf1 as out
      buf0 = &tmp_out;
      buf1 = out;
    } else {
      // even, set buf0 as out
      buf0 = out;
      buf1 = &tmp_out;
    }
  } else {
    buf0 = &tmp_out;
    buf1 = out;
  }

  for (int i = 0; i < layers; ++i) {
    // step1. layer_norm
    if (i == 0 && pre_layer_norm) {
      auto *ln_scale_data = ln_scales[i].data<U>();
      auto *ln_bias_data = ln_biases[i].data<U>();
      // TODO(wangxi): can remove mean var in inference
      ln_compute.ComputeForward(x_data,
                                ln_scale_data,
                                ln_bias_data,
                                buf1->data<T>(),
                                ln_mean_data,
                                ln_var_data);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step1";
#endif

    // step2. qkv
    const DenseTensor *qkv_bias =
        qkv_biases.size() > 0 ? &qkv_biases[i] : nullptr;
    // NOTE: in decoder stage, bias is fused in fmha
    const DenseTensor *bias = time_step ? nullptr : qkv_bias;
    if (!pre_layer_norm && i == 0) {
      qkv_compute.ComputeForward(qkv_weights[i], x, bias, &qkv_out, &qkv_out);
    } else {
      qkv_compute.ComputeForward(
          qkv_weights[i], buf1, bias, &qkv_out, &qkv_out);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step2";
#endif

    // step3. fmha
    const DenseTensor *cache_kv =
        cache_kvs.size() > 0 ? &cache_kvs[i] : nullptr;
    DenseTensor *cache_kv_out = cache_kv ? &cache_kv_outs[i] : nullptr;

    if (time_step) {  // generation decoder stage
      // [2, batch_size, num_head, max_seq_len, head_size]
      int max_seq_len = cache_kv->dims()[3];
      paddle::operators::fmha<T>(ctx,
                                 qkv_out,
                                 *qkv_bias,
                                 src_mask,
                                 cache_kv_out,
                                 &fmha_out,
                                 bsz,
                                 max_seq_len,
                                 num_head,
                                 dim_head,
                                 time_step->data<int>()[0],
                                 1. / sqrt(dim_head));
    } else if (cache_kv_out) {  // generation context stage
      const DenseTensor *pre_cache_kv_tensor =
          pre_caches.size() > 0 ? pre_caches[i] : nullptr;
      DenseTensor *pre_cache_kv_out_tmp =
          cache_offset > 0 ? &pre_cache_kv_out : nullptr;
      DenseTensor *src_mask_tmp = cache_offset > 0 ? &src_mask_out : nullptr;
      paddle::operators::qkv_bias_add_transpose_split<T>(ctx,
                                                         q_transpose_out_data,
                                                         kv_transpose_out_data,
                                                         qkv_out_data,
                                                         qkv_bias->data<T>(),
                                                         bsz,
                                                         num_head,
                                                         seq_len,
                                                         dim_head,
                                                         compute_bias);
      fmha_compute.ComputeForwardWithoutTranspose(qkv_out,
                                                  pre_cache_kv_tensor,
                                                  src_mask,
                                                  &q_transpose_out,
                                                  &kv_transpose_out,
                                                  pre_cache_kv_out_tmp,
                                                  &qk_out,
                                                  src_mask_tmp,
                                                  &softmax_out,
                                                  &attn_dropout_mask_out,
                                                  &attn_dropout_out,
                                                  &qktv_out,
                                                  &fmha_out);
      const T *k_ptr = nullptr;
      const T *v_ptr = nullptr;

      if (cache_offset > 0) {
        // [2, bsz, num_head, cache_offset + seq_len, head_dim]
        const T *kv_data = pre_cache_kv_out.data<T>();
        k_ptr = kv_data;
        int64_t k_size = bsz * num_head * (seq_len + cache_offset) * dim_head;
        v_ptr = k_ptr + k_size;
      } else {
        // [3, bsz, num_head, seq_len, head_dim]
        int64_t k_size = bsz * seq_len * num_head * dim_head;
        const T *q_ptr = q_transpose_out_data;
        k_ptr = kv_transpose_out_data;
        v_ptr = k_ptr + k_size;
      }

      // [2, bsz, num_head, max_seq_len, head_dim]
      int max_seq_len = cache_kv_out->dims()[3];
      T *cache_kv_data = cache_kv_out->data<T>();
      int64_t cache_k_size = bsz * num_head * max_seq_len * dim_head;

      T *cache_k_ptr = cache_kv_data;
      T *cache_v_ptr = cache_kv_data + cache_k_size;

      const int seq_len_tmp = seq_len + cache_offset;
      paddle::operators::write_cache_kv<T>(ctx,
                                           cache_k_ptr,
                                           cache_v_ptr,
                                           k_ptr,
                                           v_ptr,
                                           bsz,
                                           num_head,
                                           seq_len_tmp,
                                           max_seq_len,
                                           dim_head);
    } else {  // not generation
      // TODO(wangxi): can remove dropout in inference
      paddle::operators::qkv_bias_add_transpose_split<T>(ctx,
                                                         q_transpose_out_data,
                                                         kv_transpose_out_data,
                                                         qkv_out_data,
                                                         qkv_bias->data<T>(),
                                                         bsz,
                                                         num_head,
                                                         seq_len,
                                                         dim_head,
                                                         compute_bias);
      fmha_compute.ComputeForwardWithoutTranspose(qkv_out,
                                                  cache_kv,
                                                  src_mask,
                                                  &q_transpose_out,
                                                  &kv_transpose_out,
                                                  cache_kv_out,
                                                  &qk_out,
                                                  nullptr,
                                                  &softmax_out,
                                                  &attn_dropout_mask_out,
                                                  &attn_dropout_out,
                                                  &qktv_out,
                                                  &fmha_out);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step3";
#endif

    if (pre_layer_norm) {
      out_linear_compute.ComputeForward(
          &out_linear_weights[i], &fmha_out, nullptr, buf1, nullptr);
      AllReduce<T>(*buf1, ring_id, buf1->numel(), ctx);
    } else {
      out_linear_compute.ComputeForward(
          &out_linear_weights[i], &fmha_out, nullptr, buf0, nullptr);
      AllReduce<T>(*buf0, ring_id, buf0->numel(), ctx);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step4";
#endif

    // step5. ln(residual + dropout(input + bias))
    if (pre_layer_norm) {
      auto *ln_scale_data = ffn_ln_scales[i].data<U>();
      auto *ln_bias_data = ffn_ln_biases[i].data<U>();
      auto *out_linear_bias_data = out_linear_biases[i].data<T>();

      // inplace
      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          ctx,
          buf1->data<T>(),
          x_data,
          out_linear_bias_data,
          ln_scale_data,
          ln_bias_data,
          bias_dropout_residual_out_data,
          dropout_mask_out_data,
          buf1->data<T>(),
          ln_mean_data,
          ln_var_data);
    } else {
      auto *ln_scale_data = ln_scales[i].data<U>();
      auto *ln_bias_data = ln_biases[i]->data<U>();
      auto *out_linear_bias_data = out_linear_biases[i]->data<T>();
      auto *residual_data = (i == 0 ? x_data : buf1->data<T>());
      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          ctx,
          buf0->data<T>(),
          residual_data,
          out_linear_bias_data,
          ln_scale_data,
          ln_bias_data,
          buf0->data<T>(),
          dropout_mask_out_data,
          buf1->data<T>(),
          ln_mean_data,
          ln_var_data);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step5";
#endif

    // step6. ffn matmul1
    ffn1_linear_compute.ComputeForward(
        ffn1_weights[i], buf1, nullptr, &ffn1_out, nullptr);
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step6";
#endif

    // step7. act bias
    // TODO(wangxi): remove dropout mask in inference
    fused_act_dropout_helper.DropoutActBias(ctx,
                                            ffn1_out_data,
                                            ffn1_biases[i]->data<T>(),
                                            act_method,
                                            ffn1_dropout_out_data,
                                            ffn1_dropout_mask_data);
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step7";
#endif

    // step8. ffn matmul2
    if (pre_layer_norm) {
      ffn2_linear_compute.ComputeForward(
          ffn2_weights[i], &ffn1_dropout_out, nullptr, buf1, nullptr);
    } else {
      ffn2_linear_compute.ComputeForward(
          ffn2_weights[i], &ffn1_dropout_out, nullptr, buf0, nullptr);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step8.0";
#endif

    if (pre_layer_norm) {
      AllReduce<T>(*buf1, ring_id, buf1->numel(), ctx);
    } else {
      AllReduce<T>(*buf0, ring_id, buf0->numel(), ctx);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step8.1";
#endif

    // step9. residual bias
    if (pre_layer_norm) {
      // TODO(wangxi): remove dropout mask in inference
      if (i < layers - 1) {
        auto *ln_scale_data = ln_scales[i + 1]->data<U>();
        auto *ln_bias_data = ln_biases[i + 1]->data<U>();
        ffn2_fused_dropout_helper.LayernormResidualDropoutBias(
            ctx,
            buf1->data<T>(),
            bias_dropout_residual_out_data,
            ffn2_biases[i]->data<T>(),
            ln_scale_data,
            ln_bias_data,
            buf1->data<T>(),
            dropout_mask_out_data,
            buf0->data<T>(),
            ln_mean_data,
            ln_var_data);
      } else {
        ffn2_fused_dropout_helper.ResidualDropoutBias(
            ctx,
            buf1->data<T>(),
            bias_dropout_residual_out_data,
            ffn2_biases[i]->data<T>(),
            buf1->data<T>(),
            dropout_mask_out_data);
      }
    } else {
      auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
      auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
      ffn2_fused_dropout_helper.LayernormResidualDropoutBias(
          ctx,
          buf0->data<T>(),
          buf1->data<T>(),
          ffn2_biases[i]->data<T>(),
          ln_scale_data,
          ln_bias_data,
          buf0->data<T>(),
          dropout_mask_out_data,
          buf1->data<T>(),
          ln_mean_data,
          ln_var_data);
    }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    VLOG(0) << "step9";
#endif
    if (pre_layer_norm) {
      x_data = buf1->data<T>();
      std::swap(buf0, buf1);
    }
  }
}
}  // namespace phi
