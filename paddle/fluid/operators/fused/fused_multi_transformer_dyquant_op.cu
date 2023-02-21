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


#include "paddle/fluid/operators/fused/attn_gemm_int8.h"
#include "paddle/fluid/operators/fused/fused_multi_transformer_op.cu.h"

namespace paddle {
namespace operators {


template <typename T>
class FusedMultiTransformerDyquantOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    auto &dev_ctx = ctx.cuda_device_context();

    auto *time_step = ctx.Input<phi::DenseTensor>("TimeStep");
    // 0. input
    auto *input_x = ctx.Input<phi::DenseTensor>("X");
    const auto input_x_dims = input_x->dims();
    int bsz = input_x_dims[0];
    int seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];
    int bsz_seq = bsz * seq_len;
    const std::string act_method = ctx.Attr<std::string>("act_method");


    // quant round type and bound
    auto quant_round_type = ctx.Attr<int>("quant_round_type");
    auto quant_max_bound = ctx.Attr<float>("quant_max_bound");
    auto quant_min_bound = ctx.Attr<float>("quant_min_bound");

    // auto qkv_out_scales = ctx.MultiInput<phi::DenseTensor>("QKVOutScale");
    // auto out_linear_out_scales =
    //     ctx.MultiInput<phi::DenseTensor>("OutLinearOutScale");
    // auto ffn1_out_scales = ctx.MultiInput<phi::DenseTensor>("FFN1OutScale");
    auto ffn2_out_scales = ctx.MultiInput<phi::DenseTensor>("FFN2OutScale");

    auto qkv_weight_ranges = ctx.MultiInput<phi::DenseTensor>("QKVOutScale");
    auto out_linear_out_weight_ranges =
        ctx.MultiInput<phi::DenseTensor>("OutLinearOutScale");
    auto ffn1_weight_ranges= ctx.MultiInput<phi::DenseTensor>("FFN1OutScale");
    // auto ffn2_weight_ranges = ctx.MultiInput<phi::DenseTensor>("FFN2OutScale");

    // 1. layer norm
    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    const float epsilon = ctx.Attr<float>("epsilon");
    auto ln_scales = ctx.MultiInput<phi::DenseTensor>("LnScale");
    auto ln_biases = ctx.MultiInput<phi::DenseTensor>("LnBias");

    auto ln_compute = AttnLayerNorm<T>(dev_ctx, epsilon, bsz_seq, dim_embed);
    phi::DenseTensor ln_mean, ln_var;
    ln_mean.Resize({{bsz_seq}});
    auto *ln_mean_data =
        dev_ctx.Alloc<U>(&ln_mean, ln_mean.numel() * sizeof(U));
    ln_var.Resize({{bsz_seq}});
    auto *ln_var_data = dev_ctx.Alloc<U>(&ln_var, ln_var.numel() * sizeof(U));

    // 2. qkv
    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto qkv_weights = ctx.MultiInput<phi::DenseTensor>("QKVW");
    auto qkv_biases = ctx.MultiInput<phi::DenseTensor>("QKVBias");
    const bool trans_qkvw = ctx.Attr<bool>("trans_qkvw");
    const auto qkv_w_dims = qkv_weights[0]->dims();
    int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
    int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
    int hidden_size = num_head * dim_head;
    int output_size = 3 * hidden_size;
    int input_size = dim_embed;

    bool compute_bias = qkv_biases.size() > 0 && time_step == nullptr;
    // (transA, transB, compute_bias) = (false, trans_qkvw, false)
    // Since we fused QKVBias into QKVBiasAddTransposeSplit kernel, here we
    // set compute_bias as false.
    AttnMatmulINT8<T> qkv_compute(dev_ctx, bsz_seq, output_size, input_size, 
                                     /*compute_bias=*/false);

    phi::DenseTensor qkv_out;
    qkv_out.Resize({{bsz, seq_len, 3, num_head, dim_head}});
    auto *qkv_out_data =
        dev_ctx.Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

    // 2.1 rotary
    auto *rotary_tensor = ctx.Input<phi::DenseTensor>("RotaryPosEmb");
    const int rotary_emb_dims = ctx.Attr<int>("rotary_emb_dims");

    // 3. fmha
    AttnDropoutParam attn_param(
        true, "upscale_in_train", 0.0, true, true, 0, nullptr);
    auto fmha_compute =
        FMHARef<T>(dev_ctx, bsz, seq_len, num_head, dim_head, attn_param);
    auto *src_mask = ctx.Input<phi::DenseTensor>("SrcMask");
    auto cache_kvs = ctx.MultiInput<phi::DenseTensor>("CacheKV");
    auto cache_kv_outs = ctx.MultiOutput<phi::DenseTensor>("CacheKVOut");
    // auto *time_step = ctx.Input<phi::DenseTensor>("TimeStep");
    auto pre_caches = ctx.MultiInput<phi::DenseTensor>("PreCaches");
    int cache_offset = 0;
    if (pre_caches.size() > 0) {
      cache_offset = pre_caches[0]->dims()[3];
    }

    auto out_seq_len = seq_len;
    int time_step_value = 0;
    if (time_step) {
      PADDLE_ENFORCE_EQ(time_step->place(),
                        platform::CPUPlace(),
                        platform::errors::PreconditionNotMet(
                            "The place of input(TimeStep) must be CPUPlace."));
      // cache_seq_len
      time_step_value = time_step->data<int>()[0];
      PADDLE_ENFORCE_GT(time_step_value,
                        0,
                        platform::errors::PreconditionNotMet(
                            "The value of time_step must > 0, but now is %d",
                            time_step_value));
      PADDLE_ENFORCE_EQ(
          seq_len,
          1,
          platform::errors::PreconditionNotMet(
              "In decode stage, the seq_len of input must be 1, but now is %d",
              seq_len));
      out_seq_len += time_step_value;
    } else {
      out_seq_len += cache_offset;
    }

    phi::DenseTensor q_transpose_out, kv_transpose_out, qk_out;
    q_transpose_out.Resize({{bsz, num_head, seq_len, dim_head}});
    auto *q_transpose_out_data =
        dev_ctx.Alloc<T>(&q_transpose_out, q_transpose_out.numel() * sizeof(T));

    kv_transpose_out.Resize({{2, bsz, num_head, seq_len, dim_head}});
    auto *kv_transpose_out_data = dev_ctx.Alloc<T>(
        &kv_transpose_out, kv_transpose_out.numel() * sizeof(T));

    qk_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
    auto *qk_out_data = dev_ctx.Alloc<T>(&qk_out, qk_out.numel() * sizeof(T));

    phi::DenseTensor src_mask_out;
    if (cache_offset > 0) {
      src_mask_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
      auto *src_mask_out_data =
          dev_ctx.Alloc<T>(&src_mask_out, src_mask_out.numel() * sizeof(T));
    }

    // [2, bs, num_head, cache_seq_len + seq_len, head_dim]
    phi::DenseTensor pre_cache_kv_out;
    if (cache_offset > 0) {
      pre_cache_kv_out.Resize(
          {{2, bsz, num_head, seq_len + cache_offset, dim_head}});
      auto *pre_cache_kv_out_data = dev_ctx.Alloc<T>(
          &pre_cache_kv_out, pre_cache_kv_out.numel() * sizeof(T));
    }

    phi::DenseTensor softmax_out;
    phi::DenseTensor attn_dropout_mask_out, attn_dropout_out;
    phi::DenseTensor qktv_out, fmha_out;
    softmax_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
    auto *softmax_out_data =
        dev_ctx.Alloc<T>(&softmax_out, softmax_out.numel() * sizeof(T));

    attn_dropout_mask_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
    auto *attn_dropout_mask_out_data = dev_ctx.Alloc<T>(
        &attn_dropout_mask_out, attn_dropout_mask_out.numel() * sizeof(T));
    attn_dropout_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
    auto *attn_dropout_data_data = dev_ctx.Alloc<T>(
        &attn_dropout_out, attn_dropout_out.numel() * sizeof(T));

    qktv_out.Resize({{bsz, num_head, seq_len, dim_head}});
    auto *qktv_out_data =
        dev_ctx.Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
    fmha_out.Resize({{bsz, seq_len, num_head, dim_head}});
    auto *fmha_out_data =
        dev_ctx.Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));

    // 4. out_linear
    auto out_linear_weights = ctx.MultiInput<phi::DenseTensor>("OutLinearW");
    auto out_linear_biases = ctx.MultiInput<phi::DenseTensor>("OutLinearBias");
    int ring_id = ctx.Attr<int>("ring_id");
    // (transA, transB, compute_bias) = (false, false, false)
    AttnMatmulINT8<T> out_linear_compute(dev_ctx, 
        bsz_seq, dim_embed, hidden_size, false);

    // 5. ln(residual + bias)
    DropoutParam dropout_param2(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        dev_ctx, bsz_seq, dim_embed, dropout_param2, epsilon);
    auto ffn_ln_scales = ctx.MultiInput<phi::DenseTensor>("FFNLnScale");
    auto ffn_ln_biases = ctx.MultiInput<phi::DenseTensor>("FFNLnBias");
    phi::DenseTensor bias_dropout_residual_out, dropout_mask_out;
    T *bias_dropout_residual_out_data = nullptr;
    if (pre_layer_norm) {
      bias_dropout_residual_out.Resize({{bsz, seq_len, dim_embed}});
      bias_dropout_residual_out_data =
          dev_ctx.Alloc<T>(&bias_dropout_residual_out,
                           bias_dropout_residual_out.numel() * sizeof(T));
    }
    dropout_mask_out.Resize({{bsz, seq_len, dim_embed}});
    auto *dropout_mask_out_data = dev_ctx.Alloc<uint8_t>(
        &dropout_mask_out, dropout_mask_out.numel() * sizeof(uint8_t));

    // 6. ffn matmul1
    auto ffn1_weights = ctx.MultiInput<phi::DenseTensor>("FFN1Weight");
    auto ffn1_biases = ctx.MultiInput<phi::DenseTensor>("FFN1Bias");
    auto ffn1_weight_dim = ffn1_weights[0]->dims();

    int dim_ffn = ffn1_weight_dim[0];
    AttnMatmulINT8<T> ffn1_linear_compute(
        dev_ctx, bsz_seq, dim_ffn, dim_embed, false);
    phi::DenseTensor ffn1_out;
    ffn1_out.Resize({{bsz_seq, dim_ffn}});
    auto *ffn1_out_data =
        dev_ctx.Alloc<T>(&ffn1_out, ffn1_out.numel() * sizeof(T));

    // 7. ffn act + bias
    DropoutParam ffn1_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
        dev_ctx, bsz_seq, dim_ffn, ffn1_dropout_param);
    phi::DenseTensor ffn1_dropout_out, ffn1_dropout_mask;
    ffn1_dropout_out.Resize({{bsz_seq, dim_ffn}});
    auto *ffn1_dropout_out_data = dev_ctx.Alloc<T>(
        &ffn1_dropout_out, ffn1_dropout_out.numel() * sizeof(T));
    ffn1_dropout_mask.Resize({{bsz_seq, dim_ffn}});
    auto *ffn1_dropout_mask_data = dev_ctx.Alloc<uint8_t>(
        &ffn1_dropout_mask, ffn1_dropout_mask.numel() * sizeof(uint8_t));

    // 8. ffn2 matmul
    auto ffn2_weights = ctx.MultiInput<phi::DenseTensor>("FFN2Weight");
    auto ffn2_biases = ctx.MultiInput<phi::DenseTensor>("FFN2Bias");
    AttnMatmulINT8<T> ffn2_linear_compute(
        dev_ctx, bsz_seq, dim_embed, dim_ffn, false);

    // 9. ffn2 residual bias
    DropoutParam ffn2_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> ffn2_fused_dropout_helper(
        dev_ctx, bsz_seq, dim_embed, ffn2_dropout_param, epsilon);

    // []. init workspace for cublasLt transform
    phi::DenseTensor input_workspace, output_workspace;
    // for input and output transform data is CUBLASLT_ORDER_COL32 format,
    int m_max = bsz_seq, k_max = std::max(dim_embed, dim_ffn),
        n_max = std::max({output_size, dim_embed, dim_ffn});

    input_workspace.Resize({{(m_max * k_max + 31) / 32 * 32}});
    dev_ctx.Alloc<int8_t>(&input_workspace,
                          input_workspace.numel() * sizeof(int8_t));

    output_workspace.Resize({{(n_max * m_max + 31) / 32 * 32}});
    dev_ctx.Alloc<int32_t>(&output_workspace,
                           output_workspace.numel() * sizeof(int32_t));

    // calc
    auto *out = ctx.Output<phi::DenseTensor>("Out");
    auto *from_data = dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));
    phi::DenseTensor *from_tensor = out;
    phi::DenseTensor tmp_out;
    tmp_out.Resize({{bsz, seq_len, dim_embed}});
    auto *tmp_out_data =
        dev_ctx.Alloc<T>(&tmp_out, tmp_out.numel() * sizeof(T));

    auto *x_data = input_x->data<T>();
    phi::DenseTensor *buf0 = nullptr;
    phi::DenseTensor *buf1 = nullptr;

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

    phi::DenseTensor cublaslt_workspace;
    cublaslt_workspace.Resize({{3000000}});
    dev_ctx.Alloc<int8_t>(&cublaslt_workspace);

    for (int i = 0; i < layers; ++i) {
      // step1. layer_norm
      if (i == 0 && pre_layer_norm) {
        auto *ln_scale_data = ln_scales[i]->data<U>();
        auto *ln_bias_data = ln_biases[i]->data<U>();  
        PrintMatrix(ln_scale_data, ln_scales[i]->numel(), "ln_scale_" + std::to_string(i) + "_step_" + std::to_string(time_step_value) + "_device_" + std::to_string(dev_ctx.GetPlace().GetDeviceId())); 
        PrintMatrix(ln_bias_data, ln_biases[i]->numel(), "ln_bias_" + std::to_string(i) + "_step_" + std::to_string(time_step_value) + "_device_" + std::to_string(dev_ctx.GetPlace().GetDeviceId())); 
        PrintMatrix(x_data, input_x->numel(), "input_step_" + std::to_string(time_step_value) + "_device_" + std::to_string(dev_ctx.GetPlace().GetDeviceId())); 
        
        // TODO(wangxi): can remove mean var in inference
        // PrintMatrix(x_data, buf0->numel(), "X_DATA");
        // PrintMatrix(ln_scales[i]->data<U>(), ln_scales[i]->numel(), "ln_scales");
        // PrintMatrix(ln_biases[i]->data<U>(), ln_biases[i]->numel(), "ln_biases");
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
      const phi::DenseTensor *qkv_bias =
          qkv_biases.size() > 0 ? qkv_biases[i] : nullptr;
      // NOTE: in decoder stage, bias is fused in fmha
      const phi::DenseTensor *bias = time_step ? nullptr : qkv_bias;
      if (!pre_layer_norm && i == 0) {
        // qkv_compute.ComputeForwardDyquant(
        //     qkv_weights[i],
        //     input_x,
        //     &input_workspace,
        //     bias,
        //     &qkv_out,
        //     &output_workspace,
        //     &qkv_out,
        //     qkv_out_scales[i],
        //     "qkv_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
        //     quant_round_type,
        //     quant_max_bound,
        //     quant_min_bound);
        LLMGemm<T>(dev_ctx, 
             qkv_weights[i],
             input_x,
             qkv_weight_ranges[i], 
             &qkv_out,
             "qkv_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
             bsz_seq, input_size, output_size, 
             &cublaslt_workspace,
             quant_round_type,
             quant_max_bound,
             quant_min_bound);
      } else {
        // qkv_compute.ComputeForwardDyquant(
        //    qkv_weights[i],
        //     buf1,
        //     &input_workspace,
        //     bias,
        //     &qkv_out,
        //     &output_workspace,
        //     &qkv_out,
        //     qkv_out_scales[i],
        //     "qkv_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
        //     quant_round_type,
        //     quant_max_bound,
        //     quant_min_bound);
          LLMGemm<T>(dev_ctx, 
             qkv_weights[i],
             buf1,
             qkv_weight_ranges[i], 
             &qkv_out,
             "qkv_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
             bsz_seq, input_size, output_size, 
             &cublaslt_workspace,
             quant_round_type,
             quant_max_bound,
             quant_min_bound);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step2";
#endif

      // step3. fmha
      const phi::DenseTensor *cache_kv =
          cache_kvs.size() > 0 ? cache_kvs[i] : nullptr;
      phi::DenseTensor *cache_kv_out = cache_kv ? cache_kv_outs[i] : nullptr;

      if (time_step) {  // generation decoder stage
        // [2, batch_size, num_head, max_seq_len, head_size]
        int max_seq_len = cache_kv->dims()[3];
        fmha<T>(dev_ctx,
                qkv_out,
                *qkv_bias,
                *src_mask,
                rotary_tensor,
                cache_kv_out,
                &fmha_out,
                bsz,
                max_seq_len,
                num_head,
                dim_head,
                time_step->data<int>()[0],
                rotary_emb_dims,
                1. / sqrt(dim_head));
      } else if (cache_kv_out) {  // generation context stage
        const phi::DenseTensor *pre_cache_kv_tensor =
            pre_caches.size() > 0 ? pre_caches[i] : nullptr;
        phi::DenseTensor *pre_cache_kv_out_tmp =
            cache_offset > 0 ? &pre_cache_kv_out : nullptr;
        phi::DenseTensor *src_mask_tmp =
            cache_offset > 0 ? &src_mask_out : nullptr;
        qkv_bias_add_transpose_split<T>(dev_ctx,
                                        q_transpose_out_data,
                                        kv_transpose_out_data,
                                        qkv_out_data,
                                        qkv_bias->data<T>(),
                                        bsz,
                                        num_head,
                                        seq_len,
                                        dim_head,
                                        compute_bias);

        // q_transpose_out_data [bs, head_num, seq_len, dim_head]
        // kv_transpose_out_data [2， bs, head_num, seq_len, dim_head]
        if (rotary_emb_dims != 0) {
          auto *rotary_emb_data = rotary_tensor->data<T>();
          rotary_qk(dev_ctx,
                    q_transpose_out_data,
                    kv_transpose_out_data,
                    q_transpose_out_data,
                    kv_transpose_out_data,
                    rotary_emb_data,
                    rotary_emb_dims,
                    bsz,
                    num_head,
                    seq_len,
                    dim_head);
        }

        fmha_compute.ComputeForwardWithoutTranspose(pre_cache_kv_tensor,
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
        write_cache_kv<T>(dev_ctx,
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
        qkv_bias_add_transpose_split<T>(dev_ctx,
                                        q_transpose_out_data,
                                        kv_transpose_out_data,
                                        qkv_out_data,
                                        qkv_bias->data<T>(),
                                        bsz,
                                        num_head,
                                        seq_len,
                                        dim_head,
                                        compute_bias);

        // q_transpose_out_data [bs, head_num, seq_len, dim_head]
        // kv_transpose_out_data [2， bs, head_num, seq_len, dim_head]
        if (rotary_emb_dims != 0) {
          auto *rotary_emb_data = rotary_tensor->data<T>();
          rotary_qk(dev_ctx,
                    q_transpose_out_data,
                    kv_transpose_out_data,
                    q_transpose_out_data,
                    kv_transpose_out_data,
                    rotary_emb_data,
                    rotary_emb_dims,
                    bsz,
                    num_head,
                    seq_len,
                    dim_head);
        }

        fmha_compute.ComputeForwardWithoutTranspose(cache_kv,
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
        // out_linear_compute.ComputeForwardDyquant(
        //     out_linear_weights[i],
        //     &fmha_out,
        //     &input_workspace,
        //     nullptr,
        //     buf1,
        //     &output_workspace,
        //     nullptr,
        //     out_linear_out_scales[i],
        //     "out_linear_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
        //     quant_round_type,
        //     quant_max_bound,
        //     quant_min_bound);
        LLMGemm<T>(dev_ctx, 
             out_linear_weights[i],
             &fmha_out,
             out_linear_out_weight_ranges[i], 
             buf1,
             "out_linear_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
             bsz_seq, hidden_size, dim_embed, 
             &cublaslt_workspace,
             quant_round_type,
             quant_max_bound,
             quant_min_bound);
        AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
      } else {
        // out_linear_compute.ComputeForwardDyquant(
        //     out_linear_weights[i],
        //     &fmha_out,
        //     &input_workspace,
        //     nullptr,
        //     buf0,
        //     &output_workspace,
        //     nullptr,
        //     out_linear_out_scales[i],
        //     "out_linear_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
        //     quant_round_type,
        //     quant_max_bound,
        //     quant_min_bound);
        LLMGemm<T>(dev_ctx, 
             out_linear_weights[i],
             &fmha_out,
             out_linear_out_weight_ranges[i], 
             buf0,
             "out_linear_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
             bsz_seq, hidden_size, dim_embed, 
             &cublaslt_workspace,
             quant_round_type,
             quant_max_bound,
             quant_min_bound);
        AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step4";
#endif

      // step5. ln(residual + dropout(input + bias))
      if (pre_layer_norm) {
        auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
        auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
        auto *out_linear_bias_data = out_linear_biases[i]->data<T>();

        // inplace
        fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
            dev_ctx,
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
        auto *ln_scale_data = ln_scales[i]->data<U>();
        auto *ln_bias_data = ln_biases[i]->data<U>();
        auto *out_linear_bias_data = out_linear_biases[i]->data<T>();
        auto *residual_data = (i == 0 ? x_data : buf1->data<T>());
        fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
            dev_ctx,
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
      // ffn1_linear_compute.ComputeForwardDyquant(
      //     ffn1_weights[i],
      //       buf1,
      //       &input_workspace,
      //       nullptr,
      //       &ffn1_out,
      //       &output_workspace,
      //       nullptr,
      //       ffn1_out_scales[i],
      //       "ffn1_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
      //       quant_round_type,
      //       quant_max_bound,
      //       quant_min_bound);
      LLMGemm<T>(dev_ctx, 
             ffn1_weights[i],
             buf1,
             ffn1_weight_ranges[i], 
             &ffn1_out,
             "ffn1_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
             bsz_seq, dim_embed, dim_ffn, 
             &cublaslt_workspace,
             quant_round_type,
             quant_max_bound,
             quant_min_bound);
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step6";
#endif

      // step7. act bias
      // TODO(wangxi): remove dropout mask in inference
      fused_act_dropout_helper.DropoutActBias(dev_ctx,
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
        ffn2_linear_compute.ComputeForwardDyquant(
            ffn2_weights[i],
            &ffn1_dropout_out,
            &input_workspace,
            nullptr,
            buf1,
            &output_workspace,
            nullptr,
            ffn2_out_scales[i],
            "ffn2_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
            quant_round_type,
            quant_max_bound,
            quant_min_bound);
      } else {
        ffn2_linear_compute.ComputeForwardDyquant(
            ffn2_weights[i],
            &ffn1_dropout_out,
            &input_workspace,
            nullptr,
            buf0,
            &output_workspace,
            nullptr,
            ffn2_out_scales[i],
            "ffn2_"+ std::to_string(i) + "_step_" + std::to_string(time_step_value),
            quant_round_type,
            quant_max_bound,
            quant_min_bound);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step8.0";
#endif

      if (pre_layer_norm) {
        AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
      } else {
        AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
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
          PrintMatrix(ln_scale_data, ln_scales[i+1]->numel(), "ln_scale_" + std::to_string(i) + "_step_" + std::to_string(time_step_value) + "_device_" + std::to_string(dev_ctx.GetPlace().GetDeviceId())); 
          PrintMatrix(ln_bias_data, ln_biases[i+1]->numel(), "ln_bias_" + std::to_string(i) + "_step_" + std::to_string(time_step_value) + "_device_" + std::to_string(dev_ctx.GetPlace().GetDeviceId())); 
          ffn2_fused_dropout_helper.LayernormResidualDropoutBias(
              dev_ctx,
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
              dev_ctx,
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
            dev_ctx,
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
      // PADDLE_THROW(platform::errors::Unimplemented("STOP"));
    }
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_multi_transformer_dyquant,
                        ops::FusedMultiTransformerDyquantOpKernel<plat::float16>,
                        ops::FusedMultiTransformerDyquantOpKernel<float>);
