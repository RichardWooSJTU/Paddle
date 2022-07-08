// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/multi_transformer_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {


// namespace patterns {

// PDNode* MultiTransformer::operator()() {
//     /*
//      * Create old subgraph's *PDNode* including op PDNode and tensor PDNode
//      */
    
//     auto *op = pattern->NewNode(fused_multi_transformer_int8_repr())->assert_is_op("fused_multi_transformer_int8");
//     auto *ln_scale_var = pattern->NewNode(ln_scale_repr())
//                         ->AsInput()
//                         ->assert_is_persistable_var()
//                         ->assert_is_op_input("fused_multi_transformer_int8", "LnScale");
//     auto *ln_bias_var = pattern->NewNode(ln_bias_repr())
//                         ->AsInput()
//                         ->assert_is_persistable_var()
//                         ->assert_is_op_input("fused_multi_transformer_int8", "LnBias");
//     auto *qkv_w_var = pattern->NewNode(qkv_w_repr())
//                         ->AsInput()
//                         ->assert_is_persistable_var()
//                         ->assert_is_op_input("fused_multi_transformer_int8", "QKVW");
//     auto *qkv_bias_var = pattern->NewNode(qkv_bias_repr())
//                 ->AsInput()
//                 ->assert_is_persistable_var()
//                 ->assert_is_op_input("fused_multi_transformer_int8", "QKVBias");
//     auto *cache_kv_var = pattern->NewNode(cache_kv_repr())
//                 ->AsInput()
//                 ->assert_is_persistable_var()
//                 ->assert_is_op_input("fused_multi_transformer_int8", "CacheKV");
//     auto *time_stamp_var = pattern->NewNode(time_stamp_repr())
//                 ->AsInput()
//                 ->assert_is_persistable_var()
//                 ->assert_is_op_input("fused_multi_transformer_int8", "TimeStep");
//     auto *src_mask_var = pattern->NewNode(src_mask_repr())
//                 ->AsInput()
//                 ->assert_is_persistable_var()
//                 ->assert_is_op_input("fused_multi_transformer_int8", "SrcMask");
//     auto *out_linear_w_var = pattern->NewNode(out_linear_w_repr())
//                 ->AsInput()
//                 ->assert_is_persistable_var()
//                 ->assert_is_op_input("fused_multi_transformer_int8", "OutLinearW");
//     auto *out_linear_bias_var = pattern->NewNode(out_linear_bias_repr())
//                         ->AsInput()
//                         ->assert_is_persistable_var()
//                         ->assert_is_op_input("fused_multi_transformer_int8", "OutLinearBias");
//     auto *ffn_ln_scale_var = pattern->NewNode(ffn_ln_scale_repr())
//                 ->AsInput()
//                 ->assert_is_persistable_var()
//                 ->assert_is_op_input("fused_multi_transformer_int8", "FFNLnScale");
//     auto *ffn_ln_bias_var = pattern->NewNode(ffn_ln_bias_repr())
//                 ->AsInput()
//                 ->assert_is_persistable_var()
//                 ->assert_is_op_input("fused_multi_transformer_int8", "FFNLnBias");
//     auto *ffn1_weight_var = pattern->NewNode(ffn1_weight_repr())
//                 ->AsInput()
//                 ->assert_is_persistable_var()
//                 ->assert_is_op_input("fused_multi_transformer_int8", "FFN1Weight");
//     auto *ffn1_bias_var = pattern->NewNode(ffn1_bias_repr())
//                         ->AsInput()
//                         ->assert_is_persistable_var()
//                         ->assert_is_op_input("fused_multi_transformer_int8", "FFN1Bias");
//     auto *ffn2_weight_var = pattern->NewNode(ffn2_weight_repr())
//                 ->AsInput()
//                 ->assert_is_persistable_var()
//                 ->assert_is_op_input("fused_multi_transformer_int8", "FFN2Weight");
//     auto *ffn2_bias_var = pattern->NewNode(ffn2_bias_repr())
//                         ->AsInput()
//                         ->assert_is_persistable_var()
//                         ->assert_is_op_input("fused_multi_transformer_int8", "FFN2Bias");

//     auto *cache_kv_out_var = pattern->NewNode(cache_kv_out_repr())
//                         ->AsOutput()
//                         ->assert_is_op_output("fused_multi_transformer_int8");
//     auto *out_var = pattern->NewNode(out_repr())
//                         ->AsOutput()
//                         ->assert_is_op_output("fused_multi_transformer_int8");
//     op->LinksFrom({ln_scale_var,ln_bias_var,qkv_w_var,qkv_bias_var,cache_kv_var,
//                     time_stamp_var,src_mask_var,out_linear_w_var,out_linear_bias_var,
//                     ffn_ln_scale_var,ffn_ln_bias_var,ffn1_weight_var,ffn1_bias_var,ffn2_weight_var,ffn2_bias_var,
//                     cache_kv_out_var,out_var}).LinksTo({cache_kv_out_var, out_var});
//     return out_var;

// }



// } // namespace patterns

// Constructor: Some of passes invoke AddOpCompat to state which ops will be included (compatible). Here we do not need it
MultiTransformerFusePass::MultiTransformerFusePass(){}

// ApplyImpl: delete some ops of graph and add some new
void MultiTransformerFusePass::ApplyImpl(Graph* g) const {
    /*
     * 1. construct GraphPatternDetector, which is used to manage a global patten in this pass
     * 2. construct input *PDNode*, this node should be stated as input and add new assert to a assert set
     * 3. specific pattern should be constructed 
     * 4. invoke this pattern's operator() to ...
     * 5. define a lambda
     *    a. use GET_IR_NODE_FROM_SUBGRAPH to get all vars
     *    b. create new op node by create new opDesc and invoke graph function CreateOpNode
     *    c. relink nodes: 
     *       i. make new_op's input as origin first op's input, first op's inputs' output(opNode) as new op
     *       ii. first op's out: if it is second op's input, delete it, else make it as new op's output
     */
    PADDLE_ENFORCE_NOT_NULL(
    g, platform::errors::InvalidArgument("Graph cannot be nullptr."));
    FusePassBase::Init("multi_transformer_fuse" /*repr_*/, g);

    auto* scope = param_scope();
     PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "During the multi_transformer_fuse pass, The scope should not be null."));

    for (auto *node : g->Nodes()) {
        if (node->IsOp() && node->Op()-> Type() == "fused_multi_transformer_int8") {
            auto *op = node->Op();
            VLOG(1) << "[DEBUG] fused_multi_transformer_int8 weighs name";
            const std::vector<std::string> &qkv_w = op->Input("QKVW");
            for (auto n : qkv_w) std::cout << n << " ";
            std::cout << std::endl;
            const std::vector<std::string> &out_linear_w = op->Input("OutLinearW");
            for (auto n : out_linear_w) std::cout << n << " ";
            std::cout << std::endl;
            const std::vector<std::string> &ffn1_weight = op->Input("FFN1Weight");
            for (auto n : ffn1_weight) std::cout << n << " ";
            std::cout << std::endl;
            const std::vector<std::string> &ffn2_weight = op->Input("FFN2Weight");
            for (auto n : ffn2_weight) std::cout << n << " ";
            std::cout << std::endl;

            //Find var
            for (int i = 0; i < qkv_w.size(); i++) {
                auto name = qkv_w[i];
                
                auto* weight_tensor = scope->FindVar(name)->GetMutable<LoDTensor>();
                auto qkv_w_dims = weight_tensor->dims();
                if (i == 0) {
                    if (qkv_w_dims.size() != 1) {
                        g->Set("num_head_" + name, new int(static_cast<int>(qkv_w_dims[1])));
                        g->Set("dim_head_" + name, new int(static_cast<int>(qkv_w_dims[2])));
                    }
                    int num_head = g->Get<int>("num_head_" + name);
                    int dim_head = g->Get<int>("dim_head_" + name);
                    VLOG(1) << num_head << " " << dim_head;
                    op->SetAttr("num_head", num_head);
                    op->SetAttr("dim_head", dim_head);
                    if (qkv_w_dims.size() == 1) break;
                }
                // for cublasLt
                int k = qkv_w_dims[3], n = qkv_w_dims[0] * qkv_w_dims[1] * qkv_w_dims[2];
                PrepareWeights(weight_tensor, k, n);
            } 
            for (int i = 0; i < out_linear_w.size(); i++) {
                auto name = out_linear_w[i];
                
                auto* weight_tensor = scope->FindVar(name)->GetMutable<LoDTensor>();
                auto dim = weight_tensor->dims();
                if (dim.size() == 1) break;
                int k = dim[0], n = dim[1];
                PrepareWeights(weight_tensor, k, n);
            } 
            for (int i = 0; i < ffn1_weight.size(); i++) {
                auto name = ffn1_weight[i];
                
                auto* weight_tensor = scope->FindVar(name)->GetMutable<LoDTensor>();
                auto dim = weight_tensor->dims();
                if (i == 0) {
                    if (dim.size() != 1) {
                        g->Set("dim_ffn_" + name, new int(static_cast<int>(dim[1])));
                    }
                    int dim_ffn = g->Get<int>("dim_ffn_" + name);
                    VLOG(1) << "dim_ffn " << dim_ffn;
                    op->SetAttr("dim_ffn", dim_ffn);
                    if (dim.size() == 1) break;
                }
                int k = dim[0], n = dim[1];
                PrepareWeights(weight_tensor, k, n);
            } 
            for (int i = 0; i < ffn2_weight.size(); i++) {
                auto name = ffn2_weight[i];
                
                auto* weight_tensor = scope->FindVar(name)->GetMutable<LoDTensor>();
                auto dim = weight_tensor->dims();
                if (dim.size() == 1) break;
                int k = dim[0], n = dim[1];
                PrepareWeights(weight_tensor, k, n);
            } 
        }
    }

}

void MultiTransformerFusePass::PrepareWeights(framework::Tensor* weight_tensor, int k, int n) const {
    PADDLE_ENFORCE_NOT_NULL(
        weight_tensor,
        platform::errors::InvalidArgument("weight tensor should not be nullptr"));
    // quantize transpose transform
    // qkv_w do not need transpose
    framework::Tensor weight_tensor_tmp;
    auto place  =  weight_tensor->place();
    framework::TensorCopy(*weight_tensor, place, &weight_tensor_tmp);

    int ldbtransform = 32 * ((n + 8 - 1) / 8) * 8;
    weight_tensor->Resize({(k + 32 - 1) / 32 * ldbtransform});
    weight_tensor->mutable_data<int8_t>(place);
}


 // The key to change old graph to new graph is GraphPatternDetector::operator()
 


}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_transformer_fuse_pass,
              paddle::framework::ir::MultiTransformerFusePass);