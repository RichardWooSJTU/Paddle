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

#include "paddle/fluid/framework/ir/multi_transformer_quant_transform_pass.h"
#include "paddle/fluid/operators/fused/cublasLt_helper.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace framework {
namespace ir {



// Constructor: Some of passes invoke AddOpCompat to state which ops will be included (compatible). Here we do not need it
MultiTransformerQuantTransformPass::MultiTransformerQuantTransformPass(){}

// ApplyImpl: delete some ops of graph and add some new
void MultiTransformerQuantTransformPass::ApplyImpl(Graph* g) const {
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
            bool trans_qkvw = true;
            if (op->HasAttr("trans_qkvw")) {
                trans_qkvw = PADDLE_GET_CONST(bool, op->GetAttr("trans_qkvw"));
                PADDLE_ENFORCE_EQ(trans_qkvw, true, 
                    platform::errors::InvalidArgument("fused_multi_transformer_int8 do not support not trans_qkvw"));
            }

            int var_type = -1;
            //Get Origin datatype / Set var datatype
            std::unordered_set<std::string> weights_name_set(qkv_w.begin(), qkv_w.end());
            weights_name_set.insert(out_linear_w.begin(), out_linear_w.end());
            weights_name_set.insert(ffn1_weight.begin(), ffn1_weight.end());
            weights_name_set.insert(ffn2_weight.begin(), ffn2_weight.end());
            for (auto *var_node : node->inputs) {
                auto var_name = var_node->Var()->Name();
                if (weights_name_set.count(var_name) != 0) {
                    if (var_type==-1) {
                        var_type = var_node->Var()->GetDataType();
                    }
                    VLOG(1) << "set " << var_name << " var_node datatype as int8";
                    var_node->Var()->SetDataType(paddle::framework::proto::VarType::INT8);
                }
            }

            //Find var
            std::vector<float> in_scale(qkv_w.size(), 1.0f);
            // debugggggg
            op->SetAttr("qkv_in_scale", in_scale);
            op->SetAttr("out_linear_in_scale", in_scale);
            op->SetAttr("ffn1_in_scale", in_scale);
            op->SetAttr("ffn2_in_scale", in_scale);
            // debugggggg

            if (op->HasAttr("qkv_in_scale")) {
                in_scale = PADDLE_GET_CONST(std::vector<float>, op->GetAttr("qkv_in_scale"));
            } 

            LoDTensor qkv_out_scale, out_linear_out_scale, ffn1_out_scale, ffn2_out_scale;
            std::string scale_var_name;

            for (int i = 0; i < qkv_w.size(); i++) {
                auto name = qkv_w[i];
                
                // auto* weight_tensor = scope->FindVar(name)->GetMutable<LoDTensor>();
                auto* weight_var = scope->FindVar(name);
                PADDLE_ENFORCE_NOT_NULL(
                    weight_var,
                    platform::errors::NotFound(
                        "The Weight variable [%s] of fused_multi_transformer_int8 is not found.",
                        name));
                // Set var_data_type
                auto* weight_tensor = weight_var->GetMutable<LoDTensor>();
                auto qkv_w_dims = weight_tensor->dims();

                
                if (i == 0) {
                    scale_var_name = "out_scale_" + name;
                    // Main graph has transformed weight data, sub graph cannot gain origin info
                    // Set info as graph attr
                    if (qkv_w_dims.size() != 1) {
                        int num_head = qkv_w_dims[1];
                        int dim_head = qkv_w_dims[2];
                        g->Set("num_head_" + name, new int(static_cast<int>(num_head)));
                        g->Set("dim_head_" + name, new int(static_cast<int>(dim_head)));
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
                
                if (i == 0) {
                    qkv_out_scale.mutable_data<float>({qkv_w.size(), n}, platform::CPUPlace());
                    VLOG(1) << "qkv_out_scale.mutable_data dims is " << qkv_out_scale.dims();
                }

                VLOG(1) << "prepare weight " << name;
                std::vector<float> weight_scale(n, 1.0f);
                PrepareWeights(weight_tensor, k, n, weight_scale, var_type, true);
                VLOG(1) << "weight_scale[0] = " << weight_scale[0];
                for (int j = 0; j < n; ++j) {
                    qkv_out_scale.data<float>()[i * n + j] = weight_scale[j] * in_scale[i];
                }
                VLOG(1) << "qkv_out_scale.data<float>()[i * n] = " << qkv_out_scale.data<float>()[i * n];
                if (i == qkv_w.size()-1) {
                    auto* var = scope->Var(scale_var_name);
                    auto* tensor = var->GetMutable<LoDTensor>();
                    *tensor = qkv_out_scale;

                    auto* var_desc = new framework::VarDesc(scale_var_name);
                    var_desc->SetShape({qkv_w.size(), n});
                    var_desc->SetDataType(framework::proto::VarType::FP32);
                    var_desc->SetPersistable(true);
                    auto* var_node = g->CreateVarNode(var_desc);
                    node->inputs.push_back(var_node);
                }
            } 

            op->SetInput("QKVOutScale", {scale_var_name});

            if (op->HasAttr("out_linear_in_scale")) {
                in_scale = PADDLE_GET_CONST(std::vector<float>, op->GetAttr("out_linear_in_scale"));
            } 


            for (int i = 0; i < out_linear_w.size(); i++) {
                auto name = out_linear_w[i];
                
                auto* weight_var = scope->FindVar(name);
                PADDLE_ENFORCE_NOT_NULL(
                    weight_var,
                    platform::errors::NotFound(
                        "The Weight variable [%s] of fused_multi_transformer_int8 is not found.",
                        name));
                auto* weight_tensor = weight_var->GetMutable<LoDTensor>();
                auto dim = weight_tensor->dims();
                if (i == 0) {
                    scale_var_name = "out_scale_" + name;
                }
                if (dim.size() == 1) break;
                int k = dim[0], n = dim[1];

                if (i == 0) {
                    out_linear_out_scale.mutable_data<float>({out_linear_w.size(), n}, platform::CPUPlace());
                }

                std::vector<float> weight_scale(n, 1.0f);
                PrepareWeights(weight_tensor, k, n, weight_scale, var_type, false);
                for (int j = 0; j < n; ++j) {
                    out_linear_out_scale.data<float>()[i * n + j] = weight_scale[j] * in_scale[i];
                }
                if (i == out_linear_w.size()-1) {
                    auto* var = scope->Var(scale_var_name);
                    auto* tensor = var->GetMutable<LoDTensor>();
                    *tensor = out_linear_out_scale;

                    auto* var_desc = new framework::VarDesc(scale_var_name);
                    var_desc->SetShape({qkv_w.size(), n});
                    var_desc->SetDataType(framework::proto::VarType::FP32);
                    var_desc->SetPersistable(true);
                    auto* var_node = g->CreateVarNode(var_desc);
                    node->inputs.push_back(var_node);
                }
            } 
            op->SetInput("OutLinearOutScale", {scale_var_name});

            if (op->HasAttr("ffn1_in_scale")) {
                in_scale = PADDLE_GET_CONST(std::vector<float>, op->GetAttr("ffn1_in_scale"));
            } 

            for (int i = 0; i < ffn1_weight.size(); i++) {
                auto name = ffn1_weight[i];
                
                auto* weight_var = scope->FindVar(name);
                PADDLE_ENFORCE_NOT_NULL(
                    weight_var,
                    platform::errors::NotFound(
                        "The Weight variable [%s] of fused_multi_transformer_int8 is not found.",
                        name));
                auto* weight_tensor = weight_var->GetMutable<LoDTensor>();
                auto dim = weight_tensor->dims();
                if (i == 0) {
                    scale_var_name = "out_scale_" + name;
                    if (dim.size() != 1) {
                        g->Set("dim_ffn_" + name, new int(static_cast<int>(dim[1])));
                    }
                    int dim_ffn = g->Get<int>("dim_ffn_" + name);
                    VLOG(1) << "dim_ffn " << dim_ffn;
                    op->SetAttr("dim_ffn", dim_ffn);
                    if (dim.size() == 1) break;
                }
                int k = dim[0], n = dim[1];
                if (i == 0) {
                    ffn1_out_scale.mutable_data<float>({ffn1_weight.size(), n}, platform::CPUPlace());
                }
                std::vector<float> weight_scale(n, 1.0f);
                PrepareWeights(weight_tensor, k, n, weight_scale, var_type, false);
                for (int j = 0; j < n; ++j) {
                    ffn1_out_scale.data<float>()[i * n + j] = weight_scale[j] * in_scale[i];
                }
                if (i == ffn1_weight.size()-1) {
                    auto* var = scope->Var(scale_var_name);
                    auto* tensor = var->GetMutable<LoDTensor>();
                    *tensor = ffn1_out_scale;

                    auto* var_desc = new framework::VarDesc(scale_var_name);
                    var_desc->SetShape({qkv_w.size(), n});
                    var_desc->SetDataType(framework::proto::VarType::FP32);
                    var_desc->SetPersistable(true);
                    auto* var_node = g->CreateVarNode(var_desc);
                    node->inputs.push_back(var_node);
                }
            } 
            op->SetInput("FFN1OutScale", {scale_var_name});

            if (op->HasAttr("ffn2_in_scale")) {
                in_scale = PADDLE_GET_CONST(std::vector<float>, op->GetAttr("ffn2_in_scale"));
            } 
            for (int i = 0; i < ffn2_weight.size(); i++) {
                auto name = ffn2_weight[i];
                
                auto* weight_var = scope->FindVar(name);
                PADDLE_ENFORCE_NOT_NULL(
                    weight_var,
                    platform::errors::NotFound(
                        "The Weight variable [%s] of fused_multi_transformer_int8 is not found.",
                        name));
                auto* weight_tensor = weight_var->GetMutable<LoDTensor>();
                auto dim = weight_tensor->dims();
                if (i == 0) {
                    scale_var_name = "out_scale_" + name;
                }
                if (dim.size() == 1) break;
                int k = dim[0], n = dim[1];
                if (i == 0) {
                    ffn2_out_scale.mutable_data<float>({ffn2_weight.size(), n}, platform::CPUPlace());
                }
                std::vector<float> weight_scale(n, 1.0f);
                PrepareWeights(weight_tensor, k, n, weight_scale, var_type, false);
                for (int j = 0; j < n; ++j) {
                    ffn2_out_scale.data<float>()[i * n + j] = weight_scale[j] * in_scale[i];
                }
                 if (i == ffn2_weight.size()-1) {
                    auto* var = scope->Var(scale_var_name);
                    auto* tensor = var->GetMutable<LoDTensor>();
                    *tensor = ffn2_out_scale;auto* var_desc = new framework::VarDesc(scale_var_name);

                    var_desc->SetShape({qkv_w.size(), n});
                    var_desc->SetDataType(framework::proto::VarType::FP32);
                    var_desc->SetPersistable(true);
                    auto* var_node = g->CreateVarNode(var_desc);
                    node->inputs.push_back(var_node);
                }
            }
            op->SetInput("FFN2OutScale", {scale_var_name}); 
            
        }
    }

}

template <typename T>
static inline T abs(T x) {
    return x > 0 ? x : -x;
}

static inline int8_t cast_int8(float a) {
    return static_cast<int8_t>(a+0.5f);
}

template <typename T>
static void LayerwiseQuantize(const T* src, int8_t* dst, std::vector<float>& scale, int k, int n, bool trans) {
    std::vector<T> max_value(n, static_cast<T>(0.0f));
    if (!trans) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n; ++j) {
                T v = abs(src[i * n + j]);
                if ( v > max_value[j]) max_value[j] = v;
            }
        }
        for (int i = 0; i < n; ++i) {
            scale[i] = 127.0f / static_cast<float>(max_value[i]);
        }
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n; ++j) {
                dst[i * n + j] = cast_int8(static_cast<float>(src[i * n + j]) * scale[j]);
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                T v = abs(src[i * k + j]);
                if ( v > max_value[i]) max_value[i] = v;
            }
        }
        for (int i = 0; i < n; ++i) {
            scale[i] = 127.0f / static_cast<float>(max_value[i]);
        }
         for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
               dst[j * n + i] = cast_int8(static_cast<float>(src[i * k + j]) * scale[i]);
            }
        }
    }
}

void MultiTransformerQuantTransformPass::PrepareWeights(framework::Tensor* weight_tensor, int k, int n, 
                                                std::vector<float>& weight_scale, int var_type, bool trans) const {
    PADDLE_ENFORCE_NOT_NULL(
        weight_tensor,
        platform::errors::InvalidArgument("weight tensor should not be nullptr"));
    // quantize /  transpose: just transpose qkv_w
    auto place  =  weight_tensor->place();
    framework::Tensor weight_tensor_tmp;
    weight_tensor_tmp.mutable_data<int8_t>({k*n}, place);

    // if (var_type == paddle::framework::proto::VarType::FP32) {
    //     LayerwiseQuantize(weight_tensor->data<float>(), weight_tensor_tmp.data<int8_t>(), weight_scale, k, n, trans);
    // } else if (var_type == paddle::framework::proto::VarType::FP16) {
    //     LayerwiseQuantize(weight_tensor->data<platform::float16>(), weight_tensor_tmp.data<int8_t>(), weight_scale, k, n, trans);
    // }
    
    // transform: use API
    weight_tensor->Resize({k*n});
    weight_tensor->mutable_data<int8_t>(place);

}


 // The key to change old graph to new graph is GraphPatternDetector::operator()
 


}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_transformer_quant_transform_pass,
              paddle::framework::ir::MultiTransformerQuantTransformPass);