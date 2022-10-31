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

#include "paddle/fluid/framework/ir/fuse_multi_layer_transformer_pass.h"

#include <string>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

std::unordered_map<std::string, std::string> FuseMultiLayerTransformerPattern::operator()(
        bool enable_int8, int step) {
    std::string fused_multi_transformer_name = enable_int8 ? "fused_multi_transformer_int8" : "fused_multi_transformer";
    // This map is used to store node_reprs, 3 * i names will be inserted
    // cache_kv0_{i}, cache_kv1_{i}, fill_constant_batch_size_like_{i}
    std::unordered_map<std::string, std::string> node_reprs;

    VLOG(0) << "step in pattern = " << step;
    // First FusedMultiTransformer
    auto* x0 = pattern->NewNode(x0_repr());
    x0->assert_is_op_input(fused_multi_transformer_name, "X")->AsInput();

    auto* src_mask = pattern->NewNode(src_mask_repr());
    src_mask->assert_is_op_input(fused_multi_transformer_name, "SrcMask")->AsInput();


    auto* fused_multi_transformer0 =
      pattern->NewNode(fused_multi_transformer0_repr())->assert_is_op(fused_multi_transformer_name);
    auto* out0 = pattern->NewNode(out0_repr())
                                  ->AsIntermediate()
                                  ->assert_is_op_output(fused_multi_transformer_name, "Out");

    std::vector<PDNode*> cache_kv0_nodes;
    for (int i = 0; i < step; ++i) {
        auto cache_kv0_repr = PDNodeName(name_scope_, repr_, id_, "cache_kv0_" + std::to_string(i));
        node_reprs["cache_kv0_" + std::to_string(i)] = cache_kv0_repr;
        auto* cache_kv0 = pattern->NewNode(cache_kv0_repr);
        cache_kv0->assert_is_op_nth_input(fused_multi_transformer_name, "CacheKV", i);
        cache_kv0_nodes.push_back(cache_kv0);
    }

    auto fuse_op0_input_nodes = std::move(cache_kv0_nodes);
    fuse_op0_input_nodes.push_back(x0);
    fuse_op0_input_nodes.push_back(src_mask);
    
    fused_multi_transformer0->LinksFrom(fuse_op0_input_nodes).LinksTo({out0});      
            
    std::vector<PDNode*> cache_kv1_nodes;
    for (int i = 0; i < step; ++i) {
        std::string node_name = "fill_constant_batch_size_like_" + std::to_string(i);
        auto fill_const_op_repr = PDNodeName(name_scope_, repr_, id_, node_name);
        node_reprs[node_name] = fill_const_op_repr;
        auto* fill_constant_batch_size_like = 
            pattern->NewNode(fill_const_op_repr)->assert_is_op("fill_constant_batch_size_like");
        
        node_name = "cache_kv1_" + std::to_string(i);
        auto cache_kv1_repr = PDNodeName(name_scope_, repr_, id_, node_name);
        node_reprs[node_name] = cache_kv1_repr;
        auto* cache_kv1 = pattern->NewNode(cache_kv1_repr);
        cache_kv1->assert_is_op_nth_input(fused_multi_transformer_name, "CacheKV", i)
            ->assert_is_op_nth_output("fill_constant_batch_size_like", "Out", 0);
        cache_kv1_nodes.push_back(cache_kv1);
        fill_constant_batch_size_like->LinksFrom({out0}).LinksTo({cache_kv1});
    }
    
   
    auto* fused_multi_transformer1 =
      pattern->NewNode(fused_multi_transformer1_repr())->assert_is_op(fused_multi_transformer_name);
    auto* out1 = pattern->NewNode(out1_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output(fused_multi_transformer_name, "Out");

    auto fuse_op1_input_nodes = std::move(cache_kv1_nodes);
    fuse_op1_input_nodes.push_back(out0);
    fuse_op1_input_nodes.push_back(src_mask);

    fused_multi_transformer1->LinksFrom(fuse_op1_input_nodes).LinksTo({out1});
    // VLOG(0) << "Link fused_multi_transformer1";
    return node_reprs;
    // while loop
    // auto* while0 = pattern->NewNode(while0_repr())->assert_is_op("while");
    // while0->LinksFrom({catchkv_out0, catchkv_out1});
}
} // namespace pattern

inline void MergeInput(OpDesc* op, VariableNameMap& inputs_names0, VariableNameMap& inputs_names1, const std::string& var_name) {
    std::vector<std::string> tmp = inputs_names0[var_name];
    tmp.insert(tmp.end(), inputs_names1[var_name].begin(), inputs_names1[var_name].end());
    op->SetInput(var_name, tmp);
}

template <typename T>
inline void MergeAttrs(OpDesc* op0, const OpDesc* op1, const std::string& attr_name) {
    auto scale_vec_0 = PADDLE_GET_CONST(std::vector<T>, op0->GetAttr(attr_name));
    auto scale_vec_1 = PADDLE_GET_CONST(std::vector<T>, op1->GetAttr(attr_name));
    scale_vec_0.insert(scale_vec_0.end(), scale_vec_1.begin(), scale_vec_1.end());

    op0->SetAttr(attr_name, scale_vec_0);
}

int FuseMultiLayerTransformerPass::BuildFusion(
    Graph* graph, const std::string& name_scope, Scope* scope, int step) const {
    GraphPatternDetector gpd;
    auto* pattern = gpd.mutable_pattern();
    VLOG(0) << "In builf fusion";
    bool enable_int8 = graph->Get<bool>("enable_int8");
    if (enable_int8) {
        VLOG(0) << "encoder with int8";
    } else {
        VLOG(0) << "encoder with fp";
    }

    patterns::FuseMultiLayerTransformerPattern multi_layer_pattern(pattern, name_scope);
    auto node_reprs = multi_layer_pattern(enable_int8, step);
    for (auto p : node_reprs) {
        VLOG(0) << "key: " << p.first << " value: " << p.second;
    }

    VLOG(0) << "Finish build pattern";
    int fusion_count{0};
    auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
        VLOG(0) << "handle FuseMultiLayerTransformerPass";
        VLOG(0) << "subgraph.size()" << subgraph.size();
        
        GET_IR_NODE_FROM_SUBGRAPH(
            src_mask, src_mask, multi_layer_pattern);

        GET_IR_NODE_FROM_SUBGRAPH(
            x0, x0, multi_layer_pattern);
        // GET_IR_NODE_FROM_SUBGRAPH(
        //     cache_kv0, cache_kv0, multi_layer_pattern);
        std::vector<Node*> cache_kv0_nodes;
        std::vector<Node*> cache_kv1_nodes;
        std::vector<Node*> fill_op_nodes;

        VLOG(0) << " Prepare to retrieve cachekv0 node with step = "<< step;

        for (int i = 0; i < step; ++i) {
            PDNode* cache_kv0_pdnode = multi_layer_pattern.PatternBase::pattern->RetrieveNode(node_reprs["cache_kv0_" + std::to_string(i)]);
            cache_kv0_nodes.push_back(subgraph.at(cache_kv0_pdnode));

            PDNode* cache_kv1_pdnode = multi_layer_pattern.PatternBase::pattern->RetrieveNode(node_reprs["cache_kv1_" + std::to_string(i)]);
            cache_kv1_nodes.push_back(subgraph.at(cache_kv1_pdnode));

            PDNode* fill_op_pdnode = multi_layer_pattern.PatternBase::pattern->RetrieveNode(node_reprs["fill_constant_batch_size_like_" + std::to_string(i)]);
            fill_op_nodes.push_back(subgraph.at(fill_op_pdnode));
        }
        VLOG(0) << " Finish to retrieve cachekv0 node with step = "<< step;

        GET_IR_NODE_FROM_SUBGRAPH(
            fused_multi_transformer0, fused_multi_transformer0, multi_layer_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(
            out0, out0, multi_layer_pattern);

        // GET_IR_NODE_FROM_SUBGRAPH(
        //     cache_kv1, cache_kv1, multi_layer_pattern);
        // GET_IR_NODE_FROM_SUBGRAPH(
        //     fill_constant_batch_size_like, fill_constant_batch_size_like, multi_layer_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(
            fused_multi_transformer1, fused_multi_transformer1, multi_layer_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(
            out1, out1, multi_layer_pattern);
        VLOG(0) << "Finsh Get subgraph";

        // Get op desc
        auto* fused_multi_transformer0_desc = fused_multi_transformer0->Op();
        auto* fused_multi_transformer1_desc = fused_multi_transformer1->Op();
        
        auto inputs_names0 = fused_multi_transformer0_desc->Inputs();
        auto inputs_names1 = fused_multi_transformer1_desc->Inputs();

        // Merge inputs
        std::vector<std::string> inputs_names = {"CacheKV", "FFN1Bias", "FFN1OutScale", "FFN1Weight", "FFN2Bias", "FFN2OutScale",
           "FFN2OutScale",  "FFN2Weight", "FFNLnBias", "FFNLnScale", "LnBias", "LnScale", "OutLinearBias", "OutLinearOutScale", 
           "OutLinearW", "QKVBias", "QKVOutScale", "QKVW"};
        for (const auto& input_name : inputs_names) {
            MergeInput(fused_multi_transformer0_desc, inputs_names0, inputs_names1, input_name);
        }
        VLOG(0) << "Finsh Merge input";

        // Merge outputs
        auto output_names0 = fused_multi_transformer0_desc->Outputs();
        auto output_names1 = fused_multi_transformer1_desc->Outputs();
        fused_multi_transformer0_desc->SetOutput("Out", output_names1["Out"]);
        output_names0["CacheKVOut"].insert(output_names0["CacheKVOut"].end(), 
            output_names1["CacheKVOut"].begin(), output_names1["CacheKVOut"].end());
        for (auto out_name : output_names0["CacheKVOut"]) {
            VLOG(0) << "out_name " << out_name;
        }
        fused_multi_transformer0_desc->SetOutput("CacheKVOut", output_names0["CacheKVOut"]);

        // Merge inputs scale
        std::vector<std::string> attr_names = {"qkv_in_scale", "out_linear_in_scale", "ffn1_in_scale", "ffn2_in_scale"};
        for (const auto& name : attr_names) {
            MergeAttrs<float>(fused_multi_transformer0_desc, fused_multi_transformer1_desc, name);
        }
        VLOG(0) << "Finsh Merge attrs";

        // Dynamic processing
        for (int i = 0; i < step; ++i) {
            // Process fill_constant_op
            // Use x as input of fill_constant_batch_size_like instead of out0
            auto* fill_constant_batch_size_like_desc = fill_op_nodes[i]->Op();
            fill_constant_batch_size_like_desc -> SetInput("Input", {x0->Name()});
            IR_NODE_UNLINK(out1, fill_op_nodes[i]);
            IR_NODE_LINK_TO(x0, fill_op_nodes[i]);

            // Process cache kv node
            IR_NODE_UNLINK(cache_kv1_nodes[i], fused_multi_transformer1);
            IR_NODE_LINK_TO(cache_kv1_nodes[i], fused_multi_transformer0);
        }


        // Relink
        IR_NODE_UNLINK(fused_multi_transformer1, out1);
        IR_NODE_LINK_TO(fused_multi_transformer0, out1);

        IR_NODE_UNLINK(src_mask, fused_multi_transformer1);
        VLOG(0) << "Finsh relinks";

        std::unordered_set<const Node*> marked_nodes({out0, fused_multi_transformer1});
        GraphSafeRemoveNodes(graph, marked_nodes);
        VLOG(0) << "Finsh remove";
        ++fusion_count;
    };
    
    gpd(graph, handler);
    return fusion_count;
    
}

void FuseMultiLayerTransformerPass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "During the fuse_multi_layer_transformer pass, "
          "The scope should not be null."));
  int step = 1;
  int fusion_count = BuildFusion(graph, name_scope_, scope, step++);
  VLOG(0) << "fusion_count is " << fusion_count << " with step = " << step-1;
  auto tmp_fusion_count = fusion_count;
    do {
        tmp_fusion_count = fusion_count;
        fusion_count = BuildFusion(graph, name_scope_, scope, step++);
        VLOG(0) << "fusion_count is " << fusion_count << " with step = " << step-1;
    } while (tmp_fusion_count != fusion_count || fusion_count != 0);

//   PD_THROW("IMULTILAYER");

  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle


REGISTER_PASS(fuse_multi_layer_transformer_pass,
              paddle::framework::ir::FuseMultiLayerTransformerPass);