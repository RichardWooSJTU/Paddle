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

#pragma once

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

// namespace patterns {

// struct MultiTransformer : public PatternBase {
//     MultiTransformer(PDPattern* pattern, const std::string& name_scope)
//       : PatternBase(pattern, name_scope, "multi_transformer") {}
    
//     PDNode* operator()(PDNode* x);

//     // declare operator node's name
//     PATTERN_DECL_NODE(fused_multi_transformer_int8); // new op define a function mul_repr() which generate a unique name and coressponding pdNode 
//     // declare variable node's name so many input
//     PATTERN_DECL_NODE(ln_scale);
//     PATTERN_DECL_NODE(ln_bias);
    
//     PATTERN_DECL_NODE(qkv_w);  
//     PATTERN_DECL_NODE(qkv_bias);  
//     PATTERN_DECL_NODE(cache_kv);  
//     PATTERN_DECL_NODE(time_stamp);  
//     PATTERN_DECL_NODE(src_mask);  

//     PATTERN_DECL_NODE(out_linear_w);
//     PATTERN_DECL_NODE(out_linear_bias);
//     PATTERN_DECL_NODE(ffn_ln_scale);
//     PATTERN_DECL_NODE(ffn_ln_bias);
//     PATTERN_DECL_NODE(ffn1_weight);
//     PATTERN_DECL_NODE(ffn1_bias);
//     PATTERN_DECL_NODE(ffn2_weight);
//     PATTERN_DECL_NODE(ffn2_bias);

//     PATTERN_DECL_NODE(cache_kv_out);
//     PATTERN_DECL_NODE(out);
// };
// } // namespace patterns

class MultiTransformerFusePass : public FusePassBase {
    public:
     MultiTransformerFusePass();
     virtual ~MultiTransformerFusePass(){}
    protected:
     void ApplyImpl(Graph* g) const override;
     void PrepareWeights(framework::Tensor* weight_tensor, int k, int n) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle