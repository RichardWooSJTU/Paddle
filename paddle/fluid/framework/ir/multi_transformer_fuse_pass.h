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

namespace patterns {

struct MultiTransformer : public PatternBase {
    MultiTransformer(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "multi_transformer") {}
    
    PDNode* operator()(PDNode* x);

    // declare operator node's name
    PATTERN_DECL_NODE(mul); // new op define a function mul_repr() which generate a unique name and coressponding pdNode 
    PATTERN_DECL_NODE(fc);
    // declare variable node's name
    PATTERN_DECL_NODE(w);
    PATTERN_DECL_NODE(fc_out);  // (x,w) -> mul_out
};
} // namespace patterns

class MultiTransformerFusePass : public FusePassBase {
    public:
     MultiTransformerFusePass();
     virtual ~MultiTransformerFusePass(){}
    protected:
     void ApplyImpl(Graph* g) const override;
     std::vector<Node *> ReplaceNode(
        Node *cur_node, Node *new_node, const std::vector<Node *> &nodes) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle