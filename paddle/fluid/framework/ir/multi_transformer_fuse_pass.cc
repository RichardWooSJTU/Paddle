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


namespace patterns {

PDNode* MultiTransformer::operator()(PDNode* x) {
    /*
     * Create old subgraph's *PDNode* including op PDNode and tensor PDNode
     */
    
    x->assert_is_op_input("fc", "Input");
    auto *fc = pattern->NewNode(fc_repr())->assert_is_op("fc");
    auto *fc_w_var = pattern->NewNode(w_repr())
                        ->AsInput()
                        ->assert_is_persistable_var()
                        ->assert_is_op_input("fc", "W");
    auto *fc_out_var = pattern->NewNode(fc_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("fc");
    fc->LinksFrom({x, fc_w_var}).LinksTo({fc_out_var});
    return fc_out_var;

}



} // namespace patterns

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

    GraphPatternDetector gpd;
    auto* x = gpd.mutable_pattern()
                ->NewNode("multi_transformer_fuse/x")
                ->AsInput()
                ->assert_is_op_input("fc", "Input");
    
    patterns::MultiTransformer pattern(gpd.mutable_pattern(), "multi_transformer_fuse"/*name_scope_*/);

    pattern(x);

    auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {

        if (subgraph.count(x) <= 0) {
            LOG(WARNING) << "The subgraph is empty.";
            return;
        }  
        VLOG(4) << "handle MultiTransformer fuse";
        GET_IR_NODE_FROM_SUBGRAPH(w, w, pattern); // Node*
        GET_IR_NODE_FROM_SUBGRAPH(fc, fc, pattern);
        GET_IR_NODE_FROM_SUBGRAPH(fc_out, fc_out, pattern);

        // Create new op's Node
        OpDesc new_desc(fc->Op()->Block());
        new_desc.SetType("mul");


        // Set Input of new op
        new_desc.SetInput("X", {subgraph.at(x)->Name()});
        VLOG(1) << "[DEBUG] subgraph.at(x)->Name()" << subgraph.at(x)->Name();
        new_desc.SetInput("Y", {w->Name()});
        VLOG(1) << "[DEBUG] w->Name()" << w->Name();

        // Set Output of new op
        new_desc.SetOutput("Out", {fc_out->Name()});
        VLOG(1) << "[DEBUG] fc_out->Name()" << w->Name();

        // Set attrs of new op
        new_desc.SetAttr("x_num_col_dims", fc->Op()->GetAttr("in_num_col_dims"));

        new_desc.Flush();

        auto new_node = g->CreateOpNode(&new_desc);


        // Relink
        // make new_op's input equal to origin first op's input
        for (auto &in : fc->inputs) {
            new_node->inputs.emplace_back(in);
            // first op's inputs' output(opNode) set to new op (just replace old)
            in->outputs = ReplaceNode(fc, new_node, in->outputs);
        }

        // first op's out: if it is second op's input, delete it, else make it as new op's output
        std::unordered_set<const Node *> nodes2delete;
        for (auto &out : fc->outputs) {
            IR_OP_VAR_LINK(new_node, out);
        }

        nodes2delete.insert(std::move(fc));

        GraphSafeRemoveNodes(g, nodes2delete);

    };
    gpd(g, handler);
}

std::vector<Node *> MultiTransformerFusePass::ReplaceNode(
    Node *cur_node, Node *new_node, const std::vector<Node *> &nodes) const {
        std::vector<Node *> new_list(nodes.size());
        bool has_replaced = false;
        std::transform(
            nodes.begin(), nodes.end(), new_list.begin(), [&](Node *node) -> Node * {
                if (node == cur_node) {
                has_replaced = true;
                return new_node;
                }
                return node;
            });
        PADDLE_ENFORCE_EQ(has_replaced,
                            true,
                            platform::errors::NotFound("Not found %s in the node list.",
                                                    cur_node->Name()));
        return new_list;
}

 // The key to change old graph to new graph is GraphPatternDetector::operator()
 


}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_transformer_fuse_pass,
              paddle::framework::ir::MultiTransformerFusePass);