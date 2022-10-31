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

#include <memory>
#include <string>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct FuseMultiLayerTransformerPattern : public PatternBase {
  FuseMultiLayerTransformerPattern(PDPattern* pattern,
                                             const std::string& name_scope)
      : PatternBase(
            pattern, name_scope, "fuse_multi_layer_transformer") {}

  std::unordered_map<std::string, std::string> operator()(bool enable_int8, int step = 1);

  PATTERN_DECL_NODE(src_mask);
  PATTERN_DECL_NODE(x0);
  PATTERN_DECL_NODE(cache_kv0);
  PATTERN_DECL_NODE(fused_multi_transformer0);
  PATTERN_DECL_NODE(out0);
  PATTERN_DECL_NODE(catchkv_out0);

  PATTERN_DECL_NODE(fill_constant_batch_size_like)

  PATTERN_DECL_NODE(cache_kv1);
  PATTERN_DECL_NODE(fused_multi_transformer1);
  PATTERN_DECL_NODE(out1);
  PATTERN_DECL_NODE(catchkv_out1);

  // while loop
  PATTERN_DECL_NODE(while0);

};

}  // namespace patterns

class FuseMultiLayerTransformerPass : public FusePassBase {
 public:
  FuseMultiLayerTransformerPass(){};
  virtual ~FuseMultiLayerTransformerPass() {}

 protected:
  void ApplyImpl(Graph* graph) const;

  const std::string name_scope_{"fuse_multi_layer_transformer"};

 private:
  int BuildFusion(Graph* graph,
                  const std::string& name_scope,
                  Scope* scope, 
                  int step = 1) const;
};


}  // namespace ir
}  // namespace framework
}  // namespace paddle
