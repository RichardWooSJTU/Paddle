#include <memory>
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class TestAttnGemmInt8Op : public framework::OperatorWithKernel {

}

class FusedMultiTransformerINT8OpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
  }
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    test_attn_gemm_int8,
    ops::TestAttnGemmInt8Op,
    ops::TestAttnGemmInt8OpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);