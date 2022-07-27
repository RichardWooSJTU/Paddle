#pragma once

#include  "paddle/fluid/imperative/tracer.h"
#include  "paddle/fluid/platform/profiler.h"
#include  "pybind11/detail/common.h"
#include  <Python.h>


namespace paddle {
namespace pybind {

std::atomic<int> VarBaseUniqueNameID{0};

static PyObject * imperative_rsqrt(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rsqrt";
    platform::RecordEvent op_type_record_event("rsqrt pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_rsqrt_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rsqrt";
    platform::RecordEvent op_type_record_event("rsqrt pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_multihead_matmul(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multihead_matmul";
    platform::RecordEvent op_type_record_event("multihead_matmul pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    auto BiasQK = GetVarBaseFromArgs(op_type, "BiasQK", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"W", {W}},{"Bias", {Bias}},{"BiasQK", {BiasQK}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_addmm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "addmm";
    platform::RecordEvent op_type_record_event("addmm pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto X = GetVarBaseFromArgs(op_type, "X", args, 1, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_gru(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gru";
    platform::RecordEvent op_type_record_event("gru pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"BatchGate", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchResetHiddenPrev", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchHidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Hidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Weight", {Weight}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["BatchGate"][0],outs["BatchResetHiddenPrev"][0],outs["BatchHidden"][0],outs["Hidden"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_round(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "round";
    platform::RecordEvent op_type_record_event("round pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_round_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "round";
    platform::RecordEvent op_type_record_event("round pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_rank_attention(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rank_attention";
    platform::RecordEvent op_type_record_event("rank_attention pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto RankOffset = GetVarBaseFromArgs(op_type, "RankOffset", args, 1, false);
    auto RankParam = GetVarBaseFromArgs(op_type, "RankParam", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"RankOffset", {RankOffset}},{"RankParam", {RankParam}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_embedding_fc_lstm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_embedding_fc_lstm";
    platform::RecordEvent op_type_record_event("fused_embedding_fc_lstm pybind_imperative_func");
    
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 0, false);
    auto Embeddings = GetVarBaseFromArgs(op_type, "Embeddings", args, 1, false);
    auto WeightH = GetVarBaseFromArgs(op_type, "WeightH", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Hidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Cell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchedInput", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchedHidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchedCell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReorderedH0", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReorderedC0", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Ids", {Ids}},{"Embeddings", {Embeddings}},{"WeightH", {WeightH}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Hidden"][0],outs["Cell"][0],outs["XX"][0],outs["BatchedInput"][0],outs["BatchedHidden"][0],outs["BatchedCell"][0],outs["ReorderedH0"][0],outs["ReorderedC0"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_where_index(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "where_index";
    platform::RecordEvent op_type_record_event("where_index pybind_imperative_func");
    
    auto Condition = GetVarBaseFromArgs(op_type, "Condition", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Condition", {Condition}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bicubic_interp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bicubic_interp";
    platform::RecordEvent op_type_record_event("bicubic_interp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto OutSize = GetVarBaseFromArgs(op_type, "OutSize", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (OutSize != nullptr) {
      ins["OutSize"] = {OutSize};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_arg_min(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "arg_min";
    platform::RecordEvent op_type_record_event("arg_min pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_tile(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "tile";
    platform::RecordEvent op_type_record_event("tile pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_distributed_fused_lamb_init(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "distributed_fused_lamb_init";
    platform::RecordEvent op_type_record_event("distributed_fused_lamb_init pybind_imperative_func");
    
    auto Param = GetVarBaseListFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseListFromArgs(op_type, "Grad", args, 1, false);
    auto ParamOutNum = GetUnsignedLongFromArgs(op_type, "ParamOutNum", args, 2, false);
    auto MasterParamOutNum = GetUnsignedLongFromArgs(op_type, "MasterParamOutNum", args, 3, false);
    auto GradOutNum = GetUnsignedLongFromArgs(op_type, "GradOutNum", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Moment1", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Moment2", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Beta1Pow", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Beta2Pow", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"FusedParamOffsets", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"FP32ShardFusedParamOffsets", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"FP16ShardFusedParamOffsets", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ParamInfo", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ParamOrder", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ParamOut", ConstructDuplicableOutput(ParamOutNum)},{"MasterParamOut", ConstructDuplicableOutput(MasterParamOutNum)},{"GradOut", ConstructDuplicableOutput(GradOutNum)},{"GlobalScale", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Step", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Param", Param},{"Grad", Grad}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Moment1"][0],outs["Moment2"][0],outs["Beta1Pow"][0],outs["Beta2Pow"][0],outs["FusedParamOffsets"][0],outs["FP32ShardFusedParamOffsets"][0],outs["FP16ShardFusedParamOffsets"][0],outs["ParamInfo"][0],outs["ParamOrder"][0],outs["ParamOut"],outs["MasterParamOut"],outs["GradOut"],outs["GlobalScale"][0],outs["Step"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dequantize_linear(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dequantize_linear";
    platform::RecordEvent op_type_record_event("dequantize_linear pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto ZeroPoint = GetVarBaseFromArgs(op_type, "ZeroPoint", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"ZeroPoint", {ZeroPoint}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bilinear_tensor_product(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bilinear_tensor_product";
    platform::RecordEvent op_type_record_event("bilinear_tensor_product pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}},{"Weight", {Weight}}};
    
    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_ctc_align(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "ctc_align";
    platform::RecordEvent op_type_record_event("ctc_align pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pow2_decay_with_linear_warmup(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pow2_decay_with_linear_warmup";
    platform::RecordEvent op_type_record_event("pow2_decay_with_linear_warmup pybind_imperative_func");
    
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 0, false);
    auto Step = GetVarBaseFromArgs(op_type, "Step", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"LearningRateOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"StepOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"LearningRate", {LearningRate}},{"Step", {Step}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["LearningRateOut"][0],outs["StepOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reduce_amin(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reduce_amin";
    platform::RecordEvent op_type_record_event("reduce_amin pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_split(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "split";
    platform::RecordEvent op_type_record_event("split pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseListFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", Out}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fc(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fc";
    platform::RecordEvent op_type_record_event("fc pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"W", {W}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_clear_float_status(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "clear_float_status";
    platform::RecordEvent op_type_record_event("clear_float_status pybind_imperative_func");
    
    auto FloatStatus = GetVarBaseFromArgs(op_type, "FloatStatus", args, 0, false);
    auto FloatStatusOut = GetVarBaseFromArgs(op_type, "FloatStatusOut", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"FloatStatusOut", {FloatStatusOut}}};
    imperative::NameVarBaseMap ins = {{"FloatStatus", {FloatStatus}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["FloatStatusOut"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_load(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "load";
    platform::RecordEvent op_type_record_event("load pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_matmul_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "matmul_v2";
    platform::RecordEvent op_type_record_event("matmul_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_max";
    platform::RecordEvent op_type_record_event("elementwise_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_embedding(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_embedding";
    platform::RecordEvent op_type_record_event("c_embedding pybind_imperative_func");
    
    auto W = GetVarBaseFromArgs(op_type, "W", args, 0, false);
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"W", {W}},{"Ids", {Ids}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_adadelta(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "adadelta";
    platform::RecordEvent op_type_record_event("adadelta pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto AvgSquaredGrad = GetVarBaseFromArgs(op_type, "AvgSquaredGrad", args, 2, false);
    auto AvgSquaredUpdate = GetVarBaseFromArgs(op_type, "AvgSquaredUpdate", args, 3, false);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 4, false);
    auto AvgSquaredGradOut = GetVarBaseFromArgs(op_type, "AvgSquaredGradOut", args, 5, false);
    auto AvgSquaredUpdateOut = GetVarBaseFromArgs(op_type, "AvgSquaredUpdateOut", args, 6, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 7, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"AvgSquaredGradOut", {AvgSquaredGradOut}},{"AvgSquaredUpdateOut", {AvgSquaredUpdateOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"AvgSquaredGrad", {AvgSquaredGrad}},{"AvgSquaredUpdate", {AvgSquaredUpdate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["AvgSquaredGradOut"][0],outs["AvgSquaredUpdateOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_chunk_eval(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "chunk_eval";
    platform::RecordEvent op_type_record_event("chunk_eval pybind_imperative_func");
    
    auto Inference = GetVarBaseFromArgs(op_type, "Inference", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto SeqLength = GetVarBaseFromArgs(op_type, "SeqLength", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Precision", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Recall", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"F1-Score", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"NumInferChunks", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"NumLabelChunks", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"NumCorrectChunks", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Inference", {Inference}},{"Label", {Label}}};
    
    if (SeqLength != nullptr) {
      ins["SeqLength"] = {SeqLength};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Precision"][0],outs["Recall"][0],outs["F1-Score"][0],outs["NumInferChunks"][0],outs["NumLabelChunks"][0],outs["NumCorrectChunks"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_check_finite_and_unscale(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "check_finite_and_unscale";
    platform::RecordEvent op_type_record_event("check_finite_and_unscale pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto Out = GetVarBaseListFromArgs(op_type, "Out", args, 2, false);
    auto FoundInfinite = GetVarBaseFromArgs(op_type, "FoundInfinite", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", Out},{"FoundInfinite", {FoundInfinite}}};
    imperative::NameVarBaseMap ins = {{"X", X},{"Scale", {Scale}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"],outs["FoundInfinite"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sparse_momentum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sparse_momentum";
    platform::RecordEvent op_type_record_event("sparse_momentum pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto Velocity = GetVarBaseFromArgs(op_type, "Velocity", args, 2, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 3, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 4, false);
    auto MasterParam = GetVarBaseFromArgs(op_type, "MasterParam", args, 5, true);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 6, false);
    auto VelocityOut = GetVarBaseFromArgs(op_type, "VelocityOut", args, 7, false);
    auto MasterParamOut = GetVarBaseFromArgs(op_type, "MasterParamOut", args, 8, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 9, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"VelocityOut", {VelocityOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"Velocity", {Velocity}},{"Index", {Index}},{"LearningRate", {LearningRate}}};
    
    if (MasterParam != nullptr) {
      ins["MasterParam"] = {MasterParam};
    }

    outs["MasterParamOut"] = {MasterParamOut};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["VelocityOut"][0],outs["MasterParamOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_complex(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "complex";
    platform::RecordEvent op_type_record_event("complex pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_tan(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "tan";
    platform::RecordEvent op_type_record_event("tan pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_bias_dropout_residual_layer_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_bias_dropout_residual_layer_norm";
    platform::RecordEvent op_type_record_event("fused_bias_dropout_residual_layer_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Residual = GetVarBaseFromArgs(op_type, "Residual", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, true);
    auto LnScale = GetVarBaseFromArgs(op_type, "LnScale", args, 3, true);
    auto LnBias = GetVarBaseFromArgs(op_type, "LnBias", args, 4, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"BiasDropoutResidualOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"DropoutMaskOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LnMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LnVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Residual", {Residual}}};
    
    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    if (LnScale != nullptr) {
      ins["LnScale"] = {LnScale};
    }

    if (LnBias != nullptr) {
      ins["LnBias"] = {LnBias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["BiasDropoutResidualOut"][0],outs["DropoutMaskOut"][0],outs["LnMean"][0],outs["LnVariance"][0],outs["Y"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_adam(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "adam";
    platform::RecordEvent op_type_record_event("adam pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 2, false);
    auto Moment1 = GetVarBaseFromArgs(op_type, "Moment1", args, 3, false);
    auto Moment2 = GetVarBaseFromArgs(op_type, "Moment2", args, 4, false);
    auto Beta1Pow = GetVarBaseFromArgs(op_type, "Beta1Pow", args, 5, false);
    auto Beta2Pow = GetVarBaseFromArgs(op_type, "Beta2Pow", args, 6, false);
    auto MasterParam = GetVarBaseFromArgs(op_type, "MasterParam", args, 7, true);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 8, false);
    auto Moment1Out = GetVarBaseFromArgs(op_type, "Moment1Out", args, 9, false);
    auto Moment2Out = GetVarBaseFromArgs(op_type, "Moment2Out", args, 10, false);
    auto Beta1PowOut = GetVarBaseFromArgs(op_type, "Beta1PowOut", args, 11, false);
    auto Beta2PowOut = GetVarBaseFromArgs(op_type, "Beta2PowOut", args, 12, false);
    auto MasterParamOut = GetVarBaseFromArgs(op_type, "MasterParamOut", args, 13, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 14, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"Moment1Out", {Moment1Out}},{"Moment2Out", {Moment2Out}},{"Beta1PowOut", {Beta1PowOut}},{"Beta2PowOut", {Beta2PowOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"LearningRate", {LearningRate}},{"Moment1", {Moment1}},{"Moment2", {Moment2}},{"Beta1Pow", {Beta1Pow}},{"Beta2Pow", {Beta2Pow}}};
    
    if (MasterParam != nullptr) {
      ins["MasterParam"] = {MasterParam};
    }

    outs["MasterParamOut"] = {MasterParamOut};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["Moment1Out"][0],outs["Moment2Out"][0],outs["Beta1PowOut"][0],outs["Beta2PowOut"][0],outs["MasterParamOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fsp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fsp";
    platform::RecordEvent op_type_record_event("fsp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_where(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "where";
    platform::RecordEvent op_type_record_event("where pybind_imperative_func");
    
    auto Condition = GetVarBaseFromArgs(op_type, "Condition", args, 0, false);
    auto X = GetVarBaseFromArgs(op_type, "X", args, 1, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Condition", {Condition}},{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logical_xor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logical_xor";
    platform::RecordEvent op_type_record_event("logical_xor pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_multiclass_nms3(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multiclass_nms3";
    platform::RecordEvent op_type_record_event("multiclass_nms3 pybind_imperative_func");
    
    auto BBoxes = GetVarBaseFromArgs(op_type, "BBoxes", args, 0, false);
    auto Scores = GetVarBaseFromArgs(op_type, "Scores", args, 1, false);
    auto RoisNum = GetVarBaseFromArgs(op_type, "RoisNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Index", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"NmsRoisNum", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"BBoxes", {BBoxes}},{"Scores", {Scores}}};
    
    if (RoisNum != nullptr) {
      ins["RoisNum"] = {RoisNum};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Index"][0],outs["NmsRoisNum"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_one_hot_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "one_hot_v2";
    platform::RecordEvent op_type_record_event("one_hot_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_softmax(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_softmax";
    platform::RecordEvent op_type_record_event("sequence_softmax pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_affine_channel(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "affine_channel";
    platform::RecordEvent op_type_record_event("affine_channel pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_affine_channel_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "affine_channel";
    platform::RecordEvent op_type_record_event("affine_channel pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_triangular_solve(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "triangular_solve";
    platform::RecordEvent op_type_record_event("triangular_solve pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_topk_avg_pooling(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_topk_avg_pooling";
    platform::RecordEvent op_type_record_event("sequence_topk_avg_pooling pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROW = GetVarBaseFromArgs(op_type, "ROW", args, 1, false);
    auto COLUMN = GetVarBaseFromArgs(op_type, "COLUMN", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"pos", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROW", {ROW}},{"COLUMN", {COLUMN}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["pos"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_space_to_depth(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "space_to_depth";
    platform::RecordEvent op_type_record_event("space_to_depth pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reverse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reverse";
    platform::RecordEvent op_type_record_event("reverse pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_embedding_eltwise_layernorm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_embedding_eltwise_layernorm";
    platform::RecordEvent op_type_record_event("fused_embedding_eltwise_layernorm pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    auto Embs = GetVarBaseListFromArgs(op_type, "Embs", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Ids", Ids},{"Embs", Embs},{"Bias", {Bias}},{"Scale", {Scale}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_expand_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "expand_v2";
    platform::RecordEvent op_type_record_event("expand_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_repeat_interleave(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "repeat_interleave";
    platform::RecordEvent op_type_record_event("repeat_interleave pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto RepeatsTensor = GetVarBaseFromArgs(op_type, "RepeatsTensor", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (RepeatsTensor != nullptr) {
      ins["RepeatsTensor"] = {RepeatsTensor};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lgamma(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lgamma";
    platform::RecordEvent op_type_record_event("lgamma pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_solve(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "solve";
    platform::RecordEvent op_type_record_event("solve pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_deformable_psroi_pooling(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "deformable_psroi_pooling";
    platform::RecordEvent op_type_record_event("deformable_psroi_pooling pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto ROIs = GetVarBaseFromArgs(op_type, "ROIs", args, 1, false);
    auto Trans = GetVarBaseFromArgs(op_type, "Trans", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"TopCount", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"ROIs", {ROIs}},{"Trans", {Trans}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["TopCount"][0],outs["Output"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_transfer_layout(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "transfer_layout";
    platform::RecordEvent op_type_record_event("transfer_layout pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_instance_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "instance_norm";
    platform::RecordEvent op_type_record_event("instance_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, true);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Scale != nullptr) {
      ins["Scale"] = {Scale};
    }

    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["SavedMean"][0],outs["SavedVariance"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_decode_jpeg(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "decode_jpeg";
    platform::RecordEvent op_type_record_event("decode_jpeg pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_distributed_push_sparse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "distributed_push_sparse";
    platform::RecordEvent op_type_record_event("distributed_push_sparse pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    auto Shows = GetVarBaseListFromArgs(op_type, "Shows", args, 1, false);
    auto Clicks = GetVarBaseListFromArgs(op_type, "Clicks", args, 2, false);
    auto OutputsNum = GetUnsignedLongFromArgs(op_type, "OutputsNum", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Outputs", ConstructDuplicableOutput(OutputsNum)}};
    imperative::NameVarBaseMap ins = {{"Ids", Ids},{"Shows", Shows},{"Clicks", Clicks}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Outputs"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_gather_nd(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gather_nd";
    platform::RecordEvent op_type_record_event("gather_nd pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Index", {Index}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reduce_prod(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reduce_prod";
    platform::RecordEvent op_type_record_event("reduce_prod pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_matrix_rank(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "matrix_rank";
    platform::RecordEvent op_type_record_event("matrix_rank pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto TolTensor = GetVarBaseFromArgs(op_type, "TolTensor", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (TolTensor != nullptr) {
      ins["TolTensor"] = {TolTensor};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_asin(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "asin";
    platform::RecordEvent op_type_record_event("asin pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lstmp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lstmp";
    platform::RecordEvent op_type_record_event("lstmp pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 1, false);
    auto ProjWeight = GetVarBaseFromArgs(op_type, "ProjWeight", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Projection", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Cell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchGate", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchCellPreAct", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchHidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Weight", {Weight}},{"ProjWeight", {ProjWeight}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Projection"][0],outs["Cell"][0],outs["BatchGate"][0],outs["BatchCellPreAct"][0],outs["BatchHidden"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_iou_similarity(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "iou_similarity";
    platform::RecordEvent op_type_record_event("iou_similarity pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_huber_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "huber_loss";
    platform::RecordEvent op_type_record_event("huber_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Residual", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Residual"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_one_hot(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "one_hot";
    platform::RecordEvent op_type_record_event("one_hot pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_slice(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_slice";
    platform::RecordEvent op_type_record_event("sequence_slice pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Offset = GetVarBaseFromArgs(op_type, "Offset", args, 1, false);
    auto Length = GetVarBaseFromArgs(op_type, "Length", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Offset", {Offset}},{"Length", {Length}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lookup_table(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lookup_table";
    platform::RecordEvent op_type_record_event("lookup_table pybind_imperative_func");
    
    auto W = GetVarBaseFromArgs(op_type, "W", args, 0, false);
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"W", {W}},{"Ids", {Ids}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_softplus(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "softplus";
    platform::RecordEvent op_type_record_event("softplus pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_depthwise_conv2d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "depthwise_conv2d";
    platform::RecordEvent op_type_record_event("depthwise_conv2d pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_allreduce_sum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_allreduce_sum";
    platform::RecordEvent op_type_record_event("c_allreduce_sum pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_allreduce_sum_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_allreduce_sum";
    platform::RecordEvent op_type_record_event("c_allreduce_sum pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_fc_elementwise_layernorm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_fc_elementwise_layernorm";
    platform::RecordEvent op_type_record_event("fused_fc_elementwise_layernorm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 1, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"W", {W}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sigmoid_cross_entropy_with_logits(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sigmoid_cross_entropy_with_logits";
    platform::RecordEvent op_type_record_event("sigmoid_cross_entropy_with_logits pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sigmoid_cross_entropy_with_logits_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sigmoid_cross_entropy_with_logits";
    platform::RecordEvent op_type_record_event("sigmoid_cross_entropy_with_logits pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_exp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "exp";
    platform::RecordEvent op_type_record_event("exp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_exp_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "exp";
    platform::RecordEvent op_type_record_event("exp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_scatter(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "scatter";
    platform::RecordEvent op_type_record_event("scatter pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 1, false);
    auto Updates = GetVarBaseFromArgs(op_type, "Updates", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Ids", {Ids}},{"Updates", {Updates}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_scatter_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "scatter";
    platform::RecordEvent op_type_record_event("scatter pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 1, false);
    auto Updates = GetVarBaseFromArgs(op_type, "Updates", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Ids", {Ids}},{"Updates", {Updates}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_allreduce_min(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_allreduce_min";
    platform::RecordEvent op_type_record_event("c_allreduce_min pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_allreduce_min_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_allreduce_min";
    platform::RecordEvent op_type_record_event("c_allreduce_min pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_equal_all(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "equal_all";
    platform::RecordEvent op_type_record_event("equal_all pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_searchsorted(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "searchsorted";
    platform::RecordEvent op_type_record_event("searchsorted pybind_imperative_func");
    
    auto SortedSequence = GetVarBaseFromArgs(op_type, "SortedSequence", args, 0, false);
    auto Values = GetVarBaseFromArgs(op_type, "Values", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"SortedSequence", {SortedSequence}},{"Values", {Values}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_squared_mat_sub(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_squared_mat_sub";
    platform::RecordEvent op_type_record_event("fusion_squared_mat_sub pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"SquaredX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SquaredY", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SquaredXY", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["SquaredX"][0],outs["SquaredY"][0],outs["SquaredXY"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unique(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unique";
    platform::RecordEvent op_type_record_event("unique pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Index", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Indices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Counts", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Index"][0],outs["Indices"][0],outs["Counts"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_log(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "log";
    platform::RecordEvent op_type_record_event("log pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_log_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "log";
    platform::RecordEvent op_type_record_event("log pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conv_shift(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv_shift";
    platform::RecordEvent op_type_record_event("conv_shift pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_as_complex(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "as_complex";
    platform::RecordEvent op_type_record_event("as_complex pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_smooth_l1_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "smooth_l1_loss";
    platform::RecordEvent op_type_record_event("smooth_l1_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    auto InsideWeight = GetVarBaseFromArgs(op_type, "InsideWeight", args, 2, true);
    auto OutsideWeight = GetVarBaseFromArgs(op_type, "OutsideWeight", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Diff", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    if (InsideWeight != nullptr) {
      ins["InsideWeight"] = {InsideWeight};
    }

    if (OutsideWeight != nullptr) {
      ins["OutsideWeight"] = {OutsideWeight};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Diff"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_linear_interp_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "linear_interp_v2";
    platform::RecordEvent op_type_record_event("linear_interp_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_momentum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "momentum";
    platform::RecordEvent op_type_record_event("momentum pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto Velocity = GetVarBaseFromArgs(op_type, "Velocity", args, 2, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 3, false);
    auto MasterParam = GetVarBaseFromArgs(op_type, "MasterParam", args, 4, true);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 5, false);
    auto VelocityOut = GetVarBaseFromArgs(op_type, "VelocityOut", args, 6, false);
    auto MasterParamOut = GetVarBaseFromArgs(op_type, "MasterParamOut", args, 7, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"VelocityOut", {VelocityOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"Velocity", {Velocity}},{"LearningRate", {LearningRate}}};
    
    if (MasterParam != nullptr) {
      ins["MasterParam"] = {MasterParam};
    }

    outs["MasterParamOut"] = {MasterParamOut};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["VelocityOut"][0],outs["MasterParamOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_temporal_shift(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "temporal_shift";
    platform::RecordEvent op_type_record_event("temporal_shift pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_nce(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nce";
    platform::RecordEvent op_type_record_event("nce pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, true);
    auto SampleWeight = GetVarBaseFromArgs(op_type, "SampleWeight", args, 4, true);
    auto CustomDistProbs = GetVarBaseFromArgs(op_type, "CustomDistProbs", args, 5, true);
    auto CustomDistAlias = GetVarBaseFromArgs(op_type, "CustomDistAlias", args, 6, true);
    auto CustomDistAliasProbs = GetVarBaseFromArgs(op_type, "CustomDistAliasProbs", args, 7, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Cost", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SampleLogits", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SampleLabels", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Label", {Label}},{"Weight", {Weight}}};
    
    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    if (SampleWeight != nullptr) {
      ins["SampleWeight"] = {SampleWeight};
    }

    if (CustomDistProbs != nullptr) {
      ins["CustomDistProbs"] = {CustomDistProbs};
    }

    if (CustomDistAlias != nullptr) {
      ins["CustomDistAlias"] = {CustomDistAlias};
    }

    if (CustomDistAliasProbs != nullptr) {
      ins["CustomDistAliasProbs"] = {CustomDistAliasProbs};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Cost"][0],outs["SampleLogits"][0],outs["SampleLabels"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_mv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mv";
    platform::RecordEvent op_type_record_event("mv pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Vec = GetVarBaseFromArgs(op_type, "Vec", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Vec", {Vec}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_global_scatter(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "global_scatter";
    platform::RecordEvent op_type_record_event("global_scatter pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto local_count = GetVarBaseFromArgs(op_type, "local_count", args, 1, false);
    auto global_count = GetVarBaseFromArgs(op_type, "global_count", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"local_count", {local_count}},{"global_count", {global_count}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dropout_nd(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dropout_nd";
    platform::RecordEvent op_type_record_event("dropout_nd pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Mask"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_proximal_gd(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "proximal_gd";
    platform::RecordEvent op_type_record_event("proximal_gd pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"LearningRate", {LearningRate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["ParamOut"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_memcpy_h2d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "memcpy_h2d";
    platform::RecordEvent op_type_record_event("memcpy_h2d pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_add_position_encoding(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "add_position_encoding";
    platform::RecordEvent op_type_record_event("add_position_encoding pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cosh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cosh";
    platform::RecordEvent op_type_record_event("cosh pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_hash(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hash";
    platform::RecordEvent op_type_record_event("hash pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_grad_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "grad_add";
    platform::RecordEvent op_type_record_event("grad_add pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sign";
    platform::RecordEvent op_type_record_event("sign pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_prelu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "prelu";
    platform::RecordEvent op_type_record_event("prelu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Alpha = GetVarBaseFromArgs(op_type, "Alpha", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Alpha", {Alpha}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_linspace(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "linspace";
    platform::RecordEvent op_type_record_event("linspace pybind_imperative_func");
    
    auto Start = GetVarBaseFromArgs(op_type, "Start", args, 0, false);
    auto Stop = GetVarBaseFromArgs(op_type, "Stop", args, 1, false);
    auto Num = GetVarBaseFromArgs(op_type, "Num", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Start", {Start}},{"Stop", {Stop}},{"Num", {Num}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_diagonal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_diagonal";
    platform::RecordEvent op_type_record_event("fill_diagonal pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_diagonal_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_diagonal";
    platform::RecordEvent op_type_record_event("fill_diagonal pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logsigmoid(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logsigmoid";
    platform::RecordEvent op_type_record_event("logsigmoid pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_load_combine(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "load_combine";
    platform::RecordEvent op_type_record_event("load_combine pybind_imperative_func");
    
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fetch_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fetch_v2";
    platform::RecordEvent op_type_record_event("fetch_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_randperm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "randperm";
    platform::RecordEvent op_type_record_event("randperm pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_scatter(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_scatter";
    platform::RecordEvent op_type_record_event("sequence_scatter pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 1, false);
    auto Updates = GetVarBaseFromArgs(op_type, "Updates", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Ids", {Ids}},{"Updates", {Updates}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_relu6(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "relu6";
    platform::RecordEvent op_type_record_event("relu6 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_relu6_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "relu6";
    platform::RecordEvent op_type_record_event("relu6 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_partial_sum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "partial_sum";
    platform::RecordEvent op_type_record_event("partial_sum pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_partial_allgather(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "partial_allgather";
    platform::RecordEvent op_type_record_event("partial_allgather pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_partial_allgather_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "partial_allgather";
    platform::RecordEvent op_type_record_event("partial_allgather pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_scatter(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_scatter";
    platform::RecordEvent op_type_record_event("c_scatter pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_alltoall(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "alltoall";
    platform::RecordEvent op_type_record_event("alltoall pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_alltoall_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "alltoall";
    platform::RecordEvent op_type_record_event("alltoall pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conv3d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv3d";
    platform::RecordEvent op_type_record_event("conv3d pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lu_unpack(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lu_unpack";
    platform::RecordEvent op_type_record_event("lu_unpack pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Pivots = GetVarBaseFromArgs(op_type, "Pivots", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Pmat", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"L", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"U", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Pivots", {Pivots}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Pmat"][0],outs["L"][0],outs["U"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lstm_unit(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lstm_unit";
    platform::RecordEvent op_type_record_event("lstm_unit pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto C_prev = GetVarBaseFromArgs(op_type, "C_prev", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"C", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"H", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"C_prev", {C_prev}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["C"][0],outs["H"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_not_equal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "not_equal";
    platform::RecordEvent op_type_record_event("not_equal pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_transpose2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "transpose2";
    platform::RecordEvent op_type_record_event("transpose2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_sync_comm_stream(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_sync_comm_stream";
    platform::RecordEvent op_type_record_event("c_sync_comm_stream pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseListFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", Out}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_uniform_random_batch_size_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "uniform_random_batch_size_like";
    platform::RecordEvent op_type_record_event("uniform_random_batch_size_like pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_yolo_box_head(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "yolo_box_head";
    platform::RecordEvent op_type_record_event("yolo_box_head pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unfold(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unfold";
    platform::RecordEvent op_type_record_event("unfold pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lrn(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lrn";
    platform::RecordEvent op_type_record_event("lrn pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MidOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["MidOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_isclose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isclose";
    platform::RecordEvent op_type_record_event("isclose pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Other = GetVarBaseFromArgs(op_type, "Other", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Other", {Other}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_softmax_with_cross_entropy(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "softmax_with_cross_entropy";
    platform::RecordEvent op_type_record_event("softmax_with_cross_entropy pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Softmax", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Softmax"][0],outs["Loss"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_softmax_with_cross_entropy_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "softmax_with_cross_entropy";
    platform::RecordEvent op_type_record_event("softmax_with_cross_entropy pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      Logits->IsLeaf() && !Logits->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", Logits->Name()));
    Logits->BumpInplaceVersion();
    VLOG(3) << "Var(" << Logits->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Softmax", {Logits}},{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"Logits", "Softmax"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Softmax"][0],outs["Loss"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_isfinite_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isfinite_v2";
    platform::RecordEvent op_type_record_event("isfinite_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bernoulli(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bernoulli";
    platform::RecordEvent op_type_record_event("bernoulli pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_max_pool3d_with_index(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "max_pool3d_with_index";
    platform::RecordEvent op_type_record_event("max_pool3d_with_index pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Mask"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_seqpool_cvm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_seqpool_cvm";
    platform::RecordEvent op_type_record_event("fused_seqpool_cvm pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto CVM = GetVarBaseFromArgs(op_type, "CVM", args, 1, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"X", X},{"CVM", {CVM}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_gaussian_random(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gaussian_random";
    platform::RecordEvent op_type_record_event("gaussian_random pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_flatten2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "flatten2";
    platform::RecordEvent op_type_record_event("flatten2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_flatten2_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "flatten2";
    platform::RecordEvent op_type_record_event("flatten2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_matmul(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "matmul";
    platform::RecordEvent op_type_record_event("matmul pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cvm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cvm";
    platform::RecordEvent op_type_record_event("cvm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto CVM = GetVarBaseFromArgs(op_type, "CVM", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"CVM", {CVM}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_adamax(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "adamax";
    platform::RecordEvent op_type_record_event("adamax pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 2, false);
    auto Moment = GetVarBaseFromArgs(op_type, "Moment", args, 3, false);
    auto InfNorm = GetVarBaseFromArgs(op_type, "InfNorm", args, 4, false);
    auto Beta1Pow = GetVarBaseFromArgs(op_type, "Beta1Pow", args, 5, false);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 6, false);
    auto MomentOut = GetVarBaseFromArgs(op_type, "MomentOut", args, 7, false);
    auto InfNormOut = GetVarBaseFromArgs(op_type, "InfNormOut", args, 8, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 9, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"MomentOut", {MomentOut}},{"InfNormOut", {InfNormOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"LearningRate", {LearningRate}},{"Moment", {Moment}},{"InfNorm", {InfNorm}},{"Beta1Pow", {Beta1Pow}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["MomentOut"][0],outs["InfNormOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_recv_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "recv_v2";
    platform::RecordEvent op_type_record_event("recv_v2 pybind_imperative_func");
    
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_requantize(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "requantize";
    platform::RecordEvent op_type_record_event("requantize pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_masked_select(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "masked_select";
    platform::RecordEvent op_type_record_event("masked_select pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Mask = GetVarBaseFromArgs(op_type, "Mask", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Mask", {Mask}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_range(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "range";
    platform::RecordEvent op_type_record_event("range pybind_imperative_func");
    
    auto Start = GetVarBaseFromArgs(op_type, "Start", args, 0, false);
    auto End = GetVarBaseFromArgs(op_type, "End", args, 1, false);
    auto Step = GetVarBaseFromArgs(op_type, "Step", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Start", {Start}},{"End", {End}},{"Step", {Step}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bitwise_not(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bitwise_not";
    platform::RecordEvent op_type_record_event("bitwise_not pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_trace(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "trace";
    platform::RecordEvent op_type_record_event("trace pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_multinomial(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multinomial";
    platform::RecordEvent op_type_record_event("multinomial pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_modified_huber_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "modified_huber_loss";
    platform::RecordEvent op_type_record_event("modified_huber_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"IntermediateVal", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["IntermediateVal"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_reduce_prod(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_reduce_prod";
    platform::RecordEvent op_type_record_event("c_reduce_prod pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_roll(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "roll";
    platform::RecordEvent op_type_record_event("roll pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_squared_l2_distance(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "squared_l2_distance";
    platform::RecordEvent op_type_record_event("squared_l2_distance pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"sub_result", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["sub_result"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conv3d_transpose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv3d_transpose";
    platform::RecordEvent op_type_record_event("conv3d_transpose pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_share_data(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "share_data";
    platform::RecordEvent op_type_record_event("share_data pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fake_quantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_quantize_abs_max";
    platform::RecordEvent op_type_record_event("fake_quantize_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutScale", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutScale"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_rrelu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rrelu";
    platform::RecordEvent op_type_record_event("rrelu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Noise", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Noise"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unique_with_counts(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unique_with_counts";
    platform::RecordEvent op_type_record_event("unique_with_counts pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Index", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Count", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Index"][0],outs["Count"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill";
    platform::RecordEvent op_type_record_event("fill pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "concat";
    platform::RecordEvent op_type_record_event("concat pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_zeros_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_zeros_like";
    platform::RecordEvent op_type_record_event("fill_zeros_like pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_hierarchical_sigmoid(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hierarchical_sigmoid";
    platform::RecordEvent op_type_record_event("hierarchical_sigmoid pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 1, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 2, false);
    auto PathTable = GetVarBaseFromArgs(op_type, "PathTable", args, 3, true);
    auto PathCode = GetVarBaseFromArgs(op_type, "PathCode", args, 4, true);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 5, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"PreOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"W_Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"W", {W}},{"Label", {Label}}};
    
    if (PathTable != nullptr) {
      ins["PathTable"] = {PathTable};
    }

    if (PathCode != nullptr) {
      ins["PathCode"] = {PathCode};
    }

    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["PreOut"][0],outs["W_Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_isinf_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isinf_v2";
    platform::RecordEvent op_type_record_event("isinf_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_squeeze(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "squeeze";
    platform::RecordEvent op_type_record_event("squeeze pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_multiclass_nms2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multiclass_nms2";
    platform::RecordEvent op_type_record_event("multiclass_nms2 pybind_imperative_func");
    
    auto BBoxes = GetVarBaseFromArgs(op_type, "BBoxes", args, 0, false);
    auto Scores = GetVarBaseFromArgs(op_type, "Scores", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Index", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"BBoxes", {BBoxes}},{"Scores", {Scores}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Index"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bpr_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bpr_loss";
    platform::RecordEvent op_type_record_event("bpr_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fft_c2c(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fft_c2c";
    platform::RecordEvent op_type_record_event("fft_c2c pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bicubic_interp_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bicubic_interp_v2";
    platform::RecordEvent op_type_record_event("bicubic_interp_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_angle(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "angle";
    platform::RecordEvent op_type_record_event("angle pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reshape(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reshape";
    platform::RecordEvent op_type_record_event("reshape pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reshape_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reshape";
    platform::RecordEvent op_type_record_event("reshape pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_coalesce_tensor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "coalesce_tensor";
    platform::RecordEvent op_type_record_event("coalesce_tensor pybind_imperative_func");
    
    auto Input = GetVarBaseListFromArgs(op_type, "Input", args, 0, false);
    auto Output = GetVarBaseListFromArgs(op_type, "Output", args, 1, false);
    auto FusedOutput = GetVarBaseFromArgs(op_type, "FusedOutput", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", Output},{"FusedOutput", {FusedOutput}}};
    imperative::NameVarBaseMap ins = {{"Input", Input}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Output"],outs["FusedOutput"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_roi_align(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "roi_align";
    platform::RecordEvent op_type_record_event("roi_align pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROIs = GetVarBaseFromArgs(op_type, "ROIs", args, 1, false);
    auto RoisNum = GetVarBaseFromArgs(op_type, "RoisNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROIs", {ROIs}}};
    
    if (RoisNum != nullptr) {
      ins["RoisNum"] = {RoisNum};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reshape2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reshape2";
    platform::RecordEvent op_type_record_event("reshape2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Shape = GetVarBaseFromArgs(op_type, "Shape", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Shape != nullptr) {
      ins["Shape"] = {Shape};
    }

    if (ins.count("X") && outs.count("Out")) {
      HandleViewBetweenInputAndOutput(ins["X"][0], outs["Out"][0]);
    }
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reshape2_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reshape2";
    platform::RecordEvent op_type_record_event("reshape2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Shape = GetVarBaseFromArgs(op_type, "Shape", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Shape != nullptr) {
      ins["Shape"] = {Shape};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reduce_any(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reduce_any";
    platform::RecordEvent op_type_record_event("reduce_any pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_limit_by_capacity(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "limit_by_capacity";
    platform::RecordEvent op_type_record_event("limit_by_capacity pybind_imperative_func");
    
    auto expert_count = GetVarBaseFromArgs(op_type, "expert_count", args, 0, false);
    auto capacity = GetVarBaseFromArgs(op_type, "capacity", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"expert_count", {expert_count}},{"capacity", {capacity}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unstack(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unstack";
    platform::RecordEvent op_type_record_event("unstack pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto YNum = GetUnsignedLongFromArgs(op_type, "YNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", ConstructDuplicableOutput(YNum)}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_scatter_nd_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "scatter_nd_add";
    platform::RecordEvent op_type_record_event("scatter_nd_add pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    auto Updates = GetVarBaseFromArgs(op_type, "Updates", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Index", {Index}},{"Updates", {Updates}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_reshape(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_reshape";
    platform::RecordEvent op_type_record_event("sequence_reshape pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bilateral_slice(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bilateral_slice";
    platform::RecordEvent op_type_record_event("bilateral_slice pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Grid = GetVarBaseFromArgs(op_type, "Grid", args, 1, false);
    auto Guide = GetVarBaseFromArgs(op_type, "Guide", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Grid", {Grid}},{"Guide", {Guide}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_any_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_any_like";
    platform::RecordEvent op_type_record_event("fill_any_like pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_partial_recv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "partial_recv";
    platform::RecordEvent op_type_record_event("partial_recv pybind_imperative_func");
    
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_empty(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "empty";
    platform::RecordEvent op_type_record_event("empty pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pad_constant_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pad_constant_like";
    platform::RecordEvent op_type_record_event("pad_constant_like pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pool2d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pool2d";
    platform::RecordEvent op_type_record_event("pool2d pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_size(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "size";
    platform::RecordEvent op_type_record_event("size pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_imag(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "imag";
    platform::RecordEvent op_type_record_event("imag pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pull_gpups_sparse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pull_gpups_sparse";
    platform::RecordEvent op_type_record_event("pull_gpups_sparse pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"Ids", Ids}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_eigh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "eigh";
    platform::RecordEvent op_type_record_event("eigh pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Eigenvalues", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Eigenvectors", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Eigenvalues"][0],outs["Eigenvectors"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_stack(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "stack";
    platform::RecordEvent op_type_record_event("stack pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dgc_momentum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dgc_momentum";
    platform::RecordEvent op_type_record_event("dgc_momentum pybind_imperative_func");
    
    auto current_step = GetVarBaseFromArgs(op_type, "current_step", args, 0, false);
    auto nranks = GetVarBaseFromArgs(op_type, "nranks", args, 1, false);
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 2, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 3, false);
    auto Velocity = GetVarBaseFromArgs(op_type, "Velocity", args, 4, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Grad_out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ParamOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"VelocityOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"current_step", {current_step}},{"nranks", {nranks}},{"Param", {Param}},{"Grad", {Grad}},{"Velocity", {Velocity}},{"LearningRate", {LearningRate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Grad_out"][0],outs["ParamOut"][0],outs["VelocityOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lamb(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lamb";
    platform::RecordEvent op_type_record_event("lamb pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 2, false);
    auto Moment1 = GetVarBaseFromArgs(op_type, "Moment1", args, 3, false);
    auto Moment2 = GetVarBaseFromArgs(op_type, "Moment2", args, 4, false);
    auto Beta1Pow = GetVarBaseFromArgs(op_type, "Beta1Pow", args, 5, false);
    auto Beta2Pow = GetVarBaseFromArgs(op_type, "Beta2Pow", args, 6, false);
    auto MasterParam = GetVarBaseFromArgs(op_type, "MasterParam", args, 7, true);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 8, false);
    auto Moment1Out = GetVarBaseFromArgs(op_type, "Moment1Out", args, 9, false);
    auto Moment2Out = GetVarBaseFromArgs(op_type, "Moment2Out", args, 10, false);
    auto Beta1PowOut = GetVarBaseFromArgs(op_type, "Beta1PowOut", args, 11, true);
    auto Beta2PowOut = GetVarBaseFromArgs(op_type, "Beta2PowOut", args, 12, true);
    auto MasterParamOut = GetVarBaseFromArgs(op_type, "MasterParamOut", args, 13, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 14, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"Moment1Out", {Moment1Out}},{"Moment2Out", {Moment2Out}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"LearningRate", {LearningRate}},{"Moment1", {Moment1}},{"Moment2", {Moment2}},{"Beta1Pow", {Beta1Pow}},{"Beta2Pow", {Beta2Pow}}};
    
    if (MasterParam != nullptr) {
      ins["MasterParam"] = {MasterParam};
    }

    outs["Beta1PowOut"] = {Beta1PowOut};

    outs["Beta2PowOut"] = {Beta2PowOut};

    outs["MasterParamOut"] = {MasterParamOut};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["Moment1Out"][0],outs["Moment2Out"][0],outs["Beta1PowOut"][0],outs["Beta2PowOut"][0],outs["MasterParamOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_generate_proposals_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "generate_proposals_v2";
    platform::RecordEvent op_type_record_event("generate_proposals_v2 pybind_imperative_func");
    
    auto Scores = GetVarBaseFromArgs(op_type, "Scores", args, 0, false);
    auto BboxDeltas = GetVarBaseFromArgs(op_type, "BboxDeltas", args, 1, false);
    auto ImShape = GetVarBaseFromArgs(op_type, "ImShape", args, 2, false);
    auto Anchors = GetVarBaseFromArgs(op_type, "Anchors", args, 3, false);
    auto Variances = GetVarBaseFromArgs(op_type, "Variances", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"RpnRois", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RpnRoiProbs", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RpnRoisNum", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Scores", {Scores}},{"BboxDeltas", {BboxDeltas}},{"ImShape", {ImShape}},{"Anchors", {Anchors}},{"Variances", {Variances}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["RpnRois"][0],outs["RpnRoiProbs"][0],outs["RpnRoisNum"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_sync_calc_stream(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_sync_calc_stream";
    platform::RecordEvent op_type_record_event("c_sync_calc_stream pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bitwise_or(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bitwise_or";
    platform::RecordEvent op_type_record_event("bitwise_or pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_gru_unit(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gru_unit";
    platform::RecordEvent op_type_record_event("gru_unit pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto HiddenPrev = GetVarBaseFromArgs(op_type, "HiddenPrev", args, 1, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Gate", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ResetHiddenPrev", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Hidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"HiddenPrev", {HiddenPrev}},{"Weight", {Weight}}};
    
    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Gate"][0],outs["ResetHiddenPrev"][0],outs["Hidden"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fake_channel_wise_quantize_dequantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_channel_wise_quantize_dequantize_abs_max";
    platform::RecordEvent op_type_record_event("fake_channel_wise_quantize_dequantize_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    auto OutScale = GetVarBaseFromArgs(op_type, "OutScale", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}},{"OutScale", {OutScale}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutScale"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sampling_id(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sampling_id";
    platform::RecordEvent op_type_record_event("sampling_id pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unsqueeze2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unsqueeze2";
    platform::RecordEvent op_type_record_event("unsqueeze2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (ins.count("X") && outs.count("Out")) {
      HandleViewBetweenInputAndOutput(ins["X"][0], outs["Out"][0]);
    }
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unsqueeze2_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unsqueeze2";
    platform::RecordEvent op_type_record_event("unsqueeze2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_transfer_dtype(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "transfer_dtype";
    platform::RecordEvent op_type_record_event("transfer_dtype pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_allreduce(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "allreduce";
    platform::RecordEvent op_type_record_event("allreduce pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_average_accumulates(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "average_accumulates";
    platform::RecordEvent op_type_record_event("average_accumulates pybind_imperative_func");
    
    auto param = GetVarBaseFromArgs(op_type, "param", args, 0, false);
    auto in_sum_1 = GetVarBaseFromArgs(op_type, "in_sum_1", args, 1, false);
    auto in_sum_2 = GetVarBaseFromArgs(op_type, "in_sum_2", args, 2, false);
    auto in_sum_3 = GetVarBaseFromArgs(op_type, "in_sum_3", args, 3, false);
    auto in_num_accumulates = GetVarBaseFromArgs(op_type, "in_num_accumulates", args, 4, false);
    auto in_old_num_accumulates = GetVarBaseFromArgs(op_type, "in_old_num_accumulates", args, 5, false);
    auto in_num_updates = GetVarBaseFromArgs(op_type, "in_num_updates", args, 6, false);
    auto out_sum_1 = GetVarBaseFromArgs(op_type, "out_sum_1", args, 7, false);
    auto out_sum_2 = GetVarBaseFromArgs(op_type, "out_sum_2", args, 8, false);
    auto out_sum_3 = GetVarBaseFromArgs(op_type, "out_sum_3", args, 9, false);
    auto out_num_accumulates = GetVarBaseFromArgs(op_type, "out_num_accumulates", args, 10, false);
    auto out_old_num_accumulates = GetVarBaseFromArgs(op_type, "out_old_num_accumulates", args, 11, false);
    auto out_num_updates = GetVarBaseFromArgs(op_type, "out_num_updates", args, 12, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 13, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"out_sum_1", {out_sum_1}},{"out_sum_2", {out_sum_2}},{"out_sum_3", {out_sum_3}},{"out_num_accumulates", {out_num_accumulates}},{"out_old_num_accumulates", {out_old_num_accumulates}},{"out_num_updates", {out_num_updates}}};
    imperative::NameVarBaseMap ins = {{"param", {param}},{"in_sum_1", {in_sum_1}},{"in_sum_2", {in_sum_2}},{"in_sum_3", {in_sum_3}},{"in_num_accumulates", {in_num_accumulates}},{"in_old_num_accumulates", {in_old_num_accumulates}},{"in_num_updates", {in_num_updates}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["out_sum_1"][0],outs["out_sum_2"][0],outs["out_sum_3"][0],outs["out_num_accumulates"][0],outs["out_old_num_accumulates"][0],outs["out_num_updates"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_enumerate(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_enumerate";
    platform::RecordEvent op_type_record_event("sequence_enumerate pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_seqconv_eltadd_relu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_seqconv_eltadd_relu";
    platform::RecordEvent op_type_record_event("fusion_seqconv_eltadd_relu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ColMat", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Filter", {Filter}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["ColMat"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bce_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bce_loss";
    platform::RecordEvent op_type_record_event("bce_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bce_loss_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bce_loss";
    platform::RecordEvent op_type_record_event("bce_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_generate_proposal_labels(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "generate_proposal_labels";
    platform::RecordEvent op_type_record_event("generate_proposal_labels pybind_imperative_func");
    
    auto RpnRois = GetVarBaseFromArgs(op_type, "RpnRois", args, 0, false);
    auto GtClasses = GetVarBaseFromArgs(op_type, "GtClasses", args, 1, false);
    auto IsCrowd = GetVarBaseFromArgs(op_type, "IsCrowd", args, 2, false);
    auto GtBoxes = GetVarBaseFromArgs(op_type, "GtBoxes", args, 3, false);
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Rois", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LabelsInt32", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BboxTargets", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BboxInsideWeights", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BboxOutsideWeights", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MaxOverlapWithGT", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"RpnRois", {RpnRois}},{"GtClasses", {GtClasses}},{"IsCrowd", {IsCrowd}},{"GtBoxes", {GtBoxes}},{"ImInfo", {ImInfo}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Rois"][0],outs["LabelsInt32"][0],outs["BboxTargets"][0],outs["BboxInsideWeights"][0],outs["BboxOutsideWeights"][0],outs["MaxOverlapWithGT"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_im2sequence(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "im2sequence";
    platform::RecordEvent op_type_record_event("im2sequence pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_isinf(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isinf";
    platform::RecordEvent op_type_record_event("isinf pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_reducescatter(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_reducescatter";
    platform::RecordEvent op_type_record_event("c_reducescatter pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logcumsumexp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logcumsumexp";
    platform::RecordEvent op_type_record_event("logcumsumexp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_adagrad(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "adagrad";
    platform::RecordEvent op_type_record_event("adagrad pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto Moment = GetVarBaseFromArgs(op_type, "Moment", args, 2, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 3, false);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 4, false);
    auto MomentOut = GetVarBaseFromArgs(op_type, "MomentOut", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"MomentOut", {MomentOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"Moment", {Moment}},{"LearningRate", {LearningRate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["MomentOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_linear_chain_crf(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "linear_chain_crf";
    platform::RecordEvent op_type_record_event("linear_chain_crf pybind_imperative_func");
    
    auto Emission = GetVarBaseFromArgs(op_type, "Emission", args, 0, false);
    auto Transition = GetVarBaseFromArgs(op_type, "Transition", args, 1, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 2, false);
    auto Length = GetVarBaseFromArgs(op_type, "Length", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Alpha", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"EmissionExps", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TransitionExps", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LogLikelihood", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Emission", {Emission}},{"Transition", {Transition}},{"Label", {Label}}};
    
    if (Length != nullptr) {
      ins["Length"] = {Length};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Alpha"][0],outs["EmissionExps"][0],outs["TransitionExps"][0],outs["LogLikelihood"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_retinanet_target_assign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "retinanet_target_assign";
    platform::RecordEvent op_type_record_event("retinanet_target_assign pybind_imperative_func");
    
    auto Anchor = GetVarBaseFromArgs(op_type, "Anchor", args, 0, false);
    auto GtBoxes = GetVarBaseFromArgs(op_type, "GtBoxes", args, 1, false);
    auto GtLabels = GetVarBaseFromArgs(op_type, "GtLabels", args, 2, false);
    auto IsCrowd = GetVarBaseFromArgs(op_type, "IsCrowd", args, 3, false);
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"LocationIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ScoreIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TargetBBox", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TargetLabel", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BBoxInsideWeight", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ForegroundNumber", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Anchor", {Anchor}},{"GtBoxes", {GtBoxes}},{"GtLabels", {GtLabels}},{"IsCrowd", {IsCrowd}},{"ImInfo", {ImInfo}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["LocationIndex"][0],outs["ScoreIndex"][0],outs["TargetBBox"][0],outs["TargetLabel"][0],outs["BBoxInsideWeight"][0],outs["ForegroundNumber"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_group(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_group";
    platform::RecordEvent op_type_record_event("fusion_group pybind_imperative_func");
    
    auto Inputs = GetVarBaseListFromArgs(op_type, "Inputs", args, 0, false);
    auto OutsNum = GetUnsignedLongFromArgs(op_type, "OutsNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Outs", ConstructDuplicableOutput(OutsNum)}};
    imperative::NameVarBaseMap ins = {{"Inputs", Inputs}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Outs"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_teacher_student_sigmoid_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "teacher_student_sigmoid_loss";
    platform::RecordEvent op_type_record_event("teacher_student_sigmoid_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_random_crop(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "random_crop";
    platform::RecordEvent op_type_record_event("random_crop pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Seed = GetVarBaseFromArgs(op_type, "Seed", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SeedOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Seed", {Seed}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["SeedOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lookup_table_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lookup_table_v2";
    platform::RecordEvent op_type_record_event("lookup_table_v2 pybind_imperative_func");
    
    auto W = GetVarBaseFromArgs(op_type, "W", args, 0, false);
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"W", {W}},{"Ids", {Ids}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_fmax(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_fmax";
    platform::RecordEvent op_type_record_event("elementwise_fmax pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_graph_sample_neighbors(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "graph_sample_neighbors";
    platform::RecordEvent op_type_record_event("graph_sample_neighbors pybind_imperative_func");
    
    auto Row = GetVarBaseFromArgs(op_type, "Row", args, 0, false);
    auto Col_Ptr = GetVarBaseFromArgs(op_type, "Col_Ptr", args, 1, false);
    auto X = GetVarBaseFromArgs(op_type, "X", args, 2, false);
    auto Eids = GetVarBaseFromArgs(op_type, "Eids", args, 3, true);
    auto Perm_Buffer = GetVarBaseFromArgs(op_type, "Perm_Buffer", args, 4, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out_Count", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out_Eids", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Row", {Row}},{"Col_Ptr", {Col_Ptr}},{"X", {X}}};
    
    if (Eids != nullptr) {
      ins["Eids"] = {Eids};
    }

    if (Perm_Buffer != nullptr) {
      ins["Perm_Buffer"] = {Perm_Buffer};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Out_Count"][0],outs["Out_Eids"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_detection_map(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "detection_map";
    platform::RecordEvent op_type_record_event("detection_map pybind_imperative_func");
    
    auto DetectRes = GetVarBaseFromArgs(op_type, "DetectRes", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"AccumPosCount", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AccumTruePos", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AccumFalsePos", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MAP", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"DetectRes", {DetectRes}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["AccumPosCount"][0],outs["AccumTruePos"][0],outs["AccumFalsePos"][0],outs["MAP"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_l1_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "l1_norm";
    platform::RecordEvent op_type_record_event("l1_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sqrt(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sqrt";
    platform::RecordEvent op_type_record_event("sqrt pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sqrt_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sqrt";
    platform::RecordEvent op_type_record_event("sqrt pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_partial_send(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "partial_send";
    platform::RecordEvent op_type_record_event("partial_send pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    RETURN_PY_NONE
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_elemwise_activation(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_elemwise_activation";
    platform::RecordEvent op_type_record_event("fused_elemwise_activation pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"IntermediateOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["IntermediateOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_slogdeterminant(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "slogdeterminant";
    platform::RecordEvent op_type_record_event("slogdeterminant pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_share_buffer(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "share_buffer";
    platform::RecordEvent op_type_record_event("share_buffer pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    auto XOutNum = GetUnsignedLongFromArgs(op_type, "XOutNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)},{"XOut", ConstructDuplicableOutput(XOutNum)}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"],outs["XOut"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_poisson(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "poisson";
    platform::RecordEvent op_type_record_event("poisson pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bitwise_and(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bitwise_and";
    platform::RecordEvent op_type_record_event("bitwise_and pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_diag_embed(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "diag_embed";
    platform::RecordEvent op_type_record_event("diag_embed pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unbind(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unbind";
    platform::RecordEvent op_type_record_event("unbind pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dropout(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dropout";
    platform::RecordEvent op_type_record_event("dropout pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Mask"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_beam_search(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "beam_search";
    platform::RecordEvent op_type_record_event("beam_search pybind_imperative_func");
    
    auto pre_ids = GetVarBaseFromArgs(op_type, "pre_ids", args, 0, false);
    auto pre_scores = GetVarBaseFromArgs(op_type, "pre_scores", args, 1, false);
    auto scores = GetVarBaseFromArgs(op_type, "scores", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"selected_ids", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"selected_scores", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"pre_ids", {pre_ids}},{"pre_scores", {pre_scores}},{"scores", {scores}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["selected_ids"][0],outs["selected_scores"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_moving_average_abs_max_scale(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "moving_average_abs_max_scale";
    platform::RecordEvent op_type_record_event("moving_average_abs_max_scale pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto InAccum = GetVarBaseFromArgs(op_type, "InAccum", args, 1, true);
    auto InState = GetVarBaseFromArgs(op_type, "InState", args, 2, true);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 3, true);
    auto OutScale = GetVarBaseFromArgs(op_type, "OutScale", args, 4, false);
    auto OutState = GetVarBaseFromArgs(op_type, "OutState", args, 5, true);
    auto OutAccum = GetVarBaseFromArgs(op_type, "OutAccum", args, 6, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 7, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"OutScale", {OutScale}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (InAccum != nullptr) {
      ins["InAccum"] = {InAccum};
    }

    if (InState != nullptr) {
      ins["InState"] = {InState};
    }

    outs["Out"] = {Out};

    outs["OutState"] = {OutState};

    outs["OutAccum"] = {OutAccum};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutScale"][0],outs["OutState"][0],outs["OutAccum"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_greater_than(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "greater_than";
    platform::RecordEvent op_type_record_event("greater_than pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_log_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "log_loss";
    platform::RecordEvent op_type_record_event("log_loss pybind_imperative_func");
    
    auto Predicted = GetVarBaseFromArgs(op_type, "Predicted", args, 0, false);
    auto Labels = GetVarBaseFromArgs(op_type, "Labels", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Predicted", {Predicted}},{"Labels", {Labels}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Loss"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_kron(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "kron";
    platform::RecordEvent op_type_record_event("kron pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sigmoid_focal_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sigmoid_focal_loss";
    platform::RecordEvent op_type_record_event("sigmoid_focal_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto FgNum = GetVarBaseFromArgs(op_type, "FgNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}},{"FgNum", {FgNum}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_rmsprop(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rmsprop";
    platform::RecordEvent op_type_record_event("rmsprop pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto MeanSquare = GetVarBaseFromArgs(op_type, "MeanSquare", args, 1, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 2, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 3, false);
    auto Moment = GetVarBaseFromArgs(op_type, "Moment", args, 4, false);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 5, false);
    auto MomentOut = GetVarBaseFromArgs(op_type, "MomentOut", args, 6, false);
    auto MeanSquareOut = GetVarBaseFromArgs(op_type, "MeanSquareOut", args, 7, false);
    auto MeanGradOut = GetVarBaseFromArgs(op_type, "MeanGradOut", args, 8, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 9, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"MomentOut", {MomentOut}},{"MeanSquareOut", {MeanSquareOut}},{"MeanGradOut", {MeanGradOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"MeanSquare", {MeanSquare}},{"LearningRate", {LearningRate}},{"Grad", {Grad}},{"Moment", {Moment}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["MomentOut"][0],outs["MeanSquareOut"][0],outs["MeanGradOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conv2d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv2d";
    platform::RecordEvent op_type_record_event("conv2d pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_graph_reindex(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "graph_reindex";
    platform::RecordEvent op_type_record_event("graph_reindex pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Neighbors = GetVarBaseFromArgs(op_type, "Neighbors", args, 1, false);
    auto Count = GetVarBaseFromArgs(op_type, "Count", args, 2, false);
    auto HashTable_Value = GetVarBaseFromArgs(op_type, "HashTable_Value", args, 3, true);
    auto HashTable_Index = GetVarBaseFromArgs(op_type, "HashTable_Index", args, 4, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Reindex_Src", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Reindex_Dst", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out_Nodes", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Neighbors", {Neighbors}},{"Count", {Count}}};
    
    if (HashTable_Value != nullptr) {
      ins["HashTable_Value"] = {HashTable_Value};
    }

    if (HashTable_Index != nullptr) {
      ins["HashTable_Index"] = {HashTable_Index};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Reindex_Src"][0],outs["Reindex_Dst"][0],outs["Out_Nodes"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_uniform_random_inplace(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "uniform_random_inplace";
    platform::RecordEvent op_type_record_event("uniform_random_inplace pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_uniform_random_inplace_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "uniform_random_inplace";
    platform::RecordEvent op_type_record_event("uniform_random_inplace pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_maxout(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "maxout";
    platform::RecordEvent op_type_record_event("maxout pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lstsq(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lstsq";
    platform::RecordEvent op_type_record_event("lstsq pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Solution", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Rank", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SingularValues", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Solution"][0],outs["Rank"][0],outs["SingularValues"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_linear_interp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "linear_interp";
    platform::RecordEvent op_type_record_event("linear_interp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto OutSize = GetVarBaseFromArgs(op_type, "OutSize", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (OutSize != nullptr) {
      ins["OutSize"] = {OutSize};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_graph_khop_sampler(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "graph_khop_sampler";
    platform::RecordEvent op_type_record_event("graph_khop_sampler pybind_imperative_func");
    
    auto Row = GetVarBaseFromArgs(op_type, "Row", args, 0, false);
    auto Eids = GetVarBaseFromArgs(op_type, "Eids", args, 1, true);
    auto Col_Ptr = GetVarBaseFromArgs(op_type, "Col_Ptr", args, 2, false);
    auto X = GetVarBaseFromArgs(op_type, "X", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out_Src", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out_Dst", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Sample_Index", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Reindex_X", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out_Eids", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Row", {Row}},{"Col_Ptr", {Col_Ptr}},{"X", {X}}};
    
    if (Eids != nullptr) {
      ins["Eids"] = {Eids};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out_Src"][0],outs["Out_Dst"][0],outs["Sample_Index"][0],outs["Reindex_X"][0],outs["Out_Eids"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_put_along_axis(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "put_along_axis";
    platform::RecordEvent op_type_record_event("put_along_axis pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    auto Value = GetVarBaseFromArgs(op_type, "Value", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Result", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Index", {Index}},{"Value", {Value}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Result"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_put_along_axis_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "put_along_axis";
    platform::RecordEvent op_type_record_event("put_along_axis pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    auto Value = GetVarBaseFromArgs(op_type, "Value", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      Input->IsLeaf() && !Input->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", Input->Name()));
    Input->BumpInplaceVersion();
    VLOG(3) << "Var(" << Input->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Result", {Input}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Index", {Index}},{"Value", {Value}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"Input", "Result"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Result"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_auc(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "auc";
    platform::RecordEvent op_type_record_event("auc pybind_imperative_func");
    
    auto Predict = GetVarBaseFromArgs(op_type, "Predict", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto StatPos = GetVarBaseFromArgs(op_type, "StatPos", args, 2, false);
    auto StatNeg = GetVarBaseFromArgs(op_type, "StatNeg", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"AUC", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"StatPosOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"StatNegOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Predict", {Predict}},{"Label", {Label}},{"StatPos", {StatPos}},{"StatNeg", {StatNeg}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["AUC"][0],outs["StatPosOut"][0],outs["StatNegOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logical_or(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logical_or";
    platform::RecordEvent op_type_record_event("logical_or pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_batch_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "batch_norm";
    platform::RecordEvent op_type_record_event("batch_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    auto Mean = GetVarBaseFromArgs(op_type, "Mean", args, 3, false);
    auto Variance = GetVarBaseFromArgs(op_type, "Variance", args, 4, false);
    auto MomentumTensor = GetVarBaseFromArgs(op_type, "MomentumTensor", args, 5, true);
    auto MeanOut = GetVarBaseFromArgs(op_type, "MeanOut", args, 6, false);
    auto VarianceOut = GetVarBaseFromArgs(op_type, "VarianceOut", args, 7, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MeanOut", {MeanOut}},{"VarianceOut", {VarianceOut}},{"SavedMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReserveSpace", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"Bias", {Bias}},{"Mean", {Mean}},{"Variance", {Variance}}};
    
    if (MomentumTensor != nullptr) {
      ins["MomentumTensor"] = {MomentumTensor};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["MeanOut"][0],outs["VarianceOut"][0],outs["SavedMean"][0],outs["SavedVariance"][0],outs["ReserveSpace"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_reduce_sum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_reduce_sum";
    platform::RecordEvent op_type_record_event("c_reduce_sum pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_add";
    platform::RecordEvent op_type_record_event("elementwise_add pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_add_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_add";
    platform::RecordEvent op_type_record_event("elementwise_add pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_acos(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "acos";
    platform::RecordEvent op_type_record_event("acos pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_send_and_recv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "send_and_recv";
    platform::RecordEvent op_type_record_event("send_and_recv pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unpool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unpool";
    platform::RecordEvent op_type_record_event("unpool pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Indices = GetVarBaseFromArgs(op_type, "Indices", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Indices", {Indices}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cumprod(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cumprod";
    platform::RecordEvent op_type_record_event("cumprod pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sample_logits(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sample_logits";
    platform::RecordEvent op_type_record_event("sample_logits pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Labels = GetVarBaseFromArgs(op_type, "Labels", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Samples", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Probabilities", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LogitsDim", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LabelsDim", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SampledLogits", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SampledLabels", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Labels", {Labels}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Samples"][0],outs["Probabilities"][0],outs["LogitsDim"][0],outs["LabelsDim"][0],outs["SampledLogits"][0],outs["SampledLabels"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pull_box_extended_sparse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pull_box_extended_sparse";
    platform::RecordEvent op_type_record_event("pull_box_extended_sparse pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    auto OutExtendNum = GetUnsignedLongFromArgs(op_type, "OutExtendNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)},{"OutExtend", ConstructDuplicableOutput(OutExtendNum)}};
    imperative::NameVarBaseMap ins = {{"Ids", Ids}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"],outs["OutExtend"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_crop_tensor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "crop_tensor";
    platform::RecordEvent op_type_record_event("crop_tensor pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Shape = GetVarBaseFromArgs(op_type, "Shape", args, 1, true);
    auto Offsets = GetVarBaseFromArgs(op_type, "Offsets", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Shape != nullptr) {
      ins["Shape"] = {Shape};
    }

    if (Offsets != nullptr) {
      ins["Offsets"] = {Offsets};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_constant(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_constant";
    platform::RecordEvent op_type_record_event("fill_constant pybind_imperative_func");
    
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_deformable_conv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "deformable_conv";
    platform::RecordEvent op_type_record_event("deformable_conv pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Offset = GetVarBaseFromArgs(op_type, "Offset", args, 1, false);
    auto Mask = GetVarBaseFromArgs(op_type, "Mask", args, 2, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Offset", {Offset}},{"Mask", {Mask}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_generate_mask_labels(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "generate_mask_labels";
    platform::RecordEvent op_type_record_event("generate_mask_labels pybind_imperative_func");
    
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 0, false);
    auto GtClasses = GetVarBaseFromArgs(op_type, "GtClasses", args, 1, false);
    auto IsCrowd = GetVarBaseFromArgs(op_type, "IsCrowd", args, 2, false);
    auto GtSegms = GetVarBaseFromArgs(op_type, "GtSegms", args, 3, false);
    auto Rois = GetVarBaseFromArgs(op_type, "Rois", args, 4, false);
    auto LabelsInt32 = GetVarBaseFromArgs(op_type, "LabelsInt32", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"MaskRois", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RoiHasMaskInt32", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MaskInt32", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"ImInfo", {ImInfo}},{"GtClasses", {GtClasses}},{"IsCrowd", {IsCrowd}},{"GtSegms", {GtSegms}},{"Rois", {Rois}},{"LabelsInt32", {LabelsInt32}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["MaskRois"][0],outs["RoiHasMaskInt32"][0],outs["MaskInt32"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_locality_aware_nms(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "locality_aware_nms";
    platform::RecordEvent op_type_record_event("locality_aware_nms pybind_imperative_func");
    
    auto BBoxes = GetVarBaseFromArgs(op_type, "BBoxes", args, 0, false);
    auto Scores = GetVarBaseFromArgs(op_type, "Scores", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"BBoxes", {BBoxes}},{"Scores", {Scores}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_expand_as(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "expand_as";
    platform::RecordEvent op_type_record_event("expand_as pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto target_tensor = GetVarBaseFromArgs(op_type, "target_tensor", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"target_tensor", {target_tensor}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_matrix_power(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "matrix_power";
    platform::RecordEvent op_type_record_event("matrix_power pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_greater_equal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "greater_equal";
    platform::RecordEvent op_type_record_event("greater_equal pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_generate_proposals(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "generate_proposals";
    platform::RecordEvent op_type_record_event("generate_proposals pybind_imperative_func");
    
    auto Scores = GetVarBaseFromArgs(op_type, "Scores", args, 0, false);
    auto BboxDeltas = GetVarBaseFromArgs(op_type, "BboxDeltas", args, 1, false);
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 2, false);
    auto Anchors = GetVarBaseFromArgs(op_type, "Anchors", args, 3, false);
    auto Variances = GetVarBaseFromArgs(op_type, "Variances", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"RpnRois", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RpnRoiProbs", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RpnRoisNum", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Scores", {Scores}},{"BboxDeltas", {BboxDeltas}},{"ImInfo", {ImInfo}},{"Anchors", {Anchors}},{"Variances", {Variances}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["RpnRois"][0],outs["RpnRoiProbs"][0],outs["RpnRoisNum"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_number_count(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "number_count";
    platform::RecordEvent op_type_record_event("number_count pybind_imperative_func");
    
    auto numbers = GetVarBaseFromArgs(op_type, "numbers", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"numbers", {numbers}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bilinear_interp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bilinear_interp";
    platform::RecordEvent op_type_record_event("bilinear_interp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto OutSize = GetVarBaseFromArgs(op_type, "OutSize", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (OutSize != nullptr) {
      ins["OutSize"] = {OutSize};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_distributed_fused_lamb(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "distributed_fused_lamb";
    platform::RecordEvent op_type_record_event("distributed_fused_lamb pybind_imperative_func");
    
    auto Param = GetVarBaseListFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseListFromArgs(op_type, "Grad", args, 1, false);
    auto Moment1 = GetVarBaseFromArgs(op_type, "Moment1", args, 2, false);
    auto Moment2 = GetVarBaseFromArgs(op_type, "Moment2", args, 3, false);
    auto Beta1Pow = GetVarBaseFromArgs(op_type, "Beta1Pow", args, 4, false);
    auto Beta2Pow = GetVarBaseFromArgs(op_type, "Beta2Pow", args, 5, false);
    auto FusedParamOffsets = GetVarBaseFromArgs(op_type, "FusedParamOffsets", args, 6, false);
    auto FP32ShardFusedParamOffsets = GetVarBaseFromArgs(op_type, "FP32ShardFusedParamOffsets", args, 7, false);
    auto FP16ShardFusedParamOffsets = GetVarBaseFromArgs(op_type, "FP16ShardFusedParamOffsets", args, 8, false);
    auto ParamInfo = GetVarBaseFromArgs(op_type, "ParamInfo", args, 9, false);
    auto ParamOrder = GetVarBaseFromArgs(op_type, "ParamOrder", args, 10, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 11, false);
    auto GlobalScale = GetVarBaseFromArgs(op_type, "GlobalScale", args, 12, false);
    auto ParamOutNum = GetUnsignedLongFromArgs(op_type, "ParamOutNum", args, 13, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 14, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Moment1Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Moment2Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Beta1PowOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Beta2PowOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ParamOut", ConstructDuplicableOutput(ParamOutNum)},{"FoundInf", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Step", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Param", Param},{"Grad", Grad},{"Moment1", {Moment1}},{"Moment2", {Moment2}},{"Beta1Pow", {Beta1Pow}},{"Beta2Pow", {Beta2Pow}},{"FusedParamOffsets", {FusedParamOffsets}},{"FP32ShardFusedParamOffsets", {FP32ShardFusedParamOffsets}},{"FP16ShardFusedParamOffsets", {FP16ShardFusedParamOffsets}},{"ParamInfo", {ParamInfo}},{"ParamOrder", {ParamOrder}},{"LearningRate", {LearningRate}},{"GlobalScale", {GlobalScale}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Moment1Out"][0],outs["Moment2Out"][0],outs["Beta1PowOut"][0],outs["Beta2PowOut"][0],outs["ParamOut"],outs["FoundInf"][0],outs["Step"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sigmoid(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sigmoid";
    platform::RecordEvent op_type_record_event("sigmoid pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sigmoid_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sigmoid";
    platform::RecordEvent op_type_record_event("sigmoid pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_inplace_abn(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "inplace_abn";
    platform::RecordEvent op_type_record_event("inplace_abn pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    auto Mean = GetVarBaseFromArgs(op_type, "Mean", args, 3, false);
    auto Variance = GetVarBaseFromArgs(op_type, "Variance", args, 4, false);
    auto MomentumTensor = GetVarBaseFromArgs(op_type, "MomentumTensor", args, 5, true);
    auto MeanOut = GetVarBaseFromArgs(op_type, "MeanOut", args, 6, false);
    auto VarianceOut = GetVarBaseFromArgs(op_type, "VarianceOut", args, 7, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MeanOut", {MeanOut}},{"VarianceOut", {VarianceOut}},{"SavedMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReserveSpace", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"Bias", {Bias}},{"Mean", {Mean}},{"Variance", {Variance}}};
    
    if (MomentumTensor != nullptr) {
      ins["MomentumTensor"] = {MomentumTensor};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["MeanOut"][0],outs["VarianceOut"][0],outs["SavedMean"][0],outs["SavedVariance"][0],outs["ReserveSpace"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_inplace_abn_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "inplace_abn";
    platform::RecordEvent op_type_record_event("inplace_abn pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    auto Mean = GetVarBaseFromArgs(op_type, "Mean", args, 3, false);
    auto Variance = GetVarBaseFromArgs(op_type, "Variance", args, 4, false);
    auto MomentumTensor = GetVarBaseFromArgs(op_type, "MomentumTensor", args, 5, true);
    auto MeanOut = GetVarBaseFromArgs(op_type, "MeanOut", args, 6, false);
    auto VarianceOut = GetVarBaseFromArgs(op_type, "VarianceOut", args, 7, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Y", {X}},{"MeanOut", {MeanOut}},{"VarianceOut", {VarianceOut}},{"SavedMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReserveSpace", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"Bias", {Bias}},{"Mean", {Mean}},{"Variance", {Variance}}};
    
    if (MomentumTensor != nullptr) {
      ins["MomentumTensor"] = {MomentumTensor};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Y"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["MeanOut"][0],outs["VarianceOut"][0],outs["SavedMean"][0],outs["SavedVariance"][0],outs["ReserveSpace"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_softshrink(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "softshrink";
    platform::RecordEvent op_type_record_event("softshrink pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_mul(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mul";
    platform::RecordEvent op_type_record_event("mul pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_data_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "data_norm";
    platform::RecordEvent op_type_record_event("data_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto BatchSize = GetVarBaseFromArgs(op_type, "BatchSize", args, 1, false);
    auto BatchSum = GetVarBaseFromArgs(op_type, "BatchSum", args, 2, false);
    auto BatchSquareSum = GetVarBaseFromArgs(op_type, "BatchSquareSum", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Means", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Scales", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"BatchSize", {BatchSize}},{"BatchSum", {BatchSum}},{"BatchSquareSum", {BatchSquareSum}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["Means"][0],outs["Scales"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_multi_transformer(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_multi_transformer";
    platform::RecordEvent op_type_record_event("fused_multi_transformer pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto LnScale = GetVarBaseListFromArgs(op_type, "LnScale", args, 1, false);
    auto LnBias = GetVarBaseListFromArgs(op_type, "LnBias", args, 2, false);
    auto QKVW = GetVarBaseListFromArgs(op_type, "QKVW", args, 3, false);
    auto QKVBias = GetVarBaseListFromArgs(op_type, "QKVBias", args, 4, true);
    auto CacheKV = GetVarBaseListFromArgs(op_type, "CacheKV", args, 5, true);
    auto TimeStep = GetVarBaseFromArgs(op_type, "TimeStep", args, 6, true);
    auto SrcMask = GetVarBaseFromArgs(op_type, "SrcMask", args, 7, true);
    auto OutLinearW = GetVarBaseListFromArgs(op_type, "OutLinearW", args, 8, false);
    auto OutLinearBias = GetVarBaseListFromArgs(op_type, "OutLinearBias", args, 9, true);
    auto FFNLnScale = GetVarBaseListFromArgs(op_type, "FFNLnScale", args, 10, false);
    auto FFNLnBias = GetVarBaseListFromArgs(op_type, "FFNLnBias", args, 11, false);
    auto FFN1Weight = GetVarBaseListFromArgs(op_type, "FFN1Weight", args, 12, false);
    auto FFN1Bias = GetVarBaseListFromArgs(op_type, "FFN1Bias", args, 13, true);
    auto FFN2Weight = GetVarBaseListFromArgs(op_type, "FFN2Weight", args, 14, false);
    auto FFN2Bias = GetVarBaseListFromArgs(op_type, "FFN2Bias", args, 15, true);
    auto CacheKVOut = GetVarBaseListFromArgs(op_type, "CacheKVOut", args, 16, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 17, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"LnScale", LnScale},{"LnBias", LnBias},{"QKVW", QKVW},{"OutLinearW", OutLinearW},{"FFNLnScale", FFNLnScale},{"FFNLnBias", FFNLnBias},{"FFN1Weight", FFN1Weight},{"FFN2Weight", FFN2Weight}};
    
    if (QKVBias.size() != 0) {
      ins["QKVBias"] = QKVBias;
    }

    if (CacheKV.size() != 0) {
      ins["CacheKV"] = CacheKV;
    }

    if (TimeStep != nullptr) {
      ins["TimeStep"] = {TimeStep};
    }

    if (SrcMask != nullptr) {
      ins["SrcMask"] = {SrcMask};
    }

    if (OutLinearBias.size() != 0) {
      ins["OutLinearBias"] = OutLinearBias;
    }

    if (FFN1Bias.size() != 0) {
      ins["FFN1Bias"] = FFN1Bias;
    }

    if (FFN2Bias.size() != 0) {
      ins["FFN2Bias"] = FFN2Bias;
    }

    outs["CacheKVOut"] = CacheKVOut;

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["CacheKVOut"],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_asinh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "asinh";
    platform::RecordEvent op_type_record_event("asinh pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_get_tensor_from_selected_rows(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "get_tensor_from_selected_rows";
    platform::RecordEvent op_type_record_event("get_tensor_from_selected_rows pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_spp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "spp";
    platform::RecordEvent op_type_record_event("spp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_floor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "floor";
    platform::RecordEvent op_type_record_event("floor pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_floor_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "floor";
    platform::RecordEvent op_type_record_event("floor pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_as_real(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "as_real";
    platform::RecordEvent op_type_record_event("as_real pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_gelu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gelu";
    platform::RecordEvent op_type_record_event("gelu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_retinanet_detection_output(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "retinanet_detection_output";
    platform::RecordEvent op_type_record_event("retinanet_detection_output pybind_imperative_func");
    
    auto BBoxes = GetVarBaseListFromArgs(op_type, "BBoxes", args, 0, false);
    auto Scores = GetVarBaseListFromArgs(op_type, "Scores", args, 1, false);
    auto Anchors = GetVarBaseListFromArgs(op_type, "Anchors", args, 2, false);
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"BBoxes", BBoxes},{"Scores", Scores},{"Anchors", Anchors},{"ImInfo", {ImInfo}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_minus(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "minus";
    platform::RecordEvent op_type_record_event("minus pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_push_dense(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "push_dense";
    platform::RecordEvent op_type_record_event("push_dense pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {};
    imperative::NameVarBaseMap ins = {{"Ids", Ids}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    RETURN_PY_NONE
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_silu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "silu";
    platform::RecordEvent op_type_record_event("silu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_erase(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_erase";
    platform::RecordEvent op_type_record_event("sequence_erase pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_real(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "real";
    platform::RecordEvent op_type_record_event("real pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_nearest_interp_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nearest_interp_v2";
    platform::RecordEvent op_type_record_event("nearest_interp_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dgc_clip_by_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dgc_clip_by_norm";
    platform::RecordEvent op_type_record_event("dgc_clip_by_norm pybind_imperative_func");
    
    auto current_step = GetVarBaseFromArgs(op_type, "current_step", args, 0, false);
    auto X = GetVarBaseFromArgs(op_type, "X", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"current_step", {current_step}},{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_squeeze2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "squeeze2";
    platform::RecordEvent op_type_record_event("squeeze2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (ins.count("X") && outs.count("Out")) {
      HandleViewBetweenInputAndOutput(ins["X"][0], outs["Out"][0]);
    }
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_squeeze2_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "squeeze2";
    platform::RecordEvent op_type_record_event("squeeze2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conj(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conj";
    platform::RecordEvent op_type_record_event("conj pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_strided_slice(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "strided_slice";
    platform::RecordEvent op_type_record_event("strided_slice pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto StartsTensor = GetVarBaseFromArgs(op_type, "StartsTensor", args, 1, true);
    auto EndsTensor = GetVarBaseFromArgs(op_type, "EndsTensor", args, 2, true);
    auto StridesTensor = GetVarBaseFromArgs(op_type, "StridesTensor", args, 3, true);
    auto StartsTensorList = GetVarBaseListFromArgs(op_type, "StartsTensorList", args, 4, true);
    auto EndsTensorList = GetVarBaseListFromArgs(op_type, "EndsTensorList", args, 5, true);
    auto StridesTensorList = GetVarBaseListFromArgs(op_type, "StridesTensorList", args, 6, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 7, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    if (StartsTensor != nullptr) {
      ins["StartsTensor"] = {StartsTensor};
    }

    if (EndsTensor != nullptr) {
      ins["EndsTensor"] = {EndsTensor};
    }

    if (StridesTensor != nullptr) {
      ins["StridesTensor"] = {StridesTensor};
    }

    if (StartsTensorList.size() != 0) {
      ins["StartsTensorList"] = StartsTensorList;
    }

    if (EndsTensorList.size() != 0) {
      ins["EndsTensorList"] = EndsTensorList;
    }

    if (StridesTensorList.size() != 0) {
      ins["StridesTensorList"] = StridesTensorList;
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_precision_recall(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "precision_recall";
    platform::RecordEvent op_type_record_event("precision_recall pybind_imperative_func");
    
    auto MaxProbs = GetVarBaseFromArgs(op_type, "MaxProbs", args, 0, false);
    auto Indices = GetVarBaseFromArgs(op_type, "Indices", args, 1, false);
    auto Labels = GetVarBaseFromArgs(op_type, "Labels", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"BatchMetrics", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AccumMetrics", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AccumStatesInfo", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"MaxProbs", {MaxProbs}},{"Indices", {Indices}},{"Labels", {Labels}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["BatchMetrics"][0],outs["AccumMetrics"][0],outs["AccumStatesInfo"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_seqexpand_concat_fc(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_seqexpand_concat_fc";
    platform::RecordEvent op_type_record_event("fusion_seqexpand_concat_fc pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto FCWeight = GetVarBaseFromArgs(op_type, "FCWeight", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"FCOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X},{"FCWeight", {FCWeight}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["FCOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_save(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "save";
    platform::RecordEvent op_type_record_event("save pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    RETURN_PY_NONE
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_depthwise_conv2d_transpose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "depthwise_conv2d_transpose";
    platform::RecordEvent op_type_record_event("depthwise_conv2d_transpose pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fake_quantize_range_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_quantize_range_abs_max";
    platform::RecordEvent op_type_record_event("fake_quantize_range_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto InScale = GetVarBaseFromArgs(op_type, "InScale", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutScale", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"InScale", {InScale}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutScale"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_positive_negative_pair(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "positive_negative_pair";
    platform::RecordEvent op_type_record_event("positive_negative_pair pybind_imperative_func");
    
    auto Score = GetVarBaseFromArgs(op_type, "Score", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto QueryID = GetVarBaseFromArgs(op_type, "QueryID", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"PositivePair", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"NegativePair", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Score", {Score}},{"Label", {Label}},{"QueryID", {QueryID}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["PositivePair"][0],outs["NegativePair"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_square(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "square";
    platform::RecordEvent op_type_record_event("square pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_square_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "square";
    platform::RecordEvent op_type_record_event("square pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_var_conv_2d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "var_conv_2d";
    platform::RecordEvent op_type_record_event("var_conv_2d pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROW = GetVarBaseFromArgs(op_type, "ROW", args, 1, false);
    auto COLUMN = GetVarBaseFromArgs(op_type, "COLUMN", args, 2, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Col", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROW", {ROW}},{"COLUMN", {COLUMN}},{"W", {W}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Col"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_log1p(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "log1p";
    platform::RecordEvent op_type_record_event("log1p pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_channel_shuffle(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "channel_shuffle";
    platform::RecordEvent op_type_record_event("channel_shuffle pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_atan2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "atan2";
    platform::RecordEvent op_type_record_event("atan2 pybind_imperative_func");
    
    auto X1 = GetVarBaseFromArgs(op_type, "X1", args, 0, false);
    auto X2 = GetVarBaseFromArgs(op_type, "X2", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X1", {X1}},{"X2", {X2}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_softmax_mask_upper_triangle(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_softmax_mask_upper_triangle";
    platform::RecordEvent op_type_record_event("fused_softmax_mask_upper_triangle pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_clip_by_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "clip_by_norm";
    platform::RecordEvent op_type_record_event("clip_by_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_box_decoder_and_assign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "box_decoder_and_assign";
    platform::RecordEvent op_type_record_event("box_decoder_and_assign pybind_imperative_func");
    
    auto PriorBox = GetVarBaseFromArgs(op_type, "PriorBox", args, 0, false);
    auto TargetBox = GetVarBaseFromArgs(op_type, "TargetBox", args, 1, false);
    auto BoxScore = GetVarBaseFromArgs(op_type, "BoxScore", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"DecodeBox", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutputAssignBox", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"PriorBox", {PriorBox}},{"TargetBox", {TargetBox}},{"BoxScore", {BoxScore}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["DecodeBox"][0],outs["OutputAssignBox"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_roi_pool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "roi_pool";
    platform::RecordEvent op_type_record_event("roi_pool pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROIs = GetVarBaseFromArgs(op_type, "ROIs", args, 1, false);
    auto RoisNum = GetVarBaseFromArgs(op_type, "RoisNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Argmax", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROIs", {ROIs}}};
    
    if (RoisNum != nullptr) {
      ins["RoisNum"] = {RoisNum};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Argmax"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fft_r2c(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fft_r2c";
    platform::RecordEvent op_type_record_event("fft_r2c pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_overlap_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "overlap_add";
    platform::RecordEvent op_type_record_event("overlap_add pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_constant_batch_size_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_constant_batch_size_like";
    platform::RecordEvent op_type_record_event("fill_constant_batch_size_like pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_any(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_any";
    platform::RecordEvent op_type_record_event("fill_any pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_any_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_any";
    platform::RecordEvent op_type_record_event("fill_any pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dequantize_log(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dequantize_log";
    platform::RecordEvent op_type_record_event("dequantize_log pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Dict = GetVarBaseFromArgs(op_type, "Dict", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Dict", {Dict}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_split(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_split";
    platform::RecordEvent op_type_record_event("c_split pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_barrier(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "barrier";
    platform::RecordEvent op_type_record_event("barrier pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_max_pool2d_with_index(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "max_pool2d_with_index";
    platform::RecordEvent op_type_record_event("max_pool2d_with_index pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Mask"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pad3d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pad3d";
    platform::RecordEvent op_type_record_event("pad3d pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "norm";
    platform::RecordEvent op_type_record_event("norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Norm", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Norm"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_viterbi_decode(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "viterbi_decode";
    platform::RecordEvent op_type_record_event("viterbi_decode pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Transition = GetVarBaseFromArgs(op_type, "Transition", args, 1, false);
    auto Length = GetVarBaseFromArgs(op_type, "Length", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Scores", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Path", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Transition", {Transition}},{"Length", {Length}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Scores"][0],outs["Path"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_mish(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mish";
    platform::RecordEvent op_type_record_event("mish pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_box_coder(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "box_coder";
    platform::RecordEvent op_type_record_event("box_coder pybind_imperative_func");
    
    auto PriorBox = GetVarBaseFromArgs(op_type, "PriorBox", args, 0, false);
    auto PriorBoxVar = GetVarBaseFromArgs(op_type, "PriorBoxVar", args, 1, true);
    auto TargetBox = GetVarBaseFromArgs(op_type, "TargetBox", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"OutputBox", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"PriorBox", {PriorBox}},{"TargetBox", {TargetBox}}};
    
    if (PriorBoxVar != nullptr) {
      ins["PriorBoxVar"] = {PriorBoxVar};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["OutputBox"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_flatten(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "flatten";
    platform::RecordEvent op_type_record_event("flatten pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_flatten_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "flatten";
    platform::RecordEvent op_type_record_event("flatten pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_mod(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_mod";
    platform::RecordEvent op_type_record_event("elementwise_mod pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_margin_cross_entropy(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "margin_cross_entropy";
    platform::RecordEvent op_type_record_event("margin_cross_entropy pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Softmax", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Softmax"][0],outs["Loss"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pull_sparse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pull_sparse";
    platform::RecordEvent op_type_record_event("pull_sparse pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    auto W = GetVarBaseListFromArgs(op_type, "W", args, 1, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"Ids", Ids},{"W", W}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logical_and(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logical_and";
    platform::RecordEvent op_type_record_event("logical_and pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pow(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pow";
    platform::RecordEvent op_type_record_event("pow pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dirichlet(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dirichlet";
    platform::RecordEvent op_type_record_event("dirichlet pybind_imperative_func");
    
    auto Alpha = GetVarBaseFromArgs(op_type, "Alpha", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Alpha", {Alpha}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_stanh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "stanh";
    platform::RecordEvent op_type_record_event("stanh pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_label_smooth(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "label_smooth";
    platform::RecordEvent op_type_record_event("label_smooth pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto PriorDist = GetVarBaseFromArgs(op_type, "PriorDist", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (PriorDist != nullptr) {
      ins["PriorDist"] = {PriorDist};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fold(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fold";
    platform::RecordEvent op_type_record_event("fold pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_merged_momentum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "merged_momentum";
    platform::RecordEvent op_type_record_event("merged_momentum pybind_imperative_func");
    
    auto Param = GetVarBaseListFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseListFromArgs(op_type, "Grad", args, 1, false);
    auto Velocity = GetVarBaseListFromArgs(op_type, "Velocity", args, 2, false);
    auto LearningRate = GetVarBaseListFromArgs(op_type, "LearningRate", args, 3, false);
    auto MasterParam = GetVarBaseListFromArgs(op_type, "MasterParam", args, 4, true);
    auto ParamOut = GetVarBaseListFromArgs(op_type, "ParamOut", args, 5, false);
    auto VelocityOut = GetVarBaseListFromArgs(op_type, "VelocityOut", args, 6, false);
    auto MasterParamOut = GetVarBaseListFromArgs(op_type, "MasterParamOut", args, 7, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", ParamOut},{"VelocityOut", VelocityOut}};
    imperative::NameVarBaseMap ins = {{"Param", Param},{"Grad", Grad},{"Velocity", Velocity},{"LearningRate", LearningRate}};
    
    if (MasterParam.size() != 0) {
      ins["MasterParam"] = MasterParam;
    }

    outs["MasterParamOut"] = MasterParamOut;

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"],outs["VelocityOut"],outs["MasterParamOut"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_reduce_min(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_reduce_min";
    platform::RecordEvent op_type_record_event("c_reduce_min pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_ascend_trigger(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "ascend_trigger";
    platform::RecordEvent op_type_record_event("ascend_trigger pybind_imperative_func");
    
    auto FeedList = GetVarBaseListFromArgs(op_type, "FeedList", args, 0, false);
    auto FetchListNum = GetUnsignedLongFromArgs(op_type, "FetchListNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"FetchList", ConstructDuplicableOutput(FetchListNum)}};
    imperative::NameVarBaseMap ins = {{"FeedList", FeedList}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["FetchList"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_rpn_target_assign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rpn_target_assign";
    platform::RecordEvent op_type_record_event("rpn_target_assign pybind_imperative_func");
    
    auto Anchor = GetVarBaseFromArgs(op_type, "Anchor", args, 0, false);
    auto GtBoxes = GetVarBaseFromArgs(op_type, "GtBoxes", args, 1, false);
    auto IsCrowd = GetVarBaseFromArgs(op_type, "IsCrowd", args, 2, false);
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"LocationIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ScoreIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TargetBBox", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TargetLabel", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BBoxInsideWeight", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Anchor", {Anchor}},{"GtBoxes", {GtBoxes}},{"IsCrowd", {IsCrowd}},{"ImInfo", {ImInfo}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["LocationIndex"][0],outs["ScoreIndex"][0],outs["TargetBBox"][0],outs["TargetLabel"][0],outs["BBoxInsideWeight"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_feedforward(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_feedforward";
    platform::RecordEvent op_type_record_event("fused_feedforward pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Dropout1Seed = GetVarBaseFromArgs(op_type, "Dropout1Seed", args, 1, true);
    auto Dropout2Seed = GetVarBaseFromArgs(op_type, "Dropout2Seed", args, 2, true);
    auto Linear1Weight = GetVarBaseFromArgs(op_type, "Linear1Weight", args, 3, false);
    auto Linear1Bias = GetVarBaseFromArgs(op_type, "Linear1Bias", args, 4, true);
    auto Linear2Weight = GetVarBaseFromArgs(op_type, "Linear2Weight", args, 5, false);
    auto Linear2Bias = GetVarBaseFromArgs(op_type, "Linear2Bias", args, 6, true);
    auto Ln1Scale = GetVarBaseFromArgs(op_type, "Ln1Scale", args, 7, true);
    auto Ln1Bias = GetVarBaseFromArgs(op_type, "Ln1Bias", args, 8, true);
    auto Ln2Scale = GetVarBaseFromArgs(op_type, "Ln2Scale", args, 9, true);
    auto Ln2Bias = GetVarBaseFromArgs(op_type, "Ln2Bias", args, 10, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 11, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Dropout1Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Dropout2Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln1Mean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln1Variance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln2Mean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln2Variance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Linear1Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln1Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Dropout1Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Dropout2Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Linear1Weight", {Linear1Weight}},{"Linear2Weight", {Linear2Weight}}};
    
    if (Dropout1Seed != nullptr) {
      ins["Dropout1Seed"] = {Dropout1Seed};
    }

    if (Dropout2Seed != nullptr) {
      ins["Dropout2Seed"] = {Dropout2Seed};
    }

    if (Linear1Bias != nullptr) {
      ins["Linear1Bias"] = {Linear1Bias};
    }

    if (Linear2Bias != nullptr) {
      ins["Linear2Bias"] = {Linear2Bias};
    }

    if (Ln1Scale != nullptr) {
      ins["Ln1Scale"] = {Ln1Scale};
    }

    if (Ln1Bias != nullptr) {
      ins["Ln1Bias"] = {Ln1Bias};
    }

    if (Ln2Scale != nullptr) {
      ins["Ln2Scale"] = {Ln2Scale};
    }

    if (Ln2Bias != nullptr) {
      ins["Ln2Bias"] = {Ln2Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Dropout1Mask"][0],outs["Dropout2Mask"][0],outs["Ln1Mean"][0],outs["Ln1Variance"][0],outs["Ln2Mean"][0],outs["Ln2Variance"][0],outs["Linear1Out"][0],outs["Ln1Out"][0],outs["Dropout1Out"][0],outs["Dropout2Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_roi_perspective_transform(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "roi_perspective_transform";
    platform::RecordEvent op_type_record_event("roi_perspective_transform pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROIs = GetVarBaseFromArgs(op_type, "ROIs", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TransformMatrix", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out2InIdx", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out2InWeights", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROIs", {ROIs}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Mask"][0],outs["TransformMatrix"][0],outs["Out2InIdx"][0],outs["Out2InWeights"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_expand(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "expand";
    platform::RecordEvent op_type_record_event("expand pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ExpandTimes = GetVarBaseFromArgs(op_type, "ExpandTimes", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (ExpandTimes != nullptr) {
      ins["ExpandTimes"] = {ExpandTimes};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_prroi_pool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "prroi_pool";
    platform::RecordEvent op_type_record_event("prroi_pool pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROIs = GetVarBaseFromArgs(op_type, "ROIs", args, 1, false);
    auto BatchRoINums = GetVarBaseFromArgs(op_type, "BatchRoINums", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROIs", {ROIs}}};
    
    if (BatchRoINums != nullptr) {
      ins["BatchRoINums"] = {BatchRoINums};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pool3d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pool3d";
    platform::RecordEvent op_type_record_event("pool3d pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_memcpy(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "memcpy";
    platform::RecordEvent op_type_record_event("memcpy pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_distribute_fpn_proposals(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "distribute_fpn_proposals";
    platform::RecordEvent op_type_record_event("distribute_fpn_proposals pybind_imperative_func");
    
    auto FpnRois = GetVarBaseFromArgs(op_type, "FpnRois", args, 0, false);
    auto RoisNum = GetVarBaseFromArgs(op_type, "RoisNum", args, 1, true);
    auto MultiFpnRoisNum = GetUnsignedLongFromArgs(op_type, "MultiFpnRoisNum", args, 2, false);
    auto MultiLevelRoIsNumNum = GetUnsignedLongFromArgs(op_type, "MultiLevelRoIsNumNum", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"MultiFpnRois", ConstructDuplicableOutput(MultiFpnRoisNum)},{"RestoreIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MultiLevelRoIsNum", ConstructDuplicableOutput(MultiLevelRoIsNumNum)}};
    imperative::NameVarBaseMap ins = {{"FpnRois", {FpnRois}}};
    
    if (RoisNum != nullptr) {
      ins["RoisNum"] = {RoisNum};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["MultiFpnRois"],outs["RestoreIndex"][0],outs["MultiLevelRoIsNum"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_frame(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "frame";
    platform::RecordEvent op_type_record_event("frame pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bincount(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bincount";
    platform::RecordEvent op_type_record_event("bincount pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Weights = GetVarBaseFromArgs(op_type, "Weights", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Weights != nullptr) {
      ins["Weights"] = {Weights};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_shape(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "shape";
    platform::RecordEvent op_type_record_event("shape pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_mode(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mode";
    platform::RecordEvent op_type_record_event("mode pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Indices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Indices"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_group_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "group_norm";
    platform::RecordEvent op_type_record_event("group_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, true);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, true);
    auto Mean = GetVarBaseFromArgs(op_type, "Mean", args, 3, false);
    auto Variance = GetVarBaseFromArgs(op_type, "Variance", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mean", {Mean}},{"Variance", {Variance}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Scale != nullptr) {
      ins["Scale"] = {Scale};
    }

    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["Mean"][0],outs["Variance"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_softmax_with_cross_entropy(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_softmax_with_cross_entropy";
    platform::RecordEvent op_type_record_event("c_softmax_with_cross_entropy pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Softmax", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Softmax"][0],outs["Loss"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_softmax_with_cross_entropy_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_softmax_with_cross_entropy";
    platform::RecordEvent op_type_record_event("c_softmax_with_cross_entropy pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      Logits->IsLeaf() && !Logits->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", Logits->Name()));
    Logits->BumpInplaceVersion();
    VLOG(3) << "Var(" << Logits->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Softmax", {Logits}},{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"Logits", "Softmax"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Softmax"][0],outs["Loss"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_resnet_unit(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "resnet_unit";
    platform::RecordEvent op_type_record_event("resnet_unit pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto FilterX = GetVarBaseFromArgs(op_type, "FilterX", args, 1, false);
    auto ScaleX = GetVarBaseFromArgs(op_type, "ScaleX", args, 2, false);
    auto BiasX = GetVarBaseFromArgs(op_type, "BiasX", args, 3, false);
    auto MeanX = GetVarBaseFromArgs(op_type, "MeanX", args, 4, false);
    auto VarX = GetVarBaseFromArgs(op_type, "VarX", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BitMask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ConvX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedMeanX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedInvstdX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RunningMeanX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RunningVarX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"FilterX", {FilterX}},{"ScaleX", {ScaleX}},{"BiasX", {BiasX}},{"MeanX", {MeanX}},{"VarX", {VarX}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["BitMask"][0],outs["ConvX"][0],outs["SavedMeanX"][0],outs["SavedInvstdX"][0],outs["RunningMeanX"][0],outs["RunningVarX"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_expand_as(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_expand_as";
    platform::RecordEvent op_type_record_event("sequence_expand_as pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cos_sim(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cos_sim";
    platform::RecordEvent op_type_record_event("cos_sim pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XNorm", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"YNorm", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XNorm"][0],outs["YNorm"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_eigvals(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "eigvals";
    platform::RecordEvent op_type_record_event("eigvals pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_save_combine(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "save_combine";
    platform::RecordEvent op_type_record_event("save_combine pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    RETURN_PY_NONE
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_class_center_sample(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "class_center_sample";
    platform::RecordEvent op_type_record_event("class_center_sample pybind_imperative_func");
    
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"RemappedLabel", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SampledLocalClassCenter", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["RemappedLabel"][0],outs["SampledLocalClassCenter"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_fmin(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_fmin";
    platform::RecordEvent op_type_record_event("elementwise_fmin pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_read_file(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "read_file";
    platform::RecordEvent op_type_record_event("read_file pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_isfinite(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isfinite";
    platform::RecordEvent op_type_record_event("isfinite pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_arg_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "arg_max";
    platform::RecordEvent op_type_record_event("arg_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_equal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "equal";
    platform::RecordEvent op_type_record_event("equal pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fake_dequantize_max_abs(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_dequantize_max_abs";
    platform::RecordEvent op_type_record_event("fake_dequantize_max_abs pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_qr(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "qr";
    platform::RecordEvent op_type_record_event("qr pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Q", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"R", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Q"][0],outs["R"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_anchor_generator(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "anchor_generator";
    platform::RecordEvent op_type_record_event("anchor_generator pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Anchors", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Variances", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Anchors"][0],outs["Variances"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_layer_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "layer_norm";
    platform::RecordEvent op_type_record_event("layer_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, true);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Variance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Scale != nullptr) {
      ins["Scale"] = {Scale};
    }

    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["Mean"][0],outs["Variance"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_merge_selected_rows(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "merge_selected_rows";
    platform::RecordEvent op_type_record_event("merge_selected_rows pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_acosh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "acosh";
    platform::RecordEvent op_type_record_event("acosh pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_stft(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "stft";
    platform::RecordEvent op_type_record_event("stft pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Window = GetVarBaseFromArgs(op_type, "Window", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Window", {Window}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_less_equal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "less_equal";
    platform::RecordEvent op_type_record_event("less_equal pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_rnn(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rnn";
    platform::RecordEvent op_type_record_event("rnn pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto PreState = GetVarBaseListFromArgs(op_type, "PreState", args, 1, false);
    auto WeightList = GetVarBaseListFromArgs(op_type, "WeightList", args, 2, false);
    auto SequenceLength = GetVarBaseFromArgs(op_type, "SequenceLength", args, 3, true);
    auto DropoutState = GetVarBaseFromArgs(op_type, "DropoutState", args, 4, true);
    auto StateNum = GetUnsignedLongFromArgs(op_type, "StateNum", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Reserve", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"State", ConstructDuplicableOutput(StateNum)}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"PreState", PreState},{"WeightList", WeightList}};
    
    if (SequenceLength != nullptr) {
      ins["SequenceLength"] = {SequenceLength};
    }

    outs["DropoutState"] = {DropoutState};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["DropoutState"][0],outs["Reserve"][0],outs["Out"][0],outs["State"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_lstm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_lstm";
    platform::RecordEvent op_type_record_event("fusion_lstm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto WeightX = GetVarBaseFromArgs(op_type, "WeightX", args, 1, false);
    auto WeightH = GetVarBaseFromArgs(op_type, "WeightH", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Hidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Cell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchedInput", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchedHidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchedCell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReorderedH0", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReorderedC0", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"CheckedCell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"WeightX", {WeightX}},{"WeightH", {WeightH}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Hidden"][0],outs["Cell"][0],outs["XX"][0],outs["BatchedInput"][0],outs["BatchedHidden"][0],outs["BatchedCell"][0],outs["ReorderedH0"][0],outs["ReorderedC0"][0],outs["CheckedCell"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lars_momentum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lars_momentum";
    platform::RecordEvent op_type_record_event("lars_momentum pybind_imperative_func");
    
    auto Param = GetVarBaseListFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseListFromArgs(op_type, "Grad", args, 1, false);
    auto Velocity = GetVarBaseListFromArgs(op_type, "Velocity", args, 2, false);
    auto LearningRate = GetVarBaseListFromArgs(op_type, "LearningRate", args, 3, false);
    auto ParamOut = GetVarBaseListFromArgs(op_type, "ParamOut", args, 4, false);
    auto VelocityOut = GetVarBaseListFromArgs(op_type, "VelocityOut", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", ParamOut},{"VelocityOut", VelocityOut}};
    imperative::NameVarBaseMap ins = {{"Param", Param},{"Grad", Grad},{"Velocity", Velocity},{"LearningRate", LearningRate}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"],outs["VelocityOut"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_hard_sigmoid(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hard_sigmoid";
    platform::RecordEvent op_type_record_event("hard_sigmoid pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_hard_sigmoid_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hard_sigmoid";
    platform::RecordEvent op_type_record_event("hard_sigmoid pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_isnan(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isnan";
    platform::RecordEvent op_type_record_event("isnan pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_floordiv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_floordiv";
    platform::RecordEvent op_type_record_event("elementwise_floordiv pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_correlation(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "correlation";
    platform::RecordEvent op_type_record_event("correlation pybind_imperative_func");
    
    auto Input1 = GetVarBaseFromArgs(op_type, "Input1", args, 0, false);
    auto Input2 = GetVarBaseFromArgs(op_type, "Input2", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input1", {Input1}},{"Input2", {Input2}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_histogram(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "histogram";
    platform::RecordEvent op_type_record_event("histogram pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_gather_tree(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gather_tree";
    platform::RecordEvent op_type_record_event("gather_tree pybind_imperative_func");
    
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 0, false);
    auto Parents = GetVarBaseFromArgs(op_type, "Parents", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Ids", {Ids}},{"Parents", {Parents}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_nanmedian(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nanmedian";
    platform::RecordEvent op_type_record_event("nanmedian pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"MedianIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["MedianIndex"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_segment_pool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "segment_pool";
    platform::RecordEvent op_type_record_event("segment_pool pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto SegmentIds = GetVarBaseFromArgs(op_type, "SegmentIds", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SummedIds", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"SegmentIds", {SegmentIds}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["SummedIds"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_repeated_fc_relu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_repeated_fc_relu";
    platform::RecordEvent op_type_record_event("fusion_repeated_fc_relu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto W = GetVarBaseListFromArgs(op_type, "W", args, 1, false);
    auto Bias = GetVarBaseListFromArgs(op_type, "Bias", args, 2, false);
    auto ReluOutNum = GetUnsignedLongFromArgs(op_type, "ReluOutNum", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ReluOut", ConstructDuplicableOutput(ReluOutNum)},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"W", W},{"Bias", Bias}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ReluOut"],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sync_batch_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sync_batch_norm";
    platform::RecordEvent op_type_record_event("sync_batch_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    auto Mean = GetVarBaseFromArgs(op_type, "Mean", args, 3, false);
    auto Variance = GetVarBaseFromArgs(op_type, "Variance", args, 4, false);
    auto MeanOut = GetVarBaseFromArgs(op_type, "MeanOut", args, 5, false);
    auto VarianceOut = GetVarBaseFromArgs(op_type, "VarianceOut", args, 6, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 7, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MeanOut", {MeanOut}},{"VarianceOut", {VarianceOut}},{"SavedMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReserveSpace", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"Bias", {Bias}},{"Mean", {Mean}},{"Variance", {Variance}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["MeanOut"][0],outs["VarianceOut"][0],outs["SavedMean"][0],outs["SavedVariance"][0],outs["ReserveSpace"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_nop(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nop";
    platform::RecordEvent op_type_record_event("nop pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_attention(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_attention";
    platform::RecordEvent op_type_record_event("fused_attention pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto LnScale = GetVarBaseFromArgs(op_type, "LnScale", args, 1, true);
    auto LnBias = GetVarBaseFromArgs(op_type, "LnBias", args, 2, true);
    auto QKVW = GetVarBaseFromArgs(op_type, "QKVW", args, 3, false);
    auto QKVBias = GetVarBaseFromArgs(op_type, "QKVBias", args, 4, true);
    auto CacheKV = GetVarBaseFromArgs(op_type, "CacheKV", args, 5, true);
    auto SrcMask = GetVarBaseFromArgs(op_type, "SrcMask", args, 6, true);
    auto OutLinearW = GetVarBaseFromArgs(op_type, "OutLinearW", args, 7, false);
    auto OutLinearBias = GetVarBaseFromArgs(op_type, "OutLinearBias", args, 8, true);
    auto Ln2Scale = GetVarBaseFromArgs(op_type, "Ln2Scale", args, 9, true);
    auto Ln2Bias = GetVarBaseFromArgs(op_type, "Ln2Bias", args, 10, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 11, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"LnMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LnVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LnOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"QKVOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"QKVBiasOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TransposeOut2", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"QKOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"QKTVOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SoftmaxOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AttnDropoutMaskOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AttnDropoutOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SrcMaskOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"FMHAOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutLinearOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"DropoutMaskOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln2Mean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln2Variance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BiasDropoutResidualOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"CacheKVOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"QKVW", {QKVW}},{"OutLinearW", {OutLinearW}}};
    
    if (LnScale != nullptr) {
      ins["LnScale"] = {LnScale};
    }

    if (LnBias != nullptr) {
      ins["LnBias"] = {LnBias};
    }

    if (QKVBias != nullptr) {
      ins["QKVBias"] = {QKVBias};
    }

    if (CacheKV != nullptr) {
      ins["CacheKV"] = {CacheKV};
    }

    if (SrcMask != nullptr) {
      ins["SrcMask"] = {SrcMask};
    }

    if (OutLinearBias != nullptr) {
      ins["OutLinearBias"] = {OutLinearBias};
    }

    if (Ln2Scale != nullptr) {
      ins["Ln2Scale"] = {Ln2Scale};
    }

    if (Ln2Bias != nullptr) {
      ins["Ln2Bias"] = {Ln2Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["LnMean"][0],outs["LnVariance"][0],outs["LnOut"][0],outs["QKVOut"][0],outs["QKVBiasOut"][0],outs["TransposeOut2"][0],outs["QKOut"][0],outs["QKTVOut"][0],outs["SoftmaxOut"][0],outs["AttnDropoutMaskOut"][0],outs["AttnDropoutOut"][0],outs["SrcMaskOut"][0],outs["FMHAOut"][0],outs["OutLinearOut"][0],outs["DropoutMaskOut"][0],outs["Ln2Mean"][0],outs["Ln2Variance"][0],outs["BiasDropoutResidualOut"][0],outs["CacheKVOut"][0],outs["Y"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_filter_by_instag(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "filter_by_instag";
    platform::RecordEvent op_type_record_event("filter_by_instag pybind_imperative_func");
    
    auto Ins = GetVarBaseFromArgs(op_type, "Ins", args, 0, false);
    auto Ins_tag = GetVarBaseFromArgs(op_type, "Ins_tag", args, 1, false);
    auto Filter_tag = GetVarBaseFromArgs(op_type, "Filter_tag", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LossWeight", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"IndexMap", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Ins", {Ins}},{"Ins_tag", {Ins_tag}},{"Filter_tag", {Filter_tag}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["LossWeight"][0],outs["IndexMap"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_expand_as_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "expand_as_v2";
    platform::RecordEvent op_type_record_event("expand_as_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_diag_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "diag_v2";
    platform::RecordEvent op_type_record_event("diag_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pull_box_sparse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pull_box_sparse";
    platform::RecordEvent op_type_record_event("pull_box_sparse pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"Ids", Ids}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_nll_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nll_loss";
    platform::RecordEvent op_type_record_event("nll_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Total_weight", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    if (Weight != nullptr) {
      ins["Weight"] = {Weight};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Total_weight"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dot(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dot";
    platform::RecordEvent op_type_record_event("dot pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_scale(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "scale";
    platform::RecordEvent op_type_record_event("scale pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_scale_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "scale";
    platform::RecordEvent op_type_record_event("scale pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_shuffle_batch(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "shuffle_batch";
    platform::RecordEvent op_type_record_event("shuffle_batch pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Seed = GetVarBaseFromArgs(op_type, "Seed", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ShuffleIdx", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SeedOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Seed", {Seed}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["ShuffleIdx"][0],outs["SeedOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_diag(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "diag";
    platform::RecordEvent op_type_record_event("diag pybind_imperative_func");
    
    auto Diagonal = GetVarBaseFromArgs(op_type, "Diagonal", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Diagonal", {Diagonal}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_multiplex(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multiplex";
    platform::RecordEvent op_type_record_event("multiplex pybind_imperative_func");
    
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 0, false);
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Ids", {Ids}},{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_leaky_relu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "leaky_relu";
    platform::RecordEvent op_type_record_event("leaky_relu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_leaky_relu_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "leaky_relu";
    platform::RecordEvent op_type_record_event("leaky_relu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_allclose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "allclose";
    platform::RecordEvent op_type_record_event("allclose pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Other = GetVarBaseFromArgs(op_type, "Other", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Other", {Other}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_adamw(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "adamw";
    platform::RecordEvent op_type_record_event("adamw pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 2, false);
    auto Moment1 = GetVarBaseFromArgs(op_type, "Moment1", args, 3, false);
    auto Moment2 = GetVarBaseFromArgs(op_type, "Moment2", args, 4, false);
    auto Beta1Pow = GetVarBaseFromArgs(op_type, "Beta1Pow", args, 5, false);
    auto Beta2Pow = GetVarBaseFromArgs(op_type, "Beta2Pow", args, 6, false);
    auto MasterParam = GetVarBaseFromArgs(op_type, "MasterParam", args, 7, true);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 8, false);
    auto Moment1Out = GetVarBaseFromArgs(op_type, "Moment1Out", args, 9, false);
    auto Moment2Out = GetVarBaseFromArgs(op_type, "Moment2Out", args, 10, false);
    auto Beta1PowOut = GetVarBaseFromArgs(op_type, "Beta1PowOut", args, 11, false);
    auto Beta2PowOut = GetVarBaseFromArgs(op_type, "Beta2PowOut", args, 12, false);
    auto MasterParamOut = GetVarBaseFromArgs(op_type, "MasterParamOut", args, 13, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 14, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"Moment1Out", {Moment1Out}},{"Moment2Out", {Moment2Out}},{"Beta1PowOut", {Beta1PowOut}},{"Beta2PowOut", {Beta2PowOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"LearningRate", {LearningRate}},{"Moment1", {Moment1}},{"Moment2", {Moment2}},{"Beta1Pow", {Beta1Pow}},{"Beta2Pow", {Beta2Pow}}};
    
    if (MasterParam != nullptr) {
      ins["MasterParam"] = {MasterParam};
    }

    outs["MasterParamOut"] = {MasterParamOut};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["Moment1Out"][0],outs["Moment2Out"][0],outs["Beta1PowOut"][0],outs["Beta2PowOut"][0],outs["MasterParamOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_pow(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_pow";
    platform::RecordEvent op_type_record_event("elementwise_pow pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_prior_box(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "prior_box";
    platform::RecordEvent op_type_record_event("prior_box pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Image = GetVarBaseFromArgs(op_type, "Image", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Boxes", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Variances", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Image", {Image}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Boxes"][0],outs["Variances"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_p_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "p_norm";
    platform::RecordEvent op_type_record_event("p_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_concat";
    platform::RecordEvent op_type_record_event("c_concat pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_gate_attention(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_gate_attention";
    platform::RecordEvent op_type_record_event("fused_gate_attention pybind_imperative_func");
    
    auto Query = GetVarBaseFromArgs(op_type, "Query", args, 0, false);
    auto Key = GetVarBaseFromArgs(op_type, "Key", args, 1, true);
    auto QueryWeight = GetVarBaseFromArgs(op_type, "QueryWeight", args, 2, true);
    auto KeyWeight = GetVarBaseFromArgs(op_type, "KeyWeight", args, 3, true);
    auto ValueWeight = GetVarBaseFromArgs(op_type, "ValueWeight", args, 4, true);
    auto QKVWeight = GetVarBaseFromArgs(op_type, "QKVWeight", args, 5, true);
    auto NonbatchedBias = GetVarBaseFromArgs(op_type, "NonbatchedBias", args, 6, true);
    auto SrcMask = GetVarBaseFromArgs(op_type, "SrcMask", args, 7, false);
    auto GateWeight = GetVarBaseFromArgs(op_type, "GateWeight", args, 8, true);
    auto GateBias = GetVarBaseFromArgs(op_type, "GateBias", args, 9, true);
    auto OutLinearWeight = GetVarBaseFromArgs(op_type, "OutLinearWeight", args, 10, false);
    auto OutLinearBias = GetVarBaseFromArgs(op_type, "OutLinearBias", args, 11, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 12, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"QueryTransposeOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"KeyTransposeOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ValueTransposeOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"QKVTransposeOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SoftmaxOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"FMHAOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"GateOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Query", {Query}},{"SrcMask", {SrcMask}},{"OutLinearWeight", {OutLinearWeight}},{"OutLinearBias", {OutLinearBias}}};
    
    if (Key != nullptr) {
      ins["Key"] = {Key};
    }

    if (QueryWeight != nullptr) {
      ins["QueryWeight"] = {QueryWeight};
    }

    if (KeyWeight != nullptr) {
      ins["KeyWeight"] = {KeyWeight};
    }

    if (ValueWeight != nullptr) {
      ins["ValueWeight"] = {ValueWeight};
    }

    if (QKVWeight != nullptr) {
      ins["QKVWeight"] = {QKVWeight};
    }

    if (NonbatchedBias != nullptr) {
      ins["NonbatchedBias"] = {NonbatchedBias};
    }

    if (GateWeight != nullptr) {
      ins["GateWeight"] = {GateWeight};
    }

    if (GateBias != nullptr) {
      ins["GateBias"] = {GateBias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["QueryTransposeOut"][0],outs["KeyTransposeOut"][0],outs["ValueTransposeOut"][0],outs["QKVTransposeOut"][0],outs["SoftmaxOut"][0],outs["FMHAOut"][0],outs["GateOut"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unique_consecutive(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unique_consecutive";
    platform::RecordEvent op_type_record_event("unique_consecutive pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Index", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Counts", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Index"][0],outs["Counts"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lod_reset(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lod_reset";
    platform::RecordEvent op_type_record_event("lod_reset pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lod_reset_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lod_reset";
    platform::RecordEvent op_type_record_event("lod_reset pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pad(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pad";
    platform::RecordEvent op_type_record_event("pad pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_conv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_conv";
    platform::RecordEvent op_type_record_event("sequence_conv pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_set_value(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "set_value";
    platform::RecordEvent op_type_record_event("set_value pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto ValueTensor = GetVarBaseFromArgs(op_type, "ValueTensor", args, 1, true);
    auto StartsTensorList = GetVarBaseListFromArgs(op_type, "StartsTensorList", args, 2, true);
    auto EndsTensorList = GetVarBaseListFromArgs(op_type, "EndsTensorList", args, 3, true);
    auto StepsTensorList = GetVarBaseListFromArgs(op_type, "StepsTensorList", args, 4, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    if (ValueTensor != nullptr) {
      ins["ValueTensor"] = {ValueTensor};
    }

    if (StartsTensorList.size() != 0) {
      ins["StartsTensorList"] = StartsTensorList;
    }

    if (EndsTensorList.size() != 0) {
      ins["EndsTensorList"] = EndsTensorList;
    }

    if (StepsTensorList.size() != 0) {
      ins["StepsTensorList"] = StepsTensorList;
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_set_value_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "set_value";
    platform::RecordEvent op_type_record_event("set_value pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto ValueTensor = GetVarBaseFromArgs(op_type, "ValueTensor", args, 1, true);
    auto StartsTensorList = GetVarBaseListFromArgs(op_type, "StartsTensorList", args, 2, true);
    auto EndsTensorList = GetVarBaseListFromArgs(op_type, "EndsTensorList", args, 3, true);
    auto StepsTensorList = GetVarBaseListFromArgs(op_type, "StepsTensorList", args, 4, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      Input->IsLeaf() && !Input->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", Input->Name()));
    Input->BumpInplaceVersion();
    VLOG(3) << "Var(" << Input->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {Input}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    if (ValueTensor != nullptr) {
      ins["ValueTensor"] = {ValueTensor};
    }

    if (StartsTensorList.size() != 0) {
      ins["StartsTensorList"] = StartsTensorList;
    }

    if (EndsTensorList.size() != 0) {
      ins["EndsTensorList"] = EndsTensorList;
    }

    if (StepsTensorList.size() != 0) {
      ins["StepsTensorList"] = StepsTensorList;
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"Input", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_log10(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "log10";
    platform::RecordEvent op_type_record_event("log10 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_nms(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nms";
    platform::RecordEvent op_type_record_event("nms pybind_imperative_func");
    
    auto Boxes = GetVarBaseFromArgs(op_type, "Boxes", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"KeepBoxesIdxs", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Boxes", {Boxes}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["KeepBoxesIdxs"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bitwise_xor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bitwise_xor";
    platform::RecordEvent op_type_record_event("bitwise_xor pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_center_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "center_loss";
    platform::RecordEvent op_type_record_event("center_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto Centers = GetVarBaseFromArgs(op_type, "Centers", args, 2, false);
    auto CenterUpdateRate = GetVarBaseFromArgs(op_type, "CenterUpdateRate", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"CentersOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SampleCenterDiff", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}},{"Centers", {Centers}},{"CenterUpdateRate", {CenterUpdateRate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["CentersOut"][0],outs["SampleCenterDiff"][0],outs["Loss"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_randint(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "randint";
    platform::RecordEvent op_type_record_event("randint pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_attention_lstm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "attention_lstm";
    platform::RecordEvent op_type_record_event("attention_lstm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto C0 = GetVarBaseFromArgs(op_type, "C0", args, 1, false);
    auto AttentionWeight = GetVarBaseFromArgs(op_type, "AttentionWeight", args, 2, false);
    auto LSTMWeight = GetVarBaseFromArgs(op_type, "LSTMWeight", args, 3, false);
    auto LSTMBias = GetVarBaseFromArgs(op_type, "LSTMBias", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Hidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Cell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AttentionedX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AttentionFCOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LSTMX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LSTMOUT", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"C0", {C0}},{"AttentionWeight", {AttentionWeight}},{"LSTMWeight", {LSTMWeight}},{"LSTMBias", {LSTMBias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Hidden"][0],outs["Cell"][0],outs["AttentionedX"][0],outs["AttentionFCOut"][0],outs["LSTMX"][0],outs["LSTMOUT"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_uniform_random(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "uniform_random";
    platform::RecordEvent op_type_record_event("uniform_random pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_slice(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "slice";
    platform::RecordEvent op_type_record_event("slice pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto StartsTensor = GetVarBaseFromArgs(op_type, "StartsTensor", args, 1, true);
    auto EndsTensor = GetVarBaseFromArgs(op_type, "EndsTensor", args, 2, true);
    auto StartsTensorList = GetVarBaseListFromArgs(op_type, "StartsTensorList", args, 3, true);
    auto EndsTensorList = GetVarBaseListFromArgs(op_type, "EndsTensorList", args, 4, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    if (StartsTensor != nullptr) {
      ins["StartsTensor"] = {StartsTensor};
    }

    if (EndsTensor != nullptr) {
      ins["EndsTensor"] = {EndsTensor};
    }

    if (StartsTensorList.size() != 0) {
      ins["StartsTensorList"] = StartsTensorList;
    }

    if (EndsTensorList.size() != 0) {
      ins["EndsTensorList"] = EndsTensorList;
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dequantize(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dequantize";
    platform::RecordEvent op_type_record_event("dequantize pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_meshgrid(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "meshgrid";
    platform::RecordEvent op_type_record_event("meshgrid pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_hard_swish(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hard_swish";
    platform::RecordEvent op_type_record_event("hard_swish pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sin(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sin";
    platform::RecordEvent op_type_record_event("sin pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_mean_iou(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mean_iou";
    platform::RecordEvent op_type_record_event("mean_iou pybind_imperative_func");
    
    auto Predictions = GetVarBaseFromArgs(op_type, "Predictions", args, 0, false);
    auto Labels = GetVarBaseFromArgs(op_type, "Labels", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"OutMeanIou", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutWrong", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutCorrect", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Predictions", {Predictions}},{"Labels", {Labels}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["OutMeanIou"][0],outs["OutWrong"][0],outs["OutCorrect"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pad2d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pad2d";
    platform::RecordEvent op_type_record_event("pad2d pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_inverse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "inverse";
    platform::RecordEvent op_type_record_event("inverse pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_spectral_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "spectral_norm";
    platform::RecordEvent op_type_record_event("spectral_norm pybind_imperative_func");
    
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 0, false);
    auto U = GetVarBaseFromArgs(op_type, "U", args, 1, false);
    auto V = GetVarBaseFromArgs(op_type, "V", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Weight", {Weight}},{"U", {U}},{"V", {V}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_shuffle_channel(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "shuffle_channel";
    platform::RecordEvent op_type_record_event("shuffle_channel pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_multi_gru(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multi_gru";
    platform::RecordEvent op_type_record_event("multi_gru pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto WeightX = GetVarBaseListFromArgs(op_type, "WeightX", args, 1, false);
    auto WeightH = GetVarBaseListFromArgs(op_type, "WeightH", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Hidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"WeightX", WeightX},{"WeightH", WeightH}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Hidden"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_send_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "send_v2";
    platform::RecordEvent op_type_record_event("send_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    RETURN_PY_NONE
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_psroi_pool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "psroi_pool";
    platform::RecordEvent op_type_record_event("psroi_pool pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROIs = GetVarBaseFromArgs(op_type, "ROIs", args, 1, false);
    auto RoisNum = GetVarBaseFromArgs(op_type, "RoisNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROIs", {ROIs}}};
    
    if (RoisNum != nullptr) {
      ins["RoisNum"] = {RoisNum};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_seed(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "seed";
    platform::RecordEvent op_type_record_event("seed pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_ceil(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "ceil";
    platform::RecordEvent op_type_record_event("ceil pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_ceil_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "ceil";
    platform::RecordEvent op_type_record_event("ceil pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_eig(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "eig";
    platform::RecordEvent op_type_record_event("eig pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Eigenvalues", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Eigenvectors", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Eigenvalues"][0],outs["Eigenvectors"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reduce_min(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reduce_min";
    platform::RecordEvent op_type_record_event("reduce_min pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cos(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cos";
    platform::RecordEvent op_type_record_event("cos pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cudnn_lstm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cudnn_lstm";
    platform::RecordEvent op_type_record_event("cudnn_lstm pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto InitH = GetVarBaseFromArgs(op_type, "InitH", args, 1, false);
    auto InitC = GetVarBaseFromArgs(op_type, "InitC", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Reserve", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"StateOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LastH", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LastC", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"InitH", {InitH}},{"InitC", {InitC}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Reserve"][0],outs["StateOut"][0],outs["Out"][0],outs["LastH"][0],outs["LastC"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_random_routing(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "random_routing";
    platform::RecordEvent op_type_record_event("random_routing pybind_imperative_func");
    
    auto Prob = GetVarBaseFromArgs(op_type, "Prob", args, 0, false);
    auto TopK_Value = GetVarBaseFromArgs(op_type, "TopK_Value", args, 1, false);
    auto TopK_Idx = GetVarBaseFromArgs(op_type, "TopK_Idx", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Prob", {Prob}},{"TopK_Value", {TopK_Value}},{"TopK_Idx", {TopK_Idx}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_random_routing_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "random_routing";
    platform::RecordEvent op_type_record_event("random_routing pybind_imperative_func");
    
    auto Prob = GetVarBaseFromArgs(op_type, "Prob", args, 0, false);
    auto TopK_Value = GetVarBaseFromArgs(op_type, "TopK_Value", args, 1, false);
    auto TopK_Idx = GetVarBaseFromArgs(op_type, "TopK_Idx", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      TopK_Idx->IsLeaf() && !TopK_Idx->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", TopK_Idx->Name()));
    TopK_Idx->BumpInplaceVersion();
    VLOG(3) << "Var(" << TopK_Idx->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {TopK_Idx}}};
    imperative::NameVarBaseMap ins = {{"Prob", {Prob}},{"TopK_Value", {TopK_Value}},{"TopK_Idx", {TopK_Idx}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"TopK_Idx", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reduce_sum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reduce_sum";
    platform::RecordEvent op_type_record_event("reduce_sum pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_digamma(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "digamma";
    platform::RecordEvent op_type_record_event("digamma pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_quantize_linear(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "quantize_linear";
    platform::RecordEvent op_type_record_event("quantize_linear pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto ZeroPoint = GetVarBaseFromArgs(op_type, "ZeroPoint", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"ZeroPoint", {ZeroPoint}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_assign_value(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "assign_value";
    platform::RecordEvent op_type_record_event("assign_value pybind_imperative_func");
    
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_increment(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "increment";
    platform::RecordEvent op_type_record_event("increment pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logspace(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logspace";
    platform::RecordEvent op_type_record_event("logspace pybind_imperative_func");
    
    auto Start = GetVarBaseFromArgs(op_type, "Start", args, 0, false);
    auto Stop = GetVarBaseFromArgs(op_type, "Stop", args, 1, false);
    auto Num = GetVarBaseFromArgs(op_type, "Num", args, 2, false);
    auto Base = GetVarBaseFromArgs(op_type, "Base", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Start", {Start}},{"Stop", {Stop}},{"Num", {Num}},{"Base", {Base}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_tdm_sampler(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "tdm_sampler";
    platform::RecordEvent op_type_record_event("tdm_sampler pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Travel = GetVarBaseFromArgs(op_type, "Travel", args, 1, false);
    auto Layer = GetVarBaseFromArgs(op_type, "Layer", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Travel", {Travel}},{"Layer", {Layer}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Mask"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_softmax_mask(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_softmax_mask";
    platform::RecordEvent op_type_record_event("fused_softmax_mask pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Mask = GetVarBaseFromArgs(op_type, "Mask", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Mask", {Mask}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_reverse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_reverse";
    platform::RecordEvent op_type_record_event("sequence_reverse pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_eigvalsh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "eigvalsh";
    platform::RecordEvent op_type_record_event("eigvalsh pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Eigenvalues", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Eigenvectors", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Eigenvalues"][0],outs["Eigenvectors"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_diagonal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "diagonal";
    platform::RecordEvent op_type_record_event("diagonal pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_trunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "trunc";
    platform::RecordEvent op_type_record_event("trunc pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_log2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "log2";
    platform::RecordEvent op_type_record_event("log2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_marker(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "marker";
    platform::RecordEvent op_type_record_event("marker pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    RETURN_PY_NONE
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_tanh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "tanh";
    platform::RecordEvent op_type_record_event("tanh pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_tanh_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "tanh";
    platform::RecordEvent op_type_record_event("tanh pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_yolov3_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "yolov3_loss";
    platform::RecordEvent op_type_record_event("yolov3_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto GTBox = GetVarBaseFromArgs(op_type, "GTBox", args, 1, false);
    auto GTLabel = GetVarBaseFromArgs(op_type, "GTLabel", args, 2, false);
    auto GTScore = GetVarBaseFromArgs(op_type, "GTScore", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ObjectnessMask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"GTMatchMask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"GTBox", {GTBox}},{"GTLabel", {GTLabel}}};
    
    if (GTScore != nullptr) {
      ins["GTScore"] = {GTScore};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Loss"][0],outs["ObjectnessMask"][0],outs["GTMatchMask"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_graph_send_recv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "graph_send_recv";
    platform::RecordEvent op_type_record_event("graph_send_recv pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Src_index = GetVarBaseFromArgs(op_type, "Src_index", args, 1, false);
    auto Dst_index = GetVarBaseFromArgs(op_type, "Dst_index", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Dst_count", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Src_index", {Src_index}},{"Dst_index", {Dst_index}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Dst_count"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_accuracy(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "accuracy";
    platform::RecordEvent op_type_record_event("accuracy pybind_imperative_func");
    
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 0, false);
    auto Indices = GetVarBaseFromArgs(op_type, "Indices", args, 1, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 2, false);
    auto Correct = GetVarBaseFromArgs(op_type, "Correct", args, 3, false);
    auto Total = GetVarBaseFromArgs(op_type, "Total", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Accuracy", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Correct", {Correct}},{"Total", {Total}}};
    imperative::NameVarBaseMap ins = {{"Out", {Out}},{"Indices", {Indices}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Accuracy"][0],outs["Correct"][0],outs["Total"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_atan(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "atan";
    platform::RecordEvent op_type_record_event("atan pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_less_than(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "less_than";
    platform::RecordEvent op_type_record_event("less_than pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reduce_amax(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reduce_amax";
    platform::RecordEvent op_type_record_event("reduce_amax pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unsqueeze(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unsqueeze";
    platform::RecordEvent op_type_record_event("unsqueeze pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_crf_decoding(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "crf_decoding";
    platform::RecordEvent op_type_record_event("crf_decoding pybind_imperative_func");
    
    auto Emission = GetVarBaseFromArgs(op_type, "Emission", args, 0, false);
    auto Transition = GetVarBaseFromArgs(op_type, "Transition", args, 1, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 2, true);
    auto Length = GetVarBaseFromArgs(op_type, "Length", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ViterbiPath", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Emission", {Emission}},{"Transition", {Transition}}};
    
    if (Label != nullptr) {
      ins["Label"] = {Label};
    }

    if (Length != nullptr) {
      ins["Length"] = {Length};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["ViterbiPath"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_global_gather(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "global_gather";
    platform::RecordEvent op_type_record_event("global_gather pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto local_count = GetVarBaseFromArgs(op_type, "local_count", args, 1, false);
    auto global_count = GetVarBaseFromArgs(op_type, "global_count", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"local_count", {local_count}},{"global_count", {global_count}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_merged_adam(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "merged_adam";
    platform::RecordEvent op_type_record_event("merged_adam pybind_imperative_func");
    
    auto Param = GetVarBaseListFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseListFromArgs(op_type, "Grad", args, 1, false);
    auto LearningRate = GetVarBaseListFromArgs(op_type, "LearningRate", args, 2, false);
    auto Moment1 = GetVarBaseListFromArgs(op_type, "Moment1", args, 3, false);
    auto Moment2 = GetVarBaseListFromArgs(op_type, "Moment2", args, 4, false);
    auto Beta1Pow = GetVarBaseListFromArgs(op_type, "Beta1Pow", args, 5, false);
    auto Beta2Pow = GetVarBaseListFromArgs(op_type, "Beta2Pow", args, 6, false);
    auto MasterParam = GetVarBaseListFromArgs(op_type, "MasterParam", args, 7, true);
    auto ParamOut = GetVarBaseListFromArgs(op_type, "ParamOut", args, 8, false);
    auto Moment1Out = GetVarBaseListFromArgs(op_type, "Moment1Out", args, 9, false);
    auto Moment2Out = GetVarBaseListFromArgs(op_type, "Moment2Out", args, 10, false);
    auto Beta1PowOut = GetVarBaseListFromArgs(op_type, "Beta1PowOut", args, 11, false);
    auto Beta2PowOut = GetVarBaseListFromArgs(op_type, "Beta2PowOut", args, 12, false);
    auto MasterParamOut = GetVarBaseListFromArgs(op_type, "MasterParamOut", args, 13, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 14, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", ParamOut},{"Moment1Out", Moment1Out},{"Moment2Out", Moment2Out},{"Beta1PowOut", Beta1PowOut},{"Beta2PowOut", Beta2PowOut}};
    imperative::NameVarBaseMap ins = {{"Param", Param},{"Grad", Grad},{"LearningRate", LearningRate},{"Moment1", Moment1},{"Moment2", Moment2},{"Beta1Pow", Beta1Pow},{"Beta2Pow", Beta2Pow}};
    
    if (MasterParam.size() != 0) {
      ins["MasterParam"] = MasterParam;
    }

    outs["MasterParamOut"] = MasterParamOut;

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"],outs["Moment1Out"],outs["Moment2Out"],outs["Beta1PowOut"],outs["Beta2PowOut"],outs["MasterParamOut"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lerp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lerp";
    platform::RecordEvent op_type_record_event("lerp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}},{"Weight", {Weight}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lerp_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lerp";
    platform::RecordEvent op_type_record_event("lerp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}},{"Weight", {Weight}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_allreduce_prod(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_allreduce_prod";
    platform::RecordEvent op_type_record_event("c_allreduce_prod pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_allreduce_prod_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_allreduce_prod";
    platform::RecordEvent op_type_record_event("c_allreduce_prod pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_log_softmax(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "log_softmax";
    platform::RecordEvent op_type_record_event("log_softmax pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_ftrl(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "ftrl";
    platform::RecordEvent op_type_record_event("ftrl pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto SquaredAccumulator = GetVarBaseFromArgs(op_type, "SquaredAccumulator", args, 1, false);
    auto LinearAccumulator = GetVarBaseFromArgs(op_type, "LinearAccumulator", args, 2, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 3, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 4, false);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 5, false);
    auto SquaredAccumOut = GetVarBaseFromArgs(op_type, "SquaredAccumOut", args, 6, false);
    auto LinearAccumOut = GetVarBaseFromArgs(op_type, "LinearAccumOut", args, 7, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"SquaredAccumOut", {SquaredAccumOut}},{"LinearAccumOut", {LinearAccumOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"SquaredAccumulator", {SquaredAccumulator}},{"LinearAccumulator", {LinearAccumulator}},{"Grad", {Grad}},{"LearningRate", {LearningRate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["SquaredAccumOut"][0],outs["LinearAccumOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_matrix_nms(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "matrix_nms";
    platform::RecordEvent op_type_record_event("matrix_nms pybind_imperative_func");
    
    auto BBoxes = GetVarBaseFromArgs(op_type, "BBoxes", args, 0, false);
    auto Scores = GetVarBaseFromArgs(op_type, "Scores", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Index", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RoisNum", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"BBoxes", {BBoxes}},{"Scores", {Scores}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Index"][0],outs["RoisNum"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_top_k_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "top_k_v2";
    platform::RecordEvent op_type_record_event("top_k_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Indices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Indices"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cast(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cast";
    platform::RecordEvent op_type_record_event("cast pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_tanh_shrink(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "tanh_shrink";
    platform::RecordEvent op_type_record_event("tanh_shrink pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_hard_shrink(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hard_shrink";
    platform::RecordEvent op_type_record_event("hard_shrink pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logit(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logit";
    platform::RecordEvent op_type_record_event("logit pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_multiclass_nms(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multiclass_nms";
    platform::RecordEvent op_type_record_event("multiclass_nms pybind_imperative_func");
    
    auto BBoxes = GetVarBaseFromArgs(op_type, "BBoxes", args, 0, false);
    auto Scores = GetVarBaseFromArgs(op_type, "Scores", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"BBoxes", {BBoxes}},{"Scores", {Scores}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_broadcast(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_broadcast";
    platform::RecordEvent op_type_record_event("c_broadcast pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_transpose_flatten_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_transpose_flatten_concat";
    platform::RecordEvent op_type_record_event("fusion_transpose_flatten_concat pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_unpad(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_unpad";
    platform::RecordEvent op_type_record_event("sequence_unpad pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Length = GetVarBaseFromArgs(op_type, "Length", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Length", {Length}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_elemwise_add_activation(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_elemwise_add_activation";
    platform::RecordEvent op_type_record_event("fused_elemwise_add_activation pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"IntermediateOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["IntermediateOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pull_sparse_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pull_sparse_v2";
    platform::RecordEvent op_type_record_event("pull_sparse_v2 pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    auto W = GetVarBaseListFromArgs(op_type, "W", args, 1, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"Ids", Ids},{"W", W}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_einsum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "einsum";
    platform::RecordEvent op_type_record_event("einsum pybind_imperative_func");
    
    auto Operands = GetVarBaseListFromArgs(op_type, "Operands", args, 0, false);
    auto InnerCacheNum = GetUnsignedLongFromArgs(op_type, "InnerCacheNum", args, 1, false);
    auto XShapeNum = GetUnsignedLongFromArgs(op_type, "XShapeNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"InnerCache", ConstructDuplicableOutput(InnerCacheNum)},{"XShape", ConstructDuplicableOutput(XShapeNum)}};
    imperative::NameVarBaseMap ins = {{"Operands", Operands}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["InnerCache"],outs["XShape"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_frobenius_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "frobenius_norm";
    platform::RecordEvent op_type_record_event("frobenius_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_crop(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "crop";
    platform::RecordEvent op_type_record_event("crop pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, true);
    auto Offsets = GetVarBaseFromArgs(op_type, "Offsets", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Y != nullptr) {
      ins["Y"] = {Y};
    }

    if (Offsets != nullptr) {
      ins["Offsets"] = {Offsets};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cross_entropy2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cross_entropy2";
    platform::RecordEvent op_type_record_event("cross_entropy2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MatchX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["XShape"][0],outs["MatchX"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_skip_layernorm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "skip_layernorm";
    platform::RecordEvent op_type_record_event("skip_layernorm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}},{"Scale", {Scale}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_tdm_child(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "tdm_child";
    platform::RecordEvent op_type_record_event("tdm_child pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto TreeInfo = GetVarBaseFromArgs(op_type, "TreeInfo", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Child", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LeafMask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"TreeInfo", {TreeInfo}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Child"][0],outs["LeafMask"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_embedding_seq_pool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_embedding_seq_pool";
    platform::RecordEvent op_type_record_event("fused_embedding_seq_pool pybind_imperative_func");
    
    auto W = GetVarBaseFromArgs(op_type, "W", args, 0, false);
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"W", {W}},{"Ids", {Ids}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_kthvalue(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "kthvalue";
    platform::RecordEvent op_type_record_event("kthvalue pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Indices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Indices"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_erf(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "erf";
    platform::RecordEvent op_type_record_event("erf pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_yolo_box_post(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "yolo_box_post";
    platform::RecordEvent op_type_record_event("yolo_box_post pybind_imperative_func");
    
    auto Boxes0 = GetVarBaseFromArgs(op_type, "Boxes0", args, 0, false);
    auto Boxes1 = GetVarBaseFromArgs(op_type, "Boxes1", args, 1, false);
    auto Boxes2 = GetVarBaseFromArgs(op_type, "Boxes2", args, 2, false);
    auto ImageShape = GetVarBaseFromArgs(op_type, "ImageShape", args, 3, false);
    auto ImageScale = GetVarBaseFromArgs(op_type, "ImageScale", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"NmsRoisNum", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Boxes0", {Boxes0}},{"Boxes1", {Boxes1}},{"Boxes2", {Boxes2}},{"ImageShape", {ImageShape}},{"ImageScale", {ImageScale}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["NmsRoisNum"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conv2d_inception_fusion(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv2d_inception_fusion";
    platform::RecordEvent op_type_record_event("conv2d_inception_fusion pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Filter = GetVarBaseListFromArgs(op_type, "Filter", args, 1, false);
    auto Bias = GetVarBaseListFromArgs(op_type, "Bias", args, 2, false);
    auto TempOutputNum = GetUnsignedLongFromArgs(op_type, "TempOutputNum", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TempOutput", ConstructDuplicableOutput(TempOutputNum)}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Filter", Filter},{"Bias", Bias}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Output"][0],outs["TempOutput"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logsumexp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logsumexp";
    platform::RecordEvent op_type_record_event("logsumexp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_trilinear_interp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "trilinear_interp";
    platform::RecordEvent op_type_record_event("trilinear_interp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto OutSize = GetVarBaseFromArgs(op_type, "OutSize", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (OutSize != nullptr) {
      ins["OutSize"] = {OutSize};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_seqpool_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_seqpool_concat";
    platform::RecordEvent op_type_record_event("fusion_seqpool_concat pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_alloc_float_status(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "alloc_float_status";
    platform::RecordEvent op_type_record_event("alloc_float_status pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"FloatStatus", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["FloatStatus"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_concat";
    platform::RecordEvent op_type_record_event("sequence_concat pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_seqpool_cvm_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_seqpool_cvm_concat";
    platform::RecordEvent op_type_record_event("fusion_seqpool_cvm_concat pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto CVM = GetVarBaseFromArgs(op_type, "CVM", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X},{"CVM", {CVM}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unpool3d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unpool3d";
    platform::RecordEvent op_type_record_event("unpool3d pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Indices = GetVarBaseFromArgs(op_type, "Indices", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Indices", {Indices}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_similarity_focus(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "similarity_focus";
    platform::RecordEvent op_type_record_event("similarity_focus pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_allreduce_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_allreduce_max";
    platform::RecordEvent op_type_record_event("c_allreduce_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_allreduce_max_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_allreduce_max";
    platform::RecordEvent op_type_record_event("c_allreduce_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_argsort(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "argsort";
    platform::RecordEvent op_type_record_event("argsort pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Indices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Indices"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_expand(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_expand";
    platform::RecordEvent op_type_record_event("sequence_expand pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_bn_add_activation(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_bn_add_activation";
    platform::RecordEvent op_type_record_event("fused_bn_add_activation pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Z = GetVarBaseFromArgs(op_type, "Z", args, 1, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MeanOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"VarianceOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReserveSpace", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Z", {Z}},{"Scale", {Scale}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["MeanOut"][0],outs["VarianceOut"][0],outs["SavedMean"][0],outs["SavedVariance"][0],outs["ReserveSpace"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sgd(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sgd";
    platform::RecordEvent op_type_record_event("sgd pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 1, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 2, false);
    auto MasterParam = GetVarBaseFromArgs(op_type, "MasterParam", args, 3, true);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 4, false);
    auto MasterParamOut = GetVarBaseFromArgs(op_type, "MasterParamOut", args, 5, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"LearningRate", {LearningRate}},{"Grad", {Grad}}};
    
    if (MasterParam != nullptr) {
      ins["MasterParam"] = {MasterParam};
    }

    outs["MasterParamOut"] = {MasterParamOut};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["MasterParamOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_exponential(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "exponential";
    platform::RecordEvent op_type_record_event("exponential pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_exponential_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "exponential";
    platform::RecordEvent op_type_record_event("exponential pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bilinear_interp_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bilinear_interp_v2";
    platform::RecordEvent op_type_record_event("bilinear_interp_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_atanh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "atanh";
    platform::RecordEvent op_type_record_event("atanh pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_clip(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "clip";
    platform::RecordEvent op_type_record_event("clip pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_clip_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "clip";
    platform::RecordEvent op_type_record_event("clip pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_deformable_conv_v1(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "deformable_conv_v1";
    platform::RecordEvent op_type_record_event("deformable_conv_v1 pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Offset = GetVarBaseFromArgs(op_type, "Offset", args, 1, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Offset", {Offset}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_hinge_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hinge_loss";
    platform::RecordEvent op_type_record_event("hinge_loss pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Labels = GetVarBaseFromArgs(op_type, "Labels", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Labels", {Labels}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Loss"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_determinant(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "determinant";
    platform::RecordEvent op_type_record_event("determinant pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conv2d_transpose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv2d_transpose";
    platform::RecordEvent op_type_record_event("conv2d_transpose pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_memcpy_d2h(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "memcpy_d2h";
    platform::RecordEvent op_type_record_event("memcpy_d2h pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_softsign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "softsign";
    platform::RecordEvent op_type_record_event("softsign pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fake_quantize_dequantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_quantize_dequantize_abs_max";
    platform::RecordEvent op_type_record_event("fake_quantize_dequantize_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    auto OutScale = GetVarBaseFromArgs(op_type, "OutScale", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}},{"OutScale", {OutScale}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutScale"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_broadcast_tensors(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "broadcast_tensors";
    platform::RecordEvent op_type_record_event("broadcast_tensors pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cholesky_solve(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cholesky_solve";
    platform::RecordEvent op_type_record_event("cholesky_solve pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_grid_sampler(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "grid_sampler";
    platform::RecordEvent op_type_record_event("grid_sampler pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Grid = GetVarBaseFromArgs(op_type, "Grid", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Grid", {Grid}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fft_c2r(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fft_c2r";
    platform::RecordEvent op_type_record_event("fft_c2r pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pyramid_hash(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pyramid_hash";
    platform::RecordEvent op_type_record_event("pyramid_hash pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 1, false);
    auto WhiteList = GetVarBaseFromArgs(op_type, "WhiteList", args, 2, false);
    auto BlackList = GetVarBaseFromArgs(op_type, "BlackList", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"DropPos", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"X_Temp_Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"W", {W}},{"WhiteList", {WhiteList}},{"BlackList", {BlackList}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["DropPos"][0],outs["X_Temp_Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fake_quantize_dequantize_moving_average_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_quantize_dequantize_moving_average_abs_max";
    platform::RecordEvent op_type_record_event("fake_quantize_dequantize_moving_average_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto InScale = GetVarBaseFromArgs(op_type, "InScale", args, 1, false);
    auto InAccum = GetVarBaseFromArgs(op_type, "InAccum", args, 2, true);
    auto InState = GetVarBaseFromArgs(op_type, "InState", args, 3, true);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 4, false);
    auto OutScale = GetVarBaseFromArgs(op_type, "OutScale", args, 5, false);
    auto OutState = GetVarBaseFromArgs(op_type, "OutState", args, 6, true);
    auto OutAccum = GetVarBaseFromArgs(op_type, "OutAccum", args, 7, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}},{"OutScale", {OutScale}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"InScale", {InScale}}};
    
    if (InAccum != nullptr) {
      ins["InAccum"] = {InAccum};
    }

    if (InState != nullptr) {
      ins["InState"] = {InState};
    }

    outs["OutState"] = {OutState};

    outs["OutAccum"] = {OutAccum};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutScale"][0],outs["OutState"][0],outs["OutAccum"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_multi_dot(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multi_dot";
    platform::RecordEvent op_type_record_event("multi_dot pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_pool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_pool";
    platform::RecordEvent op_type_record_event("sequence_pool pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MaxIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["MaxIndex"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_broadcast(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "broadcast";
    platform::RecordEvent op_type_record_event("broadcast pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_transpose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "transpose";
    platform::RecordEvent op_type_record_event("transpose pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_top_k(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "top_k";
    platform::RecordEvent op_type_record_event("top_k pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Indices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Indices"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_renorm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "renorm";
    platform::RecordEvent op_type_record_event("renorm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pixel_unshuffle(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pixel_unshuffle";
    platform::RecordEvent op_type_record_event("pixel_unshuffle pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_take_along_axis(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "take_along_axis";
    platform::RecordEvent op_type_record_event("take_along_axis pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Result", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Index", {Index}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Result"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dist(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dist";
    platform::RecordEvent op_type_record_event("dist pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_affine_grid(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "affine_grid";
    platform::RecordEvent op_type_record_event("affine_grid pybind_imperative_func");
    
    auto Theta = GetVarBaseFromArgs(op_type, "Theta", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Theta", {Theta}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_gaussian_random_batch_size_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gaussian_random_batch_size_like";
    platform::RecordEvent op_type_record_event("gaussian_random_batch_size_like pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fake_channel_wise_dequantize_max_abs(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_channel_wise_dequantize_max_abs";
    platform::RecordEvent op_type_record_event("fake_channel_wise_dequantize_max_abs pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scales = GetVarBaseListFromArgs(op_type, "Scales", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scales", Scales}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reciprocal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reciprocal";
    platform::RecordEvent op_type_record_event("reciprocal pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reciprocal_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reciprocal";
    platform::RecordEvent op_type_record_event("reciprocal pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_mask(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_mask";
    platform::RecordEvent op_type_record_event("sequence_mask pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto MaxLenTensor = GetVarBaseFromArgs(op_type, "MaxLenTensor", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (MaxLenTensor != nullptr) {
      ins["MaxLenTensor"] = {MaxLenTensor};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_prune_gate_by_capacity(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "prune_gate_by_capacity";
    platform::RecordEvent op_type_record_event("prune_gate_by_capacity pybind_imperative_func");
    
    auto GateIdx = GetVarBaseFromArgs(op_type, "GateIdx", args, 0, false);
    auto ExpertCount = GetVarBaseFromArgs(op_type, "ExpertCount", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"NewGateIdx", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"GateIdx", {GateIdx}},{"ExpertCount", {ExpertCount}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["NewGateIdx"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_diagonal_tensor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_diagonal_tensor";
    platform::RecordEvent op_type_record_event("fill_diagonal_tensor pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_diagonal_tensor_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_diagonal_tensor";
    platform::RecordEvent op_type_record_event("fill_diagonal_tensor pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_abs(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "abs";
    platform::RecordEvent op_type_record_event("abs pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_partial_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "partial_concat";
    platform::RecordEvent op_type_record_event("partial_concat pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elu";
    platform::RecordEvent op_type_record_event("elu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elu_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elu";
    platform::RecordEvent op_type_record_event("elu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_index_select(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "index_select";
    platform::RecordEvent op_type_record_event("index_select pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Index", {Index}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_row_conv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "row_conv";
    platform::RecordEvent op_type_record_event("row_conv pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cross(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cross";
    platform::RecordEvent op_type_record_event("cross pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_mul(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_mul";
    platform::RecordEvent op_type_record_event("elementwise_mul pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_decayed_adagrad(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "decayed_adagrad";
    platform::RecordEvent op_type_record_event("decayed_adagrad pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto Moment = GetVarBaseFromArgs(op_type, "Moment", args, 2, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 3, false);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 4, false);
    auto MomentOut = GetVarBaseFromArgs(op_type, "MomentOut", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"MomentOut", {MomentOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"Moment", {Moment}},{"LearningRate", {LearningRate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["MomentOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bipartite_match(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bipartite_match";
    platform::RecordEvent op_type_record_event("bipartite_match pybind_imperative_func");
    
    auto DistMat = GetVarBaseFromArgs(op_type, "DistMat", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ColToRowMatchIndices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ColToRowMatchDist", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"DistMat", {DistMat}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ColToRowMatchIndices"][0],outs["ColToRowMatchDist"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_run_program(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "run_program";
    platform::RecordEvent op_type_record_event("run_program pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto Params = GetVarBaseListFromArgs(op_type, "Params", args, 1, true);
    auto Out = GetVarBaseListFromArgs(op_type, "Out", args, 2, false);
    auto OutScope = GetVarBaseFromArgs(op_type, "OutScope", args, 3, false);
    auto DOut = GetVarBaseListFromArgs(op_type, "DOut", args, 4, true);
    auto CUDAGraph = GetVarBaseFromArgs(op_type, "CUDAGraph", args, 5, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", Out},{"OutScope", {OutScope}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    if (Params.size() != 0) {
      ins["Params"] = Params;
    }

    outs["DOut"] = DOut;

    outs["CUDAGraph"] = {CUDAGraph};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"],outs["OutScope"][0],outs["DOut"],outs["CUDAGraph"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fake_quantize_moving_average_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_quantize_moving_average_abs_max";
    platform::RecordEvent op_type_record_event("fake_quantize_moving_average_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto InScale = GetVarBaseFromArgs(op_type, "InScale", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutScale", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"InScale", {InScale}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutScale"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_multi_transformer_int8(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_multi_transformer_int8";
    platform::RecordEvent op_type_record_event("fused_multi_transformer_int8 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto LnScale = GetVarBaseListFromArgs(op_type, "LnScale", args, 1, false);
    auto LnBias = GetVarBaseListFromArgs(op_type, "LnBias", args, 2, false);
    auto QKVW = GetVarBaseListFromArgs(op_type, "QKVW", args, 3, false);
    auto OutLinearW = GetVarBaseListFromArgs(op_type, "OutLinearW", args, 4, false);
    auto FFNLnScale = GetVarBaseListFromArgs(op_type, "FFNLnScale", args, 5, false);
    auto FFNLnBias = GetVarBaseListFromArgs(op_type, "FFNLnBias", args, 6, false);
    auto FFN1Weight = GetVarBaseListFromArgs(op_type, "FFN1Weight", args, 7, false);
    auto FFN2Weight = GetVarBaseListFromArgs(op_type, "FFN2Weight", args, 8, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 9, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"LnScale", LnScale},{"LnBias", LnBias},{"QKVW", QKVW},{"OutLinearW", OutLinearW},{"FFNLnScale", FFNLnScale},{"FFNLnBias", FFNLnBias},{"FFN1Weight", FFN1Weight},{"FFN2Weight", FFN2Weight}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_mine_hard_examples(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mine_hard_examples";
    platform::RecordEvent op_type_record_event("mine_hard_examples pybind_imperative_func");
    
    auto ClsLoss = GetVarBaseFromArgs(op_type, "ClsLoss", args, 0, false);
    auto MatchIndices = GetVarBaseFromArgs(op_type, "MatchIndices", args, 1, false);
    auto MatchDist = GetVarBaseFromArgs(op_type, "MatchDist", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"NegIndices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"UpdatedMatchIndices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"ClsLoss", {ClsLoss}},{"MatchIndices", {MatchIndices}},{"MatchDist", {MatchDist}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["NegIndices"][0],outs["UpdatedMatchIndices"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_target_assign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "target_assign";
    platform::RecordEvent op_type_record_event("target_assign pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto MatchIndices = GetVarBaseFromArgs(op_type, "MatchIndices", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutWeight", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"MatchIndices", {MatchIndices}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutWeight"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lstm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lstm";
    platform::RecordEvent op_type_record_event("lstm pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Hidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Cell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchGate", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchCellPreAct", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Weight", {Weight}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Hidden"][0],outs["Cell"][0],outs["BatchGate"][0],outs["BatchCellPreAct"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_assign_pos(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "assign_pos";
    platform::RecordEvent op_type_record_event("assign_pos pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto cum_count = GetVarBaseFromArgs(op_type, "cum_count", args, 1, false);
    auto eff_num_len = GetVarBaseFromArgs(op_type, "eff_num_len", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"cum_count", {cum_count}},{"eff_num_len", {eff_num_len}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_truncated_gaussian_random(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "truncated_gaussian_random";
    platform::RecordEvent op_type_record_event("truncated_gaussian_random pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_match_matrix_tensor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "match_matrix_tensor";
    platform::RecordEvent op_type_record_event("match_matrix_tensor pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Tmp", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}},{"W", {W}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Tmp"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_div(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_div";
    platform::RecordEvent op_type_record_event("elementwise_div pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_kldiv_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "kldiv_loss";
    platform::RecordEvent op_type_record_event("kldiv_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Target = GetVarBaseFromArgs(op_type, "Target", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Target", {Target}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Loss"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cumsum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cumsum";
    platform::RecordEvent op_type_record_event("cumsum pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sum";
    platform::RecordEvent op_type_record_event("sum pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sum_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sum";
    platform::RecordEvent op_type_record_event("sum pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X[0]->IsLeaf() && !X[0]->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X[0]->Name()));
    X[0]->BumpInplaceVersion();
    VLOG(3) << "Var(" << X[0]->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X[0]}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_proximal_adagrad(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "proximal_adagrad";
    platform::RecordEvent op_type_record_event("proximal_adagrad pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Moment = GetVarBaseFromArgs(op_type, "Moment", args, 1, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 2, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MomentOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Moment", {Moment}},{"Grad", {Grad}},{"LearningRate", {LearningRate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["MomentOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_update_loss_scaling(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "update_loss_scaling";
    platform::RecordEvent op_type_record_event("update_loss_scaling pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto FoundInfinite = GetVarBaseFromArgs(op_type, "FoundInfinite", args, 1, false);
    auto PrevLossScaling = GetVarBaseFromArgs(op_type, "PrevLossScaling", args, 2, false);
    auto InGoodSteps = GetVarBaseFromArgs(op_type, "InGoodSteps", args, 3, false);
    auto InBadSteps = GetVarBaseFromArgs(op_type, "InBadSteps", args, 4, false);
    auto Out = GetVarBaseListFromArgs(op_type, "Out", args, 5, false);
    auto LossScaling = GetVarBaseFromArgs(op_type, "LossScaling", args, 6, false);
    auto OutGoodSteps = GetVarBaseFromArgs(op_type, "OutGoodSteps", args, 7, false);
    auto OutBadSteps = GetVarBaseFromArgs(op_type, "OutBadSteps", args, 8, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 9, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", Out},{"LossScaling", {LossScaling}},{"OutGoodSteps", {OutGoodSteps}},{"OutBadSteps", {OutBadSteps}}};
    imperative::NameVarBaseMap ins = {{"X", X},{"FoundInfinite", {FoundInfinite}},{"PrevLossScaling", {PrevLossScaling}},{"InGoodSteps", {InGoodSteps}},{"InBadSteps", {InBadSteps}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"],outs["LossScaling"][0],outs["OutGoodSteps"][0],outs["OutBadSteps"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_shard_index(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "shard_index";
    platform::RecordEvent op_type_record_event("shard_index pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_selu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "selu";
    platform::RecordEvent op_type_record_event("selu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_gumbel_softmax(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gumbel_softmax";
    platform::RecordEvent op_type_record_event("gumbel_softmax pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_mean(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mean";
    platform::RecordEvent op_type_record_event("mean pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_pad(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_pad";
    platform::RecordEvent op_type_record_event("sequence_pad pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto PadValue = GetVarBaseFromArgs(op_type, "PadValue", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Length", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"PadValue", {PadValue}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Length"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_tree_conv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "tree_conv";
    platform::RecordEvent op_type_record_event("tree_conv pybind_imperative_func");
    
    auto NodesVector = GetVarBaseFromArgs(op_type, "NodesVector", args, 0, false);
    auto EdgeSet = GetVarBaseFromArgs(op_type, "EdgeSet", args, 1, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"NodesVector", {NodesVector}},{"EdgeSet", {EdgeSet}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_assign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "assign";
    platform::RecordEvent op_type_record_event("assign pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, true);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {};
    
    if (X != nullptr) {
      ins["X"] = {X};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_assign_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "assign";
    platform::RecordEvent op_type_record_event("assign pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, true);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {};
    
    if (X != nullptr) {
      ins["X"] = {X};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_flatten_contiguous_range(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "flatten_contiguous_range";
    platform::RecordEvent op_type_record_event("flatten_contiguous_range pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (ins.count("X") && outs.count("Out")) {
      HandleViewBetweenInputAndOutput(ins["X"][0], outs["Out"][0]);
    }
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_flatten_contiguous_range_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "flatten_contiguous_range";
    platform::RecordEvent op_type_record_event("flatten_contiguous_range pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_tril_triu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "tril_triu";
    platform::RecordEvent op_type_record_event("tril_triu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_celu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "celu";
    platform::RecordEvent op_type_record_event("celu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_celu_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "celu";
    platform::RecordEvent op_type_record_event("celu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reduce_mean(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reduce_mean";
    platform::RecordEvent op_type_record_event("reduce_mean pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_brelu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "brelu";
    platform::RecordEvent op_type_record_event("brelu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sinh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sinh";
    platform::RecordEvent op_type_record_event("sinh pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_rank_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rank_loss";
    platform::RecordEvent op_type_record_event("rank_loss pybind_imperative_func");
    
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 0, false);
    auto Left = GetVarBaseFromArgs(op_type, "Left", args, 1, false);
    auto Right = GetVarBaseFromArgs(op_type, "Right", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Label", {Label}},{"Left", {Left}},{"Right", {Right}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reduce_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reduce_max";
    platform::RecordEvent op_type_record_event("reduce_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_gru(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_gru";
    platform::RecordEvent op_type_record_event("fusion_gru pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto WeightX = GetVarBaseFromArgs(op_type, "WeightX", args, 1, false);
    auto WeightH = GetVarBaseFromArgs(op_type, "WeightH", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ReorderedH0", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchedInput", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchedOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Hidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"WeightX", {WeightX}},{"WeightH", {WeightH}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ReorderedH0"][0],outs["XX"][0],outs["BatchedInput"][0],outs["BatchedOut"][0],outs["Hidden"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_zeros_like2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_zeros_like2";
    platform::RecordEvent op_type_record_event("fill_zeros_like2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_expm1(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "expm1";
    platform::RecordEvent op_type_record_event("expm1 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_expm1_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "expm1";
    platform::RecordEvent op_type_record_event("expm1 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_squared_l2_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "squared_l2_norm";
    platform::RecordEvent op_type_record_event("squared_l2_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_sub(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_sub";
    platform::RecordEvent op_type_record_event("elementwise_sub pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_sub_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_sub";
    platform::RecordEvent op_type_record_event("elementwise_sub pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_margin_rank_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "margin_rank_loss";
    platform::RecordEvent op_type_record_event("margin_rank_loss pybind_imperative_func");
    
    auto X1 = GetVarBaseFromArgs(op_type, "X1", args, 0, false);
    auto X2 = GetVarBaseFromArgs(op_type, "X2", args, 1, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Activated", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X1", {X1}},{"X2", {X2}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Activated"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_faster_tokenizer(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "faster_tokenizer";
    platform::RecordEvent op_type_record_event("faster_tokenizer pybind_imperative_func");
    
    auto Vocab = GetVarBaseFromArgs(op_type, "Vocab", args, 0, false);
    auto Text = GetVarBaseFromArgs(op_type, "Text", args, 1, false);
    auto TextPair = GetVarBaseFromArgs(op_type, "TextPair", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"InputIds", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SegmentIds", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Vocab", {Vocab}},{"Text", {Text}}};
    
    if (TextPair != nullptr) {
      ins["TextPair"] = {TextPair};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["InputIds"][0],outs["SegmentIds"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_reduce_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_reduce_max";
    platform::RecordEvent op_type_record_event("c_reduce_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_identity(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_identity";
    platform::RecordEvent op_type_record_event("c_identity pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_relu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "relu";
    platform::RecordEvent op_type_record_event("relu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_relu_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "relu";
    platform::RecordEvent op_type_record_event("relu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_is_empty(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "is_empty";
    platform::RecordEvent op_type_record_event("is_empty pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reduce_all(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reduce_all";
    platform::RecordEvent op_type_record_event("reduce_all pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_edit_distance(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "edit_distance";
    platform::RecordEvent op_type_record_event("edit_distance pybind_imperative_func");
    
    auto Hyps = GetVarBaseFromArgs(op_type, "Hyps", args, 0, false);
    auto Refs = GetVarBaseFromArgs(op_type, "Refs", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"SequenceNum", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Hyps", {Hyps}},{"Refs", {Refs}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["SequenceNum"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_distributed_lookup_table(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "distributed_lookup_table";
    platform::RecordEvent op_type_record_event("distributed_lookup_table pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 1, false);
    auto OutputsNum = GetUnsignedLongFromArgs(op_type, "OutputsNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Outputs", ConstructDuplicableOutput(OutputsNum)}};
    imperative::NameVarBaseMap ins = {{"Ids", Ids},{"W", {W}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Outputs"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_tril_indices(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "tril_indices";
    platform::RecordEvent op_type_record_event("tril_indices pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bmm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bmm";
    platform::RecordEvent op_type_record_event("bmm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_yolo_box(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "yolo_box";
    platform::RecordEvent op_type_record_event("yolo_box pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ImgSize = GetVarBaseFromArgs(op_type, "ImgSize", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Boxes", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Scores", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ImgSize", {ImgSize}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Boxes"][0],outs["Scores"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_soft_relu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "soft_relu";
    platform::RecordEvent op_type_record_event("soft_relu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_soft_relu_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "soft_relu";
    platform::RecordEvent op_type_record_event("soft_relu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_density_prior_box(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "density_prior_box";
    platform::RecordEvent op_type_record_event("density_prior_box pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Image = GetVarBaseFromArgs(op_type, "Image", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Boxes", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Variances", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Image", {Image}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Boxes"][0],outs["Variances"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_swish(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "swish";
    platform::RecordEvent op_type_record_event("swish pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_eye(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "eye";
    platform::RecordEvent op_type_record_event("eye pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cross_entropy(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cross_entropy";
    platform::RecordEvent op_type_record_event("cross_entropy pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dpsgd(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dpsgd";
    platform::RecordEvent op_type_record_event("dpsgd pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 2, false);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"LearningRate", {LearningRate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["ParamOut"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_cholesky(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cholesky";
    platform::RecordEvent op_type_record_event("cholesky pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_batch_fc(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "batch_fc";
    platform::RecordEvent op_type_record_event("batch_fc pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"W", {W}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_nearest_interp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nearest_interp";
    platform::RecordEvent op_type_record_event("nearest_interp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto OutSize = GetVarBaseFromArgs(op_type, "OutSize", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (OutSize != nullptr) {
      ins["OutSize"] = {OutSize};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_gather(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gather";
    platform::RecordEvent op_type_record_event("gather pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    auto Axis = GetVarBaseFromArgs(op_type, "Axis", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Index", {Index}}};
    
    if (Axis != nullptr) {
      ins["Axis"] = {Axis};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_trilinear_interp_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "trilinear_interp_v2";
    platform::RecordEvent op_type_record_event("trilinear_interp_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_box_clip(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "box_clip";
    platform::RecordEvent op_type_record_event("box_clip pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"ImInfo", {ImInfo}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_c_allgather(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "c_allgather";
    platform::RecordEvent op_type_record_event("c_allgather pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_isnan_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isnan_v2";
    platform::RecordEvent op_type_record_event("isnan_v2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lu";
    platform::RecordEvent op_type_record_event("lu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Pivots", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Infos", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Pivots"][0],outs["Infos"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lu_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lu";
    platform::RecordEvent op_type_record_event("lu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}},{"Pivots", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Infos", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Pivots"][0],outs["Infos"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_softmax(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "softmax";
    platform::RecordEvent op_type_record_event("softmax pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_softmax_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "softmax";
    platform::RecordEvent op_type_record_event("softmax pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conv2d_fusion(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv2d_fusion";
    platform::RecordEvent op_type_record_event("conv2d_fusion pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_batch_norm_act(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_batch_norm_act";
    platform::RecordEvent op_type_record_event("fused_batch_norm_act pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    auto Mean = GetVarBaseFromArgs(op_type, "Mean", args, 3, false);
    auto Variance = GetVarBaseFromArgs(op_type, "Variance", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MeanOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"VarianceOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReserveSpace", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"Bias", {Bias}},{"Mean", {Mean}},{"Variance", {Variance}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["MeanOut"][0],outs["VarianceOut"][0],outs["SavedMean"][0],outs["SavedVariance"][0],outs["ReserveSpace"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_get_float_status(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "get_float_status";
    platform::RecordEvent op_type_record_event("get_float_status pybind_imperative_func");
    
    auto FloatStatus = GetVarBaseFromArgs(op_type, "FloatStatus", args, 0, false);
    auto FloatStatusOut = GetVarBaseFromArgs(op_type, "FloatStatusOut", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"FloatStatusOut", {FloatStatusOut}}};
    imperative::NameVarBaseMap ins = {{"FloatStatus", {FloatStatus}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["FloatStatusOut"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_index_sample(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "index_sample";
    platform::RecordEvent op_type_record_event("index_sample pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Index", {Index}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_min(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_min";
    platform::RecordEvent op_type_record_event("elementwise_min pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logical_not(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logical_not";
    platform::RecordEvent op_type_record_event("logical_not pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_erfinv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "erfinv";
    platform::RecordEvent op_type_record_event("erfinv pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_erfinv_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "erfinv";
    platform::RecordEvent op_type_record_event("erfinv pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_collect_fpn_proposals(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "collect_fpn_proposals";
    platform::RecordEvent op_type_record_event("collect_fpn_proposals pybind_imperative_func");
    
    auto MultiLevelRois = GetVarBaseListFromArgs(op_type, "MultiLevelRois", args, 0, false);
    auto MultiLevelScores = GetVarBaseListFromArgs(op_type, "MultiLevelScores", args, 1, false);
    auto MultiLevelRoIsNum = GetVarBaseListFromArgs(op_type, "MultiLevelRoIsNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"FpnRois", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RoisNum", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"MultiLevelRois", MultiLevelRois},{"MultiLevelScores", MultiLevelScores}};
    
    if (MultiLevelRoIsNum.size() != 0) {
      ins["MultiLevelRoIsNum"] = MultiLevelRoIsNum;
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["FpnRois"][0],outs["RoisNum"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pixel_shuffle(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pixel_shuffle";
    platform::RecordEvent op_type_record_event("pixel_shuffle pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_thresholded_relu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "thresholded_relu";
    platform::RecordEvent op_type_record_event("thresholded_relu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_polygon_box_transform(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "polygon_box_transform";
    platform::RecordEvent op_type_record_event("polygon_box_transform pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lookup_table_dequant(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lookup_table_dequant";
    platform::RecordEvent op_type_record_event("lookup_table_dequant pybind_imperative_func");
    
    auto W = GetVarBaseFromArgs(op_type, "W", args, 0, false);
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"W", {W}},{"Ids", {Ids}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_warpctc(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "warpctc";
    platform::RecordEvent op_type_record_event("warpctc pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto LogitsLength = GetVarBaseFromArgs(op_type, "LogitsLength", args, 2, true);
    auto LabelLength = GetVarBaseFromArgs(op_type, "LabelLength", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"WarpCTCGrad", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Label", {Label}}};
    
    if (LogitsLength != nullptr) {
      ins["LogitsLength"] = {LogitsLength};
    }

    if (LabelLength != nullptr) {
      ins["LabelLength"] = {LabelLength};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["WarpCTCGrad"][0],outs["Loss"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_heaviside(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_heaviside";
    platform::RecordEvent op_type_record_event("elementwise_heaviside pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fake_channel_wise_quantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_channel_wise_quantize_abs_max";
    platform::RecordEvent op_type_record_event("fake_channel_wise_quantize_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutScale", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutScale"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dequantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dequantize_abs_max";
    platform::RecordEvent op_type_record_event("dequantize_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_svd(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "svd";
    platform::RecordEvent op_type_record_event("svd pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"U", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"S", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"VH", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["U"][0],outs["S"][0],outs["VH"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_flip(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "flip";
    platform::RecordEvent op_type_record_event("flip pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_quantize(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "quantize";
    platform::RecordEvent op_type_record_event("quantize pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyMethodDef ExtestMethods[] = {
  {"rsqrt", (PyCFunction)(void(*)(void))imperative_rsqrt, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rsqrt in dygraph."},
  {"rsqrt_", (PyCFunction)(void(*)(void))imperative_rsqrt_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rsqrt_ in dygraph."},
  {"multihead_matmul", (PyCFunction)(void(*)(void))imperative_multihead_matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multihead_matmul in dygraph."},
  {"addmm", (PyCFunction)(void(*)(void))imperative_addmm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for addmm in dygraph."},
  {"gru", (PyCFunction)(void(*)(void))imperative_gru, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gru in dygraph."},
  {"round", (PyCFunction)(void(*)(void))imperative_round, METH_VARARGS | METH_KEYWORDS, "C++ interface function for round in dygraph."},
  {"round_", (PyCFunction)(void(*)(void))imperative_round_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for round_ in dygraph."},
  {"rank_attention", (PyCFunction)(void(*)(void))imperative_rank_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rank_attention in dygraph."},
  {"fused_embedding_fc_lstm", (PyCFunction)(void(*)(void))imperative_fused_embedding_fc_lstm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_embedding_fc_lstm in dygraph."},
  {"where_index", (PyCFunction)(void(*)(void))imperative_where_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for where_index in dygraph."},
  {"bicubic_interp", (PyCFunction)(void(*)(void))imperative_bicubic_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bicubic_interp in dygraph."},
  {"arg_min", (PyCFunction)(void(*)(void))imperative_arg_min, METH_VARARGS | METH_KEYWORDS, "C++ interface function for arg_min in dygraph."},
  {"tile", (PyCFunction)(void(*)(void))imperative_tile, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tile in dygraph."},
  {"distributed_fused_lamb_init", (PyCFunction)(void(*)(void))imperative_distributed_fused_lamb_init, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distributed_fused_lamb_init in dygraph."},
  {"dequantize_linear", (PyCFunction)(void(*)(void))imperative_dequantize_linear, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dequantize_linear in dygraph."},
  {"bilinear_tensor_product", (PyCFunction)(void(*)(void))imperative_bilinear_tensor_product, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear_tensor_product in dygraph."},
  {"ctc_align", (PyCFunction)(void(*)(void))imperative_ctc_align, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ctc_align in dygraph."},
  {"pow2_decay_with_linear_warmup", (PyCFunction)(void(*)(void))imperative_pow2_decay_with_linear_warmup, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pow2_decay_with_linear_warmup in dygraph."},
  {"reduce_amin", (PyCFunction)(void(*)(void))imperative_reduce_amin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_amin in dygraph."},
  {"split", (PyCFunction)(void(*)(void))imperative_split, METH_VARARGS | METH_KEYWORDS, "C++ interface function for split in dygraph."},
  {"fc", (PyCFunction)(void(*)(void))imperative_fc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fc in dygraph."},
  {"clear_float_status", (PyCFunction)(void(*)(void))imperative_clear_float_status, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clear_float_status in dygraph."},
  {"load", (PyCFunction)(void(*)(void))imperative_load, METH_VARARGS | METH_KEYWORDS, "C++ interface function for load in dygraph."},
  {"matmul_v2", (PyCFunction)(void(*)(void))imperative_matmul_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matmul_v2 in dygraph."},
  {"elementwise_max", (PyCFunction)(void(*)(void))imperative_elementwise_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_max in dygraph."},
  {"c_embedding", (PyCFunction)(void(*)(void))imperative_c_embedding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_embedding in dygraph."},
  {"adadelta", (PyCFunction)(void(*)(void))imperative_adadelta, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adadelta in dygraph."},
  {"chunk_eval", (PyCFunction)(void(*)(void))imperative_chunk_eval, METH_VARARGS | METH_KEYWORDS, "C++ interface function for chunk_eval in dygraph."},
  {"check_finite_and_unscale", (PyCFunction)(void(*)(void))imperative_check_finite_and_unscale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for check_finite_and_unscale in dygraph."},
  {"sparse_momentum", (PyCFunction)(void(*)(void))imperative_sparse_momentum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_momentum in dygraph."},
  {"complex", (PyCFunction)(void(*)(void))imperative_complex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for complex in dygraph."},
  {"tan", (PyCFunction)(void(*)(void))imperative_tan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tan in dygraph."},
  {"fused_bias_dropout_residual_layer_norm", (PyCFunction)(void(*)(void))imperative_fused_bias_dropout_residual_layer_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bias_dropout_residual_layer_norm in dygraph."},
  {"adam", (PyCFunction)(void(*)(void))imperative_adam, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adam in dygraph."},
  {"fsp", (PyCFunction)(void(*)(void))imperative_fsp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fsp in dygraph."},
  {"where", (PyCFunction)(void(*)(void))imperative_where, METH_VARARGS | METH_KEYWORDS, "C++ interface function for where in dygraph."},
  {"logical_xor", (PyCFunction)(void(*)(void))imperative_logical_xor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_xor in dygraph."},
  {"multiclass_nms3", (PyCFunction)(void(*)(void))imperative_multiclass_nms3, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiclass_nms3 in dygraph."},
  {"one_hot_v2", (PyCFunction)(void(*)(void))imperative_one_hot_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for one_hot_v2 in dygraph."},
  {"sequence_softmax", (PyCFunction)(void(*)(void))imperative_sequence_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_softmax in dygraph."},
  {"affine_channel", (PyCFunction)(void(*)(void))imperative_affine_channel, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_channel in dygraph."},
  {"affine_channel_", (PyCFunction)(void(*)(void))imperative_affine_channel_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_channel_ in dygraph."},
  {"triangular_solve", (PyCFunction)(void(*)(void))imperative_triangular_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triangular_solve in dygraph."},
  {"sequence_topk_avg_pooling", (PyCFunction)(void(*)(void))imperative_sequence_topk_avg_pooling, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_topk_avg_pooling in dygraph."},
  {"space_to_depth", (PyCFunction)(void(*)(void))imperative_space_to_depth, METH_VARARGS | METH_KEYWORDS, "C++ interface function for space_to_depth in dygraph."},
  {"reverse", (PyCFunction)(void(*)(void))imperative_reverse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reverse in dygraph."},
  {"fused_embedding_eltwise_layernorm", (PyCFunction)(void(*)(void))imperative_fused_embedding_eltwise_layernorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_embedding_eltwise_layernorm in dygraph."},
  {"expand_v2", (PyCFunction)(void(*)(void))imperative_expand_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand_v2 in dygraph."},
  {"repeat_interleave", (PyCFunction)(void(*)(void))imperative_repeat_interleave, METH_VARARGS | METH_KEYWORDS, "C++ interface function for repeat_interleave in dygraph."},
  {"lgamma", (PyCFunction)(void(*)(void))imperative_lgamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lgamma in dygraph."},
  {"solve", (PyCFunction)(void(*)(void))imperative_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for solve in dygraph."},
  {"deformable_psroi_pooling", (PyCFunction)(void(*)(void))imperative_deformable_psroi_pooling, METH_VARARGS | METH_KEYWORDS, "C++ interface function for deformable_psroi_pooling in dygraph."},
  {"transfer_layout", (PyCFunction)(void(*)(void))imperative_transfer_layout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transfer_layout in dygraph."},
  {"instance_norm", (PyCFunction)(void(*)(void))imperative_instance_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for instance_norm in dygraph."},
  {"decode_jpeg", (PyCFunction)(void(*)(void))imperative_decode_jpeg, METH_VARARGS | METH_KEYWORDS, "C++ interface function for decode_jpeg in dygraph."},
  {"distributed_push_sparse", (PyCFunction)(void(*)(void))imperative_distributed_push_sparse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distributed_push_sparse in dygraph."},
  {"gather_nd", (PyCFunction)(void(*)(void))imperative_gather_nd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather_nd in dygraph."},
  {"reduce_prod", (PyCFunction)(void(*)(void))imperative_reduce_prod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_prod in dygraph."},
  {"matrix_rank", (PyCFunction)(void(*)(void))imperative_matrix_rank, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_rank in dygraph."},
  {"asin", (PyCFunction)(void(*)(void))imperative_asin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asin in dygraph."},
  {"lstmp", (PyCFunction)(void(*)(void))imperative_lstmp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstmp in dygraph."},
  {"iou_similarity", (PyCFunction)(void(*)(void))imperative_iou_similarity, METH_VARARGS | METH_KEYWORDS, "C++ interface function for iou_similarity in dygraph."},
  {"huber_loss", (PyCFunction)(void(*)(void))imperative_huber_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for huber_loss in dygraph."},
  {"one_hot", (PyCFunction)(void(*)(void))imperative_one_hot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for one_hot in dygraph."},
  {"sequence_slice", (PyCFunction)(void(*)(void))imperative_sequence_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_slice in dygraph."},
  {"lookup_table", (PyCFunction)(void(*)(void))imperative_lookup_table, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lookup_table in dygraph."},
  {"softplus", (PyCFunction)(void(*)(void))imperative_softplus, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softplus in dygraph."},
  {"depthwise_conv2d", (PyCFunction)(void(*)(void))imperative_depthwise_conv2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depthwise_conv2d in dygraph."},
  {"c_allreduce_sum", (PyCFunction)(void(*)(void))imperative_c_allreduce_sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_sum in dygraph."},
  {"c_allreduce_sum_", (PyCFunction)(void(*)(void))imperative_c_allreduce_sum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_sum_ in dygraph."},
  {"fused_fc_elementwise_layernorm", (PyCFunction)(void(*)(void))imperative_fused_fc_elementwise_layernorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_fc_elementwise_layernorm in dygraph."},
  {"sigmoid_cross_entropy_with_logits", (PyCFunction)(void(*)(void))imperative_sigmoid_cross_entropy_with_logits, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_cross_entropy_with_logits in dygraph."},
  {"sigmoid_cross_entropy_with_logits_", (PyCFunction)(void(*)(void))imperative_sigmoid_cross_entropy_with_logits_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_cross_entropy_with_logits_ in dygraph."},
  {"exp", (PyCFunction)(void(*)(void))imperative_exp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exp in dygraph."},
  {"exp_", (PyCFunction)(void(*)(void))imperative_exp_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exp_ in dygraph."},
  {"scatter", (PyCFunction)(void(*)(void))imperative_scatter, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter in dygraph."},
  {"scatter_", (PyCFunction)(void(*)(void))imperative_scatter_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter_ in dygraph."},
  {"c_allreduce_min", (PyCFunction)(void(*)(void))imperative_c_allreduce_min, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_min in dygraph."},
  {"c_allreduce_min_", (PyCFunction)(void(*)(void))imperative_c_allreduce_min_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_min_ in dygraph."},
  {"equal_all", (PyCFunction)(void(*)(void))imperative_equal_all, METH_VARARGS | METH_KEYWORDS, "C++ interface function for equal_all in dygraph."},
  {"searchsorted", (PyCFunction)(void(*)(void))imperative_searchsorted, METH_VARARGS | METH_KEYWORDS, "C++ interface function for searchsorted in dygraph."},
  {"fusion_squared_mat_sub", (PyCFunction)(void(*)(void))imperative_fusion_squared_mat_sub, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_squared_mat_sub in dygraph."},
  {"unique", (PyCFunction)(void(*)(void))imperative_unique, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unique in dygraph."},
  {"log", (PyCFunction)(void(*)(void))imperative_log, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log in dygraph."},
  {"log_", (PyCFunction)(void(*)(void))imperative_log_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_ in dygraph."},
  {"conv_shift", (PyCFunction)(void(*)(void))imperative_conv_shift, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv_shift in dygraph."},
  {"as_complex", (PyCFunction)(void(*)(void))imperative_as_complex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for as_complex in dygraph."},
  {"smooth_l1_loss", (PyCFunction)(void(*)(void))imperative_smooth_l1_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for smooth_l1_loss in dygraph."},
  {"linear_interp_v2", (PyCFunction)(void(*)(void))imperative_linear_interp_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linear_interp_v2 in dygraph."},
  {"momentum", (PyCFunction)(void(*)(void))imperative_momentum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for momentum in dygraph."},
  {"temporal_shift", (PyCFunction)(void(*)(void))imperative_temporal_shift, METH_VARARGS | METH_KEYWORDS, "C++ interface function for temporal_shift in dygraph."},
  {"nce", (PyCFunction)(void(*)(void))imperative_nce, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nce in dygraph."},
  {"mv", (PyCFunction)(void(*)(void))imperative_mv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mv in dygraph."},
  {"global_scatter", (PyCFunction)(void(*)(void))imperative_global_scatter, METH_VARARGS | METH_KEYWORDS, "C++ interface function for global_scatter in dygraph."},
  {"dropout_nd", (PyCFunction)(void(*)(void))imperative_dropout_nd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dropout_nd in dygraph."},
  {"proximal_gd", (PyCFunction)(void(*)(void))imperative_proximal_gd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for proximal_gd in dygraph."},
  {"memcpy_h2d", (PyCFunction)(void(*)(void))imperative_memcpy_h2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy_h2d in dygraph."},
  {"add_position_encoding", (PyCFunction)(void(*)(void))imperative_add_position_encoding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_position_encoding in dygraph."},
  {"cosh", (PyCFunction)(void(*)(void))imperative_cosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cosh in dygraph."},
  {"hash", (PyCFunction)(void(*)(void))imperative_hash, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hash in dygraph."},
  {"grad_add", (PyCFunction)(void(*)(void))imperative_grad_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for grad_add in dygraph."},
  {"sign", (PyCFunction)(void(*)(void))imperative_sign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sign in dygraph."},
  {"prelu", (PyCFunction)(void(*)(void))imperative_prelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prelu in dygraph."},
  {"linspace", (PyCFunction)(void(*)(void))imperative_linspace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linspace in dygraph."},
  {"fill_diagonal", (PyCFunction)(void(*)(void))imperative_fill_diagonal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal in dygraph."},
  {"fill_diagonal_", (PyCFunction)(void(*)(void))imperative_fill_diagonal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_ in dygraph."},
  {"logsigmoid", (PyCFunction)(void(*)(void))imperative_logsigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsigmoid in dygraph."},
  {"load_combine", (PyCFunction)(void(*)(void))imperative_load_combine, METH_VARARGS | METH_KEYWORDS, "C++ interface function for load_combine in dygraph."},
  {"fetch_v2", (PyCFunction)(void(*)(void))imperative_fetch_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fetch_v2 in dygraph."},
  {"randperm", (PyCFunction)(void(*)(void))imperative_randperm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for randperm in dygraph."},
  {"sequence_scatter", (PyCFunction)(void(*)(void))imperative_sequence_scatter, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_scatter in dygraph."},
  {"relu6", (PyCFunction)(void(*)(void))imperative_relu6, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu6 in dygraph."},
  {"relu6_", (PyCFunction)(void(*)(void))imperative_relu6_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu6_ in dygraph."},
  {"partial_sum", (PyCFunction)(void(*)(void))imperative_partial_sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for partial_sum in dygraph."},
  {"partial_allgather", (PyCFunction)(void(*)(void))imperative_partial_allgather, METH_VARARGS | METH_KEYWORDS, "C++ interface function for partial_allgather in dygraph."},
  {"partial_allgather_", (PyCFunction)(void(*)(void))imperative_partial_allgather_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for partial_allgather_ in dygraph."},
  {"c_scatter", (PyCFunction)(void(*)(void))imperative_c_scatter, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_scatter in dygraph."},
  {"alltoall", (PyCFunction)(void(*)(void))imperative_alltoall, METH_VARARGS | METH_KEYWORDS, "C++ interface function for alltoall in dygraph."},
  {"alltoall_", (PyCFunction)(void(*)(void))imperative_alltoall_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for alltoall_ in dygraph."},
  {"conv3d", (PyCFunction)(void(*)(void))imperative_conv3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d in dygraph."},
  {"lu_unpack", (PyCFunction)(void(*)(void))imperative_lu_unpack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_unpack in dygraph."},
  {"lstm_unit", (PyCFunction)(void(*)(void))imperative_lstm_unit, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstm_unit in dygraph."},
  {"not_equal", (PyCFunction)(void(*)(void))imperative_not_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for not_equal in dygraph."},
  {"transpose2", (PyCFunction)(void(*)(void))imperative_transpose2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transpose2 in dygraph."},
  {"c_sync_comm_stream", (PyCFunction)(void(*)(void))imperative_c_sync_comm_stream, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_sync_comm_stream in dygraph."},
  {"uniform_random_batch_size_like", (PyCFunction)(void(*)(void))imperative_uniform_random_batch_size_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_random_batch_size_like in dygraph."},
  {"yolo_box_head", (PyCFunction)(void(*)(void))imperative_yolo_box_head, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_box_head in dygraph."},
  {"unfold", (PyCFunction)(void(*)(void))imperative_unfold, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unfold in dygraph."},
  {"lrn", (PyCFunction)(void(*)(void))imperative_lrn, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lrn in dygraph."},
  {"isclose", (PyCFunction)(void(*)(void))imperative_isclose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isclose in dygraph."},
  {"softmax_with_cross_entropy", (PyCFunction)(void(*)(void))imperative_softmax_with_cross_entropy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax_with_cross_entropy in dygraph."},
  {"softmax_with_cross_entropy_", (PyCFunction)(void(*)(void))imperative_softmax_with_cross_entropy_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax_with_cross_entropy_ in dygraph."},
  {"isfinite_v2", (PyCFunction)(void(*)(void))imperative_isfinite_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isfinite_v2 in dygraph."},
  {"bernoulli", (PyCFunction)(void(*)(void))imperative_bernoulli, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bernoulli in dygraph."},
  {"max_pool3d_with_index", (PyCFunction)(void(*)(void))imperative_max_pool3d_with_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool3d_with_index in dygraph."},
  {"fused_seqpool_cvm", (PyCFunction)(void(*)(void))imperative_fused_seqpool_cvm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_seqpool_cvm in dygraph."},
  {"gaussian_random", (PyCFunction)(void(*)(void))imperative_gaussian_random, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian_random in dygraph."},
  {"flatten2", (PyCFunction)(void(*)(void))imperative_flatten2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten2 in dygraph."},
  {"flatten2_", (PyCFunction)(void(*)(void))imperative_flatten2_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten2_ in dygraph."},
  {"matmul", (PyCFunction)(void(*)(void))imperative_matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matmul in dygraph."},
  {"cvm", (PyCFunction)(void(*)(void))imperative_cvm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cvm in dygraph."},
  {"adamax", (PyCFunction)(void(*)(void))imperative_adamax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adamax in dygraph."},
  {"recv_v2", (PyCFunction)(void(*)(void))imperative_recv_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for recv_v2 in dygraph."},
  {"requantize", (PyCFunction)(void(*)(void))imperative_requantize, METH_VARARGS | METH_KEYWORDS, "C++ interface function for requantize in dygraph."},
  {"masked_select", (PyCFunction)(void(*)(void))imperative_masked_select, METH_VARARGS | METH_KEYWORDS, "C++ interface function for masked_select in dygraph."},
  {"range", (PyCFunction)(void(*)(void))imperative_range, METH_VARARGS | METH_KEYWORDS, "C++ interface function for range in dygraph."},
  {"bitwise_not", (PyCFunction)(void(*)(void))imperative_bitwise_not, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_not in dygraph."},
  {"trace", (PyCFunction)(void(*)(void))imperative_trace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trace in dygraph."},
  {"multinomial", (PyCFunction)(void(*)(void))imperative_multinomial, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multinomial in dygraph."},
  {"modified_huber_loss", (PyCFunction)(void(*)(void))imperative_modified_huber_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for modified_huber_loss in dygraph."},
  {"c_reduce_prod", (PyCFunction)(void(*)(void))imperative_c_reduce_prod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_reduce_prod in dygraph."},
  {"roll", (PyCFunction)(void(*)(void))imperative_roll, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roll in dygraph."},
  {"squared_l2_distance", (PyCFunction)(void(*)(void))imperative_squared_l2_distance, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squared_l2_distance in dygraph."},
  {"conv3d_transpose", (PyCFunction)(void(*)(void))imperative_conv3d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d_transpose in dygraph."},
  {"share_data", (PyCFunction)(void(*)(void))imperative_share_data, METH_VARARGS | METH_KEYWORDS, "C++ interface function for share_data in dygraph."},
  {"fake_quantize_abs_max", (PyCFunction)(void(*)(void))imperative_fake_quantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_abs_max in dygraph."},
  {"rrelu", (PyCFunction)(void(*)(void))imperative_rrelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rrelu in dygraph."},
  {"unique_with_counts", (PyCFunction)(void(*)(void))imperative_unique_with_counts, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unique_with_counts in dygraph."},
  {"fill", (PyCFunction)(void(*)(void))imperative_fill, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill in dygraph."},
  {"concat", (PyCFunction)(void(*)(void))imperative_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for concat in dygraph."},
  {"fill_zeros_like", (PyCFunction)(void(*)(void))imperative_fill_zeros_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_zeros_like in dygraph."},
  {"hierarchical_sigmoid", (PyCFunction)(void(*)(void))imperative_hierarchical_sigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hierarchical_sigmoid in dygraph."},
  {"isinf_v2", (PyCFunction)(void(*)(void))imperative_isinf_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isinf_v2 in dygraph."},
  {"squeeze", (PyCFunction)(void(*)(void))imperative_squeeze, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze in dygraph."},
  {"multiclass_nms2", (PyCFunction)(void(*)(void))imperative_multiclass_nms2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiclass_nms2 in dygraph."},
  {"bpr_loss", (PyCFunction)(void(*)(void))imperative_bpr_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bpr_loss in dygraph."},
  {"fft_c2c", (PyCFunction)(void(*)(void))imperative_fft_c2c, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_c2c in dygraph."},
  {"bicubic_interp_v2", (PyCFunction)(void(*)(void))imperative_bicubic_interp_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bicubic_interp_v2 in dygraph."},
  {"angle", (PyCFunction)(void(*)(void))imperative_angle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for angle in dygraph."},
  {"reshape", (PyCFunction)(void(*)(void))imperative_reshape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape in dygraph."},
  {"reshape_", (PyCFunction)(void(*)(void))imperative_reshape_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape_ in dygraph."},
  {"coalesce_tensor", (PyCFunction)(void(*)(void))imperative_coalesce_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for coalesce_tensor in dygraph."},
  {"roi_align", (PyCFunction)(void(*)(void))imperative_roi_align, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_align in dygraph."},
  {"reshape2", (PyCFunction)(void(*)(void))imperative_reshape2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape2 in dygraph."},
  {"reshape2_", (PyCFunction)(void(*)(void))imperative_reshape2_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape2_ in dygraph."},
  {"reduce_any", (PyCFunction)(void(*)(void))imperative_reduce_any, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_any in dygraph."},
  {"limit_by_capacity", (PyCFunction)(void(*)(void))imperative_limit_by_capacity, METH_VARARGS | METH_KEYWORDS, "C++ interface function for limit_by_capacity in dygraph."},
  {"unstack", (PyCFunction)(void(*)(void))imperative_unstack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unstack in dygraph."},
  {"scatter_nd_add", (PyCFunction)(void(*)(void))imperative_scatter_nd_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter_nd_add in dygraph."},
  {"sequence_reshape", (PyCFunction)(void(*)(void))imperative_sequence_reshape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_reshape in dygraph."},
  {"bilateral_slice", (PyCFunction)(void(*)(void))imperative_bilateral_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilateral_slice in dygraph."},
  {"fill_any_like", (PyCFunction)(void(*)(void))imperative_fill_any_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_any_like in dygraph."},
  {"partial_recv", (PyCFunction)(void(*)(void))imperative_partial_recv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for partial_recv in dygraph."},
  {"empty", (PyCFunction)(void(*)(void))imperative_empty, METH_VARARGS | METH_KEYWORDS, "C++ interface function for empty in dygraph."},
  {"pad_constant_like", (PyCFunction)(void(*)(void))imperative_pad_constant_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad_constant_like in dygraph."},
  {"pool2d", (PyCFunction)(void(*)(void))imperative_pool2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool2d in dygraph."},
  {"size", (PyCFunction)(void(*)(void))imperative_size, METH_VARARGS | METH_KEYWORDS, "C++ interface function for size in dygraph."},
  {"imag", (PyCFunction)(void(*)(void))imperative_imag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for imag in dygraph."},
  {"pull_gpups_sparse", (PyCFunction)(void(*)(void))imperative_pull_gpups_sparse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pull_gpups_sparse in dygraph."},
  {"eigh", (PyCFunction)(void(*)(void))imperative_eigh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigh in dygraph."},
  {"stack", (PyCFunction)(void(*)(void))imperative_stack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stack in dygraph."},
  {"dgc_momentum", (PyCFunction)(void(*)(void))imperative_dgc_momentum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dgc_momentum in dygraph."},
  {"lamb", (PyCFunction)(void(*)(void))imperative_lamb, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lamb in dygraph."},
  {"generate_proposals_v2", (PyCFunction)(void(*)(void))imperative_generate_proposals_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for generate_proposals_v2 in dygraph."},
  {"c_sync_calc_stream", (PyCFunction)(void(*)(void))imperative_c_sync_calc_stream, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_sync_calc_stream in dygraph."},
  {"bitwise_or", (PyCFunction)(void(*)(void))imperative_bitwise_or, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_or in dygraph."},
  {"gru_unit", (PyCFunction)(void(*)(void))imperative_gru_unit, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gru_unit in dygraph."},
  {"fake_channel_wise_quantize_dequantize_abs_max", (PyCFunction)(void(*)(void))imperative_fake_channel_wise_quantize_dequantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_channel_wise_quantize_dequantize_abs_max in dygraph."},
  {"sampling_id", (PyCFunction)(void(*)(void))imperative_sampling_id, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sampling_id in dygraph."},
  {"unsqueeze2", (PyCFunction)(void(*)(void))imperative_unsqueeze2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze2 in dygraph."},
  {"unsqueeze2_", (PyCFunction)(void(*)(void))imperative_unsqueeze2_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze2_ in dygraph."},
  {"transfer_dtype", (PyCFunction)(void(*)(void))imperative_transfer_dtype, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transfer_dtype in dygraph."},
  {"allreduce", (PyCFunction)(void(*)(void))imperative_allreduce, METH_VARARGS | METH_KEYWORDS, "C++ interface function for allreduce in dygraph."},
  {"average_accumulates", (PyCFunction)(void(*)(void))imperative_average_accumulates, METH_VARARGS | METH_KEYWORDS, "C++ interface function for average_accumulates in dygraph."},
  {"sequence_enumerate", (PyCFunction)(void(*)(void))imperative_sequence_enumerate, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_enumerate in dygraph."},
  {"fusion_seqconv_eltadd_relu", (PyCFunction)(void(*)(void))imperative_fusion_seqconv_eltadd_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqconv_eltadd_relu in dygraph."},
  {"bce_loss", (PyCFunction)(void(*)(void))imperative_bce_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bce_loss in dygraph."},
  {"bce_loss_", (PyCFunction)(void(*)(void))imperative_bce_loss_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bce_loss_ in dygraph."},
  {"generate_proposal_labels", (PyCFunction)(void(*)(void))imperative_generate_proposal_labels, METH_VARARGS | METH_KEYWORDS, "C++ interface function for generate_proposal_labels in dygraph."},
  {"im2sequence", (PyCFunction)(void(*)(void))imperative_im2sequence, METH_VARARGS | METH_KEYWORDS, "C++ interface function for im2sequence in dygraph."},
  {"isinf", (PyCFunction)(void(*)(void))imperative_isinf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isinf in dygraph."},
  {"c_reducescatter", (PyCFunction)(void(*)(void))imperative_c_reducescatter, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_reducescatter in dygraph."},
  {"logcumsumexp", (PyCFunction)(void(*)(void))imperative_logcumsumexp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logcumsumexp in dygraph."},
  {"adagrad", (PyCFunction)(void(*)(void))imperative_adagrad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adagrad in dygraph."},
  {"linear_chain_crf", (PyCFunction)(void(*)(void))imperative_linear_chain_crf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linear_chain_crf in dygraph."},
  {"retinanet_target_assign", (PyCFunction)(void(*)(void))imperative_retinanet_target_assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for retinanet_target_assign in dygraph."},
  {"fusion_group", (PyCFunction)(void(*)(void))imperative_fusion_group, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_group in dygraph."},
  {"teacher_student_sigmoid_loss", (PyCFunction)(void(*)(void))imperative_teacher_student_sigmoid_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for teacher_student_sigmoid_loss in dygraph."},
  {"random_crop", (PyCFunction)(void(*)(void))imperative_random_crop, METH_VARARGS | METH_KEYWORDS, "C++ interface function for random_crop in dygraph."},
  {"lookup_table_v2", (PyCFunction)(void(*)(void))imperative_lookup_table_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lookup_table_v2 in dygraph."},
  {"elementwise_fmax", (PyCFunction)(void(*)(void))imperative_elementwise_fmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_fmax in dygraph."},
  {"graph_sample_neighbors", (PyCFunction)(void(*)(void))imperative_graph_sample_neighbors, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_sample_neighbors in dygraph."},
  {"detection_map", (PyCFunction)(void(*)(void))imperative_detection_map, METH_VARARGS | METH_KEYWORDS, "C++ interface function for detection_map in dygraph."},
  {"l1_norm", (PyCFunction)(void(*)(void))imperative_l1_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for l1_norm in dygraph."},
  {"sqrt", (PyCFunction)(void(*)(void))imperative_sqrt, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt in dygraph."},
  {"sqrt_", (PyCFunction)(void(*)(void))imperative_sqrt_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt_ in dygraph."},
  {"partial_send", (PyCFunction)(void(*)(void))imperative_partial_send, METH_VARARGS | METH_KEYWORDS, "C++ interface function for partial_send in dygraph."},
  {"fused_elemwise_activation", (PyCFunction)(void(*)(void))imperative_fused_elemwise_activation, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_elemwise_activation in dygraph."},
  {"slogdeterminant", (PyCFunction)(void(*)(void))imperative_slogdeterminant, METH_VARARGS | METH_KEYWORDS, "C++ interface function for slogdeterminant in dygraph."},
  {"share_buffer", (PyCFunction)(void(*)(void))imperative_share_buffer, METH_VARARGS | METH_KEYWORDS, "C++ interface function for share_buffer in dygraph."},
  {"poisson", (PyCFunction)(void(*)(void))imperative_poisson, METH_VARARGS | METH_KEYWORDS, "C++ interface function for poisson in dygraph."},
  {"bitwise_and", (PyCFunction)(void(*)(void))imperative_bitwise_and, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_and in dygraph."},
  {"diag_embed", (PyCFunction)(void(*)(void))imperative_diag_embed, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag_embed in dygraph."},
  {"unbind", (PyCFunction)(void(*)(void))imperative_unbind, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unbind in dygraph."},
  {"dropout", (PyCFunction)(void(*)(void))imperative_dropout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dropout in dygraph."},
  {"beam_search", (PyCFunction)(void(*)(void))imperative_beam_search, METH_VARARGS | METH_KEYWORDS, "C++ interface function for beam_search in dygraph."},
  {"moving_average_abs_max_scale", (PyCFunction)(void(*)(void))imperative_moving_average_abs_max_scale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for moving_average_abs_max_scale in dygraph."},
  {"greater_than", (PyCFunction)(void(*)(void))imperative_greater_than, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_than in dygraph."},
  {"log_loss", (PyCFunction)(void(*)(void))imperative_log_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_loss in dygraph."},
  {"kron", (PyCFunction)(void(*)(void))imperative_kron, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kron in dygraph."},
  {"sigmoid_focal_loss", (PyCFunction)(void(*)(void))imperative_sigmoid_focal_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_focal_loss in dygraph."},
  {"rmsprop", (PyCFunction)(void(*)(void))imperative_rmsprop, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rmsprop in dygraph."},
  {"conv2d", (PyCFunction)(void(*)(void))imperative_conv2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d in dygraph."},
  {"graph_reindex", (PyCFunction)(void(*)(void))imperative_graph_reindex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_reindex in dygraph."},
  {"uniform_random_inplace", (PyCFunction)(void(*)(void))imperative_uniform_random_inplace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_random_inplace in dygraph."},
  {"uniform_random_inplace_", (PyCFunction)(void(*)(void))imperative_uniform_random_inplace_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_random_inplace_ in dygraph."},
  {"maxout", (PyCFunction)(void(*)(void))imperative_maxout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for maxout in dygraph."},
  {"lstsq", (PyCFunction)(void(*)(void))imperative_lstsq, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstsq in dygraph."},
  {"linear_interp", (PyCFunction)(void(*)(void))imperative_linear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linear_interp in dygraph."},
  {"graph_khop_sampler", (PyCFunction)(void(*)(void))imperative_graph_khop_sampler, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_khop_sampler in dygraph."},
  {"put_along_axis", (PyCFunction)(void(*)(void))imperative_put_along_axis, METH_VARARGS | METH_KEYWORDS, "C++ interface function for put_along_axis in dygraph."},
  {"put_along_axis_", (PyCFunction)(void(*)(void))imperative_put_along_axis_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for put_along_axis_ in dygraph."},
  {"auc", (PyCFunction)(void(*)(void))imperative_auc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for auc in dygraph."},
  {"logical_or", (PyCFunction)(void(*)(void))imperative_logical_or, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_or in dygraph."},
  {"batch_norm", (PyCFunction)(void(*)(void))imperative_batch_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for batch_norm in dygraph."},
  {"c_reduce_sum", (PyCFunction)(void(*)(void))imperative_c_reduce_sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_reduce_sum in dygraph."},
  {"elementwise_add", (PyCFunction)(void(*)(void))imperative_elementwise_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_add in dygraph."},
  {"elementwise_add_", (PyCFunction)(void(*)(void))imperative_elementwise_add_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_add_ in dygraph."},
  {"acos", (PyCFunction)(void(*)(void))imperative_acos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acos in dygraph."},
  {"send_and_recv", (PyCFunction)(void(*)(void))imperative_send_and_recv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for send_and_recv in dygraph."},
  {"unpool", (PyCFunction)(void(*)(void))imperative_unpool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool in dygraph."},
  {"cumprod", (PyCFunction)(void(*)(void))imperative_cumprod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumprod in dygraph."},
  {"sample_logits", (PyCFunction)(void(*)(void))imperative_sample_logits, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sample_logits in dygraph."},
  {"pull_box_extended_sparse", (PyCFunction)(void(*)(void))imperative_pull_box_extended_sparse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pull_box_extended_sparse in dygraph."},
  {"crop_tensor", (PyCFunction)(void(*)(void))imperative_crop_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for crop_tensor in dygraph."},
  {"fill_constant", (PyCFunction)(void(*)(void))imperative_fill_constant, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_constant in dygraph."},
  {"deformable_conv", (PyCFunction)(void(*)(void))imperative_deformable_conv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for deformable_conv in dygraph."},
  {"generate_mask_labels", (PyCFunction)(void(*)(void))imperative_generate_mask_labels, METH_VARARGS | METH_KEYWORDS, "C++ interface function for generate_mask_labels in dygraph."},
  {"locality_aware_nms", (PyCFunction)(void(*)(void))imperative_locality_aware_nms, METH_VARARGS | METH_KEYWORDS, "C++ interface function for locality_aware_nms in dygraph."},
  {"expand_as", (PyCFunction)(void(*)(void))imperative_expand_as, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand_as in dygraph."},
  {"matrix_power", (PyCFunction)(void(*)(void))imperative_matrix_power, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_power in dygraph."},
  {"greater_equal", (PyCFunction)(void(*)(void))imperative_greater_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_equal in dygraph."},
  {"generate_proposals", (PyCFunction)(void(*)(void))imperative_generate_proposals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for generate_proposals in dygraph."},
  {"number_count", (PyCFunction)(void(*)(void))imperative_number_count, METH_VARARGS | METH_KEYWORDS, "C++ interface function for number_count in dygraph."},
  {"bilinear_interp", (PyCFunction)(void(*)(void))imperative_bilinear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear_interp in dygraph."},
  {"distributed_fused_lamb", (PyCFunction)(void(*)(void))imperative_distributed_fused_lamb, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distributed_fused_lamb in dygraph."},
  {"sigmoid", (PyCFunction)(void(*)(void))imperative_sigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid in dygraph."},
  {"sigmoid_", (PyCFunction)(void(*)(void))imperative_sigmoid_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_ in dygraph."},
  {"inplace_abn", (PyCFunction)(void(*)(void))imperative_inplace_abn, METH_VARARGS | METH_KEYWORDS, "C++ interface function for inplace_abn in dygraph."},
  {"inplace_abn_", (PyCFunction)(void(*)(void))imperative_inplace_abn_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for inplace_abn_ in dygraph."},
  {"softshrink", (PyCFunction)(void(*)(void))imperative_softshrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softshrink in dygraph."},
  {"mul", (PyCFunction)(void(*)(void))imperative_mul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mul in dygraph."},
  {"data_norm", (PyCFunction)(void(*)(void))imperative_data_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for data_norm in dygraph."},
  {"fused_multi_transformer", (PyCFunction)(void(*)(void))imperative_fused_multi_transformer, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_multi_transformer in dygraph."},
  {"asinh", (PyCFunction)(void(*)(void))imperative_asinh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asinh in dygraph."},
  {"get_tensor_from_selected_rows", (PyCFunction)(void(*)(void))imperative_get_tensor_from_selected_rows, METH_VARARGS | METH_KEYWORDS, "C++ interface function for get_tensor_from_selected_rows in dygraph."},
  {"spp", (PyCFunction)(void(*)(void))imperative_spp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for spp in dygraph."},
  {"floor", (PyCFunction)(void(*)(void))imperative_floor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor in dygraph."},
  {"floor_", (PyCFunction)(void(*)(void))imperative_floor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor_ in dygraph."},
  {"as_real", (PyCFunction)(void(*)(void))imperative_as_real, METH_VARARGS | METH_KEYWORDS, "C++ interface function for as_real in dygraph."},
  {"gelu", (PyCFunction)(void(*)(void))imperative_gelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gelu in dygraph."},
  {"retinanet_detection_output", (PyCFunction)(void(*)(void))imperative_retinanet_detection_output, METH_VARARGS | METH_KEYWORDS, "C++ interface function for retinanet_detection_output in dygraph."},
  {"minus", (PyCFunction)(void(*)(void))imperative_minus, METH_VARARGS | METH_KEYWORDS, "C++ interface function for minus in dygraph."},
  {"push_dense", (PyCFunction)(void(*)(void))imperative_push_dense, METH_VARARGS | METH_KEYWORDS, "C++ interface function for push_dense in dygraph."},
  {"silu", (PyCFunction)(void(*)(void))imperative_silu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for silu in dygraph."},
  {"sequence_erase", (PyCFunction)(void(*)(void))imperative_sequence_erase, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_erase in dygraph."},
  {"real", (PyCFunction)(void(*)(void))imperative_real, METH_VARARGS | METH_KEYWORDS, "C++ interface function for real in dygraph."},
  {"nearest_interp_v2", (PyCFunction)(void(*)(void))imperative_nearest_interp_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nearest_interp_v2 in dygraph."},
  {"dgc_clip_by_norm", (PyCFunction)(void(*)(void))imperative_dgc_clip_by_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dgc_clip_by_norm in dygraph."},
  {"squeeze2", (PyCFunction)(void(*)(void))imperative_squeeze2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze2 in dygraph."},
  {"squeeze2_", (PyCFunction)(void(*)(void))imperative_squeeze2_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze2_ in dygraph."},
  {"conj", (PyCFunction)(void(*)(void))imperative_conj, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conj in dygraph."},
  {"strided_slice", (PyCFunction)(void(*)(void))imperative_strided_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for strided_slice in dygraph."},
  {"precision_recall", (PyCFunction)(void(*)(void))imperative_precision_recall, METH_VARARGS | METH_KEYWORDS, "C++ interface function for precision_recall in dygraph."},
  {"fusion_seqexpand_concat_fc", (PyCFunction)(void(*)(void))imperative_fusion_seqexpand_concat_fc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqexpand_concat_fc in dygraph."},
  {"save", (PyCFunction)(void(*)(void))imperative_save, METH_VARARGS | METH_KEYWORDS, "C++ interface function for save in dygraph."},
  {"depthwise_conv2d_transpose", (PyCFunction)(void(*)(void))imperative_depthwise_conv2d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depthwise_conv2d_transpose in dygraph."},
  {"fake_quantize_range_abs_max", (PyCFunction)(void(*)(void))imperative_fake_quantize_range_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_range_abs_max in dygraph."},
  {"positive_negative_pair", (PyCFunction)(void(*)(void))imperative_positive_negative_pair, METH_VARARGS | METH_KEYWORDS, "C++ interface function for positive_negative_pair in dygraph."},
  {"square", (PyCFunction)(void(*)(void))imperative_square, METH_VARARGS | METH_KEYWORDS, "C++ interface function for square in dygraph."},
  {"square_", (PyCFunction)(void(*)(void))imperative_square_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for square_ in dygraph."},
  {"var_conv_2d", (PyCFunction)(void(*)(void))imperative_var_conv_2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for var_conv_2d in dygraph."},
  {"log1p", (PyCFunction)(void(*)(void))imperative_log1p, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log1p in dygraph."},
  {"channel_shuffle", (PyCFunction)(void(*)(void))imperative_channel_shuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for channel_shuffle in dygraph."},
  {"atan2", (PyCFunction)(void(*)(void))imperative_atan2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan2 in dygraph."},
  {"fused_softmax_mask_upper_triangle", (PyCFunction)(void(*)(void))imperative_fused_softmax_mask_upper_triangle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_softmax_mask_upper_triangle in dygraph."},
  {"clip_by_norm", (PyCFunction)(void(*)(void))imperative_clip_by_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_by_norm in dygraph."},
  {"box_decoder_and_assign", (PyCFunction)(void(*)(void))imperative_box_decoder_and_assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for box_decoder_and_assign in dygraph."},
  {"roi_pool", (PyCFunction)(void(*)(void))imperative_roi_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_pool in dygraph."},
  {"fft_r2c", (PyCFunction)(void(*)(void))imperative_fft_r2c, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_r2c in dygraph."},
  {"overlap_add", (PyCFunction)(void(*)(void))imperative_overlap_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for overlap_add in dygraph."},
  {"fill_constant_batch_size_like", (PyCFunction)(void(*)(void))imperative_fill_constant_batch_size_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_constant_batch_size_like in dygraph."},
  {"fill_any", (PyCFunction)(void(*)(void))imperative_fill_any, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_any in dygraph."},
  {"fill_any_", (PyCFunction)(void(*)(void))imperative_fill_any_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_any_ in dygraph."},
  {"dequantize_log", (PyCFunction)(void(*)(void))imperative_dequantize_log, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dequantize_log in dygraph."},
  {"c_split", (PyCFunction)(void(*)(void))imperative_c_split, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_split in dygraph."},
  {"barrier", (PyCFunction)(void(*)(void))imperative_barrier, METH_VARARGS | METH_KEYWORDS, "C++ interface function for barrier in dygraph."},
  {"max_pool2d_with_index", (PyCFunction)(void(*)(void))imperative_max_pool2d_with_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool2d_with_index in dygraph."},
  {"pad3d", (PyCFunction)(void(*)(void))imperative_pad3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad3d in dygraph."},
  {"norm", (PyCFunction)(void(*)(void))imperative_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for norm in dygraph."},
  {"viterbi_decode", (PyCFunction)(void(*)(void))imperative_viterbi_decode, METH_VARARGS | METH_KEYWORDS, "C++ interface function for viterbi_decode in dygraph."},
  {"mish", (PyCFunction)(void(*)(void))imperative_mish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mish in dygraph."},
  {"box_coder", (PyCFunction)(void(*)(void))imperative_box_coder, METH_VARARGS | METH_KEYWORDS, "C++ interface function for box_coder in dygraph."},
  {"flatten", (PyCFunction)(void(*)(void))imperative_flatten, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten in dygraph."},
  {"flatten_", (PyCFunction)(void(*)(void))imperative_flatten_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten_ in dygraph."},
  {"elementwise_mod", (PyCFunction)(void(*)(void))imperative_elementwise_mod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_mod in dygraph."},
  {"margin_cross_entropy", (PyCFunction)(void(*)(void))imperative_margin_cross_entropy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for margin_cross_entropy in dygraph."},
  {"pull_sparse", (PyCFunction)(void(*)(void))imperative_pull_sparse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pull_sparse in dygraph."},
  {"logical_and", (PyCFunction)(void(*)(void))imperative_logical_and, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_and in dygraph."},
  {"pow", (PyCFunction)(void(*)(void))imperative_pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pow in dygraph."},
  {"dirichlet", (PyCFunction)(void(*)(void))imperative_dirichlet, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dirichlet in dygraph."},
  {"stanh", (PyCFunction)(void(*)(void))imperative_stanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stanh in dygraph."},
  {"label_smooth", (PyCFunction)(void(*)(void))imperative_label_smooth, METH_VARARGS | METH_KEYWORDS, "C++ interface function for label_smooth in dygraph."},
  {"fold", (PyCFunction)(void(*)(void))imperative_fold, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fold in dygraph."},
  {"merged_momentum", (PyCFunction)(void(*)(void))imperative_merged_momentum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for merged_momentum in dygraph."},
  {"c_reduce_min", (PyCFunction)(void(*)(void))imperative_c_reduce_min, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_reduce_min in dygraph."},
  {"ascend_trigger", (PyCFunction)(void(*)(void))imperative_ascend_trigger, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ascend_trigger in dygraph."},
  {"rpn_target_assign", (PyCFunction)(void(*)(void))imperative_rpn_target_assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rpn_target_assign in dygraph."},
  {"fused_feedforward", (PyCFunction)(void(*)(void))imperative_fused_feedforward, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_feedforward in dygraph."},
  {"roi_perspective_transform", (PyCFunction)(void(*)(void))imperative_roi_perspective_transform, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_perspective_transform in dygraph."},
  {"expand", (PyCFunction)(void(*)(void))imperative_expand, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand in dygraph."},
  {"prroi_pool", (PyCFunction)(void(*)(void))imperative_prroi_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prroi_pool in dygraph."},
  {"pool3d", (PyCFunction)(void(*)(void))imperative_pool3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool3d in dygraph."},
  {"memcpy", (PyCFunction)(void(*)(void))imperative_memcpy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy in dygraph."},
  {"distribute_fpn_proposals", (PyCFunction)(void(*)(void))imperative_distribute_fpn_proposals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distribute_fpn_proposals in dygraph."},
  {"frame", (PyCFunction)(void(*)(void))imperative_frame, METH_VARARGS | METH_KEYWORDS, "C++ interface function for frame in dygraph."},
  {"bincount", (PyCFunction)(void(*)(void))imperative_bincount, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bincount in dygraph."},
  {"shape", (PyCFunction)(void(*)(void))imperative_shape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shape in dygraph."},
  {"mode", (PyCFunction)(void(*)(void))imperative_mode, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mode in dygraph."},
  {"group_norm", (PyCFunction)(void(*)(void))imperative_group_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for group_norm in dygraph."},
  {"c_softmax_with_cross_entropy", (PyCFunction)(void(*)(void))imperative_c_softmax_with_cross_entropy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_softmax_with_cross_entropy in dygraph."},
  {"c_softmax_with_cross_entropy_", (PyCFunction)(void(*)(void))imperative_c_softmax_with_cross_entropy_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_softmax_with_cross_entropy_ in dygraph."},
  {"resnet_unit", (PyCFunction)(void(*)(void))imperative_resnet_unit, METH_VARARGS | METH_KEYWORDS, "C++ interface function for resnet_unit in dygraph."},
  {"sequence_expand_as", (PyCFunction)(void(*)(void))imperative_sequence_expand_as, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_expand_as in dygraph."},
  {"cos_sim", (PyCFunction)(void(*)(void))imperative_cos_sim, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cos_sim in dygraph."},
  {"eigvals", (PyCFunction)(void(*)(void))imperative_eigvals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigvals in dygraph."},
  {"save_combine", (PyCFunction)(void(*)(void))imperative_save_combine, METH_VARARGS | METH_KEYWORDS, "C++ interface function for save_combine in dygraph."},
  {"class_center_sample", (PyCFunction)(void(*)(void))imperative_class_center_sample, METH_VARARGS | METH_KEYWORDS, "C++ interface function for class_center_sample in dygraph."},
  {"elementwise_fmin", (PyCFunction)(void(*)(void))imperative_elementwise_fmin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_fmin in dygraph."},
  {"read_file", (PyCFunction)(void(*)(void))imperative_read_file, METH_VARARGS | METH_KEYWORDS, "C++ interface function for read_file in dygraph."},
  {"isfinite", (PyCFunction)(void(*)(void))imperative_isfinite, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isfinite in dygraph."},
  {"arg_max", (PyCFunction)(void(*)(void))imperative_arg_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for arg_max in dygraph."},
  {"equal", (PyCFunction)(void(*)(void))imperative_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for equal in dygraph."},
  {"fake_dequantize_max_abs", (PyCFunction)(void(*)(void))imperative_fake_dequantize_max_abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_dequantize_max_abs in dygraph."},
  {"qr", (PyCFunction)(void(*)(void))imperative_qr, METH_VARARGS | METH_KEYWORDS, "C++ interface function for qr in dygraph."},
  {"anchor_generator", (PyCFunction)(void(*)(void))imperative_anchor_generator, METH_VARARGS | METH_KEYWORDS, "C++ interface function for anchor_generator in dygraph."},
  {"layer_norm", (PyCFunction)(void(*)(void))imperative_layer_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for layer_norm in dygraph."},
  {"merge_selected_rows", (PyCFunction)(void(*)(void))imperative_merge_selected_rows, METH_VARARGS | METH_KEYWORDS, "C++ interface function for merge_selected_rows in dygraph."},
  {"acosh", (PyCFunction)(void(*)(void))imperative_acosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acosh in dygraph."},
  {"stft", (PyCFunction)(void(*)(void))imperative_stft, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stft in dygraph."},
  {"less_equal", (PyCFunction)(void(*)(void))imperative_less_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_equal in dygraph."},
  {"rnn", (PyCFunction)(void(*)(void))imperative_rnn, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rnn in dygraph."},
  {"fusion_lstm", (PyCFunction)(void(*)(void))imperative_fusion_lstm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_lstm in dygraph."},
  {"lars_momentum", (PyCFunction)(void(*)(void))imperative_lars_momentum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lars_momentum in dygraph."},
  {"hard_sigmoid", (PyCFunction)(void(*)(void))imperative_hard_sigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hard_sigmoid in dygraph."},
  {"hard_sigmoid_", (PyCFunction)(void(*)(void))imperative_hard_sigmoid_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hard_sigmoid_ in dygraph."},
  {"isnan", (PyCFunction)(void(*)(void))imperative_isnan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isnan in dygraph."},
  {"elementwise_floordiv", (PyCFunction)(void(*)(void))imperative_elementwise_floordiv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_floordiv in dygraph."},
  {"correlation", (PyCFunction)(void(*)(void))imperative_correlation, METH_VARARGS | METH_KEYWORDS, "C++ interface function for correlation in dygraph."},
  {"histogram", (PyCFunction)(void(*)(void))imperative_histogram, METH_VARARGS | METH_KEYWORDS, "C++ interface function for histogram in dygraph."},
  {"gather_tree", (PyCFunction)(void(*)(void))imperative_gather_tree, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather_tree in dygraph."},
  {"nanmedian", (PyCFunction)(void(*)(void))imperative_nanmedian, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nanmedian in dygraph."},
  {"segment_pool", (PyCFunction)(void(*)(void))imperative_segment_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for segment_pool in dygraph."},
  {"fusion_repeated_fc_relu", (PyCFunction)(void(*)(void))imperative_fusion_repeated_fc_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_repeated_fc_relu in dygraph."},
  {"sync_batch_norm", (PyCFunction)(void(*)(void))imperative_sync_batch_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sync_batch_norm in dygraph."},
  {"nop", (PyCFunction)(void(*)(void))imperative_nop, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nop in dygraph."},
  {"fused_attention", (PyCFunction)(void(*)(void))imperative_fused_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_attention in dygraph."},
  {"filter_by_instag", (PyCFunction)(void(*)(void))imperative_filter_by_instag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for filter_by_instag in dygraph."},
  {"expand_as_v2", (PyCFunction)(void(*)(void))imperative_expand_as_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand_as_v2 in dygraph."},
  {"diag_v2", (PyCFunction)(void(*)(void))imperative_diag_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag_v2 in dygraph."},
  {"pull_box_sparse", (PyCFunction)(void(*)(void))imperative_pull_box_sparse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pull_box_sparse in dygraph."},
  {"nll_loss", (PyCFunction)(void(*)(void))imperative_nll_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nll_loss in dygraph."},
  {"dot", (PyCFunction)(void(*)(void))imperative_dot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dot in dygraph."},
  {"scale", (PyCFunction)(void(*)(void))imperative_scale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scale in dygraph."},
  {"scale_", (PyCFunction)(void(*)(void))imperative_scale_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scale_ in dygraph."},
  {"shuffle_batch", (PyCFunction)(void(*)(void))imperative_shuffle_batch, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shuffle_batch in dygraph."},
  {"diag", (PyCFunction)(void(*)(void))imperative_diag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag in dygraph."},
  {"multiplex", (PyCFunction)(void(*)(void))imperative_multiplex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiplex in dygraph."},
  {"leaky_relu", (PyCFunction)(void(*)(void))imperative_leaky_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu in dygraph."},
  {"leaky_relu_", (PyCFunction)(void(*)(void))imperative_leaky_relu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu_ in dygraph."},
  {"allclose", (PyCFunction)(void(*)(void))imperative_allclose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for allclose in dygraph."},
  {"adamw", (PyCFunction)(void(*)(void))imperative_adamw, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adamw in dygraph."},
  {"elementwise_pow", (PyCFunction)(void(*)(void))imperative_elementwise_pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_pow in dygraph."},
  {"prior_box", (PyCFunction)(void(*)(void))imperative_prior_box, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prior_box in dygraph."},
  {"p_norm", (PyCFunction)(void(*)(void))imperative_p_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for p_norm in dygraph."},
  {"c_concat", (PyCFunction)(void(*)(void))imperative_c_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_concat in dygraph."},
  {"fused_gate_attention", (PyCFunction)(void(*)(void))imperative_fused_gate_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_gate_attention in dygraph."},
  {"unique_consecutive", (PyCFunction)(void(*)(void))imperative_unique_consecutive, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unique_consecutive in dygraph."},
  {"lod_reset", (PyCFunction)(void(*)(void))imperative_lod_reset, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lod_reset in dygraph."},
  {"lod_reset_", (PyCFunction)(void(*)(void))imperative_lod_reset_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lod_reset_ in dygraph."},
  {"pad", (PyCFunction)(void(*)(void))imperative_pad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad in dygraph."},
  {"sequence_conv", (PyCFunction)(void(*)(void))imperative_sequence_conv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_conv in dygraph."},
  {"set_value", (PyCFunction)(void(*)(void))imperative_set_value, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value in dygraph."},
  {"set_value_", (PyCFunction)(void(*)(void))imperative_set_value_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for set_value_ in dygraph."},
  {"log10", (PyCFunction)(void(*)(void))imperative_log10, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log10 in dygraph."},
  {"nms", (PyCFunction)(void(*)(void))imperative_nms, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nms in dygraph."},
  {"bitwise_xor", (PyCFunction)(void(*)(void))imperative_bitwise_xor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_xor in dygraph."},
  {"center_loss", (PyCFunction)(void(*)(void))imperative_center_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for center_loss in dygraph."},
  {"randint", (PyCFunction)(void(*)(void))imperative_randint, METH_VARARGS | METH_KEYWORDS, "C++ interface function for randint in dygraph."},
  {"attention_lstm", (PyCFunction)(void(*)(void))imperative_attention_lstm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for attention_lstm in dygraph."},
  {"uniform_random", (PyCFunction)(void(*)(void))imperative_uniform_random, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_random in dygraph."},
  {"slice", (PyCFunction)(void(*)(void))imperative_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for slice in dygraph."},
  {"dequantize", (PyCFunction)(void(*)(void))imperative_dequantize, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dequantize in dygraph."},
  {"meshgrid", (PyCFunction)(void(*)(void))imperative_meshgrid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for meshgrid in dygraph."},
  {"hard_swish", (PyCFunction)(void(*)(void))imperative_hard_swish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hard_swish in dygraph."},
  {"sin", (PyCFunction)(void(*)(void))imperative_sin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sin in dygraph."},
  {"mean_iou", (PyCFunction)(void(*)(void))imperative_mean_iou, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mean_iou in dygraph."},
  {"pad2d", (PyCFunction)(void(*)(void))imperative_pad2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad2d in dygraph."},
  {"inverse", (PyCFunction)(void(*)(void))imperative_inverse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for inverse in dygraph."},
  {"spectral_norm", (PyCFunction)(void(*)(void))imperative_spectral_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for spectral_norm in dygraph."},
  {"shuffle_channel", (PyCFunction)(void(*)(void))imperative_shuffle_channel, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shuffle_channel in dygraph."},
  {"multi_gru", (PyCFunction)(void(*)(void))imperative_multi_gru, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multi_gru in dygraph."},
  {"send_v2", (PyCFunction)(void(*)(void))imperative_send_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for send_v2 in dygraph."},
  {"psroi_pool", (PyCFunction)(void(*)(void))imperative_psroi_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for psroi_pool in dygraph."},
  {"seed", (PyCFunction)(void(*)(void))imperative_seed, METH_VARARGS | METH_KEYWORDS, "C++ interface function for seed in dygraph."},
  {"ceil", (PyCFunction)(void(*)(void))imperative_ceil, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ceil in dygraph."},
  {"ceil_", (PyCFunction)(void(*)(void))imperative_ceil_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ceil_ in dygraph."},
  {"eig", (PyCFunction)(void(*)(void))imperative_eig, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eig in dygraph."},
  {"reduce_min", (PyCFunction)(void(*)(void))imperative_reduce_min, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_min in dygraph."},
  {"cos", (PyCFunction)(void(*)(void))imperative_cos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cos in dygraph."},
  {"cudnn_lstm", (PyCFunction)(void(*)(void))imperative_cudnn_lstm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cudnn_lstm in dygraph."},
  {"random_routing", (PyCFunction)(void(*)(void))imperative_random_routing, METH_VARARGS | METH_KEYWORDS, "C++ interface function for random_routing in dygraph."},
  {"random_routing_", (PyCFunction)(void(*)(void))imperative_random_routing_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for random_routing_ in dygraph."},
  {"reduce_sum", (PyCFunction)(void(*)(void))imperative_reduce_sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_sum in dygraph."},
  {"digamma", (PyCFunction)(void(*)(void))imperative_digamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for digamma in dygraph."},
  {"quantize_linear", (PyCFunction)(void(*)(void))imperative_quantize_linear, METH_VARARGS | METH_KEYWORDS, "C++ interface function for quantize_linear in dygraph."},
  {"assign_value", (PyCFunction)(void(*)(void))imperative_assign_value, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_value in dygraph."},
  {"increment", (PyCFunction)(void(*)(void))imperative_increment, METH_VARARGS | METH_KEYWORDS, "C++ interface function for increment in dygraph."},
  {"logspace", (PyCFunction)(void(*)(void))imperative_logspace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logspace in dygraph."},
  {"tdm_sampler", (PyCFunction)(void(*)(void))imperative_tdm_sampler, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tdm_sampler in dygraph."},
  {"fused_softmax_mask", (PyCFunction)(void(*)(void))imperative_fused_softmax_mask, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_softmax_mask in dygraph."},
  {"sequence_reverse", (PyCFunction)(void(*)(void))imperative_sequence_reverse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_reverse in dygraph."},
  {"eigvalsh", (PyCFunction)(void(*)(void))imperative_eigvalsh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigvalsh in dygraph."},
  {"diagonal", (PyCFunction)(void(*)(void))imperative_diagonal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diagonal in dygraph."},
  {"trunc", (PyCFunction)(void(*)(void))imperative_trunc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trunc in dygraph."},
  {"log2", (PyCFunction)(void(*)(void))imperative_log2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log2 in dygraph."},
  {"marker", (PyCFunction)(void(*)(void))imperative_marker, METH_VARARGS | METH_KEYWORDS, "C++ interface function for marker in dygraph."},
  {"tanh", (PyCFunction)(void(*)(void))imperative_tanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh in dygraph."},
  {"tanh_", (PyCFunction)(void(*)(void))imperative_tanh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_ in dygraph."},
  {"yolov3_loss", (PyCFunction)(void(*)(void))imperative_yolov3_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolov3_loss in dygraph."},
  {"graph_send_recv", (PyCFunction)(void(*)(void))imperative_graph_send_recv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_send_recv in dygraph."},
  {"accuracy", (PyCFunction)(void(*)(void))imperative_accuracy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for accuracy in dygraph."},
  {"atan", (PyCFunction)(void(*)(void))imperative_atan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan in dygraph."},
  {"less_than", (PyCFunction)(void(*)(void))imperative_less_than, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_than in dygraph."},
  {"reduce_amax", (PyCFunction)(void(*)(void))imperative_reduce_amax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_amax in dygraph."},
  {"unsqueeze", (PyCFunction)(void(*)(void))imperative_unsqueeze, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze in dygraph."},
  {"crf_decoding", (PyCFunction)(void(*)(void))imperative_crf_decoding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for crf_decoding in dygraph."},
  {"global_gather", (PyCFunction)(void(*)(void))imperative_global_gather, METH_VARARGS | METH_KEYWORDS, "C++ interface function for global_gather in dygraph."},
  {"merged_adam", (PyCFunction)(void(*)(void))imperative_merged_adam, METH_VARARGS | METH_KEYWORDS, "C++ interface function for merged_adam in dygraph."},
  {"lerp", (PyCFunction)(void(*)(void))imperative_lerp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lerp in dygraph."},
  {"lerp_", (PyCFunction)(void(*)(void))imperative_lerp_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lerp_ in dygraph."},
  {"c_allreduce_prod", (PyCFunction)(void(*)(void))imperative_c_allreduce_prod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_prod in dygraph."},
  {"c_allreduce_prod_", (PyCFunction)(void(*)(void))imperative_c_allreduce_prod_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_prod_ in dygraph."},
  {"log_softmax", (PyCFunction)(void(*)(void))imperative_log_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_softmax in dygraph."},
  {"ftrl", (PyCFunction)(void(*)(void))imperative_ftrl, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ftrl in dygraph."},
  {"matrix_nms", (PyCFunction)(void(*)(void))imperative_matrix_nms, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_nms in dygraph."},
  {"top_k_v2", (PyCFunction)(void(*)(void))imperative_top_k_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for top_k_v2 in dygraph."},
  {"cast", (PyCFunction)(void(*)(void))imperative_cast, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cast in dygraph."},
  {"tanh_shrink", (PyCFunction)(void(*)(void))imperative_tanh_shrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_shrink in dygraph."},
  {"hard_shrink", (PyCFunction)(void(*)(void))imperative_hard_shrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hard_shrink in dygraph."},
  {"logit", (PyCFunction)(void(*)(void))imperative_logit, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logit in dygraph."},
  {"multiclass_nms", (PyCFunction)(void(*)(void))imperative_multiclass_nms, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiclass_nms in dygraph."},
  {"c_broadcast", (PyCFunction)(void(*)(void))imperative_c_broadcast, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_broadcast in dygraph."},
  {"fusion_transpose_flatten_concat", (PyCFunction)(void(*)(void))imperative_fusion_transpose_flatten_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_transpose_flatten_concat in dygraph."},
  {"sequence_unpad", (PyCFunction)(void(*)(void))imperative_sequence_unpad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_unpad in dygraph."},
  {"fused_elemwise_add_activation", (PyCFunction)(void(*)(void))imperative_fused_elemwise_add_activation, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_elemwise_add_activation in dygraph."},
  {"pull_sparse_v2", (PyCFunction)(void(*)(void))imperative_pull_sparse_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pull_sparse_v2 in dygraph."},
  {"einsum", (PyCFunction)(void(*)(void))imperative_einsum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for einsum in dygraph."},
  {"frobenius_norm", (PyCFunction)(void(*)(void))imperative_frobenius_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for frobenius_norm in dygraph."},
  {"crop", (PyCFunction)(void(*)(void))imperative_crop, METH_VARARGS | METH_KEYWORDS, "C++ interface function for crop in dygraph."},
  {"cross_entropy2", (PyCFunction)(void(*)(void))imperative_cross_entropy2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross_entropy2 in dygraph."},
  {"skip_layernorm", (PyCFunction)(void(*)(void))imperative_skip_layernorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for skip_layernorm in dygraph."},
  {"tdm_child", (PyCFunction)(void(*)(void))imperative_tdm_child, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tdm_child in dygraph."},
  {"fused_embedding_seq_pool", (PyCFunction)(void(*)(void))imperative_fused_embedding_seq_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_embedding_seq_pool in dygraph."},
  {"kthvalue", (PyCFunction)(void(*)(void))imperative_kthvalue, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kthvalue in dygraph."},
  {"erf", (PyCFunction)(void(*)(void))imperative_erf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erf in dygraph."},
  {"yolo_box_post", (PyCFunction)(void(*)(void))imperative_yolo_box_post, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_box_post in dygraph."},
  {"conv2d_inception_fusion", (PyCFunction)(void(*)(void))imperative_conv2d_inception_fusion, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d_inception_fusion in dygraph."},
  {"logsumexp", (PyCFunction)(void(*)(void))imperative_logsumexp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsumexp in dygraph."},
  {"trilinear_interp", (PyCFunction)(void(*)(void))imperative_trilinear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trilinear_interp in dygraph."},
  {"fusion_seqpool_concat", (PyCFunction)(void(*)(void))imperative_fusion_seqpool_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqpool_concat in dygraph."},
  {"alloc_float_status", (PyCFunction)(void(*)(void))imperative_alloc_float_status, METH_VARARGS | METH_KEYWORDS, "C++ interface function for alloc_float_status in dygraph."},
  {"sequence_concat", (PyCFunction)(void(*)(void))imperative_sequence_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_concat in dygraph."},
  {"fusion_seqpool_cvm_concat", (PyCFunction)(void(*)(void))imperative_fusion_seqpool_cvm_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqpool_cvm_concat in dygraph."},
  {"unpool3d", (PyCFunction)(void(*)(void))imperative_unpool3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool3d in dygraph."},
  {"similarity_focus", (PyCFunction)(void(*)(void))imperative_similarity_focus, METH_VARARGS | METH_KEYWORDS, "C++ interface function for similarity_focus in dygraph."},
  {"c_allreduce_max", (PyCFunction)(void(*)(void))imperative_c_allreduce_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_max in dygraph."},
  {"c_allreduce_max_", (PyCFunction)(void(*)(void))imperative_c_allreduce_max_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allreduce_max_ in dygraph."},
  {"argsort", (PyCFunction)(void(*)(void))imperative_argsort, METH_VARARGS | METH_KEYWORDS, "C++ interface function for argsort in dygraph."},
  {"sequence_expand", (PyCFunction)(void(*)(void))imperative_sequence_expand, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_expand in dygraph."},
  {"fused_bn_add_activation", (PyCFunction)(void(*)(void))imperative_fused_bn_add_activation, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bn_add_activation in dygraph."},
  {"sgd", (PyCFunction)(void(*)(void))imperative_sgd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sgd in dygraph."},
  {"exponential", (PyCFunction)(void(*)(void))imperative_exponential, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exponential in dygraph."},
  {"exponential_", (PyCFunction)(void(*)(void))imperative_exponential_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exponential_ in dygraph."},
  {"bilinear_interp_v2", (PyCFunction)(void(*)(void))imperative_bilinear_interp_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear_interp_v2 in dygraph."},
  {"atanh", (PyCFunction)(void(*)(void))imperative_atanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atanh in dygraph."},
  {"clip", (PyCFunction)(void(*)(void))imperative_clip, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip in dygraph."},
  {"clip_", (PyCFunction)(void(*)(void))imperative_clip_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_ in dygraph."},
  {"deformable_conv_v1", (PyCFunction)(void(*)(void))imperative_deformable_conv_v1, METH_VARARGS | METH_KEYWORDS, "C++ interface function for deformable_conv_v1 in dygraph."},
  {"hinge_loss", (PyCFunction)(void(*)(void))imperative_hinge_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hinge_loss in dygraph."},
  {"determinant", (PyCFunction)(void(*)(void))imperative_determinant, METH_VARARGS | METH_KEYWORDS, "C++ interface function for determinant in dygraph."},
  {"conv2d_transpose", (PyCFunction)(void(*)(void))imperative_conv2d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d_transpose in dygraph."},
  {"memcpy_d2h", (PyCFunction)(void(*)(void))imperative_memcpy_d2h, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy_d2h in dygraph."},
  {"softsign", (PyCFunction)(void(*)(void))imperative_softsign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softsign in dygraph."},
  {"fake_quantize_dequantize_abs_max", (PyCFunction)(void(*)(void))imperative_fake_quantize_dequantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_dequantize_abs_max in dygraph."},
  {"broadcast_tensors", (PyCFunction)(void(*)(void))imperative_broadcast_tensors, METH_VARARGS | METH_KEYWORDS, "C++ interface function for broadcast_tensors in dygraph."},
  {"cholesky_solve", (PyCFunction)(void(*)(void))imperative_cholesky_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cholesky_solve in dygraph."},
  {"grid_sampler", (PyCFunction)(void(*)(void))imperative_grid_sampler, METH_VARARGS | METH_KEYWORDS, "C++ interface function for grid_sampler in dygraph."},
  {"fft_c2r", (PyCFunction)(void(*)(void))imperative_fft_c2r, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_c2r in dygraph."},
  {"pyramid_hash", (PyCFunction)(void(*)(void))imperative_pyramid_hash, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pyramid_hash in dygraph."},
  {"fake_quantize_dequantize_moving_average_abs_max", (PyCFunction)(void(*)(void))imperative_fake_quantize_dequantize_moving_average_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_dequantize_moving_average_abs_max in dygraph."},
  {"multi_dot", (PyCFunction)(void(*)(void))imperative_multi_dot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multi_dot in dygraph."},
  {"sequence_pool", (PyCFunction)(void(*)(void))imperative_sequence_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_pool in dygraph."},
  {"broadcast", (PyCFunction)(void(*)(void))imperative_broadcast, METH_VARARGS | METH_KEYWORDS, "C++ interface function for broadcast in dygraph."},
  {"transpose", (PyCFunction)(void(*)(void))imperative_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transpose in dygraph."},
  {"top_k", (PyCFunction)(void(*)(void))imperative_top_k, METH_VARARGS | METH_KEYWORDS, "C++ interface function for top_k in dygraph."},
  {"renorm", (PyCFunction)(void(*)(void))imperative_renorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for renorm in dygraph."},
  {"pixel_unshuffle", (PyCFunction)(void(*)(void))imperative_pixel_unshuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pixel_unshuffle in dygraph."},
  {"take_along_axis", (PyCFunction)(void(*)(void))imperative_take_along_axis, METH_VARARGS | METH_KEYWORDS, "C++ interface function for take_along_axis in dygraph."},
  {"dist", (PyCFunction)(void(*)(void))imperative_dist, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dist in dygraph."},
  {"affine_grid", (PyCFunction)(void(*)(void))imperative_affine_grid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_grid in dygraph."},
  {"gaussian_random_batch_size_like", (PyCFunction)(void(*)(void))imperative_gaussian_random_batch_size_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian_random_batch_size_like in dygraph."},
  {"fake_channel_wise_dequantize_max_abs", (PyCFunction)(void(*)(void))imperative_fake_channel_wise_dequantize_max_abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_channel_wise_dequantize_max_abs in dygraph."},
  {"reciprocal", (PyCFunction)(void(*)(void))imperative_reciprocal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal in dygraph."},
  {"reciprocal_", (PyCFunction)(void(*)(void))imperative_reciprocal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal_ in dygraph."},
  {"sequence_mask", (PyCFunction)(void(*)(void))imperative_sequence_mask, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_mask in dygraph."},
  {"prune_gate_by_capacity", (PyCFunction)(void(*)(void))imperative_prune_gate_by_capacity, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prune_gate_by_capacity in dygraph."},
  {"fill_diagonal_tensor", (PyCFunction)(void(*)(void))imperative_fill_diagonal_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_tensor in dygraph."},
  {"fill_diagonal_tensor_", (PyCFunction)(void(*)(void))imperative_fill_diagonal_tensor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_tensor_ in dygraph."},
  {"abs", (PyCFunction)(void(*)(void))imperative_abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for abs in dygraph."},
  {"partial_concat", (PyCFunction)(void(*)(void))imperative_partial_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for partial_concat in dygraph."},
  {"elu", (PyCFunction)(void(*)(void))imperative_elu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu in dygraph."},
  {"elu_", (PyCFunction)(void(*)(void))imperative_elu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu_ in dygraph."},
  {"index_select", (PyCFunction)(void(*)(void))imperative_index_select, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_select in dygraph."},
  {"row_conv", (PyCFunction)(void(*)(void))imperative_row_conv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for row_conv in dygraph."},
  {"cross", (PyCFunction)(void(*)(void))imperative_cross, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross in dygraph."},
  {"elementwise_mul", (PyCFunction)(void(*)(void))imperative_elementwise_mul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_mul in dygraph."},
  {"decayed_adagrad", (PyCFunction)(void(*)(void))imperative_decayed_adagrad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for decayed_adagrad in dygraph."},
  {"bipartite_match", (PyCFunction)(void(*)(void))imperative_bipartite_match, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bipartite_match in dygraph."},
  {"run_program", (PyCFunction)(void(*)(void))imperative_run_program, METH_VARARGS | METH_KEYWORDS, "C++ interface function for run_program in dygraph."},
  {"fake_quantize_moving_average_abs_max", (PyCFunction)(void(*)(void))imperative_fake_quantize_moving_average_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_moving_average_abs_max in dygraph."},
  {"fused_multi_transformer_int8", (PyCFunction)(void(*)(void))imperative_fused_multi_transformer_int8, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_multi_transformer_int8 in dygraph."},
  {"mine_hard_examples", (PyCFunction)(void(*)(void))imperative_mine_hard_examples, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mine_hard_examples in dygraph."},
  {"target_assign", (PyCFunction)(void(*)(void))imperative_target_assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for target_assign in dygraph."},
  {"lstm", (PyCFunction)(void(*)(void))imperative_lstm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstm in dygraph."},
  {"assign_pos", (PyCFunction)(void(*)(void))imperative_assign_pos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_pos in dygraph."},
  {"truncated_gaussian_random", (PyCFunction)(void(*)(void))imperative_truncated_gaussian_random, METH_VARARGS | METH_KEYWORDS, "C++ interface function for truncated_gaussian_random in dygraph."},
  {"match_matrix_tensor", (PyCFunction)(void(*)(void))imperative_match_matrix_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for match_matrix_tensor in dygraph."},
  {"elementwise_div", (PyCFunction)(void(*)(void))imperative_elementwise_div, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_div in dygraph."},
  {"kldiv_loss", (PyCFunction)(void(*)(void))imperative_kldiv_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kldiv_loss in dygraph."},
  {"cumsum", (PyCFunction)(void(*)(void))imperative_cumsum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumsum in dygraph."},
  {"sum", (PyCFunction)(void(*)(void))imperative_sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sum in dygraph."},
  {"sum_", (PyCFunction)(void(*)(void))imperative_sum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sum_ in dygraph."},
  {"proximal_adagrad", (PyCFunction)(void(*)(void))imperative_proximal_adagrad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for proximal_adagrad in dygraph."},
  {"update_loss_scaling", (PyCFunction)(void(*)(void))imperative_update_loss_scaling, METH_VARARGS | METH_KEYWORDS, "C++ interface function for update_loss_scaling in dygraph."},
  {"shard_index", (PyCFunction)(void(*)(void))imperative_shard_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shard_index in dygraph."},
  {"selu", (PyCFunction)(void(*)(void))imperative_selu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for selu in dygraph."},
  {"gumbel_softmax", (PyCFunction)(void(*)(void))imperative_gumbel_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gumbel_softmax in dygraph."},
  {"mean", (PyCFunction)(void(*)(void))imperative_mean, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mean in dygraph."},
  {"sequence_pad", (PyCFunction)(void(*)(void))imperative_sequence_pad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_pad in dygraph."},
  {"tree_conv", (PyCFunction)(void(*)(void))imperative_tree_conv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tree_conv in dygraph."},
  {"assign", (PyCFunction)(void(*)(void))imperative_assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign in dygraph."},
  {"assign_", (PyCFunction)(void(*)(void))imperative_assign_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_ in dygraph."},
  {"flatten_contiguous_range", (PyCFunction)(void(*)(void))imperative_flatten_contiguous_range, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten_contiguous_range in dygraph."},
  {"flatten_contiguous_range_", (PyCFunction)(void(*)(void))imperative_flatten_contiguous_range_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten_contiguous_range_ in dygraph."},
  {"tril_triu", (PyCFunction)(void(*)(void))imperative_tril_triu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tril_triu in dygraph."},
  {"celu", (PyCFunction)(void(*)(void))imperative_celu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for celu in dygraph."},
  {"celu_", (PyCFunction)(void(*)(void))imperative_celu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for celu_ in dygraph."},
  {"reduce_mean", (PyCFunction)(void(*)(void))imperative_reduce_mean, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_mean in dygraph."},
  {"brelu", (PyCFunction)(void(*)(void))imperative_brelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for brelu in dygraph."},
  {"sinh", (PyCFunction)(void(*)(void))imperative_sinh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sinh in dygraph."},
  {"rank_loss", (PyCFunction)(void(*)(void))imperative_rank_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rank_loss in dygraph."},
  {"reduce_max", (PyCFunction)(void(*)(void))imperative_reduce_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_max in dygraph."},
  {"fusion_gru", (PyCFunction)(void(*)(void))imperative_fusion_gru, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_gru in dygraph."},
  {"fill_zeros_like2", (PyCFunction)(void(*)(void))imperative_fill_zeros_like2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_zeros_like2 in dygraph."},
  {"expm1", (PyCFunction)(void(*)(void))imperative_expm1, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expm1 in dygraph."},
  {"expm1_", (PyCFunction)(void(*)(void))imperative_expm1_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expm1_ in dygraph."},
  {"squared_l2_norm", (PyCFunction)(void(*)(void))imperative_squared_l2_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squared_l2_norm in dygraph."},
  {"elementwise_sub", (PyCFunction)(void(*)(void))imperative_elementwise_sub, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_sub in dygraph."},
  {"elementwise_sub_", (PyCFunction)(void(*)(void))imperative_elementwise_sub_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_sub_ in dygraph."},
  {"margin_rank_loss", (PyCFunction)(void(*)(void))imperative_margin_rank_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for margin_rank_loss in dygraph."},
  {"faster_tokenizer", (PyCFunction)(void(*)(void))imperative_faster_tokenizer, METH_VARARGS | METH_KEYWORDS, "C++ interface function for faster_tokenizer in dygraph."},
  {"c_reduce_max", (PyCFunction)(void(*)(void))imperative_c_reduce_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_reduce_max in dygraph."},
  {"c_identity", (PyCFunction)(void(*)(void))imperative_c_identity, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_identity in dygraph."},
  {"relu", (PyCFunction)(void(*)(void))imperative_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu in dygraph."},
  {"relu_", (PyCFunction)(void(*)(void))imperative_relu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu_ in dygraph."},
  {"is_empty", (PyCFunction)(void(*)(void))imperative_is_empty, METH_VARARGS | METH_KEYWORDS, "C++ interface function for is_empty in dygraph."},
  {"reduce_all", (PyCFunction)(void(*)(void))imperative_reduce_all, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_all in dygraph."},
  {"edit_distance", (PyCFunction)(void(*)(void))imperative_edit_distance, METH_VARARGS | METH_KEYWORDS, "C++ interface function for edit_distance in dygraph."},
  {"distributed_lookup_table", (PyCFunction)(void(*)(void))imperative_distributed_lookup_table, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distributed_lookup_table in dygraph."},
  {"tril_indices", (PyCFunction)(void(*)(void))imperative_tril_indices, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tril_indices in dygraph."},
  {"bmm", (PyCFunction)(void(*)(void))imperative_bmm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bmm in dygraph."},
  {"yolo_box", (PyCFunction)(void(*)(void))imperative_yolo_box, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_box in dygraph."},
  {"soft_relu", (PyCFunction)(void(*)(void))imperative_soft_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for soft_relu in dygraph."},
  {"soft_relu_", (PyCFunction)(void(*)(void))imperative_soft_relu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for soft_relu_ in dygraph."},
  {"density_prior_box", (PyCFunction)(void(*)(void))imperative_density_prior_box, METH_VARARGS | METH_KEYWORDS, "C++ interface function for density_prior_box in dygraph."},
  {"swish", (PyCFunction)(void(*)(void))imperative_swish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for swish in dygraph."},
  {"eye", (PyCFunction)(void(*)(void))imperative_eye, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eye in dygraph."},
  {"cross_entropy", (PyCFunction)(void(*)(void))imperative_cross_entropy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross_entropy in dygraph."},
  {"dpsgd", (PyCFunction)(void(*)(void))imperative_dpsgd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dpsgd in dygraph."},
  {"cholesky", (PyCFunction)(void(*)(void))imperative_cholesky, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cholesky in dygraph."},
  {"batch_fc", (PyCFunction)(void(*)(void))imperative_batch_fc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for batch_fc in dygraph."},
  {"nearest_interp", (PyCFunction)(void(*)(void))imperative_nearest_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nearest_interp in dygraph."},
  {"gather", (PyCFunction)(void(*)(void))imperative_gather, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather in dygraph."},
  {"trilinear_interp_v2", (PyCFunction)(void(*)(void))imperative_trilinear_interp_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trilinear_interp_v2 in dygraph."},
  {"box_clip", (PyCFunction)(void(*)(void))imperative_box_clip, METH_VARARGS | METH_KEYWORDS, "C++ interface function for box_clip in dygraph."},
  {"c_allgather", (PyCFunction)(void(*)(void))imperative_c_allgather, METH_VARARGS | METH_KEYWORDS, "C++ interface function for c_allgather in dygraph."},
  {"isnan_v2", (PyCFunction)(void(*)(void))imperative_isnan_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isnan_v2 in dygraph."},
  {"lu", (PyCFunction)(void(*)(void))imperative_lu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu in dygraph."},
  {"lu_", (PyCFunction)(void(*)(void))imperative_lu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_ in dygraph."},
  {"softmax", (PyCFunction)(void(*)(void))imperative_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax in dygraph."},
  {"softmax_", (PyCFunction)(void(*)(void))imperative_softmax_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax_ in dygraph."},
  {"conv2d_fusion", (PyCFunction)(void(*)(void))imperative_conv2d_fusion, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d_fusion in dygraph."},
  {"fused_batch_norm_act", (PyCFunction)(void(*)(void))imperative_fused_batch_norm_act, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_batch_norm_act in dygraph."},
  {"get_float_status", (PyCFunction)(void(*)(void))imperative_get_float_status, METH_VARARGS | METH_KEYWORDS, "C++ interface function for get_float_status in dygraph."},
  {"index_sample", (PyCFunction)(void(*)(void))imperative_index_sample, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_sample in dygraph."},
  {"elementwise_min", (PyCFunction)(void(*)(void))imperative_elementwise_min, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_min in dygraph."},
  {"logical_not", (PyCFunction)(void(*)(void))imperative_logical_not, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_not in dygraph."},
  {"erfinv", (PyCFunction)(void(*)(void))imperative_erfinv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erfinv in dygraph."},
  {"erfinv_", (PyCFunction)(void(*)(void))imperative_erfinv_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erfinv_ in dygraph."},
  {"collect_fpn_proposals", (PyCFunction)(void(*)(void))imperative_collect_fpn_proposals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for collect_fpn_proposals in dygraph."},
  {"pixel_shuffle", (PyCFunction)(void(*)(void))imperative_pixel_shuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pixel_shuffle in dygraph."},
  {"thresholded_relu", (PyCFunction)(void(*)(void))imperative_thresholded_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for thresholded_relu in dygraph."},
  {"polygon_box_transform", (PyCFunction)(void(*)(void))imperative_polygon_box_transform, METH_VARARGS | METH_KEYWORDS, "C++ interface function for polygon_box_transform in dygraph."},
  {"lookup_table_dequant", (PyCFunction)(void(*)(void))imperative_lookup_table_dequant, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lookup_table_dequant in dygraph."},
  {"warpctc", (PyCFunction)(void(*)(void))imperative_warpctc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for warpctc in dygraph."},
  {"elementwise_heaviside", (PyCFunction)(void(*)(void))imperative_elementwise_heaviside, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_heaviside in dygraph."},
  {"fake_channel_wise_quantize_abs_max", (PyCFunction)(void(*)(void))imperative_fake_channel_wise_quantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_channel_wise_quantize_abs_max in dygraph."},
  {"dequantize_abs_max", (PyCFunction)(void(*)(void))imperative_dequantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dequantize_abs_max in dygraph."},
  {"svd", (PyCFunction)(void(*)(void))imperative_svd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for svd in dygraph."},
  {"flip", (PyCFunction)(void(*)(void))imperative_flip, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flip in dygraph."},
  {"quantize", (PyCFunction)(void(*)(void))imperative_quantize, METH_VARARGS | METH_KEYWORDS, "C++ interface function for quantize in dygraph."},
  {nullptr,nullptr,0,nullptr}};

inline void BindOpFunctions(pybind11::module *module) {
  auto m = module->def_submodule("ops");
  if (PyModule_AddFunctions(m.ptr(), ExtestMethods) < 0) {
    PADDLE_THROW(platform::errors::Fatal ("Add functions to core.ops failed!"));
  }

  InitOpsAttrTypeMap();}

} // namespace pybind
} // namespace paddle
