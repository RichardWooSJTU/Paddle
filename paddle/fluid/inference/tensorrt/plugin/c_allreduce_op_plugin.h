// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <stdio.h>
#include <string>
#include <vector>
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

enum ReduceType { kRedSum, kRedMax, kRedMin, kRedProd };

class CAllReducePlugin : public PluginTensorRT {
 private:
  int ring_id_;
  bool use_calc_stream_;
  ReduceType red_type_;

 protected:
  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return getBaseSerializationSize() + SerializedSize(ring_id_)
	    + SerializedSize(use_calc_stream_) + SerializedSize(red_type_);
  }

  // TRT will call this func  to serialize the configuration of TRT
  // It should not be called by users.
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
    SerializeValue(&buffer, ring_id_);
    SerializeValue(&buffer, use_calc_stream_);
    SerializeValue(&buffer, red_type_);
  }

 public:
  explicit CAllReducePlugin(const int ring_id, const bool use_calc_stream,
		  ReduceType red_type, const bool with_fp16)
      : ring_id_(ring_id), use_calc_stream_(use_calc_stream), red_type_(red_type) {
    with_fp16_ = with_fp16;
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  CAllReducePlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &ring_id_);
    DeserializeValue(&serialData, &serialLength, &use_calc_stream_);
    DeserializeValue(&serialData, &serialLength, &red_type_);
  }

  ~CAllReducePlugin() {}
  CAllReducePlugin* clone() const TRT_NOEXCEPT override {
    return new CAllReducePlugin(ring_id_, use_calc_stream_,
		    red_type_, with_fp16_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "c_allreduce_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;
  bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format)
      const TRT_NOEXCEPT override;
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nbInputDims) TRT_NOEXCEPT override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize, const void* const* inputs, void** outputs,
#else
  int enqueue(int batchSize, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;
};

class CAllReducePluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "c_allreduce_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    return new CAllReducePlugin(serial_data, serial_length);
  }
};

REGISTER_TRT_PLUGIN_V2(CAllReducePluginCreator);


class CAllReducePluginDynamic : public DynamicPluginTensorRT {
 protect:
   int ring_id_;
   bool use_calc_stream_;
   ReduceType red_type_;

 public:
  explicit CAllReducePluginDynamic(const int ring_id,
		  const bool use_calc_stream,
		  const ReduceType red_type,
		  const bool with_fp16) {
    ring_id_ = ring_id;
    use_calc_stream_ = use_calc_stream;
    red_type_ = red_type;
    with_fp16_ = with_fp16;
  }
  CAllReducePluginDynamic(void const* serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &ring_id_);
    DeserializeValue(&serialData, &serialLength, &use_calc_stream_);
    DeserializeValue(&serialData, &serialLength, &red_type_);
    DeserializeValue(&serialData, &serialLength, &with_fp16_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new CAllReducePluginDynamic(ring_id_, use_calc_stream_,
		    red_type_, with_fp16_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "c_allreduce_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override { return 0; };

  size_t getSerializationSize() const TRT_NOEXCEPT override;
  void serialize(void* buffer) const TRT_NOEXCEPT override;

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) TRT_NOEXCEPT override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* inputTypes,
      int nbInputs) const TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }
};

class CAllReducePluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "c_allreduce_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    auto plugin = new CAllReducePluginDynamic(serial_data, serial_length);
    return plugin;
  }
};

REGISTER_TRT_PLUGIN_V2(CAllReducePluginDynamicCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
