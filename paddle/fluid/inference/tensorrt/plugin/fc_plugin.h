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

#include <algorithm>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "paddle/fluid/platform/dynload/cublasLt.h"

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"


namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)
class FcPluginDynamic : public DynamicPluginTensorRT {
 public:
    explicit FcPluginDynamic(std::string weight_name, std::string bias_name, framework::Scope* scope, int dev_id);
    FcPluginDynamic(std::string weight_name, std::string bias_name, framework::Scope* scope, int dev_id, int n, int k,
                    cublasLtHandle_t handle, cublasLtMatrixLayout_t weight_transform_desc,
                    cublasLtMatrixTransformDesc_t transform_desc):
                    weight_name_(weight_name), bias_name_(bias_name), scope_(scope), dev_id_(dev_id), n_(n), k_(k), 
                    handle_(handle), weight_transform_desc_(weight_transform_desc), 
                    transform_desc_(transform_desc) {
        VLOG(1) << "[debug] clone constructor scope_ " <<  scope_;
      }

    FcPluginDynamic(void const* serial_data, size_t serial_length) {
      VLOG(1) << "[DEBUG] DeserializeValue";
      const char* weight_name_str;
      const char* bias_name_str;
      DeserializeValue(&serial_data, &serial_length, &weight_name_str);
      DeserializeValue(&serial_data, &serial_length, &bias_name_str);
      DeserializeValue(&serial_data, &serial_length, &scope_);
      DeserializeValue(&serial_data, &serial_length, &dev_id_);
      DeserializeValue(&serial_data, &serial_length, &k_);
      DeserializeValue(&serial_data, &serial_length, &n_);
      DeserializeValue(&serial_data, &serial_length, &handle_);
      DeserializeValue(&serial_data, &serial_length, &weight_transform_desc_);
      DeserializeValue(&serial_data, &serial_length, &transform_desc_);
      VLOG(1) << "[DEBUG] Deserialize weight_name_ "  << weight_name_;
      VLOG(1) << "[DEBUG] Deserialize weight_name_str "  << std::string(weight_name_str);
      weight_name_ = std::string(weight_name_str);
      bias_name_ = std::string(bias_name_str);
    }
    nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
        VLOG(1) << "[DEBUG] clone";
        return new FcPluginDynamic(weight_name_, bias_name_, scope_, dev_id_, n_, k_, handle_, weight_transform_desc_, transform_desc_);
    }

    const char* getPluginType() const TRT_NOEXCEPT override {
        return "fc_plugin";
    }
    int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
    int initialize() TRT_NOEXCEPT override { return 0; };

    size_t getSerializationSize() const TRT_NOEXCEPT override {
      VLOG(1) << "[DEBUG] getSerializationSize";
      VLOG(1) << "[DEBUG] weight_name_ SerializationSize " << SerializedSize(weight_name_.c_str());
        return SerializedSize(weight_name_.c_str()) + 
              SerializedSize(bias_name_.c_str()) + 
              SerializedSize(scope_) +
              SerializedSize(dev_id_) +
              SerializedSize(n_) +
              SerializedSize(k_) +
              SerializedSize(handle_) +
              SerializedSize(weight_transform_desc_) +
              SerializedSize(transform_desc_);
        
    }
    void serialize(void* buffer) const TRT_NOEXCEPT override {
       VLOG(1) << "[DEBUG] serialize";
       VLOG(1) << "[DEBUG] serialize weight_name_" << weight_name_;
        SerializeValue(&buffer, weight_name_.c_str());
        SerializeValue(&buffer, bias_name_.c_str());
        SerializeValue(&buffer, scope_);
        SerializeValue(&buffer, dev_id_);
        SerializeValue(&buffer, k_);
        SerializeValue(&buffer, n_); 
        SerializeValue(&buffer, handle_); 
        SerializeValue(&buffer, weight_transform_desc_); 
        SerializeValue(&buffer, transform_desc_); 

    }

    nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT override;

    bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs) TRT_NOEXCEPT override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                        int nb_inputs,
                        const nvinfer1::DynamicPluginTensorDesc* out,
                        int nb_outputs) TRT_NOEXCEPT override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                            int nb_inputs,
                            const nvinfer1::PluginTensorDesc* outputs,
                            int nb_outputs) const TRT_NOEXCEPT override {
        return 0;
    }

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* input_types,
        int nb_inputs) const TRT_NOEXCEPT override;

    void destroy() TRT_NOEXCEPT override { delete this; }

 private:
    std::string weight_name_;
    std::string bias_name_;
    framework::Scope* scope_;
    int dev_id_;
    int k_;
    int n_;
    int m_;

    //cublasLt
    cublasLtHandle_t handle_;
    cublasLtMatrixLayout_t weight_transform_desc_;
    cublasLtMatrixTransformDesc_t transform_desc_;
};

class FcPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "fc_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }


  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    auto plugin = new FcPluginDynamic(serial_data, serial_length);
    return plugin;
  }
};
REGISTER_TRT_PLUGIN_V2(FcPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle