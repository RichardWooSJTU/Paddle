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

#include "paddle/fluid/inference/tensorrt/plugin/fc_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

FcPluginDynamic::FcPluginDynamic(std::string weights_name, std::string bias_name, framework::Scope* scope, int dev_id)
    : weights_name_(weights_name), bias_name_(bias_name), scope_(scope), dev_id_(dev_id), k_(0), n_(0) {
        std::string gpu_suffix = "_gpu_for_fc";
        std::string weights_name_gpu = weights_name + gpu_suffix;
        auto* Y_v_gpu = scope_->FindVar(weights_name_gpu);
       
        if (Y_v_gpu == nullptr) {
            VLOG(1) << "[DEBUG] FcPluginDynamic";
            VLOG(1) << "[DEBUG] buildtime scope addr " << scope;
            VLOG(1) << "[debug] scope has kid " << scope_->kids().size();

            auto* Y_v = scope_->FindVar(weights_name_); // auto == Variable
            auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
            auto dims = Y_t->dims();
            k_ = dims[0];
            VLOG(1) << "[DEBUG] k_ " << k_;
            n_ = dims[1];
            VLOG(1) << "[DEBUG] n_ " << n_;

            VLOG(1) << "weights tensor type: " << Y_t->name();
            VLOG(1) << "weights tensor element num: " << Y_t->numel();
            VLOG(1) << "weights tensor dims" << Y_t->dims();

            VLOG(1) << "weights tensor dtype" << Y_t->dtype();
            VLOG(1) << "weights tensor layout"<< Y_t->layout();

            //quant weights
            float* old_weight_data = Y_t->data<float>();
            framework::LoDTensor temp_tensor;

            //create new weights var
            framework::Variable *Y_v_gpu = scope_->Var(weights_name_gpu);
            framework::LoDTensor* new_weight_t = Y_v_gpu -> GetMutable<framework::LoDTensor>();
            new_weight_t -> Resize(dims);
            int8_t* new_weight_data = new_weight_t->mutable_data<int8_t>(platform::CUDAPlace(dev_id_));
            std::vector<int8_t> new_weight_data_h(new_weight_t->numel(), 12);
            VLOG(1) << "copy " << new_weight_t->numel() * sizeof(int8_t) << "bytes to device";
            cudaMemcpy(new_weight_data, new_weight_data_h.data(), new_weight_t->numel() * sizeof(int8_t), cudaMemcpyHostToDevice);
            VLOG(1) << "new weights tensor type: " << new_weight_t->name();
            VLOG(1) << "new weights tensor element num: " << new_weight_t->numel();
            VLOG(1) << "new weights tensor dims" << new_weight_t->dims();
        } else {
            VLOG(1) << "gpu weight is initialized from scope";
            auto* Y_t_gpu = Y_v_gpu->GetMutable<framework::LoDTensor>();
            VLOG(1) << "copied weights tensor type: " << Y_t_gpu->name();
            VLOG(1) << "copied weights tensor element num: " << Y_t_gpu->numel();
            VLOG(1) << "copied weights tensor dims" << Y_t_gpu->dims();  
        }
}


nvinfer1::DimsExprs FcPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
    VLOG(1) << "[DEBUG] getOutputDimensions";
    nvinfer1::DimsExprs ret;

    ret.nbDims = inputs->nbDims;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = expr_builder.constant(n_);
    ret.d[2] = expr_builder.constant(1);
    ret.d[3] = expr_builder.constant(1);

    return ret;
}



bool FcPluginDynamic::supportsFormatCombination(int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
        VLOG(1) << "[DEBUG] supportsFormatCombination";
        PADDLE_ENFORCE_NOT_NULL(
            in_out, platform::errors::InvalidArgument(
                        "The input of fc plugin shoule not be nullptr."));
      
        PADDLE_ENFORCE_LT(
            pos, nb_inputs + nb_outputs,
            platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                              "num(%d) of the input and the output.",
                                              pos, nb_inputs + nb_outputs));
      
        const nvinfer1::PluginTensorDesc &in = in_out[pos];
        // if (pos == 0) {
        //   if (with_fp16_) {
        //     return (in.type == nvinfer1::DataType::kFLOAT ||
        //             in.type == nvinfer1::DataType::kHALF) &&
        //            (in.format == nvinfer1::TensorFormat::kLINEAR);
        //   } else {
        //     return (in.type == nvinfer1::DataType::kFLOAT) &&
        //            (in.format == nvinfer1::TensorFormat::kLINEAR);
        //   }
        // }
        // const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
        // // output
        // return in.type == prev.type && in.format == prev.format;
        return in.type == nvinfer1::DataType::kINT8 && in.format == nvinfer1::TensorFormat::kLINEAR;
}

nvinfer1::DataType FcPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
    
    VLOG(1) << "[DEBUG] getOutputDataType";
    return input_types[0];
}

void FcPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nb_outputs) TRT_NOEXCEPT {
    VLOG(1) << "[DEBUG] configurePlugin";
    //Get weights tensor
    // auto* Y_v = scope_->FindVar(weights_name_); // auto == Variable
    // auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    
}

int FcPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
    //Get weights tensor

    VLOG(1) << "fc plugin enqueue";
    VLOG(1) << "scope_ " << scope_;
    VLOG(1) << "weights_name_ " << weights_name_;

    auto* Y_v = scope_->FindVar(weights_name_); // auto == Variable
    PADDLE_ENFORCE_NOT_NULL(
        Y_v,
        platform::errors::NotFound(
            "variable %s is not found in TensorRT subgraph.", weights_name_));
    VLOG(1) << "find Y_v";
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    VLOG(1) << "find Y_t";
    int8_t* weight_data = Y_t->data<int8_t>();
    VLOG(1) << "find weight_data";
    std::vector<int8_t> weights_data_h(Y_t->numel());
    VLOG(1) << "copy " << Y_t->numel() * sizeof(int8_t) << "bytes to host";
    cudaMemcpy(weights_data_h.data(), weight_data, Y_t->numel() * sizeof(int8_t), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < Y_t->numel(); ++i) {
        VLOG(1) << static_cast<int>(weights_data_h[0]);
    // }

    if (inputDesc->type == nvinfer1::DataType::kINT8) {
         const int8_t * in = static_cast<const int8_t *>(inputs[0]);
         VLOG(1) << "get in";
        int8_t in_1;
        cudaMemcpy(&in_1, in, sizeof(int8_t), cudaMemcpyDeviceToHost);
         VLOG(1) << "input " << static_cast<int>(in_1);

    }
    return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
