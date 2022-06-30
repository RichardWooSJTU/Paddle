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

namespace dyl = paddle::platform::dynload;

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename T>
__global__ void transpose_kernel(const T* src,
                                 T* dst,
                                 int row,
                                 int col) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  int j = blockIdx.y * blockDim.y + threadIdx.y; // col

  dst[j * row + i] = src[i * col + j];
}

template <typename T>
void transpose_kernelLauncher(const T* input,
                              T* output,
                              int row,
                              int col,
                              cudaStream_t stream) {
  dim3 grid((row + 31) / 32, (col + 31) / 32);
  dim3 block(32, 32);
  transpose_kernel<<<grid, block, 0, stream>>>(input, output, row, col);
}


FcPluginDynamic::FcPluginDynamic(std::string weight_name, std::string bias_name, framework::Scope* scope, int dev_id)
    : weight_name_(weight_name), bias_name_(bias_name), scope_(scope), dev_id_(dev_id) {
        VLOG(1) << "[DEBUG] FcPluginDynamic";
        VLOG(1) << "[DEBUG] buildtime scope addr " << scope;
        std::string gpu_suffix = "_gpu_for_fc";
        std::string weight_name_gpu = weight_name + gpu_suffix;
        weight_name_ = weight_name_gpu;
        framework::Variable* Y_v_gpu = scope_->FindVar(weight_name_gpu);
        framework::LoDTensor* Y_t_gpu;
        if (Y_v_gpu == nullptr) {
            auto* Y_v = scope_->FindVar(weight_name); // auto == Variable
            auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
            auto dims = Y_t->dims();
            k_ = dims[0];
            VLOG(1) << "[DEBUG] k_ " << k_;
            n_ = dims[1];
            VLOG(1) << "[DEBUG] n_ " << n_;

            VLOG(1) << "weight tensor type: " << Y_t->name();
            VLOG(1) << "weight tensor element num: " << Y_t->numel();
            VLOG(1) << "weight tensor dims" << Y_t->dims();

            VLOG(1) << "weight tensor dtype" << Y_t->dtype();
            VLOG(1) << "weight tensor layout"<< Y_t->layout();

            //quant weight
            float* old_weight_data = Y_t->data<float>();
            float weight_max = 0.0f;
            for (int i =0; i < Y_t->numel(); ++i) {
                if (old_weight_data[i] > weight_max) {
                    weight_max = old_weight_data[i];
                }
            }
            VLOG(1) << "begin quant ";
            framework::LoDTensor scale_tensor;
            scale_tensor.Resize(dims);
            int8_t* temp_weight_data = scale_tensor.mutable_data<int8_t>(platform::CPUPlace());
            float weight_scale = weight_max / 127.0;
            VLOG(1) << "weights scale " << weight_scale;
            auto round_scale = [](float x)  { return std::floor(x + 0.5f); };
            for (int i =0; i < Y_t->numel(); ++i) {
                temp_weight_data[i] = static_cast<int8_t>(
                    round_scale( old_weight_data[i] / weight_scale));
            }

            // copy int8 weights to GPU
            int8_t * weight_int8_data;
            PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(reinterpret_cast<void**>(&weight_int8_data), 
                sizeof(int8_t) * dims[0] * dims[1]));
            PADDLE_ENFORCE_GPU_SUCCESS(
                cudaMemcpy(weight_int8_data, temp_weight_data, sizeof(int8_t) * dims[0] * dims[1], cudaMemcpyHostToDevice));

            // Transform weight
            // create desc
            VLOG(1) << "create desc ";
            PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtCreate(&handle_));
            cublasLtMatrixLayout_t weight_origin_desc;
            PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtMatrixLayoutCreate(&weight_origin_desc, CUDA_R_8I, k_, n_, k_));
            cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
            int ldbtransform = 32 * ((n_ + 8 - 1) / 8) * 8;
            PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtMatrixLayoutCreate(&weight_transform_desc_, CUDA_R_8I, n_, k_, ldbtransform));
            PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtMatrixLayoutSetAttribute(weight_transform_desc_, 
                                                                        CUBLASLT_MATRIX_LAYOUT_ORDER, 
                                                                        &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));
            // create new weight var
            VLOG(1) << "create new weight var ";
            Y_v_gpu = scope_->Var(weight_name_gpu);
            Y_t_gpu = Y_v_gpu -> GetMutable<framework::LoDTensor>();
            Y_t_gpu -> Resize({(k_ + 32 - 1) / 32 * ldbtransform});
            int8_t *weight_transform_data = Y_t_gpu->mutable_data<int8_t>(platform::CUDAPlace(dev_id_));
            // cudaMalloc(reinterpret_cast<void**>(&weight_transform), sizeof(int8_t) * (k_ + 32 - 1) / 32 * ldbtransform);
            PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtMatrixTransformDescCreate(&transform_desc_, CUDA_R_32F));
            float alpha = 1.0f, beta = 0.0f;
            cublasOperation_t op_transpose = CUBLAS_OP_T;
            dyl::cublasLtMatrixTransformDescSetAttribute(transform_desc_, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op_transpose, sizeof(op_transpose)); 
            PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtMatrixTransform(handle_, transform_desc_, (void*)&alpha, weight_int8_data, weight_origin_desc, 
                (void*)&beta, nullptr, nullptr, weight_transform_data, weight_transform_desc_, (cudaStream_t)0));
            op_transpose = CUBLAS_OP_N;
            dyl::cublasLtMatrixTransformDescSetAttribute(transform_desc_, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op_transpose, sizeof(op_transpose)); 
            cudaDeviceSynchronize();
            if (weight_int8_data) PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(weight_int8_data));
        } else {
            VLOG(1) << "gpu weight is initialized from scope";
            auto* Y_t_gpu = Y_v_gpu->GetMutable<framework::LoDTensor>();
            VLOG(1) << "copied weight tensor type: " << Y_t_gpu->name();
            VLOG(1) << "copied weight tensor element num: " << Y_t_gpu->numel();
            VLOG(1) << "copied weight tensor dims" << Y_t_gpu->dims();  
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
        return (in.type == nvinfer1::DataType::kFLOAT || in.type == nvinfer1::DataType::kINT8) && in.format == nvinfer1::TensorFormat::kLINEAR;
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
    //Get weight tensor
    // auto* Y_v = scope_->FindVar(weight_name_); // auto == Variable
    // auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    
    
}

int FcPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
    // Get m
    nvinfer1::Dims input_dims = inputDesc[0].dims, output_dims = outputDesc[0].dims;
    m_ = input_dims.d[0] * input_dims.d[1];
    VLOG(1) << "m_" << m_;
    VLOG(1) << "k_" << k_;
    VLOG(1) << "n_" << n_;

    //Get weight tensor
    VLOG(1) << "fc plugin enqueue";
    VLOG(1) << "scope_ " << scope_;
    VLOG(1) << "weight_name_ " << weight_name_;

    auto* weight_transform_var = scope_->FindVar(weight_name_); // auto == Variable
    PADDLE_ENFORCE_NOT_NULL(
        weight_transform_var,
        platform::errors::NotFound(
            "variable %s is not found in TensorRT subgraph.", weight_name_));
    VLOG(1) << "find Y_v";
    auto* weight_transform_tensor = weight_transform_var->GetMutable<framework::LoDTensor>();
    VLOG(1) << "find Y_t";
    int8_t* weight_transform_data = weight_transform_tensor->data<int8_t>();
    
    // Transform A
    const int8_t* input_data = static_cast<const int8_t*>(inputs[0]);

    cublasLtMatrixLayout_t input_desc;
    dyl::cublasLtMatrixLayoutCreate(&input_desc, CUDA_R_8I, m_, k_, m_);

    cublasLtMatrixLayout_t input_transform_desc;
    cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
    int ldatransform = 32 * m_;
    dyl::cublasLtMatrixLayoutCreate(&input_transform_desc, CUDA_R_8I, m_, k_, ldatransform);
    dyl::cublasLtMatrixLayoutSetAttribute(input_transform_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));

    int8_t *input_transpose_data;
    VLOG(1) << "transpose input";
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(reinterpret_cast<void**>(&input_transpose_data), sizeof(int8_t) * m_ * k_));
    transpose_kernelLauncher<int8_t>(input_data, input_transpose_data, m_, k_, stream);

    int8_t *input_transform_data;
    VLOG(1) << "prepare transform A " << "ldatransform " << ldatransform << " (k_ + 32 - 1) / 32 " << (k_ + 32 - 1) / 32;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(reinterpret_cast<void**>(&input_transform_data), sizeof(int8_t) * (k_ + 32 - 1) / 32 * ldatransform));

    float alpha = 1.0f, beta = 0.0f;
    PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtMatrixTransform(handle_, transform_desc_, &alpha, input_transpose_data, input_desc, 
        &beta, nullptr, nullptr, input_transform_data, input_transform_desc, stream));

    // Just malloc c_transform
    cublasLtMatrixLayout_t c_transdesc;
    int ldctransform = 32 * m_;
    dyl::cublasLtMatrixLayoutCreate(&c_transdesc, CUDA_R_8I, m_, n_, ldctransform);
    dyl::cublasLtMatrixLayoutSetAttribute(c_transdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));

    VLOG(1) << "Prepare tranform C space " << "ldctransform " << ldctransform << " (n_ + 32 - 1) / 32 " << (n_ + 32 - 1) / 32;
    int8_t *c_transform_data;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(reinterpret_cast<void**>(&c_transform_data), sizeof(int8_t) * (n_ + 32 - 1) / 32 * ldctransform));

    // Perform matmul
    // init desc
    cublasLtMatmulDesc_t matmulDesc;
    dyl::cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32F);
    cublasOperation_t op_transpose = CUBLAS_OP_T;
    dyl::cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_transpose, sizeof(op_transpose));

    // run
    VLOG(1) << "run matmul";
    PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtMatmul(handle_,
        matmulDesc,
        &alpha,
        input_transform_data,
        input_transform_desc,
        weight_transform_data,
        weight_transform_desc_,
        &beta,
        c_transform_data,
        c_transdesc,
        c_transform_data,
        c_transdesc,
        nullptr,
        nullptr,
        0,
        stream));

    // De-Transform C and write to output
    int8_t* output_data = static_cast<int8_t*>(outputs[0]);


    cublasLtMatrixLayout_t output_desc;
    dyl::cublasLtMatrixLayoutCreate(&output_desc, CUDA_R_8I, m_, n_, m_);
    VLOG(1) << "de-transform c";
    VLOG(1) << "out dims[0] " << output_dims.d[0] << " out dims[1] " << output_dims.d[1];
    PADDLE_ENFORCE_GPU_SUCCESS(dyl::cublasLtMatrixTransform(handle_, transform_desc_, &alpha, c_transform_data, c_transdesc, 
        &beta, nullptr, nullptr, output_data, output_desc, stream));

    VLOG(1) << "transpose output";
    transpose_kernelLauncher<int32_t>(output_data, output_data, n_, m_, stream);

    cudaDeviceSynchronize();
    VLOG(1) << "free c_transform_data";
    if (c_transform_data) PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(c_transform_data));
    VLOG(1) << "free input_transform_data";
    if (input_transform_data) PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(input_transform_data));
    VLOG(1) << "free input_transpose_data";
    if (input_transpose_data) PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(input_transpose_data));
    return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
