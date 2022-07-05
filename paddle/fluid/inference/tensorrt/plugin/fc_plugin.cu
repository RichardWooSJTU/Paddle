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
#include "paddle/fluid/inference/tensorrt/plugin/utils.h"

namespace dyl = paddle::platform::dynload;

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

__global__ void DequantizeKernel(int8_t* src, half* dst, half scale, int num_raws, int num_cols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_raws * num_cols) return;
    dst[tid] = (half)src[tid] * scale / (half)127.0;
}

template <typename T>
__global__ void AddBias(T* res, T* bias, int num_raws, int num_cols) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_raws * num_cols) return;
    int res_i = tid;
    int bias_i = tid % num_cols;
    res[res_i] += bias[bias_i];
}

__global__ void transpose_kernel(const int8_t* src,
    int8_t* dst,
    int row,
    int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    if (i >= row) return;
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col
    if (j >= col) return;

    dst[j * row + i] = src[i * col + j];
}

void LaunchTranspose(const int8_t* input,
 int8_t* output,
    int row,
    int col,
    cudaStream_t stream) {
    dim3 grid((row + 31) / 32, (col + 31) / 32);
    dim3 block(32, 32);
    transpose_kernel<<<grid, block, 0, stream>>>(input, output, row, col);
}


FcPluginDynamic::FcPluginDynamic(std::string weight_name, std::string bias_name, framework::Scope* scope, int dev_id, int k, int n)
    : weight_name_(weight_name), bias_name_(bias_name), scope_(scope), dev_id_(dev_id), k_(k), n_(n) {
        VLOG(1) << "[DEBUG] FcPluginDynamic";
        VLOG(1) << "[DEBUG] buildtime scope addr " << scope;
        auto* Y_v = scope_->FindVar(weight_name); // auto == Variable
        PADDLE_ENFORCE_NOT_NULL(
            Y_v, platform::errors::Fatal("Y_v should not be null, check weight name"));
        auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();

        framework::Variable* bias_v = nullptr;
        framework::LoDTensor* bias_t = nullptr;
        if (bias_name != "") {
            bias_v = scope_->FindVar(bias_name);
            PADDLE_ENFORCE_NOT_NULL(
                bias_v, platform::errors::Fatal("bias_v should not be null, check bias name"));
            bias_t = bias_v->GetMutable<framework::LoDTensor>();
            VLOG(1) << "bias tensor info: ";
            VLOG(1) << "bias tensor type: " << bias_t->name();
            VLOG(1) << "bias tensor element num: " << bias_t->numel();
            VLOG(1) << "bias tensor dims" << bias_t->dims();

            VLOG(1) << "bias tensor dtype" << bias_t->dtype();
            VLOG(1) << "bias tensor layout"<< bias_t->layout();
        }
        auto dims = Y_t->dims();
        // k_ = dims[0];
        VLOG(1) << "[DEBUG] k_ " << k_;
        // n_ = dims[1];
        VLOG(1) << "[DEBUG] n_ " << n_;

        VLOG(1) << "weight tensor type: " << Y_t->name();
        VLOG(1) << "weight tensor element num: " << Y_t->numel();
        VLOG(1) << "weight tensor dims" << Y_t->dims();

        VLOG(1) << "weight tensor dtype" << Y_t->dtype();
        VLOG(1) << "weight tensor layout"<< Y_t->layout();


        std::string gpu_suffix = "_gpu_for_fc";
        std::string weight_name_gpu = weight_name + gpu_suffix;
        std::string bias_name_gpu = bias_name + gpu_suffix;
        weight_name_ = weight_name_gpu;
        if (bias_name != "")
            bias_name_ = bias_name_gpu;
        framework::Variable* Y_v_gpu = scope_->FindVar(weight_name_gpu);
        framework::LoDTensor* Y_t_gpu;
        framework::Variable* bias_v_gpu = scope_->FindVar(bias_name_gpu);
        framework::LoDTensor* bias_t_gpu;

        // create desc
        cublasStatus_t status;
        VLOG(1) << "create desc ";
        PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
        status = dyl::cublasLtCreate(&handle_);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
        cublasLtMatrixLayout_t weight_origin_desc;
        status = dyl::cublasLtMatrixLayoutCreate(&weight_origin_desc, CUDA_R_8I, k_, n_, k_);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
        cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
        int ldbtransform = 32 * ((n_ + 8 - 1) / 8) * 8;
        status = dyl::cublasLtMatrixLayoutCreate(&weight_transform_desc_, CUDA_R_8I, n_, k_, ldbtransform);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
        status = dyl::cublasLtMatrixLayoutSetAttribute(weight_transform_desc_, 
                                                                    CUBLASLT_MATRIX_LAYOUT_ORDER, 
                                                                    &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C));
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));

        status = dyl::cublasLtMatrixTransformDescCreate(&transform_desc_, CUDA_R_32F);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixTransformDescCreate"));
        if (Y_v_gpu == nullptr) {
            //quant weight
            float* old_weight_data = Y_t->data<float>();
            float weight_max = 0.0f;
            VLOG(1) << "weights data " << weight_name;
            for (int i =0; i < Y_t->numel(); ++i) {
                // std::cout << old_weight_data[i] << " ";
                if (old_weight_data[i] > weight_max) {
                    weight_max = old_weight_data[i];
                }
            }
            // std::cout << std::endl;
            VLOG(1) << "begin quant ";
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
            framework::LoDTensor scale_tensor;
            scale_tensor.Resize(dims);
            int8_t* temp_weight_data = scale_tensor.mutable_data<int8_t>(platform::CPUPlace());
            // float weight_scale = weight_max / 127.0;
            float weight_scale = 0.007874015748031496;
            VLOG(1) << "weights scale " << weight_scale;
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
            VLOG(1) << "weights quantize data " << weight_name;
            for (int i =0; i < Y_t->numel(); ++i) {
                temp_weight_data[i] = static_cast<int8_t>(
                    round_scale( old_weight_data[i] / weight_scale));
                // std::cout << static_cast<int>(temp_weight_data[i]) << " ";
            }
            // std::cout << std::endl;

            // copy int8 weights to GPU
            int8_t * weight_int8_data, * weight_int8_data_tmp;
            // PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(reinterpret_cast<void**>(&weight_int8_data), 
            PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&weight_int8_data, 
                sizeof(int8_t) * n_ * k_));
            PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&weight_int8_data_tmp, 
                    sizeof(int8_t) * n_ * k_));
            PADDLE_ENFORCE_GPU_SUCCESS(
                cudaMemcpy(weight_int8_data_tmp, temp_weight_data, sizeof(int8_t) * n_ * k_, cudaMemcpyHostToDevice));

            LaunchTranspose(weight_int8_data_tmp, weight_int8_data, k_, n_, 0);

            // Transform weight
            
            // create new weight var
            VLOG(1) << "create new weight var ";
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
            Y_v_gpu = scope_->Var(weight_name_gpu);
            Y_t_gpu = Y_v_gpu -> GetMutable<framework::LoDTensor>();
            Y_t_gpu -> Resize({(k_ + 32 - 1) / 32 * ldbtransform});
            int8_t *weight_transform_data = Y_t_gpu->mutable_data<int8_t>(platform::CUDAPlace(dev_id_));
            // cudaMalloc(reinterpret_cast<void**>(&weight_transform), sizeof(int8_t) * (k_ + 32 - 1) / 32 * ldbtransform);
            VLOG(1) << "y_t gpu mutable data gpu";
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
            float alpha = 1.0f, beta = 0.0f;
            cublasOperation_t op_transpose = CUBLAS_OP_T;
            status = dyl::cublasLtMatrixTransformDescSetAttribute(transform_desc_, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op_transpose, sizeof(op_transpose)); 
            status = dyl::cublasLtMatrixTransform(handle_, transform_desc_, (void*)&alpha, weight_int8_data, weight_origin_desc, 
                (void*)&beta, nullptr, nullptr, weight_transform_data, weight_transform_desc_, (cudaStream_t)0);
            PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixTransform"));
            VLOG(1) << "after transform B";
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
            op_transpose = CUBLAS_OP_N;
            status = dyl::cublasLtMatrixTransformDescSetAttribute(transform_desc_, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op_transpose, sizeof(op_transpose)); 
            PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
            if (weight_int8_data) PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(weight_int8_data));
        } else {
            VLOG(1) << "gpu weight is initialized from scope";
            auto* Y_t_gpu = Y_v_gpu->GetMutable<framework::LoDTensor>();
            VLOG(1) << "copied weight tensor type: " << Y_t_gpu->name();
            VLOG(1) << "copied weight tensor element num: " << Y_t_gpu->numel();
            VLOG(1) << "copied weight tensor dims" << Y_t_gpu->dims();  
        }

        if (bias_v_gpu != nullptr) {
            VLOG(1) << "gpu bias is initialized from scope";
            auto* bias_t_gpu = bias_v_gpu->GetMutable<framework::LoDTensor>();
        } else if (bias_v != nullptr && bias_t != nullptr) {
            // quantize and sync to GPU
            // TODO(@wufeisheng): if multiple precision should be surported, 1. sync multiple bias to gpu 2. reformat when enqueue

            VLOG(1) << "bias data " << bias_name;
            float * old_bias_data = bias_t -> data<float>(); 
            float bias_max = 0.0f;
            for (int i =0; i < bias_t->numel(); ++i) {
                // std::cout << old_bias_data[i] << " ";
                if (old_bias_data[i] > bias_max) {
                    bias_max = old_bias_data[i];
                }
            }
            // std::cout << std::endl;
            VLOG(1) << "begin quant ";
            std::vector<int8_t> scaled_bias(bias_t->numel());
            // float bias_scale = bias_max / 127.0;
            float bias_scale = 0.007874015748031496;
            VLOG(1) << "bias_scale " << bias_scale;
            for (int i =0; i < bias_t->numel(); ++i) {
                scaled_bias[i] = static_cast<int8_t>(
                    round_scale( old_bias_data[i] / bias_scale));
            }

            bias_v_gpu = scope_->Var(bias_name_gpu);
            bias_t_gpu = bias_v_gpu -> GetMutable<framework::LoDTensor>();
            VLOG(1) << "bias_t->dims()" << bias_t->dims();
            bias_t_gpu -> Resize(bias_t->dims());
            int8_t *bias_data = bias_t_gpu->mutable_data<int8_t>(platform::CUDAPlace(dev_id_));
            PADDLE_ENFORCE_GPU_SUCCESS(
                cudaMemcpy(bias_data, scaled_bias.data(), sizeof(int8_t) * n_, cudaMemcpyHostToDevice));

        } else if (bias_t == nullptr) {
            VLOG(1) << "why bias_t is null while bias_v is not";
        }
        
        
}


nvinfer1::DimsExprs FcPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
    VLOG(1) << "[DEBUG] getOutputDimensions";
    nvinfer1::DimsExprs ret;

    ret.nbDims = inputs->nbDims;
    if (ret.nbDims == 4) {
        ret.d[0] = inputs[0].d[0];
        ret.d[1] = expr_builder.constant(n_);
        ret.d[2] = expr_builder.constant(1);
        ret.d[3] = expr_builder.constant(1);
    } else if (ret.nbDims == 5) {
        ret.d[0] = inputs[0].d[0];
        ret.d[1] = inputs[0].d[1];
        ret.d[2] = expr_builder.constant(n_);
        ret.d[3] = expr_builder.constant(1);
        ret.d[4] = expr_builder.constant(1);
    }
    

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
        if (pos == 0) {
            if (in.type == nvinfer1::DataType::kINT8) {
                VLOG(1) << "DEBUG: input is int8";
            } else if (in.type == nvinfer1::DataType::kHALF) {
                VLOG(1) << "DEBUG: input is half";
            } else if (in.type == nvinfer1::DataType::kFLOAT){
                VLOG(1) << "DEBUG: input is float";
            } else {
                VLOG(1) << "DEBUG: input is other";
            }
            return (in.type == nvinfer1::DataType::kINT8 ||
                in.type == nvinfer1::DataType::kHALF) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);
        }
        const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
        // output
        if (in.type == nvinfer1::DataType::kINT8) {
            VLOG(1) << "DEBUG: output is int8";
        } else if (in.type == nvinfer1::DataType::kHALF) {
            VLOG(1) << "DEBUG: output is half";
        } else if (in.type == nvinfer1::DataType::kFLOAT){
            VLOG(1) << "DEBUG: output is float";
        } else {
            VLOG(1) << "DEBUG: output is other";
        }
        if (prev.type == nvinfer1::DataType::kINT8) {
            return ((in.type == prev.type || in.type == nvinfer1::DataType::kHALF) && in.format == prev.format);
        }
        return in.type == prev.type && in.format == prev.format;
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
    auto input_type = inputDesc[0].type;
    auto output_type = outputDesc[0].type;
    // Get m
    nvinfer1::Dims input_dims = inputDesc[0].dims, output_dims = outputDesc[0].dims;
    if (input_dims.nbDims == 4) {
        m_ = input_dims.d[0];
    } else if (input_dims.nbDims == 5) {
        m_ = input_dims.d[0] * input_dims.d[1];
    }
    VLOG(1) << "m_" << m_;
    VLOG(1) << "k_" << k_;
    VLOG(1) << "n_" << n_;
    
    for (int i = 0; i < input_dims.nbDims; ++i)
        VLOG(1) << "input_dims.d[ " << i << "] " << input_dims.d[i];
    if (input_type != nvinfer1::DataType::kINT8 ) {
        VLOG(1) << "input type is not int8";
        // for (int i  = 0; i < 10000000; i++){
        //     float t = (float)(i+2)/(float)(i+1);
        //     float tt = t / 2.0;
        //     VLOG(10) << tt;
        // }
        if (input_type == nvinfer1::DataType::kHALF) {
            const half* input_data_tmp = static_cast<const half*>(inputs[0]);
            // std::vector<half> input_vec(m_ * k_);
            // VLOG(1) << "input_data_tmp with half";
            // PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(input_vec.data(), input_data_tmp, sizeof(half) * m_ * k_, cudaMemcpyDeviceToHost));
            // for (int i = 0; i < m_ * k_; ++i) {
            //     std::cout << static_cast<float>(input_vec[i]) << " ";
            // }
            // std::cout << std::endl;

            half* output_data = static_cast<half*>(outputs[0]);
            PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(output_data, input_data_tmp, sizeof(half) * m_ * k_, cudaMemcpyDeviceToDevice));
        }
        
        return cudaGetLastError() != cudaSuccess;
    }


    

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
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

    cublasStatus_t status;
    
    // Transform A
    // int8_t* input_data_origin = const_cast<int8_t*>(static_cast<const int8_t*>(inputs[0]));
    // std::vector<int8_t> input_origin_h(m_ * k_);
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(input_origin_h.data(), input_data_origin, sizeof(int8_t) * m_ * k_, cudaMemcpyDeviceToHost));
    // VLOG(1) << "in[0] " << input_origin_h[0];

    const int8_t* input_data_tmp = static_cast<const int8_t*>(inputs[0]);
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&input_data, sizeof(int8_t) * m_ * k_));
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(input_data, input_origin_h.data(), sizeof(int8_t) * m_ * k_, cudaMemcpyHostToDevice));

    //de bug
    // VLOG(1) << "input_data_tmp ";
    // std::vector<int8_t> input_vec(m_ * k_);
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(input_vec.data(), input_data_tmp, sizeof(int8_t) * m_ * k_, cudaMemcpyDeviceToHost));
    // for (int i = 0; i < m_ * k_; ++i) {
    //     std::cout << static_cast<int>(input_vec[i]) << " ";
    // }
    // std::cout << std::endl;
    //de bug

    int8_t* input_data;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&input_data, sizeof(int8_t) * m_ * k_));
    LaunchTranspose(input_data_tmp, input_data, m_, k_, stream);


    cublasLtMatrixLayout_t input_desc;
    status = dyl::cublasLtMatrixLayoutCreate(&input_desc, CUDA_R_8I, m_, k_, m_);
    PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));

    cublasLtMatrixLayout_t input_transform_desc;
    cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
    int ldatransform = 32 * m_;
    status = dyl::cublasLtMatrixLayoutCreate(&input_transform_desc, CUDA_R_8I, m_, k_, ldatransform);
    PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
    status = dyl::cublasLtMatrixLayoutSetAttribute(input_transform_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutSetAttribute"));
    int8_t *input_transform_data;
    VLOG(1) << "prepare transform A " << "ldatransform " << ldatransform << " (k_ + 32 - 1) / 32 " << (k_ + 32 - 1) / 32;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&input_transform_data, sizeof(int8_t) * (k_ + 32 - 1) / 32 * ldatransform));
    VLOG(1) << "after malloc input_transform_data";
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
    
    // framework::LoDTensor input_transform_tensor;
    // input_transform_tensor.Resize({(k_ + 32 - 1) / 32 * ldatransform});
    // input_transform_data = input_transform_tensor.mutable_data<int8_t>(platform::CUDAPlace(dev_id_));

    float alpha = 1.0f, beta = 0.0f;
    status = dyl::cublasLtMatrixTransform(handle_, transform_desc_, (void*)&alpha, input_data, input_desc, 
        (void*)&beta, nullptr, nullptr, input_transform_data, input_transform_desc, stream);
    PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixTransform"));
    VLOG(1) << "after transform A status " << status;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

    // Just malloc c_transform
    cublasLtMatrixLayout_t c_transdesc;
    int ldctransform = 32 * m_;
    status = dyl::cublasLtMatrixLayoutCreate(&c_transdesc, CUDA_R_8I, m_, n_, ldctransform);
    PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
    status = dyl::cublasLtMatrixLayoutSetAttribute(c_transdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutSetAttribute"));

    VLOG(1) << "Prepare tranform C space " << "ldctransform " << ldctransform << " (n_ + 32 - 1) / 32 " << (n_ + 32 - 1) / 32;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
    int8_t *c_transform_data;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&c_transform_data, sizeof(int8_t) * (n_ + 32 - 1) / 32 * ldctransform));
    // framework::LoDTensor c_transform_tensor;
    // c_transform_tensor.Resize({(n_ + 32 - 1) / 32 * ldctransform});
    // c_transform_data = c_transform_tensor.mutable_data<int8_t>(platform::CUDAPlace(dev_id_));


    // Perform matmul
    // init desc
    cublasLtMatmulDesc_t matmulDesc;
    status = dyl::cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32F);
    PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatmulDescCreate"));
    cublasOperation_t op_transpose = CUBLAS_OP_T;
    status = dyl::cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_transpose, sizeof(op_transpose));
    PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatmulDescSetAttribute"));

    // run
    VLOG(1) << "run matmul";
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
    status = dyl::cublasLtMatmul(handle_,
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
        stream);
    PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatmul"));
    int8_t* bias_data;

    //Get bias data
    if (bias_name_ != "") {
        auto* bias_v_gpu = scope_->FindVar(bias_name_);
        PADDLE_ENFORCE_NOT_NULL(
            bias_v_gpu,
            platform::errors::NotFound(
                "variable %s is not found in TensorRT subgraph.", bias_name_));
        auto* bias_t_gpu = bias_v_gpu->GetMutable<framework::LoDTensor>();
        bias_data = bias_t_gpu->data<int8_t>();
    }

    // De-Transform C and write to output
    cublasLtMatrixLayout_t output_desc;
    status = dyl::cublasLtMatrixLayoutCreate(&output_desc, CUDA_R_8I, m_, n_, m_);
    PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixLayoutCreate"));
    VLOG(1) << "de-transform c";
    VLOG(1) << "out dims[0] " << output_dims.d[0] << " out dims[1] " << output_dims.d[1];
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

    if (output_type == nvinfer1::DataType::kINT8) {
        int8_t* output_data = static_cast<int8_t*>(outputs[0]);
        status = dyl::cublasLtMatrixTransform(handle_, transform_desc_, &alpha, c_transform_data, c_transdesc, 
            &beta, nullptr, nullptr, output_data, output_desc, stream);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixTransform"));

        // add bias
        int grid, block;
        CalculateBlockAndGridSize(m_, n_, grid, block);
        if (bias_name_ != "")
            AddBias<int8_t><<<grid, block, 0, stream>>>(output_data, bias_data, m_, n_);

        //de bug
        // VLOG(1) << "output data ";
        // std::vector<int8_t> output_vec(m_ * n_);
        // PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(output_vec.data(), output_data, sizeof(int8_t) * m_ * n_, cudaMemcpyDeviceToHost));
        // for (int i = 0; i < m_ * n_; ++i) {
        //     std::cout << static_cast<int>(output_vec[i]) << " ";
        // }
        // std::cout << std::endl;
        //de bug
    
        
    } else if (output_type == nvinfer1::DataType::kHALF) {
        half* output_data = static_cast<half*>(outputs[0]);
        int8_t* output_data_int8;
        PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&output_data_int8, sizeof(int8_t) * m_ * n_));
        
        status = dyl::cublasLtMatrixTransform(handle_, transform_desc_, &alpha, c_transform_data, c_transdesc, 
            &beta, nullptr, nullptr, output_data_int8, output_desc, stream);
        PADDLE_ENFORCE_EQ(status, CUBLAS_STATUS_SUCCESS, platform::errors::Fatal("cublasLtMatrixTransform"));

        
        int grid, block;
        CalculateBlockAndGridSize(m_, n_, grid, block);
        if (bias_name_ != "")
            AddBias<int8_t><<<grid, block, 0, stream>>>(output_data_int8, bias_data, m_, n_);
        
        //de quantize
        half out_scale = 1.0;
        DequantizeKernel<<<grid, block, 0, stream>>>(output_data_int8, output_data, out_scale, m_, n_);

        //de bug
        // VLOG(1) << "output data ";
        // std::vector<half> output_vec(m_ * n_);
        // PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(output_vec.data(), output_data, sizeof(half) * m_ * n_, cudaMemcpyDeviceToHost));
        // for (int i = 0; i < m_ * n_; ++i) {
        //     std::cout << static_cast<float>(output_vec[i]) << " ";
        // }
        // std::cout << std::endl;
        //de bug

        VLOG(1) << "cudaDeviceSynchronize";
        PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
        VLOG(1) << "free output_data_int8";
        if (output_data_int8) PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(output_data_int8));
        return cudaGetLastError() != cudaSuccess;
    } else {
        PADDLE_ENFORCE_EQ(output_type == nvinfer1::DataType::kINT8 || output_type == nvinfer1::DataType::kHALF, true, platform::errors::Fatal("not support"));
    }   
    // if (c_transdesc) dyl::cublasLtMatrixLayoutDestroy(c_transdesc);
    // if (weight_transform_desc_) dyl::cublasLtMatrixLayoutDestroy(weight_transform_desc_);
    // if (input_transform_desc) dyl::cublasLtMatrixLayoutDestroy(input_transform_desc);
    // if (output_desc) dyl::cublasLtMatrixLayoutDestroy(output_desc);
    // if (input_desc) dyl::cublasLtMatrixLayoutDestroy(input_desc);
    // if (matmulDesc) dyl::cublasLtMatmulDescDestroy(matmulDesc);
    // if (transform_desc_) dyl::cublasLtMatrixTransformDescDestroy(transform_desc_);
    VLOG(1) << "cudaDeviceSynchronize";
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
    VLOG(1) << "free c_transform_data";
    if (c_transform_data) PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(c_transform_data));
    VLOG(1) << "free input_transform_data";
    if (input_transform_data) PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(input_transform_data));
    VLOG(1) << "free input_data";
    if (input_data) PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(input_data));

    return cudaGetLastError() != cudaSuccess;
}
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle