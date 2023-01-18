// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename Context>
void FusedMultiTransformerKernel(
    const Context &ctx,
    const DenseTensor &x,
    const DenseTensor &src_mask,
    const std::vector<const DenseTensor *> &qkv_weights,
    const std::vector<const DenseTensor *> &qkv_biases,
    const std::vector<const DenseTensor *> &out_linear_weights,
    const std::vector<const DenseTensor *> &out_linear_biases,
    const std::vector<const DenseTensor *> &ffn1_weights,
    const std::vector<const DenseTensor *> &ffn1_biases,
    const std::vector<const DenseTensor *> &ffn2_weights,
    const std::vector<const DenseTensor *> &ffn2_biases,
    const std::vector<const DenseTensor *> &ln_scales,
    const std::vector<const DenseTensor *> &ln_biases,
    const std::vector<const DenseTensor *> &ffn_ln_scales,
    const std::vector<const DenseTensor *> &ffn_ln_biases,
    int time_step,
    bool pre_layer_norm,
    float epsilon,
    std::string act_method,
    DenseTensor *out,
    std::vector<DenseTensor *> cache_kvs);

}  // namespace phi
