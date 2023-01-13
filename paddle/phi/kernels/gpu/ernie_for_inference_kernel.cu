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

#include "paddle/phi/kernels/ernie_for_inference_kernel.h"

#include <algorithm>
#include <vector>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ErnieForInferenceKernel(const Context& ctx,
                             const DenseTensor& src_ids,
                             const DenseTensor& pos_ids,
                             const DenseTensor& input_mask,
                             const DenseTensor& pos_ids_extra,
                             const DenseTensor& tgt_ids,
                             const DenseTensor& tgt_pos,
                             const DenseTensor& tgt_pos_extra,
                             const DenseTensor& init_score,
                             const DenseTensor& tgt_mask,
                             const DenseTensor& max_dec_len,
                             const DenseTensor& min_dec_len,
                             const DenseTensor& topk,
                             const DenseTensor& topp,
                             const DenseTensor& topk,
                             bool decoding_strategy,
                             DenseTensor* scores,
                             DenseTensor* indices) {
  ctx.template Alloc<T>(scores);
  ctx.template Alloc<T>(indices);
}
}  // namespace phi

PD_REGISTER_KERNEL(ernie_for_inference,
                   GPU,
                   ALL_LAYOUT,
                   phi::ErnieForInferenceKernel,
                   float,
                   phi::dtype::float16) {}
