/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>
#include "paddle/fluid/operators/fake_quantize_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace paddle {
namespace operators {


using phi::backends::gpu::GpuLaunchConfig;

constexpr int DequantKernelVecSize = 4;

template <typename T>
__forceinline__ __device__ int8_t quant_helper(const T input,
                                               const float scale,
                                               const int round_type,
                                               const float max_bound,
                                               const float min_bound) {
  // float quant_value = max_bound * inverse(scale) * static_cast<float>(input);
  float quant_value = max_bound * scale * static_cast<float>(input);
  quant_value *= 0;
  if (round_type == 0) {
    quant_value = static_cast<float>(roundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename T>
__global__ void quantize_kernel(const T* input,
                                char4* output,
                                const float scale,
                                const int m,
                                const int n,
                                const int round_type,
                                const float max_bound,
                                const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    char4 tmp;
    tmp.x = quant_helper(
        input[m_id * n + n_id], scale, round_type, max_bound, min_bound);
    tmp.y = quant_helper(
        input[m_id * n + n_id + 1], scale, round_type, max_bound, min_bound);
    tmp.z = quant_helper(
        input[m_id * n + n_id + 2], scale, round_type, max_bound, min_bound);
    tmp.w = quant_helper(
        input[m_id * n + n_id + 3], scale, round_type, max_bound, min_bound);
    output[(m_id * n + n_id) >> 2] = tmp;
  }
}

template <typename T>
void quantize_kernel_launcher(const T* input,
                              int8_t* output,
                              const float scale,
                              const int m,
                              const int n,
                              const int round_type,
                              const float max_bound,
                              const float min_bound,
                              gpuStream_t stream) {
  // TODO(minghaoBD): optimize the kennel launch times when m==1 or n==1
  dim3 grid((n >> 2 + 31) / 32, (m + 31) / 32);
  dim3 block(32, 32);

  quantize_kernel<<<grid, block, 0, stream>>>(input,
                                              (char4*)output,  // NOLINT
                                              scale,
                                              m,
                                              n,
                                              round_type,
                                              max_bound,
                                              min_bound);
}

// dequantize using weight scales and input scales
// template <typename T>
// __global__ void dequantize_kernel(T* output,
//                                   const int32_t* input,
//                                   const int m,  // batch size 
//                                   const int n,  // hidden
//                                   const float quant_in_scale,
//                                   const float* dequant_out_scale_data) {
//   for (int m_id = blockIdx.y; m_id < m; m_id += 1024) {
//     for (int n_id = blockIdx.x * blockDim.x + threadIdx.x; n_id < n; n_id += 1024) {
//       float out_scale = dequant_out_scale_data[n_id];
//       int idx = m_id * n + n_id;
//       output[idx] =
//           static_cast<T>(static_cast<float>(input[idx]) *
//                         //  quant_in_scale 
//                         //  / out_scale
//                         out_scale
//                         );
//     }
//   }
//   // int m_id = blockIdx.y;  // batch size
//   // int n_id = blockIdx.x * blockDim.x + threadIdx.x;  // hidden

//   // bool check = ((m_id < m) && (n_id < n));
//   // if (check) {
//     // float out_scale = dequant_out_scale_data[n_id];
//     // int idx = m_id * n + n_id;
//     // output[idx] =
//     //     static_cast<T>(static_cast<float>(input[idx]) *
//     //                   //  quant_in_scale 
//     //                   //  / out_scale
//     //                   out_scale
//     //                    );
//   // }
// }

// template <typename T>
// void dequantize_kernel_launcher(const int32_t* input,
//                                 T* output,
//                                 const int m,    // m
//                                 const int n,  // n
//                                 gpuStream_t stream,
//                                 const float quant_in_scale,
//                                 const float* dequant_out_scale_data) {
//   dim3 block;
//   dim3 grid;
//   int threads_per_block = n > 1024 ? 1024 : n;
//   if (m < 1024) {
//     block = dim3(threads_per_block);
//     int blocks_per_line = n > 1024 ? (n + 1023) / 1024 : 1;
//     grid = dim3(blocks_per_line, m);
//   } else {
//     block = dim3(threads_per_block);
//     grid = dim3(1, 1024);
//   }


//   dequantize_kernel<<<grid, block, 0, stream>>>(output,
//                                                 input,
//                                                 m,
//                                                 n,
//                                                 quant_in_scale,
//                                                 dequant_out_scale_data);
// }

template <typename T, int VecSize>
__global__ void dequantize_kernel(T* output,
                                  const int32_t* input,
                                  const int m,  // batch size 
                                  const int n,  // hidden
                                  const float quant_in_scale,
                                  const float* dequant_out_scale_data) {
  int numel = m * n;
  int stride = blockDim.x * gridDim.x * VecSize;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  int col_id = idx % n;

  phi::AlignedVector<int32_t, VecSize> in_vec;
  phi::AlignedVector<float, VecSize> out_scale_vec;
  phi::AlignedVector<T, VecSize> out_vec;

  for (;idx < numel; idx += stride) {
    phi::Load<int32_t, VecSize>(input + idx, &in_vec);
    phi::Load<float, VecSize>(dequant_out_scale_data + col_id, &out_scale_vec);

  #pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      out_vec[i] = static_cast<T>(static_cast<float>(in_vec[i]) * out_scale_vec[i]);
    }

    phi::Store<T, VecSize>(out_vec, output + idx);
  }                                  
}

template <typename T>
void dequantize_kernel_launcher(const int32_t* input,
                                T* output,
                                const int m,    // m
                                const int n,  // n
                                gpuStream_t stream,
                                GpuLaunchConfig* gpu_config,
                                const float quant_in_scale,
                                const float* dequant_out_scale_data) {                             

  dequantize_kernel<T, DequantKernelVecSize><<<gpu_config->block_per_grid, 
                      gpu_config->thread_per_block, 0, stream>>>(output,
                                                input,
                                                m,
                                                n,
                                                quant_in_scale,
                                                dequant_out_scale_data);
}




}  // namespace operators
}  // namespace paddle
