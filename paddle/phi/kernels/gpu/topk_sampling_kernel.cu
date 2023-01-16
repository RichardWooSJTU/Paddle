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

#include "paddle/phi/kernels/topk_sampling_kernel.h"

namespace phi {

__device__ __forceinline__ bool gt(float a, float b) { return a > b; }

__device__ __forceinline__ bool gt(half a, half b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

template <typename T>
struct TopKPair {
  __device__ __forceinline__ TopKPair() {
    v_ = -((std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX);
    id_ = -1;
  }

  __device__ __forceinline__ void insert(int id, T v) {
    if (gt(v, v_)) {
      v_ = v;
      id_ = id;
    }
  }

  T v_;
  int id_;
};

template <typename T>
__device__ __forceinline__ TopKPair<T> reduce_topk_pair(const TopKPair<T>& a,
                                                        const TopKPair<T>& b) {
  return gt(a.v_, b.v_) ? a : b;
}

__global__ void CurandInitialize(
    curandState_t* state,
    const int size,
    const unsigned long long random_seed) {  // NOLINT
  if (threadIdx.x + blockIdx.x * blockDim.x < size) {
    curand_init(
        random_seed, 0, 0, &state[blockIdx.x * blockDim.x + threadIdx.x]);
  }
}

// grid: x: num_batch, y:num_blocks_per_beam
// one beam (num_blocks_per_beam blocks) process a batch, which contains
// vocab_size items
template <typename T, int NUM_THREADS_PER_BLOCK>
__global__ void PrepareTopK(const T* src_probs,  // bsz * vocab_size
                            T* tmp_probs,        // bsz * vocab_size
                            T* tmp_vals,       // bsz * num_blocks_per_beam * k
                            int64_t* tmp_ids,  // bsz * num_blocks_per_beam * k
                            int k,
                            int vocab_size) {
  int num_blocks_per_beam = gridDim.y;
  int num_threads_per_block = blockDim.x;
  int batch_id = blockIdx.x;
  int block_id_beam = blockIdx.y;

  int stride = num_blocks_per_beam * num_threads_per_block;

  typedef cub::BlockReduce<TopKPair<T>, NUM_THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int iter = 0; iter < k; ++iter) {
    TopKPair<T> local_top;
    for (int idx = block_id_beam * num_threads_per_block + threadIdx.x;
         idx < vocab_size;
         idx += stride) {
      if (iter == 0) {
        tmp_probs[idx] = src_probs[batch_id * vocab_size + idx];
      }
      local_top.insert(idx, tmp_probs[idx]);
    }
    TopKPair<T> block_top =
        BlockReduce(temp_storage).Reduce(local_top, reduce_topk_pair<T>);

    if (threadIdx.x == 0) {
      // printf("%d, %f\n", batch_id * num_blocks_per_beam * k + block_id_beam *
      // k + iter, block_top.v_);
      tmp_ids[batch_id * num_blocks_per_beam * k + block_id_beam * k + iter] =
          block_top.id_;
      tmp_vals[batch_id * num_blocks_per_beam * k + block_id_beam * k + iter] =
          block_top.v_;
      tmp_probs[block_top.id_] =
          std::is_same<T, half>::value ? -HALF_FLT_MAX : -FLT_MAX;
    }

    __syncthreads();
  }
}

// grid: x: num_batch
// block: x: num_blocks_per_beam * k
template <typename T, int NUM_THREADS_PER_BLOCK>
__global__ void TopKSampling(
    const T* tmp_vals,       // bsz * num_blocks_per_beam * k
    const int64_t* tmp_ids,  // bsz * num_blocks_per_beam * k
    T* vals,                 // bsz * 1
    int64_t* ids,            // bsz * 1
    int k,
    int num_blocks_per_beam,
    curandState_t* curandstate) {
  int batch_id = blockIdx.x;
  int stride = NUM_THREADS_PER_BLOCK;

  typedef cub::BlockReduce<TopKPair<T>, NUM_THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  extern __shared__ unsigned char smem[];  // num_blocks_per_beam * k
  int64_t* topk_idxs = reinterpret_cast<int64_t*>(smem);
  T* topk_probs = reinterpret_cast<T*>(topk_idxs + k);
  T* tmp_probs = topk_probs + k;

  float sum_probs;

  for (int iter = 0; iter < k; ++iter) {
    TopKPair<T> local_top;
    for (int i = threadIdx.x; i < num_blocks_per_beam * k; i += stride) {
      if (iter == 0) {
        tmp_probs[i] = tmp_vals[batch_id * num_blocks_per_beam * k + i];
        // printf("%d, %d, %f\n", batch_id, i, tmp_probs[i]);
      }
      local_top.insert(i, tmp_probs[i]);
    }
    TopKPair<T> block_top =
        BlockReduce(temp_storage).Reduce(local_top, reduce_topk_pair<T>);
    if (threadIdx.x == 0) {
      // printf("%d, %d, %f, %d\n", batch_id, iter, block_top.v_,
      // tmp_ids[batch_id * num_blocks_per_beam * k + block_top.id_]);
      topk_probs[iter] = block_top.v_;
      topk_idxs[iter] =
          tmp_ids[batch_id * num_blocks_per_beam * k + block_top.id_];
      tmp_probs[block_top.id_] =
          std::is_same<T, half>::value ? -HALF_FLT_MAX : -FLT_MAX;

      sum_probs += static_cast<float>(block_top.v_);
    }

    __syncthreads();
  }

  // Sampling
  if (threadIdx.x == 0) {
    // printf("sum_probs %f\n", sum_probs);
    float random_prob = static_cast<float>(
        curand_uniform(curandstate + blockIdx.x) * sum_probs);
    // printf("%f\n", random_prob);
    for (int i = 0; i < k; ++i) {
      random_prob -= static_cast<float>(topk_probs[i]);
      if (random_prob <= 0 || i == k - 1) {
        // printf("%d\n %f, %d\n", i, topk_probs[i], topk_idxs[i]);
        vals[batch_id] = topk_probs[i];
        ids[batch_id] = topk_idxs[i];
        break;
      }
    }
  }
}

template <typename T, typename Context>
void TopKSamplingKernel(const Context& ctx,
                        const DenseTensor& probs,
                        int k,
                        int random_seed,
                        DenseTensor* topk_scores,
                        DenseTensor* topk_indices) {
  int num_batch = probs.dims[0];
  int vocab_size = probs.dims[1];

  topk_scores->Resize({{num_batch, 1}});
  ctx.template Alloc<T>(topk_scores);
  topk_indices->Resize({{num_batch, 1}});
  ctx.template Alloc<int64_t>(topk_indices);

  constexpr int NUM_THREADS_PER_BLOCK = 128;
  constexpr int NUM_BLOCKS_PER_BEAM = 8;

  // Step 1. calculate topk for per block
  DenseTensor tmp_vals, tmp_ids, workspace;
  tmp_vals.Resize({{static_cast<int64_t>(sizeof(T) * num_batch *
                                         NUM_BLOCKS_PER_BEAM * k)}});
  ctx.template Alloc<T>(&tmp_vals);

  tmp_ids.Resize({{static_cast<int64_t>(sizeof(int64_t) * num_batch *
                                        NUM_BLOCKS_PER_BEAM * k)}});
  ctx.template Alloc<int64_t>(&tmp_ids);

  workspace.Resize(
      {{static_cast<int64_t>(sizeof(nv_data_t) * num_batch * vocab_size)}});
  ctx.template Alloc<T>(&workspace);

  // nv_data_t* tmp_vals_data =
  // reinterpret_cast<nv_data_t*>(tmp_vals.data<pd_data_t>()); int64_t*
  // tmp_ids_data = tmp_ids.data<int64_t>(); nv_data_t* workspace_data =
  // reinterpret_cast<nv_data_t*>(workspace.data<pd_data_t>());

  dim3 grid(num_batch, NUM_BLOCKS_PER_BEAM);
  PrepareTopK<nv_data_t, NUM_THREADS_PER_BLOCK>
      <<<grid, NUM_THREADS_PER_BLOCK, 0, ctx.stream>>>(probs.data<T>(),
                                                       workspace.data<T>(),
                                                       tmp_vals.data<T>(),
                                                       tmp_ids.data<int64_t>(),
                                                       k,
                                                       vocab_size);

  // Step 2. reduce all top k of a batch and sampling one for each batch
  curandState_t* curand_stat;

  DenseTensor curand_states_buf;
  curand_states_buf.Resize(
      {{num_batch * static_cast<int64_t>(sizeof(curandState_t))}});
  ctx.template Alloc<uint8_t>(&curand_states_buf);

  curand_stat =
      reinterpret_cast<curandState_t*>(curand_states_buf.data<uint8_t>());

  CurandInitialize<<<num_batch, 1, 0, cu_stream>>>(
      curand_stat, num_batch, random_seed);
  TopKSampling<nv_data_t, NUM_THREADS_PER_BLOCK>
      <<<num_batch,
         NUM_THREADS_PER_BLOCK,
         NUM_BLOCKS_PER_BEAM * k * sizeof(T) + k * sizeof(T) +
             k * sizeof(int64_t),
         cu_stream>>>(tmp_vals.data<T>(),
                      tmp_ids.data<int64_t>(),
                      topk_scores.data<T>(),
                      topk_indices.data<int64_t>(),
                      k,
                      NUM_BLOCKS_PER_BEAM,
                      curand_stat);
}

}  // namespace phi
