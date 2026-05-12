// SPDX-License-Identifier: Apache-2.0
// GPU embedding lookup from INT4 chunked cache (for async pipeline).
//
// The INT4 chunked cache stores embedding rows transposed for GEMV:
//   chunk[k_group * cN + local_n]  (uint32, 8 nibbles)
//   k_group  = output element k / 8
//   local_n  = token_id - chunk_start[chunk_idx]
//   cN       = chunk_N[chunk_idx]
//
// Each work-item handles one FP16 output element (k in 0..K-1).
// Global size = roundup(K, 256). Local size = 256.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void gpu_embedding_int4chunked(
    __global const int    *token_id_svm,
    __global const uint   *chunk0,
    __global const uint   *chunk1,
    __global const uint   *chunk2,
    __global const ushort *scales,
    const int  chunk_N0,
    const int  chunk_N1,
    const int  chunk_N2,
    const int  chunk_start1,
    const int  chunk_start2,
    __global half *output,
    const int K,
    const float emb_scale
) {
    const int k = get_global_id(0);
    if (k >= K) return;

    const int token_id = token_id_svm[0];
    if (token_id < 0) {
        output[k] = (half)0.0f;
        return;
    }


    int cN, local_n;
    __global const uint *chunk;
    if (token_id < chunk_start1) {
        chunk = chunk0; cN = chunk_N0; local_n = token_id;
    } else if (token_id < chunk_start2) {
        chunk = chunk1; cN = chunk_N1; local_n = token_id - chunk_start1;
    } else {
        chunk = chunk2; cN = chunk_N2; local_n = token_id - chunk_start2;
    }

    const int k_group = k / 8;
    const int kk      = k % 8;
    const uint pack   = chunk[(long)k_group * (long)cN + (long)local_n];
    const int q       = (int)((pack >> ((uint)kk * 4u)) & 0xFu) - 8;

    // scales[token_id] is stored as raw FP16 bits in a ushort
    const float int4_scale = (float)as_half(scales[token_id]);
    output[k] = (half)(int4_scale * (float)q * emb_scale);
}

