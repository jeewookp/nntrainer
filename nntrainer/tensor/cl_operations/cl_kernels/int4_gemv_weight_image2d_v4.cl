// K-split (S=4) variant of gpu_int4_gemv_weight_image2d_v3 for the
// remaining per-FC FCs (o_proj at K=4096 N=2560, down_proj at K=9728
// N=2560). After QKV/Q_norm/K_norm got absorbed into qkv_norm_layer,
// the only per-FC FCs left have small N (2560) -- only 10 WGs at the
// existing 256-N-per-WG layout, undersubscribing Adreno's compute
// units. The fused_gemv (gate_up) which reaches 64 GB/s runs 76 WGs.
//
// Solution: split K work S=4 ways per output channel. Each WG now
// produces 64 outputs (16 N-groups × 4 channels) instead of 256, so
// for N=2560 we get 40 WGs (4x more parallelism). 4 K-lanes within
// each N-group cooperate -- each does K/4 K-positions, then a
// sub_group_shuffle_xor butterfly reduce combines partials.
//
// Layout (within WG, lane_id ∈ [0, 64)):
//   n_group = lane_id / 4   (0..15)
//   k_lane  = lane_id % 4   (0..3)
//   N output written at (wgid * 16 + n_group) * 4 + {0,1,2,3}
//   K range per lane: [k_lane * (K/4), (k_lane+1) * (K/4))
//
// Same RGBA32UI image2d layout as v2/v3 (no new data layout). Same
// 2x K-unroll per inner step (16 K-positions, 64 MACs / iter).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

__constant sampler_t weight_smp_v4 = CLK_NORMALIZED_COORDS_FALSE |
                                      CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define K_SPLIT 4

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void
gpu_int4_gemv_weight_image2d_v4(__constant half *input,
                                 __global const half *scales,
                                 __global half *output,
                                 __read_only image2d_t weights,
                                 const int K,
                                 const int N) {
  const int lid = get_local_id(0);
  const int wgid = get_group_id(0);

  const int n_group = lid / K_SPLIT;   // 0..15
  const int k_lane  = lid % K_SPLIT;   // 0..3

  const int n = (wgid * 16 + n_group) * 4;
  if (n >= N) return;

  const int n_pixel = n / 4;

  // K assumed multiple of 32 (K_SPLIT=4 × 8 K-pos per pixel × 1+).
  // Per-FC per-Qwen3 K values: 4096, 9728 -- both % 32 == 0.
  const int k_per_lane = K / K_SPLIT;
  const int k_start = k_lane * k_per_lane;
  const int k_end = k_start + k_per_lane;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  for (int k = k_start; k < k_end; k += 16) {
    const half8 in_lo = vload8(0, input + k);
    const half8 in_hi = vload8(0, input + k + 8);
    const uint4 packed_lo = read_imageui(weights, weight_smp_v4,
                                          (int2)(n_pixel, k / 8));
    const uint4 packed_hi = read_imageui(weights, weight_smp_v4,
                                          (int2)(n_pixel, k / 8 + 1));

#define STEP(packed, in_v, lane_idx, shift)                            \
  do {                                                                 \
    const float in_k = (float)in_v.s##lane_idx;                        \
    acc0 += in_k * (float)((int)((packed.s0 >> shift) & 0xF) - 8);     \
    acc1 += in_k * (float)((int)((packed.s1 >> shift) & 0xF) - 8);     \
    acc2 += in_k * (float)((int)((packed.s2 >> shift) & 0xF) - 8);     \
    acc3 += in_k * (float)((int)((packed.s3 >> shift) & 0xF) - 8);     \
  } while (0)

    STEP(packed_lo, in_lo, 0,  0);
    STEP(packed_lo, in_lo, 1,  4);
    STEP(packed_lo, in_lo, 2,  8);
    STEP(packed_lo, in_lo, 3, 12);
    STEP(packed_lo, in_lo, 4, 16);
    STEP(packed_lo, in_lo, 5, 20);
    STEP(packed_lo, in_lo, 6, 24);
    STEP(packed_lo, in_lo, 7, 28);
    STEP(packed_hi, in_hi, 0,  0);
    STEP(packed_hi, in_hi, 1,  4);
    STEP(packed_hi, in_hi, 2,  8);
    STEP(packed_hi, in_hi, 3, 12);
    STEP(packed_hi, in_hi, 4, 16);
    STEP(packed_hi, in_hi, 5, 20);
    STEP(packed_hi, in_hi, 6, 24);
    STEP(packed_hi, in_hi, 7, 28);
#undef STEP
  }

  // Butterfly reduce across the K_SPLIT lanes that share an n_group.
  // sub_group_shuffle_xor with offset 1 swaps adjacent lanes (even
  // <-> odd), offset 2 swaps the pair-pairs. After both, each of the
  // 4 lanes in the group holds the full sum.
  acc0 += sub_group_shuffle_xor(acc0, 1);
  acc1 += sub_group_shuffle_xor(acc1, 1);
  acc2 += sub_group_shuffle_xor(acc2, 1);
  acc3 += sub_group_shuffle_xor(acc3, 1);
  acc0 += sub_group_shuffle_xor(acc0, 2);
  acc1 += sub_group_shuffle_xor(acc1, 2);
  acc2 += sub_group_shuffle_xor(acc2, 2);
  acc3 += sub_group_shuffle_xor(acc3, 2);

  // Lane 0 of each n_group writes the 4 output channels.
  if (k_lane == 0) {
    const half4 scale = vload4(0, scales + n);
    output[n + 0] = (half)(acc0 * (float)scale.s0);
    output[n + 1] = (half)(acc1 * (float)scale.s1);
    output[n + 2] = (half)(acc2 * (float)scale.s2);
    output[n + 3] = (half)(acc3 * (float)scale.s3);
  }
}
