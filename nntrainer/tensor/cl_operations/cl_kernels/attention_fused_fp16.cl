// DEBUG probe: sub_group_reduce_add + sub_group_size sanity check.
//
// Step 4 and 4b both produced garbage output despite algorithmically
// matching V0.  The reduce intrinsic is the only "magic" difference:
// all logic downstream of score is identical to V0 (step 4) or to
// the equivalent V1-style redundant-exp path (step 4b).  So if the
// reduce returns a wrong value, garbage propagates.
//
// This probe verifies two things on Adreno 830 with
// qcom_reqd_sub_group_size("full") on a 1D WG of 128 lanes:
//
//   (A) get_sub_group_size() should print 128.
//       If it prints 32 or 64, "full" didn't pin the wavefront width
//       and our reduce only covers a chunk of d, not all of HD.
//
//   (B) sub_group_reduce_add(1.0f) should be 128.0 (each of 128 lanes
//       contributes 1).  sub_group_reduce_add((float)d) should be
//       sum_{d=0..127} d = 8128.0.  If they show 32 / 496 instead,
//       A is 32 too.
//
// The kernel signature stays identical to prior steps so the C++
// dispatch (V0 shape) does not change.  A single printf from
// (m, h, d) = (0, 0, 0) fires once per dispatch = 36 lines per run
// (one per layer), enough to verify without flooding.
//
// The output buffer is filled with a harmless derived value so the
// compiler can't DCE the loads and the downstream model doesn't
// dereference stale memory.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD 128

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(HD, 1, 1)))
void attention_fused_fp16(
    __global const half *Q,
    __global const half *K_cache,
    __global const half *V_cache,
    __global half *out,
    const int M,
    const int T,
    const int from,
    const int num_heads_Q,
    const int gqa_size,
    const int is_causal,
    const float scale) {

  const int d = get_local_id(0);
  const int h = get_group_id(1);
  const int m = get_group_id(2);

  // Every lane must participate in a sub-group function, so we do
  // the probe reduces unconditionally (before any early return).
  const float probe_ones = sub_group_reduce_add(1.0f);
  const float probe_iota = sub_group_reduce_add((float)d);

  if (h >= num_heads_Q || m >= M) return;

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_q  = num_heads_Q  * HD;
  const int W_k  = num_heads_KV * HD;

  // Only one lane in the first WG of the first call prints, so we
  // get exactly 36 lines per prefill (one per layer).
  if (m == 0 && h == 0 && d == 0) {
    const uint sgsz = get_sub_group_size();
    const uint mwgs = get_max_sub_group_size();
    const uint ngrp = get_num_sub_groups();
    printf("[attn-dbg] sub_group_size=%u max=%u num_groups=%u "
           "ones=%f iota=%f local_size=(%u,%u,%u)\n",
           sgsz, mwgs, ngrp, probe_ones, probe_iota,
           (uint)get_local_size(0), (uint)get_local_size(1),
           (uint)get_local_size(2));
  }

  // Fill out with a harmless value derived from live inputs so the
  // model doesn't crash on NaN / stale memory downstream.
  const float q_val = (float)Q[m * W_q + h * HD + d];
  const float k0    = (float)K_cache[h_kv * HD + d];
  out[m * W_q + h * HD + d] = (half)((q_val + k0) * scale);
}
