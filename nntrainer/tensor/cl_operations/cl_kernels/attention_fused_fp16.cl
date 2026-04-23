// Fused FlashAttention-style Qwen attention, v4 (image1d_buffer K/V).
//
// Exhaustive V3 sweep showed multi-Q WG + tree reduce is bound by
// WG-level barriers that sync all participating wavefronts; multi-Q
// WG + sub_group_reduce produced wrong output because the Qualcomm
// driver does NOT form sub-groups in linear id order for 2D WGs.
// V0's single-wavefront WG is the best correct variant (1794 ms) but
// is still 2x over NEON (866 ms).
//
// The next lever is memory, not ILP. Per-layer attention reads
//   32 heads x 437 Q positions x ~220 avg kk x 128 HD x 2 bytes
//   x (K + V) = ~1.6 GB
// of K/V data each layer, 36 layers = ~57 GB per prefill. At
// Adreno 830's ~80 GB/s peak sustained bandwidth the theoretical
// floor is ~700 ms — which explains why NEON (866 ms) is close to
// optimal and we have ~900 ms of head-room vs V0.
//
// Two structural tricks collapse that head-room:
//
//   1. image1d_buffer_t binding for K/V.  Adreno has a dedicated
//      texture-cache / fetch unit separate from the generic L1 used
//      by `__global half *` loads.  For contiguous 128-thread WG
//      reads the two paths have similar first-touch bandwidth, but
//      the texture path caches across WGs.
//
//   2. GQA share: Qwen3 has gqa_size=4 (32 Q heads / 8 KV heads).
//      Four consecutive Q heads hit the SAME K-head and the SAME
//      V-head.  Those four WGs scheduled close together on the same
//      SM will find K/V hot in the texture cache -> ~4x effective
//      bandwidth on K/V for those heads.  Plain __global reads
//      would (mostly) miss L1 for each new WG.
//
// Kernel shape is identical to V0 apart from the K/V type change.
// Input K/V are still 1-D fp16 [T, num_heads_KV, HD] laid out
// contiguously; we bind them as image1d_buffer<half4>, width =
// T * num_heads_KV * HD / 4, so one image4 fetch covers 4
// consecutive half lanes.  Each thread reads one half4 (4 halves
// out of 128) and multiplies only the lane matching its d%4 — the
// other three lanes feed d+1/+2/+3 threads, so the whole WG issues
// HD/4 = 32 texture fetches per kk vs. 128 scalar global loads in V0.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define HD 128

__kernel __attribute__((reqd_work_group_size(HD, 1, 1)))
void attention_fused_fp16(
    __global const half          *Q,
    __read_only image1d_buffer_t  K_cache_img,
    __read_only image1d_buffer_t  V_cache_img,
    __global half                *out,
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
  if (h >= num_heads_Q || m >= M) return;

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_q  = num_heads_Q  * HD;
  const int W_k  = num_heads_KV * HD;

  // d % 4 and d / 4 get reused every kk, hoist them.
  const int d_hi = d >> 2;   // 0..31
  const int d_lo = d & 3;    // 0..3

  const float q_val = (float)Q[m * W_q + h * HD + d];

  __local float rescale_l;
  __local float score_exp_l;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc         = 0.0f;

  const int kk_end = (is_causal != 0) ? (from + m + 1) : T;

  for (int kk = 0; kk < kk_end; ++kk) {
    // Image1d_buffer is addressed in half4 units.  Each WG stride
    // across d=0..127 maps to 32 distinct half4 indices; four
    // consecutive threads (d_lo = 0..3) hit the SAME half4 at the
    // same cycle, which Adreno's texture-cache coalesces into a
    // single fetch.
    const int kv_base4 = (kk * W_k + h_kv * HD) >> 2;  // half4 idx base
    const half4 k4 = read_imageh(K_cache_img, kv_base4 + d_hi);
    const half4 v4 = read_imageh(V_cache_img, kv_base4 + d_hi);

    // Pick this thread's half from the fetched half4.
    const float k_val = (float)((half*)&k4)[d_lo];
    const float v_val = (float)((half*)&v4)[d_lo];

    const float partial = q_val * k_val;
    const float score = work_group_reduce_add(partial) * scale;

    if (d == 0) {
      const float prev_max = running_max;
      const float new_max  = fmax(prev_max, score);
      const float rescale  = (isinf(prev_max) && prev_max < 0.0f)
                             ? 0.0f
                             : exp(prev_max - new_max);
      const float score_exp = exp(score - new_max);
      rescale_l   = rescale;
      score_exp_l = score_exp;
      running_max = new_max;
      running_sum = running_sum * rescale + score_exp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float rescale   = rescale_l;
    const float score_exp = score_exp_l;
    acc = acc * rescale + score_exp * v_val;
  }

  __local float sum_l;
  if (d == 0) sum_l = running_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  out[m * W_q + h * HD + d] = (half)(acc / sum_l);
}
