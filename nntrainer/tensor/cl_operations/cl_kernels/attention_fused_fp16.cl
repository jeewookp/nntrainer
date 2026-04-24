// Step 4 / incremental: step 3b + online softmax + proper acc.
//
// Timings so far (qk portion, 36 layers, M=437):
//   Step 1  =  170 ms   K load only
//   Step 2  =  341 ms   + V load + Q load + per-lane partial
//   Step 3  = 1872 ms   + work_group_reduce_add per kk
//   Step 3b =  997 ms   sub_group_reduce_add (1-wavefront WG)
//   V0      = 1794 ms   full kernel via work_group_reduce_add
//   NEON    =  866 ms   CPU reference
//
// Step 3b proved sub_group_reduce is ~2.3x cheaper than
// work_group_reduce on a 1-wavefront WG.  On top of that reduce cost
// (~655 ms after subtracting memory), the missing pieces to make
// V0 correct are:
//   * online softmax (running_max, running_sum, rescale, score_exp)
//   * acc update via `acc * rescale + score_exp * v_val`
// Both trivial in flops; any cost is from the __local broadcast of
// rescale / score_exp + the one WG-level barrier per kk.
//
// Step 4 = step 3b + that softmax / rescale / proper acc logic.  It
// should match V0 behaviour (correct output) and land close to the
// step 3b timing (~1000 ms).  If softmax + rescale costs are
// significant, delta vs step 3b shows the overhead.  The single
// barrier here is WG-wide on a 1-wavefront WG, so it's the cheap
// kind.

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
  if (h >= num_heads_Q || m >= M) return;

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_q  = num_heads_Q  * HD;
  const int W_k  = num_heads_KV * HD;

  const float q_val = (float)Q[m * W_q + h * HD + d];

  __local float rescale_l;
  __local float score_exp_l;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc         = 0.0f;

  const int kk_end = (is_causal != 0) ? (from + m + 1) : T;

  for (int kk = 0; kk < kk_end; ++kk) {
    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];
    const float score = sub_group_reduce_add(q_val * k_val) * scale;

    // d==0 advances the running softmax state and publishes
    // rescale / score_exp via __local for the other 127 lanes.
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
    const float v_val = (float)V_cache[kk * W_k + h_kv * HD + d];
    acc = acc * rescale + score_exp * v_val;
  }

  __local float sum_l;
  if (d == 0) sum_l = running_sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  out[m * W_q + h * HD + d] = (half)(acc / sum_l);
}
