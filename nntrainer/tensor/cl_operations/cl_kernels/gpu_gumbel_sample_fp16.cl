// GPU temperature sampling via Gumbel max trick.
//
// Mathematically equivalent to categorical sampling from softmax(logits/T):
//   argmax_i (logit_i / T + Gumbel_i)  ~  Categorical(softmax(logits/T))
//
// Single workgroup (256 threads), strided loop + tree reduction.
// Reads all n FP16 logits, writes the sampled index to result[0].
//
// Global: { 256, 1, 1 }   Local: { 256, 1, 1 }
// When inv_temperature == 0, degenerates to plain argmax (greedy).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void gpu_gumbel_sample_fp16(
    __global const half *logits,
    __global int        *result,
    const int            n,
    const float          inv_temperature,
    const uint           seed) {

  __local float s_val[256];
  __local int   s_idx[256];

  const int tid = get_local_id(0);
  float best_val = -1e38f;
  int   best_idx = 0;

  for (int i = tid; i < n; i += 256) {
    float v = (float)logits[i];
    if (inv_temperature > 0.0f) {
      // Per-element hash RNG (Wang hash) -> Gumbel noise
      uint h = seed ^ ((uint)i * 2654435761u);
      h ^= h >> 16; h *= 0x45d9f3bu; h ^= h >> 16;
      // Map to uniform (0,1), avoid exact 0/1 for log stability
      float u = clamp((float)(h >> 9) * (1.0f / 8388608.0f), 1e-7f, 1.0f - 1e-7f);
      float gumbel = -native_log(-native_log(u));
      v = v * inv_temperature + gumbel;
    }
    if (v > best_val) { best_val = v; best_idx = i; }
  }

  s_val[tid] = best_val;
  s_idx[tid] = best_idx;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s && s_val[tid + s] > s_val[tid]) {
      s_val[tid] = s_val[tid + s];
      s_idx[tid] = s_idx[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) result[0] = s_idx[0];
}
