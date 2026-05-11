// Variant of attention_fused_fp16_decode_v3 that writes directly to a
// __write_only image2d_t instead of a __global half * SVM buffer.
//
// This eliminates the separate svm_to_image2d kernel dispatch that normally
// follows the attention kernel, saving ~20us host overhead per decode step
// (~47ms across a 64-token generation run with 36 layers).
//
// Output image2d layout (matches svm_to_image2d and what fused_gemv_int4
// image2d expects):
//   image2d width  = M = 1  (decode)
//   image2d height = K/4  where K = num_heads_Q * HD
//   pixel (m=0, s).rgba = output[s*4 .. s*4+3]
//
// For thread d (0..WG-1) and head h:
//   i=0: writes component d%4 of pixel (0, h*(HD/4) + d/4)
//   i=1: writes component d%4 of pixel (0, h*(HD/4) + 16 + d/4)
//
// We use __local tmp[WG] to gather the 4 components from consecutive
// threads before issuing write_imageh (which requires a half4).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD  128
#define WG  64
#define DPT (HD / WG)  // 2
#define KK_BLOCK 4

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attention_fused_fp16_decode_v3_image2d(
    __global const half *Q,
    __global const half *K_cache,
    __global const half *V_cache,
    __write_only image2d_t out_img,
    __global half *out_svm,
    const int M,
    const int T,
    const int from,
    const int num_heads_Q,
    const int gqa_size,
    const int is_causal,
    const float scale) {

  const int d = get_local_id(0);
  const int h = get_group_id(1);

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_k = num_heads_KV * HD;

  const int kk_end = (is_causal != 0) ? (from + 1) : T;

  float q_val[DPT];
  #pragma unroll
  for (int i = 0; i < DPT; ++i) {
    const int dd = d + i * WG;
    q_val[i] = (float)Q[h * HD + dd];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc[DPT];
  #pragma unroll
  for (int i = 0; i < DPT; ++i) acc[i] = 0.0f;

  const int kk_blocks = kk_end / KK_BLOCK;
  const int kk_tail = kk_end & (KK_BLOCK - 1);

  int kk = 0;
  for (int p = 0; p < kk_blocks; ++p) {
    float k_v[KK_BLOCK][DPT];
    float v_v[KK_BLOCK][DPT];
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        const int dd = d + i * WG;
        const int base = (kk + b) * W_k + h_kv * HD + dd;
        k_v[b][i] = (float)K_cache[base];
        v_v[b][i] = (float)V_cache[base];
      }
    }

    float partials[KK_BLOCK];
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      float partial = 0.0f;
      #pragma unroll
      for (int i = 0; i < DPT; ++i) partial += q_val[i] * k_v[b][i];
      partials[b] = sub_group_reduce_add(partial) * scale;
    }

    const float prev_max = running_max;
    float block_max = partials[0];
    #pragma unroll
    for (int b = 1; b < KK_BLOCK; ++b) block_max = fmax(block_max, partials[b]);
    const float new_max = fmax(prev_max, block_max);
    const float rescale = (isinf(prev_max) && prev_max < 0.0f)
                            ? 0.0f
                            : native_exp(prev_max - new_max);

    float block_sum = 0.0f;
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      partials[b] = native_exp(partials[b] - new_max);
      block_sum += partials[b];
    }

    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      float a = acc[i] * rescale;
      #pragma unroll
      for (int b = 0; b < KK_BLOCK; ++b) a += partials[b] * v_v[b][i];
      acc[i] = a;
    }
    running_sum = running_sum * rescale + block_sum;
    running_max = new_max;

    kk += KK_BLOCK;
  }

  for (int t = 0; t < kk_tail; ++t) {
    float partial = 0.0f;
    float v_val[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      const int base = (kk + t) * W_k + h_kv * HD + dd;
      const float k_v_t = (float)K_cache[base];
      v_val[i]          = (float)V_cache[base];
      partial += q_val[i] * k_v_t;
    }
    const float score = sub_group_reduce_add(partial) * scale;
    const float prev_max = running_max;
    const float new_max  = fmax(prev_max, score);
    const float rescale  = (isinf(prev_max) && prev_max < 0.0f)
                             ? 0.0f
                             : native_exp(prev_max - new_max);
    const float score_exp = native_exp(score - new_max);
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      acc[i] = acc[i] * rescale + score_exp * v_val[i];
    }
    running_sum = running_sum * rescale + score_exp;
    running_max = new_max;
  }

  const float inv = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;

  // SVM write: identical to v3 kernel -- every thread stores its own
  // half directly, matching the per-element store pattern that v3 uses
  // and that Adreno SVM coherence expects for the zerocopy path.
  #pragma unroll
  for (int i = 0; i < DPT; ++i) {
    const int dd = d + i * WG;
    out_svm[h * HD + dd] = (half)(acc[i] * inv);
  }

  // image2d write: gather 4 consecutive values via local memory, then
  // write_imageh once per pixel (requires half4).
  __local float tmp[WG];

  // i=0: dd = d, positions h*HD + 0..63, pixels h*32 + 0..15
  tmp[d] = acc[0] * inv;
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((d & 3) == 0) {
    half4 pix;
    pix.x = (half)tmp[d];
    pix.y = (half)tmp[d + 1];
    pix.z = (half)tmp[d + 2];
    pix.w = (half)tmp[d + 3];
    write_imageh(out_img, (int2)(0, h * (HD / 4) + d / 4), pix);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // i=1: dd = d+64, positions h*HD + 64..127, pixels h*32 + 16..31
  tmp[d] = acc[1] * inv;
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((d & 3) == 0) {
    half4 pix;
    pix.x = (half)tmp[d];
    pix.y = (half)tmp[d + 1];
    pix.z = (half)tmp[d + 2];
    pix.w = (half)tmp[d + 3];
    write_imageh(out_img, (int2)(0, h * (HD / 4) + 16 + d / 4), pix);
  }
}
