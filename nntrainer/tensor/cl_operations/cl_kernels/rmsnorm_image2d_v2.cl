// RMSNorm on image2d tensors with cooperative workgroup reduction.
//
// Input  : image2d (width = M positions, height = K/4 slices, RGBA half)
// Gamma  : __global const float* [K]     (fp32 per-channel scale; kept fp32
//          at layer finalize time to match the checkpoint file layout)
// Output : image2d (same shape as input)
//
// For each position m:
//   sum_sq = sum_c(x[m,c]^2)
//   inv_rms = 1 / sqrt(sum_sq / K + eps)
//   y[m,c] = x[m,c] * inv_rms * gamma[c]
//
// Dispatch: global = (WG_SIZE * M, 1, 1), local = (WG_SIZE, 1, 1).
// One workgroup normalises one position; WG_SIZE=64 matches Adreno
// wavefront so the tree reduction runs one-step-per-pair on the
// subgroup. For narrow K (e.g. head_dim=128 -> slices=32) the upper
// half of the workgroup sits idle in pass 1 but still participates in
// the tree reduction with zero-initialised shared_sum slots, so the
// result stays correct.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define WG_SIZE 64

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void rmsnorm_image2d_v2(
    __read_only  image2d_t input,
    __write_only image2d_t output,
    __global const float * restrict gamma,
    const float epsilon,
    const int M,
    const int slices) {

  const int lid = get_local_id(0);
  const int m = get_group_id(0);
  if (m >= M) return;

  // -------- Pass 1: strided read + local square accumulation --------
  float sum_sq = 0.0f;
  for (int s = lid; s < slices; s += WG_SIZE) {
    half4 v = read_imageh(input, smp, (int2)(m, s));
    float4 f = convert_float4(v);
    sum_sq += f.x * f.x + f.y * f.y + f.z * f.z + f.w * f.w;
  }

  // -------- Tree reduction across the workgroup --------
  __local float shared_sum[WG_SIZE];
  shared_sum[lid] = sum_sq;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int step = WG_SIZE / 2; step > 0; step >>= 1) {
    if (lid < step) {
      shared_sum[lid] += shared_sum[lid + step];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  const float mean = shared_sum[0] / (float)(slices * 4);
  const float inv_rms = 1.0f / sqrt(mean + epsilon);

  // -------- Pass 2: normalise, scale by gamma, write --------
  for (int s = lid; s < slices; s += WG_SIZE) {
    half4 v = read_imageh(input, smp, (int2)(m, s));
    const int c = s * 4;
    float4 g = (float4)(gamma[c], gamma[c + 1], gamma[c + 2], gamma[c + 3]);
    float4 y = convert_float4(v) * inv_rms * g;
    write_imageh(output, (int2)(m, s), convert_half4(y));
  }
}
