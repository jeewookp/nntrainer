#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// RMSNorm for image2d tensors.
// Input/output: image2d (width=M, height=hidden_dim/4, RGBA half)
// Gamma: __global half* [hidden_dim]
//
// For each position m: output[m,c] = input[m,c] * gamma[c] / rms
// where rms = sqrt(mean(input[m,:]^2) + epsilon)
//
// Each work group handles one position m.
// Work items within the group cooperatively reduce across channels.

__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void rmsnorm_image2d(
    __read_only image2d_t input,
    __write_only image2d_t output,
    __global const half* gamma,
    const half epsilon,
    const int M,          // width (seq_len)
    const int slices) {   // height (hidden_dim / 4)

  int m = get_global_id(0);
  if (m >= M) return;

  // Pass 1: compute sum of squares across all channels
  float sum_sq = 0.0f;
  for (int s = 0; s < slices; ++s) {
    half4 v = read_imageh(input, smp, (int2)(m, s));
    sum_sq += (float)(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
  }

  float rms = sqrt(sum_sq / (float)(slices * 4) + (float)epsilon);
  half inv_rms = (half)(1.0f / rms);

  // Pass 2: normalize and apply gamma
  for (int s = 0; s < slices; ++s) {
    half4 v = read_imageh(input, smp, (int2)(m, s));
    int c = s * 4;
    half4 g = (half4)(gamma[c], gamma[c+1], gamma[c+2], gamma[c+3]);
    half4 out = v * inv_rms * g;
    write_imageh(output, (int2)(m, s), out);
  }
}
