#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// SwiGLU for image2d tensors.
// output[m,s] = gate[m,s] * silu(up[m,s])
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void swiglu_image2d(
    __read_only image2d_t gate,
    __read_only image2d_t up,
    __write_only image2d_t output,
    const int M,
    const int slices) {

  int m = get_global_id(0);
  int s = get_global_id(1);
  if (m >= M || s >= slices) return;

  half4 g_h = read_imageh(gate, smp, (int2)(m, s));
  half4 u_h = read_imageh(up, smp, (int2)(m, s));

  // silu(x) = x / (1 + exp(-x))
  // Compute in fp32 -- half-precision exp() accuracy is driver-
  // dependent on Adreno and was producing wrong activations in our
  // first integration test (model output became random multilingual
  // tokens).  fp32 silu + final half conversion matches the SVM
  // swiglu_fp16 path's NEON math (which also computes silu in fp32)
  // and avoids the discrepancy.
  float4 g = convert_float4(g_h);
  float4 u = convert_float4(u_h);
  float4 silu;
  silu.x = u.x / (1.0f + exp(-u.x));
  silu.y = u.y / (1.0f + exp(-u.y));
  silu.z = u.z / (1.0f + exp(-u.z));
  silu.w = u.w / (1.0f + exp(-u.w));

  write_imageh(output, (int2)(m, s), convert_half4(g * silu));
}
