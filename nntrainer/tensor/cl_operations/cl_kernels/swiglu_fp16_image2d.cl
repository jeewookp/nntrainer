// Phase A4 (activation image2d): swiglu fp16 with image2d INPUT
// reads.  Same per-element math as swiglu_fp16.cl but reads in1/in2
// from CL_RGBA + CL_HALF_FLOAT image2d views over the SVM gate/up
// output buffers (zero-copy via cl_khr_image2d_from_buffer).
//
// Output stays __global half (SVM) -- the next consumer (ffn_down
// gemv) reads it as __constant SVM.
//
// Dispatch: 1 work-item per 4-element pixel.
//   global = total / 4   (total = M * K)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t swi_smp = CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void swiglu_cl_fp16_image2d(__read_only image2d_t in1,
                                      __read_only image2d_t in2,
                                      __global half *out,
                                      const int total_pixels) {
  const int p = get_global_id(0);
  if (p >= total_pixels) return;

  const half4 v1 = read_imageh(in1, swi_smp, (int2)(p, 0));
  const half4 v2 = read_imageh(in2, swi_smp, (int2)(p, 0));

  // Match neon_impl_fp16.cpp swiglu exactly: fp32 throughout.
  const float a0 = (float)v1.s0, b0 = (float)v2.s0;
  const float a1 = (float)v1.s1, b1 = (float)v2.s1;
  const float a2 = (float)v1.s2, b2 = (float)v2.s2;
  const float a3 = (float)v1.s3, b3 = (float)v2.s3;
  const float s0 = 1.0f / (1.0f + exp(-a0));
  const float s1 = 1.0f / (1.0f + exp(-a1));
  const float s2 = 1.0f / (1.0f + exp(-a2));
  const float s3 = 1.0f / (1.0f + exp(-a3));
  half4 r;
  r.s0 = (half)(a0 * s0 * b0);
  r.s1 = (half)(a1 * s1 * b1);
  r.s2 = (half)(a2 * s2 * b2);
  r.s3 = (half)(a3 * s3 * b3);
  vstore4(r, p, out);
}
