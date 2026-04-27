#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// SwiGLU for image2d tensors.
// output[m,s] = gate[m,s] * silu(up[m,s])
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

// Renamed from swiglu_image2d -> swiglu_image2d_v2 to dodge nntrainer's
// kernel cache (cl_context.cpp:289 keys on kernel_name + compile_options
// only, NOT the source string).  The original swiglu_image2d binary is
// cached on disk under the old 5-arg signature and gets handed back
// regardless of any source change, which is why our SVM-output sentinel
// stayed 0 across edits to the .cl file.
__kernel void swiglu_image2d_v2(
    __read_only image2d_t gate,
    __read_only image2d_t up,
    __write_only image2d_t output,
    __global half *svm_output,
    const int M,
    const int slices,
    const int K) {

  int m = get_global_id(0);
  int s = get_global_id(1);

  // DIAG: confirm SVM arg binding + dispatch landed.  Every WI
  // unconditionally stamps a unique sentinel at position
  // (m * K + s*4) regardless of M/slices guard so the host can see:
  //  - svm_output[0] = 42.0           : kernel reached at all
  //  - svm_output[(M-1)*K + ...]       : last WI also reached
  //  - svm_output[non-WI position]    : remains 0 (no random writes)
  // After the diag we'd expect at most M*slices*4 sentinels.
  if (m == 0 && s == 0) {
    svm_output[0] = (half)42.0h;
  }

  if (m >= M || s >= slices) return;

  half4 g_h = read_imageh(gate, smp, (int2)(m, s));
  half4 u_h = read_imageh(up, smp, (int2)(m, s));

  // silu(x) = x / (1 + exp(-x))
  // Compute in fp32 -- half-precision exp() accuracy is driver-
  // dependent on Adreno and was producing wrong activations in our
  // first integration test.  fp32 silu + final half conversion
  // matches the SVM swiglu_fp16 path's NEON math.
  float4 g = convert_float4(g_h);
  float4 u = convert_float4(u_h);
  float4 silu;
  silu.x = u.x / (1.0f + exp(-u.x));
  silu.y = u.y / (1.0f + exp(-u.y));
  silu.z = u.z / (1.0f + exp(-u.z));
  silu.w = u.w / (1.0f + exp(-u.w));
  half4 out_v = convert_half4(g * silu);

  // Write image2d (for image-aware consumers) AND SVM (for SVM-reading
  // consumers + correctness verification).  Mirrors gpu_int4_gemv_image2d.
  vstore4(out_v, 0, svm_output + (m * K + s * 4));
  write_imageh(output, (int2)(m, s), out_v);
}
