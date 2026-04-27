// M=1 int4 gemv that reads input + writes output as image2d AND SVM.
//
// Avoids Adreno 830's coarse-grained SVM cross-kernel coherence
// hazard for the FC -> image-aware-consumer chain (which reads via
// image cache, fully ordered by OpenCL in-order queue semantics) and
// ALSO populates the SVM output so consumers that read SVM directly
// (mha_core's enqueueSVMMap entry fence on Q/K/V, addition_layer's
// add2_fp16_svm_cl, swiglu CPU NEON) see coherent data.  The double
// write costs only 4 extra half stores per work-item; the SVM
// coherence is handled by the consumer's existing SVMMap entry fence.
//
// Image layout (matches rmsnorm_image2d_v2 + svm_to_image2d_publish):
//   input  : image2d_t, width = M = 1, height = K/4 slices
//            One pixel (half4) holds input[k..k+3] in .s0..s3.
//   output : image2d_t, width = M = 1, height = N/4 slices
//   svm_out: __global half *, length N (per-channel fp16, layout
//            matches gpu_int4_gemv_adreno -- output[n] = scaled MAC).
//
// SVM args:
//   weights : ushort[(K/4) * N] channelwise int4 packed (write-once).
//   scales  : half[N] per-channel fp16 scale (write-once).
//
// Dispatch:
//   global = (N / 4)         (each WI produces 4 output channels)
//   local  = (16, 1, 1)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

kernel void
gpu_int4_gemv_image2d(__read_only image2d_t input,
                      __global const half *scales,
                      __write_only image2d_t output,
                      __global const ushort *weights,
                      __global half *svm_output,
                      const int K,
                      const int N) {
  const int n = get_global_id(0) * 4;
  if (n >= N) return;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  const int slices_k = K >> 2;
  for (int s = 0; s < slices_k; ++s) {
    const half4 in_v = read_imageh(input, smp, (int2)(0, s));
    const ushort4 packed = vload4(0, weights + s * N + n);

    // Lane k+0 (low 4 bits)
    const float in0 = (float)in_v.s0;
    acc0 += in0 * (float)((int)(packed.s0 & 0x000F) - 8);
    acc1 += in0 * (float)((int)(packed.s1 & 0x000F) - 8);
    acc2 += in0 * (float)((int)(packed.s2 & 0x000F) - 8);
    acc3 += in0 * (float)((int)(packed.s3 & 0x000F) - 8);

    // Lane k+1
    const float in1 = (float)in_v.s1;
    acc0 += in1 * (float)((int)((packed.s0 & 0x00F0) >> 4) - 8);
    acc1 += in1 * (float)((int)((packed.s1 & 0x00F0) >> 4) - 8);
    acc2 += in1 * (float)((int)((packed.s2 & 0x00F0) >> 4) - 8);
    acc3 += in1 * (float)((int)((packed.s3 & 0x00F0) >> 4) - 8);

    // Lane k+2
    const float in2 = (float)in_v.s2;
    acc0 += in2 * (float)((int)((packed.s0 & 0x0F00) >> 8) - 8);
    acc1 += in2 * (float)((int)((packed.s1 & 0x0F00) >> 8) - 8);
    acc2 += in2 * (float)((int)((packed.s2 & 0x0F00) >> 8) - 8);
    acc3 += in2 * (float)((int)((packed.s3 & 0x0F00) >> 8) - 8);

    // Lane k+3 (high 4 bits)
    const float in3 = (float)in_v.s3;
    acc0 += in3 * (float)((int)((packed.s0 & 0xF000) >> 12) - 8);
    acc1 += in3 * (float)((int)((packed.s1 & 0xF000) >> 12) - 8);
    acc2 += in3 * (float)((int)((packed.s2 & 0xF000) >> 12) - 8);
    acc3 += in3 * (float)((int)((packed.s3 & 0xF000) >> 12) - 8);
  }

  const half4 scale = vload4(0, scales + n);
  const half4 out_v = (half4)(
    (half)(acc0 * (float)scale.s0),
    (half)(acc1 * (float)scale.s1),
    (half)(acc2 * (float)scale.s2),
    (half)(acc3 * (float)scale.s3));

  // Write image2d (for image-aware consumers) + SVM (for SVM-reading
  // consumers like mha_core / addition / swiglu).  Coherence for the
  // SVM write is handled by the consumer's own SVMMap entry fence.
  write_imageh(output, (int2)(0, n >> 2), out_v);
  vstore4(out_v, 0, svm_output + n);
}
