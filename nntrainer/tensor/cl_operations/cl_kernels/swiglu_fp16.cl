#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void swiglu_cl_fp16(__global const half *in1, __global const half *in2,
                             __global half *out) {
  const int i = get_global_id(0);

  // Match the NEON reference (neon_impl_fp16.cpp: swiglu) exactly:
  // promote everything to fp32, do the whole swish(in1) * in2 in fp32,
  // cast to fp16 only when storing out[i]. The old kernel computed
  // exp(in1) in fp16 and overflowed to +Inf for moderately large
  // gate activations (|y| > ~11), producing inf/inf = NaN that wrecked
  // the subsequent down_proj. Casting the fp32 swish back to fp16
  // _before_ the * in2 multiply (intermediate step) also loses
  // precision on large magnitudes — keep it in fp32 until the last
  // step.
  const float in1_f = (float)in1[i];
  const float in2_f = (float)in2[i];
  const float sigmoid_f = 1.0f / (1.0f + exp(-in1_f));
  out[i] = (half)(in1_f * sigmoid_f * in2_f);
}
