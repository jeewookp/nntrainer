#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void swiglu_cl_fp16(__global const half *in1, __global const half *in2,
                             __global half *out) {
  const int i = get_global_id(0);

  const half in1_val = in1[i];
  const half in2_val = in2[i];

  // Compute sigmoid(in1) * in1 in fp32 to avoid the fp16 exp() overflow
  // the original kernel had: `exp(in1)` with moderate-magnitude gate
  // activations (|in1| > ~11) turns into +Inf and the swish = in1*e/(1+e)
  // then hits inf/inf = NaN. The NEON CPU reference
  // (neon_impl_fp16.cpp: swiglu) already promotes to fp32 for exp() for
  // the same reason.
  const float in1_f = (float)in1_val;
  const float sigmoid_f = 1.0f / (1.0f + exp(-in1_f));
  const half swish = (half)(in1_f * sigmoid_f);

  out[i] = swish * in2_val;
}
