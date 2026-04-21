#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 2-input elementwise fp16 add for SVM residual connections.
// out[i] = a[i] + b[i]   for i in [0, total).
__kernel void add2_fp16_svm(__global const half *a,
                            __global const half *b,
                            __global half *out,
                            int total) {
  const int i = get_global_id(0);
  if (i >= total) return;
  out[i] = a[i] + b[i];
}
