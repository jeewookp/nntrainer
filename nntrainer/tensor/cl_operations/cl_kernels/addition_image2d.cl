#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Element-wise addition for image2d tensors.
// output[m,s] = a[m,s] + b[m,s]

__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void addition_image2d(
    __read_only image2d_t a,
    __read_only image2d_t b,
    __write_only image2d_t output,
    const int M,
    const int slices) {

  int m = get_global_id(0);
  int s = get_global_id(1);
  if (m >= M || s >= slices) return;

  half4 va = read_imageh(a, smp, (int2)(m, s));
  half4 vb = read_imageh(b, smp, (int2)(m, s));
  write_imageh(output, (int2)(m, s), va + vb);
}
