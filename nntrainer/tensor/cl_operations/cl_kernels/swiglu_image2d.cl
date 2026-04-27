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

  half4 g = read_imageh(gate, smp, (int2)(m, s));
  half4 u = read_imageh(up, smp, (int2)(m, s));

  // silu(x) = x / (1 + exp(-x))
  half4 silu;
  silu.x = u.x / ((half)1.0h + exp(-u.x));
  silu.y = u.y / ((half)1.0h + exp(-u.y));
  silu.z = u.z / ((half)1.0h + exp(-u.z));
  silu.w = u.w / ((half)1.0h + exp(-u.w));

  write_imageh(output, (int2)(m, s), g * silu);
}
