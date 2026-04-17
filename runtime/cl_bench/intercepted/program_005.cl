#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__global float* buffer_buffer,
  __read_only image2d_t tensor_image2d,
  int4 shared_int4_0) {
  int linear_id = get_global_id(0);
  int x = linear_id / 1;
  int b = linear_id % 1;
  int y = get_global_id(1);
  int d = get_global_id(2);
  if (x >= shared_int4_0.w || y >= shared_int4_0.y || d >= shared_int4_0.z) return;
  half4 in_value = read_imageh(tensor_image2d, smp_zero, (int2)((x), ((y) * shared_int4_0.z + (d))));
  float out_x = convert_float(in_value.x);
  float out_y = convert_float(in_value.y);
  float out_z = convert_float(in_value.z);
  float out_w = convert_float(in_value.w);
  int c = d * 4;
  int index = ((b * shared_int4_0.y + y) * shared_int4_0.w + x) * shared_int4_0.x + c;

  buffer_buffer[index] = out_x;
  if (c + 1 < shared_int4_0.x) {
    buffer_buffer[index + 1] = out_y;
  }
  if (c + 2 < shared_int4_0.x) {
    buffer_buffer[index + 2] = out_z;
  }
  if (c + 3 < shared_int4_0.x) {
    buffer_buffer[index + 3] = out_w;
  }
}