#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__global uint4* dst_buffer_buffer,
  __global char4* src_buffer,
  int4 shared_int4_0,
  int4 shared_int4_1) {
  int linear_index = get_global_id(0);
  if (linear_index >= shared_int4_0.y) return;
  if (get_global_id(1) != 0) return;
  if (get_global_id(2) != 0) return;
  int dst_o_sp_i_ogroup = linear_index;
  int dst_ogroup = dst_o_sp_i_ogroup % shared_int4_0.x;
  int dst_o_sp_i = dst_o_sp_i_ogroup / shared_int4_0.x;
  int dst_i = dst_o_sp_i % shared_int4_0.z;
  int dst_o_sp = dst_o_sp_i / shared_int4_0.z;
  int dst_sp = dst_o_sp % shared_int4_1.z;
  int dst_o = dst_o_sp / shared_int4_1.z;
  int i_slice = dst_i;
  int o_slice = dst_o * shared_int4_0.x + dst_ogroup;
  int spatial_linear = dst_sp;
  int W = spatial_linear % shared_int4_1.w;
  int H = spatial_linear / shared_int4_1.w;
  float4 w0 = (float4)(0);
  float4 w1 = (float4)(0);
  float4 w2 = (float4)(0);
  float4 w3 = (float4)(0);
  if (o_slice < shared_int4_0.w && i_slice < shared_int4_0.z) {

    int linear_ohwi = ((o_slice * 4 * shared_int4_1.x + H) * shared_int4_1.w + W) * shared_int4_0.z + i_slice;
    int o_stride = shared_int4_1.x * shared_int4_1.w * shared_int4_0.z;
  if (o_slice * 4 + 0 < shared_int4_1.y) {
      w0 = convert_float4(src_buffer[(linear_ohwi + o_stride * 0)]);
  }
  if (o_slice * 4 + 1 < shared_int4_1.y) {
      w1 = convert_float4(src_buffer[(linear_ohwi + o_stride * 1)]);
  }
  if (o_slice * 4 + 2 < shared_int4_1.y) {
      w2 = convert_float4(src_buffer[(linear_ohwi + o_stride * 2)]);
  }
  if (o_slice * 4 + 3 < shared_int4_1.y) {
      w3 = convert_float4(src_buffer[(linear_ohwi + o_stride * 3)]);
  }
  }  
  w0 += (float4)(128.0f);
  w1 += (float4)(128.0f);
  w2 += (float4)(128.0f);
  w3 += (float4)(128.0f);
  float4 r0 = (float4)(w0.x, w1.x, w2.x, w3.x);
  float4 r1 = (float4)(w0.y, w1.y, w2.y, w3.y);
  float4 r2 = (float4)(w0.z, w1.z, w2.z, w3.z);
  float4 r3 = (float4)(w0.w, w1.w, w2.w, w3.w);

  uint4 packed_value;
  uint4 t = convert_uint4(convert_int4(r0));
  packed_value.x = ((t.w & 255u) << 24u) | ((t.z & 255u) << 16u) | ((t.y & 255u) << 8u) | (t.x & 255u);
  t = convert_uint4(convert_int4(r1));
  packed_value.y = ((t.w & 255u) << 24u) | ((t.z & 255u) << 16u) | ((t.y & 255u) << 8u) | (t.x & 255u);
  t = convert_uint4(convert_int4(r2));
  packed_value.z = ((t.w & 255u) << 24u) | ((t.z & 255u) << 16u) | ((t.y & 255u) << 8u) | (t.x & 255u);
  t = convert_uint4(convert_int4(r3));
  packed_value.w = ((t.w & 255u) << 24u) | ((t.z & 255u) << 16u) | ((t.y & 255u) << 8u) | (t.x & 255u);
  dst_buffer_buffer[linear_index] = packed_value;
}
