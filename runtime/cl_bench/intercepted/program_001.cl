#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__global half4* dst_tensor_buffer,
  __global uint4* src_buffer_buffer,
  __read_only image2d_t weights_scale_image2d,
  __read_only image2d_t weights_zero_point_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1) {
  if (get_global_id(2) != 0) return;
  int dst_i_ogroup = get_global_id(0);
  int dst_o = get_global_id(1);
  int linear_index = dst_o * shared_int4_0.y * shared_int4_0.x + dst_i_ogroup;
  int dst_ogroup = dst_i_ogroup % shared_int4_0.x;
  int dst_i = dst_i_ogroup / shared_int4_0.x;
  if (dst_i >= shared_int4_0.y) return;
  if (dst_o * shared_int4_0.x >= shared_int4_0.z) return;
  int i_slice = dst_i;
  int o_slice = dst_o * shared_int4_0.x + dst_ogroup;
  int spatial_linear = 0;
  int W = spatial_linear % shared_int4_1.y;
  int H = spatial_linear / shared_int4_1.y;
  half4 w0, w1, w2, w3;
  w0 = (half4)(0);
  w1 = (half4)(0);
  w2 = (half4)(0);
  w3 = (half4)(0);
  if (o_slice < shared_int4_0.z && i_slice < shared_int4_0.y) {
  int src_i = i_slice;
  int src_o_group = o_slice % shared_int4_0.w;
  int src_o = o_slice / shared_int4_0.w;
  int src_linear = ((src_o * shared_int4_1.x + spatial_linear) * shared_int4_0.y + src_i) * shared_int4_0.w + src_o_group;
    uint4 w = src_buffer_buffer[src_linear];
    
  w0.x = convert_half((w.x) & 255u);
  w0.y = convert_half((w.x >>  8u) & 255u);
  w0.z = convert_half((w.x >> 16u) & 255u);
  w0.w = convert_half((w.x >> 24u) & 255u);
  w1.x = convert_half((w.y) & 255u);
  w1.y = convert_half((w.y >>  8u) & 255u);
  w1.z = convert_half((w.y >> 16u) & 255u);
  w1.w = convert_half((w.y >> 24u) & 255u);
  w2.x = convert_half((w.z) & 255u);
  w2.y = convert_half((w.z >>  8u) & 255u);
  w2.z = convert_half((w.z >> 16u) & 255u);
  w2.w = convert_half((w.z >> 24u) & 255u);
  w3.x = convert_half((w.w) & 255u);
  w3.y = convert_half((w.w >>  8u) & 255u);
  w3.z = convert_half((w.w >> 16u) & 255u);
  w3.w = convert_half((w.w >> 24u) & 255u);
;
  }  
  if (o_slice < shared_int4_0.z && i_slice < shared_int4_0.y) {
    half4 weight_scale = convert_half4(read_imageh(weights_scale_image2d, smp_zero, (int2)((o_slice), 0)));
    half4 weight_zp = convert_half4(read_imageh(weights_zero_point_image2d, smp_zero, (int2)((o_slice), 0)));
  half4 weight_bias = -weight_scale * ((half4)(128.0f) + weight_zp);
  w0 = w0 * weight_scale + weight_bias;
  w1 = w1 * weight_scale + weight_bias;
  w2 = w2 * weight_scale + weight_bias;
  w3 = w3 * weight_scale + weight_bias;
  }  
  half4 r0 = w0;
  half4 r1 = w1;
  half4 r2 = w2;
  half4 r3 = w3;
  dst_tensor_buffer[linear_index * 4 + 0] = r0;
  dst_tensor_buffer[linear_index * 4 + 1] = r1;
  dst_tensor_buffer[linear_index * 4 + 2] = r2;
  dst_tensor_buffer[linear_index * 4 + 3] = r3;
}
