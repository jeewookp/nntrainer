// Load-path microbenchmark for the delegate wave-memory architecture.
//
// Same 12-subgroup / xmem-constant / mova-a0 / qcom_sub_group_constant_load8
// scaffold as delegate_conv_wave.cl, but the compute body is replaced by a
// tiny OR-reduction of the loaded bytes so the compiler cannot DCE the
// loads away. One half4 per (X, Y, Z) thread is written to dst at the end,
// matching the delegate kernel's output image2d signature so dispatch and
// timing harness stay identical.
//
// Two variants, selected by kernel name, differ only in the load `count`
// argument to qcom_sub_group_constant_load8:
//
//   probe_load_count32:
//     count=32 half8 per iter = 512 B, f_offset += 64 halves per iter.
//     Matches fp16 delegate exactly. For int4 this is 4x padded: only
//     64 of 256 loaded ushorts carry real weight data.
//
//   probe_load_count8:
//     count=8 half8 per iter = 128 B, f_offset += 16 halves per iter.
//     Just enough bytes to cover 2 src_slices of int4 (64 ushorts) with
//     no padding. Tests whether the wave-memory path still hits peak at
//     a smaller count, which is what would unlock an int4-dense layout
//     beating fp16 delegate on raw load bandwidth.
//
// Both variants run the same outer loop (K/8 iters, mirroring delegate's
// src_slice pairing), and both touch the same number of iters per thread —
// so dispatch time is dominated by load bandwidth, not iter count.

#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
#pragma OPENCL EXTENSION cl_qcom_inline_asm : enable

// shared_int4_0 = {1, dst_slices, M, 32}
// shared_int4_1 = {src_slices, 0, 0, src_slices}  // src_slices = K/4
// shared_int4_2 = {1, 1, 0, 0}
// same NDRange as delegate_conv_wave.

__attribute__((qcom_max_concurrent_subgroups(12)))
__kernel void probe_load_count32(
  __constant half8* weights_buffer  __attribute__((sub_group_uniform)),
  __constant half8* xmem_buffer  __attribute__((max_constant_size((6144)))),
  __write_only image2d_t dst_tensor_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1,
  int4 shared_int4_2) {

  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x) return;
  if (Z * 8 >= shared_int4_0.y) return;

  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  int c_offset = mul24(subgroup_id, shared_int4_0.w);
  short a0 = c_offset * 8;

__asm__ __volatile__(
  "mova a0, 256;"
  "mova a0, %[a0];"
  "(rpt5)nop ;"
  :
  : [a0]  "r" (a0)
  :
);
  __constant ushort* w_cache = (__constant ushort*)&xmem_buffer[c_offset];

  // f_offset advances at delegate's native stride (64 halves per iter).
  int f_offset = Z * shared_int4_1.x * 32;

  // OR-reduction accumulator: cheap enough that per-iter compute is tiny
  // vs. the 512-byte wave load, but not zero so loads are not DCE-able.
  ushort acc = 0;

  int coord_s = 0;
  do {
    qcom_sub_group_constant_load8(xmem_buffer, weights_buffer,
                                  c_offset, f_offset >> 1, 32);
    f_offset += 64;
    qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);
    acc |= w_cache[0];
    coord_s += 2;
  } while (coord_s < shared_int4_1.w);

  int cs = mul24(Z, 8);
  half4 res = (half4)((half)(as_short(acc)));
  if (cs < shared_int4_0.y)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * shared_int4_0.y + cs), res);
}


__attribute__((qcom_max_concurrent_subgroups(12)))
__kernel void probe_load_count8(
  __constant half8* weights_buffer  __attribute__((sub_group_uniform)),
  __constant half8* xmem_buffer  __attribute__((max_constant_size((6144)))),
  __write_only image2d_t dst_tensor_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1,
  int4 shared_int4_2) {

  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x) return;
  if (Z * 8 >= shared_int4_0.y) return;

  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  int c_offset = mul24(subgroup_id, shared_int4_0.w);
  short a0 = c_offset * 8;

__asm__ __volatile__(
  "mova a0, 256;"
  "mova a0, %[a0];"
  "(rpt5)nop ;"
  :
  : [a0]  "r" (a0)
  :
);
  __constant ushort* w_cache = (__constant ushort*)&xmem_buffer[c_offset];

  // Dense int4 stride: 8 half8 = 64 ushorts = 2 src_slices of int4 exactly.
  int f_offset = Z * shared_int4_1.x * 8;

  ushort acc = 0;

  int coord_s = 0;
  do {
    qcom_sub_group_constant_load8(xmem_buffer, weights_buffer,
                                  c_offset, f_offset >> 1, 8);
    f_offset += 16;
    qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);
    acc |= w_cache[0];
    coord_s += 2;
  } while (coord_s < shared_int4_1.w);

  int cs = mul24(Z, 8);
  half4 res = (half4)((half)(as_short(acc)));
  if (cs < shared_int4_0.y)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * shared_int4_0.y + cs), res);
}
