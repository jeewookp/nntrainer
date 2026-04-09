#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Convert FP16 buffer to FP32 buffer (element-wise)
// Dispatch: {total/4, 1, 1}
kernel void fp16_to_fp32(
    __global const half *src,
    __global float *dst,
    const int total) {
    int i4 = get_global_id(0);
    int i = i4 << 2;
    if (i >= total) return;

    half4 h = vload4(0, src + i);
    vstore4(convert_float4(h), 0, dst + i);
}
