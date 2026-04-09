#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Convert FP32 input to FP16 with zero-padding on K dimension
// Dispatch: {M, alignK/4, 1}
kernel void fp32_to_fp16_pad(
    __global const float *src,
    __global half *dst,
    const int M,
    const int K,
    const int alignK) {
    int m = get_global_id(0);
    int k4 = get_global_id(1);
    int k = k4 << 2;
    if (m >= M || k >= alignK) return;

    float4 f;
    f.s0 = (k + 0 < K) ? src[m * K + k + 0] : 0.0f;
    f.s1 = (k + 1 < K) ? src[m * K + k + 1] : 0.0f;
    f.s2 = (k + 2 < K) ? src[m * K + k + 2] : 0.0f;
    f.s3 = (k + 3 < K) ? src[m * K + k + 3] : 0.0f;

    vstore4(convert_half4(f), 0, dst + m * alignK + k);
}
