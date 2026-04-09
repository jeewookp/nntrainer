#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

// INT8 x INT4 GEMM with INT32 accumulation
// Input: INT8 quantized per-row (symmetric) with lhs_scale[M] (FP32)
// Weight: INT4 packed (ushort, nibbles biased by +8) with rhs_scale[N] (FP32)
// Output: FP32
// Each work item computes 8 M rows × 4 N cols
__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
gpu_int8_int4_gemm_adreno(
                            __global const char *input,      // [M][K] INT8
                            __global const float *lhs_scale, // [M] per-row scale
                            __global const float *rhs_scale, // [N] per-channel scale
                            __global float *output,          // [M][N] FP32
                            __global const ushort *weights,  // [K/4][N] packed INT4
                            const int K,
                            const int N,
                            const int M,
                            const int k_offset,
                            const int beta) {
    const int m = get_global_id(0) * 8;  // 8 M rows per work item
    const int n = get_global_id(1) * 4;  // 4 N cols per work item

    if (m >= M || n >= N) return;

    int4 acc[8];
    for (int i = 0; i < 8; i++) acc[i] = (int4)(0, 0, 0, 0);

    for (int k = 0; k < K; k += 4) {
        int gk = k + k_offset;

        // Load 4 INT4 weights for 4 N cols (packed in ushort4)
        ushort4 pw = vload4(0, weights + (gk >> 2) * N + n);

        // Unpack to char4 per N column (with -8 bias correction)
        char4 w0 = (char4)(
            (char)((pw.s0 & 0x000F)) - 8,
            (char)((pw.s0 & 0x00F0) >> 4) - 8,
            (char)((pw.s0 & 0x0F00) >> 8) - 8,
            (char)((pw.s0 & 0xF000) >> 12) - 8);
        char4 w1 = (char4)(
            (char)((pw.s1 & 0x000F)) - 8,
            (char)((pw.s1 & 0x00F0) >> 4) - 8,
            (char)((pw.s1 & 0x0F00) >> 8) - 8,
            (char)((pw.s1 & 0xF000) >> 12) - 8);
        char4 w2 = (char4)(
            (char)((pw.s2 & 0x000F)) - 8,
            (char)((pw.s2 & 0x00F0) >> 4) - 8,
            (char)((pw.s2 & 0x0F00) >> 8) - 8,
            (char)((pw.s2 & 0xF000) >> 12) - 8);
        char4 w3 = (char4)(
            (char)((pw.s3 & 0x000F)) - 8,
            (char)((pw.s3 & 0x00F0) >> 4) - 8,
            (char)((pw.s3 & 0x0F00) >> 8) - 8,
            (char)((pw.s3 & 0xF000) >> 12) - 8);

        // Convert weights to int4 for wider arithmetic (compiler may use dp4a)
        int4 wi0 = convert_int4(w0);
        int4 wi1 = convert_int4(w1);
        int4 wi2 = convert_int4(w2);
        int4 wi3 = convert_int4(w3);

        // For each of 8 M rows: load 4 INT8 input values and dot-product with 4 N cols
        for (int mi = 0; mi < 8; mi++) {
            if (m + mi >= M) break;
            char4 in = vload4(0, input + (m + mi) * K + gk);
            int4 ini = convert_int4(in);
            int4 p0 = ini * wi0;
            int4 p1 = ini * wi1;
            int4 p2 = ini * wi2;
            int4 p3 = ini * wi3;
            acc[mi].s0 += p0.s0 + p0.s1 + p0.s2 + p0.s3;
            acc[mi].s1 += p1.s0 + p1.s1 + p1.s2 + p1.s3;
            acc[mi].s2 += p2.s0 + p2.s1 + p2.s2 + p2.s3;
            acc[mi].s3 += p3.s0 + p3.s1 + p3.s2 + p3.s3;
        }
    }

    // Load rhs_scale once (4 N cols)
    float4 rs = vload4(0, rhs_scale + n);

    // Dequantize and store
    for (int mi = 0; mi < 8; mi++) {
        if (m + mi >= M) break;
        float ls = lhs_scale[m + mi];
        float4 out = convert_float4(acc[mi]) * (ls * rs);
        int idx = (m + mi) * N + n;
        if (beta == 1) {
            float4 prev = vload4(0, output + idx);
            out += prev;
        }
        vstore4(out, 0, output + idx);
    }
}
