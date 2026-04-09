#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

// Tile: 4M x 4N per work item
// Reduced from 8M to halve register pressure and double occupancy
__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
gpu_int4_gemm_adreno(
                            __read_only image1d_buffer_t input,
                            __global const half *scales,
                            __global half *output,
                            __global const ushort *weights,
                            const int K,
                            const int N,
                            const int M,
                            const int quantization_group_size) {
    const int align_N = ALIGN(N, 32);

    const int m = get_global_id(0);
    const int n = get_global_id(1) * 4;
    const int M_4 = CEIL_DIV(M, 4);

    float4 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    half4 c0, c1, c2, c3;
    half4 input_reg;
    half4 dq;
    ushort4 pw;
    half4 sc;

    for(int k=0; k<K; k+=4){
        if((k&0x1F) == 0) {
            if(k > 0) {
                acc0 += convert_float4(c0);
                acc1 += convert_float4(c1);
                acc2 += convert_float4(c2);
                acc3 += convert_float4(c3);
            }
            c0 = 0; c1 = 0; c2 = 0; c3 = 0;
            sc = vload4(0, scales + (k/quantization_group_size)*align_N + n);
        }
        pw = vload4(0, weights + (k/4) * N + n);

        // k+0
        input_reg = read_imageh(input, k * M_4 + m);
        dq.s0 = ((pw.s0 & 0x000F)-8) * sc.s0;
        dq.s1 = ((pw.s1 & 0x000F)-8) * sc.s1;
        dq.s2 = ((pw.s2 & 0x000F)-8) * sc.s2;
        dq.s3 = ((pw.s3 & 0x000F)-8) * sc.s3;
        c0 += input_reg * dq.s0;
        c1 += input_reg * dq.s1;
        c2 += input_reg * dq.s2;
        c3 += input_reg * dq.s3;

        // k+1
        input_reg = read_imageh(input, (k+1) * M_4 + m);
        dq.s0 = (((pw.s0 & 0x00F0) >> 4)-8) * sc.s0;
        dq.s1 = (((pw.s1 & 0x00F0) >> 4)-8) * sc.s1;
        dq.s2 = (((pw.s2 & 0x00F0) >> 4)-8) * sc.s2;
        dq.s3 = (((pw.s3 & 0x00F0) >> 4)-8) * sc.s3;
        c0 += input_reg * dq.s0;
        c1 += input_reg * dq.s1;
        c2 += input_reg * dq.s2;
        c3 += input_reg * dq.s3;

        // k+2
        input_reg = read_imageh(input, (k+2) * M_4 + m);
        dq.s0 = (((pw.s0 & 0x0F00) >> 8)-8) * sc.s0;
        dq.s1 = (((pw.s1 & 0x0F00) >> 8)-8) * sc.s1;
        dq.s2 = (((pw.s2 & 0x0F00) >> 8)-8) * sc.s2;
        dq.s3 = (((pw.s3 & 0x0F00) >> 8)-8) * sc.s3;
        c0 += input_reg * dq.s0;
        c1 += input_reg * dq.s1;
        c2 += input_reg * dq.s2;
        c3 += input_reg * dq.s3;

        // k+3
        input_reg = read_imageh(input, (k+3) * M_4 + m);
        dq.s0 = (((pw.s0 & 0xF000) >> 12)-8) * sc.s0;
        dq.s1 = (((pw.s1 & 0xF000) >> 12)-8) * sc.s1;
        dq.s2 = (((pw.s2 & 0xF000) >> 12)-8) * sc.s2;
        dq.s3 = (((pw.s3 & 0xF000) >> 12)-8) * sc.s3;
        c0 += input_reg * dq.s0;
        c1 += input_reg * dq.s1;
        c2 += input_reg * dq.s2;
        c3 += input_reg * dq.s3;
    }

    acc0 += convert_float4(c0);
    acc1 += convert_float4(c1);
    acc2 += convert_float4(c2);
    acc3 += convert_float4(c3);

    // Store 4M x 4N
    int m4 = m << 2;
    int idx = m4 * N + n;

    if (m4 < M){
    vstore4((half4)((half)acc0.s0, (half)acc1.s0, (half)acc2.s0, (half)acc3.s0), 0, output + idx);
    }
    if (m4+1 < M){
    vstore4((half4)((half)acc0.s1, (half)acc1.s1, (half)acc2.s1, (half)acc3.s1), 0, output + idx + N);
    }
    if (m4+2 < M){
    vstore4((half4)((half)acc0.s2, (half)acc1.s2, (half)acc2.s2, (half)acc3.s2), 0, output + idx + 2*N);
    }
    if (m4+3 < M){
    vstore4((half4)((half)acc0.s3, (half)acc1.s3, (half)acc2.s3, (half)acc3.s3), 0, output + idx + 3*N);
    }
}
