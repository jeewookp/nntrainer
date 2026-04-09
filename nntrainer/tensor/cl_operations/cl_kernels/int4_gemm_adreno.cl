#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

// Tile: 8M x 4N per work item
// Reads input directly in row-major [M, K] layout — no transpose needed.
// Each read_imageh returns 4 consecutive K values for one M row.
// 8 reads (one per M row) give all data for 4 K steps.
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

    const int m = get_global_id(0) * 2;  // 2 pixels = 8 M rows
    const int n = get_global_id(1) * 4;
    const int M_4 = CEIL_DIV(M, 4);
    const int K_4 = CEIL_DIV(K, 4);

    float8 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    half8 c0, c1, c2, c3;
    half8 input_reg;
    half4 dq;
    ushort4 pw;
    half4 sc;

    // Row-major: pixel at row*K_4 + k/4 contains {input[row][k..k+3]}
    // m*4 .. m*4+7 are the 8 M rows
    const int row_base0 = (m << 2) * K_4;       // row m*4
    const int row_base1 = ((m << 2) + 1) * K_4; // row m*4+1
    const int row_base2 = ((m << 2) + 2) * K_4;
    const int row_base3 = ((m << 2) + 3) * K_4;
    const int row_base4 = ((m << 2) + 4) * K_4;
    const int row_base5 = ((m << 2) + 5) * K_4;
    const int row_base6 = ((m << 2) + 6) * K_4;
    const int row_base7 = ((m << 2) + 7) * K_4;

    for(int k=0; k<K; k+=4){
        if((k&0x1F) == 0) {
            if(k > 0) {
                acc0 += convert_float8(c0);
                acc1 += convert_float8(c1);
                acc2 += convert_float8(c2);
                acc3 += convert_float8(c3);
            }
            c0 = 0; c1 = 0; c2 = 0; c3 = 0;
            sc = vload4(0, scales + (k/quantization_group_size)*align_N + n);
        }

        pw = vload4(0, weights + (k/4) * N + n);
        int kk = k >> 2;

        // Read 4 consecutive K values for each of 8 M rows
        half4 r0 = read_imageh(input, row_base0 + kk);
        half4 r1 = read_imageh(input, row_base1 + kk);
        half4 r2 = read_imageh(input, row_base2 + kk);
        half4 r3 = read_imageh(input, row_base3 + kk);
        half4 r4 = read_imageh(input, row_base4 + kk);
        half4 r5 = read_imageh(input, row_base5 + kk);
        half4 r6 = read_imageh(input, row_base6 + kk);
        half4 r7 = read_imageh(input, row_base7 + kk);

        // k+0: gather .s0 from each row
        input_reg = (half8)(r0.s0, r1.s0, r2.s0, r3.s0, r4.s0, r5.s0, r6.s0, r7.s0);
        dq.s0 = ((pw.s0 & 0x000F)-8) * sc.s0;
        dq.s1 = ((pw.s1 & 0x000F)-8) * sc.s1;
        dq.s2 = ((pw.s2 & 0x000F)-8) * sc.s2;
        dq.s3 = ((pw.s3 & 0x000F)-8) * sc.s3;
        c0 += input_reg * dq.s0;
        c1 += input_reg * dq.s1;
        c2 += input_reg * dq.s2;
        c3 += input_reg * dq.s3;

        // k+1: gather .s1
        input_reg = (half8)(r0.s1, r1.s1, r2.s1, r3.s1, r4.s1, r5.s1, r6.s1, r7.s1);
        dq.s0 = (((pw.s0 & 0x00F0) >> 4)-8) * sc.s0;
        dq.s1 = (((pw.s1 & 0x00F0) >> 4)-8) * sc.s1;
        dq.s2 = (((pw.s2 & 0x00F0) >> 4)-8) * sc.s2;
        dq.s3 = (((pw.s3 & 0x00F0) >> 4)-8) * sc.s3;
        c0 += input_reg * dq.s0;
        c1 += input_reg * dq.s1;
        c2 += input_reg * dq.s2;
        c3 += input_reg * dq.s3;

        // k+2: gather .s2
        input_reg = (half8)(r0.s2, r1.s2, r2.s2, r3.s2, r4.s2, r5.s2, r6.s2, r7.s2);
        dq.s0 = (((pw.s0 & 0x0F00) >> 8)-8) * sc.s0;
        dq.s1 = (((pw.s1 & 0x0F00) >> 8)-8) * sc.s1;
        dq.s2 = (((pw.s2 & 0x0F00) >> 8)-8) * sc.s2;
        dq.s3 = (((pw.s3 & 0x0F00) >> 8)-8) * sc.s3;
        c0 += input_reg * dq.s0;
        c1 += input_reg * dq.s1;
        c2 += input_reg * dq.s2;
        c3 += input_reg * dq.s3;

        // k+3: gather .s3
        input_reg = (half8)(r0.s3, r1.s3, r2.s3, r3.s3, r4.s3, r5.s3, r6.s3, r7.s3);
        dq.s0 = (((pw.s0 & 0xF000) >> 12)-8) * sc.s0;
        dq.s1 = (((pw.s1 & 0xF000) >> 12)-8) * sc.s1;
        dq.s2 = (((pw.s2 & 0xF000) >> 12)-8) * sc.s2;
        dq.s3 = (((pw.s3 & 0xF000) >> 12)-8) * sc.s3;
        c0 += input_reg * dq.s0;
        c1 += input_reg * dq.s1;
        c2 += input_reg * dq.s2;
        c3 += input_reg * dq.s3;
    }

    acc0 += convert_float8(c0);
    acc1 += convert_float8(c1);
    acc2 += convert_float8(c2);
    acc3 += convert_float8(c3);

    int idx = (m<<2) * N + n;

    if (idx+3<M*N){
    vstore4((half4)((half)acc0.s0, (half)acc1.s0, (half)acc2.s0, (half)acc3.s0), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)acc0.s1, (half)acc1.s1, (half)acc2.s1, (half)acc3.s1), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)acc0.s2, (half)acc1.s2, (half)acc2.s2, (half)acc3.s2), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)acc0.s3, (half)acc1.s3, (half)acc2.s3, (half)acc3.s3), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)acc0.s4, (half)acc1.s4, (half)acc2.s4, (half)acc3.s4), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)acc0.s5, (half)acc1.s5, (half)acc2.s5, (half)acc3.s5), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)acc0.s6, (half)acc1.s6, (half)acc2.s6, (half)acc3.s6), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)acc0.s7, (half)acc1.s7, (half)acc2.s7, (half)acc3.s7), 0, output + idx);
    }
}
