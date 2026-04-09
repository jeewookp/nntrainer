#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define DQ(pw, mask, shift, sc) (half)((((pw) & (mask)) >> (shift)) - 8) * (sc)

// Tile: 8M x 8N per work item (2x N vs original)
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
    const int m = get_global_id(0) * 2;
    const int n = get_global_id(1) * 8;
    const int M_4 = CEIL_DIV(M, 4);

    float8 a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    half8 c0,c1,c2,c3,c4,c5,c6,c7;
    half8 in;
    ushort4 pw0, pw1;
    half4 s0, s1;

    for(int k=0; k<K; k+=4){
        if((k&0x1F)==0){
            if(k>0){
                a0+=convert_float8(c0); a1+=convert_float8(c1);
                a2+=convert_float8(c2); a3+=convert_float8(c3);
                a4+=convert_float8(c4); a5+=convert_float8(c5);
                a6+=convert_float8(c6); a7+=convert_float8(c7);
            }
            c0=0;c1=0;c2=0;c3=0;c4=0;c5=0;c6=0;c7=0;
            int so = (k/quantization_group_size)*align_N + n;
            s0 = vload4(0, scales + so);
            s1 = vload4(0, scales + so + 4);
        }

        int wo = (k/4)*N + n;
        pw0 = vload4(0, weights + wo);
        pw1 = vload4(0, weights + wo + 4);

        // k+0
        in.s0123 = read_imageh(input, k*M_4 + m);
        in.s4567 = read_imageh(input, k*M_4 + m + 1);

        c0 += in * DQ(pw0.s0, 0x000F, 0, s0.s0);
        c1 += in * DQ(pw0.s1, 0x000F, 0, s0.s1);
        c2 += in * DQ(pw0.s2, 0x000F, 0, s0.s2);
        c3 += in * DQ(pw0.s3, 0x000F, 0, s0.s3);
        c4 += in * DQ(pw1.s0, 0x000F, 0, s1.s0);
        c5 += in * DQ(pw1.s1, 0x000F, 0, s1.s1);
        c6 += in * DQ(pw1.s2, 0x000F, 0, s1.s2);
        c7 += in * DQ(pw1.s3, 0x000F, 0, s1.s3);

        // k+1
        in.s0123 = read_imageh(input, (k+1)*M_4 + m);
        in.s4567 = read_imageh(input, (k+1)*M_4 + m + 1);

        c0 += in * DQ(pw0.s0, 0x00F0, 4, s0.s0);
        c1 += in * DQ(pw0.s1, 0x00F0, 4, s0.s1);
        c2 += in * DQ(pw0.s2, 0x00F0, 4, s0.s2);
        c3 += in * DQ(pw0.s3, 0x00F0, 4, s0.s3);
        c4 += in * DQ(pw1.s0, 0x00F0, 4, s1.s0);
        c5 += in * DQ(pw1.s1, 0x00F0, 4, s1.s1);
        c6 += in * DQ(pw1.s2, 0x00F0, 4, s1.s2);
        c7 += in * DQ(pw1.s3, 0x00F0, 4, s1.s3);

        // k+2
        in.s0123 = read_imageh(input, (k+2)*M_4 + m);
        in.s4567 = read_imageh(input, (k+2)*M_4 + m + 1);

        c0 += in * DQ(pw0.s0, 0x0F00, 8, s0.s0);
        c1 += in * DQ(pw0.s1, 0x0F00, 8, s0.s1);
        c2 += in * DQ(pw0.s2, 0x0F00, 8, s0.s2);
        c3 += in * DQ(pw0.s3, 0x0F00, 8, s0.s3);
        c4 += in * DQ(pw1.s0, 0x0F00, 8, s1.s0);
        c5 += in * DQ(pw1.s1, 0x0F00, 8, s1.s1);
        c6 += in * DQ(pw1.s2, 0x0F00, 8, s1.s2);
        c7 += in * DQ(pw1.s3, 0x0F00, 8, s1.s3);

        // k+3
        in.s0123 = read_imageh(input, (k+3)*M_4 + m);
        in.s4567 = read_imageh(input, (k+3)*M_4 + m + 1);

        c0 += in * DQ(pw0.s0, 0xF000, 12, s0.s0);
        c1 += in * DQ(pw0.s1, 0xF000, 12, s0.s1);
        c2 += in * DQ(pw0.s2, 0xF000, 12, s0.s2);
        c3 += in * DQ(pw0.s3, 0xF000, 12, s0.s3);
        c4 += in * DQ(pw1.s0, 0xF000, 12, s1.s0);
        c5 += in * DQ(pw1.s1, 0xF000, 12, s1.s1);
        c6 += in * DQ(pw1.s2, 0xF000, 12, s1.s2);
        c7 += in * DQ(pw1.s3, 0xF000, 12, s1.s3);
    }

    // Final flush
    a0+=convert_float8(c0); a1+=convert_float8(c1);
    a2+=convert_float8(c2); a3+=convert_float8(c3);
    a4+=convert_float8(c4); a5+=convert_float8(c5);
    a6+=convert_float8(c6); a7+=convert_float8(c7);

    // Store 8M x 8N tile
    int idx = (m<<2) * N + n;

    if(idx+7<M*N){
    vstore8((half8)((half)a0.s0,(half)a1.s0,(half)a2.s0,(half)a3.s0,(half)a4.s0,(half)a5.s0,(half)a6.s0,(half)a7.s0), 0, output+idx);
    idx += N;
    }
    if(idx+7<M*N){
    vstore8((half8)((half)a0.s1,(half)a1.s1,(half)a2.s1,(half)a3.s1,(half)a4.s1,(half)a5.s1,(half)a6.s1,(half)a7.s1), 0, output+idx);
    idx += N;
    }
    if(idx+7<M*N){
    vstore8((half8)((half)a0.s2,(half)a1.s2,(half)a2.s2,(half)a3.s2,(half)a4.s2,(half)a5.s2,(half)a6.s2,(half)a7.s2), 0, output+idx);
    idx += N;
    }
    if(idx+7<M*N){
    vstore8((half8)((half)a0.s3,(half)a1.s3,(half)a2.s3,(half)a3.s3,(half)a4.s3,(half)a5.s3,(half)a6.s3,(half)a7.s3), 0, output+idx);
    idx += N;
    }
    if(idx+7<M*N){
    vstore8((half8)((half)a0.s4,(half)a1.s4,(half)a2.s4,(half)a3.s4,(half)a4.s4,(half)a5.s4,(half)a6.s4,(half)a7.s4), 0, output+idx);
    idx += N;
    }
    if(idx+7<M*N){
    vstore8((half8)((half)a0.s5,(half)a1.s5,(half)a2.s5,(half)a3.s5,(half)a4.s5,(half)a5.s5,(half)a6.s5,(half)a7.s5), 0, output+idx);
    idx += N;
    }
    if(idx+7<M*N){
    vstore8((half8)((half)a0.s6,(half)a1.s6,(half)a2.s6,(half)a3.s6,(half)a4.s6,(half)a5.s6,(half)a6.s6,(half)a7.s6), 0, output+idx);
    idx += N;
    }
    if(idx+7<M*N){
    vstore8((half8)((half)a0.s7,(half)a1.s7,(half)a2.s7,(half)a3.s7,(half)a4.s7,(half)a5.s7,(half)a6.s7,(half)a7.s7), 0, output+idx);
    }
}
