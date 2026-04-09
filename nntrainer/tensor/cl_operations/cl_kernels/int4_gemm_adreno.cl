#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define LDS_STRIDE 32

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
    const int n = get_global_id(1) * 4;
    const int M_4 = CEIL_DIV(M,4);
    const int lid = get_local_id(1);

    // Float accumulators in local memory (not registers)
    // 32 floats per work item: 4 accumulators × 8 values
    __local float lacc[128 * LDS_STRIDE];
    const int lb = lid * LDS_STRIDE;
    for (int i = 0; i < LDS_STRIDE; i++) lacc[lb + i] = 0.0f;

    // Pure half registers only - same as original fast kernel
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 input_reg;
    half4 dq_weights_reg;
    ushort4 packed_w;
    half4 scale;

    for(int k=0; k<K; k+=4){
        // Flush half→float(local mem) every 256 K values
        if((k&0xFF) == 0 && k > 0) {
            lacc[lb+ 0]+=(float)c0.s0; lacc[lb+ 1]+=(float)c0.s1;
            lacc[lb+ 2]+=(float)c0.s2; lacc[lb+ 3]+=(float)c0.s3;
            lacc[lb+ 4]+=(float)c0.s4; lacc[lb+ 5]+=(float)c0.s5;
            lacc[lb+ 6]+=(float)c0.s6; lacc[lb+ 7]+=(float)c0.s7;
            lacc[lb+ 8]+=(float)c1.s0; lacc[lb+ 9]+=(float)c1.s1;
            lacc[lb+10]+=(float)c1.s2; lacc[lb+11]+=(float)c1.s3;
            lacc[lb+12]+=(float)c1.s4; lacc[lb+13]+=(float)c1.s5;
            lacc[lb+14]+=(float)c1.s6; lacc[lb+15]+=(float)c1.s7;
            lacc[lb+16]+=(float)c2.s0; lacc[lb+17]+=(float)c2.s1;
            lacc[lb+18]+=(float)c2.s2; lacc[lb+19]+=(float)c2.s3;
            lacc[lb+20]+=(float)c2.s4; lacc[lb+21]+=(float)c2.s5;
            lacc[lb+22]+=(float)c2.s6; lacc[lb+23]+=(float)c2.s7;
            lacc[lb+24]+=(float)c3.s0; lacc[lb+25]+=(float)c3.s1;
            lacc[lb+26]+=(float)c3.s2; lacc[lb+27]+=(float)c3.s3;
            lacc[lb+28]+=(float)c3.s4; lacc[lb+29]+=(float)c3.s5;
            lacc[lb+30]+=(float)c3.s6; lacc[lb+31]+=(float)c3.s7;
            c0 = 0; c1 = 0; c2 = 0; c3 = 0;
        }
        if((k&0x1F) == 0) {
            scale = vload4(0,scales + (k/quantization_group_size)*align_N + n);
        }
        packed_w = vload4(0,weights + (k/4) * N + n);

        input_reg.s0123 = read_imageh(input, k * M_4 + m);
        input_reg.s4567 = read_imageh(input, k * M_4 + m + 1);

        dq_weights_reg.s0 = ((packed_w.s0 & (0x000F))-8) * scale.s0;
        dq_weights_reg.s1 = ((packed_w.s1 & (0x000F))-8) * scale.s1;
        dq_weights_reg.s2 = ((packed_w.s2 & (0x000F))-8) * scale.s2;
        dq_weights_reg.s3 = ((packed_w.s3 & (0x000F))-8) * scale.s3;

        c0 += input_reg * dq_weights_reg.s0;
        c1 += input_reg * dq_weights_reg.s1;
        c2 += input_reg * dq_weights_reg.s2;
        c3 += input_reg * dq_weights_reg.s3;

        input_reg.s0123 = read_imageh(input, (k+1) * M_4 + m);
        input_reg.s4567 = read_imageh(input, (k+1) * M_4 + m + 1);

        dq_weights_reg.s0 = (((packed_w.s0 & (0x00F0)) >> 4) - 8) * scale.s0;
        dq_weights_reg.s1 = (((packed_w.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
        dq_weights_reg.s2 = (((packed_w.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
        dq_weights_reg.s3 = (((packed_w.s3 & (0x00F0)) >> 4) - 8) * scale.s3;

        c0 += input_reg * dq_weights_reg.s0;
        c1 += input_reg * dq_weights_reg.s1;
        c2 += input_reg * dq_weights_reg.s2;
        c3 += input_reg * dq_weights_reg.s3;

        input_reg.s0123 = read_imageh(input, (k+2) * M_4 + m);
        input_reg.s4567 = read_imageh(input, (k+2) * M_4 + m + 1);

        dq_weights_reg.s0 = (((packed_w.s0 & (0x0F00)) >> 8) - 8) * scale.s0;
        dq_weights_reg.s1 = (((packed_w.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
        dq_weights_reg.s2 = (((packed_w.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
        dq_weights_reg.s3 = (((packed_w.s3 & (0x0F00)) >> 8) - 8) * scale.s3;

        c0 += input_reg * dq_weights_reg.s0;
        c1 += input_reg * dq_weights_reg.s1;
        c2 += input_reg * dq_weights_reg.s2;
        c3 += input_reg * dq_weights_reg.s3;

        input_reg.s0123 = read_imageh(input, (k+3) * M_4 + m);
        input_reg.s4567 = read_imageh(input, (k+3) * M_4 + m + 1);

        dq_weights_reg.s0 = (((packed_w.s0 & (0xF000)) >> 12) - 8) * scale.s0;
        dq_weights_reg.s1 = (((packed_w.s1 & (0xF000)) >> 12) - 8) * scale.s1;
        dq_weights_reg.s2 = (((packed_w.s2 & (0xF000)) >> 12) - 8) * scale.s2;
        dq_weights_reg.s3 = (((packed_w.s3 & (0xF000)) >> 12) - 8) * scale.s3;

        c0 += input_reg * dq_weights_reg.s0;
        c1 += input_reg * dq_weights_reg.s1;
        c2 += input_reg * dq_weights_reg.s2;
        c3 += input_reg * dq_weights_reg.s3;
    }

    // Final flush to local memory
    lacc[lb+ 0]+=(float)c0.s0; lacc[lb+ 1]+=(float)c0.s1;
    lacc[lb+ 2]+=(float)c0.s2; lacc[lb+ 3]+=(float)c0.s3;
    lacc[lb+ 4]+=(float)c0.s4; lacc[lb+ 5]+=(float)c0.s5;
    lacc[lb+ 6]+=(float)c0.s6; lacc[lb+ 7]+=(float)c0.s7;
    lacc[lb+ 8]+=(float)c1.s0; lacc[lb+ 9]+=(float)c1.s1;
    lacc[lb+10]+=(float)c1.s2; lacc[lb+11]+=(float)c1.s3;
    lacc[lb+12]+=(float)c1.s4; lacc[lb+13]+=(float)c1.s5;
    lacc[lb+14]+=(float)c1.s6; lacc[lb+15]+=(float)c1.s7;
    lacc[lb+16]+=(float)c2.s0; lacc[lb+17]+=(float)c2.s1;
    lacc[lb+18]+=(float)c2.s2; lacc[lb+19]+=(float)c2.s3;
    lacc[lb+20]+=(float)c2.s4; lacc[lb+21]+=(float)c2.s5;
    lacc[lb+22]+=(float)c2.s6; lacc[lb+23]+=(float)c2.s7;
    lacc[lb+24]+=(float)c3.s0; lacc[lb+25]+=(float)c3.s1;
    lacc[lb+26]+=(float)c3.s2; lacc[lb+27]+=(float)c3.s3;
    lacc[lb+28]+=(float)c3.s4; lacc[lb+29]+=(float)c3.s5;
    lacc[lb+30]+=(float)c3.s6; lacc[lb+31]+=(float)c3.s7;

    // Store from local memory to global output
    int idx = (m<<2) * N + n;

    if (idx+3<M*N){
    vstore4((half4)((half)lacc[lb+0], (half)lacc[lb+8], (half)lacc[lb+16], (half)lacc[lb+24]), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)lacc[lb+1], (half)lacc[lb+9], (half)lacc[lb+17], (half)lacc[lb+25]), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)lacc[lb+2], (half)lacc[lb+10], (half)lacc[lb+18], (half)lacc[lb+26]), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)lacc[lb+3], (half)lacc[lb+11], (half)lacc[lb+19], (half)lacc[lb+27]), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)lacc[lb+4], (half)lacc[lb+12], (half)lacc[lb+20], (half)lacc[lb+28]), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)lacc[lb+5], (half)lacc[lb+13], (half)lacc[lb+21], (half)lacc[lb+29]), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)lacc[lb+6], (half)lacc[lb+14], (half)lacc[lb+22], (half)lacc[lb+30]), 0, output + idx);
    idx += N;
    }
    if (idx+3<M*N){
    vstore4((half4)((half)lacc[lb+7], (half)lacc[lb+15], (half)lacc[lb+23], (half)lacc[lb+31]), 0, output + idx);
    }
}
