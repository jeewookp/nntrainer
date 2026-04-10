#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

// FP32 input / FP32 output / FP32 scales
// Weight image is still CL_HALF_FLOAT (used as raw 16-bit storage for packed INT4)
__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
gpu_int4_gemm_adreno(
                            __read_only image1d_buffer_t input,
                            __global const float *scales,
                            __global float *output,
                            __read_only image1d_buffer_t weight_img,
                            const int K,
                            const int N,
                            const int M,
                            const int quantization_group_size,
                            const int k_offset,
                            const int beta) {
    const int align_N = ALIGN(N, 32);

    const int m = get_global_id(0) * 2;
    const int n = get_global_id(1) * 4;
    const int M_4 = CEIL_DIV(M,4);
    const int N_4 = N >> 2;

    // FP32 accumulators for precision on large K
    float8 c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float8 input_reg;
    float4 dq_weights_reg;
    ushort4 packed_w;

    // Per-channel scale: load once (constant across all K)
    float4 scale = vload4(0, scales + n);

    for(int k=0; k<K; k+=4){
        int gk = k + k_offset;

        // Weight via texture cache (image1d_buffer holds raw 16-bit packed INT4 as half)
        packed_w = as_ushort4(read_imageh(weight_img, (gk >> 2) * N_4 + (n >> 2)));

        input_reg.s0123 = read_imagef(input, gk * M_4 + m);
        input_reg.s4567 = read_imagef(input, gk * M_4 + m + 1);

        dq_weights_reg.s0 = ((packed_w.s0 & (0x000F))-8) * scale.s0;
        dq_weights_reg.s1 = ((packed_w.s1 & (0x000F))-8) * scale.s1;
        dq_weights_reg.s2 = ((packed_w.s2 & (0x000F))-8) * scale.s2;
        dq_weights_reg.s3 = ((packed_w.s3 & (0x000F))-8) * scale.s3;

        c0 += input_reg * dq_weights_reg.s0;
        c1 += input_reg * dq_weights_reg.s1;
        c2 += input_reg * dq_weights_reg.s2;
        c3 += input_reg * dq_weights_reg.s3;

        input_reg.s0123 = read_imagef(input, (gk+1) * M_4 + m);
        input_reg.s4567 = read_imagef(input, (gk+1) * M_4 + m + 1);

        dq_weights_reg.s0 = (((packed_w.s0 & (0x00F0)) >> 4) - 8) * scale.s0;
        dq_weights_reg.s1 = (((packed_w.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
        dq_weights_reg.s2 = (((packed_w.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
        dq_weights_reg.s3 = (((packed_w.s3 & (0x00F0)) >> 4) - 8) * scale.s3;

        c0 += input_reg * dq_weights_reg.s0;
        c1 += input_reg * dq_weights_reg.s1;
        c2 += input_reg * dq_weights_reg.s2;
        c3 += input_reg * dq_weights_reg.s3;

        input_reg.s0123 = read_imagef(input, (gk+2) * M_4 + m);
        input_reg.s4567 = read_imagef(input, (gk+2) * M_4 + m + 1);

        dq_weights_reg.s0 = (((packed_w.s0 & (0x0F00)) >> 8) - 8) * scale.s0;
        dq_weights_reg.s1 = (((packed_w.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
        dq_weights_reg.s2 = (((packed_w.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
        dq_weights_reg.s3 = (((packed_w.s3 & (0x0F00)) >> 8) - 8) * scale.s3;

        c0 += input_reg * dq_weights_reg.s0;
        c1 += input_reg * dq_weights_reg.s1;
        c2 += input_reg * dq_weights_reg.s2;
        c3 += input_reg * dq_weights_reg.s3;

        input_reg.s0123 = read_imagef(input, (gk+3) * M_4 + m);
        input_reg.s4567 = read_imagef(input, (gk+3) * M_4 + m + 1);

        dq_weights_reg.s0 = (((packed_w.s0 & (0xF000)) >> 12) - 8) * scale.s0;
        dq_weights_reg.s1 = (((packed_w.s1 & (0xF000)) >> 12) - 8) * scale.s1;
        dq_weights_reg.s2 = (((packed_w.s2 & (0xF000)) >> 12) - 8) * scale.s2;
        dq_weights_reg.s3 = (((packed_w.s3 & (0xF000)) >> 12) - 8) * scale.s3;

        c0 += input_reg * dq_weights_reg.s0;
        c1 += input_reg * dq_weights_reg.s1;
        c2 += input_reg * dq_weights_reg.s2;
        c3 += input_reg * dq_weights_reg.s3;
    }

    int idx = (m<<2) * N + n;

    if (beta == 1) {
        float4 prev;
        if (idx+3<M*N){ prev = vload4(0, output+idx);
        vstore4(prev + (float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, output + idx); idx += N; }
        if (idx+3<M*N){ prev = vload4(0, output+idx);
        vstore4(prev + (float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, output + idx); idx += N; }
        if (idx+3<M*N){ prev = vload4(0, output+idx);
        vstore4(prev + (float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, output + idx); idx += N; }
        if (idx+3<M*N){ prev = vload4(0, output+idx);
        vstore4(prev + (float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, output + idx); idx += N; }
        if (idx+3<M*N){ prev = vload4(0, output+idx);
        vstore4(prev + (float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, output + idx); idx += N; }
        if (idx+3<M*N){ prev = vload4(0, output+idx);
        vstore4(prev + (float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, output + idx); idx += N; }
        if (idx+3<M*N){ prev = vload4(0, output+idx);
        vstore4(prev + (float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, output + idx); idx += N; }
        if (idx+3<M*N){ prev = vload4(0, output+idx);
        vstore4(prev + (float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, output + idx); }
    } else {
        if (idx+3<M*N){ vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, output + idx); idx += N; }
        if (idx+3<M*N){ vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, output + idx); idx += N; }
        if (idx+3<M*N){ vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, output + idx); idx += N; }
        if (idx+3<M*N){ vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, output + idx); idx += N; }
        if (idx+3<M*N){ vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, output + idx); idx += N; }
        if (idx+3<M*N){ vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, output + idx); idx += N; }
        if (idx+3<M*N){ vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, output + idx); idx += N; }
        if (idx+3<M*N){ vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, output + idx); }
    }
}
