#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define NR 4
#define ALIGN_32(x) (((x) + 31) & ~31)

// Dispatch: {K/32, N/4, 1}
// Optimized: vectorized reads (vload8) + no division/modulo in inner loop
__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
repack_kai_to_adreno(const __global uchar* kai_packed_data,
    __global ushort* weights,
    __global float* scales,
    const int N, const int K, const int rhs_packed_stride, const int quantization_group_size) {

    const int k_id = get_global_id(0);
    const int n_id = get_global_id(1);

    const int K_aligned = ALIGN_32(K);
    const int base = n_id * rhs_packed_stride;
    // block_base = k_id * (K_INTERLEAVED_V * NR) = k_id * 64
    const int block_base = k_id << 6;
    const int k_start = k_id << 5;

    for (int sub_n = 0; sub_n < NR; sub_n++) {
        int global_n = n_id * NR + sub_n;
        int byte_off = base + block_base + (sub_n << 3);

        // Read 2 groups of 8 consecutive bytes from KAI packed data
        // b0[0..7]: contains K values 0..7 (lower nib) and 16..23 (upper nib)
        // b1[0..7]: contains K values 8..15 (lower nib) and 24..31 (upper nib)
        uchar8 b0 = vload8(0, kai_packed_data + byte_off);
        uchar8 b1 = vload8(0, kai_packed_data + byte_off + 32);

        // Pack 4 nibbles per uint16_t, XOR 0x8 for unsigned affine encoding
        // sub_k 0..3: b0.s0123 lower nibble
        ushort p0 = ((ushort)((b0.s0&0xF)^0x8))      | ((ushort)((b0.s1&0xF)^0x8)<<4) |
                    ((ushort)((b0.s2&0xF)^0x8)<<8)    | ((ushort)((b0.s3&0xF)^0x8)<<12);
        // sub_k 4..7: b0.s4567 lower nibble
        ushort p1 = ((ushort)((b0.s4&0xF)^0x8))      | ((ushort)((b0.s5&0xF)^0x8)<<4) |
                    ((ushort)((b0.s6&0xF)^0x8)<<8)    | ((ushort)((b0.s7&0xF)^0x8)<<12);
        // sub_k 8..11: b1.s0123 lower nibble
        ushort p2 = ((ushort)((b1.s0&0xF)^0x8))      | ((ushort)((b1.s1&0xF)^0x8)<<4) |
                    ((ushort)((b1.s2&0xF)^0x8)<<8)    | ((ushort)((b1.s3&0xF)^0x8)<<12);
        // sub_k 12..15: b1.s4567 lower nibble
        ushort p3 = ((ushort)((b1.s4&0xF)^0x8))      | ((ushort)((b1.s5&0xF)^0x8)<<4) |
                    ((ushort)((b1.s6&0xF)^0x8)<<8)    | ((ushort)((b1.s7&0xF)^0x8)<<12);
        // sub_k 16..19: b0.s0123 upper nibble
        ushort p4 = ((ushort)((b0.s0>>4)^0x8))       | ((ushort)((b0.s1>>4)^0x8)<<4) |
                    ((ushort)((b0.s2>>4)^0x8)<<8)     | ((ushort)((b0.s3>>4)^0x8)<<12);
        // sub_k 20..23: b0.s4567 upper nibble
        ushort p5 = ((ushort)((b0.s4>>4)^0x8))       | ((ushort)((b0.s5>>4)^0x8)<<4) |
                    ((ushort)((b0.s6>>4)^0x8)<<8)     | ((ushort)((b0.s7>>4)^0x8)<<12);
        // sub_k 24..27: b1.s0123 upper nibble
        ushort p6 = ((ushort)((b1.s0>>4)^0x8))       | ((ushort)((b1.s1>>4)^0x8)<<4) |
                    ((ushort)((b1.s2>>4)^0x8)<<8)     | ((ushort)((b1.s3>>4)^0x8)<<12);
        // sub_k 28..31: b1.s4567 upper nibble
        ushort p7 = ((ushort)((b1.s4>>4)^0x8))       | ((ushort)((b1.s5>>4)^0x8)<<4) |
                    ((ushort)((b1.s6>>4)^0x8)<<8)     | ((ushort)((b1.s7>>4)^0x8)<<12);

        // Write 8 packed uint16_t values
        int w_base = (k_start >> 2) * N + global_n;
        weights[w_base]         = p0;
        weights[w_base + N]     = p1;
        weights[w_base + 2*N]   = p2;
        weights[w_base + 3*N]   = p3;
        weights[w_base + 4*N]   = p4;
        weights[w_base + 5*N]   = p5;
        weights[w_base + 6*N]   = p6;
        weights[w_base + 7*N]   = p7;
    }

    // Per-channel scale: write N values only
    if (k_id == 0) {
        int scale_offset_bytes = NR * (K_aligned / 2 + 4);
        const __global float* scale_src = (const __global float*)(kai_packed_data + base + scale_offset_bytes);
        for (int sub_n = 0; sub_n < NR; sub_n++) {
            scales[n_id * NR + sub_n] = scale_src[sub_n] * 16.0f;
        }
    }
}
