#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define NR 4
#define K_INTERLEAVED_V 16
#define BLOCK_LENGTH_IN_BYTES 8
#define ALIGN_32(x) (((x) + 31) & ~31)

// Dispatch: {K/32, N/4, 1}
// Output: dequantized FP16 weights [K][N] (pre-multiplied by scale)
__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
repack_kai_to_adreno(const __global uchar* kai_packed_data,
    __global half* dequant_weights,
    __global half* scales_out,
    const int N, const int K, const int rhs_packed_stride, const int quantization_group_size) {

    const int k_id = get_global_id(0);
    const int n_id = get_global_id(1);

    const int K_aligned = ALIGN_32(K);
    const int base = n_id * rhs_packed_stride;
    const int k_start = k_id * 32;

    // Read per-channel scale (constant across K)
    int scale_offset_bytes = NR * (K_aligned / 2 + 4);
    const __global float* scale_src = (const __global float*)(kai_packed_data + base + scale_offset_bytes);

    for (int sub_n = 0; sub_n < NR; sub_n++) {
        half scale_h = (half)(scale_src[sub_n] * 16.0f);
        int global_n = n_id * NR + sub_n;

        // Write scale once (k_id==0 only)
        if (k_id == 0) {
            scales_out[global_n] = scale_h;
        }

        for (int sub_k = 0; sub_k < 32; sub_k++) {
            int x = k_start + sub_k;
            if (x >= K) break;

            // Extract nibble from KAI packed data
            int block_base = (x / (2 * K_INTERLEAVED_V)) * (K_INTERLEAVED_V * NR);
            int block_index = ((x % K_INTERLEAVED_V) / BLOCK_LENGTH_IN_BYTES) * BLOCK_LENGTH_IN_BYTES * NR
                            + sub_n * BLOCK_LENGTH_IN_BYTES
                            + x % BLOCK_LENGTH_IN_BYTES;

            uchar byte_val = kai_packed_data[base + block_base + block_index];

            uchar nibble;
            if ((x / K_INTERLEAVED_V) % 2 == 0)
                nibble = byte_val & 0x0F;
            else
                nibble = (byte_val >> 4) & 0x0F;

            // Full dequantization: (nibble^0x8 - 8) * scale
            int signed_val = (int)(nibble ^ 0x8) - 8;
            half dq = (half)signed_val * scale_h;

            dequant_weights[x * N + global_n] = dq;
        }
    }
}
