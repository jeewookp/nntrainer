#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define NR 4
#define K_INTERLEAVED_V 16
#define BLOCK_LENGTH_IN_BYTES 8
#define ALIGN_32(x) (((x) + 31) & ~31)

// Dispatch: {K/32, N/4, 1}
// k_id: K/32 block index (each handles 32 K values)
// n_id: N/4 group index (each handles 4 N channels)
__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
repack_kai_to_adreno(const __global uchar* kai_packed_data,
    __global ushort* weights,
    __global half* scales,
    const int N, const int K, const int rhs_packed_stride, const int quantization_group_size) {

    const int k_id = get_global_id(0);
    const int n_id = get_global_id(1);

    const int K_aligned = ALIGN_32(K);
    const int align_N = ALIGN_32(N);
    const int base = n_id * rhs_packed_stride;
    const int k_start = k_id * 32;

    // Process 32 K values in groups of 4, for 4 N channels
    for (int sub_n = 0; sub_n < NR; sub_n++) {
        int global_n = n_id * NR + sub_n;

        for (int step = 0; step < 8; step++) {
            int k_base = k_start + step * 4;
            ushort packed = 0;

            for (int nibble_idx = 0; nibble_idx < 4; nibble_idx++) {
                int x = k_base + nibble_idx;
                if (x >= K) break;

                // Decode KAI packed data layout
                int block_base = (x / (2 * K_INTERLEAVED_V)) * (K_INTERLEAVED_V * NR);
                int block_index = ((x % K_INTERLEAVED_V) / BLOCK_LENGTH_IN_BYTES) * BLOCK_LENGTH_IN_BYTES * NR
                                + sub_n * BLOCK_LENGTH_IN_BYTES
                                + x % BLOCK_LENGTH_IN_BYTES;

                uchar byte_val = kai_packed_data[base + block_base + block_index];

                uchar nibble;
                if ((x / K_INTERLEAVED_V) % 2 == 0) {
                    nibble = byte_val & 0x0F;
                } else {
                    nibble = (byte_val >> 4) & 0x0F;
                }

                // XOR 0x8 to convert KAI encoding to unsigned affine [0,15]
                // Adreno kernel applies (val - 8) to get signed [-8,7]
                uchar val = nibble ^ 0x8;
                packed |= ((ushort)val) << (nibble_idx * 4);
            }

            weights[(k_base / 4) * N + global_n] = packed;
        }
    }

    // Extract scales: stored as float at offset NR*(K_aligned/2 + 4) bytes from block start
    // KAI stores scale as original_scale * 0.0625, multiply by 16 to recover
    int scale_offset_bytes = NR * (K_aligned / 2 + 4);
    const __global float* scale_src = (const __global float*)(kai_packed_data + base + scale_offset_bytes);

    // KAI uses per-channel scale: replicate same scale for all K/32 groups
    if (k_id == 0) {
        for (int sub_n = 0; sub_n < NR; sub_n++) {
            float s = scale_src[sub_n] * 16.0f;
            int global_n = n_id * NR + sub_n;
            for (int kg = 0; kg < (K + 31) / 32; kg++) {
                scales[kg * align_N + global_n] = (half)s;
            }
        }
    }
}
