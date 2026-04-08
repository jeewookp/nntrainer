#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define NR 4
#define K_INTERLEAVED_V 16
#define BLOCK_LENGTH_IN_BYTES 8

// Dispatch: {K/32, N/4, 1}
// k_id: K/32 block index (each handles 32 K values)
// n_id: N/4 group index (each handles 4 N channels)
__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
repack_kai_to_adreno(const __global uchar* kai_packed_data,
    __global ushort* weights,
    __global half* scales,
    const int N, const int K, const int rhs_packed_stride, const int quantization_group_size) {

    const int k_id = get_global_id(0);  // K/32 block index
    const int n_id = get_global_id(1);  // N/4 group index

    const int K_aligned = ((K + 31) / 32) * 32;
    const int base = n_id * rhs_packed_stride;
    const int k_start = k_id * 32;

    // Extract and repack 32 K values for 4 N channels
    // Adreno format: weight_ptr[(k/4)*N + n] = uint16_t with 4 nibbles
    for (int sub_k = 0; sub_k < 32; sub_k++) {
        int x = k_start + sub_k;
        if (x >= K) break;

        ushort4 packed = (ushort4)(0, 0, 0, 0);

        for (int sub_n = 0; sub_n < NR; sub_n++) {
            // Decode KAI packed data layout
            int block_base = (x / (2 * K_INTERLEAVED_V)) * (K_INTERLEAVED_V * NR);
            int block_index = ((x % K_INTERLEAVED_V) / BLOCK_LENGTH_IN_BYTES) * BLOCK_LENGTH_IN_BYTES * NR
                            + sub_n * BLOCK_LENGTH_IN_BYTES
                            + x % BLOCK_LENGTH_IN_BYTES;

            uchar byte_val = kai_packed_data[base + block_base + block_index];

            uchar nibble;
            if ((x / K_INTERLEAVED_V) % 2 == 0) {
                nibble = byte_val & 0x0F;  // lower nibble
            } else {
                nibble = (byte_val >> 4) & 0x0F;  // upper nibble
            }

            // nibble is already in [0,15] with bias +8 (value ^ 0x8 gives signed)
            // Adreno kernel expects the same biased format
            int global_n = n_id * NR + sub_n;
            int shift = (x % 4) * 4;

            if (sub_n == 0) packed.s0 |= ((ushort)nibble) << shift;
            else if (sub_n == 1) packed.s1 |= ((ushort)nibble) << shift;
            else if (sub_n == 2) packed.s2 |= ((ushort)nibble) << shift;
            else packed.s3 |= ((ushort)nibble) << shift;
        }

        // Write to Adreno weight buffer at aligned k/4 positions
        if (x % 4 == 3) {
            int weight_row = x / 4;
            weights[weight_row * N + n_id * NR + 0] = packed.s0;
            weights[weight_row * N + n_id * NR + 1] = packed.s1;
            weights[weight_row * N + n_id * NR + 2] = packed.s2;
            weights[weight_row * N + n_id * NR + 3] = packed.s3;
        }
    }

    // Handle the last partial group of 4 if K is not multiple of 4
    // (K is expected to be multiple of 32 from KAI, so this is rarely needed)

    // Extract scales: stored as float at offset 4*(K_aligned/2 + 4) from block start
    // Each n_id block has 4 float scales, stored as original_scale * 0.0625
    // We need to multiply by 16 and convert to FP16
    int scale_offset_bytes = NR * (K_aligned / 2 + 4);
    const __global float* scale_src = (const __global float*)(kai_packed_data + base + scale_offset_bytes);

    int scale_row = k_id;  // k_id corresponds to K/32 block
    // KAI has one scale per channel (not per group), so only write once for k_id==0
    if (k_id == 0) {
        for (int sub_n = 0; sub_n < NR; sub_n++) {
            float s = scale_src[sub_n] * 16.0f;
            int global_n = n_id * NR + sub_n;
            // Write the same scale for all K/32 groups since KAI uses per-channel scale
            for (int kg = 0; kg < (K + 31) / 32; kg++) {
                scales[kg * N + global_n] = (half)s;
            }
        }
    }
}
