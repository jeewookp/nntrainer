// Q dot K^T per-head attention scoring, v2.
//
// Replaces the naive compute_kcaches_qwen_fp16.cl (which was 5x slower
// than NEON due to uncoalesced K reads — each kk thread fetched its
// own 256 B K row from DDR).  This variant uses __local tiling in the
// style of hgemm_trans_b: a workgroup loads a TM x head_dim Q tile and
// a TN x head_dim K tile cooperatively, then every thread in the WG
// reuses both tiles for its own (m, kk) output.
//
//   Q       : fp16 [M, num_heads_Q, head_dim]
//   K cache : fp16 [T_max, num_heads_KV, head_dim]   (only [0, T) valid)
//   out     : fp16 triangle-layout attention scores
//
// Dispatch:
//   global = (T_round, M_round, num_heads_Q)
//   local  = (TN, TM, 1)
//   axis 0 (inner) : kk  = group_id(0)*TN + local_id(0)
//   axis 1         : m   = group_id(1)*TM + local_id(1)
//   axis 2 (outer) : h   = group_id(2)      (one head per WG)
//
// Output indexing matches the NEON impl (and the naive kernel):
//   if is_causal:
//     out_start_row(m) = (from + m)*(from + m + 1)/2 - from*(from + 1)/2
//     out[(out_start_row(m) + kk) * num_heads_Q + h]
//   else:
//     out[(m * T + kk) * num_heads_Q + h]

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TM 8
#define TN 32
#define HD 128          // Qwen3-4B head_dim; the dispatcher enforces this.
// Pad the inner __local dim by one half so the row stride (2*(HD+PAD)
// bytes) is no longer a multiple of 32*4 = 128 bytes (Adreno's
// __local bank count * bank width).  Without the pad, k_tile[tn][d]
// with fixed d and varying tn hit the same bank across all 32 tn's
// of a wavefront -> 32-way serialisation per dot iteration, which
// measured at qk = 1777 ms on device (vs ~500 ms projected at the
// same GFLOPS after the bank pad lands).
#define PAD 1

__kernel __attribute__((reqd_work_group_size(TN, TM, 1)))
void attn_qk_wave_fp16(
    __global const half *q,
    __global const half *k_cache,
    __global half *out,
    const int M,
    const int T,
    const int from,
    const int num_heads_Q,
    const int gqa_size,
    const int head_dim,      // accepted for symmetry; must equal HD
    const int is_causal,
    const float scale) {

  const int tn = get_local_id(0);
  const int tm = get_local_id(1);
  const int gn = get_group_id(0);
  const int gm = get_group_id(1);
  const int h  = get_group_id(2);

  const int kk_block = gn * TN;
  const int m_block  = gm * TM;
  const int kk       = kk_block + tn;
  const int m        = m_block  + tm;
  const int h_kv     = h / gqa_size;

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int W_q = num_heads_Q  * HD;
  const int W_k = num_heads_KV * HD;

  // __local tiles.  Q: TM rows x (HD+PAD) cols; K: TN rows x (HD+PAD).
  // The PAD column is never touched in compute; it only shifts the
  // row stride off the 128-byte boundary to break __local bank
  // conflicts (see comment on PAD above).
  __local half q_tile[TM][HD + PAD];
  __local half k_tile[TN][HD + PAD];

  // Cooperative Q load: TM*HD = 1024 halves across 256 threads = 4/thread.
  {
    const int tid = tm * TN + tn;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = tid * 4 + i;
      const int row = idx / HD;       // 0..TM-1
      const int col = idx % HD;
      const int m_abs = m_block + row;
      const half v = (m_abs < M)
        ? q[m_abs * W_q + h * HD + col]
        : (half)0;
      q_tile[row][col] = v;
    }
  }

  // Cooperative K load: TN*HD = 4096 halves / 256 = 16/thread.
  {
    const int tid = tm * TN + tn;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int idx = tid * 16 + i;
      const int row = idx / HD;       // 0..TN-1
      const int col = idx % HD;
      const int kk_abs = kk_block + row;
      const half v = (kk_abs < T)
        ? k_cache[kk_abs * W_k + h_kv * HD + col]
        : (half)0;
      k_tile[row][col] = v;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (m >= M || kk >= T) return;
  if (is_causal != 0 && kk > from + m) return;

  // 128-dot from __local tiles, fp32 accumulator.
  float acc = 0.0f;
  #pragma unroll 8
  for (int d = 0; d < HD; ++d) {
    acc += (float)q_tile[tm][d] * (float)k_tile[tn][d];
  }

  int out_start_row;
  if (is_causal != 0) {
    const int A = (from + m) * (from + m + 1) / 2;
    const int B = from * (from + 1) / 2;
    out_start_row = A - B;
  } else {
    out_start_row = m * T;
  }

  out[(out_start_row + kk) * num_heads_Q + h] = (half)(acc * scale);
}
