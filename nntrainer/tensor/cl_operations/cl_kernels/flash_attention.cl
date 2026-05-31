// SPDX-License-Identifier: Apache-2.0
// Flash-attention-style single-kernel prefill / decode attention for
// Adreno (paper ML Drift §3.6 fusion + Dao et al. 2022 online softmax).
//
// REPLACES: the three-kernel two_conv_attention.cl path which is
// (a) slower than CPU NEON on Adreno 830 due to VGPR spill (each WI's
// TM_QK*TN_QK accumulator floods the register file), and (b) routes
// the full scores[H, M, N_kv] tensor through global memory between
// kernels, which is ~6.7 MB bandwidth per prefill that we don't need.
//
// PIPELINE (one kernel):
//   For each (head_q, query_row):
//     - online maximum / sum / output accumulator (head_dim FP32 reg)
//     - serial loop over K/V rows n=0..N_kv-1 (causal: n<=m):
//       * s = scale * dot(Q[m,head_q], K[n,head_kv])   (fp32 accum)
//       * m_new = max(m_i, s); alpha = exp(m_i - m_new); p = exp(s - m_new)
//       * l_i = alpha*l_i + p
//       * acc[:] = alpha*acc[:] + p*V[n,head_kv,:]
//       * m_i = m_new
//     - write O[query_row, head_q*d : (head_q+1)*d] = acc / l_i
//
// CORRECTNESS-FIRST DESIGN (Step #2):
//   ONE work-item per (head_q, query_row). The full d=128 fp32 output
//   accumulator (512 B) + Q row (256 B fp16) live in private memory.
//   No LDS, no inter-WI reduction, no scores DRAM materialization — the
//   whole point (removing the [H,M,N_kv] global traffic) is achieved by
//   the single-WI serial form. Tiling/LDS cooperation is a follow-up.
//   gws = (num_heads_q * M,); lws chosen by host (small, e.g. 64).
//
// K-LAYOUT NOTE: the gpu_native NNTR_OHWI_IMG=0 path stores cache_k_svm
//   in OHWI form  K[head_kv * S_max * d + n * d + x]  (qk_matmul_f16_ohwi
//   layout, NOT the pure concat [N_kv, HD_KV]). V stays concat
//   V[n * HD_KV + head_kv * d + x] (sv_matmul_f16 layout). To feed the
//   EXACT SAME buffers as the 3-kernel _ohwi_cl fallback, this kernel
//   takes a k_stride param: if k_stride > 0, K is OHWI with that S_max
//   row-stride; if k_stride <= 0, K is pure concat (HD_KV stride). Q and
//   O are always concat. This keeps the flash path bit-comparable to the
//   baseline it replaces.
//
// QWEN3-0.6B SHAPES (for reference; kernel is parameterized):
//   M = step_size (prefill: input length; decode: 1)
//   N_kv = cache_to (running cache fill, <= MAX_SEQ_LEN)
//   d = head_dim = 128
//   num_heads_q = 16, num_heads_kv = 8 (GQA = 2)
//   HD_Q = num_heads_q * d = 2048
//   HD_KV = num_heads_kv * d = 1024

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef FLASH_BLOCK_KV
#define FLASH_BLOCK_KV 32
#endif

#ifndef FLASH_MAX_D
#define FLASH_MAX_D 128
#endif

// Single-WI-per-(head_q, query_row) flash attention prefill. Online
// softmax (Dao et al.) so scores are never materialized to global memory.
// Signature matches the host wrapper flash_attention_prefill_f16_cl.
__kernel void flash_attention_prefill_f16_skeleton(
    __global const half *Q,           // [M, HD_Q] fp16, row-major (concat)
    __global const half *K,           // OHWI [H_kv,S_max,d] or concat [N_kv,HD_KV]
    __global const half *V,           // [N_kv, HD_KV] fp16, row-major (concat)
    __global       half *O,           // [M, HD_Q] fp16, row-major (concat)
    const int M,
    const int N_kv,
    const int d,                       // head_dim
    const int HD_Q,                    // num_heads_q * d
    const int HD_KV,                   // num_heads_kv * d
    const int gqa,                     // num_heads_q / num_heads_kv
    const int is_causal,
    const float scale,                 // 1 / sqrt(d), precomputed
    const int k_stride                 // >0: K OHWI S_max row-stride; <=0: concat
) {
  const int gid = get_global_id(0);     // decodes to (head_q, query_row)
  const int head_q = gid / M;
  const int m = gid % M;
  if (m >= M) return;
  const int total = (HD_Q / d) * M;     // num_heads_q * M
  if (gid >= total) return;

  const int head_kv = head_q / gqa;

  // K base offset for this (head_kv). OHWI: head_kv*S_max*d + n*d + x.
  // concat: n*HD_KV + head_kv*d + x. We fold the per-n stride into k_row.
  const long k_head_base =
      (k_stride > 0) ? ((long)head_kv * (long)k_stride * (long)d)
                     : ((long)head_kv * (long)d);
  const long k_row_stride = (k_stride > 0) ? (long)d : (long)HD_KV;

  const long q_base = (long)m * HD_Q + (long)head_q * d;

  // Load this query row into private fp32 registers.
  float q_row[FLASH_MAX_D];
  for (int x = 0; x < d; ++x)
    q_row[x] = (float)Q[q_base + x];

  // Online-softmax state.
  float m_i = -INFINITY;
  float l_i = 0.0f;
  float acc[FLASH_MAX_D];
  for (int x = 0; x < d; ++x)
    acc[x] = 0.0f;

  // Causal: key n is masked when is_causal && n > m. So the valid range
  // is n in [0, n_last] where n_last = is_causal ? min(N_kv-1, m) : N_kv-1.
  const int n_last = is_causal ? min(N_kv - 1, m) : (N_kv - 1);

  for (int n = 0; n <= n_last; ++n) {
    const long k_base = k_head_base + (long)n * k_row_stride;
    const long v_base = (long)n * HD_KV + (long)head_kv * d;

    // s = scale * dot(Q[m,head_q], K[n,head_kv]) in fp32.
    float dot = 0.0f;
    for (int x = 0; x < d; ++x)
      dot += q_row[x] * (float)K[k_base + x];
#ifdef FLASH_FP16_SCORE
    // Match the 3-kernel baseline, which writes scores as fp16 before
    // softmax (qk_matmul_f16 stores (half)(acc*scale)). Truncating here
    // makes the flash path bit-comparable to that baseline.
    const float s = (float)((half)(scale * dot));
#else
    const float s = scale * dot;
#endif

    // Online softmax update (Dao et al.).
    const float m_new = fmax(m_i, s);
    const float alpha = exp(m_i - m_new);   // m_i==-inf, m_new finite => 0
    const float p = exp(s - m_new);
    l_i = alpha * l_i + p;
    for (int x = 0; x < d; ++x)
      acc[x] = alpha * acc[x] + p * (float)V[v_base + x];
    m_i = m_new;
  }

  // Normalize and write out. l_i == 0 only when no key was attended
  // (e.g. causal row with N_kv==0); guard to avoid NaN.
  const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  const long o_base = (long)m * HD_Q + (long)head_q * d;
  for (int x = 0; x < d; ++x)
    O[o_base + x] = (half)(acc[x] * inv);
}

// ===========================================================================
// COOPERATIVE flash attention prefill — d-AXIS-TILED online softmax.
//
// Motivation (measured on Intel Arc 0x7d55): the naive 1-WI variant and the
// first "split-the-key-loop + tree-reduce" coop attempt BOTH keep a full
// private acc[d=128] + q_row[d=128] per work-item. clGetKernelWorkGroupInfo
// reported CL_KERNEL_PRIVATE_MEM_SIZE = 16384 B/WI for both => the compiler
// spills that to global scratch, and the kernel becomes scratch-bandwidth
// bound (8.2 s @ M=1024 vs the 3-kernel path's ~1.0 s). The key-split coop
// made it WORSE by stacking a 32 KB LDS acc reduction on top of the same
// 16 KB private spill (max_wg collapsed to 64, occupancy died).
//
// FIX: tile the head_dim across the work-group so NO work-item ever holds a
// full d-wide vector. One WORKGROUP per (head_q, query_row), LWS work-items:
//   - Q row -> LDS once (q_sh[d], cooperative load), reused for every key.
//   - acc[d] -> LDS (acc_sh[d]), shared online-softmax output accumulator;
//     WI `lid` owns the d-lanes  x = lid, lid+LWS, ...   (tiny private state).
//   - m_i, l_i -> LDS scalars (shared online-softmax running max / denom).
//   - For each key n (serial over the WG, all WIs in lockstep):
//       * each WI computes a PARTIAL dot over its d-lanes (q_sh[x]*K[..x]);
//       * tree-reduce the LWS partials in LDS -> scalar score s (red_sh[]);
//       * one online-softmax step: alpha=exp(m_i-m_new), p=exp(s-m_new);
//         each WI updates ITS acc lanes acc_sh[x]=alpha*acc_sh[x]+p*V[..x],
//         and l_i=alpha*l_i+p ; m_i=m_new (scalars updated by all, identical).
//   - Finally each WI writes O[.. x] = acc_sh[x]/l_i for its d-lanes.
//
// Private footprint per WI is now O(1) floats (a couple of scalars + a small
// strided loop index) => NO spill. LDS = q_sh[d] + acc_sh[d] + red_sh[LWS]
// + a few scalars = 128*4 + 128*4 + LWS*4 + 16 ≈ 1.3 KB (LWS=64). Fits both
// Intel (64 KB) and Adreno (32 KB) with huge occupancy headroom.
//
// Score-reduction tree is a portable log-step LDS reduction (NO subgroup-64
// assumption) => Adreno-portable. LWS must be a power of two and divide d
// reasonably (each WI owns ceil(d/LWS) lanes); LWS in {16,32,64,128}.
//
// Same signature / layout contract as the naive variant (K OHWI via k_stride,
// V concat, causal via n_last, FLASH_FP16_SCORE diag off by default).
// ===========================================================================
#ifndef FLASH_COOP_LWS
#define FLASH_COOP_LWS 64
#endif

// Keys processed per reduction phase (amortizes the log-step dot-reduction
// barriers over a tile of keys). LDS red_sh = BLOCK_KV*LWS*4 B
// (4*64*4 = 1 KB). Must keep total LDS <= 32 KB for Adreno. Default 4 is the
// measured Intel Arc sweet spot (BLOCK_KV 2-4 ~equal; 1 and >=8 slower).
#ifndef FLASH_COOP_BLOCK_KV
#define FLASH_COOP_BLOCK_KV 4
#endif

__attribute__((reqd_work_group_size(FLASH_COOP_LWS, 1, 1)))
__kernel void flash_attention_prefill_f16_coop(
    __global const half *Q,           // [M, HD_Q] fp16, row-major (concat)
    __global const half *K,           // OHWI [H_kv,S_max,d] or concat [N_kv,HD_KV]
    __global const half *V,           // [N_kv, HD_KV] fp16, row-major (concat)
    __global       half *O,           // [M, HD_Q] fp16, row-major (concat)
    const int M,
    const int N_kv,
    const int d,                       // head_dim
    const int HD_Q,                    // num_heads_q * d
    const int HD_KV,                   // num_heads_kv * d
    const int gqa,                     // num_heads_q / num_heads_kv
    const int is_causal,
    const float scale,                 // 1 / sqrt(d), precomputed
    const int k_stride                 // >0: K OHWI S_max row-stride; <=0: concat
) {
  const int lid = get_local_id(0);
  const int grp = get_group_id(0);      // decodes to (head_q, query_row)
  const int head_q = grp / M;
  const int m = grp % M;
  const int total_groups = (HD_Q / d) * M;   // num_heads_q * M
  if (grp >= total_groups || m >= M)
    return;

  const int head_kv = head_q / gqa;

  const long k_head_base =
      (k_stride > 0) ? ((long)head_kv * (long)k_stride * (long)d)
                     : ((long)head_kv * (long)d);
  const long k_row_stride = (k_stride > 0) ? (long)d : (long)HD_KV;
  const long q_base = (long)m * HD_Q + (long)head_q * d;

  // Shared per-(head_q,m) state in LDS. q_sh: query row; acc_sh: output
  // accumulator (WI lid owns disjoint d-lanes lid,lid+LWS,...). red_sh:
  // score-dot reduction scratch, sized [BLOCK_KV][LWS] so a whole key tile
  // reduces with ONE set of log-step barriers instead of one set per key.
  __local float q_sh[FLASH_MAX_D];
  __local float acc_sh[FLASH_MAX_D];
  __local float red_sh[FLASH_COOP_BLOCK_KV * FLASH_COOP_LWS];

  // m_i / l_i are kept PRIVATE in every WI and stay identical across the WG
  // (all WIs read the same reduced score) — this removes the per-key cross-WI
  // barrier the shared-scalar version needed. acc_sh lanes are WI-private
  // (disjoint), so consecutive keys need NO barrier between acc updates; the
  // only barriers are inside the dot reduction.
  float m_i = -INFINITY;
  float l_i = 0.0f;

  // Cooperative load of Q row + zero acc. WI lid owns d-lanes lid, lid+LWS...
  for (int x = lid; x < d; x += FLASH_COOP_LWS) {
    q_sh[x] = (float)Q[q_base + x];
    acc_sh[x] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const int n_last = is_causal ? min(N_kv - 1, m) : (N_kv - 1);

  // Key-blocked loop: process up to BLOCK_KV keys per reduction phase.
  for (int n0 = 0; n0 <= n_last; n0 += FLASH_COOP_BLOCK_KV) {
    const int nb = min(FLASH_COOP_BLOCK_KV, n_last - n0 + 1);

    // (1) Each WI computes its PARTIAL d-dot for every key in the tile and
    //     stages them into red_sh[j*LWS + lid].
    for (int j = 0; j < nb; ++j) {
      const long k_base = k_head_base + (long)(n0 + j) * k_row_stride;
      float part = 0.0f;
      for (int x = lid; x < d; x += FLASH_COOP_LWS)
        part += q_sh[x] * (float)K[k_base + x];
      red_sh[j * FLASH_COOP_LWS + lid] = part;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // (2) Tree-reduce ALL nb columns together: log2(LWS) barrier rounds for
    //     the whole tile (not per key). Each active WI folds nb partials.
    for (int off = FLASH_COOP_LWS >> 1; off > 0; off >>= 1) {
      if (lid < off)
        for (int j = 0; j < nb; ++j)
          red_sh[j * FLASH_COOP_LWS + lid] +=
              red_sh[j * FLASH_COOP_LWS + lid + off];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    // red_sh[j*LWS] now holds the full fp32 dot for key (n0+j).

    // (3) Serial online-softmax over the tile. No barrier between keys: each
    //     WI updates only its own acc_sh lanes and its private m_i/l_i.
    for (int j = 0; j < nb; ++j) {
#ifdef FLASH_FP16_SCORE
      const float s = (float)((half)(scale * red_sh[j * FLASH_COOP_LWS]));
#else
      const float s = scale * red_sh[j * FLASH_COOP_LWS];
#endif
      const float m_new = fmax(m_i, s);
      const float alpha = exp(m_i - m_new);   // m_i=-inf, m_new finite => 0
      const float p = exp(s - m_new);
      const long v_base = (long)(n0 + j) * HD_KV + (long)head_kv * d;
      for (int x = lid; x < d; x += FLASH_COOP_LWS)
        acc_sh[x] = alpha * acc_sh[x] + p * (float)V[v_base + x];
      l_i = alpha * l_i + p;
      m_i = m_new;
    }
    // acc_sh lanes are WI-private; next tile's reduction barrier (step 1->2)
    // re-fences red_sh. A barrier here ensures the tile's acc writes are
    // settled before red_sh is overwritten by the next tile (red_sh and
    // acc_sh are distinct, but the staging write of step (1) must not race
    // a still-reading WI of step (3) — they read red_sh, step (1) writes it).
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Normalize and write O for this WI's d-lanes.
  const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  const long o_base = (long)m * HD_Q + (long)head_q * d;
  for (int x = lid; x < d; x += FLASH_COOP_LWS)
    O[o_base + x] = (half)(acc_sh[x] * inv);
}
