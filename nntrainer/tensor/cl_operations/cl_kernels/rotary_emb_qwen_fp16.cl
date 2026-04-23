// Rotary embedding for Qwen-style attention. fp16 in/out SVM tensors,
// fp16 precomputed cos/sin tables (size seq_len x dim, cos[t][k] and
// cos[t][k+half] are duplicated by precompute_freqs_fp16 so we can index
// with k < half_ uniformly). Each work-item handles ONE half-pair, so
// the dispatch covers (M * num_heads, half_, 1) and writes BOTH
// out[k] and out[k + half_] in the same call.
//
// Math (per (m, head, k) with k < half_, pair index k):
//   x = in[m, head, k]
//   y = in[m, head, k + half_]
//   c = cos_table[(from + m) * dim + k]
//   s = sin_table[(from + m) * dim + k]
//   out[m, head, k]         = x * c - y * s
//   out[m, head, k + half_] = x * s + y * c
//
// In-place is safe (the read of x happens before writing the same
// index from the same work-item, and pair indices don't overlap).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rotary_emb_qwen_fp16(
    __global half *input, __global half *output,
    __global const half *cos_table, __global const half *sin_table,
    const int M, const int W, const int dim, const int from) {
  const int half_ = dim >> 1;
  const int num_heads = W / dim;

  const int m_head = get_global_id(0);
  const int k = get_global_id(1);
  if (k >= half_) return;
  if (m_head >= M * num_heads) return;

  const int m = m_head / num_heads;
  const int head = m_head - m * num_heads;
  const int row_base = m * W + head * dim;

  const half x = input[row_base + k];
  const half y = input[row_base + k + half_];
  const half c = cos_table[(from + m) * dim + k];
  const half s = sin_table[(from + m) * dim + k];

  output[row_base + k]         = x * c - y * s;
  output[row_base + k + half_] = x * s + y * c;
}
