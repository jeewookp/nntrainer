__kernel void embedding_cl(
    __global const float *input,   // token IDs (float -> cast to int)
    __global const float *weight,  // embedding table [vocab_size * out_dim]
    __global float *output,        // output [num_tokens * out_dim]
    const int out_dim,
    const float scale) {
  const int gid = get_global_id(0);
  const int token_idx = gid / out_dim;
  const int dim_idx = gid % out_dim;
  const int embed_idx = (int)input[token_idx];
  output[gid] = weight[embed_idx * out_dim + dim_idx] * scale;
}
