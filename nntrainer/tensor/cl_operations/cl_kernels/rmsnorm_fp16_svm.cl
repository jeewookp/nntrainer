// SVM-direct RMSNorm with cooperative workgroup reduction. Same
// algorithm as rmsnorm_image2d_v2 but reads/writes SVM fp16 tensors
// directly, no image2d round-trip. Built specifically for the M==1
// decode path where the image2d ceremony's overhead (svm_to_image2d
// reformat -> rmsnorm_image2d_v2 -> image2d_to_svm readback ->
// svm_to_image2d_publish + blocking SVMMap fence = 4 dispatches +
// drain) dominates the actual compute by ~17x.
//
// Input  : __global const half* [H_rows * W] flat row-major
// Gamma  : __global const float* [W] (fp32 per-channel, matches the
//          model's gamma layout pinned in RMSNormLayer::finalize)
// Output : __global half* [H_rows * W]
//
// For each row r:
//   sum_sq   = sum_w(input[r * W + w]^2)
//   inv_rms  = 1 / sqrt(sum_sq / W + eps)
//   output[r * W + w] = input[r * W + w] * inv_rms * gamma[w]
//
// Dispatch: global = (WG_SIZE * H_rows, 1, 1), local = (WG_SIZE, 1, 1)
// One workgroup normalises one row. WG_SIZE = 64 = Adreno wavefront
// so the tree reduction runs one-pair-per-step on the subgroup.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void rmsnorm_fp16_svm(
    __global const half * restrict input,
    __global half * restrict output,
    __global const float * restrict gamma,
    const float epsilon,
    const int W) {

  const int lid = get_local_id(0);
  const int row = get_group_id(0);

  __global const half *in_row = input + (size_t)row * (size_t)W;
  __global half *out_row = output + (size_t)row * (size_t)W;

  // Pass 1: strided read + local square accumulation.
  float sum_sq = 0.0f;
  for (int j = lid; j < W; j += WG_SIZE) {
    float v = (float)in_row[j];
    sum_sq += v * v;
  }

  // Tree reduction across the workgroup.
  __local float shared_sum[WG_SIZE];
  shared_sum[lid] = sum_sq;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int step = WG_SIZE / 2; step > 0; step >>= 1) {
    if (lid < step) {
      shared_sum[lid] += shared_sum[lid + step];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  const float mean = shared_sum[0] / (float)W;
  const float inv_rms = 1.0f / sqrt(mean + epsilon);

  // Pass 2: normalise, scale by gamma, write.
  for (int j = lid; j < W; j += WG_SIZE) {
    float v = (float)in_row[j];
    out_row[j] = (half)(v * inv_rms * gamma[j]);
  }
}
