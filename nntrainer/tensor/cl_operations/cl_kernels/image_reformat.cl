#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Transpose SVM [M][K] → image2d (width=M, height=K/4, RGBA half)
// pixel(m, s).c = input[m * K + s*4 + c]
__kernel void svm_to_image2d(
    __global const half* restrict input,
    __write_only image2d_t dst,
    const int M,
    const int K) {
  int m = get_global_id(0);
  int s = get_global_id(1);
  if (m >= M || s >= K/4) return;
  half4 v;
  int base = m * K + s * 4;
  v.x = input[base];
  v.y = (s*4+1 < K) ? input[base+1] : (half)0;
  v.z = (s*4+2 < K) ? input[base+2] : (half)0;
  v.w = (s*4+3 < K) ? input[base+3] : (half)0;
  write_imageh(dst, (int2)(m, s), v);
}

// Transpose image2d (width=M, height=N/4, RGBA half) → SVM [M][N]
// output[m * N + s*4 + c] = pixel(m, s).c
__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void image2d_to_svm(
    __read_only image2d_t src,
    __global half* restrict output,
    const int M,
    const int N) {
  const int m = get_global_id(0);
  const int s = get_global_id(1);
  if (m >= M || s >= N / 4) return;
  const half4 v = read_imageh(src, smp, (int2)(m, s));
  // N % 4 == 0 is guaranteed (delegate requires N % 32 == 0), so base is
  // always a 4-half-aligned offset and we can vector-store. This cuts the
  // SVM write from 4 scalar half stores (= 4 coherent-SVM transactions on
  // Adreno's coarse-grained SVM) to one half4 store.
  __global half4 *out4 = (__global half4 *)(output + (m * N + s * 4));
  *out4 = v;
}
