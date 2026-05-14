// SPDX-License-Identifier: Apache-2.0
//
// Fused fp16 RMSNorm with fp32 gamma, shared between RMSNormLayer and
// ReshapedRMSNormLayer to avoid their prior fp16 -> fp32 Tensor -> fp32
// temp -> Tensor chain -> fp32 -> fp16 round-trip.
//
// Stage 0 breakdown on Qwen3-4B prefill showed the round-trip path spent
//   compute 160 + alloc 75 + gamma_mul 22 + fp16<->fp32 24 = ~280 ms
// per layer type. This helper collapses that to two passes over the
// hidden dimension (sum-of-squares then normalise+scale+write fp16) with
// fp32 accumulator, reading input + gamma and writing output in place.
//
// Shape contract:
//   X      : fp16, H * W contiguous
//   Y      : fp16, H * W contiguous
//   gamma  : fp32, length W (per-channel scale, fp32 pinned at
//            finalize() to match the .bin checkpoint layout)
//   H, W   : row count and hidden/feature width
//   eps    : fp32 epsilon added to variance before rsqrt

#ifndef CAUSALLM_RMSNORM_FUSED_FP16_H
#define CAUSALLM_RMSNORM_FUSED_FP16_H

#include <cmath>
#include <cstddef>
#include <tensor_dim.h> // _FP16

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace causallm {

#ifdef ENABLE_FP16
static inline void rmsnorm_fused_fp16(const _FP16 *__restrict__ X,
                                      _FP16 *__restrict__ Y,
                                      const float *__restrict__ gamma,
                                      std::size_t H, std::size_t W,
                                      float epsilon) {
  const float inv_W = 1.0f / static_cast<float>(W);
  for (std::size_t h = 0; h < H; ++h) {
    const _FP16 *rx = X + h * W;
    _FP16 *ry = Y + h * W;

    // -------- Pass 1: sum of squares in fp32 --------
    float sum_sq = 0.0f;
#if defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    float32x4_t acc2 = vdupq_n_f32(0.f);
    float32x4_t acc3 = vdupq_n_f32(0.f);
    std::size_t w = 0;
    for (; w + 16 <= W; w += 16) {
      float16x8_t h01 = vld1q_f16(rx + w);
      float16x8_t h23 = vld1q_f16(rx + w + 8);
      float32x4_t f0 = vcvt_f32_f16(vget_low_f16(h01));
      float32x4_t f1 = vcvt_f32_f16(vget_high_f16(h01));
      float32x4_t f2 = vcvt_f32_f16(vget_low_f16(h23));
      float32x4_t f3 = vcvt_f32_f16(vget_high_f16(h23));
      acc0 = vfmaq_f32(acc0, f0, f0);
      acc1 = vfmaq_f32(acc1, f1, f1);
      acc2 = vfmaq_f32(acc2, f2, f2);
      acc3 = vfmaq_f32(acc3, f3, f3);
    }
    for (; w + 8 <= W; w += 8) {
      float16x8_t h01 = vld1q_f16(rx + w);
      float32x4_t f0 = vcvt_f32_f16(vget_low_f16(h01));
      float32x4_t f1 = vcvt_f32_f16(vget_high_f16(h01));
      acc0 = vfmaq_f32(acc0, f0, f0);
      acc1 = vfmaq_f32(acc1, f1, f1);
    }
    float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    sum_sq = vaddvq_f32(acc);
    for (; w < W; ++w) {
      float xv = static_cast<float>(rx[w]);
      sum_sq += xv * xv;
    }
#else
    for (std::size_t w = 0; w < W; ++w) {
      float xv = static_cast<float>(rx[w]);
      sum_sq += xv * xv;
    }
#endif

    const float inv_rms = 1.0f / std::sqrt(sum_sq * inv_W + epsilon);

    // -------- Pass 2: normalise, scale by gamma, write fp16 --------
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t vinv = vdupq_n_f32(inv_rms);
    std::size_t w2 = 0;
    for (; w2 + 8 <= W; w2 += 8) {
      float16x8_t h01 = vld1q_f16(rx + w2);
      float32x4_t f0 = vcvt_f32_f16(vget_low_f16(h01));
      float32x4_t f1 = vcvt_f32_f16(vget_high_f16(h01));
      float32x4_t g0 = vld1q_f32(gamma + w2);
      float32x4_t g1 = vld1q_f32(gamma + w2 + 4);
      float32x4_t o0 = vmulq_f32(vmulq_f32(f0, vinv), g0);
      float32x4_t o1 = vmulq_f32(vmulq_f32(f1, vinv), g1);
      float16x4_t oh0 = vcvt_f16_f32(o0);
      float16x4_t oh1 = vcvt_f16_f32(o1);
      vst1q_f16(ry + w2, vcombine_f16(oh0, oh1));
    }
    for (; w2 < W; ++w2) {
      float xv = static_cast<float>(rx[w2]);
      ry[w2] = static_cast<_FP16>(xv * inv_rms * gamma[w2]);
    }
#else
    for (std::size_t w = 0; w < W; ++w) {
      float xv = static_cast<float>(rx[w]);
      ry[w] = static_cast<_FP16>(xv * inv_rms * gamma[w]);
    }
#endif
  }
}

} // namespace causallm
#endif // ENABLE_FP16

#endif // CAUSALLM_RMSNORM_FUSED_FP16_H
