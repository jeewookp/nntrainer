# GPU Matmul Optimization Status

## 달성한 것

### 1. Delegate 커널 캡처 (litert_lm 브랜치)
- Google의 closed-source `libLiteRtGpuAccelerator.so`에서 실제 `conv_wave_memory` CL 커널 소스 캡처 성공
- 바이너리 패치 방식: delegate .so의 `libOpenCL.so` → `libOpenCX.so` 문자열 교체, interceptor를 `libOpenCX.so`로 배포
- 빌드 옵션: `-qcom-accelerate-16-bit=true -cl-std=CL2.0`
- Qualcomm 확장: `qcom_sub_group_constant_load`, `ucl_wave_memory`, `cl_qcom_inline_asm`

### 2. Delegate 커널 검증 (litert_lm 브랜치)
- `delegate_kernel_bench.cc`: 캡처된 커널을 standalone으로 실행
- Weight 레이아웃 역산: `W[out_ch, in_ch]` → buffer 내 인덱스 공식 도출
- CPU fp64 reference와 비교: 127/128 match (99.2%)
- 성능: **3.45 TFLOPS** (clGetEventProfilingInfo START→END)
- Auto-tuning: 13개 local size 시도, `(32,1,2)`가 최적

### 3. 모델 통합 시도 (all_gpu 브랜치)
- `gemm_delegate_fp16_cl`: int4→fp16 dequant + delegate conv 커널 dispatch
- GPU dequant 커널 (`dequant_int4_to_fp16.cl`): SVM int4 → fp16 delegate layout
- GPU reformat 커널 (`image_reformat.cl`): SVM [M][K] ↔ image2d
- 출력 정상 확인 (Qwen3-4B 모델)
- 하지만 per-call dequant+reformat 오버헤드로 기존 int4보다 느림

### 4. GPU 파이프라인 최적화 (all_gpu 브랜치)
- SwiGLU fp16 SVM path 활성화 → WriteDataRegion/ReadDataRegion 제거
- matmul SVMMap 제거 시도 → 289 TPS 달성 (단 출력 garbage)
- SVMMap 복원 + SwiGLU SVM → **97-125 TPS** (정상 출력)

### 5. Image2d 커널 작성 (all_gpu 브랜치, 미연동)
- `rmsnorm_image2d.cl`: image2d 입출력 RMSNorm
- `swiglu_image2d.cl`: image2d 입출력 SwiGLU
- `addition_image2d.cl`: image2d 입출력 element-wise add
- `GpuImagePool`: 레이어 간 image2d 공유 매니저

## 현재 성능

| 접근 | prefill TPS | 비고 |
|------|------------|------|
| 기존 int4 v1 | ~97 | baseline (warm device) |
| SwiGLU SVM + int4 | ~125 | SwiGLU host copy 제거 |
| delegate fp16 (pipeline) | ~60 | dequant+reformat 오버헤드 |
| SVMMap 제거 (unsafe) | ~289 | 출력 garbage |

## 핵심 발견

1. **delegate 커널 자체는 int4보다 빠름** (2.70 vs 2.22 TFLOPS)
2. **느린 이유는 커널이 아닌 데이터 변환** (int4→fp16, SVM↔image2d)
3. **LiteRT가 빠른 이유: per-layer sync 0** (전체 그래프를 GPU에서 한 번에 실행)
4. **SVMMap 제거만으로 3x speedup** 가능하지만, host-side output 복사 전 sync 필요
5. **image2d 포맷으로 통일하면 reformat 완전 제거** 가능 (LiteRT 방식)

## 진행 중인 작업

### Phase 0: Delegate 커널 프로덕션 연결 + 배치 reformat 공유  (2026-04-20)

**시도한 것**
- `gemm_delegate_fp16_cl_batched` 추가 — 하나의 SVM 입력을 image2d로
  **배치당 한 번만** 변환해서 여러 weight gemm에 공유. N번의 blocking
  SVMMap을 배치 끝의 1번으로 축소.
- `HalfTensor::dot(vector, vector)` (배치 FC API) → 배치 delegate 호출로
  교체. N%32==0 && K%8==0 만족 못하면 기존 int4 경로로 자동 fallback.
- `FloatTensor` 쪽 배치 경로도 동시에 연결.
- 안전 스위치: `NNTRAINER_DISABLE_DELEGATE_GEMM=1` 로 전체 비활성 가능.

**실측 결과 (Qwen3-4B, prefill 437 토큰)**
| 경로 | prefill TPS | 비고 |
|---|---|---|
| int4 baseline | ~97-125 (이전 문서) | |
| **초기 L1 (delegate single-call 포함)** | **58.6** | dotQInteger 를 delegate 로 바꿨더니 **regression** |
| L1 revert (single-call 만 되돌림, 배치만 유지) | 측정 예정 | |

**원인 분석**
1. Qwen3 모델은 `qwen3_causallm.cpp` 에서 Q/K/V 를 **3개의 독립 FC 레이어**로
   선언 (L48/L63/L78) → `HalfTensor::dot(vector, vector)` 가 호출되지 않고
   각 FC 마다 `dotQInteger` 가 호출됨 → **배치 경로가 한 번도 실행되지 않음**.
2. Single-call delegate 는 int4 보다 per-call dispatch 수가 많음
   (`svm_to_image2d + conv + image2d_to_svm + SVMMap` vs `transpose + gemm
   + SVMMap`). 252 calls 프로파일에서 `readback: 3024.7 ms (97.6%)` —
   blocking SVMMap 이 지배적이고, delegate 쪽이 더 긴 파이프라인을 기다림.

**남긴 것**
- 배치 함수는 그대로 둠 (비용 0 — 호출되지 않음). 앞으로 QKV fused FC
  레이어를 만들거나 gate/up fused 레이어를 만들면 곧장 혜택.
- `dotQInteger` 는 기존 int4 경로로 원복.

**진짜 속도를 내려면 필요한 것 (다음 단계)**
- 옵션 1: Qwen3 모델 정의에 batched FC (Q/K/V 하나 + gate/up 하나) 추가
  → 배치 delegate 가 호출되기 시작해서 3x savings.
- 옵션 2: SVMMap blocking 제거 (Phase A-B 로 RMSNorm/SwiGLU 를 image2d
  소비자로 전환). 현재 readback 이 97% 인데 그 중 대부분이 SVM coherence
  강제 sync. image2d 체인이 되면 sync 자체가 제거됨.
- 옵션 3: tensor 페이지 정렬 + getSVMOutput staging 제거. host 스칼라
  복사가 없어지면 blocking 가 필요 없어서 in-order queue 로 자연 직렬화됨.

## 다음 단계

### Phase A: RMSNorm image2d 연동
- `rmsnorm_layer_cl.cpp` 수정: GpuImagePool에서 input image2d 확인
- 있으면: rmsnorm_image2d 커널 사용, output을 GpuImagePool에 등록
- 없으면: 기존 cl_mem path (fallback)
- 전제: pool 엔트리 invalidation 규약이 필요 (tensor 재할당 시 stale image2d
  방지). 현재 Phase 0 에서는 pool 을 건드리지 않음 — producer/consumer 양쪽이
  동시에 image2d를 지원할 때 의미 있는 최적화.

### Phase B: SwiGLU, Addition image2d 연동
- 동일 패턴

### Phase C: MHACore image2d
- 가장 복잡: QK matmul + softmax + V matmul
- 별도 image2d attention 커널 필요

### Phase D: Embedding/LM Head image2d
- 첫 입력과 최종 출력만 SVM↔image2d 변환

## 파일 구조

### litert_lm 브랜치
```
runtime/cl_bench/
  intercepted/program_002.cl   ← 캡처된 5+ TFLOPS 커널
  delegate_kernel_bench.cc     ← standalone 벤치마크
  cl_intercept.cc              ← OpenCL 인터셉터
  conv_generic_bench.cc        ← 오픈소스 ConvGeneric 벤치
  matmul_cl_bench.cc           ← naive/wave GEMM 벤치
```

### all_gpu 브랜치
```
nntrainer/tensor/cl_operations/cl_kernels/
  delegate_conv_wave.cl        ← 캡처된 delegate 커널
  int4_gemm_wave.cl            ← int4 + wave memory 시도
  dequant_int4_to_fp16.cl      ← GPU dequant 커널
  image_reformat.cl            ← SVM↔image2d 변환
  rmsnorm_image2d.cl           ← RMSNorm image2d (미연동)
  swiglu_image2d.cl            ← SwiGLU image2d (미연동)
  addition_image2d.cl          ← Addition image2d (미연동)

nntrainer/opencl/
  gpu_image_pool.h             ← 레이어간 image2d 공유

nntrainer/tensor/cl_operations/
  blas_kernels.cpp             ← gemm_delegate_fp16_cl 추가

test/unittest/
  unittest_delegate_conv_wave.cpp  ← delegate 커널 unit test
```
