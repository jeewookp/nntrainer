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

## 다음 단계

### Phase A: RMSNorm image2d 연동
- `rmsnorm_layer_cl.cpp` 수정: GpuImagePool에서 input image2d 확인
- 있으면: rmsnorm_image2d 커널 사용, output을 GpuImagePool에 등록
- 없으면: 기존 cl_mem path (fallback)

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
