fs = import('fs')

cl_op_kernels = [
  'addition.cl',
  'concat_axis_1.cl',
  'concat_axis_2.cl',
  'concat_axis_3.cl',
  'convert_block_q4_0.cl',
  'copy.cl',
  'dot.cl',
  'embedding.cl',
  'int4_gemv.cl',
  'int4_gemv_adreno.cl',
  'int4_gemv_adreno_v2.cl',
  'int4_gemv_adreno_v3.cl',
  'int4_gemv_adreno_v4.cl',
  'int4_gemv_weight_image2d.cl',
  'int4_gemv_weight_image2d_v2.cl',
  'int4_gemv_weight_image2d_v3.cl',
  'int4_gemv_weight_image2d_v4.cl',
  'int4_gemv_image_v1.cl',
  'int4_gemv_image_v2.cl',
  'int4_gemv_image_v3.cl',
  'int4_gemv_image_v4.cl',
  'int4_gemv_image_v5.cl',
  'int4_gemv_image2d.cl',
  'fused_rmsnorm_qkv.cl',
  'fused_rmsnorm_gate_up.cl',
  'fused_gemv_int4.cl',
  'fused_gemv_int4_v2.cl',
  'fused_gemv_int4_image2d.cl',
  'fused_gemv_int4_image2d_v2.cl',
  'gateup_swiglu_int4_image2d.cl',
  'int4_quantize_input.cl',
  'int4_gemm.cl',
  'delegate_conv_wave.cl',
  'dequant_int4_to_fp16.cl',
  'image_reformat.cl',
  'rmsnorm_image2d.cl',
  'rmsnorm_image2d_v2.cl',
  'addition_image2d.cl',
  'swiglu_image2d.cl',
  'int4_gemm_adreno.cl',
  'int4_gemm_adreno_v2.cl',
  'int4_gemm_adreno_v3.cl',
  'int8_int4_gemm_adreno.cl',
  'input_transpose.cl',
  'q4_0_ab_bi_8x4.cl',
  'q6_k_sgemv.cl',
  'q6_k_sgemv_fp16.cl',
  'restore_block_q4_0.cl',
  'rmsnorm.cl',
  'rotary_emb.cl',
  'sgemm_no_trans.cl',
  'sgemm_trans_a.cl',
  'sgemm_trans_ab.cl',
  'sgemm_trans_b.cl',
  'sgemv_no_trans.cl',
  'sgemv.cl',
  'sscal.cl',
  'swiglu.cl',
  'transpose_16bit.cl',
  'transpose_32bit_16bit.cl',
  'transpose_axis_0.cl',
  'transpose_axis_1.cl',
  'transpose_axis_2.cl',
]

if get_option('enable-fp16')
  cl_op_kernels += [
    'add2_fp16_svm.cl',
    'addition_fp16.cl',
    'concat_axis_1_fp16.cl',
    'concat_axis_2_fp16.cl',
    'concat_axis_3_fp16.cl',
    'copy_fp16.cl',
    'dot_fp16.cl',
    'embedding_fp16.cl',
    'hgemm_no_trans.cl',
    'hgemm_trans_a.cl',
    'hgemm_trans_ab.cl',
    'hgemm_trans_b.cl',
    'hgemv_no_trans.cl',
    'hgemv.cl',
    'hscal.cl',
    'rmsnorm_fp16.cl',
    'rmsnorm_fp16_svm.cl',
    'rotary_emb_fp16.cl',
    'attention_fused_fp16.cl',
    'attention_fused_fp16_decode.cl',
    'attention_fused_fp16_decode_v2.cl',
    'attention_fused_fp16_decode_v3.cl',
    'attention_fused_fp16_decode_gqa.cl',
    'swiglu_fp16.cl',
    'swiglu_fp16_image2d.cl',
    'transpose_axis_0_fp16.cl',
    'transpose_axis_1_fp16.cl',
    'transpose_axis_2_fp16.cl',
  ]
endif

all_includes = ''
cl_kernel_generated_headers = []
foreach k : cl_op_kernels
  basename = fs.stem(k)
  kernel_code = fs.read(k).split('\n')
  modified_code = []
  foreach line : kernel_code
    modified_code += ('R"(' + line + ')"' + ' "\\n"')
  endforeach
  variable_name = basename + '_kernel'
  if basename == 'int4_gemm'
    variable_name = 'gemm_int4_kernel'
  endif
  conf_data = configuration_data(
    {
      'namespace': 'nntrainer',
      'variable_name': variable_name,
      'header_path': basename + '.h',
      'kernel_code': '\n'.join(modified_code),
    },
  )

  configure_file(
    input: 'kernel.cpp.in',
    output: basename + '.cpp',
    configuration: conf_data,
  )
  generated_h = configure_file(
    input: 'kernel.h.in',
    output: basename + '.h',
    configuration: conf_data,
  )

  # source.full_path() does not work: https://github.com/mesonbuild/meson/issues/5273#issuecomment-1851811417
  # can be used when meson is upgraded to 1.4.0
  # NOTE: Generated kernel headers are NOT added to nntrainer_headers because
  # they have generic names (swiglu.h, embedding.h, rmsnorm.h, ...) that
  # would collide with application-side headers when installed flat into
  # include/nntrainer/. Instead, they are collected here and installed under
  # include/nntrainer/cl_kernels/ via a separate install_headers() call below.
  cl_kernel_generated_headers += generated_h
  nntrainer_sources += meson.current_build_dir() / basename + '.cpp'

  all_includes += f'#include "@basename@.h"\n'
endforeach
generated_kernels_h = configure_file(
  input: 'kernels.h.in',
  output: 'cl_kernels.h',
  configuration: {'includes': all_includes},
)
cl_kernel_generated_headers += generated_kernels_h

# Install all generated CL kernel headers into a dedicated subdirectory so they
# do not pollute include/nntrainer/ with names that may collide with public
# layer headers (e.g. swiglu.h, embedding.h, rmsnorm.h).
install_headers(cl_kernel_generated_headers,
  install_dir: nntrainer_includedir / 'cl_kernels',
)

nntrainer_inc += include_directories('..')
nntrainer_inc_abs += meson.current_build_dir() / '..'