kai_headers = [
    'kai_common.h'
]

kai_sources = [
    'kai_common_sve_asm.S'
]

subdir('matmul_clamp_f32_qai8dxp_qsi4cxp')
nntrainer_inc += include_directories('matmul_clamp_f32_qai8dxp_qsi4cxp')
nntrainer_inc_abs += meson.current_source_dir() / 'matmul_clamp_f32_qai8dxp_qsi4cxp'

subdir('matmul_clamp_f32_qsi8d32p_qsi4c32p')
nntrainer_inc += include_directories('matmul_clamp_f32_qsi8d32p_qsi4c32p')
nntrainer_inc_abs += meson.current_source_dir() / 'matmul_clamp_f32_qsi8d32p_qsi4c32p'

subdir('pack')
nntrainer_inc += include_directories('pack')
nntrainer_inc_abs += meson.current_source_dir() / 'pack'

arm_arch_opt = get_option('arm-arch')

if arm_arch_opt == 'armv9.2-a'
  foreach s : kai_sources
    nntrainer_sources += meson.current_source_dir() / s
  endforeach
endif

foreach h : kai_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
