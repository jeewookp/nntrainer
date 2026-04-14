# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# 1. Writer Utils
#    Bazel: cc_library(name = "litertlm_writer_utils" ...)
# ==============================================================================
add_litertlm_library(schema_cc_litertlm_writer_utils STATIC
  litertlm_writer_utils.cc
)
add_library(LiteRTLM::Schema::CC::WriterUtils ALIAS schema_cc_litertlm_writer_utils)

target_include_directories(schema_cc_litertlm_writer_utils
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(schema_cc_litertlm_writer_utils
  PUBLIC
    LiteRTLM::Schema::Core::Export
    LiteRTLM::Schema::Core::Header
    LiteRTLM::Schema::Core::HeaderSchema
    LiteRTLM::Schema::Core::Print
    LiteRTLM::Schema::Core::Section
    LiteRTLM::Schema::Core::Utils
)


if(_unverified_targets)
  # ==============================================================================
  # 2. Export Main Executable
  # ==============================================================================
  add_litertlm_executable(schema_cc_litertlm_export_main
    litertlm_export_main.cc
  )

  target_include_directories(schema_cc_litertlm_export_main
    PRIVATE
      ${GENERATED_SRC_DIR}
      ${LITERTLM_INCLUDE_PATHS}
  )

  target_link_libraries(schema_cc_litertlm_export_main
    PUBLIC
      LiteRTLM::Schema::Core::Export
      LiteRTLM::Schema::Core::Header
      LiteRTLM::Schema::Core::HeaderSchema
      LiteRTLM::Schema::Core::Section
  )

  # ==============================================================================
  # 3. Peek Executable
  # ==============================================================================
  add_litertlm_executable(schema_cc_litertlm_peek
    litertlm_peek.cc
  )

  target_include_directories(schema_cc_litertlm_peek
    PRIVATE
      ${GENERATED_SRC_DIR}
      ${CMAKE_BINARY_DIR}
      # ${LITERT_INCLUDE_PATHS}
      ${LITERTLM_INCLUDE_PATHS}
  )

  target_link_libraries(schema_cc_litertlm_peek
    PUBLIC
      LiteRTLM::Schema::Core::Read
      LiteRTLM::Schema::Core::Header
      LiteRTLM::Schema::Core::HeaderSchema
      LiteRTLM::Schema::Core::Print
      LiteRTLM::Schema::Core::Section
  )

  # ==============================================================================
  # 4. Writer Executable
  # ==============================================================================
  add_litertlm_executable(schema_cc_litertlm_writer
    litertlm_writer.cc
  )

  target_include_directories(schema_cc_litertlm_writer
    PRIVATE
      ${THIRD_PARTY_DIR}
      ${GENERATED_SRC_DIR}
      ${CMAKE_BINARY_DIR}
      ${LITERT_INCLUDE_DIR}
      ${LITERTLM_INCLUDE_PATHS}
  )

  target_link_libraries(schema_cc_litertlm_writer
    PUBLIC
      LiteRTLM::Schema::CC::WriterUtils
      LiteRTLM::Schema::Core::Export
      LiteRTLM::Schema::Core::Header
      LiteRTLM::Schema::Core::HeaderSchema
      LiteRTLM::Schema::Core::Section
  )
endif()



# ==============================================================================
# 5. Folder Facade
# ==============================================================================
add_library(schema_cc_libs INTERFACE)
add_library(LiteRTLM::Schema::CC ALIAS schema_cc_libs)

target_link_libraries(schema_cc_libs INTERFACE
  LiteRTLM::Schema::CC::WriterUtils
)