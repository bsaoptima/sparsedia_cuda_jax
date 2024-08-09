cmake_minimum_required(VERSION 3.18...3.27)
project(dia_cuda LANGUAGES CUDA)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax.extend import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

enable_language(CUDA)
add_library(dia_cuda SHARED "dia_cuda.cu")
target_include_directories(dia_cuda PUBLIC ${XLA_DIR})
target_compile_features(dia_cuda PUBLIC cxx_std_17)
install(TARGETS dia_cuda LIBRARY DESTINATION ${CMAKE_CURRENT_LIST_DIR})