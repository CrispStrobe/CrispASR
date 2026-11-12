# Optional GPU STFT research dependency. Never enable the product CUDA backend.
option(CRISPASR_REUSE_CUDA_FRONTEND "Enable the experimental cuFFT frontend in RE-USE research tools" OFF)
if(NOT CRISPASR_REUSE_CUDA_FRONTEND)
    return()
endif()
if(NOT TARGET crispasr-reuse-mkl)
    message(FATAL_ERROR "CRISPASR_REUSE_CUDA_FRONTEND=ON requires CRISPASR_REUSE_MKL_FRONTEND=ON and an external oneMKL SDK for independent CPU Hann generation.")
endif()

# Host APIs only: a matching SDK's headers and shared libraries suffice; nvcc
# and GGML_CUDA are not required. Explicit paths also support split SDK layouts.
find_path(CRISPASR_REUSE_CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h
    HINTS "$ENV{CUDA_HOME}" "$ENV{CUDA_PATH}" "${CUDAToolkit_ROOT}" /usr/local/cuda
    PATH_SUFFIXES include targets/x86_64-linux/include)
find_path(CRISPASR_REUSE_CUFFT_INCLUDE_DIR cufft.h
    HINTS "$ENV{CUDA_HOME}" "$ENV{CUDA_PATH}" "${CUDAToolkit_ROOT}" /usr/local/cuda
    PATH_SUFFIXES include targets/x86_64-linux/include)
find_library(CRISPASR_REUSE_CUDART_LIBRARY NAMES cudart
    HINTS "$ENV{CUDA_HOME}" "$ENV{CUDA_PATH}" "${CUDAToolkit_ROOT}" /usr/local/cuda
    PATH_SUFFIXES lib64 lib targets/x86_64-linux/lib)
find_library(CRISPASR_REUSE_CUFFT_LIBRARY NAMES cufft
    HINTS "$ENV{CUDA_HOME}" "$ENV{CUDA_PATH}" "${CUDAToolkit_ROOT}" /usr/local/cuda
    PATH_SUFFIXES lib64 lib targets/x86_64-linux/lib)
if(NOT EXISTS "${CRISPASR_REUSE_CUDA_RUNTIME_INCLUDE_DIR}/cuda_runtime_api.h"
        OR NOT EXISTS "${CRISPASR_REUSE_CUDA_RUNTIME_INCLUDE_DIR}/crt/host_defines.h"
        OR NOT EXISTS "${CRISPASR_REUSE_CUFFT_INCLUDE_DIR}/cufft.h"
        OR NOT EXISTS "${CRISPASR_REUSE_CUDART_LIBRARY}"
        OR NOT EXISTS "${CRISPASR_REUSE_CUFFT_LIBRARY}")
    message(FATAL_ERROR "CRISPASR_REUSE_CUDA_FRONTEND=ON requires an external CUDA SDK (complete cudart/cuFFT headers and libraries). Set CUDAToolkit_ROOT or CRISPASR_REUSE_CUDA_RUNTIME_INCLUDE_DIR/CRISPASR_REUSE_CUFFT_INCLUDE_DIR/CRISPASR_REUSE_CUDART_LIBRARY/CRISPASR_REUSE_CUFFT_LIBRARY; disable the option for a dependency-free default build.")
endif()

get_filename_component(_reuse_cuda_source_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(crispasr-reuse-cuda STATIC "${_reuse_cuda_source_root}/tools/ci-heavy/issue579_reuse/frontend_cuda.cpp")
target_compile_features(crispasr-reuse-cuda PUBLIC cxx_std_17)
target_compile_definitions(crispasr-reuse-cuda PUBLIC CRISPASR_REUSE_HAVE_CUDA_FRONTEND=1)
target_include_directories(crispasr-reuse-cuda PRIVATE "${CRISPASR_REUSE_CUDA_RUNTIME_INCLUDE_DIR}" "${CRISPASR_REUSE_CUFFT_INCLUDE_DIR}")
target_include_directories(crispasr-reuse-cuda PUBLIC "${_reuse_cuda_source_root}/src")
target_link_libraries(crispasr-reuse-cuda PUBLIC crispasr-reuse-mkl "${CRISPASR_REUSE_CUFFT_LIBRARY}" "${CRISPASR_REUSE_CUDART_LIBRARY}")
set_target_properties(crispasr-reuse-cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_executable(reuse-frontend-cuda-probe "${_reuse_cuda_source_root}/tools/ci-heavy/issue579_reuse/frontend_cuda_probe.cpp")
target_link_libraries(reuse-frontend-cuda-probe PRIVATE crispasr-reuse-cuda)
