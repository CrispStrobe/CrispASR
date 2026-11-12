cmake_minimum_required(VERSION 3.18)
project(reuse_research_tools LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
get_filename_component(REUSE_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(CRISPASR_REUSE_GGML_SOURCE_DIR "${REUSE_SOURCE_ROOT}/ggml" CACHE PATH "Pinned ggml checkout for isolated CPU research tools")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
add_subdirectory("${CRISPASR_REUSE_GGML_SOURCE_DIR}" ggml)
include("${REUSE_SOURCE_ROOT}/cmake/CrispasrReuseMkl.cmake")
include("${REUSE_SOURCE_ROOT}/cmake/CrispasrReuseCuda.cmake")

# This library belongs only to this standalone project. Do not add the loader
# to crispasr-lib: the scalar reference is not a supported inference backend.
add_library(reuse-research-loader STATIC
    "${REUSE_SOURCE_ROOT}/src/core/gguf_loader.cpp"
    "${REUSE_SOURCE_ROOT}/src/core/reuse_model_loader.cpp")
target_include_directories(reuse-research-loader PUBLIC "${REUSE_SOURCE_ROOT}/src"
    PRIVATE "${CRISPASR_REUSE_GGML_SOURCE_DIR}/src")
target_link_libraries(reuse-research-loader PUBLIC ggml ggml-cpu)

add_executable(reuse-research-diff diff_probe.cpp
    "${REUSE_SOURCE_ROOT}/examples/cli/crispasr_diff.cpp")
target_include_directories(reuse-research-diff PRIVATE "${REUSE_SOURCE_ROOT}/examples/cli")
target_link_libraries(reuse-research-diff PRIVATE reuse-research-loader)
if(TARGET crispasr-reuse-mkl)
    target_link_libraries(reuse-research-diff PRIVATE crispasr-reuse-mkl)
endif()
if(TARGET crispasr-reuse-cuda)
    target_link_libraries(reuse-research-diff PRIVATE crispasr-reuse-cuda)
endif()

enable_testing()
# Explicit five-second research comparator; default comparator remains 1 s.
add_executable(reuse-long-diff long_diff_probe.cpp
    "${REUSE_SOURCE_ROOT}/examples/cli/crispasr_diff.cpp")
target_include_directories(reuse-long-diff PRIVATE "${REUSE_SOURCE_ROOT}/examples/cli")
target_link_libraries(reuse-long-diff PRIVATE reuse-research-loader)
if(TARGET crispasr-reuse-mkl)
    target_link_libraries(reuse-long-diff PRIVATE crispasr-reuse-mkl)
endif()
if(TARGET crispasr-reuse-cuda)
    target_link_libraries(reuse-long-diff PRIVATE crispasr-reuse-cuda)
endif()

add_executable(reuse-diff-controls "${REUSE_SOURCE_ROOT}/tests/test-reuse-diff-controls.cpp"
    "${REUSE_SOURCE_ROOT}/examples/cli/crispasr_diff.cpp")
target_link_libraries(reuse-diff-controls PRIVATE reuse-research-loader)
add_test(NAME reuse-diff-controls COMMAND reuse-diff-controls
    "${CMAKE_CURRENT_BINARY_DIR}/reuse-diff-controls.gguf")
set_tests_properties(reuse-diff-controls PROPERTIES LABELS "unit;reuse;research" TIMEOUT 60)
set(REUSE_FIXTURES "${REUSE_SOURCE_ROOT}/tests/fixtures")
function(reuse_fixture_test target)
    add_executable(${target} "${REUSE_SOURCE_ROOT}/tests/test-${target}.cpp")
    add_test(NAME ${target} COMMAND ${target} ${ARGN})
    set_tests_properties(${target} PROPERTIES LABELS "unit;reuse;research" TIMEOUT 60)
endfunction()
reuse_fixture_test(reuse-frontend "${REUSE_SOURCE_ROOT}/samples/jfk.wav"
    "${REUSE_FIXTURES}/reuse-stft-reference-f32le.bin"
    "${REUSE_FIXTURES}/reuse-stft-44100-reference-f32le.bin"
    "${REUSE_FIXTURES}/reuse-istft-reference-f32le.bin")
foreach(component dense-encoder scan mamba bimamba tf-mamba decoder)
    reuse_fixture_test(reuse-${component} "${REUSE_FIXTURES}/reuse-${component}-f32le.bin")
endforeach()
reuse_fixture_test(reuse-inventory "${REUSE_FIXTURES}/reuse-inventory.json")
reuse_fixture_test(reuse-model-loader)
target_link_libraries(reuse-model-loader PRIVATE reuse-research-loader)
foreach(component dense scan mamba bimamba tf-mamba decoder network)
    add_executable(test-reuse-${component}-oracle
        "${REUSE_SOURCE_ROOT}/tests/test-reuse-${component}-oracle.cpp")
    if(component STREQUAL "network")
        target_link_libraries(test-reuse-${component}-oracle PRIVATE reuse-research-loader)
    endif()
endforeach()
add_executable(reuse-profile-default "${REUSE_SOURCE_ROOT}/tests/test-reuse-frontend-profile.cpp")
target_include_directories(reuse-profile-default PRIVATE "${REUSE_SOURCE_ROOT}/src")
add_test(NAME reuse-profile-default COMMAND reuse-profile-default)
add_executable(reuse-network-controls "${REUSE_SOURCE_ROOT}/tests/test-reuse-reference-network.cpp")
target_link_libraries(reuse-network-controls PRIVATE ggml)
add_test(NAME reuse-network-controls COMMAND reuse-network-controls)
if(TARGET crispasr-reuse-mkl)
    add_executable(reuse-profile-mkl "${REUSE_SOURCE_ROOT}/tests/test-reuse-frontend-profile.cpp")
    target_include_directories(reuse-profile-mkl PRIVATE "${CRISPASR_REUSE_MKL_INCLUDE_DIR}")
    target_link_libraries(reuse-profile-mkl PRIVATE crispasr-reuse-mkl)
    add_test(NAME reuse-profile-mkl COMMAND reuse-profile-mkl)
endif()
if(TARGET crispasr-reuse-cuda)
    add_executable(reuse-profile-cuda "${REUSE_SOURCE_ROOT}/tests/test-reuse-frontend-profile.cpp")
    target_include_directories(reuse-profile-cuda PRIVATE "${CRISPASR_REUSE_MKL_INCLUDE_DIR}")
    target_link_libraries(reuse-profile-cuda PRIVATE crispasr-reuse-cuda)
    add_test(NAME reuse-profile-cuda COMMAND reuse-profile-cuda)
endif()
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    foreach(control converter network_compare frontend_stability reference_dumper phase_stability chunked quality_metrics)
        add_test(NAME reuse-${control}-controls COMMAND ${Python3_EXECUTABLE}
            "${REUSE_SOURCE_ROOT}/tests/test_reuse_${control}.py")
        set_tests_properties(reuse-${control}-controls PROPERTIES
            LABELS "unit;reuse;research" SKIP_RETURN_CODE 77 TIMEOUT 60)
    endforeach()
    if(TARGET reuse-frontend-cuda-probe)
        add_test(NAME reuse-cuda-probe-controls
            COMMAND ${CMAKE_COMMAND} -E env "REUSE_CUDA_FRONTEND_PROBE=$<TARGET_FILE:reuse-frontend-cuda-probe>"
                "REUSE_CUDA_PROFILE_CONTROL=$<TARGET_FILE:reuse-profile-cuda>"
                ${Python3_EXECUTABLE} "${REUSE_SOURCE_ROOT}/tests/test_reuse_cuda_frontend.py")
    else()
        add_test(NAME reuse-cuda-dependency-controls
            COMMAND ${Python3_EXECUTABLE} "${REUSE_SOURCE_ROOT}/tests/test_reuse_cuda_frontend.py")
    endif()
    if(TARGET reuse-frontend-mkl-probe)
        add_test(NAME reuse-mkl-probe-controls
            COMMAND ${CMAKE_COMMAND} -E env "REUSE_MKL_FRONTEND_PROBE=$<TARGET_FILE:reuse-frontend-mkl-probe>"
                "REUSE_MKL_PROFILE_CONTROL=$<TARGET_FILE:reuse-profile-mkl>"
                ${Python3_EXECUTABLE} "${REUSE_SOURCE_ROOT}/tests/test_reuse_mkl_probe.py")
    else()
        add_test(NAME reuse-mkl-dependency-controls
            COMMAND ${Python3_EXECUTABLE} "${REUSE_SOURCE_ROOT}/tests/test_reuse_mkl_probe.py")
    endif()
endif()
