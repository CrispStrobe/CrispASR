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

add_executable(reuse-research-diff diff_probe.cpp
    "${REUSE_SOURCE_ROOT}/src/core/gguf_loader.cpp"
    "${REUSE_SOURCE_ROOT}/src/core/reuse_model_loader.cpp"
    "${REUSE_SOURCE_ROOT}/examples/cli/crispasr_diff.cpp")
target_include_directories(reuse-research-diff PRIVATE "${REUSE_SOURCE_ROOT}/src"
    "${REUSE_SOURCE_ROOT}/examples/cli" "${CRISPASR_REUSE_GGML_SOURCE_DIR}/src")
target_link_libraries(reuse-research-diff PRIVATE ggml ggml-cpu)
if(TARGET crispasr-reuse-mkl)
    target_link_libraries(reuse-research-diff PRIVATE crispasr-reuse-mkl)
endif()

enable_testing()
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
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    if(TARGET reuse-frontend-mkl-probe)
        add_test(NAME reuse-mkl-probe-controls
            COMMAND ${CMAKE_COMMAND} -E env "REUSE_MKL_FRONTEND_PROBE=$<TARGET_FILE:reuse-frontend-mkl-probe>"
                ${Python3_EXECUTABLE} "${REUSE_SOURCE_ROOT}/tests/test_reuse_mkl_probe.py")
    else()
        add_test(NAME reuse-mkl-dependency-controls
            COMMAND ${Python3_EXECUTABLE} "${REUSE_SOURCE_ROOT}/tests/test_reuse_mkl_probe.py")
    endif()
endif()
