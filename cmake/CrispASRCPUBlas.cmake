# Shared host-kernel CBLAS capability. A library without its C header (or only
# Fortran BLAS symbols) must not enable a source-level fast path.
# crispasr_prefer_cblas(), defined by src/CMakeLists.txt, retains the project's
# explicit MKL selection and sequential-MKL compatibility policy.
if(NOT DEFINED CRISPASR_MEL_BLAS)
    set(_crispasr_cpu_blas_default ON)
else()
    set(_crispasr_cpu_blas_default "${CRISPASR_MEL_BLAS}")
endif()
option(CRISPASR_CPU_BLAS "Use CBLAS in backend host kernels and inverse STFT" ${_crispasr_cpu_blas_default})
unset(_crispasr_cpu_blas_default)

add_library(crispasr-cpu-blas INTERFACE)
set(CRISPASR_CPU_BLAS_AVAILABLE OFF)
set_property(TARGET crispasr-cpu-blas PROPERTY CRISPASR_CPU_BLAS_AVAILABLE OFF)

function(crispasr_configure_cpu_blas)
    if(NOT CRISPASR_CPU_BLAS)
        return()
    endif()
    if(APPLE)
        find_library(CRISPASR_CPU_ACCELERATE Accelerate REQUIRED)
        target_link_libraries(crispasr-cpu-blas INTERFACE ${CRISPASR_CPU_ACCELERATE})
        target_compile_definitions(crispasr-cpu-blas INTERFACE
            HAVE_ACCELERATE HAVE_BLAS CRISPASR_HAVE_BLAS=1 ACCELERATE_NEW_LAPACK)
        set(CRISPASR_CPU_BLAS_AVAILABLE ON PARENT_SCOPE)
        set_property(TARGET crispasr-cpu-blas PROPERTY CRISPASR_CPU_BLAS_AVAILABLE ON)
        return()
    endif()

    find_path(CRISPASR_CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
    if(NOT COHERE_MKL)
        find_library(CRISPASR_CPU_OPENBLAS NAMES openblas openblasp openblaso)
    endif()
    if(COHERE_MKL)
        # The root project sets BLA_VENDOR=Intel10_64lp for this explicit
        # choice. Do not let an unrelated standalone libcblas bypass it.
        find_package(BLAS QUIET)
    elseif(CRISPASR_CPU_OPENBLAS)
        set(BLAS_FOUND ON)
        set(BLAS_LIBRARIES ${CRISPASR_CPU_OPENBLAS})
    else()
        crispasr_prefer_cblas(QUIET)
    endif()
    if(BLAS_FOUND AND CRISPASR_CBLAS_INCLUDE_DIR)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_INCLUDES ${CRISPASR_CBLAS_INCLUDE_DIR})
        set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
        check_cxx_source_compiles("
            #include <cblas.h>
            int main() {
                float a = 2, b = 3, c = 0;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            1, 1, 1, 1, &a, 1, &b, 1, 0, &c, 1);
                return c == 6 ? 0 : 1;
            }" CRISPASR_CBLAS_LINKS)
        if(CRISPASR_CBLAS_LINKS)
            target_include_directories(crispasr-cpu-blas INTERFACE ${CRISPASR_CBLAS_INCLUDE_DIR})
            target_link_libraries(crispasr-cpu-blas INTERFACE ${BLAS_LIBRARIES})
            target_compile_definitions(crispasr-cpu-blas INTERFACE HAVE_BLAS CRISPASR_HAVE_BLAS=1)
            set(CRISPASR_CPU_BLAS_AVAILABLE ON PARENT_SCOPE)
            set_property(TARGET crispasr-cpu-blas PROPERTY CRISPASR_CPU_BLAS_AVAILABLE ON)
            message(STATUS "Backend host kernels: CBLAS enabled (${BLAS_LIBRARIES})")
            return()
        endif()
    endif()
    if(COHERE_MKL)
        message(FATAL_ERROR "COHERE_MKL=ON but the backend CBLAS header/symbol check failed")
    endif()
    message(STATUS "Backend host kernels: CBLAS header/symbols unavailable; using scalar kernels")
endfunction()

crispasr_configure_cpu_blas()
