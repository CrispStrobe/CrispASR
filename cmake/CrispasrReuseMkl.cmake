# Optional dependency for RE-USE research tools, never a product backend.
option(CRISPASR_REUSE_MKL_FRONTEND "Enable the experimental oneMKL CPU frontend in RE-USE research tools" OFF)
if(NOT CRISPASR_REUSE_MKL_FRONTEND)
    return()
endif()

find_path(CRISPASR_REUSE_MKL_INCLUDE_DIR mkl.h HINTS "$ENV{MKLROOT}" PATH_SUFFIXES include)
find_library(CRISPASR_REUSE_MKL_LIBRARY NAMES mkl_rt HINTS "$ENV{MKLROOT}" PATH_SUFFIXES lib lib/intel64)
if(NOT EXISTS "${CRISPASR_REUSE_MKL_INCLUDE_DIR}/mkl.h" OR NOT EXISTS "${CRISPASR_REUSE_MKL_LIBRARY}")
    message(FATAL_ERROR "CRISPASR_REUSE_MKL_FRONTEND=ON requires an external oneMKL SDK (mkl.h and mkl_rt). Set MKLROOT or CRISPASR_REUSE_MKL_INCLUDE_DIR/CRISPASR_REUSE_MKL_LIBRARY; disable the option for a dependency-free default build.")
endif()

get_filename_component(_reuse_source_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(crispasr-reuse-mkl STATIC "${_reuse_source_root}/tools/ci-heavy/issue579_reuse/frontend_mkl.cpp")
target_compile_features(crispasr-reuse-mkl PUBLIC cxx_std_17)
target_compile_definitions(crispasr-reuse-mkl PUBLIC CRISPASR_REUSE_HAVE_MKL=1)
target_include_directories(crispasr-reuse-mkl PRIVATE "${CRISPASR_REUSE_MKL_INCLUDE_DIR}")
target_include_directories(crispasr-reuse-mkl PUBLIC "${_reuse_source_root}/src")
target_link_libraries(crispasr-reuse-mkl PUBLIC "${CRISPASR_REUSE_MKL_LIBRARY}")
set_target_properties(crispasr-reuse-mkl PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(reuse-frontend-mkl-probe "${_reuse_source_root}/tools/ci-heavy/issue579_reuse/frontend_mkl_probe.cpp")
target_link_libraries(reuse-frontend-mkl-probe PRIVATE crispasr-reuse-mkl)
