# crisp_punc — shared C++ punctuation restoration library.
#
# Two backends: FireRedPunc (BERT, Chinese+multilingual) and
# PCS (XLM-R, punctuation + capitalization + segmentation).
#
# Used by CrispASR (post-ASR cleanup) and CrispEmbed (post-OCR cleanup).
# Linked into CrispEmbed via add_subdirectory("../CrispASR/crisp_punc")
# from CrispEmbed's top-level CMakeLists.txt.

cmake_minimum_required(VERSION 3.14)

# crisp_punc links against the consumer's "core" target — which provides
# core_gguf::* symbols. CrispASR's crispasr-core and CrispEmbed's
# crispembed-core both supply these from identical sources.
if(TARGET crispasr-core)
    set(_crisp_punc_core_target crispasr-core)
    set(_crisp_punc_core_inc    ${CMAKE_CURRENT_SOURCE_DIR}/../src)
elseif(TARGET crispembed-core)
    set(_crisp_punc_core_target crispembed-core)
    set(_crisp_punc_core_inc "")
else()
    message(STATUS "crisp_punc: no core target (crispasr-core or crispembed-core) — skipping")
    return()
endif()

add_library(crisp_punc STATIC
    src/fireredpunc.cpp
    src/pcs.cpp
    include/crisp_punc.h
)

target_include_directories(crisp_punc
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(_crisp_punc_core_inc)
    target_include_directories(crisp_punc PRIVATE ${_crisp_punc_core_inc})
endif()

target_link_libraries(crisp_punc
    PUBLIC  ${_crisp_punc_core_target} ggml ggml-base
)

target_compile_features(crisp_punc PUBLIC cxx_std_17)
set_target_properties(crisp_punc PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
    target_compile_options(crisp_punc PRIVATE -O3)
endif()
