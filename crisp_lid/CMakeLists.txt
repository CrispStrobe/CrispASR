# crisp_lid — shared C++ text-based language identification library.
#
# Two backends: lid-fasttext (GlotLID/LID-176) and lid-cld3 (Google CLD3).
# Auto-detects from GGUF `general.architecture` key.
#
# Used by CrispASR (post-ASR LID) and CrispEmbed (OCR language routing).

cmake_minimum_required(VERSION 3.14)

if(TARGET crispasr-core)
    set(_crisp_lid_core_target crispasr-core)
    set(_crisp_lid_core_inc    ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    set(_crisp_lid_defs        CRISPASR_BUILD)
elseif(TARGET crispembed-core)
    set(_crisp_lid_core_target crispembed-core)
    set(_crisp_lid_core_inc "")
    set(_crisp_lid_defs "")
else()
    message(STATUS "crisp_lid: no core target — skipping")
    return()
endif()

add_library(crisp_lid STATIC
    src/lid_fasttext.cpp
    src/lid_cld3.cpp
    src/text_lid_dispatch.cpp
)

target_include_directories(crisp_lid
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(_crisp_lid_core_inc)
    target_include_directories(crisp_lid PRIVATE ${_crisp_lid_core_inc})
endif()
if(_crisp_lid_defs)
    target_compile_definitions(crisp_lid PRIVATE ${_crisp_lid_defs})
endif()

target_link_libraries(crisp_lid
    PUBLIC  ${_crisp_lid_core_target} ggml ggml-base
)

target_compile_features(crisp_lid PUBLIC cxx_std_17)
set_target_properties(crisp_lid PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
    target_compile_options(crisp_lid PRIVATE -O3)
endif()
