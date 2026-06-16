# crisp_truecase — shared C++ truecasing library.
#
# Three backends: statistical, CRF, BiLSTM.
# Used by CrispASR (post-ASR German truecasing) and CrispEmbed (post-OCR).

cmake_minimum_required(VERSION 3.14)

if(TARGET crispasr-core)
    set(_crisp_tc_core_target crispasr-core)
    set(_crisp_tc_core_inc    ${CMAKE_CURRENT_SOURCE_DIR}/../src)
elseif(TARGET crispembed-core)
    set(_crisp_tc_core_target crispembed-core)
    set(_crisp_tc_core_inc "")
else()
    message(STATUS "crisp_truecase: no core target — skipping")
    return()
endif()

add_library(crisp_truecase STATIC
    src/truecaser.cpp
    src/truecaser_crf.cpp
    src/truecaser_lstm.cpp
)

target_include_directories(crisp_truecase
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(_crisp_tc_core_inc)
    target_include_directories(crisp_truecase PRIVATE ${_crisp_tc_core_inc})
endif()

target_link_libraries(crisp_truecase
    PUBLIC  ${_crisp_tc_core_target}
)

target_compile_features(crisp_truecase PUBLIC cxx_std_17)
set_target_properties(crisp_truecase PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
    target_compile_options(crisp_truecase PRIVATE -O3)
endif()
