# dependencies

find_package(Threads REQUIRED)

# third-party

if (CRISPASR_SDL2)
    # SDL2
    find_package(SDL2 REQUIRED)

    string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

    message(STATUS "SDL2_INCLUDE_DIRS = ${SDL2_INCLUDE_DIRS}")
    message(STATUS "SDL2_LIBRARIES    = ${SDL2_LIBRARIES}")
endif()

# common

set(TARGET common)

unset(COMMON_EXTRA_LIBS)

if (CRISPASR_FFMPEG)
    find_package(FFmpeg REQUIRED)

    if (NOT ${FFMPEG_FOUND})
        message(FATAL_ERROR "Cannot find ffmpeg libs/headers")
    endif()

    message(STATUS "Found ffmpeg libs:       ${FFMPEG_LIBRARIES}")
    message(STATUS "Found ffmpeg headers in: ${FFMPEG_INCLUDE_DIRS}")
    message(STATUS "ffmpeg definitions:      ${FFMPEG_DEFINITIONS}")
    message(STATUS "Found avformat           ${AVFORMAT_VERSION}")

    include_directories(${FFMPEG_INCLUDE_DIRS})
    add_compile_definitions(CRISPASR_FFMPEG)

    list(APPEND COMMON_EXTRA_LIBS ${FFMPEG_LIBRARIES})

    set(COMMON_SOURCES_FFMPEG ffmpeg-transcode.cpp)
endif()


add_library(${TARGET} STATIC
    common.h
    common.cpp
    common-ggml.h
    common-ggml.cpp
    common-crispasr.h
    common-crispasr.cpp
    grammar-parser.h
    grammar-parser.cpp
    ${COMMON_SOURCES_FFMPEG}
    )

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE crispasr ${COMMON_EXTRA_LIBS} ${CMAKE_DL_LIBS})

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${TARGET} PROPERTIES FOLDER "libs")

if (CRISPASR_SDL2)
    # common-sdl

    set(TARGET common-sdl)

    add_library(${TARGET} STATIC
        common-sdl.h
        common-sdl.cpp
        )

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PUBLIC  ${SDL2_INCLUDE_DIRS})
    target_link_libraries     (${TARGET} PRIVATE ${SDL2_LIBRARIES})

    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

# add json lib
add_library(json_cpp INTERFACE)
target_include_directories(json_cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
    add_subdirectory(crispasr.wasm)
    add_subdirectory(stream.wasm)
    add_subdirectory(command.wasm)
    add_subdirectory(bench.wasm)
    add_subdirectory(wchess)
elseif(CMAKE_JS_VERSION)
    add_subdirectory(addon.node)
else()
    # ------------------------------------------------------------------
    # Unified CrispASR binary + support tools
    # ------------------------------------------------------------------
    add_subdirectory(cli)               # the `crispasr` binary
    add_subdirectory(bench)
    add_subdirectory(server)
    add_subdirectory(quantize)
    add_subdirectory(vad-speech-segments)
    add_subdirectory(crispasr-quantize) # GGUF re-quantize (was cohere-quantize)

    # Standalone forced-alignment CLIs (complementary to `crispasr -am`):
    #   cohere-align  wav2vec2 char-level CTC aligner (English only)
    #   nfa-align     canary_ctc multilingual subword CTC aligner
    add_subdirectory(cohere-align)
    add_subdirectory(nfa-align)

    # Differential test fixtures kept from the original per-model ports.
    # They load pre-dumped PyTorch tensors and diff against the ggml
    # forward pass, stage by stage. Useful for regression on future
    # architectural changes; not a user-facing feature.
    add_subdirectory(qwen3-asr-test-conv)
    add_subdirectory(qwen3-asr-test-llm)
    add_subdirectory(qwen3-asr-test-trace)
    add_subdirectory(qwen3-asr-test-bpe)
    add_subdirectory(voxtral-test-llm)
    add_subdirectory(voxtral-test-encoder)
    add_subdirectory(voxtral-test-e2e)
    add_subdirectory(vibevoice-test-stages)

    if (CRISPASR_SDL2)
        add_subdirectory(stream)
        add_subdirectory(command)
        add_subdirectory(talk-llama)
        add_subdirectory(lsp)
        if (GGML_SYCL)
            add_subdirectory(sycl)
        endif()
    endif (CRISPASR_SDL2)

    add_subdirectory(deprecation-warning)
endif()

if (CRISPASR_SDL2)
    add_subdirectory(wchess)
endif (CRISPASR_SDL2)
