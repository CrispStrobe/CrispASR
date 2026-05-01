#
# crispasr — unified ASR CLI (formerly crispasr).
#
# Drives all model backends in src/ (crispasr, parakeet, canary, cohere,
# qwen3, voxtral, voxtral4b, granite) from a single binary via the
# --backend flag. When --backend is omitted or set to "whisper" the
# historical crispasr code path is used unchanged.
#
# The CMake target is named `crispasr-cli`, but the shipped binary stays
# `crispasr` via OUTPUT_NAME for user-facing compatibility.

set(TARGET crispasr-cli)

add_executable(${TARGET}
    cli.cpp
    crispasr_backend.cpp
    crispasr_backend_crispasr.cpp
    crispasr_backend_parakeet.cpp
    crispasr_backend_canary.cpp
    crispasr_backend_cohere.cpp
    crispasr_backend_granite.cpp
    crispasr_backend_granite_nle.cpp
    crispasr_backend_voxtral.cpp
    crispasr_backend_voxtral4b.cpp
    crispasr_backend_qwen3.cpp
    crispasr_backend_fastconformer_ctc.cpp
    crispasr_backend_wav2vec2.cpp
    crispasr_backend_vibevoice.cpp
    crispasr_backend_qwen3_tts.cpp
    crispasr_backend_kokoro.cpp
    crispasr_backend_glm_asr.cpp
    crispasr_backend_kyutai_stt.cpp
    crispasr_backend_firered_asr.cpp
    crispasr_backend_moonshine.cpp
    crispasr_backend_moonshine_streaming.cpp
    crispasr_backend_gemma4_e2b.cpp
    crispasr_backend_omniasr.cpp
    crispasr_aligner_cli.cpp
    crispasr_vad_cli.cpp
    crispasr_lid_cli.cpp
    crispasr_diarize_cli.cpp
    crispasr_output.cpp
    crispasr_model_mgr_cli.cpp
    crispasr_diagnostics.cpp
    crispasr_run.cpp
    crispasr_server.cpp
)

# ---------------------------------------------------------------------------
# Build-info macros surfaced to crispasr_diagnostics.cpp. Together they
# answer "what exact bits am I running?" — important for bug reports
# from prebuilt Docker tags where the user has no source tree handy.
#
# `git-vars.cmake` populates GIT_SHA1 / GIT_DATE / GIT_COMMIT_SUBJECT.
# Re-include defensively in case a downstream consumer of this file
# imports it without the top-level CMakeLists having run yet.
# ---------------------------------------------------------------------------
if (NOT DEFINED GIT_SHA1)
    include(${CMAKE_SOURCE_DIR}/cmake/git-vars.cmake)
endif()
string(TIMESTAMP CRISPASR_BUILD_DATE_STR "%Y-%m-%dT%H:%M:%SZ" UTC)
target_compile_definitions(${TARGET} PRIVATE
    CRISPASR_VERSION_STR="${PROJECT_VERSION}"
    CRISPASR_GIT_SHA1="${GIT_SHA1}"
    CRISPASR_GIT_DATE="${GIT_DATE}"
    CRISPASR_GIT_SUBJECT="${GIT_COMMIT_SUBJECT}"
    CRISPASR_BUILD_DATE="${CRISPASR_BUILD_DATE_STR}"
    CRISPASR_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
)
if (GGML_CUDA)
    target_compile_definitions(${TARGET} PRIVATE
        CRISPASR_HAVE_CUDA=1
        CRISPASR_CUDA_VERSION="${CUDAToolkit_VERSION}"
        CRISPASR_CUDA_ARCHS="${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()
if (GGML_METAL)
    target_compile_definitions(${TARGET} PRIVATE CRISPASR_HAVE_METAL=1)
endif()
if (GGML_VULKAN)
    target_compile_definitions(${TARGET} PRIVATE CRISPASR_HAVE_VULKAN=1)
endif()
if (GGML_HIP)
    target_compile_definitions(${TARGET} PRIVATE CRISPASR_HAVE_HIP=1)
endif()
if (GGML_SYCL)
    target_compile_definitions(${TARGET} PRIVATE CRISPASR_HAVE_SYCL=1)
endif()
if (GGML_OPENCL)
    target_compile_definitions(${TARGET} PRIVATE CRISPASR_HAVE_OPENCL=1)
endif()
if (GGML_BLAS)
    target_compile_definitions(${TARGET} PRIVATE CRISPASR_HAVE_BLAS=1)
endif()
if (GGML_MUSA)
    target_compile_definitions(${TARGET} PRIVATE CRISPASR_HAVE_MUSA=1)
endif()

# ---------------------------------------------------------------------------
# crispasr-diff-lib — shared ground-truth diff harness (companion to
# tools/dump_reference.py).
#
# Exposed as a small static library so both the `crispasr-diff` binary
# and the older differential-test drivers under
# examples/{qwen3,voxtral}-test-*/ can link it without re-compiling the
# same sources. The library has no per-model dependencies — it only
# needs core_gguf from crispasr-core and the ggml CPU backend for its
# tensor loader.
# ---------------------------------------------------------------------------
add_library(crispasr-diff-lib STATIC
    crispasr_diff.cpp
    crispasr_diff.h
)
target_include_directories(crispasr-diff-lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(crispasr-diff-lib PUBLIC
    crispasr-core
    ggml
)
set_target_properties(crispasr-diff-lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# crispasr-diff — the CLI wrapper binary. Separate target so downstream
# users who don't care about differential testing don't pay the extra
# link time.
# ---------------------------------------------------------------------------
add_executable(crispasr-diff
    crispasr_diff_main.cpp
)
target_link_libraries(crispasr-diff PRIVATE
    common
    crispasr
    parakeet
    canary
    cohere
    granite_speech
    granite_nle
    voxtral
    voxtral4b
    qwen3_asr
    qwen3_tts
    kokoro
    gemma4_e2b
    mimo_tokenizer
    mimo_asr
    crispasr-diff-lib
    ${FFMPEG_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)
install(TARGETS crispasr-diff RUNTIME BUNDLE DESTINATION bin)

# ---------------------------------------------------------------------------
# mimo-tokenizer-smoke — finiteness/magnitude smoke test for the MiMo
# tokenizer encoder forward path. Used during PLAN #51 step 2 development
# while the upstream safetensors download is in flight.
# ---------------------------------------------------------------------------
add_executable(mimo-tokenizer-smoke mimo_tokenizer_smoke.cpp)
target_link_libraries(mimo-tokenizer-smoke PRIVATE
    common
    crispasr
    mimo_tokenizer
    ${FFMPEG_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME crispasr)

# httplib.h is in the server example directory
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/server)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE
    common
    crispasr
    parakeet
    canary
    cohere
    granite_speech
    granite_nle
    voxtral
    voxtral4b
    qwen3_asr
    qwen3_tts
    kokoro
    canary_ctc
    wav2vec2-ggml
    vibevoice
    silero-lid
    pyannote-seg
    ${FFMPEG_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    $<$<BOOL:${WIN32}>:winhttp>
    $<$<BOOL:${CRISPASR_CURL}>:CURL::libcurl>
)

if (CRISPASR_CURL)
    target_compile_definitions(${TARGET} PRIVATE CRISPASR_USE_LIBCURL)
endif()

install(TARGETS ${TARGET} RUNTIME BUNDLE DESTINATION bin)
