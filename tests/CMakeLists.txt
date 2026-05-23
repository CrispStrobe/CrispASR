set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Catch2 — unit - test framework(pytest analogue) ──────────────────────────
#Fetched at configure time; pinned to a specific tag for reproducibility.
#The resulting targets(Catch2::Catch2WithMain, Catch2::Catch2) are then
#available for any test in this directory.
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)

if (EMSCRIPTEN)
#
#test - crispasr - js

    set(TEST_TARGET test-crispasr-js)

    add_test(NAME ${TEST_TARGET}
        COMMAND node test-crispasr.js --experimental-wasm-threads
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

    return()
endif()

set(TEST_TARGET test-crispasr-cli-tiny)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "tiny;gh")

set(TEST_TARGET test-crispasr-cli-tiny.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "tiny;en")

set(TEST_TARGET test-crispasr-cli-base)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-base.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "base")

set(TEST_TARGET test-crispasr-cli-base.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-base.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "base;en")

set(TEST_TARGET test-crispasr-cli-small)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-small.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "small")

set(TEST_TARGET test-crispasr-cli-small.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-small.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "small;en")

set(TEST_TARGET test-crispasr-cli-medium)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-medium.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "medium")

set(TEST_TARGET test-crispasr-cli-medium.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-medium.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "medium;en")

set(TEST_TARGET test-crispasr-cli-large)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-large.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "large")

if (CRISPASR_FFMPEG)
    set(TEST_TARGET test-crispasr-cli-tiny-mp3)
#Check with reviewers : any way to check the output transcription via ctest(diff, ...) ?
    add_test(NAME ${TEST_TARGET}
      COMMAND $<TARGET_FILE:crispasr-cli>
      -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.en.bin
      -f ${PROJECT_SOURCE_DIR}/samples/jfk.mp3)
    set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "tiny;mp3")
endif()

#VAD test tests VAD in isolation
set(VAD_TEST test-vad)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../ggml/include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common)
target_compile_definitions(${VAD_TEST} PRIVATE
    VAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TEST} PROPERTIES LABELS "unit")

#VAD test full uses crispasr_full with VAD enabled
set(VAD_TEST test-vad-full)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../ggml/include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common)
target_compile_definitions(${VAD_TEST} PRIVATE
    CRISPASR_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/ggml-tiny.en.bin"
    VAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TEST} PROPERTIES LABELS "base;en")

# ─── test - vad - thresholds — Verify VAD behavior with different thresholds ────────
set(VAD_TH_TEST test-vad-thresholds)
add_executable(${VAD_TH_TEST} ${VAD_TH_TEST}.cpp)
target_include_directories(${VAD_TH_TEST} PRIVATE ../include ../ggml/include ../examples)
target_link_libraries(${VAD_TH_TEST} PRIVATE common)
target_compile_definitions(${VAD_TH_TEST} PRIVATE
    CRISPASR_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/ggml-tiny.en.bin"
    VAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${VAD_TH_TEST} COMMAND ${VAD_TH_TEST})
set_tests_properties(${VAD_TH_TEST} PROPERTIES LABELS "base;en")

# ─── test - crispasr - cache — Catch2 unit tests for download / cache helpers ───────
#
#Compiles crispasr_cache.cpp directly into the test binary so the test is
#self - contained and does not depend on the full crispasr link closure.
#Tests are pure filesystem operations — no network requests are made.
add_executable(test-crispasr-cache
    test-crispasr-cache.cpp
    ${PROJECT_SOURCE_DIR}/src/crispasr_cache.cpp
)
target_include_directories(test-crispasr-cache PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-crispasr-cache PRIVATE
    Catch2::Catch2WithMain
    $<$<BOOL:${WIN32}>:winhttp>
)
if (CRISPASR_CURL)
    target_compile_definitions(test-crispasr-cache PRIVATE CRISPASR_USE_LIBCURL)
    target_link_libraries(test-crispasr-cache PRIVATE CURL::libcurl)
endif()
#catch_discover_tests registers one CTest entry per Catch2 TEST_CASE.
catch_discover_tests(test-crispasr-cache
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test - params — Catch2 unit tests for crispasr_params struct defaults ────────
#
#Default - constructs crispasr_params and checks every crispasr - specific field
#carries the right initial value.Links crispasr + common because
#crispasr_params inlines calls to crispasr_full_default_params() in its field
#initialisers.
add_executable(test-params
    test-params.cpp
)
target_include_directories(test-params PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/examples
    ${PROJECT_SOURCE_DIR}/examples/cli
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-params PRIVATE
    Catch2::Catch2WithMain
    crispasr
    common
)
catch_discover_tests(test-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test - stream - finalize — Catch2 unit tests for the streaming
# `final.text` stitcher used by `--stream-json` finalize_utterance.
# Header-only target (no crispasr lib needed): the helper is a pure
# string function in examples/cli/crispasr_stream_finalize.h.
add_executable(test-stream-finalize
    test-stream-finalize.cpp
)
target_include_directories(test-stream-finalize PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test-stream-finalize PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-stream-finalize
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;regression-84"
)

# Pre-decode finalized-slice skip added in PR #90 (stream JSON mode).
# Pure arithmetic, no model load.
add_executable(test-stream-vad-skip
    test-stream-vad-skip.cpp
)
target_link_libraries(test-stream-vad-skip PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-stream-vad-skip
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;stream-json"
)

# JSON streaming VAD post-merge policy. Pure arithmetic, no model load.
add_executable(test-stream-vad-merge-policy
    test-stream-vad-merge-policy.cpp
)
target_include_directories(test-stream-vad-merge-policy PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-stream-vad-merge-policy PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-stream-vad-merge-policy
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;stream-json"
)

# JSON streaming punctuation policy (PR #112). Pure string-to-bool
# predicate covered exhaustively; no model load, no audio.
add_executable(test-stream-punc
    test-stream-punc.cpp
)
target_include_directories(test-stream-punc PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test-stream-punc PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-stream-punc
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;stream-json"
)

# JSON streaming partial-decode cadence (PR #113). Pure arithmetic on
# the throttle predicate + interval conversion; no model, no audio.
add_executable(test-stream-partial-decode
    test-stream-partial-decode.cpp
)
target_include_directories(test-stream-partial-decode PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test-stream-partial-decode PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-stream-partial-decode
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;stream-json"
)

# Pyannote-seg → ASR-segment speaker scoring (issue #107). Drives the
# internal helper with synthetic log-softmax posteriors; no model, no
# audio. Links against libcrispasr so it tests the actually-shipped
# logic rather than a fork of it.
add_executable(test-diarize-pyannote
    test-diarize-pyannote.cpp
)
target_include_directories(test-diarize-pyannote PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-diarize-pyannote PRIVATE
    Catch2::Catch2WithMain
    crispasr
)
catch_discover_tests(test-diarize-pyannote
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;diarize"
)

# Live end-to-end checks that --diarize-method pyannote (and optional
# --diarize-embedder) produce sensible labels on a real multi-speaker
# WAV. Opt-in via env vars; skips cleanly otherwise. Compiles the
# CLI-side cluster helper directly so the embedder variant doesn't
# need to link the whole crispasr-cli executable.
add_executable(test-diarize-pyannote-live
    test-diarize-pyannote-live.cpp
)
target_include_directories(test-diarize-pyannote-live PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-diarize-pyannote-live PRIVATE
    Catch2::Catch2WithMain
    crispasr
)
catch_discover_tests(test-diarize-pyannote-live
    TEST_SPEC "[live]"
    PROPERTIES LABELS "live;diarize"
)

# Agglomerative cosine clustering used by --diarize-embedder (#107 P3b).
# Pure arithmetic on synthetic embeddings; no model, no audio.
add_executable(test-speaker-cluster
    test-speaker-cluster.cpp
)
target_include_directories(test-speaker-cluster PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-speaker-cluster PRIVATE
    Catch2::Catch2WithMain
    crispasr
)
catch_discover_tests(test-speaker-cluster
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;diarize"
)

# ─── test - punctuation - policy — Catch2 unit tests for punctuation auto - enable
#policy.Pure logic, no model load.
add_executable(test-punctuation-policy
    test_punctuation_policy.cpp
)
target_include_directories(test-punctuation-policy PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/examples
    ${PROJECT_SOURCE_DIR}/examples/cli
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-punctuation-policy PRIVATE
    Catch2::Catch2WithMain
    common
)
catch_discover_tests(test-punctuation-policy
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test - diarize - global — Catch2 unit tests for global diarization logic (#110)
add_executable(test-diarize-global
    test_diarize_global.cpp
)
target_include_directories(test-diarize-global PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test-diarize-global PRIVATE
    Catch2::Catch2WithMain
    common
)
catch_discover_tests(test-diarize-global
    TEST_SPEC "[diarize]"
    PROPERTIES LABELS "unit;diarize"
)

# ─── test - paraformer — live integration tests for paraformer backend
add_executable(test-paraformer
    test_paraformer.cpp
)
target_include_directories(test-paraformer PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples
)
target_link_libraries(test-paraformer PRIVATE
    Catch2::Catch2WithMain
    paraformer
)
catch_discover_tests(test-paraformer
    TEST_SPEC "[paraformer]"
    PROPERTIES LABELS "live;paraformer"
)

# ─── test - context - bias — Catch2 unit tests for CTC-WS phrase-boost trie (PLAN #98)
add_executable(test-context-bias
    test_context_bias.cpp
)
target_include_directories(test-context-bias PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-context-bias PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-context-bias
    TEST_SPEC "[context_bias]"
    PROPERTIES LABELS "unit"
)

# ─── test - windows - mic — regression test for Windows dshow mic selection
add_executable(test-windows-mic
    test_windows_mic.cpp
)
target_include_directories(test-windows-mic PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test-windows-mic PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-windows-mic
    TEST_SPEC "[windows][mic]"
    PROPERTIES LABELS "unit;windows"
)

# ─── test - chatterbox - params — Catch2 unit tests for chatterbox_context_params defaults ────────
add_executable(test-chatterbox-params
    test-chatterbox-params.cpp
)
target_include_directories(test-chatterbox-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-chatterbox-params PRIVATE
    Catch2::Catch2WithMain
    chatterbox
)
catch_discover_tests(test-chatterbox-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;chatterbox"
)

# ─── test-qwen3-tts-cv-style — Catch2 unit tests for qwen3_tts_set_cv_style_instruct
# (issue #91: --instruct silently ignored for CustomVoice models).
# Exercises null-ctx guards and default-params sanity — no GGUF needed. ─────
add_executable(test-qwen3-tts-cv-style
    test-qwen3-tts-cv-style.cpp
)
target_include_directories(test-qwen3-tts-cv-style PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-qwen3-tts-cv-style PRIVATE
    Catch2::Catch2WithMain
    qwen3_tts
)
catch_discover_tests(test-qwen3-tts-cv-style
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;qwen3-tts"
)

# ─── test - flash - attn - defaults — Catch2 unit tests for the PLAN #89
# flash_attn-field plumbing across every backend with use_gpu.
# Auto-skips backends not built into the current libcrispasr slice
# (each section is guarded by `__has_include("backend.h")`), so the
# test runs on every CMake configuration without hand-tuning. ────────
add_executable(test-flash-attn-defaults
    test-flash-attn-defaults.cpp
)
target_include_directories(test-flash-attn-defaults PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
# Link every backend whose header we touch. CMake auto-skips
# missing targets via the if(TARGET …) guards so a slim build
# (e.g. iOS device-only with fewer backends) still configures.
set(_flash_attn_test_libs Catch2::Catch2WithMain)
foreach(_be parakeet canary cohere qwen3_asr voxtral voxtral4b
             granite_speech vibevoice qwen3_tts orpheus kokoro
             chatterbox)
    if (TARGET ${_be})
        list(APPEND _flash_attn_test_libs ${_be})
    endif()
endforeach()
target_link_libraries(test-flash-attn-defaults PRIVATE ${_flash_attn_test_libs})
unset(_flash_attn_test_libs)
catch_discover_tests(test-flash-attn-defaults
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;flash_attn"
)

# ─── test - metal - pipeline - cache — Apple-only smoke test for the
# persistent MTLBinaryArchive cache (PLAN #88 / CrisperWeaver §5.18).
# Compiled with -ObjC++ when Metal is enabled; on Linux / Windows
# the .mm body collapses to a single "skipped" TEST_CASE so the
# binary still builds and registers cleanly. ────────
if (APPLE AND GGML_METAL)
    add_executable(test-metal-pipeline-cache
        test-metal-pipeline-cache.mm
    )
    target_include_directories(test-metal-pipeline-cache PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal
    )
    target_link_libraries(test-metal-pipeline-cache PRIVATE
        Catch2::Catch2WithMain
        ggml
    )
    # Match the host C++ standard the rest of the tests use (the
    # .mm file mixes Catch2 templates + filesystem with Obj-C
    # frameworks, needs C++17 minimum for <filesystem>).
    set_target_properties(test-metal-pipeline-cache PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    catch_discover_tests(test-metal-pipeline-cache
        TEST_SPEC "[unit]"
        PROPERTIES LABELS "unit;metal;pipeline-cache"
    )
endif()

# ─── test_chunk_context — Catch2 unit tests for the energy-minima chunker (issue #89)
#
# Covers: audio_chunking::split_at_energy_minima slice coverage / contiguity
# / single-slice cases. The ctx_l/ctx_r expansion and word-trim that the
# original 617cd02 round added were reverted (see issue #89 round 2); the
# tests that pinned them are gone with the code. Pure CPU, no model load.
add_executable(test_chunk_context
    test_chunk_context.cpp
)
target_include_directories(test_chunk_context PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test_chunk_context PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test_chunk_context
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;chunk-context"
)

# ─── test-issue-114-chunk-context-gate — issue #114 regression guard
#
# Pins crispasr_chunk_context::should_use_chunk_context so a future
# refactor cannot silently re-introduce the cad4c28a gate that
# extended VAD-derived slices with neighbour audio and produced
# kanji → hiragana on parakeet-tdt-0.6b-ja. Pure CPU, no model load.
add_executable(test-issue-114-chunk-context-gate
    test-issue-114-chunk-context-gate.cpp
)
target_include_directories(test-issue-114-chunk-context-gate PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test-issue-114-chunk-context-gate PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-issue-114-chunk-context-gate
    TEST_SPEC "[issue-114]"
    PROPERTIES LABELS "unit;chunk-context;issue-114"
)

# ─── test-issue-89-long-audio-fallback — issue #89 regression guard
#
# Pins crispasr_long_audio::should_auto_chunk_long so a future refactor
# cannot silently restore the a069018f "always full-encode for
# CAP_UNBOUNDED_INPUT" behaviour. That produced 35 tokens of output on
# 300 s of audio when the user provided neither --vad nor --chunk-seconds
# because the FastConformer + TDT decoder collapse past ~30-60 s in a
# single pass.
add_executable(test-issue-89-long-audio-fallback
    test-issue-89-long-audio-fallback.cpp
)
target_include_directories(test-issue-89-long-audio-fallback PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test-issue-89-long-audio-fallback PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-issue-89-long-audio-fallback
    TEST_SPEC "[issue-89]"
    PROPERTIES LABELS "unit;long-audio;issue-89"
)

# ─── test-internal-chunking-gate — CAP_INTERNAL_CHUNKING regression guard
#
# Pins the auto-chunk fallback skip for backends that declare
# CAP_INTERNAL_CHUNKING (parakeet, canary, fastconformer-ctc).
# These backends handle their own long-audio chunking internally via
# parakeet_transcribe_streamed or single-pass encoding.
add_executable(test-internal-chunking-gate
    test-internal-chunking-gate.cpp
)
target_include_directories(test-internal-chunking-gate PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test-internal-chunking-gate PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-internal-chunking-gate
    TEST_SPEC "[internal-chunking]"
    PROPERTIES LABELS "unit;internal-chunking;issue-89"
)

# ─── test-lcs-chunk-merge — Catch2 unit tests for LCS chunk-merge dedup
#
# Ports of NeMo's `longest_common_subsequence_merge` test cases
# (`nemo/collections/asr/parts/utils/streaming_utils.py`). Pin the
# algorithm so subsequent refactors keep behavioural parity with the
# upstream BatchedFrameASRTDT hypothesis stitcher.
add_executable(test-lcs-chunk-merge
    test-lcs-chunk-merge.cpp
)
target_include_directories(test-lcs-chunk-merge PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-lcs-chunk-merge PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-lcs-chunk-merge
    TEST_SPEC "[lcs]"
    PROPERTIES LABELS "unit;lcs"
)

# ─── test-lcs-dedup-driver — Catch2 unit tests for the segment-aware
# driver that wraps the LCS algorithm with token / word / text book-
# keeping. Together with test-lcs-chunk-merge this gives end-to-end
# coverage of the chunk-boundary dedup feature.
add_executable(test-lcs-dedup-driver
    test-lcs-dedup-driver.cpp
)
target_include_directories(test-lcs-dedup-driver PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-lcs-dedup-driver PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-lcs-dedup-driver
    TEST_SPEC "[lcs-dedup]"
    PROPERTIES LABELS "unit;lcs;lcs-dedup"
)

# ─── test-lcs-c-abi — pin the public C ABI surface for the LCS dedup.
# Verifies that `crispasr_lcs_dedup_prefix_count` is exported from
# libcrispasr and agrees with the in-tree algorithm on the same
# fixtures. Catches regressions where the symbol gets dropped from the
# dynamic library (CMake link-order slip, missing CA_EXPORT).
add_executable(test-lcs-c-abi
    test-lcs-c-abi.cpp
)
target_include_directories(test-lcs-c-abi PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test-lcs-c-abi PRIVATE
    crispasr
    Catch2::Catch2WithMain
)
catch_discover_tests(test-lcs-c-abi
    TEST_SPEC "[c-abi]"
    PROPERTIES LABELS "unit;lcs;c-abi"
)

# ─── test - math - consistency — Catch2 unit tests for math constants ────────
add_executable(test-math-consistency
    test-math-consistency.cpp
)
target_link_libraries(test-math-consistency PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-math-consistency
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test - link - odr — Verify no ODR violations when linking multiple backends ────────
add_executable(test-link-odr
    test-link-odr.cpp
)
target_include_directories(test-link-odr PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-link-odr PRIVATE
    Catch2::Catch2WithMain
    chatterbox
    vibevoice
)
catch_discover_tests(test-link-odr
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test - issue - 79 — Regression test for VibeVoice ASR KV cache resize ────────
add_executable(test-issue-79
    test-issue-79.cpp
)
target_include_directories(test-issue-79 PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-issue-79 PRIVATE
    vibevoice
)
#Note : No add_test() here as it requires a large VibeVoice GGUF model path as an argument.
#Usage :./ bin / test - issue - 79 / path / to / vibevoice - asr - 7b.gguf

# ─── test-session-setters — Catch2 unit tests for every generation-control
# session setter declared in include/crispasr.h (PLAN §88 parity sweep).
#
# Verifies the null-handle guard (return -1) for all 19 setters.
# No model loaded — pure null-arg path, sub-millisecond.
add_executable(test-session-setters
    test-session-setters.cpp
)
target_include_directories(test-session-setters PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/include
)
target_link_libraries(test-session-setters PRIVATE
    Catch2::Catch2WithMain
    crispasr
)
catch_discover_tests(test-session-setters
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test - registry — Catch2 unit tests for crispasr_model_registry
#
#Verifies registry lookup by backend name, reverse lookup by filename,
#and that all expected entries exist.No network, no models.
add_executable(test-registry test-registry.cpp
    ${PROJECT_SOURCE_DIR}/src/crispasr_model_registry.cpp
    ${PROJECT_SOURCE_DIR}/src/crispasr_cache.cpp
)
target_include_directories(test-registry PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-registry PRIVATE
    Catch2::Catch2WithMain
)
if(WIN32)
#crispasr_cache.cpp uses WinHTTP for model downloads
    target_link_libraries(test-registry PRIVATE winhttp)
endif()
catch_discover_tests(test-registry
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test_server_wav_writer — Catch2 unit tests for crispasr_make_wav_int16
#
#Header - only WAV serializer used by / v1 / audio / speech and crispasr_diff.
#Pure CPU, no model dependencies — sub - millisecond per case.Pins the
#RIFF byte layout so OpenAI clients keep working.
add_executable(test_server_wav_writer test_server_wav_writer.cpp)
target_include_directories(test_server_wav_writer PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test_server_wav_writer PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test_server_wav_writer
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test_server_chunking — Catch2 unit tests for the sentence splitter
#used by / v1 / audio / speech long - form input(PLAN §75d / issue #66).
#Pure functions, no backend dependency.
add_executable(test_server_chunking
    test_server_chunking.cpp
    ${PROJECT_SOURCE_DIR}/examples/cli/crispasr_tts_chunking.cpp
)
target_include_directories(test_server_chunking PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test_server_chunking PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test_server_chunking
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test_vibevoice_wav_ref — Catch2 unit tests for VibeVoice voice cloning
#WAV reference parsing and normalization.Pure helper tests, no model load.
add_executable(test_vibevoice_wav_ref test_vibevoice_wav_ref.cpp)
target_include_directories(test_vibevoice_wav_ref PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test_vibevoice_wav_ref PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test_vibevoice_wav_ref
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test - core - decode — Catch2 unit tests for core_greedy_decode + core_beam_decode
#
#Tests the shared decode helpers in isolation with a mock LLM.No models
#loaded, no GPU — pure CPU, sub - millisecond.Covers : argmax, softmax_of,
#greedy run_with_probs, beam run_with_probs(beam_size = 1, 4), max_new_tokens
#cap, multi-EOS termination, token-id invariant (PR #97).
add_executable(test-core-decode test-core-decode.cpp)
target_include_directories(test-core-decode PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-core-decode PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-core-decode
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test-audio-chunking — Catch2 unit tests for audio_chunking::find_energy_min_split
#
# Covers the energy-minimum detection helper directly: quiet window selection,
# tie-breaking, search range, degenerate spans.  Complements test_chunk_context
# which tests the higher-level split_at_energy_minima wrapper.
# Pure CPU, no model load, sub-millisecond.
add_executable(test-audio-chunking test-audio-chunking.cpp)
target_include_directories(test-audio-chunking PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-audio-chunking PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-audio-chunking
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;chunking"
)

# ─── test_slice_timestamps — Catch2 unit tests for crispasr_fixed_chunk_slices
# t0_cs / t1_cs timestamp math.  Pins the invariant that PR #96 depends on:
# each slice's t0_cs = start_sample / SR * 100, with no gaps between adjacent
# slices.  Links libcrispasr for crispasr_fixed_chunk_slices.  Pure CPU, no
# model load.
add_executable(test_slice_timestamps
    test_slice_timestamps.cpp
)
target_include_directories(test_slice_timestamps PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test_slice_timestamps PRIVATE
    Catch2::Catch2WithMain
    crispasr
)
catch_discover_tests(test_slice_timestamps
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;slice-ts"
)

# ─── test_qwen3_audio_tower — pre - refactor numeric baseline for the audio
#tower.Captures sum / sum_sq / max_abs + first - 32 / last - 32 windows on the encoder
#output for a fixed PCM(jfk.raw) so that extracting the audio tower into the
#CrispAudio shared library is verifiably non - disruptive.CPU - only — output
#must be deterministic across machines.
add_executable(test_qwen3_audio_tower test_qwen3_audio_tower.cpp)
target_include_directories(test_qwen3_audio_tower PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_qwen3_audio_tower PRIVATE qwen3_asr)

# ─── test_qwen3_tts_talker — sanity check the Qwen3 - TTS talker forward.
#Not auto - registered with add_test — needs a real GGUF on disk.
add_executable(test_qwen3_tts_talker test_qwen3_tts_talker.cpp)
target_include_directories(test_qwen3_tts_talker PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_qwen3_tts_talker PRIVATE qwen3_tts)
#Not auto - registered with add_test — needs a model file + jfk.raw which the
#test runner / dev produces locally.Run manually:
#./ build / tests / test_qwen3_audio_tower model.gguf samples / jfk.raw #> tests / fixtures / qwen3_audio_tower_jfk.txt
#./ build / tests / test_qwen3_audio_tower -- check model.gguf samples / jfk.raw
#> tests / fixtures / qwen3_audio_tower_jfk.txt

# ─── test-chat-ggml — end-to-end smoke for the crispasr_chat_* C ABI
#
# Loads a small GGUF chat model (env CRISPASR_CHAT_TEST_MODEL → path) and
# verifies open / generate (one-shot) / generate_stream / reset / close.
# Skipped when the env var is unset so unrelated builds stay green.
add_executable(test-chat-ggml test-chat-ggml.cpp)
target_link_libraries(test-chat-ggml PRIVATE
    Catch2::Catch2WithMain
    crispasr
)
catch_discover_tests(test-chat-ggml
    TEST_SPEC "[chat]"
    PROPERTIES LABELS "chat;integration"
)

# ─── test_metal_norm_repro — ggml_norm Metal vs CPU bit-exactness ────────────
# Standalone repro for the kernel_norm cross-simdgroup reduction bug
# (see tools/upstream-prs/08-metal-norm-cross-simdgroup). Sweeps a
# range of T values around the failure boundary and asserts CPU and
# Metal outputs match bit-exactly. Guards against ggml bumps reverting
# the patch.
if(APPLE)
    add_executable(test_metal_norm_repro test_metal_norm_repro.cpp)
    target_link_libraries(test_metal_norm_repro PRIVATE
        ggml
        ggml-base
        ggml-cpu
        ggml-metal
    )
    # T values that triggered the original bug; each must return rc=0.
    foreach(T 33 65 66 67 97 129 130 131 132 257)
        add_test(NAME metal_norm_T${T}
            COMMAND test_metal_norm_repro ${T} 512)
        set_tests_properties(metal_norm_T${T} PROPERTIES
            LABELS "unit;metal"
            TIMEOUT 30)
    endforeach()
endif()

# ─── test - backends — integration tests for all CrispASR backends ────────────
#Requires network(downloads models).Not run in CI by default.
#Usage : ctest - L integration-- output - on - failure
#Or :./ tests / test - backends.sh[backend...]
set(TEST_TARGET test-backends)
add_test(NAME ${TEST_TARGET}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-backends.sh
        parakeet moonshine wav2vec2 fastconformer-ctc data2vec hubert
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(${TEST_TARGET} PROPERTIES
    LABELS "integration"
    TIMEOUT 600)

# ─── benchmark-quick — ASR benchmark framework smoke test ───────────────────
# Runs the Python benchmark driver on JFK + parakeet (11s, fast). Verifies
# the framework works end-to-end: JSON parse → metric computation → JSONL
# storage. Full matrix runs are done interactively or in nightly CI.
add_test(NAME benchmark-quick
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run-benchmark.sh --quick
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(benchmark-quick PROPERTIES
    LABELS "benchmark"
    TIMEOUT 600)
