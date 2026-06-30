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

# CLI integration tests — use auto-download to fetch real whisper models.
# Set CRISPASR_MODELS_DIR to a local cache dir to avoid network downloads.
set(TEST_TARGET test-crispasr-cli-medium)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    --backend whisper -m auto --auto-download -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES
    LABELS "medium;live"
    TIMEOUT 300)

set(TEST_TARGET test-crispasr-cli-medium.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    --backend whisper -m auto --auto-download
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES
    LABELS "medium;en;live"
    TIMEOUT 300)

set(TEST_TARGET test-crispasr-cli-large)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr-cli>
    --backend whisper -m auto --auto-download -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES
    LABELS "large;live"
    TIMEOUT 300)

if (CRISPASR_FFMPEG)
    set(TEST_TARGET test-crispasr-cli-tiny-mp3)
#Check with reviewers : any way to check the output transcription via ctest(diff, ...) ?
    add_test(NAME ${TEST_TARGET}
      COMMAND $<TARGET_FILE:crispasr-cli>
      -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.en.bin
      -f ${PROJECT_SOURCE_DIR}/samples/jfk.mp3)
    set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "tiny;mp3")

    # ─── test-ffmpeg-transcode — Catch2 unit tests for container decode (issue #129) ─
    set(FFMPEG_TEST test-ffmpeg-transcode)
    add_executable(${FFMPEG_TEST} ${FFMPEG_TEST}.cpp)
    target_include_directories(${FFMPEG_TEST} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/examples
        ${FFMPEG_INCLUDE_DIRS}
    )
    target_link_libraries(${FFMPEG_TEST} PRIVATE
        Catch2::Catch2WithMain
        common
    )
    target_compile_definitions(${FFMPEG_TEST} PRIVATE
        SAMPLES_DIR="${PROJECT_SOURCE_DIR}/samples"
    )
    catch_discover_tests(${FFMPEG_TEST}
        TEST_SPEC "[ffmpeg]"
        PROPERTIES LABELS "unit;ffmpeg"
    )
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
    crispasr-lib
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
    crispasr-lib
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
    crispasr-lib
)
catch_discover_tests(test-diarize-pyannote-live
    TEST_SPEC "[live]"
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
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
    crispasr-lib
)
catch_discover_tests(test-speaker-cluster
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;diarize"
)

# Named-voiceprint profile DB used by the opt-in --speaker-db path. Drives
# enroll/load/match on synthetic embeddings; no model, no audio, no network.
add_executable(test-speaker-db
    test-speaker-db.cpp
)
target_include_directories(test-speaker-db PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-speaker-db PRIVATE
    Catch2::Catch2WithMain
    crispasr-lib
)
catch_discover_tests(test-speaker-db
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;diarize"
)

# test-titanet — TitaNet speaker-embedding smoke/live tool. Plain main(); takes
# <model.gguf> <audio...> and prints embeddings + a cosine-sim matrix. Builds
# the orphaned smoke test so it can serve as a live check (same-speaker high /
# cross-speaker low cosine; GEMM vs TITANET_FORCE_SCALAR equivalence).
add_executable(test-titanet test-titanet.cpp)
target_include_directories(test-titanet PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-titanet PRIVATE titanet crispasr-core ggml ggml-base ggml-cpu)

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

# ─── test - chatterbox - text - prep — unit tests for the shared text
# normalizer used by the multilingual Chatterbox path.
add_executable(test-chatterbox-text-prep
    test-chatterbox-text-prep.cpp
)
target_include_directories(test-chatterbox-text-prep PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-chatterbox-text-prep PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-chatterbox-text-prep
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;chatterbox"
)

# ─── test - punc - resolve — Catch2 unit tests for the shared --punc-model
# resolver (CLI/server parity). Pure string mapping, header-only, no model load.
add_executable(test-punc-resolve
    test_punc_resolve.cpp
)
target_include_directories(test-punc-resolve PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/examples
    ${PROJECT_SOURCE_DIR}/examples/cli
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-punc-resolve PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-punc-resolve
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
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# ─── test - nemotron — live integration tests for nemotron ASR
add_executable(test-nemotron
    test_nemotron_live.cpp
)
target_include_directories(test-nemotron PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples
)
target_link_libraries(test-nemotron PRIVATE
    Catch2::Catch2WithMain
    nemotron
)
catch_discover_tests(test-nemotron
    TEST_SPEC "[nemotron]"
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# ─── test - ark-asr — live integration tests for ARK-ASR-3B (experimental/WIP)
add_executable(test-ark-asr
    test_ark_asr_live.cpp
)
target_include_directories(test-ark-asr PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples
)
target_link_libraries(test-ark-asr PRIVATE
    Catch2::Catch2WithMain
    ark_asr
)
catch_discover_tests(test-ark-asr
    TEST_SPEC "[ark-asr]"
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# ─── test - lfm2-audio — live integration tests for LFM2-Audio ASR + TTS
add_executable(test-lfm2-audio
    test_lfm2_audio_live.cpp
)
target_include_directories(test-lfm2-audio PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples
)
target_link_libraries(test-lfm2-audio PRIVATE
    Catch2::Catch2WithMain
    lfm2_audio
)
catch_discover_tests(test-lfm2-audio
    TEST_SPEC "[lfm2]"
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# ─── test - snac — unit + live tests for core/snac.h (SNAC 24kHz decoder)
add_executable(test-snac
    test_snac_unit.cpp
)
target_include_directories(test-snac PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-snac PRIVATE
    Catch2::Catch2WithMain
    snac
)
catch_discover_tests(test-snac
    TEST_SPEC "[unit][snac]"
    PROPERTIES
        LABELS unit
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)
catch_discover_tests(test-snac
    TEST_SPEC "[live][snac]"
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# ─── test - mini-omni2 — live integration tests for Mini-Omni2 ASR + TTS + S2S
add_executable(test-mini-omni2
    test_mini_omni2_live.cpp
)
target_include_directories(test-mini-omni2 PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples
)
target_link_libraries(test-mini-omni2 PRIVATE
    Catch2::Catch2WithMain
    mini-omni2
    orpheus
)
catch_discover_tests(test-mini-omni2
    TEST_SPEC "[mini-omni2]"
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# ─── test - higgs-stt — live integration tests for higgs-audio-v3-stt ASR
add_executable(test-higgs-stt
    test_higgs_stt_live.cpp
)
target_include_directories(test-higgs-stt PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples
)
target_link_libraries(test-higgs-stt PRIVATE
    Catch2::Catch2WithMain
    higgs_stt
)
catch_discover_tests(test-higgs-stt
    TEST_SPEC "[higgs-stt]"
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# ─── test - moss-audio — live integration tests for MOSS-Audio-4B audio understanding
add_executable(test-moss-audio
    test_moss_audio_live.cpp
)
target_include_directories(test-moss-audio PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples
)
target_link_libraries(test-moss-audio PRIVATE
    Catch2::Catch2WithMain
    moss_audio
)
catch_discover_tests(test-moss-audio
    TEST_SPEC "[moss-audio]"
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# ─── test - moss-transcribe — live ASR integration test for MOSS-Transcribe-preview-2B
add_executable(test-moss-transcribe
    test_moss_transcribe_live.cpp
)
target_include_directories(test-moss-transcribe PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples
)
target_link_libraries(test-moss-transcribe PRIVATE
    Catch2::Catch2WithMain
    moss_transcribe
)
catch_discover_tests(test-moss-transcribe
    TEST_SPEC "[moss-transcribe]"
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
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

# ─── test-firered-params — Catch2 unit tests for firered_asr_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-firered-params
    test-firered-params.cpp
)
target_include_directories(test-firered-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-firered-params PRIVATE
    Catch2::Catch2WithMain
    firered-asr
)
catch_discover_tests(test-firered-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;firered"
)

# ─── test-cosyvoice3-params — Catch2 unit tests for cosyvoice3_tts_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-cosyvoice3-params
    test-cosyvoice3-params.cpp
)
target_include_directories(test-cosyvoice3-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-cosyvoice3-params PRIVATE
    Catch2::Catch2WithMain
    cosyvoice3_tts
    chatterbox
)
catch_discover_tests(test-cosyvoice3-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;cosyvoice3"
)

# ─── test-dots-tts-params — Catch2 unit tests for dots_tts_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
if (TARGET dots-tts)
add_executable(test-dots-tts-params
    test-dots-tts-params.cpp
)
target_include_directories(test-dots-tts-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-dots-tts-params PRIVATE
    Catch2::Catch2WithMain
    dots-tts
    chatterbox
)
catch_discover_tests(test-dots-tts-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;dots-tts"
)
endif()

# ─── test-funasr-params — Catch2 unit tests for funasr_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-funasr-params
    test-funasr-params.cpp
)
target_include_directories(test-funasr-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-funasr-params PRIVATE
    Catch2::Catch2WithMain
    funasr
)
catch_discover_tests(test-funasr-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;funasr"
)

# ─── test-f5tts-params — Catch2 unit tests for f5_tts_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-f5tts-params
    test-f5tts-params.cpp
)
target_include_directories(test-f5tts-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-f5tts-params PRIVATE
    Catch2::Catch2WithMain
    f5-tts
)
catch_discover_tests(test-f5tts-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;f5tts"
)

# ─── test-parakeet-params — Catch2 unit tests for parakeet_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-parakeet-params
    test-parakeet-params.cpp
)
target_include_directories(test-parakeet-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-parakeet-params PRIVATE
    Catch2::Catch2WithMain
    parakeet
)
catch_discover_tests(test-parakeet-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;parakeet"
)

# ─── test-piper-params — Catch2 unit tests for piper_tts_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-piper-params
    test-piper-params.cpp
)
target_include_directories(test-piper-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-piper-params PRIVATE
    Catch2::Catch2WithMain
    piper-tts
)
catch_discover_tests(test-piper-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;piper"
)

# ─── test-canary-params — Catch2 unit tests for canary_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-canary-params
    test-canary-params.cpp
)
target_include_directories(test-canary-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-canary-params PRIVATE
    Catch2::Catch2WithMain
    canary
)
catch_discover_tests(test-canary-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;canary"
)

# ─── test-orpheus-params — Catch2 unit tests for orpheus_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-orpheus-params
    test-orpheus-params.cpp
)
target_include_directories(test-orpheus-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-orpheus-params PRIVATE
    Catch2::Catch2WithMain
    orpheus
)
catch_discover_tests(test-orpheus-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;orpheus"
)

# ─── test-zonos-params — Catch2 unit tests for zonos_tts_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-zonos-params
    test-zonos-params.cpp
)
target_include_directories(test-zonos-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-zonos-params PRIVATE
    Catch2::Catch2WithMain
    zonos-tts
)
catch_discover_tests(test-zonos-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;zonos"
)

# ─── test-bark-params — Catch2 unit tests for bark_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-bark-params
    test-bark-params.cpp
)
target_include_directories(test-bark-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-bark-params PRIVATE
    Catch2::Catch2WithMain
    bark-tts
)
catch_discover_tests(test-bark-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;bark"
)

# ─── test-canaryctc-params — Catch2 unit tests for canary_ctc_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-canaryctc-params
    test-canaryctc-params.cpp
)
target_include_directories(test-canaryctc-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-canaryctc-params PRIVATE
    Catch2::Catch2WithMain
    canary_ctc
)
catch_discover_tests(test-canaryctc-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;canary_ctc"
)

# ─── test-csm-params — Catch2 unit tests for csm_tts_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-csm-params
    test-csm-params.cpp
)
target_include_directories(test-csm-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-csm-params PRIVATE
    Catch2::Catch2WithMain
    csm-tts
)
catch_discover_tests(test-csm-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;csm"
)

# ─── test-kokoro-params — Catch2 unit tests for kokoro_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-kokoro-params
    test-kokoro-params.cpp
)
target_include_directories(test-kokoro-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-kokoro-params PRIVATE
    Catch2::Catch2WithMain
    kokoro
)
catch_discover_tests(test-kokoro-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;kokoro"
)

# ─── test-sensevoice-params — Catch2 unit tests for sensevoice_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-sensevoice-params
    test-sensevoice-params.cpp
)
target_include_directories(test-sensevoice-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-sensevoice-params PRIVATE
    Catch2::Catch2WithMain
    sensevoice
)
catch_discover_tests(test-sensevoice-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;sensevoice"
)

# ─── test-speecht5-params — Catch2 unit tests for speecht5_tts_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-speecht5-params
    test-speecht5-params.cpp
)
target_include_directories(test-speecht5-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-speecht5-params PRIVATE
    Catch2::Catch2WithMain
    speecht5-tts
)
catch_discover_tests(test-speecht5-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;speecht5"
)

# ─── test-dia-params — Catch2 unit tests for dia_tts_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-dia-params
    test-dia-params.cpp
)
target_include_directories(test-dia-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-dia-params PRIVATE
    Catch2::Catch2WithMain
    dia-tts
)
catch_discover_tests(test-dia-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;dia"
)

# ─── test-melotts-params — Catch2 unit tests for melotts_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-melotts-params
    test-melotts-params.cpp
)
target_include_directories(test-melotts-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-melotts-params PRIVATE
    Catch2::Catch2WithMain
    melotts
)
catch_discover_tests(test-melotts-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;melotts"
)

# ─── test-mimoasr-params — Catch2 unit tests for mimo_asr_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-mimoasr-params
    test-mimoasr-params.cpp
)
target_include_directories(test-mimoasr-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-mimoasr-params PRIVATE
    Catch2::Catch2WithMain
    mimo_asr
)
catch_discover_tests(test-mimoasr-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;mimo_asr"
)

# ─── test-nemotron-params — Catch2 unit tests for nemotron_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-nemotron-params
    test-nemotron-params.cpp
)
target_include_directories(test-nemotron-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-nemotron-params PRIVATE
    Catch2::Catch2WithMain
    nemotron
)
catch_discover_tests(test-nemotron-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;nemotron"
)

# ─── test-omniasr-params — Catch2 unit tests for omniasr_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-omniasr-params
    test-omniasr-params.cpp
)
target_include_directories(test-omniasr-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-omniasr-params PRIVATE
    Catch2::Catch2WithMain
    omniasr
)
catch_discover_tests(test-omniasr-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;omniasr"
)

# ─── test-outetts-params — Catch2 unit tests for outetts_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-outetts-params
    test-outetts-params.cpp
)
target_include_directories(test-outetts-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-outetts-params PRIVATE
    Catch2::Catch2WithMain
    outetts
)
catch_discover_tests(test-outetts-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;outetts"
)

# ─── test-parlertts-params — Catch2 unit tests for parler_tts_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-parlertts-params
    test-parlertts-params.cpp
)
target_include_directories(test-parlertts-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-parlertts-params PRIVATE
    Catch2::Catch2WithMain
    parler-tts
)
catch_discover_tests(test-parlertts-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;parler_tts"
)

# ─── test-pockettts-params — Catch2 unit tests for pocket_tts_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-pockettts-params
    test-pockettts-params.cpp
)
target_include_directories(test-pockettts-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-pockettts-params PRIVATE
    Catch2::Catch2WithMain
    pocket-tts
)
catch_discover_tests(test-pockettts-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;pocket_tts"
)

# ─── test-qwen3asr-params — Catch2 unit tests for qwen3_asr_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-qwen3asr-params
    test-qwen3asr-params.cpp
)
target_include_directories(test-qwen3asr-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-qwen3asr-params PRIVATE
    Catch2::Catch2WithMain
    qwen3_asr
)
catch_discover_tests(test-qwen3asr-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;qwen3_asr"
)

# ─── test-qwen3tts-params — Catch2 unit tests for qwen3_tts_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-qwen3tts-params
    test-qwen3tts-params.cpp
)
target_include_directories(test-qwen3tts-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-qwen3tts-params PRIVATE
    Catch2::Catch2WithMain
    qwen3_tts
)
catch_discover_tests(test-qwen3tts-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;qwen3_tts"
)

# ─── test-tada-params — Catch2 unit tests for tada_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-tada-params
    test-tada-params.cpp
)
target_include_directories(test-tada-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-tada-params PRIVATE
    Catch2::Catch2WithMain
    tada-tts
)
catch_discover_tests(test-tada-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;tada"
)

# ─── test-vibevoice-params — Catch2 unit tests for vibevoice_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-vibevoice-params
    test-vibevoice-params.cpp
)
target_include_directories(test-vibevoice-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-vibevoice-params PRIVATE
    Catch2::Catch2WithMain
    vibevoice
)
catch_discover_tests(test-vibevoice-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;vibevoice"
)

# ─── test-voxcpm2-params — Catch2 unit tests for voxcpm2_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-voxcpm2-params
    test-voxcpm2-params.cpp
)
target_include_directories(test-voxcpm2-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-voxcpm2-params PRIVATE
    Catch2::Catch2WithMain
    voxcpm2_tts
)
catch_discover_tests(test-voxcpm2-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;voxcpm2"
)

# ─── test-cohere-params — Catch2 unit tests for cohere_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-cohere-params
    test-cohere-params.cpp
)
target_include_directories(test-cohere-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-cohere-params PRIVATE
    Catch2::Catch2WithMain
    cohere
)
catch_discover_tests(test-cohere-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;cohere"
)

# ─── test-fastpitch-params — Catch2 unit tests for fastpitch_tts_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-fastpitch-params
    test-fastpitch-params.cpp
)
target_include_directories(test-fastpitch-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-fastpitch-params PRIVATE
    Catch2::Catch2WithMain
    fastpitch-tts
)
catch_discover_tests(test-fastpitch-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;fastpitch"
)

# ─── test-glmasr-params — Catch2 unit tests for glm_asr_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-glmasr-params
    test-glmasr-params.cpp
)
target_include_directories(test-glmasr-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-glmasr-params PRIVATE
    Catch2::Catch2WithMain
    glm-asr
)
catch_discover_tests(test-glmasr-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;glm_asr"
)

# ─── test-granitespeech-params — Catch2 unit tests for granite_speech_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-granitespeech-params
    test-granitespeech-params.cpp
)
target_include_directories(test-granitespeech-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-granitespeech-params PRIVATE
    Catch2::Catch2WithMain
    granite_speech
)
catch_discover_tests(test-granitespeech-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;granite_speech"
)

# ─── test-indextts-params — Catch2 unit tests for indextts_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-indextts-params
    test-indextts-params.cpp
)
target_include_directories(test-indextts-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-indextts-params PRIVATE
    Catch2::Catch2WithMain
    indextts
)
catch_discover_tests(test-indextts-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;indextts"
)

# ─── test-kyutaistt-params — Catch2 unit tests for kyutai_stt_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-kyutaistt-params
    test-kyutaistt-params.cpp
)
target_include_directories(test-kyutaistt-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-kyutaistt-params PRIVATE
    Catch2::Catch2WithMain
    kyutai-stt
)
catch_discover_tests(test-kyutaistt-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;kyutai_stt"
)

# ─── test-m2m100-params — Catch2 unit tests for m2m100_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-m2m100-params
    test-m2m100-params.cpp
)
target_include_directories(test-m2m100-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-m2m100-params PRIVATE
    Catch2::Catch2WithMain
    m2m100
)
catch_discover_tests(test-m2m100-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;m2m100"
)

# ─── test-moonshine-stream-params — Catch2 unit tests for moonshine_streaming_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-moonshine-stream-params
    test-moonshine-stream-params.cpp
)
target_include_directories(test-moonshine-stream-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-moonshine-stream-params PRIVATE
    Catch2::Catch2WithMain
    moonshine_streaming
    crispasr-core
)
catch_discover_tests(test-moonshine-stream-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;moonshine_streaming"
)

# ─── test-openvoice2-params — Catch2 unit tests for openvoice2_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-openvoice2-params
    test-openvoice2-params.cpp
)
target_include_directories(test-openvoice2-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-openvoice2-params PRIVATE
    Catch2::Catch2WithMain
    openvoice2
)
catch_discover_tests(test-openvoice2-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;openvoice2"
)

# ─── test-t5translate-params — Catch2 unit tests for t5_translate_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-t5translate-params
    test-t5translate-params.cpp
)
target_include_directories(test-t5translate-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-t5translate-params PRIVATE
    Catch2::Catch2WithMain
    t5_translate
)
catch_discover_tests(test-t5translate-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;t5_translate"
)

# ─── test-gemma4e2b-params — Catch2 unit tests for gemma4_e2b_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-gemma4e2b-params
    test-gemma4e2b-params.cpp
)
target_include_directories(test-gemma4e2b-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-gemma4e2b-params PRIVATE
    Catch2::Catch2WithMain
    gemma4_e2b
    crispasr-core
)
catch_discover_tests(test-gemma4e2b-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;gemma4_e2b"
)

# ─── test-granitenle-params — Catch2 unit tests for granite_nle_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-granitenle-params
    test-granitenle-params.cpp
)
target_include_directories(test-granitenle-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-granitenle-params PRIVATE
    Catch2::Catch2WithMain
    granite_nle
)
catch_discover_tests(test-granitenle-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;granite_nle"
)

# ─── test-orpheussnac-params — Catch2 unit tests for snac_decoder_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-orpheussnac-params
    test-orpheussnac-params.cpp
)
target_include_directories(test-orpheussnac-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-orpheussnac-params PRIVATE
    Catch2::Catch2WithMain
    snac
    crispasr-core
)
catch_discover_tests(test-orpheussnac-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;orpheus_snac"
)

# ─── test-voxtral-params — Catch2 unit tests for voxtral_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-voxtral-params
    test-voxtral-params.cpp
)
target_include_directories(test-voxtral-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-voxtral-params PRIVATE
    Catch2::Catch2WithMain
    voxtral
)
catch_discover_tests(test-voxtral-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;voxtral"
)

# ─── test-voxtral4b-params — Catch2 unit tests for voxtral4b_context_params defaults
# and null-guard coverage. No GGUF required. ───────────────────────────────────
add_executable(test-voxtral4b-params
    test-voxtral4b-params.cpp
)
target_include_directories(test-voxtral4b-params PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-voxtral4b-params PRIVATE
    Catch2::Catch2WithMain
    voxtral4b
)
catch_discover_tests(test-voxtral4b-params
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;voxtral4b"
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
# Link crispasr-lib (which already pulls every backend) rather than
# enumerating individual backend libraries — this avoids missing
# transitive dependencies (e.g. kokoro → phonemizer → crispasr_cache).
target_link_libraries(test-flash-attn-defaults PRIVATE
    Catch2::Catch2WithMain
    crispasr-lib
)
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

# ─── test-issue-150-cjk-fallback — issue #150 regression guard
#
# Pins split_text_at_punct via crispasr_make_disp_segments: the 42-char CJK
# subtitle fallback must only fire for text containing CJK/kana/Hangul, not
# for long English sentences where it would split words at raw character
# boundaries (e.g. "about" → "abo" / "ut").  Compiles crispasr_output.cpp
# directly — no model load, pure CPU.
add_executable(test-issue-150-cjk-fallback
    test-issue-150-cjk-fallback.cpp
    ${PROJECT_SOURCE_DIR}/examples/cli/crispasr_output.cpp
)
target_include_directories(test-issue-150-cjk-fallback PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-issue-150-cjk-fallback PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-issue-150-cjk-fallback
    TEST_SPEC "[issue-150]"
    PROPERTIES LABELS "unit;output;issue-150"
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
    crispasr-lib
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
    crispasr-lib
)
catch_discover_tests(test-session-setters
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test-session-beam — functional beam-search regression test (§90).
#
# [unit][beam]  — setter API (null guard, width clamping). No model.
# [beam][.live] — end-to-end: beam_size=1 no-regression + beam 2-4
#                 produce non-empty output. Gated on CRISPASR_MODEL_WHISPER
#                 (+ optionally CRISPASR_MODEL_GLM_ASR).
add_executable(test-session-beam
    test-session-beam.cpp
)
target_include_directories(test-session-beam PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/include
)
target_link_libraries(test-session-beam PRIVATE
    Catch2::Catch2WithMain
    crispasr-lib
)
catch_discover_tests(test-session-beam
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;beam"
)
catch_discover_tests(test-session-beam
    TEST_SPEC "[.live]"
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
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

# ─── test-companion-resolve — Catch2 unit tests for companion resolution (#146 / #148)
#
# Verifies the companion pre-download guard: sibling discovery, cache probe,
# and correct companion size in download prompts. No network, no models.
add_executable(test-companion-resolve test-companion-resolve.cpp
    ${PROJECT_SOURCE_DIR}/src/crispasr_model_registry.cpp
    ${PROJECT_SOURCE_DIR}/src/crispasr_cache.cpp
)
target_include_directories(test-companion-resolve PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-companion-resolve PRIVATE
    Catch2::Catch2WithMain
)
if(WIN32)
    target_link_libraries(test-companion-resolve PRIVATE winhttp)
endif()
catch_discover_tests(test-companion-resolve
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

# ─── test_watermark — Catch2 unit tests for the spread-spectrum audio
#watermark (embed + detect round-trip, SNR, robustness).
#Pure CPU, no model dependencies — ~10 ms total.
add_executable(test_watermark test_watermark.cpp)
target_include_directories(test_watermark PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test_watermark PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test_watermark
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test_tts_provenance — Catch2 unit tests for TTS AI-provenance
#compliance (WAV LIST/INFO, ID3v2, C2PA, consent gate, watermark edge cases).
#Pure CPU, no model dependencies.
add_executable(test_tts_provenance test_tts_provenance.cpp)
target_include_directories(test_tts_provenance PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
)
target_link_libraries(test_tts_provenance PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test_tts_provenance
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test_audioseal — Catch2 unit tests for AudioSeal ggml watermark.
#API surface tests run unconditionally. Live model tests require
#AUDIOSEAL_GGUF env var pointing at an audioseal.gguf.
add_executable(test_audioseal test_audioseal.cpp)
target_include_directories(test_audioseal PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test_audioseal PRIVATE
    Catch2::Catch2WithMain
    audioseal
)
catch_discover_tests(test_audioseal
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test_audioseal_cosine — standalone cosine comparison against PyTorch ref.
# Not a Catch2 test — standalone main(). Run manually after generating
# reference with Python. Requires AUDIOSEAL_GGUF env var.
add_executable(test_audioseal_cosine test_audioseal_cosine.cpp)
target_include_directories(test_audioseal_cosine PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test_audioseal_cosine PRIVATE
    audioseal
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

# ─── test-core-sentencepiece — core/sentencepiece.h Viterbi tokenizer ─────────
# Verifies the extracted SentencePiece unigram tokenizer against an
# independent brute-force Viterbi (fuzzed) + hand-computed segmentations.
add_executable(test-core-sentencepiece test-core-sentencepiece.cpp)
target_include_directories(test-core-sentencepiece PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-core-sentencepiece PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-core-sentencepiece
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test-core-lang-names — core/lang_names.h ISO-639-1 → English map ─────────
# Locks the shared language-name map that replaced four copy-pasted copies
# (PLAN §175): the common 15 codes (bit-identical to the former CLI + C-ABI
# maps), the gemma4-only union (uk/vi), and unknown-input passthrough.
add_executable(test-core-lang-names test-core-lang-names.cpp)
target_include_directories(test-core-lang-names PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-core-lang-names PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-core-lang-names
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test-bench-enabled — unit tests for the bench_enabled() + BenchStage RAII
# pattern shared by 70+ backends. Pure env-var logic, no model load.
add_executable(test-bench-enabled test-bench-enabled.cpp)
target_link_libraries(test-bench-enabled PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test-bench-enabled
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;bench"
)

# ─── test-core-bpe — core/bpe.h GPT-2 byte-level BPE tokenizer ──────────────
# Verifies byte_encoder, utf8_encode, bytes_to_unicode, bpe_one merge,
# token_bytes_to_utf8, detokenize round-trips. No model load, pure CPU.
add_executable(test-core-bpe test-core-bpe.cpp)
target_include_directories(test-core-bpe PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-core-bpe PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test-core-bpe
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;bpe"
)

# ─── test-core-wordpiece — core/wordpiece.h BERT WordPiece tokenizer ─────────
# Verifies build_map, lookup, cased/uncased detection, BasicTokenizer,
# WordPiece greedy subword splitting, [UNK] fallback. No model load, pure CPU.
add_executable(test-core-wordpiece test-core-wordpiece.cpp)
target_include_directories(test-core-wordpiece PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-core-wordpiece PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test-core-wordpiece
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;wordpiece"
)

# ─── test-core-istft — core/istft.h inverse-STFT overlap-add ──────────────────
# Verifies the extracted iSTFT against an independent full-DFT reference
# (fuzzed) + a forward-STFT→iSTFT sinusoid round-trip.
add_executable(test-core-istft test-core-istft.cpp)
target_include_directories(test-core-istft PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-core-istft PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test-core-istft
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test-core-mel-parallel — core/mel.cpp §176f parallel-STFT bit-identity ───
# Asserts the OpenMP-parallel STFT (Params::allow_parallel_stft) is bit-identical
# to the serial path. Links crispasr-core (core_mel::compute is not header-only).
add_executable(test-core-mel-parallel test-core-mel-parallel.cpp)
target_include_directories(test-core-mel-parallel PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test-core-mel-parallel PRIVATE
    Catch2::Catch2WithMain
    crispasr-core
)
catch_discover_tests(test-core-mel-parallel
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;mel"
)

# ─── test-core-cross-attn — core/cross_attn.h manual path (ggml graph) ────────
# Computes the header's cross-attention graph and compares to a plain-C++ MHA
# reference (verifies the bug-fixed V-transpose + unscaled T5 contract).
add_executable(test-core-cross-attn test-core-cross-attn.cpp)
target_include_directories(test-core-cross-attn PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test-core-cross-attn PRIVATE
    Catch2::Catch2WithMain
    ggml ggml-base ggml-cpu
)
catch_discover_tests(test-core-cross-attn
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test-core-adaln — core/adaln.h AdaLN-Zero modulation (ggml graph) ────────
add_executable(test-core-adaln test-core-adaln.cpp)
target_include_directories(test-core-adaln PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test-core-adaln PRIVATE
    Catch2::Catch2WithMain
    ggml ggml-base ggml-cpu
)
catch_discover_tests(test-core-adaln
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test-core-seanet — core/seanet_decoder.h convt1d_crop fix (ggml graph) ───
add_executable(test-core-seanet test-core-seanet.cpp)
target_include_directories(test-core-seanet PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test-core-seanet PRIVATE
    Catch2::Catch2WithMain
    ggml ggml-base ggml-cpu
)
catch_discover_tests(test-core-seanet
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test-audio-formats — Catch2 unit tests for crispasr_audio_load format coverage
#
# Validates AU (µ-law), AMR-NB, and WebM (Opus) decode against the WAV
# reference (jfk.wav). Cross-correlation + length + energy checks. Pure CPU,
# no model load — only requires sample files in samples/.
add_executable(test-audio-formats test-audio-formats.cpp)
target_include_directories(test-audio-formats PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/include
)
target_link_libraries(test-audio-formats PRIVATE
    Catch2::Catch2WithMain
    crispasr-lib
)
target_compile_definitions(test-audio-formats PRIVATE
    SAMPLES_DIR="${PROJECT_SOURCE_DIR}/samples"
)
catch_discover_tests(test-audio-formats
    TEST_SPEC "[audio]"
    PROPERTIES LABELS "unit;audio"
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
    crispasr-lib
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
    crispasr-lib
)
catch_discover_tests(test-chat-ggml
    TEST_SPEC "[chat]"
    PROPERTIES
        LABELS live
        SKIP_RETURN_CODE 4
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# ─── test_metal_norm_repro — ggml_norm Metal vs CPU bit-exactness ────────────
# Standalone repro for the kernel_norm cross-simdgroup reduction bug
# (see tools/upstream-prs/08-metal-norm-cross-simdgroup). Sweeps a
# range of T values around the failure boundary and asserts CPU and
# Metal outputs match bit-exactly. Guards against ggml bumps reverting
# the patch.
if(APPLE AND TARGET ggml-metal)
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

# ─── test_ggml_audio_ops_backend — carried ggml audio-op backend patches ─────
#
# These run conv_transpose_1d and col2im_1d directly on a single GPU backend
# without a CPU fallback scheduler. They catch ggml bumps that drop CrispASR's
# backend kernels or dtype support for audio codec/vocoder paths.
function(add_ggml_audio_ops_backend_test TEST_TARGET BACKEND_DEF BACKEND_TARGET)
    add_executable(${TEST_TARGET} test_ggml_audio_ops_backend.cpp)
    target_include_directories(${TEST_TARGET} PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
    )
    target_compile_definitions(${TEST_TARGET} PRIVATE ${BACKEND_DEF})
    target_link_libraries(${TEST_TARGET} PRIVATE
        ggml
        ggml-base
        ggml-cpu
        ${BACKEND_TARGET}
    )
    add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})
    set_tests_properties(${TEST_TARGET} PROPERTIES
        LABELS "unit;ggml;backend"
        TIMEOUT 45
        SKIP_RETURN_CODE 4)
endfunction()

if(APPLE AND TARGET ggml-metal)
    add_ggml_audio_ops_backend_test(test_ggml_audio_ops_metal
        "CRISPASR_TEST_BACKEND_METAL;CRISPASR_TEST_COL2IM_F16"
        ggml-metal)
    set_tests_properties(test_ggml_audio_ops_metal PROPERTIES LABELS "unit;ggml;backend;metal")
endif()

if(TARGET ggml-vulkan)
    add_ggml_audio_ops_backend_test(test_ggml_audio_ops_vulkan
        CRISPASR_TEST_BACKEND_VULKAN
        ggml-vulkan)
    set_tests_properties(test_ggml_audio_ops_vulkan PROPERTIES LABELS "unit;ggml;backend;vulkan")
endif()

if(TARGET ggml-cuda)
    add_ggml_audio_ops_backend_test(test_ggml_audio_ops_cuda
        "CRISPASR_TEST_BACKEND_CUDA;CRISPASR_TEST_COL2IM_F16"
        ggml-cuda)
    set_tests_properties(test_ggml_audio_ops_cuda PROPERTIES LABELS "unit;ggml;backend;cuda")
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

# --- test-f5-tts --- standalone F5-TTS DiT+Vocos smoke test
add_executable(test-f5-tts test-f5-tts.cpp)
target_include_directories(test-f5-tts PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test-f5-tts PRIVATE f5-tts)

# ─── test-piper-tts — standalone Piper VITS TTS smoke test ─────────────────
add_executable(test-piper-tts test-piper-tts.cpp)
target_include_directories(test-piper-tts PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test-piper-tts PRIVATE piper-tts)

# ─── test-g2p-en — Catch2 tests for English G2P (ARPAbet, LTS, phonemizer) ──
add_executable(test-g2p-en test-g2p-en.cpp
    ${PROJECT_SOURCE_DIR}/src/phonemizer.cpp)
target_include_directories(test-g2p-en PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test-g2p-en PRIVATE
    Catch2::Catch2WithMain
)
# Need dl for dlopen on Linux
if(NOT WIN32 AND NOT APPLE)
    target_link_libraries(test-g2p-en PRIVATE dl)
endif()
catch_discover_tests(test-g2p-en
    EXTRA_ARGS --colour-mode ansi
    PROPERTIES LABELS "unit;g2p;phonemizer"
    PROPERTIES TIMEOUT 30
)

# ─── test-g2p-de — Catch2 tests for German G2P ──────────────────────────────
add_executable(test-g2p-de test-g2p-de.cpp)
target_include_directories(test-g2p-de PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test-g2p-de PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test-g2p-de
    EXTRA_ARGS --colour-mode ansi
    PROPERTIES LABELS "unit;g2p;german"
    PROPERTIES TIMEOUT 30
)

# ─── test-g2p-fr — Catch2 tests for French G2P ──────────────────────────────
add_executable(test-g2p-fr test-g2p-fr.cpp)
target_include_directories(test-g2p-fr PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test-g2p-fr PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test-g2p-fr
    EXTRA_ARGS --colour-mode ansi
    PROPERTIES LABELS "unit;g2p;french"
    PROPERTIES TIMEOUT 30
)

# ─── test-g2p-es — Catch2 tests for Spanish G2P ─────────────────────────────
add_executable(test-g2p-es test-g2p-es.cpp)
target_include_directories(test-g2p-es PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test-g2p-es PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test-g2p-es
    EXTRA_ARGS --colour-mode ansi
    PROPERTIES LABELS "unit;g2p;spanish"
    PROPERTIES TIMEOUT 30
)

# ─── test-espeak-phonemize — Catch2 tests for espeak-ng phonemization ───────
# Tests espeak availability, piper text/IPA synthesis, input validation.
# Requires PIPER_TEST_MODEL env var or -DPIPER_TEST_MODEL=... for live tests.
add_executable(test-espeak-phonemize test-espeak-phonemize.cpp)
target_include_directories(test-espeak-phonemize PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test-espeak-phonemize PRIVATE
    Catch2::Catch2WithMain
    piper-tts
)
catch_discover_tests(test-espeak-phonemize
    EXTRA_ARGS --colour-mode ansi
    PROPERTIES LABELS "espeak;piper;tts"
    PROPERTIES TIMEOUT 120
)

# ─── test-piper-roundtrip — live TTS→ASR integration test ───────────────────
# Synthesizes text with piper, transcribes with whisper, checks output.
if(TARGET crispasr-cli)
add_test(NAME test-piper-roundtrip
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-piper-roundtrip.sh
            $<TARGET_FILE:crispasr-cli>
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-piper-roundtrip PROPERTIES
    LABELS "live;tts;piper;espeak;roundtrip"
    TIMEOUT 300)
endif()

# ─── test-speaker-id-live — live test: consent gate (--speaker-db-consent) +
# session-scoped --diarize-speakers. SKIPs cleanly without audio/models. ──────
if(TARGET crispasr-cli)
add_test(NAME test-speaker-id-live
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-speaker-id-live.sh
            $<TARGET_FILE:crispasr-cli>
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-speaker-id-live PROPERTIES
    LABELS "live;diarize;speaker"
    TIMEOUT 600)
endif()

# ─── test-progress-output — live test: -pp/--print-progress emits per-slice
# progress on stderr for unified backends (issue #130).
add_test(NAME test-progress-output
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-progress-output.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-progress-output PROPERTIES
    LABELS "live;progress"
    TIMEOUT 300)

# ─── test-dry-run-resolve — offline regression: --dry-run-resolve must preview
# the same model a real `-m <alias>` would load (sub-variant keys, §166).
# Needs only the built binary (no models / network); SKIPs if absent.
add_test(NAME test-dry-run-resolve
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-dry-run-resolve.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-dry-run-resolve PROPERTIES
    LABELS "cli"
    TIMEOUT 60)

# ─── test-server-ws-stream — live: --ws-port real-time WebSocket ASR streaming.
# Raw-socket client (stdlib only) verifies the RFC 6455 handshake + a streaming
# transcription. Needs a whisper ggml model; SKIPs otherwise.
add_test(NAME test-server-ws-stream
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-server-ws-stream.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-server-ws-stream PROPERTIES
    LABELS "live;server"
    TIMEOUT 300)

# ─── test-wyoming-proto — Catch2 unit tests for Wyoming wire format, PCM
# conversion (f32↔s16), and linear-interpolation resampler.
# Pure CPU, no model or audio device needed. The socket-pair framing tests
# are POSIX-only and compiled out on Windows.
add_executable(test-wyoming-proto test-wyoming-proto.cpp)
target_link_libraries(test-wyoming-proto PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test-wyoming-proto
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;wyoming")

# ─── test-crispasr-speaker-resample — Catch2 unit tests for the pre-resample
# math inside crispasr_speaker_play(): upsample/downsample, mono expansion,
# stereo mix-down, DC preservation. No audio device or model needed.
add_executable(test-crispasr-speaker-resample test-crispasr-speaker-resample.cpp)
target_link_libraries(test-crispasr-speaker-resample PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test-crispasr-speaker-resample
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;speaker")

# ─── test-server-wyoming — live: --wyoming-port Wyoming protocol TCP server.
# Python stdlib client exercises describe→info, ASR transcription, disconnect
# resilience, and unknown-event resilience. Needs a whisper ggml model; SKIPs
# otherwise.
add_test(NAME test-server-wyoming
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-server-wyoming.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-server-wyoming PROPERTIES
    LABELS "live;server;wyoming"
    TIMEOUT 300)

# ─── test-server-tts — live: /v1/audio/speech + /v1/voices smoke, incl. the
# PR #177 streaming path (stream=true per-chunk PCM, mp3 rejection, log marker).
# Needs a qwen3-tts CustomVoice GGUF + tokenizer (set CRISPASR_MODELS); SKIPs
# (exit 0) otherwise.
add_test(NAME test-server-tts
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-server-tts.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-server-tts PROPERTIES
    LABELS "live;server;tts"
    TIMEOUT 600)

# ─── test-server-stream-resilience — live: the detached streaming-TTS worker
# must not crash the server on an exception. Forces a worker throw
# (CRISPASR_TEST_STREAM_THROW=1 + magic input) and asserts the server survives +
# keeps serving. Auto-discovers a qwen3-tts GGUF; SKIPs otherwise.
add_test(NAME test-server-stream-resilience
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-server-stream-resilience.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-server-stream-resilience PROPERTIES
    LABELS "live;server;tts;qwen3-tts"
    TIMEOUT 400)

# ─── test-qwen3-tts-codes-determinism — live: CLI whole-clip synthesis gates.
# (1) two identical --tts runs are bit-identical (guards the shared talker AR
# loop / streaming dedup); (2) QWEN3_TTS_O15=1 is byte-identical to O15=0
# (guards #56). Auto-discovers a qwen3-tts GGUF; SKIPs (exit 0) otherwise.
add_test(NAME test-qwen3-tts-codes-determinism
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-qwen3-tts-codes-determinism.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-qwen3-tts-codes-determinism PROPERTIES
    LABELS "live;tts;qwen3-tts;determinism"
    TIMEOUT 300)

# ─── test-qwen3-tts-seed-live — live: server-side seed determinism on
# /v1/audio/speech (same request seed → bit-identical WAV; different seed →
# different). Needs CRISPASR_TEST_QWEN3_TTS_MODEL/CODEC (CustomVoice); SKIPs.
add_test(NAME test-qwen3-tts-seed-live
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-qwen3-tts-seed-live.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-qwen3-tts-seed-live PROPERTIES
    LABELS "live;server;tts;qwen3-tts;determinism"
    TIMEOUT 300)

# ─── test-release-workflow — validate release.yml targets (issue #131).
# Static test: no binary or model needed, just parses YAML.
add_test(NAME test-release-workflow
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_release_workflow.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-release-workflow PROPERTIES
    LABELS "unit;ci"
    TIMEOUT 30)

# ─── test_diarized_json — Catch2 unit tests for the diarized_json formatter
# (issue #206) and max_len display segment splitting (issue #205).
# Pure functions, no backend dependency.
add_executable(test_diarized_json
    test_diarized_json.cpp
    ${PROJECT_SOURCE_DIR}/examples/cli/crispasr_output.cpp
)
target_include_directories(test_diarized_json PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/cli
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test_diarized_json PRIVATE
    Catch2::Catch2WithMain
)
catch_discover_tests(test_diarized_json
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit;diarized-json;max-len"
)

# ─── test-server-diarized-json — live: /v1/audio/transcriptions with
# response_format=diarized_json (issue #206). Boots whisper server, validates
# speaker labels, type field, start/end timestamps. SKIPs without model.
add_test(NAME test-server-diarized-json
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-server-diarized-json.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-server-diarized-json PROPERTIES
    LABELS "live;server;diarized-json"
    TIMEOUT 300)

# ─── test-granite-maxlen — live: --max-len splitting on granite-speech
# (issue #205). Runs CLI with --max-len and verifies segments are split.
# SKIPs without a granite-speech PLUS GGUF.
add_test(NAME test-granite-maxlen
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-granite-maxlen.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(test-granite-maxlen PROPERTIES
    LABELS "live;granite;max-len"
    TIMEOUT 300)
