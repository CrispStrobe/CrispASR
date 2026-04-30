set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Catch2 — unit-test framework (pytest analogue) ──────────────────────────
# Fetched at configure time; pinned to a specific tag for reproducibility.
# The resulting targets (Catch2::Catch2WithMain, Catch2::Catch2) are then
# available for any test in this directory.
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
    # test-crispasr-js

    set(TEST_TARGET test-crispasr-js)

    add_test(NAME ${TEST_TARGET}
        COMMAND node test-crispasr.js --experimental-wasm-threads
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

    return()
endif()

set(TEST_TARGET test-crispasr-cli-tiny)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "tiny;gh")

set(TEST_TARGET test-crispasr-cli-tiny.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "tiny;en")

set(TEST_TARGET test-crispasr-cli-base)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-base.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "base")

set(TEST_TARGET test-crispasr-cli-base.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-base.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "base;en")

set(TEST_TARGET test-crispasr-cli-small)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-small.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "small")

set(TEST_TARGET test-crispasr-cli-small.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-small.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "small;en")

set(TEST_TARGET test-crispasr-cli-medium)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-medium.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "medium")

set(TEST_TARGET test-crispasr-cli-medium.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-medium.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "medium;en")

set(TEST_TARGET test-crispasr-cli-large)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:crispasr>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-large.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "large")

if (CRISPASR_FFMPEG)
    set(TEST_TARGET test-crispasr-cli-tiny-mp3)
    # Check with reviewers: any way to check the output transcription via ctest (diff, ...)?
    add_test(NAME ${TEST_TARGET}
      COMMAND $<TARGET_FILE:crispasr>
      -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.en.bin
      -f ${PROJECT_SOURCE_DIR}/samples/jfk.mp3)
    set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "tiny;mp3")
endif()

# VAD test tests VAD in isolation
set(VAD_TEST test-vad)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../ggml/include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common)
target_compile_definitions(${VAD_TEST} PRIVATE
    VAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TEST} PROPERTIES LABELS "unit")

# VAD test full uses crispasr_full with VAD enabled
set(VAD_TEST test-vad-full)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../ggml/include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common)
target_compile_definitions(${VAD_TEST} PRIVATE
    CRISPASR_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/ggml-base.en.bin"
    VAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TEST} PROPERTIES LABELS "base;en")

# ─── test-crispasr-cache — Catch2 unit tests for download/cache helpers ───────
#
# Compiles crispasr_cache.cpp directly into the test binary so the test is
# self-contained and does not depend on the full crispasr link closure.
# Tests are pure filesystem operations — no network requests are made.
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
# catch_discover_tests registers one CTest entry per Catch2 TEST_CASE.
catch_discover_tests(test-crispasr-cache
    TEST_SPEC "[unit]"
    PROPERTIES LABELS "unit"
)

# ─── test-params — Catch2 unit tests for crispasr_params struct defaults ────────
#
# Default-constructs crispasr_params and checks every crispasr-specific field
# carries the right initial value.  Links crispasr + common because
# crispasr_params inlines calls to crispasr_full_default_params() in its field
# initialisers.
add_executable(test-params
    test-params.cpp
)
target_include_directories(test-params PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/examples
    ${PROJECT_SOURCE_DIR}/examples/cli
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

# ─── test_qwen3_audio_tower — pre-refactor numeric baseline for the audio
# tower. Captures sum/sum_sq/max_abs + first-32/last-32 windows on the encoder
# output for a fixed PCM (jfk.raw) so that extracting the audio tower into the
# CrispAudio shared library is verifiably non-disruptive. CPU-only — output
# must be deterministic across machines.
add_executable(test_qwen3_audio_tower test_qwen3_audio_tower.cpp)
target_include_directories(test_qwen3_audio_tower PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_qwen3_audio_tower PRIVATE qwen3_asr)

# ─── test_qwen3_tts_talker — sanity check the Qwen3-TTS talker forward.
# Not auto-registered with add_test — needs a real GGUF on disk.
add_executable(test_qwen3_tts_talker test_qwen3_tts_talker.cpp)
target_include_directories(test_qwen3_tts_talker PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_qwen3_tts_talker PRIVATE qwen3_tts)
# Not auto-registered with add_test — needs a model file + jfk.raw which the
# test runner / dev produces locally. Run manually:
#   ./build/tests/test_qwen3_audio_tower model.gguf samples/jfk.raw \
#       > tests/fixtures/qwen3_audio_tower_jfk.txt
#   ./build/tests/test_qwen3_audio_tower --check model.gguf samples/jfk.raw \
#       tests/fixtures/qwen3_audio_tower_jfk.txt

# ─── test-backends — integration tests for all CrispASR backends ────────────
# Requires network (downloads models). Not run in CI by default.
# Usage: ctest -L integration --output-on-failure
# Or:    ./tests/test-backends.sh [backend ...]
set(TEST_TARGET test-backends)
add_test(NAME ${TEST_TARGET}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-backends.sh
        parakeet moonshine wav2vec2 fastconformer-ctc data2vec hubert
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(${TEST_TARGET} PROPERTIES
    LABELS "integration"
    TIMEOUT 600)
