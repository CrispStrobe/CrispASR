#
# crispasr — unified ASR CLI (formerly whisper-cli).
#
# Drives all model backends in src/ (whisper, parakeet, canary, cohere,
# qwen3, voxtral, voxtral4b, granite) from a single binary via the
# --backend flag. When --backend is omitted or set to "whisper" the
# historical whisper-cli code path is used unchanged.
#
# The CMake target was renamed from `whisper-cli` to `crispasr`.
# A backward-compat `whisper-cli` symlink is created for legacy scripts.

set(TARGET crispasr)

add_executable(${TARGET}
    cli.cpp
    crispasr_backend.cpp
    crispasr_backend_whisper.cpp
    crispasr_backend_parakeet.cpp
    crispasr_backend_canary.cpp
    crispasr_backend_cohere.cpp
    crispasr_backend_granite.cpp
    crispasr_backend_voxtral.cpp
    crispasr_backend_voxtral4b.cpp
    crispasr_backend_qwen3.cpp
    crispasr_backend_fastconformer_ctc.cpp
    crispasr_backend_wav2vec2.cpp
    crispasr_backend_vibevoice.cpp
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
    crispasr_run.cpp
    crispasr_server.cpp
)

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
    whisper
    parakeet
    canary
    cohere
    granite_speech
    voxtral
    voxtral4b
    qwen3_asr
    gemma4_e2b
    crispasr-diff-lib
    ${FFMPEG_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)
install(TARGETS crispasr-diff RUNTIME BUNDLE DESTINATION bin)

set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME crispasr)

# httplib.h is in the server example directory
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/server)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE
    common
    whisper
    parakeet
    canary
    cohere
    granite_speech
    voxtral
    voxtral4b
    qwen3_asr
    canary_ctc
    wav2vec2-ggml
    vibevoice
    silero-lid
    pyannote-seg
    ${FFMPEG_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    $<$<BOOL:${WIN32}>:winhttp>
    $<$<BOOL:${WHISPER_CURL}>:CURL::libcurl>
)

if (WHISPER_CURL)
    target_compile_definitions(${TARGET} PRIVATE CRISPASR_USE_LIBCURL)
endif()

install(TARGETS ${TARGET} RUNTIME BUNDLE DESTINATION bin)

# ---------------------------------------------------------------------------
# Backward-compat alias: `whisper-cli` binary name -> `crispasr`.
# ---------------------------------------------------------------------------

# 1) Build-tree alias so ./build/bin/whisper-cli works during development.
#
#    Some storage backends (network mounts, sandbox overlays) don't support
#    symlinks. We try symlink first, then fall back to hardlink, then to a
#    plain file copy. Any of these produces a usable whisper-cli binary.
if (UNIX)
    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -f $<TARGET_FILE_DIR:${TARGET}>/whisper-cli
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            crispasr $<TARGET_FILE_DIR:${TARGET}>/whisper-cli
            || ${CMAKE_COMMAND} -E create_hardlink
                $<TARGET_FILE_DIR:${TARGET}>/crispasr
                $<TARGET_FILE_DIR:${TARGET}>/whisper-cli
            || ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE_DIR:${TARGET}>/crispasr
                $<TARGET_FILE_DIR:${TARGET}>/whisper-cli
        VERBATIM USES_TERMINAL)
endif()

# 2) Install-tree alias (symlink preferred, hardlink/copy fallback).
if (UNIX)
    install(CODE "
        set(_alias \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/whisper-cli\")
        file(REMOVE \"\${_alias}\")
        execute_process(
            COMMAND \${CMAKE_COMMAND} -E create_symlink crispasr \"\${_alias}\"
            RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
        if (NOT _rc EQUAL 0)
            execute_process(
                COMMAND \${CMAKE_COMMAND} -E create_hardlink
                    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/crispasr\" \"\${_alias}\"
                RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
        endif()
        if (NOT _rc EQUAL 0)
            execute_process(
                COMMAND \${CMAKE_COMMAND} -E copy
                    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/crispasr\" \"\${_alias}\")
        endif()
    ")
endif()
