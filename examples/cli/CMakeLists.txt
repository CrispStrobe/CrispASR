#
# crispasr — unified ASR CLI (formerly whisper-cli).
#
# Drives all model backends in src/ (whisper, parakeet, canary, cohere,
# qwen3, voxtral, voxtral4b, granite) from a single binary via the
# --backend flag. When --backend is omitted or set to "whisper" the
# historical whisper-cli code path is used unchanged.
#
# The CMake target is kept as `whisper-cli` so tests/CMakeLists.txt and any
# other CMake rules referring to $<TARGET_FILE:whisper-cli> continue to
# work. The produced binary is renamed to `crispasr` via OUTPUT_NAME, and a
# `whisper-cli` symlink is created next to it for backward compatibility.

set(TARGET whisper-cli)

add_executable(${TARGET}
    cli.cpp
    crispasr_backend.cpp
    crispasr_backend_parakeet.cpp
    crispasr_backend_canary.cpp
    crispasr_backend_cohere.cpp
    crispasr_vad.cpp
    crispasr_output.cpp
    crispasr_model_mgr.cpp
    crispasr_run.cpp
)

set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME crispasr)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE
    common
    whisper
    parakeet
    canary
    cohere
    ${FFMPEG_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS ${TARGET} RUNTIME)

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
