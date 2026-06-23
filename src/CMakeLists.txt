# TODO: should not use this
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    # _USE_MATH_DEFINES is set globally in the top-level CMakeLists.txt

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()


if (CRISPASR_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

if (CRISPASR_COREML)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREML_FRAMEWORK CoreML)

    if (COREML_FRAMEWORK)
        message(STATUS "CoreML framework found")

        set(CRISPASR_EXTRA_FLAGS ${CRISPASR_EXTRA_FLAGS} -DCRISPASR_USE_COREML)
    else()
        message(FATAL_ERROR "CoreML framework not found")
    endif()

    if (CRISPASR_COREML_ALLOW_FALLBACK)
        set(CRISPASR_EXTRA_FLAGS ${CRISPASR_EXTRA_FLAGS} -DCRISPASR_COREML_ALLOW_FALLBACK)
    endif()
endif()

if (CRISPASR_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()

#
# libraries
#

# crispasr.coreml

if (CRISPASR_COREML)
    set(TARGET crispasr.coreml)

    add_library(${TARGET}
        coreml/whisper-compat.m
        coreml/whisper-encoder.h
        coreml/whisper-encoder.mm
        coreml/whisper-encoder-impl.h
        coreml/whisper-encoder-impl.m
        )

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PUBLIC
        .
        )

    target_link_libraries(${TARGET} PRIVATE ${FOUNDATION_FRAMEWORK} ${COREML_FRAMEWORK})

    set_target_properties(${TARGET} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
        )

    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
    install(TARGETS ${TARGET} LIBRARY)
endif()

if (CRISPASR_OPENVINO)
    set(TARGET crispasr.openvino)

    add_library(${TARGET} OBJECT
        openvino/whisper-openvino-encoder.h
        openvino/whisper-openvino-encoder.cpp
        )

    target_include_directories(${TARGET} PUBLIC
        .
        )

    set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set(CRISPASR_EXTRA_FLAGS ${CRISPASR_EXTRA_FLAGS} -DCRISPASR_USE_OPENVINO)

    target_link_libraries(${TARGET} PRIVATE ggml openvino::runtime)
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

# crispasr-lib (formerly whisper)

add_library(crispasr-lib
            ../include/crispasr.h
            crispasr-arch.h
            crispasr.cpp
            # C-ABI shared by every CrispASR consumer (CLI, Dart, Python,
            # Rust). Declared extern "C" with `crispasr_*` prefixes so they
            # don't collide with upstream crispasr.h identifiers.
            crispasr_c_api.cpp
            # Audio file decoder (WAV / MP3 / FLAC via miniaudio).
            # Lets language wrappers feed `crispasr_session_transcribe`
            # any audio format without bundling an external ffmpeg.
            crispasr_audio.cpp
            # Library-level mic capture (PLAN #62d) — wraps miniaudio
            # ma_device for cross-platform live PCM. Symbols share the
            # MINIAUDIO_IMPLEMENTATION already linked from crispasr_audio.cpp.
            crispasr_mic.cpp
            # Local speaker playback (issue #173) — miniaudio ma_device
            # playback mode, mirrors crispasr_mic.cpp. Used by the CLI's
            # --tts-play flag; server mode routes audio via HTTP.
            crispasr_speaker.cpp
            # Silero-VAD-driven segmentation + crispasr.cpp-style stitching.
            # Consumed by the CLI and by the C-ABI `crispasr_session_transcribe_vad`
            # wrapper so every language binding benefits without re-implementing.
            crispasr_vad.cpp
            # Shared diarization methods (energy, xcorr, vad-turns, pyannote).
            # Consumed by the CLI and by the C-ABI `crispasr_diarize_segments`.
            # The sherpa-ONNX subprocess method stays in the CLI shim.
            crispasr_diarize.cpp
            # Shared language identification (crispasr-tiny + native silero).
            # Consumed by the CLI and by the C-ABI
            # `crispasr_detect_language_pcm`. The sherpa-ONNX subprocess
            # method stays in the CLI shim.
            crispasr_lid.cpp
            # Shared CTC / qwen3-fa forced alignment. Consumed by the CLI
            # and by the C-ABI `crispasr_align_words_abi`. Both canary-ctc
            # and qwen3-forced-aligner paths route through one call.
            crispasr_aligner.cpp
            # Built-in G2P phonemizers (EN/DE/FR/ES) live in the kokoro
            # target (phonemizer.cpp) — only kokoro calls them.
            # HF download + filesystem cache (shared with CLI). Consumed
            # by the library's VAD / LID / diarize auto-download paths
            # and by the CLI's `-m auto` resolver.
            crispasr_cache.cpp
            # Known-model registry lookup (backend → canonical GGUF URL).
            # CLI keeps the TTY prompting shim; the library does the
            # non-interactive resolve.
            crispasr_model_registry.cpp
            # Pluggable speaker-embedding interface + concrete adapters
            # (TitaNet, IndexTTS-ECAPA) and agglomerative cosine
            # clustering, shared between the CLI's --diarize-embedder
            # path and the C ABI's crispasr_speaker_embedder_* surface
            # (#107 P6). The adapters depend on the lib-local titanet
            # + indextts_voc runtimes also compiled in here.
            crispasr_speaker_embedder.cpp
            crispasr_speaker_cluster.cpp
            # Text-LLM chat C ABI (docs/prompts/chat-abi.md). Calls into
            # the private crispasr-llama-core static lib; the public
            # surface lives in include/crispasr_chat.h and exposes only
            # POD structs + opaque handles — no llama.h types leak out.
            chat.cpp
            ../include/crispasr_chat.h
            # GBNF grammar parser — moved out of examples/ so the library
            # can expose grammar-constrained sampling via
            # `crispasr_session_set_grammar_text`. The CLI examples now
            # depend on libcrispasr for grammar parsing too instead of
            # building their own copy.
            grammar-parser.h
            grammar-parser.cpp
            )

# RNNoise uses C99 VLAs and M_PI — both unsupported by MSVC out of the box.
# Exclude the whole feature on Windows rather than patching vendored sources.
if (NOT MSVC)
    target_sources(crispasr-lib PRIVATE
        crispasr_enhance.h
        crispasr_enhance.cpp
        rnnoise/rnnoise.h
        rnnoise/denoise.c
        rnnoise/rnn.c
        rnnoise/rnn_data.c
        rnnoise/pitch.c
        rnnoise/celt_lpc.c
        rnnoise/kiss_fft.c
    )
    target_compile_definitions(crispasr-lib PRIVATE CRISPASR_RNNOISE=1)
endif()

# RNNoise's C sources predate the strict implicit-decl errors that
# clang 16+ defaults to. Silence those warnings on a per-file basis
# rather than relaxing global -Werror policy.
if (NOT MSVC)
    set_source_files_properties(
        rnnoise/denoise.c rnnoise/rnn.c rnnoise/rnn_data.c
        rnnoise/pitch.c rnnoise/celt_lpc.c rnnoise/kiss_fft.c
        PROPERTIES
            COMPILE_FLAGS "-Wno-implicit-function-declaration -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-variable"
    )
endif()
add_library(whisper ALIAS crispasr-lib)

# miniaudio.h on Apple includes <AVFoundation/AVFoundation.h> which
# contains ObjC syntax (@class, @protocol). When compiling as pure
# C++ this fails on visionOS/tvOS/iOS xcframework builds. Setting the
# source language to OBJCXX lets clang accept the ObjC headers.
if(APPLE)
    set_source_files_properties(crispasr_audio.cpp crispasr_mic.cpp crispasr_speaker.cpp
        PROPERTIES LANGUAGE OBJCXX)
endif()

set_target_properties(crispasr-lib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${SOVERSION}
    OUTPUT_NAME crispasr
)

target_include_directories(crispasr-lib PUBLIC . ../include)

set(CRISPASR_LIB_CXX_STANDARD cxx_std_11)
if (GGML_CUDA AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    set(CRISPASR_LIB_CXX_STANDARD cxx_std_17)
endif()
target_compile_features   (crispasr-lib PUBLIC ${CRISPASR_LIB_CXX_STANDARD}) # don't bump

# ── Optional .opus (Ogg/Opus) decode in crispasr_audio.cpp ──────────────────
# Adds Opus to the miniaudio decoder path (custom backend in
# src/miniaudio_libopus.c) so crispasr_audio_load handles .opus alongside
# WAV/MP3/FLAC/OGG — no ffmpeg. libopus + opusfile + libogg are BSD-3-Clause.
# Primary: system libs via pkg-config (macOS/Linux). Fallback for platforms
# without system libs (Windows/iOS/Android/WASM): build them statically via
# FetchContent. Disable entirely with -DCRISPASR_OPUS=OFF; force the static
# build (also used to validate the fallback locally) with -DCRISPASR_OPUS_FETCH=ON.
option(CRISPASR_OPUS       "Decode .opus (Ogg/Opus) via libopus + opusfile" ON)
option(CRISPASR_OPUS_FETCH "Skip system pkg-config; build opus/ogg/opusfile statically via FetchContent" OFF)

if (CRISPASR_OPUS)
    set(_crispasr_opus_ok OFF)

    if (NOT CRISPASR_OPUS_FETCH)
        find_package(PkgConfig QUIET)
        if (PkgConfig_FOUND)
            pkg_check_modules(OPUSFILE QUIET IMPORTED_TARGET opusfile)
            if (OPUSFILE_FOUND)
                target_link_libraries(crispasr-lib PRIVATE PkgConfig::OPUSFILE)
                set(_crispasr_opus_ok ON)
                message(STATUS "CrispASR: .opus via system opusfile ${OPUSFILE_VERSION} (pkg-config)")
            endif()
        endif()
    endif()

    if (NOT _crispasr_opus_ok)
        # Static fallback. libogg + libopus ship upstream CMake; opusfile is
        # autotools-only, so compile its (HTTP-free) core sources directly.
        include(FetchContent)
        # Suppress Opus/Ogg test targets without clobbering the global
        # BUILD_TESTING that CTest relies on (include(CTest) runs later
        # and needs BUILD_TESTING to default to ON).
        if (DEFINED CACHE{BUILD_TESTING})
            set(_crispasr_save_build_testing "$CACHE{BUILD_TESTING}")
        else()
            set(_crispasr_save_build_testing "__UNSET__")
        endif()
        set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
        set(OPUS_BUILD_TESTING  OFF CACHE BOOL "" FORCE)
        # Emscripten: Opus cmake auto-detects x86 and enables AVX2/FMA sources
        # that fail under wasm32 clang. Disable all intrinsics.
        if (EMSCRIPTEN)
            set(OPUS_DISABLE_INTRINSICS ON CACHE BOOL "" FORCE)
        endif()
        # libogg/opus ship pre-3.5 cmake_minimum_required, which CMake >= 4.0
        # rejects outright; loosen the floor just for these vendored subbuilds.
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
        FetchContent_Declare(crispasr_ogg  GIT_REPOSITORY https://github.com/xiph/ogg.git  GIT_TAG v1.3.5  GIT_SHALLOW TRUE)
        FetchContent_Declare(crispasr_opus GIT_REPOSITORY https://github.com/xiph/opus.git GIT_TAG v1.5.2 GIT_SHALLOW TRUE)
        FetchContent_Declare(crispasr_opusfile GIT_REPOSITORY https://github.com/xiph/opusfile.git GIT_TAG v0.12 GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(crispasr_ogg crispasr_opus)
        FetchContent_GetProperties(crispasr_opusfile)
        if (NOT crispasr_opusfile_POPULATED)
            FetchContent_Populate(crispasr_opusfile)
        endif()
        add_library(crispasr_opusfile STATIC
            ${crispasr_opusfile_SOURCE_DIR}/src/info.c
            ${crispasr_opusfile_SOURCE_DIR}/src/internal.c
            ${crispasr_opusfile_SOURCE_DIR}/src/opusfile.c
            ${crispasr_opusfile_SOURCE_DIR}/src/stream.c)
        target_include_directories(crispasr_opusfile PUBLIC ${crispasr_opusfile_SOURCE_DIR}/include)
        target_compile_definitions(crispasr_opusfile PRIVATE OP_DISABLE_HTTP)      # no openssl
        target_link_libraries(crispasr_opusfile PUBLIC Ogg::ogg Opus::opus)
        set_target_properties(crispasr_opusfile PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(crispasr-lib PRIVATE crispasr_opusfile)
        set(_crispasr_opus_ok ON)
        message(STATUS "CrispASR: .opus via FetchContent static opus/ogg/opusfile")
        # Restore BUILD_TESTING so include(CTest) in the root CMakeLists
        # defaults correctly (it checks the cache variable).
        if (_crispasr_save_build_testing STREQUAL "__UNSET__")
            unset(BUILD_TESTING CACHE)
        else()
            set(BUILD_TESTING "${_crispasr_save_build_testing}" CACHE BOOL "" FORCE)
        endif()
    endif()

    if (_crispasr_opus_ok)
        # The custom miniaudio backend, compiled as a separate C TU and linked
        # against this lib's MINIAUDIO_IMPLEMENTATION. opusfile.h comes from the
        # linked opusfile target; miniaudio.h is the src/ sibling.
        target_sources(crispasr-lib PRIVATE miniaudio_libopus.c)
        target_compile_definitions(crispasr-lib PRIVATE CRISPASR_HAVE_OPUS=1)
    elseif (NOT CRISPASR_OPUS_FETCH)
        message(STATUS "CrispASR: .opus disabled — opusfile not found (set -DCRISPASR_OPUS_FETCH=ON to build it)")
    endif()
endif()

# ── Optional AMR-NB/WB decode in crispasr_audio.cpp ──────────────────────────
# Telephony / voicemail speech via opencore-amr (Apache-2.0). AMR-NB (8 kHz)
# and AMR-WB (16 kHz) are the standard codecs for mobile voice recordings and
# voicemail. Decoded PCM is resampled to 16 kHz via miniaudio.
# Disable with -DCRISPASR_AMR=OFF; force static build with -DCRISPASR_AMR_FETCH=ON.
option(CRISPASR_AMR       "Decode AMR-NB/WB via opencore-amr" ON)
option(CRISPASR_AMR_FETCH "Skip system pkg-config; build opencore-amr statically via FetchContent" OFF)

if (CRISPASR_AMR)
    set(_crispasr_amr_ok OFF)

    if (NOT CRISPASR_AMR_FETCH)
        find_package(PkgConfig QUIET)
        if (PkgConfig_FOUND)
            pkg_check_modules(OPENCORE_AMRNB QUIET IMPORTED_TARGET opencore-amrnb)
            pkg_check_modules(OPENCORE_AMRWB QUIET IMPORTED_TARGET opencore-amrwb)
            if (OPENCORE_AMRNB_FOUND AND OPENCORE_AMRWB_FOUND)
                target_link_libraries(crispasr-lib PRIVATE PkgConfig::OPENCORE_AMRNB PkgConfig::OPENCORE_AMRWB)
                set(_crispasr_amr_ok ON)
                message(STATUS "CrispASR: AMR via system opencore-amr (pkg-config)")
            endif()
        endif()
    endif()

    if (NOT _crispasr_amr_ok AND CRISPASR_AMR_FETCH)
        # Static fallback (opt-in via -DCRISPASR_AMR_FETCH=ON): build
        # opencore-amr decoder from CrispStrobe/opencore-amr (fork of
        # feuvan/opencore-amr-iOS v0.1.6 + OSCL shim headers in oscl/).
        include(FetchContent)
        FetchContent_Declare(crispasr_opencore_amr
            GIT_REPOSITORY https://github.com/CrispStrobe/opencore-amr.git
            GIT_TAG v0.1.7
            GIT_SHALLOW TRUE)
        FetchContent_GetProperties(crispasr_opencore_amr)
        if (NOT crispasr_opencore_amr_POPULATED)
            FetchContent_Populate(crispasr_opencore_amr)
        endif()

        set(_amr_src "${crispasr_opencore_amr_SOURCE_DIR}")
        # The fork ships oscl/ shim headers and amrnb/amrwb wrappers.
        # Create symlinks for system-style <opencore-amrnb/...> includes.
        if (NOT EXISTS "${_amr_src}/opencore-amrnb")
            file(CREATE_LINK "${_amr_src}/amrnb" "${_amr_src}/opencore-amrnb" SYMBOLIC)
        endif()
        if (NOT EXISTS "${_amr_src}/opencore-amrwb")
            file(CREATE_LINK "${_amr_src}/amrwb" "${_amr_src}/opencore-amrwb" SYMBOLIC)
        endif()

        # Check if this is the CMake-enabled fork (has CMakeLists.txt)
        # or the raw AOSP source (autotools only).
        # Build decoder-only from amrnb/amrwb wrappers + opencore sources.
        # The fork includes oscl/ shim headers for standalone compilation.
        # Glob the raw codec C/C++ sources but exclude the PV framework
        # wrappers (pvgsm*.cpp, decoder_gsm_amr.cpp) which depend on
        # case-insensitive includes and OSCL C++ framework classes we
        # don't need — the amrnb/wrapper.cpp already wraps the codec API.
        file(GLOB _amrnb_srcs
            "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_nb/dec/src/*.cpp"
            "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_nb/dec/src/*.c"
            "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_nb/common/src/*.cpp"
            "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_nb/common/src/*.c"
            "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/common/dec/src/*.cpp"
            "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/common/dec/src/*.c")
        list(FILTER _amrnb_srcs EXCLUDE REGEX "pvgsm[^/]*\\.cpp$|decoder_gsm_amr\\.cpp$")
        file(GLOB _amrwb_srcs
            "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_wb/dec/src/*.cpp"
            "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_wb/dec/src/*.c")
        list(FILTER _amrwb_srcs EXCLUDE REGEX "pvgsm[^/]*\\.cpp$|decoder_amr_wb\\.cpp$")
        if (_amrnb_srcs AND _amrwb_srcs AND EXISTS "${_amr_src}/amrnb/wrapper.cpp")
            add_library(crispasr_amrnb STATIC
                "${_amr_src}/amrnb/wrapper.cpp" ${_amrnb_srcs})
            target_compile_definitions(crispasr_amrnb PRIVATE DISABLE_AMRNB_ENCODER)
            target_include_directories(crispasr_amrnb SYSTEM PUBLIC
                "${_amr_src}" "${_amr_src}/amrnb" "${_amr_src}/oscl"
                "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_nb/dec/src"
                "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_nb/dec/include"
                "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_nb/enc/src"
                "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_nb/enc/include"
                "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_nb/common/include"
                "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/common/dec/include")
            set_target_properties(crispasr_amrnb PROPERTIES POSITION_INDEPENDENT_CODE ON)

            add_library(crispasr_amrwb STATIC
                "${_amr_src}/amrwb/wrapper.cpp" ${_amrwb_srcs})
            target_include_directories(crispasr_amrwb SYSTEM PUBLIC
                "${_amr_src}" "${_amr_src}/amrwb" "${_amr_src}/oscl"
                "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_wb/dec/src"
                "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/amr_wb/dec/include"
                "${_amr_src}/opencore/codecs_v2/audio/gsm_amr/common/dec/include")
            set_target_properties(crispasr_amrwb PROPERTIES POSITION_INDEPENDENT_CODE ON)

            target_link_libraries(crispasr-lib PRIVATE crispasr_amrnb crispasr_amrwb)
            set(_crispasr_amr_ok ON)
            message(STATUS "CrispASR: AMR via FetchContent CrispStrobe/opencore-amr v0.1.7")
        else()
            message(STATUS "CrispASR: AMR disabled (FetchContent: source layout not recognized)")
        endif()
    endif()

    if (_crispasr_amr_ok)
        target_compile_definitions(crispasr-lib PRIVATE CRISPASR_HAVE_AMR=1)
    else()
        message(STATUS "CrispASR: AMR disabled — opencore-amr not found (install libopencore-amrnb-dev libopencore-amrwb-dev, or set -DCRISPASR_AMR_FETCH=ON)")
    endif()
endif()

# ── AAC/M4A decode via fdk-aac (runtime dlopen, MIT-clean) ───────────────────
# The project binary never links against fdk-aac. At runtime, libfdk-aac.so is
# loaded via dlopen if installed — the user's choice and license responsibility.
# On Apple, AudioToolbox is preferred; on Windows, Media Foundation; on
# Android, NDK MediaCodec. This check is informational only.
if (NOT APPLE)
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        pkg_check_modules(_FDK_AAC_CHECK QUIET fdk-aac)
        if (_FDK_AAC_CHECK_FOUND)
            message(STATUS "CrispASR: AAC/M4A — libfdk-aac ${_FDK_AAC_CHECK_VERSION} found (will dlopen at runtime)")
        else()
            message(STATUS "CrispASR: AAC/M4A — libfdk-aac not found (install libfdk-aac-dev for M4A/AAC decode)")
        endif()
    endif()
endif()

# Apple: AudioToolbox ExtAudioFile decodes AAC / M4A / ALAC / CAF — formats the
# permissive miniaudio path can't — with no ffmpeg and no GPL (crispasr_audio.cpp
# falls back to it under __APPLE__). miniaudio's own CoreAudio backend dlopen's
# its frameworks, but our direct ExtAudioFile calls must be linked.
if (APPLE)
    # PUBLIC so consumers (tests, CLI, xcframework) transitively pick up the
    # framework linkage — PRIVATE breaks the Xcode/iOS all-targets build.
    target_link_libraries(crispasr-lib PUBLIC
        "-framework AudioToolbox" "-framework CoreFoundation" "-framework CoreAudio")
endif()

if (CMAKE_CXX_BYTE_ORDER STREQUAL "BIG_ENDIAN")
    target_compile_definitions(crispasr-lib PRIVATE CRISPASR_BIG_ENDIAN)
endif()

if (CRISPASR_EXTRA_FLAGS)
    target_compile_options(crispasr-lib PRIVATE ${CRISPASR_EXTRA_FLAGS})
endif()

find_package(Threads REQUIRED)
target_link_libraries(crispasr-lib PUBLIC ggml Threads::Threads audioseal crispasr-core snac ${CMAKE_DL_LIBS})

# `crispasr_cache.cpp` calls WinHTTP for HF auto-download on Windows. MSVC
# auto-links winhttp.lib via `#pragma comment(lib, "winhttp.lib")` inside
# the .cpp, but MinGW / GCC ignore that pragma — so any downstream test or
# executable that links libcrispasr.a (and therefore pulls in
# crispasr_cache.cpp.obj) needs winhttp on its link line. Propagate it as
# PUBLIC here so every consumer transitively picks it up, and individual
# tests don't have to remember to add it. Mirrors how Threads::Threads
# already flows through.
if (WIN32)
    # winhttp: HF auto-download in crispasr_cache.cpp
    # mfplat/mfreadwrite/mfuuid: Media Foundation AAC decoder in crispasr_audio.cpp
    # ole32: CoInitializeEx for MF
    # MSVC auto-links via #pragma comment(lib,...) but MinGW/Clang/lld ignores those.
    target_link_libraries(crispasr-lib PUBLIC winhttp mfplat mfreadwrite mfuuid ole32)
endif()

if (CRISPASR_COREML)
    target_link_libraries(crispasr-lib PRIVATE crispasr.coreml)
endif()

if (CRISPASR_OPENVINO)
    target_link_libraries(crispasr-lib PRIVATE crispasr.openvino)
endif()

if (CRISPASR_MKL)
    target_link_libraries(crispasr-lib PRIVATE MKL::MKL)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(crispasr-lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(crispasr-lib PRIVATE CRISPASR_SHARED CRISPASR_BUILD)
endif()

# Prefer a standalone libcblas (reference/Netlib BLAS splits the C interface
# into libcblas; find_package(BLAS) only finds the Fortran libblas, leaving
# cblas_sgemm undefined at link time — see cmake/FindCBLAS.cmake). Falls back
# to find_package(BLAS) for OpenBLAS/MKL/Accelerate, which bundle the cblas
# symbols. Sets BLAS_FOUND / BLAS_LIBRARIES for the caller. Pass QUIET to
# suppress the "found" message. Header (cblas.h) discovery stays at call sites.
macro(crispasr_prefer_cblas)
    find_package(CBLAS ${ARGN})
    if(CBLAS_FOUND)
        set(BLAS_FOUND ON)
        set(BLAS_LIBRARIES ${CBLAS_LIBRARIES})
    else()
        find_package(BLAS ${ARGN})
    endif()
endmacro()

# Cohere Transcribe library (Conformer encoder + transformer decoder)
add_library(cohere STATIC
            cohere-arch.h
            cohere.h
            cohere.cpp
            )
target_include_directories(cohere PUBLIC . ../include)
target_compile_features   (cohere PUBLIC cxx_std_20)
set_target_properties(cohere PROPERTIES POSITION_INDEPENDENT_CODE ON)
# BLAS for mel filterbank SGEMM (cblas_sgemm)
if(APPLE)
    # Accelerate framework provides cblas and vDSP; no external deps needed.
    # ACCELERATE_NEW_LAPACK silences the macOS 13.3+ cblas_sgemm deprecation warning.
    find_library(ACCELERATE_LIB Accelerate REQUIRED)
    target_link_libraries(cohere PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(cohere PRIVATE HAVE_BLAS ACCELERATE_NEW_LAPACK)
else()
    # If COHERE_MKL=ON the parent CMakeLists.txt sets BLA_VENDOR=Intel10_64lp
    # so this find_package(BLAS) call is steered toward MKL automatically.
    # cohere.cpp also does a bare `#include <cblas.h>` under HAVE_BLAS — gate on
    # the header being reachable (see crispasr-core note above re: MSYS2).
    crispasr_prefer_cblas()
    find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
    if(BLAS_FOUND AND CBLAS_INCLUDE_DIR)
        target_link_libraries(cohere PUBLIC ${BLAS_LIBRARIES})
        target_include_directories(cohere PRIVATE ${CBLAS_INCLUDE_DIR})
        target_compile_definitions(cohere PRIVATE HAVE_BLAS)
        if(COHERE_MKL)
            message(STATUS "cohere: linking MKL: ${BLAS_LIBRARIES}")
        endif()
    elseif(COHERE_MKL)
        message(FATAL_ERROR
            "COHERE_MKL=ON but Intel MKL was not found. "
            "Install MKL (e.g. apt install intel-mkl, or via oneAPI), "
            "or build without -DCOHERE_MKL=ON.")
    else()
        find_library(OPENBLAS_LIB openblas)
        if(OPENBLAS_LIB AND CBLAS_INCLUDE_DIR)
            target_link_libraries(cohere PUBLIC ${OPENBLAS_LIB})
            target_include_directories(cohere PRIVATE ${CBLAS_INCLUDE_DIR})
            target_compile_definitions(cohere PRIVATE HAVE_BLAS)
        endif()
    endif()
endif()
# FFT: self-contained Cooley-Tukey in cohere.cpp — no external FFT library required.
if(OpenMP_CXX_FOUND)
    target_link_libraries(cohere PUBLIC ggml ggml-base Threads::Threads OpenMP::OpenMP_CXX crispasr-core)
    target_compile_options(cohere PRIVATE -O3)
else()
    target_link_libraries(cohere PUBLIC ggml ggml-base Threads::Threads crispasr-core)
    target_compile_options(cohere PRIVATE -O3)
endif()
# Metal backend (macOS / Apple Silicon)
if(GGML_METAL)
    target_compile_definitions(cohere PRIVATE GGML_USE_METAL)
    target_link_libraries(cohere PUBLIC ggml-metal)
endif()
# CUDA backend
if(GGML_CUDA)
    target_compile_definitions(cohere PRIVATE GGML_USE_CUDA)
    target_link_libraries(cohere PUBLIC ggml-cuda)
endif()

# ---------------------------------------------------------------------------
# Canary 1B v2 — nvidia/canary-1b-v2 (FastConformer enc + Transformer dec, ASR + AST)
# ---------------------------------------------------------------------------
add_library(canary STATIC
            canary.h
            canary.cpp
            )
target_include_directories(canary PUBLIC . ../include)
target_compile_features   (canary PUBLIC cxx_std_17)
set_target_properties(canary PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries     (canary PUBLIC ggml ggml-base Threads::Threads crispasr-core)
if(NOT MSVC)
    target_compile_options(canary PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(canary PRIVATE GGML_USE_METAL)
    target_link_libraries(canary PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(canary PRIVATE GGML_USE_CUDA)
    target_link_libraries(canary PUBLIC ggml-cuda)
endif()

# ---------------------------------------------------------------------------
# Canary CTC aligner — auxiliary CTC model from inside canary-1b-v2.nemo,
# used for SentencePiece subword forced alignment of any transcript text.
# ---------------------------------------------------------------------------
add_library(canary_ctc STATIC
            canary_ctc.h
            canary_ctc.cpp
            )
target_include_directories(canary_ctc PUBLIC . ../include)
target_compile_features   (canary_ctc PUBLIC cxx_std_17)
set_target_properties(canary_ctc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries     (canary_ctc PUBLIC ggml ggml-base Threads::Threads crispasr-core)
if(NOT MSVC)
    target_compile_options(canary_ctc PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# LFM2 Audio — LiquidAI LFM2.5-Audio-1.5B-JP (FastConformer enc + LFM2 hybrid
# conv+attention backbone + depthformer). ASR + TTS in one model.
# ---------------------------------------------------------------------------
add_library(lfm2_audio STATIC
            lfm2_audio.h
            lfm2_audio.cpp
            )
target_include_directories(lfm2_audio PUBLIC . ../include)
target_compile_features   (lfm2_audio PUBLIC cxx_std_17)
set_target_properties(lfm2_audio PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries     (lfm2_audio PUBLIC ggml ggml-base Threads::Threads crispasr-core)
if(NOT MSVC)
    target_compile_options(lfm2_audio PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(lfm2_audio PRIVATE GGML_USE_METAL)
    target_link_libraries(lfm2_audio PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(lfm2_audio PRIVATE GGML_USE_CUDA)
    target_link_libraries(lfm2_audio PUBLIC ggml-cuda)
endif()

# ---------------------------------------------------------------------------
# OpenMP — optional shared dependency for parallelisable hot loops in the
# per-model encoders (currently cohere; granite linked as preparation for
# follow-up parallelisation). find_package runs once here so every model
# below can reference OpenMP_CXX_FOUND without re-probing.
find_package(OpenMP QUIET)

# ---------------------------------------------------------------------------
# crispasr-core — shared primitives used by multiple model implementations.
#
# This is the first piece of the model-level DRY refactor: one log-mel
# spectrogram helper (src/core/mel.{h,cpp}) that replaces what were 9
# copy-pasted mel functions across the model files. Additional helpers
# (gguf_loader, ffn, attention, greedy_decode) will land here in
# subsequent commits as we peel off more duplicated patterns.
# ---------------------------------------------------------------------------
add_library(crispasr-core STATIC
            core/mel.h
            core/mel.cpp
            core/ffn.h
            core/attention.h
            core/greedy_decode.h
            core/gguf_loader.h
            core/gguf_loader.cpp
            core/rvq.h
            core/rvq.cpp
            core/kaldi_fbank.h
            core/kaldi_fbank.cpp
            core/audio_resample.h
            core/audio_resample.cpp
            )
target_include_directories(crispasr-core PUBLIC . ../include)
# PLAN #51a: gguf_loader uses ggml_backend_buffer_init from
# ggml-backend-impl.h to register a custom mmap-owning CPU buffer. The
# header lives in ggml/src/ rather than ggml/include/, so we expose it
# here as a private include path for this target only.
target_include_directories(crispasr-core PRIVATE ../ggml/src)
target_compile_features   (crispasr-core PUBLIC cxx_std_17)
target_link_libraries     (crispasr-core PUBLIC ggml-base ggml)
# §176f: the STFT frame loop in core/mel.cpp is OpenMP-parallel (each frame's
# FFT is independent → bit-identical output). Link OpenMP when available; the
# pragma is #ifdef _OPENMP-guarded so non-OMP builds keep the serial path.
if(OpenMP_CXX_FOUND)
    target_link_libraries(crispasr-core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(NOT MSVC)
    target_compile_options(crispasr-core PRIVATE -O3)
endif()
set_target_properties(crispasr-core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# §176f: BLAS for mel filterbank SGEMM (cblas_sgemm) — affects all 40+ backends.
# Default ON. Consumers that link libcrispasr-core.a via raw -l flags rather than
# CMake target propagation (e.g. the cgo Go bindings) can't pull in the PRIVATE
# BLAS dependency and would hit 'undefined reference to cblas_sgemm'; they set
# -DCRISPASR_MEL_BLAS=OFF to compile the scalar mel projection (same output,
# slightly slower). MSVC has no portable cblas wired here.
option(CRISPASR_MEL_BLAS "Use cblas_sgemm for the mel filterbank projection (§176f)" ON)
if(CRISPASR_MEL_BLAS AND NOT MSVC)
    if(APPLE)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
        target_link_libraries     (crispasr-core PRIVATE ${ACCELERATE_LIB})
        target_compile_definitions(crispasr-core PRIVATE HAVE_BLAS ACCELERATE_NEW_LAPACK)
    else()
        # mel.cpp's BLAS path does a bare `#include <cblas.h>`. find_package(BLAS)
        # locates the library but NOT the header, and on some toolchains (MSYS2
        # OpenBLAS) cblas.h lives under an `openblas/` subdir not on the default
        # include path — so only enable HAVE_BLAS when cblas.h is actually
        # reachable, otherwise fall back to the scalar mel projection.
        crispasr_prefer_cblas(QUIET)
        find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
        if(BLAS_FOUND AND CBLAS_INCLUDE_DIR)
            target_link_libraries     (crispasr-core PRIVATE ${BLAS_LIBRARIES})
            target_include_directories(crispasr-core PRIVATE ${CBLAS_INCLUDE_DIR})
            target_compile_definitions(crispasr-core PRIVATE HAVE_BLAS)
        else()
            find_library(OPENBLAS_LIB openblas QUIET)
            if(OPENBLAS_LIB AND CBLAS_INCLUDE_DIR)
                target_link_libraries     (crispasr-core PRIVATE ${OPENBLAS_LIB})
                target_include_directories(crispasr-core PRIVATE ${CBLAS_INCLUDE_DIR})
                target_compile_definitions(crispasr-core PRIVATE HAVE_BLAS)
            else()
                message(STATUS "crispasr-core: cblas.h not found — using scalar mel projection")
            endif()
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# snac — Standalone SNAC 24 kHz codec decoder (shared by orpheus, mini-omni2)
# ---------------------------------------------------------------------------
add_library(snac STATIC core/snac.h core/snac.cpp)
set_target_properties(snac PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(snac PUBLIC .)
target_compile_features   (snac PUBLIC cxx_std_17)
target_link_libraries     (snac PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(snac PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# crispasr-llama-core — vendored llama.cpp built as a private STATIC lib
# so the chat / text-LLM C-ABI (src/chat.cpp, include/crispasr_chat.h) can
# call into it without exposing `llama.h` types in the public surface and
# without pulling a second copy of ggml.
#
# Sources live under `examples/talk-llama/` (they originally landed there
# only to power the SDL2-gated `crispasr-talk-llama` example). The example
# now consumes this lib too — one source set, two targets — see
# examples/talk-llama/CMakeLists.txt.
#
# Audio-LLM backends (voxtral / qwen3_asr / granite_speech / …) ship their
# own hand-rolled ggml graph builders and intentionally do NOT link this
# target; they stay free of the per-architecture llama-model code.
# ---------------------------------------------------------------------------
set(LLAMA_CORE_DIR ${CMAKE_SOURCE_DIR}/examples/talk-llama)
file(GLOB LLAMA_CORE_MODELS CONFIGURE_DEPENDS ${LLAMA_CORE_DIR}/models/*.cpp)
add_library(crispasr-llama-core STATIC
            ${LLAMA_CORE_DIR}/llama.cpp
            ${LLAMA_CORE_DIR}/llama-adapter.cpp
            ${LLAMA_CORE_DIR}/llama-arch.cpp
            ${LLAMA_CORE_DIR}/llama-batch.cpp
            ${LLAMA_CORE_DIR}/llama-chat.cpp
            ${LLAMA_CORE_DIR}/llama-context.cpp
            ${LLAMA_CORE_DIR}/llama-cparams.cpp
            ${LLAMA_CORE_DIR}/llama-grammar.cpp
            ${LLAMA_CORE_DIR}/llama-graph.cpp
            ${LLAMA_CORE_DIR}/llama-hparams.cpp
            ${LLAMA_CORE_DIR}/llama-impl.cpp
            ${LLAMA_CORE_DIR}/llama-io.cpp
            ${LLAMA_CORE_DIR}/llama-kv-cache.cpp
            ${LLAMA_CORE_DIR}/llama-kv-cache-iswa.cpp
            ${LLAMA_CORE_DIR}/llama-memory.cpp
            ${LLAMA_CORE_DIR}/llama-memory-recurrent.cpp
            ${LLAMA_CORE_DIR}/llama-memory-hybrid.cpp
            ${LLAMA_CORE_DIR}/llama-memory-hybrid-iswa.cpp
            ${LLAMA_CORE_DIR}/llama-mmap.cpp
            ${LLAMA_CORE_DIR}/llama-model-loader.cpp
            ${LLAMA_CORE_DIR}/llama-model-saver.cpp
            ${LLAMA_CORE_DIR}/llama-model.cpp
            ${LLAMA_CORE_DIR}/llama-quant.cpp
            ${LLAMA_CORE_DIR}/llama-sampler.cpp
            ${LLAMA_CORE_DIR}/llama-vocab.cpp
            ${LLAMA_CORE_DIR}/unicode.cpp
            ${LLAMA_CORE_DIR}/unicode-data.cpp
            ${LLAMA_CORE_MODELS})
# Headers live next to the .cpp files. PUBLIC so src/chat.cpp (built into
# libcrispasr) sees llama.h, but the header is NOT installed — only the
# `crispasr_chat_*` ABI in include/crispasr_chat.h is.
target_include_directories(crispasr-llama-core PUBLIC ${LLAMA_CORE_DIR})
target_compile_features   (crispasr-llama-core PUBLIC cxx_std_17)
target_link_libraries     (crispasr-llama-core PUBLIC ggml ggml-base Threads::Threads)
set_target_properties(crispasr-llama-core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    # Hide llama.cpp's C ABI from libcrispasr's export table — chat.cpp
    # calls into it INTERNALLY via the static-archive link, but external
    # consumers of libcrispasr.dylib only see the crispasr_chat_* surface.
    # Without this, hundreds of llama_* / ggml_* symbols leak and could
    # collide with another llama.cpp linked into the same process.
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET   hidden
    VISIBILITY_INLINES_HIDDEN ON)
if(NOT MSVC)
    target_compile_options(crispasr-llama-core PRIVATE -O3)
    # Vendored upstream code — silence the strict warnings we apply to
    # our own sources so a ggml/llama.cpp bump doesn't break CI lint.
    target_compile_options(crispasr-llama-core PRIVATE
        -Wno-cast-qual
        -Wno-unused-function
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-shadow
        -Wno-missing-declarations
        -Wno-missing-noreturn
        -Wno-pedantic
        -Wno-deprecated-declarations)
endif()
if(WIN32)
    # llama-mmap.cpp uses PrefetchVirtualMemory which needs Win 8.1+.
    target_compile_definitions(crispasr-llama-core PRIVATE _WIN32_WINNT=0x0602)
endif()

# ---------------------------------------------------------------------------
# Nemotron — nvidia/nemotron-3.5-asr-streaming-0.6b (Cache-Aware FastConformer + RNNT)
# ---------------------------------------------------------------------------
add_library(nemotron STATIC
            nemotron.h
            nemotron.cpp
            )
target_include_directories(nemotron PUBLIC . ../include)
target_compile_features   (nemotron PUBLIC cxx_std_17)
set_target_properties(nemotron PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries     (nemotron PUBLIC ggml ggml-base Threads::Threads crispasr-core)
if(NOT MSVC)
    target_compile_options(nemotron PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(nemotron PRIVATE GGML_USE_METAL)
    target_link_libraries(nemotron PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(nemotron PRIVATE GGML_USE_CUDA)
    target_link_libraries(nemotron PUBLIC ggml-cuda)
endif()
# §176d: Accelerate cblas_sgemv for LSTM predictor + joint head.
if(APPLE)
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(nemotron PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(nemotron PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()

# ---------------------------------------------------------------------------
# Parakeet TDT — nvidia/parakeet-tdt-0.6b-v3 (FastConformer encoder + TDT decoder)
# ---------------------------------------------------------------------------
add_library(parakeet STATIC
            parakeet.h
            parakeet.cpp
            )
target_include_directories(parakeet PUBLIC . ../include)
target_compile_features   (parakeet PUBLIC cxx_std_17)
set_target_properties(parakeet PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries     (parakeet PUBLIC ggml ggml-base Threads::Threads crispasr-core)
if(NOT MSVC)
    target_compile_options(parakeet PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(parakeet PRIVATE GGML_USE_METAL)
    target_link_libraries(parakeet PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(parakeet PRIVATE GGML_USE_CUDA)
    target_link_libraries(parakeet PUBLIC ggml-cuda)
endif()
if(APPLE)
    # §194 Accelerate cblas_sgemv for LSTM predictor + joint head (§176d).
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(parakeet PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(parakeet PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()

# ---------------------------------------------------------------------------
# Qwen3-ASR — Qwen/Qwen3-ASR-0.6B (Whisper-style audio encoder + Qwen3 LLM)
# ---------------------------------------------------------------------------
add_library(qwen3_asr STATIC
            qwen3_asr.h
            qwen3_asr.cpp
            )
target_include_directories(qwen3_asr PUBLIC . ../include)
target_compile_features   (qwen3_asr PUBLIC cxx_std_17)
set_target_properties(qwen3_asr PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries     (qwen3_asr PUBLIC ggml ggml-base Threads::Threads crispasr-core crisp_audio)
if(NOT MSVC)
    target_compile_options(qwen3_asr PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(qwen3_asr PRIVATE GGML_USE_METAL)
    target_link_libraries(qwen3_asr PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(qwen3_asr PRIVATE GGML_USE_CUDA)
    target_link_libraries(qwen3_asr PUBLIC ggml-cuda)
endif()

# ---------------------------------------------------------------------------
# FunASR — FunAudioLLM/Fun-ASR-Nano-2512 (SANM encoder + Transformer adaptor
#          + Qwen3-0.6B LLM decoder)
# ---------------------------------------------------------------------------
add_library(funasr STATIC
            funasr.h
            funasr.cpp
            )
target_include_directories(funasr PUBLIC . ../include)
target_compile_features   (funasr PUBLIC cxx_std_17)
set_target_properties(funasr PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries     (funasr PUBLIC ggml ggml-base Threads::Threads crispasr-core)
if(NOT MSVC)
    target_compile_options(funasr PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(funasr PRIVATE GGML_USE_METAL)
    target_link_libraries(funasr PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(funasr PRIVATE GGML_USE_CUDA)
    target_link_libraries(funasr PUBLIC ggml-cuda)
endif()

add_library(paraformer STATIC
            paraformer.h
            paraformer.cpp
            )
target_include_directories(paraformer PUBLIC . ../include)
target_compile_features   (paraformer PUBLIC cxx_std_17)
set_target_properties(paraformer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries     (paraformer PUBLIC ggml ggml-base Threads::Threads crispasr-core)
if(NOT MSVC)
    target_compile_options(paraformer PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(paraformer PRIVATE GGML_USE_METAL)
    target_link_libraries(paraformer PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(paraformer PRIVATE GGML_USE_CUDA)
    target_link_libraries(paraformer PUBLIC ggml-cuda)
endif()

# ---------------------------------------------------------------------------
# SenseVoice — FunAudioLLM/SenseVoiceSmall (encoder-only multi-task ASR:
#              transcript + LID + emotion + audio-event via a single CTC head,
#              non-AR, 15× faster than Whisper-Large, 50+ languages)
# ---------------------------------------------------------------------------
add_library(sensevoice STATIC
            sensevoice.h
            sensevoice.cpp
            )
target_include_directories(sensevoice PUBLIC . ../include)
target_compile_features   (sensevoice PUBLIC cxx_std_17)
set_target_properties(sensevoice PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries     (sensevoice PUBLIC ggml ggml-base Threads::Threads crispasr-core)
if(NOT MSVC)
    target_compile_options(sensevoice PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(sensevoice PRIVATE GGML_USE_METAL)
    target_link_libraries(sensevoice PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(sensevoice PRIVATE GGML_USE_CUDA)
    target_link_libraries(sensevoice PUBLIC ggml-cuda)
endif()

# ---------------------------------------------------------------------------
# Voxtral — mistralai/Voxtral-Mini-3B-2507 (Whisper encoder + Llama 3 LLM)
# ---------------------------------------------------------------------------
add_library(voxtral STATIC
            voxtral.h
            voxtral.cpp
            )
target_include_directories(voxtral PUBLIC . ../include)
target_compile_features   (voxtral PUBLIC cxx_std_17)
set_target_properties(voxtral PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries     (voxtral PUBLIC ggml ggml-base Threads::Threads crispasr-core)
if(NOT MSVC)
    target_compile_options(voxtral PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(voxtral PRIVATE GGML_USE_METAL)
    target_link_libraries(voxtral PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(voxtral PRIVATE GGML_USE_CUDA)
    target_link_libraries(voxtral PUBLIC ggml-cuda)
endif()

add_library(voxtral4b STATIC
            voxtral4b.h
            voxtral4b.cpp
            )
target_include_directories(voxtral4b PUBLIC . ../include)
target_compile_features   (voxtral4b PUBLIC cxx_std_17)
set_target_properties(voxtral4b PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries     (voxtral4b PUBLIC ggml ggml-base Threads::Threads crispasr-core)
if(NOT MSVC)
    target_compile_options(voxtral4b PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(voxtral4b PRIVATE GGML_USE_METAL)
    target_link_libraries(voxtral4b PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(voxtral4b PRIVATE GGML_USE_CUDA)
    target_link_libraries(voxtral4b PUBLIC ggml-cuda)
endif()

add_library(granite_speech STATIC
            granite_speech.h
            granite_speech.cpp
            )
target_include_directories(granite_speech PUBLIC . ../include)
target_compile_features   (granite_speech PUBLIC cxx_std_17)
set_target_properties(granite_speech PROPERTIES POSITION_INDEPENDENT_CODE ON)
# OpenMP: linked when available so future `#pragma omp parallel for` on the
# encoder hot loops can land without a CMake change. Adding the pragmas
# themselves is a follow-up — see TODO.md → "Per-model follow-ups → granite".
if(OpenMP_CXX_FOUND)
    target_link_libraries (granite_speech PUBLIC ggml ggml-base Threads::Threads OpenMP::OpenMP_CXX crispasr-core)
else()
    target_link_libraries (granite_speech PUBLIC ggml ggml-base Threads::Threads crispasr-core)
endif()
if(NOT MSVC)
    target_compile_options(granite_speech PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(granite_speech PRIVATE GGML_USE_METAL)
    target_link_libraries(granite_speech PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(granite_speech PRIVATE GGML_USE_CUDA)
    target_link_libraries(granite_speech PUBLIC ggml-cuda)
endif()
# §176d: Accelerate cblas_sgemm for cpu_linear.
if(APPLE)
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(granite_speech PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(granite_speech PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()

# ---------------------------------------------------------------------------
# granite_nle — runtime for ibm-granite/granite-speech-4.1-2b-nar
# (NLENARDecoder: non-autoregressive LLM refiner). Forward-pass functions
# are scaffolded as stubs returning nullptr; the model loads and the
# CTC tokenizer is wired up. See src/granite_nle.cpp for the roadmap.

add_library(granite_nle STATIC
            granite_nle.h
            granite_nle.cpp
            )
target_include_directories(granite_nle PUBLIC . ../include)
target_compile_features   (granite_nle PUBLIC cxx_std_17)
set_target_properties(granite_nle PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(OpenMP_CXX_FOUND)
    target_link_libraries (granite_nle PUBLIC ggml ggml-base Threads::Threads OpenMP::OpenMP_CXX crispasr-core)
else()
    target_link_libraries (granite_nle PUBLIC ggml ggml-base Threads::Threads crispasr-core)
endif()
if(NOT MSVC)
    target_compile_options(granite_nle PRIVATE -O3)
endif()
if(GGML_METAL)
    target_compile_definitions(granite_nle PRIVATE GGML_USE_METAL)
    target_link_libraries(granite_nle PUBLIC ggml-metal)
endif()
if(GGML_CUDA)
    target_compile_definitions(granite_nle PRIVATE GGML_USE_CUDA)
    target_link_libraries(granite_nle PUBLIC ggml-cuda)
endif()

# ---------------------------------------------------------------------------
# wav2vec2-ggml — Wav2Vec2ForCTC encoder (GGUF, ggml-based)
# ---------------------------------------------------------------------------
# glm-asr — GLM-ASR-Nano (zai-org/GLM-ASR-Nano-2512)
# ---------------------------------------------------------------------------
add_library(glm-asr STATIC glm_asr.h glm_asr.cpp)
set_target_properties(glm-asr PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(glm-asr PUBLIC .)
target_compile_features   (glm-asr PUBLIC cxx_std_17)
target_link_libraries     (glm-asr PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(glm-asr PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# mini-omni2 — gpt-omni/mini-omni2 (Whisper-small + Qwen2-0.5B)
# ---------------------------------------------------------------------------
add_library(mini-omni2 STATIC mini_omni2.h mini_omni2.cpp)
set_target_properties(mini-omni2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mini-omni2 PUBLIC .)
target_compile_features   (mini-omni2 PUBLIC cxx_std_17)
target_link_libraries     (mini-omni2 PUBLIC crispasr-core snac ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(mini-omni2 PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# kyutai-stt — Kyutai STT (stt-1b-en_fr, stt-2.6b-en)
# ---------------------------------------------------------------------------
add_library(kyutai-stt STATIC kyutai_stt.h kyutai_stt.cpp)
set_target_properties(kyutai-stt PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kyutai-stt PUBLIC .)
target_compile_features   (kyutai-stt PUBLIC cxx_std_17)
target_link_libraries     (kyutai-stt PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(kyutai-stt PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# moonshine — Moonshine ASR (UsefulSensors)
# Tokenizer is its own static lib so both moonshine and moonshine_streaming
# can depend on it without bundling moonshine-tokenizer.o into both archives.
# Without the split, libtool -static packs combined.a with duplicate symbols
# and the iOS xcframework link fails.
# ---------------------------------------------------------------------------
add_library(moonshine_tokenizer STATIC moonshine-tokenizer.h moonshine-tokenizer.cpp)
set_target_properties(moonshine_tokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(moonshine_tokenizer PUBLIC .)
target_compile_features   (moonshine_tokenizer PUBLIC cxx_std_17)
target_link_libraries     (moonshine_tokenizer PUBLIC ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(moonshine_tokenizer PRIVATE -O3)
endif()

add_library(moonshine STATIC moonshine.h moonshine.cpp moonshine-impl.h
            moonshine-audio.h)
set_target_properties(moonshine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(moonshine PUBLIC .)
target_compile_features   (moonshine PUBLIC cxx_std_17)
target_link_libraries     (moonshine PUBLIC ggml ggml-base ggml-cpu moonshine_tokenizer)
if(NOT MSVC)
    target_compile_options(moonshine PRIVATE -O3)
endif()

add_library(moonshine_streaming STATIC moonshine_streaming.h moonshine_streaming.cpp)
set_target_properties(moonshine_streaming PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(moonshine_streaming PUBLIC .)
target_compile_features   (moonshine_streaming PUBLIC cxx_std_17)
target_link_libraries     (moonshine_streaming PUBLIC ggml ggml-base ggml-cpu moonshine_tokenizer)
if(NOT MSVC)
    target_compile_options(moonshine_streaming PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# gemma4_e2b — Google Gemma-4-E2B (USM Conformer + Gemma4 LLM)
# ---------------------------------------------------------------------------
add_library(gemma4_e2b STATIC gemma4_e2b.h gemma4_e2b.cpp)
set_target_properties(gemma4_e2b PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gemma4_e2b PUBLIC .)
target_compile_features   (gemma4_e2b PUBLIC cxx_std_17)
target_link_libraries     (gemma4_e2b PUBLIC ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(gemma4_e2b PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# mimo_tokenizer — XiaomiMiMo/MiMo-Audio-Tokenizer (encoder-only, PCM → RVQ)
# ---------------------------------------------------------------------------
add_library(mimo_tokenizer STATIC mimo_tokenizer.h mimo_tokenizer.cpp)
set_target_properties(mimo_tokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mimo_tokenizer PUBLIC .)
target_compile_features   (mimo_tokenizer PUBLIC cxx_std_17)
target_link_libraries     (mimo_tokenizer PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(mimo_tokenizer PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# mimo_asr — XiaomiMiMo/MiMo-V2.5-ASR (input_local_transformer + Qwen2 LLM)
# ---------------------------------------------------------------------------
add_library(mimo_asr STATIC mimo_asr.h mimo_asr.cpp)
set_target_properties(mimo_asr PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mimo_asr PUBLIC .)
target_compile_features   (mimo_asr PUBLIC cxx_std_17)
target_link_libraries     (mimo_asr PUBLIC crispasr-core mimo_tokenizer ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(mimo_asr PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# moss_audio — OpenMOSS-Team/MOSS-Audio-4B-Instruct (Whisper encoder + DeepStack + Qwen3 LLM)
# ---------------------------------------------------------------------------
add_library(moss_audio STATIC moss_audio.h moss_audio.cpp)
set_target_properties(moss_audio PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(moss_audio PUBLIC .)
target_compile_features   (moss_audio PUBLIC cxx_std_17)
target_link_libraries     (moss_audio PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(moss_audio PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# qwen3_tts — Qwen/Qwen3-TTS (Qwen3 talker + 16-codebook RVQ output)
# ---------------------------------------------------------------------------
add_library(qwen3_tts STATIC qwen3_tts.h qwen3_tts.cpp)
set_target_properties(qwen3_tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(qwen3_tts PUBLIC .)
target_compile_features   (qwen3_tts PUBLIC cxx_std_17)
target_link_libraries     (qwen3_tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(qwen3_tts PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# orpheus — canopylabs/orpheus-3b-0.1-ft (Llama-3.2-3B + SNAC 24 kHz codec)
# ---------------------------------------------------------------------------
add_library(orpheus STATIC orpheus.h orpheus.cpp)
set_target_properties(orpheus PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(orpheus PUBLIC .)
target_compile_features   (orpheus PUBLIC cxx_std_17)
target_link_libraries     (orpheus PUBLIC crispasr-core snac ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(orpheus PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# audioseal — Meta AudioSeal watermark generator + detector (MIT, ~5M params)
# ---------------------------------------------------------------------------
add_library(audioseal STATIC audioseal.h audioseal.cpp)
set_target_properties(audioseal PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(audioseal PUBLIC .)
target_compile_features   (audioseal PUBLIC cxx_std_17)
target_link_libraries     (audioseal PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(audioseal PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# chatterbox — ResembleAI/chatterbox (MIT, 520M Llama AR + CFM + HiFTNet)
# ---------------------------------------------------------------------------
add_library(chatterbox STATIC chatterbox.h chatterbox.cpp chatterbox_s3gen.h chatterbox_s3gen.cpp
            chatterbox_ve.h chatterbox_ve.cpp
            chatterbox_s3tok.h chatterbox_s3tok.cpp
            chatterbox_campplus.h chatterbox_campplus.cpp)
set_target_properties(chatterbox PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(chatterbox PUBLIC .)
target_compile_features   (chatterbox PUBLIC cxx_std_17)
target_link_libraries     (chatterbox PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(chatterbox PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# tada-tts — HumeAI/tada-3b-ml (Llama-3.2-3B + FM head + TADA codec)
# ---------------------------------------------------------------------------
add_library(tada-tts STATIC tada_tts.h tada_tts.cpp)
set_target_properties(tada-tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tada-tts PUBLIC .)
target_compile_features   (tada-tts PUBLIC cxx_std_17)
target_link_libraries     (tada-tts PUBLIC tada-codec crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(tada-tts PRIVATE -O3)
endif()

# tada-codec — TADA codec decoder (local-attention + DAC upsampler)
add_library(tada-codec STATIC tada_codec.h tada_codec.cpp)
set_target_properties(tada-codec PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tada-codec PUBLIC .)
target_compile_features   (tada-codec PUBLIC cxx_std_17)
target_link_libraries     (tada-codec PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(tada-codec PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# indextts — IndexTeam/IndexTTS-1.5 (GPT-2 AR + BigVGAN vocoder)
# ---------------------------------------------------------------------------
add_library(indextts STATIC indextts.h indextts.cpp indextts_voc.h indextts_voc.cpp)
set_target_properties(indextts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(indextts PUBLIC .)
target_compile_features   (indextts PUBLIC cxx_std_17)
target_link_libraries     (indextts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(indextts PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# m2m100 — Facebook M2M-100 multilingual machine translation
# ---------------------------------------------------------------------------
add_library(m2m100 STATIC m2m100.h m2m100.cpp)
set_target_properties(m2m100 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(m2m100 PUBLIC .)
target_compile_features   (m2m100 PUBLIC cxx_std_17)
target_link_libraries     (m2m100 PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(m2m100 PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# t5_translate — T5 encoder-decoder translation (MADLAD-400, mT5, Flan-T5)
# ---------------------------------------------------------------------------
add_library(t5_translate STATIC t5_translate.h t5_translate.cpp)
set_target_properties(t5_translate PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(t5_translate PUBLIC .)
target_compile_features   (t5_translate PUBLIC cxx_std_17)
target_link_libraries     (t5_translate PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(t5_translate PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# omniasr — Facebook OmniASR-CTC (wav2vec2-style, 1600+ languages)
# ---------------------------------------------------------------------------
add_library(omniasr STATIC omniasr.h omniasr.cpp)
set_target_properties(omniasr PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(omniasr PUBLIC .)
target_compile_features   (omniasr PUBLIC cxx_std_17)
target_link_libraries     (omniasr PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(OpenMP_CXX_FOUND)
    target_link_libraries(omniasr PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# ecapa-lid — ECAPA-TDNN language identification (107 languages, 43 MB)
# ---------------------------------------------------------------------------
add_library(ecapa-lid STATIC ecapa_lid.h ecapa_lid.cpp)
set_target_properties(ecapa-lid PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ecapa-lid PUBLIC .)
target_compile_features   (ecapa-lid PUBLIC cxx_std_17)
target_link_libraries     (ecapa-lid PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(APPLE)
    # Accelerate cblas_sgemm for the ASP pooling head (the CPU-scalar hotspot).
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(ecapa-lid PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(ecapa-lid PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()
if(NOT MSVC)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ecapa-lid PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# ---------------------------------------------------------------------------
# titanet — TitaNet-Large speaker verification + speaker profile DB
# ---------------------------------------------------------------------------
add_library(titanet STATIC titanet.h titanet.cpp speaker_db.h speaker_db.cpp)
set_target_properties(titanet PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(titanet PUBLIC .)
target_compile_features   (titanet PUBLIC cxx_std_17)
target_link_libraries     (titanet PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(APPLE)
    # Accelerate cblas_sgemm for the (CPU-scalar) speaker-embedding forward.
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(titanet PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(titanet PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()

# ---------------------------------------------------------------------------
# vibevoice — Microsoft VibeVoice-ASR (σ-VAE tokenizers + Qwen2 LM)
# ---------------------------------------------------------------------------
add_library(vibevoice STATIC vibevoice.h vibevoice.cpp)
set_target_properties(vibevoice PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(vibevoice PUBLIC .)
target_compile_features   (vibevoice PUBLIC cxx_std_17)
target_link_libraries     (vibevoice PUBLIC crispasr-core ggml ggml-base ggml-cpu)

# ---------------------------------------------------------------------------
# kugelaudio — KugelAudio-0-Open TTS (Qwen2.5-7B + DiT diffusion + VAE)
# ---------------------------------------------------------------------------
add_library(kugelaudio STATIC kugelaudio.h kugelaudio.cpp)
set_target_properties(kugelaudio PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kugelaudio PUBLIC .)
target_compile_features   (kugelaudio PUBLIC cxx_std_17)
target_link_libraries     (kugelaudio PUBLIC crispasr-core ggml ggml-base ggml-cpu)

# ---------------------------------------------------------------------------
# voxcpm2_tts — openbmb/VoxCPM2 (diffusion AR TTS, 30 languages, 48kHz)
# ---------------------------------------------------------------------------
add_library(voxcpm2_tts STATIC voxcpm2_tts.h voxcpm2_tts.cpp)
set_target_properties(voxcpm2_tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(voxcpm2_tts PUBLIC .)
target_compile_features   (voxcpm2_tts PUBLIC cxx_std_17)
target_link_libraries     (voxcpm2_tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(OpenMP_FOUND)
    target_link_libraries     (voxcpm2_tts PUBLIC OpenMP::OpenMP_CXX)
endif()
if(APPLE)
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(voxcpm2_tts PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(voxcpm2_tts PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()
if(NOT MSVC)
    target_compile_options(voxcpm2_tts PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# cosyvoice3_tts — FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (Qwen2-0.5B LLM +
# DiT-CFM flow + HiFi-GAN-iSTFT vocoder). Phase 2 ships the LLM only.
# ---------------------------------------------------------------------------
add_library(cosyvoice3_tts STATIC cosyvoice3_tts.h cosyvoice3_tts.cpp)
set_target_properties(cosyvoice3_tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(cosyvoice3_tts PUBLIC .)
target_compile_features   (cosyvoice3_tts PUBLIC cxx_std_17)
target_link_libraries     (cosyvoice3_tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(cosyvoice3_tts PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# kokoro — hexgrad/Kokoro-82M + yl4579/StyleTTS2-LJSpeech (iSTFTNet TTS)
# ---------------------------------------------------------------------------
# phonemizer.cpp lives here — only kokoro calls phonemize_builtin_*.
add_library(kokoro STATIC kokoro.h kokoro.cpp phonemizer.cpp phonemizer.h)
set_target_properties(kokoro PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kokoro PUBLIC .)
target_compile_features   (kokoro PUBLIC cxx_std_17)
target_link_libraries     (kokoro PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(kokoro PRIVATE -O3)
endif()

# Optional libespeak-ng linkage. When ON we replace the popen("espeak-ng …")
# shell-out with a direct espeak_TextToPhonemes() call (saves ~30–50 ms per
# call, removes the shell-quoting surface, works in sandboxed processes).
#
# NOTE: espeak-ng is GPLv3. Static/dynamic linking at build time makes the
# binary a GPL derivative. The default mode (AUTO) therefore uses **dlopen**
# at runtime — the binary ships MIT-clean and loads libespeak-ng only if the
# user has it installed (or auto-downloaded). This follows the same pattern
# as GIMP/VLC/FFmpeg for optional GPL plugins.
#
# AUTO (default): try system headers for compile-time API; load the shared
#   library at runtime via dlopen. Falls back to popen(espeak-ng) subprocess.
# ON:  link libespeak-ng at build time (makes the binary GPLv3).
# OFF: disable completely — popen(espeak-ng) only.
set(CRISPASR_WITH_ESPEAK_NG "AUTO" CACHE STRING
    "espeak-ng phonemization: AUTO=dlopen at runtime, ON=link at build time (GPLv3!), OFF=popen only")
set_property(CACHE CRISPASR_WITH_ESPEAK_NG PROPERTY STRINGS "AUTO" "ON" "OFF")

if(NOT CRISPASR_WITH_ESPEAK_NG STREQUAL "OFF")
    set(_espeak_found FALSE)
    set(_espeak_include "")
    set(_espeak_lib "")

    if(CRISPASR_WITH_ESPEAK_NG STREQUAL "ON")
        # --- Explicit build-time linking (user accepts GPLv3) ---
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(ESPEAK_NG QUIET espeak-ng)
            if(ESPEAK_NG_FOUND)
                set(_espeak_found TRUE)
                set(_espeak_include ${ESPEAK_NG_INCLUDE_DIRS})
                set(_espeak_lib     ${ESPEAK_NG_LIBRARIES})
                set(_espeak_libdir  ${ESPEAK_NG_LIBRARY_DIRS})
            endif()
        endif()
        if(NOT _espeak_found)
            find_path(ESPEAK_NG_INCLUDE_DIR espeak-ng/speak_lib.h
                      HINTS /opt/homebrew/include /usr/local/include /usr/include)
            find_library(ESPEAK_NG_LIBRARY NAMES espeak-ng
                         HINTS /opt/homebrew/lib /usr/local/lib /usr/lib)
            if(ESPEAK_NG_INCLUDE_DIR AND ESPEAK_NG_LIBRARY)
                set(_espeak_found TRUE)
                set(_espeak_include ${ESPEAK_NG_INCLUDE_DIR})
                set(_espeak_lib     ${ESPEAK_NG_LIBRARY})
            endif()
        endif()
        if(NOT _espeak_found)
            message(FATAL_ERROR "CRISPASR_WITH_ESPEAK_NG=ON but libespeak-ng not found. Install espeak-ng or set AUTO/OFF.")
        endif()
    endif()

    if(_espeak_found)
        # Build-time linking path (GPLv3)
        target_compile_definitions(kokoro PRIVATE CRISPASR_HAVE_ESPEAK_NG=1)
        target_include_directories(kokoro PRIVATE ${_espeak_include})
        if(_espeak_libdir)
            target_link_directories(kokoro PRIVATE ${_espeak_libdir})
        endif()
        target_link_libraries(kokoro PRIVATE ${_espeak_lib})
        message(STATUS "kokoro: libespeak-ng linked (in-process phonemizer) — binary is GPLv3!")
    else()
        # AUTO mode: dlopen at runtime (MIT-clean binary).
        # Define CRISPASR_ESPEAK_DLOPEN so the runtime uses dlopen/dlsym
        # instead of direct function calls.
        target_compile_definitions(kokoro PRIVATE CRISPASR_ESPEAK_DLOPEN=1)
        message(STATUS "kokoro: espeak-ng via dlopen at runtime (MIT-clean)")
    endif()
endif()

# ---------------------------------------------------------------------------
# outetts — OuteAI/OuteTTS LLM-based TTS + WavTokenizer codec (native ggml runtime)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/outetts.cpp")
add_library(outetts STATIC outetts.h outetts.cpp outetts_wavtok.h outetts_wavtok.cpp)
set_target_properties(outetts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(outetts PUBLIC .)
target_compile_features   (outetts PUBLIC cxx_std_17)
target_link_libraries     (outetts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(outetts PRIVATE -O3)
endif()
endif()

# ---------------------------------------------------------------------------
# f5-tts — SWivid/F5-TTS DiT-based flow matching TTS (native ggml runtime)
# ---------------------------------------------------------------------------
add_library(f5-tts STATIC f5_tts.h f5_tts.cpp)
set_target_properties(f5-tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(f5-tts PUBLIC .)
target_compile_features   (f5-tts PUBLIC cxx_std_17)
target_link_libraries     (f5-tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(APPLE)
    # Accelerate provides cblas_sgemm for the Vocos vocoder pointwise Linears
    # (PLAN §182). Falls back to the scalar loops on non-Apple.
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(f5-tts PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(f5-tts PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()
if(NOT MSVC)
    target_compile_options(f5-tts PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# csm-tts — sesame/csm-1b (1B Llama backbone + depth decoder + Mimi codec)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/csm_tts.cpp")
    add_library(csm-tts STATIC csm_tts.h csm_tts.cpp)
    set_target_properties(csm-tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(csm-tts PUBLIC .)
    target_compile_features   (csm-tts PUBLIC cxx_std_17)
    target_link_libraries     (csm-tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
    if(NOT MSVC)
        target_compile_options(csm-tts PRIVATE -O3)
    endif()
endif()

# ---------------------------------------------------------------------------
# pocket-tts — kyutai/pocket-tts (100M continuous-latent AR TTS, 24 kHz)
# ---------------------------------------------------------------------------
add_library(pocket-tts STATIC pocket_tts.h pocket_tts.cpp)
set_target_properties(pocket-tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pocket-tts PUBLIC .)
target_compile_features   (pocket-tts PUBLIC cxx_std_17)
target_link_libraries     (pocket-tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(pocket-tts PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# parler-tts — Parler TTS prompt-conditioned TTS (T5 + MusicGen decoder + DAC)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/parler_tts.cpp")
add_library(parler-tts STATIC parler_tts.h parler_tts.cpp)
set_target_properties(parler-tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(parler-tts PUBLIC .)
target_compile_features   (parler-tts PUBLIC cxx_std_17)
target_link_libraries     (parler-tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(parler-tts PRIVATE -O3)
endif()
endif()

# ---------------------------------------------------------------------------
# zonos-tts — Zyphra/Zonos-v0.1-transformer AR TTS + DAC 44.1 kHz codec
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/zonos_tts.cpp")
add_library(zonos-tts STATIC zonos_tts.h zonos_tts.cpp)
set_target_properties(zonos-tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(zonos-tts PUBLIC .)
target_compile_features   (zonos-tts PUBLIC cxx_std_17)
target_link_libraries     (zonos-tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(zonos-tts PRIVATE -O3)
endif()
endif()

# ---------------------------------------------------------------------------
# bark-tts — Suno Bark TTS (native ggml runtime)
# ---------------------------------------------------------------------------
add_library(bark-tts STATIC bark_tts.h bark_tts.cpp)
set_target_properties(bark-tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(bark-tts PUBLIC .)
target_compile_features   (bark-tts PUBLIC cxx_std_17)
target_link_libraries     (bark-tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(bark-tts PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# dia-tts — Nari Labs Dia TTS (native ggml runtime)
# ---------------------------------------------------------------------------
add_library(dia-tts STATIC dia_tts.h dia_tts.cpp)
set_target_properties(dia-tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(dia-tts PUBLIC .)
target_compile_features   (dia-tts PUBLIC cxx_std_17)
target_link_libraries     (dia-tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(dia-tts PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# fastpitch-tts — NVIDIA FastPitch parallel TTS (native ggml runtime)
# ---------------------------------------------------------------------------
add_library(fastpitch-tts STATIC fastpitch_tts.h fastpitch_tts.cpp)
set_target_properties(fastpitch-tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fastpitch-tts PUBLIC .)
target_compile_features   (fastpitch-tts PUBLIC cxx_std_17)
target_link_libraries     (fastpitch-tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(fastpitch-tts PRIVATE -O3)
endif()

# bert-encoder — BERT encoder for MeloTTS conditioning
# ---------------------------------------------------------------------------
add_library(bert-encoder STATIC bert_encoder.h bert_encoder.cpp)
set_target_properties(bert-encoder PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(bert-encoder PUBLIC .)
target_compile_features   (bert-encoder PUBLIC cxx_std_17)
target_link_libraries     (bert-encoder PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(bert-encoder PRIVATE -O3)
endif()

# melotts — myshell-ai/MeloTTS VITS2 TTS (native ggml runtime)
# ---------------------------------------------------------------------------
add_library(melotts STATIC melotts.h melotts.cpp)
set_target_properties(melotts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(melotts PUBLIC .)
target_compile_features   (melotts PUBLIC cxx_std_17)
target_link_libraries     (melotts PUBLIC crispasr-core ggml ggml-base ggml-cpu bert-encoder)
if(NOT MSVC)
    target_compile_options(melotts PRIVATE -O3)
endif()
if(APPLE)
    # §176d Accelerate cblas_sgemm for QKV projections + per-head attention matmuls.
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(melotts PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(melotts PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()

# openvoice2 — myshell-ai/OpenVoice V2 Tone Color Converter (voice cloning)
# ---------------------------------------------------------------------------
add_library(openvoice2 STATIC openvoice2.h openvoice2.cpp)
set_target_properties(openvoice2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(openvoice2 PUBLIC .)
target_compile_features   (openvoice2 PUBLIC cxx_std_17)
target_link_libraries     (openvoice2 PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(APPLE)
    # Accelerate cblas_sgemm for the (CPU-scalar) WaveNet voice-conversion forward.
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(openvoice2 PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(openvoice2 PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()
if(NOT MSVC)
    target_compile_options(openvoice2 PRIVATE -O3)
endif()

# piper-tts — rhasspy/piper VITS TTS (native ggml runtime)
# ---------------------------------------------------------------------------
add_library(piper-tts STATIC piper_tts.h piper_tts.cpp)
set_target_properties(piper-tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(piper-tts PUBLIC .)
target_compile_features   (piper-tts PUBLIC cxx_std_17)
target_link_libraries     (piper-tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(piper-tts PRIVATE -O3)
endif()
if(APPLE)
    # §176d Accelerate cblas_sgemm for QKV projections + per-head attention matmuls.
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(piper-tts PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(piper-tts PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()

# Piper reuses the same espeak-ng integration as kokoro.
if(_espeak_found)
    target_compile_definitions(piper-tts PRIVATE CRISPASR_HAVE_ESPEAK_NG=1)
    target_include_directories(piper-tts PRIVATE ${_espeak_include})
    if(_espeak_libdir)
        target_link_directories(piper-tts PRIVATE ${_espeak_libdir})
    endif()
    target_link_libraries(piper-tts PRIVATE ${_espeak_lib})
    message(STATUS "piper-tts: libespeak-ng linked — binary is GPLv3!")
elseif(NOT CRISPASR_WITH_ESPEAK_NG STREQUAL "OFF")
    target_compile_definitions(piper-tts PRIVATE CRISPASR_ESPEAK_DLOPEN=1)
    message(STATUS "piper-tts: espeak-ng via dlopen at runtime (MIT-clean)")
endif()

# speecht5-tts — Microsoft SpeechT5 TTS (encoder-decoder mel + HiFi-GAN)
# ---------------------------------------------------------------------------
add_library(speecht5-tts STATIC speecht5_tts.h speecht5_tts.cpp)
set_target_properties(speecht5-tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(speecht5-tts PUBLIC .)
target_compile_features   (speecht5-tts PUBLIC cxx_std_17)
target_link_libraries     (speecht5-tts PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(speecht5-tts PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# firered-vad — FireRedVAD (DFSMN voice activity detection)
# ---------------------------------------------------------------------------
add_library(firered-vad STATIC firered_vad.h firered_vad.cpp)
set_target_properties(firered-vad PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(firered-vad PUBLIC .)
target_compile_features   (firered-vad PUBLIC cxx_std_17)
target_link_libraries     (firered-vad PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(APPLE)
    # §193 Accelerate cblas_sgemm for DFSMN linear layers (§176d).
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(firered-vad PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(firered-vad PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()

# marblenet-vad — NVIDIA MarbleNet VAD (1D separable CNN, 91.5K params)
add_library(marblenet-vad STATIC marblenet_vad.h marblenet_vad.cpp)
set_target_properties(marblenet-vad PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(marblenet-vad PUBLIC .)
target_compile_features   (marblenet-vad PUBLIC cxx_std_17)
target_link_libraries     (marblenet-vad PUBLIC ggml ggml-base ggml-cpu)

# crispasr-vad-encdec — Whisper-encoder + TransformerDecoder VAD
add_library(crispasr-vad-encdec STATIC crispasr_vad_encdec.h crispasr_vad_encdec.cpp)
set_target_properties(crispasr-vad-encdec PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(crispasr-vad-encdec PUBLIC .)
target_compile_features   (crispasr-vad-encdec PUBLIC cxx_std_17)
target_link_libraries     (crispasr-vad-encdec PUBLIC ggml ggml-base ggml-cpu)

# ---------------------------------------------------------------------------
# firered-lid — FireRedLID (120-language identification)
# ---------------------------------------------------------------------------
add_library(firered-lid STATIC firered_lid.h firered_lid.cpp)
set_target_properties(firered-lid PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(firered-lid PUBLIC .)
target_compile_features   (firered-lid PUBLIC cxx_std_17)
target_link_libraries     (firered-lid PUBLIC firered-asr crispasr-core ggml ggml-base ggml-cpu)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# crisp_lid — shared text LID library (fastText + CLD3).
# Prefer shared library from crisp_lid/; fall back to local copies.
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../crisp_lid/CMakeLists.txt" AND NOT TARGET crisp_lid)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../crisp_lid
                     ${CMAKE_BINARY_DIR}/crisp_lid)
    message(STATUS "crispasr: using shared crisp_lid library")
else()
    # Local fallback: lid-fasttext
    add_library(lid-fasttext STATIC lid_fasttext.h lid_fasttext.cpp)
    set_target_properties(lid-fasttext PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(lid-fasttext PUBLIC .)
    target_compile_features   (lid-fasttext PUBLIC cxx_std_17)
    target_link_libraries     (lid-fasttext PUBLIC crispasr-core ggml ggml-base ggml-cpu)
    if(NOT MSVC)
        target_compile_options(lid-fasttext PRIVATE -O3)
    endif()
endif()

# ---------------------------------------------------------------------------
# firered-asr — FireRedASR2-AED (FireRedTeam/FireRedASR2-AED)
# ---------------------------------------------------------------------------
add_library(firered-asr STATIC firered_asr.h firered_asr.cpp)
set_target_properties(firered-asr PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(firered-asr PUBLIC .)
target_compile_features   (firered-asr PUBLIC cxx_std_17)
target_link_libraries     (firered-asr PUBLIC crispasr-core ggml ggml-base ggml-cpu)
# firered_asr.cpp's CPU decoder/encoder matmuls (cpu_matmul_bt) are
# `#pragma omp parallel for`, but the target never linked OpenMP — so the
# pragmas were no-ops and the AED beam decoder ran single-threaded
# (~1.4 s/token at beam 5). Link OpenMP so the matmuls actually parallelize.
if (OpenMP_CXX_FOUND)
    target_link_libraries(firered-asr PUBLIC OpenMP::OpenMP_CXX)
endif()
if(NOT MSVC)
    target_compile_options(firered-asr PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
add_library(wav2vec2-ggml STATIC wav2vec2-ggml.h wav2vec2-ggml.cpp)
target_include_directories(wav2vec2-ggml PUBLIC .)
target_compile_features   (wav2vec2-ggml PUBLIC cxx_std_17)
target_link_libraries     (wav2vec2-ggml PUBLIC ggml ggml-base ggml-cpu)
if(OpenMP_CXX_FOUND)
    target_link_libraries(wav2vec2-ggml PUBLIC OpenMP::OpenMP_CXX)
endif()
set_target_properties(wav2vec2-ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(wav2vec2-ggml PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# pyannote-seg — native pyannote-segmentation-3.0 (GGUF, CPU-only)
# ---------------------------------------------------------------------------
add_library(pyannote-seg STATIC pyannote_seg.h pyannote_seg.cpp)
set_target_properties(pyannote-seg PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pyannote-seg PUBLIC .)
target_compile_features   (pyannote-seg PUBLIC cxx_std_17)
target_link_libraries     (pyannote-seg PUBLIC crispasr-core ggml ggml-base ggml-cpu)
set_target_properties(pyannote-seg PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(pyannote-seg PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# silero-lid — native Silero Language Classifier 95 (GGUF, ggml-based)
# ---------------------------------------------------------------------------
add_library(silero-lid STATIC silero_lid.h silero_lid.cpp)
set_target_properties(silero-lid PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(silero-lid PUBLIC .)
target_compile_features   (silero-lid PUBLIC cxx_std_17)
target_link_libraries     (silero-lid PUBLIC crispasr-core ggml ggml-base ggml-cpu)
set_target_properties(silero-lid PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(APPLE)
    # Accelerate cblas_sgemm for the (CPU-scalar) LID forward.
    if(NOT ACCELERATE_LIB)
        find_library(ACCELERATE_LIB Accelerate REQUIRED)
    endif()
    target_link_libraries(silero-lid PUBLIC ${ACCELERATE_LIB})
    target_compile_definitions(silero-lid PRIVATE HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()
if(NOT MSVC)
    target_compile_options(silero-lid PRIVATE -O3)
endif()

# lid-cld3 — local fallback (only built if shared crisp_lid not available)
if(NOT TARGET crisp_lid)
    add_library(lid-cld3 STATIC lid_cld3.h lid_cld3.cpp)
    set_target_properties(lid-cld3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(lid-cld3 PUBLIC .)
    target_compile_features   (lid-cld3 PUBLIC cxx_std_17)
    target_link_libraries     (lid-cld3 PUBLIC crispasr-core ggml ggml-base ggml-cpu)
    if(NOT MSVC)
        target_compile_options(lid-cld3 PRIVATE -O3)
    endif()
endif()

# ---------------------------------------------------------------------------
# text-lid-dispatch — backend-agnostic façade over lid-fasttext + lid-cld3.
# Peeks general.architecture and dispatches; one C ABI for any text-LID GGUF.
# ---------------------------------------------------------------------------
if(NOT TARGET crisp_lid)
    add_library(text-lid-dispatch STATIC text_lid_dispatch.h text_lid_dispatch.cpp)
    set_target_properties(text-lid-dispatch PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(text-lid-dispatch PUBLIC .)
    target_compile_features   (text-lid-dispatch PUBLIC cxx_std_17)
    target_compile_definitions(text-lid-dispatch PRIVATE CRISPASR_BUILD)
    target_link_libraries     (text-lid-dispatch PUBLIC crispasr-core lid-fasttext lid-cld3)
    if(NOT MSVC)
        target_compile_options(text-lid-dispatch PRIVATE -O3)
    endif()
endif()

# ---------------------------------------------------------------------------
# crisp_punc — shared punctuation library (FireRedPunc + PCS).
# Prefer the shared library from crisp_punc/; fall back to local copies.
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../crisp_punc/CMakeLists.txt" AND NOT TARGET crisp_punc)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../crisp_punc
                     ${CMAKE_BINARY_DIR}/crisp_punc)
    message(STATUS "crispasr: using shared crisp_punc library")
else()
    # Local fallback: fireredpunc
    add_library(fireredpunc STATIC fireredpunc.h fireredpunc.cpp)
    set_target_properties(fireredpunc PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(fireredpunc PUBLIC .)
    target_compile_features   (fireredpunc PUBLIC cxx_std_17)
    target_link_libraries     (fireredpunc PUBLIC crispasr-core ggml ggml-base ggml-cpu)
    if(NOT MSVC)
        target_compile_options(fireredpunc PRIVATE -O3)
    endif()
endif()

# ---------------------------------------------------------------------------
# crisp_truecase — shared truecaser library (stat + CRF + BiLSTM).
# Prefer shared library from crisp_truecase/; fall back to local copies.
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../crisp_truecase/CMakeLists.txt" AND NOT TARGET crisp_truecase)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../crisp_truecase
                     ${CMAKE_BINARY_DIR}/crisp_truecase)
    message(STATUS "crispasr: using shared crisp_truecase library")
else()
    add_library(truecaser STATIC truecaser.h truecaser.cpp)
    set_target_properties(truecaser PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(truecaser PUBLIC .)
    target_compile_features   (truecaser PUBLIC cxx_std_17)
    if(NOT MSVC)
        target_compile_options(truecaser PRIVATE -O3)
    endif()

    add_library(truecaser_crf STATIC truecaser_crf.h truecaser_crf.cpp)
    set_target_properties(truecaser_crf PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(truecaser_crf PUBLIC .)
    target_compile_features   (truecaser_crf PUBLIC cxx_std_17)
    if(NOT MSVC)
        target_compile_options(truecaser_crf PRIVATE -O3)
    endif()

    add_library(truecaser_lstm STATIC truecaser_lstm.h truecaser_lstm.cpp)
    set_target_properties(truecaser_lstm PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(truecaser_lstm PUBLIC .)
    target_compile_features   (truecaser_lstm PUBLIC cxx_std_17)
    if(NOT MSVC)
        target_compile_options(truecaser_lstm PRIVATE -O3)
    endif()
endif()

# ---------------------------------------------------------------------------
# pcs — Punctuation + Capitalization + Segmentation (1-800-BAD-CODE XLM-R)
# ---------------------------------------------------------------------------
if(NOT TARGET crisp_punc)
    # Only build local pcs if shared crisp_punc wasn't loaded
    add_library(pcs STATIC pcs.h pcs.cpp)
    set_target_properties(pcs PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(pcs PUBLIC .)
    target_compile_features   (pcs PUBLIC cxx_std_17)
    target_link_libraries     (pcs PUBLIC crispasr-core ggml ggml-base ggml-cpu)
    if(NOT MSVC)
        target_compile_options(pcs PRIVATE -O3)
    endif()
endif()

# ---------------------------------------------------------------------------
# ctc-align — CTC forced alignment (Viterbi DP, CPU-only, no ggml dependency)
# ---------------------------------------------------------------------------
add_library(ctc-align STATIC align.h align.cpp)
set_target_properties(ctc-align PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ctc-align PUBLIC .)
target_compile_features   (ctc-align PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(ctc-align PRIVATE -O3)
endif()
# ---------------------------------------------------------------------------
# Multi-backend linkage (consumed by CLI + Dart + Python + Rust)
#
# Non-Whisper CrispASR backends become reachable through the `crispasr_*`
# C-ABI wrappers in crispasr_c_api.cpp. Each backend header is #included
# under `#if __has_include(...)`, and when the corresponding CMake target
# exists we PUBLIC-link it into libcrispasr so the resulting shared library
# carries a DT_NEEDED on its sibling dylib. Consumers only have to open
# libcrispasr; the platform loader pulls the rest automatically.
#
# Adding a new backend is a three-line change here: (1) build the
# backend target as usual elsewhere in this file, (2) add
# `target_link_libraries(crispasr-lib PUBLIC <target>)` below, (3) add the
# `crispasr_<backend>_*` wrappers to crispasr_c_api.cpp.
# ---------------------------------------------------------------------------
if (TARGET nemotron)
    target_link_libraries(crispasr-lib PUBLIC nemotron)
endif()
if (TARGET parakeet)
    target_link_libraries(crispasr-lib PUBLIC parakeet)
endif()
if (TARGET canary)
    target_link_libraries(crispasr-lib PUBLIC canary)
endif()
if (TARGET lfm2_audio)
    target_link_libraries(crispasr-lib PUBLIC lfm2_audio)
endif()
if (TARGET qwen3_asr)
    target_link_libraries(crispasr-lib PUBLIC qwen3_asr)
endif()
if (TARGET cohere)
    target_link_libraries(crispasr-lib PUBLIC cohere)
endif()
if (TARGET granite_speech)
    target_link_libraries(crispasr-lib PUBLIC granite_speech)
endif()
if (TARGET granite_nle)
    target_link_libraries(crispasr-lib PUBLIC granite_nle)
endif()
if (TARGET canary_ctc)
    target_link_libraries(crispasr-lib PUBLIC canary_ctc)
endif()
if (TARGET voxtral)
    target_link_libraries(crispasr-lib PUBLIC voxtral)
endif()
if (TARGET voxtral4b)
    target_link_libraries(crispasr-lib PUBLIC voxtral4b)
endif()
if (TARGET wav2vec2-ggml)
    target_link_libraries(crispasr-lib PUBLIC wav2vec2-ggml)
endif()
if (TARGET ctc-align)
    target_link_libraries(crispasr-lib PUBLIC ctc-align)
endif()
if (TARGET glm-asr)
    target_link_libraries(crispasr-lib PUBLIC glm-asr)
endif()
if (TARGET mini-omni2)
    target_link_libraries(crispasr-lib PUBLIC mini-omni2)
endif()
if (TARGET kyutai-stt)
    target_link_libraries(crispasr-lib PUBLIC kyutai-stt)
endif()
if (TARGET firered-asr)
    target_link_libraries(crispasr-lib PUBLIC firered-asr)
endif()
if (TARGET firered-vad)
    target_link_libraries(crispasr-lib PUBLIC firered-vad)
endif()
if (TARGET marblenet-vad)
    target_link_libraries(crispasr-lib PUBLIC marblenet-vad)
endif()
if (TARGET crispasr-vad-encdec)
    target_link_libraries(crispasr-lib PUBLIC crispasr-vad-encdec)
endif()
if (TARGET firered-lid)
    target_link_libraries(crispasr-lib PUBLIC firered-lid)
endif()
if (TARGET crisp_lid)
    target_link_libraries(crispasr-lib PUBLIC crisp_lid)
elseif (TARGET lid-fasttext)
    target_link_libraries(crispasr-lib PUBLIC lid-fasttext)
endif()
if (TARGET omniasr)
    target_link_libraries(crispasr-lib PUBLIC omniasr)
endif()
if (TARGET vibevoice)
    target_link_libraries(crispasr-lib PUBLIC vibevoice)
endif()
if (TARGET kugelaudio)
    target_link_libraries(crispasr-lib PUBLIC kugelaudio)
endif()
if (TARGET ecapa-lid)
    target_link_libraries(crispasr-lib PUBLIC ecapa-lid)
endif()
if (TARGET titanet)
    target_link_libraries(crispasr-lib PUBLIC titanet)
endif()
if (TARGET moonshine)
    target_link_libraries(crispasr-lib PUBLIC moonshine)
endif()
if (TARGET moonshine_streaming)
    target_link_libraries(crispasr-lib PUBLIC moonshine_streaming)
endif()
if (TARGET gemma4_e2b)
    target_link_libraries(crispasr-lib PUBLIC gemma4_e2b)
endif()
if (TARGET mimo_tokenizer)
    target_link_libraries(crispasr-lib PUBLIC mimo_tokenizer)
endif()
if (TARGET mimo_asr)
    target_link_libraries(crispasr-lib PUBLIC mimo_asr)
endif()
if (TARGET moss_audio)
    target_link_libraries(crispasr-lib PUBLIC moss_audio)
endif()
if (TARGET qwen3_tts)
    target_link_libraries(crispasr-lib PUBLIC qwen3_tts)
endif()
if (TARGET orpheus)
    target_link_libraries(crispasr-lib PUBLIC orpheus)
endif()
if (TARGET tada-tts)
    target_link_libraries(crispasr-lib PUBLIC tada-tts)
endif()
if (TARGET chatterbox)
    # libcrispasr.so calls chatterbox_set_temperature / set_top_p /
    # set_min_p / set_repetition_penalty / set_max_speech_tokens from
    # crispasr_c_api.cpp. With BUILD_SHARED_LIBS=ON on Linux (the
    # default the Docker Smoke build hits), ld leaves those refs
    # unresolved in libcrispasr.so because chatterbox.cpp.o that
    # defines them wasn't being pulled in from libchatterbox.a at
    # SO-link time. Static archive members after a shared library on
    # the executable link line can't satisfy the shared library's
    # externals, so the exec link then errored:
    #   "libcrispasr.so.0.6.2: undefined reference to chatterbox_set_temperature"
    # (Docker Smoke failure on ff7bcc50 → d0d95e48 attempted fix.)
    #
    # The d0d95e48 attempt was `-Wl,--whole-archive chatterbox -Wl,--no-whole-archive`,
    # but CMake deduplicated the `chatterbox` target out of the wrap
    # (treating it as a library name) and put it elsewhere on the link
    # line — leaving an empty `-Wl,--whole-archive -Wl,--no-whole-archive`
    # pair, so the wrap had no effect. Using `$<TARGET_FILE:chatterbox>`
    # passes the resolved `.a` file path as a literal, which CMake
    # doesn't recognise as a library and therefore doesn't deduplicate
    # against the original `target_link_libraries(... chatterbox)` call.
    # add_dependencies() restores the build-order edge the generator
    # expression alone wouldn't establish.
    if (UNIX AND NOT APPLE AND BUILD_SHARED_LIBS)
        target_link_libraries(crispasr-lib PUBLIC
            "-Wl,--whole-archive"
            "$<TARGET_FILE:chatterbox>"
            "-Wl,--no-whole-archive")
        add_dependencies(crispasr-lib chatterbox)
    else()
        target_link_libraries(crispasr-lib PUBLIC chatterbox)
    endif()
endif()
if (TARGET indextts)
    target_link_libraries(crispasr-lib PUBLIC indextts)
endif()
if (TARGET m2m100)
    target_link_libraries(crispasr-lib PUBLIC m2m100)
endif()
if (TARGET t5_translate)
    target_link_libraries(crispasr-lib PUBLIC t5_translate)
endif()
if (TARGET voxcpm2_tts)
    target_link_libraries(crispasr-lib PUBLIC voxcpm2_tts)
endif()
if (TARGET cosyvoice3_tts)
    target_link_libraries(crispasr-lib PUBLIC cosyvoice3_tts)
endif()
if (TARGET csm-tts)
    target_link_libraries(crispasr-lib PUBLIC csm-tts)
endif()
if (TARGET bark-tts)
    target_link_libraries(crispasr-lib PUBLIC bark-tts)
endif()
if (TARGET dia-tts)
    target_link_libraries(crispasr-lib PUBLIC dia-tts)
endif()
if (TARGET fastpitch-tts)
    target_link_libraries(crispasr-lib PUBLIC fastpitch-tts)
endif()
if (TARGET parler-tts)
    target_link_libraries(crispasr-lib PUBLIC parler-tts)
endif()
if (TARGET speecht5-tts)
    target_link_libraries(crispasr-lib PUBLIC speecht5-tts)
endif()
if (TARGET kokoro)
    target_link_libraries(crispasr-lib PUBLIC kokoro)
endif()
if (TARGET outetts)
    target_link_libraries(crispasr-lib PUBLIC outetts)
endif()
if (TARGET zonos-tts)
    target_link_libraries(crispasr-lib PUBLIC zonos-tts)
endif()
if (TARGET f5-tts)
    target_link_libraries(crispasr-lib PUBLIC f5-tts)
endif()
if (TARGET pocket-tts)
    target_link_libraries(crispasr-lib PUBLIC pocket-tts)
endif()
if (TARGET bert-encoder)
    target_link_libraries(crispasr-lib PUBLIC bert-encoder)
endif()
if (TARGET melotts)
    target_link_libraries(crispasr-lib PUBLIC melotts)
endif()
if (TARGET openvoice2)
    target_link_libraries(crispasr-lib PUBLIC openvoice2)
endif()
if (TARGET piper-tts)
    target_link_libraries(crispasr-lib PUBLIC piper-tts)
endif()
if (TARGET funasr)
    target_link_libraries(crispasr-lib PUBLIC funasr)
endif()
if (TARGET paraformer)
    target_link_libraries(crispasr-lib PUBLIC paraformer)
endif()
if (TARGET sensevoice)
    target_link_libraries(crispasr-lib PUBLIC sensevoice)
endif()
if (TARGET crisp_punc)
    target_link_libraries(crispasr-lib PUBLIC crisp_punc)
elseif (TARGET fireredpunc)
    target_link_libraries(crispasr-lib PUBLIC fireredpunc)
endif()
if (TARGET truecaser AND NOT TARGET crisp_truecase)
    target_link_libraries(crispasr-lib PUBLIC truecaser)
endif()
if (TARGET crisp_truecase)
    target_link_libraries(crispasr-lib PUBLIC crisp_truecase)
else()
    if (TARGET truecaser_crf)
        target_link_libraries(crispasr-lib PUBLIC truecaser_crf)
    endif()
    if (TARGET truecaser_lstm)
        target_link_libraries(crispasr-lib PUBLIC truecaser_lstm)
    endif()
endif()
if (TARGET pcs AND NOT TARGET crisp_punc)
    target_link_libraries(crispasr-lib PUBLIC pcs)
endif()

# crisp_punc / crisp_truecase / crisp_lid are pulled in via add_subdirectory
# from sibling dirs and define crispasr_c_api.cpp's fireredpunc_*/pcs_*/
# truecaser_lstm_*/text_lid_* symbols. Under the Xcode generator the
# target_link_libraries() dependency above is NOT honored as a hard build-order
# constraint: the ios-xcode-build job's Ld of libcrispasr.dylib started ~9 ms
# after `Libtool libcrisp_punc.a` *began*, i.e. it linked the archive before it
# finished writing → those exact 12 symbols came up undefined (even with
# -force_load, which then just force-loaded an empty archive).
#   1. add_dependencies forces Xcode to finish archiving these before the link.
#   2. -force_load keeps the now-complete archive members past -dead_strip.
# Apple-only; the GNU/MSVC linkers resolve these fine.
if(APPLE)
    foreach(_force_lib crisp_punc crisp_truecase crisp_lid)
        if(TARGET ${_force_lib})
            add_dependencies(crispasr-lib ${_force_lib})
            target_link_options(crispasr-lib PRIVATE
                "LINKER:-force_load,$<TARGET_FILE:${_force_lib}>")
        endif()
    endforeach()
endif()

# Pyannote segmentation is not a backend; it's consumed by
# crispasr_diarize.cpp for the "pyannote" method. Still needs linking in
# so the native diarizer's symbols resolve at libcrispasr load time.
if (TARGET pyannote-seg)
    target_link_libraries(crispasr-lib PUBLIC pyannote-seg)
endif()
# Silero LID engine used by crispasr_lid.cpp's native Silero path.
if (TARGET silero-lid)
    target_link_libraries(crispasr-lib PUBLIC silero-lid)
endif()
# CLD3 text-LID — sibling of lid-fasttext, routed via the auto-routing
# text-LID dispatcher in src/text_lid_dispatch.cpp.
if (TARGET lid-cld3 AND NOT TARGET crisp_lid)
    target_link_libraries(crispasr-lib PUBLIC lid-cld3)
endif()
# Text-LID dispatcher — selects between fastText and CLD3 by GGUF arch.
# crispasr-lib (the main library) and crispasr-lid both go through this.
if (TARGET text-lid-dispatch AND NOT TARGET crisp_lid)
    target_link_libraries(crispasr-lib PUBLIC text-lid-dispatch)
endif()

# Text-LLM chat backend — PRIVATE link: llama.h identifiers are
# absorbed into libcrispasr and don't bleed into the public surface.
# Symbols downstream consumers see are only the `crispasr_chat_*` ABI
# from include/crispasr_chat.h. See docs/prompts/chat-abi.md.
if (TARGET crispasr-llama-core)
    target_link_libraries(crispasr-lib PRIVATE crispasr-llama-core)
endif()

install(TARGETS crispasr-lib LIBRARY)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/crispasr_chat.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Convenience alias: produce `libwhisper.{dylib,so,dll}` as a symlink (or
# copy on Windows) next to `libcrispasr`, pointing at the same file. Lets
# legacy consumers that know the project as crispasr — not as
# CrispASR — open the library under its old name.
if (BUILD_SHARED_LIBS)
    set_target_properties(crispasr-lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(crispasr-lib PRIVATE CRISPASR_SHARED CRISPASR_BUILD)
    if (APPLE)
        add_custom_command(TARGET crispasr-lib POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:crispasr-lib>
                $<TARGET_FILE_DIR:crispasr-lib>/libwhisper.dylib
            COMMENT "Creating libwhisper.dylib -> $<TARGET_FILE_NAME:crispasr-lib> alias"
        )
    elseif (UNIX)
        add_custom_command(TARGET crispasr-lib POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:crispasr-lib>
                $<TARGET_FILE_DIR:crispasr-lib>/libwhisper.so
            COMMENT "Creating libwhisper.so -> $<TARGET_FILE_NAME:crispasr-lib> alias"
        )
    elseif (WIN32)
        add_custom_command(TARGET crispasr-lib POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:crispasr-lib>
                $<TARGET_FILE_DIR:crispasr-lib>/whisper.dll
            COMMENT "Copying $<TARGET_FILE:crispasr-lib> to whisper.dll alias"
        )
    endif()
endif()
