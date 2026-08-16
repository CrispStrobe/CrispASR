cmake_minimum_required(VERSION 3.10...3.29)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION_CONTENT)
string(STRIP "${PROJECT_VERSION_CONTENT}" PROJECT_VERSION_CONTENT)

project(crispasr VERSION ${PROJECT_VERSION_CONTENT} LANGUAGES C CXX)

# src/CMakeLists.txt promotes crispasr_audio.cpp and crispasr_mic.cpp to
# LANGUAGE OBJCXX on Apple so miniaudio's CoreAudio backend can include
# <AVFoundation/AVFoundation.h>. CMake refuses to honor that property
# unless the language is enabled at project scope.
if(APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

include(CheckIncludeFileCXX)

set(SOVERSION 1)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# C2PA (Content Credentials) enablement — defines crispasr_enable_c2pa(<target>),
# called for both the core lib (src/) and the CLI (examples/cli/).
include(CrispasrC2pa)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CRISPASR_STANDALONE ON)

    include(git-vars)

    # configure project version
    configure_file(${CMAKE_SOURCE_DIR}/bindings/javascript/package-tmpl.json ${CMAKE_SOURCE_DIR}/bindings/javascript/package.json @ONLY)

    # Sync the pure-JS C2PA signer/verifier from the c2pa-audio submodule (single
    # source of truth) into bindings/javascript so `crispasr/c2pa` and the JS
    # tests use the current WAV/MP3/M4A/FLAC implementation. The copies are
    # generated (gitignored); the originals live in third_party/c2pa-audio/js.
    set(_c2pa_js_src ${CMAKE_SOURCE_DIR}/third_party/c2pa-audio/js)
    if (EXISTS ${_c2pa_js_src}/c2pa.mjs)
        foreach(_f c2pa.mjs c2pa-verify.mjs default-cert.mjs)
            configure_file(${_c2pa_js_src}/${_f} ${CMAKE_SOURCE_DIR}/bindings/javascript/${_f} COPYONLY)
        endforeach()
    endif()
else()
    set(CRISPASR_STANDALONE OFF)
endif()

# Always populate GIT_SHA1 / GIT_DATE / GIT_COMMIT_SUBJECT — the CLI
# build-info banner uses them whether or not we're the top-level build.
if (NOT DEFINED GIT_SHA1)
    include(git-vars)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    option(CRISPASR_WASM            "crispasr: build WASM/JS target (Emscripten)"              ON)
    option(CRISPASR_WASM_SINGLE_FILE "crispasr: embed WASM inside the generated crispasr.js"   ON)
    # Single-threaded WASM: no pthreads, so NO SharedArrayBuffer / COOP-COEP required, and no
    # blocking-compute deadlock when driven from a Web Worker. Slower (1 thread), but it just runs
    # anywhere. Multithreaded (-pthread) stays the default.
    option(CRISPASR_WASM_SINGLE_THREAD "crispasr: single-threaded WASM (no pthreads, no COOP/COEP)" OFF)

    # Multithreaded WASM: -pthread enables SharedArrayBuffer-based threading
    # in the browser. The hosting page MUST set COOP/COEP headers:
    #   Cross-Origin-Opener-Policy: same-origin
    #   Cross-Origin-Embedder-Policy: require-corp
    if (NOT CRISPASR_WASM_SINGLE_THREAD)
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -pthread")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    endif()

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s TOTAL_STACK=5242880")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s TOTAL_STACK=5242880")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

#
# option list
#

# debug
option(CRISPASR_ALL_WARNINGS           "crispasr: enable all compiler warnings"                   ON)
option(CRISPASR_ALL_WARNINGS_3RD_PARTY "crispasr: enable all compiler warnings in 3rd party libs" OFF)

# build
option(CRISPASR_FATAL_WARNINGS  "crispasr: enable -Werror flag"               OFF)
option(CRISPASR_USE_SYSTEM_GGML "crispasr: use system-installed GGML library" OFF)
option(CRISPASR_PORTABLE_CPU
       "crispasr: compile bundled ggml-cpu for generic x86-64 (no optional ISA)"
       OFF)
option(CRISPASR_USE_ONNXRUNTIME
       "crispasr: enable onnxruntime inference for pyannote-seg/TitaNet (model paths ending in .onnx); GPU via CUDA EP with CPU fallback"
       OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH
       "crispasr: path to the onnxruntime C++ distribution (contains include/ and lib/); required when CRISPASR_USE_ONNXRUNTIME=ON")

# sanitizers
option(CRISPASR_SANITIZE_THREAD    "crispasr: enable thread sanitizer"    OFF)
option(CRISPASR_SANITIZE_ADDRESS   "crispasr: enable address sanitizer"   OFF)
option(CRISPASR_SANITIZE_UNDEFINED "crispasr: enable undefined sanitizer" OFF)

# libFuzzer harnesses (clang only). Combine with -DCRISPASR_SANITIZE_ADDRESS=ON
# for OOB detection. `-fsanitize=fuzzer-no-link` instruments every TU compiled
# after this for coverage; each harness target adds `-fsanitize=fuzzer` for the
# libFuzzer driver. See tests/fuzz/.
option(CRISPASR_FUZZ "crispasr: build libFuzzer harnesses (clang only)" OFF)
if (CRISPASR_FUZZ)
    add_compile_options(-fsanitize=fuzzer-no-link)
endif()

# extra artifacts
option(CRISPASR_BUILD_TESTS    "crispasr: build tests"          ${CRISPASR_STANDALONE})
option(CRISPASR_BUILD_EXAMPLES "crispasr: build examples"       ${CRISPASR_STANDALONE})
option(CRISPASR_BUILD_SERVER   "crispasr: build server example" ${CRISPASR_STANDALONE})

# 3rd party libs
option(CRISPASR_CURL "crispasr: use libcurl to download model from an URL" OFF)
option(CRISPASR_SDL2 "crispasr: support for libSDL2" OFF)

if (CRISPASR_CURL)
    find_package(CURL REQUIRED)
    message(STATUS "libcurl found: ${CURL_VERSION_STRING} (${CURL_LIBRARIES})")
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    option(CRISPASR_FFMPEG "crispasr: support building and linking with ffmpeg libs (avcodec, swresample, ...)" OFF)
endif()

option(CRISPASR_MEDIA_NDK              "crispasr: enable Android Media NDK (auto-detected)" ON)
option(CRISPASR_COREML                "crispasr: enable Core ML framework"  OFF)
option(CRISPASR_COREML_ALLOW_FALLBACK "crispasr: allow non-CoreML fallback" OFF)
option(CRISPASR_OPENVINO              "crispasr: support for OpenVINO"      OFF)

# ---------------------------------------------------------------------------
# Cohere Transcribe options
# ---------------------------------------------------------------------------
# COHERE_MKL: route both ggml's GEMM and the Cohere mel filterbank through
# Intel MKL. This typically gives a 1.5-2× speedup of the F16 / quantised
# matmul paths on x86_64 (Cascade Lake / Ice Lake / Zen4) by using MLAS-class
# INT8/F32 kernels with VNNI. Default OFF — turning it on adds a ~200 MB MKL
# runtime dependency. Recommended only for x86_64 server deployments.
option(COHERE_MKL "cohere: link Intel MKL for ggml GEMM + mel filterbank" OFF)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)

# Bump tensor name limit for models with long hierarchical names (e.g.
# Pocket TTS Mimi decoder transformer: 71 chars). Must be set before
# add_subdirectory(ggml) so the definition propagates to all targets.
add_compile_definitions(GGML_MAX_NAME=128)

# MSVC's <cmath> doesn't expose M_PI unless _USE_MATH_DEFINES is set.
# NOMINMAX prevents <windows.h> from defining min/max macros that conflict
# with std::min/std::max (#152). Both are defined globally so all targets
# (src, tests, examples) get them on Windows.
if (WIN32)
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
endif()

# Large generated tables (e.g. src/core/pinyin_g2p.cpp, #294) overflow MSVC's
# default per-object section limit → "error C1128: number of sections exceeded
# object file format limit". /bigobj raises the limit; harmless for every other
# host TU. Guard to C/CXX (cl.exe) ONLY — a bare /bigobj handed to nvcc for .cu
# files ("nvcc fatal: Unknown option") would break the Windows CUDA build; nvcc
# would need it via -Xcompiler, and no .cu currently needs it anyway (#309).
if (MSVC)
    # Single-value COMPILE_LANGUAGE genexes (CMake >=3.3) — the multi-value
    # "C,CXX" form needs >=3.15, above this project's declared 3.10 floor.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:/bigobj>)
endif()

# Wire COHERE_MKL into ggml's BLAS backend before add_subdirectory(ggml).
# This must happen here so the values are visible to the ggml subproject.
if (COHERE_MKL)
    set(GGML_BLAS        ON  CACHE BOOL "ggml: use BLAS"               FORCE)
    set(GGML_BLAS_VENDOR "Intel10_64lp" CACHE STRING "ggml: BLAS vendor" FORCE)
    # Hint our own find_package(BLAS) call (in src/CMakeLists.txt) toward MKL too.
    set(BLA_VENDOR       "Intel10_64lp")
    message(STATUS "COHERE_MKL=ON: enabling GGML_BLAS with vendor Intel10_64lp")
endif()

# override ggml options
set(GGML_SANITIZE_THREAD    ${CRISPASR_SANITIZE_THREAD})
set(GGML_SANITIZE_ADDRESS   ${CRISPASR_SANITIZE_ADDRESS})
set(GGML_SANITIZE_UNDEFINED ${CRISPASR_SANITIZE_UNDEFINED})
set(GGML_ALL_WARNINGS       ${CRISPASR_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS     ${CRISPASR_FATAL_WARNINGS})

# #261/#302: GPU inference still enumerates and initializes ggml-cpu before it
# selects CUDA/Vulkan. A GPU artifact whose CPU helper was compiled for AVX2 or
# BMI2 therefore raises SIGILL at model startup on older workstations, before
# the server's runtime ISA diagnostic can run. `GGML_NATIVE=OFF` is NOT enough:
# in a normal build it makes ggml's INS_ENB default true, silently enabling
# SSE4.2/AVX/AVX2/BMI2 and friends. Keep the portable contract in one CMake
# option so release jobs cannot omit one of the growing set of ggml ISA knobs.
# This is the generic x86-64/SSE2 floor; non-x86 builds simply leave irrelevant
# cache entries disabled.
if (CRISPASR_PORTABLE_CPU)
    if (CRISPASR_USE_SYSTEM_GGML)
        message(FATAL_ERROR
            "CRISPASR_PORTABLE_CPU cannot control a system-installed ggml; "
            "use the bundled submodule or provide a baseline-built ggml")
    endif()

    foreach(_ggml_isa
            NATIVE
            SSE42
            AVX
            AVX_VNNI
            AVX2
            FMA
            F16C
            BMI2
            AVX512
            AVX512_VBMI
            AVX512_VNNI
            AVX512_BF16
            AMX_TILE
            AMX_INT8
            AMX_BF16)
        set(GGML_${_ggml_isa} OFF CACHE BOOL
            "Disabled by CRISPASR_PORTABLE_CPU" FORCE)
    endforeach()
    message(STATUS
        "CRISPASR_PORTABLE_CPU=ON: ggml-cpu uses the generic x86-64 baseline")
endif()

# §232: Enable tinyBLAS/llamafile GEMM by default — ~15-29% faster CPU encoder
# matmuls across all backends. transcribe.cpp forces this ON; ggml defaults OFF.
# No risk: tinyBLAS is a drop-in SIMD-optimized GEMM replacement.
set(GGML_LLAMAFILE_DEFAULT ON)

# transition helpers
function (crispasr_option_depr TYPE OLD NEW)
    if (${OLD})
        message(${TYPE} "${OLD} is deprecated and will be removed in the future.\nUse ${NEW} instead\n")
        set(${NEW} ON)
    endif()
endfunction()

crispasr_option_depr(FATAL_ERROR CRISPASR_CUBLAS              GGML_CUDA)
crispasr_option_depr(WARNING     CRISPASR_CUDA                GGML_CUDA)
crispasr_option_depr(WARNING     CRISPASR_KOMPUTE             GGML_KOMPUTE)
crispasr_option_depr(WARNING     CRISPASR_METAL               GGML_METAL)
crispasr_option_depr(WARNING     CRISPASR_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY)
crispasr_option_depr(WARNING     CRISPASR_NATIVE              GGML_NATIVE)
crispasr_option_depr(WARNING     CRISPASR_OPENMP              GGML_OPENMP)
crispasr_option_depr(WARNING     CRISPASR_RPC                 GGML_RPC)
crispasr_option_depr(WARNING     CRISPASR_SYCL                GGML_SYCL)
crispasr_option_depr(WARNING     CRISPASR_SYCL_F16            GGML_SYCL_F16)
crispasr_option_depr(WARNING     CRISPASR_CCACHE              GGML_CCACHE)

if (GGML_CUDA AND NOT MSVC)
    #GGML_CUDA enabled, add the necessary compile options -Wno-deprecated-gpu-targets
    add_compile_options(-Wno-deprecated-gpu-targets)
endif()

#
# build the library
#

if (NOT TARGET ggml)
    if (CRISPASR_USE_SYSTEM_GGML)
        find_package(ggml REQUIRED)
        if (NOT ggml_FOUND)
            message(FATAL_ERROR "System-installed GGML library not found.")
        endif()
        add_library(ggml ALIAS ggml::ggml)
    else()
        # #298: a clone without submodules leaves ggml/ empty, so add_subdirectory
        # fails with a cryptic "does not contain a CMakeLists.txt file" (and then
        # cascades into an empty-install-DESTINATION error). Fail early with an
        # actionable message instead.
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
            message(FATAL_ERROR
                "The bundled 'ggml' submodule is not initialized "
                "(${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt is missing).\n"
                "Fetch the submodules and re-run cmake:\n"
                "    git submodule update --init --recursive\n"
                "or re-clone with:\n"
                "    git clone --recursive https://github.com/CrispStrobe/CrispASR")
        endif()
        add_subdirectory(ggml)
        if(WIN32)
            # The following adds a _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR macro and is a workaround for
            # the Windows C++ standard library which does not support constexpr mutexes.
            # From the release notes://github.com/microsoft/STL/wiki/Changelog
            #  Disable constexpr mutex constructor on Windows
            #  Fixed mutex's constructor to be constexpr. #3824 #4000 #4339
            #  Note: Programs that aren't following the documented restrictions on binary compatibility may encounter
            #  null dereferences in mutex machinery. You must follow this rule:
            #  When you mix binaries built by different supported versions of the toolset, the Redistributable version
            #  must be at least as new as the latest toolset used by any app component.
            #  You can define _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR as an escape hatch.
            #
            # Specifically to crispasr this would cause a crash when using the Java bindings.
            # resulting in a Invalid memory access error.
            target_compile_definitions(ggml-base PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
        endif()
    endif()
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

# Define CMAKE_INSTALL_* (LIBDIR / INCLUDEDIR / …) BEFORE add_subdirectory(src):
# src/CMakeLists.txt's install() rules reference them. This previously relied on
# ggml's own include(GNUInstallDirs) running first via add_subdirectory(ggml), so
# an absent/incomplete ggml left those DESTINATIONs empty and install() errored
# with "install FILES given no DESTINATION" (#298). Including it here is
# idempotent with the later include in the install section.
include(GNUInstallDirs)
# ---------------------------------------------------------------------------
# Backend link shims (#355)
#
# Every per-model library links `ggml-cuda` / `ggml-metal` under `if(GGML_CUDA)`
# / `if(GGML_METAL)`. Nothing in this tree calls ggml_backend_cuda_init() or its
# Metal twin directly — the link exists so a STATIC build keeps the backend's
# object files, since backends register themselves from static initialisers and
# a linker that sees no referenced symbol drops them (#316).
#
# It is also why the shipped CUDA binary carries libcuda.so.1 as a hard
# DT_NEEDED and exits 127 on a host without the NVIDIA driver, before
# ggml_backend_init_best() can fall back to CPU (#355).
#
# With GGML_BACKEND_DL the backends are MODULE libraries, dlopen'd at runtime by
# ggml_backend_load_all() — which this tree already calls. CMake refuses a
# MODULE_LIBRARY as a link target, and it must not be linked anyway: the whole
# point is that a missing backend is a runtime no-op rather than a load-time
# abort. So route the link sites through interface targets that are empty in a
# DL build and the real backend otherwise.
add_library(crispasr_link_ggml_metal INTERFACE)
add_library(crispasr_link_ggml_cuda INTERFACE)
add_library(crispasr_link_ggml_vulkan INTERFACE)
# ggml-cpu is a MODULE under DL too — the CPU backend is dlopen'd like any
# other, which is what makes the fallback work when CUDA is absent. The call
# sites go through src/core/ggml_cpu_backend.h.
add_library(crispasr_link_ggml_cpu INTERFACE)
if(NOT GGML_BACKEND_DL)
    if(GGML_METAL)
        target_link_libraries(crispasr_link_ggml_metal INTERFACE ggml-metal)
    endif()
    if(GGML_CUDA)
        target_link_libraries(crispasr_link_ggml_cuda INTERFACE ggml-cuda)
    endif()
    if(GGML_VULKAN)
        target_link_libraries(crispasr_link_ggml_vulkan INTERFACE ggml-vulkan)
    endif()
    target_link_libraries(crispasr_link_ggml_cpu INTERFACE ggml-cpu)
endif()

add_subdirectory(src)

# ---------------------------------------------------------------------------
# WASM / JS bindings (Emscripten only, all backends)
# ---------------------------------------------------------------------------
if(EMSCRIPTEN AND CRISPASR_WASM)
    add_subdirectory(bindings/javascript)
endif()

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CRISPASR_BUILD_NUMBER        ${BUILD_NUMBER})
set(CRISPASR_BUILD_COMMIT        ${BUILD_COMMIT})
set(CRISPASR_INSTALL_VERSION     ${CMAKE_PROJECT_VERSION})

set(CRISPASR_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(CRISPASR_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(CRISPASR_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

get_directory_property(CRISPASR_TRANSIENT_DEFINES COMPILE_DEFINITIONS)

set_target_properties(crispasr-lib PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/crispasr.h)
install(TARGETS crispasr-lib LIBRARY PUBLIC_HEADER)

target_compile_definitions(crispasr-lib PRIVATE
    CRISPASR_VERSION="${PROJECT_VERSION}"
)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/crispasr-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/crispasr-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/crispasr
    PATH_VARS
    CRISPASR_INCLUDE_INSTALL_DIR
    CRISPASR_LIB_INSTALL_DIR
    CRISPASR_BIN_INSTALL_DIR )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/crispasr-version.cmake
    VERSION ${CRISPASR_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/crispasr-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/crispasr-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/crispasr)

configure_file(cmake/crispasr.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/crispasr.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/crispasr.pc"
        DESTINATION lib/pkgconfig)

#
# programs, examples and tests
#

# glint — our clean-room MP3 encoder (in-tree copy of the sibling glint
# project, MP3 core only) for TTS output (server response_format=mp3,
# CLI --tts-output *.mp3). Static lib, only built when a consumer
# (cli/server/tests) links it.
add_subdirectory(glint EXCLUDE_FROM_ALL)

add_subdirectory(crisp_audio)

if (CRISPASR_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif ()

if (CRISPASR_BUILD_EXAMPLES)
    add_subdirectory(examples)
else()
    # The Ruby/Python/Rust/Dart bindings need the `common` static
    # library (audio I/O helpers etc.) but NOT the example executables
    # (cli, server) and not Catch2. examples/CMakeLists.txt has a
    # `if (CRISPASR_BUILD_COMMON_ONLY)` early-return that builds just
    # the common+common-crispasr libraries when CRISPASR_BUILD_EXAMPLES
    # is OFF. Always include it — the early-return is cheap when no
    # binding is consuming it.
    set(CRISPASR_BUILD_COMMON_ONLY ON)
    add_subdirectory(examples)
endif()

if (MSVC)
    set(MSVC_WARNING_FLAGS
        /wd4101  # Unreferenced local variable
        /wd4005  # Macro redefinition
        /wd4065  # switch statement contains 'default' but no 'case' labels
        /wd4267  # Conversion from 'size_t' to a smaller type, possible loss of data
        /wd4244  # Conversion from one type to another type, possible loss of ata
        /wd4805  # Unsafe mix of type
        /wd4305  # Truncation from 'type1' to 'type2' (often double to float)
        /wd4996  # Function or variable may be unsafe/deprecated
    )
    function(disable_msvc_warnings target_name)
        if(TARGET ${target_name})
            target_compile_options(${target_name} PRIVATE ${MSVC_WARNING_FLAGS})
        endif()
    endfunction()

    if (CRISPASR_BUILD_EXAMPLES)
        disable_msvc_warnings(crispasr-lib)
        disable_msvc_warnings(common)
        disable_msvc_warnings(common-sdl)
        disable_msvc_warnings(lsp)
        disable_msvc_warnings(wchess-core)
        disable_msvc_warnings(crispasr-command)
        disable_msvc_warnings(crispasr-cli)
        disable_msvc_warnings(crispasr-server)
        disable_msvc_warnings(crispasr-stream)
        disable_msvc_warnings(crispasr-talk-llama)
        disable_msvc_warnings(crispasr-bench)
        disable_msvc_warnings(quantize)
        disable_msvc_warnings(vad-speech-segments)
    endif()
endif()
