cmake_minimum_required(VERSION 3.10...3.29)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION_CONTENT)
string(STRIP "${PROJECT_VERSION_CONTENT}" PROJECT_VERSION_CONTENT)

project(crispasr VERSION ${PROJECT_VERSION_CONTENT} LANGUAGES C CXX)
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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CRISPASR_STANDALONE ON)

    include(git-vars)

    # configure project version
    configure_file(${CMAKE_SOURCE_DIR}/bindings/javascript/package-tmpl.json ${CMAKE_SOURCE_DIR}/bindings/javascript/package.json @ONLY)
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

    option(CRISPASR_WASM_SINGLE_FILE "crispasr: embed WASM inside the generated crispasr.js" ON)

    # TODO: without these, we get the following error:
    #       wasm-ld: error: --shared-memory is disallowed by crispasr.o because it was not compiled with 'atomics' or 'bulk-memory' features.
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -pthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

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

# sanitizers
option(CRISPASR_SANITIZE_THREAD    "crispasr: enable thread sanitizer"    OFF)
option(CRISPASR_SANITIZE_ADDRESS   "crispasr: enable address sanitizer"   OFF)
option(CRISPASR_SANITIZE_UNDEFINED "crispasr: enable undefined sanitizer" OFF)

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
add_subdirectory(src)

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

set_target_properties(crispasr PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/crispasr.h)
install(TARGETS crispasr LIBRARY PUBLIC_HEADER)

target_compile_definitions(crispasr PRIVATE
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

add_subdirectory(crisp_audio)

if (CRISPASR_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif ()

if (CRISPASR_BUILD_EXAMPLES)
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
        disable_msvc_warnings(crispasr)
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
