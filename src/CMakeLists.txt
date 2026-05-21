# TODO: should not use this
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

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

# crispasr (formerly whisper)

add_library(crispasr
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
    target_sources(crispasr PRIVATE
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
    target_compile_definitions(crispasr PRIVATE CRISPASR_RNNOISE=1)
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
add_library(whisper ALIAS crispasr)

# miniaudio.h on Apple includes <AVFoundation/AVFoundation.h> which
# contains ObjC syntax (@class, @protocol). When compiling as pure
# C++ this fails on visionOS/tvOS/iOS xcframework builds. Setting the
# source language to OBJCXX lets clang accept the ObjC headers.
if(APPLE)
    set_source_files_properties(crispasr_audio.cpp crispasr_mic.cpp
        PROPERTIES LANGUAGE OBJCXX)
endif()

set_target_properties(crispasr PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${SOVERSION}
    OUTPUT_NAME crispasr
)

target_include_directories(crispasr PUBLIC . ../include)
target_compile_features   (crispasr PUBLIC cxx_std_11) # don't bump

if (CMAKE_CXX_BYTE_ORDER STREQUAL "BIG_ENDIAN")
    target_compile_definitions(crispasr PRIVATE CRISPASR_BIG_ENDIAN)
endif()

if (CRISPASR_EXTRA_FLAGS)
    target_compile_options(crispasr PRIVATE ${CRISPASR_EXTRA_FLAGS})
endif()

find_package(Threads REQUIRED)
target_link_libraries(crispasr PUBLIC ggml Threads::Threads)

# `crispasr_cache.cpp` calls WinHTTP for HF auto-download on Windows. MSVC
# auto-links winhttp.lib via `#pragma comment(lib, "winhttp.lib")` inside
# the .cpp, but MinGW / GCC ignore that pragma — so any downstream test or
# executable that links libcrispasr.a (and therefore pulls in
# crispasr_cache.cpp.obj) needs winhttp on its link line. Propagate it as
# PUBLIC here so every consumer transitively picks it up, and individual
# tests don't have to remember to add it. Mirrors how Threads::Threads
# already flows through.
if (WIN32)
    target_link_libraries(crispasr PUBLIC winhttp)
endif()

if (CRISPASR_COREML)
    target_link_libraries(crispasr PRIVATE crispasr.coreml)
endif()

if (CRISPASR_OPENVINO)
    target_link_libraries(crispasr PRIVATE crispasr.openvino)
endif()

if (CRISPASR_MKL)
    target_link_libraries(crispasr PRIVATE MKL::MKL)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(crispasr PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(crispasr PRIVATE CRISPASR_SHARED CRISPASR_BUILD)
endif()

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
    find_package(BLAS)
    if(BLAS_FOUND)
        target_link_libraries(cohere PUBLIC ${BLAS_LIBRARIES})
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
        if(OPENBLAS_LIB)
            target_link_libraries(cohere PUBLIC ${OPENBLAS_LIB})
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
if(NOT MSVC)
    target_compile_options(crispasr-core PRIVATE -O3)
endif()
set_target_properties(crispasr-core PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
add_library(orpheus STATIC orpheus.h orpheus.cpp orpheus_snac.h orpheus_snac.cpp)
set_target_properties(orpheus PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(orpheus PUBLIC .)
target_compile_features   (orpheus PUBLIC cxx_std_17)
target_link_libraries     (orpheus PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(orpheus PRIVATE -O3)
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

# ---------------------------------------------------------------------------
# vibevoice — Microsoft VibeVoice-ASR (σ-VAE tokenizers + Qwen2 LM)
# ---------------------------------------------------------------------------
add_library(vibevoice STATIC vibevoice.h vibevoice.cpp)
set_target_properties(vibevoice PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(vibevoice PUBLIC .)
target_compile_features   (vibevoice PUBLIC cxx_std_17)
target_link_libraries     (vibevoice PUBLIC crispasr-core ggml ggml-base ggml-cpu)

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
if(NOT MSVC)
    target_compile_options(voxcpm2_tts PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# kokoro — hexgrad/Kokoro-82M + yl4579/StyleTTS2-LJSpeech (iSTFTNet TTS)
# ---------------------------------------------------------------------------
add_library(kokoro STATIC kokoro.h kokoro.cpp)
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
# AUTO: link if pkg-config finds it OR if Homebrew layout is present;
# otherwise silently fall back to the popen path.
set(CRISPASR_WITH_ESPEAK_NG "AUTO" CACHE STRING
    "Link kokoro against libespeak-ng for in-process phonemization (AUTO/ON/OFF)")
set_property(CACHE CRISPASR_WITH_ESPEAK_NG PROPERTY STRINGS "AUTO" "ON" "OFF")

if(NOT CRISPASR_WITH_ESPEAK_NG STREQUAL "OFF")
    set(_espeak_found FALSE)
    set(_espeak_include "")
    set(_espeak_lib "")

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

    # Homebrew/macOS fallback: pkg-config sometimes isn't on PATH but the lib is.
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

    if(_espeak_found)
        target_compile_definitions(kokoro PRIVATE CRISPASR_HAVE_ESPEAK_NG=1)
        target_include_directories(kokoro PRIVATE ${_espeak_include})
        if(_espeak_libdir)
            target_link_directories(kokoro PUBLIC ${_espeak_libdir})
        endif()
        target_link_libraries(kokoro PUBLIC ${_espeak_lib})
        message(STATUS "kokoro: libespeak-ng enabled (in-process phonemizer)")
    elseif(CRISPASR_WITH_ESPEAK_NG STREQUAL "ON")
        message(FATAL_ERROR "CRISPASR_WITH_ESPEAK_NG=ON but libespeak-ng not found. Install espeak-ng or set OFF.")
    else()
        message(STATUS "kokoro: libespeak-ng not found, falling back to popen(espeak-ng) at runtime")
    endif()
endif()

# ---------------------------------------------------------------------------
# firered-vad — FireRedVAD (DFSMN voice activity detection)
# ---------------------------------------------------------------------------
add_library(firered-vad STATIC firered_vad.h firered_vad.cpp)
set_target_properties(firered-vad PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(firered-vad PUBLIC .)
target_compile_features   (firered-vad PUBLIC cxx_std_17)
target_link_libraries     (firered-vad PUBLIC crispasr-core ggml ggml-base ggml-cpu)

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
target_link_libraries     (firered-lid PUBLIC crispasr-core ggml ggml-base ggml-cpu)

# ---------------------------------------------------------------------------
# lid-fasttext — fastText supervised LID (GlotLID-V3 + Facebook LID-176).
# Text-only — runs on transcripts, not audio. Tiny F32/F16 forward
# (manual gather + matmul + softmax), no ggml graph for the F32/F16 path;
# quants will need a small ggml graph for the matmul.
# ---------------------------------------------------------------------------
add_library(lid-fasttext STATIC lid_fasttext.h lid_fasttext.cpp)
set_target_properties(lid-fasttext PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(lid-fasttext PUBLIC .)
target_compile_features   (lid-fasttext PUBLIC cxx_std_17)
target_link_libraries     (lid-fasttext PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(lid-fasttext PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# firered-asr — FireRedASR2-AED (FireRedTeam/FireRedASR2-AED)
# ---------------------------------------------------------------------------
add_library(firered-asr STATIC firered_asr.h firered_asr.cpp)
set_target_properties(firered-asr PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(firered-asr PUBLIC .)
target_compile_features   (firered-asr PUBLIC cxx_std_17)
target_link_libraries     (firered-asr PUBLIC crispasr-core ggml ggml-base ggml-cpu)
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
if(NOT MSVC)
    target_compile_options(silero-lid PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# lid-cld3 — Google CLD3 (Apache-2.0) text-LID, 109 ISO 639-1 labels
# Sibling backend to lid-fasttext; the auto-routing dispatcher in
# src/text_lid_dispatch.cpp picks between them by reading the GGUF's
# general.architecture key. Pure manual F32 forward, no ggml graph.
# ---------------------------------------------------------------------------
add_library(lid-cld3 STATIC lid_cld3.h lid_cld3.cpp)
set_target_properties(lid-cld3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(lid-cld3 PUBLIC .)
target_compile_features   (lid-cld3 PUBLIC cxx_std_17)
target_link_libraries     (lid-cld3 PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(lid-cld3 PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# text-lid-dispatch — backend-agnostic façade over lid-fasttext + lid-cld3.
# Peeks general.architecture and dispatches; one C ABI for any text-LID GGUF.
# ---------------------------------------------------------------------------
add_library(text-lid-dispatch STATIC text_lid_dispatch.h text_lid_dispatch.cpp)
set_target_properties(text-lid-dispatch PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(text-lid-dispatch PUBLIC .)
target_compile_features   (text-lid-dispatch PUBLIC cxx_std_17)
target_link_libraries     (text-lid-dispatch PUBLIC crispasr-core lid-fasttext lid-cld3)
# `text_lid_resolve_path()` calls into crispasr_cache::ensure_cached_file
# and crispasr_registry_lookup — both compiled into the crispasr shared
# library, not separate targets. The symbols resolve when the consumer
# (crispasr-lid, libcrispasr) links text-lid-dispatch + crispasr together.
if(NOT MSVC)
    target_compile_options(text-lid-dispatch PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# fireredpunc — FireRedPunc BERT-based punctuation restoration
# ---------------------------------------------------------------------------
add_library(fireredpunc STATIC fireredpunc.h fireredpunc.cpp)
set_target_properties(fireredpunc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fireredpunc PUBLIC .)
target_compile_features   (fireredpunc PUBLIC cxx_std_17)
target_link_libraries     (fireredpunc PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(fireredpunc PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# truecaser — statistical truecaser from word-frequency data
# ---------------------------------------------------------------------------
add_library(truecaser STATIC truecaser.h truecaser.cpp)
set_target_properties(truecaser PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(truecaser PUBLIC .)
target_compile_features   (truecaser PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(truecaser PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# truecaser_crf — CRF-based truecaser with context features
# ---------------------------------------------------------------------------
add_library(truecaser_crf STATIC truecaser_crf.h truecaser_crf.cpp)
set_target_properties(truecaser_crf PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(truecaser_crf PUBLIC .)
target_compile_features   (truecaser_crf PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(truecaser_crf PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# truecaser_lstm — BiLSTM character-level truecaser (mayhewsw/pytorch-truecaser)
# ---------------------------------------------------------------------------
add_library(truecaser_lstm STATIC truecaser_lstm.h truecaser_lstm.cpp)
set_target_properties(truecaser_lstm PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(truecaser_lstm PUBLIC .)
target_compile_features   (truecaser_lstm PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(truecaser_lstm PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# pcs — Punctuation + Capitalization + Segmentation (1-800-BAD-CODE XLM-R)
# ---------------------------------------------------------------------------
add_library(pcs STATIC pcs.h pcs.cpp)
set_target_properties(pcs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pcs PUBLIC .)
target_compile_features   (pcs PUBLIC cxx_std_17)
target_link_libraries     (pcs PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(pcs PRIVATE -O3)
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
# `target_link_libraries(crispasr PUBLIC <target>)` below, (3) add the
# `crispasr_<backend>_*` wrappers to crispasr_c_api.cpp.
# ---------------------------------------------------------------------------
if (TARGET parakeet)
    target_link_libraries(crispasr PUBLIC parakeet)
endif()
if (TARGET canary)
    target_link_libraries(crispasr PUBLIC canary)
endif()
if (TARGET qwen3_asr)
    target_link_libraries(crispasr PUBLIC qwen3_asr)
endif()
if (TARGET cohere)
    target_link_libraries(crispasr PUBLIC cohere)
endif()
if (TARGET granite_speech)
    target_link_libraries(crispasr PUBLIC granite_speech)
endif()
if (TARGET canary_ctc)
    target_link_libraries(crispasr PUBLIC canary_ctc)
endif()
if (TARGET voxtral)
    target_link_libraries(crispasr PUBLIC voxtral)
endif()
if (TARGET voxtral4b)
    target_link_libraries(crispasr PUBLIC voxtral4b)
endif()
if (TARGET wav2vec2-ggml)
    target_link_libraries(crispasr PUBLIC wav2vec2-ggml)
endif()
if (TARGET glm-asr)
    target_link_libraries(crispasr PUBLIC glm-asr)
endif()
if (TARGET kyutai-stt)
    target_link_libraries(crispasr PUBLIC kyutai-stt)
endif()
if (TARGET firered-asr)
    target_link_libraries(crispasr PUBLIC firered-asr)
endif()
if (TARGET firered-vad)
    target_link_libraries(crispasr PUBLIC firered-vad)
endif()
if (TARGET marblenet-vad)
    target_link_libraries(crispasr PUBLIC marblenet-vad)
endif()
if (TARGET crispasr-vad-encdec)
    target_link_libraries(crispasr PUBLIC crispasr-vad-encdec)
endif()
if (TARGET firered-lid)
    target_link_libraries(crispasr PUBLIC firered-lid)
endif()
if (TARGET lid-fasttext)
    target_link_libraries(crispasr PUBLIC lid-fasttext)
endif()
if (TARGET omniasr)
    target_link_libraries(crispasr PUBLIC omniasr)
endif()
if (TARGET vibevoice)
    target_link_libraries(crispasr PUBLIC vibevoice)
endif()
if (TARGET ecapa-lid)
    target_link_libraries(crispasr PUBLIC ecapa-lid)
endif()
if (TARGET titanet)
    target_link_libraries(crispasr PUBLIC titanet)
endif()
if (TARGET moonshine)
    target_link_libraries(crispasr PUBLIC moonshine)
endif()
if (TARGET moonshine_streaming)
    target_link_libraries(crispasr PUBLIC moonshine_streaming)
endif()
if (TARGET gemma4_e2b)
    target_link_libraries(crispasr PUBLIC gemma4_e2b)
endif()
if (TARGET mimo_tokenizer)
    target_link_libraries(crispasr PUBLIC mimo_tokenizer)
endif()
if (TARGET mimo_asr)
    target_link_libraries(crispasr PUBLIC mimo_asr)
endif()
if (TARGET qwen3_tts)
    target_link_libraries(crispasr PUBLIC qwen3_tts)
endif()
if (TARGET orpheus)
    target_link_libraries(crispasr PUBLIC orpheus)
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
        target_link_libraries(crispasr PUBLIC
            "-Wl,--whole-archive"
            "$<TARGET_FILE:chatterbox>"
            "-Wl,--no-whole-archive")
        add_dependencies(crispasr chatterbox)
    else()
        target_link_libraries(crispasr PUBLIC chatterbox)
    endif()
endif()
if (TARGET indextts)
    target_link_libraries(crispasr PUBLIC indextts)
endif()
if (TARGET m2m100)
    target_link_libraries(crispasr PUBLIC m2m100)
endif()
if (TARGET t5_translate)
    target_link_libraries(crispasr PUBLIC t5_translate)
endif()
if (TARGET voxcpm2_tts)
    target_link_libraries(crispasr PUBLIC voxcpm2_tts)
endif()
if (TARGET kokoro)
    target_link_libraries(crispasr PUBLIC kokoro)
endif()
if (TARGET fireredpunc)
    target_link_libraries(crispasr PUBLIC fireredpunc)
endif()
if (TARGET truecaser)
    target_link_libraries(crispasr PUBLIC truecaser)
endif()
if (TARGET truecaser_crf)
    target_link_libraries(crispasr PUBLIC truecaser_crf)
endif()
if (TARGET truecaser_lstm)
    target_link_libraries(crispasr PUBLIC truecaser_lstm)
endif()
if (TARGET pcs)
    target_link_libraries(crispasr PUBLIC pcs)
endif()
# Pyannote segmentation is not a backend; it's consumed by
# crispasr_diarize.cpp for the "pyannote" method. Still needs linking in
# so the native diarizer's symbols resolve at libcrispasr load time.
if (TARGET pyannote-seg)
    target_link_libraries(crispasr PUBLIC pyannote-seg)
endif()
# Silero LID engine used by crispasr_lid.cpp's native Silero path.
if (TARGET silero-lid)
    target_link_libraries(crispasr PUBLIC silero-lid)
endif()
# CLD3 text-LID — sibling of lid-fasttext, routed via the auto-routing
# text-LID dispatcher in src/text_lid_dispatch.cpp.
if (TARGET lid-cld3)
    target_link_libraries(crispasr PUBLIC lid-cld3)
endif()
# Text-LID dispatcher — selects between fastText and CLD3 by GGUF arch.
# crispasr (the main library) and crispasr-lid both go through this.
if (TARGET text-lid-dispatch)
    target_link_libraries(crispasr PUBLIC text-lid-dispatch)
endif()

# Text-LLM chat backend — PRIVATE link: llama.h identifiers are
# absorbed into libcrispasr and don't bleed into the public surface.
# Symbols downstream consumers see are only the `crispasr_chat_*` ABI
# from include/crispasr_chat.h. See docs/prompts/chat-abi.md.
if (TARGET crispasr-llama-core)
    target_link_libraries(crispasr PRIVATE crispasr-llama-core)
endif()

install(TARGETS crispasr LIBRARY)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/crispasr_chat.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Convenience alias: produce `libwhisper.{dylib,so,dll}` as a symlink (or
# copy on Windows) next to `libcrispasr`, pointing at the same file. Lets
# legacy consumers that know the project as crispasr — not as
# CrispASR — open the library under its old name.
if (BUILD_SHARED_LIBS)
    set_target_properties(crispasr PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(crispasr PRIVATE CRISPASR_SHARED CRISPASR_BUILD)
    if (APPLE)
        add_custom_command(TARGET crispasr POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:crispasr>
                $<TARGET_FILE_DIR:crispasr>/libwhisper.dylib
            COMMENT "Creating libwhisper.dylib -> $<TARGET_FILE_NAME:crispasr> alias"
        )
    elseif (UNIX)
        add_custom_command(TARGET crispasr POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:crispasr>
                $<TARGET_FILE_DIR:crispasr>/libwhisper.so
            COMMENT "Creating libwhisper.so -> $<TARGET_FILE_NAME:crispasr> alias"
        )
    elseif (WIN32)
        add_custom_command(TARGET crispasr POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:crispasr>
                $<TARGET_FILE_DIR:crispasr>/whisper.dll
            COMMENT "Copying $<TARGET_FILE:crispasr> to whisper.dll alias"
        )
    endif()
endif()
