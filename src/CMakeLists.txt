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
            )
add_library(whisper ALIAS crispasr)

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
            )
target_include_directories(crispasr-core PUBLIC . ../include)
target_compile_features   (crispasr-core PUBLIC cxx_std_17)
target_link_libraries     (crispasr-core PUBLIC ggml-base ggml)
if(NOT MSVC)
    target_compile_options(crispasr-core PRIVATE -O3)
endif()
set_target_properties(crispasr-core PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
# ---------------------------------------------------------------------------
add_library(moonshine STATIC moonshine.h moonshine.cpp moonshine-impl.h
            moonshine-audio.h moonshine-tokenizer.h moonshine-tokenizer.cpp)
set_target_properties(moonshine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(moonshine PUBLIC .)
target_compile_features   (moonshine PUBLIC cxx_std_17)
target_link_libraries     (moonshine PUBLIC ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(moonshine PRIVATE -O3)
endif()

add_library(moonshine_streaming STATIC moonshine_streaming.h moonshine_streaming.cpp
            moonshine-tokenizer.h moonshine-tokenizer.cpp)
set_target_properties(moonshine_streaming PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(moonshine_streaming PUBLIC .)
target_compile_features   (moonshine_streaming PUBLIC cxx_std_17)
target_link_libraries     (moonshine_streaming PUBLIC ggml ggml-base ggml-cpu)
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
# vibevoice — Microsoft VibeVoice-ASR (σ-VAE tokenizers + Qwen2 LM)
# ---------------------------------------------------------------------------
add_library(vibevoice STATIC vibevoice.h vibevoice.cpp)
set_target_properties(vibevoice PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(vibevoice PUBLIC .)
target_compile_features   (vibevoice PUBLIC cxx_std_17)
target_link_libraries     (vibevoice PUBLIC crispasr-core ggml ggml-base ggml-cpu)

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
if (TARGET omniasr)
    target_link_libraries(crispasr PUBLIC omniasr)
endif()
if (TARGET vibevoice)
    target_link_libraries(crispasr PUBLIC vibevoice)
endif()
if (TARGET ecapa-lid)
    target_link_libraries(crispasr PUBLIC ecapa-lid)
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
if (TARGET kokoro)
    target_link_libraries(crispasr PUBLIC kokoro)
endif()
if (TARGET fireredpunc)
    target_link_libraries(crispasr PUBLIC fireredpunc)
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

install(TARGETS crispasr LIBRARY)

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
                libwhisper.dylib
            WORKING_DIRECTORY $<TARGET_FILE_DIR:crispasr>
            COMMENT "Creating libwhisper.dylib -> $<TARGET_FILE_NAME:crispasr> alias"
        )
    elseif (UNIX)
        add_custom_command(TARGET crispasr POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:crispasr>
                libwhisper.so
            WORKING_DIRECTORY $<TARGET_FILE_DIR:crispasr>
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
