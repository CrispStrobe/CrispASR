# TODO: should not use this
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()


if (WHISPER_ALL_WARNINGS)
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

if (WHISPER_COREML)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREML_FRAMEWORK CoreML)

    if (COREML_FRAMEWORK)
        message(STATUS "CoreML framework found")

        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_USE_COREML)
    else()
        message(FATAL_ERROR "CoreML framework not found")
    endif()

    if (WHISPER_COREML_ALLOW_FALLBACK)
        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_COREML_ALLOW_FALLBACK)
    endif()
endif()

if (WHISPER_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()

#
# libraries
#

# whisper.coreml

if (WHISPER_COREML)
    set(TARGET whisper.coreml)

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

if (WHISPER_OPENVINO)
    set(TARGET whisper.openvino)

    add_library(${TARGET} OBJECT
        openvino/whisper-openvino-encoder.h
        openvino/whisper-openvino-encoder.cpp
        )

    target_include_directories(${TARGET} PUBLIC
        .
        )

    set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_USE_OPENVINO)

    target_link_libraries(${TARGET} PRIVATE ggml openvino::runtime)
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

# whisper

add_library(whisper
            ../include/whisper.h
            whisper-arch.h
            whisper.cpp
            )

# Cohere Transcribe library (Conformer encoder + transformer decoder)
add_library(cohere
            cohere-arch.h
            cohere.h
            cohere.cpp
            )
target_include_directories(cohere PUBLIC . ../include)
target_compile_features   (cohere PUBLIC cxx_std_20)
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
add_library(canary
            canary.h
            canary.cpp
            )
target_include_directories(canary PUBLIC . ../include)
target_compile_features   (canary PUBLIC cxx_std_17)
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
add_library(canary_ctc
            canary_ctc.h
            canary_ctc.cpp
            )
target_include_directories(canary_ctc PUBLIC . ../include)
target_compile_features   (canary_ctc PUBLIC cxx_std_17)
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
add_library(parakeet
            parakeet.h
            parakeet.cpp
            )
target_include_directories(parakeet PUBLIC . ../include)
target_compile_features   (parakeet PUBLIC cxx_std_17)
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
add_library(qwen3_asr
            qwen3_asr.h
            qwen3_asr.cpp
            )
target_include_directories(qwen3_asr PUBLIC . ../include)
target_compile_features   (qwen3_asr PUBLIC cxx_std_17)
target_link_libraries     (qwen3_asr PUBLIC ggml ggml-base Threads::Threads crispasr-core)
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
add_library(voxtral
            voxtral.h
            voxtral.cpp
            )
target_include_directories(voxtral PUBLIC . ../include)
target_compile_features   (voxtral PUBLIC cxx_std_17)
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

add_library(voxtral4b
            voxtral4b.h
            voxtral4b.cpp
            )
target_include_directories(voxtral4b PUBLIC . ../include)
target_compile_features   (voxtral4b PUBLIC cxx_std_17)
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

add_library(granite_speech
            granite_speech.h
            granite_speech.cpp
            )
target_include_directories(granite_speech PUBLIC . ../include)
target_compile_features   (granite_speech PUBLIC cxx_std_17)
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
# wav2vec2-ggml — Wav2Vec2ForCTC encoder (GGUF, ggml-based)
# ---------------------------------------------------------------------------
add_library(wav2vec2-ggml STATIC wav2vec2-ggml.h wav2vec2-ggml.cpp)
target_include_directories(wav2vec2-ggml PUBLIC .)
target_compile_features   (wav2vec2-ggml PUBLIC cxx_std_17)
target_link_libraries     (wav2vec2-ggml PUBLIC ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(wav2vec2-ggml PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# pyannote-seg — native pyannote-segmentation-3.0 (GGUF, CPU-only)
# ---------------------------------------------------------------------------
add_library(pyannote-seg STATIC pyannote_seg.h pyannote_seg.cpp)
target_include_directories(pyannote-seg PUBLIC .)
target_compile_features   (pyannote-seg PUBLIC cxx_std_17)
target_link_libraries     (pyannote-seg PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(pyannote-seg PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# silero-lid — native Silero Language Classifier 95 (GGUF, ggml-based)
# ---------------------------------------------------------------------------
add_library(silero-lid STATIC silero_lid.h silero_lid.cpp)
target_include_directories(silero-lid PUBLIC .)
target_compile_features   (silero-lid PUBLIC cxx_std_17)
target_link_libraries     (silero-lid PUBLIC crispasr-core ggml ggml-base ggml-cpu)
if(NOT MSVC)
    target_compile_options(silero-lid PRIVATE -O3)
endif()

# ---------------------------------------------------------------------------
# ctc-align — CTC forced alignment (Viterbi DP, CPU-only, no ggml dependency)
# ---------------------------------------------------------------------------
add_library(ctc-align STATIC align.h align.cpp)
target_include_directories(ctc-align PUBLIC .)
target_compile_features   (ctc-align PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(ctc-align PRIVATE -O3)
endif()

# Set the version numbers
set_target_properties(whisper PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${SOVERSION}
)

target_include_directories(whisper PUBLIC . ../include)
target_compile_features   (whisper PUBLIC cxx_std_11) # don't bump

if (CMAKE_CXX_BYTE_ORDER STREQUAL "BIG_ENDIAN")
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_BIG_ENDIAN)
endif()

if (WHISPER_EXTRA_FLAGS)
    target_compile_options(whisper PRIVATE ${WHISPER_EXTRA_FLAGS})
endif()

find_package(Threads REQUIRED)
target_link_libraries(whisper PUBLIC ggml Threads::Threads)

if (WHISPER_COREML)
    target_link_libraries(whisper PRIVATE whisper.coreml)
endif()

if (WHISPER_OPENVINO)
    target_link_libraries(whisper PRIVATE whisper.openvino)
endif()

if (WHISPER_MKL)
    target_link_libraries(whisper PRIVATE MKL::MKL)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(whisper PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(whisper PRIVATE WHISPER_SHARED WHISPER_BUILD)
endif()
