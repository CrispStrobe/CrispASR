#
# crispasr-quantize — GGUF re-quantization tool.
#
# Takes any GGUF model (whisper, parakeet, canary, cohere, voxtral, qwen3,
# granite, wav2vec2, …) and re-quantizes all eligible tensors to a new
# ggml_ftype. Originally shipped as `cohere-quantize` under
# examples/cohere-main/ but the tool itself is model-agnostic, so it
# moved out into its own directory when the per-model CLIs were
# consolidated into `crispasr`.
#
# Usage:
#   crispasr-quantize input.gguf output-q4_k.gguf q4_k
#

set(TARGET crispasr-quantize)
add_executable(${TARGET} main.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common ggml ggml-base ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME BUNDLE DESTINATION bin)
