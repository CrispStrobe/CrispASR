#pragma once

// MiMo-V2.5-ASR public C ABI.
//
// XiaomiMiMo/MiMo-V2.5-ASR pairs a 6-layer "input_local_transformer"
// audio-token processor with a 36-layer Qwen2 LLM. Audio enters as
// 8-channel RVQ codes from the SEPARATE `cstr/mimo-tokenizer-GGUF`
// model, which the host wires up via `mimo_asr_set_audio_tokens`
// (or via the backend-level helper that does it for you).

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mimo_asr_context;

struct mimo_asr_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature; // 0 = greedy
};

struct mimo_asr_context_params mimo_asr_context_default_params(void);

// Initialise from the LM GGUF file (cstr/mimo-asr-GGUF).
// Returns nullptr on failure.
struct mimo_asr_context* mimo_asr_init_from_file(const char* path_model,
                                                 struct mimo_asr_context_params params);

// Transcribe PCM audio (16 kHz mono float32). The runtime handles
// the call into the audio-tokeniser GGUF internally — point it at
// the path via `mimo_asr_set_tokenizer_path` before the first call.
char* mimo_asr_transcribe(struct mimo_asr_context* ctx, const float* pcm, int n_samples);

// Set the path to the audio-tokeniser GGUF (cstr/mimo-tokenizer-GGUF).
// Required before the first transcribe call. Returns 0 on success.
int mimo_asr_set_tokenizer_path(struct mimo_asr_context* ctx, const char* path);

// Free context and all associated memory.
void mimo_asr_free(struct mimo_asr_context* ctx);

void mimo_asr_set_n_threads(struct mimo_asr_context* ctx, int n_threads);

#ifdef __cplusplus
}
#endif
