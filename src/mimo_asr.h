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
struct mimo_asr_context* mimo_asr_init_from_file(const char* path_model, struct mimo_asr_context_params params);

// Transcribe PCM audio (16 kHz mono float32). The runtime handles
// the call into the audio-tokeniser GGUF internally — point it at
// the path via `mimo_asr_set_tokenizer_path` before the first call.
char* mimo_asr_transcribe(struct mimo_asr_context* ctx, const float* pcm, int n_samples);

// Set the path to the audio-tokeniser GGUF (cstr/mimo-tokenizer-GGUF).
// Required before the first transcribe call. Returns 0 on success.
int mimo_asr_set_tokenizer_path(struct mimo_asr_context* ctx, const char* path);

// Stage extraction for the diff harness. Runs the prefill graph on the
// caller-supplied [9, T_total] input_ids tensor (channel 0 = text mostly
// <|empty|>, channels 1..8 = audio codes per channel) and returns a
// freshly-malloc'd float buffer holding one of:
//   prefill_audio_features      [llm_hidden, T_groups]   F32 (post group_proj)
//   prefill_inputs_embeds       [llm_hidden, T_groups]   F32 (LM input)
//   prefill_last_hidden         [llm_hidden]             F32 (post final norm)
//   prefill_text_logits_step0   [vocab]                  F32 (lm_head out)
// Writes the total element count to *n_out. Caller must free().
// Returns nullptr on failure or unknown stage.
float* mimo_asr_extract_stage(struct mimo_asr_context* ctx, const int32_t* input_ids_9xT, int T_total,
                              const char* stage, int* n_out);

// Read out a few hyperparameters needed by the diff harness for layout
// validation (each pointer may be NULL to skip).
int mimo_asr_get_hparams(struct mimo_asr_context* ctx, uint32_t* llm_hidden, uint32_t* llm_vocab, uint32_t* audio_dim,
                        uint32_t* audio_channels, uint32_t* audio_group_size);

// Free context and all associated memory.
void mimo_asr_free(struct mimo_asr_context* ctx);

void mimo_asr_set_n_threads(struct mimo_asr_context* ctx, int n_threads);

#ifdef __cplusplus
}
#endif
