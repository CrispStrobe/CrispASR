// voxtral4b.h — public C API for Mistral Voxtral-Mini-4B-Realtime-2602
//
// Audio-LLM with a RoPE+SwiGLU Whisper-style encoder + 4-frame stack
// projector + Llama 3 (Mistral) 3.4B LLM with adaptive RMSNorm and
// sliding window attention. Models loaded from GGUF files produced by:
//   `python models/convert-voxtral4b-to-gguf.py --input <hf_dir> --output X.gguf`
//
// Key differences from Voxtral-Mini-3B (voxtral.h):
//   - Encoder uses RoPE (not learned absolute pos embed)
//   - Encoder uses SwiGLU FFN (not GELU fc1/fc2)
//   - Encoder uses RMSNorm (not LayerNorm)
//   - Encoder + LLM use sliding window attention
//   - LLM has adaptive RMSNorm (time-conditioned)
//   - LLM uses tied embeddings (output = token_embd transposed)
//   - LLM: 26 layers, FFN=9216, RoPE θ=1e6

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct voxtral4b_context;

struct voxtral4b_context_params {
    int n_threads;
    int verbosity;     // 0=silent 1=normal 2=verbose
};

struct voxtral4b_context_params voxtral4b_context_default_params(void);

struct voxtral4b_context * voxtral4b_init_from_file(const char * path_model,
                                                    struct voxtral4b_context_params params);

void voxtral4b_free(struct voxtral4b_context * ctx);

const uint8_t * voxtral4b_token_text(struct voxtral4b_context * ctx, int id, int * out_len);

int32_t * voxtral4b_tokenize(struct voxtral4b_context * ctx,
                             const char * text, int * out_n_tokens);

float * voxtral4b_compute_mel(struct voxtral4b_context * ctx,
                              const float * samples, int n_samples,
                              int * out_n_mels, int * out_T_mel);

float * voxtral4b_run_encoder(struct voxtral4b_context * ctx,
                              const float * mel_features,
                              int n_mels, int T_mel,
                              int * out_N, int * out_dim);

float * voxtral4b_embed_tokens(struct voxtral4b_context * ctx,
                               const int32_t * input_ids, int n_tokens);

bool voxtral4b_kv_init(struct voxtral4b_context * ctx, int max_ctx);
void voxtral4b_kv_reset(struct voxtral4b_context * ctx);

float * voxtral4b_run_llm_kv(struct voxtral4b_context * ctx,
                             const float * inputs_embeds,
                             int n_tokens, int n_past,
                             int * out_n_tokens, int * out_vocab_size);

#ifdef __cplusplus
}
#endif
