// src/lfm2_audio.h — public C API for LiquidAI LFM2.5-Audio ggml runtime
//
// End-to-end multimodal audio model: FastConformer encoder → audio adapter
// → LFM2 hybrid conv+attention backbone → depthformer. Supports ASR (audio
// → text) and TTS (text → audio via depthformer + Mimi detokenizer).
//
// Models are loaded from GGUF files produced by:
//   python models/convert-lfm2-audio-to-gguf.py
//       --input LiquidAI/LFM2.5-Audio-1.5B-JP --output lfm2-audio.gguf
//
// License: LFM Open License v1.0 ($10M revenue cap for commercial use).

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct lfm2_audio_context;

struct lfm2_audio_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
};

struct lfm2_audio_context_params lfm2_audio_context_default_params(void);

struct lfm2_audio_context* lfm2_audio_init_from_file(const char* path_model, struct lfm2_audio_context_params params);

void lfm2_audio_free(struct lfm2_audio_context* ctx);

// Transcribe raw 16 kHz mono PCM. Returns malloc'd string the caller must free().
// prompt: system prompt (e.g. "Perform ASR in japanese."). NULL for default.
// max_tokens: max text tokens to generate (0 = default 512).
char* lfm2_audio_transcribe(struct lfm2_audio_context* ctx, const float* samples, int n_samples, const char* prompt,
                            int max_tokens);

// Synthesize speech from text. Returns malloc'd PCM at 24 kHz mono.
// text: input text to speak. language: "ja" or "en". NULL = default.
// out_n_samples: receives the sample count.
// Returns NULL on failure. Caller frees with free().
float* lfm2_audio_synthesize(struct lfm2_audio_context* ctx, const char* text, const char* language,
                             int* out_n_samples);

// Speech-to-speech: audio in → interleaved text+audio out → PCM at 24 kHz.
// Combines ASR (conformer encoder on input) with TTS (depthformer + detokenizer).
// out_text: if non-NULL, receives malloc'd transcript text. Caller frees.
// out_n_samples: receives output PCM sample count.
// Returns output PCM at 24 kHz mono. Caller frees with free().
float* lfm2_audio_speech_to_speech(struct lfm2_audio_context* ctx, const float* in_samples, int n_in_samples,
                                   const char* language, char** out_text, int* out_n_samples);

// Streaming TTS: synthesize with per-chunk audio callback.
// The callback receives PCM chunks (~1920 samples = 80ms at 24 kHz)
// as they're generated. Returns 0 on success, -1 on failure.
// Set cb to NULL for batch mode (equivalent to lfm2_audio_synthesize).
typedef void (*lfm2_audio_stream_cb)(const float* pcm_chunk, int n_samples, void* userdata);
int lfm2_audio_synthesize_stream(struct lfm2_audio_context* ctx, const char* text, const char* language,
                                 lfm2_audio_stream_cb cb, void* userdata);

// Hyper-parameters
int lfm2_audio_n_mels(struct lfm2_audio_context* ctx);
int lfm2_audio_sample_rate(struct lfm2_audio_context* ctx);

// ---- Stage-level entry points (for crispasr-diff testing) ----

// Log-mel spectrogram. Row-major (T_mel, n_mels). Caller frees.
float* lfm2_audio_compute_mel(struct lfm2_audio_context* ctx, const float* samples, int n_samples, int* out_T_mel,
                              int* out_n_mels);

// FastConformer encoder. Row-major (T_enc, d_model). Caller frees.
float* lfm2_audio_run_encoder(struct lfm2_audio_context* ctx, const float* mel, int T_mel, int n_mels, int* out_T_enc,
                              int* out_d_model);

// Audio adapter MLP. Row-major (T_enc, hidden_size). Caller frees.
float* lfm2_audio_run_adapter(struct lfm2_audio_context* ctx, const float* encoder_out, int T_enc, int d_model,
                              int* out_hidden_size);

// LFM2 backbone. Row-major (T, hidden_size). Caller frees.
// Runs on the full input sequence (text tokens + audio embeddings).
float* lfm2_audio_run_lfm(struct lfm2_audio_context* ctx, const float* samples, int n_samples, int* out_T,
                          int* out_hidden_size);

// Staged LFM backbone: invokes cb at every layer boundary.
typedef void (*lfm2_audio_stage_cb)(const char* stage_name, const float* data, int rows, int cols, void* userdata);
int lfm2_audio_run_lfm_staged(struct lfm2_audio_context* ctx, const float* samples, int n_samples,
                              lfm2_audio_stage_cb cb, void* userdata);

// Internal smoke test.
int lfm2_audio_test_load(struct lfm2_audio_context* ctx);

// Beam search for ASR decode. 1 = greedy (default). >1 = beam search
// via core_beam_decode::run_with_probs_branched with KV + conv state
// snapshot/restore (§167h). Needs Kaggle for testing (no local model).
void lfm2_audio_set_beam_size(struct lfm2_audio_context* ctx, int beam_size);

#ifdef __cplusplus
}
#endif
