#pragma once

// Supertonic-3 TTS (Supertone/supertonic-3, OpenRAIL-M) public C ABI.
//
// Non-autoregressive flow-matching TTS, 44.1 kHz, 31 languages, 10 preset
// voices, ~99 M params. The upstream distribution is ONNX-only; this runtime
// is a native ggml port of the four graphs (duration predictor, text
// encoder, vector estimator, vocoder) converted by
// models/convert-supertonic3-to-gguf.py into ONE GGUF (arch
// "supertonic-tts") that embeds the unicode indexer, NFKD tables and all
// ten voices.
//
// Pipeline (mirrors supertone-inc/supertonic py/helper.py exactly):
//   text -> preprocess (NFKD, cleanup, "<lang>...</lang>" wrap) -> char ids
//        -> duration predictor (CPU)   -> seconds / speed
//        -> text encoder (CPU)         -> text_emb [256, L]
//        -> noisy latent  randn(144, N), N = ceil(dur*sr/3072)
//        -> vector estimator x total_steps (ggml graph; classifier-free
//           guidance INSIDE each step: v = 4*v_cond - 3*v_uncond,
//           xt += v/total_steps)
//        -> vocoder (ggml graph)       -> wav, trimmed to sr*dur
//
// Long text is chunked at sentence boundaries (300 chars, 120 for ko/ja)
// and joined with 0.3 s silences, as upstream does.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct supertonic_context;

struct supertonic_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    int total_steps;  // flow-matching steps, default 8
    float speed;      // speech speed, default 1.05 (upstream default)
    char voice[16];   // one of F1..F5, M1..M5; default "M1"
    char language[8]; // ISO code from the supported set; default "en"
    uint64_t seed;    // 0 = nondeterministic noise
};

struct supertonic_context_params supertonic_context_default_params(void);

struct supertonic_context* supertonic_init_from_file(const char* path_model, struct supertonic_context_params params);
void supertonic_free(struct supertonic_context* ctx);

void supertonic_set_n_threads(struct supertonic_context* ctx, int n_threads);
int supertonic_set_voice(struct supertonic_context* ctx, const char* voice);   // 0 on success
int supertonic_set_language(struct supertonic_context* ctx, const char* lang); // 0 on success
void supertonic_set_speed(struct supertonic_context* ctx, float speed);
void supertonic_set_seed(struct supertonic_context* ctx, uint64_t seed);
void supertonic_set_total_steps(struct supertonic_context* ctx, int steps);

int supertonic_sample_rate(const struct supertonic_context* ctx);

// Synthesize. Returns malloc'd mono PCM at supertonic_sample_rate();
// caller frees. NULL on failure.
float* supertonic_synthesize(struct supertonic_context* ctx, const char* text, int* out_n_samples);

// ── diff-harness stage APIs (crispasr-diff supertonic-tts) ──────────────
// All buffers are float32. Layout of 2-D stages matches the reference
// dumper: [C, T] with C fastest (flat index t*C + c).

// Preprocess + indexer. Returns id count (<= cap written to out_ids).
int supertonic_debug_text_ids(struct supertonic_context* ctx, const char* text, const char* lang, int32_t* out_ids,
                              int cap);
// Duration AFTER /speed for the current voice/speed.
float supertonic_debug_duration(struct supertonic_context* ctx, const int32_t* ids, int n);
// Text encoder output [256, L]; out must hold 256*n floats. Also exposes
// the two interior probes when non-NULL (each 256*n floats).
int supertonic_debug_text_encoder(struct supertonic_context* ctx, const int32_t* ids, int n, float* out_text_emb,
                                  float* out_convnext, float* out_pre_spte);
// Run the flow with INJECTED noise xt0 [144, N]. out_steps must hold
// total_steps*144*N floats (xt after each step, concatenated).
// out_projin_s0 (optional, 512*N) = cond-half proj_in output at step 0.
int supertonic_debug_flow(struct supertonic_context* ctx, const float* text_emb, int n_text, const float* xt0,
                          int n_latent, float* out_steps, float* out_projin_s0);
// Self-contained per-stage diff vs a reference GGUF from
// tools/reference_backends/supertonic_tts.py. Returns 0 = ALL PASS.
int supertonic_tts_diff(const char* model_path, const char* ref_path, int verbosity);
// Vocoder: latent [144, N] -> wav. Returns sample count (<= cap).
int supertonic_debug_vocoder(struct supertonic_context* ctx, const float* latent, int n_latent, float* out_wav,
                             int cap);

#ifdef __cplusplus
}
#endif
