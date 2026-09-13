#pragma once

// fireredtts3_tts.h — public C ABI for the FireRedTTS3 TTS backend (#377).
//
// FireRedTTS3 (FireRedTeam/FireRedTTS3, Apache-2.0) is a continuous-latent
// AR zero-shot voice-cloning TTS system:
//   - Qwen3-1.7B LLM backbone (28L, hidden=2048, 16Q/8KV, head_dim=128,
//     QK-norm, NEOX RoPE theta=1e6) over [spk_emb, text tokens, patches]
//   - PatchEncoder: 8L bidirectional transformer (hidden=1024); each patch
//     of 4 RedAE latent frames + [CLS] → one 2048-d LLM embedding
//   - DiT flow-matching head: 11L AdaLN blocks (hidden=1024, +conv branch),
//     input concat(latents 64, LLM cond 1024, spk cond 512); denoises one
//     4-frame patch per AR step over a 2-patch clean history, cosine
//     t-schedule, CFG 2.0, sigmoid stop head
//   - RedAE: 64-d 25 Hz latent autoencoder @ 24 kHz (Qwen3 stacks with
//     sliding-window 64 + Vocos-style ISTFT head, n_fft=1920 hop=480)
//   - CAM++ speaker encoder (512-d x-vector from 80-bin Kaldi fbank)
//
// Two GGUF files:
//   fireredtts3-base-*.gguf   — LLM + PatchEncoder + DiT + heads + tokenizer
//                               (+ optional baked default voice prompt)
//   fireredtts3-redae-*.gguf  — RedAE encoder+decoder + CAM++ (campplus.*)
//
// Voice cloning is ICL: the prompt transcript is REQUIRED and is prepended
// to the target text as `<|Lang|><|sot|>{ref_text}{text}<|eot|>` while the
// prompt's RedAE latents prefill the decoder. Without set_voice_*, the
// default prompt baked into the core GGUF is used.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct fireredtts3_tts_context;

struct fireredtts3_tts_context_params {
    int n_threads;
    int verbosity;        // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    uint64_t seed;        // RNG seed; 0 = upstream default 1234
    int n_timesteps;      // flow ODE steps; 0 = default 10
    float cfg_scale;      // classifier-free guidance; < 0 = default 2.0
    float stop_threshold; // stop when sigmoid(stop_head) >= this; 0 = 0.5
    int max_patches;      // max AR steps (1 patch = 160 ms); 0 = default 400
};

struct fireredtts3_tts_context_params fireredtts3_tts_context_default_params(void);

// Load the core model (LLM + PatchEncoder + DiT + tokenizer).
struct fireredtts3_tts_context* fireredtts3_tts_init_from_file(const char* path_model,
                                                               struct fireredtts3_tts_context_params params);

// Load the RedAE + CAM++ companion GGUF (required before synthesis).
// Returns 0 on success.
int fireredtts3_tts_set_redae_path(struct fireredtts3_tts_context* ctx, const char* path);

// Set the reference voice from 16 kHz mono float PCM (requires the redae
// companion: RedAE-encodes the prompt and computes the CAM++ x-vector).
// The prompt transcript must be provided via set_ref_text. Returns 0.
int fireredtts3_tts_set_voice_pcm(struct fireredtts3_tts_context* ctx, const float* pcm_16k, int n_samples);

// Transcript of the reference audio (ICL prompt text). Required whenever a
// voice was set via set_voice_pcm.
int fireredtts3_tts_set_ref_text(struct fireredtts3_tts_context* ctx, const char* text);

// Language tag ("English", "Chinese", "German", ... — upstream tag names).
// Default: the baked prompt's language, else "English".
int fireredtts3_tts_set_language(struct fireredtts3_tts_context* ctx, const char* language);

// Synthesize text to 24 kHz mono float32 PCM. Returns malloc'd
// float[*out_n_samples]; caller frees with fireredtts3_tts_pcm_free().
float* fireredtts3_tts_synthesize(struct fireredtts3_tts_context* ctx, const char* text, int* out_n_samples);

void fireredtts3_tts_pcm_free(float* pcm);

int fireredtts3_tts_sample_rate(const struct fireredtts3_tts_context* ctx); // 24000

void fireredtts3_tts_free(struct fireredtts3_tts_context* ctx);

// Self-contained per-stage diff runner against a reference GGUF produced by
// tools/reference_backends/fireredtts3_tts.py (crispasr-diff "fireredtts3").
// prompt_wav_path is the same prompt WAV the reference used (16 kHz mono).
// Returns 0 when every checked stage passes.
int fireredtts3_tts_diff(const char* core_gguf, const char* redae_gguf, const char* ref_gguf,
                         const char* prompt_wav_path, int verbosity);

#ifdef __cplusplus
}
#endif
