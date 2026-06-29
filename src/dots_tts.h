#pragma once

// dots_tts.h — public C ABI for rednote-hilab/dots.tts TTS backend.
//
// dots.tts is a 2B-parameter continuous AR text-to-speech system:
//   - Qwen2.5-1.5B LLM backbone (28L, 12Q/2KV heads, hidden=1536)
//   - PatchEncoder: 24L causal transformer (VAESemanticEncoder, in=128→1024)
//   - DiT: 18L AdaLN flow-matching head (hidden=1024)
//   - BigVGAN vocoder: 6-stage upsample (960x) with SnakeBeta, 48 kHz
//   - CAM++ speaker encoder: 80-mel → 512-d embedding
//
// The inference loop generates audio patch-by-patch:
//   1. Text tokens → LLM → hidden states at audio-span positions
//   2. LLM hidden → DiT conditions; noise → Euler ODE → denoised latent
//   3. Latent → PatchEncoder → embedding → feed back to LLM
//   4. All latents → BigVGAN decoder → 48 kHz PCM
//
// Three GGUF files:
//   - Core (LLM + DiT + PatchEncoder + tokenizer): dots-tts-soar-f16.gguf
//   - Vocoder (BigVGAN): dots-tts-soar-vocoder-f16.gguf
//   - Speaker (CAM++): dots-tts-soar-spk-f16.gguf (optional, for voice cloning)

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct dots_tts_context;

struct dots_tts_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature;   // sampling temperature for LLM decode (0 = greedy)
    uint64_t seed;       // RNG seed (0 = default 42)
    int max_patches;     // max audio patches to generate; 0 = default (256 ≈ 40s)
    int ode_steps;       // flow-matching ODE steps; 0 = default (16)
    float cfg_scale;     // classifier-free guidance scale; 0 = default (3.0)
    float eos_threshold; // stop when eos_proj stop-prob exceeds this; 0 = default (0.8)
    bool flash_attn;     // flash attention for LLM/PatchEncoder
};

struct dots_tts_context_params dots_tts_context_default_params(void);

// Load core model (LLM + DiT + PatchEncoder + tokenizer).
struct dots_tts_context* dots_tts_init_from_file(const char* path_model, struct dots_tts_context_params params);

// Load vocoder GGUF (required before synthesis). Returns 0 on success.
int dots_tts_set_vocoder_path(struct dots_tts_context* ctx, const char* path);

// Load speaker encoder GGUF (optional, for voice cloning). Returns 0 on success.
int dots_tts_set_speaker_path(struct dots_tts_context* ctx, const char* path);

// Set reference audio for voice cloning (requires speaker encoder).
// wav_path must be a mono WAV file. Returns 0 on success.
int dots_tts_set_voice_prompt(struct dots_tts_context* ctx, const char* wav_path);

// Set reference voice directly from 16 kHz mono float PCM (requires speaker
// encoder). Computes the CAM++ x-vector → xvec_proj → g_cond. Returns 0 on
// success. Used by the diff harness and CLI adapters that decode audio
// themselves.
int dots_tts_set_speaker_pcm(struct dots_tts_context* ctx, const float* pcm_16k, int n_samples);

// Enable/disable voice conditioning for subsequent synthesis without discarding
// the computed g_cond. Lets a caller synthesize a neutral utterance (e.g. the
// spoken AI-disclaimer) and then restore the cloned voice. Returns 0 on success.
int dots_tts_set_voice_enabled(struct dots_tts_context* ctx, int on);

// Synthesize text to 48 kHz mono float32 PCM.
// Returns malloc'd float[*out_n_samples]. Caller frees with dots_tts_pcm_free().
// Returns nullptr on failure.
float* dots_tts_synthesize(struct dots_tts_context* ctx, const char* text, int* out_n_samples);

// Generate latents only (no vocoder decode). For debugging.
// Returns malloc'd float[*out_n] in (n_patches * patch_size, latent_dim) layout.
float* dots_tts_generate_latents(struct dots_tts_context* ctx, const char* text, int* out_n);

void dots_tts_pcm_free(float* pcm);
void dots_tts_free(struct dots_tts_context* ctx);

// Runtime parameter setters
void dots_tts_set_n_threads(struct dots_tts_context* ctx, int n_threads);
void dots_tts_set_temperature(struct dots_tts_context* ctx, float temperature);
void dots_tts_set_seed(struct dots_tts_context* ctx, uint64_t seed);

// Diff-harness entry: validate the PatchEncoder decode_patch (patch 0) against
// a reference GGUF carrying penc_in_patch0 / penc_out_patch0 (produced by
// tools/dots_penc_reference.py). Prints cosine/max_abs; returns 0 on PASS.
int dots_tts_penc_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

// Diff-harness entry: validate the LLM (Qwen2.5) forward against a reference
// GGUF carrying llm_ids_prefill / llm_hidden_prefill / llm_step_embed /
// llm_hidden_step (prefill + one KV-cached incremental step). Returns 0 on PASS.
int dots_tts_llm_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

// Diff-harness entry: validate the DiT (velocity_field_predictor) one forward
// against a reference GGUF carrying dit_x / dit_t / dit_gcond / dit_vel.
// Prints cosine/max_abs; returns 0 on PASS.
int dots_tts_dit_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

// Diff-harness entry: validate the flow-matching ODE driver against a reference
// GGUF carrying fm_input_seq / fm_cfg_seq / fm_mask / fm_pos / fm_noise /
// fm_latent_out / fm_meta. Prints cosine/max_abs; returns 0 on PASS.
int dots_tts_flowmatch_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

// Diff-harness entry: validate the BigVGAN vocoder. voc_gguf is the VOCODER
// GGUF; ref carries voc_latent_in / voc_mi_out / voc_conv_pre / voc_audio.
// Validates the post_proj → dec_mi LSTM → conv_pre front-end; returns 0 on PASS.
int dots_tts_vocoder_diff(const char* voc_gguf, const char* ref_gguf, int verbosity);

// Diff-harness entry: validate the isolated alias-free Activation1d against a
// reference carrying act_in/act_out/act_alpha/act_beta/act_up_filter/
// act_down_filter. Returns 0 on PASS.
int dots_tts_act_diff(const char* ref_gguf, int verbosity);

// Diff-harness entry: validate the CAM++ speaker path. spk_gguf is the speaker
// encoder GGUF; ref carries spk_pcm_16k / spk_xvector / spk_g_cond plus the
// xvec_proj_{0,1}_{w,b} weights. Validates wav→x-vector→g_cond. Returns 0 on PASS.
int dots_tts_spk_diff(const char* spk_gguf, const char* ref_gguf, int verbosity);

#ifdef __cplusplus
}
#endif
