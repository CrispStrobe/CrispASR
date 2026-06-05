#pragma once

// Chatterbox public C ABI.
//
// ResembleAI/chatterbox (MIT) — a multi-stage TTS pipeline:
//   1. T3 (520M Llama AR) — text + conditioning → speech tokens @25 Hz
//   2. S3Gen (CFM) — speech tokens → mel-spectrogram via flow matching
//   3. HiFTGenerator — mel → 24 kHz waveform
//
// Voice cloning uses a reference WAV processed through:
//   - VoiceEncoder (3-layer LSTM) → 256D speaker embedding for T3
//   - S3Tokenizer → speech tokens for T3 prompt conditioning
//   - CAMPPlus → 192D x-vector for S3Gen speaker conditioning
//
// The default built-in voice uses precomputed conditioning baked into
// the T3 GGUF (from conds.pt), so no reference audio is needed for
// basic synthesis.
//
// Two GGUFs are required:
//   - chatterbox-t3-f16.gguf   (T3 model + VE + tokenizer + conds)
//   - chatterbox-s3gen-f16.gguf (S3Gen flow + vocoder + CAMPPlus)

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct chatterbox_context;

struct chatterbox_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature;        // AR sampling temperature (default 0.8)
    float cfg_weight;         // classifier-free guidance weight (default 0.5)
    float exaggeration;       // emotion exaggeration factor (default 0.5)
    float repetition_penalty; // repetition penalty (default 1.2)
    float min_p;              // min_p sampling (default 0.05)
    float top_p;              // top_p sampling (default 1.0)
    int top_k;                // top_k sampling, 0 = disabled (default 0).
                              // chatterbox-turbo enables top_k=1000 + top_p=0.95
                              // + min_p=0 to match HF inference_turbo defaults
                              // (tts_turbo.py:248-260, t3.py:415).
    int max_speech_tokens;    // upper bound on T3 AR decode (default 1000)
    int cfm_steps;            // number of CFM Euler steps (default 10)
    bool flash_attn;          // PLAN #89 plumbing — T3 Llama-style AR
                              // loop. Highest-impact target alongside
                              // orpheus for the kernel wiring in #86.
};

struct chatterbox_context_params chatterbox_context_default_params(void);

// Initialise from the T3 GGUF (arch="chatterbox", produced by
// models/convert-chatterbox-to-gguf.py).
struct chatterbox_context* chatterbox_init_from_file(const char* path_model, struct chatterbox_context_params params);

// Point the runtime at the S3Gen GGUF (arch="chatterbox-s3gen").
// Required before the first chatterbox_synthesize call. Returns 0
// on success.
int chatterbox_set_s3gen_path(struct chatterbox_context* ctx, const char* path);

// Synthesise text → 24 kHz mono float32 PCM using the built-in voice.
// Caller frees with chatterbox_pcm_free. *out_n_samples is set on
// success. Returns nullptr on error.
float* chatterbox_synthesize(struct chatterbox_context* ctx, const char* text, int* out_n_samples);

// Run T3 + S3Gen stages: text → mel spectrogram (80 channels).
// Returns channel-first float array (80 * T_mel), caller frees with free().
// *out_T_mel is set to the number of mel frames.
float* chatterbox_synthesize_mel(struct chatterbox_context* ctx, const char* text, int* out_T_mel);

// Run only the T3 stage: text → speech tokens. Caller frees with
// chatterbox_tokens_free. *out_n is set to token count.
int32_t* chatterbox_synthesize_tokens(struct chatterbox_context* ctx, const char* text, int* out_n);

// Synthesise from pre-generated speech tokens (bypasses T3, runs S3Gen+vocoder).
// Uses precomputed conditioning from conds.pt. Caller frees with chatterbox_pcm_free.
float* chatterbox_synthesize_from_tokens(struct chatterbox_context* ctx, const int32_t* speech_tokens,
                                         int n_speech_tokens, int* out_n_samples);

// Run S3Gen only on pre-generated speech tokens and return the
// generated mel-spectrogram (channel-first, 80 * T_mel).
float* chatterbox_synthesize_mel_from_tokens(struct chatterbox_context* ctx, const int32_t* speech_tokens,
                                             int n_speech_tokens, int* out_T_mel);

// Diff-only entry point: replay S3Gen from caller-provided full initial
// diffusion noise in channel-first layout (80 * T_total, including prompt).
float* chatterbox_synthesize_mel_from_tokens_with_noise(struct chatterbox_context* ctx, const int32_t* speech_tokens,
                                                        int n_speech_tokens, const float* init_noise_cf,
                                                        int init_noise_T_total, int* out_T_mel);

// Run only the HiFT vocoder on a channel-first mel tensor (80 * T_mel).
float* chatterbox_vocode_mel(struct chatterbox_context* ctx, const float* mel_cf, int T_mel, int* out_n_samples);

// Run the HiFT vocoder on a channel-first mel tensor with an externally
// supplied upstream source STFT (18 * T_src, channel-first).
float* chatterbox_vocode_mel_with_source_stft(struct chatterbox_context* ctx, const float* mel_cf, int T_mel,
                                              const float* source_stft_cf, int T_src, int* out_n_samples);

float* chatterbox_vocode_mel_dump_with_source_stft(struct chatterbox_context* ctx, const float* mel_cf, int T_mel,
                                                   const float* source_stft_cf, int T_src, int* out_n_samples,
                                                   const char** stage_names, float** stage_data, int* stage_sizes,
                                                   int n_stages);

// Diff/debug: reconstruct the final HiFT waveform directly from a dumped
// conv_post tensor. stft_cf uses channel-first layout (18 * T_stft),
// matching the "voc_conv_post" reference archive tensor.
float* chatterbox_hift_from_conv_post(const float* stft_cf, int T_stft, int T_mel, int* out_n_samples);

// Set voice from a reference WAV path for voice cloning.
// Requires VE (in T3 GGUF) + S3Tokenizer + CAMPPlus (in S3Gen GGUF).
// Returns 0 on success.
int chatterbox_set_voice_from_wav(struct chatterbox_context* ctx, const char* wav_path);

// Set the emotion exaggeration factor (0.0–2.0).
void chatterbox_set_exaggeration(struct chatterbox_context* ctx, float exaggeration);

// Set classifier-free guidance weight.
void chatterbox_set_cfg_weight(struct chatterbox_context* ctx, float cfg_weight);

// Set number of CFM denoising steps (1–100).
void chatterbox_set_cfm_steps(struct chatterbox_context* ctx, int steps);

// Runtime sampling knobs — read on every chatterbox_synthesize call,
// so mutation between calls is safe. Pre-sweep the call site already
// reads `ctx->params.X` on each AR sample, hence the runtime path.
void chatterbox_set_temperature(struct chatterbox_context* ctx, float temperature);
void chatterbox_set_top_p(struct chatterbox_context* ctx, float top_p);
void chatterbox_set_min_p(struct chatterbox_context* ctx, float min_p);
void chatterbox_set_top_k(struct chatterbox_context* ctx, int top_k);
void chatterbox_set_repetition_penalty(struct chatterbox_context* ctx, float r);
void chatterbox_set_max_speech_tokens(struct chatterbox_context* ctx, int n);
void chatterbox_set_seed(struct chatterbox_context* ctx, uint32_t seed);

void chatterbox_tokens_free(int32_t* tokens);
void chatterbox_pcm_free(float* pcm);

void chatterbox_free(struct chatterbox_context* ctx);
void chatterbox_set_n_threads(struct chatterbox_context* ctx, int n_threads);

// Diff/debug: VoiceEncoder (Module 2 of the native voice clone path) stages.
// `pcm_16k` is mono float32 PCM at 16 kHz — same format the python reference
// dumper feeds `model.ve.embeds_from_wavs([audio], 16000)`. Each call returns
// a malloc'd buffer the caller releases with `free()`:
//   - chatterbox_dump_ve_mel        → (T * 40) f32 row-major (raw-amp Slaney mel)
//   - chatterbox_dump_ve_partial_emb→ (n_partials * 256) f32 row-major (L2-normed per partial)
//   - chatterbox_dump_ve_speaker_emb→ (256,) f32 (mean across partials, L2-normed)
float* chatterbox_dump_ve_mel(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples, int* out_T);
float* chatterbox_dump_ve_partial_emb(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples,
                                      int* out_n_partials);
float* chatterbox_dump_ve_speaker_emb(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples);

// Diff/debug: S3Tokenizer V2 stages (Module 3 of native voice clone).
// Forwarders to the same hooks on the underlying s3gen context — keeps the
// diff harness pointed at the chatterbox handle rather than reaching for
// the s3gen sub-context directly.
float* chatterbox_dump_s3tok_log_mel(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples, int* out_T);
float* chatterbox_dump_s3tok_proj_down(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples,
                                       int max_tokens, int* out_T_tok);
float* chatterbox_dump_s3tok_tokens(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples, int max_tokens,
                                    int* out_T_tok);

// Diff/debug: CAMPPlus fbank front-end (Module 4 phase 1). Forwarder to
// the s3gen sub-context. Returns (T_frames * 80) f32 row-major; caller
// frees with `free()`.
float* chatterbox_dump_campplus_fbank(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples, int* out_T);

// Diff/debug: CAMPPlus 192-d speaker x-vector (Module 4 phase 2).
// Forwarder to the s3gen sub-context. Returns a malloc'd 192-float buffer;
// caller frees with `free()`.
float* chatterbox_dump_campplus_xvector(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples);

// Diff/debug: 24 kHz Matcha-TTS prompt mel for `gen.prompt_feat`
// (Module 4 phase 3). Forwarder to the s3gen sub-context. Returns a
// malloc'd (T_mel * 80) f32 row-major buffer; caller frees with `free()`.
float* chatterbox_dump_prompt_feat_24k(struct chatterbox_context* ctx, const float* pcm_24k, int n_samples,
                                       int max_samples, int* out_T_mel);

// Diff/debug: Conformer encoder output (Module 5 phase 1). Forwarder
// to the s3gen sub-context's `chatterbox_s3gen_dump_encoder_out`,
// supplying the voice's `gen.prompt_token` from the loaded conds.
// Returns a malloc'd (80 * T_mel) f32 channel-first buffer; T_mel
// written into *out_T_mel. Caller frees with `free()`. Used by
// crispasr-diff to split "Conformer encoder breaks on GPU" from
// "CFM denoiser breaks on GPU" when downstream `s3gen_mel` cos drops.
float* chatterbox_dump_s3gen_encoder_out(struct chatterbox_context* ctx, const int32_t* speech_tokens,
                                         int n_speech_tokens, int* out_T_mel);

// Diff/debug: return the T3 prefill embeddings for the given text (output of
// build_prefill_embeds, excluding the extra BOS). Shape: (*out_T, *out_D).
// Also sets *out_cond_T to the number of conditioning tokens (cond_len).
// Caller frees result with free(). Returns nullptr on error.
float* chatterbox_dump_t3_prefill_emb(struct chatterbox_context* ctx, const char* text, int* out_T, int* out_D,
                                      int* out_cond_T);

// Diff/debug: run the T3 prefill for the given text and return the step-0
// speech logits. Returned buffers have shape (*out_V,). Any of the output
// pointers may be null; if non-null, the caller frees them with free().
// For CFG runs, `out_logits_blended = cond + cfg * (cond - uncond)`.
int chatterbox_dump_t3_step0_logits(struct chatterbox_context* ctx, const char* text, float** out_logits_cond,
                                    float** out_logits_uncond, float** out_logits_blended, int* out_V);

// Diff/debug: return next-step T3 logits after forcing a speech-token prefix.
// `prefix_tokens` are generated speech tokens after BOS, using speech positions
// 1..n_prefix. When n_prefix == 0, this is equivalent to step-0 logits.
int chatterbox_dump_t3_next_logits(struct chatterbox_context* ctx, const char* text, const int32_t* prefix_tokens,
                                   int n_prefix, float** out_logits_cond, float** out_logits_uncond,
                                   float** out_logits_blended, int* out_V);

#ifdef __cplusplus
}
#endif
