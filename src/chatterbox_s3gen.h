#pragma once

// chatterbox_s3gen.h — S3Gen model (CFM flow matching) for Chatterbox TTS.
//
// Converts speech tokens from T3 → mel-spectrogram via:
//   1. UpsampleConformerEncoder (6+4 conformer blocks, 2x upsample)
//   2. ConditionalDecoder (UNet1D with causal convolutions)
//   3. Euler ODE solver (10 steps, cosine schedule)
//   4. HiFTGenerator vocoder (mel → 24 kHz waveform)

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct chatterbox_s3gen_context;

struct chatterbox_s3gen_context* chatterbox_s3gen_init_from_file(const char* path, int n_threads, int verbosity,
                                                                 bool use_gpu);
void chatterbox_s3gen_set_seed(struct chatterbox_s3gen_context* ctx, uint32_t seed);

// Run the full S3Gen pipeline: speech tokens → 24 kHz PCM.
// Conditioning: prompt_token (ref speech tokens), prompt_feat (ref mel),
// embedding (speaker x-vector). These come from conds.pt or live VE/CAMPPlus.
// Returns malloc'd float array, caller frees with chatterbox_s3gen_pcm_free.
float* chatterbox_s3gen_synthesize(struct chatterbox_s3gen_context* ctx, const int32_t* speech_tokens,
                                   int n_speech_tokens,
                                   // S3Gen conditioning (from conds.pt precomputed or live)
                                   const int32_t* prompt_tokens, int n_prompt_tokens, const float* prompt_feat,
                                   int prompt_feat_len,        // (T, 80)
                                   const float* spk_embedding, // (192,)
                                   int n_cfm_steps,            // 0 = default (10)
                                   int* out_n_samples);

// Run S3Gen through the CFM decoder and return the generated mel only,
// excluding the prompt-conditioning region. Returned layout is
// channel-first (80 * T_mel). Caller frees with chatterbox_s3gen_pcm_free.
float* chatterbox_s3gen_synthesize_mel(struct chatterbox_s3gen_context* ctx, const int32_t* speech_tokens,
                                       int n_speech_tokens, const int32_t* prompt_tokens, int n_prompt_tokens,
                                       const float* prompt_feat, int prompt_feat_len, const float* spk_embedding,
                                       int n_cfm_steps, int* out_T_mel);

// Diff-only entry point: same as chatterbox_s3gen_synthesize_mel but
// starts the Euler solver from a caller-provided full initial latent
// noise tensor in channel-first layout (80 * T_total, including prompt).
float* chatterbox_s3gen_synthesize_mel_with_noise(struct chatterbox_s3gen_context* ctx, const int32_t* speech_tokens,
                                                  int n_speech_tokens, const int32_t* prompt_tokens,
                                                  int n_prompt_tokens, const float* prompt_feat, int prompt_feat_len,
                                                  const float* spk_embedding, int n_cfm_steps,
                                                  const float* init_noise_cf, int init_noise_T_total, int* out_T_mel);

// Run only the vocoder on externally-provided mel.
// mel_cf: channel-first (80 * T_mel) float array.
float* chatterbox_s3gen_vocode(struct chatterbox_s3gen_context* ctx, const float* mel_cf, int T_mel,
                               int* out_n_samples);

// Run the vocoder on externally-provided mel plus an externally-provided
// source STFT from the upstream HiFT path. source_stft_cf uses
// channel-first layout (18 * T_src).
float* chatterbox_s3gen_vocode_with_source_stft(struct chatterbox_s3gen_context* ctx, const float* mel_cf, int T_mel,
                                                const float* source_stft_cf, int T_src, int* out_n_samples);

// Run vocoder and dump per-stage intermediate outputs.
// stage_names: array of C strings (e.g. "voc_conv_pre", "voc_ups_0", ...),
// stage_data: caller-allocated array of float* (set to malloc'd buffers on return),
// stage_sizes: caller-allocated array filled with element counts.
// Returns PCM like chatterbox_s3gen_vocode. Caller frees stage_data[i] with free().
float* chatterbox_s3gen_vocode_dump(struct chatterbox_s3gen_context* ctx, const float* mel_cf, int T_mel,
                                    int* out_n_samples, const char** stage_names, float** stage_data, int* stage_sizes,
                                    int n_stages);

float* chatterbox_s3gen_vocode_dump_with_source_stft(struct chatterbox_s3gen_context* ctx, const float* mel_cf,
                                                     int T_mel, const float* source_stft_cf, int T_src,
                                                     int* out_n_samples, const char** stage_names, float** stage_data,
                                                     int* stage_sizes, int n_stages);

// Diff/debug: reconstruct the final HiFT waveform directly from the
// conv_post output tensor. stft_cf is channel-first (18 * T_stft),
// matching the dumped "voc_conv_post" reference layout.
float* chatterbox_s3gen_hift_from_conv_post(const float* stft_cf, int T_stft, int T_mel, int* out_n_samples);

// S3Tokenizer V2 — module 3 of the native voice clone path. Encodes a
// 16 kHz mono float32 PCM buffer into 25 Hz speech tokens (codebook
// size = 3^8 = 6561) using the FSMN-augmented Whisper encoder + FSQ
// codebook stored in the S3Gen GGUF. `max_tokens > 0` truncates the
// output (matches `S3Tokenizer.forward(..., max_len=...)`); pass 0 / a
// negative value for the full token stream. On success returns a malloc'd
// int32 buffer of `*out_n_tokens` entries; caller frees with `free()`.
int32_t* chatterbox_s3gen_tokenize_pcm(struct chatterbox_s3gen_context* ctx, const float* pcm_16k, int n_samples,
                                       int max_tokens, int* out_n_tokens);

// Diff/debug: each stage of the S3Tokenizer pipeline. Reference dumper
// stages: `s3tok_log_mel`, `s3tok_proj_down`, `s3tok_tokens`. Each
// function returns a malloc'd buffer the caller releases with `free()`.
//   - s3tok_log_mel    → (n_mels=128 * T) f32, channel-first
//   - s3tok_proj_down  → (T_tok * 8) f32 row-major, post-FSQ-projdown
//                        pre-tanh/round (the float input to the codebook)
//   - s3tok_tokens     → (T_tok,) f32 holding int values in [0, 6561)
//                        — float32 for the GGUF reference archive's
//                        single-dtype contract
float* chatterbox_s3gen_dump_s3tok_log_mel(struct chatterbox_s3gen_context* ctx, const float* pcm_16k, int n_samples,
                                           int* out_T);
float* chatterbox_s3gen_dump_s3tok_proj_down(struct chatterbox_s3gen_context* ctx, const float* pcm_16k, int n_samples,
                                             int max_tokens, int* out_T_tok);
float* chatterbox_s3gen_dump_s3tok_tokens(struct chatterbox_s3gen_context* ctx, const float* pcm_16k, int n_samples,
                                          int max_tokens, int* out_T_tok);

// CAMPPlus fbank front-end (Module 4 phase 1). Computes 80-bin Kaldi-style
// fbank features for the 16 kHz mono ref audio and applies the per-utterance
// mean subtraction that `xvector.extract_feature` does. Returns a malloc'd
// (T_frames * 80) f32 row-major buffer; caller frees with `free()`.
float* chatterbox_s3gen_dump_campplus_fbank(struct chatterbox_s3gen_context* ctx, const float* pcm_16k, int n_samples,
                                            int* out_T);

// CAMPPlus speaker encoder forward (Module 4 phase 2). Runs the full FCM
// head + xvector chain on the 16 kHz mono ref audio and returns the 192-d
// speaker x-vector that S3Gen consumes as `gen.embedding`. Returns a
// malloc'd 192-float buffer; caller frees with `free()`.
float* chatterbox_s3gen_dump_campplus_xvector(struct chatterbox_s3gen_context* ctx, const float* pcm_16k,
                                              int n_samples);

// Conformer encoder output (Module 5 phase 1). Runs `flow.encoder` then
// `flow.encoder_proj` over [prompt_tokens | speech_tokens], returning
// the (80, T_mel) f32 channel-first pre-CFM-denoiser hidden states.
// T_mel = 2 * (n_prompt + n_speech_tokens). Writes T_mel into
// *out_T_mel. Caller frees with `free()`. Used by crispasr-diff to
// split "Conformer encoder breaks on GPU" from "CFM denoiser breaks
// on GPU" when the downstream `s3gen_mel` cos drops.
float* chatterbox_s3gen_dump_encoder_out(struct chatterbox_s3gen_context* ctx, const int32_t* speech_tokens,
                                         int n_speech_tokens, const int32_t* prompt_tokens, int n_prompt_tokens,
                                         int* out_T_mel);

// 24 kHz Matcha-TTS prompt mel (Module 4 phase 3). Computes the
// (T_mel, 80) row-major mel spectrogram for the 24 kHz mono ref audio
// — the `gen.prompt_feat` cond S3Gen's CFM denoiser consumes. Truncates
// the input to `max_samples` if non-zero (set to 240000 = 10 * 24 kHz
// to match `prepare_conditionals.DEC_COND_LEN`; pass 0 for no
// truncation). Returns a malloc'd (T_mel * 80) f32 buffer; caller
// frees with `free()`.
float* chatterbox_s3gen_dump_prompt_feat_24k(struct chatterbox_s3gen_context* ctx, const float* pcm_24k, int n_samples,
                                             int max_samples, int* out_T_mel);

void chatterbox_s3gen_pcm_free(float* pcm);
void chatterbox_s3gen_free(struct chatterbox_s3gen_context* ctx);

#ifdef __cplusplus
}
#endif
