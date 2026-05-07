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

float* chatterbox_s3gen_vocode_dump_with_source_stft(struct chatterbox_s3gen_context* ctx, const float* mel_cf, int T_mel,
                                                     const float* source_stft_cf, int T_src, int* out_n_samples,
                                                     const char** stage_names, float** stage_data, int* stage_sizes,
                                                     int n_stages);

void chatterbox_s3gen_pcm_free(float* pcm);
void chatterbox_s3gen_free(struct chatterbox_s3gen_context* ctx);

#ifdef __cplusplus
}
#endif
