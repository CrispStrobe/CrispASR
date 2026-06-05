// melotts.h — MeloTTS (VITS2) native ggml runtime.
//
// myshell-ai/MeloTTS: VITS2 architecture with multilingual support,
// BERT conditioning, transformer coupling flow, dual duration predictor
// (SDP + DP), speaker conditioning, and HiFi-GAN decoder.
//
// ~50MB GGUF (F16), 44.1 kHz mono output, 256 speakers (EN-US/BR/INDIA/AU).
//
// Text processing: built-in English G2P via embedded CMU dictionary +
// ARPAbet phoneme mapping. BERT conditioning disabled by default (zeros).

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct melotts_context;

struct melotts_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float noise_scale;  // latent noise variance (default 0.667)
    float length_scale; // duration stretch (>1 = slower)
    float noise_w;      // SDP noise variance (default 0.8)
    float sdp_ratio;    // blend SDP vs DP (0=all DP, 1=all SDP, default 0.2)
    int speaker_id;     // speaker index (0=EN-US, 1=EN-BR, etc.)
    uint32_t seed;      // RNG seed (0 = random)
};

struct melotts_params melotts_default_params(void);

// Load a MeloTTS GGUF. Returns nullptr on failure.
struct melotts_context* melotts_init_from_file(const char* path_model, struct melotts_params params);

void melotts_free(struct melotts_context* ctx);

// Load a companion BERT model for contextual conditioning.
// If loaded, synthesize() produces higher quality output.
// path: GGUF from convert-bert-base-to-gguf.py (~238 MB).
bool melotts_load_bert(struct melotts_context* ctx, const char* bert_gguf_path);

// Synthesize text to mono PCM at 44100 Hz.
// Returns number of samples, 0 on failure.
// *pcm_out is malloc'd by this function; caller frees with free().
int melotts_synthesize(struct melotts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out);

// Free PCM buffer returned by melotts_synthesize.
void melotts_pcm_free(float* pcm);

// Runtime parameter setters.
void melotts_set_noise_scale(struct melotts_context* ctx, float v);
void melotts_set_length_scale(struct melotts_context* ctx, float v);
void melotts_set_noise_w(struct melotts_context* ctx, float v);
void melotts_set_sdp_ratio(struct melotts_context* ctx, float v);
void melotts_set_speaker_id(struct melotts_context* ctx, int id);
void melotts_set_seed(struct melotts_context* ctx, uint32_t seed);

// Query model info.
int melotts_sample_rate(const struct melotts_context* ctx);
int melotts_num_speakers(const struct melotts_context* ctx);

// Diff harness: dump intermediates to this directory.
void melotts_set_dump_dir(struct melotts_context* ctx, const char* dir);

#ifdef __cplusplus
}
#endif
