// outetts.h -- OuteTTS public C ABI.
//
// OuteAI/OuteTTS-0.3-1B: OLMo-1B (Llama-compatible) finetuned to emit
// interleaved text + audio tokens, decoded by the WavTokenizer single-
// codebook VQ-GAN into 24 kHz PCM.  CC BY 4.0 license.
//
// V2 prompt format:
//   <|im_start|>\n
//   <|text_start|>word1<|space|>word2<|text_end|>\n
//   <|audio_start|>\n
//   [AR generates: word1<|t_0.24|><|147|><|523|>... <|space|>\n ...]
//   <|audio_end|>\n<|im_end|>
//
// Audio tokens <|0|>..<|4099|> map to WavTokenizer codebook indices 0-4095
// (with a 4-index offset for the first 4 reserved entries in some variants).
// The LLM generates until <|audio_end|>; caller extracts audio code IDs,
// subtracts the audio_token_offset, and decodes via the WavTokenizer decoder.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct outetts_context;

struct outetts_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature;    // 0 = greedy (default 0.4 per upstream)
    uint64_t seed;        // RNG seed (0 = default)
    int max_audio_tokens; // upper bound on AR decode steps; 0 = built-in default (4096)
    bool flash_attn;
};

struct outetts_context_params outetts_context_default_params(void);

// Load the OuteTTS LLM GGUF (arch="outetts").
struct outetts_context* outetts_init_from_file(const char* path_model, struct outetts_context_params params);

// Point at the WavTokenizer decoder GGUF. Required before synthesize.
int outetts_set_codec_path(struct outetts_context* ctx, const char* path);

// Synthesize text -> 24 kHz mono float32 PCM.
// Caller frees with outetts_pcm_free. Returns nullptr on failure.
float* outetts_synthesize(struct outetts_context* ctx, const char* text, int* out_n_samples);

// Run the LLM, return raw audio code indices (WavTokenizer codebook).
int32_t* outetts_synthesize_codes(struct outetts_context* ctx, const char* text, int* out_n);

void outetts_codes_free(int32_t* codes);
void outetts_pcm_free(float* pcm);
void outetts_free(struct outetts_context* ctx);
void outetts_set_n_threads(struct outetts_context* ctx, int n_threads);
void outetts_set_temperature(struct outetts_context* ctx, float temperature);
void outetts_set_seed(struct outetts_context* ctx, uint64_t seed);

// Load a speaker profile JSON for voice cloning.
// JSON format: {"text": "...", "words": [{"word": "...", "duration": 0.53, "codes": [123, 456, ...]}, ...]}
// Created by tools/reference_backends/outetts_create_speaker.py or the outetts library's create_speaker().
int outetts_load_speaker(struct outetts_context* ctx, const char* json_path);

#ifdef __cplusplus
}
#endif
