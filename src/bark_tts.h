#pragma once

// Bark TTS public C ABI.
//
// Suno Bark (MIT license) -- 3-stage hierarchical TTS:
//   Stage 1: text -> semantic tokens     (GPT-2 causal transformer)
//   Stage 2: semantic -> coarse tokens   (GPT-2 causal, 2 EnCodec codebooks)
//   Stage 3: coarse -> fine tokens       (non-causal transformer, all 8 codebooks)
//   Decode:  fine tokens -> 24 kHz PCM   (EnCodec SEANet decoder)
//
// Speaker conditioning via .npz prompt files (semantic + coarse + fine
// history tokens prepended to context). 10 German presets (v2/de_speaker_0-9).
//
// All 3 sub-models + EnCodec decoder packed into one GGUF
// (models/convert-bark-to-gguf.py).

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct bark_context;

struct bark_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature_semantic; // 0 = greedy, default 0.7
    float temperature_coarse;   // 0 = greedy, default 0.7
    float temperature_fine;     // 0 = greedy, default 0.5
    uint64_t seed;              // RNG seed (0 = non-deterministic)
    int max_semantic_tokens;    // upper bound; 0 = default (768)
    bool flash_attn;
};

struct bark_context_params bark_context_default_params(void);

// Initialise from the all-in-one GGUF (arch="bark").
struct bark_context* bark_init_from_file(const char* path_model, struct bark_context_params params);

// Returns 24000.
uint32_t bark_sample_rate(const struct bark_context* ctx);

// Set speaker prompt from an .npz file path. The .npz must contain
// "semantic_prompt", "coarse_prompt", "fine_prompt" arrays.
// Returns 0 on success.
int bark_set_speaker_npz(struct bark_context* ctx, const char* npz_path);

// Clear speaker prompt (unconditional generation).
void bark_clear_speaker(struct bark_context* ctx);

// Synthesise text -> 24 kHz mono float32 PCM. Caller frees with
// bark_pcm_free. *out_n_samples is set on success.
float* bark_synthesize(struct bark_context* ctx, const char* text, int* out_n_samples);

void bark_pcm_free(float* pcm);
void bark_free(struct bark_context* ctx);

void bark_set_n_threads(struct bark_context* ctx, int n_threads);
void bark_set_temperature_semantic(struct bark_context* ctx, float temp);
void bark_set_temperature_coarse(struct bark_context* ctx, float temp);
void bark_set_temperature_fine(struct bark_context* ctx, float temp);
void bark_set_seed(struct bark_context* ctx, uint64_t seed);

#ifdef __cplusplus
}
#endif
