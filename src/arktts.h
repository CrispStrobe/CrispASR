#pragma once

// ArkTTS (Audio8-TTS-Realistic-0.6b) public C ABI.
//
// Edge0/Audio8-TTS-Realistic-0.6b:
//   Slow Autoregressive LM: 24 layers, dim 896, 14 heads, 2 KV heads (GQA),
//     vocab 155776 (Qwen2 base text + semantic audio tokens 151678..155773).
//   Fast Subnetwork (Residual Audio Synthesis / RAS): 4 layers, dim 896,
//     predicts 10 acoustic codebooks (codebook size 4096).
//   Codec Decoder: 44.1 kHz multi-scale Snake-activated causal vocoder.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct arktts_context;

struct arktts_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature; // 0 = greedy
    float top_p;       // nucleus sampling
    int top_k;         // top-k sampling
    uint64_t seed;     // RNG seed (0 = default 42)
    int max_tokens;    // max generation steps
    bool flash_attn;
};

struct arktts_context_params arktts_context_default_params(void);

// Initialize context from unified GGUF model.
struct arktts_context* arktts_init_from_file(const char* path_model, struct arktts_context_params params);

// Load tokenizer from tokenizer.json (if not embedded).
int arktts_load_tokenizer(struct arktts_context* ctx, const char* tokenizer_json_path);

// Set reference audio for zero-shot voice cloning.
int arktts_set_reference(struct arktts_context* ctx, const float* audio_44k, int n_samples, const char* reference_text);

// Output sample rate (44100 Hz).
int arktts_get_sample_rate(const struct arktts_context* ctx);

// Sampling controls
void arktts_set_temperature(struct arktts_context* ctx, float temperature);
void arktts_set_seed(struct arktts_context* ctx, uint64_t seed);

// Synthesize text to 44.1 kHz mono float PCM. Caller frees with arktts_free_audio.
float* arktts_synthesize(struct arktts_context* ctx, const char* text, int* out_n);

void arktts_free_audio(float* pcm);
void arktts_free(struct arktts_context* ctx);

#ifdef __cplusplus
}
#endif
