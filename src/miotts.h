#pragma once

// MioTTS public C ABI.
//
// Aratako/MioTTS-{0.6B,1.7B} — LLM-based TTS (Qwen3 backbone) that generates
// speech tokens from text, decoded by MioCodec at the sample rate stored in the GGUF.
//
// Architecture (0.6B):
//   LLM: Qwen3ForCausalLM, 28 layers, 1024 hidden, GQA 16/8, vocab 164480
//        (Qwen3 BPE text tokens + 12800 speech tokens <|s_0|>..<|s_12799|>)
//   Codec: MioCodec (25 Hz tokens, 24 kHz or v2 44.1 kHz audio) — FSQ(levels=[8,8,8,5,5]) dequant → wave_prenet
//        transformer → conv_upsample → ResNet → wave_decoder transformer
//        (AdaLN-Zero conditioned on 128-d speaker embedding) → ResNet → iSTFT
//
// The model generates speech tokens by autoregressive sampling from the
// LLM's full vocabulary. Speaker identity (voice cloning) is injected at
// codec decode time via a global embedding extracted from reference audio.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct miotts_context;

struct miotts_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature; // 0 = greedy
    uint64_t seed;     // RNG seed (0 = use default 42)
    int max_tokens;    // upper bound on AR decode steps; 0 = default (750 = 30s at 25Hz)
    bool flash_attn;
};

struct miotts_context_params miotts_context_default_params(void);

// Initialise from a GGUF containing both LLM + codec weights.
// tokenizer_json_path is optional — if non-null, loads the Qwen3 BPE
// tokenizer from this file. If null, the tokenizer is loaded from the
// GGUF (if embedded) or synthesis requires pre-tokenized input.
struct miotts_context* miotts_init_from_file(const char* path_model, struct miotts_context_params params);

// Load tokenizer from a tokenizer.json file (Qwen3 BPE). Returns 0 on success.
int miotts_load_tokenizer(struct miotts_context* ctx, const char* tokenizer_json_path);

// Set reference audio for voice cloning. The codec extracts a 128-d global
// embedding from this audio to condition the waveform decoder.
// Pass nullptr/0 to clear (uses a zero embedding = default voice).
// Returns 0 on success.
int miotts_set_reference(struct miotts_context* ctx, const float* audio_24k, int n_samples);

// Load a preset speaker embedding from a .emb.gguf or raw binary file.
int miotts_load_preset_embedding(struct miotts_context* ctx, const char* emb_path);

// Output sample rate from miotts.codec.sample_rate (legacy models default to
// 24000 Hz). A null context returns the legacy default.
int miotts_get_sample_rate(const struct miotts_context* ctx);

// Synthesize speech from text. Returns a freshly allocated float buffer of
// mono PCM at miotts_get_sample_rate(ctx) (caller must free with miotts_free_audio). *out_n receives
// the sample count. Returns nullptr on failure.
float* miotts_synthesize(struct miotts_context* ctx, const char* text, int* out_n);

void miotts_free_audio(float* pcm);
void miotts_free(struct miotts_context* ctx);

// For the diff harness: run the LLM forward on the given token IDs and return
// the logits for the last position. Caller must free with miotts_free_audio.
float* miotts_forward_logits(struct miotts_context* ctx, const int32_t* token_ids, int n_tokens, int* out_vocab);

// For the diff harness: run FSQ dequant on speech token indices and return
// the content embedding. Caller must free with miotts_free_audio.
float* miotts_fsq_dequant(struct miotts_context* ctx, const int32_t* indices, int n_indices, int* out_dim);

// For the diff harness: run MioCodec wave_prenet on FSQ embeddings.
// Input: fsq_emb is (T * 768) floats. Output: (T * 512) floats.
float* miotts_wave_prenet_forward(struct miotts_context* ctx, const float* fsq_emb, int T, int* out_dim);

// Run the full codec decode: wave_prenet output → conv_upsample → ResNet →
// wave_decoder (AdaLN-Zero) → ResNet → iSTFT → model-rate PCM.
// Input: prenet_out is (T * 512) floats. Output: PCM samples at miotts_get_sample_rate(ctx).
float* miotts_codec_decode(struct miotts_context* ctx, const float* prenet_out, int T_prenet, int* out_n);

#ifdef __cplusplus
}
#endif
