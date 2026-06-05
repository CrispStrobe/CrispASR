// csm_tts.h -- C API for Sesame CSM-1B TTS (Conversational Speech Model).
//
// Architecture:
//   Backbone:       Llama-3.2 1B (16L, 32H/8KVH, 2048d, RoPE, SwiGLU, RMSNorm)
//                   Autoregressively generates first-codebook Mimi tokens.
//   Depth decoder:  Llama-3.2 100M (4L, 8H/2KVH, 1024d)
//                   Fills remaining 31 codebooks from backbone hidden state.
//   Mimi codec:     Kyutai Mimi decoder (SEANet + 8L transformer + upsample)
//                   32-codebook RVQ -> upsampling conv -> 24 kHz PCM.
//
// Text flow:
//   Llama-3.2 BPE tokenize -> backbone AR loop (one frame = 32 codebooks)
//   -> depth decoder (7 iterations per frame) -> Mimi decode -> 24 kHz PCM
//
// Speaker conditioning: encode reference audio with Mimi encoder, prepend
//   the encoded tokens as context to the backbone prompt.
//
// Reference: github.com/SesameAILabs/csm (Apache 2.0)
//            HuggingFace: sesame/csm-1b

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct csm_tts_context;

struct csm_tts_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    float temperature;    // 0 = greedy, >0 = top-k sampling (default 0.9)
    int topk;             // top-k parameter for sampling (default 50)
    uint64_t seed;        // RNG seed (0 = non-deterministic)
    int max_audio_tokens; // max backbone AR steps (0 = default 2048)
};

struct csm_tts_context_params csm_tts_context_default_params(void);

// Load CSM-1B from a single GGUF file (backbone + depth decoder + Mimi codec).
struct csm_tts_context* csm_tts_init_from_file(const char* path_model, struct csm_tts_context_params params);

void csm_tts_free(struct csm_tts_context* ctx);

// Synthesize text to 24 kHz mono float32 PCM.
// Caller frees with csm_tts_pcm_free(). *out_n_samples is set on success.
// Returns nullptr on failure.
float* csm_tts_synthesize(struct csm_tts_context* ctx, const char* text, int* out_n_samples);

// Synthesize with speaker conditioning from reference audio.
// ref_pcm: 24 kHz mono float32 PCM of reference speaker.
// ref_text: transcript of reference audio (for text-audio alignment).
float* csm_tts_synthesize_with_reference(struct csm_tts_context* ctx, const char* text, const float* ref_pcm,
                                         int ref_n_samples, const char* ref_text, int* out_n_samples);

void csm_tts_pcm_free(float* pcm);

// ---------------------------------------------------------------------------
// Diagnostic entry point for the diff harness (crispasr-diff csm).
// ---------------------------------------------------------------------------
// Runs the backbone PREFILL on `text` (text-only frames, audio masked out)
// and dumps per-layer activations so they can be compared against the
// PyTorch manual reference (csm_reference_manual.py) stage by stage. No
// sampling — fully deterministic given the tokenized prompt.
//
// Caller pre-allocates buffers sized with `max_T` capacity (T = number of
// text-frame positions; the caller usually takes it from the reference
// tensor shape). Buffers filled on success:
//   layer0_normed  : [max_T * d_model]        layer-0 pre-attn RMSNorm output
//   layer_outputs  : n_layers pointers, each [max_T * d_model]
//                    residual stream after each backbone layer
//   last_h         : [d_model]                final-norm output, last position
//   c0_logits      : [audio_vocab_size]       codebook0_head @ last_h
// Any of the above may be NULL to skip that capture.
//
// Returns the actual number of text-frame positions T (>0), or <=0 on error.
int csm_tts_run_backbone_dump(struct csm_tts_context* ctx, const char* text, int max_T, float* layer0_normed,
                              float** layer_outputs, int n_layers, float* last_h, float* c0_logits);

// Diagnostic: depth-decoder frame-0 dump. Feed a backbone hidden state
// `last_h` (e.g. the reference backbone_prefill_last_h, to isolate the depth
// decoder from any backbone drift) plus the codebook-0 token, run the depth
// decoder's initial 2-position step, and dump:
//   initial_proj : [2 * dd_d_model]   projection of [last_h, c0_embed] -> dd dim
//   c1_logits    : [audio_vocab_size] codebook-1 logits (codebooks_head[0])
// Either output may be NULL. Returns 0 on success, <0 on error.
int csm_tts_run_depth_dump(struct csm_tts_context* ctx, const float* last_h, int c0_token, float* initial_proj,
                           float* c1_logits);

// Diagnostic: run the FULL greedy generation (backbone AR loop + depth decoder
// for all 32 codebooks per frame) via the real synth path, and copy the
// resulting codes into out_codes as a row-major [n_frames * audio_num_codebooks]
// int array (caller allocates max_frames_cap * audio_num_codebooks). Greedy
// (temperature forced to 0) so it matches a greedy PyTorch reference. Returns
// the number of frames generated, or <0 on error.
int csm_tts_run_generate_codes(struct csm_tts_context* ctx, const char* text, int32_t* out_codes, int max_frames_cap);

// Diagnostic: feed reference codes (row-major [T * n_cb]) into the Mimi decode
// path to isolate it from token-generation drift. Dumps the RVQ-dequantized
// continuous representation into rvq_out ([T * mimi_dim]). Returns 0 on success.
int csm_tts_run_mimi_dump(struct csm_tts_context* ctx, const int32_t* codes_T_cb, int T, int n_cb, float* rvq_out);

// Diagnostic: synthesize `text` (capped to frame_cap frames so greedy runs
// can't run away) and write a 24 kHz mono 16-bit PCM WAV to wav_path
// (peak-normalised). Returns the number of PCM samples written, or <0 on error.
int csm_tts_diag_synth_wav(struct csm_tts_context* ctx, const char* text, const char* wav_path, float temperature,
                           int frame_cap);

// Runtime parameter setters.
void csm_tts_set_temperature(struct csm_tts_context* ctx, float temperature);
void csm_tts_set_topk(struct csm_tts_context* ctx, int topk);
void csm_tts_set_seed(struct csm_tts_context* ctx, uint64_t seed);
void csm_tts_set_n_threads(struct csm_tts_context* ctx, int n_threads);

#ifdef __cplusplus
}
#endif
