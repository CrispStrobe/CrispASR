// breeze_tts_2.h -- C API for the Breeze TTS 2 autoregressive TTS runtime.
//
// ⚠ NON-COMMERCIAL WEIGHTS. The architecture ported here is described by
// Apache-2.0 inference code (breezeblue-ai/breeze-tts, excluded from "Model
// Materials" by LICENSE §1.2), but the CHECKPOINT is under the BreezeBlue
// Research and Non-Commercial License Agreement, and §1.3 makes a quantized
// GGUF a Derivative Model that inherits those terms. The model-registry entry
// carries the licence prose so crispasr_license_requires_acceptance() forces
// CRISPASR_ACCEPT_LICENSE before an auto-download.
//
//   "Derived from Breeze TTS 2 by BreezeBlue and licensed for research and
//    non-commercial use only."
//
// The backend key is `bt2-tts`, not a name led by "Breeze": §4 of the
// Agreement bars the licensor's marks as the PRIMARY name of a derivative,
// while descriptive attribution (as in this comment, the registry description
// and the model card) is permitted.
//
// Architecture — a CSM (Sesame) fork with three components swapped:
//
//   text encoder   T5Gemma2, 26L, d=1152, 4H/1KVH, head_dim 256, GeGLU.
//                  BIDIRECTIONAL (the config key `use_bidirectional_attention:
//                  false` is a misnomer that the registered implementation
//                  inverts), with a SYMMETRIC sliding window of [i-255, i+256]
//                  on 22 layers and no mask at all on layers 5/11/17/23.
//                  Two RoPE configs: theta 1e4 on the sliding layers, and
//                  theta 1e6 with "linear" factor 8 on the full ones.
//   backbone       genuine Qwen3, 28L, d=2048, 16H/8KVH, SwiGLU, q/k-norm,
//                  theta 1e6, eps 1e-6 — all four read from the NESTED
//                  backbone_config, never the top-level decoys.
//   depth decoder  12L, d=1024, 8H/2KVH, llama3-scaled RoPE with an
//                  original context of 16 over a 33-slot sequence. Emits
//                  codebooks 1..15 through 15 separate heads.
//   codec          NOT in this file. Breeze's bundled audio tokenizer is
//                  bit-identical to Qwen3-TTS-Tokenizer-12Hz, which CrispASR
//                  already ships; it is wired in as a registry companion and
//                  driven through qwen3_tts.h.
//
// One backbone step emits codebook 0; the depth decoder then runs 15 steps to
// fill the frame. 16 codebooks at 12.5 Hz, decoded to 24 kHz mono.
//
// Languages: English and Chinese only.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct breeze_tts_2_context;

// Classifier-free guidance branches. Breeze runs the backbone over 2-3
// INDEPENDENT prompts per step — not one prompt with different conditioning
// vectors — and combines the logits as
//     logits = uncond + s_ref * (ref - uncond) + s_ins * (ins - uncond)
// at every backbone step AND again inside the depth decoder, per codebook.
// Which capabilities are reachable is therefore a function of how many
// branches the runtime can carry:
//
//   BREEZE_CFG_NONE   1 branch   Voice Clone, and plain text-to-speech
//   BREEZE_CFG_REF    2 branches Voice Clone with guidance
//   BREEZE_CFG_INS    2 branches Voice Design   (instruction only)
//   BREEZE_CFG_BOTH   3 branches Voice Direction (reference + instruction)
//
// See breeze_tts_2_capabilities() for what this build actually implements —
// the enum describes the model, not a promise about the port.
enum breeze_cfg_mode {
    BREEZE_CFG_NONE = 0,
    BREEZE_CFG_REF = 1,
    BREEZE_CFG_INS = 2,
    BREEZE_CFG_BOTH = 3,
};

struct breeze_tts_2_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    float temperature;        // backbone sampling temperature (default 0.9)
    float depth_temperature;  // depth-decoder temperature   (default 0.9)
    int topk;                 // default 50
    float topp;               // default 1.0
    float repetition_penalty; // default 1.1 (infer.py:26)
    uint64_t seed;            // 0 = non-deterministic
    int max_new_tokens;       // max backbone AR frames (0 = default 750)
    // Path to the qwen3-tts-tokenizer-12hz codec GGUF. Without it the runtime
    // can still produce CODES (and the diff harness only needs codes), but it
    // cannot produce audio.
    const char* codec_path;
};

struct breeze_tts_2_context_params breeze_tts_2_context_default_params(void);

struct breeze_tts_2_context* breeze_tts_2_init_from_file(const char* path_model,
                                                         struct breeze_tts_2_context_params params);

void breeze_tts_2_free(struct breeze_tts_2_context* ctx);

// Bitmask of the CFG modes this BUILD can actually run, so a caller can refuse
// a request instead of silently downgrading it. A capability that is not
// implemented must be visibly absent — never quietly answered with the
// single-branch result, which sounds plausible and is not what was asked for.
uint32_t breeze_tts_2_capabilities(const struct breeze_tts_2_context* ctx);

// Plain text-to-speech. Caller frees with breeze_tts_2_pcm_free().
float* breeze_tts_2_synthesize(struct breeze_tts_2_context* ctx, const char* text, int* out_n_samples);

// Voice Clone: condition on a reference clip plus its transcript.
// ref_pcm is 24 kHz mono float32.
float* breeze_tts_2_synthesize_with_reference(struct breeze_tts_2_context* ctx, const char* text, const float* ref_pcm,
                                              int ref_n_samples, const char* ref_text, int* out_n_samples);

// Voice Design / Voice Direction. `instruction` is the natural-language style
// prompt; ref_pcm may be NULL for Voice Design. Returns NULL and logs when the
// requested CFG mode is not in breeze_tts_2_capabilities().
float* breeze_tts_2_synthesize_guided(struct breeze_tts_2_context* ctx, const char* text, const char* instruction,
                                      const float* ref_pcm, int ref_n_samples, const char* ref_text,
                                      float cfg_scale_ref, float cfg_scale_ins, int* out_n_samples);

void breeze_tts_2_pcm_free(float* pcm);

// Runtime setters.
void breeze_tts_2_set_temperature(struct breeze_tts_2_context* ctx, float t);
void breeze_tts_2_set_topk(struct breeze_tts_2_context* ctx, int k);
void breeze_tts_2_set_seed(struct breeze_tts_2_context* ctx, uint64_t seed);
void breeze_tts_2_set_n_threads(struct breeze_tts_2_context* ctx, int n);
void breeze_tts_2_set_max_new_tokens(struct breeze_tts_2_context* ctx, int n);

// ---------------------------------------------------------------------------
// Diff-harness entry points (crispasr-diff bt2-tts).
//
// These mirror the stage names the Kaggle reference oracle dumps
// (tools/kaggle/breeze-refdump), so each one can be fed the ORACLE's input and
// judged on its own output — a stage fed known-good input that produces bad
// output is indicted; one fed a drifting input is not.
// ---------------------------------------------------------------------------

// Stage 1: run the text encoder over ONE segment of token ids and dump the
// final hidden state [seg_len, 1152], and optionally every layer's hidden
// state (n_layers+1 of them; index 0 is the scaled embedding).
// Segments are encoded INDEPENDENTLY upstream — concatenating them changes
// every hidden state — so this takes exactly one segment.
// Returns seg_len, or <0 on error.
int breeze_tts_2_run_text_encoder_dump(struct breeze_tts_2_context* ctx, const int32_t* seg_ids, int seg_len,
                                       float* out_hidden, float** out_layers, int n_layer_slots);

// Stage 2: text_encoder_proj applied to [n, 1152] -> [n, 2048].
int breeze_tts_2_run_text_proj_dump(struct breeze_tts_2_context* ctx, const float* hidden, int n, float* out_proj);

// Stage 3: backbone prefill over pre-assembled embeddings [L, 2048]. Dumps the
// final-norm hidden of the LAST position and the 2052-wide codebook-0 logits,
// plus per-layer hidden at that position when out_layers is non-NULL.
// Taking embeddings rather than text lets the harness substitute the oracle's
// backbone_inputs_embeds and so separate prompt assembly from the backbone.
int breeze_tts_2_run_backbone_dump(struct breeze_tts_2_context* ctx, const float* embeds, int L, float* out_hidden,
                                   float* out_logits, float** out_layers, int n_layer_slots);

// Stage 4: depth decoder, one frame. Feed a backbone hidden state (the
// oracle's backbone_hidden_frame0 isolates the depth decoder from backbone
// drift) and the codebook-0 code; dumps the per-codebook logits for
// codebooks 1..15 into out_logits_cb[c-1] ([2051] each) and the chosen codes
// into out_codes ([16], including cb0).
int breeze_tts_2_run_depth_dump(struct breeze_tts_2_context* ctx, const float* backbone_hidden, int32_t cb0,
                                float** out_logits_cb, int32_t* out_codes);

// Stage 5: full greedy generation. Writes row-major [n_frames * 16] codes.
// Greedy regardless of the context's temperature so it can be held to
// exact-integer equality against the oracle.
int breeze_tts_2_run_generate_codes(struct breeze_tts_2_context* ctx, const char* text, const float* ref_pcm,
                                    int ref_n_samples, const char* ref_text, int32_t* out_codes, int max_frames_cap);

// Stage 2b: prompt assembly ALL the way to backbone_inputs_embeds, driven by
// the oracle's own prompt ids / text mask / reference codes rather than by our
// tokenizer and our resampler. That separation is the point: if this matches
// and the ids do not, the bug is in tokenization; if this fails on oracle ids,
// the bug is in the scatter of text-encoder rows and audio embeddings.
// out_embeds is [L, hidden_size]. Returns L, or <0 on error.
int breeze_tts_2_run_prefill_embeds_dump(struct breeze_tts_2_context* ctx, const int32_t* ids, const int32_t* text_mask,
                                         int L, const int32_t* ref_codes, int ref_frames, float* out_embeds);

// Stage 5b: greedy generation from PRE-ENCODED reference codes, skipping the
// codec encoder entirely. The oracle hands its reference clip to the tokenizer
// at 16 kHz and lets it resample; the runtime pre-resamples to 24 kHz. Feeding
// the oracle's ref_codes removes that difference from the comparison, so a
// code mismatch here cannot be blamed on resampling.
int breeze_tts_2_run_generate_codes_ref(struct breeze_tts_2_context* ctx, const char* text, const char* ref_text,
                                        const int32_t* ref_codes, int ref_frames, int32_t* out_codes,
                                        int max_frames_cap);

// Prompt assembly, exposed on its own because it is invisible until the audio
// is garbage. Writes the flattened prompt token ids, the per-position text
// mask, and the per-segment text lengths, matching prompt_input_ids /
// prompt_text_ids_mask / prompt_text_ids_len in the fixture.
// Returns L (prompt length), or <0 on error.
int breeze_tts_2_run_prompt_dump(struct breeze_tts_2_context* ctx, const char* text, const char* ref_text,
                                 int n_ref_audio_frames, const char* instruction, int32_t* out_ids,
                                 int32_t* out_text_mask, int32_t* out_seg_lens, int max_L, int max_segs,
                                 int* out_n_segs);

#ifdef __cplusplus
}
#endif
