#pragma once

// Qwen3-TTS public C ABI.
//
// Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base is a "discrete multi-codebook
// LM" — a Qwen3 backbone (28 layers, 16Q/8KV, head_dim 128) with a
// `codec_head` that emits codebook-0 of a 16-codebook RVQ, plus a
// 5-layer `code_predictor` AR LM that fills in codebooks 1..15 given
// the talker's hidden state and the previous codes. The codec that
// turns codes back into 24 kHz waveform lives in the SEPARATE
// Qwen/Qwen3-TTS-Tokenizer-12Hz repo and gets its own context (loaded
// via `qwen3_tts_set_codec_path`).
//
// Status (April 2026): the talker forward is implemented and produces
// codebook-0 streams for a text prompt. The 15-codebook code_predictor
// and the codec decoder are still pending — see PLAN #52 step 3.
// `qwen3_tts_synthesize` returns nullptr until the codec lands;
// `qwen3_tts_synthesize_codes` works end-to-end for codebook-0 today.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct qwen3_tts_context;

struct qwen3_tts_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature;  // 0 = greedy
    int max_codec_steps; // upper bound on AR decode steps; 0 = use built-in default (1500)
};

struct qwen3_tts_context_params qwen3_tts_context_default_params(void);

// Initialise from the talker LM GGUF file.
struct qwen3_tts_context* qwen3_tts_init_from_file(const char* path_model, struct qwen3_tts_context_params params);

// Point the runtime at the codec GGUF (cstr/qwen3-tts-tokenizer-12hz-GGUF).
// Required before the first `qwen3_tts_synthesize` call. Returns 0 on success.
int qwen3_tts_set_codec_path(struct qwen3_tts_context* ctx, const char* path);

// Optional: reference-audio path (16 kHz mono WAV) for voice cloning
// (Base / CustomVoice variants). Pass nullptr / "" to use the model's
// built-in default voice. Currently a no-op until the speaker_encoder
// forward lands.
int qwen3_tts_set_voice_prompt(struct qwen3_tts_context* ctx, const char* wav_path);

// Load a voice pack GGUF (produced by `models/bake-qwen3-tts-voice-pack.py`)
// containing one or more `(spk_embedding, ref_code)` pairs extracted via
// the official qwen-tts package. Required for voice-clone synthesis
// until the runtime ECAPA speaker_encoder + codec encoder forwards
// land.  Returns 0 on success.
int qwen3_tts_load_voice_pack(struct qwen3_tts_context* ctx, const char* path);

// Select an active voice from the loaded voice pack by name.
// Returns 0 on success, -1 if no voice pack is loaded, -2 if the
// name is not in the pack.
int qwen3_tts_select_voice(struct qwen3_tts_context* ctx, const char* name);

// Set the synthesis language: 0=auto (no language hint, "nothink"
// path), >0 = codec_language_id from the model config (e.g. English=2050,
// Chinese=2055, Japanese=2058 — see the `codec_language_id` field in the
// HF config.json's talker_config). Returns 0 on success.
int qwen3_tts_set_language(struct qwen3_tts_context* ctx, int codec_language_id);

// ---------------------------------------------------------------------------
// Diff-harness stage APIs (PLAN #52 step 4)
//
// These expose intermediate activations without driving the AR decode
// loop, so `crispasr-diff qwen3-tts` can verify each stage of the
// talker against the qwen_tts PyTorch reference. They mirror the
// stage names that `tools/reference_backends/qwen3_tts.py` dumps.
// ---------------------------------------------------------------------------

// text_embedding(ids) → text_projection: returns the post-resize-MLP
// activations of shape (n_tokens, hidden_size). Caller frees with free().
// *out_T = n_tokens, *out_d = hidden_size on success.
//
// Pure-text path that doesn't depend on the speaker_embed / codec
// splice, so a numerical mismatch here implicates only the
// text_embedding lookup or the text_proj fc1/fc2.
float* qwen3_tts_run_text_proj(struct qwen3_tts_context* ctx, const int32_t* ids, int n_tokens, int* out_T, int* out_d);

// Run the talker prefill on a caller-supplied embedding tensor of shape
// (n_tokens, hidden_size). Returns the codec_head logits at the LAST
// position (= what greedy AR decode would sample first). *out_vocab is
// set to vocab_size (3072). Caller frees with free().
//
// Decouples "is the talker graph numerically correct" from "is the
// prefill builder semantically correct" — feed in a PyTorch-prebuilt
// embedding, expect bit-equivalent logits at the tail.
float* qwen3_tts_run_talker_with_embeds(struct qwen3_tts_context* ctx, const float* embeds, int n_tokens,
                                        int* out_vocab);

// Build the full ICL prefill embedding from a (syn_text, ref_text) pair
// using the active voice pack's spk_embedding + ref_code. Returns a
// freshly malloc'd float buffer of shape (T, hidden_size). *out_T is
// set to T on success. Caller frees with free().
//
// Mirrors `Qwen3TTSForConditionalGeneration.generate_icl_prompt` for
// the non_streaming_mode=False voice-clone Base path.
float* qwen3_tts_build_icl_prefill(struct qwen3_tts_context* ctx, const char* syn_text, const char* ref_text,
                                   int* out_T);

// Run the talker on `text`, AR-decode codebook-0 until <eos> or the
// step limit, and return the resulting code stream. *out_n_codes is
// set to the number of codes produced. Caller frees with
// `qwen3_tts_codes_free`. Returns nullptr on failure.
//
// This is the path you can use today even without the codec — the
// codes are valid Qwen3-TTS codec inputs; you can render them via the
// HF python codec for audio.
int32_t* qwen3_tts_synthesize_codes(struct qwen3_tts_context* ctx, const char* text, int* out_n_codes);

void qwen3_tts_codes_free(int32_t* codes);

// Decode a flat code array (T_frames * 16 codes, row-major [T, 16]) to
// 24 kHz mono float32 PCM. Requires `qwen3_tts_set_codec_path` to have
// been called first. Caller frees with `qwen3_tts_pcm_free`.
// *out_n_samples is set on success; returns nullptr on failure.
float* qwen3_tts_decode_codes(struct qwen3_tts_context* ctx,
                              const int32_t* codes, int n_codes,
                              int* out_n_samples);

// Synthesise text → 24 kHz mono float32 PCM. Caller frees with
// `qwen3_tts_pcm_free`. *out_n_samples is set on success.
//
// Returns nullptr until the codec decoder lands (PLAN #52 step 3).
float* qwen3_tts_synthesize(struct qwen3_tts_context* ctx, const char* text, int* out_n_samples);

void qwen3_tts_pcm_free(float* pcm);

void qwen3_tts_free(struct qwen3_tts_context* ctx);

void qwen3_tts_set_n_threads(struct qwen3_tts_context* ctx, int n_threads);

#ifdef __cplusplus
}
#endif
