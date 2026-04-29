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
    float temperature;   // 0 = greedy
    int max_codec_steps; // upper bound on AR decode steps; 0 = use built-in default (1500)
};

struct qwen3_tts_context_params qwen3_tts_context_default_params(void);

// Initialise from the talker LM GGUF file.
struct qwen3_tts_context* qwen3_tts_init_from_file(const char* path_model, struct qwen3_tts_context_params params);

// Point the runtime at the codec GGUF (cstr/qwen3-tts-tokenizer-12hz-GGUF).
// Required before the first `qwen3_tts_synthesize` call. Returns 0 on success.
int qwen3_tts_set_codec_path(struct qwen3_tts_context* ctx, const char* path);

// Set a reference voice from a 24 kHz mono WAV plus its transcription.
// Computes both the ECAPA speaker embedding AND the RVQ codec codes from
// the audio (replacing the baked voice-pack workflow). The ref_text is
// the transcription of wav_path — required for the ICL prefill to match
// the reference audio with text. Pass nullptr / "" to clear the prompt.
// Returns 0 on success.
int qwen3_tts_set_voice_prompt(struct qwen3_tts_context* ctx, const char* wav_path);

// Same as set_voice_prompt but also stores the reference transcription.
// Required for synthesis when no voice pack is loaded.
int qwen3_tts_set_voice_prompt_with_text(struct qwen3_tts_context* ctx, const char* wav_path, const char* ref_text);

// Debug: get the runtime ref codes after set_voice_prompt. Returns pointer
// to internal int32 buffer of *out_n elements ([T_codec, 16] row-major).
// Do NOT free — buffer is owned by ctx.
const int32_t* qwen3_tts_get_runtime_ref_codes(struct qwen3_tts_context* ctx, int* out_n);

// Read-only pointer to the currently active runtime speaker embedding
// (set by qwen3_tts_set_voice_prompt[_with_text]). Returns nullptr if no
// runtime prompt is active. Buffer is owned by ctx; do not free.
const float* qwen3_tts_get_runtime_spk_emb(struct qwen3_tts_context* ctx, int* out_n);

// Run the codec encoder graph on `audio` (24 kHz mono float32) and extract
// a named intermediate tensor by `stage_name`. Stage names match those set
// via ggml_set_name in build_cenc_graph:
//   "cenc_seanet_out"  — SEANet output [T_enc, 512]
//   "cenc_xfmr_out"    — Encoder transformer output [T_enc, 512]
//   "cenc_ds_out"      — After stride-2 downsample [T_frames, 512]
//   "enc_emb"          — Final embeddings (channels-first) [512, T_frames]
// Returns malloc'd float[*out_n] array. Caller frees with free().
float* qwen3_tts_cenc_extract_stage(struct qwen3_tts_context* ctx, const float* audio, int n_samples,
                                    const char* stage_name, int* out_n);

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
float* qwen3_tts_decode_codes(struct qwen3_tts_context* ctx, const int32_t* codes, int n_codes, int* out_n_samples);

// Run the codec graph on `codes` and extract a named intermediate tensor
// by `stage_name`. Useful for the diff harness — matches stage names that
// `build_graph_codec_decode` sets via ggml_set_name:
//   "codec_rvq_out", "codec_pre_conv_out", "codec_xfmr_out",
//   "codec_up0_out", "codec_up1_out", "codec_in_conv_out",
//   "codec_blk0_out", "pcm"
// Returns malloc'd float array of *out_n elements. Caller frees with free().
float* qwen3_tts_codec_extract_stage(struct qwen3_tts_context* ctx, const int32_t* codes, int n_codes,
                                     const char* stage_name, int* out_n);

// Synthesise text → 24 kHz mono float32 PCM. Caller frees with
// `qwen3_tts_pcm_free`. *out_n_samples is set on success.
//
// Returns nullptr until the codec decoder lands (PLAN #52 step 3).
float* qwen3_tts_synthesize(struct qwen3_tts_context* ctx, const char* text, int* out_n_samples);

void qwen3_tts_pcm_free(float* pcm);

void qwen3_tts_free(struct qwen3_tts_context* ctx);

void qwen3_tts_set_n_threads(struct qwen3_tts_context* ctx, int n_threads);

// Compute the 128-mel log-mel spectrogram used by the speaker encoder
// from 24 kHz mono audio. Returns malloc'd (T_mel × 128) row-major float32.
// *out_T_mel is set to the number of mel frames. Caller frees with free().
float* qwen3_tts_compute_speaker_mel(struct qwen3_tts_context* ctx, const float* audio, int n_samples, int* out_T_mel,
                                     int* out_n_mels);

// Run the ECAPA speaker encoder on a pre-computed mel spectrogram.
// mel is (T_mel × n_mels=128) row-major float32. Returns malloc'd float[1024].
float* qwen3_tts_run_speaker_enc_on_mel(struct qwen3_tts_context* ctx, const float* mel, int T_mel, int* out_dim);

// Compute a 1024-d speaker embedding from 24 kHz mono float32 audio
// via the ECAPA-TDNN speaker encoder. Returns a malloc'd float[1024]
// array that the caller frees with free(). Returns nullptr on failure.
// Does NOT set the context's active voice — call qwen3_tts_set_voice_prompt
// to both compute and activate the embedding for synthesis.
float* qwen3_tts_compute_speaker_embedding(struct qwen3_tts_context* ctx, const float* audio, int n_samples,
                                           int* out_dim);

#ifdef __cplusplus
}
#endif
