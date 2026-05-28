#pragma once

// CosyVoice3-0.5B-2512 TTS — public C ABI.
//
// FunAudioLLM/Fun-CosyVoice3-0.5B-2512: multilingual TTS (9 langs +
// 18 Chinese dialects), Apache-2.0, 24 kHz output, zero-shot voice
// cloning. Pipeline:
//
//   text  → CosyVoice3LM (Qwen2-0.5B + speech_embd + speech_lm_head)
//         → speech tokens ∈ [0, 6561)
//        → Flow (DiT + CausalConditionalCFM)
//         → mel @ T_mel = 2 · T_tok
//        → CausalHiFTGenerator (HiFi-GAN + iSTFT)
//         → 24 kHz PCM
//
// Phase 2 (this header at first cut): the LLM side only. The flow and
// hift sub-models load through separate calls landed in later phases.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct cosyvoice3_tts_context;

struct cosyvoice3_tts_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    bool flash_attn;
    float temperature; // 0 = greedy
    uint64_t seed;     // RNG seed; 0 = use default 42
    int max_tokens;    // upper bound on AR decode steps; 0 = use built-in default (1500)
    int ras_top_k;     // RAS sampler top-k (default 25; 0 → use default)
    float ras_top_p;   // RAS sampler top-p (default 0.8f; 0 → use default)
    int ras_win_size;  // RAS repetition window (default 10; 0 → use default)
    float ras_tau_r;   // RAS repetition threshold (default 0.1f)
};

struct cosyvoice3_tts_context_params cosyvoice3_tts_context_default_params(void);

// Initialise from the LLM GGUF file
// (e.g. cosyvoice3-llm-f16.gguf from `cstr/cosyvoice3-0.5b-2512-GGUF`).
// Returns nullptr on failure.
struct cosyvoice3_tts_context* cosyvoice3_tts_init_from_file(const char* path_model,
                                                             struct cosyvoice3_tts_context_params params);

void cosyvoice3_tts_free(struct cosyvoice3_tts_context* ctx);

void cosyvoice3_tts_set_n_threads(struct cosyvoice3_tts_context* ctx, int n_threads);
void cosyvoice3_tts_set_seed(struct cosyvoice3_tts_context* ctx, uint64_t seed);
void cosyvoice3_tts_set_temperature(struct cosyvoice3_tts_context* ctx, float temperature);

// Read out LLM hparams for the diff harness (each pointer may be NULL).
// d_model, n_layers, n_heads, n_kv_heads, head_dim, text_vocab,
// speech_vocab (the head dimension, 6761) and speech_codebook (the AR
// emit range, 6561).
int cosyvoice3_tts_get_hparams(struct cosyvoice3_tts_context* ctx, uint32_t* d_model, uint32_t* n_layers,
                               uint32_t* n_heads, uint32_t* n_kv_heads, uint32_t* head_dim, uint32_t* text_vocab,
                               uint32_t* speech_vocab, uint32_t* speech_codebook);

// Build a text-mode input embedding (text_token_id -> token_embd[id])
// for `n_tokens` ids. Returns a malloc'd float buffer [n_tokens, d_model]
// in row-major order. Caller frees with free(). Useful for the diff
// harness and for the higher-level synth path.
float* cosyvoice3_tts_embed_text(struct cosyvoice3_tts_context* ctx, const int32_t* ids, int n_tokens);

// Build a speech-mode input embedding (speech_token_id ->
// speech_embd[id]) for `n_tokens` ids. Same return contract.
float* cosyvoice3_tts_embed_speech(struct cosyvoice3_tts_context* ctx, const int32_t* ids, int n_tokens);

// Run the 24-layer Qwen2 LM on caller-supplied [n_tokens, d_model]
// row-major float32 embeddings. Writes K/V into the persistent KV
// cache at positions [n_past, n_past + n_tokens). Returns the
// last-position logits over the speech codebook head (6761 entries).
// Caller frees with free(). Returns nullptr on failure.
//
// This is the diff-harness entry: feed in PyTorch-prebuilt embeds and
// expect bit-equivalent logits at the tail.
float* cosyvoice3_tts_prefill_with_embeds(struct cosyvoice3_tts_context* ctx, const float* embeds, int n_tokens,
                                          int n_past);

// Single-step speech-token forward: speech_embd[id] -> Qwen2 ->
// speech_lm_head -> logits[6761]. Reads/writes the persistent KV cache
// at slot n_past. Caller frees with free().
float* cosyvoice3_tts_step_speech(struct cosyvoice3_tts_context* ctx, int32_t speech_id, int n_past);

// Reset the persistent KV cache so the next prefill starts from n_past=0.
void cosyvoice3_tts_reset_kv(struct cosyvoice3_tts_context* ctx);

// Repetition-Aware Sampling — port of upstream
// `CosyVoice/cosyvoice/utils/common.py::ras_sampling`. Samples ONE
// speech token from `logits[speech_vocab]` using nucleus sampling
// (top_p, top_k from ctx->params); if the sampled token already
// appears in the last `win_size` entries of `decoded_history` at
// least `win_size * tau_r` times, suppresses it and falls back to
// random sampling over the full distribution.
//
// `decoded_history` may be NULL (e.g. for the first AR step); in
// that case the repetition check is skipped and the function reduces
// to nucleus sampling.
//
// Modifies the RNG state on the context (advances ctx->seed via
// std::mt19937).  Returns -1 on failure (e.g. logits all -INF).
int32_t cosyvoice3_tts_sample_ras(struct cosyvoice3_tts_context* ctx, const float* logits,
                                  const int32_t* decoded_history, int n_history);

// End-to-end speech-token AR loop: caller supplies an [n_tokens, d_model]
// embedding tensor (built externally — typically text token_embd lookups
// plus optionally a speech-token prompt), the runtime prefills, then
// AR-samples up to `max_tokens` speech tokens via RAS (when
// ctx->params.temperature > 0) or greedy argmax (temperature == 0)
// until `stop_token_id` is sampled.
//
// Returns malloc'd int32_t[*out_n] of speech token ids. Caller frees
// with free(). Returns nullptr on failure.
//
// `stop_token_id < 0` disables the stop check (runs to max_tokens).
// `max_tokens <= 0` uses the context's default (params.max_tokens or
// the built-in 1500).
int32_t* cosyvoice3_tts_generate_tokens_from_embeds(struct cosyvoice3_tts_context* ctx, const float* embeds,
                                                    int n_tokens, int max_tokens, int stop_token_id, int* out_n);

// ---------------------------------------------------------------------------
// Phase 3 — Flow (DiT-CFM) sub-model API
// ---------------------------------------------------------------------------
//
// The flow sub-model converts speech tokens (T_tok,) + a 192-dim
// speaker embedding into a (T_mel = 2·T_tok, 80) log-mel via:
//   input_embd(speech_tokens) → pre_lookahead causal conv (k=4 then k=3)
//   → concat [pre_la, spk_affine(spk_emb), 0-cond] → (T_tok, 320)
//   → CausalConditionalCFM (10-step Euler ODE, cosine t-schedule):
//       22-block DiT estimator @ dim=1024, heads=16, head_dim=64,
//       ff_mult=2, AdaLN-Zero modulation projected from
//       sinusoidal time-embed via 2-layer MLP. RoPE inside MHA.
//   → mel
//
// Load the flow GGUF (cosyvoice3-flow-f16.gguf, ~670 MB) into an
// already-initialised context AFTER the LLM init. Returns 0 on
// success, -1 on failure (missing tensors, …).

int cosyvoice3_tts_init_flow_from_file(struct cosyvoice3_tts_context* ctx, const char* path);

// Read flow-side hparams. Each pointer may be NULL.
int cosyvoice3_tts_get_flow_hparams(struct cosyvoice3_tts_context* ctx, uint32_t* n_dit_layers, uint32_t* dit_dim,
                                    uint32_t* dit_heads, uint32_t* dit_head_dim, uint32_t* dit_ff_dim,
                                    uint32_t* dit_input_dim, uint32_t* mel_dim, uint32_t* spk_dim_in,
                                    uint32_t* spk_dim_out, uint32_t* cfm_n_steps, float* cfm_cfg_rate);

// Run ONE DiT block (block_idx in [0, n_dit_layers)) on caller-supplied
// input. Inputs (row-major):
//   x      [T, dit_dim]  F32
//   t_emb  [dit_dim]     F32  — already passed through time_mlp
// Returns malloc'd float[T * dit_dim] (caller frees with free()). The
// per-block AdaLN-Zero linear (silu(t_emb) → 6×dit_dim chunked into
// shift/scale/gate × {attn, ffn}) is applied inside; LayerNorm is
// affine-free, bidirectional MHA uses RoPE θ=10000, FFN is plain
// Linear→GELU(tanh)→Linear with no GLU gating. Diff stage for
// verifying one block against PyTorch before wiring up the 22-block
// stack.
float* cosyvoice3_tts_run_flow_dit_block(struct cosyvoice3_tts_context* ctx, int block_idx, const float* x, int T,
                                         const float* t_emb);

// Solve the cosine-schedule CFM Euler ODE (`n_steps` steps, classifier-free
// guidance at `cfg_rate`) that converts speech-token-derived `mu` + speaker
// embedding into a mel spectrogram. Inputs are caller-supplied and post-
// projection — see the corresponding extract_stage flow_euler doc-comment.
//   mu          [T_mel, mel_dim]   F32 — pre_la + repeat_interleave output
//   spks_proj   [spk_dim_out]      F32 — already through normalize + spk_affine
//   cond        [T_mel, mel_dim]   F32 — prompt-prefix conditioning
//                                          (zeros for zero-shot)
//   x_init      [T_mel, mel_dim]   F32 — initial Euler noise (for bit-equivalence
//                                          to upstream, pass
//                                          `torch.manual_seed(0);
//                                           randn([1, 80, 50*300])[:, :, :T_mel]`)
//   n_steps                                — defaults to 10 upstream
//   cfg_rate                                — defaults to 0.7 upstream
// Returns malloc'd float[T_mel * mel_dim] (caller frees) with the final mel.
float* cosyvoice3_tts_solve_flow_euler(struct cosyvoice3_tts_context* ctx, const float* mu, int T_mel,
                                       const float* spks_proj, const float* cond, const float* x_init, int n_steps,
                                       float cfg_rate);

// Diff-harness stage extractor. Returns malloc'd float[*out_n].
// Phase 2 supports:
//   "lm_step0_logits"   — single-step logits after prefilling on
//                          caller-supplied embeds; pass embeds via the
//                          `embeds_in` buffer of length n_tokens*d_model
//   "lm_token_embd"     — token_embd[ids] lookup verification
//   "lm_speech_embd"    — speech_embd[ids] lookup verification
// Phase 3 (partial):
//   "flow_inventory"    — returns sentinel buffer; verifies flow GGUF
//                          is loaded and binds.
//   "flow_dit_blk_<N>_out"
//   "flow_dit_blk_<N>_{lnx_a,h_a,attn,xattn,ff}"
//                       — single-block forward (block N) or per-stage
//                          intermediate. `embeds_in` packs [x | t_emb]:
//                          first n_embed_tokens*dit_dim floats are x
//                          [T, dit_dim], next dit_dim floats are t_emb
//                          (post time_mlp). Returns malloc'd
//                          float[T*dit_dim].
float* cosyvoice3_tts_extract_stage(struct cosyvoice3_tts_context* ctx, const char* stage_name, const int32_t* ids,
                                    int n_ids, const float* embeds_in, int n_embed_tokens, int* out_n);

#ifdef __cplusplus
}
#endif
