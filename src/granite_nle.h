// granite_nle.h — public C API for ibm-granite/granite-speech-4.1-2b-nar
//
// NLENARDecoder: a non-autoregressive speech-LLM. Same 16-layer Conformer
// encoder + Granite 4.0-1B LLM as granite-speech base, but the decoding
// pipeline is fundamentally different:
//
//   1. Encoder forward (with self-conditioning at layer 8). Outputs:
//        - logits (348-dim char-level CTC head)
//        - logits_bpe (100353-dim aux BPE CTC head, on posterior-pooled hidden)
//        - hidden states at layers [4, 8, 12, -1] (concatenated for projector)
//
//   2. Greedy CTC decode of the BPE logits → coarse text predictions
//      via the LLM tokenizer (with consecutive-collapse + blank removal).
//
//   3. Window-based projector (block_size=15, downsample_rate=5, 2-layer
//      Q-Former with 32-head SDPA cross-attention) → audio embeddings at
//      LLM dim (2048). 3 audio token slots per 15 frames of encoder output.
//
//   4. Build flat LLM input:
//        [audio_embs, text_embs_with_eos_slots]
//      where text_embs_with_eos_slots = [eos, t1, eos, t2, eos, ...] —
//      the "insertion slots" that absorb the editing logits.
//
//   5. ONE LLM forward pass (non-causal, all attention layers patched
//      `is_causal=False`). The slot positions absorb the LLM's edit votes.
//
//   6. Argmax + unique_consecutive on the slot logits, drop EOS, decode
//      with the LLM tokenizer → final transcript.
//
// Why "NAR": the LLM runs ONCE over the full sequence instead of token-by-
// token. Throughput is 5–20× higher than the autoregressive variants.
// Quality on JFK is comparable; trickier on long-form / noisy audio.
//
// Models loaded from GGUF files produced by:
//   `python models/convert-granite-nle-to-gguf.py --input <hf_dir> --output X.gguf`

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct granite_nle_context;

struct granite_nle_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;  // false => force CPU backend
};

struct granite_nle_context_params granite_nle_context_default_params(void);

struct granite_nle_context* granite_nle_init_from_file(const char* path_model,
                                                       struct granite_nle_context_params params);

void granite_nle_free(struct granite_nle_context* ctx);

// High-level: transcribe raw 16 kHz mono PCM. Returns malloc'd UTF-8.
// Caller frees with free(). Returns NULL on failure.
char* granite_nle_transcribe(struct granite_nle_context* ctx, const float* samples, int n_samples);

// Pipeline building blocks for differential testing — each returns a
// malloc'd buffer. Caller frees with free().

// 80-bin log-mel spectrogram, (80, T) F32 row-major.
float* granite_nle_compute_mel(struct granite_nle_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                               int* out_T_mel);

// Run encoder. Returns the CONCATENATED hidden states at the configured
// encoder_layer_indices (default [4, 8, 12, -1]) → (T, n_layers * d_model)
// where d_model = 1024 and n_layers = 4 → 4096-wide. Aux outputs (CTC
// logits and BPE logits) are also produced and may be retrieved via the
// `_last_*` accessors below if non-null.
float* granite_nle_run_encoder(struct granite_nle_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T,
                               int* out_dim);

// Last encoder CTC head outputs. Pointers are valid until the next
// granite_nle_run_encoder call. May be NULL for a freshly-loaded ctx.
const float* granite_nle_last_ctc_logits(struct granite_nle_context* ctx, int* out_T, int* out_vocab);
const float* granite_nle_last_bpe_logits(struct granite_nle_context* ctx, int* out_T, int* out_vocab);

// Run window-based Q-Former projector. Input: (T, num_encoder_layers * d_model).
// Output: (T_out, llm_dim) where T_out = ceil(T / block_size) * (block_size / downsample_rate).
float* granite_nle_run_projector(struct granite_nle_context* ctx, const float* enc_concat, int T, int dim, int* out_T,
                                 int* out_dim);

// Run the LLM in a SINGLE non-autoregressive forward pass. Caller passes:
//   audio_embs: (n_audio, llm_dim)  -- output of granite_nle_run_projector
//   text_ids:   (n_text,)           -- LLM-tokenized CTC text + EOS slots
// Returns the editing logits at the slot positions only:
//   (n_slot, vocab_size) F32.  n_slot = (n_text + 1) / 2 if alternating
//   slot/token layout from add_insertion_slots().
float* granite_nle_run_llm_editing(struct granite_nle_context* ctx, const float* audio_embs, int n_audio,
                                   const int32_t* text_ids, int n_text, int* out_n, int* out_vocab);

// Tokenize an arbitrary UTF-8 string with the LLM tokenizer (Granite 4.0
// BPE, vocab 100352). Returns malloc'd int32 array; caller frees.
int32_t* granite_nle_tokenize(struct granite_nle_context* ctx, const char* text, int* out_n);

// Decode an LLM token id sequence to UTF-8. Returns malloc'd; caller frees.
char* granite_nle_detokenize(struct granite_nle_context* ctx, const int32_t* ids, int n);

// Decode CTC char-level ids (348-vocab) to UTF-8 (greedy, no collapse).
// Useful for diffing against the upstream `text_ctc_preds` output.
char* granite_nle_ctc_decode(struct granite_nle_context* ctx, const int32_t* ids, int n);

// EOS / pad token id used to fill insertion slots between text tokens
// before the LLM's editing forward pass.
int granite_nle_eos_token_id(struct granite_nle_context* ctx);

#ifdef __cplusplus
}
#endif
