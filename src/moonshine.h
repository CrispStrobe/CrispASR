#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct moonshine_context;

struct moonshine_timing {
    double encode_ms; // encoder + cross-KV precompute
    double decode_ms; // decode loop
    int n_tokens;     // tokens decoded
    int n_samples;    // audio samples
};

struct moonshine_init_params {
    const char* model_path;
    const char* tokenizer_path; // NULL = auto-detect from model directory
    int n_threads;              // 0 = default (4)
};

struct moonshine_context* moonshine_init(const char* model_path);
struct moonshine_context* moonshine_init_with_params(struct moonshine_init_params params);
const char* moonshine_transcribe(struct moonshine_context* ctx, const float* audio, int n_samples);
// Run encoder conv stem. Caller must free(*out_features) when done.
int moonshine_encode(struct moonshine_context* ctx, const float* audio, int n_samples, float** out_features,
                     int* out_seq_len, int* out_hidden_dim);
void moonshine_free(struct moonshine_context* ctx);

// Sticky sampling temperature. 0 = greedy argmax (default). > 0 enables
// multinomial sampling from softmax(logits/temperature).
void moonshine_set_temperature(struct moonshine_context* ctx, float temperature);

// Sticky per-call seed for the multinomial sampler. 0 (default) = derive
// deterministically from the input audio (repeated calls give identical
// samples). Non-zero values let best-of-N callers draw independent samples
// from the same audio by injecting a run-index salt.
void moonshine_set_seed(struct moonshine_context* ctx, uint64_t seed);

// Sticky beam size for the decoder. 1 = greedy/sampled (default). >1 = beam
// search via per-beam KV snapshot/restore (O(B × T) single-token forwards).
// Beam search is mutually exclusive with temperature sampling — the beam
// path always picks deterministically by cumulative log-prob.
void moonshine_set_beam_size(struct moonshine_context* ctx, int beam_size);

// Single-token piece lookup. The returned pointer is owned by the context
// and stable until the next call to this function. Returns empty string
// for special tokens / out-of-range ids.
const char* moonshine_token_text(struct moonshine_context* ctx, int token_id);

// Result of `moonshine_transcribe_with_probs`: full decoded text + parallel
// arrays of token ids and per-token softmax probabilities. `n_tokens`
// excludes BOS / EOS. All pointers are malloc'd; free with
// `moonshine_result_free`.
struct moonshine_result {
    char* text;
    int* token_ids;
    float* token_probs;
    int n_tokens;
};

struct moonshine_result* moonshine_transcribe_with_probs(struct moonshine_context* ctx, const float* audio,
                                                         int n_samples);

void moonshine_result_free(struct moonshine_result* r);
void moonshine_print_model_info(struct moonshine_context* ctx);

void moonshine_set_n_threads(struct moonshine_context* ctx, int n_threads);
int moonshine_get_n_threads(struct moonshine_context* ctx);
int moonshine_get_timing(struct moonshine_context* ctx, struct moonshine_timing* timing);

#ifdef __cplusplus
}
#endif
