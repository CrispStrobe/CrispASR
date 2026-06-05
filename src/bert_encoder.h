// bert_encoder.h — Minimal BERT encoder for MeloTTS conditioning.
//
// Loads a GGUF containing bert-base-uncased (layers 0-9) and runs
// forward inference to produce per-token hidden states from layer -3.
// Used by MeloTTS to produce contextual phoneme embeddings.
//
// The BERT forward pass:
//   1. Tokenize text → token IDs (WordPiece via embedded vocab)
//   2. Token + position + type embeddings → LayerNorm
//   3. 10 transformer layers (self-attention + FFN)
//   4. Extract layer 9 output (= hidden_states[-3] in Python)
//
// Output: (hidden_size=768, n_tokens) float tensor

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct bert_encoder_context;

struct bert_encoder_context* bert_encoder_init(const char* gguf_path, int n_threads);

void bert_encoder_free(struct bert_encoder_context* ctx);

// Tokenize text and run BERT forward pass.
// Returns the hidden states from layer -3 (layer 9 of 12).
// out_features: (hidden_size=768, n_tokens) row-major, allocated by callee.
// out_n_tokens: number of BERT tokens (including [CLS] and [SEP]).
// Caller must free out_features with free().
bool bert_encoder_forward(struct bert_encoder_context* ctx, const char* text, float** out_features, int* out_n_tokens);

// Get the hidden size (768 for bert-base-uncased).
int bert_encoder_hidden_size(const struct bert_encoder_context* ctx);

// Tokenize text into BERT token IDs (including [CLS] and [SEP]).
// Returns number of tokens. out_ids must be freed by caller.
int bert_encoder_tokenize(const struct bert_encoder_context* ctx, const char* text, int** out_ids);

// Get the number of whitespace words that a BERT token sequence spans.
// For each whitespace word, returns the number of BERT subword tokens.
// E.g. "seashells" → ["seas", "##hell", "##s"] → n_subtokens=3.
// out_subtokens: array of subtokens-per-word. Caller frees.
// Returns number of words.
int bert_encoder_word_subtokens(const struct bert_encoder_context* ctx, const char* text, int** out_subtokens);

#ifdef __cplusplus
}
#endif
