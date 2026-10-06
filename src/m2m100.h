#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct m2m100_context;

struct m2m100_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
};

struct m2m100_context_params m2m100_context_default_params(void);

// Generation default declared by every supported M2M100/WMT21 checkpoint.
// Exposed so adapters and tests use the runtime's single source of truth.
int m2m100_default_beam_size(void);

// Load model from GGUF file produced by convert-m2m100-to-gguf.py, or a
// MarianMT / Opus-MT one from convert-marian-to-gguf.py (general.architecture
// "marian"): the same runtime with post-norm layers, Marian's positions and
// its SentencePiece-plus-vocab.json tokenizer, all selected by the GGUF.
struct m2m100_context* m2m100_init_from_file(const char* path_model, struct m2m100_context_params params);

void m2m100_free(struct m2m100_context* ctx);

// Beam search width. 1 = greedy; >1 = replay-from-prefix beam. Default is 5,
// matching the supported checkpoints' generation_config (#439).
void m2m100_set_beam_size(struct m2m100_context* ctx, int beam_size);

// The beam width the loaded checkpoint declares: generation_config num_beams
// for a Marian GGUF (4 for Opus-MT), m2m100_default_beam_size() otherwise.
// A freshly loaded context already uses it.
int m2m100_model_beam_size(struct m2m100_context* ctx);

// 1 when the loaded GGUF is a Marian / Opus-MT model.
int m2m100_is_marian(struct m2m100_context* ctx);

// The encoder input ids for `text`, exactly as m2m100_translate builds them
// (language token / `>>xx<<` code, pieces, </s>). Writes at most `capacity`
// ids and returns the full count, or -1 on a NULL argument. For parity tests
// against the reference tokenizer.
int m2m100_tokenize(struct m2m100_context* ctx, const char* text, const char* src_lang, const char* tgt_lang,
                    int32_t* out_ids, int capacity);

// Translate text from src_lang to tgt_lang.
// src_lang/tgt_lang: ISO-639-1 codes ("en", "de", "fr", ...)
// A single-pair Marian model translates its own direction regardless and warns
// once if asked for another. max_new_tokens <= 0 selects the default (200;
// the checkpoint's max_length for Marian).
// Returns a newly allocated UTF-8 string (caller must free()).
// Returns NULL on failure.
char* m2m100_translate(struct m2m100_context* ctx, const char* text, const char* src_lang, const char* tgt_lang,
                       int max_new_tokens);

// Like m2m100_translate, with `target_prefix` (target-language text, whole
// words) as the fixed start of the translation: the decoder is fed it and
// continues after it, and the returned text starts with it. NULL or "" = no
// prefix. For live drafts, to keep words already shown from changing.
char* m2m100_translate_prefixed(struct m2m100_context* ctx, const char* text, const char* src_lang,
                                const char* tgt_lang, int max_new_tokens, const char* target_prefix);

// Get the list of supported language codes.
int m2m100_n_languages(struct m2m100_context* ctx);
const char* m2m100_language(struct m2m100_context* ctx, int index);

#ifdef __cplusplus
}
#endif
