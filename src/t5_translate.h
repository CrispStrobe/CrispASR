#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct t5_translate_context;

struct t5_translate_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
};

struct t5_translate_context_params t5_translate_context_default_params(void);

// Load model from GGUF file produced by convert-madlad-to-gguf.py
struct t5_translate_context* t5_translate_init_from_file(const char* path_model,
                                                         struct t5_translate_context_params params);

void t5_translate_free(struct t5_translate_context* ctx);

// Beam search width. 1 = greedy (default); >1 = replay-from-prefix beam.
void t5_translate_set_beam_size(struct t5_translate_context* ctx, int beam_size);

// Translate text. For MADLAD-400, prefix text with "<2xx> " where xx is the
// target language code (e.g. "<2de> Hello world" → German translation).
// Returns a newly allocated UTF-8 string (caller must free()).
char* t5_translate(struct t5_translate_context* ctx, const char* text, int max_new_tokens);

// Returns true if the tokenizer's vocab contains `token_str` as an
// exact-match piece. Used by the CLI adapter to decide whether to
// prepend the MADLAD "<2xx>" target-language tag — MADLAD-400 has all
// 419 lang tags as single-piece vocab entries; flan-t5 / mT5 / etc.
// don't, so prepending the tag on those just adds garbage [▁, <unk>]
// tokens to the input and corrupts the encoder context.
bool t5_has_token(struct t5_translate_context* ctx, const char* token_str);

#ifdef __cplusplus
}
#endif
