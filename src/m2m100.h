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

// Load model from GGUF file produced by convert-m2m100-to-gguf.py
struct m2m100_context* m2m100_init_from_file(const char* path_model, struct m2m100_context_params params);

void m2m100_free(struct m2m100_context* ctx);

// Translate text from src_lang to tgt_lang.
// src_lang/tgt_lang: ISO-639-1 codes ("en", "de", "fr", ...)
// Returns a newly allocated UTF-8 string (caller must free()).
// Returns NULL on failure.
char* m2m100_translate(struct m2m100_context* ctx, const char* text, const char* src_lang, const char* tgt_lang,
                       int max_new_tokens);

// Get the list of supported language codes.
int m2m100_n_languages(struct m2m100_context* ctx);
const char* m2m100_language(struct m2m100_context* ctx, int index);

#ifdef __cplusplus
}
#endif
