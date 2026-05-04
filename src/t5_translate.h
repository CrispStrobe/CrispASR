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

// Translate text. For MADLAD-400, prefix text with "<2xx> " where xx is the
// target language code (e.g. "<2de> Hello world" → German translation).
// Returns a newly allocated UTF-8 string (caller must free()).
char* t5_translate(struct t5_translate_context* ctx, const char* text, int max_new_tokens);

#ifdef __cplusplus
}
#endif
