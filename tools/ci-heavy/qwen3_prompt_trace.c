/* Test-only ELF interposer. Only synthetic fixtures are run with this loaded. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t *(*tokenize_fn)(void *, const char *, int *);

int32_t *qwen3_asr_tokenize(void *ctx, const char *text, int *count) {
    const char *lib = getenv("CRISPASR_TEST_QWEN_LIB");
    void *handle = lib ? dlopen(lib, RTLD_NOW | RTLD_NOLOAD) : NULL;
    tokenize_fn next = (tokenize_fn)dlsym(handle ? handle : RTLD_NEXT, "qwen3_asr_tokenize");
    if (!next) {
        fputs("test interposer: tokenizer lookup failed\n", stderr);
        abort();
    }
    const char *path = getenv("CRISPASR_TEST_PROMPT_TRACE");
    if (path && text && strncmp(text, "<|im_start|>system\n", 19) == 0) {
        FILE *f = fopen(path, "ab");
        if (!f || fprintf(f, "\n--- PROMPT ---\n%s\n--- END ---\n", text) < 0 || fclose(f) != 0) {
            fputs("test interposer: cannot write trace\n", stderr);
            abort();
        }
    }
    return next(ctx, text, count);
}
