#ifndef FIREREDPUNC_H
#define FIREREDPUNC_H

#include <string>

#ifdef __cplusplus
extern "C" {
#endif

struct fireredpunc_context;

// Load a FireRedPunc GGUF model.
struct fireredpunc_context* fireredpunc_init(const char* model_path);

// Add punctuation to unpunctuated text. Returns newly allocated string (caller frees).
char* fireredpunc_process(struct fireredpunc_context* ctx, const char* text);

// Free context.
void fireredpunc_free(struct fireredpunc_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // FIREREDPUNC_H
