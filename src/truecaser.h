#ifndef TRUECASER_H
#define TRUECASER_H

#ifdef __cplusplus
extern "C" {
#endif

struct truecaser_context;

// Load a statistical truecaser binary model.
struct truecaser_context* truecaser_init(const char* model_path);

// Apply truecasing to text. Returns newly allocated string (caller frees).
char* truecaser_process(struct truecaser_context* ctx, const char* text);

// Free context.
void truecaser_free(struct truecaser_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // TRUECASER_H
