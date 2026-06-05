#ifndef TRUECASER_CRF_H
#define TRUECASER_CRF_H

#ifdef __cplusplus
extern "C" {
#endif

struct truecaser_crf_context;

// Load a CRF truecaser binary model.
struct truecaser_crf_context* truecaser_crf_init(const char* model_path);

// Apply truecasing to text. Returns newly allocated string (caller frees).
char* truecaser_crf_process(struct truecaser_crf_context* ctx, const char* text);

// Free context.
void truecaser_crf_free(struct truecaser_crf_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // TRUECASER_CRF_H
