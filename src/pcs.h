#ifndef PCS_H
#define PCS_H

#ifdef __cplusplus
extern "C" {
#endif

struct pcs_context;

// Load a PCS (Punctuation + Capitalization + Segmentation) GGUF model.
struct pcs_context* pcs_init(const char* model_path);

// Apply punctuation, truecasing, and sentence boundary detection.
// Returns newly allocated string (caller frees).
char* pcs_process(struct pcs_context* ctx, const char* text);

// Free context.
void pcs_free(struct pcs_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // PCS_H
