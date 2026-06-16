// crisp_punc.h — shared punctuation restoration library.
//
// Two backends:
//   FireRedPunc — BERT Chinese+multilingual, 5 classes
//   PCS — XLM-R, punctuation + capitalization + segmentation
//
// Used by both CrispASR and CrispEmbed for post-OCR/ASR text cleanup.

#ifndef CRISP_PUNC_H
#define CRISP_PUNC_H

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// FireRedPunc
// ---------------------------------------------------------------------------

struct fireredpunc_context;

// Load a FireRedPunc GGUF model.
struct fireredpunc_context* fireredpunc_init(const char* model_path);

// Add punctuation to unpunctuated text. Returns newly allocated string (caller frees).
char* fireredpunc_process(struct fireredpunc_context* ctx, const char* text);

// Free context.
void fireredpunc_free(struct fireredpunc_context* ctx);

// ---------------------------------------------------------------------------
// PCS (Punctuation + Capitalization + Segmentation)
// ---------------------------------------------------------------------------

struct pcs_context;

// Load a PCS GGUF model.
struct pcs_context* pcs_init(const char* model_path);

// Apply punctuation, truecasing, and sentence boundary detection.
// Returns newly allocated string (caller frees).
char* pcs_process(struct pcs_context* ctx, const char* text);

// Free context.
void pcs_free(struct pcs_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // CRISP_PUNC_H
