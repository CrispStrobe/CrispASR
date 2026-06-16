// crisp_truecase.h — shared truecasing library.
//
// Three backends:
//   truecaser      — statistical word-frequency based (9 MB)
//   truecaser_crf  — CRF with context features (24 MB)
//   truecaser_lstm — BiLSTM character-level (3.2 MB, recommended)
//
// Used by CrispASR (post-ASR German truecasing) and CrispEmbed (post-OCR).

#ifndef CRISP_TRUECASE_H
#define CRISP_TRUECASE_H

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Statistical truecaser (word-frequency)
// ---------------------------------------------------------------------------
struct truecaser_context;
struct truecaser_context* truecaser_init(const char* model_path);
char* truecaser_process(struct truecaser_context* ctx, const char* text);
void truecaser_free(struct truecaser_context* ctx);

// ---------------------------------------------------------------------------
// CRF truecaser (context features)
// ---------------------------------------------------------------------------
struct truecaser_crf_context;
struct truecaser_crf_context* truecaser_crf_init(const char* model_path);
char* truecaser_crf_process(struct truecaser_crf_context* ctx, const char* text);
void truecaser_crf_free(struct truecaser_crf_context* ctx);

// ---------------------------------------------------------------------------
// BiLSTM truecaser (character-level, highest quality)
// ---------------------------------------------------------------------------
struct truecaser_lstm_context;
struct truecaser_lstm_context* truecaser_lstm_init(const char* model_path);
char* truecaser_lstm_process(struct truecaser_lstm_context* ctx, const char* text);
void truecaser_lstm_free(struct truecaser_lstm_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // CRISP_TRUECASE_H
