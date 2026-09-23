// dolphin.h — Dolphin (DataoceanAI) ASR runtime, #436.
//
// E-Branchformer encoder + Transformer decoder + CTC, WeNet layout. Decoding
// follows the upstream recipe (dolphin/transcribe.py): CTC prefix beam search
// (beam 10) rescored by the attention decoder with the prompt
// [sos, <lang>, <region>, <asr>, <notimestamp>]; language and region are
// predicted by the decoder unless the caller forces them.
//
// docs/dolphin/PLAN.md lists the blueprint facts this runtime encodes.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct dolphin_context;

struct dolphin_context_params {
    int n_threads;
    int verbosity; // 0 silent, 1 normal, 2 verbose
    bool use_gpu;
    int beam_size; // CTC prefix beam + rescoring width (upstream default 10)
};

struct dolphin_context_params dolphin_context_default_params(void);
struct dolphin_context* dolphin_init_from_file(const char* path_model, struct dolphin_context_params params);
void dolphin_free(struct dolphin_context* ctx);

struct dolphin_result {
    char* text;        // transcript without special tokens, '▁' rendered as spaces (malloc'd)
    char* raw_text;    // upstream's `text`: every token joined, specials included (malloc'd)
    char language[32]; // e.g. "zh"
    char region[32];   // e.g. "CN"
    int32_t* tokens;   // attention-rescoring output, prompt tokens included
    int n_tokens;
};

// lang / region: NULL (or "") lets the decoder predict them, as upstream does.
struct dolphin_result* dolphin_transcribe_ex(struct dolphin_context* ctx, const float* samples, int n_samples,
                                             const char* lang, const char* region);
void dolphin_result_free(struct dolphin_result* r);
char* dolphin_transcribe(struct dolphin_context* ctx, const float* samples, int n_samples);

// ---- diff-harness hooks (crispasr-diff dolphin) --------------------------
// Kaldi fbank of waveform*32768, BEFORE CMVN: (T, n_mels) row-major, malloc'd.
float* dolphin_compute_fbank(struct dolphin_context* ctx, const float* samples, int n_samples, int* out_T);
// Encoder on a (T, n_mels) pre-CMVN fbank. Returns (T', d) malloc'd.
// dumps (optional, n_dumps = 1 + n_layers slots): [0] subsample output,
// [1+i] block i output, each (T', d) and caller-allocated.
float* dolphin_run_encoder(struct dolphin_context* ctx, const float* fbank, int T, int* out_T_enc, int* out_d,
                           float** dumps, int n_dumps);
// CTC log-probs (T', V) for an encoder output. malloc'd.
float* dolphin_ctc_logprobs(struct dolphin_context* ctx, const float* enc, int T_enc, int* out_V);
int dolphin_n_layers(struct dolphin_context* ctx);
int dolphin_n_mels(struct dolphin_context* ctx);
const char* dolphin_token_text(struct dolphin_context* ctx, int id);

#ifdef __cplusplus
}
#endif
