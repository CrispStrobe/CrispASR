// xasr.h — X-ASR (icefall streaming Zipformer2 transducer) runtime, #436.
//
// Chunked streaming encoder + stateless decoder + joiner, decoded exactly as
// sherpa-onnx's OnlineRecognizer greedy search: fixed windows of
// T = 2*chunk + 13 feature frames shifted by 2*chunk, every Zipformer cache
// carried from chunk to chunk, at most one symbol per encoder frame.
// docs/xasr/PLAN.md lists the blueprint facts this runtime encodes.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct xasr_context;

struct xasr_context_params {
    int n_threads;
    int verbosity; // 0 silent, 1 normal, 2 verbose
    bool use_gpu;
    int chunk_ms;    // 160 / 480 / 960 / 1920 (the upstream exports); 0 = 480
    int tail_pad_ms; // silence appended before the end of input; < 0 = T*10 + 1000
};

struct xasr_context_params xasr_context_default_params(void);
struct xasr_context* xasr_init_from_file(const char* path_model, struct xasr_context_params params);
void xasr_free(struct xasr_context* ctx);

// Transcript of 16 kHz mono PCM ('▁' rendered as spaces; malloc'd). One
// flushed xasr_stream_accept, so offline and streaming share one code path.
char* xasr_transcribe(struct xasr_context* ctx, const float* samples, int n_samples);

// Streaming: feed 16 kHz PCM in pieces of any size; every call returns the
// full transcript so far (malloc'd, append-only). flush = end of input: the
// tail padding is added and the remaining frames are decoded. Pieces of any
// size give the same tokens as one call.
struct xasr_stream;
struct xasr_stream* xasr_stream_init(struct xasr_context* ctx);
char* xasr_stream_accept(struct xasr_stream* s, const float* samples, int n_samples, bool flush);
void xasr_stream_reset(struct xasr_stream* s);
void xasr_stream_free(struct xasr_stream* s);

// ---- diff-harness hooks (crispasr-diff xasr) -----------------------------
// Kaldi fbank of samples + the configured tail padding: (T, 80) row-major, malloc'd.
float* xasr_compute_fbank(struct xasr_context* ctx, const float* samples, int n_samples, int* out_T);
// The chunk loop over a (T, 80) fbank. Returns encoder_out (n_enc, joiner_dim),
// malloc'd. Optional dumps (caller-allocated, concatenated over chunks, rows
// time-major): dumps[0] embed_out (n_chunks*chunk, dims[0]), dumps[1+s] stack s
// output (n_chunks*chunk, dims[s]), dumps[1+S] enc_full (n_enc, max dim).
float* xasr_run_encoder(struct xasr_context* ctx, const float* fbank, int T, int* out_n_enc, int* out_dim,
                        float** dumps, int n_dumps);
// Greedy transducer search over encoder_out; returns the token ids (malloc'd).
int32_t* xasr_greedy(struct xasr_context* ctx, const float* enc_out, int n_enc, int* out_n_tokens, float* first_logits);
char* xasr_tokens_to_text(struct xasr_context* ctx, const int32_t* tokens, int n);
int xasr_n_stacks(struct xasr_context* ctx);
int xasr_stack_dim(struct xasr_context* ctx, int s);
int xasr_chunk_frames(struct xasr_context* ctx); // encoder_embed frames per chunk (50 Hz)
int xasr_vocab(struct xasr_context* ctx);

#ifdef __cplusplus
}
#endif
