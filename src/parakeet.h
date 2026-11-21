// parakeet.h — public C API for nvidia/parakeet-tdt-0.6b-v3 ggml runtime
//
// Multilingual ASR (25 European languages) using FastConformer encoder +
// Token-and-Duration Transducer (TDT) decoder. Word-level timestamps come
// for free from the duration head — no separate CTC alignment needed.
//
// Models are loaded from GGUF files produced by:
//   python models/convert-parakeet-to-gguf.py --nemo X.nemo --output X.gguf

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct parakeet_context;

// Model metadata declares English-only training (Phonon-2), even after renaming.
bool parakeet_is_english_only(const struct parakeet_context* ctx);

struct parakeet_context_params {
    int n_threads;
    bool use_flash; // flash attention in encoder (default: true)
    int verbosity;  // 0=silent 1=normal 2=verbose
    bool use_gpu;   // false => force CPU backend
};

struct parakeet_context_params parakeet_context_default_params(void);

// Load model from GGUF (produced by convert-parakeet-to-gguf.py)
struct parakeet_context* parakeet_init_from_file(const char* path_model, struct parakeet_context_params params);

void parakeet_free(struct parakeet_context* ctx);

// ---- Per-token data returned by parakeet_transcribe_ex() ----

struct parakeet_token_data {
    int id;        // SentencePiece token id (0 .. vocab_size-1)
    char text[48]; // decoded text (SentencePiece '▁' converted to ' ')
    int64_t t0;    // start time, centiseconds (absolute, includes t_offset_cs)
    int64_t t1;    // end time,   centiseconds (start + duration*frame_dur_cs)
    float p;       // softmax probability of the emitted token [0,1]
};

// Word-level data: sub-word tokens grouped at SentencePiece '▁' boundaries.
struct parakeet_word_data {
    char text[64]; // word text (no leading space)
    int64_t t0;    // start time, centiseconds (from first sub-word)
    int64_t t1;    // end time,   centiseconds (from last sub-word)
    float p;       // mean softmax probability across the word's sub-word tokens [0,1]
};

struct parakeet_result {
    char* text;                         // full transcript (malloc'd, caller owns)
    struct parakeet_token_data* tokens; // per-token timing (malloc'd)
    int n_tokens;
    struct parakeet_word_data* words; // grouped word timings (malloc'd)
    int n_words;
};

void parakeet_result_free(struct parakeet_result* r);

// Transcribe raw 16 kHz mono PCM, returning a malloc'd UTF-8 string.
char* parakeet_transcribe(struct parakeet_context* ctx, const float* samples, int n_samples);

// Like parakeet_transcribe but returns per-token TDT timestamps.
//
// t_offset_cs: absolute start of this audio slice in centiseconds.
//   Token t0/t1 = t_offset_cs + (encoder_frame * frame_dur_cs).
//   For long audio with VAD, pass (vad_segment_t0_seconds * 100).
//
// Unlike Cohere's cross-attention DTW path, these timestamps come directly
// from the TDT decoder's duration head and are accurate to one encoder
// frame (~80 ms for parakeet-tdt-0.6b-v3).
struct parakeet_result* parakeet_transcribe_ex(struct parakeet_context* ctx, const float* samples, int n_samples,
                                               int64_t t_offset_cs);

// Like parakeet_transcribe_ex but splits long audio into overlapping
// chunks (default 8 s with 2 s overlap), encodes each with per-chunk
// z-norm, concatenates the encoder output, and runs one TDT decode
// over the whole sequence.  Avoids both z-norm drift and decoder
// cold-start.  Issue #89 / PLAN #104.
//
// chunk_seconds <= 0 → default 8;  overlap_seconds < 0 → default 2.
struct parakeet_result* parakeet_transcribe_chunked(struct parakeet_context* ctx, const float* samples, int n_samples,
                                                    int64_t t_offset_cs, int chunk_seconds, int overlap_seconds);

// NeMo-style streamed pipeline: compute mel over the FULL audio with
// global z-norm (identical to single-pass), then encode in overlapping
// chunks and decode in one TDT pass.  Best quality on long audio — the
// z-norm matches single-pass exactly, and the encoder gets per-chunk
// context windows while the decoder sees a continuous sequence.
struct parakeet_result* parakeet_transcribe_streamed(struct parakeet_context* ctx, const float* samples, int n_samples,
                                                     int64_t t_offset_cs, int chunk_seconds, int overlap_seconds);

// Issue #385: the streamed pipeline with a per-encoder-window progress hook.
// `progress_cb` is invoked on the CALLING thread once per finished encoder
// window with (input samples encoded so far, total samples) — monotonically
// non-decreasing, and exactly (n_samples, n_samples) on the last window.
//
// NOTE the asymmetry with the LONGFORM route: here the single TDT decode runs
// AFTER every window, so the last tick means "encoding done", not "transcript
// done". A caller that wants 100 % to mean finished should suppress the
// terminal tick and emit it itself once the decode returns — see the
// CHUNK_SEGMENTED branch of parakeet_transcribe_segments().
//
// parakeet_transcribe_streamed() is exactly this with progress_cb == NULL.
struct parakeet_result* parakeet_transcribe_streamed_progress(
    struct parakeet_context* ctx, const float* samples, int n_samples, int64_t t_offset_cs, int chunk_seconds,
    int overlap_seconds, void (*progress_cb)(int processed, int total, void* user_data), void* progress_ud);

// Vocabulary helpers
int parakeet_n_vocab(struct parakeet_context* ctx);
// Issue #257: 1 iff the model's vocab is predominantly Japanese script (JA-only
// model). Detects JA by content, not vocab size (small English vocabs were
// misclassified). Drives the JA-specific short-chunk decode path.
int parakeet_vocab_is_japanese(struct parakeet_context* ctx);
int parakeet_blank_id(struct parakeet_context* ctx);
const char* parakeet_token_to_str(struct parakeet_context* ctx, int token_id);

// Sampling: when temperature > 0, the TDT decoder draws each non-blank
// token via stable-softmax(logits / temperature) instead of argmax.
// Temperature == 0 (the default) keeps the bit-identical pure-greedy
// path. Set per-call as needed; the setting is sticky on the context
// until the next call. seed == 0 means time-based RNG.
void parakeet_set_temperature(struct parakeet_context* ctx, float temperature, uint64_t seed);

// Beam search for TDT/RNNT decode. beam_size == 1 (default) is greedy.
// beam_size > 1 runs a label-looping beam search that keeps multiple
// hypotheses alive — especially beneficial with hotword phrase boost.
void parakeet_set_beam_size(struct parakeet_context* ctx, int beam_size);

// MAES (Modified Adaptive Expansion Search) beam decoding. Requires
// beam_size > 1. More efficient than label-looping beam search for
// transducers: processes one frame at a time with adaptive expansion.
// gamma = pruning threshold (lower = more pruning); num_steps = max
// non-blank expansions per frame; beta = extra candidates beyond beam.
void parakeet_set_maes(struct parakeet_context* ctx, bool enable, int num_steps, float gamma, int beta);

// CTC decode mode (hybrid TDT+CTC models only).
void parakeet_set_ctc_mode(struct parakeet_context* ctx, bool ctc);
// Issue #257: local (windowed) attention window in encoder frames — NeMo's
// change_attention_model("rel_pos_local_attn", [left, right]). Bounds long-audio
// encoder memory to O(T·window). left/right < 0 = full attention. Applies to the
// next transcribe.
void parakeet_set_att_context(struct parakeet_context* ctx, int left, int right);
bool parakeet_has_ctc(struct parakeet_context* ctx);
// True for a model whose prediction network is the Kimi Delta Attention stack
// (Haas-ja) rather than the NeMo LSTM. Used to keep parakeet-ja-specific
// repairs (#89 gap-fill) off models that do not share its encoder defect.
bool parakeet_predictor_is_kda(const struct parakeet_context* ctx);

// CTC-WS hotword phrase boost (PLAN #98). Builds an Aho-Corasick trie
// from the hotword strings; during CTC/TDT decode, tokens that continue
// an active hotword prefix get a log-prob boost. Call before transcribe.
void parakeet_set_hotwords(struct parakeet_context* ctx, const char** hotwords, int n_hotwords, float boost);

// Split encode / decode for full-audio-encode + chunked-decode.
// parakeet_encode: mel → encoder, returns malloc'd float[T_enc * d_model].
// parakeet_decode_frames: run TDT/CTC decode on pre-encoded frames.
// Caller must free() the returned encoder buffer.
float* parakeet_encode(struct parakeet_context* ctx, const float* samples, int n_samples, int* out_T_enc,
                       int* out_d_model);
struct parakeet_result* parakeet_decode_frames(struct parakeet_context* ctx, const float* enc_frames, int T_enc,
                                               int d_model, int64_t t_offset_cs);

// 1 when parakeet_decode_frames() runs its predictor/joint as ggml graphs on
// ctx->backend (CUDA/Vulkan default), 0 when it uses the cblas CPU path
// (CPU + Metal default). Callers that want to overlap encode and decode on
// separate threads MUST NOT do so when this returns 1 — both would drive the
// same ggml backend and scheduler concurrently.
int parakeet_decode_uses_backend(struct parakeet_context* ctx);

// Hyper-parameters needed by callers (frame duration for stamping etc.)
int parakeet_frame_dur_cs(struct parakeet_context* ctx); // centiseconds per encoder frame
int parakeet_n_mels(struct parakeet_context* ctx);
int parakeet_sample_rate(struct parakeet_context* ctx);
int parakeet_n_heads(struct parakeet_context* ctx); // encoder attention heads (Phase 2 mem policy)
int parakeet_est_enc_frames(struct parakeet_context* ctx, int n_samples); // approx T_enc for n_samples

// ---- Stage-level entry points (for crispasr-diff testing) ----
// These let the diff harness compare intermediate activations against
// a PyTorch reference. They are NOT needed by the normal transcribe
// path — use parakeet_transcribe(_ex) for inference.
//
// Returns a malloc'd F32 buffer that the caller must free(). Shape is
// reported via the out_* parameters. Returns nullptr on failure.

// Log-mel spectrogram of raw 16 kHz mono PCM.
// Output layout: row-major (n_mels, T_mel).
float* parakeet_compute_mel(struct parakeet_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                            int* out_T_mel);

// Run just the audio encoder on a mel spectrogram. Takes the output of
// parakeet_compute_mel() (or any externally-produced reference mel with
// the same layout) and returns the encoder hidden state.
// Output layout: row-major (T_enc, d_model).
float* parakeet_run_encoder(struct parakeet_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                            int* out_d_model);

// Internal smoke test: build encoder graph on a zero mel of `T_mel` frames,
// run it, and report the output T_enc. Returns T_enc on success or -1.
int parakeet_test_encoder(struct parakeet_context* ctx, int T_mel);

// Run the encoder and capture per-layer intermediates for diff testing.
// Caller passes pre-allocated row-major (T_enc, d_model) buffers in `out`:
//   out[0]    : after pre-encode (subsampling + projection)
//   out[1..N] : after each conformer layer (where N = n_layers)
// `out_count` must be at least n_layers+1; extra slots are ignored.
// Sizes are reported back via *out_T_enc / *out_d_model. Returns 0 on
// success, non-zero on failure. The C-side allocates and frees its own
// scratch; the caller-provided buffers are written into directly.
int parakeet_run_encoder_dump(struct parakeet_context* ctx, const float* mel, int n_mels, int T_mel, float** out,
                              int out_count, int* out_T_enc, int* out_d_model);

// Internal smoke test: take raw 16 kHz mono PCM, run mel + encoder, print
// encoder-output statistics. Returns T_enc on success or -1.
int parakeet_test_audio(struct parakeet_context* ctx, const float* samples, int n_samples);

// ---- Transducer component entry points (for crispasr-diff testing) ----
// These expose the predictor (LSTM) and joint network intermediates so
// crispasr-diff can validate each component against a PyTorch reference
// before testing the full decode loop.

// Run the joint encoder projection on encoder output.
// Input:  enc_frames row-major (T_enc, d_model)
// Output: malloc'd (T_enc, joint_hidden) row-major. Caller must free().
float* parakeet_joint_project_encoder(struct parakeet_context* ctx, const float* enc_frames, int T_enc, int d_model,
                                      int* out_joint_hidden);

// Legacy NeMo capture: two blank steps from zero LSTM state (SOS + pad).
// Output: malloc'd (1, pred_hidden).
// Caller must free().
float* parakeet_predictor_initial(struct parakeet_context* ctx, int* out_pred_hidden);
// Production greedy start: one blank token, zero LSTM state.
float* parakeet_predictor_sos(struct parakeet_context* ctx, int* out_pred_hidden);

// Run full joint step at a single encoder frame + predictor output.
// proj_enc: (joint_hidden,) — output of parakeet_joint_project_encoder for one frame
// pred_out: (pred_hidden,) — output of predictor step
// Output: malloc'd (vocab_total,) raw logits. Caller must free().
float* parakeet_joint_step(struct parakeet_context* ctx, const float* proj_enc, const float* pred_out,
                           int* out_vocab_total);

#ifdef __cplusplus
}
#endif
