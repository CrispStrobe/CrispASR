// crispasr_session.h — forward declarations for the session C ABI.
//
// AUTO-GENERATED from CA_EXPORT definitions in src/crispasr_c_api.cpp.
// Suppresses -Wmissing-declarations when crispasr_c_api.cpp includes this.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef CRISPASR_SHARED
#ifdef _WIN32
#ifdef CRISPASR_BUILD
#define CRISPASR_SESSION_API __declspec(dllexport)
#else
#define CRISPASR_SESSION_API __declspec(dllimport)
#endif
#else
#define CRISPASR_SESSION_API __attribute__((visibility("default")))
#endif
#else
#define CRISPASR_SESSION_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Whisper types (defined in whisper.h, forward-declared here so this
// header is self-contained without pulling in the full whisper API).
struct whisper_context;
struct whisper_context_params;
struct whisper_full_params;

struct crispasr_align_result;
typedef struct crispasr_align_result crispasr_align_result;
struct crispasr_diarize_opts_abi;
typedef struct crispasr_diarize_opts_abi crispasr_diarize_opts_abi;
struct crispasr_diarize_seg_abi;
typedef struct crispasr_diarize_seg_abi crispasr_diarize_seg_abi;
struct crispasr_open_params_v1;
typedef struct crispasr_open_params_v1 crispasr_open_params_v1;
struct crispasr_session;
typedef struct crispasr_session crispasr_session;
struct crispasr_session_result;
typedef struct crispasr_session_result crispasr_session_result;
struct crispasr_stream;
typedef struct crispasr_stream crispasr_stream;
struct crispasr_vad_abi_opts;
typedef struct crispasr_vad_abi_opts crispasr_vad_abi_opts;
struct parakeet_context;
typedef struct parakeet_context parakeet_context;
struct parakeet_result;
typedef struct parakeet_result parakeet_result;
struct whisper_context;
typedef struct whisper_context whisper_context;
struct whisper_context_params;
typedef struct whisper_context_params whisper_context_params;

CRISPASR_SESSION_API int crispasr_get_progress(void);
CRISPASR_SESSION_API void crispasr_reset_progress(void);

// 0.10.3+ (issue #208): per-session progress callback for long-form
// (chunked) transcription. Invoked once per finished window from within
// crispasr_session_transcribe_chunked[_lang] (and any auto-chunked long
// Parakeet transcribe) with the number of input samples processed so far
// and the total. `processed` is monotonically non-decreasing and ends at
// `total`. It is invoked on the calling (transcribe) thread — keep the
// callback fast and non-blocking; do not re-enter the session from it.
// Single-pass and non-Parakeet backends do not fire it. The module-level
// atomic (crispasr_get_progress) is updated in lockstep, so pure pollers
// (e.g. Dart FFI) get chunked progress without registering a callback.
typedef void (*crispasr_progress_callback)(int processed, int total, void* user_data);

// Register (or clear, with cb == NULL) the session progress callback.
// user_data is passed back verbatim to every invocation. The pointer is
// stored, not copied; keep it valid for the duration of transcribe calls.
CRISPASR_SESSION_API void crispasr_session_set_progress_callback(crispasr_session* s, crispasr_progress_callback cb,
                                                                 void* user_data);

/// Per-segment streaming callback. Fired each time a new segment is
/// committed during transcription. The segment text and timing are
/// passed directly — the callback must copy any data it needs (the
/// pointers are valid only for the duration of the call).
typedef void (*crispasr_segment_callback)(const char* text,  // segment text (UTF-8, null-terminated)
                                          int64_t t0_cs,     // start time in centiseconds
                                          int64_t t1_cs,     // end time in centiseconds
                                          int segment_index, // 0-based segment index within this transcription
                                          void* user_data    // opaque pointer from registration
);

/// Register a per-segment streaming callback on the session.
/// Pass NULL to clear. The callback is invoked on the transcription
/// thread — it must be fast and non-blocking.
CRISPASR_SESSION_API void crispasr_session_set_segment_callback(crispasr_session* s, crispasr_segment_callback cb,
                                                                void* user_data);

/// Number of streamed segments available for polling (Dart FFI path).
/// The polling buffers are per-session; this session-less API reads the
/// session that most recently streamed via the default callbacks.
CRISPASR_SESSION_API int crispasr_get_streamed_segment_count(void);

/// Drain all buffered streamed segments into a new result. Caller owns the
/// returned pointer (free with crispasr_session_result_free). Returns NULL
/// when the buffer is empty.
CRISPASR_SESSION_API crispasr_session_result* crispasr_drain_streamed_segments(void);

/// Clear the streamed-segment buffer. Call before starting a new
/// transcription to discard stale segments from the previous run.
CRISPASR_SESSION_API void crispasr_reset_streamed_segments(void);

/// Per-token streaming callback. Fired each time the decoder produces a
/// new text token during LLM-based ASR. The token text may be a partial
/// word (BPE subword). The callback must be fast and non-blocking.
typedef void (*crispasr_token_callback)(const char* token_text, // decoded token text (UTF-8)
                                        int token_index,        // 0-based token index within current segment
                                        void* user_data);

CRISPASR_SESSION_API void crispasr_session_set_token_callback(crispasr_session* s, crispasr_token_callback cb,
                                                              void* user_data);

/// Number of streamed tokens available for polling (Dart FFI path).
CRISPASR_SESSION_API int crispasr_get_streamed_token_count(void);

/// Drain all buffered streamed tokens. Returns a buffer of null-separated
/// UTF-8 strings; *out_count receives the number of tokens. Returns NULL
/// when the buffer is empty. The returned pointer is valid until the next
/// drain call on the same session, or until the session is closed.
CRISPASR_SESSION_API const char* crispasr_drain_streamed_tokens(int* out_count);

/// Clear the streamed-token buffer.
CRISPASR_SESSION_API void crispasr_reset_streamed_tokens(void);

CRISPASR_SESSION_API void crispasr_params_set_language(whisper_full_params* p, const char* lang);
CRISPASR_SESSION_API void crispasr_params_set_translate(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_detect_language(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_token_timestamps(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_n_threads(whisper_full_params* p, int n);
CRISPASR_SESSION_API void crispasr_params_set_max_len(whisper_full_params* p, int n);
CRISPASR_SESSION_API void crispasr_params_set_best_of(whisper_full_params* p, int n);
CRISPASR_SESSION_API void crispasr_params_set_split_on_word(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_no_context(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_single_segment(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_print_realtime(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_print_progress(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_print_timestamps(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_print_special(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_suppress_blank(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_temperature(whisper_full_params* p, float t);
CRISPASR_SESSION_API void crispasr_params_set_max_tokens(whisper_full_params* p, int n);
CRISPASR_SESSION_API void crispasr_params_set_initial_prompt(whisper_full_params* p, const char* prompt);
CRISPASR_SESSION_API void crispasr_params_set_vad(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_params_set_vad_model_path(whisper_full_params* p, const char* path);
CRISPASR_SESSION_API void crispasr_params_set_vad_threshold(whisper_full_params* p, float t);
CRISPASR_SESSION_API void crispasr_params_set_vad_min_speech_ms(whisper_full_params* p, int ms);
CRISPASR_SESSION_API void crispasr_params_set_vad_min_silence_ms(whisper_full_params* p, int ms);
CRISPASR_SESSION_API void crispasr_params_set_tdrz(whisper_full_params* p, int v);
CRISPASR_SESSION_API void crispasr_ctx_params_set_dtw(whisper_context_params* p, bool enable, int aheads_preset,
                                                      int n_top);
CRISPASR_SESSION_API int64_t crispasr_token_t0(whisper_context* ctx, int i_seg, int i_tok);
CRISPASR_SESSION_API int64_t crispasr_token_t1(whisper_context* ctx, int i_seg, int i_tok);
CRISPASR_SESSION_API float crispasr_token_p(whisper_context* ctx, int i_seg, int i_tok);
CRISPASR_SESSION_API int64_t crispasr_token_dtw_t(whisper_context* ctx, int i_segment, int i_token);
CRISPASR_SESSION_API void crispasr_params_set_alt_n(whisper_full_params* p, int n);
CRISPASR_SESSION_API int crispasr_token_n_alts(whisper_context* ctx, int i_seg, int i_tok);
CRISPASR_SESSION_API int32_t crispasr_token_alt_id(whisper_context* ctx, int i_seg, int i_tok, int i_alt);
CRISPASR_SESSION_API float crispasr_token_alt_p(whisper_context* ctx, int i_seg, int i_tok, int i_alt);
CRISPASR_SESSION_API int crispasr_token_alt_text(whisper_context* ctx, int i_seg, int i_tok, int i_alt, char* out,
                                                 int out_cap);
CRISPASR_SESSION_API float crispasr_detect_language(whisper_context* ctx, const float* pcm, int n_samples,
                                                    int n_threads, char* out_code, int out_cap);
CRISPASR_SESSION_API int crispasr_vad_segments(const char* vad_model_path, const float* pcm, int n_samples,
                                               int sample_rate, float threshold, int min_speech_ms, int min_silence_ms,
                                               int n_threads, bool use_gpu, float** out_spans);
CRISPASR_SESSION_API int crispasr_vad_slices(const char* vad_model_path, const float* pcm, int n_samples,
                                             int sample_rate, float threshold, int min_speech_ms, int min_silence_ms,
                                             int speech_pad_ms, float max_chunk_duration_s, int n_threads,
                                             float** out_spans);
CRISPASR_SESSION_API void crispasr_vad_free(float* spans);
CRISPASR_SESSION_API int crispasr_watermark_load_model(const char* gguf_path);
CRISPASR_SESSION_API float crispasr_watermark_detect(const float* pcm, int n_samples);
CRISPASR_SESSION_API void crispasr_watermark_embed(float* pcm, int n_samples, float alpha);
CRISPASR_SESSION_API int crispasr_lcs_dedup_prefix_count(const int32_t* prev_tail_tokens, int n_prev,
                                                         const int32_t* curr_tokens, int n_curr, int min_lcs_length);
CRISPASR_SESSION_API crispasr_stream* crispasr_stream_open(whisper_context* ctx, int n_threads, int step_ms,
                                                           int length_ms, int keep_ms, const char* language,
                                                           int translate);
CRISPASR_SESSION_API void crispasr_stream_close(crispasr_stream* s);
CRISPASR_SESSION_API int crispasr_stream_feed(crispasr_stream* s, const float* pcm, int n_samples);
CRISPASR_SESSION_API int crispasr_stream_get_text(crispasr_stream* s, char* out_text, int out_cap, double* out_t0_s,
                                                  double* out_t1_s, int64_t* out_counter);
CRISPASR_SESSION_API int crispasr_stream_flush(crispasr_stream* s);
CRISPASR_SESSION_API void crispasr_stream_set_live_decode(crispasr_stream* s, int enabled);
CRISPASR_SESSION_API parakeet_context* crispasr_parakeet_init(const char* model_path, int n_threads, int use_flash);
CRISPASR_SESSION_API void crispasr_parakeet_free(parakeet_context* ctx);
CRISPASR_SESSION_API parakeet_result* crispasr_parakeet_transcribe(parakeet_context* ctx, const float* pcm,
                                                                   int n_samples, int64_t t_offset_cs);
CRISPASR_SESSION_API const char* crispasr_parakeet_result_text(parakeet_result* r);
CRISPASR_SESSION_API int crispasr_parakeet_result_n_words(parakeet_result* r);
CRISPASR_SESSION_API const char* crispasr_parakeet_result_word_text(parakeet_result* r, int i);
CRISPASR_SESSION_API int64_t crispasr_parakeet_result_word_t0(parakeet_result* r, int i);
CRISPASR_SESSION_API int64_t crispasr_parakeet_result_word_t1(parakeet_result* r, int i);
CRISPASR_SESSION_API int crispasr_parakeet_result_n_tokens(parakeet_result* r);
CRISPASR_SESSION_API const char* crispasr_parakeet_result_token_text(parakeet_result* r, int i);
CRISPASR_SESSION_API int64_t crispasr_parakeet_result_token_t0(parakeet_result* r, int i);
CRISPASR_SESSION_API int64_t crispasr_parakeet_result_token_t1(parakeet_result* r, int i);
CRISPASR_SESSION_API float crispasr_parakeet_result_token_p(parakeet_result* r, int i);
CRISPASR_SESSION_API void crispasr_parakeet_result_free(parakeet_result* r);
// Issue #214: set the preferred GPU backend name ("cuda", "vulkan",
// "metal"). Call before any crispasr_session_open*. NULL or "" = auto.
CRISPASR_SESSION_API void crispasr_set_gpu_backend(const char* name);

CRISPASR_SESSION_API int crispasr_detect_backend_from_gguf(const char* path, char* out_name, int out_cap);
CRISPASR_SESSION_API crispasr_session* crispasr_session_open_explicit(const char* model_path, const char* backend_name,
                                                                      int n_threads);
CRISPASR_SESSION_API crispasr_session* crispasr_session_open(const char* model_path, int n_threads);
CRISPASR_SESSION_API crispasr_session* crispasr_session_open_with_params(const char* model_path,
                                                                         const char* backend_name,
                                                                         const crispasr_open_params_v1* params);
CRISPASR_SESSION_API const char* crispasr_session_backend(crispasr_session* s);
CRISPASR_SESSION_API int crispasr_session_available_backends(char* out_csv, int out_cap);
// Acoustic language detected by the last transcribe, as an ISO-639-1 code
// (whisper only; other backends fall back to the source-language hint, then
// "unknown"). Writes into out_buf (NUL-terminated, truncated to out_cap) and
// returns the code length in bytes, or -1 on bad args. Distinct from the
// text-LID pass crispasr_text_detect_language.
CRISPASR_SESSION_API int crispasr_session_detected_language(crispasr_session* s, char* out_buf, int out_cap);
// CTC vocabulary access (Omni CTC backend). crispasr_session_n_vocab returns
// the number of SentencePiece pieces in the loaded model (0 for backends that
// don't expose a CTC vocab); crispasr_session_token_text maps a token id in
// [0, n_vocab) to its raw piece (U+2581 word-boundary marker intact), or ""
// when out of range or unsupported. Pairs with crispasr_session_result_logits
// to detokenize a greedy CTC decode.
CRISPASR_SESSION_API int crispasr_session_n_vocab(crispasr_session* s);
CRISPASR_SESSION_API const char* crispasr_session_token_text(crispasr_session* s, int id);
CRISPASR_SESSION_API crispasr_session_result* crispasr_session_transcribe_lang(crispasr_session* s, const float* pcm,
                                                                               int n_samples, const char* language);
CRISPASR_SESSION_API crispasr_session_result* crispasr_session_transcribe(crispasr_session* s, const float* pcm,
                                                                          int n_samples);
// 0.8.7+: chunked-encode transcribe (issue #208). Forces the Parakeet
// backend through its bounded long-form path (overlapping short-window
// transcribe-and-merge for non-JA models, streamed encoder for the JA-only
// model) regardless of audio length, so long files transcribe in bounded
// time AND recover the sections a single full-length pass drops.
// `chunk_seconds <= 0` keeps the per-model defaults; otherwise it sets the
// non-JA window length / the JA streamed window. `overlap_seconds < 0`
// uses the default. For non-Parakeet backends the chunk params are inert
// and this behaves exactly like crispasr_session_transcribe[_lang].
CRISPASR_SESSION_API crispasr_session_result* crispasr_session_transcribe_chunked_lang(
    crispasr_session* s, const float* pcm, int n_samples, int chunk_seconds, int overlap_seconds, const char* language);
CRISPASR_SESSION_API crispasr_session_result* crispasr_session_transcribe_chunked(crispasr_session* s, const float* pcm,
                                                                                  int n_samples, int chunk_seconds,
                                                                                  int overlap_seconds);
CRISPASR_SESSION_API crispasr_session_result* crispasr_session_transcribe_vad_lang(
    crispasr_session* s, const float* pcm, int n_samples, int sample_rate, const char* vad_model_path,
    const crispasr_vad_abi_opts* opts_or_null, const char* language);
CRISPASR_SESSION_API crispasr_session_result* crispasr_session_transcribe_vad(
    crispasr_session* s, const float* pcm, int n_samples, int sample_rate, const char* vad_model_path,
    const crispasr_vad_abi_opts* opts_or_null);
CRISPASR_SESSION_API int crispasr_diarize_segments_abi(const float* left_pcm, const float* right_pcm, int32_t n_samples,
                                                       int32_t is_stereo, crispasr_diarize_seg_abi* segs,
                                                       int32_t n_segs, const crispasr_diarize_opts_abi* opts);
CRISPASR_SESSION_API int crispasr_detect_language_pcm(const float* samples, int32_t n_samples, int32_t method,
                                                      const char* model_path, int32_t n_threads, int32_t use_gpu,
                                                      int32_t gpu_device, int32_t flash_attn, char* out_lang_buf,
                                                      int32_t out_lang_cap, float* out_confidence);
CRISPASR_SESSION_API int crispasr_enhance_audio_rnnoise(const float* in_pcm, int32_t n_samples, float* out_pcm,
                                                        int32_t out_cap);
CRISPASR_SESSION_API int crispasr_text_detect_language(const char* text, const char* model_path, int32_t n_threads,
                                                       char* out_label_buf, int32_t out_label_cap,
                                                       float* out_confidence);
CRISPASR_SESSION_API crispasr_align_result* crispasr_align_words_abi(const char* aligner_model, const char* transcript,
                                                                     const float* samples, int32_t n_samples,
                                                                     int64_t t_offset_cs, int32_t n_threads);
CRISPASR_SESSION_API int crispasr_align_result_n_words(crispasr_align_result* r);
CRISPASR_SESSION_API const char* crispasr_align_result_word_text(crispasr_align_result* r, int i);
CRISPASR_SESSION_API int64_t crispasr_align_result_word_t0(crispasr_align_result* r, int i);
CRISPASR_SESSION_API int64_t crispasr_align_result_word_t1(crispasr_align_result* r, int i);
CRISPASR_SESSION_API void crispasr_align_result_free(crispasr_align_result* r);
CRISPASR_SESSION_API int crispasr_cache_ensure_file_abi(const char* filename, const char* url, int32_t quiet,
                                                        const char* cache_dir_override, char* out_buf, int32_t out_cap);
CRISPASR_SESSION_API int crispasr_cache_dir_abi(const char* cache_dir_override, char* out_buf, int32_t out_cap);
CRISPASR_SESSION_API int crispasr_registry_lookup_abi(const char* backend, char* out_filename, int32_t filename_cap,
                                                      char* out_url, int32_t url_cap, char* out_size, int32_t size_cap);
CRISPASR_SESSION_API int crispasr_registry_lookup_by_filename_abi(const char* filename, char* out_filename,
                                                                  int32_t filename_cap, char* out_url, int32_t url_cap,
                                                                  char* out_size, int32_t size_cap);
CRISPASR_SESSION_API int crispasr_registry_list_backends_abi(char* out_csv, int32_t out_cap);
CRISPASR_SESSION_API int crispasr_session_result_n_segments(crispasr_session_result* r);
CRISPASR_SESSION_API const char* crispasr_session_result_segment_text(crispasr_session_result* r, int i);
CRISPASR_SESSION_API int64_t crispasr_session_result_segment_t0(crispasr_session_result* r, int i);
CRISPASR_SESSION_API int64_t crispasr_session_result_segment_t1(crispasr_session_result* r, int i);
CRISPASR_SESSION_API int crispasr_session_result_n_words(crispasr_session_result* r, int i_seg);
CRISPASR_SESSION_API const char* crispasr_session_result_word_text(crispasr_session_result* r, int i_seg, int i_word);
CRISPASR_SESSION_API int64_t crispasr_session_result_word_t0(crispasr_session_result* r, int i_seg, int i_word);
CRISPASR_SESSION_API int64_t crispasr_session_result_word_t1(crispasr_session_result* r, int i_seg, int i_word);
CRISPASR_SESSION_API float crispasr_session_result_word_p(crispasr_session_result* r, int i_seg, int i_word);
// Whisper's per-segment no-speech probability (the <|nospeech|> token
// posterior) in [0, 1]. Only the whisper backend populates it; other backends
// and out-of-range indices return the -1.0 sentinel ("no data").
CRISPASR_SESSION_API float crispasr_session_result_segment_no_speech_prob(crispasr_session_result* r, int i_seg);
// Per-frame CTC logits (opted in via crispasr_session_set_return_logits) for
// backends that produce a dense CTC grid (Omni CTC, wav2vec2/hubert/data2vec,
// canary-ctc). Frame-major: logits[t * n_logit_vocab + v]. Raw pre-softmax for
// Omni & wav2vec2; log-probabilities for canary-ctc. _logits returns NULL when
// none captured.
CRISPASR_SESSION_API int crispasr_session_result_n_logit_frames(crispasr_session_result* r);
CRISPASR_SESSION_API int crispasr_session_result_n_logit_vocab(crispasr_session_result* r);
CRISPASR_SESSION_API const float* crispasr_session_result_logits(crispasr_session_result* r);
CRISPASR_SESSION_API int crispasr_session_result_word_n_alts(crispasr_session_result* r, int i_seg, int i_word);
CRISPASR_SESSION_API const char* crispasr_session_result_word_alt_text(crispasr_session_result* r, int i_seg,
                                                                       int i_word, int i_alt);
CRISPASR_SESSION_API float crispasr_session_result_word_alt_p(crispasr_session_result* r, int i_seg, int i_word,
                                                              int i_alt);
CRISPASR_SESSION_API void crispasr_session_result_free(crispasr_session_result* r);
// Issue #257: parakeet/canary local-attention window in encoder frames (~80 ms
// each) — NeMo change_attention_model("rel_pos_local_attn", [left, right]).
// Bounds long-audio encoder memory to O(T·window) instead of O(T²). Negative
// values = full attention; INT_MIN,INT_MIN = clear (use the model default).
// No-op for non-parakeet backends. Returns 0 on success.
CRISPASR_SESSION_API int crispasr_session_set_parakeet_att_context(crispasr_session* s, int left, int right);
CRISPASR_SESSION_API int crispasr_session_set_codec_path(crispasr_session* s, const char* path);
CRISPASR_SESSION_API int crispasr_session_set_voice(crispasr_session* s, const char* path,
                                                    const char* ref_text_or_null);
// #201: configure the TADA encoder + aligner GGUFs used for on-the-fly voice
// cloning, i.e. crispasr_session_set_voice(s, "ref.wav", "<transcript>") on a
// TADA session. The .wav clone path is opt-in (experimental) — enable it with
// the env var CRISPASR_TADA_WAV_CLONE=1; otherwise a .wav voice is rejected as
// before. Either path may be NULL to clear it and fall back to
// auto-resolution (next to the model, then the cache dir). The aligner is
// language-specific (tada-aligner-<lang>.gguf); the language follows the
// session's source-language hint. No-op (returns -1) for non-TADA sessions.
CRISPASR_SESSION_API int crispasr_session_tada_set_makeref_models(crispasr_session* s, const char* encoder_gguf,
                                                                  const char* aligner_gguf);
CRISPASR_SESSION_API int crispasr_session_set_speaker_name(crispasr_session* s, const char* name);
CRISPASR_SESSION_API int crispasr_session_set_speaker_id(crispasr_session* s, int id);
CRISPASR_SESSION_API int crispasr_session_n_speakers(crispasr_session* s);
CRISPASR_SESSION_API const char* crispasr_session_get_speaker_name(crispasr_session* s, int i);
CRISPASR_SESSION_API int crispasr_session_set_instruct(crispasr_session* s, const char* instruct);
CRISPASR_SESSION_API int crispasr_session_is_custom_voice(crispasr_session* s);
CRISPASR_SESSION_API int crispasr_session_is_voice_design(crispasr_session* s);
CRISPASR_SESSION_API float* crispasr_session_synthesize_raw(crispasr_session* s, const char* text, int* out_n_samples);
CRISPASR_SESSION_API float* crispasr_session_synthesize(crispasr_session* s, const char* text, int* out_n_samples);

// Streaming synthesis: fires `cb` once per sentence chunk with that chunk's
// watermarked PCM (backend-native sample rate, same as synthesize) as it is
// produced. The PCM is owned by the call and freed after `cb` returns — copy
// it if you need to keep it. `is_final` is 1 on the last chunk. Returns 0 on
// success, -1 on bad args.
typedef void (*crispasr_pcm_stream_cb)(const float* pcm, int n_samples, int is_final, void* user_data);
CRISPASR_SESSION_API int crispasr_session_synthesize_streaming(crispasr_session* s, const char* text,
                                                               crispasr_pcm_stream_cb cb, void* user_data);

CRISPASR_SESSION_API void crispasr_pcm_free(float* pcm);
CRISPASR_SESSION_API float* crispasr_session_speech_to_speech(crispasr_session* s, const float* in_samples,
                                                              int n_in_samples, char** out_text, int* out_n_samples);
CRISPASR_SESSION_API int crispasr_session_set_hotwords(crispasr_session* s, const char* hotwords, float boost);

// Source separation: split audio into N stems (drums, bass, other, vocals).
// Input: stereo interleaved PCM at the model's native rate (44100 Hz for htdemucs).
// Returns stem count (>0) on success, -1 on error. Call crispasr_session_separate_stem
// to retrieve individual stem PCM after a successful separate call.
CRISPASR_SESSION_API int crispasr_session_separate(crispasr_session* s, const float* pcm_stereo, int n_samples);
CRISPASR_SESSION_API int crispasr_session_separate_n_stems(crispasr_session* s);
CRISPASR_SESSION_API const char* crispasr_session_separate_stem_name(crispasr_session* s, int stem_idx);
// Returns pointer to interleaved stereo PCM for stem_idx. Owned by the session;
// valid until the next separate() call or session close. *out_n_samples receives
// the per-channel sample count.
CRISPASR_SESSION_API const float* crispasr_session_separate_stem(crispasr_session* s, int stem_idx, int* out_n_samples);
CRISPASR_SESSION_API int crispasr_session_separate_sample_rate(crispasr_session* s);

// Pitch (F0) estimation: monophonic pitch track from mono PCM at the model's
// native rate (16000 Hz for crepe). `hop_ms` <= 0 uses the model default (10 ms).
// Returns the frame count (>0) on success, -1 on error. Frames stay valid until
// the next pitch() call or session close.
CRISPASR_SESSION_API int crispasr_session_pitch(crispasr_session* s, const float* pcm_16k, int n_samples, float hop_ms);
CRISPASR_SESSION_API int crispasr_session_pitch_n_frames(crispasr_session* s);
// Single frame by index. Any out_* pointer may be NULL. Returns 0 on success.
CRISPASR_SESSION_API int crispasr_session_pitch_frame(crispasr_session* s, int idx, float* out_time_ms,
                                                      float* out_f0_hz, float* out_voiced_prob);
// Flat view of every frame: 3 floats per frame {time_ms, f0_hz, voiced_prob},
// frame-major. Owned by the session. *out_n_frames receives the frame count.
CRISPASR_SESSION_API const float* crispasr_session_pitch_frames(crispasr_session* s, int* out_n_frames);
CRISPASR_SESSION_API int crispasr_session_pitch_sample_rate(crispasr_session* s);
CRISPASR_SESSION_API const char* crispasr_session_last_synth_error(crispasr_session* s);
CRISPASR_SESSION_API char* crispasr_session_translate_text(crispasr_session* s, const char* text, const char* src_lang,
                                                           const char* tgt_lang, int max_tokens);
CRISPASR_SESSION_API void crispasr_session_translate_text_free(char* text);
CRISPASR_SESSION_API crispasr_stream* crispasr_session_stream_open(crispasr_session* s, int n_threads, int step_ms,
                                                                   int length_ms, int keep_ms, const char* language,
                                                                   int translate);
CRISPASR_SESSION_API void crispasr_session_close(crispasr_session* s);
CRISPASR_SESSION_API void* crispasr_punc_init(const char* model_path);
CRISPASR_SESSION_API const char* crispasr_punc_process(void* ctx, const char* text);
CRISPASR_SESSION_API void crispasr_punc_free_text(const char* text);
CRISPASR_SESSION_API void crispasr_punc_free(void* ctx);
CRISPASR_SESSION_API void* crispasr_punc_init(const char*);
CRISPASR_SESSION_API const char* crispasr_punc_process(void*, const char*);
CRISPASR_SESSION_API void crispasr_punc_free_text(const char*);
CRISPASR_SESSION_API void crispasr_punc_free(void*);
CRISPASR_SESSION_API void* crispasr_truecase_init(const char* model_path);
CRISPASR_SESSION_API const char* crispasr_truecase_process(void* ctx, const char* text);
CRISPASR_SESSION_API void crispasr_truecase_free_text(const char* text);
CRISPASR_SESSION_API void crispasr_truecase_free(void* ctx);
CRISPASR_SESSION_API void* crispasr_truecase_init(const char* model_path);
CRISPASR_SESSION_API const char* crispasr_truecase_process(void* ctx, const char* text);
CRISPASR_SESSION_API void crispasr_truecase_free_text(const char* text);
CRISPASR_SESSION_API void crispasr_truecase_free(void* ctx);
CRISPASR_SESSION_API void* crispasr_truecase_init(const char*);
CRISPASR_SESSION_API const char* crispasr_truecase_process(void*, const char*);
CRISPASR_SESSION_API void crispasr_truecase_free_text(const char*);
CRISPASR_SESSION_API void crispasr_truecase_free(void*);
CRISPASR_SESSION_API void* crispasr_pcs_init(const char* model_path);
CRISPASR_SESSION_API const char* crispasr_pcs_process(void* ctx, const char* text);
CRISPASR_SESSION_API void crispasr_pcs_free_text(const char* text);
CRISPASR_SESSION_API void crispasr_pcs_free(void* ctx);
CRISPASR_SESSION_API void* crispasr_pcs_init(const char*);
CRISPASR_SESSION_API const char* crispasr_pcs_process(void*, const char*);
CRISPASR_SESSION_API void crispasr_pcs_free_text(const char*);
CRISPASR_SESSION_API void crispasr_pcs_free(void*);
CRISPASR_SESSION_API int crispasr_transcribe_parallel(struct whisper_context* ctx, struct whisper_full_params params,
                                                      const float* samples, int n_samples, int n_processors);
CRISPASR_SESSION_API const char* crispasr_c_api_version(void);
CRISPASR_SESSION_API const char* crispasr_dart_helpers_version(void);
CRISPASR_SESSION_API bool crispasr_kokoro_lang_is_german_abi(const char* lang);
CRISPASR_SESSION_API bool crispasr_kokoro_lang_has_native_voice_abi(const char* lang);
CRISPASR_SESSION_API int crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang,
                                                                    char* out_path, int out_path_len);
CRISPASR_SESSION_API int crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang,
                                                                    char* out_path, int out_path_len, char* out_picked,
                                                                    int out_picked_len);
CRISPASR_SESSION_API int crispasr_session_kokoro_clear_phoneme_cache(crispasr_session* s);
CRISPASR_SESSION_API bool crispasr_kokoro_lang_is_german_abi(const char*);
CRISPASR_SESSION_API bool crispasr_kokoro_lang_has_native_voice_abi(const char*);
CRISPASR_SESSION_API int crispasr_kokoro_resolve_model_for_lang_abi(const char*, const char*, char*, int);
CRISPASR_SESSION_API int crispasr_kokoro_resolve_fallback_voice_abi(const char*, const char*, char*, int, char*, int);
CRISPASR_SESSION_API int crispasr_session_kokoro_clear_phoneme_cache(crispasr_session*);
CRISPASR_SESSION_API int crispasr_session_set_source_language(crispasr_session* s, const char* lang);
CRISPASR_SESSION_API int crispasr_session_set_target_language(crispasr_session* s, const char* lang);
CRISPASR_SESSION_API int crispasr_session_set_punctuation(crispasr_session* s, int enable);
// Select + load a punctuation-restoration model (alias auto|firered|fullstop|
// punctuate-all|pcs, or a .gguf path; "none"/NULL unloads). Auto-downloads on
// first use. Restores punctuation on backends that emit none (parakeet, CTC).
// Returns 0 on success/unload, -1 bad handle, -2 load failed, -3 not compiled.
CRISPASR_SESSION_API int crispasr_session_set_punc_model(crispasr_session* s, const char* punc_model);
CRISPASR_SESSION_API int crispasr_session_set_translate(crispasr_session* s, int enable);
CRISPASR_SESSION_API int crispasr_session_set_ask(crispasr_session* s, const char* prompt);
CRISPASR_SESSION_API int crispasr_session_set_temperature(crispasr_session* s, float temperature, uint64_t seed);
CRISPASR_SESSION_API int crispasr_session_set_tts_seed(crispasr_session* s, uint64_t seed);
CRISPASR_SESSION_API int crispasr_session_set_tts_steps(crispasr_session* s, int steps);
CRISPASR_SESSION_API int crispasr_session_set_tts_cfg_scale(crispasr_session* s, float scale);
CRISPASR_SESSION_API int crispasr_session_set_tts_num_candidates(crispasr_session* s, int n);
CRISPASR_SESSION_API int crispasr_session_set_g2p_dict(crispasr_session* s, const char* source);
CRISPASR_SESSION_API int crispasr_session_set_top_p(crispasr_session* s, float top_p);
CRISPASR_SESSION_API int crispasr_session_set_min_p(crispasr_session* s, float min_p);
CRISPASR_SESSION_API int crispasr_session_set_repetition_penalty(crispasr_session* s, float r);
CRISPASR_SESSION_API int crispasr_session_set_top_k(crispasr_session* s, int top_k);
CRISPASR_SESSION_API int crispasr_session_set_do_sample(crispasr_session* s, int enable);
CRISPASR_SESSION_API int crispasr_session_set_cfg_weight(crispasr_session* s, float cfg_weight);
CRISPASR_SESSION_API int crispasr_session_set_tts_noise_temp(crispasr_session* s, float noise_temp);
CRISPASR_SESSION_API int crispasr_session_set_exaggeration(crispasr_session* s, float exaggeration);
CRISPASR_SESSION_API int crispasr_session_set_max_speech_tokens(crispasr_session* s, int n);
CRISPASR_SESSION_API int crispasr_session_set_length_scale(crispasr_session* s, float scale);
CRISPASR_SESSION_API int crispasr_session_set_best_of(crispasr_session* s, int n);
CRISPASR_SESSION_API int crispasr_session_set_max_new_tokens(crispasr_session* s, int n);
CRISPASR_SESSION_API int crispasr_session_set_frequency_penalty(crispasr_session* s, float penalty);
CRISPASR_SESSION_API int crispasr_session_set_beam_size(crispasr_session* s, int n);
CRISPASR_SESSION_API int crispasr_session_set_return_logits(crispasr_session* s, int enable);
CRISPASR_SESSION_API int crispasr_session_set_grammar_text(crispasr_session* s, const char* gbnf_text,
                                                           const char* root_rule, float penalty);
CRISPASR_SESSION_API int crispasr_session_set_fallback_thresholds(crispasr_session* s, float entropy_thold,
                                                                  float logprob_thold, float no_speech_thold,
                                                                  float temperature_inc);
CRISPASR_SESSION_API int crispasr_session_set_alt_n(crispasr_session* s, int n);
CRISPASR_SESSION_API int crispasr_session_set_whisper_decode_extras(crispasr_session* s, int suppress_nst,
                                                                    const char* suppress_regex,
                                                                    int carry_initial_prompt);
CRISPASR_SESSION_API int crispasr_session_detect_language(crispasr_session* s, const float* pcm, int n_samples,
                                                          const char* lid_model_path, int method, char* out_lang,
                                                          int out_lang_cap, float* out_prob);
CRISPASR_SESSION_API void* crispasr_titanet_init(const char* model_path, int32_t n_threads);
CRISPASR_SESSION_API void crispasr_titanet_free(void* ctx);
CRISPASR_SESSION_API int32_t crispasr_titanet_embed(void* ctx, const float* pcm_16k, int32_t n_samples, float* out);
CRISPASR_SESSION_API float crispasr_titanet_cosine_sim(const float* a, const float* b, int32_t dim);
// Speaker database — closed-roster, consent-gated (issue #266).
// `expected_names_csv` is the comma-separated roster of enrolled
// participants the caller asserts are present in the audio (e.g.
// "Alice,Bob"); the db is narrowed to exactly those profiles.
// `consent_attested` affirms a lawful basis + explicit consent from
// every enrolled person (GDPR Art. 9). Open 1:N identification is
// deliberately unsupported: crispasr_speaker_db_open refuses without
// both, and the legacy ungated _load/_enroll symbols below refuse at
// runtime (kept so old callers fail loudly, not at link time).
CRISPASR_SESSION_API void* crispasr_speaker_db_open(const char* dir_path, const char* expected_names_csv,
                                                    int32_t consent_attested);
CRISPASR_SESSION_API void crispasr_speaker_db_free(void* db);
CRISPASR_SESSION_API int32_t crispasr_speaker_db_count(const void* db);
CRISPASR_SESSION_API float crispasr_speaker_db_match(const void* db, const float* embedding, int32_t dim,
                                                     float threshold, char* out_name, int32_t out_cap);
CRISPASR_SESSION_API int32_t crispasr_speaker_db_enroll2(const char* dir_path, const char* name, const float* embedding,
                                                         int32_t dim, int32_t consent_attested);
// Legacy (pre-#266) entry points — always refuse at runtime.
CRISPASR_SESSION_API void* crispasr_speaker_db_load(const char* dir_path);
CRISPASR_SESSION_API int32_t crispasr_speaker_db_enroll(const char* dir_path, const char* name, const float* embedding,
                                                        int32_t dim);
CRISPASR_SESSION_API void* crispasr_speaker_embedder_make_abi(const char* model_spec, int32_t n_threads,
                                                              const char* cache_dir);
CRISPASR_SESSION_API void crispasr_speaker_embedder_free_abi(void* embedder);
CRISPASR_SESSION_API int32_t crispasr_speaker_embedder_dim_abi(const void* embedder);
CRISPASR_SESSION_API int32_t crispasr_speaker_embedder_embed_abi(void* embedder, const float* pcm_16k,
                                                                 int32_t n_samples, float* out);
CRISPASR_SESSION_API const char* crispasr_speaker_embedder_name_abi(const void* embedder);
CRISPASR_SESSION_API int32_t crispasr_speaker_cluster_abi(const float* embeddings, int32_t n, int32_t dim,
                                                          float merge_threshold, int32_t max_speakers,
                                                          int32_t* labels_out);
CRISPASR_SESSION_API void* crispasr_pyannote_cache_compute_abi(const float* full_audio, int32_t n_samples,
                                                               const char* model_path, int32_t n_threads);
CRISPASR_SESSION_API void crispasr_pyannote_cache_free_abi(void* cache);
CRISPASR_SESSION_API int32_t crispasr_pyannote_cache_apply_abi(const void* cache, int64_t slice_t0_cs,
                                                               crispasr_diarize_seg_abi* segs, int32_t n_segs);

#ifdef __cplusplus
}
#endif
