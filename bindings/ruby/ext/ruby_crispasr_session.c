// Minimal TTS surface for the Ruby binding. Exposes the unified
// CrispASR Session API for TTS-capable backends (kokoro, vibevoice,
// qwen3-tts, orpheus) plus the kokoro per-language model + voice
// resolver (PLAN #56 opt 2b).
//
// Surface (under module `CrispASR::Session`):
//   open(model_path, n_threads) -> handle
//   close(handle)
//   set_codec_path(handle, path)
//   set_voice(handle, path, ref_text=nil)
//   set_speaker_name(handle, name)             # orpheus + qwen3-tts CV
//   speakers(handle) -> Array<String>
//   set_instruct(handle, instruct)             # qwen3-tts VoiceDesign
//   is_custom_voice(handle) -> Boolean         # qwen3-tts variant detect
//   is_voice_design(handle) -> Boolean         # qwen3-tts variant detect
//   synthesize(handle, text) -> Array<Float>   # 24 kHz mono PCM
//
// And a singleton method:
//   CrispASR::Session.kokoro_resolve_for_lang(model_path, lang)
//     -> { model_path:, voice_path:, voice_name:, backbone_swapped: }

#include <ruby.h>
#include <ruby/thread.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Forward-declare the C ABI exported by libcrispasr. Full prototypes
// live in src/crispasr_c_api.cpp and src/kokoro.h.
struct CrispasrSession;
extern struct CrispasrSession* crispasr_session_open(const char* model_path, int n_threads);
extern void                    crispasr_session_close(struct CrispasrSession* s);
extern int                     crispasr_session_set_codec_path(struct CrispasrSession* s, const char* path);
extern int                     crispasr_session_set_voice(struct CrispasrSession* s, const char* path,
                                                          const char* ref_text_or_null);
extern int                     crispasr_session_set_speaker_name(struct CrispasrSession* s, const char* name);
extern int                     crispasr_session_n_speakers(struct CrispasrSession* s);
extern const char*             crispasr_session_get_speaker_name(struct CrispasrSession* s, int i);
extern int                     crispasr_session_set_instruct(struct CrispasrSession* s, const char* instruct);
extern int                     crispasr_session_is_custom_voice(struct CrispasrSession* s);
extern int                     crispasr_session_is_voice_design(struct CrispasrSession* s);
extern float*                  crispasr_session_synthesize(struct CrispasrSession* s, const char* text,
                                                           int* out_n_samples);
extern void                    crispasr_pcm_free(float* pcm);
extern int                     crispasr_session_kokoro_clear_phoneme_cache(struct CrispasrSession* s);
extern int                     crispasr_session_set_source_language(struct CrispasrSession* s, const char* lang);
extern int                     crispasr_session_set_target_language(struct CrispasrSession* s, const char* lang);
extern int                     crispasr_session_set_punctuation(struct CrispasrSession* s, int enable);
extern int                     crispasr_session_set_translate(struct CrispasrSession* s, int enable);
extern int                     crispasr_session_set_temperature(struct CrispasrSession* s, float temperature,
                                                                unsigned long long seed);
extern int                     crispasr_session_set_tts_seed(struct CrispasrSession* s, unsigned long long seed);
extern int                     crispasr_session_set_max_new_tokens(struct CrispasrSession* s, int n);
extern int                     crispasr_session_set_frequency_penalty(struct CrispasrSession* s, float penalty);
extern int                     crispasr_session_set_tts_steps(struct CrispasrSession* s, int steps);
extern int                     crispasr_session_set_top_p(struct CrispasrSession* s, float top_p);
extern int                     crispasr_session_set_min_p(struct CrispasrSession* s, float min_p);
extern int                     crispasr_session_set_repetition_penalty(struct CrispasrSession* s, float r);
extern int                     crispasr_session_set_cfg_weight(struct CrispasrSession* s, float cfg_weight);
extern int                     crispasr_session_set_exaggeration(struct CrispasrSession* s, float exaggeration);
extern int                     crispasr_session_set_max_speech_tokens(struct CrispasrSession* s, int n);
extern int                     crispasr_session_set_length_scale(struct CrispasrSession* s, float scale);
extern int                     crispasr_session_set_best_of(struct CrispasrSession* s, int n);
extern int                     crispasr_session_set_beam_size(struct CrispasrSession* s, int n);
extern int                     crispasr_session_set_grammar_text(struct CrispasrSession* s, const char* gbnf_text,
                                                                  const char* root_rule, float penalty);
extern int                     crispasr_session_set_fallback_thresholds(struct CrispasrSession* s,
                                                                         float entropy_thold, float logprob_thold,
                                                                         float no_speech_thold,
                                                                         float temperature_inc);
extern int                     crispasr_session_set_alt_n(struct CrispasrSession* s, int n);
extern int                     crispasr_session_set_whisper_decode_extras(struct CrispasrSession* s,
                                                                           int suppress_nst,
                                                                           const char* suppress_regex,
                                                                           int carry_initial_prompt);
extern int                     crispasr_session_set_ask(struct CrispasrSession* s, const char* prompt);
extern int                     crispasr_session_detect_language(struct CrispasrSession* s, const float* pcm,
                                                                int n_samples, const char* lid_model_path, int method,
                                                                char* out_lang, int out_lang_cap, float* out_prob);
extern int                     crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang,
                                                                          char* out_path, int out_path_len);
extern int                     crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang,
                                                                          char* out_path, int out_path_len,
                                                                          char* out_picked, int out_picked_len);

// --- ASR transcription (PLAN #59) ---
struct crispasr_session_result;
extern struct crispasr_session_result* crispasr_session_transcribe(struct CrispasrSession* s,
                                                                    const float* pcm, int n_samples);
extern struct crispasr_session_result* crispasr_session_transcribe_lang(struct CrispasrSession* s,
                                                                        const float* pcm, int n_samples,
                                                                        const char* language);
extern int          crispasr_session_result_n_segments(struct crispasr_session_result* r);
extern const char*  crispasr_session_result_segment_text(struct crispasr_session_result* r, int i);
extern int64_t      crispasr_session_result_segment_t0(struct crispasr_session_result* r, int i);
extern int64_t      crispasr_session_result_segment_t1(struct crispasr_session_result* r, int i);
extern int          crispasr_session_result_n_words(struct crispasr_session_result* r, int i_seg);
extern const char*  crispasr_session_result_word_text(struct crispasr_session_result* r, int i_seg, int i_word);
extern int64_t      crispasr_session_result_word_t0(struct crispasr_session_result* r, int i_seg, int i_word);
extern int64_t      crispasr_session_result_word_t1(struct crispasr_session_result* r, int i_seg, int i_word);
extern float        crispasr_session_result_word_p(struct crispasr_session_result* r, int i_seg, int i_word);
extern void         crispasr_session_result_free(struct crispasr_session_result* r);

// --- Punctuation (PLAN #59) ---
extern void*        crispasr_punc_init(const char* model_path);
extern const char*  crispasr_punc_process(void* ctx, const char* text);
extern void         crispasr_punc_free_text(const char* text);
extern void         crispasr_punc_free(void* ctx);

// --- VAD (PLAN #59) ---
extern int  crispasr_vad_segments(const char* vad_model_path, const float* pcm, int n_samples,
                                  int sample_rate, float threshold, int min_speech_ms, int min_silence_ms,
                                  int n_threads, int use_gpu, float** out_spans);
extern void crispasr_vad_free(float* spans);

// --- Alignment (PLAN #59) ---
struct crispasr_align_result;
extern struct crispasr_align_result* crispasr_align_words_abi(const char* aligner_model, const char* transcript,
                                                               const float* samples, int n_samples, int64_t t_offset_cs,
                                                               int n_threads);
extern int          crispasr_align_result_n_words(struct crispasr_align_result* r);
extern const char*  crispasr_align_result_word_text(struct crispasr_align_result* r, int i);
extern int64_t      crispasr_align_result_word_t0(struct crispasr_align_result* r, int i);
extern int64_t      crispasr_align_result_word_t1(struct crispasr_align_result* r, int i);
extern void         crispasr_align_result_free(struct crispasr_align_result* r);

// --- Streaming (PLAN #62b): rolling-window decoder. Whisper-only at the C-ABI today.
struct CrispasrStream;
extern struct CrispasrStream* crispasr_session_stream_open(struct CrispasrSession* s, int n_threads,
                                                           int step_ms, int length_ms, int keep_ms,
                                                           const char* language, int translate);
extern int                    crispasr_stream_feed(struct CrispasrStream* s, const float* pcm, int n_samples);
extern int                    crispasr_stream_get_text(struct CrispasrStream* s, char* out_text, int out_cap,
                                                       double* out_t0_s, double* out_t1_s,
                                                       long long* out_counter);
extern int                    crispasr_stream_flush(struct CrispasrStream* s);
extern void                   crispasr_stream_close(struct CrispasrStream* s);

// --- Microphone capture (PLAN #62d): miniaudio-backed cross-platform.
struct crispasr_mic;
typedef void (*crispasr_mic_callback)(const float* pcm, int n_samples, void* userdata);
extern struct crispasr_mic*   crispasr_mic_open(int sample_rate, int channels,
                                                crispasr_mic_callback cb, void* userdata);
extern int                    crispasr_mic_start(struct crispasr_mic* m);
extern int                    crispasr_mic_stop(struct crispasr_mic* m);
extern void                   crispasr_mic_close(struct crispasr_mic* m);
extern const char*            crispasr_mic_default_device_name(void);

// --- Full C-ABI parity additions ---
extern void crispasr_stream_set_live_decode(struct CrispasrStream* s, int enabled);

// params_set_* on whisper_full_params
extern void crispasr_params_set_language(void* p, const char* lang);
extern void crispasr_params_set_translate(void* p, int v);
extern void crispasr_params_set_detect_language(void* p, int v);
extern void crispasr_params_set_token_timestamps(void* p, int v);
extern void crispasr_params_set_n_threads(void* p, int n);
extern void crispasr_params_set_max_len(void* p, int n);
extern void crispasr_params_set_best_of(void* p, int n);
extern void crispasr_params_set_split_on_word(void* p, int v);
extern void crispasr_params_set_no_context(void* p, int v);
extern void crispasr_params_set_single_segment(void* p, int v);
extern void crispasr_params_set_print_realtime(void* p, int v);
extern void crispasr_params_set_print_progress(void* p, int v);
extern void crispasr_params_set_print_timestamps(void* p, int v);
extern void crispasr_params_set_print_special(void* p, int v);
extern void crispasr_params_set_suppress_blank(void* p, int v);
extern void crispasr_params_set_temperature(void* p, float t);
extern void crispasr_params_set_max_tokens(void* p, int n);
extern void crispasr_params_set_initial_prompt(void* p, const char* prompt);
extern void crispasr_params_set_alt_n(void* p, int n);
extern void crispasr_params_set_vad(void* p, int v);
extern void crispasr_params_set_vad_model_path(void* p, const char* path);
extern void crispasr_params_set_vad_threshold(void* p, float t);
extern void crispasr_params_set_vad_min_speech_ms(void* p, int ms);
extern void crispasr_params_set_vad_min_silence_ms(void* p, int ms);
extern void crispasr_params_set_tdrz(void* p, int v);

// Token-level accessors
extern int64_t crispasr_token_t0(void* ctx, int i_seg, int i_tok);
extern int64_t crispasr_token_t1(void* ctx, int i_seg, int i_tok);
extern float   crispasr_token_p(void* ctx, int i_seg, int i_tok);
extern int     crispasr_token_n_alts(void* ctx, int i_seg, int i_tok);
extern int32_t crispasr_token_alt_id(void* ctx, int i_seg, int i_tok, int i_alt);
extern float   crispasr_token_alt_p(void* ctx, int i_seg, int i_tok, int i_alt);
extern int     crispasr_token_alt_text(void* ctx, int i_seg, int i_tok, int i_alt, char* out, int out_cap);

// Language detection (whisper context)
extern float crispasr_detect_language(void* ctx, const float* pcm, int n_samples,
                                      int n_threads, char* out_code, int out_cap);

// VAD slices
extern int crispasr_vad_slices(const char* vad_model_path, const float* pcm, int n_samples,
                               int sample_rate, float threshold, int min_speech_ms, int min_silence_ms,
                               int speech_pad_ms, float max_chunk_duration_s, int n_threads,
                               float** out_spans);

// LCS dedup
extern int crispasr_lcs_dedup_prefix_count(const int32_t* prev_tail_tokens, int n_prev,
                                           const int32_t* curr_tokens, int n_curr, int min_lcs_length);

// Direct Parakeet API
extern void* crispasr_parakeet_init(const char* model_path, int n_threads, int use_flash);
extern void  crispasr_parakeet_free(void* ctx);
extern void* crispasr_parakeet_transcribe(void* ctx, const float* pcm, int n_samples, const char* language);
extern const char* crispasr_parakeet_result_text(void* r);
extern int         crispasr_parakeet_result_n_words(void* r);
extern const char* crispasr_parakeet_result_word_text(void* r, int i);
extern int64_t     crispasr_parakeet_result_word_t0(void* r, int i);
extern int64_t     crispasr_parakeet_result_word_t1(void* r, int i);
extern int         crispasr_parakeet_result_n_tokens(void* r);
extern const char* crispasr_parakeet_result_token_text(void* r, int i);
extern int64_t     crispasr_parakeet_result_token_t0(void* r, int i);
extern int64_t     crispasr_parakeet_result_token_t1(void* r, int i);
extern float       crispasr_parakeet_result_token_p(void* r, int i);
extern void        crispasr_parakeet_result_free(void* r);

// TitaNet
extern void* crispasr_titanet_init(const char* model_path, int32_t n_threads);
extern void  crispasr_titanet_free(void* ctx);
extern int32_t crispasr_titanet_embed(void* ctx, const float* pcm_16k, int32_t n_samples, float* out);
extern float   crispasr_titanet_cosine_sim(const float* a, const float* b, int32_t dim);

// Speaker database
extern void* crispasr_speaker_db_load(const char* dir_path);
extern void  crispasr_speaker_db_free(void* db);
extern int32_t crispasr_speaker_db_count(const void* db);
extern float   crispasr_speaker_db_match(const void* db, const float* embedding, int32_t dim,
                                         float threshold, char* out_name, int32_t out_cap);
extern int32_t crispasr_speaker_db_enroll(const char* dir_path, const char* name,
                                          const float* embedding, int32_t dim);

// Pluggable speaker embedder + clustering + pyannote cache
extern void*   crispasr_speaker_embedder_make_abi(const char* model_spec, int32_t n_threads, const char* cache_dir);
extern void    crispasr_speaker_embedder_free_abi(void* embedder);
extern int32_t crispasr_speaker_embedder_dim_abi(const void* embedder);
extern int32_t crispasr_speaker_embedder_embed_abi(void* embedder, const float* pcm_16k, int32_t n_samples, float* out);
extern const char* crispasr_speaker_embedder_name_abi(const void* embedder);
extern int32_t crispasr_speaker_cluster_abi(const float* embeddings, int32_t n, int32_t dim,
                                            float merge_threshold, int32_t max_speakers, int32_t* labels_out);
extern void*   crispasr_pyannote_cache_compute_abi(const float* full_audio, int32_t n_samples,
                                                   const char* model_path, int32_t n_threads);
extern void    crispasr_pyannote_cache_free_abi(void* cache);
extern int32_t crispasr_pyannote_cache_apply_abi(const void* cache, int64_t slice_t0_cs,
                                                 void* segs, int32_t n_segs);

// Kokoro lang helpers
extern int  crispasr_kokoro_lang_is_german_abi(const char* lang);
extern int  crispasr_kokoro_lang_has_native_voice_abi(const char* lang);

// Backend detection
extern int crispasr_detect_backend_from_gguf(const char* path, char* out_name, int out_cap);
extern int crispasr_detect_language_pcm(const float* samples, int32_t n_samples, int32_t method,
                                        const char* model_path, int32_t n_threads, int32_t use_gpu,
                                        int32_t gpu_device, int32_t flash_attn,
                                        char* out_lang, int32_t out_lang_cap, float* out_confidence);

// RNNoise audio enhancement
extern int crispasr_enhance_audio_rnnoise(const float* in_pcm, int32_t n_samples,
                                          float* out_pcm, int32_t out_cap);

// Text-LID
extern int crispasr_text_detect_language(const char* text, const char* model_path, int32_t n_threads,
                                         char* out_label, int32_t out_label_cap, float* out_confidence);

// Registry extras
extern int crispasr_registry_lookup_abi(const char* backend, char* out_filename, int32_t filename_cap,
                                        char* out_url, int32_t url_cap, char* out_size, int32_t size_cap);
extern int crispasr_registry_lookup_by_filename_abi(const char* filename, char* out_filename, int32_t filename_cap,
                                                    char* out_url, int32_t url_cap, char* out_size, int32_t size_cap);
extern int crispasr_registry_list_backends_abi(char* out_csv, int32_t out_cap);
extern int crispasr_cache_ensure_file_abi(const char* filename, const char* url, int32_t quiet,
                                          const char* cache_dir_override, char* out_buf, int32_t out_cap);
extern int crispasr_cache_dir_abi(const char* cache_dir_override, char* out_buf, int32_t out_cap);

// Session extras
extern int crispasr_session_available_backends(char* out_csv, int out_cap);
extern struct CrispasrSession* crispasr_session_open_explicit(const char* model_path, const char* backend_name, int n_threads);
extern struct CrispasrSession* crispasr_session_open_with_params(const char* model_path, const char* backend_name, const void* params);
extern struct crispasr_session_result* crispasr_session_transcribe_vad(struct CrispasrSession* s, const float* pcm, int n_samples,
                                                                       int sample_rate, const char* vad_model_path, void* opts);
extern struct crispasr_session_result* crispasr_session_transcribe_vad_lang(struct CrispasrSession* s, const float* pcm, int n_samples,
                                                                            int sample_rate, const char* vad_model_path, void* opts,
                                                                            const char* language);
extern char* crispasr_session_translate_text(struct CrispasrSession* s, const char* text, const char* src_lang,
                                             const char* tgt_lang, int max_tokens);
extern void  crispasr_session_translate_text_free(char* text);
extern int   crispasr_session_result_word_n_alts(struct crispasr_session_result* r, int i_seg, int i_word);
extern const char* crispasr_session_result_word_alt_text(struct crispasr_session_result* r, int i_seg, int i_word, int i_alt);
extern float crispasr_session_result_word_alt_p(struct crispasr_session_result* r, int i_seg, int i_word, int i_alt);

// Streaming (whisper context)
extern struct CrispasrStream* crispasr_stream_open(void* ctx, int n_threads, int step_ms,
                                                    int length_ms, int keep_ms, const char* language, int translate);

// Diarization
extern int crispasr_diarize_segments_abi(const float* left_pcm, const float* right_pcm, int32_t n_samples,
                                         int32_t is_stereo, void* segs, int32_t n_segs, const void* opts);

static VALUE mCrispASR;
static VALUE mSession;
static VALUE mStream;
static VALUE mMic;

static VALUE rb_session_open(VALUE self, VALUE model_path, VALUE n_threads) {
    struct CrispasrSession* s =
        crispasr_session_open(StringValueCStr(model_path), NUM2INT(n_threads));
    if (!s) rb_raise(rb_eRuntimeError, "crispasr_session_open: failed to open %s",
                     StringValueCStr(model_path));
    return ULL2NUM((uintptr_t)s);
}

static VALUE rb_session_close(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    if (s) crispasr_session_close(s);
    return Qnil;
}

static VALUE rb_session_set_codec_path(VALUE self, VALUE handle, VALUE path) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_codec_path(s, StringValueCStr(path));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_codec_path failed (rc=%d)", rc);
    return Qnil;
}

// Drop the kokoro per-session phoneme cache. (PLAN #56 #5)
static VALUE rb_session_clear_phoneme_cache(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_kokoro_clear_phoneme_cache(s);
    if (rc != 0) rb_raise(rb_eRuntimeError, "clear_phoneme_cache failed (rc=%d)", rc);
    return Qnil;
}

// ---- Sticky session-state setters (PLAN #59 partial unblock) ----

static VALUE rb_session_set_source_language(VALUE self, VALUE handle, VALUE lang) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_source_language(s, NIL_P(lang) ? "" : StringValueCStr(lang));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_source_language failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_target_language(VALUE self, VALUE handle, VALUE lang) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_target_language(s, NIL_P(lang) ? "" : StringValueCStr(lang));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_target_language failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_punctuation(VALUE self, VALUE handle, VALUE enable) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_punctuation(s, RTEST(enable) ? 1 : 0);
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_punctuation failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_translate(VALUE self, VALUE handle, VALUE enable) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_translate(s, RTEST(enable) ? 1 : 0);
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_translate failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_temperature(VALUE self, VALUE handle, VALUE temperature, VALUE seed) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_temperature(s, (float)NUM2DBL(temperature),
                                              (unsigned long long)NUM2ULL(seed));
    // rc == -2 = no backend supports it; soft no-op.
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_temperature failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_tts_seed(VALUE self, VALUE handle, VALUE seed) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_tts_seed(s, (unsigned long long)NUM2ULL(seed));
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_tts_seed failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_max_new_tokens(VALUE self, VALUE handle, VALUE n) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_max_new_tokens(s, NUM2INT(n));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_max_new_tokens failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_frequency_penalty(VALUE self, VALUE handle, VALUE penalty) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_frequency_penalty(s, (float)NUM2DBL(penalty));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_frequency_penalty failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_tts_steps(VALUE self, VALUE handle, VALUE steps) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_tts_steps(s, NUM2INT(steps));
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_tts_steps failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_top_p(VALUE self, VALUE handle, VALUE top_p) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_top_p(s, (float)NUM2DBL(top_p));
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_top_p failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_min_p(VALUE self, VALUE handle, VALUE min_p) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_min_p(s, (float)NUM2DBL(min_p));
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_min_p failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_repetition_penalty(VALUE self, VALUE handle, VALUE r) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_repetition_penalty(s, (float)NUM2DBL(r));
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_repetition_penalty failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_cfg_weight(VALUE self, VALUE handle, VALUE cfg) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_cfg_weight(s, (float)NUM2DBL(cfg));
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_cfg_weight failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_exaggeration(VALUE self, VALUE handle, VALUE exag) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_exaggeration(s, (float)NUM2DBL(exag));
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_exaggeration failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_max_speech_tokens(VALUE self, VALUE handle, VALUE n) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_max_speech_tokens(s, NUM2INT(n));
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_max_speech_tokens failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_length_scale(VALUE self, VALUE handle, VALUE scale) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_length_scale(s, (float)NUM2DBL(scale));
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_length_scale failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_best_of(VALUE self, VALUE handle, VALUE n) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_best_of(s, NUM2INT(n));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_best_of failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_beam_size(VALUE self, VALUE handle, VALUE n) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_beam_size(s, NUM2INT(n));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_beam_size failed (rc=%d)", rc);
    return Qnil;
}

// set_grammar_text(handle, gbnf_text, root_rule, penalty)
// Pass nil or "" for gbnf_text to clear the grammar.
static VALUE rb_session_set_grammar_text(VALUE self, VALUE handle, VALUE gbnf, VALUE root, VALUE penalty) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    const char* gbnf_str  = NIL_P(gbnf)  ? NULL : StringValueCStr(gbnf);
    const char* root_str  = NIL_P(root)  ? NULL : StringValueCStr(root);
    int rc = crispasr_session_set_grammar_text(s, gbnf_str, root_str, (float)NUM2DBL(penalty));
    if (rc == -2) rb_raise(rb_eArgError, "set_grammar_text: invalid GBNF or root rule not found");
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_grammar_text failed (rc=%d)", rc);
    return Qnil;
}

// set_fallback_thresholds(handle, entropy_thold, logprob_thold, no_speech_thold, temperature_inc)
static VALUE rb_session_set_fallback_thresholds(VALUE self, VALUE handle,
                                                VALUE entropy, VALUE logprob,
                                                VALUE no_speech, VALUE temp_inc) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_fallback_thresholds(s,
        (float)NUM2DBL(entropy), (float)NUM2DBL(logprob),
        (float)NUM2DBL(no_speech), (float)NUM2DBL(temp_inc));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_fallback_thresholds failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_alt_n(VALUE self, VALUE handle, VALUE n) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_alt_n(s, NUM2INT(n));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_alt_n failed (rc=%d)", rc);
    return Qnil;
}

// set_whisper_decode_extras(handle, suppress_nst, suppress_regex, carry_initial_prompt)
static VALUE rb_session_set_whisper_decode_extras(VALUE self, VALUE handle,
                                                  VALUE suppress_nst, VALUE suppress_regex,
                                                  VALUE carry_prompt) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    const char* regex = NIL_P(suppress_regex) ? "" : StringValueCStr(suppress_regex);
    int rc = crispasr_session_set_whisper_decode_extras(s,
        RTEST(suppress_nst) ? 1 : 0, regex, RTEST(carry_prompt) ? 1 : 0);
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_whisper_decode_extras failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_ask(VALUE self, VALUE handle, VALUE prompt) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_ask(s, StringValueCStr(prompt));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_ask failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_detect_language(VALUE self, VALUE handle, VALUE pcm_arr, VALUE lid_path, VALUE method) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    Check_Type(pcm_arr, T_ARRAY);
    long n = RARRAY_LEN(pcm_arr);
    float* pcm = (float*)malloc(sizeof(float) * (size_t)n);
    if (!pcm) rb_raise(rb_eNoMemError, "alloc failed");
    for (long i = 0; i < n; i++) pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    char out_lang[16] = {0};
    float prob = 0.0f;
    int rc = crispasr_session_detect_language(s, pcm, (int)n, StringValueCStr(lid_path),
                                              NUM2INT(method), out_lang, sizeof(out_lang), &prob);
    free(pcm);
    if (rc != 0) rb_raise(rb_eRuntimeError, "detect_language failed (rc=%d)", rc);
    VALUE pair = rb_ary_new_capa(2);
    rb_ary_push(pair, rb_str_new_cstr(out_lang));
    rb_ary_push(pair, DBL2NUM((double)prob));
    return pair;
}

static VALUE rb_session_set_voice(int argc, VALUE* argv, VALUE self) {
    VALUE handle, path, ref_text;
    rb_scan_args(argc, argv, "21", &handle, &path, &ref_text);
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    const char* rt = NIL_P(ref_text) ? NULL : StringValueCStr(ref_text);
    int rc = crispasr_session_set_voice(s, StringValueCStr(path), rt);
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_voice failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_speaker_name(VALUE self, VALUE handle, VALUE name) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_speaker_name(s, StringValueCStr(name));
    if (rc == -2) rb_raise(rb_eArgError, "unknown speaker: %s", StringValueCStr(name));
    if (rc == -3) rb_raise(rb_eRuntimeError, "backend has no preset speakers; use set_voice instead");
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_speaker_name failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_speakers(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int n = crispasr_session_n_speakers(s);
    VALUE arr = rb_ary_new_capa(n);
    for (int i = 0; i < n; i++) {
        const char* name = crispasr_session_get_speaker_name(s, i);
        rb_ary_push(arr, name ? rb_str_new_cstr(name) : rb_str_new_cstr(""));
    }
    return arr;
}

static VALUE rb_session_set_instruct(VALUE self, VALUE handle, VALUE instruct) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_instruct(s, StringValueCStr(instruct));
    if (rc == -3) rb_raise(rb_eRuntimeError,
            "backend is not a VoiceDesign variant; set_instruct only applies to qwen3-tts VoiceDesign models");
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_instruct failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_is_custom_voice(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    return crispasr_session_is_custom_voice(s) ? Qtrue : Qfalse;
}

static VALUE rb_session_is_voice_design(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    return crispasr_session_is_voice_design(s) ? Qtrue : Qfalse;
}

static VALUE rb_session_synthesize(VALUE self, VALUE handle, VALUE text) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int n = 0;
    float* pcm = crispasr_session_synthesize(s, StringValueCStr(text), &n);
    if (!pcm || n <= 0) {
        if (pcm) crispasr_pcm_free(pcm);
        rb_raise(rb_eRuntimeError, "synthesize returned no audio");
    }
    VALUE arr = rb_ary_new_capa(n);
    for (int i = 0; i < n; i++) rb_ary_push(arr, DBL2NUM((double)pcm[i]));
    crispasr_pcm_free(pcm);
    return arr;
}

// --- ASR transcription (PLAN #59) ---
// CrispASR::Session.transcribe(handle, pcm_array) -> [{text:, t0:, t1:, words: [{text:, t0:, t1:, p:}]}]
static VALUE rb_session_transcribe(VALUE self, VALUE handle, VALUE pcm_arr) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    long n = RARRAY_LEN(pcm_arr);
    float* pcm = (float*)malloc(sizeof(float) * (size_t)n);
    for (long i = 0; i < n; i++)
        pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));

    struct crispasr_session_result* r = crispasr_session_transcribe(s, pcm, (int)n);
    free(pcm);
    if (!r) rb_raise(rb_eRuntimeError, "transcription failed");

    int n_segs = crispasr_session_result_n_segments(r);
    VALUE segments = rb_ary_new_capa(n_segs);
    for (int i = 0; i < n_segs; i++) {
        VALUE seg = rb_hash_new();
        const char* text = crispasr_session_result_segment_text(r, i);
        rb_hash_aset(seg, ID2SYM(rb_intern("text")), rb_utf8_str_new_cstr(text ? text : ""));
        rb_hash_aset(seg, ID2SYM(rb_intern("t0")), LL2NUM(crispasr_session_result_segment_t0(r, i)));
        rb_hash_aset(seg, ID2SYM(rb_intern("t1")), LL2NUM(crispasr_session_result_segment_t1(r, i)));

        int n_words = crispasr_session_result_n_words(r, i);
        VALUE words = rb_ary_new_capa(n_words);
        for (int j = 0; j < n_words; j++) {
            VALUE w = rb_hash_new();
            const char* wt = crispasr_session_result_word_text(r, i, j);
            rb_hash_aset(w, ID2SYM(rb_intern("text")), rb_utf8_str_new_cstr(wt ? wt : ""));
            rb_hash_aset(w, ID2SYM(rb_intern("t0")), LL2NUM(crispasr_session_result_word_t0(r, i, j)));
            rb_hash_aset(w, ID2SYM(rb_intern("t1")), LL2NUM(crispasr_session_result_word_t1(r, i, j)));
            rb_hash_aset(w, ID2SYM(rb_intern("p")), DBL2NUM((double)crispasr_session_result_word_p(r, i, j)));
            rb_ary_push(words, w);
        }
        rb_hash_aset(seg, ID2SYM(rb_intern("words")), words);
        rb_ary_push(segments, seg);
    }
    crispasr_session_result_free(r);
    return segments;
}

// CrispASR::Session.vad_segments(vad_model_path, pcm, sample_rate, threshold, min_speech_ms, min_silence_ms, n_threads)
// -> [{t0:, t1:}]
static VALUE rb_session_vad_segments(int argc, VALUE* argv, VALUE self) {
    if (argc < 3) rb_raise(rb_eArgError, "vad_segments needs at least 3 args: vad_model_path, pcm, sample_rate");
    const char* vad_path = StringValueCStr(argv[0]);
    VALUE pcm_arr = argv[1];
    int sr = NUM2INT(argv[2]);
    float threshold = argc > 3 ? (float)NUM2DBL(argv[3]) : 0.5f;
    int min_speech = argc > 4 ? NUM2INT(argv[4]) : 250;
    int min_silence = argc > 5 ? NUM2INT(argv[5]) : 100;
    int n_threads = argc > 6 ? NUM2INT(argv[6]) : 4;

    long n = RARRAY_LEN(pcm_arr);
    float* pcm = (float*)malloc(sizeof(float) * (size_t)n);
    for (long i = 0; i < n; i++) pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));

    float* spans = NULL;
    int n_segs = crispasr_vad_segments(vad_path, pcm, (int)n, sr, threshold, min_speech, min_silence, n_threads, 0, &spans);
    free(pcm);
    if (n_segs < 0) rb_raise(rb_eRuntimeError, "VAD failed (rc=%d)", n_segs);
    VALUE result = rb_ary_new_capa(n_segs);
    for (int i = 0; i < n_segs; i++) {
        VALUE seg = rb_hash_new();
        rb_hash_aset(seg, ID2SYM(rb_intern("t0")), DBL2NUM((double)spans[i * 2]));
        rb_hash_aset(seg, ID2SYM(rb_intern("t1")), DBL2NUM((double)spans[i * 2 + 1]));
        rb_ary_push(result, seg);
    }
    if (spans) crispasr_vad_free(spans);
    return result;
}

// CrispASR::Session.align_words(aligner_model, transcript, pcm, n_threads) -> [{text:, t0:, t1:}]
static VALUE rb_session_align_words(VALUE self, VALUE aligner_model, VALUE transcript, VALUE pcm_arr, VALUE n_threads_v) {
    const char* model = StringValueCStr(aligner_model);
    const char* text = StringValueCStr(transcript);
    int n_threads = NUM2INT(n_threads_v);
    long n = RARRAY_LEN(pcm_arr);
    float* pcm = (float*)malloc(sizeof(float) * (size_t)n);
    for (long i = 0; i < n; i++) pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));

    struct crispasr_align_result* r = crispasr_align_words_abi(model, text, pcm, (int)n, 0, n_threads);
    free(pcm);
    if (!r) rb_raise(rb_eRuntimeError, "alignment failed");

    int nw = crispasr_align_result_n_words(r);
    VALUE result = rb_ary_new_capa(nw);
    for (int i = 0; i < nw; i++) {
        VALUE w = rb_hash_new();
        const char* wt = crispasr_align_result_word_text(r, i);
        rb_hash_aset(w, ID2SYM(rb_intern("text")), rb_utf8_str_new_cstr(wt ? wt : ""));
        rb_hash_aset(w, ID2SYM(rb_intern("t0")), LL2NUM(crispasr_align_result_word_t0(r, i)));
        rb_hash_aset(w, ID2SYM(rb_intern("t1")), LL2NUM(crispasr_align_result_word_t1(r, i)));
        rb_ary_push(result, w);
    }
    crispasr_align_result_free(r);
    return result;
}

static VALUE rb_kokoro_resolve_for_lang(VALUE self, VALUE model_path, VALUE lang) {
    char out_model[1024]  = {0};
    char out_voice[1024]  = {0};
    char out_picked[64]   = {0};

    const char* mp = StringValueCStr(model_path);
    const char* lg = NIL_P(lang) ? "" : StringValueCStr(lang);

    int rc = crispasr_kokoro_resolve_model_for_lang_abi(mp, lg, out_model, sizeof(out_model));
    if (rc < 0) rb_raise(rb_eRuntimeError, "kokoro_resolve_model_for_lang: buffer too small");
    int swapped = (rc == 0);
    const char* resolved = (out_model[0] != 0) ? out_model : mp;

    rc = crispasr_kokoro_resolve_fallback_voice_abi(mp, lg,
                                                    out_voice, sizeof(out_voice),
                                                    out_picked, sizeof(out_picked));
    if (rc < 0) rb_raise(rb_eRuntimeError, "kokoro_resolve_fallback_voice: buffer too small");

    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("model_path")),       rb_str_new_cstr(resolved));
    rb_hash_aset(h, ID2SYM(rb_intern("voice_path")),       rc == 0 ? rb_str_new_cstr(out_voice) : Qnil);
    rb_hash_aset(h, ID2SYM(rb_intern("voice_name")),       rc == 0 ? rb_str_new_cstr(out_picked) : Qnil);
    rb_hash_aset(h, ID2SYM(rb_intern("backbone_swapped")), swapped ? Qtrue : Qfalse);
    return h;
}

// =====================================================================
// Streaming (PLAN #62b) — rolling-window decoder. Whisper-only at the C-ABI today.
// =====================================================================

static VALUE rb_stream_open(VALUE self, VALUE session_h, VALUE step_ms, VALUE length_ms,
                            VALUE keep_ms, VALUE language, VALUE translate) {
    struct CrispasrSession* sess = (struct CrispasrSession*)NUM2ULL(session_h);
    const char* lang = NIL_P(language) ? "" : StringValueCStr(language);
    struct CrispasrStream* st = crispasr_session_stream_open(
        sess, 4, NUM2INT(step_ms), NUM2INT(length_ms), NUM2INT(keep_ms),
        lang, RTEST(translate) ? 1 : 0);
    if (!st) {
        rb_raise(rb_eRuntimeError, "crispasr_session_stream_open failed (whisper-only today)");
    }
    return ULL2NUM((uintptr_t)st);
}

static VALUE rb_stream_feed(VALUE self, VALUE handle, VALUE pcm_arr) {
    struct CrispasrStream* st = (struct CrispasrStream*)NUM2ULL(handle);
    Check_Type(pcm_arr, T_ARRAY);
    long n = RARRAY_LEN(pcm_arr);
    if (n == 0) return INT2NUM(0);
    float* pcm = (float*)malloc(sizeof(float) * (size_t)n);
    if (!pcm) rb_raise(rb_eNoMemError, "alloc failed");
    for (long i = 0; i < n; i++) pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    int rc = crispasr_stream_feed(st, pcm, (int)n);
    free(pcm);
    if (rc < 0) rb_raise(rb_eRuntimeError, "crispasr_stream_feed failed (rc=%d)", rc);
    return INT2NUM(rc);
}

static VALUE rb_stream_get_text(VALUE self, VALUE handle) {
    struct CrispasrStream* st = (struct CrispasrStream*)NUM2ULL(handle);
    char buf[8192];
    buf[0] = '\0';
    double t0 = 0.0, t1 = 0.0;
    long long counter = 0;
    int rc = crispasr_stream_get_text(st, buf, (int)sizeof(buf), &t0, &t1, &counter);
    if (rc < 0) rb_raise(rb_eRuntimeError, "crispasr_stream_get_text failed (rc=%d)", rc);
    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("text")),    rb_utf8_str_new_cstr(buf));
    rb_hash_aset(h, ID2SYM(rb_intern("t0")),      DBL2NUM(t0));
    rb_hash_aset(h, ID2SYM(rb_intern("t1")),      DBL2NUM(t1));
    rb_hash_aset(h, ID2SYM(rb_intern("counter")), LL2NUM(counter));
    return h;
}

static VALUE rb_stream_flush(VALUE self, VALUE handle) {
    struct CrispasrStream* st = (struct CrispasrStream*)NUM2ULL(handle);
    int rc = crispasr_stream_flush(st);
    if (rc < 0) rb_raise(rb_eRuntimeError, "crispasr_stream_flush failed (rc=%d)", rc);
    return INT2NUM(rc);
}

static VALUE rb_stream_close(VALUE self, VALUE handle) {
    struct CrispasrStream* st = (struct CrispasrStream*)NUM2ULL(handle);
    if (st) crispasr_stream_close(st);
    return Qnil;
}

// =====================================================================
// Microphone (PLAN #62d). The audio thread is owned by miniaudio — it
// has no Ruby context and no GVL. We can't call into MRI from there.
//
// Instead: the native callback pushes copies into a bounded ring
// buffer guarded by pthread mu+cv. A dedicated Ruby pump thread
// (created via `rb_thread_create`) loops:
//   1. release the GVL via `rb_thread_call_without_gvl`
//   2. wait on the cv until a buffer is ready or the handle is closed
//   3. re-acquire the GVL and dispatch to the user's `Proc` with
//      the audio as a fresh `Array<Float>`.
//
// Closing flips `closed=1`, broadcasts the cv, joins the pump, and
// drains anything left in the ring.
// =====================================================================

#define MIC_RING_CAP 32

typedef struct mic_handle {
    pthread_mutex_t       mu;
    pthread_cond_t        cv;
    float*                bufs[MIC_RING_CAP];
    int                   lens[MIC_RING_CAP];
    int                   head;
    int                   tail;
    volatile int          closed;
    VALUE                 user_proc;
    VALUE                 pump_thread;
    struct crispasr_mic*  mic;
} mic_handle_t;

static void mic_native_cb(const float* pcm, int n, void* ud) {
    mic_handle_t* h = (mic_handle_t*)ud;
    if (n <= 0 || !pcm) return;
    pthread_mutex_lock(&h->mu);
    if (!h->closed) {
        int next = (h->tail + 1) % MIC_RING_CAP;
        if (next == h->head) {
            // Ring full: drop oldest. Better to lose a chunk than to
            // block the audio thread.
            free(h->bufs[h->head]);
            h->bufs[h->head] = NULL;
            h->head = (h->head + 1) % MIC_RING_CAP;
        }
        float* copy = (float*)malloc(sizeof(float) * (size_t)n);
        if (copy) {
            memcpy(copy, pcm, sizeof(float) * (size_t)n);
            h->bufs[h->tail] = copy;
            h->lens[h->tail] = n;
            h->tail = next;
        }
        pthread_cond_signal(&h->cv);
    }
    pthread_mutex_unlock(&h->mu);
}

struct mic_dequeue {
    mic_handle_t* h;
    float*        buf;  // out
    int           n;    // out
    int           closed; // out: 1 if woken because handle closed with no data
};

static void* mic_wait_blocking(void* p) {
    struct mic_dequeue* d = (struct mic_dequeue*)p;
    pthread_mutex_lock(&d->h->mu);
    while (d->h->head == d->h->tail && !d->h->closed) {
        pthread_cond_wait(&d->h->cv, &d->h->mu);
    }
    if (d->h->head != d->h->tail) {
        d->buf = d->h->bufs[d->h->head];
        d->n   = d->h->lens[d->h->head];
        d->h->bufs[d->h->head] = NULL;
        d->h->head = (d->h->head + 1) % MIC_RING_CAP;
    } else {
        d->closed = 1;
    }
    pthread_mutex_unlock(&d->h->mu);
    return NULL;
}

static void mic_wait_unblock(void* p) {
    mic_handle_t* h = (mic_handle_t*)p;
    pthread_mutex_lock(&h->mu);
    h->closed = 1;
    pthread_cond_broadcast(&h->cv);
    pthread_mutex_unlock(&h->mu);
}

struct mic_call_args { VALUE proc; VALUE arg; };
static VALUE mic_call_proc(VALUE p) {
    struct mic_call_args* a = (struct mic_call_args*)p;
    return rb_funcall(a->proc, rb_intern("call"), 1, a->arg);
}

static VALUE mic_pump_body(void* p) {
    mic_handle_t* h = (mic_handle_t*)p;
    while (!h->closed) {
        struct mic_dequeue d = { h, NULL, 0, 0 };
        rb_thread_call_without_gvl(mic_wait_blocking, &d, mic_wait_unblock, h);
        if (d.closed && !d.buf) break;
        if (d.buf && d.n > 0) {
            VALUE arr = rb_ary_new_capa(d.n);
            for (int i = 0; i < d.n; i++) rb_ary_push(arr, DBL2NUM((double)d.buf[i]));
            free(d.buf);
            int state = 0;
            struct mic_call_args ca = { h->user_proc, arr };
            rb_protect(mic_call_proc, (VALUE)&ca, &state);
            // Swallow user-side exceptions — the audio thread can't
            // be allowed to die because the consumer raised. The user
            // can wrap their block with their own error handling.
            (void)state;
        }
    }
    return Qnil;
}

static VALUE rb_mic_open(int argc, VALUE* argv, VALUE self) {
    VALUE sample_rate, channels, blk;
    rb_scan_args(argc, argv, "20&", &sample_rate, &channels, &blk);
    if (NIL_P(blk)) rb_raise(rb_eArgError, "CrispASR::Mic.open requires a block");

    mic_handle_t* h = (mic_handle_t*)calloc(1, sizeof(mic_handle_t));
    if (!h) rb_raise(rb_eNoMemError, "mic_handle_t alloc failed");
    pthread_mutex_init(&h->mu, NULL);
    pthread_cond_init(&h->cv, NULL);
    h->user_proc = blk;
    rb_gc_register_address(&h->user_proc);

    h->mic = crispasr_mic_open(NUM2INT(sample_rate), NUM2INT(channels), mic_native_cb, h);
    if (!h->mic) {
        rb_gc_unregister_address(&h->user_proc);
        pthread_mutex_destroy(&h->mu);
        pthread_cond_destroy(&h->cv);
        free(h);
        rb_raise(rb_eRuntimeError, "crispasr_mic_open failed");
    }

    // Cast to silence prototype-mismatch warnings on older Ruby
    // headers that still typedef rb_thread_create's fn as `(ANYARGS)`.
    h->pump_thread = rb_thread_create((VALUE (*)(void *))mic_pump_body, h);
    rb_gc_register_address(&h->pump_thread);

    return ULL2NUM((uintptr_t)h);
}

static VALUE rb_mic_start(VALUE self, VALUE handle) {
    mic_handle_t* h = (mic_handle_t*)NUM2ULL(handle);
    if (!h || !h->mic) rb_raise(rb_eRuntimeError, "mic is closed");
    int rc = crispasr_mic_start(h->mic);
    if (rc != 0) rb_raise(rb_eRuntimeError, "crispasr_mic_start failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_mic_stop(VALUE self, VALUE handle) {
    mic_handle_t* h = (mic_handle_t*)NUM2ULL(handle);
    if (!h || !h->mic) return Qnil;
    crispasr_mic_stop(h->mic);
    return Qnil;
}

static VALUE rb_mic_close(VALUE self, VALUE handle) {
    mic_handle_t* h = (mic_handle_t*)NUM2ULL(handle);
    if (!h) return Qnil;

    // 1. Stop + close the audio thread first so the native callback
    //    can never fire again. This must happen before we tear down
    //    the queue or the callback would touch freed memory.
    if (h->mic) {
        crispasr_mic_close(h->mic);
        h->mic = NULL;
    }

    // 2. Wake the pump and join it.
    pthread_mutex_lock(&h->mu);
    h->closed = 1;
    pthread_cond_broadcast(&h->cv);
    pthread_mutex_unlock(&h->mu);
    if (h->pump_thread != Qnil && h->pump_thread != 0) {
        rb_funcall(h->pump_thread, rb_intern("join"), 0);
        rb_gc_unregister_address(&h->pump_thread);
    }

    // 3. Drain anything the audio thread enqueued before close.
    while (h->head != h->tail) {
        free(h->bufs[h->head]);
        h->bufs[h->head] = NULL;
        h->head = (h->head + 1) % MIC_RING_CAP;
    }

    rb_gc_unregister_address(&h->user_proc);
    pthread_mutex_destroy(&h->mu);
    pthread_cond_destroy(&h->cv);
    free(h);
    return Qnil;
}

static VALUE rb_mic_default_device_name(VALUE self) {
    const char* s = crispasr_mic_default_device_name();
    return rb_utf8_str_new_cstr(s ? s : "");
}

// ---------------------------------------------------------------------------
// Full C-ABI parity wrappers (PLAN #59 bindings-parity milestone)
// ---------------------------------------------------------------------------

static VALUE rb_detect_backend_from_gguf(VALUE self, VALUE path) {
    char out[128] = {0};
    int rc = crispasr_detect_backend_from_gguf(StringValueCStr(path), out, sizeof(out));
    if (rc != 0) rb_raise(rb_eRuntimeError, "detect_backend_from_gguf failed");
    return rb_utf8_str_new_cstr(out);
}

static VALUE rb_lcs_dedup_prefix_count(VALUE self, VALUE prev_arr, VALUE curr_arr, VALUE min_len) {
    long pn = RARRAY_LEN(prev_arr);
    long cn = RARRAY_LEN(curr_arr);
    int32_t* prev = (int32_t*)calloc(pn > 0 ? pn : 1, sizeof(int32_t));
    int32_t* curr = (int32_t*)calloc(cn > 0 ? cn : 1, sizeof(int32_t));
    for (long i = 0; i < pn; i++) prev[i] = (int32_t)NUM2INT(rb_ary_entry(prev_arr, i));
    for (long i = 0; i < cn; i++) curr[i] = (int32_t)NUM2INT(rb_ary_entry(curr_arr, i));
    int rc = crispasr_lcs_dedup_prefix_count(prev, (int)pn, curr, (int)cn, NUM2INT(min_len));
    free(prev); free(curr);
    return INT2NUM(rc);
}

static VALUE rb_kokoro_lang_is_german(VALUE self, VALUE lang) {
    return crispasr_kokoro_lang_is_german_abi(StringValueCStr(lang)) ? Qtrue : Qfalse;
}

static VALUE rb_kokoro_lang_has_native_voice(VALUE self, VALUE lang) {
    return crispasr_kokoro_lang_has_native_voice_abi(StringValueCStr(lang)) ? Qtrue : Qfalse;
}

static VALUE rb_session_translate_text(VALUE self, VALUE handle, VALUE text,
                                       VALUE src_lang, VALUE tgt_lang, VALUE max_tokens) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    char* res = crispasr_session_translate_text(s, StringValueCStr(text),
                                                StringValueCStr(src_lang),
                                                StringValueCStr(tgt_lang),
                                                NUM2INT(max_tokens));
    if (!res) return Qnil;
    VALUE out = rb_utf8_str_new_cstr(res);
    crispasr_session_translate_text_free(res);
    return out;
}

static VALUE rb_session_available_backends(VALUE self) {
    char buf[1024] = {0};
    crispasr_session_available_backends(buf, sizeof(buf));
    return rb_utf8_str_new_cstr(buf);
}

static VALUE rb_session_open_explicit(VALUE self, VALUE model_path, VALUE backend, VALUE n_threads) {
    struct CrispasrSession* s = crispasr_session_open_explicit(
        StringValueCStr(model_path), StringValueCStr(backend), NUM2INT(n_threads));
    if (!s) rb_raise(rb_eRuntimeError, "crispasr_session_open_explicit failed");
    return ULL2NUM((uintptr_t)s);
}

static VALUE rb_vad_slices(int argc, VALUE* argv, VALUE self) {
    VALUE model_path, pcm_arr, opts;
    rb_scan_args(argc, argv, "21", &model_path, &pcm_arr, &opts);
    long n = RARRAY_LEN(pcm_arr);
    float* pcm = (float*)calloc(n > 0 ? n : 1, sizeof(float));
    for (long i = 0; i < n; i++) pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    float* out_spans = NULL;
    int count = crispasr_vad_slices(StringValueCStr(model_path), pcm, (int)n,
                                    16000, 0.0f, 250, 100, 30, 30.0f, 1, &out_spans);
    free(pcm);
    if (count < 0) rb_raise(rb_eRuntimeError, "crispasr_vad_slices failed");
    VALUE result = rb_ary_new_capa(count);
    for (int i = 0; i < count; i++) {
        VALUE span = rb_ary_new3(2, DBL2NUM(out_spans[2*i]), DBL2NUM(out_spans[2*i+1]));
        rb_ary_push(result, span);
    }
    if (count > 0) crispasr_vad_free(out_spans);
    return result;
}

static VALUE rb_enhance_audio_rnnoise(VALUE self, VALUE pcm_arr) {
    long n = RARRAY_LEN(pcm_arr);
    float* in = (float*)calloc(n > 0 ? n : 1, sizeof(float));
    float* out = (float*)calloc(n > 0 ? n : 1, sizeof(float));
    for (long i = 0; i < n; i++) in[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    int rc = crispasr_enhance_audio_rnnoise(in, (int32_t)n, out, (int32_t)n);
    free(in);
    if (rc != 0) { free(out); rb_raise(rb_eRuntimeError, "enhance_audio_rnnoise failed"); }
    VALUE result = rb_ary_new_capa(n);
    for (long i = 0; i < n; i++) rb_ary_push(result, DBL2NUM(out[i]));
    free(out);
    return result;
}

static VALUE rb_text_detect_language(VALUE self, VALUE text, VALUE model_path, VALUE n_threads) {
    char label[64] = {0};
    float conf = 0.0f;
    int rc = crispasr_text_detect_language(StringValueCStr(text), StringValueCStr(model_path),
                                           NUM2INT(n_threads), label, 64, &conf);
    if (rc != 0) rb_raise(rb_eRuntimeError, "text_detect_language failed");
    VALUE result = rb_ary_new3(2, rb_utf8_str_new_cstr(label), DBL2NUM(conf));
    return result;
}

static VALUE rb_registry_list_backends(VALUE self) {
    char buf[8192] = {0};
    crispasr_registry_list_backends_abi(buf, sizeof(buf));
    return rb_utf8_str_new_cstr(buf);
}

// --- Parakeet direct API ---
static VALUE rb_parakeet_init(VALUE self, VALUE model_path, VALUE n_threads, VALUE use_flash) {
    void* h = crispasr_parakeet_init(StringValueCStr(model_path), NUM2INT(n_threads), RTEST(use_flash) ? 1 : 0);
    if (!h) rb_raise(rb_eRuntimeError, "crispasr_parakeet_init failed");
    return ULL2NUM((uintptr_t)h);
}
static VALUE rb_parakeet_free(VALUE self, VALUE handle) {
    void* h = (void*)NUM2ULL(handle);
    if (h) crispasr_parakeet_free(h);
    return Qnil;
}
static VALUE rb_parakeet_transcribe(VALUE self, VALUE handle, VALUE pcm_arr, VALUE language) {
    void* h = (void*)NUM2ULL(handle);
    long n = RARRAY_LEN(pcm_arr);
    float* pcm = (float*)calloc(n > 0 ? n : 1, sizeof(float));
    for (long i = 0; i < n; i++) pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    const char* lang = NIL_P(language) ? NULL : StringValueCStr(language);
    void* res = crispasr_parakeet_transcribe(h, pcm, (int)n, lang);
    free(pcm);
    if (!res) rb_raise(rb_eRuntimeError, "crispasr_parakeet_transcribe failed");
    // Build a hash with text, words, tokens
    VALUE hash = rb_hash_new();
    const char* txt = crispasr_parakeet_result_text(res);
    rb_hash_aset(hash, ID2SYM(rb_intern("text")), rb_utf8_str_new_cstr(txt ? txt : ""));
    int nw = crispasr_parakeet_result_n_words(res);
    VALUE words = rb_ary_new_capa(nw);
    for (int i = 0; i < nw; i++) {
        const char* wt = crispasr_parakeet_result_word_text(res, i);
        VALUE w = rb_ary_new3(3,
            rb_utf8_str_new_cstr(wt ? wt : ""),
            LL2NUM(crispasr_parakeet_result_word_t0(res, i)),
            LL2NUM(crispasr_parakeet_result_word_t1(res, i)));
        rb_ary_push(words, w);
    }
    rb_hash_aset(hash, ID2SYM(rb_intern("words")), words);
    int nt = crispasr_parakeet_result_n_tokens(res);
    VALUE tokens = rb_ary_new_capa(nt);
    for (int i = 0; i < nt; i++) {
        const char* tt = crispasr_parakeet_result_token_text(res, i);
        VALUE t = rb_ary_new3(4,
            rb_utf8_str_new_cstr(tt ? tt : ""),
            LL2NUM(crispasr_parakeet_result_token_t0(res, i)),
            LL2NUM(crispasr_parakeet_result_token_t1(res, i)),
            DBL2NUM(crispasr_parakeet_result_token_p(res, i)));
        rb_ary_push(tokens, t);
    }
    rb_hash_aset(hash, ID2SYM(rb_intern("tokens")), tokens);
    crispasr_parakeet_result_free(res);
    return hash;
}

// --- TitaNet ---
static VALUE rb_titanet_init(VALUE self, VALUE model_path, VALUE n_threads) {
    void* h = crispasr_titanet_init(StringValueCStr(model_path), NUM2INT(n_threads));
    if (!h) rb_raise(rb_eRuntimeError, "crispasr_titanet_init failed");
    return ULL2NUM((uintptr_t)h);
}
static VALUE rb_titanet_free(VALUE self, VALUE handle) {
    void* h = (void*)NUM2ULL(handle);
    if (h) crispasr_titanet_free(h);
    return Qnil;
}
static VALUE rb_titanet_embed(VALUE self, VALUE handle, VALUE pcm_arr) {
    void* h = (void*)NUM2ULL(handle);
    long n = RARRAY_LEN(pcm_arr);
    float* pcm = (float*)calloc(n > 0 ? n : 1, sizeof(float));
    for (long i = 0; i < n; i++) pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    float out[192];
    int32_t dim = crispasr_titanet_embed(h, pcm, (int32_t)n, out);
    free(pcm);
    if (dim <= 0) rb_raise(rb_eRuntimeError, "titanet_embed failed");
    VALUE result = rb_ary_new_capa(dim);
    for (int i = 0; i < dim; i++) rb_ary_push(result, DBL2NUM(out[i]));
    return result;
}

// --- Speaker database ---
static VALUE rb_speaker_db_load(VALUE self, VALUE dir_path) {
    void* h = crispasr_speaker_db_load(StringValueCStr(dir_path));
    if (!h) rb_raise(rb_eRuntimeError, "crispasr_speaker_db_load failed");
    return ULL2NUM((uintptr_t)h);
}
static VALUE rb_speaker_db_free(VALUE self, VALUE handle) {
    void* h = (void*)NUM2ULL(handle);
    if (h) crispasr_speaker_db_free(h);
    return Qnil;
}
static VALUE rb_speaker_db_count(VALUE self, VALUE handle) {
    void* h = (void*)NUM2ULL(handle);
    return INT2NUM(crispasr_speaker_db_count(h));
}
static VALUE rb_speaker_db_match(VALUE self, VALUE handle, VALUE emb_arr, VALUE threshold) {
    void* h = (void*)NUM2ULL(handle);
    long dim = RARRAY_LEN(emb_arr);
    float* emb = (float*)calloc(dim > 0 ? dim : 1, sizeof(float));
    for (long i = 0; i < dim; i++) emb[i] = (float)NUM2DBL(rb_ary_entry(emb_arr, i));
    char name[256] = {0};
    float score = crispasr_speaker_db_match(h, emb, (int32_t)dim,
                                            (float)NUM2DBL(threshold), name, 256);
    free(emb);
    VALUE result = rb_ary_new3(2,
        score >= (float)NUM2DBL(threshold) ? rb_utf8_str_new_cstr(name) : Qnil,
        DBL2NUM(score));
    return result;
}
static VALUE rb_speaker_db_enroll(VALUE self, VALUE dir_path, VALUE name, VALUE emb_arr) {
    long dim = RARRAY_LEN(emb_arr);
    float* emb = (float*)calloc(dim > 0 ? dim : 1, sizeof(float));
    for (long i = 0; i < dim; i++) emb[i] = (float)NUM2DBL(rb_ary_entry(emb_arr, i));
    int32_t rc = crispasr_speaker_db_enroll(StringValueCStr(dir_path),
                                            StringValueCStr(name), emb, (int32_t)dim);
    free(emb);
    if (rc != 0) rb_raise(rb_eRuntimeError, "speaker_db_enroll failed");
    return Qtrue;
}

// --- Session transcribe_vad_lang ---
static VALUE rb_session_transcribe_vad_lang(int argc, VALUE* argv, VALUE self) {
    VALUE handle, pcm_arr, vad_model, language;
    rb_scan_args(argc, argv, "4", &handle, &pcm_arr, &vad_model, &language);
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    long n = RARRAY_LEN(pcm_arr);
    float* pcm = (float*)calloc(n > 0 ? n : 1, sizeof(float));
    for (long i = 0; i < n; i++) pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    struct crispasr_session_result* res = crispasr_session_transcribe_vad_lang(
        s, pcm, (int)n, 16000, StringValueCStr(vad_model), NULL,
        NIL_P(language) ? NULL : StringValueCStr(language));
    free(pcm);
    if (!res) rb_raise(rb_eRuntimeError, "transcribe_vad_lang failed");
    int ns = crispasr_session_result_n_segments(res);
    VALUE segs = rb_ary_new_capa(ns);
    for (int i = 0; i < ns; i++) {
        const char* t = crispasr_session_result_segment_text(res, i);
        VALUE seg = rb_hash_new();
        rb_hash_aset(seg, ID2SYM(rb_intern("text")), rb_utf8_str_new_cstr(t ? t : ""));
        rb_hash_aset(seg, ID2SYM(rb_intern("t0")), DBL2NUM(crispasr_session_result_segment_t0(res, i) / 100.0));
        rb_hash_aset(seg, ID2SYM(rb_intern("t1")), DBL2NUM(crispasr_session_result_segment_t1(res, i) / 100.0));
        rb_ary_push(segs, seg);
    }
    crispasr_session_result_free(res);
    return segs;
}

// --- Session word alts ---
static VALUE rb_session_result_word_n_alts(VALUE self, VALUE res_handle, VALUE i_seg, VALUE i_word) {
    struct crispasr_session_result* r = (struct crispasr_session_result*)NUM2ULL(res_handle);
    return INT2NUM(crispasr_session_result_word_n_alts(r, NUM2INT(i_seg), NUM2INT(i_word)));
}
static VALUE rb_session_result_word_alt_p(VALUE self, VALUE res_handle, VALUE i_seg, VALUE i_word, VALUE i_alt) {
    struct crispasr_session_result* r = (struct crispasr_session_result*)NUM2ULL(res_handle);
    return DBL2NUM(crispasr_session_result_word_alt_p(r, NUM2INT(i_seg), NUM2INT(i_word), NUM2INT(i_alt)));
}

// --- Stream set_live_decode ---
static VALUE rb_stream_set_live_decode(VALUE self, VALUE handle, VALUE enabled) {
    struct CrispasrStream* s = (struct CrispasrStream*)NUM2ULL(handle);
    crispasr_stream_set_live_decode(s, RTEST(enabled) ? 1 : 0);
    return Qnil;
}

// --- Diarize segments ---
static VALUE rb_diarize_segments(VALUE self, VALUE left_arr, VALUE right_arr, VALUE is_stereo) {
    // Minimal wrapper — full diarize needs segment structs which are complex in Ruby.
    // This is a placeholder that validates the symbol is callable.
    return Qnil;
}

static VALUE rb_titanet_cosine_sim(VALUE self, VALUE a_arr, VALUE b_arr) {
    long na = RARRAY_LEN(a_arr);
    long nb = RARRAY_LEN(b_arr);
    long dim = na < nb ? na : nb;
    float* a = (float*)calloc(dim > 0 ? dim : 1, sizeof(float));
    float* b = (float*)calloc(dim > 0 ? dim : 1, sizeof(float));
    for (long i = 0; i < dim; i++) {
        a[i] = (float)NUM2DBL(rb_ary_entry(a_arr, i));
        b[i] = (float)NUM2DBL(rb_ary_entry(b_arr, i));
    }
    float sim = crispasr_titanet_cosine_sim(a, b, (int32_t)dim);
    free(a); free(b);
    return DBL2NUM(sim);
}

void init_ruby_crispasr_session(VALUE* mWhisper) {
    // Define module path under the existing Whisper module so existing
    // code keeps working: CrispASR::Session, but we also alias it under
    // Whisper::CrispASR::Session.
    mCrispASR = rb_define_module_under(*mWhisper, "CrispASR");
    mSession  = rb_define_module_under(mCrispASR, "Session");

    rb_define_singleton_method(mSession, "open",                 rb_session_open,             2);
    rb_define_singleton_method(mSession, "close",                rb_session_close,            1);
    rb_define_singleton_method(mSession, "set_codec_path",       rb_session_set_codec_path,   2);
    rb_define_singleton_method(mSession, "set_voice",            rb_session_set_voice,       -1);
    rb_define_singleton_method(mSession, "set_speaker_name",     rb_session_set_speaker_name, 2);
    rb_define_singleton_method(mSession, "speakers",             rb_session_speakers,         1);
    rb_define_singleton_method(mSession, "set_instruct",         rb_session_set_instruct,     2);
    rb_define_singleton_method(mSession, "is_custom_voice",      rb_session_is_custom_voice,  1);
    rb_define_singleton_method(mSession, "is_voice_design",      rb_session_is_voice_design,  1);
    rb_define_singleton_method(mSession, "synthesize",           rb_session_synthesize,       2);
    rb_define_singleton_method(mSession, "transcribe",           rb_session_transcribe,       2);
    rb_define_singleton_method(mSession, "vad_segments",         rb_session_vad_segments,    -1);
    rb_define_singleton_method(mSession, "align_words",          rb_session_align_words,      4);
    rb_define_singleton_method(mSession, "clear_phoneme_cache",  rb_session_clear_phoneme_cache, 1);
    rb_define_singleton_method(mSession, "set_source_language",  rb_session_set_source_language, 2);
    rb_define_singleton_method(mSession, "set_target_language",  rb_session_set_target_language, 2);
    rb_define_singleton_method(mSession, "set_punctuation",      rb_session_set_punctuation, 2);
    rb_define_singleton_method(mSession, "set_translate",        rb_session_set_translate, 2);
    rb_define_singleton_method(mSession, "set_temperature",           rb_session_set_temperature,           3);
    rb_define_singleton_method(mSession, "set_tts_seed",              rb_session_set_tts_seed,              2);
    rb_define_singleton_method(mSession, "set_max_new_tokens",        rb_session_set_max_new_tokens,        2);
    rb_define_singleton_method(mSession, "set_frequency_penalty",     rb_session_set_frequency_penalty,     2);
    rb_define_singleton_method(mSession, "set_tts_steps",             rb_session_set_tts_steps,             2);
    rb_define_singleton_method(mSession, "set_top_p",                 rb_session_set_top_p,                 2);
    rb_define_singleton_method(mSession, "set_min_p",                 rb_session_set_min_p,                 2);
    rb_define_singleton_method(mSession, "set_repetition_penalty",    rb_session_set_repetition_penalty,    2);
    rb_define_singleton_method(mSession, "set_cfg_weight",            rb_session_set_cfg_weight,            2);
    rb_define_singleton_method(mSession, "set_exaggeration",          rb_session_set_exaggeration,          2);
    rb_define_singleton_method(mSession, "set_max_speech_tokens",     rb_session_set_max_speech_tokens,     2);
    rb_define_singleton_method(mSession, "set_length_scale",          rb_session_set_length_scale,          2);
    rb_define_singleton_method(mSession, "set_best_of",               rb_session_set_best_of,               2);
    rb_define_singleton_method(mSession, "set_beam_size",             rb_session_set_beam_size,             2);
    rb_define_singleton_method(mSession, "set_grammar_text",          rb_session_set_grammar_text,          4);
    rb_define_singleton_method(mSession, "set_fallback_thresholds",   rb_session_set_fallback_thresholds,   5);
    rb_define_singleton_method(mSession, "set_alt_n",                 rb_session_set_alt_n,                 2);
    rb_define_singleton_method(mSession, "set_whisper_decode_extras", rb_session_set_whisper_decode_extras, 4);
    rb_define_singleton_method(mSession, "set_ask",                   rb_session_set_ask,                   2);
    rb_define_singleton_method(mSession, "detect_language",           rb_session_detect_language,           4);
    rb_define_singleton_method(mSession, "kokoro_resolve_for_lang", rb_kokoro_resolve_for_lang, 2);

    // Streaming (PLAN #62b) — CrispASR::Session::Stream.{open, feed, get_text, flush, close}.
    mStream = rb_define_module_under(mSession, "Stream");
    rb_define_singleton_method(mStream, "open",     rb_stream_open,     6);
    rb_define_singleton_method(mStream, "feed",     rb_stream_feed,     2);
    rb_define_singleton_method(mStream, "get_text", rb_stream_get_text, 1);
    rb_define_singleton_method(mStream, "flush",    rb_stream_flush,    1);
    rb_define_singleton_method(mStream, "close",    rb_stream_close,    1);
    rb_define_singleton_method(mStream, "set_live_decode", rb_stream_set_live_decode, 2);

    // Full C-ABI parity (PLAN #59).
    rb_define_singleton_method(mSession, "detect_backend_from_gguf", rb_detect_backend_from_gguf, 1);
    rb_define_singleton_method(mSession, "lcs_dedup_prefix_count", rb_lcs_dedup_prefix_count, 3);
    rb_define_singleton_method(mSession, "kokoro_lang_is_german", rb_kokoro_lang_is_german, 1);
    rb_define_singleton_method(mSession, "kokoro_lang_has_native_voice", rb_kokoro_lang_has_native_voice, 1);
    rb_define_singleton_method(mSession, "translate_text", rb_session_translate_text, 5);
    rb_define_singleton_method(mSession, "available_backends", rb_session_available_backends, 0);
    rb_define_singleton_method(mSession, "open_explicit", rb_session_open_explicit, 3);
    rb_define_singleton_method(mSession, "vad_slices", rb_vad_slices, -1);
    rb_define_singleton_method(mSession, "enhance_audio_rnnoise", rb_enhance_audio_rnnoise, 1);
    rb_define_singleton_method(mSession, "text_detect_language", rb_text_detect_language, 3);
    rb_define_singleton_method(mSession, "registry_list_backends", rb_registry_list_backends, 0);
    rb_define_singleton_method(mSession, "titanet_cosine_sim", rb_titanet_cosine_sim, 2);
    rb_define_singleton_method(mSession, "transcribe_vad_lang", rb_session_transcribe_vad_lang, -1);
    rb_define_singleton_method(mSession, "result_word_n_alts", rb_session_result_word_n_alts, 3);
    rb_define_singleton_method(mSession, "result_word_alt_p", rb_session_result_word_alt_p, 4);

    // Parakeet direct API
    rb_define_singleton_method(mSession, "parakeet_init", rb_parakeet_init, 3);
    rb_define_singleton_method(mSession, "parakeet_free", rb_parakeet_free, 1);
    rb_define_singleton_method(mSession, "parakeet_transcribe", rb_parakeet_transcribe, 3);

    // TitaNet
    rb_define_singleton_method(mSession, "titanet_init", rb_titanet_init, 2);
    rb_define_singleton_method(mSession, "titanet_free", rb_titanet_free, 1);
    rb_define_singleton_method(mSession, "titanet_embed", rb_titanet_embed, 2);

    // Speaker database
    rb_define_singleton_method(mSession, "speaker_db_load", rb_speaker_db_load, 1);
    rb_define_singleton_method(mSession, "speaker_db_free", rb_speaker_db_free, 1);
    rb_define_singleton_method(mSession, "speaker_db_count", rb_speaker_db_count, 1);
    rb_define_singleton_method(mSession, "speaker_db_match", rb_speaker_db_match, 3);
    rb_define_singleton_method(mSession, "speaker_db_enroll", rb_speaker_db_enroll, 3);

    // Mic (PLAN #62d) — CrispASR::Mic.{open(rate, channels) { |pcm| ... }, start, stop, close, default_device_name}.
    mMic = rb_define_module_under(mCrispASR, "Mic");
    rb_define_singleton_method(mMic, "open",                rb_mic_open,                -1);
    rb_define_singleton_method(mMic, "start",               rb_mic_start,                1);
    rb_define_singleton_method(mMic, "stop",                rb_mic_stop,                 1);
    rb_define_singleton_method(mMic, "close",               rb_mic_close,                1);
    rb_define_singleton_method(mMic, "default_device_name", rb_mic_default_device_name,  0);
}
