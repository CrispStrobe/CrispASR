package whisper

// Minimal TTS + S2S surface for the Go binding. Exposes the unified
// CrispASR Session API for TTS-capable backends (kokoro, vibevoice,
// qwen3-tts, moss-tts, orpheus, chatterbox, csm, dia, zonos-tts, speecht5, fastpitch,
// bananamind-tts, melotts, piper, parler-tts, outetts, indextts, voxcpm2-tts,
// cosyvoice3-tts, pocket-tts, f5-tts, bark, kugelaudio, tada, lfm2-audio,
// dots-tts, ...)
// and S2S-capable backends (lfm2-audio, mini-omni2), plus the kokoro
// per-language model + voice resolver (PLAN #56 opt 2b).

/*
// LDFLAGS for libcrispasr + all conditionally-built sub-libs are set in
// whisper.go (the canonical cgo block). Don't re-list here to avoid
// `ld: warning: ignoring duplicate libraries: '-lcrispasr'`.
#include <stdlib.h>
#include <string.h>

// Forward-declare the C ABI symbols. The full signatures live in
// src/crispasr_c_api.cpp (Session) and src/kokoro.h
// (kokoro_resolve_*_abi). They're exported by libcrispasr.dylib /
// .so and linked in via cgo's -lcrispasr.

typedef struct CrispasrSession CrispasrSession;

CrispasrSession* crispasr_session_open(const char* model_path, int n_threads);
void             crispasr_session_close(CrispasrSession* s);
int              crispasr_session_set_codec_path(CrispasrSession* s, const char* path);
int              crispasr_session_set_source_language(CrispasrSession* s, const char* lang);
int              crispasr_session_set_target_language(CrispasrSession* s, const char* lang);
int              crispasr_session_set_tts_reference_language(CrispasrSession* s, const char* lang);
int              crispasr_session_set_punctuation(CrispasrSession* s, int enable);
int              crispasr_session_set_punc_model(CrispasrSession* s, const char* punc_model);
int              crispasr_session_set_hotwords(CrispasrSession* s, const char* hotwords, float boost);
// Source separation (#359): stereo interleaved PCM in at the model's own rate.
int              crispasr_session_separate(CrispasrSession* s, const float* pcm_stereo, int n_samples);
int              crispasr_session_separate_n_stems(CrispasrSession* s);
const char*      crispasr_session_separate_stem_name(CrispasrSession* s, int stem_idx);
const float*     crispasr_session_separate_stem(CrispasrSession* s, int stem_idx, int* out_n_samples);
int              crispasr_session_separate_sample_rate(CrispasrSession* s);
int              crispasr_session_input_sample_rate(CrispasrSession* s);
int              crispasr_session_output_sample_rate(CrispasrSession* s);
int              crispasr_session_set_sensitivity(CrispasrSession* s, const char* preset);
int              crispasr_session_set_translate(CrispasrSession* s, int enable);
int              crispasr_session_set_temperature(CrispasrSession* s, float temperature, unsigned long long seed);
int              crispasr_session_set_tts_seed(CrispasrSession* s, unsigned long long seed);
int              crispasr_session_set_max_new_tokens(CrispasrSession* s, int max_new_tokens);
int              crispasr_session_set_frequency_penalty(CrispasrSession* s, float penalty);
int              crispasr_session_set_tts_steps(CrispasrSession* s, int steps);
int              crispasr_session_set_tts_num_candidates(CrispasrSession* s, int n);
int              crispasr_session_set_top_p(CrispasrSession* s, float top_p);
int              crispasr_session_set_top_k(CrispasrSession* s, int top_k);
int              crispasr_session_set_do_sample(CrispasrSession* s, int enable);
int              crispasr_session_set_min_p(CrispasrSession* s, float min_p);
int              crispasr_session_set_repetition_penalty(CrispasrSession* s, float r);
int              crispasr_session_set_cfg_weight(CrispasrSession* s, float cfg_weight);
int              crispasr_session_set_tts_noise_temp(CrispasrSession* s, float noise_temp);
int              crispasr_session_set_exaggeration(CrispasrSession* s, float exaggeration);
int              crispasr_session_set_max_speech_tokens(CrispasrSession* s, int n);
int              crispasr_session_set_min_speech_tokens(CrispasrSession* s, int n);
int              crispasr_session_set_length_scale(CrispasrSession* s, float scale);
int              crispasr_session_set_g2p_dict(CrispasrSession* s, const char* source);
int              crispasr_session_set_best_of(CrispasrSession* s, int n);
int              crispasr_session_set_beam_size(CrispasrSession* s, int n);
int              crispasr_session_set_return_logits(CrispasrSession* s, int enable);
int              crispasr_session_set_grammar_text(CrispasrSession* s, const char* gbnf_text,
                                                   const char* root_rule, float penalty);
int              crispasr_session_set_fallback_thresholds(CrispasrSession* s, float entropy_thold,
                                                          float logprob_thold, float no_speech_thold,
                                                          float temperature_inc);
int              crispasr_session_set_alt_n(CrispasrSession* s, int n);
int              crispasr_session_set_whisper_decode_extras(CrispasrSession* s, int suppress_nst,
                                                            const char* suppress_regex,
                                                            int carry_initial_prompt);
int              crispasr_session_set_ask(CrispasrSession* s, const char* prompt);
int              crispasr_session_detect_language(CrispasrSession* s, const float* pcm, int n_samples,
                                                  const char* lid_model_path, int method,
                                                  char* out_lang, int out_lang_cap, float* out_prob);
int              crispasr_session_set_voice(CrispasrSession* s, const char* path, const char* ref_text_or_null);
int              crispasr_session_set_speaker_name(CrispasrSession* s, const char* name);
int              crispasr_session_set_speaker_id(CrispasrSession* s, int id);
int              crispasr_session_n_speakers(CrispasrSession* s);
const char*      crispasr_session_get_speaker_name(CrispasrSession* s, int i);
int              crispasr_session_set_instruct(CrispasrSession* s, const char* instruct);
int              crispasr_session_set_tts_phonemes(CrispasrSession* s, const char* phonemes);
void             crispasr_session_set_tts_pad_silence_ms(CrispasrSession* s, int ms);
int              crispasr_session_is_custom_voice(CrispasrSession* s);
int              crispasr_session_is_voice_design(CrispasrSession* s);
float*           crispasr_session_synthesize(CrispasrSession* s, const char* text, int* out_n_samples);
float*           crispasr_session_synthesize_raw(CrispasrSession* s, const char* text, int* out_n_samples);
int              crispasr_session_accept_marking_responsibility(CrispasrSession* s, const char* attestation);
int              crispasr_session_set_speaker_identity(CrispasrSession* s, const char* identity);
float*           crispasr_session_speech_to_speech(CrispasrSession* s, const float* in_samples, int n_in_samples,
                                                    char** out_text, int* out_n_samples);
void             crispasr_session_translate_text_free(char* text);
void             crispasr_pcm_free(float* pcm);
int              crispasr_session_kokoro_clear_phoneme_cache(CrispasrSession* s);

int crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang,
                                               char* out_path, int out_path_len);
int crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang,
                                               char* out_path, int out_path_len,
                                               char* out_picked, int out_picked_len);

// --- ASR transcription (PLAN #59) ---
typedef struct crispasr_session_result crispasr_session_result;
crispasr_session_result* crispasr_session_transcribe(CrispasrSession* s, const float* pcm, int n_samples);
crispasr_session_result* crispasr_session_transcribe_lang(CrispasrSession* s, const float* pcm, int n_samples,
                                                          const char* language);
crispasr_session_result* crispasr_session_transcribe_chunked_lang(CrispasrSession* s, const float* pcm, int n_samples,
                                                                  int chunk_seconds, int overlap_seconds,
                                                                  const char* language);
crispasr_session_result* crispasr_session_transcribe_vad(CrispasrSession* s, const float* pcm, int n_samples,
                                                         int sample_rate, const char* vad_model_path, void* opts);
int          crispasr_get_progress(void);
int          crispasr_session_result_n_segments(crispasr_session_result* r);
const char*  crispasr_session_result_segment_text(crispasr_session_result* r, int i);
long long    crispasr_session_result_segment_t0(crispasr_session_result* r, int i);
long long    crispasr_session_result_segment_t1(crispasr_session_result* r, int i);
int          crispasr_session_result_n_words(crispasr_session_result* r, int i_seg);
const char*  crispasr_session_result_word_text(crispasr_session_result* r, int i_seg, int i_word);
long long    crispasr_session_result_word_t0(crispasr_session_result* r, int i_seg, int i_word);
long long    crispasr_session_result_word_t1(crispasr_session_result* r, int i_seg, int i_word);
float        crispasr_session_result_word_p(crispasr_session_result* r, int i_seg, int i_word);
float        crispasr_session_result_segment_no_speech_prob(crispasr_session_result* r, int i_seg);
const char*  crispasr_session_result_segment_speaker(crispasr_session_result* r, int i);
// Per-frame CTC logits (opted in via crispasr_session_set_return_logits) for
// backends that produce a dense CTC grid (Omni CTC, wav2vec2/hubert/data2vec,
// canary-ctc). Frame-major: logits[t * n_logit_vocab + v]. Raw pre-softmax for
// Omni & wav2vec2; log-probabilities for canary-ctc. _logits returns NULL when
// none captured.
int          crispasr_session_result_n_logit_frames(crispasr_session_result* r);
int          crispasr_session_result_n_logit_vocab(crispasr_session_result* r);
const float* crispasr_session_result_logits(crispasr_session_result* r);
void         crispasr_session_result_free(crispasr_session_result* r);

// --- Punctuation (PLAN #59) ---
void*        crispasr_punc_init(const char* model_path);
const char*  crispasr_punc_process(void* ctx, const char* text);
void         crispasr_punc_free_text(const char* text);
void         crispasr_punc_free(void* ctx);

// --- Alignment (PLAN #59) ---
typedef struct crispasr_align_result crispasr_align_result;
crispasr_align_result* crispasr_align_words_abi(const char* aligner_model, const char* transcript,
                                                 const float* samples, int n_samples, long long t_offset_cs,
                                                 int n_threads);
int          crispasr_align_result_n_words(crispasr_align_result* r);
const char*  crispasr_align_result_word_text(crispasr_align_result* r, int i);
long long    crispasr_align_result_word_t0(crispasr_align_result* r, int i);
long long    crispasr_align_result_word_t1(crispasr_align_result* r, int i);
void         crispasr_align_result_free(crispasr_align_result* r);

// --- Watermark ---
int   crispasr_watermark_load_model(const char* gguf_path);
void  crispasr_watermark_embed(float* pcm, int n_samples, float alpha);
float crispasr_watermark_detect(const float* pcm, int n_samples);

// --- VAD (PLAN #59) ---
int crispasr_vad_segments(const char* vad_model_path, const float* pcm, int n_samples,
                          int sample_rate, float threshold, int min_speech_ms, int min_silence_ms,
                          int n_threads, int use_gpu, float** out_spans);
void crispasr_vad_free(float* spans);

// --- Diarization (PLAN #59) ---
struct crispasr_diarize_seg_abi {
    long long t0_cs;
    long long t1_cs;
    int       speaker;
    int       _pad;
};
struct crispasr_diarize_opts_abi {
    int         method;
    int         n_threads;
    long long   slice_t0_cs;
    const char* pyannote_model_path;
    // #324 foxnose (method 4). This layout is hand-maintained and MUST match
    // struct crispasr_diarize_opts_abi in src/crispasr_c_api.cpp exactly —
    // Go allocates the struct, so a missing field here means the C side reads
    // past the end of it.
    const char* foxnose_embedder_path;
    int         min_speakers;
    int         max_speakers;
    int         num_speakers;
    int         _pad2;
};
int crispasr_diarize_segments_abi(const float* left_pcm, const float* right_pcm, int n_samples,
                                  int is_stereo, struct crispasr_diarize_seg_abi* segs, int n_segs,
                                  const struct crispasr_diarize_opts_abi* opts);
// #395: a speaker turn the METHOD derived from the audio, independent of the
// caller's segment grid. Same hand-maintained-layout warning as the opts
// struct above: Go allocates these, so this must match
// struct crispasr_diarize_turn_abi in src/crispasr_c_api.cpp exactly.
struct crispasr_diarize_turn_abi {
    long long t0_cs;
    long long t1_cs;
    int       speaker;
    int       _pad;
};
int crispasr_diarize_segments_turns_abi(const float* left_pcm, const float* right_pcm, int n_samples,
                                        int is_stereo, struct crispasr_diarize_seg_abi* segs, int n_segs,
                                        const struct crispasr_diarize_opts_abi* opts,
                                        struct crispasr_diarize_turn_abi* out_turns, int n_turns_cap,
                                        int* out_n_turns);

// --- Pluggable speaker embedder, clustering, pyannote cache (#107 P6) ---
void*       crispasr_speaker_embedder_make_abi(const char* model_spec, int n_threads, const char* cache_dir);
void        crispasr_speaker_embedder_free_abi(void* embedder);
int         crispasr_speaker_embedder_dim_abi(const void* embedder);
int         crispasr_speaker_embedder_embed_abi(void* embedder, const float* pcm_16k, int n_samples, float* out);
const char* crispasr_speaker_embedder_name_abi(const void* embedder);
int         crispasr_speaker_cluster_abi(const float* embeddings, int n, int dim,
                                         float merge_threshold, int max_speakers, int* labels_out);
void*       crispasr_pyannote_cache_compute_abi(const float* full_audio, int n_samples,
                                                const char* model_path, int n_threads);
void        crispasr_pyannote_cache_free_abi(void* cache);
int         crispasr_pyannote_cache_apply_abi(const void* cache, long long slice_t0_cs,
                                              struct crispasr_diarize_seg_abi* segs, int n_segs);

// --- Standalone LID (PLAN #59) ---
int crispasr_detect_language_pcm(const float* samples, int n_samples, int method,
                                  const char* model_path, int n_threads, int use_gpu,
                                  int gpu_device, int flash_attn,
                                  char* out_lang, int out_lang_cap, float* out_prob);

// --- Registry + cache (PLAN #59) ---
int crispasr_registry_lookup_abi(const char* backend, char* out_filename, int filename_cap,
                                 char* out_url, int url_cap, char* out_size, int size_cap);
int crispasr_cache_ensure_file_abi(const char* filename, const char* url, int quiet,
                                   const char* cache_dir_override, char* out_buf, int out_cap);
int crispasr_cache_dir_abi(const char* cache_dir_override, char* out_buf, int out_cap);

// --- Streaming (PLAN #62) ---
typedef struct CrispasrStream CrispasrStream;
CrispasrStream* crispasr_session_stream_open(CrispasrSession* s, int n_threads, int step_ms,
                                             int length_ms, int keep_ms, const char* language, int translate);
CrispasrStream* crispasr_stream_open(void* ctx, int n_threads, int step_ms,
                                     int length_ms, int keep_ms, const char* language, int translate);
int             crispasr_stream_feed(CrispasrStream* s, const float* pcm, int n_samples);
int             crispasr_stream_get_text(CrispasrStream* s, char* out_text, int out_cap,
                                         double* out_t0_s, double* out_t1_s, long long* out_counter);
int             crispasr_stream_flush(CrispasrStream* s);
void            crispasr_stream_close(CrispasrStream* s);
void            crispasr_stream_set_live_decode(CrispasrStream* s, int enabled);

// --- Text-LID ---
int crispasr_text_detect_language(const char* text, const char* model_path, int n_threads,
                                  char* out_label, int out_label_cap, float* out_confidence);

// --- Backend detection ---
int crispasr_detect_backend_from_gguf(const char* path, char* out_name, int out_cap);

// --- RNNoise audio enhancement ---
int crispasr_enhance_audio_rnnoise(const float* in_pcm, int n_samples, float* out_pcm, int out_cap);

// --- params_set_* on whisper_full_params ---
void crispasr_params_set_language(void* p, const char* lang);
void crispasr_params_set_translate(void* p, int v);
void crispasr_params_set_detect_language(void* p, int v);
void crispasr_params_set_token_timestamps(void* p, int v);
void crispasr_params_set_n_threads(void* p, int n);
void crispasr_params_set_max_len(void* p, int n);
void crispasr_params_set_best_of(void* p, int n);
void crispasr_params_set_split_on_word(void* p, int v);
void crispasr_params_set_no_context(void* p, int v);
void crispasr_params_set_single_segment(void* p, int v);
void crispasr_params_set_print_realtime(void* p, int v);
void crispasr_params_set_print_progress(void* p, int v);
void crispasr_params_set_print_timestamps(void* p, int v);
void crispasr_params_set_print_special(void* p, int v);
void crispasr_params_set_suppress_blank(void* p, int v);
void crispasr_params_set_temperature(void* p, float t);
void crispasr_params_set_max_tokens(void* p, int n);
void crispasr_params_set_initial_prompt(void* p, const char* prompt);
void crispasr_params_set_alt_n(void* p, int n);
void crispasr_params_set_vad(void* p, int v);
void crispasr_params_set_vad_model_path(void* p, const char* path);
void crispasr_params_set_vad_threshold(void* p, float t);
void crispasr_params_set_vad_min_speech_ms(void* p, int ms);
void crispasr_params_set_vad_min_silence_ms(void* p, int ms);
void crispasr_params_set_tdrz(void* p, int v);

// --- Token-level accessors ---
long long crispasr_token_t0(void* ctx, int i_seg, int i_tok);
long long crispasr_token_t1(void* ctx, int i_seg, int i_tok);
float     crispasr_token_p(void* ctx, int i_seg, int i_tok);
int       crispasr_token_n_alts(void* ctx, int i_seg, int i_tok);
int       crispasr_token_alt_id(void* ctx, int i_seg, int i_tok, int i_alt);
float     crispasr_token_alt_p(void* ctx, int i_seg, int i_tok, int i_alt);
int       crispasr_token_alt_text(void* ctx, int i_seg, int i_tok, int i_alt, char* out, int out_cap);

// --- Language detection (whisper context) ---
float crispasr_detect_language(void* ctx, const float* pcm, int n_samples,
                               int n_threads, char* out_code, int out_cap);

// --- VAD slices ---
int crispasr_vad_slices(const char* vad_model_path, const float* pcm, int n_samples,
                        int sample_rate, float threshold, int min_speech_ms, int min_silence_ms,
                        int speech_pad_ms, float max_chunk_duration_s, int n_threads,
                        float** out_spans);

// --- LCS dedup ---
int crispasr_lcs_dedup_prefix_count(const int* prev_tail_tokens, int n_prev,
                                    const int* curr_tokens, int n_curr, int min_lcs_length);

// --- Direct Parakeet API ---
// Note: nemotron, lfm2-audio, and other recent backends are accessed
// via the Session API (Open → Transcribe → Close). Standalone wrappers
// below are for backwards compatibility with early parakeet integrations.
void* crispasr_parakeet_init(const char* model_path, int n_threads, int use_flash);
void  crispasr_parakeet_free(void* ctx);
void* crispasr_parakeet_transcribe(void* ctx, const float* pcm, int n_samples, const char* language);
const char* crispasr_parakeet_result_text(void* r);
int         crispasr_parakeet_result_n_words(void* r);
const char* crispasr_parakeet_result_word_text(void* r, int i);
long long   crispasr_parakeet_result_word_t0(void* r, int i);
long long   crispasr_parakeet_result_word_t1(void* r, int i);
int         crispasr_parakeet_result_n_tokens(void* r);
const char* crispasr_parakeet_result_token_text(void* r, int i);
long long   crispasr_parakeet_result_token_t0(void* r, int i);
long long   crispasr_parakeet_result_token_t1(void* r, int i);
float       crispasr_parakeet_result_token_p(void* r, int i);
void        crispasr_parakeet_result_free(void* r);

// --- TitaNet ---
void* crispasr_titanet_init(const char* model_path, int n_threads);
void  crispasr_titanet_free(void* ctx);
int   crispasr_titanet_embed(void* ctx, const float* pcm_16k, int n_samples, float* out);
float crispasr_titanet_cosine_sim(const float* a, const float* b, int dim);

// --- Speaker database (closed-roster, consent-gated — issue #266) ---
void* crispasr_speaker_db_open(const char* dir_path, const char* expected_names_csv,
                               int consent_attested);
void  crispasr_speaker_db_free(void* db);
int   crispasr_speaker_db_count(const void* db);
float crispasr_speaker_db_match(const void* db, const float* embedding, int dim,
                                float threshold, char* out_name, int out_cap);
int   crispasr_speaker_db_enroll2(const char* dir_path, const char* name,
                                  const float* embedding, int dim, int consent_attested);

// --- Kokoro lang helpers ---
int  crispasr_kokoro_lang_is_german_abi(const char* lang);
int  crispasr_kokoro_lang_has_native_voice_abi(const char* lang);

// --- Registry by filename ---
int crispasr_registry_lookup_by_filename_abi(const char* filename, char* out_filename, int filename_cap,
                                             char* out_url, int url_cap, char* out_size, int size_cap);
int crispasr_registry_list_backends_abi(char* out_csv, int out_cap);
int crispasr_registry_default_bundle_info_abi(const char* backend, char* out_backend, int backend_cap,
                                              char* out_license, int license_cap, int* out_requires_acceptance);
int crispasr_registry_default_bundle_artifact_abi(const char* backend, int index, int* out_kind,
                                                  char* out_filename, int filename_cap, char* out_url,
                                                  int url_cap, char* out_size, int size_cap);

// --- Session extras ---
int crispasr_session_available_backends(char* out_csv, int out_cap);
int crispasr_session_detected_language(CrispasrSession* s, char* out_buf, int out_cap);
// CTC vocabulary access (Omni CTC backend): n_vocab piece count, token_text
// maps an id to its raw piece (word-boundary marker intact) or "" when out of
// range / unsupported. Pairs with the result logits accessor for detokenization.
int          crispasr_session_n_vocab(CrispasrSession* s);
const char*  crispasr_session_token_text(CrispasrSession* s, int id);
CrispasrSession* crispasr_session_open_explicit(const char* model_path, const char* backend_name, int n_threads);
CrispasrSession* crispasr_session_open_with_params(const char* model_path, const char* backend_name, const void* params);
crispasr_session_result* crispasr_session_transcribe_vad_lang(CrispasrSession* s, const float* pcm, int n_samples,
                                                              int sample_rate, const char* vad_model_path, void* opts,
                                                              const char* language);
char* crispasr_session_translate_text(CrispasrSession* s, const char* text, const char* src_lang,
                                      const char* tgt_lang, int max_tokens);
void  crispasr_session_translate_text_free(char* text);
int   crispasr_session_result_word_n_alts(crispasr_session_result* r, int i_seg, int i_word);
const char* crispasr_session_result_word_alt_text(crispasr_session_result* r, int i_seg, int i_word, int i_alt);
float crispasr_session_result_word_alt_p(crispasr_session_result* r, int i_seg, int i_word, int i_alt);
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// CrispasrSession is a TTS/S2S-capable session (kokoro, vibevoice, qwen3-tts, moss-tts, orpheus, parler-tts, pocket-tts, tada, lfm2-audio, mini-omni2).
type CrispasrSession struct {
	handle *C.CrispasrSession
}

// SessionOpen opens a backend session for the given model file.
// Detects the backend automatically from the GGUF metadata.
// Returns an error if the model can't be loaded.
func SessionOpen(modelPath string, nThreads int) (*CrispasrSession, error) {
	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))
	h := C.crispasr_session_open(cpath, C.int(nThreads))
	if h == nil {
		return nil, errors.New("crispasr_session_open: failed to open " + modelPath)
	}
	return &CrispasrSession{handle: h}, nil
}

// Close releases the session handle.
func (s *CrispasrSession) Close() {
	if s != nil && s.handle != nil {
		C.crispasr_session_close(s.handle)
		s.handle = nil
	}
}

// ClearPhonemeCache drops the kokoro per-session phoneme cache.
// No-op for non-kokoro backends. Useful for long-running daemons that
// resynthesize across many speakers and want bounded memory. (PLAN #56 #5)
func (s *CrispasrSession) ClearPhonemeCache() error {
	rc := C.crispasr_session_kokoro_clear_phoneme_cache(s.handle)
	if rc != 0 {
		return errors.New("crispasr_session_kokoro_clear_phoneme_cache failed")
	}
	return nil
}

// ---------------------------------------------------------------------------
// Sticky session-state setters (PLAN #59 partial unblock).
// ---------------------------------------------------------------------------

// SetSourceLanguage sets the sticky source-language hint (canary, cohere,
// voxtral, whisper). Empty string clears. Per-call language args still win.
func (s *CrispasrSession) SetSourceLanguage(lang string) error {
	cl := C.CString(lang)
	defer C.free(unsafe.Pointer(cl))
	rc := C.crispasr_session_set_source_language(s.handle, cl)
	if rc != 0 {
		return errors.New("crispasr_session_set_source_language failed")
	}
	return nil
}

// SetTargetLanguage sets the sticky target-language. When ≠ source on
// canary/cohere, the backend emits a translation. For whisper, pair with
// SetTranslate(true).
func (s *CrispasrSession) SetTargetLanguage(lang string) error {
	cl := C.CString(lang)
	defer C.free(unsafe.Pointer(cl))
	rc := C.crispasr_session_set_target_language(s.handle, cl)
	if rc != 0 {
		return errors.New("crispasr_session_set_target_language failed")
	}
	return nil
}

// SetTTSReferenceLanguage declares the language a voice-cloning REFERENCE clip
// is spoken in (issue #329). Cross-lingual TTS backends (cosyvoice3) compare it
// to the requested output language — SetTargetLanguage, falling back to
// SetSourceLanguage — and drop the reference transcript when they differ, so the
// clone speaks the target language instead of carrying the reference's accent.
//
// Optional: the backend otherwise infers it from the voice bank or the reference
// transcript, and that inference declines rather than guesses on a short
// transcript. When it declines, the requested target language has no effect —
// set this to make it explicit. Empty string clears.
func (s *CrispasrSession) SetTTSReferenceLanguage(lang string) error {
	cl := C.CString(lang)
	defer C.free(unsafe.Pointer(cl))
	rc := C.crispasr_session_set_tts_reference_language(s.handle, cl)
	if rc != 0 {
		return errors.New("crispasr_session_set_tts_reference_language failed")
	}
	return nil
}

// SetPunctuation toggles punctuation + capitalisation in the output
// (canary/cohere natively; LLM backends via post-process strip). Default true.
func (s *CrispasrSession) SetPunctuation(enable bool) error {
	v := C.int(0)
	if enable {
		v = 1
	}
	rc := C.crispasr_session_set_punctuation(s.handle, v)
	if rc != 0 {
		return errors.New("crispasr_session_set_punctuation failed")
	}
	return nil
}

// SetPuncModel selects + loads a punctuation-restoration model on the session.
// model is an alias (auto|firered|fullstop|punctuate-all|pcs) or a .gguf path;
// "none" or "" unloads. Auto-downloads on first use. Restores punctuation on
// backends that emit none (parakeet RNNT/CTC, etc.) — the same post-processor
// the CLI --punc-model and the server apply.
func (s *CrispasrSession) SetPuncModel(model string) error {
	cm := C.CString(model)
	defer C.free(unsafe.Pointer(cm))
	rc := C.crispasr_session_set_punc_model(s.handle, cm)
	if rc != 0 {
		return errors.New("crispasr_session_set_punc_model failed")
	}
	return nil
}

// SetHotwords sets comma-separated hotwords for contextual biasing, boosted by
// `boost` log-prob per token match (parakeet CTC/TDT trie, LLM backends prompt
// injection). Empty string clears.
func (s *CrispasrSession) SetHotwords(hotwords string, boost float32) error {
	ch := C.CString(hotwords)
	defer C.free(unsafe.Pointer(ch))
	rc := C.crispasr_session_set_hotwords(s.handle, ch, C.float(boost))
	if rc != 0 {
		return errors.New("crispasr_session_set_hotwords failed")
	}
	return nil
}

// SetSensitivity applies a named bundle of the four decoder fallback
// thresholds: "conservative", "balanced" (the shipped defaults, a no-op) or
// "aggressive". "strict"/"default"/"loose" are aliases. Mirrors the CLI's
// --sensitivity.
//
// The four thresholds interact — a decode is only retried when the logprob and
// no-speech bars are both crossed — so they move as a set. A later
// SetFallbackThresholds overrides this. An unrecognised name is rejected
// rather than silently treated as "balanced".
func (s *CrispasrSession) SetSensitivity(preset string) error {
	cp := C.CString(preset)
	defer C.free(unsafe.Pointer(cp))
	rc := C.crispasr_session_set_sensitivity(s.handle, cp)
	if rc == -2 {
		return errors.New("unknown sensitivity preset (expected: conservative, balanced, aggressive)")
	}
	if rc != 0 {
		return errors.New("crispasr_session_set_sensitivity failed")
	}
	return nil
}

// SetTranslate enables whisper sticky --translate. For canary/cohere/voxtral
// the equivalent is SetTargetLanguage ≠ source.
func (s *CrispasrSession) SetTranslate(enable bool) error {
	v := C.int(0)
	if enable {
		v = 1
	}
	rc := C.crispasr_session_set_translate(s.handle, v)
	if rc != 0 {
		return errors.New("crispasr_session_set_translate failed")
	}
	return nil
}

// SetTemperature sets the decoder temperature on backends that support
// runtime control (canary, cohere, parakeet, moonshine). Other backends
// silently no-op. seed is the RNG seed; pass 0 for time-based.
func (s *CrispasrSession) SetTemperature(temperature float32, seed uint64) error {
	rc := C.crispasr_session_set_temperature(s.handle, C.float(temperature), C.ulonglong(seed))
	// rc == -2 means no backend in this session honours it — soft no-op.
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_temperature failed")
	}
	return nil
}

// SetTTSSeed sets the seed on TTS session backends that support runtime
// reseeding (chatterbox, vibevoice, qwen3-tts, orpheus). Other backends
// silently no-op.
func (s *CrispasrSession) SetTTSSeed(seed uint64) error {
	rc := C.crispasr_session_set_tts_seed(s.handle, C.ulonglong(seed))
	// rc == -2 means no backend in this session honours it — soft no-op.
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_tts_seed failed")
	}
	return nil
}

// SetMaxNewTokens sets a generated-token cap for autoregressive session
// backends. Pass <= 0 to clear the override and use the backend default.
func (s *CrispasrSession) SetMaxNewTokens(maxNewTokens int) error {
	rc := C.crispasr_session_set_max_new_tokens(s.handle, C.int(maxNewTokens))
	if rc != 0 {
		return errors.New("crispasr_session_set_max_new_tokens failed")
	}
	return nil
}

// SetFrequencyPenalty sets an opt-in repeated generated-token penalty for
// autoregressive session backends. Pass <= 0 to disable it.
func (s *CrispasrSession) SetFrequencyPenalty(penalty float32) error {
	rc := C.crispasr_session_set_frequency_penalty(s.handle, C.float(penalty))
	if rc != 0 {
		return errors.New("crispasr_session_set_frequency_penalty failed")
	}
	return nil
}

// SetTTSSteps sets the diffusion / CFM / masked-iterative step count for
// step-based TTS backends (chatterbox, vibevoice, kugelaudio, tada, irodori, supertonic,
// omnivoice). Higher = better fidelity, slower. Other backends silently no-op.
func (s *CrispasrSession) SetTTSSteps(steps int) error {
	rc := C.crispasr_session_set_tts_steps(s.handle, C.int(steps))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_tts_steps failed")
	}
	return nil
}

// SetTTSNumCandidates sets the number of flow-matching timing candidates
// ranked per token (TADA). Higher = more reliable multilingual timing at
// higher cost. Other backends silently no-op.
func (s *CrispasrSession) SetTTSNumCandidates(n int) error {
	rc := C.crispasr_session_set_tts_num_candidates(s.handle, C.int(n))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_tts_num_candidates failed")
	}
	return nil
}

// SetTopP sets the top-p nucleus-sampling threshold. Honoured by chatterbox.
func (s *CrispasrSession) SetTopP(topP float32) error {
	rc := C.crispasr_session_set_top_p(s.handle, C.float(topP))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_top_p failed")
	}
	return nil
}

// SetTopK sets the top-k sampling cutoff (0 = disabled). Honoured by TADA.
func (s *CrispasrSession) SetTopK(topK int) error {
	rc := C.crispasr_session_set_top_k(s.handle, C.int(topK))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_top_k failed")
	}
	return nil
}

// SetDoSample enables/disables sampling (false = greedy). Honoured by TADA.
func (s *CrispasrSession) SetDoSample(enable bool) error {
	cEnable := C.int(0)
	if enable {
		cEnable = C.int(1)
	}
	rc := C.crispasr_session_set_do_sample(s.handle, cEnable)
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_do_sample failed")
	}
	return nil
}

// SetMinP sets the min-p sampling threshold. Honoured by chatterbox.
func (s *CrispasrSession) SetMinP(minP float32) error {
	rc := C.crispasr_session_set_min_p(s.handle, C.float(minP))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_min_p failed")
	}
	return nil
}

// SetRepetitionPenalty sets the repetition penalty (1.0 = no penalty).
// Honoured by chatterbox.
func (s *CrispasrSession) SetRepetitionPenalty(r float32) error {
	rc := C.crispasr_session_set_repetition_penalty(s.handle, C.float(r))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_repetition_penalty failed")
	}
	return nil
}

// SetCFGWeight sets the classifier-free-guidance weight (chatterbox).
// 0 disables CFG; 0.5 is the upstream default.
func (s *CrispasrSession) SetCFGWeight(cfgWeight float32) error {
	rc := C.crispasr_session_set_cfg_weight(s.handle, C.float(cfgWeight))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_cfg_weight failed")
	}
	return nil
}

// SetTtsNoiseTemp sets the TADA flow-matching noise temperature
// (Python noise_temp, default 0.9).
func (s *CrispasrSession) SetTtsNoiseTemp(noiseTemp float32) error {
	rc := C.crispasr_session_set_tts_noise_temp(s.handle, C.float(noiseTemp))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_tts_noise_temp failed")
	}
	return nil
}

// SetExaggeration sets the emotion-exaggeration scalar (chatterbox).
// 0.5 is the upstream default.
func (s *CrispasrSession) SetExaggeration(exaggeration float32) error {
	rc := C.crispasr_session_set_exaggeration(s.handle, C.float(exaggeration))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_exaggeration failed")
	}
	return nil
}

// SetMaxSpeechTokens sets the upper bound on speech tokens generated per
// synthesize call (chatterbox). Default ≈1000 tokens ≈ 20 s.
func (s *CrispasrSession) SetMaxSpeechTokens(n int) error {
	rc := C.crispasr_session_set_max_speech_tokens(s.handle, C.int(n))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_max_speech_tokens failed")
	}
	return nil
}

// SetMinSpeechTokens sets the floor on generated audio length (MOSS TTS). Units are codec frames at 12.5 Hz (80 ms each), so n=25 floors at ~2 s; other backends no-op (rc=-2).
func (s *CrispasrSession) SetMinSpeechTokens(n int) error {
	rc := C.crispasr_session_set_min_speech_tokens(s.handle, C.int(n))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_min_speech_tokens failed")
	}
	return nil
}

// SetLengthScale sets the per-phoneme length-scale / speaking-rate scalar for
// TTS backends with a duration model. Honoured by kokoro today; other backends
// silently no-op. 1.0 = upstream default; >1.0 = slower; <1.0 = faster.
func (s *CrispasrSession) SetLengthScale(scale float32) error {
	rc := C.crispasr_session_set_length_scale(s.handle, C.float(scale))
	if rc != 0 && rc != -2 {
		return errors.New("crispasr_session_set_length_scale failed")
	}
	return nil
}

// SetG2PDict selects the G2P pronunciation dictionary source for TTS backends:
// "olaph" (MIT, default), "open-dict" (CC-BY-SA), or a file path.
func (s *CrispasrSession) SetG2PDict(source string) error {
	cs := C.CString(source)
	defer C.free(unsafe.Pointer(cs))
	rc := C.crispasr_session_set_g2p_dict(s.handle, cs)
	if rc != 0 {
		return errors.New("crispasr_session_set_g2p_dict failed")
	}
	return nil
}

// SetBestOf sets the best-of-N sampling count for ASR backends.
func (s *CrispasrSession) SetBestOf(n int) error {
	rc := C.crispasr_session_set_best_of(s.handle, C.int(n))
	if rc != 0 {
		return errors.New("crispasr_session_set_best_of failed")
	}
	return nil
}

// SetBeamSize sets the beam-search width for ASR backends that support it.
func (s *CrispasrSession) SetBeamSize(n int) error {
	rc := C.crispasr_session_set_beam_size(s.handle, C.int(n))
	if rc != 0 {
		return errors.New("crispasr_session_set_beam_size failed")
	}
	return nil
}

// SetReturnLogits opts in to capturing the per-frame CTC logits on
// subsequent transcribe calls (backends with a dense CTC grid: Omni CTC,
// wav2vec2/hubert/data2vec, canary-ctc). Off by default: capture
// copies an NFrames × NVocab float grid per call, so leave it off unless a
// consumer (e.g. forced alignment) needs the logits. Retrieve them with
// TranscribeWithLogits.
func (s *CrispasrSession) SetReturnLogits(on bool) error {
	enable := C.int(0)
	if on {
		enable = 1
	}
	rc := C.crispasr_session_set_return_logits(s.handle, enable)
	if rc != 0 {
		return errors.New("crispasr_session_set_return_logits failed")
	}
	return nil
}

// SetGrammarText sets a GBNF grammar for constrained whisper decoding.
// Pass an empty gbnfText to clear the grammar. penalty is the grammar
// penalty scalar (default 100.0).
func (s *CrispasrSession) SetGrammarText(gbnfText, rootRule string, penalty float32) error {
	var cgbnf *C.char
	if gbnfText != "" {
		cgbnf = C.CString(gbnfText)
		defer C.free(unsafe.Pointer(cgbnf))
	}
	var croot *C.char
	if rootRule != "" {
		croot = C.CString(rootRule)
		defer C.free(unsafe.Pointer(croot))
	}
	rc := C.crispasr_session_set_grammar_text(s.handle, cgbnf, croot, C.float(penalty))
	if rc == -2 {
		return errors.New("set_grammar_text: invalid GBNF or root rule not found")
	}
	if rc != 0 {
		return errors.New("crispasr_session_set_grammar_text failed")
	}
	return nil
}

// SetFallbackThresholds sets whisper decoder fallback thresholds.
// temperatureInc=0 disables fallback entirely (equivalent to --no-fallback).
func (s *CrispasrSession) SetFallbackThresholds(entropyThold, logprobThold, noSpeechThold, temperatureInc float32) error {
	rc := C.crispasr_session_set_fallback_thresholds(s.handle,
		C.float(entropyThold), C.float(logprobThold),
		C.float(noSpeechThold), C.float(temperatureInc))
	if rc != 0 {
		return errors.New("crispasr_session_set_fallback_thresholds failed")
	}
	return nil
}

// SetAltN sets per-token top-N alternative-candidate capture for whisper
// greedy decode. 0 disables it.
func (s *CrispasrSession) SetAltN(n int) error {
	rc := C.crispasr_session_set_alt_n(s.handle, C.int(n))
	if rc != 0 {
		return errors.New("crispasr_session_set_alt_n failed")
	}
	return nil
}

// SetWhisperDecodeExtras sets whisper-only text-suppression and prompt-carry
// extras. suppressRegex may be empty to clear any prior regex.
func (s *CrispasrSession) SetWhisperDecodeExtras(suppressNST bool, suppressRegex string, carryInitialPrompt bool) error {
	var cregex *C.char
	if suppressRegex != "" {
		cregex = C.CString(suppressRegex)
		defer C.free(unsafe.Pointer(cregex))
	} else {
		cregex = C.CString("")
		defer C.free(unsafe.Pointer(cregex))
	}
	snst := 0
	if suppressNST {
		snst = 1
	}
	carry := 0
	if carryInitialPrompt {
		carry = 1
	}
	rc := C.crispasr_session_set_whisper_decode_extras(s.handle, C.int(snst), cregex, C.int(carry))
	if rc != 0 {
		return errors.New("crispasr_session_set_whisper_decode_extras failed")
	}
	return nil
}

// SetAsk sets a free-form prompt / question passed to the backend on the next
// transcribe or synthesize call (used by LLM-style backends).
func (s *CrispasrSession) SetAsk(prompt string) error {
	cprompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cprompt))
	rc := C.crispasr_session_set_ask(s.handle, cprompt)
	if rc != 0 {
		return errors.New("crispasr_session_set_ask failed")
	}
	return nil
}

// ---------------------------------------------------------------------------
// Streaming (PLAN #62) — rolling-window decoder for whisper today.
// ---------------------------------------------------------------------------

// Stream is a streaming-decoder handle returned by Session.StreamOpen.
// Feed PCM, pull text. Whisper-only at the C-ABI level today.
type Stream struct {
	handle *C.CrispasrStream
}

// StreamingUpdate is one commit from a streaming session — the latest
// concatenated text + its absolute audio-time bounds. Counter increments
// per commit; same value = no new text.
type StreamingUpdate struct {
	Text    string
	T0      float64
	T1      float64
	Counter int64
}

// StreamOpen opens a rolling-window streaming decoder for this session.
// Currently whisper-only at the C-ABI level. stepMs (default 3000) is
// how often to commit a partial transcript; lengthMs (default 10000) is
// the rolling window; keepMs (default 200) is the trailing audio carried.
func (s *CrispasrSession) StreamOpen(stepMs, lengthMs, keepMs int, language string, translate bool) (*Stream, error) {
	clang := C.CString(language)
	defer C.free(unsafe.Pointer(clang))
	tr := C.int(0)
	if translate {
		tr = 1
	}
	h := C.crispasr_session_stream_open(s.handle, C.int(4), C.int(stepMs), C.int(lengthMs), C.int(keepMs), clang, tr)
	if h == nil {
		return nil, errors.New("crispasr_session_stream_open failed (whisper-only today)")
	}
	return &Stream{handle: h}, nil
}

// Feed pushes 16 kHz mono float32 PCM. Returns 0 if still buffering, 1 if
// a new partial transcript is ready (call GetText).
func (st *Stream) Feed(pcm []float32) (int, error) {
	if len(pcm) == 0 {
		return 0, nil
	}
	rc := C.crispasr_stream_feed(st.handle, (*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)))
	if rc < 0 {
		return 0, errors.New("crispasr_stream_feed failed")
	}
	return int(rc), nil
}

// GetText returns the latest committed transcript + absolute audio-time bounds.
func (st *Stream) GetText() (StreamingUpdate, error) {
	var buf [8192]C.char
	var t0, t1 C.double
	var counter C.longlong
	rc := C.crispasr_stream_get_text(st.handle, &buf[0], C.int(len(buf)), &t0, &t1, &counter)
	if rc < 0 {
		return StreamingUpdate{}, errors.New("crispasr_stream_get_text failed")
	}
	return StreamingUpdate{
		Text:    C.GoString(&buf[0]),
		T0:      float64(t0),
		T1:      float64(t1),
		Counter: int64(counter),
	}, nil
}

// Flush finalises any remaining buffered audio.
func (st *Stream) Flush() error {
	if rc := C.crispasr_stream_flush(st.handle); rc < 0 {
		return errors.New("crispasr_stream_flush failed")
	}
	return nil
}

// Close releases the streaming handle.
func (st *Stream) Close() {
	if st != nil && st.handle != nil {
		C.crispasr_stream_close(st.handle)
		st.handle = nil
	}
}

// DetectLanguage auto-detects the spoken language on raw 16 kHz mono PCM.
// method: 0=Whisper, 1=Silero (default), 2=Firered, 3=Ecapa.
// Returns the ISO 639-1 code and the model's confidence in [0, 1].
func (s *CrispasrSession) DetectLanguage(pcm []float32, lidModelPath string, method int) (string, float32, error) {
	cpath := C.CString(lidModelPath)
	defer C.free(unsafe.Pointer(cpath))
	var outLang [16]C.char
	var outProb C.float
	pcmPtr := (*C.float)(nil)
	if len(pcm) > 0 {
		pcmPtr = (*C.float)(unsafe.Pointer(&pcm[0]))
	}
	rc := C.crispasr_session_detect_language(s.handle, pcmPtr, C.int(len(pcm)), cpath, C.int(method),
		&outLang[0], C.int(len(outLang)), &outProb)
	if rc != 0 {
		return "", 0, errors.New("crispasr_session_detect_language failed")
	}
	return C.GoString(&outLang[0]), float32(outProb), nil
}

// SetCodecPath loads a separate codec GGUF.
// Required for qwen3-tts (12 Hz tokenizer) and orpheus (SNAC codec);
// no-op for other backends.
func (s *CrispasrSession) SetCodecPath(path string) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	rc := C.crispasr_session_set_codec_path(s.handle, cpath)
	if rc != 0 {
		return errors.New("crispasr_session_set_codec_path failed")
	}
	return nil
}

// SetVoice loads a voice prompt: a baked GGUF voice pack OR a *.wav reference.
// `refText` is required for qwen3-tts when `path` is a WAV; pass an empty
// string otherwise.
//
// For orpheus voice selection is BY NAME — use SetSpeakerName instead.
func (s *CrispasrSession) SetVoice(path, refText string) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var rtPtr *C.char
	if refText != "" {
		crt := C.CString(refText)
		defer C.free(unsafe.Pointer(crt))
		rtPtr = crt
	}
	rc := C.crispasr_session_set_voice(s.handle, cpath, rtPtr)
	if rc != 0 {
		return errors.New("crispasr_session_set_voice failed")
	}
	return nil
}

// SetSpeakerName selects a fixed/preset speaker by NAME for backends
// that bake speaker names into the GGUF (orpheus today). Names are
// e.g. "tara"/"leo" for the canopylabs English finetune; "Anton"/"Sophie"
// for the Kartoffel_Orpheus DE finetunes. Use Speakers() to enumerate.
func (s *CrispasrSession) SetSpeakerName(name string) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	rc := C.crispasr_session_set_speaker_name(s.handle, cname)
	switch rc {
	case 0:
		return nil
	case -2:
		return fmt.Errorf("unknown speaker %q; call Speakers() to enumerate", name)
	case -3:
		return errors.New("backend has no preset speakers; use SetVoice instead")
	default:
		return fmt.Errorf("crispasr_session_set_speaker_name failed (rc=%d)", int(rc))
	}
}

// SetSpeakerID selects a speaker by integer index for multi-speaker
// TTS backends (melotts, piper, fastpitch). Index is 0-based; valid
// range is [0, NSpeakers()-1]. Use SetSpeakerName for name-based
// backends like orpheus.
func (s *CrispasrSession) SetSpeakerID(id int) error {
	rc := C.crispasr_session_set_speaker_id(s.handle, C.int(id))
	switch rc {
	case 0:
		return nil
	case -2:
		return fmt.Errorf("speaker id %d out of range; call NSpeakers() to check", id)
	case -3:
		return errors.New("backend has no integer-speaker contract; use SetSpeakerName instead")
	default:
		return fmt.Errorf("crispasr_session_set_speaker_id failed (rc=%d)", int(rc))
	}
}

// NSpeakers returns the number of preset speakers for the active backend.
// Works for both name-based (orpheus, qwen3-tts) and index-based
// (melotts, piper, fastpitch) backends.
func (s *CrispasrSession) NSpeakers() int {
	return int(C.crispasr_session_n_speakers(s.handle))
}

// SetInstruct sets the natural-language voice description for
// instruct-tuned TTS backends (qwen3-tts VoiceDesign today).
// Required before Synthesize when the loaded backend is VoiceDesign.
// Detect via IsVoiceDesign().
func (s *CrispasrSession) SetInstruct(instruct string) error {
	cins := C.CString(instruct)
	defer C.free(unsafe.Pointer(cins))
	rc := C.crispasr_session_set_instruct(s.handle, cins)
	switch rc {
	case 0:
		return nil
	case -3:
		return errors.New("backend is not a VoiceDesign variant; SetInstruct only applies to qwen3-tts VoiceDesign")
	default:
		return fmt.Errorf("crispasr_session_set_instruct failed (rc=%d)", int(rc))
	}
}

// SetTTSPhonemes synthesizes the given phonemes verbatim instead of
// phonemizing the text — the seam between text processing and the acoustic
// model. Use it to reproduce another implementation's pronunciation exactly, or
// to tell a G2P bug from a model bug (#316). Empty clears.
// Honoured by kokoro and piper; other backends return a soft no-op error.
func (s *CrispasrSession) SetTTSPhonemes(phonemes string) error {
	cps := C.CString(phonemes)
	defer C.free(unsafe.Pointer(cps))
	rc := C.crispasr_session_set_tts_phonemes(s.handle, cps)
	switch rc {
	case 0:
		return nil
	case -2:
		return errors.New("backend has no phonemes-in entry point; SetTTSPhonemes applies to kokoro and piper")
	default:
		return fmt.Errorf("SetTTSPhonemes failed (rc=%d)", int(rc))
	}
}

// SetTTSPadSilenceMs pads N ms of silence at the beginning of TTS output.
// Useful to bypass VLC playback bugs where it drops the first ~1.5s of audio
// while parsing a large C2PA chunk.
func (s *CrispasrSession) SetTTSPadSilenceMs(ms int) {
	C.crispasr_session_set_tts_pad_silence_ms(s.handle, C.int(ms))
}

// IsCustomVoice reports whether the loaded model is a qwen3-tts
// CustomVoice variant (use SetSpeakerName for it).
func (s *CrispasrSession) IsCustomVoice() bool {
	return C.crispasr_session_is_custom_voice(s.handle) != 0
}

// IsVoiceDesign reports whether the loaded model is a qwen3-tts
// VoiceDesign variant (use SetInstruct for it).
func (s *CrispasrSession) IsVoiceDesign() bool {
	return C.crispasr_session_is_voice_design(s.handle) != 0
}

// Speakers returns the list of preset speaker names for the active
// backend. Empty if the backend has no preset-speaker contract.
func (s *CrispasrSession) Speakers() []string {
	n := int(C.crispasr_session_n_speakers(s.handle))
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		ptr := C.crispasr_session_get_speaker_name(s.handle, C.int(i))
		if ptr != nil {
			out = append(out, C.GoString(ptr))
		}
	}
	return out
}

// Synthesize converts `text` to 24 kHz mono PCM. Requires a TTS-capable
// backend (kokoro / vibevoice / qwen3-tts / orpheus / tada).
func (s *CrispasrSession) Synthesize(text string) ([]float32, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	var n C.int
	ptr := C.crispasr_session_synthesize(s.handle, ctext, &n)
	if ptr == nil || n <= 0 {
		return nil, errors.New("crispasr_session_synthesize: no audio produced")
	}
	defer C.crispasr_pcm_free(ptr)
	samples := make([]float32, int(n))
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), int(n))
	copy(samples, src)
	return samples, nil
}

// SetSpeakerIdentity declares whose voice a PRESET voice is: "real_person",
// "synthetic" or "unknown".
//
// Cloning is not the only way to produce a deep fake. A preset voice shipped
// inside a model can be an identifiable individual — a named donor, a corpus
// speaker — and EU AI Act Art. 3(60) attaches to the audio resembling that
// person, not to which pipeline made it. Setting real_person makes the
// Art. 50(4) reminder fire for a non-cloned voice.
//
// It does NOT require a consent attestation: whether that donor agreed to the
// model being trained is a licensing matter settled upstream that you cannot
// attest to.
//
// Returns an error on an unrecognised value rather than silently downgrading
// it to "unknown".
func (s *CrispasrSession) SetSpeakerIdentity(identity string) error {
	if s == nil || s.handle == nil {
		return errors.New("session is closed")
	}
	cid := C.CString(identity)
	defer C.free(unsafe.Pointer(cid))
	switch rc := C.crispasr_session_set_speaker_identity(s.handle, cid); rc {
	case 0:
		return nil
	case -2:
		return fmt.Errorf("unrecognised speaker_identity %q (expected real_person, synthetic or unknown)", identity)
	default:
		return fmt.Errorf("crispasr_session_set_speaker_identity failed (rc=%d)", int(rc))
	}
}

// AcceptMarkingResponsibility attests that the caller accepts AI-content
// marking/disclosure responsibility (EU AI Act Art. 50). It is REQUIRED before
// SynthesizeRaw will return UNMARKED audio; the default Synthesize (watermarked)
// is unaffected. `attestation` is a human-readable affirmation recorded for audit.
func (s *CrispasrSession) AcceptMarkingResponsibility(attestation string) error {
	ca := C.CString(attestation)
	defer C.free(unsafe.Pointer(ca))
	if C.crispasr_session_accept_marking_responsibility(s.handle, ca) != 0 {
		return errors.New("crispasr_session_accept_marking_responsibility failed")
	}
	return nil
}

// SynthesizeRaw converts `text` to UNMARKED 24 kHz mono PCM (no watermark), for
// callers that must post-process (speed/mix/concat) before embedding the mark
// themselves. It is hard-refused unless AcceptMarkingResponsibility was called
// first. Most callers should use Synthesize, which watermarks by default.
func (s *CrispasrSession) SynthesizeRaw(text string) ([]float32, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	var n C.int
	ptr := C.crispasr_session_synthesize_raw(s.handle, ctext, &n)
	if ptr == nil || n <= 0 {
		return nil, errors.New("crispasr_session_synthesize_raw: no audio (attestation required? call AcceptMarkingResponsibility first)")
	}
	defer C.crispasr_pcm_free(ptr)
	samples := make([]float32, int(n))
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), int(n))
	copy(samples, src)
	return samples, nil
}

// SpeechToSpeechResult holds the output of a speech-to-speech call.
type SpeechToSpeechResult struct {
	PCM        []float32 // output audio at 24 kHz mono
	Transcript string    // intermediate ASR transcript (may be empty)
}

// SpeechToSpeech runs end-to-end audio-in → audio-out on backends with
// S2S capability (lfm2-audio, mini-omni2, sidon, voxcpm2-vae). Input is
// 16 kHz mono float32 PCM.
func (s *CrispasrSession) SpeechToSpeech(samples []float32) (*SpeechToSpeechResult, error) {
	if len(samples) == 0 {
		return nil, errors.New("SpeechToSpeech: empty input")
	}
	var textOut *C.char
	var nOut C.int
	ptr := C.crispasr_session_speech_to_speech(
		s.handle,
		(*C.float)(unsafe.Pointer(&samples[0])),
		C.int(len(samples)),
		&textOut, &nOut)
	if ptr == nil || nOut <= 0 {
		return nil, errors.New("crispasr_session_speech_to_speech: no audio produced " +
			"(separation models like htdemucs / mel-band-roformer are not S2S — use Separate() instead, #359)")
	}
	defer C.crispasr_pcm_free(ptr)
	out := make([]float32, int(nOut))
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), int(nOut))
	copy(out, src)
	var transcript string
	if textOut != nil {
		transcript = C.GoString(textOut)
		C.crispasr_session_translate_text_free(textOut)
	}
	return &SpeechToSpeechResult{PCM: out, Transcript: transcript}, nil
}

// InputSampleRate is the rate (Hz) this backend expects for input PCM.
//
// SpeechToSpeech and the other PCM entry points want audio at the backend's
// native rate, and it varies by backend — so "resample to the native rate"
// was not actionable from Go, C# or Ruby until this was bound (#321).
func (s *CrispasrSession) InputSampleRate() int {
	return int(C.crispasr_session_input_sample_rate(s.handle))
}

// OutputSampleRate is the rate (Hz) of PCM this backend returns — what
// SpeechToSpeech and Synthesize hand back. 24 kHz for conversational S2S,
// 48 kHz for Sidon and VoxCPM2 AudioVAE.
func (s *CrispasrSession) OutputSampleRate() int {
	return int(C.crispasr_session_output_sample_rate(s.handle))
}

// Stem is one separated source from Separate: its name ("vocals", "drums", …)
// and interleaved-stereo PCM at SeparateSampleRate.
type Stem struct {
	Name string
	PCM  []float32
}

// Separate splits stereo audio into its stems (#359).
//
// This is the verb for htdemucs and mel-band-roformer. They are not
// speech-to-speech models, so SpeechToSpeech returns no audio for them — the C
// ABI has always had a separate five-function surface, and it was not bound
// here.
//
// pcmStereo is INTERLEAVED stereo at the model's own rate, which is not
// 16 kHz: read SeparateSampleRate after loading (44100 for the shipped
// separation models). Each stem comes back interleaved stereo. The C side owns
// the stem buffers only until the next call, so they are copied out here.
func (s *CrispasrSession) Separate(pcmStereo []float32) ([]Stem, error) {
	if len(pcmStereo) < 2 {
		return nil, errors.New("Separate: needs interleaved stereo PCM")
	}
	// The C API counts PER-CHANNEL frames, not floats.
	nFrames := len(pcmStereo) / 2
	nStems := C.crispasr_session_separate(
		s.handle,
		(*C.float)(unsafe.Pointer(&pcmStereo[0])),
		C.int(nFrames))
	if nStems <= 0 {
		return nil, errors.New("crispasr_session_separate: no stems produced (is this a separation model?)")
	}
	stems := make([]Stem, 0, int(nStems))
	for i := 0; i < int(nStems); i++ {
		name := C.GoString(C.crispasr_session_separate_stem_name(s.handle, C.int(i)))
		if name == "" {
			name = fmt.Sprintf("stem%d", i)
		}
		var nOut C.int
		ptr := C.crispasr_session_separate_stem(s.handle, C.int(i), &nOut)
		if ptr == nil || nOut <= 0 {
			return nil, fmt.Errorf("crispasr_session_separate: stem %d (%s) came back empty", i, name)
		}
		// nOut is per-channel; the buffer is interleaved stereo.
		n := int(nOut) * 2
		out := make([]float32, n)
		copy(out, unsafe.Slice((*float32)(unsafe.Pointer(ptr)), n))
		stems = append(stems, Stem{Name: name, PCM: out})
	}
	return stems, nil
}

// SeparateSampleRate is the rate (Hz) of the stems from Separate, and the rate
// its input must be at. 0 before a separation backend is loaded.
func (s *CrispasrSession) SeparateSampleRate() int {
	return int(C.crispasr_session_separate_sample_rate(s.handle))
}

// KokoroResolved is the result of KokoroResolveForLang. Mirrors the
// Python wrapper's KokoroResolved dataclass and the Rust crate's
// kokoro_resolve_for_lang() return type.
type KokoroResolved struct {
	ModelPath       string // path to actually load (may differ from input)
	VoicePath       string // fallback voice path; empty if not applicable
	VoiceName       string // basename of the picked voice (e.g. "df_victoria")
	BackboneSwapped bool   // true iff the model path was rewritten
}

// ---------------------------------------------------------------------------
// ASR Transcription (PLAN #59 — the most critical missing capability).
// ---------------------------------------------------------------------------

// TranscribeResult holds the output of a transcription call.
type TranscribeResult struct {
	Segments []TranscribeSegment
}

// TranscribeSegment is one segment from a transcription.
type TranscribeSegment struct {
	Text  string
	T0    int64 // centiseconds
	T1    int64
	Words []TranscribeWord
	// NoSpeechProb is Whisper's per-segment no-speech probability (the
	// <|nospeech|> posterior) in [0, 1]. Whisper-only; other backends leave
	// the -1.0 "no data" sentinel.
	NoSpeechProb float32
	// Speaker is a native per-segment speaker label from a backend that
	// diarizes on its own, in the "(Speaker N) " form the CLI prefixes into
	// text/srt/vtt output, or "" when the backend produced none. Populated
	// today by vibevoice, whose model answers with a Start/End/Speaker/Content
	// array. The ordinals are CHUNK-LOCAL: "Speaker 1" in one transcribe call
	// is not necessarily the same voice as "Speaker 1" in the next.
	Speaker string
}

// TranscribeWord is one word with timing and confidence.
type TranscribeWord struct {
	Text string
	T0   int64 // centiseconds
	T1   int64
	P    float32 // confidence
}

// CtcLogits holds the per-frame CTC logits captured from a CTC backend (Omni
// CTC, wav2vec2/hubert/data2vec, or canary-ctc). Data is frame-major:
// Data[t*NVocab + v] is the score for vocabulary entry v at encoder frame t, so
// len(Data) == NFrames*NVocab. The Omni and wav2vec2 grids are raw logits
// (pre-softmax); the canary-ctc grid is log-probabilities. Produced only by
// TranscribeWithLogits; other backends yield no grid.
type CtcLogits struct {
	NVocab  int
	NFrames int
	Data    []float32
}

// Transcribe runs ASR on 16 kHz mono float32 PCM.
func (s *CrispasrSession) Transcribe(pcm []float32) (*TranscribeResult, error) {
	return s.TranscribeLang(pcm, "")
}

// TranscribeLang transcribes with an explicit language hint (e.g. "en", "de").
// Pass empty string for auto-detect.
func (s *CrispasrSession) TranscribeLang(pcm []float32, lang string) (*TranscribeResult, error) {
	if s.handle == nil {
		return nil, errors.New("session is closed")
	}
	pcmPtr := (*C.float)(nil)
	if len(pcm) > 0 {
		pcmPtr = (*C.float)(unsafe.Pointer(&pcm[0]))
	}
	var clang *C.char
	if lang != "" {
		clang = C.CString(lang)
		defer C.free(unsafe.Pointer(clang))
	}
	r := C.crispasr_session_transcribe_lang(s.handle, pcmPtr, C.int(len(pcm)), clang)
	if r == nil {
		return nil, errors.New("transcription failed")
	}
	defer C.crispasr_session_result_free(r)
	return extractResult(r), nil
}

// TranscribeWithLogits transcribes 16 kHz mono float32 PCM and also returns the
// per-frame CTC logits captured for this call. It opts logit capture in for
// the duration (no prior SetReturnLogits needed). The returned *CtcLogits is
// nil for backends that don't produce a dense CTC grid or when the transcript
// is empty.
func (s *CrispasrSession) TranscribeWithLogits(pcm []float32) (*TranscribeResult, *CtcLogits, error) {
	if s.handle == nil {
		return nil, nil, errors.New("session is closed")
	}
	if len(pcm) == 0 {
		return &TranscribeResult{}, nil, nil
	}
	if err := s.SetReturnLogits(true); err != nil {
		return nil, nil, err
	}
	defer s.SetReturnLogits(false)
	pcmPtr := (*C.float)(unsafe.Pointer(&pcm[0]))
	r := C.crispasr_session_transcribe(s.handle, pcmPtr, C.int(len(pcm)))
	if r == nil {
		return nil, nil, errors.New("transcription failed")
	}
	defer C.crispasr_session_result_free(r)
	return extractResult(r), extractLogits(r), nil
}

// TranscribeChunked runs chunked-encode ASR (issue #208): it forces the
// Parakeet backend through its bounded overlapping-window long-form path so
// long audio transcribes in bounded time without dropping sections. Inert
// (== TranscribeLang) on non-Parakeet backends. chunkSeconds <= 0 keeps the
// per-model default window; overlapSeconds < 0 keeps the default overlap.
// Poll GetProgress() (0..100) from another goroutine to render progress.
func (s *CrispasrSession) TranscribeChunked(pcm []float32, chunkSeconds, overlapSeconds int, lang string) (*TranscribeResult, error) {
	if s.handle == nil {
		return nil, errors.New("session is closed")
	}
	pcmPtr := (*C.float)(nil)
	if len(pcm) > 0 {
		pcmPtr = (*C.float)(unsafe.Pointer(&pcm[0]))
	}
	var clang *C.char
	if lang != "" {
		clang = C.CString(lang)
		defer C.free(unsafe.Pointer(clang))
	}
	r := C.crispasr_session_transcribe_chunked_lang(s.handle, pcmPtr, C.int(len(pcm)),
		C.int(chunkSeconds), C.int(overlapSeconds), clang)
	if r == nil {
		return nil, errors.New("chunked transcription failed")
	}
	defer C.crispasr_session_result_free(r)
	return extractResult(r), nil
}

// GetProgress polls long-form (chunked) transcription progress: 0..100, or -1
// when idle. Updated in lockstep with TranscribeChunked windows (issue #208),
// so a UI goroutine can render a progress bar without a callback.
func GetProgress() int {
	return int(C.crispasr_get_progress())
}

// TranscribeVAD transcribes with VAD segmentation.
// vadModelPath can be empty for auto-download of default Silero model.
func (s *CrispasrSession) TranscribeVAD(pcm []float32, sampleRate int, vadModelPath string) (*TranscribeResult, error) {
	if s.handle == nil {
		return nil, errors.New("session is closed")
	}
	pcmPtr := (*C.float)(nil)
	if len(pcm) > 0 {
		pcmPtr = (*C.float)(unsafe.Pointer(&pcm[0]))
	}
	cvad := C.CString(vadModelPath)
	defer C.free(unsafe.Pointer(cvad))
	r := C.crispasr_session_transcribe_vad(s.handle, pcmPtr, C.int(len(pcm)), C.int(sampleRate), cvad, nil)
	if r == nil {
		return nil, errors.New("transcription with VAD failed")
	}
	defer C.crispasr_session_result_free(r)
	return extractResult(r), nil
}

func extractResult(r *C.crispasr_session_result) *TranscribeResult {
	nSegs := int(C.crispasr_session_result_n_segments(r))
	result := &TranscribeResult{Segments: make([]TranscribeSegment, nSegs)}
	for i := 0; i < nSegs; i++ {
		seg := &result.Segments[i]
		seg.Text = C.GoString(C.crispasr_session_result_segment_text(r, C.int(i)))
		seg.T0 = int64(C.crispasr_session_result_segment_t0(r, C.int(i)))
		seg.T1 = int64(C.crispasr_session_result_segment_t1(r, C.int(i)))
		seg.NoSpeechProb = float32(C.crispasr_session_result_segment_no_speech_prob(r, C.int(i)))
		seg.Speaker = C.GoString(C.crispasr_session_result_segment_speaker(r, C.int(i)))
		nWords := int(C.crispasr_session_result_n_words(r, C.int(i)))
		seg.Words = make([]TranscribeWord, nWords)
		for j := 0; j < nWords; j++ {
			w := &seg.Words[j]
			w.Text = C.GoString(C.crispasr_session_result_word_text(r, C.int(i), C.int(j)))
			w.T0 = int64(C.crispasr_session_result_word_t0(r, C.int(i), C.int(j)))
			w.T1 = int64(C.crispasr_session_result_word_t1(r, C.int(i), C.int(j)))
			w.P = float32(C.crispasr_session_result_word_p(r, C.int(i), C.int(j)))
		}
	}
	return result
}

// extractLogits lifts out the CTC logits attached to a result (see
// CtcLogits) into a Go-owned slice before the result is freed. The C buffer is
// owned by the result, so the data is copied out here. Returns nil unless the
// session opted in via SetReturnLogits and the backend produced a grid.
func extractLogits(r *C.crispasr_session_result) *CtcLogits {
	nFrames := int(C.crispasr_session_result_n_logit_frames(r))
	nVocab := int(C.crispasr_session_result_n_logit_vocab(r))
	ptr := C.crispasr_session_result_logits(r)
	if nFrames <= 0 || nVocab <= 0 || ptr == nil {
		return nil
	}
	n := nFrames * nVocab
	data := make([]float32, n)
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), n)
	copy(data, src)
	return &CtcLogits{NVocab: nVocab, NFrames: nFrames, Data: data}
}

// CtcVocab returns the Omni CTC vocabulary as raw pieces indexed by token id
// (vocab[id]). Pieces keep their word-boundary marker intact (the v2 Omni vocab
// uses a literal space, v1 uses U+2581), so a consumer can detokenize a greedy
// CTC decode over the grid from TranscribeWithLogits. Returns nil for backends
// that don't expose a CTC vocab.
func (s *CrispasrSession) CtcVocab() []string {
	if s.handle == nil {
		return nil
	}
	n := int(C.crispasr_session_n_vocab(s.handle))
	if n <= 0 {
		return nil
	}
	vocab := make([]string, n)
	for i := 0; i < n; i++ {
		vocab[i] = C.GoString(C.crispasr_session_token_text(s.handle, C.int(i)))
	}
	return vocab
}

// ---------------------------------------------------------------------------
// Forced alignment — word-level timestamps from transcript + audio
// ---------------------------------------------------------------------------

// AlignedWord holds one word from forced alignment.
type AlignedWord struct {
	Text string
	T0   int64 // centiseconds
	T1   int64
}

// AlignWords runs CTC forced alignment on a transcript + audio pair.
// alignerModel is the path to a CTC aligner GGUF (e.g. canary-ctc-aligner.gguf).
func AlignWords(alignerModel, transcript string, pcm []float32, tOffsetCs int64, nThreads int) ([]AlignedWord, error) {
	cm := C.CString(alignerModel)
	defer C.free(unsafe.Pointer(cm))
	ct := C.CString(transcript)
	defer C.free(unsafe.Pointer(ct))
	pcmPtr := (*C.float)(nil)
	if len(pcm) > 0 {
		pcmPtr = (*C.float)(unsafe.Pointer(&pcm[0]))
	}
	r := C.crispasr_align_words_abi(cm, ct, pcmPtr, C.int(len(pcm)), C.longlong(tOffsetCs), C.int(nThreads))
	if r == nil {
		return nil, errors.New("alignment failed")
	}
	defer C.crispasr_align_result_free(r)
	n := int(C.crispasr_align_result_n_words(r))
	words := make([]AlignedWord, n)
	for i := 0; i < n; i++ {
		words[i] = AlignedWord{
			Text: C.GoString(C.crispasr_align_result_word_text(r, C.int(i))),
			T0:   int64(C.crispasr_align_result_word_t0(r, C.int(i))),
			T1:   int64(C.crispasr_align_result_word_t1(r, C.int(i))),
		}
	}
	return words, nil
}

// ---------------------------------------------------------------------------
// Standalone language detection (not session-bound)
// ---------------------------------------------------------------------------

// DetectLanguagePCM detects the spoken language from raw 16 kHz mono PCM.
// method: 0=Whisper, 1=Silero, 2=Firered, 3=Ecapa.
// modelPath can be empty for auto-download of the default model.
func DetectLanguagePCM(pcm []float32, method int, modelPath string, nThreads int) (string, float32, error) {
	pcmPtr := (*C.float)(nil)
	if len(pcm) > 0 {
		pcmPtr = (*C.float)(unsafe.Pointer(&pcm[0]))
	}
	cm := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cm))
	var outLang [16]C.char
	var outProb C.float
	rc := C.crispasr_detect_language_pcm(pcmPtr, C.int(len(pcm)), C.int(method), cm,
		C.int(nThreads), 0, 0, 0, &outLang[0], C.int(len(outLang)), &outProb)
	if rc != 0 {
		return "", 0, errors.New("language detection failed")
	}
	return C.GoString(&outLang[0]), float32(outProb), nil
}

// ---------------------------------------------------------------------------
// Standalone VAD — speech segment detection
// ---------------------------------------------------------------------------

// VADSpan is one speech segment detected by VAD.
type VADSpan struct {
	T0 float64 // seconds
	T1 float64
}

// VADSegments runs standalone VAD on 16 kHz mono PCM.
// vadModelPath can be empty for auto-download of default Silero model.
// Returns detected speech spans.
func VADSegments(vadModelPath string, pcm []float32, sampleRate int, threshold float32,
	minSpeechMs, minSilenceMs, nThreads int) ([]VADSpan, error) {
	cm := C.CString(vadModelPath)
	defer C.free(unsafe.Pointer(cm))
	pcmPtr := (*C.float)(nil)
	if len(pcm) > 0 {
		pcmPtr = (*C.float)(unsafe.Pointer(&pcm[0]))
	}
	var outSpans *C.float
	n := C.crispasr_vad_segments(cm, pcmPtr, C.int(len(pcm)), C.int(sampleRate),
		C.float(threshold), C.int(minSpeechMs), C.int(minSilenceMs), C.int(nThreads), 0, &outSpans)
	if n < 0 {
		return nil, fmt.Errorf("VAD failed (rc=%d)", int(n))
	}
	if n == 0 || outSpans == nil {
		return nil, nil
	}
	defer C.crispasr_vad_free(outSpans)
	// outSpans is a flat array of [t0, t1, t0, t1, ...] with n pairs
	spans := make([]VADSpan, int(n))
	raw := unsafe.Slice((*float32)(unsafe.Pointer(outSpans)), int(n)*2)
	for i := 0; i < int(n); i++ {
		spans[i] = VADSpan{T0: float64(raw[i*2]), T1: float64(raw[i*2+1])}
	}
	return spans, nil
}

// ---------------------------------------------------------------------------
// Speaker diarization
// ---------------------------------------------------------------------------

// DiarizeMethod selects the diarization algorithm.
type DiarizeMethod int

const (
	DiarizeEnergy   DiarizeMethod = 0 // stereo-only, energy-based
	DiarizeXCorr    DiarizeMethod = 1 // stereo-only, cross-correlation
	DiarizeVADTurns DiarizeMethod = 2 // mono-friendly, gap-based
	DiarizePyannote DiarizeMethod = 3 // pyannote v3 segmentation model
	// DiarizeMethodFoxNose: WeSpeaker embeddings + spectral clustering (#324).
	// Requires FoxNoseOpts.EmbedderPath.
	DiarizeMethodFoxNose DiarizeMethod = 4
)

// DiarizeSeg is one input/output segment for diarization.
type DiarizeSeg struct {
	T0      int64 // centiseconds
	T1      int64
	Speaker int32 // output: -1 if unassigned
}

// DiarizeSegments assigns speaker labels to pre-segmented audio.
// leftPCM is the mono or left-channel audio; rightPCM is the right channel
// (nil for mono). segs are modified in-place with Speaker fields filled.
// FoxNoseOpts configures DiarizeMethodFoxNose (#324). Nil for other methods.
type FoxNoseOpts struct {
	EmbedderPath string // WeSpeaker GGUF; required
	MinSpeakers  int    // 0 -> 1
	MaxSpeakers  int    // 0 -> 8
	NumSpeakers  int    // >0 pins the count and skips estimation
}

// DiarizeSegments keeps its original signature for source compatibility;
// DiarizeSegmentsFoxNose adds the #324 options.
func DiarizeSegments(leftPCM, rightPCM []float32, isStereo bool, segs []DiarizeSeg,
	method DiarizeMethod, nThreads int, pyannoteModel string) error {
	return diarizeSegments(leftPCM, rightPCM, isStereo, segs, method, nThreads, pyannoteModel, nil, nil)
}

// DiarizeSegmentsFoxNose runs the WeSpeaker + spectral-clustering diarizer.
func DiarizeSegmentsFoxNose(leftPCM, rightPCM []float32, isStereo bool, segs []DiarizeSeg,
	nThreads int, fox *FoxNoseOpts) error {
	return diarizeSegments(leftPCM, rightPCM, isStereo, segs, DiarizeMethodFoxNose, nThreads, "", fox, nil)
}

// DiarizeTurn is one speaker turn the METHOD derived from the audio, on the
// caller's absolute timeline (slice_t0_cs already added back), independent of
// the segment grid that was passed in.
type DiarizeTurn struct {
	T0      int64 // centiseconds
	T1      int64
	Speaker int32 // dense, zero-based; never -1
}

// DiarizeSegmentsWithTurns labels segs AND returns the speaker turns the
// method derived from the audio (#395).
//
// Labelling alone can never resolve finer than the segment grid the caller
// sent in: apply_foxnose awards each segment to the turn it overlaps MOST, so
// a segment straddling a speaker change goes wholly to the majority speaker.
// The turns let a caller split such a segment itself. Sending a finer grid is
// not an alternative — FoxNose skips spans under kMinSegmentSeconds (0.4 s).
//
// Only DiarizeMethodFoxNose derives turns; the others return an empty slice,
// which is not an error.
func DiarizeSegmentsWithTurns(leftPCM, rightPCM []float32, isStereo bool, segs []DiarizeSeg,
	method DiarizeMethod, nThreads int, pyannoteModel string, fox *FoxNoseOpts) ([]DiarizeTurn, error) {
	var turns []DiarizeTurn
	err := diarizeSegments(leftPCM, rightPCM, isStereo, segs, method, nThreads, pyannoteModel, fox, &turns)
	return turns, err
}

func diarizeSegments(leftPCM, rightPCM []float32, isStereo bool, segs []DiarizeSeg,
	method DiarizeMethod, nThreads int, pyannoteModel string, fox *FoxNoseOpts,
	outTurns *[]DiarizeTurn) error {
	if len(segs) == 0 {
		return nil
	}
	leftPtr := (*C.float)(unsafe.Pointer(&leftPCM[0]))
	var rightPtr *C.float
	nSamples := len(leftPCM)
	if isStereo && len(rightPCM) > 0 {
		rightPtr = (*C.float)(unsafe.Pointer(&rightPCM[0]))
	}
	cSegs := make([]C.struct_crispasr_diarize_seg_abi, len(segs))
	for i, s := range segs {
		cSegs[i].t0_cs = C.longlong(s.T0)
		cSegs[i].t1_cs = C.longlong(s.T1)
		cSegs[i].speaker = C.int(s.Speaker)
	}
	var cPyannote *C.char
	if pyannoteModel != "" {
		cPyannote = C.CString(pyannoteModel)
		defer C.free(unsafe.Pointer(cPyannote))
	}
	opts := C.struct_crispasr_diarize_opts_abi{
		method:              C.int(method),
		n_threads:           C.int(nThreads),
		slice_t0_cs:         0,
		pyannote_model_path: cPyannote,
	}
	if fox != nil {
		// DiarizeMethodFoxNose (#324) consumes the embedder path and speaker
		// bounds; every other method ignores them.
		if fox.EmbedderPath != "" {
			cEmb := C.CString(fox.EmbedderPath)
			defer C.free(unsafe.Pointer(cEmb))
			opts.foxnose_embedder_path = cEmb
		}
		opts.min_speakers = C.int(fox.MinSpeakers)
		opts.max_speakers = C.int(fox.MaxSpeakers)
		opts.num_speakers = C.int(fox.NumSpeakers)
	}
	stereo := C.int(0)
	if isStereo {
		stereo = 1
	}
	var rc C.int
	if outTurns == nil {
		rc = C.crispasr_diarize_segments_abi(leftPtr, rightPtr, C.int(nSamples), stereo,
			&cSegs[0], C.int(len(cSegs)), &opts)
	} else {
		// Size the turn buffer the way the Rust wrapper does: FoxNose's
		// embedding hop is 0.6 s and no turn is shorter than one hop, so one
		// slot per 0.5 s of audio is an over-estimate. That pair of braces is
		// the belt; the single retry below is the suspenders — rc 2 means the
		// buffer was short and n_turns holds the capacity actually needed.
		turnCap := int(math.Ceil(float64(nSamples)/16000.0/0.5)) + len(segs) + 16
		for attempt := 0; ; attempt++ {
			if turnCap < 1 {
				turnCap = 1
			}
			cTurns := make([]C.struct_crispasr_diarize_turn_abi, turnCap)
			var nTurns C.int
			rc = C.crispasr_diarize_segments_turns_abi(leftPtr, rightPtr, C.int(nSamples), stereo,
				&cSegs[0], C.int(len(cSegs)), &opts, &cTurns[0], C.int(turnCap), &nTurns)
			if rc == 2 && attempt == 0 {
				turnCap = int(nTurns)
				continue
			}
			if rc == 0 {
				n := int(nTurns)
				if n > turnCap {
					n = turnCap
				}
				*outTurns = make([]DiarizeTurn, n)
				for i := 0; i < n; i++ {
					(*outTurns)[i] = DiarizeTurn{
						T0:      int64(cTurns[i].t0_cs),
						T1:      int64(cTurns[i].t1_cs),
						Speaker: int32(cTurns[i].speaker),
					}
				}
			}
			break
		}
	}
	if rc != 0 {
		return fmt.Errorf("diarize failed (rc=%d)", int(rc))
	}
	for i := range segs {
		segs[i].Speaker = int32(cSegs[i].speaker)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Pluggable speaker embedder + cosine clustering + pyannote cache (#107 P6).
// Same building blocks the CLI's --diarize-embedder path uses; expose them
// here so Go callers can compose the diarize pipeline without shelling out.
// ---------------------------------------------------------------------------

// SpeakerEmbedder wraps a pluggable speaker-embedding model.
//
// Known aliases (case-insensitive):
//
//	"auto" / "titanet"                            -> TitaNet-Large (192-d)
//	"indextts" / "indextts-bigvgan" / "ecapa"     -> IndexTTS-BigVGAN ECAPA-TDNN (512-d)
//	any .gguf path                                -> TitaNet (or IndexTTS if "indextts" in name)
//
// Always call Close() — the C-side context owns model weights.
type SpeakerEmbedder struct {
	handle unsafe.Pointer
}

// NewSpeakerEmbedder builds an embedder from a CLI-style model spec.
// cacheDir overrides the auto-download cache root (empty for the
// CrispASR default of ~/.cache/crispasr/).
func NewSpeakerEmbedder(modelSpec string, nThreads int, cacheDir string) (*SpeakerEmbedder, error) {
	cSpec := C.CString(modelSpec)
	defer C.free(unsafe.Pointer(cSpec))
	cCache := C.CString(cacheDir)
	defer C.free(unsafe.Pointer(cCache))
	h := C.crispasr_speaker_embedder_make_abi(cSpec, C.int(nThreads), cCache)
	if h == nil {
		return nil, fmt.Errorf("failed to build speaker embedder %q", modelSpec)
	}
	return &SpeakerEmbedder{handle: h}, nil
}

// Dim returns the output embedding dimension (e.g. 192 for TitaNet).
func (e *SpeakerEmbedder) Dim() int {
	if e == nil || e.handle == nil {
		return 0
	}
	return int(C.crispasr_speaker_embedder_dim_abi(e.handle))
}

// Name returns the adapter name for logging.
func (e *SpeakerEmbedder) Name() string {
	if e == nil || e.handle == nil {
		return ""
	}
	cName := C.crispasr_speaker_embedder_name_abi(e.handle)
	if cName == nil {
		return ""
	}
	return C.GoString(cName)
}

// Embed extracts one embedding from mono 16 kHz float32 PCM. Returns
// nil when the underlying model rejected the input (e.g. too short).
func (e *SpeakerEmbedder) Embed(pcm16k []float32) []float32 {
	if e == nil || e.handle == nil || len(pcm16k) == 0 {
		return nil
	}
	d := e.Dim()
	if d <= 0 {
		return nil
	}
	out := make([]float32, d)
	ok := C.crispasr_speaker_embedder_embed_abi(
		e.handle,
		(*C.float)(unsafe.Pointer(&pcm16k[0])),
		C.int(len(pcm16k)),
		(*C.float)(unsafe.Pointer(&out[0])),
	)
	if ok == 0 {
		return nil
	}
	return out
}

// Close frees the C-side context. Safe to call multiple times.
func (e *SpeakerEmbedder) Close() {
	if e == nil || e.handle == nil {
		return
	}
	C.crispasr_speaker_embedder_free_abi(e.handle)
	e.handle = nil
}

// AgglomerativeCluster groups n embeddings of dimension dim into
// clusters by single-linkage cosine similarity. mergeThreshold stops
// the merging when the closest pair falls below the threshold;
// maxSpeakers is a hard cap.
//
// Returns one cluster ID per input in [0, k) assigned in
// first-appearance order.
func AgglomerativeCluster(embeddings []float32, n, dim int,
	mergeThreshold float32, maxSpeakers int) ([]int32, error) {
	if n <= 0 || dim <= 0 || len(embeddings) < n*dim {
		return nil, fmt.Errorf("invalid arguments to AgglomerativeCluster")
	}
	out := make([]int32, n)
	rc := C.crispasr_speaker_cluster_abi(
		(*C.float)(unsafe.Pointer(&embeddings[0])),
		C.int(n), C.int(dim), C.float(mergeThreshold), C.int(maxSpeakers),
		(*C.int)(unsafe.Pointer(&out[0])),
	)
	if rc < 0 {
		return nil, fmt.Errorf("crispasr_speaker_cluster_abi failed")
	}
	return out, nil
}

// PyannoteCache holds pyannote-seg posteriors over a full audio buffer.
// Compute once at the start of a pipeline, then call Apply for each
// set of segment ranges. Gives cross-slice speaker-ID consistency.
type PyannoteCache struct {
	handle unsafe.Pointer
}

// NewPyannoteCache pre-computes pyannote-seg posteriors over pcm16k.
// modelPath must point at a pyannote-seg .gguf.
func NewPyannoteCache(pcm16k []float32, modelPath string, nThreads int) (*PyannoteCache, error) {
	if len(pcm16k) == 0 {
		return nil, fmt.Errorf("empty audio")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	h := C.crispasr_pyannote_cache_compute_abi(
		(*C.float)(unsafe.Pointer(&pcm16k[0])),
		C.int(len(pcm16k)), cPath, C.int(nThreads),
	)
	if h == nil {
		return nil, fmt.Errorf("failed to compute pyannote cache from %q", modelPath)
	}
	return &PyannoteCache{handle: h}, nil
}

// Apply scores segs against the cached posteriors. Each segment's
// Speaker is set to 0/1/2 (local pyannote-seg track index) or -1 for
// silence. sliceT0Cs is the absolute centisecond at which the cache
// buffer starts (usually 0 — the cache covers the whole input audio).
func (c *PyannoteCache) Apply(segs []DiarizeSeg, sliceT0Cs int64) error {
	if c == nil || c.handle == nil {
		return fmt.Errorf("nil pyannote cache")
	}
	if len(segs) == 0 {
		return nil
	}
	cSegs := make([]C.struct_crispasr_diarize_seg_abi, len(segs))
	for i, s := range segs {
		cSegs[i].t0_cs = C.longlong(s.T0)
		cSegs[i].t1_cs = C.longlong(s.T1)
		cSegs[i].speaker = C.int(s.Speaker)
	}
	rc := C.crispasr_pyannote_cache_apply_abi(c.handle, C.longlong(sliceT0Cs),
		&cSegs[0], C.int(len(cSegs)))
	if rc != 0 {
		return fmt.Errorf("crispasr_pyannote_cache_apply_abi rc=%d", int(rc))
	}
	for i := range segs {
		segs[i].Speaker = int32(cSegs[i].speaker)
	}
	return nil
}

// Close frees the C-side cache. Safe to call multiple times.
func (c *PyannoteCache) Close() {
	if c == nil || c.handle == nil {
		return
	}
	C.crispasr_pyannote_cache_free_abi(c.handle)
	c.handle = nil
}

// ---------------------------------------------------------------------------
// Punctuation restoration (FireRedPunc post-processor)
// ---------------------------------------------------------------------------

// PuncModel wraps a loaded FireRedPunc punctuation restoration model.
type PuncModel struct {
	handle unsafe.Pointer
}

// PuncInit loads a FireRedPunc GGUF model. Pass "auto" to auto-download.
func PuncInit(modelPath string) (*PuncModel, error) {
	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))
	h := C.crispasr_punc_init(cpath)
	if h == nil {
		return nil, errors.New("crispasr_punc_init failed for " + modelPath)
	}
	return &PuncModel{handle: h}, nil
}

// Process restores punctuation + capitalization to the input text.
func (p *PuncModel) Process(text string) (string, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	result := C.crispasr_punc_process(p.handle, ctext)
	if result == nil {
		return text, errors.New("crispasr_punc_process failed")
	}
	out := C.GoString(result)
	C.crispasr_punc_free_text(result)
	return out, nil
}

// Close frees the punctuation model.
func (p *PuncModel) Close() {
	if p != nil && p.handle != nil {
		C.crispasr_punc_free(p.handle)
		p.handle = nil
	}
}

// ---------------------------------------------------------------------------
// Model registry + cache
// ---------------------------------------------------------------------------

// RegistryEntry holds the auto-download metadata for a backend.
type RegistryEntry struct {
	Filename string
	URL      string
	Size     string
}

// RegistryArtifactKind identifies an artifact's role in a default bundle.
type RegistryArtifactKind int

const (
	RegistryArtifactPrimary RegistryArtifactKind = iota
	RegistryArtifactCompanion
	RegistryArtifactExtra
)

// RegistryArtifact is one file in a backend's canonical default bundle.
type RegistryArtifact struct {
	Kind     RegistryArtifactKind
	Filename string
	URL      string
	Size     string
}

// RegistryBundle is the exact artifact set downloaded by `-m auto`.
type RegistryBundle struct {
	Backend            string
	License            string
	RequiresAcceptance bool
	Artifacts          []RegistryArtifact
}

// RegistryLookup returns the default model filename + download URL for a backend.
func RegistryLookup(backend string) (RegistryEntry, error) {
	cb := C.CString(backend)
	defer C.free(unsafe.Pointer(cb))
	var fname, url, sz [512]C.char
	rc := C.crispasr_registry_lookup_abi(cb, &fname[0], 512, &url[0], 512, &sz[0], 512)
	if rc != 0 {
		return RegistryEntry{}, fmt.Errorf("no registry entry for backend %q", backend)
	}
	return RegistryEntry{
		Filename: C.GoString(&fname[0]),
		URL:      C.GoString(&url[0]),
		Size:     C.GoString(&sz[0]),
	}, nil
}

// RegistryDefaultBundle returns the backend's exact canonical `-m auto`
// artifact bundle. It does not apply a preferred quant.
func RegistryDefaultBundle(backend string) (RegistryBundle, error) {
	cb := C.CString(backend)
	defer C.free(unsafe.Pointer(cb))
	var canonical [256]C.char
	var license [1024]C.char
	var requiresAcceptance C.int
	count := C.crispasr_registry_default_bundle_info_abi(
		cb, &canonical[0], C.int(len(canonical)), &license[0], C.int(len(license)),
		&requiresAcceptance,
	)
	if count == 0 {
		return RegistryBundle{}, fmt.Errorf("no default registry bundle for backend %q", backend)
	}
	if count < 0 {
		return RegistryBundle{}, fmt.Errorf(
			"default registry bundle lookup failed for backend %q (rc=%d)",
			backend, int(count),
		)
	}

	bundle := RegistryBundle{
		Backend:            C.GoString(&canonical[0]),
		License:            C.GoString(&license[0]),
		RequiresAcceptance: requiresAcceptance != 0,
		Artifacts:          make([]RegistryArtifact, 0, int(count)),
	}
	for index := C.int(0); index < count; index++ {
		var kind C.int
		var filename [256]C.char
		var url [2048]C.char
		var size [64]C.char
		rc := C.crispasr_registry_default_bundle_artifact_abi(
			cb, index, &kind,
			&filename[0], C.int(len(filename)),
			&url[0], C.int(len(url)),
			&size[0], C.int(len(size)),
		)
		if rc != 0 || kind < 0 || kind > 2 {
			return RegistryBundle{}, fmt.Errorf(
				"default registry bundle artifact %d failed (rc=%d, kind=%d)",
				int(index), int(rc), int(kind),
			)
		}
		bundle.Artifacts = append(bundle.Artifacts, RegistryArtifact{
			Kind:     RegistryArtifactKind(kind),
			Filename: C.GoString(&filename[0]),
			URL:      C.GoString(&url[0]),
			Size:     C.GoString(&size[0]),
		})
	}
	return bundle, nil
}

// CacheEnsureFile downloads a file into the model cache if not already present.
// Returns the local path. cacheDirOverride can be empty for the default.
func CacheEnsureFile(filename, url, cacheDirOverride string) (string, error) {
	cf := C.CString(filename)
	defer C.free(unsafe.Pointer(cf))
	cu := C.CString(url)
	defer C.free(unsafe.Pointer(cu))
	cdir := C.CString(cacheDirOverride)
	defer C.free(unsafe.Pointer(cdir))
	var out [1024]C.char
	rc := C.crispasr_cache_ensure_file_abi(cf, cu, 0, cdir, &out[0], 1024)
	if rc != 0 {
		return "", fmt.Errorf("cache_ensure_file failed for %q", filename)
	}
	return C.GoString(&out[0]), nil
}

// CacheDir returns the resolved model cache directory path.
func CacheDir(override string) (string, error) {
	co := C.CString(override)
	defer C.free(unsafe.Pointer(co))
	var out [1024]C.char
	rc := C.crispasr_cache_dir_abi(co, &out[0], 1024)
	if rc != 0 {
		return "", errors.New("crispasr_cache_dir_abi failed")
	}
	return C.GoString(&out[0]), nil
}

// KokoroResolveForLang returns the kokoro model + fallback voice that
// CrispASR's CLI would pick for `lang`. Mirrors PLAN #56 opt 2b. Wrappers
// should call this *before* SessionOpen so the routing kicks in even
// outside the CLI entry point.
func KokoroResolveForLang(modelPath, lang string) (KokoroResolved, error) {
	cmodel := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cmodel))
	clang := C.CString(lang)
	defer C.free(unsafe.Pointer(clang))

	outModel := (*C.char)(C.malloc(1024))
	defer C.free(unsafe.Pointer(outModel))
	outVoice := (*C.char)(C.malloc(1024))
	defer C.free(unsafe.Pointer(outVoice))
	outPicked := (*C.char)(C.malloc(64))
	defer C.free(unsafe.Pointer(outPicked))

	swapped := false
	rc := C.crispasr_kokoro_resolve_model_for_lang_abi(cmodel, clang, outModel, 1024)
	if rc < 0 {
		return KokoroResolved{}, errors.New("kokoro_resolve_model_for_lang: buffer too small")
	}
	if rc == 0 {
		swapped = true
	}
	resolvedModel := C.GoString(outModel)
	if resolvedModel == "" {
		resolvedModel = modelPath
	}

	rc = C.crispasr_kokoro_resolve_fallback_voice_abi(cmodel, clang, outVoice, 1024, outPicked, 64)
	if rc < 0 {
		return KokoroResolved{}, errors.New("kokoro_resolve_fallback_voice: buffer too small")
	}
	out := KokoroResolved{ModelPath: resolvedModel, BackboneSwapped: swapped}
	if rc == 0 {
		out.VoicePath = C.GoString(outVoice)
		out.VoiceName = C.GoString(outPicked)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Full C-ABI parity wrappers (PLAN #59 bindings-parity milestone)
// ---------------------------------------------------------------------------

// SessionOpenExplicit opens a session with an explicit backend name override.
func SessionOpenExplicit(modelPath, backendName string, nThreads int) (*CrispasrSession, error) {
	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))
	cbe := C.CString(backendName)
	defer C.free(unsafe.Pointer(cbe))
	h := C.crispasr_session_open_explicit(cpath, cbe, C.int(nThreads))
	if h == nil {
		return nil, fmt.Errorf("crispasr_session_open_explicit: failed to open %s (backend=%s)", modelPath, backendName)
	}
	return &CrispasrSession{handle: h}, nil
}

// AvailableBackends returns the comma-separated list of backend names
// the loaded libcrispasr was built with.
func AvailableBackends() []string {
	var buf [1024]C.char
	C.crispasr_session_available_backends(&buf[0], 1024)
	csv := C.GoString(&buf[0])
	if csv == "" {
		return nil
	}
	var out []string
	for _, s := range splitCSV(csv) {
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

// DetectedLanguage returns the acoustic language Whisper detected on the last
// transcribe as an ISO-639-1 code (e.g. "en"). Whisper-only; other backends
// return the session's source-language hint, or "unknown".
func (s *CrispasrSession) DetectedLanguage() string {
	var buf [32]C.char
	C.crispasr_session_detected_language(s.handle, &buf[0], 32)
	return C.GoString(&buf[0])
}

func splitCSV(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == ',' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

// TranslateText translates text via the session's MT-capable backend.
// Returns empty string if the backend doesn't support translation.
func (s *CrispasrSession) TranslateText(text, srcLang, tgtLang string, maxTokens int) (string, error) {
	ct := C.CString(text)
	defer C.free(unsafe.Pointer(ct))
	cs := C.CString(srcLang)
	defer C.free(unsafe.Pointer(cs))
	ctg := C.CString(tgtLang)
	defer C.free(unsafe.Pointer(ctg))
	res := C.crispasr_session_translate_text(s.handle, ct, cs, ctg, C.int(maxTokens))
	if res == nil {
		return "", nil
	}
	out := C.GoString(res)
	C.crispasr_session_translate_text_free(res)
	return out, nil
}

// TranscribeVadLang transcribes with VAD segmentation and a language hint.
func (s *CrispasrSession) TranscribeVadLang(pcm []float32, sampleRate int,
	vadModelPath string, language string) (*TranscribeResult, error) {
	if len(pcm) == 0 {
		return nil, nil
	}
	cvad := C.CString(vadModelPath)
	defer C.free(unsafe.Pointer(cvad))
	clang := C.CString(language)
	defer C.free(unsafe.Pointer(clang))
	res := C.crispasr_session_transcribe_vad_lang(s.handle,
		(*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)),
		C.int(sampleRate), cvad, nil, clang)
	if res == nil {
		return nil, fmt.Errorf("crispasr_session_transcribe_vad_lang failed")
	}
	defer C.crispasr_session_result_free(res)
	return extractResult(res), nil
}

// DetectBackendFromGGUF returns the backend name from GGUF metadata.
func DetectBackendFromGGUF(path string) (string, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var out [128]C.char
	rc := C.crispasr_detect_backend_from_gguf(cpath, &out[0], 128)
	// rc > 0 = detected (strlen of name); rc == 0 = valid GGUF but no backend
	// mapping (empty name); rc < 0 = error. The prior `rc != 0` reported every
	// successful detection as a failure.
	if rc < 0 {
		return "", fmt.Errorf("detect_backend_from_gguf failed for %s", path)
	}
	return C.GoString(&out[0]), nil
}

// EnhanceAudioRnnoise applies RNNoise denoising to 48 kHz mono PCM.
func EnhanceAudioRnnoise(pcm []float32) ([]float32, error) {
	out := make([]float32, len(pcm))
	rc := C.crispasr_enhance_audio_rnnoise(
		(*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)),
		(*C.float)(unsafe.Pointer(&out[0])), C.int(len(out)))
	if rc != 0 {
		return nil, fmt.Errorf("enhance_audio_rnnoise failed (rc=%d)", int(rc))
	}
	return out, nil
}

// TextDetectLanguage classifies the language of a text string.
// Returns (langCode, confidence).
func TextDetectLanguage(text, modelPath string, nThreads int) (string, float32, error) {
	ct := C.CString(text)
	defer C.free(unsafe.Pointer(ct))
	cm := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cm))
	var label [64]C.char
	var conf C.float
	rc := C.crispasr_text_detect_language(ct, cm, C.int(nThreads), &label[0], 64, &conf)
	if rc != 0 {
		return "", 0, fmt.Errorf("text_detect_language failed (rc=%d)", int(rc))
	}
	return C.GoString(&label[0]), float32(conf), nil
}

// RegistryLookupByFilename looks up a model by filename.
func RegistryLookupByFilename(filename string) (RegistryEntry, error) {
	cf := C.CString(filename)
	defer C.free(unsafe.Pointer(cf))
	var fname, url, sz [512]C.char
	rc := C.crispasr_registry_lookup_by_filename_abi(cf, &fname[0], 512, &url[0], 512, &sz[0], 512)
	if rc != 0 {
		return RegistryEntry{}, fmt.Errorf("no registry entry for filename %q", filename)
	}
	return RegistryEntry{
		Filename: C.GoString(&fname[0]),
		URL:      C.GoString(&url[0]),
		Size:     C.GoString(&sz[0]),
	}, nil
}

// ListKnownModels returns every backend name in the registry.
func ListKnownModels() []string {
	var buf [8192]C.char
	n := C.crispasr_registry_list_backends_abi(&buf[0], 8192)
	if n <= 0 {
		return nil
	}
	return splitCSV(C.GoString(&buf[0]))
}

// KokoroLangIsGerman returns true if the lang code maps to German.
func KokoroLangIsGerman(lang string) bool {
	cl := C.CString(lang)
	defer C.free(unsafe.Pointer(cl))
	return C.crispasr_kokoro_lang_is_german_abi(cl) != 0
}

// KokoroLangHasNativeVoice returns true if the lang has a native Kokoro voice.
func KokoroLangHasNativeVoice(lang string) bool {
	cl := C.CString(lang)
	defer C.free(unsafe.Pointer(cl))
	return C.crispasr_kokoro_lang_has_native_voice_abi(cl) != 0
}

// VadSlices runs the unified VAD dispatcher returning speech spans in seconds.
func VadSlices(modelPath string, pcm []float32, sampleRate int,
	threshold float32, minSpeechMs, minSilenceMs, speechPadMs int,
	maxChunkDurationS float32, nThreads int) ([][2]float32, error) {
	cm := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cm))
	var outSpans *C.float
	n := C.crispasr_vad_slices(cm,
		(*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)),
		C.int(sampleRate), C.float(threshold),
		C.int(minSpeechMs), C.int(minSilenceMs), C.int(speechPadMs),
		C.float(maxChunkDurationS), C.int(nThreads), &outSpans)
	if n < 0 {
		return nil, fmt.Errorf("crispasr_vad_slices failed (rc=%d)", int(n))
	}
	spans := make([][2]float32, int(n))
	if n > 0 {
		raw := (*[1 << 28]C.float)(unsafe.Pointer(outSpans))
		for i := 0; i < int(n); i++ {
			spans[i] = [2]float32{float32(raw[2*i]), float32(raw[2*i+1])}
		}
		C.crispasr_vad_free(outSpans)
	}
	return spans, nil
}

// LcsDedup returns the number of leading tokens to drop from curr to
// remove overlap with prevTail.
func LcsDedup(prevTail, curr []int32, minLcsLength int) int {
	if len(prevTail) == 0 || len(curr) == 0 {
		return 0
	}
	return int(C.crispasr_lcs_dedup_prefix_count(
		(*C.int)(unsafe.Pointer(&prevTail[0])), C.int(len(prevTail)),
		(*C.int)(unsafe.Pointer(&curr[0])), C.int(len(curr)),
		C.int(minLcsLength)))
}

// ---------------------------------------------------------------------------
// Direct Parakeet API
// ---------------------------------------------------------------------------

// ParakeetContext wraps a direct Parakeet ASR context.
type ParakeetContext struct {
	handle unsafe.Pointer
}

// ParakeetWord holds word-level timing from a Parakeet result.
type ParakeetWord struct {
	Text string
	T0   int64
	T1   int64
}

// ParakeetToken holds token-level timing from a Parakeet result.
type ParakeetToken struct {
	Text string
	T0   int64
	T1   int64
	P    float32
}

// ParakeetResult holds the full Parakeet transcription output.
type ParakeetResult struct {
	Text   string
	Words  []ParakeetWord
	Tokens []ParakeetToken
}

// ParakeetInit loads a Parakeet model.
func ParakeetInit(modelPath string, nThreads int, useFlash bool) (*ParakeetContext, error) {
	cm := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cm))
	flash := C.int(0)
	if useFlash {
		flash = 1
	}
	h := C.crispasr_parakeet_init(cm, C.int(nThreads), flash)
	if h == nil {
		return nil, fmt.Errorf("crispasr_parakeet_init failed for %s", modelPath)
	}
	return &ParakeetContext{handle: h}, nil
}

// Transcribe runs Parakeet ASR on mono 16 kHz float32 PCM.
func (p *ParakeetContext) Transcribe(pcm []float32, language string) (*ParakeetResult, error) {
	var clang *C.char
	if language != "" {
		clang = C.CString(language)
		defer C.free(unsafe.Pointer(clang))
	}
	res := C.crispasr_parakeet_transcribe(p.handle,
		(*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)), clang)
	if res == nil {
		return nil, fmt.Errorf("crispasr_parakeet_transcribe returned null")
	}
	defer C.crispasr_parakeet_result_free(res)

	textPtr := C.crispasr_parakeet_result_text(res)
	text := ""
	if textPtr != nil {
		text = C.GoString(textPtr)
	}

	nw := int(C.crispasr_parakeet_result_n_words(res))
	words := make([]ParakeetWord, nw)
	for i := 0; i < nw; i++ {
		wp := C.crispasr_parakeet_result_word_text(res, C.int(i))
		wt := ""
		if wp != nil {
			wt = C.GoString(wp)
		}
		words[i] = ParakeetWord{
			Text: wt,
			T0:   int64(C.crispasr_parakeet_result_word_t0(res, C.int(i))),
			T1:   int64(C.crispasr_parakeet_result_word_t1(res, C.int(i))),
		}
	}

	nt := int(C.crispasr_parakeet_result_n_tokens(res))
	tokens := make([]ParakeetToken, nt)
	for i := 0; i < nt; i++ {
		tp := C.crispasr_parakeet_result_token_text(res, C.int(i))
		tt := ""
		if tp != nil {
			tt = C.GoString(tp)
		}
		tokens[i] = ParakeetToken{
			Text: tt,
			T0:   int64(C.crispasr_parakeet_result_token_t0(res, C.int(i))),
			T1:   int64(C.crispasr_parakeet_result_token_t1(res, C.int(i))),
			P:    float32(C.crispasr_parakeet_result_token_p(res, C.int(i))),
		}
	}

	return &ParakeetResult{Text: text, Words: words, Tokens: tokens}, nil
}

// Close frees the Parakeet context.
func (p *ParakeetContext) Close() {
	if p != nil && p.handle != nil {
		C.crispasr_parakeet_free(p.handle)
		p.handle = nil
	}
}

// ---------------------------------------------------------------------------
// TitaNet speaker verification
// ---------------------------------------------------------------------------

// TitaNetContext wraps a TitaNet-Large speaker embedding model.
type TitaNetContext struct {
	handle unsafe.Pointer
}

// TitaNetInit loads a TitaNet model.
func TitaNetInit(modelPath string, nThreads int) (*TitaNetContext, error) {
	cm := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cm))
	h := C.crispasr_titanet_init(cm, C.int(nThreads))
	if h == nil {
		return nil, fmt.Errorf("crispasr_titanet_init failed for %s", modelPath)
	}
	return &TitaNetContext{handle: h}, nil
}

// Embed extracts a 192-d speaker embedding from 16 kHz mono PCM.
func (t *TitaNetContext) Embed(pcm16k []float32) ([]float32, error) {
	out := make([]float32, 192)
	dim := C.crispasr_titanet_embed(t.handle,
		(*C.float)(unsafe.Pointer(&pcm16k[0])), C.int(len(pcm16k)),
		(*C.float)(unsafe.Pointer(&out[0])))
	if dim <= 0 {
		return nil, fmt.Errorf("titanet_embed failed")
	}
	return out[:int(dim)], nil
}

// Close frees the TitaNet context.
func (t *TitaNetContext) Close() {
	if t != nil && t.handle != nil {
		C.crispasr_titanet_free(t.handle)
		t.handle = nil
	}
}

// TitaNetCosineSim returns the cosine similarity between two embeddings.
func TitaNetCosineSim(a, b []float32) float32 {
	dim := len(a)
	if len(b) < dim {
		dim = len(b)
	}
	return float32(C.crispasr_titanet_cosine_sim(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.int(dim)))
}

// ---------------------------------------------------------------------------
// Speaker database
// ---------------------------------------------------------------------------

// SpeakerDB wraps a file-based speaker profile database.
type SpeakerDB struct {
	handle  unsafe.Pointer
	dirPath string
}

// SpeakerDBOpen opens a speaker database directory for closed-roster
// matching (issue #266). expectedNames is the comma-separated list of
// enrolled participants you assert are present in the audio (e.g.
// "Alice,Bob") — the db is narrowed to exactly those profiles; open 1:N
// identification is deliberately unsupported. consentAttested affirms a
// lawful basis + explicit consent from every enrolled person (GDPR
// Art. 9); the call fails without it.
func SpeakerDBOpen(dirPath, expectedNames string, consentAttested bool) (*SpeakerDB, error) {
	if !consentAttested {
		return nil, fmt.Errorf("speaker db requires an explicit consent attestation (GDPR Art. 9)")
	}
	cd := C.CString(dirPath)
	defer C.free(unsafe.Pointer(cd))
	cn := C.CString(expectedNames)
	defer C.free(unsafe.Pointer(cn))
	h := C.crispasr_speaker_db_open(cd, cn, 1)
	if h == nil {
		return nil, fmt.Errorf("crispasr_speaker_db_open failed for %s", dirPath)
	}
	return &SpeakerDB{handle: h, dirPath: dirPath}, nil
}

// Count returns the number of enrolled speakers.
func (db *SpeakerDB) Count() int {
	return int(C.crispasr_speaker_db_count(db.handle))
}

// Match returns (name, score) for the best match, or ("", score) if below threshold.
func (db *SpeakerDB) Match(embedding []float32, threshold float32) (string, float32) {
	var name [256]C.char
	score := C.crispasr_speaker_db_match(db.handle,
		(*C.float)(unsafe.Pointer(&embedding[0])), C.int(len(embedding)),
		C.float(threshold), &name[0], 256)
	n := ""
	if float32(score) >= threshold {
		n = C.GoString(&name[0])
	}
	return n, float32(score)
}

// Enroll adds a speaker to the database. consentAttested affirms the
// enrolled person's explicit consent (GDPR Art. 9); it is recorded in
// the profile and enrollment refuses without it.
func (db *SpeakerDB) Enroll(name string, embedding []float32, consentAttested bool) error {
	if !consentAttested {
		return fmt.Errorf("enrollment requires an explicit consent attestation (GDPR Art. 9)")
	}
	cd := C.CString(db.dirPath)
	defer C.free(unsafe.Pointer(cd))
	cn := C.CString(name)
	defer C.free(unsafe.Pointer(cn))
	rc := C.crispasr_speaker_db_enroll2(cd, cn,
		(*C.float)(unsafe.Pointer(&embedding[0])), C.int(len(embedding)), 1)
	if rc != 0 {
		return fmt.Errorf("speaker_db_enroll failed (rc=%d)", int(rc))
	}
	return nil
}

// Close frees the speaker database.
func (db *SpeakerDB) Close() {
	if db != nil && db.handle != nil {
		C.crispasr_speaker_db_free(db.handle)
		db.handle = nil
	}
}

// ---------------------------------------------------------------------------
// Watermark — AI-generated audio marking
// ---------------------------------------------------------------------------

// WatermarkLoadModel loads an AudioSeal GGUF for neural watermarking.
// Once loaded, WatermarkEmbed/Detect dispatch to AudioSeal automatically.
// Returns nil on success or an error if the model fails to load.
func WatermarkLoadModel(ggufPath string) error {
	cp := C.CString(ggufPath)
	defer C.free(unsafe.Pointer(cp))
	rc := C.crispasr_watermark_load_model(cp)
	if rc != 0 {
		return fmt.Errorf("crispasr_watermark_load_model failed (rc=%d)", int(rc))
	}
	return nil
}

// WatermarkEmbed embeds an AI-generated watermark into PCM audio in-place.
// Uses AudioSeal if a model was loaded, otherwise spread-spectrum.
//
// Passes alpha <= 0, which selects the band-limited default strength that makes
// the mark reliably DETECTABLE — the property EU AI Act Art. 50(2) requires, and
// the reason this is the call SynthesizeRaw users need to discharge marking
// themselves. It used to hardcode 0.005, the strength the C ABI documents as
// too faint to reliably detect on real speech: an explicit positive alpha is
// used verbatim and bypasses the robust default.
func WatermarkEmbed(pcm []float32) {
	if len(pcm) == 0 {
		return
	}
	C.crispasr_watermark_embed((*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)), -1.0)
}

// WatermarkDetect returns a confidence score [0, 1] indicating whether
// the audio contains an AI-generated watermark. >0.65 = present.
func WatermarkDetect(pcm []float32) float32 {
	if len(pcm) == 0 {
		return 0
	}
	return float32(C.crispasr_watermark_detect((*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm))))
}
