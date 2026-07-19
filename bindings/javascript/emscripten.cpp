//
// This is the Javascript API of crispasr
//
// Very crude at the moment.
// Feel free to contribute and make this better!
//
// See tests/test-crispasr.js for sample usage.
//

#include "crispasr.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <thread>
#include <vector>

// PROXY_TO_PTHREAD path: the browser deadlock is that a thread which owns the
// event loop (the "servicer") may not block in pthread_join — but ggml's compute
// threads do exactly that. The fix (we own the code, so we can): run the heavy
// synth on a NON-servicer thread. Under -sPROXY_TO_PTHREAD, main() runs on a
// dedicated pthread; we keep it alive and use it as the compute thread, driving
// it from the servicer via a proxying queue. The servicer never blocks, so ggml's
// pthread_create/join for compute workers is serviced normally. No deadlock.
#ifdef __EMSCRIPTEN_PTHREADS__
#include <emscripten/proxying.h>
#include <emscripten/threading.h>
#include <pthread.h>
static emscripten::ProxyingQueue g_proxy_queue;
static pthread_t g_compute_thread; // pthread-0 under PROXY_TO_PTHREAD
static bool g_compute_thread_set = false;
#endif

// The unified Session C-ABI is declared in crispasr.h (included above)
// as `struct crispasr_session`. Legacy name alias for the Embind wrappers:
extern "C" {
typedef struct crispasr_session CrispasrSession;
CrispasrSession* crispasr_session_open(const char* model_path, int n_threads);
void crispasr_session_close(CrispasrSession* s);
int crispasr_session_set_codec_path(CrispasrSession* s, const char* path);
int crispasr_session_set_voice(CrispasrSession* s, const char* path, const char* ref_text_or_null);
int crispasr_session_set_speaker_name(CrispasrSession* s, const char* name);
int crispasr_session_n_speakers(CrispasrSession* s);
const char* crispasr_session_get_speaker_name(CrispasrSession* s, int i);
int crispasr_session_set_instruct(CrispasrSession* s, const char* instruct);
int crispasr_session_is_custom_voice(CrispasrSession* s);
int crispasr_session_is_voice_design(CrispasrSession* s);
float* crispasr_session_synthesize(CrispasrSession* s, const char* text, int* out_n_samples);
void crispasr_pcm_free(float* pcm);
unsigned char* crispasr_c2pa_sign(const unsigned char* data, size_t len, const char* format, const char* cert_path,
                                  const char* key_path, size_t* out_len);
void crispasr_c2pa_free(unsigned char* p);
unsigned char* crispasr_pcm_to_wav(const float* pcm, int n_samples, int sample_rate, size_t* out_len);
int crispasr_session_kokoro_clear_phoneme_cache(CrispasrSession* s);
int crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang, char* out_path,
                                               int out_path_len);
int crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang, char* out_path,
                                               int out_path_len, char* out_picked, int out_picked_len);

// --- Full C-ABI parity declarations ---
// Session extras
int crispasr_session_available_backends(char* out_csv, int out_cap);
int crispasr_session_detected_language(CrispasrSession* s, char* out_buf, int out_cap);
// CTC vocabulary access (Omni CTC backend): n_vocab piece count, token_text
// maps an id to its model-owned raw piece (do not free) or "" when out of
// range / unsupported.
int crispasr_session_n_vocab(CrispasrSession* s);
const char* crispasr_session_token_text(CrispasrSession* s, int id);
// Pitch (F0) estimation — CREPE. `pitch` runs the track and returns the
// frame count; `pitch_frames` hands back a session-owned flat array of
// 3 floats per frame {time_ms, f0_hz, voiced_prob}, frame-major.
// Referencing these here is also what keeps the `crepe` objects from
// being dead-stripped out of the wasm module at link time.
int crispasr_session_pitch(CrispasrSession* s, const float* pcm_16k, int n_samples, float hop_ms);
const float* crispasr_session_pitch_frames(CrispasrSession* s, int* out_n_frames);
int crispasr_session_pitch_sample_rate(CrispasrSession* s);
CrispasrSession* crispasr_session_open_explicit(const char* model_path, const char* backend_name, int n_threads);
CrispasrSession* crispasr_session_open_with_params(const char* model_path, const char* backend_name,
                                                   const void* params);
const char* crispasr_session_backend(CrispasrSession* s);
int crispasr_session_set_source_language(CrispasrSession* s, const char* lang);
int crispasr_session_set_target_language(CrispasrSession* s, const char* lang);
int crispasr_session_set_punctuation(CrispasrSession* s, int enable);
int crispasr_session_set_punc_model(CrispasrSession* s, const char* punc_model);
int crispasr_session_set_hotwords(CrispasrSession* s, const char* hotwords, float boost);
int crispasr_session_set_translate(CrispasrSession* s, int enable);
int crispasr_session_set_temperature(CrispasrSession* s, float temperature, unsigned long long seed);
int crispasr_session_set_tts_seed(CrispasrSession* s, unsigned long long seed);
int crispasr_session_set_tts_steps(CrispasrSession* s, int steps);
int crispasr_session_set_tts_num_candidates(CrispasrSession* s, int n);
int crispasr_session_set_max_new_tokens(CrispasrSession* s, int n);
int crispasr_session_set_frequency_penalty(CrispasrSession* s, float penalty);
int crispasr_session_set_top_p(CrispasrSession* s, float top_p);
int crispasr_session_set_top_k(CrispasrSession* s, int top_k);
int crispasr_session_set_do_sample(CrispasrSession* s, int enable);
int crispasr_session_set_min_p(CrispasrSession* s, float min_p);
int crispasr_session_set_repetition_penalty(CrispasrSession* s, float r);
int crispasr_session_set_cfg_weight(CrispasrSession* s, float cfg_weight);
int crispasr_session_set_tts_noise_temp(CrispasrSession* s, float noise_temp);
int crispasr_session_set_exaggeration(CrispasrSession* s, float exaggeration);
int crispasr_session_set_max_speech_tokens(CrispasrSession* s, int n);
int crispasr_session_set_length_scale(CrispasrSession* s, float scale);
int crispasr_session_set_best_of(CrispasrSession* s, int n);
int crispasr_session_set_beam_size(CrispasrSession* s, int n);
int crispasr_session_set_return_logits(CrispasrSession* s, int enable);
int crispasr_session_set_grammar_text(CrispasrSession* s, const char* gbnf_text, const char* root_rule, float penalty);
int crispasr_session_set_fallback_thresholds(CrispasrSession* s, float entropy_thold, float logprob_thold,
                                             float no_speech_thold, float temperature_inc);
int crispasr_session_set_alt_n(CrispasrSession* s, int n);
int crispasr_session_set_whisper_decode_extras(CrispasrSession* s, int suppress_nst, const char* suppress_regex,
                                               int carry_initial_prompt);
int crispasr_session_set_ask(CrispasrSession* s, const char* prompt);
int crispasr_session_detect_language(CrispasrSession* s, const float* pcm, int n_samples, const char* lid_model_path,
                                     int method, char* out_lang, int out_lang_cap, float* out_prob);

// Session ASR transcription
struct crispasr_session_result;
struct crispasr_session_result* crispasr_session_transcribe(CrispasrSession* s, const float* pcm, int n_samples);
struct crispasr_session_result* crispasr_session_transcribe_lang(CrispasrSession* s, const float* pcm, int n_samples,
                                                                 const char* language);
struct crispasr_session_result* crispasr_session_transcribe_chunked_lang(CrispasrSession* s, const float* pcm,
                                                                         int n_samples, int chunk_seconds,
                                                                         int overlap_seconds, const char* language);
int crispasr_get_progress(void);
struct crispasr_session_result* crispasr_session_transcribe_vad(CrispasrSession* s, const float* pcm, int n_samples,
                                                                int sample_rate, const char* vad_model_path,
                                                                void* opts);
struct crispasr_session_result* crispasr_session_transcribe_vad_lang(CrispasrSession* s, const float* pcm,
                                                                     int n_samples, int sample_rate,
                                                                     const char* vad_model_path, void* opts,
                                                                     const char* language);
int crispasr_session_result_n_segments(struct crispasr_session_result* r);
const char* crispasr_session_result_segment_text(struct crispasr_session_result* r, int i);
long long crispasr_session_result_segment_t0(struct crispasr_session_result* r, int i);
long long crispasr_session_result_segment_t1(struct crispasr_session_result* r, int i);
int crispasr_session_result_n_words(struct crispasr_session_result* r, int i_seg);
const char* crispasr_session_result_word_text(struct crispasr_session_result* r, int i_seg, int i_word);
long long crispasr_session_result_word_t0(struct crispasr_session_result* r, int i_seg, int i_word);
long long crispasr_session_result_word_t1(struct crispasr_session_result* r, int i_seg, int i_word);
float crispasr_session_result_word_p(struct crispasr_session_result* r, int i_seg, int i_word);
float crispasr_session_result_segment_no_speech_prob(struct crispasr_session_result* r, int i_seg);
int crispasr_session_result_word_n_alts(struct crispasr_session_result* r, int i_seg, int i_word);
const char* crispasr_session_result_word_alt_text(struct crispasr_session_result* r, int i_seg, int i_word, int i_alt);
float crispasr_session_result_word_alt_p(struct crispasr_session_result* r, int i_seg, int i_word, int i_alt);
int crispasr_session_result_n_logit_frames(struct crispasr_session_result* r);
int crispasr_session_result_n_logit_vocab(struct crispasr_session_result* r);
const float* crispasr_session_result_logits(struct crispasr_session_result* r);
void crispasr_session_result_free(struct crispasr_session_result* r);
char* crispasr_session_translate_text(CrispasrSession* s, const char* text, const char* src_lang, const char* tgt_lang,
                                      int max_tokens);
void crispasr_session_translate_text_free(char* text);

// Streaming
struct CrispasrStream;
struct CrispasrStream* crispasr_session_stream_open(CrispasrSession* s, int n_threads, int step_ms, int length_ms,
                                                    int keep_ms, const char* language, int translate);
struct CrispasrStream* crispasr_stream_open(void* ctx, int n_threads, int step_ms, int length_ms, int keep_ms,
                                            const char* language, int translate);
int crispasr_stream_feed(struct CrispasrStream* s, const float* pcm, int n_samples);
int crispasr_stream_get_text(struct CrispasrStream* s, char* out_text, int out_cap, double* out_t0_s, double* out_t1_s,
                             long long* out_counter);
int crispasr_stream_flush(struct CrispasrStream* s);
void crispasr_stream_close(struct CrispasrStream* s);
void crispasr_stream_set_live_decode(struct CrispasrStream* s, int enabled);

// Punctuation
void* crispasr_punc_init(const char* model_path);
const char* crispasr_punc_process(void* ctx, const char* text);
void crispasr_punc_free_text(const char* text);
void crispasr_punc_free(void* ctx);

// Alignment
struct crispasr_align_result;
struct crispasr_align_result* crispasr_align_words_abi(const char* aligner_model, const char* transcript,
                                                       const float* samples, int n_samples, long long t_offset_cs,
                                                       int n_threads);
int crispasr_align_result_n_words(struct crispasr_align_result* r);
const char* crispasr_align_result_word_text(struct crispasr_align_result* r, int i);
long long crispasr_align_result_word_t0(struct crispasr_align_result* r, int i);
long long crispasr_align_result_word_t1(struct crispasr_align_result* r, int i);
void crispasr_align_result_free(struct crispasr_align_result* r);

// VAD — declared in crispasr.h

// LCS dedup
int crispasr_lcs_dedup_prefix_count(const int* prev_tail_tokens, int n_prev, const int* curr_tokens, int n_curr,
                                    int min_lcs_length);

// params_set_* — declared in crispasr.h (use whisper_full_params*, not void*)

// Token-level accessors
long long crispasr_token_t0(void* ctx, int i_seg, int i_tok);
long long crispasr_token_t1(void* ctx, int i_seg, int i_tok);
float crispasr_token_p(void* ctx, int i_seg, int i_tok);
int crispasr_token_n_alts(void* ctx, int i_seg, int i_tok);
int crispasr_token_alt_id(void* ctx, int i_seg, int i_tok, int i_alt);
float crispasr_token_alt_p(void* ctx, int i_seg, int i_tok, int i_alt);
int crispasr_token_alt_text(void* ctx, int i_seg, int i_tok, int i_alt, char* out, int out_cap);

// Language detection
float crispasr_detect_language(void* ctx, const float* pcm, int n_samples, int n_threads, char* out_code, int out_cap);
int crispasr_detect_language_pcm(const float* samples, int n_samples, int method, const char* model_path, int n_threads,
                                 int use_gpu, int gpu_device, int flash_attn, char* out_lang, int out_lang_cap,
                                 float* out_confidence);

// Direct Parakeet API
void* crispasr_parakeet_init(const char* model_path, int n_threads, int use_flash);
void crispasr_parakeet_free(void* ctx);
void* crispasr_parakeet_transcribe(void* ctx, const float* pcm, int n_samples, const char* language);
const char* crispasr_parakeet_result_text(void* r);
int crispasr_parakeet_result_n_words(void* r);
const char* crispasr_parakeet_result_word_text(void* r, int i);
long long crispasr_parakeet_result_word_t0(void* r, int i);
long long crispasr_parakeet_result_word_t1(void* r, int i);
int crispasr_parakeet_result_n_tokens(void* r);
const char* crispasr_parakeet_result_token_text(void* r, int i);
long long crispasr_parakeet_result_token_t0(void* r, int i);
long long crispasr_parakeet_result_token_t1(void* r, int i);
float crispasr_parakeet_result_token_p(void* r, int i);
void crispasr_parakeet_result_free(void* r);

// Backend detection
int crispasr_detect_backend_from_gguf(const char* path, char* out_name, int out_cap);

// RNNoise audio enhancement
int crispasr_enhance_audio_rnnoise(const float* in_pcm, int n_samples, float* out_pcm, int out_cap);

// Text-LID
int crispasr_text_detect_language(const char* text, const char* model_path, int n_threads, char* out_label,
                                  int out_label_cap, float* out_confidence);

// TitaNet
void* crispasr_titanet_init(const char* model_path, int n_threads);
void crispasr_titanet_free(void* ctx);
int crispasr_titanet_embed(void* ctx, const float* pcm_16k, int n_samples, float* out);
float crispasr_titanet_cosine_sim(const float* a, const float* b, int dim);

// Speaker database (closed-roster, consent-gated — issue #266)
void* crispasr_speaker_db_open(const char* dir_path, const char* expected_names_csv, int consent_attested);
void crispasr_speaker_db_free(void* db);
int crispasr_speaker_db_count(const void* db);
float crispasr_speaker_db_match(const void* db, const float* embedding, int dim, float threshold, char* out_name,
                                int out_cap);
int crispasr_speaker_db_enroll2(const char* dir_path, const char* name, const float* embedding, int dim,
                                int consent_attested);

// Pluggable speaker embedder + clustering + pyannote cache
void* crispasr_speaker_embedder_make_abi(const char* model_spec, int n_threads, const char* cache_dir);
void crispasr_speaker_embedder_free_abi(void* embedder);
int crispasr_speaker_embedder_dim_abi(const void* embedder);
int crispasr_speaker_embedder_embed_abi(void* embedder, const float* pcm_16k, int n_samples, float* out);
const char* crispasr_speaker_embedder_name_abi(const void* embedder);
int crispasr_speaker_cluster_abi(const float* embeddings, int n, int dim, float merge_threshold, int max_speakers,
                                 int* labels_out);
void* crispasr_pyannote_cache_compute_abi(const float* full_audio, int n_samples, const char* model_path,
                                          int n_threads);
void crispasr_pyannote_cache_free_abi(void* cache);
int crispasr_pyannote_cache_apply_abi(const void* cache, long long slice_t0_cs, void* segs, int n_segs);

// Kokoro lang helpers
int crispasr_kokoro_lang_is_german_abi(const char* lang);
int crispasr_kokoro_lang_has_native_voice_abi(const char* lang);

// Registry + cache
int crispasr_registry_lookup_abi(const char* backend, char* out_filename, int filename_cap, char* out_url, int url_cap,
                                 char* out_size, int size_cap);
int crispasr_registry_lookup_by_filename_abi(const char* filename, char* out_filename, int filename_cap, char* out_url,
                                             int url_cap, char* out_size, int size_cap);
int crispasr_registry_list_backends_abi(char* out_csv, int out_cap);
int crispasr_cache_ensure_file_abi(const char* filename, const char* url, int quiet, const char* cache_dir_override,
                                   char* out_buf, int out_cap);
int crispasr_cache_dir_abi(const char* cache_dir_override, char* out_buf, int out_cap);

// Diarization
int crispasr_diarize_segments_abi(const float* left_pcm, const float* right_pcm, int n_samples, int is_stereo,
                                  void* segs, int n_segs, const void* opts);
}

static CrispasrSession* g_tts_session = nullptr;
// Backend-agnostic ASR session (parakeet/canary/whisper/…) for the asr* surface
// below — distinct from the whisper-only g_context used by init()/full_default().
static CrispasrSession* g_asr_session = nullptr;

struct whisper_context* g_context;

#ifdef __EMSCRIPTEN_PTHREADS__
// Present only in threaded builds. Under -sPROXY_TO_PTHREAD this runs on pthread-0
// (NOT the servicer); we record it as the compute thread and keep the runtime live
// so it can process the proxying queue. In a non-proxy threaded build this runs on
// the servicer itself, so ttsSynthesizeAsync would still block there — the async
// path is only correct with PROXY_TO_PTHREAD (that's the whole point).
int main() {
    g_compute_thread = pthread_self();
    g_compute_thread_set = true;
    emscripten_exit_with_live_runtime();
    return 0;
}
#endif

EMSCRIPTEN_BINDINGS(whisper) {
    emscripten::function("init", emscripten::optional_override([](const std::string& path_model) {
                             if (g_context == nullptr) {
                                 g_context = whisper_init_from_file_with_params(path_model.c_str(),
                                                                                whisper_context_default_params());
                                 if (g_context != nullptr) {
                                     return true;
                                 } else {
                                     return false;
                                 }
                             }

                             return false;
                         }));

    emscripten::function("free", emscripten::optional_override([]() {
                             if (g_context) {
                                 whisper_free(g_context);
                                 g_context = nullptr;
                             }
                         }));

    emscripten::function(
        "full_default",
        emscripten::optional_override([](const emscripten::val& audio, const std::string& lang, bool translate) {
            if (g_context == nullptr) {
                return -1;
            }

            struct whisper_full_params params =
                whisper_full_default_params(whisper_sampling_strategy::CRISPASR_SAMPLING_GREEDY);

            params.print_realtime = true;
            params.print_progress = false;
            params.print_timestamps = true;
            params.print_special = false;
            params.translate = translate;
            params.language = whisper_is_multilingual(g_context) ? lang.c_str() : "en";
            params.n_threads = std::min(8, (int)std::thread::hardware_concurrency());
            params.offset_ms = 0;

            std::vector<float> pcmf32;
            const int n = audio["length"].as<int>();

            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];

            pcmf32.resize(n);

            emscripten::val memoryView =
                audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
            memoryView.call<void>("set", audio);

            // print system information
            {
                printf("\n");
                printf("system_info: n_threads = %d / %d | %s\n", params.n_threads, std::thread::hardware_concurrency(),
                       whisper_print_system_info());

                printf("\n");
                printf("%s: processing %d samples, %.1f sec, %d threads, %d processors, lang = %s, task = %s ...\n",
                       __func__, int(pcmf32.size()), float(pcmf32.size()) / CRISPASR_SAMPLE_RATE, params.n_threads, 1,
                       params.language, params.translate ? "translate" : "transcribe");

                printf("\n");
            }

            // run whisper
            {
                whisper_reset_timings(g_context);
                whisper_full(g_context, params, pcmf32.data(), pcmf32.size());
                whisper_print_timings(g_context);
            }

            return 0;
        }));

    // -------------------------------------------------------------------
    // Backend-agnostic ASR session surface (parakeet, canary, …) over the
    // crispasr_session C-ABI — the WASM analogue of the native bindings'
    // Session. Unlike init()/full_default() above (whisper-only), this reaches
    // every ASR backend plus the session post-processors (punctuation,
    // punc-model, beam, hotwords, translate). Mirrors the tts* surface.
    // -------------------------------------------------------------------
    emscripten::function(
        "asrOpen",
        emscripten::optional_override([](const std::string& model_path, const std::string& backend, int n_threads) {
            if (g_asr_session) {
                crispasr_session_close(g_asr_session);
                g_asr_session = nullptr;
            }
            g_asr_session = backend.empty()
                                ? crispasr_session_open(model_path.c_str(), n_threads)
                                : crispasr_session_open_explicit(model_path.c_str(), backend.c_str(), n_threads);
            return g_asr_session != nullptr;
        }));

    emscripten::function("asrClose", emscripten::optional_override([]() {
                             if (g_asr_session) {
                                 crispasr_session_close(g_asr_session);
                                 g_asr_session = nullptr;
                             }
                         }));

    emscripten::function("asrSetSourceLanguage", emscripten::optional_override([](const std::string& lang) {
                             return g_asr_session ? crispasr_session_set_source_language(g_asr_session, lang.c_str())
                                                  : -1;
                         }));
    emscripten::function("asrSetTargetLanguage", emscripten::optional_override([](const std::string& lang) {
                             return g_asr_session ? crispasr_session_set_target_language(g_asr_session, lang.c_str())
                                                  : -1;
                         }));
    emscripten::function("asrSetTranslate", emscripten::optional_override([](bool enable) {
                             return g_asr_session ? crispasr_session_set_translate(g_asr_session, enable ? 1 : 0) : -1;
                         }));
    emscripten::function("asrSetPunctuation", emscripten::optional_override([](bool enable) {
                             return g_asr_session ? crispasr_session_set_punctuation(g_asr_session, enable ? 1 : 0)
                                                  : -1;
                         }));
    emscripten::function("asrSetPuncModel", emscripten::optional_override([](const std::string& m) {
                             return g_asr_session ? crispasr_session_set_punc_model(g_asr_session, m.c_str()) : -1;
                         }));
    emscripten::function("asrSetBeamSize", emscripten::optional_override([](int n) {
                             return g_asr_session ? crispasr_session_set_beam_size(g_asr_session, n) : -1;
                         }));
    emscripten::function("asrSetHotwords", emscripten::optional_override([](const std::string& w, double boost) {
                             return g_asr_session
                                        ? crispasr_session_set_hotwords(g_asr_session, w.c_str(), (float)boost)
                                        : -1;
                         }));
    emscripten::function("asrSetAsk", emscripten::optional_override([](const std::string& prompt) {
                             return g_asr_session ? crispasr_session_set_ask(g_asr_session, prompt.c_str()) : -1;
                         }));
    emscripten::function("asrSetTemperature", emscripten::optional_override([](double temp, double seed) {
                             return g_asr_session ? crispasr_session_set_temperature(g_asr_session, (float)temp,
                                                                                     (unsigned long long)seed)
                                                  : -1;
                         }));

    // Transcribe a Float32Array of 16 kHz mono PCM. Returns an array of
    // { t0, t1, text } segments (timestamps in centiseconds). Empty on failure.
    emscripten::function(
        "asrTranscribe",
        emscripten::optional_override([](const emscripten::val& audio, const std::string& lang) -> emscripten::val {
            emscripten::val out = emscripten::val::array();
            if (!g_asr_session)
                return out;
            const int n = audio["length"].as<int>();
            std::vector<float> pcmf32(n);
            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];
            emscripten::val view = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
            view.call<void>("set", audio);

            crispasr_session_result* r = crispasr_session_transcribe_lang(g_asr_session, pcmf32.data(), n,
                                                                          lang.empty() ? nullptr : lang.c_str());
            if (!r)
                return out;
            const int ns = crispasr_session_result_n_segments(r);
            for (int i = 0; i < ns; i++) {
                emscripten::val seg = emscripten::val::object();
                seg.set("t0", (double)crispasr_session_result_segment_t0(r, i));
                seg.set("t1", (double)crispasr_session_result_segment_t1(r, i));
                const char* text = crispasr_session_result_segment_text(r, i);
                seg.set("text", std::string(text ? text : ""));
                out.call<void>("push", seg);
            }
            crispasr_session_result_free(r);
            return out;
        }));

    // asrTranscribeChunked(audio, lang, chunkSeconds, overlapSeconds) — issue
    // #208 chunked-encode long-form path (bounded windows on Parakeet; inert on
    // other backends). chunkSeconds<=0 keeps the per-model default window;
    // overlapSeconds<0 keeps the default. Poll asrGetProgress() for a bar.
    emscripten::function(
        "asrTranscribeChunked",
        emscripten::optional_override([](const emscripten::val& audio, const std::string& lang, int chunkSeconds,
                                         int overlapSeconds) -> emscripten::val {
            emscripten::val out = emscripten::val::array();
            if (!g_asr_session)
                return out;
            const int n = audio["length"].as<int>();
            std::vector<float> pcmf32(n);
            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];
            emscripten::val view = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
            view.call<void>("set", audio);

            crispasr_session_result* r = crispasr_session_transcribe_chunked_lang(
                g_asr_session, pcmf32.data(), n, chunkSeconds, overlapSeconds, lang.empty() ? nullptr : lang.c_str());
            if (!r)
                return out;
            const int ns = crispasr_session_result_n_segments(r);
            for (int i = 0; i < ns; i++) {
                emscripten::val seg = emscripten::val::object();
                seg.set("t0", (double)crispasr_session_result_segment_t0(r, i));
                seg.set("t1", (double)crispasr_session_result_segment_t1(r, i));
                const char* text = crispasr_session_result_segment_text(r, i);
                seg.set("text", std::string(text ? text : ""));
                out.call<void>("push", seg);
            }
            crispasr_session_result_free(r);
            return out;
        }));

    // asrGetProgress() -> 0..100 (or -1 idle). Long-form progress poll, updated
    // in lockstep with asrTranscribeChunked windows (issue #208).
    emscripten::function("asrGetProgress", emscripten::optional_override([]() { return crispasr_get_progress(); }));

    // -------------------------------------------------------------------
    // TTS surface (kokoro / vibevoice / qwen3-tts) + kokoro per-language
    // routing (PLAN #56 opt 2b).
    // -------------------------------------------------------------------

    emscripten::function("ttsOpen", emscripten::optional_override([](const std::string& model_path, int n_threads) {
                             if (g_tts_session != nullptr) {
                                 crispasr_session_close(g_tts_session);
                                 g_tts_session = nullptr;
                             }
                             g_tts_session = crispasr_session_open(model_path.c_str(), n_threads <= 0 ? 1 : n_threads);
                             return g_tts_session != nullptr;
                         }));

    // Like ttsOpen but with an explicit backend name (e.g. "kokoro", "piper") — some TTS backends
    // aren't auto-detectable from the GGUF and need it named, exactly as `--backend` does on the CLI.
    emscripten::function(
        "ttsOpenExplicit",
        emscripten::optional_override([](const std::string& model_path, const std::string& backend, int n_threads) {
            if (g_tts_session != nullptr) {
                crispasr_session_close(g_tts_session);
                g_tts_session = nullptr;
            }
            const int nt = n_threads <= 0 ? 1 : n_threads;
            g_tts_session = backend.empty() ? crispasr_session_open(model_path.c_str(), nt)
                                            : crispasr_session_open_explicit(model_path.c_str(), backend.c_str(), nt);
            return g_tts_session != nullptr;
        }));

    emscripten::function("ttsClose", emscripten::optional_override([]() {
                             if (g_tts_session) {
                                 crispasr_session_close(g_tts_session);
                                 g_tts_session = nullptr;
                             }
                         }));

    emscripten::function("ttsSetCodecPath", emscripten::optional_override([](const std::string& path) {
                             return g_tts_session ? crispasr_session_set_codec_path(g_tts_session, path.c_str()) : -1;
                         }));

    // Drop the kokoro per-session phoneme cache. (PLAN #56 #5)
    emscripten::function("ttsClearPhonemeCache", emscripten::optional_override([]() {
                             return g_tts_session ? crispasr_session_kokoro_clear_phoneme_cache(g_tts_session) : -1;
                         }));

    emscripten::function("ttsSetVoice",
                         emscripten::optional_override([](const std::string& path, const std::string& ref_text) {
                             if (!g_tts_session)
                                 return -1;
                             const char* rt = ref_text.empty() ? nullptr : ref_text.c_str();
                             return crispasr_session_set_voice(g_tts_session, path.c_str(), rt);
                         }));

    // Orpheus preset speakers — set by NAME, not by file path.
    emscripten::function("ttsSetSpeakerName", emscripten::optional_override([](const std::string& name) {
                             if (!g_tts_session)
                                 return -1;
                             return crispasr_session_set_speaker_name(g_tts_session, name.c_str());
                         }));

    // Returns the list of preset speaker names for the active backend
    // (orpheus today). Empty array if the backend has no preset speakers.
    emscripten::function("ttsSpeakers", emscripten::optional_override([]() -> emscripten::val {
                             emscripten::val out = emscripten::val::array();
                             if (!g_tts_session)
                                 return out;
                             int n = crispasr_session_n_speakers(g_tts_session);
                             for (int i = 0; i < n; i++) {
                                 const char* name = crispasr_session_get_speaker_name(g_tts_session, i);
                                 if (name)
                                     out.call<void>("push", std::string(name));
                             }
                             return out;
                         }));

    // qwen3-tts VoiceDesign — natural-language voice description.
    emscripten::function("ttsSetInstruct", emscripten::optional_override([](const std::string& instruct) {
                             if (!g_tts_session)
                                 return -1;
                             return crispasr_session_set_instruct(g_tts_session, instruct.c_str());
                         }));

    // qwen3-tts variant detection (returns false also when the active
    // backend isn't qwen3-tts).
    emscripten::function("ttsIsCustomVoice", emscripten::optional_override([]() -> bool {
                             return g_tts_session && crispasr_session_is_custom_voice(g_tts_session) != 0;
                         }));

    emscripten::function("ttsIsVoiceDesign", emscripten::optional_override([]() -> bool {
                             return g_tts_session && crispasr_session_is_voice_design(g_tts_session) != 0;
                         }));

    // Returns a Float32Array of 24 kHz mono PCM. Empty array on failure.
    emscripten::function("ttsSynthesize", emscripten::optional_override([](const std::string& text) -> emscripten::val {
                             if (!g_tts_session)
                                 return emscripten::val::array();
                             int n = 0;
                             float* pcm = crispasr_session_synthesize(g_tts_session, text.c_str(), &n);
                             if (!pcm || n <= 0) {
                                 if (pcm)
                                     crispasr_pcm_free(pcm);
                                 return emscripten::val::array();
                             }
                             emscripten::val out = emscripten::val::global("Float32Array").new_(n);
                             emscripten::val memoryView = emscripten::val(emscripten::typed_memory_view(n, pcm));
                             out.call<void>("set", memoryView);
                             crispasr_pcm_free(pcm);
                             return out;
                         }));

#ifdef __EMSCRIPTEN_PTHREADS__
    // Multithreaded, deadlock-free synth for the browser. Delivers a Float32Array
    // (24 kHz mono PCM, empty on failure) to `cb`. The servicer only enqueues work
    // and returns immediately; the blocking compute happens on the proxied pthread.
    emscripten::function(
        "ttsSynthesizeAsync", emscripten::optional_override([](const std::string& text, emscripten::val cb) {
            auto* text_copy = new std::string(text);
            auto* cbp = new emscripten::val(cb);
            if (!g_compute_thread_set) {
                // No proxied runtime thread (shouldn't happen once
                // the factory has resolved) — run inline as fallback.
                int n = 0;
                float* pcm =
                    g_tts_session ? crispasr_session_synthesize(g_tts_session, text_copy->c_str(), &n) : nullptr;
                emscripten::val out = emscripten::val::array();
                if (pcm && n > 0) {
                    out = emscripten::val::global("Float32Array").new_(n);
                    out.call<void>("set", emscripten::val(emscripten::typed_memory_view(n, pcm)));
                }
                if (pcm)
                    crispasr_pcm_free(pcm);
                (*cbp)(out);
                delete text_copy;
                delete cbp;
                return;
            }
            pthread_t servicer = pthread_self();
            // Enqueue the blocking synth onto the compute thread (pthread-0).
            g_proxy_queue.proxyAsync(g_compute_thread, [text_copy, cbp, servicer]() {
                int n = 0;
                float* pcm =
                    g_tts_session ? crispasr_session_synthesize(g_tts_session, text_copy->c_str(), &n) : nullptr;
                delete text_copy;
                // Hand the result (shared WASM heap) back to the servicer; only
                // touch the JS callback `val` on the thread that created it.
                g_proxy_queue.proxyAsync(servicer, [pcm, n, cbp]() {
                    emscripten::val out = emscripten::val::array();
                    if (pcm && n > 0) {
                        out = emscripten::val::global("Float32Array").new_(n);
                        out.call<void>("set", emscripten::val(emscripten::typed_memory_view(n, pcm)));
                    }
                    if (pcm)
                        crispasr_pcm_free(pcm);
                    (*cbp)(out);
                    delete cbp;
                });
            });
        }));
#endif

    // Mirrors python crispasr.kokoro_resolve_for_lang() — returns
    // {modelPath, voicePath, voiceName, backboneSwapped}.
    emscripten::function(
        "kokoroResolveForLang",
        emscripten::optional_override([](const std::string& model_path, const std::string& lang) -> emscripten::val {
            char out_model[1024] = {0};
            char out_voice[1024] = {0};
            char out_picked[64] = {0};

            int rc = crispasr_kokoro_resolve_model_for_lang_abi(model_path.c_str(), lang.c_str(), out_model,
                                                                sizeof(out_model));
            bool swapped = (rc == 0);
            std::string resolved = (out_model[0] != 0) ? std::string(out_model) : model_path;

            std::string vp, vn;
            rc = crispasr_kokoro_resolve_fallback_voice_abi(model_path.c_str(), lang.c_str(), out_voice,
                                                            sizeof(out_voice), out_picked, sizeof(out_picked));
            if (rc == 0) {
                vp = out_voice;
                vn = out_picked;
            }

            emscripten::val r = emscripten::val::object();
            r.set("modelPath", resolved);
            r.set("voicePath", vp);
            r.set("voiceName", vn);
            r.set("backboneSwapped", swapped);
            return r;
        }));

    // -------------------------------------------------------------------
    // Full C-ABI parity wrappers (PLAN #59)
    // -------------------------------------------------------------------

    // --- Session setters ---
    emscripten::function("sessionSetSourceLanguage", emscripten::optional_override([](const std::string& lang) {
                             return g_tts_session ? crispasr_session_set_source_language(g_tts_session, lang.c_str())
                                                  : -1;
                         }));
    emscripten::function("sessionSetTargetLanguage", emscripten::optional_override([](const std::string& lang) {
                             return g_tts_session ? crispasr_session_set_target_language(g_tts_session, lang.c_str())
                                                  : -1;
                         }));
    emscripten::function("sessionSetPunctuation", emscripten::optional_override([](bool enable) {
                             return g_tts_session ? crispasr_session_set_punctuation(g_tts_session, enable ? 1 : 0)
                                                  : -1;
                         }));
    emscripten::function("sessionSetTranslate", emscripten::optional_override([](bool enable) {
                             return g_tts_session ? crispasr_session_set_translate(g_tts_session, enable ? 1 : 0) : -1;
                         }));
    emscripten::function(
        "sessionSetTemperature", emscripten::optional_override([](float temp, int seed) {
            return g_tts_session ? crispasr_session_set_temperature(g_tts_session, temp, (unsigned long long)seed) : -1;
        }));
    emscripten::function("sessionSetTtsSeed", emscripten::optional_override([](int seed) {
                             return g_tts_session
                                        ? crispasr_session_set_tts_seed(g_tts_session, (unsigned long long)seed)
                                        : -1;
                         }));
    emscripten::function("sessionSetTtsSteps", emscripten::optional_override([](int steps) {
                             return g_tts_session ? crispasr_session_set_tts_steps(g_tts_session, steps) : -1;
                         }));
    emscripten::function("sessionSetTtsNumCandidates", emscripten::optional_override([](int n) {
                             return g_tts_session ? crispasr_session_set_tts_num_candidates(g_tts_session, n) : -1;
                         }));
    emscripten::function("sessionSetMaxNewTokens", emscripten::optional_override([](int n) {
                             return g_tts_session ? crispasr_session_set_max_new_tokens(g_tts_session, n) : -1;
                         }));
    emscripten::function("sessionSetFrequencyPenalty", emscripten::optional_override([](float p) {
                             return g_tts_session ? crispasr_session_set_frequency_penalty(g_tts_session, p) : -1;
                         }));
    emscripten::function("sessionSetTopP", emscripten::optional_override([](float p) {
                             return g_tts_session ? crispasr_session_set_top_p(g_tts_session, p) : -1;
                         }));
    emscripten::function("sessionSetTopK", emscripten::optional_override([](int k) {
                             return g_tts_session ? crispasr_session_set_top_k(g_tts_session, k) : -1;
                         }));
    emscripten::function("sessionSetDoSample", emscripten::optional_override([](bool enable) {
                             return g_tts_session ? crispasr_session_set_do_sample(g_tts_session, enable ? 1 : 0) : -1;
                         }));
    emscripten::function("sessionSetMinP", emscripten::optional_override([](float p) {
                             return g_tts_session ? crispasr_session_set_min_p(g_tts_session, p) : -1;
                         }));
    emscripten::function("sessionSetRepetitionPenalty", emscripten::optional_override([](float r) {
                             return g_tts_session ? crispasr_session_set_repetition_penalty(g_tts_session, r) : -1;
                         }));
    emscripten::function("sessionSetCfgWeight", emscripten::optional_override([](float w) {
                             return g_tts_session ? crispasr_session_set_cfg_weight(g_tts_session, w) : -1;
                         }));
    emscripten::function("sessionSetTtsNoiseTemp", emscripten::optional_override([](float t) {
                             return g_tts_session ? crispasr_session_set_tts_noise_temp(g_tts_session, t) : -1;
                         }));
    emscripten::function("sessionSetExaggeration", emscripten::optional_override([](float e) {
                             return g_tts_session ? crispasr_session_set_exaggeration(g_tts_session, e) : -1;
                         }));
    emscripten::function("sessionSetMaxSpeechTokens", emscripten::optional_override([](int n) {
                             return g_tts_session ? crispasr_session_set_max_speech_tokens(g_tts_session, n) : -1;
                         }));
    emscripten::function("sessionSetLengthScale", emscripten::optional_override([](float s) {
                             return g_tts_session ? crispasr_session_set_length_scale(g_tts_session, s) : -1;
                         }));
    emscripten::function("sessionSetBestOf", emscripten::optional_override([](int n) {
                             return g_tts_session ? crispasr_session_set_best_of(g_tts_session, n) : -1;
                         }));
    emscripten::function("sessionSetBeamSize", emscripten::optional_override([](int n) {
                             return g_tts_session ? crispasr_session_set_beam_size(g_tts_session, n) : -1;
                         }));
    emscripten::function("sessionSetReturnLogits", emscripten::optional_override([](bool enable) {
                             return g_tts_session ? crispasr_session_set_return_logits(g_tts_session, enable ? 1 : 0)
                                                  : -1;
                         }));
    emscripten::function(
        "sessionSetGrammarText",
        emscripten::optional_override([](const std::string& text, const std::string& root, float penalty) {
            return g_tts_session ? crispasr_session_set_grammar_text(g_tts_session, text.c_str(), root.c_str(), penalty)
                                 : -1;
        }));
    emscripten::function("sessionSetFallbackThresholds",
                         emscripten::optional_override([](float entropy, float logprob, float noSpeech, float tempInc) {
                             return g_tts_session ? crispasr_session_set_fallback_thresholds(g_tts_session, entropy,
                                                                                             logprob, noSpeech, tempInc)
                                                  : -1;
                         }));
    emscripten::function("sessionSetAltN", emscripten::optional_override([](int n) {
                             return g_tts_session ? crispasr_session_set_alt_n(g_tts_session, n) : -1;
                         }));
    emscripten::function(
        "sessionSetWhisperDecodeExtras",
        emscripten::optional_override([](bool suppressNst, const std::string& regex, bool carryPrompt) {
            return g_tts_session ? crispasr_session_set_whisper_decode_extras(g_tts_session, suppressNst ? 1 : 0,
                                                                              regex.c_str(), carryPrompt ? 1 : 0)
                                 : -1;
        }));
    emscripten::function("sessionSetAsk", emscripten::optional_override([](const std::string& prompt) {
                             return g_tts_session ? crispasr_session_set_ask(g_tts_session, prompt.c_str()) : -1;
                         }));

    // --- Session ASR ---
    emscripten::function(
        "sessionTranscribe",
        emscripten::optional_override([](const emscripten::val& audio, const std::string& lang) -> emscripten::val {
            if (!g_tts_session)
                return emscripten::val::array();
            const int n = audio["length"].as<int>();
            std::vector<float> pcm(n);
            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];
            emscripten::val mv = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcm.data()), n);
            mv.call<void>("set", audio);

            crispasr_session_result* res;
            if (!lang.empty()) {
                res = crispasr_session_transcribe_lang(g_tts_session, pcm.data(), n, lang.c_str());
            } else {
                res = crispasr_session_transcribe(g_tts_session, pcm.data(), n);
            }
            if (!res)
                return emscripten::val::array();

            int ns = crispasr_session_result_n_segments(res);
            emscripten::val out = emscripten::val::array();
            for (int i = 0; i < ns; i++) {
                emscripten::val seg = emscripten::val::object();
                const char* t = crispasr_session_result_segment_text(res, i);
                seg.set("text", std::string(t ? t : ""));
                seg.set("t0", crispasr_session_result_segment_t0(res, i) / 100.0);
                seg.set("t1", crispasr_session_result_segment_t1(res, i) / 100.0);
                seg.set("noSpeechProb", (double)crispasr_session_result_segment_no_speech_prob(res, i));
                int nw = crispasr_session_result_n_words(res, i);
                emscripten::val words = emscripten::val::array();
                for (int j = 0; j < nw; j++) {
                    emscripten::val w = emscripten::val::object();
                    const char* wt = crispasr_session_result_word_text(res, i, j);
                    w.set("text", std::string(wt ? wt : ""));
                    w.set("t0", crispasr_session_result_word_t0(res, i, j) / 100.0);
                    w.set("t1", crispasr_session_result_word_t1(res, i, j) / 100.0);
                    w.set("p", (double)crispasr_session_result_word_p(res, i, j));
                    words.call<void>("push", w);
                }
                seg.set("words", words);
                out.call<void>("push", seg);
            }
            crispasr_session_result_free(res);
            return out;
        }));

    // Transcribe + CTC logits (backends with a dense CTC grid: Omni CTC,
    // wav2vec2/hubert/data2vec, canary-ctc; opted in via
    // sessionSetReturnLogits). Returns { segments, logits } where segments
    // mirrors sessionTranscribe and logits is { nFrames, nVocab, data } with
    // data a frame-major Float32Array (logits[t*nVocab + v]) — raw pre-softmax
    // for Omni & wav2vec2, log-probabilities for canary-ctc — or null for
    // backends with no dense CTC grid / when no grid was captured.
    emscripten::function(
        "sessionTranscribeWithLogits",
        emscripten::optional_override([](const emscripten::val& audio, const std::string& lang) -> emscripten::val {
            if (!g_tts_session)
                return emscripten::val::null();
            const int n = audio["length"].as<int>();
            std::vector<float> pcm(n);
            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];
            emscripten::val mv = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcm.data()), n);
            mv.call<void>("set", audio);

            crispasr_session_set_return_logits(g_tts_session, 1);
            crispasr_session_result* res;
            if (!lang.empty()) {
                res = crispasr_session_transcribe_lang(g_tts_session, pcm.data(), n, lang.c_str());
            } else {
                res = crispasr_session_transcribe(g_tts_session, pcm.data(), n);
            }
            if (!res) {
                crispasr_session_set_return_logits(g_tts_session, 0);
                return emscripten::val::null();
            }

            emscripten::val out = emscripten::val::object();
            int ns = crispasr_session_result_n_segments(res);
            emscripten::val segs = emscripten::val::array();
            for (int i = 0; i < ns; i++) {
                emscripten::val seg = emscripten::val::object();
                const char* t = crispasr_session_result_segment_text(res, i);
                seg.set("text", std::string(t ? t : ""));
                seg.set("t0", crispasr_session_result_segment_t0(res, i) / 100.0);
                seg.set("t1", crispasr_session_result_segment_t1(res, i) / 100.0);
                seg.set("noSpeechProb", (double)crispasr_session_result_segment_no_speech_prob(res, i));
                int nw = crispasr_session_result_n_words(res, i);
                emscripten::val words = emscripten::val::array();
                for (int j = 0; j < nw; j++) {
                    emscripten::val w = emscripten::val::object();
                    const char* wt = crispasr_session_result_word_text(res, i, j);
                    w.set("text", std::string(wt ? wt : ""));
                    w.set("t0", crispasr_session_result_word_t0(res, i, j) / 100.0);
                    w.set("t1", crispasr_session_result_word_t1(res, i, j) / 100.0);
                    w.set("p", (double)crispasr_session_result_word_p(res, i, j));
                    words.call<void>("push", w);
                }
                seg.set("words", words);
                segs.call<void>("push", seg);
            }
            out.set("segments", segs);

            // Copy the result-owned logit grid into a JS Float32Array before the
            // result (and its buffer) is freed — same owned-then-freed idiom as
            // ttsSynthesize.
            int nf = crispasr_session_result_n_logit_frames(res);
            int nv = crispasr_session_result_n_logit_vocab(res);
            const float* lg = crispasr_session_result_logits(res);
            if (nf > 0 && nv > 0 && lg) {
                const int total = nf * nv;
                emscripten::val data = emscripten::val::global("Float32Array").new_(total);
                emscripten::val memoryView = emscripten::val(emscripten::typed_memory_view(total, lg));
                data.call<void>("set", memoryView);
                emscripten::val logits = emscripten::val::object();
                logits.set("nFrames", nf);
                logits.set("nVocab", nv);
                logits.set("data", data);
                out.set("logits", logits);
            } else {
                out.set("logits", emscripten::val::null());
            }

            crispasr_session_set_return_logits(g_tts_session, 0);
            crispasr_session_result_free(res);
            return out;
        }));

    // CTC vocabulary access (Omni CTC backend). Returns the vocab as a JS array
    // of raw piece strings indexed by token id (word-boundary marker intact —
    // v2 uses a literal space, v1 uses U+2581), for detokenizing a greedy CTC
    // decode over sessionTranscribeWithLogits' grid. null when no session is
    // open or the backend exposes no CTC vocab.
    emscripten::function("sessionCtcVocab", emscripten::optional_override([]() -> emscripten::val {
                             if (!g_tts_session)
                                 return emscripten::val::null();
                             const int n = crispasr_session_n_vocab(g_tts_session);
                             if (n <= 0)
                                 return emscripten::val::null();
                             emscripten::val vocab = emscripten::val::array();
                             for (int i = 0; i < n; i++) {
                                 const char* p = crispasr_session_token_text(g_tts_session, i);
                                 vocab.call<void>("push", std::string(p ? p : ""));
                             }
                             return vocab;
                         }));

    // --- Session translate ---
    emscripten::function(
        "sessionTranslateText", emscripten::optional_override([](const std::string& text, const std::string& src,
                                                                 const std::string& tgt, int maxTokens) -> std::string {
            if (!g_tts_session)
                return "";
            char* res =
                crispasr_session_translate_text(g_tts_session, text.c_str(), src.c_str(), tgt.c_str(), maxTokens);
            if (!res)
                return "";
            std::string out(res);
            crispasr_session_translate_text_free(res);
            return out;
        }));

    // --- Pitch (F0) estimation — CREPE ---
    // Open the model with ttsOpenExplicit(path, "crepe", nThreads) -- always
    // correct. Plain ttsOpen also works from 0.8.15 on, where the GGUF
    // architecture auto-detect gained a `crepe` case; older builds return
    // false for a CREPE model even though the backend is compiled in.
    //
    // `audio` is a mono Float32Array at 16 kHz (query sessionPitchSampleRate
    // rather than hard-coding it); `hopMs` <= 0 uses the model default of
    // 10 ms. Returns [{timeMs, f0Hz, voicedProb}, ...] — the same field names
    // the Dart `PitchFrame` record uses, so the seam is identical across
    // bindings — or [] on failure.
    emscripten::function(
        "sessionPitch", emscripten::optional_override([](const emscripten::val& audio, float hopMs) -> emscripten::val {
            emscripten::val out = emscripten::val::array();
            if (!g_tts_session)
                return out;
            const int n = audio["length"].as<int>();
            if (n <= 0)
                return out;
            std::vector<float> pcmf32(n);
            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];
            emscripten::val view = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
            view.call<void>("set", audio);

            if (crispasr_session_pitch(g_tts_session, pcmf32.data(), n, hopMs) <= 0)
                return out;
            int n_frames = 0;
            const float* frames = crispasr_session_pitch_frames(g_tts_session, &n_frames);
            if (!frames || n_frames <= 0)
                return out;
            for (int i = 0; i < n_frames; i++) {
                emscripten::val f = emscripten::val::object();
                f.set("timeMs", (double)frames[i * 3 + 0]);
                f.set("f0Hz", (double)frames[i * 3 + 1]);
                f.set("voicedProb", (double)frames[i * 3 + 2]);
                out.call<void>("push", f);
            }
            return out;
        }));

    // Native input rate the loaded pitch model expects (16000 for CREPE), or
    // 0 when the session has no pitch arm — doubles as a capability probe.
    emscripten::function("sessionPitchSampleRate", emscripten::optional_override([]() {
                             return g_tts_session ? crispasr_session_pitch_sample_rate(g_tts_session) : 0;
                         }));

    // --- Available backends ---
    emscripten::function("availableBackends", emscripten::optional_override([]() -> std::string {
                             char buf[1024] = {0};
                             crispasr_session_available_backends(buf, sizeof(buf));
                             return std::string(buf);
                         }));

    // --- Detected language (Whisper acoustic; "unknown" for other backends) ---
    emscripten::function("detectedLanguage", emscripten::optional_override([]() -> std::string {
                             if (!g_tts_session)
                                 return std::string("unknown");
                             char buf[32] = {0};
                             crispasr_session_detected_language(g_tts_session, buf, sizeof(buf));
                             return std::string(buf);
                         }));

    // --- Detect backend from GGUF ---
    emscripten::function("detectBackendFromGguf",
                         emscripten::optional_override([](const std::string& path) -> std::string {
                             char out[128] = {0};
                             int rc = crispasr_detect_backend_from_gguf(path.c_str(), out, sizeof(out));
                             // rc > 0 = detected (strlen of name); rc == 0 = valid GGUF, no backend
                             // mapping (name ""); rc < 0 = error. The prior `rc == 0` check returned
                             // the empty name on success and "" on unknown-arch — i.e. always "".
                             return rc > 0 ? std::string(out) : std::string();
                         }));

    // --- LCS dedup ---
    emscripten::function("lcsDedup", emscripten::optional_override([](const emscripten::val& prev,
                                                                      const emscripten::val& curr, int minLen) -> int {
                             int pn = prev["length"].as<int>();
                             int cn = curr["length"].as<int>();
                             std::vector<int> pvec(pn), cvec(cn);
                             for (int i = 0; i < pn; i++)
                                 pvec[i] = prev[i].as<int>();
                             for (int i = 0; i < cn; i++)
                                 cvec[i] = curr[i].as<int>();
                             return crispasr_lcs_dedup_prefix_count(pvec.data(), pn, cvec.data(), cn, minLen);
                         }));

    // --- Kokoro lang helpers ---
    emscripten::function("kokoroLangIsGerman", emscripten::optional_override([](const std::string& lang) -> bool {
                             return crispasr_kokoro_lang_is_german_abi(lang.c_str()) != 0;
                         }));
    emscripten::function("kokoroLangHasNativeVoice", emscripten::optional_override([](const std::string& lang) -> bool {
                             return crispasr_kokoro_lang_has_native_voice_abi(lang.c_str()) != 0;
                         }));

    // --- Text-LID ---
    emscripten::function(
        "textDetectLanguage", emscripten::optional_override([](const std::string& text, const std::string& modelPath,
                                                               int nThreads) -> emscripten::val {
            char label[64] = {0};
            float conf = 0.0f;
            int rc = crispasr_text_detect_language(text.c_str(), modelPath.c_str(), nThreads, label, 64, &conf);
            emscripten::val r = emscripten::val::object();
            r.set("rc", rc);
            r.set("lang", std::string(label));
            r.set("confidence", (double)conf);
            return r;
        }));

    // --- Registry ---
    emscripten::function("registryListBackends", emscripten::optional_override([]() -> std::string {
                             char buf[8192] = {0};
                             crispasr_registry_list_backends_abi(buf, sizeof(buf));
                             return std::string(buf);
                         }));

    // --- Punctuation ---
    emscripten::function("puncInit", emscripten::optional_override([](const std::string& modelPath) -> int {
                             void* h = crispasr_punc_init(modelPath.c_str());
                             return h ? (int)(uintptr_t)h : 0;
                         }));
    emscripten::function("puncProcess",
                         emscripten::optional_override([](int handle, const std::string& text) -> std::string {
                             if (!handle)
                                 return text;
                             const char* r = crispasr_punc_process((void*)(uintptr_t)handle, text.c_str());
                             if (!r)
                                 return text;
                             std::string out(r);
                             crispasr_punc_free_text(r);
                             return out;
                         }));
    emscripten::function("puncFree", emscripten::optional_override([](int handle) {
                             if (handle)
                                 crispasr_punc_free((void*)(uintptr_t)handle);
                         }));

    // Sign an audio CONTAINER (WAV/MP3 bytes — e.g. build a WAV from the
    // Float32Array returned by ttsSynthesize) with C2PA Content Credentials.
    // Args: a Uint8Array of container bytes + a C2PA MIME string ("audio/wav" /
    // "audio/mpeg"). Signs with the bundled self-signed default cert (baked in —
    // no filesystem/openssl needed). Returns a signed Uint8Array, or an empty one
    // on failure / unsupported container (AAC/Opus). #260.
    //
    // WAV is signed by the built-in native C++ signer (crispasr_c2pa_native) —
    // it compiles into every wasm build, so c2paSign("audio/wav") works WITHOUT
    // ./build-wasm.sh --c2pa and adds no ~10 MB c2pa-rs weight. (The pure-JS
    // signer in bindings/javascript/c2pa.mjs is an equivalent module-free option.)
    // MP3/M4A still require --c2pa (the c2pa-rs stack).
    // Wrap a Float32Array of mono PCM into a WAV Uint8Array carrying the
    // interoperable AI-provenance metadata tag (standard WAV LIST/INFO chunk) —
    // the zero-cost provenance floor for the browser (ttsSynthesize returns raw
    // watermarked PCM; wrap it to also get a machine-readable, any-tool-readable
    // AI marker). Optionally feed the result to c2paSign() for a full manifest.
    emscripten::function(
        "pcmToWav", emscripten::optional_override([](const emscripten::val& pcm, int sampleRate) -> emscripten::val {
            std::vector<float> in = emscripten::vecFromJSArray<float>(pcm);
            size_t out_len = 0;
            unsigned char* out = crispasr_pcm_to_wav(in.data(), (int)in.size(), sampleRate, &out_len);
            if (!out || out_len == 0)
                return emscripten::val::global("Uint8Array").new_(0);
            emscripten::val view(emscripten::typed_memory_view(out_len, out));
            emscripten::val result = emscripten::val::global("Uint8Array").new_(view);
            crispasr_c2pa_free(out);
            return result;
        }));

    emscripten::function("c2paSign", emscripten::optional_override([](const emscripten::val& data,
                                                                      const std::string& format) -> emscripten::val {
                             // Copy the JS Uint8Array into wasm memory (no reliance on HEAPU8 being
                             // exposed as a Module property, which varies by build config).
                             std::vector<unsigned char> in = emscripten::vecFromJSArray<unsigned char>(data);
                             size_t out_len = 0;
                             unsigned char* out_ptr =
                                 crispasr_c2pa_sign(in.data(), in.size(), format.c_str(), nullptr, nullptr, &out_len);
                             if (!out_ptr || out_len == 0)
                                 return emscripten::val::global("Uint8Array").new_(0);
                             // View over the signed bytes in the wasm heap, then copy into a fresh
                             // JS Uint8Array so we can free the wasm buffer.
                             emscripten::val view(emscripten::typed_memory_view(out_len, out_ptr));
                             emscripten::val result = emscripten::val::global("Uint8Array").new_(view);
                             crispasr_c2pa_free(out_ptr);
                             return result;
                         }));
}
