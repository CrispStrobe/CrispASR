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

// Forward-declare the unified Session + kokoro routing C ABI exported
// by libcrispasr. The full prototypes live in src/crispasr_c_api.cpp
// and src/kokoro.h. Including those headers from emscripten would pull
// in C++/STL declarations we don't need here.
extern "C" {
struct CrispasrSession;
struct CrispasrSession* crispasr_session_open(const char* model_path, int n_threads);
void                    crispasr_session_close(struct CrispasrSession* s);
int                     crispasr_session_set_codec_path(struct CrispasrSession* s, const char* path);
int                     crispasr_session_set_voice(struct CrispasrSession* s, const char* path,
                                                   const char* ref_text_or_null);
int                     crispasr_session_set_speaker_name(struct CrispasrSession* s, const char* name);
int                     crispasr_session_n_speakers(struct CrispasrSession* s);
const char*             crispasr_session_get_speaker_name(struct CrispasrSession* s, int i);
int                     crispasr_session_set_instruct(struct CrispasrSession* s, const char* instruct);
int                     crispasr_session_is_custom_voice(struct CrispasrSession* s);
int                     crispasr_session_is_voice_design(struct CrispasrSession* s);
float*                  crispasr_session_synthesize(struct CrispasrSession* s, const char* text,
                                                    int* out_n_samples);
void                    crispasr_pcm_free(float* pcm);
int                     crispasr_session_kokoro_clear_phoneme_cache(struct CrispasrSession* s);
int                     crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang,
                                                                   char* out_path, int out_path_len);
int                     crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang,
                                                                   char* out_path, int out_path_len,
                                                                   char* out_picked, int out_picked_len);

// --- Full C-ABI parity declarations ---
// Session extras
int          crispasr_session_available_backends(char* out_csv, int out_cap);
struct CrispasrSession* crispasr_session_open_explicit(const char* model_path, const char* backend_name, int n_threads);
struct CrispasrSession* crispasr_session_open_with_params(const char* model_path, const char* backend_name, const void* params);
const char*  crispasr_session_backend(struct CrispasrSession* s);
int          crispasr_session_set_source_language(struct CrispasrSession* s, const char* lang);
int          crispasr_session_set_target_language(struct CrispasrSession* s, const char* lang);
int          crispasr_session_set_punctuation(struct CrispasrSession* s, int enable);
int          crispasr_session_set_translate(struct CrispasrSession* s, int enable);
int          crispasr_session_set_temperature(struct CrispasrSession* s, float temperature, unsigned long long seed);
int          crispasr_session_set_tts_seed(struct CrispasrSession* s, unsigned long long seed);
int          crispasr_session_set_tts_steps(struct CrispasrSession* s, int steps);
int          crispasr_session_set_max_new_tokens(struct CrispasrSession* s, int n);
int          crispasr_session_set_frequency_penalty(struct CrispasrSession* s, float penalty);
int          crispasr_session_set_top_p(struct CrispasrSession* s, float top_p);
int          crispasr_session_set_min_p(struct CrispasrSession* s, float min_p);
int          crispasr_session_set_repetition_penalty(struct CrispasrSession* s, float r);
int          crispasr_session_set_cfg_weight(struct CrispasrSession* s, float cfg_weight);
int          crispasr_session_set_exaggeration(struct CrispasrSession* s, float exaggeration);
int          crispasr_session_set_max_speech_tokens(struct CrispasrSession* s, int n);
int          crispasr_session_set_length_scale(struct CrispasrSession* s, float scale);
int          crispasr_session_set_best_of(struct CrispasrSession* s, int n);
int          crispasr_session_set_beam_size(struct CrispasrSession* s, int n);
int          crispasr_session_set_grammar_text(struct CrispasrSession* s, const char* gbnf_text,
                                               const char* root_rule, float penalty);
int          crispasr_session_set_fallback_thresholds(struct CrispasrSession* s, float entropy_thold,
                                                      float logprob_thold, float no_speech_thold,
                                                      float temperature_inc);
int          crispasr_session_set_alt_n(struct CrispasrSession* s, int n);
int          crispasr_session_set_whisper_decode_extras(struct CrispasrSession* s, int suppress_nst,
                                                        const char* suppress_regex, int carry_initial_prompt);
int          crispasr_session_set_ask(struct CrispasrSession* s, const char* prompt);
int          crispasr_session_detect_language(struct CrispasrSession* s, const float* pcm, int n_samples,
                                              const char* lid_model_path, int method,
                                              char* out_lang, int out_lang_cap, float* out_prob);

// Session ASR transcription
struct crispasr_session_result;
struct crispasr_session_result* crispasr_session_transcribe(struct CrispasrSession* s, const float* pcm, int n_samples);
struct crispasr_session_result* crispasr_session_transcribe_lang(struct CrispasrSession* s, const float* pcm, int n_samples,
                                                                  const char* language);
struct crispasr_session_result* crispasr_session_transcribe_vad(struct CrispasrSession* s, const float* pcm, int n_samples,
                                                                 int sample_rate, const char* vad_model_path, void* opts);
struct crispasr_session_result* crispasr_session_transcribe_vad_lang(struct CrispasrSession* s, const float* pcm, int n_samples,
                                                                      int sample_rate, const char* vad_model_path, void* opts,
                                                                      const char* language);
int          crispasr_session_result_n_segments(struct crispasr_session_result* r);
const char*  crispasr_session_result_segment_text(struct crispasr_session_result* r, int i);
long long    crispasr_session_result_segment_t0(struct crispasr_session_result* r, int i);
long long    crispasr_session_result_segment_t1(struct crispasr_session_result* r, int i);
int          crispasr_session_result_n_words(struct crispasr_session_result* r, int i_seg);
const char*  crispasr_session_result_word_text(struct crispasr_session_result* r, int i_seg, int i_word);
long long    crispasr_session_result_word_t0(struct crispasr_session_result* r, int i_seg, int i_word);
long long    crispasr_session_result_word_t1(struct crispasr_session_result* r, int i_seg, int i_word);
float        crispasr_session_result_word_p(struct crispasr_session_result* r, int i_seg, int i_word);
int          crispasr_session_result_word_n_alts(struct crispasr_session_result* r, int i_seg, int i_word);
const char*  crispasr_session_result_word_alt_text(struct crispasr_session_result* r, int i_seg, int i_word, int i_alt);
float        crispasr_session_result_word_alt_p(struct crispasr_session_result* r, int i_seg, int i_word, int i_alt);
void         crispasr_session_result_free(struct crispasr_session_result* r);
char*        crispasr_session_translate_text(struct CrispasrSession* s, const char* text, const char* src_lang,
                                             const char* tgt_lang, int max_tokens);
void         crispasr_session_translate_text_free(char* text);

// Streaming
struct CrispasrStream;
struct CrispasrStream* crispasr_session_stream_open(struct CrispasrSession* s, int n_threads, int step_ms,
                                                     int length_ms, int keep_ms, const char* language, int translate);
struct CrispasrStream* crispasr_stream_open(void* ctx, int n_threads, int step_ms,
                                             int length_ms, int keep_ms, const char* language, int translate);
int                    crispasr_stream_feed(struct CrispasrStream* s, const float* pcm, int n_samples);
int                    crispasr_stream_get_text(struct CrispasrStream* s, char* out_text, int out_cap,
                                                double* out_t0_s, double* out_t1_s, long long* out_counter);
int                    crispasr_stream_flush(struct CrispasrStream* s);
void                   crispasr_stream_close(struct CrispasrStream* s);
void                   crispasr_stream_set_live_decode(struct CrispasrStream* s, int enabled);

// Punctuation
void*        crispasr_punc_init(const char* model_path);
const char*  crispasr_punc_process(void* ctx, const char* text);
void         crispasr_punc_free_text(const char* text);
void         crispasr_punc_free(void* ctx);

// Alignment
struct crispasr_align_result;
struct crispasr_align_result* crispasr_align_words_abi(const char* aligner_model, const char* transcript,
                                                       const float* samples, int n_samples, long long t_offset_cs,
                                                       int n_threads);
int          crispasr_align_result_n_words(struct crispasr_align_result* r);
const char*  crispasr_align_result_word_text(struct crispasr_align_result* r, int i);
long long    crispasr_align_result_word_t0(struct crispasr_align_result* r, int i);
long long    crispasr_align_result_word_t1(struct crispasr_align_result* r, int i);
void         crispasr_align_result_free(struct crispasr_align_result* r);

// VAD
int  crispasr_vad_segments(const char* vad_model_path, const float* pcm, int n_samples,
                           int sample_rate, float threshold, int min_speech_ms, int min_silence_ms,
                           int n_threads, int use_gpu, float** out_spans);
int  crispasr_vad_slices(const char* vad_model_path, const float* pcm, int n_samples,
                         int sample_rate, float threshold, int min_speech_ms, int min_silence_ms,
                         int speech_pad_ms, float max_chunk_duration_s, int n_threads,
                         float** out_spans);
void crispasr_vad_free(float* spans);

// LCS dedup
int crispasr_lcs_dedup_prefix_count(const int* prev_tail_tokens, int n_prev,
                                    const int* curr_tokens, int n_curr, int min_lcs_length);

// params_set_* on whisper_full_params
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

// Token-level accessors
long long crispasr_token_t0(void* ctx, int i_seg, int i_tok);
long long crispasr_token_t1(void* ctx, int i_seg, int i_tok);
float     crispasr_token_p(void* ctx, int i_seg, int i_tok);
int       crispasr_token_n_alts(void* ctx, int i_seg, int i_tok);
int       crispasr_token_alt_id(void* ctx, int i_seg, int i_tok, int i_alt);
float     crispasr_token_alt_p(void* ctx, int i_seg, int i_tok, int i_alt);
int       crispasr_token_alt_text(void* ctx, int i_seg, int i_tok, int i_alt, char* out, int out_cap);

// Language detection
float crispasr_detect_language(void* ctx, const float* pcm, int n_samples,
                               int n_threads, char* out_code, int out_cap);
int   crispasr_detect_language_pcm(const float* samples, int n_samples, int method,
                                    const char* model_path, int n_threads, int use_gpu,
                                    int gpu_device, int flash_attn,
                                    char* out_lang, int out_lang_cap, float* out_confidence);

// Direct Parakeet API
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

// Backend detection
int crispasr_detect_backend_from_gguf(const char* path, char* out_name, int out_cap);

// RNNoise audio enhancement
int crispasr_enhance_audio_rnnoise(const float* in_pcm, int n_samples, float* out_pcm, int out_cap);

// Text-LID
int crispasr_text_detect_language(const char* text, const char* model_path, int n_threads,
                                   char* out_label, int out_label_cap, float* out_confidence);

// TitaNet
void* crispasr_titanet_init(const char* model_path, int n_threads);
void  crispasr_titanet_free(void* ctx);
int   crispasr_titanet_embed(void* ctx, const float* pcm_16k, int n_samples, float* out);
float crispasr_titanet_cosine_sim(const float* a, const float* b, int dim);

// Speaker database
void* crispasr_speaker_db_load(const char* dir_path);
void  crispasr_speaker_db_free(void* db);
int   crispasr_speaker_db_count(const void* db);
float crispasr_speaker_db_match(const void* db, const float* embedding, int dim,
                                float threshold, char* out_name, int out_cap);
int   crispasr_speaker_db_enroll(const char* dir_path, const char* name,
                                 const float* embedding, int dim);

// Pluggable speaker embedder + clustering + pyannote cache
void*  crispasr_speaker_embedder_make_abi(const char* model_spec, int n_threads, const char* cache_dir);
void   crispasr_speaker_embedder_free_abi(void* embedder);
int    crispasr_speaker_embedder_dim_abi(const void* embedder);
int    crispasr_speaker_embedder_embed_abi(void* embedder, const float* pcm_16k, int n_samples, float* out);
const char* crispasr_speaker_embedder_name_abi(const void* embedder);
int    crispasr_speaker_cluster_abi(const float* embeddings, int n, int dim,
                                    float merge_threshold, int max_speakers, int* labels_out);
void*  crispasr_pyannote_cache_compute_abi(const float* full_audio, int n_samples,
                                            const char* model_path, int n_threads);
void   crispasr_pyannote_cache_free_abi(void* cache);
int    crispasr_pyannote_cache_apply_abi(const void* cache, long long slice_t0_cs,
                                         void* segs, int n_segs);

// Kokoro lang helpers
int crispasr_kokoro_lang_is_german_abi(const char* lang);
int crispasr_kokoro_lang_has_native_voice_abi(const char* lang);

// Registry + cache
int crispasr_registry_lookup_abi(const char* backend, char* out_filename, int filename_cap,
                                 char* out_url, int url_cap, char* out_size, int size_cap);
int crispasr_registry_lookup_by_filename_abi(const char* filename, char* out_filename, int filename_cap,
                                             char* out_url, int url_cap, char* out_size, int size_cap);
int crispasr_registry_list_backends_abi(char* out_csv, int out_cap);
int crispasr_cache_ensure_file_abi(const char* filename, const char* url, int quiet,
                                   const char* cache_dir_override, char* out_buf, int out_cap);
int crispasr_cache_dir_abi(const char* cache_dir_override, char* out_buf, int out_cap);

// Diarization
int crispasr_diarize_segments_abi(const float* left_pcm, const float* right_pcm, int n_samples,
                                  int is_stereo, void* segs, int n_segs, const void* opts);
}

static struct CrispasrSession* g_tts_session = nullptr;

struct whisper_context* g_context;

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
    // TTS surface (kokoro / vibevoice / qwen3-tts) + kokoro per-language
    // routing (PLAN #56 opt 2b).
    // -------------------------------------------------------------------

    emscripten::function("ttsOpen", emscripten::optional_override([](const std::string& model_path,
                                                                     int n_threads) {
                             if (g_tts_session != nullptr) {
                                 crispasr_session_close(g_tts_session);
                                 g_tts_session = nullptr;
                             }
                             g_tts_session = crispasr_session_open(model_path.c_str(),
                                                                   n_threads <= 0 ? 1 : n_threads);
                             return g_tts_session != nullptr;
                         }));

    emscripten::function("ttsClose", emscripten::optional_override([]() {
                             if (g_tts_session) {
                                 crispasr_session_close(g_tts_session);
                                 g_tts_session = nullptr;
                             }
                         }));

    emscripten::function("ttsSetCodecPath", emscripten::optional_override([](const std::string& path) {
                             return g_tts_session ? crispasr_session_set_codec_path(g_tts_session,
                                                                                    path.c_str())
                                                  : -1;
                         }));

    // Drop the kokoro per-session phoneme cache. (PLAN #56 #5)
    emscripten::function("ttsClearPhonemeCache", emscripten::optional_override([]() {
                             return g_tts_session ? crispasr_session_kokoro_clear_phoneme_cache(g_tts_session)
                                                  : -1;
                         }));

    emscripten::function("ttsSetVoice", emscripten::optional_override([](const std::string& path,
                                                                         const std::string& ref_text) {
                             if (!g_tts_session) return -1;
                             const char* rt = ref_text.empty() ? nullptr : ref_text.c_str();
                             return crispasr_session_set_voice(g_tts_session, path.c_str(), rt);
                         }));

    // Orpheus preset speakers — set by NAME, not by file path.
    emscripten::function("ttsSetSpeakerName",
                         emscripten::optional_override([](const std::string& name) {
                             if (!g_tts_session) return -1;
                             return crispasr_session_set_speaker_name(g_tts_session, name.c_str());
                         }));

    // Returns the list of preset speaker names for the active backend
    // (orpheus today). Empty array if the backend has no preset speakers.
    emscripten::function("ttsSpeakers",
                         emscripten::optional_override([]() -> emscripten::val {
                             emscripten::val out = emscripten::val::array();
                             if (!g_tts_session) return out;
                             int n = crispasr_session_n_speakers(g_tts_session);
                             for (int i = 0; i < n; i++) {
                                 const char* name = crispasr_session_get_speaker_name(g_tts_session, i);
                                 if (name) out.call<void>("push", std::string(name));
                             }
                             return out;
                         }));

    // qwen3-tts VoiceDesign — natural-language voice description.
    emscripten::function("ttsSetInstruct",
                         emscripten::optional_override([](const std::string& instruct) {
                             if (!g_tts_session) return -1;
                             return crispasr_session_set_instruct(g_tts_session, instruct.c_str());
                         }));

    // qwen3-tts variant detection (returns false also when the active
    // backend isn't qwen3-tts).
    emscripten::function("ttsIsCustomVoice",
                         emscripten::optional_override([]() -> bool {
                             return g_tts_session && crispasr_session_is_custom_voice(g_tts_session) != 0;
                         }));

    emscripten::function("ttsIsVoiceDesign",
                         emscripten::optional_override([]() -> bool {
                             return g_tts_session && crispasr_session_is_voice_design(g_tts_session) != 0;
                         }));

    // Returns a Float32Array of 24 kHz mono PCM. Empty array on failure.
    emscripten::function("ttsSynthesize",
                         emscripten::optional_override([](const std::string& text) -> emscripten::val {
                             if (!g_tts_session) return emscripten::val::array();
                             int n = 0;
                             float* pcm = crispasr_session_synthesize(g_tts_session, text.c_str(), &n);
                             if (!pcm || n <= 0) {
                                 if (pcm) crispasr_pcm_free(pcm);
                                 return emscripten::val::array();
                             }
                             emscripten::val out = emscripten::val::global("Float32Array").new_(n);
                             emscripten::val memoryView = emscripten::val(emscripten::typed_memory_view(n, pcm));
                             out.call<void>("set", memoryView);
                             crispasr_pcm_free(pcm);
                             return out;
                         }));

    // Mirrors python crispasr.kokoro_resolve_for_lang() — returns
    // {modelPath, voicePath, voiceName, backboneSwapped}.
    emscripten::function("kokoroResolveForLang",
                         emscripten::optional_override([](const std::string& model_path,
                                                          const std::string& lang) -> emscripten::val {
                             char out_model[1024]  = {0};
                             char out_voice[1024]  = {0};
                             char out_picked[64]   = {0};

                             int rc = crispasr_kokoro_resolve_model_for_lang_abi(
                                 model_path.c_str(), lang.c_str(), out_model, sizeof(out_model));
                             bool swapped = (rc == 0);
                             std::string resolved = (out_model[0] != 0) ? std::string(out_model) : model_path;

                             std::string vp, vn;
                             rc = crispasr_kokoro_resolve_fallback_voice_abi(
                                 model_path.c_str(), lang.c_str(),
                                 out_voice, sizeof(out_voice),
                                 out_picked, sizeof(out_picked));
                             if (rc == 0) {
                                 vp = out_voice;
                                 vn = out_picked;
                             }

                             emscripten::val r = emscripten::val::object();
                             r.set("modelPath",       resolved);
                             r.set("voicePath",       vp);
                             r.set("voiceName",       vn);
                             r.set("backboneSwapped", swapped);
                             return r;
                         }));

    // -------------------------------------------------------------------
    // Full C-ABI parity wrappers (PLAN #59)
    // -------------------------------------------------------------------

    // --- Session setters ---
    emscripten::function("sessionSetSourceLanguage", emscripten::optional_override([](const std::string& lang) {
        return g_tts_session ? crispasr_session_set_source_language(g_tts_session, lang.c_str()) : -1;
    }));
    emscripten::function("sessionSetTargetLanguage", emscripten::optional_override([](const std::string& lang) {
        return g_tts_session ? crispasr_session_set_target_language(g_tts_session, lang.c_str()) : -1;
    }));
    emscripten::function("sessionSetPunctuation", emscripten::optional_override([](bool enable) {
        return g_tts_session ? crispasr_session_set_punctuation(g_tts_session, enable ? 1 : 0) : -1;
    }));
    emscripten::function("sessionSetTranslate", emscripten::optional_override([](bool enable) {
        return g_tts_session ? crispasr_session_set_translate(g_tts_session, enable ? 1 : 0) : -1;
    }));
    emscripten::function("sessionSetTemperature", emscripten::optional_override([](float temp, int seed) {
        return g_tts_session ? crispasr_session_set_temperature(g_tts_session, temp, (unsigned long long)seed) : -1;
    }));
    emscripten::function("sessionSetTtsSeed", emscripten::optional_override([](int seed) {
        return g_tts_session ? crispasr_session_set_tts_seed(g_tts_session, (unsigned long long)seed) : -1;
    }));
    emscripten::function("sessionSetTtsSteps", emscripten::optional_override([](int steps) {
        return g_tts_session ? crispasr_session_set_tts_steps(g_tts_session, steps) : -1;
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
    emscripten::function("sessionSetMinP", emscripten::optional_override([](float p) {
        return g_tts_session ? crispasr_session_set_min_p(g_tts_session, p) : -1;
    }));
    emscripten::function("sessionSetRepetitionPenalty", emscripten::optional_override([](float r) {
        return g_tts_session ? crispasr_session_set_repetition_penalty(g_tts_session, r) : -1;
    }));
    emscripten::function("sessionSetCfgWeight", emscripten::optional_override([](float w) {
        return g_tts_session ? crispasr_session_set_cfg_weight(g_tts_session, w) : -1;
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
    emscripten::function("sessionSetGrammarText", emscripten::optional_override(
        [](const std::string& text, const std::string& root, float penalty) {
            return g_tts_session ? crispasr_session_set_grammar_text(
                g_tts_session, text.c_str(), root.c_str(), penalty) : -1;
        }));
    emscripten::function("sessionSetFallbackThresholds", emscripten::optional_override(
        [](float entropy, float logprob, float noSpeech, float tempInc) {
            return g_tts_session ? crispasr_session_set_fallback_thresholds(
                g_tts_session, entropy, logprob, noSpeech, tempInc) : -1;
        }));
    emscripten::function("sessionSetAltN", emscripten::optional_override([](int n) {
        return g_tts_session ? crispasr_session_set_alt_n(g_tts_session, n) : -1;
    }));
    emscripten::function("sessionSetWhisperDecodeExtras", emscripten::optional_override(
        [](bool suppressNst, const std::string& regex, bool carryPrompt) {
            return g_tts_session ? crispasr_session_set_whisper_decode_extras(
                g_tts_session, suppressNst ? 1 : 0, regex.c_str(), carryPrompt ? 1 : 0) : -1;
        }));
    emscripten::function("sessionSetAsk", emscripten::optional_override([](const std::string& prompt) {
        return g_tts_session ? crispasr_session_set_ask(g_tts_session, prompt.c_str()) : -1;
    }));

    // --- Session ASR ---
    emscripten::function("sessionTranscribe", emscripten::optional_override(
        [](const emscripten::val& audio, const std::string& lang) -> emscripten::val {
            if (!g_tts_session) return emscripten::val::array();
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
            if (!res) return emscripten::val::array();

            int ns = crispasr_session_result_n_segments(res);
            emscripten::val out = emscripten::val::array();
            for (int i = 0; i < ns; i++) {
                emscripten::val seg = emscripten::val::object();
                const char* t = crispasr_session_result_segment_text(res, i);
                seg.set("text", std::string(t ? t : ""));
                seg.set("t0", crispasr_session_result_segment_t0(res, i) / 100.0);
                seg.set("t1", crispasr_session_result_segment_t1(res, i) / 100.0);
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

    // --- Session translate ---
    emscripten::function("sessionTranslateText", emscripten::optional_override(
        [](const std::string& text, const std::string& src, const std::string& tgt, int maxTokens) -> std::string {
            if (!g_tts_session) return "";
            char* res = crispasr_session_translate_text(g_tts_session, text.c_str(),
                                                        src.c_str(), tgt.c_str(), maxTokens);
            if (!res) return "";
            std::string out(res);
            crispasr_session_translate_text_free(res);
            return out;
        }));

    // --- Available backends ---
    emscripten::function("availableBackends", emscripten::optional_override([]() -> std::string {
        char buf[1024] = {0};
        crispasr_session_available_backends(buf, sizeof(buf));
        return std::string(buf);
    }));

    // --- Detect backend from GGUF ---
    emscripten::function("detectBackendFromGguf", emscripten::optional_override(
        [](const std::string& path) -> std::string {
            char out[128] = {0};
            int rc = crispasr_detect_backend_from_gguf(path.c_str(), out, sizeof(out));
            return rc == 0 ? std::string(out) : "";
        }));

    // --- LCS dedup ---
    emscripten::function("lcsDedup", emscripten::optional_override(
        [](const emscripten::val& prev, const emscripten::val& curr, int minLen) -> int {
            int pn = prev["length"].as<int>();
            int cn = curr["length"].as<int>();
            std::vector<int> pvec(pn), cvec(cn);
            for (int i = 0; i < pn; i++) pvec[i] = prev[i].as<int>();
            for (int i = 0; i < cn; i++) cvec[i] = curr[i].as<int>();
            return crispasr_lcs_dedup_prefix_count(pvec.data(), pn, cvec.data(), cn, minLen);
        }));

    // --- Kokoro lang helpers ---
    emscripten::function("kokoroLangIsGerman", emscripten::optional_override(
        [](const std::string& lang) -> bool {
            return crispasr_kokoro_lang_is_german_abi(lang.c_str()) != 0;
        }));
    emscripten::function("kokoroLangHasNativeVoice", emscripten::optional_override(
        [](const std::string& lang) -> bool {
            return crispasr_kokoro_lang_has_native_voice_abi(lang.c_str()) != 0;
        }));

    // --- Text-LID ---
    emscripten::function("textDetectLanguage", emscripten::optional_override(
        [](const std::string& text, const std::string& modelPath, int nThreads) -> emscripten::val {
            char label[64] = {0};
            float conf = 0.0f;
            int rc = crispasr_text_detect_language(text.c_str(), modelPath.c_str(),
                                                   nThreads, label, 64, &conf);
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
    emscripten::function("puncInit", emscripten::optional_override(
        [](const std::string& modelPath) -> int {
            void* h = crispasr_punc_init(modelPath.c_str());
            return h ? (int)(uintptr_t)h : 0;
        }));
    emscripten::function("puncProcess", emscripten::optional_override(
        [](int handle, const std::string& text) -> std::string {
            if (!handle) return text;
            const char* r = crispasr_punc_process((void*)(uintptr_t)handle, text.c_str());
            if (!r) return text;
            std::string out(r);
            crispasr_punc_free_text(r);
            return out;
        }));
    emscripten::function("puncFree", emscripten::optional_override([](int handle) {
        if (handle) crispasr_punc_free((void*)(uintptr_t)handle);
    }));
}
