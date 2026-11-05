//
// This is the Javascript API of crispasr
//
// Very crude at the moment.
// Feel free to contribute and make this better!
//
// See tests/test-crispasr.js for sample usage.
//

#include "crispasr.h"
#include "crispasr_session.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <atomic>
#include <functional>
#include <memory>
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
static std::atomic<bool> g_compute_thread_set{false};
#endif

// The unified Session C-ABI is declared in crispasr.h (included above)
// as `struct crispasr_session`. Legacy name alias for the Embind wrappers:
typedef struct crispasr_session CrispasrSession;


#include "session_results.h"
#include "session_transcription.h"

static CrispasrSession* g_tts_session = nullptr;
// Backend-agnostic ASR session (parakeet/canary/whisper/…) for the asr* surface
// below — distinct from the whisper-only g_context used by init()/full_default().
static CrispasrSession* g_asr_session = nullptr;

struct whisper_context* g_context;
#include "session_lifetime.h"
#include "session_tasks.h"
#include "session_utilities.h"
#include "session_chat.h"
#include "session_analysis.h"
#include "session_speaker.h"
#include "session_options.h"
#include "session_registry.h"
#include "session_diarization.h"
#include "session_vad.h"

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

#include "session_dispatch.h"
#include "session_async_options.h"
#include "session_async_audio.h"
#include "session_streams.h"

EMSCRIPTEN_BINDINGS(whisper) {
    binding_register_session_tasks();
    binding_register_utilities();
    binding_register_chat();
    binding_register_analysis();
    binding_register_speaker();
    binding_register_open_options();
    binding_register_async_options();
    binding_register_async_audio();
    binding_register_registry();
    binding_register_diarization();
    binding_register_vad();
    binding_register_streams();
    binding_register_asr_callbacks();
#ifdef __EMSCRIPTEN_PTHREADS__
    emscripten::function("browserComputeReady", emscripten::optional_override([]() {
                             binding_check_callback();
                             return g_compute_thread_set.load() && g_compute_thread != pthread_self();
                         }));
    emscripten::function(
        "asrOpenAsync", emscripten::optional_override([](const std::string& path, const std::string& backend,
                                                         int threads, emscripten::val cb) {
            binding_check_reentry();
            browser_dispatch<bool>(
                [path, backend, threads]() {
                    binding_native_close_session(g_asr_session);
                    g_asr_session = backend.empty()
                                        ? crispasr_session_open(path.c_str(), threads)
                                        : crispasr_session_open_explicit(path.c_str(), backend.c_str(), threads);
                    return g_asr_session != nullptr;
                },
                [](bool& value) { return emscripten::val(value); }, cb);
        }));
    emscripten::function(
        "ttsOpenExplicitAsync", emscripten::optional_override([](const std::string& path, const std::string& backend,
                                                                 int threads, emscripten::val cb) {
            binding_check_reentry();
            browser_dispatch<bool>(
                [path, backend, threads]() {
                    binding_native_close_session(g_tts_session);
                    g_tts_session = crispasr_session_open_explicit(path.c_str(), backend.c_str(), threads);
                    return g_tts_session != nullptr;
                },
                [](bool& value) { return emscripten::val(value); }, cb);
        }));
    emscripten::function(
        "asrTranscribeAsync",
        emscripten::optional_override([](const emscripten::val& audio, const std::string& lang, emscripten::val cb) {
            binding_check_reentry();
            auto pcm = std::make_shared<std::vector<float>>(binding_pcm(audio));
            browser_dispatch<std::vector<binding_segment>>(
                [pcm, lang]() {
                    binding_require_no_stream(g_asr_session);
                    std::vector<binding_segment> segments;
                    crispasr_session_result* result =
                        g_asr_session ? crispasr_session_transcribe_lang(g_asr_session, pcm->data(), (int)pcm->size(),
                                                                         lang.empty() ? nullptr : lang.c_str())
                                      : nullptr;
                    if (result) {
                        segments = binding_take_segments(result);
                    }
                    return segments;
                },
                [](std::vector<binding_segment>& segments) { return binding_encode_segments(segments, 1.0); }, cb);
        }));
#endif
    emscripten::function("init", emscripten::optional_override([](const std::string& path_model) {
                             binding_check_reentry();
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
                             binding_check_reentry();
                             if (g_context) {
                                 whisper_free(g_context);
                                 g_context = nullptr;
                             }
                         }));

    emscripten::function(
        "full_default",
        emscripten::optional_override([](const emscripten::val& audio, const std::string& lang, bool translate) {
            binding_check_reentry();
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
            binding_check_reentry();
            binding_native_close_session(g_asr_session);
            g_asr_session = backend.empty()
                                ? crispasr_session_open(model_path.c_str(), n_threads)
                                : crispasr_session_open_explicit(model_path.c_str(), backend.c_str(), n_threads);
            return g_asr_session != nullptr;
        }));

    emscripten::function("asrClose", emscripten::optional_override([]() {
                             binding_check_reentry();
                             binding_native_close_session(g_asr_session);
                         }));

    emscripten::function("asrSetSourceLanguage", emscripten::optional_override([](const std::string& lang) {
                             binding_check_reentry();
                             return g_asr_session ? crispasr_session_set_source_language(g_asr_session, lang.c_str())
                                                  : -1;
                         }));
    emscripten::function("asrSetTargetLanguage", emscripten::optional_override([](const std::string& lang) {
                             binding_check_reentry();
                             return g_asr_session ? crispasr_session_set_target_language(g_asr_session, lang.c_str())
                                                  : -1;
                         }));
    emscripten::function("asrSetTranslate", emscripten::optional_override([](bool enable) {
                             binding_check_reentry();
                             return g_asr_session ? crispasr_session_set_translate(g_asr_session, enable ? 1 : 0) : -1;
                         }));
    emscripten::function("asrSetPunctuation", emscripten::optional_override([](bool enable) {
                             binding_check_reentry();
                             return g_asr_session ? crispasr_session_set_punctuation(g_asr_session, enable ? 1 : 0)
                                                  : -1;
                         }));
    emscripten::function("asrSetPuncModel", emscripten::optional_override([](const std::string& m) {
                             binding_check_reentry();
                             return g_asr_session ? crispasr_session_set_punc_model(g_asr_session, m.c_str()) : -1;
                         }));
    emscripten::function("asrSetBeamSize", emscripten::optional_override([](int n) {
                             binding_check_reentry();
                             return g_asr_session ? crispasr_session_set_beam_size(g_asr_session, n) : -1;
                         }));
    emscripten::function("asrSetHotwords", emscripten::optional_override([](const std::string& w, double boost) {
                             binding_check_reentry();
                             return g_asr_session
                                        ? crispasr_session_set_hotwords(g_asr_session, w.c_str(), (float)boost)
                                        : -1;
                         }));
    // Named bundle of the four decoder fallback thresholds:
    // "conservative" | "balanced" | "aggressive". Returns -2 for an unknown
    // preset (JS callers should treat that as an error, not a no-op) and -1
    // when no session is open.
    emscripten::function("asrSetSensitivity", emscripten::optional_override([](const std::string& preset) {
                             binding_check_reentry();
                             return g_asr_session ? crispasr_session_set_sensitivity(g_asr_session, preset.c_str())
                                                  : -1;
                         }));
    emscripten::function("asrSetAsk", emscripten::optional_override([](const std::string& prompt) {
                             binding_check_reentry();
                             return g_asr_session ? crispasr_session_set_ask(g_asr_session, prompt.c_str()) : -1;
                         }));
    emscripten::function("asrSetTemperature", emscripten::optional_override([](double temp, double seed) {
                             binding_check_reentry();
                             return g_asr_session ? crispasr_session_set_temperature(g_asr_session, (float)temp,
                                                                                     (unsigned long long)seed)
                                                  : -1;
                         }));

    // Transcribe a Float32Array of 16 kHz mono PCM. Returns an array of
    // { t0, t1, text } segments (timestamps in centiseconds). Empty on failure.
    emscripten::function(
        "asrTranscribe",
        emscripten::optional_override([](const emscripten::val& audio, const std::string& lang) -> emscripten::val {
            binding_check_reentry();
            emscripten::val out = emscripten::val::array();
            if (!g_asr_session)
                return out;
            const int n = audio["length"].as<int>();
            std::vector<float> pcmf32(n);
            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];
            emscripten::val view = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
            view.call<void>("set", audio);

            binding_check_reentry();
            binding_require_no_stream(g_asr_session);
            crispasr_session_result* r = crispasr_session_transcribe_lang(g_asr_session, pcmf32.data(), n,
                                                                          lang.empty() ? nullptr : lang.c_str());
            if (!r)
                return out;
            out = binding_encode_segments(binding_take_segments(r), 1.0);
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
            binding_check_reentry();
            emscripten::val out = emscripten::val::array();
            if (!g_asr_session)
                return out;
            const int n = audio["length"].as<int>();
            std::vector<float> pcmf32(n);
            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];
            emscripten::val view = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
            view.call<void>("set", audio);

            binding_check_reentry();
            binding_require_no_stream(g_asr_session);
            crispasr_session_result* r = crispasr_session_transcribe_chunked_lang(
                g_asr_session, pcmf32.data(), n, chunkSeconds, overlapSeconds, lang.empty() ? nullptr : lang.c_str());
            if (!r)
                return out;
            out = binding_encode_segments(binding_take_segments(r), 1.0);
            return out;
        }));

    // asrGetProgress() -> 0..100 (or -1 idle). Long-form progress poll, updated
    // in lockstep with asrTranscribeChunked windows (issue #208).
    emscripten::function("asrGetProgress", emscripten::optional_override([]() {
                             binding_check_callback();
                             return crispasr_get_progress();
                         }));

    // -------------------------------------------------------------------
    // TTS surface (kokoro / vibevoice / qwen3-tts) + kokoro per-language
    // routing (PLAN #56 opt 2b).
    // -------------------------------------------------------------------

    emscripten::function("ttsOpen", emscripten::optional_override([](const std::string& model_path, int n_threads) {
                             binding_check_reentry();
                             binding_native_close_session(g_tts_session);
                             g_tts_session = crispasr_session_open(model_path.c_str(), n_threads <= 0 ? 1 : n_threads);
                             return g_tts_session != nullptr;
                         }));

    // Like ttsOpen but with an explicit backend name (e.g. "kokoro", "piper") — some TTS backends
    // aren't auto-detectable from the GGUF and need it named, exactly as `--backend` does on the CLI.
    emscripten::function(
        "ttsOpenExplicit",
        emscripten::optional_override([](const std::string& model_path, const std::string& backend, int n_threads) {
            binding_check_reentry();
            binding_native_close_session(g_tts_session);
            const int nt = n_threads <= 0 ? 1 : n_threads;
            g_tts_session = backend.empty() ? crispasr_session_open(model_path.c_str(), nt)
                                            : crispasr_session_open_explicit(model_path.c_str(), backend.c_str(), nt);
            return g_tts_session != nullptr;
        }));

    emscripten::function("ttsClose", emscripten::optional_override([]() {
                             binding_check_reentry();
                             binding_native_close_session(g_tts_session);
                         }));

    emscripten::function("ttsSetCodecPath", emscripten::optional_override([](const std::string& path) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_codec_path(g_tts_session, path.c_str()) : -1;
                         }));

    // Drop the kokoro per-session phoneme cache. (PLAN #56 #5)
    emscripten::function("ttsClearPhonemeCache", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_kokoro_clear_phoneme_cache(g_tts_session) : -1;
                         }));

    emscripten::function("ttsSetVoice",
                         emscripten::optional_override([](const std::string& path, const std::string& ref_text) {
                             binding_check_reentry();
                             if (!g_tts_session)
                                 return -1;
                             const char* rt = ref_text.empty() ? nullptr : ref_text.c_str();
                             return crispasr_session_set_voice(g_tts_session, path.c_str(), rt);
                         }));

    // Orpheus preset speakers — set by NAME, not by file path.
    emscripten::function("ttsSetSpeakerName", emscripten::optional_override([](const std::string& name) {
                             binding_check_reentry();
                             if (!g_tts_session)
                                 return -1;
                             return crispasr_session_set_speaker_name(g_tts_session, name.c_str());
                         }));

    // Returns the list of preset speaker names for the active backend
    // (orpheus today). Empty array if the backend has no preset speakers.
    emscripten::function("ttsSpeakers", emscripten::optional_override([]() -> emscripten::val {
                             binding_check_reentry();
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
                             binding_check_reentry();
                             if (!g_tts_session)
                                 return -1;
                             return crispasr_session_set_instruct(g_tts_session, instruct.c_str());
                         }));

    // #316: drive the acoustic model with phonemes, skipping the G2P.
    // kokoro and piper; -2 from any other backend.
    emscripten::function("ttsSetPhonemes", emscripten::optional_override([](const std::string& phonemes) {
                             binding_check_reentry();
                             if (!g_tts_session)
                                 return -1;
                             return crispasr_session_set_tts_phonemes(g_tts_session, phonemes.c_str());
                         }));

    // Pad N ms of silence at the beginning of TTS output. Useful to bypass VLC playback bugs.
    emscripten::function("ttsSetPadSilenceMs", emscripten::optional_override([](int ms) {
                             binding_check_reentry();
                             if (!g_tts_session)
                                 return;
                             crispasr_session_set_tts_pad_silence_ms(g_tts_session, ms);
                         }));

    // qwen3-tts variant detection (returns false also when the active
    // backend isn't qwen3-tts).
    emscripten::function("ttsIsCustomVoice", emscripten::optional_override([]() -> bool {
                             binding_check_reentry();
                             return g_tts_session && crispasr_session_is_custom_voice(g_tts_session) != 0;
                         }));

    emscripten::function("ttsIsVoiceDesign", emscripten::optional_override([]() -> bool {
                             binding_check_reentry();
                             return g_tts_session && crispasr_session_is_voice_design(g_tts_session) != 0;
                         }));

    // Returns a Float32Array of 24 kHz mono PCM. Empty array on failure.
    emscripten::function("ttsSynthesize", emscripten::optional_override([](const std::string& text) -> emscripten::val {
                             binding_check_reentry();
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


    // Declare whose voice a PRESET voice is: "real_person" | "synthetic" |
    // "unknown". Cloning is not the only way to produce a deep fake — a preset
    // voice shipped inside a model can be an identifiable individual, and
    // Art. 3(60) attaches to the audio resembling that person rather than to
    // which pipeline made it. Setting real_person makes the Art. 50(4) reminder
    // fire for a non-cloned voice; it does NOT require a consent attestation.
    // Returns 0, -1 on no session, -2 on an unrecognised value.
    emscripten::function("ttsSetSpeakerIdentity", emscripten::optional_override([](const std::string& identity) {
                             binding_check_reentry();
                             return g_tts_session
                                        ? crispasr_session_set_speaker_identity(g_tts_session, identity.c_str())
                                        : -1;
                         }));

    // #321 parity: attest AI-content marking/disclosure responsibility (EU AI Act
    // Art. 50). REQUIRED before ttsSynthesizeRaw() will return unmarked audio.
    // `attestation` is recorded for audit. Returns 0 on success, -1 otherwise.
    emscripten::function(
        "ttsAcceptMarkingResponsibility", emscripten::optional_override([](const std::string& attestation) {
            binding_check_reentry();
            return g_tts_session ? crispasr_session_accept_marking_responsibility(g_tts_session, attestation.c_str())
                                 : -1;
        }));

    // #321 parity: UNMARKED synthesis (no watermark). Mirrors ttsSynthesize but
    // hard-refused (returns an empty Float32Array) unless
    // ttsAcceptMarkingResponsibility() was called first. Prefer ttsSynthesize
    // for the default watermarked output.
    emscripten::function("ttsSynthesizeRaw",
                         emscripten::optional_override([](const std::string& text) -> emscripten::val {
                             binding_check_reentry();
                             if (!g_tts_session)
                                 return emscripten::val::array();
                             int n = 0;
                             float* pcm = crispasr_session_synthesize_raw(g_tts_session, text.c_str(), &n);
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

    // #321 parity: speech-to-speech (audio in → audio out via a single model
    // pass; lfm2-audio, mini-omni2, sidon, voxcpm2-vae). `audio` is a mono
    // Float32Array at the backend's input rate (see sessionInputSampleRate).
    // Returns { pcm: Float32Array (watermarked, backend-native rate), transcript:
    // string } — an empty pcm + "" transcript on failure or when the backend
    // has no S2S arm. Same owned-then-freed idiom as ttsSynthesize; the optional
    // intermediate transcript is freed with crispasr_session_translate_text_free.
    emscripten::function(
        "ttsSpeechToSpeech", emscripten::optional_override([](const emscripten::val& audio) -> emscripten::val {
            binding_check_reentry();
            emscripten::val result = emscripten::val::object();
            result.set("pcm", emscripten::val::array());
            result.set("transcript", std::string());
            if (!g_tts_session)
                return result;
            const int n = audio["length"].as<int>();
            if (n <= 0)
                return result;
            std::vector<float> in(n);
            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];
            emscripten::val view = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(in.data()), n);
            view.call<void>("set", audio);

            char* out_text = nullptr;
            int out_n = 0;
            float* pcm = crispasr_session_speech_to_speech(g_tts_session, in.data(), n, &out_text, &out_n);
            if (out_text) {
                result.set("transcript", std::string(out_text));
                crispasr_session_translate_text_free(out_text);
            }
            if (!pcm || out_n <= 0) {
                if (pcm)
                    crispasr_pcm_free(pcm);
                return result;
            }
            emscripten::val outPcm = emscripten::val::global("Float32Array").new_(out_n);
            emscripten::val memoryView = emscripten::val(emscripten::typed_memory_view(out_n, pcm));
            outPcm.call<void>("set", memoryView);
            crispasr_pcm_free(pcm);
            result.set("pcm", outPcm);
            return result;
        }));

    // Mirrors python crispasr.kokoro_resolve_for_lang() — returns
    // {modelPath, voicePath, voiceName, backboneSwapped}.
    emscripten::function(
        "kokoroResolveForLang",
        emscripten::optional_override([](const std::string& model_path, const std::string& lang) -> emscripten::val {
            binding_check_reentry();
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
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_source_language(g_tts_session, lang.c_str())
                                                  : -1;
                         }));
    emscripten::function("sessionSetTargetLanguage", emscripten::optional_override([](const std::string& lang) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_target_language(g_tts_session, lang.c_str())
                                                  : -1;
                         }));
    // #329: the language the cloning REFERENCE is spoken in (not the output
    // language) — cosyvoice3 drops the reference transcript when they differ.
    emscripten::function("sessionSetTtsReferenceLanguage", emscripten::optional_override([](const std::string& lang) {
                             binding_check_reentry();
                             return g_tts_session
                                        ? crispasr_session_set_tts_reference_language(g_tts_session, lang.c_str())
                                        : -1;
                         }));
    emscripten::function("sessionSetPunctuation", emscripten::optional_override([](bool enable) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_punctuation(g_tts_session, enable ? 1 : 0)
                                                  : -1;
                         }));
    emscripten::function("sessionSetTranslate", emscripten::optional_override([](bool enable) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_translate(g_tts_session, enable ? 1 : 0) : -1;
                         }));
    emscripten::function(
        "sessionSetTemperature", emscripten::optional_override([](float temp, int seed) {
            binding_check_reentry();
            return g_tts_session ? crispasr_session_set_temperature(g_tts_session, temp, (unsigned long long)seed) : -1;
        }));
    emscripten::function("sessionSetTtsSeed", emscripten::optional_override([](int seed) {
                             binding_check_reentry();
                             return g_tts_session
                                        ? crispasr_session_set_tts_seed(g_tts_session, (unsigned long long)seed)
                                        : -1;
                         }));
    emscripten::function("sessionSetTtsSteps", emscripten::optional_override([](int steps) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_tts_steps(g_tts_session, steps) : -1;
                         }));
    emscripten::function("sessionSetTtsNumCandidates", emscripten::optional_override([](int n) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_tts_num_candidates(g_tts_session, n) : -1;
                         }));
    emscripten::function("sessionSetMaxNewTokens", emscripten::optional_override([](int n) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_max_new_tokens(g_tts_session, n) : -1;
                         }));
    emscripten::function("sessionSetAutochunk", emscripten::optional_override([](bool enabled, int chunk_seconds) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_autochunk(g_tts_session, enabled ? 1 : 0, chunk_seconds) : -1;
                         }));
    emscripten::function("sessionSetMaxNewTokensPerSecond", emscripten::optional_override([](float rate, int minimum) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_max_new_tokens_per_second(g_tts_session, rate, minimum) : -1;
                         }));
    emscripten::function("sessionSetFrequencyPenalty", emscripten::optional_override([](float p) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_frequency_penalty(g_tts_session, p) : -1;
                         }));
    emscripten::function("sessionSetTopP", emscripten::optional_override([](float p) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_top_p(g_tts_session, p) : -1;
                         }));
    emscripten::function("sessionSetTopK", emscripten::optional_override([](int k) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_top_k(g_tts_session, k) : -1;
                         }));
    emscripten::function("sessionSetDoSample", emscripten::optional_override([](bool enable) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_do_sample(g_tts_session, enable ? 1 : 0) : -1;
                         }));
    emscripten::function("sessionSetMinP", emscripten::optional_override([](float p) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_min_p(g_tts_session, p) : -1;
                         }));
    emscripten::function("sessionSetRepetitionPenalty", emscripten::optional_override([](float r) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_repetition_penalty(g_tts_session, r) : -1;
                         }));
    emscripten::function("sessionSetCfgWeight", emscripten::optional_override([](float w) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_cfg_weight(g_tts_session, w) : -1;
                         }));
    emscripten::function("sessionSetTtsNoiseTemp", emscripten::optional_override([](float t) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_tts_noise_temp(g_tts_session, t) : -1;
                         }));
    emscripten::function("sessionSetExaggeration", emscripten::optional_override([](float e) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_exaggeration(g_tts_session, e) : -1;
                         }));
    emscripten::function("sessionSetMaxSpeechTokens", emscripten::optional_override([](int n) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_max_speech_tokens(g_tts_session, n) : -1;
                         }));
    emscripten::function("sessionSetMinSpeechTokens", emscripten::optional_override([](int n) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_min_speech_tokens(g_tts_session, n) : -1;
                         }));
    emscripten::function("sessionSetLengthScale", emscripten::optional_override([](float s) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_length_scale(g_tts_session, s) : -1;
                         }));
    // #321 parity: G2P dict source — "olaph" (MIT), "open-dict" (CC-BY-SA), or a file path.
    emscripten::function("sessionSetG2pDict", emscripten::optional_override([](const std::string& source) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_g2p_dict(g_tts_session, source.c_str()) : -1;
                         }));
    // #321 parity: select a preset speaker by integer id (backends with an id-indexed roster).
    emscripten::function("sessionSetSpeakerId", emscripten::optional_override([](int id) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_speaker_id(g_tts_session, id) : -1;
                         }));
    // #321 parity: sample rate the backend expects for input PCM (16000 for
    // Whisper-family; feed it to S2S/transcribe input). 0 when no session is open.
    emscripten::function("sessionInputSampleRate", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_input_sample_rate(g_tts_session) : 0;
                         }));
    // #332: rate of the PCM ttsSynthesize/speech-to-speech produce (0 = the
    // backend has no audio output), plus the mono channel-count getters.
    emscripten::function("sessionOutputSampleRate", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_output_sample_rate(g_tts_session) : 0;
                         }));
    emscripten::function("sessionInputChannels", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_input_channels(g_tts_session) : 0;
                         }));
    emscripten::function("sessionOutputChannels", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_output_channels(g_tts_session) : 0;
                         }));
    emscripten::function("sessionSetBestOf", emscripten::optional_override([](int n) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_best_of(g_tts_session, n) : -1;
                         }));
    emscripten::function("sessionSetBeamSize", emscripten::optional_override([](int n) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_beam_size(g_tts_session, n) : -1;
                         }));
    emscripten::function("sessionSetReturnLogits", emscripten::optional_override([](bool enable) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_return_logits(g_tts_session, enable ? 1 : 0)
                                                  : -1;
                         }));
    emscripten::function(
        "sessionSetGrammarText",
        emscripten::optional_override([](const std::string& text, const std::string& root, float penalty) {
            binding_check_reentry();
            return g_tts_session ? crispasr_session_set_grammar_text(g_tts_session, text.c_str(), root.c_str(), penalty)
                                 : -1;
        }));
    emscripten::function("sessionSetFallbackThresholds",
                         emscripten::optional_override([](float entropy, float logprob, float noSpeech, float tempInc) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_fallback_thresholds(g_tts_session, entropy,
                                                                                             logprob, noSpeech, tempInc)
                                                  : -1;
                         }));
    emscripten::function("sessionSetAltN", emscripten::optional_override([](int n) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_alt_n(g_tts_session, n) : -1;
                         }));
    emscripten::function(
        "sessionSetWhisperDecodeExtras",
        emscripten::optional_override([](bool suppressNst, const std::string& regex, bool carryPrompt) {
            binding_check_reentry();
            return g_tts_session ? crispasr_session_set_whisper_decode_extras(g_tts_session, suppressNst ? 1 : 0,
                                                                              regex.c_str(), carryPrompt ? 1 : 0)
                                 : -1;
        }));
    emscripten::function("sessionSetAsk", emscripten::optional_override([](const std::string& prompt) {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_set_ask(g_tts_session, prompt.c_str()) : -1;
                         }));

    // --- Session ASR ---
    emscripten::function(
        "sessionTranscribe",
        emscripten::optional_override([](const emscripten::val& audio, const std::string& lang) -> emscripten::val {
            binding_check_reentry();
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

            auto out = binding_encode_segments(binding_take_segments(res), 100.0);
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
            binding_check_reentry();
            if (!g_tts_session)
                return emscripten::val::null();
            auto pcm = binding_pcm(audio);
            return binding_encode_transcription(binding_transcribe_logits(g_tts_session, pcm, lang));
        }));

    // CTC vocabulary access (Omni CTC backend). Returns the vocab as a JS array
    // of raw piece strings indexed by token id (word-boundary marker intact —
    // v2 uses a literal space, v1 uses U+2581), for detokenizing a greedy CTC
    // decode over sessionTranscribeWithLogits' grid. null when no session is
    // open or the backend exposes no CTC vocab.
    emscripten::function("sessionCtcVocab", emscripten::optional_override([]() -> emscripten::val {
                             binding_check_reentry();
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
            binding_check_reentry();
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
            binding_check_reentry();
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

    // Streaming mode of the active backend session (0 when none is loaded).
    emscripten::function("sessionStreamKind", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_stream_kind(g_tts_session) : 0;
                         }));

    // Native input rate the loaded pitch model expects (16000 for CREPE), or
    // 0 when the session has no pitch arm — doubles as a capability probe.
    emscripten::function("sessionPitchSampleRate", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_pitch_sample_rate(g_tts_session) : 0;
                         }));

    // --- Note transcription — basic-pitch, piano-transcription, mt3,
    // onsets-and-frames, hft-transformer ---
    // Open the model with ttsOpenExplicit(path, "<backend>", nThreads), the
    // same way sessionPitch does.
    //
    // `audio` is a mono Float32Array at sessionPianoSampleRate() — 22050 for
    // basic-pitch, 16000 for the other four; ask rather than assume. Returns
    // [{onMs, offMs, midi, velocity, program}, ...], the fields of the Dart
    // `PianoNote` record, or [] on failure. `program` is the General MIDI
    // program MT3 identified, or -1 when the model names no instrument.
    emscripten::function(
        "sessionPianoNotes", emscripten::optional_override([](const emscripten::val& audio) -> emscripten::val {
            binding_check_reentry();
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

            if (crispasr_session_piano(g_tts_session, pcmf32.data(), n) < 0)
                return out;
            int n_notes = 0;
            const float* notes = crispasr_session_piano_notes(g_tts_session, &n_notes);
            if (!notes || n_notes <= 0)
                return out;
            int n_programs = 0;
            const int* programs = crispasr_session_piano_note_programs(g_tts_session, &n_programs);
            for (int i = 0; i < n_notes; i++) {
                emscripten::val e = emscripten::val::object();
                e.set("onMs", (double)notes[i * 4 + 0]);
                e.set("offMs", (double)notes[i * 4 + 1]);
                e.set("midi", (int)notes[i * 4 + 2]);
                e.set("velocity", (int)notes[i * 4 + 3]);
                e.set("program", (programs && i < n_programs) ? programs[i] : -1);
                out.call<void>("push", e);
            }
            return out;
        }));

    // Native input rate of the loaded note model, or 0 when the session has
    // no note arm — a capability probe, mirroring sessionPitchSampleRate.
    emscripten::function("sessionPianoSampleRate", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_piano_sample_rate(g_tts_session) : 0;
                         }));

    // --- Chord recognition (BTC) ---
    //
    // NOTE: the shipped BTC weights are CC-BY-NC-SA, not MIT like this
    // library. A commercial deployment must supply its own weights; the
    // registry will not download these without an explicit licence
    // acceptance. See docs/music-transcription/PLAN.md.
    emscripten::function("sessionChords", emscripten::optional_override([](emscripten::val audio, int sampleRate) {
                             binding_check_reentry();
                             emscripten::val out = emscripten::val::array();
                             if (!g_tts_session)
                                 return out;
                             const int n = audio["length"].as<int>();
                             if (n <= 0 || sampleRate <= 0)
                                 return out;
                             std::vector<float> pcmf32(n);
                             emscripten::val heap = emscripten::val::module_property("HEAPU8");
                             emscripten::val memory = heap["buffer"];
                             emscripten::val view =
                                 audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
                             view.call<void>("set", audio);

                             if (crispasr_session_chords(g_tts_session, pcmf32.data(), n, sampleRate) <= 0)
                                 return out;
                             int n_spans = 0;
                             const float* spans = crispasr_session_chords_spans(g_tts_session, &n_spans);
                             if (!spans || n_spans <= 0)
                                 return out;
                             for (int i = 0; i < n_spans; i++) {
                                 emscripten::val c = emscripten::val::object();
                                 c.set("startMs", (double)spans[i * 4 + 0]);
                                 c.set("endMs", (double)spans[i * 4 + 1]);
                                 const char* nm = crispasr_session_chords_span_name(g_tts_session, i);
                                 c.set("chord", std::string(nm ? nm : "N"));
                                 c.set("confidence", (double)spans[i * 4 + 3]);
                                 out.call<void>("push", c);
                             }
                             return out;
                         }));

    // 25 or 170, or 0 when the session has no chord arm — capability probe,
    // mirroring sessionPitchSampleRate.
    emscripten::function("sessionChordsVocabSize", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return g_tts_session ? crispasr_session_chords_vocab_size(g_tts_session) : 0;
                         }));

    // --- Available backends ---
    emscripten::function("availableBackends", emscripten::optional_override([]() -> std::string {
                             binding_check_reentry();
                             char buf[1024] = {0};
                             crispasr_session_available_backends(buf, sizeof(buf));
                             return std::string(buf);
                         }));

    // --- Detected language (Whisper acoustic; "unknown" for other backends) ---
    emscripten::function("detectedLanguage", emscripten::optional_override([]() -> std::string {
                             binding_check_reentry();
                             if (!g_tts_session)
                                 return std::string("unknown");
                             char buf[32] = {0};
                             crispasr_session_detected_language(g_tts_session, buf, sizeof(buf));
                             return std::string(buf);
                         }));

    // --- Detect backend from GGUF ---
    emscripten::function("detectBackendFromGguf",
                         emscripten::optional_override([](const std::string& path) -> std::string {
                             binding_check_reentry();
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
                             binding_check_reentry();
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
                             binding_check_reentry();
                             return crispasr_kokoro_lang_is_german_abi(lang.c_str()) != 0;
                         }));
    emscripten::function("kokoroLangHasNativeVoice", emscripten::optional_override([](const std::string& lang) -> bool {
                             binding_check_reentry();
                             return crispasr_kokoro_lang_has_native_voice_abi(lang.c_str()) != 0;
                         }));

    // --- Text-LID ---
    emscripten::function(
        "textDetectLanguage", emscripten::optional_override([](const std::string& text, const std::string& modelPath,
                                                               int nThreads) -> emscripten::val {
            binding_check_reentry();
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
                             binding_check_reentry();
                             char buf[8192] = {0};
                             crispasr_registry_list_backends_abi(buf, sizeof(buf));
                             return std::string(buf);
                         }));

    // --- Punctuation ---
    emscripten::function("puncInit", emscripten::optional_override([](const std::string& modelPath) -> int {
                             binding_check_reentry();
                             void* h = crispasr_punc_init(modelPath.c_str());
                             return h ? (int)(uintptr_t)h : 0;
                         }));
    emscripten::function("puncProcess",
                         emscripten::optional_override([](int handle, const std::string& text) -> std::string {
                             binding_check_reentry();
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
                             binding_check_reentry();
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
            binding_check_reentry();
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
                             binding_check_reentry();
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
