// CrispASR — C-ABI consumed by every CrispASR consumer: the CLI in
// `examples/cli/`, the Dart FFI binding in `flutter/crispasr/`, the Python
// ctypes binding in `python/crispasr/`, and the Rust `crispasr-sys` crate.
// These wrap the handful of whisper.h entry points that external callers
// can't reach directly (functions that take or return structs by value,
// plus convenience wrappers that would otherwise force each binding to
// mirror the full `whisper_full_params` / `whisper_token_data` layouts),
// and also expose the higher-level CrispASR session/VAD/diarize surface.
//
// Every symbol here is plain C linkage, prefixed `crispasr_` so it can't
// collide with upstream whisper.h identifiers. Keep signatures stable once
// published — these are part of CrispASR's published ABI contract shared
// across all four consumers above.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "whisper.h"
#include "crispasr_vad.h"            // VAD slicing + stitching (shared with CLI)
#include "crispasr_diarize.h"        // Speaker diarization (shared with CLI)
#include "crispasr_lid.h"            // Language identification (shared with CLI)
#include "crispasr_aligner.h"        // CTC / forced-aligner word timings (shared with CLI)
#include "crispasr_cache.h"          // HF download + filesystem cache (shared with CLI)
#include "crispasr_model_registry.h" // Known-model lookup (shared with CLI)
// Non-Whisper backend headers. Each of these lives in `src/` and is built as
// its own shared library — we link them into libwhisper privately so Dart
// only has to open one library to reach every backend. Any missing header
// in a slim build is skipped cleanly below.
#if __has_include("parakeet.h")
#include "parakeet.h"
#define CA_HAVE_PARAKEET 1
#endif
#if __has_include("canary.h")
#include "canary.h"
#define CA_HAVE_CANARY 1
#endif
#if __has_include("qwen3_asr.h")
#include "qwen3_asr.h"
#define CA_HAVE_QWEN3 1
#endif
#if __has_include("cohere.h")
#include "cohere.h"
#define CA_HAVE_COHERE 1
#endif
#if __has_include("granite_speech.h")
#include "granite_speech.h"
#define CA_HAVE_GRANITE 1
#endif
#if __has_include("canary_ctc.h")
#include "canary_ctc.h"
#define CA_HAVE_CTC 1
#endif
#if __has_include("voxtral.h")
#include "voxtral.h"
#define CA_HAVE_VOXTRAL 1
#endif
#if __has_include("voxtral4b.h")
#include "voxtral4b.h"
#define CA_HAVE_VOXTRAL4B 1
#endif
#if __has_include("wav2vec2-ggml.h")
#include "wav2vec2-ggml.h"
#define CA_HAVE_WAV2VEC2 1
#endif
#if __has_include("vibevoice.h")
#include "vibevoice.h"
#define CA_HAVE_VIBEVOICE 1
#endif
#if __has_include("glm_asr.h")
#include "glm_asr.h"
#define CA_HAVE_GLMASR 1
#endif
#if __has_include("kyutai_stt.h")
#include "kyutai_stt.h"
#define CA_HAVE_KYUTAI 1
#endif
#if __has_include("firered_asr.h")
#include "firered_asr.h"
#define CA_HAVE_FIRERED 1
#endif
#if __has_include("moonshine.h")
#include "moonshine.h"
#define CA_HAVE_MOONSHINE 1
#endif
#if __has_include("omniasr.h")
#include "omniasr.h"
#define CA_HAVE_OMNIASR 1
#endif
#if __has_include("fireredpunc.h")
#include "fireredpunc.h"
#define CA_HAVE_FIREREDPUNC 1
#endif

#ifdef _WIN32
#define CA_EXPORT extern "C" __declspec(dllexport)
#else
#define CA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// =========================================================================
// whisper_full_params setters
// =========================================================================
//
// Dart holds the params as an opaque `Pointer<Void>` returned by
// `whisper_full_default_params_by_ref`. Rather than mirror the struct layout
// (~40 fields, volatile across upstream bumps), we expose a setter per field
// we actually care about.

CA_EXPORT void crispasr_params_set_language(whisper_full_params* p, const char* lang) {
    if (p)
        p->language = lang; // caller must keep the string alive
}
CA_EXPORT void crispasr_params_set_translate(whisper_full_params* p, int v) {
    if (p)
        p->translate = v != 0;
}
CA_EXPORT void crispasr_params_set_detect_language(whisper_full_params* p, int v) {
    if (p)
        p->detect_language = v != 0;
}
CA_EXPORT void crispasr_params_set_token_timestamps(whisper_full_params* p, int v) {
    if (p)
        p->token_timestamps = v != 0;
}
CA_EXPORT void crispasr_params_set_n_threads(whisper_full_params* p, int n) {
    if (p)
        p->n_threads = n;
}
CA_EXPORT void crispasr_params_set_max_len(whisper_full_params* p, int n) {
    if (p)
        p->max_len = n;
}
CA_EXPORT void crispasr_params_set_split_on_word(whisper_full_params* p, int v) {
    if (p)
        p->split_on_word = v != 0;
}
CA_EXPORT void crispasr_params_set_no_context(whisper_full_params* p, int v) {
    if (p)
        p->no_context = v != 0;
}
CA_EXPORT void crispasr_params_set_single_segment(whisper_full_params* p, int v) {
    if (p)
        p->single_segment = v != 0;
}
CA_EXPORT void crispasr_params_set_print_realtime(whisper_full_params* p, int v) {
    if (p)
        p->print_realtime = v != 0;
}
CA_EXPORT void crispasr_params_set_print_progress(whisper_full_params* p, int v) {
    if (p)
        p->print_progress = v != 0;
}
CA_EXPORT void crispasr_params_set_print_timestamps(whisper_full_params* p, int v) {
    if (p)
        p->print_timestamps = v != 0;
}
CA_EXPORT void crispasr_params_set_print_special(whisper_full_params* p, int v) {
    if (p)
        p->print_special = v != 0;
}
CA_EXPORT void crispasr_params_set_suppress_blank(whisper_full_params* p, int v) {
    if (p)
        p->suppress_blank = v != 0;
}
CA_EXPORT void crispasr_params_set_temperature(whisper_full_params* p, float t) {
    if (p)
        p->temperature = t;
}
CA_EXPORT void crispasr_params_set_initial_prompt(whisper_full_params* p, const char* prompt) {
    if (p)
        p->initial_prompt = prompt; // caller owns the string
}

// VAD (whisper.cpp built-in Silero pipeline). When enabled, whisper_full
// detects speech spans internally and only decodes those regions —
// timestamps are adjusted for the caller. Skips costly decode on silence.
CA_EXPORT void crispasr_params_set_vad(whisper_full_params* p, int v) {
    if (p)
        p->vad = v != 0;
}
CA_EXPORT void crispasr_params_set_vad_model_path(whisper_full_params* p, const char* path) {
    if (p)
        p->vad_model_path = path; // caller owns the string
}
CA_EXPORT void crispasr_params_set_vad_threshold(whisper_full_params* p, float t) {
    if (p)
        p->vad_params.threshold = t;
}
CA_EXPORT void crispasr_params_set_vad_min_speech_ms(whisper_full_params* p, int ms) {
    if (p)
        p->vad_params.min_speech_duration_ms = ms;
}
CA_EXPORT void crispasr_params_set_vad_min_silence_ms(whisper_full_params* p, int ms) {
    if (p)
        p->vad_params.min_silence_duration_ms = ms;
}

// tinydiarize (`tdrz`) — whisper's own experimental speaker-turn marker
// injection. Requires a whisper *.en.tdrz finetune. Emits `[SPEAKER_TURN]`
// tokens in-segment which the host can split on.
CA_EXPORT void crispasr_params_set_tdrz(whisper_full_params* p, int v) {
    if (p)
        p->tdrz_enable = v != 0;
}

// NOTE — DTW (Dynamic Time Warping) fields for precise per-token timing
// live on `whisper_context_params`, set at context init, not
// `whisper_full_params`. Exposing them needs a new
// `crispasr_init_with_dtw_params` entry point and wider binding work;
// tracked separately.

// =========================================================================
// Token-level timestamp getters
// =========================================================================
//
// `whisper_full_get_token_data` returns a `whisper_token_data` *by value*,
// which Dart FFI can't handle portably. Expose each field we need as a
// scalar-returning helper.

CA_EXPORT int64_t crispasr_token_t0(whisper_context* ctx, int i_seg, int i_tok) {
    if (!ctx)
        return 0;
    return whisper_full_get_token_data(ctx, i_seg, i_tok).t0;
}
CA_EXPORT int64_t crispasr_token_t1(whisper_context* ctx, int i_seg, int i_tok) {
    if (!ctx)
        return 0;
    return whisper_full_get_token_data(ctx, i_seg, i_tok).t1;
}
CA_EXPORT float crispasr_token_p(whisper_context* ctx, int i_seg, int i_tok) {
    if (!ctx)
        return 0.0f;
    return whisper_full_get_token_data(ctx, i_seg, i_tok).p;
}

// =========================================================================
// Language detection
// =========================================================================
//
// Mel + encode + `whisper_lang_auto_detect`. Writes the ISO-639 code into
// `out_code` (e.g. "de") and returns the detected-language probability.
// Returns negative on error.

CA_EXPORT float crispasr_detect_language(whisper_context* ctx, const float* pcm, int n_samples, int n_threads,
                                         char* out_code, int out_cap) {
    if (!ctx || !pcm || n_samples <= 0 || !out_code || out_cap <= 0) {
        return -1.0f;
    }

    // whisper requires mel + encode before lang auto-detect can run.
    if (whisper_pcm_to_mel(ctx, pcm, n_samples, n_threads > 0 ? n_threads : 4) != 0) {
        return -2.0f;
    }
    if (whisper_encode(ctx, 0, n_threads > 0 ? n_threads : 4) != 0) {
        return -3.0f;
    }

    std::vector<float> probs(whisper_lang_max_id() + 1, 0.0f);
    const int lang_id = whisper_lang_auto_detect(ctx, 0, n_threads > 0 ? n_threads : 4, probs.data());
    if (lang_id < 0)
        return -4.0f;

    const char* code = whisper_lang_str(lang_id);
    if (!code)
        return -5.0f;

    std::strncpy(out_code, code, out_cap - 1);
    out_code[out_cap - 1] = '\0';
    return probs[lang_id];
}

// =========================================================================
// VAD — run Silero on PCM, return [start_s, end_s] pairs
// =========================================================================
//
// `out_spans` is a malloc'd array of floats (2 per span). The caller must
// pass the pointer back to `crispasr_vad_free` when done. Returns the number
// of speech segments detected (>= 0), or a negative error.
//
//   -1  bad arguments
//   -2  model init failed
//   -3  VAD inference failed

CA_EXPORT int crispasr_vad_segments(const char* vad_model_path, const float* pcm, int n_samples, int sample_rate,
                                    float threshold, int min_speech_ms, int min_silence_ms, int n_threads, bool use_gpu,
                                    float** out_spans) {
    if (!vad_model_path || !pcm || n_samples <= 0 || !out_spans)
        return -1;
    *out_spans = nullptr;

    whisper_vad_context_params cparams = whisper_vad_default_context_params();
    cparams.n_threads = n_threads > 0 ? n_threads : 4;
    cparams.use_gpu = use_gpu;
    cparams.gpu_device = 0;

    whisper_vad_context* vctx = whisper_vad_init_from_file_with_params(vad_model_path, cparams);
    if (!vctx)
        return -2;

    whisper_vad_params vparams = whisper_vad_default_params();
    if (threshold > 0.0f)
        vparams.threshold = threshold;
    if (min_speech_ms > 0)
        vparams.min_speech_duration_ms = min_speech_ms;
    if (min_silence_ms > 0)
        vparams.min_silence_duration_ms = min_silence_ms;

    whisper_vad_segments* segs = whisper_vad_segments_from_samples(vctx, vparams, pcm, n_samples);
    if (!segs) {
        whisper_vad_free(vctx);
        return -3;
    }

    const int n = whisper_vad_segments_n_segments(segs);
    if (n > 0) {
        float* buf = (float*)std::malloc(sizeof(float) * 2 * n);
        if (!buf) {
            whisper_vad_free_segments(segs);
            whisper_vad_free(vctx);
            return -2;
        }
        for (int i = 0; i < n; ++i) {
            buf[2 * i + 0] = whisper_vad_segments_get_segment_t0(segs, i);
            buf[2 * i + 1] = whisper_vad_segments_get_segment_t1(segs, i);
        }
        *out_spans = buf;
    }

    whisper_vad_free_segments(segs);
    whisper_vad_free(vctx);
    // `sample_rate` parameter is accepted for API future-proofing even though
    // Silero VAD internally assumes 16 kHz — if we later add automatic
    // resampling here, callers don't have to change.
    (void)sample_rate;
    return n;
}

CA_EXPORT void crispasr_vad_free(float* spans) {
    if (spans)
        std::free(spans);
}

// =========================================================================
// Streaming transcription
// =========================================================================
//
// Port of `examples/stream/stream.cpp`'s rolling-window approach, but
// packaged as a pure-C-ABI struct so Dart can drive it without spinning
// its own threads. Non-blocking: caller feeds PCM in chunks of any size,
// and each feed whose accumulation crosses `step_ms` runs a single
// `whisper_full` on the last `length_ms` of audio (plus a small `keep_ms`
// context carry-over) and returns the concatenated text.
//
// This is the same "sliding-window" trick the CLI uses. It is not true
// token-level streaming — it is chunked batch with context carry, which
// is what whisper.cpp itself supports.

struct crispasr_stream {
    whisper_context* ctx = nullptr; // not owned
    int n_threads = 4;
    int step_ms = 3000;
    int length_ms = 10000;
    int keep_ms = 200;
    std::string language; // empty = auto
    bool translate = false;

    int n_samples_step = 0; // cached from step_ms
    int n_samples_length = 0;
    int n_samples_keep = 0;

    std::vector<float> accum;   // samples fed since last decode
    std::vector<float> history; // last decoded window (for carry)

    // Last decode output, held here until caller pulls it with
    // `crispasr_stream_get_text`.
    std::string out_text;
    double out_t0_s = 0.0;
    double out_t1_s = 0.0;
    bool has_output = false;

    // Monotonic counter so callers can detect when output has been replaced
    // by a subsequent decode even if the text didn't visibly change.
    int64_t decode_counter = 0;

    double stream_time_s = 0.0; // total audio fed, in seconds
};

CA_EXPORT crispasr_stream* crispasr_stream_open(whisper_context* ctx, int n_threads, int step_ms, int length_ms,
                                                int keep_ms, const char* language, int translate) {
    if (!ctx)
        return nullptr;
    auto* s = new crispasr_stream();
    s->ctx = ctx;
    s->n_threads = n_threads > 0 ? n_threads : 4;
    s->step_ms = step_ms > 0 ? step_ms : 3000;
    s->length_ms = length_ms > 0 ? length_ms : 10000;
    s->keep_ms = keep_ms >= 0 ? keep_ms : 200;
    s->translate = translate != 0;
    if (language && language[0] != '\0')
        s->language = language;

    constexpr int kSampleRate = 16000;
    s->n_samples_step = (int)(1e-3 * s->step_ms * kSampleRate);
    s->n_samples_length = (int)(1e-3 * s->length_ms * kSampleRate);
    s->n_samples_keep = (int)(1e-3 * s->keep_ms * kSampleRate);
    return s;
}

CA_EXPORT void crispasr_stream_close(crispasr_stream* s) {
    if (!s)
        return;
    delete s;
}

static int crispasr_stream_run_decode(crispasr_stream* s) {
    // Assemble the decode window: tail of `history` (length `n_samples_take`)
    // + all of `accum`.
    const int n_new = (int)s->accum.size();
    const int n_take = std::min((int)s->history.size(), std::max(0, s->n_samples_keep + s->n_samples_length - n_new));

    std::vector<float> pcm;
    pcm.reserve(n_take + n_new);
    if (n_take > 0) {
        const size_t start = s->history.size() - (size_t)n_take;
        pcm.insert(pcm.end(), s->history.begin() + start, s->history.end());
    }
    pcm.insert(pcm.end(), s->accum.begin(), s->accum.end());

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.print_special = false;
    wparams.single_segment = true; // mirror stream.cpp non-VAD path
    wparams.no_timestamps = false;
    wparams.translate = s->translate;
    wparams.n_threads = s->n_threads;
    wparams.language = s->language.empty() ? nullptr : s->language.c_str();
    wparams.detect_language = s->language.empty();
    wparams.no_context = true;

    if (whisper_full(s->ctx, wparams, pcm.data(), (int)pcm.size()) != 0) {
        return -1;
    }

    // Concatenate all segments produced by this decode.
    const int n_seg = whisper_full_n_segments(s->ctx);
    std::string text;
    double t0_s = 1e18;
    double t1_s = 0.0;
    for (int i = 0; i < n_seg; ++i) {
        const char* segtext = whisper_full_get_segment_text(s->ctx, i);
        if (segtext)
            text += segtext;

        const double t0 = whisper_full_get_segment_t0(s->ctx, i) / 100.0;
        const double t1 = whisper_full_get_segment_t1(s->ctx, i) / 100.0;
        if (t0 < t0_s)
            t0_s = t0;
        if (t1 > t1_s)
            t1_s = t1;
    }
    if (n_seg == 0) {
        t0_s = 0.0;
        t1_s = 0.0;
    }

    // Re-base timestamps onto absolute stream time: the last sample fed
    // sits at `stream_time_s`; the start of the decode window sits
    // `pcm.size() / 16000` seconds before that.
    const double win_end_abs = s->stream_time_s;
    const double win_start_abs = win_end_abs - (double)pcm.size() / 16000.0;
    s->out_text = std::move(text);
    s->out_t0_s = win_start_abs + t0_s;
    s->out_t1_s = win_start_abs + t1_s;
    s->has_output = true;
    s->decode_counter += 1;

    // Keep the last ~`length_ms + keep_ms` of audio as history so the next
    // decode can carry context. Anything older is dropped.
    s->history = pcm;
    const int max_hist = s->n_samples_length + s->n_samples_keep;
    if ((int)s->history.size() > max_hist) {
        s->history.erase(s->history.begin(), s->history.begin() + ((int)s->history.size() - max_hist));
    }
    s->accum.clear();
    return 0;
}

CA_EXPORT int crispasr_stream_feed(crispasr_stream* s, const float* pcm, int n_samples) {
    if (!s || !pcm || n_samples <= 0)
        return -1;
    s->accum.insert(s->accum.end(), pcm, pcm + n_samples);
    s->stream_time_s += (double)n_samples / 16000.0;

    if ((int)s->accum.size() < s->n_samples_step) {
        return 0; // still buffering
    }

    if (crispasr_stream_run_decode(s) != 0)
        return -2;
    return 1; // new output ready
}

CA_EXPORT int crispasr_stream_get_text(crispasr_stream* s, char* out_text, int out_cap, double* out_t0_s,
                                       double* out_t1_s, int64_t* out_counter) {
    if (!s || !out_text || out_cap <= 0)
        return -1;
    if (!s->has_output) {
        out_text[0] = '\0';
        if (out_t0_s)
            *out_t0_s = 0.0;
        if (out_t1_s)
            *out_t1_s = 0.0;
        if (out_counter)
            *out_counter = 0;
        return 0;
    }
    std::strncpy(out_text, s->out_text.c_str(), out_cap - 1);
    out_text[out_cap - 1] = '\0';
    if (out_t0_s)
        *out_t0_s = s->out_t0_s;
    if (out_t1_s)
        *out_t1_s = s->out_t1_s;
    if (out_counter)
        *out_counter = s->decode_counter;
    return (int)s->out_text.size();
}

/// Force a decode on whatever audio is currently buffered, regardless of
/// whether we hit the step threshold. Useful when the caller knows the
/// audio has ended and wants a final flush.
CA_EXPORT int crispasr_stream_flush(crispasr_stream* s) {
    if (!s)
        return -1;
    if (s->accum.empty())
        return 0;
    return crispasr_stream_run_decode(s) == 0 ? 1 : -2;
}

// =========================================================================
// Parakeet (nvidia/parakeet-tdt-0.6b-v3) — C-ABI wrappers for Dart
// =========================================================================
//
// Parakeet's C API already has clean C linkage (see parakeet.h), but Dart
// FFI can't deal with the returned `parakeet_result *` whose fields
// include `parakeet_token_data[]` and `parakeet_word_data[]` by value.
// These helpers wrap the handful of calls Dart needs: open / free,
// transcribe → opaque result handle, iterate words with scalar getters.

#ifdef CA_HAVE_PARAKEET

CA_EXPORT parakeet_context* crispasr_parakeet_init(const char* model_path, int n_threads, int use_flash) {
    if (!model_path)
        return nullptr;
    parakeet_context_params p = parakeet_context_default_params();
    p.n_threads = n_threads > 0 ? n_threads : 4;
    p.use_flash = use_flash != 0;
    p.verbosity = 0;
    return parakeet_init_from_file(model_path, p);
}

CA_EXPORT void crispasr_parakeet_free(parakeet_context* ctx) {
    if (ctx)
        parakeet_free(ctx);
}

CA_EXPORT parakeet_result* crispasr_parakeet_transcribe(parakeet_context* ctx, const float* pcm, int n_samples,
                                                        int64_t t_offset_cs) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;
    return parakeet_transcribe_ex(ctx, pcm, n_samples, t_offset_cs);
}

CA_EXPORT const char* crispasr_parakeet_result_text(parakeet_result* r) {
    return (r && r->text) ? r->text : "";
}

CA_EXPORT int crispasr_parakeet_result_n_words(parakeet_result* r) {
    return r ? r->n_words : 0;
}
CA_EXPORT const char* crispasr_parakeet_result_word_text(parakeet_result* r, int i) {
    if (!r || i < 0 || i >= r->n_words)
        return "";
    return r->words[i].text;
}
CA_EXPORT int64_t crispasr_parakeet_result_word_t0(parakeet_result* r, int i) {
    return (r && i >= 0 && i < r->n_words) ? r->words[i].t0 : 0;
}
CA_EXPORT int64_t crispasr_parakeet_result_word_t1(parakeet_result* r, int i) {
    return (r && i >= 0 && i < r->n_words) ? r->words[i].t1 : 0;
}

CA_EXPORT int crispasr_parakeet_result_n_tokens(parakeet_result* r) {
    return r ? r->n_tokens : 0;
}
CA_EXPORT const char* crispasr_parakeet_result_token_text(parakeet_result* r, int i) {
    if (!r || i < 0 || i >= r->n_tokens)
        return "";
    return r->tokens[i].text;
}
CA_EXPORT int64_t crispasr_parakeet_result_token_t0(parakeet_result* r, int i) {
    return (r && i >= 0 && i < r->n_tokens) ? r->tokens[i].t0 : 0;
}
CA_EXPORT int64_t crispasr_parakeet_result_token_t1(parakeet_result* r, int i) {
    return (r && i >= 0 && i < r->n_tokens) ? r->tokens[i].t1 : 0;
}
CA_EXPORT float crispasr_parakeet_result_token_p(parakeet_result* r, int i) {
    return (r && i >= 0 && i < r->n_tokens) ? r->tokens[i].p : 0.0f;
}

CA_EXPORT void crispasr_parakeet_result_free(parakeet_result* r) {
    if (r)
        parakeet_result_free(r);
}

#endif // CA_HAVE_PARAKEET

// =========================================================================
// Backend auto-detection from GGUF metadata
// =========================================================================
//
// Reads `general.architecture` from a GGUF file and returns one of the
// backend names used by CrispASR ("whisper" / "parakeet" / "canary" /
// "qwen3" / ...). Returns an empty string if the file is unreadable or
// the architecture is unknown.

#include "ggml.h"
#include "gguf.h"

CA_EXPORT int crispasr_detect_backend_from_gguf(const char* path, char* out_name, int out_cap) {
    if (!path || !out_name || out_cap <= 0)
        return -1;
    out_name[0] = '\0';

    gguf_init_params p = {/*no_alloc*/ true, /*ctx*/ nullptr};
    gguf_context* gctx = gguf_init_from_file(path, p);
    if (!gctx)
        return -2;

    const int key_id = gguf_find_key(gctx, "general.architecture");
    if (key_id < 0) {
        gguf_free(gctx);
        return -3;
    }
    const char* arch = gguf_get_val_str(gctx, key_id);
    if (!arch) {
        gguf_free(gctx);
        return -4;
    }

    // Map known architecture strings to CrispASR backend names.
    const char* backend = "";
    if (strcmp(arch, "whisper") == 0)
        backend = "whisper";
    else if (strcmp(arch, "parakeet") == 0 || strcmp(arch, "parakeet-tdt") == 0)
        backend = "parakeet";
    else if (strcmp(arch, "canary") == 0)
        backend = "canary";
    else if (strcmp(arch, "cohere-transcribe") == 0)
        backend = "cohere";
    else if (strcmp(arch, "qwen3-asr") == 0)
        backend = "qwen3";
    else if (strcmp(arch, "voxtral") == 0)
        backend = "voxtral";
    else if (strcmp(arch, "voxtral4b") == 0)
        backend = "voxtral4b";
    else if (strcmp(arch, "granite-speech") == 0)
        backend = "granite";
    else if (strcmp(arch, "fastconformer-ctc") == 0)
        backend = "fastconformer-ctc";
    else if (strcmp(arch, "canary-ctc") == 0)
        backend = "canary-ctc";
    else if (strcmp(arch, "wav2vec2") == 0)
        backend = "wav2vec2";
    else if (strcmp(arch, "vibevoice-asr") == 0 || strcmp(arch, "vibevoice") == 0)
        backend = "vibevoice";

    std::strncpy(out_name, backend, out_cap - 1);
    out_name[out_cap - 1] = '\0';
    gguf_free(gctx);
    return (int)std::strlen(out_name);
}

// =========================================================================
// Unified session API — one entry point for every backend
// =========================================================================
//
// Callers (Dart, Python, Rust) open a GGUF, we auto-detect the backend
// from its `general.architecture` metadata, construct the right native
// context internally, and expose a common segment/word/token surface.
// No caller code needs to know which backend a given model uses.
//
// Internally a `crispasr_session` owns exactly one of the per-backend
// contexts — we route every call to the matching per-backend wrapper.
// Adding a backend to the unified API is therefore the same three steps
// as adding it to the per-backend API, plus one more: a case in the big
// switch statement in `crispasr_session_open_explicit`.

struct crispasr_session {
    std::string backend; // "whisper", "parakeet", ...
    std::string model_path;
    int n_threads = 4;

    // Exactly one of these pointers is non-null based on `backend`.
    whisper_context* whisper_ctx = nullptr;
#ifdef CA_HAVE_PARAKEET
    parakeet_context* parakeet_ctx = nullptr;
#endif
#ifdef CA_HAVE_CANARY
    canary_context* canary_ctx = nullptr;
#endif
#ifdef CA_HAVE_QWEN3
    qwen3_asr_context* qwen3_ctx = nullptr;
#endif
#ifdef CA_HAVE_COHERE
    cohere_context* cohere_ctx = nullptr;
#endif
#ifdef CA_HAVE_GRANITE
    granite_speech_context* granite_ctx = nullptr;
#endif
#ifdef CA_HAVE_CTC
    // Shared between the fastconformer-ctc and canary-ctc backends — they
    // load different GGUFs but go through the same canary_ctc_* compute
    // pipeline.
    canary_ctc_context* ctc_ctx = nullptr;
#endif
#ifdef CA_HAVE_VOXTRAL
    voxtral_context* voxtral_ctx = nullptr;
#endif
#ifdef CA_HAVE_VOXTRAL4B
    voxtral4b_context* voxtral4b_ctx = nullptr;
#endif
#ifdef CA_HAVE_WAV2VEC2
    // wav2vec2_model is a C++ struct by-value; we heap-allocate it so
    // Dart can carry a pointer. `nullptr` means this slot is unused.
    wav2vec2_model* wav2vec2_ctx = nullptr;
#endif
#ifdef CA_HAVE_VIBEVOICE
    vibevoice_context* vibevoice_ctx = nullptr;
#endif
#ifdef CA_HAVE_GLMASR
    void* glmasr_ctx = nullptr;
#endif
#ifdef CA_HAVE_KYUTAI
    void* kyutai_ctx = nullptr;
#endif
#ifdef CA_HAVE_FIRERED
    void* firered_ctx = nullptr;
#endif
#ifdef CA_HAVE_MOONSHINE
    void* moonshine_ctx = nullptr;
#endif
#ifdef CA_HAVE_OMNIASR
    void* omniasr_ctx = nullptr;
#endif
};

struct crispasr_session_seg {
    std::string text;
    int64_t t0 = 0; // centiseconds absolute
    int64_t t1 = 0;
    struct word {
        std::string text;
        int64_t t0;
        int64_t t1;
        float p;
    };
    std::vector<word> words;
};

struct crispasr_session_result {
    std::vector<crispasr_session_seg> segments;
    std::string backend;
};

CA_EXPORT crispasr_session* crispasr_session_open_explicit(const char* model_path, const char* backend_name,
                                                           int n_threads) {
    if (!model_path || !backend_name)
        return nullptr;

    auto* s = new crispasr_session();
    s->model_path = model_path;
    s->backend = backend_name;
    s->n_threads = n_threads > 0 ? n_threads : 4;

    if (s->backend == "whisper") {
        whisper_context_params cparams = whisper_context_default_params();
        s->whisper_ctx = whisper_init_from_file_with_params(model_path, cparams);
        if (!s->whisper_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#ifdef CA_HAVE_PARAKEET
    if (s->backend == "parakeet") {
        parakeet_context_params pp = parakeet_context_default_params();
        pp.n_threads = s->n_threads;
        pp.verbosity = 0;
        s->parakeet_ctx = parakeet_init_from_file(model_path, pp);
        if (!s->parakeet_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_CANARY
    if (s->backend == "canary") {
        canary_context_params p = canary_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = 0;
        s->canary_ctx = canary_init_from_file(model_path, p);
        if (!s->canary_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_QWEN3
    if (s->backend == "qwen3") {
        qwen3_asr_context_params p = qwen3_asr_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = 0;
        s->qwen3_ctx = qwen3_asr_init_from_file(model_path, p);
        if (!s->qwen3_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_COHERE
    if (s->backend == "cohere") {
        cohere_context_params p = cohere_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = 0;
        s->cohere_ctx = cohere_init_from_file(model_path, p);
        if (!s->cohere_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_GRANITE
    if (s->backend == "granite") {
        granite_speech_context_params p = granite_speech_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = 0;
        s->granite_ctx = granite_speech_init_from_file(model_path, p);
        if (!s->granite_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_CTC
    if (s->backend == "fastconformer-ctc" || s->backend == "canary-ctc") {
        canary_ctc_context_params p = canary_ctc_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = 0;
        s->ctc_ctx = canary_ctc_init_from_file(model_path, p);
        if (!s->ctc_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_VOXTRAL
    if (s->backend == "voxtral") {
        voxtral_context_params p = voxtral_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = 0;
        s->voxtral_ctx = voxtral_init_from_file(model_path, p);
        if (!s->voxtral_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_VOXTRAL4B
    if (s->backend == "voxtral4b") {
        voxtral4b_context_params p = voxtral4b_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = 0;
        s->voxtral4b_ctx = voxtral4b_init_from_file(model_path, p);
        if (!s->voxtral4b_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_WAV2VEC2
    if (s->backend == "wav2vec2" || s->backend == "hubert" || s->backend == "data2vec") {
        s->wav2vec2_ctx = new wav2vec2_model();
        if (!wav2vec2_load(model_path, *s->wav2vec2_ctx)) {
            delete s->wav2vec2_ctx;
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_VIBEVOICE
    if (s->backend == "vibevoice") {
        vibevoice_context_params p = vibevoice_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = 0;
        p.use_gpu = false;
        s->vibevoice_ctx = vibevoice_init_from_file(model_path, p);
        if (!s->vibevoice_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_GLMASR
    if (s->backend == "glm-asr" || s->backend == "glmasr") {
        glm_asr_context_params p = glm_asr_context_default_params();
        p.n_threads = s->n_threads;
        s->glmasr_ctx = glm_asr_init_from_file(model_path, p);
        if (!s->glmasr_ctx) { delete s; return nullptr; }
        return s;
    }
#endif
#ifdef CA_HAVE_KYUTAI
    if (s->backend == "kyutai-stt" || s->backend == "kyutai" || s->backend == "moshi-stt") {
        kyutai_stt_context_params p = kyutai_stt_context_default_params();
        p.n_threads = s->n_threads;
        s->kyutai_ctx = kyutai_stt_init_from_file(model_path, p);
        if (!s->kyutai_ctx) { delete s; return nullptr; }
        return s;
    }
#endif
#ifdef CA_HAVE_FIRERED
    if (s->backend == "firered-asr" || s->backend == "firered") {
        firered_asr_context_params p = firered_asr_context_default_params();
        p.n_threads = s->n_threads;
        s->firered_ctx = firered_asr_init_from_file(model_path, p);
        if (!s->firered_ctx) { delete s; return nullptr; }
        return s;
    }
#endif
#ifdef CA_HAVE_MOONSHINE
    if (s->backend == "moonshine") {
        s->moonshine_ctx = moonshine_init(model_path);
        if (!s->moonshine_ctx) { delete s; return nullptr; }
        moonshine_set_n_threads((moonshine_context*)s->moonshine_ctx, s->n_threads);
        return s;
    }
#endif
#ifdef CA_HAVE_OMNIASR
    if (s->backend == "omniasr" || s->backend == "omniasr-ctc" || s->backend == "omniasr-llm") {
        omniasr_context_params p = omniasr_context_default_params();
        p.n_threads = s->n_threads;
        s->omniasr_ctx = omniasr_init_from_file(model_path, p);
        if (!s->omniasr_ctx) { delete s; return nullptr; }
        return s;
    }
#endif

    // Unknown or unsupported-in-this-build backend.
    delete s;
    return nullptr;
}

CA_EXPORT crispasr_session* crispasr_session_open(const char* model_path, int n_threads) {
    if (!model_path)
        return nullptr;
    char detected[64] = {0};
    if (crispasr_detect_backend_from_gguf(model_path, detected, (int)sizeof(detected)) <= 0) {
        // GGUF detection failed — check if this is a whisper GGML file
        // (magic "lmgg" or "ggjt"). Whisper models use the legacy GGML
        // format, not GGUF.
        FILE* f = fopen(model_path, "rb");
        if (f) {
            char magic[4] = {0};
            if (fread(magic, 1, 4, f) == 4 && (memcmp(magic, "lmgg", 4) == 0 || memcmp(magic, "ggjt", 4) == 0)) {
                snprintf(detected, sizeof(detected), "whisper");
            }
            fclose(f);
        }
        if (detected[0] == '\0')
            return nullptr;
    }
    return crispasr_session_open_explicit(model_path, detected, n_threads);
}

CA_EXPORT const char* crispasr_session_backend(crispasr_session* s) {
    return s ? s->backend.c_str() : "";
}

/// Comma-separated list of backend names compiled into this libwhisper.
/// e.g. "whisper,parakeet". Slim builds expose fewer. Used by language
/// bindings to show the user which formats are runtime-ready.
CA_EXPORT int crispasr_session_available_backends(char* out_csv, int out_cap) {
    if (!out_csv || out_cap <= 0)
        return -1;
    std::string list = "whisper";
#ifdef CA_HAVE_PARAKEET
    list += ",parakeet";
#endif
#ifdef CA_HAVE_CANARY
    list += ",canary";
#endif
#ifdef CA_HAVE_QWEN3
    list += ",qwen3";
#endif
#ifdef CA_HAVE_COHERE
    list += ",cohere";
#endif
#ifdef CA_HAVE_GRANITE
    list += ",granite";
#endif
#ifdef CA_HAVE_CTC
    list += ",fastconformer-ctc,canary-ctc";
#endif
#ifdef CA_HAVE_VOXTRAL
    list += ",voxtral";
#endif
#ifdef CA_HAVE_VOXTRAL4B
    list += ",voxtral4b";
#endif
#ifdef CA_HAVE_WAV2VEC2
    list += ",wav2vec2";
#endif
#ifdef CA_HAVE_VIBEVOICE
    list += ",vibevoice";
#endif
#ifdef CA_HAVE_GLMASR
    list += ",glm-asr";
#endif
#ifdef CA_HAVE_KYUTAI
    list += ",kyutai-stt";
#endif
#ifdef CA_HAVE_FIRERED
    list += ",firered-asr";
#endif
#ifdef CA_HAVE_MOONSHINE
    list += ",moonshine";
#endif
#ifdef CA_HAVE_OMNIASR
    list += ",omniasr";
#endif
    std::strncpy(out_csv, list.c_str(), out_cap - 1);
    out_csv[out_cap - 1] = '\0';
    return (int)list.size();
}

// Shared greedy generation loop for Voxtral-family audio-LLM backends.
// Each backend provides its own function pointers via the VoxtralOps trait
// struct below so we can share the code without pulling in the full
// CLI's crispasr_llm_pipeline.h (which depends on whisper_params and
// other CLI-only machinery).
//
// Prompt convention matches the Tekken template the CLI uses:
//   "<s>[INST][BEGIN_AUDIO]" + audio-pad×N_enc + "[/INST]lang:<LANG>[TRANSCRIBE]"
// The audio-pad slot embeddings are replaced in place with the encoder
// output so the LLM attends to the real audio features.
template <typename Ctx> struct VoxtralFamilyOps {
    // Function-pointer plumbing — populated via factory methods below so
    // we can template over either voxtral_* or voxtral4b_* without
    // macro-pasting.
    typedef float* (*ComputeMelFn)(Ctx*, const float*, int, int*, int*);
    typedef float* (*RunEncoderFn)(Ctx*, const float*, int, int, int*, int*);
    typedef int32_t* (*TokenizeFn)(Ctx*, const char*, int*);
    typedef float* (*EmbedTokensFn)(Ctx*, const int32_t*, int);
    typedef bool (*KvInitFn)(Ctx*, int);
    typedef void (*KvResetFn)(Ctx*);
    typedef float* (*RunLlmKvFn)(Ctx*, const float*, int, int, int*, int*);
    typedef const uint8_t* (*TokenTextFn)(Ctx*, int, int*);

    ComputeMelFn compute_mel = nullptr;
    RunEncoderFn run_encoder = nullptr;
    TokenizeFn tokenize = nullptr;
    EmbedTokensFn embed_tokens = nullptr;
    KvInitFn kv_init = nullptr;
    KvResetFn kv_reset = nullptr;
    RunLlmKvFn run_llm_kv = nullptr;
    TokenTextFn token_text = nullptr;

    int audio_pad_id = 24; // Tekken <audio_pad>
    int eos_id = 2;        // Tekken </s>
};

template <typename Ctx>
static crispasr_session_result* run_voxtral_family(Ctx* ctx, const VoxtralFamilyOps<Ctx>& ops, const float* pcm,
                                                   int n_samples, const std::string& language) {
    auto* r = new crispasr_session_result();
    r->segments.reserve(1);

    // 1. Mel spectrogram.
    int n_mels = 0, T_mel = 0;
    float* mel = ops.compute_mel(ctx, pcm, n_samples, &n_mels, &T_mel);
    if (!mel) {
        delete r;
        return nullptr;
    }

    // 2. Audio encoder.
    int N_enc = 0, enc_dim = 0;
    float* audio_embeds = ops.run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &enc_dim);
    std::free(mel);
    if (!audio_embeds) {
        delete r;
        return nullptr;
    }

    // 3. Tokenize prefix + build audio-pad run + tokenize suffix.
    const char* prefix = "<s>[INST][BEGIN_AUDIO]";
    const std::string suffix = std::string("[/INST]lang:") + (language.empty() ? "en" : language) + "[TRANSCRIBE]";

    int n_pref = 0;
    int32_t* pref_ids = ops.tokenize(ctx, prefix, &n_pref);
    int n_suf = 0;
    int32_t* suf_ids = ops.tokenize(ctx, suffix.c_str(), &n_suf);
    if (!pref_ids || !suf_ids) {
        if (pref_ids)
            std::free(pref_ids);
        if (suf_ids)
            std::free(suf_ids);
        std::free(audio_embeds);
        delete r;
        return nullptr;
    }

    // 4. Embed prefix.
    float* pref_embeds = ops.embed_tokens(ctx, pref_ids, n_pref);
    std::free(pref_ids);
    if (!pref_embeds) {
        std::free(suf_ids);
        std::free(audio_embeds);
        delete r;
        return nullptr;
    }

    // 5. Embed suffix.
    float* suf_embeds = ops.embed_tokens(ctx, suf_ids, n_suf);
    std::free(suf_ids);
    if (!suf_embeds) {
        std::free(pref_embeds);
        std::free(audio_embeds);
        delete r;
        return nullptr;
    }

    // 6. Splice [prefix][audio][suffix] into one embedding buffer, then
    //    prefill the KV cache with it in one shot.
    const int total_tokens = n_pref + N_enc + n_suf;
    std::vector<float> spliced((size_t)total_tokens * (size_t)enc_dim);
    std::memcpy(spliced.data(), pref_embeds, (size_t)n_pref * (size_t)enc_dim * sizeof(float));
    std::memcpy(spliced.data() + (size_t)n_pref * (size_t)enc_dim, audio_embeds,
                (size_t)N_enc * (size_t)enc_dim * sizeof(float));
    std::memcpy(spliced.data() + (size_t)(n_pref + N_enc) * (size_t)enc_dim, suf_embeds,
                (size_t)n_suf * (size_t)enc_dim * sizeof(float));
    std::free(pref_embeds);
    std::free(audio_embeds);
    std::free(suf_embeds);

    // 7. KV-cache prefill. Allow enough room for ~512 new tokens.
    constexpr int kMaxNewTokens = 512;
    if (!ops.kv_init(ctx, total_tokens + kMaxNewTokens + 16)) {
        delete r;
        return nullptr;
    }
    ops.kv_reset(ctx);

    int out_n_tok = 0, out_vocab = 0;
    float* logits = ops.run_llm_kv(ctx, spliced.data(), total_tokens, 0, &out_n_tok, &out_vocab);
    if (!logits || out_vocab <= 0) {
        delete r;
        return nullptr;
    }

    // 8. Greedy generation loop. Pick argmax at each step, embed the new
    //    token, feed it through run_llm_kv with n_past=total_tokens+step,
    //    stop on EOS or kMaxNewTokens.
    std::string generated;
    generated.reserve(512);
    int n_past = total_tokens;
    for (int step = 0; step < kMaxNewTokens; ++step) {
        // argmax over the last position's logits.
        const float* last = logits + (size_t)(out_n_tok - 1) * (size_t)out_vocab;
        int best = 0;
        float best_score = last[0];
        for (int i = 1; i < out_vocab; ++i) {
            if (last[i] > best_score) {
                best_score = last[i];
                best = i;
            }
        }
        std::free(logits);
        logits = nullptr;

        if (best == ops.eos_id)
            break;

        int tok_len = 0;
        const uint8_t* tok_bytes = ops.token_text(ctx, best, &tok_len);
        if (tok_bytes && tok_len > 0) {
            generated.append(reinterpret_cast<const char*>(tok_bytes), (size_t)tok_len);
        }

        // Embed the newly-chosen token and step the KV cache.
        int32_t next_id = best;
        float* next_emb = ops.embed_tokens(ctx, &next_id, 1);
        if (!next_emb)
            break;
        logits = ops.run_llm_kv(ctx, next_emb, 1, n_past, &out_n_tok, &out_vocab);
        std::free(next_emb);
        if (!logits)
            break;
        n_past += 1;
    }
    if (logits)
        std::free(logits);

    crispasr_session_seg seg;
    seg.text = std::move(generated);
    seg.t0 = 0;
    seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
    r->segments.push_back(std::move(seg));
    return r;
}

// ---------------------------------------------------------------------------
// Language-aware session transcribe. `language` is an ISO 639-1 code
// ("en", "de", "ja", ...). Passing NULL or empty keeps each backend's
// historical default (usually "en") so this is a strict superset of
// `crispasr_session_transcribe`. Backends that don't take a language
// input (parakeet, qwen3, granite, wav2vec2, fastconformer-ctc) ignore
// the hint silently — parakeet/qwen3 auto-detect, granite is instruction-
// tuned with its own prompt, wav2vec2 is usually mono-lingual.
// ---------------------------------------------------------------------------
CA_EXPORT crispasr_session_result* crispasr_session_transcribe_lang(crispasr_session* s, const float* pcm,
                                                                    int n_samples, const char* language) {
    if (!s || !pcm || n_samples <= 0)
        return nullptr;

    const std::string lang = (language && *language) ? language : "en";
    const bool lang_set = (language && *language);

    auto* r = new crispasr_session_result();
    r->backend = s->backend;

    if (s->backend == "whisper" && s->whisper_ctx) {
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_progress = false;
        wparams.print_realtime = false;
        wparams.print_timestamps = false;
        wparams.print_special = false;
        wparams.n_threads = s->n_threads;
        if (lang_set)
            wparams.language = lang.c_str();

        if (whisper_full(s->whisper_ctx, wparams, pcm, n_samples) != 0) {
            delete r;
            return nullptr;
        }
        const int n = whisper_full_n_segments(s->whisper_ctx);
        for (int i = 0; i < n; ++i) {
            crispasr_session_seg seg;
            const char* t = whisper_full_get_segment_text(s->whisper_ctx, i);
            if (t)
                seg.text = t;
            seg.t0 = whisper_full_get_segment_t0(s->whisper_ctx, i);
            seg.t1 = whisper_full_get_segment_t1(s->whisper_ctx, i);
            r->segments.push_back(std::move(seg));
        }
        return r;
    }
#ifdef CA_HAVE_PARAKEET
    if (s->backend == "parakeet" && s->parakeet_ctx) {
        parakeet_result* pr = parakeet_transcribe_ex(s->parakeet_ctx, pcm, n_samples, 0);
        if (!pr) {
            delete r;
            return nullptr;
        }

        // Parakeet produces one logical segment covering the whole input;
        // we package word-level timings into a single segment for the
        // unified shape.
        crispasr_session_seg seg;
        seg.text = pr->text ? pr->text : "";
        if (pr->n_words > 0) {
            seg.t0 = pr->words[0].t0;
            seg.t1 = pr->words[pr->n_words - 1].t1;
            seg.words.reserve(pr->n_words);
            for (int i = 0; i < pr->n_words; ++i) {
                crispasr_session_seg::word w;
                w.text = pr->words[i].text;
                w.t0 = pr->words[i].t0;
                w.t1 = pr->words[i].t1;
                w.p = 1.0f;
                seg.words.push_back(std::move(w));
            }
        }
        r->segments.push_back(std::move(seg));
        parakeet_result_free(pr);
        return r;
    }
#endif

    // Backends below all return a `char * malloc`'d transcript — we package
    // the whole thing into a single segment with no word timings. They're
    // LLM-style decoders (or ASR without native word-level alignment);
    // word timestamps would need CTC alignment as a post-step.
    auto run_char_transcribe = [&](char* raw) -> crispasr_session_result* {
        if (!raw) {
            delete r;
            return nullptr;
        }
        crispasr_session_seg seg;
        seg.text = raw;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
        r->segments.push_back(std::move(seg));
        std::free(raw);
        return r;
    };

#ifdef CA_HAVE_CANARY
    if (s->backend == "canary" && s->canary_ctx) {
        // Canary supports source/target language explicitly; when the
        // caller hasn't supplied a language the historical default of
        // en→en is preserved. Full translation control (src != tgt)
        // still needs the backend-specific helpers.
        return run_char_transcribe(canary_transcribe(s->canary_ctx, pcm, n_samples, lang.c_str(), lang.c_str(), true));
    }
#endif
#ifdef CA_HAVE_QWEN3
    if (s->backend == "qwen3" && s->qwen3_ctx) {
        return run_char_transcribe(qwen3_asr_transcribe(s->qwen3_ctx, pcm, n_samples));
    }
#endif
#ifdef CA_HAVE_COHERE
    if (s->backend == "cohere" && s->cohere_ctx) {
        return run_char_transcribe(cohere_transcribe(s->cohere_ctx, pcm, n_samples, lang.c_str()));
    }
#endif
#ifdef CA_HAVE_GRANITE
    if (s->backend == "granite" && s->granite_ctx) {
        return run_char_transcribe(granite_speech_transcribe(s->granite_ctx, pcm, n_samples));
    }
#endif
#ifdef CA_HAVE_VOXTRAL
    if (s->backend == "voxtral" && s->voxtral_ctx) {
        delete r; // run_voxtral_family creates its own
        VoxtralFamilyOps<voxtral_context> ops;
        ops.compute_mel = &voxtral_compute_mel;
        ops.run_encoder = &voxtral_run_encoder;
        ops.tokenize = &voxtral_tokenize;
        ops.embed_tokens = &voxtral_embed_tokens;
        ops.kv_init = &voxtral_kv_init;
        ops.kv_reset = &voxtral_kv_reset;
        ops.run_llm_kv = &voxtral_run_llm_kv;
        ops.token_text = &voxtral_token_text;
        ops.audio_pad_id = 24; // Tekken <audio_pad>
        ops.eos_id = 2;        // Tekken </s>
        return run_voxtral_family(s->voxtral_ctx, ops, pcm, n_samples, lang);
    }
#endif
#ifdef CA_HAVE_VOXTRAL4B
    if (s->backend == "voxtral4b" && s->voxtral4b_ctx) {
        delete r;
        VoxtralFamilyOps<voxtral4b_context> ops;
        ops.compute_mel = &voxtral4b_compute_mel;
        ops.run_encoder = &voxtral4b_run_encoder;
        ops.tokenize = &voxtral4b_tokenize;
        ops.embed_tokens = &voxtral4b_embed_tokens;
        ops.kv_init = &voxtral4b_kv_init;
        ops.kv_reset = &voxtral4b_kv_reset;
        ops.run_llm_kv = &voxtral4b_run_llm_kv;
        ops.token_text = &voxtral4b_token_text;
        ops.audio_pad_id = 24;
        ops.eos_id = 2;
        return run_voxtral_family(s->voxtral4b_ctx, ops, pcm, n_samples, lang);
    }
#endif
#ifdef CA_HAVE_WAV2VEC2
    if (s->backend == "wav2vec2" && s->wav2vec2_ctx) {
        // Encoder + CTC head → logits → greedy decode, same pipeline
        // wav2vec2-ggml.h advertises.
        auto logits = wav2vec2_compute_logits(*s->wav2vec2_ctx, pcm, n_samples, s->n_threads);
        if (logits.empty()) {
            delete r;
            return nullptr;
        }
        const int V = (int)s->wav2vec2_ctx->hparams.vocab_size;
        const int T = (int)(logits.size() / (size_t)V);
        const std::string text = wav2vec2_greedy_decode(*s->wav2vec2_ctx, logits.data(), T);

        crispasr_session_seg seg;
        seg.text = text;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
        r->segments.push_back(std::move(seg));
        return r;
    }
#endif
#ifdef CA_HAVE_VIBEVOICE
    if (s->backend == "vibevoice" && s->vibevoice_ctx) {
        auto resample_16k_to_24k = [](const float* in, int n_in) {
            std::vector<float> out;
            if (!in || n_in <= 0)
                return out;

            const int n_out = (int)((double)n_in * 24000.0 / 16000.0);
            out.resize((size_t)n_out);
            for (int i = 0; i < n_out; ++i) {
                const double pos = (double)i * 16000.0 / 24000.0;
                int i0 = (int)pos;
                int i1 = i0 + 1;
                if (i0 < 0)
                    i0 = 0;
                if (i1 >= n_in)
                    i1 = n_in - 1;
                const float frac = (float)(pos - (double)i0);
                out[(size_t)i] = in[i0] * (1.0f - frac) + in[i1] * frac;
            }
            return out;
        };

        const std::vector<float> pcm24 = resample_16k_to_24k(pcm, n_samples);
        return run_char_transcribe(vibevoice_transcribe(s->vibevoice_ctx, pcm24.data(), (int)pcm24.size()));
    }
#endif
#ifdef CA_HAVE_CTC
    if ((s->backend == "fastconformer-ctc" || s->backend == "canary-ctc") && s->ctc_ctx) {
        float* logits = nullptr;
        int T_enc = 0, V = 0;
        if (canary_ctc_compute_logits(s->ctc_ctx, pcm, n_samples, &logits, &T_enc, &V) != 0 || !logits) {
            delete r;
            return nullptr;
        }
        char* text = canary_ctc_greedy_decode(s->ctc_ctx, logits, T_enc, V);
        std::free(logits);
        if (!text) {
            delete r;
            return nullptr;
        }
        crispasr_session_seg seg;
        seg.text = text;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
        r->segments.push_back(std::move(seg));
        std::free(text);
        return r;
    }
#endif

    // Generic text-returning backends: glm-asr, kyutai-stt, firered-asr,
    // moonshine, omniasr — all return a malloc'd/static string from transcribe().
    {
        char* text = nullptr;
        bool need_free = true;
#ifdef CA_HAVE_GLMASR
        if ((s->backend == "glm-asr" || s->backend == "glmasr") && s->glmasr_ctx)
            text = glm_asr_transcribe((glm_asr_context*)s->glmasr_ctx, pcm, n_samples);
#endif
#ifdef CA_HAVE_KYUTAI
        if (!text && (s->backend == "kyutai-stt" || s->backend == "kyutai" || s->backend == "moshi-stt") && s->kyutai_ctx)
            text = kyutai_stt_transcribe((kyutai_stt_context*)s->kyutai_ctx, pcm, n_samples);
#endif
#ifdef CA_HAVE_FIRERED
        if (!text && (s->backend == "firered-asr" || s->backend == "firered") && s->firered_ctx)
            text = firered_asr_transcribe((firered_asr_context*)s->firered_ctx, pcm, n_samples);
#endif
#ifdef CA_HAVE_MOONSHINE
        if (!text && s->backend == "moonshine" && s->moonshine_ctx) {
            text = (char*)moonshine_transcribe((moonshine_context*)s->moonshine_ctx, pcm, n_samples);
            need_free = false; // moonshine returns internal pointer
        }
#endif
#ifdef CA_HAVE_OMNIASR
        if (!text && (s->backend == "omniasr" || s->backend == "omniasr-ctc" || s->backend == "omniasr-llm") && s->omniasr_ctx)
            text = omniasr_transcribe((omniasr_context*)s->omniasr_ctx, pcm, n_samples);
#endif
        if (text) {
            crispasr_session_seg seg;
            seg.text = text;
            seg.t0 = 0;
            seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
            r->segments.push_back(std::move(seg));
            if (need_free) std::free(text);
            return r;
        }
    }

    delete r;
    return nullptr;
}

// Back-compat wrapper. Existing 0.4.x consumers called the 3-arg shape;
// now that's a thin forward to `_lang` with a null language hint, which
// reproduces the historical per-backend defaults (usually "en").
CA_EXPORT crispasr_session_result* crispasr_session_transcribe(crispasr_session* s, const float* pcm, int n_samples) {
    return crispasr_session_transcribe_lang(s, pcm, n_samples, nullptr);
}

// ---------------------------------------------------------------------------
// VAD-driven transcription over the session API.
//
// Runs Silero VAD on the PCM buffer, merges short / overlong slices into
// usable chunks, stitches them into a single contiguous buffer with 0.1s
// silence gaps (whisper.cpp-style), calls crispasr_session_transcribe on
// the stitched buffer, and remaps segment + word timestamps from
// stitched-buffer space back to original-audio positions.
//
// The same algorithm the CLI uses (see examples/cli/crispasr_run.cpp) is
// now reachable from every binding via a single call.
//
// Falls back to a direct crispasr_session_transcribe(pcm) when VAD
// produces no slices (no speech / model load failure). Callers should
// pass sample_rate = 16000 for all currently-supported backends.
// ---------------------------------------------------------------------------
struct crispasr_vad_abi_opts {
    float threshold;                 // 0.5 typical
    int32_t min_speech_duration_ms;  // 250
    int32_t min_silence_duration_ms; // 100
    int32_t speech_pad_ms;           // 30
    int32_t chunk_seconds;           // 30 (0 = no max-split)
    int32_t n_threads;               // 4
};

// 0.4.9+: language-aware VAD transcribe. Passing a non-empty ISO 639-1
// code forwards it into whichever backend accepts one (whisper / canary /
// cohere / voxtral / voxtral4b). NULL or empty keeps each backend's
// historical default so this function is a strict superset of
// `crispasr_session_transcribe_vad`.
CA_EXPORT crispasr_session_result* crispasr_session_transcribe_vad_lang(crispasr_session* s, const float* pcm,
                                                                        int n_samples, int sample_rate,
                                                                        const char* vad_model_path,
                                                                        const crispasr_vad_abi_opts* opts_or_null,
                                                                        const char* language) {
    if (!s || !pcm || n_samples <= 0 || sample_rate <= 0)
        return nullptr;

    // Fill a library opts struct from the ABI struct, or use defaults.
    crispasr_vad_options opts;
    if (opts_or_null) {
        opts.threshold = opts_or_null->threshold;
        opts.min_speech_duration_ms = opts_or_null->min_speech_duration_ms;
        opts.min_silence_duration_ms = opts_or_null->min_silence_duration_ms;
        opts.speech_pad_ms = opts_or_null->speech_pad_ms;
        opts.chunk_seconds = opts_or_null->chunk_seconds;
        if (opts_or_null->n_threads > 0)
            opts.n_threads = opts_or_null->n_threads;
    }

    // Compute speech slices. Empty slices ⇒ VAD model missing or no speech
    // detected — fall back to a plain transcribe so callers always get some
    // result when audio exists.
    std::vector<crispasr_audio_slice> slices;
    if (vad_model_path && *vad_model_path) {
        slices = crispasr_compute_vad_slices(pcm, n_samples, sample_rate, vad_model_path, opts);
    }
    if (slices.empty()) {
        return crispasr_session_transcribe_lang(s, pcm, n_samples, language);
    }

    // One slice ⇒ no stitching needed, but still clip to the speech region
    // so the backend doesn't burn cycles on leading / trailing silence.
    if (slices.size() == 1) {
        const auto& sl = slices.front();
        return crispasr_session_transcribe_lang(s, pcm + sl.start, sl.end - sl.start, language);
    }

    // Multiple slices ⇒ stitch with 0.1s silence gaps, transcribe once,
    // remap timestamps back to original-audio positions.
    auto stitched = crispasr_stitch_vad_slices(pcm, n_samples, sample_rate, slices);
    crispasr_session_result* r =
        crispasr_session_transcribe_lang(s, stitched.samples.data(), (int)stitched.samples.size(), language);
    if (!r)
        return nullptr;

    for (auto& seg : r->segments) {
        seg.t0 = crispasr_vad_remap_timestamp(stitched.mapping, seg.t0);
        seg.t1 = crispasr_vad_remap_timestamp(stitched.mapping, seg.t1);
        for (auto& w : seg.words) {
            w.t0 = crispasr_vad_remap_timestamp(stitched.mapping, w.t0);
            w.t1 = crispasr_vad_remap_timestamp(stitched.mapping, w.t1);
        }
    }
    return r;
}

// Back-compat wrapper for 0.4.4–0.4.8 consumers. Forwards to the
// language-aware variant with `language = NULL` (historical defaults).
CA_EXPORT crispasr_session_result* crispasr_session_transcribe_vad(crispasr_session* s, const float* pcm, int n_samples,
                                                                   int sample_rate, const char* vad_model_path,
                                                                   const crispasr_vad_abi_opts* opts_or_null) {
    return crispasr_session_transcribe_vad_lang(s, pcm, n_samples, sample_rate, vad_model_path, opts_or_null, nullptr);
}

// ---------------------------------------------------------------------------
// Speaker diarization (shared across all 4 consumers).
//
// Operates on a PCM buffer + a caller-supplied array of segment timings,
// writes a zero-based speaker index into each segment. Four methods:
//   0 Energy    — stereo only, |L| vs |R| per segment
//   1 Xcorr     — stereo only, TDOA via cross-correlation
//   2 VadTurns  — mono-friendly, alternates every >600 ms gap
//   3 Pyannote  — mono-friendly, ML via GGUF pyannote seg model
//
// `right_pcm` may be null when `is_stereo == 0`. `opts->pyannote_model_path`
// must point at a concrete GGUF for the Pyannote method; other methods
// ignore it.
//
// Returns 0 on success, 1 when Pyannote was requested but the model
// failed to load, -1 on invalid arguments. `speaker = -1` in a seg
// means the method had no information to pick a label for that segment.
// ---------------------------------------------------------------------------
struct crispasr_diarize_seg_abi {
    int64_t t0_cs;
    int64_t t1_cs;
    int32_t speaker; // out: -1 if unassigned
    int32_t _pad;
};

struct crispasr_diarize_opts_abi {
    int32_t method; // 0..3 from crispasr_diarize_method_t
    int32_t n_threads;
    int64_t slice_t0_cs;
    const char* pyannote_model_path; // required for method 3, ignored otherwise
};

CA_EXPORT int crispasr_diarize_segments_abi(const float* left_pcm, const float* right_pcm, int32_t n_samples,
                                            int32_t is_stereo, crispasr_diarize_seg_abi* segs, int32_t n_segs,
                                            const crispasr_diarize_opts_abi* opts) {
    if (!left_pcm || !segs || n_segs <= 0 || !opts)
        return -1;
    if (opts->method < 0 || opts->method > 3)
        return -1;

    CrispasrDiarizeOptions lib_opts;
    lib_opts.method = static_cast<CrispasrDiarizeMethod>(opts->method);
    lib_opts.n_threads = opts->n_threads > 0 ? opts->n_threads : 4;
    lib_opts.slice_t0_cs = opts->slice_t0_cs;
    if (opts->pyannote_model_path)
        lib_opts.pyannote_model_path = opts->pyannote_model_path;

    std::vector<CrispasrDiarizeSegment> lib_segs;
    lib_segs.reserve(n_segs);
    for (int i = 0; i < n_segs; i++)
        lib_segs.push_back({segs[i].t0_cs, segs[i].t1_cs, segs[i].speaker});

    const float* r = (is_stereo && right_pcm) ? right_pcm : left_pcm;
    const bool ok = crispasr_diarize_segments(left_pcm, r, n_samples, is_stereo != 0, lib_segs, lib_opts);
    if (!ok)
        return 1;

    for (int i = 0; i < n_segs; i++) {
        segs[i].speaker = lib_segs[i].speaker;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Language identification (shared across all 4 consumers).
//
// Runs LID on a 16 kHz mono float PCM buffer. Two methods:
//   0 Whisper — encoder + lang head on a multilingual ggml-*.bin
//   1 Silero  — GGUF-packed Silero 95-language classifier
//
// `model_path` must point to a concrete file on disk (callers handle
// auto-download themselves — the CLI has a shim for that; wrappers can
// ship the model as an asset).
//
// Returns 0 on success. `out_lang_buf` is populated with a null-terminated
// ISO-639-1 code (e.g. "en", "de"). `out_confidence` gets the posterior
// probability ([0, 1]) on whisper or silero's softmax peak.
//
// Error codes: -1 = invalid args, 1 = model load / detect failure, 2 =
// output buffer too small.
// ---------------------------------------------------------------------------
CA_EXPORT int crispasr_detect_language_pcm(const float* samples, int32_t n_samples,
                                           int32_t method,         // 0 = whisper, 1 = silero
                                           const char* model_path, // concrete path (required)
                                           int32_t n_threads,
                                           int32_t use_gpu, // 0 / 1
                                           int32_t gpu_device,
                                           int32_t flash_attn, // 0 / 1
                                           char* out_lang_buf, int32_t out_lang_cap, float* out_confidence) {
    if (!samples || n_samples <= 0 || !model_path || !out_lang_buf || out_lang_cap <= 0)
        return -1;
    if (method < 0 || method > 3)
        return -1;

    CrispasrLidOptions opts;
    opts.method = static_cast<CrispasrLidMethod>(method);
    opts.model_path = model_path;
    opts.n_threads = n_threads > 0 ? n_threads : 4;
    opts.use_gpu = use_gpu != 0;
    opts.gpu_device = gpu_device;
    opts.flash_attn = flash_attn != 0;
    opts.verbose = false;

    CrispasrLidResult r;
    if (!crispasr_detect_language(samples, n_samples, opts, r))
        return 1;

    if ((int)r.lang_code.size() + 1 > out_lang_cap)
        return 2;
    std::memcpy(out_lang_buf, r.lang_code.c_str(), r.lang_code.size());
    out_lang_buf[r.lang_code.size()] = '\0';
    if (out_confidence)
        *out_confidence = r.confidence;
    return 0;
}

// ---------------------------------------------------------------------------
// CTC / forced-aligner word timings (shared across all 4 consumers).
//
// Runs a CTC aligner (canary-ctc by default, qwen3-forced-aligner when
// the filename matches) on a transcript + audio pair and emits one
// per-word entry with centisecond timings. Useful for LLM-based
// backends (qwen3, voxtral, voxtral4b, granite) that don't produce
// per-word timestamps on their own.
//
// Because each aligned word carries a dynamically-sized UTF-8 text
// string, the result is returned as an opaque handle that the caller
// frees with `crispasr_align_result_free`. Accessors below mirror the
// session-result accessor pattern.
// ---------------------------------------------------------------------------
struct crispasr_align_result {
    std::vector<CrispasrAlignedWord> words;
};

CA_EXPORT crispasr_align_result* crispasr_align_words_abi(const char* aligner_model, const char* transcript,
                                                          const float* samples, int32_t n_samples, int64_t t_offset_cs,
                                                          int32_t n_threads) {
    if (!aligner_model || !transcript || !samples || n_samples <= 0)
        return nullptr;
    auto* r = new crispasr_align_result();
    r->words =
        crispasr_align_words(aligner_model, transcript, samples, n_samples, t_offset_cs, n_threads > 0 ? n_threads : 4);
    if (r->words.empty()) {
        delete r;
        return nullptr;
    }
    return r;
}

CA_EXPORT int crispasr_align_result_n_words(crispasr_align_result* r) {
    return r ? (int)r->words.size() : 0;
}

CA_EXPORT const char* crispasr_align_result_word_text(crispasr_align_result* r, int i) {
    return (r && i >= 0 && i < (int)r->words.size()) ? r->words[i].text.c_str() : "";
}

CA_EXPORT int64_t crispasr_align_result_word_t0(crispasr_align_result* r, int i) {
    return (r && i >= 0 && i < (int)r->words.size()) ? r->words[i].t0_cs : 0;
}

CA_EXPORT int64_t crispasr_align_result_word_t1(crispasr_align_result* r, int i) {
    return (r && i >= 0 && i < (int)r->words.size()) ? r->words[i].t1_cs : 0;
}

CA_EXPORT void crispasr_align_result_free(crispasr_align_result* r) {
    if (r)
        delete r;
}

// ---------------------------------------------------------------------------
// HF download + filesystem cache (shared across all 4 consumers).
//
// Writes the resolved path into `out_buf` (null-terminated) and returns 0
// on success. Returns -1 on invalid args, 1 on download failure, 2 when
// the output buffer is too small to hold the resolved path.
//
// `cache_dir_override` may be nullptr / empty to use the platform default
// (~/.cache/crispasr on POSIX, %USERPROFILE%/.cache/crispasr on Windows).
// ---------------------------------------------------------------------------
CA_EXPORT int crispasr_cache_ensure_file_abi(const char* filename, const char* url, int32_t quiet,
                                             const char* cache_dir_override, char* out_buf, int32_t out_cap) {
    if (!filename || !url || !out_buf || out_cap <= 0)
        return -1;
    const std::string override_s = cache_dir_override ? cache_dir_override : "";
    const std::string path = crispasr_cache::ensure_cached_file(filename, url, quiet != 0, "crispasr", override_s);
    if (path.empty())
        return 1;
    if ((int)path.size() + 1 > out_cap)
        return 2;
    std::memcpy(out_buf, path.c_str(), path.size());
    out_buf[path.size()] = '\0';
    return 0;
}

// Write the resolved cache dir (creating it if missing) into `out_buf`.
// Same return convention as above.
CA_EXPORT int crispasr_cache_dir_abi(const char* cache_dir_override, char* out_buf, int32_t out_cap) {
    if (!out_buf || out_cap <= 0)
        return -1;
    const std::string override_s = cache_dir_override ? cache_dir_override : "";
    const std::string d = crispasr_cache::dir(override_s);
    if (d.empty())
        return 1;
    if ((int)d.size() + 1 > out_cap)
        return 2;
    std::memcpy(out_buf, d.c_str(), d.size());
    out_buf[d.size()] = '\0';
    return 0;
}

// ---------------------------------------------------------------------------
// Known-model registry lookup.
//
// Writes the canonical filename, HF URL, and human-readable approx size
// into caller-provided buffers. Returns 0 on hit, 1 on miss, -1 on
// invalid args, 2 when any of the output buffers is too small.
// ---------------------------------------------------------------------------
static int write_entry(const CrispasrRegistryEntry& e, char* out_filename, int32_t filename_cap, char* out_url,
                       int32_t url_cap, char* out_size, int32_t size_cap) {
    if ((int)e.filename.size() + 1 > filename_cap || (int)e.url.size() + 1 > url_cap ||
        (int)e.approx_size.size() + 1 > size_cap)
        return 2;
    std::memcpy(out_filename, e.filename.c_str(), e.filename.size());
    out_filename[e.filename.size()] = '\0';
    std::memcpy(out_url, e.url.c_str(), e.url.size());
    out_url[e.url.size()] = '\0';
    std::memcpy(out_size, e.approx_size.c_str(), e.approx_size.size());
    out_size[e.approx_size.size()] = '\0';
    return 0;
}

CA_EXPORT int crispasr_registry_lookup_abi(const char* backend, char* out_filename, int32_t filename_cap, char* out_url,
                                           int32_t url_cap, char* out_size, int32_t size_cap) {
    if (!backend || !out_filename || !out_url || !out_size || filename_cap <= 0 || url_cap <= 0 || size_cap <= 0)
        return -1;
    CrispasrRegistryEntry e;
    if (!crispasr_registry_lookup(backend, e))
        return 1;
    return write_entry(e, out_filename, filename_cap, out_url, url_cap, out_size, size_cap);
}

CA_EXPORT int crispasr_registry_lookup_by_filename_abi(const char* filename, char* out_filename, int32_t filename_cap,
                                                       char* out_url, int32_t url_cap, char* out_size,
                                                       int32_t size_cap) {
    if (!filename || !out_filename || !out_url || !out_size || filename_cap <= 0 || url_cap <= 0 || size_cap <= 0)
        return -1;
    CrispasrRegistryEntry e;
    if (!crispasr_registry_lookup_by_filename(filename, e))
        return 1;
    return write_entry(e, out_filename, filename_cap, out_url, url_cap, out_size, size_cap);
}

CA_EXPORT int crispasr_session_result_n_segments(crispasr_session_result* r) {
    return r ? (int)r->segments.size() : 0;
}
CA_EXPORT const char* crispasr_session_result_segment_text(crispasr_session_result* r, int i) {
    return (r && i >= 0 && i < (int)r->segments.size()) ? r->segments[i].text.c_str() : "";
}
CA_EXPORT int64_t crispasr_session_result_segment_t0(crispasr_session_result* r, int i) {
    return (r && i >= 0 && i < (int)r->segments.size()) ? r->segments[i].t0 : 0;
}
CA_EXPORT int64_t crispasr_session_result_segment_t1(crispasr_session_result* r, int i) {
    return (r && i >= 0 && i < (int)r->segments.size()) ? r->segments[i].t1 : 0;
}
CA_EXPORT int crispasr_session_result_n_words(crispasr_session_result* r, int i_seg) {
    if (!r || i_seg < 0 || i_seg >= (int)r->segments.size())
        return 0;
    return (int)r->segments[i_seg].words.size();
}
CA_EXPORT const char* crispasr_session_result_word_text(crispasr_session_result* r, int i_seg, int i_word) {
    if (!r || i_seg < 0 || i_seg >= (int)r->segments.size())
        return "";
    auto& ws = r->segments[i_seg].words;
    return (i_word >= 0 && i_word < (int)ws.size()) ? ws[i_word].text.c_str() : "";
}
CA_EXPORT int64_t crispasr_session_result_word_t0(crispasr_session_result* r, int i_seg, int i_word) {
    if (!r || i_seg < 0 || i_seg >= (int)r->segments.size())
        return 0;
    auto& ws = r->segments[i_seg].words;
    return (i_word >= 0 && i_word < (int)ws.size()) ? ws[i_word].t0 : 0;
}
CA_EXPORT int64_t crispasr_session_result_word_t1(crispasr_session_result* r, int i_seg, int i_word) {
    if (!r || i_seg < 0 || i_seg >= (int)r->segments.size())
        return 0;
    auto& ws = r->segments[i_seg].words;
    return (i_word >= 0 && i_word < (int)ws.size()) ? ws[i_word].t1 : 0;
}

CA_EXPORT void crispasr_session_result_free(crispasr_session_result* r) {
    if (r)
        delete r;
}

CA_EXPORT void crispasr_session_close(crispasr_session* s) {
    if (!s)
        return;
    if (s->whisper_ctx)
        whisper_free(s->whisper_ctx);
#ifdef CA_HAVE_PARAKEET
    if (s->parakeet_ctx)
        parakeet_free(s->parakeet_ctx);
#endif
#ifdef CA_HAVE_CANARY
    if (s->canary_ctx)
        canary_free(s->canary_ctx);
#endif
#ifdef CA_HAVE_QWEN3
    if (s->qwen3_ctx)
        qwen3_asr_free(s->qwen3_ctx);
#endif
#ifdef CA_HAVE_COHERE
    if (s->cohere_ctx)
        cohere_free(s->cohere_ctx);
#endif
#ifdef CA_HAVE_GRANITE
    if (s->granite_ctx)
        granite_speech_free(s->granite_ctx);
#endif
#ifdef CA_HAVE_CTC
    if (s->ctc_ctx)
        canary_ctc_free(s->ctc_ctx);
#endif
#ifdef CA_HAVE_VOXTRAL
    if (s->voxtral_ctx)
        voxtral_free(s->voxtral_ctx);
#endif
#ifdef CA_HAVE_VOXTRAL4B
    if (s->voxtral4b_ctx)
        voxtral4b_free(s->voxtral4b_ctx);
#endif
#ifdef CA_HAVE_WAV2VEC2
    if (s->wav2vec2_ctx) {
        delete s->wav2vec2_ctx;
    }
#endif
#ifdef CA_HAVE_VIBEVOICE
    if (s->vibevoice_ctx)
        vibevoice_free(s->vibevoice_ctx);
#endif
#ifdef CA_HAVE_GLMASR
    if (s->glmasr_ctx)
        glm_asr_free((glm_asr_context*)s->glmasr_ctx);
#endif
#ifdef CA_HAVE_KYUTAI
    if (s->kyutai_ctx)
        kyutai_stt_free((kyutai_stt_context*)s->kyutai_ctx);
#endif
#ifdef CA_HAVE_FIRERED
    if (s->firered_ctx)
        firered_asr_free((firered_asr_context*)s->firered_ctx);
#endif
#ifdef CA_HAVE_MOONSHINE
    if (s->moonshine_ctx)
        moonshine_free((moonshine_context*)s->moonshine_ctx);
#endif
#ifdef CA_HAVE_OMNIASR
    if (s->omniasr_ctx)
        omniasr_free((omniasr_context*)s->omniasr_ctx);
#endif
    delete s;
}

// =========================================================================
// FireRedPunc — punctuation restoration post-processor
// =========================================================================
// These are standalone entry points (not part of the session API) so any
// consumer can load a punc model once and call it on arbitrary text.

#ifdef CA_HAVE_FIREREDPUNC
CA_EXPORT void* crispasr_punc_init(const char* model_path) {
    return (void*)fireredpunc_init(model_path);
}

CA_EXPORT const char* crispasr_punc_process(void* ctx, const char* text) {
    return fireredpunc_process((fireredpunc_context*)ctx, text);
}

CA_EXPORT void crispasr_punc_free_text(const char* text) {
    free((void*)text);
}

CA_EXPORT void crispasr_punc_free(void* ctx) {
    fireredpunc_free((fireredpunc_context*)ctx);
}
#else
CA_EXPORT void* crispasr_punc_init(const char*) { return nullptr; }
CA_EXPORT const char* crispasr_punc_process(void*, const char*) { return nullptr; }
CA_EXPORT void crispasr_punc_free_text(const char*) {}
CA_EXPORT void crispasr_punc_free(void*) {}
#endif

// =========================================================================
// Version reporting — identifies the C-ABI build to every consumer
// (CLI, Dart, Python, Rust). Bump when breaking or extending the surface.
// =========================================================================

CA_EXPORT const char* crispasr_c_api_version(void) {
    return "0.5.0";
}

// Backwards-compatibility alias. The Dart smoke test and any 0.4.x-era
// consumer probed `crispasr_dart_helpers_version`. The symbol was renamed
// when the file moved to `crispasr_c_api.cpp` (no longer Dart-specific).
// TODO: remove once all in-tree consumers are updated and a major-version
// bump is cut.
CA_EXPORT const char* crispasr_dart_helpers_version(void) {
    return crispasr_c_api_version();
}
