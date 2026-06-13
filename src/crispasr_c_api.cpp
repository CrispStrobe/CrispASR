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

#include "crispasr_session.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>

#include "crispasr.h"
#include "crispasr_vad.h"     // VAD slicing + stitching (shared with CLI)
#include "crispasr_diarize.h" // Speaker diarization (shared with CLI)
#include "crispasr_lid.h"     // Language identification (shared with CLI)
#if defined(CRISPASR_RNNOISE)
#include "crispasr_enhance.h" // RNNoise audio enhancement (shared with CLI)
#endif
#include "text_lid_dispatch.h"       // Text-LID backend-agnostic façade (CLD3 + fastText)
#include "crispasr_aligner.h"        // CTC / forced-aligner word timings (shared with CLI)
#include "crispasr_cache.h"          // HF download + filesystem cache (shared with CLI)
#include "crispasr_model_registry.h" // Known-model lookup (shared with CLI)
#include "crispasr_punc_model.h"     // shared --punc-model alias resolution (CLI/server/C-ABI parity)
#include "core/beam_decode.h"        // Shared autoregressive beam-search decode helper
#include "core/greedy_decode.h"      // Shared autoregressive greedy decode helper
#include "grammar-parser.h"          // GBNF parser for grammar-constrained sampling
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
#if __has_include("lfm2_audio.h")
#include "lfm2_audio.h"
#define CA_HAVE_LFM2_AUDIO 1
#endif
#if __has_include("mini_omni2.h")
#include "mini_omni2.h"
#define CA_HAVE_MINI_OMNI2 1
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
#if __has_include("granite_nle.h")
#include "granite_nle.h"
#define CA_HAVE_GRANITE_NLE 1
#endif
#if __has_include("funasr.h")
#include "funasr.h"
#define CA_HAVE_FUNASR 1
#endif
#if __has_include("paraformer.h")
#include "paraformer.h"
#define CA_HAVE_PARAFORMER 1
#endif
#if __has_include("sensevoice.h")
#include "sensevoice.h"
#define CA_HAVE_SENSEVOICE 1
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
#if __has_include("kugelaudio.h")
#include "kugelaudio.h"
#define CA_HAVE_KUGELAUDIO 1
#endif
#if __has_include("zonos_tts.h")
#include "zonos_tts.h"
#define CA_HAVE_ZONOS 1
#endif
#if __has_include("qwen3_tts.h")
#include "qwen3_tts.h"
#define CA_HAVE_QWEN3_TTS 1
#endif
#if __has_include("kokoro.h")
#include "kokoro.h"
#define CA_HAVE_KOKORO 1
#endif
#if __has_include("piper_tts.h")
#include "piper_tts.h"
#define CA_HAVE_PIPER 1
#endif
#if __has_include("melotts.h")
#include "melotts.h"
#define CA_HAVE_MELOTTS 1
#endif
#if __has_include("chatterbox.h")
#include "chatterbox.h"
#define CA_HAVE_CHATTERBOX 1
#endif
#if __has_include("outetts.h")
#include "outetts.h"
#define CA_HAVE_OUTETTS 1
#endif
#if __has_include("csm_tts.h")
#include "csm_tts.h"
#define CA_HAVE_CSM 1
#endif
#if __has_include("dia_tts.h")
#include "dia_tts.h"
#define CA_HAVE_DIA 1
#endif
#if __has_include("pocket_tts.h")
#include "pocket_tts.h"
#define CA_HAVE_POCKET 1
#endif
#if __has_include("fastpitch_tts.h")
#include "fastpitch_tts.h"
#define CA_HAVE_FASTPITCH 1
#endif
#if __has_include("parler_tts.h")
#include "parler_tts.h"
#define CA_HAVE_PARLER 1
#endif
#if __has_include("speecht5_tts.h")
#include "speecht5_tts.h"
#define CA_HAVE_SPEECHT5 1
#endif
#if __has_include("bark_tts.h")
#include "bark_tts.h"
#define CA_HAVE_BARK 1
#endif
#if __has_include("parler_tts.h")
#include "parler_tts.h"
#define CA_HAVE_PARLER_TTS 1
#endif
#if __has_include("voxcpm2_tts.h")
#include "voxcpm2_tts.h"
#define CA_HAVE_VOXCPM2 1
#endif
#if __has_include("cosyvoice3_tts.h")
#include "cosyvoice3_tts.h"
#define CA_HAVE_COSYVOICE3 1
#endif
#if __has_include("indextts.h")
#include "indextts.h"
#define CA_HAVE_INDEXTTS 1
#endif
#if __has_include("f5_tts.h")
#include "f5_tts.h"
#define CA_HAVE_F5TTS 1
#endif
#if __has_include("m2m100.h")
#include "m2m100.h"
#define CA_HAVE_M2M100 1
#endif
#if __has_include("t5_translate.h")
#include "t5_translate.h"
#define CA_HAVE_T5_TRANSLATE 1
#endif
#if __has_include("orpheus.h")
#include "orpheus.h"
#define CA_HAVE_ORPHEUS 1
#endif
#if __has_include("tada_tts.h")
#include "tada_tts.h"
#define CA_HAVE_TADA 1
#endif
#if __has_include("mimo_asr.h")
#include "mimo_asr.h"
#define CA_HAVE_MIMO_ASR 1
#endif
#if __has_include("moss_audio.h")
#include "moss_audio.h"
#define CA_HAVE_MOSS_AUDIO 1
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
#if __has_include("moonshine_streaming.h")
#include "moonshine_streaming.h"
#define CA_HAVE_MOONSHINE_STREAMING 1
#endif
#if __has_include("gemma4_e2b.h")
#include "gemma4_e2b.h"
#define CA_HAVE_GEMMA4_E2B 1
#endif
#if __has_include("fireredpunc.h")
#include "fireredpunc.h"
#define CA_HAVE_FIREREDPUNC 1
#endif
#if __has_include("truecaser.h")
#include "truecaser.h"
#define CA_HAVE_TRUECASER 1
#endif
#if __has_include("truecaser_lstm.h")
#include "truecaser_lstm.h"
#define CA_HAVE_TRUECASER_LSTM 1
#endif
#if __has_include("truecaser_crf.h")
#include "truecaser_crf.h"
#define CA_HAVE_TRUECASER_CRF 1
#endif
#if __has_include("pcs.h")
#include "pcs.h"
#define CA_HAVE_PCS 1
#endif
#if __has_include("titanet.h")
#include "titanet.h"
#include "speaker_db.h"
#define CA_HAVE_TITANET 1
#endif
#include "crispasr_speaker_cluster.h"
#include "crispasr_speaker_embedder.h"
#include "crispasr_diarize_internal.h"
#include "pyannote_seg.h"

// CA_EXPORT decorates every C-ABI definition in this file. It MUST expand
// to the same linkage attributes that `CRISPASR_API` (from
// `include/crispasr.h`) puts on the public declaration, otherwise MSVC
// raises `C2375: redefinition; different linkage` (Windows static build
// CI failure 2026-05-20, caught at b6ab1655). Static builds set
// CRISPASR_API to empty; shared builds set __declspec(dllexport) on the
// library side and __declspec(dllimport) on the consumer side. Reusing
// the same macro for the impl side keeps decl and definition in lock-step
// across all three configurations.
#define CA_EXPORT extern "C" CRISPASR_API

// =========================================================================
// Module-level atomic progress (0-100, -1 = idle)
// =========================================================================
//
// Dart FFI cannot use C function pointers as callbacks, so for progress
// reporting we use a shared atomic integer that Dart polls via
// crispasr_get_progress(). The whisper progress_callback writes here;
// Dart reads it at leisure.

static std::atomic<int> g_progress{-1};

CA_EXPORT int crispasr_get_progress(void) {
    return g_progress.load(std::memory_order_relaxed);
}

CA_EXPORT void crispasr_reset_progress(void) {
    g_progress.store(-1, std::memory_order_relaxed);
}

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
CA_EXPORT void crispasr_params_set_best_of(whisper_full_params* p, int n) {
    if (p && n > 0)
        p->greedy.best_of = n;
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
CA_EXPORT void crispasr_params_set_max_tokens(whisper_full_params* p, int n) {
    if (p)
        p->max_tokens = n > 0 ? n : 0;
}
CA_EXPORT void crispasr_params_set_initial_prompt(whisper_full_params* p, const char* prompt) {
    if (p)
        p->initial_prompt = prompt; // caller owns the string
}

// VAD (crispasr built-in Silero pipeline). When enabled, whisper_full
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

// =========================================================================
// DTW (Dynamic Time Warping) context-params setter
// =========================================================================
//
// DTW fields live on `whisper_context_params`, set at context init. This
// setter lets Dart configure DTW token-level timestamps via a pointer to
// the params struct without mirroring its layout.

CA_EXPORT void crispasr_ctx_params_set_dtw(whisper_context_params* p, bool enable,
                                           int aheads_preset, // cast to whisper_alignment_heads_preset
                                           int n_top) {
    if (!p)
        return;
    p->dtw_token_timestamps = enable;
    p->dtw_aheads_preset = static_cast<whisper_alignment_heads_preset>(aheads_preset);
    p->dtw_n_top = n_top;
}

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

CA_EXPORT int64_t crispasr_token_dtw_t(whisper_context* ctx, int i_segment, int i_token) {
    if (!ctx)
        return 0;
    return whisper_full_get_token_data(ctx, i_segment, i_token).t_dtw;
}

// =========================================================================
// Alternative-candidate tokens (`alt_n` knob, greedy decode only)
// =========================================================================
//
// Whisper's per-step softmax produces a full distribution over the vocab;
// `whisper_full_get_token_data` returns just the chosen token. When
// `wparams.alt_n > 0` (set via crispasr_params_set_alt_n), the decoder
// also stashes the top-N runner-up candidates so consumers can build
// tap-to-pick UIs for ambiguous proper nouns / technical jargon. Beam
// search is excluded (siblings are beam-conditional, not greedy alts).
CA_EXPORT void crispasr_params_set_alt_n(whisper_full_params* p, int n) {
    if (p)
        p->alt_n = n < 0 ? 0 : (n > 32 ? 32 : n); // sanity clamp; UI caps at 5
}

CA_EXPORT int crispasr_token_n_alts(whisper_context* ctx, int i_seg, int i_tok) {
    if (!ctx)
        return 0;
    return whisper_full_get_token_n_alts(ctx, i_seg, i_tok);
}

CA_EXPORT int32_t crispasr_token_alt_id(whisper_context* ctx, int i_seg, int i_tok, int i_alt) {
    if (!ctx)
        return 0;
    return (int32_t)whisper_full_get_token_alt_id(ctx, i_seg, i_tok, i_alt);
}

CA_EXPORT float crispasr_token_alt_p(whisper_context* ctx, int i_seg, int i_tok, int i_alt) {
    if (!ctx)
        return 0.0f;
    return whisper_full_get_token_alt_p(ctx, i_seg, i_tok, i_alt);
}

// Resolve alt token id to its display string via whisper's vocab. Writes
// into the caller's buffer; returns bytes written (excluding NUL), 0
// on empty / out-of-range, or -1 when the buffer is too small. Mirrors
// the registry-lookup ABI's "fill caller buffer" convention.
CA_EXPORT int crispasr_token_alt_text(whisper_context* ctx, int i_seg, int i_tok, int i_alt, char* out, int out_cap) {
    if (!ctx || !out || out_cap <= 0)
        return -1;
    const int n = whisper_full_get_token_n_alts(ctx, i_seg, i_tok);
    if (i_alt < 0 || i_alt >= n) {
        if (out_cap > 0)
            out[0] = '\0';
        return 0;
    }
    const whisper_token id = whisper_full_get_token_alt_id(ctx, i_seg, i_tok, i_alt);
    const char* t = whisper_token_to_str(ctx, id);
    if (!t) {
        out[0] = '\0';
        return 0;
    }
    const int len = (int)std::strlen(t);
    if (len + 1 > out_cap) {
        return -1;
    }
    std::memcpy(out, t, (size_t)len);
    out[len] = '\0';
    return len;
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
// VAD — run Silero on PCM, return [start_cs, end_cs] pairs
// =========================================================================
//
// `out_spans` is a malloc'd array of centisecond floats (2 per span). The
// caller must pass the pointer back to `crispasr_vad_free` when done. Returns
// the number of speech segments detected (>= 0), or a negative error.
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

// Dispatcher-backed VAD slicing. Unlike crispasr_vad_segments above, this
// routes through crispasr_compute_vad_slices so GGUF VAD backends such as
// Whisper-VAD-EncDec, FireRedVAD, and MarbleNet use the same path as the CLI.
//
// `out_spans` is a malloc'd array of [start_s, end_s] float pairs. The caller
// must pass it to crispasr_vad_free. Returns the number of slices (>= 0), or a
// negative error.
//
//   -1  bad arguments
//   -2  allocation failed
CA_EXPORT int crispasr_vad_slices(const char* vad_model_path, const float* pcm, int n_samples, int sample_rate,
                                  float threshold, int min_speech_ms, int min_silence_ms, int speech_pad_ms,
                                  float max_chunk_duration_s, int n_threads, float** out_spans) {
    if (!vad_model_path || !*vad_model_path || !pcm || n_samples <= 0 || sample_rate <= 0 || !out_spans)
        return -1;
    *out_spans = nullptr;

    crispasr_vad_options opts;
    if (threshold > 0.0f) {
        opts.threshold = threshold;
        opts.threshold_explicit = true;
    }
    if (min_speech_ms > 0)
        opts.min_speech_duration_ms = min_speech_ms;
    if (min_silence_ms > 0)
        opts.min_silence_duration_ms = min_silence_ms;
    const int pad_ms = speech_pad_ms > 0 ? speech_pad_ms : 0;
    // Apply padding below for all dispatcher backends. Some implementations
    // (for example Whisper-VAD-EncDec) ignore opts.speech_pad_ms internally.
    opts.speech_pad_ms = 0;
    if (max_chunk_duration_s > 0.0f)
        opts.chunk_seconds = (int)std::ceil(max_chunk_duration_s);
    else
        opts.chunk_seconds = 0;
    if (n_threads > 0)
        opts.n_threads = n_threads;

    std::vector<crispasr_audio_slice> slices =
        crispasr_compute_vad_slices(pcm, n_samples, sample_rate, vad_model_path, opts);
    const int n = (int)slices.size();
    if (n == 0)
        return 0;

    float* buf = (float*)std::malloc(sizeof(float) * 2 * n);
    if (!buf)
        return -2;

    const float duration_s = (float)n_samples / (float)sample_rate;
    const float pad_s = (float)pad_ms / 1000.0f;
    for (int i = 0; i < n; ++i) {
        float start_s = (float)slices[i].t0_cs / 100.0f;
        float end_s = (float)slices[i].t1_cs / 100.0f;
        if (pad_s > 0.0f) {
            start_s = std::max(0.0f, start_s - pad_s);
            end_s = std::min(duration_s, end_s + pad_s);
        }
        buf[2 * i + 0] = start_s;
        buf[2 * i + 1] = end_s;
    }
    *out_spans = buf;
    return n;
}

CA_EXPORT void crispasr_vad_free(float* spans) {
    if (spans)
        std::free(spans);
}

// =========================================================================
// LCS chunk-boundary deduplication
// =========================================================================
//
// Public-API entry point for the NeMo-style LCS hypothesis stitcher used
// internally by the CLI's overlap-save chunking path. Exposed so bindings
// that drive `libcrispasr` chunk-by-chunk (Go cgo, Rust, Dart FFI, Python
// ctypes) can run the same dedup on their own per-chunk token streams
// without re-implementing the algorithm.
//
// Pure function over the input arrays — no state, no thread safety
// concerns. See `src/core/crispasr_lcs.h` for the algorithm itself; this
// is a 4-line C-ABI wrapper that does input validation + namespace
// stripping.

// =========================================================================
// AI-generated audio watermark
// =========================================================================
//
// Dispatches to AudioSeal (neural) if a model has been loaded via
// crispasr_watermark_load_model(), otherwise falls back to the built-in
// spread-spectrum watermark.

#include "../examples/cli/crispasr_watermark.h"
#include "audioseal.h"

// Global AudioSeal context for C ABI callers.
static audioseal_ctx* g_audioseal_ctx = nullptr;

CA_EXPORT int crispasr_watermark_load_model(const char* gguf_path) {
    if (!gguf_path || !gguf_path[0])
        return -1;
    if (g_audioseal_ctx) {
        audioseal_free(g_audioseal_ctx);
        g_audioseal_ctx = nullptr;
    }
    auto params = audioseal_default_params();
    params.verbosity = 1;
    g_audioseal_ctx = audioseal_init_from_file(gguf_path, params);
    return g_audioseal_ctx ? 0 : -1;
}

CA_EXPORT float crispasr_watermark_detect(const float* pcm, int n_samples) {
    if (!pcm || n_samples <= 0)
        return 0.0f;
    if (g_audioseal_ctx) {
        int n_frames = 0;
        float* probs = audioseal_detect(g_audioseal_ctx, pcm, n_samples, &n_frames, nullptr);
        if (probs && n_frames > 0) {
            double avg = 0.0;
            for (int i = 0; i < n_frames; i++)
                avg += probs[i];
            avg /= (double)n_frames;
            free(probs);
            return (float)avg;
        }
        if (probs)
            free(probs);
    }
    return ::crispasr_watermark_detect_impl(pcm, n_samples);
}

CA_EXPORT void crispasr_watermark_embed(float* pcm, int n_samples, float alpha) {
    if (!pcm || n_samples <= 0)
        return;
    if (g_audioseal_ctx) {
        float* watermarked = audioseal_embed(g_audioseal_ctx, pcm, n_samples, nullptr);
        if (watermarked) {
            memcpy(pcm, watermarked, (size_t)n_samples * sizeof(float));
            free(watermarked);
            return;
        }
    }
    ::crispasr_watermark_embed_impl(pcm, n_samples, alpha > 0.0f ? alpha : 0.005f);
}

#include "core/crispasr_lcs.h"

CA_EXPORT int crispasr_lcs_dedup_prefix_count(const int32_t* prev_tail_tokens, int n_prev, const int32_t* curr_tokens,
                                              int n_curr, int min_lcs_length) {
    if (!prev_tail_tokens || !curr_tokens || n_prev <= 0 || n_curr <= 0)
        return 0;
    std::vector<int32_t> prev(prev_tail_tokens, prev_tail_tokens + n_prev);
    std::vector<int32_t> curr(curr_tokens, curr_tokens + n_curr);
    const int min_l = min_lcs_length > 0 ? min_lcs_length : crispasr_lcs::kMinMergeSubsequenceLen;
    return crispasr_lcs::lcs_dedup_prefix_count(prev, curr, min_l);
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
// is what crispasr itself supports.

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

    // PLAN #62c — opaque kyutai_stt_stream*; when set, all crispasr_stream_*
    // functions route to the kyutai backend instead of whisper. Mutually
    // exclusive with `ctx`.
    void* kyutai_stream_state = nullptr;

    // PLAN #62c follow-on — opaque moonshine_streaming_stream*; same pattern.
    void* moonshine_streaming_state = nullptr;

    // PLAN #7 — opaque voxtral4b_stream*; native incremental encoder + LLM
    // decode-on-flush. Mutually exclusive with `ctx`.
    void* voxtral4b_stream_state = nullptr;
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
#if __has_include("kyutai_stt.h")
    if (s->kyutai_stream_state) {
        kyutai_stt_stream_close((kyutai_stt_stream*)s->kyutai_stream_state);
        s->kyutai_stream_state = nullptr;
    }
#endif
#if __has_include("moonshine_streaming.h")
    if (s->moonshine_streaming_state) {
        moonshine_streaming_stream_close((moonshine_streaming_stream*)s->moonshine_streaming_state);
        s->moonshine_streaming_state = nullptr;
    }
#endif
#if __has_include("voxtral4b.h")
    if (s->voxtral4b_stream_state) {
        voxtral4b_stream_close((voxtral4b_stream*)s->voxtral4b_stream_state);
        s->voxtral4b_stream_state = nullptr;
    }
#endif
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

    whisper_full_params wparams = whisper_full_default_params(CRISPASR_SAMPLING_GREEDY);
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
#if __has_include("kyutai_stt.h")
    if (s->kyutai_stream_state) {
        return kyutai_stt_stream_feed((kyutai_stt_stream*)s->kyutai_stream_state, pcm, n_samples);
    }
#endif
#if __has_include("moonshine_streaming.h")
    if (s->moonshine_streaming_state) {
        return moonshine_streaming_stream_feed((moonshine_streaming_stream*)s->moonshine_streaming_state, pcm,
                                               n_samples);
    }
#endif
#if __has_include("voxtral4b.h")
    if (s->voxtral4b_stream_state) {
        return voxtral4b_stream_feed((voxtral4b_stream*)s->voxtral4b_stream_state, pcm, n_samples);
    }
#endif
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
#if __has_include("kyutai_stt.h")
    if (s->kyutai_stream_state) {
        return kyutai_stt_stream_get_text((kyutai_stt_stream*)s->kyutai_stream_state, out_text, out_cap, out_t0_s,
                                          out_t1_s, out_counter);
    }
#endif
#if __has_include("moonshine_streaming.h")
    if (s->moonshine_streaming_state) {
        return moonshine_streaming_stream_get_text((moonshine_streaming_stream*)s->moonshine_streaming_state, out_text,
                                                   out_cap, out_t0_s, out_t1_s, out_counter);
    }
#endif
#if __has_include("voxtral4b.h")
    if (s->voxtral4b_stream_state) {
        return voxtral4b_stream_get_text((voxtral4b_stream*)s->voxtral4b_stream_state, out_text, out_cap, out_t0_s,
                                         out_t1_s, out_counter);
    }
#endif
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
#if __has_include("kyutai_stt.h")
    if (s->kyutai_stream_state) {
        return kyutai_stt_stream_flush((kyutai_stt_stream*)s->kyutai_stream_state);
    }
#endif
#if __has_include("moonshine_streaming.h")
    if (s->moonshine_streaming_state) {
        return moonshine_streaming_stream_flush((moonshine_streaming_stream*)s->moonshine_streaming_state);
    }
#endif
#if __has_include("voxtral4b.h")
    if (s->voxtral4b_stream_state) {
        return voxtral4b_stream_flush((voxtral4b_stream*)s->voxtral4b_stream_state);
    }
#endif
    if (s->accum.empty())
        return 0;
    return crispasr_stream_run_decode(s) == 0 ? 1 : -2;
}

// PLAN #7 phase 3 — voxtral4b live-captions toggle on a unified stream.
// Forwards to voxtral4b_stream_set_live_decode if the underlying stream is
// voxtral4b; no-op for other backends. Set BEFORE the first feed for
// clean semantics. Idempotent.
CA_EXPORT void crispasr_stream_set_live_decode(crispasr_stream* s, int enabled) {
    if (!s)
        return;
#if __has_include("voxtral4b.h")
    if (s->voxtral4b_stream_state) {
        voxtral4b_stream_set_live_decode((voxtral4b_stream*)s->voxtral4b_stream_state, enabled);
    }
#endif
    (void)enabled; // silence unused-warning when no streaming-capable backend is built in
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
    else if (strcmp(arch, "lfm2-audio") == 0)
        backend = "lfm2-audio";
    else if (strcmp(arch, "cohere-transcribe") == 0)
        backend = "cohere";
    else if (strcmp(arch, "qwen3-asr") == 0 || strcmp(arch, "qwen3asr") == 0)
        backend = "qwen3";
    else if (strcmp(arch, "voxtral") == 0)
        backend = "voxtral";
    else if (strcmp(arch, "voxtral4b") == 0)
        backend = "voxtral4b";
    else if (strcmp(arch, "granite-speech") == 0)
        backend = "granite";
    else if (strcmp(arch, "granite_nle") == 0 || strcmp(arch, "granite-nle") == 0)
        backend = "granite-4.1-nar";
    else if (strcmp(arch, "fastconformer-ctc") == 0)
        backend = "fastconformer-ctc";
    else if (strcmp(arch, "canary-ctc") == 0)
        backend = "canary-ctc";
    else if (strcmp(arch, "wav2vec2") == 0)
        backend = "wav2vec2";
    else if (strcmp(arch, "vibevoice-asr") == 0 || strcmp(arch, "vibevoice") == 0 || strcmp(arch, "vibevoice-tts") == 0)
        backend = "vibevoice";
    else if (strcmp(arch, "qwen3-tts") == 0 || strcmp(arch, "qwen3_tts") == 0)
        backend = "qwen3-tts";
    else if (strcmp(arch, "orpheus") == 0)
        backend = "orpheus";
    else if (strcmp(arch, "chatterbox") == 0 || strcmp(arch, "chatterbox_turbo") == 0 ||
             strcmp(arch, "kartoffelbox") == 0)
        backend = "chatterbox";
    else if (strcmp(arch, "outetts") == 0)
        backend = "outetts";
    else if (strcmp(arch, "voxcpm2") == 0 || strcmp(arch, "voxcpm2-tts") == 0)
        backend = "voxcpm2-tts";
    else if (strcmp(arch, "cosyvoice3-llm") == 0 || strcmp(arch, "cosyvoice3") == 0 ||
             strcmp(arch, "cosyvoice3-tts") == 0)
        backend = "cosyvoice3-tts";
    else if (strcmp(arch, "indextts") == 0)
        backend = "indextts";
    else if (strcmp(arch, "f5-tts") == 0 || strcmp(arch, "f5tts") == 0)
        backend = "f5-tts";
    else if (strcmp(arch, "m2m100") == 0)
        backend = "m2m100";
    else if (strcmp(arch, "parler-tts") == 0 || strcmp(arch, "parler_tts") == 0 || strcmp(arch, "parlertts") == 0)
        backend = "parler-tts";
    else if (strcmp(arch, "t5") == 0)
        backend = "madlad";
    else if (strcmp(arch, "moss_audio") == 0 || strcmp(arch, "moss-audio") == 0)
        backend = "moss-audio";
    else if (strcmp(arch, "kugelaudio") == 0 || strcmp(arch, "kugelaudio-tts") == 0)
        backend = "kugelaudio";
    else if (strcmp(arch, "zonos") == 0 || strcmp(arch, "zonos-tts") == 0)
        backend = "zonos";

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

// ─────────────────────────────────────────────────────────────────────
// Open-time params (CrispASR 0.6.1). The previous open ABI took only
// `(model_path, backend, n_threads)`; backend-specific knobs like
// `use_gpu` were either compile-time (CMake flags) or default-true.
// To let host apps toggle at runtime we now thread two extra flags
// through `crispasr_session_open_explicit` via thread-local storage:
// the new `crispasr_session_open_with_params` export sets them
// before delegating, then resets them on the way out.
//
// Why thread-locals instead of an explicit parameter? `open_explicit`
// is the central choke-point used by the auto-detect path
// (`crispasr_session_open` → detect → open_explicit) and by every
// language binding. Adding a 4th positional arg would break those
// callers; appending another ABI is its own risk. A thread-local pair
// lets the new export set defaults for its own call without disturbing
// the existing API surface, and the helpers below ensure the values
// reset to their static defaults afterwards.
//
// `use_gpu` defaults to true so unmodified callers behave like
// pre-0.6.1 builds (which always passed GPU-on params). `verbosity`
// defaults to 0 (silent) for the same reason.
//
// `flash_attn` and `n_gpu_layers` (added in 0.6.2) follow the same
// pattern. Today only the whisper backend's `whisper_context_params`
// has a native `flash_attn` field; other backends accept the toggle
// but their compute graphs don't yet branch on it (a per-backend
// kernel-level commit lands those incrementally). `n_gpu_layers`
// is reserved for backends with a llama.cpp-style layer-offload
// concept (orpheus / voxtral / qwen3 / granite LLM); -1 means
// "as many as possible", 0 = CPU-only inference.
// ─────────────────────────────────────────────────────────────────────
static thread_local bool g_open_use_gpu_tls = true;
static thread_local int g_open_verbosity_tls = 0;
static thread_local bool g_open_flash_attn_tls = true;
static thread_local int g_open_n_gpu_layers_tls = -1;
static thread_local float g_open_temperature_tls = 0.0f;
static thread_local uint64_t g_open_seed_tls = 0;

struct crispasr_session {
    std::string backend; // "whisper", "parakeet", ...
    std::string model_path;
    int n_threads = 4;

    // Last synthesize error — populated by synthesize_raw_impl when it
    // returns nullptr so callers can surface a meaningful reason instead
    // of the generic "no audio produced". Cleared on every synthesize call.
    std::string last_synth_error;

    // Sticky session-level state (PLAN #59 partial unblock — the
    // capabilities matrix items that were previously CLI-only). Per-call
    // args still win when supplied; these are the fallback.
    std::string source_language; // canary/cohere/voxtral source-lang hint
    std::string target_language; // canary/cohere/voxtral target-lang (≠ source ⇒ translate)
    bool punctuation = true;     // canary/cohere per-call arg + post-process gate
    bool translate = false;      // whisper sticky --translate (others: use src/tgt mismatch)

    // --punc-model post-processor (set via crispasr_session_set_punc_model).
    // Held as void* so the struct doesn't depend on the optionally-compiled
    // fireredpunc/pcs headers; at most one is non-null. Applied per segment
    // after transcription (gated on `punctuation`), mirroring the CLI/server.
    void* punc_ctx = nullptr; // fireredpunc_context*
    void* pcs_ctx = nullptr;  // pcs_context*
    // Free-form audio Q&A prompt for instruct-tuned audio-LLM backends
    // (voxtral / voxtral4b / qwen3-asr). When non-empty, replaces the
    // standard "lang:<X>[TRANSCRIBE]" suffix with `[/INST]<prompt>`,
    // causing the LLM to answer the question instead of transcribing
    // verbatim. Empty means "transcribe normally" — the historical
    // default.
    std::string ask;
    // Best-of-N: run N independent decodes and keep the lowest-perplexity
    // one. Only effective when temperature > 0. Default 1 (no resampling).
    int best_of = 1;
    int max_new_tokens = 0;
    float frequency_penalty = 0.0f;
    float temperature = 0.0f; // 0 = greedy / backend default
    uint64_t seed = 0;        // 0 = time-based

    // ── §90 session-level beam-search width ──────────────────────────────
    // beam_size > 1 activates beam search for backends that support it:
    //   whisper        — native BEAM_SEARCH strategy
    //   qwen3-asr      — via core_beam_decode::run_with_probs (replay)
    //   granite*        — via core_beam_decode::run_with_probs
    //   voxtral         — via core_beam_decode::run_with_probs
    //   glm-asr         — glm_asr_set_beam_size (per-backend setter)
    //   kyutai-stt      — kyutai_stt_set_beam_size
    //   firered         — firered_asr_set_beam_size
    //   moonshine       — moonshine_set_beam_size
    //   omniasr-llm     — omniasr_set_beam_size
    //   canary          — canary_set_beam_size (branched-KV AED beam)
    //   cohere          — cohere_set_beam_size (branched-KV AED beam)
    // Silent no-op for: voxtral4b (streaming, no beam hook), CTC/NAR backends.
    // Default 1 preserves greedy bit-identical output (no-regression contract).
    int beam_size = 1;

    // Whisper text-suppression + prompt-carry extras (whisper-only).
    // Map 1-to-1 onto wparams.suppress_nst / suppress_regex /
    // carry_initial_prompt on every transcribe dispatch.
    //
    // Defaults match whisper_full_default_params: nst off,
    // regex empty (no suppression), carry off. Set via the
    // matching C-ABI `crispasr_session_set_whisper_decode_extras`.
    bool whisper_suppress_nst = false;
    std::string whisper_suppress_regex;
    bool whisper_carry_initial_prompt = false;

    // Whisper decoder-fallback thresholds (whisper-only — none of
    // these fields exist on the other backends' wparams equivalent).
    //
    // Defaults are the same as `whisper_full_default_params` so an
    // unmodified session matches whisper.cpp's stock behaviour. The
    // values get written into wparams.{entropy,logprob,no_speech}
    // _thold + wparams.temperature_inc on every whisper transcribe
    // dispatch — same shape as the other sticky setters.
    //
    // Set `temperature_inc = 0.0f` to disable the temperature-
    // fallback loop entirely (= the CLI's `--no-fallback`).
    float entropy_thold = 2.4f;
    float logprob_thold = -1.0f;
    float no_speech_thold = 0.6f;
    float temperature_inc = 0.2f;

    // Per-token top-N alternative-candidate capture (whisper greedy
    // decode only). 0 = off (default). Written into wparams.alt_n on
    // every whisper dispatch. UI caps at 5 to keep memory tame at
    // ~50 KB/min of audio.
    int alt_n = 0;

    // GBNF grammar-constrained sampling state (whisper backend only —
    // wparams.grammar_rules lives in whisper_full_params, no analog
    // on other backends today).
    //
    // Lifecycle:
    //   * `crispasr_session_set_grammar_text(s, "<gbnf>", "root", 100.0f)`
    //     re-parses the source, populates `grammar_parsed` + the cached
    //     `grammar_rules_ptrs` vector, and stores the root rule name.
    //   * An empty `grammar_text` means "no grammar"; the transcribe
    //     path skips the rules-wiring branch and runs unconstrained.
    //   * `grammar_rules_ptrs` is a vector of POINTERS into
    //     `grammar_parsed.rules`. Both must outlive the transcribe call,
    //     so they're members of the session, not stack locals.
    std::string grammar_text;
    std::string grammar_root_rule;
    float grammar_penalty = 100.0f; // whisper.cpp default
    grammar_parser::parse_state grammar_parsed;
    std::vector<const whisper_grammar_element*> grammar_rules_ptrs;
    uint32_t grammar_root_rule_id = 0;
    bool grammar_active = false;

    // Session-level hotwords for contextual biasing (PLAN §5.26.2).
    // Stored as a comma-separated string; parsed into per-backend form
    // on the next transcribe call. For parakeet CTC/TDT, the parsed
    // words are fed to parakeet_set_hotwords(); for LLM backends, they
    // are injected into the ask prompt.
    std::string hotwords;
    float hotwords_boost = 1.5f;

    // Exactly one of these pointers is non-null based on `backend`.
    whisper_context* whisper_ctx = nullptr;
#ifdef CA_HAVE_PARAKEET
    parakeet_context* parakeet_ctx = nullptr;
#endif
#ifdef CA_HAVE_CANARY
    canary_context* canary_ctx = nullptr;
#endif
#ifdef CA_HAVE_LFM2_AUDIO
    lfm2_audio_context* lfm2_audio_ctx = nullptr;
#endif
#ifdef CA_HAVE_MINI_OMNI2
    mini_omni2_context* mini_omni2_ctx = nullptr;
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
#ifdef CA_HAVE_GRANITE_NLE
    // granite-speech-4.1-2b-nar — non-autoregressive variant. Different
    // pipeline (single LLM forward pass + edit-slot greedy decode); the
    // simpler high-level `granite_nle_transcribe` entry handles it, no
    // need to mirror the granite_speech multi-step plumbing.
    granite_nle_context* granite_nle_ctx = nullptr;
#endif
#ifdef CA_HAVE_FUNASR
    funasr_context* funasr_ctx = nullptr;
#endif
#ifdef CA_HAVE_PARAFORMER
    paraformer_context* paraformer_ctx = nullptr;
#endif
#ifdef CA_HAVE_SENSEVOICE
    sensevoice_context* sensevoice_ctx = nullptr;
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
#ifdef CA_HAVE_KUGELAUDIO
    kugelaudio_context* kugelaudio_ctx = nullptr;
#endif
#ifdef CA_HAVE_ZONOS
    zonos_tts_context* zonos_ctx = nullptr;
#endif
#ifdef CA_HAVE_QWEN3_TTS
    qwen3_tts_context* qwen3_tts_ctx = nullptr;
    bool qwen3_tts_voice_loaded = false;
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
#ifdef CA_HAVE_MOONSHINE_STREAMING
    void* moonshine_streaming_ctx = nullptr;
#endif
#ifdef CA_HAVE_GEMMA4_E2B
    void* gemma4_e2b_ctx = nullptr;
#endif
#ifdef CA_HAVE_OMNIASR
    void* omniasr_ctx = nullptr;
#endif
#ifdef CA_HAVE_ORPHEUS
    orpheus_context* orpheus_ctx = nullptr;
    bool orpheus_codec_loaded = false;
#endif
#ifdef CA_HAVE_TADA
    tada_context* tada_ctx = nullptr;
#endif
#ifdef CA_HAVE_KOKORO
    kokoro_context* kokoro_ctx = nullptr;
#endif
#ifdef CA_HAVE_CHATTERBOX
    chatterbox_context* chatterbox_ctx = nullptr;
#endif
#ifdef CA_HAVE_OUTETTS
    outetts_context* outetts_ctx = nullptr;
#endif
#ifdef CA_HAVE_CSM
    csm_tts_context* csm_tts_ctx = nullptr;
#endif
#ifdef CA_HAVE_DIA
    dia_tts_context* dia_tts_ctx = nullptr;
#endif
#ifdef CA_HAVE_POCKET
    pocket_tts_context* pocket_tts_ctx = nullptr;
#endif
#ifdef CA_HAVE_FASTPITCH
    fastpitch_tts_context* fastpitch_ctx = nullptr;
#endif
#ifdef CA_HAVE_PARLER
    parler_tts_context* parler_ctx = nullptr;
#endif
#ifdef CA_HAVE_SPEECHT5
    speecht5_tts_context* speecht5_ctx = nullptr;
    std::vector<float> speecht5_speaker; // 512-d x-vector
#endif
#ifdef CA_HAVE_BARK
    bark_context* bark_ctx = nullptr;
#endif
#ifdef CA_HAVE_PARLER_TTS
    parler_tts_context* parler_tts_ctx = nullptr;
    std::string parler_description; // voice description for T5 conditioning
#endif
#ifdef CA_HAVE_VOXCPM2
    voxcpm2_context* voxcpm2_ctx = nullptr;
    std::vector<float> voxcpm2_ref_pcm; // 16 kHz mono cloning reference
#endif
#ifdef CA_HAVE_COSYVOICE3
    cosyvoice3_tts_context* cosyvoice3_ctx = nullptr;
    std::string cosyvoice3_voice;    // bank voice name OR *.wav clone path (set_voice)
    std::string cosyvoice3_ref_text; // ref transcription for *.wav cloning
#endif
#ifdef CA_HAVE_INDEXTTS
    indextts_context* indextts_ctx = nullptr;
    std::vector<float> indextts_ref_pcm; // 24 kHz mono cloning reference
#endif
#ifdef CA_HAVE_F5TTS
    f5_tts_context* f5tts_ctx = nullptr;
#endif
#ifdef CA_HAVE_PIPER
    piper_tts_context* piper_ctx = nullptr;
#endif
#ifdef CA_HAVE_MELOTTS
    melotts_context* melotts_ctx = nullptr;
#endif
#ifdef CA_HAVE_M2M100
    m2m100_context* m2m100_ctx = nullptr;
#endif
#ifdef CA_HAVE_T5_TRANSLATE
    t5_translate_context* t5_translate_ctx = nullptr;
#endif
#ifdef CA_HAVE_MIMO_ASR
    mimo_asr_context* mimo_asr_ctx = nullptr;
#endif
#ifdef CA_HAVE_MOSS_AUDIO
    moss_audio_context* moss_audio_ctx = nullptr;
#endif
};

struct crispasr_session_seg {
    std::string text;
    int64_t t0 = 0; // centiseconds absolute
    int64_t t1 = 0;
    struct word_alt {
        std::string text;
        float p = 0.0f;
    };
    struct word {
        std::string text;
        int64_t t0 = 0; // centiseconds absolute
        int64_t t1 = 0;
        float p = 1.0f;
        // Top-N alternative candidates for the first content token of
        // this word (whisper greedy decode only, when alt_n > 0).
        // Empty when alts weren't captured or the backend doesn't
        // produce them. Ordered descending by p.
        std::vector<word_alt> alts;
    };
    std::vector<word> words;
};

struct crispasr_session_result {
    std::vector<crispasr_session_seg> segments;
    std::string backend;
};

// Per-token data fed into emit_words_from_tokens. Backends with their own
// token-prob APIs project into this shape so the word-grouping logic stays
// in one place.
struct ca_token_record {
    std::string text;
    int64_t t0;
    int64_t t1;
    float p;
    // Optional per-token top-N alternative candidates. Only the
    // whisper-greedy path populates these (when alt_n > 0); other
    // backends leave it empty. emit_words_from_tokens attaches the
    // alts of each word's first content-bearing token to the emitted
    // word — see the inline note there for why first-token only.
    std::vector<crispasr_session_seg::word_alt> alts;
};

// GPT-2 byte-level BPE decoder. Mirrors HF's bytes_to_unicode reverse map.
// Used by qwen3 / granite tokenizers and any GPT-2 / Llama-3 family.
static const std::vector<int>& gpt2_byte_decoder() {
    static std::vector<int> dec;
    static bool initialized = false;
    if (initialized)
        return dec;
    dec.assign(0x200, -1);
    std::vector<int> bs, cs;
    for (int b = 0x21; b <= 0x7e; b++) {
        bs.push_back(b);
        cs.push_back(b);
    }
    for (int b = 0xa1; b <= 0xac; b++) {
        bs.push_back(b);
        cs.push_back(b);
    }
    for (int b = 0xae; b <= 0xff; b++) {
        bs.push_back(b);
        cs.push_back(b);
    }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool present = false;
        for (int x : bs)
            if (x == b) {
                present = true;
                break;
            }
        if (!present) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    for (size_t i = 0; i < bs.size(); i++)
        if ((size_t)cs[i] < dec.size())
            dec[cs[i]] = bs[i];
    initialized = true;
    return dec;
}

static std::string gpt2_byte_decode(const std::string& s) {
    const auto& dec = gpt2_byte_decoder();
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        int cp = 0, len = 1;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            len = 4;
        } else {
            i++;
            continue;
        }
        if (i + len > s.size())
            break;
        for (int k = 1; k < len; k++)
            cp = (cp << 6) | (s[i + k] & 0x3F);
        i += len;
        if (cp >= 0 && cp < (int)dec.size() && dec[cp] >= 0)
            out.push_back((char)dec[cp]);
    }
    return out;
}

// SentencePiece-style word grouping. Each token's `text` either starts with a
// leading space (Latin convention: token is the start of a new word) or
// continues the previous word. Punctuation-only tokens attach to the previous
// word. Per-word probability is the arithmetic mean of contributing tokens'
// softmax probs — matches parakeet's word grouping convention.
static std::vector<crispasr_session_seg::word> emit_words_from_tokens(const std::vector<ca_token_record>& toks) {
    auto is_punct_only = [](const std::string& s) {
        if (s.empty())
            return false;
        for (char c : s) {
            unsigned char u = (unsigned char)c;
            if ((u >= 'A' && u <= 'Z') || (u >= 'a' && u <= 'z') || (u >= '0' && u <= '9') || u >= 0x80)
                return false;
        }
        return true;
    };

    std::vector<crispasr_session_seg::word> out;
    crispasr_session_seg::word cur;
    bool have_cur = false;
    float cur_p_sum = 0.0f;
    int cur_p_cnt = 0;

    auto flush = [&]() {
        if (cur_p_cnt > 0)
            cur.p = cur_p_sum / (float)cur_p_cnt;
        else
            cur.p = 1.0f;
        out.push_back(std::move(cur));
        cur = {};
        cur_p_sum = 0.0f;
        cur_p_cnt = 0;
        have_cur = false;
    };

    for (const auto& tk : toks) {
        if (tk.text.empty())
            continue;
        if (tk.text == " ") {
            // Standalone space (CTC-style ▁ → ' ' emission) marks a word
            // boundary. Flush the current accumulating word but don't add
            // anything else.
            if (have_cur)
                flush();
            continue;
        }

        const bool has_leading_space = (tk.text[0] == ' ');
        const bool punct = is_punct_only(tk.text);

        if (has_leading_space && !punct && have_cur)
            flush();

        if (!have_cur) {
            cur.t0 = tk.t0;
            // Attribute the first content token's top-N alts to the
            // emitted word. Whisper tokens are sub-word (BPE-ish), so
            // for a multi-token word like "kubectl" → ["kub","ect","l"]
            // we surface alternatives of "kub" only. That's the
            // discriminating token in practice — if the user sees
            // "cubicle" as an alt for "kub", they know the model
            // wavered there. Full word-level enumeration would require
            // expanding a token-tree per word; out of scope for v1.
            cur.alts = tk.alts;
            have_cur = true;
        }
        cur.t1 = tk.t1;
        cur_p_sum += tk.p;
        cur_p_cnt += 1;
        cur.text += has_leading_space ? tk.text.substr(1) : tk.text;
    }
    if (have_cur)
        flush();
    return out;
}

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
        cparams.use_gpu = g_open_use_gpu_tls;
        cparams.flash_attn = g_open_flash_attn_tls;
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
        pp.verbosity = g_open_verbosity_tls;
        pp.use_gpu = g_open_use_gpu_tls;
        // Parakeet's pre-existing toggle is named `use_flash`; the
        // unified open-params calls it `flash_attn`. Both map to the
        // same kernel switch in the encoder SA blocks.
        pp.use_flash = g_open_flash_attn_tls;
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
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.use_flash = g_open_flash_attn_tls;
        s->canary_ctx = canary_init_from_file(model_path, p);
        if (!s->canary_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_LFM2_AUDIO
    if (s->backend == "lfm2-audio") {
        lfm2_audio_context_params p = lfm2_audio_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->lfm2_audio_ctx = lfm2_audio_init_from_file(model_path, p);
        if (!s->lfm2_audio_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_MINI_OMNI2
    if (s->backend == "mini-omni2" || s->backend == "mini_omni2" || s->backend == "miniomni2") {
        mini_omni2_context_params p = mini_omni2_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->mini_omni2_ctx = mini_omni2_init_from_file(model_path, p);
        if (!s->mini_omni2_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_QWEN3
    // mega-asr ships as Qwen3-ASR-1.7B with a merged robustness LoRA —
    // architecturally identical to qwen3, so it loads through the same
    // dispatch. Same alias set the CLI accepts in
    // examples/cli/crispasr_backend.cpp::resolve_make_fn().
    if (s->backend == "qwen3" || s->backend == "mega-asr" || s->backend == "mega_asr" || s->backend == "megaasr") {
        qwen3_asr_context_params p = qwen3_asr_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
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
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.use_flash = g_open_flash_attn_tls;
        s->cohere_ctx = cohere_init_from_file(model_path, p);
        if (!s->cohere_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_GRANITE
    if (s->backend == "granite" || s->backend == "granite-4.1" || s->backend == "granite-4.1-plus") {
        granite_speech_context_params p = granite_speech_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
        s->granite_ctx = granite_speech_init_from_file(model_path, p);
        if (!s->granite_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_GRANITE_NLE
    if (s->backend == "granite-4.1-nar" || s->backend == "granite-nle" || s->backend == "granite_nle") {
        granite_nle_context_params p = granite_nle_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->granite_nle_ctx = granite_nle_init_from_file(model_path, p);
        if (!s->granite_nle_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_FUNASR
    if (s->backend == "funasr") {
        funasr_context_params p = funasr_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->funasr_ctx = funasr_init_from_file(model_path, p);
        if (!s->funasr_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_PARAFORMER
    if (s->backend == "paraformer") {
        paraformer_context_params p = paraformer_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.flash_attn = g_open_flash_attn_tls;
        s->paraformer_ctx = paraformer_init_from_file(model_path, p);
        if (!s->paraformer_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_SENSEVOICE
    if (s->backend == "sensevoice") {
        sensevoice_context_params p = sensevoice_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->sensevoice_ctx = sensevoice_init_from_file(model_path, p);
        if (!s->sensevoice_ctx) {
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
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
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
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
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
    if (s->backend == "vibevoice" || s->backend == "vibevoice-tts" || s->backend == "vibevoice-1.5b" ||
        s->backend == "vibevoice-tts-1.5b" || s->backend == "vibevoice-tts-base") {
        s->backend = "vibevoice";
        vibevoice_context_params p = vibevoice_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
        s->vibevoice_ctx = vibevoice_init_from_file(model_path, p);
        if (!s->vibevoice_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_KUGELAUDIO
    if (s->backend == "kugelaudio" || s->backend == "kugelaudio-tts") {
        s->backend = "kugelaudio";
        kugelaudio_context_params p = kugelaudio_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
        s->kugelaudio_ctx = kugelaudio_init_from_file(model_path, p);
        if (!s->kugelaudio_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_ZONOS
    if (s->backend == "zonos" || s->backend == "zonos-tts") {
        s->backend = "zonos";
        zonos_tts_params p = zonos_tts_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->zonos_ctx = zonos_tts_init_from_file(model_path, p);
        if (!s->zonos_ctx) {
            delete s;
            return nullptr;
        }
        // Auto-discover the DAC codec GGUF as a sibling of the transformer file.
        {
            std::string mp = model_path ? model_path : "";
            auto sep = mp.find_last_of("/\\");
            std::string dir = (sep == std::string::npos) ? std::string(".") : mp.substr(0, sep);
            for (const char* name : {"dac-44khz-f16.gguf", "dac-44khz.gguf", "dac_44khz.gguf"}) {
                std::string cp = dir + "/" + name;
                FILE* f = fopen(cp.c_str(), "rb");
                if (f) {
                    fclose(f);
                    zonos_tts_set_codec_path(s->zonos_ctx, cp.c_str());
                    break;
                }
            }
        }
        return s;
    }
#endif
#ifdef CA_HAVE_QWEN3_TTS
    if (s->backend == "qwen3-tts" || s->backend == "qwen3_tts" || s->backend == "qwen3tts" ||
        s->backend == "qwen3-tts-base" || s->backend == "qwen3-tts-1.7b-base" ||
        s->backend == "qwen3-tts-customvoice" || s->backend == "gwen-tts") {
        qwen3_tts_context_params p = qwen3_tts_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
        s->qwen3_tts_ctx = qwen3_tts_init_from_file(model_path, p);
        if (!s->qwen3_tts_ctx) {
            delete s;
            return nullptr;
        }
        // Codec must be loaded before synthesise. Caller does so via
        // `crispasr_session_set_codec_path` after open.
        return s;
    }
#endif
#ifdef CA_HAVE_GLMASR
    if (s->backend == "glm-asr" || s->backend == "glmasr" || s->backend == "glm" || s->backend == "glm_asr") {
        glm_asr_context_params p = glm_asr_context_default_params();
        p.n_threads = s->n_threads;
        s->glmasr_ctx = glm_asr_init_from_file(model_path, p);
        if (!s->glmasr_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_KYUTAI
    if (s->backend == "kyutai-stt" || s->backend == "kyutai" || s->backend == "moshi-stt") {
        kyutai_stt_context_params p = kyutai_stt_context_default_params();
        p.n_threads = s->n_threads;
        s->kyutai_ctx = kyutai_stt_init_from_file(model_path, p);
        if (!s->kyutai_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_FIRERED
    if (s->backend == "firered-asr" || s->backend == "firered") {
        firered_asr_context_params p = firered_asr_context_default_params();
        p.n_threads = s->n_threads;
        s->firered_ctx = firered_asr_init_from_file(model_path, p);
        if (!s->firered_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_MOONSHINE
    if (s->backend == "moonshine") {
        moonshine_init_params mp = {};
        mp.model_path = model_path;
        mp.tokenizer_path = nullptr;
        mp.n_threads = s->n_threads;
        mp.use_gpu = g_open_use_gpu_tls;
        s->moonshine_ctx = moonshine_init_with_params(mp);
        if (!s->moonshine_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_MOONSHINE_STREAMING
    if (s->backend == "moonshine-streaming") {
        moonshine_streaming_context_params p = moonshine_streaming_context_default_params();
        p.n_threads = s->n_threads;
        p.use_gpu = g_open_use_gpu_tls;
        s->moonshine_streaming_ctx = moonshine_streaming_init_from_file(model_path, p);
        if (!s->moonshine_streaming_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_GEMMA4_E2B
    if (s->backend == "gemma4-e2b") {
        gemma4_e2b_context_params p = gemma4_e2b_context_default_params();
        p.n_threads = s->n_threads;
        s->gemma4_e2b_ctx = gemma4_e2b_init_from_file(model_path, p);
        if (!s->gemma4_e2b_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_OMNIASR
    if (s->backend.rfind("omniasr", 0) == 0) {
        omniasr_context_params p = omniasr_context_default_params();
        p.n_threads = s->n_threads;
        s->omniasr_ctx = omniasr_init_from_file(model_path, p);
        if (!s->omniasr_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_ORPHEUS
    if (s->backend == "orpheus" || s->backend == "orpheus-tts") {
        s->backend = "orpheus";
        orpheus_context_params p = orpheus_context_default_params();
        p.use_gpu = g_open_use_gpu_tls;
        p.verbosity = g_open_verbosity_tls;
        p.flash_attn = g_open_flash_attn_tls;
        s->orpheus_ctx = orpheus_init_from_file(model_path, p);
        if (!s->orpheus_ctx) {
            delete s;
            return nullptr;
        }
        orpheus_set_n_threads(s->orpheus_ctx, s->n_threads);
        // SNAC codec must be loaded before synthesise. Caller does so via
        // `crispasr_session_set_codec_path` after open.
        return s;
    }
#endif
#ifdef CA_HAVE_KOKORO
    if (s->backend == "kokoro" || s->backend == "kokoro-tts") {
        s->backend = "kokoro";
        kokoro_context_params p = kokoro_context_default_params();
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
        s->kokoro_ctx = kokoro_init_from_file(model_path, p);
        if (!s->kokoro_ctx) {
            delete s;
            return nullptr;
        }
        kokoro_set_n_threads(s->kokoro_ctx, s->n_threads);
        // Voicepack must be loaded before synthesise. Caller does so via
        // `crispasr_session_set_voice` after open.
        return s;
    }
#endif
#ifdef CA_HAVE_CHATTERBOX
    if (s->backend == "chatterbox" || s->backend == "chatterbox-tts" || s->backend == "kartoffelbox" ||
        s->backend == "chatterbox_turbo") {
        s->backend = "chatterbox";
        chatterbox_context_params p = chatterbox_context_default_params();
        p.n_threads = s->n_threads;
        // Chatterbox's verbosity default is 1 (chatty); honour the user
        // override when set, otherwise keep the upstream "tell me what
        // you're doing" log level so first-time bake users see progress.
        p.verbosity = g_open_verbosity_tls > 0 ? g_open_verbosity_tls : 1;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
        s->chatterbox_ctx = chatterbox_init_from_file(model_path, p);
        if (!s->chatterbox_ctx) {
            delete s;
            return nullptr;
        }
        // S3Gen GGUF must be loaded via crispasr_session_set_s3gen_path
        return s;
    }
#endif
#ifdef CA_HAVE_TADA
    if (s->backend == "tada" || s->backend == "tada-tts" || s->backend == "tada-3b" || s->backend == "tada-3b-ml") {
        s->backend = "tada";
        tada_context_params p = tada_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls > 0 ? g_open_verbosity_tls : 1;
        p.use_gpu = g_open_use_gpu_tls;
        s->tada_ctx = tada_init_from_file(model_path, p);
        if (!s->tada_ctx) {
            fprintf(stderr, "crispasr: failed to init tada from '%s'\n", model_path);
            delete s;
            return nullptr;
        }
        // TADA codec must be loaded before synthesise. Caller does so via
        // `crispasr_session_set_codec_path` after open.
        return s;
    }
#endif
#ifdef CA_HAVE_OUTETTS
    if (s->backend == "outetts" || s->backend == "oute-tts" || s->backend == "oute_tts") {
        s->backend = "outetts";
        outetts_context_params p = outetts_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
        p.temperature = (g_open_temperature_tls > 0.0f) ? g_open_temperature_tls : 0.4f;
        p.seed = g_open_seed_tls;
        s->outetts_ctx = outetts_init_from_file(model_path, p);
        if (!s->outetts_ctx) {
            delete s;
            return nullptr;
        }
        // WavTokenizer codec set via crispasr_session_set_companion_path
        // or auto-discovered as sibling wavtokenizer-decoder-f16.gguf
        {
            std::string dir(model_path);
            auto sep = dir.find_last_of("/\\");
            if (sep != std::string::npos)
                dir.resize(sep);
            else
                dir = ".";
            const char* names[] = {"wavtokenizer-decoder-f16.gguf", "wavtokenizer-decoder.gguf"};
            for (const char* n : names) {
                std::string cp = dir + "/" + n;
                FILE* f = fopen(cp.c_str(), "rb");
                if (f) {
                    fclose(f);
                    outetts_set_codec_path(s->outetts_ctx, cp.c_str());
                    break;
                }
            }
        }
        return s;
    }
#endif
#ifdef CA_HAVE_CSM
    if (s->backend == "csm" || s->backend == "csm-tts" || s->backend == "sesame" || s->backend == "sesame-csm") {
        s->backend = "csm";
        csm_tts_context_params p = csm_tts_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->csm_tts_ctx = csm_tts_init_from_file(model_path, p);
        if (!s->csm_tts_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_DIA
    if (s->backend == "dia" || s->backend == "dia-tts" || s->backend == "dia-1.6b" || s->backend == "dia_tts") {
        s->backend = "dia";
        dia_tts_context_params p = dia_tts_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->dia_tts_ctx = dia_tts_init_from_file(model_path, p);
        if (!s->dia_tts_ctx) {
            delete s;
            return nullptr;
        }
        // Resolve the DAC codec — the common case is the codec GGUF sitting next
        // to the model. (The CLI additionally resolves the registry companion;
        // callers wanting a custom codec can extend with a set_codec_path ABI.)
        {
            std::string mp = model_path ? model_path : "";
            auto sep = mp.find_last_of("/\\");
            std::string dir = (sep == std::string::npos) ? std::string(".") : mp.substr(0, sep);
            for (const char* name : {"dac-44khz-f16.gguf", "dac-44khz.gguf", "dac_44khz.gguf", "dia-dac-44khz.gguf"}) {
                std::string cp = dir + "/" + name;
                FILE* f = fopen(cp.c_str(), "rb");
                if (f) {
                    fclose(f);
                    dia_tts_set_codec_path(s->dia_tts_ctx, cp.c_str());
                    break;
                }
            }
        }
        return s;
    }
#endif
#ifdef CA_HAVE_POCKET
    if (s->backend == "pocket-tts" || s->backend == "pocket_tts" || s->backend == "pocket") {
        s->backend = "pocket-tts";
        pocket_tts_context_params p = pocket_tts_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->pocket_tts_ctx = pocket_tts_init_from_file(model_path, p);
        if (!s->pocket_tts_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_FASTPITCH
    if (s->backend == "fastpitch" || s->backend == "fastpitch-tts" || s->backend == "fastpitch_tts") {
        s->backend = "fastpitch";
        fastpitch_tts_params p = fastpitch_tts_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->fastpitch_ctx = fastpitch_tts_init_from_file(model_path, p);
        if (!s->fastpitch_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_PARLER
    if (s->backend == "parler-tts" || s->backend == "parler_tts" || s->backend == "parler") {
        s->backend = "parler-tts";
        parler_tts_context_params p = parler_tts_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->parler_ctx = parler_tts_init_from_file(model_path, p);
        if (!s->parler_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_SPEECHT5
    if (s->backend == "speecht5" || s->backend == "speecht5-tts" || s->backend == "speecht5_tts") {
        s->backend = "speecht5";
        speecht5_tts_params p = speecht5_tts_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->speecht5_ctx = speecht5_tts_init(model_path, p);
        if (!s->speecht5_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_BARK
    if (s->backend == "bark" || s->backend == "bark-tts" || s->backend == "bark_tts") {
        s->backend = "bark";
        bark_context_params p = bark_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->bark_ctx = bark_init_from_file(model_path, p);
        if (!s->bark_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_PARLER_TTS
    if (s->backend == "parler-tts" || s->backend == "parler_tts" || s->backend == "parler" ||
        s->backend == "parlertts") {
        s->backend = "parler-tts";
        parler_tts_context_params p = parler_tts_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
        // Parler requires stochastic sampling (temp=1.0); greedy is degenerate
        p.temperature = (g_open_temperature_tls > 0.0f) ? g_open_temperature_tls : 1.0f;
        p.seed = g_open_seed_tls;
        s->parler_tts_ctx = parler_tts_init_from_file(model_path, p);
        if (!s->parler_tts_ctx) {
            delete s;
            return nullptr;
        }
        // Set default voice description (user overrides via set_instruct)
        s->parler_description = "A female speaker delivers her words at a moderate pace "
                                "with a clear and natural tone in a quiet environment.";
        parler_tts_set_description(s->parler_tts_ctx, s->parler_description.c_str());
        return s;
    }
#endif
#ifdef CA_HAVE_VOXCPM2
    if (s->backend == "voxcpm2-tts" || s->backend == "voxcpm2" || s->backend == "voxcpm2_tts") {
        s->backend = "voxcpm2-tts";
        voxcpm2_context_params p = voxcpm2_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
        s->voxcpm2_ctx = voxcpm2_init_from_file(model_path, p);
        if (!s->voxcpm2_ctx) {
            delete s;
            return nullptr;
        }
        // Zero-shot synthesis works immediately; voice cloning kicks in
        // when a 16 kHz reference is supplied via
        // crispasr_session_set_voice (decoded from a WAV there).
        return s;
    }
#endif
#ifdef CA_HAVE_COSYVOICE3
    if (s->backend == "cosyvoice3-tts" || s->backend == "cosyvoice3" || s->backend == "cosyvoice3-llm") {
        s->backend = "cosyvoice3-tts";
        cosyvoice3_tts_context_params p = cosyvoice3_tts_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.flash_attn = g_open_flash_attn_tls;
        // CV3 greedy decode (temperature 0) falls into a documented
        // silent_tokens loop within ~5 AR steps and emits silence; the RAS
        // sampler needs temperature > 0. Mirror the CLI's 0->0.8 default.
        p.temperature = 0.8f;
        s->cosyvoice3_ctx = cosyvoice3_tts_init_from_file(model_path, p);
        if (!s->cosyvoice3_ctx) {
            delete s;
            return nullptr;
        }
        // Companion GGUFs auto-discover as siblings of the LLM (or via
        // COSYVOICE3_*_PATH env vars). flow/hift/voices are required;
        // s3tok+campplus are optional (enable native arbitrary-WAV cloning).
        std::string cv3_dir = model_path;
        {
            auto sep = cv3_dir.find_last_of("/\\");
            cv3_dir = (sep == std::string::npos) ? std::string(".") : cv3_dir.substr(0, sep);
        }
        auto cv3_exists = [](const std::string& q) {
            FILE* f = std::fopen(q.c_str(), "rb");
            if (f) {
                std::fclose(f);
                return true;
            }
            return false;
        };
        auto cv3_sib = [&](const char* env, std::initializer_list<const char*> names) -> std::string {
            if (env) {
                const char* e = std::getenv(env);
                if (e && *e)
                    return e;
            }
            for (const char* n : names) {
                std::string q = cv3_dir + "/" + n;
                if (cv3_exists(q))
                    return q;
            }
            return "";
        };
        std::string cv3_flow = cv3_sib(
            "COSYVOICE3_FLOW_PATH", {"cosyvoice3-flow-f16.gguf", "cosyvoice3-flow-q8_0.gguf", "cosyvoice3-flow.gguf"});
        std::string cv3_hift = cv3_sib("COSYVOICE3_HIFT_PATH", {"cosyvoice3-hift-f16.gguf", "cosyvoice3-hift.gguf"});
        std::string cv3_voices = cv3_sib("COSYVOICE3_VOICES_PATH", {"cosyvoice3-voices.gguf", "voices.gguf"});
        std::string cv3_camp =
            cv3_sib("COSYVOICE3_CAMPPLUS_PATH", {"cosyvoice3-campplus-f16.gguf", "cosyvoice3-campplus.gguf"});
        std::string cv3_s3tok =
            cv3_sib(nullptr, {"cosyvoice3-s3tok-f16.gguf", "cosyvoice3-s3tok-q4_k.gguf", "cosyvoice3-s3tok.gguf"});
        if (cv3_flow.empty() || cosyvoice3_tts_init_flow_from_file(s->cosyvoice3_ctx, cv3_flow.c_str()) != 0 ||
            cv3_hift.empty() || cosyvoice3_tts_init_hift_from_file(s->cosyvoice3_ctx, cv3_hift.c_str()) != 0 ||
            cv3_voices.empty() || cosyvoice3_tts_init_voices_from_file(s->cosyvoice3_ctx, cv3_voices.c_str()) != 0) {
            cosyvoice3_tts_free(s->cosyvoice3_ctx);
            delete s;
            return nullptr;
        }
        if (!cv3_camp.empty())
            cosyvoice3_tts_init_campplus_from_file(s->cosyvoice3_ctx, cv3_camp.c_str());
        if (!cv3_s3tok.empty())
            cosyvoice3_tts_init_s3tok_from_file(s->cosyvoice3_ctx, cv3_s3tok.c_str());
        return s;
    }
#endif
#ifdef CA_HAVE_INDEXTTS
    if (s->backend == "indextts" || s->backend == "indextts-1.5") {
        s->backend = "indextts";
        indextts_context_params p = indextts_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->indextts_ctx = indextts_init_from_file(model_path, p);
        if (!s->indextts_ctx) {
            delete s;
            return nullptr;
        }
        // BigVGAN vocoder (indextts-bigvgan) MUST be loaded before the
        // first synthesize — caller does so via
        // crispasr_session_set_codec_path. The cloning reference (24 kHz)
        // is supplied via crispasr_session_set_voice.
        return s;
    }
#endif
#ifdef CA_HAVE_F5TTS
    if (s->backend == "f5-tts" || s->backend == "f5tts" || s->backend == "f5") {
        s->backend = "f5-tts";
        f5_tts_params p = f5_tts_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        p.seed = 42;
        s->f5tts_ctx = f5_tts_init_from_file(model_path, p);
        if (!s->f5tts_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_PIPER
    if (s->backend == "piper" || s->backend == "piper-tts") {
        s->backend = "piper";
        piper_tts_params p = piper_tts_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->piper_ctx = piper_tts_init_from_file(model_path, p);
        if (!s->piper_ctx) {
            delete s;
            return nullptr;
        }
        // Single-file VITS: the phoneme-id map and the default espeak
        // voice are baked into the GGUF, so it synthesises immediately
        // after open — no codec / voice companion required.
        return s;
    }
#endif
#ifdef CA_HAVE_MELOTTS
    if (s->backend == "melotts" || s->backend == "melo-tts" || s->backend == "melo") {
        s->backend = "melotts";
        melotts_params p = melotts_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->melotts_ctx = melotts_init_from_file(model_path, p);
        if (!s->melotts_ctx) {
            delete s;
            return nullptr;
        }
        // Try to load BERT companion from MELOTTS_BERT env var or
        // bert-base-uncased.gguf next to the model.
        {
            const char* bert_env = std::getenv("MELOTTS_BERT");
            std::string bert_path;
            if (bert_env && *bert_env) {
                bert_path = bert_env;
            } else {
                std::string mp = model_path;
                size_t sep = mp.find_last_of("/\\");
                std::string dir = (sep != std::string::npos) ? mp.substr(0, sep + 1) : "./";
                std::string candidate = dir + "bert-base-uncased.gguf";
                FILE* test = fopen(candidate.c_str(), "rb");
                if (test) {
                    fclose(test);
                    bert_path = candidate;
                }
            }
            if (!bert_path.empty())
                melotts_load_bert(s->melotts_ctx, bert_path.c_str());
        }
        return s;
    }
#endif
#ifdef CA_HAVE_M2M100
    if (s->backend == "m2m100" || s->backend == "m2m-100" || s->backend == "translate" ||
        s->backend == "m2m100-wmt21") {
        // WMT21 Dense (24-wide) shares m2m100's runtime — m2m100.cpp
        // supports facebook/m2m100_418M, m2m100_1.2B AND wmt21-dense-24-wide,
        // and WMT21 GGUFs carry the `m2m100` architecture. The catalogue
        // tags them backend="m2m100-wmt21" (direction picked from the
        // model's prefix at translate time), so accept that string and
        // normalise to the shared m2m100 context.
        s->backend = "m2m100";
        m2m100_context_params p = m2m100_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = 1;
        s->m2m100_ctx = m2m100_init_from_file(model_path, p);
        if (!s->m2m100_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_T5_TRANSLATE
    if (s->backend == "madlad" || s->backend == "t5") {
        // MADLAD-400 (google/madlad400-3b-mt) is a T5 translator — GGUFs
        // carry arch="t5". Target language is selected via a "<2xx> " tag
        // prepended to the source at translate time (see
        // crispasr_session_translate_text).
        s->backend = "madlad";
        t5_translate_context_params p = t5_translate_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->t5_translate_ctx = t5_translate_init_from_file(model_path, p);
        if (!s->t5_translate_ctx) {
            delete s;
            return nullptr;
        }
        return s;
    }
#endif
#ifdef CA_HAVE_MIMO_ASR
    if (s->backend == "mimo-asr" || s->backend == "mimo_asr" || s->backend == "mimo") {
        s->backend = "mimo-asr";
        mimo_asr_context_params p = mimo_asr_context_default_params();
        p.n_threads = s->n_threads;
        s->mimo_asr_ctx = mimo_asr_init_from_file(model_path, p);
        if (!s->mimo_asr_ctx) {
            delete s;
            return nullptr;
        }
        // mimo_tokenizer companion must be loaded before transcribe.
        // Caller does so via `crispasr_session_set_codec_path` after open
        // (we route the tokenizer through that setter — same shape as
        // qwen3-tts/orpheus's codec companion).
        return s;
    }
#endif
#ifdef CA_HAVE_MOSS_AUDIO
    if (s->backend == "moss-audio" || s->backend == "moss_audio" || s->backend == "mossaudio") {
        s->backend = "moss-audio";
        moss_audio_context_params p = moss_audio_context_default_params();
        p.n_threads = s->n_threads;
        p.verbosity = g_open_verbosity_tls;
        p.use_gpu = g_open_use_gpu_tls;
        s->moss_audio_ctx = moss_audio_init_from_file(model_path, p);
        if (!s->moss_audio_ctx) {
            delete s;
            return nullptr;
        }
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

// ─────────────────────────────────────────────────────────────────
// Open with explicit runtime params (CrispASR 0.6.1, extended 0.6.2).
//
// Layout-stable struct via a leading version int — host languages
// can extend by reading the version field and skipping unknown
// trailing bytes. v2 (0.6.2) adds `flash_attn` and `n_gpu_layers`
// in the v1 reserved padding; v1 callers see those fields as zero
// (which is interpreted as "use defaults" — flash_attn defaults
// true, n_gpu_layers defaults -1).
//
// `backend` may be "" / NULL to delegate to GGUF arch detection
// (same path as `crispasr_session_open`).
// ─────────────────────────────────────────────────────────────────
struct crispasr_open_params_v1 {
    int abi_version; // = 1 or 2
    int n_threads;
    int use_gpu;   // 0 = CPU only, non-zero = GPU when available
    int verbosity; // 0 = silent, 1+ = chatty
    // ── v2 (0.6.2) additions ───────────────────────────────────────
    // Set abi_version >= 2 to opt into these fields. v1 callers
    // get the historical defaults.
    int flash_attn;   // 0 = off, non-zero = on (default on)
    int n_gpu_layers; // -1 = max, 0 = CPU-only LLM, >0 = bound
    int reserved[6];  // future-compat padding (was 8 in v1; -2 here)
};

CA_EXPORT crispasr_session* crispasr_session_open_with_params(const char* model_path, const char* backend_name,
                                                              const crispasr_open_params_v1* params) {
    if (!model_path)
        return nullptr;

    // Default values mirror the pre-0.6.1 behaviour so a NULL params
    // (or one whose version we don't recognise yet) lands you in the
    // same place crispasr_session_open does.
    int n_threads = 4;
    bool use_gpu = true;
    int verbosity = 0;
    bool flash_attn = true;
    int n_gpu_layers = -1;
    if (params && params->abi_version >= 1) {
        n_threads = params->n_threads > 0 ? params->n_threads : 4;
        use_gpu = params->use_gpu != 0;
        verbosity = params->verbosity;
        if (params->abi_version >= 2) {
            // v2 fields: 0 in flash_attn means "explicitly off"; we
            // can't distinguish "v1 caller, struct memset to 0" from
            // "v2 caller, asked for off". The version gate above is
            // the disambiguator — only read these when v2.
            flash_attn = params->flash_attn != 0;
            n_gpu_layers = params->n_gpu_layers;
        }
    }

    // Stash the runtime overrides for the duration of the open call.
    // Reset on the way out so subsequent calls that don't pass params
    // see the static defaults again. RAII would be tidier but the
    // function has multiple early-return paths through delete-and-fail
    // and this scoped pair is the simplest correct shape.
    const bool prev_use_gpu = g_open_use_gpu_tls;
    const int prev_verbosity = g_open_verbosity_tls;
    const bool prev_flash_attn = g_open_flash_attn_tls;
    const int prev_n_gpu_layers = g_open_n_gpu_layers_tls;
    g_open_use_gpu_tls = use_gpu;
    g_open_verbosity_tls = verbosity;
    g_open_flash_attn_tls = flash_attn;
    g_open_n_gpu_layers_tls = n_gpu_layers;

    crispasr_session* s = nullptr;
    if (backend_name && backend_name[0] != '\0') {
        s = crispasr_session_open_explicit(model_path, backend_name, n_threads);
    } else {
        // Explicit-detection path matches `crispasr_session_open`.
        char detected[64] = {0};
        if (crispasr_detect_backend_from_gguf(model_path, detected, (int)sizeof(detected)) > 0) {
            s = crispasr_session_open_explicit(model_path, detected, n_threads);
        } else {
            // Whisper GGML magic check (legacy non-GGUF format).
            FILE* f = fopen(model_path, "rb");
            if (f) {
                char magic[4] = {0};
                if (fread(magic, 1, 4, f) == 4 && (memcmp(magic, "lmgg", 4) == 0 || memcmp(magic, "ggjt", 4) == 0)) {
                    s = crispasr_session_open_explicit(model_path, "whisper", n_threads);
                }
                fclose(f);
            }
        }
    }

    g_open_use_gpu_tls = prev_use_gpu;
    g_open_verbosity_tls = prev_verbosity;
    g_open_flash_attn_tls = prev_flash_attn;
    g_open_n_gpu_layers_tls = prev_n_gpu_layers;
    return s;
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
#ifdef CA_HAVE_LFM2_AUDIO
    list += ",lfm2-audio";
#endif
#ifdef CA_HAVE_MINI_OMNI2
    list += ",mini-omni2";
#endif
#ifdef CA_HAVE_QWEN3
    list += ",qwen3";
#endif
#ifdef CA_HAVE_COHERE
    list += ",cohere";
#endif
#ifdef CA_HAVE_GRANITE
    list += ",granite,granite-4.1,granite-4.1-plus,granite-4.1-nar";
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
    list += ",vibevoice,vibevoice-tts,vibevoice-1.5b";
#endif
#ifdef CA_HAVE_KUGELAUDIO
    list += ",kugelaudio";
#endif
#ifdef CA_HAVE_ZONOS
    list += ",zonos";
#endif
#ifdef CA_HAVE_QWEN3_TTS
    list += ",qwen3-tts";
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
#ifdef CA_HAVE_MOONSHINE_STREAMING
    list += ",moonshine-streaming";
#endif
#ifdef CA_HAVE_GEMMA4_E2B
    list += ",gemma4-e2b";
#endif
#ifdef CA_HAVE_OMNIASR
    // The session-open dispatcher uses a prefix match
    // (`backend.rfind("omniasr", 0) == 0`) so every omniasr-* variant
    // routes through the same handler. Advertise the variants
    // CrisperWeaver's catalogue uses so the front-door
    // available_backends check (which is strict string equality)
    // accepts them.
    list += ",omniasr,omniasr-300m,omniasr-llm,omniasr-llm-unlimited";
#endif
#ifdef CA_HAVE_ORPHEUS
    list += ",orpheus";
#endif
#ifdef CA_HAVE_KOKORO
    list += ",kokoro";
#endif
#ifdef CA_HAVE_CHATTERBOX
    list += ",chatterbox";
#endif
#ifdef CA_HAVE_TADA
    list += ",tada";
#endif
#ifdef CA_HAVE_OUTETTS
    list += ",outetts";
#endif
#ifdef CA_HAVE_CSM
    list += ",csm";
#endif
#ifdef CA_HAVE_DIA
    list += ",dia";
#endif
#ifdef CA_HAVE_POCKET
    list += ",pocket-tts";
#endif
#ifdef CA_HAVE_FASTPITCH
    list += ",fastpitch";
#endif
#ifdef CA_HAVE_PARLER
    list += ",parler-tts";
#endif
#ifdef CA_HAVE_SPEECHT5
    list += ",speecht5";
#endif
#ifdef CA_HAVE_BARK
    list += ",bark";
#endif
#ifdef CA_HAVE_PARLER_TTS
    list += ",parler-tts";
#endif
#ifdef CA_HAVE_VOXCPM2
    list += ",voxcpm2-tts";
#endif
#ifdef CA_HAVE_COSYVOICE3
    list += ",cosyvoice3-tts";
#endif
#ifdef CA_HAVE_INDEXTTS
    list += ",indextts";
#endif
#ifdef CA_HAVE_F5TTS
    list += ",f5-tts";
#endif
#ifdef CA_HAVE_PIPER
    list += ",piper";
#endif
#ifdef CA_HAVE_MELOTTS
    list += ",melotts";
#endif
#ifdef CA_HAVE_M2M100
    // m2m100-wmt21 routes through the same m2m100 engine (WMT21 Dense
    // support) — advertise it so CrisperWeaver's strict front-door check
    // accepts ModelDefinitions tagged backend='m2m100-wmt21'.
    list += ",m2m100,m2m100-wmt21";
#endif
#ifdef CA_HAVE_T5_TRANSLATE
    list += ",madlad";
#endif
#ifdef CA_HAVE_MIMO_ASR
    list += ",mimo-asr";
#endif
#ifdef CA_HAVE_MOSS_AUDIO
    list += ",moss-audio";
#endif
#ifdef CA_HAVE_QWEN3
    // mega-asr is a Qwen3-ASR variant (LoRA merged offline) — dispatch
    // through the qwen3 path in session_open_explicit / transcribe.
    // Advertised separately so CrisperWeaver's strict front-door check
    // accepts ModelDefinitions tagged backend='mega-asr'.
    list += ",mega-asr";
#endif
#ifdef CA_HAVE_FUNASR
    list += ",funasr";
#endif
#ifdef CA_HAVE_PARAFORMER
    list += ",paraformer";
#endif
#ifdef CA_HAVE_SENSEVOICE
    list += ",sensevoice";
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
                                                   int n_samples, const std::string& language,
                                                   const std::string& ask = std::string(), int beam_size = 1) {
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
    //
    // Suffix branches on whether the user set a Q&A prompt via
    // `crispasr_session_set_ask`. With it: `[/INST]<question>` so the
    // audio-LLM answers free-form ("Summarize the speaker's tone",
    // "What did they say about Z"). Without it: the historical
    // `[/INST]lang:<X>[TRANSCRIBE]` template that asks for a verbatim
    // transcript in the target language.
    const char* prefix = "<s>[INST][BEGIN_AUDIO]";
    const std::string suffix =
        !ask.empty() ? std::string("[/INST]") + ask
                     : std::string("[/INST]lang:") + (language.empty() ? "en" : language) + "[TRANSCRIBE]";

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

    // 8. Decode: beam search (PLAN §90) or greedy, then detokenize.
    // U+2581 (Tekken word-leading marker) → ASCII space, shared by both paths.
    auto decode_piece = [&](int id, float p, std::string& out_text, std::vector<ca_token_record>& out_toks) {
        int tok_len = 0;
        const uint8_t* tok_bytes = ops.token_text(ctx, id, &tok_len);
        std::string piece;
        if (tok_bytes && tok_len > 0)
            piece.assign(reinterpret_cast<const char*>(tok_bytes), (size_t)tok_len);
        std::string decoded;
        decoded.reserve(piece.size());
        for (size_t ci = 0; ci < piece.size(); ci++) {
            if ((unsigned char)piece[ci] == 0xE2 && ci + 2 < piece.size() && (unsigned char)piece[ci + 1] == 0x96 &&
                (unsigned char)piece[ci + 2] == 0x81) {
                decoded += ' ';
                ci += 2;
            } else {
                decoded += piece[ci];
            }
        }
        out_text += decoded;
        ca_token_record tk;
        tk.text = std::move(decoded);
        tk.t0 = -1;
        tk.t1 = -1;
        tk.p = p;
        out_toks.push_back(std::move(tk));
    };

    std::string generated;
    generated.reserve(512);
    std::vector<ca_token_record> toks;
    toks.reserve(64);

    if (beam_size > 1) {
        // PLAN §90: session beam_size → voxtral beam decode.
        const float* prefill_last = logits + (size_t)(out_n_tok - 1) * (size_t)out_vocab;
        auto replay = [&](Ctx* c, const int32_t* tok_ids, int n, int pl) -> float* {
            float* emb = ops.embed_tokens(c, tok_ids, n);
            if (!emb)
                return nullptr;
            float* lg = ops.run_llm_kv(c, emb, n, pl, nullptr, nullptr);
            std::free(emb);
            return lg;
        };
        core_beam_decode::Config bcfg;
        bcfg.max_new_tokens = kMaxNewTokens;
        bcfg.eos_id = ops.eos_id;
        bcfg.vocab_size = out_vocab;
        bcfg.beam_size = beam_size;
        bcfg.prompt_len = total_tokens;
        auto br = core_beam_decode::run_with_probs(ctx, prefill_last, replay, bcfg);
        ops.kv_reset(ctx);
        std::free(logits);
        for (size_t i = 0; i < br.tokens.size(); i++) {
            if (br.tokens[i] == ops.eos_id)
                break;
            decode_piece(br.tokens[i], i < br.probs.size() ? br.probs[i] : -1.0f, generated, toks);
        }
    } else {
        int n_past = total_tokens;
        for (int step = 0; step < kMaxNewTokens; ++step) {
            const float* last = logits + (size_t)(out_n_tok - 1) * (size_t)out_vocab;
            int best = 0;
            float best_score = last[0];
            for (int i = 1; i < out_vocab; ++i) {
                if (last[i] > best_score) {
                    best_score = last[i];
                    best = i;
                }
            }
            float sum_exp = 0.f;
            for (int i = 0; i < out_vocab; ++i)
                sum_exp += expf(last[i] - best_score);
            const float picked_p = (sum_exp > 0.f) ? (1.0f / sum_exp) : 0.0f;
            std::free(logits);
            logits = nullptr;

            if (best == ops.eos_id)
                break;

            decode_piece(best, picked_p, generated, toks);

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
    }

    crispasr_session_seg seg;
    seg.text = std::move(generated);
    seg.t0 = 0;
    seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
    seg.words = emit_words_from_tokens(toks);
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
// Internal single-pass transcribe (used by best-of-N wrapper below).
static crispasr_session_result* transcribe_single(crispasr_session* s, const float* pcm, int n_samples,
                                                  const char* language);

// Applies the session's resident --punc-model (if any) to a result in place.
// Defined further down next to crispasr_session_set_punc_model.
static void apply_session_punc_model(crispasr_session* s, crispasr_session_result* r);

CA_EXPORT crispasr_session_result* crispasr_session_transcribe_lang(crispasr_session* s, const float* pcm,
                                                                    int n_samples, const char* language) {
    if (!s || !pcm || n_samples <= 0)
        return nullptr;

    // Best-of-N: run N independent transcriptions and keep the one with the
    // highest average per-token confidence. Whisper handles best_of internally
    // via greedy.best_of, so we only loop externally for non-whisper backends.
    const int n_runs = (s->best_of > 1 && s->backend != "whisper") ? s->best_of : 1;
    if (n_runs <= 1) {
        crispasr_session_result* r = transcribe_single(s, pcm, n_samples, language);
        apply_session_punc_model(s, r);
        return r;
    }

    crispasr_session_result* best = nullptr;
    double best_avg_p = -1.0;
    for (int run = 0; run < n_runs; run++) {
        crispasr_session_result* candidate = transcribe_single(s, pcm, n_samples, language);
        if (!candidate)
            continue;
        // Compute average per-word confidence
        double sum_p = 0.0;
        int n_words = 0;
        for (auto& seg : candidate->segments) {
            for (auto& w : seg.words) {
                sum_p += w.p;
                n_words++;
            }
        }
        double avg_p = n_words > 0 ? sum_p / n_words : 0.0;
        if (!best || avg_p > best_avg_p) {
            if (best)
                delete best;
            best = candidate;
            best_avg_p = avg_p;
        } else {
            delete candidate;
        }
    }
    apply_session_punc_model(s, best);
    return best;
}

static crispasr_session_result* transcribe_single(crispasr_session* s, const float* pcm, int n_samples,
                                                  const char* language) {
    const std::string lang = (language && *language) ? language : "en";
    const bool lang_set = (language && *language);

    // §5.26.2 — Hotword injection for LLM backends. Temporarily prepend
    // the hotword phrasing to s->ask so every LLM dispatch path picks it
    // up through the existing ask-prompt injection. Parakeet CTC/TDT
    // hotwords are applied directly via parakeet_set_hotwords() in
    // crispasr_session_set_hotwords() — no ask-prompt injection needed.
    std::string saved_ask;
    if (!s->hotwords.empty()) {
        saved_ask = s->ask;
        const std::string hw_hint = "The following words may appear in the audio: " + s->hotwords + ". ";
        s->ask = s->ask.empty() ? hw_hint : hw_hint + s->ask;
    }
    // Scope guard: restore original ask on all exit paths.
    struct AskGuard {
        crispasr_session* s;
        std::string* saved;
        bool active;
        ~AskGuard() {
            if (active)
                s->ask = std::move(*saved);
        }
    } ask_guard{s, &saved_ask, !s->hotwords.empty()};

    auto* r = new crispasr_session_result();
    r->backend = s->backend;

    if (s->backend == "whisper" && s->whisper_ctx) {
        // Beam search vs greedy. The session API's sticky `beam_size`
        // selects the strategy: > 1 → beam search with that width;
        // otherwise stay greedy and let `best_of` drive sampling
        // breadth (best_of and beam_size are alternative knobs in
        // upstream whisper.cpp — beam search uses `beam_search.beam_size`,
        // greedy uses `greedy.best_of`).
        // GBNF grammar-constrained sampling requires beam search per
        // whisper.cpp — fall back to beam=5 when the user enabled
        // grammar but left beam_size at its default 1. Otherwise use
        // beam search only when the user explicitly asked for it.
        const bool use_beam = s->beam_size > 1 || s->grammar_active;
        whisper_full_params wparams =
            whisper_full_default_params(use_beam ? CRISPASR_SAMPLING_BEAM_SEARCH : CRISPASR_SAMPLING_GREEDY);
        wparams.print_progress = false;
        wparams.print_realtime = false;
        wparams.print_timestamps = false;
        wparams.print_special = false;
        wparams.n_threads = s->n_threads;
        if (use_beam) {
            // Honour the user's explicit beam_size when set; otherwise
            // pick a sensible default (5) so grammar-constrained
            // sampling has enough beam width to be useful.
            wparams.beam_search.beam_size = s->beam_size > 1 ? s->beam_size : 5;
        } else if (s->best_of > 1) {
            // Best-of-N for whisper greedy sampling.
            wparams.greedy.best_of = s->best_of;
        }
        // Per-call language hint wins; sticky source_language is the
        // fallback (PLAN #59 unblock).
        if (lang_set)
            wparams.language = lang.c_str();
        else if (!s->source_language.empty())
            wparams.language = s->source_language.c_str();
        // Sticky --translate toggle (PLAN #59 unblock); whisper's
        // wparams.translate also activates the EN target language.
        if (s->translate)
            wparams.translate = true;
        // Decoder-fallback thresholds — write the sticky session
        // values into wparams on every dispatch so a slider tweak
        // takes effect on the next transcribe. Defaults match
        // whisper_full_default_params, so a user who never touches
        // the AdvancedOptions UI sees identical behaviour to the
        // stock library.
        wparams.entropy_thold = s->entropy_thold;
        wparams.logprob_thold = s->logprob_thold;
        wparams.no_speech_thold = s->no_speech_thold;
        wparams.temperature_inc = s->temperature_inc;
        // Alt-token capture (greedy decode only). 0 = off. The whisper
        // backend writes top-N runners-up onto each chosen token so
        // session_result_word_alt_* can surface them for ambiguous
        // word tap-to-pick UIs.
        wparams.alt_n = s->alt_n;
        // We need token-level data to build the per-word records below
        // (whisper otherwise reports only segment text). Token
        // timestamps are cheap once whisper_full has the tokens
        // resident; turn them on unconditionally on the session path
        // so word-level UIs work the same way they do for parakeet /
        // canary.
        wparams.token_timestamps = true;
        // Whisper text-suppression + prompt-carry extras. All three
        // map directly onto wparams; an empty regex passes nullptr
        // (whisper's "no suppression" sentinel) instead of an empty
        // string so wparams.suppress_regex doesn't end up pointing
        // at a heap blob with zero length.
        wparams.suppress_nst = s->whisper_suppress_nst;
        wparams.carry_initial_prompt = s->whisper_carry_initial_prompt;
        wparams.suppress_regex = s->whisper_suppress_regex.empty() ? nullptr : s->whisper_suppress_regex.c_str();
        // GBNF grammar-constrained sampling (whisper-only). The
        // `grammar_rules_ptrs` vector and the parsed rules it points
        // into both live on the session struct so they outlive the
        // whisper_full call.
        if (s->grammar_active) {
            wparams.grammar_rules = s->grammar_rules_ptrs.data();
            wparams.n_grammar_rules = s->grammar_rules_ptrs.size();
            wparams.i_start_rule = s->grammar_root_rule_id;
            wparams.grammar_penalty = s->grammar_penalty;
        }

        // Set progress callback — writes to the module-level atomic so
        // Dart can poll via crispasr_get_progress().
        g_progress.store(0, std::memory_order_relaxed);
        static auto progress_cb = [](struct whisper_context*, struct whisper_state*, int progress, void*) {
            g_progress.store(progress, std::memory_order_relaxed);
        };
        wparams.progress_callback = progress_cb;
        wparams.progress_callback_user_data = nullptr;

        if (whisper_full(s->whisper_ctx, wparams, pcm, n_samples) != 0) {
            g_progress.store(-1, std::memory_order_relaxed);
            delete r;
            return nullptr;
        }
        g_progress.store(-1, std::memory_order_relaxed);
        const int n = whisper_full_n_segments(s->whisper_ctx);
        for (int i = 0; i < n; ++i) {
            crispasr_session_seg seg;
            const char* t = whisper_full_get_segment_text(s->whisper_ctx, i);
            if (t)
                seg.text = t;
            seg.t0 = whisper_full_get_segment_t0(s->whisper_ctx, i);
            seg.t1 = whisper_full_get_segment_t1(s->whisper_ctx, i);

            // Convert whisper's per-token output into the unified
            // ca_token_record shape and run it through
            // emit_words_from_tokens — same grouping logic the
            // parakeet/canary paths use. Special / EOT / timestamp
            // tokens are filtered so they don't appear as garbage
            // words. When alt_n > 0, each content token also carries
            // its top-N runner-up candidates which flow through to
            // word.alts (attached to the word's first content token).
            const int n_tok = whisper_full_n_tokens(s->whisper_ctx, i);
            std::vector<ca_token_record> toks;
            toks.reserve((size_t)std::max(0, n_tok));
            for (int j = 0; j < n_tok; ++j) {
                const whisper_token_data td = whisper_full_get_token_data(s->whisper_ctx, i, j);
                if (td.id >= whisper_token_eot(s->whisper_ctx)) {
                    continue; // skip EOT / timestamp / lang / special tokens
                }
                const char* ttext = whisper_full_get_token_text(s->whisper_ctx, i, j);
                if (!ttext || ttext[0] == '\0') {
                    continue;
                }
                ca_token_record rec;
                rec.text = ttext;
                rec.t0 = td.t0;
                rec.t1 = td.t1;
                rec.p = td.p;
                const int n_alts = whisper_full_get_token_n_alts(s->whisper_ctx, i, j);
                if (n_alts > 0) {
                    rec.alts.reserve((size_t)n_alts);
                    for (int k = 0; k < n_alts; ++k) {
                        const whisper_token alt_id = whisper_full_get_token_alt_id(s->whisper_ctx, i, j, k);
                        const float alt_p = whisper_full_get_token_alt_p(s->whisper_ctx, i, j, k);
                        const char* alt_text = whisper_token_to_str(s->whisper_ctx, alt_id);
                        crispasr_session_seg::word_alt wa;
                        wa.text = alt_text ? alt_text : "";
                        wa.p = alt_p;
                        rec.alts.push_back(std::move(wa));
                    }
                }
                toks.push_back(std::move(rec));
            }
            seg.words = emit_words_from_tokens(toks);
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
                // Mean of sub-word token softmax probs (parakeet.cpp word
                // grouping). 0 if the word came from no-token source.
                w.p = pr->words[i].p > 0.0f ? pr->words[i].p : 1.0f;
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
    // (Historical run_char_transcribe lambda removed — every backend now
    // either has a token-prob path or uses the explicit text-only fallback
    // block at the bottom of this function. See PLAN #65.)

#ifdef CA_HAVE_CANARY
    if (s->backend == "canary" && s->canary_ctx) {
        // Canary supports source/target language + punctuation explicitly.
        // Resolution order (most specific wins): per-call `language` arg
        // → sticky `s->source_language` / `s->target_language` → historical
        // default of en→en. Same for punctuation: sticky `s->punctuation`
        // (default true).
        const std::string src = lang_set ? lang : (!s->source_language.empty() ? s->source_language : "en");
        const std::string tgt = !s->target_language.empty() ? s->target_language : src;
        if (s->beam_size > 1)
            canary_set_beam_size(s->canary_ctx, s->beam_size);
        canary_result* cr =
            canary_transcribe_ex(s->canary_ctx, pcm, n_samples, src.c_str(), tgt.c_str(), s->punctuation, 0);
        if (!cr) {
            delete r;
            return nullptr;
        }
        crispasr_session_seg seg;
        seg.text = cr->text ? cr->text : "";
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);

        std::vector<ca_token_record> toks;
        toks.reserve((size_t)cr->n_tokens);
        for (int i = 0; i < cr->n_tokens; i++) {
            ca_token_record tk;
            tk.text = cr->tokens[i].text; // already ▁→' ' decoded
            tk.t0 = cr->tokens[i].t0;
            tk.t1 = cr->tokens[i].t1;
            tk.p = cr->tokens[i].p;
            toks.push_back(std::move(tk));
        }
        seg.words = emit_words_from_tokens(toks);
        canary_result_free(cr);
        r->segments.push_back(std::move(seg));
        return r;
    }
#endif
#ifdef CA_HAVE_LFM2_AUDIO
    if (s->backend == "lfm2-audio" && s->lfm2_audio_ctx) {
        // Forward language hint or ask prompt; fall back to the session language
        const char* lfm2_prompt = nullptr;
        if (!s->ask.empty())
            lfm2_prompt = s->ask.c_str();
        else if (lang_set)
            lfm2_prompt = lang.c_str();
        char* text = lfm2_audio_transcribe(s->lfm2_audio_ctx, pcm, n_samples, lfm2_prompt, 0);
        if (!text) {
            delete r;
            return nullptr;
        }
        crispasr_session_seg seg;
        seg.text = text;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
        free(text);
        r->segments.push_back(std::move(seg));
        return r;
    }
#endif
#ifdef CA_HAVE_MINI_OMNI2
    if ((s->backend == "mini-omni2" || s->backend == "mini_omni2" || s->backend == "miniomni2") && s->mini_omni2_ctx) {
        mini_omni2_set_ask(s->mini_omni2_ctx, s->ask.empty() ? nullptr : s->ask.c_str());
        char* text = mini_omni2_transcribe(s->mini_omni2_ctx, pcm, n_samples);
        if (!text) {
            delete r;
            return nullptr;
        }
        crispasr_session_seg seg;
        seg.text = text;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
        free(text);
        r->segments.push_back(std::move(seg));
        return r;
    }
#endif
#ifdef CA_HAVE_QWEN3
    // mega-asr is handled here too via the qwen3_ctx — it's just
    // qwen3 weights with a merged robustness LoRA. See the matching
    // alias set in crispasr_session_open_explicit.
    if ((s->backend == "qwen3" || s->backend == "mega-asr" || s->backend == "mega_asr" || s->backend == "megaasr") &&
        s->qwen3_ctx) {
        // qwen3-asr's runtime _transcribe() is a stub. Drive the building
        // blocks the CLI adapter uses (compute_mel → run_encoder → tokenize
        // → embed+splice → kv_init → run_llm_kv prefill → greedy decode).
        // Capture per-step softmax probability via core_greedy_decode.
        int n_mels = 0, T_mel = 0;
        float* mel = qwen3_asr_compute_mel(s->qwen3_ctx, pcm, n_samples, &n_mels, &T_mel);
        if (!mel) {
            delete r;
            return nullptr;
        }
        int N_enc = 0, pdim = 0;
        float* audio_embeds = qwen3_asr_run_encoder(s->qwen3_ctx, mel, n_mels, T_mel, &N_enc, &pdim);
        std::free(mel);
        if (!audio_embeds) {
            delete r;
            return nullptr;
        }

        // ChatML prompt: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n
        // <|audio_start|><|audio_pad|>×N<|audio_end|>{question}<|im_end|>\n
        // <|im_start|>assistant\n
        //
        // When `s->ask` is set, inject the question between the audio
        // close token and the user-turn end so the LLM answers it
        // instead of producing a verbatim transcript. Empty ask keeps
        // the historical transcribe-only template.
        std::string text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>";
        text.reserve(text.size() + (size_t)N_enc * 13 + 64 + s->ask.size());
        for (int i = 0; i < N_enc; i++)
            text += "<|audio_pad|>";
        text += "<|audio_end|>";
        if (!s->ask.empty()) {
            text += '\n';
            text += s->ask;
        }
        text += "<|im_end|>\n<|im_start|>assistant\n";

        int n_prompt = 0;
        int32_t* raw_ids = qwen3_asr_tokenize(s->qwen3_ctx, text.c_str(), &n_prompt);
        if (!raw_ids) {
            std::free(audio_embeds);
            delete r;
            return nullptr;
        }
        std::vector<int32_t> ids(raw_ids, raw_ids + n_prompt);
        std::free(raw_ids);

        int n_pad_id = 0;
        int32_t* pad_id_arr = qwen3_asr_tokenize(s->qwen3_ctx, "<|audio_pad|>", &n_pad_id);
        const int audio_pad_id = (pad_id_arr && n_pad_id >= 1) ? pad_id_arr[0] : -1;
        std::free(pad_id_arr);
        if (audio_pad_id < 0) {
            std::free(audio_embeds);
            delete r;
            return nullptr;
        }

        float* text_embeds = qwen3_asr_embed_tokens(s->qwen3_ctx, ids.data(), (int)ids.size());
        if (!text_embeds) {
            std::free(audio_embeds);
            delete r;
            return nullptr;
        }
        int spliced = 0;
        for (size_t i = 0; i < ids.size() && spliced < N_enc; i++) {
            if (ids[i] == audio_pad_id) {
                std::memcpy(text_embeds + i * pdim, audio_embeds + (size_t)spliced * pdim, pdim * sizeof(float));
                spliced++;
            }
        }
        std::free(audio_embeds);

        if (!qwen3_asr_kv_init(s->qwen3_ctx, 4096)) {
            std::free(text_embeds);
            delete r;
            return nullptr;
        }
        qwen3_asr_kv_reset(s->qwen3_ctx);

        int n_t = 0, vocab = 0;
        float* logits = qwen3_asr_run_llm_kv(s->qwen3_ctx, text_embeds, (int)ids.size(), 0, &n_t, &vocab);
        std::free(text_embeds);
        if (!logits) {
            delete r;
            return nullptr;
        }

        int eos_id = -1;
        int n_eos = 0;
        int32_t* eos_arr = qwen3_asr_tokenize(s->qwen3_ctx, "<|im_end|>", &n_eos);
        if (eos_arr && n_eos >= 1)
            eos_id = eos_arr[0];
        std::free(eos_arr);

        const int last_off = (n_t - 1) * vocab;
        const int prompt_len_q3 = (int)ids.size();

        core_greedy_decode::Result dec;
        if (s->beam_size > 1) {
            // PLAN §90: session beam_size → qwen3-asr beam decode.
            auto replay = [](qwen3_asr_context* ctx, const int32_t* toks, int n, int pl) -> float* {
                float* emb = qwen3_asr_embed_tokens(ctx, toks, n);
                if (!emb)
                    return nullptr;
                float* lg = qwen3_asr_run_llm_kv(ctx, emb, n, pl, nullptr, nullptr);
                std::free(emb);
                return lg;
            };
            core_beam_decode::Config bcfg;
            bcfg.max_new_tokens = s->max_new_tokens > 0 ? s->max_new_tokens : 256;
            bcfg.eos_id = eos_id;
            bcfg.vocab_size = vocab;
            bcfg.beam_size = s->beam_size;
            bcfg.prompt_len = prompt_len_q3;
            auto br = core_beam_decode::run_with_probs(s->qwen3_ctx, logits + last_off, replay, bcfg);
            qwen3_asr_kv_reset(s->qwen3_ctx);
            dec.tokens = std::move(br.tokens);
            dec.probs = std::move(br.probs);
        } else {
            const int first_tok = core_greedy_decode::argmax(logits + last_off, vocab);
            const float first_p =
                core_greedy_decode::softmax_of(logits + last_off, vocab, first_tok, logits[last_off + first_tok]);
            core_greedy_decode::Config dec_cfg;
            dec_cfg.max_new_tokens = s->max_new_tokens > 0 ? s->max_new_tokens : 256;
            dec_cfg.eos_id = eos_id;
            dec_cfg.vocab_size = vocab;
            dec_cfg.frequency_penalty = s->frequency_penalty;
            dec = core_greedy_decode::run_with_probs(s->qwen3_ctx, first_tok, first_p, prompt_len_q3,
                                                     qwen3_asr_embed_tokens, qwen3_asr_run_llm_kv, dec_cfg);
        }
        std::free(logits);

        // Detokenize, filtering out qwen3's metadata wrapper tokens
        // (<|im_start|>, <asr_text>, "language <name>", etc.) the same
        // way the CLI adapter does, and project surviving tokens into
        // ca_token_record for word grouping.
        std::string transcript;
        std::vector<ca_token_record> toks;
        toks.reserve(dec.tokens.size());
        bool capture_language = false;
        for (size_t i = 0; i < dec.tokens.size(); i++) {
            const int32_t id = dec.tokens[i];
            if (id == eos_id)
                break;
            const char* raw_piece = qwen3_asr_token_text(s->qwen3_ctx, id);
            if (!raw_piece || !*raw_piece)
                continue;
            std::string raw = raw_piece;
            // Skip qwen3 special tokens and structured tags.
            if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|')
                continue;
            if (raw.size() >= 2 && raw[0] == '<' && raw.back() == '>')
                continue;
            if (raw.size() >= 5 && raw[0] == '[' && raw[1] == 'P' && raw[2] == 'A' && raw[3] == 'D')
                continue;
            std::string piece = gpt2_byte_decode(raw);
            if (piece == "language") {
                capture_language = true;
                continue;
            }
            if (capture_language) {
                capture_language = false; // language name eaten, no transcript contribution
                continue;
            }
            transcript += piece;
            ca_token_record tk;
            tk.text = piece;
            tk.t0 = -1;
            tk.t1 = -1;
            tk.p = (i < dec.probs.size()) ? dec.probs[i] : -1.0f;
            toks.push_back(std::move(tk));
        }
        while (!transcript.empty() && (transcript.front() == ' ' || transcript.front() == '\n'))
            transcript.erase(transcript.begin());

        crispasr_session_seg seg;
        seg.text = transcript;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
        seg.words = emit_words_from_tokens(toks);
        r->segments.push_back(std::move(seg));
        return r;
    }
#endif
#ifdef CA_HAVE_COHERE
    if (s->backend == "cohere" && s->cohere_ctx) {
        // Cohere takes a single `lang` (source); per-call wins, sticky next.
        const std::string src = lang_set ? lang : (!s->source_language.empty() ? s->source_language : "en");
        cohere_set_max_new_tokens(s->cohere_ctx, s->max_new_tokens);
        cohere_set_frequency_penalty(s->cohere_ctx, s->frequency_penalty);
        if (s->beam_size > 1)
            cohere_set_beam_size(s->cohere_ctx, s->beam_size);
        cohere_result* cr = cohere_transcribe_ex(s->cohere_ctx, pcm, n_samples, src.c_str(), 0);
        if (!cr) {
            delete r;
            return nullptr;
        }
        crispasr_session_seg seg;
        seg.text = cr->text ? cr->text : "";
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);

        std::vector<ca_token_record> toks;
        toks.reserve((size_t)cr->n_tokens);
        for (int i = 0; i < cr->n_tokens; i++) {
            ca_token_record tk;
            tk.text = cr->tokens[i].text; // already ▁→' ' decoded
            tk.t0 = cr->tokens[i].t0;
            tk.t1 = cr->tokens[i].t1;
            tk.p = cr->tokens[i].p;
            toks.push_back(std::move(tk));
        }
        seg.words = emit_words_from_tokens(toks);
        cohere_result_free(cr);
        r->segments.push_back(std::move(seg));
        return r;
    }
#endif
#ifdef CA_HAVE_GRANITE
    if ((s->backend == "granite" || s->backend == "granite-4.1" || s->backend == "granite-4.1-plus") &&
        s->granite_ctx) {
        // granite_speech_transcribe is a stub. Drive the building blocks:
        // mel → encoder → projector → tokenize prompt → splice projector
        // output into the audio-pad slots → kv prefill → greedy decode.
        int n_mels = 0, T_mel = 0;
        float* mel = granite_speech_compute_mel(s->granite_ctx, pcm, n_samples, &n_mels, &T_mel);
        if (!mel) {
            delete r;
            return nullptr;
        }
        int N_enc = 0, enc_dim = 0;
        float* enc = granite_speech_run_encoder(s->granite_ctx, mel, n_mels, T_mel, &N_enc, &enc_dim);
        std::free(mel);
        if (!enc) {
            delete r;
            return nullptr;
        }
        int N_proj = 0, proj_dim = 0;
        float* proj = granite_speech_run_projector(s->granite_ctx, enc, N_enc, enc_dim, &N_proj, &proj_dim);
        std::free(enc);
        if (!proj) {
            delete r;
            return nullptr;
        }

        int audio_tok = granite_speech_audio_token_id(s->granite_ctx);
        int eos_tok = granite_speech_eos_token_id(s->granite_ctx);
        // Legacy granite-4.0 fallbacks (mirrors crispasr_backend_granite.cpp).
        if (audio_tok < 0)
            audio_tok = 100352;
        if (eos_tok < 0)
            eos_tok = 100257;

        // granite-3.x uses control-token chat template; granite-4.0 uses
        // "USER: …\n ASSISTANT:". Discriminator: audio_token < 50000 ⇒ v3.
        const bool use_v3_template = (audio_tok < 50000);
        std::vector<int32_t> prefix_ids, suffix_ids;
        if (use_v3_template) {
            const std::string prefix_str = "<|start_of_role|>user<|end_of_role|>";
            const std::string suffix_str = (!s->ask.empty() ? s->ask
                                                            : std::string("can you transcribe the speech into a "
                                                                          "written format?")) +
                                           "<|end_of_text|>\n"
                                           "<|start_of_role|>assistant<|end_of_role|>";
            int n = 0;
            int32_t* a = granite_speech_tokenize(s->granite_ctx, prefix_str.c_str(), &n);
            if (a && n > 0) {
                prefix_ids.assign(a, a + n);
                std::free(a);
            } else if (a)
                std::free(a);
            a = granite_speech_tokenize(s->granite_ctx, suffix_str.c_str(), &n);
            if (a && n > 0) {
                suffix_ids.assign(a, a + n);
                std::free(a);
            } else if (a)
                std::free(a);
        } else {
            // granite-4.0-1b legacy hardcoded ids: "USER: " + transcription request.
            static const int32_t kPrefix4[] = {6584, 25, 220};
            prefix_ids.assign(kPrefix4, kPrefix4 + (sizeof(kPrefix4) / sizeof(kPrefix4[0])));
            if (!s->ask.empty()) {
                const std::string suffix4_str = s->ask + "\nASSISTANT:";
                int n = 0;
                int32_t* a = granite_speech_tokenize(s->granite_ctx, suffix4_str.c_str(), &n);
                if (a && n > 0) {
                    suffix_ids.assign(a, a + n);
                    std::free(a);
                } else if (a)
                    std::free(a);
            } else {
                static const int32_t kSuffix4[] = {4919, 499,  1380, 3191, 279,   8982, 1139, 264,
                                                   5439, 3645, 30,   198,  36660, 3931, 2891, 25};
                suffix_ids.assign(kSuffix4, kSuffix4 + (sizeof(kSuffix4) / sizeof(kSuffix4[0])));
            }
        }
        if (prefix_ids.empty() || suffix_ids.empty()) {
            std::free(proj);
            delete r;
            return nullptr;
        }

        const int n_prefix = (int)prefix_ids.size();
        const int n_suffix = (int)suffix_ids.size();
        const int total_prompt = n_prefix + N_proj + n_suffix;
        std::vector<int32_t> prompt_ids;
        prompt_ids.reserve(total_prompt);
        for (int id : prefix_ids)
            prompt_ids.push_back(id);
        for (int i = 0; i < N_proj; i++)
            prompt_ids.push_back(audio_tok);
        for (int id : suffix_ids)
            prompt_ids.push_back(id);

        float* all_embeds = granite_speech_embed_tokens(s->granite_ctx, prompt_ids.data(), total_prompt);
        if (!all_embeds) {
            std::free(proj);
            delete r;
            return nullptr;
        }
        for (int i = 0; i < N_proj; i++)
            std::memcpy(all_embeds + (size_t)(n_prefix + i) * proj_dim, proj + (size_t)i * proj_dim,
                        proj_dim * sizeof(float));
        std::free(proj);

        if (!granite_speech_kv_init(s->granite_ctx, 4096)) {
            std::free(all_embeds);
            delete r;
            return nullptr;
        }
        granite_speech_kv_reset(s->granite_ctx);

        int vocab = 0;
        float* logits = granite_speech_run_llm_kv(s->granite_ctx, all_embeds, total_prompt, 0, nullptr, &vocab);
        std::free(all_embeds);
        if (!logits) {
            delete r;
            return nullptr;
        }
        core_greedy_decode::Result dec;
        if (s->beam_size > 1) {
            // PLAN §90: session beam_size → granite beam decode.
            auto replay = [](granite_speech_context* ctx, const int32_t* toks, int n, int pl) -> float* {
                float* emb = granite_speech_embed_tokens(ctx, toks, n);
                if (!emb)
                    return nullptr;
                float* lg = granite_speech_run_llm_kv(ctx, emb, n, pl, nullptr, nullptr);
                std::free(emb);
                return lg;
            };
            core_beam_decode::Config bcfg;
            bcfg.max_new_tokens = s->max_new_tokens > 0 ? s->max_new_tokens : 200;
            bcfg.eos_id = eos_tok;
            bcfg.vocab_size = vocab;
            bcfg.beam_size = s->beam_size;
            bcfg.prompt_len = total_prompt;
            auto br = core_beam_decode::run_with_probs(s->granite_ctx, logits, replay, bcfg);
            granite_speech_kv_reset(s->granite_ctx);
            dec.tokens = std::move(br.tokens);
            dec.probs = std::move(br.probs);
        } else {
            const int first_tok = core_greedy_decode::argmax(logits, vocab);
            const float first_p = core_greedy_decode::softmax_of(logits, vocab, first_tok, logits[first_tok]);
            core_greedy_decode::Config dec_cfg;
            dec_cfg.max_new_tokens = s->max_new_tokens > 0 ? s->max_new_tokens : 200;
            dec_cfg.eos_id = eos_tok;
            dec_cfg.vocab_size = vocab;
            dec_cfg.frequency_penalty = s->frequency_penalty;
            dec = core_greedy_decode::run_with_probs(s->granite_ctx, first_tok, first_p, total_prompt,
                                                     granite_speech_embed_tokens, granite_speech_run_llm_kv, dec_cfg);
        }
        std::free(logits);

        // Detokenize batch via granite's own merge logic for the segment
        // text. Per-token text comes from gpt2_byte_decode of single ids
        // for the word-grouping pass.
        std::vector<int32_t> text_ids;
        text_ids.reserve(dec.tokens.size());
        for (int32_t id : dec.tokens)
            if (id != eos_tok)
                text_ids.push_back(id);
        char* batch_text = granite_speech_decode_tokens(s->granite_ctx, text_ids.data(), (int)text_ids.size());
        std::string transcript = batch_text ? batch_text : "";
        if (batch_text)
            std::free(batch_text);
        while (!transcript.empty() && (transcript.front() == ' ' || transcript.front() == '\n'))
            transcript.erase(transcript.begin());

        std::vector<ca_token_record> toks;
        toks.reserve(dec.tokens.size());
        for (size_t i = 0; i < dec.tokens.size(); i++) {
            const int32_t id = dec.tokens[i];
            if (id == eos_tok)
                break;
            const char* raw = granite_speech_token_text(s->granite_ctx, id);
            std::string piece = raw ? gpt2_byte_decode(raw) : "";
            ca_token_record tk;
            tk.text = piece;
            tk.t0 = -1;
            tk.t1 = -1;
            tk.p = (i < dec.probs.size()) ? dec.probs[i] : -1.0f;
            toks.push_back(std::move(tk));
        }

        crispasr_session_seg seg;
        seg.text = transcript;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
        seg.words = emit_words_from_tokens(toks);
        r->segments.push_back(std::move(seg));
        return r;
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
        return run_voxtral_family(s->voxtral_ctx, ops, pcm, n_samples, lang, s->ask, s->beam_size);
    }
#endif
#ifdef CA_HAVE_VOXTRAL4B
    if (s->backend == "voxtral4b" && s->voxtral4b_ctx) {
        // Voxtral-Mini-4B-Realtime-2602 uses a streaming-prompt convention
        // (BOS + 38 STREAMING_PAD + audio-injection pre_hook) that's
        // qualitatively different from voxtral-3B's [INST]...[TRANSCRIBE]
        // template. The unified `run_voxtral_family` orchestrator above
        // assumes the 3B template and crashes on arbitrary audio sizes
        // (projector stride-8 misalignment). PLAN #7 phase 1+1.5's streaming
        // implementation has the right prompt convention; route session
        // transcribe through it. Bit-exact match to the CLI batch path,
        // validated via tools/bench_streaming_latency.py --check-batch-equality.
        (void)lang;   // voxtral4b-realtime is en-only via the CLI adapter
        (void)s->ask; // streaming path doesn't take Q&A prompts (yet)
        voxtral4b_stream* vs = voxtral4b_stream_open(s->voxtral4b_ctx, /*step_ms*/ 0, /*length_ms*/ 0);
        if (!vs) {
            delete r;
            return nullptr;
        }
        if (voxtral4b_stream_feed(vs, pcm, n_samples) != 0 || voxtral4b_stream_flush(vs) != 1) {
            voxtral4b_stream_close(vs);
            delete r;
            return nullptr;
        }
        std::vector<char> buf(8192, 0);
        double t0_s = 0.0, t1_s = 0.0;
        int64_t counter = 0;
        voxtral4b_stream_get_text(vs, buf.data(), (int)buf.size(), &t0_s, &t1_s, &counter);
        voxtral4b_stream_close(vs);
        crispasr_session_seg seg;
        seg.text = std::string(buf.data());
        seg.t0 = (int64_t)(t0_s * 100.0);
        seg.t1 = (int64_t)(t1_s * 100.0);
        // No per-token records from the streaming API today; word splitting
        // falls back to whitespace via emit_words_from_tokens(empty).
        seg.words = emit_words_from_tokens({});
        r->segments.push_back(std::move(seg));
        return r;
    }
#endif
#ifdef CA_HAVE_WAV2VEC2
    if (s->backend == "wav2vec2" && s->wav2vec2_ctx) {
        // Encoder + CTC head → logits → greedy decode with per-emission
        // probabilities. Each non-blank emission is a CTC frame; we group
        // them into words on the SentencePiece "|" (= space) boundary.
        auto logits = wav2vec2_compute_logits(*s->wav2vec2_ctx, pcm, n_samples, s->n_threads);
        if (logits.empty()) {
            delete r;
            return nullptr;
        }
        const int V = (int)s->wav2vec2_ctx->hparams.vocab_size;
        const int T = (int)(logits.size() / (size_t)V);
        auto emits = (s->beam_size > 1)
                         ? wav2vec2_beam_decode_with_probs(*s->wav2vec2_ctx, logits.data(), T, s->beam_size, 2.3f)
                         : wav2vec2_greedy_decode_with_probs(*s->wav2vec2_ctx, logits.data(), T);
        const float frame_dur_s = wav2vec2_frame_dur(*s->wav2vec2_ctx);

        // Build the transcript text and project emissions into the
        // ca_token_record shape that emit_words_from_tokens consumes.
        std::vector<ca_token_record> toks;
        toks.reserve(emits.size());
        std::string text;
        for (const auto& e : emits) {
            text += e.text;
            ca_token_record tk;
            tk.text = e.text;
            tk.t0 = (int64_t)(e.frame_start * frame_dur_s * 100.0);
            tk.t1 = (int64_t)((e.frame_end + 1) * frame_dur_s * 100.0);
            tk.p = e.prob;
            toks.push_back(std::move(tk));
        }
        // Trim leading/trailing spaces from the assembled transcript.
        auto lo = text.find_first_not_of(' ');
        auto hi = text.find_last_not_of(' ');
        text = (lo == std::string::npos) ? "" : text.substr(lo, hi - lo + 1);

        crispasr_session_seg seg;
        seg.text = text;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
        seg.words = emit_words_from_tokens(toks);
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
        vibevoice_result* vr = vibevoice_transcribe_with_probs(s->vibevoice_ctx, pcm24.data(), (int)pcm24.size());
        if (!vr || !vr->text) {
            if (vr)
                vibevoice_result_free(vr);
            delete r;
            return nullptr;
        }
        std::vector<ca_token_record> toks;
        toks.reserve((size_t)vr->n_tokens);
        for (int i = 0; i < vr->n_tokens; i++) {
            ca_token_record tk;
            const char* raw = vibevoice_token_text(s->vibevoice_ctx, vr->token_ids[i]);
            if (raw && *raw) {
                std::string s_raw = raw;
                // Skip Qwen2 special tokens like <|im_end|>; the runtime
                // filter in vibevoice_transcribe_impl strips them from the
                // segment text but the per-token list keeps them with empty
                // text so confidence indices stay aligned with the ids.
                if (!(s_raw.size() >= 4 && s_raw[0] == '<' && s_raw[1] == '|'))
                    tk.text = gpt2_byte_decode(s_raw); // Qwen2 byte-level BPE
            }
            tk.t0 = -1;
            tk.t1 = -1;
            tk.p = vr->token_probs[i];
            toks.push_back(std::move(tk));
        }
        crispasr_session_seg seg;
        seg.text = vr->text;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
        seg.words = emit_words_from_tokens(toks);
        vibevoice_result_free(vr);
        r->segments.push_back(std::move(seg));
        return r;
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
        canary_ctc_decode_result* dr = canary_ctc_greedy_decode_with_probs(s->ctc_ctx, logits, T_enc, V);
        std::free(logits);
        if (!dr || !dr->text) {
            if (dr)
                canary_ctc_decode_result_free(dr);
            delete r;
            return nullptr;
        }
        const int frame_dur_cs = canary_ctc_frame_dur_cs(s->ctc_ctx);
        crispasr_session_seg seg;
        seg.text = dr->text;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);

        std::vector<ca_token_record> toks;
        toks.reserve((size_t)dr->n_tokens);
        for (int i = 0; i < dr->n_tokens; i++) {
            ca_token_record tk;
            if (dr->text_lengths[i] > 0)
                tk.text.assign(dr->text + dr->text_offsets[i], (size_t)dr->text_lengths[i]);
            tk.t0 = (int64_t)dr->frame_starts[i] * frame_dur_cs;
            tk.t1 = (int64_t)(dr->frame_ends[i] + 1) * frame_dur_cs;
            tk.p = dr->token_probs[i];
            toks.push_back(std::move(tk));
        }
        seg.words = emit_words_from_tokens(toks);
        canary_ctc_decode_result_free(dr);
        r->segments.push_back(std::move(seg));
        return r;
    }
#endif

    // Helper: package a text-only result with the standard segment span.
    auto package_text_only = [&](char* text, bool need_free) -> crispasr_session_result* {
        if (!text) {
            delete r;
            return nullptr;
        }
        crispasr_session_seg seg;
        seg.text = text;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
        r->segments.push_back(std::move(seg));
        if (need_free)
            std::free(text);
        return r;
    };

    // Helper: package a text + per-token result. Constructs ca_token_record
    // entries from typed token arrays and runs the SentencePiece word grouping.
    auto package_with_tokens = [&](char* text, std::vector<ca_token_record>&& toks) -> crispasr_session_result* {
        if (!text) {
            delete r;
            return nullptr;
        }
        crispasr_session_seg seg;
        seg.text = text;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n_samples * 100.0 / 16000.0);
        seg.words = emit_words_from_tokens(toks);
        r->segments.push_back(std::move(seg));
        return r;
    };

#ifdef CA_HAVE_GLMASR
    if ((s->backend == "glm-asr" || s->backend == "glmasr" || s->backend == "glm" || s->backend == "glm_asr") &&
        s->glmasr_ctx) {
        // PLAN §90: session beam_size → glm-asr's per-context setter.
        if (s->beam_size > 1) {
            glm_asr_set_beam_size((glm_asr_context*)s->glmasr_ctx, s->beam_size);
        }
        glm_asr_set_ask((glm_asr_context*)s->glmasr_ctx, s->ask.empty() ? nullptr : s->ask.c_str());
        glm_asr_result* gr = glm_asr_transcribe_with_probs((glm_asr_context*)s->glmasr_ctx, pcm, n_samples);
        if (!gr || !gr->text) {
            if (gr)
                glm_asr_result_free(gr);
            delete r;
            return nullptr;
        }
        std::vector<ca_token_record> toks;
        toks.reserve((size_t)gr->n_tokens);
        for (int i = 0; i < gr->n_tokens; i++) {
            ca_token_record tk;
            // GLM uses GPT-2 byte-level BPE: Ġ→space, Ċ→newline.
            const char* raw = glm_asr_token_text((glm_asr_context*)s->glmasr_ctx, gr->token_ids[i]);
            if (raw) {
                for (size_t ci = 0; raw[ci] != '\0';) {
                    unsigned char c = (unsigned char)raw[ci];
                    if (c == 0xC4 && raw[ci + 1] != '\0') {
                        unsigned char c2 = (unsigned char)raw[ci + 1];
                        if (c2 == 0xA0) {
                            tk.text += ' ';
                            ci += 2;
                            continue;
                        }
                        if (c2 == 0x8A) {
                            tk.text += '\n';
                            ci += 2;
                            continue;
                        }
                    }
                    tk.text += (char)c;
                    ci++;
                }
            }
            tk.t0 = -1;
            tk.t1 = -1;
            tk.p = gr->token_probs[i];
            toks.push_back(std::move(tk));
        }
        char* text = strdup(gr->text);
        glm_asr_result_free(gr);
        return package_with_tokens(text, std::move(toks));
    }
#endif
#ifdef CA_HAVE_KYUTAI
    if ((s->backend == "kyutai-stt" || s->backend == "kyutai" || s->backend == "moshi-stt") && s->kyutai_ctx) {
        // PLAN §90: forward sticky session beam_size into kyutai-stt's
        // per-context setter so session-API consumers (CrisperWeaver's
        // worker pool, Rust/Node bindings) get the same beam search
        // the CLI does. 1 = greedy = no-op at the backend level.
        if (s->beam_size > 1) {
            kyutai_stt_set_beam_size((kyutai_stt_context*)s->kyutai_ctx, s->beam_size);
        }
        kyutai_stt_result* kr = kyutai_stt_transcribe_with_probs((kyutai_stt_context*)s->kyutai_ctx, pcm, n_samples);
        if (!kr || !kr->text) {
            if (kr)
                kyutai_stt_result_free(kr);
            delete r;
            return nullptr;
        }
        std::vector<ca_token_record> toks;
        toks.reserve((size_t)kr->n_tokens);
        for (int i = 0; i < kr->n_tokens; i++) {
            ca_token_record tk;
            const char* piece = kyutai_stt_token_text((kyutai_stt_context*)s->kyutai_ctx, kr->token_ids[i]);
            if (piece) {
                std::string p = piece;
                for (size_t ci = 0; ci < p.size(); ci++) {
                    if ((unsigned char)p[ci] == 0xE2 && ci + 2 < p.size() && (unsigned char)p[ci + 1] == 0x96 &&
                        (unsigned char)p[ci + 2] == 0x81) {
                        tk.text += ' ';
                        ci += 2;
                    } else {
                        tk.text += p[ci];
                    }
                }
            }
            tk.t0 = -1;
            tk.t1 = -1;
            tk.p = kr->token_probs[i];
            toks.push_back(std::move(tk));
        }
        char* text = strdup(kr->text);
        kyutai_stt_result_free(kr);
        return package_with_tokens(text, std::move(toks));
    }
#endif
#ifdef CA_HAVE_FIRERED
    if ((s->backend == "firered-asr" || s->backend == "firered") && s->firered_ctx) {
        // PLAN §90: session beam_size → firered's per-context setter.
        if (s->beam_size > 1) {
            firered_asr_set_beam_size((firered_asr_context*)s->firered_ctx, s->beam_size);
        }
        firered_asr_result* fr =
            firered_asr_transcribe_with_probs((firered_asr_context*)s->firered_ctx, pcm, n_samples);
        if (!fr || !fr->text) {
            if (fr)
                firered_asr_result_free(fr);
            delete r;
            return nullptr;
        }
        std::vector<ca_token_record> toks;
        toks.reserve((size_t)fr->n_tokens);
        for (int i = 0; i < fr->n_tokens; i++) {
            ca_token_record tk;
            const char* piece = firered_asr_token_text((firered_asr_context*)s->firered_ctx, fr->token_ids[i]);
            if (piece) {
                std::string p = piece;
                for (size_t ci = 0; ci < p.size(); ci++) {
                    if ((unsigned char)p[ci] == 0xE2 && ci + 2 < p.size() && (unsigned char)p[ci + 1] == 0x96 &&
                        (unsigned char)p[ci + 2] == 0x81) {
                        tk.text += ' ';
                        ci += 2;
                    } else {
                        tk.text += p[ci];
                    }
                }
            }
            tk.t0 = -1;
            tk.t1 = -1;
            tk.p = fr->token_probs[i];
            toks.push_back(std::move(tk));
        }
        char* text = strdup(fr->text);
        firered_asr_result_free(fr);
        return package_with_tokens(text, std::move(toks));
    }
#endif
#ifdef CA_HAVE_MOONSHINE
    if (s->backend == "moonshine" && s->moonshine_ctx) {
        // PLAN §90: session beam_size → moonshine's per-context setter.
        if (s->beam_size > 1) {
            moonshine_set_beam_size((moonshine_context*)s->moonshine_ctx, s->beam_size);
        }
        moonshine_result* mr = moonshine_transcribe_with_probs((moonshine_context*)s->moonshine_ctx, pcm, n_samples);
        if (!mr || !mr->text) {
            if (mr)
                moonshine_result_free(mr);
            delete r;
            return nullptr;
        }
        std::vector<ca_token_record> toks;
        toks.reserve((size_t)mr->n_tokens);
        for (int i = 0; i < mr->n_tokens; i++) {
            ca_token_record tk;
            const char* piece = moonshine_token_text((moonshine_context*)s->moonshine_ctx, mr->token_ids[i]);
            if (piece && piece[0])
                tk.text = piece;
            tk.t0 = -1;
            tk.t1 = -1;
            tk.p = mr->token_probs[i];
            toks.push_back(std::move(tk));
        }
        char* text = strdup(mr->text);
        moonshine_result_free(mr);
        return package_with_tokens(text, std::move(toks));
    }
#endif
#ifdef CA_HAVE_OMNIASR
    if ((s->backend.rfind("omniasr", 0) == 0) && s->omniasr_ctx) {
        // PLAN §90: session beam_size → omniasr's per-context setter.
        // Effective only on the LLM variant (CTC has no beam path);
        // the CTC fall-through below silently ignores the setting.
        if (s->beam_size > 1) {
            omniasr_set_beam_size((omniasr_context*)s->omniasr_ctx, s->beam_size);
        }
        // LLM variant produces per-token probs; CTC variant returns nullptr
        // here — fall through to the plain-text path below.
        omniasr_result* oar = omniasr_transcribe_with_probs((omniasr_context*)s->omniasr_ctx, pcm, n_samples);
        if (oar && oar->text) {
            std::vector<ca_token_record> toks;
            toks.reserve((size_t)oar->n_tokens);
            for (int i = 0; i < oar->n_tokens; i++) {
                ca_token_record tk;
                const char* piece = omniasr_token_text((omniasr_context*)s->omniasr_ctx, oar->token_ids[i]);
                if (piece) {
                    std::string p = piece;
                    for (size_t ci = 0; ci < p.size(); ci++) {
                        if ((unsigned char)p[ci] == 0xE2 && ci + 2 < p.size() && (unsigned char)p[ci + 1] == 0x96 &&
                            (unsigned char)p[ci + 2] == 0x81) {
                            tk.text += ' ';
                            ci += 2;
                        } else {
                            tk.text += p[ci];
                        }
                    }
                }
                tk.t0 = -1;
                tk.t1 = -1;
                tk.p = oar->token_probs[i];
                toks.push_back(std::move(tk));
            }
            char* text = strdup(oar->text);
            omniasr_result_free(oar);
            return package_with_tokens(text, std::move(toks));
        }
        if (oar)
            omniasr_result_free(oar);
        // CTC variant: text only.
        return package_text_only(omniasr_transcribe((omniasr_context*)s->omniasr_ctx, pcm, n_samples), true);
    }
#endif

    // Backends without a token-prob API yet: text-only segment.
    {
        char* text = nullptr;
        bool need_free = true;
#ifdef CA_HAVE_GRANITE_NLE
        if (!text && (s->backend == "granite-4.1-nar" || s->backend == "granite-nle" || s->backend == "granite_nle") &&
            s->granite_nle_ctx) {
            // granite_nle_transcribe is the single high-level entry point —
            // returns malloc'd UTF-8, caller frees. Token-prob accessors
            // aren't exposed on the C-side yet (the LLM editing forward pass
            // doesn't emit per-token p values the same way greedy decoders
            // do), so text-only is the right shape for now.
            text = granite_nle_transcribe(s->granite_nle_ctx, pcm, n_samples);
            need_free = true;
        }
#endif
#ifdef CA_HAVE_FUNASR
        if (!text && s->backend == "funasr" && s->funasr_ctx) {
            if (s->beam_size > 1)
                funasr_set_beam_size(s->funasr_ctx, s->beam_size);
            if (!s->source_language.empty())
                funasr_set_language(s->funasr_ctx, s->source_language.c_str());
            text = funasr_transcribe(s->funasr_ctx, pcm, n_samples);
            need_free = true;
        }
#endif
#ifdef CA_HAVE_PARAFORMER
        if (!text && s->backend == "paraformer" && s->paraformer_ctx) {
            text = paraformer_transcribe(s->paraformer_ctx, pcm, n_samples);
            need_free = true;
        }
#endif
#ifdef CA_HAVE_SENSEVOICE
        if (!text && s->backend == "sensevoice" && s->sensevoice_ctx) {
            if (s->beam_size > 1)
                sensevoice_set_beam_size(s->sensevoice_ctx, s->beam_size, /*gamma=*/2.3f);
            const char* lang = s->source_language.empty() ? nullptr : s->source_language.c_str();
            text = sensevoice_transcribe(s->sensevoice_ctx, pcm, n_samples, lang, /*use_itn=*/true);
            need_free = true;
        }
#endif
#ifdef CA_HAVE_MOONSHINE_STREAMING
        if (!text && s->backend == "moonshine-streaming" && s->moonshine_streaming_ctx) {
            if (s->beam_size > 1)
                moonshine_streaming_set_beam_size((moonshine_streaming_context*)s->moonshine_streaming_ctx,
                                                  s->beam_size);
            moonshine_streaming_result* msr = moonshine_streaming_transcribe_with_probs(
                (moonshine_streaming_context*)s->moonshine_streaming_ctx, pcm, n_samples);
            if (msr && msr->text) {
                std::vector<ca_token_record> toks;
                toks.reserve((size_t)msr->n_tokens);
                for (int i = 0; i < msr->n_tokens; i++) {
                    ca_token_record tk;
                    const char* piece = moonshine_streaming_token_text(
                        (moonshine_streaming_context*)s->moonshine_streaming_ctx, msr->token_ids[i]);
                    if (piece && piece[0])
                        tk.text = piece;
                    tk.t0 = -1;
                    tk.t1 = -1;
                    tk.p = msr->token_probs[i];
                    toks.push_back(std::move(tk));
                }
                char* dup = strdup(msr->text);
                moonshine_streaming_result_free(msr);
                return package_with_tokens(dup, std::move(toks));
            }
            if (msr)
                moonshine_streaming_result_free(msr);
        }
#endif
#ifdef CA_HAVE_GEMMA4_E2B
        if (!text && s->backend == "gemma4-e2b" && s->gemma4_e2b_ctx) {
            if (s->beam_size > 1)
                gemma4_e2b_set_beam_size((gemma4_e2b_context*)s->gemma4_e2b_ctx, s->beam_size);
            gemma4_e2b_set_ask((gemma4_e2b_context*)s->gemma4_e2b_ctx, s->ask.empty() ? nullptr : s->ask.c_str());
            const std::string src = lang_set ? lang : (!s->source_language.empty() ? s->source_language : "");
            const std::string tgt = !s->target_language.empty() ? s->target_language : (s->translate ? "en" : src);
            if (s->translate || (!tgt.empty() && tgt != src)) {
                char* text = gemma4_e2b_transcribe_ex((gemma4_e2b_context*)s->gemma4_e2b_ctx, pcm, n_samples, 1,
                                                      src.c_str(), tgt.c_str());
                if (text) {
                    return package_text_only(text, true);
                }
            } else {
                gemma4_e2b_result* gr =
                    gemma4_e2b_transcribe_with_probs((gemma4_e2b_context*)s->gemma4_e2b_ctx, pcm, n_samples);
                if (gr && gr->text) {
                    std::vector<ca_token_record> toks;
                    toks.reserve((size_t)gr->n_tokens);
                    for (int i = 0; i < gr->n_tokens; i++) {
                        ca_token_record tk;
                        const char* piece =
                            gemma4_e2b_token_text((gemma4_e2b_context*)s->gemma4_e2b_ctx, gr->token_ids[i]);
                        if (piece && piece[0]) {
                            // Gemma uses SentencePiece-style ▁ markers (U+2581).
                            std::string p = piece;
                            for (size_t ci = 0; ci < p.size(); ci++) {
                                if ((unsigned char)p[ci] == 0xE2 && ci + 2 < p.size() &&
                                    (unsigned char)p[ci + 1] == 0x96 && (unsigned char)p[ci + 2] == 0x81) {
                                    tk.text += ' ';
                                    ci += 2;
                                } else {
                                    tk.text += p[ci];
                                }
                            }
                        }
                        tk.t0 = -1;
                        tk.t1 = -1;
                        tk.p = gr->token_probs[i];
                        toks.push_back(std::move(tk));
                    }
                    char* dup = strdup(gr->text);
                    gemma4_e2b_result_free(gr);
                    return package_with_tokens(dup, std::move(toks));
                }
                if (gr)
                    gemma4_e2b_result_free(gr);
            }
        }
#endif
#ifdef CA_HAVE_MIMO_ASR
        if (!text && s->backend == "mimo-asr" && s->mimo_asr_ctx) {
            // mimo_asr returns null + logs to stderr if the tokenizer companion
            // wasn't set via crispasr_session_set_codec_path. We surface a clean
            // "no transcription" rather than hanging.
            mimo_asr_set_ask(s->mimo_asr_ctx, s->ask.empty() ? nullptr : s->ask.c_str());
            mimo_asr_result* mr = mimo_asr_transcribe_with_probs(s->mimo_asr_ctx, pcm, n_samples);
            if (mr && mr->text) {
                std::vector<ca_token_record> toks;
                toks.reserve((size_t)mr->n_tokens);
                for (int i = 0; i < mr->n_tokens; i++) {
                    ca_token_record tk;
                    const char* piece = mimo_asr_token_text(s->mimo_asr_ctx, mr->token_ids[i]);
                    if (piece) {
                        std::string p = piece;
                        // Mimo uses Qwen2 tokenizer (GPT-2 byte-level BPE):
                        // Ġ (0xC4 0xA0) → space, Ċ (0xC4 0x8A) → newline.
                        for (size_t ci = 0; ci < p.size();) {
                            unsigned char c = (unsigned char)p[ci];
                            if (c == 0xC4 && ci + 1 < p.size()) {
                                unsigned char c2 = (unsigned char)p[ci + 1];
                                if (c2 == 0xA0) {
                                    tk.text += ' ';
                                    ci += 2;
                                    continue;
                                }
                                if (c2 == 0x8A) {
                                    tk.text += '\n';
                                    ci += 2;
                                    continue;
                                }
                            }
                            tk.text += (char)c;
                            ci++;
                        }
                    }
                    tk.t0 = -1;
                    tk.t1 = -1;
                    tk.p = mr->token_probs[i];
                    toks.push_back(std::move(tk));
                }
                char* dup = strdup(mr->text);
                mimo_asr_result_free(mr);
                return package_with_tokens(dup, std::move(toks));
            }
            if (mr)
                mimo_asr_result_free(mr);
        }
#endif
#ifdef CA_HAVE_MOSS_AUDIO
        if (!text && s->moss_audio_ctx) {
            const char* prompt = s->ask.empty() ? "Transcribe this audio." : s->ask.c_str();
            text = moss_audio_process(s->moss_audio_ctx, pcm, n_samples, prompt);
            need_free = true;
        }
#endif
        if (text)
            return package_text_only(text, need_free);
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
// silence gaps (crispasr-style), calls crispasr_session_transcribe on
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
    if (!crispasr_detect_language(samples, n_samples, opts, r)) {
        crispasr_lid_free_cache(); // free GPU memory even on failure
        return 1;
    }

    // Free cached LID context to release GPU VRAM for subsequent ASR calls
    crispasr_lid_free_cache();

    if ((int)r.lang_code.size() + 1 > out_lang_cap)
        return 2;
    std::memcpy(out_lang_buf, r.lang_code.c_str(), r.lang_code.size());
    out_lang_buf[r.lang_code.size()] = '\0';
    if (out_confidence)
        *out_confidence = r.confidence;
    return 0;
}

// ---------------------------------------------------------------------------
// RNNoise audio enhancement (transcribe pre-step).
//
// Takes a 16 kHz mono float32 PCM buffer in [-1, 1] and writes the
// denoised result into a caller-allocated output buffer of the same
// length. Internally upsamples to 48 kHz, runs RNNoise's 480-sample
// frame loop, and downsamples back. State is allocated and freed per
// call so concurrent worker isolates can invoke this safely.
//
// Returns:
//   *  0 — success; out_pcm populated with denoised samples.
//   * -1 — invalid args (null pointer, n_samples <= 0, out_cap < n_samples).
//   * -2 — RNNoise init / processing failure (resampler init, etc).
// ---------------------------------------------------------------------------
CA_EXPORT int crispasr_enhance_audio_rnnoise(const float* in_pcm, int32_t n_samples, float* out_pcm, int32_t out_cap) {
#if defined(CRISPASR_RNNOISE)
    if (!in_pcm || !out_pcm || n_samples <= 0 || out_cap < n_samples)
        return -1;

    CrispasrEnhanceOptions opts;
    opts.method = CrispasrEnhanceMethod::Rnnoise;
    opts.verbose = false;

    if (!crispasr_enhance_audio(in_pcm, n_samples, out_pcm, opts))
        return -2;
    return 0;
#else
    (void)in_pcm;
    (void)n_samples;
    (void)out_pcm;
    (void)out_cap;
    return -2; // RNNoise not compiled on this platform
#endif
}

// ---------------------------------------------------------------------------
// Text-LID (P13.5 Phase 7 — downstream consumers' text-LID needs).
//
// Wraps the existing internal `text_lid_dispatch.h` façade (CLD3 +
// GlotLID/LID-176 fastText) as a stable C-ABI export so the Rust /
// Dart / Python bindings can drop the CLI-shell-out fallback they
// were using for text language detection.  Mirrors the audio-side
// `crispasr_detect_language_pcm` shape: caller supplies a model
// path + output buffer, function returns an int status code and
// writes the label + confidence via out-params.
//
// Label format depends on the loaded GGUF:
//   * CLD3 backend (lid-cld3 arch) — ISO 639-1 two-letter codes
//     ('en', 'de', 'zh-Latn') across 109 labels.
//   * fastText backend (lid-fasttext arch) — ISO 639-3 + script
//     ('eng_Latn', 'sco_Latn') across 2102 (GlotLID-V3) or 176
//     (LID-176) labels.
// Callers that need ISO 639-1 normalisation must do it on their side;
// the dispatcher intentionally returns the model's native label space
// to preserve information (CLD3's `zh-Latn` and GlotLID's `eng_Latn`
// both carry script tags that a naive 2-letter normalisation would
// drop).
//
// Returns:
//   *  0 — success; out_label_buf + out_confidence populated.
//   * -1 — invalid args (null pointer, n_threads <= 0, out_cap < 1).
//   *  1 — dispatcher init failure (bad GGUF, unsupported architecture).
//   *  2 — output buffer too small for the predicted label.
CA_EXPORT int crispasr_text_detect_language(const char* text, const char* model_path, int32_t n_threads,
                                            char* out_label_buf, int32_t out_label_cap, float* out_confidence) {
    if (!text || !model_path || !out_label_buf || out_label_cap <= 0)
        return -1;
    if (n_threads <= 0)
        n_threads = 1;

    text_lid_context* ctx = text_lid_init_from_file(model_path, n_threads);
    if (!ctx)
        return 1;

    float conf = 0.0f;
    const char* label = text_lid_predict(ctx, text, &conf);
    if (!label) {
        text_lid_free(ctx);
        return 1;
    }

    const size_t label_len = std::strlen(label);
    if (static_cast<int32_t>(label_len) + 1 > out_label_cap) {
        text_lid_free(ctx);
        return 2;
    }

    std::memcpy(out_label_buf, label, label_len);
    out_label_buf[label_len] = '\0';
    if (out_confidence)
        *out_confidence = conf;

    text_lid_free(ctx);
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

// Write a comma-separated list of every backend name in the registry to
// `out_csv`. Returns the number of bytes written (excluding NUL) on
// success, or a negative error. Wrappers iterate the CSV then call
// crispasr_registry_lookup_abi(name) for full details. Mirrors the
// shape of crispasr_session_available_backends().
CA_EXPORT int crispasr_registry_list_backends_abi(char* out_csv, int32_t out_cap) {
    if (!out_csv || out_cap <= 0)
        return -1;
    std::string acc;
    const int n = crispasr_registry_count();
    for (int i = 0; i < n; i++) {
        CrispasrRegistryEntry e;
        if (!crispasr_registry_get_at(i, e))
            continue;
        if (!acc.empty())
            acc.push_back(',');
        acc += e.backend;
    }
    if ((int)acc.size() + 1 > out_cap)
        return -2;
    std::memcpy(out_csv, acc.data(), acc.size());
    out_csv[acc.size()] = '\0';
    return (int)acc.size();
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
// Per-word probability (confidence) in [0, 1]. Backends that don't
// emit per-word probabilities populate this with 1.0 at construction
// time, so consumers can render uniformly when the backend is silent.
// Returns -1.0 on out-of-range so callers can distinguish "no data"
// from "100% confident".
CA_EXPORT float crispasr_session_result_word_p(crispasr_session_result* r, int i_seg, int i_word) {
    if (!r || i_seg < 0 || i_seg >= (int)r->segments.size())
        return -1.0f;
    auto& ws = r->segments[i_seg].words;
    return (i_word >= 0 && i_word < (int)ws.size()) ? ws[i_word].p : -1.0f;
}

// Top-N alternative candidates for the word's first content token.
// Returns 0 / "" / 0.0f when alts weren't captured (alt_n was 0, the
// backend doesn't produce alts, or indices are out of range). Ordered
// descending by p.
CA_EXPORT int crispasr_session_result_word_n_alts(crispasr_session_result* r, int i_seg, int i_word) {
    if (!r || i_seg < 0 || i_seg >= (int)r->segments.size())
        return 0;
    auto& ws = r->segments[i_seg].words;
    if (i_word < 0 || i_word >= (int)ws.size())
        return 0;
    return (int)ws[i_word].alts.size();
}

CA_EXPORT const char* crispasr_session_result_word_alt_text(crispasr_session_result* r, int i_seg, int i_word,
                                                            int i_alt) {
    if (!r || i_seg < 0 || i_seg >= (int)r->segments.size())
        return "";
    auto& ws = r->segments[i_seg].words;
    if (i_word < 0 || i_word >= (int)ws.size())
        return "";
    auto& alts = ws[i_word].alts;
    if (i_alt < 0 || i_alt >= (int)alts.size())
        return "";
    return alts[i_alt].text.c_str();
}

CA_EXPORT float crispasr_session_result_word_alt_p(crispasr_session_result* r, int i_seg, int i_word, int i_alt) {
    if (!r || i_seg < 0 || i_seg >= (int)r->segments.size())
        return 0.0f;
    auto& ws = r->segments[i_seg].words;
    if (i_word < 0 || i_word >= (int)ws.size())
        return 0.0f;
    auto& alts = ws[i_word].alts;
    if (i_alt < 0 || i_alt >= (int)alts.size())
        return 0.0f;
    return alts[i_alt].p;
}

CA_EXPORT void crispasr_session_result_free(crispasr_session_result* r) {
    if (r)
        delete r;
}

// ---------------------------------------------------------------------------
// TTS synthesis (vibevoice, qwen3-tts)
// ---------------------------------------------------------------------------
//
// `crispasr_session_synthesize` returns malloc'd float32 PCM at 24 kHz mono.
// `*out_n_samples` is set on success. Caller frees with `crispasr_pcm_free`.
// Returns nullptr if the active backend doesn't support TTS or synthesis fails.
//
// `crispasr_session_set_voice` accepts:
//   - a *.gguf voice pack (vibevoice or qwen3-tts), or
//   - a *.wav reference audio. For qwen3-tts the reference transcription is
//     required and goes through `ref_text_or_null`. Pass nullptr for a
//     voice pack.
//
// `crispasr_session_set_codec_path` forwards the codec GGUF path to the
// active backend (qwen3-tts, orpheus, zonos, dia, tada, outetts, indextts).
// For Zonos and Dia the codec is auto-discovered as a sibling on open;
// call this only to override the discovered path.

CA_EXPORT int crispasr_session_set_codec_path(crispasr_session* s, const char* path) {
    if (!s || !path)
        return -1;
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx)
        return qwen3_tts_set_codec_path(s->qwen3_tts_ctx, path);
#endif
#ifdef CA_HAVE_ORPHEUS
    if (s->orpheus_ctx) {
        int rc = orpheus_set_codec_path(s->orpheus_ctx, path);
        if (rc == 0)
            s->orpheus_codec_loaded = true;
        return rc;
    }
#endif
#ifdef CA_HAVE_MIMO_ASR
    // mimo-asr needs the mimo_tokenizer companion before transcribe
    // can run. Reuse the same setter the other companion-needing
    // backends do — call right after open(), pass mimo-tokenizer-*.gguf.
    if (s->mimo_asr_ctx)
        return mimo_asr_set_tokenizer_path(s->mimo_asr_ctx, path);
#endif
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx)
        return chatterbox_set_s3gen_path(s->chatterbox_ctx, path);
#endif
#ifdef CA_HAVE_TADA
    if (s->tada_ctx)
        return tada_set_codec_path(s->tada_ctx, path);
#endif
#ifdef CA_HAVE_OUTETTS
    if (s->outetts_ctx)
        return outetts_set_codec_path(s->outetts_ctx, path);
#endif
#ifdef CA_HAVE_ZONOS
    if (s->zonos_ctx) {
        zonos_tts_set_codec_path(s->zonos_ctx, path);
        return 0;
    }
#endif
#ifdef CA_HAVE_DIA
    if (s->dia_tts_ctx) {
        dia_tts_set_codec_path(s->dia_tts_ctx, path);
        return 0;
    }
#endif
#ifdef CA_HAVE_INDEXTTS
    // indextts routes its BigVGAN vocoder companion (indextts-bigvgan)
    // through the shared codec-path setter, same as qwen3-tts/orpheus.
    if (s->indextts_ctx)
        return indextts_set_vocoder_path(s->indextts_ctx, path);
#endif
    return 0; // not applicable
}

#if defined(CA_HAVE_INDEXTTS) || defined(CA_HAVE_VOXCPM2) || defined(CA_HAVE_POCKET)
// crispasr_audio_load lives in crispasr_audio.cpp (same shared lib);
// forward-declare it so set_voice can decode a reference WAV without
// pulling in the audio header. Returns 16 kHz mono f32 — voxcpm2 uses it
// directly; indextts upsamples to 24 kHz below; pocket-tts needs 24 kHz
// (the Mimi encoder expects 24 kHz, but crispasr_audio_load returns 16 kHz
// — pocket_tts_set_voice handles the resample internally if needed).
extern "C" int crispasr_audio_load(const char* path, float** out_pcm, int* out_samples, int* out_sample_rate);
#endif

#ifdef CA_HAVE_INDEXTTS
// Linear-resample 16 kHz → 24 kHz (3:2). indextts's ECAPA speaker
// encoder + conditioning mel expect a 24 kHz reference ("resampled by
// the backend caller", indextts_voc.cpp); the shared decoder only emits
// 16 kHz, so upsample here before handing the clip to indextts.
static std::vector<float> indextts_resample_16k_to_24k(const float* in, int n) {
    std::vector<float> out;
    if (!in || n <= 0)
        return out;
    const int outN = (int)((int64_t)n * 24000 / 16000);
    out.resize(outN);
    for (int j = 0; j < outN; ++j) {
        const double srcPos = (double)j * 16000.0 / 24000.0;
        const int i0 = (int)srcPos;
        const int i1 = (i0 + 1 < n) ? i0 + 1 : n - 1;
        const double frac = srcPos - (double)i0;
        out[j] = (float)((double)in[i0] * (1.0 - frac) + (double)in[i1] * frac);
    }
    return out;
}
#endif

CA_EXPORT int crispasr_session_set_voice(crispasr_session* s, const char* path, const char* ref_text_or_null) {
    if (!s || !path)
        return -1;
    auto ends_with_wav = [](const char* p) {
        size_t n = std::strlen(p);
        if (n < 4)
            return false;
        const char* tail = p + n - 4;
        return (tail[0] == '.' && (tail[1] == 'w' || tail[1] == 'W') && (tail[2] == 'a' || tail[2] == 'A') &&
                (tail[3] == 'v' || tail[3] == 'V'));
    };
#ifdef CA_HAVE_COSYVOICE3
    if (s->cosyvoice3_ctx) {
        // `path` is either a baked-bank voice name (e.g. "fleurs-en") or a
        // *.wav clone reference; for a WAV, ref_text_or_null is required and
        // is consumed at synthesize time by cosyvoice3_tts_synth_from_wav.
        s->cosyvoice3_voice = path;
        s->cosyvoice3_ref_text = ref_text_or_null ? ref_text_or_null : "";
        if (ends_with_wav(path) && s->cosyvoice3_ref_text.empty())
            return -2; // WAV cloning needs a reference transcription
        return 0;
    }
#endif
#ifdef CA_HAVE_VIBEVOICE
    if (s->vibevoice_ctx) {
        if (ends_with_wav(path)) {
            // 1.5B/7B base model: WAV reference → env var for vibevoice_synthesize
#if defined(_WIN32)
            _putenv_s("VIBEVOICE_VOICE_AUDIO", path);
#else
            setenv("VIBEVOICE_VOICE_AUDIO", path, 1);
#endif
            return 0;
        }
        return vibevoice_load_voice(s->vibevoice_ctx, path);
    }
#endif
#ifdef CA_HAVE_KUGELAUDIO
    if (s->kugelaudio_ctx) {
        return kugelaudio_load_voice(s->kugelaudio_ctx, path);
    }
#endif
#ifdef CA_HAVE_ZONOS
    if (s->zonos_ctx) {
        return zonos_tts_set_voice(s->zonos_ctx, path);
    }
#endif
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx) {
        if (ends_with_wav(path)) {
            if (!ref_text_or_null)
                return -2;
            int rc = qwen3_tts_set_voice_prompt_with_text(s->qwen3_tts_ctx, path, ref_text_or_null);
            if (rc == 0)
                s->qwen3_tts_voice_loaded = true;
            return rc;
        }
        int rc = qwen3_tts_load_voice_pack(s->qwen3_tts_ctx, path);
        if (rc == 0)
            s->qwen3_tts_voice_loaded = true;
        return rc;
    }
#endif
#ifdef CA_HAVE_KOKORO
    if (s->kokoro_ctx) {
        // Kokoro voicepacks are GGUF only; .wav reference audio is not
        // a thing for this backend. ref_text_or_null is ignored.
        return kokoro_load_voice_pack(s->kokoro_ctx, path);
    }
#endif
#ifdef CA_HAVE_SPEECHT5
    if (s->speecht5_ctx) {
        // SpeechT5 needs a 512-d x-vector (raw float32 .bin file).
        if (ends_with_wav(path))
            return -2;
        FILE* f = fopen(path, "rb");
        if (!f)
            return -1;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        int n = (int)(sz / sizeof(float));
        s->speecht5_speaker.resize(n);
        size_t rd = fread(s->speecht5_speaker.data(), sizeof(float), n, f);
        fclose(f);
        if ((int)rd != n)
            return -1;
        return speecht5_tts_set_speaker(s->speecht5_ctx, s->speecht5_speaker.data(), n);
    }
#endif
#ifdef CA_HAVE_INDEXTTS
    if (s->indextts_ctx) {
        // indextts clones from a reference clip. Decode the WAV (16 kHz
        // mono via the shared loader), upsample to the 24 kHz the encoder
        // expects, and stash it for the next synthesize. ref_text is
        // unused — indextts conditions on audio, not a transcript.
        if (!ends_with_wav(path))
            return -2;
        float* pcm = nullptr;
        int n = 0, sr = 0;
        if (crispasr_audio_load(path, &pcm, &n, &sr) != 0 || !pcm || n <= 0) {
            if (pcm)
                free(pcm);
            return -1;
        }
        s->indextts_ref_pcm = indextts_resample_16k_to_24k(pcm, n);
        free(pcm);
        return s->indextts_ref_pcm.empty() ? -1 : 0;
    }
#endif
#ifdef CA_HAVE_F5TTS
    if (s->f5tts_ctx) {
        // F5-TTS clones from a reference WAV + its transcript. Load the
        // audio (16 kHz from shared loader), resample to 24 kHz, and pass
        // to the library which computes the mel spectrogram internally.
        if (!ends_with_wav(path))
            return -2;
        float* pcm = nullptr;
        int n = 0, sr = 0;
        if (crispasr_audio_load(path, &pcm, &n, &sr) != 0 || !pcm || n <= 0) {
            if (pcm)
                free(pcm);
            return -1;
        }
        // Resample 16 kHz → 24 kHz (linear interp)
        int n24 = (int)((float)n * 24000.0f / 16000.0f);
        std::vector<float> pcm24(n24);
        for (int i = 0; i < n24; i++) {
            float pos = (float)i * 16000.0f / 24000.0f;
            int idx = (int)pos;
            float frac = pos - (float)idx;
            pcm24[i] = (idx + 1 < n) ? pcm[idx] * (1 - frac) + pcm[idx + 1] * frac : pcm[std::min(idx, n - 1)];
        }
        free(pcm);
        // RMS normalize to 0.1
        float rms = 0;
        for (float v : pcm24)
            rms += v * v;
        rms = sqrtf(rms / (float)pcm24.size());
        if (rms < 0.1f && rms > 1e-10f) {
            float s2 = 0.1f / rms;
            for (float& v : pcm24)
                v *= s2;
        }
        const char* rt = ref_text_or_null ? ref_text_or_null : "";
        int rc = f5_tts_set_reference(s->f5tts_ctx, pcm24.data(), n24, rt);
        return (rc == 0) ? 0 : -1;
    }
#endif
#ifdef CA_HAVE_VOXCPM2
    if (s->voxcpm2_ctx) {
        // VoxCPM2 zero-shot voice cloning: stash a 16 kHz mono reference
        // clip (exactly what voxcpm2_synthesize_clone wants, so no
        // resample) for the next synthesize. ref_text is unused.
        if (!ends_with_wav(path))
            return -2;
        float* pcm = nullptr;
        int n = 0, sr = 0;
        if (crispasr_audio_load(path, &pcm, &n, &sr) != 0 || !pcm || n <= 0) {
            if (pcm)
                free(pcm);
            return -1;
        }
        s->voxcpm2_ref_pcm.assign(pcm, pcm + n);
        free(pcm);
        return s->voxcpm2_ref_pcm.empty() ? -1 : 0;
    }
#endif
#ifdef CA_HAVE_POCKET
    if (s->pocket_tts_ctx) {
        if (!ends_with_wav(path))
            return -2;
        float* pcm = nullptr;
        int n = 0, sr = 0;
        if (crispasr_audio_load(path, &pcm, &n, &sr) != 0 || !pcm || n <= 0) {
            if (pcm)
                free(pcm);
            return -1;
        }
        // crispasr_audio_load returns 16 kHz mono; pocket_tts_set_voice
        // expects 24 kHz (Mimi encoder native rate). Resample 16→24 kHz
        // with linear interpolation.
        if (sr <= 0)
            sr = 16000;
        int rc;
        if (sr != 24000) {
            const int n24 = (int)((int64_t)n * 24000 / sr);
            float* pcm24 = (float*)malloc((size_t)(n24 > 0 ? n24 : 1) * sizeof(float));
            if (!pcm24) {
                free(pcm);
                return -1;
            }
            const double ratio = (double)sr / 24000.0;
            for (int j = 0; j < n24; ++j) {
                const double pos = (double)j * ratio;
                const int i0 = (int)pos;
                const int i1 = (i0 + 1 < n) ? i0 + 1 : n - 1;
                const double frac = pos - (double)i0;
                pcm24[j] = (float)((double)pcm[i0] * (1.0 - frac) + (double)pcm[i1] * frac);
            }
            free(pcm);
            rc = pocket_tts_set_voice(s->pocket_tts_ctx, pcm24, n24);
            free(pcm24);
        } else {
            rc = pocket_tts_set_voice(s->pocket_tts_ctx, pcm, n);
            free(pcm);
        }
        return rc;
    }
#endif
#ifdef CA_HAVE_OUTETTS
    if (s->outetts_ctx) {
        // OuteTTS speaker profiles are JSON files (not WAV).
        return outetts_load_speaker(s->outetts_ctx, path);
    }
#endif
    return -3;
}

// Select a fixed/preset speaker by NAME for backends that bake speakers
// into the GGUF (orpheus + qwen3-tts CustomVoice today).
//
// For orpheus, canonical names are "tara"/"leo"/"leah" etc. for the
// canopylabs English finetune; "Anton"/"Sophie" etc. for the
// SebastianBodza/Kartoffel_Orpheus DE finetunes. For qwen3-tts
// CustomVoice the names are the 9 baked speakers (vivian, aiden,
// dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu). Enumerate
// at runtime via crispasr_session_n_speakers /
// crispasr_session_get_speaker_name.
//
// Returns 0 on success, -1 if the session isn't valid, -2 if the name
// is unknown for the active backend, -3 if the active backend has no
// preset-speaker contract.
CA_EXPORT int crispasr_session_set_speaker_name(crispasr_session* s, const char* name) {
    if (!s || !name)
        return -1;
#ifdef CA_HAVE_ORPHEUS
    if (s->orpheus_ctx) {
        return orpheus_set_speaker_by_name(s->orpheus_ctx, name);
    }
#endif
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx && qwen3_tts_is_custom_voice(s->qwen3_tts_ctx)) {
        int rc = qwen3_tts_set_speaker_by_name(s->qwen3_tts_ctx, name);
        if (rc == 0)
            s->qwen3_tts_voice_loaded = true;
        return rc;
    }
#endif
    return -3;
}

// Select a speaker by integer index for multi-speaker TTS backends
// (melotts, piper, fastpitch). Index is 0-based; valid range is
// [0, crispasr_session_n_speakers() - 1].
//
// Returns 0 on success, -1 if the session isn't valid, -2 if the id
// is out of range, -3 if the active backend has no integer-speaker
// contract.
CA_EXPORT int crispasr_session_set_speaker_id(crispasr_session* s, int id) {
    if (!s)
        return -1;
    if (id < 0)
        return -2;
#ifdef CA_HAVE_MELOTTS
    if (s->melotts_ctx) {
        if (id >= melotts_num_speakers(s->melotts_ctx))
            return -2;
        melotts_set_speaker_id(s->melotts_ctx, id);
        return 0;
    }
#endif
#ifdef CA_HAVE_PIPER
    if (s->piper_ctx) {
        if (id >= piper_tts_num_speakers(s->piper_ctx))
            return -2;
        piper_tts_set_speaker_id(s->piper_ctx, id);
        return 0;
    }
#endif
#ifdef CA_HAVE_FASTPITCH
    if (s->fastpitch_ctx) {
        if (id >= fastpitch_tts_n_speakers(s->fastpitch_ctx))
            return -2;
        fastpitch_tts_set_speaker(s->fastpitch_ctx, id);
        return 0;
    }
#endif
    return -3;
}

// Number of fixed/preset speakers baked into the active backend's GGUF.
// Returns 0 if the backend has no preset speakers (or isn't loaded).
CA_EXPORT int crispasr_session_n_speakers(crispasr_session* s) {
    if (!s)
        return 0;
#ifdef CA_HAVE_ORPHEUS
    if (s->orpheus_ctx)
        return orpheus_n_speakers(s->orpheus_ctx);
#endif
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx && qwen3_tts_is_custom_voice(s->qwen3_tts_ctx))
        return qwen3_tts_n_speakers(s->qwen3_tts_ctx);
#endif
#ifdef CA_HAVE_MELOTTS
    if (s->melotts_ctx)
        return melotts_num_speakers(s->melotts_ctx);
#endif
#ifdef CA_HAVE_PIPER
    if (s->piper_ctx)
        return piper_tts_num_speakers(s->piper_ctx);
#endif
#ifdef CA_HAVE_FASTPITCH
    if (s->fastpitch_ctx)
        return fastpitch_tts_n_speakers(s->fastpitch_ctx);
#endif
    return 0;
}

// Returns the i-th preset speaker name (0-indexed). Buffer is owned by
// the session; do not free. Returns nullptr for out-of-range indices
// or backends without preset speakers.
CA_EXPORT const char* crispasr_session_get_speaker_name(crispasr_session* s, int i) {
    if (!s || i < 0)
        return nullptr;
#ifdef CA_HAVE_ORPHEUS
    if (s->orpheus_ctx)
        return orpheus_get_speaker_name(s->orpheus_ctx, i);
#endif
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx && qwen3_tts_is_custom_voice(s->qwen3_tts_ctx))
        return qwen3_tts_get_speaker_name(s->qwen3_tts_ctx, i);
#endif
    return nullptr;
}

// Set the natural-language voice description for instruct-tuned TTS
// backends (qwen3-tts VoiceDesign today). Required before
// crispasr_session_synthesize when the loaded backend is VoiceDesign.
//
// Returns 0 on success, -1 on invalid args, -3 if the active backend
// has no instruct contract (or isn't a VoiceDesign variant).
CA_EXPORT int crispasr_session_set_instruct(crispasr_session* s, const char* instruct) {
    if (!s || !instruct)
        return -1;
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx && qwen3_tts_is_voice_design(s->qwen3_tts_ctx)) {
        int rc = qwen3_tts_set_instruct(s->qwen3_tts_ctx, instruct);
        if (rc == 0)
            s->qwen3_tts_voice_loaded = true;
        return rc;
    }
#endif
#ifdef CA_HAVE_PARLER_TTS
    if (s->parler_tts_ctx) {
        s->parler_description = instruct;
        return parler_tts_set_description(s->parler_tts_ctx, instruct);
    }
#endif
    return -3;
}

// Variant detection for the qwen3-tts backend. Returns 0/1; 0 also
// covers "active backend isn't qwen3-tts". Lets wrappers branch on
// which voice-prompt API to call (`set_voice` vs `set_speaker_name`
// vs `set_instruct`) without parsing GGUF metadata themselves.
CA_EXPORT int crispasr_session_is_custom_voice(crispasr_session* s) {
    if (!s)
        return 0;
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx)
        return qwen3_tts_is_custom_voice(s->qwen3_tts_ctx);
#endif
    return 0;
}

CA_EXPORT int crispasr_session_is_voice_design(crispasr_session* s) {
    if (!s)
        return 0;
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx)
        return qwen3_tts_is_voice_design(s->qwen3_tts_ctx);
#endif
    return 0;
}

// Raw synthesis — no watermark. Used internally; the public API wraps this
// and applies the watermark automatically.
static float* crispasr_session_synthesize_raw_impl(crispasr_session* s, const char* text, int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!s || !text)
        return nullptr;
    s->last_synth_error.clear();
#ifdef CA_HAVE_COSYVOICE3
    if (s->cosyvoice3_ctx) {
        // Voice is a bank name (default zero_shot) unless set_voice supplied a
        // *.wav path, in which case clone from it (needs cosyvoice3_ref_text).
        // Output is already malloc'd float32 @ 24 kHz mono — the C API contract.
        const std::string& v = s->cosyvoice3_voice;
        const bool is_wav =
            v.size() >= 4 && (v.compare(v.size() - 4, 4, ".wav") == 0 || v.compare(v.size() - 4, 4, ".WAV") == 0);
        int n = 0;
        float* pcm = is_wav ? cosyvoice3_tts_synth_from_wav(s->cosyvoice3_ctx, text, v.c_str(),
                                                            s->cosyvoice3_ref_text.c_str(), &n)
                            : cosyvoice3_tts_synth(s->cosyvoice3_ctx, text, v.empty() ? nullptr : v.c_str(), &n);
        if (!pcm || n <= 0) {
            if (pcm)
                free(pcm);
            return nullptr;
        }
        if (out_n_samples)
            *out_n_samples = n;
        return pcm;
    }
#endif
#ifdef CA_HAVE_VIBEVOICE
    if (s->vibevoice_ctx) {
        return vibevoice_synthesize(s->vibevoice_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_KUGELAUDIO
    if (s->kugelaudio_ctx) {
        return kugelaudio_synthesize(s->kugelaudio_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_ZONOS
    if (s->zonos_ctx) {
        return zonos_tts_synthesize(s->zonos_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx) {
        // Pre-flight: surface a clear error when the Base model is used
        // without a voice reference — qwen3_tts_synthesize_codes prints
        // to stderr (invisible on Android) and returns nullptr, which the
        // Dart side can only report as a generic "no audio produced".
        if (!s->qwen3_tts_voice_loaded) {
            const bool is_cv = qwen3_tts_is_custom_voice(s->qwen3_tts_ctx);
            const bool is_vd = qwen3_tts_is_voice_design(s->qwen3_tts_ctx);
            if (!is_cv && !is_vd) {
                s->last_synth_error = "qwen3-tts Base requires a voice — "
                                      "select a voice pack or reference WAV in the voice picker";
                return nullptr;
            }
        }
        float* pcm = qwen3_tts_synthesize(s->qwen3_tts_ctx, text, out_n_samples);
        if (!pcm && s->last_synth_error.empty()) {
            s->last_synth_error = "qwen3-tts synthesis failed — "
                                  "try q8_0 quantisation or a different model variant";
        }
        return pcm;
    }
#endif
#ifdef CA_HAVE_ORPHEUS
    if (s->orpheus_ctx) {
        if (!s->orpheus_codec_loaded) {
            s->last_synth_error = "orpheus requires the SNAC codec — "
                                  "download the codec companion model";
            return nullptr;
        }
        return orpheus_synthesize(s->orpheus_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_KOKORO
    if (s->kokoro_ctx) {
        float* pcm = kokoro_synthesize(s->kokoro_ctx, text, out_n_samples);
        if (!pcm && s->last_synth_error.empty()) {
            s->last_synth_error = "kokoro synthesis failed — "
                                  "this is usually because the built-in phonemizer could not "
                                  "process the text (check that a voice pack is loaded)";
        }
        return pcm;
    }
#endif
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx) {
        return chatterbox_synthesize(s->chatterbox_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_TADA
    if (s->tada_ctx) {
        return tada_synthesize(s->tada_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_LFM2_AUDIO
    if (s->lfm2_audio_ctx) {
        return lfm2_audio_synthesize(s->lfm2_audio_ctx, text, nullptr, out_n_samples);
    }
#endif
#ifdef CA_HAVE_MINI_OMNI2
    if (s->mini_omni2_ctx) {
        return mini_omni2_synthesize(s->mini_omni2_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_CSM
    if (s->csm_tts_ctx) {
        // CSM emits 24 kHz mono float — same convention as the other TTS
        // backends; PCM is malloc'd and freed via crispasr_pcm_free.
        return csm_tts_synthesize(s->csm_tts_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_DIA
    if (s->dia_tts_ctx) {
        // Dia emits 44.1 kHz mono float (DAC codec); PCM is malloc'd and freed
        // via crispasr_pcm_free, same as the other TTS backends.
        return dia_tts_synthesize(s->dia_tts_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_POCKET
    if (s->pocket_tts_ctx) {
        return pocket_tts_synthesize(s->pocket_tts_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_FASTPITCH
    if (s->fastpitch_ctx) {
        // FastPitch emits 22050 Hz mono float; deterministic (no sampling).
        float* pcm = nullptr;
        int sr = 0;
        int n = fastpitch_tts_synthesize(s->fastpitch_ctx, text, &pcm, &sr);
        if (n <= 0 || !pcm)
            return nullptr;
        if (out_n_samples)
            *out_n_samples = n;
        return pcm;
    }
#endif
#ifdef CA_HAVE_PARLER
    if (s->parler_ctx) {
        // Parler TTS emits 24 kHz mono float; PCM is malloc'd, freed via crispasr_pcm_free.
        return parler_tts_synthesize(s->parler_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_SPEECHT5
    if (s->speecht5_ctx) {
        // SpeechT5 emits 16 kHz mono float; PCM is malloc'd, freed via crispasr_pcm_free.
        return speecht5_tts_synthesize(s->speecht5_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_BARK
    if (s->bark_ctx) {
        return bark_synthesize(s->bark_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_OUTETTS
    if (s->outetts_ctx) {
        return outetts_synthesize(s->outetts_ctx, text, out_n_samples);
    }
#endif
#ifdef CA_HAVE_VOXCPM2
    if (s->voxcpm2_ctx) {
        // VoxCPM2 synthesises at 48 kHz mono; every other CrispASR TTS
        // backend (and the Dart `synthesize` contract) emits 24 kHz.
        // Decimate 2:1 with a pairwise average — a cheap half-band low
        // pass — so the host's fixed-24 kHz playback path stays correct.
        // When a 16 kHz reference was set via set_voice, clone that voice;
        // otherwise fall back to the zero-shot default speaker.
        int n48 = 0;
        float* pcm48 = s->voxcpm2_ref_pcm.empty()
                           ? voxcpm2_synthesize(s->voxcpm2_ctx, text, &n48)
                           : voxcpm2_synthesize_clone(s->voxcpm2_ctx, text, s->voxcpm2_ref_pcm.data(),
                                                      (int)s->voxcpm2_ref_pcm.size(), &n48);
        if (!pcm48 || n48 <= 0) {
            if (pcm48)
                voxcpm2_pcm_free(pcm48);
            return nullptr;
        }
        const int n24 = n48 / 2;
        float* pcm24 = (float*)malloc((size_t)(n24 > 0 ? n24 : 1) * sizeof(float));
        if (!pcm24) {
            voxcpm2_pcm_free(pcm48);
            return nullptr;
        }
        for (int i = 0; i < n24; ++i)
            pcm24[i] = 0.5f * (pcm48[2 * i] + pcm48[2 * i + 1]);
        voxcpm2_pcm_free(pcm48);
        if (out_n_samples)
            *out_n_samples = n24;
        return pcm24;
    }
#endif
#ifdef CA_HAVE_INDEXTTS
    if (s->indextts_ctx) {
        // indextts outputs 24 kHz mono f32 (BigVGAN) — already the host's
        // playback rate, no resample needed. The cloning reference (24 kHz,
        // set via set_voice) is passed through; with none, indextts falls
        // back to dummy conditioning. Buffer is malloc'd / crispasr_pcm_free
        // compatible, same convention as kokoro / chatterbox.
        const float* ref = s->indextts_ref_pcm.empty() ? nullptr : s->indextts_ref_pcm.data();
        const int refN = (int)s->indextts_ref_pcm.size();
        return indextts_synthesize(s->indextts_ctx, text, ref, refN, out_n_samples);
    }
#endif
#ifdef CA_HAVE_F5TTS
    if (s->f5tts_ctx) {
        // F5-TTS outputs 24 kHz mono. Reference was already set via set_voice.
        float* pcm = nullptr;
        int sr = 0;
        int n = f5_tts_synthesize(s->f5tts_ctx, text, &pcm, &sr);
        if (n <= 0 || !pcm)
            return nullptr;
        *out_n_samples = n;
        return pcm;
    }
#endif
#ifdef CA_HAVE_PARLER_TTS
    if (s->parler_tts_ctx) {
        // Parler outputs 44.1 kHz mono. Resample to 24 kHz for the session
        // contract (Dart/Python playback path assumes 24 kHz).
        // Temperature/seed set at open time or via set_temperature/set_tts_seed
        int n44 = 0;
        float* pcm44 = parler_tts_synthesize(s->parler_tts_ctx, text, &n44);
        if (!pcm44 || n44 <= 0) {
            if (pcm44)
                parler_tts_pcm_free(pcm44);
            return nullptr;
        }
        // 44100 → 24000 linear interpolation
        const int64_t nOut = (int64_t)n44 * 24000 / 44100;
        float* dst = (float*)malloc((size_t)(nOut > 0 ? nOut : 1) * sizeof(float));
        if (!dst) {
            parler_tts_pcm_free(pcm44);
            return nullptr;
        }
        const double ratio = 44100.0 / 24000.0;
        for (int64_t j = 0; j < nOut; ++j) {
            const double pos = (double)j * ratio;
            const int64_t i0 = (int64_t)pos;
            const int64_t i1 = (i0 + 1 < n44) ? i0 + 1 : n44 - 1;
            const double frac = pos - (double)i0;
            dst[j] = (float)((double)pcm44[i0] * (1.0 - frac) + (double)pcm44[i1] * frac);
        }
        parler_tts_pcm_free(pcm44);
        if (out_n_samples)
            *out_n_samples = (int)nOut;
        return dst;
    }
#endif
#ifdef CA_HAVE_PIPER
    if (s->piper_ctx) {
        // Piper synthesises at 22.05 kHz mono; the session synth contract
        // (and the Dart `synthesize` path) is a fixed 24 kHz. Resample with
        // linear interpolation so the host's 24 kHz playback stays correct.
        float* src = nullptr;
        int sr = 0;
        const int nIn = piper_tts_synthesize(s->piper_ctx, text, &src, &sr);
        if (!src || nIn <= 0) {
            if (src)
                free(src);
            return nullptr;
        }
        if (sr <= 0)
            sr = 22050;
        if (sr == 24000) {
            if (out_n_samples)
                *out_n_samples = nIn;
            return src; // already the host rate — pass through
        }
        const int64_t nOut = (int64_t)nIn * 24000 / sr;
        float* dst = (float*)malloc((size_t)(nOut > 0 ? nOut : 1) * sizeof(float));
        if (!dst) {
            free(src);
            return nullptr;
        }
        const double ratio = (double)sr / 24000.0;
        for (int64_t j = 0; j < nOut; ++j) {
            const double pos = (double)j * ratio;
            const int64_t i0 = (int64_t)pos;
            const int64_t i1 = (i0 + 1 < nIn) ? i0 + 1 : nIn - 1;
            const double frac = pos - (double)i0;
            dst[j] = (float)((double)src[i0] * (1.0 - frac) + (double)src[i1] * frac);
        }
        free(src);
        if (out_n_samples)
            *out_n_samples = (int)nOut;
        return dst;
    }
#endif
#ifdef CA_HAVE_MELOTTS
    if (s->melotts_ctx) {
        // MeloTTS synthesises at 44.1 kHz; resample to 24 kHz for session contract.
        float* src = nullptr;
        int sr = 0;
        const int nIn = melotts_synthesize(s->melotts_ctx, text, &src, &sr);
        if (!src || nIn <= 0) {
            if (src)
                free(src);
            return nullptr;
        }
        if (sr <= 0)
            sr = 44100;
        const int64_t nOut = (int64_t)nIn * 24000 / sr;
        float* dst = (float*)malloc((size_t)(nOut > 0 ? nOut : 1) * sizeof(float));
        if (!dst) {
            free(src);
            return nullptr;
        }
        const double ratio = (double)sr / 24000.0;
        for (int64_t j = 0; j < nOut; ++j) {
            const double pos = (double)j * ratio;
            const int64_t i0 = (int64_t)pos;
            const int64_t i1 = (i0 + 1 < nIn) ? i0 + 1 : nIn - 1;
            const double frac = pos - (double)i0;
            dst[j] = (float)((double)src[i0] * (1.0 - frac) + (double)src[i1] * frac);
        }
        melotts_pcm_free(src);
        if (out_n_samples)
            *out_n_samples = (int)nOut;
        return dst;
    }
#endif
    return nullptr;
}

// Synthesize without watermark — for callers that need DSP (speed change,
// mixing, concatenation) before embedding the watermark themselves via
// crispasr_watermark_embed(). Most callers should use
// crispasr_session_synthesize() instead, which auto-watermarks.
CA_EXPORT float* crispasr_session_synthesize_raw(crispasr_session* s, const char* text, int* out_n_samples) {
    return crispasr_session_synthesize_raw_impl(s, text, out_n_samples);
}

// Synthesize + auto-watermark. The default API — all TTS output is
// watermarked for EU AI Act provenance compliance. Use synthesize_raw()
// only when you need to post-process PCM before watermarking.
CA_EXPORT float* crispasr_session_synthesize(crispasr_session* s, const char* text, int* out_n_samples) {
    float* pcm = crispasr_session_synthesize_raw_impl(s, text, out_n_samples);
    if (pcm && out_n_samples && *out_n_samples > 0) {
        crispasr_watermark_embed(pcm, *out_n_samples, 0.005f);
    }
    return pcm;
}

CA_EXPORT void crispasr_pcm_free(float* pcm) {
    free(pcm);
}

// =========================================================================
// Speech-to-Speech — audio in → audio out via a single model pass.
// =========================================================================

CA_EXPORT float* crispasr_session_speech_to_speech(crispasr_session* s, const float* in_samples, int n_in_samples,
                                                   char** out_text, int* out_n_samples) {
    if (!s || !in_samples || n_in_samples <= 0)
        return nullptr;
    if (out_n_samples)
        *out_n_samples = 0;
    if (out_text)
        *out_text = nullptr;

#ifdef CA_HAVE_LFM2_AUDIO
    if (s->lfm2_audio_ctx) {
        char* text = nullptr;
        int n = 0;
        float* pcm =
            lfm2_audio_speech_to_speech(s->lfm2_audio_ctx, in_samples, n_in_samples,
                                        s->source_language.empty() ? nullptr : s->source_language.c_str(), &text, &n);
        if (out_n_samples)
            *out_n_samples = n;
        if (out_text)
            *out_text = text;
        else
            free(text);
        return pcm;
    }
#endif
#ifdef CA_HAVE_MINI_OMNI2
    if (s->mini_omni2_ctx) {
        char* text = nullptr;
        int n = 0;
        float* pcm = mini_omni2_speech_to_speech(s->mini_omni2_ctx, in_samples, n_in_samples, &text, &n);
        if (out_n_samples)
            *out_n_samples = n;
        if (out_text)
            *out_text = text;
        else
            free(text);
        return pcm;
    }
#endif

    s->last_synth_error = "backend '" + s->backend + "' does not support speech-to-speech";
    return nullptr;
}

// =========================================================================
// Hotwords / contextual biasing — session-level setter.
// =========================================================================

CA_EXPORT int crispasr_session_set_hotwords(crispasr_session* s, const char* hotwords, float boost) {
    if (!s)
        return -1;
    s->hotwords = hotwords ? hotwords : "";
    s->hotwords_boost = boost > 0.0f ? boost : 1.5f;

    // For parakeet CTC/TDT, apply immediately to the trie.
#ifdef CA_HAVE_PARAKEET
    if (s->parakeet_ctx) {
        if (s->hotwords.empty()) {
            parakeet_set_hotwords(s->parakeet_ctx, nullptr, 0, 0.0f);
        } else {
            // Parse comma-separated hotwords into an array of C strings.
            std::vector<std::string> hw_strings;
            std::istringstream iss(s->hotwords);
            std::string token;
            while (std::getline(iss, token, ',')) {
                // Trim whitespace.
                size_t start = token.find_first_not_of(" \t");
                size_t end = token.find_last_not_of(" \t");
                if (start != std::string::npos)
                    hw_strings.push_back(token.substr(start, end - start + 1));
            }
            std::vector<const char*> ptrs;
            ptrs.reserve(hw_strings.size());
            for (auto& w : hw_strings)
                ptrs.push_back(w.c_str());
            parakeet_set_hotwords(s->parakeet_ctx, ptrs.data(), (int)ptrs.size(), s->hotwords_boost);
        }
    }
#endif

    // For LLM backends, hotwords are injected into the ask prompt at
    // transcribe time. The stored string is consumed in the transcribe
    // dispatch path via s->hotwords. No immediate action needed.
    return 0;
}

// Returns a human-readable error description when the last synthesize call
// returned nullptr. Empty string when the last call succeeded or no error
// detail is available. The returned pointer is owned by the session — valid
// until the next synthesize call or session close.
CA_EXPORT const char* crispasr_session_last_synth_error(crispasr_session* s) {
    if (!s)
        return "";
    return s->last_synth_error.c_str();
}

// =========================================================================
// Translation API — M2M-100 text-to-text translation
// =========================================================================

CA_EXPORT char* crispasr_session_translate_text(crispasr_session* s, const char* text, const char* src_lang,
                                                const char* tgt_lang, int max_tokens) {
    if (!s || !text || !src_lang || !tgt_lang)
        return nullptr;
#ifdef CA_HAVE_GEMMA4_E2B
    if (s->gemma4_e2b_ctx)
        return gemma4_e2b_translate_text((gemma4_e2b_context*)s->gemma4_e2b_ctx, text, src_lang, tgt_lang);
#endif
#ifdef CA_HAVE_M2M100
    if (s->m2m100_ctx) {
        if (s->beam_size > 1)
            m2m100_set_beam_size(s->m2m100_ctx, s->beam_size);
        return m2m100_translate(s->m2m100_ctx, text, src_lang, tgt_lang, max_tokens > 0 ? max_tokens : 200);
    }
#endif
#ifdef CA_HAVE_T5_TRANSLATE
    if (s->t5_translate_ctx) {
        if (s->beam_size > 1)
            t5_translate_set_beam_size(s->t5_translate_ctx, s->beam_size);
        // MADLAD-400 picks the target language from a "<2xx> " tag
        // prepended to the source (t5_translate expects the caller to do
        // this; src_lang is unused — the encoder is language-agnostic).
        // Guard with t5_has_token so a plain flan-t5/mT5 GGUF, whose vocab
        // lacks the tag, isn't fed a garbage [▁,<unk>] prefix.
        std::string in = text;
        if (tgt_lang && tgt_lang[0]) {
            std::string tag = std::string("<2") + tgt_lang + ">";
            if (t5_has_token(s->t5_translate_ctx, tag.c_str()))
                in = tag + " " + in;
        }
        (void)src_lang;
        return t5_translate(s->t5_translate_ctx, in.c_str(), max_tokens > 0 ? max_tokens : 200);
    }
#endif
    (void)max_tokens;
    return nullptr;
}

// Free a string returned by `crispasr_session_translate_text`.  Mirrors
// the punc-side `crispasr_punc_free_text` symmetric-ownership pattern —
// without this, safe-Rust callers would need to drag in libc::free just
// to release a single malloc'd buffer.  No-op when `text` is nullptr.
CA_EXPORT void crispasr_session_translate_text_free(char* text) {
    free(text);
}

// =========================================================================
// Streaming session API (PLAN #62b — generalize stream_open from
// whisper_context* to crispasr_session*).
//
// Today only the whisper backend is wired through (it's what the
// rolling-window engine in `crispasr_stream_*` was built for); future
// backends (moonshine-streaming, kyutai-stt, voxtral4b) plug in here
// by routing `crispasr_session_stream_feed` to their native streaming
// entry points.
//
// Returns nullptr if the session's backend doesn't support streaming.
// =========================================================================

CA_EXPORT crispasr_stream* crispasr_session_stream_open(crispasr_session* s, int n_threads, int step_ms, int length_ms,
                                                        int keep_ms, const char* language, int translate) {
    if (!s)
        return nullptr;
    if (s->whisper_ctx)
        return crispasr_stream_open(s->whisper_ctx, n_threads, step_ms, length_ms, keep_ms, language, translate);
#if __has_include("kyutai_stt.h")
    if (s->kyutai_ctx) {
        // Chunked-batch over a rolling window — see PLAN #62c.
        // n_threads/keep_ms/language/translate are unused here: kyutai
        // doesn't have language/translate flags, threads are configured at
        // model init, and the rolling window already carries left context.
        (void)n_threads;
        (void)keep_ms;
        (void)language;
        (void)translate;
        kyutai_stt_stream* ks = kyutai_stt_stream_open((kyutai_stt_context*)s->kyutai_ctx, step_ms, length_ms);
        if (!ks)
            return nullptr;
        auto* w = new crispasr_stream();
        w->kyutai_stream_state = ks;
        return w;
    }
#endif
#if __has_include("moonshine_streaming.h")
    if (s->moonshine_streaming_ctx) {
        // Same chunked-batch pattern as kyutai (PLAN #62c follow-on).
        // Despite the backend name, moonshine_streaming_transcribe is single-shot
        // — the "streaming" refers to the model architecture, not the API.
        (void)n_threads;
        (void)keep_ms;
        (void)language;
        (void)translate;
        moonshine_streaming_stream* ms = moonshine_streaming_stream_open(
            (moonshine_streaming_context*)s->moonshine_streaming_ctx, step_ms, length_ms);
        if (!ms)
            return nullptr;
        auto* w = new crispasr_stream();
        w->moonshine_streaming_state = ms;
        return w;
    }
#endif
#ifdef CA_HAVE_VOXTRAL4B
    if (s->voxtral4b_ctx) {
        // PLAN #7 — native incremental encoder + decode-on-flush.
        // step_ms / length_ms are accepted for ABI parity but currently
        // ignored (decode happens at flush in phase 1).
        (void)n_threads;
        (void)keep_ms;
        (void)language;
        (void)translate;
        voxtral4b_stream* vs = voxtral4b_stream_open(s->voxtral4b_ctx, step_ms, length_ms);
        if (!vs)
            return nullptr;
        auto* w = new crispasr_stream();
        w->voxtral4b_stream_state = vs;
        return w;
    }
#endif
    return nullptr;
}

CA_EXPORT void crispasr_session_close(crispasr_session* s) {
    if (!s)
        return;
#ifdef CA_HAVE_FIREREDPUNC
    if (s->punc_ctx)
        fireredpunc_free((fireredpunc_context*)s->punc_ctx);
#endif
#ifdef CA_HAVE_PCS
    if (s->pcs_ctx)
        pcs_free((pcs_context*)s->pcs_ctx);
#endif
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
#ifdef CA_HAVE_LFM2_AUDIO
    if (s->lfm2_audio_ctx)
        lfm2_audio_free(s->lfm2_audio_ctx);
#endif
#ifdef CA_HAVE_MINI_OMNI2
    if (s->mini_omni2_ctx)
        mini_omni2_free(s->mini_omni2_ctx);
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
#ifdef CA_HAVE_GRANITE_NLE
    if (s->granite_nle_ctx)
        granite_nle_free(s->granite_nle_ctx);
#endif
#ifdef CA_HAVE_FUNASR
    if (s->funasr_ctx)
        funasr_free(s->funasr_ctx);
#endif
#ifdef CA_HAVE_PARAFORMER
    if (s->paraformer_ctx)
        paraformer_free(s->paraformer_ctx);
#endif
#ifdef CA_HAVE_SENSEVOICE
    if (s->sensevoice_ctx)
        sensevoice_free(s->sensevoice_ctx);
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
#ifdef CA_HAVE_KUGELAUDIO
    if (s->kugelaudio_ctx)
        kugelaudio_free(s->kugelaudio_ctx);
#endif
#ifdef CA_HAVE_ZONOS
    if (s->zonos_ctx)
        zonos_tts_free(s->zonos_ctx);
#endif
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx)
        qwen3_tts_free(s->qwen3_tts_ctx);
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
#ifdef CA_HAVE_MOONSHINE_STREAMING
    if (s->moonshine_streaming_ctx)
        moonshine_streaming_free((moonshine_streaming_context*)s->moonshine_streaming_ctx);
#endif
#ifdef CA_HAVE_GEMMA4_E2B
    if (s->gemma4_e2b_ctx)
        gemma4_e2b_free((gemma4_e2b_context*)s->gemma4_e2b_ctx);
#endif
#ifdef CA_HAVE_OMNIASR
    if (s->omniasr_ctx)
        omniasr_free((omniasr_context*)s->omniasr_ctx);
#endif
#ifdef CA_HAVE_ORPHEUS
    if (s->orpheus_ctx)
        orpheus_free(s->orpheus_ctx);
#endif
#ifdef CA_HAVE_KOKORO
    if (s->kokoro_ctx)
        kokoro_free(s->kokoro_ctx);
#endif
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx)
        chatterbox_free(s->chatterbox_ctx);
#endif
#ifdef CA_HAVE_TADA
    if (s->tada_ctx)
        tada_free(s->tada_ctx);
#endif
#ifdef CA_HAVE_OUTETTS
    if (s->outetts_ctx)
        outetts_free(s->outetts_ctx);
#endif
#ifdef CA_HAVE_CSM
    if (s->csm_tts_ctx)
        csm_tts_free(s->csm_tts_ctx);
#endif
#ifdef CA_HAVE_DIA
    if (s->dia_tts_ctx)
        dia_tts_free(s->dia_tts_ctx);
#endif
#ifdef CA_HAVE_POCKET
    if (s->pocket_tts_ctx)
        pocket_tts_free(s->pocket_tts_ctx);
#endif
#ifdef CA_HAVE_FASTPITCH
    if (s->fastpitch_ctx)
        fastpitch_tts_free(s->fastpitch_ctx);
#endif
#ifdef CA_HAVE_PARLER
    if (s->parler_ctx)
        parler_tts_free(s->parler_ctx);
#endif
#ifdef CA_HAVE_SPEECHT5
    if (s->speecht5_ctx)
        speecht5_tts_free(s->speecht5_ctx);
#endif
#ifdef CA_HAVE_BARK
    if (s->bark_ctx)
        bark_free(s->bark_ctx);
#endif
#ifdef CA_HAVE_PARLER_TTS
    if (s->parler_tts_ctx)
        parler_tts_free(s->parler_tts_ctx);
#endif
#ifdef CA_HAVE_VOXCPM2
    if (s->voxcpm2_ctx)
        voxcpm2_free(s->voxcpm2_ctx);
#endif
#ifdef CA_HAVE_COSYVOICE3
    if (s->cosyvoice3_ctx)
        cosyvoice3_tts_free(s->cosyvoice3_ctx);
#endif
#ifdef CA_HAVE_INDEXTTS
    if (s->indextts_ctx)
        indextts_free(s->indextts_ctx);
#endif
#ifdef CA_HAVE_F5TTS
    if (s->f5tts_ctx)
        f5_tts_free(s->f5tts_ctx);
#endif
#ifdef CA_HAVE_PIPER
    if (s->piper_ctx)
        piper_tts_free(s->piper_ctx);
#endif
#ifdef CA_HAVE_MELOTTS
    if (s->melotts_ctx)
        melotts_free(s->melotts_ctx);
#endif
#ifdef CA_HAVE_M2M100
    if (s->m2m100_ctx)
        m2m100_free(s->m2m100_ctx);
#endif
#ifdef CA_HAVE_T5_TRANSLATE
    if (s->t5_translate_ctx)
        t5_translate_free(s->t5_translate_ctx);
#endif
#ifdef CA_HAVE_MIMO_ASR
    if (s->mimo_asr_ctx)
        mimo_asr_free(s->mimo_asr_ctx);
#endif
#ifdef CA_HAVE_MOSS_AUDIO
    if (s->moss_audio_ctx)
        moss_audio_free(s->moss_audio_ctx);
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
    free(const_cast<char*>(text));
}

CA_EXPORT void crispasr_punc_free(void* ctx) {
    fireredpunc_free((fireredpunc_context*)ctx);
}
#else
CA_EXPORT void* crispasr_punc_init(const char*) {
    return nullptr;
}
CA_EXPORT const char* crispasr_punc_process(void*, const char*) {
    return nullptr;
}
CA_EXPORT void crispasr_punc_free_text(const char*) {}
CA_EXPORT void crispasr_punc_free(void*) {}
#endif

// =========================================================================
// Truecaser — standalone text post-processing (init → process → free).
// Three backends: statistical (`truecaser`), BiLSTM (`truecaser_lstm`),
// and CRF (`truecaser_crf`). The `crispasr_truecase_init` dispatcher
// probes the GGUF architecture to pick the right backend automatically.
// All three share the same C-ABI surface. Process returns a malloc'd
// string; free it with `crispasr_truecase_free_text`.
// =========================================================================

#if defined(CA_HAVE_TRUECASER_LSTM)
CA_EXPORT void* crispasr_truecase_init(const char* model_path) {
    // Prefer LSTM when available (97.9% F1 on German).
    return (void*)truecaser_lstm_init(model_path);
}
CA_EXPORT const char* crispasr_truecase_process(void* ctx, const char* text) {
    return truecaser_lstm_process((truecaser_lstm_context*)ctx, text);
}
CA_EXPORT void crispasr_truecase_free_text(const char* text) {
    free(const_cast<char*>(text));
}
CA_EXPORT void crispasr_truecase_free(void* ctx) {
    truecaser_lstm_free((truecaser_lstm_context*)ctx);
}
#elif defined(CA_HAVE_TRUECASER)
CA_EXPORT void* crispasr_truecase_init(const char* model_path) {
    return (void*)truecaser_init(model_path);
}
CA_EXPORT const char* crispasr_truecase_process(void* ctx, const char* text) {
    return truecaser_process((truecaser_context*)ctx, text);
}
CA_EXPORT void crispasr_truecase_free_text(const char* text) {
    free(const_cast<char*>(text));
}
CA_EXPORT void crispasr_truecase_free(void* ctx) {
    truecaser_free((truecaser_context*)ctx);
}
#else
CA_EXPORT void* crispasr_truecase_init(const char*) {
    return nullptr;
}
CA_EXPORT const char* crispasr_truecase_process(void*, const char*) {
    return nullptr;
}
CA_EXPORT void crispasr_truecase_free_text(const char*) {}
CA_EXPORT void crispasr_truecase_free(void*) {}
#endif

// =========================================================================
// PCS — Punctuation + Capitalization + Sentence-boundary Detection.
// Single GGUF model, standalone post-processing.
// =========================================================================

#ifdef CA_HAVE_PCS
CA_EXPORT void* crispasr_pcs_init(const char* model_path) {
    return (void*)pcs_init(model_path);
}
CA_EXPORT const char* crispasr_pcs_process(void* ctx, const char* text) {
    return pcs_process((pcs_context*)ctx, text);
}
CA_EXPORT void crispasr_pcs_free_text(const char* text) {
    free(const_cast<char*>(text));
}
CA_EXPORT void crispasr_pcs_free(void* ctx) {
    pcs_free((pcs_context*)ctx);
}
#else
CA_EXPORT void* crispasr_pcs_init(const char*) {
    return nullptr;
}
CA_EXPORT const char* crispasr_pcs_process(void*, const char*) {
    return nullptr;
}
CA_EXPORT void crispasr_pcs_free_text(const char*) {}
CA_EXPORT void crispasr_pcs_free(void*) {}
#endif

// =========================================================================
// Parallel transcription wrapper
// =========================================================================
//
// Thin C-ABI wrapper around whisper_full_parallel for Dart FFI. Tracks
// progress via the module-level g_progress atomic (same as the session
// path's progress callback).

CA_EXPORT int crispasr_transcribe_parallel(struct whisper_context* ctx, struct whisper_full_params params,
                                           const float* samples, int n_samples, int n_processors) {
    g_progress.store(0, std::memory_order_relaxed);
    int rc = whisper_full_parallel(ctx, params, samples, n_samples, n_processors);
    g_progress.store(-1, std::memory_order_relaxed);
    return rc;
}

// =========================================================================
// Version reporting — identifies the C-ABI build to every consumer
// (CLI, Dart, Python, Rust). Bump when breaking or extending the surface.
// =========================================================================

CA_EXPORT const char* crispasr_c_api_version(void) {
    // 0.6.0 — Adds CrisperWeaver parity: crispasr_get_progress /
    // crispasr_reset_progress (atomic progress polling for Dart FFI),
    // crispasr_audio_load_stereo (stereo PCM decode),
    // crispasr_transcribe_parallel (whisper_full_parallel wrapper),
    // crispasr_ctx_params_set_dtw / crispasr_token_dtw_t (DTW
    // timestamp init + getter). Pure addition; no symbol renames.
    // 0.5.3 — Adds `crispasr_truecase_*` (init/process/free/free_text)
    // and `crispasr_pcs_*` (init/process/free/free_text) standalone
    // text post-processors. Pure addition; no symbol renames.
    // 0.5.2 — Adds `crispasr_text_detect_language` (text-LID via the
    // internal `text_lid_dispatch` façade — CLD3 + GlotLID-V3 +
    // LID-176 routed by GGUF architecture).  Mirrors the audio-side
    // `crispasr_detect_language_pcm` return-code contract.
    // 0.5.1 — Adds `crispasr_session_translate_text_free`.
    // Pure addition; no symbol renames or signature changes.
    return "0.6.0";
}

// Backwards-compatibility alias. The Dart smoke test and any 0.4.x-era
// consumer probed `crispasr_dart_helpers_version`. The symbol was renamed
// when the file moved to `crispasr_c_api.cpp` (no longer Dart-specific).
// TODO: remove once all in-tree consumers are updated and a major-version
// bump is cut.
CA_EXPORT const char* crispasr_dart_helpers_version(void) {
    return crispasr_c_api_version();
}

// =========================================================================
// Kokoro per-language model + voice routing — re-exports of the
// crispasr_kokoro_* helpers from src/kokoro.cpp so they're visible to
// every wrapper. See src/kokoro.h for full semantics. (PLAN #56 opt 2b)
// =========================================================================

#ifdef CA_HAVE_KOKORO
CA_EXPORT bool crispasr_kokoro_lang_is_german_abi(const char* lang) {
    return crispasr_kokoro_lang_is_german(lang);
}

CA_EXPORT bool crispasr_kokoro_lang_has_native_voice_abi(const char* lang) {
    return crispasr_kokoro_lang_has_native_voice(lang);
}

CA_EXPORT int crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang, char* out_path,
                                                         int out_path_len) {
    return crispasr_kokoro_resolve_model_for_lang(model_path, lang, out_path, out_path_len);
}

CA_EXPORT int crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang, char* out_path,
                                                         int out_path_len, char* out_picked, int out_picked_len) {
    return crispasr_kokoro_resolve_fallback_voice(model_path, lang, out_path, out_path_len, out_picked, out_picked_len);
}

// Drop the per-session phoneme cache. No-op if the session has no kokoro
// context loaded. Returns 0 on success, -1 if `s` is null. (PLAN #56 #5)
CA_EXPORT int crispasr_session_kokoro_clear_phoneme_cache(crispasr_session* s) {
    if (!s)
        return -1;
    if (s->kokoro_ctx)
        kokoro_phoneme_cache_clear(s->kokoro_ctx);
    return 0;
}
#else
CA_EXPORT bool crispasr_kokoro_lang_is_german_abi(const char*) {
    return false;
}
CA_EXPORT bool crispasr_kokoro_lang_has_native_voice_abi(const char*) {
    return false;
}
CA_EXPORT int crispasr_kokoro_resolve_model_for_lang_abi(const char*, const char*, char*, int) {
    return 1;
}
CA_EXPORT int crispasr_kokoro_resolve_fallback_voice_abi(const char*, const char*, char*, int, char*, int) {
    return 2;
}
CA_EXPORT int crispasr_session_kokoro_clear_phoneme_cache(crispasr_session*) {
    return 0;
}
#endif

// =========================================================================
// Sticky session-state setters (PLAN #59 partial unblock).
//
// These close gaps between CLI flags and what wrappers can reach.
// Per-call args (e.g. transcribe_lang's `language`) still win when
// supplied; these are the fallback. Returns 0 on success, -1 on null
// session, -2 if backend doesn't accept the value at runtime.
// =========================================================================

// Sticky source-language hint. Used by canary, cohere, voxtral, voxtral4b,
// whisper. Empty string clears.
CA_EXPORT int crispasr_session_set_source_language(crispasr_session* s, const char* lang) {
    if (!s)
        return -1;
    s->source_language = (lang ? lang : "");
    return 0;
}

// Sticky target-language. When set and ≠ source_language, canary/cohere
// emit a translation instead of an ASR transcript. Whisper uses
// translate=true with target=en. Empty string clears.
CA_EXPORT int crispasr_session_set_target_language(crispasr_session* s, const char* lang) {
    if (!s)
        return -1;
    s->target_language = (lang ? lang : "");
    return 0;
}

// Sticky punctuation toggle. canary/cohere honour it natively (per-call
// arg); LLM-style backends rely on the post-process strip. Default true.
CA_EXPORT int crispasr_session_set_punctuation(crispasr_session* s, int enable) {
    if (!s)
        return -1;
    s->punctuation = (enable != 0);
    return 0;
}

// Select + load a punctuation-restoration model on the session, the same way
// the CLI `--punc-model` and the server do. `punc_model` is an alias
// (auto|firered|fullstop|punctuate-all|pcs) or a direct .gguf path; the model
// auto-downloads on first use. Pass "none"/""/NULL to unload. Restores
// punctuation on non-PnC backends (parakeet RNNT/CTC, etc.) that emit none.
// Returns 0 on success (incl. unload), -1 on bad handle, -2 if the requested
// model failed to load, -3 if punctuation support wasn't compiled in.
CA_EXPORT int crispasr_session_set_punc_model(crispasr_session* s, const char* punc_model) {
    if (!s)
        return -1;
        // Unload any currently-resident context first.
#ifdef CA_HAVE_FIREREDPUNC
    if (s->punc_ctx) {
        fireredpunc_free((fireredpunc_context*)s->punc_ctx);
        s->punc_ctx = nullptr;
    }
#endif
#ifdef CA_HAVE_PCS
    if (s->pcs_ctx) {
        pcs_free((pcs_context*)s->pcs_ctx);
        s->pcs_ctx = nullptr;
    }
#endif

    const crispasr_punc_spec spec = crispasr_resolve_punc_model(punc_model ? punc_model : "");
    if (spec.kind == crispasr_punc_kind::none)
        return 0; // unloaded / disabled

    std::string path = spec.direct_path;
    if (path.empty() && !spec.cache_filename.empty())
        path = crispasr_cache::ensure_cached_file(spec.cache_filename, spec.url, /*quiet=*/true, "crispasr[punc]", "");
    if (path.empty())
        return -2;

    if (spec.kind == crispasr_punc_kind::fireredpunc) {
#ifdef CA_HAVE_FIREREDPUNC
        s->punc_ctx = (void*)fireredpunc_init(path.c_str());
        return s->punc_ctx ? 0 : -2;
#else
        return -3;
#endif
    }
    if (spec.kind == crispasr_punc_kind::pcs) {
#ifdef CA_HAVE_PCS
        s->pcs_ctx = (void*)pcs_init(path.c_str());
        return s->pcs_ctx ? 0 : -2;
#else
        return -3;
#endif
    }
    return -2;
}

// Apply the session's resident punctuation model (if any) to every segment's
// text, in place. PCS takes precedence over FireRedPunc. Gated on
// `s->punctuation` so a caller that disabled punctuation still gets plain text.
static void apply_session_punc_model(crispasr_session* s, crispasr_session_result* r) {
    if (!s || !r || !s->punctuation)
        return;
    for (auto& seg : r->segments) {
        char* out = nullptr;
#ifdef CA_HAVE_PCS
        if (s->pcs_ctx)
            out = pcs_process((pcs_context*)s->pcs_ctx, seg.text.c_str());
#endif
#ifdef CA_HAVE_FIREREDPUNC
        if (!out && s->punc_ctx)
            out = fireredpunc_process((fireredpunc_context*)s->punc_ctx, seg.text.c_str());
#endif
        if (out) {
            seg.text = out;
            free(out);
        }
    }
}

// Sticky --translate toggle (whisper). For canary/cohere/voxtral the
// equivalent is set_target_language() ≠ source. Default false.
CA_EXPORT int crispasr_session_set_translate(crispasr_session* s, int enable) {
    if (!s)
        return -1;
    s->translate = (enable != 0);
    return 0;
}

// Sticky audio Q&A prompt for instruct-tuned audio-LLM backends
// (granite / voxtral / voxtral4b / qwen3-asr / glm-asr / gemma4-e2b /
// mimo-asr). Pass an empty string to clear and resume verbatim
// transcription. Other backends ignore — set is cheap so we don't error.
CA_EXPORT int crispasr_session_set_ask(crispasr_session* s, const char* prompt) {
    if (!s)
        return -1;
    s->ask = prompt ? prompt : "";
    return 0;
}

// Set decoder temperature on backends that expose runtime control:
// canary, cohere, parakeet, moonshine. Other backends silently no-op.
// `seed` is used by the temperature-sampling RNG; pass 0 for time-based.
CA_EXPORT int crispasr_session_set_temperature(crispasr_session* s, float temperature, uint64_t seed) {
    if (!s)
        return -1;
    s->temperature = temperature;
    s->seed = seed;
    int touched = 0;
#ifdef CA_HAVE_CANARY
    if (s->canary_ctx) {
        canary_set_temperature(s->canary_ctx, temperature, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_COHERE
    if (s->cohere_ctx) {
        cohere_set_temperature(s->cohere_ctx, temperature, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_PARAKEET
    if (s->parakeet_ctx) {
        parakeet_set_temperature(s->parakeet_ctx, temperature, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_MOONSHINE
    if (s->moonshine_ctx) {
        // moonshine's setter takes (ctx, temperature) — no seed parameter.
        moonshine_set_temperature((moonshine_context*)s->moonshine_ctx, temperature);
        touched++;
    }
#endif
#ifdef CA_HAVE_ORPHEUS
    if (s->orpheus_ctx) {
        // Orpheus AR sampler reads ctx->params.temperature on every
        // sample; the runtime setter (added 2026-05) just mutates it.
        // No seed argument — orpheus uses its own RNG bound at init.
        orpheus_set_temperature((orpheus_context*)s->orpheus_ctx, temperature);
        (void)seed;
        touched++;
    }
#endif
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx) {
        chatterbox_set_temperature((chatterbox_context*)s->chatterbox_ctx, temperature);
        (void)seed;
        touched++;
    }
#endif
#ifdef CA_HAVE_TADA
    if (s->tada_ctx) {
        tada_set_temperature(s->tada_ctx, temperature);
        touched++;
    }
#endif
#ifdef CA_HAVE_OUTETTS
    if (s->outetts_ctx) {
        outetts_set_temperature(s->outetts_ctx, temperature);
        outetts_set_seed(s->outetts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_CSM
    if (s->csm_tts_ctx) {
        csm_tts_set_temperature(s->csm_tts_ctx, temperature);
        csm_tts_set_seed(s->csm_tts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_DIA
    if (s->dia_tts_ctx) {
        dia_tts_set_temperature(s->dia_tts_ctx, temperature);
        dia_tts_set_seed(s->dia_tts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_POCKET
    if (s->pocket_tts_ctx) {
        pocket_tts_set_temperature(s->pocket_tts_ctx, temperature);
        pocket_tts_set_seed(s->pocket_tts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_BARK
    if (s->bark_ctx) {
        bark_set_temperature_semantic(s->bark_ctx, temperature);
        bark_set_temperature_coarse(s->bark_ctx, temperature);
        bark_set_temperature_fine(s->bark_ctx, std::min(temperature, 0.5f));
        bark_set_seed(s->bark_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_PARLER_TTS
    if (s->parler_tts_ctx) {
        parler_tts_set_temperature(s->parler_tts_ctx, temperature > 0.0f ? temperature : 1.0f);
        touched++;
    }
#endif
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx) {
        // qwen3-tts's code-predictor sampler reads cparams.temperature
        // on every step (after the 0.6.2 wiring); 0.0 means "use the
        // upstream 0.9 default" — pass any other value to override.
        qwen3_tts_set_temperature((qwen3_tts_context*)s->qwen3_tts_ctx, temperature);
        (void)seed;
        touched++;
    }
#endif
#ifdef CA_HAVE_ZONOS
    if (s->zonos_ctx) {
        zonos_tts_set_temperature(s->zonos_ctx, temperature);
        touched++;
    }
#endif
    return touched > 0 ? 0 : -2;
}

// Set the seed for sampling-capable TTS backends. This currently
// covers chatterbox, vibevoice, qwen3-tts, and orpheus. Other
// backends silently no-op (rc=-2).
CA_EXPORT int crispasr_session_set_tts_seed(crispasr_session* s, uint64_t seed) {
    if (!s)
        return -1;
    int touched = 0;
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx) {
        chatterbox_set_seed((chatterbox_context*)s->chatterbox_ctx, (uint32_t)seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_TADA
    if (s->tada_ctx) {
        tada_set_seed(s->tada_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_OUTETTS
    if (s->outetts_ctx) {
        outetts_set_seed(s->outetts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_CSM
    if (s->csm_tts_ctx) {
        csm_tts_set_seed(s->csm_tts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_DIA
    if (s->dia_tts_ctx) {
        dia_tts_set_seed(s->dia_tts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_POCKET
    if (s->pocket_tts_ctx) {
        pocket_tts_set_seed(s->pocket_tts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_BARK
    if (s->bark_ctx) {
        bark_set_seed(s->bark_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_PARLER_TTS
    if (s->parler_tts_ctx) {
        parler_tts_set_seed(s->parler_tts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_VIBEVOICE
    if (s->vibevoice_ctx) {
        vibevoice_set_seed((vibevoice_context*)s->vibevoice_ctx, (uint32_t)seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_KUGELAUDIO
    if (s->kugelaudio_ctx) {
        kugelaudio_set_seed(s->kugelaudio_ctx, (uint32_t)seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_QWEN3_TTS
    if (s->qwen3_tts_ctx) {
        qwen3_tts_set_seed((qwen3_tts_context*)s->qwen3_tts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_ORPHEUS
    if (s->orpheus_ctx) {
        orpheus_set_seed((orpheus_context*)s->orpheus_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_VOXCPM2
    if (s->voxcpm2_ctx) {
        voxcpm2_set_seed((voxcpm2_context*)s->voxcpm2_ctx, (uint32_t)seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_COSYVOICE3
    if (s->cosyvoice3_ctx) {
        cosyvoice3_tts_set_seed(s->cosyvoice3_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_INDEXTTS
    if (s->indextts_ctx) {
        indextts_set_seed((indextts_context*)s->indextts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_F5TTS
    if (s->f5tts_ctx) {
        f5_tts_set_seed(s->f5tts_ctx, seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_MELOTTS
    if (s->melotts_ctx) {
        melotts_set_seed(s->melotts_ctx, (uint32_t)seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_MOSS_AUDIO
    if (s->moss_audio_ctx) {
        moss_audio_set_seed(s->moss_audio_ctx, (uint32_t)seed);
        touched++;
    }
#endif
#ifdef CA_HAVE_ZONOS
    if (s->zonos_ctx) {
        zonos_tts_set_seed(s->zonos_ctx, seed);
        touched++;
    }
#endif
    return touched > 0 ? 0 : -2;
}

// ─────────────────────────────────────────────────────────────────
// CrispASR 0.6.1 parity additions — TTS sampling knobs reachable at
// runtime so the CrisperWeaver Synthesize screen can drive them
// without reopening the session per setting change.
// ─────────────────────────────────────────────────────────────────

// Set the diffusion / CFM step count for diffusion-based TTS
// backends. Today only chatterbox honours this (its CFM mel-decoder
// is a 10-step Euler solver by default; raising to 20-30 trades
// latency for fidelity). Other TTS backends silently no-op (rc=-2).
CA_EXPORT int crispasr_session_set_tts_steps(crispasr_session* s, int steps) {
    if (!s)
        return -1;
    int touched = 0;
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx) {
        chatterbox_set_cfm_steps((chatterbox_context*)s->chatterbox_ctx, steps);
        touched++;
    }
#endif
#ifdef CA_HAVE_VIBEVOICE
    if (s->vibevoice_ctx) {
        // VibeVoice's DPM-Solver++ step count — read on every
        // synthesize() call, so post-init mutation changes the next
        // call's schedule density.
        vibevoice_set_tts_steps((vibevoice_context*)s->vibevoice_ctx, steps);
        touched++;
    }
#endif
#ifdef CA_HAVE_KUGELAUDIO
    if (s->kugelaudio_ctx) {
        kugelaudio_set_tts_steps(s->kugelaudio_ctx, steps);
        touched++;
    }
#endif
    return touched > 0 ? 0 : -2;
}

// G2P dictionary source: "olaph" (MIT), "open-dict" (CC-BY-SA), or file path.
// Applies to piper + kokoro backends. Returns 0 on success, -1 if session
// is null, -2 if no TTS backend is active.
CA_EXPORT int crispasr_session_set_g2p_dict(crispasr_session* s, const char* source) {
    if (!s)
        return -1;
    // piper_tts_set_g2p_dict is process-global (not per-context) so call
    // it unconditionally — it affects the next phonemize_builtin() call.
    piper_tts_set_g2p_dict(source);
    return 0;
}

// Set the top-p nucleus-sampling threshold. Honoured by chatterbox;
// other backends no-op (their AR loops use top-k or hardcoded
// sampling parameters today).
CA_EXPORT int crispasr_session_set_top_p(crispasr_session* s, float top_p) {
    if (!s)
        return -1;
    int touched = 0;
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx) {
        chatterbox_set_top_p((chatterbox_context*)s->chatterbox_ctx, top_p);
        touched++;
    }
#endif
    return touched > 0 ? 0 : -2;
}

// Set the min-p sampling threshold. Honoured by chatterbox.
CA_EXPORT int crispasr_session_set_min_p(crispasr_session* s, float min_p) {
    if (!s)
        return -1;
    int touched = 0;
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx) {
        chatterbox_set_min_p((chatterbox_context*)s->chatterbox_ctx, min_p);
        touched++;
    }
#endif
    return touched > 0 ? 0 : -2;
}

// Set the repetition penalty (1.0 = no penalty). Honoured by
// chatterbox.
CA_EXPORT int crispasr_session_set_repetition_penalty(crispasr_session* s, float r) {
    if (!s)
        return -1;
    int touched = 0;
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx) {
        chatterbox_set_repetition_penalty((chatterbox_context*)s->chatterbox_ctx, r);
        touched++;
    }
#endif
    return touched > 0 ? 0 : -2;
}

// Set the classifier-free-guidance weight (chatterbox). 0 disables
// CFG; 0.5 is the upstream default; values up to 2.0 amplify the
// conditional path.
CA_EXPORT int crispasr_session_set_cfg_weight(crispasr_session* s, float cfg_weight) {
    if (!s)
        return -1;
    int touched = 0;
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx) {
        chatterbox_set_cfg_weight((chatterbox_context*)s->chatterbox_ctx, cfg_weight);
        touched++;
    }
#endif
    return touched > 0 ? 0 : -2;
}

// Set the emotion-exaggeration scalar (chatterbox). 0.5 is the
// upstream default; raise for more dramatic delivery, lower for
// flat / monotone.
CA_EXPORT int crispasr_session_set_exaggeration(crispasr_session* s, float exaggeration) {
    if (!s)
        return -1;
    int touched = 0;
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx) {
        chatterbox_set_exaggeration((chatterbox_context*)s->chatterbox_ctx, exaggeration);
        touched++;
    }
#endif
    return touched > 0 ? 0 : -2;
}

// Set the upper bound on speech tokens generated per synthesize call
// (chatterbox AR loop). Default 1000 ≈ 20 s of audio at 50 Hz codes.
// Raise for very long single-shot synth; lower to bound runaway
// hallucinations.
CA_EXPORT int crispasr_session_set_max_speech_tokens(crispasr_session* s, int n) {
    if (!s)
        return -1;
    int touched = 0;
#ifdef CA_HAVE_CHATTERBOX
    if (s->chatterbox_ctx) {
        chatterbox_set_max_speech_tokens((chatterbox_context*)s->chatterbox_ctx, n);
        touched++;
    }
#endif
    return touched > 0 ? 0 : -2;
}

// Set the per-phoneme length-scale / speaking-rate scalar for TTS
// backends that have a duration model. Today only kokoro consumes
// it (PLAN #88). Other backends silently no-op (rc=-2). 1.0 =
// upstream default; >1.0 = slower / longer; <1.0 = faster / shorter.
// Clamped to [0.25, 4.0] inside the per-backend setter.
CA_EXPORT int crispasr_session_set_length_scale(crispasr_session* s, float scale) {
    if (!s)
        return -1;
    int touched = 0;
#ifdef CA_HAVE_KOKORO
    if (s->kokoro_ctx) {
        kokoro_set_length_scale((kokoro_context*)s->kokoro_ctx, scale);
        touched++;
    }
#endif
    return touched > 0 ? 0 : -2;
}

CA_EXPORT int crispasr_session_set_best_of(crispasr_session* s, int n) {
    if (!s)
        return -1;
    s->best_of = n > 0 ? n : 1;
    return 0;
}

CA_EXPORT int crispasr_session_set_max_new_tokens(crispasr_session* s, int n) {
    if (!s)
        return -1;
    s->max_new_tokens = n > 0 ? n : 0;
#ifdef CA_HAVE_COHERE
    if (s->cohere_ctx)
        cohere_set_max_new_tokens(s->cohere_ctx, s->max_new_tokens);
#endif
    return 0;
}

CA_EXPORT int crispasr_session_set_frequency_penalty(crispasr_session* s, float penalty) {
    if (!s)
        return -1;
    s->frequency_penalty = penalty > 0.0f ? penalty : 0.0f;
#ifdef CA_HAVE_COHERE
    if (s->cohere_ctx)
        cohere_set_frequency_penalty(s->cohere_ctx, s->frequency_penalty);
#endif
    return 0;
}

// §90 Sticky beam_size for beam-search sampling. > 1 activates beam search
// on 11 backends wired in transcribe_single:
//   whisper (native BEAM_SEARCH), qwen3-asr / granite / voxtral (replay via
//   core_beam_decode::run_with_probs), glm-asr / kyutai-stt / firered /
//   moonshine / omniasr (per-backend _set_beam_size setter),
//   canary / cohere (branched-KV AED beam via _set_beam_size),
//   funasr / gemma4-e2b / moonshine-streaming / m2m100 / t5 (§139).
// voxtral4b beam is CLI-adapter-only; session streaming path TBD.
// Silent no-op for CTC/NAR backends.
// Returns 0 on a non-null session; width <= 0 clamped to 1 (greedy).
CA_EXPORT int crispasr_session_set_beam_size(crispasr_session* s, int n) {
    if (!s)
        return -1;
    s->beam_size = n > 0 ? n : 1;
    return 0;
}

// GBNF grammar-constrained sampling (whisper-only — wparams.grammar_rules
// has no analog on other backends today).
//
// Pass `gbnf_text == nullptr` (or empty) to disable grammar constraints
// and resume unconstrained decoding. Otherwise the GBNF source is parsed
// once at setter-time and the resulting whisper_grammar_element graph is
// stored on the session for reuse across every subsequent transcribe call.
//
// `root_rule` is the symbol name to start parsing from (typically "root");
// `penalty` is whisper's grammar_penalty scalar (the CLI default is 100.0).
//
// When grammar is active, the whisper transcribe path automatically
// switches to beam search (grammar-constrained sampling requires beam ≥ 2);
// a beam_size left at the default 1 gets bumped to 5.
//
// Return codes:
//    0 = grammar parsed and stored (or cleared, when text was empty)
//   -1 = null session
//   -2 = parse failed (invalid GBNF) or root_rule not found in parsed grammar
CA_EXPORT int crispasr_session_set_grammar_text(crispasr_session* s, const char* gbnf_text, const char* root_rule,
                                                float penalty) {
    if (!s)
        return -1;
    // Clear-grammar path: empty text disables grammar-constrained
    // sampling for subsequent transcribe calls.
    if (!gbnf_text || gbnf_text[0] == '\0') {
        s->grammar_text.clear();
        s->grammar_root_rule.clear();
        s->grammar_parsed = grammar_parser::parse_state{};
        s->grammar_rules_ptrs.clear();
        s->grammar_root_rule_id = 0;
        s->grammar_active = false;
        return 0;
    }
    // Parse the GBNF. The parser writes to stderr on syntax errors but
    // doesn't throw; we detect failure by checking the resulting
    // rules vector + symbol table.
    grammar_parser::parse_state parsed = grammar_parser::parse(gbnf_text);
    if (parsed.rules.empty()) {
        return -2;
    }
    const std::string root = (root_rule && root_rule[0]) ? root_rule : "root";
    auto it = parsed.symbol_ids.find(root);
    if (it == parsed.symbol_ids.end()) {
        return -2;
    }
    // Commit to the session. `c_rules()` materialises a fresh vector
    // of pointers into `parsed.rules`; both the vector and the
    // underlying rules must outlive the next transcribe call, which
    // is why both fields live on the session.
    s->grammar_text = gbnf_text;
    s->grammar_root_rule = root;
    s->grammar_parsed = std::move(parsed);
    s->grammar_rules_ptrs = s->grammar_parsed.c_rules();
    s->grammar_root_rule_id = it->second;
    s->grammar_penalty = penalty > 0.0f ? penalty : 100.0f;
    s->grammar_active = true;
    return 0;
}

// Whisper decoder-fallback thresholds. All four are written into
// the session struct here and applied to wparams on every whisper
// transcribe dispatch. Non-whisper backends silently ignore — the
// fields have no analog in their wparams equivalent.
//
// Defaults from whisper_full_default_params (the values the
// session struct ships with):
//   entropy_thold     = 2.4f   (per-token entropy fallback trigger)
//   logprob_thold     = -1.0f  (avg-logprob fallback trigger)
//   no_speech_thold   = 0.6f   (silence detector cutoff)
//   temperature_inc   = 0.2f   (temperature step per fallback pass;
//                                0.0 disables fallback entirely =
//                                the CLI's `--no-fallback`)
//
// Caller passes whatever values they want — there's no "leave
// default" sentinel because every value in this set is a real
// float with meaningful semantics, not a presence flag.
CA_EXPORT int crispasr_session_set_fallback_thresholds(crispasr_session* s, float entropy_thold, float logprob_thold,
                                                       float no_speech_thold, float temperature_inc) {
    if (!s)
        return -1;
    s->entropy_thold = entropy_thold;
    s->logprob_thold = logprob_thold;
    s->no_speech_thold = no_speech_thold;
    // Clamp temperature_inc to [0, 1] — values outside that range
    // either disable fallback (= 0, fine) or cause the fallback
    // loop to never terminate in some versions of whisper.cpp.
    if (temperature_inc < 0.0f)
        temperature_inc = 0.0f;
    if (temperature_inc > 1.0f)
        temperature_inc = 1.0f;
    s->temperature_inc = temperature_inc;
    return 0;
}

// Per-token top-N alternative-candidate capture (whisper greedy
// decode only). Writes the sticky value onto the session; the
// transcribe path forwards it into wparams.alt_n on every dispatch.
// 0 = off (the upstream default). Bumped beyond 5 is allowed but
// the UI caps at 5; greater values just cost more memory.
//
// Non-whisper backends silently ignore — none of the other engines'
// wparams equivalents have a runner-ups concept today (parakeet's
// hypothesis lattice is closest in shape but exposed via a
// different API).
CA_EXPORT int crispasr_session_set_alt_n(crispasr_session* s, int n) {
    if (!s)
        return -1;
    s->alt_n = n < 0 ? 0 : (n > 32 ? 32 : n);
    return 0;
}

// Whisper text-suppression + prompt-carry extras. All three map
// onto whisper_full_params fields with no analog on other
// backends, so this setter is whisper-only at apply time. The
// session struct holds the values and the transcribe path
// writes them into wparams on every dispatch.
//
// Defaults from whisper_full_default_params:
//   suppress_nst         = false  ("emit non-speech tokens like
//                                    [LAUGHTER], [MUSIC] when
//                                    whisper produces them")
//   suppress_regex       = ""     (no suppression)
//   carry_initial_prompt = false  ("only prepend initial_prompt
//                                    to the FIRST decode window")
//
// `suppress_regex` is copied into a std::string on the session;
// the caller can free their copy after this returns. Empty
// string clears any prior regex.
CA_EXPORT int crispasr_session_set_whisper_decode_extras(crispasr_session* s, int suppress_nst,
                                                         const char* suppress_regex, int carry_initial_prompt) {
    if (!s)
        return -1;
    s->whisper_suppress_nst = suppress_nst != 0;
    s->whisper_carry_initial_prompt = carry_initial_prompt != 0;
    s->whisper_suppress_regex = suppress_regex ? suppress_regex : "";
    return 0;
}

// Auto-detect spoken language on raw 16 kHz mono PCM. Wraps the
// standalone crispasr_detect_language() from src/crispasr_lid.h so
// wrappers can invoke LID via their session handle.
//
// `lid_model_path` is the LID GGUF (whisper-tiny for the whisper
// method, silero-lid for silero). `method`: 0=Whisper, 1=Silero,
// 2=Firered, 3=Ecapa.
//
// `out_lang` receives a NUL-terminated ISO 639-1 code; `out_lang_cap`
// must be ≥ 8. `out_prob` (optional) gets the model's confidence.
//
// Returns 0 on success, -1 on null args / buffer too small, -2 if LID
// failed.
CA_EXPORT int crispasr_session_detect_language(crispasr_session* s, const float* pcm, int n_samples,
                                               const char* lid_model_path, int method, char* out_lang, int out_lang_cap,
                                               float* out_prob) {
    if (!s || !pcm || n_samples <= 0 || !lid_model_path || !out_lang || out_lang_cap < 8)
        return -1;
    CrispasrLidOptions opts;
    opts.n_threads = s->n_threads;
    opts.model_path = lid_model_path;
    opts.method = (CrispasrLidMethod)method;
    CrispasrLidResult lid_out;
    if (!::crispasr_detect_language(pcm, n_samples, opts, lid_out) || lid_out.lang_code.empty())
        return -2;
    if ((int)lid_out.lang_code.size() + 1 > out_lang_cap)
        return -1;
    std::memcpy(out_lang, lid_out.lang_code.data(), lid_out.lang_code.size());
    out_lang[lid_out.lang_code.size()] = '\0';
    if (out_prob)
        *out_prob = lid_out.confidence;
    return 0;
}

// =========================================================================
// Speaker verification — TitaNet + speaker profile DB
// =========================================================================

#ifdef CA_HAVE_TITANET

CA_EXPORT void* crispasr_titanet_init(const char* model_path, int32_t n_threads) {
    return (void*)titanet_init(model_path, n_threads);
}

CA_EXPORT void crispasr_titanet_free(void* ctx) {
    titanet_free((struct titanet_context*)ctx);
}

// Extract speaker embedding from PCM. Returns embedding dimension (192) on
// success, 0 on error. `out` must hold at least 192 floats.
CA_EXPORT int32_t crispasr_titanet_embed(void* ctx, const float* pcm_16k, int32_t n_samples, float* out) {
    return (int32_t)titanet_embed((struct titanet_context*)ctx, pcm_16k, n_samples, out);
}

CA_EXPORT float crispasr_titanet_cosine_sim(const float* a, const float* b, int32_t dim) {
    return titanet_cosine_sim(a, b, dim);
}

CA_EXPORT void* crispasr_speaker_db_load(const char* dir_path) {
    return (void*)speaker_db_load(dir_path);
}

CA_EXPORT void crispasr_speaker_db_free(void* db) {
    speaker_db_free((struct speaker_db*)db);
}

CA_EXPORT int32_t crispasr_speaker_db_count(const void* db) {
    return (int32_t)speaker_db_count((const struct speaker_db*)db);
}

// Match embedding against speaker DB. Writes the speaker name into
// `out_name` (up to `out_cap` bytes including NUL). Returns cosine
// similarity score on match, or a negative value if no match.
CA_EXPORT float crispasr_speaker_db_match(const void* db, const float* embedding, int32_t dim, float threshold,
                                          char* out_name, int32_t out_cap) {
    float score = -1.0f;
    const char* name = speaker_db_match((const struct speaker_db*)db, embedding, dim, threshold, &score);
    if (name && out_name && out_cap > 0) {
        int len = (int)std::strlen(name);
        if (len >= out_cap)
            len = out_cap - 1;
        std::memcpy(out_name, name, len);
        out_name[len] = '\0';
    }
    return name ? score : -1.0f;
}

CA_EXPORT int32_t crispasr_speaker_db_enroll(const char* dir_path, const char* name, const float* embedding,
                                             int32_t dim) {
    return speaker_db_enroll(dir_path, name, embedding, dim) ? 0 : 1;
}

#endif // CA_HAVE_TITANET

// =========================================================================
// Pluggable speaker embedder, agglomerative clustering, and pyannote-seg
// cache (issue #107 P6 — pipeline primitives so every language binding
// can compose the same diarize flow the CLI does).
// =========================================================================

// ---- pluggable embedder ----
// `model_spec` accepts "auto", "titanet", "indextts", "indextts-bigvgan",
// "ecapa", or a path to a `.gguf`. Returns an opaque handle, or null on
// failure. See crispasr_speaker_embedder.h for the dispatch rules.
CA_EXPORT void* crispasr_speaker_embedder_make_abi(const char* model_spec, int32_t n_threads, const char* cache_dir) {
    if (!model_spec)
        return nullptr;
    auto p =
        crispasr_make_speaker_embedder(std::string(model_spec), n_threads, cache_dir ? std::string(cache_dir) : "");
    return p.release(); // ownership transfers to the caller via free_abi
}

CA_EXPORT void crispasr_speaker_embedder_free_abi(void* embedder) {
    if (embedder)
        delete static_cast<CrispasrSpeakerEmbedder*>(embedder);
}

CA_EXPORT int32_t crispasr_speaker_embedder_dim_abi(const void* embedder) {
    if (!embedder)
        return 0;
    return static_cast<const CrispasrSpeakerEmbedder*>(embedder)->dim();
}

// Embed one mono 16 kHz PCM range. `out` must hold at least dim() floats.
// Returns 1 on success, 0 on failure (e.g. clip too short for the model).
CA_EXPORT int32_t crispasr_speaker_embedder_embed_abi(void* embedder, const float* pcm_16k, int32_t n_samples,
                                                      float* out) {
    if (!embedder || !pcm_16k || n_samples <= 0 || !out)
        return 0;
    return static_cast<CrispasrSpeakerEmbedder*>(embedder)->embed(pcm_16k, n_samples, out) ? 1 : 0;
}

CA_EXPORT const char* crispasr_speaker_embedder_name_abi(const void* embedder) {
    if (!embedder)
        return "";
    return static_cast<const CrispasrSpeakerEmbedder*>(embedder)->name();
}

// ---- agglomerative cosine clustering ----
// Pure arithmetic. `embeddings` is a row-major n×dim buffer of (ideally
// L2-normalized) speaker embeddings; `labels_out` is filled with one
// cluster ID per input in [0, k). Returns the number of clusters k, or
// -1 on invalid args.
CA_EXPORT int32_t crispasr_speaker_cluster_abi(const float* embeddings, int32_t n, int32_t dim, float merge_threshold,
                                               int32_t max_speakers, int32_t* labels_out) {
    if (!embeddings || n <= 0 || dim <= 0 || !labels_out)
        return -1;
    std::vector<float> e(embeddings, embeddings + (size_t)n * (size_t)dim);
    auto labels = crispasr_agglomerative_cluster(e, n, dim, merge_threshold, max_speakers);
    int max_label = -1;
    for (int i = 0; i < n; i++) {
        labels_out[i] = (int32_t)labels[i];
        if (labels[i] > max_label)
            max_label = labels[i];
    }
    return max_label + 1; // total cluster count (0 means none assigned)
}

// ---- pyannote-seg cache ----
// Pre-compute pyannote-seg posteriors over the FULL audio buffer once,
// then apply them to per-segment scoring with stable local track IDs.
// The cache is opaque; callers free it with the matching _free_abi.
//
// Returns null on model-load failure or empty audio.
struct crispasr_pyannote_cache_abi {
    std::vector<float> log_probs;
    int T = 0;
    double frame_dur_s = 0.0;
};

CA_EXPORT void* crispasr_pyannote_cache_compute_abi(const float* full_audio, int32_t n_samples, const char* model_path,
                                                    int32_t n_threads) {
    if (!full_audio || n_samples <= 0 || !model_path || !*model_path)
        return nullptr;
    pyannote_seg_context* pctx = pyannote_seg_init(model_path, n_threads > 0 ? n_threads : 4);
    if (!pctx)
        return nullptr;
    int T = 0;
    float* probs = pyannote_seg_run(pctx, full_audio, n_samples, &T);
    pyannote_seg_free(pctx);
    if (!probs || T <= 0) {
        if (probs)
            std::free(probs);
        return nullptr;
    }
    auto* cache = new crispasr_pyannote_cache_abi();
    cache->log_probs.assign(probs, probs + (size_t)T * 7);
    cache->T = T;
    cache->frame_dur_s = 270.0 / 16000.0;
    std::free(probs);
    return cache;
}

CA_EXPORT void crispasr_pyannote_cache_free_abi(void* cache) {
    if (cache)
        delete static_cast<crispasr_pyannote_cache_abi*>(cache);
}

// Score segs against a precomputed pyannote cache. Each seg's `speaker`
// is set to 0/1/2 (local pyannote-seg track index) or -1 for silence.
// `slice_t0_cs` is the absolute centisecond at which the cache buffer
// starts (usually 0 — the cache covers the whole input audio).
//
// Returns 0 on success, -1 on invalid args.
CA_EXPORT int32_t crispasr_pyannote_cache_apply_abi(const void* cache, int64_t slice_t0_cs,
                                                    crispasr_diarize_seg_abi* segs, int32_t n_segs) {
    if (!cache || !segs || n_segs <= 0)
        return -1;
    const auto* c = static_cast<const crispasr_pyannote_cache_abi*>(cache);
    std::vector<CrispasrDiarizeSegment> lib_segs;
    lib_segs.reserve(n_segs);
    for (int i = 0; i < n_segs; i++)
        lib_segs.push_back({segs[i].t0_cs, segs[i].t1_cs, segs[i].speaker});
    crispasr_diarize_internal::assign_speakers_from_log_posteriors(c->log_probs.data(), c->T, c->frame_dur_s,
                                                                   slice_t0_cs, lib_segs);
    for (int i = 0; i < n_segs; i++)
        segs[i].speaker = lib_segs[i].speaker;
    return 0;
}
