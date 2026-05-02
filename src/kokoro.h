#pragma once

// Kokoro / StyleTTS2 (iSTFTNet) public C ABI.
//
// hexgrad/Kokoro-82M and yl4579/StyleTTS2-LJSpeech share the same
// architecture (custom ALBERT BERT + ProsodyPredictor + iSTFTNet
// decoder + phoneme TextEncoder) and feed through this single runtime.
// Pre-trained voices ship as separate per-voice GGUFs (arch =
// "kokoro-voice") containing one F32 tensor `voice.pack[max_phon, 1,
// 256]`. Index by phoneme length L: ref_s = voice.pack[L-1, 0, :],
// split as [predictor_style 0:128 | decoder_style 128:256].
//
// Phonemizer: when the build links libespeak-ng (CMake option
// CRISPASR_WITH_ESPEAK_NG, defaults to AUTO-detect via pkg-config or the
// homebrew layout), we call espeak_TextToPhonemes() in-process. Otherwise
// we shell out to popen("espeak-ng -q --ipa=3 -v LANG TEXT"). Results are
// memoised in a per-context LRU cache keyed on (lang, text).
// Set CRISPASR_ESPEAK_DATA_PATH to override the espeak-ng-data location.
// Pass already-IPA strings via kokoro_synthesize_phonemes to skip both
// paths. The vocab map (178 IPA symbols) lives in the GGUF as
// `tokenizer.ggml.tokens`.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kokoro_context;

struct kokoro_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    bool gen_force_metal; // KOKORO_GEN_FORCE_METAL=1 — debug only; default false
                          // pins the iSTFTNet generator to backend_cpu to avoid
                          // a known Metal hang on stride-10 ConvTranspose1d
    char espeak_lang[32]; // espeak-ng -v LANG, default "en-us"
};

struct kokoro_context_params kokoro_context_default_params(void);

// Load Kokoro / StyleTTS2 GGUF. Returns nullptr on failure.
struct kokoro_context* kokoro_init_from_file(const char* path_model, struct kokoro_context_params params);

void kokoro_free(struct kokoro_context* ctx);
void kokoro_set_n_threads(struct kokoro_context* ctx, int n_threads);

// Load a voice-pack GGUF (arch="kokoro-voice"). Each pack stores ONE
// voice — single tensor `voice.pack[max_phon, 1, 256]`, plus
// `kokoro_voice.name` metadata. Returns 0 on success.
int kokoro_load_voice_pack(struct kokoro_context* ctx, const char* path);

// Override the espeak-ng language (default "en-us"). Returns 0 on success.
int kokoro_set_language(struct kokoro_context* ctx, const char* espeak_lang);

// Drop every cached (lang, text) → IPA entry. Cheap and safe to call
// concurrently. Intended for long-running daemons that resynthesize
// across many speakers and want bounded memory.
void kokoro_phoneme_cache_clear(struct kokoro_context* ctx);

// Diff-harness entry points (PLAN #56 #4). Both return malloc'd UTF-8
// IPA strings — caller frees with free(). Stateless, no kokoro_context
// needed.
//
// Lib path is gated on CRISPASR_HAVE_ESPEAK_NG at compile time and on
// successful runtime init; returns nullptr if the lib isn't compiled
// in or its in-process init fails. Popen path shells out to
// `espeak-ng` on PATH; returns nullptr if the binary isn't available.
//
// `lang` is an espeak-ng voice (e.g. "en-us", "de", "fr", "cmn", "ja").
// `text` is UTF-8 plain text — pre-phonemizer raw input.
char* kokoro_phonemize_text_lib(const char* lang, const char* text);
char* kokoro_phonemize_text_popen(const char* lang, const char* text);

// Tokenise a phoneme string (already-IPA) into the model's vocab.
// Returns malloc'd int32_t[*out_n] — caller frees with free().
int32_t* kokoro_phonemes_to_ids(struct kokoro_context* ctx, const char* phonemes, int* out_n);

// Synthesise text → 24 kHz mono float32 PCM. Runs the espeak-ng
// phonemizer first. Returns malloc'd buffer; caller frees with
// kokoro_pcm_free. *out_n_samples set on success; nullptr on failure.
float* kokoro_synthesize(struct kokoro_context* ctx, const char* text, int* out_n_samples);

// Same as kokoro_synthesize but the input is already IPA — skips espeak-ng.
float* kokoro_synthesize_phonemes(struct kokoro_context* ctx, const char* phonemes, int* out_n_samples);

// Diff-harness stage extractor. Pass a phoneme string and a stage name;
// returns the stage's activations as malloc'd float[*out_n]. Stage names
// match the ggml_set_name labels in src/kokoro.cpp:
//
// All "L"-shaped stages below use the StyleTTS2 pad-wrap convention —
// the raw phoneme ids are wrapped as [pad, *raw, pad] before being fed
// to BERT / text_enc / predictor, so each L is the raw token count + 2.
//
//   "token_ids"        — int32 cast to float (raw tokenisation, length L_raw)
//   "bert_pooler_out"  — (768, L) BERT last_hidden_state (NOT the pooled
//                        vector — name kept for ABI stability; "pooler"
//                        weight is loaded but unused by the synth path)
//   "bert_proj_out"    — (512, L) bert_proj per-token Linear of pooler_out
//   "text_enc_out"     — (512, L) text encoder output
//   "dur_enc_out"      — (L, 512) duration-encoder output
//   "durations"        — (L,) integer durations cast to float
//   "align_out"        — (T_frames, 512) duration-aligned features
//   "f0_curve"         — (T_frames,) F0 prediction
//   "n_curve"          — (T_frames,) energy prediction
//   "dec_encode_out"   — (T_frames, 512) decoder pre-generator features
//   "dec_decode_3_out" — (T_frames, 256) last decode-stack output
//   "gen_pre_post_out" — (T_audio_frames, 512) generator output before conv_post
//   "mag"              — (11, T_audio_frames) iSTFT magnitude
//   "phase"            — (11, T_audio_frames) iSTFT phase
//   "audio_out"        — (T_samples,) final 24 kHz waveform
//
// Caller frees with free().
float* kokoro_extract_stage(struct kokoro_context* ctx, const char* phonemes, const char* stage_name, int* out_n);

void kokoro_pcm_free(float* pcm);

// ---------------------------------------------------------------------------
// Per-language model + voice routing (PLAN #56 opt 2b).
//
// Kokoro-82M ships voice packs only for {a/b, e, f, h, i, j, p, z}. For the
// languages without a native voice (de, ru, ko, ar, ...) we keep a small
// policy table here so every wrapper can reuse it instead of re-implementing
// the CLI's logic.
//
// Two layers of routing:
//
//  1. Backbone swap (German only).  When `lang` starts with "de" AND
//     `model_path` is the official kokoro-82m baseline (basename starts with
//     "kokoro-82m"), and a sibling `kokoro-de-hui-base-f16.gguf` exists,
//     callers should prefer it. The German backbone is a Stage-1 fine-tune
//     of Kokoro-82M on HUI-Audio-Corpus-German
//     (dida-80b/kokoro-german-hui-multispeaker-base, Apache-2.0; HUI corpus
//     CC0).
//
//  2. Voice fallback.  Returns the first existing
//     `<model_dir>/kokoro-voice-<name>.gguf` from the per-language candidate
//     cascade. German tries `df_victoria` (kikiri-tts/kikiri-german-victoria
//     voicepack — Apache-2.0, in-distribution to the dida-80b lineage) ->
//     `df_eva` (Tundragoon German, Apache-2.0) -> `ff_siwis` (French —
//     non-silence baseline). Other unsupported langs fall through to
//     `ff_siwis` directly.

// True iff `lang` begins with "de" followed by '\0', '-' or '_'.
bool crispasr_kokoro_lang_is_german(const char* lang);

// True iff `lang` is one of the languages Kokoro-82M ships native voice packs
// for: en, es, fr, hi, it, ja, pt, cmn, zh.
bool crispasr_kokoro_lang_has_native_voice(const char* lang);

// Resolve the model path to load for `lang` given `model_path` as the
// user-provided baseline. Writes the result into `out_path` if non-NULL.
// Returns: 0 if a German backbone swap was applied (path rewritten),
// 1 if no swap is applicable (caller should keep `model_path` — also
// copied into `out_path` for convenience), -1 on buffer-too-small.
int crispasr_kokoro_resolve_model_for_lang(const char* model_path, const char* lang, char* out_path, int out_path_len);

// Resolve the fallback voice path for `lang`. Walks the per-language voice
// cascade and returns the first existing
// `<dirname(model_path)>/kokoro-voice-<name>.gguf`. Writes the path into
// `out_path` and the picked basename ("df_victoria", "df_eva", "ff_siwis",
// ...) into `out_picked` (both optional).
// Returns: 0 if a fallback was found and written, 1 if `lang` already has a
// native voice (no fallback needed), 2 if no candidate file exists,
// -1 on buffer-too-small.
int crispasr_kokoro_resolve_fallback_voice(const char* model_path, const char* lang, char* out_path, int out_path_len,
                                           char* out_picked, int out_picked_len);

#ifdef __cplusplus
}
#endif
