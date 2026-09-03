#pragma once

// ---------------------------------------------------------------------------
// `general.architecture` → CrispASR backend name: the SINGLE source of truth.
//
// Issue #335: `Session::open()` (Rust/Python/Go/Dart, i.e. anything on the C
// ABI) could not open granite-speech-4.1-2b-plus while the CLI opened it fine.
// The reason was pure table drift: two independent copies of this mapping had
// grown side by side —
//
//   * `examples/cli/crispasr_backend.cpp` (CLI pass 2, after its filename pass)
//   * `src/crispasr_c_api.cpp::crispasr_detect_backend_from_gguf` (every binding)
//
// — and the C-ABI copy only knew `"granite-speech"` while every converter has
// always written `"granite_speech"` (models/convert-granite-speech-to-gguf.py).
// The CLI never noticed because its *filename* pass matches "granite"+"speech"
// first and short-circuits pass 2; the bindings have no filename pass, so the
// detect returned "" and `crispasr_session_open` returned NULL.
//
// granite was not alone — an audit of the two tables found 113 architecture
// strings the CLI knew and the C ABI did not, including whole backends
// (nemotron, moonshine, kokoro, piper, melotts, sensevoice, funasr, paraformer,
// glm-asr, kyutai-stt, mini-omni2, csm, dia, bark, speecht5, fastpitch,
// pocket-tts, gemma4-e2b, mimo-asr, piano-transcription, voxtral-tts, …) whose
// GGUFs no binding could auto-detect at all.
//
// So both surfaces now call `core_arch::backend_for_arch()` and there is
// nowhere left for the two to disagree. When you add a backend, add its
// architecture string(s) HERE and both surfaces get it (checklist point 3 in
// docs/contributing.md).
//
// Rules for entries:
//   * Match the string the CONVERTER actually writes (grep the converter for
//     `GGUFWriter(..., arch=...)` / `add_string("general.architecture", ...)`),
//     not the one you would have chosen. Underscore/hyphen/squashed spellings
//     all exist in the wild, so list every spelling that ships.
//   * The value is a backend name accepted by BOTH `crispasr_create_backend()`
//     (CLI) and the `s->backend ==` dispatch in `crispasr_session_open_explicit`
//     — tests/test-arch-backend-map.cpp pins that they stay in sync.
//   * Component/auxiliary GGUFs (voice packs, tokenizers, codecs, VAD/LID
//     helpers, `crispasr.reference` diff archives) are deliberately ABSENT:
//     they are not standalone sessions, and an unknown architecture correctly
//     yields "" rather than a backend that would fail on load.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstring>
#include <string>

namespace core_arch {

struct entry {
    const char* arch;    // exact `general.architecture` value in the GGUF
    const char* backend; // CrispASR backend name
};

// clang-format off
inline const entry* table(size_t* n_out) {
    static const entry k[] = {
        // ── ASR ────────────────────────────────────────────────────────────
        {"whisper",                   "whisper"},
        {"sidon",                     "sidon"},
        {"nemotron",                  "nemotron"},
        {"nemotron-asr",              "nemotron"},
        {"nemotron-streaming",        "nemotron"},
        {"gigaam",                    "gigaam"},
        {"parakeet",                  "parakeet"},
        {"parakeet-tdt",              "parakeet"},
        {"parakeet-ja",               "parakeet"},
        {"parakeet_ja",               "parakeet"},
        {"canary",                    "canary"},
        {"canary_qwen",               "canary-qwen"},
        {"canary-qwen",               "canary-qwen"},
        // A canary-*-ctc GGUF is a FastConformer-CTC model and runs on the
        // canary_ctc runtime, NOT the AED encoder-decoder "canary" backend
        // (which has no CTC head). Both surfaces accept "fastconformer-ctc"
        // and "canary-ctc" as names for that one runtime; we emit the former.
        {"canary-ctc",                "fastconformer-ctc"},
        {"fastconformer-ctc",         "fastconformer-ctc"},
        {"stt-fastconformer-ctc",     "fastconformer-ctc"},
        {"stt_fastconformer_ctc",     "fastconformer-ctc"},
        {"lfm2-audio",                "lfm2-audio"},
        {"mini-omni2",                "mini-omni2"},
        // "cohere-ar" is a registry/CLI alias only — the Arabic model shares the
        // runtime and its GGUF architecture is plain "cohere"/"cohere-transcribe".
        {"cohere",                    "cohere"},
        {"cohere-transcribe",         "cohere"},
        {"qwen3-asr",                 "qwen3"},
        {"qwen3_asr",                 "qwen3"},
        {"qwen3asr",                  "qwen3"},
        {"voxtral",                   "voxtral"},
        {"voxtral4b",                 "voxtral4b"},
        {"voxtral-4b",                "voxtral4b"},
        {"voxtral_4b",                "voxtral4b"},
        {"higgs-stt",                 "higgs-stt"},
        {"higgs_stt",                 "higgs-stt"},
        {"higgs-audio-v3-stt",        "higgs-stt"},
        {"granite-speech",            "granite"},
        {"granite_speech",            "granite"}, // ← what the converter writes (#335)
        {"granitespeech",             "granite"},
        {"granite-nle",               "granite-4.1-nar"},
        {"granite_nle",               "granite-4.1-nar"},
        {"granitenle",                "granite-4.1-nar"},
        {"wav2vec2",                  "wav2vec2"},
        {"wav2vec2-ctc",              "wav2vec2"},
        {"hubert",                    "wav2vec2"},
        {"hubert-ctc",                "wav2vec2"},
        {"data2vec",                  "wav2vec2"},
        {"data2vec-audio",            "wav2vec2"},
        {"data2vec_audio",            "wav2vec2"},
        {"glmasr",                    "glm-asr"},
        {"glm-asr",                   "glm-asr"},
        {"glm_asr",                   "glm-asr"},
        {"kyutai-stt",                "kyutai-stt"},
        {"kyutai_stt",                "kyutai-stt"},
        {"kyutaistt",                 "kyutai-stt"},
        {"firered-asr",               "firered-asr"},
        {"firered_asr",               "firered-asr"},
        {"firered",                   "firered-asr"},
        {"firereadasr",               "firered-asr"},
        {"firered-lid",               "firered-asr"},
        {"firered_lid",               "firered-asr"},
        // Detect-only: a punctuation restorer is driven through `--punc-model`,
        // never opened as a session/CLI backend. Naming it still beats "" —
        // "backend fireredpunc is not available" tells the user what they
        // handed us, where "could not auto-detect" does not.
        {"fireredpunc",               "fireredpunc"},
        {"moonshine",                 "moonshine"},
        {"moonshine-tiny",            "moonshine"},
        {"moonshine-base",            "moonshine"},
        {"moonshine_streaming",       "moonshine-streaming"},
        {"gemma4e2b",                 "gemma4-e2b"},
        {"gemma4_e2b",                "gemma4-e2b"},
        // Both the Omni ASR CTC and LLM converters write "omniasr-ctc"; the
        // omniasr backend reads model_type from the GGUF to route CTC vs LLM.
        {"omniasr",                   "omniasr"},
        {"omniasr-ctc",               "omniasr"},
        {"omniasr_ctc",               "omniasr"},
        {"omniasr-300m",              "omniasr"},
        {"omniasr-llm",               "omniasr"},
        {"omniasr_llm",               "omniasr"},
        {"mimo_asr",                  "mimo-asr"},
        {"mimo-asr",                  "mimo-asr"},
        {"arkasr",                    "ark-asr"},
        {"ark-asr",                   "ark-asr"},
        {"ark_asr",                   "ark-asr"},
        {"moss_audio",                "moss-audio"},
        {"moss-audio",                "moss-audio"},
        {"moss_transcribe",           "moss-transcribe"},
        {"moss-transcribe",           "moss-transcribe"},
        {"moss_transcribe_diarize",   "moss-diarize"},
        {"moss-transcribe-diarize",   "moss-diarize"},
        {"moss_diarize",              "moss-diarize"},
        {"moss-diarize",              "moss-diarize"},
        {"funasr",                    "funasr"},
        {"fun_asr",                   "funasr"},
        {"fun-asr",                   "funasr"},
        {"paraformer",                "paraformer"},
        {"sensevoice",                "sensevoice"},
        {"sense_voice",               "sensevoice"},
        {"sense-voice",               "sensevoice"},
        {"sensevoicesmall",           "sensevoice"},
        // VibeVoice ships an ASR and a TTS variant from one runtime; the session
        // collapses every vibevoice* name onto "vibevoice", the CLI has a
        // separate TTS adapter — so keep the TTS architecture distinct here.
        {"vibevoice",                 "vibevoice"},
        {"vibevoice-asr",             "vibevoice"},
        {"vibevoice_asr",             "vibevoice"},
        {"vibevoice-bitnet",          "vibevoice"},
        {"vibevoice-asr-bitnet",      "vibevoice"},
        {"vibevoice_bitnet",          "vibevoice"},
        {"vibevoice-tts",             "vibevoice-tts"},

        // ── TTS ────────────────────────────────────────────────────────────
        {"qwen3-tts",                 "qwen3-tts"},
        {"qwen3_tts",                 "qwen3-tts"},
        {"qwen3tts",                  "qwen3-tts"},
        {"miotts",                    "miotts"},
        {"mio-tts",                   "miotts"},
        {"moss-tts",                  "moss-tts"},
        {"moss_tts",                  "moss-tts"},
        {"moss-tts-delay",            "moss-tts"},
        {"moss-tts-local",            "moss-tts-local"},
        {"moss_tts_local",            "moss-tts-local"},
        {"orpheus",                   "orpheus"},
        {"kokoro",                    "kokoro"},
        {"styletts2",                 "kokoro"},
        {"styletts2-ljspeech",        "kokoro"},
        {"voxcpm2",                   "voxcpm2-tts"},
        {"voxcpm2-tts",               "voxcpm2-tts"},
        {"voxcpm2_tts",               "voxcpm2-tts"},
        {"voxcpm2-vae",               "voxcpm2-vae"},
        {"voxcpm2_vae",               "voxcpm2-vae"},
        {"cosyvoice3",                "cosyvoice3-tts"},
        {"cosyvoice3-tts",            "cosyvoice3-tts"},
        {"cosyvoice3_tts",            "cosyvoice3-tts"},
        {"cosyvoice3-llm",            "cosyvoice3-tts"},
        {"cosyvoice3-rl",             "cosyvoice3-tts-rl"},
        {"cosyvoice3-tts-rl",         "cosyvoice3-tts-rl"},
        {"cv3-rl",                    "cosyvoice3-tts-rl"},
        {"fastpitch",                 "fastpitch"},
        {"fastpitch-tts",             "fastpitch"},
        {"fastpitch_tts",             "fastpitch"},
        {"bananamind_tts",            "bananamind-tts"},
        {"bananamind-tts",            "bananamind-tts"},
        {"omnivoice",                 "omnivoice"},
        {"omnivoice-tts",             "omnivoice"},
        {"omnivoice_tts",             "omnivoice"},
        {"piper",                     "piper"},
        {"piper-tts",                 "piper"},
        {"piper_tts",                 "piper"},
        {"vits",                      "piper"},
        {"melotts",                   "melotts"},
        {"melo-tts",                  "melotts"},
        {"melo_tts",                  "melotts"},
        {"vits2",                     "melotts"},
        {"f5-tts",                    "f5-tts"},
        {"f5_tts",                    "f5-tts"},
        {"f5tts",                     "f5-tts"},
        {"irodori-tts",               "irodori-tts"},
        {"irodori_tts",               "irodori-tts"},
        {"irodori",                   "irodori-tts"},
        {"chatterbox",                "chatterbox"},
        {"chatterbox_turbo",          "chatterbox"},
        {"kartoffelbox",              "chatterbox"},
        {"tada",                      "tada"},
        {"tada-tts",                  "tada"},
        {"tada-1b",                   "tada"},
        {"tada-tts-1b",               "tada"},
        {"tada-3b-ml",                "tada"},
        {"voxtral-tts",               "voxtral-tts"},
        {"voxtral_tts",               "voxtral-tts"},
        {"kugelaudio",                "kugelaudio"},
        {"kugelaudio-tts",            "kugelaudio"},
        {"kugelaudio_tts",            "kugelaudio"},
        {"zonos",                     "zonos"},
        {"zonos-tts",                 "zonos"},
        {"zonos_tts",                 "zonos"},
        // The IndexTTS converter emits two GGUFs; only the GPT one is the model
        // a session opens (the bigvgan vocoder is a component, deliberately
        // absent so pointing a session at it fails loudly instead of silently).
        {"indextts",                  "indextts"},
        {"indextts.gpt",              "indextts"},
        {"indextts-1.5",              "indextts"},
        {"indextts_1_5",              "indextts"},
        {"outetts",                   "outetts"},
        {"oute-tts",                  "outetts"},
        {"oute_tts",                  "outetts"},
        {"pocket-tts",                "pocket-tts"},
        {"pocket_tts",                "pocket-tts"},
        {"pockettts",                 "pocket-tts"},
        {"speecht5",                  "speecht5"},
        {"speecht5-tts",              "speecht5"},
        {"speecht5_tts",              "speecht5"},
        {"bark",                      "bark"},
        {"bark-tts",                  "bark"},
        {"bark_tts",                  "bark"},
        {"dia",                       "dia"},
        {"dia-tts",                   "dia"},
        {"dia_tts",                   "dia"},
        {"dots-tts",                  "dots-tts"},
        {"dots_tts",                  "dots-tts"},
        {"dots.tts",                  "dots-tts"},
        {"confucius4-tts",            "confucius4-tts"},
        {"confucius4_tts",            "confucius4-tts"},
        {"confucius4",                "confucius4-tts"},
        {"csm",                       "csm"},
        {"csm-tts",                   "csm"},
        {"csm_tts",                   "csm"},
        {"parler-tts",                "parler-tts"},
        {"parler_tts",                "parler-tts"},
        {"parlertts",                 "parler-tts"},

        // ── Translation ────────────────────────────────────────────────────
        {"m2m100",                    "m2m100"},
        {"m2m_100",                   "m2m100"},
        {"t5",                        "madlad"},

        // ── Music / audio analysis ─────────────────────────────────────────
        {"htdemucs",                  "htdemucs"},
        {"demucs",                    "htdemucs"},
        {"mel-band-roformer",         "mel-band-roformer"},
        {"mel_band_roformer",         "mel-band-roformer"},
        {"crepe",                     "crepe"},
        {"btc",                       "btc-chords"},
        {"tabcnn",                    "tabcnn"},
        {"rvc",                       "rvc-svc"},
        {"beat-this",                 "beat-this"},
        {"piano-transcription",       "piano-transcription"},
        {"piano_transcription",       "piano-transcription"},
        {"basic-pitch",               "basic-pitch"},
        {"basic_pitch",               "basic-pitch"},
        {"mt3",                       "mt3"},
    };
    if (n_out)
        *n_out = sizeof(k) / sizeof(k[0]);
    return k;
}
// clang-format on

// Exact-match lookup. Returns "" when the architecture is unknown (an
// auxiliary/component GGUF, or a model this build predates) — callers treat
// that as "could not auto-detect", never as a backend name.
inline const char* backend_for_arch(const char* arch) {
    if (!arch || !*arch)
        return "";
    size_t n = 0;
    const entry* k = table(&n);
    for (size_t i = 0; i < n; i++) {
        if (std::strcmp(k[i].arch, arch) == 0)
            return k[i].backend;
    }
    return "";
}

inline std::string backend_for_arch(const std::string& arch) {
    return std::string(backend_for_arch(arch.c_str()));
}

} // namespace core_arch
