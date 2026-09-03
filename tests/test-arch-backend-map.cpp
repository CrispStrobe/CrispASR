// test-arch-backend-map.cpp — hermetic guards for the shared
// `general.architecture` → backend table (src/core/arch_backend_map.h).
//
// Issue #335: the C ABI and the CLI each carried their own copy of this
// mapping. The copies drifted; the C-ABI one never learned "granite_speech"
// (the spelling every granite converter writes), so `Session::open()` returned
// NULL from Rust/Python/Go/Dart for a model the CLI opened fine — the CLI's
// filename pass masked the gap for CLI users. No model files needed: the
// mapping is a pure table, and the ABI test below synthesises a 0-tensor GGUF
// carrying nothing but `general.architecture`.
//
// The properties here are the ones a TYPO breaks (docs rule 3b): a wrong
// spelling in the key column, a shadowed duplicate, a backend name in the value
// column that no surface can open, and — the one that actually shipped — a
// surface that does not consult the table at all.

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/arch_backend_map.h"

#include "gguf.h"

// The C ABI's detector — the surface issue #335 was reported against.
extern "C" int crispasr_detect_backend_from_gguf(const char* path, char* out_name, int out_cap);

// ---------------------------------------------------------------------------
// The architectures our own converters write for a standalone (openable) model.
// Each pair is grep-able back to models/convert-*-to-gguf.py. This list is the
// point of the whole exercise: every one of these must resolve, on EVERY
// surface, without a filename hint.
// ---------------------------------------------------------------------------
static const std::vector<std::pair<std::string, std::string>>& converter_archs() {
    static const std::vector<std::pair<std::string, std::string>> k = {
        // ── the #335 regression itself ──
        {"granite_speech", "granite"},      // convert-granite-speech-to-gguf.py
        {"granite_nle", "granite-4.1-nar"}, // convert-granite-nle-to-gguf.py
        // ── ASR ──
        {"whisper", "whisper"},
        {"parakeet", "parakeet"},
        {"nemotron", "nemotron"},
        {"gigaam", "gigaam"},
        {"canary", "canary"},
        {"canary_qwen", "canary-qwen"},
        {"canary-ctc", "fastconformer-ctc"},
        {"cohere-transcribe", "cohere"},
        {"qwen3asr", "qwen3"},
        {"voxtral", "voxtral"},
        {"voxtral4b", "voxtral4b"},
        {"higgs-stt", "higgs-stt"},
        {"glmasr", "glm-asr"},
        {"kyutai-stt", "kyutai-stt"},
        {"firered-asr", "firered-asr"},
        {"moonshine", "moonshine"},
        {"moonshine_streaming", "moonshine-streaming"},
        {"gemma4e2b", "gemma4-e2b"},
        {"omniasr-ctc", "omniasr"},
        {"mimo_asr", "mimo-asr"},
        {"arkasr", "ark-asr"},
        {"moss_audio", "moss-audio"},
        {"moss_transcribe", "moss-transcribe"},
        {"moss_transcribe_diarize", "moss-diarize"},
        {"funasr", "funasr"},
        {"paraformer", "paraformer"},
        {"sensevoice", "sensevoice"},
        {"sidon", "sidon"},
        {"mini-omni2", "mini-omni2"},
        {"lfm2-audio", "lfm2-audio"},
        {"wav2vec2", "wav2vec2"},
        {"vibevoice-asr", "vibevoice"},
        {"vibevoice-tts", "vibevoice-tts"},
        // ── TTS ──
        {"qwen3tts", "qwen3-tts"},
        {"miotts", "miotts"},
        {"moss-tts", "moss-tts"},
        {"moss-tts-local", "moss-tts-local"},
        {"orpheus", "orpheus"},
        {"kokoro", "kokoro"},
        {"voxcpm2", "voxcpm2-tts"},
        {"voxcpm2-vae", "voxcpm2-vae"},
        {"cosyvoice3-llm", "cosyvoice3-tts"},
        {"fastpitch", "fastpitch"},
        {"bananamind_tts", "bananamind-tts"},
        {"omnivoice", "omnivoice"},
        {"piper", "piper"},
        {"melotts", "melotts"},
        {"f5-tts", "f5-tts"},
        {"irodori-tts", "irodori-tts"},
        {"chatterbox", "chatterbox"},
        {"tada-tts", "tada"},
        {"voxtral_tts", "voxtral-tts"},
        {"kugelaudio", "kugelaudio"},
        {"zonos-tts", "zonos"},
        {"indextts.gpt", "indextts"},
        {"outetts", "outetts"},
        {"pocket-tts", "pocket-tts"},
        {"speecht5-tts", "speecht5"},
        {"bark", "bark"},
        {"dia", "dia"},
        {"dots-tts", "dots-tts"},
        {"csm-tts", "csm"},
        {"parler-tts", "parler-tts"},
        // ── translation / music ──
        {"m2m100", "m2m100"},
        {"t5", "madlad"},
        {"htdemucs", "htdemucs"},
        {"mel-band-roformer", "mel-band-roformer"},
        {"crepe", "crepe"},
        {"btc", "btc-chords"},
        {"tabcnn", "tabcnn"},
        {"rvc", "rvc-svc"},
        {"beat-this", "beat-this"},
        {"piano-transcription", "piano-transcription"},
    };
    return k;
}

TEST_CASE("granite-speech resolves on every spelling the wild contains", "[unit][arch-map]") {
    // The bug: only the hyphen spelling was known to the C ABI, while
    // models/convert-granite-speech-to-gguf.py has always written the
    // underscore one — so granite-speech-4.1-2b-plus-q4_k.gguf was
    // undetectable from Rust/Python/Go/Dart.
    CHECK(core_arch::backend_for_arch("granite_speech") == std::string("granite"));
    CHECK(core_arch::backend_for_arch("granite-speech") == std::string("granite"));
    CHECK(core_arch::backend_for_arch("granitespeech") == std::string("granite"));
    // The NAR sibling is a different runtime and must not collapse onto it.
    CHECK(core_arch::backend_for_arch("granite_nle") == std::string("granite-4.1-nar"));
}

TEST_CASE("every converter-written architecture resolves", "[unit][arch-map]") {
    for (const auto& kv : converter_archs()) {
        INFO("architecture: " << kv.first);
        CHECK(core_arch::backend_for_arch(kv.first) == kv.second);
    }
}

TEST_CASE("unknown / empty architectures resolve to the empty string", "[unit][arch-map]") {
    // "" is the caller's signal for "could not auto-detect" — never a backend
    // name, and never a guess. Component GGUFs (voice packs, tokenizers,
    // codecs, diff-reference archives) land here on purpose.
    CHECK(core_arch::backend_for_arch("") == std::string(""));
    CHECK(core_arch::backend_for_arch(static_cast<const char*>(nullptr)) == std::string(""));
    CHECK(core_arch::backend_for_arch("bert") == std::string(""));
    CHECK(core_arch::backend_for_arch("crispasr.reference") == std::string(""));
    CHECK(core_arch::backend_for_arch("kokoro-voice") == std::string(""));
    CHECK(core_arch::backend_for_arch("cosyvoice3-s3tok") == std::string(""));
    CHECK(core_arch::backend_for_arch("not-a-real-architecture") == std::string(""));
}

TEST_CASE("no architecture key is listed twice", "[unit][arch-map]") {
    // A duplicate key is invisible at runtime — the first entry wins and the
    // second silently does nothing, which is exactly how a mapping ends up
    // pointing at the wrong backend after a copy-paste.
    size_t n = 0;
    const core_arch::entry* k = core_arch::table(&n);
    REQUIRE(n > 100); // the table is the union of both former copies
    std::map<std::string, std::string> seen;
    for (size_t i = 0; i < n; i++) {
        INFO("architecture: " << k[i].arch);
        auto it = seen.find(k[i].arch);
        CHECK(it == seen.end());
        seen[k[i].arch] = k[i].backend;
    }
}

TEST_CASE("every emitted backend name is a name a surface can open", "[unit][arch-map]") {
    // The value column is the other half of the contract: a name here that no
    // dispatch matches produces the same NULL session as an unknown
    // architecture, just with a more convincing error message. This set is
    // written independently of the table on purpose — it is the law, not a
    // re-derivation of it. ("fireredpunc" is the one detect-only name: a
    // punctuation restorer runs behind --punc-model, never as a session.)
    static const std::set<std::string> known = {
        "whisper",
        "parakeet",
        "nemotron",
        "gigaam",
        "canary",
        "canary-qwen",
        "fastconformer-ctc",
        "lfm2-audio",
        "mini-omni2",
        "cohere",
        "qwen3",
        "voxtral",
        "voxtral4b",
        "higgs-stt",
        "granite",
        "granite-4.1-nar",
        "wav2vec2",
        "glm-asr",
        "kyutai-stt",
        "firered-asr",
        "fireredpunc",
        "moonshine",
        "moonshine-streaming",
        "gemma4-e2b",
        "omniasr",
        "mimo-asr",
        "ark-asr",
        "moss-audio",
        "moss-transcribe",
        "moss-diarize",
        "funasr",
        "paraformer",
        "sensevoice",
        "sidon",
        "vibevoice",
        "vibevoice-tts",
        "qwen3-tts",
        "miotts",
        "moss-tts",
        "moss-tts-local",
        "orpheus",
        "kokoro",
        "voxcpm2-tts",
        "voxcpm2-vae",
        "cosyvoice3-tts",
        "cosyvoice3-tts-rl",
        "fastpitch",
        "bananamind-tts",
        "omnivoice",
        "piper",
        "melotts",
        "f5-tts",
        "irodori-tts",
        "chatterbox",
        "tada",
        "voxtral-tts",
        "kugelaudio",
        "zonos",
        "indextts",
        "outetts",
        "pocket-tts",
        "speecht5",
        "bark",
        "dia",
        "dots-tts",
        "confucius4-tts",
        "csm",
        "parler-tts",
        "m2m100",
        "madlad",
        "htdemucs",
        "mel-band-roformer",
        "crepe",
        "btc-chords",
        "tabcnn",
        "rvc-svc",
        "beat-this",
        "piano-transcription",
        "basic-pitch",
        "mt3",
    };
    size_t n = 0;
    const core_arch::entry* k = core_arch::table(&n);
    for (size_t i = 0; i < n; i++) {
        INFO(k[i].arch << " -> " << k[i].backend);
        CHECK(known.count(k[i].backend) == 1);
    }
}

// ---------------------------------------------------------------------------
// The ABI-level guard. Everything above would still have passed if
// `crispasr_detect_backend_from_gguf` had kept its own private copy of the
// table — which is precisely the shape of the shipped bug. So drive the real
// export, on a real GGUF file, through the real code path.
// ---------------------------------------------------------------------------

static std::string write_stub_gguf(const std::string& arch) {
    std::string path = std::string(std::tmpnam(nullptr)) + "-arch-map.gguf";
    gguf_context* ctx = gguf_init_empty();
    REQUIRE(ctx != nullptr);
    gguf_set_val_str(ctx, "general.architecture", arch.c_str());
    const bool ok = gguf_write_to_file(ctx, path.c_str(), /*only_meta=*/true);
    gguf_free(ctx);
    REQUIRE(ok);
    return path;
}

TEST_CASE("crispasr_detect_backend_from_gguf reads the shared table", "[unit][arch-map]") {
    // A metadata-only GGUF with nothing but general.architecture: no tensors,
    // no weights, no model download. This is the exact reproduction of #335 —
    // before the fix, the granite_speech case returned 0 / "" here while the
    // CLI returned "granite" from its filename pass.
    struct probe {
        const char* arch;
        const char* expect;
    };
    const probe probes[] = {
        {"granite_speech", "granite"}, // #335
        {"nemotron", "nemotron"},      // was CLI-only too
        {"qwen3tts", "qwen3-tts"},     // ditto
        {"whisper", "whisper"},        // was already shared — must not regress
        {"not-a-real-architecture", ""},
    };
    for (const auto& p : probes) {
        INFO("architecture: " << p.arch);
        const std::string path = write_stub_gguf(p.arch);
        char out[64] = {0};
        const int rc = crispasr_detect_backend_from_gguf(path.c_str(), out, (int)sizeof(out));
        std::remove(path.c_str());
        CHECK(std::string(out) == std::string(p.expect));
        CHECK(rc == (int)std::string(p.expect).size());
    }
}
