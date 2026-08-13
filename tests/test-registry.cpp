// test-registry.cpp — unit tests for crispasr_model_registry.
//
// Verifies registry lookup, backend listing, and filename-based reverse
// lookup. No network, no models — pure in-memory registry queries.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_model_registry.h"

#include <cstring>
#include <string>

TEST_CASE("registry: lookup known backend returns valid entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("whisper", e);
    REQUIRE(found);
    REQUIRE(std::string(e.filename).find("ggml") != std::string::npos);
    REQUIRE(std::string(e.url).find("huggingface") != std::string::npos);
}

TEST_CASE("registry: lookup unknown backend returns false", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("nonexistent-backend-xyz", e);
    REQUIRE_FALSE(found);
}

TEST_CASE("registry: default bundle reports the exact canonical artifacts", "[unit][registry]") {
    CrispasrRegistryBundle bundle;
    REQUIRE(crispasr_registry_default_bundle("omnivoice", bundle));
    REQUIRE(bundle.backend == "omnivoice");
    REQUIRE_FALSE(bundle.requires_license_acceptance);
    REQUIRE(bundle.artifacts.size() == 2);
    REQUIRE(bundle.artifacts[0].kind == CrispasrRegistryArtifactKind::Primary);
    REQUIRE(bundle.artifacts[0].filename == "omnivoice-f16.gguf");
    REQUIRE(bundle.artifacts[1].kind == CrispasrRegistryArtifactKind::Companion);
    REQUIRE(bundle.artifacts[1].filename == "omnivoice-tokenizer-f16.gguf");
}

TEST_CASE("registry: default bundle resolves aliases and includes extras", "[unit][registry]") {
    CrispasrRegistryBundle bundle;
    REQUIRE(crispasr_registry_default_bundle("cosyvoice3", bundle));
    REQUIRE(bundle.backend == "cosyvoice3-tts");
    REQUIRE(bundle.artifacts.size() == 6);
    REQUIRE(bundle.artifacts[0].kind == CrispasrRegistryArtifactKind::Primary);
    REQUIRE(bundle.artifacts[1].kind == CrispasrRegistryArtifactKind::Companion);
    for (size_t i = 2; i < bundle.artifacts.size(); ++i) {
        REQUIRE(bundle.artifacts[i].kind == CrispasrRegistryArtifactKind::Extra);
        REQUIRE(bundle.artifacts[i].approx_size.empty());
    }
}

TEST_CASE("registry: default bundle preserves license metadata", "[unit][registry]") {
    CrispasrRegistryBundle bundle;
    REQUIRE(crispasr_registry_default_bundle("voxtral-tts", bundle));
    REQUIRE_FALSE(bundle.license.empty());
    REQUIRE(bundle.requires_license_acceptance);
}

TEST_CASE("registry: non-permissive supported weights carry policy metadata", "[unit][registry][license]") {
    struct Expected {
        const char* backend;
        const char* needle;
        bool requires_acceptance;
    } cases[] = {
        {"parakeet", "CC-BY-4.0", false},
        {"canary", "CC-BY-4.0", false},
        {"fastconformer-ctc", "CC-BY-4.0", false},
        {"fastpitch", "CC-BY-4.0", false},
        {"outetts", "CC-BY-NC-SA-4.0", true},
        {"wespeaker", "CC-BY-4.0", false},
        {"lid-fasttext176", "CC-BY-NC-4.0", true},
        {"lfm2-audio", "lfm1.0", true},
        {"funasr", "funasr-v1.1", true},
        {"sensevoice", "funasr-v1.1", true},
        {"gemma4-e2b", "gemma-terms", true},
        {"pocket-tts", "pocket-tts-terms", true},
        {"tada", "llama3.2", true},
        {"orpheus", "llama3.2", true},
    };
    for (const auto& c : cases) {
        CrispasrRegistryEntry e;
        REQUIRE(crispasr_registry_lookup(c.backend, e));
        REQUIRE(std::string(e.license).find(c.needle) != std::string::npos);
        REQUIRE(crispasr_license_requires_acceptance(e.license) == c.requires_acceptance);
    }
}

TEST_CASE("registry: default bundle rejects unknown backends", "[unit][registry]") {
    CrispasrRegistryBundle bundle;
    REQUIRE_FALSE(crispasr_registry_default_bundle("nonexistent-backend-xyz", bundle));
}

TEST_CASE("registry: every default bundle starts with its lookup result", "[unit][registry]") {
    for (int i = 0; i < crispasr_registry_count(); ++i) {
        CrispasrRegistryEntry entry;
        REQUIRE(crispasr_registry_get_at(i, entry));

        CrispasrRegistryEntry canonical;
        REQUIRE(crispasr_registry_lookup(entry.backend, canonical));
        CrispasrRegistryBundle bundle;
        REQUIRE(crispasr_registry_default_bundle(entry.backend, bundle));
        REQUIRE_FALSE(bundle.artifacts.empty());
        REQUIRE(bundle.backend == canonical.backend);
        REQUIRE(bundle.license == canonical.license);
        REQUIRE(bundle.requires_license_acceptance == crispasr_license_requires_acceptance(canonical.license));
        REQUIRE(bundle.artifacts[0].kind == CrispasrRegistryArtifactKind::Primary);
        REQUIRE(bundle.artifacts[0].filename == canonical.filename);
        REQUIRE(bundle.artifacts[0].url == canonical.url);
        REQUIRE(bundle.artifacts[0].approx_size == canonical.approx_size);
        if (!canonical.companion_filename.empty()) {
            REQUIRE(bundle.artifacts.size() >= 2);
            REQUIRE(bundle.artifacts[1].kind == CrispasrRegistryArtifactKind::Companion);
            REQUIRE(bundle.artifacts[1].filename == canonical.companion_filename);
            REQUIRE(bundle.artifacts[1].url == canonical.companion_url);
            REQUIRE(bundle.artifacts[1].approx_size == canonical.companion_approx_size);
        }
    }
}

TEST_CASE("registry: parakeet entry has correct filename", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("parakeet", e);
    REQUIRE(found);
    REQUIRE(std::string(e.filename).find("parakeet") != std::string::npos);
}

TEST_CASE("registry: mimo-asr has entry (added in #63)", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("mimo-asr", e);
    REQUIRE(found);
    REQUIRE(std::string(e.filename).find("mimo-asr") != std::string::npos);
}

TEST_CASE("registry: omniasr has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("omniasr", e);
    REQUIRE(found);
}

TEST_CASE("registry: omniasr-300m has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("omniasr-300m", e);
    REQUIRE(found);
    REQUIRE(std::string(e.filename).find("300m") != std::string::npos);
}

TEST_CASE("registry: omniasr-llm has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("omniasr-llm", e);
    REQUIRE(found);
}

TEST_CASE("registry: omniasr-llm-1b has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("omniasr-llm-1b", e);
    REQUIRE(found);
    REQUIRE(std::string(e.filename).find("1b") != std::string::npos);
}

TEST_CASE("registry: granite-4.1 has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("granite-4.1", e);
    REQUIRE(found);
    REQUIRE(std::string(e.filename).find("granite") != std::string::npos);
}

TEST_CASE("registry: gemma4-e2b has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("gemma4-e2b", e);
    REQUIRE(found);
}

TEST_CASE("registry: vibevoice has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("vibevoice", e);
    REQUIRE(found);
}

TEST_CASE("registry: wav2vec2 aligner aliases resolve", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("wav2vec2-aligner", e));
    REQUIRE(e.backend == "wav2vec2-aligner");
    REQUIRE(e.filename.find("wav2vec2") != std::string::npos);

    REQUIRE(crispasr_registry_lookup("wav2vec2-aligner-en", e));
    REQUIRE(e.filename == "wav2vec2-xlsr-en-q4_k.gguf");

    REQUIRE(crispasr_registry_lookup("wav2vec2-aligner-de", e));
    REQUIRE(e.filename.find("german") != std::string::npos);

    for (const auto& [alias, filename_part] : {
             std::pair{"wav2vec2-aligner-fr", "french"},
             std::pair{"wav2vec2-aligner-es", "spanish"},
             std::pair{"wav2vec2-aligner-it", "italian"},
             std::pair{"wav2vec2-aligner-ja", "japanese"},
             std::pair{"wav2vec2-aligner-zh", "chinese-zh-cn"},
             std::pair{"wav2vec2-aligner-nl", "dutch"},
             std::pair{"wav2vec2-aligner-uk", "uk-with-small-lm"},
             std::pair{"wav2vec2-aligner-pt", "portuguese"},
             std::pair{"wav2vec2-aligner-ar", "arabic"},
             std::pair{"wav2vec2-aligner-cs", "cs-250"},
         }) {
        REQUIRE(crispasr_registry_lookup(alias, e));
        REQUIRE(std::string(e.filename).find(filename_part) != std::string::npos);
        REQUIRE(e.backend == alias);
    }
}

TEST_CASE("registry: preferred quant rewrites primary filename", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("chatterbox", e, "q4_k");
    REQUIRE(found);
    REQUIRE(e.filename == "chatterbox-v3-t3-q4_k.gguf");
}

TEST_CASE("registry: companion quant can be resolved independently", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("chatterbox", e, "q4_k");
    REQUIRE(found);
    REQUIRE(e.companion_filename == "chatterbox-v3-s3gen-q4_k.gguf");
}

TEST_CASE("registry: non-quantized companion remains unchanged", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("qwen3-tts", e, "q4_k");
    REQUIRE(found);
    REQUIRE(e.companion_filename == "qwen3-tts-tokenizer-12hz.gguf");
}

TEST_CASE("registry: companion filename lookup resolves the companion entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup_by_filename("qwen3-tts-tokenizer-12hz.gguf", e);
    REQUIRE(found);
    REQUIRE(e.backend == "qwen3-tts");
    REQUIRE(e.filename == "qwen3-tts-tokenizer-12hz.gguf");
    REQUIRE(e.url.find("qwen3-tts-tokenizer-12hz-GGUF") != std::string::npos);
}

TEST_CASE("registry: quantized companion filename lookup preserves the requested quant", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup_by_filename("qwen3-tts-tokenizer-12hz-q8_0.gguf", e);
    REQUIRE(found);
    REQUIRE(e.backend == "qwen3-tts");
    REQUIRE(e.filename == "qwen3-tts-tokenizer-12hz-q8_0.gguf");
    REQUIRE(e.url.find("qwen3-tts-tokenizer-12hz-q8_0.gguf") != std::string::npos);
}

// ── companion_approx_size (#146 / #148) ──────────────────────────────

TEST_CASE("registry: companion_approx_size populated for mimo-asr", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("mimo-asr", e));
    REQUIRE(!e.companion_filename.empty());
    REQUIRE(!e.companion_approx_size.empty());
    REQUIRE(e.companion_approx_size != e.approx_size); // tokenizer != LM size
    REQUIRE(e.companion_approx_size.find("MB") != std::string::npos);
}

TEST_CASE("registry: companion_approx_size populated for qwen3-tts", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("qwen3-tts", e));
    REQUIRE(!e.companion_approx_size.empty());
    REQUIRE(e.companion_approx_size != e.approx_size);
}

TEST_CASE("registry: companion_approx_size populated for orpheus", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("orpheus", e));
    REQUIRE(!e.companion_approx_size.empty());
    REQUIRE(e.companion_approx_size != e.approx_size);
}

TEST_CASE("registry: companion_approx_size populated for chatterbox", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("chatterbox", e));
    REQUIRE(!e.companion_approx_size.empty());
    REQUIRE(e.companion_approx_size != e.approx_size);
}

TEST_CASE("registry: chatterbox family keeps multilingual and finetunes separate", "[unit][registry]") {
    CrispasrRegistryEntry e;

    REQUIRE(crispasr_registry_lookup("chatterbox", e));
    REQUIRE(e.filename == "chatterbox-v3-t3-q8_0.gguf");
    REQUIRE(e.companion_filename == "chatterbox-v3-s3gen-q8_0.gguf");

    REQUIRE(crispasr_registry_lookup("kartoffelbox-turbo", e));
    REQUIRE(e.filename.find("kartoffelbox-turbo-t3") != std::string::npos);
    REQUIRE(e.companion_filename == "chatterbox-turbo-s3gen-f16.gguf");

    REQUIRE(crispasr_registry_lookup("lahgtna-chatterbox", e));
    REQUIRE(e.filename == "chatterbox-t3-f16.gguf");
    REQUIRE(e.companion_filename == "chatterbox-s3gen-q8_0.gguf");
}

TEST_CASE("registry: companion_approx_size empty for backends without companion", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("whisper", e));
    REQUIRE(e.companion_filename.empty());
    REQUIRE(e.companion_approx_size.empty());
}

TEST_CASE("registry: companion filename lookup uses companion size, not LM size (#146)", "[unit][registry]") {
    CrispasrRegistryEntry e;
    // Look up mimo-tokenizer by filename — should get the tokenizer's
    // size (~395 MB), not the LM's size (~4.2 GB).
    REQUIRE(crispasr_registry_lookup_by_filename("mimo-tokenizer-q4_k.gguf", e));
    REQUIRE(e.approx_size.find("MB") != std::string::npos);
    REQUIRE(e.approx_size.find("GB") == std::string::npos); // must NOT be the LM's 4.2 GB
}

TEST_CASE("registry: qwen3-tts tokenizer filename lookup uses companion size (#146)", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup_by_filename("qwen3-tts-tokenizer-12hz.gguf", e));
    // The tokenizer is ~60 MB, the LM is ~986 MB. The size should
    // reflect the tokenizer, not the LM.
    REQUIRE(e.approx_size.find("60") != std::string::npos);
}

TEST_CASE("registry: all entries with companions have companion_approx_size set", "[unit][registry]") {
    const int n = crispasr_registry_count();
    for (int i = 0; i < n; ++i) {
        CrispasrRegistryEntry e;
        if (!crispasr_registry_get_at(i, e))
            continue;
        if (!e.companion_filename.empty()) {
            REQUIRE(!e.companion_approx_size.empty());
        }
    }
}

// ── #152: fill() robustness — no entry may crash on lookup ──────────

TEST_CASE("registry: every entry fills without crash (#152)", "[unit][registry]") {
    // The SIGSEGV in #152 was heap corruption, but this test ensures
    // fill() handles every Entry in the static table without tripping
    // on NULL fields or malformed strings.
    const int n = crispasr_registry_count();
    REQUIRE(n > 0);
    for (int i = 0; i < n; ++i) {
        CrispasrRegistryEntry e;
        bool ok = crispasr_registry_get_at(i, e);
        REQUIRE(ok);
        REQUIRE(!e.backend.empty());
        REQUIRE(!e.filename.empty());
        REQUIRE(!e.url.empty());
        REQUIRE(!e.approx_size.empty());
        // companion fields: either both set or both empty
        REQUIRE((e.companion_filename.empty() == e.companion_url.empty()));
    }
}

TEST_CASE("registry: fill with preferred quant does not crash on any entry (#152)", "[unit][registry]") {
    const int n = crispasr_registry_count();
    for (int i = 0; i < n; ++i) {
        CrispasrRegistryEntry e;
        // Exercise the quant-rewriting path for every entry
        crispasr_registry_get_at(i, e);
        CrispasrRegistryEntry eq;
        crispasr_registry_lookup(e.backend, eq, "q4_k");
        REQUIRE(!eq.backend.empty());
        REQUIRE(!eq.filename.empty());
    }
}

// ── v0.7.0 new backends: verify they exist in the registry ──────────

TEST_CASE("registry: zonos has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("zonos", e));
    REQUIRE(e.filename.find("zonos") != std::string::npos);
}

TEST_CASE("registry: kugelaudio has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("kugelaudio", e));
}

TEST_CASE("registry: melotts has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("melotts", e));
}

TEST_CASE("registry: cosyvoice3-tts has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("cosyvoice3-tts", e));
}

// The CLI / kaggle benchmark pass the short `--backend cosyvoice3` alias to
// `-m auto`; the registry must resolve it to the canonical `cosyvoice3-tts`
// entry via the `-tts`-suffix fallback (else `-m auto` fails instantly with
// "no default model registered" — full-backend-sweep cosyvoice3 FAIL).
TEST_CASE("registry: cosyvoice3 short alias resolves via -tts fallback", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("cosyvoice3", e));
    REQUIRE(e.filename == "cosyvoice3-llm-q4_k.gguf");
}

// #334: upstream ships a second talker (llm.rl.pt, RL-tuned). It is the same
// engine with a different LLM GGUF, so the alias must resolve to the RL file
// while pulling the SAME flow/HiFT/CAMPPlus/s3tok/voices companions — a copy
// of the base row that forgot to swap `filename` would silently hand the user
// the base talker under the RL name.
TEST_CASE("registry: cosyvoice3-tts-rl swaps only the talker", "[unit][registry]") {
    CrispasrRegistryEntry base;
    CrispasrRegistryEntry rl;
    REQUIRE(crispasr_registry_lookup("cosyvoice3-tts", base));
    REQUIRE(crispasr_registry_lookup("cosyvoice3-tts-rl", rl));
    REQUIRE(rl.filename == "cosyvoice3-llm-rl-q4_k.gguf");
    REQUIRE(rl.filename != base.filename);
    REQUIRE(rl.url != base.url);
    // Companion (flow) is shared, not a second copy under a new name.
    REQUIRE(rl.companion_filename == base.companion_filename);
    REQUIRE(rl.companion_url == base.companion_url);
}

TEST_CASE("registry: dia has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("dia", e));
}

TEST_CASE("registry: f5-tts has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("f5-tts", e));
}

TEST_CASE("registry: bark has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("bark", e));
}

TEST_CASE("registry: piper has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("piper", e));
}

TEST_CASE("registry: csm (sesame) has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("csm", e));
}

TEST_CASE("registry: pocket-tts has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("pocket-tts", e));
}

TEST_CASE("registry: speecht5 has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("speecht5", e));
}

TEST_CASE("registry: fastpitch has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("fastpitch", e));
}

TEST_CASE("registry: parler-tts has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("parler-tts", e));
}

TEST_CASE("registry: voxcpm2-tts has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("voxcpm2-tts", e));
}

TEST_CASE("registry: moss-audio has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("moss-audio", e));
}

TEST_CASE("registry: moss-transcribe has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("moss-transcribe", e));
}

TEST_CASE("registry: sensevoice has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("sensevoice", e));
}

TEST_CASE("registry: paraformer has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("paraformer", e));
}

TEST_CASE("registry: outetts has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("outetts", e));
}
