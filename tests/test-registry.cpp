// test-registry.cpp — unit tests for crispasr_model_registry.
//
// Verifies registry lookup, backend listing, and filename-based reverse
// lookup. No network, no models — pure in-memory registry queries.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_model_registry.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

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

TEST_CASE("registry: Nemotron defaults only to the pinned qualified F16", "[unit][registry][nemotron]") {
    CrispasrRegistryEntry entry;
    REQUIRE(crispasr_registry_lookup("nemotron", entry));
    REQUIRE(entry.filename == "nemotron-3.5-asr-streaming-0.6b-f16.gguf");
    REQUIRE(entry.url == "https://huggingface.co/cstr/nemotron-3.5-asr-streaming-GGUF/resolve/"
                         "bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2/nemotron-3.5-asr-streaming-0.6b-f16.gguf");
    REQUIRE(entry.approx_size == "~1.3 GB");
    REQUIRE(entry.license == "OpenMDW-1.1");
    REQUIRE(crispasr_registry_lookup("nemotron", entry, "f16"));
    REQUIRE(entry.filename == "nemotron-3.5-asr-streaming-0.6b-f16.gguf");
    for (const char* quant : {"q4_k", "q8_0", "q6_k", "bf16", "f32"}) {
        REQUIRE_FALSE(crispasr_registry_lookup("nemotron", entry, quant));
        REQUIRE_FALSE(crispasr_registry_lookup_by_filename("nemotron-3.5-asr-streaming-0.6b-f16.gguf", entry, quant));
        REQUIRE_FALSE(crispasr_registry_lookup_by_filename(
            std::string("nemotron-3.5-asr-streaming-0.6b-") + quant + ".gguf", entry));
    }
}

TEST_CASE("registry: Nemotron default retains full OpenMDW terms and origin", "[unit][registry][nemotron]") {
    CrispasrRegistryBundle bundle;
    REQUIRE(crispasr_registry_default_bundle("nemotron", bundle));
    REQUIRE(bundle.license == "OpenMDW-1.1");
    REQUIRE_FALSE(bundle.requires_license_acceptance);
    REQUIRE(bundle.artifacts.size() == 3);
    REQUIRE(bundle.artifacts[0].filename == "nemotron-3.5-asr-streaming-0.6b-f16.gguf");
    REQUIRE(bundle.artifacts[0].kind == CrispasrRegistryArtifactKind::Primary);
    const char* names[] = {"LICENSE-OpenMDW-1.1.txt", "NOTICE.txt"};
    for (size_t i = 1; i < bundle.artifacts.size(); ++i) {
        REQUIRE(bundle.artifacts[i].kind == CrispasrRegistryArtifactKind::Extra);
        REQUIRE(bundle.artifacts[i].filename == std::string("nemotron-3.5-asr-streaming-0.6b/") + names[i - 1]);
        REQUIRE(bundle.artifacts[i].url ==
                std::string("https://raw.githubusercontent.com/CrispStrobe/CrispASR/"
                            "102cd6bfc233c3d5d211966718d3de0e4401f836/docs/nemotron-model-notices/") +
                    names[i - 1]);
    }
}

TEST_CASE("resolver: Nemotron Q4 cache and filenames cannot substitute for the F16 default",
          "[unit][registry][nemotron]") {
    const std::filesystem::path cache = "test-registry-nemotron-cache";
    const std::string prefix = "nemotron-3.5-asr-streaming-0.6b-";
    const std::string f16 = prefix + "f16.gguf", q4 = prefix + "q4_k.gguf";
    std::filesystem::create_directories(cache);
    {
        std::ofstream fixture(cache / q4, std::ios::binary);
        fixture << "existing explicitly selected diagnostic model fixture";
    }
    CrispasrRegistryEntry entry;
    REQUIRE(crispasr_registry_lookup("nemotron", entry));
    REQUIRE(entry.filename == f16);
    CHECK_FALSE(crispasr_find_cached_model(entry, cache.string()));
    CHECK(crispasr_resolve_model(q4, "nemotron", true, cache.string(), false) == (cache / q4).string());
    std::filesystem::remove(cache / q4);
    {
        std::ofstream fixture(cache / f16, std::ios::binary);
        fixture << "qualified default identity fixture";
    }
    REQUIRE(crispasr_find_cached_model(entry, cache.string()));
    CHECK(entry.filename == f16);
    for (const char* quant : {"q4_k", "q8_0", "q6_k"}) {
        const std::string requested = prefix + quant + ".gguf";
        CHECK(crispasr_resolve_model(requested, "nemotron", true, cache.string(), false) == requested);
        CHECK_FALSE(crispasr_find_cached_model(entry, cache.string(), quant));
    }
    CHECK(crispasr_resolve_model(f16, "nemotron", true, cache.string(), false) == (cache / f16).string());
    std::filesystem::remove_all(cache);
}

TEST_CASE("registry: Phonon-2 selects Q8 and resolves alternate GGUF quants", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("phonon2", e));
    REQUIRE(e.filename == "phonon2-q8_0.gguf");
    REQUIRE(e.approx_size == "~674 MB");
    REQUIRE(e.license.find("CC-BY-4.0") != std::string::npos);
    REQUIRE(crispasr_registry_lookup("phonon2", e, "f16"));
    REQUIRE(e.filename == "phonon2-f16.gguf");
    REQUIRE(e.url.find("phonon2-f16.gguf") != std::string::npos);
    REQUIRE(e.approx_size.empty());
    REQUIRE(crispasr_registry_lookup("phonon2", e, "q4_k"));
    REQUIRE(e.backend == "phonon2");
    REQUIRE(e.filename == "phonon2-q4_k.gguf");
    REQUIRE(crispasr_registry_lookup_by_filename("phonon2-q8_0.gguf", e));
    REQUIRE(e.backend == "phonon2");
}

TEST_CASE("registry: default bundle reports the exact canonical artifacts", "[unit][registry]") {
    CrispasrRegistryBundle bundle;
    REQUIRE(crispasr_registry_default_bundle("omnivoice", bundle));
    REQUIRE(bundle.backend == "omnivoice");
    REQUIRE(bundle.requires_license_acceptance);
    REQUIRE(bundle.license.find("omnivoice-terms") == 0);
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
        {"omnivoice", "omnivoice-terms", true},
        {"supertonic", "OpenRAIL-M", true},
    };
    for (const auto& c : cases) {
        CrispasrRegistryEntry e;
        REQUIRE(crispasr_registry_lookup(c.backend, e));
        REQUIRE(std::string(e.license).find(c.needle) != std::string::npos);
        REQUIRE(crispasr_license_requires_acceptance(e.license) == c.requires_acceptance);
    }
}

TEST_CASE("registry: NC acceptance does not accept a custom bundle license", "[unit][registry][license]") {
    REQUIRE(crispasr_license_requires_acceptance("CC-BY-NC-4.0"));
    REQUIRE(crispasr_license_accepted("CC-BY-NC-4.0", "cc-by-nc-4.0"));
    REQUIRE_FALSE(crispasr_license_accepted("omnivoice-terms", "cc-by-nc-4.0"));
    REQUIRE_FALSE(crispasr_license_accepted("OpenRAIL-M", "cc-by-nc-4.0"));
    REQUIRE(crispasr_license_requires_acceptance("CreativeML-OpenRAIL-M"));
    REQUIRE(crispasr_license_accepted("OpenRAIL-M", "openrail-m"));
    REQUIRE_FALSE(crispasr_license_requires_acceptance("CC-BY-4.0"));
    REQUIRE_FALSE(crispasr_license_requires_acceptance("MIT"));
}

TEST_CASE("registry: NVIDIA and OmniVoice acceptance affirm distinct full terms", "[unit][registry][license]") {
    REQUIRE(crispasr_license_requires_acceptance("NVIDIA-Open-Model-License"));
    REQUIRE(crispasr_license_accepted("NVIDIA-Open-Model-License", "nvidia-open-model-license"));
    REQUIRE_FALSE(crispasr_license_accepted("NVIDIA-Open-Model-License", "cc-by-nc-4.0"));
    REQUIRE_FALSE(crispasr_license_accepted("NVIDIA-Open-Model-License", "apache-2.0"));
    REQUIRE(crispasr_license_requires_acceptance("omnivoice-terms"));
    REQUIRE(crispasr_license_accepted("omnivoice-terms", "omnivoice-terms"));
    REQUIRE_FALSE(crispasr_license_accepted("omnivoice-terms", "nvidia-open-model-license"));
    REQUIRE_FALSE(crispasr_license_accepted("NVIDIA-Open-Model-License", "omnivoice-terms"));
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("omnivoice", e));
    REQUIRE(e.license.find("100,000") != std::string::npos);
    REQUIRE(e.license.find("materials/outputs") != std::string::npos);
}

TEST_CASE("registry: the Breeze TTS 2 non-commercial gate actually blocks", "[unit][registry][license]") {
    // #412. The weights are non-commercial and this is the only thing standing
    // between `-m auto` and downloading them without acceptance, so it gets its
    // own test rather than a row in the table above.
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("bt2-tts", e));

    // The prose a user is shown must actually say what the restriction is.
    REQUIRE(std::string(e.license).find("NON-COMMERCIAL") != std::string::npos);
    REQUIRE(std::string(e.license).find("BreezeBlue") != std::string::npos);

    // The gate fires.
    REQUIRE(crispasr_license_requires_acceptance(e.license));

    // ...and the bundle -m auto consults agrees. A gate that is true in one
    // accessor and false in the one the download path reads is not a gate.
    CrispasrRegistryBundle bundle;
    REQUIRE(crispasr_registry_default_bundle("bt2-tts", bundle));
    REQUIRE(bundle.requires_license_acceptance);

    // The codec companion must be listed, or a -m auto user gets a model that
    // loads and cannot render audio.
    REQUIRE(e.companion_filename == "qwen3-tts-tokenizer-12hz.gguf");

    // CONTROL. Everything above would also pass if
    // crispasr_license_requires_acceptance() simply returned true for every
    // input — so prove it can say no. This is the pairing the "readout must be
    // able to report failure" rule exists for.
    REQUIRE_FALSE(crispasr_license_requires_acceptance("apache-2.0"));
    REQUIRE_FALSE(crispasr_license_requires_acceptance("CC-BY-4.0 — attribution required"));

    // SECOND CONTROL, on the tag parser rather than the predicate: the gate
    // keys on the FIRST WORD of the licence string. If someone reworded the
    // entry to lead with a permissive word and left "other" later in the
    // sentence, the gate would silently stop firing while the prose still read
    // as restrictive.
    REQUIRE_FALSE(crispasr_license_requires_acceptance("apache-2.0 (not other, despite the word other here)"));
}

TEST_CASE("registry: default bundle rejects unknown backends", "[unit][registry]") {
    CrispasrRegistryBundle bundle;
    REQUIRE_FALSE(crispasr_registry_default_bundle("nonexistent-backend-xyz", bundle));
}

TEST_CASE("registry: Melo V2 bundle uses the qualified discoverable BERT companion", "[unit][registry]") {
    CrispasrRegistryBundle bundle;
    REQUIRE(crispasr_registry_default_bundle("melotts", bundle));
    REQUIRE_FALSE(bundle.requires_license_acceptance);
    REQUIRE(bundle.license.find("Apache-2.0") != std::string::npos);
    REQUIRE(bundle.artifacts.size() == 2);
    REQUIRE(bundle.artifacts[0].filename == "melotts-en-v2-f16.gguf");
    REQUIRE(bundle.artifacts[1].filename == "bert-base-uncased.gguf");
    for (const auto& artifact : bundle.artifacts) {
        REQUIRE(artifact.url.find("/resolve/35522347b1bed9a913d6894803c5cab929ae42a5/") != std::string::npos);
    }
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

TEST_CASE("registry: haas-ja has entry (#561)", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("haas", e));
    REQUIRE(std::string(e.filename) == "haas-ja-v1.0-q8_0.gguf");
    REQUIRE(crispasr_registry_lookup("haas-ja", e));
    REQUIRE(std::string(e.filename) == "haas-ja-v1.0-q8_0.gguf");
    REQUIRE(crispasr_registry_lookup("haas-ja", e, "q4_k"));
    REQUIRE(std::string(e.filename) == "haas-ja-v1.0-q4_k.gguf");
    REQUIRE(crispasr_registry_lookup("haas-ja", e, "f16"));
    REQUIRE(std::string(e.filename) == "haas-ja-v1.0-f16.gguf");
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

TEST_CASE("registry: Granite5 defaults to qualified Q4 and preserves explicit F16", "[unit][registry][granite5]") {
    CrispasrRegistryEntry entry;
    REQUIRE(crispasr_registry_lookup("granite", entry));
    REQUIRE(entry.filename == "granite-speech-5.0-2b-q4_k.gguf");
    REQUIRE(entry.license == "Apache-2.0");
    REQUIRE(entry.url.find("/granite-speech-5.0-2b-GGUF/resolve/") != std::string::npos);
    REQUIRE(entry.url.find("/resolve/main/") == std::string::npos);
    REQUIRE_FALSE(entry.approx_size.empty());
    const std::string default_url = entry.url;
    for (const char* quant : {"q4_k", "f16"}) {
        REQUIRE(crispasr_registry_lookup("granite", entry, quant));
        REQUIRE(entry.filename == std::string("granite-speech-5.0-2b-") + quant + ".gguf");
        REQUIRE_FALSE(entry.approx_size.empty());
        REQUIRE(crispasr_registry_lookup_by_filename(entry.filename, entry));
        REQUIRE(entry.backend == "granite");
        if (std::string(quant) == "f16")
            REQUIRE(entry.url.find("/resolve/701f5529d8459412cc1b6053b5880c630125b138/") != std::string::npos);
        else
            REQUIRE(entry.url == default_url);
    }
    for (const char* quant : {"q8_0", "q6_k", "q5_k", "q2_k", "bf16", "f32"}) {
        REQUIRE_FALSE(crispasr_registry_lookup("granite", entry, quant));
        REQUIRE_FALSE(crispasr_registry_lookup_by_filename("granite-speech-5.0-2b-f16.gguf", entry, quant));
        REQUIRE_FALSE(
            crispasr_registry_lookup_by_filename(std::string("granite-speech-5.0-2b-") + quant + ".gguf", entry));
    }
    REQUIRE(crispasr_registry_lookup("granite-4.1", entry, "q4_k"));
    REQUIRE(entry.filename == "granite-speech-4.1-2b-q4_k.gguf");
    REQUIRE(crispasr_registry_lookup_by_filename("granite-speech-4.0-1b-q4_k.gguf", entry));
    REQUIRE(entry.url.find("/granite-speech-4.0-1b-GGUF/") != std::string::npos);
    int profiles = 0;
    for (int i = 0; i < crispasr_registry_count(); ++i) {
        REQUIRE(crispasr_registry_get_at(i, entry));
        if (entry.filename.find("granite-speech-5.0-2b-") != 0)
            continue;
        ++profiles;
        REQUIRE(crispasr_registry_get_at(i, entry, "q4_k"));
        REQUIRE(entry.filename == "granite-speech-5.0-2b-q4_k.gguf");
        REQUIRE(crispasr_registry_get_at(i, entry, "f16"));
        REQUIRE(entry.filename == "granite-speech-5.0-2b-f16.gguf");
        REQUIRE_FALSE(crispasr_registry_get_at(i, entry, "q8_0"));
        REQUIRE_FALSE(crispasr_registry_get_at(i, entry, "q6_k"));
    }
    REQUIRE(profiles == 2);
}

TEST_CASE("registry: Granite5 downloads the complete pinned legal bundle", "[unit][registry][granite5]") {
    CrispasrRegistryBundle bundle;
    REQUIRE(crispasr_registry_default_bundle("granite", bundle));
    REQUIRE(bundle.license == "Apache-2.0");
    REQUIRE_FALSE(bundle.requires_license_acceptance);
    REQUIRE(bundle.artifacts.size() == 6);
    REQUIRE(bundle.artifacts.front().filename == "granite-speech-5.0-2b-q4_k.gguf");
    REQUIRE(bundle.artifacts.front().kind == CrispasrRegistryArtifactKind::Primary);
    const std::string primary_url = bundle.artifacts.front().url;
    const std::string release_prefix = primary_url.substr(0, primary_url.rfind('/') + 1);
    const char* names[] = {"LICENSE", "LICENSE-RECIPE.txt", "NOTICE", "original-publisher-README.md",
                           "validation-evidence.json"};
    for (size_t i = 1; i < bundle.artifacts.size(); ++i) {
        const auto& artifact = bundle.artifacts[i];
        REQUIRE(artifact.kind == CrispasrRegistryArtifactKind::Extra);
        REQUIRE(artifact.filename == std::string("granite-speech-5.0-2b/") + names[i - 1]);
        REQUIRE(artifact.url.find("/granite-speech-5.0-2b-GGUF/resolve/") != std::string::npos);
        REQUIRE(artifact.url.find("/resolve/main/") == std::string::npos);
        REQUIRE(artifact.url == release_prefix + names[i - 1]);
    }
}

TEST_CASE("resolver: cached F16 cannot satisfy the Granite5 Q4 default", "[unit][registry][granite5]") {
    const std::filesystem::path cache = "test-registry-granite5-cache";
    const std::string f16 = "granite-speech-5.0-2b-f16.gguf";
    const std::string q4 = "granite-speech-5.0-2b-q4_k.gguf";
    std::filesystem::create_directories(cache);
    {
        std::ofstream fixture(cache / f16, std::ios::binary);
        fixture << "fixture";
    }
    CHECK(crispasr_resolve_model(q4, "granite", true, cache.string(), false) == q4);
    CrispasrRegistryEntry entry;
    REQUIRE(crispasr_registry_lookup("granite", entry));
    CHECK(entry.filename == q4);
    CHECK_FALSE(std::filesystem::exists(cache / entry.filename));
    // Inspect the cache with the default's exact quant; auto-resolution can
    // fetch model/notice files and does not belong in a model-free unit test.
    CHECK_FALSE(crispasr_find_cached_model(entry, cache.string(), "q4_k"));
    CHECK(crispasr_find_cached_model(entry, cache.string(), "f16"));
    CHECK(entry.filename == f16);
    CHECK(crispasr_resolve_model(f16, "granite", true, cache.string(), false) == (cache / f16).string());
    for (const char* rejected : {"q8_0", "q6_k"}) {
        CHECK_FALSE(crispasr_find_cached_model(entry, cache.string(), rejected));
        CHECK(crispasr_resolve_model("auto", "granite", true, cache.string(), false, rejected).empty());
    }
    {
        std::ofstream fixture(cache / q4, std::ios::binary);
        fixture << "qualified Q4 cache fixture";
    }
    CHECK(crispasr_find_cached_model(entry, cache.string(), "q4_k"));
    CHECK(entry.filename == q4);
    CHECK(crispasr_find_cached_model(entry, cache.string(), "f16"));
    CHECK(entry.filename == f16);
    std::filesystem::remove(cache / q4);
    std::filesystem::remove(cache / f16);
    std::filesystem::remove(cache);
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

TEST_CASE("registry: vibevoice streaming resolves the validated Q4", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("vibevoice-streaming", e));
    REQUIRE(e.filename == "vibevoice-asr-streaming-1.5b-q4_k.gguf");
    REQUIRE(e.url.find("vibevoice-asr-streaming-1.5b-GGUF") != std::string::npos);
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

    REQUIRE(crispasr_registry_lookup("chatterbox-finnish-nano", e));
    REQUIRE(e.filename == "chatterbox-finnish-nano-v0.1.3-t3-q8_0.gguf");
    REQUIRE(e.url.find("JJarvinen/chatterbox-finnish-nano-GGUF") != std::string::npos);
    REQUIRE(e.companion_filename == "chatterbox-turbo-s3gen-q8_0.gguf");

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
        const bool found = crispasr_registry_lookup(e.backend, eq, "q4_k");
        if (e.backend == "nemotron") {
            // Only F16 is qualified; manufacturing a Q4 download is rejected.
            REQUIRE_FALSE(found);
            continue;
        }
        REQUIRE(found);
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

TEST_CASE("resolver: exact unregistered Piper voice wins over backend default (#397)", "[unit][registry]") {
    const std::filesystem::path cache = "test-registry-piper-cache";
    const std::string filename = "piper-en_GB-cori-medium-f16.gguf";
    std::filesystem::create_directories(cache);
    {
        std::ofstream fixture(cache / filename, std::ios::binary);
        fixture << "fixture";
    }

    const std::string resolved = crispasr_resolve_model(filename, "piper", /*quiet=*/true, cache.string(),
                                                        /*allow_download=*/false);
    CHECK(resolved == (cache / filename).string());
    CHECK(resolved.find("lessac") == std::string::npos);

    std::filesystem::remove(cache / filename);
    std::filesystem::remove(cache);
}

TEST_CASE("registry: csm (sesame) has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("csm", e));
}

TEST_CASE("registry: pocket-tts has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    REQUIRE(crispasr_registry_lookup("pocket-tts", e));
}

TEST_CASE("registry: Pocket-TTS language checkpoints are wired", "[unit][registry]") {
    const std::pair<const char*, const char*> variants[] = {
        {"pocket-tts-de", "pocket-tts-german-q8_0.gguf"},     {"pocket-tts-es", "pocket-tts-spanish-q8_0.gguf"},
        {"pocket-tts-it", "pocket-tts-italian-q8_0.gguf"},    {"pocket-tts-pt", "pocket-tts-portuguese-q8_0.gguf"},
        {"pocket-tts-fr", "pocket-tts-french_24l-q8_0.gguf"},
    };
    for (const auto& [backend, filename] : variants) {
        CAPTURE(backend);
        CrispasrRegistryEntry e;
        REQUIRE(crispasr_registry_lookup(backend, e));
        REQUIRE(e.filename == filename);
        REQUIRE(e.url.find("cstr/pocket-tts-GGUF") != std::string::npos);
    }
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

TEST_CASE("registry: Index-Echo downloads a matching pair and source VAD", "[unit][registry]") {
    CrispasrRegistryBundle bundle;
    REQUIRE(crispasr_registry_default_bundle("index-echo", bundle));
    REQUIRE_FALSE(bundle.requires_license_acceptance);
    REQUIRE(bundle.artifacts.size() == 3);
    REQUIRE(bundle.artifacts[0].filename == "index-echo-2b-q8_0.gguf");
    REQUIRE(bundle.artifacts[1].filename == "index-echo-2b-decoder-q8_0.gguf");
    REQUIRE(bundle.artifacts[2].filename == "ggml-silero-v6.2.0.bin");
    CrispasrRegistryEntry entry;
    REQUIRE(crispasr_registry_lookup("index-echo", entry, "f16"));
    REQUIRE(entry.filename == "index-echo-2b-f16.gguf");
    REQUIRE(entry.companion_filename == "index-echo-2b-decoder-f16.gguf");
    REQUIRE(crispasr_registry_lookup_by_filename("index-echo-9b-f16.gguf", entry));
    REQUIRE(entry.backend == "index-echo");
    REQUIRE(entry.filename == "index-echo-9b-f16.gguf");
    REQUIRE(entry.companion_filename == "index-echo-9b-decoder-f16.gguf");
    REQUIRE(entry.url.find("/index-echo-9b-GGUF/") != std::string::npos);
    REQUIRE(entry.companion_url.find("/index-echo-9b-GGUF/") != std::string::npos);
    REQUIRE_FALSE(crispasr_license_requires_acceptance(entry.license));
}

TEST_CASE("registry: German Moonshine bundles own every companion", "[unit][registry]") {
    for (const auto& key : {"moonshine-streaming-tiny-de-onnx", "moonshine-streaming-small-de-onnx",
                            "moonshine-streaming-tiny-de-onnx-f32", "moonshine-streaming-small-de-onnx-f32"}) {
        CAPTURE(key);
        CrispasrRegistryBundle bundle;
        REQUIRE(crispasr_registry_default_bundle(key, bundle));
        REQUIRE(bundle.backend == key);
        REQUIRE(bundle.artifacts.size() == 8);
        REQUIRE(bundle.license == "MIT");
        REQUIRE_FALSE(bundle.requires_license_acceptance);
        for (const auto& artifact : bundle.artifacts) {
            REQUIRE(artifact.filename.find(std::string(key) + "/") == 0);
            REQUIRE(artifact.url.find("/resolve/main/") == std::string::npos);
        }
        const bool f32 = std::string(key).find("-f32") != std::string::npos;
        REQUIRE(bundle.artifacts.front().filename == std::string(key) + (f32 ? "/encoder.onnx" : "/encoder_int8.onnx"));
    }
}

TEST_CASE("registry: Base bundles download model-sized voices with notices (#555)", "[unit][registry]") {
    for (const auto& pair : {std::make_pair("qwen3-tts", "0.6b"), std::make_pair("qwen3-tts-1.7b-base", "1.7b")}) {
        CrispasrRegistryBundle bundle;
        REQUIRE(crispasr_registry_default_bundle(pair.first, bundle));
        CrispasrRegistryEntry entry;
        REQUIRE(crispasr_registry_lookup(pair.first, entry));
        REQUIRE(entry.license == "Apache-2.0");
        const std::string voice = std::string("qwen3-tts-voice-") + pair.second + "-designed.gguf";
        int companions = 0;
        for (const auto& item : bundle.artifacts) {
            REQUIRE(item.filename != "qwen3-tts-voice-default.gguf");
            if (item.filename.find("qwen3-tts-voice-") == 0) {
                REQUIRE(item.filename == voice);
                ++companions;
            }
            if (item.url.find("qwen3-tts-voices-GGUF/") != std::string::npos) {
                REQUIRE(item.url.find("/resolve/9ab96c5ba3022bd2ed46306ded0489a24096f8b5/") != std::string::npos);
                if (item.filename != voice)
                    ++companions;
            }
        }
        REQUIRE(companions == 5); // Matching pack, Apache/MIT/NOTICE and full provenance.
    }
}
