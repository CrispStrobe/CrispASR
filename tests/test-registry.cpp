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

TEST_CASE("registry: omniasr-llm has entry", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("omniasr-llm", e);
    REQUIRE(found);
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

TEST_CASE("registry: preferred quant rewrites primary filename", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("chatterbox", e, "q4_k");
    REQUIRE(found);
    REQUIRE(e.filename == "chatterbox-t3-q4_k.gguf");
}

TEST_CASE("registry: companion quant can be resolved independently", "[unit][registry]") {
    CrispasrRegistryEntry e;
    bool found = crispasr_registry_lookup("chatterbox", e, "q4_k");
    REQUIRE(found);
    REQUIRE(e.companion_filename == "chatterbox-s3gen-q4_k.gguf");
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
