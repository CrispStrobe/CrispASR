// test_punc_resolve.cpp — unit tests for the shared --punc-model resolver.
//
// The resolver (crispasr_punc_loader.h) is the single source of truth for the
// alias → (cache filename, download URL) table used by both the CLI one-shot
// path and the HTTP server. Pin every alias and edge case here so the two
// front-ends can't silently disagree on which model a flag value selects.

#include "crispasr_punc_loader.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("disabled values resolve to none", "[unit][punctuation]") {
    for (const char* v : {"", "none", "off"}) {
        const auto s = crispasr_resolve_punc_model(v);
        REQUIRE(s.kind == crispasr_punc_kind::none);
        REQUIRE(s.cache_filename.empty());
        REQUIRE(s.url.empty());
        REQUIRE(s.direct_path.empty());
    }
}

TEST_CASE("FireRedPunc aliases map to their GGUFs", "[unit][punctuation]") {
    SECTION("auto and firered share the default FireRedPunc model") {
        for (const char* v : {"auto", "firered"}) {
            const auto s = crispasr_resolve_punc_model(v);
            REQUIRE(s.kind == crispasr_punc_kind::fireredpunc);
            REQUIRE(s.cache_filename == "fireredpunc-q4_k.gguf");
            REQUIRE(s.url == "https://huggingface.co/cstr/fireredpunc-GGUF/resolve/main/fireredpunc-q4_k.gguf");
            REQUIRE(s.direct_path.empty());
        }
    }
    SECTION("fullstop") {
        const auto s = crispasr_resolve_punc_model("fullstop");
        REQUIRE(s.kind == crispasr_punc_kind::fireredpunc);
        REQUIRE(s.cache_filename == "fullstop-punc-q4_k.gguf");
        REQUIRE(s.url ==
                "https://huggingface.co/cstr/fullstop-punc-multilang-GGUF/resolve/main/fullstop-punc-q4_k.gguf");
    }
    SECTION("punctuate-all") {
        const auto s = crispasr_resolve_punc_model("punctuate-all");
        REQUIRE(s.kind == crispasr_punc_kind::fireredpunc);
        REQUIRE(s.cache_filename == "punctuate-all-q4_k.gguf");
        REQUIRE(s.url == "https://huggingface.co/cstr/punctuate-all-GGUF/resolve/main/punctuate-all-q4_k.gguf");
    }
}

TEST_CASE("pcs alias maps to the mixed-precision PCS GGUF", "[unit][punctuation]") {
    const auto s = crispasr_resolve_punc_model("pcs");
    REQUIRE(s.kind == crispasr_punc_kind::pcs);
    REQUIRE(s.cache_filename == "pcs-xlmr-base-q4_k-mixed.gguf");
    REQUIRE(s.url == "https://huggingface.co/cstr/pcs-xlmr-base-GGUF/resolve/"
                     "a685c2ead4c14a590ba657c1ea5eaa1b7da7f955/pcs-xlmr-base-q4_k-mixed.gguf");
    REQUIRE(s.direct_path.empty());
}

TEST_CASE("previous PCS defaults stay selectable", "[unit][punctuation]") {
    SECTION("pcs-q4k") {
        const auto s = crispasr_resolve_punc_model("pcs-q4k");
        REQUIRE(s.kind == crispasr_punc_kind::pcs);
        REQUIRE(s.cache_filename == "pcs-xlmr-base-q4_k.gguf");
        REQUIRE(s.url == "https://huggingface.co/cstr/pcs-xlmr-base-GGUF/resolve/main/pcs-xlmr-base-q4_k.gguf");
    }
    SECTION("pcs-q4k-imatrix") {
        const auto s = crispasr_resolve_punc_model("pcs-q4k-imatrix");
        REQUIRE(s.kind == crispasr_punc_kind::pcs);
        REQUIRE(s.cache_filename == "pcs-xlmr-base-q4_k-imatrix.gguf");
        REQUIRE(s.url ==
                "https://huggingface.co/cstr/pcs-xlmr-base-GGUF/resolve/main/pcs-xlmr-base-q4_k-imatrix.gguf");
    }
}

TEST_CASE("direct paths are passed through, not downloaded", "[unit][punctuation]") {
    SECTION("a plain .gguf path is a FireRedPunc model") {
        const auto s = crispasr_resolve_punc_model("/models/my-fireredpunc.gguf");
        REQUIRE(s.kind == crispasr_punc_kind::fireredpunc);
        REQUIRE(s.direct_path == "/models/my-fireredpunc.gguf");
        REQUIRE(s.cache_filename.empty());
        REQUIRE(s.url.empty());
    }
    SECTION("a path containing 'pcs' AND .gguf is a PCS model") {
        const auto s = crispasr_resolve_punc_model("/models/pcs-xlmr-custom.gguf");
        REQUIRE(s.kind == crispasr_punc_kind::pcs);
        REQUIRE(s.direct_path == "/models/pcs-xlmr-custom.gguf");
        REQUIRE(s.cache_filename.empty());
        REQUIRE(s.url.empty());
    }
}

TEST_CASE("a 'pcs' keyword without a .gguf path resolves to none", "[unit][punctuation]") {
    // Matches the historical behaviour of both front-ends: a bare token that
    // merely mentions pcs (e.g. "pcs-de") loaded neither model.
    const auto s = crispasr_resolve_punc_model("pcs-de");
    REQUIRE(s.kind == crispasr_punc_kind::none);
    REQUIRE(s.direct_path.empty());
    REQUIRE(s.cache_filename.empty());
}
