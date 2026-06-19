// test-core-lang-names.cpp — unit tests for core/lang_names.h
//
// core_lang::iso_to_english() is the single source of truth that replaced
// four copy-pasted ISO-639-1 → English maps (PLAN §175). These tests lock
// in:
//   (a) every mapping the four former copies agreed on (bit-identical);
//   (b) the union coverage (uk/vi from gemma4; ar/ko/pl/tr/hi from the CLI
//       + C-ABI maps) so a future trim can't silently regress a caller;
//   (c) passthrough for unknown / already-spelled-out inputs.
//
// Pure CPU, no models.

#include <catch2/catch_test_macros.hpp>

#include "core/lang_names.h"

#include <string>

using core_lang::iso_to_english;

TEST_CASE("core_lang: common 15 codes map identically", "[unit][core-lang]") {
    // These are the codes the CLI (crispasr_backend_utils.h) and the C ABI
    // (ca_iso_to_english_lang) both mapped — the migration must reproduce
    // them byte-for-byte.
    CHECK(iso_to_english("en") == "English");
    CHECK(iso_to_english("de") == "German");
    CHECK(iso_to_english("fr") == "French");
    CHECK(iso_to_english("es") == "Spanish");
    CHECK(iso_to_english("it") == "Italian");
    CHECK(iso_to_english("pt") == "Portuguese");
    CHECK(iso_to_english("ru") == "Russian");
    CHECK(iso_to_english("ja") == "Japanese");
    CHECK(iso_to_english("ko") == "Korean");
    CHECK(iso_to_english("zh") == "Chinese");
    CHECK(iso_to_english("nl") == "Dutch");
    CHECK(iso_to_english("pl") == "Polish");
    CHECK(iso_to_english("tr") == "Turkish");
    CHECK(iso_to_english("ar") == "Arabic");
    CHECK(iso_to_english("hi") == "Hindi");
}

TEST_CASE("core_lang: gemma4-only union codes (uk/vi)", "[unit][core-lang]") {
    // gemma4's g4e_lang_name added these two; they must survive the merge
    // so gemma4 stays bit-identical after deferring to the shared map.
    CHECK(iso_to_english("uk") == "Ukrainian");
    CHECK(iso_to_english("vi") == "Vietnamese");
}

TEST_CASE("core_lang: unknown/spelled-out passthrough", "[unit][core-lang]") {
    // The former maps all returned the input verbatim for codes they did
    // not know — callers rely on this (e.g. voxtral translate targets).
    CHECK(iso_to_english("xx") == "xx");
    CHECK(iso_to_english("German") == "German"); // already spelled out
    CHECK(iso_to_english("") == "");
    CHECK(iso_to_english("auto") == "auto"); // gemma4 handles "auto" in its wrapper
}
