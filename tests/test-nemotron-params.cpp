// test-nemotron-params.cpp — unit tests for nemotron_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "nemotron.h"
#include "core/nemotron_text.h"

TEST_CASE("nemotron_params: default values are sensible", "[unit][nemotron]") {
    struct nemotron_context_params p = nemotron_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197 + PLAN #89). Pin
// the shipped use_flash/use_gpu defaults so a silent flip fails CI.
TEST_CASE("nemotron_params: gpu/flash defaults are pinned", "[unit][nemotron]") {
    struct nemotron_context_params p = nemotron_context_default_params();
    REQUIRE(p.use_gpu == false);
    REQUIRE(p.use_flash == false);
}

TEST_CASE("nemotron_init_from_file: null path returns nullptr", "[unit][nemotron]") {
    struct nemotron_context_params p = nemotron_context_default_params();
    struct nemotron_context* ctx = nemotron_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("nemotron_init_from_file: empty path returns nullptr", "[unit][nemotron]") {
    struct nemotron_context_params p = nemotron_context_default_params();
    struct nemotron_context* ctx = nemotron_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("nemotron_free: NULL context is a no-op", "[unit][nemotron]") {
    nemotron_free(nullptr);
    SUCCEED("nemotron_free tolerated a NULL ctx.");
}

TEST_CASE("nemotron: language metadata never becomes transcript or words", "[unit][nemotron]") {
    using core_nemotron_text::strip_lang_tags;
    CHECK(strip_lang_tags("Guten Morgen. <de-DE>") == "Guten Morgen.");
    CHECK(strip_lang_tags("Hallo. <de-DE> Guten Tag.") == "Hallo. Guten Tag.");
    CHECK(strip_lang_tags("<de-DE>").empty());
    CHECK(strip_lang_tags("<eng-US>").empty());
    CHECK(strip_lang_tags("Grüße <de-DE> für alle") == "Grüße für alle");
    CHECK(strip_lang_tags("x<de-DE><en-US>y") == "xy");
    CHECK(strip_lang_tags("<DE-de> <d-DE> <de-DEU> <word>") == "<DE-de> <d-DE> <de-DEU> <word>");
    CHECK(strip_lang_tags("unvollständig <de-") == "unvollständig <de-");
}
