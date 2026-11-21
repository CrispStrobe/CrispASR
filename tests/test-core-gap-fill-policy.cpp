// test-core-gap-fill-policy.cpp — src/core/gap_fill_policy.h, the gap-fill
// decisions shared by the CLI/server pass and the session C ABI (#605).
// Hermetic: no model, no audio.

#include "core/gap_fill_policy.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

struct scoped_env {
    const char* name;
    std::string old;
    bool had;
    scoped_env(const char* n, const char* v) : name(n) {
        const char* o = std::getenv(n);
        had = o != nullptr;
        if (had)
            old = o;
        if (v)
            setenv(n, v, 1);
        else
            unsetenv(n);
    }
    ~scoped_env() {
        if (had)
            setenv(name, old.c_str(), 1);
        else
            unsetenv(name);
    }
};

} // namespace

TEST_CASE("gap-fill policy: CRISPASR_GAP_FILL is a tri-state over the backend default", "[unit][gap-fill-policy]") {
    {
        scoped_env e("CRISPASR_GAP_FILL", nullptr);
        REQUIRE(core_gap_fill::enabled(true));
        REQUIRE_FALSE(core_gap_fill::enabled(false));
    }
    {
        scoped_env e("CRISPASR_GAP_FILL", "");
        REQUIRE(core_gap_fill::enabled(true));
        REQUIRE_FALSE(core_gap_fill::enabled(false));
    }
    for (const char* off : {"0", "false", "OFF", "no"}) {
        scoped_env e("CRISPASR_GAP_FILL", off);
        INFO("CRISPASR_GAP_FILL=" << off);
        REQUIRE_FALSE(core_gap_fill::enabled(true));
    }
    for (const char* on : {"1", "true", "yes"}) {
        scoped_env e("CRISPASR_GAP_FILL", on);
        INFO("CRISPASR_GAP_FILL=" << on);
        // Forcing on overrides a backend that defaults off (Haas-ja).
        REQUIRE(core_gap_fill::enabled(false));
    }
}

TEST_CASE("gap-fill policy: confidence threshold default and override", "[unit][gap-fill-policy]") {
    {
        scoped_env e("CRISPASR_GAP_FILL_MIN_CONF", nullptr);
        REQUIRE(core_gap_fill::min_run_confidence() == core_gap_fill::kDefaultMinRunConfidence);
    }
    {
        scoped_env e("CRISPASR_GAP_FILL_MIN_CONF", "0");
        REQUIRE(core_gap_fill::min_run_confidence() == 0.0f);
    }
    {
        scoped_env e("CRISPASR_GAP_FILL_MIN_CONF", "0.5");
        REQUIRE(core_gap_fill::min_run_confidence() == 0.5f);
    }
    {
        scoped_env e("CRISPASR_GAP_FILL_MIN_CONF", "7");
        REQUIRE(core_gap_fill::min_run_confidence() == 1.0f);
    }
    {
        scoped_env e("CRISPASR_GAP_FILL_MIN_CONF", "garbage");
        REQUIRE(core_gap_fill::min_run_confidence() == core_gap_fill::kDefaultMinRunConfidence);
    }
}

TEST_CASE("gap-fill policy: the gate averages available confidences", "[unit][gap-fill-policy]") {
    // The #605 junk: "隠しのように" at 0.11 / 0.54 / 0.17 / 0.33.
    REQUIRE_FALSE(core_gap_fill::run_is_confident({0.11f, 0.54f, 0.17f, 0.33f}, 0.85f));
    // The confirmed #605 recovery "投げね" at 1.00.
    REQUIRE(core_gap_fill::run_is_confident({1.0f, 1.0f, 0.999f}, 0.85f));
    // Exactly at the threshold passes.
    REQUIRE(core_gap_fill::run_is_confident({0.85f}, 0.85f));
    // Unavailable confidences are skipped; none at all keeps the run.
    REQUIRE(core_gap_fill::run_is_confident({-1.0f, -1.0f}, 0.85f));
    REQUIRE(core_gap_fill::run_is_confident({}, 0.85f));
    REQUIRE_FALSE(core_gap_fill::run_is_confident({-1.0f, 0.2f}, 0.85f));
    // A gate of 0 keeps everything.
    REQUIRE(core_gap_fill::run_is_confident({0.0f}, 0.0f));
}

TEST_CASE("gap-fill policy: norm_word ignores ASCII and CJK punctuation", "[unit][gap-fill-policy]") {
    REQUIRE(core_gap_fill::norm_word("はい。") == "はい");
    REQUIRE(core_gap_fill::norm_word("はい") == "はい");
    REQUIRE(core_gap_fill::norm_word("ねえ!") == "ねえ");
    REQUIRE(core_gap_fill::norm_word("ねえ！") == "ねえ");     // full-width !
    REQUIRE(core_gap_fill::norm_word("「そう」、") == "そう"); // corner brackets, ideographic comma
    REQUIRE(core_gap_fill::norm_word(" And,") == "and");
    REQUIRE(core_gap_fill::norm_word("ーー") == "ーー"); // long-vowel mark is a letter
    REQUIRE(core_gap_fill::norm_word("。、") == "");
}

TEST_CASE("gap-fill policy: seam duplicates are same text within 0.5 s", "[unit][gap-fill-policy]") {
    const std::vector<std::pair<int64_t, std::string>> kept = {{8821, "はい"}, {8989, "あ違った"}};
    REQUIRE(core_gap_fill::is_seam_duplicate(8840, core_gap_fill::norm_word("はい。"), kept));
    REQUIRE_FALSE(core_gap_fill::is_seam_duplicate(8821 + core_gap_fill::kSeamCs, "はい", kept));
    REQUIRE_FALSE(core_gap_fill::is_seam_duplicate(8840, "いいえ", kept));
    REQUIRE_FALSE(core_gap_fill::is_seam_duplicate(8840, "", {{8840, ""}}));
}
