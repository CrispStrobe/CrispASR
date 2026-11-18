// test-sentence-end-abbrev.cpp — --split-on-punct must not break subtitles at
// German ordinals, initials or abbreviations ("am 24. | November",
// "Dr. | Sieben", "z. | B."), on both display-segment paths: text-only
// segments (no word timings) and word-timed segments. The '.' rule is shared
// with TTS chunking and live translation (examples/cli/crispasr_sentence_end.h).
// Measured on 1,776 UD German GSD sentences: false splits 72 -> 8.

#include "crispasr_output.h"

#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <vector>

static const char* kText = "Am 24. November kam Dr. Sieben z. B. nach Berlin. Er blieb! Das war 1990.";

static std::vector<std::string> texts(const std::vector<crispasr_disp_segment>& d) {
    std::vector<std::string> out;
    for (const auto& s : d)
        out.push_back(s.text);
    return out;
}

TEST_CASE("split-on-punct keeps ordinals and abbreviations (text-only segment)", "[unit][output][sentence-end]") {
    crispasr_segment seg;
    seg.t0 = 0;
    seg.t1 = 1000;
    seg.text = kText;
    auto out = texts(crispasr_make_disp_segments({seg}, 0, true));
    REQUIRE(out.size() == 3);
    REQUIRE(out[0] == "Am 24. November kam Dr. Sieben z. B. nach Berlin.");
    REQUIRE(out[1] == "Er blieb!");
    REQUIRE(out[2] == "Das war 1990.");
}

TEST_CASE("split-on-punct keeps ordinals and abbreviations (word-timed segment)", "[unit][output][sentence-end]") {
    crispasr_segment seg;
    seg.t0 = 0;
    seg.t1 = 1500;
    seg.text = kText;
    std::istringstream in(kText);
    std::string w;
    int64_t t = 0;
    while (in >> w) {
        seg.words.push_back({(seg.words.empty() ? "" : " ") + w, t, t + 100});
        t += 100;
    }
    auto out = crispasr_make_disp_segments({seg}, 0, true);
    REQUIRE(out.size() == 3);
    REQUIRE(out[0].text.find("Berlin.") != std::string::npos);
    REQUIRE(out[0].text.find("November") != std::string::npos);
    REQUIRE(out[1].text.find("blieb!") != std::string::npos);
}
