// test-issue-465-hangul-spaces.cpp - Korean keeps its word spaces through
// crispasr_make_disp_segments (-sp / -ml / SRT splitting), issue #465.
//
// The forced aligner splits Hangul into syllables; the syllable that follows a
// space now carries it (" 오", whisper's own convention) and the display builder
// must keep it, trim it at a line start, count -ml in characters, and not break
// inside a word. Chinese / Japanese (no inter-word spaces) and English are the
// controls. Compiles crispasr_output.cpp directly - no model.
#include "crispasr_output.h"

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

namespace {

crispasr_segment seg_from_words(const std::vector<std::string>& words) {
    crispasr_segment s;
    s.t0 = 0;
    int64_t t = 0;
    for (const auto& w : words) {
        crispasr_word cw;
        cw.text = w;
        cw.t0 = t;
        cw.t1 = t + 20;
        t += 20;
        s.words.push_back(cw);
        s.text += w;
    }
    s.t1 = t;
    return s;
}

std::vector<std::string> lines(const std::vector<crispasr_disp_segment>& d) {
    std::vector<std::string> out;
    for (const auto& x : d)
        out.push_back(x.text);
    return out;
}

// what tokenise_display_words produces for the issue's sentence
const std::vector<std::string> kKo = {"내", "일", " 오", "전", "에", " 회", "의", " 자",
                                      "료", "를", " 보", "내", "주", "세",  "요."};

} // namespace

TEST_CASE("#465 -sp keeps the Hangul word spaces", "[issue-465]") {
    const auto d = crispasr_make_disp_segments({seg_from_words(kKo)}, 0, /*split_on_punct=*/true);
    REQUIRE(d.size() == 1);
    CHECK(d[0].text == "내일 오전에 회의 자료를 보내주세요.");
}

TEST_CASE("#465 -ml counts characters and does not break inside a Hangul word", "[issue-465]") {
    const auto d = lines(crispasr_make_disp_segments({seg_from_words(kKo)}, 10));
    // every line boundary falls on a space in the original, no line opens with one
    REQUIRE(d == std::vector<std::string>{"내일 오전에 회의", "자료를 보내주세요."});
}

TEST_CASE("#465 controls: Chinese stays glued, English keeps its spacing", "[issue-465]") {
    // split_on_punct routes through the word rebuild, the path the fix touches
    const auto zh = crispasr_make_disp_segments({seg_from_words({"你", "好", "世", "界"})}, 0, true);
    REQUIRE(zh.size() == 1);
    CHECK(zh[0].text == "你好世界");
    const auto en = crispasr_make_disp_segments({seg_from_words({"hello", "world", "again"})}, 0, true);
    REQUIRE(en.size() == 1);
    CHECK(en[0].text == "hello world again");
    const auto ws = crispasr_make_disp_segments({seg_from_words({" hello", " world"})}, 0, true); // whisper-style
    REQUIRE(ws.size() == 1);
    CHECK(ws[0].text == "hello world");
}
