// test-ngram-loop-fix.cpp — Catch2 unit tests for core_ngram::fix_loops, the
// shared greedy-n-gram-loop collapse used by the higgs-audio-v3-stt and
// moss-transcribe backends (issue #218).
//
// Pure text transform — no model load. The fixtures are the exact degenerate
// outputs observed on the issue #218 sample (t32-145s.wav) plus invariants
// (clean text untouched, empty input safe).

#include <catch2/catch_test_macros.hpp>

#include "core/ngram_loop_fix.h"

#include <string>

using core_ngram::fix_loops;

TEST_CASE("clean transcripts pass through byte-identical", "[ngram-loop]") {
    // The transform must be a no-op on non-degenerate text so it never harms a
    // good slice.
    const std::string clean = "Fast! This is some sort of a test.";
    REQUIRE(fix_loops(clean) == clean);

    const std::string clean2 = "Come on! Don't move much of the fence.";
    REQUIRE(fix_loops(clean2) == clean2);

    // A legitimate short repeat (2 reps of a bigram, 3 of a unigram) is under
    // the collapse threshold and survives.
    const std::string ok_rep = "no no no more games";
    REQUIRE(fix_loops(ok_rep) == ok_rep);
}

TEST_CASE("degenerate unigram loop collapses to a bounded run", "[ngram-loop]") {
    // issue #218 slice 2: "Hey," repeated ~490× until the token cap.
    std::string loop = "just get him moving.";
    for (int i = 0; i < 40; i++)
        loop += " hey,";
    const std::string out = fix_loops(loop);
    // The good prefix survives; the "hey," run is trimmed to at most a few reps.
    REQUIRE(out.find("just get him moving.") == 0);
    // Count surviving "hey," tokens — must be a small bounded number, not 40.
    size_t count = 0;
    for (size_t p = out.find("hey,"); p != std::string::npos; p = out.find("hey,", p + 1))
        count++;
    REQUIRE(count <= 3);
    REQUIRE(out.size() < loop.size());
}

TEST_CASE("long single-word chunk loop collapses after larger n-grams", "[ngram-loop]") {
    std::string loop = "just get him moving";
    for (int i = 0; i < 80; i++)
        loop += " back";
    loop += " It is a joke.";
    const std::string out = fix_loops(loop);
    REQUIRE(out.find("just get him moving") == 0);
    REQUIRE(out.find("It is a joke.") != std::string::npos);
    REQUIRE(out.find("back back back back") == std::string::npos);
    REQUIRE(out.size() < loop.size());
}

TEST_CASE("degenerate multi-word cycle collapses", "[ngram-loop]") {
    // issue #218 slice 5: "run hey hey hey hey hey run" cycles.
    const std::string loop =
        "run hey hey hey hey hey run hey hey hey hey hey run hey hey hey hey hey run Alex you okay?";
    const std::string out = fix_loops(loop);
    REQUIRE(out.size() < loop.size());
    REQUIRE(out.find("Alex you okay?") != std::string::npos);
    // No five-in-a-row "hey" survives the unigram collapse.
    REQUIRE(out.find("hey hey hey hey hey") == std::string::npos);
}

TEST_CASE("degenerate phrase repetition collapses", "[ngram-loop]") {
    // issue #218 follow-ups: multiple AR ASR backends can loop on short
    // phrases such as "come on" after a valid prefix.
    const std::string loop = "move much of the fence. come on come on come on come on come on come on come on come on";
    const std::string out = fix_loops(loop);
    REQUIRE(out.find("move much of the fence.") == 0);
    REQUIRE(out.find("come on come on come on come on") == std::string::npos);
    REQUIRE(out.size() < loop.size());
}

TEST_CASE("edge cases are safe", "[ngram-loop]") {
    REQUIRE(fix_loops("") == "");
    REQUIRE(fix_loops("word") == "word");
    // Whitespace is normalised to single spaces on rejoin.
    REQUIRE(fix_loops("  a   b  ") == "a b");
}
