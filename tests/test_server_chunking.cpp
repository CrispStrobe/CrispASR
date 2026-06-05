// test_server_chunking.cpp — sentence splitter + concat unit tests.
//
// Pins the splitter behaviour the /v1/audio/speech long-form path
// depends on (PLAN §75d / issue #66): boundary detection, abbreviation
// over-split (acceptable), Unicode terminator support, max_chars
// fallback, concat with silence pad.

#include "crispasr_tts_chunking.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

TEST_CASE("splitter empty / whitespace-only input → empty vector", "[unit][chunking]") {
    REQUIRE(crispasr_tts_split_sentences("").empty());
    REQUIRE(crispasr_tts_split_sentences("   ").empty());
    REQUIRE(crispasr_tts_split_sentences("\n\t  \r\n").empty());
}

TEST_CASE("splitter single sentence stays whole (with terminator)", "[unit][chunking]") {
    auto out = crispasr_tts_split_sentences("Hello there.");
    REQUIRE(out.size() == 1);
    REQUIRE(out[0] == "Hello there.");
}

TEST_CASE("splitter single sentence stays whole (no terminator)", "[unit][chunking]") {
    auto out = crispasr_tts_split_sentences("Hello there");
    REQUIRE(out.size() == 1);
    REQUIRE(out[0] == "Hello there");
}

TEST_CASE("splitter handles all three ASCII terminators", "[unit][chunking]") {
    auto out = crispasr_tts_split_sentences("Wait. Really? Yes!");
    REQUIRE(out.size() == 3);
    REQUIRE(out[0] == "Wait.");
    REQUIRE(out[1] == "Really?");
    REQUIRE(out[2] == "Yes!");
}

TEST_CASE("splitter trims whitespace from each chunk", "[unit][chunking]") {
    auto out = crispasr_tts_split_sentences("  First.    Second.   Third.  ");
    REQUIRE(out.size() == 3);
    REQUIRE(out[0] == "First.");
    REQUIRE(out[1] == "Second.");
    REQUIRE(out[2] == "Third.");
}

TEST_CASE("splitter does NOT split decimal numbers", "[unit][chunking]") {
    // The period in "1.5" is followed by a digit, not whitespace, so
    // it must stay intact.
    auto out = crispasr_tts_split_sentences("The value is 1.5 today.");
    REQUIRE(out.size() == 1);
    REQUIRE(out[0] == "The value is 1.5 today.");
}

TEST_CASE("splitter over-splits English abbreviations (documented)", "[unit][chunking]") {
    // "Mr. Smith" gets split — the period is followed by whitespace.
    // Documented limitation; produces an extra ~200 ms pause but
    // doesn't break audio. Real fix is ICU BreakIterator.
    auto out = crispasr_tts_split_sentences("Mr. Smith said hi.");
    REQUIRE(out.size() == 2);
    REQUIRE(out[0] == "Mr.");
    REQUIRE(out[1] == "Smith said hi.");
}

TEST_CASE("splitter handles CJK ideographic full stop U+3002", "[unit][chunking]") {
    // Two Japanese sentences: 「これは最初。」「これは二番目。」
    const std::string ja = "\xe3\x81\x93\xe3\x82\x8c\xe3\x81\xaf\xe6\x9c\x80\xe5\x88\x9d\xe3\x80\x82 "
                           "\xe3\x81\x93\xe3\x82\x8c\xe3\x81\xaf\xe4\xba\x8c\xe7\x95\xaa\xe7\x9b\xae\xe3\x80\x82";
    auto out = crispasr_tts_split_sentences(ja);
    REQUIRE(out.size() == 2);
    // First chunk ends with U+3002.
    REQUIRE(out[0].substr(out[0].size() - 3) == "\xe3\x80\x82");
    REQUIRE(out[1].substr(out[1].size() - 3) == "\xe3\x80\x82");
}

TEST_CASE("splitter handles Devanagari danda U+0964", "[unit][chunking]") {
    // Two Hindi sentences: यह पहला है। यह दूसरा है।
    const std::string hi = "\xe0\xa4\xaf\xe0\xa4\xb9 \xe0\xa4\xaa\xe0\xa4\xb9\xe0\xa4\xb2\xe0\xa4\xbe "
                           "\xe0\xa4\xb9\xe0\xa5\x88\xe0\xa5\xa4 \xe0\xa4\xaf\xe0\xa4\xb9 "
                           "\xe0\xa4\xa6\xe0\xa5\x82\xe0\xa4\xb8\xe0\xa4\xb0\xe0\xa4\xbe "
                           "\xe0\xa4\xb9\xe0\xa5\x88\xe0\xa5\xa4";
    auto out = crispasr_tts_split_sentences(hi);
    REQUIRE(out.size() == 2);
    REQUIRE(out[0].substr(out[0].size() - 3) == "\xe0\xa5\xa4");
    REQUIRE(out[1].substr(out[1].size() - 3) == "\xe0\xa5\xa4");
}

TEST_CASE("splitter max_chars fallback splits run-on input", "[unit][chunking]") {
    // No terminator anywhere. With max_chars=20 and word-boundaries,
    // we should get multiple chunks of <= 20 chars each.
    std::string run_on;
    for (int i = 0; i < 10; i++)
        run_on += "wordA wordB wordC ";
    auto out = crispasr_tts_split_sentences(run_on, /*max_chars=*/20);
    REQUIRE(out.size() >= 2);
    for (const auto& chunk : out) {
        REQUIRE(chunk.size() <= 20);
        REQUIRE(!chunk.empty());
    }
}

TEST_CASE("splitter max_chars splits a too-long single sentence", "[unit][chunking]") {
    std::string long_sent;
    for (int i = 0; i < 20; i++)
        long_sent += "alpha beta ";
    long_sent += "end.";
    auto out = crispasr_tts_split_sentences(long_sent, /*max_chars=*/30);
    REQUIRE(out.size() >= 2);
}

TEST_CASE("splitter handles trailing terminator with no whitespace after", "[unit][chunking]") {
    auto out = crispasr_tts_split_sentences("Just one thing.");
    REQUIRE(out.size() == 1);
    REQUIRE(out[0] == "Just one thing.");
}

TEST_CASE("splitter handles multiple paragraphs", "[unit][chunking]") {
    auto out = crispasr_tts_split_sentences("First sentence. Second sentence.\n\nThird sentence. Fourth.");
    REQUIRE(out.size() == 4);
}

TEST_CASE("splitter pathological no-whitespace blob still splits at max_chars", "[unit][chunking]") {
    // 200 chars of `x` with no space anywhere. Must still produce
    // <=max_chars chunks; falling back to mid-word split is acceptable
    // — we'd rather degrade speech quality than OOM the synth loop.
    std::string blob(200, 'x');
    auto out = crispasr_tts_split_sentences(blob, /*max_chars=*/50);
    REQUIRE(out.size() >= 4);
    for (const auto& chunk : out)
        REQUIRE(chunk.size() <= 50);
}

TEST_CASE("server TTS policy keeps VibeVoice voice cloning single-shot", "[unit][chunking]") {
    auto out = crispasr_tts_plan_chunks_for_backend("First sentence. Second sentence. Third sentence.", "vibevoice");
    REQUIRE(out.size() == 1);
    REQUIRE(out[0] == "First sentence. Second sentence. Third sentence.");
}

TEST_CASE("server TTS policy chunks non-VibeVoice backends", "[unit][chunking]") {
    auto out = crispasr_tts_plan_chunks_for_backend("First sentence. Second sentence.", "qwen3-tts");
    REQUIRE(out.size() == 2);
    REQUIRE(out[0] == "First sentence.");
    REQUIRE(out[1] == "Second sentence.");
}

TEST_CASE("server TTS policy preserves whitespace-only requests", "[unit][chunking]") {
    auto out = crispasr_tts_plan_chunks_for_backend("   ", "qwen3-tts");
    REQUIRE(out.size() == 1);
    REQUIRE(out[0] == "   ");
}

// ──────────────────────────────────────────────────────────────────────────
// crispasr_tts_concat_with_silence
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("concat empty input → empty output", "[unit][chunking]") {
    auto out = crispasr_tts_concat_with_silence({}, 4800);
    REQUIRE(out.empty());
}

TEST_CASE("concat single chunk has no silence pad", "[unit][chunking]") {
    std::vector<std::vector<float>> chunks = {{0.1f, 0.2f, 0.3f}};
    auto out = crispasr_tts_concat_with_silence(chunks, 4800);
    REQUIRE(out.size() == 3);
    REQUIRE(out[0] == 0.1f);
    REQUIRE(out[1] == 0.2f);
    REQUIRE(out[2] == 0.3f);
}

TEST_CASE("concat two chunks inserts silence between but not at boundaries", "[unit][chunking]") {
    std::vector<std::vector<float>> chunks = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto out = crispasr_tts_concat_with_silence(chunks, 5);
    REQUIRE(out.size() == 2 + 5 + 2);
    REQUIRE(out[0] == 1.0f);
    REQUIRE(out[1] == 2.0f);
    for (int i = 0; i < 5; i++)
        REQUIRE(out[2 + i] == 0.0f);
    REQUIRE(out[7] == 3.0f);
    REQUIRE(out[8] == 4.0f);
}

TEST_CASE("concat with silence_samples=0 just concatenates", "[unit][chunking]") {
    std::vector<std::vector<float>> chunks = {{1.0f}, {2.0f}, {3.0f}};
    auto out = crispasr_tts_concat_with_silence(chunks, 0);
    REQUIRE(out.size() == 3);
    REQUIRE(out[0] == 1.0f);
    REQUIRE(out[1] == 2.0f);
    REQUIRE(out[2] == 3.0f);
}

TEST_CASE("concat negative silence_samples treated as zero", "[unit][chunking]") {
    std::vector<std::vector<float>> chunks = {{1.0f}, {2.0f}};
    auto out = crispasr_tts_concat_with_silence(chunks, -100);
    REQUIRE(out.size() == 2);
}
