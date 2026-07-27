// test-vibevoice-transcript.cpp — unit tests for the VibeVoice-ASR answer parser.
//
// The model answers with a JSON array of utterances (Start / End / Speaker /
// Content). #300: those fields were never read, so speaker labels reached the
// user as literal JSON and `seg.speaker` stayed empty. This pins the parse —
// including the cases a strict JSON reader gets wrong for LLM output: a decode
// truncated by the token cap, and the long key spellings the prompt itself uses.
//
// Header-only, no model, no audio.

#include <catch2/catch_test_macros.hpp>

#include "vibevoice_transcript_parse.h"

using vibevoice_transcript::parse;

TEST_CASE("the recorded jfk answer parses to one utterance", "[unit][vibevoice]") {
    // Verbatim from the 2026-04-24 Q4_K end-to-end run on samples/jfk.wav.
    const std::string raw =
        R"([{"Start":0.0,"End":11.0,"Speaker":0,"Content":"And so, my fellow Americans, ask not what your )"
        R"(country can do for you, ask what you can do for your country."}])";
    auto u = parse(raw);
    REQUIRE(u.size() == 1);
    REQUIRE(u[0].start_s == 0.0);
    REQUIRE(u[0].end_s == 11.0);
    REQUIRE(u[0].speaker == 0);
    REQUIRE(u[0].text.rfind("And so, my fellow Americans", 0) == 0);
    REQUIRE(u[0].text.find('{') == std::string::npos); // no JSON leaks into the text
}

TEST_CASE("multiple speakers become multiple utterances", "[unit][vibevoice]") {
    const std::string raw = R"([{"Start":0.0,"End":1.5,"Speaker":0,"Content":"welcome everyone"},)"
                            R"({"Start":1.8,"End":3.2,"Speaker":1,"Content":"thanks, glad to be here"}])";
    auto u = parse(raw);
    REQUIRE(u.size() == 2);
    REQUIRE(u[0].speaker == 0);
    REQUIRE(u[0].text == "welcome everyone");
    REQUIRE(u[1].speaker == 1);
    REQUIRE(u[1].end_s == 3.2);
}

TEST_CASE("the prompt's long key spellings are accepted", "[unit][vibevoice]") {
    // src/vibevoice.cpp asks for "Start time, End time, Speaker ID, Content".
    const std::string raw = R"([{"Start time":2.0,"End time":4.0,"Speaker ID":3,"Content":"long keys"}])";
    auto u = parse(raw);
    REQUIRE(u.size() == 1);
    REQUIRE(u[0].start_s == 2.0);
    REQUIRE(u[0].end_s == 4.0);
    REQUIRE(u[0].speaker == 3);
}

TEST_CASE("a truncated decode keeps every complete utterance", "[unit][vibevoice]") {
    // What a max-new-tokens cut looks like: two good objects, then a cut mid-string.
    const std::string raw = R"([{"Start":0.0,"End":1.0,"Speaker":0,"Content":"first"},)"
                            R"({"Start":1.0,"End":2.0,"Speaker":1,"Content":"second"},)"
                            R"({"Start":2.0,"End":3.0,"Speaker":0,"Content":"thi)";
    auto u = parse(raw);
    REQUIRE(u.size() == 2); // a strict JSON parse would yield ZERO here
    REQUIRE(u[0].text == "first");
    REQUIRE(u[1].text == "second");
}

TEST_CASE("escapes inside Content are unescaped", "[unit][vibevoice]") {
    const std::string raw = R"([{"Speaker":0,"Content":"he said \"hi\",\nthen left — quickly"}])";
    auto u = parse(raw);
    REQUIRE(u.size() == 1);
    REQUIRE(u[0].text == "he said \"hi\",\nthen left \xe2\x80\x94 quickly"); // U+2014 as UTF-8
}

TEST_CASE("a quoted or prefixed speaker id still yields the number", "[unit][vibevoice]") {
    auto a = parse(R"([{"Speaker":"1","Content":"quoted"}])");
    REQUIRE(a.size() == 1);
    REQUIRE(a[0].speaker == 1);
    auto b = parse(R"([{"Speaker":"SPEAKER_02","Content":"prefixed"}])");
    REQUIRE(b.size() == 1);
    REQUIRE(b[0].speaker == 2);
}

TEST_CASE("missing fields are reported as absent, not as zero", "[unit][vibevoice]") {
    // The caller distinguishes "no speaker" from "speaker 0" — labelling an
    // unlabelled utterance "(Speaker 0)" would invent diarization.
    auto u = parse(R"([{"Content":"no metadata at all"}])");
    REQUIRE(u.size() == 1);
    REQUIRE(u[0].speaker == -1);
    REQUIRE(u[0].start_s < 0.0);
    REQUIRE(u[0].end_s < 0.0);
}

TEST_CASE("plain prose is not a transcript blob", "[unit][vibevoice]") {
    // Empty result is the signal to keep the raw string as one segment.
    REQUIRE(parse("And so, my fellow Americans.").empty());
    REQUIRE(parse("").empty());
    REQUIRE(parse("[]").empty());
    REQUIRE(parse(R"([{"Start":0.0,"End":1.0,"Speaker":0}])").empty()); // no Content
}
