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

#include <initializer_list>

#include "core/vibevoice_transcript.h"

using core_vibevoice::parse;

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
    REQUIRE(u.size() == 3); // a strict JSON parse would yield ZERO here
    REQUIRE(u[0].text == "first");
    REQUIRE(u[1].text == "second");
    REQUIRE_FALSE(u[1].truncated);
    // #610: the cut fell inside a Content, so its decoded part is kept.
    REQUIRE(u[2].text == "thi");
    REQUIRE(u[2].truncated);
    REQUIRE(u[2].speaker == 0);
}

// #610: the shapes reported from Japanese drama audio, where a repetition loop
// ran into the 512-token limit inside a Content.
static std::string repeat(const std::string& s, int n) {
    std::string out;
    for (int i = 0; i < n; i++)
        out += s;
    return out;
}

TEST_CASE("#610: a first object cut off in a loop keeps its text, loop collapsed", "[unit][vibevoice]") {
    const std::string raw = R"([{"Start":0,"End":10.43,"Speaker":0,"Content":")" + repeat("あ、", 270);
    auto u = parse(raw);
    REQUIRE(u.size() == 1);
    REQUIRE(u[0].truncated);
    REQUIRE(u[0].start_s == 0.0);
    REQUIRE(u[0].end_s == 10.43);
    REQUIRE(u[0].raw_text == repeat("あ、", 270));
    REQUIRE(u[0].text.size() < 30); // collapsed, not 540 characters
    REQUIRE(u[0].text.rfind("あ、", 0) == 0);
}

TEST_CASE("#610: lines decoded before the loop are not lost", "[unit][vibevoice]") {
    const std::string said = "何してるの。早く行こうよ。わかった。";
    const std::string raw = R"([{"Start":0,"End":3.1,"Speaker":0,"Content":"[Human Sounds]"},)"
                            R"({"Start":3.1,"End":22.6,"Speaker":1,"Content":")" +
                            said + repeat("うわ、", 148);
    auto u = parse(raw);
    REQUIRE(u.size() == 2);
    REQUIRE(u[1].truncated);
    REQUIRE(u[1].text.rfind(said, 0) == 0); // the three lines survive
    REQUIRE(u[1].text.size() < said.size() + 40);
}

TEST_CASE("#610: a Latin annotation loop collapses on the word path", "[unit][vibevoice]") {
    const std::string raw = R"([{"Start":0,"End":29.55,"Speaker":0,"Content":")" + repeat("[Gasp] ", 120);
    auto u = parse(raw);
    REQUIRE(u.size() == 1);
    REQUIRE(u[0].text == "[Gasp] [Gasp] [Gasp]");
}

TEST_CASE("#610: a loop inside a complete Content collapses too", "[unit][vibevoice]") {
    const std::string raw =
        R"([{"Start":0,"End":10.39,"Speaker":0,"Content":")" + repeat("何の話だよ。", 40) + R"("}])";
    auto u = parse(raw);
    REQUIRE(u.size() == 1);
    REQUIRE_FALSE(u[0].truncated);
    REQUIRE(u[0].text == repeat("何の話だよ。", 2));
}

TEST_CASE("#610: clean Content passes through byte for byte", "[unit][vibevoice]") {
    const std::string c = "Two  spaces,\ta tab and a\nnewline stay as they are.";
    auto u = parse(R"([{"Start":0,"End":1,"Speaker":0,"Content":")" + c + R"("}])");
    REQUIRE(u.size() == 1);
    REQUIRE(u[0].text == "Two  spaces,\ta tab and a\nnewline stay as they are.");
    REQUIRE(u[0].text == u[0].raw_text);
}

TEST_CASE("#610: a cut before any Content is still a transcript, never raw text", "[unit][vibevoice]") {
    using core_vibevoice::looks_like_transcript;
    REQUIRE(parse(R"([{"Start":0,"End":1.5,"Spea)").empty());
    REQUIRE(looks_like_transcript(R"([{"Start":0,"End":1.5,"Spea)"));
    REQUIRE(looks_like_transcript(" [\n {\"Start\":0"));
    REQUIRE(looks_like_transcript("["));
    REQUIRE_FALSE(looks_like_transcript("And so, my fellow Americans."));
    REQUIRE_FALSE(looks_like_transcript("[Music] and then she said"));
    REQUIRE_FALSE(looks_like_transcript(""));
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

// ── assign_tokens: split the per-token confidence list the same way as the text
//
// The session ABI hands callers a word/confidence list per segment. Once the
// answer is split into utterances that list has to be split too, or segment 2's
// "words" are segment 0's tokens plus every `[`, `{` and `"Speaker"` of the JSON.

using core_vibevoice::assign_tokens;
using core_vibevoice::Utterance;

static std::vector<std::string> toks(std::initializer_list<const char*> l) {
    return std::vector<std::string>(l.begin(), l.end());
}

TEST_CASE("tokens map to the utterance whose Content they spell", "[unit][vibevoice]") {
    const std::string raw = R"([{"Speaker":0,"Content":"hello there"},{"Speaker":1,"Content":"bye now"}])";
    auto u = parse(raw);
    REQUIRE(u.size() == 2);
    // Decode order, scaffolding included — exactly what the runtime emits.
    auto t = toks({"[{\"", "Speaker", "\":0,\"", "Content", "\":\"", "hello", " there", "\"},{\"", "Speaker", "\":1,\"",
                   "Content", "\":\"", "bye", " now", "\"}]"});
    auto a = assign_tokens(u, t);
    REQUIRE(a.size() == 2);
    REQUIRE(a[0] == std::vector<int>{5, 6});   // "hello", " there"
    REQUIRE(a[1] == std::vector<int>{12, 13}); // "bye", " now"
}

TEST_CASE("a repeated line still maps to distinct token spans", "[unit][vibevoice]") {
    // samples/multispeaker.wav really does repeat the same sentence per speaker,
    // so the second search must not re-find the first occurrence.
    std::vector<Utterance> u(2);
    u[0].text = "same line";
    u[1].text = "same line";
    auto t = toks({"same", " line", "|", "same", " line"});
    auto a = assign_tokens(u, t);
    REQUIRE(a[0] == std::vector<int>{0, 1});
    REQUIRE(a[1] == std::vector<int>{3, 4});
}

TEST_CASE("a Content that cannot be located yields no tokens, not wrong ones", "[unit][vibevoice]") {
    std::vector<Utterance> u(1);
    u[0].text = "not present anywhere";
    auto a = assign_tokens(u, toks({"something", " else"}));
    REQUIRE(a.size() == 1);
    REQUIRE(a[0].empty());
}

TEST_CASE("a token straddling a boundary is claimed by both spans it covers", "[unit][vibevoice]") {
    // Half-open overlap: a merged token is real (BPE merges across the JSON
    // quote), and dropping it would lose that confidence value entirely.
    std::vector<Utterance> u(2);
    u[0].text = "ab";
    u[1].text = "cd";
    auto a = assign_tokens(u, toks({"a", "bc", "d"}));
    REQUIRE(a[0] == std::vector<int>{0, 1});
    REQUIRE(a[1] == std::vector<int>{1, 2});
}

// ── Non-speech markers (#369) ────────────────────────────────────────────────
// "[Silence]" is CONTENT the model emits — it appears nowhere in this codebase.
// It reached users' transcripts as literal text over plainly non-silent audio,
// and because there WAS text the CLI's "no text produced for N s of non-silent
// audio" warning could not fire, so it was invisible as well as wrong.
TEST_CASE("vibevoice transcript: the model's own [Silence] is not transcript text", "[unit][vibevoice]") {
    REQUIRE(core_vibevoice::is_non_speech_marker("[Silence]"));
    REQUIRE(core_vibevoice::is_non_speech_marker("  [silence]  "));
    REQUIRE(core_vibevoice::is_non_speech_marker("[SILENCE]"));
    REQUIRE(core_vibevoice::is_non_speech_marker("[Silence.]"));
    REQUIRE(core_vibevoice::is_non_speech_marker("[BLANK_AUDIO]"));
    REQUIRE(core_vibevoice::is_non_speech_marker("[Inaudible]"));
}

// Deliberately narrow: only a WHOLE bracketed token from the observed set. Over-
// reaching here deletes real content, which is a worse failure than the one
// being fixed.
TEST_CASE("vibevoice transcript: real speech is never mistaken for a marker", "[unit][vibevoice]") {
    REQUIRE_FALSE(core_vibevoice::is_non_speech_marker("내일 오전에 회의 자료를 보내주세요."));
    REQUIRE_FALSE(core_vibevoice::is_non_speech_marker("[Music]"));    // may be wanted
    REQUIRE_FALSE(core_vibevoice::is_non_speech_marker("[Laughter]")); // may be wanted
    REQUIRE_FALSE(core_vibevoice::is_non_speech_marker("The silence was total."));
    REQUIRE_FALSE(core_vibevoice::is_non_speech_marker("[Silence] and then she spoke."));
    REQUIRE_FALSE(core_vibevoice::is_non_speech_marker(""));
    REQUIRE_FALSE(core_vibevoice::is_non_speech_marker("[]"));
}

// The blob still has to PARSE — a response carrying only markers must be
// distinguishable from one that was not a transcript at all, or the caller
// falls back to handing over the raw JSON, which is exactly how "[Silence]"
// reached the transcript.
TEST_CASE("vibevoice transcript: a markers-only answer still parses", "[unit][vibevoice]") {
    const auto utts =
        core_vibevoice::parse(R"([{"Start time":0.0,"End time":6.01,"Speaker ID":0,"Content":"[Silence]"}])");
    REQUIRE(utts.size() == 1);
    REQUIRE(core_vibevoice::is_non_speech_marker(utts[0].text));
    REQUIRE_FALSE(core_vibevoice::parse("the model answered in prose").size() > 0);
}

// A marker between two real utterances must not take them with it.
TEST_CASE("vibevoice transcript: a marker between utterances drops only itself", "[unit][vibevoice]") {
    const auto utts = core_vibevoice::parse(R"([{"Start":0.0,"End":1.0,"Speaker":0,"Content":"hello"},)"
                                            R"({"Start":1.0,"End":2.0,"Speaker":0,"Content":"[Silence]"},)"
                                            R"({"Start":2.0,"End":3.0,"Speaker":1,"Content":"world"}])");
    REQUIRE(utts.size() == 3);
    int kept = 0;
    for (const auto& u : utts)
        if (!core_vibevoice::is_non_speech_marker(u.text))
            kept++;
    REQUIRE(kept == 2);
}
