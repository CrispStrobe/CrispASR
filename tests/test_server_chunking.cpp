// test_server_chunking.cpp — sentence splitter + concat unit tests.
//
// Pins the splitter behaviour the /v1/audio/speech long-form path
// depends on (PLAN §75d / issue #66): boundary detection, abbreviation
// and ordinal handling (shared with live translation via
// crispasr_sentence_end.h), Unicode terminator support, max_chars
// fallback, concat with silence pad.

#include "crispasr_tts_chunking.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
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

TEST_CASE("splitter keeps abbreviations, initials and ordinals inside the sentence", "[unit][chunking]") {
    // Formerly a documented over-split ("Mr." | "Smith said hi."). The '.'
    // rule is now shared with live translation (lt_ends_sentence).
    auto en = crispasr_tts_split_sentences("Mr. Smith said hi.");
    REQUIRE(en.size() == 1);
    REQUIRE(en[0] == "Mr. Smith said hi.");
    auto de = crispasr_tts_split_sentences("Am 3. Oktober kam Dr. Müller z. B. nach Berlin. Er blieb! Was nun?");
    REQUIRE(de.size() == 3);
    REQUIRE(de[0] == "Am 3. Oktober kam Dr. Müller z. B. nach Berlin.");
    REQUIRE(de[1] == "Er blieb!");
    REQUIRE(de[2] == "Was nun?");
}

TEST_CASE("splitter still ends sentences at years, ellipses and closing brackets", "[unit][chunking]") {
    auto out = crispasr_tts_split_sentences("Das war 1990. Das hatte ich gehofft... Die Akte (Nr. 216/94). Ende.");
    REQUIRE(out.size() == 4);
    REQUIRE(out[0] == "Das war 1990.");
    REQUIRE(out[1] == "Das hatte ich gehofft...");
    REQUIRE(out[2] == "Die Akte (Nr. 216/94).");
    REQUIRE(out[3] == "Ende.");
}

TEST_CASE("CRISPASR_SENTENCE_ABBREV=0 restores the bare '. ' rule", "[unit][chunking]") {
#if defined(_WIN32)
    _putenv_s("CRISPASR_SENTENCE_ABBREV", "0");
#else
    setenv("CRISPASR_SENTENCE_ABBREV", "0", 1);
#endif
    auto out = crispasr_tts_split_sentences("Mr. Smith said hi.");
#if defined(_WIN32)
    _putenv_s("CRISPASR_SENTENCE_ABBREV", "");
#else
    unsetenv("CRISPASR_SENTENCE_ABBREV");
#endif
    REQUIRE(out.size() == 2);
    REQUIRE(out[0] == "Mr.");
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

// GH #171 regression: the production TTS backends register as
// "vibevoice-tts" / "vibevoice-1.5b" / "vibevoice-tts-base", NOT the bare
// "vibevoice" (that's the ASR backend). The guard must match every variant
// by prefix or the server sentence-splits each request (the CLI never does),
// degrading voice-clone continuity. These cases FAIL on a bare-string guard.
TEST_CASE("server TTS policy keeps ALL vibevoice* TTS variants single-shot", "[unit][chunking]") {
    const char* text = "First sentence. Second sentence. Third sentence.";
    for (const char* name : {"vibevoice-tts", "vibevoice-1.5b", "vibevoice-tts-1.5b", "vibevoice-tts-base"}) {
        CAPTURE(name);
        auto out = crispasr_tts_plan_chunks_for_backend(text, name);
        REQUIRE(out.size() == 1);
        REQUIRE(out[0] == text);
    }
}

TEST_CASE("server TTS policy keeps qwen3-tts variants single-shot", "[unit][chunking]") {
    const char* text = "First sentence. Second sentence. Third sentence.";
    for (const char* name : {"qwen3-tts", "qwen3-tts-1.7b-base", "qwen3-tts-customvoice", "qwen3-tts-voicedesign"}) {
        CAPTURE(name);
        auto out = crispasr_tts_plan_chunks_for_backend(text, name);
        REQUIRE(out.size() == 1);
        REQUIRE(out[0] == text);
    }
}

// GH #197 regression: TADA's AR talker generates multi-sentence utterances in a
// single pass like HumeAI's reference tada.py. Splitting at punctuation made each
// isolated sentence's duration head over-predict the trailing pause (e.g. "Hi."
// alone → ~9 s of silence+hum) and inserted extra inter-chunk silence — both
// diverge from the reference. The backend registers as bare "tada".
TEST_CASE("server TTS policy keeps tada single-shot", "[unit][chunking]") {
    const char* text = "Hi. My name is Bob. It's really nice to meet you!";
    for (const char* name : {"tada", "tada-1b", "tada-tts-1b", "tada-3b-ml"}) {
        CAPTURE(name);
        auto out = crispasr_tts_plan_chunks_for_backend(text, name);
        REQUIRE(out.size() == 1);
        REQUIRE(out[0] == text);
    }
}

// PLAN #200: dots.tts is a continuous-latent AR model that generates
// multi-sentence utterances in one pass with its own EOS, like the reference
// dots.tts generate(text). Splitting re-runs the per-sentence trailing-pause
// over-prediction AND re-emits the spoken AI-disclaimer that
// dots_tts_synthesize prepends on each call (one per chunk) when cloning. The
// backend registers as "dots-tts".
TEST_CASE("server TTS policy keeps dots-tts single-shot", "[unit][chunking]") {
    const char* text = "First sentence. Second sentence. Third sentence.";
    auto out = crispasr_tts_plan_chunks_for_backend(text, "dots-tts");
    REQUIRE(out.size() == 1);
    REQUIRE(out[0] == text);
}

TEST_CASE("server TTS policy preserves Dia dialogue context and one generation limit", "[unit][chunking]") {
    const std::string text = "[S1] Hello there, how are you doing today? I hope you are having a wonderful time. "
                             "[S2] Thank you! The weather outside is lovely and bright.";
    for (const char* name : {"dia", "dia-tts", "dia-1.6b", "dia_tts"}) {
        CAPTURE(name);
        // A small generic chunk bound must not fragment the model's dialogue.
        const auto chunks = crispasr_tts_plan_chunks_for_backend(text, name, 16);
        REQUIRE(chunks.size() == 1);
        REQUIRE(chunks.front() == text);
    }
}

TEST_CASE("server TTS policy chunks sentence-safe backends", "[unit][chunking]") {
    auto out = crispasr_tts_plan_chunks_for_backend("First sentence. Second sentence.", "kokoro");
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

// #594: one bt2-tts generation per sentence made the model read its style
// instruction aloud in place of short fragments ("Good God!", "Chapter 14").
// The policy must hand it paragraph-sized chunks of whole sentences.
TEST_CASE("server TTS policy packs sentences into paragraph chunks for bt2-tts", "[unit][chunking]") {
    const std::string text = "\"My dear Victor,\" cried he, \"what, for God's sake, is the matter? Do not laugh in "
                             "that manner. How ill you are! What is the cause of all this?\"";
    auto out = crispasr_tts_plan_chunks_for_backend(text, "bt2-tts");
    REQUIRE(out.size() == 1);
    REQUIRE(out[0].find("Do not laugh in that manner.") != std::string::npos);

    std::string longer;
    for (int i = 0; i < 40; i++)
        longer += "Upon hearing this he appeared satisfied and consented. Good God! ";
    auto packed = crispasr_tts_plan_chunks_for_backend(longer, "bt2-tts", 600);
    REQUIRE(packed.size() > 1);
    for (const auto& c : packed) {
        REQUIRE(c.size() <= 600);
        REQUIRE(c.size() > 200); // never a lone short sentence, including the tail
        REQUIRE((c.back() == '.' || c.back() == '!'));
    }
}

TEST_CASE("pack_sentences joins greedily and keeps an oversize sentence alone", "[unit][chunking]") {
    auto out = crispasr_tts_pack_sentences({"Aa.", "Bb.", "Cc.", std::string(20, 'x'), "Dd."}, 8);
    REQUIRE(out == std::vector<std::string>{"Aa. Bb.", "Cc.", std::string(20, 'x'), "Dd."});
    // A short tail borrows trailing sentences from the previous chunk.
    auto bal = crispasr_tts_pack_sentences({"Aaaa.", "Bbbb.", "Cccc.", "Dddd.", "Ee."}, 23);
    REQUIRE(bal == std::vector<std::string>{"Aaaa. Bbbb. Cccc.", "Dddd. Ee."});
    REQUIRE(crispasr_tts_pack_sentences({}, 8).empty());
}

// #609 (chunk 000522): a sentence longer than max_chars used to be cut at the
// last whitespace before the cap, i.e. mid-phrase. The bt2-tts server then
// synthesised "... the base of the mountain, where the priest and" and
// "his mistress were overwhelmed ..." as two separate generations, and the
// model dropped "his" at the start of the second one. Over-long sentences now
// break at the strongest clause boundary (; : dash, then comma) in a window
// that keeps the pieces balanced, and fall back to whitespace only when the
// sentence has no punctuation to break at. CRISPASR_TTS_CLAUSE_SPLIT=0
// restores the old whitespace split.
namespace {

// Frankenstein (1818), vol. III ch. 1: the only sentence in the book longer
// than 600 bytes, as the #609 reporter's corpus spells it (ASCII quotes).
const char* k_long_sentence =
    "\"I have seen,\" he said, \"the most beautiful scenes of my own country; I have visited the lakes of Lucerne "
    "and Uri, where the snowy mountains descend almost perpendicularly to the water, casting black and "
    "impenetrable shades, which would cause a gloomy and mournful appearance were it not for the most verdant "
    "islands that relieve the eye by their gay appearance; I have seen this lake agitated by a tempest, when the "
    "wind tore up whirlwinds of water and gave you an idea of what the water-spout must be on the great ocean; "
    "and the waves dash with fury the base of the mountain, where the priest and his mistress were overwhelmed by "
    "an avalanche and where their dying voices are still said to be heard amid the pauses of the nightly wind; I "
    "have seen the mountains of La Valais, and the Pays de Vaud; but this country, Victor, pleases me more than all "
    "those wonders.";

void set_clause_split_env(const char* v) {
#if defined(_WIN32)
    _putenv_s("CRISPASR_TTS_CLAUSE_SPLIT", v ? v : "");
#else
    if (v)
        setenv("CRISPASR_TTS_CLAUSE_SPLIT", v, 1);
    else
        unsetenv("CRISPASR_TTS_CLAUSE_SPLIT");
#endif
}

bool ends_with(const std::string& s, const std::string& tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// Last character of a piece, ignoring closing quotes.
char last_mark(const std::string& s) {
    std::size_t i = s.size();
    while (i > 0 && (s[i - 1] == '"' || s[i - 1] == '\''))
        i--;
    return i > 0 ? s[i - 1] : '\0';
}

bool valid_utf8(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        std::size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
        if (len == 0 || i + len > s.size())
            return false;
        for (std::size_t k = 1; k < len; k++)
            if (((unsigned char)s[i + k] & 0xC0) != 0x80)
                return false;
        i += len;
    }
    return true;
}

} // namespace

TEST_CASE("#609: an over-long sentence breaks at clause punctuation, not mid-phrase", "[unit][chunking]") {
    set_clause_split_env(nullptr);
    const std::string s = k_long_sentence;
    REQUIRE(s.size() > 600);
    const auto pieces = crispasr_tts_split_long_sentence(s, 600);
    REQUIRE(pieces.size() == 2);
    std::string joined;
    for (const auto& p : pieces) {
        REQUIRE(!p.empty());
        REQUIRE(p.size() <= 600);
        REQUIRE(p.size() >= 150); // never a short fragment behind the instruction (#594)
        joined += (joined.empty() ? "" : " ") + p;
    }
    REQUIRE(joined == s); // nothing lost or reordered
    // Every piece but the last ends at a semicolon, the strongest break here.
    REQUIRE(last_mark(pieces[0]) == ';');
    REQUIRE(!ends_with(pieces[0], "where the priest and"));
    REQUIRE(pieces[1].rfind("his mistress", 0) != 0);
}

TEST_CASE("#609 negative control: CRISPASR_TTS_CLAUSE_SPLIT=0 reproduces the mid-phrase cut", "[unit][chunking]") {
    // Proves the test above can fail: the old whitespace split is exactly the
    // boundary the reporter heard as a dropped word.
    set_clause_split_env("0");
    const auto pieces = crispasr_tts_split_long_sentence(k_long_sentence, 600);
    set_clause_split_env(nullptr);
    REQUIRE(pieces.size() == 2);
    REQUIRE(ends_with(pieces[0], "where the priest and"));
    REQUIRE(pieces[1].rfind("his mistress were overwhelmed", 0) == 0);
}

TEST_CASE("#609: the bt2-tts chunk plan of the 000522 paragraph has no mid-phrase boundary", "[unit][chunking]") {
    set_clause_split_env(nullptr);
    const std::string para = std::string("He felt as if he had been transported to Fairy-land and enjoyed a happiness "
                                         "seldom tasted by man. ") +
                             k_long_sentence +
                             " The mountains of Switzerland are more majestic and strange, but there is a charm in "
                             "the banks of this divine river that I never before saw equalled.";
    const auto chunks = crispasr_tts_plan_chunks_for_backend(para, "bt2-tts", 600);
    REQUIRE(chunks.size() >= 2);
    std::string joined;
    for (const auto& c : chunks) {
        REQUIRE(c.size() <= 600);
        const char m = last_mark(c);
        INFO("chunk: " << c);
        REQUIRE((m == '.' || m == '!' || m == '?' || m == ';' || m == ':' || m == ','));
        joined += (joined.empty() ? "" : " ") + c;
    }
    REQUIRE(joined == para);
}

TEST_CASE("#609: clause breaks outrank commas, commas outrank bare spaces", "[unit][chunking]") {
    set_clause_split_env(nullptr);
    // 4 x 30-byte clauses; the semicolon is off-centre, the comma central.
    const std::string a = "alpha beta gamma delta epsilon"; // 30 bytes
    const std::string s1 = a + "; " + a + ", " + a + " " + a + ".";
    auto p1 = crispasr_tts_split_long_sentence(s1, 100);
    REQUIRE(p1.size() == 2);
    REQUIRE(last_mark(p1[0]) == ';');
    // No clause mark: the comma nearest the middle wins over plain spaces.
    const std::string s2 = a + " " + a + ", " + a + " " + a + ".";
    auto p2 = crispasr_tts_split_long_sentence(s2, 100);
    REQUIRE(p2.size() == 2);
    REQUIRE(last_mark(p2[0]) == ',');
    // Em dash (U+2014) counts as a clause break, with or without spaces.
    const std::string s3 = a + " " + a + "\xe2\x80\x94" + a + " " + a + ".";
    auto p3 = crispasr_tts_split_long_sentence(s3, 100);
    REQUIRE(p3.size() == 2);
    REQUIRE(ends_with(p3[0], "\xe2\x80\x94"));
}

TEST_CASE("#609: a clause mark before a conjunction ranks with the commas", "[unit][chunking]") {
    set_clause_split_env(nullptr);
    const std::string a = "alpha beta gamma delta epsilon"; // 30 bytes
    // "; and" sits nearer the balanced middle than "; we", but only the
    // second starts a fresh clause.
    const std::string s = a + " " + a + "; we " + a + "; and " + a + " " + a + ".";
    auto p = crispasr_tts_split_long_sentence(s, 120);
    REQUIRE(p.size() == 2);
    REQUIRE(p[1].rfind("we ", 0) == 0);
    // The real sentence: the break lands before "I have seen this lake",
    // not before "and the waves dash".
    auto q = crispasr_tts_split_long_sentence(k_long_sentence, 600);
    REQUIRE(q.size() == 2);
    REQUIRE(q[1].rfind("I have seen this lake", 0) == 0);
}

TEST_CASE("#609: a period the sentence splitter kept (M., Mr.) is the last resort", "[unit][chunking]") {
    // The last space before the 100-byte cap follows "M.", so the old greedy
    // cut synthesised "... M." and "Krempe ..." as two generations.
    const std::string s = std::string(40, 'x') + " " + std::string(55, 'y') + " M. Krempe and the rest of it.";
    set_clause_split_env("0");
    auto old_pieces = crispasr_tts_split_long_sentence(s, 100);
    set_clause_split_env(nullptr);
    REQUIRE(ends_with(old_pieces[0], " M.")); // negative control: the old rule does split there
    auto p = crispasr_tts_split_long_sentence(s, 100);
    REQUIRE(p.size() == 2);
    REQUIRE(!ends_with(p[0], "M."));
    REQUIRE(p[1].find("M. Krempe") != std::string::npos);
}

TEST_CASE("#609: a sentence just over the cap does not leave a tiny tail", "[unit][chunking]") {
    set_clause_split_env(nullptr);
    std::string s;
    while (s.size() < 590)
        s += "word, ";
    s += "end of it all.";
    REQUIRE(s.size() > 600);
    set_clause_split_env("0");
    auto old_pieces = crispasr_tts_split_long_sentence(s, 600);
    set_clause_split_env(nullptr);
    REQUIRE(old_pieces.back().size() < 150); // negative control: the old cut leaves a fragment
    auto p = crispasr_tts_split_long_sentence(s, 600);
    REQUIRE(p.size() == 2);
    for (const auto& piece : p) {
        REQUIRE(piece.size() <= 600);
        REQUIRE(piece.size() >= 150);
    }
}

TEST_CASE("#609: the hard cut of a blob without spaces never splits a UTF-8 character", "[unit][chunking]") {
    set_clause_split_env(nullptr);
    std::string blob;
    for (int i = 0; i < 100; i++)
        blob += "\xc3\xa9";                              // é, 2 bytes: 200 bytes, no whitespace
    auto p = crispasr_tts_split_long_sentence(blob, 51); // odd cap: a byte cut would land mid-character
    REQUIRE(p.size() >= 4);
    std::string joined;
    for (const auto& piece : p) {
        REQUIRE(piece.size() <= 51);
        REQUIRE(valid_utf8(piece));
        joined += piece;
    }
    REQUIRE(joined == blob);
}

TEST_CASE("#609: CJK sentences without spaces break after fullwidth punctuation", "[unit][chunking]") {
    set_clause_split_env(nullptr);
    // 30 x "\u5b57" (3 bytes each) = 90 bytes per clause; no spaces anywhere.
    std::string clause;
    for (int i = 0; i < 30; i++)
        clause += "\xe5\xad\x97";
    const std::string comma = "\xef\xbc\x8c";     // U+FF0C FULLWIDTH COMMA
    const std::string semicolon = "\xef\xbc\x9b"; // U+FF1B FULLWIDTH SEMICOLON
    const std::string s = clause + comma + clause + semicolon + clause + comma + clause + "\xe3\x80\x82";
    auto p = crispasr_tts_split_long_sentence(s, 240);
    REQUIRE(p.size() == 2);
    REQUIRE(ends_with(p[0], semicolon)); // the clause mark beats the commas
    std::string joined;
    for (const auto& piece : p) {
        REQUIRE(piece.size() <= 240);
        REQUIRE(valid_utf8(piece));
        joined += piece;
    }
    REQUIRE(joined == s);
}

TEST_CASE("#609: short sentences pass through the long-sentence splitter unchanged", "[unit][chunking]") {
    set_clause_split_env(nullptr);
    REQUIRE(crispasr_tts_split_long_sentence("Hello there.", 600) == std::vector<std::string>{"Hello there."});
    REQUIRE(crispasr_tts_split_long_sentence("   ", 600).empty());
}
