// test-kokoro-misaki-wiring.cpp — #316: does the misaki G2P the PRODUCT calls
// actually apply misaki's rules?
//
// This test exists because the answer was no for a whole release. The
// contextual function-word rules, the capitalisation-stress rule and the
// phrase-final lexicon all shipped in 0.8.24/0.8.25 behind
// `g2p_en::context::context_words`, which nothing anywhere set — so
// `phonemize_misaki_en` read "the" as `ði` in every position and the article
// "a" as the LETTER, `ˈA` with primary stress. Every unit test passed: they
// called `core_g2p_ctxwords::lookup` directly, which is not the code path the
// product uses.
//
// So this one goes through `crispasr::phonemize_misaki_en` — the function
// kokoro.cpp calls — with a tiny lexicon in misaki's own JSON shape pointed at
// by CRISPASR_MISAKI_DICT_PATH. Hermetic: no download, no model, no audio.

#include <catch2/catch_test_macros.hpp>

#include "phonemizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "portable_env.h"

namespace {

// misaki's own entries for these words, verbatim (misaki 0.9.4, us_gold.json).
// `a` really is stored as the letter; the article reading is a CONTEXT rule,
// which is the whole point of this test.
const char* kLexicon = R"({
  "a": "A",
  "the": "ði",
  "to": "tu",
  "I": "ˈI",
  "box": "bˈɑks",
  "apple": "ˈæpᵊl",
  "dramatic": "dɹəmˈæɾɪk",
  "d": "dˈi",
  "f": "ˈɛf",
  "knight": "nˈIt",
  "three": "θɹˈi",
  "four": "fˈɔɹ",
  "eight": "ˈAt"
})";

std::string phonemize(const char* text) {
    std::string out;
    REQUIRE(crispasr::phonemize_misaki_en("en", text, out));
    return out;
}

} // namespace

TEST_CASE("phonemize_misaki_en applies misaki's rules, not just loads its words", "[unit][kokoro][g2p]") {
    // Written next to the test binary, not /tmp: the VPS root partition is
    // tiny and CLAUDE.md forbids it.
    const char* path = "test-kokoro-misaki-wiring.lexicon.json";
    {
        FILE* f = fopen(path, "wb");
        REQUIRE(f != nullptr);
        fwrite(kLexicon, 1, strlen(kLexicon), f);
        fclose(f);
    }
    // Must be set before the FIRST call — the loader runs once per process.
    setenv("CRISPASR_MISAKI_DICT_PATH", path, 1);
    REQUIRE(crispasr::misaki_lexicon_available());

    SECTION("the contextual function words are applied") {
        // The reported "old English" diction. `ði` before a consonant is the
        // bug; `ðə` is misaki.
        CHECK(phonemize("the box") == "ðə bˈɑks");
        CHECK(phonemize("the apple") == "ði ˈæpᵊl");
        // The article, not the letter. `ˈA` here is "EIGH" with primary stress
        // — the "unnecessary emphasis on a" of the #316 follow-up.
        CHECK(phonemize("a box") == "ɐ bˈɑks");
        // Secondary stress on the pronoun, not primary.
        CHECK(phonemize("I box") == "ˌI bˈɑks");
    }

    SECTION("punctuation reaches the model") {
        // Every one of these is in Kokoro's 178-symbol vocabulary and is how it
        // knows to pause.
        CHECK(phonemize("the box, the apple.") == "ðə bˈɑks, ði ˈæpᵊl.");
    }

    SECTION("letters followed by digits keep the digits") {
        // Chess squares, "B52", "Win10": the number speller skipped digits
        // after a letter (to leave "mp3" to other rules) and nothing else
        // read them, so "d4" phonemized to `d` - the square was silent and
        // the letter read as a sound. misaki splits the token: the letter
        // by its name, the digits as a number.
        CHECK(phonemize("d4") == "dˈi fˈɔɹ");
        CHECK(phonemize("knight f3") == "nˈIt ˈɛf θɹˈi");
        // misaki reads a lone "a" before digits as the article, and so do we.
        CHECK(phonemize("a8") == "ɐ ˈAt");
    }

    SECTION("a quoted word is looked up without its quotes") {
        // `"dramatic"` used to miss every tier and fall through to the
        // letter-to-sound rules.
        CHECK(phonemize("\"dramatic\"") == "“dɹəmˈæɾɪk”");
    }

    remove(path);
}
