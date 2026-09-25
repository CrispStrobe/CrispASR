// tests/test-g2p-de.cpp — unit tests for German G2P.

#include <catch2/catch_test_macros.hpp>
#include "core/g2p_de.h"
#include <string>

// ── German LTS: digraphs ─────────────────────────────────────────────

TEST_CASE("German LTS: sch/ch/tsch digraphs", "[g2p_de][lts]") {
    SECTION("sch → ʃ") {
        std::string ipa = g2p_de::lts_word_to_ipa("schule");
        CHECK(ipa.find("\xCA\x83") != std::string::npos);
    }
    SECTION("tsch → tʃ") {
        std::string ipa = g2p_de::lts_word_to_ipa("deutsch");
        CHECK(ipa.find("t\xCA\x83") != std::string::npos);
    }
    SECTION("ich-Laut: ch after front vowel → ç") {
        std::string ipa = g2p_de::lts_word_to_ipa("ich");
        CHECK(ipa.find("\xC3\xA7") != std::string::npos);
    }
    SECTION("ach-Laut: ch after back vowel → x") {
        std::string ipa = g2p_de::lts_word_to_ipa("dach");
        CHECK(ipa.find("x") != std::string::npos);
    }
}

// ── Vowel digraphs ──────────────────────────────────────────────────

TEST_CASE("German LTS: vowel digraphs", "[g2p_de][lts]") {
    SECTION("ei → aɪ̯") {
        std::string ipa = g2p_de::lts_word_to_ipa("mein");
        CHECK(ipa.find("a\xC9\xAA") != std::string::npos);
    }
    SECTION("eu → ɔʏ̯") {
        std::string ipa = g2p_de::lts_word_to_ipa("heute");
        CHECK(ipa.find("\xC9\x94\xCA\x8F") != std::string::npos);
    }
    SECTION("au → aʊ̯") {
        std::string ipa = g2p_de::lts_word_to_ipa("haus");
        CHECK(ipa.find("a\xCA\x8A") != std::string::npos);
    }
    SECTION("ie → iː") {
        std::string ipa = g2p_de::lts_word_to_ipa("liebe");
        CHECK(ipa.find("i\xCB\x90") != std::string::npos);
    }
}

// ── Consonant specifics ─────────────────────────────────────────────

TEST_CASE("German LTS: consonant specifics", "[g2p_de][lts]") {
    SECTION("z → ts") {
        std::string ipa = g2p_de::lts_word_to_ipa("zeit");
        CHECK(ipa.find("ts") != std::string::npos);
    }
    SECTION("w → v") {
        std::string ipa = g2p_de::lts_word_to_ipa("welt");
        CHECK(ipa[0] == 'v');
    }
    SECTION("v → f") {
        std::string ipa = g2p_de::lts_word_to_ipa("vater");
        CHECK(ipa[0] == 'f');
    }
    SECTION("initial sp → ʃp") {
        std::string ipa = g2p_de::lts_word_to_ipa("sprechen");
        CHECK(ipa.find("\xCA\x83"
                       "p") != std::string::npos);
    }
    SECTION("initial st → ʃt") {
        std::string ipa = g2p_de::lts_word_to_ipa("stunde");
        CHECK(ipa.find("\xCA\x83"
                       "t") != std::string::npos);
    }
    SECTION("ng → ŋ") {
        std::string ipa = g2p_de::lts_word_to_ipa("lang");
        CHECK(ipa.find("\xC5\x8B") != std::string::npos);
    }
    SECTION("r → r (espeak-ng DE uses plain r)") {
        std::string ipa = g2p_de::lts_word_to_ipa("rot");
        CHECK(ipa.find("r") != std::string::npos);
    }
}

// ── Final schwa and -er ─────────────────────────────────────────────

TEST_CASE("German LTS: schwa and -er", "[g2p_de][lts]") {
    SECTION("final -e → ə") {
        std::string ipa = g2p_de::lts_word_to_ipa("habe");
        CHECK(ipa.find("\xC9\x99") != std::string::npos);
    }
    SECTION("final -er → ɜ (espeak-ng DE)") {
        std::string ipa = g2p_de::lts_word_to_ipa("besser");
        CHECK(ipa.find("\xC9\x9C") != std::string::npos); // ɜ
    }
}

// ── S voicing ───────────────────────────────────────────────────────

TEST_CASE("German LTS: s voicing", "[g2p_de][lts]") {
    SECTION("s before vowel → z") {
        std::string ipa = g2p_de::lts_word_to_ipa("sonne");
        CHECK(ipa[0] == 'z');
    }
    SECTION("ss → s") {
        std::string ipa = g2p_de::lts_word_to_ipa("wasser");
        CHECK(ipa.find("s") != std::string::npos);
    }
}

// ══════════════════════════════════════════════════════════════════════
// NEW: Auslautverhärtung (final devoicing)
// ══════════════════════════════════════════════════════════════════════

TEST_CASE("German LTS: Auslautverhärtung", "[g2p_de][devoicing]") {
    SECTION("final -b → p (Grab → ɡʁaːp)") {
        std::string ipa = g2p_de::lts_word_to_ipa("grab");
        // Last char should be 'p', not 'b'
        CHECK(ipa.back() == 'p');
    }
    SECTION("final -d → t (Rad → ʁaːt)") {
        std::string ipa = g2p_de::lts_word_to_ipa("rad");
        CHECK(ipa.back() == 't');
    }
    SECTION("final -g → k (Tag → taːk)") {
        std::string ipa = g2p_de::lts_word_to_ipa("tag");
        CHECK(ipa.back() == 'k');
    }
    SECTION("non-final b stays voiced") {
        std::string ipa = g2p_de::lts_word_to_ipa("bitte");
        // First char should be 'b'
        CHECK(ipa[0] == 'b');
    }
}

// ══════════════════════════════════════════════════════════════════════
// NEW: Open-syllable vowel lengthening
// ══════════════════════════════════════════════════════════════════════

TEST_CASE("German LTS: open-syllable lengthening", "[g2p_de][vowel_length]") {
    SECTION("open syllable: 'name' → ɑː (long)") {
        std::string ipa = g2p_de::lts_word_to_ipa("name");
        CHECK(ipa.find("\xC9\x91\xCB\x90") != std::string::npos); // ɑː
    }
    SECTION("closed syllable: 'mann' → a is short") {
        std::string ipa = g2p_de::lts_word_to_ipa("mann");
        // Should NOT have ɑː (double nn closes syllable)
        bool has_long_a = ipa.find("\xC9\x91\xCB\x90") != std::string::npos;
        CHECK(!has_long_a);
    }
    SECTION("open syllable ö: 'mögen' → øː") {
        std::string ipa = g2p_de::lts_word_to_ipa("m\xC3\xB6gen"); // mögen
        CHECK(ipa.find("\xC3\xB8\xCB\x90") != std::string::npos);  // øː
    }
    SECTION("closed syllable ö: 'möchte' → œ") {
        std::string ipa = g2p_de::lts_word_to_ipa("m\xC3\xB6"
                                                  "chte"); // möchte
        CHECK(ipa.find("\xC5\x93") != std::string::npos);  // œ
    }
    SECTION("open syllable ü: 'über' → yː") {
        std::string ipa = g2p_de::lts_word_to_ipa("\xC3\xBC"
                                                  "ber");  // über
        CHECK(ipa.find("y\xCB\x90") != std::string::npos); // yː
    }
}

// ══════════════════════════════════════════════════════════════════════
// NEW: Compound word splitting
// ══════════════════════════════════════════════════════════════════════

TEST_CASE("German compound splitting", "[g2p_de][compound]") {
    g2p_de::dictionary dict;
    // Set up a minimal test dictionary
    dict.entries["haus"] = "haʊ̯s";
    dict.entries["tür"] = "tyːɐ̯";
    dict.entries["auto"] = "aʊ̯to";
    dict.entries["bahn"] = "baːn";
    dict.loaded = true;

    SECTION("known compound splits") {
        auto parts = g2p_de::split_compound(dict, "autobahn");
        REQUIRE(parts.size() == 2);
        CHECK(parts[0] == "auto");
        CHECK(parts[1] == "bahn");
    }
    SECTION("non-compound returns single") {
        auto parts = g2p_de::split_compound(dict, "hallo");
        CHECK(parts.size() == 1);
    }
    SECTION("too short for compound") {
        auto parts = g2p_de::split_compound(dict, "haus");
        CHECK(parts.size() == 1);
    }
    SECTION("word_to_ipa uses compound splitting") {
        g2p_de::context ctx;
        ctx.dict = dict;
        std::string ipa = g2p_de::word_to_ipa(ctx, "autobahn");
        // Should contain parts from both dictionary entries
        bool has_ipa_chars = false;
        for (unsigned char c : ipa) {
            if (c >= 0x80) {
                has_ipa_chars = true;
                break;
            }
        }
        CHECK(has_ipa_chars);
    }
}

// ── Full word/text ──────────────────────────────────────────────────

TEST_CASE("German word_to_ipa produces IPA", "[g2p_de][word]") {
    g2p_de::context ctx;
    SECTION("common words produce non-empty IPA with Unicode") {
        std::string ipa = g2p_de::word_to_ipa(ctx, "hallo");
        REQUIRE(!ipa.empty());
    }
    SECTION("welt") {
        std::string ipa = g2p_de::word_to_ipa(ctx, "welt");
        CHECK(ipa[0] == 'v');
    }
}

TEST_CASE("German text_to_ipa handles sentences", "[g2p_de][sentence]") {
    g2p_de::context ctx;
    SECTION("hallo welt") {
        std::string ipa = g2p_de::text_to_ipa(ctx, "hallo welt");
        CHECK(!ipa.empty());
        CHECK(ipa.find(' ') != std::string::npos);
    }
    SECTION("longer sentence") {
        std::string ipa = g2p_de::text_to_ipa(ctx, "Ich bin ein Berliner");
        CHECK(!ipa.empty());
    }
}

// ── Dictionary loading ──────────────────────────────────────────────

TEST_CASE("German IPA dict loading", "[g2p_de][dict]") {
    g2p_de::context ctx;
    int n = g2p_de::load_ipa_dict_file(ctx.dict, "/tmp/ipa_dict_de.txt");
    if (n == 0) {
        SKIP("German IPA dict not available at /tmp/ipa_dict_de.txt");
    }
    INFO("Loaded " << n << " entries");
    CHECK(n > 100000);

    SECTION("hallo lookup") {
        std::string ipa = g2p_de::word_to_ipa(ctx, "hallo");
        CHECK(!ipa.empty());
        bool has_ipa = false;
        for (unsigned char c : ipa) {
            if (c >= 0x80) {
                has_ipa = true;
                break;
            }
        }
        CHECK(has_ipa);
    }
    SECTION("loanword: Restaurant") {
        std::string ipa = g2p_de::word_to_ipa(ctx, "Restaurant");
        CHECK(!ipa.empty());
    }
}

// ── #316 round 2: punctuation is the consumer's choice ──────────────────────
//
// Kokoro's 178-symbol vocabulary contains `,.;:!?` and they are how it pauses;
// dropping them delivered a paragraph in one breath. English proved it against
// misaki; this is the same defect in the same shape, one language over. Off by
// default so piper — whose espeak inventory has never been fed punctuation — is
// unchanged.

TEST_CASE("de: punctuation is dropped by default, kept on request", "[g2p_de][unit][punct]") {
    g2p_de::context ctx;
    const std::string plain = g2p_de::text_to_ipa(ctx, "a b, c.");
    CHECK(plain.find(',') == std::string::npos);
    CHECK(plain.find('.') == std::string::npos);
    // A dropped mark still separates its neighbours — with ONE space, not the
    // two the old loop emitted around every comma.
    CHECK(plain.find("  ") == std::string::npos);

    ctx.emit_punctuation = true;
    const std::string kept = g2p_de::text_to_ipa(ctx, "a b, c.");
    CHECK(kept.find(',') != std::string::npos);
    CHECK(kept.find('.') != std::string::npos);
    // A mark sits flush against the word before it and is followed by a space.
    CHECK(kept.find(" ,") == std::string::npos);
    CHECK(kept.find(", ") != std::string::npos);
}

TEST_CASE("de: the hyphen is a separator, never a mark", "[g2p_de][unit][punct]") {
    g2p_de::context ctx;
    ctx.emit_punctuation = true;
    // No TTS vocabulary here has an ASCII hyphen, and misaki drops it.
    CHECK(g2p_de::text_to_ipa(ctx, "a-b").find('-') == std::string::npos);
}

TEST_CASE("de: a quoted word is looked up without its quotes", "[g2p_de][unit][punct]") {
    // The tokenizer split on ,.!?;:- only, so a quoted word reached every
    // lookup tier WEARING its quotes and fell through to the letter-to-sound
    // rules — the same defect that made English read "dramatic" as DRAM-atic.
    g2p_de::context ctx;
    const std::string quoted = g2p_de::text_to_ipa(ctx, "\"ab\"");
    const std::string bare = g2p_de::text_to_ipa(ctx, "ab");
    CHECK(quoted == bare);
}

// ── #316: the citation-form stress our per-word dictionary bakes in ─────────

#include "core/g2p_de_unstressed.h"

TEST_CASE("de: the closed class loses its stress in running text", "[g2p_de][unit][stress]") {
    // Every one of these is espeak's own answer, taken from it in two carrier
    // frames by tools/gen-g2p-de-unstressed.py. espeak says `zˈiː` for "sie"
    // alone and `ziː` inside a sentence; our dictionary was generated one word
    // at a time, so it only ever had the first.
    CHECK(core_g2p_de_unstressed::lookup("sie") == "ziː");
    CHECK(core_g2p_de_unstressed::lookup("der") == "dɛɾ");
    CHECK(core_g2p_de_unstressed::lookup("nach") == "nɑːx");
    CHECK(core_g2p_de_unstressed::lookup("und") == "ʊnt");
    // A content word is not in the class and must never be reduced.
    CHECK(core_g2p_de_unstressed::lookup("hause").empty());
    CHECK(core_g2p_de_unstressed::lookup("wissenschaftlerin").empty());
    // Every entry is the citation form minus its stress marks and nothing else
    // — the generator rejects a candidate whose SEGMENTS change.
    for (const auto& kv : core_g2p_de_unstressed::table()) {
        CHECK(kv.second.find("ˈ") == std::string::npos);
        CHECK(kv.second.find("ˌ") == std::string::npos);
        CHECK(!kv.second.empty());
    }
}

TEST_CASE("de: the un-stressing is on, on source evidence", "[g2p_de][unit][stress]") {
    // Default ON since dida-80b/kokoro-deutsch's own dataset script was found:
    // it phonemizes the WHOLE SENTENCE through misaki's
    // EspeakG2P(language="de"), so espeak's sentence-level de-stressing is in
    // the training data and the citation form our per-word dictionary stores is
    // a spelling the model never saw. Token agreement with that recipe:
    // 41.2% -> 81.2%. CRISPASR_G2P_DE_UNSTRESS=0 restores the old forms.
    g2p_de::context ctx;
    CHECK(ctx.unstress_function_words);
}

// ── #316: symbols the German Kokoro vocabulary cannot receive ───────────────

#include "core/phoneme_dialect.h"

TEST_CASE("de: ʏ is not in the model vocabulary and must not reach it", "[g2p_de][unit][vocab]") {
    // A symbol the vocabulary lacks is not approximated by
    // kokoro_phonemes_to_ids — it is DROPPED, so the sound leaves the
    // utterance. `ʏ` is in every München, Frühstück, fünf, Glück and zurück,
    // and we were deleting the vowel out of all of them. Measured on the real
    // model: "Frühstück" came back from the ASR as "Frisch" and "Küche" as
    // "Kerre"; with the substitution they are "Frühstück" and "Kühe".
    //
    // dida-80b's own dataset script makes the same one ("ʏ → y … the duration
    // difference is learned from audio"), which is what confirms it is the
    // intended spelling rather than a workaround of ours.
    const std::string in = "mˈʏnçən frˈyːʃtʏk";
    const std::string out = core_phoneme::convert(in, core_phoneme::Dialect::DeVocab);
    CHECK(out.find("ʏ") == std::string::npos);
    CHECK(out == "mˈynçən frˈyːʃtyk");
    // The non-syllabic mark is not in the vocabulary either; dropping it leaves
    // the two vowels of the diphthong, which is what we want.
    CHECK(core_phoneme::convert("tsaɪ̯tʊŋ", core_phoneme::Dialect::DeVocab) == "tsaɪtʊŋ");
}

TEST_CASE("de: the misaki tied-sequence collapse is a separate, opt-in step", "[g2p_de][unit][vocab]") {
    // What the published recipe does (misaki's EspeakG2P e2m map). Kept
    // separate from the vocabulary fixups because the evidence differs: the
    // fixups are unambiguous, this one made the round-trip worse on the model
    // we ship. Both apply the fixups.
    CHECK(core_phoneme::convert("tsvˈaɪ", core_phoneme::Dialect::MisakiDe) == "ʦvˈI");
    CHECK(core_phoneme::convert("hˈaʊzə", core_phoneme::Dialect::MisakiDe) == "hˈWzə");
    CHECK(core_phoneme::convert("mˈʏnçən", core_phoneme::Dialect::MisakiDe) == "mˈynçən");
    // ...and DeVocab leaves the diphthongs alone.
    CHECK(core_phoneme::convert("tsvˈaɪ", core_phoneme::Dialect::DeVocab) == "tsvˈaɪ");
}

TEST_CASE("de: a single letter is read by its name, digits after it are kept", "[g2p_de][unit][letters]") {
    // Chess squares and moves: "d4", "Springer f3". The dictionary was built
    // one word at a time and has no single letters, so "d" fell through to the
    // letter-to-sound rules and came out `t` (final devoicing), and the digits
    // were dropped. espeak-ng, which the model was trained on, reads the name.
    g2p_de::context ctx; // no dictionary: tier 0 must not depend on it
    CHECK(g2p_de::word_to_ipa(ctx, "d") == "dˈeː");
    CHECK(g2p_de::word_to_ipa(ctx, "F") == "ˈɛf");
    CHECK(g2p_de::word_to_ipa(ctx, "ä") == "ˈɛː");
    const std::string d4 = g2p_de::text_to_ipa(ctx, "d4");
    CHECK(d4.rfind("dˈeː ", 0) == 0); // the letter's name, then the number
    CHECK(d4.size() > std::string("dˈeː ").size());
    CHECK(g2p_de::text_to_ipa(ctx, "f3").rfind("ˈɛf ", 0) == 0);
}
