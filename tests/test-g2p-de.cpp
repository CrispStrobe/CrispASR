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
        CHECK(ipa.find("\xCA\x83""p") != std::string::npos);
    }
    SECTION("initial st → ʃt") {
        std::string ipa = g2p_de::lts_word_to_ipa("stunde");
        CHECK(ipa.find("\xCA\x83""t") != std::string::npos);
    }
    SECTION("ng → ŋ") {
        std::string ipa = g2p_de::lts_word_to_ipa("lang");
        CHECK(ipa.find("\xC5\x8B") != std::string::npos);
    }
    SECTION("r → ʁ") {
        std::string ipa = g2p_de::lts_word_to_ipa("rot");
        CHECK(ipa.find("\xCA\x81") != std::string::npos);
    }
}

// ── Final schwa and -er ─────────────────────────────────────────────

TEST_CASE("German LTS: schwa and -er", "[g2p_de][lts]") {
    SECTION("final -e → ə") {
        std::string ipa = g2p_de::lts_word_to_ipa("habe");
        CHECK(ipa.find("\xC9\x99") != std::string::npos);
    }
    SECTION("final -er → ɐ") {
        std::string ipa = g2p_de::lts_word_to_ipa("besser");
        CHECK(ipa.find("\xC9\x90") != std::string::npos);
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
    SECTION("open syllable: 'name' → a is long (aː)") {
        std::string ipa = g2p_de::lts_word_to_ipa("name");
        CHECK(ipa.find("a\xCB\x90") != std::string::npos); // aː
    }
    SECTION("closed syllable: 'mann' → a is short") {
        std::string ipa = g2p_de::lts_word_to_ipa("mann");
        // Should NOT have aː (double nn closes syllable)
        bool has_long_a = ipa.find("a\xCB\x90") != std::string::npos;
        CHECK(!has_long_a);
    }
    SECTION("open syllable ö: 'mögen' → øː") {
        std::string ipa = g2p_de::lts_word_to_ipa("m\xC3\xB6gen"); // mögen
        CHECK(ipa.find("\xC3\xB8\xCB\x90") != std::string::npos); // øː
    }
    SECTION("closed syllable ö: 'möchte' → œ") {
        std::string ipa = g2p_de::lts_word_to_ipa("m\xC3\xB6""chte"); // möchte
        CHECK(ipa.find("\xC5\x93") != std::string::npos); // œ
    }
    SECTION("open syllable ü: 'über' → yː") {
        std::string ipa = g2p_de::lts_word_to_ipa("\xC3\xBC""ber"); // über
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
        for (unsigned char c : ipa) { if (c >= 0x80) { has_ipa_chars = true; break; } }
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
        for (unsigned char c : ipa) { if (c >= 0x80) { has_ipa = true; break; } }
        CHECK(has_ipa);
    }
    SECTION("loanword: Restaurant") {
        std::string ipa = g2p_de::word_to_ipa(ctx, "Restaurant");
        CHECK(!ipa.empty());
    }
}
