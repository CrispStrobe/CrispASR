// tests/test-g2p-fr.cpp — unit tests for French G2P.

#include <catch2/catch_test_macros.hpp>
#include "core/g2p_fr.h"
#include <string>

TEST_CASE("French LTS: ch/ph/gn digraphs", "[g2p_fr][lts]") {
    SECTION("ch → ʃ") {
        std::string ipa = g2p_fr::lts_word_to_ipa("chat");
        CHECK(ipa.find("\xCA\x83") != std::string::npos);
    }
    SECTION("ph → f") {
        std::string ipa = g2p_fr::lts_word_to_ipa("photo");
        CHECK(ipa[0] == 'f');
    }
    SECTION("gn → ɲ") {
        std::string ipa = g2p_fr::lts_word_to_ipa("montagne");
        CHECK(ipa.find("\xC9\xB2") != std::string::npos);
    }
}

TEST_CASE("French LTS: nasal vowels", "[g2p_fr][lts]") {
    SECTION("on → ɔ̃") {
        std::string ipa = g2p_fr::lts_word_to_ipa("bon");
        CHECK(ipa.find("\xC9\x94\xCC\x83") != std::string::npos);
    }
    SECTION("an → ɑ̃") {
        std::string ipa = g2p_fr::lts_word_to_ipa("grand");
        CHECK(ipa.find("\xC9\x91\xCC\x83") != std::string::npos);
    }
    SECTION("in → ɛ̃") {
        std::string ipa = g2p_fr::lts_word_to_ipa("vin");
        CHECK(ipa.find("\xC9\x9B\xCC\x83") != std::string::npos);
    }
}

TEST_CASE("French LTS: vowel combinations", "[g2p_fr][lts]") {
    SECTION("ou → u") {
        std::string ipa = g2p_fr::lts_word_to_ipa("jour");
        CHECK(ipa.find("u") != std::string::npos);
    }
    SECTION("oi → wa") {
        std::string ipa = g2p_fr::lts_word_to_ipa("moi");
        CHECK(ipa.find("wa") != std::string::npos);
    }
    SECTION("eau → o") {
        std::string ipa = g2p_fr::lts_word_to_ipa("beau");
        CHECK(ipa.find("o") != std::string::npos);
    }
    SECTION("ai → ɛ") {
        std::string ipa = g2p_fr::lts_word_to_ipa("lait");
        CHECK(ipa.find("\xC9\x9B") != std::string::npos);
    }
}

TEST_CASE("French LTS: silent final consonants", "[g2p_fr][lts]") {
    SECTION("final -s is silent") {
        std::string ipa = g2p_fr::lts_word_to_ipa("pas");
        // Should end with 'a', not 's'
        CHECK(ipa.back() != 's');
    }
    SECTION("final -t is silent") {
        std::string ipa = g2p_fr::lts_word_to_ipa("fait");
        bool no_final_t = (ipa.find("t") == std::string::npos) || (ipa.back() != 't');
        CHECK(no_final_t);
    }
}

TEST_CASE("French LTS: accented vowels", "[g2p_fr][lts]") {
    SECTION("é → e") {
        std::string ipa = g2p_fr::lts_word_to_ipa("\xC3\xA9t\xC3\xA9"); // été
        // Should contain 'e' sounds
        CHECK(ipa.find("e") != std::string::npos);
    }
    SECTION("è → ɛ") {
        std::string ipa = g2p_fr::lts_word_to_ipa("m\xC3\xA8re"); // mère
        CHECK(ipa.find("\xC9\x9B") != std::string::npos);
    }
}

TEST_CASE("French LTS: consonant specifics", "[g2p_fr][lts]") {
    SECTION("j → ʒ") {
        std::string ipa = g2p_fr::lts_word_to_ipa("jour");
        CHECK(ipa.find("\xCA\x92") != std::string::npos);
    }
    SECTION("h is silent") {
        std::string ipa = g2p_fr::lts_word_to_ipa("homme");
        CHECK(ipa.find("h") == std::string::npos);
    }
    SECTION("r → ʁ") {
        std::string ipa = g2p_fr::lts_word_to_ipa("rouge");
        CHECK(ipa.find("\xCA\x81") != std::string::npos);
    }
    SECTION("u → y") {
        std::string ipa = g2p_fr::lts_word_to_ipa("lune");
        CHECK(ipa.find("y") != std::string::npos);
    }
    SECTION("s between vowels → z") {
        std::string ipa = g2p_fr::lts_word_to_ipa("rose");
        CHECK(ipa.find("z") != std::string::npos);
    }
}

TEST_CASE("French text_to_ipa", "[g2p_fr][sentence]") {
    g2p_fr::context ctx;
    SECTION("bonjour le monde") {
        std::string ipa = g2p_fr::text_to_ipa(ctx, "bonjour le monde");
        CHECK(!ipa.empty());
        CHECK(ipa.find(' ') != std::string::npos);
    }
    SECTION("empty") {
        CHECK(g2p_fr::text_to_ipa(ctx, "").empty());
    }
}
