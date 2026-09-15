// tests/test-g2p-en.cpp — unit tests for English G2P phonemizer.
// Tests all three tiers: ARPAbet→IPA table, LTS rules, and the
// combined phonemizer pipeline.

#include <catch2/catch_test_macros.hpp>

#include "core/g2p_en.h"
#include "phonemizer.h"

#include <string>

// ── ARPAbet → IPA conversion ─────────────────────────────────────────

TEST_CASE("ARPAbet to IPA conversion", "[g2p][arpabet]") {
    SECTION("basic vowels") {
        CHECK(g2p_en::arpa_to_ipa("AA0") == "ɑː");
        CHECK(g2p_en::arpa_to_ipa("AE1") == "ˈæ");
        CHECK(g2p_en::arpa_to_ipa("IY0") == "i");   // unstressed → short
        CHECK(g2p_en::arpa_to_ipa("UW2") == "ˌuː"); // secondary stress emitted
        CHECK(g2p_en::arpa_to_ipa("AH0") == "ə");   // unstressed → schwa
        CHECK(g2p_en::arpa_to_ipa("EY1") == "ˈeɪ");
    }
    SECTION("basic consonants") {
        CHECK(g2p_en::arpa_to_ipa("B") == "b");
        CHECK(g2p_en::arpa_to_ipa("CH") == "tʃ");
        CHECK(g2p_en::arpa_to_ipa("SH") == "ʃ");
        CHECK(g2p_en::arpa_to_ipa("TH") == "θ");
        CHECK(g2p_en::arpa_to_ipa("DH") == "ð");
        CHECK(g2p_en::arpa_to_ipa("NG") == "ŋ");
        CHECK(g2p_en::arpa_to_ipa("ZH") == "ʒ");
        CHECK(g2p_en::arpa_to_ipa("HH") == "h");
        CHECK(g2p_en::arpa_to_ipa("JH") == "dʒ");
    }
    SECTION("stress markers and reductions") {
        CHECK(g2p_en::arpa_to_ipa("AH0") == "ə");   // unstressed → schwa
        CHECK(g2p_en::arpa_to_ipa("AH1") == "ˈʌ");  // primary stress → ʌ
        CHECK(g2p_en::arpa_to_ipa("AH2") == "ˌʌ");  // secondary stress
        CHECK(g2p_en::arpa_to_ipa("IH0") == "ɪ");   // espeak uses ɪ for unstressed
        CHECK(g2p_en::arpa_to_ipa("ER1") == "ˈɜː"); // stressed → ɜː
    }
    SECTION("case insensitivity") {
        CHECK(g2p_en::arpa_to_ipa("ah0") == "ə");
        CHECK(g2p_en::arpa_to_ipa("Sh") == "ʃ");
    }
    SECTION("unknown phoneme") {
        CHECK(g2p_en::arpa_to_ipa("XX") == "");
        CHECK(g2p_en::arpa_to_ipa("") == "");
    }
}

// ── LTS rules ────────────────────────────────────────────────────────

TEST_CASE("LTS rule-based phonemization", "[g2p][lts]") {
    SECTION("simple words produce non-empty ARPAbet") {
        auto phs = g2p_en::lts_predict("hello");
        REQUIRE(!phs.empty());
        // Should contain at least an H and a vowel
        bool has_hh = false, has_vowel = false;
        for (auto& p : phs) {
            std::string upper;
            for (char c : p)
                upper += (char)toupper((unsigned char)c);
            if (upper == "HH" || upper == "H")
                has_hh = true;
            if (upper.find("AH") != std::string::npos || upper.find("EH") != std::string::npos ||
                upper.find("OW") != std::string::npos || upper.find("IY") != std::string::npos)
                has_vowel = true;
        }
        CHECK(has_hh);
        CHECK(has_vowel);
    }
    SECTION("digraphs handled correctly") {
        auto phs = g2p_en::lts_predict("the");
        // 'th' should produce TH, not T+H
        REQUIRE(!phs.empty());
        bool has_th = false;
        for (auto& p : phs) {
            std::string upper;
            for (char c : p)
                upper += (char)toupper((unsigned char)c);
            if (upper == "TH")
                has_th = true;
        }
        CHECK(has_th);
    }
    SECTION("silent final e") {
        auto phs_make = g2p_en::lts_predict("make");
        auto phs_mak = g2p_en::lts_predict("mak");
        // "make" should have the same consonants but silent e
        CHECK(phs_make.size() <= phs_mak.size() + 1);
    }
    SECTION("sh digraph") {
        auto phs = g2p_en::lts_predict("she");
        REQUIRE(!phs.empty());
        bool has_sh = false;
        for (auto& p : phs) {
            std::string upper;
            for (char c : p)
                upper += (char)toupper((unsigned char)c);
            if (upper == "SH")
                has_sh = true;
        }
        CHECK(has_sh);
    }
}

// ── Word-to-IPA (full pipeline) ──────────────────────────────────────

TEST_CASE("word_to_ipa produces IPA output", "[g2p][ipa]") {
    g2p_en::context ctx; // no CMUdict or neural — pure LTS

    SECTION("common words produce IPA with Unicode characters") {
        std::string ipa = g2p_en::word_to_ipa(ctx, "hello");
        REQUIRE(!ipa.empty());
        // IPA should contain non-ASCII characters (ɛ, ʌ, etc.)
        bool has_nonascii = false;
        for (unsigned char c : ipa) {
            if (c >= 0x80) {
                has_nonascii = true;
                break;
            }
        }
        CHECK(has_nonascii);
    }

    SECTION("the produces ð") {
        std::string ipa = g2p_en::word_to_ipa(ctx, "the");
        CHECK(ipa.find("ð") != std::string::npos); // voiced th
    }

    SECTION("she produces ʃ") {
        std::string ipa = g2p_en::word_to_ipa(ctx, "she");
        CHECK(ipa.find("ʃ") != std::string::npos);
    }
}

// ── Text-to-IPA (full sentence) ──────────────────────────────────────

TEST_CASE("text_to_ipa handles full sentences", "[g2p][sentence]") {
    g2p_en::context ctx;

    SECTION("hello world produces non-empty IPA") {
        std::string ipa = g2p_en::text_to_ipa(ctx, "hello world");
        REQUIRE(!ipa.empty());
        // Should have a space separating two words
        CHECK(ipa.find(' ') != std::string::npos);
    }

    SECTION("punctuation is handled") {
        std::string ipa1 = g2p_en::text_to_ipa(ctx, "hello, world!");
        std::string ipa2 = g2p_en::text_to_ipa(ctx, "hello world");
        // Both should produce IPA (punctuation stripped/preserved)
        CHECK(!ipa1.empty());
        CHECK(!ipa2.empty());
    }

    SECTION("empty input") {
        std::string ipa = g2p_en::text_to_ipa(ctx, "");
        CHECK(ipa.empty());
    }

    SECTION("mixed case") {
        std::string ipa1 = g2p_en::text_to_ipa(ctx, "Hello");
        std::string ipa2 = g2p_en::text_to_ipa(ctx, "hello");
        CHECK(ipa1 == ipa2);
    }
}

// ── Tokenizer ────────────────────────────────────────────────────────

TEST_CASE("tokenizer splits correctly", "[g2p][tokenizer]") {
    SECTION("basic words") {
        auto t = g2p_en::tokenize("hello world");
        REQUIRE(t.size() == 2);
        CHECK(t[0] == "hello");
        CHECK(t[1] == "world");
    }
    SECTION("punctuation as separate tokens") {
        auto t = g2p_en::tokenize("hello, world!");
        REQUIRE(t.size() == 4);
        CHECK(t[0] == "hello");
        CHECK(t[1] == ",");
        CHECK(t[2] == "world");
        CHECK(t[3] == "!");
    }
    SECTION("multiple spaces") {
        auto t = g2p_en::tokenize("a  b");
        REQUIRE(t.size() == 2);
    }
}

// ── Neural G2P ──────────────────────────────────────────────────────

TEST_CASE("base64 decode", "[g2p][neural]") {
    // "hello" in base64 = "aGVsbG8="
    auto raw = g2p_en::base64_decode("aGVsbG8=");
    REQUIRE(raw.size() == 5);
    CHECK(raw[0] == 'h');
    CHECK(raw[1] == 'e');
    CHECK(raw[2] == 'l');
    CHECK(raw[3] == 'l');
    CHECK(raw[4] == 'o');
}

TEST_CASE("neural G2P JSON loading", "[g2p][neural]") {
    g2p_en::neural_model nm;

    SECTION("empty JSON returns false") {
        CHECK(!g2p_en::load_neural_g2p_json(nm, ""));
        CHECK(!nm.loaded);
    }

    SECTION("file loading from path") {
        const char* env = std::getenv("CRISPASR_G2P_MODEL_PATH");
        std::string path = env ? env : "";
        if (path.empty()) {
            // Try cache dir
            const char* home = std::getenv("HOME");
            if (home)
                path = std::string(home) + "/.cache/crispasr/g2p_en.json";
        }
        if (path.empty() || !g2p_en::load_neural_g2p_file(nm, path)) {
            SKIP("g2p_en.json not available — set CRISPASR_G2P_MODEL_PATH");
        }
        REQUIRE(nm.loaded);
        CHECK(nm.graphemes.size() == 29);
        CHECK(nm.phonemes.size() == 74);
        CHECK(!nm.enc_emb.empty());

        SECTION("neural predict produces phonemes") {
            auto phs = g2p_en::neural_predict(nm, "hello");
            REQUIRE(!phs.empty());
            INFO("neural prediction for 'hello': ");
            for (auto& p : phs)
                INFO("  " << p);
        }
    }
}

// ── CMUdict loading ──────────────────────────────────────────────────

TEST_CASE("CMUdict file loading", "[g2p][cmudict]") {
    g2p_en::context ctx;

    SECTION("load from file") {
        // Try loading CMUdict from known locations
        const char* paths[] = {"/tmp/cmudict.dict", "models/cmudict.dict", nullptr};
        bool loaded = false;
        for (int i = 0; paths[i]; i++) {
            int n = g2p_en::load_cmudict_file(ctx.dict, paths[i]);
            if (n > 0) {
                INFO("Loaded " << n << " entries from " << paths[i]);
                loaded = true;
                break;
            }
        }
        if (!loaded) {
            SKIP("CMUdict file not found — skipping (place cmudict.dict in /tmp/)");
        }
        REQUIRE(ctx.dict.loaded);
        REQUIRE(ctx.dict.entries.size() > 100000);

        SECTION("HELLO lookup") {
            auto it = ctx.dict.entries.find("HELLO");
            REQUIRE(it != ctx.dict.entries.end());
            // HELLO -> HH AH0 L OW1 or HH EH0 L OW1
            CHECK(!it->second.empty());
            CHECK(it->second[0] == "HH");
        }

        SECTION("word_to_ipa with dict produces stressed output") {
            std::string ipa_dict = g2p_en::word_to_ipa(ctx, "hello");
            // With CMUdict, "hello" should have proper stress
            CHECK(ipa_dict.find("ˈ") != std::string::npos);
        }

        SECTION("THE lookup") {
            auto it = ctx.dict.entries.find("THE");
            REQUIRE(it != ctx.dict.entries.end());
            // THE -> DH AH0 or DH AH1 or DH IY0
            bool has_dh = false;
            for (auto& ph : it->second) {
                if (ph == "DH")
                    has_dh = true;
            }
            CHECK(has_dh);
        }
    }
}

// ── CMUdict IPA quality ──────────────────────────────────────────────

TEST_CASE("CMUdict IPA output quality", "[g2p][cmudict][quality]") {
    g2p_en::context ctx;
    int n = g2p_en::load_cmudict_file(ctx.dict, "/tmp/cmudict.dict");
    if (n == 0) {
        SKIP("CMUdict not available");
    }

    SECTION("common words produce expected IPA") {
        // "hello" should contain h and a stressed vowel
        std::string ipa = g2p_en::word_to_ipa(ctx, "hello");
        CHECK(ipa.find("h") != std::string::npos);
        CHECK(!ipa.empty());
    }

    SECTION("world has ɜː (stressed ER)") {
        std::string ipa = g2p_en::word_to_ipa(ctx, "world");
        bool has_r = ipa.find("ɜː") != std::string::npos || ipa.find("ɚ") != std::string::npos;
        CHECK(has_r);
    }

    SECTION("fox has f") {
        std::string ipa = g2p_en::word_to_ipa(ctx, "fox");
        CHECK(ipa.find("f") != std::string::npos);
    }
}

// ── Phoneme inventory filtering ──────────────────────────────────────

TEST_CASE("filter_to_inventory strips unmapped chars", "[phonemizer][inventory]") {
    // Simulated piper phoneme map (subset)
    std::set<std::string> valid = {"t", "s", "ʃ", "d", "ʒ", "a", "e", "i", "o", "u",
                                   "ɪ", "ɛ", "ɔ", "ʊ", "ə", "ˈ", "ˌ", "ː", "ŋ"};

    SECTION("passes valid IPA through") {
        std::string filtered = crispasr::filter_to_inventory("tʃaɪ", valid);
        CHECK(filtered == "tʃaɪ");
    }

    SECTION("strips unknown combining marks") {
        // U+0361 combining tie should be stripped if not in valid set
        std::string with_tie = "t\xCD\xA1s"; // t͡s
        std::string filtered = crispasr::filter_to_inventory(with_tie, valid);
        CHECK(filtered == "ts");
    }

    SECTION("preserves spaces") {
        std::string filtered = crispasr::filter_to_inventory("a e", valid);
        CHECK(filtered == "a e");
    }
}

// ── Phonemizer interface ─────────────────────────────────────────────

TEST_CASE("phonemizer builtin_en works without espeak", "[phonemizer]") {
    std::string out;

    SECTION("English text produces IPA") {
        bool ok = crispasr::phonemize_builtin_en("en-us", "hello world", out);
        CHECK(ok);
        CHECK(!out.empty());
    }

    SECTION("auto language works") {
        bool ok = crispasr::phonemize_builtin_en("auto", "hello", out);
        CHECK(ok);
        CHECK(!out.empty());
    }

    SECTION("empty language works") {
        bool ok = crispasr::phonemize_builtin_en("", "hello", out);
        CHECK(ok);
    }

    SECTION("non-English returns false") {
        bool ok = crispasr::phonemize_builtin_en("de", "hallo", out);
        CHECK(!ok);
    }
}

TEST_CASE("phonemize() cascade works", "[phonemizer]") {
    std::string out;
    // Even without espeak, the built-in English G2P should produce output
    bool ok = crispasr::phonemize("en-us", "The quick brown fox", out);
    CHECK(ok);
    CHECK(!out.empty());
    // Should contain IPA characters
    bool has_ipa = false;
    for (unsigned char c : out) {
        if (c >= 0x80) {
            has_ipa = true;
            break;
        }
    }
    CHECK(has_ipa);
}

// ── espeak language-marker stripping (#169) ─────────────────────────

TEST_CASE("strip espeak language markers", "[phonemizer][espeak]") {
    using crispasr::strip_espeak_lang_markers;

    SECTION("basic (en) and (it) markers") {
        std::string s = "lassistɛnte vokʈale dɪ (en)wˈeɪ(it)";
        strip_espeak_lang_markers(s);
        CHECK(s.find("(en)") == std::string::npos);
        CHECK(s.find("(it)") == std::string::npos);
        CHECK(s.find("wˈeɪ") != std::string::npos); // IPA preserved
    }

    SECTION("region codes (de-AT), (pt-BR)") {
        std::string s = "fü(de-AT)nf (pt-BR)teste";
        strip_espeak_lang_markers(s);
        CHECK(s.find("(de-AT)") == std::string::npos);
        CHECK(s.find("(pt-BR)") == std::string::npos);
    }

    SECTION("three-letter codes (cmn)") {
        std::string s = "hello (cmn)ni hao (en)world";
        strip_espeak_lang_markers(s);
        CHECK(s.find("(cmn)") == std::string::npos);
        CHECK(s.find("(en)") == std::string::npos);
        CHECK(s.find("hello") != std::string::npos);
        CHECK(s.find("world") != std::string::npos);
    }

    SECTION("no markers passes through unchanged") {
        std::string s = "həlˈoʊ wˈɜːld";
        std::string orig = s;
        strip_espeak_lang_markers(s);
        CHECK(s == orig);
    }

    SECTION("non-language parens preserved") {
        std::string s = "test (1234) and (AB) keep";
        std::string orig = s;
        strip_espeak_lang_markers(s);
        CHECK(s == orig); // not language codes
    }

    SECTION("empty string") {
        std::string s;
        strip_espeak_lang_markers(s);
        CHECK(s.empty());
    }
}

// ── Technical token normalization (#216) ─────────────────────────────

TEST_CASE("Technical token normalization", "[g2p][normalize]") {
    SECTION("C++ expanded") {
        auto r = g2p_en::normalize_technical_tokens("modern C++, Linux");
        CHECK(r == "modern C plus plus, Linux");
    }

    SECTION("C# expanded") {
        auto r = g2p_en::normalize_technical_tokens("wrote it in C# and F#");
        CHECK(r == "wrote it in C sharp and F sharp");
    }

    SECTION(".NET expanded") {
        auto r = g2p_en::normalize_technical_tokens("using .NET for web");
        CHECK(r == "using dot net for web");
    }

    SECTION("Node.js expanded") {
        auto r = g2p_en::normalize_technical_tokens("built with Node.js");
        CHECK(r == "built with Node J S");
    }

    SECTION("case insensitive") {
        auto r = g2p_en::normalize_technical_tokens("c++ and node.js");
        CHECK(r == "C plus plus and Node J S");
    }

    SECTION("mid-word not matched") {
        // "AC++" should not match — C++ must be at a word boundary
        auto r = g2p_en::normalize_technical_tokens("AC++ test");
        CHECK(r == "AC++ test");
    }

    SECTION("no technical tokens passes through") {
        std::string input = "hello world, this is a test.";
        auto r = g2p_en::normalize_technical_tokens(input);
        CHECK(r == input);
    }

    SECTION("multiple tokens in one sentence") {
        auto r = g2p_en::normalize_technical_tokens("C++ and C# with .NET on Node.js");
        CHECK(r == "C plus plus and C sharp with dot net on Node J S");
    }

    SECTION("CI/CD expanded") {
        auto r = g2p_en::normalize_technical_tokens("setup CI/CD pipeline");
        CHECK(r == "setup C I C D pipeline");
    }

    SECTION("empty string") {
        auto r = g2p_en::normalize_technical_tokens("");
        CHECK(r.empty());
    }
}

// ── #435: language dispatch + the TTS convention ─────────────────────
//
// These two properties are what zonos (and any future non-kokoro consumer)
// depends on, and both are the kind a refactor breaks without failing anything
// else: the dispatch is a string comparison, and the punctuation flag is a bool
// that defaults to the wrong value for this consumer.

TEST_CASE("builtin_g2p_language matches the PRIMARY SUBTAG, not a substring", "[unit][g2p][phonemizer]") {
    using crispasr::builtin_g2p_language;

    SECTION("the four covered languages, bare and with a region") {
        CHECK(std::string(builtin_g2p_language("en")) == "en");
        CHECK(std::string(builtin_g2p_language("en-us")) == "en");
        CHECK(std::string(builtin_g2p_language("en-gb")) == "en");
        CHECK(std::string(builtin_g2p_language("de")) == "de");
        CHECK(std::string(builtin_g2p_language("de-at")) == "de");
        CHECK(std::string(builtin_g2p_language("fr")) == "fr");
        CHECK(std::string(builtin_g2p_language("fr-fr")) == "fr");
        CHECK(std::string(builtin_g2p_language("es")) == "es");
        CHECK(std::string(builtin_g2p_language("es-419")) == "es");
        CHECK(std::string(builtin_g2p_language("ES-419")) == "es"); // case-insensitive
        CHECK(std::string(builtin_g2p_language("en_US")) == "en");  // underscore form
    }

    SECTION("Russian is covered now") {
        // This section used to assert `builtin_g2p_language("ru") == nullptr`.
        // That was true when written and is not any more: a built-in Russian
        // G2P ships (812,953-entry CC-BY-4.0 dictionary + LTS fallback), so the
        // dispatcher must route ru rather than fall through to espeak.
        //
        // Note this says nothing about zonos's DEFAULT, which still prefers
        // espeak for ru — zonos-v0.1 does not list Russian, so its roundtrip
        // could not certify the swap in either direction. Coverage and default
        // are separate questions and this test only answers the first.
        CHECK(std::string(builtin_g2p_language("ru")) == "ru");
        CHECK(std::string(builtin_g2p_language("RU")) == "ru");
    }

    SECTION("uncovered languages return nullptr, so the caller falls through") {
        CHECK(builtin_g2p_language("ja") == nullptr);
        CHECK(builtin_g2p_language("cmn") == nullptr);
        CHECK(builtin_g2p_language("") == nullptr);
    }

    SECTION("the substring trap the per-language functions still have") {
        // phonemize_builtin_en tests `lang.find("en") != npos`, so a code that
        // merely CONTAINS the digraph is claimed by the wrong G2P. Assert both
        // halves: that the raw function really does claim it (otherwise this
        // test is guarding a hazard that no longer exists and would keep
        // passing after the dispatch regressed), and that the dispatch does not.
        std::string out;
        const bool raw_claims_it = crispasr::phonemize_builtin_en("ben", "hello", out);
        CHECK(raw_claims_it); // the hazard is real
        CHECK(builtin_g2p_language("ben") == nullptr);
        CHECK(builtin_g2p_language("hi-en") == nullptr); // primary subtag is "hi"
        CHECK(builtin_g2p_language("nl") == nullptr);
    }
}

TEST_CASE("phonemize_builtin_tts carries punctuation through", "[unit][g2p][phonemizer]") {
    // Zonos's Python reference phonemizes with preserve_punctuation=True and its
    // symbol table contains `;:,.!?`, which is how the model pauses. The
    // built-in G2P defaults that flag OFF (piper's inventory never saw
    // punctuation), so this entry point has to turn it on — and if it ever
    // stops, the failure is a paragraph delivered in one breath, which no
    // roundtrip metric would call a failure.
    std::string out;
    REQUIRE(crispasr::phonemize_builtin_tts("en-us", "Hello, world. Is it?", out));
    CHECK(out.find(',') != std::string::npos);
    CHECK(out.find('.') != std::string::npos);
    CHECK(out.find('?') != std::string::npos);

    SECTION("and the default English entry point still does NOT — piper's contract") {
        // The positive control for the check above: if emit_punctuation were
        // globally on, both assertions would pass and neither would mean
        // anything.
        std::string plain;
        REQUIRE(crispasr::phonemize_builtin_en("en-us", "Hello, world. Is it?", plain));
        CHECK(plain.find(',') == std::string::npos);
        CHECK(plain.find('?') == std::string::npos);
    }

    SECTION("an uncovered language is refused, not silently mis-phonemised") {
        // ru is no longer the example — it is covered. Japanese still is not,
        // and refusing is the point: a silent mis-phonemisation is what made
        // zonos emit confident noise for Cyrillic in #435.
        std::string ja;
        CHECK_FALSE(crispasr::phonemize_builtin_tts("ja", "\xe3\x81\x93\xe3\x82\x93", ja));
    }
}
