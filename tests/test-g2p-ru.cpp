// tests/test-g2p-ru.cpp — unit tests for Russian G2P and Russian number words.
//
// Weight-free and hermetic: no model, no downloaded dictionary. Where a test
// needs a dictionary it builds a three-entry one inline, because what is being
// tested is the LOOKUP BEHAVIOUR (the ё fold, the analogy tier, the heteronym
// fall-through), not the 813K entries themselves.
//
// The last TEST_CASE is the one that matters most and is easiest to skip
// writing: an INVENTORY INVARIANT. Every symbol the rule path can emit must be
// inside the same 37-codepoint set the dictionary uses. A rule that emits one
// codepoint outside it does not crash and does not look wrong in a log — the
// consumer (zonos's text_to_phoneme_ids, piper's phoneme_id_map) DROPS it
// silently, which is exactly how #435 shipped a backend that produced noise at
// a success exit code. It has its own positive control, so a checker that
// cannot fail is caught before it certifies anything.

#include <catch2/catch_test_macros.hpp>
#include "core/g2p_ru.h"
#include <set>
#include <string>

using g2p_ru::utf8_to_cps;

static std::string lts(const std::string& w, int stress = -2) {
    const auto cps = g2p_ru::normalize_word(w);
    g2p_ru::context ctx;
    const int s = (stress == -2) ? g2p_ru::guess_stress(ctx, cps) : stress;
    return g2p_ru::lts_word_to_ipa(cps, s);
}

static bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// The 36 IPA codepoints the dictionary uses, plus U+02C8 for the stress mark
// that replaced its backtick. A rule that emits anything outside this set is
// DROPPED without comment by every consumer downstream.
static const std::set<uint32_t>& ru_inventory() {
    static const std::set<uint32_t> s = {
        'a',    'b', 'd', 'e', 'f', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'x', 'z',
        0x00E6, // æ
        0x0250, // ɐ
        0x0255, // ɕ
        0x0259, // ə
        0x025B, // ɛ
        0x0261, // ɡ
        0x0268, // ɨ
        0x026A, // ɪ
        0x026B, // ɫ
        0x0275, // ɵ
        0x0282, // ʂ
        0x0289, // ʉ
        0x028A, // ʊ
        0x0290, // ʐ
        0x02B2, // ʲ
        0x02D0, // ː
        0x02C8, // ˈ  (was ` U+0060 upstream)
    };
    return s;
}

// espeak-ng's `ru` spelling of the same sounds. A DIFFERENT set — that is the
// whole point of the conversion — and every member was checked against zonos's
// conditioning.py symbol list, so converting loses nothing downstream either.
static const std::set<uint32_t>& ru_espeak_inventory() {
    static const std::set<uint32_t> s = [] {
        std::set<uint32_t> v = ru_inventory();
        // Symbols the conversion REMOVES from the output.
        for (uint32_t cp : {(uint32_t)'a', (uint32_t)0x00E6, (uint32_t)0x0250, (uint32_t)0x0259, (uint32_t)0x0268,
                            (uint32_t)0x028A, (uint32_t)0x0289, (uint32_t)0x0275, (uint32_t)0x0282, (uint32_t)0x0290,
                            (uint32_t)0x026B, (uint32_t)'l', (uint32_t)0x02D0})
            v.erase(cp);
        // ...and the ones it introduces: ɑ ʌ y ʃ ʒ ɭ.
        for (uint32_t cp :
             {(uint32_t)0x0251, (uint32_t)0x028C, (uint32_t)'y', (uint32_t)0x0283, (uint32_t)0x0292, (uint32_t)0x026D})
            v.insert(cp);
        return v;
    }();
    return s;
}

static std::set<uint32_t> outside(const std::string& ipa, const std::set<uint32_t>& inv) {
    std::set<uint32_t> bad;
    for (uint32_t cp : utf8_to_cps(ipa))
        if (cp != ' ' && !inv.count(cp))
            bad.insert(cp);
    return bad;
}

static std::set<uint32_t> outside_inventory(const std::string& ipa) {
    std::set<uint32_t> bad;
    for (uint32_t cp : utf8_to_cps(ipa))
        if (cp != ' ' && !ru_inventory().count(cp))
            bad.insert(cp);
    return bad;
}

// ── numbers ─────────────────────────────────────────────────────────────────

TEST_CASE("Russian numerals: the three that no pattern produces", "[g2p_ru][num]") {
    // 40 and 90 are not built from 4 and 9, and 200 is not built from "два"+"сто".
    CHECK(core_num2words_ru::cardinal(40) == "сорок");
    CHECK(core_num2words_ru::cardinal(90) == "девяносто");
    CHECK(core_num2words_ru::cardinal(200) == "двести");
    CHECK(core_num2words_ru::cardinal(300) == "триста");
    CHECK(core_num2words_ru::cardinal(500) == "пятьсот");
    CHECK(core_num2words_ru::cardinal(0) == "ноль");
}

TEST_CASE("Russian numerals: the counted-noun form, including 11-14", "[g2p_ru][num]") {
    // n % 10 alone gives "одиннадцать тысяча" and "двенадцать тысячи". The
    // 11-14 window is the whole reason plural_form() exists.
    CHECK(core_num2words_ru::cardinal(1000) == "одна тысяча");
    CHECK(core_num2words_ru::cardinal(2000) == "две тысячи");
    CHECK(core_num2words_ru::cardinal(5000) == "пять тысяч");
    CHECK(core_num2words_ru::cardinal(11000) == "одиннадцать тысяч");
    CHECK(core_num2words_ru::cardinal(12000) == "двенадцать тысяч");
    CHECK(core_num2words_ru::cardinal(14000) == "четырнадцать тысяч");
    CHECK(core_num2words_ru::cardinal(21000) == "двадцать одна тысяча");
    CHECK(core_num2words_ru::cardinal(22000) == "двадцать две тысячи");
}

TEST_CASE("Russian numerals: тысяча is feminine and the millions are not", "[g2p_ru][num]") {
    // One gender assumption gets exactly one of these two wrong.
    CHECK(core_num2words_ru::cardinal(2000) == "две тысячи");
    CHECK(core_num2words_ru::cardinal(2000000) == "два миллиона");
    CHECK(core_num2words_ru::cardinal(1000000) == "один миллион");
    CHECK(core_num2words_ru::cardinal(5000000) == "пять миллионов");
}

TEST_CASE("Russian numerals: 2024 reads as a year does", "[g2p_ru][num]") {
    CHECK(core_num2words_ru::cardinal(2024) == "две тысячи двадцать четыре");
}

TEST_CASE("Russian numerals: a digit run does not vanish", "[g2p_ru][num]") {
    // The defect this file exists for: an unexpanded digit is in no dictionary,
    // matches no rule, and is absent from the TTS phoneme inventory, so it is
    // dropped twice over without a word.
    g2p_ru::context ctx;
    const std::string ipa = g2p_ru::text_to_ipa(ctx, "5 книг");
    CHECK_FALSE(ipa.empty());
    CHECK_FALSE(has(ipa, "5"));
    // "пять" -> pʲætʲ, so the fronted vowel has to be in there.
    CHECK(has(ipa, "\xc3\xa6"));
}

// ── letter-to-sound ─────────────────────────────────────────────────────────

TEST_CASE("Russian LTS: final devoicing", "[g2p_ru][lts]") {
    CHECK(lts("сад") == "sat");
    CHECK(lts("год") == "\xc9\xa1ot"); // ɡot
}

TEST_CASE("Russian LTS: regressive voicing assimilation, and в as the exception", "[g2p_ru][lts]") {
    // в devoices before a voiceless obstruent...
    CHECK(lts("все") == "fs\xca\xb2"
                        "e"); // fsʲe
    // ...but does NOT voice what precedes it: свой is [svoj], never [zvoj].
    CHECK(has(lts("свой"), "svoj"));
}

TEST_CASE("Russian LTS: palatalisation, and the three letters that never take it", "[g2p_ru][lts]") {
    CHECK(has(lts("мать"), "mat\xca\xb2"));                // matʲ
    CHECK(has(lts("конь"), "kon\xca\xb2"));                // konʲ
    CHECK(has(lts("жизнь"), "\xca\x90\xc9\xa8z\xca\xb2")); // ʐɨzʲ — ж stays hard, з softens
    CHECK_FALSE(has(lts("шесть"), "\xca\x82\xca\xb2"));    // ʂ never carries ʲ
}

TEST_CASE("Russian LTS: regressive softening of dentals, and its narrower trigger set", "[g2p_ru][lts]") {
    // A soft т/д/н/л pulls the dental before it soft...
    CHECK(has(lts("жизнь"), "z\xca\xb2n\xca\xb2"));   // зʲнʲ
    CHECK(has(lts("сегодня"), "d\xca\xb2n\xca\xb2")); // дʲнʲ
    // ...a soft с/з does NOT. `отсюда` is ɐtsʲˈudə, not ɐtʲsʲ…
    CHECK_FALSE(has(lts("отсюда"), "t\xca\xb2s\xca\xb2"));
}

TEST_CASE("Russian LTS: /a/ and /u/ front only BETWEEN two soft segments", "[g2p_ru][lts]") {
    // Measured over the whole dictionary: stressed `я` is [æ] before a soft
    // consonant 100.0% of the time and [a] otherwise 97.2%. Conditioning on the
    // PRECEDING consonant instead splits it ~50/50, i.e. it is no rule at all.
    CHECK(has(lts("пять"), "\xc3\xa6"));           // пʲætʲ — soft on both sides
    CHECK(has(lts("мяч"), "\xc3\xa6"));            // мʲætɕ
    CHECK_FALSE(has(lts("говяжьей"), "\xc3\xa6")); // ɡɐvʲˈaʐjɪj — ж is hard
    CHECK_FALSE(has(lts("яблоко"), "\xc3\xa6"));   // ˈjabɫəkə — б is hard
}

TEST_CASE("Russian LTS: л is two different sounds", "[g2p_ru][lts]") {
    CHECK(has(lts("молоко"), "\xc9\xab")); // ɫ, dark
    CHECK(has(lts("люди"), "l\xca\xb2"));  // lʲ, light
    CHECK_FALSE(has(lts("люди"), "\xc9\xab"));
}

TEST_CASE("Russian LTS: ё is always stressed and carries its own vowel", "[g2p_ru][lts]") {
    // The one stress signal the input can carry for free. Both the position and
    // the quality come out of it.
    const std::string yo = lts("ёлка");
    CHECK(has(yo, "\xc9\xb5")); // ɵ
    // Extra parens: Catch2's expression decomposer cannot take `a == b || c < d`
    // and rejects it as a chained comparison at compile time.
    CHECK((yo.find("\xcb\x88") == std::string::npos ||
           yo.find("\xcb\x88") < yo.find("\xc9\xb5"))); // the mark precedes it
    CHECK(has(lts("жёлтый"), "o"));                     // after hard ж it is [o], not [ɵ]
    CHECK_FALSE(has(lts("жёлтый"), "\xc9\xb5"));
    // всё vs все — the ё is the ONLY thing separating them, in the input and
    // in the output. (The dictionary cannot: its keys fold ё to е.)
    CHECK(lts("всё") != lts("все"));
}

TEST_CASE("Russian LTS: the spellings that fuse", "[g2p_ru][lts]") {
    CHECK(has(lts("счастье"), "\xc9\x95\xcb\x90"));   // сч  -> ɕː
    CHECK(has(lts("изживала"), "\xca\x90\xcb\x90"));  // зж  -> ʐː
    CHECK(has(lts("отца"), "ts\xcb\x90"));            // тц  -> tsː
    CHECK(has(lts("спиться"), "ts\xcb\x90\xc9\x99")); // -ться -> [tsːə]
    CHECK(has(lts("тесанные"), "n\xcb\x90"));         // нн  -> nː
}

TEST_CASE("Russian LTS: -ого/-его is read with [v], reflexive included", "[g2p_ru][lts]") {
    CHECK(has(lts("которого"), "v"));
    CHECK_FALSE(has(lts("которого"), "\xc9\xa1\xc9\x99")); // not ɡə
    // The reflexive -ся sits AFTER the ending, two letters further from the end.
    CHECK(has(lts("отвергнувшегося"), "v\xc9\x99s\xca\xb2"));
}

TEST_CASE("Russian LTS: vowel reduction is driven by distance from the stress", "[g2p_ru][lts]") {
    // молоко: the SAME letter о three times, three different vowels, and the
    // only thing choosing between them is where the stress is. This is why a
    // wrong stress guess is a segmental error and not a prosodic one.
    const std::string m = lts("молоко", 2);            // stress on the third vowel
    CHECK(m == "m\xc9\x99\xc9\xab\xc9\x90k\xcb\x88o"); // məɫɐkˈo
    const std::string m0 = lts("молоко", 0);           // stress on the first
    CHECK(m0 != m);
    CHECK(has(m0, "\xcb\x88o")); // the mark moved with it
}

// ── dictionary behaviour ────────────────────────────────────────────────────

static g2p_ru::context with_mini_dict() {
    g2p_ru::context ctx;
    // Keys deliberately written the way the real file writes them: NO ё.
    ctx.dict.entries["елка"] = "\xcb\x88jj";                           // sentinel, not real IPA
    ctx.dict.entries["тесты"] = "t\xca\xb2\xcb\x88\xc9\x9bst\xc9\xa8"; // tʲˈɛstɨ
    ctx.dict.loaded = true;
    return ctx;
}

TEST_CASE("Russian dictionary: lookup folds ё to е, because the keys do", "[g2p_ru][dict]") {
    // The upstream vocabulary contains no ё at all. A lookup that does not fold
    // misses every ё word in the input — a 100% miss on a letter that only ever
    // appears in stressed syllables, and it fails silently by falling through
    // to the rules.
    g2p_ru::context ctx = with_mini_dict();
    CHECK(g2p_ru::word_to_ipa(ctx, "ёлка") == "\xcb\x88jj");
    CHECK(g2p_ru::word_to_ipa(ctx, "елка") == "\xcb\x88jj");
    CHECK(g2p_ru::word_to_ipa(ctx, "Ёлка") == "\xcb\x88jj"); // and it lowercases
}

TEST_CASE("Russian stress analogy: an absent bare form borrows from a present inflection", "[g2p_ru][dict]") {
    // `тест` is genuinely absent from the vocabulary while `тесты` is present.
    // This tier is worth 47 points of stressed-syllable accuracy on held-out
    // words (93.9% with, 47.1% without), so it gets a test that distinguishes
    // the two states rather than just running.
    g2p_ru::context ctx = with_mini_dict();
    const auto w = g2p_ru::normalize_word("тестом");
    bool used = false;
    const int s = g2p_ru::guess_stress(ctx, w, &used);
    CHECK(used);
    CHECK(s == 0); // тесты is stressed on its first vowel

    ctx.stress_analogy = false;
    used = false;
    g2p_ru::guess_stress(ctx, w, &used);
    CHECK_FALSE(used); // the control: the tier can be off, and then it is
}

TEST_CASE("Russian stress: index read back out of a dictionary entry", "[g2p_ru][dict]") {
    CHECK(g2p_ru::stressed_vowel_index("m\xc9\x99\xc9\xab\xc9\x90k\xcb\x88o") == 2); // məɫɐkˈo
    CHECK(g2p_ru::stressed_vowel_index("t\xca\xb2\xcb\x88\xc9\x9bst\xc9\xa8") == 0); // tʲˈɛstɨ
    // A monosyllable carries no mark in this dictionary, and -1 is how that is
    // reported — NOT 0, which would silently claim a mark that is not there.
    CHECK(g2p_ru::stressed_vowel_index("sat") == -1);
}

TEST_CASE("Russian heteronyms: they are absent from the dictionary, so they reach the rules", "[g2p_ru][dict]") {
    // The upstream heteronym list and vocabulary are DISJOINT: 0 of the 17,359
    // heteronyms appear in the vocabulary. They were removed because upstream
    // could not choose a reading, so "take the dictionary's reading" is not an
    // option that exists for them. What must hold is that they still produce
    // phonemes rather than nothing.
    g2p_ru::context ctx = with_mini_dict();
    ctx.het.words.insert("замок");
    ctx.het.loaded = true;
    const std::string ipa = g2p_ru::word_to_ipa(ctx, "замок");
    CHECK_FALSE(ipa.empty());
    CHECK(ipa == lts("замок")); // i.e. the rule path answered, not a lookup
}

TEST_CASE("Russian input: an explicit combining acute is honoured and never leaks", "[g2p_ru][dict]") {
    // U+0301 is how Russian reference text, dictionaries and learner material
    // write stress (`Росси́я`). Two things must happen to it and neither is
    // optional:
    //   1. it must be STRIPPED before the dictionary lookup — the vocabulary is
    //      written without it, so a marked word otherwise misses every entry;
    //   2. it must never reach the phoneme string, where it is outside every
    //      TTS inventory and would be dropped without a word.
    // And it carries information worth using: it is the writer disambiguating a
    // heteronym on purpose, which is the one case where the input knows
    // something the dictionary cannot.
    int st = -1;
    const auto w = g2p_ru::normalize_word("замо́к", &st);
    CHECK(st == 1); // the mark sat on the second vowel
    CHECK(g2p_ru::cps_to_utf8(w) == "замок");

    g2p_ru::context ctx;
    const std::string castle = g2p_ru::word_to_ipa(ctx, "за́мок");
    const std::string lock = g2p_ru::word_to_ipa(ctx, "замо́к");
    CHECK(castle != lock);                    // the one distinction that matters
    CHECK(outside_inventory(castle).empty()); // and no U+0301 survived into either
    CHECK(outside_inventory(lock).empty());
    CHECK(has(castle, "\xcb\x88"
                      "a"));       // ˈa — stressed first syllable
    CHECK(has(lock, "\xcb\x88o")); // ˈo — stressed second
}

// ── punctuation, the consumer's convention ──────────────────────────────────

TEST_CASE("Russian text: punctuation is carried only when the consumer wants it", "[g2p_ru][text]") {
    g2p_ru::context ctx;
    ctx.emit_punctuation = true;
    CHECK(has(g2p_ru::text_to_ipa(ctx, "Привет, мир."), ","));
    ctx.emit_punctuation = false;
    CHECK_FALSE(has(g2p_ru::text_to_ipa(ctx, "Привет, мир."), ","));
}

// ── THE INVENTORY INVARIANT ─────────────────────────────────────────────────

TEST_CASE("the inventory checker can actually fail", "[g2p_ru][inventory]") {
    // A checker that returns "clean" for everything certifies nothing. Prove it
    // fires BEFORE trusting it on real output — U+0060 is the exact codepoint
    // the upstream file uses for stress and the exact one that must never
    // survive into a phoneme string.
    CHECK(outside_inventory("`a").count(0x0060) == 1);
    CHECK(outside_inventory("t\xca\x83").size() == 1); // tʃ — English, not Russian
    CHECK(outside_inventory("123").size() == 3);
    CHECK(outside_inventory("sat").empty()); // and it does not cry wolf
}

TEST_CASE("Russian LTS emits nothing outside the dictionary's own inventory", "[g2p_ru][inventory]") {
    // Both paths must speak the same dialect, or an OOV word arrives in a
    // different accent from its neighbours — and the symbols that differ are
    // dropped rather than mispronounced, which is far harder to notice.
    static const char* const kWords[] = {
        "привет",
        "здравствуйте",
        "молоко",
        "хорошо",
        "синтеза",
        "речи",
        "отец",
        "сад",
        "год",
        "москва",
        "ёлка",
        "счастье",
        "сегодня",
        "яблоко",
        "жизнь",
        "цирк",
        "шесть",
        "все",
        "всё",
        "люди",
        "учится",
        "спиться",
        "жёлтый",
        "большого",
        "которого",
        "отвергнувшегося",
        "изживала",
        "отсюда",
        "объявление",
        "съешь",
        "мягких",
        "французских",
        "выпей",
        "чаю",
        "щи",
        "экзамен",
        "пять",
        "мяч",
        "говяжьей",
        "впечатляюще",
        "многообещающий",
        "тесанные",
        "отца",
        "отчество",
        "лучший",
        "театр",
    };
    for (const char* w : kWords) {
        const std::string ipa = lts(w);
        INFO("word=" << w << " ipa=" << ipa);
        CHECK_FALSE(ipa.empty());
        CHECK(outside_inventory(ipa).empty());
    }
}

TEST_CASE("espeak dialect: the conversion changes the spelling and not the inventory", "[g2p_ru][dialect]") {
    // A consumer trained on espeak's Russian spelling needs espeak's symbols.
    // Neither spelling loses anything to a TTS inventory — which is exactly why
    // this mismatch is invisible to a drop counter and needs its own test.
    g2p_ru::context ctx;
    const std::string native = g2p_ru::text_to_ipa(ctx, "молоко и хлеб лежат на столе");
    const std::string esp = g2p_ru::to_espeak_dialect(native);
    CHECK(native != esp); // the control: it must actually do something
    CHECK(outside_inventory(native).empty());
    CHECK(outside(esp, ru_espeak_inventory()).empty());
    // And the two sets really are different, so the check above is not the one
    // above it wearing a different name: the converted string must contain
    // symbols the NATIVE inventory does not have.
    CHECK_FALSE(outside_inventory(esp).empty());

    // Each rule, on its own, against the symbol espeak actually writes.
    CHECK(g2p_ru::to_espeak_dialect("\xc9\xab") == "\xc9\xad"); // ɫ -> ɭ
    CHECK(g2p_ru::to_espeak_dialect("l") == "\xc9\xad");        // l -> ɭ (espeak spells both the same)
    CHECK(g2p_ru::to_espeak_dialect("\xca\x82") == "\xca\x83"); // ʂ -> ʃ
    CHECK(g2p_ru::to_espeak_dialect("\xca\x90") == "\xca\x92"); // ʐ -> ʒ
    CHECK(g2p_ru::to_espeak_dialect("\xc9\xa8") == "y");        // ɨ -> y
    CHECK(g2p_ru::to_espeak_dialect("\xc9\x90") == "\xca\x8c"); // ɐ -> ʌ
    CHECK(g2p_ru::to_espeak_dialect("\xc9\x99") == "\xca\x8c"); // ə -> ʌ
    CHECK(g2p_ru::to_espeak_dialect("a") == "\xc9\x91");        // a -> ɑ
    CHECK(g2p_ru::to_espeak_dialect("\xc3\xa6") == "\xc9\x91"); // æ -> ɑ
    CHECK(g2p_ru::to_espeak_dialect("\xca\x8a") == "u");        // ʊ -> u
    CHECK(g2p_ru::to_espeak_dialect("\xca\x89") == "u");        // ʉ -> u

    // щ: ours ɕː, espeak ɕʲ.
    CHECK(g2p_ru::to_espeak_dialect("\xc9\x95\xcb\x90") == "\xc9\x95\xca\xb2");
    // A long consonant is written twice, not with a length mark.
    CHECK(g2p_ru::to_espeak_dialect("n\xcb\x90") == "nn");
    // ɪ survives word-finally and becomes plain i inside a word.
    CHECK(g2p_ru::to_espeak_dialect("r\xca\xb2"
                                    "e"
                                    "t\xc9\x95\xc9\xaa") == "r\xca\xb2"
                                                            "e"
                                                            "t\xc9\x95\xc9\xaa");
    CHECK(g2p_ru::to_espeak_dialect("\xc9\xaa"
                                    "n") == "in");
    // ...and a mark at a WORD boundary inside a phrase is still word-final.
    CHECK(g2p_ru::to_espeak_dialect("\xc9\xaa \xc9\xaa"
                                    "n") == "\xc9\xaa in");

    // Idempotence is NOT claimed and must not be assumed: the conversion maps
    // several source symbols onto one target, so it is lossy by construction.
    // Running it twice is a bug waiting to happen, and this records that the
    // second pass is a no-op only because the targets are outside the map.
    CHECK(g2p_ru::to_espeak_dialect(esp) == esp);
}

TEST_CASE("Russian number words also stay inside the inventory", "[g2p_ru][inventory]") {
    // The numerals go through the same rules, but they are text this file
    // GENERATES rather than text a user typed — a typo in the numeral table
    // could introduce a letter the rules have no case for, and it would vanish
    // instead of erroring.
    g2p_ru::context ctx;
    for (int64_t n :
         {0, 1, 2, 4, 9, 11, 14, 21, 40, 90, 100, 200, 500, 1000, 2000, 5000, 11000, 21000, 1000000, 2024, 999999999}) {
        const std::string ipa = g2p_ru::text_to_ipa(ctx, core_num2words_ru::cardinal(n));
        INFO("n=" << n << " words=" << core_num2words_ru::cardinal(n) << " ipa=" << ipa);
        CHECK_FALSE(ipa.empty());
        CHECK(outside_inventory(ipa).empty());
    }
}
