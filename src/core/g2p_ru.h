// core/g2p_ru.h — Russian grapheme-to-phoneme (text → IPA).
//
// WHY THIS EXISTS: zonos needed espeak-ng (GPL-3.0) for Russian and for nothing
// else once en/de/fr moved to the built-ins. This is the same two-tier shape as
// g2p_de/fr/es — a pronunciation dictionary in front of letter-to-sound rules —
// with the tiers doing rather different amounts of work than they do there,
// for reasons that are properties of the DATA and are spelled out below.
//
// ── TIER 1: the dictionary ──────────────────────────────────────────────────
// 812,953 entries from bene-ges/ru_g2p_ipa_bert_large (CC-BY-4.0), hosted as
// `ru_g2p_ipa.tsv` on cstr/g2p-dicts. Lexical stress is already resolved in it,
// which is the hard half of Russian G2P: stress is not predictable from
// spelling and it DRIVES vowel reduction, so getting it wrong changes the
// vowels, not just the prosody.
//
// Two facts about that file that a lookup must handle or it silently misfires:
//
//   1. THE KEYS CONTAIN NO `ё`. It is folded to `е` throughout (`ёлка` is
//      stored as `елка`, with its IPA `/ˈjɵɫkə/` still carrying the `ɵ`). So
//      every lookup folds `ё`→`е` first. Without that fold, every `ё` word in
//      the input misses the dictionary and falls to the rules — a 100% miss on
//      a letter that only ever appears in stressed syllables.
//   2. MONOSYLLABLES CARRY NO STRESS MARK (`сад` → `/sat/`). espeak-ng does the
//      same, so the two agree here rather than differing by one symbol on every
//      short word.
//
// ── TIER 2: the rules, and why they carry more than usual ───────────────────
// The upstream project shipped a SECOND file, `heteronyms.txt` (17,359 words),
// and the relationship between the two is the single most important thing to
// know about this backend:
//
//      THE HETERONYM LIST AND THE VOCABULARY ARE DISJOINT.
//      Measured: 0 of the 17,359 heteronyms appear in the vocabulary.
//
// They are not entries carrying a chosen reading. They are the words upstream
// REMOVED because it could not choose. So "on a heteronym, take the
// dictionary's reading" is not an option that exists — there is no reading to
// take, and every heteronym is an OOV word by construction. The list is
// therefore carried here as DIAGNOSTIC data (see `heteronym_list` and
// `warn_heteronyms`), not as a lookup table.
//
// That set is not exotic vocabulary either. `все`, `уже`, `потом`, `чем`,
// `небо`, `дорога`, `стоит`, `тест` are all in it. A large part of the reason
// is mechanical: because the keys fold `ё`→`е`, every `ё`/`е` minimal pair
// (`всё`/`все`, `нёбо`/`небо`) collapses into one ambiguous key and gets
// dropped. But the ambiguity is only in the DICTIONARY's keys: when the input
// text writes `ё`, it says which word it means, and the rules below use that —
// which recovers a good share of that set for free. An explicit combining
// acute (`замо́к`) does the same for the rest.
//
// A DICTIONARY CANNOT CARRY SENTENCE CONTEXT. For a genuine heteronym
// (`замок` = castle or lock, `мука` = flour or torment) the rules below pick
// ONE reading from spelling alone and will be wrong roughly half the time on
// those words. No lookup table fixes that; it needs a model that sees the
// sentence. This is a stated limitation of this backend, not a bug to file.
//
// ── The rules themselves ────────────────────────────────────────────────────
// Russian orthography is deep in a very specific way: the letters say what the
// morpheme is, and four systematic processes say what is actually pronounced.
// All four are implemented here because each of them changes phonemes, not
// just quality:
//
//   1. PALATALISATION. A consonant before `е ё и ю я ь` is soft (`ʲ`). `ж ш ц`
//      never are; `ч щ й` always are and take no mark. Dentals additionally
//      soften before a following SOFT dental — `жизнь` is `ʐɨzʲnʲ`, with the
//      `з` softened by the `нь` after it, not by anything to its right that is
//      written.
//   2. FINAL DEVOICING and REGRESSIVE VOICING ASSIMILATION. `сад`→`sat`,
//      `любовь`→`lʲʊbˈofʲ`, `все`→`fsʲe` (the `в` devoices before `с`), and the
//      other direction too: `экзамен` has `к`→`ɡ`. `в` is the classic
//      exception — it assimilates but does not TRIGGER voicing.
//   3. VOWEL REDUCTION, driven by the stress position: `о`/`а` are `ɐ` in the
//      syllable immediately before the stress and `ə` everywhere else
//      unstressed (`молоко` → `məɫɐkˈo` — the same letter, three different
//      vowels, chosen by distance from the stress). This is why a wrong stress
//      guess is not a prosody error but a segmental one.
//   4. `л` is dark `ɫ` when hard and `lʲ` when soft — two different sounds from
//      one letter, and both are in every inventory we target.
//
// ── Guessing the stress for an OOV word ─────────────────────────────────────
// In descending order of how much it can be trusted:
//   a. `ё` is ALWAYS stressed. Free, exact, and it is the input's own signal.
//   b. One vowel ⇒ that vowel, and no mark is emitted (matching the dictionary
//      and espeak).
//   c. ANALOGY against the dictionary: strip up to 3 trailing letters, or
//      append a short list of inflectional endings, and if the result is a
//      known word take ITS stressed-vowel index. `тест` is absent but `тесты`
//      is present, so `тест` gets its stress from `тесты`.
//   d. Last resort, a constant by syllable count, measured over all 812,953
//      entries rather than assumed (see kStressBySyllables). It is right 60.6%
//      of the time on 2-syllable words and 27-53% higher up — a coin-flip
//      dressed as a rule, which is exactly why it is last and why (c) exists.
//
// MEASURED, by holding 10,000 random dictionary words OUT of the dictionary and
// asking the rules for each of them (tools have no business claiming a number
// they did not compute):
//
//                          analogy ON     analogy OFF (control)
//   stressed index right      93.9%            47.1%
//   exact IPA match           79.4%            40.9%
//   symbol accuracy           96.6%            82.3%
//
// The control arm is the point: with tier (c) disabled the same code drops to a
// coin flip, so the 93.9% is the analogy working and not the rules being
// flattered by an easy sample. What that sample CANNOT show is the genuinely
// unrelated word — a surname, a neologism — for which (d) is all there is.
//
// Everything this file emits is inside the 37-codepoint set the dictionary
// itself uses, so the rules and the dictionary speak the same dialect and an
// OOV word does not arrive in a different accent from its neighbours.
// tests/test-g2p-ru.cpp asserts that mechanically, over both paths.
//
// Worth knowing, because it is the reverse of the assumption that the GPL
// dependency is the safe option: espeak-ng's own `ru` voice does NOT have this
// property. Over the same 2,200 words it emits `^` (U+005E) five times, and
// U+005E is in no TTS phoneme inventory we target — zonos drops it without a
// word. Measured drop against zonos's symbol list: espeak 0.0172%, this file
// 0.0000%. A tiny number, but it is the built-in that can be proven to stay
// inside the inventory and espeak that cannot.

#pragma once

#include "core/num2words_ru.h" // digits are in no dictionary and no rule
#include "core/g2p_alnum.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace g2p_ru {

// ── UTF-8 <-> codepoints ────────────────────────────────────────────────────

inline std::vector<uint32_t> utf8_to_cps(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp;
        size_t len;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else {
            cp = c & 0x07;
            len = 4;
        }
        if (i + len > s.size())
            break;
        for (size_t k = 1; k < len; k++)
            cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        out.push_back(cp);
        i += len;
    }
    return out;
}

inline void append_cp(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
}

inline std::string cps_to_utf8(const std::vector<uint32_t>& cps) {
    std::string s;
    for (uint32_t cp : cps)
        append_cp(s, cp);
    return s;
}

// ── Cyrillic letter classification ──────────────────────────────────────────

enum : uint32_t {
    RU_A = 0x430,    // а
    RU_YO = 0x451,   // ё
    RU_YA = 0x44F,   // я
    RU_E = 0x435,    // е
    RU_I = 0x438,    // и
    RU_O = 0x43E,    // о
    RU_U = 0x443,    // у
    RU_Y = 0x44B,    // ы
    RU_EH = 0x44D,   // э
    RU_YU = 0x44E,   // ю
    RU_SOFT = 0x44C, // ь
    RU_HARD = 0x44A, // ъ
    RU_ZH = 0x436,   // ж
    RU_SH = 0x448,   // ш
    RU_TS = 0x446,   // ц
    RU_CH = 0x447,   // ч
    RU_SHCH = 0x449, // щ
    RU_J = 0x439,    // й
    RU_L = 0x43B,    // л
    RU_G = 0x433,    // г
    RU_T = 0x442,    // т
    RU_S = 0x441,    // с
    RU_V = 0x432,    // в
};

inline bool is_cyrillic(uint32_t c) {
    return (c >= 0x410 && c <= 0x44F) || c == 0x401 || c == 0x451;
}

inline uint32_t to_lower_ru(uint32_t c) {
    if (c == 0x401)
        return 0x451; // Ё -> ё
    if (c >= 0x410 && c <= 0x42F)
        return c + 0x20;
    if (c >= 'A' && c <= 'Z')
        return c + 32;
    return c;
}

inline bool is_vowel(uint32_t c) {
    switch (c) {
    case RU_A:
    case RU_E:
    case RU_YO:
    case RU_I:
    case RU_O:
    case RU_U:
    case RU_Y:
    case RU_EH:
    case RU_YU:
    case RU_YA:
        return true;
    default:
        return false;
    }
}

// е ё ю я: the four that both soften a preceding consonant AND grow a /j/ when
// nothing precedes them.
inline bool is_iotating(uint32_t c) {
    return c == RU_E || c == RU_YO || c == RU_YU || c == RU_YA;
}

// The letters that make a preceding consonant soft.
inline bool softens_previous(uint32_t c) {
    return is_iotating(c) || c == RU_I || c == RU_SOFT;
}

inline bool always_hard(uint32_t c) {
    return c == RU_ZH || c == RU_SH || c == RU_TS;
}

inline bool always_soft(uint32_t c) {
    return c == RU_CH || c == RU_SHCH || c == RU_J;
}

inline bool is_consonant(uint32_t c) {
    return is_cyrillic(c) && !is_vowel(c) && c != RU_SOFT && c != RU_HARD;
}

inline bool is_dental(uint32_t c) {
    return c == 0x434 /*д*/ || c == RU_T || c == RU_S || c == 0x437 /*з*/ || c == 0x43D /*н*/ || c == RU_L;
}

// ── dictionary ──────────────────────────────────────────────────────────────

struct dictionary {
    std::map<std::string, std::string> entries;
    bool loaded = false;
};

// Same TSV shape the de/fr/es dicts use: `word<TAB>/ipa/`, '#' comments, a
// header line that parses harmlessly as one junk entry.
//
// COST, measured rather than guessed: 812,953 entries in a std::map cost ~137 MB
// of RSS once loaded. That is the same order as the 667K-entry German dict this
// mirrors, and it is paid ONCE, lazily, on the first Russian phonemisation —
// but it is real, and a caller on a memory-tight device should know it before
// it surprises them. CRISPASR_RU_DICT_PATH can point at a smaller subset.
inline int load_ipa_dict_file(dictionary& dict, const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f)
        return 0;
    char line[1024];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = 0;
        char* tab = strchr(line, '\t');
        if (!tab)
            continue;
        *tab = 0;
        std::string word = line;
        if (word.empty() || dict.entries.count(word))
            continue;
        char* ipa_start = tab + 1;
        while (*ipa_start == '/' || *ipa_start == ' ')
            ipa_start++;
        std::string ipa;
        for (char* p = ipa_start; *p && *p != '/' && *p != ','; p++)
            ipa += *p;
        while (!ipa.empty() && (ipa.back() == ' ' || ipa.back() == '/'))
            ipa.pop_back();
        if (!ipa.empty()) {
            dict.entries[word] = ipa;
            count++;
        }
    }
    fclose(f);
    dict.loaded = count > 0;
    return count;
}

// The heteronym list. NOT a lookup table — see the header comment. It exists so
// the ambiguity can be REPORTED instead of being invisible.
struct heteronym_list {
    std::set<std::string> words;
    bool loaded = false;
};

inline int load_heteronyms_file(heteronym_list& het, const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f)
        return 0;
    char line[512];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' '))
            line[--len] = 0;
        if (len == 0 || line[0] == '#')
            continue;
        het.words.insert(std::string(line, len));
        count++;
    }
    fclose(f);
    het.loaded = count > 0;
    return count;
}

// ── stress ──────────────────────────────────────────────────────────────────

inline const char* const kStressMark = "\xcb\x88"; // ˈ U+02C8

// The IPA vowel set the dictionary uses. Needed to read a stress POSITION back
// out of a dictionary entry, which is what the analogy tier does.
inline bool is_ipa_vowel_cp(uint32_t c) {
    switch (c) {
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
    case 0x00E6: // æ
    case 0x0250: // ɐ
    case 0x0259: // ə
    case 0x025B: // ɛ
    case 0x0268: // ɨ
    case 0x026A: // ɪ
    case 0x0275: // ɵ
    case 0x0289: // ʉ
    case 0x028A: // ʊ
        return true;
    default:
        return false;
    }
}

// Index (0-based, among IPA vowels) of the vowel the stress mark precedes.
// -1 when the entry carries no mark, which for this dictionary means a
// monosyllable.
inline int stressed_vowel_index(const std::string& ipa) {
    const std::vector<uint32_t> cps = utf8_to_cps(ipa);
    bool seen_mark = false;
    int vi = 0;
    for (uint32_t c : cps) {
        if (c == 0x02C8) {
            seen_mark = true;
            continue;
        }
        if (is_ipa_vowel_cp(c)) {
            if (seen_mark)
                return vi;
            vi++;
        }
    }
    return -1;
}

// Measured over all 812,953 dictionary entries: the single most frequent
// stressed-vowel index for each syllable count, and how often it is right.
//
//   2 syl -> 0  (60.6%)    5 syl -> 2  (50.4%)     8 syl -> 3  (30.6%)
//   3 syl -> 1  (53.3%)    6 syl -> 2  (45.4%)     9 syl -> 4  (27.1%)
//   4 syl -> 2  (42.2%)    7 syl -> 2  (30.4%)    10 syl -> 5  (39.8%)
//
// Reported here because a number that says "wrong 40-70% of the time" is the
// honest label for this tier, and it stops anyone reading a rule-path result as
// if it were dictionary quality.
inline int kStressBySyllables(int nvowels) {
    static const int kTable[11] = {0, 0, 0, 1, 2, 2, 2, 2, 3, 4, 5};
    if (nvowels <= 0)
        return 0;
    if (nvowels <= 10)
        return kTable[nvowels];
    return nvowels - 5;
}

// ── context ─────────────────────────────────────────────────────────────────

struct context {
    dictionary dict;
    heteronym_list het;

    // Carry `,.;:!?` through into the phoneme string. Same split as the other
    // languages: a phoneme-conditioned TTS model has them in its symbol table
    // and uses them to pause; piper's espeak inventory has never been fed them.
    bool emit_punctuation = false;

    // Print one line per input word that the upstream project flagged as
    // genuinely ambiguous. Off by default (it is noise in normal use), on when
    // someone is diagnosing a mispronunciation — which is exactly when "this
    // word HAS no single right answer" is the useful thing to learn.
    bool warn_heteronyms = false;

    // Analogy tier (c). On by default; an escape hatch because it is the one
    // tier that can pull a stress from a word that merely LOOKS related.
    bool stress_analogy = true;
};

// ── letter-to-sound ─────────────────────────────────────────────────────────

namespace detail {

// Voicing pairs, by letter. `.first` = voiced form, `.second` = voiceless.
inline bool voiced_obstruent(uint32_t c) {
    return c == 0x431 /*б*/ || c == RU_V || c == RU_G || c == 0x434 /*д*/ || c == RU_ZH || c == 0x437 /*з*/;
}

inline bool voiceless_obstruent(uint32_t c) {
    return c == 0x43F /*п*/ || c == 0x444 /*ф*/ || c == 0x43A /*к*/ || c == RU_T || c == RU_SH || c == RU_S ||
           c == 0x445 /*х*/ || c == RU_TS || c == RU_CH || c == RU_SHCH;
}

inline bool is_obstruent(uint32_t c) {
    return voiced_obstruent(c) || voiceless_obstruent(c);
}

// The voiced counterpart of a voiceless letter, and vice versa. Unpaired
// letters map to themselves.
inline uint32_t voiced_pair(uint32_t c) {
    switch (c) {
    case 0x43F:
        return 0x431; // п -> б
    case 0x444:
        return RU_V; // ф -> в
    case 0x43A:
        return RU_G; // к -> г
    case RU_T:
        return 0x434; // т -> д
    case RU_SH:
        return RU_ZH; // ш -> ж
    case RU_S:
        return 0x437; // с -> з
    default:
        return c;
    }
}

inline uint32_t devoiced_pair(uint32_t c) {
    switch (c) {
    case 0x431:
        return 0x43F; // б -> п
    case RU_V:
        return 0x444; // в -> ф
    case RU_G:
        return 0x43A; // г -> к
    case 0x434:
        return RU_T; // д -> т
    case RU_ZH:
        return RU_SH; // ж -> ш
    case 0x437:
        return RU_S; // з -> с
    default:
        return c;
    }
}

// Hard base IPA for each consonant letter, AFTER voicing has been resolved.
inline const char* consonant_ipa(uint32_t c, bool soft) {
    switch (c) {
    case 0x431:
        return "b";
    case RU_V:
        return "v";
    case RU_G:
        return "\xc9\xa1"; // ɡ
    case 0x434:
        return "d";
    case RU_ZH:
        return "\xca\x90"; // ʐ — never soft
    case 0x437:
        return "z";
    case RU_J:
        return "j"; // inherently soft, no ʲ
    case 0x43A:
        return "k";
    case RU_L:
        return soft ? "l" : "\xc9\xab"; // lʲ vs dark ɫ
    case 0x43C:
        return "m";
    case 0x43D:
        return "n";
    case 0x43F:
        return "p";
    case 0x440:
        return "r";
    case RU_S:
        return "s";
    case RU_T:
        return "t";
    case 0x444:
        return "f";
    case 0x445:
        return "x";
    case RU_TS:
        return "ts"; // never soft
    case RU_CH:
        return "t\xc9\x95"; // tɕ — inherently soft, no ʲ
    case RU_SH:
        return "\xca\x82"; // ʂ — never soft
    case RU_SHCH:
        return "\xc9\x95\xcb\x90"; // ɕː — inherently soft, no ʲ
    default:
        return "";
    }
}

// A consonant that carries its softness in the symbol itself takes no ʲ.
inline bool takes_palatal_mark(uint32_t c) {
    return !always_soft(c) && !always_hard(c);
}

// Latin letters inside Russian text. Not a transliteration scheme — just a
// reading, so a stray "Wi-Fi" or "CrispASR" produces SOMETHING rather than
// being dropped without a word, which is the failure mode this whole file
// exists to stop.
inline const char* latin_ipa(uint32_t c) {
    switch (c) {
    case 'a':
        return "a";
    case 'b':
        return "b";
    case 'c':
        return "k";
    case 'd':
        return "d";
    case 'e':
        return "e";
    case 'f':
        return "f";
    case 'g':
        return "\xc9\xa1";
    case 'h':
        return "x";
    case 'i':
        return "i";
    case 'j':
        return "j";
    case 'k':
        return "k";
    case 'l':
        return "l";
    case 'm':
        return "m";
    case 'n':
        return "n";
    case 'o':
        return "o";
    case 'p':
        return "p";
    case 'q':
        return "k";
    case 'r':
        return "r";
    case 's':
        return "s";
    case 't':
        return "t";
    case 'u':
        return "u";
    case 'v':
        return "v";
    case 'w':
        return "v";
    case 'x':
        return "ks";
    case 'y':
        return "i";
    case 'z':
        return "z";
    default:
        return "";
    }
}

} // namespace detail

// Normalise a token to lowercase Cyrillic codepoints, keeping ё, and strip the
// COMBINING ACUTE (U+0301) — the explicit stress mark Russian reference text,
// dictionaries and learner material use (`Росси́я`). Two things depend on
// removing it:
//
//   * the dictionary key. The vocabulary is written without it, so a word
//     carrying one misses every entry and drops to the rules — and Russian
//     Wikipedia puts one on the lede spelling of nearly every article.
//   * the phoneme string. U+0301 is in no TTS inventory, so it would be
//     dropped downstream WITHOUT a word, which is the failure this file exists
//     to stop.
//
// `explicit_stress` (optional) receives the 0-based index, among the word's
// vowels, of the vowel the mark sat on — free, exact stress information that
// the writer put there deliberately, and the caller uses it in preference to
// any guess.
inline std::vector<uint32_t> normalize_word(const std::string& word, int* explicit_stress = nullptr) {
    if (explicit_stress)
        *explicit_stress = -1;
    std::vector<uint32_t> out;
    int vowels = 0;
    for (uint32_t raw : utf8_to_cps(word)) {
        const uint32_t c = to_lower_ru(raw);
        // Invisible characters that break a lookup while showing nothing:
        // SOFT HYPHEN (U+00AD) is scattered through Wikipedia text, and the
        // zero-width family arrives from copy-paste. They are not stress marks,
        // so they are dropped without touching the vowel count.
        if (c == 0x00AD || c == 0x200B || c == 0x200C || c == 0x200D || c == 0xFEFF)
            continue;
        if (c == 0x0301 || c == 0x0300 || c == 0x0341) {
            // The mark follows the vowel it marks, so the vowel just counted is
            // the stressed one.
            if (explicit_stress && vowels > 0)
                *explicit_stress = vowels - 1;
            continue;
        }
        if (is_vowel(c))
            vowels++;
        out.push_back(c);
    }
    return out;
}

// The dictionary key for a word: ё folded to е, since the upstream vocabulary
// has no ё in its keys at all.
inline std::string dict_key(const std::vector<uint32_t>& cps) {
    std::vector<uint32_t> k = cps;
    for (uint32_t& c : k)
        if (c == RU_YO)
            c = RU_E;
    return cps_to_utf8(k);
}

// Letter-to-sound, given a stress position already decided by the caller.
// `stress_vi` is a 0-based index among the word's vowel LETTERS, or -1 for
// "no stress mark" (a monosyllable, matching the dictionary and espeak).
inline std::string lts_word_to_ipa(const std::vector<uint32_t>& in, int stress_vi) {
    std::vector<uint32_t> w = in;
    const size_t n0 = w.size();

    // -тся / -ться → [tsə]. A morphological spelling the letter rules cannot
    // reach: the т and the с fuse, and the я is a plain reduced vowel. Rewrite
    // to ц+а, which the rules below then handle correctly and which keeps the
    // vowel COUNT unchanged so the stress index stays valid.
    bool long_final_ts = false;
    if (n0 >= 4 && w[n0 - 4] == RU_T && w[n0 - 3] == RU_SOFT && w[n0 - 2] == RU_S && w[n0 - 1] == RU_YA) {
        w.resize(n0 - 4);
        w.push_back(RU_TS);
        w.push_back(RU_A);
        long_final_ts = true; // -ться is [tsːə], the infinitive's т and с fuse LONG
    } else if (n0 >= 3 && w[n0 - 3] == RU_T && w[n0 - 2] == RU_S && w[n0 - 1] == RU_YA) {
        w.resize(n0 - 3);
        w.push_back(RU_TS);
        w.push_back(RU_A);
    }

    const size_t n = w.size();

    // -ого / -его: the genitive ending where `г` is pronounced [v]. Not an
    // exception list — it is a productive ending on every adjective and
    // pronoun in the language.
    // The reflexive -ся/-сь sits AFTER the ending, so `отвергнувшегося` is
    // ɐtvʲˈerɡnʊfʂɨvəsʲə — the same [v], two letters further from the end.
    size_t tail = n;
    if (tail >= 2 && w[tail - 2] == RU_S && w[tail - 1] == RU_YA)
        tail -= 2;
    else if (tail >= 2 && w[tail - 2] == RU_S && w[tail - 1] == RU_SOFT)
        tail -= 2;
    size_t g_as_v = (size_t)-1;
    if (tail >= 4 && w[tail - 1] == RU_O && w[tail - 2] == RU_G && (w[tail - 3] == RU_O || w[tail - 3] == RU_E))
        g_as_v = tail - 2;

    // Vowel positions, and the stressed one.
    std::vector<size_t> vpos;
    for (size_t i = 0; i < n; i++)
        if (is_vowel(w[i]))
            vpos.push_back(i);
    const int nv = (int)vpos.size();
    const size_t stress_at = (stress_vi >= 0 && stress_vi < nv) ? vpos[stress_vi] : (size_t)-1;

    // Pass 1: softness. A consonant is soft if the next letter softens it,
    // never if it is ж/ш/ц, always if it is ч/щ/й.
    std::vector<bool> soft(n, false);
    for (size_t i = 0; i < n; i++) {
        if (!is_consonant(w[i]))
            continue;
        if (always_soft(w[i])) {
            soft[i] = true;
            continue;
        }
        if (always_hard(w[i]))
            continue;
        if (i + 1 < n && softens_previous(w[i + 1]))
            soft[i] = true;
    }
    // Pass 1b: regressive softening of dentals before a soft dental. `жизнь` is
    // ʐɨzʲnʲ — the з is softened by the нь to its right, and nothing in the
    // spelling of the з itself says so.
    for (size_t i = n; i-- > 0;) {
        if (!is_consonant(w[i]) || !is_dental(w[i]) || soft[i] || always_hard(w[i]))
            continue;
        // The TRIGGER set is narrower than the target set. `от-` before a soft
        // с stays hard (`отсюда` ɐtsʲˈudə, not ɐtʲsʲ…), while `сегодня`
        // sʲɪvˈodʲnʲə and `жизнь` ʐɨzʲnʲ do soften — so a soft с/з does not
        // pull its neighbour with it, and a soft т/д/н/л does.
        const uint32_t nxt = i + 1 < n ? w[i + 1] : 0;
        const bool trigger = nxt == RU_T || nxt == 0x434 /*д*/ || nxt == 0x43D /*н*/ || nxt == RU_L;
        if (trigger && is_consonant(nxt) && soft[i + 1])
            soft[i] = true;
    }

    // Pass 2: voicing. Right to left, because the assimilation is regressive.
    std::vector<uint32_t> voiced = w;
    for (size_t i = n; i-- > 0;) {
        if (!detail::is_obstruent(w[i]))
            continue;
        // Look right past ь/ъ for the next consonant.
        size_t j = i + 1;
        while (j < n && (w[j] == RU_SOFT || w[j] == RU_HARD))
            j++;
        if (j >= n) {
            voiced[i] = detail::devoiced_pair(voiced[i]); // word-final devoicing
            continue;
        }
        if (!detail::is_obstruent(w[j]))
            continue; // a vowel or sonorant to the right: no assimilation
        // Use the ALREADY-RESOLVED voicing of the neighbour, so a chain
        // devoices as a unit rather than one link at a time.
        if (detail::voiceless_obstruent(voiced[j])) {
            voiced[i] = detail::devoiced_pair(voiced[i]);
        } else if (w[j] != RU_V) {
            // в assimilates but does not TRIGGER voicing — `свой` is [svoj],
            // not [zvoj].
            voiced[i] = detail::voiced_pair(voiced[i]);
        }
    }

    // Pass 3: emit.
    std::string ipa;
    int vi = 0;
    for (size_t i = 0; i < n; i++) {
        const uint32_t c = w[i];

        if (c == RU_SOFT || c == RU_HARD)
            continue; // softness already applied; the /j/ is emitted at the vowel

        if (is_vowel(c)) {
            const bool stressed = (i == stress_at) || (nv == 1 && stress_vi < 0);
            const bool mark = (i == stress_at) && nv > 1;
            // /j/ onset: an iotating vowel with no consonant immediately
            // before it, or one separated from it by ь/ъ. `и` gets one too
            // after ь (`соловьи`).
            const bool after_sign = i > 0 && (w[i - 1] == RU_SOFT || w[i - 1] == RU_HARD);
            const bool bare = i == 0 || is_vowel(w[i - 1]);
            const bool j_onset = (is_iotating(c) && (bare || after_sign)) || (c == RU_I && after_sign);
            // Which consonant, if any, colours this vowel.
            const bool prev_hard = i > 0 && is_consonant(w[i - 1]) && always_hard(w[i - 1]);
            const bool prev_soft = i > 0 && is_consonant(w[i - 1]) && soft[i - 1];
            const bool soft_ctx = j_onset || prev_soft;
            // FRONTING is conditioned on the consonant AFTER the vowel, not the
            // one before it. Measured over the whole dictionary, stressed `я`
            // is [æ] before a soft consonant 100.0% of the time and [a]
            // otherwise 97.2%; stressed `ю` is [ʉ] before a soft consonant
            // 99.9% and [u] otherwise 84.3%. Conditioning on the PRECEDING
            // consonant instead (the intuitive reading of "after a soft
            // consonant") splits both roughly 50/50 and is therefore no rule at
            // all — `пять` is pʲætʲ but `говяжьей` is ɡɐvʲˈaʐjɪj, and what
            // separates them is entirely to the right of the vowel.
            // A following iotating vowel brings its own /j/, and /j/ is a soft
            // consonant — `впечатляюще` is fʲpʲɪtɕɪtlʲˈæjʉɕːe, fronted by the
            // ю that follows, not by anything spelled as a consonant.
            const bool next_soft = (i + 1 < n && is_consonant(w[i + 1]) && soft[i + 1]) ||
                                   (i + 1 < n && is_vowel(w[i + 1]) && is_iotating(w[i + 1]));
            // For the vowel's own colouring a /j/ onset counts as a soft
            // consonant on the left, the same way it does on the right.
            const bool prev_soft_v = prev_soft || j_onset;
            // The preceding VOWEL letter, needed for one ending: word-final
            // unstressed `-ое` is [əjə] (79.5%) while `-ие`/`-ые`/`-ее` are
            // [je] (98-100%). Measured, not assumed — it is the neuter
            // adjective ending behaving differently from every other -е.
            uint32_t prev_vowel_letter = 0;
            for (size_t q = i; q-- > 0;)
                if (is_vowel(w[q])) {
                    prev_vowel_letter = w[q];
                    break;
                }
            const bool pretonic = stress_at != (size_t)-1 && vi + 1 == stress_vi;
            const bool final_vowel = (i + 1 == n);

            if (mark)
                ipa += kStressMark;
            if (j_onset)
                ipa += "j";

            const char* v = "";
            if (stressed) {
                switch (c) {
                case RU_A:
                case RU_YA:
                    // /a/ fronts to [æ] only BETWEEN two soft segments. One
                    // side is not enough and the near-50/50 split in the data
                    // is what says so: `пять` pʲætʲ (soft-soft) against
                    // `говяжьей` ɡɐvʲˈaʐjɪj (soft-HARD) — both spelled `я`,
                    // both after a palatalized consonant.
                    v = (prev_soft_v && next_soft) ? "\xc3\xa6" : "a"; // æ / a
                    break;
                case RU_O:
                    v = "o";
                    break;
                case RU_YO:
                    v = prev_hard ? "o" : "\xc9\xb5"; // ɵ, but o after ж/ш
                    break;
                case RU_E:
                    v = prev_hard ? "\xc9\x9b" : "e"; // ɛ after ж/ш/ц, else e
                    break;
                case RU_EH:
                    v = "\xc9\x9b"; // ɛ
                    break;
                case RU_I:
                    v = prev_hard ? "\xc9\xa8" : "i"; // ɨ after ж/ш/ц
                    break;
                case RU_Y:
                    v = "\xc9\xa8"; // ɨ
                    break;
                case RU_U:
                    // Same fronting, same condition: `чудивший` tɕʉdʲˈifʂɨj.
                    v = (prev_soft_v && next_soft) ? "\xca\x89" : "u"; // ʉ / u
                    break;
                case RU_YU:
                    // ю already softens whatever is to its left, so only the
                    // right side is free to vary: [ʉ] before a soft segment
                    // (100.0%), [u] otherwise (84.2%).
                    v = next_soft ? "\xca\x89" : "u"; // ʉ / u
                    break;
                default:
                    break;
                }
            } else {
                switch (c) {
                case RU_A:
                case RU_O:
                    if (soft_ctx)
                        v = "\xc9\xaa"; // ɪ — `часы` tɕɪsɨ
                    else if (pretonic || i == 0)
                        v = "\xc9\x90"; // ɐ
                    else
                        v = "\xc9\x99"; // ə
                    break;
                case RU_YA:
                    // [ɪ] in the syllable before the stress, [ə] elsewhere
                    // (including word-final `-ня` → nʲə).
                    v = pretonic ? "\xc9\xaa" : "\xc9\x99";
                    break;
                case RU_E:
                    if (prev_hard)
                        v = "\xc9\xa8"; // ɨ after ж/ш/ц — `жена` ʐɨna
                    else if (final_vowel)
                        v = (prev_vowel_letter == RU_O) ? "\xc9\x99" : "e"; // -ое is [əjə]
                    else
                        v = "\xc9\xaa"; // ɪ
                    break;
                case RU_YO:
                    // ё is stressed by definition; reaching here means the
                    // caller overrode the stress, so keep its own quality.
                    v = prev_hard ? "o" : "\xc9\xb5";
                    break;
                case RU_EH:
                    v = "\xc9\xaa";
                    break;
                case RU_I:
                    v = prev_hard ? "\xc9\xa8" : "\xc9\xaa"; // ɨ / ɪ
                    break;
                case RU_Y:
                    v = "\xc9\xa8";
                    break;
                case RU_U:
                    v = (prev_soft_v && next_soft) ? "\xca\x89" : "\xca\x8a"; // ʉ / ʊ
                    break;
                case RU_YU:
                    v = next_soft ? "\xca\x89" : "\xca\x8a"; // ʉ / ʊ
                    break;
                default:
                    break;
                }
            }
            ipa += v;
            vi++;
            continue;
        }

        if (!is_consonant(c)) {
            // A Latin letter or a stray symbol inside a Cyrillic token.
            ipa += detail::latin_ipa(c);
            continue;
        }

        // т/д before ц or ч fuse into ONE long affricate: `отца` is ɐtːsˈa-class
        // [tsː], `отчество` is ɐtːɕ… — not a stop followed by an affricate.
        if ((c == RU_T || c == 0x434 /*д*/) && i + 1 < n && (w[i + 1] == RU_TS || w[i + 1] == RU_CH)) {
            ipa += w[i + 1] == RU_TS ? "ts\xcb\x90" : "t\xc9\x95\xcb\x90"; // tsː / tɕː
            i++;
            continue;
        }

        // зж / жж spell a long [ʐː] and сш / зш a long [ʂː] — `изживала` is
        // ɪʐːɨvˈaɫə, not ɪzʐɨ. Assimilation across the pair, not two segments.
        if ((c == 0x437 /*з*/ || c == RU_ZH) && i + 1 < n && w[i + 1] == RU_ZH) {
            ipa += "\xca\x90\xcb\x90"; // ʐː
            i++;
            continue;
        }
        if ((c == RU_S || c == 0x437 /*з*/) && i + 1 < n && w[i + 1] == RU_SH) {
            ipa += "\xca\x82\xcb\x90"; // ʂː
            i++;
            continue;
        }

        // сч / зч / жч spell a single long [ɕː] — `счастье` is ɕːˈæsʲtʲje,
        // not stɕ. A productive spelling (счёт, считать, счастье), not an
        // exception list.
        if ((c == RU_S || c == 0x437 /*з*/ || c == RU_ZH) && i + 1 < n && w[i + 1] == RU_CH) {
            ipa += "\xc9\x95\xcb\x90"; // ɕː
            i++;                       // consume the ч as well
            continue;
        }

        // A doubled letter is a LONG consonant, not two of them: `тесанные`
        // ends nːɨje.
        if (i > 0 && w[i - 1] == c) {
            ipa += "\xcb\x90"; // ː
            continue;
        }

        if (i == g_as_v) {
            ipa += "v";
            if (soft[i])
                ipa += "\xca\xb2";
            continue;
        }

        ipa += detail::consonant_ipa(voiced[i], soft[i]);
        if (c == RU_TS && long_final_ts && i + 2 == n)
            ipa += "\xcb\x90"; // ː — see long_final_ts above
        if (soft[i] && detail::takes_palatal_mark(c))
            ipa += "\xca\xb2"; // ʲ
    }
    return ipa;
}

// ── stress guessing for OOV words ───────────────────────────────────────────

namespace detail {

// Endings worth trying when the bare form is missing but an inflected one is
// present. `тест` is absent from the vocabulary; `тесты` is there.
inline const char* const kTryEndings[] = {
    "\xd1\x8b",         // ы
    "\xd0\xb0",         // а
    "\xd0\xb8",         // и
    "\xd1\x83",         // у
    "\xd0\xb5",         // е
    "\xd0\xbe\xd0\xbc", // ом
    "\xd0\xbe\xd0\xb2", // ов
    "\xd0\xb0\xd1\x8f", // ая
    "\xd1\x8b\xd0\xb9", // ый
};

} // namespace detail

// Number of vowel LETTERS in a normalised word.
inline int count_vowels(const std::vector<uint32_t>& w) {
    int n = 0;
    for (uint32_t c : w)
        if (is_vowel(c))
            n++;
    return n;
}

// Decide the stressed-vowel index for a word the dictionary does not have.
// The tiers and their trustworthiness are documented at the top of this file;
// `used_analogy` reports which one answered, so a caller (or a test) can tell a
// measured guess from a coin flip.
inline int guess_stress(const context& ctx, const std::vector<uint32_t>& w, bool* used_analogy = nullptr) {
    if (used_analogy)
        *used_analogy = false;
    const int nv = count_vowels(w);
    if (nv <= 1)
        return nv == 1 ? 0 : -1;

    // (a) ё is always stressed.
    int vi = 0;
    for (uint32_t c : w) {
        if (c == RU_YO)
            return vi;
        if (is_vowel(c))
            vi++;
    }

    // (c) analogy against the dictionary: strip up to 3 letters off the end,
    // then optionally glue one of a short list of inflectional endings back on,
    // and take the stress of whatever known word that lands on. The two moves
    // have to COMBINE, not just alternate — `тестом` is missing and `тесты` is
    // present, and no amount of truncating `тестом` or appending to it reaches
    // `тесты`; only `тест` + `ы` does.
    //
    // Nearest first (cut 0 before cut 1, bare stem before any added ending), so
    // the closest relative wins rather than whichever ending happens to be
    // listed first.
    if (ctx.stress_analogy && ctx.dict.loaded) {
        for (size_t cut = 0; cut <= 3 && w.size() > cut + 2; cut++) {
            const std::vector<uint32_t> stem(w.begin(), w.end() - cut);
            for (size_t e = 0; e <= sizeof(detail::kTryEndings) / sizeof(detail::kTryEndings[0]); e++) {
                if (cut == 0 && e == 0)
                    continue; // that is the word itself, already known missing
                std::vector<uint32_t> t = stem;
                if (e > 0)
                    for (uint32_t c : utf8_to_cps(detail::kTryEndings[e - 1]))
                        t.push_back(c);
                const int tnv = count_vowels(t);
                if (tnv < 1)
                    continue;
                auto it = ctx.dict.entries.find(dict_key(t));
                if (it == ctx.dict.entries.end())
                    continue;
                int sv = stressed_vowel_index(it->second);
                // A monosyllabic relative carries no mark in this dictionary,
                // and its one vowel IS the stressed one.
                if (sv < 0 && tnv == 1)
                    sv = 0;
                if (sv >= 0 && sv < nv) {
                    if (used_analogy)
                        *used_analogy = true;
                    return sv;
                }
            }
        }
    }

    // (d) the measured constant. Right 27-61% of the time; see kStressBySyllables.
    return kStressBySyllables(nv);
}

// ── word / text ─────────────────────────────────────────────────────────────

inline std::string word_to_ipa(const context& ctx, const std::string& word) {
    int explicit_stress = -1;
    const std::vector<uint32_t> w = normalize_word(word, &explicit_stress);
    if (w.empty())
        return "";

    bool any_cyrillic = false;
    for (uint32_t c : w)
        if (is_cyrillic(c))
            any_cyrillic = true;

    if (any_cyrillic && ctx.dict.loaded) {
        const std::string key = dict_key(w);
        auto it = ctx.dict.entries.find(key);
        if (it != ctx.dict.entries.end()) {
            // An explicit combining acute in the INPUT outranks the dictionary
            // when the two disagree, because it is the writer disambiguating on
            // purpose — `за́мок` (castle) against `замо́к` (lock). When they
            // agree, or when nothing was marked, the dictionary wins: it also
            // carries vowel quality, which the rules only approximate.
            if (explicit_stress < 0 || stressed_vowel_index(it->second) == explicit_stress || count_vowels(w) <= 1)
                return it->second;
        }
        if (ctx.warn_heteronyms && ctx.het.loaded && ctx.het.words.count(key)) {
            // The upstream project could not choose a reading for this word,
            // so neither can a lookup. Say so rather than let the rules pick
            // one silently.
            fprintf(stderr,
                    "g2p_ru: '%s' is a known heteronym — its pronunciation depends on sentence context, "
                    "which a dictionary cannot carry. Falling back to letter-to-sound rules, which pick ONE reading.\n",
                    key.c_str());
        }
    }

    if (!any_cyrillic) {
        // Pure Latin/other. Read it letter by letter rather than dropping it.
        std::string ipa;
        for (uint32_t c : w)
            ipa += detail::latin_ipa(c);
        return ipa;
    }

    return lts_word_to_ipa(w, explicit_stress >= 0 ? explicit_stress : guess_stress(ctx, w));
}

// A CODEPOINT-aware tokenizer, unlike the ASCII-byte one the other languages
// use. Russian prose is full of punctuation that lives above U+007F — the em
// dash in particular is the ordinary copula ("Россия — государство"), and an
// ASCII-only splitter leaves it glued to the word beside it. `россия —` is then
// one token, misses the dictionary, and drops to the rules for no reason at
// all. With these marks split (and with the soft hyphens and combining acutes
// that the same text is full of stripped in normalize_word), dictionary
// coverage over 1,073 word tokens of Russian Wikipedia summaries is 89.7%; the
// 10.3% that misses is dominated by unstressed function words (`на`, `не`,
// `из`, `во`) that are on the heteronym list and so were never in the
// vocabulary to begin with.
inline bool is_separator_cp(uint32_t c) {
    switch (c) {
    case ' ':
    case '\t':
    case '\n':
    case '\r':
    case 0x00A0: // no-break space
    case ',':
    case '.':
    case '!':
    case '?':
    case ';':
    case ':':
    case '-':
    case '"':
    case '\'':
    case '(':
    case ')':
    case '[':
    case ']':
    case 0x2010: // ‐ hyphen
    case 0x2011: // ‑ non-breaking hyphen
    case 0x2012: // ‒ figure dash
    case 0x2013: // – en dash
    case 0x2014: // — em dash
    case 0x2026: // … ellipsis
    case 0x00AB: // «
    case 0x00BB: // »
    case 0x201C: // “
    case 0x201D: // ”
    case 0x201E: // „
    case 0x2018: // ‘
    case 0x2019: // ’
        return true;
    default:
        return false;
    }
}

// Whitespace separates but is not itself a token.
inline bool is_space_cp(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x00A0;
}

// Which marks a phoneme-conditioned model actually has in its symbol table and
// uses to pause. Hyphens and apostrophes are separators only — they are word
// joiners in the orthography, not prosody.
inline bool is_emittable_punct_cp(uint32_t c) {
    switch (c) {
    case ',':
    case '.':
    case '!':
    case '?':
    case ';':
    case ':':
    case '"':
    case '(':
    case ')':
    case 0x2014: // —
    case 0x2026: // …
    case 0x00AB: // «
    case 0x00BB: // »
    case 0x201C: // “
    case 0x201D: // ”
        return true;
    default:
        return false;
    }
}

inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (uint32_t c : utf8_to_cps(text)) {
        if (!is_separator_cp(c)) {
            append_cp(cur, c);
            continue;
        }
        if (!cur.empty()) {
            tokens.push_back(cur);
            cur.clear();
        }
        if (!is_space_cp(c)) {
            std::string p;
            append_cp(p, c);
            tokens.push_back(p);
        }
    }
    if (!cur.empty())
        tokens.push_back(cur);
    return tokens;
}

// An OPENING mark leads the phrase that follows it; a closing mark trails the
// word before it. Attaching both to the word before produces `mʲirʲe,« stalʲitsə»`,
// where the guillemet is glued to the comma instead of to the word it opens.
inline bool is_opening_punct_cp(uint32_t c) {
    return c == '(' || c == 0x00AB /*«*/ || c == 0x201C /*“*/ || c == 0x201E /*„*/;
}

// True when the token is a single punctuation codepoint (so it is a mark, not a
// word). `emit` says whether the consumer's symbol table has it.
inline bool token_is_punct(const std::string& t, bool* emit) {
    const std::vector<uint32_t> cps = utf8_to_cps(t);
    if (cps.size() != 1 || !is_separator_cp(cps[0]))
        return false;
    *emit = is_emittable_punct_cp(cps[0]);
    return true;
}

inline std::string text_to_ipa(const context& ctx, const std::string& text) {
    // Digits first: they are in no dictionary and match no rule, and Zonos's
    // inventory has no digit symbols either, so an unexpanded number is dropped
    // twice over without a word. Must run before tokenize(), which splits on
    // ',' — the Russian decimal mark.
    const std::vector<std::string> words =
        tokenize(core_num2words_ru::expand(core_g2p_alnum::split_letters_digits(text)));
    std::string ipa;
    bool pending_space = false;
    for (const std::string& w : words) {
        bool emit_this = false;
        const bool punct = token_is_punct(w, &emit_this);
        if (punct && (!ctx.emit_punctuation || !emit_this)) {
            if (!ipa.empty())
                pending_space = true;
            continue;
        }
        if (punct) {
            const uint32_t cp = utf8_to_cps(w)[0];
            if (is_opening_punct_cp(cp)) {
                if (!ipa.empty() && ipa.back() != ' ')
                    ipa += ' ';
                ipa += w;
                pending_space = false; // the word it opens attaches to it
            } else {
                ipa += w; // a closing mark attaches to the word before it
                pending_space = true;
            }
            continue;
        }
        const std::string ph = word_to_ipa(ctx, w);
        if (ph.empty())
            continue;
        if (!ipa.empty() && (pending_space || ipa.back() != ' '))
            ipa += ' ';
        pending_space = false;
        ipa += ph;
    }
    return ipa;
}

// ── espeak's Russian spelling, for consumers trained on it ──────────────────
//
// This file's own transcription and espeak-ng's `ru` voice describe the SAME
// sounds in different symbols, and the difference is large: measured over 2,200
// dictionary words phonemised both ways, raw symbol agreement between them is
// 57.7% and not one word matches exactly.
//
//   ours   məɫɐkˈo i xlʲep lʲɪʐˈat na stɐlʲˈe v bɐlʲʂˈoj kˈomnətʲe
//   espeak mʌɭʌkˈo ɪ xɭʲˈep ɭʲiʒˈɑt nə stʌɭʲˈe v bʌɭʃˈoj kˈomnʌtʲi
//
// That matters because a TTS model conditions on the SPELLING it was trained
// on, not on the sounds. Zonos was phonemised with espeak, so feeding it this
// file's symbols hands it combinations it never saw — and because every symbol
// on both sides is inside zonos's inventory, the drop counter cannot see the
// mismatch at all. It is the #316 problem (Kokoro trained on misaki's spelling,
// 58% agreement from CMUdict) one language further on, and the fix has the same
// shape: the G2P keeps emitting its own, more accurate transcription, and the
// CONSUMER converts.
//
// The map is DERIVED, not assumed: each rule is the majority alignment over
// those 2,200 word pairs. Measured effect, with the unconverted arm as the
// control:
//
//                             symbol agreement    exact word match
//   unconverted (control)          57.7%                0.0%
//   converted                      88.0%               32.6%
//
// One tempting rule was measured and REJECTED: moving `ˈ` from before the
// stressed vowel to the syllable onset drops agreement to 75.1%. espeak's ru
// voice marks the vowel too, so the placement was already right.
//
// ── AND THEN THE AUDIO SAID NO ───────────────────────────────────────────────
//
// THIS CONVERSION IS OFF BY DEFAULT BECAUSE IT WAS MEASURED AND IT LOSES.
//
// It does exactly what it was built to do — phoneme-ID agreement with the
// espeak arm rose on all three test sentences, 0.773/0.759/0.627 → 0.818/0.852/
// 0.847 — and the ASR roundtrip through zonos went from 0.293 to **0.000**, on
// every sentence, in both the with-espeak and the espeak-removed runs. Six arms,
// six zeros. It is the only one of the three arms that never once produced a
// recognisable transcript; the ASR read its output as Polish and Dutch.
//
// So the reasoning this function was built on — "the model was trained on
// espeak's spelling, therefore matching that spelling will help" — is FALSE
// here, and it is worth stating why it was so persuasive: it is a mechanism
// story with a measured 30-point number attached, which is exactly the shape
// that gets shipped without a roundtrip. **Agreement with the tool a model was
// trained on is not a proxy for the quality of the audio**, and no amount of
// symbol-level evidence substitutes for listening to the output.
//
// It is kept, gated and off, for the same reason CRISPASR_G2P_DE_UNSTRESS and
// CRISPASR_KOKORO_DE_MISAKI_ALPHABET are kept: the lever is the evidence, and a
// different consumer (a piper or kokoro Russian voice trained on espeak) may
// yet want it. It must not be turned on for zonos without a new measurement.
//
// One further rule is measured, better on AGREEMENT, and deliberately not here:
// rewriting `tɕ` (ч) to espeak's `tʃʲ` takes agreement to 89.6% and exact
// matches to 36.1%. Given the result above, "better on agreement" is no longer
// a reason to add anything. Add it with the next run, not before it.
inline std::string to_espeak_dialect(const std::string& ipa) {
    const std::vector<uint32_t> cps = utf8_to_cps(ipa);
    std::vector<uint32_t> out;
    out.reserve(cps.size() + 8);
    for (size_t i = 0; i < cps.size(); i++) {
        const uint32_t c = cps[i];
        // щ: ours ɕː, espeak ɕʲ.
        if (c == 0x0255 && i + 1 < cps.size() && cps[i + 1] == 0x02D0) {
            out.push_back(0x0255);
            out.push_back(0x02B2);
            i++;
            continue;
        }
        // A long consonant is written twice rather than with a length mark.
        if (c == 0x02D0) {
            if (!out.empty())
                out.push_back(out.back());
            continue;
        }
        // ɪ survives only word-finally (`речи` rʲˈetʃʲɪ); elsewhere espeak
        // writes plain i.
        if (c == 0x026A) {
            const bool word_final = (i + 1 == cps.size()) || cps[i + 1] == ' ' || is_separator_cp(cps[i + 1]);
            out.push_back(word_final ? 0x026A : 'i');
            continue;
        }
        switch (c) {
        case 'a':
        case 0x00E6:               // æ
            out.push_back(0x0251); // ɑ
            break;
        case 0x0250:               // ɐ
        case 0x0259:               // ə
            out.push_back(0x028C); // ʌ
            break;
        case 0x0268: // ɨ
            out.push_back('y');
            break;
        case 0x028A: // ʊ
        case 0x0289: // ʉ
            out.push_back('u');
            break;
        case 0x0275: // ɵ
            out.push_back('e');
            break;
        case 0x0282:               // ʂ
            out.push_back(0x0283); // ʃ
            break;
        case 0x0290:               // ʐ
            out.push_back(0x0292); // ʒ
            break;
        case 0x026B: // ɫ
        case 'l':
            out.push_back(0x026D); // ɭ — espeak writes both hard and soft л this way
            break;
        default:
            out.push_back(c);
            break;
        }
    }
    return cps_to_utf8(out);
}

} // namespace g2p_ru
