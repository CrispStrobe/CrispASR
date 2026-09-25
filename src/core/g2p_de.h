// core/g2p_de.h — German grapheme-to-phoneme (text → IPA).
//
// Pipeline:
//   1. Dictionary lookup (787K IPA entries, CC-BY-SA, auto-downloadable)
//   2. Compound word splitting (greedy longest-match against dictionary)
//   3. Rule-based LTS with:
//      a. Open-syllable vowel lengthening (ö/ü/ä/a/e/i/o/u)
//      b. Auslautverhärtung (final devoicing: b→p, d→t, g→k, v→f, z→s)
//      c. Full digraph/trigraph rules (sch/ch/tsch/ei/eu/au/äu/sp/st/...)

#pragma once

#include "core/g2p_de_unstressed.h"
#include "core/num2words_de.h" // #316: spell digits out before phonemizing
#include "core/g2p_alnum.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace g2p_de {

// ── Dictionary ──────────────────────────────────────────────────────

struct dictionary {
    std::map<std::string, std::string> entries;
    bool loaded = false;
};

// Load open-dict-data format: "word\t/IPA1/, /IPA2/\n"
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
        for (auto& c : word)
            c = (char)tolower((unsigned char)c);
        if (dict.entries.count(word))
            continue;
        char* ipa_start = tab + 1;
        while (*ipa_start == '/' || *ipa_start == ' ')
            ipa_start++;
        std::string ipa;
        for (char* p = ipa_start; *p && *p != '/' && *p != ','; p++)
            ipa += *p;
        while (!ipa.empty() && (ipa.back() == ' ' || ipa.back() == '/'))
            ipa.pop_back();
        if (!ipa.empty() && !word.empty()) {
            dict.entries[word] = ipa;
            count++;
        }
    }
    fclose(f);
    dict.loaded = count > 0;
    return count;
}

// ── Helpers ─────────────────────────────────────────────────────────

// UTF-8 aware lowercase (handles ä/ö/ü/ß/Ä/Ö/Ü)
inline std::string to_lower_de(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 'A' && c <= 'Z') {
            out += (char)(c + 32);
            continue;
        }
        if (c == 0xC3 && i + 1 < s.size()) {
            unsigned char c2 = (unsigned char)s[i + 1];
            // Ä(C384)→ä(C3A4), Ö(C396)→ö(C3B6), Ü(C39C)→ü(C3BC)
            if (c2 >= 0x80 && c2 <= 0x9E) {
                out += (char)c;
                out += (char)(c2 + 0x20);
                i++;
                continue;
            }
        }
        out += s[i];
    }
    return out;
}

// Check if a byte position is at a vowel letter (ASCII only for syllable check)
inline bool is_vowel_ascii(char c) {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

// Count consonant PHONEME units after position i.
// Treats digraphs (ch, sch, ng, ck, pf, tz, th) as single units.
inline int count_following_consonant_units(const std::string& w, int i) {
    int n = 0;
    int j = i + 1;
    int wlen = (int)w.size();
    while (j < wlen) {
        char c = w[j];
        if ((unsigned char)c >= 0x80)
            break;
        if (is_vowel_ascii(c))
            break;
        if (c < 'a' || c > 'z')
            break;
        // Check for digraphs that form one phoneme
        char c1 = (j + 1 < wlen) ? w[j + 1] : 0;
        if ((c == 'c' && c1 == 'h') || (c == 'n' && c1 == 'g') || (c == 'c' && c1 == 'k') || (c == 'p' && c1 == 'f') ||
            (c == 't' && c1 == 'z') || (c == 't' && c1 == 'h') || (c == 's' && c1 == 'c')) {
            n++;
            j += 2;
            // sch is 3 chars
            if (c == 's' && c1 == 'c' && j < wlen && w[j] == 'h')
                j++;
            continue;
        }
        n++;
        j++;
    }
    return n;
}

// Check if vowel at position i is in an open syllable (followed by
// at most 1 consonant before next vowel). Open syllable → long vowel.
inline bool is_open_syllable(const std::string& w, int i) {
    int cons = count_following_consonant_units(w, i);
    if (cons == 0)
        return true; // word-final vowel or before another vowel
    if (cons == 1) {
        // Find position after the consonant unit
        int j = i + 1;
        int wlen = (int)w.size();
        while (j < wlen && (unsigned char)w[j] < 0x80 && !is_vowel_ascii(w[j]) && w[j] >= 'a' && w[j] <= 'z')
            j++;
        if (j < wlen && is_vowel_ascii(w[j]))
            return true;
        if (j + 1 < wlen && (unsigned char)w[j] == 0xC3)
            return true;
    }
    return false;
}

// ── Auslautverhärtung (final devoicing) ─────────────────────────────
// Applied as post-processing to the IPA string.

inline std::string apply_final_devoicing(const std::string& ipa) {
    if (ipa.empty())
        return ipa;
    std::string out = ipa;

    // Map voiced → voiceless for final position
    // b→p, d→t, ɡ→k, v→f, z→s
    // These are all single-byte ASCII or specific UTF-8 sequences
    size_t len = out.size();

    // Check last IPA character (may be multi-byte)
    if (len >= 1) {
        char last = out[len - 1];
        if (last == 'b') {
            out[len - 1] = 'p';
        } else if (last == 'd') {
            out[len - 1] = 't';
        } else if (last == 'z') {
            out[len - 1] = 's';
        } else if (last == 'v') {
            out[len - 1] = 'f';
        }
    }
    // ɡ (C9 A1) → k
    if (len >= 2 && (unsigned char)out[len - 2] == 0xC9 && (unsigned char)out[len - 1] == 0xA1) {
        out.resize(len - 2);
        out += 'k';
    }
    // ʁ (CA 81) at end — not devoiced, but preceding consonant might be
    // This is handled naturally since we check the last char

    return out;
}

// ── Compound word splitting ─────────────────────────────────────────
// Greedy longest-match: try to split a word into known dictionary parts.
// Returns vector of subwords, or single element if no split found.

inline std::vector<std::string> split_compound(const dictionary& dict, const std::string& word) {
    if (!dict.loaded || word.size() < 6)
        return {word}; // too short to be compound

    int len = (int)word.size();
    // Try splitting at each position, preferring longer left parts
    for (int split = len - 3; split >= 3; split--) {
        std::string left = word.substr(0, split);
        std::string right = word.substr(split);
        // Check Fugen-s: if left ends in 's', try without it too
        bool fugen_s = false;
        if (left.back() == 's' && left.size() > 3) {
            std::string left_no_s = left.substr(0, left.size() - 1);
            if (dict.entries.count(left_no_s) && dict.entries.count(right)) {
                fugen_s = true;
                left = left_no_s;
            }
        }
        if (!fugen_s && dict.entries.count(left) && dict.entries.count(right)) {
            return {left, right};
        }
        if (fugen_s) {
            return {left, right};
        }
    }
    return {word};
}

// ── Rule-based German G2P ───────────────────────────────────────────

inline std::string lts_word_to_ipa(const std::string& word) {
    std::string ipa;
    std::string w = to_lower_de(word);
    int len = (int)w.size();

    for (int i = 0; i < len;) {
        auto at = [&](int offset) -> char {
            int idx = i + offset;
            return (idx >= 0 && idx < len) ? w[idx] : 0;
        };
        char c = at(0), c1 = at(1), c2 = at(2), c3 = at(3);

        // --- 4-char sequences ---
        if (c == 't' && c1 == 's' && c2 == 'c' && c3 == 'h') {
            ipa += "t\xCA\x83";
            i += 4;
            continue;
        }

        // --- 3-char sequences ---
        if (c == 's' && c1 == 'c' && c2 == 'h') {
            ipa += "\xCA\x83";
            i += 3;
            continue;
        }
        if (c == 'c' && c1 == 'h' && c2 == 's') {
            ipa += "ks";
            i += 3;
            continue;
        }

        // --- 2-char sequences ---
        if (c == 'c' && c1 == 'h') {
            char prev = (i > 0) ? w[i - 1] : 0;
            if (prev == 'a' || prev == 'o' || prev == 'u')
                ipa += "x";
            else
                ipa += "\xC3\xA7"; // ç
            i += 2;
            continue;
        }
        if (c == 'c' && c1 == 'k') {
            ipa += "k";
            i += 2;
            continue;
        }
        if (c == 'p' && c1 == 'h') {
            ipa += "f";
            i += 2;
            continue;
        }
        if (c == 'p' && c1 == 'f') {
            ipa += "pf";
            i += 2;
            continue;
        }
        if (c == 't' && c1 == 'h') {
            ipa += "t";
            i += 2;
            continue;
        }
        if (c == 't' && c1 == 'z') {
            ipa += "ts";
            i += 2;
            continue;
        }
        if (c == 'd' && c1 == 't') {
            ipa += "t";
            i += 2;
            continue;
        }
        if (c == 'n' && c1 == 'g') {
            ipa += "\xC5\x8B";
            i += 2;
            continue;
        }
        if (c == 'n' && c1 == 'k') {
            ipa += "\xC5\x8B"
                   "k";
            i += 2;
            continue;
        }
        if (c == 'q' && c1 == 'u') {
            ipa += "kv";
            i += 2;
            continue;
        }

        // Vowel digraphs
        if (c == 'e' && c1 == 'i') {
            ipa += "a\xC9\xAA\xCC\xAF";
            i += 2;
            continue;
        }
        if (c == 'e' && c1 == 'u') {
            ipa += "\xC9\x94\xCA\x8F\xCC\xAF";
            i += 2;
            continue;
        }
        if (c == 'a' && c1 == 'u') {
            ipa += "a\xCA\x8A\xCC\xAF";
            i += 2;
            continue;
        }
        if (c == 'i' && c1 == 'e') {
            ipa += "i\xCB\x90";
            i += 2;
            continue;
        }
        if (c == 'e' && c1 == 'e') {
            ipa += "e\xCB\x90";
            i += 2;
            continue;
        }
        if (c == 'o' && c1 == 'o') {
            ipa += "o\xCB\x90";
            i += 2;
            continue;
        }
        if (c == 'a' && c1 == 'a') {
            ipa += "\xC9\x91\xCB\x90";
            i += 2;
            continue;
        } // aa → ɑː
        // Lengthening-h digraphs
        if (c == 'e' && c1 == 'h') {
            ipa += "e\xCB\x90";
            i += 2;
            continue;
        }
        if (c == 'a' && c1 == 'h') {
            ipa += "\xC9\x91\xCB\x90";
            i += 2;
            continue;
        } // ah → ɑː
        if (c == 'o' && c1 == 'h') {
            ipa += "o\xCB\x90";
            i += 2;
            continue;
        }
        if (c == 'u' && c1 == 'h') {
            ipa += "u\xCB\x90";
            i += 2;
            continue;
        }
        if (c == 'i' && c1 == 'h') {
            ipa += "i\xCB\x90";
            i += 2;
            continue;
        }

        // ä-digraphs (ä = C3 A4)
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xA4) {
            if (at(2) == 'u') {
                ipa += "\xC9\x94\xCA\x8F\xCC\xAF";
                i += 3;
                continue;
            } // äu → ɔʏ̯
            if (at(2) == 'h') {
                ipa += "\xC9\x9B\xCB\x90";
                i += 3;
                continue;
            } // äh → ɛː
            // Open syllable check: i+1 is the second byte of ä, so check from i+1
            if (is_open_syllable(w, i + 1))
                ipa += "\xC9\x9B\xCB\x90"; // ɛː
            else
                ipa += "\xC9\x9B"; // ɛ
            i += 2;
            continue;
        }
        // ö = C3 B6
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xB6) {
            if (at(2) == 'h') {
                ipa += "\xC3\xB8\xCB\x90";
                i += 3;
                continue;
            } // öh → øː
            if (is_open_syllable(w, i + 1))
                ipa += "\xC3\xB8\xCB\x90"; // øː
            else
                ipa += "\xC5\x93"; // œ
            i += 2;
            continue;
        }
        // ü = C3 BC
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xBC) {
            if (at(2) == 'h') {
                ipa += "y\xCB\x90";
                i += 3;
                continue;
            } // üh → yː
            if (is_open_syllable(w, i + 1))
                ipa += "y\xCB\x90"; // yː
            else
                ipa += "y"; // y (short)
            i += 2;
            continue;
        }
        // ß = C3 9F
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0x9F) {
            ipa += "s";
            i += 2;
            continue;
        }

        // st/sp at start → ʃt/ʃp
        if (i == 0 || (i > 0 && (w[i - 1] == ' ' || w[i - 1] == '-'))) {
            if (c == 's' && c1 == 't') {
                ipa += "\xCA\x83"
                       "t";
                i += 2;
                continue;
            }
            if (c == 's' && c1 == 'p') {
                ipa += "\xCA\x83"
                       "p";
                i += 2;
                continue;
            }
        }
        // ss → voiceless s (must be before double-consonant rule)
        if (c == 's' && c1 == 's') {
            ipa += "s";
            i += 2;
            continue;
        }

        // Double consonants → single (signals short vowel, handled by open-syllable check)
        if (c == c1 && c >= 'a' && c <= 'z' && !is_vowel_ascii(c)) {
            // Fall through to single-char rules, skip one
            i++;
            continue;
        }

        // --- Single vowels with open-syllable lengthening ---
        if (c == 'a') {
            if (is_open_syllable(w, i))
                ipa += "\xC9\x91\xCB\x90"; // ɑː (espeak-ng DE)
            else
                ipa += "a";
            i++;
            continue;
        }
        if (c == 'e') {
            if (i == len - 1) {
                ipa += "\xC9\x99";
                i++;
                continue;
            } // final schwa
            if (c1 == 'r' && (i + 2 == len || at(2) == ' ' || at(2) == '-')) {
                ipa += "\xC9\x9C";
                i += 2;
                continue; // -er → ɜ (espeak-ng DE)
            }
            if (is_open_syllable(w, i))
                ipa += "e\xCB\x90"; // eː
            else
                ipa += "\xC9\x9B"; // ɛ
            i++;
            continue;
        }
        if (c == 'i') {
            if (is_open_syllable(w, i))
                ipa += "i\xCB\x90"; // iː
            else
                ipa += "\xC9\xAA"; // ɪ
            i++;
            continue;
        }
        if (c == 'o') {
            if (is_open_syllable(w, i))
                ipa += "o\xCB\x90"; // oː
            else
                ipa += "\xC9\x94"; // ɔ
            i++;
            continue;
        }
        if (c == 'u') {
            if (is_open_syllable(w, i))
                ipa += "u\xCB\x90"; // uː
            else
                ipa += "\xCA\x8A"; // ʊ
            i++;
            continue;
        }

        // --- Single consonants ---
        if (c == 'b') {
            ipa += "b";
            i++;
            continue;
        }
        if (c == 'c') {
            ipa += "k";
            i++;
            continue;
        }
        if (c == 'd') {
            ipa += "d";
            i++;
            continue;
        }
        if (c == 'f') {
            ipa += "f";
            i++;
            continue;
        }
        if (c == 'g') {
            ipa += "\xC9\xA1";
            i++;
            continue;
        }
        if (c == 'h') {
            ipa += "h";
            i++;
            continue;
        }
        if (c == 'j') {
            ipa += "j";
            i++;
            continue;
        }
        if (c == 'k') {
            ipa += "k";
            i++;
            continue;
        }
        if (c == 'l') {
            ipa += "l";
            i++;
            continue;
        }
        if (c == 'm') {
            ipa += "m";
            i++;
            continue;
        }
        if (c == 'n') {
            ipa += "n";
            i++;
            continue;
        }
        if (c == 'p') {
            ipa += "p";
            i++;
            continue;
        }
        if (c == 'r') {
            ipa += "r";
            i++;
            continue;
        } // espeak-ng DE uses plain r
        if (c == 's') {
            if (c1 == 'a' || c1 == 'e' || c1 == 'i' || c1 == 'o' || c1 == 'u' || (unsigned char)c1 == 0xC3) {
                ipa += "z";
                i++;
                continue;
            }
            ipa += "s";
            i++;
            continue;
        }
        if (c == 't') {
            ipa += "t";
            i++;
            continue;
        }
        if (c == 'v') {
            ipa += "f";
            i++;
            continue;
        }
        if (c == 'w') {
            ipa += "v";
            i++;
            continue;
        }
        if (c == 'x') {
            ipa += "ks";
            i++;
            continue;
        }
        if (c == 'y') {
            ipa += "y";
            i++;
            continue;
        }
        if (c == 'z') {
            ipa += "ts";
            i++;
            continue;
        }
        i++; // skip unknown
    }

    // Apply Auslautverhärtung (final devoicing)
    ipa = apply_final_devoicing(ipa);

    return ipa;
}

// ── Context ─────────────────────────────────────────────────────────

struct context {
    dictionary dict;

    // #316 round 2: carry punctuation through into the phoneme string. Kokoro's
    // 178-symbol vocabulary contains `,.;:!?…—"«»“”` — they are how the model
    // knows to pause, and dropping them delivers a paragraph in one breath.
    // English proved it (misaki emits them and we did not); this is the same
    // defect in the same shape. Off by default: piper's phoneme inventory is
    // espeak's, and that consumer has never been fed punctuation, so the flag
    // is the caller's to set. See g2p_en::style.
    bool emit_punctuation = false;

    // Read the German closed class the way espeak reads it IN A SENTENCE.
    // `espeak_de.tsv` was generated one word at a time, so every entry is the
    // citation form — and espeak stresses "sie" as `zˈiː` alone but `ziː` in a
    // sentence. We therefore put a primary stress on every article, pronoun,
    // preposition and auxiliary, which is the German shape of the #316 English
    // bug. Turning this on takes token agreement with espeak's sentence output
    // from 45.9% to 87.1%.
    //
    // ON by default since the training recipe was found: dida-80b/kokoro-deutsch
    // phonemizes the WHOLE SENTENCE through espeak (misaki's
    // `EspeakG2P(language="de")`), so espeak's sentence-level de-stressing is
    // in the training data and the citation form is a spelling the model never
    // saw. That is source evidence, not a preference.
    // CRISPASR_G2P_DE_UNSTRESS=0 restores the citation forms for A/B.
    bool unstress_function_words = true;
};

// ── Tokenizer ───────────────────────────────────────────────────────

inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        if (c == ' ' || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':' || c == '-' || c == '\n' ||
            c == '"' || c == '(' || c == ')' || c == '[' || c == ']') {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
            if (c != ' ')
                tokens.push_back(std::string(1, c));
        } else {
            cur += c;
        }
    }
    if (!cur.empty())
        tokens.push_back(cur);
    return tokens;
}

// ── Main API ────────────────────────────────────────────────────────

// A single letter on its own is read by its NAME, as espeak-ng (the model's
// training phonemizer) reads it: chess squares ("d4" -> "d vier" after the
// digit split), grades, list items. The espeak_de dictionary was generated one
// word at a time and has no single letters, so they used to fall through to
// the letter-to-sound rules - "d" came out `t` (final devoicing), "f" as `f`.
// Values: `espeak-ng -v de -q --ipa <letter>`.
inline const char* letter_name_de(const std::string& lower) {
    static const std::map<std::string, const char*> names = {
        {"a", "ˈɑː"},       {"b", "bˈeː"},  {"c", "tsˈeː"}, {"d", "dˈeː"}, {"e", "ˈeː"},  {"f", "ˈɛf"},
        {"g", "ɡˈeː"},      {"h", "hˈɑː"},  {"i", "ˈiː"},   {"j", "jˈɔt"}, {"k", "kˈɑː"}, {"l", "ˈɛl"},
        {"m", "ˈɛm"},       {"n", "ˈɛn"},   {"o", "ˈoː"},   {"p", "pˈeː"}, {"q", "kˈuː"}, {"r", "ˈɛɾ"},
        {"s", "ˈɛs"},       {"t", "tˈeː"},  {"u", "ˈuː"},   {"v", "fˈaʊ"}, {"w", "vˈeː"}, {"x", "ˈɪks"},
        {"y", "ˈʏpsɪlˌɔn"}, {"z", "tsˈɛt"}, {"ä", "ˈɛː"},   {"ö", "ˈøː"},  {"ü", "ˈyː"},
    };
    auto it = names.find(lower);
    return it == names.end() ? nullptr : it->second;
}

inline std::string word_to_ipa(const context& ctx, const std::string& word) {
    std::string lower = to_lower_de(word);

    // Tier 0: a lone letter
    if (const char* name = letter_name_de(lower))
        return name;

    // Tier 1: full-word dictionary lookup
    if (ctx.dict.loaded) {
        auto it = ctx.dict.entries.find(lower);
        if (it != ctx.dict.entries.end())
            return it->second;
    }

    // Tier 2: compound splitting (try dictionary parts)
    if (ctx.dict.loaded) {
        auto parts = split_compound(ctx.dict, lower);
        if (parts.size() > 1) {
            std::string ipa;
            for (const auto& part : parts) {
                auto it = ctx.dict.entries.find(part);
                if (it != ctx.dict.entries.end()) {
                    if (!ipa.empty())
                        ipa += "."; // syllable boundary
                    ipa += it->second;
                } else {
                    if (!ipa.empty())
                        ipa += ".";
                    ipa += lts_word_to_ipa(part);
                }
            }
            return ipa;
        }
    }

    // Tier 3: rules
    return lts_word_to_ipa(lower);
}

inline std::string text_to_ipa(const context& ctx, const std::string& text) {
    // #316: spell numbers out FIRST. Digits are in no pronunciation dictionary
    // and no letter-to-sound rule, so a numeric token phonemized to the EMPTY
    // string and vanished from the audio — "Ich habe 82 Euro" was spoken as
    // "Ich habe Euro". Nothing errored; the number was simply gone.
    //
    // This must run before tokenize(), which splits on ',' and '.' — the two
    // characters German uses as decimal mark and thousands separator. Expanding
    // afterwards would already have "3,14" torn into "3", "," and "14".
    auto words = tokenize(core_num2words_de::expand(core_g2p_alnum::split_letters_digits(text)));
    // A mark that the consumer wants goes flush against the word before it and
    // is followed by a space, the way misaki emits English. A mark it does not
    // want still separates its neighbours — with ONE space, not the two the old
    // loop produced around every comma. `-` is a separator only: it is in no
    // TTS vocabulary here.
    std::string ipa;
    bool pending_space = false;
    for (const auto& w : words) {
        const bool punct = w.size() == 1 && strchr(",.!?;:-\"()[]'", w[0]) != nullptr;
        if (punct && (!ctx.emit_punctuation || w[0] == '-' || w[0] == '\'')) {
            if (!ipa.empty())
                pending_space = true;
            continue;
        }
        if (punct) {
            ipa += w; // attaches to the word before it
            pending_space = true;
            continue;
        }
        std::string ph = word_to_ipa(ctx, w);
        if (ph.empty())
            continue;
        if (ctx.unstress_function_words) {
            std::string lower;
            for (char c : w)
                lower += (char)tolower((unsigned char)c);
            const std::string un = core_g2p_de_unstressed::lookup(lower);
            if (!un.empty())
                ph = un;
        }
        if (!ipa.empty() && (pending_space || ipa.back() != ' '))
            ipa += ' ';
        pending_space = false;
        ipa += ph;
    }
    return ipa;
}

} // namespace g2p_de
