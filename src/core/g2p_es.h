// core/g2p_es.h — Spanish grapheme-to-phoneme (text → IPA).
//
// Spanish has nearly phonemic orthography — rules are very accurate
// (~98% for native words). Latin American seseo (c/z → s) by default.
// Main features: b/d/g allophonic lenition, ñ, ll→ʝ (yeísmo),
// ch→tʃ, silent h, stress rules.

#pragma once

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace g2p_es {

struct dictionary {
    std::map<std::string, std::string> entries;
    bool loaded = false;
};

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

// ── Rule-based Spanish G2P ──────────────────────────────────────────

inline std::string lts_word_to_ipa(const std::string& word) {
    std::string ipa;
    std::string w;
    for (size_t i = 0; i < word.size(); i++) {
        unsigned char c = (unsigned char)word[i];
        // Lowercase ASCII
        if (c >= 'A' && c <= 'Z')
            w += (char)(c + 32);
        // Lowercase accented (C3 xx range)
        else if (c == 0xC3 && i + 1 < word.size()) {
            unsigned char c2 = (unsigned char)word[i + 1];
            // Á→á, É→é, Í→í, Ó→ó, Ú→ú, Ñ→ñ, Ü→ü
            if (c2 >= 0x80 && c2 <= 0x9E) {
                w += (char)c;
                w += (char)(c2 + 0x20);
                i++;
            } else {
                w += word[i];
            }
        } else
            w += word[i];
    }
    int len = (int)w.size();

    // Helper: check if position i is at a vowel
    auto is_vowel = [](char c) -> bool { return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'; };
    // Check if previous char was a vowel (for lenition)
    auto prev_vowel = [&](int pos) -> bool { return pos > 0 && is_vowel(w[pos - 1]); };
    // Check if word-initial or after nasal (for stop vs fricative)
    auto is_stop_context = [&](int pos) -> bool {
        if (pos == 0)
            return true;
        char prev = w[pos - 1];
        return prev == 'n' || prev == 'm' || prev == 'l';
    };

    for (int i = 0; i < len;) {
        auto at = [&](int offset) -> char {
            int idx = i + offset;
            return (idx >= 0 && idx < len) ? w[idx] : 0;
        };
        char c = at(0), c1 = at(1), c2 = at(2);

        // UTF-8 2-byte sequences (C3 xx)
        if ((unsigned char)c == 0xC3 && i + 1 < len) {
            unsigned char c2b = (unsigned char)w[i + 1];
            // ñ = C3 B1
            if (c2b == 0xB1) {
                ipa += "\xC9\xB2";
                i += 2;
                continue;
            } // ɲ
            // á = C3 A1
            if (c2b == 0xA1) {
                ipa += "a";
                i += 2;
                continue;
            }
            // é = C3 A9
            if (c2b == 0xA9) {
                ipa += "e";
                i += 2;
                continue;
            }
            // í = C3 AD
            if (c2b == 0xAD) {
                ipa += "i";
                i += 2;
                continue;
            }
            // ó = C3 B3
            if (c2b == 0xB3) {
                ipa += "o";
                i += 2;
                continue;
            }
            // ú = C3 BA
            if (c2b == 0xBA) {
                ipa += "u";
                i += 2;
                continue;
            }
            // ü = C3 BC
            if (c2b == 0xBC) {
                ipa += "w";
                i += 2;
                continue;
            }
            i += 2;
            continue; // skip unknown accented
        }

        // --- Multi-char ---
        if (c == 'c' && c1 == 'h') {
            ipa += "t\xCA\x83";
            i += 2;
            continue;
        } // ch → tʃ
        if (c == 'l' && c1 == 'l') {
            ipa += "\xCA\x9D";
            i += 2;
            continue;
        } // ll → ʝ (yeísmo)
        if (c == 'r' && c1 == 'r') {
            ipa += "r";
            i += 2;
            continue;
        } // rr → r (trill)
        if (c == 'q' && c1 == 'u') {
            ipa += "k";
            i += 2;
            continue;
        }
        if (c == 'g' && c1 == 'u') {
            // gu before e/i: silent u
            if (c2 == 'e' || c2 == 'i') {
                if (is_stop_context(i))
                    ipa += "\xC9\xA1"; // ɡ
                else
                    ipa += "\xC9\xA3"; // ɣ
                i += 2;
                continue;
            }
        }

        // --- Single chars ---
        if (c == 'a') {
            ipa += "a";
            i++;
            continue;
        }
        if (c == 'b' || c == 'v') {
            // Stop after pause/nasal/l, fricative elsewhere
            if (is_stop_context(i))
                ipa += "b";
            else
                ipa += "\xCE\xB2"; // β
            i++;
            continue;
        }
        if (c == 'c') {
            if (c1 == 'e' || c1 == 'i') {
                ipa += "s";
                i++;
                continue;
            } // seseo
            ipa += "k";
            i++;
            continue;
        }
        if (c == 'd') {
            if (is_stop_context(i))
                ipa += "d";
            else
                ipa += "\xC3\xB0"; // ð
            i++;
            continue;
        }
        if (c == 'e') {
            ipa += "e";
            i++;
            continue;
        }
        if (c == 'f') {
            ipa += "f";
            i++;
            continue;
        }
        if (c == 'g') {
            if (c1 == 'e' || c1 == 'i') {
                ipa += "x";
                i++;
                continue;
            } // jota
            if (is_stop_context(i))
                ipa += "\xC9\xA1"; // ɡ
            else
                ipa += "\xC9\xA3"; // ɣ
            i++;
            continue;
        }
        if (c == 'h') {
            i++;
            continue;
        } // silent
        if (c == 'i') {
            ipa += "i";
            i++;
            continue;
        }
        if (c == 'j') {
            ipa += "x";
            i++;
            continue;
        } // jota
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
        if (c == 'o') {
            ipa += "o";
            i++;
            continue;
        }
        if (c == 'p') {
            ipa += "p";
            i++;
            continue;
        }
        if (c == 'r') {
            if (i == 0)
                ipa += "r"; // trill word-initial
            else
                ipa += "\xC9\xBE"; // ɾ tap
            i++;
            continue;
        }
        if (c == 's') {
            ipa += "s";
            i++;
            continue;
        }
        if (c == 't') {
            ipa += "t";
            i++;
            continue;
        }
        if (c == 'u') {
            ipa += "u";
            i++;
            continue;
        }
        if (c == 'w') {
            ipa += "w";
            i++;
            continue;
        }
        if (c == 'x') {
            ipa += "ks";
            i++;
            continue;
        }
        if (c == 'y') {
            if (i == len - 1)
                ipa += "i"; // word-final y → vowel
            else
                ipa += "\xCA\x9D"; // ʝ
            i++;
            continue;
        }
        if (c == 'z') {
            ipa += "s";
            i++;
            continue;
        }    // seseo
        i++; // skip unknown
    }
    return ipa;
}

struct context {
    dictionary dict;
};

inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        if (c == ' ' || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':' || c == '-' || c == '\n') {
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

inline std::string word_to_ipa(const context& ctx, const std::string& word) {
    std::string lower;
    for (size_t i = 0; i < word.size(); i++) {
        unsigned char c = (unsigned char)word[i];
        if (c >= 'A' && c <= 'Z')
            lower += (char)(c + 32);
        else
            lower += word[i];
    }
    if (ctx.dict.loaded) {
        auto it = ctx.dict.entries.find(lower);
        if (it != ctx.dict.entries.end())
            return it->second;
    }
    return lts_word_to_ipa(lower);
}

inline std::string text_to_ipa(const context& ctx, const std::string& text) {
    auto words = tokenize(text);
    std::string ipa;
    for (const auto& w : words) {
        if (w.size() == 1 && strchr(",.!?;:-", w[0]))
            continue;
        if (!ipa.empty())
            ipa += ' ';
        ipa += word_to_ipa(ctx, w);
    }
    return ipa;
}

} // namespace g2p_es
