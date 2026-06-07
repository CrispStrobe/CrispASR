// core/g2p_fr.h — French grapheme-to-phoneme (text → IPA).
//
// Rule-based French G2P. French orthography is fairly regular for
// reading (more predictable than English, less than German/Spanish).
// Main challenges: silent final consonants, nasal vowels, liaison.
//
// Covers standard metropolitan French pronunciation.

#pragma once

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace g2p_fr {

// ── Dictionary ──────────────────────────────────────────────────────

struct dictionary {
    std::map<std::string, std::string> entries; // word → IPA
    bool loaded = false;
};

// Load open-dict-data format: "word\t/IPA1/, /IPA2/\n"
inline int load_ipa_dict_file(dictionary& dict, const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return 0;
    char line[1024];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        char* tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        std::string word = line;
        for (auto& c : word) c = (char)tolower((unsigned char)c);
        if (dict.entries.count(word)) continue;
        char* ipa_start = tab + 1;
        while (*ipa_start == '/' || *ipa_start == ' ') ipa_start++;
        std::string ipa;
        for (char* p = ipa_start; *p && *p != '/' && *p != ','; p++) ipa += *p;
        while (!ipa.empty() && (ipa.back() == ' ' || ipa.back() == '/')) ipa.pop_back();
        if (!ipa.empty() && !word.empty()) { dict.entries[word] = ipa; count++; }
    }
    fclose(f);
    dict.loaded = count > 0;
    return count;
}

// ── Rule-based French G2P ───────────────────────────────────────────

inline std::string lts_word_to_ipa(const std::string& word) {
    std::string ipa;
    std::string w;
    for (char c : word) w += (char)tolower((unsigned char)c);
    int len = (int)w.size();

    for (int i = 0; i < len;) {
        auto at = [&](int offset) -> char {
            int idx = i + offset;
            return (idx >= 0 && idx < len) ? w[idx] : 0;
        };
        // UTF-8 helper: check for 2-byte accented chars (C3 xx)
        auto is_utf8_2 = [&](int offset, unsigned char byte2) -> bool {
            int idx = i + offset;
            return (idx + 1 < len && (unsigned char)w[idx] == 0xC3 && (unsigned char)w[idx+1] == byte2);
        };
        char c = at(0), c1 = at(1), c2 = at(2), c3 = at(3);

        // --- 4-char ---
        if (c == 't' && c1 == 'i' && c2 == 'o' && c3 == 'n') {
            // -tion → sjɔ̃ (before end or consonant)
            if (i + 4 == len || (i + 4 < len && !strchr("aeiouy", at(4)))) {
                ipa += "sj\xC9\x94\xCC\x83"; i += 4; continue; // sjɔ̃
            }
        }

        // --- 3-char ---
        if (c == 'e' && c1 == 'a' && c2 == 'u') { ipa += "o"; i += 3; continue; } // eau → o
        if (c == 'o' && c1 == 'u' && c2 == 'i') { ipa += "wi"; i += 3; continue; } // oui → wi
        if (c == 'a' && c1 == 'i' && c2 == 'n') {
            if (i + 3 == len || (at(3) != 'e' && !strchr("aeiouy", at(3)))) {
                ipa += "\xC9\x9B\xCC\x83"; i += 3; continue; // ain → ɛ̃
            }
        }
        if (c == 'e' && c1 == 'i' && c2 == 'n') {
            if (i + 3 == len || !strchr("aeiouy", at(3))) {
                ipa += "\xC9\x9B\xCC\x83"; i += 3; continue; // ein → ɛ̃
            }
        }
        if (c == 'o' && c1 == 'i' && c2 == 'n') {
            if (i + 3 == len || !strchr("aeiouy", at(3))) {
                ipa += "w\xC9\x9B\xCC\x83"; i += 3; continue; // oin → wɛ̃
            }
        }

        // --- 2-char ---
        if (c == 'c' && c1 == 'h') { ipa += "\xCA\x83"; i += 2; continue; }  // ch → ʃ
        if (c == 'p' && c1 == 'h') { ipa += "f"; i += 2; continue; }
        if (c == 'g' && c1 == 'n') { ipa += "\xC9\xB2"; i += 2; continue; }  // gn → ɲ
        if (c == 'q' && c1 == 'u') { ipa += "k"; i += 2; continue; }
        if (c == 'g' && c1 == 'u') {
            if (i + 2 < len && strchr("eiy", at(2))) { ipa += "\xC9\xA1"; i += 2; continue; } // gu+e/i → ɡ
        }
        if (c == 'o' && c1 == 'u') { ipa += "u"; i += 2; continue; }
        if (c == 'o' && c1 == 'i') { ipa += "wa"; i += 2; continue; }         // oi → wa
        if (c == 'a' && c1 == 'u') { ipa += "o"; i += 2; continue; }          // au → o
        if (c == 'a' && c1 == 'i') { ipa += "\xC9\x9B"; i += 2; continue; }   // ai → ɛ
        if (c == 'e' && c1 == 'i') { ipa += "\xC9\x9B"; i += 2; continue; }   // ei → ɛ
        if (c == 'e' && c1 == 'u') { ipa += "\xC3\xB8"; i += 2; continue; }   // eu → ø
        if (c == 'a' && c1 == 'n') {
            if (i + 2 == len || !strchr("aeiouy", at(2))) {
                ipa += "\xC9\x91\xCC\x83"; i += 2; continue; // an → ɑ̃
            }
        }
        if (c == 'e' && c1 == 'n') {
            if (i + 2 == len || !strchr("aeiouy", at(2))) {
                ipa += "\xC9\x91\xCC\x83"; i += 2; continue; // en → ɑ̃
            }
        }
        if (c == 'e' && c1 == 'm') {
            if (i + 2 == len || !strchr("aeiouy", at(2))) {
                ipa += "\xC9\x91\xCC\x83"; i += 2; continue; // em → ɑ̃
            }
        }
        if (c == 'a' && c1 == 'm') {
            if (i + 2 == len || !strchr("aeiouy", at(2))) {
                ipa += "\xC9\x91\xCC\x83"; i += 2; continue; // am → ɑ̃
            }
        }
        if (c == 'o' && c1 == 'n') {
            if (i + 2 == len || !strchr("aeiouy", at(2))) {
                ipa += "\xC9\x94\xCC\x83"; i += 2; continue; // on → ɔ̃
            }
        }
        if (c == 'o' && c1 == 'm') {
            if (i + 2 == len || !strchr("aeiouy", at(2))) {
                ipa += "\xC9\x94\xCC\x83"; i += 2; continue; // om → ɔ̃
            }
        }
        if (c == 'i' && c1 == 'n') {
            if (i + 2 == len || !strchr("aeiouy", at(2))) {
                ipa += "\xC9\x9B\xCC\x83"; i += 2; continue; // in → ɛ̃
            }
        }
        if (c == 'i' && c1 == 'm') {
            if (i + 2 == len || !strchr("aeiouy", at(2))) {
                ipa += "\xC9\x9B\xCC\x83"; i += 2; continue; // im → ɛ̃
            }
        }
        if (c == 'u' && c1 == 'n') {
            if (i + 2 == len || !strchr("aeiouy", at(2))) {
                ipa += "\xC5\x93\xCC\x83"; i += 2; continue; // un → œ̃
            }
        }
        if (c == 'l' && c1 == 'l') {
            // ill → j (after vowel)
            ipa += "l"; i += 2; continue;
        }
        if (c == 's' && c1 == 's') { ipa += "s"; i += 2; continue; }
        if (c == 't' && c1 == 't') { ipa += "t"; i += 2; continue; }
        if (c == 'n' && c1 == 'n') { ipa += "n"; i += 2; continue; }
        if (c == 'r' && c1 == 'r') { ipa += "\xCA\x81"; i += 2; continue; }   // rr → ʁ
        if (c == 'c' && c1 == 'c') { ipa += "ks"; i += 2; continue; }

        // é = C3 A9
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xA9) { ipa += "e"; i += 2; continue; }
        // è = C3 A8
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xA8) { ipa += "\xC9\x9B"; i += 2; continue; }
        // ê = C3 AA
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xAA) { ipa += "\xC9\x9B"; i += 2; continue; }
        // ë = C3 AB
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xAB) { ipa += "\xC9\x9B"; i += 2; continue; }
        // à = C3 A0
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xA0) { ipa += "a"; i += 2; continue; }
        // â = C3 A2
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xA2) { ipa += "\xC9\x91"; i += 2; continue; } // ɑ
        // ù = C3 B9
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xB9) { ipa += "y"; i += 2; continue; }
        // û = C3 BB
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xBB) { ipa += "y"; i += 2; continue; }
        // ô = C3 B4
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xB4) { ipa += "o"; i += 2; continue; }
        // î = C3 AE
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xAE) { ipa += "i"; i += 2; continue; }
        // ç = C3 A7
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xA7) { ipa += "s"; i += 2; continue; }

        // --- Silent final consonants ---
        if (i == len - 1 && (c == 's' || c == 't' || c == 'd' || c == 'x' || c == 'z' || c == 'p' || c == 'g')) {
            i++; continue; // silent
        }

        // Final -e is mute (schwa dropped in most positions)
        if (c == 'e' && i == len - 1) { i++; continue; }
        // Final -es is silent
        if (c == 'e' && c1 == 's' && i + 2 == len) { i += 2; continue; }
        // Final -ent (3rd person plural) is silent
        if (c == 'e' && c1 == 'n' && c2 == 't' && i + 3 == len) { i += 3; continue; }

        // --- Single characters ---
        if (c == 'a') { ipa += "a"; i++; continue; }
        if (c == 'b') { ipa += "b"; i++; continue; }
        if (c == 'c') {
            if (c1 == 'e' || c1 == 'i' || c1 == 'y') { ipa += "s"; i++; continue; }
            ipa += "k"; i++; continue;
        }
        if (c == 'd') { ipa += "d"; i++; continue; }
        if (c == 'e') { ipa += "\xC9\x99"; i++; continue; }  // schwa ə
        if (c == 'f') { ipa += "f"; i++; continue; }
        if (c == 'g') {
            if (c1 == 'e' || c1 == 'i' || c1 == 'y') { ipa += "\xCA\x92"; i++; continue; } // ʒ
            ipa += "\xC9\xA1"; i++; continue; // ɡ
        }
        if (c == 'h') { i++; continue; }  // always silent in French
        if (c == 'i') { ipa += "i"; i++; continue; }
        if (c == 'j') { ipa += "\xCA\x92"; i++; continue; }  // ʒ
        if (c == 'k') { ipa += "k"; i++; continue; }
        if (c == 'l') { ipa += "l"; i++; continue; }
        if (c == 'm') { ipa += "m"; i++; continue; }
        if (c == 'n') { ipa += "n"; i++; continue; }
        if (c == 'o') { ipa += "\xC9\x94"; i++; continue; }  // ɔ
        if (c == 'p') { ipa += "p"; i++; continue; }
        if (c == 'r') { ipa += "\xCA\x81"; i++; continue; }   // ʁ
        if (c == 's') {
            // s between vowels → z
            if (i > 0 && i + 1 < len) {
                char prev = w[i-1];
                if (strchr("aeiouy", prev) && strchr("aeiouy", c1)) {
                    ipa += "z"; i++; continue;
                }
            }
            ipa += "s"; i++; continue;
        }
        if (c == 't') { ipa += "t"; i++; continue; }
        if (c == 'u') { ipa += "y"; i++; continue; }  // French u → y
        if (c == 'v') { ipa += "v"; i++; continue; }
        if (c == 'w') { ipa += "w"; i++; continue; }
        if (c == 'x') { ipa += "ks"; i++; continue; }
        if (c == 'y') { ipa += "i"; i++; continue; }
        if (c == 'z') { ipa += "z"; i++; continue; }

        i++; // skip unknown
    }
    return ipa;
}

// ── Context ─────────────────────────────────────────────────────────

struct context {
    dictionary dict;
};

inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        if (c == ' ' || c == ',' || c == '.' || c == '!' || c == '?' ||
            c == ';' || c == ':' || c == '-' || c == '\n' || c == '\'') {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
            if (c != ' ') tokens.push_back(std::string(1, c));
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

inline std::string word_to_ipa(const context& ctx, const std::string& word) {
    std::string lower;
    for (size_t i = 0; i < word.size(); i++) {
        unsigned char c = (unsigned char)word[i];
        if (c >= 'A' && c <= 'Z') lower += (char)(c + 32);
        else lower += word[i];
    }
    if (ctx.dict.loaded) {
        auto it = ctx.dict.entries.find(lower);
        if (it != ctx.dict.entries.end()) return it->second;
    }
    return lts_word_to_ipa(lower);
}

inline std::string text_to_ipa(const context& ctx, const std::string& text) {
    auto words = tokenize(text);
    std::string ipa;
    for (const auto& w : words) {
        if (w.size() == 1 && strchr(",.!?;:-'", w[0])) continue;
        if (!ipa.empty()) ipa += ' ';
        ipa += word_to_ipa(ctx, w);
    }
    return ipa;
}

} // namespace g2p_fr
