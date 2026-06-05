// truecaser.cpp — statistical truecaser from word-frequency data.
//
// Model format (little-endian):
//   uint32  n_entries
//   for each entry:
//     uint16  key_len        (UTF-8 byte count of the lowercased key)
//     char[]  key_bytes      (lowercased word, UTF-8)
//     uint32  lc             (count: all-lowercase)
//     uint32  u1             (count: first-letter-capitalised)
//     uint32  uc             (count: all-uppercase)
//
// At inference, each whitespace-delimited word is looked up (lowercased)
// and the casing variant with the highest count is applied.

#include "truecaser.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

struct CaseStats {
    uint32_t lc;
    uint32_t u1;
    uint32_t uc;
};

struct truecaser_context {
    std::unordered_map<std::string, CaseStats> dict;
};

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

// Return the byte length of a UTF-8 character starting at *p.
static int utf8_char_len(const char* p) {
    unsigned char c = (unsigned char)*p;
    if (c < 0x80)
        return 1;
    if ((c & 0xE0) == 0xC0)
        return 2;
    if ((c & 0xF0) == 0xE0)
        return 3;
    if ((c & 0xF8) == 0xF0)
        return 4;
    return 1; // invalid → skip one byte
}

// Lowercase a single ASCII character. Non-ASCII passes through.
static char ascii_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
}

// Uppercase a single ASCII character.
static char ascii_upper(char c) {
    return (c >= 'a' && c <= 'z') ? (char)(c - 'a' + 'A') : c;
}

// Lowercase a UTF-8 string (ASCII letters only — fine for German/Latin).
static std::string utf8_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        out += ascii_lower(c);
    return out;
}

// Capitalise the first letter (ASCII) of a UTF-8 string.
static std::string capitalise_first(const std::string& s) {
    if (s.empty())
        return s;
    std::string out = s;
    // Find first ASCII letter
    for (size_t i = 0; i < out.size(); i++) {
        unsigned char c = (unsigned char)out[i];
        if (c >= 'a' && c <= 'z') {
            out[i] = ascii_upper(out[i]);
            break;
        }
        if (c >= 'A' && c <= 'Z')
            break; // already upper
        if (c >= 0x80) {
            // skip multi-byte UTF-8 char
            i += utf8_char_len(&out[i]) - 1;
        }
    }
    return out;
}

// Uppercase all ASCII letters.
static std::string to_upper(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        out += ascii_upper(c);
    return out;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

truecaser_context* truecaser_init(const char* model_path) {
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "truecaser: cannot open '%s'\n", model_path);
        return nullptr;
    }

    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) {
        fclose(f);
        return nullptr;
    }

    auto* ctx = new truecaser_context();
    ctx->dict.reserve(n);

    for (uint32_t i = 0; i < n; i++) {
        uint16_t klen = 0;
        if (fread(&klen, 2, 1, f) != 1)
            break;

        std::string key(klen, '\0');
        if (fread(&key[0], 1, klen, f) != klen)
            break;

        CaseStats st{};
        if (fread(&st.lc, 4, 1, f) != 1)
            break;
        if (fread(&st.u1, 4, 1, f) != 1)
            break;
        if (fread(&st.uc, 4, 1, f) != 1)
            break;

        ctx->dict[key] = st;
    }

    fclose(f);
    fprintf(stderr, "truecaser: loaded %zu entries from '%s'\n", ctx->dict.size(), model_path);
    return ctx;
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

char* truecaser_process(truecaser_context* ctx, const char* text) {
    if (!ctx || !text)
        return nullptr;

    std::string input(text);
    std::string result;
    result.reserve(input.size() + 64);

    // Split on whitespace, truecase each word, preserve punctuation attached
    // to words (e.g. "hund." → look up "hund", keep ".").
    bool cap_next = true; // capitalise first word + after sentence enders
    size_t i = 0;
    while (i < input.size()) {
        // Skip whitespace
        if (input[i] == ' ' || input[i] == '\t' || input[i] == '\n' || input[i] == '\r') {
            result += input[i++];
            continue;
        }

        // Extract word (until whitespace)
        size_t start = i;
        while (i < input.size() && input[i] != ' ' && input[i] != '\t' && input[i] != '\n' && input[i] != '\r')
            i++;

        std::string token = input.substr(start, i - start);

        // Strip trailing punctuation for lookup
        size_t alpha_end = token.size();
        while (alpha_end > 0) {
            char c = token[alpha_end - 1];
            if (c == '.' || c == ',' || c == '?' || c == '!' || c == ':' || c == ';' || c == '-')
                alpha_end--;
            else
                break;
        }

        std::string word = token.substr(0, alpha_end);
        std::string trail = token.substr(alpha_end);

        if (word.empty()) {
            result += token;
            // Check trailing punc for sentence enders
            for (char c : trail) {
                if (c == '.' || c == '?' || c == '!')
                    cap_next = true;
            }
            continue;
        }

        std::string key = utf8_lower(word);
        auto it = ctx->dict.find(key);

        std::string cased;
        if (it != ctx->dict.end()) {
            const auto& st = it->second;
            if (st.u1 >= st.lc && st.u1 >= st.uc) {
                cased = capitalise_first(utf8_lower(word));
            } else if (st.uc > st.lc && st.uc > st.u1) {
                cased = to_upper(word);
            } else {
                cased = utf8_lower(word);
            }
        } else {
            // Unknown word — keep original casing
            cased = word;
        }

        // Force capitalise after sentence enders
        if (cap_next)
            cased = capitalise_first(cased);

        result += cased;
        result += trail;

        // Check trailing punctuation for sentence enders
        cap_next = false;
        for (char c : trail) {
            if (c == '.' || c == '?' || c == '!')
                cap_next = true;
        }
    }

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------

void truecaser_free(truecaser_context* ctx) {
    delete ctx;
}
