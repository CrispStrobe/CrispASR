// src/core/bpe.h — shared GPT-2 byte-level BPE tokenizer.
//
// Replaces the per-model copies of the same byte_encoder + bytes_to_unicode
// + bpe_one + tokenize loop that qwen3_asr.cpp and granite_speech.cpp each
// have. Both models use the OpenAI GPT-2 byte-level BPE family
// (vocab.json + merges.txt loaded into the GGUF as
// `tokenizer.ggml.tokens` + `tokenizer.ggml.merges`), so the encode side
// is identical down to the byte-permutation table and the greedy
// lowest-rank merge loop.
//
// The decode side (id -> text) lives in each model already because the
// merging-space → utf-8 conversion can in principle differ between
// tokenizers; this header only covers encode for now.
//
// Header-only: each consumer compiles its own copy. The byte_encoder
// table and the per-call BPE merge work are tiny enough that the
// indirection cost of a function-pointer interface isn't worth it.

#pragma once

#include <climits>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace core_bpe {

// GPT-2 byte → unicode codepoint table. Built lazily on first call.
// Maps each of the 256 raw bytes to a printable unicode codepoint that
// can survive a roundtrip through json/utf-8 layers. Standard
// definition from `bytes_to_unicode()` in OpenAI's GPT-2 tokenizer.
inline const std::vector<int>& byte_encoder() {
    static std::vector<int> bs(256, 0);
    static bool initialized = false;
    if (initialized)
        return bs;
    std::vector<int> printable;
    for (int b = 0x21; b <= 0x7e; b++)
        printable.push_back(b);
    for (int b = 0xa1; b <= 0xac; b++)
        printable.push_back(b);
    for (int b = 0xae; b <= 0xff; b++)
        printable.push_back(b);
    int next_extra = 256;
    for (int b = 0; b < 256; b++) {
        bool is_printable = false;
        for (int p : printable)
            if (p == b) {
                is_printable = true;
                break;
            }
        if (is_printable)
            bs[b] = b;
        else
            bs[b] = next_extra++;
    }
    initialized = true;
    return bs;
}

// Encode a single Unicode codepoint as a UTF-8 byte sequence.
inline void utf8_encode(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back((char)cp);
    } else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

// Apply the byte→unicode encoder to a raw byte buffer. Each input byte
// becomes one Unicode codepoint via the GPT-2 byte_encoder() map, then
// is encoded as UTF-8.
inline std::string bytes_to_unicode(const char* bytes, size_t n) {
    auto& enc = byte_encoder();
    std::string out;
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        utf8_encode((uint32_t)enc[(unsigned char)bytes[i]], out);
    }
    return out;
}

// Greedy lowest-rank BPE merge for a single byte-encoded pre-token.
// Appends the resulting vocab IDs to `out`. When merge_rank is empty
// (older converter that didn't write tokenizer.ggml.merges), only
// complete-token vocab lookups work — sub-words fall back to per-byte.
//
// Symbol identity check uses string concatenation with a literal space
// separator ("left right") to match the textual representation in the
// merges table.
inline void bpe_one(const std::unordered_map<std::string, int32_t>& token_to_id,
                    const std::unordered_map<std::string, int32_t>& merge_rank, const std::string& word,
                    std::vector<int32_t>& out) {
    if (word.empty())
        return;

    // Split into UTF-8 codepoint substrings — each codepoint is one symbol.
    std::vector<std::string> symbols;
    {
        size_t i = 0;
        while (i < word.size()) {
            unsigned char c = (unsigned char)word[i];
            size_t len;
            if (c < 0x80)
                len = 1;
            else if ((c & 0xE0) == 0xC0)
                len = 2;
            else if ((c & 0xF0) == 0xE0)
                len = 3;
            else if ((c & 0xF8) == 0xF0)
                len = 4;
            else
                len = 1;
            if (i + len > word.size())
                len = 1;
            symbols.emplace_back(word, i, len);
            i += len;
        }
    }
    if (symbols.empty())
        return;

    if (!merge_rank.empty()) {
        const int max_iter = (int)symbols.size();
        for (int iter = 0; iter < max_iter && symbols.size() >= 2; iter++) {
            int best_i = -1;
            int best_rank = INT_MAX;
            for (size_t k = 0; k + 1 < symbols.size(); k++) {
                std::string pair = symbols[k] + " " + symbols[k + 1];
                auto it = merge_rank.find(pair);
                if (it != merge_rank.end() && (int)it->second < best_rank) {
                    best_rank = it->second;
                    best_i = (int)k;
                }
            }
            if (best_i < 0)
                break;
            symbols[best_i] += symbols[best_i + 1];
            symbols.erase(symbols.begin() + best_i + 1);
        }
    }

    for (const auto& s : symbols) {
        auto it = token_to_id.find(s);
        if (it != token_to_id.end()) {
            out.push_back(it->second);
        } else {
            // Per-byte fallback: split into individual codepoints.
            size_t i = 0;
            while (i < s.size()) {
                unsigned char c = (unsigned char)s[i];
                size_t len;
                if (c < 0x80)
                    len = 1;
                else if ((c & 0xE0) == 0xC0)
                    len = 2;
                else if ((c & 0xF0) == 0xE0)
                    len = 3;
                else if ((c & 0xF8) == 0xF0)
                    len = 4;
                else
                    len = 1;
                std::string single(s, i, len);
                auto jt = token_to_id.find(single);
                if (jt != token_to_id.end())
                    out.push_back(jt->second);
                i += len;
            }
        }
    }
}

// Inverse of byte_encoder(): codepoint → original byte. Built lazily on
// first call. The forward map sends each of the 256 raw bytes to a
// printable Unicode codepoint; this reverses it. Codepoints not in the
// table (e.g. real Unicode in synthesised text) are skipped.
inline const std::unordered_map<uint32_t, uint8_t>& byte_decoder() {
    static std::unordered_map<uint32_t, uint8_t> table;
    static bool initialized = false;
    if (initialized)
        return table;
    auto& enc = byte_encoder();
    table.reserve(256);
    for (int b = 0; b < 256; b++) {
        table[(uint32_t)enc[b]] = (uint8_t)b;
    }
    initialized = true;
    return table;
}

// Decode one BPE token's textual form (a UTF-8 string of byte-encoded
// codepoints) back to the raw byte sequence it represents. Codepoints
// that don't appear in the byte_decoder() map are skipped silently —
// this matches HuggingFace's `errors="replace"` decode behaviour for
// the "didn't see this in training" tail.
inline std::string token_bytes_to_utf8(const std::string& token) {
    auto& dec = byte_decoder();
    std::string out;
    out.reserve(token.size());
    size_t i = 0;
    while (i < token.size()) {
        unsigned char c = (unsigned char)token[i];
        uint32_t cp;
        size_t len;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < token.size()) {
            cp = ((c & 0x1F) << 6) | ((unsigned char)token[i + 1] & 0x3F);
            len = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < token.size()) {
            cp =
                ((c & 0x0F) << 12) | (((unsigned char)token[i + 1] & 0x3F) << 6) | ((unsigned char)token[i + 2] & 0x3F);
            len = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < token.size()) {
            cp = ((c & 0x07) << 18) | (((unsigned char)token[i + 1] & 0x3F) << 12) |
                 (((unsigned char)token[i + 2] & 0x3F) << 6) | ((unsigned char)token[i + 3] & 0x3F);
            len = 4;
        } else {
            i++;
            continue;
        }
        i += len;
        auto it = dec.find(cp);
        if (it != dec.end())
            out.push_back((char)it->second);
    }
    return out;
}

// Decode a sequence of BPE token IDs to a UTF-8 string. Pass an
// id_to_token vector (the GGUF `tokenizer.ggml.tokens` table) and the
// id sequence; out-of-range / negative IDs are silently skipped (the
// caller is responsible for special-token filtering).
inline std::string detokenize(const std::vector<std::string>& id_to_token, const int32_t* ids, size_t n) {
    std::string out;
    out.reserve(n * 4);
    for (size_t i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || (size_t)id >= id_to_token.size())
            continue;
        out += token_bytes_to_utf8(id_to_token[(size_t)id]);
    }
    return out;
}

// Whitespace-split pre-tokenizer + BPE merge pass for arbitrary text.
// Pre-tokenization: collect runs of non-whitespace, prepend a leading
// space to all but the first run (matches GPT-2's "treat space as part
// of the token" convention), byte-encode each run, then BPE-merge it.
//
// This is the simple pre-tokenizer good for prompt fragments. Models
// that need full GPT-2 regex pre-tokenization (with letter / number /
// punctuation runs split separately) should call bpe_one directly.
inline std::vector<int32_t> tokenize_simple(const std::unordered_map<std::string, int32_t>& token_to_id,
                                            const std::unordered_map<std::string, int32_t>& merge_rank,
                                            const std::string& text) {
    std::vector<int32_t> result;
    size_t i = 0;
    bool first = true;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n'))
            i++;
        if (i >= text.size())
            break;
        size_t j = i;
        while (j < text.size() && text[j] != ' ' && text[j] != '\t' && text[j] != '\n')
            j++;
        std::string word = text.substr(i, j - i);
        if (!first)
            word = std::string(" ") + word;
        first = false;
        std::string encoded = bytes_to_unicode(word.data(), word.size());
        bpe_one(token_to_id, merge_rank, encoded, result);
        i = j;
    }
    return result;
}

} // namespace core_bpe
