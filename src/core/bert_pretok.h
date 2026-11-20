// src/core/bert_pretok.h — HuggingFace BertNormalizer + BertPreTokenizer.
//
// The historical WordPiece pre-tokenization in src/tokenizer.cpp is per-BYTE
// `isspace`/`ispunct`, which cannot match HF on anything outside ASCII: CJK
// runs stay glued together (HF spaces every ideograph apart), typographic
// quotes and dashes ride along inside the word, an NBSP is a letter and a
// soft hyphen survives. That approximation is FROZEN for every shipped GGUF —
// this path is selected only by `tokenizer.ggml.pre = "bert"`, which the
// converter writes when tokenizer.json itself declares
// BertNormalizer/BertPreTokenizer without lowercasing or accent stripping
// (the LaBSE class). An absent key keeps the historical behavior exactly.
//
// What HF does, in order (tokenizers/src/normalizers/bert.rs + pre_tokenizers/
// bert.rs), and what this header reproduces:
//   1. clean_text: drop U+0000, U+FFFD and control/format chars (Cc, Cf —
//      NOT \t \n \r); map every whitespace char to a plain space.
//   2. handle_chinese_chars: wrap every CJK ideograph in spaces, so each
//      ideograph becomes its own word (kana/Hangul are NOT ideographs and
//      stay in their word).
//   3. Whitespace split (delimiter removed).
//   4. Punctuation split, delimiter ISOLATED as its own word. HF
//      `is_bert_punc` = ASCII punctuation (which includes the ASCII symbols
//      $ + < = > ^ ` | ~) OR Unicode \p{P}. Unicode SYMBOLS are not split:
//      a currency sign stays attached to its number.
//
// Hermetic goldens for this file live in tests/test_bert_pretokenize.cpp,
// captured from HF's own normalizer/pre-tokenizer output for
// sentence-transformers/LaBSE.

#pragma once

#include "unicode_categ.h"

#include <string>
#include <vector>

namespace core_bert {

// BERT's fixed CJK-ideograph ranges (BasicTokenizer._is_chinese_char — a
// hardcoded list in both the Python and Rust implementations, NOT a Unicode
// category query; kana and Hangul are deliberately outside it).
inline bool is_cjk_ideograph(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

// HF is_bert_punc: all ASCII punctuation AND symbols, plus Unicode \p{P}
// (punctuation proper — Unicode symbols S* excluded).
inline bool is_bert_punct(uint32_t cp) {
    if (cp < 0x80)
        return core_unicode::category(cp) == core_unicode::CAT_P;
    return core_unicode::is_punct_nonascii(cp);
}

inline void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

// BertNormalizer(clean_text=true, handle_chinese_chars=true, lowercase=false,
// strip_accents=false) + BertPreTokenizer. Returns the pre-token words; the
// caller runs WordPiece over each.
//
// `out_space_before`, when non-null, receives one flag per returned word: 1
// when whitespace separated it from the previous word in `text`. Callers that
// rebuild the original text from the words (fireredpunc) need it to keep the
// input's word spacing; the pre-tokenizer itself discards it.
inline std::vector<std::string> pretokenize(const std::string& text, std::vector<uint8_t>* out_space_before = nullptr) {
    std::vector<std::string> words;
    std::string cur;
    bool pending_space = false;
    auto push = [&](std::string w) {
        words.push_back(std::move(w));
        if (out_space_before)
            out_space_before->push_back(pending_space ? 1 : 0);
        pending_space = false;
    };
    auto flush = [&]() {
        if (!cur.empty()) {
            push(cur);
            cur.clear();
        }
    };
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        const uint32_t cp = core_unicode::utf8_next(text.data(), n, i);
        if (cp == 0 || cp == 0xFFFD)
            continue; // clean_text: dropped outright
        const uint8_t cat = core_unicode::category(cp);
        if (cat == core_unicode::CAT_C)
            continue;                      // control/format (soft hyphen, ZWJ, ...)
        if (cat == core_unicode::CAT_WS) { // every whitespace -> word break
            flush();
            pending_space = true;
            continue;
        }
        if (is_cjk_ideograph(cp) || is_bert_punct(cp)) { // isolated single-cp word
            flush();
            std::string one;
            append_utf8(one, cp);
            push(one);
            continue;
        }
        append_utf8(cur, cp);
    }
    flush();
    return words;
}

} // namespace core_bert
