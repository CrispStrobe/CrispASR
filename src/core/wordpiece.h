#pragma once

// Shared BERT WordPiece tokenizer for CrispASR backends.
//
// Used by bark_tts (bert-base-multilingual-cased) and fireredpunc
// (chinese-bert-wwm-ext).  Supports both cased and uncased variants.
//
// Usage:
//   core_wordpiece::Tokenizer tok;
//   tok.id_to_token = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
//   tok.build_map();
//   std::vector<int32_t> ids = tok.tokenize("Hello world");

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace core_wordpiece {

struct Tokenizer {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;
    int unk_id = 100;      // [UNK]
    int cls_id = 101;      // [CLS]
    int sep_id = 102;      // [SEP]
    int pad_id = 0;        // [PAD]
    bool do_lower = false; // false for cased BERT variants
    bool loaded = false;

    void build_map() {
        token_to_id.clear();
        for (int i = 0; i < (int)id_to_token.size(); i++) {
            if (!id_to_token[(size_t)i].empty())
                token_to_id[id_to_token[(size_t)i]] = i;
        }
        // Auto-detect cased vs uncased: if vocab has "Hello" it's cased
        if (token_to_id.find("Hello") == token_to_id.end() && token_to_id.find("hello") != token_to_id.end()) {
            do_lower = true;
        }
        loaded = !id_to_token.empty();
    }

    int lookup(const std::string& tok) const {
        auto it = token_to_id.find(tok);
        return it != token_to_id.end() ? it->second : unk_id;
    }

    // BERT BasicTokenizer + WordPiece.
    // Returns token IDs (no [CLS]/[SEP] wrapping — caller adds if needed).
    std::vector<int32_t> tokenize(const std::string& text) const {
        std::vector<int32_t> ids;

        // Split on whitespace and punctuation (BERT BasicTokenizer)
        std::vector<std::string> words;
        std::string cur;
        for (size_t i = 0; i < text.size();) {
            uint8_t c = (uint8_t)text[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (!cur.empty()) {
                    words.push_back(cur);
                    cur.clear();
                }
                i++;
            } else if ((c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) || (c >= 0x5B && c <= 0x60) ||
                       (c >= 0x7B && c <= 0x7E)) {
                // ASCII punctuation — split as its own token
                if (!cur.empty()) {
                    words.push_back(cur);
                    cur.clear();
                }
                cur += (char)c;
                words.push_back(cur);
                cur.clear();
                i++;
            } else {
                char ch = (char)c;
                if (do_lower && ch >= 'A' && ch <= 'Z')
                    ch = ch - 'A' + 'a';
                cur += ch;
                i++;
                // Consume UTF-8 continuation bytes
                while (i < text.size() && ((uint8_t)text[i] & 0xC0) == 0x80) {
                    cur += text[i];
                    i++;
                }
            }
        }
        if (!cur.empty())
            words.push_back(cur);

        // WordPiece each word
        for (const auto& word : words) {
            auto it = token_to_id.find(word);
            if (it != token_to_id.end()) {
                ids.push_back(it->second);
                continue;
            }
            // Greedy longest-match subword
            size_t start = 0;
            bool is_bad = false;
            while (start < word.size()) {
                size_t end = word.size();
                int best_id = -1;
                while (end > start) {
                    std::string sub = (start == 0) ? word.substr(0, end) : ("##" + word.substr(start, end - start));
                    auto sit = token_to_id.find(sub);
                    if (sit != token_to_id.end()) {
                        best_id = sit->second;
                        break;
                    }
                    // Back up to previous UTF-8 char boundary
                    end--;
                    while (end > start && ((uint8_t)word[end] & 0xC0) == 0x80)
                        end--;
                }
                if (best_id < 0) {
                    is_bad = true;
                    break;
                }
                ids.push_back(best_id);
                start = end;
            }
            if (is_bad) {
                ids.push_back(unk_id);
            }
        }
        return ids;
    }
};

} // namespace core_wordpiece
