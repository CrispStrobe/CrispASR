#pragma once
// Qwen tokenization for base VibeVoice TTS prompts, including older GGUFs
// without merge metadata. Keep whitespace and punctuation before applying BPE.
#include "bpe.h"

namespace core_vibevoice_tts {
inline void bpe_one_vocab_rank(const std::unordered_map<std::string, int32_t>& token_to_id, const std::string& word,
                               std::vector<int32_t>& out) {
    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < word.size()) {
        unsigned char c = (unsigned char)word[i];
        size_t len = 1;
        if ((c & 0xE0) == 0xC0)
            len = 2;
        else if ((c & 0xF0) == 0xE0)
            len = 3;
        else if ((c & 0xF8) == 0xF0)
            len = 4;
        if (i + len > word.size())
            len = 1;
        symbols.emplace_back(word, i, len);
        i += len;
    }

    while (symbols.size() >= 2) {
        int best_i = -1;
        int32_t best_id = INT32_MAX;
        for (size_t k = 0; k + 1 < symbols.size(); k++) {
            std::string merged = symbols[k] + symbols[k + 1];
            auto it = token_to_id.find(merged);
            if (it != token_to_id.end() && it->second < best_id) {
                best_id = it->second;
                best_i = (int)k;
            }
        }
        if (best_i < 0)
            break;
        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }

    for (const auto& s : symbols) {
        auto it = token_to_id.find(s);
        if (it != token_to_id.end()) {
            out.push_back(it->second);
            continue;
        }
        size_t j = 0;
        while (j < s.size()) {
            unsigned char c = (unsigned char)s[j];
            size_t len = 1;
            if ((c & 0xE0) == 0xC0)
                len = 2;
            else if ((c & 0xF0) == 0xE0)
                len = 3;
            else if ((c & 0xF8) == 0xF0)
                len = 4;
            if (j + len > s.size())
                len = 1;
            auto jt = token_to_id.find(std::string(s, j, len));
            if (jt != token_to_id.end())
                out.push_back(jt->second);
            j += len;
        }
    }
}

// Processor _parse_script strips the supplied sentence and restores exactly
// one separator space after the speaker colon and one terminal newline.
inline std::string speaker_line(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n\v\f");
    const auto last = text.find_last_not_of(" \t\r\n\v\f");
    return " Speaker 0: " + (first == std::string::npos ? std::string() : text.substr(first, last - first + 1)) + "\n";
}

inline std::vector<int32_t> tokenize(const std::unordered_map<std::string, int32_t>& vocab,
                                     const std::unordered_map<std::string, int32_t>& merges, const std::string& text) {
    if (!merges.empty())
        return core_bpe::tokenize_qwen(vocab, merges, text);
    std::vector<int32_t> result;
    for (const auto& piece : core_bpe::qwen_pretokenize(text)) {
        const auto encoded = core_bpe::bytes_to_unicode(piece.data(), piece.size());
        bpe_one_vocab_rank(vocab, encoded, result);
    }
    return result;
}
} // namespace core_vibevoice_tts
