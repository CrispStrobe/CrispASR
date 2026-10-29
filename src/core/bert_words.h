#pragma once

#include "bert_norm.h"
#include "bert_pretok.h"

namespace core_bert {

// Share word boundaries between BERT and the phoneme-conditioning caller.
inline std::vector<std::string> uncased_words(const std::string& text) {
    std::vector<std::string> result;
    for (const auto& word : pretokenize(text)) {
        auto normalized = lower_strip_accents(word);
        if (!normalized.empty())
            result.push_back(normalized);
    }
    return result;
}

} // namespace core_bert
