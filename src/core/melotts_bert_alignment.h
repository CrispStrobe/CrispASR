#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace core_melotts {

// Original Melo distributes raw phones across WordPieces BEFORE interspersing
// blanks. g2p_word2ph already includes blanks, so undo the factor of two first;
// preserve the leading extra blank and both padding-token counts separately.
inline bool bert_phone_counts(const std::vector<int>& g2p_word2ph, const std::vector<int>& subwords, int n_tokens,
                              int n_phones, std::vector<int>& result) {
    result.clear();
    if (n_tokens < 2 || n_phones < 0 || g2p_word2ph.size() != subwords.size() + 2)
        return false;
    if (g2p_word2ph.front() < 0 || g2p_word2ph.back() < 0)
        return false;
    result.push_back(g2p_word2ph.front());
    int64_t total = g2p_word2ph.front();
    for (std::size_t i = 0; i < subwords.size(); ++i) {
        const int doubled = g2p_word2ph[i + 1];
        const int count = subwords[i];
        if (doubled < 0 || doubled % 2 != 0 || count <= 0 || count > n_tokens ||
            result.size() + static_cast<std::size_t>(count) >= static_cast<std::size_t>(n_tokens)) {
            result.clear();
            return false;
        }
        const int phones = doubled / 2;
        for (int j = 0; j < count; ++j)
            result.push_back(2 * (phones / count + (j < phones % count ? 1 : 0)));
        total += doubled;
    }
    result.push_back(g2p_word2ph.back());
    total += g2p_word2ph.back();
    if (result.size() != static_cast<std::size_t>(n_tokens) || total != n_phones) {
        result.clear();
        return false;
    }
    return true;
}

} // namespace core_melotts
