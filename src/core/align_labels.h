#pragma once
#include "uroman.h"
#include <string>
#include <vector>

namespace core_align_labels {
// Preserve a script which the loaded CTC vocabulary actually supports.
// Romanization remains a fallback for a Latin-only aligner; those transformed
// labels cannot be presented as measured original-script character spans.
inline std::string for_vocab(const std::string& word, const std::vector<std::string>& vocab, bool romanize) {
    if (!romanize || !core_uroman::needs_romanization(word))
        return word;
    for (const auto& token : vocab) {
        if (!token.empty() && static_cast<unsigned char>(token[0]) >= 0x80 && word.find(token) != std::string::npos)
            return word;
    }
    return core_uroman::romanize(word);
}
} // namespace core_align_labels
