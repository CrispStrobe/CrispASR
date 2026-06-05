#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct moonshine_tokenizer {
    std::vector<std::vector<uint8_t>> vocab;

    bool load(const char* path);
    std::string tokens_to_text(const std::vector<int32_t>& tokens) const;
    // Detokenise a single id to its raw piece (no trim, no special-token strip).
    // Returns empty string for out-of-range / special-token (`<...>`) ids.
    // U+2581 ("▁") is converted to a leading space so the piece is renderable
    // alongside others without further joining.
    std::string token_to_piece(int32_t token) const;
    size_t vocab_size() const;
};
