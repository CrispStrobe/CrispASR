#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct moonshine_tokenizer {
    std::vector<std::vector<uint8_t>> vocab;

    bool load(const char* path);
    std::string tokens_to_text(const std::vector<int32_t>& tokens) const;
    size_t vocab_size() const;
};
