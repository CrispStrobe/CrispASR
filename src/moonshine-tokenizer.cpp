#include "moonshine-tokenizer.h"

#include <cstdio>

static std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return str;
}

static std::string trim(const std::string& str) {
    const char* ws = " \t";
    size_t start = str.find_first_not_of(ws);
    if (start == std::string::npos) {
        return "";
    }
    size_t end = str.find_last_not_of(ws);
    return str.substr(start, end - start + 1);
}

bool moonshine_tokenizer::load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, path);
        return false;
    }

    vocab.clear();

    while (true) {
        int first = fgetc(f);
        if (first == EOF) {
            break;
        }

        if (first == 0x00) {
            // empty token
            vocab.push_back({});
            continue;
        }

        size_t length;
        if (first < 128) {
            length = (size_t)first;
        } else {
            int second = fgetc(f);
            if (second == EOF) {
                fprintf(stderr, "%s: unexpected EOF reading length\n", __func__);
                fclose(f);
                return false;
            }
            length = (size_t)(second * 128) + (size_t)first - 128;
        }

        std::vector<uint8_t> bytes(length);
        if (fread(bytes.data(), 1, length, f) != length) {
            fprintf(stderr, "%s: unexpected EOF reading token bytes\n", __func__);
            fclose(f);
            return false;
        }

        vocab.push_back(std::move(bytes));
    }

    fclose(f);

    if (vocab.empty()) {
        fprintf(stderr, "%s: no tokens found in '%s'\n", __func__, path);
        return false;
    }

    return true;
}

std::string moonshine_tokenizer::tokens_to_text(const std::vector<int32_t>& tokens) const {
    std::vector<uint8_t> result_bytes;

    for (int32_t token : tokens) {
        if (token < 0 || (size_t)token >= vocab.size()) {
            continue;
        }

        const auto& bytes = vocab[token];

        // skip special tokens: <...>
        if (bytes.size() > 2 && bytes.front() == '<' && bytes.back() == '>') {
            continue;
        }

        result_bytes.insert(result_bytes.end(), bytes.begin(), bytes.end());
    }

    std::string text(result_bytes.begin(), result_bytes.end());

    // replace U+2581 (Lower One Eighth Block "▁") with space
    // UTF-8 encoding: 0xE2 0x96 0x81
    text = replace_all(text, "\xE2\x96\x81", " ");

    return trim(text);
}

size_t moonshine_tokenizer::vocab_size() const {
    return vocab.size();
}
