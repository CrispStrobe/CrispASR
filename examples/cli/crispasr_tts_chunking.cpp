// crispasr_tts_chunking.cpp — sentence splitter + silence-padded concat.
//
// See crispasr_tts_chunking.h for the design rationale (issue #66).

#include "crispasr_tts_chunking.h"

#include <algorithm>
#include <cstdint>

namespace {

// True if `s[i ..]` begins with a sentence terminator. ASCII .!? are
// single-byte; the two Unicode terminators we recognise are 3-byte
// UTF-8 sequences. Caller guarantees i < s.size().
bool is_terminator_at(const std::string& s, std::size_t i, std::size_t& out_len) {
    char c = s[i];
    if (c == '.' || c == '!' || c == '?') {
        out_len = 1;
        return true;
    }
    if (i + 2 < s.size()) {
        const std::uint8_t b0 = (std::uint8_t)s[i];
        const std::uint8_t b1 = (std::uint8_t)s[i + 1];
        const std::uint8_t b2 = (std::uint8_t)s[i + 2];
        // U+3002 IDEOGRAPHIC FULL STOP: E3 80 82
        if (b0 == 0xE3 && b1 == 0x80 && b2 == 0x82) {
            out_len = 3;
            return true;
        }
        // U+0964 DEVANAGARI DANDA: E0 A5 A4
        if (b0 == 0xE0 && b1 == 0xA5 && b2 == 0xA4) {
            out_len = 3;
            return true;
        }
    }
    return false;
}

// True if position `i` is whitespace or end-of-input. Treats ASCII
// space, tab, newline, carriage return as whitespace; we don't expand
// to Unicode whitespace because the only tokens we care about are
// terminators followed by SOMETHING that ends the sentence visually.
bool is_space_or_end(const std::string& s, std::size_t i) {
    if (i >= s.size())
        return true;
    char c = s[i];
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r'))
        a++;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r'))
        b--;
    return s.substr(a, b - a);
}

// Break a chunk longer than max_chars into pieces along whitespace
// boundaries. If we can't find a whitespace boundary inside the
// window, split mid-word (last resort — protects against pathological
// no-whitespace input like a 10 KB hex dump in `input`).
void whitespace_split(const std::string& chunk, std::size_t max_chars, std::vector<std::string>& out) {
    if (chunk.size() <= max_chars) {
        if (!chunk.empty())
            out.push_back(chunk);
        return;
    }
    std::size_t start = 0;
    while (start < chunk.size()) {
        std::size_t end = std::min(start + max_chars, chunk.size());
        if (end < chunk.size()) {
            // Walk back to a whitespace boundary if one exists in the
            // [start+1, end) window.
            std::size_t back = end;
            while (back > start + 1 && !is_space_or_end(chunk, back))
                back--;
            if (back > start + 1)
                end = back;
        }
        std::string piece = trim(chunk.substr(start, end - start));
        if (!piece.empty())
            out.push_back(piece);
        start = end;
    }
}

} // namespace

std::vector<std::string> crispasr_tts_split_sentences(const std::string& text, std::size_t max_chars) {
    std::vector<std::string> result;
    if (text.empty())
        return result;

    std::size_t chunk_start = 0;
    std::size_t i = 0;
    while (i < text.size()) {
        std::size_t term_len = 0;
        if (is_terminator_at(text, i, term_len)) {
            // Terminator must be followed by whitespace or end-of-input
            // for it to count as a sentence break — otherwise it's a
            // decimal point ("1.5") or in-word punctuation.
            if (is_space_or_end(text, i + term_len)) {
                std::string chunk = trim(text.substr(chunk_start, i + term_len - chunk_start));
                if (!chunk.empty())
                    whitespace_split(chunk, max_chars, result);
                chunk_start = i + term_len;
                i = chunk_start;
                continue;
            }
            i += term_len;
            continue;
        }
        i++;
    }
    // Tail (input that doesn't end with a terminator).
    if (chunk_start < text.size()) {
        std::string tail = trim(text.substr(chunk_start));
        if (!tail.empty())
            whitespace_split(tail, max_chars, result);
    }
    return result;
}

std::vector<std::string> crispasr_tts_plan_chunks_for_backend(const std::string& text, const std::string& backend_name,
                                                              std::size_t max_chars) {
    if (backend_name == "vibevoice")
        return {text};

    std::vector<std::string> result = crispasr_tts_split_sentences(text, max_chars);
    if (result.empty())
        result.push_back(text);
    return result;
}

std::vector<float> crispasr_tts_concat_with_silence(const std::vector<std::vector<float>>& chunks,
                                                    int silence_samples) {
    if (chunks.empty())
        return {};
    if (silence_samples < 0)
        silence_samples = 0;

    std::size_t total = 0;
    for (const auto& c : chunks)
        total += c.size();
    if (chunks.size() > 1)
        total += (chunks.size() - 1) * (std::size_t)silence_samples;

    std::vector<float> out;
    out.reserve(total);
    for (std::size_t i = 0; i < chunks.size(); i++) {
        if (i > 0 && silence_samples > 0)
            out.insert(out.end(), (std::size_t)silence_samples, 0.0f);
        out.insert(out.end(), chunks[i].begin(), chunks[i].end());
    }
    return out;
}
