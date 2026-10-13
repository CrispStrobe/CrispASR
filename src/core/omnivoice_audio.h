#pragma once
// Native mono/24-kHz port of k2-fsa/OmniVoice audio/text utilities at
// 08be0b4ccbac3e13e374e86fbfead4b4cac343e2 (Apache-2.0, Xiaomi Corp.).
// Silence detection deliberately uses pydub's PCM16 integer RMS, sliding
// windows and overlapping keep-silence ranges, rather than frame classification.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace core_omnivoice_audio {
using Range = std::pair<int, int>;
inline int milliseconds(size_t samples) {
    // Python round(): nearest integer with ties to even.
    const size_t q = samples / 24, r = samples % 24;
    return (int)(q + (r > 12 || (r == 12 && (q & 1))));
}
inline std::vector<int16_t> slice(const std::vector<int16_t>& pcm, int start, int end) {
    const int len = milliseconds(pcm.size());
    start = std::max(0, std::min(start, len));
    end = std::max(start, std::min(end, len));
    std::vector<int16_t> out((size_t)(end - start) * 24, 0);
    const size_t begin = std::min(pcm.size(), (size_t)start * 24);
    const size_t count = std::min(out.size(), pcm.size() - begin);
    std::copy_n(pcm.begin() + begin, count, out.begin());
    return out;
}
inline std::vector<float> remove_silence(const std::vector<float>& input, int mid, int lead, int trail) {
    std::vector<int16_t> pcm;
    pcm.reserve(input.size());
    for (float v : input) {
        const float scaled = v * 32768.f;
        pcm.push_back((int16_t)std::max(-32768.f, std::min(32767.f, scaled)));
    }
    const int len = milliseconds(pcm.size());
    std::vector<uint64_t> squares(pcm.size() + 1, 0);
    for (size_t i = 0; i < pcm.size(); ++i)
        squares[i + 1] = squares[i] + (int64_t)pcm[i] * pcm[i];
    auto silent = [&](int start, int end) {
        const size_t first = std::min(pcm.size(), (size_t)start * 24);
        const size_t last = std::min(pcm.size(), (size_t)end * 24);
        const size_t count = (size_t)(end - start) * 24;
        // pydub audioop.rms truncates to integer; -50 dBFS == 103.6215.
        return count == 0 || std::floor(std::sqrt((double)(squares[last] - squares[first]) / count)) <= 103;
    };
    std::vector<Range> silence;
    if (mid > 0 && len >= mid) {
        int begin = -1, previous = -1;
        const int last = len - mid;
        for (int start = 0;; start = std::min(start + 10, last)) {
            if (silent(start, start + mid)) {
                if (begin < 0)
                    begin = start;
                else if (start != previous + 10 && start > previous + mid) {
                    silence.emplace_back(begin, previous + mid);
                    begin = start;
                }
                previous = start;
            }
            if (start == last)
                break;
        }
        if (begin >= 0)
            silence.emplace_back(begin, previous + mid);
    }
    if (mid > 0) {
        std::vector<Range> ranges;
        if (silence.empty()) {
            ranges.emplace_back(-mid, len + mid);
        } else {
            int previous = 0;
            for (const auto& gap : silence) {
                if (gap.first > previous)
                    ranges.emplace_back(previous - mid, gap.first + mid);
                previous = gap.second;
            }
            if (previous != len)
                ranges.emplace_back(previous - mid, len + mid);
        }
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first < ranges[i - 1].second) {
                const int boundary = (ranges[i].first + ranges[i - 1].second) / 2;
                ranges[i - 1].second = ranges[i].first = boundary;
            }
        }
        std::vector<int16_t> joined;
        for (const auto& range : ranges) {
            auto part = slice(pcm, range.first, range.second);
            joined.insert(joined.end(), part.begin(), part.end());
        }
        pcm.swap(joined);
    }
    auto trim = [&](int keep) {
        const int duration = milliseconds(pcm.size());
        int start = 0;
        while (start < duration) {
            auto chunk = slice(pcm, start, std::min(start + 10, duration));
            uint64_t sum = 0;
            for (int64_t v : chunk)
                sum += v * v;
            if (!chunk.empty() && std::floor(std::sqrt((double)sum / chunk.size())) > 103)
                break;
            start += 10;
        }
        start = std::max(0, std::min(start, duration) - keep);
        pcm = slice(pcm, start, duration);
    };
    trim(lead);
    std::reverse(pcm.begin(), pcm.end());
    trim(trail);
    std::reverse(pcm.begin(), pcm.end());
    std::vector<float> out(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i)
        out[i] = (float)pcm[i] / 32768.f;
    return out;
}
inline void fade_and_pad(std::vector<float>& pcm, float pad_s, float fade_s) {
    if (pcm.empty())
        return;
    const size_t k = std::min((size_t)(fade_s * 24000), pcm.size() / 2);
    if (k) {
        const double step = k > 1 ? 1.0 / (k - 1) : 0.0;
        for (size_t i = 0; i < k; ++i) {
            pcm[i] *= (float)(i * step);
            pcm[pcm.size() - k + i] *= (float)(1.0 - i * step);
        }
        if (k > 1) {
            pcm.back() = 0.f;
        }
    }
    const size_t pad = (size_t)(pad_s * 24000);
    pcm.insert(pcm.begin(), pad, 0.f);
    pcm.insert(pcm.end(), pad, 0.f);
}
// Decode valid UTF-8; invalid bytes remain in the returned original text.
inline uint32_t codepoint(const std::string& text, size_t& pos) {
    unsigned char c = text[pos++];
    int n = c < 0x80 ? 0 : (c & 0xe0) == 0xc0 ? 1 : (c & 0xf0) == 0xe0 ? 2 : (c & 0xf8) == 0xf0 ? 3 : 0;
    uint32_t cp = n ? c & ((1u << (6 - n)) - 1) : c;
    while (n-- && pos < text.size()) {
        const unsigned char next = text[pos];
        if ((next & 0xc0) != 0x80)
            break;
        cp = (cp << 6) | (next & 0x3f);
        ++pos;
    }
    return cp;
}
inline bool whitespace(uint32_t cp) {
    return (cp >= 9 && cp <= 13) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f ||
           cp == 0x3000;
}
inline std::string add_punctuation(const std::string& text) {
    size_t begin = text.size(), end = 0, pos = 0;
    uint32_t last = 0;
    bool chinese = false;
    while (pos < text.size()) {
        const size_t start = pos;
        const uint32_t cp = codepoint(text, pos);
        chinese = chinese || (cp >= 0x4e00 && cp <= 0x9fff);
        if (!whitespace(cp)) {
            begin = std::min(begin, start);
            end = pos;
            last = cp;
        }
    }
    if (end == 0)
        return {};
    const std::string terminals = ";:,.!?…)]}\"'“”‘’；：，。！？、）】";
    bool terminal = false;
    for (pos = 0; pos < terminals.size();)
        terminal = codepoint(terminals, pos) == last || terminal;
    return text.substr(begin, end - begin) + (terminal ? "" : chinese ? "。" : ".");
}
} // namespace core_omnivoice_audio
