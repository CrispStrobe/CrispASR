#pragma once

// Qwen3-ForcedAligner-Ultra local-head post-processing, based on the
// qwen_fa_ultra 0.1.1 reference at model revision 765dc1d714e4c80196782a5df5bfbcb3e1416736.
// Times and offsets are milliseconds. The optional local head is not yet wired
// into the aligner; this component exists so the port can be tested in stages.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace qwen3_ultra_boundary {
struct Interval {
    double start;
    double end;
};

// Upstream masks classes after the audio end before timestamp repair.
inline int valid_timestamp_classes(int n_samples, int class_count) {
    if (n_samples < 0 || class_count <= 0)
        return 0;
    return std::min(class_count, n_samples / 1280 + 1);
}

inline bool decode_offset(const std::array<float, 13>& logits, double& offset) {
    for (float x : logits)
        if (!std::isfinite(x))
            return false;
    const double maximum = *std::max_element(logits.begin(), logits.end()) / 1.5;
    double weight_sum = 0.0;
    double weighted = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const double weight = std::exp(static_cast<double>(logits[i]) / 1.5 - maximum);
        weight_sum += weight;
        weighted += weight * (static_cast<int>(i) * 10 - 60);
    }
    offset = weighted / weight_sum;
    return std::isfinite(offset);
}

inline bool repair(const std::vector<Interval>& coarse, const std::vector<double>& offsets, double duration_ms,
                   std::vector<Interval>& result) {
    if (!std::isfinite(duration_ms) || duration_ms <= 0.0 || offsets.size() != 2 * coarse.size())
        return false;
    constexpr double min_word = 10.0;
    result.clear();
    result.reserve(coarse.size());
    for (size_t i = 0; i < coarse.size(); ++i) {
        double start = coarse[i].start + offsets[2 * i];
        double end = coarse[i].end + offsets[2 * i + 1];
        if (!std::isfinite(start) || !std::isfinite(end)) {
            start = coarse[i].start;
            end = coarse[i].end;
        } else if (end - start < min_word) {
            const double midpoint = (start + end) / 2.0;
            if (midpoint < min_word / 2.0) {
                start = 0.0;
                end = min_word;
            } else {
                start = midpoint - min_word / 2.0;
                end = midpoint + min_word / 2.0;
            }
        } else if (start < 0.0) {
            end -= start;
            start = 0.0;
        }
        if (!std::isfinite(start) || !std::isfinite(end) || start < 0.0 || end < 0.0 || end - start < min_word) {
            start = 0.0;
            end = min_word;
        }
        result.push_back({start, end});
    }
    for (size_t i = 0; i + 1 < result.size(); ++i) {
        auto& current = result[i];
        auto& next = result[i + 1];
        if (current.end <= next.start)
            continue;
        double cut = next.start;
        const double minimum_cut = current.start + min_word;
        if (cut < minimum_cut) {
            cut = minimum_cut;
            next.start = cut;
            next.end = std::max(next.end, cut + min_word);
        }
        current.end = cut;
    }
    if (result.empty())
        return true;
    const bool in_bounds = std::all_of(result.begin(), result.end(), [duration_ms](const Interval& i) {
        return std::isfinite(i.start) && std::isfinite(i.end) && 0.0 <= i.start && i.start < i.end &&
               i.end <= duration_ms;
    });
    if (in_bounds)
        return true;
    const double minimum = std::min(min_word, duration_ms / result.size());
    double cursor = 0.0;
    for (size_t i = 0; i < result.size(); ++i) {
        auto& item = result[i];
        if (!std::isfinite(item.start) || !std::isfinite(item.end))
            item = {0.0, duration_ms};
        const size_t remaining = result.size() - i - 1;
        const double latest_start = duration_ms - minimum * (remaining + 1);
        item.start = std::min(std::max(item.start, cursor), latest_start);
        const double earliest_end = item.start + minimum;
        const double latest_end = duration_ms - minimum * remaining;
        item.end = std::min(std::max(item.end, earliest_end), latest_end);
        cursor = item.end;
    }
    return true;
}
} // namespace qwen3_ultra_boundary
