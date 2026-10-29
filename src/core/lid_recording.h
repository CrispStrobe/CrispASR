// Opt-in recording-wide language detection. Each classifier call sees at most
// 15 seconds, so model-specific input limits and the normal short-clip path
// stay unchanged. The callback is the same LID dispatch used for one window.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace core_lid {

struct WindowResult {
    std::string lang_code;
    float confidence = -1.0f;
    std::string source;
};

template <typename Detect>
bool detect_recording(const float* samples, int n_samples, Detect&& detect, WindowResult& out) {
    constexpr int kWindowSamples = 16000 * 15;
    constexpr float kSilencePeak = 1e-4f;
    if (!samples || n_samples <= 0)
        return false;

    struct Vote {
        std::string lang_code;
        std::string source;
        double duration = 0.0;
        double confidence_sum = 0.0;
    };
    std::vector<Vote> votes;
    double total_duration = 0.0;
    for (int start = 0; start < n_samples;) {
        const int count = std::min(kWindowSamples, n_samples - start);
        float peak = 0.0f;
        for (int i = 0; i < count; ++i)
            peak = std::max(peak, std::abs(samples[start + i]));
        if (peak >= kSilencePeak) {
            // A tiny final fragment is weak evidence for several providers.
            // Give it up to 15 s of context while counting only its own
            // duration in the recording-wide vote.
            const bool short_tail = count < 5 * 16000 && start > 0;
            const int window_start = short_tail ? std::max(0, n_samples - kWindowSamples) : start;
            const int window_count = short_tail ? n_samples - window_start : count;
            WindowResult result;
            if (!detect(samples + window_start, window_count, result) || result.lang_code.empty() ||
                !std::isfinite(result.confidence) || result.confidence < 0.0f || result.confidence > 1.0f)
                return false;
            auto it = std::find_if(votes.begin(), votes.end(),
                                   [&](const Vote& vote) { return vote.lang_code == result.lang_code; });
            if (it == votes.end()) {
                votes.push_back({result.lang_code, result.source, 0.0, 0.0});
                it = votes.end() - 1;
            } else if (it->source != result.source) {
                it->source = "mixed";
            }
            // Provider confidences are not comparable: Whisper, Silero and
            // ECAPA report posteriors, while FireRed and sherpa report only
            // an argmax (represented as 1.0). Choose by non-silent window
            // duration; an equal-duration tie keeps the first language seen.
            it->duration += count;
            it->confidence_sum += static_cast<double>(count) * result.confidence;
            total_duration += count;
        }
        start += count;
    }
    if (votes.empty())
        return false;
    const auto best = std::max_element(votes.begin(), votes.end(),
                                       [](const Vote& a, const Vote& b) { return a.duration < b.duration; });
    out.lang_code = best->lang_code;
    // An uncalibrated support score for the dominant language, bounded [0,1].
    out.confidence = static_cast<float>(best->confidence_sum / total_duration);
    out.source = best->source;
    return true;
}

} // namespace core_lid
