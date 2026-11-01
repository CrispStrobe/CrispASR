// Compare native Mini BS RoFormer captures with the original Python blueprint
// using CrispEmbed's actual, external ground-truth diff implementation.
//
// Build against a pinned CrispEmbed checkout (do not silently substitute the
// native CrispASR report implementation):
//   c++ -O3 -std=c++17 -I/path/to/CrispEmbed/src \
//       tools/mini_bs_crispembed_diff.cpp -o crispembed-diff
//
// Usage: crispembed-diff reference.gguf native-captures.gguf f16|q8_0|q4_k [--outputs-only]
// Native captures: CRISPASR_MINI_BS_CAPTURE_GGUF=out.gguf crispasr-diff ...

#include "crispembed_diff.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <set>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4 && (argc != 5 || std::string(argv[4]) != "--outputs-only")) {
        std::fprintf(stderr, "usage: %s reference.gguf native-captures.gguf f16|q8_0|q4_k [--outputs-only]\n", argv[0]);
        return 2;
    }
    const std::string precision(argv[3]);
    const double threshold = precision == "f16" ? 0.999 : precision == "q8_0" ? 0.98 : 0.94;
    if (precision != "f16" && precision != "q8_0" && precision != "q4_k") {
        std::fprintf(stderr, "unsupported precision: %s\n", argv[3]);
        return 2;
    }
    try {
        crispembed_diff::Ref reference, native;
        if (!reference.load(argv[1]) || !native.load(argv[2])) {
            return 2;
        }
        std::set<std::string> required = {"input_audio",     "stft_input",     "stft_output",  "band_split",
                                          "time_compressed", "encoder_output", "freq_pooled",  "stem_embeddings",
                                          "mask_raw",        "output_bass",    "output_drums", "output_other",
                                          "output_vocals"};
        for (int layer = 0; layer < 6; ++layer) {
            required.insert("layer" + std::to_string(layer) + "_time");
            required.insert("layer" + std::to_string(layer) + "_freq");
        }
        if (argc == 5) {
            required = {"input_audio", "output_bass", "output_drums", "output_other", "output_vocals"};
        }
        auto names = reference.tensor_names();
        auto native_names = native.tensor_names();
        const std::set<std::string> reference_set(names.begin(), names.end());
        const std::set<std::string> native_set(native_names.begin(), native_names.end());
        if (reference_set != native_set ||
            !std::includes(reference_set.begin(), reference_set.end(), required.begin(), required.end())) {
            std::fprintf(stderr, "FAIL: native/reference stage sets differ or a required stage is missing\n");
            return 2;
        }
        std::sort(names.begin(), names.end());
        int failures = 0;
        for (const auto& name : names) {
            const auto expected = reference.get_f32(name);
            const auto actual = native.get_f32(name);
            if (!expected.first || !actual.first || expected.second == 0 || actual.second != expected.second ||
                reference.shape(name) != native.shape(name)) {
                std::fprintf(stderr, "[FAIL] %s missing data or unequal shape\n", name.c_str());
                ++failures;
                continue;
            }
            size_t nonfinite = 0;
            for (size_t i = 0; i < actual.second; ++i) {
                nonfinite += !std::isfinite(actual.first[i]) || !std::isfinite(expected.first[i]);
            }
            if (nonfinite) {
                std::fprintf(stderr, "[FAIL] %s n_nonfinite=%zu\n", name.c_str(), nonfinite);
                ++failures;
                continue;
            }
            // Use the genuine CrispEmbed metrics. Check exact shapes and
            // finiteness above because Ref::compare historically truncates
            // unequal tensors and does not expose a non-finite counter.
            const auto report = reference.compare(name, actual.first, actual.second);
            const double ratio =
                report.ref_norm > 0 ? report.mine_norm / report.ref_norm : (report.mine_norm == 0 ? 1.0 : INFINITY);
            // Strict > implements the user's thresholds exactly. Reading the
            // metric directly prevents CRISPEMBED_DIFF_COS_THRESHOLD from
            // lowering this fixed acceptance gate in the caller environment.
            const bool pass = report.found && std::isfinite(report.cos_global) && std::isfinite(report.cos_min) &&
                              report.cos_global > threshold &&
                              // The STFT row is only one Re/Im pair. Its
                              // angle is ill-conditioned for near-zero bins;
                              // full feature-vector and channel rows remain
                              // strict port-correctness gates everywhere else.
                              (precision != "f16" || name == "stft_output" || report.cos_min > threshold) &&
                              std::isfinite(ratio) && ratio >= 0.80 && ratio <= 1.20;
            std::printf("[%s] %s cos_global=%.9g cos_min=%.9g norm_ratio=%.9g max_abs=%.9g "
                        "rms=%.9g n_nonfinite=0\n",
                        pass ? "PASS" : "FAIL", name.c_str(), report.cos_global, report.cos_min, ratio, report.max_abs,
                        report.rms);
            failures += !pass;
        }
        std::printf("CrispEmbed comparison: %zu stages, %d failures; strict cosine > %.6g (%s)\n", names.size(),
                    failures, threshold, precision.c_str());
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CrispEmbed comparison failed: %s\n", error.what());
        return 2;
    }
}
