#pragma once

// RE-USE diagnostic path. Native CPU reference, not a production backend.
#include "crispasr_diff.h"
#include "core/ggml_cpu_backend.h"
#include "core/reuse_reference_network.h"
#include "reuse_frontend_profile.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace reuse_diff {
// These observations localize frontend arithmetic; the existing raw phase and
// model comparisons below remain the acceptance gates, including at branch cuts.
inline void diagnose(const crispasr_diff::Ref& ref, const char* name, const std::vector<float>& values,
                     const std::vector<int64_t>& shape) {
    if (!ref.get_f32(name).first)
        return; // Older archives have the complete original acceptance stages.
    if (ref.shape(name) != shape) {
        std::fprintf(stderr, "diagnostic/%s: shape mismatch\n", name);
        return;
    }
    const auto report = ref.compare(name, values.data(), values.size(), crispasr_diff::Ref::COS_FIRST_DIM);
    std::printf("diagnostic/%-22s DIAGNOSTIC rms=%.8g max_abs=%.8g cos_min=%.8g norm_ratio=%.8g\n", name, report.rms,
                report.max_abs, report.cos_min, report.norm_ratio);
}

inline bool compare(const crispasr_diff::Ref& ref, const std::string& name, const std::vector<float>& values,
                    const std::vector<int64_t>& expected_shape, const char* prefix, double tolerance) {
    const auto shape = ref.shape(name);
    if (shape != expected_shape) {
        std::fprintf(stderr, "%s%s: missing tensor or shape mismatch\n", prefix, name.c_str());
        return false;
    }
    const auto report = ref.compare(name, values.data(), values.size(), crispasr_diff::Ref::COS_FIRST_DIM);
    const double relative = report.rms_ref > 1e-12f ? report.rms / report.rms_ref : report.rms;
    const bool pass = report.is_pass(0.999f) && std::isfinite(relative) && relative <= tolerance &&
                      std::isfinite(report.norm_ratio) &&
                      (report.rms_ref <= 1e-12f || (report.norm_ratio >= 0.98f && report.norm_ratio <= 1.02f));
    std::printf("%s%-22s %s rel_rms=%.8g max_abs=%.8g cos_min=%.8g norm_ratio=%.8g\n", prefix, name.c_str(),
                pass ? "PASS" : "FAIL", relative, report.max_abs, report.cos_min, report.norm_ratio);
    return pass;
}

inline int run(const std::string& model_path, const std::string& ref_path) {
    reuse_frontend_profile::Analyze frontend = nullptr;
    const char* profile = nullptr;
    if (!reuse_frontend_profile::select(std::getenv("CRISPASR_REUSE_FRONTEND"), frontend, profile))
        return 2;
#ifdef CRISPASR_REUSE_HAVE_MKL
    if (std::string(profile) == "mkl")
        std::printf("reuse: experimental CPU frontend mkl; %s; parity is reference-environment-specific\n",
                    reuse_frontend_mkl::version().c_str());
#endif
    crispasr_diff::Ref ref;
    if (!ref.load(ref_path))
        return 2;
    const auto audio = ref.get_f32("raw_audio");
    const auto phase = ref.get_f32("frontend_phase");
    if (!audio.first || audio.second <= 320 || audio.second > 16000 || audio.second % 80 || !phase.first) {
        std::fprintf(stderr, "reuse: missing/invalid bounded raw_audio or frontend_phase reference\n");
        return 2;
    }
    std::vector<float> samples(audio.first, audio.first + audio.second);
    if (ref.get_f32("frontend_window").first || ref.get_f32("frontend_real").first ||
        ref.get_f32("frontend_imaginary").first) {
        reuse_frontend::Spectrum spectrum;
        reuse_frontend::Diagnostics observations;
        if (!frontend(samples, 16000, spectrum, &observations))
            return 2;
        diagnose(ref, "frontend_window", observations.window, {640});
        const std::vector<int64_t> shape = {static_cast<int64_t>(spectrum.bins), static_cast<int64_t>(spectrum.frames)};
        diagnose(ref, "frontend_real", observations.real, shape);
        diagnose(ref, "frontend_imaginary", observations.imaginary, shape);
    }
    auto* backend = core_cpu_backend::init();
    if (!backend)
        return 2;
    reuse_model_loader::Model model;
    const bool loaded = reuse_model_loader::load(model_path.c_str(), backend, model);
    ggml_backend_free(backend);
    if (!loaded)
        return 2;
    bool passed = true;
    for (bool conditioned : {false, true}) {
        std::vector<float> output, reference_phase(phase.first, phase.first + phase.second);
        reuse_reference_network::Trace trace;
        if (!reuse_reference_network::run(model, samples, 16000, output, &trace,
                                          conditioned ? &reference_phase : nullptr, frontend) ||
            trace.encoder_layers.size() != 6 || trace.block_time.size() != 30 || trace.blocks.size() != 30)
            return 2;
        const std::string prefix = std::string(profile) == "native" ? (conditioned ? "conditioned/" : "native/")
                                                                    : (conditioned ? "mkl-conditioned/" : "mkl/");
        bool group = true;
        const auto check = [&](const std::string& name, const std::vector<float>& data,
                               const std::vector<int64_t>& shape, double tolerance = 0.002) {
            group = compare(ref, name, data, shape, prefix.c_str(), tolerance) && group;
        };
        const int64_t frames = static_cast<int64_t>(1 + samples.size() / 80), bins = 321;
        check("frontend_magnitude", trace.magnitude, {bins, frames});
        check("frontend_phase", trace.phase, {bins, frames});
        for (size_t i = 0; i < trace.encoder_layers.size(); ++i) {
            const auto& x = trace.encoder_layers[i];
            check("encoder_layer_" + std::to_string(i), x.values,
                  {static_cast<int64_t>(x.freq), static_cast<int64_t>(x.time), static_cast<int64_t>(x.channels)});
        }
        for (size_t i = 0; i < trace.blocks.size(); ++i) {
            const auto& t = trace.block_time[i];
            check("block_time_" + std::to_string(i), t.values,
                  {static_cast<int64_t>(t.freq), static_cast<int64_t>(t.time), static_cast<int64_t>(t.channels)});
            const auto& x = trace.blocks[i];
            check("block_" + std::to_string(i), x.values,
                  {static_cast<int64_t>(x.freq), static_cast<int64_t>(x.time), static_cast<int64_t>(x.channels)});
        }
        check("amplitude", trace.amplitude, {bins, frames});
        check("output_phase", trace.output_phase, {bins, frames});
        check("waveform", output, {static_cast<int64_t>(samples.size())}, 0.005);
        passed = group && passed;
    }
    return passed ? 0 : 1;
}
} // namespace reuse_diff
