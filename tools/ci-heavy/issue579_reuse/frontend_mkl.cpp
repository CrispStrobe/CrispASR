// Optional RE-USE research frontend; public oneMKL VML/DFTI interfaces only.
#include "../../../src/core/reuse_frontend_mkl.h"
#include <mkl.h>
#include <cstdio>
#include <stdexcept>

namespace reuse_frontend_mkl {
namespace {
constexpr int fft_size = 640, hop = 80, bins = 321;
struct ThreadScope {
    int previous = MKL_Set_Num_Threads_Local(2);
    ~ThreadScope() { MKL_Set_Num_Threads_Local(previous); }
};
} // namespace
void check(MKL_LONG status) {
    if (status)
        throw std::runtime_error(DftiErrorMessage(status));
}

struct Plan {
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;
    ~Plan() {
        if (handle)
            DftiFreeDescriptor(&handle);
    }
    Plan() = default;
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
};

Analysis transform(const std::vector<float>& audio, Window mode, bool batch) {
    ThreadScope threads;
    if (audio.size() < 400 || audio.size() > 16000 || audio.size() % hop)
        throw std::runtime_error("expected 400..16000 samples, divisible by 80");
    for (float sample : audio)
        if (!std::isfinite(sample))
            throw std::runtime_error("nonfinite audio");
    const size_t frames = 1 + audio.size() / hop, count = frames * bins;
    Analysis result;
    result.window.resize(fft_size);
    result.real.resize(count);
    result.imaginary.resize(count);
    result.magnitude.resize(count);
    result.phase.resize(count);
    std::vector<float> angles(fft_size), cosine(fft_size), input(frames * fft_size), output(2 * count);
    const float step = static_cast<float>(2.0 * std::acos(-1.0) / fft_size);
    for (int i = 0; i < fft_size; ++i)
        angles[i] = step * static_cast<float>(i);
    if (mode == Window::Libm) {
        for (int i = 0; i < fft_size; ++i)
            cosine[i] = std::cos(angles[i]);
    } else {
        const MKL_INT64 accuracy = mode == Window::HighAccuracy  ? VML_HA
                                   : mode == Window::LowAccuracy ? VML_LA
                                                                 : VML_EP;
        vmsCos(fft_size, angles.data(), cosine.data(), accuracy);
    }
    for (int i = 0; i < fft_size; ++i)
        result.window[i] = 0.5f - 0.5f * cosine[i];
    for (size_t frame = 0; frame < frames; ++frame) {
        for (int i = 0; i < fft_size; ++i) {
            int64_t index = static_cast<int64_t>(frame * hop) + i - fft_size / 2;
            if (index < 0)
                index = -index;
            else if (index >= static_cast<int64_t>(audio.size()))
                index = 2 * static_cast<int64_t>(audio.size()) - 2 - index;
            input[frame * fft_size + i] = audio[index] * result.window[i];
        }
    }
    Plan plan;
    check(DftiCreateDescriptor(&plan.handle, DFTI_SINGLE, DFTI_REAL, 1, static_cast<MKL_LONG>(fft_size)));
    check(DftiSetValue(plan.handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    check(DftiSetValue(plan.handle, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(batch ? frames : 1)));
    check(DftiSetValue(plan.handle, DFTI_INPUT_DISTANCE, static_cast<MKL_LONG>(fft_size)));
    check(DftiSetValue(plan.handle, DFTI_OUTPUT_DISTANCE, static_cast<MKL_LONG>(bins)));
    MKL_LONG stride[2] = {0, 1};
    check(DftiSetValue(plan.handle, DFTI_INPUT_STRIDES, stride));
    check(DftiSetValue(plan.handle, DFTI_OUTPUT_STRIDES, stride));
    check(DftiSetValue(plan.handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    check(DftiCommitDescriptor(plan.handle));
    if (batch)
        check(DftiComputeForward(plan.handle, input.data(), output.data()));
    else
        for (size_t frame = 0; frame < frames; ++frame)
            check(DftiComputeForward(plan.handle, input.data() + frame * fft_size, output.data() + 2 * frame * bins));
    for (size_t i = 0; i < count; ++i) {
        result.real[i] = output[2 * i];
        result.imaginary[i] = output[2 * i + 1];
        result.magnitude[i] = std::log1p(std::hypot(result.real[i], result.imaginary[i]));
        result.phase[i] = std::atan2(result.imaginary[i], result.real[i]);
    }
    return result;
}


std::string version() {
    char value[256] = {};
    MKL_Get_Version_String(value, sizeof(value));
    return value;
}
bool analyze(const std::vector<float>& audio, int sample_rate, reuse_frontend::Spectrum& output,
             reuse_frontend::Diagnostics* diagnostics) {
    if (sample_rate != 16000)
        return false;
    try {
        auto value = transform(audio);
        reuse_frontend::Spectrum result;
        result.sample_rate = sample_rate;
        result.n_fft = fft_size;
        result.hop = hop;
        result.frames = 1 + audio.size() / hop;
        result.bins = bins;
        result.magnitude = std::move(value.magnitude);
        result.phase = std::move(value.phase);
        if (diagnostics) {
            reuse_frontend::Diagnostics observed;
            observed.window = std::move(value.window);
            observed.real = std::move(value.real);
            observed.imaginary = std::move(value.imaginary);
            *diagnostics = std::move(observed);
        }
        output = std::move(result);
        return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "reuse: oneMKL frontend failed: %s\n", error.what());
        return false;
    }
}
} // namespace reuse_frontend_mkl
