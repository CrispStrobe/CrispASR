#include "../src/core/reuse_frontend.h"
#include "../src/core/wav_reader.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

static bool test_raw_endpoint_phase() {
    // Independent real-DFT invariant: DC and even-N Nyquist have +0 imaginary
    // part. A negative real coefficient must therefore give raw +pi, not -pi.
    // Wrapped phase comparisons cannot detect this model-input regression.
    const float pi = static_cast<float>(std::acos(-1.0));
    for (bool alternating : {false, true}) {
        for (float sign : {-1.0f, 1.0f}) {
            std::vector<float> audio(1600);
            for (size_t i = 0; i < audio.size(); ++i)
                audio[i] = sign * 0.25f * (alternating && i % 2 ? -1.0f : 1.0f);
            reuse_frontend::Spectrum observed, plain;
            reuse_frontend::Diagnostics diagnostics;
            if (!reuse_frontend::analyze(audio, 16000, observed, &diagnostics) ||
                !reuse_frontend::analyze(audio, 16000, plain) || observed.phase != plain.phase ||
                observed.magnitude != plain.magnitude || diagnostics.window.size() != 640 ||
                diagnostics.real.size() != observed.phase.size() ||
                diagnostics.imaginary.size() != observed.phase.size())
                return false;
            const size_t bin = alternating ? observed.bins - 1 : 0;
            for (size_t frame = 0; frame < observed.frames; ++frame) {
                const size_t index = frame * observed.bins + bin;
                if (observed.phase[index] != (sign < 0 ? pi : 0.0f) || diagnostics.imaginary[index] != 0.0f ||
                    std::signbit(diagnostics.imaginary[index]) || diagnostics.real[index] * sign <= 1.0f)
                    return false;
            }
        }
    }
    std::puts("RE-USE raw DC/Nyquist phase: 84 exact branch and diagnostic invariants passed");
    return true;
}

static bool compare_reference(const std::vector<float>& audio, int rate, const char* fixture_path, int expected_fft,
                              int expected_hop) {
    reuse_frontend::Spectrum out;
    if (!reuse_frontend::analyze(audio, rate, out) || out.n_fft != expected_fft || out.hop != expected_hop ||
        out.frames != 1 + audio.size() / static_cast<size_t>(expected_hop) ||
        out.bins != static_cast<size_t>(expected_fft / 2 + 1) || out.magnitude.size() != out.frames * out.bins ||
        out.phase.size() != out.frames * out.bins)
        return false;
    FILE* fixture = std::fopen(fixture_path, "rb");
    if (!fixture)
        return false;
    std::vector<float> ref_mag(out.magnitude.size()), ref_phase(out.phase.size());
    for (auto* values : {&ref_mag, &ref_phase}) {
        for (auto& value : *values) {
            unsigned char bytes[4];
            if (std::fread(bytes, 1, 4, fixture) != 4) {
                std::fclose(fixture);
                return false;
            }
            const uint32_t bits = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) | (uint32_t(bytes[2]) << 16) |
                                  (uint32_t(bytes[3]) << 24);
            std::memcpy(&value, &bits, sizeof(value));
        }
    }
    const bool extra = std::fgetc(fixture) != EOF;
    std::fclose(fixture);
    if (extra)
        return false;
    double max_mag = 0.0, max_phase = 0.0, squared = 0.0, baseline = 0.0;
    for (size_t i = 0; i < ref_mag.size(); ++i) {
        max_mag = std::max(max_mag, std::abs(double(out.magnitude[i]) - ref_mag[i]));
        const double diff = double(out.phase[i]) - ref_phase[i];
        if (std::expm1(ref_mag[i]) > 0.01)
            max_phase = std::max(max_phase, std::abs(std::atan2(std::sin(diff), std::cos(diff))));
        const double expected_mag = std::expm1(ref_mag[i]);
        const double actual_mag = std::expm1(out.magnitude[i]);
        const double re = actual_mag * std::cos(out.phase[i]) - expected_mag * std::cos(ref_phase[i]);
        const double im = actual_mag * std::sin(out.phase[i]) - expected_mag * std::sin(ref_phase[i]);
        squared += re * re + im * im;
        baseline += expected_mag * expected_mag;
    }
    const double relative = std::sqrt(squared / baseline);
    std::printf("RE-USE %d Hz frontend: max magnitude error %.9g, max phase error %.9g, relative error %.9g\n", rate,
                max_mag, max_phase, relative);
    return max_mag <= 0.0001 && max_phase <= 0.001 && relative <= 0.00001;
}

static bool test_synthesis(const std::vector<float>& audio, int rate) {
    reuse_frontend::Spectrum spec;
    if (!reuse_frontend::analyze(audio, rate, spec))
        return false;
    const size_t stride = spec.bins + spec.bins % 2;
    std::vector<float> amp(spec.frames * stride), real(amp.size()), imag(amp.size());
    for (size_t t = 0; t < spec.frames; ++t)
        for (size_t f = 0; f < spec.bins; ++f) {
            const size_t in = t * spec.bins + f, out = t * stride + f;
            amp[out] = spec.magnitude[in];
            real[out] = std::cos(spec.phase[in]);
            imag[out] = std::sin(spec.phase[in]);
        }
    std::vector<float> reconstructed;
    if (!reuse_frontend::synthesize(spec, amp, real, imag, audio.size(), reconstructed))
        return false;
    double max_error = 0;
    const size_t count = (spec.frames - 1) * spec.hop;
    for (size_t i = 0; i < count; ++i)
        max_error = std::max(max_error, std::abs(double(audio[i]) - reconstructed[i]));
    for (size_t i = count; i < audio.size(); ++i)
        if (reconstructed[i] != 1e-8f)
            return false;
    std::printf("RE-USE %d Hz synthesis round trip: max error %.9g\n", rate, max_error);
    if (max_error > 0.00001)
        return false;
    std::fill(amp.begin(), amp.end(), -1.0f);
    if (!reuse_frontend::synthesize(spec, amp, real, imag, audio.size(), reconstructed))
        return false;
    for (size_t i = 0; i < count; ++i)
        if (reconstructed[i] != 0.0f)
            return false;
    reconstructed.assign(1, 42.0f);
    amp[0] = std::numeric_limits<float>::quiet_NaN();
    if (reuse_frontend::synthesize(spec, amp, real, imag, audio.size(), reconstructed) || reconstructed.size() != 1 ||
        reconstructed[0] != 42.0f)
        return false;
    amp[0] = 0;
    if (reuse_frontend::synthesize(spec, amp, real, imag, count + spec.hop, reconstructed) ||
        reuse_frontend::synthesize(spec, amp, real, imag, count - 1, reconstructed))
        return false;
    amp.pop_back();
    return !reuse_frontend::synthesize(spec, amp, real, imag, audio.size(), reconstructed);
}

static bool test_synthesis_reference(const char* path) {
    reuse_frontend::Spectrum spec;
    spec.sample_rate = 16000;
    spec.n_fft = 640;
    spec.hop = 80;
    spec.frames = 11;
    spec.bins = 321;
    const size_t stride = 322;
    std::vector<float> amp(spec.frames * stride), real(amp.size()), imag(amp.size());
    for (size_t i = 0; i < amp.size(); ++i) {
        amp[i] = (static_cast<int>(i % 29) - 3) / 50.0f;
        if (i / stride == 3)
            amp[i] = -1.0f;
        const float phase = (static_cast<int>(i % 101) - 50) / 20.0f;
        real[i] = std::cos(phase);
        imag[i] = std::sin(phase);
    }
    std::vector<float> output;
    if (!reuse_frontend::synthesize(spec, amp, real, imag, 803, output))
        return false;
    FILE* fixture = std::fopen(path, "rb");
    if (!fixture)
        return false;
    double max_error = 0;
    for (float actual : output) {
        unsigned char bytes[4];
        if (std::fread(bytes, 1, 4, fixture) != 4) {
            std::fclose(fixture);
            return false;
        }
        const uint32_t bits =
            uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) | (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
        float expected;
        std::memcpy(&expected, &bits, sizeof(expected));
        max_error = std::max(max_error, std::abs(double(actual) - expected));
    }
    const bool extra = std::fgetc(fixture) != EOF;
    std::fclose(fixture);
    std::printf("RE-USE synthesis PyTorch reference: max error %.9g\n", max_error);
    return !extra && max_error <= 0.00001;
}

int main(int argc, char** argv) {
    if (argc != 5)
        return 1;
    if (!test_raw_endpoint_phase())
        return 12;
    if (!test_synthesis_reference(argv[4]))
        return 11;
    std::vector<float> audio;
    int rate = 0;
    if (!crispasr::core::read_wav_mono_pcm16(argv[1], audio, rate) || rate != 16000 || audio.size() < 1600)
        return 2;
    audio.resize(1600);
    if (!compare_reference(audio, rate, argv[2], 640, 80))
        return 3;
    if (!test_synthesis(audio, rate))
        return 9;
    std::vector<float> synthetic(4400);
    for (int i = 0; i < static_cast<int>(synthetic.size()); ++i)
        synthetic[i] = static_cast<float>((i * 37) % 2001 - 1000) / 32768.0f;
    if (!compare_reference(synthetic, 44100, argv[3], 1764, 220))
        return 4;
    synthetic.push_back(0.0f); // Exercise the upstream sub-hop tail padding.
    if (!test_synthesis(synthetic, 44100))
        return 10;
    reuse_frontend::Spectrum untouched;
    untouched.sample_rate = 123;
    if (reuse_frontend::analyze(audio, 7999, untouched) || untouched.sample_rate != 123 ||
        reuse_frontend::analyze(std::vector<float>(320), 16000, untouched))
        return 5;
    audio[0] = std::numeric_limits<float>::quiet_NaN();
    if (reuse_frontend::analyze(audio, 16000, untouched))
        return 6;
    std::vector<float> silence(2000, 0.0f);
    reuse_frontend::Spectrum out;
    if (!reuse_frontend::analyze(silence, 44100, out) || out.n_fft != 1764 || out.hop != 220)
        return 7;
    for (float x : out.magnitude)
        if (x != 0.0f)
            return 8;
    return 0;
}
