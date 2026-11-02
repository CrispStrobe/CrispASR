// FRCRN_SE_16K (frcrn-se) live integration test.
//
// Requires CRISPASR_MODEL_FRCRN_SE pointing at a GGUF converted with
// models/convert-frcrn-se-to-gguf.py. SKIPs cleanly when unset.
//
// Drives the shared C ABI (crispasr_enhance_audio_spec / Session ABI) that the
// CLI --enhance, the server and every wrapper use — not frcrn_se.h directly.
// Bit-level parity against PyTorch is tools/frcrn_se_parity.py; this pins the
// contracts: same length at any rate, finite output, real denoising on speech,
// the segmented (>120 s) path, arch detection, and that the GGUF is rejected
// as a transcription model.

#include <catch2/catch_test_macros.hpp>

#include "crispasr.h"
#include "crispasr_session.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef CRISPASR_SOURCE_DIR
#define CRISPASR_SOURCE_DIR "."
#endif

namespace {

// The env script points at a conventional path; a missing file skips.
const char* model_path() {
    const char* m = std::getenv("CRISPASR_MODEL_FRCRN_SE");
    if (!m || !*m)
        return nullptr;
    FILE* f = std::fopen(m, "rb");
    if (!f)
        return nullptr;
    std::fclose(f);
    return m;
}

std::vector<float> load_jfk() {
    float* pcm = nullptr;
    int n = 0, sr = 0;
    const std::string path = std::string(CRISPASR_SOURCE_DIR) + "/samples/jfk.wav";
    if (crispasr_audio_load(path.c_str(), &pcm, &n, &sr) != 0 || !pcm)
        return {};
    std::vector<float> v(pcm, pcm + n);
    crispasr_audio_free(pcm);
    return v;
}

// Deterministic white noise (LCG + Box-Muller), sigma 0.05.
std::vector<float> add_noise(const std::vector<float>& x) {
    std::vector<float> y(x);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    auto u = [&]() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return ((s >> 11) + 0.5) / 9007199254740992.0;
    };
    for (size_t i = 0; i < y.size(); i++)
        y[i] += (float)(0.05 * std::sqrt(-2.0 * std::log(u())) * std::cos(6.283185307179586 * u()));
    return y;
}

double snr_db(const std::vector<float>& ref, const std::vector<float>& est) {
    double s = 0, e = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        s += (double)ref[i] * ref[i];
        const double d = (double)est[i] - ref[i];
        e += d * d;
    }
    return 10.0 * std::log10(s / (e + 1e-30));
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x))
            return false;
    return true;
}

} // namespace

TEST_CASE("frcrn-se is an accepted enhancement architecture", "[frcrn-se]") {
    const std::string archs = crispasr_enhance_model_archs();
    CHECK(archs.find("frcrn-se") != std::string::npos);
    const char* model = model_path();
    if (!model)
        SKIP("CRISPASR_MODEL_FRCRN_SE not set or missing");
    char name[64] = {0};
    REQUIRE(crispasr_detect_backend_from_gguf(model, name, (int)sizeof(name)) > 0);
    CHECK(std::string(name) == "frcrn-se");
    // Not a transcription model: the session must refuse it.
    CHECK(crispasr_session_open_explicit(model, "frcrn-se", 2) == nullptr);
}

TEST_CASE("frcrn-se denoises speech and keeps length", "[frcrn-se]") {
    const char* model = model_path();
    if (!model)
        SKIP("CRISPASR_MODEL_FRCRN_SE not set or missing");
    const std::vector<float> clean = load_jfk();
    REQUIRE(clean.size() > 16000 * 5);
    const std::vector<float> noisy = add_noise(clean);
    std::vector<float> out(noisy.size(), 0.0f);
    REQUIRE(crispasr_enhance_audio_spec(model, noisy.data(), (int32_t)noisy.size(), 16000, 4, out.data(),
                                        (int32_t)out.size()) == 0);
    REQUIRE(all_finite(out));
    const double before = snr_db(clean, noisy);
    const double after = snr_db(clean, out);
    INFO("SNR before " << before << " dB, after " << after << " dB");
    CHECK(after > before + 5.0);
}

TEST_CASE("frcrn-se handles other rates, short clips and the segmented path", "[frcrn-se]") {
    const char* model = model_path();
    if (!model)
        SKIP("CRISPASR_MODEL_FRCRN_SE not set or missing");
    const std::vector<float> noisy = add_noise(load_jfk());
    REQUIRE(!noisy.empty());

    // 0.3 s (shorter than one 1 s decode window) and 2.5 s at 44.1 kHz.
    for (int n : {4800, 40000}) {
        std::vector<float> x(noisy.begin() + 16000, noisy.begin() + 16000 + n);
        std::vector<float> y(x.size(), 0.0f);
        REQUIRE(crispasr_enhance_audio_spec(model, x.data(), n, 16000, 2, y.data(), n) == 0);
        CHECK(all_finite(y));
    }
    {
        float* hi = nullptr;
        int n_hi = 0;
        REQUIRE(crispasr_audio_resample(noisy.data(), 16000 * 3, 16000, 44100, &hi, &n_hi) == 0);
        std::vector<float> y(n_hi, 0.0f);
        REQUIRE(crispasr_enhance_audio_spec(model, hi, n_hi, 44100, 2, y.data(), n_hi) == 0);
        CHECK(all_finite(y));
        crispasr_audio_free(hi);
    }

    // The > 120 s path (1 s windows, 0.75 s stride) on a short clip.
    setenv("CRISPASR_FRCRN_SE_FORCE_SEGMENT", "1", 1);
    std::vector<float> seg(noisy.size(), 0.0f);
    const int rc = crispasr_enhance_audio_spec(model, noisy.data(), (int32_t)noisy.size(), 16000, 4, seg.data(),
                                               (int32_t)seg.size());
    unsetenv("CRISPASR_FRCRN_SE_FORCE_SEGMENT");
    REQUIRE(rc == 0);
    CHECK(all_finite(seg));
    const std::vector<float> clean = load_jfk();
    CHECK(snr_db(clean, seg) > snr_db(clean, noisy) + 3.0);
}

TEST_CASE("frcrn-se is selectable on a transcription session", "[frcrn-se]") {
    const char* model = model_path();
    const char* asr = std::getenv("CRISPASR_MODEL_WHISPER");
    if (!model || !asr || !*asr)
        SKIP("CRISPASR_MODEL_FRCRN_SE / CRISPASR_MODEL_WHISPER not set");
    crispasr_session* s = crispasr_session_open(asr, 2);
    REQUIRE(s != nullptr);
    CHECK(crispasr_session_set_enhance(s, model) == 0);
    CHECK(std::string(crispasr_session_get_enhance(s)) == "frcrn-se");
    crispasr_session_close(s);
}
