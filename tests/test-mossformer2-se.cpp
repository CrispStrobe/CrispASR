// test-mossformer2-se.cpp — MossFormer2_SE_48K enhancement.
//
// [unit]  model-free: the enhancer table advertises "mossformer2-se"; a
//         metadata-only GGUF with that architecture is rejected cleanly; the
//         native front/back end matches its torch definitions (compute_deltas,
//         a unit mask through STFT -> iSTFT reconstructs the input, frame
//         counts) and the segmentation plan reproduces upstream's padding.
// [live]  MOSSFORMER2_SE_MODEL=<gguf>: enhance samples/jfk.wav twice over +
//         white noise through the C ABI at 16 kHz (resampled to 48 kHz, 22 s:
//         the segmented > 20 s decode path) and check a real SNR gain; 48 kHz
//         input, a sub-window input and digital silence keep the same-length
//         contract.  Without the variable the live cases SKIP, never pass.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "crispasr_session.h"
#include "gguf.h"
#include "mossformer2_se.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool read_wav16(const std::string& path, std::vector<float>& out, int& sr) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    char riff[12];
    f.read(riff, 12);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(riff + 8, "WAVE", 4) != 0)
        return false;
    int channels = 0, bits = 0;
    while (f) {
        char id[4];
        uint32_t size = 0;
        f.read(id, 4);
        f.read((char*)&size, 4);
        if (!f)
            return false;
        if (!std::memcmp(id, "fmt ", 4)) {
            std::vector<char> b(size);
            f.read(b.data(), size);
            channels = *(uint16_t*)&b[2];
            sr = (int)*(uint32_t*)&b[4];
            bits = *(uint16_t*)&b[14];
        } else if (!std::memcmp(id, "data", 4)) {
            if (channels != 1 || bits != 16)
                return false;
            std::vector<int16_t> pcm(size / 2);
            f.read((char*)pcm.data(), size);
            out.resize(pcm.size());
            for (size_t i = 0; i < pcm.size(); i++)
                out[i] = pcm[i] / 32768.0f;
            return true;
        } else {
            f.seekg(size + (size & 1), std::ios::cur);
        }
    }
    return false;
}

std::vector<float> gaussian(size_t n, float sigma, uint64_t seed) {
    std::vector<float> y(n);
    uint64_t s = seed;
    auto uni = [&]() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return ((s >> 11) + 0.5) / 9007199254740992.0;
    };
    for (size_t i = 0; i < n; i++)
        y[i] = sigma * (float)(std::sqrt(-2.0 * std::log(uni())) * std::cos(6.283185307179586 * uni()));
    return y;
}

double snr_db(const std::vector<float>& ref, const std::vector<float>& est) {
    double s = 0, e = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        s += (double)ref[i] * ref[i];
        e += ((double)est[i] - ref[i]) * ((double)est[i] - ref[i]);
    }
    return 10.0 * std::log10(s / e);
}

const char* live_model() {
    const char* m = std::getenv("MOSSFORMER2_SE_MODEL");
    return (m && m[0]) ? m : nullptr;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x))
            return false;
    return true;
}

} // namespace

TEST_CASE("mossformer2-se is a registered enhancement architecture", "[unit][enhance][mossformer2-se]") {
    const char* archs = crispasr_enhance_model_archs();
    REQUIRE(archs != nullptr);
    CHECK(std::string(archs).find("mossformer2-se") != std::string::npos);
}

TEST_CASE("mossformer2-se rejects a GGUF without weights", "[unit][enhance][mossformer2-se]") {
    const std::string path = std::string(std::tmpnam(nullptr)) + "-mossformer2-se-stub.gguf";
    gguf_context* g = gguf_init_empty();
    REQUIRE(g != nullptr);
    gguf_set_val_str(g, "general.architecture", "mossformer2-se");
    gguf_set_val_u32(g, "mossformer2_se.sample_rate", 48000);
    REQUIRE(gguf_write_to_file(g, path.c_str(), /*only_meta=*/true));
    gguf_free(g);
    std::vector<float> in(4800, 0.1f), out(4800, 0.0f);
    CHECK(crispasr_enhance_audio_spec(path.c_str(), in.data(), (int32_t)in.size(), 48000, 1, out.data(),
                                      (int32_t)out.size()) == -2);
    std::remove(path.c_str());
}

TEST_CASE("mossformer2-se compute_deltas matches torchaudio", "[unit][mossformer2-se]") {
    // T = 4, C = 1, ramp 0, 1, 4, 9; replicate padding by 2, denominator 10.
    const std::vector<float> x = {0, 1, 4, 9};
    const auto d = mf2se::compute_deltas(x, 4, 1);
    // t=0: -2*0 - 1*0 + 1*1 + 2*4 = 9   -> 0.9
    // t=1: -2*0 - 1*0 + 1*4 + 2*9 = 22  -> 2.2
    // t=2: -2*0 - 1*1 + 1*9 + 2*9 = 26  -> 2.6
    // t=3: -2*1 - 1*4 + 1*9 + 2*9 = 21  -> 2.1
    CHECK(d[0] == Catch::Approx(0.9f));
    CHECK(d[1] == Catch::Approx(2.2f));
    CHECK(d[2] == Catch::Approx(2.6f));
    CHECK(d[3] == Catch::Approx(2.1f));
}

TEST_CASE("mossformer2-se front end frame count and finiteness", "[unit][mossformer2-se]") {
    std::vector<float> fb, feats;
    std::vector<float> x(1919, 0.1f);
    CHECK(mf2se::features(x.data(), (int)x.size(), fb, feats) == 0);
    x = gaussian(48000, 0.1f, 7);
    const int T = mf2se::features(x.data(), (int)x.size(), fb, feats);
    CHECK(T == 1 + (48000 - 1920) / 384);
    REQUIRE(feats.size() == (size_t)T * 180);
    CHECK(all_finite(feats));
    // digital silence: log(FLT_EPSILON) everywhere, zero deltas
    std::vector<float> zeros(4000, 0.0f);
    REQUIRE(mf2se::features(zeros.data(), 4000, fb, feats) == 6);
    CHECK(feats[0] == Catch::Approx(std::log(1.1920929e-7f)));
    CHECK(std::abs(feats[60]) < 1e-5f);
}

TEST_CASE("mossformer2-se unit mask reconstructs the input", "[unit][mossformer2-se]") {
    for (int n : {1920, 48000, 48000 + 100}) {
        INFO("n = " << n);
        const auto x = gaussian((size_t)n, 0.2f, 11 + n);
        const int T = 1 + (n - 1920) / 384, cover = (T - 1) * 384 + 1920;
        const std::vector<float> ones((size_t)T * 961, 1.0f);
        const auto y = mf2se::apply_mask(x.data(), n, ones.data(), T, 2);
        REQUIRE((int)y.size() == n);
        double err = 0;
        for (int i = 0; i < cover; i++)
            err = std::max(err, (double)std::abs(y[i] - x[i]));
        CHECK(err < 1e-4);
        for (int i = cover; i < n; i++)
            REQUIRE(y[i] == 0.0f); // torch.istft(length=n) zero-fills past the last frame
    }
}

TEST_CASE("mossformer2-se segmentation reproduces upstream padding", "[unit][mossformer2-se]") {
    // 33 s at 48 kHz, 4 s windows, 3 s stride: (t - window) % stride != 0 branch
    const auto p = mf2se::segment_plan(1584000, 192000, 144000);
    CHECK(p.give_up == 24000);
    CHECK(p.padded == 1584000 + (1584000 - (1584000 - 192000) / 144000 * 144000));
    REQUIRE(!p.starts.empty());
    CHECK(p.starts.front() == 0);
    CHECK(p.starts.back() + 192000 <= p.padded);
    CHECK(p.starts.back() + 192000 + 144000 > p.padded);
    const auto q = mf2se::segment_plan(192000 + 144000 * 3, 192000, 144000); // aligned: no padding
    CHECK(q.padded == 192000 + 144000 * 3);
    CHECK(q.starts.size() == 4);
}

TEST_CASE("mossformer2-se live: segmented 22 s decode improves SNR", "[live][enhance][mossformer2-se]") {
    const char* model = live_model();
    if (!model)
        SKIP("MOSSFORMER2_SE_MODEL not set");
    std::vector<float> jfk;
    int sr = 0;
    REQUIRE(read_wav16(std::string(CRISPASR_SOURCE_DIR) + "/samples/jfk.wav", jfk, sr));
    REQUIRE(sr == 16000);
    std::vector<float> clean(jfk);
    clean.insert(clean.end(), jfk.begin(), jfk.end()); // 22 s > 20 s: upstream's segmented path
    const auto noise = gaussian(clean.size(), 0.05f, 0x9E3779B97F4A7C15ull);
    std::vector<float> noisy(clean.size());
    for (size_t i = 0; i < clean.size(); i++)
        noisy[i] = clean[i] + noise[i];
    std::vector<float> out(noisy.size());
    REQUIRE(crispasr_enhance_audio_spec(model, noisy.data(), (int32_t)noisy.size(), 16000, 2, out.data(),
                                        (int32_t)out.size()) == 0);
    REQUIRE(all_finite(out));
    const double before = snr_db(clean, noisy), after = snr_db(clean, out);
    INFO("SNR " << before << " dB -> " << after << " dB");
    CHECK(after > before + 5.0);
}

TEST_CASE("mossformer2-se live: same-length contract at 48 kHz and edge sizes", "[live][enhance][mossformer2-se]") {
    const char* model = live_model();
    if (!model)
        SKIP("MOSSFORMER2_SE_MODEL not set");
    const auto noise = gaussian(2 * 48000, 0.05f, 3);
    std::vector<float> out(noise.size());
    REQUIRE(crispasr_enhance_audio_spec(model, noise.data(), (int32_t)noise.size(), 48000, 2, out.data(),
                                        (int32_t)out.size()) == 0);
    CHECK(all_finite(out));

    // fewer samples than one 1920-sample analysis window (upstream cannot decode it)
    std::vector<float> tiny(500, 0.05f), tiny_out(500, 1.0f);
    REQUIRE(crispasr_enhance_audio_spec(model, tiny.data(), 500, 48000, 2, tiny_out.data(), 500) == 0);
    CHECK(all_finite(tiny_out));

    // digital silence stays silent
    std::vector<float> zeros(24000, 0.0f), zeros_out(24000, 1.0f);
    REQUIRE(crispasr_enhance_audio_spec(model, zeros.data(), 24000, 48000, 2, zeros_out.data(), 24000) == 0);
    for (float v : zeros_out)
        REQUIRE(v == 0.0f);
}
