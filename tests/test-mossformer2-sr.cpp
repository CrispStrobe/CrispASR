// test-mossformer2-sr.cpp — MossFormer2_SR_48K speech super-resolution.
//
// [unit]  model-free: the enhancer table advertises "mossformer2-sr"; a
//         metadata-only GGUF with that architecture is rejected cleanly; the
//         scipy parts of bandwidth_sub (butter, lfilter_zi, filtfilt) match
//         scipy 1.18 to 1e-10, detect_bandwidth finds a tone's band, the
//         degenerate cut-offs return the input, and get_mel's frame count /
//         spectral peak are right.
// [live]  MOSSFORMER2_SR_MODEL=<gguf>: super-resolve 4 s of samples/jfk.wav
//         (16 kHz, polyphase-resampled to 48 kHz) and check that the band
//         below the input's cut-off is kept, energy appears above 8 kHz, the
//         chunked generator equals the one-pass generator, and the C ABI keeps
//         the same-length contract for short inputs and silence.  Without the
//         variable the live cases SKIP, never pass.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/audio_resample.h"
#include "crispasr_session.h"
#include "gguf.h"
#include "mossformer2_sr.h"

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

const char* live_model() {
    const char* m = std::getenv("MOSSFORMER2_SR_MODEL");
    return (m && m[0]) ? m : nullptr;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x))
            return false;
    return true;
}

double energy(const std::vector<double>& v) {
    double e = 0;
    for (double x : v)
        e += x * x;
    return e;
}

void check_close(const std::vector<double>& got, const std::vector<double>& want, double tol) {
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < got.size(); i++)
        CHECK(got[i] == Catch::Approx(want[i]).margin(tol));
}

} // namespace

TEST_CASE("mossformer2-sr is a registered enhancement architecture", "[unit][enhance][mossformer2-sr]") {
    const char* archs = crispasr_enhance_model_archs();
    REQUIRE(archs != nullptr);
    CHECK(std::string(archs).find("mossformer2-sr") != std::string::npos);
}

TEST_CASE("mossformer2-sr rejects a GGUF without weights", "[unit][enhance][mossformer2-sr]") {
    const std::string path = std::string(std::tmpnam(nullptr)) + "-mossformer2-sr-stub.gguf";
    gguf_context* g = gguf_init_empty();
    REQUIRE(g != nullptr);
    gguf_set_val_str(g, "general.architecture", "mossformer2-sr");
    gguf_set_val_u32(g, "mossformer2_sr.sample_rate", 48000);
    REQUIRE(gguf_write_to_file(g, path.c_str(), /*only_meta=*/true));
    gguf_free(g);
    std::vector<float> in(4800, 0.1f), out(4800, 0.0f);
    CHECK(crispasr_enhance_audio_spec(path.c_str(), in.data(), (int32_t)in.size(), 48000, 1, out.data(),
                                      (int32_t)out.size()) == -2);
    std::remove(path.c_str());
}

TEST_CASE("mossformer2-sr butter / lfilter_zi / filtfilt match scipy", "[unit][mossformer2-sr]") {
    // scipy.signal.butter(4, 0.3, 'low' / 'high'), lfilter_zi, filtfilt (scipy 1.18).
    const auto lo = mf2sr::butter4(0.3, false);
    check_close(
        lo.b, {0.01856301062689717, 0.07425204250758868, 0.11137806376138303, 0.07425204250758868, 0.01856301062689717},
        1e-12);
    check_close(lo.a, {1.0, -1.570398851228172, 1.27561332498328, -0.4844033683350857, 0.07619706461033242}, 1e-12);
    const auto hi = mf2sr::butter4(7875.0 / 24000.0, true);
    check_close(
        hi.b, {0.24073016199506803, -0.9629206479802721, 1.4443809719704082, -0.9629206479802721, 0.24073016199506803},
        1e-12);
    check_close(hi.a, {1.0, -1.3477472848817313, 1.0656103457435984, -0.37975931522063167, 0.05856564607512728}, 1e-12);
    check_close(mf2sr::lfilter_zi(lo),
                {0.9814369893731029, -0.6632139043626578, 0.5010213568592391, -0.05763405398343525}, 1e-12);

    std::vector<double> x(40);
    for (int i = 0; i < 40; i++)
        x[(size_t)i] = std::sin(i * 0.7) + 0.1 * i;
    const auto y = mf2sr::filtfilt(lo, x);
    REQUIRE(y.size() == x.size());
    CHECK(y[0] == Catch::Approx(-0.0012362774583620575).margin(1e-10));
    CHECK(y[1] == Catch::Approx(0.7020940612909373).margin(1e-10));
    CHECK(y[17] == Catch::Approx(1.1217252491489331).margin(1e-10));
    CHECK(y[38] == Catch::Approx(4.560784806009568).margin(1e-10));
    CHECK(y[39] == Catch::Approx(4.720456772855966).margin(1e-10));
    const auto yh = mf2sr::filtfilt(mf2sr::butter4(0.3, true), x);
    CHECK(yh[0] == Catch::Approx(0.0012363097646730936).margin(1e-10));
    CHECK(yh[5] == Catch::Approx(-0.022729695475473904).margin(1e-10));
    CHECK(yh[39] == Catch::Approx(-0.0013302019162166057).margin(1e-10));
    CHECK_THROWS(mf2sr::butter4(1.0, false));
}

TEST_CASE("mossformer2-sr detect_bandwidth and degenerate cut-offs", "[unit][mossformer2-sr]") {
    // A 3 kHz tone: 99.96 % of the energy sits at or below the 3 kHz bin (+ leakage).
    std::vector<double> tone(48000);
    for (size_t i = 0; i < tone.size(); i++)
        tone[i] = 0.5 * std::sin(2.0 * M_PI * 3000.0 * (double)i / 48000.0);
    const double f = mf2sr::detect_bandwidth(tone.data(), (int64_t)tone.size(), 48000);
    CHECK(f >= 3000.0);
    CHECK(f <= 3600.0);
    std::vector<double> zeros(4800, 0.0);
    CHECK(mf2sr::detect_bandwidth(zeros.data(), (int64_t)zeros.size(), 48000) < 0);

    // Silence: returned unchanged (upstream's butter() would raise).
    std::vector<double> gen(4800, 0.25);
    const auto s = mf2sr::bandwidth_sub(zeros, gen, 48000);
    REQUIRE(s.size() == zeros.size());
    CHECK(energy(s) == 0.0);

    // Tone + generated noise: the tone band is kept, the generated band is added
    // above the cut-off, the first 100 ms fade in from the plain input.
    std::vector<double> noise(tone.size());
    uint64_t st = 0x9E3779B97F4A7C15ull;
    for (double& v : noise) {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        v = ((double)(st >> 11) / 9007199254740992.0 - 0.5) * 0.2;
    }
    double fh = 0;
    const auto y = mf2sr::bandwidth_sub(tone, noise, 48000, 0.9996, 100.0, &fh);
    REQUIRE(y.size() == tone.size());
    CHECK(fh == Catch::Approx(f));
    CHECK(y[0] == Catch::Approx(tone[0]).margin(1e-12)); // crossfade starts at the input
    std::vector<double> diff(y.size());
    for (size_t i = 0; i < y.size(); i++)
        diff[i] = y[i] - tone[i];
    const auto lowband = mf2sr::filtfilt(mf2sr::butter4(2000.0 / 24000.0, false), diff);
    CHECK(energy(lowband) < 1e-3 * energy(tone));                     // the tone band is untouched
    CHECK(energy(diff) > 0.3 * energy(noise) * (1.0 - fh / 24000.0)); // generated band added
}

TEST_CASE("mossformer2-sr log-mel frame count and peak", "[unit][mossformer2-sr]") {
    std::vector<float> mel;
    std::vector<float> x(384, 0.1f);
    CHECK(mf2sr::log_mel(x.data(), (int)x.size(), mel) == 0); // reflect pad needs n > 384
    for (int n : {385, 1024, 4800, 48000 + 17}) {
        x.assign((size_t)n, 0.0f);
        for (int i = 0; i < n; i++)
            x[(size_t)i] = 0.3f * (float)std::sin(2.0 * M_PI * 1000.0 * i / 48000.0);
        const int T = mf2sr::log_mel(x.data(), n, mel);
        CHECK(T == (n - 256) / 256 + 1);
        REQUIRE(mel.size() == (size_t)T * 80);
    }
    // 1 kHz tone: the strongest bin of a middle frame is a low band (0-8 kHz over 80 bins).
    const float* row = mel.data() + (size_t)(mel.size() / 80 / 2) * 80;
    int best = 0;
    for (int m = 1; m < 80; m++)
        if (row[m] > row[best])
            best = m;
    CHECK(best > 5);
    CHECK(best < 30);
    // silence -> log(1e-5 clip) everywhere... except the sqrt(1e-9) magnitude floor
    x.assign(4800, 0.0f);
    mf2sr::log_mel(x.data(), 4800, mel);
    for (float v : mel)
        REQUIRE(v == Catch::Approx(std::log(1e-5f)).margin(1e-3));
}

TEST_CASE("mossformer2-sr live: 16 kHz speech gains a 48 kHz band", "[live][enhance][mossformer2-sr]") {
    const char* model = live_model();
    if (!model)
        SKIP("MOSSFORMER2_SR_MODEL not set");
    std::vector<float> jfk;
    int sr = 0;
    REQUIRE(read_wav16(std::string(CRISPASR_SOURCE_DIR) + "/samples/jfk.wav", jfk, sr));
    REQUIRE(sr == 16000);
    jfk.resize(4 * 16000);
    const auto in = core_audio::resample_polyphase(jfk.data(), (int)jfk.size(), 16000, 48000);

    unsetenv("CRISPASR_MF2SR_GEN_CHUNK");
    mossformer2_sr_context* ctx = mossformer2_sr_init(model, 2, false);
    REQUIRE(ctx != nullptr);
    CHECK(mossformer2_sr_sample_rate(ctx) == 48000);
    std::vector<float> out;
    REQUIRE(mossformer2_sr_process(ctx, in.data(), (int)in.size(), out));
    mossformer2_sr_free(ctx);
    REQUIRE(out.size() == in.size());
    REQUIRE(all_finite(out));

    std::vector<double> xd(in.begin(), in.end()), yd(out.begin(), out.end());
    const auto hp = mf2sr::butter4(9000.0 / 24000.0, true);
    const double hi_in = energy(mf2sr::filtfilt(hp, xd)), hi_out = energy(mf2sr::filtfilt(hp, yd));
    INFO("energy above 9 kHz: " << hi_in << " -> " << hi_out);
    CHECK(hi_out > 100.0 * hi_in);
    // Below 6 kHz the output is the input (bandwidth_sub keeps the input band).
    const auto lp = mf2sr::butter4(6000.0 / 24000.0, false);
    auto lin = mf2sr::filtfilt(lp, xd), lout = mf2sr::filtfilt(lp, yd);
    double dot = 0, a = 0, b = 0;
    for (size_t i = 4800; i < lin.size(); i++) { // after the 100-ms crossfade
        dot += lin[i] * lout[i];
        a += lin[i] * lin[i];
        b += lout[i] * lout[i];
    }
    CHECK(dot / std::sqrt(a * b) > 0.999);

    // Chunked generator (tiny chunks) == one-pass generator.
    setenv("CRISPASR_MF2SR_GEN_CHUNK", "64", 1);
    ctx = mossformer2_sr_init(model, 2, false);
    unsetenv("CRISPASR_MF2SR_GEN_CHUNK");
    REQUIRE(ctx != nullptr);
    std::vector<float> chunked;
    REQUIRE(mossformer2_sr_process(ctx, in.data(), (int)in.size(), chunked));
    mossformer2_sr_free(ctx);
    REQUIRE(chunked.size() == out.size());
    double max_abs = 0;
    for (size_t i = 0; i < out.size(); i++)
        max_abs = std::max(max_abs, (double)std::fabs(chunked[i] - out[i]));
    CHECK(max_abs < 1e-4);
}

TEST_CASE("mossformer2-sr live: same-length contract and edge sizes", "[live][enhance][mossformer2-sr]") {
    const char* model = live_model();
    if (!model)
        SKIP("MOSSFORMER2_SR_MODEL not set");
    // fewer samples than one 1024-sample frame (upstream cannot decode it)
    std::vector<float> tiny(500), tiny_out(500, 1.0f);
    for (size_t i = 0; i < tiny.size(); i++)
        tiny[i] = 0.1f * (float)std::sin(0.05 * (double)i);
    REQUIRE(crispasr_enhance_audio_spec(model, tiny.data(), 500, 48000, 2, tiny_out.data(), 500) == 0);
    CHECK(all_finite(tiny_out));

    // digital silence stays silent
    std::vector<float> zeros(24000, 0.0f), zeros_out(24000, 1.0f);
    REQUIRE(crispasr_enhance_audio_spec(model, zeros.data(), 24000, 48000, 2, zeros_out.data(), 24000) == 0);
    for (float v : zeros_out)
        REQUIRE(v == 0.0f);

    // a 16 kHz caller gets 16 kHz back (the pre-ASR contract)
    std::vector<float> s16(8000), s16_out(8000);
    for (size_t i = 0; i < s16.size(); i++)
        s16[i] = 0.2f * (float)std::sin(2.0 * M_PI * 440.0 * (double)i / 16000.0);
    REQUIRE(crispasr_enhance_audio_spec(model, s16.data(), 8000, 16000, 2, s16_out.data(), 8000) == 0);
    CHECK(all_finite(s16_out));
}
