// Sidon live integration test — model load + 16 kHz speech restoration.
//
// Requires CRISPASR_MODEL_SIDON to point to a Sidon GGUF. Skips cleanly
// when the model is not available.

#include <catch2/catch_test_macros.hpp>

#include "sidon.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

struct ScopedTestEnv {
    std::string name;
    std::string previous;
    bool had_previous = false;

    ScopedTestEnv(const char* key, const char* value) : name(key) {
        if (const char* current = std::getenv(key)) {
            previous = current;
            had_previous = true;
        }
        set_test_env(key, value);
    }

    ~ScopedTestEnv() { set_test_env(name.c_str(), had_previous ? previous.c_str() : nullptr); }
};

// Walk the RIFF chunk list to find `data`. A fixed 44-byte skip is wrong for
// real files: samples/jfk.wav carries a 26-byte LIST chunk between `fmt ` and
// `data`, so audio actually starts at byte 78 and a 44-byte skip feeds 17
// samples of chunk header into the model as though they were loud audio.
static std::vector<float> load_wav_16k(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};

    char riff[12];
    if (fread(riff, 1, sizeof(riff), f) != sizeof(riff) || memcmp(riff, "RIFF", 4) != 0 ||
        memcmp(riff + 8, "WAVE", 4) != 0) {
        fclose(f);
        return {};
    }

    uint32_t data_bytes = 0;
    for (;;) {
        char id[4];
        uint32_t chunk_size = 0;
        if (fread(id, 1, sizeof(id), f) != sizeof(id) || fread(&chunk_size, sizeof(chunk_size), 1, f) != 1) {
            fclose(f);
            return {};
        }
        if (memcmp(id, "data", 4) == 0) {
            data_bytes = chunk_size;
            break;
        }
        // Chunks are word-aligned: an odd size is followed by a pad byte.
        if (fseek(f, (long)chunk_size + (chunk_size & 1u), SEEK_CUR) != 0) {
            fclose(f);
            return {};
        }
    }

    if (data_bytes == 0 || data_bytes % sizeof(int16_t) != 0) {
        fclose(f);
        return {};
    }

    std::vector<int16_t> raw(data_bytes / sizeof(int16_t));
    const size_t read = fread(raw.data(), sizeof(int16_t), raw.size(), f);
    fclose(f);
    if (read != raw.size())
        return {};

    std::vector<float> pcm(raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
        pcm[i] = raw[i] / 32768.0f;
    return pcm;
}

TEST_CASE("sidon speech restoration", "[integration][sidon]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_SIDON");
    if (!model_path || !*model_path)
        SKIP("CRISPASR_MODEL_SIDON not set");

    auto params = sidon_context_default_params();
    params.verbosity = 0;
    auto* ctx = sidon_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    const auto input = load_wav_16k("samples/jfk.wav");
    REQUIRE(!input.empty());

    const auto output = sidon_restore(ctx, input.data(), (int)input.size());
    REQUIRE(output.size() == input.size() * 3);
    REQUIRE(std::all_of(output.begin(), output.end(), [](float sample) { return std::isfinite(sample); }));

    float peak = 0.0f;
    for (float sample : output)
        peak = std::max(peak, std::fabs(sample));
    CHECK(peak > 0.01f);

    // samples/jfk.wav ends in silence. Decoding without Sidon's prescribed
    // right-side lookahead used to expose the model boundary response here,
    // producing a near-full-scale transient in the final ~12 ms.
    float tail_peak = 0.0f;
    const size_t tail_samples = std::min<size_t>(output.size(), 48 * 12);
    for (size_t i = output.size() - tail_samples; i < output.size(); ++i)
        tail_peak = std::max(tail_peak, std::fabs(output[i]));
    CHECK(tail_peak < 0.05f);

    // Restoration must be time-ALIGNED with the input, not merely the right
    // length. Inference pads the input (a leading predictor frame plus 1.5 s of
    // lookahead) and crops the padding back off afterwards; cropping only the
    // tail leaves the whole result delayed by the leading pad and silently drops
    // the same amount of real audio off the end. Length checks and the
    // chunked/whole parity check are both blind to that, so compare the speech
    // onset on both sides.
    {
        // Speech onset from a 5 ms windowed-RMS envelope. A bare
        // first-sample-over-threshold test is useless here: jfk.wav opens on
        // broadband noise whose very first sample already clears any sane
        // fraction of the peak.
        auto onset = [](const std::vector<float>& x, int win) {
            std::vector<float> env;
            for (size_t i = 0; i + (size_t)win <= x.size(); i += (size_t)win) {
                double sum = 0.0;
                for (int j = 0; j < win; ++j)
                    sum += (double)x[i + (size_t)j] * x[i + (size_t)j];
                env.push_back((float)std::sqrt(sum / win));
            }
            if (env.empty())
                return (long)0;
            const float loudest = *std::max_element(env.begin(), env.end());
            const float threshold = 0.10f * loudest;
            for (size_t i = 0; i < env.size(); ++i)
                if (env[i] > threshold)
                    return (long)(i * (size_t)win);
            return (long)x.size();
        };
        const long in_onset = onset(input, 80) * 3; // 5 ms @ 16 kHz -> 48 kHz
        const long out_onset = onset(output, 240);  // 5 ms @ 48 kHz
        const long skew = std::labs(out_onset - in_onset);
        INFO("onset input(x3)=" << in_onset << " output=" << out_onset << " skew=" << skew << " samples");
        CHECK(skew < 480); // < 10 ms @ 48 kHz
    }

    // The bounded DAC path must reproduce the whole-utterance decode EXACTLY.
    // It decodes each core with the decoder's full latent receptive field, so
    // the only difference is where the graph is cut — not the arithmetic. Assert
    // bit-exactness rather than a cosine: a global cosine over the whole
    // waveform is the wrong instrument here, because a join discontinuity is
    // local and a handful of bad samples out of 528000 barely moves it. When
    // graph reuse silently corrupted the joins, max|diff| was 0.476 while the
    // cosine still read 0.978.
    //
    // This also exercises scheduler teardown/recreation on a persistent context.
    {
        ScopedTestEnv full_decode("CRISPASR_SIDON_DECODER_CHUNK_FRAMES", "0");
        const auto full_output = sidon_restore(ctx, input.data(), (int)input.size());
        REQUIRE(full_output.size() == output.size());
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < output.size(); ++i)
            max_abs_diff = std::max(max_abs_diff, std::fabs(output[i] - full_output[i]));
        INFO("max|chunked - whole-utterance| = " << max_abs_diff);
        CHECK(max_abs_diff == 0.0f);
    }

    // The relative-position-bias formulations are algebraically identical; they
    // differ only in what the graph materialises. `expand` keeps the legacy
    // [head_dim, T, T] expansion (and the Vulkan-specific branch), `bucket`
    // evaluates one dot product per distance bucket, `bucket-direct` also skips
    // the in-graph REPEAT of the gather index. Reordered float arithmetic means
    // these are close, not identical, and the DAC amplifies the difference — so
    // check the magnitudes too, since cosine alone is scale-blind.
    //
    // NOTE: until the #416 follow-up this loop compared bucket-direct against
    // itself. CRISPASR_SIDON_RPE was read once in sidon_init (parse_rpe_mode)
    // and never again, so setting it here — on an already-loaded ctx — changed
    // nothing and both iterations ran whatever init had chosen. The mode is now
    // resolved per graph build (resolve_rpe_mode), so these env settings take
    // effect and the comparison is real. `bucket` is included for the same
    // reason: all three are claimed equivalent, so all three get checked.
    for (const char* mode : {"expand", "bucket", "bucket-direct"}) {
        ScopedTestEnv rpe("CRISPASR_SIDON_RPE", mode);
        const auto alt = sidon_restore(ctx, input.data(), (int)input.size());
        REQUIRE(alt.size() == output.size());
        double dot = 0.0, alt_sq = 0.0, ref_sq = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            dot += (double)output[i] * alt[i];
            alt_sq += (double)alt[i] * alt[i];
            ref_sq += (double)output[i] * output[i];
        }
        const double cosine = dot / std::sqrt(alt_sq * ref_sq);
        const double magnitude_ratio = std::sqrt(alt_sq) / std::sqrt(ref_sq);
        INFO("RPE mode " << mode << ": cos=" << cosine << " |alt|/|ref|=" << magnitude_ratio);
        CHECK(cosine > 0.99);
        CHECK(magnitude_ratio > 0.98);
        CHECK(magnitude_ratio < 1.02);
    }

    // O(T^2) length cap: an over-long input (well past the ~58.5 s / 3000-frame
    // default) must fail cleanly with an empty result — never OOM/crash. The
    // guard trips right after the cheap STFT front-end, so this stays fast even
    // with the model loaded. ~90 s of silence plus lookahead ⇒ T ≈ 4575 > 3000.
    {
        std::vector<float> too_long(16000 * 90, 0.0f);
        // A little energy so peak-normalization doesn't divide near-zero.
        for (size_t i = 0; i < too_long.size(); i += 160)
            too_long[i] = 0.1f;
        const auto capped = sidon_restore(ctx, too_long.data(), (int)too_long.size());
        CHECK(capped.empty());
    }

    sidon_free(ctx);
}

// Issue #416: a QUANTIZED Sidon GGUF produced a full-length file of pure
// silence while the f16 model restored the same clip correctly. Nothing caught
// it because every Sidon test above runs a single model, and
// tests/env-live-tests.sh pointed CRISPASR_MODEL_SIDON at the f16 build only —
// so no quantized Sidon was ever executed.
//
// The reported failure is silence, not drift, so this deliberately does NOT
// assert a tolerance against the f16 output: q4_k genuinely differs, and a
// cosine band wide enough to admit q4_k would still have to be narrower than
// "all zeros" to catch anything. Assert the properties a silent/NaN decode
// breaks instead — full length, all-finite, real energy, and energy that
// actually tracks the input envelope rather than a constant hum.
//
// Point CRISPASR_MODEL_SIDON_QUANT at a q8_0/q6_k/q4_k Sidon GGUF; the case
// skips when unset so CI without the artifact stays green.
TEST_CASE("sidon quantized restoration is not silent", "[integration][sidon]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_SIDON_QUANT");
    if (!model_path || !*model_path)
        SKIP("CRISPASR_MODEL_SIDON_QUANT not set");

    auto params = sidon_context_default_params();
    params.verbosity = 0;
    auto* ctx = sidon_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    const auto input = load_wav_16k("samples/jfk.wav");
    REQUIRE(!input.empty());

    const auto output = sidon_restore(ctx, input.data(), (int)input.size());
    REQUIRE(output.size() == input.size() * 3);

    // A NaN decode reaches the WAV writer as zeros, so the silence and the
    // not-finite failures look identical downstream. Separate them here.
    REQUIRE(std::all_of(output.begin(), output.end(), [](float s) { return std::isfinite(s); }));

    double sum_sq = 0.0;
    float peak = 0.0f;
    for (float s : output) {
        sum_sq += (double)s * s;
        peak = std::max(peak, std::fabs(s));
    }
    const double rms = std::sqrt(sum_sq / (double)output.size());
    INFO("quantized output rms=" << rms << " peak=" << peak);
    // The f16 path measures rms ~0.128 / peak ~0.61 on this clip; every quant
    // lands within a few percent. 0.01 is far below any real decode and far
    // above the reported failure (identically zero).
    CHECK(rms > 0.01);
    CHECK(peak > 0.05f);

    // Energy must FOLLOW the source. A constant tone or DC offset would clear
    // the rms floor above while still being a broken decode, so require the
    // loud half of the input to be louder than the quiet half on the output
    // too. jfk.wav is speech into trailing silence, so this ordering is stable.
    {
        auto rms_of = [](const float* p, size_t n) {
            double acc = 0.0;
            for (size_t i = 0; i < n; ++i)
                acc += (double)p[i] * p[i];
            return std::sqrt(acc / (double)n);
        };
        const size_t half = output.size() / 2;
        const double head = rms_of(output.data(), half);
        const double tail = rms_of(output.data() + half, output.size() - half);
        const double in_head = rms_of(input.data(), input.size() / 2);
        const double in_tail = rms_of(input.data() + input.size() / 2, input.size() - input.size() / 2);
        INFO("halves: out " << head << "/" << tail << "  in " << in_head << "/" << in_tail);
        CHECK((head > tail) == (in_head > in_tail));
    }

    sidon_free(ctx);
}
