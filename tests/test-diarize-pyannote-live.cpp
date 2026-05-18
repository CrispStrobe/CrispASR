// Live end-to-end test for --diarize-method pyannote (issue #107).
//
// Loads the pyannote-seg GGUF model and a multi-speaker WAV, runs
// apply_pyannote, and verifies that at least two distinct speaker
// labels surface across the file. This is the regression check for
// the originally reported "all segments collapsed to one speaker"
// failure, plus the post-fix "labels exist but are bad" case from #107.
//
// The test is opt-in: it skips cleanly when either env var is unset,
// so CI without the model/audio fixtures still runs.
//
//   CRISPASR_TEST_DIARIZE_WAV=/path/to/multispeaker.wav         (required)
//   CRISPASR_TEST_DIARIZE_MODEL=/path/to/pyannote-seg-3.0.gguf  (required)
//
// The wav must be 16 kHz mono PCM s16le (the format crispasr feeds the
// pyannote front-end). Standard sample: any clip with ≥2 speakers,
// e.g. a podcast snippet, an interview, or a sherpa-onnx demo file.

#include "../src/crispasr_diarize.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

// Minimal 16-bit PCM WAV loader, mono/stereo → float32 mono at the
// file's native rate. Adapted from tests/test-titanet.cpp.
bool load_wav_mono_16k(const std::string& path, std::vector<float>& out, int& sample_rate) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return false;

    char riff[4], wave[4];
    uint32_t file_size = 0;
    if (std::fread(riff, 1, 4, f) != 4 || std::fread(&file_size, 4, 1, f) != 1 || std::fread(wave, 1, 4, f) != 4 ||
        std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        std::fclose(f);
        return false;
    }

    int channels = 0, bits = 0;
    sample_rate = 0;
    std::vector<int16_t> pcm;

    while (true) {
        char id[4];
        uint32_t sz = 0;
        if (std::fread(id, 1, 4, f) != 4 || std::fread(&sz, 4, 1, f) != 1)
            break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            uint16_t fmt = 0, ch = 0;
            uint32_t sr = 0;
            uint16_t bps = 0;
            std::fread(&fmt, 2, 1, f);
            std::fread(&ch, 2, 1, f);
            std::fread(&sr, 4, 1, f);
            std::fseek(f, 4 + 2, SEEK_CUR);
            std::fread(&bps, 2, 1, f);
            channels = ch;
            sample_rate = (int)sr;
            bits = bps;
            if (sz > 16)
                std::fseek(f, sz - 16, SEEK_CUR);
        } else if (std::memcmp(id, "data", 4) == 0) {
            int n = (int)(sz / (bits / 8) / channels);
            pcm.resize((size_t)n * channels);
            std::fread(pcm.data(), bits / 8, (size_t)n * channels, f);
            break;
        } else {
            std::fseek(f, sz, SEEK_CUR);
        }
    }
    std::fclose(f);

    if (bits != 16 || channels < 1 || sample_rate <= 0 || pcm.empty())
        return false;

    out.resize(pcm.size() / channels);
    for (size_t i = 0; i < out.size(); i++) {
        int32_t acc = 0;
        for (int c = 0; c < channels; c++)
            acc += pcm[i * channels + c];
        out[i] = (float)acc / (float)channels / 32768.0f;
    }
    return true;
}

const char* getenv_or_null(const char* name) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

} // namespace

TEST_CASE("apply_pyannote live: multi-speaker WAV yields ≥2 distinct speaker labels", "[live][diarize][pyannote]") {
    const char* wav_path = getenv_or_null("CRISPASR_TEST_DIARIZE_WAV");
    const char* model_path = getenv_or_null("CRISPASR_TEST_DIARIZE_MODEL");
    if (!wav_path || !model_path) {
        SKIP("set CRISPASR_TEST_DIARIZE_WAV + CRISPASR_TEST_DIARIZE_MODEL "
             "to a multi-speaker 16-bit PCM WAV and a pyannote-seg-3.0 GGUF "
             "to run this live test");
    }

    std::vector<float> mono;
    int sr = 0;
    REQUIRE(load_wav_mono_16k(wav_path, mono, sr));
    REQUIRE(sr == 16000);
    REQUIRE(mono.size() > (size_t)sr * 5); // at least 5 s of audio

    // Build coarse 1-second ASR-segment stand-ins covering the file.
    const int64_t dur_cs = (int64_t)(mono.size() * 100 / 16000);
    std::vector<CrispasrDiarizeSegment> segs;
    constexpr int64_t kSegCs = 100; // 1 second per segment
    for (int64_t t = 0; t + kSegCs <= dur_cs; t += kSegCs)
        segs.push_back({t, t + kSegCs, -1});
    REQUIRE(segs.size() >= 5);

    CrispasrDiarizeOptions opts;
    opts.method = CrispasrDiarizeMethod::Pyannote;
    opts.pyannote_model_path = model_path;
    opts.n_threads = 4;
    opts.slice_t0_cs = 0;

    const bool ok = crispasr_diarize_segments(mono.data(), mono.data(), (int)mono.size(),
                                              /*is_stereo=*/false, segs, opts);
    REQUIRE(ok);

    std::set<int> speakers;
    int n_labeled = 0;
    for (const auto& s : segs) {
        if (s.speaker >= 0) {
            speakers.insert(s.speaker);
            n_labeled++;
        }
    }

    INFO("labeled " << n_labeled << " / " << segs.size() << " segments");
    INFO("distinct speakers: " << speakers.size());

    // Sanity: any reasonable multi-speaker clip should label most segments.
    REQUIRE(n_labeled > (int)segs.size() / 2);

    // The regression check: for a true multi-speaker file, at least
    // two speaker IDs must surface. If this fails on a known multi-
    // speaker fixture, we've reintroduced the collapse from #107.
    REQUIRE(speakers.size() >= 2);
}
