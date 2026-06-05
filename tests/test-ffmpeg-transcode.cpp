/*
 * test-ffmpeg-transcode.cpp — Catch2 tests for the in-memory FFmpeg
 * decode/resample pipeline (examples/ffmpeg-transcode.cpp).
 *
 * Validates the fixes from GitHub issue #129 (theaerotoad):
 *   1. AVIO EOF handling  — read_packet returns AVERROR_EOF
 *   2. Stream filtering   — non-audio packets are skipped
 *   3. Multi-frame decode — inner while loop on avcodec_receive_frame
 *   4. Ref-count hygiene  — av_packet_unref / av_frame_unref
 *   5. Decoder drain      — NULL-packet flush before resampler flush
 *   6. Null-safe flush    — convert_frame guards frame->nb_samples
 *
 * Test strategy:
 *   - Decode jfk.wav via ffmpeg_decode_audio as the reference signal.
 *   - Decode container formats (.webm, .m4a, .mp3) through the same path.
 *   - Compare each result against the reference: length similarity, non-zero
 *     samples, and cross-correlation above a threshold (lossy codecs will not
 *     be bit-exact but must be perceptually close).
 */

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// The function under test — defined in examples/ffmpeg-transcode.cpp
extern int ffmpeg_decode_audio(const std::string & ifname,
                               std::vector<uint8_t> & owav_data);

// ── helpers ──────────────────────────────────────────────────────────

struct wave_hdr {
    char riff_header[4];
    int  wav_size;
    char wav_header[4];
    char fmt_header[4];
    int  fmt_chunk_size;
    int16_t audio_format;
    int16_t num_channels;
    int  sample_rate;
    int  byte_rate;
    int16_t sample_alignment;
    int16_t bit_depth;
    char data_header[4];
    int  data_bytes;
} __attribute__((__packed__));

static_assert(sizeof(wave_hdr) == 44, "wave_hdr must be 44 bytes");

// Extract s16 PCM samples from the wav blob produced by ffmpeg_decode_audio.
static std::vector<int16_t> extract_pcm(const std::vector<uint8_t> & wav) {
    REQUIRE(wav.size() >= sizeof(wave_hdr));
    const wave_hdr * hdr = reinterpret_cast<const wave_hdr *>(wav.data());
    REQUIRE(std::memcmp(hdr->riff_header, "RIFF", 4) == 0);
    REQUIRE(std::memcmp(hdr->wav_header, "WAVE", 4) == 0);
    REQUIRE(hdr->audio_format == 1);        // PCM
    REQUIRE(hdr->num_channels == 1);        // mono
    REQUIRE(hdr->sample_rate  == 16000);    // 16 kHz
    REQUIRE(hdr->bit_depth    == 16);

    size_t n_samples = hdr->data_bytes / sizeof(int16_t);
    REQUIRE(n_samples > 0);
    std::vector<int16_t> pcm(n_samples);
    std::memcpy(pcm.data(), wav.data() + sizeof(wave_hdr),
                n_samples * sizeof(int16_t));
    return pcm;
}

// Normalised cross-correlation between two s16 signals (peak over ±lag window).
static double cross_correlation(const std::vector<int16_t> & a,
                                const std::vector<int16_t> & b,
                                int max_lag = 160 /* 10 ms at 16 kHz */) {
    size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0.0;

    // energy of each signal
    double ea = 0, eb = 0;
    for (size_t i = 0; i < n; ++i) {
        ea += double(a[i]) * a[i];
        eb += double(b[i]) * b[i];
    }
    double norm = std::sqrt(ea * eb);
    if (norm < 1e-12) return 0.0;

    double best = -1.0;
    for (int lag = -max_lag; lag <= max_lag; ++lag) {
        double sum = 0;
        size_t count = 0;
        for (size_t i = 0; i < n; ++i) {
            int j = int(i) + lag;
            if (j < 0 || j >= int(n)) continue;
            sum += double(a[i]) * b[j];
            ++count;
        }
        if (count > 0) {
            double cc = sum / norm;
            if (cc > best) best = cc;
        }
    }
    return best;
}

// Check that a signal is not silent (has energy).
static bool has_energy(const std::vector<int16_t> & pcm, double min_rms = 50.0) {
    if (pcm.empty()) return false;
    double sum_sq = 0;
    for (auto s : pcm) sum_sq += double(s) * s;
    return std::sqrt(sum_sq / pcm.size()) >= min_rms;
}

// ── test cases ───────────────────────────────────────────────────────

#ifndef SAMPLES_DIR
#define SAMPLES_DIR "."
#endif

static std::string sample(const char * name) {
    return std::string(SAMPLES_DIR) + "/" + name;
}

TEST_CASE("ffmpeg_decode_audio decodes WAV reference", "[ffmpeg][unit]") {
    std::vector<uint8_t> wav;
    int ret = ffmpeg_decode_audio(sample("jfk.wav"), wav);
    REQUIRE(ret == 0);

    auto pcm = extract_pcm(wav);
    // jfk.wav is ~11 seconds at 16 kHz → ~176000 samples
    REQUIRE(pcm.size() > 100000);
    REQUIRE(pcm.size() < 250000);
    REQUIRE(has_energy(pcm));
}

TEST_CASE("ffmpeg_decode_audio decodes WebM (opus)", "[ffmpeg][unit][webm]") {
    // First decode reference
    std::vector<uint8_t> ref_wav;
    REQUIRE(ffmpeg_decode_audio(sample("jfk.wav"), ref_wav) == 0);
    auto ref_pcm = extract_pcm(ref_wav);

    // Decode webm
    std::vector<uint8_t> webm_wav;
    int ret = ffmpeg_decode_audio(sample("jfk.webm"), webm_wav);
    REQUIRE(ret == 0);

    auto pcm = extract_pcm(webm_wav);
    REQUIRE(has_energy(pcm));

    // Length should be within 5% of reference
    double ratio = double(pcm.size()) / ref_pcm.size();
    REQUIRE(ratio > 0.95);
    REQUIRE(ratio < 1.05);

    // Cross-correlation should be high (>0.85 — lossy codec)
    double cc = cross_correlation(ref_pcm, pcm);
    INFO("WebM cross-correlation: " << cc);
    REQUIRE(cc > 0.85);
}

TEST_CASE("ffmpeg_decode_audio decodes M4A (AAC)", "[ffmpeg][unit][m4a]") {
    std::vector<uint8_t> ref_wav;
    REQUIRE(ffmpeg_decode_audio(sample("jfk.wav"), ref_wav) == 0);
    auto ref_pcm = extract_pcm(ref_wav);

    std::vector<uint8_t> m4a_wav;
    int ret = ffmpeg_decode_audio(sample("jfk.m4a"), m4a_wav);
    REQUIRE(ret == 0);

    auto pcm = extract_pcm(m4a_wav);
    REQUIRE(has_energy(pcm));

    double ratio = double(pcm.size()) / ref_pcm.size();
    REQUIRE(ratio > 0.95);
    REQUIRE(ratio < 1.05);

    double cc = cross_correlation(ref_pcm, pcm);
    INFO("M4A cross-correlation: " << cc);
    REQUIRE(cc > 0.85);
}

TEST_CASE("ffmpeg_decode_audio decodes MP3", "[ffmpeg][unit][mp3]") {
    std::vector<uint8_t> ref_wav;
    REQUIRE(ffmpeg_decode_audio(sample("jfk.wav"), ref_wav) == 0);
    auto ref_pcm = extract_pcm(ref_wav);

    std::vector<uint8_t> mp3_wav;
    int ret = ffmpeg_decode_audio(sample("jfk.mp3"), mp3_wav);
    REQUIRE(ret == 0);

    auto pcm = extract_pcm(mp3_wav);
    REQUIRE(has_energy(pcm));

    double ratio = double(pcm.size()) / ref_pcm.size();
    REQUIRE(ratio > 0.95);
    REQUIRE(ratio < 1.05);

    double cc = cross_correlation(ref_pcm, pcm);
    INFO("MP3 cross-correlation: " << cc);
    REQUIRE(cc > 0.85);
}

TEST_CASE("ffmpeg_decode_audio rejects missing file", "[ffmpeg][unit]") {
    std::vector<uint8_t> wav;
    int ret = ffmpeg_decode_audio("/nonexistent/file.wav", wav);
    REQUIRE(ret != 0);
}
