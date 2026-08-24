// Issue #350 regression guard — parakeet non-JA long-form coverage.
//
// Two defects, one symptom (whole spans of speech silently missing from a
// 30-300 s transcript), both pinned here through the session ABI:
//
//   1. ROUTING. crispasr_session_transcribe_chunked[_lang] documents
//      `chunk_seconds = 0` as "use per-model defaults", but the unified dispatch
//      read that as "not an explicit chunk request" and fell through to the
//      300 s single-pass cap — so an explicitly chunked long-form call took ONE
//      full-length decode. v0.8.9 reached the bounded path; v0.8.24 did not.
//   2. REPAIR. The TDT decoder loses track past ~30 s and emits nothing for
//      spans it did hear, so no length cap alone can rule the symptom out.
//      gap_fill_segments re-transcribes holes in the word timeline.
//
// The fixture is the repo's own samples/jfk.wav repeated 21× (231 s), which puts
// the input squarely in the dead zone AND gives exact ground truth: 21 × 22 =
// 462 words, "country" exactly 42 times. Measured on parakeet-tdt-0.6b-v3-q4_k
// (M1 Metal): one unbounded pass 364 words / 34 "country"; fixed 463 / 42.
//
// Requires:
//   CRISPASR_MODEL_PARAKEET — a NON-Japanese parakeet GGUF (e.g.
//       parakeet-tdt-0.6b-v3-q4_k.gguf)
// SKIPs cleanly when it is missing. Run from the repo root (samples/jfk.wav).

#include <catch2/catch_test_macros.hpp>

#include "crispasr_session.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

// Mono 16-bit PCM WAV → float. The header is NOT assumed to be 44 bytes:
// samples/jfk.wav carries a LIST/INFO chunk, and reading that as audio shifts
// every sample and quietly changes what the decoder does.
std::vector<float> load_wav_16k(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};
    unsigned char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        fclose(f);
        return {};
    }
    std::vector<float> pcm;
    unsigned char ck[8];
    while (fread(ck, 1, 8, f) == 8) {
        const uint32_t sz =
            (uint32_t)ck[4] | ((uint32_t)ck[5] << 8) | ((uint32_t)ck[6] << 16) | ((uint32_t)ck[7] << 24);
        if (memcmp(ck, "data", 4) == 0) {
            std::vector<int16_t> raw(sz / 2);
            const size_t n = fread(raw.data(), 2, raw.size(), f);
            pcm.resize(n);
            for (size_t i = 0; i < n; i++)
                pcm[i] = raw[i] / 32768.0f;
            break;
        }
        fseek(f, (long)sz + (sz & 1), SEEK_CUR); // chunks are word-aligned
    }
    fclose(f);
    return pcm;
}

int count_occurrences(std::string hay, std::string needle) {
    std::transform(hay.begin(), hay.end(), hay.begin(), [](unsigned char c) { return (char)tolower(c); });
    std::transform(needle.begin(), needle.end(), needle.begin(), [](unsigned char c) { return (char)tolower(c); });
    int n = 0;
    for (size_t pos = hay.find(needle); pos != std::string::npos; pos = hay.find(needle, pos + needle.size()))
        n++;
    return n;
}

// jfk.wav × reps of known content, or empty if the sample is missing.
std::vector<float> repeated_fixture(int reps) {
    const auto one = load_wav_16k("samples/jfk.wav");
    if (one.size() < 16000 * 10)
        return {};
    std::vector<float> pcm;
    pcm.reserve(one.size() * reps);
    for (int i = 0; i < reps; i++)
        pcm.insert(pcm.end(), one.begin(), one.end());
    return pcm;
}

// 231 s — squarely in the 30-300 s dead zone the #350 cases need.
std::vector<float> long_fixture() { return repeated_fixture(21); }

std::string transcript_of(crispasr_session_result* r) {
    std::string text;
    const int n = crispasr_session_result_n_segments(r);
    for (int i = 0; i < n; i++) {
        const char* t = crispasr_session_result_segment_text(r, i);
        if (t) {
            if (!text.empty())
                text += ' ';
            text += t;
        }
    }
    return text;
}

// Set an env var for the duration of a scope and put the old value back — the
// Catch2 binary runs every case in one process.
struct scoped_env {
    std::string name;
    bool had;
    std::string old;
    scoped_env(const char* n, const char* v) : name(n) {
        const char* p = std::getenv(n);
        had = p != nullptr;
        if (had)
            old = p;
        setenv(n, v, 1);
    }
    ~scoped_env() {
        if (had)
            setenv(name.c_str(), old.c_str(), 1);
        else
            unsetenv(name.c_str());
    }
};

const char* parakeet_model() {
    const char* p = std::getenv("CRISPASR_MODEL_PARAKEET");
    if (!p || !*p)
        return nullptr;
    FILE* f = fopen(p, "rb");
    if (!f)
        return nullptr;
    fclose(f);
    return p;
}

} // namespace

TEST_CASE("parakeet long-form: chunked entry point stays bounded (issue #350)", "[integration][parakeet-longform]") {
    const char* model_path = parakeet_model();
    if (!model_path)
        SKIP("CRISPASR_MODEL_PARAKEET not set or not readable");
    const auto pcm = long_fixture();
    if (pcm.empty())
        SKIP("samples/jfk.wav not found — run from the repo root");

    // Gap-fill OFF: this case is about ROUTING alone. chunk_seconds = 0 means
    // "per-model defaults", so the call must be sliced, not run as one decode —
    // with the repair pass disabled, a single unbounded pass has no way back.
    const scoped_env no_gap_fill("CRISPASR_GAP_FILL", "0");

    crispasr_session* s = crispasr_session_open(model_path, 4);
    REQUIRE(s != nullptr);
    crispasr_session_result* r = crispasr_session_transcribe_chunked(s, pcm.data(), (int)pcm.size(), 0, -1);
    REQUIRE(r != nullptr);
    const std::string text = transcript_of(r);
    crispasr_session_result_free(r);
    crispasr_session_close(s);

    // 21 repetitions × 2 "country" = 42. One unbounded pass scored 34.
    INFO("transcript(" << text.size() << " bytes): " << text);
    CHECK(count_occurrences(text, "country") >= 40);
}

TEST_CASE("parakeet long-form: dropped spans are repaired (issue #350)", "[integration][parakeet-longform]") {
    const char* model_path = parakeet_model();
    if (!model_path)
        SKIP("CRISPASR_MODEL_PARAKEET not set or not readable");
    const auto pcm = long_fixture();
    if (pcm.empty())
        SKIP("samples/jfk.wav not found — run from the repo root");

    // Plain transcribe: 231 s is under the 300 s cap, so this IS the single
    // full-length pass — the one the gap-fill repair has to rescue.
    crispasr_session* s = crispasr_session_open(model_path, 4);
    REQUIRE(s != nullptr);
    crispasr_session_result* r = crispasr_session_transcribe(s, pcm.data(), (int)pcm.size());
    REQUIRE(r != nullptr);
    const std::string text = transcript_of(r);
    crispasr_session_result_free(r);
    crispasr_session_close(s);

    INFO("transcript(" << text.size() << " bytes): " << text);
    CHECK(count_occurrences(text, "country") >= 40);
    // ...and the sentence survives intact, not just the keyword.
    CHECK(count_occurrences(text, "ask what you can do for your country") >= 18);
}

// Issue #385 regression guard — the #208 progress contract on the unified
// dispatch. #350's fix restored the CHUNKING through the shared orchestrator
// but the orchestrator carried no progress hook, so
// crispasr_session_set_progress_callback + transcribe_chunked reported nothing
// (callback silent, g_progress atomic stuck at idle) until the call returned.
TEST_CASE("parakeet long-form: chunked entry point reports progress (issue #385)",
          "[integration][parakeet-longform]") {
    const char* model_path = parakeet_model();
    if (!model_path)
        SKIP("CRISPASR_MODEL_PARAKEET not set or not readable");
    // 66 s — past the chunked entry point's ~30 s bounded cap, so the run
    // takes the multi-window LONGFORM route with at least two windows.
    const auto pcm = repeated_fixture(6);
    if (pcm.empty())
        SKIP("samples/jfk.wav not found — run from the repo root");

    struct prog_log {
        std::vector<std::pair<int, int>> fires; // (processed, total)
        int polled_max = -1;                    // g_progress seen from inside the callback
    } log;
    crispasr_reset_progress();

    crispasr_session* s = crispasr_session_open(model_path, 4);
    REQUIRE(s != nullptr);
    crispasr_session_set_progress_callback(
        s,
        [](int processed, int total, void* ud) {
            auto* l = static_cast<prog_log*>(ud);
            l->fires.push_back({processed, total});
            l->polled_max = std::max(l->polled_max, crispasr_get_progress());
        },
        &log);
    crispasr_session_result* r = crispasr_session_transcribe_chunked(s, pcm.data(), (int)pcm.size(), 0, -1);
    REQUIRE(r != nullptr);
    crispasr_session_result_free(r);
    crispasr_session_close(s);

    // Once per finished window: a 66 s run over ~30 s windows has >= 2.
    REQUIRE(log.fires.size() >= 2);
    // Monotonically non-decreasing, ending at (total, total) with the real
    // sample count — the documented crispasr_progress_callback contract.
    for (size_t i = 1; i < log.fires.size(); i++)
        CHECK(log.fires[i].first >= log.fires[i - 1].first);
    for (const auto& f : log.fires)
        CHECK(f.second == (int)pcm.size());
    CHECK(log.fires.back().first == (int)pcm.size());
    // The pollable atomic moves in lockstep with the callback, and returns to
    // idle when the call is done.
    CHECK(log.polled_max > 0);
    CHECK(crispasr_get_progress() == -1);
}
