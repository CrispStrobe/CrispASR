// tests/test_align_only.cpp — unit + live tests for standalone alignment (issue #217).
//
// Unit tests: SRT parsing, tokenise_words, empty/null safety.
// Live tests: actual CTC alignment with canary-ctc-aligner or wav2vec2.
//
// Env vars:
//   CRISPASR_MODEL_ALIGNER       — CTC aligner GGUF (canary-ctc-aligner-q4_k, wav2vec2, etc.)
//   CRISPASR_TEST_AUDIO_JFK      — path to jfk.wav (defaults to samples/jfk.wav)

#include <catch2/catch_test_macros.hpp>

#include "crispasr_aligner.h"
#include "core/qwen3_forced_aligner.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Unit tests — no model needed
// ---------------------------------------------------------------------------

TEST_CASE("align-only: crispasr_align_words null/empty safety", "[unit][align]") {
    // All null/empty inputs must return an empty vector, not crash.
    auto r1 = crispasr_align_words("", "hello world", nullptr, 0, 0, 1);
    CHECK(r1.empty());

    auto r2 = crispasr_align_words("/nonexistent.gguf", "", nullptr, 0, 0, 1);
    CHECK(r2.empty());

    auto r3 = crispasr_align_words("/nonexistent.gguf", "hello", nullptr, 0, 0, 1);
    CHECK(r3.empty());

    float dummy = 0.0f;
    auto r4 = crispasr_align_words("/nonexistent.gguf", "hello", &dummy, 1, 0, 1);
    CHECK(r4.empty());
}

TEST_CASE("align-only: crispasr_aligner_free_cache does not crash when empty", "[unit][align]") {
    crispasr_aligner_free_cache(); // must not crash
    crispasr_aligner_free_cache(); // double-free safety
}

TEST_CASE("align-only: each segment gets its own audio interval", "[unit][align][issue444]") {
    constexpr int sr = 16000;
    const int slice_start = 10 * sr;
    const int slice_end = 100 * sr;

    // Two segments in one VAD slice must not both restart at the slice origin.
    const auto first = crispasr_alignment_audio_range(7197, 8509, slice_start, slice_end, sr);
    const auto second = crispasr_alignment_audio_range(8893, 8901, slice_start, slice_end, sr);
    REQUIRE(first.valid());
    REQUIRE(second.valid());
    CHECK(first.start == 7197 * sr / 100);
    CHECK(first.end == 8509 * sr / 100);
    CHECK(first.offset_cs == 7197);
    CHECK(second.start >= first.end);
    CHECK(second.offset_cs == 8893);

    // Context-expanded timestamps are clipped to the actual slice samples.
    const auto clipped = crispasr_alignment_audio_range(500, 12000, slice_start, slice_end, sr);
    REQUIRE(clipped.valid());
    CHECK(clipped.start == slice_start);
    CHECK(clipped.end == slice_end);
    CHECK(clipped.offset_cs == 1000);

    CHECK_FALSE(crispasr_alignment_audio_range(5000, 5000, slice_start, slice_end, sr).valid());
}

// Join cue texts back into the flat transcript --align-only feeds the aligner.
static std::string extract_srt_text(const std::string& raw) {
    std::string text_only;
    for (const auto& cue : crispasr_parse_srt_cues(raw)) {
        if (!text_only.empty())
            text_only += ' ';
        text_only += cue;
    }
    return text_only;
}

TEST_CASE("align-only: SRT cue parsing", "[unit][align]") {
    SECTION("standard SRT") {
        const char* srt = "1\r\n"
                          "00:00:00,000 --> 00:00:02,500\r\n"
                          "Hello world\r\n"
                          "\r\n"
                          "2\r\n"
                          "00:00:02,500 --> 00:00:05,000\r\n"
                          "This is a test\r\n"
                          "\r\n";
        auto cues = crispasr_parse_srt_cues(srt);
        REQUIRE(cues.size() == 2);
        CHECK(cues[0] == "Hello world");
        CHECK(cues[1] == "This is a test");
        CHECK(extract_srt_text(srt) == "Hello world This is a test");
    }

    SECTION("SRT with multi-line subtitle text") {
        const char* srt = "1\n"
                          "00:00:00,000 --> 00:00:03,000\n"
                          "Line one\n"
                          "Line two\n"
                          "\n"
                          "2\n"
                          "00:00:03,000 --> 00:00:05,000\n"
                          "Single line\n"
                          "\n";
        auto cues = crispasr_parse_srt_cues(srt);
        REQUIRE(cues.size() == 2);
        CHECK(cues[0] == "Line one Line two");
        CHECK(cues[1] == "Single line");
        CHECK(extract_srt_text(srt) == "Line one Line two Single line");
    }

    SECTION("empty SRT") {
        CHECK(crispasr_parse_srt_cues("").empty());
        CHECK(extract_srt_text("").empty());
    }

    SECTION("SRT with no trailing blank line") {
        const char* srt = "1\n"
                          "00:00:00,000 --> 00:00:01,000\n"
                          "No trailing blank";
        auto cues = crispasr_parse_srt_cues(srt);
        REQUIRE(cues.size() == 1);
        CHECK(cues[0] == "No trailing blank");
    }

    SECTION("whitespace-only cue is dropped") {
        const char* srt = "1\n"
                          "00:00:00,000 --> 00:00:01,000\n"
                          "   \n"
                          "\n"
                          "2\n"
                          "00:00:01,000 --> 00:00:02,000\n"
                          "Real text\n"
                          "\n";
        auto cues = crispasr_parse_srt_cues(srt);
        REQUIRE(cues.size() == 1);
        CHECK(cues[0] == "Real text");
    }
}

TEST_CASE("align-only: JSON segment parsing (#317)", "[unit][align]") {
    SECTION("CrispASR --output-json format") {
        const char* json = R"({
  "systeminfo": "...",
  "model": {},
  "transcription": [
    {"timestamps": {"from": "00:00:00,400", "to": "00:00:02,160"}, "text": "And so my fellow Americans"},
    {"timestamps": {"from": "00:00:03,520", "to": "00:00:07,280"}, "text": "ask not what your country can do for you"}
  ]
})";
        auto segs = crispasr_parse_json_segments(json);
        REQUIRE(segs.size() == 2);
        CHECK(segs[0] == "And so my fellow Americans");
        CHECK(segs[1] == "ask not what your country can do for you");
    }

    SECTION("align-only JSON output format (array of objects)") {
        const char* json = R"([
  {"text": "And so my fellow Americans", "start": 0.400, "end": 2.160},
  {"text": "ask not what your country can do for you", "start": 3.520, "end": 7.280}
])";
        auto segs = crispasr_parse_json_segments(json);
        REQUIRE(segs.size() == 2);
        CHECK(segs[0] == "And so my fellow Americans");
        CHECK(segs[1] == "ask not what your country can do for you");
    }

    SECTION("escaped characters in text") {
        const char* json = R"({"transcription": [{"text": "He said \"hello\" and\\left"}]})";
        auto segs = crispasr_parse_json_segments(json);
        REQUIRE(segs.size() == 1);
        CHECK(segs[0] == "He said \"hello\" and\\left");
    }

    SECTION("empty transcription array") {
        auto segs = crispasr_parse_json_segments(R"({"transcription": []})");
        CHECK(segs.empty());
    }

    SECTION("whitespace-only text is skipped") {
        auto segs = crispasr_parse_json_segments(R"({"transcription": [{"text": "  "}, {"text": "real"}]})");
        REQUIRE(segs.size() == 1);
        CHECK(segs[0] == "real");
    }

    SECTION("no transcription key returns empty") {
        auto segs = crispasr_parse_json_segments(R"({"something": "else"})");
        CHECK(segs.empty());
    }

    SECTION("invalid JSON returns empty") {
        auto segs = crispasr_parse_json_segments("not json at all");
        CHECK(segs.empty());
    }
}

TEST_CASE("align-only: tokenise_align_words", "[unit][align]") {
    SECTION("space-delimited") {
        auto w = crispasr_tokenise_align_words("Hello  world\tfoo\nbar");
        REQUIRE(w.size() == 4);
        CHECK(w[0] == "Hello");
        CHECK(w[3] == "bar");
    }
    SECTION("CJK splits per character") {
        auto w = crispasr_tokenise_align_words("你好，world!");
        REQUIRE(w.size() == 3);
        CHECK(w[0] == "你");
        CHECK(w[1] == "好");
        CHECK(w[2] == "world");
    }
    SECTION("empty") {
        CHECK(crispasr_tokenise_align_words("").empty());
        CHECK(crispasr_tokenise_align_words("  \n\t").empty());
    }
}

TEST_CASE("align-only: display units restore punctuation without adding timestamp slots", "[unit][align][issue444]") {
    SECTION("CJK sentence punctuation attaches to the preceding character") {
        const auto labels = crispasr_tokenise_align_words("你好，world!再见？");
        const auto display = crispasr_tokenise_align_display_words("你好，world!再见？");
        REQUIRE(labels == std::vector<std::string>{"你", "好", "world", "再", "见"});
        REQUIRE(display == std::vector<std::string>{"你", "好，", "world!", "再", "见？"});
        CHECK(display.size() == labels.size());
    }

    SECTION("leading and repeated punctuation preserve the one-to-one map") {
        const auto labels = crispasr_tokenise_align_words("“你好……” test-case.");
        const auto display = crispasr_tokenise_align_display_words("“你好……” test-case.");
        REQUIRE(display.size() == labels.size());
        REQUIRE(display.size() == 4);
        CHECK(display[0] == "“你");
        CHECK(display[1] == "好……”");
        CHECK(display[2] == "test-");
        CHECK(display[3] == "case.");
    }

    SECTION("an opening quote after whitespace attaches to the following word") {
        const auto labels = crispasr_tokenise_align_words("hello “world”");
        const auto display = crispasr_tokenise_align_display_words("hello “world”");
        REQUIRE(display.size() == labels.size());
        REQUIRE(display == std::vector<std::string>{"hello", "“world”"});
    }
}

TEST_CASE("align-only: Qwen3 timestamp repair matches Python blueprint", "[unit][align][issue444]") {
    SECTION("first-maximum LIS tie and short anomaly snapping") {
        std::vector<int> ts{0, 0, 1, 0};
        core_qwen3_forced_aligner::fix_timestamps(ts);
        CHECK(ts == std::vector<int>{0, 0, 1, 1});
    }
    SECTION("long anomaly run interpolates") {
        std::vector<int> ts{0, 8, 7, 6, 5, 16};
        core_qwen3_forced_aligner::fix_timestamps(ts);
        CHECK(ts == std::vector<int>{0, 8, 10, 12, 14, 16});
    }
}

// Build a synthetic word alignment: word i spans [i*100, i*100+80) cs.
static std::vector<CrispasrAlignedWord> make_words(const std::vector<std::string>& texts) {
    std::vector<CrispasrAlignedWord> out;
    for (size_t i = 0; i < texts.size(); i++)
        out.push_back({texts[i], (int64_t)(i * 100), (int64_t)(i * 100 + 80)});
    return out;
}

TEST_CASE("align-only: group aligned words into segments", "[unit][align]") {
    SECTION("exact word counts") {
        std::vector<std::string> segs = {"Hello world", "This is a test"};
        auto words = make_words({"Hello", "world", "This", "is", "a", "test"});
        auto grouped = crispasr_group_aligned_segments(segs, words);
        REQUIRE(grouped.size() == 2);
        CHECK(grouped[0].text == "Hello world");
        CHECK(grouped[0].t0_cs == 0);
        CHECK(grouped[0].t1_cs == 180);
        CHECK(grouped[0].word_begin == 0);
        CHECK(grouped[0].word_end == 2);
        CHECK(grouped[1].t0_cs == 200);
        CHECK(grouped[1].t1_cs == 580);
        CHECK(grouped[1].word_begin == 2);
        CHECK(grouped[1].word_end == 6);
    }

    SECTION("leftover words extend the last segment") {
        std::vector<std::string> segs = {"one", "two three"};
        auto words = make_words({"one", "two", "three", "extra"});
        auto grouped = crispasr_group_aligned_segments(segs, words);
        REQUIRE(grouped.size() == 2);
        CHECK(grouped[1].word_end == 4);
        CHECK(grouped[1].t1_cs == 380);
    }

    SECTION("alignment shorter than transcript drops uncovered tail") {
        std::vector<std::string> segs = {"one two", "three four"};
        auto words = make_words({"one", "two"});
        auto grouped = crispasr_group_aligned_segments(segs, words);
        REQUIRE(grouped.size() == 1);
        CHECK(grouped[0].text == "one two");
    }

    SECTION("CJK segment consumes per-character words") {
        std::vector<std::string> segs = {"你好", "world"};
        auto words = make_words({"你", "好", "world"});
        auto grouped = crispasr_group_aligned_segments(segs, words);
        REQUIRE(grouped.size() == 2);
        CHECK(grouped[0].word_end == 2);
        CHECK(grouped[1].word_begin == 2);
    }

    SECTION("empty inputs") {
        CHECK(crispasr_group_aligned_segments({}, {}).empty());
        CHECK(crispasr_group_aligned_segments({"text"}, {}).empty());
    }
}

// ---------------------------------------------------------------------------
// Live tests — need CRISPASR_MODEL_ALIGNER + audio
// ---------------------------------------------------------------------------

static std::string get_env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

static std::vector<float> load_wav_16k_mono(const std::string& path) {
    std::vector<float> pcm;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return pcm;

    char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        fclose(f);
        return pcm;
    }

    int channels = 0, sample_rate = 0, bits = 0;
    int32_t data_size = 0;
    bool found_fmt = false, found_data = false;

    while (!found_data) {
        char chunk_id[4];
        int32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4)
            break;
        if (fread(&chunk_size, 4, 1, f) != 1)
            break;
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            char fmt[16];
            if (chunk_size < 16 || fread(fmt, 1, 16, f) != 16)
                break;
            channels = *(int16_t*)(fmt + 2);
            sample_rate = *(int32_t*)(fmt + 4);
            bits = *(int16_t*)(fmt + 14);
            found_fmt = true;
            if (chunk_size > 16)
                fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data || bits != 16 || channels < 1) {
        fclose(f);
        return pcm;
    }

    int n_samples = data_size / (channels * (bits / 8));
    std::vector<int16_t> raw(n_samples * channels);
    if ((int)fread(raw.data(), sizeof(int16_t), n_samples * channels, f) != n_samples * channels) {
        fclose(f);
        return pcm;
    }
    fclose(f);

    pcm.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int c = 0; c < channels; c++)
            sum += raw[i * channels + c];
        pcm[i] = sum / (channels * 32768.0f);
    }
    return pcm;
}

TEST_CASE("align-only: live alignment with canary-ctc-aligner", "[live][align]") {
    std::string model = get_env("CRISPASR_MODEL_ALIGNER");
    if (model.empty()) {
        // Fallback: try well-known paths.
        const char* dir = std::getenv("CRISPASR_MODELS_DIR");
        if (dir) {
            model = std::string(dir) + "/canary-ctc-aligner-q4_k.gguf";
            FILE* f = fopen(model.c_str(), "rb");
            if (f)
                fclose(f);
            else
                model.clear();
        }
    }
    if (model.empty())
        SKIP("CRISPASR_MODEL_ALIGNER not set and no fallback found");

    std::string audio_path = get_env("CRISPASR_TEST_AUDIO_JFK", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio_path);
    if (pcm.empty())
        SKIP("Cannot load audio: " + audio_path);

    REQUIRE(pcm.size() > 16000); // at least 1 second

    const std::string transcript = "And so my fellow Americans ask not what your country can do for you "
                                   "ask what you can do for your country";

    auto aligned = crispasr_align_words(model, transcript, pcm.data(), (int)pcm.size(), 0, 4);

    REQUIRE(!aligned.empty());

    // Verify basic properties:
    // 1. Number of words matches transcript word count.
    int word_count = 1;
    for (char c : transcript)
        if (c == ' ')
            word_count++;
    CHECK(aligned.size() == (size_t)word_count);

    // 2. Timestamps are monotonically non-decreasing.
    for (size_t i = 1; i < aligned.size(); i++) {
        CHECK(aligned[i].t0_cs >= aligned[i - 1].t0_cs);
    }

    // 3. All timestamps are within the audio duration.
    int64_t max_cs = (int64_t)(pcm.size() * 100 / 16000) + 10; // +10cs tolerance
    for (auto& w : aligned) {
        CHECK(w.t0_cs >= 0);
        CHECK(w.t0_cs <= max_cs);
        CHECK(w.t1_cs >= 0);
        CHECK(w.t1_cs <= max_cs);
    }

    // 4. First word starts near the beginning (within first 3 seconds).
    CHECK(aligned[0].t0_cs < 300);

    // 5. Words have non-empty text.
    for (auto& w : aligned) {
        CHECK(!w.text.empty());
    }

    crispasr_aligner_free_cache();
}

// §176e (wav2vec2): the aligner keeps one model resident across calls so a
// 300 MB–1 GB GGUF is not reloaded on every crispasr_align_words. This test
// aligns twice with the SAME wav2vec2 model and checks (a) the two calls are
// byte-identical (a stale/corrupted cache would drift) and (b) the second
// call is faster (the first pays the one-time load + Metal pipeline compile;
// the second reuses the resident model). Needs a wav2vec2 aligner GGUF.
TEST_CASE("align-only: wav2vec2 aligner cache reuse", "[live][align]") {
    std::string model = get_env("CRISPASR_MODEL_ALIGNER_W2V");
    if (model.empty()) {
        const char* dir = std::getenv("CRISPASR_MODELS_DIR");
        if (dir) {
            model = std::string(dir) + "/wav2vec2-xlsr-en-q4_k.gguf";
            FILE* f = fopen(model.c_str(), "rb");
            if (f)
                fclose(f);
            else
                model.clear();
        }
    }
    if (model.empty())
        SKIP("CRISPASR_MODEL_ALIGNER_W2V not set and no wav2vec2 fallback found");

    std::string audio_path = get_env("CRISPASR_TEST_AUDIO_JFK", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio_path);
    if (pcm.empty())
        SKIP("Cannot load audio: " + audio_path);
    REQUIRE(pcm.size() > 16000);

    const std::string transcript = "And so my fellow Americans ask not what your country can do for you "
                                   "ask what you can do for your country";

    // Start from a clean cache so call 1 definitely pays the load.
    crispasr_aligner_free_cache();

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    auto first = crispasr_align_words(model, transcript, pcm.data(), (int)pcm.size(), 0, 4);
    auto t1 = clock::now();
    auto second = crispasr_align_words(model, transcript, pcm.data(), (int)pcm.size(), 0, 4);
    auto t2 = clock::now();

    REQUIRE(!first.empty());
    REQUIRE(!second.empty());

    double ms_load = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_cached = std::chrono::duration<double, std::milli>(t2 - t1).count();
    INFO("call1 (load+align) = " << ms_load << " ms, call2 (cached align) = " << ms_cached << " ms");

    // (a) Byte-identical results across the two calls — the resident model +
    //     its mutable dequant tensor_cache must not perturb the output.
    REQUIRE(first.size() == second.size());
    for (size_t i = 0; i < first.size(); i++) {
        CHECK(first[i].text == second[i].text);
        CHECK(first[i].t0_cs == second[i].t0_cs);
        CHECK(first[i].t1_cs == second[i].t1_cs);
    }

    // (b) The cached second call skips the model reload, so it is faster. Use a
    //     modest margin (5 %) to stay robust to timing noise on a loaded box;
    //     reloading a multi-hundred-MB model dwarfs one short-clip forward pass.
    CHECK(ms_cached < ms_load * 0.95);

    crispasr_aligner_free_cache();
}

TEST_CASE("align-only: live alignment from SRT text input", "[live][align]") {
    std::string model = get_env("CRISPASR_MODEL_ALIGNER");
    if (model.empty()) {
        const char* dir = std::getenv("CRISPASR_MODELS_DIR");
        if (dir) {
            model = std::string(dir) + "/canary-ctc-aligner-q4_k.gguf";
            FILE* f = fopen(model.c_str(), "rb");
            if (f)
                fclose(f);
            else
                model.clear();
        }
    }
    if (model.empty())
        SKIP("CRISPASR_MODEL_ALIGNER not set and no fallback found");

    std::string audio_path = get_env("CRISPASR_TEST_AUDIO_JFK", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio_path);
    if (pcm.empty())
        SKIP("Cannot load audio: " + audio_path);

    // Simulate text extracted from an unaligned SRT.
    const char* srt_content = "1\n"
                              "00:00:00,000 --> 00:00:30,000\n"
                              "And so my fellow Americans ask not what your country can do for you "
                              "ask what you can do for your country\n"
                              "\n";
    std::string transcript = extract_srt_text(srt_content);
    REQUIRE(!transcript.empty());

    auto aligned = crispasr_align_words(model, transcript, pcm.data(), (int)pcm.size(), 0, 4);

    REQUIRE(!aligned.empty());
    // The alignment from SRT text should produce the same result as from plain text.
    CHECK(aligned.size() > 10); // at least 10 words aligned

    // Timestamps should span most of the audio duration.
    int64_t last_t1 = aligned.back().t1_cs;
    int64_t audio_dur_cs = (int64_t)(pcm.size() * 100 / 16000);
    // Last word should end within the last 3 seconds of the audio.
    CHECK(last_t1 > audio_dur_cs - 300);

    crispasr_aligner_free_cache();
}
