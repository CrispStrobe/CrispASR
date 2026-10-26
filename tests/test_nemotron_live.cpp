// tests/test_nemotron_live.cpp — integration tests for nemotron backend.
//
// Live tests: require model + audio files on disk. Gated behind [.live].
// Run: ctest -R test-nemotron --output-on-failure
//
// Env vars:
//   CRISPASR_MODEL_NEMOTRON    — qualified default GGUF path (F16)
//   CRISPASR_MODEL_NEMOTRON_F16 — F16 GGUF path (optional)
//   CRISPASR_MODEL_NEMOTRON_Q4  — explicitly selected historical Q4 diagnostic

#include <catch2/catch_test_macros.hpp>
#include "nemotron.h"

#include <algorithm>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

static std::string get_env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

struct scoped_env {
    std::string name;
    std::string old_value;
    bool had_value;
    scoped_env(const char* key, const char* value) : name(key), had_value(std::getenv(key) != nullptr) {
        if (had_value)
            old_value = std::getenv(key);
#ifdef _WIN32
        _putenv_s(key, value);
#else
        setenv(key, value, 1);
#endif
    }
    ~scoped_env() {
#ifdef _WIN32
        _putenv_s(name.c_str(), had_value ? old_value.c_str() : "");
#else
        if (had_value)
            setenv(name.c_str(), old_value.c_str(), 1);
        else
            unsetenv(name.c_str());
#endif
    }
};

// Chunk-skipping 16-bit PCM WAV loader.
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
    fread(raw.data(), sizeof(int16_t), raw.size(), f);
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

TEST_CASE("nemotron: init and free", "[nemotron][.live]") {
    std::string model = get_env("CRISPASR_MODEL_NEMOTRON");
    if (model.empty()) {
        SKIP("CRISPASR_MODEL_NEMOTRON not set");
    }

    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    nemotron_context* ctx = nemotron_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);
    nemotron_free(ctx);
}

TEST_CASE("nemotron: JFK English transcription", "[nemotron][.live]") {
    std::string model = get_env("CRISPASR_MODEL_NEMOTRON");
    if (model.empty()) {
        SKIP("CRISPASR_MODEL_NEMOTRON not set");
    }

    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    nemotron_context* ctx = nemotron_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    char* text = nemotron_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(text != nullptr);

    std::string got(text);
    std::free(text);
    INFO("Transcription: " << got);

    // Check key phrases are present (case-insensitive substring match)
    auto contains_ci = [](const std::string& hay, const char* needle) {
        std::string h = hay, n = needle;
        for (auto& c : h)
            c = (char)tolower((unsigned char)c);
        for (auto& c : n)
            c = (char)tolower((unsigned char)c);
        return h.find(n) != std::string::npos;
    };

    CHECK(contains_ci(got, "fellow americans"));
    CHECK(contains_ci(got, "country"));
    CHECK(got.size() > 30);

    nemotron_free(ctx);
}

TEST_CASE("nemotron: persistent stream matches one-shot chunked decode", "[nemotron][.live][streaming]") {
    std::string model = get_env("CRISPASR_MODEL_NEMOTRON");
    if (model.empty())
        SKIP("CRISPASR_MODEL_NEMOTRON not set");
    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());
    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    nemotron_context* ctx = nemotron_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    nemotron_set_context_preset(ctx, 0);
    scoped_env streaming_env("CRISPASR_NEMOTRON_STREAMING", "1");
    scoped_env step_env("CRISPASR_NEMOTRON_STREAM_CHUNKS_PER_STEP", "4");
    char* expected_raw = nemotron_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(expected_raw != nullptr);
    std::string expected(expected_raw);
    std::free(expected_raw);

    nemotron_stream* stream = nemotron_stream_create(ctx);
    REQUIRE(stream != nullptr);
    std::string actual;
    int tokens_before_flush = 0;
    auto cb = [](int id, float, void* userdata) {
        auto* pair = static_cast<std::pair<nemotron_context*, std::string*>*>(userdata);
        std::string piece = nemotron_token_to_str(pair->first, id);
        size_t pos = 0;
        while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
            piece.replace(pos, 3, " ");
            pos++;
        }
        *pair->second += piece;
    };
    std::pair<nemotron_context*, std::string*> state{ctx, &actual};
    const int append_samples = 320 * 16;
    int previous_frames = 0;
    for (size_t offset = 0; offset < pcm.size(); offset += append_samples) {
        int count = (int)std::min((size_t)append_samples, pcm.size() - offset);
        REQUIRE(nemotron_stream_append(stream, pcm.data() + offset, count, false, cb, &state));
        const int current_frames = nemotron_stream_processed_frames(stream);
        REQUIRE(current_frames >= previous_frames);
        // Four native chunks per step (pinned above). Growth is bounded; a
        // growing-prefix implementation would jump by total T.
        REQUIRE(current_frames - previous_frames <= 16);
        previous_frames = current_frames;
        if (offset + count < pcm.size())
            tokens_before_flush = (int)actual.size();
    }
    REQUIRE(tokens_before_flush > 0);
    REQUIRE(nemotron_stream_append(stream, nullptr, 0, true, cb, &state));
    while (!actual.empty() && actual.front() == ' ')
        actual.erase(actual.begin());
    INFO("one-shot: " << expected);
    INFO("streamed: " << actual);
    CHECK(actual == expected);

    nemotron_stream_reset(stream);
    CHECK(nemotron_stream_processed_frames(stream) == 0);
    nemotron_stream_free(stream);
    nemotron_free(ctx);
}

// A real conversation is short utterance -> silence -> short utterance -> ... A
// turn that transcribes fine alone must also transcribe as the 2nd/3rd turn of
// a session. Issue #547's reporter suspected the encoder cache was corrupted
// turn-to-turn; this pins that invariant directly, on both the shared one-shot
// context and the persistent stream reset between turns. (The empty `final` the
// report actually saw was the CLI cadence throttle skipping a short step's
// decode, not backend state — see `crispasr_stream_partial_decode.h` and
// `tools/ci-heavy/issue547_nemotron_multiturn.py` for that guard.)
static std::vector<std::vector<float>> jfk_utterance_segments(const std::vector<float>& jfk, int sample_rate) {
    auto seg = [&](double a, double b) {
        const int s = (int)(a * sample_rate);
        const int e = std::min<int>((int)(b * sample_rate), (int)jfk.size());
        return std::vector<float>(jfk.begin() + s, jfk.begin() + e);
    };
    // Four distinct speech regions of the 11 s JFK clip.
    return {seg(0.0, 2.0), seg(2.5, 5.8), seg(6.2, 8.2), seg(8.6, 11.0)};
}

static std::string collect_transcribe(nemotron_context* ctx, const float* pcm, int n) {
    std::string text;
    auto cb = [](int id, float, void* ud) {
        auto* pair = static_cast<std::pair<nemotron_context*, std::string*>*>(ud);
        std::string piece = nemotron_token_to_str(pair->first, id);
        size_t pos = 0;
        while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
            piece.replace(pos, 3, " ");
            pos++;
        }
        *pair->second += piece;
    };
    std::pair<nemotron_context*, std::string*> state{ctx, &text};
    nemotron_transcribe_cb(ctx, pcm, n, cb, &state);
    while (!text.empty() && text.front() == ' ')
        text.erase(text.begin());
    return text;
}

TEST_CASE("nemotron: every turn of a multi-utterance session transcribes", "[nemotron][.live][streaming]") {
    std::string model = get_env("CRISPASR_MODEL_NEMOTRON");
    if (model.empty())
        SKIP("CRISPASR_MODEL_NEMOTRON not set");
    auto jfk = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!jfk.empty());
    auto utts = jfk_utterance_segments(jfk, 16000);
    REQUIRE(utts.size() == 4);

    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = std::max(1, std::atoi(get_env("CRISPASR_TEST_N_THREADS", "4").c_str()));
    cp.use_gpu = get_env("CRISPASR_TEST_CPU_ONLY") != "1";
    cp.verbosity = 0;
    scoped_env step_env("CRISPASR_NEMOTRON_STREAM_CHUNKS_PER_STEP", "4");

    // (a) Shared one-shot context: consecutive calls must stay independent.
    {
        nemotron_context* ctx = nemotron_init_from_file(model.c_str(), cp);
        REQUIRE(ctx != nullptr);
        nemotron_set_context_preset(ctx, 0);
        for (size_t i = 0; i < utts.size(); i++) {
            const std::string got = collect_transcribe(ctx, utts[i].data(), (int)utts[i].size());
            INFO("one-shot turn " << i + 1 << ": " << got);
            CHECK(!got.empty());
        }
        nemotron_free(ctx);
    }

    // (b) Persistent stream reset between turns, fed in live-client-sized
    // pieces. This is the path the streaming session uses and the one whose
    // cache reset the report suspected.
    {
        nemotron_context* ctx = nemotron_init_from_file(model.c_str(), cp);
        REQUIRE(ctx != nullptr);
        nemotron_set_context_preset(ctx, 0);
        nemotron_stream* stream = nemotron_stream_create(ctx);
        REQUIRE(stream != nullptr);
        for (size_t i = 0; i < utts.size(); i++) {
            const auto& u = utts[i];
            std::string text;
            auto cb = [](int id, float, void* ud) {
                auto* pair = static_cast<std::pair<nemotron_context*, std::string*>*>(ud);
                std::string piece = nemotron_token_to_str(pair->first, id);
                size_t pos = 0;
                while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
                    piece.replace(pos, 3, " ");
                    pos++;
                }
                *pair->second += piece;
            };
            std::pair<nemotron_context*, std::string*> state{ctx, &text};
            const int piece = 1600; // 100 ms, like a live client
            for (size_t off = 0; off < u.size(); off += piece) {
                const int count = (int)std::min((size_t)piece, u.size() - off);
                REQUIRE(nemotron_stream_append(stream, u.data() + off, count, false, cb, &state));
            }
            REQUIRE(nemotron_stream_append(stream, nullptr, 0, true, cb, &state));
            while (!text.empty() && text.front() == ' ')
                text.erase(text.begin());
            INFO("stream turn " << i + 1 << ": " << text);
            CHECK(!text.empty());
            nemotron_stream_reset(stream);
            CHECK(nemotron_stream_processed_frames(stream) == 0);
        }
        nemotron_stream_free(stream);
        nemotron_free(ctx);
    }
}

struct stream_output {
    std::string text;
    std::string tokens; // "id:probability" in hex float, so rounding differences show
};

using stream_capture = std::pair<nemotron_context*, stream_output>;

static void collect_token(int id, float p, void* userdata) {
    auto* st = static_cast<stream_capture*>(userdata);
    std::string piece = nemotron_token_to_str(st->first, id);
    size_t pos = 0;
    while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
        piece.replace(pos, 3, " ");
        pos++;
    }
    st->second.text += piece;
    char buf[48];
    snprintf(buf, sizeof(buf), "%d:%a ", id, p);
    st->second.tokens += buf;
}

// Streams pcm in 100 ms pieces, like a live client. Checks that frames only move
// forward, that one step adds at most two chunks, and that text arrives before
// the end.
static stream_output stream_like_live_client(nemotron_context* ctx, const std::vector<float>& pcm, int chunk) {
    nemotron_stream* stream = nemotron_stream_create(ctx);
    REQUIRE(stream != nullptr);
    stream_capture state{ctx, {}};
    int tokens_before_flush = 0;
    const int append_samples = 1600;
    int previous_frames = 0;
    for (size_t offset = 0; offset < pcm.size(); offset += append_samples) {
        int count = (int)std::min((size_t)append_samples, pcm.size() - offset);
        REQUIRE(nemotron_stream_append(stream, pcm.data() + offset, count, false, collect_token, &state));
        const int current_frames = nemotron_stream_processed_frames(stream);
        REQUIRE(current_frames >= previous_frames);
        REQUIRE(current_frames - previous_frames <= 2 * chunk);
        previous_frames = current_frames;
        if (offset + count < pcm.size())
            tokens_before_flush = (int)state.second.text.size();
    }
    REQUIRE(tokens_before_flush > 0);
    REQUIRE(nemotron_stream_append(stream, nullptr, 0, true, collect_token, &state));
    nemotron_stream_free(stream);
    std::string& text = state.second.text;
    while (!text.empty() && text.front() == ' ')
        text.erase(text.begin());
    return state.second;
}

// One-shot and incremental chunk feeding already differ in punctuation for
// Q4 on the full-recompute control. Keep that auxiliary JFK comparison about
// spoken words; the window/control text and every token confidence stay exact.
static std::string jfk_words(const std::string& text) {
    std::string words;
    for (unsigned char c : text) {
        if (std::isalpha(c))
            words += (char)std::tolower(c);
        else if (!words.empty() && words.back() != ' ')
            words += ' ';
    }
    if (!words.empty() && words.back() == ' ')
        words.pop_back();
    return words;
}

TEST_CASE("nemotron: realtime stream gives the same output as a full recompute", "[nemotron][.live][streaming]") {
    std::string model = get_env("CRISPASR_MODEL_NEMOTRON");
    if (model.empty())
        SKIP("CRISPASR_MODEL_NEMOTRON not set");
    auto jfk = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!jfk.empty());
    // About 33 s, so the window moves and old audio is dropped many times.
    std::vector<float> pcm;
    for (int i = 0; i < 3; i++)
        pcm.insert(pcm.end(), jfk.begin(), jfk.end());

    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = std::max(1, std::atoi(get_env("CRISPASR_TEST_N_THREADS", "4").c_str()));
    cp.use_gpu = get_env("CRISPASR_TEST_CPU_ONLY") != "1";
    cp.verbosity = cp.use_gpu ? 1 : 0;
    nemotron_context* ctx = nemotron_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);
    scoped_env step_env("CRISPASR_NEMOTRON_STREAM_CHUNKS_PER_STEP", "1");

    // Exercise the window explicitly on every backend, including GPU, where
    // it remains experimental after failing exact confidence parity.
    scoped_env window_env("CRISPASR_NEMOTRON_STREAM_INCREMENTAL_FRONTEND", "1");

    // Same words and the same confidence for every token as a full recompute.
    // Presets 2 and 3 have 7 and 14 frame chunks, so they exercise the
    // multiple-of-4 start; preset 0's 4 frame chunks always land on one.
    for (const auto& [preset, chunk] :
         {std::pair<int, int>{0, 4}, std::pair<int, int>{2, 7}, std::pair<int, int>{3, 14}}) {
        INFO("preset " << preset);
        nemotron_set_context_preset(ctx, preset);
        const stream_output newest_only = stream_like_live_client(ctx, pcm, chunk);
        stream_output full;
        {
            scoped_env full_env("CRISPASR_NEMOTRON_STREAM_FULL_RECOMPUTE", "1");
            full = stream_like_live_client(ctx, pcm, chunk);
        }
        INFO("full recompute: " << full.text);
        INFO("newest only:    " << newest_only.text);
        CHECK(newest_only.text == full.text);
        CHECK(newest_only.tokens == full.tokens);

        // At preset 0 the text also matches decoding the whole clip at once.
        if (preset == 0) {
            scoped_env streaming_env("CRISPASR_NEMOTRON_STREAMING", "1");
            char* expected_raw = nemotron_transcribe(ctx, pcm.data(), (int)pcm.size());
            REQUIRE(expected_raw != nullptr);
            std::string expected(expected_raw);
            std::free(expected_raw);
            INFO("one-shot: " << expected);
            INFO("full control already differs from one-shot: " << (full.text != expected));
            CHECK(jfk_words(full.text) == jfk_words(expected));
            CHECK(jfk_words(newest_only.text) == jfk_words(expected));
        }
    }
    nemotron_free(ctx);
}

// Capture the production GPU default independently of the experimental window.
// The hosted harness runs this against both the frozen pre-PR source and the
// candidate, then compares all text and token confidence bytes.
TEST_CASE("nemotron: GPU production stream captures complete speech", "[nemotron][.live][streaming]") {
    const std::string model = get_env("CRISPASR_MODEL_NEMOTRON");
    if (model.empty())
        SKIP("CRISPASR_MODEL_NEMOTRON not set");
    auto jfk = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!jfk.empty());
    std::vector<float> pcm;
    for (int i = 0; i < 3; i++)
        pcm.insert(pcm.end(), jfk.begin(), jfk.end());
    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 4;
    cp.use_gpu = true;
    cp.verbosity = 1;
    auto* ctx = nemotron_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);
    scoped_env window_env("CRISPASR_NEMOTRON_STREAM_INCREMENTAL_FRONTEND", "0");
    scoped_env full_env("CRISPASR_NEMOTRON_STREAM_FULL_RECOMPUTE", "0");
    scoped_env step_env("CRISPASR_NEMOTRON_STREAM_CHUNKS_PER_STEP", "1");
    const std::string output = get_env("CRISPASR_TEST_STREAM_CAPTURE");
    REQUIRE(!output.empty());
    std::ofstream capture(output, std::ios::binary);
    REQUIRE(capture.good());
    for (const auto& [preset, chunk] :
         {std::pair<int, int>{0, 4}, std::pair<int, int>{2, 7}, std::pair<int, int>{3, 14}}) {
        INFO("preset " << preset);
        nemotron_set_context_preset(ctx, preset);
        const auto actual = stream_like_live_client(ctx, pcm, chunk);
        INFO(actual.text);
        const std::string words = jfk_words(actual.text);
        for (const std::string phrase : {"my fellow americans", "ask not what your country can do for you",
                                         "ask what you can do for your country"}) {
            size_t count = 0, pos = 0;
            while ((pos = words.find(phrase, pos)) != std::string::npos) {
                count++;
                pos += phrase.size();
            }
            CHECK(count == 3);
        }
        capture << preset << '\n' << actual.text << '\n' << actual.tokens << '\n';
    }
    REQUIRE(capture.good());
    nemotron_free(ctx);
}

// This one-clip diagnostic does not qualify Q4: the fixed original-reference
// matrix rejects it. Keep the quant input explicit when the default is F16.
TEST_CASE("nemotron: explicit Q4_K diagnostic matches F16 on JFK only", "[nemotron][.live]") {
    std::string model_q4k = get_env("CRISPASR_MODEL_NEMOTRON_Q4");
    std::string model_f16 = get_env("CRISPASR_MODEL_NEMOTRON_F16");
    if (model_q4k.empty() || model_f16.empty()) {
        SKIP("Explicit CRISPASR_MODEL_NEMOTRON_Q4 or CRISPASR_MODEL_NEMOTRON_F16 not set");
    }
    REQUIRE(model_q4k != model_f16);

    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;

    // Q4_K
    nemotron_context* ctx_q = nemotron_init_from_file(model_q4k.c_str(), cp);
    REQUIRE(ctx_q != nullptr);
    char* t_q = nemotron_transcribe(ctx_q, pcm.data(), (int)pcm.size());
    REQUIRE(t_q != nullptr);
    std::string text_q(t_q);
    std::free(t_q);
    nemotron_free(ctx_q);

    // F16
    nemotron_context* ctx_f = nemotron_init_from_file(model_f16.c_str(), cp);
    REQUIRE(ctx_f != nullptr);
    char* t_f = nemotron_transcribe(ctx_f, pcm.data(), (int)pcm.size());
    REQUIRE(t_f != nullptr);
    std::string text_f(t_f);
    std::free(t_f);
    nemotron_free(ctx_f);

    INFO("Q4_K: " << text_q);
    INFO("F16:  " << text_f);
    CHECK(text_q == text_f);
}
