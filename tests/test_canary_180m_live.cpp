// Live acceptance coverage for NVIDIA Canary 180M Flash.
//
// Required ASR model:
//   CRISPASR_MODEL_CANARY_180M=/path/to/canary-180m-flash-Q4_K_M.gguf
// Optional translation model (Q5_K_M or better; Q4 immediately emits EOS):
//   CRISPASR_MODEL_CANARY_180M_TRANSLATE=/path/to/canary-180m-flash-Q5_K_M.gguf
// Audio:
//   CRISPASR_AUDIO_CANARY_180M=/path/to/jfk-16k-mono.wav
//   CRISPASR_AUDIO_CANARY_180M_JFK_X4=/path/to/jfk-repeated-4-times-44s.wav
// Optional legacy load regression:
//   CRISPASR_MODEL_CANARY_LEGACY=/path/to/canary-1b-v2-q4_k.gguf
//   (falls back to CRISPASR_MODEL_CANARY)

#include <catch2/catch_test_macros.hpp>

#include "canary.h"
#include "crispasr_session.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr const char* kEnglishPnc =
    "And so my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.";
constexpr const char* kEnglishNoPnc =
    "and so my fellow americans ask not what your country can do for you ask what you can do for your country";
constexpr const char* kGermanPnc =
    "Und so, meine Landsleute, fragen Sie nicht, was Ihr Land für Sie tun kann. Fragen Sie, was Sie für Ihr Land tun "
    "können.";

std::string env(const char* name, const char* fallback = "") {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

bool file_exists(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file)
        return false;
    std::fclose(file);
    return true;
}

std::string normalize_space(const std::string& text) {
    std::string result;
    bool pending_space = false;
    for (unsigned char ch : text) {
        if (std::isspace(ch)) {
            pending_space = !result.empty();
            continue;
        }
        if (pending_space)
            result.push_back(' ');
        result.push_back((char)ch);
        pending_space = false;
    }
    return result;
}

std::string normalize_words(const std::string& text) {
    std::string words;
    bool pending_space = false;
    for (unsigned char ch : text) {
        if (std::isalnum(ch)) {
            if (pending_space && !words.empty())
                words.push_back(' ');
            words.push_back((char)std::tolower(ch));
            pending_space = false;
        } else {
            pending_space = !words.empty();
        }
    }
    return words;
}

int count_occurrences(const std::string& text, const std::string& needle) {
    int count = 0;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

std::vector<float> load_wav_16k_mono(const std::string& path) {
    std::vector<float> pcm;
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file)
        return pcm;

    char riff[12];
    if (std::fread(riff, 1, sizeof(riff), file) != sizeof(riff) || std::memcmp(riff, "RIFF", 4) != 0 ||
        std::memcmp(riff + 8, "WAVE", 4) != 0) {
        std::fclose(file);
        return pcm;
    }

    int channels = 0;
    int sample_rate = 0;
    int bits = 0;
    int32_t data_size = 0;
    bool found_fmt = false;
    bool found_data = false;
    while (!found_data) {
        char chunk_id[4];
        int32_t chunk_size = 0;
        if (std::fread(chunk_id, 1, sizeof(chunk_id), file) != sizeof(chunk_id) ||
            std::fread(&chunk_size, sizeof(chunk_size), 1, file) != 1 || chunk_size < 0) {
            break;
        }
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            unsigned char fmt[16];
            if (chunk_size < (int32_t)sizeof(fmt) || std::fread(fmt, 1, sizeof(fmt), file) != sizeof(fmt))
                break;
            std::memcpy(&channels, fmt + 2, sizeof(int16_t));
            std::memcpy(&sample_rate, fmt + 4, sizeof(int32_t));
            std::memcpy(&bits, fmt + 14, sizeof(int16_t));
            found_fmt = true;
            if (chunk_size > (int32_t)sizeof(fmt))
                std::fseek(file, chunk_size - (long)sizeof(fmt), SEEK_CUR);
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
        } else {
            std::fseek(file, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data || channels < 1 || sample_rate != 16000 || bits != 16) {
        std::fclose(file);
        return pcm;
    }

    const int n_samples = data_size / (channels * (int)sizeof(int16_t));
    std::vector<int16_t> raw((size_t)n_samples * channels);
    const size_t n_read = std::fread(raw.data(), sizeof(int16_t), raw.size(), file);
    std::fclose(file);
    if (n_read != raw.size())
        return {};

    pcm.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        float sum = 0.0f;
        for (int channel = 0; channel < channels; ++channel)
            sum += raw[(size_t)i * channels + channel];
        pcm[i] = sum / (channels * 32768.0f);
    }
    return pcm;
}

canary_context* open_canary(const std::string& model) {
    canary_context_params params = canary_context_default_params();
    params.n_threads = 4;
    params.verbosity = 0;
    params.use_gpu = false;
    return canary_init_from_file(model.c_str(), params);
}

std::string session_text(crispasr_session_result* result) {
    std::string text;
    const int n_segments = crispasr_session_result_n_segments(result);
    for (int i = 0; i < n_segments; ++i) {
        const char* segment = crispasr_session_result_segment_text(result, i);
        if (segment)
            text += segment;
    }
    return normalize_space(text);
}

struct StageCapture {
    int layer_callbacks = 0;
    std::set<std::string> layer_names;
    std::vector<int> layer_widths;
};

void capture_stage(const char* name, const float*, int, int d_model, void* userdata) {
    auto* capture = static_cast<StageCapture*>(userdata);
    if (name && std::strncmp(name, "enc_L", 5) == 0) {
        ++capture->layer_callbacks;
        capture->layer_names.emplace(name);
        capture->layer_widths.push_back(d_model);
    }
}

void require_q4_k_m_file(const std::string& model) {
    gguf_init_params params = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* gguf = gguf_init_from_file(model.c_str(), params);
    REQUIRE(gguf != nullptr);

    const int64_t file_type_key = gguf_find_key(gguf, "general.file_type");
    REQUIRE(file_type_key >= 0);
    REQUIRE(gguf_get_kv_type(gguf, file_type_key) == GGUF_TYPE_UINT32);
    CHECK(gguf_get_val_u32(gguf, file_type_key) == 15); // llama.cpp's MOSTLY_Q4_K_M

    int q4_k_tensors = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        if (gguf_get_tensor_type(gguf, i) == GGML_TYPE_Q4_K)
            ++q4_k_tensors;
    }
    CHECK(q4_k_tensors > 0);
    gguf_free(gguf);
}

} // namespace

TEST_CASE("canary 180M Q4_K_M ASR covers low-level and auto-detected session APIs", "[canary-180m][.live]") {
    const std::string model = env("CRISPASR_MODEL_CANARY_180M");
    if (model.empty())
        SKIP("CRISPASR_MODEL_CANARY_180M not set");
    if (!file_exists(model))
        SKIP("CRISPASR_MODEL_CANARY_180M does not exist");

    const std::string audio_path = env("CRISPASR_AUDIO_CANARY_180M");
    if (audio_path.empty())
        SKIP("CRISPASR_AUDIO_CANARY_180M not set");
    if (!file_exists(audio_path))
        SKIP("CRISPASR_AUDIO_CANARY_180M does not exist");
    const std::vector<float> pcm = load_wav_16k_mono(audio_path);
    REQUIRE(!pcm.empty());

    require_q4_k_m_file(model);

    canary_context* ctx = open_canary(model);
    REQUIRE(ctx != nullptr);
    CHECK(canary_n_vocab(ctx) == 5248);
    CHECK(canary_n_mels(ctx) == 128);
    CHECK(canary_sample_rate(ctx) == 16000);

    int n_mels = 0;
    int n_frames = 0;
    float* mel = canary_compute_mel(ctx, pcm.data(), (int)pcm.size(), &n_mels, &n_frames);
    REQUIRE(mel != nullptr);
    CHECK(n_mels == 128);

    int n_encoded = 0;
    int native_width = 0;
    float* encoded = canary_run_encoder(ctx, mel, n_mels, n_frames, &n_encoded, &native_width);
    REQUIRE(encoded != nullptr);
    CHECK(n_encoded > 0);
    CHECK(native_width == 512);
    std::free(encoded);

    StageCapture stages;
    REQUIRE(canary_run_encoder_staged(ctx, mel, n_mels, n_frames, capture_stage, &stages) == 0);
    CHECK(stages.layer_callbacks == 17);
    CHECK(stages.layer_names.size() == 17);
    CHECK(std::all_of(stages.layer_widths.begin(), stages.layer_widths.end(), [](int width) { return width == 512; }));
    std::free(mel);

    char* no_pnc = canary_transcribe(ctx, pcm.data(), (int)pcm.size(), "en", "en", false);
    REQUIRE(no_pnc != nullptr);
    const std::string no_pnc_text = normalize_space(no_pnc);
    std::free(no_pnc);
    INFO("Canary 180M no-PNC: " << no_pnc_text);
    CHECK(no_pnc_text == kEnglishNoPnc);

    char* unsupported_language = canary_transcribe(ctx, pcm.data(), (int)pcm.size(), "zz", "zz", true);
    CHECK(unsupported_language == nullptr);
    std::free(unsupported_language);
    char* unsupported_pair = canary_transcribe(ctx, pcm.data(), (int)pcm.size(), "de", "es", true);
    CHECK(unsupported_pair == nullptr);
    std::free(unsupported_pair);
    canary_free(ctx);

    char detected[32] = {};
    REQUIRE(crispasr_detect_backend_from_gguf(model.c_str(), detected, (int)sizeof(detected)) == 6);
    CHECK(std::string(detected) == "canary");

    crispasr_session* session = crispasr_session_open(model.c_str(), 4);
    REQUIRE(session != nullptr);
    REQUIRE(crispasr_session_backend(session) != nullptr);
    CHECK(std::string(crispasr_session_backend(session)) == "canary");
    REQUIRE(crispasr_session_set_source_language(session, "en") == 0);
    REQUIRE(crispasr_session_set_target_language(session, "en") == 0);
    REQUIRE(crispasr_session_set_punctuation(session, 1) == 0);

    crispasr_session_result* result = crispasr_session_transcribe(session, pcm.data(), (int)pcm.size());
    REQUIRE(result != nullptr);
    const std::string text = session_text(result);
    INFO("Canary 180M session ASR: " << text);
    CHECK(text == kEnglishPnc);
    crispasr_session_result_free(result);
    crispasr_session_close(session);
}

TEST_CASE("canary 180M Q5+ translates JFK from English to German", "[canary-180m][.live][translation]") {
    const std::string model = env("CRISPASR_MODEL_CANARY_180M_TRANSLATE");
    if (model.empty())
        SKIP("CRISPASR_MODEL_CANARY_180M_TRANSLATE not set (Q4 translation intentionally is not required)");
    if (!file_exists(model))
        SKIP("CRISPASR_MODEL_CANARY_180M_TRANSLATE does not exist");

    const std::string audio_path = env("CRISPASR_AUDIO_CANARY_180M");
    if (audio_path.empty())
        SKIP("CRISPASR_AUDIO_CANARY_180M not set");
    if (!file_exists(audio_path))
        SKIP("CRISPASR_AUDIO_CANARY_180M does not exist");
    const std::vector<float> pcm = load_wav_16k_mono(audio_path);
    REQUIRE(!pcm.empty());

    canary_context* ctx = open_canary(model);
    REQUIRE(ctx != nullptr);
    char* translated = canary_transcribe(ctx, pcm.data(), (int)pcm.size(), "en", "de", true);
    REQUIRE(translated != nullptr);
    const std::string text = normalize_space(translated);
    std::free(translated);
    INFO("Canary 180M EN->DE: " << text);
    CHECK(text == kGermanPnc);
    canary_free(ctx);
}

TEST_CASE("canary 180M long-form centered stitching preserves repeated speech and timings",
          "[canary-180m][.live][long-form]") {
    const std::string model = env("CRISPASR_MODEL_CANARY_180M");
    if (model.empty())
        SKIP("CRISPASR_MODEL_CANARY_180M not set");
    if (!file_exists(model))
        SKIP("CRISPASR_MODEL_CANARY_180M does not exist");

    const std::string audio_path = env("CRISPASR_AUDIO_CANARY_180M_JFK_X4");
    if (audio_path.empty())
        SKIP("CRISPASR_AUDIO_CANARY_180M_JFK_X4 not set");
    if (!file_exists(audio_path))
        SKIP("CRISPASR_AUDIO_CANARY_180M_JFK_X4 does not exist");
    const std::vector<float> pcm = load_wav_16k_mono(audio_path);
    REQUIRE(!pcm.empty());

    canary_context* ctx = open_canary(model);
    REQUIRE(ctx != nullptr);
    canary_result* result = canary_transcribe_streamed(ctx, pcm.data(), (int)pcm.size(), "en", "en", true, 0, 0, -1);
    REQUIRE(result != nullptr);
    REQUIRE(result->text != nullptr);

    const std::string words = normalize_words(result->text);
    const std::string expected_repetition = kEnglishNoPnc;
    std::string expected;
    for (int i = 0; i < 4; ++i) {
        if (!expected.empty())
            expected.push_back(' ');
        expected += expected_repetition;
    }
    INFO("Canary 180M long-form: " << result->text);
    CHECK(words == expected);
    CHECK(count_occurrences(words, expected_repetition) == 4);
    CHECK(count_occurrences(words, "ask not") == 4);

    const int64_t duration_cs = (int64_t)pcm.size() * 100 / 16000;
    REQUIRE(result->n_tokens > 0);
    REQUIRE(result->tokens != nullptr);
    int64_t previous_t0 = 0;
    int64_t previous_t1 = 0;
    for (int i = 0; i < result->n_tokens; ++i) {
        INFO("token " << i << " timing " << result->tokens[i].t0 << ".." << result->tokens[i].t1);
        CHECK(result->tokens[i].t0 >= previous_t0);
        CHECK(result->tokens[i].t1 >= previous_t1);
        CHECK(result->tokens[i].t1 >= result->tokens[i].t0);
        CHECK(result->tokens[i].t0 >= 0);
        CHECK(result->tokens[i].t1 <= duration_cs);
        previous_t0 = result->tokens[i].t0;
        previous_t1 = result->tokens[i].t1;
    }

    REQUIRE(result->n_words == 88);
    REQUIRE(result->words != nullptr);
    previous_t0 = 0;
    previous_t1 = 0;
    for (int i = 0; i < result->n_words; ++i) {
        INFO("word " << i << " timing " << result->words[i].t0 << ".." << result->words[i].t1);
        CHECK(result->words[i].t0 >= previous_t0);
        CHECK(result->words[i].t1 >= previous_t1);
        CHECK(result->words[i].t1 >= result->words[i].t0);
        CHECK(result->words[i].t0 >= 0);
        CHECK(result->words[i].t1 <= duration_cs);
        previous_t0 = result->words[i].t0;
        previous_t1 = result->words[i].t1;
    }

    // The validated four-JFK fixture spans 0.80–44.00 s. Do not silently
    // accept the runtime's documented all-zero fallback for invalid chunk
    // timings here: this fixture has stable, meaningful DTW timings.
    CHECK(result->tokens[0].t0 > 0);
    CHECK(result->tokens[result->n_tokens - 1].t1 >= duration_cs * 9 / 10);
    CHECK(result->words[0].t0 > 0);
    CHECK(result->words[result->n_words - 1].t1 >= duration_cs * 9 / 10);

    canary_result_free(result);
    canary_free(ctx);
}

TEST_CASE("legacy Canary Q4 still loads and transcribes JFK", "[canary-180m][canary-legacy][.live]") {
    std::string model = env("CRISPASR_MODEL_CANARY_LEGACY");
    if (model.empty())
        model = env("CRISPASR_MODEL_CANARY");
    if (model.empty())
        SKIP("CRISPASR_MODEL_CANARY_LEGACY/CRISPASR_MODEL_CANARY not set");
    if (!file_exists(model))
        SKIP("legacy Canary model does not exist");

    const std::string audio_path = env("CRISPASR_AUDIO_CANARY_180M");
    if (audio_path.empty())
        SKIP("CRISPASR_AUDIO_CANARY_180M not set");
    if (!file_exists(audio_path))
        SKIP("CRISPASR_AUDIO_CANARY_180M does not exist");
    const std::vector<float> pcm = load_wav_16k_mono(audio_path);
    REQUIRE(!pcm.empty());

    canary_context* ctx = open_canary(model);
    REQUIRE(ctx != nullptr);
    char* transcript = canary_transcribe(ctx, pcm.data(), (int)pcm.size(), "en", "en", true);
    REQUIRE(transcript != nullptr);
    const std::string words = normalize_words(transcript);
    INFO("Legacy Canary JFK: " << transcript);
    std::free(transcript);
    CHECK(words == kEnglishNoPnc);
    canary_free(ctx);
}
