#pragma once

#include "crispasr_diff.h"
#include "dia_tts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dia_diff {

inline void set_env(const std::string& name, const char* value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value ? value : "");
#else
    if (value)
        setenv(name.c_str(), value, 1);
    else
        unsetenv(name.c_str());
#endif
}

// Replay must neither inherit unrelated debug hooks nor leak them to callers.
struct Environment {
    struct Saved {
        std::string name, value;
        bool present;
    };
    std::vector<Saved> saved;
    Environment() {
        for (const char* suffix : {"DUMP_DIR", "FORCE_TOKENS", "FORCE_OUTPUT_TOKENS", "DUMP_INPUT_TOKENS",
                                   "DUMP_STEPLOGITS", "DECODE_CODES", "MAX_STEPS", "GREEDY", "TTS_GPU"}) {
            for (const char* prefix : {"CRISPASR_DIA_", "DIA_"}) {
                std::string name = std::string(prefix) + suffix;
                const char* value = std::getenv(name.c_str());
                saved.push_back({name, value ? value : "", value != nullptr});
                set_env(name, nullptr);
            }
        }
    }
    ~Environment() {
        for (const auto& item : saved)
            set_env(item.name, item.present ? item.value.c_str() : nullptr);
    }
};

struct Scratch {
    std::filesystem::path path;
    Scratch() {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int i = 0; i < 20; ++i) {
            auto candidate = std::filesystem::temp_directory_path() /
                             ("crispasr-dia-diff-" + std::to_string(nonce) + "-" + std::to_string(i));
            if (std::filesystem::create_directory(candidate)) {
                path = candidate;
                return;
            }
        }
        throw std::runtime_error("Cannot create Dia diff scratch directory");
    }
    ~Scratch() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

inline bool write_codes(const crispasr_diff::Ref& ref, const char* name, const std::filesystem::path& path) {
    const auto data = ref.get_f32(name);
    const auto shape = ref.shape(name);
    if (!data.first || data.second == 0 || shape.size() != 2 || shape[0] != 9 || data.second % 9)
        return false;
    std::vector<int32_t> codes;
    for (size_t i = 0; i < data.second; ++i) {
        const float value = data.first[i];
        if (!std::isfinite(value) || value < 0 || value > 1027 || std::floor(value) != value)
            return false;
        codes.push_back((int32_t)value);
    }
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(codes.data()), codes.size() * sizeof(int32_t));
    return file.good();
}

inline std::vector<float> read_floats(const std::filesystem::path& path, bool integers = false) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() % 4 != 0)
        return {};
    const size_t count = (size_t)file.tellg() / 4;
    file.seekg(0);
    std::vector<float> values(count);
    if (integers) {
        std::vector<int32_t> ids(count);
        file.read(reinterpret_cast<char*>(ids.data()), count * sizeof(int32_t));
        for (size_t i = 0; i < count; ++i)
            values[i] = (float)ids[i];
    } else {
        file.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
    }
    return file ? values : std::vector<float>{};
}

// Preserve the established F16 source-parity gates; integer boundaries are
// exact. Check complete shapes and per-row magnitudes, not just global cosine.
inline bool compare(const crispasr_diff::Ref& ref, const char* name, const std::vector<float>& values,
                    bool exact = false) {
    const auto expected = ref.get_f32(name);
    const auto shape = ref.shape(name);
    if (!expected.first || values.empty() || values.size() != expected.second || shape.empty()) {
        fprintf(stderr, "[FAIL] %-23s missing or mismatched complete tensor (%zu vs %zu)\n", name, values.size(),
                expected.second);
        return false;
    }
    auto result = ref.compare(name, values.data(), values.size(), crispasr_diff::Ref::COS_FIRST_DIM);
    double max_norm_error = 0;
    const size_t width = (size_t)shape.front();
    if (width == 0 || values.size() % width != 0)
        return false;
    for (size_t row = 0; row < values.size(); row += width) {
        double mine = 0, original = 0;
        for (size_t j = 0; j < width; ++j) {
            mine += (double)values[row + j] * values[row + j];
            original += (double)expected.first[row + j] * expected.first[row + j];
        }
        const double ratio = original > 0 ? std::sqrt(mine / original) : (mine == 0 ? 1 : INFINITY);
        max_norm_error = std::max(max_norm_error, std::abs(ratio - 1));
    }
    const bool pass =
        exact ? result.n_nonfinite == 0 && result.max_abs == 0 : result.is_pass(.9995f) && max_norm_error < .02;
    printf("[%s] %-23s cos_min=%.7f |mine|=%.7g |ref|=%.7g max_row_norm_error=%.7g max_abs=%.7g\n",
           pass ? "PASS" : "FAIL", name, result.cos_min, result.rms_data, result.rms_ref, max_norm_error,
           result.max_abs);
    return pass;
}

inline int run(const std::string& model, const std::string& archive) {
    crispasr_diff::Ref ref;
    if (!ref.load(archive))
        return 1;
    const std::string text = ref.meta("dia_text");
    if (ref.meta("dia_source_pin") != "4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8" || text.empty()) {
        fprintf(stderr, "Dia diff requires a pinned original archive and its exact dia_text\n");
        return 1;
    }
    try {
        Environment environment;
        Scratch scratch;
        const auto inputs = scratch.path / "inputs.i32";
        const auto codes = scratch.path / "codes.i32";
        if (!write_codes(ref, "decoder_input_tokens", inputs) || !write_codes(ref, "decoder_codes_rev", codes)) {
            fprintf(stderr, "Dia diff requires complete valid input/code grids\n");
            return 1;
        }
        auto params = dia_tts_context_default_params();
        params.use_gpu = false;
        params.n_threads = 4;
        params.verbosity = 0;
        std::unique_ptr<dia_tts_context, decltype(&dia_tts_free)> ctx(dia_tts_init_from_file(model.c_str(), params),
                                                                      dia_tts_free);
        if (!ctx)
            return 1;
        set_env("CRISPASR_DIA_FORCE_TOKENS", inputs.string().c_str());
        set_env("CRISPASR_DIA_DUMP_DIR", scratch.path.string().c_str());
        set_env("CRISPASR_DIA_DUMP_STEPLOGITS", (scratch.path / "logits.f32").string().c_str());
        set_env("CRISPASR_DIA_DUMP_INPUT_TOKENS", (scratch.path / "feedback.i32").string().c_str());
        int samples = 0;
        // A separate DAC is unnecessary for the forced-logit pass. Null PCM
        // is expected for those files; complete stage files are mandatory.
        dia_tts_pcm_free(dia_tts_synthesize(ctx.get(), text.c_str(), &samples));
        bool pass = compare(ref, "encoder_tokens", read_floats(scratch.path / "cpp_encoder_tokens.f32"), true);
        pass = compare(ref, "decoder_input_tokens", read_floats(scratch.path / "feedback.i32", true), true) && pass;
        if (!pass) {
            fprintf(stderr, "Dia diff: input alignment failed; numerical comparisons are not meaningful\n");
            return 1;
        }
        std::vector<std::pair<std::string, std::string>> stages = {{"encoder_embedding", "cpp_encoder_embedding.f32"}};
        for (int i = 0; i < 12; ++i) {
            const std::string name = "encoder_layer_" + std::to_string(i);
            stages.push_back({name, "cpp_" + name + ".f32"});
        }
        stages.push_back({"encoder_output", "cpp_encoder_output.f32"});
        stages.push_back({"cross_k0", "cpp_cross_k0.f32"});
        stages.push_back({"cross_v0", "cpp_cross_v0.f32"});
        stages.push_back({"decoder_ca0", "cpp_ca0_out.f32"});
        for (int i = 0; i < 18; ++i) {
            const std::string name = "decoder_layer_" + std::to_string(i);
            stages.push_back({name, "cpp_" + name + ".f32"});
        }
        stages.push_back({"decoder_hidden", "cpp_final_hidden.f32"});
        stages.push_back({"decoder_step_logits", "logits.f32"});
        for (const auto& stage : stages)
            pass = compare(ref, stage.first.c_str(), read_floats(scratch.path / stage.second)) && pass;
        const char* codec = std::getenv("DIA_CODEC");
        const std::string codec_path =
            codec ? codec : (std::filesystem::path(model).parent_path() / "dac-44khz.gguf").string();
        if (dia_tts_set_codec_path(ctx.get(), codec_path.c_str()) != 0)
            return 1;
        set_env("CRISPASR_DIA_FORCE_TOKENS", nullptr);
        set_env("CRISPASR_DIA_DUMP_DIR", nullptr);
        set_env("CRISPASR_DIA_DUMP_STEPLOGITS", nullptr);
        set_env("CRISPASR_DIA_DUMP_INPUT_TOKENS", nullptr);
        set_env("CRISPASR_DIA_DECODE_CODES", codes.string().c_str());
        printf("[REPLAY] decoder_codes_rev: exact original codes drive the isolated DAC; delay reversal is tested "
               "separately.\n");
        std::unique_ptr<float, decltype(&dia_tts_pcm_free)> pcm(dia_tts_synthesize(ctx.get(), text.c_str(), &samples),
                                                                dia_tts_pcm_free);
        if (!pcm || samples <= 0)
            return 1;
        pass = compare(ref, "decoded_audio", std::vector<float>(pcm.get(), pcm.get() + samples)) && pass;
        printf("Dia diff scope: original decoder-input replay and DAC-code replay; free-running speech is separate.\n");
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        fprintf(stderr, "Dia diff: %s\n", error.what());
        return 1;
    }
}

} // namespace dia_diff
