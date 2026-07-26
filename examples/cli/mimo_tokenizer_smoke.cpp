// tools/mimo_tokenizer_smoke.cpp — smoke test for the MiMo-Audio-Tokenizer
// encoder forward path. Loads a 16 kHz mono PCM WAV and runs each declared
// stage through `mimo_tokenizer_extract_stage`, printing per-stage shape +
// finiteness + min/max/mean stats. Used during PLAN #51 step 2 development
// before the upstream safetensors are available for cosine ground truth.
//
// Usage:
//   mimo-tokenizer-smoke <model.gguf> <audio.wav>

#include "mimo_tokenizer.h"
#include "common-crispasr.h"
#include "core/win_compat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/crispasr_env.h"

namespace {

struct Stats {
    bool any_nonfinite = false;
    float min_v = 1e30f;
    float max_v = -1e30f;
    double mean = 0.0;
};

static Stats summarize(const float* d, int n) {
    Stats s;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float v = d[i];
        if (!std::isfinite(v))
            s.any_nonfinite = true;
        if (v < s.min_v)
            s.min_v = v;
        if (v > s.max_v)
            s.max_v = v;
        sum += (double)v;
    }
    s.mean = n > 0 ? sum / (double)n : 0.0;
    return s;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <audio.wav>\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* audio_path = argv[2];

    std::vector<float> samples;
    std::vector<std::vector<float>> stereo;
    if (!read_audio_data(audio_path, samples, stereo, /*stereo=*/false)) {
        fprintf(stderr, "smoke: failed to read audio '%s'\n", audio_path);
        return 2;
    }
    printf("smoke: loaded %zu samples (%.2fs @ 16 kHz)\n", samples.size(), samples.size() / 16000.0);

    auto cp = mimo_tokenizer_context_default_params();
    cp.n_threads = 4;
    cp.verbosity = 1;
    // CPU-pin until the Metal conv-stem watchdog hang is verified absent on
    // forward conv (cf. qwen3-tts kernel_conv_transpose_1d). Set
    // MIMO_SMOKE_GPU=1 to exercise the GPU path.
    cp.use_gpu = crispasr_env::get("CRISPASR_MIMO_SMOKE_GPU") != nullptr;
    mimo_tokenizer_context* ctx = mimo_tokenizer_init_from_file(model_path, cp);
    if (!ctx) {
        fprintf(stderr, "smoke: failed to load tokenizer '%s'\n", model_path);
        return 3;
    }
    static const char* stages[] = {
        "tok_mel", "tok_conv1_out", "tok_conv2_out", "tok_xfmr_out", "tok_pool_out", "tok_codes",
    };
    int n_pass = 0, n_fail = 0;
    for (const char* stage : stages) {
        int n = 0;
        float* data = mimo_tokenizer_extract_stage(ctx, samples.data(), (int)samples.size(), stage, &n);
        if (!data) {
            printf("[FAIL] %-22s extract returned null\n", stage);
            n_fail++;
            continue;
        }
        Stats s = summarize(data, n);
        const char* tag = s.any_nonfinite ? "[NaN ]" : "[ ok ]";
        if (s.any_nonfinite)
            n_fail++;
        else
            n_pass++;
        printf("%s %-22s n=%-8d min=%-12.4f max=%-12.4f mean=%-12.4f\n", tag, stage, n, s.min_v, s.max_v, s.mean);
        if (crispasr_env::get("CRISPASR_MIMO_SMOKE_DUMP")) {
            char path[256];
            std::snprintf(path, sizeof(path), "/tmp/mimo_cpp_%s.bin", stage);
            FILE* f = std::fopen(path, "wb");
            if (f) {
                std::fwrite(data, sizeof(float), (size_t)n, f);
                std::fclose(f);
                fprintf(stderr, "  → dumped %s (%d floats)\n", path, n);
            }
        }
        std::free(data);
    }
    if (crispasr_env::truthy("CRISPASR_MIMO_TOK_VERIFY_RVQ")) {
        constexpr const char* kCpuRvqEnv = "CRISPASR_MIMO_TOK_CPU_RVQ";
        const char* old_value_ptr = std::getenv(kCpuRvqEnv);
        const bool had_old_value = old_value_ptr != nullptr;
        const std::string old_value = old_value_ptr ? old_value_ptr : "";

        unsetenv(kCpuRvqEnv);
        int gpu_frames = 0;
        int32_t* gpu_codes = mimo_tokenizer_encode_pcm16k(ctx, samples.data(), (int)samples.size(), &gpu_frames);

        setenv(kCpuRvqEnv, "1", /*overwrite=*/1);
        int cpu_frames = 0;
        int32_t* cpu_codes = mimo_tokenizer_encode_pcm16k(ctx, samples.data(), (int)samples.size(), &cpu_frames);

        if (had_old_value)
            setenv(kCpuRvqEnv, old_value.c_str(), /*overwrite=*/1);
        else
            unsetenv(kCpuRvqEnv);

        size_t mismatches = 0;
        if (!gpu_codes || !cpu_codes || gpu_frames != cpu_frames) {
            mismatches = 1;
        } else {
            const size_t n_codes = (size_t)cpu_frames * 8;
            for (size_t i = 0; i < n_codes; i++)
                mismatches += gpu_codes[i] != cpu_codes[i];
        }
        if (mismatches == 0) {
            printf("[ ok ] rvq_cpu_gpu_compare    codes=%zu mismatches=0\n", (size_t)cpu_frames * 8);
            n_pass++;
        } else {
            printf("[FAIL] rvq_cpu_gpu_compare    gpu_frames=%d cpu_frames=%d mismatches=%zu\n", gpu_frames, cpu_frames,
                   mismatches);
            n_fail++;
        }
        std::free(gpu_codes);
        std::free(cpu_codes);
    }
    mimo_tokenizer_free(ctx);
    printf("smoke: %d ok, %d fail\n", n_pass, n_fail);
    return n_fail ? 4 : 0;
}
