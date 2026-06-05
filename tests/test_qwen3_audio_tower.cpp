// test_qwen3_audio_tower.cpp — numeric regression baseline for the qwen3-asr
// audio tower (mel preprocessor + conv stem + 18-layer encoder + ln_post +
// proj1/GELU/proj2 → 1024-d frame embeddings).
//
// This pre-refactor baseline locks in the current numerical behavior so the
// upcoming extraction into the shared CrispAudio library can be verified to
// preserve bit-by-bit semantics (within 1e-5 tolerance).
//
// Usage:
//   # Generate fixture (one-time, pre-refactor):
//   ./build/tests/test_qwen3_audio_tower MODEL.gguf jfk.raw \
//       > tests/fixtures/qwen3_audio_tower_jfk.txt
//
//   # Verify after refactor:
//   ./build/tests/test_qwen3_audio_tower --check MODEL.gguf jfk.raw \
//       tests/fixtures/qwen3_audio_tower_jfk.txt
//
// jfk.raw must be 16 kHz mono float32 little-endian PCM. Generate with:
//   ffmpeg -i samples/jfk.wav -ar 16000 -ac 1 -f f32le samples/jfk.raw

#include "qwen3_asr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Stats {
    int n_frames = 0;
    int dim = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    // First and last 32 values of the flattened (n_frames, dim) tensor.
    // Tight window onto the actual numerical output — catches changes in
    // either the early frames (mel/conv) or late frames (encoder/projector).
    std::vector<float> first32;
    std::vector<float> last32;
};

Stats compute_stats(const float* data, int n_frames, int dim) {
    const size_t total = (size_t)n_frames * (size_t)dim;
    Stats s;
    s.n_frames = n_frames;
    s.dim = dim;
    s.first32.assign(data, data + std::min<size_t>(32, total));
    if (total > 32) {
        s.last32.assign(data + total - 32, data + total);
    } else {
        s.last32 = s.first32;
    }
    for (size_t i = 0; i < total; i++) {
        const double x = (double)data[i];
        s.sum += x;
        s.sum_sq += x * x;
        const double a = std::fabs(x);
        if (a > s.max_abs) s.max_abs = a;
    }
    return s;
}

void print_stats(const Stats& s) {
    printf("# qwen3-asr audio tower regression baseline\n");
    printf("n_frames %d\n", s.n_frames);
    printf("dim %d\n", s.dim);
    // Fixed precision so diff-based comparison is stable across runs.
    printf("sum %.6e\n", s.sum);
    printf("sum_sq %.6e\n", s.sum_sq);
    printf("max_abs %.6e\n", s.max_abs);
    printf("first32");
    for (float v : s.first32) printf(" %.6e", v);
    printf("\n");
    printf("last32");
    for (float v : s.last32) printf(" %.6e", v);
    printf("\n");
}

bool load_pcm(const char* path, std::vector<float>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", path);
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % sizeof(float) != 0) {
        fprintf(stderr, "error: %s size=%ld is not a valid f32le stream\n", path, sz);
        std::fclose(f);
        return false;
    }
    out.resize(sz / sizeof(float));
    if (std::fread(out.data(), sizeof(float), out.size(), f) != out.size()) {
        fprintf(stderr, "error: short read on %s\n", path);
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

// Tolerance for re-validation. Tight enough to catch any code change that
// alters the math, loose enough to absorb floating-point reassociation that
// can happen between ggml versions on the same backend.
constexpr double kAbsTolScalar = 1e-4;
constexpr double kAbsTolRel    = 1e-4;
constexpr double kAbsTolMaxAbs = 1e-4;

bool approx_equal(double a, double b, double abs_tol, double rel_tol) {
    const double d = std::fabs(a - b);
    if (d <= abs_tol) return true;
    const double m = std::max(std::fabs(a), std::fabs(b));
    return d <= rel_tol * m;
}

bool check_against_fixture(const Stats& actual, const char* fixture_path) {
    FILE* f = std::fopen(fixture_path, "r");
    if (!f) {
        fprintf(stderr, "error: cannot open fixture %s\n", fixture_path);
        return false;
    }
    int exp_n_frames = -1, exp_dim = -1;
    double exp_sum = 0, exp_sum_sq = 0, exp_max_abs = 0;
    std::vector<float> exp_first32, exp_last32;
    char line[8192];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char key[32];
        if (std::sscanf(line, "%31s", key) != 1) continue;
        if (std::strcmp(key, "n_frames") == 0) {
            std::sscanf(line, "%*s %d", &exp_n_frames);
        } else if (std::strcmp(key, "dim") == 0) {
            std::sscanf(line, "%*s %d", &exp_dim);
        } else if (std::strcmp(key, "sum") == 0) {
            std::sscanf(line, "%*s %lf", &exp_sum);
        } else if (std::strcmp(key, "sum_sq") == 0) {
            std::sscanf(line, "%*s %lf", &exp_sum_sq);
        } else if (std::strcmp(key, "max_abs") == 0) {
            std::sscanf(line, "%*s %lf", &exp_max_abs);
        } else if (std::strcmp(key, "first32") == 0 || std::strcmp(key, "last32") == 0) {
            std::vector<float>& v = (key[0] == 'f') ? exp_first32 : exp_last32;
            const char* p = line + std::strlen(key);
            while (*p) {
                char* end = nullptr;
                double x = std::strtod(p, &end);
                if (end == p) break;
                v.push_back((float)x);
                p = end;
            }
        }
    }
    std::fclose(f);

    bool ok = true;
    auto fail = [&](const char* msg) {
        fprintf(stderr, "FAIL: %s\n", msg);
        ok = false;
    };

    if (exp_n_frames != actual.n_frames) {
        fprintf(stderr, "FAIL: n_frames %d vs expected %d\n", actual.n_frames, exp_n_frames);
        ok = false;
    }
    if (exp_dim != actual.dim) {
        fprintf(stderr, "FAIL: dim %d vs expected %d\n", actual.dim, exp_dim);
        ok = false;
    }
    if (!approx_equal(exp_sum, actual.sum, kAbsTolRel * std::fabs(exp_sum) + kAbsTolScalar, kAbsTolRel)) {
        fprintf(stderr, "FAIL: sum %.6e vs expected %.6e (delta %.3e)\n",
                actual.sum, exp_sum, std::fabs(actual.sum - exp_sum));
        ok = false;
    }
    if (!approx_equal(exp_sum_sq, actual.sum_sq, kAbsTolRel * std::fabs(exp_sum_sq) + kAbsTolScalar, kAbsTolRel)) {
        fprintf(stderr, "FAIL: sum_sq %.6e vs expected %.6e (delta %.3e)\n",
                actual.sum_sq, exp_sum_sq, std::fabs(actual.sum_sq - exp_sum_sq));
        ok = false;
    }
    if (!approx_equal(exp_max_abs, actual.max_abs, kAbsTolMaxAbs, kAbsTolRel)) {
        fprintf(stderr, "FAIL: max_abs %.6e vs expected %.6e\n", actual.max_abs, exp_max_abs);
        ok = false;
    }
    auto cmp_window = [&](const std::vector<float>& a, const std::vector<float>& e, const char* tag) {
        if (a.size() != e.size()) {
            fprintf(stderr, "FAIL: %s window size %zu vs %zu\n", tag, a.size(), e.size());
            ok = false;
            return;
        }
        double max_d = 0;
        int max_i = -1;
        for (size_t i = 0; i < a.size(); i++) {
            double d = std::fabs((double)a[i] - (double)e[i]);
            if (d > max_d) {
                max_d = d;
                max_i = (int)i;
            }
        }
        if (max_d > kAbsTolScalar) {
            fprintf(stderr, "FAIL: %s window max delta %.3e at index %d\n", tag, max_d, max_i);
            ok = false;
        }
    };
    cmp_window(actual.first32, exp_first32, "first32");
    cmp_window(actual.last32, exp_last32, "last32");

    return ok;
}

int usage(const char* argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s          MODEL.gguf PCM.raw          # write fixture to stdout\n"
            "  %s --check  MODEL.gguf PCM.raw FIXTURE  # verify against fixture\n"
            "PCM.raw = 16 kHz mono float32 little-endian. ffmpeg recipe in source comments.\n",
            argv0, argv0);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    bool check_mode = false;
    int arg = 1;
    if (argc >= 2 && std::strcmp(argv[1], "--check") == 0) {
        check_mode = true;
        arg = 2;
    }
    if ((!check_mode && argc - arg != 2) || (check_mode && argc - arg != 3)) {
        return usage(argv[0]);
    }

    const char* model_path = argv[arg + 0];
    const char* pcm_path   = argv[arg + 1];
    const char* fixture    = check_mode ? argv[arg + 2] : nullptr;

    std::vector<float> samples;
    if (!load_pcm(pcm_path, samples)) return 1;
    fprintf(stderr, "loaded %zu samples (%.2f s @ 16kHz)\n",
            samples.size(), (double)samples.size() / 16000.0);

    qwen3_asr_context_params params = qwen3_asr_context_default_params();
    params.use_gpu = false;     // CPU only — deterministic across machines
    params.verbosity = 0;
    qwen3_asr_context* ctx = qwen3_asr_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "error: failed to load model %s\n", model_path);
        return 1;
    }

    int n_mels = 0, T_mel = 0;
    float* mel = qwen3_asr_compute_mel(ctx, samples.data(), (int)samples.size(),
                                       &n_mels, &T_mel);
    if (!mel) {
        fprintf(stderr, "error: compute_mel failed\n");
        qwen3_asr_free(ctx);
        return 1;
    }
    fprintf(stderr, "mel: %d x %d\n", n_mels, T_mel);

    int n_total = 0, proj_dim = 0;
    float* enc = qwen3_asr_run_encoder(ctx, mel, n_mels, T_mel, &n_total, &proj_dim);
    std::free(mel);
    if (!enc) {
        fprintf(stderr, "error: run_encoder failed\n");
        qwen3_asr_free(ctx);
        return 1;
    }
    fprintf(stderr, "encoder: %d frames x %d dim\n", n_total, proj_dim);

    Stats stats = compute_stats(enc, n_total, proj_dim);
    std::free(enc);
    qwen3_asr_free(ctx);

    if (check_mode) {
        if (!check_against_fixture(stats, fixture)) {
            fprintf(stderr, "regression FAILED\n");
            return 1;
        }
        fprintf(stderr, "regression OK\n");
        return 0;
    }

    print_stats(stats);
    return 0;
}
