// test-f5-tts.cpp — smoke test + diff harness for F5-TTS.
//
// Usage:
//   test-f5-tts <model.gguf> <text> <output.wav> [--dump <dir>]
//                [--ref-mel <ref_mel.bin>] [--ref-mel-T <T>]
//
// With --dump, dumps all intermediate tensors as .bin files.
// With --ref-mel, loads external reference mel (T, 100) as conditioning.
// The Python diff script then compares these against the reference GGUF.

#include "f5_tts.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── Cosine similarity helper ─────────────────────────────────────

static float cosine_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

// Per-row cosine: treat data as (rows, cols), compute cos per row, return min
static float cosine_per_row(const float* a, const float* b, int rows, int cols) {
    float min_cos = 1.0f;
    for (int r = 0; r < rows; r++) {
        float c = cosine_sim(a + r * cols, b + r * cols, cols);
        if (c < min_cos) min_cos = c;
    }
    return min_cos;
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// ── Read reference tensor from GGUF ──────────────────────────────

static std::vector<float> read_ref_tensor(const core_gguf::WeightLoad& wl, const char* name,
                                          std::vector<int64_t>& shape) {
    auto it = wl.tensors.find(name);
    if (it == wl.tensors.end()) return {};
    ggml_tensor* t = it->second;
    int n = (int)ggml_nelements(t);
    // Store shape (GGUF ne[] is innermost-first; we want row-major order)
    shape.clear();
    int ndims = GGML_MAX_DIMS;
    while (ndims > 1 && t->ne[ndims - 1] == 1) ndims--;
    for (int i = ndims - 1; i >= 0; i--) {
        shape.push_back(t->ne[i]);
    }
    std::vector<float> data(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> idata(n);
        ggml_backend_tensor_get(t, idata.data(), 0, n * sizeof(int32_t));
        for (int i = 0; i < n; i++) data[i] = (float)idata[i];
    } else {
        // F16
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        const ggml_fp16_t* src = (const ggml_fp16_t*)raw.data();
        for (int i = 0; i < n; i++) data[i] = ggml_fp16_to_fp32(src[i]);
    }
    return data;
}

// ── Compare and report ───────────────────────────────────────────

static void compare_stage(const char* name, const float* cpp, int cpp_n,
                          const float* ref, int ref_n, int last_dim) {
    int n = std::min(cpp_n, ref_n);
    if (n == 0) {
        fprintf(stderr, "  %-25s  SKIP (empty)\n", name);
        return;
    }
    float max_d = max_abs_diff(cpp, ref, n);
    int rows = n / last_dim;
    float cos_min = cosine_per_row(cpp, ref, rows, last_dim);
    const char* status = (cos_min >= 0.999f) ? "PASS" : (cos_min >= 0.99f ? "SOFT" : "FAIL");
    fprintf(stderr, "  %-25s  cos_min=%.6f  max_abs=%.2e  n=%d  %s\n",
            name, cos_min, max_d, n, status);
}

// ── WAV writer ───────────────────────────────────────────────────

static void write_wav(const char* path, const float* pcm, int n, int sr) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    int data_size = n * 2;
    int file_size = 36 + data_size;
    fwrite("RIFF", 4, 1, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVEfmt ", 8, 1, f);
    int chunk_size = 16; fwrite(&chunk_size, 4, 1, f);
    short audio_format = 1; fwrite(&audio_format, 2, 1, f);
    short channels = 1; fwrite(&channels, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    int byte_rate = sr * 2; fwrite(&byte_rate, 4, 1, f);
    short block_align = 2; fwrite(&block_align, 2, 1, f);
    short bits = 16; fwrite(&bits, 2, 1, f);
    fwrite("data", 4, 1, f);
    fwrite(&data_size, 4, 1, f);
    for (int i = 0; i < n; i++) {
        float s = pcm[i];
        if (s > 1.0f) s = 1.0f; if (s < -1.0f) s = -1.0f;
        short v = (short)(s * 32767.0f);
        fwrite(&v, 2, 1, f);
    }
    fclose(f);
    fprintf(stderr, "Wrote %s (%d samples, %.2f s)\n", path, n, (float)n / sr);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <text> <output.wav> [--ref-gguf <ref>] [--dump <dir>]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* text = argv[2];
    const char* wav_path = argv[3];
    const char* ref_gguf_path = nullptr;
    const char* dump_dir = nullptr;
    const char* ref_text = "";
    int seed = 42;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--ref-gguf") == 0 && i + 1 < argc) ref_gguf_path = argv[++i];
        else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) dump_dir = argv[++i];
        else if (strcmp(argv[i], "--ref-text") == 0 && i + 1 < argc) ref_text = argv[++i];
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
    }

    // ── Load model ──
    f5_tts_params params = f5_tts_default_params();
    params.verbosity = 2;
    params.seed = seed;

    f5_tts_context* ctx = f5_tts_init_from_file(model_path, params);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    if (dump_dir) f5_tts_set_dump_dir(ctx, dump_dir);

    fprintf(stderr, "Model loaded. vocab=%d sr=%d\n", f5_tts_vocab_size(ctx), f5_tts_sample_rate(ctx));

    // ── Load reference GGUF if provided ──
    core_gguf::WeightLoad ref_wl = {};
    bool have_ref = false;
    if (ref_gguf_path) {
        ggml_backend_t ref_be = ggml_backend_cpu_init();
        if (ref_be && core_gguf::load_weights(ref_gguf_path, ref_be, "ref", ref_wl)) {
            have_ref = true;
            fprintf(stderr, "Loaded reference GGUF: %s (%zu tensors)\n",
                    ref_gguf_path, ref_wl.tensors.size());
        } else {
            fprintf(stderr, "WARNING: failed to load ref GGUF: %s\n", ref_gguf_path);
        }
    }

    // ── Set reference mel from ref GGUF ──
    if (have_ref) {
        std::vector<int64_t> shape;
        auto ref_mel = read_ref_tensor(ref_wl, "ref_mel", shape);
        if (!ref_mel.empty() && shape.size() == 2) {
            int T_ref = (int)shape[0];
            int mel_dim = (int)shape[1];
            fprintf(stderr, "Setting ref_mel from GGUF: T=%d mel_dim=%d\n", T_ref, mel_dim);
            // f5_tts_set_reference expects 24kHz PCM, but we're injecting mel directly.
            // We need an internal API. For now, access ctx internals.
            // This is a test-only hack; the real pipeline computes mel from audio.
            extern void f5_tts_set_ref_mel(f5_tts_context* ctx, const float* mel,
                                            int T, int mel_dim, const char* ref_text);
            f5_tts_set_ref_mel(ctx, ref_mel.data(), T_ref, mel_dim, ref_text);
        }

        // Inject reference initial noise (ode_step_0)
        std::vector<int64_t> noise_shape;
        auto ref_noise = read_ref_tensor(ref_wl, "ode_step_0", noise_shape);
        if (!ref_noise.empty()) {
            fprintf(stderr, "Injecting ref initial noise: %zu elements\n", ref_noise.size());
            extern void f5_tts_set_init_noise(f5_tts_context* ctx, const float* noise, int n);
            f5_tts_set_init_noise(ctx, ref_noise.data(), (int)ref_noise.size());
        }
    }

    // ── Synthesize ──
    float* pcm = nullptr;
    int sr = 0;
    int n = f5_tts_synthesize(ctx, text, &pcm, &sr);

    if (n > 0 && pcm) {
        fprintf(stderr, "Generated %d samples at %d Hz (%.2f s)\n", n, sr, (float)n / sr);
        write_wav(wav_path, pcm, n, sr);
        free(pcm);
    } else {
        fprintf(stderr, "Synthesis returned 0 samples\n");
    }

    // ── Diff comparison against reference ──
    if (have_ref && dump_dir) {
        fprintf(stderr, "\n=== Stage-by-stage comparison ===\n");

        const char* stages[] = {
            "text_embed", "time_embed", "input_embed",
            "dit_layer_0", "dit_layer_5", "dit_layer_10", "dit_layer_15", "dit_layer_21",
            "dit_output",
            "ode_step_0", "ode_step_8", "ode_step_16", "ode_step_24", "ode_step_31",
            nullptr
        };
        int last_dims[] = {
            512, 1024, 1024,
            1024, 1024, 1024, 1024, 1024,
            100,
            100, 100, 100, 100, 100,
        };

        for (int s = 0; stages[s]; s++) {
            std::vector<int64_t> ref_shape;
            auto ref_data = read_ref_tensor(ref_wl, stages[s], ref_shape);
            if (ref_data.empty()) {
                fprintf(stderr, "  %-25s  SKIP (not in ref)\n", stages[s]);
                continue;
            }

            // Load C++ dump
            std::string bin_path = std::string(dump_dir) + "/" + stages[s] + ".bin";
            FILE* f = fopen(bin_path.c_str(), "rb");
            if (!f) {
                fprintf(stderr, "  %-25s  SKIP (no dump)\n", stages[s]);
                continue;
            }
            fseek(f, 0, SEEK_END);
            long fsize = ftell(f);
            fseek(f, 0, SEEK_SET);
            int cpp_n = (int)(fsize / sizeof(float));
            std::vector<float> cpp_data(cpp_n);
            size_t nr = fread(cpp_data.data(), sizeof(float), cpp_n, f);
            (void)nr;
            fclose(f);

            compare_stage(stages[s], cpp_data.data(), cpp_n,
                          ref_data.data(), (int)ref_data.size(), last_dims[s]);
        }
    }

    // Cleanup
    if (have_ref) core_gguf::free_weights(ref_wl);
    f5_tts_free(ctx);
    return 0;
}
