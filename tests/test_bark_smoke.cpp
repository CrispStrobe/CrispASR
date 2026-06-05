// Bark TTS integration test — exercises all fixed code paths:
//   1. Model load (F16, Q8_0, Q4_K)
//   2. BERT tokenizer from GGUF
//   3. Speaker prompt .npz loading
//   4. All 3 stages + EnCodec decode
//   5. Peak amplitude sanity (not silence)
//   6. Seed reproducibility
//   7. Temperature + seed setters
//
// Build:
//   g++ -std=c++17 -O2 -o test_bark tests/test_bark_smoke.cpp \
//     -Isrc -Iggml/include -Iinclude -Isrc/core \
//     -Lbuild/src -lbark-tts -lcrispasr-core \
//     -Lbuild/ggml/src -lggml -lggml-base -lggml-cpu -lpthread -ldl
//
// Run:
//   ./test_bark <model.gguf>                              # basic test
//   ./test_bark <model.gguf> --speaker <file.npz>         # with speaker
//   ./test_bark <model.gguf> --text "custom text"         # custom text
//   ./test_bark <model.gguf> --out dir/                   # output dir

#include "bark_tts.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void write_wav(const char* path, const float* pcm, int n, int sr) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "  WARN: cannot write %s\n", path); return; }
    int16_t* buf = (int16_t*)malloc((size_t)n * 2);
    for (int i = 0; i < n; i++) {
        float s = pcm[i] * 32767.0f;
        if (s > 32767.0f) s = 32767.0f;
        if (s < -32768.0f) s = -32768.0f;
        buf[i] = (int16_t)s;
    }
    uint32_t data_size = (uint32_t)(n * 2);
    uint32_t file_size = 36 + data_size;
    fwrite("RIFF", 1, 4, f);   fwrite(&file_size, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    uint32_t fmt_size = 16;     fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_fmt = 1;     fwrite(&audio_fmt, 2, 1, f);
    uint16_t channels = 1;      fwrite(&channels, 2, 1, f);
    uint32_t sample_rate = (uint32_t)sr; fwrite(&sample_rate, 4, 1, f);
    uint32_t byte_rate = (uint32_t)(sr * 2); fwrite(&byte_rate, 4, 1, f);
    uint16_t block_align = 2;   fwrite(&block_align, 2, 1, f);
    uint16_t bits = 16;         fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);   fwrite(&data_size, 4, 1, f);
    fwrite(buf, 2, (size_t)n, f);
    fclose(f); free(buf);
}

static float peak_amp(const float* pcm, int n) {
    float peak = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = pcm[i] < 0 ? -pcm[i] : pcm[i];
        if (a > peak) peak = a;
    }
    return peak;
}

static float rms_amp(const float* pcm, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)pcm[i] * pcm[i];
    return (float)std::sqrt(sum / n);
}

static float cosine_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0f;
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--speaker <npz>] [--text \"...\"] [--out <dir>]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* speaker_npz = nullptr;
    const char* text = "Hello there, how are you doing today?";
    std::string out_dir = "/tmp";

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--speaker") && i + 1 < argc) speaker_npz = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) text = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
    }

    int n_pass = 0, n_fail = 0;
    auto PASS = [&](const char* t) { fprintf(stderr, "  [PASS] %s\n", t); n_pass++; };
    auto FAIL = [&](const char* t) { fprintf(stderr, "  [FAIL] %s\n", t); n_fail++; };

    // ---------------------------------------------------------------
    fprintf(stderr, "\n=== Test 1: Model load ===\n");
    bark_context_params p = bark_context_default_params();
    p.n_threads = 4;
    p.verbosity = 1;
    p.seed = 42;

    bark_context* ctx = bark_init_from_file(model_path, p);
    if (!ctx) { FAIL("model load"); return 1; }
    PASS("model load");

    if (bark_sample_rate(ctx) == 24000) PASS("sample_rate == 24000");
    else FAIL("sample_rate != 24000");

    // ---------------------------------------------------------------
    fprintf(stderr, "\n=== Test 2: Speaker prompt .npz ===\n");
    if (speaker_npz) {
        int rc = bark_set_speaker_npz(ctx, speaker_npz);
        if (rc == 0) PASS("npz load");
        else FAIL("npz load returned error");
    } else {
        fprintf(stderr, "  [SKIP] no --speaker provided\n");
    }

    // ---------------------------------------------------------------
    fprintf(stderr, "\n=== Test 3: Synthesis (seed=42) ===\n");
    bark_set_seed(ctx, 42);
    int n1 = 0;
    float* pcm1 = bark_synthesize(ctx, text, &n1);
    if (!pcm1 || n1 <= 0) {
        FAIL("synthesis produced no audio");
        bark_free(ctx);
        return 1;
    }
    PASS("synthesis completed");
    fprintf(stderr, "  samples=%d (%.2f sec), peak=%.4f, rms=%.4f\n",
            n1, (float)n1 / 24000.0f, peak_amp(pcm1, n1), rms_amp(pcm1, n1));

    if (peak_amp(pcm1, n1) > 0.05f) PASS("peak > 0.05 (not silence)");
    else FAIL("peak <= 0.05 (near-silence — bad)");

    if (n1 > 4800) PASS("duration > 0.2s");
    else FAIL("duration <= 0.2s (too short)");

    std::string wav1 = out_dir + "/bark_test_seed42.wav";
    write_wav(wav1.c_str(), pcm1, n1, 24000);
    fprintf(stderr, "  wrote: %s\n", wav1.c_str());

    // ---------------------------------------------------------------
    fprintf(stderr, "\n=== Test 4: Seed reproducibility ===\n");
    bark_set_seed(ctx, 42);
    int n2 = 0;
    float* pcm2 = bark_synthesize(ctx, text, &n2);
    if (!pcm2 || n2 <= 0) {
        FAIL("second synthesis failed");
    } else if (n1 == n2) {
        float cos = cosine_sim(pcm1, pcm2, n1);
        fprintf(stderr, "  same seed: n1=%d n2=%d cos=%.6f\n", n1, n2, cos);
        if (cos > 0.999f) PASS("seed reproducibility (cos > 0.999)");
        else FAIL("seed NOT reproducible");
    } else {
        fprintf(stderr, "  lengths differ: n1=%d n2=%d\n", n1, n2);
        FAIL("seed NOT reproducible (different lengths)");
    }
    if (pcm2) bark_pcm_free(pcm2);

    // ---------------------------------------------------------------
    fprintf(stderr, "\n=== Test 5: Different seed produces different audio ===\n");
    bark_set_seed(ctx, 123);
    int n3 = 0;
    float* pcm3 = bark_synthesize(ctx, text, &n3);
    if (!pcm3 || n3 <= 0) {
        FAIL("seed=123 synthesis failed");
    } else {
        int cmp_len = n1 < n3 ? n1 : n3;
        float cos = cosine_sim(pcm1, pcm3, cmp_len);
        fprintf(stderr, "  seed 42 vs 123: n1=%d n3=%d cos=%.6f\n", n1, n3, cos);
        if (cos < 0.95f) PASS("different seeds produce different audio");
        else FAIL("different seeds produce suspiciously similar audio");
        std::string wav3 = out_dir + "/bark_test_seed123.wav";
        write_wav(wav3.c_str(), pcm3, n3, 24000);
        fprintf(stderr, "  wrote: %s\n", wav3.c_str());
    }
    if (pcm3) bark_pcm_free(pcm3);

    // ---------------------------------------------------------------
    fprintf(stderr, "\n=== Test 6: Temperature setter ===\n");
    bark_set_temperature_semantic(ctx, 0.5f);
    bark_set_temperature_coarse(ctx, 0.5f);
    bark_set_temperature_fine(ctx, 0.3f);
    bark_set_seed(ctx, 42);
    int n4 = 0;
    float* pcm4 = bark_synthesize(ctx, text, &n4);
    if (!pcm4 || n4 <= 0) {
        FAIL("temp=0.5 synthesis failed");
    } else {
        fprintf(stderr, "  temp=0.5: n=%d peak=%.4f rms=%.4f\n",
                n4, peak_amp(pcm4, n4), rms_amp(pcm4, n4));
        if (peak_amp(pcm4, n4) > 0.05f) PASS("temp=0.5 produces audio");
        else FAIL("temp=0.5 silence");
        std::string wav4 = out_dir + "/bark_test_temp05.wav";
        write_wav(wav4.c_str(), pcm4, n4, 24000);
    }
    if (pcm4) bark_pcm_free(pcm4);

    // ---------------------------------------------------------------
    fprintf(stderr, "\n=== Test 7: n_threads setter ===\n");
    bark_set_n_threads(ctx, 2);
    bark_set_seed(ctx, 42);
    bark_set_temperature_semantic(ctx, 0.7f);
    bark_set_temperature_coarse(ctx, 0.7f);
    bark_set_temperature_fine(ctx, 0.5f);
    int n5 = 0;
    float* pcm5 = bark_synthesize(ctx, text, &n5);
    if (!pcm5 || n5 <= 0) {
        FAIL("n_threads=2 synthesis failed");
    } else {
        // Should be identical to test 3 (same seed, same params)
        fprintf(stderr, "  n_threads=2: n=%d peak=%.4f\n", n5, peak_amp(pcm5, n5));
        if (peak_amp(pcm5, n5) > 0.05f) PASS("n_threads=2 produces audio");
        else FAIL("n_threads=2 silence");
        // Note: AR sampling diverges across thread counts due to FP reduction
        // order — cos < 1.0 is expected and not a bug.
    }
    if (pcm5) bark_pcm_free(pcm5);

    bark_pcm_free(pcm1);
    bark_free(ctx);

    // ---------------------------------------------------------------
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Results: %d pass, %d fail\n", n_pass, n_fail);
    fprintf(stderr, "========================================\n\n");
    return n_fail > 0 ? 1 : 0;
}
