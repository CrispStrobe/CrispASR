// Minimal test binary for MeloTTS — links only melotts lib, no CLI deps.
#include "melotts.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Minimal WAV header writer
static void write_wav(const char* path, const float* pcm, int n, int sr) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot write %s\n", path);
        return;
    }
    int16_t* buf = (int16_t*)malloc(n * sizeof(int16_t));
    for (int i = 0; i < n; i++) {
        float s = pcm[i] * 32767.0f;
        if (s > 32767.0f)
            s = 32767.0f;
        if (s < -32768.0f)
            s = -32768.0f;
        buf[i] = (int16_t)s;
    }
    int data_size = n * 2;
    int file_size = 36 + data_size;
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    int fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    short fmt = 1;
    fwrite(&fmt, 2, 1, f); // PCM
    short ch = 1;
    fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    int bps = sr * 2;
    fwrite(&bps, 4, 1, f);
    short ba = 2;
    fwrite(&ba, 2, 1, f);
    short bits = 16;
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    fwrite(buf, 2, n, f);
    fclose(f);
    free(buf);
    fprintf(stderr, "Wrote %s (%d samples, %.2f s)\n", path, n, (float)n / sr);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <text> [output.wav] [--dump-dir DIR]\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* text = argv[2];
    const char* wav_path = argc > 3 ? argv[3] : "/tmp/melotts_out.wav";

    struct melotts_params params = melotts_default_params();
    params.n_threads = 4;
    params.verbosity = 2;
    params.seed = 42;

    // Check for --dump-dir
    const char* dump_dir = nullptr;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--dump-dir") == 0 && i + 1 < argc) {
            dump_dir = argv[i + 1];
            i++;
        }
    }

    struct melotts_context* ctx = melotts_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    if (dump_dir) {
        melotts_set_dump_dir(ctx, dump_dir);
    }

    float* pcm = nullptr;
    int sr = 0;
    int n = melotts_synthesize(ctx, text, &pcm, &sr);
    if (n <= 0) {
        fprintf(stderr, "Synthesis failed\n");
        melotts_free(ctx);
        return 1;
    }

    write_wav(wav_path, pcm, n, sr);
    melotts_pcm_free(pcm);
    melotts_free(ctx);
    return 0;
}
