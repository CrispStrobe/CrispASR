// tests/test-piper-tts.cpp — smoke test for piper TTS runtime.
// Loads a GGUF, synthesizes from pre-phonemized IPA, writes a WAV file.
//
// Usage:
//   build/bin/test-piper-tts <model.gguf> <ipa_string> <output.wav>
//
// Example:
//   build/bin/test-piper-tts /mnt/storage/piper/piper-en_US-lessac-medium-f16.gguf \
//       "həlˈoʊ wˈɜːld" /mnt/storage/piper/cpp_hello_world.wav

#include "piper_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>

static bool write_wav(const char * path, const float * data, int n_samples, int sample_rate) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;

    int16_t * pcm16 = (int16_t *)malloc(n_samples * sizeof(int16_t));
    for (int i = 0; i < n_samples; i++) {
        float v = data[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm16[i] = (int16_t)(v * 32767.0f);
    }

    // WAV header
    uint32_t data_size = n_samples * 2;
    uint32_t file_size = 36 + data_size;
    uint16_t channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
    uint16_t block_align = channels * bits_per_sample / 8;

    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1;
    fwrite(&audio_format, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    fwrite(pcm16, 2, n_samples, f);

    fclose(f);
    free(pcm16);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <ipa_string> <output.wav> [noise_scale] [noise_w] [--dump <dir>]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * ipa = argv[2];
    const char * wav_path = argv[3];
    float noise_scale = (argc > 4) ? atof(argv[4]) : 0.0f;
    float noise_w = (argc > 5) ? atof(argv[5]) : 0.0f;
    const char * dump_dir = nullptr;
    for (int i = 4; i < argc - 1; i++) {
        if (strcmp(argv[i], "--dump") == 0) { dump_dir = argv[i+1]; break; }
    }

    struct piper_tts_params params = piper_tts_default_params();
    params.verbosity = 2;
    params.noise_scale = noise_scale;
    params.noise_w = noise_w;

    fprintf(stderr, "Loading model: %s\n", model_path);
    struct piper_tts_context * ctx = piper_tts_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    if (dump_dir) {
        piper_tts_set_dump_dir(ctx, dump_dir);
        fprintf(stderr, "Dumping intermediates to: %s\n", dump_dir);
    }

    fprintf(stderr, "Synthesizing IPA: %s\n", ipa);
    float * pcm = nullptr;
    int sr = 0;
    int n = piper_tts_synthesize_phonemes(ctx, ipa, &pcm, &sr);
    if (n <= 0 || !pcm) {
        fprintf(stderr, "Synthesis failed\n");
        piper_tts_free(ctx);
        return 1;
    }

    fprintf(stderr, "Output: %d samples @ %d Hz (%.3f s)\n", n, sr, (float)n / sr);

    // Print audio stats
    float min_v = pcm[0], max_v = pcm[0], rms = 0;
    for (int i = 0; i < n; i++) {
        if (pcm[i] < min_v) min_v = pcm[i];
        if (pcm[i] > max_v) max_v = pcm[i];
        rms += pcm[i] * pcm[i];
    }
    rms = sqrtf(rms / n);
    fprintf(stderr, "Audio range: [%.4f, %.4f], RMS: %.4f\n", min_v, max_v, rms);

    if (!write_wav(wav_path, pcm, n, sr)) {
        fprintf(stderr, "Failed to write WAV: %s\n", wav_path);
        free(pcm);
        piper_tts_free(ctx);
        return 1;
    }

    fprintf(stderr, "Written: %s\n", wav_path);
    free(pcm);
    piper_tts_free(ctx);
    return 0;
}
