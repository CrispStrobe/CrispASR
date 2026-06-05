// Quick test: run vocoder on ref mel, write WAV, compare with Python reference.
#include "chatterbox_s3gen.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

static void write_wav(const char* path, const float* data, int n_samples, int sample_rate) {
    FILE* f = fopen(path, "wb");
    if (!f)
        return;
    int16_t* pcm16 = (int16_t*)malloc(n_samples * 2);
    for (int i = 0; i < n_samples; i++) {
        float v = data[i];
        if (v > 1.0f)
            v = 1.0f;
        if (v < -1.0f)
            v = -1.0f;
        pcm16[i] = (int16_t)(v * 32767.0f);
    }
    uint32_t data_size = n_samples * 2;
    uint32_t file_size = 36 + data_size;
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_fmt = 1;
    fwrite(&audio_fmt, 2, 1, f);
    uint16_t channels = 1;
    fwrite(&channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    uint32_t byte_rate = sample_rate * 2;
    fwrite(&byte_rate, 4, 1, f);
    uint16_t block_align = 2;
    fwrite(&block_align, 2, 1, f);
    uint16_t bits = 16;
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    fwrite(pcm16, 2, n_samples, f);
    free(pcm16);
    fclose(f);
}

int main() {
    auto* ctx = chatterbox_s3gen_init_from_file("/mnt/storage/chatterbox/chatterbox-s3gen-f16.gguf", 4, 2, false);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    std::vector<float> mel(80 * 62);
    FILE* f = fopen("/mnt/storage/chatterbox/ref_mel_80x62.bin", "rb");
    fread(mel.data(), 4, mel.size(), f);
    fclose(f);

    int n_samples = 0;
    float* pcm = chatterbox_s3gen_vocode(ctx, mel.data(), 62, &n_samples);
    fprintf(stderr, "Produced %d samples at 24 kHz (%.2f sec)\n", n_samples, n_samples / 24000.0f);

    write_wav("/mnt/storage/chatterbox/cpp_voc_fixed.wav", pcm, n_samples, 24000);
    fprintf(stderr, "Wrote /mnt/storage/chatterbox/cpp_voc_fixed.wav\n");

    free(pcm);
    chatterbox_s3gen_free(ctx);
    return 0;
}
