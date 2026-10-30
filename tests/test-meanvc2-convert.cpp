// test-meanvc2-convert.cpp — standalone convert harness for MeanVC2.
#include "meanvc2.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <meanvc2.gguf> [input.wav] [out.wav]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* src_path = (argc >= 3) ? argv[2] : "samples/jfk.wav";
    const char* out_path = (argc >= 4) ? argv[3] : "/tmp/meanvc2-output.wav";

    auto cp = meanvc2_context_default_params();
    cp.verbosity = 1;
    cp.use_gpu = true;
    auto* ctx = meanvc2_init_from_file(model_path, cp);
    if (!ctx) {
        fprintf(stderr, "Failed to load model '%s'\n", model_path);
        return 1;
    }

    FILE* f = fopen(src_path, "rb");
    if (!f) {
        const char* alt = "../samples/jfk.wav";
        f = fopen(alt, "rb");
        if (f)
            src_path = alt;
    }
    if (!f) {
        fprintf(stderr, "Cannot open '%s'\n", src_path);
        meanvc2_free(ctx);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> wav_data(fsize);
    fread(wav_data.data(), 1, fsize, f);
    fclose(f);

    int sr = 16000, bits = 16;
    std::vector<float> pcm;
    size_t pos = 12;
    while (pos < wav_data.size() - 8) {
        uint32_t csz = *(uint32_t*)(wav_data.data() + pos + 4);
        if (memcmp(wav_data.data() + pos, "fmt ", 4) == 0) {
            sr = *(int32_t*)(wav_data.data() + pos + 12);
            bits = *(int16_t*)(wav_data.data() + pos + 22);
        } else if (memcmp(wav_data.data() + pos, "data", 4) == 0) {
            int n = csz / (bits / 8);
            pcm.resize(n);
            const int16_t* samples = (const int16_t*)(wav_data.data() + pos + 8);
            for (int i = 0; i < n; i++)
                pcm[i] = samples[i] / 32768.0f;
            break;
        }
        pos += 8 + csz;
    }
    fprintf(stderr, "Loaded source: %zu samples @ %d Hz from %s\n", pcm.size(), sr, src_path);

    // Process full audio if requested, else default to 25600 samples (1.6s) for quick test
    int test_samples = (getenv("CRISPASR_MEANVC2_ALL") != nullptr)
        ? (int)pcm.size()
        : std::min((int)pcm.size(), 25600);

    int out_samples = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    float* out = meanvc2_convert(ctx, pcm.data(), test_samples, &out_samples);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double audio_sec = (double)test_samples / 16000.0;
    double rtf = (elapsed_ms / 1000.0) / audio_sec;

    fprintf(stderr, "Conversion time: %.2f ms for %.2f s audio (RTF: %.4f, %.2fx realtime)\n",
            elapsed_ms, audio_sec, rtf, 1.0 / rtf);

    if (!out || out_samples <= 0) {
        fprintf(stderr, "FAIL: meanvc2_convert returned null or 0 samples\n");
        meanvc2_free(ctx);
        return 1;
    }

    float mn = out[0], mx = out[0];
    double sum = 0.0;
    for (int i = 0; i < out_samples; i++) {
        if (out[i] < mn) mn = out[i];
        if (out[i] > mx) mx = out[i];
        sum += out[i];
    }
    fprintf(stderr, "Output: %d samples, min=%.6f, max=%.6f, mean=%.6f\n", out_samples, mn, mx, (float)(sum / out_samples));

    if (mx <= 0.01f && mn >= -0.01f) {
        fprintf(stderr, "FAIL: output is near-silent\n");
        meanvc2_free_audio(out);
        meanvc2_free(ctx);
        return 1;
    }

    // Write output WAV (16 kHz mono 16-bit PCM)
    FILE* wf = fopen(out_path, "wb");
    if (wf) {
        std::vector<int16_t> pcm16(out_samples);
        for (int i = 0; i < out_samples; i++) {
            float v = out[i] * 32767.0f;
            if (v > 32767.0f) v = 32767.0f;
            if (v < -32768.0f) v = -32768.0f;
            pcm16[i] = (int16_t)v;
        }
        uint32_t data_size = out_samples * 2;
        uint32_t riff_size = 36 + data_size;
        fwrite("RIFF", 1, 4, wf);
        fwrite(&riff_size, 4, 1, wf);
        fwrite("WAVE", 1, 4, wf);
        fwrite("fmt ", 1, 4, wf);
        uint32_t fmt_size = 16;
        fwrite(&fmt_size, 4, 1, wf);
        uint16_t audio_fmt = 1;
        fwrite(&audio_fmt, 2, 1, wf);
        uint16_t channels = 1;
        fwrite(&channels, 2, 1, wf);
        uint32_t sr_out = 16000;
        fwrite(&sr_out, 4, 1, wf);
        uint32_t byte_rate = 32000;
        fwrite(&byte_rate, 4, 1, wf);
        uint16_t block_align = 2;
        fwrite(&block_align, 2, 1, wf);
        uint16_t bps = 16;
        fwrite(&bps, 2, 1, wf);
        fwrite("data", 1, 4, wf);
        fwrite(&data_size, 4, 1, wf);
        fwrite(pcm16.data(), 2, out_samples, wf);
        fclose(wf);
        fprintf(stderr, "Wrote %s (%d samples)\n", out_path, out_samples);
    }

    meanvc2_free_audio(out);
    meanvc2_free(ctx);
    fprintf(stderr, "PASS: meanvc2 conversion successful\n");
    return 0;
}
