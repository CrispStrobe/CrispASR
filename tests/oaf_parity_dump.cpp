// oaf_parity_dump.cpp — dump Onsets & Frames intermediates for the ONNX diff.
//
// Not a Catch2 test: it is the C++ half of tools/oaf_parity.py, which runs the
// same audio through native onnxruntime and diffs the head activations. The
// split exists because the reference is a Python runtime and the thing under
// test is not, and because §35.1's rule applies here too — a front-end
// mismatch does not raise, it just scores worse, so the mel is dumped and
// compared FIRST and a model difference cannot hide behind it.
//
//   oaf-parity-dump <model.gguf> <audio.wav> <out-prefix> [n_threads]
//
// Writes <prefix>.mel.f32 (T × 229) and <prefix>.<head>.f32 (T × 88, after the
// sigmoid) for head in onset, offset, frame, activation, velocity, plus
// <prefix>.meta.txt carrying T and the timing.

#include "onsets_and_frames.h"

#include <chrono>
#include <cstdio>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// 16-bit PCM WAV, any chunk layout. Returns mono float in [-1, 1) and the
// sample rate; no resampling — the caller is expected to hand over 16 kHz.
bool read_wav16(const char* path, std::vector<float>& out, int& sample_rate) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "oaf-parity-dump: cannot open %s\n", path);
        return false;
    }
    char riff[12];
    if (std::fread(riff, 1, 12, f) != 12 || std::memcmp(riff, "RIFF", 4) != 0 ||
        std::memcmp(riff + 8, "WAVE", 4) != 0) {
        std::fclose(f);
        std::fprintf(stderr, "oaf-parity-dump: %s is not a RIFF/WAVE file\n", path);
        return false;
    }
    int channels = 1, bits = 16;
    sample_rate = 0;
    std::vector<uint8_t> data;
    for (;;) {
        char id[4];
        uint32_t sz = 0;
        if (std::fread(id, 1, 4, f) != 4 || std::fread(&sz, 4, 1, f) != 1)
            break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            std::vector<uint8_t> fmt(sz);
            if (std::fread(fmt.data(), 1, sz, f) != sz)
                break;
            std::memcpy(&channels, fmt.data() + 2, 2);
            std::memcpy(&sample_rate, fmt.data() + 4, 4);
            std::memcpy(&bits, fmt.data() + 14, 2);
            channels &= 0xffff;
            bits &= 0xffff;
        } else if (std::memcmp(id, "data", 4) == 0) {
            data.resize(sz);
            if (std::fread(data.data(), 1, sz, f) != sz)
                data.clear();
            break;
        } else {
            std::fseek(f, (long)((sz + 1) & ~1u), SEEK_CUR);
        }
    }
    std::fclose(f);
    if (data.empty() || bits != 16 || channels < 1) {
        std::fprintf(stderr, "oaf-parity-dump: %s: need 16-bit PCM (got %d-bit, %d ch, %zu bytes)\n", path, bits,
                     channels, data.size());
        return false;
    }
    const size_t n = data.size() / 2 / (size_t)channels;
    out.resize(n);
    const int16_t* s = (const int16_t*)data.data();
    for (size_t i = 0; i < n; i++) {
        float acc = 0.0f;
        for (int c = 0; c < channels; c++)
            acc += (float)s[i * channels + c] / 32768.0f;
        out[i] = acc / (float)channels;
    }
    return true;
}

bool write_f32(const std::string& path, const float* data, size_t n) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f)
        return false;
    const bool ok = data && std::fwrite(data, sizeof(float), n, f) == n;
    std::fclose(f);
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: oaf-parity-dump <model.gguf> <audio.wav> <out-prefix> [n_threads]\n");
        return 2;
    }
    const std::string model = argv[1];
    const std::string wav = argv[2];
    const std::string prefix = argv[3];
    const int nthreads = argc > 4 ? std::atoi(argv[4]) : 4;

    std::vector<float> pcm;
    int sr = 0;
    if (!read_wav16(wav.c_str(), pcm, sr))
        return 3;
    if (sr != 16000)
        std::fprintf(stderr, "oaf-parity-dump: WARNING %s is %d Hz, the model wants 16000\n", wav.c_str(), sr);

    onsets_and_frames_params p = onsets_and_frames_default_params();
    p.n_threads = nthreads;
    p.verbosity = 2; // keeps the raw head outputs
    onsets_and_frames_ctx* ctx = onsets_and_frames_init_from_file(model.c_str(), p);
    if (!ctx) {
        std::fprintf(stderr, "oaf-parity-dump: failed to load %s\n", model.c_str());
        return 4;
    }

    int mel_frames = 0;
    float* mel = onsets_and_frames_mel(ctx, pcm.data(), (int)pcm.size(), &mel_frames);
    if (mel) {
        write_f32(prefix + ".mel.f32", mel, (size_t)mel_frames * 229);
        std::free(mel);
    }

    // Wall clock AND CPU time. This box runs several sessions at once, so a
    // wall-clock realtime factor measures the load average as much as the
    // model; CPU-seconds per audio-second is the number that survives that.
    auto cpu_ms = []() -> double {
#ifndef _WIN32
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        return (double)(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000.0 +
               (double)(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1000.0;
#else
        return 0.0;
#endif
    };
    const double c0 = cpu_ms();
    const auto t0 = std::chrono::steady_clock::now();
    onsets_and_frames_result res{};
    const int rc = onsets_and_frames_transcribe(ctx, pcm.data(), (int)pcm.size(), &res);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double cpu = cpu_ms() - c0;
    if (rc != 0) {
        std::fprintf(stderr, "oaf-parity-dump: transcribe failed (%d)\n", rc);
        onsets_and_frames_free(ctx);
        return 5;
    }

    const size_t n = (size_t)res.n_frames * res.n_classes;
    write_f32(prefix + ".onset.f32", res.onset_output, n);
    write_f32(prefix + ".offset.f32", res.offset_output, n);
    write_f32(prefix + ".frame.f32", res.frame_output, n);
    write_f32(prefix + ".activation.f32", res.activation_output, n);
    write_f32(prefix + ".velocity.f32", res.velocity_output, n);

    const double audio_sec = (double)pcm.size() / 16000.0;
    FILE* meta = std::fopen((prefix + ".meta.txt").c_str(), "w");
    if (meta) {
        std::fprintf(meta,
                     "frames %d\nclasses %d\nmel_frames %d\nnotes %d\naudio_seconds %.4f\n"
                     "elapsed_ms %.2f\nrealtime_factor %.4f\ncpu_ms %.2f\ncpu_factor %.4f\nthreads %d\n",
                     res.n_frames, res.n_classes, mel_frames, res.n_notes, audio_sec, ms,
                     ms / 1000.0 / (audio_sec > 0 ? audio_sec : 1.0), cpu,
                     cpu / 1000.0 / (audio_sec > 0 ? audio_sec : 1.0), nthreads);
        std::fclose(meta);
    }
    std::printf("frames=%d notes=%d %.1f ms wall / %.1f ms cpu for %.2f s audio "
                "(%.4f x real time, %.4f cpu-s per audio-s, %d threads)\n",
                res.n_frames, res.n_notes, ms, cpu, audio_sec, ms / 1000.0 / (audio_sec > 0 ? audio_sec : 1.0),
                cpu / 1000.0 / (audio_sec > 0 ? audio_sec : 1.0), nthreads);

    // Note events, so the F1 scorer does not have to re-implement the decoder.
    FILE* nf = std::fopen((prefix + ".notes.tsv").c_str(), "w");
    if (nf) {
        for (int i = 0; i < res.n_notes; i++) {
            const onsets_and_frames_note_event& e = res.note_events[i];
            std::fprintf(nf, "%.4f\t%.4f\t%d\t%d\n", e.onset_time, e.offset_time, e.midi_note, e.velocity);
        }
        std::fclose(nf);
    }

    onsets_and_frames_result_free(&res);
    onsets_and_frames_free(ctx);
    return 0;
}
