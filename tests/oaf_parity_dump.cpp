// oaf_parity_dump.cpp — dump Onsets & Frames intermediates for the ONNX diff.
//
// Not a Catch2 test: it is the C++ half of tools/oaf_parity.py, which runs the
// same audio through native onnxruntime and diffs the head activations. The
// split exists because the reference is a Python runtime and the thing under
// test is not, and because §35.1's rule applies here too — a front-end
// mismatch does not raise, it just scores worse, so the mel is dumped and
// compared FIRST and a model difference cannot hide behind it.
//
//   oaf-parity-dump <model.gguf> <audio.wav> <out-prefix> [n_threads] [n_repeats]
//
// n_repeats > 1 transcribes the same audio that many times on the SAME context
// and asserts every run is bitwise identical to the first. That is the
// repeated-call validation docs/ggml-optimisation-playbook.md §5.7 item 5 and
// §6.7 require of any persistent-allocator change: a gallocr that outlives the
// call may alias an input tensor's slot with a later intermediate, and the
// symptom is run 0 correct and run 1 onward quietly wrong -- invisible to a
// one-shot CLI invocation, which is how this class of bug has shipped before
// (#208, and LEARNINGS.md L14367 on omnivoice). It also reports the per-run
// CPU cost, so a timing arm gets a median instead of a single sample.
//
// Writes <prefix>.mel.f32 (T × 229) and <prefix>.<head>.f32 (T × 88, after the
// sigmoid) for head in onset, offset, frame, activation, velocity, plus
// <prefix>.meta.txt carrying T and the timing.

#include "onsets_and_frames.h"

#include <algorithm>
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
    const int nrepeats = argc > 5 ? std::max(1, std::atoi(argv[5])) : 1;

    std::vector<float> pcm;
    int sr = 0;
    if (!read_wav16(wav.c_str(), pcm, sr))
        return 3;
    if (sr != 16000)
        std::fprintf(stderr, "oaf-parity-dump: WARNING %s is %d Hz, the model wants 16000\n", wav.c_str(), sr);

    onsets_and_frames_params p = onsets_and_frames_default_params();
    p.n_threads = nthreads;
    p.verbosity = 2; // keeps the raw head outputs
    // Parity is a CPU question by default, so nothing changes for existing
    // callers on a CUDA/Metal box. CRISPASR_PARITY_USE_GPU=1 opts the dump
    // onto the GPU backend so the SAME harness can answer "does the GPU path
    // produce the same numbers?" -- which is the only way that question gets
    // an honest answer. CRISPASR_OAF_NO_GPU=1 still wins over it.
    p.use_gpu = std::getenv("CRISPASR_PARITY_USE_GPU") != nullptr;
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

    // Repeated-call validation. See the header comment: this is the only thing
    // that surfaces second-use allocator corruption, and it compares the FULL
    // head tensors bitwise, not the note list, because a drifted activation can
    // decode to the same notes and still be a bug.
    for (int rep = 1; rep < nrepeats; rep++) {
        const double rc0 = cpu_ms();
        const auto rt0 = std::chrono::steady_clock::now();
        onsets_and_frames_result r2{};
        if (onsets_and_frames_transcribe(ctx, pcm.data(), (int)pcm.size(), &r2) != 0) {
            std::fprintf(stderr, "oaf-parity-dump: repeat %d failed\n", rep);
            onsets_and_frames_free(ctx);
            return 5;
        }
        const double rms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rt0).count();
        const double rcpu = cpu_ms() - rc0;
        bool same = r2.n_frames == res.n_frames && r2.n_notes == res.n_notes && r2.n_classes == res.n_classes;
        const size_t nn = (size_t)res.n_frames * res.n_classes;
        const float* mine[5] = {r2.onset_output, r2.offset_output, r2.frame_output, r2.activation_output,
                                r2.velocity_output};
        const float* first[5] = {res.onset_output, res.offset_output, res.frame_output, res.activation_output,
                                 res.velocity_output};
        const char* hname[5] = {"onset", "offset", "frame", "activation", "velocity"};
        for (int k = 0; same && k < 5; k++) {
            if (!mine[k] || !first[k]) {
                same = false;
                break;
            }
            for (size_t i = 0; i < nn; i++) {
                if (mine[k][i] != first[k][i]) {
                    std::fprintf(stderr,
                                 "oaf-parity-dump: REPEAT %d DIVERGED at %s[%zu]: %.9g vs %.9g on run 0 -- this is "
                                 "second-use allocator corruption (playbook §6.7)\n",
                                 rep, hname[k], i, (double)mine[k][i], (double)first[k][i]);
                    same = false;
                    break;
                }
            }
        }
        std::printf("  repeat %d: %s  (%d frames, %d notes, %.1f ms wall / %.1f ms cpu)\n", rep,
                    same ? "bitwise identical to run 0" : "*** DIVERGED ***", r2.n_frames, r2.n_notes, rms_, rcpu);
        onsets_and_frames_result_free(&r2);
        if (!same) {
            onsets_and_frames_result_free(&res);
            onsets_and_frames_free(ctx);
            return 6;
        }
    }

    const size_t n = (size_t)res.n_frames * res.n_classes;
    write_f32(prefix + ".onset.f32", res.onset_output, n);
    write_f32(prefix + ".offset.f32", res.offset_output, n);
    write_f32(prefix + ".frame.f32", res.frame_output, n);
    write_f32(prefix + ".activation.f32", res.activation_output, n);
    write_f32(prefix + ".velocity.f32", res.velocity_output, n);

    // Peak RSS, the same field hft-parity-dump has emitted since it was
    // written. Added here so the two piano arms can be put in one table --
    // a CPU-vs-GPU A/B has to report memory as well as time, because moving
    // weights onto a device changes where the bytes live, not just how fast
    // they are read.
    auto peak_rss_mib = []() -> double {
#ifndef _WIN32
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
        return (double)ru.ru_maxrss / 1048576.0; // bytes on macOS
#else
        return (double)ru.ru_maxrss / 1024.0; // kilobytes on Linux
#endif
#else
        return 0.0;
#endif
    };

    const double audio_sec = (double)pcm.size() / 16000.0;
    FILE* meta = std::fopen((prefix + ".meta.txt").c_str(), "w");
    if (meta) {
        std::fprintf(meta,
                     "frames %d\nclasses %d\nmel_frames %d\nnotes %d\naudio_seconds %.4f\n"
                     "elapsed_ms %.2f\nrealtime_factor %.4f\ncpu_ms %.2f\ncpu_factor %.4f\nthreads %d\n"
                     "peak_rss_mib %.1f\n",
                     res.n_frames, res.n_classes, mel_frames, res.n_notes, audio_sec, ms,
                     ms / 1000.0 / (audio_sec > 0 ? audio_sec : 1.0), cpu,
                     cpu / 1000.0 / (audio_sec > 0 ? audio_sec : 1.0), nthreads, peak_rss_mib());
        std::fclose(meta);
    }
    std::printf("frames=%d notes=%d %.1f ms wall / %.1f ms cpu for %.2f s audio "
                "(%.4f x real time, %.4f cpu-s per audio-s, %d threads, peak RSS %.0f MiB)\n",
                res.n_frames, res.n_notes, ms, cpu, audio_sec, ms / 1000.0 / (audio_sec > 0 ? audio_sec : 1.0),
                cpu / 1000.0 / (audio_sec > 0 ? audio_sec : 1.0), nthreads, peak_rss_mib());

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
