// vibevoice-test-stages — stage-by-stage differential test for VibeVoice-ASR.
//
// Loads a reference archive produced by tools/dump_reference.py --backend vibevoice
// and compares each pipeline stage against the C++ ggml forward pass.
//
// Stages tested (in order):
//   audio_norm     → normalised 24 kHz PCM used by both encoders
//   at_enc_mean    → acoustic σ-VAE encoder mean  (T', 64)
//   st_enc_mean    → semantic encoder mean          (T', 128)
//   at_conn_out    → acoustic SpeechConnector out   (T', 3584)
//   st_conn_out    → semantic SpeechConnector out   (T', 3584)
//   speech_features→ elementwise sum of both conn   (T', 3584)
//
// Reference archive is produced by:
//
//   python tools/dump_reference.py --backend vibevoice \
//       --model-dir /path/to/microsoft/VibeVoice-ASR \
//       --audio samples/jfk_24k.wav \
//       --output /tmp/vibevoice-ref.gguf
//
// Usage:
//   vibevoice-test-stages model.gguf reference.gguf [audio.wav]
//
// If audio.wav is omitted the normalised PCM from the reference archive
// (key "audio_norm") is used as the encoder input.

#include "crispasr_diff.h"
#include "vibevoice.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ── tiny WAV loader (PCM 16-bit only, mono) ──────────────────────────────────
static std::vector<float> load_wav_mono(const char* path, int* out_sr) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return {};
    }

    char riff[4];
    fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4)) {
        fclose(f);
        fprintf(stderr, "%s: not a RIFF file\n", path);
        return {};
    }

    uint32_t chunk_size;
    fread(&chunk_size, 4, 1, f);
    char wave[4];
    fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4)) {
        fclose(f);
        fprintf(stderr, "%s: not WAVE\n", path);
        return {};
    }

    uint16_t audio_fmt = 0, n_ch = 0, bps = 0;
    uint32_t sr = 0;
    std::vector<float> samples;

    char id[4];
    uint32_t sz;
    while (fread(id, 1, 4, f) == 4 && fread(&sz, 4, 1, f) == 1) {
        if (!memcmp(id, "fmt ", 4)) {
            fread(&audio_fmt, 2, 1, f);
            fread(&n_ch, 2, 1, f);
            fread(&sr, 4, 1, f);
            uint32_t byte_rate;
            fread(&byte_rate, 4, 1, f);
            uint16_t block_align;
            fread(&block_align, 2, 1, f);
            fread(&bps, 2, 1, f);
            if (sz > 16)
                fseek(f, sz - 16, SEEK_CUR);
        } else if (!memcmp(id, "data", 4)) {
            if (audio_fmt != 1 || bps != 16) {
                fprintf(stderr, "%s: only PCM 16-bit supported\n", path);
                fclose(f);
                return {};
            }
            size_t n = sz / sizeof(int16_t);
            std::vector<int16_t> raw(n);
            fread(raw.data(), sizeof(int16_t), n, f);
            // downmix to mono
            size_t n_mono = (n_ch > 1) ? n / n_ch : n;
            samples.resize(n_mono);
            for (size_t i = 0; i < n_mono; ++i) {
                float s = 0;
                for (int c = 0; c < (int)n_ch; ++c)
                    s += raw[i * n_ch + c];
                samples[i] = s / (n_ch * 32768.0f);
            }
        } else {
            fseek(f, sz, SEEK_CUR);
        }
    }
    fclose(f);
    if (out_sr)
        *out_sr = (int)sr;
    return samples;
}

// ── print a stage result ─────────────────────────────────────────────────────
static bool print_stage(const char* name, const crispasr_diff::Report& rep, float cos_threshold,
                        const crispasr_diff::Ref& ref, const float* cpp_data) {
    const bool pass = rep.is_pass(cos_threshold);
    fprintf(stderr, "\n[%s]\n", name);
    fprintf(stderr, "  elements:        %zu\n", rep.n_elem);
    fprintf(stderr, "  max abs diff:    %.4e\n", rep.max_abs);
    fprintf(stderr, "  mean abs diff:   %.4e\n", rep.mean_abs);
    fprintf(stderr, "  RMS error:       %.4e\n", rep.rms);
    fprintf(stderr, "  cos min:         %.6f\n", rep.cos_min);
    fprintf(stderr, "  cos mean:        %.6f\n", rep.cos_mean);
    fprintf(stderr, "  verdict: %s  (threshold %.3f)\n", pass ? "PASS" : "FAIL", cos_threshold);
    if (!pass && cpp_data) {
        // Print first 8 values at frame 0 for both C++ and reference (LEARNINGS.md step 3)
        auto [ref_ptr, ref_n] = ref.get_f32(name);
        int n_show = 8;
        fprintf(stderr, "  frame-0 C++: ");
        for (int i = 0; i < n_show && (size_t)i < rep.n_elem; i++)
            fprintf(stderr, "%.5f ", cpp_data[i]);
        fprintf(stderr, "\n");
        if (ref_ptr) {
            fprintf(stderr, "  frame-0 ref: ");
            for (int i = 0; i < n_show && (size_t)i < ref_n; i++)
                fprintf(stderr, "%.5f ", ref_ptr[i]);
            fprintf(stderr, "\n");
        }
    }
    return pass;
}

int main(int argc, char** argv) {
    // Parse flags before positional args
    bool use_gpu = false; // CPU by default for safe memory use during validation
    int arg_start = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
            arg_start = i + 1;
        } else if (strcmp(argv[i], "--cpu") == 0) {
            use_gpu = false;
            arg_start = i + 1;
        } else {
            arg_start = i;
            break;
        }
    }

    if (argc - arg_start < 2) {
        fprintf(stderr,
                "usage: %s [--cpu|--gpu] model.gguf reference.gguf [audio.wav]\n"
                "\n"
                "  --cpu          use CPU backend (default, OOM-safe for large models)\n"
                "  --gpu          use GPU/Metal backend\n"
                "  model.gguf     VibeVoice-ASR GGUF weights\n"
                "  reference.gguf archive from tools/dump_reference.py --backend vibevoice\n"
                "  audio.wav      optional 24 kHz mono PCM (uses ref audio_norm otherwise)\n",
                argv[0]);
        return 1;
    }

    // ── 1. Load reference archive ─────────────────────────────────────────────
    crispasr_diff::Ref ref;
    if (!ref.load(argv[arg_start + 1]))
        return 2;

    // ── 2. Obtain input audio ─────────────────────────────────────────────────
    std::vector<float> audio;
    if (argc - arg_start >= 3) {
        int sr = 0;
        audio = load_wav_mono(argv[arg_start + 2], &sr);
        if (audio.empty())
            return 3;
        if (sr != 24000) {
            fprintf(stderr, "WARNING: audio sample rate is %d Hz, expected 24000 Hz\n", sr);
        }
        fprintf(stderr, "loaded %s: %zu samples @ %d Hz (%.2f s)\n", argv[arg_start + 2], audio.size(), sr,
                audio.size() / (float)sr);
    } else if (ref.has("audio_norm")) {
        auto [ptr, n] = ref.get_f32("audio_norm");
        audio.assign(ptr, ptr + n);
        fprintf(stderr, "using reference audio_norm: %zu samples\n", audio.size());
    } else {
        fprintf(stderr, "no audio provided and reference has no 'audio_norm' — "
                        "pass an audio.wav argument\n");
        return 3;
    }

    // ── 3. Init model ─────────────────────────────────────────────────────────
    auto cp = vibevoice_context_default_params();
    cp.n_threads = 4;
    cp.verbosity = 1;
    cp.use_gpu = use_gpu;
    auto* ctx = vibevoice_init_from_file(argv[arg_start], cp);
    if (!ctx) {
        fprintf(stderr, "failed to load model from %s\n", argv[1]);
        return 4;
    }

    const float* pcm = audio.data();
    const int n_pcm = (int)audio.size();
    bool all_pass = true;
    // Encoder stages: conv weights are 3D → stay F16 in the quantized model.
    // 26 ConvNeXt blocks with F16 dw_conv accumulate ~cos 0.99 at the output.
    // Connector stages: FC weights are 2D → quantized to Q4_K by crispasr-quantize,
    // adding another noise layer (fc2 is 3584×3584=12.8M weights). cos_min ~0.986
    // with Q4_K; for F16 models expect cos_min >0.999.
    const float ENC_COS_THRESHOLD = 0.90f;
    const float CONN_COS_THRESHOLD = 0.985f;

    // ── Stage 1: acoustic encoder ─────────────────────────────────────────────
    if (ref.has("at_enc_mean")) {
        int T_at = 0, vd_at = 0;
        float* at_mean = vibevoice_run_acoustic_encoder(ctx, pcm, n_pcm, &T_at, &vd_at);
        if (!at_mean) {
            fprintf(stderr, "[at_enc_mean] FAIL — vibevoice_run_acoustic_encoder returned null\n");
            all_pass = false;
        } else {
            fprintf(stderr, "acoustic encoder: T=%d, vae_dim=%d\n", T_at, vd_at);
            auto rep = ref.compare("at_enc_mean", at_mean, (size_t)T_at * vd_at);
            all_pass &= print_stage("at_enc_mean", rep, ENC_COS_THRESHOLD, ref, at_mean);
            free(at_mean);
        }
    } else {
        fprintf(stderr, "[at_enc_mean] skipped (not in reference archive)\n");
    }

    // ── Stage 2: semantic encoder ─────────────────────────────────────────────
    if (ref.has("st_enc_mean")) {
        int T_st = 0, vd_st = 0;
        float* st_mean = vibevoice_run_semantic_encoder(ctx, pcm, n_pcm, &T_st, &vd_st);
        if (!st_mean) {
            fprintf(stderr, "[st_enc_mean] FAIL — vibevoice_run_semantic_encoder returned null\n");
            all_pass = false;
        } else {
            fprintf(stderr, "semantic encoder: T=%d, vae_dim=%d\n", T_st, vd_st);
            auto rep = ref.compare("st_enc_mean", st_mean, (size_t)T_st * vd_st);
            all_pass &= print_stage("st_enc_mean", rep, ENC_COS_THRESHOLD, ref, st_mean);
            free(st_mean);
        }
    } else {
        fprintf(stderr, "[st_enc_mean] skipped (not in reference archive)\n");
    }

    // ── Stages 3+4: connectors need encoder means ─────────────────────────────
    // Re-run both encoders to get means needed for connector inputs.
    int T_at = 0, vd_at = 0, T_st = 0, vd_st = 0;
    float* at_mean2 = vibevoice_run_acoustic_encoder(ctx, pcm, n_pcm, &T_at, &vd_at);
    float* st_mean2 = vibevoice_run_semantic_encoder(ctx, pcm, n_pcm, &T_st, &vd_st);

    // ── Stage 3: acoustic connector ───────────────────────────────────────────
    if (ref.has("at_conn_out") && at_mean2) {
        int d_lm = 0;
        float* at_conn = vibevoice_run_connector(ctx, "at_conn", at_mean2, T_at, vd_at, &d_lm);
        if (!at_conn) {
            fprintf(stderr, "[at_conn_out] FAIL — vibevoice_run_connector returned null\n");
            all_pass = false;
        } else {
            fprintf(stderr, "acoustic connector: T=%d, d_lm=%d\n", T_at, d_lm);
            auto rep = ref.compare("at_conn_out", at_conn, (size_t)T_at * d_lm);
            all_pass &= print_stage("at_conn_out", rep, CONN_COS_THRESHOLD, ref, at_conn);
            free(at_conn);
        }
    } else if (!ref.has("at_conn_out")) {
        fprintf(stderr, "[at_conn_out] skipped (not in reference archive)\n");
    }

    // ── Stage 4: semantic connector ───────────────────────────────────────────
    if (ref.has("st_conn_out") && st_mean2) {
        int d_lm = 0;
        float* st_conn = vibevoice_run_connector(ctx, "se_conn", st_mean2, T_st, vd_st, &d_lm);
        if (!st_conn) {
            fprintf(stderr, "[st_conn_out] FAIL — vibevoice_run_connector returned null\n");
            all_pass = false;
        } else {
            fprintf(stderr, "semantic connector: T=%d, d_lm=%d\n", T_st, d_lm);
            auto rep = ref.compare("st_conn_out", st_conn, (size_t)T_st * d_lm);
            all_pass &= print_stage("st_conn_out", rep, CONN_COS_THRESHOLD, ref, st_conn);
            free(st_conn);
        }
    } else if (!ref.has("st_conn_out")) {
        fprintf(stderr, "[st_conn_out] skipped (not in reference archive)\n");
    }

    free(at_mean2);
    free(st_mean2);

    // ── Stage 5: combined speech features ────────────────────────────────────
    if (ref.has("speech_features")) {
        int T_sf = 0, d_sf = 0;
        float* sf = vibevoice_encode_speech(ctx, pcm, n_pcm, &T_sf, &d_sf);
        if (!sf) {
            fprintf(stderr, "[speech_features] FAIL — vibevoice_encode_speech returned null\n");
            all_pass = false;
        } else {
            fprintf(stderr, "speech features: T=%d, d=%d\n", T_sf, d_sf);
            auto rep = ref.compare("speech_features", sf, (size_t)T_sf * d_sf);
            // Combined features depend on encoder means (F16 drift) + connector (F32 exact).
            // Use the encoder threshold since the dominant error source is the encoder.
            all_pass &= print_stage("speech_features", rep, ENC_COS_THRESHOLD, ref, sf);
            free(sf);
        }
    } else {
        fprintf(stderr, "[speech_features] skipped (not in reference archive)\n");
    }

    vibevoice_free(ctx);

    fprintf(stderr, "\n══ overall: %s ══\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
