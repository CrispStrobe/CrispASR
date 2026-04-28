// crispasr_diff_main.cpp — CLI frontend for the ground-truth diff harness.
//
// Companion to tools/dump_reference.py. Given a reference GGUF archive
// produced by the Python dumper and a crispasr backend + model, runs the
// backend's public stage API (currently: mel spectrogram) and reports how
// closely the C++ forward path matches the PyTorch reference at every
// named stage.
//
// This is an incremental tool — it covers the stages the backends
// currently expose through their C headers. As more per-stage functions
// are exposed (audio_encoder, projector, embed_tokens, run_llm_kv, ...)
// the diff tool grows to call each of them and report at every
// architectural boundary. The C++ code for stage comparisons lives in
// crispasr_diff.{h,cpp}.
//
// Usage:
//   crispasr-diff <backend> <model.gguf> <reference.gguf> <audio.wav>
//
// Example:
//   python tools/dump_reference.py --backend voxtral \
//       --model-dir /hf/voxtral-mini-3b-2507 \
//       --audio samples/jfk.wav \
//       --output /tmp/voxtral-ref.gguf
//   build/bin/crispasr-diff voxtral \
//       voxtral-mini-3b-2507-q4_k.gguf \
//       /tmp/voxtral-ref.gguf \
//       samples/jfk.wav
//
// Typical output:
//   [PASS] mel_spectrogram     shape=[128,3000]  cos_min=0.99998  max_abs=3.1e-5
//   [FAIL] encoder_output      shape=[375,1280]  cos_min=0.92     max_abs=0.87
//   [SKIP] projector_output    (stage not exposed by backend API)

#include "crispasr_diff.h"

#include "voxtral.h"
#include "voxtral4b.h"
#include "qwen3_asr.h"
#include "granite_speech.h"
#include "parakeet.h"
#include "canary.h"
#include "cohere.h"
#include "gemma4_e2b.h"

#include "common-whisper.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Per-backend stage runners
// ---------------------------------------------------------------------------
//
// Each "stage runner" below takes a loaded model, some input tensor, and
// returns a freshly-allocated float buffer that can be compared against a
// reference. Stages are named to match the Python side
// (mel_spectrogram, encoder_output, projector_output, llm_logits).
//
// We only wire up the backends + stages whose C headers expose a
// standalone entry point. Everything else is reported as [SKIP].

namespace {

struct StageResult {
    bool ok = false;
    std::vector<float> data;
    std::vector<int> shape; // canonical order: outer..inner
    std::string note;       // filled when ok=false to explain skip
};

// ---- voxtral 3B ----

static StageResult voxtral_mel(voxtral_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = voxtral_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "voxtral_compute_mel returned null";
        return r;
    }
    r.shape = {n_mels, T_mel};
    r.data.assign(mel, mel + (size_t)n_mels * T_mel);
    free(mel);
    r.ok = true;
    return r;
}

static StageResult voxtral_encoder(voxtral_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = voxtral_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "mel failed";
        return r;
    }
    int N_enc = 0, pdim = 0;
    float* enc = voxtral_run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &pdim);
    free(mel);
    if (!enc) {
        r.note = "voxtral_run_encoder returned null";
        return r;
    }
    r.shape = {N_enc, pdim};
    r.data.assign(enc, enc + (size_t)N_enc * pdim);
    free(enc);
    r.ok = true;
    return r;
}

// ---- voxtral4b ----

static StageResult voxtral4b_mel(voxtral4b_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = voxtral4b_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "voxtral4b_compute_mel returned null";
        return r;
    }
    r.shape = {n_mels, T_mel};
    r.data.assign(mel, mel + (size_t)n_mels * T_mel);
    free(mel);
    r.ok = true;
    return r;
}

// ---- qwen3 ----

static StageResult qwen3_mel(qwen3_asr_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = qwen3_asr_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "qwen3_asr_compute_mel returned null";
        return r;
    }
    r.shape = {n_mels, T_mel};
    r.data.assign(mel, mel + (size_t)n_mels * T_mel);
    free(mel);
    r.ok = true;
    return r;
}

// ---- granite ----

static StageResult granite_mel(granite_speech_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = granite_speech_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "granite_speech_compute_mel returned null";
        return r;
    }
    r.shape = {n_mels, T_mel};
    r.data.assign(mel, mel + (size_t)n_mels * T_mel);
    free(mel);
    r.ok = true;
    return r;
}

// ---- parakeet (NeMo FastConformer + TDT) ----

static StageResult parakeet_mel_r(parakeet_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = parakeet_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "parakeet_compute_mel returned null";
        return r;
    }
    r.shape = {n_mels, T_mel};
    r.data.assign(mel, mel + (size_t)n_mels * T_mel);
    free(mel);
    r.ok = true;
    return r;
}

static StageResult parakeet_encoder_r(parakeet_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = parakeet_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "mel failed";
        return r;
    }
    int T_enc = 0, d_model = 0;
    float* enc = parakeet_run_encoder(ctx, mel, n_mels, T_mel, &T_enc, &d_model);
    free(mel);
    if (!enc) {
        r.note = "parakeet_run_encoder returned null";
        return r;
    }
    r.shape = {T_enc, d_model};
    r.data.assign(enc, enc + (size_t)T_enc * d_model);
    free(enc);
    r.ok = true;
    return r;
}

// ---- canary (NeMo FastConformer + Transformer decoder) ----

static StageResult canary_mel_r(canary_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = canary_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "canary_compute_mel returned null";
        return r;
    }
    r.shape = {n_mels, T_mel};
    r.data.assign(mel, mel + (size_t)n_mels * T_mel);
    free(mel);
    r.ok = true;
    return r;
}

static StageResult canary_encoder_r(canary_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = canary_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "mel failed";
        return r;
    }
    int T_enc = 0, d_model = 0;
    float* enc = canary_run_encoder(ctx, mel, n_mels, T_mel, &T_enc, &d_model);
    free(mel);
    if (!enc) {
        r.note = "canary_run_encoder returned null";
        return r;
    }
    r.shape = {T_enc, d_model};
    r.data.assign(enc, enc + (size_t)T_enc * d_model);
    free(enc);
    r.ok = true;
    return r;
}

// ---- cohere (Conformer + Transformer) ----

static StageResult cohere_mel_r(cohere_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = cohere_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "cohere_compute_mel returned null";
        return r;
    }
    r.shape = {n_mels, T_mel};
    r.data.assign(mel, mel + (size_t)n_mels * T_mel);
    free(mel);
    r.ok = true;
    return r;
}

static StageResult cohere_encoder_r(cohere_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = cohere_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "mel failed";
        return r;
    }
    int T_enc = 0, d_model = 0;
    float* enc = cohere_run_encoder(ctx, mel, n_mels, T_mel, &T_enc, &d_model);
    free(mel);
    if (!enc) {
        r.note = "cohere_run_encoder returned null";
        return r;
    }
    r.shape = {T_enc, d_model};
    r.data.assign(enc, enc + (size_t)T_enc * d_model);
    free(enc);
    r.ok = true;
    return r;
}

// ---- gemma4-e2b (USM Conformer + Gemma4 LLM) ----

static StageResult gemma4_mel_r(gemma4_e2b_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = gemma4_e2b_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "gemma4_e2b_compute_mel returned null";
        return r;
    }
    r.shape = {n_mels, T_mel};
    r.data.assign(mel, mel + (size_t)n_mels * T_mel);
    free(mel);
    r.ok = true;
    return r;
}

static StageResult gemma4_encoder_r(gemma4_e2b_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = gemma4_e2b_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "mel failed";
        return r;
    }
    int T_enc = 0, d_model = 0;
    float* enc = gemma4_e2b_run_encoder(ctx, mel, n_mels, T_mel, &T_enc, &d_model);
    free(mel);
    if (!enc) {
        r.note = "gemma4_e2b_run_encoder returned null";
        return r;
    }
    // The Python reference returns encoder_output as (T_enc, d_model) where
    // d_model is the LLM hidden size (1536) — i.e. post audio_embed_proj.
    // Our buffer is laid out [d_model, T_enc] in row-major; the diff
    // harness treats it as a flat float array so the contents must match
    // the reference flat layout. The Python reference dump uses
    // [T_enc, d_model] as a (T,d) matrix; ggml stores [d, T] which is
    // numerically the SAME contiguous bytes if you read it row-major and
    // interpret as (T, d). cos_min is invariant under shape
    // interpretation, so as long as both sides are consistent the
    // comparison is meaningful. Report shape as (T_enc, d_model) to
    // match the Python convention.
    r.shape = {T_enc, d_model};
    r.data.assign(enc, enc + (size_t)T_enc * d_model);
    free(enc);
    r.ok = true;
    return r;
}

} // namespace


static void print_row(const char* name, const crispasr_diff::Report& r, float cos_threshold, const char* extra = "") {
    const char* tag = r.found ? (r.is_pass(cos_threshold) ? "[PASS]" : "[FAIL]") : "[SKIP]";
    std::string shape_str = "[";
    for (size_t i = 0; i < r.shape.size(); i++) {
        shape_str += std::to_string(r.shape[i]);
        if (i + 1 < r.shape.size())
            shape_str += ",";
    }
    shape_str += "]";
    if (!r.found) {
        printf("%s %-22s %s  (reference not in archive)%s%s\n", tag, name, shape_str.c_str(), *extra ? "  " : "",
               extra);
        return;
    }
    printf("%s %-22s shape=%-16s cos_min=%.6f  cos_mean=%.6f  max_abs=%.2e  rms=%.2e%s%s\n", tag, name,
           shape_str.c_str(), r.cos_min, r.cos_mean, r.max_abs, r.rms, *extra ? "  " : "", extra);
}


int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s <backend> <model.gguf> <reference.gguf> <audio.wav>\n"
                "\n"
                "  backend       one of: voxtral, voxtral4b, qwen3, granite, parakeet, canary, cohere, gemma4\n"
                "  model.gguf    crispasr-compatible model weights\n"
                "  reference.gguf  archive produced by tools/dump_reference.py\n"
                "  audio.wav     16 kHz mono WAV\n",
                argv[0]);
        return 1;
    }
    const std::string backend_name = argv[1];
    const std::string model_path = argv[2];
    const std::string ref_path = argv[3];
    const std::string audio_path = argv[4];

    // Load the reference archive.
    crispasr_diff::Ref ref;
    if (!ref.load(ref_path)) {
        return 2;
    }
    const std::string ref_backend = ref.meta("backend");
    if (!ref_backend.empty() && ref_backend != backend_name) {
        fprintf(stderr,
                "crispasr-diff: warning: reference archive was dumped for backend '%s' "
                "but you asked for '%s'\n",
                ref_backend.c_str(), backend_name.c_str());
    }

    // Load audio (any common format, via read_audio_data).
    std::vector<float> samples;
    std::vector<std::vector<float>> stereo;
    if (!read_audio_data(audio_path, samples, stereo, /*stereo=*/false)) {
        fprintf(stderr, "crispasr-diff: failed to read audio '%s'\n", audio_path.c_str());
        return 3;
    }
    printf("crispasr-diff: audio %zu samples (%.2fs), reference %s, backend %s\n", samples.size(),
           samples.size() / 16000.0, ref_path.c_str(), backend_name.c_str());

    const float COS_THRESHOLD = 0.999f;
    int n_pass = 0, n_fail = 0, n_skip = 0;

    auto record = [&](const crispasr_diff::Report& r) {
        if (!r.found) {
            n_skip++;
            return;
        }
        if (r.is_pass(COS_THRESHOLD)) {
            n_pass++;
            return;
        }
        n_fail++;
    };

    // -------- Dispatch to the right backend runner --------
    if (backend_name == "voxtral") {
        auto cp = voxtral_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        voxtral_context* ctx = voxtral_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load voxtral model\n");
            return 4;
        }

        auto mel_r = voxtral_mel(ctx, samples.data(), (int)samples.size());
        if (mel_r.ok) {
            auto rep = ref.compare("mel_spectrogram", mel_r.data.data(), mel_r.data.size());
            print_row("mel_spectrogram", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] mel_spectrogram         %s\n", mel_r.note.c_str());
            n_fail++;
        }

        auto enc_r = voxtral_encoder(ctx, samples.data(), (int)samples.size());
        if (enc_r.ok) {
            // voxtral's run_encoder returns the projector output directly,
            // so compare it against projector_output in the reference.
            auto rep = ref.compare("projector_output", enc_r.data.data(), enc_r.data.size());
            print_row("projector_output", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] projector_output        %s\n", enc_r.note.c_str());
            n_fail++;
        }

        voxtral_free(ctx);
    } else if (backend_name == "voxtral4b") {
        auto cp = voxtral4b_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        voxtral4b_context* ctx = voxtral4b_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load voxtral4b model\n");
            return 4;
        }
        auto mel_r = voxtral4b_mel(ctx, samples.data(), (int)samples.size());
        if (mel_r.ok) {
            auto rep = ref.compare("mel_spectrogram", mel_r.data.data(), mel_r.data.size());
            print_row("mel_spectrogram", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] mel_spectrogram         %s\n", mel_r.note.c_str());
            n_fail++;
        }
        voxtral4b_free(ctx);
    } else if (backend_name == "qwen3") {
        auto cp = qwen3_asr_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        qwen3_asr_context* ctx = qwen3_asr_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load qwen3 model\n");
            return 4;
        }
        auto mel_r = qwen3_mel(ctx, samples.data(), (int)samples.size());
        if (mel_r.ok) {
            auto rep = ref.compare("mel_spectrogram", mel_r.data.data(), mel_r.data.size());
            print_row("mel_spectrogram", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] mel_spectrogram         %s\n", mel_r.note.c_str());
            n_fail++;
        }
        qwen3_asr_free(ctx);
    } else if (backend_name == "granite") {
        auto cp = granite_speech_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        granite_speech_context* ctx = granite_speech_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load granite model\n");
            return 4;
        }
        auto mel_r = granite_mel(ctx, samples.data(), (int)samples.size());
        if (mel_r.ok) {
            auto rep = ref.compare("mel_spectrogram", mel_r.data.data(), mel_r.data.size());
            print_row("mel_spectrogram", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] mel_spectrogram         %s\n", mel_r.note.c_str());
            n_fail++;
        }
        granite_speech_free(ctx);
    } else if (backend_name == "parakeet") {
        auto cp = parakeet_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        parakeet_context* ctx = parakeet_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load parakeet model\n");
            return 4;
        }

        auto mel_r = parakeet_mel_r(ctx, samples.data(), (int)samples.size());
        if (mel_r.ok) {
            auto rep = ref.compare("mel_spectrogram", mel_r.data.data(), mel_r.data.size());
            print_row("mel_spectrogram", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] mel_spectrogram         %s\n", mel_r.note.c_str());
            n_fail++;
        }

        auto enc_r = parakeet_encoder_r(ctx, samples.data(), (int)samples.size());
        if (enc_r.ok) {
            auto rep = ref.compare("encoder_output", enc_r.data.data(), enc_r.data.size());
            print_row("encoder_output", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] encoder_output          %s\n", enc_r.note.c_str());
            n_fail++;
        }

        parakeet_free(ctx);
    } else if (backend_name == "canary") {
        auto cp = canary_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        canary_context* ctx = canary_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load canary model\n");
            return 4;
        }

        auto mel_r = canary_mel_r(ctx, samples.data(), (int)samples.size());
        if (mel_r.ok) {
            auto rep = ref.compare("mel_spectrogram", mel_r.data.data(), mel_r.data.size());
            print_row("mel_spectrogram", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] mel_spectrogram         %s\n", mel_r.note.c_str());
            n_fail++;
        }

        auto enc_r = canary_encoder_r(ctx, samples.data(), (int)samples.size());
        if (enc_r.ok) {
            auto rep = ref.compare("encoder_output", enc_r.data.data(), enc_r.data.size());
            print_row("encoder_output", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] encoder_output          %s\n", enc_r.note.c_str());
            n_fail++;
        }

        canary_free(ctx);
    } else if (backend_name == "cohere") {
        auto cp = cohere_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        cohere_context* ctx = cohere_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load cohere model\n");
            return 4;
        }

        auto mel_r = cohere_mel_r(ctx, samples.data(), (int)samples.size());
        if (mel_r.ok) {
            auto rep = ref.compare("mel_spectrogram", mel_r.data.data(), mel_r.data.size());
            print_row("mel_spectrogram", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] mel_spectrogram         %s\n", mel_r.note.c_str());
            n_fail++;
        }

        auto enc_r = cohere_encoder_r(ctx, samples.data(), (int)samples.size());
        if (enc_r.ok) {
            auto rep = ref.compare("encoder_output", enc_r.data.data(), enc_r.data.size());
            print_row("encoder_output", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] encoder_output          %s\n", enc_r.note.c_str());
            n_fail++;
        }

        cohere_free(ctx);
    } else if (backend_name == "gemma4" || backend_name == "gemma4-e2b") {
        auto cp = gemma4_e2b_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        gemma4_e2b_context* ctx = gemma4_e2b_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load gemma4-e2b model\n");
            return 4;
        }

        auto mel_r = gemma4_mel_r(ctx, samples.data(), (int)samples.size());
        if (mel_r.ok) {
            auto rep = ref.compare("mel_spectrogram", mel_r.data.data(), mel_r.data.size());
            print_row("mel_spectrogram", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] mel_spectrogram         %s\n", mel_r.note.c_str());
            n_fail++;
        }

        auto enc_r = gemma4_encoder_r(ctx, samples.data(), (int)samples.size());
        if (enc_r.ok) {
            auto rep = ref.compare("encoder_output", enc_r.data.data(), enc_r.data.size());
            print_row("encoder_output", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[ERR ] encoder_output          %s\n", enc_r.note.c_str());
            n_fail++;
        }

        gemma4_e2b_free(ctx);
    } else {
        fprintf(stderr,
                "crispasr-diff: backend '%s' is not recognised. "
                "Supported: voxtral, voxtral4b, qwen3, granite, parakeet, canary, cohere, gemma4.\n",
                backend_name.c_str());
        return 5;
    }

    printf("\nsummary: %d pass, %d fail, %d skip (cos threshold %.3f)\n", n_pass, n_fail, n_skip, COS_THRESHOLD);
    return n_fail == 0 ? 0 : 6;
}
