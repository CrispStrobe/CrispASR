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

// MSVC's <cmath> doesn't define M_PI unless _USE_MATH_DEFINES is set
// BEFORE the first include. We use M_PI in the synthetic-audio test
// path (line ~944). MinGW + Linux + macOS toolchains define it
// unconditionally, so this is a Windows-build-only nudge.
#define _USE_MATH_DEFINES
#include <cmath>

#include "crispasr_diff.h"

#include "voxtral.h"
#include "voxtral4b.h"
#include "qwen3_asr.h"
#include "qwen3_tts.h"
#include "kokoro.h"
#include "granite_speech.h"
#include "granite_nle.h"
#include "parakeet.h"
#include "canary.h"
#include "cohere.h"
#include "gemma4_e2b.h"
#include "mimo_asr.h"
#include "mimo_tokenizer.h"
#include "orpheus_snac.h"
#include "chatterbox.h"

#include "common-crispasr.h"

#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sys/stat.h>
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

// ---- granite-nle (granite-speech-4.1-2b-nar) ----

static StageResult granite_nle_mel(granite_nle_context* ctx, const float* samples, int n_samples) {
    StageResult r;
    int n_mels = 0, T_mel = 0;
    float* mel = granite_nle_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        r.note = "granite_nle_compute_mel returned null";
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

// Run the parakeet encoder on the REFERENCE mel rather than our C++ mel.
// This isolates encoder-internal divergence from preprocessor divergence:
// if `encoder_output_ref_mel` cos_mean ≈ 1.0, the residual encoder error
// is pure mel propagation through residuals; if it stays at ~0.8, there's
// a bug inside the FastConformer encoder itself. Reference mel is stored
// as (T_mel, n_mels) row-major (matching parakeet_run_encoder's input
// layout exactly — see tools/reference_backends/parakeet.py).
static StageResult parakeet_encoder_with_ref_mel_r(parakeet_context* ctx, const crispasr_diff::Ref& ref) {
    StageResult r;
    auto pair = ref.get_f32("mel_spectrogram");
    auto shp = ref.shape("mel_spectrogram");
    if (!pair.first || shp.size() < 2) {
        r.note = "reference mel_spectrogram not in archive";
        return r;
    }
    // GGUF ne[0] is the fast axis; the dumper writes (T_mel, n_mels) with
    // n_mels contiguous, so ne = [n_mels, T_mel].
    const int n_mels = (int)shp[0];
    const int T_mel = (int)shp[1];
    int T_enc = 0, d_model = 0;
    float* enc = parakeet_run_encoder(ctx, pair.first, n_mels, T_mel, &T_enc, &d_model);
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

// ---- qwen3-tts (Qwen3 talker, codec_head, code_predictor) ----

static StageResult qwen3_tts_text_proj_r(qwen3_tts_context* ctx, const int32_t* ids, int n_tokens) {
    StageResult r;
    int T = 0, d = 0;
    float* h = qwen3_tts_run_text_proj(ctx, ids, n_tokens, &T, &d);
    if (!h) {
        r.note = "qwen3_tts_run_text_proj returned null";
        return r;
    }
    r.shape = {T, d};
    r.data.assign(h, h + (size_t)T * d);
    free(h);
    r.ok = true;
    return r;
}

static StageResult qwen3_tts_talker_logits_r(qwen3_tts_context* ctx, const float* embeds, int n_tokens) {
    StageResult r;
    int vocab = 0;
    float* logits = qwen3_tts_run_talker_with_embeds(ctx, embeds, n_tokens, &vocab);
    if (!logits) {
        r.note = "qwen3_tts_run_talker_with_embeds returned null";
        return r;
    }
    r.shape = {1, vocab};
    r.data.assign(logits, logits + (size_t)vocab);
    free(logits);
    r.ok = true;
    return r;
}

// ---- chatterbox ----

static StageResult chatterbox_tokens_r(chatterbox_context* ctx, const char* text) {
    StageResult r;
    int n = 0;
    int32_t* tokens = chatterbox_synthesize_tokens(ctx, text, &n);
    if (!tokens) {
        r.note = "chatterbox_synthesize_tokens returned null";
        return r;
    }
    r.shape = {n};
    r.data.resize((size_t)n);
    for (int i = 0; i < n; ++i)
        r.data[(size_t)i] = (float)tokens[i];
    chatterbox_tokens_free(tokens);
    r.ok = true;
    return r;
}

static StageResult chatterbox_mel_r(chatterbox_context* ctx, const char* text) {
    StageResult r;
    int T_mel = 0;
    float* mel_cf = chatterbox_synthesize_mel(ctx, text, &T_mel);
    if (!mel_cf) {
        r.note = "chatterbox_synthesize_mel returned null";
        return r;
    }
    r.shape = {T_mel, 80};
    r.data.resize((size_t)T_mel * 80);
    for (int t = 0; t < T_mel; ++t) {
        for (int c = 0; c < 80; ++c) {
            r.data[(size_t)t * 80 + c] = mel_cf[(size_t)c * T_mel + t];
        }
    }
    free(mel_cf);
    r.ok = true;
    return r;
}

static StageResult chatterbox_mel_from_tokens_r(chatterbox_context* ctx, const int32_t* tokens, int n_tokens) {
    StageResult r;
    int T_mel = 0;
    float* mel_cf = chatterbox_synthesize_mel_from_tokens(ctx, tokens, n_tokens, &T_mel);
    if (!mel_cf) {
        r.note = "chatterbox_synthesize_mel_from_tokens returned null";
        return r;
    }
    r.shape = {T_mel, 80};
    r.data.resize((size_t)T_mel * 80);
    for (int t = 0; t < T_mel; ++t) {
        for (int c = 0; c < 80; ++c) {
            r.data[(size_t)t * 80 + c] = mel_cf[(size_t)c * T_mel + t];
        }
    }
    free(mel_cf);
    r.ok = true;
    return r;
}

static StageResult chatterbox_pcm_r(chatterbox_context* ctx, const char* text) {
    StageResult r;
    int n = 0;
    float* pcm = chatterbox_synthesize(ctx, text, &n);
    if (!pcm) {
        r.note = "chatterbox_synthesize returned null";
        return r;
    }
    r.shape = {n};
    r.data.assign(pcm, pcm + n);
    chatterbox_pcm_free(pcm);
    r.ok = true;
    return r;
}

static StageResult chatterbox_pcm_from_tokens_r(chatterbox_context* ctx, const int32_t* tokens, int n_tokens) {
    StageResult r;
    int n = 0;
    float* pcm = chatterbox_synthesize_from_tokens(ctx, tokens, n_tokens, &n);
    if (!pcm) {
        r.note = "chatterbox_synthesize_from_tokens returned null";
        return r;
    }
    r.shape = {n};
    r.data.assign(pcm, pcm + n);
    chatterbox_pcm_free(pcm);
    r.ok = true;
    return r;
}

static StageResult chatterbox_vocode_mel_with_source_stft_r(chatterbox_context* ctx, const float* mel_cf, int T_mel,
                                                            const float* source_stft_cf, int T_src);

static StageResult chatterbox_vocode_mel_r(chatterbox_context* ctx, const float* mel_cf, int T_mel) {
    return chatterbox_vocode_mel_with_source_stft_r(ctx, mel_cf, T_mel, nullptr, 0);
}

static StageResult chatterbox_vocode_mel_with_source_stft_r(chatterbox_context* ctx, const float* mel_cf, int T_mel,
                                                            const float* source_stft_cf, int T_src) {
    StageResult r;
    int n = 0;
    float* pcm = chatterbox_vocode_mel_with_source_stft(ctx, mel_cf, T_mel, source_stft_cf, T_src, &n);
    if (!pcm) {
        r.note = "chatterbox_vocode_mel_with_source_stft returned null";
        return r;
    }
    r.shape = {n};
    r.data.assign(pcm, pcm + n);
    chatterbox_pcm_free(pcm);
    r.ok = true;
    return r;
}

static StageResult chatterbox_vocode_dump_stage_r(chatterbox_context* ctx, const float* mel_cf, int T_mel,
                                                  const float* source_stft_cf, int T_src, const char* stage_name,
                                                  int row_width) {
    StageResult r;
    int n = 0;
    const char* stage_names[1] = {stage_name};
    float* stage_data[1] = {nullptr};
    int stage_sizes[1] = {0};
    float* pcm = chatterbox_vocode_mel_dump_with_source_stft(ctx, mel_cf, T_mel, source_stft_cf, T_src, &n, stage_names,
                                                             stage_data, stage_sizes, 1);
    if (!pcm) {
        r.note = "chatterbox_vocode_mel_dump_with_source_stft returned null";
        return r;
    }
    chatterbox_pcm_free(pcm);
    if (!stage_data[0] || stage_sizes[0] <= 0) {
        r.note = "requested stage missing from dump";
        return r;
    }
    const int n_rows = row_width > 0 ? (stage_sizes[0] / row_width) : stage_sizes[0];
    if (row_width > 0 && n_rows * row_width != stage_sizes[0]) {
        free(stage_data[0]);
        r.note = "stage size is not divisible by row width";
        return r;
    }
    r.shape = row_width > 0 ? std::vector<int>{n_rows, row_width} : std::vector<int>{stage_sizes[0]};
    if (row_width > 0) {
        r.data.resize((size_t)stage_sizes[0]);
        for (int t = 0; t < n_rows; ++t) {
            for (int c = 0; c < row_width; ++c) {
                r.data[(size_t)t * row_width + (size_t)c] = stage_data[0][(size_t)c * n_rows + (size_t)t];
            }
        }
    } else {
        r.data.assign(stage_data[0], stage_data[0] + stage_sizes[0]);
    }
    free(stage_data[0]);
    r.ok = true;
    return r;
}

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static std::string chatterbox_find_s3gen(const std::string& model_path) {
    const bool turbo_like = model_path.find("turbo") != std::string::npos ||
                            model_path.find("kartoffel") != std::string::npos;
    const char* const* candidates = nullptr;
    static const char* turbo_candidates[] = {
        "chatterbox-turbo-s3gen-f16.gguf",
        nullptr,
    };
    static const char* base_candidates[] = {
        "chatterbox-s3gen-q8_0.gguf",
        "chatterbox-s3gen-f16.gguf",
        nullptr,
    };
    candidates = turbo_like ? turbo_candidates : base_candidates;

    const size_t sep = model_path.find_last_of("/\\");
    const std::string dir = (sep == std::string::npos) ? "." : model_path.substr(0, sep);
    for (const char* const* it = candidates; *it; ++it) {
        const std::string path = dir + "/" + *it;
        if (file_exists(path))
            return path;
    }
    return "";
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

static crispasr_diff::Report compare_with_row_width(const crispasr_diff::Ref& ref, const std::string& name,
                                                    const float* data, size_t n_elem, int row_w) {
    crispasr_diff::Report r;
    auto pair = ref.get_f32(name);
    if (!pair.first || pair.second == 0 || row_w <= 0)
        return r;
    r.found = true;
    r.shape = ref.shape(name);
    const size_t n = std::min(n_elem, pair.second);
    r.n_elem = n;
    if (n == 0)
        return r;
    double sum_abs = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float d = data[i] - pair.first[i];
        const float ad = std::fabs(d);
        if (ad > r.max_abs)
            r.max_abs = ad;
        sum_abs += ad;
        sum_sq += (double)d * (double)d;
    }
    r.mean_abs = (float)(sum_abs / n);
    r.rms = (float)std::sqrt(sum_sq / n);
    const size_t n_rows = n / (size_t)row_w;
    r.cos_min = 1.0f;
    double cos_sum = 0.0;
    size_t cos_rows = 0;
    for (size_t i = 0; i < n_rows; ++i) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int k = 0; k < row_w; ++k) {
            const float a = data[i * (size_t)row_w + (size_t)k];
            const float b = pair.first[i * (size_t)row_w + (size_t)k];
            dot += (double)a * b;
            na += (double)a * a;
            nb += (double)b * b;
        }
        const double denom = std::sqrt(na) * std::sqrt(nb);
        if (denom > 1e-12) {
            const float cs = (float)(dot / denom);
            if (cs < r.cos_min)
                r.cos_min = cs;
            cos_sum += cs;
            cos_rows++;
        }
    }
    if (cos_rows > 0)
        r.cos_mean = (float)(cos_sum / cos_rows);
    return r;
}


int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s <backend> <model.gguf> <reference.gguf> <audio.wav>\n"
                "\n"
                "  backend       one of: voxtral, voxtral4b, qwen3, qwen3-tts, qwen3-tts-codec, kokoro, granite, "
                "granite-4.1, "
                "granite-nle, parakeet, chatterbox, "
                "canary, cohere, gemma4, mimo-tokenizer, mimo-asr, orpheus\n"
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
    } else if (backend_name == "chatterbox") {
        constexpr float CHATTERBOX_MEAN_THRESHOLD = 0.95f;
        auto print_row_mean = [&](const char* name, const crispasr_diff::Report& r, float cos_threshold,
                                  const char* extra = "") {
            const bool pass = r.found && r.cos_mean >= cos_threshold;
            const char* tag = r.found ? (pass ? "[PASS]" : "[FAIL]") : "[SKIP]";
            std::string shape_str = "[";
            for (size_t i = 0; i < r.shape.size(); i++) {
                shape_str += std::to_string(r.shape[i]);
                if (i + 1 < r.shape.size())
                    shape_str += ",";
            }
            shape_str += "]";
            if (!r.found) {
                printf("%s %-22s %s  (reference not in archive)%s%s\n", tag, name, shape_str.c_str(),
                       *extra ? "  " : "", extra);
                return pass;
            }
            printf("%s %-22s shape=%-16s cos_min=%.6f  cos_mean=%.6f  max_abs=%.2e  rms=%.2e%s%s\n", tag, name,
                   shape_str.c_str(), r.cos_min, r.cos_mean, r.max_abs, r.rms, *extra ? "  " : "", extra);
            return pass;
        };
        auto record_mean = [&](const crispasr_diff::Report& r, float cos_threshold) {
            if (!r.found) {
                n_skip++;
            } else if (r.cos_mean >= cos_threshold) {
                n_pass++;
            } else {
                n_fail++;
            }
        };

        auto cp = chatterbox_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        cp.use_gpu = false;
        chatterbox_context* ctx = chatterbox_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load chatterbox model\n");
            return 4;
        }
        const std::string s3gen_path = chatterbox_find_s3gen(model_path);
        if (s3gen_path.empty() || chatterbox_set_s3gen_path(ctx, s3gen_path.c_str()) != 0) {
            fprintf(stderr, "failed to load chatterbox companion S3Gen model\n");
            chatterbox_free(ctx);
            return 4;
        }
        auto ref_tok_pair = ref.get_f32("t3_speech_tokens");
        if (!ref_tok_pair.first || ref_tok_pair.second == 0) {
            printf("[SKIP] t3_speech_tokens       exact upstream T3 path is stochastic; replaying downstream stages "
                   "from reference tokens requires t3_speech_tokens in the archive\n");
            n_skip++;
        } else {
            std::vector<int32_t> ref_tokens(ref_tok_pair.second);
            for (size_t i = 0; i < ref_tok_pair.second; ++i)
                ref_tokens[i] = (int32_t)std::lrint(ref_tok_pair.first[i]);

            printf("[SKIP] t3_speech_tokens       exact upstream T3 path is stochastic; comparing S3Gen/HiFT using "
                   "reference tokens from the official path\n");
            n_skip++;

            auto mel_r = chatterbox_mel_from_tokens_r(ctx, ref_tokens.data(), (int)ref_tokens.size());
            if (mel_r.ok) {
                auto rep = compare_with_row_width(ref, "s3gen_mel", mel_r.data.data(), mel_r.data.size(), 80);
                print_row_mean("s3gen_mel", rep, CHATTERBOX_MEAN_THRESHOLD, "criterion=cos_mean>=0.95");
                record_mean(rep, CHATTERBOX_MEAN_THRESHOLD);
            } else {
                printf("[ERR ] s3gen_mel              %s\n", mel_r.note.c_str());
                n_fail++;
            }

            printf("[SKIP] hift_pcm               compounded token->mel + mel->wave drift; use s3gen_mel and "
                   "hift_pcm(ref_mel) for apples-to-apples parity\n");
            n_skip++;

            auto ref_mel_pair = ref.get_f32("s3gen_mel");
            auto ref_mel_shape = ref.shape("s3gen_mel");
            if (ref_mel_pair.first && ref_mel_shape.size() >= 2) {
                const int T_mel = (int)ref_mel_shape[1];
                const int C_mel = (int)ref_mel_shape[0];
                if (C_mel == 80) {
                    std::vector<float> mel_cf((size_t)T_mel * 80);
                    for (int t = 0; t < T_mel; ++t) {
                        for (int c = 0; c < 80; ++c) {
                            mel_cf[(size_t)c * T_mel + (size_t)t] = ref_mel_pair.first[(size_t)t * 80 + (size_t)c];
                        }
                    }

                    const float* ref_source_stft = nullptr;
                    int T_src = 0;
                    auto ref_source_pair = ref.get_f32("hift_source_stft");
                    auto ref_source_shape = ref.shape("hift_source_stft");
                    if (ref_source_pair.first && ref_source_shape.size() >= 2 && (int)ref_source_shape[0] == 18) {
                        ref_source_stft = ref_source_pair.first;
                        T_src = (int)ref_source_shape[1];
                    }

                    if (ref_source_stft && T_src > 0) {
                        struct VocStage {
                            const char* name;
                            int row_width;
                        };
                        static const VocStage voc_stages[] = {
                            {"voc_conv_pre", 512},
                            {"voc_ups_0", 256},
                            {"voc_rb_0", 256},
                            {"voc_ups_1", 128},
                            {"voc_rb_1", 128},
                            {"voc_ups_2", 64},
                            {"voc_rb_2", 64},
                            {"voc_conv_post", 18},
                        };
                        for (const auto& s : voc_stages) {
                            if (ref.shape(s.name).empty()) {
                                printf("[SKIP] %-20s missing from reference archive\n", s.name);
                                n_skip++;
                                continue;
                            }
                            auto stage_r = chatterbox_vocode_dump_stage_r(ctx, mel_cf.data(), T_mel, ref_source_stft,
                                                                          T_src, s.name, s.row_width);
                            if (!stage_r.ok) {
                                printf("[ERR ] %-20s %s\n", s.name, stage_r.note.c_str());
                                n_fail++;
                                continue;
                            }
                            auto rep = compare_with_row_width(ref, s.name, stage_r.data.data(), stage_r.data.size(),
                                                              s.row_width);
                            print_row_mean(s.name, rep, CHATTERBOX_MEAN_THRESHOLD, "criterion=cos_mean>=0.95");
                            record_mean(rep, CHATTERBOX_MEAN_THRESHOLD);
                        }
                    } else {
                        printf("[SKIP] hift_source_stft      not in reference archive; re-dump with latest "
                               "tools/reference_backends/chatterbox.py\n");
                        n_skip++;
                    }

                    auto voc_r = chatterbox_vocode_mel_with_source_stft_r(ctx, mel_cf.data(), T_mel, ref_source_stft,
                                                                          T_src);
                    if (voc_r.ok) {
                        auto rep = ref.compare("hift_pcm", voc_r.data.data(), voc_r.data.size());
                        print_row_mean("hift_pcm(ref_mel)", rep, CHATTERBOX_MEAN_THRESHOLD,
                                       "criterion=cos_mean>=0.95");
                        record_mean(rep, CHATTERBOX_MEAN_THRESHOLD);
                    } else {
                        printf("[ERR ] hift_pcm(ref_mel)      %s\n", voc_r.note.c_str());
                        n_fail++;
                    }
                }
            }
        }
        chatterbox_free(ctx);
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
    } else if (backend_name == "qwen3-tts") {
        // TTS backend: the 4th positional arg is the reference WAV used
        // for voice-clone prompt building. The C++ side doesn't consume
        // the audio directly — input ids come from the reference
        // archive's `text_input_ids` tensor (deterministic, written by
        // tools/reference_backends/qwen3_tts.py).
        auto cp = qwen3_tts_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        qwen3_tts_context* ctx = qwen3_tts_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load qwen3-tts model\n");
            return 4;
        }
        const char* codec_gguf = std::getenv("QWEN3_TTS_CODEC_GGUF");
        if (codec_gguf && *codec_gguf) {
            if (qwen3_tts_set_codec_path(ctx, codec_gguf) != 0) {
                fprintf(stderr, "failed to load qwen3-tts codec '%s'\n", codec_gguf);
                qwen3_tts_free(ctx);
                return 4;
            }
        }

        // Stage: text_proj_out — text_embedding + text_projection on the
        // tokenised synth prompt. Read the int32 ids from the reference
        // archive (written as F32 by the dumper for GGUF compatibility)
        // and feed them through qwen3_tts_run_text_proj.
        auto ids_pair = ref.get_f32("text_input_ids");
        if (!ids_pair.first) {
            printf("[ERR ] text_proj_out            text_input_ids missing from reference\n");
            n_fail++;
        } else {
            std::vector<int32_t> ids(ids_pair.second);
            for (size_t i = 0; i < ids_pair.second; i++)
                ids[i] = (int32_t)ids_pair.first[i];
            auto tp_r = qwen3_tts_text_proj_r(ctx, ids.data(), (int)ids.size());
            if (tp_r.ok) {
                auto rep = ref.compare("text_proj_out", tp_r.data.data(), tp_r.data.size());
                print_row("text_proj_out", rep, COS_THRESHOLD);
                record(rep);
            } else {
                printf("[ERR ] text_proj_out            %s\n", tp_r.note.c_str());
                n_fail++;
            }
        }

        // Stage: talker_logits — read the PyTorch-built ICL prefill
        // embedding (talker_inputs_embeds) directly from the reference
        // archive, run our talker graph on it, and compare the
        // codec_head logits at position[-1] against the reference's
        // talker_logits[-1]. This isolates "talker graph correctness"
        // from "prefill builder correctness" — perfect cosine here
        // means the 28L Qwen3 forward + Q/K-norm + flash_attn + RoPE
        // + SwiGLU all match PyTorch given identical inputs, even
        // before our own ICL prefill builder is wired up.
        auto embeds_pair = ref.get_f32("talker_inputs_embeds");
        auto embeds_shape = ref.shape("talker_inputs_embeds");
        // GGUF ne[] is reverse of numpy: tensor saved as numpy (T, d) has
        // ne[0]=d, ne[1]=T. We want (T, d) here.
        if (!embeds_pair.first || embeds_shape.size() < 2) {
            printf("[SKIP] talker_logits           talker_inputs_embeds not in reference (re-dump with that stage)\n");
            n_skip++;
        } else {
            const int T = (int)embeds_shape[1];
            const int d = (int)embeds_shape[0];
            (void)d;
            auto tl_r = qwen3_tts_talker_logits_r(ctx, embeds_pair.first, T);
            if (tl_r.ok) {
                // Reference talker_logits is numpy (1, T, vocab); GGUF
                // stores ne=[vocab, T, 1]. Compare position[-1] of the
                // T axis against our (vocab,) output at position[-1].
                auto ref_logits_pair = ref.get_f32("talker_logits");
                auto ref_logits_shape = ref.shape("talker_logits");
                if (!ref_logits_pair.first || ref_logits_shape.size() < 2) {
                    printf("[SKIP] talker_logits           talker_logits ref tensor missing/wrong shape\n");
                    n_skip++;
                } else {
                    const int vocab = (int)ref_logits_shape[0];
                    const int Tref = (int)ref_logits_shape[1];
                    const float* ref_last = ref_logits_pair.first + (size_t)(Tref - 1) * vocab;
                    // ref.compare expects the named tensor to match buffer length;
                    // since we've already loaded talker_logits, just compute the
                    // metrics inline against the last row.
                    crispasr_diff::Report rep;
                    rep.found = true;
                    rep.n_elem = (size_t)vocab;
                    rep.shape = {1, vocab};
                    double dot = 0, na = 0, nb = 0, max_abs = 0, sum_abs = 0, sum_sq = 0;
                    for (int i = 0; i < vocab; i++) {
                        double a = tl_r.data[i], b = ref_last[i];
                        dot += a * b;
                        na += a * a;
                        nb += b * b;
                        double d = a - b;
                        if (std::fabs(d) > max_abs)
                            max_abs = std::fabs(d);
                        sum_abs += std::fabs(d);
                        sum_sq += d * d;
                    }
                    rep.cos_min = (na > 0 && nb > 0) ? (float)(dot / std::sqrt(na * nb)) : 1.0f;
                    rep.cos_mean = rep.cos_min;
                    rep.max_abs = (float)max_abs;
                    rep.mean_abs = (float)(sum_abs / vocab);
                    rep.rms = (float)std::sqrt(sum_sq / vocab);
                    print_row("talker_logits", rep, COS_THRESHOLD);
                    record(rep);
                }
            } else {
                printf("[ERR ] talker_logits           %s\n", tl_r.note.c_str());
                n_fail++;
            }
        }
        // Stage: talker_logits_via_icl_prefill — the full self-test.
        // Build the ICL prefill on the C++ side (text_embed + text_proj
        // for the chat template + codec sentinels + speaker_embed (from
        // baked voice pack) + per-frame summed codec embeddings of
        // ref_code), feed it through the talker, compare codec_head[-1]
        // against PyTorch's talker_logits[-1]. If our prefill builder
        // matches, this passes at cos_min=1.000000 just like the prior
        // stage that consumed PyTorch's prefill verbatim.
        const std::string syn_text = ref.meta("qwen3_tts_syn_text");
        const std::string ref_text = ref.meta("qwen3_tts_ref_text");
        if (syn_text.empty() || ref_text.empty()) {
            printf("[SKIP] talker_logits_via_icl  qwen3_tts_syn_text / qwen3_tts_ref_text not in reference (set "
                   "env vars at dump time)\n");
            n_skip++;
        } else {
            // Need a voice pack — load the canonical clone pack.
            int rc = qwen3_tts_load_voice_pack(ctx, "/tmp/qwen3-tts-voice-pack.gguf");
            if (rc != 0) {
                printf("[SKIP] talker_logits_via_icl  voice pack not loaded (run bake-qwen3-tts-voice-pack first)\n");
                n_skip++;
            } else {
                int Tprefill = 0;
                float* my_prefill = qwen3_tts_build_icl_prefill(ctx, syn_text.c_str(), ref_text.c_str(), &Tprefill);
                if (!my_prefill) {
                    printf("[ERR ] talker_logits_via_icl  build_icl_prefill returned null\n");
                    n_fail++;
                } else {
                    // Sanity: compare our prefill to the reference prefill before
                    // running the talker. If they differ here, the bug is in
                    // build_icl_prefill; if they match here, the bug is in
                    // talker invocation state.
                    if (embeds_pair.first && (int)embeds_shape[1] == Tprefill) {
                        const int dd = (int)embeds_shape[0];
                        double max_pre = 0;
                        for (size_t i = 0; i < (size_t)Tprefill * dd; i++) {
                            double diff = std::fabs(my_prefill[i] - embeds_pair.first[i]);
                            if (diff > max_pre)
                                max_pre = diff;
                        }
                        printf("[INFO] icl_prefill_vs_ref     max_abs=%.4e (over T=%d × d=%d)\n", max_pre, Tprefill,
                               dd);
                    }
                    auto tl_r = qwen3_tts_talker_logits_r(ctx, my_prefill, Tprefill);
                    free(my_prefill);
                    if (tl_r.ok) {
                        auto ref_logits_pair = ref.get_f32("talker_logits");
                        auto ref_logits_shape = ref.shape("talker_logits");
                        if (!ref_logits_pair.first || ref_logits_shape.size() < 2) {
                            printf("[SKIP] talker_logits_via_icl  ref talker_logits missing\n");
                            n_skip++;
                        } else {
                            const int vocab = (int)ref_logits_shape[0];
                            const int Tref = (int)ref_logits_shape[1];
                            const float* ref_last = ref_logits_pair.first + (size_t)(Tref - 1) * vocab;
                            crispasr_diff::Report rep;
                            rep.found = true;
                            rep.n_elem = (size_t)vocab;
                            rep.shape = {1, vocab};
                            double dot = 0, na = 0, nb = 0, max_abs = 0, sum_abs = 0, sum_sq = 0;
                            for (int i = 0; i < vocab; i++) {
                                double a = tl_r.data[i], b = ref_last[i];
                                dot += a * b;
                                na += a * a;
                                nb += b * b;
                                double d = a - b;
                                if (std::fabs(d) > max_abs)
                                    max_abs = std::fabs(d);
                                sum_abs += std::fabs(d);
                                sum_sq += d * d;
                            }
                            rep.cos_min = (na > 0 && nb > 0) ? (float)(dot / std::sqrt(na * nb)) : 1.0f;
                            rep.cos_mean = rep.cos_min;
                            rep.max_abs = (float)max_abs;
                            rep.mean_abs = (float)(sum_abs / vocab);
                            rep.rms = (float)std::sqrt(sum_sq / vocab);
                            char extra[64];
                            snprintf(extra, sizeof(extra), "T_prefill=%d  argmax=%d", Tprefill,
                                     (int)(std::max_element(tl_r.data.begin(), tl_r.data.end()) - tl_r.data.begin()));
                            print_row("talker_logits_via_icl", rep, COS_THRESHOLD, extra);
                            record(rep);
                        }
                    } else {
                        printf("[ERR ] talker_logits_via_icl  %s\n", tl_r.note.c_str());
                        n_fail++;
                    }
                }
            }
        }

        // Stage: runtime_voice_prompt — run the real WAV-prompt path
        // used by the CLI (`set_voice_prompt_with_text`) and compare
        // the resulting runtime ref_code / prefill / talker logits
        // against the official prompt item dumped in the reference.
        if (syn_text.empty() || ref_text.empty()) {
            printf("[SKIP] runtime_voice_prompt    qwen3_tts_syn_text / qwen3_tts_ref_text not in reference\n");
            n_skip++;
        } else {
            int rc = qwen3_tts_set_voice_prompt_with_text(ctx, audio_path.c_str(), ref_text.c_str());
            if (rc != 0) {
                printf("[SKIP] runtime_voice_prompt    set_voice_prompt_with_text failed on '%s'\n",
                       audio_path.c_str());
                n_skip++;
            } else {
                auto ref_codes_pair = ref.get_f32("ref_codes");
                auto ref_codes_shape = ref.shape("ref_codes");
                auto ref_spk_pair = ref.get_f32("ref_spk_embedding");
                int n_spk = 0;
                const float* my_spk = qwen3_tts_get_runtime_spk_emb(ctx, &n_spk);
                if (ref_spk_pair.first && my_spk && n_spk > 0) {
                    auto rep = ref.compare("ref_spk_embedding", my_spk, (size_t)n_spk);
                    print_row("runtime_spk_emb", rep, COS_THRESHOLD);
                    record(rep);
                } else {
                    printf(
                        "[SKIP] runtime_spk_emb       ref_spk_embedding missing or runtime speaker emb unavailable\n");
                    n_skip++;
                }
                int n_codes = 0;
                const int32_t* my_codes = qwen3_tts_get_runtime_ref_codes(ctx, &n_codes);
                if (ref_codes_pair.first && ref_codes_shape.size() >= 2 && my_codes) {
                    const int n_ref = (int)(ref_codes_shape[0] * ref_codes_shape[1]);
                    std::vector<float> my_codes_f((size_t)n_codes);
                    for (int i = 0; i < n_codes; i++)
                        my_codes_f[i] = (float)my_codes[i];
                    auto rep = ref.compare("ref_codes", my_codes_f.data(), (size_t)std::min(n_codes, n_ref));
                    print_row("runtime_ref_codes", rep, COS_THRESHOLD);
                    record(rep);
                } else {
                    printf("[SKIP] runtime_ref_codes     ref_codes missing or runtime codes unavailable\n");
                    n_skip++;
                }

                int Tprefill_rt = 0;
                float* rt_prefill = qwen3_tts_build_icl_prefill(ctx, syn_text.c_str(), ref_text.c_str(), &Tprefill_rt);
                if (!rt_prefill) {
                    printf("[ERR ] runtime_icl_prefill    build_icl_prefill returned null\n");
                    n_fail++;
                } else {
                    if (embeds_pair.first && (int)embeds_shape[1] == Tprefill_rt) {
                        const int dd = (int)embeds_shape[0];
                        double max_pre = 0;
                        for (size_t i = 0; i < (size_t)Tprefill_rt * dd; i++) {
                            double diff = std::fabs(rt_prefill[i] - embeds_pair.first[i]);
                            if (diff > max_pre)
                                max_pre = diff;
                        }
                        printf("[INFO] runtime_icl_vs_ref     max_abs=%.4e (over T=%d × d=%d)\n", max_pre, Tprefill_rt,
                               dd);
                    }
                    auto tl_r = qwen3_tts_talker_logits_r(ctx, rt_prefill, Tprefill_rt);
                    free(rt_prefill);
                    if (tl_r.ok) {
                        auto ref_logits_pair = ref.get_f32("talker_logits");
                        auto ref_logits_shape = ref.shape("talker_logits");
                        if (!ref_logits_pair.first || ref_logits_shape.size() < 2) {
                            printf("[SKIP] runtime_talker_logits  ref talker_logits missing\n");
                            n_skip++;
                        } else {
                            const int vocab = (int)ref_logits_shape[0];
                            const int Tref = (int)ref_logits_shape[1];
                            const float* ref_last = ref_logits_pair.first + (size_t)(Tref - 1) * vocab;
                            crispasr_diff::Report rep;
                            rep.found = true;
                            rep.n_elem = (size_t)vocab;
                            rep.shape = {1, vocab};
                            double dot = 0, na = 0, nb = 0, max_abs = 0, sum_abs = 0, sum_sq = 0;
                            for (int i = 0; i < vocab; i++) {
                                double a = tl_r.data[i], b = ref_last[i];
                                dot += a * b;
                                na += a * a;
                                nb += b * b;
                                double d = a - b;
                                if (std::fabs(d) > max_abs)
                                    max_abs = std::fabs(d);
                                sum_abs += std::fabs(d);
                                sum_sq += d * d;
                            }
                            rep.cos_min = (na > 0 && nb > 0) ? (float)(dot / std::sqrt(na * nb)) : 1.0f;
                            rep.cos_mean = rep.cos_min;
                            rep.max_abs = (float)max_abs;
                            rep.mean_abs = (float)(sum_abs / vocab);
                            rep.rms = (float)std::sqrt(sum_sq / vocab);
                            char extra[64];
                            snprintf(extra, sizeof(extra), "T_prefill=%d  argmax=%d", Tprefill_rt,
                                     (int)(std::max_element(tl_r.data.begin(), tl_r.data.end()) - tl_r.data.begin()));
                            print_row("runtime_talker_logits", rep, COS_THRESHOLD, extra);
                            record(rep);
                        }
                    } else {
                        printf("[ERR ] runtime_talker_logits  %s\n", tl_r.note.c_str());
                        n_fail++;
                    }
                }
            }
        }

        // ---- cp_step{0..14}: per-step code-predictor diff ----
        //
        // Drives the 15-step AR loop on the C++ side using PyTorch-dumped
        // input embeds at each step, so any divergence between paths
        // (default / O15 / FUSED_QKV / ...) is localised to the exact step
        // it first appears at instead of being smeared across the prefill.
        //
        // Schedule mirrors run_code_pred_kv inside code_pred_generate_15:
        //   step 0        : T=2, n_past=0,    lm_head_idx=0
        //   step k (1..14): T=1, n_past=k+1,  lm_head_idx=k
        //
        // The cp_kv cache state persists across calls — so we MUST run them
        // in order from step 0. The cache was zero-initialised at init and
        // none of the prior stages above touch cp_kv, so step 0 starts on
        // a clean slate.
        for (int k = 0; k < 15; k++) {
            char in_name[32], out_name[32];
            snprintf(in_name, sizeof(in_name), "cp_step%d_input_embed", k);
            snprintf(out_name, sizeof(out_name), "cp_step%d_logits", k);
            char stage_label[32];
            snprintf(stage_label, sizeof(stage_label), "cp_step%d", k);

            auto in_pair = ref.get_f32(in_name);
            if (!in_pair.first) {
                printf("[SKIP] %-15s %s missing (re-dump ref with cp_step stages)\n", stage_label, in_name);
                n_skip++;
                break; // subsequent steps depend on this step's cp_kv state
            }
            const int T_in = (k == 0) ? 2 : 1;
            const int n_past = (k == 0) ? 0 : (k + 1);
            int vocab = 0;
            float* logits = qwen3_tts_run_code_pred_step(ctx, in_pair.first, T_in, n_past, /*lm_head_idx=*/k, &vocab);
            if (!logits || vocab <= 0) {
                printf("[ERR ] %-15s qwen3_tts_run_code_pred_step returned null\n", stage_label);
                n_fail++;
                break;
            }
            auto rep = ref.compare(out_name, logits, (size_t)vocab);
            print_row(stage_label, rep, COS_THRESHOLD);
            record(rep);
            free(logits);
        }

        qwen3_tts_free(ctx);

        // ---- qwen3-tts-spk (ECAPA speaker encoder) ----
        // Uses the same 440 Hz sine wave as the Python reference backend.
    } else if (backend_name == "qwen3-tts-spk") {
        auto cp = qwen3_tts_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        cp.use_gpu = false;
        qwen3_tts_context* ctx = qwen3_tts_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load model\n");
            return 4;
        }

        // Generate the same 440 Hz sine the Python backend uses
        const int sr = 24000, n = sr * 3;
        std::vector<float> audio(n);
        for (int i = 0; i < n; i++)
            audio[i] = 0.5f * std::sin(2.0f * (float)M_PI * 440.0f * i / sr);

        // Stage 1: mel spectrogram
        int T_mel = 0, n_mels = 0;
        float* mel = qwen3_tts_compute_speaker_mel(ctx, audio.data(), n, &T_mel, &n_mels);
        if (!mel) {
            printf("[ERR ] spk_mel  mel computation failed\n");
            n_fail++;
        } else {
            // Python stores (T, 128) time-first, C++ also (T, 128) — compare flat
            auto rep = ref.compare("spk_mel", mel, (size_t)T_mel * n_mels);
            print_row("spk_mel", rep, COS_THRESHOLD);
            record(rep);
            free(mel);
        }

        // Stage 2a: ECAPA on the EXACT Python mel (isolates ECAPA network error).
        // Read spk_mel from the reference archive (Python-computed).
        {
            auto mel_pair = ref.get_f32("spk_mel");
            auto mel_shape = ref.shape("spk_mel");
            if (mel_pair.first && mel_shape.size() >= 2) {
                // GGUF ne=[128, T] = (C, T) in ggml. We need (T, 128) row-major for run_speaker_enc_on_mel.
                // mel_pair.first is flat in (C, T) ggml order: element [c,t] at c + t*128.
                // run_spk_enc expects (T, 128) row-major: element [t,c] at t*128 + c.
                const int C = (int)mel_shape[0]; // 128
                const int T = (int)mel_shape[1]; // T_mel
                std::vector<float> mel_TC((size_t)T * C);
                for (int t = 0; t < T; t++)
                    for (int c = 0; c < C; c++)
                        mel_TC[(size_t)t * C + c] = mel_pair.first[c + (size_t)t * C];
                int dim2 = 0;
                float* emb2 = qwen3_tts_run_speaker_enc_on_mel(ctx, mel_TC.data(), T, &dim2);
                if (emb2) {
                    auto rep = ref.compare("spk_emb", emb2, (size_t)dim2);
                    print_row("spk_emb(ref_mel)", rep, COS_THRESHOLD, "  ECAPA-only");
                    record(rep);
                    free(emb2);
                }
            }
        }

        // Stage 2b: full embedding (C++ mel → ECAPA)
        int dim = 0;
        float* emb = qwen3_tts_compute_speaker_embedding(ctx, audio.data(), n, &dim);
        if (!emb) {
            fprintf(stderr, "[ERR ] spk_emb  ECAPA forward failed\n");
            n_fail++;
        } else {
            auto rep = ref.compare("spk_emb", emb, (size_t)dim);
            print_row("spk_emb(cpp_mel)", rep, COS_THRESHOLD, "  full pipeline");
            record(rep);
            free(emb);
        }
        qwen3_tts_free(ctx);

        // ---- qwen3-tts-cenc (codec ENCODER: audio → codes) ----
        // Uses the same fixed 3s slice of clone.wav as the Python reference.
    } else if (backend_name == "qwen3-tts-cenc") {
        const char* codec_gguf = std::getenv("QWEN3_TTS_CODEC_GGUF");
        if (!codec_gguf) {
            fprintf(stderr, "qwen3-tts-cenc: set QWEN3_TTS_CODEC_GGUF=<codec.gguf>\n");
            return 4;
        }
        auto cp = qwen3_tts_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        cp.use_gpu = false;
        qwen3_tts_context* qctx = qwen3_tts_init_from_file(model_path.c_str(), cp);
        if (!qctx) {
            fprintf(stderr, "failed to load talker\n");
            return 4;
        }
        if (qwen3_tts_set_codec_path(qctx, codec_gguf) != 0) {
            fprintf(stderr, "failed to load codec\n");
            qwen3_tts_free(qctx);
            return 4;
        }

        // Read input audio from the reference (cenc_input_audio is the fixed 3s slice)
        auto audio_pair = ref.get_f32("cenc_input_audio");
        if (!audio_pair.first) {
            printf("[ERR ] cenc_input_audio not in reference\n");
            qwen3_tts_free(qctx);
            return 5;
        }
        const int n_samp = (int)audio_pair.second;
        std::vector<float> audio_buf(audio_pair.first, audio_pair.first + n_samp);

        // Compare each stage (intra-SEANet first to localize drift)
        static const char* stages[] = {
            "cenc_se_init", "cenc_se_s0",      "cenc_se_s1",    "cenc_se_s2",
            "cenc_se_s3",   "cenc_seanet_out", "cenc_xfmr_out", "cenc_ds_out",
        };
        for (const char* s : stages) {
            int n = 0;
            float* mine = qwen3_tts_cenc_extract_stage(qctx, audio_buf.data(), n_samp, s, &n);
            if (!mine) {
                printf("[ERR ] %-22s extract returned null\n", s);
                n_fail++;
                continue;
            }
            auto rep = ref.compare(s, mine, (size_t)n);
            print_row(s, rep, COS_THRESHOLD);
            record(rep);
            free(mine);
        }

        // Final codes from the raw-audio buffer path. This isolates the RVQ
        // encoder math from the WAV reader used by set_voice_prompt().
        {
            int n = 0;
            float* mine = qwen3_tts_cenc_extract_stage(qctx, audio_buf.data(), n_samp, "cenc_codes", &n);
            if (!mine) {
                printf("[ERR ] cenc_codes(raw)         extract returned null\n");
                n_fail++;
            } else {
                auto rep = ref.compare("cenc_codes", mine, (size_t)n);
                print_row("cenc_codes(raw)", rep, COS_THRESHOLD);
                record(rep);
                free(mine);
            }
        }

        // Final codes — use the real WAV-path runtime prompt builder so we
        // compare the exact row-major [T,16] codes the CLI path will later
        // feed into ICL prefill.
        {
            int rc = qwen3_tts_set_voice_prompt(qctx, audio_path.c_str());
            if (rc != 0) {
                printf("[ERR ] cenc_codes(wav)         set_voice_prompt failed on '%s'\n", audio_path.c_str());
                n_fail++;
            } else {
                auto ref_codes_pair = ref.get_f32("cenc_codes");
                auto ref_codes_shape = ref.shape("cenc_codes");
                int n_codes = 0;
                const int32_t* my_codes = qwen3_tts_get_runtime_ref_codes(qctx, &n_codes);
                if (!ref_codes_pair.first || ref_codes_shape.size() < 2 || !my_codes) {
                    printf("[SKIP] cenc_codes(wav)         ref/runtime codes unavailable\n");
                    n_skip++;
                } else {
                    const int T_ref = (int)ref_codes_shape[1];
                    const int Q_ref = (int)ref_codes_shape[0];
                    const int n_ref = T_ref * Q_ref;
                    std::vector<float> my_codes_f((size_t)n_codes);
                    for (int i = 0; i < n_codes; i++)
                        my_codes_f[i] = (float)my_codes[i];
                    auto rep = ref.compare("cenc_codes", my_codes_f.data(), (size_t)std::min(n_codes, n_ref));
                    print_row("cenc_codes(wav)", rep, COS_THRESHOLD);
                    record(rep);
                }
            }
        }

        qwen3_tts_free(qctx);
        // Runs the codec decoder on T=10 all-zero codes and compares
        // each named intermediate tensor against the Python reference dump.
    } else if (backend_name == "qwen3-tts-codec") {
        const char* codec_gguf = std::getenv("QWEN3_TTS_CODEC_GGUF");
        if (!codec_gguf) {
            fprintf(stderr, "qwen3-tts-codec: set QWEN3_TTS_CODEC_GGUF=<path/to/codec.gguf>\n");
            return 4;
        }
        auto cp = qwen3_tts_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        cp.use_gpu = true; // codec is pinned to CPU internally via codec_sched
        qwen3_tts_context* ctx = qwen3_tts_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load qwen3-tts model '%s'\n", model_path.c_str());
            return 4;
        }
        if (qwen3_tts_set_codec_path(ctx, codec_gguf) != 0) {
            fprintf(stderr, "failed to load codec from '%s'\n", codec_gguf);
            qwen3_tts_free(ctx);
            return 4;
        }

        // Build the same deterministic codes the Python dump used:
        // T=10 frames × 16 codebooks, all code_val (default 0).
        const int T_codec = 10, N_Q = 16;
        const int code_val = 0;
        std::vector<int32_t> codes(T_codec * N_Q, code_val); // [T, n_q] row-major

        // Stage list matches DEFAULT_STAGES in qwen3_tts_codec.py
        static const char* codec_stages[] = {
            "codec_rvq_out", "codec_pre_conv_out", "codec_xfmr_out", "codec_up0_out",
            "codec_up1_out", "codec_in_conv_out",  "codec_blk0_out",
            "pcm", // full PCM — matches "codec_pcm" in Python (renamed to "pcm" internally)
        };
        // The Python dump names the final output "codec_pcm", but internally it's "pcm".
        // We compare "pcm" (C++ graph name) against "codec_pcm" (Python dump name).
        const char* ref_name_override[] = {
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "codec_pcm",
        };
        static_assert(sizeof(codec_stages) / sizeof(*codec_stages) == 8, "");

        for (int si = 0; si < 8; si++) {
            const char* stage = codec_stages[si];
            const char* ref_name = ref_name_override[si] ? ref_name_override[si] : stage;

            int n_stage = 0;
            float* our_data = qwen3_tts_codec_extract_stage(ctx, codes.data(), T_codec * N_Q, stage, &n_stage);
            if (!our_data) {
                printf("[ERR ] %-22s  extract returned null\n", ref_name);
                n_fail++;
                continue;
            }

            auto rep = ref.compare(ref_name, our_data, (size_t)n_stage);
            print_row(ref_name, rep, COS_THRESHOLD);
            record(rep);
            free(our_data);
        }
        qwen3_tts_free(ctx);

    } else if (backend_name == "mimo-tokenizer") {
        auto cp = mimo_tokenizer_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        // CPU-pin until the Metal graph-build path on the forward conv stem is
        // verified safe. The qwen3-tts kernel_conv_transpose_1d watchdog hang
        // shape is comparable to MiMo's conv2 / down_sample. Opt in with
        // MIMO_TOKENIZER_GPU=1.
        cp.use_gpu = std::getenv("MIMO_TOKENIZER_GPU") != nullptr;
        mimo_tokenizer_context* ctx = mimo_tokenizer_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load mimo-tokenizer model '%s'\n", model_path.c_str());
            return 4;
        }
        // Stage list matches DEFAULT_STAGES in tools/reference_backends/mimo_tokenizer.py
        static const char* stages[] = {
            "tok_mel", "tok_conv1_out", "tok_conv2_out", "tok_xfmr_out", "tok_pool_out", "tok_codes",
        };
        for (const char* stage : stages) {
            int n_stage = 0;
            float* our_data = mimo_tokenizer_extract_stage(ctx, samples.data(), (int)samples.size(), stage, &n_stage);
            if (!our_data) {
                printf("[ERR ] %-22s  extract returned null\n", stage);
                n_fail++;
                continue;
            }
            auto rep = ref.compare(stage, our_data, (size_t)n_stage);
            print_row(stage, rep, COS_THRESHOLD);
            record(rep);
            free(our_data);
        }
        mimo_tokenizer_free(ctx);

    } else if (backend_name == "mimo-asr") {
        // MiMo-V2.5-ASR LM-half: pulls input_ids from the ref GGUF (the
        // Python dumper saved the full [9, T_total] prompt under
        // `prefill_input_ids` as F32). Casts to int32 and runs the C++
        // prefill graph stage-by-stage. The audio tokenizer GGUF path is
        // optional for the diff harness — the ref input_ids already
        // capture the codes the Python tokenizer produced, and the C++
        // LM-half graph reads them directly. Set MIMO_TOKENIZER_GGUF to
        // also exercise the codes-side bit-equality check.
        auto cp = mimo_asr_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        cp.use_gpu = std::getenv("MIMO_ASR_GPU") != nullptr;
        mimo_asr_context* ctx = mimo_asr_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load mimo-asr model '%s'\n", model_path.c_str());
            return 4;
        }

        // Pull the [9, T_total] input_ids tensor from the ref archive.
        auto ids_pair = ref.get_f32("prefill_input_ids");
        auto ids_shape = ref.shape("prefill_input_ids");
        if (!ids_pair.first || ids_shape.empty()) {
            fprintf(stderr, "mimo-asr: ref archive missing 'prefill_input_ids' — re-dump with the mimo-asr backend\n");
            mimo_asr_free(ctx);
            return 4;
        }
        // Shape is [9, T_total] (gguf row-major == ne=[T_total, 9] in ggml's
        // column-major ne convention). The dumper writes via
        // `out["prefill_input_ids"] = input_ids.detach().cpu().numpy().astype(int32).astype(float32)`
        // where input_ids is shape [9, T_total]. Numpy default order is C,
        // so the data lays out as 9 rows × T_total cols row-major. GGUF
        // stores ne[0]=T_total, ne[1]=9, which matches what we read here.
        const int T_total = (int)ids_shape[0];
        const int n_chan = ids_shape.size() >= 2 ? (int)ids_shape[1] : 9;
        if (n_chan != 9) {
            fprintf(stderr, "mimo-asr: prefill_input_ids has %d channels, expected 9\n", n_chan);
            mimo_asr_free(ctx);
            return 4;
        }
        std::vector<int32_t> input_ids((size_t)9 * T_total);
        for (size_t i = 0; i < input_ids.size(); i++) {
            input_ids[i] = (int32_t)std::lround(ids_pair.first[i]);
        }

        // Stage list matches DEFAULT_STAGES in tools/reference_backends/mimo_asr.py
        // (skipping the ones that require running the wrapper.asr_sft like
        // generated_text — those are handled separately).
        static const char* stages[] = {
            "prefill_audio_features", "prefill_text_embeds",       "prefill_inputs_embeds",
            "prefill_last_hidden",    "prefill_text_logits_step0",
        };
        std::vector<float> saved_text_embeds, saved_audio_features;
        for (const char* stage : stages) {
            int n_stage = 0;
            float* our_data = mimo_asr_extract_stage(ctx, input_ids.data(), T_total, stage, &n_stage);
            if (!our_data) {
                printf("[ERR ] %-26s  extract returned null\n", stage);
                n_fail++;
                continue;
            }
            auto rep = ref.compare(stage, our_data, (size_t)n_stage);
            print_row(stage, rep, COS_THRESHOLD);
            record(rep);
            // Save for an out-of-graph sum check below.
            if (std::string(stage) == "prefill_text_embeds")
                saved_text_embeds.assign(our_data, our_data + n_stage);
            else if (std::string(stage) == "prefill_audio_features")
                saved_audio_features.assign(our_data, our_data + n_stage);
            free(our_data);
        }
        // Bisect: compute (extracted text_embeds + extracted audio_features)
        // out-of-graph and compare to ref's prefill_inputs_embeds. If the
        // in-graph ggml_add is buggy, this should match the ref while the
        // graph's prefill_inputs_embeds does not.
        if (!saved_text_embeds.empty() && saved_text_embeds.size() == saved_audio_features.size()) {
            std::vector<float> sum_check(saved_text_embeds.size());
            for (size_t i = 0; i < sum_check.size(); i++)
                sum_check[i] = saved_text_embeds[i] + saved_audio_features[i];
            auto rep = ref.compare("prefill_inputs_embeds", sum_check.data(), sum_check.size());
            print_row("dbg_extracted_sum", rep, COS_THRESHOLD);
        }
        mimo_asr_free(ctx);

    } else if (backend_name == "granite" || backend_name == "granite-4.1") {
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

            int enc_N = 0, enc_dim = 0;
            float* enc_out =
                granite_speech_run_encoder(ctx, mel_r.data.data(), mel_r.shape[0], mel_r.shape[1], &enc_N, &enc_dim);
            if (enc_out) {
                auto rep2 = ref.compare("encoder_out", enc_out, (size_t)enc_N * enc_dim);
                print_row("encoder_out", rep2, COS_THRESHOLD);
                record(rep2);

                int proj_N = 0, proj_dim = 0;
                float* proj_out = granite_speech_run_projector(ctx, enc_out, enc_N, enc_dim, &proj_N, &proj_dim);
                free(enc_out);
                if (proj_out) {
                    auto rep3 = ref.compare("projector_out", proj_out, (size_t)proj_N * proj_dim);
                    print_row("projector_out", rep3, COS_THRESHOLD);
                    record(rep3);
                    free(proj_out);
                }
            }
        } else {
            printf("[ERR ] mel_spectrogram         %s\n", mel_r.note.c_str());
            n_fail++;
        }
        granite_speech_free(ctx);
    } else if (backend_name == "granite-nle") {
        auto cp = granite_nle_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        granite_nle_context* ctx = granite_nle_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load granite-nle model\n");
            return 4;
        }
        auto mel_r = granite_nle_mel(ctx, samples.data(), (int)samples.size());
        if (mel_r.ok) {
            auto rep = ref.compare("mel_spectrogram", mel_r.data.data(), mel_r.data.size());
            print_row("mel_spectrogram", rep, COS_THRESHOLD);
            record(rep);

            int enc_T = 0, enc_dim = 0;
            float* enc_out =
                granite_nle_run_encoder(ctx, mel_r.data.data(), mel_r.shape[0], mel_r.shape[1], &enc_T, &enc_dim);
            if (enc_out) {
                auto rep2 = ref.compare("encoder_output", enc_out, (size_t)enc_T * enc_dim);
                print_row("encoder_output", rep2, COS_THRESHOLD);
                record(rep2);

                int ctc_T = 0, ctc_V = 0;
                const float* ctc = granite_nle_last_ctc_logits(ctx, &ctc_T, &ctc_V);
                if (ctc && ctc_V > 0) {
                    auto rep3 = ref.compare("encoder_logits", ctc, (size_t)ctc_T * ctc_V);
                    print_row("encoder_logits", rep3, COS_THRESHOLD);
                    record(rep3);
                }

                int proj_T = 0, proj_dim = 0;
                float* proj_out = granite_nle_run_projector(ctx, enc_out, enc_T, enc_dim, &proj_T, &proj_dim);
                if (proj_out && proj_T > 0 && proj_dim > 0) {
                    auto rep4 = ref.compare("projector_output", proj_out, (size_t)proj_T * proj_dim);
                    print_row("projector_output", rep4, COS_THRESHOLD);
                    record(rep4);
                    free(proj_out);
                } else {
                    printf("[ERR ] projector_output        granite_nle_run_projector returned null\n");
                    n_fail++;
                }
                free(enc_out);

                // ---- LLM editing forward ----
                // Reference-input path: fetch the upstream's audio embeds
                // and slot IDs from the dump and run our LLM forward
                // against them. Isolates LLM-only error from upstream
                // mel/encoder/projector divergence.
                auto audio_pair = ref.get_f32("audio_embs_for_llm");
                auto ids_pair = ref.get_f32("text_ids_with_slots");
                if (audio_pair.first && ids_pair.first) {
                    // ggml shape convention: ne[0] = feature dim (innermost
                    // in memory), ne[1] = sequence length.
                    auto audio_shape = ref.shape("audio_embs_for_llm");
                    int audio_d = audio_shape.empty() ? 0 : (int)audio_shape[0];
                    int n_audio = audio_shape.size() >= 2 ? (int)audio_shape[1] : 0;
                    int n_text = (int)ids_pair.second;

                    std::vector<int32_t> text_ids((size_t)n_text);
                    for (int i = 0; i < n_text; i++)
                        text_ids[i] = (int32_t)std::lround(ids_pair.first[i]);

                    int edit_n = 0, edit_V = 0;
                    float* edit_logits = granite_nle_run_llm_editing(ctx, audio_pair.first, n_audio, text_ids.data(),
                                                                     n_text, &edit_n, &edit_V);
                    if (edit_logits && edit_n > 0 && edit_V > 0) {
                        auto rep5 = ref.compare("editing_logits", edit_logits, (size_t)edit_n * edit_V);
                        print_row("editing_logits", rep5, COS_THRESHOLD);
                        record(rep5);
                        auto rep5b = ref.compare_argmax("editing_logits", edit_logits, (size_t)edit_n * edit_V);
                        print_row("editing_logits_top1", rep5b, COS_THRESHOLD);
                        free(edit_logits);
                    } else {
                        printf("[ERR ] editing_logits          granite_nle_run_llm_editing returned null\n");
                        n_fail++;
                    }
                    (void)audio_d;
                } else {
                    printf("[SKIP] editing_logits          ref missing audio_embs_for_llm/text_ids_with_slots\n");
                }

                // ---- end-to-end transcribe ----
                // Runs the full pipeline (mel → encoder → BPE-CTC → projector
                // → LLM editing → slot decode) from raw samples and compares
                // against the upstream `final_text` metadata string.
                {
                    std::string ref_text = ref.meta("final_text");
                    if (ref_text.empty())
                        ref_text = ref.meta("generated_text");
                    char* my_text = granite_nle_transcribe(ctx, samples.data(), (int)samples.size());
                    if (my_text) {
                        bool match = (!ref_text.empty()) && (ref_text == std::string(my_text));
                        if (match) {
                            printf("[PASS] transcribe              %s\n", my_text);
                        } else if (ref_text.empty()) {
                            printf("[INFO] transcribe              %s (no ref)\n", my_text);
                        } else {
                            printf("[FAIL] transcribe              cpp: %s\n", my_text);
                            printf("                              ref: %s\n", ref_text.c_str());
                            n_fail++;
                        }
                        free(my_text);
                    } else {
                        printf("[ERR ] transcribe              granite_nle_transcribe returned null\n");
                        n_fail++;
                    }
                }
            } else {
                printf("[ERR ] encoder_output          granite_nle_run_encoder returned null\n");
                n_fail++;
            }
        } else {
            printf("[ERR ] mel_spectrogram         %s\n", mel_r.note.c_str());
            n_fail++;
        }
        granite_nle_free(ctx);
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

        // Diagnostic: feed the reference mel to our encoder. If this passes
        // while encoder_output (with our mel) fails, the encoder is OK and
        // residual mel error is responsible. If it also fails, encoder bug.
        auto enc_ref_r = parakeet_encoder_with_ref_mel_r(ctx, ref);
        if (enc_ref_r.ok) {
            auto rep = ref.compare("encoder_output", enc_ref_r.data.data(), enc_ref_r.data.size());
            print_row("encoder_output_ref_mel", rep, COS_THRESHOLD);
            record(rep);
        } else {
            printf("[SKIP] encoder_output_ref_mel  %s\n", enc_ref_r.note.c_str());
        }

        // Per-layer diff: localise the encoder bug. Uses the reference mel
        // as input so we measure encoder-internal divergence only (no mel
        // bleed-through). Captures pre_encode + every conformer layer.
        if (ref.has("pre_encode_output") || ref.has("encoder_layer_0")) {
            auto mel_pair = ref.get_f32("mel_spectrogram");
            auto mel_shp = ref.shape("mel_spectrogram");
            if (mel_pair.first && mel_shp.size() >= 2) {
                const int n_mels = (int)mel_shp[0];
                const int T_mel = (int)mel_shp[1];
                const int n_layers = 24;
                const int d_model = 1024;
                // Predict T_enc as ceil_div(T_mel, 8) for sizing buffers — the
                // exact value comes back from the runner. Allocate generous.
                const int T_enc_max = (T_mel + 7) / 8 + 4;
                std::vector<std::vector<float>> bufs(n_layers + 1, std::vector<float>((size_t)d_model * T_enc_max));
                std::vector<float*> ptrs(n_layers + 1);
                for (int i = 0; i < n_layers + 1; i++)
                    ptrs[i] = bufs[i].data();
                int T_enc = 0, d_out = 0;
                int rc = parakeet_run_encoder_dump(ctx, mel_pair.first, n_mels, T_mel, ptrs.data(), (int)ptrs.size(),
                                                   &T_enc, &d_out);
                if (rc == 0 && T_enc > 0) {
                    auto rep0 = ref.compare("pre_encode_output", ptrs[0], (size_t)T_enc * d_out);
                    print_row("pre_encode_output", rep0, COS_THRESHOLD);
                    for (int il = 0; il < n_layers; il++) {
                        char nm[64];
                        snprintf(nm, sizeof(nm), "encoder_layer_%d", il);
                        auto rep = ref.compare(nm, ptrs[il + 1], (size_t)T_enc * d_out);
                        print_row(nm, rep, COS_THRESHOLD);
                    }
                } else {
                    printf("[SKIP] encoder_layer_*       parakeet_run_encoder_dump rc=%d\n", rc);
                }
            }
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
    } else if (backend_name == "kokoro") {
        // Kokoro / StyleTTS2: text-driven TTS, the 4th positional arg
        // (audio.wav) is unused — phonemes come from the reference
        // archive's `kokoro_phonemes` metadata (written by the Python
        // dumper). The voice-pack GGUF path comes from the
        // KOKORO_VOICE_GGUF env var, defaulting to the canonical
        // af_heart pack.
        const std::string phonemes = ref.meta("kokoro_phonemes");
        if (phonemes.empty()) {
            fprintf(stderr, "crispasr-diff kokoro: reference is missing kokoro_phonemes metadata. "
                            "Re-dump with KOKORO_PHONEMES=<ipa> set.\n");
            return 4;
        }
        const char* voice_env = std::getenv("KOKORO_VOICE_GGUF");
        const std::string voice_gguf =
            (voice_env && *voice_env) ? voice_env : "/tmp/kokoro_voices/kokoro-voice-af_heart.gguf";

        auto cp = kokoro_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        kokoro_context* ctx = kokoro_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "failed to load kokoro model '%s'\n", model_path.c_str());
            return 4;
        }
        if (kokoro_load_voice_pack(ctx, voice_gguf.c_str()) != 0) {
            fprintf(stderr, "failed to load voice pack '%s'\n", voice_gguf.c_str());
            kokoro_free(ctx);
            return 4;
        }
        printf("crispasr-diff kokoro: phonemes=%s  voice=%s\n", phonemes.c_str(), voice_gguf.c_str());

        // Stage list mirrors DEFAULT_STAGES in tools/reference_backends/kokoro.py.
        // Tolerance split: stages 0..11 are deterministic and must hit
        // cos ≥ 0.999; stages 12..15 (gen_pre_post_out, mag, phase,
        // audio_out) depend on SineGen's RNG, which the Python and C++
        // sides cannot match exactly (PyTorch RNG ≠ std::mt19937), so
        // they only need to clear cos ≥ 0.95.
        struct KStage {
            const char* name;
            float threshold;
        };
        static const KStage kokoro_stages[] = {
            {"token_ids", COS_THRESHOLD},
            {"bert_pooler_out", COS_THRESHOLD},
            {"bert_proj_out", COS_THRESHOLD},
            {"text_enc_out", COS_THRESHOLD},
            {"dur_enc_out", COS_THRESHOLD},
            {"pred_lstm_out", COS_THRESHOLD},
            {"durations", COS_THRESHOLD},
            {"align_out", COS_THRESHOLD},
            {"f0_curve", COS_THRESHOLD},
            {"n_curve", COS_THRESHOLD},
            {"dec_encode_out", COS_THRESHOLD},
            {"dec_decode_3_out", COS_THRESHOLD},
            // RNG-divergent stages — looser cosine threshold (still
            // catches structural breakage; deterministic content is the
            // bulk of magnitude in mag/phase/audio).
            {"gen_pre_post_out", 0.95f},
            {"mag", 0.95f},
            {"phase", 0.95f},
            {"audio_out", 0.95f},
        };
        const char* dump_dir = std::getenv("KOKORO_DUMP_STAGES");
        for (const auto& s : kokoro_stages) {
            int n_stage = 0;
            float* mine = kokoro_extract_stage(ctx, phonemes.c_str(), s.name, &n_stage);
            if (!mine || n_stage <= 0) {
                printf("[ERR ] %-22s kokoro_extract_stage returned null\n", s.name);
                if (mine)
                    free(mine);
                n_fail++;
                continue;
            }
            if (dump_dir && *dump_dir) {
                char path[512];
                snprintf(path, sizeof(path), "%s/cpp_%s.bin", dump_dir, s.name);
                FILE* fp = fopen(path, "wb");
                if (fp) {
                    fwrite(mine, sizeof(float), (size_t)n_stage, fp);
                    fclose(fp);
                }
            }
            auto rep = ref.compare(s.name, mine, (size_t)n_stage);
            // print_row uses a fixed threshold for PASS/FAIL; route the
            // looser-tolerance stages through their own threshold so
            // they're tagged correctly.
            print_row(s.name, rep, s.threshold);
            if (!rep.found) {
                n_skip++;
            } else if (rep.is_pass(s.threshold)) {
                n_pass++;
            } else {
                n_fail++;
            }
            free(mine);
        }
        kokoro_free(ctx);
    } else if (backend_name == "orpheus") {
        // Orpheus SNAC 24 kHz codec-decoder diff. The talker is out of
        // scope for this slice; we drive the C++ SNAC graph directly
        // with the same deterministic 7N-token stream the Python
        // reference (tools/reference_backends/orpheus_snac.py) uses,
        // de-interleaved into 3 codebook tensors per the canonical
        // 7-slot super-frame layout.
        //
        // model_path is the SNAC GGUF (cstr/snac-24khz-GGUF or built
        // locally via models/convert-snac-to-gguf.py). The audio.wav
        // arg is unused (codec-only).
        snac_decoder_params sp = snac_decoder_default_params();
        sp.n_threads = 4;
        sp.verbosity = 0;
        sp.use_gpu = std::getenv("ORPHEUS_SNAC_GPU") != nullptr;
        snac_decoder_ctx* ctx = snac_decoder_init_from_file(model_path.c_str(), sp);
        if (!ctx) {
            fprintf(stderr, "failed to load SNAC codec from '%s'\n", model_path.c_str());
            return 4;
        }

        // Build the same deterministic codes the Python ref builds
        // (orpheus_snac.py:_build_codes). 7 LM tokens per super-frame:
        //   slot 0     → codes_0    (1 entry / super-frame)
        //   slot 1, 4  → codes_1    (2 / super-frame)
        //   slot 2,3,5,6 → codes_2  (4 / super-frame)
        const int T_super = std::getenv("ORPHEUS_SNAC_T_SUPER") ? std::atoi(std::getenv("ORPHEUS_SNAC_T_SUPER")) : 4;
        const int fill_code = std::getenv("ORPHEUS_SNAC_CODE") ? std::atoi(std::getenv("ORPHEUS_SNAC_CODE")) : 0;
        const int code = ((fill_code % 4096) + 4096) % 4096;
        std::vector<int32_t> c0((size_t)T_super, code);
        std::vector<int32_t> c1((size_t)T_super * 2, code);
        std::vector<int32_t> c2((size_t)T_super * 4, code);

        printf("crispasr-diff: orpheus SNAC: T_super=%d  code=%d  → %zu+%zu+%zu codebook entries\n", T_super, code,
               c0.size(), c1.size(), c2.size());

        // Stage list matches DEFAULT_STAGES in orpheus_snac.py minus
        // the trivially-equal codes_{0,1,2}. snac_pcm_emit is derived
        // from snac_pcm by slicing [:, :, 2048:4096] when T_super == 4.
        static const char* stages[] = {
            "snac_quant_out", "snac_dec_pre",  "snac_dec_blk0", "snac_dec_blk1",
            "snac_dec_blk2",  "snac_dec_blk3", "snac_pcm",
        };
        for (const char* stage : stages) {
            int n_stage = 0;
            float* our_data = snac_decoder_extract_stage(ctx, c0.data(), (int)c0.size(), c1.data(), (int)c1.size(),
                                                         c2.data(), (int)c2.size(), stage, &n_stage);
            if (!our_data) {
                printf("[ERR ] %-22s  extract returned null\n", stage);
                n_fail++;
                continue;
            }
            auto rep = ref.compare(stage, our_data, (size_t)n_stage);
            print_row(stage, rep, COS_THRESHOLD);
            record(rep);

            // Streaming-window slice. Compare against snac_pcm_emit when
            // T_super == 4 — the canonical orpheus streaming case.
            if (std::strcmp(stage, "snac_pcm") == 0 && T_super == 4 && n_stage >= 4096) {
                const int slice_n = 2048;
                auto rep_emit = ref.compare("snac_pcm_emit", our_data + 2048, (size_t)slice_n);
                print_row("snac_pcm_emit", rep_emit, COS_THRESHOLD);
                record(rep_emit);
            }
            free(our_data);
        }
        snac_decoder_free(ctx);
    } else {
        fprintf(stderr,
                "crispasr-diff: backend '%s' is not recognised. "
                "Supported: voxtral, voxtral4b, qwen3, qwen3-tts, qwen3-tts-codec, kokoro, granite, granite-4.1, "
                "granite-nle, parakeet, canary, cohere, gemma4, mimo-tokenizer, mimo-asr, orpheus.\n",
                backend_name.c_str());
        return 5;
    }

    printf("\nsummary: %d pass, %d fail, %d skip (cos threshold %.3f)\n", n_pass, n_fail, n_skip, COS_THRESHOLD);
    return n_fail == 0 ? 0 : 6;
}
