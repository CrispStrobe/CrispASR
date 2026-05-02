// voxtral.cpp — Mistral Voxtral-Mini-3B-2507 ggml runtime
//
// Stage V1: GGUF loader + Llama 3 / Mistral LLM forward graph + smoke test API.
// Audio encoder, projector, Tekken tokenizer, and audio injection are all
// deferred to later stages — see voxtral-todo.md.
//
// The LLM forward is structurally a strict subset of qwen3_asr.cpp's
// build_llm_body: same SwiGLU + GQA + NEOX RoPE + RMSNorm pattern, just
// without the Qwen3-specific Q-norm/K-norm and with different hyperparams
// (30 layers, d=3072, GQA 32/8, head_dim=128, FFN=8192, RoPE θ=1e8,
// vocab=131072). No biases anywhere.

#include "voxtral.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// ===========================================================================
// Hyper-parameters (filled from voxtral.* GGUF kv)
// ===========================================================================

struct voxtral_hparams {
    // Audio encoder
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 400;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;
    uint32_t audio_n_layers = 32;
    uint32_t audio_d_model = 1280;
    uint32_t audio_n_heads = 20;
    uint32_t audio_head_dim = 64;
    uint32_t audio_ff_dim = 5120;
    uint32_t audio_max_pos = 1500;

    // Projector
    uint32_t proj_in_dim = 5120;
    uint32_t proj_out_dim = 3072;
    uint32_t proj_frame_stack = 4;

    // LLM (Llama 3 / Mistral)
    uint32_t llm_n_layers = 30;
    uint32_t llm_d_model = 3072;
    uint32_t llm_n_heads = 32;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 8192;
    float llm_rope_theta = 1e8f;
    float llm_rms_eps = 1e-5f;
    uint32_t llm_vocab_size = 131072;
    uint32_t llm_max_pos = 131072;

    uint32_t audio_token_id = 24;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct voxtral_audio_block {
    // Pre-LN self-attention (Whisper-style: biased q/v/out, no bias on K)
    ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr; // NO bias (Whisper quirk)
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    // Pre-LN FFN (GELU)
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;     // fc1
    ggml_tensor *ffn_down_w = nullptr, *ffn_down_b = nullptr; // fc2
};

struct voxtral_audio_tower {
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;
    ggml_tensor* embed_positions = nullptr; // (max_pos, d_model) F32
    std::vector<voxtral_audio_block> blocks;
    ggml_tensor *ln_post_w = nullptr, *ln_post_b = nullptr;
    // Mel preprocessor (baked from WhisperFeatureExtractor)
    ggml_tensor* mel_filters = nullptr; // (n_freqs=201, n_mels=128) F32
    ggml_tensor* mel_window = nullptr;  // (400,) F32 hann window
};

struct voxtral_projector {
    ggml_tensor* proj1 = nullptr; // (in_dim=5120, out_dim=3072)
    ggml_tensor* proj2 = nullptr; // (out_dim=3072, out_dim=3072)
};

struct voxtral_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    // PLAN #60d: optional fused QKV. Built at load time if all q/k/v
    // weights share a quantization-compatible type. Routes through
    // core_attn::kv_self_attn's qkv_w branch for one matmul/layer instead
    // of three. ~7-8 % decode speedup on M1 Q4_K.
    ggml_tensor* attn_qkv_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct voxtral_llm {
    ggml_tensor* token_embd_w = nullptr;
    std::vector<voxtral_llm_block> blocks;
    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* output_w = nullptr;
};

struct voxtral_model {
    voxtral_hparams hparams;
    voxtral_audio_tower audio;
    voxtral_projector projector;
    voxtral_llm llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

// Tekken tokenizer state — populated from GGUF blobs at load time, but the
// actual encode/decode logic is deferred to Stage V3. For Stage V1 we just
// stash the blobs and provide voxtral_token_text() that walks the vocab.
struct voxtral_vocab {
    // Length-prefixed concatenation of token_bytes entries (one per rank).
    // Each entry: u16 length + raw bytes.
    std::vector<uint8_t> tekken_vocab_blob;
    // Per-rank offsets into tekken_vocab_blob (offset of the bytes, not the
    // length prefix). offsets[i+1] - offsets[i] would walk to the next entry
    // but we store length explicitly per rank.
    std::vector<uint32_t> rank_offset;
    std::vector<uint32_t> rank_length;

    // Special tokens (rank 0..999): index → string
    std::vector<std::string> specials;
    // Reverse: string → rank
    std::unordered_map<std::string, int32_t> special_to_rank;

    int n_specials = 0;
    int n_vocab = 0;

    std::string pre_pattern; // tiktoken-style pre-tokenizer regex

    // Reverse lookup: byte_sequence → rank (built lazily on first tokenize call)
    std::unordered_map<std::string, int32_t> bytes_to_rank;
    bool reverse_built = false;
};

struct voxtral_context {
    voxtral_context_params params;

    voxtral_model model;
    voxtral_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;

    // KV cache (F16, same pattern as qwen3_asr)
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;

    // PLAN #60d: fused QKV buffer (allocated at load time).
    ggml_context* fused_ctx = nullptr;
    ggml_backend_buffer_t fused_buf = nullptr;

    int n_threads = 4;
};

// ===========================================================================
// Loader helpers
// ===========================================================================

#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static ggml_tensor* try_get(voxtral_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(voxtral_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "voxtral");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool voxtral_load_model(voxtral_model& model, voxtral_vocab& vocab, const char* path, ggml_backend_t backend) {
    // ---- pass 1: read hparams + vocab via metadata-only context ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "voxtral.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "voxtral.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "voxtral.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "voxtral.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "voxtral.hop_length", hp.hop_length);
        hp.audio_n_layers = core_gguf::kv_u32(gctx, "voxtral.audio.n_layers", hp.audio_n_layers);
        hp.audio_d_model = core_gguf::kv_u32(gctx, "voxtral.audio.d_model", hp.audio_d_model);
        hp.audio_n_heads = core_gguf::kv_u32(gctx, "voxtral.audio.n_heads", hp.audio_n_heads);
        hp.audio_head_dim = core_gguf::kv_u32(gctx, "voxtral.audio.head_dim", hp.audio_head_dim);
        hp.audio_ff_dim = core_gguf::kv_u32(gctx, "voxtral.audio.ff_dim", hp.audio_ff_dim);
        hp.audio_max_pos = core_gguf::kv_u32(gctx, "voxtral.audio.max_pos", hp.audio_max_pos);

        hp.proj_in_dim = core_gguf::kv_u32(gctx, "voxtral.proj.in_dim", hp.proj_in_dim);
        hp.proj_out_dim = core_gguf::kv_u32(gctx, "voxtral.proj.out_dim", hp.proj_out_dim);
        hp.proj_frame_stack = core_gguf::kv_u32(gctx, "voxtral.proj.frame_stack", hp.proj_frame_stack);

        hp.llm_n_layers = core_gguf::kv_u32(gctx, "voxtral.llm.n_layers", hp.llm_n_layers);
        hp.llm_d_model = core_gguf::kv_u32(gctx, "voxtral.llm.d_model", hp.llm_d_model);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "voxtral.llm.n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "voxtral.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "voxtral.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "voxtral.llm.ff_dim", hp.llm_ff_dim);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "voxtral.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "voxtral.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "voxtral.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "voxtral.llm.max_pos", hp.llm_max_pos);
        hp.audio_token_id = core_gguf::kv_u32(gctx, "voxtral.audio_token_id", hp.audio_token_id);

        // ---- Tekken tokenizer blobs ----
        vocab.pre_pattern = core_gguf::kv_str(gctx, "tokenizer.tekken.pattern", "");

        auto specials = core_gguf::kv_str_array(gctx, "tokenizer.tekken.specials");
        if (!specials.empty()) {
            vocab.specials = std::move(specials);
            vocab.special_to_rank.reserve(vocab.specials.size());
            for (int i = 0; i < (int)vocab.specials.size(); i++) {
                vocab.special_to_rank[vocab.specials[i]] = i;
            }
        }
        vocab.n_specials = core_gguf::kv_u32(gctx, "tokenizer.tekken.n_specials", 1000);
        vocab.n_vocab = core_gguf::kv_u32(gctx, "tokenizer.tekken.n_vocab", 150000);

        // The vocab blob is stored as a 1D F32 tensor (one float per byte)
        // because the GGUF KV array path loses uint8 precision. We'll read
        // it from the tensor data in pass 2 after the weights are loaded.

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: tensor data via shared helper ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "voxtral", wl)) {
        return false;
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // ---- bind named tensors into the per-layer structs ----
    auto& a = model.audio;
    a.conv1_w = require(model, "audio.conv.1.weight");
    a.conv1_b = require(model, "audio.conv.1.bias");
    a.conv2_w = require(model, "audio.conv.2.weight");
    a.conv2_b = require(model, "audio.conv.2.bias");
    a.embed_positions = require(model, "audio.embed_positions");
    a.ln_post_w = require(model, "audio.ln_post.weight");
    a.ln_post_b = require(model, "audio.ln_post.bias");
    a.mel_filters = try_get(model, "audio.mel_filters");
    a.mel_window = try_get(model, "audio.mel_window");
    a.blocks.resize(model.hparams.audio_n_layers);
    for (uint32_t i = 0; i < model.hparams.audio_n_layers; i++) {
        char buf[128];
        auto& b = a.blocks[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "audio.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_norm_b = get("attn_norm.bias");
        b.attn_q_w = get("attn_q.weight");
        b.attn_q_b = get("attn_q.bias");
        b.attn_k_w = get("attn_k.weight");
        // K has no bias (Whisper quirk) — skip
        b.attn_v_w = get("attn_v.weight");
        b.attn_v_b = get("attn_v.bias");
        b.attn_out_w = get("attn_out.weight");
        b.attn_out_b = get("attn_out.bias");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_norm_b = get("ffn_norm.bias");
        b.ffn_up_w = get("ffn_up.weight");
        b.ffn_up_b = get("ffn_up.bias");
        b.ffn_down_w = get("ffn_down.weight");
        b.ffn_down_b = get("ffn_down.bias");
    }

    // ---- Reconstruct the Tekken vocab blob from the F32 tensor ----
    {
        ggml_tensor* vt = try_get(model, "tokenizer.tekken.vocab_tensor");
        if (vt) {
            size_t n = (size_t)vt->ne[0];
            std::vector<float> f32(n);
            ggml_backend_tensor_get(vt, f32.data(), 0, n * sizeof(float));
            vocab.tekken_vocab_blob.resize(n);
            for (size_t i = 0; i < n; i++)
                vocab.tekken_vocab_blob[i] = (uint8_t)(int)f32[i];

            // Build per-rank offset/length tables by walking length prefixes
            vocab.rank_offset.reserve(vocab.n_vocab);
            vocab.rank_length.reserve(vocab.n_vocab);
            size_t pos = 0;
            for (int r = 0; r < vocab.n_vocab; r++) {
                if (pos + 2 > n)
                    break;
                uint16_t len;
                std::memcpy(&len, vocab.tekken_vocab_blob.data() + pos, 2);
                pos += 2;
                if (pos + len > n)
                    break;
                vocab.rank_offset.push_back((uint32_t)pos);
                vocab.rank_length.push_back(len);
                pos += len;
            }
        }
    }

    auto& p = model.projector;
    p.proj1 = require(model, "proj1.weight");
    p.proj2 = require(model, "proj2.weight");

    auto& l = model.llm;
    l.token_embd_w = require(model, "token_embd.weight");
    l.output_norm_w = require(model, "output_norm.weight");
    l.output_w = require(model, "output.weight");
    l.blocks.resize(model.hparams.llm_n_layers);
    for (uint32_t i = 0; i < model.hparams.llm_n_layers; i++) {
        char buf[128];
        auto& b = l.blocks[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_q_w = get("attn_q.weight");
        b.attn_k_w = get("attn_k.weight");
        b.attn_v_w = get("attn_v.weight");
        b.attn_output_w = get("attn_output.weight");
        // NO Q-norm/K-norm for Llama / Mistral
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn_gate.weight");
        b.ffn_up_w = get("ffn_up.weight");
        b.ffn_down_w = get("ffn_down.weight");
    }

    return true;
}

// ===========================================================================
// FFT + Mel computation (cargo-cult from qwen3_asr.cpp — same parameters)
// ===========================================================================

static void voxtral_dft(const float* in, int N, float* out) {
    for (int k = 0; k < N; k++) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; n++) {
            float ang = -2.0f * (float)M_PI * (float)k * (float)n / (float)N;
            re += in[n] * std::cos(ang);
            im += in[n] * std::sin(ang);
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

static void voxtral_fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    int half = N / 2;
    if (N - half * 2 == 1) {
        voxtral_dft(in, N, out);
        return;
    }
    float* even = in + N;
    for (int i = 0; i < half; i++)
        even[i] = in[2 * i];
    float* ef = out + 2 * N;
    voxtral_fft(even, half, ef);
    float* odd = even;
    for (int i = 0; i < half; i++)
        odd[i] = in[2 * i + 1];
    float* of = ef + N;
    voxtral_fft(odd, half, of);
    for (int k = 0; k < half; k++) {
        float ang = -2.0f * (float)M_PI * (float)k / (float)N;
        float re = std::cos(ang), im = std::sin(ang);
        float reo = of[2 * k], imo = of[2 * k + 1];
        out[2 * k] = ef[2 * k] + re * reo - im * imo;
        out[2 * k + 1] = ef[2 * k + 1] + re * imo + im * reo;
        out[2 * (k + half)] = ef[2 * k] - re * reo + im * imo;
        out[2 * (k + half) + 1] = ef[2 * k + 1] - re * imo - im * reo;
    }
}

#include "core/mel.h"
#include "core/ffn.h"
#include "core/attention.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// core_mel::FftR2C expects a const-input function and passes a buffer of
// exactly N floats. voxtral_fft() uses `in` as scratch during recursion
// and needs 4*N slots, so we wrap it here with a local scratch vector
// that copies the input and discards any writes.
static void voxtral_fft_wrapper(const float* in, int N, float* out) {
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size() < 4 * N)
        scratch_in.assign((size_t)4 * N, 0.0f);
    if ((int)scratch_out.size() < 8 * N)
        scratch_out.assign((size_t)8 * N, 0.0f);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    voxtral_fft(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}

extern "C" float* voxtral_compute_mel(voxtral_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                      int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    if (!ctx->model.audio.mel_filters || !ctx->model.audio.mel_window) {
        fprintf(stderr, "voxtral: GGUF missing audio.mel_filters/mel_window — re-convert\n");
        return nullptr;
    }
    const int n_fft = 400, hop = 160, n_mels = 128, n_freqs = 201;
    const int T_out = 3000; // Voxtral always pads to 30s

    // Window is stored already padded to n_fft in the GGUF, so win_length
    // == n_fft and core_mel's center-pad step is a no-op for it.
    std::vector<float> hann(n_fft);
    ggml_backend_tensor_get(ctx->model.audio.mel_window, hann.data(), 0, n_fft * sizeof(float));
    // Filterbank is stored (n_freqs, n_mels) — the HF layout — so we tell
    // core_mel to use FbLayout::FreqsMels.
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->model.audio.mel_filters, filt.data(), 0, filt.size() * sizeof(float));

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Log10;
    p.log_guard = core_mel::LogGuard::MaxClip; // log10(max(x, 1e-10))
    p.norm = core_mel::Normalization::GlobalClipMax;
    p.layout = core_mel::Layout::MelsTime;
    p.fb_layout = core_mel::FbLayout::FreqsMels; // HF storage order
    p.matmul = core_mel::MatmulPrecision::Double;
    p.log_eps = 1e-10f;
    p.center_pad = true;
    p.drop_last_frame = true; // Whisper convention
    p.pad_to_T = T_out;

    int T_ret = 0;
    auto mel =
        core_mel::compute(samples, n_samples, hann.data(), n_fft, filt.data(), n_freqs, voxtral_fft_wrapper, p, T_ret);

    if (mel.empty())
        return nullptr;

    if (out_n_mels)
        *out_n_mels = n_mels;
    if (out_T_mel)
        *out_T_mel = T_ret;

    float* result = (float*)malloc(mel.size() * sizeof(float));
    std::memcpy(result, mel.data(), mel.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Audio encoder + projector graph (Stage V2)
//
// Architecture (from the reference dump):
//   mel (128, 3000) F32 — padded to 30s
//   → Conv1d(128→1280, k=3, stride=1, pad=1) + GELU → (1280, 3000)
//   → Conv1d(1280→1280, k=3, stride=2, pad=1) + GELU → (1280, 1500)
//   → transpose to (1500, 1280), add learned pos embed (1500, 1280)
//   → 32 × Whisper-style pre-LN encoder block
//   → layer_norm → (1500, 1280)
//   → reshape to (375, 5120) — stack 4 adjacent frames
//   → linear_1(5120→3072) + GELU + linear_2(3072→3072)
//   → (375, 3072) audio embeddings for the LLM
//
// Note: Conv1d weights in ggml are stored as (K, IC, OC) after the
// gguf-py dim reversal from PyTorch's (OC, IC, K). ggml_conv_1d expects
// kernel shape (K_w, IC, OC) = (ne[0], ne[1], ne[2]).
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

static ggml_cgraph* voxtral_build_graph_encoder(voxtral_context* ctx) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.audio_d_model;         // 1280
    const int n_heads = (int)hp.audio_n_heads;   // 20
    const int head_dim = (int)hp.audio_head_dim; // 64
    const int n_layers = (int)hp.audio_n_layers; // 32
    const int proj_in = (int)hp.proj_in_dim;     // 5120
    const int n_mels = (int)hp.n_mels;           // 128
    const int T_mel = 3000;                      // padded to 30s
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: mel spectrogram (n_mels=128, T_mel=3000) F32
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // ---- Conv1d front-end ----
    // conv1: (128→1280, k=3, stride=1, pad=1). Output T stays 3000.
    // ggml_conv_1d(kernel, input, stride, pad, dilation)
    // kernel ne = (3, 128, 1280), input ne = (T, 128) → need 3d input
    // Actually ggml_conv_1d expects input (T, C_in) and kernel (K, C_in, C_out)
    // BUT ggml_conv_1d is actually defined for 1D convolution:
    //   a = kernel (KW, C_in, C_out)
    //   b = input  (IW, C_in)  → BUT actually needs 3D for batched: (IW, C_in, N)
    // For unbatched: just pass (IW, C_in, 1) or use the 2D input directly.
    // Let's check: mel is ne=(T_mel, n_mels) = (3000, 128).
    // conv1_w is ne=(3, 128, 1280) from the GGUF.
    // ggml_conv_1d_s1_ph computes 1D conv with stride=1, pad=half_kernel.
    // But we need stride=1, pad=1 which is half of kernel 3.
    // ggml_conv_1d output ne = (OL, OC, N). For unbatched 2D input the
    // batch dim N=1, so output is (OL, OC, 1). Bias (d=1280,) must be
    // reshaped to (1, d, 1) to broadcast over OL and N.
    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };

    ggml_tensor* cur = ggml_conv_1d(ctx0, m.audio.conv1_w, mel, /*s*/ 1, /*p*/ 1, /*d*/ 1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);
    // cur ne = (3000, 1280, 1)

    cur = ggml_conv_1d(ctx0, m.audio.conv2_w, cur, /*s*/ 2, /*p*/ 1, /*d*/ 1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);
    // cur ne = (1500, 1280, 1)

    // ggml_conv_1d output is (OL, OC, 1) = (1500, 1280, 1). We need (d, T_enc)
    // = (1280, 1500) for the norm/mul/matmul ops downstream, which all expect
    // ne[0] = feature_dim. Transpose:
    const int T_enc = T_mel / 2;
    cur = ggml_reshape_2d(ctx0, cur, T_enc, d);       // (1500, 1280) squeeze batch
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (1280, 1500) = (d, T_enc)

    // ---- Add learned positional embedding ----
    // embed_positions ggml ne = (d=1280, max_pos=1500) = (d, T_enc). Matches cur.
    cur = ggml_add(ctx0, cur, m.audio.embed_positions);

    // ---- 32 × Whisper encoder block ----
    for (int il = 0; il < n_layers; il++) {
        const auto& b = m.audio.blocks[il];
        ggml_tensor* residual = cur;

        // Pre-LN
        ggml_tensor* x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        x = ggml_add(ctx0, x, b.attn_norm_b);

        // Self-attention (biased Q, V, out_proj; NO bias on K — Whisper quirk).
        // No RoPE — learned positional embedding was added above.
        core_attn::EncoderSelfAttnParams eap;
        eap.n_heads = n_heads;
        eap.n_kv_heads = n_heads; // MHA
        eap.head_dim = head_dim;
        eap.n_kv_grp = 1;
        eap.attn_scale = attn_scale;
        eap.n_ctx_orig = 0;
        eap.rope_theta = 0.0f;
        ggml_tensor* attn = core_attn::encoder_self_attn(ctx0, x, b.attn_q_w, b.attn_q_b, b.attn_k_w,
                                                         /*k_b*/ nullptr, b.attn_v_w, b.attn_v_b, b.attn_out_w,
                                                         b.attn_out_b, /*positions*/ nullptr, /*mask*/ nullptr, eap);
        cur = ggml_add(ctx0, residual, attn);

        // Pre-LN + FFN (GELU, biased)
        residual = cur;
        x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        x = ggml_add(ctx0, x, b.ffn_norm_b);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_up_w, x), b.ffn_up_b);
        x = ggml_gelu_erf(ctx0, x);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_down_w, x), b.ffn_down_b);
        cur = ggml_add(ctx0, residual, x);
    }

    // ---- Final layer norm ----
    cur = ggml_norm(ctx0, cur, kLayerNormEps);
    cur = ggml_mul(ctx0, cur, m.audio.ln_post_w);
    cur = ggml_add(ctx0, cur, m.audio.ln_post_b);
    // cur ne = (T_enc=1500, d=1280)

    // ---- Projector: stack-4-frames + 2× Linear ----
    // Reshape (1500, 1280) → (375, 5120) = stack 4 adjacent frames.
    // In ggml memory (T_enc, d) row-major, 4 consecutive rows of 1280 become
    // one row of 5120. ggml_reshape_2d just relabels.
    cur = ggml_reshape_2d(ctx0, cur, proj_in, T_enc / 4);
    // linear_1: (5120 → 3072)
    cur = ggml_mul_mat(ctx0, m.projector.proj1, cur);
    cur = ggml_gelu_erf(ctx0, cur);
    // linear_2: (3072 → 3072)
    cur = ggml_mul_mat(ctx0, m.projector.proj2, cur);
    // cur ne = (proj_out=3072, 375)

    ggml_set_name(cur, "encoder_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// LLM forward graph (Stage V1)
//
// Pure text-only Llama 3 / Mistral forward, no audio injection, no KV cache.
// This is structurally a strict subset of qwen3_asr's build_llm_body:
//   - same SwiGLU + GQA + NEOX RoPE + RMSNorm pattern
//   - NO Q-norm/K-norm
//   - NO biases anywhere
//   - different dims (30 layers, d=3072, GQA 32/8, head_dim=128, FFN=8192)
//   - RoPE θ=1e8 instead of 1e6
// ===========================================================================

static const float kRmsEps = 1e-5f;

static ggml_cgraph* voxtral_build_graph_llm(voxtral_context* ctx, int n_tokens) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.llm_d_model;       // 3072
    const int n_q = (int)hp.llm_n_heads;     // 32
    const int n_kv = (int)hp.llm_n_kv_heads; // 8
    const int hd = (int)hp.llm_head_dim;     // 128
    const int n_kv_grp = n_q / n_kv;         // 4
    const float eps = hp.llm_rms_eps;
    const float theta = hp.llm_rope_theta;
    const int T = n_tokens;
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // ------- Inputs -------
    ggml_tensor* input_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(input_ids, "input_ids");
    ggml_set_input(input_ids);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    // ------- Token embedding lookup -------
    ggml_tensor* cur = ggml_get_rows(ctx0, m.llm.token_embd_w, input_ids);
    // cur ne = (d, T)

    // ------- 30 × Llama block -------
    for (uint32_t il = 0; il < hp.llm_n_layers; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        // ---- LN1 (RMSNorm + multiplicative weight, no bias) ----
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // ---- Q/K/V projections + reshape + NEOX RoPE + GQA expand +
        //      permute + flash-attn + output projection — all the
        //      Llama / Mistral boilerplate lives in core_attn now. ----
        core_attn::LlamaSelfAttnParams ap;
        ap.n_heads = n_q;
        ap.n_kv_heads = n_kv;
        ap.head_dim = hd;
        ap.n_kv_grp = n_kv_grp;
        ap.n_ctx_orig = (int)hp.llm_max_pos;
        ap.rope_theta = theta;
        ap.attn_scale = attn_scale;
        ggml_tensor* attn = core_attn::llama_self_attn(ctx0, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w,
                                                       positions, causal_mask, ap);
        cur = ggml_add(ctx0, residual, attn);

        // ---- FFN: down(silu(gate(x)) * up(x)) ----
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    // ------- Output norm + lm_head (last-token-only slice) -------
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" voxtral_context_params voxtral_context_default_params(void) {
    voxtral_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

extern "C" voxtral_context* voxtral_init_from_file(const char* path, voxtral_context_params params) {
    voxtral_context* ctx = new voxtral_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Try GPU backend first (Metal, CUDA, Vulkan...), fall back to CPU.
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu) {
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    }
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    }

    if (!voxtral_load_model(ctx->model, ctx->vocab, path, ctx->backend)) {
        delete ctx;
        return nullptr;
    }

    // Create backend scheduler once with worst-case node budget.
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    // PLAN #60d — runtime fused QKV for the LLM. Concat each layer's
    // q/k/v along the output axis at load time. Works for F16/F32 and
    // for row-wise quantized formats (Q4_K, Q4_0, Q5_K, Q8_0, ...) via
    // byte-concat. Routes through core_attn::kv_self_attn's qkv_w branch
    // for one matmul/layer instead of three.
    //
    // **Opt-in** (CRISPASR_VOXTRAL_FUSED_QKV=1) because A/B measurement on
    // JFK 11 s with Q4_K showed no measurable speedup vs the unfused path
    // (mean 16.37 s vs 16.43 s, within run-to-run noise). The voxtral4b
    // fuse delivered 7-8 % because its decode loop is much longer
    // (141 tokens incl. streaming-pad warmup) so the saved kernel-launch
    // overhead per step compounds. Voxtral-3B's normal-prompt decode is
    // ~30 tokens — the fuse savings disappear into the noise. Cost when
    // enabled: ~500 MB extra for the duplicated fused weights. Kept
    // opt-in so memory-tight deployments aren't penalised, but available
    // for long-form workloads where decode dominates.
    {
        const char* fuse_env = getenv("CRISPASR_VOXTRAL_FUSED_QKV");
        const bool fuse_enabled = (fuse_env != nullptr) && (atoi(fuse_env) != 0);
        auto& blocks = ctx->model.llm.blocks;
        bool can_fuse = fuse_enabled && !blocks.empty();
        if (can_fuse) {
            const ggml_type t0 = blocks[0].attn_q_w ? blocks[0].attn_q_w->type : GGML_TYPE_F32;
            for (auto& b : blocks) {
                if (!b.attn_q_w || !b.attn_k_w || !b.attn_v_w || b.attn_q_w->type != t0 ||
                    b.attn_k_w->type != t0 || b.attn_v_w->type != t0 ||
                    b.attn_q_w->ne[0] != b.attn_k_w->ne[0] || b.attn_q_w->ne[0] != b.attn_v_w->ne[0]) {
                    can_fuse = false;
                    break;
                }
            }
        }
        if (can_fuse) {
            const int hidden = (int)blocks[0].attn_q_w->ne[0];
            const int q_out = (int)blocks[0].attn_q_w->ne[1];
            const int kv_out = (int)blocks[0].attn_k_w->ne[1];
            const int qkv_out = q_out + 2 * kv_out;
            const ggml_type t0 = blocks[0].attn_q_w->type;
            ggml_init_params fgp = {ggml_tensor_overhead() * blocks.size() + 256, nullptr, true};
            ctx->fused_ctx = ggml_init(fgp);
            if (ctx->fused_ctx) {
                for (auto& b : blocks) {
                    b.attn_qkv_w = ggml_new_tensor_2d(ctx->fused_ctx, t0, hidden, qkv_out);
                }
                ctx->fused_buf = ggml_backend_alloc_ctx_tensors_from_buft(
                    ctx->fused_ctx, ggml_backend_get_default_buffer_type(ctx->backend));
                if (ctx->fused_buf) {
                    for (auto& b : blocks) {
                        const size_t qb = ggml_nbytes(b.attn_q_w);
                        const size_t kb = ggml_nbytes(b.attn_k_w);
                        const size_t vb = ggml_nbytes(b.attn_v_w);
                        std::vector<uint8_t> tmp(qb + kb + vb);
                        ggml_backend_tensor_get(b.attn_q_w, tmp.data(), 0, qb);
                        ggml_backend_tensor_get(b.attn_k_w, tmp.data() + qb, 0, kb);
                        ggml_backend_tensor_get(b.attn_v_w, tmp.data() + qb + kb, 0, vb);
                        ggml_backend_tensor_set(b.attn_qkv_w, tmp.data(), 0, tmp.size());
                    }
                    if (params.verbosity >= 1)
                        fprintf(stderr, "voxtral: fused QKV for %zu LLM layers (%d+%d+%d→%d, type=%s)\n",
                                blocks.size(), q_out, kv_out, kv_out, qkv_out, ggml_type_name(t0));
                } else {
                    ggml_free(ctx->fused_ctx);
                    ctx->fused_ctx = nullptr;
                    for (auto& b : blocks)
                        b.attn_qkv_w = nullptr;
                }
            }
        }
    }

    if (params.verbosity >= 1) {
        fprintf(stderr,
                "voxtral: loaded %s  (audio %u layers, llm %u layers, vocab %u, "
                "tekken %d specials + %d BPE)\n",
                path, ctx->model.hparams.audio_n_layers, ctx->model.hparams.llm_n_layers,
                ctx->model.hparams.llm_vocab_size, ctx->vocab.n_specials, ctx->vocab.n_vocab);
    }
    return ctx;
}

extern "C" void voxtral_free(voxtral_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->fused_buf)
        ggml_backend_buffer_free(ctx->fused_buf);
    if (ctx->fused_ctx)
        ggml_free(ctx->fused_ctx);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    // Free the primary backend last — buffers above were allocated against it,
    // and on Metal an unreleased backend leaves the residency set live and
    // trips ggml_metal_rsets_free's assert at process exit.
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" const uint8_t* voxtral_token_text(voxtral_context* ctx, int id, int* out_len) {
    if (!ctx) {
        if (out_len)
            *out_len = 0;
        return nullptr;
    }
    const auto& v = ctx->vocab;
    // Special tokens live at ranks [0, n_specials).
    if (id >= 0 && id < v.n_specials) {
        if (out_len)
            *out_len = (int)v.specials[id].size();
        return (const uint8_t*)v.specials[id].data();
    }
    // Regular vocab entries are stored at indices [n_specials, n_specials+n_vocab)
    // in the model's logical id space, but the rank_offset table uses a 0-based
    // index into the vocab blob. So translate: rank = id - n_specials.
    int r = id - v.n_specials;
    if (r < 0 || r >= (int)v.rank_offset.size()) {
        if (out_len)
            *out_len = 0;
        return nullptr;
    }
    if (out_len)
        *out_len = (int)v.rank_length[r];
    return v.tekken_vocab_blob.data() + v.rank_offset[r];
}

// ===========================================================================
// KV-cached LLM graph (Stage V3) — same pattern as qwen3_asr's build_graph_llm_kv
// ===========================================================================

static ggml_cgraph* voxtral_build_graph_llm_kv(voxtral_context* ctx, int n_past, int n_tokens) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.llm_d_model;
    const int n_q = (int)hp.llm_n_heads;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.llm_rms_eps;
    const float theta = hp.llm_rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = n_past + T;

    GGML_ASSERT(ctx->kv_k && ctx->kv_v && Lk <= ctx->kv_max_ctx);

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor* cur = embeds;

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.llm_max_pos,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 32.0f,
        /*rope_beta_slow*/ 1.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ 0.0f, // voxtral has no Q/K norm
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    for (uint32_t il = 0; il < hp.llm_n_layers; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // Decode path (T==1) passes no mask to flash-attn; prefill (T>1)
        // passes the causal mask. core_attn::kv_self_attn threads whichever
        // we give it down to ggml_flash_attn_ext.
        ggml_tensor* attn =
            core_attn::kv_self_attn(ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w,
                                    /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions,
                                    (T == 1) ? nullptr : causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, kvp,
                                    /*qkv_w*/ b.attn_qkv_w);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);
    if (T > 1)
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Tiny embed lookup graph
static ggml_cgraph* voxtral_build_graph_embed(voxtral_context* ctx, int n_tokens) {
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);
    ggml_tensor* out = ggml_get_rows(ctx0, ctx->model.llm.token_embd_w, ids);
    ggml_set_name(out, "embeds");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Public C API — KV cache + embed + run_llm_kv
// ===========================================================================

extern "C" bool voxtral_kv_init(voxtral_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    if (ctx->kv_k)
        return true;
    const auto& hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim, n_kv = (int)hp.llm_n_kv_heads, nl = (int)hp.llm_n_layers;
    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    ctx->kv_ctx = ggml_init(kp);
    // PLAN #60e: KV dtype from CRISPASR_KV_QUANT (default f16).
    const ggml_type kv_dtype = core_attn::kv_dtype_from_env("voxtral");
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_dtype, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_dtype, hd, max_ctx, n_kv, nl);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");
    size_t kb = ggml_nbytes(ctx->kv_k), vb = ggml_nbytes(ctx->kv_v);
    ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, kb + vb);
    if (!ctx->kv_buf) {
        fprintf(stderr, "voxtral: kv alloc failed\n");
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kb);
    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used = 0;
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "voxtral: kv cache %d MiB (hd=%d max=%d n_kv=%d nl=%d)\n", (int)((kb + vb) / 1048576), hd,
                max_ctx, n_kv, nl);
    return true;
}

extern "C" void voxtral_kv_reset(voxtral_context* ctx) {
    if (ctx)
        ctx->kv_n_used = 0;
}

extern "C" float* voxtral_embed_tokens(voxtral_context* ctx, const int32_t* input_ids, int n_tokens) {
    if (!ctx || !input_ids || n_tokens <= 0)
        return nullptr;
    const int d = (int)ctx->model.hparams.llm_d_model;
    ggml_cgraph* gf = voxtral_build_graph_embed(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input_ids"), input_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* r = (float*)malloc((size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)d * n_tokens * sizeof(float));
    return r;
}

extern "C" float* voxtral_run_llm_kv(voxtral_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                                     int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0 || !ctx->kv_k)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model, vocab = (int)hp.llm_vocab_size, Lk = n_past + n_tokens;

    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos[i] = n_past + i;
    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        mask.assign((size_t)Lk * n_tokens, ggml_fp32_to_fp16(0.0f));
        ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++)
            for (int k = n_past + q + 1; k < Lk; k++)
                mask[(size_t)q * Lk + k] = neg_inf;
    }

    ggml_cgraph* gf = voxtral_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "voxtral: failed to alloc llm_kv graph\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), inputs_embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), pos.data(), 0, pos.size() * sizeof(int32_t));
    if (n_tokens > 1)
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "voxtral: llm_kv graph compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out)
        return nullptr;
    ctx->kv_n_used = Lk;
    if (out_n_tokens)
        *out_n_tokens = 1;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    float* r = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)vocab * sizeof(float));
    return r;
}

extern "C" float* voxtral_run_encoder(voxtral_context* ctx, const float* mel_features, int n_mels, int T_mel,
                                      int* out_N, int* out_dim) {
    if (!ctx || !mel_features)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    if (n_mels != (int)hp.n_mels || T_mel != 3000) {
        fprintf(stderr, "voxtral: encoder expects (128, 3000) mel, got (%d, %d)\n", n_mels, T_mel);
        return nullptr;
    }

    ggml_cgraph* gf = voxtral_build_graph_encoder(ctx);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "voxtral: failed to alloc encoder graph\n");
        return nullptr;
    }

    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel_features, 0, (size_t)n_mels * T_mel * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "voxtral: encoder graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "encoder_out");
    if (!out) {
        fprintf(stderr, "voxtral: missing encoder_out\n");
        return nullptr;
    }
    const int pdim = (int)out->ne[0]; // 3072
    const int N = (int)out->ne[1];    // 375
    if (out_N)
        *out_N = N;
    if (out_dim)
        *out_dim = pdim;
    const size_t total = (size_t)pdim * N;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// Tekken BPE encoder (tiktoken-style rank-based byte BPE)
// ---------------------------------------------------------------------------

// Build reverse lookup table on first use.
static void tekken_build_reverse(voxtral_vocab& v) {
    if (v.reverse_built)
        return;
    v.bytes_to_rank.reserve(v.rank_offset.size());
    for (size_t r = 0; r < v.rank_offset.size(); r++) {
        std::string key((const char*)v.tekken_vocab_blob.data() + v.rank_offset[r], v.rank_length[r]);
        v.bytes_to_rank[key] = (int32_t)r;
    }
    v.reverse_built = true;
}

// Look up rank for a byte sequence. Returns -1 if not found.
static int32_t tekken_rank(const voxtral_vocab& v, const uint8_t* data, size_t len) {
    auto it = v.bytes_to_rank.find(std::string((const char*)data, len));
    return it != v.bytes_to_rank.end() ? it->second : -1;
}

// Encode a single pre-token (byte sequence) into BPE token IDs.
// tiktoken algorithm: start with individual bytes, repeatedly merge the
// adjacent pair with the lowest rank until no more merges are possible.
static void tekken_bpe_encode(const voxtral_vocab& v, const uint8_t* data, size_t len, std::vector<int32_t>& out) {
    if (len == 0)
        return;
    if (len == 1) {
        int32_t r = tekken_rank(v, data, 1);
        out.push_back(r >= 0 ? r + v.n_specials : 0);
        return;
    }

    // Start with individual bytes as "pieces"
    struct piece {
        size_t start;
        size_t len;
    };
    std::vector<piece> pieces(len);
    for (size_t i = 0; i < len; i++)
        pieces[i] = {i, 1};

    while (pieces.size() > 1) {
        // Find the pair with the lowest merge rank
        int32_t best_rank = INT32_MAX;
        size_t best_idx = SIZE_MAX;
        for (size_t i = 0; i + 1 < pieces.size(); i++) {
            size_t merged_len = pieces[i].len + pieces[i + 1].len;
            int32_t r = tekken_rank(v, data + pieces[i].start, merged_len);
            if (r >= 0 && r < best_rank) {
                best_rank = r;
                best_idx = i;
            }
        }
        if (best_idx == SIZE_MAX)
            break; // no more merges
        // Merge pieces[best_idx] and pieces[best_idx+1]
        pieces[best_idx].len += pieces[best_idx + 1].len;
        pieces.erase(pieces.begin() + best_idx + 1);
    }

    // Convert pieces to token IDs
    for (const auto& p : pieces) {
        int32_t r = tekken_rank(v, data + p.start, p.len);
        out.push_back(r >= 0 ? r + v.n_specials : 0);
    }
}

// Simple pre-tokenizer: split on whitespace boundaries in a way compatible
// with tiktoken's regex. For the transcription use case we primarily need to
// handle special tokens like [INST], [BEGIN_AUDIO] and simple text.
// This is a simplified version that handles: letters+digits as words,
// whitespace chunks, punctuation individually, and special bracket tokens.
static std::vector<std::string> tekken_pre_tokenize(const std::string& text) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = text[i];
        // Whitespace: group consecutive whitespace
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            size_t j = i;
            while (j < text.size() && (text[j] == ' ' || text[j] == '\t' || text[j] == '\n' || text[j] == '\r'))
                j++;
            out.push_back(text.substr(i, j - i));
            i = j;
        }
        // ASCII letter or digit: group with following letters/digits
        else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            // Prepend leading space if previous char was space (tiktoken style)
            size_t j = i;
            while (j < text.size()) {
                unsigned char d = text[j];
                if ((d >= 'A' && d <= 'Z') || (d >= 'a' && d <= 'z') || (d >= '0' && d <= '9'))
                    j++;
                else
                    break;
            }
            out.push_back(text.substr(i, j - i));
            i = j;
        }
        // UTF-8 multibyte: group entire codepoint
        else if (c >= 0x80) {
            size_t j = i + 1;
            while (j < text.size() && (text[j] & 0xC0) == 0x80)
                j++;
            // Keep grouping continuation letters (for CJK, accented, etc.)
            while (j < text.size() && ((unsigned char)text[j]) >= 0x80) {
                size_t k = j + 1;
                while (k < text.size() && (text[k] & 0xC0) == 0x80)
                    k++;
                j = k;
            }
            out.push_back(text.substr(i, j - i));
            i = j;
        }
        // Other (punctuation): single char
        else {
            out.push_back(text.substr(i, 1));
            i++;
        }
    }
    return out;
}

extern "C" int32_t* voxtral_tokenize(voxtral_context* ctx, const char* text, int* out_n_tokens) {
    if (!ctx || !text) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }
    auto& v = ctx->vocab;
    tekken_build_reverse(v);

    std::string input(text);
    std::vector<int32_t> ids;

    // Check for special tokens first — scan for exact matches
    size_t pos = 0;
    while (pos < input.size()) {
        // Try to match a special token at this position
        bool found_special = false;
        for (int si = 0; si < (int)v.specials.size(); si++) {
            const auto& sp = v.specials[si];
            if (sp.empty())
                continue;
            if (pos + sp.size() <= input.size() && input.compare(pos, sp.size(), sp) == 0) {
                ids.push_back(si);
                pos += sp.size();
                found_special = true;
                break;
            }
        }
        if (found_special)
            continue;

        // Find the next special token position (or end of string)
        size_t next_special = input.size();
        for (int si = 0; si < (int)v.specials.size(); si++) {
            const auto& sp = v.specials[si];
            if (sp.empty())
                continue;
            size_t f = input.find(sp, pos);
            if (f != std::string::npos && f < next_special)
                next_special = f;
        }

        // Pre-tokenize + BPE the non-special text segment
        std::string segment = input.substr(pos, next_special - pos);
        auto pre_tokens = tekken_pre_tokenize(segment);
        for (const auto& pt : pre_tokens) {
            tekken_bpe_encode(v, (const uint8_t*)pt.data(), pt.size(), ids);
        }
        pos = next_special;
    }

    if (ids.empty()) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }

    int32_t* result = (int32_t*)malloc(ids.size() * sizeof(int32_t));
    std::memcpy(result, ids.data(), ids.size() * sizeof(int32_t));
    if (out_n_tokens)
        *out_n_tokens = (int)ids.size();
    return result;
}

extern "C" float* voxtral_run_llm(voxtral_context* ctx, const int32_t* input_ids, int n_tokens, int* out_n_tokens,
                                  int* out_vocab_size) {
    if (!ctx || !input_ids || n_tokens <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int vocab = (int)hp.llm_vocab_size;

    // Positions [0, T)
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = i;

    // Causal mask: F16 (T, T), -inf above diagonal, 0 elsewhere.
    // ggml ne[0]=k (key, fast), ne[1]=q (query). mask[k > q] = -inf.
    std::vector<ggml_fp16_t> mask((size_t)n_tokens * n_tokens, ggml_fp32_to_fp16(0.0f));
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < n_tokens; q++) {
        for (int k = q + 1; k < n_tokens; k++) {
            mask[(size_t)q * n_tokens + k] = neg_inf;
        }
    }

    ggml_cgraph* gf = voxtral_build_graph_llm(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "voxtral: failed to alloc llm graph\n");
        return nullptr;
    }

    ggml_tensor* ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, input_ids, 0, (size_t)n_tokens * sizeof(int32_t));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));

    ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "voxtral: llm graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out) {
        fprintf(stderr, "voxtral: missing logits tensor\n");
        return nullptr;
    }
    // Last-token-only output: returns (vocab,) for the final position.
    if (out_n_tokens)
        *out_n_tokens = 1;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)vocab * sizeof(float));
    return result;
}
