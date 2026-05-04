// voxtral4b.cpp — Mistral Voxtral-Mini-4B-Realtime-2602 ggml runtime
//
// Key architectural differences from voxtral.cpp (3B):
//   - Audio encoder: RoPE, SwiGLU FFN, RMSNorm, sliding window (750)
//   - LLM: 26 layers, FFN=9216, SWA(8192), adaptive RMSNorm, tied embeddings
//   - Same projector topology (stack-4-frames + 2×Linear)

#include "voxtral4b.h"

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
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct voxtral4b_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 400;
    uint32_t hop_length = 160;

    // Audio encoder (RoPE + SwiGLU + RMSNorm)
    uint32_t audio_n_layers = 32;
    uint32_t audio_d_model = 1280;
    uint32_t audio_n_heads = 32;
    uint32_t audio_head_dim = 64;
    uint32_t audio_ff_dim = 5120;
    uint32_t audio_max_pos = 1500;
    float audio_rope_theta = 1e6f;
    uint32_t audio_swa = 750;

    // Projector
    uint32_t proj_in_dim = 5120;
    uint32_t proj_out_dim = 3072;
    uint32_t proj_frame_stack = 4;

    // LLM (Llama-style + ada_rms_norm + SWA + tied embed)
    uint32_t llm_n_layers = 26;
    uint32_t llm_d_model = 3072;
    uint32_t llm_n_heads = 32;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 9216;
    float llm_rope_theta = 1e6f;
    float llm_rms_eps = 1e-5f;
    uint32_t llm_vocab_size = 131072;
    uint32_t llm_max_pos = 131072;
    uint32_t llm_swa = 8192;
    uint32_t ada_norm_dim = 32;
    bool tied_embeddings = true;

    uint32_t audio_token_id = 24;
};

// ===========================================================================
// Model tensors
// ===========================================================================

struct voxtral4b_audio_block {
    ggml_tensor* attn_norm_w = nullptr; // RMSNorm (no bias)
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr; // no bias
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    ggml_tensor* ffn_norm_w = nullptr; // RMSNorm (no bias)
    // SwiGLU: gate + up + down
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
};

struct voxtral4b_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    // Optional fused QKV (PLAN #7 phase 2). When non-null, the LLM forward
    // takes the single-matmul path through `core_attn::kv_self_attn`'s
    // `qkv_w` branch instead of three separate Q/K/V matmuls. Built at load
    // time by byte-concatenating the per-projection weight tensors along
    // the output axis (works for F16/F32 and quantized types since each
    // output row is independent in row-wise quantized formats).
    ggml_tensor* attn_qkv_w = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    // Adaptive RMSNorm
    ggml_tensor* ada_down_w = nullptr; // (ada_dim, d_model)
    ggml_tensor* ada_up_w = nullptr;   // (d_model, ada_dim)
};

struct voxtral4b_model {
    voxtral4b_hparams hparams;

    struct {
        ggml_tensor* conv1_w = nullptr;
        ggml_tensor* conv1_b = nullptr;
        ggml_tensor* conv2_w = nullptr;
        ggml_tensor* conv2_b = nullptr;
        // No embed_positions — RoPE instead
        ggml_tensor* ln_post_w = nullptr; // RMSNorm post-encoder
        ggml_tensor* mel_filters = nullptr;
        ggml_tensor* mel_window = nullptr;
        std::vector<voxtral4b_audio_block> blocks;
    } audio;

    struct {
        ggml_tensor* proj1 = nullptr;
        ggml_tensor* proj2 = nullptr;
    } projector;

    struct {
        ggml_tensor* token_embd_w = nullptr;
        ggml_tensor* output_norm_w = nullptr;
        // No output_w — tied to token_embd_w
        std::vector<voxtral4b_llm_block> blocks;
    } llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    // PLAN #69a: optional second buffer for layers spilled to CPU.
    // Non-null only when CRISPASR_N_GPU_LAYERS triggered a split load.
    ggml_backend_buffer_t buf_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

// Vocab (same structure as 3B)
struct voxtral4b_vocab {
    std::vector<uint8_t> tekken_vocab_blob;
    std::vector<uint32_t> rank_offset;
    std::vector<uint32_t> rank_length;
    std::vector<std::string> specials;
    std::unordered_map<std::string, int32_t> special_to_rank;
    int n_specials = 0;
    int n_vocab = 0;
    std::string pre_pattern;
    std::unordered_map<std::string, int32_t> bytes_to_rank;
    bool reverse_built = false;
};

struct voxtral4b_context {
    voxtral4b_context_params params;
    voxtral4b_model model;
    voxtral4b_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;

    // PLAN #7 phase 2: fused QKV LLM-side. Allocated once at load time.
    ggml_context* fused_ctx = nullptr;
    ggml_backend_buffer_t fused_buf = nullptr;

    int n_threads = 4;
    int delay_tokens = 6;          // 480ms default
    std::vector<float> ada_scales; // (n_layers × d_model), precomputed
};

// ===========================================================================
// GGUF loader
// ===========================================================================

#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// Parse a transformer-block layer index out of a tensor name. Returns
// -1 if the name is not a layered tensor (audio encoder, projection,
// embeddings, output norm). Layered tensors look like "blk.<N>.<field>".
static int voxtral4b_layer_of(const char* name) {
    if (!name)
        return -1;
    constexpr const char* kPfx = "blk.";
    constexpr size_t kPfxLen = 4;
    if (std::strncmp(name, kPfx, kPfxLen) != 0)
        return -1;
    char* end = nullptr;
    long il = std::strtol(name + kPfxLen, &end, 10);
    if (!end || *end != '.' || il < 0)
        return -1;
    return (int)il;
}

struct voxtral4b_split_ctx {
    int n_gpu_layers; // layers [0..n_gpu_layers) on GPU; rest on CPU
};

static bool voxtral4b_is_gpu_tensor(const char* tensor_name, void* user) {
    auto* sc = static_cast<voxtral4b_split_ctx*>(user);
    const int il = voxtral4b_layer_of(tensor_name);
    if (il < 0)
        return true; // non-layered tensors (audio enc, proj, embd, output_norm) stay on GPU
    return il < sc->n_gpu_layers;
}

static bool voxtral4b_load_model(voxtral4b_model& model, voxtral4b_vocab& vocab, const char* path,
                                 ggml_backend_t backend, ggml_backend_t backend_cpu) {
    // Pass 1: metadata
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;
        auto& hp = model.hparams;

        hp.audio_n_layers = core_gguf::kv_u32(gctx, "voxtral4b.audio.n_layers", hp.audio_n_layers);
        hp.audio_d_model = core_gguf::kv_u32(gctx, "voxtral4b.audio.d_model", hp.audio_d_model);
        hp.audio_n_heads = core_gguf::kv_u32(gctx, "voxtral4b.audio.n_heads", hp.audio_n_heads);
        hp.audio_head_dim = core_gguf::kv_u32(gctx, "voxtral4b.audio.head_dim", hp.audio_head_dim);
        hp.audio_ff_dim = core_gguf::kv_u32(gctx, "voxtral4b.audio.ff_dim", hp.audio_ff_dim);
        hp.audio_max_pos = core_gguf::kv_u32(gctx, "voxtral4b.audio.max_pos", hp.audio_max_pos);
        hp.audio_rope_theta = core_gguf::kv_f32(gctx, "voxtral4b.audio.rope_theta", hp.audio_rope_theta);
        hp.audio_swa = core_gguf::kv_u32(gctx, "voxtral4b.audio.sliding_window", hp.audio_swa);

        hp.proj_in_dim = core_gguf::kv_u32(gctx, "voxtral4b.proj.in_dim", hp.proj_in_dim);
        hp.proj_out_dim = core_gguf::kv_u32(gctx, "voxtral4b.proj.out_dim", hp.proj_out_dim);
        hp.proj_frame_stack = core_gguf::kv_u32(gctx, "voxtral4b.proj.frame_stack", hp.proj_frame_stack);

        hp.llm_n_layers = core_gguf::kv_u32(gctx, "voxtral4b.llm.n_layers", hp.llm_n_layers);
        hp.llm_d_model = core_gguf::kv_u32(gctx, "voxtral4b.llm.d_model", hp.llm_d_model);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "voxtral4b.llm.n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "voxtral4b.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "voxtral4b.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "voxtral4b.llm.ff_dim", hp.llm_ff_dim);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "voxtral4b.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "voxtral4b.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "voxtral4b.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "voxtral4b.llm.max_pos", hp.llm_max_pos);
        hp.llm_swa = core_gguf::kv_u32(gctx, "voxtral4b.llm.sliding_window", hp.llm_swa);
        hp.ada_norm_dim = core_gguf::kv_u32(gctx, "voxtral4b.llm.ada_norm_dim", hp.ada_norm_dim);
        hp.audio_token_id = core_gguf::kv_u32(gctx, "voxtral4b.audio_token_id", hp.audio_token_id);

        // Tekken tokenizer
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

        core_gguf::free_metadata(gctx);
    }

    // Pass 2: load tensors via shared helper.
    // PLAN #69a: when CRISPASR_N_GPU_LAYERS is set and < total layers,
    // route layers [N..total) onto the CPU backend so VRAM-tight users
    // can fit models larger than their GPU. -1 (default) or any value
    // >= n_layers preserves the legacy single-backend load.
    core_gguf::WeightLoad wl;
    int n_gpu_layers_env = -1;
    if (const char* s = std::getenv("CRISPASR_N_GPU_LAYERS")) {
        n_gpu_layers_env = std::atoi(s);
    }
    const int total_layers = (int)model.hparams.llm_n_layers;
    const bool do_split = backend_cpu && backend_cpu != backend && n_gpu_layers_env >= 0 &&
                          n_gpu_layers_env < total_layers;
    if (do_split) {
        voxtral4b_split_ctx sc{n_gpu_layers_env};
        if (!core_gguf::load_weights_split(path, backend, backend_cpu, voxtral4b_is_gpu_tensor, &sc, "voxtral4b",
                                           wl)) {
            return false;
        }
        fprintf(stderr, "voxtral4b: layer offload: gpu=[0,%d), cpu=[%d,%d) (CRISPASR_N_GPU_LAYERS=%d)\n",
                n_gpu_layers_env, n_gpu_layers_env, total_layers, n_gpu_layers_env);
    } else {
        if (!core_gguf::load_weights(path, backend, "voxtral4b", wl)) {
            return false;
        }
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.buf_cpu = wl.buf_cpu;
    model.tensors = std::move(wl.tensors);

    // Bind tensors
    auto get = [&](const std::string& n) -> ggml_tensor* {
        auto it = model.tensors.find(n);
        return it != model.tensors.end() ? it->second : nullptr;
    };
    auto require = [&](const std::string& n) -> ggml_tensor* {
        auto* t = get(n);
        if (!t)
            fprintf(stderr, "voxtral4b: missing tensor '%s'\n", n.c_str());
        return t;
    };

    auto& a = model.audio;
    a.conv1_w = require("audio.conv.1.weight");
    a.conv1_b = require("audio.conv.1.bias");
    a.conv2_w = require("audio.conv.2.weight");
    a.conv2_b = require("audio.conv.2.bias");
    a.ln_post_w = require("audio.ln_post.weight");
    a.mel_filters = get("audio.mel_filters");
    a.mel_window = get("audio.mel_window");

    a.blocks.resize(model.hparams.audio_n_layers);
    for (uint32_t il = 0; il < model.hparams.audio_n_layers; il++) {
        auto pfx = "audio.blk." + std::to_string(il) + ".";
        auto& b = a.blocks[il];
        b.attn_norm_w = require(pfx + "attn_norm.weight");
        b.attn_q_w = require(pfx + "attn_q.weight");
        b.attn_q_b = get(pfx + "attn_q.bias");
        b.attn_k_w = require(pfx + "attn_k.weight");
        b.attn_v_w = require(pfx + "attn_v.weight");
        b.attn_v_b = get(pfx + "attn_v.bias");
        b.attn_out_w = require(pfx + "attn_out.weight");
        b.attn_out_b = get(pfx + "attn_out.bias");
        b.ffn_norm_w = require(pfx + "ffn_norm.weight");
        b.ffn_gate_w = require(pfx + "ffn_gate.weight");
        b.ffn_up_w = require(pfx + "ffn_up.weight");
        b.ffn_down_w = require(pfx + "ffn_down.weight");
        b.ffn_down_b = get(pfx + "ffn_down.bias");
    }

    // Tekken vocab blob
    {
        ggml_tensor* vt = get("tokenizer.tekken.vocab_tensor");
        if (vt) {
            size_t n = (size_t)vt->ne[0];
            std::vector<float> f32(n);
            ggml_backend_tensor_get(vt, f32.data(), 0, n * sizeof(float));
            vocab.tekken_vocab_blob.resize(n);
            for (size_t i = 0; i < n; i++)
                vocab.tekken_vocab_blob[i] = (uint8_t)(int)f32[i];
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
    p.proj1 = require("proj1.weight");
    p.proj2 = require("proj2.weight");

    auto& l = model.llm;
    l.token_embd_w = require("token_embd.weight");
    l.output_norm_w = require("output_norm.weight");
    // No output_w — tied to token_embd_w

    l.blocks.resize(model.hparams.llm_n_layers);
    for (uint32_t il = 0; il < model.hparams.llm_n_layers; il++) {
        auto pfx = "blk." + std::to_string(il) + ".";
        auto& b = l.blocks[il];
        b.attn_norm_w = require(pfx + "attn_norm.weight");
        b.attn_q_w = require(pfx + "attn_q.weight");
        b.attn_k_w = require(pfx + "attn_k.weight");
        b.attn_v_w = require(pfx + "attn_v.weight");
        b.attn_out_w = require(pfx + "attn_output.weight");
        b.ffn_norm_w = require(pfx + "ffn_norm.weight");
        b.ffn_gate_w = require(pfx + "ffn_gate.weight");
        b.ffn_up_w = require(pfx + "ffn_up.weight");
        b.ffn_down_w = require(pfx + "ffn_down.weight");
        b.ada_down_w = get(pfx + "ada_norm_down.weight");
        b.ada_up_w = get(pfx + "ada_norm_up.weight");
    }

    // Count actually-loaded tensors (non-null required tensors)
    int n_missing = 0;
    auto check = [&](ggml_tensor* t, const char* name) {
        if (!t) {
            n_missing++;
        }
    };
    check(p.proj1, "proj1");
    check(p.proj2, "proj2");
    check(l.token_embd_w, "token_embd");
    check(l.output_norm_w, "output_norm");
    for (uint32_t il = 0; il < model.hparams.llm_n_layers; il++) {
        const auto& b = l.blocks[il];
        check(b.attn_norm_w, "");
        check(b.attn_q_w, "");
        check(b.attn_k_w, "");
        check(b.attn_v_w, "");
        check(b.attn_out_w, "");
        check(b.ffn_norm_w, "");
        check(b.ffn_gate_w, "");
        check(b.ffn_up_w, "");
        check(b.ffn_down_w, "");
    }
    fprintf(stderr, "voxtral4b: loaded %d audio tensors + %d LLM tensors\n",
            (int)(model.hparams.audio_n_layers * 13 + 5), (int)(model.hparams.llm_n_layers * 11 + 2));
    if (n_missing > 0) {
        fprintf(stderr,
                "voxtral4b: ERROR: %d required tensors missing — GGUF file is corrupt or truncated.\n"
                "           If the file is > 2 GB, this may be a Windows fseek overflow bug.\n"
                "           Re-download the model or update CrispASR to the latest version.\n",
                n_missing);
        return false;
    }
    return true;
}

// ===========================================================================
// Mel / FFT (same as 3B — CPU-only)
// ===========================================================================

static void voxtral4b_dft(const float* in, int N, float* out) {
    for (int k = 0; k < N; k++) {
        double re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * k * n / N;
            re += in[n] * cos(a);
            im += in[n] * sin(a);
        }
        out[2 * k] = (float)re;
        out[2 * k + 1] = (float)im;
    }
}

static void voxtral4b_fft(float* in, int N, float* out) {
    if (N <= 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }
    if (N % 2 != 0) {
        voxtral4b_dft(in, N, out);
        return;
    }
    int half = N / 2;
    std::vector<float> even(half), odd(half);
    for (int i = 0; i < half; i++) {
        even[i] = in[2 * i];
        odd[i] = in[2 * i + 1];
    }
    std::vector<float> E(2 * half), O(2 * half);
    voxtral4b_fft(even.data(), half, E.data());
    voxtral4b_fft(odd.data(), half, O.data());
    for (int k = 0; k < half; k++) {
        double a = -2.0 * M_PI * k / N;
        float wr = (float)cos(a), wi = (float)sin(a);
        float tre = wr * O[2 * k] - wi * O[2 * k + 1];
        float tim = wr * O[2 * k + 1] + wi * O[2 * k];
        out[2 * k] = E[2 * k] + tre;
        out[2 * k + 1] = E[2 * k + 1] + tim;
        out[2 * (k + half)] = E[2 * k] - tre;
        out[2 * (k + half) + 1] = E[2 * k + 1] - tim;
    }
}

#include "core/mel.h"
#include "core/ffn.h"
#include "core/attention.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// Same in-place FFT quirk as voxtral 3B: voxtral4b_fft writes into its
// input buffer during recursion, so we wrap it with a thread-local
// scratch copy to satisfy core_mel::FftR2C's const input contract.
static void voxtral4b_fft_wrapper(const float* in, int N, float* out) {
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size() < 4 * N)
        scratch_in.assign((size_t)4 * N, 0.0f);
    if ((int)scratch_out.size() < 8 * N)
        scratch_out.assign((size_t)8 * N, 0.0f);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    voxtral4b_fft(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}

extern "C" float* voxtral4b_compute_mel(voxtral4b_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                        int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    if (!ctx->model.audio.mel_filters || !ctx->model.audio.mel_window)
        return nullptr;
    const int n_fft = 400, hop = 160, n_mels = 128, n_freqs = 201;

    std::vector<float> hann(n_fft);
    ggml_backend_tensor_get(ctx->model.audio.mel_window, hann.data(), 0, n_fft * sizeof(float));
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->model.audio.mel_filters, filt.data(), 0, filt.size() * sizeof(float));

    // VoxtralRealtime specifics:
    //  - Whisper drops the last frame (HF convention)
    //  - If the remaining T is odd, also drop the first frame (stride-2 conv)
    //  - Log guard is log10(max(x, 1e-10)); matmul is double-accumulated
    //  - Normalization uses a fixed global_log_mel_max=1.5, not per-audio max
    //  - Filterbank is stored [n_freqs, n_mels]
    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Log10;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.norm = core_mel::Normalization::GlobalClipFixed;
    p.layout = core_mel::Layout::MelsTime;
    p.fb_layout = core_mel::FbLayout::FreqsMels;
    p.matmul = core_mel::MatmulPrecision::Double;
    p.log_eps = 1e-10f;
    p.fixed_max = 1.5f;
    p.center_pad = true;
    p.drop_last_frame = true;
    p.drop_first_frame_if_odd = true;

    int T_ret = 0;
    auto mel = core_mel::compute(samples, n_samples, hann.data(), n_fft, filt.data(), n_freqs, voxtral4b_fft_wrapper, p,
                                 T_ret);

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
// Audio encoder graph — RoPE + SwiGLU + RMSNorm + sliding window
// ===========================================================================

static const float kRmsEps = 1e-5f;

static ggml_cgraph* voxtral4b_build_graph_encoder(voxtral4b_context* ctx, int T_mel) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.audio_d_model;
    const int n_heads = (int)hp.audio_n_heads;
    const int head_dim = (int)hp.audio_head_dim;
    const int n_layers = (int)hp.audio_n_layers;
    const int proj_in = (int)hp.proj_in_dim;
    const int n_mels = (int)hp.n_mels;
    // hp.audio_swa is consumed later on the CPU-side swa_mask build
    // (line ~1020). Read it there directly rather than stashing it here,
    // so the build stays warning-free.
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    ggml_init_params ip = {
        ctx->compute_meta.size(),
        ctx->compute_meta.data(),
        true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: mel spectrogram (n_mels=128, T_mel=3000) F32
    // ggml layout: ne[0]=T_mel, ne[1]=n_mels → row-major (T_mel, n_mels)
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // Causal Conv1d front-end: left-pad only (padding_total = kernel_size - stride)
    // Conv0: k=3, s=1 → left_pad=2, Conv1: k=3, s=2 → left_pad=1
    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };

    // Causal padding for conv0: pad 2 zeros on the left of the time dimension
    // mel is (T_mel, n_mels), conv_1d treats ne[0] as time
    // ggml_pad pads at the end; for left-padding we use ggml_concat with a zero tensor
    ggml_tensor* pad0 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, n_mels);
    ggml_set_name(pad0, "conv0_lpad");
    ggml_set_input(pad0);                                      // will be set to zeros
    ggml_tensor* mel_padded = ggml_concat(ctx0, pad0, mel, 0); // concat along dim 0 (time)

    ggml_tensor* cur = ggml_conv_1d(ctx0, m.audio.conv1_w, mel_padded, 1, 0, 1); // pad=0!
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);

    // Causal padding for conv1: pad 1 zero on the left
    ggml_tensor* pad1 = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, d, 1);
    ggml_set_name(pad1, "conv1_lpad");
    ggml_set_input(pad1); // will be set to zeros
    // cur is (T, d, 1) from conv_1d output; concat a (1, d, 1) zero on the left
    cur = ggml_concat(ctx0, ggml_reshape_3d(ctx0, pad1, 1, d, 1), cur, 0);

    cur = ggml_conv_1d(ctx0, m.audio.conv2_w, cur, 2, 0, 1); // pad=0!
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);

    // Output length: conv0 out = T_mel (same with causal pad), conv1 out = ceil(T_mel/2)
    const int T_enc = (T_mel + 1) / 2; // ceil division for stride 2
    cur = ggml_reshape_2d(ctx0, cur, T_enc, d);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (d, T_enc)

    // Debug: name the conv stem output for extraction
    ggml_set_name(cur, "conv_stem_out");
    ggml_build_forward_expand(gf, cur); // ensure it's computed

    // RoPE positions for encoder
    ggml_tensor* pos_enc = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_enc);
    ggml_set_name(pos_enc, "enc_positions");
    ggml_set_input(pos_enc);

    // Causal attention mask (ALWAYS required — encoder is causal, not bidirectional).
    // When T_enc > swa, also apply sliding window restriction.
    ggml_tensor* swa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T_enc, T_enc);
    ggml_set_name(swa_mask, "swa_mask");
    ggml_set_input(swa_mask);

    // 32 × encoder blocks
    for (int il = 0; il < n_layers; il++) {
        const auto& b = m.audio.blocks[il];
        ggml_tensor* residual = cur;

        // Pre-RMSNorm
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, kRmsEps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // Self-attention via shared helper (no cont after permute to
        // match the original voxtral4b graph structure).
        core_attn::EncoderSelfAttnParams ap = {};
        ap.n_heads = n_heads;
        ap.n_kv_heads = n_heads;
        ap.head_dim = head_dim;
        ap.n_kv_grp = 1;
        ap.attn_scale = attn_scale;
        ap.n_ctx_orig = 0;
        ap.rope_theta = hp.audio_rope_theta;
        ap.permute_cont = false;
        ggml_tensor* attn =
            core_attn::encoder_self_attn(ctx0, x, b.attn_q_w, b.attn_q_b, b.attn_k_w, nullptr, // no K bias
                                         b.attn_v_w, b.attn_v_b, b.attn_out_w, b.attn_out_b, pos_enc, swa_mask, ap);
        cur = ggml_add(ctx0, residual, attn);

        // FFN: Pre-RMSNorm + SwiGLU (audio encoder may carry an optional
        // down bias on some checkpoints — swiglu_down_bias() no-ops it
        // when b.ffn_down_b is null).
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, kRmsEps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* ffn = core_ffn::swiglu_down_bias(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w, b.ffn_down_b);
        cur = ggml_add(ctx0, residual, ffn);
    }

    // Final RMSNorm
    cur = ggml_rms_norm(ctx0, cur, kRmsEps);
    cur = ggml_mul(ctx0, cur, m.audio.ln_post_w);

    // Projector: stack-4-frames + 2× Linear + GELU
    cur = ggml_reshape_2d(ctx0, cur, proj_in, T_enc / 4);
    cur = ggml_mul_mat(ctx0, m.projector.proj1, cur);
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_mul_mat(ctx0, m.projector.proj2, cur);

    ggml_set_name(cur, "encoder_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Ada-RMSNorm time conditioning
//
// The adaptive RMSNorm takes a sinusoidal time embedding of `delay_tokens`,
// projects it through a small bottleneck (3072→32→3072), and scales the
// post-attention hidden state: `h = h * (1 + scale)`.
// ===========================================================================

// Compute sinusoidal time embedding (same as voxtral.c / RoPE-style freqs)
static void voxtral4b_compute_t_cond(float* out, int d, float t_value) {
    int half = d / 2;
    float log_theta = std::log(10000.0f);
    for (int i = 0; i < half; i++) {
        float inv_freq = std::exp(-log_theta * (float)i / (float)half);
        float emb = t_value * inv_freq;
        out[i] = std::cos(emb);
        out[i + half] = std::sin(emb);
    }
}

// Precompute per-layer ada_scale: scale[l] = ada_up(gelu(ada_down(t_cond)))
// Returns (n_layers, d_model) F32 on CPU.
static std::vector<float> voxtral4b_compute_ada_scales(voxtral4b_context* ctx, int delay_tokens) {
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int ada_dim = (int)hp.ada_norm_dim;
    const int n_layers = (int)hp.llm_n_layers;

    // t_cond: sinusoidal embedding of delay_tokens
    std::vector<float> t_cond(d);
    voxtral4b_compute_t_cond(t_cond.data(), d, (float)delay_tokens);

    std::vector<float> all_scales((size_t)n_layers * d, 0.0f);

    for (int il = 0; il < n_layers; il++) {
        const auto& b = ctx->model.llm.blocks[il];
        if (!b.ada_down_w || !b.ada_up_w)
            continue;

        // Download weights to CPU
        std::vector<float> ada_down((size_t)ada_dim * d);
        std::vector<float> ada_up((size_t)d * ada_dim);
        ggml_backend_tensor_get(b.ada_down_w, ada_down.data(), 0, ada_down.size() * sizeof(float));
        ggml_backend_tensor_get(b.ada_up_w, ada_up.data(), 0, ada_up.size() * sizeof(float));

        // hidden = ada_down @ t_cond  (ada_dim × d) @ (d,) → (ada_dim,)
        std::vector<float> hidden(ada_dim, 0.0f);
        for (int i = 0; i < ada_dim; i++) {
            double s = 0;
            for (int j = 0; j < d; j++)
                s += (double)ada_down[(size_t)i * d + j] * t_cond[j];
            hidden[i] = (float)s;
        }

        // GELU
        for (int i = 0; i < ada_dim; i++) {
            float x = hidden[i];
            hidden[i] = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
        }

        // scale = 1 + ada_up @ hidden  (precompute the 1+ for direct mul in graph)
        float* scale = all_scales.data() + (size_t)il * d;
        for (int i = 0; i < d; i++) {
            double s = 0;
            for (int j = 0; j < ada_dim; j++)
                s += (double)ada_up[(size_t)i * ada_dim + j] * hidden[j];
            scale[i] = 1.0f + (float)s;
        }
    }

    return all_scales;
}

// ===========================================================================
// LLM KV-cached graph — 26 layers, FFN=9216, SWA(8192), tied embeddings,
// ada_rms_norm time conditioning
// ===========================================================================

static ggml_cgraph* voxtral4b_build_graph_llm_kv(voxtral4b_context* ctx, int n_past, int n_tokens) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.llm_d_model;
    const int n_q = (int)hp.llm_n_heads;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;
    const int n_layers = (int)hp.llm_n_layers;

    ggml_init_params ip = {
        ctx->compute_meta.size(),
        ctx->compute_meta.data(),
        true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_tokens);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past + n_tokens, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    // Ada-scale per layer: (n_layers, d) — precomputed on CPU, passed as input
    ggml_tensor* ada_scales = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_layers);
    ggml_set_name(ada_scales, "ada_scales");
    ggml_set_input(ada_scales);

    ggml_tensor* cur = embeds;

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_q / n_kv,
        /*n_ctx_orig*/ 0,
        /*rope_theta*/ hp.llm_rope_theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ 1.0f / std::sqrt((float)hd),
        /*qk_norm_eps*/ 0.0f, // no Q/K norm
        /*gqa_mode*/ core_attn::GQA_MANUAL_NOCONT,
    };

    for (int il = 0; il < n_layers; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        // Pre-RMSNorm
        cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        cur = ggml_mul(ctx0, cur, b.attn_norm_w);

        // KV-cached GQA self-attention — shared core_attn helper. voxtral4b
        // uses RoPE n_ctx_orig=0 and the manual-no-cont GQA mode, both of
        // which diverge from the other models and are therefore per-model
        // knobs in KvSelfAttnParams.
        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, cur, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_out_w,
                                                    /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions, causal_mask,
                                                    ctx->kv_k, ctx->kv_v, il, n_past, kvp,
                                                    /*qkv_w*/ b.attn_qkv_w);
        cur = ggml_add(ctx0, residual, attn);

        // FFN: Post-attention RMSNorm + ada_rms_norm conditioning + SwiGLU
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        cur = ggml_mul(ctx0, cur, b.ffn_norm_w);

        // Ada-scale: cur = cur * (1 + scale[il])  — precomputed as (1+scale) in ada_scales
        {
            ggml_tensor* scale = ggml_view_1d(ctx0, ada_scales, d, (size_t)il * d * sizeof(float));
            cur = ggml_mul(ctx0, cur, scale);
        }

        // cur here is the pre-FFN norm + ffn_norm_w scale + ada-scale.
        ggml_tensor* ffn = core_ffn::swiglu(ctx0, cur, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, ffn);
    }

    // Final RMSNorm
    cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);

    // LM head — tied to token_embd (transposed)
    if (n_tokens > 1) {
        // Only take the last token's hidden state for logits
        cur = ggml_view_1d(ctx0, cur, d, (size_t)(n_tokens - 1) * d * sizeof(float));
        cur = ggml_reshape_2d(ctx0, cur, d, 1);
    }
    cur = ggml_mul_mat(ctx0, m.llm.token_embd_w, cur); // tied!

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Embed graph (same as 3B)
// ===========================================================================

static ggml_cgraph* voxtral4b_build_graph_embed(voxtral4b_context* ctx, int n_tokens) {
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
// Public API
// ===========================================================================

extern "C" struct voxtral4b_context_params voxtral4b_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1, /*use_gpu=*/true};
}

extern "C" struct voxtral4b_context* voxtral4b_init_from_file(const char* path,
                                                              struct voxtral4b_context_params params) {
    auto* ctx = new voxtral4b_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!voxtral4b_load_model(ctx->model, ctx->vocab, path, ctx->backend, ctx->backend_cpu)) {
        delete ctx;
        return nullptr;
    }

    // Create scheduler once
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    // Precompute ada_rms_norm scales from delay_tokens
    ctx->ada_scales = voxtral4b_compute_ada_scales(ctx, ctx->delay_tokens);
    if (params.verbosity >= 1)
        fprintf(stderr, "voxtral4b: ada_scales computed for delay=%d (%d ms)\n", ctx->delay_tokens,
                ctx->delay_tokens * 80);

    // PLAN #7 phase 2 — runtime fused QKV for the LLM. Concat each layer's
    // q/k/v along the output axis into a single (d_model, q_dim+2*kv_dim)
    // tensor. Works for F16/F32 directly and for row-wise quantized formats
    // (Q4_K, Q4_0, Q8_0, etc.) by byte-concat: each output row is a self-
    // contained quantized block group, so concatenation along the output
    // axis is a pure memcpy. Skipped if any layer is missing q/k/v or has
    // mismatched types/input-dims.
    // Opt-out: set CRISPASR_VOXTRAL4B_FUSED_QKV=0.
    {
        const char* fuse_env = getenv("CRISPASR_VOXTRAL4B_FUSED_QKV");
        const bool fuse_enabled = (fuse_env == nullptr) || (atoi(fuse_env) != 0);
        auto& blocks = ctx->model.llm.blocks;
        bool can_fuse = fuse_enabled && !blocks.empty();
        if (can_fuse) {
            const ggml_type t0 = blocks[0].attn_q_w ? blocks[0].attn_q_w->type : GGML_TYPE_F32;
            for (auto& b : blocks) {
                if (!b.attn_q_w || !b.attn_k_w || !b.attn_v_w || b.attn_q_w->type != t0 || b.attn_k_w->type != t0 ||
                    b.attn_v_w->type != t0 || b.attn_q_w->ne[0] != b.attn_k_w->ne[0] ||
                    b.attn_q_w->ne[0] != b.attn_v_w->ne[0]) {
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
                        fprintf(stderr, "voxtral4b: fused QKV for %zu LLM layers (%d+%d+%d→%d, type=%s)\n",
                                blocks.size(), q_out, kv_out, kv_out, qkv_out, ggml_type_name(t0));
                } else {
                    ggml_free(ctx->fused_ctx);
                    ctx->fused_ctx = nullptr;
                    for (auto& b : blocks)
                        b.attn_qkv_w = nullptr;
                }
            }
        }
        // FFN gate+up fuse was tried (PLAN #7 phase 2) but delivered no
        // measurable speedup on M1 Q4_K voxtral4b — Metal's Q4_K matmul
        // kernel appears already memory-bandwidth-bound for the
        // (3072, 9216) shape, so combining two of those into (3072, 18432)
        // doesn't help the per-step decode budget. Removed; the
        // `core_ffn::swiglu_fused_gate_up` helper stays in place for any
        // future caller (e.g. a different backend or larger model where
        // the savings might be real). See HISTORY §71 for the
        // measurement notes.
    }

    if (params.verbosity >= 1) {
        const auto& hp = ctx->model.hparams;
        fprintf(stderr,
                "voxtral4b: loaded %s  (audio %u layers, llm %u layers, vocab %u, "
                "tekken %d specials + %d BPE)\n",
                path, hp.audio_n_layers, hp.llm_n_layers, hp.llm_vocab_size, ctx->vocab.n_specials, ctx->vocab.n_vocab);
    }
    return ctx;
}

extern "C" void voxtral4b_free(voxtral4b_context* ctx) {
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
    if (ctx->model.buf_cpu)
        ggml_backend_buffer_free(ctx->model.buf_cpu);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" const uint8_t* voxtral4b_token_text(voxtral4b_context* ctx, int id, int* out_len) {
    if (!ctx) {
        if (out_len)
            *out_len = 0;
        return nullptr;
    }
    const auto& v = ctx->vocab;
    if (id >= 0 && id < v.n_specials) {
        if (out_len)
            *out_len = (int)v.specials[id].size();
        return (const uint8_t*)v.specials[id].data();
    }
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

// Tekken BPE tokenizer — reuse same algorithm as 3B (see voxtral.cpp)
static void tekken4b_build_reverse(voxtral4b_vocab& v) {
    if (v.reverse_built)
        return;
    v.bytes_to_rank.reserve(v.rank_offset.size());
    for (size_t r = 0; r < v.rank_offset.size(); r++) {
        std::string key((const char*)v.tekken_vocab_blob.data() + v.rank_offset[r], v.rank_length[r]);
        v.bytes_to_rank[key] = (int32_t)r;
    }
    v.reverse_built = true;
}

static int32_t tekken4b_rank(const voxtral4b_vocab& v, const uint8_t* data, size_t len) {
    auto it = v.bytes_to_rank.find(std::string((const char*)data, len));
    return it != v.bytes_to_rank.end() ? it->second : -1;
}

static void tekken4b_bpe_encode(const voxtral4b_vocab& v, const uint8_t* data, size_t len, std::vector<int32_t>& out) {
    if (len == 0)
        return;
    if (len == 1) {
        int32_t r = tekken4b_rank(v, data, 1);
        out.push_back(r >= 0 ? r + v.n_specials : 0);
        return;
    }
    struct piece {
        size_t start;
        size_t len;
    };
    std::vector<piece> pieces(len);
    for (size_t i = 0; i < len; i++)
        pieces[i] = {i, 1};
    while (pieces.size() > 1) {
        int32_t best_rank = INT32_MAX;
        size_t best_idx = SIZE_MAX;
        for (size_t i = 0; i + 1 < pieces.size(); i++) {
            size_t ml = pieces[i].len + pieces[i + 1].len;
            int32_t r = tekken4b_rank(v, data + pieces[i].start, ml);
            if (r >= 0 && r < best_rank) {
                best_rank = r;
                best_idx = i;
            }
        }
        if (best_idx == SIZE_MAX)
            break;
        pieces[best_idx].len += pieces[best_idx + 1].len;
        pieces.erase(pieces.begin() + best_idx + 1);
    }
    for (const auto& p : pieces) {
        int32_t r = tekken4b_rank(v, data + p.start, p.len);
        out.push_back(r >= 0 ? r + v.n_specials : 0);
    }
}

static std::vector<std::string> tekken4b_pre_tokenize(const std::string& text) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = text[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            size_t j = i;
            while (j < text.size() && (text[j] == ' ' || text[j] == '\t' || text[j] == '\n' || text[j] == '\r'))
                j++;
            out.push_back(text.substr(i, j - i));
            i = j;
        } else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
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
        } else if (c >= 0x80) {
            size_t j = i + 1;
            while (j < text.size() && (text[j] & 0xC0) == 0x80)
                j++;
            while (j < text.size() && ((unsigned char)text[j]) >= 0x80) {
                size_t k = j + 1;
                while (k < text.size() && (text[k] & 0xC0) == 0x80)
                    k++;
                j = k;
            }
            out.push_back(text.substr(i, j - i));
            i = j;
        } else {
            out.push_back(text.substr(i, 1));
            i++;
        }
    }
    return out;
}

extern "C" int32_t* voxtral4b_tokenize(voxtral4b_context* ctx, const char* text, int* out_n_tokens) {
    if (!ctx || !text) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }
    auto& v = ctx->vocab;
    tekken4b_build_reverse(v);
    std::string input(text);
    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < input.size()) {
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
        size_t next_special = input.size();
        for (int si = 0; si < (int)v.specials.size(); si++) {
            const auto& sp = v.specials[si];
            if (sp.empty())
                continue;
            size_t f = input.find(sp, pos);
            if (f != std::string::npos && f < next_special)
                next_special = f;
        }
        auto pre_tokens = tekken4b_pre_tokenize(input.substr(pos, next_special - pos));
        for (const auto& pt : pre_tokens)
            tekken4b_bpe_encode(v, (const uint8_t*)pt.data(), pt.size(), ids);
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

// Encoder run
extern "C" float* voxtral4b_run_encoder(voxtral4b_context* ctx, const float* mel, int n_mels, int T_mel, int* out_N,
                                        int* out_dim) {
    if (!ctx || !mel || n_mels != 128 || T_mel <= 0 || T_mel % 2 != 0)
        return nullptr;
    const int T_enc = (T_mel + 1) / 2;
    // Truncate to be divisible by 4 (downsample factor)
    const int T_enc_ds = (T_enc / 4) * 4; // round down
    const int N_out = T_enc_ds / 4;
    const int dim = (int)ctx->model.hparams.proj_out_dim;

    ggml_cgraph* gf = voxtral4b_build_graph_encoder(ctx, T_mel);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    // Set mel input
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel"), mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    // Set causal conv padding to zeros
    {
        ggml_tensor* p0 = ggml_graph_get_tensor(gf, "conv0_lpad");
        if (p0) {
            std::vector<float> zeros(2 * n_mels, 0.0f);
            ggml_backend_tensor_set(p0, zeros.data(), 0, zeros.size() * sizeof(float));
        }
        ggml_tensor* p1 = ggml_graph_get_tensor(gf, "conv1_lpad");
        if (p1) {
            int d = (int)ctx->model.hparams.audio_d_model;
            std::vector<float> zeros(d, 0.0f);
            ggml_backend_tensor_set(p1, zeros.data(), 0, zeros.size() * sizeof(float));
        }
    }

    // Set encoder positions [0, 1, ..., T_enc-1]
    std::vector<int32_t> pos(T_enc);
    for (int i = 0; i < T_enc; i++)
        pos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_positions"), pos.data(), 0, pos.size() * sizeof(int32_t));

    // Set SWA mask if present
    ggml_tensor* swa_t = ggml_graph_get_tensor(gf, "swa_mask");
    if (swa_t) {
        int swa = (int)ctx->model.hparams.audio_swa;
        std::vector<ggml_fp16_t> mask((size_t)T_enc * T_enc, ggml_fp32_to_fp16(0.0f));
        ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        // Causal + sliding window: mask k > q (causal) and k <= q - swa (outside window)
        for (int q = 0; q < T_enc; q++)
            for (int k = 0; k < T_enc; k++)
                if (k > q || k <= q - swa)
                    mask[(size_t)q * T_enc + k] = neg_inf;
        ggml_backend_tensor_set(swa_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* out = ggml_graph_get_tensor(gf, "encoder_out");
    size_t total = (size_t)N_out * dim;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    if (out_N)
        *out_N = N_out;
    if (out_dim)
        *out_dim = dim;
    return result;
}

// Embed tokens
extern "C" float* voxtral4b_embed_tokens(voxtral4b_context* ctx, const int32_t* input_ids, int n_tokens) {
    if (!ctx || !input_ids || n_tokens <= 0)
        return nullptr;
    const int d = (int)ctx->model.hparams.llm_d_model;
    ggml_cgraph* gf = voxtral4b_build_graph_embed(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input_ids"), input_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* result = (float*)malloc((size_t)n_tokens * d * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)n_tokens * d * sizeof(float));
    return result;
}

// KV cache
extern "C" bool voxtral4b_kv_init(voxtral4b_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    // Idempotent: callers (notably crispasr_backend_voxtral4b's
    // per-chunk transcribe path) re-init on every audio chunk. Without
    // this guard, each call replaces ctx->kv_buf with a fresh backend
    // allocation while leaking the previous one — ~256-512 MiB of
    // VRAM per chunk depending on KV dtype, n_layers, and max_ctx
    // (issue #54: voxtral4b OOMs after ~49 chunks of 30 s audio).
    if (ctx->kv_k)
        return true;
    const auto& hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int nl = (int)hp.llm_n_layers;

    ggml_init_params ip = {2 * ggml_tensor_overhead(), nullptr, true};
    ctx->kv_ctx = ggml_init(ip);
    // PLAN #60e + #69e: per-half KV dtype. CRISPASR_KV_QUANT sets both,
    // CRISPASR_KV_QUANT_{K,V} override per half. Default f16/f16.
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("voxtral4b");
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, nl);
    // Size the backend buffer from the actual ggml_nbytes after creation
    // (was hardcoded F16 byte count, which over-allocated for quant types).
    const size_t k_size = ggml_nbytes(ctx->kv_k);
    const size_t v_size = ggml_nbytes(ctx->kv_v);

    // PLAN #69b: optional KV-on-CPU spill for long-context / tight-VRAM users.
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "voxtral4b");
    ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, k_size + v_size);
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + ggml_nbytes(ctx->kv_k));

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "voxtral4b: kv cache %.0f MiB k=%s v=%s (on %s, head_dim=%d max_ctx=%d n_kv=%d n_layers=%d)\n",
                (k_size + v_size) / 1048576.0, ggml_type_name(kv_pair.k), ggml_type_name(kv_pair.v),
                kv_backend == ctx->backend_cpu ? "cpu" : "gpu", hd, max_ctx, n_kv, nl);
    return true;
}

extern "C" void voxtral4b_kv_reset(voxtral4b_context* ctx) {
    if (ctx && ctx->kv_buf)
        ggml_backend_buffer_clear(ctx->kv_buf, 0);
}

// LLM forward with KV cache
extern "C" float* voxtral4b_run_llm_kv(voxtral4b_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                                       int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int vocab = (int)hp.llm_vocab_size;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        int Lk = n_past + n_tokens;
        mask.resize((size_t)n_tokens * Lk, ggml_fp32_to_fp16(0.0f));
        ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        int swa = (int)hp.llm_swa;
        for (int q = 0; q < n_tokens; q++)
            for (int k = 0; k < Lk; k++)
                if (k > n_past + q || k <= (n_past + q) - swa)
                    mask[(size_t)q * Lk + k] = neg_inf;
    }

    ggml_cgraph* gf = voxtral4b_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), inputs_embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ada_scales"), ctx->ada_scales.data(), 0,
                            ctx->ada_scales.size() * sizeof(float));
    if (n_tokens > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(logits_t, result, 0, (size_t)vocab * sizeof(float));
    if (out_n_tokens)
        *out_n_tokens = 1;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    return result;
}

// ===========================================================================
// Native streaming (PLAN #7) — synchronous incremental encoder + LLM
// decode-on-flush. PTT/dictation phase 1.
// ===========================================================================

namespace {

// One-time mel computation for a streaming chunk. Mirrors `voxtral4b_compute_mel`
// but with `center_pad=false`, `drop_last_frame=false`, `drop_first_frame_if_odd=false`.
// The caller manages center-pad emulation by prepending n_fft/2 zeros to the very
// first chunk's PCM and per-call left-overlap (n_fft - hop = 240 samples) to
// subsequent chunks.
static std::vector<float> compute_mel_streaming(voxtral4b_context* ctx, const float* samples, int n_samples,
                                                int* out_T_mel) {
    const int n_fft = 400, hop = 160, n_mels = 128, n_freqs = 201;
    std::vector<float> hann(n_fft);
    ggml_backend_tensor_get(ctx->model.audio.mel_window, hann.data(), 0, n_fft * sizeof(float));
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->model.audio.mel_filters, filt.data(), 0, filt.size() * sizeof(float));

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Log10;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.norm = core_mel::Normalization::GlobalClipFixed;
    p.layout = core_mel::Layout::MelsTime;
    p.fb_layout = core_mel::FbLayout::FreqsMels;
    p.matmul = core_mel::MatmulPrecision::Double;
    p.log_eps = 1e-10f;
    p.fixed_max = 1.5f;
    p.center_pad = false; // caller pre-pads
    p.drop_last_frame = false;
    p.drop_first_frame_if_odd = false;

    int T_ret = 0;
    auto mel = core_mel::compute(samples, n_samples, hann.data(), n_fft, filt.data(), n_freqs, voxtral4b_fft_wrapper, p,
                                 T_ret);
    if (out_T_mel)
        *out_T_mel = T_ret;
    return mel;
}

// Streaming encoder graph builder.
// Inputs:
//   - mel input: [n_mels, T_chunk_mel + 2_lctx_mel]   (lctx prepended by caller)
//   - conv0_lctx is part of the mel input (first 2 frames)
//   - conv1_lctx tensor: [d_model, 1] (separate input; prepended in graph)
//   - K/V cache tensors per layer: [head_dim, max_T_enc, n_kv, n_layers]
//   - positions: [T_chunk_enc] I32 — global encoder-frame positions
//   - causal_mask: [T_chunk_enc, Lk] F16 — Lk = n_past + T_chunk_enc
// Outputs:
//   - "stream_enc_out": [d_model, T_chunk_enc] post-RMSNorm (pre-projector)
//   - "stream_conv0_out_last_d": [d_model, 1] — capture last conv0-out frame for next call's conv1_lctx
static ggml_cgraph* voxtral4b_build_graph_encoder_stream(voxtral4b_context* ctx, int T_chunk_mel, int n_past_enc,
                                                         ggml_tensor* kv_k, ggml_tensor* kv_v) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.audio_d_model;
    const int n_heads = (int)hp.audio_n_heads;
    const int head_dim = (int)hp.audio_head_dim;
    const int n_layers = (int)hp.audio_n_layers;
    const int n_mels = (int)hp.n_mels;
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);
    const int T_chunk_enc = T_chunk_mel / 2; // conv1 stride-2 with 1-frame lctx

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Mel input includes 2 left-context mel frames at the front. Total time = T_chunk_mel + 2.
    const int T_mel_in = T_chunk_mel + 2;
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel_in, n_mels);
    ggml_set_name(mel, "stream_mel");
    ggml_set_input(mel);

    // Conv0: k=3, s=1, no pad (caller supplied 2 lctx mel frames). Out length = T_chunk_mel.
    ggml_tensor* cur = ggml_conv_1d(ctx0, m.audio.conv1_w, mel, 1, 0, 1);
    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur); // shape (T_chunk_mel, d, 1)

    // Capture last conv0-out frame as next call's conv1_lctx. View slice (last frame across d).
    {
        // cur is (T_chunk_mel, d, 1) row-major. Last frame is offset (T_chunk_mel - 1) along ne[0].
        ggml_tensor* last =
            ggml_view_3d(ctx0, cur, 1, d, 1, cur->nb[1], cur->nb[2], (size_t)(T_chunk_mel - 1) * cur->nb[0]);
        last = ggml_cont(ctx0, last);
        ggml_set_name(last, "stream_conv0_last_d");
        ggml_set_output(last);
        ggml_build_forward_expand(gf, last);
    }

    // Conv1 left-context: 1-frame [d, 1] supplied as a separate input. Concat at front.
    ggml_tensor* conv1_lctx = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, d, 1);
    ggml_set_name(conv1_lctx, "stream_conv1_lctx");
    ggml_set_input(conv1_lctx);
    cur = ggml_concat(ctx0, conv1_lctx, cur, 0); // (T_chunk_mel + 1, d, 1)

    // Conv1: k=3, s=2, no pad. Out = (T_chunk_mel + 1 - 3) / 2 + 1 = T_chunk_mel / 2 = T_chunk_enc.
    cur = ggml_conv_1d(ctx0, m.audio.conv2_w, cur, 2, 0, 1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);

    // Reshape to (d, T_chunk_enc).
    cur = ggml_reshape_2d(ctx0, cur, T_chunk_enc, d);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    // Position ids for the chunk: [n_past, n_past + T_chunk_enc).
    ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_chunk_enc);
    ggml_set_name(pos, "stream_positions");
    ggml_set_input(pos);

    // Causal+SWA mask of shape [Lk, T_chunk_enc] (ne[0]=Lk, ne[1]=T_chunk_enc).
    const int Lk = n_past_enc + T_chunk_enc;
    ggml_tensor* mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T_chunk_enc);
    ggml_set_name(mask, "stream_mask");
    ggml_set_input(mask);

    // Encoder transformer layers using kv_self_attn (persistent K/V cache).
    core_attn::KvSelfAttnParams ap = {};
    ap.n_heads = n_heads;
    ap.n_kv_heads = n_heads; // encoder is MHA
    ap.head_dim = head_dim;
    ap.n_kv_grp = 1;
    ap.n_ctx_orig = 0;
    ap.rope_theta = hp.audio_rope_theta;
    ap.rope_beta_fast = 0.0f;
    ap.rope_beta_slow = 0.0f;
    ap.attn_scale = attn_scale;
    ap.qk_norm_eps = 0.0f;
    ap.gqa_mode = core_attn::GQA_NATIVE;
    ap.rope_type = GGML_ROPE_TYPE_NEOX;
    ap.n_rot = 0;
    ap.v_rms_norm = false;

    for (int il = 0; il < n_layers; il++) {
        const auto& b = m.audio.blocks[il];
        ggml_tensor* residual = cur;

        // Pre-RMSNorm + scale.
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, kRmsEps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // KV-cached self-attention. Voxtral4b encoder has Q-bias and V-bias (no K, optional O).
        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_out_w,
                                                    /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, pos, mask, kv_k, kv_v,
                                                    il, n_past_enc, ap, /*qkv_w*/ nullptr, /*fixed_kv_len*/ 0,
                                                    /*kv_indices*/ nullptr, /*q_b*/ b.attn_q_b, /*k_b*/ nullptr,
                                                    /*v_b*/ b.attn_v_b, /*o_b*/ b.attn_out_b);
        cur = ggml_add(ctx0, residual, attn);

        // FFN.
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, kRmsEps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* ffn = core_ffn::swiglu_down_bias(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w, b.ffn_down_b);
        cur = ggml_add(ctx0, residual, ffn);
    }

    // Final RMSNorm — pre-projector. Caller does the stack-4 + 2×Linear externally
    // because projector input frames may span chunks.
    cur = ggml_rms_norm(ctx0, cur, kRmsEps);
    cur = ggml_mul(ctx0, cur, m.audio.ln_post_w);
    ggml_set_name(cur, "stream_enc_out");
    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);
    return gf;
}

// Build a graph that just runs the projector (stack-4 + 2×Linear + GELU) over
// `n_groups` × 4 encoder frames. Lets the streaming flusher project accumulated
// frames in batches without re-running the encoder.
static ggml_cgraph* voxtral4b_build_graph_projector(voxtral4b_context* ctx, int n_groups) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.audio_d_model;
    const int proj_in = (int)hp.proj_in_dim;
    const int T_in = n_groups * 4;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_in);
    ggml_set_name(x, "proj_in");
    ggml_set_input(x);

    ggml_tensor* cur = ggml_reshape_2d(ctx0, x, proj_in, n_groups);
    cur = ggml_mul_mat(ctx0, m.projector.proj1, cur);
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_mul_mat(ctx0, m.projector.proj2, cur);
    ggml_set_name(cur, "proj_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

} // namespace

// ── Stream object + entrypoints ─────────────────────────────────────────────

struct voxtral4b_stream {
    voxtral4b_context* ctx; // not owned

    // Configuration.
    int chunk_mel_frames; // must be even (conv1 stride-2). default 8 = 80ms audio.
    int max_audio_seconds;

    // ── Mel state ──────────────────────────────────────────────────────────
    // pcm_with_pad holds (n_fft/2 zero-prefix) + all PCM fed so far. Frames
    // emitted = floor((pcm_with_pad.size() - n_fft) / hop) + 1 once primed.
    std::vector<float> pcm_with_pad;
    int pcm_next_frame_start; // = mel_frames_emitted * hop
    int mel_frames_emitted;
    bool mel_primed;

    // Mel staging: frames emitted but not yet consumed by the encoder.
    std::vector<float> mel_pending; // shape (n_mels, T_pending) row-major in (T,M) layout

    // ── Conv left-context ──────────────────────────────────────────────────
    std::vector<float> conv0_lctx; // (n_mels) × 2 mel frames
    std::vector<float> conv1_lctx; // (d_model) × 1 frame
    bool first_chunk_seen;

    // ── Encoder K/V cache (separate from LLM's) ────────────────────────────
    ggml_context* enc_kv_ctx;
    ggml_backend_buffer_t enc_kv_buf;
    ggml_tensor* enc_kv_k;
    ggml_tensor* enc_kv_v;
    int enc_T_so_far;
    int enc_max_T;

    // ── Projector pending ─────────────────────────────────────────────────
    // Encoder frames awaiting stack-4 alignment. Up to 3 frames × d_model.
    std::vector<float> enc_frames_pending; // row-major (d_model, n)
    int enc_frames_pending_count;

    // Audio embeddings emitted by the projector. Fed to LLM at flush.
    std::vector<float> audio_embeds; // row-major (proj_out_dim, N_audio)

    // ── LLM / decode state ────────────────────────────────────────────────
    bool kv_initialised;
    int n_past;

    // ── Speculative prefill (PLAN #7 phase 3) ──────────────────────────────
    // Once we have ≥ kStreamingPromptLen audio_embeds during feed(), run
    // the LLM prefill speculatively so flush() only does the decode loop.
    // Saves ~250ms off first-text-token wall-clock at flush time.
    bool prefill_done;
    std::vector<float> prefill_logits; // vocab_size × 1 (last position's logits)
    int prefill_n_past;
    int prefill_out_vocab;

    // ── Live decode state (PLAN #7 phase 3 — live captions) ────────────────
    // After speculative prefill, every new audio_embed enables one more
    // greedy decode step. With live_decode_enabled, those steps run during
    // feed() and tokens get appended to out_text / out_text_unread as they
    // commit; get_text() then returns progressive transcript.
    // Off by default (PTT semantics: decode happens at flush). Set
    // CRISPASR_VOXTRAL4B_STREAM_LIVE=1 at stream_open to opt in.
    //
    // No stable-prefix heuristic needed: voxtral4b's audio-injection
    // pre_hook makes each decoded token a deterministic function of the
    // audio context up to that point. Tokens commit immediately — no
    // retraction. (This is the key architectural property that distinguishes
    // injection-prompt audio-LLMs from encoder-decoder ASR where stable-
    // prefix heuristics matter.)
    bool live_decode_enabled;
    bool decode_started;              // first decode step ran (logits/state initialised)
    bool decode_finished;             // EOS emitted
    bool decode_logits_committed;     // true if decode_logits's argmax has already been emitted
                                      // (prevents double-emission across multiple drain calls)
    int decode_n_past;                // current LLM position cursor (advances per step)
    int decode_adapter_pos;           // current audio_embed cursor (advances per step)
    int decode_out_vocab;             // cached for the current logits buffer
    int decode_last_argmax_id;        // most-recent argmax id (committed but not yet forwarded)
    std::vector<float> decode_logits; // last-position logits for next argmax
    int decode_steps_done;            // total decode steps run (for timing telemetry)

    // ── Decoder thread (PLAN #7 phase 4) ───────────────────────────────────
    // When `decoder_thread_enabled` is true, vox_stream_drain_decode runs
    // on a worker thread instead of the caller's thread. Feed pushes
    // audio_embeds and signals the worker; the worker drains decode steps
    // while audio is available. Lets feed() return without waiting for
    // decode — critical for mic-driven live captions where the audio
    // thread must stay current.
    //
    // sched_mutex serializes all ggml_backend_sched access (the encoder
    // graphs from feed and the LLM forward from worker share ctx->sched).
    // On Metal that means the encoder and LLM still run sequentially on
    // the GPU, but the CPU-side feed() returns immediately after queuing
    // audio_embeds. On a faster GPU with kernel parallelism the
    // encoder + LLM may actually overlap; the architecture supports it
    // without further code changes.
    bool decoder_thread_enabled;
    std::thread worker_thread;
    std::mutex sched_mutex;        // serializes encoder + decoder access to ctx->sched
    std::mutex decode_state_mutex; // protects decode_*, audio_embeds, out_text*
    std::condition_variable cond_var;
    std::atomic<bool> shutdown_requested;
    std::atomic<bool> worker_idle; // true when worker is waiting on cond_var (no pending work)

    // ── Output ────────────────────────────────────────────────────────────
    std::string out_text;
    std::string out_text_unread;
    double out_t0_s;
    double out_t1_s;
    bool has_output;
    int decode_counter;

    // Total PCM samples ever fed (for the t1 timestamp).
    int64_t total_samples_fed;
};

namespace {

static int vox_stream_alloc_enc_kv(voxtral4b_stream* s) {
    const auto& hp = s->ctx->model.hparams;
    const int hd = (int)hp.audio_head_dim;
    const int n_kv = (int)hp.audio_n_heads; // MHA
    const int nl = (int)hp.audio_n_layers;
    s->enc_max_T = (int)hp.audio_swa; // 750 frames = 15s

    ggml_init_params ip = {2 * ggml_tensor_overhead(), nullptr, true};
    s->enc_kv_ctx = ggml_init(ip);
    // F32 cache: the batch encoder runs in F32 throughout (no cache), so
    // matching that precision is necessary for bit-exact-batch. F16 caches
    // (used by the LLM, which was trained with F16 KV) introduce precision
    // loss that compounds across the encoder's 32 layers and degrades the
    // audio embeddings enough that the LLM reads them as silence.
    s->enc_kv_k = ggml_new_tensor_4d(s->enc_kv_ctx, GGML_TYPE_F32, hd, s->enc_max_T, n_kv, nl);
    s->enc_kv_v = ggml_new_tensor_4d(s->enc_kv_ctx, GGML_TYPE_F32, hd, s->enc_max_T, n_kv, nl);
    const size_t k_size = ggml_nbytes(s->enc_kv_k);
    const size_t v_size = ggml_nbytes(s->enc_kv_v);
    s->enc_kv_buf = ggml_backend_alloc_buffer(s->ctx->backend, k_size + v_size);
    if (!s->enc_kv_buf)
        return -1;
    char* base = (char*)ggml_backend_buffer_get_base(s->enc_kv_buf);
    ggml_backend_tensor_alloc(s->enc_kv_buf, s->enc_kv_k, base);
    ggml_backend_tensor_alloc(s->enc_kv_buf, s->enc_kv_v, base + k_size);
    ggml_backend_buffer_clear(s->enc_kv_buf, 0);
    if (s->ctx->params.verbosity >= 1)
        fprintf(stderr, "voxtral4b_stream: enc kv cache %.1f MiB (max_T=%d)\n", (k_size + v_size) / 1048576.0,
                s->enc_max_T);
    return 0;
}

// Compute and emit any new mel frames that fit in pcm_with_pad's current contents.
// Appends new frames (n_mels, T_new) row-major-as-(T,M) to s->mel_pending.
static int vox_stream_advance_mel(voxtral4b_stream* s) {
    const int n_fft = 400, hop = 160, n_mels = 128;
    const int avail = (int)s->pcm_with_pad.size();
    if (avail < n_fft)
        return 0;
    const int last_possible_start = avail - n_fft;
    if (last_possible_start < s->pcm_next_frame_start)
        return 0;
    const int n_new = (last_possible_start - s->pcm_next_frame_start) / hop + 1;
    const int slice_len = (n_new - 1) * hop + n_fft;
    int T_out = 0;
    auto mel = compute_mel_streaming(s->ctx, s->pcm_with_pad.data() + s->pcm_next_frame_start, slice_len, &T_out);
    if (T_out != n_new) {
        fprintf(stderr, "voxtral4b_stream: mel mismatch: expected %d frames, got %d\n", n_new, T_out);
        return -1;
    }
    // mel layout is MelsTime: ggml ne=(T, n_mels), memory stride T fast,
    // n_mels slow. Index `mel[m*T + t]` means for band m at time t. To
    // accumulate chunks into a (n_mels, T_total) tensor we have to
    // interleave per-band rows: each new chunk's row m gets appended
    // after the previous chunks' row m. So mel_pending is laid out as
    // (n_mels, T_total) and per-band slices need contiguous time.
    const size_t prev_T = (n_mels > 0 && !s->mel_pending.empty()) ? s->mel_pending.size() / n_mels : 0;
    const size_t new_T = prev_T + (size_t)T_out;
    std::vector<float> rebuilt((size_t)n_mels * new_T);
    for (int m = 0; m < n_mels; m++) {
        // Copy old per-band row.
        if (prev_T > 0)
            std::memcpy(rebuilt.data() + (size_t)m * new_T, s->mel_pending.data() + (size_t)m * prev_T,
                        prev_T * sizeof(float));
        // Append new per-band row.
        std::memcpy(rebuilt.data() + (size_t)m * new_T + prev_T, mel.data() + (size_t)m * T_out,
                    (size_t)T_out * sizeof(float));
    }
    s->mel_pending = std::move(rebuilt);
    s->pcm_next_frame_start += n_new * hop;
    s->mel_frames_emitted += n_new;
    return n_new;
}

// Build causal+SWA mask in [Lk, T_chunk_enc] layout. Voxtral4b mask convention:
// Q at position q (chunk-local) maps to global position n_past + q. Allowed K
// indices: [max(0, n_past+q - swa + 1), n_past+q].
static std::vector<ggml_fp16_t> vox_stream_build_mask(int T_chunk_enc, int n_past, int swa) {
    const int Lk = n_past + T_chunk_enc;
    std::vector<ggml_fp16_t> mask((size_t)T_chunk_enc * Lk, ggml_fp32_to_fp16(0.0f));
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < T_chunk_enc; q++) {
        const int qg = n_past + q;
        for (int k = 0; k < Lk; k++) {
            // ne[0]=Lk so layout is mask[q*Lk + k]
            if (k > qg || k <= qg - swa)
                mask[(size_t)q * Lk + k] = neg_inf;
        }
    }
    return mask;
}

// Run the streaming encoder graph on s->mel_pending's first chunk_mel_frames worth.
// Consumes those frames, advances enc_T_so_far by chunk_mel_frames/2, appends
// the post-RMSNorm encoder output to enc_frames_pending.
//
// Layout convention: mel_pending and conv0_lctx are both stored in
// (n_mels, T) row-major — per-band time series contiguous, matching what
// `core_mel::compute` emits with `Layout::MelsTime` and what the encoder
// graph's mel input expects (ggml ne=(T, n_mels) with T as the fast axis).
static int vox_stream_run_encoder_chunk(voxtral4b_stream* s) {
    const auto& hp = s->ctx->model.hparams;
    const int n_mels = (int)hp.n_mels;
    const int d = (int)hp.audio_d_model;
    const int T_chunk_mel = s->chunk_mel_frames;
    const int T_chunk_enc = T_chunk_mel / 2;

    if ((int)s->mel_pending.size() < T_chunk_mel * n_mels)
        return 0; // not enough yet
    const int T_pending = (int)(s->mel_pending.size() / (size_t)n_mels);

    if (s->enc_T_so_far + T_chunk_enc > s->enc_max_T) {
        // SWA window full — silently drop the oldest chunk. Phase 1 limitation.
        return 0;
    }

    // Build mel input buffer: per-band [conv0_lctx (2 frames) ; mel_pending[0..T_chunk_mel)]
    // Layout (n_mels, T_chunk_mel + 2) — per-band-row contiguous.
    const int T_in = T_chunk_mel + 2;
    std::vector<float> mel_in((size_t)n_mels * T_in);
    for (int m = 0; m < n_mels; m++) {
        std::memcpy(mel_in.data() + (size_t)m * T_in, s->conv0_lctx.data() + (size_t)m * 2, 2 * sizeof(float));
        std::memcpy(mel_in.data() + (size_t)m * T_in + 2, s->mel_pending.data() + (size_t)m * T_pending,
                    (size_t)T_chunk_mel * sizeof(float));
    }

    ggml_cgraph* gf =
        voxtral4b_build_graph_encoder_stream(s->ctx, T_chunk_mel, s->enc_T_so_far, s->enc_kv_k, s->enc_kv_v);
    ggml_backend_sched_reset(s->ctx->sched);
    if (!ggml_backend_sched_alloc_graph(s->ctx->sched, gf))
        return -2;

    // Set inputs.
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "stream_mel"), mel_in.data(), 0, mel_in.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "stream_conv1_lctx"), s->conv1_lctx.data(), 0,
                            s->conv1_lctx.size() * sizeof(float));
    std::vector<int32_t> pos(T_chunk_enc);
    for (int i = 0; i < T_chunk_enc; i++)
        pos[i] = s->enc_T_so_far + i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "stream_positions"), pos.data(), 0, pos.size() * sizeof(int32_t));
    auto mask = vox_stream_build_mask(T_chunk_enc, s->enc_T_so_far, (int)hp.audio_swa);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "stream_mask"), mask.data(), 0,
                            mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(s->ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return -3;

    // Read outputs.
    ggml_tensor* enc_out = ggml_graph_get_tensor(gf, "stream_enc_out");
    const size_t enc_chunk_floats = (size_t)d * T_chunk_enc;
    const size_t pad_in_pending = (size_t)s->enc_frames_pending_count * d;
    s->enc_frames_pending.resize(pad_in_pending + enc_chunk_floats);
    ggml_backend_tensor_get(enc_out, s->enc_frames_pending.data() + pad_in_pending, 0,
                            enc_chunk_floats * sizeof(float));
    s->enc_frames_pending_count += T_chunk_enc;

    ggml_tensor* conv0_last = ggml_graph_get_tensor(gf, "stream_conv0_last_d");
    s->conv1_lctx.assign(d, 0.0f);
    ggml_backend_tensor_get(conv0_last, s->conv1_lctx.data(), 0, (size_t)d * sizeof(float));

    // Update conv0_lctx = last 2 mel frames of this chunk, per-band layout.
    s->conv0_lctx.assign((size_t)n_mels * 2, 0.0f);
    for (int m = 0; m < n_mels; m++) {
        s->conv0_lctx[(size_t)m * 2 + 0] = s->mel_pending[(size_t)m * T_pending + (T_chunk_mel - 2)];
        s->conv0_lctx[(size_t)m * 2 + 1] = s->mel_pending[(size_t)m * T_pending + (T_chunk_mel - 1)];
    }

    // Drop consumed mel frames: shift each per-band row left by T_chunk_mel,
    // dropping the leading T_chunk_mel samples and keeping the remaining
    // T_pending - T_chunk_mel.
    const int T_remain = T_pending - T_chunk_mel;
    if (T_remain > 0) {
        std::vector<float> rebuilt((size_t)n_mels * T_remain);
        for (int m = 0; m < n_mels; m++) {
            std::memcpy(rebuilt.data() + (size_t)m * T_remain,
                        s->mel_pending.data() + (size_t)m * T_pending + T_chunk_mel, (size_t)T_remain * sizeof(float));
        }
        s->mel_pending = std::move(rebuilt);
    } else {
        s->mel_pending.clear();
    }

    s->enc_T_so_far += T_chunk_enc;
    s->first_chunk_seen = true;
    return 1;
}

// Run the projector over any complete groups of 4 enc frames sitting in
// enc_frames_pending. Append projected audio embeddings to s->audio_embeds.
static int vox_stream_advance_projector(voxtral4b_stream* s) {
    const auto& hp = s->ctx->model.hparams;
    const int d = (int)hp.audio_d_model;
    const int proj_out = (int)hp.proj_out_dim;
    const int n_groups = s->enc_frames_pending_count / 4;
    if (n_groups == 0)
        return 0;
    ggml_cgraph* gf = voxtral4b_build_graph_projector(s->ctx, n_groups);
    ggml_backend_sched_reset(s->ctx->sched);
    if (!ggml_backend_sched_alloc_graph(s->ctx->sched, gf))
        return -1;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "proj_in"), s->enc_frames_pending.data(), 0,
                            (size_t)d * (n_groups * 4) * sizeof(float));
    if (ggml_backend_sched_graph_compute(s->ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return -2;
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "proj_out");
    const size_t prev = s->audio_embeds.size();
    s->audio_embeds.resize(prev + (size_t)proj_out * n_groups);
    ggml_backend_tensor_get(out_t, s->audio_embeds.data() + prev, 0, (size_t)proj_out * n_groups * sizeof(float));
    // Drop consumed enc frames.
    const int consumed = n_groups * 4;
    if (consumed < s->enc_frames_pending_count) {
        std::memmove(s->enc_frames_pending.data(), s->enc_frames_pending.data() + (size_t)consumed * d,
                     (size_t)(s->enc_frames_pending_count - consumed) * d * sizeof(float));
    }
    s->enc_frames_pending_count -= consumed;
    s->enc_frames_pending.resize((size_t)s->enc_frames_pending_count * d);
    return n_groups;
}

} // namespace

// Voxtral4B realtime model expects 32 "STREAMING_PAD" tokens worth of left-pad
// silence at the start of the audio (matches `crispasr_backend_voxtral4b.cpp`
// adapter at line 75). One token = hop * conv_stride * stack_4 = 160 * 2 * 4 =
// 1280 samples.
static constexpr int kSamplesPerToken = 1280;
static constexpr int kLeftPadTokens = 32;
static constexpr int kRightPadTokens = 10;
static constexpr int kStreamingPromptLen = 39; // 1 BOS + 32 STREAMING_PAD + 6 delay

namespace {
// PLAN #7 phase 3: speculative LLM prefill during feed.
// Once feed has produced ≥ kStreamingPromptLen audio_embeds, run the
// streaming-prompt prefill once and stash the resulting last-position
// logits + n_past on the stream. Flush then skips the prefill (a
// ~250 ms cost) and jumps straight to the decode loop. Idempotent —
// only runs once per stream lifetime.
//
// Returns 0 on success / not-yet-ready, <0 on error.
static int vox_stream_maybe_prefill(voxtral4b_stream* s) {
    if (s->prefill_done)
        return 0;
    const auto& hp = s->ctx->model.hparams;
    const int proj_out = (int)hp.proj_out_dim;
    const int N_audio = (int)(s->audio_embeds.size() / proj_out);
    if (N_audio < kStreamingPromptLen)
        return 0;

    // Build streaming-prompt: BOS + 38 × STREAMING_PAD = 39 tokens.
    std::vector<int32_t> prompt_ids((size_t)kStreamingPromptLen);
    prompt_ids[0] = 1;
    for (int i = 1; i < kStreamingPromptLen; i++)
        prompt_ids[i] = 32;

    float* prompt_emb = voxtral4b_embed_tokens(s->ctx, prompt_ids.data(), kStreamingPromptLen);
    if (!prompt_emb)
        return -1;
    // Element-wise add the first kStreamingPromptLen audio embeds to the prompt.
    for (int i = 0; i < kStreamingPromptLen; i++) {
        const float* src = s->audio_embeds.data() + (size_t)i * proj_out;
        float* dst = prompt_emb + (size_t)i * proj_out;
        for (int j = 0; j < proj_out; j++)
            dst[j] += src[j];
    }

    constexpr int kMaxNewTokens = 512;
    if (!s->kv_initialised) {
        if (!voxtral4b_kv_init(s->ctx, kStreamingPromptLen + kMaxNewTokens + 16)) {
            std::free(prompt_emb);
            return -2;
        }
        s->kv_initialised = true;
    }
    voxtral4b_kv_reset(s->ctx);

    int out_n_tok = 0, out_vocab = 0;
    float* logits = voxtral4b_run_llm_kv(s->ctx, prompt_emb, kStreamingPromptLen, 0, &out_n_tok, &out_vocab);
    std::free(prompt_emb);
    if (!logits || out_vocab <= 0)
        return -3;

    // Stash the last position's logits — that's the input to decode step 0.
    const float* last = logits + (size_t)(out_n_tok - 1) * (size_t)out_vocab;
    s->prefill_logits.assign(last, last + out_vocab);
    std::free(logits);
    s->prefill_n_past = kStreamingPromptLen;
    s->prefill_out_vocab = out_vocab;
    s->prefill_done = true;

    // Initialise the live-decode state from prefill — same data, decoupled
    // so the decode loop can mutate independently.
    s->decode_logits = s->prefill_logits;
    s->decode_n_past = kStreamingPromptLen;
    s->decode_adapter_pos = kStreamingPromptLen;
    s->decode_out_vocab = out_vocab;
    s->decode_started = true;
    return 0;
}

// PLAN #7 phase 3: shared decode-drain. Runs greedy decode steps from the
// stream's current state, emitting text into `out_text` / `out_text_unread`.
// Stops when one of: (a) EOS hit, (b) audio_embeds[adapter_pos] not yet
// available (live mode — caller will resume on next feed), (c) max
// new-tokens limit. Idempotent across feed/flush calls — flush just
// calls it after the encoder/projector are drained, with the same state.
//
// Returns 0 on success (incl. "no progress because no audio yet"), <0 on error.
static int vox_stream_drain_decode(voxtral4b_stream* s) {
    if (!s->decode_started || s->decode_finished)
        return 0;
    const auto& hp = s->ctx->model.hparams;
    const int proj_out = (int)hp.proj_out_dim;
    const int N_audio = (int)(s->audio_embeds.size() / proj_out);
    constexpr int kMaxNewTokens = 512;
    constexpr int eos_id = 2;
    if (s->decode_logits.empty() || s->decode_out_vocab <= 0)
        return -1;

    // Per-step state machine to avoid double-emission across multiple
    // drain calls (live mode invokes drain once per feed):
    //
    // 1. If decode_logits is uncommitted: argmax → emit → set committed=true.
    //    `last_argmax_id` is stashed on the stream so step 3 can use it.
    // 2. Check stop conditions (EOS, audio exhausted) — break out of loop.
    // 3. Embed `last_argmax_id` + inject audio_embeds[adapter_pos] → forward.
    //    The forward updates decode_logits and clears committed=false.
    //
    // Cross-drain-call invariant: when drain returns, decode_logits is
    // ALWAYS committed (its argmax has been emitted). The next drain call
    // skips step 1 on its first iteration and goes straight to step 2.
    while (s->decode_steps_done < kMaxNewTokens) {
        if (!s->decode_logits_committed) {
            // Step 1: argmax + emit.
            const float* last = s->decode_logits.data();
            int best = 0;
            float best_score = last[0];
            for (int i = 1; i < s->decode_out_vocab; ++i) {
                if (last[i] > best_score) {
                    best_score = last[i];
                    best = i;
                }
            }
            s->decode_last_argmax_id = best;
            if (best == eos_id) {
                s->decode_finished = true;
                s->decode_logits_committed = true;
                break;
            }
            // Filter streaming control tokens (id < 1000) from emitted text.
            if (best >= 1000) {
                int tok_len = 0;
                const uint8_t* tok_bytes = voxtral4b_token_text(s->ctx, best, &tok_len);
                std::string piece;
                if (tok_bytes && tok_len > 0)
                    piece.assign(reinterpret_cast<const char*>(tok_bytes), (size_t)tok_len);
                std::string decoded;
                decoded.reserve(piece.size());
                for (size_t ci = 0; ci < piece.size(); ci++) {
                    if ((unsigned char)piece[ci] == 0xE2 && ci + 2 < piece.size() &&
                        (unsigned char)piece[ci + 1] == 0x96 && (unsigned char)piece[ci + 2] == 0x81) {
                        decoded += ' ';
                        ci += 2;
                    } else {
                        decoded += piece[ci];
                    }
                }
                s->out_text += decoded;
                s->out_text_unread += decoded;
            }
            s->decode_logits_committed = true;
        }

        // Step 2: stop if EOS or no audio left to inject.
        if (s->decode_finished)
            break;
        if (s->decode_adapter_pos >= N_audio)
            break;
        // Step 3: embed the just-committed argmax + inject audio + forward.
        int32_t next_id = s->decode_last_argmax_id;
        float* next_emb = voxtral4b_embed_tokens(s->ctx, &next_id, 1);
        if (!next_emb)
            return -2;
        const float* aud = s->audio_embeds.data() + (size_t)s->decode_adapter_pos * proj_out;
        for (int j = 0; j < proj_out; j++)
            next_emb[j] += aud[j];
        s->decode_adapter_pos++;

        int out_n_tok = 0, out_vocab = 0;
        float* logits = voxtral4b_run_llm_kv(s->ctx, next_emb, 1, s->decode_n_past, &out_n_tok, &out_vocab);
        std::free(next_emb);
        if (!logits)
            return -3;
        s->decode_logits.assign(logits, logits + out_vocab);
        s->decode_out_vocab = out_vocab;
        std::free(logits);
        s->decode_n_past++;
        s->decode_steps_done++;
        s->decode_logits_committed = false; // new logits, need to argmax+emit on next iter
    }
    return 0;
}

// PLAN #7 phase 4 worker thread main loop. Sleeps on cond_var until
// either shutdown or there's audio to decode. Drains while it can,
// then sleeps again. Holds sched_mutex + decode_state_mutex for each
// drain call — feed() acquires sched_mutex around encoder ops, so
// they alternate (no GPU parallelism on Metal, but feed() doesn't
// wait for the entire decode loop the way it does in single-threaded
// live mode).
static void vox_stream_decoder_worker(voxtral4b_stream* s) {
    while (!s->shutdown_requested.load()) {
        {
            std::unique_lock<std::mutex> lk(s->decode_state_mutex);
            s->worker_idle.store(true);
            s->cond_var.wait(lk, [&]() {
                if (s->shutdown_requested.load())
                    return true;
                if (!s->decode_started || s->decode_finished)
                    return false;
                const int proj_out = (int)s->ctx->model.hparams.proj_out_dim;
                const int N_audio = (int)(s->audio_embeds.size() / proj_out);
                return N_audio > s->decode_adapter_pos;
            });
            if (s->shutdown_requested.load())
                break;
            s->worker_idle.store(false);
        }
        // Acquire sched_mutex around the LLM forward calls inside drain.
        // The drain helper itself reads/writes decode state while holding
        // decode_state_mutex's contract loosely — we hold both throughout
        // for safety. On M1 the LLM forward dominates so the lock-hold
        // duration is the LLM step time.
        std::lock_guard<std::mutex> sched_lk(s->sched_mutex);
        std::lock_guard<std::mutex> state_lk(s->decode_state_mutex);
        (void)vox_stream_drain_decode(s);
    }
    s->worker_idle.store(true);
}

} // namespace

extern "C" struct voxtral4b_stream* voxtral4b_stream_open(struct voxtral4b_context* ctx, int /*step_ms*/,
                                                          int /*length_ms*/) {
    if (!ctx)
        return nullptr;
    auto* s = new voxtral4b_stream();
    s->ctx = ctx;
    // Chunk size in mel frames. Must be even (conv1 stride 2) AND divisible
    // by 8 (so the projector's stack-4 alignment lands at a chunk boundary
    // and we never carry partial-projector state between chunks). At 100 fps
    // mel = 10 ms/frame, so 8 = 80 ms audio, 24 = 240 ms, 40 = 400 ms.
    // Larger chunks amortize Metal kernel-launch overhead: at 80 ms each
    // chunk pays ~170 ms on M1 Q4_K (2.1× realtime); at 240 ms it drops
    // to ~250 ms per 240 ms chunk (~1× realtime). Bit-exact-batch holds
    // for any valid size.
    if (const char* env = getenv("CRISPASR_VOXTRAL4B_STREAM_CHUNK_MS")) {
        const int ms = atoi(env);
        const int frames = (ms / 10 / 8) * 8; // round to multiple of 8 mel frames
        s->chunk_mel_frames = frames > 0 ? frames : 8;
    } else {
        s->chunk_mel_frames = 24; // 240 ms default — phase 2 perf pass
    }
    s->max_audio_seconds = 15;
    s->pcm_next_frame_start = 0;
    s->mel_frames_emitted = 0;
    s->mel_primed = false;
    s->first_chunk_seen = false;
    s->enc_T_so_far = 0;
    s->enc_frames_pending_count = 0;
    s->kv_initialised = false;
    s->n_past = 0;
    s->prefill_done = false;
    s->prefill_n_past = 0;
    s->prefill_out_vocab = 0;
    s->live_decode_enabled = (getenv("CRISPASR_VOXTRAL4B_STREAM_LIVE") != nullptr);
    s->decode_started = false;
    s->decode_finished = false;
    s->decode_logits_committed = false;
    s->decode_n_past = 0;
    s->decode_adapter_pos = 0;
    s->decode_out_vocab = 0;
    s->decode_last_argmax_id = 0;
    s->decode_steps_done = 0;
    s->decoder_thread_enabled = (getenv("CRISPASR_VOXTRAL4B_STREAM_DECODER_THREAD") != nullptr);
    s->shutdown_requested = false;
    s->worker_idle = true;
    s->out_t0_s = 0.0;
    s->out_t1_s = 0.0;
    s->has_output = false;
    s->decode_counter = 0;
    s->total_samples_fed = 0;
    s->conv0_lctx.assign((size_t)2 * ctx->model.hparams.n_mels, 0.0f);
    s->conv1_lctx.assign((size_t)ctx->model.hparams.audio_d_model, 0.0f);
    if (vox_stream_alloc_enc_kv(s) != 0) {
        delete s;
        return nullptr;
    }
    // PLAN #7 phase 4: optionally start a decoder worker thread. Only
    // useful in live-decode mode (in PTT mode all decode happens in flush,
    // no parallelism opportunity). Implies live decode.
    if (s->decoder_thread_enabled) {
        s->live_decode_enabled = true; // decoder thread implies live mode
        s->worker_thread = std::thread(vox_stream_decoder_worker, s);
    }
    // Pre-feed 32 STREAMING_PAD-tokens worth of zero audio so the encoder's
    // first 32 frames are silence-encoded, matching the batch CLI adapter.
    // This is internal padding — not counted toward total_samples_fed.
    {
        const int left_pad_samples = kLeftPadTokens * kSamplesPerToken; // 40960
        std::vector<float> zeros((size_t)left_pad_samples, 0.0f);
        if (voxtral4b_stream_feed(s, zeros.data(), left_pad_samples) != 0) {
            voxtral4b_stream_close(s);
            return nullptr;
        }
        // Don't count the silent prefix in user-visible audio time.
        s->total_samples_fed = 0;
    }
    return s;
}

extern "C" int voxtral4b_stream_feed(struct voxtral4b_stream* s, const float* pcm, int n_samples) {
    if (!s || !pcm || n_samples < 0)
        return -1;
    if (n_samples == 0)
        return 0;
    const int n_fft = 400;
    if (!s->mel_primed) {
        s->pcm_with_pad.assign((size_t)(n_fft / 2), 0.0f); // simulate center_pad-left
        s->pcm_next_frame_start = 0;
        s->mel_primed = true;
    }
    const size_t prev = s->pcm_with_pad.size();
    s->pcm_with_pad.resize(prev + n_samples);
    std::memcpy(s->pcm_with_pad.data() + prev, pcm, (size_t)n_samples * sizeof(float));
    s->total_samples_fed += n_samples;

    // Default: incremental encoder runs during feed (audio embeds bit-exact
    // vs the batch encoder, validated on JFK). Set
    // `CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER=1` to fall back to running the
    // whole encoder at flush time — useful when chasing a regression.
    static const bool use_batch_encoder = (getenv("CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER") != nullptr);
    if (use_batch_encoder)
        return 0;

    // PLAN #7 phase 4: when the decoder thread is running, the worker
    // also calls into ggml's sched on the LLM. Hold sched_mutex around
    // the encoder + projector + prefill so encoder/decoder serialize
    // cleanly on Metal. (No GPU parallelism on M1; the win is feed()
    // returning before the decode catches up — the worker drains in
    // the background.)
    {
        std::lock_guard<std::mutex> sched_lk(s->sched_mutex);
        if (vox_stream_advance_mel(s) < 0)
            return -2;
        const int n_mels = (int)s->ctx->model.hparams.n_mels;
        while ((int)(s->mel_pending.size() / n_mels) >= s->chunk_mel_frames) {
            const int rc = vox_stream_run_encoder_chunk(s);
            if (rc <= 0)
                break;
        }
        if (vox_stream_advance_projector(s) < 0)
            return -3;
        if (vox_stream_maybe_prefill(s) < 0)
            return -4;
    }

    if (s->decoder_thread_enabled) {
        // Worker thread is responsible for decode. Notify it that
        // audio_embeds have grown. Feed returns immediately.
        s->cond_var.notify_one();
    } else if (s->live_decode_enabled && s->decode_started && !s->decode_finished) {
        // Single-threaded live decode: drain inline.
        if (vox_stream_drain_decode(s) < 0)
            return -5;
    }
    return 0;
}

extern "C" int voxtral4b_stream_flush(struct voxtral4b_stream* s) {
    if (!s)
        return -1;
    const auto& hp = s->ctx->model.hparams;
    const int proj_out = (int)hp.proj_out_dim;
    const bool timing = (getenv("CRISPASR_VOXTRAL4B_STREAM_TIMING") != nullptr);
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed_ms = [&t0]() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    // PLAN #7 phase 1.5 default: incremental encoder ran during feed(), so
    // s->audio_embeds is already populated. Flush just drains any residual
    // mel/encoder/projector + runs the LLM decode. Set
    // CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER=1 to ignore the streaming
    // encoder's output and re-run the batch encoder at flush — regression-
    // debug switch matching the same env var on the feed path.
    const bool use_incremental = (getenv("CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER") == nullptr);

    // Right-pad the user audio: align to a SAMPLES_PER_TOKEN boundary, then
    // append kRightPadTokens worth of trailing zeros. Internal padding —
    // not counted toward total_samples_fed.
    //
    // PLAN #7 phase 3: append the right-pad zeros directly to pcm_with_pad
    // WITHOUT triggering encoder chunks per-feed. The right-pad's ~80 mel
    // frames + any residual are then encoded in a single larger combined
    // chunk below, saving ~6 kernel launches (3 encoder + 3 projector) vs
    // the previous "feed-and-chunk" path that ran 3-4 separate encoder
    // chunks during the right-pad feed.
    const int64_t user_samples = s->total_samples_fed;
    const int right_align = (kSamplesPerToken - (int)(user_samples % kSamplesPerToken)) % kSamplesPerToken;
    const int right_pad_total = right_align + kRightPadTokens * kSamplesPerToken;
    if (use_incremental && right_pad_total > 0) {
        // Direct append, bypassing voxtral4b_stream_feed's encoder loop.
        if (!s->mel_primed) {
            const int n_fft = 400;
            s->pcm_with_pad.assign((size_t)(n_fft / 2), 0.0f);
            s->pcm_next_frame_start = 0;
            s->mel_primed = true;
        }
        const size_t prev = s->pcm_with_pad.size();
        s->pcm_with_pad.resize(prev + right_pad_total, 0.0f);
    } else if (right_pad_total > 0) {
        std::vector<float> zeros((size_t)right_pad_total, 0.0f);
        if (voxtral4b_stream_feed(s, zeros.data(), right_pad_total) != 0)
            return -2;
        s->total_samples_fed = user_samples; // un-count the right-pad
    }

    int N_audio = 0;
    std::vector<float> batch_audio_embeds;

    if (use_incremental) {
        const int n_mels = (int)hp.n_mels;
        if (vox_stream_advance_mel(s) < 0)
            return -3;
        // PLAN #7 phase 3: run all remaining mel through ONE larger combined
        // encoder chunk. mel_pending now holds (residual from feed) +
        // (right-pad's mel frames). Round up to a multiple of 8 so the
        // projector's stack-4 alignment works (8 mel = 4 enc = 1 projector
        // group). One graph build instead of N saves ~6 kernel launches
        // (3 encoder + 3 projector) and amortises Metal launch overhead
        // across one larger matmul per layer.
        const int T_pending = (int)(s->mel_pending.size() / (size_t)n_mels);
        if (T_pending > 0) {
            const int T_padded = ((T_pending + 7) / 8) * 8; // round up to multiple of 8
            if (T_padded > T_pending) {
                std::vector<float> padded((size_t)n_mels * T_padded, 0.0f);
                for (int m = 0; m < n_mels; m++) {
                    std::memcpy(padded.data() + (size_t)m * T_padded, s->mel_pending.data() + (size_t)m * T_pending,
                                (size_t)T_pending * sizeof(float));
                }
                s->mel_pending = std::move(padded);
            }
            if (s->enc_T_so_far + T_padded / 2 <= s->enc_max_T) {
                const int saved_chunk = s->chunk_mel_frames;
                s->chunk_mel_frames = T_padded;
                const int rc = vox_stream_run_encoder_chunk(s);
                s->chunk_mel_frames = saved_chunk;
                if (rc < 0)
                    return -4;
            }
        }
        if (vox_stream_advance_projector(s) < 0)
            return -5;
        N_audio = (int)(s->audio_embeds.size() / proj_out);
    } else {
        // Batch path: rebuild the full PCM from pcm_with_pad (excluding the
        // 200-zero center_pad-emulation prefix at the front), run batch mel +
        // encoder, use those embeddings.
        const int n_fft = 400;
        const float* full_pcm = s->pcm_with_pad.data() + (n_fft / 2);
        const int full_n = (int)s->pcm_with_pad.size() - (n_fft / 2);
        if (full_n <= 0)
            return 1; // no audio
        int n_mels_b = 0, T_mel_b = 0;
        float* mel = voxtral4b_compute_mel(s->ctx, full_pcm, full_n, &n_mels_b, &T_mel_b);
        if (!mel)
            return -3;
        int N_enc = 0, dim_b = 0;
        float* aud = voxtral4b_run_encoder(s->ctx, mel, n_mels_b, T_mel_b, &N_enc, &dim_b);
        std::free(mel);
        if (!aud)
            return -4;
        batch_audio_embeds.assign(aud, aud + (size_t)N_enc * dim_b);
        std::free(aud);
        N_audio = N_enc;
    }
    const float* audio_src = use_incremental ? s->audio_embeds.data() : batch_audio_embeds.data();
    if (getenv("CRISPASR_VOXTRAL4B_STREAM_DEBUG")) {
        fprintf(stderr, "voxtral4b_stream: flush enc_T=%d N_audio=%d user_samples=%lld\n", s->enc_T_so_far, N_audio,
                (long long)user_samples);
    }

    // Side-by-side: run BOTH encoders and compare. Useful only in incremental
    // mode (else batch is the source of truth). Prints first divergent embed
    // index, cosine similarity per-embed, and the worst absolute element diff.
    if (use_incremental && getenv("CRISPASR_VOXTRAL4B_STREAM_DIFF")) {
        const int n_fft = 400;
        const float* full_pcm = s->pcm_with_pad.data() + (n_fft / 2);
        const int full_n = (int)s->pcm_with_pad.size() - (n_fft / 2);
        int n_mels_b = 0, T_mel_b = 0;
        float* mel = voxtral4b_compute_mel(s->ctx, full_pcm, full_n, &n_mels_b, &T_mel_b);
        if (mel) {
            int N_enc_b = 0, dim_b = 0;
            float* aud_b = voxtral4b_run_encoder(s->ctx, mel, n_mels_b, T_mel_b, &N_enc_b, &dim_b);
            std::free(mel);
            if (aud_b) {
                fprintf(stderr,
                        "voxtral4b_stream DIFF: stream_N=%d batch_N=%d dim=%d "
                        "stream_T_mel=%d batch_T_mel=%d\n",
                        N_audio, N_enc_b, dim_b, s->enc_T_so_far * 2 /*approx*/, T_mel_b);
                const int n_compare = std::min(N_audio, N_enc_b);
                int first_div = -1;
                for (int i = 0; i < n_compare; i++) {
                    double dot = 0.0, na = 0.0, nb = 0.0, max_abs = 0.0;
                    const float* a = audio_src + (size_t)i * dim_b;
                    const float* b = aud_b + (size_t)i * dim_b;
                    for (int j = 0; j < dim_b; j++) {
                        dot += (double)a[j] * (double)b[j];
                        na += (double)a[j] * (double)a[j];
                        nb += (double)b[j] * (double)b[j];
                        const double d = std::abs((double)a[j] - (double)b[j]);
                        if (d > max_abs)
                            max_abs = d;
                    }
                    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
                    if (cos < 0.999 && first_div < 0)
                        first_div = i;
                    if (i < 8 || (first_div >= 0 && i < first_div + 4))
                        fprintf(stderr,
                                "  embed[%3d] cos=%.6f max_abs=%.4f stream[0..3]=[%.4f,%.4f,%.4f,%.4f] "
                                "batch[0..3]=[%.4f,%.4f,%.4f,%.4f]\n",
                                i, cos, max_abs, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
                }
                fprintf(stderr, "voxtral4b_stream DIFF: first divergent embed = %d\n", first_div);
                std::free(aud_b);
            }
        }
    }
    if (N_audio < kStreamingPromptLen) {
        // Not enough audio for even the streaming prompt.
        return 1;
    }

    if (timing)
        fprintf(stderr, "voxtral4b_stream timing: encoder+projector drain @ %.1f ms\n", elapsed_ms());

    // For the batch (regression-debug) path: overwrite s->audio_embeds
    // with the batch encoder's result so vox_stream_drain_decode reads
    // from the same buffer either way. Discards the streaming-encoder's
    // (probably stale) audio_embeds in the process — fine, we're in the
    // explicit fall-back path.
    if (!use_incremental && !batch_audio_embeds.empty()) {
        s->audio_embeds = std::move(batch_audio_embeds);
        // Drop any speculative-prefill state — it was based on the streaming
        // encoder's audio_embeds and is now stale.
        s->prefill_done = false;
        s->decode_started = false;
        s->decode_logits.clear();
    }

    // Run the streaming-prompt prefill if it didn't happen during feed
    // (very short utterances, or batch path). Sets up s->prefill_logits
    // + decode_n_past + decode_adapter_pos for vox_stream_drain_decode.
    constexpr int kMaxNewTokens = 512;
    if (!s->prefill_done) {
        std::vector<int32_t> prompt_ids((size_t)kStreamingPromptLen);
        prompt_ids[0] = 1;
        for (int i = 1; i < kStreamingPromptLen; i++)
            prompt_ids[i] = 32;
        float* prompt_emb = voxtral4b_embed_tokens(s->ctx, prompt_ids.data(), kStreamingPromptLen);
        if (!prompt_emb)
            return -6;
        for (int i = 0; i < kStreamingPromptLen && i < N_audio; i++) {
            const float* src = s->audio_embeds.data() + (size_t)i * proj_out;
            float* dst = prompt_emb + (size_t)i * proj_out;
            for (int j = 0; j < proj_out; j++)
                dst[j] += src[j];
        }
        if (!s->kv_initialised) {
            if (!voxtral4b_kv_init(s->ctx, kStreamingPromptLen + kMaxNewTokens + 16)) {
                std::free(prompt_emb);
                return -7;
            }
            s->kv_initialised = true;
        }
        voxtral4b_kv_reset(s->ctx);
        auto t_pre = std::chrono::steady_clock::now();
        int out_n_tok = 0, out_vocab = 0;
        float* logits = voxtral4b_run_llm_kv(s->ctx, prompt_emb, kStreamingPromptLen, 0, &out_n_tok, &out_vocab);
        std::free(prompt_emb);
        if (!logits || out_vocab <= 0)
            return -8;
        if (timing)
            fprintf(stderr, "voxtral4b_stream timing: prefill (39 tokens) %.1f ms; total @ %.1f ms\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pre).count(),
                    elapsed_ms());
        const float* last = logits + (size_t)(out_n_tok - 1) * (size_t)out_vocab;
        s->prefill_logits.assign(last, last + out_vocab);
        std::free(logits);
        s->prefill_n_past = kStreamingPromptLen;
        s->prefill_out_vocab = out_vocab;
        s->prefill_done = true;
        s->decode_logits = s->prefill_logits;
        s->decode_n_past = kStreamingPromptLen;
        s->decode_adapter_pos = kStreamingPromptLen;
        s->decode_out_vocab = out_vocab;
        s->decode_started = true;
    } else if (timing) {
        fprintf(stderr, "voxtral4b_stream timing: prefill REUSED (saved ~250 ms); total @ %.1f ms\n", elapsed_ms());
    }

    // Drain decode — runs all remaining tokens until EOS or audio exhausted.
    // In live mode many will already be drained from feed; in PTT mode this
    // does the entire decode loop. With decoder_thread_enabled, signal
    // the worker and wait for it to be idle.
    auto t_decode_start = std::chrono::steady_clock::now();
    const int decode_steps_before = s->decode_steps_done;

    if (s->decoder_thread_enabled) {
        // Wake worker so it picks up any final audio_embeds from the
        // flush-time encoder drain above. Then wait until it goes idle.
        s->cond_var.notify_one();
        for (;;) {
            // Hand off briefly so worker can grab the lock if needed.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::unique_lock<std::mutex> lk(s->decode_state_mutex);
            const int proj_out = (int)s->ctx->model.hparams.proj_out_dim;
            const int N_audio_now = (int)(s->audio_embeds.size() / proj_out);
            const bool caught_up = s->decode_finished || s->decode_adapter_pos >= N_audio_now;
            if (caught_up && s->worker_idle.load())
                break;
            // Re-notify in case worker missed our earlier signal due to
            // a tight wait/check race.
            lk.unlock();
            s->cond_var.notify_one();
        }
    } else {
        if (vox_stream_drain_decode(s) < 0)
            return -9;
    }
    const int new_decode_steps = s->decode_steps_done - decode_steps_before;
    if (timing && new_decode_steps > 0) {
        const double decode_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_decode_start).count();
        fprintf(stderr,
                "voxtral4b_stream timing: drain %d decode steps in %.1f ms (avg %.1f ms/step); flush total %.1f ms\n",
                new_decode_steps, decode_ms, decode_ms / std::max(1, new_decode_steps), elapsed_ms());
    }

    // Trim leading/trailing whitespace from out_text (live mode may have
    // committed leading spaces from `▁` SP-marker tokens).
    while (!s->out_text.empty() && (s->out_text.front() == ' ' || s->out_text.front() == '\t'))
        s->out_text.erase(s->out_text.begin());
    while (!s->out_text.empty() && (s->out_text.back() == ' ' || s->out_text.back() == '\t'))
        s->out_text.pop_back();
    s->out_t0_s = 0.0;
    s->out_t1_s = (double)s->total_samples_fed / 16000.0;
    s->has_output = true;
    s->decode_counter += 1;
    return 1;
}

extern "C" int voxtral4b_stream_get_text(struct voxtral4b_stream* s, char* out, int cap, double* out_t0_s,
                                         double* out_t1_s, int64_t* out_decode_counter) {
    if (!s || !out || cap <= 0)
        return -1;
    // PLAN #7 phase 4: protect out_text_unread + counter from concurrent
    // worker-thread writes.
    std::lock_guard<std::mutex> lk(s->decode_state_mutex);
    if (s->out_text_unread.empty()) {
        out[0] = '\0';
        if (out_t0_s)
            *out_t0_s = s->out_t0_s;
        if (out_t1_s)
            *out_t1_s = s->out_t1_s;
        if (out_decode_counter)
            *out_decode_counter = s->decode_counter;
        return 0;
    }
    const int copy_n = std::min((int)s->out_text_unread.size(), cap - 1);
    std::memcpy(out, s->out_text_unread.data(), copy_n);
    out[copy_n] = '\0';
    s->out_text_unread.erase(0, copy_n);
    if (out_t0_s)
        *out_t0_s = s->out_t0_s;
    if (out_t1_s)
        *out_t1_s = s->out_t1_s;
    if (out_decode_counter)
        *out_decode_counter = s->decode_counter;
    return copy_n;
}

extern "C" void voxtral4b_stream_set_live_decode(struct voxtral4b_stream* s, int enabled) {
    if (!s)
        return;
    s->live_decode_enabled = (enabled != 0);
}

extern "C" void voxtral4b_stream_close(struct voxtral4b_stream* s) {
    if (!s)
        return;
    // PLAN #7 phase 4: shut down + join the decoder worker thread.
    if (s->worker_thread.joinable()) {
        s->shutdown_requested.store(true);
        s->cond_var.notify_all();
        s->worker_thread.join();
    }
    if (s->enc_kv_buf)
        ggml_backend_buffer_free(s->enc_kv_buf);
    if (s->enc_kv_ctx)
        ggml_free(s->enc_kv_ctx);
    delete s;
}
