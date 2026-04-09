// voxtral4b.cpp — Mistral Voxtral-Mini-4B-Realtime-2602 ggml runtime
//
// Key architectural differences from voxtral.cpp (3B):
//   - Audio encoder: RoPE, SwiGLU FFN, RMSNorm, sliding window (750)
//   - LLM: 26 layers, FFN=9216, SWA(8192), adaptive RMSNorm, tied embeddings
//   - Same projector topology (stack-4-frames + 2×Linear)

#include "voxtral4b.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct voxtral4b_hparams {
    uint32_t sample_rate    = 16000;
    uint32_t n_mels         = 128;
    uint32_t n_fft          = 400;
    uint32_t hop_length     = 160;

    // Audio encoder (RoPE + SwiGLU + RMSNorm)
    uint32_t audio_n_layers = 32;
    uint32_t audio_d_model  = 1280;
    uint32_t audio_n_heads  = 32;
    uint32_t audio_head_dim = 64;
    uint32_t audio_ff_dim   = 5120;
    uint32_t audio_max_pos  = 1500;
    float    audio_rope_theta = 1e6f;
    uint32_t audio_swa      = 750;

    // Projector
    uint32_t proj_in_dim    = 5120;
    uint32_t proj_out_dim   = 3072;
    uint32_t proj_frame_stack = 4;

    // LLM (Llama-style + ada_rms_norm + SWA + tied embed)
    uint32_t llm_n_layers   = 26;
    uint32_t llm_d_model    = 3072;
    uint32_t llm_n_heads    = 32;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim   = 128;
    uint32_t llm_ff_dim     = 9216;
    float    llm_rope_theta = 1e6f;
    float    llm_rms_eps    = 1e-5f;
    uint32_t llm_vocab_size = 131072;
    uint32_t llm_max_pos    = 131072;
    uint32_t llm_swa        = 8192;
    uint32_t ada_norm_dim   = 32;
    bool     tied_embeddings = true;

    uint32_t audio_token_id = 24;
};

// ===========================================================================
// Model tensors
// ===========================================================================

struct voxtral4b_audio_block {
    ggml_tensor * attn_norm_w  = nullptr;  // RMSNorm (no bias)
    ggml_tensor * attn_q_w    = nullptr;
    ggml_tensor * attn_q_b    = nullptr;
    ggml_tensor * attn_k_w    = nullptr;  // no bias
    ggml_tensor * attn_v_w    = nullptr;
    ggml_tensor * attn_v_b    = nullptr;
    ggml_tensor * attn_out_w  = nullptr;
    ggml_tensor * attn_out_b  = nullptr;
    ggml_tensor * ffn_norm_w  = nullptr;  // RMSNorm (no bias)
    // SwiGLU: gate + up + down
    ggml_tensor * ffn_gate_w  = nullptr;
    ggml_tensor * ffn_up_w    = nullptr;
    ggml_tensor * ffn_down_w  = nullptr;
    ggml_tensor * ffn_down_b  = nullptr;
};

struct voxtral4b_llm_block {
    ggml_tensor * attn_norm_w   = nullptr;
    ggml_tensor * attn_q_w     = nullptr;
    ggml_tensor * attn_k_w     = nullptr;
    ggml_tensor * attn_v_w     = nullptr;
    ggml_tensor * attn_out_w   = nullptr;
    ggml_tensor * ffn_norm_w   = nullptr;
    ggml_tensor * ffn_gate_w   = nullptr;
    ggml_tensor * ffn_up_w     = nullptr;
    ggml_tensor * ffn_down_w   = nullptr;
    // Adaptive RMSNorm
    ggml_tensor * ada_down_w   = nullptr;  // (ada_dim, d_model)
    ggml_tensor * ada_up_w     = nullptr;  // (d_model, ada_dim)
};

struct voxtral4b_model {
    voxtral4b_hparams hparams;

    struct {
        ggml_tensor * conv1_w = nullptr;
        ggml_tensor * conv1_b = nullptr;
        ggml_tensor * conv2_w = nullptr;
        ggml_tensor * conv2_b = nullptr;
        // No embed_positions — RoPE instead
        ggml_tensor * ln_post_w = nullptr;  // RMSNorm post-encoder
        ggml_tensor * mel_filters = nullptr;
        ggml_tensor * mel_window  = nullptr;
        std::vector<voxtral4b_audio_block> blocks;
    } audio;

    struct {
        ggml_tensor * proj1 = nullptr;
        ggml_tensor * proj2 = nullptr;
    } projector;

    struct {
        ggml_tensor * token_embd_w  = nullptr;
        ggml_tensor * output_norm_w = nullptr;
        // No output_w — tied to token_embd_w
        std::vector<voxtral4b_llm_block> blocks;
    } llm;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
};

// Vocab (same structure as 3B)
struct voxtral4b_vocab {
    std::vector<uint8_t> tekken_vocab_blob;
    std::vector<uint32_t> rank_offset;
    std::vector<uint32_t> rank_length;
    std::vector<std::string> specials;
    std::unordered_map<std::string, int32_t> special_to_rank;
    int n_specials = 0;
    int n_vocab    = 0;
    std::string pre_pattern;
    std::unordered_map<std::string, int32_t> bytes_to_rank;
    bool reverse_built = false;
};

struct voxtral4b_context {
    voxtral4b_context_params params;
    voxtral4b_model model;
    voxtral4b_vocab vocab;

    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    std::vector<uint8_t> compute_meta;

    ggml_context *        kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor *         kv_k   = nullptr;
    ggml_tensor *         kv_v   = nullptr;

    int n_threads = 4;
    int delay_tokens = 6;  // 480ms default
    std::vector<float> ada_scales;  // (n_layers × d_model), precomputed
};

// ===========================================================================
// GGUF loader
// ===========================================================================

static uint32_t kv_u32(gguf_context * g, const char * key, uint32_t def) {
    int i = gguf_find_key(g, key);
    return i >= 0 ? (uint32_t)gguf_get_val_u32(g, i) : def;
}
static float kv_f32(gguf_context * g, const char * key, float def) {
    int i = gguf_find_key(g, key);
    return i >= 0 ? gguf_get_val_f32(g, i) : def;
}

static bool voxtral4b_load_model(voxtral4b_model & model, voxtral4b_vocab & vocab,
                                 const char * path, ggml_backend_t backend) {
    // Pass 1: metadata
    {
        gguf_init_params mp = { /*no_alloc=*/true, /*ctx=*/nullptr };
        gguf_context * gctx = gguf_init_from_file(path, mp);
        if (!gctx) { fprintf(stderr, "voxtral4b: failed to open '%s'\n", path); return false; }
        auto & hp = model.hparams;

        hp.audio_n_layers = kv_u32(gctx, "voxtral4b.audio.n_layers", hp.audio_n_layers);
        hp.audio_d_model  = kv_u32(gctx, "voxtral4b.audio.d_model",  hp.audio_d_model);
        hp.audio_n_heads  = kv_u32(gctx, "voxtral4b.audio.n_heads",  hp.audio_n_heads);
        hp.audio_head_dim = kv_u32(gctx, "voxtral4b.audio.head_dim", hp.audio_head_dim);
        hp.audio_ff_dim   = kv_u32(gctx, "voxtral4b.audio.ff_dim",   hp.audio_ff_dim);
        hp.audio_max_pos  = kv_u32(gctx, "voxtral4b.audio.max_pos",  hp.audio_max_pos);
        hp.audio_rope_theta = kv_f32(gctx, "voxtral4b.audio.rope_theta", hp.audio_rope_theta);
        hp.audio_swa      = kv_u32(gctx, "voxtral4b.audio.sliding_window", hp.audio_swa);

        hp.proj_in_dim    = kv_u32(gctx, "voxtral4b.proj.in_dim",      hp.proj_in_dim);
        hp.proj_out_dim   = kv_u32(gctx, "voxtral4b.proj.out_dim",     hp.proj_out_dim);
        hp.proj_frame_stack = kv_u32(gctx, "voxtral4b.proj.frame_stack", hp.proj_frame_stack);

        hp.llm_n_layers   = kv_u32(gctx, "voxtral4b.llm.n_layers",     hp.llm_n_layers);
        hp.llm_d_model    = kv_u32(gctx, "voxtral4b.llm.d_model",      hp.llm_d_model);
        hp.llm_n_heads    = kv_u32(gctx, "voxtral4b.llm.n_heads",      hp.llm_n_heads);
        hp.llm_n_kv_heads = kv_u32(gctx, "voxtral4b.llm.n_kv_heads",   hp.llm_n_kv_heads);
        hp.llm_head_dim   = kv_u32(gctx, "voxtral4b.llm.head_dim",     hp.llm_head_dim);
        hp.llm_ff_dim     = kv_u32(gctx, "voxtral4b.llm.ff_dim",       hp.llm_ff_dim);
        hp.llm_rope_theta = kv_f32(gctx, "voxtral4b.llm.rope_theta",   hp.llm_rope_theta);
        hp.llm_rms_eps    = kv_f32(gctx, "voxtral4b.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = kv_u32(gctx, "voxtral4b.llm.vocab_size",   hp.llm_vocab_size);
        hp.llm_max_pos    = kv_u32(gctx, "voxtral4b.llm.max_pos",      hp.llm_max_pos);
        hp.llm_swa        = kv_u32(gctx, "voxtral4b.llm.sliding_window", hp.llm_swa);
        hp.ada_norm_dim   = kv_u32(gctx, "voxtral4b.llm.ada_norm_dim", hp.ada_norm_dim);
        hp.audio_token_id = kv_u32(gctx, "voxtral4b.audio_token_id",   hp.audio_token_id);

        // Tekken tokenizer
        int kp = gguf_find_key(gctx, "tokenizer.tekken.pattern");
        if (kp >= 0) vocab.pre_pattern = gguf_get_val_str(gctx, kp);
        int ks = gguf_find_key(gctx, "tokenizer.tekken.specials");
        if (ks >= 0) {
            int n = gguf_get_arr_n(gctx, ks);
            vocab.specials.resize(n);
            vocab.special_to_rank.reserve(n);
            for (int i = 0; i < n; i++) {
                vocab.specials[i] = gguf_get_arr_str(gctx, ks, i);
                vocab.special_to_rank[vocab.specials[i]] = i;
            }
        }
        vocab.n_specials = kv_u32(gctx, "tokenizer.tekken.n_specials", 1000);
        vocab.n_vocab    = kv_u32(gctx, "tokenizer.tekken.n_vocab",    150000);

        gguf_free(gctx);
    }

    // Pass 2: load tensors
    ggml_context * weight_ctx = nullptr;
    {
        gguf_init_params lp = { /*no_alloc=*/true, /*ctx=*/&weight_ctx };
        gguf_context * gctx = gguf_init_from_file(path, lp);
        if (!gctx || !weight_ctx) return false;

        model.buf = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);

        int fd = open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st; fstat(fd, &st);
        void * mmap_base = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);

        int n_tensors = gguf_get_n_tensors(gctx);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(gctx, i);
            ggml_tensor * t = ggml_get_tensor(weight_ctx, name);
            if (!t) continue;
            size_t offset = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, i);
            ggml_backend_tensor_set(t, (const char *)mmap_base + offset, 0, ggml_nbytes(t));
            model.tensors[name] = t;
        }
        munmap(mmap_base, (size_t)st.st_size);
        gguf_free(gctx);
    }
    model.ctx = weight_ctx;

    // Bind tensors
    auto get = [&](const std::string & n) -> ggml_tensor * {
        auto it = model.tensors.find(n);
        return it != model.tensors.end() ? it->second : nullptr;
    };
    auto require = [&](const std::string & n) -> ggml_tensor * {
        auto * t = get(n);
        if (!t) fprintf(stderr, "voxtral4b: missing tensor '%s'\n", n.c_str());
        return t;
    };

    auto & a = model.audio;
    a.conv1_w = require("audio.conv.1.weight");
    a.conv1_b = require("audio.conv.1.bias");
    a.conv2_w = require("audio.conv.2.weight");
    a.conv2_b = require("audio.conv.2.bias");
    a.ln_post_w = require("audio.ln_post.weight");
    a.mel_filters = get("audio.mel_filters");
    a.mel_window  = get("audio.mel_window");

    a.blocks.resize(model.hparams.audio_n_layers);
    for (uint32_t il = 0; il < model.hparams.audio_n_layers; il++) {
        auto pfx = "audio.blk." + std::to_string(il) + ".";
        auto & b = a.blocks[il];
        b.attn_norm_w  = require(pfx + "attn_norm.weight");
        b.attn_q_w     = require(pfx + "attn_q.weight");
        b.attn_q_b     = get(pfx + "attn_q.bias");
        b.attn_k_w     = require(pfx + "attn_k.weight");
        b.attn_v_w     = require(pfx + "attn_v.weight");
        b.attn_v_b     = get(pfx + "attn_v.bias");
        b.attn_out_w   = require(pfx + "attn_out.weight");
        b.attn_out_b   = get(pfx + "attn_out.bias");
        b.ffn_norm_w   = require(pfx + "ffn_norm.weight");
        b.ffn_gate_w   = require(pfx + "ffn_gate.weight");
        b.ffn_up_w     = require(pfx + "ffn_up.weight");
        b.ffn_down_w   = require(pfx + "ffn_down.weight");
        b.ffn_down_b   = get(pfx + "ffn_down.bias");
    }

    // Tekken vocab blob
    {
        ggml_tensor * vt = get("tokenizer.tekken.vocab_tensor");
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
                if (pos + 2 > n) break;
                uint16_t len;
                std::memcpy(&len, vocab.tekken_vocab_blob.data() + pos, 2);
                pos += 2;
                if (pos + len > n) break;
                vocab.rank_offset.push_back((uint32_t)pos);
                vocab.rank_length.push_back(len);
                pos += len;
            }
        }
    }

    auto & p = model.projector;
    p.proj1 = require("proj1.weight");
    p.proj2 = require("proj2.weight");

    auto & l = model.llm;
    l.token_embd_w  = require("token_embd.weight");
    l.output_norm_w = require("output_norm.weight");
    // No output_w — tied to token_embd_w

    l.blocks.resize(model.hparams.llm_n_layers);
    for (uint32_t il = 0; il < model.hparams.llm_n_layers; il++) {
        auto pfx = "blk." + std::to_string(il) + ".";
        auto & b = l.blocks[il];
        b.attn_norm_w   = require(pfx + "attn_norm.weight");
        b.attn_q_w      = require(pfx + "attn_q.weight");
        b.attn_k_w      = require(pfx + "attn_k.weight");
        b.attn_v_w      = require(pfx + "attn_v.weight");
        b.attn_out_w    = require(pfx + "attn_output.weight");
        b.ffn_norm_w    = require(pfx + "ffn_norm.weight");
        b.ffn_gate_w    = require(pfx + "ffn_gate.weight");
        b.ffn_up_w      = require(pfx + "ffn_up.weight");
        b.ffn_down_w    = require(pfx + "ffn_down.weight");
        b.ada_down_w    = get(pfx + "ada_norm_down.weight");
        b.ada_up_w      = get(pfx + "ada_norm_up.weight");
    }

    fprintf(stderr, "voxtral4b: loaded %d audio tensors + %d LLM tensors\n",
            (int)(model.hparams.audio_n_layers * 13 + 5),
            (int)(model.hparams.llm_n_layers * 11 + 2));
    return true;
}

// ===========================================================================
// Mel / FFT (same as 3B — CPU-only)
// ===========================================================================

static void voxtral4b_dft(const float * in, int N, float * out) {
    for (int k = 0; k < N; k++) {
        double re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * k * n / N;
            re += in[n] * cos(a);
            im += in[n] * sin(a);
        }
        out[2*k] = (float)re;
        out[2*k+1] = (float)im;
    }
}

static void voxtral4b_fft(float * in, int N, float * out) {
    if (N <= 1) { out[0] = in[0]; out[1] = 0; return; }
    if (N % 2 != 0) { voxtral4b_dft(in, N, out); return; }
    int half = N / 2;
    std::vector<float> even(half), odd(half);
    for (int i = 0; i < half; i++) { even[i] = in[2*i]; odd[i] = in[2*i+1]; }
    std::vector<float> E(2*half), O(2*half);
    voxtral4b_fft(even.data(), half, E.data());
    voxtral4b_fft(odd.data(),  half, O.data());
    for (int k = 0; k < half; k++) {
        double a = -2.0 * M_PI * k / N;
        float wr = (float)cos(a), wi = (float)sin(a);
        float tre = wr*O[2*k] - wi*O[2*k+1];
        float tim = wr*O[2*k+1] + wi*O[2*k];
        out[2*k]         = E[2*k] + tre;
        out[2*k+1]       = E[2*k+1] + tim;
        out[2*(k+half)]   = E[2*k] - tre;
        out[2*(k+half)+1] = E[2*k+1] - tim;
    }
}

extern "C" float * voxtral4b_compute_mel(voxtral4b_context * ctx,
                                         const float * samples, int n_samples,
                                         int * out_n_mels, int * out_T_mel) {
    if (!ctx || !samples || n_samples <= 0) return nullptr;
    if (!ctx->model.audio.mel_filters || !ctx->model.audio.mel_window) return nullptr;
    const int n_fft = 400, hop = 160, n_mels = 128, n_freqs = 201;

    std::vector<float> hann(n_fft);
    ggml_backend_tensor_get(ctx->model.audio.mel_window, hann.data(), 0, n_fft*sizeof(float));
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->model.audio.mel_filters, filt.data(), 0, filt.size()*sizeof(float));

    const int pad = n_fft / 2;
    std::vector<float> padded((size_t)n_samples + 2*pad, 0.0f);
    std::memcpy(padded.data() + pad, samples, n_samples*sizeof(float));

    const int T_full = (int)((padded.size() - n_fft) / hop + 1);
    const int T = T_full - 1;

    std::vector<float> power((size_t)n_freqs * T, 0.0f);
    {
        std::vector<float> fi((size_t)n_fft*4, 0.0f), fo((size_t)n_fft*8, 0.0f);
        for (int t = 0; t < T; t++) {
            const float * frame = padded.data() + (size_t)t*hop;
            for (int n = 0; n < n_fft; n++) fi[n] = frame[n]*hann[n];
            voxtral4b_fft(fi.data(), n_fft, fo.data());
            for (int k = 0; k < n_freqs; k++) {
                float re = fo[2*k], im = fo[2*k+1];
                power[(size_t)k*T+t] = re*re + im*im;
            }
        }
    }

    // Make T even for conv stride-2 (drop first frame if odd, matching voxmlx)
    int T_even = T;
    int t_offset = 0;
    if (T % 2 != 0) { T_even = T - 1; t_offset = 1; }

    // Compute mel spectrogram — no padding to fixed size!
    std::vector<float> mel((size_t)n_mels * T_even, 0.0f);
    for (int m = 0; m < n_mels; m++) {
        for (int t = 0; t < T_even; t++) {
            double s = 0.0;
            for (int k = 0; k < n_freqs; k++)
                s += (double)filt[(size_t)k*n_mels+m] * power[(size_t)k*(T) + (t + t_offset)];
            float lv = std::log10(std::max((float)s, 1e-10f));
            mel[(size_t)m*T_even+t] = lv;
        }
    }

    // VoxtralRealtime normalization with FIXED global_log_mel_max=1.5
    const float global_log_mel_max = 1.5f;
    const float floor_v = global_log_mel_max - 8.0f;
    for (size_t i = 0; i < mel.size(); i++) {
        float v = mel[i]; if (v < floor_v) v = floor_v;
        mel[i] = (v + 4.0f) / 4.0f;
    }

    if (out_n_mels) *out_n_mels = n_mels;
    if (out_T_mel)  *out_T_mel  = T_even;
    float * result = (float*)malloc(mel.size()*sizeof(float));
    std::memcpy(result, mel.data(), mel.size()*sizeof(float));
    return result;
}

// ===========================================================================
// Audio encoder graph — RoPE + SwiGLU + RMSNorm + sliding window
// ===========================================================================

static const float kRmsEps = 1e-5f;

static ggml_cgraph * voxtral4b_build_graph_encoder(voxtral4b_context * ctx, int T_mel) {
    const auto & m  = ctx->model;
    const auto & hp = m.hparams;
    const int d         = (int)hp.audio_d_model;
    const int n_heads   = (int)hp.audio_n_heads;
    const int head_dim  = (int)hp.audio_head_dim;
    const int n_layers  = (int)hp.audio_n_layers;
    const int proj_in   = (int)hp.proj_in_dim;
    const int n_mels    = (int)hp.n_mels;
    const int swa       = (int)hp.audio_swa;
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    ggml_init_params ip = {
        ctx->compute_meta.size(), ctx->compute_meta.data(), true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: mel spectrogram (n_mels=128, T_mel=3000) F32
    // ggml layout: ne[0]=T_mel, ne[1]=n_mels → row-major (T_mel, n_mels)
    ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // Causal Conv1d front-end: left-pad only (padding_total = kernel_size - stride)
    // Conv0: k=3, s=1 → left_pad=2, Conv1: k=3, s=2 → left_pad=1
    auto bias_1d = [&](ggml_tensor * b) {
        return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1);
    };

    // Causal padding for conv0: pad 2 zeros on the left of the time dimension
    // mel is (T_mel, n_mels), conv_1d treats ne[0] as time
    // ggml_pad pads at the end; for left-padding we use ggml_concat with a zero tensor
    ggml_tensor * pad0 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, n_mels);
    ggml_set_name(pad0, "conv0_lpad");
    ggml_set_input(pad0);  // will be set to zeros
    ggml_tensor * mel_padded = ggml_concat(ctx0, pad0, mel, 0);  // concat along dim 0 (time)

    ggml_tensor * cur = ggml_conv_1d(ctx0, m.audio.conv1_w, mel_padded, 1, 0, 1);  // pad=0!
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);

    // Causal padding for conv1: pad 1 zero on the left
    ggml_tensor * pad1 = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, d, 1);
    ggml_set_name(pad1, "conv1_lpad");
    ggml_set_input(pad1);  // will be set to zeros
    // cur is (T, d, 1) from conv_1d output; concat a (1, d, 1) zero on the left
    cur = ggml_concat(ctx0, ggml_reshape_3d(ctx0, pad1, 1, d, 1), cur, 0);

    cur = ggml_conv_1d(ctx0, m.audio.conv2_w, cur, 2, 0, 1);  // pad=0!
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);

    // Output length: conv0 out = T_mel (same with causal pad), conv1 out = ceil(T_mel/2)
    const int T_enc = (T_mel + 1) / 2;  // ceil division for stride 2
    cur = ggml_reshape_2d(ctx0, cur, T_enc, d);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));  // (d, T_enc)

    // RoPE positions for encoder
    ggml_tensor * pos_enc = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_enc);
    ggml_set_name(pos_enc, "enc_positions");
    ggml_set_input(pos_enc);

    // Sliding window attention mask (optional — only if swa < T_enc)
    ggml_tensor * swa_mask = nullptr;
    if (swa > 0 && swa < T_enc) {
        swa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T_enc, T_enc);
        ggml_set_name(swa_mask, "swa_mask");
        ggml_set_input(swa_mask);
    }

    // 32 × encoder blocks
    for (int il = 0; il < n_layers; il++) {
        const auto & b = m.audio.blocks[il];
        ggml_tensor * residual = cur;

        // Pre-RMSNorm
        ggml_tensor * x = ggml_rms_norm(ctx0, cur, kRmsEps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // Q, K, V projections
        ggml_tensor * Q = ggml_mul_mat(ctx0, b.attn_q_w, x);
        if (b.attn_q_b) Q = ggml_add(ctx0, Q, b.attn_q_b);
        ggml_tensor * K = ggml_mul_mat(ctx0, b.attn_k_w, x);
        ggml_tensor * V = ggml_mul_mat(ctx0, b.attn_v_w, x);
        if (b.attn_v_b) V = ggml_add(ctx0, V, b.attn_v_b);

        // Reshape to (head_dim, n_heads, T_enc)
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, T_enc);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_heads, T_enc);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T_enc);

        // Apply RoPE to Q and K
        Q = ggml_rope_ext(ctx0, Q, pos_enc, nullptr, head_dim, 0, 0,
                          hp.audio_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, pos_enc, nullptr, head_dim, 0, 0,
                          hp.audio_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute for attention: (hd, T, n_h)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        // Flash attention with optional SWA mask
        ggml_tensor * attn = ggml_flash_attn_ext(ctx0, Q, K, V, swa_mask,
                                                  attn_scale, 0.0f, 0.0f);
        // Flash attn output: (hd, nh, T, 1) → reshape to (nh*hd, T) = (2048, T_enc)
        attn = ggml_reshape_2d(ctx0, attn, n_heads * head_dim, T_enc);

        // Output projection
        attn = ggml_mul_mat(ctx0, b.attn_out_w, attn);
        if (b.attn_out_b) attn = ggml_add(ctx0, attn, b.attn_out_b);
        cur = ggml_add(ctx0, residual, attn);

        // FFN: Pre-RMSNorm + SwiGLU
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, kRmsEps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);

        ggml_tensor * gate = ggml_mul_mat(ctx0, b.ffn_gate_w, x);
        gate = ggml_silu(ctx0, gate);
        ggml_tensor * up = ggml_mul_mat(ctx0, b.ffn_up_w, x);
        ggml_tensor * ffn = ggml_mul(ctx0, gate, up);
        ffn = ggml_mul_mat(ctx0, b.ffn_down_w, ffn);
        if (b.ffn_down_b) ffn = ggml_add(ctx0, ffn, b.ffn_down_b);
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
static void voxtral4b_compute_t_cond(float * out, int d, float t_value) {
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
static std::vector<float> voxtral4b_compute_ada_scales(
    voxtral4b_context * ctx, int delay_tokens)
{
    const auto & hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int ada_dim = (int)hp.ada_norm_dim;
    const int n_layers = (int)hp.llm_n_layers;

    // t_cond: sinusoidal embedding of delay_tokens
    std::vector<float> t_cond(d);
    voxtral4b_compute_t_cond(t_cond.data(), d, (float)delay_tokens);

    std::vector<float> all_scales((size_t)n_layers * d, 0.0f);

    for (int il = 0; il < n_layers; il++) {
        const auto & b = ctx->model.llm.blocks[il];
        if (!b.ada_down_w || !b.ada_up_w) continue;

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
        float * scale = all_scales.data() + (size_t)il * d;
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

static ggml_cgraph * voxtral4b_build_graph_llm_kv(voxtral4b_context * ctx,
                                                   int n_past, int n_tokens) {
    const auto & m  = ctx->model;
    const auto & hp = m.hparams;
    const int d        = (int)hp.llm_d_model;
    const int n_q      = (int)hp.llm_n_heads;
    const int n_kv     = (int)hp.llm_n_kv_heads;
    const int hd       = (int)hp.llm_head_dim;
    const int n_layers = (int)hp.llm_n_layers;
    const int ff       = (int)hp.llm_ff_dim;
    const int vocab    = (int)hp.llm_vocab_size;

    ggml_init_params ip = {
        ctx->compute_meta.size(), ctx->compute_meta.data(), true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor * embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_tokens);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor * causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past + n_tokens, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    // Ada-scale per layer: (n_layers, d) — precomputed on CPU, passed as input
    ggml_tensor * ada_scales = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_layers);
    ggml_set_name(ada_scales, "ada_scales");
    ggml_set_input(ada_scales);

    ggml_tensor * cur = embeds;

    for (int il = 0; il < n_layers; il++) {
        const auto & b = m.llm.blocks[il];
        ggml_tensor * residual = cur;

        // Pre-RMSNorm
        cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        cur = ggml_mul(ctx0, cur, b.attn_norm_w);

        // GQA self-attention
        ggml_tensor * Q = ggml_mul_mat(ctx0, b.attn_q_w, cur);
        ggml_tensor * K = ggml_mul_mat(ctx0, b.attn_k_w, cur);
        ggml_tensor * V = ggml_mul_mat(ctx0, b.attn_v_w, cur);

        Q = ggml_reshape_3d(ctx0, Q, hd, n_q,  n_tokens);
        K = ggml_reshape_3d(ctx0, K, hd, n_kv, n_tokens);
        V = ggml_reshape_3d(ctx0, V, hd, n_kv, n_tokens);

        // RoPE
        Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, 0, 0,
                          hp.llm_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, 0, 0,
                          hp.llm_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // KV cache write
        const size_t kv_offset = (size_t)il * ctx->kv_k->nb[3] +
                                 (size_t)n_past * ctx->kv_k->nb[1];
        ggml_tensor * k_dst = ggml_view_3d(ctx0, ctx->kv_k, hd, n_tokens, n_kv,
                                           ctx->kv_k->nb[1], ctx->kv_k->nb[2], kv_offset);
        ggml_tensor * v_dst = ggml_view_3d(ctx0, ctx->kv_v, hd, n_tokens, n_kv,
                                           ctx->kv_v->nb[1], ctx->kv_v->nb[2],
                                           (size_t)il * ctx->kv_v->nb[3] +
                                           (size_t)n_past * ctx->kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V, v_dst));

        // KV cache read
        int Lk = n_past + n_tokens;
        ggml_tensor * Kfull = ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_k, hd, Lk, n_kv,
                                        ctx->kv_k->nb[1], ctx->kv_k->nb[2],
                                        (size_t)il * ctx->kv_k->nb[3]));
        ggml_tensor * Vfull = ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_v, hd, Lk, n_kv,
                                        ctx->kv_v->nb[1], ctx->kv_v->nb[2],
                                        (size_t)il * ctx->kv_v->nb[3]));

        // GQA expansion: repeat KV heads to match Q heads (32/8=4× repeat)
        const int n_kv_grp = n_q / n_kv;
        if (n_kv_grp > 1) {
            ggml_tensor * K4 = ggml_reshape_4d(ctx0, Kfull, hd, Lk, 1, n_kv);
            ggml_tensor * V4 = ggml_reshape_4d(ctx0, Vfull, hd, Lk, 1, n_kv);
            K4 = ggml_repeat_4d(ctx0, K4, hd, Lk, n_kv_grp, n_kv);
            V4 = ggml_repeat_4d(ctx0, V4, hd, Lk, n_kv_grp, n_kv);
            Kfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, Lk, n_q));
            Vfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, Lk, n_q));
        }

        // Flash attention
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor * attn = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, causal_mask,
                                                  scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, n_q * hd, n_tokens);
        attn = ggml_mul_mat(ctx0, b.attn_out_w, attn);
        cur = ggml_add(ctx0, residual, attn);

        // FFN: Post-attention RMSNorm + ada_rms_norm conditioning + SwiGLU
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        cur = ggml_mul(ctx0, cur, b.ffn_norm_w);

        // Ada-scale: cur = cur * (1 + scale[il])  — precomputed as (1+scale) in ada_scales
        {
            ggml_tensor * scale = ggml_view_1d(ctx0, ada_scales, d,
                                               (size_t)il * d * sizeof(float));
            cur = ggml_mul(ctx0, cur, scale);
        }

        ggml_tensor * gate = ggml_silu(ctx0, ggml_mul_mat(ctx0, b.ffn_gate_w, cur));
        ggml_tensor * up   = ggml_mul_mat(ctx0, b.ffn_up_w, cur);
        cur = ggml_mul_mat(ctx0, b.ffn_down_w, ggml_mul(ctx0, gate, up));
        cur = ggml_add(ctx0, residual, cur);
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
    cur = ggml_mul_mat(ctx0, m.llm.token_embd_w, cur);  // tied!

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Embed graph (same as 3B)
// ===========================================================================

static ggml_cgraph * voxtral4b_build_graph_embed(voxtral4b_context * ctx, int n_tokens) {
    ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids"); ggml_set_input(ids);
    ggml_tensor * out = ggml_get_rows(ctx0, ctx->model.llm.token_embd_w, ids);
    ggml_set_name(out, "embeds");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" struct voxtral4b_context_params voxtral4b_context_default_params(void) {
    return { /*n_threads=*/4, /*verbosity=*/1 };
}

extern "C" struct voxtral4b_context * voxtral4b_init_from_file(
    const char * path, struct voxtral4b_context_params params)
{
    auto * ctx = new voxtral4b_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend)) ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!voxtral4b_load_model(ctx->model, ctx->vocab, path, ctx->backend)) {
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
    ctx->compute_meta.resize(
        ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    // Precompute ada_rms_norm scales from delay_tokens
    ctx->ada_scales = voxtral4b_compute_ada_scales(ctx, ctx->delay_tokens);
    if (params.verbosity >= 1)
        fprintf(stderr, "voxtral4b: ada_scales computed for delay=%d (%d ms)\n",
                ctx->delay_tokens, ctx->delay_tokens * 80);

    if (params.verbosity >= 1) {
        const auto & hp = ctx->model.hparams;
        fprintf(stderr,
                "voxtral4b: loaded %s  (audio %u layers, llm %u layers, vocab %u, "
                "tekken %d specials + %d BPE)\n",
                path, hp.audio_n_layers, hp.llm_n_layers, hp.llm_vocab_size,
                ctx->vocab.n_specials, ctx->vocab.n_vocab);
    }
    return ctx;
}

extern "C" void voxtral4b_free(voxtral4b_context * ctx) {
    if (!ctx) return;
    if (ctx->sched)       ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)      ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)      ggml_free(ctx->kv_ctx);
    if (ctx->model.buf)   ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)   ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)     ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" const uint8_t * voxtral4b_token_text(voxtral4b_context * ctx, int id, int * out_len) {
    if (!ctx) { if (out_len) *out_len = 0; return nullptr; }
    const auto & v = ctx->vocab;
    if (id >= 0 && id < v.n_specials) {
        if (out_len) *out_len = (int)v.specials[id].size();
        return (const uint8_t *)v.specials[id].data();
    }
    int r = id - v.n_specials;
    if (r < 0 || r >= (int)v.rank_offset.size()) {
        if (out_len) *out_len = 0;
        return nullptr;
    }
    if (out_len) *out_len = (int)v.rank_length[r];
    return v.tekken_vocab_blob.data() + v.rank_offset[r];
}

// Tekken BPE tokenizer — reuse same algorithm as 3B (see voxtral.cpp)
static void tekken4b_build_reverse(voxtral4b_vocab & v) {
    if (v.reverse_built) return;
    v.bytes_to_rank.reserve(v.rank_offset.size());
    for (size_t r = 0; r < v.rank_offset.size(); r++) {
        std::string key((const char *)v.tekken_vocab_blob.data() + v.rank_offset[r], v.rank_length[r]);
        v.bytes_to_rank[key] = (int32_t)r;
    }
    v.reverse_built = true;
}

static int32_t tekken4b_rank(const voxtral4b_vocab & v, const uint8_t * data, size_t len) {
    auto it = v.bytes_to_rank.find(std::string((const char *)data, len));
    return it != v.bytes_to_rank.end() ? it->second : -1;
}

static void tekken4b_bpe_encode(const voxtral4b_vocab & v, const uint8_t * data, size_t len,
                                std::vector<int32_t> & out) {
    if (len == 0) return;
    if (len == 1) { int32_t r = tekken4b_rank(v, data, 1); out.push_back(r >= 0 ? r + v.n_specials : 0); return; }
    struct piece { size_t start; size_t len; };
    std::vector<piece> pieces(len);
    for (size_t i = 0; i < len; i++) pieces[i] = {i, 1};
    while (pieces.size() > 1) {
        int32_t best_rank = INT32_MAX; size_t best_idx = SIZE_MAX;
        for (size_t i = 0; i + 1 < pieces.size(); i++) {
            size_t ml = pieces[i].len + pieces[i+1].len;
            int32_t r = tekken4b_rank(v, data + pieces[i].start, ml);
            if (r >= 0 && r < best_rank) { best_rank = r; best_idx = i; }
        }
        if (best_idx == SIZE_MAX) break;
        pieces[best_idx].len += pieces[best_idx+1].len;
        pieces.erase(pieces.begin() + best_idx + 1);
    }
    for (const auto & p : pieces) {
        int32_t r = tekken4b_rank(v, data + p.start, p.len);
        out.push_back(r >= 0 ? r + v.n_specials : 0);
    }
}

static std::vector<std::string> tekken4b_pre_tokenize(const std::string & text) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = text[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            size_t j = i; while (j < text.size() && (text[j]==' '||text[j]=='\t'||text[j]=='\n'||text[j]=='\r')) j++;
            out.push_back(text.substr(i, j-i)); i = j;
        } else if ((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')) {
            size_t j = i;
            while (j < text.size()) { unsigned char d = text[j]; if ((d>='A'&&d<='Z')||(d>='a'&&d<='z')||(d>='0'&&d<='9')) j++; else break; }
            out.push_back(text.substr(i, j-i)); i = j;
        } else if (c >= 0x80) {
            size_t j = i+1; while (j < text.size() && (text[j]&0xC0)==0x80) j++;
            while (j < text.size() && ((unsigned char)text[j])>=0x80) { size_t k=j+1; while(k<text.size()&&(text[k]&0xC0)==0x80) k++; j=k; }
            out.push_back(text.substr(i, j-i)); i = j;
        } else { out.push_back(text.substr(i, 1)); i++; }
    }
    return out;
}

extern "C" int32_t * voxtral4b_tokenize(voxtral4b_context * ctx, const char * text, int * out_n_tokens) {
    if (!ctx || !text) { if (out_n_tokens) *out_n_tokens = 0; return nullptr; }
    auto & v = ctx->vocab;
    tekken4b_build_reverse(v);
    std::string input(text);
    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < input.size()) {
        bool found_special = false;
        for (int si = 0; si < (int)v.specials.size(); si++) {
            const auto & sp = v.specials[si];
            if (sp.empty()) continue;
            if (pos + sp.size() <= input.size() && input.compare(pos, sp.size(), sp) == 0) {
                ids.push_back(si); pos += sp.size(); found_special = true; break;
            }
        }
        if (found_special) continue;
        size_t next_special = input.size();
        for (int si = 0; si < (int)v.specials.size(); si++) {
            const auto & sp = v.specials[si]; if (sp.empty()) continue;
            size_t f = input.find(sp, pos);
            if (f != std::string::npos && f < next_special) next_special = f;
        }
        auto pre_tokens = tekken4b_pre_tokenize(input.substr(pos, next_special - pos));
        for (const auto & pt : pre_tokens)
            tekken4b_bpe_encode(v, (const uint8_t *)pt.data(), pt.size(), ids);
        pos = next_special;
    }
    if (ids.empty()) { if (out_n_tokens) *out_n_tokens = 0; return nullptr; }
    int32_t * result = (int32_t *)malloc(ids.size() * sizeof(int32_t));
    std::memcpy(result, ids.data(), ids.size() * sizeof(int32_t));
    if (out_n_tokens) *out_n_tokens = (int)ids.size();
    return result;
}

// Encoder run
extern "C" float * voxtral4b_run_encoder(voxtral4b_context * ctx,
                                         const float * mel, int n_mels, int T_mel,
                                         int * out_N, int * out_dim) {
    if (!ctx || !mel || n_mels != 128 || T_mel <= 0 || T_mel % 2 != 0) return nullptr;
    const int T_enc = (T_mel + 1) / 2;
    // Truncate to be divisible by 4 (downsample factor)
    const int T_enc_ds = (T_enc / 4) * 4;  // round down
    const int N_out = T_enc_ds / 4;
    const int dim   = (int)ctx->model.hparams.proj_out_dim;

    ggml_cgraph * gf = voxtral4b_build_graph_encoder(ctx, T_mel);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return nullptr;

    // Set mel input
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel"), mel, 0,
                            (size_t)n_mels * T_mel * sizeof(float));

    // Set causal conv padding to zeros
    {
        ggml_tensor * p0 = ggml_graph_get_tensor(gf, "conv0_lpad");
        if (p0) { std::vector<float> zeros(2 * n_mels, 0.0f); ggml_backend_tensor_set(p0, zeros.data(), 0, zeros.size() * sizeof(float)); }
        ggml_tensor * p1 = ggml_graph_get_tensor(gf, "conv1_lpad");
        if (p1) { int d = (int)ctx->model.hparams.audio_d_model; std::vector<float> zeros(d, 0.0f); ggml_backend_tensor_set(p1, zeros.data(), 0, zeros.size() * sizeof(float)); }
    }

    // Set encoder positions [0, 1, ..., T_enc-1]
    std::vector<int32_t> pos(T_enc);
    for (int i = 0; i < T_enc; i++) pos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_positions"), pos.data(), 0,
                            pos.size() * sizeof(int32_t));

    // Set SWA mask if present
    ggml_tensor * swa_t = ggml_graph_get_tensor(gf, "swa_mask");
    if (swa_t) {
        int swa = (int)ctx->model.hparams.audio_swa;
        std::vector<ggml_fp16_t> mask((size_t)T_enc * T_enc, ggml_fp32_to_fp16(0.0f));
        ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        // Causal + sliding window: mask k > q (causal) and k < q - swa (outside window)
        for (int q = 0; q < T_enc; q++)
            for (int k = 0; k < T_enc; k++)
                if (k > q || k < q - swa) mask[(size_t)q * T_enc + k] = neg_inf;
        ggml_backend_tensor_set(swa_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return nullptr;

    ggml_tensor * out = ggml_graph_get_tensor(gf, "encoder_out");
    size_t total = (size_t)N_out * dim;
    float * result = (float *)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    if (out_N) *out_N = N_out;
    if (out_dim) *out_dim = dim;
    return result;
}

// Embed tokens
extern "C" float * voxtral4b_embed_tokens(voxtral4b_context * ctx,
                                          const int32_t * input_ids, int n_tokens) {
    if (!ctx || !input_ids || n_tokens <= 0) return nullptr;
    const int d = (int)ctx->model.hparams.llm_d_model;
    ggml_cgraph * gf = voxtral4b_build_graph_embed(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input_ids"), input_ids, 0,
                            (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return nullptr;
    ggml_tensor * out = ggml_graph_get_tensor(gf, "embeds");
    float * result = (float *)malloc((size_t)n_tokens * d * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)n_tokens * d * sizeof(float));
    return result;
}

// KV cache
extern "C" bool voxtral4b_kv_init(voxtral4b_context * ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0) return false;
    const auto & hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int nl = (int)hp.llm_n_layers;

    size_t k_size = (size_t)ggml_type_size(GGML_TYPE_F16) * hd * max_ctx * n_kv * nl;
    size_t v_size = k_size;

    ggml_init_params ip = { 2 * ggml_tensor_overhead(), nullptr, true };
    ctx->kv_ctx = ggml_init(ip);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);

    ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, k_size + v_size);
    char * base = (char *)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + ggml_nbytes(ctx->kv_k));

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "voxtral4b: kv cache %.0f MiB (head_dim=%d max_ctx=%d n_kv=%d n_layers=%d)\n",
                (k_size + v_size) / 1048576.0, hd, max_ctx, n_kv, nl);
    return true;
}

extern "C" void voxtral4b_kv_reset(voxtral4b_context * ctx) {
    if (ctx && ctx->kv_buf) ggml_backend_buffer_clear(ctx->kv_buf, 0);
}

// LLM forward with KV cache
extern "C" float * voxtral4b_run_llm_kv(voxtral4b_context * ctx,
                                        const float * inputs_embeds,
                                        int n_tokens, int n_past,
                                        int * out_n_tokens, int * out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0) return nullptr;
    const auto & hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int vocab = (int)hp.llm_vocab_size;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) positions[i] = n_past + i;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        int Lk = n_past + n_tokens;
        mask.resize((size_t)n_tokens * Lk, ggml_fp32_to_fp16(0.0f));
        ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++)
            for (int k = 0; k < Lk; k++)
                if (k > n_past + q) mask[(size_t)q * Lk + k] = neg_inf;
    }

    ggml_cgraph * gf = voxtral4b_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return nullptr;

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

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return nullptr;

    ggml_tensor * logits_t = ggml_graph_get_tensor(gf, "logits");
    float * result = (float *)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(logits_t, result, 0, (size_t)vocab * sizeof(float));
    if (out_n_tokens) *out_n_tokens = 1;
    if (out_vocab_size) *out_vocab_size = vocab;
    return result;
}
