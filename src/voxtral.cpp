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
// Hyper-parameters (filled from voxtral.* GGUF kv)
// ===========================================================================

struct voxtral_hparams {
    // Audio encoder
    uint32_t sample_rate    = 16000;
    uint32_t n_mels         = 128;
    uint32_t n_fft          = 400;
    uint32_t win_length     = 400;
    uint32_t hop_length     = 160;
    uint32_t audio_n_layers = 32;
    uint32_t audio_d_model  = 1280;
    uint32_t audio_n_heads  = 20;
    uint32_t audio_head_dim = 64;
    uint32_t audio_ff_dim   = 5120;
    uint32_t audio_max_pos  = 1500;

    // Projector
    uint32_t proj_in_dim      = 5120;
    uint32_t proj_out_dim     = 3072;
    uint32_t proj_frame_stack = 4;

    // LLM (Llama 3 / Mistral)
    uint32_t llm_n_layers   = 30;
    uint32_t llm_d_model    = 3072;
    uint32_t llm_n_heads    = 32;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim   = 128;
    uint32_t llm_ff_dim     = 8192;
    float    llm_rope_theta = 1e8f;
    float    llm_rms_eps    = 1e-5f;
    uint32_t llm_vocab_size = 131072;
    uint32_t llm_max_pos    = 131072;

    uint32_t audio_token_id = 24;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct voxtral_audio_block {
    // Pre-LN self-attention (Whisper-style: biased q/v/out, no bias on K)
    ggml_tensor * attn_norm_w  = nullptr, * attn_norm_b  = nullptr;
    ggml_tensor * attn_q_w     = nullptr, * attn_q_b     = nullptr;
    ggml_tensor * attn_k_w     = nullptr;  // NO bias (Whisper quirk)
    ggml_tensor * attn_v_w     = nullptr, * attn_v_b     = nullptr;
    ggml_tensor * attn_out_w   = nullptr, * attn_out_b   = nullptr;
    // Pre-LN FFN (GELU)
    ggml_tensor * ffn_norm_w   = nullptr, * ffn_norm_b   = nullptr;
    ggml_tensor * ffn_up_w     = nullptr, * ffn_up_b     = nullptr;  // fc1
    ggml_tensor * ffn_down_w   = nullptr, * ffn_down_b   = nullptr;  // fc2
};

struct voxtral_audio_tower {
    ggml_tensor * conv1_w = nullptr, * conv1_b = nullptr;
    ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;
    ggml_tensor * embed_positions = nullptr;  // (max_pos, d_model) F32
    std::vector<voxtral_audio_block> blocks;
    ggml_tensor * ln_post_w = nullptr, * ln_post_b = nullptr;
    // Mel preprocessor (baked from WhisperFeatureExtractor)
    ggml_tensor * mel_filters = nullptr;  // (n_freqs=201, n_mels=128) F32
    ggml_tensor * mel_window  = nullptr;  // (400,) F32 hann window
};

struct voxtral_projector {
    ggml_tensor * proj1 = nullptr;  // (in_dim=5120, out_dim=3072)
    ggml_tensor * proj2 = nullptr;  // (out_dim=3072, out_dim=3072)
};

struct voxtral_llm_block {
    ggml_tensor * attn_norm_w   = nullptr;
    ggml_tensor * attn_q_w      = nullptr;
    ggml_tensor * attn_k_w      = nullptr;
    ggml_tensor * attn_v_w      = nullptr;
    ggml_tensor * attn_output_w = nullptr;
    ggml_tensor * ffn_norm_w    = nullptr;
    ggml_tensor * ffn_gate_w    = nullptr;
    ggml_tensor * ffn_up_w      = nullptr;
    ggml_tensor * ffn_down_w    = nullptr;
};

struct voxtral_llm {
    ggml_tensor * token_embd_w = nullptr;
    std::vector<voxtral_llm_block> blocks;
    ggml_tensor * output_norm_w = nullptr;
    ggml_tensor * output_w      = nullptr;
};

struct voxtral_model {
    voxtral_hparams     hparams;
    voxtral_audio_tower audio;
    voxtral_projector   projector;
    voxtral_llm         llm;

    ggml_context        * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
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
    int n_vocab    = 0;

    std::string pre_pattern;  // tiktoken-style pre-tokenizer regex
};

struct voxtral_context {
    voxtral_context_params params;

    voxtral_model model;
    voxtral_vocab vocab;

    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    std::vector<uint8_t> compute_meta;

    // KV cache (F16, same pattern as qwen3_asr)
    ggml_context        * kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor         * kv_k   = nullptr;
    ggml_tensor         * kv_v   = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used  = 0;

    int n_threads = 4;
};

// ===========================================================================
// Loader helpers
// ===========================================================================

static ggml_tensor * try_get(voxtral_model & m, const char * name) {
    auto it = m.tensors.find(name);
    return it != m.tensors.end() ? it->second : nullptr;
}

static ggml_tensor * require(voxtral_model & m, const char * name) {
    auto t = try_get(m, name);
    if (!t) {
        fprintf(stderr, "voxtral: required tensor '%s' not found in GGUF\n", name);
    }
    return t;
}

static uint32_t kv_u32(gguf_context * gctx, const char * key, uint32_t def = 0) {
    int ki = gguf_find_key(gctx, key);
    return ki >= 0 ? (uint32_t)gguf_get_val_u32(gctx, ki) : def;
}

static float kv_f32(gguf_context * gctx, const char * key, float def = 0.0f) {
    int ki = gguf_find_key(gctx, key);
    return ki >= 0 ? gguf_get_val_f32(gctx, ki) : def;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool voxtral_load_model(voxtral_model & model,
                               voxtral_vocab & vocab,
                               const char * path,
                               ggml_backend_t backend) {
    // ---- pass 1: read hparams + vocab via metadata-only context ----
    {
        ggml_init_params meta_params = {
            /*mem_size=*/   4 * 1024 * 1024,
            /*mem_buffer=*/ nullptr,
            /*no_alloc=*/   true,
        };
        ggml_context * meta_ctx = ggml_init(meta_params);
        gguf_init_params load_params_meta = { /*no_alloc=*/true, /*ctx=*/&meta_ctx };
        gguf_context * gctx = gguf_init_from_file(path, load_params_meta);
        if (!gctx) {
            fprintf(stderr, "voxtral: failed to open '%s'\n", path);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }

        auto & hp = model.hparams;
        hp.sample_rate    = kv_u32(gctx, "voxtral.sample_rate",    hp.sample_rate);
        hp.n_mels         = kv_u32(gctx, "voxtral.n_mels",         hp.n_mels);
        hp.n_fft          = kv_u32(gctx, "voxtral.n_fft",          hp.n_fft);
        hp.win_length     = kv_u32(gctx, "voxtral.win_length",     hp.win_length);
        hp.hop_length     = kv_u32(gctx, "voxtral.hop_length",     hp.hop_length);
        hp.audio_n_layers = kv_u32(gctx, "voxtral.audio.n_layers", hp.audio_n_layers);
        hp.audio_d_model  = kv_u32(gctx, "voxtral.audio.d_model",  hp.audio_d_model);
        hp.audio_n_heads  = kv_u32(gctx, "voxtral.audio.n_heads",  hp.audio_n_heads);
        hp.audio_head_dim = kv_u32(gctx, "voxtral.audio.head_dim", hp.audio_head_dim);
        hp.audio_ff_dim   = kv_u32(gctx, "voxtral.audio.ff_dim",   hp.audio_ff_dim);
        hp.audio_max_pos  = kv_u32(gctx, "voxtral.audio.max_pos",  hp.audio_max_pos);

        hp.proj_in_dim      = kv_u32(gctx, "voxtral.proj.in_dim",      hp.proj_in_dim);
        hp.proj_out_dim     = kv_u32(gctx, "voxtral.proj.out_dim",     hp.proj_out_dim);
        hp.proj_frame_stack = kv_u32(gctx, "voxtral.proj.frame_stack", hp.proj_frame_stack);

        hp.llm_n_layers   = kv_u32(gctx, "voxtral.llm.n_layers",   hp.llm_n_layers);
        hp.llm_d_model    = kv_u32(gctx, "voxtral.llm.d_model",    hp.llm_d_model);
        hp.llm_n_heads    = kv_u32(gctx, "voxtral.llm.n_heads",    hp.llm_n_heads);
        hp.llm_n_kv_heads = kv_u32(gctx, "voxtral.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim   = kv_u32(gctx, "voxtral.llm.head_dim",   hp.llm_head_dim);
        hp.llm_ff_dim     = kv_u32(gctx, "voxtral.llm.ff_dim",     hp.llm_ff_dim);
        hp.llm_rope_theta = kv_f32(gctx, "voxtral.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps    = kv_f32(gctx, "voxtral.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = kv_u32(gctx, "voxtral.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos    = kv_u32(gctx, "voxtral.llm.max_pos",    hp.llm_max_pos);
        hp.audio_token_id = kv_u32(gctx, "voxtral.audio_token_id", hp.audio_token_id);

        // ---- Tekken tokenizer blobs ----
        // pattern (string)
        int kp = gguf_find_key(gctx, "tokenizer.tekken.pattern");
        if (kp >= 0) vocab.pre_pattern = gguf_get_val_str(gctx, kp);

        // specials (string array)
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

        // The vocab blob is stored as a 1D F32 tensor (one float per byte)
        // because the GGUF KV array path loses uint8 precision. We'll read
        // it from the tensor data in pass 2 after the weights are loaded.

        gguf_free(gctx);
        ggml_free(meta_ctx);
    }

    // ---- pass 2: load tensor metadata + bind into a backend buffer ----
    ggml_context * weight_ctx = nullptr;
    {
        gguf_init_params load_params = { /*no_alloc=*/true, /*ctx=*/&weight_ctx };
        gguf_context * gctx = gguf_init_from_file(path, load_params);
        if (!gctx || !weight_ctx) {
            fprintf(stderr, "voxtral: failed to load tensor metadata\n");
            return false;
        }

        model.buf = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);

        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "voxtral: open failed\n"); return false; }
        struct stat st; fstat(fd, &st);
        size_t file_size = (size_t)st.st_size;
        void * mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (mmap_base == MAP_FAILED) {
            fprintf(stderr, "voxtral: mmap failed\n");
            return false;
        }

        size_t data_offset = gguf_get_data_offset(gctx);

        for (ggml_tensor * t = ggml_get_first_tensor(weight_ctx); t;
             t = ggml_get_next_tensor(weight_ctx, t)) {
            model.tensors[ggml_get_name(t)] = t;
            int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0) continue;
            size_t off    = gguf_get_tensor_offset(gctx, tid);
            size_t nbytes = ggml_nbytes(t);
            ggml_backend_tensor_set(t, (const char *)mmap_base + data_offset + off, 0, nbytes);
        }

        munmap(mmap_base, file_size);
        model.ctx = weight_ctx;
        gguf_free(gctx);
    }

    // ---- bind named tensors into the per-layer structs ----
    auto & a = model.audio;
    a.conv1_w = require(model, "audio.conv.1.weight");
    a.conv1_b = require(model, "audio.conv.1.bias");
    a.conv2_w = require(model, "audio.conv.2.weight");
    a.conv2_b = require(model, "audio.conv.2.bias");
    a.embed_positions = require(model, "audio.embed_positions");
    a.ln_post_w = require(model, "audio.ln_post.weight");
    a.ln_post_b = require(model, "audio.ln_post.bias");
    a.mel_filters = try_get(model, "audio.mel_filters");
    a.mel_window  = try_get(model, "audio.mel_window");
    a.blocks.resize(model.hparams.audio_n_layers);
    for (uint32_t i = 0; i < model.hparams.audio_n_layers; i++) {
        char buf[128];
        auto & b = a.blocks[i];
        auto get = [&](const char * suf) {
            snprintf(buf, sizeof(buf), "audio.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_norm_b = get("attn_norm.bias");
        b.attn_q_w    = get("attn_q.weight");
        b.attn_q_b    = get("attn_q.bias");
        b.attn_k_w    = get("attn_k.weight");
        // K has no bias (Whisper quirk) — skip
        b.attn_v_w    = get("attn_v.weight");
        b.attn_v_b    = get("attn_v.bias");
        b.attn_out_w  = get("attn_out.weight");
        b.attn_out_b  = get("attn_out.bias");
        b.ffn_norm_w  = get("ffn_norm.weight");
        b.ffn_norm_b  = get("ffn_norm.bias");
        b.ffn_up_w    = get("ffn_up.weight");
        b.ffn_up_b    = get("ffn_up.bias");
        b.ffn_down_w  = get("ffn_down.weight");
        b.ffn_down_b  = get("ffn_down.bias");
    }

    // ---- Reconstruct the Tekken vocab blob from the F32 tensor ----
    {
        ggml_tensor * vt = try_get(model, "tokenizer.tekken.vocab_tensor");
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
    p.proj1 = require(model, "proj1.weight");
    p.proj2 = require(model, "proj2.weight");

    auto & l = model.llm;
    l.token_embd_w  = require(model, "token_embd.weight");
    l.output_norm_w = require(model, "output_norm.weight");
    l.output_w      = require(model, "output.weight");
    l.blocks.resize(model.hparams.llm_n_layers);
    for (uint32_t i = 0; i < model.hparams.llm_n_layers; i++) {
        char buf[128];
        auto & b = l.blocks[i];
        auto get = [&](const char * suf) {
            snprintf(buf, sizeof(buf), "blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w   = get("attn_norm.weight");
        b.attn_q_w      = get("attn_q.weight");
        b.attn_k_w      = get("attn_k.weight");
        b.attn_v_w      = get("attn_v.weight");
        b.attn_output_w = get("attn_output.weight");
        // NO Q-norm/K-norm for Llama / Mistral
        b.ffn_norm_w    = get("ffn_norm.weight");
        b.ffn_gate_w    = get("ffn_gate.weight");
        b.ffn_up_w      = get("ffn_up.weight");
        b.ffn_down_w    = get("ffn_down.weight");
    }

    return true;
}

// ===========================================================================
// FFT + Mel computation (cargo-cult from qwen3_asr.cpp — same parameters)
// ===========================================================================

static void voxtral_dft(const float * in, int N, float * out) {
    for (int k = 0; k < N; k++) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; n++) {
            float ang = -2.0f * (float)M_PI * (float)k * (float)n / (float)N;
            re += in[n] * std::cos(ang); im += in[n] * std::sin(ang);
        }
        out[2*k] = re; out[2*k+1] = im;
    }
}

static void voxtral_fft(float * in, int N, float * out) {
    if (N == 1) { out[0] = in[0]; out[1] = 0.0f; return; }
    int half = N / 2;
    if (N - half*2 == 1) { voxtral_dft(in, N, out); return; }
    float * even = in + N;
    for (int i = 0; i < half; i++) even[i] = in[2*i];
    float * ef = out + 2*N; voxtral_fft(even, half, ef);
    float * odd = even;
    for (int i = 0; i < half; i++) odd[i] = in[2*i+1];
    float * of = ef + N; voxtral_fft(odd, half, of);
    for (int k = 0; k < half; k++) {
        float ang = -2.0f*(float)M_PI*(float)k/(float)N;
        float re = std::cos(ang), im = std::sin(ang);
        float reo = of[2*k], imo = of[2*k+1];
        out[2*k]          = ef[2*k]   + re*reo - im*imo;
        out[2*k+1]        = ef[2*k+1] + re*imo + im*reo;
        out[2*(k+half)]   = ef[2*k]   - re*reo + im*imo;
        out[2*(k+half)+1] = ef[2*k+1] - re*imo - im*reo;
    }
}

extern "C" float * voxtral_compute_mel(voxtral_context * ctx,
                                       const float * samples, int n_samples,
                                       int * out_n_mels, int * out_T_mel) {
    if (!ctx || !samples || n_samples <= 0) return nullptr;
    if (!ctx->model.audio.mel_filters || !ctx->model.audio.mel_window) {
        fprintf(stderr, "voxtral: GGUF missing audio.mel_filters/mel_window — re-convert\n");
        return nullptr;
    }
    const int n_fft = 400, hop = 160, n_mels = 128, n_freqs = 201;
    const int T_out = 3000;  // Voxtral always pads to 30s

    std::vector<float> hann(n_fft);
    ggml_backend_tensor_get(ctx->model.audio.mel_window, hann.data(), 0, n_fft*sizeof(float));
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->model.audio.mel_filters, filt.data(), 0, filt.size()*sizeof(float));

    const int pad = n_fft / 2;
    std::vector<float> padded((size_t)n_samples + 2*pad, 0.0f);
    std::memcpy(padded.data() + pad, samples, n_samples*sizeof(float));

    const int T_full = (int)((padded.size() - n_fft) / hop + 1);
    const int T = T_full - 1;  // Whisper drops last frame

    std::vector<float> power((size_t)n_freqs * T, 0.0f);
    {
        std::vector<float> fi((size_t)n_fft*4, 0.0f), fo((size_t)n_fft*8, 0.0f);
        for (int t = 0; t < T; t++) {
            const float * frame = padded.data() + (size_t)t*hop;
            for (int n = 0; n < n_fft; n++) fi[n] = frame[n]*hann[n];
            voxtral_fft(fi.data(), n_fft, fo.data());
            for (int k = 0; k < n_freqs; k++) {
                float re = fo[2*k], im = fo[2*k+1];
                power[(size_t)k*T+t] = re*re + im*im;
            }
        }
    }

    // mel + log10 + clip + normalize — pad to T_out=3000
    std::vector<float> mel((size_t)n_mels * T_out, 0.0f);
    float mel_max = -1e30f;
    for (int m = 0; m < n_mels; m++) {
        for (int t = 0; t < std::min(T, T_out); t++) {
            double s = 0.0;
            for (int k = 0; k < n_freqs; k++)
                s += (double)filt[(size_t)k*n_mels+m] * power[(size_t)k*T+t];
            float lv = std::log10(std::max((float)s, 1e-10f));
            mel[(size_t)m*T_out+t] = lv;
            if (lv > mel_max) mel_max = lv;
        }
        // T..T_out remain 0 → log10(1e-10) = -10 after clipping
        for (int t = T; t < T_out; t++) {
            float lv = std::log10(1e-10f);
            mel[(size_t)m*T_out+t] = lv;
            if (lv > mel_max) mel_max = lv;
        }
    }
    const float floor_v = mel_max - 8.0f;
    for (size_t i = 0; i < mel.size(); i++) {
        float v = mel[i]; if (v < floor_v) v = floor_v;
        mel[i] = (v + 4.0f) / 4.0f;
    }

    if (out_n_mels) *out_n_mels = n_mels;
    if (out_T_mel)  *out_T_mel  = T_out;
    float * result = (float*)malloc(mel.size()*sizeof(float));
    std::memcpy(result, mel.data(), mel.size()*sizeof(float));
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

static ggml_cgraph * voxtral_build_graph_encoder(voxtral_context * ctx) {
    const auto & m  = ctx->model;
    const auto & hp = m.hparams;
    const int d         = (int)hp.audio_d_model;   // 1280
    const int n_heads   = (int)hp.audio_n_heads;   // 20
    const int head_dim  = (int)hp.audio_head_dim;  // 64
    const int n_layers  = (int)hp.audio_n_layers;  // 32
    const int proj_in   = (int)hp.proj_in_dim;     // 5120
    const int n_mels    = (int)hp.n_mels;          // 128
    const int T_mel     = 3000;                    // padded to 30s
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    ggml_init_params ip = {
        /*mem_size=*/   ctx->compute_meta.size(),
        /*mem_buffer=*/ ctx->compute_meta.data(),
        /*no_alloc=*/   true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: mel spectrogram (n_mels=128, T_mel=3000) F32
    ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, n_mels);
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
    auto bias_1d = [&](ggml_tensor * b) {
        return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1);
    };

    ggml_tensor * cur = ggml_conv_1d(ctx0, m.audio.conv1_w, mel, /*s*/1, /*p*/1, /*d*/1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);
    // cur ne = (3000, 1280, 1)

    cur = ggml_conv_1d(ctx0, m.audio.conv2_w, cur, /*s*/2, /*p*/1, /*d*/1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);
    // cur ne = (1500, 1280, 1)

    // ggml_conv_1d output is (OL, OC, 1) = (1500, 1280, 1). We need (d, T_enc)
    // = (1280, 1500) for the norm/mul/matmul ops downstream, which all expect
    // ne[0] = feature_dim. Transpose:
    const int T_enc = T_mel / 2;
    cur = ggml_reshape_2d(ctx0, cur, T_enc, d);         // (1500, 1280) squeeze batch
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));    // (1280, 1500) = (d, T_enc)

    // ---- Add learned positional embedding ----
    // embed_positions ggml ne = (d=1280, max_pos=1500) = (d, T_enc). Matches cur.
    cur = ggml_add(ctx0, cur, m.audio.embed_positions);

    // ---- 32 × Whisper encoder block ----
    for (int il = 0; il < n_layers; il++) {
        const auto & b = m.audio.blocks[il];
        ggml_tensor * residual = cur;

        // Pre-LN
        ggml_tensor * x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        x = ggml_add(ctx0, x, b.attn_norm_b);

        // Self-attention (biased Q, V, out_proj; NO bias on K — Whisper quirk)
        ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, x), b.attn_q_b);
        ggml_tensor * K = ggml_mul_mat(ctx0, b.attn_k_w, x);  // no bias
        ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, x), b.attn_v_b);

        // Reshape to (head_dim, n_heads, T_enc)
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, T_enc);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_heads, T_enc);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T_enc);

        // Permute to (hd, T, n_h) for attention
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        // No causal mask (encoder self-attention is bidirectional)
        ggml_tensor * attn = ggml_flash_attn_ext(ctx0, Q, K, V, /*mask*/nullptr,
                                                  attn_scale, 0.0f, 0.0f);
        // attn ne = (hd, n_h, T, 1) → reshape to (d, T)
        attn = ggml_reshape_2d(ctx0, attn, d, T_enc);

        attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_out_w, attn), b.attn_out_b);
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

static ggml_cgraph * voxtral_build_graph_llm(voxtral_context * ctx, int n_tokens) {
    const auto & m  = ctx->model;
    const auto & hp = m.hparams;
    const int d        = (int)hp.llm_d_model;     // 3072
    const int n_q      = (int)hp.llm_n_heads;     // 32
    const int n_kv     = (int)hp.llm_n_kv_heads;  // 8
    const int hd       = (int)hp.llm_head_dim;    // 128
    const int n_kv_grp = n_q / n_kv;              // 4
    const float eps    = hp.llm_rms_eps;
    const float theta  = hp.llm_rope_theta;
    const int T        = n_tokens;
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = {
        /*mem_size=*/   ctx->compute_meta.size(),
        /*mem_buffer=*/ ctx->compute_meta.data(),
        /*no_alloc=*/   true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    // ------- Inputs -------
    ggml_tensor * input_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(input_ids, "input_ids");
    ggml_set_input(input_ids);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor * causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    // ------- Token embedding lookup -------
    ggml_tensor * cur = ggml_get_rows(ctx0, m.llm.token_embd_w, input_ids);
    // cur ne = (d, T)

    // ------- 30 × Llama block -------
    for (uint32_t il = 0; il < hp.llm_n_layers; il++) {
        const auto & b = m.llm.blocks[il];
        ggml_tensor * residual = cur;

        // ---- LN1 (RMSNorm + multiplicative weight, no bias) ----
        ggml_tensor * x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // ---- Q, K, V projections (no biases) ----
        ggml_tensor * Q = ggml_mul_mat(ctx0, b.attn_q_w, x);
        ggml_tensor * K = ggml_mul_mat(ctx0, b.attn_k_w, x);
        ggml_tensor * V = ggml_mul_mat(ctx0, b.attn_v_w, x);
        Q = ggml_reshape_3d(ctx0, Q, hd, n_q,  T);
        K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

        // (No Q-norm/K-norm — that's the Qwen3-specific bit we drop)

        // ---- RoPE (NEOX style: rotate first/second half along head_dim) ----
        Q = ggml_rope_ext(ctx0, Q, positions, /*freq_factors*/nullptr,
                          hd, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig*/(int)hp.llm_max_pos,
                          theta, /*freq_scale*/1.0f, /*ext_factor*/0.0f,
                          /*attn_factor*/1.0f, /*beta_fast*/32.0f, /*beta_slow*/1.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr,
                          hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos,
                          theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // ---- GQA: repeat each KV head n_kv_grp=4 times to match n_q=32 ----
        if (n_kv_grp > 1) {
            ggml_tensor * K4 = ggml_reshape_4d(ctx0, K, hd, 1, n_kv, T);
            ggml_tensor * V4 = ggml_reshape_4d(ctx0, V, hd, 1, n_kv, T);
            K4 = ggml_repeat_4d(ctx0, K4, hd, n_kv_grp, n_kv, T);
            V4 = ggml_repeat_4d(ctx0, V4, hd, n_kv_grp, n_kv, T);
            K = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, n_q, T));
            V = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, n_q, T));
        }

        // ---- Permute Q/K/V to (head_dim, T, n_heads) for flash-attn ----
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        // ---- Attention via ggml_flash_attn_ext (F16 mask) ----
        // Output ne = (hd, n_q, T, 1) → reshape (hd*n_q, T) = (4096, T)
        ggml_tensor * attn = ggml_flash_attn_ext(ctx0, Q, K, V, causal_mask,
                                                  attn_scale, /*max_bias*/0.0f,
                                                  /*logit_softcap*/0.0f);
        attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

        // ---- O projection (no bias) — projects (4096, T) → (3072, T) ----
        attn = ggml_mul_mat(ctx0, b.attn_output_w, attn);
        cur = ggml_add(ctx0, residual, attn);

        // ---- FFN: down(silu(gate(x)) * up(x)) ----
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor * gate = ggml_mul_mat(ctx0, b.ffn_gate_w, x);
        ggml_tensor * up   = ggml_mul_mat(ctx0, b.ffn_up_w,   x);
        ggml_tensor * mlp  = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
        mlp = ggml_mul_mat(ctx0, b.ffn_down_w, mlp);
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
    return p;
}

extern "C" voxtral_context * voxtral_init_from_file(const char * path,
                                                    voxtral_context_params params) {
    voxtral_context * ctx = new voxtral_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Try GPU backend first (Metal, CUDA, Vulkan...), fall back to CPU.
    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
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
    if (params.verbosity >= 1) {
        fprintf(stderr,
                "voxtral: loaded %s  (audio %u layers, llm %u layers, vocab %u, "
                "tekken %d specials + %d BPE)\n",
                path, ctx->model.hparams.audio_n_layers,
                ctx->model.hparams.llm_n_layers,
                ctx->model.hparams.llm_vocab_size,
                ctx->vocab.n_specials, ctx->vocab.n_vocab);
    }
    return ctx;
}

extern "C" void voxtral_free(voxtral_context * ctx) {
    if (!ctx) return;
    if (ctx->sched)       ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)      ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)      ggml_free(ctx->kv_ctx);
    if (ctx->model.buf)   ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)   ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" const uint8_t * voxtral_token_text(voxtral_context * ctx, int id, int * out_len) {
    if (!ctx) { if (out_len) *out_len = 0; return nullptr; }
    const auto & v = ctx->vocab;
    // Special tokens live at ranks [0, n_specials).
    if (id >= 0 && id < v.n_specials) {
        if (out_len) *out_len = (int)v.specials[id].size();
        return (const uint8_t *)v.specials[id].data();
    }
    // Regular vocab entries are stored at indices [n_specials, n_specials+n_vocab)
    // in the model's logical id space, but the rank_offset table uses a 0-based
    // index into the vocab blob. So translate: rank = id - n_specials.
    int r = id - v.n_specials;
    if (r < 0 || r >= (int)v.rank_offset.size()) {
        if (out_len) *out_len = 0;
        return nullptr;
    }
    if (out_len) *out_len = (int)v.rank_length[r];
    return v.tekken_vocab_blob.data() + v.rank_offset[r];
}

// ===========================================================================
// KV-cached LLM graph (Stage V3) — same pattern as qwen3_asr's build_graph_llm_kv
// ===========================================================================

static ggml_cgraph * voxtral_build_graph_llm_kv(voxtral_context * ctx,
                                                int n_past, int n_tokens) {
    const auto & m  = ctx->model;
    const auto & hp = m.hparams;
    const int d        = (int)hp.llm_d_model;
    const int n_q      = (int)hp.llm_n_heads;
    const int n_kv     = (int)hp.llm_n_kv_heads;
    const int hd       = (int)hp.llm_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps    = hp.llm_rms_eps;
    const float theta  = hp.llm_rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T  = n_tokens;
    const int Lk = n_past + T;

    GGML_ASSERT(ctx->kv_k && ctx->kv_v && Lk <= ctx->kv_max_ctx);

    ggml_init_params ip = {
        /*mem_size=*/ ctx->compute_meta.size(),
        /*mem_buffer=*/ ctx->compute_meta.data(),
        /*no_alloc=*/ true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor * embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds"); ggml_set_input(embeds);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions"); ggml_set_input(positions);
    ggml_tensor * causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask"); ggml_set_input(causal_mask);
    }

    ggml_tensor * cur = embeds;

    for (uint32_t il = 0; il < hp.llm_n_layers; il++) {
        const auto & b = m.llm.blocks[il];
        ggml_tensor * residual = cur;

        ggml_tensor * x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor * Q = ggml_mul_mat(ctx0, b.attn_q_w, x);
        ggml_tensor * K = ggml_mul_mat(ctx0, b.attn_k_w, x);
        ggml_tensor * V = ggml_mul_mat(ctx0, b.attn_v_w, x);
        Q = ggml_reshape_3d(ctx0, Q, hd, n_q,  T);
        K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

        Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX,
                          (int)hp.llm_max_pos, theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX,
                          (int)hp.llm_max_pos, theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // Write new K/V into the persistent cache
        ggml_tensor * K_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
        ggml_tensor * V_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);
        ggml_tensor * k_view = ggml_view_4d(ctx0, ctx->kv_k, hd, T, n_kv, 1,
            ctx->kv_k->nb[1], ctx->kv_k->nb[2], ctx->kv_k->nb[3],
            il * ctx->kv_k->nb[3] + n_past * ctx->kv_k->nb[1]);
        ggml_tensor * v_view = ggml_view_4d(ctx0, ctx->kv_v, hd, T, n_kv, 1,
            ctx->kv_v->nb[1], ctx->kv_v->nb[2], ctx->kv_v->nb[3],
            il * ctx->kv_v->nb[3] + n_past * ctx->kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_perm, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_perm, v_view));

        // Read full history
        ggml_tensor * Kfull = ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_k,
            hd, Lk, n_kv, ctx->kv_k->nb[1], ctx->kv_k->nb[2], il * ctx->kv_k->nb[3]));
        ggml_tensor * Vfull = ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_v,
            hd, Lk, n_kv, ctx->kv_v->nb[1], ctx->kv_v->nb[2], il * ctx->kv_v->nb[3]));

        // GQA expand
        if (n_kv_grp > 1) {
            ggml_tensor * K4 = ggml_reshape_4d(ctx0, Kfull, hd, Lk, 1, n_kv);
            ggml_tensor * V4 = ggml_reshape_4d(ctx0, Vfull, hd, Lk, 1, n_kv);
            K4 = ggml_repeat_4d(ctx0, K4, hd, Lk, n_kv_grp, n_kv);
            V4 = ggml_repeat_4d(ctx0, V4, hd, Lk, n_kv_grp, n_kv);
            Kfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, Lk, n_q));
            Vfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, Lk, n_q));
        }

        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

        ggml_tensor * attn;
        if (T == 1) {
            attn = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, nullptr,
                                       attn_scale, 0.0f, 0.0f);
            attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);
        } else {
            attn = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, causal_mask,
                                       attn_scale, 0.0f, 0.0f);
            attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);
        }

        attn = ggml_mul_mat(ctx0, b.attn_output_w, attn);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor * gate = ggml_mul_mat(ctx0, b.ffn_gate_w, x);
        ggml_tensor * up   = ggml_mul_mat(ctx0, b.ffn_up_w, x);
        ggml_tensor * mlp  = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
        mlp = ggml_mul_mat(ctx0, b.ffn_down_w, mlp);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);
    if (T > 1) cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T-1)*cur->nb[1]);
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Tiny embed lookup graph
static ggml_cgraph * voxtral_build_graph_embed(voxtral_context * ctx, int n_tokens) {
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
// Public C API — KV cache + embed + run_llm_kv
// ===========================================================================

extern "C" bool voxtral_kv_init(voxtral_context * ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0) return false;
    if (ctx->kv_k) return true;
    const auto & hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim, n_kv = (int)hp.llm_n_kv_heads, nl = (int)hp.llm_n_layers;
    ggml_init_params kp = { ggml_tensor_overhead()*4+1024, nullptr, true };
    ctx->kv_ctx = ggml_init(kp);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
    ggml_set_name(ctx->kv_k, "kv_k"); ggml_set_name(ctx->kv_v, "kv_v");
    size_t kb = ggml_nbytes(ctx->kv_k), vb = ggml_nbytes(ctx->kv_v);
    ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, kb + vb);
    if (!ctx->kv_buf) { fprintf(stderr, "voxtral: kv alloc failed\n"); return false; }
    char * base = (char *)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kb);
    ctx->kv_max_ctx = max_ctx; ctx->kv_n_used = 0;
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "voxtral: kv cache %d MiB (hd=%d max=%d n_kv=%d nl=%d)\n",
                (int)((kb+vb)/1048576), hd, max_ctx, n_kv, nl);
    return true;
}

extern "C" void voxtral_kv_reset(voxtral_context * ctx) { if (ctx) ctx->kv_n_used = 0; }

extern "C" float * voxtral_embed_tokens(voxtral_context * ctx,
                                        const int32_t * input_ids, int n_tokens) {
    if (!ctx || !input_ids || n_tokens <= 0) return nullptr;
    const int d = (int)ctx->model.hparams.llm_d_model;
    if (ctx->sched) { ggml_backend_sched_free(ctx->sched); ctx->sched = nullptr; }
    ctx->compute_meta.assign(ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false), 0);
    ggml_backend_t be[2] = { ctx->backend, ctx->backend_cpu };
    int nb = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(be, nullptr, nb, 64, false, false);
    ggml_cgraph * gf = voxtral_build_graph_embed(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf,"input_ids"), input_ids, 0,
                            (size_t)n_tokens*sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return nullptr;
    ggml_tensor * out = ggml_graph_get_tensor(gf, "embeds");
    float * r = (float*)malloc((size_t)d*n_tokens*sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)d*n_tokens*sizeof(float));
    return r;
}

extern "C" float * voxtral_run_llm_kv(voxtral_context * ctx,
                                      const float * inputs_embeds,
                                      int n_tokens, int n_past,
                                      int * out_n_tokens, int * out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0 || !ctx->kv_k) return nullptr;
    const auto & hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model, vocab = (int)hp.llm_vocab_size, Lk = n_past+n_tokens;

    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++) pos[i] = n_past + i;
    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        mask.assign((size_t)Lk*n_tokens, ggml_fp32_to_fp16(0.0f));
        ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++)
            for (int k = n_past+q+1; k < Lk; k++)
                mask[(size_t)q*Lk+k] = neg_inf;
    }

    if (ctx->sched) { ggml_backend_sched_free(ctx->sched); ctx->sched = nullptr; }
    ctx->compute_meta.assign(ggml_tensor_overhead()*16384+ggml_graph_overhead_custom(16384,false), 0);
    ggml_backend_t be[2] = { ctx->backend, ctx->backend_cpu };
    int nb = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(be, nullptr, nb, 16384, false, false);

    ggml_cgraph * gf = voxtral_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "voxtral: failed to alloc llm_kv graph\n"); return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf,"inputs_embeds"), inputs_embeds, 0,
                            (size_t)d*n_tokens*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf,"positions"), pos.data(), 0,
                            pos.size()*sizeof(int32_t));
    if (n_tokens > 1)
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf,"causal_mask"), mask.data(), 0,
                                mask.size()*sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "voxtral: llm_kv graph compute failed\n"); return nullptr;
    }
    ggml_tensor * out = ggml_graph_get_tensor(gf, "logits");
    if (!out) return nullptr;
    ctx->kv_n_used = Lk;
    if (out_n_tokens) *out_n_tokens = 1;
    if (out_vocab_size) *out_vocab_size = vocab;
    float * r = (float*)malloc((size_t)vocab*sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)vocab*sizeof(float));
    return r;
}

extern "C" float * voxtral_run_encoder(voxtral_context * ctx,
                                      const float * mel_features,
                                      int n_mels, int T_mel,
                                      int * out_N, int * out_dim) {
    if (!ctx || !mel_features) return nullptr;
    const auto & hp = ctx->model.hparams;
    if (n_mels != (int)hp.n_mels || T_mel != 3000) {
        fprintf(stderr, "voxtral: encoder expects (128, 3000) mel, got (%d, %d)\n",
                n_mels, T_mel);
        return nullptr;
    }

    if (ctx->sched) { ggml_backend_sched_free(ctx->sched); ctx->sched = nullptr; }
    ctx->compute_meta.assign(
        ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), 0);
    ggml_backend_t backends[2] = { ctx->backend, ctx->backend_cpu };
    int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);

    ggml_cgraph * gf = voxtral_build_graph_encoder(ctx);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "voxtral: failed to alloc encoder graph\n"); return nullptr;
    }

    ggml_tensor * mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel_features, 0,
                            (size_t)n_mels * T_mel * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "voxtral: encoder graph compute failed\n"); return nullptr;
    }

    ggml_tensor * out = ggml_graph_get_tensor(gf, "encoder_out");
    if (!out) { fprintf(stderr, "voxtral: missing encoder_out\n"); return nullptr; }
    const int pdim = (int)out->ne[0];   // 3072
    const int N    = (int)out->ne[1];   // 375
    if (out_N)   *out_N   = N;
    if (out_dim) *out_dim = pdim;
    const size_t total = (size_t)pdim * N;
    float * result = (float *)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" int32_t * voxtral_tokenize(voxtral_context * /*ctx*/,
                                      const char * /*text*/, int * out_n_tokens) {
    // Stage V3: not yet implemented. The Tekken vocab blobs are loaded into
    // ctx->vocab but the encode logic (rank-based byte BPE + tiktoken-style
    // pre-tokenizer regex with Unicode property classes) is deferred.
    if (out_n_tokens) *out_n_tokens = 0;
    return nullptr;
}

extern "C" float * voxtral_run_llm(voxtral_context * ctx,
                                   const int32_t * input_ids, int n_tokens,
                                   int * out_n_tokens, int * out_vocab_size) {
    if (!ctx || !input_ids || n_tokens <= 0) return nullptr;
    const auto & hp = ctx->model.hparams;
    const int vocab = (int)hp.llm_vocab_size;

    // Positions [0, T)
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) positions[i] = i;

    // Causal mask: F16 (T, T), -inf above diagonal, 0 elsewhere.
    // ggml ne[0]=k (key, fast), ne[1]=q (query). mask[k > q] = -inf.
    std::vector<ggml_fp16_t> mask((size_t)n_tokens * n_tokens,
                                  ggml_fp32_to_fp16(0.0f));
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < n_tokens; q++) {
        for (int k = q + 1; k < n_tokens; k++) {
            mask[(size_t)q * n_tokens + k] = neg_inf;
        }
    }

    // Lazy sched + compute_meta init
    if (ctx->sched) {
        ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }
    ctx->compute_meta.assign(
        ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), 0);
    ggml_backend_t backends[2] = { ctx->backend, ctx->backend_cpu };
    int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);

    ggml_cgraph * gf = voxtral_build_graph_llm(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "voxtral: failed to alloc llm graph\n");
        return nullptr;
    }

    ggml_tensor * ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, input_ids, 0, (size_t)n_tokens * sizeof(int32_t));

    ggml_tensor * pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0,
                            positions.size() * sizeof(int32_t));

    ggml_tensor * mask_in = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0,
                            mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "voxtral: llm graph compute failed\n");
        return nullptr;
    }

    ggml_tensor * out = ggml_graph_get_tensor(gf, "logits");
    if (!out) { fprintf(stderr, "voxtral: missing logits tensor\n"); return nullptr; }
    // Last-token-only output: returns (vocab,) for the final position.
    if (out_n_tokens)   *out_n_tokens   = 1;
    if (out_vocab_size) *out_vocab_size = vocab;
    float * result = (float *)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)vocab * sizeof(float));
    return result;
}
