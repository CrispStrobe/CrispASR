// qwen3_asr.cpp — Qwen/Qwen3-ASR-0.6B ggml runtime
//
// STAGE 1 (current commit): loader + audio encoder conv front-end only.
//   - Loads the GGUF produced by models/convert-qwen3-asr-to-gguf.py
//   - Computes the per-chunk Conv2D subsampler (conv2d1/2/3 + GELU) and the
//     conv_out linear projection. Output shape (num_chunks, T_chunk_out, 896).
//   - Exposed via qwen3_asr_run_conv() for differential testing against
//     /tmp/qwen3-asr-ref/jfk/conv_out.npy
//
// Subsequent stages will add the chunked self-attention encoder body, the
// projector head, the Qwen3 0.6B LLM forward, and the audio-injection glue.
//
// See qwen3-asr-todo.md for the full plan.

#include "qwen3_asr.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
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
#include <vector>

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct qwen3_asr_hparams {
    // Audio
    uint32_t sample_rate    = 16000;
    uint32_t n_mels         = 128;
    uint32_t n_fft          = 400;
    uint32_t win_length     = 400;
    uint32_t hop_length     = 160;
    uint32_t audio_n_layers = 18;
    uint32_t audio_d_model  = 896;
    uint32_t audio_n_heads  = 14;
    uint32_t audio_head_dim = 64;
    uint32_t audio_ff_dim   = 3584;
    uint32_t audio_conv_ch  = 480;
    uint32_t audio_proj_dim = 1024;
    uint32_t audio_max_pos  = 1500;

    // Chunking parameters (from reference impl: n_window=50, n_window_infer=800)
    uint32_t n_window       = 50;
    uint32_t n_window_infer = 800;

    // LLM (Qwen3 0.6B)
    uint32_t llm_n_layers   = 28;
    uint32_t llm_d_model    = 1024;
    uint32_t llm_n_heads    = 16;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim   = 128;
    uint32_t llm_ff_dim     = 3072;
    float    llm_rope_theta = 1e6f;
    float    llm_rms_eps    = 1e-6f;
    uint32_t llm_vocab_size = 151936;
    uint32_t llm_max_pos    = 65536;

    // Special tokens
    uint32_t audio_start_token_id = 151669;
    uint32_t audio_end_token_id   = 151670;
    uint32_t audio_pad_token_id   = 151676;
    uint32_t eos_token_id         = 151645;
    uint32_t pad_token_id         = 151643;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct qwen3_asr_audio_block {
    // Pre-LN self-attention
    ggml_tensor * attn_norm_w  = nullptr, * attn_norm_b  = nullptr;
    ggml_tensor * attn_q_w     = nullptr, * attn_q_b     = nullptr;
    ggml_tensor * attn_k_w     = nullptr, * attn_k_b     = nullptr;
    ggml_tensor * attn_v_w     = nullptr, * attn_v_b     = nullptr;
    ggml_tensor * attn_out_w   = nullptr, * attn_out_b   = nullptr;
    // Pre-LN FFN (GELU)
    ggml_tensor * ffn_norm_w   = nullptr, * ffn_norm_b   = nullptr;
    ggml_tensor * ffn_up_w     = nullptr, * ffn_up_b     = nullptr;
    ggml_tensor * ffn_down_w   = nullptr, * ffn_down_b   = nullptr;
};

struct qwen3_asr_audio_tower {
    // Conv subsampler front-end (4 stride-2 freq convs as 2D over the mel image)
    ggml_tensor * conv1_w = nullptr, * conv1_b = nullptr;  // (480, 1,   3, 3)
    ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;  // (480, 480, 3, 3)
    ggml_tensor * conv3_w = nullptr, * conv3_b = nullptr;  // (480, 480, 3, 3)
    ggml_tensor * conv_out_w = nullptr, * conv_out_b = nullptr; // (896, 7680)

    // Encoder body
    std::vector<qwen3_asr_audio_block> blocks;

    // Final norm + projector head (896 → 896 → GELU → 1024)
    ggml_tensor * ln_post_w = nullptr, * ln_post_b = nullptr;
    ggml_tensor * proj1_w   = nullptr, * proj1_b   = nullptr;
    ggml_tensor * proj2_w   = nullptr, * proj2_b   = nullptr;

    // Mel preprocessor (baked from WhisperFeatureExtractor by the converter)
    ggml_tensor * mel_filters = nullptr;  // (n_freqs=201, n_mels=128) F32
    ggml_tensor * mel_window  = nullptr;  // (400,) F32 hann window
};

struct qwen3_asr_llm_block {
    ggml_tensor * attn_norm_w   = nullptr;
    ggml_tensor * attn_q_w      = nullptr;
    ggml_tensor * attn_k_w      = nullptr;
    ggml_tensor * attn_v_w      = nullptr;
    ggml_tensor * attn_output_w = nullptr;
    ggml_tensor * attn_q_norm_w = nullptr;  // Qwen3 per-head Q RMSNorm
    ggml_tensor * attn_k_norm_w = nullptr;
    ggml_tensor * ffn_norm_w    = nullptr;
    ggml_tensor * ffn_gate_w    = nullptr;
    ggml_tensor * ffn_up_w      = nullptr;
    ggml_tensor * ffn_down_w    = nullptr;
};

struct qwen3_asr_llm {
    ggml_tensor * token_embd_w = nullptr;     // (151936, 1024)
    std::vector<qwen3_asr_llm_block> blocks;
    ggml_tensor * output_norm_w = nullptr;
    ggml_tensor * output_w      = nullptr;
};

struct qwen3_asr_model {
    qwen3_asr_hparams     hparams;
    qwen3_asr_audio_tower audio;
    qwen3_asr_llm         llm;

    ggml_context        * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;

    // Sinusoidal positional embedding for the audio encoder, computed once
    // at load time. Layout: row-major (max_pos, d_model) where row p is the
    // pos embed for position p.
    std::vector<float> audio_pe;   // size = audio_max_pos * audio_d_model
};

struct qwen3_asr_vocab {
    std::vector<std::string> id_to_token;
};

struct qwen3_asr_context {
    qwen3_asr_context_params params;

    qwen3_asr_model model;
    qwen3_asr_vocab vocab;

    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    std::vector<uint8_t> compute_meta;

    // KV cache (Stage 5). Single tensor for K, single for V, both shape
    // (head_dim, max_ctx, n_kv_heads, n_layers). Allocated to backend.
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

static ggml_tensor * try_get(qwen3_asr_model & m, const char * name) {
    auto it = m.tensors.find(name);
    return it != m.tensors.end() ? it->second : nullptr;
}

static ggml_tensor * require(qwen3_asr_model & m, const char * name) {
    auto t = try_get(m, name);
    if (!t) {
        fprintf(stderr, "qwen3_asr: required tensor '%s' not found in GGUF\n", name);
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

static bool qwen3_asr_load_model(qwen3_asr_model & model,
                                 qwen3_asr_vocab & vocab,
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
            fprintf(stderr, "qwen3_asr: failed to open '%s'\n", path);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }

        auto & hp = model.hparams;
        hp.sample_rate    = kv_u32(gctx, "qwen3asr.sample_rate",          hp.sample_rate);
        hp.n_mels         = kv_u32(gctx, "qwen3asr.n_mels",               hp.n_mels);
        hp.n_fft          = kv_u32(gctx, "qwen3asr.n_fft",                hp.n_fft);
        hp.win_length     = kv_u32(gctx, "qwen3asr.win_length",           hp.win_length);
        hp.hop_length     = kv_u32(gctx, "qwen3asr.hop_length",           hp.hop_length);
        hp.audio_n_layers = kv_u32(gctx, "qwen3asr.audio.n_layers",       hp.audio_n_layers);
        hp.audio_d_model  = kv_u32(gctx, "qwen3asr.audio.d_model",        hp.audio_d_model);
        hp.audio_n_heads  = kv_u32(gctx, "qwen3asr.audio.n_heads",        hp.audio_n_heads);
        hp.audio_head_dim = kv_u32(gctx, "qwen3asr.audio.head_dim",       hp.audio_head_dim);
        hp.audio_ff_dim   = kv_u32(gctx, "qwen3asr.audio.ff_dim",         hp.audio_ff_dim);
        hp.audio_conv_ch  = kv_u32(gctx, "qwen3asr.audio.conv_channels",  hp.audio_conv_ch);
        hp.audio_proj_dim = kv_u32(gctx, "qwen3asr.audio.proj_dim",       hp.audio_proj_dim);
        hp.audio_max_pos  = kv_u32(gctx, "qwen3asr.audio.max_source_pos", hp.audio_max_pos);

        hp.llm_n_layers   = kv_u32(gctx, "qwen3asr.llm.n_layers",   hp.llm_n_layers);
        hp.llm_d_model    = kv_u32(gctx, "qwen3asr.llm.d_model",    hp.llm_d_model);
        hp.llm_n_heads    = kv_u32(gctx, "qwen3asr.llm.n_heads",    hp.llm_n_heads);
        hp.llm_n_kv_heads = kv_u32(gctx, "qwen3asr.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim   = kv_u32(gctx, "qwen3asr.llm.head_dim",   hp.llm_head_dim);
        hp.llm_ff_dim     = kv_u32(gctx, "qwen3asr.llm.ff_dim",     hp.llm_ff_dim);
        hp.llm_rope_theta = kv_f32(gctx, "qwen3asr.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps    = kv_f32(gctx, "qwen3asr.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = kv_u32(gctx, "qwen3asr.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos    = kv_u32(gctx, "qwen3asr.llm.max_pos",    hp.llm_max_pos);

        hp.audio_start_token_id = kv_u32(gctx, "qwen3asr.audio_start_token_id", hp.audio_start_token_id);
        hp.audio_end_token_id   = kv_u32(gctx, "qwen3asr.audio_end_token_id",   hp.audio_end_token_id);
        hp.audio_pad_token_id   = kv_u32(gctx, "qwen3asr.audio_pad_token_id",   hp.audio_pad_token_id);
        hp.eos_token_id         = kv_u32(gctx, "qwen3asr.eos_token_id",         hp.eos_token_id);
        hp.pad_token_id         = kv_u32(gctx, "qwen3asr.pad_token_id",         hp.pad_token_id);

        int ki = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (ki >= 0) {
            int n = gguf_get_arr_n(gctx, ki);
            vocab.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                vocab.id_to_token[i] = gguf_get_arr_str(gctx, ki, i);
            }
        }

        gguf_free(gctx);
        ggml_free(meta_ctx);
    }

    // ---- pass 2: load tensor metadata + bind into a backend buffer ----
    ggml_context * weight_ctx = nullptr;
    {
        gguf_init_params load_params = { /*no_alloc=*/true, /*ctx=*/&weight_ctx };
        gguf_context * gctx = gguf_init_from_file(path, load_params);
        if (!gctx || !weight_ctx) {
            fprintf(stderr, "qwen3_asr: failed to load tensor metadata\n");
            return false;
        }

        model.buf = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);

        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "qwen3_asr: open failed\n"); return false; }
        struct stat st; fstat(fd, &st);
        size_t file_size = (size_t)st.st_size;
        void * mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (mmap_base == MAP_FAILED) {
            fprintf(stderr, "qwen3_asr: mmap failed\n");
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
    a.conv3_w = require(model, "audio.conv.3.weight");
    a.conv3_b = require(model, "audio.conv.3.bias");
    a.conv_out_w = require(model, "audio.conv_out.weight");
    a.conv_out_b = try_get(model, "audio.conv_out.bias");  // bias may be absent
    a.ln_post_w  = require(model, "audio.ln_post.weight");
    a.ln_post_b  = require(model, "audio.ln_post.bias");
    a.proj1_w    = require(model, "audio.proj1.weight");
    a.proj1_b    = require(model, "audio.proj1.bias");
    a.proj2_w    = require(model, "audio.proj2.weight");
    a.proj2_b    = require(model, "audio.proj2.bias");
    a.mel_filters = try_get(model, "audio.mel_filters");  // optional (may be missing in older GGUFs)
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
        b.attn_k_b    = get("attn_k.bias");
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
        b.attn_q_norm_w = get("attn_q_norm.weight");
        b.attn_k_norm_w = get("attn_k_norm.weight");
        b.ffn_norm_w    = get("ffn_norm.weight");
        b.ffn_gate_w    = get("ffn_gate.weight");
        b.ffn_up_w      = get("ffn_up.weight");
        b.ffn_down_w    = get("ffn_down.weight");
    }

    // ---- precompute sinusoidal positional embedding for the audio encoder ----
    // Reference: SinusoidsPositionEmbedding in modeling_qwen3_asr.py
    //   log_inc = log(10000) / (C/2 - 1)
    //   inv_t   = exp(-log_inc * arange(C/2))
    //   pe[p, :C/2] = sin(p * inv_t)
    //   pe[p, C/2:] = cos(p * inv_t)
    {
        const int C = (int)model.hparams.audio_d_model;
        const int L = (int)model.hparams.audio_max_pos;
        const int half = C / 2;
        const float log_inc = std::log(10000.0f) / (float)(half - 1);
        std::vector<float> inv_t(half);
        for (int i = 0; i < half; i++) inv_t[i] = std::exp(-log_inc * (float)i);
        model.audio_pe.assign((size_t)L * C, 0.0f);
        for (int p = 0; p < L; p++) {
            float * row = model.audio_pe.data() + (size_t)p * C;
            for (int i = 0; i < half; i++) {
                float angle = (float)p * inv_t[i];
                row[i]        = std::sin(angle);
                row[half + i] = std::cos(angle);
            }
        }
    }

    return true;
}

// ===========================================================================
// FFT (Cooley-Tukey for even sizes, falls back to DFT for odd leaves).
// Handles n_fft=400 (= 2^4 * 25) by recursing down to a 25-point DFT.
// ===========================================================================

static void qwen3_asr_dft(const float * in, int N, float * out) {
    for (int k = 0; k < N; k++) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; n++) {
            float ang = -2.0f * (float)M_PI * (float)k * (float)n / (float)N;
            re += in[n] * std::cos(ang);
            im += in[n] * std::sin(ang);
        }
        out[2*k]   = re;
        out[2*k+1] = im;
    }
}

// Real-input FFT, output complex (out has 2*N floats interleaved real/imag).
// in/out are scratch buffers; in must have at least 2*N floats of writable space.
static void qwen3_asr_fft(float * in, int N, float * out) {
    if (N == 1) { out[0] = in[0]; out[1] = 0.0f; return; }
    int half_N = N / 2;
    if (N - half_N * 2 == 1) { qwen3_asr_dft(in, N, out); return; }

    float * even = in + N;
    for (int i = 0; i < half_N; i++) even[i] = in[2*i];
    float * even_fft = out + 2 * N;
    qwen3_asr_fft(even, half_N, even_fft);

    float * odd = even;
    for (int i = 0; i < half_N; i++) odd[i] = in[2*i + 1];
    float * odd_fft = even_fft + N;
    qwen3_asr_fft(odd, half_N, odd_fft);

    for (int k = 0; k < half_N; k++) {
        float ang = -2.0f * (float)M_PI * (float)k / (float)N;
        float re = std::cos(ang);
        float im = std::sin(ang);
        float re_odd = odd_fft[2*k];
        float im_odd = odd_fft[2*k+1];
        out[2*k]               = even_fft[2*k]   + re*re_odd - im*im_odd;
        out[2*k+1]             = even_fft[2*k+1] + re*im_odd + im*re_odd;
        out[2*(k + half_N)]    = even_fft[2*k]   - re*re_odd + im*im_odd;
        out[2*(k + half_N)+1]  = even_fft[2*k+1] - re*im_odd - im*re_odd;
    }
}

// ===========================================================================
// Whisper-style log-mel spectrogram
//
// Pipeline (matches WhisperFeatureExtractor._np_extract_fbank_features):
//   1. center-pad audio with n_fft/2 zeros on each side
//   2. STFT: hann window length 400, hop 160, n_fft 400 → (n_freqs=201, T)
//   3. power = |STFT|^2
//   4. mel = power @ filters^T → (n_mels=128, T)
//   5. log10(max(mel, 1e-10))
//   6. drop the last frame
//   7. clip: log_spec = max(log_spec, log_spec.max() - 8.0)
//   8. normalize: log_spec = (log_spec + 4) / 4
//
// Returns a flat (n_mels, T) row-major buffer.
// ===========================================================================

extern "C" float * qwen3_asr_compute_mel(qwen3_asr_context * ctx,
                                         const float * samples, int n_samples,
                                         int * out_n_mels, int * out_T_mel) {
    if (!ctx || !samples || n_samples <= 0) return nullptr;
    const auto & hp = ctx->model.hparams;
    if (!ctx->model.audio.mel_filters || !ctx->model.audio.mel_window) {
        fprintf(stderr, "qwen3_asr: model GGUF missing audio.mel_filters / audio.mel_window\n");
        return nullptr;
    }

    const int n_fft   = (int)hp.n_fft;        // 400
    const int hop     = (int)hp.hop_length;   // 160
    const int n_mels  = (int)hp.n_mels;       // 128
    const int n_freqs = n_fft / 2 + 1;        // 201

    // Pull window and filterbank from the backend
    std::vector<float> hann(n_fft);
    ggml_backend_tensor_get(ctx->model.audio.mel_window, hann.data(), 0,
                            n_fft * sizeof(float));
    std::vector<float> filt((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(ctx->model.audio.mel_filters, filt.data(), 0,
                            filt.size() * sizeof(float));

    // Center-pad audio with n_fft/2 zeros on each side
    const int pad = n_fft / 2;
    std::vector<float> padded((size_t)n_samples + 2 * pad, 0.0f);
    std::memcpy(padded.data() + pad, samples, n_samples * sizeof(float));

    // Number of STFT frames before dropping the last one
    const int T_full = (int)((padded.size() - n_fft) / hop + 1);
    const int T = T_full - 1;  // Whisper drops the last frame
    if (T <= 0) return nullptr;

    // STFT → power spectrum, store as (n_freqs, T) row-major
    std::vector<float> power((size_t)n_freqs * T, 0.0f);
    {
        // Scratch space for the recursive FFT (worst case ~6N floats)
        std::vector<float> fft_in((size_t)n_fft * 4, 0.0f);
        std::vector<float> fft_out((size_t)n_fft * 8, 0.0f);
        for (int t = 0; t < T; t++) {
            const float * frame = padded.data() + (size_t)t * hop;
            for (int n = 0; n < n_fft; n++) fft_in[n] = frame[n] * hann[n];
            qwen3_asr_fft(fft_in.data(), n_fft, fft_out.data());
            for (int k = 0; k < n_freqs; k++) {
                float re = fft_out[2*k], im = fft_out[2*k+1];
                power[(size_t)k * T + t] = re * re + im * im;
            }
        }
    }

    // Mel projection. WhisperFeatureExtractor.mel_filters has shape (n_freqs, n_mels)
    // (i.e. filters[k, m] = coefficient for mel band m at freq bin k). The numpy
    // array is row-major so the byte layout is filters[k * n_mels + m].
    //
    // log_mel[m, t] = log10(max(sum_k filters[k, m] * power[k, t], 1e-10))
    std::vector<float> mel((size_t)n_mels * T, 0.0f);
    float mel_max = -1e30f;
    for (int m = 0; m < n_mels; m++) {
        for (int t = 0; t < T; t++) {
            double s = 0.0;
            for (int k = 0; k < n_freqs; k++) {
                s += (double)filt[(size_t)k * n_mels + m] * power[(size_t)k * T + t];
            }
            float lv = std::log10(std::max((float)s, 1e-10f));
            mel[(size_t)m * T + t] = lv;
            if (lv > mel_max) mel_max = lv;
        }
    }

    // Clip + normalize
    const float floor_v = mel_max - 8.0f;
    for (size_t i = 0; i < mel.size(); i++) {
        float v = mel[i];
        if (v < floor_v) v = floor_v;
        mel[i] = (v + 4.0f) / 4.0f;
    }

    if (out_n_mels) *out_n_mels = n_mels;
    if (out_T_mel)  *out_T_mel  = T;

    float * result = (float *)malloc(mel.size() * sizeof(float));
    std::memcpy(result, mel.data(), mel.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Conv front-end graph (Stage 1)
//
// Input  (set on the CPU side as a contiguous F32 buffer):
//   mel_batched: shape (T_chunk, n_mels, 1, num_chunks)  in ggml ne order
//                = num_chunks chunks of (1, n_mels, T_chunk) per the
//                  reference impl's per-chunk processing
//
// Output:
//   conv_out: shape (audio_d_model, T_chunk_out, num_chunks)
//             = num_chunks chunks of (T_chunk_out, audio_d_model) frames
//
// Each chunk is processed independently through:
//   conv2d1 + bias + GELU      (in_ch=1,   out_ch=480, k=3, stride=2, pad=1)
//   conv2d2 + bias + GELU      (in_ch=480, out_ch=480, k=3, stride=2, pad=1)
//   conv2d3 + bias + GELU      (in_ch=480, out_ch=480, k=3, stride=2, pad=1)
//   permute + flatten freq → (num_chunks * T_chunk_out, 480 * F_out)
//   conv_out linear (480*16=7680 → 896) + optional bias
//
// For our reference test on jfk.wav:
//   T_chunk=100, n_mels=128, num_chunks=11 → conv1: (50,64,480), conv2:
//   (25,32,480), conv3: (13,16,480) → flatten: (13, 7680) → linear: (13, 896)
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

static ggml_cgraph * qwen3_asr_build_graph_conv(qwen3_asr_context * ctx,
                                                int T_chunk, int num_chunks) {
    const auto & m  = ctx->model;
    const auto & hp = m.hparams;
    const int n_mels = (int)hp.n_mels;
    const int d      = (int)hp.audio_d_model;

    ggml_init_params ip = {
        /*mem_size=*/   ctx->compute_meta.size(),
        /*mem_buffer=*/ ctx->compute_meta.data(),
        /*no_alloc=*/   true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);

    // Input: ggml 2D conv expects (W, H, C, N) where ne[0]=W (fast), ne[3]=N
    // For per-chunk processing of (1, n_mels=128, T_chunk=100):
    //   ne[0] = T_chunk  (time, varies fastest)
    //   ne[1] = n_mels   (frequency)
    //   ne[2] = 1        (in channels)
    //   ne[3] = num_chunks (batch)
    ggml_tensor * mel = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
                                           T_chunk, n_mels, 1, num_chunks);
    ggml_set_name(mel, "mel_batched");
    ggml_set_input(mel);

    auto bias_4d = [&](ggml_context * c0, ggml_tensor * b) {
        // bias is (out_ch,) — broadcast as (1, 1, out_ch, 1) for elementwise add
        return ggml_cast(c0, ggml_reshape_4d(c0, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    // Conv1: in=1, out=480, k=3, stride=2, pad=1
    ggml_tensor * cur = ggml_conv_2d(ctx0, m.audio.conv1_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);

    // Conv2: in=480, out=480, k=3, stride=2, pad=1
    cur = ggml_conv_2d(ctx0, m.audio.conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);

    // Conv3: in=480, out=480, k=3, stride=2, pad=1
    cur = ggml_conv_2d(ctx0, m.audio.conv3_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv3_b));
    cur = ggml_gelu_erf(ctx0, cur);

    // After conv3: shape (T_out, F_out, 480, num_chunks)
    // For T_chunk=100, n_mels=128: T_out=13, F_out=16
    const int T_out = (int)cur->ne[0];
    const int F_out = (int)cur->ne[1];
    const int C_out = (int)cur->ne[2];   // 480
    GGML_ASSERT(C_out == (int)hp.audio_conv_ch);

    // Reference does: padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c*f)
    // PyTorch shape (B, C, F, T) → permute(0, 3, 1, 2) → (B, T, C, F) → flatten last two
    // Our ggml shape is (T, F, C, B). We want (T, C*F, B) where C*F is contiguous so
    // that the linear in conv_out (which expects 7680 input dim) gets the right
    // memory layout. PyTorch's view(b, t, c*f) over (B, T, C, F) means C is the
    // outer index, F is the inner index → memory order: f0c0, f1c0, ..., f15c0,
    // f0c1, ... = (F + F*C). To match, our final layout should be (F + F*C) along
    // the fast axis = ne[0] = F*C with inner stride F.
    //
    // Currently: ne = (T, F, C, B). We want ne = (F*C, T, B) with C as inner.
    // Permute to put C inner: (T, F, C, B) → (C, F, T, B)? No, we want C inner of F.
    // Let's permute so axes order becomes (C, F, T, B) — then C is fast (ne[0]),
    // F is next (ne[1]), so memory is c0f0, c1f0, ..., c479f0, c0f1, ...
    // That's the order f outer, c inner = c + C*f. Reshape (C*F, T, B) gives us
    // (c+C*f, t, b) — which equals PyTorch's (b, t, c*F + f) — wait that's NOT
    // what PyTorch does. PyTorch's view(b, t, c*f) treats it as a flat dim where
    // PyTorch's prior layout was (B, T, C, F) → memory: t outer, c middle, f inner
    // → flat index along last dim = c*F + f. So fast-axis index = c*F + f, with
    // c outer and f inner. Our target ggml memory is therefore (f + F*c) along
    // ne[0]. Permute (T, F, C, B) → axes (1, 2, 0, 3): puts F at ne[0], C at ne[1].
    // Memory order: f0c0, f1c0, ..., F-1 c0, f0c1, ... = (f + F*c). YES.
    //
    // ggml_permute(t, p0, p1, p2, p3) semantics: source axis i goes to NEW
    // position p_i. So to get new ne = (F, C, T, B) from source (T, F, C, B):
    //   source 0 (T) → new pos 2  → p0 = 2
    //   source 1 (F) → new pos 0  → p1 = 0
    //   source 2 (C) → new pos 1  → p2 = 1
    //   source 3 (B) → new pos 3  → p3 = 3
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
    cur = ggml_reshape_3d(ctx0, cur, F_out * C_out, T_out, num_chunks);

    // Linear: cur is (F*C, T, B) = (7680, 13, 11). conv_out_w is stored as
    // ggml shape (7680, 896) — i.e. ne[0]=7680, ne[1]=896. ggml_mul_mat(A, B)
    // computes B^T @ A^T with output ne[0] = A->ne[1], ne[1] = B->ne[1].
    // We want output (896, T, B). With cur as B (7680, T*B effectively), and
    // mul_mat(conv_out_w, cur): output ne[0] = conv_out_w->ne[1] = 896,
    // ne[1..] inherit from cur. ✓
    cur = ggml_mul_mat(ctx0, m.audio.conv_out_w, cur);
    if (m.audio.conv_out_b) {
        cur = ggml_add(ctx0, cur, m.audio.conv_out_b);
    }
    // cur shape now: (896, T_out, num_chunks)

    ggml_set_name(cur, "conv_front_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Full encoder graph (Stage 2)
//
// Pipeline (matching modeling_qwen3_asr.Qwen3ASRAudioEncoder.forward):
//   1. Per-chunk Conv2D subsampler  → (896, 13, num_chunks)        [as Stage 1]
//   2. Add sinusoidal pos embed (broadcast over chunks)            → same shape
//   3. Reshape to flat (896, N_padded) where N_padded = 13*num_chunks
//      [Stage-2 simplification: assumes all chunks are full, no padding mask]
//   4. 18 × Whisper-style pre-LN encoder block:
//        residual = x
//        x = LN1(x)
//        Q,K,V = x @ {Wq,Wk,Wv} + bias
//        attn = softmax((Q @ K^T)/sqrt(hd) + window_mask) @ V
//        x = residual + Wo @ attn + bo
//        residual = x
//        x = LN2(x); x = GELU(W1 x + b1); x = W2 x + b2
//        x = residual + x
//   5. ln_post → proj1 → GELU → proj2  →  (1024, N_padded)
//
// The "window_mask" implements the chunked attention from the reference:
// each position only attends within its window of size 104. The mask is
// supplied as an input tensor (N_padded, N_padded) F32 with -inf in
// disallowed positions and 0 in allowed ones.
// ===========================================================================

static ggml_cgraph * qwen3_asr_build_graph_encoder(qwen3_asr_context * ctx,
                                                   int T_chunk,
                                                   int num_chunks,
                                                   int T_chunk_out_expected) {
    const auto & m  = ctx->model;
    const auto & hp = m.hparams;
    const int n_mels   = (int)hp.n_mels;
    const int d        = (int)hp.audio_d_model;        // 896
    const int n_heads  = (int)hp.audio_n_heads;        // 14
    const int head_dim = (int)hp.audio_head_dim;       // 64
    const int proj_dim = (int)hp.audio_proj_dim;       // 1024

    ggml_init_params ip = {
        /*mem_size=*/   ctx->compute_meta.size(),
        /*mem_buffer=*/ ctx->compute_meta.data(),
        /*no_alloc=*/   true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 8192, false);

    // ------- Inputs -------
    // mel_batched ne = (T_chunk, n_mels, 1, num_chunks)
    ggml_tensor * mel = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
                                           T_chunk, n_mels, 1, num_chunks);
    ggml_set_name(mel, "mel_batched");
    ggml_set_input(mel);

    // pe_input ne = (d, T_chunk_out, 1, 1)  — broadcasts over batch
    ggml_tensor * pe_in = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32,
                                             d, T_chunk_out_expected, 1);
    ggml_set_name(pe_in, "pe_input");
    ggml_set_input(pe_in);

    // attn_mask ne = (N_padded, N_padded)  F32 with 0 / -inf
    const int N_padded = T_chunk_out_expected * num_chunks;
    ggml_tensor * mask_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                                               N_padded, N_padded);
    ggml_set_name(mask_in, "attn_mask");
    ggml_set_input(mask_in);

    // ------- Conv front-end (same as Stage 1) -------
    auto bias_4d = [&](ggml_context * c0, ggml_tensor * b) {
        return ggml_cast(c0, ggml_reshape_4d(c0, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    ggml_tensor * cur = ggml_conv_2d(ctx0, m.audio.conv1_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_conv_2d(ctx0, m.audio.conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_conv_2d(ctx0, m.audio.conv3_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv3_b));
    cur = ggml_gelu_erf(ctx0, cur);
    // cur ne = (T_out, F_out, 480, num_chunks)
    const int T_out = (int)cur->ne[0];
    const int F_out = (int)cur->ne[1];
    const int C_out = (int)cur->ne[2];
    GGML_ASSERT(T_out == T_chunk_out_expected);

    // Permute (T,F,C,B) → (F,C,T,B): source axis 0(T)→pos 2, 1(F)→0, 2(C)→1, 3(B)→3
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
    cur = ggml_reshape_3d(ctx0, cur, F_out * C_out, T_out, num_chunks);
    cur = ggml_mul_mat(ctx0, m.audio.conv_out_w, cur);  // (d, T_out, num_chunks)

    // ------- Add positional embedding (broadcasts over batch) -------
    // pe_in ne = (d, T_out, 1) → broadcast against (d, T_out, num_chunks)
    cur = ggml_add(ctx0, cur, pe_in);

    // ------- Flatten chunks into a single sequence -------
    // cur ne = (d, T_out, num_chunks). Want (d, N_padded). Memory layout for
    // (d, T_out, num_chunks) row-major (d fastest) is identical to
    // (d, N_padded=T_out*num_chunks) where chunk-major order is preserved.
    // ggml_reshape_2d just relabels strides.
    cur = ggml_cont(ctx0, cur);  // ensure contiguous before reshape
    cur = ggml_reshape_2d(ctx0, cur, d, N_padded);

    // ------- 18 × encoder blocks -------
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);
    for (uint32_t il = 0; il < hp.audio_n_layers; il++) {
        const auto & b = m.audio.blocks[il];
        ggml_tensor * residual = cur;

        // ---- LN1 (pre-attention) ----
        ggml_tensor * x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        x = ggml_add(ctx0, x, b.attn_norm_b);

        // ---- Q, K, V projections (with biases) ----
        ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, x), b.attn_q_b);
        ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_k_w, x), b.attn_k_b);
        ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, x), b.attn_v_b);
        // Q/K/V ne = (d, N_padded). Reshape to (head_dim, n_heads, N_padded),
        // then permute to (head_dim, N_padded, n_heads).
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, N_padded);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_heads, N_padded);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, N_padded);
        // Permute (hd, n_h, N) → (hd, N, n_h): source 0→0, 1→2, 2→1
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        // V layout for the attn @ V step: we want V as (hd, N, n_h) too, then
        // reshape later. Use the same permute.
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        // ---- Scores = (Q @ K^T) ----
        // ggml_mul_mat(K, Q): K ne=(hd, N, n_h), Q ne=(hd, N, n_h)
        // result ne = (N, N, n_h) where result[j, i, h] = dot(K[:, j, h], Q[:, i, h])
        // So result[j, i, h] = sum_d K[d,j,h] * Q[d,i,h] = (Q @ K^T)[i, j, h]
        // ne[0]=j (key index, varies fast), ne[1]=i (query index)
        ggml_tensor * scores = ggml_mul_mat(ctx0, K, Q);

        // Add window mask. mask_in ne=(N, N) F32. ggml_add broadcasts over the
        // n_heads dim (size 1 in mask, size n_h in scores).
        scores = ggml_add(ctx0, scores, mask_in);

        // Softmax along key axis (ne[0]) with scale baked in.
        scores = ggml_soft_max_ext(ctx0, scores, /*mask*/nullptr, attn_scale, 0.0f);

        // ---- attn = scores @ V ----
        // We need: out[d, i, h] = sum_j scores[j, i, h] * V[d, j, h]
        // ggml_mul_mat(V_perm, scores) where V_perm is (j, d, h) so dot is over j.
        // Currently V ne=(hd, N, n_h). We want V indexed as (j, d, h) with j fast.
        // Permute V (hd, N, n_h) → (N, hd, n_h): source 0→1, 1→0, 2→2
        ggml_tensor * V2 = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
        // ggml_mul_mat(V2, scores): V2 ne=(N, hd, n_h), scores ne=(N, N, n_h)
        // dot over ne[0]=N (the j axis). Result ne=(hd, N, n_h) where result[d, i, h]
        // = sum_j V2[j, d, h] * scores[j, i, h] = sum_j V[d, j, h] * scores[j, i, h] ✓
        ggml_tensor * attn = ggml_mul_mat(ctx0, V2, scores);
        // attn ne=(hd, N, n_h). Permute back to (hd, n_h, N) and reshape (d, N).
        // src 0(hd)→0, 1(N)→2, 2(n_h)→1
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, d, N_padded);

        // ---- Output projection (with bias) ----
        attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_out_w, attn), b.attn_out_b);
        cur = ggml_add(ctx0, residual, attn);

        // ---- LN2 + FFN ----
        residual = cur;
        x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        x = ggml_add(ctx0, x, b.ffn_norm_b);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_up_w, x), b.ffn_up_b);
        x = ggml_gelu_erf(ctx0, x);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_down_w, x), b.ffn_down_b);
        cur = ggml_add(ctx0, residual, x);
    }

    // ------- ln_post → proj1 → GELU → proj2 -------
    {
        ggml_tensor * x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, m.audio.ln_post_w);
        x = ggml_add(ctx0, x, m.audio.ln_post_b);
        cur = x;
    }
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.audio.proj1_w, cur), m.audio.proj1_b);
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.audio.proj2_w, cur), m.audio.proj2_b);
    // cur ne = (proj_dim=1024, N_padded)
    (void)proj_dim;

    ggml_set_name(cur, "encoder_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Qwen3 LLM forward graph (Stage 3)
//
// Architecture: 28 layers, hidden=1024, GQA(16/8), head_dim=128, RMSNorm,
//   SwiGLU FFN, RoPE θ=1e6 NEOX-style, Q-norm/K-norm per-head along head_dim.
//
// Pipeline:
//   x = embed(input_ids)              # (1024, T)
//   for layer in 28 layers:
//     residual = x
//     x = RMSNorm(x) * attn_norm_w
//     Q = q_proj(x).view(head_dim, n_q,  T)
//     K = k_proj(x).view(head_dim, n_kv, T)
//     V = v_proj(x).view(head_dim, n_kv, T)
//     Q = q_norm(Q) along head_dim
//     K = k_norm(K) along head_dim
//     Q = rope_neox(Q, positions)
//     K = rope_neox(K, positions)
//     # GQA: repeat K, V from n_kv to n_q heads
//     K_rep = repeat_each(K, n_q / n_kv)   # (head_dim, n_q, T)
//     V_rep = repeat_each(V, n_q / n_kv)
//     # Standard attention
//     scores = (Q @ K_rep^T) * (1/sqrt(head_dim)) + causal_mask
//     attn   = softmax(scores) @ V_rep
//     attn   = o_proj(attn.reshape(d, T))
//     x = residual + attn
//     residual = x
//     x = RMSNorm(x) * ffn_norm_w
//     x = down_proj(silu(gate_proj(x)) * up_proj(x))
//     x = residual + x
//   x = RMSNorm(x) * output_norm_w
//   logits = lm_head(x)               # (vocab, T)
//
// First iteration: no KV cache, full forward each call. Used for diff testing.
// ===========================================================================

// Internal: builds the 28-layer transformer + lm_head graph starting from
// a (d, T) hidden state. Used by both build_graph_llm (which prepends a
// get_rows token-embed lookup) and build_graph_llm_from_embeds (which takes
// pre-computed embeddings as input).
static void qwen3_asr_build_llm_body(qwen3_asr_context * ctx,
                                     ggml_context * ctx0,
                                     ggml_cgraph * gf,
                                     ggml_tensor * cur,        // (d, T) input hidden state
                                     ggml_tensor * positions,
                                     ggml_tensor * causal_mask,
                                     int T) {
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

    for (uint32_t il = 0; il < hp.llm_n_layers; il++) {
        const auto & b = m.llm.blocks[il];
        ggml_tensor * residual = cur;

        // ---- LN1 (RMSNorm + multiplicative weight, no bias) ----
        ggml_tensor * x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // ---- Q, K, V projections ----
        ggml_tensor * Q = ggml_mul_mat(ctx0, b.attn_q_w, x);
        ggml_tensor * K = ggml_mul_mat(ctx0, b.attn_k_w, x);
        ggml_tensor * V = ggml_mul_mat(ctx0, b.attn_v_w, x);
        Q = ggml_reshape_3d(ctx0, Q, hd, n_q,  T);
        K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

        // ---- Q-norm / K-norm ----
        Q = ggml_rms_norm(ctx0, Q, eps);
        Q = ggml_mul(ctx0, Q, b.attn_q_norm_w);
        K = ggml_rms_norm(ctx0, K, eps);
        K = ggml_mul(ctx0, K, b.attn_k_norm_w);

        // ---- RoPE NEOX ----
        Q = ggml_rope_ext(ctx0, Q, positions, nullptr,
                          hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos,
                          theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr,
                          hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos,
                          theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // ---- GQA expand ----
        if (n_kv_grp > 1) {
            ggml_tensor * K4 = ggml_reshape_4d(ctx0, K, hd, 1, n_kv, T);
            ggml_tensor * V4 = ggml_reshape_4d(ctx0, V, hd, 1, n_kv, T);
            K4 = ggml_repeat_4d(ctx0, K4, hd, n_kv_grp, n_kv, T);
            V4 = ggml_repeat_4d(ctx0, V4, hd, n_kv_grp, n_kv, T);
            K = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, n_q, T));
            V = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, n_q, T));
        }

        // ---- Permute for attention ----
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        ggml_tensor * scores = ggml_mul_mat(ctx0, K, Q);
        scores = ggml_add(ctx0, scores, causal_mask);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, attn_scale, 0.0f);

        ggml_tensor * V2 = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
        ggml_tensor * attn = ggml_mul_mat(ctx0, V2, scores);
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

        attn = ggml_mul_mat(ctx0, b.attn_output_w, attn);
        cur = ggml_add(ctx0, residual, attn);

        // ---- FFN ----
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor * gate = ggml_mul_mat(ctx0, b.ffn_gate_w, x);
        ggml_tensor * up   = ggml_mul_mat(ctx0, b.ffn_up_w,   x);
        ggml_tensor * mlp  = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
        mlp = ggml_mul_mat(ctx0, b.ffn_down_w, mlp);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    (void)d;
}

static ggml_cgraph * qwen3_asr_build_graph_llm(qwen3_asr_context * ctx,
                                               int n_tokens) {
    const auto & m  = ctx->model;
    const int T     = n_tokens;

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

    ggml_tensor * causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    // Token embedding lookup → (d, T)
    ggml_tensor * cur = ggml_get_rows(ctx0, m.llm.token_embd_w, input_ids);
    qwen3_asr_build_llm_body(ctx, ctx0, gf, cur, positions, causal_mask, T);
    ggml_free(ctx0);
    return gf;
}

// Variant: takes pre-computed inputs_embeds (d, T) F32 instead of input_ids.
// Used by the audio-injection path after splicing audio frames into the
// text-token embedding sequence.
static ggml_cgraph * qwen3_asr_build_graph_llm_from_embeds(qwen3_asr_context * ctx,
                                                           int n_tokens) {
    const auto & hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int T = n_tokens;

    ggml_init_params ip = {
        /*mem_size=*/   ctx->compute_meta.size(),
        /*mem_buffer=*/ ctx->compute_meta.data(),
        /*no_alloc=*/   true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor * embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor * causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    qwen3_asr_build_llm_body(ctx, ctx0, gf, embeds, positions, causal_mask, T);
    ggml_free(ctx0);
    return gf;
}

// Graph builder for the KV-cached LLM forward. Used by both prefill
// (n_past=0, n_tokens=T_prompt) and incremental decode (n_past>0, n_tokens=1).
//
// Inputs:
//   inputs_embeds: F32 (d, n_tokens)
//   positions:     I32 (n_tokens,) — absolute positions n_past, n_past+1, ...
//   causal_mask:   F32 (n_kv_total, n_tokens) where n_kv_total = n_past+n_tokens
//                  mask[k, q] = 0 if k <= n_past+q else -inf
//
// Per layer, the new K/V are written into the persistent cache at positions
// [n_past, n_past+n_tokens) and attention reads from [0, n_past+n_tokens).
static ggml_cgraph * qwen3_asr_build_graph_llm_kv(qwen3_asr_context * ctx,
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
    const int T        = n_tokens;
    const int Lk       = n_past + T;          // total cache length after this call

    GGML_ASSERT(ctx->kv_k && ctx->kv_v);
    GGML_ASSERT(Lk <= ctx->kv_max_ctx);

    ggml_init_params ip = {
        /*mem_size=*/   ctx->compute_meta.size(),
        /*mem_buffer=*/ ctx->compute_meta.data(),
        /*no_alloc=*/   true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor * embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor * causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, Lk, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

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

        Q = ggml_rms_norm(ctx0, Q, eps);
        Q = ggml_mul(ctx0, Q, b.attn_q_norm_w);
        K = ggml_rms_norm(ctx0, K, eps);
        K = ggml_mul(ctx0, K, b.attn_k_norm_w);

        Q = ggml_rope_ext(ctx0, Q, positions, nullptr,
                          hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos,
                          theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr,
                          hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos,
                          theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // ---- Permute new K/V to (hd, T, n_kv) before writing into cache ----
        // Cache layout: ne=(hd, max_ctx, n_kv, n_layers) — same as cohere.
        ggml_tensor * K_new_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);  // (hd, T, n_kv)
        ggml_tensor * V_new_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);

        // ---- Write into the persistent KV cache ----
        ggml_tensor * k_view = ggml_view_4d(ctx0, ctx->kv_k,
            hd, T, n_kv, 1,
            ctx->kv_k->nb[1], ctx->kv_k->nb[2], ctx->kv_k->nb[3],
            il * ctx->kv_k->nb[3] + n_past * ctx->kv_k->nb[1]);
        ggml_tensor * v_view = ggml_view_4d(ctx0, ctx->kv_v,
            hd, T, n_kv, 1,
            ctx->kv_v->nb[1], ctx->kv_v->nb[2], ctx->kv_v->nb[3],
            il * ctx->kv_v->nb[3] + n_past * ctx->kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_new_perm, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_new_perm, v_view));

        // ---- Read full K/V history from cache: (hd, Lk, n_kv) ----
        ggml_tensor * Kfull = ggml_view_3d(ctx0, ctx->kv_k,
            hd, Lk, n_kv,
            ctx->kv_k->nb[1], ctx->kv_k->nb[2],
            il * ctx->kv_k->nb[3]);
        ggml_tensor * Vfull = ggml_view_3d(ctx0, ctx->kv_v,
            hd, Lk, n_kv,
            ctx->kv_v->nb[1], ctx->kv_v->nb[2],
            il * ctx->kv_v->nb[3]);
        Kfull = ggml_cont(ctx0, Kfull);
        Vfull = ggml_cont(ctx0, Vfull);

        // ---- GQA expand cached K/V from n_kv to n_q heads ----
        if (n_kv_grp > 1) {
            ggml_tensor * K4 = ggml_reshape_4d(ctx0, Kfull, hd, Lk, 1, n_kv);
            ggml_tensor * V4 = ggml_reshape_4d(ctx0, Vfull, hd, Lk, 1, n_kv);
            // We want (hd, Lk, n_q) where heads (0,1) = kv head 0, (2,3) = kv head 1, etc.
            // Insert singleton at axis 2, repeat by n_kv_grp, reshape.
            K4 = ggml_repeat_4d(ctx0, K4, hd, Lk, n_kv_grp, n_kv);
            V4 = ggml_repeat_4d(ctx0, V4, hd, Lk, n_kv_grp, n_kv);
            Kfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, Lk, n_q));
            Vfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, Lk, n_q));
        }

        // ---- Permute Q to (hd, T, n_q) for attention ----
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

        // ---- scores = Kfull^T @ Q ----  ne=(Lk, T, n_q)
        ggml_tensor * scores = ggml_mul_mat(ctx0, Kfull, Q);
        scores = ggml_add(ctx0, scores, causal_mask);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, attn_scale, 0.0f);

        // ---- attn = scores @ Vfull ----
        // Vfull ne=(hd, Lk, n_q). Permute to (Lk, hd, n_q) for the dot.
        ggml_tensor * V2 = ggml_cont(ctx0, ggml_permute(ctx0, Vfull, 1, 0, 2, 3));
        ggml_tensor * attn = ggml_mul_mat(ctx0, V2, scores);
        // attn ne=(hd, T, n_q) → permute back → reshape (hd*n_q, T)
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

        attn = ggml_mul_mat(ctx0, b.attn_output_w, attn);
        cur = ggml_add(ctx0, residual, attn);

        // ---- FFN ----
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor * gate = ggml_mul_mat(ctx0, b.ffn_gate_w, x);
        ggml_tensor * up   = ggml_mul_mat(ctx0, b.ffn_up_w,   x);
        ggml_tensor * mlp  = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
        mlp = ggml_mul_mat(ctx0, b.ffn_down_w, mlp);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);

    // Last-token-only lm_head: slice (d, T) → (d, 1) before the big matmul.
    // We only ever need the next-token logits, never historical ones, so the
    // (151936, 2048) lm_head matmul is run on a 1-column input instead of T.
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);
    // logits ne = (vocab, 1)

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Tiny standalone graph for token embedding lookup (used by run_embed_tokens).
static ggml_cgraph * qwen3_asr_build_graph_embed(qwen3_asr_context * ctx, int n_tokens) {
    ggml_init_params ip = {
        /*mem_size=*/   ctx->compute_meta.size(),
        /*mem_buffer=*/ ctx->compute_meta.data(),
        /*no_alloc=*/   true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);
    ggml_tensor * out = ggml_get_rows(ctx0, ctx->model.llm.token_embd_w, ids);
    ggml_set_name(out, "embeds");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" const char * qwen3_asr_token_text(qwen3_asr_context * ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->vocab.id_to_token.size()) return "";
    return ctx->vocab.id_to_token[id].c_str();
}

extern "C" qwen3_asr_context_params qwen3_asr_context_default_params(void) {
    qwen3_asr_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    return p;
}

extern "C" qwen3_asr_context * qwen3_asr_init_from_file(const char * path,
                                                        qwen3_asr_context_params params) {
    qwen3_asr_context * ctx = new qwen3_asr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend_cpu = ggml_backend_cpu_init();
    ctx->backend     = ctx->backend_cpu;
    if (ctx->backend_cpu) {
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    }

    if (!qwen3_asr_load_model(ctx->model, ctx->vocab, path, ctx->backend)) {
        delete ctx;
        return nullptr;
    }
    if (params.verbosity >= 1) {
        fprintf(stderr, "qwen3_asr: loaded %s  (audio %u layers, llm %u layers, vocab %u)\n",
                path, ctx->model.hparams.audio_n_layers, ctx->model.hparams.llm_n_layers,
                (uint32_t)ctx->vocab.id_to_token.size());
    }
    return ctx;
}

extern "C" void qwen3_asr_free(qwen3_asr_context * ctx) {
    if (!ctx) return;
    if (ctx->sched)       ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)      ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)      ggml_free(ctx->kv_ctx);
    if (ctx->model.buf)   ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)   ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" char * qwen3_asr_transcribe(qwen3_asr_context * /*ctx*/,
                                       const float * /*samples*/, int /*n_samples*/) {
    // Stage 1: not yet implemented end-to-end. Use qwen3_asr_run_conv for now.
    return strdup("");
}

extern "C" float * qwen3_asr_run_conv(qwen3_asr_context * ctx,
                                      const float * mel_features,
                                      int n_mels, int T_mel,
                                      int * out_n_chunks,
                                      int * out_T_chunk_out,
                                      int * out_d) {
    if (!ctx || !mel_features) return nullptr;
    const auto & hp = ctx->model.hparams;
    if (n_mels != (int)hp.n_mels) {
        fprintf(stderr, "qwen3_asr: mel feature mismatch (%d vs %d)\n", n_mels, (int)hp.n_mels);
        return nullptr;
    }

    // Chunking: split T_mel into chunks of n_window*2. The final chunk is
    // padded with zeros to n_window*2 to match the reference impl, which
    // pad_sequences chunks before batching them through the convs.
    const int chunk_T   = (int)hp.n_window * 2;          // 100
    const int num_chunks = (T_mel + chunk_T - 1) / chunk_T;

    // Build (T_chunk=100, n_mels=128, 1, num_chunks) F32 buffer, padded with zeros.
    std::vector<float> mel_padded((size_t)chunk_T * n_mels * num_chunks, 0.0f);
    // Source layout: mel_features is (n_mels, T_mel), row-major (mel as outer,
    // time as inner). Per the reference dump it's saved that way.
    // Target ggml layout: ne[0]=T_chunk varies fastest, ne[1]=n_mels, ne[3]=batch.
    // Memory index: t + chunk_T*(f + n_mels*chunk_idx)
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int t_start = chunk * chunk_T;
        const int t_end   = std::min(t_start + chunk_T, T_mel);
        const int t_len   = t_end - t_start;
        for (int f = 0; f < n_mels; f++) {
            for (int t = 0; t < t_len; t++) {
                // src: (f, t_start + t) — mel_features[f * T_mel + (t_start + t)]
                // dst: (t, f, 0, chunk) — mel_padded[t + chunk_T*(f + n_mels*chunk)]
                mel_padded[(size_t)t + chunk_T*((size_t)f + n_mels*(size_t)chunk)]
                    = mel_features[(size_t)f * T_mel + (size_t)(t_start + t)];
            }
        }
    }

    // Lazy sched + compute_meta init
    if (!ctx->sched) {
        ggml_backend_t backends[2] = { ctx->backend, ctx->backend_cpu };
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 4096, false, false);
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(
            ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false));
    }

    ggml_cgraph * gf = qwen3_asr_build_graph_conv(ctx, chunk_T, num_chunks);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc conv graph\n");
        return nullptr;
    }

    ggml_tensor * mel_in = ggml_graph_get_tensor(gf, "mel_batched");
    ggml_backend_tensor_set(mel_in, mel_padded.data(), 0,
                            mel_padded.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: conv graph compute failed\n");
        return nullptr;
    }

    ggml_tensor * out = ggml_graph_get_tensor(gf, "conv_front_out");
    if (!out) {
        fprintf(stderr, "qwen3_asr: missing conv_front_out tensor\n");
        return nullptr;
    }
    const int d   = (int)out->ne[0];      // 896
    const int T   = (int)out->ne[1];      // 13
    const int B   = (int)out->ne[2];      // num_chunks
    if (out_n_chunks)    *out_n_chunks    = B;
    if (out_T_chunk_out) *out_T_chunk_out = T;
    if (out_d)           *out_d           = d;

    const size_t total = (size_t)d * T * B;
    float * result = (float *)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" float * qwen3_asr_run_encoder(qwen3_asr_context * ctx,
                                         const float * mel_features,
                                         int n_mels, int T_mel,
                                         int * out_N_total,
                                         int * out_proj_dim) {
    if (!ctx || !mel_features) return nullptr;
    const auto & hp = ctx->model.hparams;
    if (n_mels != (int)hp.n_mels) {
        fprintf(stderr, "qwen3_asr: mel feature mismatch (%d vs %d)\n", n_mels, (int)hp.n_mels);
        return nullptr;
    }

    // Chunking (same as Stage 1).
    const int chunk_T    = (int)hp.n_window * 2;
    const int num_chunks = (T_mel + chunk_T - 1) / chunk_T;
    // Stage 2 simplification: assume the input is an exact multiple of chunk_T,
    // so all chunks are full and there's no per-chunk padding to mask out. The
    // jfk fixture (1100 mel frames, 11 chunks of 100) satisfies this.
    if (T_mel % chunk_T != 0) {
        fprintf(stderr, "qwen3_asr: stage-2 requires T_mel %% %d == 0 (got %d)\n",
                chunk_T, T_mel);
        return nullptr;
    }

    // After three stride-2 convs the time dim shrinks by 8 (with rounding).
    // Reference: 100 → 50 → 25 → 13.
    auto conv_out_len = [](int in_len) {
        // (in + 2*pad - k)/stride + 1, with pad=1, k=3, stride=2
        return (in_len + 2 - 3) / 2 + 1;
    };
    const int T_chunk_out = conv_out_len(conv_out_len(conv_out_len(chunk_T)));
    const int N_padded = T_chunk_out * num_chunks;

    // Pack mel into the (T_chunk, n_mels, 1, num_chunks) ggml layout.
    std::vector<float> mel_padded((size_t)chunk_T * n_mels * num_chunks, 0.0f);
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int t_start = chunk * chunk_T;
        for (int f = 0; f < n_mels; f++) {
            for (int t = 0; t < chunk_T; t++) {
                mel_padded[(size_t)t + chunk_T*((size_t)f + n_mels*(size_t)chunk)]
                    = mel_features[(size_t)f * T_mel + (size_t)(t_start + t)];
            }
        }
    }

    // The reference's eager_attention_forward IGNORES cu_seqlens and uses
    // standard full self-attention with attention_mask=None. cu_seqlens is
    // only consumed by FlashAttention2 on GPU. So on CPU we just need a
    // zero mask. (We keep the input tensor in the graph so the structure
    // is ready when we add real per-chunk padding masking later.)
    std::vector<float> mask((size_t)N_padded * N_padded, 0.0f);

    // Lazy sched + compute_meta init (encoder graph is much bigger than conv-only,
    // bump the meta budget for the 18 layers).
    if (ctx->sched) {
        ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }
    ctx->compute_meta.assign(
        ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), 0);
    ggml_backend_t backends[2] = { ctx->backend, ctx->backend_cpu };
    int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);

    ggml_cgraph * gf = qwen3_asr_build_graph_encoder(ctx, chunk_T, num_chunks, T_chunk_out);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc encoder graph\n");
        return nullptr;
    }

    // Set inputs
    ggml_tensor * mel_in = ggml_graph_get_tensor(gf, "mel_batched");
    ggml_backend_tensor_set(mel_in, mel_padded.data(), 0,
                            mel_padded.size() * sizeof(float));

    // pe_input ne=(d, T_chunk_out). Pull rows [0, T_chunk_out) from model.audio_pe.
    ggml_tensor * pe_in = ggml_graph_get_tensor(gf, "pe_input");
    {
        const int d = (int)hp.audio_d_model;
        std::vector<float> pe_buf((size_t)d * T_chunk_out);
        // model.audio_pe row p starts at offset p*d. We need to write into ggml
        // ne=(d, T_chunk_out) which has d as ne[0] (fast). Memory layout matches
        // a row-major (T_chunk_out, d) buffer. So just copy [0, T_chunk_out*d).
        std::memcpy(pe_buf.data(), ctx->model.audio_pe.data(),
                    pe_buf.size() * sizeof(float));
        ggml_backend_tensor_set(pe_in, pe_buf.data(), 0,
                                pe_buf.size() * sizeof(float));
    }

    ggml_tensor * mask_in = ggml_graph_get_tensor(gf, "attn_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: encoder graph compute failed\n");
        return nullptr;
    }

    ggml_tensor * out = ggml_graph_get_tensor(gf, "encoder_out");
    if (!out) { fprintf(stderr, "qwen3_asr: missing encoder_out tensor\n"); return nullptr; }
    const int pdim = (int)out->ne[0];   // 1024
    const int N    = (int)out->ne[1];   // N_padded
    if (out_N_total) *out_N_total = N;
    if (out_proj_dim) *out_proj_dim = pdim;

    const size_t total = (size_t)pdim * N;
    float * result = (float *)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" bool qwen3_asr_kv_init(qwen3_asr_context * ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0) return false;
    if (ctx->kv_k) return true;  // already initialized

    const auto & hp = ctx->model.hparams;
    const int hd     = (int)hp.llm_head_dim;
    const int n_kv   = (int)hp.llm_n_kv_heads;
    const int n_lay  = (int)hp.llm_n_layers;

    ggml_init_params kp = {
        /*mem_size=*/   ggml_tensor_overhead() * 4 + 1024,
        /*mem_buffer=*/ nullptr,
        /*no_alloc=*/   true,
    };
    ctx->kv_ctx = ggml_init(kp);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F32, hd, max_ctx, n_kv, n_lay);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F32, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    const size_t kbytes = ggml_nbytes(ctx->kv_k);
    const size_t vbytes = ggml_nbytes(ctx->kv_v);
    ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, kbytes + vbytes);
    if (!ctx->kv_buf) {
        fprintf(stderr, "qwen3_asr: failed to allocate kv buffer\n");
        return false;
    }
    char * base = (char *)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);
    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used  = 0;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "qwen3_asr: kv cache %d MiB (head_dim=%d max_ctx=%d n_kv=%d n_layers=%d)\n",
                (int)((kbytes + vbytes) / 1048576), hd, max_ctx, n_kv, n_lay);
    }
    return true;
}

extern "C" void qwen3_asr_kv_reset(qwen3_asr_context * ctx) {
    if (ctx) ctx->kv_n_used = 0;
}

extern "C" float * qwen3_asr_run_llm_kv(qwen3_asr_context * ctx,
                                        const float * inputs_embeds,
                                        int n_tokens, int n_past,
                                        int * out_n_tokens, int * out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0) return nullptr;
    if (!ctx->kv_k) {
        fprintf(stderr, "qwen3_asr: kv cache not initialized — call qwen3_asr_kv_init first\n");
        return nullptr;
    }
    if (n_past + n_tokens > ctx->kv_max_ctx) {
        fprintf(stderr, "qwen3_asr: kv overflow (n_past=%d + n_tokens=%d > max_ctx=%d)\n",
                n_past, n_tokens, ctx->kv_max_ctx);
        return nullptr;
    }
    const auto & hp = ctx->model.hparams;
    const int d     = (int)hp.llm_d_model;
    const int vocab = (int)hp.llm_vocab_size;
    const int Lk    = n_past + n_tokens;

    // Positions [n_past, n_past+T)
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) positions[i] = n_past + i;

    // Causal mask (Lk, T): mask[k, q] = 0 if k <= n_past+q else -inf.
    // ggml ne[0]=k (key, fast), ne[1]=q (query). Memory layout: q outer, k inner.
    // Index = q*Lk + k.
    std::vector<float> mask((size_t)Lk * n_tokens, 0.0f);
    for (int q = 0; q < n_tokens; q++) {
        for (int k = n_past + q + 1; k < Lk; k++) {
            mask[(size_t)q * Lk + k] = -INFINITY;
        }
    }

    // Reset sched + compute_meta with bigger budget for KV graph
    if (ctx->sched) { ggml_backend_sched_free(ctx->sched); ctx->sched = nullptr; }
    ctx->compute_meta.assign(
        ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), 0);
    ggml_backend_t backends[2] = { ctx->backend, ctx->backend_cpu };
    int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);

    ggml_cgraph * gf = qwen3_asr_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc llm_kv graph\n"); return nullptr;
    }

    ggml_tensor * embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor * pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    ggml_tensor * mask_in = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: llm_kv graph compute failed\n"); return nullptr;
    }

    ggml_tensor * out = ggml_graph_get_tensor(gf, "logits");
    if (!out) return nullptr;
    ctx->kv_n_used = n_past + n_tokens;
    // Output is only the last token's logits — see lm_head slice in build_graph_llm_kv.
    if (out_n_tokens)   *out_n_tokens   = 1;
    if (out_vocab_size) *out_vocab_size = vocab;
    float * result = (float *)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)vocab * sizeof(float));
    return result;
}

extern "C" float * qwen3_asr_embed_tokens(qwen3_asr_context * ctx,
                                          const int32_t * input_ids, int n_tokens) {
    if (!ctx || !input_ids || n_tokens <= 0) return nullptr;
    const int d = (int)ctx->model.hparams.llm_d_model;

    if (ctx->sched) {
        ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }
    ctx->compute_meta.assign(
        ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), 0);
    ggml_backend_t backends[2] = { ctx->backend, ctx->backend_cpu };
    int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 64, false, false);

    ggml_cgraph * gf = qwen3_asr_build_graph_embed(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc embed graph\n"); return nullptr;
    }
    ggml_tensor * ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, input_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: embed graph compute failed\n"); return nullptr;
    }
    ggml_tensor * out = ggml_graph_get_tensor(gf, "embeds");
    const size_t total = (size_t)d * n_tokens;
    float * result = (float *)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" float * qwen3_asr_run_llm_from_embeds(qwen3_asr_context * ctx,
                                                 const float * inputs_embeds,
                                                 int n_tokens,
                                                 int * out_n_tokens,
                                                 int * out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0) return nullptr;
    const auto & hp = ctx->model.hparams;
    const int d     = (int)hp.llm_d_model;
    const int vocab = (int)hp.llm_vocab_size;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) positions[i] = i;
    std::vector<float> mask((size_t)n_tokens * n_tokens, 0.0f);
    for (int i = 0; i < n_tokens; i++)
        for (int j = i + 1; j < n_tokens; j++)
            mask[(size_t)i * n_tokens + j] = -INFINITY;

    if (ctx->sched) { ggml_backend_sched_free(ctx->sched); ctx->sched = nullptr; }
    ctx->compute_meta.assign(
        ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), 0);
    ggml_backend_t backends[2] = { ctx->backend, ctx->backend_cpu };
    int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);

    ggml_cgraph * gf = qwen3_asr_build_graph_llm_from_embeds(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc llm-from-embeds graph\n");
        return nullptr;
    }

    ggml_tensor * embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_tensor * pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    ggml_tensor * mask_in = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: llm-from-embeds graph compute failed\n");
        return nullptr;
    }

    ggml_tensor * out = ggml_graph_get_tensor(gf, "logits");
    if (!out) { fprintf(stderr, "missing logits\n"); return nullptr; }
    if (out_n_tokens)   *out_n_tokens   = n_tokens;
    if (out_vocab_size) *out_vocab_size = vocab;
    const size_t total = (size_t)vocab * n_tokens;
    float * result = (float *)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" float * qwen3_asr_run_llm(qwen3_asr_context * ctx,
                                     const int32_t * input_ids,
                                     int n_tokens,
                                     int * out_n_tokens,
                                     int * out_vocab_size) {
    if (!ctx || !input_ids || n_tokens <= 0) return nullptr;
    const auto & hp = ctx->model.hparams;
    const int vocab = (int)hp.llm_vocab_size;

    // Build positions = [0, 1, ..., T-1]
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) positions[i] = i;

    // Build causal mask: (T, T) F32. mask[i, j] = 0 if j <= i else -inf.
    // ggml ne[0]=j (key, fast), ne[1]=i (query). Disallowed → -inf.
    std::vector<float> mask((size_t)n_tokens * n_tokens, 0.0f);
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < n_tokens; j++) {
            if (j > i) mask[(size_t)i * n_tokens + j] = -INFINITY;
        }
    }

    // Lazy sched + compute_meta init (LLM graph is bigger than encoder)
    if (ctx->sched) {
        ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }
    ctx->compute_meta.assign(
        ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), 0);
    ggml_backend_t backends[2] = { ctx->backend, ctx->backend_cpu };
    int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);

    ggml_cgraph * gf = qwen3_asr_build_graph_llm(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc llm graph\n");
        return nullptr;
    }

    ggml_tensor * ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, input_ids, 0, (size_t)n_tokens * sizeof(int32_t));

    ggml_tensor * pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));

    ggml_tensor * mask_in = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: llm graph compute failed\n");
        return nullptr;
    }

    ggml_tensor * out = ggml_graph_get_tensor(gf, "logits");
    if (!out) { fprintf(stderr, "qwen3_asr: missing logits tensor\n"); return nullptr; }
    if (out_n_tokens)    *out_n_tokens    = n_tokens;
    if (out_vocab_size) *out_vocab_size  = vocab;

    const size_t total = (size_t)vocab * n_tokens;
    float * result = (float *)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}
