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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct qwen3_asr_hparams {
    // Audio
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 400;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;
    uint32_t audio_n_layers = 18;
    uint32_t audio_d_model = 896;
    uint32_t audio_n_heads = 14;
    uint32_t audio_head_dim = 64;
    uint32_t audio_ff_dim = 3584;
    uint32_t audio_conv_ch = 480;
    uint32_t audio_proj_dim = 1024;
    uint32_t audio_max_pos = 1500;

    // Chunking parameters (from reference impl: n_window=50, n_window_infer=800)
    uint32_t n_window = 50;
    uint32_t n_window_infer = 800;

    // LLM (Qwen3 0.6B)
    uint32_t llm_n_layers = 28;
    uint32_t llm_d_model = 1024;
    uint32_t llm_n_heads = 16;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 3072;
    float llm_rope_theta = 1e6f;
    float llm_rms_eps = 1e-6f;
    uint32_t llm_vocab_size = 151936;
    uint32_t llm_max_pos = 65536;

    // The LM head's actual output dimension. For the standard ASR
    // models this equals llm_vocab_size (152064 for 1.7B, 151936 for
    // 0.6B). For the Qwen3-ForcedAligner-0.6B variant the head
    // outputs 5000 timestamp classes instead — we read the real
    // value from the loaded `output.weight` tensor's shape rather
    // than asserting equality with vocab_size.
    uint32_t llm_lm_head_dim = 0; // 0 = "use vocab_size" sentinel

    // Special tokens
    uint32_t audio_start_token_id = 151669;
    uint32_t audio_end_token_id = 151670;
    uint32_t audio_pad_token_id = 151676;
    uint32_t eos_token_id = 151645;
    uint32_t pad_token_id = 151643;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct qwen3_asr_audio_block {
    // Pre-LN self-attention
    ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    // Pre-LN FFN (GELU)
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;
    ggml_tensor *ffn_down_w = nullptr, *ffn_down_b = nullptr;
};

struct qwen3_asr_audio_tower {
    // Conv subsampler front-end (4 stride-2 freq convs as 2D over the mel image)
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;       // (480, 1,   3, 3)
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;       // (480, 480, 3, 3)
    ggml_tensor *conv3_w = nullptr, *conv3_b = nullptr;       // (480, 480, 3, 3)
    ggml_tensor *conv_out_w = nullptr, *conv_out_b = nullptr; // (896, 7680)

    // Encoder body
    std::vector<qwen3_asr_audio_block> blocks;

    // Final norm + projector head (896 → 896 → GELU → 1024)
    ggml_tensor *ln_post_w = nullptr, *ln_post_b = nullptr;
    ggml_tensor *proj1_w = nullptr, *proj1_b = nullptr;
    ggml_tensor *proj2_w = nullptr, *proj2_b = nullptr;

    // Mel preprocessor (baked from WhisperFeatureExtractor by the converter)
    ggml_tensor* mel_filters = nullptr; // (n_freqs=201, n_mels=128) F32
    ggml_tensor* mel_window = nullptr;  // (400,) F32 hann window
};

struct qwen3_asr_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr; // Qwen3 per-head Q RMSNorm
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct qwen3_asr_llm {
    ggml_tensor* token_embd_w = nullptr; // (151936, 1024)
    std::vector<qwen3_asr_llm_block> blocks;
    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* output_w = nullptr;
};

struct qwen3_asr_model {
    qwen3_asr_hparams hparams;
    qwen3_asr_audio_tower audio;
    qwen3_asr_llm llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Sinusoidal positional embedding for the audio encoder, computed once
    // at load time. Layout: row-major (max_pos, d_model) where row p is the
    // pos embed for position p.
    std::vector<float> audio_pe; // size = audio_max_pos * audio_d_model
};

struct qwen3_asr_vocab {
    std::vector<std::string> id_to_token;

    // Reverse lookup: byte-encoded vocab string → token id (for BPE encode).
    std::unordered_map<std::string, int32_t> token_to_id;

    // BPE merges loaded from the GGUF. Each merge is a (left, right) pair
    // of byte-encoded strings; the rank is the index in the list (lower = earlier
    // merge = higher priority, matching GPT-2 / Qwen2 BPE convention).
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" → rank
};

struct qwen3_asr_context {
    qwen3_asr_context_params params;

    qwen3_asr_model model;
    qwen3_asr_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;

    // KV cache (Stage 5). Single tensor for K, single for V, both shape
    // (head_dim, max_ctx, n_kv_heads, n_layers). Allocated to backend.
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;

    int n_threads = 4;
};

// ===========================================================================
// Loader helpers
// ===========================================================================

#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static ggml_tensor* try_get(qwen3_asr_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(qwen3_asr_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "qwen3_asr");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool qwen3_asr_load_model(qwen3_asr_model& model, qwen3_asr_vocab& vocab, const char* path,
                                 ggml_backend_t backend) {
    // ---- pass 1: read hparams + vocab via metadata-only context ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "qwen3asr.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "qwen3asr.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "qwen3asr.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "qwen3asr.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "qwen3asr.hop_length", hp.hop_length);
        hp.audio_n_layers = core_gguf::kv_u32(gctx, "qwen3asr.audio.n_layers", hp.audio_n_layers);
        hp.audio_d_model = core_gguf::kv_u32(gctx, "qwen3asr.audio.d_model", hp.audio_d_model);
        hp.audio_n_heads = core_gguf::kv_u32(gctx, "qwen3asr.audio.n_heads", hp.audio_n_heads);
        hp.audio_head_dim = core_gguf::kv_u32(gctx, "qwen3asr.audio.head_dim", hp.audio_head_dim);
        hp.audio_ff_dim = core_gguf::kv_u32(gctx, "qwen3asr.audio.ff_dim", hp.audio_ff_dim);
        hp.audio_conv_ch = core_gguf::kv_u32(gctx, "qwen3asr.audio.conv_channels", hp.audio_conv_ch);
        hp.audio_proj_dim = core_gguf::kv_u32(gctx, "qwen3asr.audio.proj_dim", hp.audio_proj_dim);
        hp.audio_max_pos = core_gguf::kv_u32(gctx, "qwen3asr.audio.max_source_pos", hp.audio_max_pos);

        hp.llm_n_layers = core_gguf::kv_u32(gctx, "qwen3asr.llm.n_layers", hp.llm_n_layers);
        hp.llm_d_model = core_gguf::kv_u32(gctx, "qwen3asr.llm.d_model", hp.llm_d_model);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "qwen3asr.llm.n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "qwen3asr.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "qwen3asr.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "qwen3asr.llm.ff_dim", hp.llm_ff_dim);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "qwen3asr.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "qwen3asr.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "qwen3asr.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "qwen3asr.llm.max_pos", hp.llm_max_pos);

        hp.audio_start_token_id = core_gguf::kv_u32(gctx, "qwen3asr.audio_start_token_id", hp.audio_start_token_id);
        hp.audio_end_token_id = core_gguf::kv_u32(gctx, "qwen3asr.audio_end_token_id", hp.audio_end_token_id);
        hp.audio_pad_token_id = core_gguf::kv_u32(gctx, "qwen3asr.audio_pad_token_id", hp.audio_pad_token_id);
        hp.eos_token_id = core_gguf::kv_u32(gctx, "qwen3asr.eos_token_id", hp.eos_token_id);
        hp.pad_token_id = core_gguf::kv_u32(gctx, "qwen3asr.pad_token_id", hp.pad_token_id);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            vocab.token_to_id.reserve(vocab.id_to_token.size());
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        // Register Qwen2/Qwen3 special tokens explicitly. The original
        // vocab.json that the converter pulls strings from only has 151 643
        // regular tokens; the special tokens (<|im_start|>, <|audio_pad|>,
        // ...) live in tokenizer_config.json's added_tokens list, which the
        // converter currently doesn't propagate, so they end up as
        // "[PAD151644]" etc. in vocab.id_to_token. Patch them in here so
        // qwen3_asr_tokenize and qwen3_asr_token_text can find them.
        struct SpecialTok {
            int id;
            const char* text;
        };
        static const SpecialTok specials[] = {
            {151643, "<|endoftext|>"},        {151644, "<|im_start|>"},       {151645, "<|im_end|>"},
            {151646, "<|object_ref_start|>"}, {151647, "<|object_ref_end|>"}, {151648, "<|box_start|>"},
            {151649, "<|box_end|>"},          {151650, "<|quad_start|>"},     {151651, "<|quad_end|>"},
            {151652, "<|vision_start|>"},     {151653, "<|vision_end|>"},     {151654, "<|vision_pad|>"},
            {151655, "<|image_pad|>"},        {151656, "<|video_pad|>"},      {151669, "<|audio_start|>"},
            {151670, "<|audio_end|>"},        {151676, "<|audio_pad|>"},
        };
        for (const auto& sp : specials) {
            if (sp.id < (int)vocab.id_to_token.size()) {
                // Drop the old [PAD<id>] reverse-map entry if present
                auto old_it = vocab.token_to_id.find(vocab.id_to_token[sp.id]);
                if (old_it != vocab.token_to_id.end() && old_it->second == sp.id) {
                    vocab.token_to_id.erase(old_it);
                }
                vocab.id_to_token[sp.id] = sp.text;
                vocab.token_to_id[sp.text] = sp.id;
            }
        }
        // Merges (BPE encode side). Each entry is a "left right" pair string;
        // the ARRAY index is the merge's rank (lowest rank = highest priority).
        auto merges = core_gguf::kv_str_array(gctx, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++) {
            vocab.merge_rank[merges[i]] = i;
        }

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: tensor data via shared helper ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "qwen3_asr", wl)) {
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
    a.conv3_w = require(model, "audio.conv.3.weight");
    a.conv3_b = require(model, "audio.conv.3.bias");
    a.conv_out_w = require(model, "audio.conv_out.weight");
    a.conv_out_b = try_get(model, "audio.conv_out.bias"); // bias may be absent
    a.ln_post_w = require(model, "audio.ln_post.weight");
    a.ln_post_b = require(model, "audio.ln_post.bias");
    a.proj1_w = require(model, "audio.proj1.weight");
    a.proj1_b = require(model, "audio.proj1.bias");
    a.proj2_w = require(model, "audio.proj2.weight");
    a.proj2_b = require(model, "audio.proj2.bias");
    a.mel_filters = try_get(model, "audio.mel_filters"); // optional (may be missing in older GGUFs)
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
        b.attn_k_b = get("attn_k.bias");
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

    auto& l = model.llm;
    l.token_embd_w = require(model, "token_embd.weight");
    l.output_norm_w = require(model, "output_norm.weight");
    l.output_w = require(model, "output.weight");
    // Read the actual lm_head output dimension from the loaded tensor
    // shape rather than asserting it equals vocab_size. The standard
    // Qwen3-ASR-{0.6B,1.7B} models have lm_head = (vocab, d), but the
    // Qwen3-ForcedAligner variant has lm_head = (5000, d) — same body,
    // different head. ne[1] is the row count after ggml's [in, out]
    // storage convention.
    model.hparams.llm_lm_head_dim = (uint32_t)l.output_w->ne[1];
    if (model.hparams.llm_lm_head_dim == 0) {
        model.hparams.llm_lm_head_dim = model.hparams.llm_vocab_size;
    }
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
        b.attn_q_norm_w = get("attn_q_norm.weight");
        b.attn_k_norm_w = get("attn_k_norm.weight");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn_gate.weight");
        b.ffn_up_w = get("ffn_up.weight");
        b.ffn_down_w = get("ffn_down.weight");
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
        for (int i = 0; i < half; i++)
            inv_t[i] = std::exp(-log_inc * (float)i);
        model.audio_pe.assign((size_t)L * C, 0.0f);
        for (int p = 0; p < L; p++) {
            float* row = model.audio_pe.data() + (size_t)p * C;
            for (int i = 0; i < half; i++) {
                float angle = (float)p * inv_t[i];
                row[i] = std::sin(angle);
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

static void qwen3_asr_dft(const float* in, int N, float* out) {
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

// Real-input FFT, output complex (out has 2*N floats interleaved real/imag).
// in/out are scratch buffers; in must have at least 2*N floats of writable space.
static void qwen3_asr_fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        qwen3_asr_dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; i++)
        even[i] = in[2 * i];
    float* even_fft = out + 2 * N;
    qwen3_asr_fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; i++)
        odd[i] = in[2 * i + 1];
    float* odd_fft = even_fft + N;
    qwen3_asr_fft(odd, half_N, odd_fft);

    for (int k = 0; k < half_N; k++) {
        float ang = -2.0f * (float)M_PI * (float)k / (float)N;
        float re = std::cos(ang);
        float im = std::sin(ang);
        float re_odd = odd_fft[2 * k];
        float im_odd = odd_fft[2 * k + 1];
        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + half_N)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
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

#include "core/mel.h"
#include "core/ffn.h"
#include "core/attention.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// qwen3_asr_fft uses its input buffer as scratch during recursion (needs
// ~4N extra floats past the input pointer). Wrap it to match core_mel's
// const-input FftR2C signature, same trick as voxtral / voxtral4b.
static void qwen3_asr_fft_wrapper(const float* in, int N, float* out) {
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size() < 4 * N)
        scratch_in.assign((size_t)4 * N, 0.0f);
    if ((int)scratch_out.size() < 8 * N)
        scratch_out.assign((size_t)8 * N, 0.0f);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    qwen3_asr_fft(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}

extern "C" float* qwen3_asr_compute_mel(qwen3_asr_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                        int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    if (!ctx->model.audio.mel_filters || !ctx->model.audio.mel_window) {
        fprintf(stderr, "qwen3_asr: model GGUF missing audio.mel_filters / audio.mel_window\n");
        return nullptr;
    }

    const int n_fft = (int)hp.n_fft;    // 400
    const int hop = (int)hp.hop_length; // 160
    const int n_mels = (int)hp.n_mels;  // 128
    const int n_freqs = n_fft / 2 + 1;  // 201

    std::vector<float> hann(n_fft);
    ggml_backend_tensor_get(ctx->model.audio.mel_window, hann.data(), 0, n_fft * sizeof(float));
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->model.audio.mel_filters, filt.data(), 0, filt.size() * sizeof(float));

    // Qwen3-ASR / Whisper HF feature extractor: log10 + max-clip guard,
    // double-accumulator matmul, drop last STFT frame, fb in (n_freqs, n_mels)
    // layout. No fixed-size padding — output T is whatever the audio yields.
    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Log10;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.norm = core_mel::Normalization::GlobalClipMax;
    p.layout = core_mel::Layout::MelsTime;
    p.fb_layout = core_mel::FbLayout::FreqsMels;
    p.matmul = core_mel::MatmulPrecision::Double;
    p.log_eps = 1e-10f;
    p.center_pad = true;
    p.drop_last_frame = true;

    int T_ret = 0;
    auto mel = core_mel::compute(samples, n_samples, hann.data(), n_fft, filt.data(), n_freqs, qwen3_asr_fft_wrapper, p,
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

static ggml_cgraph* qwen3_asr_build_graph_conv(qwen3_asr_context* ctx, int T_chunk, int num_chunks) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // Input: ggml 2D conv expects (W, H, C, N) where ne[0]=W (fast), ne[3]=N
    // For per-chunk processing of (1, n_mels=128, T_chunk=100):
    //   ne[0] = T_chunk  (time, varies fastest)
    //   ne[1] = n_mels   (frequency)
    //   ne[2] = 1        (in channels)
    //   ne[3] = num_chunks (batch)
    ggml_tensor* mel = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T_chunk, n_mels, 1, num_chunks);
    ggml_set_name(mel, "mel_batched");
    ggml_set_input(mel);

    auto bias_4d = [&](ggml_context* c0, ggml_tensor* b) {
        // bias is (out_ch,) — broadcast as (1, 1, out_ch, 1) for elementwise add
        return ggml_cast(c0, ggml_reshape_4d(c0, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    // Conv1: in=1, out=480, k=3, stride=2, pad=1
    ggml_tensor* cur = ggml_conv_2d(ctx0, m.audio.conv1_w, mel, 2, 2, 1, 1, 1, 1);
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
    const int C_out = (int)cur->ne[2]; // 480
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

static ggml_cgraph* qwen3_asr_build_graph_encoder(qwen3_asr_context* ctx, int T_chunk, int num_chunks,
                                                  int T_chunk_out_expected) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;
    const int d = (int)hp.audio_d_model;         // 896
    const int n_heads = (int)hp.audio_n_heads;   // 14
    const int head_dim = (int)hp.audio_head_dim; // 64
    const int proj_dim = (int)hp.audio_proj_dim; // 1024

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // ------- Inputs -------
    // mel_batched ne = (T_chunk, n_mels, 1, num_chunks)
    ggml_tensor* mel = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T_chunk, n_mels, 1, num_chunks);
    ggml_set_name(mel, "mel_batched");
    ggml_set_input(mel);

    // pe_input ne = (d, T_chunk_out, 1, 1)  — broadcasts over batch
    ggml_tensor* pe_in = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d, T_chunk_out_expected, 1);
    ggml_set_name(pe_in, "pe_input");
    ggml_set_input(pe_in);

    // attn_mask ne = (N_padded, N_padded)  F32 with 0 / -inf
    const int N_padded = T_chunk_out_expected * num_chunks;
    ggml_tensor* mask_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N_padded, N_padded);
    ggml_set_name(mask_in, "attn_mask");
    ggml_set_input(mask_in);

    // ------- Conv front-end (same as Stage 1) -------
    auto bias_4d = [&](ggml_context* c0, ggml_tensor* b) {
        return ggml_cast(c0, ggml_reshape_4d(c0, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    ggml_tensor* cur = ggml_conv_2d(ctx0, m.audio.conv1_w, mel, 2, 2, 1, 1, 1, 1);
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
    cur = ggml_mul_mat(ctx0, m.audio.conv_out_w, cur); // (d, T_out, num_chunks)

    // ------- Add positional embedding (broadcasts over batch) -------
    // pe_in ne = (d, T_out, 1) → broadcast against (d, T_out, num_chunks)
    cur = ggml_add(ctx0, cur, pe_in);

    // ------- Flatten chunks into a single sequence -------
    // cur ne = (d, T_out, num_chunks). Want (d, N_padded). Memory layout for
    // (d, T_out, num_chunks) row-major (d fastest) is identical to
    // (d, N_padded=T_out*num_chunks) where chunk-major order is preserved.
    // ggml_reshape_2d just relabels strides.
    cur = ggml_cont(ctx0, cur); // ensure contiguous before reshape
    cur = ggml_reshape_2d(ctx0, cur, d, N_padded);

    // ------- 18 × encoder blocks -------
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);
    for (uint32_t il = 0; il < hp.audio_n_layers; il++) {
        const auto& b = m.audio.blocks[il];
        ggml_tensor* residual = cur;

        // ---- LN1 (pre-attention) ----
        ggml_tensor* x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        x = ggml_add(ctx0, x, b.attn_norm_b);

        // ---- Q, K, V projections (with biases) ----
        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, x), b.attn_q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_k_w, x), b.attn_k_b);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, x), b.attn_v_b);
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
        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);

        // Add window mask. mask_in ne=(N, N) F32. ggml_add broadcasts over the
        // n_heads dim (size 1 in mask, size n_h in scores).
        scores = ggml_add(ctx0, scores, mask_in);

        // Softmax along key axis (ne[0]) with scale baked in.
        scores = ggml_soft_max_ext(ctx0, scores, /*mask*/ nullptr, attn_scale, 0.0f);

        // ---- attn = scores @ V ----
        // We need: out[d, i, h] = sum_j scores[j, i, h] * V[d, j, h]
        // ggml_mul_mat(V_perm, scores) where V_perm is (j, d, h) so dot is over j.
        // Currently V ne=(hd, N, n_h). We want V indexed as (j, d, h) with j fast.
        // Permute V (hd, N, n_h) → (N, hd, n_h): source 0→1, 1→0, 2→2
        ggml_tensor* V2 = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
        // ggml_mul_mat(V2, scores): V2 ne=(N, hd, n_h), scores ne=(N, N, n_h)
        // dot over ne[0]=N (the j axis). Result ne=(hd, N, n_h) where result[d, i, h]
        // = sum_j V2[j, d, h] * scores[j, i, h] = sum_j V[d, j, h] * scores[j, i, h] ✓
        ggml_tensor* attn = ggml_mul_mat(ctx0, V2, scores);
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
        ggml_tensor* x = ggml_norm(ctx0, cur, kLayerNormEps);
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
static void qwen3_asr_build_llm_body(qwen3_asr_context* ctx, ggml_context* ctx0, ggml_cgraph* gf,
                                     ggml_tensor* cur, // (d, T) input hidden state
                                     ggml_tensor* positions, ggml_tensor* causal_mask, int T) {
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

    for (uint32_t il = 0; il < hp.llm_n_layers; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        // ---- LN1 (RMSNorm + multiplicative weight, no bias) ----
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // ---- Q, K, V projections ----
        ggml_tensor* Q = ggml_mul_mat(ctx0, b.attn_q_w, x);
        ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, x);
        ggml_tensor* V = ggml_mul_mat(ctx0, b.attn_v_w, x);
        Q = ggml_reshape_3d(ctx0, Q, hd, n_q, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

        // ---- Q-norm / K-norm ----
        Q = ggml_rms_norm(ctx0, Q, eps);
        Q = ggml_mul(ctx0, Q, b.attn_q_norm_w);
        K = ggml_rms_norm(ctx0, K, eps);
        K = ggml_mul(ctx0, K, b.attn_k_norm_w);

        // ---- RoPE NEOX ----
        Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos, theta, 1.0f, 0.0f,
                          1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos, theta, 1.0f, 0.0f,
                          1.0f, 32.0f, 1.0f);

        // ---- GQA expand ----
        if (n_kv_grp > 1) {
            ggml_tensor* K4 = ggml_reshape_4d(ctx0, K, hd, 1, n_kv, T);
            ggml_tensor* V4 = ggml_reshape_4d(ctx0, V, hd, 1, n_kv, T);
            K4 = ggml_repeat_4d(ctx0, K4, hd, n_kv_grp, n_kv, T);
            V4 = ggml_repeat_4d(ctx0, V4, hd, n_kv_grp, n_kv, T);
            K = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, n_q, T));
            V = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, n_q, T));
        }

        // ---- Permute for attention ----
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);
        scores = ggml_add(ctx0, scores, causal_mask);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, attn_scale, 0.0f);

        ggml_tensor* V2 = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_mul_mat(ctx0, V2, scores);
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

        attn = ggml_mul_mat(ctx0, b.attn_output_w, attn);
        cur = ggml_add(ctx0, residual, attn);

        // ---- FFN ----
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    (void)d;
}

static ggml_cgraph* qwen3_asr_build_graph_llm(qwen3_asr_context* ctx, int n_tokens) {
    const auto& m = ctx->model;
    const int T = n_tokens;

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

    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    // Token embedding lookup → (d, T)
    ggml_tensor* cur = ggml_get_rows(ctx0, m.llm.token_embd_w, input_ids);
    qwen3_asr_build_llm_body(ctx, ctx0, gf, cur, positions, causal_mask, T);
    ggml_free(ctx0);
    return gf;
}

// Variant: takes pre-computed inputs_embeds (d, T) F32 instead of input_ids.
// Used by the audio-injection path after splicing audio frames into the
// text-token embedding sequence.
static ggml_cgraph* qwen3_asr_build_graph_llm_from_embeds(qwen3_asr_context* ctx, int n_tokens) {
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int T = n_tokens;

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

    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
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
static ggml_cgraph* qwen3_asr_build_graph_llm_kv(qwen3_asr_context* ctx, int n_past, int n_tokens,
                                                 bool last_token_only = true) {
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
    const int Lk = n_past + T; // total cache length after this call

    GGML_ASSERT(ctx->kv_k && ctx->kv_v);
    GGML_ASSERT(Lk <= ctx->kv_max_ctx);

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

    // Only the prefill path (T > 1) needs the causal mask. Decode (T = 1)
    // uses ggml_flash_attn_ext with no mask — the single new query attends
    // to all cached keys including itself. If we always declared the mask
    // input, the scheduler would optimize it away on the decode path and
    // ggml_graph_get_tensor("causal_mask") would return null.
    //
    // For prefill we use flash_attn_ext too, which requires the mask to
    // be F16 (and contiguous, broadcast-compatible with Q's trailing dims).
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
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    for (uint32_t il = 0; il < hp.llm_n_layers; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // KV-cached GQA self-attention with Q/K norm — qwen3 is the only
        // backend currently passing non-null q_norm_w/k_norm_w to the
        // shared helper.
        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, b.attn_q_norm_w, b.attn_k_norm_w,
            positions, (T == 1) ? nullptr : causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, kvp);
        cur = ggml_add(ctx0, residual, attn);

        // ---- FFN ----
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);

    // Last-token-only lm_head (default): slice (d, T) → (d, 1) before
    // the big matmul. The autoregressive decode loop only ever needs the
    // next-token logits, so we save the (152064, 2048) matmul on T-1
    // columns.
    //
    // Forced-alignment mode (last_token_only=false) keeps all T columns
    // because the FA model needs the lm_head output at every position
    // where input == <|timestamp|>. The lm_head shape itself is read
    // from output_w (5000 for FA, vocab_size for ASR).
    if (last_token_only && T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);
    // logits ne = (lm_head_dim, 1)        (decode / last-token mode)
    //         or (lm_head_dim, T)         (FA / full-T mode)

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Tiny standalone graph for token embedding lookup (used by run_embed_tokens).
static ggml_cgraph* qwen3_asr_build_graph_embed(qwen3_asr_context* ctx, int n_tokens) {
    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
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

extern "C" const char* qwen3_asr_token_text(qwen3_asr_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[id].c_str();
}

// ===========================================================================
// BPE tokenizer (GPT-2 byte-level, Qwen2/Qwen3 compatible)
//
// Two-stage encode:
//   1. Pre-tokenize: split the input on `<|special|>` markers and on
//      whitespace boundaries. Special tokens are looked up directly in the
//      vocab as full strings; the remaining text segments are byte-encoded
//      via the GPT-2 byte→unicode mapping and then BPE-merged.
//   2. BPE merge loop: for each pre-token, find the lowest-rank merge in
//      the merge table and apply it, repeating until no merges apply.
//      The final symbols are then looked up in the vocab.
// ===========================================================================

// All four building blocks (byte_encoder, utf8_encode, bytes_to_unicode,
// bpe_one) used to live inline here as `qwen3_*` statics. They now live
// once in src/core/bpe.h — qwen3 calls into them via the namespace alias
// below, granite uses the same primitives, and any future GPT-2-family
// model gets them for free.
#include "core/bpe.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
extern "C" int32_t* qwen3_asr_tokenize(qwen3_asr_context* ctx, const char* text, int* out_n_tokens) {
    if (!ctx || !text) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }
    const auto& v = ctx->vocab;
    std::vector<int32_t> result;

    const std::string s = text;
    size_t i = 0;
    while (i < s.size()) {
        // 1. Special-token check: if the next chars are "<|...|>" and the
        //    full token exists in the vocab, emit it directly.
        if (s[i] == '<' && i + 1 < s.size() && s[i + 1] == '|') {
            size_t end = s.find("|>", i + 2);
            if (end != std::string::npos) {
                std::string special = s.substr(i, end + 2 - i);
                auto it = v.token_to_id.find(special);
                if (it != v.token_to_id.end()) {
                    result.push_back(it->second);
                    i = end + 2;
                    continue;
                }
            }
        }

        // 2. Plain text segment: collect chars up to the next "<|...|>" we
        //    can recognize. We treat a "<|" as a candidate boundary only if
        //    it's an actual special token we know — otherwise we keep
        //    extending the plain-text segment past it (a literal "<|" in
        //    user text isn't a special token).
        size_t j = i;
        // Ensure we always advance by at least one char on every outer
        // iteration, even if step 1 just failed on a "<|...|>" lookalike
        // that isn't actually a special token.
        if (s[j] == '<' && j + 1 < s.size() && s[j + 1] == '|')
            j++;
        while (j < s.size()) {
            if (s[j] == '<' && j + 1 < s.size() && s[j + 1] == '|') {
                size_t end = s.find("|>", j + 2);
                if (end != std::string::npos) {
                    std::string special = s.substr(j, end + 2 - j);
                    if (v.token_to_id.find(special) != v.token_to_id.end())
                        break;
                }
            }
            j++;
        }
        std::string chunk = s.substr(i, j - i);
        i = j;
        if (chunk.empty())
            continue;

        // 3. Pre-split the chunk on whitespace boundaries the way GPT-2
        //    does: each pre-token starts at a non-space char and includes
        //    the leading space if present (Ġ marker). We approximate this
        //    by splitting on transitions between "leading space + word",
        //    "word", "punctuation", etc.
        size_t k = 0;
        while (k < chunk.size()) {
            size_t start = k;
            // Optional leading whitespace (single char)
            if (chunk[k] == ' ' || chunk[k] == '\t' || chunk[k] == '\n')
                k++;
            // The "word body" — everything until the next whitespace OR a
            // standalone punctuation transition.
            while (k < chunk.size() && chunk[k] != ' ' && chunk[k] != '\t' && chunk[k] != '\n') {
                k++;
            }
            if (k == start)
                k++; // pure whitespace boundary, advance one
            std::string pre(chunk, start, k - start);
            // 4. Byte-encode and BPE-merge via the shared helpers.
            std::string encoded = core_bpe::bytes_to_unicode(pre.data(), pre.size());
            core_bpe::bpe_one(v.token_to_id, v.merge_rank, encoded, result);
        }
    }

    if (out_n_tokens)
        *out_n_tokens = (int)result.size();
    int32_t* out = (int32_t*)malloc(result.size() * sizeof(int32_t));
    if (!out) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }
    std::memcpy(out, result.data(), result.size() * sizeof(int32_t));
    return out;
}

extern "C" qwen3_asr_context_params qwen3_asr_context_default_params(void) {
    qwen3_asr_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

extern "C" qwen3_asr_context* qwen3_asr_init_from_file(const char* path, qwen3_asr_context_params params) {
    qwen3_asr_context* ctx = new qwen3_asr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Try GPU backend first (Metal, CUDA, Vulkan...), fall back to CPU.
    // ggml_backend_init_best() picks the highest-priority available backend.
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

    if (!qwen3_asr_load_model(ctx->model, ctx->vocab, path, ctx->backend)) {
        delete ctx;
        return nullptr;
    }

    // Create the backend scheduler once with the worst-case node budget.
    // All compute functions reuse this scheduler via ggml_backend_sched_reset().
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        fprintf(stderr, "qwen3_asr: loaded %s  (audio %u layers, llm %u layers, vocab %u)\n", path,
                ctx->model.hparams.audio_n_layers, ctx->model.hparams.llm_n_layers,
                (uint32_t)ctx->vocab.id_to_token.size());
    }
    return ctx;
}

extern "C" void qwen3_asr_free(qwen3_asr_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" char* qwen3_asr_transcribe(qwen3_asr_context* /*ctx*/, const float* /*samples*/, int /*n_samples*/) {
    // Stage 1: not yet implemented end-to-end. Use qwen3_asr_run_conv for now.
    return strdup("");
}

extern "C" float* qwen3_asr_run_conv(qwen3_asr_context* ctx, const float* mel_features, int n_mels, int T_mel,
                                     int* out_n_chunks, int* out_T_chunk_out, int* out_d) {
    if (!ctx || !mel_features)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    if (n_mels != (int)hp.n_mels) {
        fprintf(stderr, "qwen3_asr: mel feature mismatch (%d vs %d)\n", n_mels, (int)hp.n_mels);
        return nullptr;
    }

    // Chunking: split T_mel into chunks of n_window*2. The final chunk is
    // padded with zeros to n_window*2 to match the reference impl, which
    // pad_sequences chunks before batching them through the convs.
    const int chunk_T = (int)hp.n_window * 2; // 100
    const int num_chunks = (T_mel + chunk_T - 1) / chunk_T;

    // Build (T_chunk=100, n_mels=128, 1, num_chunks) F32 buffer, padded with zeros.
    std::vector<float> mel_padded((size_t)chunk_T * n_mels * num_chunks, 0.0f);
    // Source layout: mel_features is (n_mels, T_mel), row-major (mel as outer,
    // time as inner). Per the reference dump it's saved that way.
    // Target ggml layout: ne[0]=T_chunk varies fastest, ne[1]=n_mels, ne[3]=batch.
    // Memory index: t + chunk_T*(f + n_mels*chunk_idx)
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int t_start = chunk * chunk_T;
        const int t_end = std::min(t_start + chunk_T, T_mel);
        const int t_len = t_end - t_start;
        for (int f = 0; f < n_mels; f++) {
            for (int t = 0; t < t_len; t++) {
                // src: (f, t_start + t) — mel_features[f * T_mel + (t_start + t)]
                // dst: (t, f, 0, chunk) — mel_padded[t + chunk_T*(f + n_mels*chunk)]
                mel_padded[(size_t)t + chunk_T * ((size_t)f + n_mels * (size_t)chunk)] =
                    mel_features[(size_t)f * T_mel + (size_t)(t_start + t)];
            }
        }
    }

    ggml_cgraph* gf = qwen3_asr_build_graph_conv(ctx, chunk_T, num_chunks);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc conv graph\n");
        return nullptr;
    }

    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel_batched");
    ggml_backend_tensor_set(mel_in, mel_padded.data(), 0, mel_padded.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: conv graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "conv_front_out");
    if (!out) {
        fprintf(stderr, "qwen3_asr: missing conv_front_out tensor\n");
        return nullptr;
    }
    const int d = (int)out->ne[0]; // 896
    const int T = (int)out->ne[1]; // 13
    const int B = (int)out->ne[2]; // num_chunks
    if (out_n_chunks)
        *out_n_chunks = B;
    if (out_T_chunk_out)
        *out_T_chunk_out = T;
    if (out_d)
        *out_d = d;

    const size_t total = (size_t)d * T * B;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" float* qwen3_asr_run_encoder(qwen3_asr_context* ctx, const float* mel_features, int n_mels, int T_mel,
                                        int* out_N_total, int* out_proj_dim) {
    if (!ctx || !mel_features)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    if (n_mels != (int)hp.n_mels) {
        fprintf(stderr, "qwen3_asr: mel feature mismatch (%d vs %d)\n", n_mels, (int)hp.n_mels);
        return nullptr;
    }

    // Chunking. Round T_mel up to the nearest multiple of chunk_T = 100 and
    // zero-pad the trailing partial chunk. The padding shows up as "silence"
    // encoder frames at the end of the sequence; the LLM handles them
    // naturally (it's trained on audio with silence). For long audio there
    // are typically only 0..99 padding frames out of thousands.
    const int chunk_T = (int)hp.n_window * 2;
    const int num_chunks = (T_mel + chunk_T - 1) / chunk_T;

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
        const int t_end = std::min(t_start + chunk_T, T_mel);
        const int t_len = t_end - t_start; // valid (non-padded) frames in this chunk
        for (int f = 0; f < n_mels; f++) {
            for (int t = 0; t < t_len; t++) {
                mel_padded[(size_t)t + chunk_T * ((size_t)f + n_mels * (size_t)chunk)] =
                    mel_features[(size_t)f * T_mel + (size_t)(t_start + t)];
            }
            // remaining (chunk_T - t_len) entries are already zero from the
            // initial assignment — silence padding for the trailing partial chunk.
        }
    }

    // The reference's eager_attention_forward IGNORES cu_seqlens and uses
    // standard full self-attention with attention_mask=None. cu_seqlens is
    // only consumed by FlashAttention2 on GPU. So on CPU we just need a
    // zero mask. (We keep the input tensor in the graph so the structure
    // is ready when we add real per-chunk padding masking later.)
    std::vector<float> mask((size_t)N_padded * N_padded, 0.0f);

    ggml_cgraph* gf = qwen3_asr_build_graph_encoder(ctx, chunk_T, num_chunks, T_chunk_out);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc encoder graph\n");
        return nullptr;
    }

    // Set inputs
    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel_batched");
    ggml_backend_tensor_set(mel_in, mel_padded.data(), 0, mel_padded.size() * sizeof(float));

    // pe_input ne=(d, T_chunk_out). Pull rows [0, T_chunk_out) from model.audio_pe.
    ggml_tensor* pe_in = ggml_graph_get_tensor(gf, "pe_input");
    {
        const int d = (int)hp.audio_d_model;
        std::vector<float> pe_buf((size_t)d * T_chunk_out);
        // model.audio_pe row p starts at offset p*d. We need to write into ggml
        // ne=(d, T_chunk_out) which has d as ne[0] (fast). Memory layout matches
        // a row-major (T_chunk_out, d) buffer. So just copy [0, T_chunk_out*d).
        std::memcpy(pe_buf.data(), ctx->model.audio_pe.data(), pe_buf.size() * sizeof(float));
        ggml_backend_tensor_set(pe_in, pe_buf.data(), 0, pe_buf.size() * sizeof(float));
    }

    ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "attn_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: encoder graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "encoder_out");
    if (!out) {
        fprintf(stderr, "qwen3_asr: missing encoder_out tensor\n");
        return nullptr;
    }
    const int pdim = (int)out->ne[0]; // 1024
    const int N = (int)out->ne[1];    // N_padded
    if (out_N_total)
        *out_N_total = N;
    if (out_proj_dim)
        *out_proj_dim = pdim;

    const size_t total = (size_t)pdim * N;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" bool qwen3_asr_kv_init(qwen3_asr_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    if (ctx->kv_k)
        return true; // already initialized

    const auto& hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int n_lay = (int)hp.llm_n_layers;

    ggml_init_params kp = {
        /*mem_size=*/ggml_tensor_overhead() * 4 + 1024,
        /*mem_buffer=*/nullptr,
        /*no_alloc=*/true,
    };
    ctx->kv_ctx = ggml_init(kp);
    // F16 KV cache: halves memory + ~2× cache read bandwidth on decode.
    // Conversion happens at the ggml_cpy() write into the cache view, and
    // ggml_mul_mat handles F16-on-F32 dot products natively for the read path.
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    const size_t kbytes = ggml_nbytes(ctx->kv_k);
    const size_t vbytes = ggml_nbytes(ctx->kv_v);
    ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, kbytes + vbytes);
    if (!ctx->kv_buf) {
        fprintf(stderr, "qwen3_asr: failed to allocate kv buffer\n");
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);
    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used = 0;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "qwen3_asr: kv cache %d MiB (head_dim=%d max_ctx=%d n_kv=%d n_layers=%d)\n",
                (int)((kbytes + vbytes) / 1048576), hd, max_ctx, n_kv, n_lay);
    }
    return true;
}

extern "C" void qwen3_asr_kv_reset(qwen3_asr_context* ctx) {
    if (ctx)
        ctx->kv_n_used = 0;
}

extern "C" float* qwen3_asr_run_llm_kv(qwen3_asr_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                                       int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    if (!ctx->kv_k) {
        fprintf(stderr, "qwen3_asr: kv cache not initialized — call qwen3_asr_kv_init first\n");
        return nullptr;
    }
    if (n_past + n_tokens > ctx->kv_max_ctx) {
        fprintf(stderr, "qwen3_asr: kv overflow (n_past=%d + n_tokens=%d > max_ctx=%d)\n", n_past, n_tokens,
                ctx->kv_max_ctx);
        return nullptr;
    }
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    // Use the LM head's actual output dim, not the token vocab size.
    // For ASR models the two are equal; for the ForcedAligner variant
    // the head is (5000, d) while vocab_size stays 152064.
    const int vocab = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);
    const int Lk = n_past + n_tokens;

    // Positions [n_past, n_past+T)
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

    // Causal mask: only needed for prefill (T > 1). Decode (T = 1) uses
    // ggml_flash_attn_ext with no mask — the single new query attends to
    // all cached keys including itself, so no masking is needed.
    // ggml_flash_attn_ext requires the mask to be F16.
    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * n_tokens, zero_h);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = n_past + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neginf_h;
            }
        }
    }

    ggml_cgraph* gf = qwen3_asr_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc llm_kv graph\n");
        return nullptr;
    }

    ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: llm_kv graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out)
        return nullptr;
    ctx->kv_n_used = n_past + n_tokens;
    // Output is only the last token's logits — see lm_head slice in build_graph_llm_kv.
    if (out_n_tokens)
        *out_n_tokens = 1;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)vocab * sizeof(float));
    return result;
}

extern "C" float* qwen3_asr_embed_tokens(qwen3_asr_context* ctx, const int32_t* input_ids, int n_tokens) {
    if (!ctx || !input_ids || n_tokens <= 0)
        return nullptr;
    const int d = (int)ctx->model.hparams.llm_d_model;

    ggml_cgraph* gf = qwen3_asr_build_graph_embed(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc embed graph\n");
        return nullptr;
    }
    ggml_tensor* ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, input_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: embed graph compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    const size_t total = (size_t)d * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" float* qwen3_asr_run_llm_from_embeds(qwen3_asr_context* ctx, const float* inputs_embeds, int n_tokens,
                                                int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int vocab = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = i;
    std::vector<float> mask((size_t)n_tokens * n_tokens, 0.0f);
    for (int i = 0; i < n_tokens; i++)
        for (int j = i + 1; j < n_tokens; j++)
            mask[(size_t)i * n_tokens + j] = -INFINITY;

    ggml_cgraph* gf = qwen3_asr_build_graph_llm_from_embeds(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc llm-from-embeds graph\n");
        return nullptr;
    }

    ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: llm-from-embeds graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out) {
        fprintf(stderr, "missing logits\n");
        return nullptr;
    }
    if (out_n_tokens)
        *out_n_tokens = n_tokens;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    const size_t total = (size_t)vocab * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" float* qwen3_asr_run_llm(qwen3_asr_context* ctx, const int32_t* input_ids, int n_tokens, int* out_n_tokens,
                                    int* out_vocab_size) {
    if (!ctx || !input_ids || n_tokens <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int vocab = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);

    // Build positions = [0, 1, ..., T-1]
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = i;

    // Build causal mask: (T, T) F32. mask[i, j] = 0 if j <= i else -inf.
    // ggml ne[0]=j (key, fast), ne[1]=i (query). Disallowed → -inf.
    std::vector<float> mask((size_t)n_tokens * n_tokens, 0.0f);
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < n_tokens; j++) {
            if (j > i)
                mask[(size_t)i * n_tokens + j] = -INFINITY;
        }
    }

    ggml_cgraph* gf = qwen3_asr_build_graph_llm(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc llm graph\n");
        return nullptr;
    }

    ggml_tensor* ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, input_ids, 0, (size_t)n_tokens * sizeof(int32_t));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));

    ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: llm graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out) {
        fprintf(stderr, "qwen3_asr: missing logits tensor\n");
        return nullptr;
    }
    if (out_n_tokens)
        *out_n_tokens = n_tokens;
    if (out_vocab_size)
        *out_vocab_size = vocab;

    const size_t total = (size_t)vocab * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" int qwen3_asr_lm_head_dim(struct qwen3_asr_context* ctx) {
    if (!ctx)
        return 0;
    const auto& hp = ctx->model.hparams;
    return (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);
}

// High-level forced-alignment dispatch.
//
// Mirrors qwen_asr/inference/qwen3_forced_aligner.py::Qwen3ForcedAligner.align
// minus the language-specific tokenizers (we whitespace-split words and let
// each word go through the standard byte-level BPE encoder; CJK char-level
// tokenization is a follow-up).
extern "C" int qwen3_asr_align_words(struct qwen3_asr_context* ctx, const float* samples, int n_samples,
                                     const char** words, int n_words, int64_t* out_start_ms, int64_t* out_end_ms) {
    if (!ctx || !samples || n_samples <= 0 || !words || n_words <= 0 || !out_start_ms || !out_end_ms)
        return -1;

    constexpr int TIMESTAMP_TOKEN_ID = 151705;
    // Default ms-per-class. Forced-aligner config sets this to 80; the
    // value isn't carried in the GGUF metadata yet, so we hardcode it
    // for now and add a kv field on the next converter bump.
    constexpr float TIMESTAMP_SEGMENT_TIME_MS = 80.0f;

    // 1. Mel
    int n_mels = 0, T_mel = 0;
    float* mel = qwen3_asr_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        fprintf(stderr, "qwen3_asr[align]: mel failed\n");
        return -2;
    }

    // 2. Audio encoder
    int N_enc = 0, pdim = 0;
    float* audio_embeds = qwen3_asr_run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &pdim);
    free(mel);
    if (!audio_embeds) {
        fprintf(stderr, "qwen3_asr[align]: encoder failed\n");
        return -3;
    }

    // 3. Build the prompt token-id sequence:
    //      <|audio_start|>  <|audio_pad|>×N_enc  <|audio_end|>
    //      word_1 <timestamp> <timestamp>
    //      word_2 <timestamp> <timestamp>
    //      ...
    //      word_M <timestamp> <timestamp>
    const auto& hp = ctx->model.hparams;
    const int audio_start_id = (int)hp.audio_start_token_id;
    const int audio_end_id = (int)hp.audio_end_token_id;
    const int audio_pad_id = (int)hp.audio_pad_token_id;

    std::vector<int32_t> ids;
    ids.reserve((size_t)(N_enc + n_words * 6 + 4));
    ids.push_back(audio_start_id);
    for (int i = 0; i < N_enc; i++)
        ids.push_back(audio_pad_id);
    ids.push_back(audio_end_id);

    // Tokenize each word separately and append two timestamp markers
    // after it. Whitespace-split English / Latin scripts work fine
    // through the standard BPE encoder; CJK languages need char-level
    // pre-tokenization which is tracked as a follow-up. The leading
    // space convention matches GPT-2 BPE: each non-first word starts
    // with a space so the tokenizer recognises it as a word boundary.
    for (int w = 0; w < n_words; w++) {
        const std::string word = (w == 0) ? std::string(words[w]) : std::string(" ") + words[w];
        int n = 0;
        int32_t* arr = qwen3_asr_tokenize(ctx, word.c_str(), &n);
        if (arr && n > 0) {
            for (int i = 0; i < n; i++)
                ids.push_back(arr[i]);
        }
        free(arr);
        ids.push_back(TIMESTAMP_TOKEN_ID);
        ids.push_back(TIMESTAMP_TOKEN_ID);
    }

    const int T_prompt = (int)ids.size();

    // 4. Embed and splice audio
    float* text_embeds = qwen3_asr_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) {
        free(audio_embeds);
        fprintf(stderr, "qwen3_asr[align]: embed failed\n");
        return -4;
    }
    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc; i++) {
        if (ids[i] == audio_pad_id) {
            std::memcpy(text_embeds + (size_t)i * pdim, audio_embeds + (size_t)spliced * pdim, pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);

    // 5. KV cache + aligner forward
    if (!qwen3_asr_kv_init(ctx, /*max_ctx*/ std::max(4096, T_prompt + 16))) {
        free(text_embeds);
        fprintf(stderr, "qwen3_asr[align]: kv_init failed\n");
        return -5;
    }
    int n_t_out = 0, H = 0;
    float* logits = qwen3_asr_run_aligner(ctx, text_embeds, T_prompt, &n_t_out, &H);
    free(text_embeds);
    if (!logits) {
        fprintf(stderr, "qwen3_asr[align]: aligner forward failed\n");
        return -6;
    }

    // 6. argmax over the lm_head_dim classes at each <timestamp>
    // position. The graph stores logits as ne[0]=H, ne[1]=T_prompt
    // row-major, i.e. logits[t * H + k] = score(k, t).
    std::vector<int> ts_classes;
    ts_classes.reserve((size_t)(n_words * 2));
    for (int t = 0; t < T_prompt; t++) {
        if (ids[t] != TIMESTAMP_TOKEN_ID)
            continue;
        const float* row = logits + (size_t)t * H;
        int best = 0;
        float mx = row[0];
        for (int k = 1; k < H; k++) {
            if (row[k] > mx) {
                mx = row[k];
                best = k;
            }
        }
        ts_classes.push_back(best);
    }
    free(logits);

    if ((int)ts_classes.size() != 2 * n_words) {
        fprintf(stderr,
                "qwen3_asr[align]: timestamp marker count mismatch — "
                "got %zu placeholders, expected %d\n",
                ts_classes.size(), 2 * n_words);
        return -7;
    }

    // 6b. Fix timestamp monotonicity via LIS (longest increasing subsequence).
    // The reference Qwen3-ForcedAligner uses LIS to find the longest monotone
    // chain, then interpolates outliers. This is more robust than a simple
    // forward clamp for cases where large inversions occur mid-sequence.
    {
        const int M = (int)ts_classes.size();
        // O(n log n) LIS — find indices of the longest non-decreasing subsequence
        std::vector<int> dp; // dp[i] = smallest tail value for IS of length i+1
        std::vector<int> parent(M, -1);
        std::vector<int> idx_map; // which index produced each dp entry

        for (int i = 0; i < M; i++) {
            int val = ts_classes[i];
            // Binary search for first dp entry > val (upper_bound for non-decreasing)
            auto it = std::upper_bound(dp.begin(), dp.end(), val);
            int pos = (int)(it - dp.begin());
            if (pos == (int)dp.size()) {
                dp.push_back(val);
                idx_map.push_back(i);
            } else {
                dp[pos] = val;
                idx_map[pos] = i;
            }
            parent[i] = (pos > 0) ? idx_map[pos - 1] : -1;
        }

        // Traceback: find which indices are in the LIS
        std::vector<bool> in_lis(M, false);
        int k = idx_map.back();
        while (k >= 0) {
            in_lis[k] = true;
            k = parent[k];
        }

        // Interpolate outliers: for each non-LIS element, set it to the
        // value of the nearest LIS neighbor (linear interpolation between
        // the previous and next LIS values).
        int prev_lis = -1;
        int prev_val = 0;
        for (int i = 0; i < M; i++) {
            if (in_lis[i]) {
                // Fill any gap between prev_lis and i
                if (prev_lis >= 0) {
                    for (int j = prev_lis + 1; j < i; j++) {
                        // Linear interpolation
                        float frac = (float)(j - prev_lis) / (float)(i - prev_lis);
                        ts_classes[j] = prev_val + (int)(frac * (float)(ts_classes[i] - prev_val));
                    }
                }
                prev_lis = i;
                prev_val = ts_classes[i];
            }
        }
        // Fill trailing non-LIS elements
        for (int j = prev_lis + 1; j < M; j++)
            ts_classes[j] = prev_val;
    }

    // Ensure each word's end >= start.
    for (int w = 0; w < n_words; w++) {
        if (ts_classes[2 * w + 1] < ts_classes[2 * w])
            ts_classes[2 * w + 1] = ts_classes[2 * w];
    }

    // 7. Convert classes → ms and write into the caller's parallel arrays.
    for (int w = 0; w < n_words; w++) {
        out_start_ms[w] = (int64_t)((float)ts_classes[2 * w + 0] * TIMESTAMP_SEGMENT_TIME_MS);
        out_end_ms[w] = (int64_t)((float)ts_classes[2 * w + 1] * TIMESTAMP_SEGMENT_TIME_MS);
    }
    return 0;
}

// One full-T forward pass for the Qwen3-ForcedAligner. Same KV-cached
// graph the ASR backend uses for prefill, but with last_token_only=false
// so the lm_head sees every position. The KV cache is reset at the start
// because forced alignment is one-shot — no autoregressive decode loop.
extern "C" float* qwen3_asr_run_aligner(struct qwen3_asr_context* ctx, const float* inputs_embeds, int n_tokens,
                                        int* out_n_tokens, int* out_lm_head_dim) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    if (!ctx->kv_k) {
        fprintf(stderr, "qwen3_asr: kv cache not initialized — call qwen3_asr_kv_init first\n");
        return nullptr;
    }
    if (n_tokens > ctx->kv_max_ctx) {
        fprintf(stderr, "qwen3_asr: aligner needs %d tokens but kv max_ctx is %d\n", n_tokens, ctx->kv_max_ctx);
        return nullptr;
    }
    qwen3_asr_kv_reset(ctx);

    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int H = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);
    const int Lk = n_tokens;

    // Positions 0..n_tokens-1.
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = i;

    // Causal mask for the prefill path. Same encoding the run_llm_kv
    // path uses (F16, ne[0]=Lk for fast key axis, ne[1]=n_tokens for
    // queries; -inf above the diagonal).
    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * n_tokens, zero_h);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neginf_h;
            }
        }
    }

    ggml_cgraph* gf = qwen3_asr_build_graph_llm_kv(ctx, /*n_past=*/0, n_tokens, /*last_token_only=*/false);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc aligner graph\n");
        return nullptr;
    }

    ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: aligner graph compute failed\n");
        return nullptr;
    }

    ctx->kv_n_used = n_tokens;

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out)
        return nullptr;
    if (out_n_tokens)
        *out_n_tokens = n_tokens;
    if (out_lm_head_dim)
        *out_lm_head_dim = H;

    const size_t total = (size_t)H * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    if (!result)
        return nullptr;
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}
