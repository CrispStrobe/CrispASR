// src/ark_asr.cpp — ARK-ASR-3B runtime (see PLAN.md §ARK).
//
// Pipeline: Whisper mel -> Whisper-large-v3 encoder with partial interleaved
// RoPE (rot_dim 32 of head_dim 64, theta 10000) + no final LN -> MLP adapter
// (LayerNorm -> merge-4 -> Linear 5120->4096 -> GELU -> Linear 4096->2048) ->
// audio embeddings injected at <|audio|> placeholder positions in a Qwen2.5-3B
// decoder, greedy-decoded to text.

#include "ark_asr.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>

#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/mel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Hyperparameters
// ===========================================================================
struct ark_asr_hp {
    // decoder (Qwen2.5-3B)
    uint32_t llm_hidden = 2048;
    uint32_t llm_layers = 36;
    uint32_t llm_heads = 16;
    uint32_t llm_kv_heads = 2;
    uint32_t llm_head_dim = 128;
    uint32_t llm_intermediate = 11008;
    uint32_t llm_vocab = 151936;
    uint32_t llm_max_pos = 32768;
    float llm_rope_theta = 1000000.0f;
    float llm_rms_eps = 1e-6f;

    // encoder (Whisper)
    uint32_t enc_d_model = 1280;
    uint32_t enc_layers = 32;
    uint32_t enc_heads = 20;
    uint32_t enc_head_dim = 64;
    uint32_t enc_ffn = 5120;
    uint32_t n_mels = 128;
    uint32_t enc_max_pos = 1500;
    uint32_t enc_rot_dim = 32;
    float enc_rope_theta = 10000.0f;
    float enc_ln_eps = 1e-5f;

    // adapter / audio / mel
    uint32_t merge_factor = 4;
    uint32_t audio_token_id = 151663;
    uint32_t eos_token_id = 151645;
    uint32_t n_fft = 400;
    uint32_t hop_length = 160;
    uint32_t sample_rate = 16000;
};

// ===========================================================================
// Tensor structs
// ===========================================================================
struct ark_enc_block {
    ggml_tensor* attn_ln_w = nullptr;
    ggml_tensor* attn_ln_b = nullptr;
    ggml_tensor* q_w = nullptr;
    ggml_tensor* q_b = nullptr;
    ggml_tensor* k_w = nullptr; // no bias
    ggml_tensor* v_w = nullptr;
    ggml_tensor* v_b = nullptr;
    ggml_tensor* o_w = nullptr;
    ggml_tensor* o_b = nullptr;
    ggml_tensor* ffn_ln_w = nullptr;
    ggml_tensor* ffn_ln_b = nullptr;
    ggml_tensor* fc1_w = nullptr;
    ggml_tensor* fc1_b = nullptr;
    ggml_tensor* fc2_w = nullptr;
    ggml_tensor* fc2_b = nullptr;
};

struct ark_dec_block {
    ggml_tensor* attn_norm_w = nullptr; // RMSNorm
    ggml_tensor* q_w = nullptr;
    ggml_tensor* q_b = nullptr;
    ggml_tensor* k_w = nullptr;
    ggml_tensor* k_b = nullptr;
    ggml_tensor* v_w = nullptr;
    ggml_tensor* v_b = nullptr;
    ggml_tensor* o_w = nullptr; // no bias
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct ark_asr_model {
    // encoder
    ggml_tensor* conv1_w = nullptr;
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* conv2_w = nullptr;
    ggml_tensor* conv2_b = nullptr;
    std::vector<ark_enc_block> enc;
    // adapter
    ggml_tensor* adapter_ln_w = nullptr;
    ggml_tensor* adapter_ln_b = nullptr;
    ggml_tensor* adapter_fc1_w = nullptr;
    ggml_tensor* adapter_fc1_b = nullptr;
    ggml_tensor* adapter_fc2_w = nullptr;
    ggml_tensor* adapter_fc2_b = nullptr;
    // mel
    ggml_tensor* mel_filters = nullptr; // (n_freqs, n_mels) F32
    ggml_tensor* mel_window = nullptr;  // (n_fft,) F32
    // decoder
    ggml_tensor* embed_w = nullptr; // [hidden, vocab]; also tied lm_head
    ggml_tensor* norm_w = nullptr;
    std::vector<ark_dec_block> dec;
};

struct ark_asr_context {
    ark_asr_context_params params{};
    int n_threads = 4;
    ark_asr_hp hp;
    ark_asr_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf_w_cpu = nullptr;
    core_gguf::tensor_map tensors;
    std::vector<uint8_t> compute_meta;

    // KV cache (decoder)
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;

    // tokenizer
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
    int32_t id_user = 151665, id_boa = 151666, id_audio = 151663, id_eoa = 151667;
    int32_t id_assistant = 151668, id_im_end = 151645;

    std::string language;
    std::string ask; // EXPERIMENTAL transcription instruction (see ark_asr_set_ask)
};

// ===========================================================================
// Direct real-input DFT (n_fft=400 is not a power of two). Fills out[2k],
// out[2k+1] for k in [0, N/2]; the rest is zeroed. Matches core_mel::FftR2C.
// ===========================================================================
static void ark_fft(const float* in, int N, float* out) {
    std::memset(out, 0, (size_t)2 * N * sizeof(float));
    const int half = N / 2;
    for (int k = 0; k <= half; k++) {
        double re = 0.0, im = 0.0;
        const double w = -2.0 * M_PI * (double)k / (double)N;
        for (int n = 0; n < N; n++) {
            const double a = w * (double)n;
            re += (double)in[n] * std::cos(a);
            im += (double)in[n] * std::sin(a);
        }
        out[2 * k] = (float)re;
        out[2 * k + 1] = (float)im;
    }
}

// ===========================================================================
// Backend / scheduler helpers
// ===========================================================================
static bool ark_kv_init(ark_asr_context* ctx, int max_ctx) {
    if (ctx->kv_k && ctx->kv_max_ctx >= max_ctx)
        return true;
    if (ctx->kv_buf) {
        ggml_backend_buffer_free(ctx->kv_buf);
        ctx->kv_buf = nullptr;
    }
    if (ctx->kv_ctx) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
    }
    const auto& hp = ctx->hp;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_kv_heads;
    const int n_lay = (int)hp.llm_layers;

    const auto kvp = core_attn::kv_dtype_pair_from_env("ark_asr");
    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    ctx->kv_ctx = ggml_init(kp);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kvp.k, hd, max_ctx, n_kv, n_lay);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kvp.v, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->kv_k, "ark_kv_k");
    ggml_set_name(ctx->kv_v, "ark_kv_v");

    ggml_backend_t kv_be = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "ark_asr");
    const size_t kbytes = ggml_nbytes(ctx->kv_k);
    const size_t vbytes = ggml_nbytes(ctx->kv_v);
    ctx->kv_buf = ggml_backend_alloc_buffer(kv_be, kbytes + vbytes);
    if (!ctx->kv_buf)
        return false;
    {
        char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
        ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
        ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);
    }
    // Zero-clear so never-written tail slots can't leak NaN/garbage.
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    ctx->kv_max_ctx = max_ctx;
    return true;
}

// ===========================================================================
// Init
// ===========================================================================
extern "C" struct ark_asr_context_params ark_asr_context_default_params(void) {
    ark_asr_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.temperature = 0.0f;
    p.beam_size = 1;
    return p;
}

extern "C" struct ark_asr_context* ark_asr_init_from_file(const char* path_model,
                                                          struct ark_asr_context_params params) {
    auto* ctx = new ark_asr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    gguf_context* g = core_gguf::open_metadata(path_model);
    if (!g) {
        fprintf(stderr, "ark_asr: failed to read GGUF '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    auto& hp = ctx->hp;
    hp.llm_hidden = core_gguf::kv_u32(g, "arkasr.llm.hidden_size", hp.llm_hidden);
    hp.llm_layers = core_gguf::kv_u32(g, "arkasr.llm.num_layers", hp.llm_layers);
    hp.llm_heads = core_gguf::kv_u32(g, "arkasr.llm.num_heads", hp.llm_heads);
    hp.llm_kv_heads = core_gguf::kv_u32(g, "arkasr.llm.num_kv_heads", hp.llm_kv_heads);
    hp.llm_head_dim = core_gguf::kv_u32(g, "arkasr.llm.head_dim", hp.llm_head_dim);
    hp.llm_intermediate = core_gguf::kv_u32(g, "arkasr.llm.intermediate_size", hp.llm_intermediate);
    hp.llm_vocab = core_gguf::kv_u32(g, "arkasr.llm.vocab_size", hp.llm_vocab);
    hp.llm_max_pos = core_gguf::kv_u32(g, "arkasr.llm.max_position_embeddings", hp.llm_max_pos);
    hp.llm_rope_theta = core_gguf::kv_f32(g, "arkasr.llm.rope_theta", hp.llm_rope_theta);
    hp.llm_rms_eps = core_gguf::kv_f32(g, "arkasr.llm.rms_norm_eps", hp.llm_rms_eps);
    hp.enc_d_model = core_gguf::kv_u32(g, "arkasr.whisper.d_model", hp.enc_d_model);
    hp.enc_layers = core_gguf::kv_u32(g, "arkasr.whisper.num_layers", hp.enc_layers);
    hp.enc_heads = core_gguf::kv_u32(g, "arkasr.whisper.num_heads", hp.enc_heads);
    hp.enc_head_dim = core_gguf::kv_u32(g, "arkasr.whisper.head_dim", hp.enc_head_dim);
    hp.enc_ffn = core_gguf::kv_u32(g, "arkasr.whisper.ffn_dim", hp.enc_ffn);
    hp.n_mels = core_gguf::kv_u32(g, "arkasr.whisper.num_mel_bins", hp.n_mels);
    hp.enc_max_pos = core_gguf::kv_u32(g, "arkasr.whisper.max_source_positions", hp.enc_max_pos);
    hp.enc_rot_dim = core_gguf::kv_u32(g, "arkasr.whisper.rot_dim", hp.enc_rot_dim);
    hp.enc_rope_theta = core_gguf::kv_f32(g, "arkasr.whisper.rope_theta", hp.enc_rope_theta);
    hp.enc_ln_eps = core_gguf::kv_f32(g, "arkasr.whisper.ln_eps", hp.enc_ln_eps);
    hp.merge_factor = core_gguf::kv_u32(g, "arkasr.adapter.merge_factor", hp.merge_factor);
    hp.audio_token_id = core_gguf::kv_u32(g, "arkasr.audio_token_id", hp.audio_token_id);
    hp.eos_token_id = core_gguf::kv_u32(g, "arkasr.eos_token_id", hp.eos_token_id);
    hp.n_fft = core_gguf::kv_u32(g, "arkasr.n_fft", hp.n_fft);
    hp.hop_length = core_gguf::kv_u32(g, "arkasr.hop_length", hp.hop_length);
    hp.sample_rate = core_gguf::kv_u32(g, "arkasr.sample_rate", hp.sample_rate);

    ctx->vocab = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    ctx->token_to_id.reserve(ctx->vocab.size());
    for (size_t i = 0; i < ctx->vocab.size(); i++)
        ctx->token_to_id.emplace(ctx->vocab[i], (int32_t)i);
    {
        auto merges = core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
        ctx->merge_rank.reserve(merges.size());
        for (size_t i = 0; i < merges.size(); i++)
            ctx->merge_rank.emplace(merges[i], (int32_t)i);
    }
    core_gguf::free_metadata(g);

    auto find_id = [&](const char* name, int32_t fallback) {
        auto it = ctx->token_to_id.find(name);
        return it != ctx->token_to_id.end() ? it->second : fallback;
    };
    ctx->id_user = find_id("<|user|>", ctx->id_user);
    ctx->id_boa = find_id("<|begin_of_audio|>", ctx->id_boa);
    ctx->id_audio = (int32_t)hp.audio_token_id;
    ctx->id_eoa = find_id("<|end_of_audio|>", ctx->id_eoa);
    ctx->id_assistant = find_id("<|assistant|>", ctx->id_assistant);
    ctx->id_im_end = (int32_t)hp.eos_token_id;

    // Backends
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "ark_asr: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    // GPU default. Validated verbatim on M1 Metal (en jfk + de De-Abwasch,
    // 2026-06-29): ~5.6x faster prefill, ~neutral per-token decode (single-token
    // decode is bandwidth/dispatch-bound on unified memory), ~1.7x faster overall
    // on jfk. (An earlier port build "emitted no tokens" on GPU; the current
    // flash-attn + KV path no longer reproduces it.) Force CPU with
    // CRISPASR_ARKASR_CPU=1. CUDA/other GPUs not yet validated.
    const bool force_cpu = std::getenv("CRISPASR_ARKASR_CPU") != nullptr;
    if (params.use_gpu && !force_cpu) {
        ctx->backend = ggml_backend_init_best();
        if (!ctx->backend) {
            ctx->backend = ctx->backend_cpu;
        } else if (params.verbosity >= 1) {
            fprintf(stderr, "ark_asr: GPU backend active (Metal-validated; CRISPASR_ARKASR_CPU=1 to force CPU)\n");
        }
    } else {
        ctx->backend = ctx->backend_cpu;
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "ark_asr", wl)) {
        fprintf(stderr, "ark_asr: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->buf_w_cpu = wl.buf_cpu;
    ctx->tensors = std::move(wl.tensors);

    auto req = [&](const std::string& n) { return core_gguf::require(ctx->tensors, n.c_str(), "ark_asr"); };
    auto opt = [&](const std::string& n) { return core_gguf::try_get(ctx->tensors, n.c_str()); };

    auto& m = ctx->model;
    bool ok = true;
    m.conv1_w = req("enc.conv1.weight");
    m.conv1_b = req("enc.conv1.bias");
    m.conv2_w = req("enc.conv2.weight");
    m.conv2_b = req("enc.conv2.bias");
    m.mel_filters = req("enc.mel_filters");
    m.mel_window = req("enc.mel_window");
    m.enc.resize(hp.enc_layers);
    for (uint32_t i = 0; i < hp.enc_layers; i++) {
        char p[48];
        snprintf(p, sizeof(p), "enc.blk.%u", i);
        std::string s(p);
        auto& b = m.enc[i];
        b.attn_ln_w = req(s + ".attn_ln.weight");
        b.attn_ln_b = req(s + ".attn_ln.bias");
        b.q_w = req(s + ".attn.q.weight");
        b.q_b = req(s + ".attn.q.bias");
        b.k_w = req(s + ".attn.k.weight");
        b.v_w = req(s + ".attn.v.weight");
        b.v_b = req(s + ".attn.v.bias");
        b.o_w = req(s + ".attn.o.weight");
        b.o_b = req(s + ".attn.o.bias");
        b.ffn_ln_w = req(s + ".ffn_ln.weight");
        b.ffn_ln_b = req(s + ".ffn_ln.bias");
        b.fc1_w = req(s + ".fc1.weight");
        b.fc1_b = req(s + ".fc1.bias");
        b.fc2_w = req(s + ".fc2.weight");
        b.fc2_b = req(s + ".fc2.bias");
    }
    m.adapter_ln_w = req("adapter.ln.weight");
    m.adapter_ln_b = req("adapter.ln.bias");
    m.adapter_fc1_w = req("adapter.fc1.weight");
    m.adapter_fc1_b = req("adapter.fc1.bias");
    m.adapter_fc2_w = req("adapter.fc2.weight");
    m.adapter_fc2_b = req("adapter.fc2.bias");
    m.embed_w = req("dec.embed.weight");
    m.norm_w = req("dec.norm.weight");
    m.dec.resize(hp.llm_layers);
    for (uint32_t i = 0; i < hp.llm_layers; i++) {
        char p[48];
        snprintf(p, sizeof(p), "dec.blk.%u", i);
        std::string s(p);
        auto& b = m.dec[i];
        b.attn_norm_w = req(s + ".attn_norm.weight");
        b.q_w = req(s + ".attn.q.weight");
        b.q_b = req(s + ".attn.q.bias");
        b.k_w = req(s + ".attn.k.weight");
        b.k_b = req(s + ".attn.k.bias");
        b.v_w = req(s + ".attn.v.weight");
        b.v_b = req(s + ".attn.v.bias");
        b.o_w = req(s + ".attn.o.weight");
        b.ffn_norm_w = req(s + ".ffn_norm.weight");
        b.ffn_gate_w = req(s + ".ffn.gate.weight");
        b.ffn_up_w = req(s + ".ffn.up.weight");
        b.ffn_down_w = req(s + ".ffn.down.weight");
    }
    (void)opt;

    // Validate all bound (require() already logged any miss).
    for (auto* t : {m.conv1_w, m.conv2_w, m.embed_w, m.norm_w, m.mel_filters, m.mel_window})
        ok = ok && (t != nullptr);
    if (!ok) {
        fprintf(stderr, "ark_asr: missing required tensors\n");
        ark_asr_free(ctx);
        return nullptr;
    }

    // Scheduler
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1)
        fprintf(stderr, "ark_asr: loaded %zu tensors  dec=%uL/%u  enc=%uL/%u  vocab=%zu merges=%zu\n",
                ctx->tensors.size(), hp.llm_layers, hp.llm_hidden, hp.enc_layers, hp.enc_d_model, ctx->vocab.size(),
                ctx->merge_rank.size());
    return ctx;
}

extern "C" void ark_asr_set_language(struct ark_asr_context* ctx, const char* lang_iso) {
    if (ctx && lang_iso)
        ctx->language = lang_iso;
}

extern "C" void ark_asr_set_ask(struct ark_asr_context* ctx, const char* instruction) {
    if (ctx)
        ctx->ask = instruction ? instruction : "";
}

// ===========================================================================
// Mel
// ===========================================================================
extern "C" float* ark_asr_compute_mel(struct ark_asr_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                      int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->hp;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int n_mels = (int)hp.n_mels;
    const int n_freqs = n_fft / 2 + 1;

    std::vector<float> hann((size_t)n_fft);
    ggml_backend_tensor_get(ctx->model.mel_window, hann.data(), 0, (size_t)n_fft * sizeof(float));
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->model.mel_filters, filt.data(), 0, filt.size() * sizeof(float));

    // Stock WhisperFeatureExtractor: log10 + max-clip guard, double-accum
    // matmul, drop last STFT frame, fb in (n_freqs, n_mels) layout.
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
    auto mel = core_mel::compute(samples, n_samples, hann.data(), n_fft, filt.data(), n_freqs, ark_fft, p, T_ret);
    if (mel.empty())
        return nullptr;
    if (out_n_mels)
        *out_n_mels = n_mels;
    if (out_T_mel)
        *out_T_mel = T_ret;
    float* r = (float*)malloc(mel.size() * sizeof(float));
    std::memcpy(r, mel.data(), mel.size() * sizeof(float));
    return r;
}

// ===========================================================================
// Graph helpers
// ===========================================================================
static ggml_tensor* ark_layernorm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    ggml_tensor* h = ggml_norm(c, x, eps);
    h = ggml_mul(c, h, w);
    h = ggml_add(c, h, b);
    return h;
}

static ggml_tensor* ark_bias2d(ggml_context* c, ggml_tensor* bias) {
    // reshape a 1D [C] bias to [1, C] so it broadcasts over the time axis of
    // a conv output [T, C].
    return ggml_reshape_2d(c, bias, 1, bias->ne[0]);
}

// Whisper encoder self-attention with partial interleaved RoPE (NORMAL,
// n_dims=rot_dim) on Q+K, no mask (bidirectional), biased q/v/o, no k bias.
static ggml_tensor* ark_enc_attn(ggml_context* c, ggml_tensor* x, const ark_enc_block& b, const ark_asr_hp& hp,
                                 ggml_tensor* positions) {
    const int hd = (int)hp.enc_head_dim;
    const int nh = (int)hp.enc_heads;
    const int T = (int)x->ne[1];
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_tensor* Q = ggml_add(c, ggml_mul_mat(c, b.q_w, x), b.q_b);
    ggml_tensor* K = ggml_mul_mat(c, b.k_w, x); // no bias
    ggml_tensor* V = ggml_add(c, ggml_mul_mat(c, b.v_w, x), b.v_b);

    Q = ggml_reshape_3d(c, Q, hd, nh, T);
    K = ggml_reshape_3d(c, K, hd, nh, T);
    V = ggml_reshape_3d(c, V, hd, nh, T);

    Q = ggml_rope_ext(c, Q, positions, nullptr, (int)hp.enc_rot_dim, GGML_ROPE_TYPE_NORMAL, 0, hp.enc_rope_theta, 1.0f,
                      0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(c, K, positions, nullptr, (int)hp.enc_rot_dim, GGML_ROPE_TYPE_NORMAL, 0, hp.enc_rope_theta, 1.0f,
                      0.0f, 1.0f, 0.0f, 0.0f);

    Q = ggml_cont(c, ggml_permute(c, Q, 0, 2, 1, 3)); // (hd, T, nh)
    K = ggml_cont(c, ggml_permute(c, K, 0, 2, 1, 3));
    V = ggml_cont(c, ggml_permute(c, V, 0, 2, 1, 3));

    ggml_tensor* attn = ggml_flash_attn_ext(c, Q, K, V, nullptr, scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(c, attn, hd * nh, T);
    attn = ggml_add(c, ggml_mul_mat(c, b.o_w, attn), b.o_b);
    return attn;
}

// Build the encoder+adapter graph. mel input is [T_mel, n_mels]. Output tensor
// "audio_embeds" has shape [llm_hidden, N] where N = T_enc / merge_factor.
static ggml_cgraph* ark_build_encoder_graph(ark_asr_context* ctx, int T_mel) {
    const auto& hp = ctx->hp;
    const auto& m = ctx->model;
    const int d = (int)hp.enc_d_model;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* c = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(c, 16384, false);

    ggml_tensor* mel = ggml_new_tensor_2d(c, GGML_TYPE_F32, T_mel, n_mels);
    ggml_set_input(mel);
    ggml_set_name(mel, "mel");

    // conv stem: conv1 (k3 s1 p1) gelu, conv2 (k3 s2 p1) gelu.
    ggml_tensor* cur = ggml_conv_1d(c, m.conv1_w, mel, 1, 1, 1); // [T_mel, d]
    cur = ggml_add(c, cur, ark_bias2d(c, m.conv1_b));
    cur = ggml_gelu_erf(c, cur);
    cur = ggml_conv_1d(c, m.conv2_w, cur, 2, 1, 1); // [T_enc, d]
    cur = ggml_add(c, cur, ark_bias2d(c, m.conv2_b));
    cur = ggml_gelu_erf(c, cur);
    const int T_enc = (int)cur->ne[0];
    cur = ggml_cont(c, ggml_transpose(c, cur)); // [d, T_enc]

    ggml_tensor* positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, T_enc);
    ggml_set_input(positions);
    ggml_set_name(positions, "enc_positions");

    for (uint32_t il = 0; il < hp.enc_layers; il++) {
        const auto& b = m.enc[il];
        ggml_tensor* res = cur;
        ggml_tensor* h = ark_layernorm(c, cur, b.attn_ln_w, b.attn_ln_b, hp.enc_ln_eps);
        h = ark_enc_attn(c, h, b, hp, positions);
        cur = ggml_add(c, res, h);
        res = cur;
        h = ark_layernorm(c, cur, b.ffn_ln_w, b.ffn_ln_b, hp.enc_ln_eps);
        h = core_ffn::gelu_erf_ffn(c, h, b.fc1_w, b.fc1_b, b.fc2_w, b.fc2_b);
        cur = ggml_add(c, res, h);
    }
    // no final encoder LN (nn.Identity upstream)

    // adapter: LayerNorm -> merge-4 -> Linear -> GELU -> Linear
    cur = ark_layernorm(c, cur, m.adapter_ln_w, m.adapter_ln_b, hp.enc_ln_eps); // [d, T_enc]
    const int mf = (int)hp.merge_factor;
    const int T4 = (T_enc / mf) * mf;
    if (T4 != T_enc) {
        cur = ggml_cont(c, ggml_view_2d(c, cur, d, T4, cur->nb[1], 0)); // [d, T4]
    }
    const int N = T4 / mf;
    cur = ggml_reshape_2d(c, cur, d * mf, N); // [d*mf, N]
    cur = ggml_add(c, ggml_mul_mat(c, m.adapter_fc1_w, cur), m.adapter_fc1_b);
    cur = ggml_gelu_erf(c, cur);
    cur = ggml_add(c, ggml_mul_mat(c, m.adapter_fc2_w, cur), m.adapter_fc2_b); // [hidden, N]
    ggml_set_name(cur, "audio_embeds");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// Returns malloc'd [hidden, N] audio embeddings; sets *out_hidden and *out_n.
static float* ark_encode(ark_asr_context* ctx, const float* mel, int T_mel, int* out_hidden, int* out_n) {
    ggml_cgraph* gf = ark_build_encoder_graph(ctx, T_mel);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_t, mel, 0, (size_t)ggml_nelements(mel_t) * sizeof(float));
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "enc_positions");
    {
        std::vector<int32_t> p((size_t)pos_t->ne[0]);
        for (int i = 0; i < (int)p.size(); i++)
            p[i] = i;
        ggml_backend_tensor_set(pos_t, p.data(), 0, p.size() * sizeof(int32_t));
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* out = ggml_graph_get_tensor(gf, "audio_embeds");
    const int hidden = (int)out->ne[0];
    const int N = (int)out->ne[1];
    float* r = (float*)malloc((size_t)hidden * N * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)hidden * N * sizeof(float));
    if (out_hidden)
        *out_hidden = hidden;
    if (out_n)
        *out_n = N;
    return r;
}

extern "C" float* ark_asr_run_encoder(struct ark_asr_context* ctx, const float* pcm, int n_samples, int* out_hidden,
                                      int* out_n) {
    int n_mels = 0, T_mel = 0;
    float* mel = ark_asr_compute_mel(ctx, pcm, n_samples, &n_mels, &T_mel);
    if (!mel)
        return nullptr;
    float* r = ark_encode(ctx, mel, T_mel, out_hidden, out_n);
    free(mel);
    return r;
}

// ===========================================================================
// Decoder graph (prefill T>1 with audio injection, or step T=1)
// ===========================================================================
static ggml_cgraph* ark_build_decoder_graph(ark_asr_context* ctx, int T, int n_past, bool inject_audio) {
    const auto& hp = ctx->hp;
    const auto& m = ctx->model;
    const int hidden = (int)hp.llm_hidden;
    const float eps = hp.llm_rms_eps;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* c = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(c, 16384, false);

    ggml_tensor* text_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_input(text_ids);
    ggml_set_name(text_ids, "text_ids");

    ggml_tensor* cur = ggml_get_rows(c, m.embed_w, text_ids); // [hidden, T]

    if (inject_audio) {
        ggml_tensor* keep = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, T);
        ggml_set_input(keep);
        ggml_set_name(keep, "keep_mask");
        ggml_tensor* audio = ggml_new_tensor_2d(c, GGML_TYPE_F32, hidden, T);
        ggml_set_input(audio);
        ggml_set_name(audio, "audio_features");
        cur = ggml_add(c, ggml_mul(c, cur, keep), audio);
    }
    ggml_set_name(cur, "inputs_embeds");

    ggml_tensor* positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_input(positions);
    ggml_set_name(positions, "lm_positions");

    const int Lk = n_past + T;
    ggml_tensor* mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, Lk, T);
    ggml_set_input(mask);
    ggml_set_name(mask, "lm_causal_mask");

    core_attn::KvSelfAttnParams ap{};
    ap.n_heads = (int)hp.llm_heads;
    ap.n_kv_heads = (int)hp.llm_kv_heads;
    ap.head_dim = (int)hp.llm_head_dim;
    ap.n_kv_grp = (int)(hp.llm_heads / hp.llm_kv_heads);
    ap.n_ctx_orig = (int)hp.llm_max_pos;
    ap.rope_theta = hp.llm_rope_theta;
    ap.rope_beta_fast = 32.0f;
    ap.rope_beta_slow = 1.0f;
    ap.attn_scale = 1.0f / std::sqrt((float)hp.llm_head_dim);
    ap.gqa_mode = core_attn::GQA_MANUAL_CONT;
    ap.rope_type = GGML_ROPE_TYPE_NEOX;

    for (uint32_t il = 0; il < hp.llm_layers; il++) {
        const auto& b = m.dec[il];
        ggml_tensor* res = cur;
        ggml_tensor* h = ggml_mul(c, ggml_rms_norm(c, cur, eps), b.attn_norm_w);
        ggml_tensor* attn = core_attn::kv_self_attn(c, gf, h, b.q_w, b.k_w, b.v_w, b.o_w, nullptr, nullptr, positions,
                                                    mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, ap, nullptr, 0,
                                                    nullptr, b.q_b, b.k_b, b.v_b, nullptr, nullptr);
        cur = ggml_add(c, res, attn);
        res = cur;
        h = ggml_mul(c, ggml_rms_norm(c, cur, eps), b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(c, h, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(c, res, mlp);
    }
    cur = ggml_mul(c, ggml_rms_norm(c, cur, eps), m.norm_w);

    // logits at the last position only.
    ggml_tensor* last = ggml_view_2d(c, cur, hidden, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    last = ggml_cont(c, last);
    ggml_tensor* logits = ggml_mul_mat(c, m.embed_w, last); // tied lm_head -> [vocab, 1]
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

// Fill an F16 causal mask [Lk, T] (row q, col k): 0 if k <= n_past+q else -inf.
static void ark_fill_mask(std::vector<ggml_fp16_t>& mask, int T, int n_past) {
    const int Lk = n_past + T;
    mask.resize((size_t)T * Lk);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < T; q++)
        for (int k = 0; k < Lk; k++)
            mask[(size_t)q * Lk + k] = (k <= n_past + q) ? z : ninf;
}

// Run one decoder graph; returns malloc'd logits [vocab] for the last position.
static float* ark_run_decoder(ark_asr_context* ctx, const int32_t* ids, int T, int n_past, bool inject,
                              const float* audio_features, const float* keep_mask) {
    static const bool timing = std::getenv("CRISPASR_ARKASR_TIMING") != nullptr;
    const int64_t t0 = timing ? ggml_time_us() : 0;
    ggml_cgraph* gf = ark_build_decoder_graph(ctx, T, n_past, inject);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;
    const int64_t t1 = timing ? ggml_time_us() : 0;

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };
    if (!set_t("text_ids", ids, (size_t)T * sizeof(int32_t)))
        return nullptr;
    {
        std::vector<int32_t> p((size_t)T);
        for (int i = 0; i < T; i++)
            p[i] = n_past + i;
        if (!set_t("lm_positions", p.data(), p.size() * sizeof(int32_t)))
            return nullptr;
    }
    {
        std::vector<ggml_fp16_t> mask;
        ark_fill_mask(mask, T, n_past);
        if (!set_t("lm_causal_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)))
            return nullptr;
    }
    if (inject) {
        const int hidden = (int)ctx->hp.llm_hidden;
        if (!set_t("keep_mask", keep_mask, (size_t)T * sizeof(float)))
            return nullptr;
        if (!set_t("audio_features", audio_features, (size_t)hidden * T * sizeof(float)))
            return nullptr;
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    if (timing) {
        const int64_t t2 = ggml_time_us();
        fprintf(stderr, "[ark-timing] T=%d n_past=%d build+alloc=%.2fms compute=%.2fms (%.1f%% build)\n", T, n_past,
                (t1 - t0) / 1000.0, (t2 - t1) / 1000.0, 100.0 * (t1 - t0) / (double)(t2 - t0));
    }

    ggml_tensor* lt = ggml_graph_get_tensor(gf, "logits");
    const int vocab = (int)ctx->hp.llm_vocab;
    float* out = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(lt, out, 0, (size_t)vocab * sizeof(float));
    return out;
}

// Build the ASR prompt + audio-feature/keep buffers from raw PCM. Returns the
// prompt token ids (with N <|audio|> placeholders) and fills feats[hidden*T] /
// keep[T]. Returns false on failure.
static bool ark_build_prefill_inputs(ark_asr_context* ctx, const float* pcm, int n_samples, std::vector<int32_t>& ids,
                                     std::vector<float>& feats, std::vector<float>& keep, int* out_hidden) {
    int n_mels = 0, T_mel = 0;
    float* mel = ark_asr_compute_mel(ctx, pcm, n_samples, &n_mels, &T_mel);
    if (!mel)
        return false;
    int hidden = 0, N = 0;
    float* audio = ark_encode(ctx, mel, T_mel, &hidden, &N);
    free(mel);
    if (!audio || N <= 0) {
        free(audio);
        return false;
    }
    ids.clear();
    ids.reserve((size_t)N + 16);
    ids.push_back(ctx->id_user);
    // EXPERIMENTAL: prepend a tokenised instruction (e.g. language steering)
    // when set. Default (no ask) is promptless — the validated path.
    if (!ctx->ask.empty()) {
        std::vector<int32_t> tids = core_bpe::tokenize_simple(ctx->token_to_id, ctx->merge_rank, ctx->ask);
        ids.insert(ids.end(), tids.begin(), tids.end());
    }
    ids.push_back(ctx->id_boa);
    const int audio_start = (int)ids.size();
    for (int i = 0; i < N; i++)
        ids.push_back(ctx->id_audio);
    ids.push_back(ctx->id_eoa);
    ids.push_back(ctx->id_assistant);
    const int T = (int)ids.size();
    feats.assign((size_t)hidden * T, 0.0f);
    keep.assign((size_t)T, 1.0f);
    for (int i = 0; i < N; i++) {
        const int col = audio_start + i;
        std::memcpy(&feats[(size_t)col * hidden], &audio[(size_t)i * hidden], (size_t)hidden * sizeof(float));
        keep[col] = 0.0f;
    }
    free(audio);
    if (out_hidden)
        *out_hidden = hidden;
    return true;
}

// diff-harness stage: prefill logits at the last prompt position ([vocab]).
extern "C" float* ark_asr_prefill_logits(struct ark_asr_context* ctx, const float* pcm, int n_samples, int* out_vocab) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;
    std::vector<int32_t> ids;
    std::vector<float> feats, keep;
    if (!ark_build_prefill_inputs(ctx, pcm, n_samples, ids, feats, keep, nullptr))
        return nullptr;
    const int T = (int)ids.size();
    if (!ark_kv_init(ctx, T + 16))
        return nullptr;
    float* logits = ark_run_decoder(ctx, ids.data(), T, 0, true, feats.data(), keep.data());
    if (out_vocab)
        *out_vocab = (int)ctx->hp.llm_vocab;
    return logits;
}

// ===========================================================================
// Transcribe (one <=30s window)
// ===========================================================================
static std::string ark_transcribe_window(ark_asr_context* ctx, const float* pcm, int n_samples, int max_new) {
    const auto& hp = ctx->hp;
    int n_mels = 0, T_mel = 0;
    float* mel = ark_asr_compute_mel(ctx, pcm, n_samples, &n_mels, &T_mel);
    if (!mel)
        return std::string();
    int hidden = 0, N = 0;
    float* audio = ark_encode(ctx, mel, T_mel, &hidden, &N);
    free(mel);
    if (!audio || N <= 0) {
        free(audio);
        return std::string();
    }

    // Prompt: <|user|> <|begin_of_audio|> <|audio|>xN <|end_of_audio|> <|assistant|>
    std::vector<int32_t> ids;
    ids.reserve((size_t)N + 8);
    ids.push_back(ctx->id_user);
    ids.push_back(ctx->id_boa);
    const int audio_start = (int)ids.size();
    for (int i = 0; i < N; i++)
        ids.push_back(ctx->id_audio);
    ids.push_back(ctx->id_eoa);
    ids.push_back(ctx->id_assistant);
    const int T = (int)ids.size();

    // audio_features [hidden, T] (zeros except audio positions) + keep_mask [T].
    std::vector<float> feats((size_t)hidden * T, 0.0f);
    std::vector<float> keep((size_t)T, 1.0f);
    for (int i = 0; i < N; i++) {
        const int col = audio_start + i;
        std::memcpy(&feats[(size_t)col * hidden], &audio[(size_t)i * hidden], (size_t)hidden * sizeof(float));
        keep[col] = 0.0f;
    }
    free(audio);

    if (!ark_kv_init(ctx, T + max_new + 16))
        return std::string();
    core_gguf::mmap_advise_random(ctx->buf_w);

    const int vocab = (int)hp.llm_vocab;
    auto argmax = [&](const float* L) {
        int best = 0;
        float bv = L[0];
        for (int i = 1; i < vocab; i++)
            if (L[i] > bv) {
                bv = L[i];
                best = i;
            }
        return best;
    };

    float* logits = ark_run_decoder(ctx, ids.data(), T, 0, true, feats.data(), keep.data());
    if (!logits)
        return std::string();
    int next = argmax(logits);
    free(logits);

    std::vector<int32_t> gen;
    gen.reserve((size_t)max_new);
    gen.push_back(next);
    int n_past = T;
    for (int step = 1; step < max_new; step++) {
        if (next == ctx->id_im_end)
            break;
        int32_t tok = next;
        float* L = ark_run_decoder(ctx, &tok, 1, n_past, false, nullptr, nullptr);
        if (!L)
            break;
        next = argmax(L);
        free(L);
        n_past++;
        gen.push_back(next);
    }
    if (!gen.empty() && gen.back() == ctx->id_im_end)
        gen.pop_back();

    std::string txt = core_bpe::detokenize(ctx->vocab, gen.data(), gen.size());
    // trim leading whitespace
    size_t lead = 0;
    while (lead < txt.size() && (txt[lead] == ' ' || txt[lead] == '\t' || txt[lead] == '\n'))
        lead++;
    txt.erase(0, lead);
    return txt;
}

// Budget for greedy decode: output tokens are ~0.3x the audio-token count, so
// the audio-token estimate (n_samples / (hop*2*merge) = n_samples/1280) is a
// safe upper bound with headroom. Clamp to [256, 4096] (floor for short clips,
// backstop against a no-EOS runaway).
static int ark_max_new_for(int n_samples) {
    int est = n_samples / 1280 + 64;
    if (est < 256)
        est = 256;
    if (est > 4096)
        est = 4096;
    return est;
}

extern "C" char* ark_asr_transcribe(struct ark_asr_context* ctx, const float* pcm, int n_samples) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;
    const int sr = (int)ctx->hp.sample_rate;

    // Single-pass whole-audio, matching the reference (modeling_arkasr.py): the
    // RoPE encoder has no positional cap, so the whole clip is one encoder pass +
    // one decode. This avoids the per-window language drift that independent
    // chunks cause. The encoder uses flash-attn (bounded memory), but attention
    // is O(T^2) compute and decode is O(transcript) — so cap single-pass length
    // and fall back to internal chunking for very long audio to stay off the
    // 16 GB M1's OOM/slowdown cliff. Override the cap (seconds) with
    // CRISPASR_ARKASR_MAX_SINGLE_PASS_S (0 = never chunk).
    int cap_s = 300;
    if (const char* e = std::getenv("CRISPASR_ARKASR_MAX_SINGLE_PASS_S"))
        cap_s = std::atoi(e);
    const long cap_samples = (long)cap_s * sr;

    std::string full;
    if (cap_s <= 0 || (long)n_samples <= cap_samples) {
        full = ark_transcribe_window(ctx, pcm, n_samples, ark_max_new_for(n_samples));
    } else {
        // Fallback for very long audio: internal 30 s windows (re-introduces the
        // per-window drift, but bounds memory/time). Pass --vad for cleaner
        // silence-bounded segments, or raise CRISPASR_ARKASR_MAX_SINGLE_PASS_S.
        const int win = 30 * sr;
        for (int off = 0; off < n_samples; off += win) {
            const int len = std::min(win, n_samples - off);
            std::string seg = ark_transcribe_window(ctx, pcm + off, len, ark_max_new_for(len));
            if (!seg.empty()) {
                if (!full.empty())
                    full += " ";
                full += seg;
            }
        }
    }
    char* out = (char*)malloc(full.size() + 1);
    if (out)
        std::memcpy(out, full.c_str(), full.size() + 1);
    return out;
}

extern "C" void ark_asr_free(struct ark_asr_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->buf_w_cpu)
        ggml_backend_buffer_free(ctx->buf_w_cpu);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}
