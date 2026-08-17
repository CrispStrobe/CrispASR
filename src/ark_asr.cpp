// src/ark_asr.cpp — ARK-ASR-3B runtime (see PLAN.md §ARK).
//
// Pipeline: Whisper mel -> Whisper-large-v3 encoder with partial interleaved
// RoPE (rot_dim 32 of head_dim 64, theta 10000) + no final LN -> MLP adapter
// (LayerNorm -> merge-4 -> Linear 5120->4096 -> GELU -> Linear 4096->2048) ->
// audio embeddings injected at <|audio|> placeholder positions in a Qwen2.5-3B
// decoder, greedy-decoded to text.

#include "ark_asr.h"

#include "ggml-backend.h"
#include "crispasr_imatrix.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>

#include "core/attention.h"
#include "core/beam_decode.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/mel.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/ggml_cpu_backend.h"

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
    // Upstream's bad_words_ids: every special / <|…|> added token except EOS,
    // banned at EVERY decode step. Built once at init (see ark_build_banned_ids).
    std::vector<int32_t> banned_ids;

    std::string language;
    std::string ask; // EXPERIMENTAL transcription instruction (see ark_asr_set_ask)
};

// Defined below with the decode helpers; called from init once the vocab and
// the special-token ids are resolved.
static void ark_build_banned_ids(ark_asr_context* ctx);

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
    // #367: size and place via ggml, not ggml_nbytes() arithmetic. CUDA's
    // get_alloc_size() pads a quantized row up to MATRIX_ROW_PADDING (512),
    // and these KV rows are head_dim wide (128), so each q8_0 tensor needs
    // 408 bytes more than nbytes — the hand-sized buffer came up short and
    // ggml_backend_tensor_alloc aborted. f16 is not quantized, so this only
    // ever fired with CRISPASR_KV_QUANT set, and only on CUDA.
    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, kv_be);
    if (!ctx->kv_buf)
        return false;
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
    ark_build_banned_ids(ctx);
    ctx->id_im_end = (int32_t)hp.eos_token_id;

    // Backends
    ctx->backend_cpu = core_cpu_backend::init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "ark_asr: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->n_threads);
    // GPU default. Validated verbatim on M1 Metal (en jfk + de De-Abwasch,
    // 2026-06-29): ~5.6x faster prefill, ~neutral per-token decode (single-token
    // decode is bandwidth/dispatch-bound on unified memory), ~1.7x faster overall
    // on jfk. (An earlier port build "emitted no tokens" on GPU; the current
    // flash-attn + KV path no longer reproduces it.) Force CPU with
    // CRISPASR_ARKASR_CPU=1. CUDA/other GPUs not yet validated.
    const bool force_cpu = std::getenv("CRISPASR_ARKASR_CPU") != nullptr;
    if (params.use_gpu && !force_cpu) {
        ctx->backend = crispasr_init_gpu_backend();
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
        crispasr_imatrix_install(ctx->sched); // no-op unless CRISPASR_IMATRIX_OUT is set
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

// Upstream's `bad_words_ids`, which its README builds as
//
//     bad = set(tokenizer.all_special_ids) - {eos}
//     bad |= {id for tok, id in tokenizer.get_added_vocab().items()
//             if tok.startswith("<") and tok.endswith(">") and id != eos}
//
// and passes to every generate() call, so the ban applies at EVERY step, not
// just the first. We had no equivalent: nothing stopped ark emitting <|audio|>
// or <|user|> mid-transcript, which decodes to either literal junk or (with
// skip_special_tokens-style cleanup) a silent hole in the text.
//
// Approximated from the GGUF vocab by shape — `<|…|>` — because the added-token
// table upstream reads is not carried into the GGUF. That covers Qwen2.5's
// specials and ark's five extra role/audio markers. EOS is kept: it is how
// generation stops. Disable with CRISPASR_ARKASR_NO_SPECIAL_SUPPRESS=1.
static void ark_build_banned_ids(ark_asr_context* ctx) {
    ctx->banned_ids.clear();
    if (std::getenv("CRISPASR_ARKASR_NO_SPECIAL_SUPPRESS") != nullptr)
        return;
    for (size_t i = 0; i < ctx->vocab.size(); i++) {
        const std::string& t = ctx->vocab[i];
        if (t.size() >= 4 && t.compare(0, 2, "<|") == 0 && t.compare(t.size() - 2, 2, "|>") == 0 &&
            (int32_t)i != ctx->id_im_end)
            ctx->banned_ids.push_back((int32_t)i);
    }
}

// Upstream additionally blocks every id at or above `asr_block_token_id_from`
// (scripts/infer/ark_asr_transformers.py, default 151670) via
// BlockTokenIdsFromLogitsProcessor. The text vocabulary ends at 151669
// (<|system|>); everything above it is bicodec / semantic-token space that the
// shared multi-task vocabulary carries but ASR must never emit. Our vocab is
// 151936, so 266 ids were emittable that upstream forbids. Override or disable
// with CRISPASR_ARKASR_BLOCK_FROM_ID (negative = off).
static int ark_block_from_id() {
    static const int s_from = []() {
        const char* e = std::getenv("CRISPASR_ARKASR_BLOCK_FROM_ID");
        return e ? atoi(e) : 151670;
    }();
    return s_from;
}

// Apply both bans to one logits row, in place.
static void ark_mask_specials(const ark_asr_context* ctx, float* logits) {
    const int vocab = (int)ctx->hp.llm_vocab;
    for (int32_t id : ctx->banned_ids)
        if (id >= 0 && id < vocab)
            logits[id] = -INFINITY;
    const int from = ark_block_from_id();
    if (from >= 0)
        for (int i = from; i < vocab; i++)
            logits[i] = -INFINITY;
}

// Append the user's text instruction, which upstream places BETWEEN
// <|end_of_audio|> and <|assistant|>.
//
// ArkAsrProcessor._build_templates_and_audios concatenates each content part in
// order, so the documented ASR conversation
//
//     {"role": "user", "content": [{"type": "audio", ...},
//                                  {"type": "text", "text": "Please transcribe this audio."}]}
//
// renders as
//
//     <|user|><|begin_of_audio|>…<|end_of_audio|>Please transcribe this audio.<|assistant|>
//
// We used to jump straight from <|end_of_audio|> to <|assistant|>, i.e. decode
// with a prompt the model never saw in training. It mostly worked, which is why
// it survived — but it left the first decode step marginal, and a marginal step
// is one any small numeric perturbation can tip. That is the whole shape of
// the ark empty-transcript symptom: <im_end> right after a single "." token,
// flipping with the KV cache
// dtype and with audio length, on a model that is otherwise fine. The existing
// `ctx->ask` knob is NOT this: it inserts text BEFORE <|begin_of_audio|>, which
// is a different position from upstream's, and it is off by default.
//
// Precedence: ark_asr_set_ask() > CRISPASR_ARKASR_INSTRUCTION > upstream's
// default. Setting any of them empty restores the old promptless behaviour.
//
// `ask` used to be spliced in before <|begin_of_audio|> and ONLY in
// ark_build_prefill_inputs — ark_transcribe_window never looked at it — so the
// caller-supplied instruction reached the diff/logits harness and never a real
// transcript. Routing it here fixes both halves at once: it now applies to
// transcription, and it lands in the slot upstream actually puts text in.
static void ark_push_instruction(ark_asr_context* ctx, std::vector<int32_t>& ids) {
    // The env var is an explicit operator override, so it outranks `ask` —
    // which the CLI derives automatically from the detected language and would
    // otherwise always set, leaving the env knob unreachable in practice.
    // Present-but-empty is a deliberate "no instruction at all".
    static const bool s_env_set = std::getenv("CRISPASR_ARKASR_INSTRUCTION") != nullptr;
    static const std::string s_env = s_env_set ? std::getenv("CRISPASR_ARKASR_INSTRUCTION") : std::string();
    static const std::string s_default = "Please transcribe this audio.";

    const std::string& instruction = s_env_set ? s_env : (!ctx->ask.empty() ? ctx->ask : s_default);
    if (instruction.empty())
        return;
    const std::vector<int32_t> tids = core_bpe::tokenize_simple(ctx->token_to_id, ctx->merge_rank, instruction);
    if (std::getenv("CRISPASR_ARKASR_DEBUG_GEN")) {
        // tokenize_simple does NOT do GPT-2 regex pre-tokenization (its own
        // docstring says so): it splits on whitespace only, so a trailing "."
        // stays glued to the last word and can BPE-merge across a boundary the
        // reference tokenizer would have split. Print what we actually feed the
        // model so a non-canonical prompt is visible rather than inferred.
        fprintf(stderr, "[ark-instr] \"%s\" -> %zu tokens:", instruction.c_str(), tids.size());
        for (int32_t t : tids)
            fprintf(stderr, " %d(%s)", t,
                    (t >= 0 && (size_t)t < ctx->vocab.size()) ? ctx->vocab[(size_t)t].c_str() : "?");
        fprintf(stderr, "\n");
    }
    ids.insert(ids.end(), tids.begin(), tids.end());
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
    // `ask` is no longer spliced in here: it goes through ark_push_instruction
    // below, at the position upstream puts text in, so this path and
    // ark_transcribe_window build the same prompt.
    ids.push_back(ctx->id_boa);
    const int audio_start = (int)ids.size();
    for (int i = 0; i < N; i++)
        ids.push_back(ctx->id_audio);
    ids.push_back(ctx->id_eoa);
    ark_push_instruction(ctx, ids);
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

// #253: Qwen2.5's added/special tokens start here (<|endoftext|>=151643); the
// base BPE text vocab is below. The model can emit these on out-of-distribution
// chunks and core_bpe::detokenize would render them as literal text ("endoftext",
// "im_start", ...) — the reference decodes with skip_special_tokens=True.
static constexpr int32_t ARK_FIRST_SPECIAL_TOKEN = 151643;

// #253: collapse a phrase repeated >=3x consecutively down to a single copy.
// The cross-chunk language seed (ark_transcribe_chunked) can make an
// uninformative window (silence/music/OOD) echo the seeded phrase, which greedy
// decode then loops. Real speech never repeats a multi-word phrase verbatim 3+
// times, so this is safe; applied per window so the seed carried to the next
// window stays clean and the loop can't grow across windows. Returns the input
// unchanged when no loop is present (preserves original spacing).
static std::string ark_deloop(const std::string& s) {
    std::vector<std::string> w;
    for (size_t i = 0; i < s.size();) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n'))
            i++;
        size_t j = i;
        while (j < s.size() && s[j] != ' ' && s[j] != '\t' && s[j] != '\n')
            j++;
        if (j > i)
            w.emplace_back(s.substr(i, j - i));
        i = j;
    }
    if (w.size() < 4)
        return s;
    std::vector<std::string> out;
    out.reserve(w.size());
    bool changed = false;
    size_t i = 0;
    while (i < w.size()) {
        int bestk = 0;
        size_t bestreps = 0;
        // Scan phrase lengths up to a full sentence — seed-echo loops repeat an
        // entire clause (observed ~16 words), which a short k window can't see.
        for (int k = 1; k <= 40 && i + (size_t)2 * k <= w.size(); k++) {
            size_t reps = 1, j = i + (size_t)k;
            while (j + (size_t)k <= w.size() && std::equal(w.begin() + i, w.begin() + i + k, w.begin() + j)) {
                reps++;
                j += (size_t)k;
            }
            // >=3 reps of any phrase, OR >=2 reps of a >=4-word phrase (a long
            // verbatim repeat is a loop, not natural speech — a cross-window seed
            // echo shows up as the phrase appearing exactly twice back-to-back).
            if ((reps >= 3 || (reps >= 2 && k >= 4)) && reps * (size_t)k > bestreps * (size_t)bestk) {
                bestk = k;
                bestreps = reps;
            }
        }
        if (bestk) {
            for (int m = 0; m < bestk; m++)
                out.push_back(w[i + (size_t)m]); // keep one copy of the looped phrase
            i += bestreps * (size_t)bestk;
            changed = true;
        } else {
            out.push_back(w[i]);
            i++;
        }
    }
    if (!changed)
        return s;
    std::string res;
    for (size_t k = 0; k < out.size(); k++) {
        if (k)
            res += ' ';
        res += out[k];
    }
    return res;
}

// ===========================================================================
// Transcribe (one <=30s window)
// ===========================================================================
static std::string ark_transcribe_window(ark_asr_context* ctx, const float* pcm, int n_samples, int max_new,
                                         const std::vector<int32_t>& prefix) {
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

    // Prompt, mirroring upstream's ArkAsrProcessor._build_templates_and_audios:
    //   <|user|> <|begin_of_audio|> <|audio|>xN <|end_of_audio|> INSTRUCTION <|assistant|>
    std::vector<int32_t> ids;
    ids.reserve((size_t)N + 16);
    ids.push_back(ctx->id_user);
    ids.push_back(ctx->id_boa);
    const int audio_start = (int)ids.size();
    for (int i = 0; i < N; i++)
        ids.push_back(ctx->id_audio);
    ids.push_back(ctx->id_eoa);
    ark_push_instruction(ctx, ids);
    ids.push_back(ctx->id_assistant);
    // Cross-chunk conditioning: seed the assistant turn with the tail of the
    // previous chunk's transcript so the model continues in the same language
    // (the chunked fallback otherwise re-detects language per window — a German
    // clip's next chunk would get *translated* to English). These tokens are
    // context only; generation continues after them and we keep just the new
    // tokens. Empty for the first chunk / single-pass. See ark_asr_transcribe.
    for (int32_t t : prefix)
        ids.push_back(t);
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

    // #253: some windows (esp. short ones) degenerate — the model emits <im_end>
    // as the very FIRST token, yielding an empty transcript for clearly-audible
    // speech. Suppress <im_end> on the first step so the decoder must emit real
    // content (Whisper's "suppress-EOT-at-start"); later steps allow <im_end>
    // normally, so genuine endings are unaffected. On a truly empty/silent window
    // the forced first token is the model's opening "." which the output cleanup
    // strips back to empty, so this doesn't invent words out of silence. Opt out
    // with CRISPASR_ARKASR_NO_EOS_SUPPRESS=1.
    if (ctx->id_im_end >= 0 && ctx->id_im_end < (int)ctx->hp.llm_vocab &&
        std::getenv("CRISPASR_ARKASR_NO_EOS_SUPPRESS") == nullptr)
        logits[ctx->id_im_end] = -INFINITY;
    // Upstream bans the other special tokens at every step (see
    // ark_build_banned_ids); this is the first of those steps.
    ark_mask_specials(ctx, logits);

    std::vector<int32_t> gen;
    gen.reserve((size_t)max_new);
    const int beam = ctx->params.beam_size > 0 ? ctx->params.beam_size : 1;

    if (beam > 1) {
        // Beam search via core_beam_decode replay-from-prefix. The prompt's K/V
        // (incl. the injected audio frames) occupy slots [0, T); each beam-step
        // rebuilds its suffix by replaying from that anchor. ark_run_decoder
        // embeds the token ids itself and returns the last-position logits.
        auto replay = [](ark_asr_context* c, const int32_t* toks, int n, int prompt_len) -> float* {
            float* L = ark_run_decoder(c, toks, n, prompt_len, false, nullptr, nullptr);
            if (L)
                ark_mask_specials(c, L); // same ban as the greedy path
            return L;
        };
        core_beam_decode::Config cfg;
        cfg.max_new_tokens = max_new;
        cfg.eos_id = ctx->id_im_end;
        cfg.vocab_size = vocab;
        cfg.beam_size = beam;
        cfg.prompt_len = T;
        auto br = core_beam_decode::run_with_probs(ctx, logits, replay, cfg);
        logits = nullptr; // ownership transferred to core_beam_decode
        gen = std::move(br.tokens);
        if (!gen.empty() && gen.back() == ctx->id_im_end)
            gen.pop_back();
    } else {
        int next = argmax(logits);
        free(logits);
        gen.push_back(next);
        int n_past = T;
        for (int step = 1; step < max_new; step++) {
            if (next == ctx->id_im_end)
                break;
            int32_t tok = next;
            float* L = ark_run_decoder(ctx, &tok, 1, n_past, false, nullptr, nullptr);
            if (!L)
                break;
            ark_mask_specials(ctx, L);
            next = argmax(L);
            free(L);
            n_past++;
            gen.push_back(next);
        }
        if (!gen.empty() && gen.back() == ctx->id_im_end)
            gen.pop_back();
    }

    // #253: strip special/added tokens (>= ARK_FIRST_SPECIAL_TOKEN) the model can
    // emit on OOD chunks (<|endoftext|>, <|im_start|>, ...) before detokenising —
    // core_bpe::detokenize renders them as literal text, unlike the reference's
    // tokenizer.decode(skip_special_tokens=True).
    std::vector<int32_t> vis;
    vis.reserve(gen.size());
    for (int32_t t : gen)
        if (t < ARK_FIRST_SPECIAL_TOKEN)
            vis.push_back(t);
    std::string txt = core_bpe::detokenize(ctx->vocab, vis.data(), vis.size());
    if (std::getenv("CRISPASR_ARKASR_DEBUG_GEN")) {
        fprintf(stderr, "[ark-gen] window %.1fs: %zu tokens, first_tok=%d (im_end=%d) raw='%.90s'\n",
                (double)n_samples / (double)ctx->hp.sample_rate, gen.size(), gen.empty() ? -1 : gen[0], ctx->id_im_end,
                txt.c_str());
    }
    // Output cleanup. ARK opens every transcript with a bare "." token (the
    // reference .generate() output shows the same leading ". " — it's the model's
    // trained format, not a bug here), which is noise in an ASR transcript. Trim
    // leading whitespace, then drop a single leading bare period + its trailing
    // whitespace. The numerical diff gate (mel/audio_embeds/first_logits) runs
    // before detokenisation, so this normalisation doesn't affect it.
    auto ltrim = [](std::string& s) {
        size_t i = 0;
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n'))
            i++;
        s.erase(0, i);
    };
    ltrim(txt);
    if (!txt.empty() && txt[0] == '.' && (txt.size() == 1 || txt[1] == ' ' || txt[1] == '\t' || txt[1] == '\n')) {
        txt.erase(0, 1);
        ltrim(txt);
    }
    txt = ark_deloop(txt); // #253: collapse seed-echo repetition loops
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

// Internal 30 s windowed decode. The Whisper encoder was trained on 30 s
// (max_source_positions=1500); a single pass over much longer audio extrapolates
// its RoPE far past that window and the decoder degenerates (issue #253 — often
// to empty). 30 s windows keep the encoder in-distribution. Cross-chunk language
// conditioning carries the previous window's transcript tail forward so the model
// keeps the same language instead of re-detecting per window (disable with
// CRISPASR_ARKASR_NO_CHUNK_CONTEXT=1).
static std::string ark_transcribe_chunked(ark_asr_context* ctx, const float* pcm, int n_samples, int sr) {
    const bool ctx_carry = std::getenv("CRISPASR_ARKASR_NO_CHUNK_CONTEXT") == nullptr;
    const int kPrefixTokens = 32; // tail length to seed (enough to fix language)
    const int win = 30 * sr;
    std::vector<int32_t> prefix;
    std::string full;
    for (int off = 0; off < n_samples; off += win) {
        const int len = std::min(win, n_samples - off);
        std::string seg = ark_transcribe_window(ctx, pcm + off, len, ark_max_new_for(len), prefix);
        if (!seg.empty()) {
            if (!full.empty())
                full += " ";
            full += seg;
            if (ctx_carry) {
                // Seed the next window with the tail of this chunk's transcript.
                std::vector<int32_t> toks = core_bpe::tokenize_simple(ctx->token_to_id, ctx->merge_rank, seg);
                const int keep = std::min((int)toks.size(), kPrefixTokens);
                prefix.assign(toks.end() - keep, toks.end());
            }
        }
    }
    // #253: final pass to collapse a seed echo that spans window boundaries (the
    // same phrase emitted once per window shows up as a back-to-back repeat only
    // in the concatenated transcript, so the per-window de-loop can't see it).
    return ark_deloop(full);
}

extern "C" char* ark_asr_transcribe(struct ark_asr_context* ctx, const float* pcm, int n_samples) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;
    const int sr = (int)ctx->hp.sample_rate;

    // #253: cap single-pass at the Whisper encoder's TRAINING window (30 s /
    // max_source_positions=1500). Feeding a longer clip in one pass extrapolates
    // the encoder's RoPE far past 1500 frames and the decoder degenerates — often
    // to an empty transcript (reproduced: a 118 s clip → no text; the same clip in
    // 30 s windows → full transcript). So decode audio > 30 s in 30 s windows,
    // with cross-chunk language conditioning (below) carrying the language forward
    // to avoid the per-window drift that motivated the old whole-audio pass.
    // Raise CRISPASR_ARKASR_MAX_SINGLE_PASS_S to force a longer single pass
    // (0 = never chunk); it will degrade past ~30 s.
    int cap_s = 30;
    if (const char* e = std::getenv("CRISPASR_ARKASR_MAX_SINGLE_PASS_S"))
        cap_s = std::atoi(e);
    const long cap_samples = (long)cap_s * sr;

    const std::vector<int32_t> no_prefix;
    std::string full;
    if (cap_s <= 0 || (long)n_samples <= cap_samples) {
        full = ark_transcribe_window(ctx, pcm, n_samples, ark_max_new_for(n_samples), no_prefix);
        // #253: a single pass over audio well past the encoder's 30 s / 1500-frame
        // training window can degenerate to EMPTY (RoPE extrapolation → the decoder
        // emits nothing). Rather than returning no transcript at all, retry with
        // 30 s windowed decode, which keeps the encoder in-distribution.
        if (full.empty() && (long)n_samples > (long)30 * sr) {
            if (ctx->params.verbosity >= 1)
                fprintf(stderr,
                        "ark_asr: single pass produced no text for %.1f s of audio; "
                        "retrying with 30 s windows (issue #253)\n",
                        (double)n_samples / sr);
            full = ark_transcribe_chunked(ctx, pcm, n_samples, sr);
        }
    } else {
        // Very long audio: skip the doomed single pass, decode in 30 s windows.
        full = ark_transcribe_chunked(ctx, pcm, n_samples, sr);
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
        core_gguf::release_weight_buffer(ctx->buf_w);
    if (ctx->buf_w_cpu)
        core_gguf::release_weight_buffer(ctx->buf_w_cpu);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}
