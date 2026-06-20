// nemotron.cpp — nvidia/nemotron-3.5-asr-streaming-0.6b ggml runtime
//
// Cache-Aware Streaming FastConformer encoder + RNN-T decoder.
// Architecture closely matches parakeet-tdt-0.6b-v3 with these key differences:
//   - Causal downsampling in pre-encode
//   - LayerNorm in conv module (not BatchNorm)
//   - Cache-aware self-attention with configurable context window
//   - Pure RNNT decoder (no TDT durations)
//   - Prompt features for language selection
//   - normalize="NA" in preprocessor (no per-feature z-norm)
//   - xscaling=false
//
// The encoder body reuses core_conformer::BlockWeights and build_block()
// from src/core/fastconformer.h — the same shared code that drives parakeet
// and canary. The cache-aware streaming wrapping (context windowing, state
// management) is handled here rather than in a separate header to keep the
// first implementation simple and contained.

#include "nemotron.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif
#include "gguf.h"

#include "core/fastconformer.h"
#include "core/gguf_loader.h"
#include "core/mel.h"

#if defined(HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

// §176d: env-gated fallback to scalar LSTM/joint loops for A/B testing.
static bool nemotron_force_scalar() {
    static int v = -1;
    if (v < 0)
        v = (std::getenv("NEMOTRON_FORCE_SCALAR") != nullptr) ? 1 : 0;
    return v != 0;
}

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Bench instrumentation — `NEMOTRON_BENCH=1` for per-stage timings.
// ===========================================================================

static bool nemotron_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("NEMOTRON_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct nemotron_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit nemotron_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~nemotron_bench_stage() {
        if (!nemotron_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  nemotron_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct nemotron_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 512;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;
    uint32_t d_model = 1024;
    uint32_t n_layers = 24;
    uint32_t n_heads = 8;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 4096;
    uint32_t subsampling_factor = 8;
    uint32_t subsampling_channels = 256;
    uint32_t conv_kernel = 9;
    bool xscaling = false;
    bool causal_downsampling = true;
    uint32_t pred_hidden = 640;
    uint32_t pred_layers = 2;
    uint32_t joint_hidden = 640;
    uint32_t vocab_size = 13087;
    uint32_t blank_id = 13087;
    uint32_t frame_dur_cs = 8;
    uint32_t num_prompts = 128;
    uint32_t prompt_kernel_in = 1152;
    uint32_t prompt_kernel_mid = 2048;

    // Streaming context presets
    uint32_t n_att_context_presets = 4;
    std::vector<int32_t> att_context_left = {56, 56, 56, 56};
    std::vector<int32_t> att_context_right = {3, 0, 6, 13};
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

using nemotron_pre_encode = core_conformer::PreEncodeWeights;

struct nemotron_enc_layer : core_conformer::BlockWeights {
    // LayerNorm in conv module (used instead of BatchNorm)
    ggml_tensor *conv_ln_w = nullptr, *conv_ln_b = nullptr;
};

struct nemotron_predictor {
    ggml_tensor* embed_w = nullptr;
    ggml_tensor *lstm0_w_ih = nullptr, *lstm0_w_hh = nullptr;
    ggml_tensor *lstm0_b_ih = nullptr, *lstm0_b_hh = nullptr;
    ggml_tensor *lstm1_w_ih = nullptr, *lstm1_w_hh = nullptr;
    ggml_tensor *lstm1_b_ih = nullptr, *lstm1_b_hh = nullptr;
};

struct nemotron_joint {
    ggml_tensor *enc_w = nullptr, *enc_b = nullptr;
    ggml_tensor *pred_w = nullptr, *pred_b = nullptr;
    ggml_tensor *out_w = nullptr, *out_b = nullptr;
};

struct nemotron_prompt_kernel {
    ggml_tensor *l0_w = nullptr, *l0_b = nullptr;
    ggml_tensor *l2_w = nullptr, *l2_b = nullptr;
};

// ===========================================================================
// CPU weight caches for predictor LSTM and joint head
// ===========================================================================

struct nemotron_predictor_weights {
    std::vector<float> embed;
    std::vector<float> w_ih_0, w_hh_0, b_ih_0, b_hh_0;
    std::vector<float> w_ih_1, w_hh_1, b_ih_1, b_hh_1;
    int H = 0;
    bool initialised = false;
};

struct nemotron_joint_weights {
    std::vector<float> enc_w, enc_b;
    std::vector<float> pred_w, pred_b;
    std::vector<float> out_w, out_b;
    int joint_hidden = 0;
    int d_model = 0;
    int pred_hidden = 0;
    int vocab_total = 0;
    bool initialised = false;
};

// ===========================================================================
// LSTM state for RNN-T predictor
// ===========================================================================

struct nemotron_lstm_state {
    std::vector<float> h0, c0;
    std::vector<float> h1, c1;
    void init(int H) {
        h0.assign(H, 0.0f);
        c0.assign(H, 0.0f);
        h1.assign(H, 0.0f);
        c1.assign(H, 0.0f);
    }
};

// ===========================================================================
// Model and vocabulary
// ===========================================================================

struct nemotron_model {
    nemotron_hparams hparams;

    ggml_tensor* mel_fb = nullptr;
    ggml_tensor* mel_window = nullptr;

    nemotron_pre_encode pre_encode;
    std::vector<nemotron_enc_layer> enc;
    nemotron_predictor predictor;
    nemotron_joint joint;
    nemotron_prompt_kernel prompt_kernel;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::map<std::string, ggml_tensor*> tensors;
};

struct nemotron_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
};

struct nemotron_context {
    nemotron_context_params params;

    nemotron_model model;
    nemotron_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;

    nemotron_predictor_weights pred_w;
    nemotron_joint_weights joint_w;

    int n_threads = 4;

    // Streaming state
    int att_context_preset = 0;
    int prompt_id = 0; // language prompt index (0 = en-US in NeMo's prompt_dictionary)

    // Decode controls
    float decode_temperature = 0.0f;
    uint64_t decode_seed = 0;
    int decode_beam_size = 1;

    // MAES controls
    bool decode_maes = false;
    int maes_num_steps = 2;
    float maes_gamma = 2.3f;
    int maes_beta = 2;

    // Prompt language map: lang_code -> prompt_id
    std::unordered_map<std::string, int> lang_to_prompt;

    // Per-layer streaming cache for cache-aware chunked encoding.
    // Populated by nemotron_run_encoder_chunked().
    struct layer_cache {
        // K/V cache: last L frames' K and V after projection.
        // Shape: (head_dim, n_heads, L) each, stored as flat float arrays.
        std::vector<float> k_cache, v_cache;
        int n_cached = 0; // frames currently in cache (0..L)

        // Conv state: last (K-1) frames before the depthwise conv.
        // Shape: (d_model, K-1) stored as flat float array.
        std::vector<float> conv_cache;
        int conv_cached = 0;
    };
    std::vector<layer_cache> enc_cache; // size = n_layers
};

// ===========================================================================
// Helpers
// ===========================================================================

static ggml_tensor* try_get(nemotron_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(nemotron_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "nemotron");
}

static std::vector<float> tensor_to_f32(ggml_tensor* t) {
    if (!t)
        return {};
    int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++)
            out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else if (ggml_is_quantized(t->type)) {
        size_t raw_sz = ggml_nbytes(t);
        std::vector<uint8_t> raw(raw_sz);
        ggml_backend_tensor_get(t, raw.data(), 0, raw_sz);
        const auto* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), out.data(), n);
        } else {
            fprintf(stderr, "nemotron: no dequant for type %d\n", t->type);
            out.assign(n, 0.0f);
        }
    } else {
        fprintf(stderr, "nemotron: unsupported tensor type %d\n", t->type);
        out.assign(n, 0.0f);
    }
    return out;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool nemotron_load_model(nemotron_model& model, nemotron_vocab& vocab,
                                std::unordered_map<std::string, int>& lang_to_prompt, const char* path,
                                ggml_backend_t backend) {
    // ---- pass 1: read hparams + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "nemotron.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "nemotron.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "nemotron.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "nemotron.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "nemotron.hop_length", hp.hop_length);
        hp.d_model = core_gguf::kv_u32(gctx, "nemotron.d_model", hp.d_model);
        hp.n_layers = core_gguf::kv_u32(gctx, "nemotron.n_layers", hp.n_layers);
        hp.n_heads = core_gguf::kv_u32(gctx, "nemotron.n_heads", hp.n_heads);
        hp.head_dim = core_gguf::kv_u32(gctx, "nemotron.head_dim", hp.head_dim);
        hp.ff_dim = core_gguf::kv_u32(gctx, "nemotron.ff_dim", hp.ff_dim);
        hp.subsampling_factor = core_gguf::kv_u32(gctx, "nemotron.subsampling_factor", hp.subsampling_factor);
        hp.subsampling_channels = core_gguf::kv_u32(gctx, "nemotron.subsampling_channels", hp.subsampling_channels);
        hp.conv_kernel = core_gguf::kv_u32(gctx, "nemotron.conv_kernel", hp.conv_kernel);
        hp.xscaling = core_gguf::kv_bool(gctx, "nemotron.xscaling", hp.xscaling);
        hp.causal_downsampling = core_gguf::kv_bool(gctx, "nemotron.causal_downsampling", hp.causal_downsampling);
        hp.pred_hidden = core_gguf::kv_u32(gctx, "nemotron.pred_hidden", hp.pred_hidden);
        hp.pred_layers = core_gguf::kv_u32(gctx, "nemotron.pred_layers", hp.pred_layers);
        hp.joint_hidden = core_gguf::kv_u32(gctx, "nemotron.joint_hidden", hp.joint_hidden);
        hp.vocab_size = core_gguf::kv_u32(gctx, "nemotron.vocab_size", hp.vocab_size);
        hp.blank_id = core_gguf::kv_u32(gctx, "nemotron.blank_id", hp.blank_id);
        hp.frame_dur_cs = core_gguf::kv_u32(gctx, "nemotron.frame_dur_cs", hp.frame_dur_cs);
        hp.num_prompts = core_gguf::kv_u32(gctx, "nemotron.num_prompts", hp.num_prompts);
        hp.prompt_kernel_in = core_gguf::kv_u32(gctx, "nemotron.prompt_kernel_in", hp.prompt_kernel_in);
        hp.prompt_kernel_mid = core_gguf::kv_u32(gctx, "nemotron.prompt_kernel_mid", hp.prompt_kernel_mid);
        hp.n_att_context_presets = core_gguf::kv_u32(gctx, "nemotron.n_att_context_presets", hp.n_att_context_presets);

        // Vocab
        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        // Note: att_context_left/right arrays are stored as GGUF int32 arrays.
        // Reading them requires the GGUFv3 array API which core_gguf doesn't
        // expose directly for int arrays. Use defaults for now; the runtime
        // picks the preset index and the C++ code handles the mapping.

        // Populate language → prompt_id mapping from NeMo's prompt_dictionary.
        // This is NOT alphabetical — it's the order from the model_config.yaml
        // training config. The prompt_kernel one-hot encoding must match this
        // exact mapping or the encoder output will be conditioned on the wrong
        // language (#81).
        {
            // clang-format off
            const struct { const char* code; int id; } prompts[] = {
                {"en-US", 0}, {"en-GB", 1}, {"es-ES", 2}, {"es-US", 3},
                {"zh-CN", 4}, {"zh-TW", 5}, {"hi-IN", 6}, {"ar-AR", 7},
                {"fr-FR", 8}, {"de-DE", 9}, {"ja-JP", 10}, {"ru-RU", 11},
                {"pt-BR", 12}, {"pt-PT", 13}, {"ko-KR", 14}, {"it-IT", 15},
                {"nl-NL", 16}, {"pl-PL", 17}, {"tr-TR", 18}, {"uk-UA", 19},
                {"ro-RO", 20}, {"el-GR", 21}, {"cs-CZ", 22}, {"hu-HU", 23},
                {"sv-SE", 24}, {"da-DK", 25}, {"fi-FI", 26}, {"nb-NO", 27},
                {"sk-SK", 28}, {"hr-HR", 29}, {"bg-BG", 30}, {"lt-LT", 31},
                {"th-TH", 32}, {"vi-VN", 33}, {"et-EE", 60}, {"lv-LV", 61},
                {"sl-SI", 62}, {"he-IL", 64}, {"fr-CA", 100}, {"nn-NO", 104},
            };
            // clang-format on
            for (const auto& p : prompts) {
                lang_to_prompt[p.code] = p.id;
                std::string lo = p.code;
                for (auto& c : lo)
                    c = (char)std::tolower((unsigned char)c);
                lang_to_prompt[lo] = p.id;
            }
            // Short ISO 639-1 aliases → first matching locale
            lang_to_prompt["en"] = 0;   // en-US
            lang_to_prompt["es"] = 3;   // es-US
            lang_to_prompt["zh"] = 4;   // zh-CN
            lang_to_prompt["hi"] = 6;   // hi-IN
            lang_to_prompt["ar"] = 7;   // ar-AR
            lang_to_prompt["fr"] = 8;   // fr-FR
            lang_to_prompt["de"] = 9;   // de-DE
            lang_to_prompt["ja"] = 10;  // ja-JP
            lang_to_prompt["ru"] = 11;  // ru-RU
            lang_to_prompt["pt"] = 13;  // pt-PT
            lang_to_prompt["ko"] = 14;  // ko-KR
            lang_to_prompt["it"] = 15;  // it-IT
            lang_to_prompt["nl"] = 16;  // nl-NL
            lang_to_prompt["pl"] = 17;  // pl-PL
            lang_to_prompt["tr"] = 18;  // tr-TR
            lang_to_prompt["uk"] = 19;  // uk-UA
            lang_to_prompt["ro"] = 20;  // ro-RO
            lang_to_prompt["el"] = 21;  // el-GR
            lang_to_prompt["cs"] = 22;  // cs-CZ
            lang_to_prompt["hu"] = 23;  // hu-HU
            lang_to_prompt["sv"] = 24;  // sv-SE
            lang_to_prompt["da"] = 25;  // da-DK
            lang_to_prompt["fi"] = 26;  // fi-FI
            lang_to_prompt["nb"] = 27;  // nb-NO
            lang_to_prompt["sk"] = 28;  // sk-SK
            lang_to_prompt["hr"] = 29;  // hr-HR
            lang_to_prompt["bg"] = 30;  // bg-BG
            lang_to_prompt["lt"] = 31;  // lt-LT
            lang_to_prompt["th"] = 32;  // th-TH
            lang_to_prompt["vi"] = 33;  // vi-VN
            lang_to_prompt["et"] = 60;  // et-EE
            lang_to_prompt["lv"] = 61;  // lv-LV
            lang_to_prompt["sl"] = 62;  // sl-SI
            lang_to_prompt["he"] = 64;  // he-IL
            lang_to_prompt["nn"] = 104; // nn-NO
        }

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: load tensor data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "nemotron", wl)) {
        return false;
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // ---- bind named tensors ----

    // Mel preprocessor
    model.mel_fb = try_get(model, "preprocessor.fb");
    model.mel_window = try_get(model, "preprocessor.window");

    // Pre-encode
    model.pre_encode.conv0_w = require(model, "encoder.pre.conv.0.weight");
    model.pre_encode.conv0_b = require(model, "encoder.pre.conv.0.bias");
    model.pre_encode.conv2_w = require(model, "encoder.pre.conv.2.weight");
    model.pre_encode.conv2_b = require(model, "encoder.pre.conv.2.bias");
    model.pre_encode.conv3_w = require(model, "encoder.pre.conv.3.weight");
    model.pre_encode.conv3_b = require(model, "encoder.pre.conv.3.bias");
    model.pre_encode.conv5_w = require(model, "encoder.pre.conv.5.weight");
    model.pre_encode.conv5_b = require(model, "encoder.pre.conv.5.bias");
    model.pre_encode.conv6_w = require(model, "encoder.pre.conv.6.weight");
    model.pre_encode.conv6_b = require(model, "encoder.pre.conv.6.bias");
    model.pre_encode.out_w = require(model, "encoder.pre.out.weight");
    model.pre_encode.out_b = require(model, "encoder.pre.out.bias");

    // Encoder layers
    model.enc.resize(model.hparams.n_layers);
    for (uint32_t i = 0; i < model.hparams.n_layers; i++) {
        char buf[128];
        auto& e = model.enc[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return require(model, buf);
        };
        auto try_ = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return try_get(model, buf);
        };

        e.norm_ff1_w = get("norm_ff1.weight");
        e.norm_ff1_b = get("norm_ff1.bias");
        e.ff1_l1_w = get("ff1.linear1.weight");
        e.ff1_l1_b = try_("ff1.linear1.bias");
        e.ff1_l2_w = get("ff1.linear2.weight");
        e.ff1_l2_b = try_("ff1.linear2.bias");

        e.norm_attn_w = get("norm_attn.weight");
        e.norm_attn_b = get("norm_attn.bias");
        e.attn_q_w = get("attn.q.weight");
        e.attn_q_b = try_("attn.q.bias");
        e.attn_k_w = get("attn.k.weight");
        e.attn_k_b = try_("attn.k.bias");
        e.attn_v_w = get("attn.v.weight");
        e.attn_v_b = try_("attn.v.bias");
        e.attn_out_w = get("attn.out.weight");
        e.attn_out_b = try_("attn.out.bias");
        e.attn_pos_w = get("attn.pos.weight");
        e.pos_bias_u = get("attn.pos_bias_u");
        e.pos_bias_v = get("attn.pos_bias_v");

        e.norm_conv_w = get("norm_conv.weight");
        e.norm_conv_b = get("norm_conv.bias");
        e.conv_pw1_w = get("conv.pw1.weight");
        e.conv_pw1_b = try_("conv.pw1.bias");
        e.conv_dw_w = get("conv.dw.weight");
        // LayerNorm in conv module
        e.conv_ln_w = get("conv.ln.weight");
        e.conv_ln_b = get("conv.ln.bias");
        // Synthesize a zero dw.bias since the shared build_block expects one
        e.conv_dw_b = try_("conv.dw.bias");
        e.conv_pw2_w = get("conv.pw2.weight");
        e.conv_pw2_b = try_("conv.pw2.bias");

        e.norm_ff2_w = get("norm_ff2.weight");
        e.norm_ff2_b = get("norm_ff2.bias");
        e.ff2_l1_w = get("ff2.linear1.weight");
        e.ff2_l1_b = try_("ff2.linear1.bias");
        e.ff2_l2_w = get("ff2.linear2.weight");
        e.ff2_l2_b = try_("ff2.linear2.bias");

        e.norm_out_w = get("norm_out.weight");
        e.norm_out_b = get("norm_out.bias");
    }

    // Prompt kernel
    model.prompt_kernel.l0_w = try_get(model, "prompt_kernel.0.weight");
    model.prompt_kernel.l0_b = try_get(model, "prompt_kernel.0.bias");
    model.prompt_kernel.l2_w = try_get(model, "prompt_kernel.2.weight");
    model.prompt_kernel.l2_b = try_get(model, "prompt_kernel.2.bias");

    // Predictor
    auto& p = model.predictor;
    p.embed_w = require(model, "decoder.embed.weight");
    p.lstm0_w_ih = require(model, "decoder.lstm.0.w_ih");
    p.lstm0_w_hh = require(model, "decoder.lstm.0.w_hh");
    p.lstm0_b_ih = require(model, "decoder.lstm.0.b_ih");
    p.lstm0_b_hh = require(model, "decoder.lstm.0.b_hh");
    const bool has_lstm1 = model.hparams.pred_layers >= 2;
    p.lstm1_w_ih = has_lstm1 ? require(model, "decoder.lstm.1.w_ih") : nullptr;
    p.lstm1_w_hh = has_lstm1 ? require(model, "decoder.lstm.1.w_hh") : nullptr;
    p.lstm1_b_ih = has_lstm1 ? require(model, "decoder.lstm.1.b_ih") : nullptr;
    p.lstm1_b_hh = has_lstm1 ? require(model, "decoder.lstm.1.b_hh") : nullptr;

    // Joint
    auto& j = model.joint;
    j.enc_w = require(model, "joint.enc.weight");
    j.enc_b = require(model, "joint.enc.bias");
    j.pred_w = require(model, "joint.pred.weight");
    j.pred_b = require(model, "joint.pred.bias");
    j.out_w = require(model, "joint.out.weight");
    j.out_b = require(model, "joint.out.bias");

    fprintf(stderr, "nemotron: vocab=%u  d_model=%u  n_layers=%u  n_heads=%u  ff=%u  pred=%u  joint=%u\n",
            model.hparams.vocab_size, model.hparams.d_model, model.hparams.n_layers, model.hparams.n_heads,
            model.hparams.ff_dim, model.hparams.pred_hidden, model.hparams.joint_hidden);
    fprintf(stderr,
            "nemotron: streaming: %u presets, causal_ds=%d, conv_norm=layer_norm, conv_ln=%s, prompt_kernel=%s\n",
            model.hparams.n_att_context_presets, model.hparams.causal_downsampling ? 1 : 0,
            model.enc[0].conv_ln_w ? "loaded" : "NULL", model.prompt_kernel.l0_w ? "loaded" : "NULL");
    return true;
}

// ===========================================================================
// FFT
// ===========================================================================

static void nemotron_fft_r2c(const float* in, int N, float* out) {
    int bits = 0;
    for (int n = N; n > 1; n >>= 1)
        bits++;
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int b = 0; b < bits; b++)
            rev = (rev << 1) | ((i >> b) & 1);
        out[2 * rev] = in[i];
        out[2 * rev + 1] = 0.0f;
    }
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, b = i + j + len / 2;
                float are = out[2 * a], aim = out[2 * a + 1];
                float bre = out[2 * b], bim = out[2 * b + 1];
                float tre = ure * bre - uim * bim, tim = ure * bim + uim * bre;
                out[2 * a] = are + tre;
                out[2 * a + 1] = aim + tim;
                out[2 * b] = are - tre;
                out[2 * b + 1] = aim - tim;
                float new_ure = ure * wre - uim * wim;
                uim = ure * wim + uim * wre;
                ure = new_ure;
            }
        }
    }
}

// ===========================================================================
// Mel spectrogram — NeMo-style, NO per-feature z-norm (normalize="NA")
// ===========================================================================

static std::vector<float> nemotron_compute_mel_impl(nemotron_context* ctx, const float* samples, int n_samples,
                                                    int& T_out) {
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int win = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels = (int)hp.n_mels;

    if (!ctx->model.mel_fb || !ctx->model.mel_window) {
        fprintf(stderr, "nemotron: missing preprocessor.fb or preprocessor.window in GGUF\n");
        return {};
    }

    std::vector<float> window_raw((size_t)win);
    ggml_backend_tensor_get(ctx->model.mel_window, window_raw.data(), 0, win * sizeof(float));

    std::vector<float> mel_fb((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(ctx->model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    // NeMo AudioToMelSpectrogramPreprocessor with normalize="NA":
    // No per-feature normalization. Log-mel only.
    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = win;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.norm = core_mel::Normalization::None; // normalize="NA"
    p.layout = core_mel::Layout::TimeMels;
    p.log_eps = (float)(1.0 / (1 << 24));
    p.center_pad = true;
    p.drop_last_frame = false; // NeMo default: keep all frames (1101 not 1100)
    p.preemph = 0.97f;

    auto mel = core_mel::compute(samples, n_samples, window_raw.data(), win, mel_fb.data(), n_freqs, nemotron_fft_r2c,
                                 p, T_out);
    return mel;
}

// ===========================================================================
// Causal pre-encode builder
//
// Nemotron uses CausalConv2D for subsampling with asymmetric padding:
//   left_pad = kernel_size - 1 = 2
//   right_pad = stride - 1     = 1
// This differs from the shared build_pre_encode which uses symmetric (1,1).
// The asymmetric padding produces 17 freq bins (not 16) after 3-stage 8x
// temporal downsampling from 128 mel bins:
//   128 -> pad(2,1)=131 -> conv k=3 s=2 -> 65
//    65 -> pad(2,1)=68  -> dw  k=3 s=2 -> 33
//    33 -> pw -> 33
//    33 -> pad(2,1)=36  -> dw  k=3 s=2 -> 17
//    17 -> pw -> 17
// Flatten: 17 * 256 = 4352 input to the final linear.
// ===========================================================================

static ggml_tensor* nemotron_build_pre_encode(ggml_context* ctx0, ggml_tensor* mel,
                                              const core_conformer::PreEncodeWeights& w, int subsampling_channels,
                                              int* out_T_enc) {
    auto bias_4d = [&](ggml_tensor* b) {
        return ggml_cast(ctx0, ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    // CausalConv2D: pad left=k-1=2, right=s-1=1 on BOTH freq (ne[0]) and time (ne[1]).
    // Use ggml_pad_ext for true asymmetric padding, then conv with p=0.
    // This produces correct per-element values (symmetric pad=(2,2) changes values
    // because the extra right-side zeros participate in the convolution).
    auto causal_pad = [&](ggml_tensor* x) -> ggml_tensor* {
        return ggml_pad_ext(ctx0, x, /*lp0*/ 2, /*rp0*/ 1, /*lp1*/ 2, /*rp1*/ 1, 0, 0, 0, 0);
    };

    // Stage 0: Conv2d(1, C, k=3, s=2) with causal padding
    ggml_tensor* cur = ggml_conv_2d(ctx0, w.conv0_w, causal_pad(mel), 2, 2, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv0_b));
    cur = ggml_relu(ctx0, cur);

    // Stage 2: DW Conv2d(C, k=3, s=2) with causal padding
    cur = ggml_conv_2d_dw(ctx0, w.conv2_w, causal_pad(cur), 2, 2, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv2_b));

    // Stage 3: PW Conv2d(C, C, k=1, s=1) — no padding
    cur = ggml_conv_2d(ctx0, w.conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv3_b));
    cur = ggml_relu(ctx0, cur);

    // Stage 5: DW Conv2d(C, k=3, s=2) with causal padding
    cur = ggml_conv_2d_dw(ctx0, w.conv5_w, causal_pad(cur), 2, 2, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv5_b));

    // Stage 6: PW Conv2d(C, C, k=1, s=1)
    cur = ggml_conv_2d(ctx0, w.conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv6_b));
    cur = ggml_relu(ctx0, cur);

    // Flatten and linear: (OW, OH, C, 1) -> permute to (OH, C, OW, 1) -> reshape to (C*OW, OH)
    const int H3 = (int)cur->ne[1]; // time (T_enc)
    const int W3 = (int)cur->ne[0]; // freq (should be 17)
    const int C = subsampling_channels;
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, W3 * C, H3);

    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, w.out_w, cur), w.out_b);

    if (out_T_enc)
        *out_T_enc = H3;
    return cur;
}

// ===========================================================================
// Nemotron Conformer block builder
//
// Almost identical to core_conformer::build_block but replaces the BN-folded
// conv_dw_b add with a proper LayerNorm using the conv_ln_w/conv_ln_b tensors.
// ===========================================================================

// Build a chunked_limited attention mask (F16) matching NeMo's att_context_style.
// Each frame belongs to a chunk of size (right+1). A frame can attend to frames
// in its own chunk and up to left_chunks_num previous chunks.
// mask[q, k] = 0 if visible, -inf if masked.
// Shape: (T, T) broadcast across heads.
static std::vector<ggml_fp16_t> build_window_mask(int T, int left, int right) {
    const int chunk_size = right + 1;
    const int left_chunks_num = left >= 0 ? left / chunk_size : 10000;
    std::vector<ggml_fp16_t> mask((size_t)T * T);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-1e9f);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    for (int q = 0; q < T; q++) {
        int q_chunk = q / chunk_size;
        for (int k = 0; k < T; k++) {
            int k_chunk = k / chunk_size;
            int diff = q_chunk - k_chunk;
            bool visible = (diff >= 0) && (diff <= left_chunks_num);
            mask[(size_t)q * T + k] = visible ? zero : neg_inf;
        }
    }
    return mask;
}

// ---- Streaming block: split into stages with separate cache inputs ----
// new_in:    (d, T_new) — new frames for this chunk
// cache_ch:  (d, T_cache) — cached post-FFN1 frames from previous chunks (or nullptr)
// pos_enc:   (d, 2*(T_cache+T_new)-1) — rel-pos for the full window
// Output:    (d, T_new) — block output for new frames only
// Also tags "cache_ch_out" = post-FFN1 new frames for caching.
// conv_cache_in: (d, K-1) tensor of pre-DW-conv signal from previous chunk (or nullptr for first chunk).
// NeMo's CausalConv1D.update_cache prepends this instead of zero-padding.
static ggml_tensor* nemotron_build_block_streaming(ggml_context* ctx0, ggml_tensor* new_in, ggml_tensor* cache_ch,
                                                   ggml_tensor* conv_cache_in, ggml_tensor* pos_enc, int T_new,
                                                   int T_cache, const nemotron_enc_layer& e,
                                                   const core_conformer::BlockParams& p) {
    const int d = p.d;
    const int n_heads = p.n_heads;
    const int head_dim = p.head_dim;
    const int K = p.K;
    const float eps = p.ln_eps;
    const int T_full = T_cache + T_new; // full attention window

    auto mm_bias = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
        ggml_tensor* y = ggml_mul_mat(ctx0, w, x);
        return b ? ggml_add(ctx0, y, b) : y;
    };

    // ---- FFN1 on new frames only ----
    ggml_tensor* cur = new_in;
    ggml_tensor* inpL = cur;
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, e.norm_ff1_w, e.norm_ff1_b, eps);
    x = mm_bias(e.ff1_l1_w, x, e.ff1_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff1_l2_w, x, e.ff1_l2_b);
    cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, x, 0.5f));
    // cur = post-FFN1 for new frames (d, T_new)

    // Tag post-FFN1 for caching
    ggml_tensor* post_ffn1_new = ggml_dup(ctx0, cur);
    ggml_set_name(post_ffn1_new, "cache_ch_out");
    ggml_set_output(post_ffn1_new);

    // ---- Self-attention: Q from new, K/V from [cache, new] ----
    ggml_tensor* inpAttn = cur;

    // Build full input for K/V: concat(cache_ch, cur) along time axis (dim 1)
    ggml_tensor* full_kv_in;
    if (cache_ch) {
        full_kv_in = ggml_concat(ctx0, cache_ch, cur, 1); // (d, T_cache + T_new)
    } else {
        full_kv_in = cur; // no cache yet
    }

    // Norm for attention — apply to both cached and new
    ggml_tensor* norm_new = ggml_norm_affine(ctx0, cur, e.norm_attn_w, e.norm_attn_b, eps);
    ggml_tensor* norm_full;
    if (cache_ch) {
        norm_full = ggml_norm_affine(ctx0, full_kv_in, e.norm_attn_w, e.norm_attn_b, eps);
    } else {
        norm_full = norm_new;
    }

    // Q from new frames only, K/V from full window
    ggml_tensor* Q = mm_bias(e.attn_q_w, norm_new, e.attn_q_b);   // (d, T_new)
    ggml_tensor* K_ = mm_bias(e.attn_k_w, norm_full, e.attn_k_b); // (d, T_full)
    ggml_tensor* V = mm_bias(e.attn_v_w, norm_full, e.attn_v_b);  // (d, T_full)

    // Rel-pos: pos_enc covers the full window
    ggml_tensor* R = ggml_mul_mat(ctx0, e.attn_pos_w, pos_enc); // (d, 2*T_full-1)

    ggml_tensor* Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_u, d));
    ggml_tensor* Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_v, d));

    // Reshape for multi-head: Q is (head_dim, T_new, n_heads), K/V is (head_dim, T_full, n_heads)
    Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T_new), 0, 2, 1, 3);
    Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T_new), 0, 2, 1, 3);
    K_ = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K_, head_dim, n_heads, T_full), 0, 2, 1, 3);
    R = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R, head_dim, n_heads, 2 * T_full - 1), 0, 2, 1, 3);
    ggml_tensor* V_ =
        ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, head_dim, n_heads, T_full), 0, 2, 1, 3));

    // BD = rel_shift(Q_v @ R^T): asymmetric version for Q(T_new) × K(T_full).
    // BD_raw shape: (2*T_full-1, T_new, n_heads).
    // Asymmetric rel_shift: BD[k, q, h] = BD_raw[(T_new-1)+k-q, q, h]
    //   = data[(T_new-1)*s0 + k*s0 + q*(s1-s0) + h*s2]
    // This is view_3d(BD_raw, T_full, T_new, H, s1-s0, s2, (T_new-1)*s0).
    ggml_tensor* BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v); // (2*T_full-1, T_new, n_heads)
    ggml_tensor* BD = ggml_view_3d(ctx0, BD_raw, T_full, T_new, n_heads, BD_raw->nb[1] - BD_raw->nb[0], BD_raw->nb[2],
                                   (T_new - 1) * BD_raw->nb[0]);

    const float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor* BD_c = ggml_cont(ctx0, BD);
    ggml_tensor* BD_scaled = ggml_scale(ctx0, BD_c, scale);
    ggml_tensor* BD_mask = ggml_cast(ctx0, BD_scaled, GGML_TYPE_F16);

    ggml_tensor* attn_out =
        ggml_flash_attn_ext(ctx0, ggml_cont(ctx0, Q_u), ggml_cont(ctx0, K_), V_, BD_mask, scale, 0.0f, 0.0f);
    attn_out = ggml_reshape_2d(ctx0, attn_out, d, T_new);

    attn_out = mm_bias(e.attn_out_w, attn_out, e.attn_out_b);
    cur = ggml_add(ctx0, inpAttn, attn_out);

    // ---- Conv module (new frames only, with conv state cache) ----
    ggml_tensor* inpConv = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_conv_w, e.norm_conv_b, eps);
    ggml_tensor* pw1_w = ggml_reshape_2d(ctx0, e.conv_pw1_w, d, 2 * d);
    ggml_tensor* cnv = mm_bias(pw1_w, x, e.conv_pw1_b);
    cnv = ggml_siglu_swapped(ctx0, cnv);

    // Save post-GLU signal for conv cache output (last K-1 frames)
    // NeMo's CausalConv1D.update_cache stores this for the next chunk.
    {
        // cnv is (d, T_new). We need the last (K-1) frames.
        int cache_frames = K - 1;
        ggml_tensor* conv_cache_new;
        if (T_new >= cache_frames) {
            int offset_frames = T_new - cache_frames;
            conv_cache_new = ggml_view_2d(ctx0, cnv, d, cache_frames, cnv->nb[1], (size_t)offset_frames * cnv->nb[1]);
        } else {
            // T_new < K-1: prepend part of old cache + all new frames
            // This case handles very small chunks; for simplicity, just save what we have
            conv_cache_new = cnv;
        }
        conv_cache_new = ggml_cont(ctx0, conv_cache_new);
        ggml_set_name(conv_cache_new, "conv_cache_out");
        ggml_set_output(conv_cache_new);
    }

    // DW conv: prepend conv cache (or zero-pad) for left context
    ggml_tensor* dw_w_f32 = ggml_cast(ctx0, e.conv_dw_w, GGML_TYPE_F32);
    ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, K, 1, 1, d);

    if (conv_cache_in) {
        // Prepend cached frames: [conv_cache_in (d,K-1), cnv (d,T_new)] → (d, K-1+T_new)
        ggml_tensor* cnv_padded = ggml_concat(ctx0, conv_cache_in, cnv, 1);
        cnv_padded = ggml_cont(ctx0, ggml_transpose(ctx0, cnv_padded));
        cnv_padded = ggml_reshape_4d(ctx0, cnv_padded, K - 1 + T_new, 1, d, 1);
        // Conv with no padding — output is (T_new, 1, d, 1)
        cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv_padded, 1, 1, 0, 0, 1, 1);
    } else {
        // First chunk: zero-pad left by K-1 (same as NeMo's first-call behavior)
        cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv));
        cnv = ggml_reshape_4d(ctx0, cnv, T_new, 1, d, 1);
        cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, K - 1, 0, 1, 1);
    }
    {
        cnv = ggml_view_4d(ctx0, cnv, T_new, cnv->ne[1], cnv->ne[2], cnv->ne[3], cnv->nb[1], cnv->nb[2], cnv->nb[3], 0);
        cnv = ggml_cont(ctx0, cnv);
    }
    cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
    cnv = ggml_reshape_2d(ctx0, cnv, d, T_new);

    if (e.conv_ln_w && e.conv_ln_b) {
        cnv = ggml_norm_affine(ctx0, cnv, e.conv_ln_w, e.conv_ln_b, eps);
    }
    cnv = ggml_silu(ctx0, cnv);

    ggml_tensor* pw2_w = ggml_reshape_2d(ctx0, e.conv_pw2_w, d, d);
    cnv = mm_bias(pw2_w, cnv, e.conv_pw2_b);
    cur = ggml_add(ctx0, inpConv, cnv);

    // ---- FFN2 + LN (new frames only) ----
    ggml_tensor* inpFF2 = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_ff2_w, e.norm_ff2_b, eps);
    x = mm_bias(e.ff2_l1_w, x, e.ff2_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff2_l2_w, x, e.ff2_l2_b);
    cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, x, 0.5f));
    cur = ggml_norm_affine(ctx0, cur, e.norm_out_w, e.norm_out_b, eps);

    return cur;
}

// ---- Non-streaming block (full sequence, for batch/debug) ----
// window_mask: optional (T, T) F16 tensor with -inf outside the context window.
// When nullptr, attention is bidirectional (fallback for debugging).
static ggml_tensor* nemotron_build_block(ggml_context* ctx0, ggml_tensor* cur, ggml_tensor* pos_enc, int T,
                                         const nemotron_enc_layer& e, const core_conformer::BlockParams& p,
                                         ggml_tensor* window_mask = nullptr) {
    const int d = p.d;
    const int n_heads = p.n_heads;
    const int head_dim = p.head_dim;
    const int K = p.K;
    const float eps = p.ln_eps;

    auto mm_bias = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
        ggml_tensor* y = ggml_mul_mat(ctx0, w, x);
        return b ? ggml_add(ctx0, y, b) : y;
    };

    ggml_tensor* inpL = cur;

    // ---- FFN1 (macaron half) ----
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, e.norm_ff1_w, e.norm_ff1_b, eps);
    x = mm_bias(e.ff1_l1_w, x, e.ff1_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff1_l2_w, x, e.ff1_l2_b);
    cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, x, 0.5f));

    ggml_tensor* inpAttn = cur;

    // ---- Self-Attention (rel_pos with untied biases) ----
    x = ggml_norm_affine(ctx0, cur, e.norm_attn_w, e.norm_attn_b, eps);

    ggml_tensor* Q = mm_bias(e.attn_q_w, x, e.attn_q_b);
    ggml_tensor* K_ = mm_bias(e.attn_k_w, x, e.attn_k_b);
    ggml_tensor* V = mm_bias(e.attn_v_w, x, e.attn_v_b);
    ggml_tensor* R = ggml_mul_mat(ctx0, e.attn_pos_w, pos_enc);

    ggml_tensor* Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_u, d));
    ggml_tensor* Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_v, d));

    Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T), 0, 2, 1, 3);
    Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T), 0, 2, 1, 3);
    K_ = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K_, head_dim, n_heads, T), 0, 2, 1, 3);
    R = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R, head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3);

    ggml_tensor* BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v);
    ggml_tensor* BD = core_conformer::rel_shift(ctx0, BD_raw);

    const float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor* BD_c = ggml_cont(ctx0, BD);
    ggml_tensor* BD_scaled = ggml_scale(ctx0, BD_c, scale);

    // Combine rel-pos bias with the streaming window mask
    ggml_tensor* attn_mask;
    if (window_mask) {
        // window_mask is (T, T) F16 with -inf outside the window, 0 inside.
        // BD_scaled is (T, T, n_heads) F32. Add window_mask (broadcast over heads)
        // then cast to F16 for flash_attn_ext.
        ggml_tensor* wm_f32 = ggml_cast(ctx0, window_mask, GGML_TYPE_F32);
        // Reshape window_mask to (T, T, 1) for broadcast
        ggml_tensor* wm_3d = ggml_reshape_3d(ctx0, wm_f32, T, T, 1);
        ggml_tensor* combined =
            ggml_add(ctx0, BD_scaled, ggml_repeat(ctx0, wm_3d, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, T, T, n_heads)));
        attn_mask = ggml_cast(ctx0, combined, GGML_TYPE_F16);
    } else {
        attn_mask = ggml_cast(ctx0, BD_scaled, GGML_TYPE_F16);
    }

    ggml_tensor* V_ = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, head_dim, n_heads, T), 0, 2, 1, 3));

    ggml_tensor* attn_out =
        ggml_flash_attn_ext(ctx0, ggml_cont(ctx0, Q_u), ggml_cont(ctx0, K_), V_, attn_mask, scale, 0.0f, 0.0f);
    attn_out = ggml_reshape_2d(ctx0, attn_out, d, T);

    attn_out = mm_bias(e.attn_out_w, attn_out, e.attn_out_b);
    cur = ggml_add(ctx0, inpAttn, attn_out);

    // ---- Conformer convolution module (with LayerNorm instead of BN) ----
    ggml_tensor* inpConv = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_conv_w, e.norm_conv_b, eps);

    ggml_tensor* pw1_w = ggml_reshape_2d(ctx0, e.conv_pw1_w, d, 2 * d);
    ggml_tensor* cnv = mm_bias(pw1_w, x, e.conv_pw1_b);
    // NeMo Conformer GLU: first_half * sigmoid(second_half) = swapped siglu
    // ggml_siglu does sigmoid(first) * second; ggml_siglu_swapped does first * sigmoid(second)
    cnv = ggml_siglu_swapped(ctx0, cnv);

    // DW conv (kernel K, CAUSAL padding: left=K-1, right=0)
    // ggml_conv_2d_dw_direct only supports symmetric padding, so we use
    // p0=K-1 (adds K-1 on both sides) then trim the last K-1 time frames.
    ggml_tensor* dw_w_f32 = ggml_cast(ctx0, e.conv_dw_w, GGML_TYPE_F32);
    ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, K, 1, 1, d);
    cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv));
    cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, 1);
    cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, K - 1, 0, 1, 1);
    // Output has T + (K-1) time frames due to excess right padding. Trim to T.
    // Conv output shape: (T+K-1, 1, d, 1). Take view of first T frames.
    {
        int64_t T_conv = cnv->ne[0]; // T + K - 1
        cnv = ggml_view_4d(ctx0, cnv, T, cnv->ne[1], cnv->ne[2], cnv->ne[3], cnv->nb[1], cnv->nb[2], cnv->nb[3], 0);
        cnv = ggml_cont(ctx0, cnv);
    }
    cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
    cnv = ggml_reshape_2d(ctx0, cnv, d, T);
    // LayerNorm (replaces BN-folded bias add in parakeet)
    if (e.conv_ln_w && e.conv_ln_b) {
        cnv = ggml_norm_affine(ctx0, cnv, e.conv_ln_w, e.conv_ln_b, eps);
    }
    cnv = ggml_silu(ctx0, cnv);

    ggml_tensor* pw2_w = ggml_reshape_2d(ctx0, e.conv_pw2_w, d, d);
    cnv = mm_bias(pw2_w, cnv, e.conv_pw2_b);
    cur = ggml_add(ctx0, inpConv, cnv);

    // ---- FFN2 (macaron half) ----
    ggml_tensor* inpFF2 = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_ff2_w, e.norm_ff2_b, eps);
    x = mm_bias(e.ff2_l1_w, x, e.ff2_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff2_l2_w, x, e.ff2_l2_b);
    cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, x, 0.5f));

    // ---- Block final LN ----
    cur = ggml_norm_affine(ctx0, cur, e.norm_out_w, e.norm_out_b, eps);

    return cur;
}

// ===========================================================================
// Encoder graph builder
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

static ggml_cgraph* nemotron_build_graph_encoder(nemotron_context* ctx, int T_mel) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // ----- Input -----
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // ----- Causal pre-encode (asymmetric padding, 17 freq bins) -----
    int T = 0;
    ggml_tensor* cur = nemotron_build_pre_encode(ctx0, mel, m.pre_encode, (int)hp.subsampling_channels, &T);

    // nemotron has xscaling=false, so no scaling step

    // ----- Sinusoidal rel-pos table [d, 2T-1] -----
    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    // ----- Window mask for cache-aware streaming attention -----
    // CRISPASR_NEMOTRON_NO_WINDOW_MASK=1 → bidirectional attention (for A/B testing).
    // Default: banded attention with att_context_left/right.
    ggml_tensor* window_mask_t = nullptr;
    const bool use_window_mask = !getenv("CRISPASR_CRISPASR_NEMOTRON_NO_WINDOW_MASK");
    if (use_window_mask && T > 0) {
        window_mask_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
        ggml_set_name(window_mask_t, "window_mask");
        ggml_set_input(window_mask_t);
    }

    // ----- Conformer layers -----
    core_conformer::BlockParams bp;
    bp.d = (int)hp.d_model;
    bp.n_heads = (int)hp.n_heads;
    bp.head_dim = (int)hp.head_dim;
    bp.K = (int)hp.conv_kernel;
    bp.ln_eps = kLayerNormEps;

    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = nemotron_build_block(ctx0, cur, pos_enc, T, m.enc[il], bp, window_mask_t);
    }

    ggml_set_name(cur, "enc_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ===========================================================================
// Lazy-init backend scheduler (replaces per-call gallocr)
// ===========================================================================

static bool nemotron_ensure_sched(nemotron_context* ctx) {
    if (ctx->sched)
        return true;
    ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
    int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
    return ctx->sched != nullptr;
}

// ===========================================================================
// Run encoder: mel → encoder output
// ===========================================================================

static bool nemotron_run_encoder(nemotron_context* ctx, const float* mel, int n_mels, int T_mel,
                                 std::vector<float>& enc_out, int& T_enc, int& d_model_out) {
    ggml_cgraph* gf = nemotron_build_graph_encoder(ctx, T_mel);

    if (!nemotron_ensure_sched(ctx))
        return false;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "nemotron: sched alloc encoder graph failed\n");
        return false;
    }

    // Set mel input
    ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_t, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    // Compute T_enc from the encoder graph output shape
    ggml_tensor* enc_out_t = ggml_graph_get_tensor(gf, "enc_out");
    T_enc = (int)enc_out_t->ne[1];
    d_model_out = (int)enc_out_t->ne[0];

    // Build and set pos_enc
    auto pe = core_conformer::make_pos_enc(d_model_out, T_enc);
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "pos_enc");
    ggml_backend_tensor_set(pos_t, pe.data(), 0, pe.size() * sizeof(float));

    // Set window mask for streaming attention
    ggml_tensor* wm_t = ggml_graph_get_tensor(gf, "window_mask");
    if (wm_t) {
        int preset = ctx->att_context_preset;
        int left = ctx->model.hparams.att_context_left[preset];
        int right = ctx->model.hparams.att_context_right[preset];
        auto wm = build_window_mask(T_enc, left, right);
        ggml_backend_tensor_set(wm_t, wm.data(), 0, wm.size() * sizeof(ggml_fp16_t));
        fprintf(stderr, "nemotron: streaming attention L=%d R=%d (CRISPASR_NEMOTRON_NO_WINDOW_MASK to disable)\n", left,
                right);
    }

    // Run
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "nemotron: encoder graph compute failed\n");
        return false;
    }

    // Read output: (d_model, T_enc) in column-major → row-major (T_enc, d_model)
    enc_out.resize((size_t)T_enc * d_model_out);
    ggml_backend_tensor_get(enc_out_t, enc_out.data(), 0, enc_out.size() * sizeof(float));

    if (getenv("CRISPASR_NEMOTRON_DEBUG")) {
        float emin = 1e30f, emax = -1e30f, esum = 0.0f;
        for (size_t i = 0; i < enc_out.size(); i++) {
            float v = enc_out[i];
            if (v < emin)
                emin = v;
            if (v > emax)
                emax = v;
            esum += v;
        }
        fprintf(stderr, "nemotron: default enc_out T=%d d=%d min=%.4f max=%.4f mean=%.6f\n", T_enc, d_model_out, emin,
                emax, esum / (float)enc_out.size());
    }

    return true;
}

// ===========================================================================
// Cache-aware chunked encoder
//
// Process pre-encoded frames in non-overlapping chunks of (R+1) frames.
// Each chunk sees up to L cached frames of left context per layer.
// Per-layer: build a ggml graph for the (cached+new) window, run one
// conformer block, extract the last (R+1) frames, update cache.
// ===========================================================================

static bool nemotron_run_encoder_chunked(nemotron_context* ctx, const float* pre_enc, int T_enc, int d_model,
                                         std::vector<float>& enc_out) {
    const auto& hp = ctx->model.hparams;
    const auto& m = ctx->model;
    const int n_layers = (int)hp.n_layers;
    const int preset = ctx->att_context_preset;
    const int L = hp.att_context_left[preset];  // left context frames (cache_last_channel)
    const int R = hp.att_context_right[preset]; // right context frames
    const int chunk_size = R + 1;               // new frames per chunk
    const int K = (int)hp.conv_kernel;
    const int d = d_model;

    fprintf(stderr, "nemotron: chunked encoder (streaming) L=%d R=%d chunk=%d T_enc=%d layers=%d\n", L, R, chunk_size,
            T_enc, n_layers);

    // Initialize per-layer caches.
    // k_cache stores post-FFN1 output (NeMo's cache_last_channel): the input
    // to the self-attention K/V projections.  Up to L frames per layer.
    ctx->enc_cache.resize(n_layers);
    for (int il = 0; il < n_layers; il++) {
        auto& c = ctx->enc_cache[il];
        c.k_cache.clear();
        c.v_cache.clear();
        c.n_cached = 0;
        c.conv_cache.assign((size_t)(K - 1) * d, 0.0f);
        c.conv_cached = 0;
    }

    int n_chunks = (T_enc + chunk_size - 1) / chunk_size;
    enc_out.resize((size_t)T_enc * d);

    core_conformer::BlockParams bp;
    bp.d = d;
    bp.n_heads = (int)hp.n_heads;
    bp.head_dim = (int)hp.head_dim;
    bp.K = K;
    bp.ln_eps = kLayerNormEps;

    // Streaming graph cache keyed by (layer, T_new, T_cache).
    // Uses nemotron_build_block_streaming: FFN1 on new only, Q from new,
    // K/V from [cache_ch + new_post_ffn1], conv with causal pad, FFN2+LN on new.
    struct layer_graph {
        ggml_context* ctx0 = nullptr;
        ggml_cgraph* gf = nullptr;
    };
    std::map<std::tuple<int, int, int>, layer_graph> graph_cache;

    auto get_or_build = [&](int il, int T_new, int T_cache, bool has_conv_cache) -> layer_graph& {
        // Key includes has_conv_cache (bit-packed into T_cache sign is awkward; use a 4th field)
        auto key = std::make_tuple(il, T_new, T_cache + (has_conv_cache ? 10000 : 0));
        auto it = graph_cache.find(key);
        if (it != graph_cache.end())
            return it->second;
        size_t msz = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(2048, false);
        auto* meta = new std::vector<uint8_t>(msz);
        ggml_init_params ip2 = {msz, meta->data(), true};
        layer_graph lg;
        lg.ctx0 = ggml_init(ip2);
        lg.gf = ggml_new_graph_custom(lg.ctx0, 2048, false);

        // Input: new frames (pre-FFN1 input for this layer)
        ggml_tensor* inp_new = ggml_new_tensor_2d(lg.ctx0, GGML_TYPE_F32, d, T_new);
        ggml_set_name(inp_new, "block_in");
        ggml_set_input(inp_new);

        // Cache: post-FFN1 output from previous chunks (nullptr for first chunk)
        ggml_tensor* cache_ch = nullptr;
        if (T_cache > 0) {
            cache_ch = ggml_new_tensor_2d(lg.ctx0, GGML_TYPE_F32, d, T_cache);
            ggml_set_name(cache_ch, "cache_ch");
            ggml_set_input(cache_ch);
        }

        // Conv cache: last K-1 frames of pre-DW-conv signal from previous chunk
        ggml_tensor* conv_cache = nullptr;
        if (has_conv_cache) {
            conv_cache = ggml_new_tensor_2d(lg.ctx0, GGML_TYPE_F32, d, K - 1);
            ggml_set_name(conv_cache, "conv_cache_in");
            ggml_set_input(conv_cache);
        }

        // Pos enc covers the full attention window
        int T_full = T_cache + T_new;
        ggml_tensor* pos2 = ggml_new_tensor_2d(lg.ctx0, GGML_TYPE_F32, d, 2 * T_full - 1);
        ggml_set_name(pos2, "pos_enc");
        ggml_set_input(pos2);

        ggml_tensor* out2 =
            nemotron_build_block_streaming(lg.ctx0, inp_new, cache_ch, conv_cache, pos2, T_new, T_cache, m.enc[il], bp);
        ggml_set_name(out2, "block_out");
        ggml_set_output(out2);
        ggml_build_forward_expand(lg.gf, out2);

        // Also expand cache outputs — they're not reachable from block_out's
        // dependency tree. Find them by scanning the ggml context's tensor list.
        for (ggml_tensor* t = ggml_get_first_tensor(lg.ctx0); t; t = ggml_get_next_tensor(lg.ctx0, t)) {
            if (t->name && (std::string(t->name) == "cache_ch_out" || std::string(t->name) == "conv_cache_out")) {
                ggml_build_forward_expand(lg.gf, t);
            }
        }

        graph_cache[key] = lg;
        return graph_cache[key];
    };

    if (!nemotron_ensure_sched(ctx))
        return false;

    // Process each chunk
    for (int ci = 0; ci < n_chunks; ci++) {
        int t_start = ci * chunk_size;
        int t_end = std::min(t_start + chunk_size, T_enc);
        int n_new = t_end - t_start;

        // chunk_in: this chunk's input to each layer (starts as pre_enc slice,
        // then gets replaced with each layer's output)
        std::vector<float> chunk_in(pre_enc + (size_t)t_start * d, pre_enc + (size_t)t_end * d);

        for (int il = 0; il < n_layers; il++) {
            auto& cache = ctx->enc_cache[il];
            int n_ctx = std::min(cache.n_cached, L);

            bool has_conv = (cache.conv_cached > 0);
            auto& lg = get_or_build(il, n_new, n_ctx, has_conv);

            ggml_backend_sched_reset(ctx->sched);
            if (!ggml_backend_sched_alloc_graph(ctx->sched, lg.gf)) {
                fprintf(stderr, "nemotron: sched alloc streaming layer %d failed\n", il);
                return false;
            }

            // Set new frames input
            ggml_tensor* inp_t = ggml_graph_get_tensor(lg.gf, "block_in");
            if (!inp_t) {
                fprintf(stderr, "nemotron: BUG block_in nil ci=%d il=%d n_new=%d n_ctx=%d\n", ci, il, n_new, n_ctx);
                return false;
            }
            ggml_backend_tensor_set(inp_t, chunk_in.data(), 0, (size_t)n_new * d * sizeof(float));

            // Set cache_last_channel (post-FFN1 from previous chunks)
            if (n_ctx > 0) {
                ggml_tensor* cache_t = ggml_graph_get_tensor(lg.gf, "cache_ch");
                if (!cache_t) {
                    fprintf(stderr, "nemotron: BUG cache_ch nil ci=%d il=%d n_ctx=%d\n", ci, il, n_ctx);
                    return false;
                }
                int off = cache.n_cached - n_ctx;
                ggml_backend_tensor_set(cache_t, cache.k_cache.data() + (size_t)off * d, 0,
                                        (size_t)n_ctx * d * sizeof(float));
            }

            // Set conv cache (pre-DW-conv signal from previous chunk)
            if (has_conv) {
                ggml_tensor* cc_t = ggml_graph_get_tensor(lg.gf, "conv_cache_in");
                if (cc_t) {
                    ggml_backend_tensor_set(cc_t, cache.conv_cache.data(), 0, (size_t)(K - 1) * d * sizeof(float));
                }
            }

            // Set pos enc for the full attention window (cache + new)
            int T_full = n_ctx + n_new;
            auto pe = core_conformer::make_pos_enc(d, T_full);
            ggml_tensor* pos_t = ggml_graph_get_tensor(lg.gf, "pos_enc");
            if (!pos_t) {
                fprintf(stderr, "nemotron: pos_enc tensor missing (il=%d)\n", il);
                return false;
            }
            ggml_backend_tensor_set(pos_t, pe.data(), 0, pe.size() * sizeof(float));

            if (ggml_backend_sched_graph_compute(ctx->sched, lg.gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "nemotron: streaming layer %d compute failed\n", il);
                return false;
            }

            // Read block output (new frames only, shape (d, n_new))
            ggml_tensor* out_t = ggml_graph_get_tensor(lg.gf, "block_out");
            chunk_in.resize((size_t)n_new * d);
            ggml_backend_tensor_get(out_t, chunk_in.data(), 0, chunk_in.size() * sizeof(float));

            // Read post-FFN1 output for cache update
            ggml_tensor* cache_out = ggml_graph_get_tensor(lg.gf, "cache_ch_out");
            if (cache_out) {
                std::vector<float> new_cache((size_t)n_new * d);
                ggml_backend_tensor_get(cache_out, new_cache.data(), 0, new_cache.size() * sizeof(float));

                // Append to cache and trim to L frames. Use memmove
                // instead of vector::erase to avoid O(N) element shifting
                // on every chunk (§176m).
                cache.k_cache.insert(cache.k_cache.end(), new_cache.begin(), new_cache.end());
                cache.n_cached += n_new;
                if (cache.n_cached > L) {
                    int excess = cache.n_cached - L;
                    size_t keep = (size_t)L * d;
                    std::memmove(cache.k_cache.data(), cache.k_cache.data() + (size_t)excess * d, keep * sizeof(float));
                    cache.k_cache.resize(keep);
                    cache.n_cached = L;
                }
            }

            // Read conv cache output (last K-1 frames of pre-DW-conv signal)
            ggml_tensor* conv_out = ggml_graph_get_tensor(lg.gf, "conv_cache_out");
            if (conv_out) {
                int conv_frames = (int)conv_out->ne[1];
                if (conv_frames == K - 1) {
                    cache.conv_cache.resize((size_t)(K - 1) * d);
                    ggml_backend_tensor_get(conv_out, cache.conv_cache.data(), 0, (size_t)(K - 1) * d * sizeof(float));
                    cache.conv_cached = K - 1;
                } else {
                    // Small chunk: save what we have, pad rest with zeros
                    cache.conv_cache.assign((size_t)(K - 1) * d, 0.0f);
                    size_t copy = (size_t)conv_frames * d;
                    size_t offset = (size_t)(K - 1 - conv_frames) * d;
                    ggml_backend_tensor_get(conv_out, cache.conv_cache.data() + offset, 0, copy * sizeof(float));
                    cache.conv_cached = K - 1;
                }
            }

            // Debug: print per-layer stats for first chunk
            if (ci == 0 && getenv("CRISPASR_NEMOTRON_DEBUG")) {
                float lmin = 1e30f, lmax = -1e30f;
                for (size_t i = 0; i < chunk_in.size(); i++) {
                    if (chunk_in[i] < lmin)
                        lmin = chunk_in[i];
                    if (chunk_in[i] > lmax)
                        lmax = chunk_in[i];
                }
                fprintf(stderr, "  layer %d output: min=%.2f max=%.2f\n", il, lmin, lmax);
            }
        }

        memcpy(enc_out.data() + (size_t)t_start * d, chunk_in.data(), (size_t)n_new * d * sizeof(float));

        if (ci % 5 == 0 || ci == n_chunks - 1) {
            fprintf(stderr, "  chunk %d/%d (frames %d-%d, graphs=%zu)\n", ci + 1, n_chunks, t_start, t_end - 1,
                    graph_cache.size());
        }
    }

    // Cleanup
    for (auto& [key, lg] : graph_cache) {
        ggml_free(lg.ctx0);
    }

    if (getenv("CRISPASR_NEMOTRON_DEBUG")) {
        float emin = 1e30f, emax = -1e30f, esum = 0.0f;
        for (size_t i = 0; i < enc_out.size(); i++) {
            float v = enc_out[i];
            if (v < emin)
                emin = v;
            if (v > emax)
                emax = v;
            esum += v;
        }
        fprintf(stderr, "nemotron: chunked enc_out T=%d d=%d min=%.4f max=%.4f mean=%.6f\n", T_enc, d, emin, emax,
                esum / (float)enc_out.size());
    }
    fprintf(stderr, "nemotron: chunked encoder done\n");
    return true;
}

// ===========================================================================
// LSTM step (CPU F32) — identical to parakeet's
// ===========================================================================

static void lstm_step_layer(const float* x_in, const float* w_ih, const float* b_ih, const float* w_hh,
                            const float* b_hh, float* h, float* c, float* h_out, int H, int in_dim) {
    auto sig = [](float v) { return 1.0f / (1.0f + expf(-v)); };
    const int H4 = 4 * H;
    std::vector<float> gates((size_t)H4);
    // gates = b_ih + b_hh
    for (int j = 0; j < H4; j++)
        gates[(size_t)j] = b_ih[j] + b_hh[j];
#if defined(HAVE_ACCELERATE)
    if (!nemotron_force_scalar()) {
        // gates += w_ih @ x_in + w_hh @ h
        cblas_sgemv(CblasRowMajor, CblasNoTrans, H4, in_dim, 1.0f, w_ih, in_dim, x_in, 1, 1.0f, gates.data(), 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, H4, H, 1.0f, w_hh, H, h, 1, 1.0f, gates.data(), 1);
    } else
#endif
    {
        for (int j = 0; j < H4; j++) {
            float s = gates[(size_t)j];
            for (int k = 0; k < in_dim; k++)
                s += w_ih[j * in_dim + k] * x_in[k];
            for (int k = 0; k < H; k++)
                s += w_hh[j * H + k] * h[k];
            gates[(size_t)j] = s;
        }
    }
    for (int j = 0; j < H; j++) {
        float i_g = sig(gates[0 * H + j]);
        float f_g = sig(gates[1 * H + j]);
        float g_g = tanhf(gates[2 * H + j]);
        float o_g = sig(gates[3 * H + j]);
        c[j] = f_g * c[j] + i_g * g_g;
        h_out[j] = o_g * tanhf(c[j]);
    }
}

static void predictor_step(const nemotron_predictor_weights& W, int token_id, nemotron_lstm_state& state,
                           std::vector<float>& pred_out) {
    const int H = W.H;
    pred_out.assign(H, 0.0f);

    std::vector<float> x(W.embed.data() + (size_t)token_id * H, W.embed.data() + (size_t)(token_id + 1) * H);

    std::vector<float> h0_new(H);
    lstm_step_layer(x.data(), W.w_ih_0.data(), W.b_ih_0.data(), W.w_hh_0.data(), W.b_hh_0.data(), state.h0.data(),
                    state.c0.data(), h0_new.data(), H, H);
    state.h0 = h0_new;

    std::vector<float> h1_new(H);
    lstm_step_layer(state.h0.data(), W.w_ih_1.data(), W.b_ih_1.data(), W.w_hh_1.data(), W.b_hh_1.data(),
                    state.h1.data(), state.c1.data(), h1_new.data(), H, H);
    state.h1 = h1_new;

    pred_out = state.h1;
}

// ===========================================================================
// Joint head (CPU F32) — ReLU activation (NeMo default)
// ===========================================================================

static void joint_proj_enc(const nemotron_joint_weights& J, const float* enc_t, std::vector<float>& out) {
    out.assign(J.joint_hidden, 0.0f);
#if defined(HAVE_ACCELERATE)
    if (!nemotron_force_scalar()) {
        std::memcpy(out.data(), J.enc_b.data(), (size_t)J.joint_hidden * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, J.joint_hidden, J.d_model, 1.0f, J.enc_w.data(), J.d_model, enc_t, 1,
                    1.0f, out.data(), 1);
    } else
#endif
    {
        for (int i = 0; i < J.joint_hidden; i++) {
            float s = J.enc_b[i];
            const float* row = J.enc_w.data() + (size_t)i * J.d_model;
            for (int k = 0; k < J.d_model; k++)
                s += row[k] * enc_t[k];
            out[i] = s;
        }
    }
}

static void joint_step(const nemotron_joint_weights& J, const float* proj_enc, const float* pred_u,
                       std::vector<float>& logits) {
    std::vector<float> mid(J.joint_hidden);
#if defined(HAVE_ACCELERATE)
    if (!nemotron_force_scalar()) {
        // mid = pred_b + pred_w @ pred_u
        std::memcpy(mid.data(), J.pred_b.data(), (size_t)J.joint_hidden * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, J.joint_hidden, J.pred_hidden, 1.0f, J.pred_w.data(), J.pred_hidden,
                    pred_u, 1, 1.0f, mid.data(), 1);
        // mid = ReLU(proj_enc + mid)
        for (int i = 0; i < J.joint_hidden; i++) {
            float v = proj_enc[i] + mid[i];
            mid[i] = v > 0.0f ? v : 0.0f;
        }
        // logits = out_b + out_w @ mid
        logits.assign(J.vocab_total, 0.0f);
        std::memcpy(logits.data(), J.out_b.data(), (size_t)J.vocab_total * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, J.vocab_total, J.joint_hidden, 1.0f, J.out_w.data(), J.joint_hidden,
                    mid.data(), 1, 1.0f, logits.data(), 1);
    } else
#endif
    {
        for (int i = 0; i < J.joint_hidden; i++) {
            float s = J.pred_b[i];
            const float* row = J.pred_w.data() + (size_t)i * J.pred_hidden;
            for (int k = 0; k < J.pred_hidden; k++)
                s += row[k] * pred_u[k];
            float v = proj_enc[i] + s;
            mid[i] = v > 0.0f ? v : 0.0f; // ReLU
        }
        logits.assign(J.vocab_total, 0.0f);
        for (int v = 0; v < J.vocab_total; v++) {
            float s = J.out_b[v];
            const float* row = J.out_w.data() + (size_t)v * J.joint_hidden;
            for (int k = 0; k < J.joint_hidden; k++)
                s += row[k] * mid[k];
            logits[v] = s;
        }
    }
}

// ===========================================================================
// Lazy weight cache init
// ===========================================================================

static void nemotron_init_pred_weights(nemotron_context* ctx) {
    if (ctx->pred_w.initialised)
        return;
    auto& p = ctx->model.predictor;
    auto& W = ctx->pred_w;
    const int H = (int)ctx->model.hparams.pred_hidden;

    W.embed = tensor_to_f32(p.embed_w);
    W.w_ih_0 = tensor_to_f32(p.lstm0_w_ih);
    W.w_hh_0 = tensor_to_f32(p.lstm0_w_hh);
    W.b_ih_0 = tensor_to_f32(p.lstm0_b_ih);
    W.b_hh_0 = tensor_to_f32(p.lstm0_b_hh);
    W.w_ih_1 = tensor_to_f32(p.lstm1_w_ih);
    W.w_hh_1 = tensor_to_f32(p.lstm1_w_hh);
    W.b_ih_1 = tensor_to_f32(p.lstm1_b_ih);
    W.b_hh_1 = tensor_to_f32(p.lstm1_b_hh);
    W.H = H;
    W.initialised = true;
}

static void nemotron_init_joint_weights(nemotron_context* ctx) {
    if (ctx->joint_w.initialised)
        return;
    auto& j = ctx->model.joint;
    auto& J = ctx->joint_w;
    const auto& hp = ctx->model.hparams;

    J.enc_w = tensor_to_f32(j.enc_w);
    J.enc_b = tensor_to_f32(j.enc_b);
    J.pred_w = tensor_to_f32(j.pred_w);
    J.pred_b = tensor_to_f32(j.pred_b);
    J.out_w = tensor_to_f32(j.out_w);
    J.out_b = tensor_to_f32(j.out_b);
    J.joint_hidden = (int)hp.joint_hidden;
    J.d_model = (int)hp.d_model;
    J.pred_hidden = (int)hp.pred_hidden;
    J.vocab_total = (int)j.out_b->ne[0]; // vocab + 1 blank
    J.initialised = true;
}

// ===========================================================================
// RNN-T greedy decode (pure RNNT, no TDT durations)
// ===========================================================================

struct nemotron_emitted_token {
    int id;
    int t_start;
    int t_end;
    float p;
};

static std::vector<nemotron_emitted_token> nemotron_rnnt_decode(nemotron_context* ctx, const float* enc, int T_enc,
                                                                int d_model, nemotron_token_cb on_tok = nullptr,
                                                                void* on_tok_ud = nullptr) {
    nemotron_init_pred_weights(ctx);
    nemotron_init_joint_weights(ctx);

    const auto& W = ctx->pred_w;
    const auto& J = ctx->joint_w;
    const int blank_id = (int)ctx->model.hparams.blank_id;
    const int max_symbols_per_frame = 10;

    std::vector<nemotron_emitted_token> emitted;
    nemotron_lstm_state state;
    state.init(W.H);

    std::vector<float> pred_out;
    predictor_step(W, blank_id, state, pred_out);

    for (int t = 0; t < T_enc; t++) {
        const float* enc_t = enc + (size_t)t * d_model;
        std::vector<float> proj_e;
        joint_proj_enc(J, enc_t, proj_e);

        int sym_count = 0;
        while (sym_count < max_symbols_per_frame) {
            std::vector<float> logits;
            joint_step(J, proj_e.data(), pred_out.data(), logits);

            // Softmax for probability
            float maxl = *std::max_element(logits.begin(), logits.end());
            float sum = 0.0f;
            for (auto& l : logits) {
                l = expf(l - maxl);
                sum += l;
            }
            for (auto& l : logits)
                l /= sum;

            int tok = 0;
            float maxp = logits[0];
            for (int v = 1; v < J.vocab_total; v++) {
                if (logits[v] > maxp) {
                    maxp = logits[v];
                    tok = v;
                }
            }

            if (tok == blank_id)
                break;

            nemotron_emitted_token et;
            et.id = tok;
            et.t_start = t;
            et.t_end = t + 1;
            et.p = logits[tok];
            emitted.push_back(et);
            if (on_tok)
                on_tok(tok, logits[tok], on_tok_ud);

            predictor_step(W, tok, state, pred_out);
            sym_count++;
        }
    }
    return emitted;
}

// ===========================================================================
// RNN-T beam search decode
// ===========================================================================
// Same structure as greedy decode above but expands top-B hypotheses at
// each step and prunes globally by cumulative log-probability.
// When beam_size==1 produces bit-identical output to greedy.
// LSTM state snapshots are plain vector copies (~few KB per beam).

static std::vector<nemotron_emitted_token> nemotron_rnnt_beam_decode(nemotron_context* ctx, const float* enc, int T_enc,
                                                                     int d_model, int beam_size) {
    nemotron_init_pred_weights(ctx);
    nemotron_init_joint_weights(ctx);

    const auto& W = ctx->pred_w;
    const auto& J = ctx->joint_w;
    const int blank_id = (int)ctx->model.hparams.blank_id;
    const int n_vocab = J.vocab_total; // vocab + blank
    const int max_per_step = 10;
    const int B = std::max(1, beam_size);

    struct Hyp {
        nemotron_lstm_state lstm;
        std::vector<float> pred_out;
        int t = 0;
        int n_inner = 0;
        double cum_logprob = 0.0;
        std::vector<nemotron_emitted_token> emitted;
        bool active = true;
    };

    // Seed: single hypothesis at t=0 with SOS predictor state
    std::vector<Hyp> beam(1);
    {
        auto& h = beam[0];
        h.lstm.init(W.H);
        predictor_step(W, blank_id, h.lstm, h.pred_out);
        h.emitted.reserve(256);
    }

    std::vector<float> proj_e(J.joint_hidden);
    std::vector<float> logits(n_vocab);

    for (;;) {
        bool any_active = false;
        for (const auto& h : beam)
            if (h.active) {
                any_active = true;
                break;
            }
        if (!any_active)
            break;

        struct Candidate {
            int parent;
            int token;
            double cum_logprob;
            float tok_p;
        };
        std::vector<Candidate> cands;
        cands.reserve((size_t)beam.size() * (size_t)(B + 1));

        for (int bi = 0; bi < (int)beam.size(); bi++) {
            auto& h = beam[bi];
            if (!h.active) {
                cands.push_back({bi, -1, h.cum_logprob, 0.0f});
                continue;
            }

            joint_proj_enc(J, enc + (size_t)h.t * d_model, proj_e);
            joint_step(J, proj_e.data(), h.pred_out.data(), logits);

            // Log-partition (log-sum-exp) over all tokens
            float max_logit = logits[0];
            for (int v = 1; v < n_vocab; v++)
                if (logits[v] > max_logit)
                    max_logit = logits[v];
            double logZ = 0.0;
            for (int v = 0; v < n_vocab; v++)
                logZ += std::exp((double)(logits[v] - max_logit));
            logZ = (double)max_logit + std::log(logZ);

            // Top-B tokens
            std::vector<int> top_ids(std::min(B, n_vocab));
            std::vector<float> top_vals(top_ids.size(), -1e30f);
            for (int v = 0; v < n_vocab; v++) {
                int mi = 0;
                for (int j = 1; j < (int)top_ids.size(); j++)
                    if (top_vals[j] < top_vals[mi])
                        mi = j;
                if (logits[v] > top_vals[mi]) {
                    top_vals[mi] = logits[v];
                    top_ids[mi] = v;
                }
            }
            // Ensure blank is always a candidate
            bool has_blank = false;
            for (int id : top_ids)
                if (id == blank_id) {
                    has_blank = true;
                    break;
                }
            if (!has_blank)
                top_ids.push_back(blank_id);

            for (int id : top_ids) {
                double log_p = (double)logits[id] - logZ;
                float tok_p = (float)std::exp(log_p);
                cands.push_back({bi, id, h.cum_logprob + log_p, tok_p});
            }
        }

        // Global prune: keep top-B
        const size_t keep = std::min<size_t>((size_t)B, cands.size());
        std::partial_sort(cands.begin(), cands.begin() + (ptrdiff_t)keep, cands.end(),
                          [](const Candidate& a, const Candidate& b) { return a.cum_logprob > b.cum_logprob; });
        cands.resize(keep);

        std::vector<Hyp> next_beam;
        next_beam.reserve(keep);

        for (auto& c : cands) {
            const auto& parent = beam[c.parent];

            if (c.token < 0) {
                next_beam.push_back(parent);
                next_beam.back().active = false;
                continue;
            }

            Hyp nh;
            nh.lstm = parent.lstm;
            nh.pred_out = parent.pred_out;
            nh.cum_logprob = c.cum_logprob;
            nh.emitted = parent.emitted;

            if (c.token == blank_id) {
                // Blank: advance frame by 1
                nh.t = parent.t + 1;
                nh.n_inner = 0;
            } else {
                // Real token: emit, advance predictor, stay on frame
                nh.emitted.push_back({c.token, parent.t, parent.t + 1, c.tok_p});
                predictor_step(W, c.token, nh.lstm, nh.pred_out);
                nh.t = parent.t;
                nh.n_inner = parent.n_inner + 1;
                if (nh.n_inner >= max_per_step) {
                    nh.t = parent.t + 1;
                    nh.n_inner = 0;
                }
            }

            nh.active = (nh.t < T_enc);
            next_beam.push_back(std::move(nh));
        }

        beam = std::move(next_beam);
    }

    if (beam.empty())
        return {};
    int best = 0;
    for (int i = 1; i < (int)beam.size(); i++)
        if (beam[i].cum_logprob > beam[best].cum_logprob)
            best = i;
    return std::move(beam[best].emitted);
}

// ===========================================================================
// MAES (Modified Adaptive Expansion Search) for RNNT
// ===========================================================================
// Time-synchronous beam search: process ALL beams at the same encoder frame,
// allow up to maes_num_steps non-blank expansions per frame, prune with
// gamma-threshold. Blank always advances by 1 frame.

static std::vector<nemotron_emitted_token> nemotron_rnnt_maes_decode(nemotron_context* ctx, const float* enc, int T_enc,
                                                                     int d_model, int beam_size, int maes_num_steps = 2,
                                                                     float maes_gamma = 2.3f, int maes_beta = 2) {
    nemotron_init_pred_weights(ctx);
    nemotron_init_joint_weights(ctx);

    const auto& W = ctx->pred_w;
    const auto& J = ctx->joint_w;
    const int blank_id = (int)ctx->model.hparams.blank_id;
    const int n_vocab = J.vocab_total;
    const int B = std::max(1, beam_size);
    const int topk = B + maes_beta;

    struct Hyp {
        nemotron_lstm_state lstm;
        std::vector<float> pred_out;
        double score = 0.0;
        std::vector<nemotron_emitted_token> emitted;
    };

    std::vector<Hyp> kept(1);
    {
        auto& h = kept[0];
        h.lstm.init(W.H);
        predictor_step(W, blank_id, h.lstm, h.pred_out);
        h.emitted.reserve(256);
    }

    std::vector<float> proj_e(J.joint_hidden);
    std::vector<float> logits(n_vocab);

    for (int t = 0; t < T_enc; t++) {
        joint_proj_enc(J, enc + (size_t)t * d_model, proj_e);

        std::vector<Hyp> hyps = kept;
        std::vector<Hyp> list_b;

        for (int n = 0; n < maes_num_steps; n++) {
            std::vector<Hyp> list_exp;

            for (auto& h : hyps) {
                joint_step(J, proj_e.data(), h.pred_out.data(), logits);

                // Log-softmax
                float max_l = logits[0];
                for (int v = 1; v < n_vocab; v++)
                    if (logits[v] > max_l)
                        max_l = logits[v];
                double logZ = 0.0;
                for (int v = 0; v < n_vocab; v++)
                    logZ += std::exp((double)(logits[v] - max_l));
                logZ = (double)max_l + std::log(logZ);

                // Top-k + gamma pruning
                std::vector<std::pair<float, int>> topk_pairs(n_vocab);
                for (int v = 0; v < n_vocab; v++)
                    topk_pairs[v] = {logits[v], v};
                std::partial_sort(topk_pairs.begin(), topk_pairs.begin() + std::min(topk, n_vocab), topk_pairs.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });

                double best_exp = h.score + ((double)topk_pairs[0].first - logZ);
                for (int k = 0; k < std::min(topk, n_vocab); k++) {
                    double new_score = h.score + ((double)topk_pairs[k].first - logZ);
                    if (new_score < best_exp - (double)maes_gamma)
                        continue;

                    int tok = topk_pairs[k].second;
                    if (tok == blank_id) {
                        Hyp bh;
                        bh.lstm = h.lstm;
                        bh.pred_out = h.pred_out;
                        bh.score = new_score;
                        bh.emitted = h.emitted;
                        list_b.push_back(std::move(bh));
                    } else {
                        Hyp nh;
                        nh.lstm = h.lstm;
                        nh.score = new_score;
                        nh.emitted = h.emitted;
                        nh.emitted.push_back({tok, t, t + 1, (float)std::exp(new_score - h.score)});
                        predictor_step(W, tok, nh.lstm, nh.pred_out);
                        list_exp.push_back(std::move(nh));
                    }
                }
            }

            if (list_exp.empty())
                break;

            if (n < maes_num_steps - 1) {
                hyps = std::move(list_exp);
            } else {
                // Last expansion step: score remaining expansions with blank
                for (auto& nh : list_exp) {
                    joint_step(J, proj_e.data(), nh.pred_out.data(), logits);
                    float max_l = logits[0];
                    for (int v = 1; v < n_vocab; v++)
                        if (logits[v] > max_l)
                            max_l = logits[v];
                    double logZ2 = 0.0;
                    for (int v = 0; v < n_vocab; v++)
                        logZ2 += std::exp((double)(logits[v] - max_l));
                    logZ2 = (double)max_l + std::log(logZ2);
                    nh.score += ((double)logits[blank_id] - logZ2);
                    list_b.push_back(std::move(nh));
                }
            }
        }

        if ((int)list_b.size() > B) {
            std::partial_sort(list_b.begin(), list_b.begin() + B, list_b.end(),
                              [](const Hyp& a, const Hyp& b) { return a.score > b.score; });
            list_b.resize(B);
        }
        kept = std::move(list_b);
    }

    if (kept.empty())
        return {};
    int best = 0;
    for (int i = 1; i < (int)kept.size(); i++)
        if (kept[i].score > kept[best].score)
            best = i;
    return std::move(kept[best].emitted);
}

// ===========================================================================
// Token → text conversion
// ===========================================================================

static std::string nemotron_detokenize(const nemotron_vocab& vocab, const std::vector<nemotron_emitted_token>& tokens) {
    std::string out;
    for (const auto& t : tokens) {
        if (t.id < 0 || t.id >= (int)vocab.id_to_token.size())
            continue;
        std::string piece = vocab.id_to_token[t.id];
        // SentencePiece: replace '▁' with space
        size_t pos = 0;
        while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
            piece.replace(pos, 3, " ");
            pos += 1;
        }
        out += piece;
    }
    // Trim leading space
    if (!out.empty() && out[0] == ' ')
        out.erase(0, 1);
    return out;
}

// Group emitted sub-word tokens into words at '▁' boundaries.
static void nemotron_group_words(const nemotron_vocab& vocab, const std::vector<nemotron_emitted_token>& tokens,
                                 int frame_dur_cs, int64_t t_offset_cs, std::vector<nemotron_word_data>& words) {
    words.clear();
    nemotron_word_data cur_word = {};
    cur_word.t0 = 0;
    cur_word.t1 = 0;
    cur_word.p = 0.0f;
    int n_sub = 0;
    bool have_word = false;

    for (const auto& t : tokens) {
        if (t.id < 0 || t.id >= (int)vocab.id_to_token.size())
            continue;
        const std::string& piece = vocab.id_to_token[t.id];
        bool starts_word = (piece.find("\xe2\x96\x81") == 0);

        if (starts_word && have_word && n_sub > 0) {
            cur_word.p /= (float)n_sub;
            words.push_back(cur_word);
            cur_word = {};
            n_sub = 0;
        }

        if (!have_word || starts_word) {
            have_word = true;
            cur_word.t0 = t_offset_cs + (int64_t)t.t_start * frame_dur_cs;
        }

        // Append text
        std::string text = piece;
        size_t pos = 0;
        while ((pos = text.find("\xe2\x96\x81", pos)) != std::string::npos) {
            text.replace(pos, 3, "");
        }
        size_t len = strlen(cur_word.text);
        size_t avail = sizeof(cur_word.text) - len - 1;
        if (text.size() <= avail)
            memcpy(cur_word.text + len, text.c_str(), text.size() + 1);

        cur_word.t1 = t_offset_cs + (int64_t)t.t_end * frame_dur_cs;
        cur_word.p += t.p;
        n_sub++;
    }

    if (have_word && n_sub > 0) {
        cur_word.p /= (float)n_sub;
        words.push_back(cur_word);
    }
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" struct nemotron_context_params nemotron_context_default_params(void) {
    nemotron_context_params p;
    p.n_threads = 4;
    p.use_flash = false;
    p.verbosity = 1;
    p.use_gpu = false;
    return p;
}

extern "C" struct nemotron_context* nemotron_init_from_file(const char* path_model,
                                                            struct nemotron_context_params params) {
    auto* ctx = new nemotron_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Backend selection — use ggml_backend_init_best() for portable GPU init
    ctx->backend = nullptr;
    ctx->backend_cpu = ggml_backend_cpu_init();

    if (params.use_gpu) {
        ctx->backend = ggml_backend_init_best();
    }
    if (!ctx->backend) {
        ctx->backend = ctx->backend_cpu;
    }
    if (params.verbosity > 0) {
        fprintf(stderr, "nemotron: backend = %s\n", ggml_backend_name(ctx->backend));
    }

    // compute_meta buffer
    ctx->compute_meta.resize(16 * 1024 * 1024);

    // Load model
    if (!nemotron_load_model(ctx->model, ctx->vocab, ctx->lang_to_prompt, path_model, ctx->backend)) {
        fprintf(stderr, "nemotron: failed to load model from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    // CRISPASR_NEMOTRON_CONTEXT_PRESET=N selects attention context preset
    // 0: L=56, R=3  (streaming, chunk=4)
    // 1: L=56, R=0  (left-only)
    // 2: L=56, R=6  (chunk=7)
    // 3: L=56, R=13 (chunk=14, best quality)
    if (const char* s = getenv("CRISPASR_NEMOTRON_CONTEXT_PRESET")) {
        int p = atoi(s);
        if (p >= 0 && p < (int)ctx->model.hparams.n_att_context_presets) {
            ctx->att_context_preset = p;
            fprintf(stderr, "nemotron: context preset %d (L=%d, R=%d)\n", p, ctx->model.hparams.att_context_left[p],
                    ctx->model.hparams.att_context_right[p]);
        }
    }

    return ctx;
}

extern "C" void nemotron_free(struct nemotron_context* ctx) {
    if (!ctx)
        return;

    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);

    delete ctx;
}

extern "C" void nemotron_result_free(struct nemotron_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->tokens);
    free(r->words);
    free(r);
}

static nemotron_result* nemotron_transcribe_impl(nemotron_context* ctx, const float* samples, int n_samples,
                                                 int64_t t_offset_cs, nemotron_token_cb on_tok, void* on_tok_ud);

extern "C" char* nemotron_transcribe(struct nemotron_context* ctx, const float* samples, int n_samples) {
    nemotron_result* r = nemotron_transcribe_impl(ctx, samples, n_samples, 0, nullptr, nullptr);
    if (!r)
        return nullptr;
    char* text = r->text;
    r->text = nullptr;
    nemotron_result_free(r);
    return text;
}

static nemotron_result* nemotron_transcribe_impl(nemotron_context* ctx, const float* samples, int n_samples,
                                                 int64_t t_offset_cs, nemotron_token_cb on_tok, void* on_tok_ud) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    // Compute mel
    int T_mel = 0;
    std::vector<float> mel;
    {
        nemotron_bench_stage _b("mel");
        mel = nemotron_compute_mel_impl(ctx, samples, n_samples, T_mel);
    }
    if (mel.empty() || T_mel <= 0)
        return nullptr;

    // Run encoder — default: full-sequence with chunked_limited attention mask.
    // CRISPASR_NEMOTRON_STREAMING=1 selects the cache-aware streaming path.
    std::vector<float> enc_out;
    int T_enc = 0, d_model = 0;
    const bool use_chunked = getenv("CRISPASR_NEMOTRON_STREAMING");
    {
        nemotron_bench_stage _b("encoder");
        if (!use_chunked) {
            if (!nemotron_run_encoder(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, enc_out, T_enc, d_model))
                return nullptr;
        } else {
            // Cache-aware streaming encoder: pre-encode → chunked conformer layers.
            // Step 1: run pre-encode only, step 2: chunked conformer with cache_last_channel.
            fprintf(stderr, "nemotron: running streaming chunked encoder path\n");

            // Build pre-encode-only graph
            {
                const auto& hp2 = ctx->model.hparams;
                size_t meta_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(1024, false);
                std::vector<uint8_t> meta(meta_size);
                ggml_init_params ip = {meta_size, meta.data(), true};
                ggml_context* ctx0 = ggml_init(ip);
                ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 1024, false);

                ggml_tensor* mel_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp2.n_mels, T_mel);
                ggml_set_name(mel_t, "mel");
                ggml_set_input(mel_t);

                int T_pre = 0;
                ggml_tensor* pre = nemotron_build_pre_encode(ctx0, mel_t, ctx->model.pre_encode,
                                                             (int)hp2.subsampling_channels, &T_pre);
                ggml_set_name(pre, "pre_enc");
                ggml_set_output(pre);
                ggml_build_forward_expand(gf, pre);

                if (!nemotron_ensure_sched(ctx)) {
                    ggml_free(ctx0);
                    return nullptr;
                }
                ggml_backend_sched_reset(ctx->sched);
                if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
                    fprintf(stderr, "nemotron: sched alloc pre-encode graph failed\n");
                    ggml_free(ctx0);
                    return nullptr;
                }

                ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
                ggml_backend_tensor_set(mel_in, mel.data(), 0, mel.size() * sizeof(float));

                if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "nemotron: pre-encode compute failed\n");
                    ggml_free(ctx0);
                    return nullptr;
                }

                ggml_tensor* pre_out = ggml_graph_get_tensor(gf, "pre_enc");
                T_enc = (int)pre_out->ne[1];
                d_model = (int)pre_out->ne[0];
                std::vector<float> pre_enc((size_t)T_enc * d_model);
                ggml_backend_tensor_get(pre_out, pre_enc.data(), 0, pre_enc.size() * sizeof(float));

                {
                    float pmin = 1e30f, pmax = -1e30f, psum = 0.0f;
                    for (size_t i = 0; i < pre_enc.size(); i++) {
                        float v = pre_enc[i];
                        if (v < pmin)
                            pmin = v;
                        if (v > pmax)
                            pmax = v;
                        psum += v;
                    }
                    fprintf(stderr, "nemotron: pre-encode T=%d d=%d min=%.2f max=%.2f mean=%.4f\n", T_enc, d_model,
                            pmin, pmax, psum / (float)pre_enc.size());
                }

                ggml_free(ctx0);

                // Step 2: run chunked encoder on pre-encode output
                enc_out.clear();
                if (!nemotron_run_encoder_chunked(ctx, pre_enc.data(), T_enc, d_model, enc_out))
                    return nullptr;
            }
        }

    } // nemotron_bench_stage encoder

    if (T_enc <= 0)
        return nullptr;

    // Apply prompt kernel (language conditioning) — CPU F32
    // concat(enc_out[d_model], lang_onehot[n_prompts]) → Linear(in→mid) → ReLU → Linear(mid→d_model)
    if (ctx->model.prompt_kernel.l0_w) {
        const auto& pk = ctx->model.prompt_kernel;
        const int n_prompts = (int)ctx->model.hparams.num_prompts;
        const int pk_in = d_model + n_prompts;
        const int pk_mid = (int)ctx->model.hparams.prompt_kernel_mid;
        const int prompt_id = ctx->prompt_id;

        // Load prompt kernel weights to CPU F32 (lazy)
        auto l0_w = tensor_to_f32(pk.l0_w); // (pk_mid, pk_in)
        auto l0_b = tensor_to_f32(pk.l0_b); // (pk_mid,)
        auto l2_w = tensor_to_f32(pk.l2_w); // (d_model, pk_mid)
        auto l2_b = tensor_to_f32(pk.l2_b); // (d_model,)

        // Build language one-hot
        std::vector<float> lang(n_prompts, 0.0f);
        if (prompt_id >= 0 && prompt_id < n_prompts)
            lang[prompt_id] = 1.0f;

        // Apply per-frame: for each t, concat(enc[t], lang) → linear1 → relu → linear2
        std::vector<float> prompted((size_t)T_enc * d_model);
        for (int t = 0; t < T_enc; t++) {
            // Concat: [enc[t][0..d_model], lang[0..n_prompts]]
            std::vector<float> cat(pk_in);
            memcpy(cat.data(), enc_out.data() + (size_t)t * d_model, d_model * sizeof(float));
            memcpy(cat.data() + d_model, lang.data(), n_prompts * sizeof(float));

            // Linear1 + ReLU
            std::vector<float> mid(pk_mid);
            for (int i = 0; i < pk_mid; i++) {
                float s = l0_b[i];
                const float* row = l0_w.data() + (size_t)i * pk_in;
                for (int k = 0; k < pk_in; k++)
                    s += row[k] * cat[k];
                mid[i] = s > 0.0f ? s : 0.0f; // ReLU
            }

            // Linear2
            float* out = prompted.data() + (size_t)t * d_model;
            for (int i = 0; i < d_model; i++) {
                float s = l2_b[i];
                const float* row = l2_w.data() + (size_t)i * pk_mid;
                for (int k = 0; k < pk_mid; k++)
                    s += row[k] * mid[k];
                out[i] = s;
            }
        }
        enc_out = std::move(prompted);
        fprintf(stderr, "nemotron: prompt kernel applied (prompt_id=%d)\n", prompt_id);
    }

    if (getenv("CRISPASR_NEMOTRON_DEBUG")) {
        float emin = 1e30f, emax = -1e30f;
        for (size_t i = 0; i < enc_out.size(); i++) {
            if (enc_out[i] < emin)
                emin = enc_out[i];
            if (enc_out[i] > emax)
                emax = enc_out[i];
        }
        fprintf(stderr, "nemotron: post-prompt enc_out min=%.4f max=%.4f\n", emin, emax);
        // Dump first 8 values of frames 0, 10, 50, 100
        for (int tf : {0, 10, 50, 100}) {
            if (tf < T_enc) {
                fprintf(stderr, "  frame %d:", tf);
                for (int k = 0; k < 8; k++)
                    fprintf(stderr, " %.4f", enc_out[(size_t)tf * d_model + k]);
                fprintf(stderr, "\n");
            }
        }
    }

    // RNN-T decode
    const int beam_sz = ctx->decode_beam_size;
    const bool use_maes = ctx->decode_maes && beam_sz > 1;
    decltype(nemotron_rnnt_decode(ctx, enc_out.data(), T_enc, d_model, on_tok, on_tok_ud)) emitted;
    {
        nemotron_bench_stage _b("rnnt_decode");
        emitted = use_maes        ? nemotron_rnnt_maes_decode(ctx, enc_out.data(), T_enc, d_model, beam_sz,
                                                              ctx->maes_num_steps, ctx->maes_gamma, ctx->maes_beta)
                  : (beam_sz > 1) ? nemotron_rnnt_beam_decode(ctx, enc_out.data(), T_enc, d_model, beam_sz)
                                  : nemotron_rnnt_decode(ctx, enc_out.data(), T_enc, d_model, on_tok, on_tok_ud);
    }

    if (getenv("CRISPASR_NEMOTRON_DEBUG")) {
        fprintf(stderr, "nemotron: RNNT emitted %zu tokens\n", emitted.size());
        for (size_t i = 0; i < std::min(emitted.size(), (size_t)5); i++) {
            fprintf(stderr, "  tok %zu: id=%d t=%d-%d p=%.3f\n", i, emitted[i].id, emitted[i].t_start, emitted[i].t_end,
                    emitted[i].p);
        }
    }

    // Build result
    auto* r = (nemotron_result*)calloc(1, sizeof(nemotron_result));
    std::string text = nemotron_detokenize(ctx->vocab, emitted);
    r->text = strdup(text.c_str());

    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;

    // Tokens
    r->n_tokens = (int)emitted.size();
    if (r->n_tokens > 0) {
        r->tokens = (nemotron_token_data*)calloc(r->n_tokens, sizeof(nemotron_token_data));
        for (int i = 0; i < r->n_tokens; i++) {
            auto& et = emitted[i];
            auto& td = r->tokens[i];
            td.id = et.id;
            td.t0 = t_offset_cs + (int64_t)et.t_start * frame_dur_cs;
            td.t1 = t_offset_cs + (int64_t)et.t_end * frame_dur_cs;
            td.p = et.p;
            if (et.id >= 0 && et.id < (int)ctx->vocab.id_to_token.size()) {
                std::string piece = ctx->vocab.id_to_token[et.id];
                size_t pos = 0;
                while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
                    piece.replace(pos, 3, " ");
                    pos += 1;
                }
                snprintf(td.text, sizeof(td.text), "%s", piece.c_str());
            }
        }
    }

    // Words
    std::vector<nemotron_word_data> words;
    nemotron_group_words(ctx->vocab, emitted, frame_dur_cs, t_offset_cs, words);
    r->n_words = (int)words.size();
    if (r->n_words > 0) {
        r->words = (nemotron_word_data*)malloc(r->n_words * sizeof(nemotron_word_data));
        memcpy(r->words, words.data(), r->n_words * sizeof(nemotron_word_data));
    }

    return r;
}

extern "C" struct nemotron_result* nemotron_transcribe_ex(struct nemotron_context* ctx, const float* samples,
                                                          int n_samples, int64_t t_offset_cs) {
    return nemotron_transcribe_impl(ctx, samples, n_samples, t_offset_cs, nullptr, nullptr);
}

extern "C" void nemotron_transcribe_cb(struct nemotron_context* ctx, const float* samples, int n_samples,
                                       nemotron_token_cb cb, void* userdata) {
    if (!ctx || !samples || n_samples <= 0 || !cb)
        return;
    nemotron_result* r = nemotron_transcribe_impl(ctx, samples, n_samples, 0, cb, userdata);
    nemotron_result_free(r);
}

extern "C" void nemotron_set_context_preset(struct nemotron_context* ctx, int preset) {
    if (!ctx)
        return;
    if (preset >= 0 && preset < (int)ctx->model.hparams.n_att_context_presets)
        ctx->att_context_preset = preset;
}

extern "C" void nemotron_set_language(struct nemotron_context* ctx, const char* lang_code) {
    if (!ctx || !lang_code)
        return;
    auto it = ctx->lang_to_prompt.find(lang_code);
    if (it != ctx->lang_to_prompt.end()) {
        ctx->prompt_id = it->second;
    } else {
        // Default to English (en-US = index 0 in NeMo's prompt_dictionary)
        fprintf(stderr, "nemotron: unknown language '%s', defaulting to en-US\n", lang_code);
        ctx->prompt_id = 0;
    }
}

extern "C" void nemotron_set_temperature(struct nemotron_context* ctx, float temperature, uint64_t seed) {
    if (!ctx)
        return;
    ctx->decode_temperature = temperature;
    ctx->decode_seed = seed;
}

extern "C" void nemotron_set_beam_size(struct nemotron_context* ctx, int beam_size) {
    if (!ctx)
        return;
    ctx->decode_beam_size = beam_size > 0 ? beam_size : 1;
}

extern "C" void nemotron_set_maes(struct nemotron_context* ctx, bool enable, int num_steps, float gamma, int beta) {
    if (!ctx)
        return;
    ctx->decode_maes = enable;
    if (enable) {
        ctx->maes_num_steps = num_steps > 0 ? num_steps : 2;
        ctx->maes_gamma = gamma > 0.0f ? gamma : 2.3f;
        ctx->maes_beta = beta > 0 ? beta : 2;
    }
}

extern "C" int nemotron_n_vocab(struct nemotron_context* ctx) {
    return ctx ? (int)ctx->model.hparams.vocab_size : 0;
}

extern "C" int nemotron_blank_id(struct nemotron_context* ctx) {
    return ctx ? (int)ctx->model.hparams.blank_id : 0;
}

extern "C" const char* nemotron_token_to_str(struct nemotron_context* ctx, int token_id) {
    if (!ctx || token_id < 0 || token_id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[token_id].c_str();
}

extern "C" int nemotron_frame_dur_cs(struct nemotron_context* ctx) {
    return ctx ? (int)ctx->model.hparams.frame_dur_cs : 0;
}

extern "C" int nemotron_n_mels(struct nemotron_context* ctx) {
    return ctx ? (int)ctx->model.hparams.n_mels : 0;
}

extern "C" int nemotron_sample_rate(struct nemotron_context* ctx) {
    return ctx ? (int)ctx->model.hparams.sample_rate : 0;
}
