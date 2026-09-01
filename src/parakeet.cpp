// parakeet.cpp — nvidia/parakeet-tdt-0.6b-v3 ggml runtime
//
// First iteration: loader + public C API skeleton.
// Encoder forward (FastConformer), predictor LSTM, joint head, and the
// TDT greedy decode loop will land in subsequent commits.
//
// Architecture summary (see parakeet-todo.md for the full plan):
//   Mel:       128 mels @ 16 kHz, n_fft=512, win=400, hop=160 (Hann window)
//   Encoder:   24× FastConformer block, d_model=1024, 8 heads, head_dim=128,
//              ff_dim=4096, conv kernel=9, 8× temporal subsampling via dw_striding
//   Predictor: embed(8193, 640) + 2-layer LSTM(640, 640)
//   Joint:     enc(1024→640) + pred(640→640) → tanh → linear(640 → 8198)
//              8198 = 8192 vocab + 1 blank + 5 TDT durations {0,1,2,3,4}

#include "parakeet.h"
#include "core/crispasr_env.h"
#include "parakeet_ja_detect.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "crispasr_imatrix.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif
#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "core/asr_context_bias.h"
#include "core/cpu_ops.h" // core_cpu::to_f32 (quantized-safe weight read)
#include "core/ctc.h"
#include "core/fastconformer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
static bool parakeet_use_scalar() {
    static int v = -1;
    if (v < 0)
        v = (crispasr_env::get("CRISPASR_PARAKEET_FORCE_SCALAR") != nullptr) ? 1 : 0;
    return v != 0;
}
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation — `PARAKEET_BENCH=1` for per-stage timings.
// ===========================================================================

static bool parakeet_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_PARAKEET_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct parakeet_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit parakeet_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~parakeet_bench_stage() {
        if (!parakeet_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  parakeet_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// CPU weight caches for the predictor LSTM and joint head
//
// Both are populated lazily (parakeet_init_pred_weights / _joint_weights)
// from the model's GGUF tensors and used by the manual F32 stepping in
// predictor_step / joint_step. Stored on the parakeet_context by value.
// ===========================================================================

struct parakeet_predictor_weights {
    std::vector<float> embed; // [vocab+1, H]
    std::vector<float> w_ih_0, w_hh_0, b_ih_0, b_hh_0;
    std::vector<float> w_ih_1, w_hh_1, b_ih_1, b_hh_1;
    int H = 0;
    int n_layers = 2;
    bool initialised = false;
};

struct parakeet_joint_weights {
    std::vector<float> enc_w, enc_b;   // (joint_hidden, d_model), (joint_hidden,)
    std::vector<float> pred_w, pred_b; // (joint_hidden, pred_hidden), (joint_hidden,)
    std::vector<float> out_w, out_b;   // (vocab_total, joint_hidden), (vocab_total,)
    int joint_hidden = 0;
    int d_model = 0;
    int pred_hidden = 0;
    int vocab_total = 0;
    bool initialised = false;
};

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct parakeet_hparams {
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
    bool xscaling = true; // RelPositionalEncoding scales encoder input by sqrt(d_model)
    uint32_t pred_hidden = 640;
    uint32_t pred_layers = 2;
    uint32_t joint_hidden = 640;
    uint32_t vocab_size = 8192;
    uint32_t blank_id = 8192;
    uint32_t n_tdt_durations = 5;
    uint32_t frame_dur_cs = 8; // 80 ms per encoder frame
    std::vector<int32_t> tdt_durations = {0, 1, 2, 3, 4};

    // Local attention (rel_pos_local_attn). -1 = full (global) attention.
    int32_t att_context_left = -1;
    int32_t att_context_right = -1;
    uint32_t global_tokens = 0; // positions that all tokens can attend to
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

// Pre-encode weights: exactly the shared FastConformer layout.
using parakeet_pre_encode = core_conformer::PreEncodeWeights;

// Per-layer tensor container: inherits the shared Conformer block weights
// and adds the BN tensors used only at load time (BN folding).
struct parakeet_enc_layer : core_conformer::BlockWeights {
    // Raw BN tensors (consumed by parakeet_fold_batchnorm and then unused).
    ggml_tensor *conv_bn_w = nullptr, *conv_bn_b = nullptr;
    ggml_tensor *conv_bn_rm = nullptr, *conv_bn_rv = nullptr;
};

struct parakeet_predictor {
    ggml_tensor* embed_w = nullptr; // (vocab+1, pred_hidden)
    // 2-layer LSTM (PyTorch convention: w_ih [4H, in], w_hh [4H, H], b_ih [4H], b_hh [4H])
    ggml_tensor *lstm0_w_ih = nullptr, *lstm0_w_hh = nullptr;
    ggml_tensor *lstm0_b_ih = nullptr, *lstm0_b_hh = nullptr;
    ggml_tensor *lstm1_w_ih = nullptr, *lstm1_w_hh = nullptr;
    ggml_tensor *lstm1_b_ih = nullptr, *lstm1_b_hh = nullptr;
};

struct parakeet_joint {
    ggml_tensor *enc_w = nullptr, *enc_b = nullptr;   // (joint_hidden, d_model)
    ggml_tensor *pred_w = nullptr, *pred_b = nullptr; // (joint_hidden, pred_hidden)
    ggml_tensor *out_w = nullptr, *out_b = nullptr;   // (vocab+1+n_dur, joint_hidden)
};

// ===========================================================================
// Model and vocabulary
// ===========================================================================

struct parakeet_model {
    parakeet_hparams hparams;

    // Mel preprocessor weights (baked into the .nemo checkpoint)
    ggml_tensor* mel_fb = nullptr;     // (1, n_mels, n_fft/2+1)
    ggml_tensor* mel_window = nullptr; // (win_length,)

    parakeet_pre_encode pre_encode;
    std::vector<parakeet_enc_layer> enc;
    parakeet_predictor predictor;
    parakeet_joint joint;

    // CTC head (hybrid TDT+CTC models). nullptr when not present.
    ggml_tensor* ctc_w = nullptr; // Conv1d(d_model, ctc_vocab, 1) → (ctc_vocab, d_model, 1)
    ggml_tensor* ctc_b = nullptr; // (ctc_vocab,)
    bool has_ctc = false;
    uint32_t ctc_vocab_size = 0;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // F32 conv_dw_w copies (created during BN fold, avoids per-forward F16→F32 cast)
    ggml_context* ctx_f32 = nullptr;
    ggml_backend_buffer_t buf_f32 = nullptr;

    // Q8_0 repack of the F16 conv pw1/pw2 weights (issue #81, CRISPASR_FC_PW_Q8)
    core_conformer::PwRepackBuf pw_q8;
    // Fused Q/K/V weight concat (issue #81, CRISPASR_FC_FUSED_QKV)
    core_conformer::PwRepackBuf qkv_fused;

    std::map<std::string, ggml_tensor*> tensors;
};

struct parakeet_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
};

struct parakeet_context {
    parakeet_context_params params;

    parakeet_model model;
    parakeet_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta; // metadata buffer for graph allocation

    // CPU-side weight caches for the predictor LSTM and joint head.
    // Lazy-initialised on first transcribe call.
    parakeet_predictor_weights pred_w;
    parakeet_joint_weights joint_w;

    int n_threads = 4;

    // CTC decode mode. When true and the model has a CTC head, use CTC
    // greedy decode instead of TDT. CTC is frame-synchronous and doesn't
    // suffer from chunk-boundary text loss on bidirectional encoders.
    bool decode_ctc = false;

    // Decode-time sampling controls. Default temperature == 0 keeps the
    // bit-identical pure-argmax path; > 0 switches the TDT decoder over
    // to numerically-stable softmax sampling on the vocab+blank logits.
    // Set via parakeet_set_temperature(); the field is sticky on the ctx.
    float decode_temperature = 0.0f;
    uint64_t decode_seed = 0;
    int decode_beam_size = 1; // 1 = greedy (default); >1 = TDT/RNNT beam search
    bool decode_maes = false; // MAES beam search (requires beam_size > 1)
    int maes_num_steps = 2;   // max non-blank expansions per frame
    float maes_gamma = 2.3f;  // gamma-threshold pruning
    int maes_beta = 2;        // extra candidates beyond beam_size

    // CTC-WS phrase-boost trie (PLAN #98). Set via parakeet_set_hotwords().
    core_context_bias::Trie hotword_trie;
    float hotword_boost = 2.0f; // per-frame prefix continuation boost

    // §176s: cached encoder graph — reused when T_mel matches.
    ggml_cgraph* cached_enc_gf = nullptr;
    std::vector<uint8_t> cached_enc_meta;
    int cached_enc_T_mel = 0;
};

// ---------------------------------------------------------------------------
// Transformer-XL relative-position shift.
// Same trick as cohere.cpp: a single ggml_view_3d that walks the BD matrix
// with stride (nb[1] - nb[0]) along the time axis, dropping the upper
// triangle so each row picks up the rel-pos score r_{j-i}. Zero-cost view.
// ---------------------------------------------------------------------------
// rel_shift moved to core_conformer::rel_shift in src/core/fastconformer.h.

// ===========================================================================
// Loader helpers — thin wrappers around core_gguf:: to preserve the
// existing call-site syntax (try_get(model, name) / require(model, name))
// while the underlying work lives in src/core/gguf_loader.
// ===========================================================================

#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static ggml_tensor* try_get(parakeet_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(parakeet_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "parakeet");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool parakeet_load_model(parakeet_model& model, parakeet_vocab& vocab, const char* path,
                                ggml_backend_t backend) {
    // ---- pass 1: read hparams + vocab via metadata-only context ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "parakeet.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "parakeet.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "parakeet.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "parakeet.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "parakeet.hop_length", hp.hop_length);
        hp.d_model = core_gguf::kv_u32(gctx, "parakeet.d_model", hp.d_model);
        hp.n_layers = core_gguf::kv_u32(gctx, "parakeet.n_layers", hp.n_layers);
        hp.n_heads = core_gguf::kv_u32(gctx, "parakeet.n_heads", hp.n_heads);
        hp.head_dim = core_gguf::kv_u32(gctx, "parakeet.head_dim", hp.head_dim);
        hp.ff_dim = core_gguf::kv_u32(gctx, "parakeet.ff_dim", hp.ff_dim);
        hp.subsampling_factor = core_gguf::kv_u32(gctx, "parakeet.subsampling_factor", hp.subsampling_factor);
        hp.subsampling_channels = core_gguf::kv_u32(gctx, "parakeet.subsampling_channels", hp.subsampling_channels);
        hp.conv_kernel = core_gguf::kv_u32(gctx, "parakeet.conv_kernel", hp.conv_kernel);
        hp.xscaling = core_gguf::kv_bool(gctx, "parakeet.xscaling", hp.xscaling);
        hp.pred_hidden = core_gguf::kv_u32(gctx, "parakeet.pred_hidden", hp.pred_hidden);
        hp.pred_layers = core_gguf::kv_u32(gctx, "parakeet.pred_layers", hp.pred_layers);
        hp.joint_hidden = core_gguf::kv_u32(gctx, "parakeet.joint_hidden", hp.joint_hidden);
        hp.vocab_size = core_gguf::kv_u32(gctx, "parakeet.vocab_size", hp.vocab_size);
        hp.blank_id = core_gguf::kv_u32(gctx, "parakeet.blank_id", hp.blank_id);
        hp.n_tdt_durations = core_gguf::kv_u32(gctx, "parakeet.n_tdt_durations", hp.n_tdt_durations);
        hp.frame_dur_cs = core_gguf::kv_u32(gctx, "parakeet.frame_dur_cs", hp.frame_dur_cs);

        // Local attention context (rel_pos_local_attn models).
        hp.att_context_left = core_gguf::kv_i32(gctx, "parakeet.att_context_left", hp.att_context_left);
        hp.att_context_right = core_gguf::kv_i32(gctx, "parakeet.att_context_right", hp.att_context_right);
        hp.global_tokens = core_gguf::kv_u32(gctx, "parakeet.global_tokens", hp.global_tokens);
        // Issue #89: NeMo supports switching a full-attention FastConformer to
        // rel_pos_local_attn at inference time (change_attention_model) for
        // long-form audio; the mask math is identical, only the window
        // changes. CRISPASR_PARAKEET_ATT_CONTEXT="L,R" (encoder frames,
        // 1 frame = 80 ms) applies the same switch here. "-1,-1" forces full
        // attention even for GGUFs that ship a local-attn context.
        if (const char* e = getenv("CRISPASR_PARAKEET_ATT_CONTEXT")) {
            int l = -1, r = -1;
            if (sscanf(e, "%d,%d", &l, &r) == 2) {
                hp.att_context_left = l;
                hp.att_context_right = r;
            }
        }

        // CTC head metadata (hybrid TDT+CTC models).
        model.has_ctc = core_gguf::kv_bool(gctx, "parakeet.has_ctc", false);
        if (model.has_ctc)
            model.ctc_vocab_size = core_gguf::kv_u32(gctx, "parakeet.ctc_vocab_size", 0);

        // Vocab: tokenizer.ggml.tokens is a string array.
        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: load tensor data via the shared helper ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "parakeet", wl)) {
        return false;
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // Pure-CTC guard. A NeMo EncDecCTCModelBPE (parakeet-ctc-*,
    // stt_*_fastconformer_ctc_*) has only an encoder + a CTC classification
    // head — no RNN-T prediction network and no joint. The parakeet backend is
    // the transducer (TDT/RNN-T) runtime and cannot run such a model. Detect it
    // up front and point the user at the CTC backend, rather than emitting a wall
    // of "required tensor not found" errors and then dereferencing null weights.
    // Such a GGUF carries arch "canary-ctc" and runs on --backend fastconformer-ctc.
    if (model.tensors.find("decoder.embed.weight") == model.tensors.end() &&
        model.tensors.find("decoder.lstm.0.w_ih") == model.tensors.end()) {
        fprintf(stderr,
                "parakeet: this GGUF has no RNN-T decoder/joint tensors — it is a pure CTC\n"
                "parakeet: model (EncDecCTCModelBPE), not a transducer, so it cannot run on the\n"
                "parakeet: parakeet (transducer) backend. Run it with the CTC backend instead:\n"
                "parakeet: drop '--backend parakeet' to auto-detect, or pass '--backend fastconformer-ctc'.\n"
                "parakeet: (If you converted it yourself, use\n"
                "parakeet:  models/convert-stt-fastconformer-ctc-to-gguf.py, not convert-parakeet-to-gguf.py.)\n");
        return false;
    }

    // ---- bind named tensors into the per-layer structs ----

    // Mel preprocessor (optional — may be absent if recomputed at runtime)
    model.mel_fb = try_get(model, "preprocessor.fb");
    model.mel_window = try_get(model, "preprocessor.window");

    // Pre-encode (subsampling)
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
        e.conv_dw_b = get("conv.dw.bias"); // synthetic — populated by BN fold
        e.conv_pw2_w = get("conv.pw2.weight");
        e.conv_pw2_b = try_("conv.pw2.bias");
        e.conv_bn_w = get("conv.bn.weight");
        e.conv_bn_b = get("conv.bn.bias");
        e.conv_bn_rm = get("conv.bn.running_mean");
        e.conv_bn_rv = get("conv.bn.running_var");

        e.norm_ff2_w = get("norm_ff2.weight");
        e.norm_ff2_b = get("norm_ff2.bias");
        e.ff2_l1_w = get("ff2.linear1.weight");
        e.ff2_l1_b = try_("ff2.linear1.bias");
        e.ff2_l2_w = get("ff2.linear2.weight");
        e.ff2_l2_b = try_("ff2.linear2.bias");

        e.norm_out_w = get("norm_out.weight");
        e.norm_out_b = get("norm_out.bias");
    }

    // Predictor
    auto& p = model.predictor;
    p.embed_w = require(model, "decoder.embed.weight");
    p.lstm0_w_ih = require(model, "decoder.lstm.0.w_ih");
    p.lstm0_w_hh = require(model, "decoder.lstm.0.w_hh");
    p.lstm0_b_ih = require(model, "decoder.lstm.0.b_ih");
    p.lstm0_b_hh = require(model, "decoder.lstm.0.b_hh");
    // lstm.1 only exists when pred_layers >= 2 (v2/v3/0.6b-ja). Single-LSTM
    // hybrids/RNNTs keep them null; predictor_step returns layer 0 directly.
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

    // CTC head (optional)
    if (model.has_ctc) {
        auto it_w = model.tensors.find("ctc.weight");
        auto it_b = model.tensors.find("ctc.bias");
        if (it_w != model.tensors.end() && it_b != model.tensors.end()) {
            model.ctc_w = it_w->second;
            model.ctc_b = it_b->second;
            fprintf(stderr, "parakeet: CTC head loaded (vocab=%u)\n", model.ctc_vocab_size);
        } else {
            fprintf(stderr, "parakeet: has_ctc=true but ctc tensors missing — falling back to TDT\n");
            model.has_ctc = false;
        }
    }

    fprintf(stderr, "parakeet: vocab=%u  d_model=%u  n_layers=%u  n_heads=%u  ff=%u  pred=%u  joint=%u\n",
            model.hparams.vocab_size, model.hparams.d_model, model.hparams.n_layers, model.hparams.n_heads,
            model.hparams.ff_dim, model.hparams.pred_hidden, model.hparams.joint_hidden);

    // Issue #89 follow-up (#221d): the JA model's TDT decode is degenerate at
    // q4-class quantisation (joint.pred / decoder.embed fall back to q4_0
    // inside q4_k mode and the autoregressive decode compounds the noise into
    // a fixed-point repetition loop) — while CTC decode over the same
    // quantised encoder is clean, and q8_0 TDT is byte-identical to F16.
    // Warn so q4_k users don't ship garbage silently.
    //
    // Issue #257: gate on vocab CONTENT (kana/kanji fraction), not vocab_size
    // <= 4096 — small-vocab ENGLISH models (parakeet-tdt-1.1b, vocab 1024) are
    // NOT JA and decode cleanly at q4_k, so the size heuristic mis-warned them
    // that their (correct) output was garbage.
    if (crispasr_parakeet::vocab_looks_japanese(vocab.id_to_token) && j.out_w && ggml_is_quantized(j.out_w->type) &&
        j.out_w->type != GGML_TYPE_Q8_0) {
        fprintf(stderr,
                "parakeet: WARNING: JA model with %s weights — TDT decode degrades to repetition "
                "loops at this quantisation. Use %sthe q8_0/F16 file for reliable TDT output.\n",
                ggml_type_name(j.out_w->type), model.has_ctc ? "--parakeet-decoder ctc (clean at q4) or " : "");
    }
    return true;
}

// ===========================================================================
// FFT (iterative Cooley-Tukey, real-input, N must be a power of 2)
// ===========================================================================

static void parakeet_fft_r2c(const float* in, int N, float* out) {
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
// NeMo-style mel spectrogram
//
// AudioToMelSpectrogramPreprocessor defaults for parakeet-tdt-0.6b-v3:
//   sample_rate=16000  features=128  n_fft=512
//   window_size=0.025 (= 400 samples)  window_stride=0.010 (= 160 samples)
//   window=hann  log=True  normalize=per_feature  dither=1e-5
//   mag_power=2.0  log_zero_guard_value=2^-24 ≈ 5.96e-8
//
// No pre-emphasis (unlike Cohere). Window and mel filterbank are loaded
// directly from the GGUF preprocessor.* tensors.
//
// The heavy lifting lives in src/core/mel.cpp now — this function just
// pulls the GGUF-stored window and filterbank, then delegates. The FFT
// function pointer keeps parakeet's own Cooley-Tukey implementation so
// numerical output is bit-exact with the pre-refactor version.
//
// Returns mel as a flat row-major [T, n_mels] (the layout the parakeet
// encoder expects — ne[0]=n_mels fastest means each frame's 128 mels are
// contiguous in memory).
// ===========================================================================

#include "core/mel.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)
#include "core/rnnt_ggml.h"        // §232 GPU transducer decode (shared)
#include "core/ggml_cpu_backend.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static std::vector<float> parakeet_compute_mel_impl(parakeet_context* ctx, const float* samples, int n_samples,
                                                    int& T_out) {
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int win = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels = (int)hp.n_mels;

    if (!ctx->model.mel_fb || !ctx->model.mel_window) {
        fprintf(stderr, "parakeet: missing preprocessor.fb or preprocessor.window in GGUF\n");
        return {};
    }

    // Pull window and filterbank from the GGUF.
    std::vector<float> window_raw((size_t)win);
    ggml_backend_tensor_get(ctx->model.mel_window, window_raw.data(), 0, win * sizeof(float));

    std::vector<float> mel_fb((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(ctx->model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    // Configure the shared helper for the NeMo cluster:
    //   ln + per-mel z-score, (T, n_mels) output, center-padded input,
    //   log_eps = 2^-24 (NeMo log_zero_guard_value), 0.97 pre-emphasis
    //   (NeMo AudioToMelSpectrogramPreprocessor default — applied at
    //   inference, missing it caused JA token deletions: issue #37).
    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = win;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.norm = core_mel::Normalization::PerFeatureZ;
    p.layout = core_mel::Layout::TimeMels;
    p.log_eps = (float)(1.0 / (1 << 24));
    p.center_pad = true;
    p.drop_last_frame = true; // NeMo returns feat_len = floor(n_samples/hop) frames
    p.preemph = 0.97f;

    auto mel = core_mel::compute(samples, n_samples, window_raw.data(), win, mel_fb.data(), n_freqs, parakeet_fft_r2c,
                                 p, T_out);
    return mel;
}

// Compute mel WITHOUT z-norm — returns raw log-mel in (T, n_mels) layout.
// Caller accumulates statistics across chunks and normalizes at the end.
static std::vector<float> parakeet_compute_mel_raw(parakeet_context* ctx, const float* samples, int n_samples,
                                                   int& T_out) {
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int win = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels = (int)hp.n_mels;

    if (!ctx->model.mel_fb || !ctx->model.mel_window)
        return {};

    std::vector<float> window_raw((size_t)win);
    ggml_backend_tensor_get(ctx->model.mel_window, window_raw.data(), 0, win * sizeof(float));
    std::vector<float> mel_fb((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(ctx->model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = win;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.norm = core_mel::Normalization::None; // <-- NO z-norm
    p.layout = core_mel::Layout::TimeMels;
    p.log_eps = (float)(1.0 / (1 << 24));
    p.center_pad = true;
    p.drop_last_frame = true;
    p.preemph = 0.97f;

    auto mel = core_mel::compute(samples, n_samples, window_raw.data(), win, mel_fb.data(), n_freqs, parakeet_fft_r2c,
                                 p, T_out);
    return mel;
}

// Apply PerFeatureZ normalization to a (T, n_mels) mel buffer in-place,
// using pre-computed per-band mean and inverse-std.
static void parakeet_apply_znorm(float* mel, int T, int n_mels, const double* band_mean, const float* band_inv_std) {
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < n_mels; m++) {
            mel[(size_t)t * n_mels + m] = (float)(mel[(size_t)t * n_mels + m] - band_mean[m]) * band_inv_std[m];
        }
    }
}

// ===========================================================================
// BatchNorm folding (load-time, once)
//
// Inference-time BN: y = (x - mean) / sqrt(var + eps) * gamma + beta
//                     = x * s + (beta - mean * s)   where s = gamma/sqrt(var+eps)
//
// Since mean / var are fixed after training, we fold s into the depthwise conv
// weights and absorb the bias shift into the synthetic conv_dw_b tensor (which
// the converter pre-allocated as zeros). After this the encoder graph drops
// the BN block entirely.
// ===========================================================================

static void parakeet_fold_batchnorm(parakeet_model& model, ggml_backend_t backend) {
    const int d = (int)model.hparams.d_model;
    const int K = (int)model.hparams.conv_kernel;
    const float eps = 1e-5f;
    const uint32_t n_layers = model.hparams.n_layers;

    // Collect folded F32 weights for F32 pre-cast allocation.
    std::vector<std::vector<float>> all_w_f32(n_layers);
    int n_f16_layers = 0;

    for (uint32_t il = 0; il < n_layers; il++) {
        auto& e = model.enc[il];
        if (!e.conv_dw_w || !e.conv_dw_b || !e.conv_bn_w || !e.conv_bn_b || !e.conv_bn_rm || !e.conv_bn_rv) {
            fprintf(stderr, "parakeet: BN fold: missing tensor on layer %u\n", il);
            return;
        }

        std::vector<float> bn_mean(d), bn_var(d), bn_w(d), bn_b(d);
        ggml_backend_tensor_get(e.conv_bn_rm, bn_mean.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_rv, bn_var.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_w, bn_w.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_b, bn_b.data(), 0, d * sizeof(float));

        std::vector<float> s(d);
        for (int c = 0; c < d; c++)
            s[c] = bn_w[c] / sqrtf(bn_var[c] + eps);

        // Fold s into conv_dw_w (ggml shape [K, 1, d]).
        {
            const size_t n = (size_t)K * d;
            std::vector<float> w_f32(n);
            if (e.conv_dw_w->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(e.conv_dw_w, w_f32.data(), 0, n * sizeof(float));
            } else {
                std::vector<ggml_fp16_t> tmp(n);
                ggml_backend_tensor_get(e.conv_dw_w, tmp.data(), 0, n * sizeof(ggml_fp16_t));
                for (size_t i = 0; i < n; i++)
                    w_f32[i] = ggml_fp16_to_fp32(tmp[i]);
            }
            for (int c = 0; c < d; c++)
                for (int ki = 0; ki < K; ki++)
                    w_f32[ki + c * K] *= s[c];
            if (e.conv_dw_w->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(e.conv_dw_w, w_f32.data(), 0, n * sizeof(float));
            } else {
                std::vector<ggml_fp16_t> tmp(n);
                ggml_fp32_to_fp16_row(w_f32.data(), tmp.data(), (int)n);
                ggml_backend_tensor_set(e.conv_dw_w, tmp.data(), 0, n * sizeof(ggml_fp16_t));
                // Save F32 for pre-cast
                all_w_f32[il] = std::move(w_f32);
                n_f16_layers++;
            }
        }

        // Fold into conv_dw_b: b[c] = (existing_b[c] - mean[c]) * s[c] + bn_b[c]
        std::vector<float> dw_b(d, 0.0f);
        ggml_backend_tensor_get(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
        for (int c = 0; c < d; c++)
            dw_b[c] = (dw_b[c] - bn_mean[c]) * s[c] + bn_b[c];
        ggml_backend_tensor_set(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
    }

    // Allocate F32 copies of F16 conv_dw_w so build_block skips the per-forward cast.
    if (n_f16_layers > 0) {
        ggml_init_params ip = {(size_t)n_f16_layers * ggml_tensor_overhead(), nullptr, true};
        model.ctx_f32 = ggml_init(ip);
        for (uint32_t il = 0; il < n_layers; il++) {
            if (all_w_f32[il].empty())
                continue;
            auto& e = model.enc[il];
            e.conv_dw_w_f32 = ggml_new_tensor_1d(model.ctx_f32, GGML_TYPE_F32, (int64_t)K * d);
        }
        model.buf_f32 = ggml_backend_alloc_ctx_tensors(model.ctx_f32, backend);
        for (uint32_t il = 0; il < n_layers; il++) {
            if (all_w_f32[il].empty())
                continue;
            auto& e = model.enc[il];
            ggml_backend_tensor_set(e.conv_dw_w_f32, all_w_f32[il].data(), 0, all_w_f32[il].size() * sizeof(float));
        }
    }

    fprintf(stderr, "parakeet: BN folded into conv_dw weights for %u layers\n", n_layers);
}

// ===========================================================================
// Encoder graph builder
//
// Input:  mel [n_mels, T_mel]
// Output: enc_out [d_model, T_enc]   where T_enc = T_mel / subsampling_factor
// ===========================================================================

static const float kLayerNormEps = 1e-5f;
static const float kBatchNormEps = 1e-5f;

static ggml_cgraph* parakeet_build_graph_encoder(parakeet_context* ctx, int T_mel) {
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

    // ----- Pre-encode (dw_striding subsampling 8×) -----
    int T = 0;
    ggml_tensor* cur = core_conformer::build_pre_encode(ctx0, mel, m.pre_encode, (int)hp.subsampling_channels, &T);

    // ----- xscaling: NeMo's RelPositionalEncoding multiplies the encoder
    // input by sqrt(d_model) before the conformer layers when the model's
    // `encoder.xscaling: true` (default for parakeet-tdt-0.6b-v3 and -ja).
    // Without it, every layer's input is 32× too small, the rel-pos
    // sinusoid is the same scale, and the model produces near-random
    // activations downstream.
    if (hp.xscaling) {
        const float xscale = sqrtf((float)hp.d_model);
        cur = ggml_scale(ctx0, cur, xscale);
    }

    // ----- Sinusoidal rel-pos table [d, 2T-1] -----
    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    // ----- Local attention mask (rel_pos_local_attn models) -----
    ggml_tensor* local_mask = nullptr;
    ggml_tensor* window_band_mask = nullptr;
    const bool use_local_attn =
        hp.att_context_left >= 0 && hp.att_context_right >= 0 && T > hp.att_context_left + hp.att_context_right + 1;
    // TRUE windowed attention: build the O(T·window) band mask instead of the T×T
    // full mask when the gate is on and the clip is long enough to benefit.
    const bool use_windowed_attn =
        core_conformer::fc_windowed_attn() &&
        core_conformer::fc_window_attn_applicable(T, hp.att_context_left, hp.att_context_right);
    if (getenv("CRISPASR_FC_MEM_DEBUG"))
        fprintf(stderr, "[fc-encode] T=%d windowed=%d local=%d\n", T, (int)use_windowed_attn, (int)use_local_attn);
    if (use_windowed_attn) {
        const int BS = core_conformer::fc_window_block_size(hp.att_context_left, hp.att_context_right);
        const int NB = (T + BS - 1) / BS;
        window_band_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 3 * BS, BS, NB);
        ggml_set_name(window_band_mask, "window_band_mask");
        ggml_set_input(window_band_mask);
    } else if (use_local_attn) {
        local_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
        ggml_set_name(local_mask, "local_attn_mask");
        ggml_set_input(local_mask);
    }

    // ----- 24× FastConformer block -----
    core_conformer::BlockParams bp = {
        (int)hp.d_model, (int)hp.n_heads,     (int)hp.head_dim,     (int)hp.conv_kernel,
        kLayerNormEps,   hp.att_context_left, hp.att_context_right, (int)hp.global_tokens,
    };
    bp.manual_attn = core_conformer::fc_gpu_manual_attn(ctx->backend);
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = core_conformer::build_block(ctx0, cur, pos_enc, T, m.enc[il], bp, local_mask, nullptr, window_band_mask);
    }

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// make_pos_enc moved to core_conformer::make_pos_enc in src/core/fastconformer.h.

// Run the encoder once. Returns enc_out as a flat row-major [T_enc, d_model].
// Caller computes T_enc as ceil(T_mel / subsampling_factor) (approximately —
// the actual value depends on the conv arithmetic and is reported back).
static std::vector<float> parakeet_encode_mel(parakeet_context* ctx, const float* mel, int n_mels, int T_mel,
                                              int* out_T_enc) {
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "parakeet: mel feature mismatch (%d vs %d)\n", n_mels, (int)ctx->model.hparams.n_mels);
        return {};
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
        crispasr_imatrix_install(ctx->sched); // no-op unless CRISPASR_IMATRIX_OUT is set
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    }

    // §176s cached the built encoder graph and reused it whenever T_mel
    // matched. That is NOT safe with the shared ggml_backend_sched: reusing
    // the same cgraph object across sched_reset / sched_alloc_graph cycles
    // leaves the cached compute tensors with stale buffer/data pointers from
    // the previous allocation, so the SECOND and every later encode of the
    // same T_mel reads partially-stale memory. The encoder output silently
    // shifts (e.g. std 0.020 → 0.014 on jfk) and the TDT decoder collapses to
    // all-blank → empty transcript. This breaks every repeated encode:
    // long-lived sessions (issue #208 batch callers / the server), the
    // streamed + chunked long-audio paths, and the silence-split long-form
    // path. It went unnoticed because the CLI does exactly one same-T_mel
    // encode per process.
    //
    // Default is therefore the correct rebuild-every-call path. The cache is
    // kept behind an opt-in env for benchmarking only.
    //
    // #208 follow-up (2026-07-01): the reporter asked whether a re-entrant
    // cache is worth building to speed up chunked long audio. Measured on M1
    // Metal (Q4_K 0.6B-v3, PARAKEET_ENC_PROBE) per uniform 20 s window:
    //   graph build ~0.3 ms | sched_alloc ~1 ms | GPU compute ~1000 ms.
    // Graph build+alloc is ~0.13 % of per-window time; a re-entrant cache
    // could only skip the ~0.3 ms build (sched_alloc still runs every call),
    // so it is a DUD — reimplementing it re-entrantly gains nothing. The
    // headroom in chunked long-audio is the encoder GPU compute itself (and
    // the ~40 % overlap re-encode in parakeet_session_chunked_merge), NOT the
    // graph cache. Kept opt-in-OFF as a benchmark hook, not on the roadmap.
    // Enable PARAKEET_ENC_PROBE to reproduce the measurement and the 2nd-reuse
    // corruption (reuse windows collapse enc std 0.020 → 0.008). See issue #208.
    ggml_cgraph* gf;
    const bool use_enc_cache = getenv("CRISPASR_PARAKEET_ENC_CACHE") != nullptr;
    const bool probe_time = crispasr_env::get("CRISPASR_PARAKEET_ENC_PROBE") != nullptr;
    bool enc_cache_hit = false;
    int64_t t_build0 = probe_time ? ggml_time_us() : 0;
    if (use_enc_cache && ctx->cached_enc_gf && ctx->cached_enc_T_mel == T_mel) {
        gf = ctx->cached_enc_gf;
        enc_cache_hit = true;
    } else if (use_enc_cache) {
        ctx->cached_enc_meta.assign(ctx->compute_meta.size(), 0);
        std::swap(ctx->compute_meta, ctx->cached_enc_meta);
        gf = parakeet_build_graph_encoder(ctx, T_mel);
        std::swap(ctx->compute_meta, ctx->cached_enc_meta);
        ctx->cached_enc_gf = gf;
        ctx->cached_enc_T_mel = T_mel;
    } else {
        gf = parakeet_build_graph_encoder(ctx, T_mel);
    }
    int64_t t_build_us = probe_time ? ggml_time_us() - t_build0 : 0;

    int64_t t_alloc0 = probe_time ? ggml_time_us() : 0;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "parakeet: failed to alloc encoder graph\n");
        return {};
    }
    int64_t t_alloc_us = probe_time ? ggml_time_us() - t_alloc0 : 0;

    // Set inputs
    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2; // pos_enc has 2T-1 columns; recover T
    auto pe = core_conformer::make_pos_enc((int)ctx->model.hparams.d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    // Local attention mask (when present in the graph).
    ggml_tensor* local_mask_in = ggml_graph_get_tensor(gf, "local_attn_mask");
    if (local_mask_in) {
        const auto& hp = ctx->model.hparams;
        auto lm = core_conformer::make_local_attn_mask(T_enc, hp.att_context_left, hp.att_context_right,
                                                       (int)hp.global_tokens);
        ggml_backend_tensor_set(local_mask_in, lm.data(), 0, lm.size() * sizeof(float));
    }

    // Windowed-attention band mask (when present in the graph).
    ggml_tensor* band_mask_in = ggml_graph_get_tensor(gf, "window_band_mask");
    if (band_mask_in) {
        const auto& hp = ctx->model.hparams;
        const int BS = core_conformer::fc_window_block_size(hp.att_context_left, hp.att_context_right);
        auto bm = core_conformer::make_window_band_mask(T_enc, hp.att_context_left, hp.att_context_right,
                                                        (int)hp.global_tokens, BS);
        ggml_backend_tensor_set(band_mask_in, bm.data(), 0, bm.size() * sizeof(float));
    }

    // Compute
    int64_t t_comp0 = probe_time ? ggml_time_us() : 0;
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "parakeet: encoder graph compute failed\n");
        return {};
    }
    int64_t t_comp_us = probe_time ? ggml_time_us() - t_comp0 : 0;

    ggml_tensor* out = ggml_graph_get_tensor(gf, "enc_out");
    if (!out) {
        fprintf(stderr, "parakeet: missing enc_out tensor\n");
        return {};
    }
    const int d = (int)out->ne[0];
    const int Te = (int)out->ne[1];
    if (out_T_enc)
        *out_T_enc = Te;

    std::vector<float> result((size_t)d * Te);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    if (crispasr_env::get("CRISPASR_PARAKEET_ENC_PROBE")) {
        double s = 0, s2 = 0;
        const size_t n = result.size();
        for (float v : result) {
            s += v;
            s2 += (double)v * v;
        }
        const double mean = s / n;
        const double var = s2 / n - mean * mean;
        static int call_idx = 0;
        fprintf(stderr,
                "[enc-probe] call=%d T_mel=%d T_enc=%d cache=%s mean=%.6f std=%.6f first=%.6f,%.6f "
                "build=%.2fms alloc=%.2fms compute=%.2fms\n",
                call_idx++, T_mel, Te, !use_enc_cache ? "off" : (enc_cache_hit ? "reuse" : "fill"), mean,
                var > 0 ? sqrt(var) : 0.0, n > 0 ? result[0] : 0.0, n > 1 ? result[1] : 0.0, t_build_us / 1000.0,
                t_alloc_us / 1000.0, t_comp_us / 1000.0);
    }
    return result;
}

// ===========================================================================
// LSTM predictor (manual F32 step on the CPU)
//
// We don't go through ggml here — the predictor runs once per *emitted token*
// (not per encoder frame), the input is a single 640-vector, and the work
// per step is two small (640 → 4*640) GEMMs. A direct loop is simpler than
// building a per-step graph and the performance ceiling is identical.
//
// PyTorch LSTM weight layout:
//   weight_ih [4H, in_dim]   gates packed as [i, f, g, o]
//   weight_hh [4H, H]
//   bias_ih   [4H]
//   bias_hh   [4H]
// Forward (per layer):
//   gates = weight_ih @ x + bias_ih + weight_hh @ h + bias_hh
//   i = sigmoid(gates[0..H])
//   f = sigmoid(gates[H..2H])
//   g = tanh   (gates[2H..3H])
//   o = sigmoid(gates[3H..4H])
//   c' = f * c + i * g
//   h' = o * tanh(c')
// ===========================================================================

struct parakeet_lstm_state {
    std::vector<float> h0, c0; // layer 0
    std::vector<float> h1, c1; // layer 1
};

static void lstm_init_state(parakeet_lstm_state& s, int H) {
    s.h0.assign(H, 0.0f);
    s.c0.assign(H, 0.0f);
    s.h1.assign(H, 0.0f);
    s.c1.assign(H, 0.0f);
}

// Read an F16/F32 ggml tensor into a flat F32 std::vector for CPU stepping.
static std::vector<float> tensor_to_f32(ggml_tensor* t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++)
            out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        // Quantised types: dequantise via ggml-cpu helper
        const struct ggml_type_traits* tr = ggml_get_type_traits(t->type);
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        tr->to_float(raw.data(), out.data(), n);
    }
    return out;
}

static void lstm_step_layer(const float* x, // [in_dim]
                            const float* w_ih, const float* b_ih, const float* w_hh, const float* b_hh, float* h,
                            float* c,     // [H]   in/out
                            float* h_out, // [H]   out
                            int in_dim, int H) {
    const int H4 = 4 * H;
    std::vector<float> gates(H4);

    // gates = b_ih + b_hh + w_ih @ x + w_hh @ h
    for (int i = 0; i < H4; i++)
        gates[i] = b_ih[i] + b_hh[i];

#if defined(HAVE_ACCELERATE)
    if (!parakeet_use_scalar()) {
        // w_ih[4H, in_dim] @ x[in_dim] and w_hh[4H, H] @ h[H], adding into gates
        cblas_sgemv(CblasRowMajor, CblasNoTrans, H4, in_dim, 1.0f, w_ih, in_dim, x, 1, 1.0f, gates.data(), 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, H4, H, 1.0f, w_hh, H, h, 1, 1.0f, gates.data(), 1);
    } else {
#endif
        for (int i = 0; i < H4; i++) {
            const float* row = w_ih + (size_t)i * in_dim;
            float s = 0.0f;
            for (int k = 0; k < in_dim; k++)
                s += row[k] * x[k];
            gates[i] += s;
        }
        for (int i = 0; i < H4; i++) {
            const float* row = w_hh + (size_t)i * H;
            float s = 0.0f;
            for (int k = 0; k < H; k++)
                s += row[k] * h[k];
            gates[i] += s;
        }
#if defined(HAVE_ACCELERATE)
    }
#endif

    auto sig = [](float x) { return 1.0f / (1.0f + expf(-x)); };

    for (int j = 0; j < H; j++) {
        float i_g = sig(gates[0 * H + j]);
        float f_g = sig(gates[1 * H + j]);
        float g_g = tanhf(gates[2 * H + j]);
        float o_g = sig(gates[3 * H + j]);
        c[j] = f_g * c[j] + i_g * g_g;
        h_out[j] = o_g * tanhf(c[j]);
    }
}

// Run one predictor step:  input token id  →  pred_out [H]
static void predictor_step(const parakeet_predictor_weights& W, int token_id, parakeet_lstm_state& state,
                           std::vector<float>& pred_out) {
    const int H = W.H;
    pred_out.assign(H, 0.0f);

    // Embed token
    std::vector<float> x(W.embed.data() + (size_t)token_id * H, W.embed.data() + (size_t)(token_id + 1) * H);

    // Layer 0
    std::vector<float> h0_new(H);
    lstm_step_layer(x.data(), W.w_ih_0.data(), W.b_ih_0.data(), W.w_hh_0.data(), W.b_hh_0.data(), state.h0.data(),
                    state.c0.data(), h0_new.data(), H, H);
    state.h0 = h0_new;

    if (W.n_layers >= 2) {
        // Layer 1 — input is layer 0's hidden.
        std::vector<float> h1_new(H);
        lstm_step_layer(state.h0.data(), W.w_ih_1.data(), W.b_ih_1.data(), W.w_hh_1.data(), W.b_hh_1.data(),
                        state.h1.data(), state.c1.data(), h1_new.data(), H, H);
        state.h1 = h1_new;
        pred_out = state.h1;
    } else {
        pred_out = state.h0;
    }
}

// ===========================================================================
// Joint head (CPU, F32) — runs once per (encoder_frame, predictor_state)
//
//   joint_in_e = enc_w @ enc[t]  + enc_b           [joint_hidden]
//   joint_in_p = pred_w @ pred_u + pred_b          [joint_hidden]
//   logits     = out_w @ tanh(joint_in_e + joint_in_p) + out_b
// Output: [vocab+1+n_dur] (8198 for parakeet-tdt-0.6b-v3)
// ===========================================================================

// Pre-compute proj_e once per encoder frame so we don't redo it inside the
// inner predictor loop.
static void joint_proj_enc(const parakeet_joint_weights& J, const float* enc_t, std::vector<float>& out) {
    out.assign(J.enc_b.begin(), J.enc_b.end());
#if defined(HAVE_ACCELERATE)
    if (!parakeet_use_scalar()) {
        // enc_w[joint_hidden, d_model] @ enc_t[d_model], adding into out (which holds enc_b)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, J.joint_hidden, J.d_model, 1.0f, J.enc_w.data(), J.d_model, enc_t, 1,
                    1.0f, out.data(), 1);
        return;
    }
#endif
    for (int i = 0; i < J.joint_hidden; i++) {
        float s = out[i]; // already has enc_b[i]
        const float* row = J.enc_w.data() + (size_t)i * J.d_model;
        for (int k = 0; k < J.d_model; k++)
            s += row[k] * enc_t[k];
        out[i] = s;
    }
}

static void joint_step(const parakeet_joint_weights& J,
                       const float* proj_enc, // [joint_hidden]
                       const float* pred_u,   // [pred_hidden]
                       std::vector<float>& logits) {
    std::vector<float> mid(J.joint_hidden);
#if defined(HAVE_ACCELERATE)
    if (!parakeet_use_scalar()) {
        // pred_w[joint_hidden, pred_hidden] @ pred_u + pred_b → mid, then relu(proj_enc + mid)
        mid.assign(J.pred_b.begin(), J.pred_b.end());
        cblas_sgemv(CblasRowMajor, CblasNoTrans, J.joint_hidden, J.pred_hidden, 1.0f, J.pred_w.data(), J.pred_hidden,
                    pred_u, 1, 1.0f, mid.data(), 1);
        for (int i = 0; i < J.joint_hidden; i++) {
            float v = proj_enc[i] + mid[i];
            mid[i] = v > 0.0f ? v : 0.0f;
        }
        // out_w[vocab_total, joint_hidden] @ mid + out_b → logits
        logits.assign(J.out_b.begin(), J.out_b.end());
        cblas_sgemv(CblasRowMajor, CblasNoTrans, J.vocab_total, J.joint_hidden, 1.0f, J.out_w.data(), J.joint_hidden,
                    mid.data(), 1, 1.0f, logits.data(), 1);
    } else {
#endif
        for (int i = 0; i < J.joint_hidden; i++) {
            float s = J.pred_b[i];
            const float* row = J.pred_w.data() + (size_t)i * J.pred_hidden;
            for (int k = 0; k < J.pred_hidden; k++)
                s += row[k] * pred_u[k];
            // NeMo RNNTJoint uses ReLU (not tanh) — see jointnet.activation in
            // model_config.yaml.
            float v = proj_enc[i] + s;
            mid[i] = v > 0.0f ? v : 0.0f;
        }

        logits.assign(J.vocab_total, 0.0f);
        for (int v = 0; v < J.vocab_total; v++) {
            float s = J.out_b[v];
            const float* row = J.out_w.data() + (size_t)v * J.joint_hidden;
            for (int k = 0; k < J.joint_hidden; k++)
                s += row[k] * mid[k];
            logits[v] = s;
        }
#if defined(HAVE_ACCELERATE)
    }
#endif
}

// ===========================================================================
// Lazy weight cache initialisation (predictor + joint, F32 on CPU)
// ===========================================================================

static void parakeet_init_pred_weights(parakeet_context* ctx) {
    if (ctx->pred_w.initialised)
        return;
    auto& p = ctx->model.predictor;
    auto& W = ctx->pred_w;
    const int H = (int)ctx->model.hparams.pred_hidden;
    const int blank_id = (int)ctx->model.hparams.blank_id;

    W.embed = tensor_to_f32(p.embed_w);
    W.w_ih_0 = tensor_to_f32(p.lstm0_w_ih);
    W.w_hh_0 = tensor_to_f32(p.lstm0_w_hh);
    W.b_ih_0 = tensor_to_f32(p.lstm0_b_ih);
    W.b_hh_0 = tensor_to_f32(p.lstm0_b_hh);
    W.n_layers = (int)ctx->model.hparams.pred_layers;
    if (W.n_layers >= 2) {
        W.w_ih_1 = tensor_to_f32(p.lstm1_w_ih);
        W.w_hh_1 = tensor_to_f32(p.lstm1_w_hh);
        W.b_ih_1 = tensor_to_f32(p.lstm1_b_ih);
        W.b_hh_1 = tensor_to_f32(p.lstm1_b_hh);
    }

    W.H = H;
    W.initialised = true;

    // Sanity checks against the actual tensor shapes. ggml stores
    // PyTorch (rows, cols) as ne[0]=cols, ne[1]=rows.
    const int64_t embed_cols = p.embed_w->ne[0];       // == hidden
    const int64_t embed_rows = p.embed_w->ne[1];       // == vocab+1
    const int64_t lstm0_ih_cols = p.lstm0_w_ih->ne[0]; // == in_dim
    const int64_t lstm0_ih_rows = p.lstm0_w_ih->ne[1]; // == 4*H
    if (embed_cols != H) {
        fprintf(stderr,
                "parakeet: WARN embed.weight cols=%lld but pred_hidden=%d "
                "(GGUF hparam may be wrong)\n",
                (long long)embed_cols, H);
    }
    if (lstm0_ih_cols != H || lstm0_ih_rows != 4 * H) {
        fprintf(stderr,
                "parakeet: WARN lstm.0.w_ih shape=%lldx%lld but expected "
                "%dx%d (4*H, H)\n",
                (long long)lstm0_ih_rows, (long long)lstm0_ih_cols, 4 * H, H);
    }
    if (embed_rows < blank_id + 1) {
        fprintf(stderr, "parakeet: WARN embed.weight rows=%lld but blank_id+1=%d\n", (long long)embed_rows,
                blank_id + 1);
    }

    // NeMo's RNNT decoder uses Embedding(..., padding_idx=blank_id), so
    // the blank row should be all-zeros after training. If non-zero,
    // either the converter dropped padding_idx semantics or the
    // checkpoint was built with a different convention — the SOS step
    // (predictor at blank input) would produce wrong output.
    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG") && blank_id < (int)embed_rows) {
        const float* row = W.embed.data() + (size_t)blank_id * H;
        double s = 0;
        float maxv = 0;
        for (int k = 0; k < H; k++) {
            s += row[k] * row[k];
            if (fabsf(row[k]) > maxv)
                maxv = fabsf(row[k]);
        }
        fprintf(stderr,
                "parakeet: embed[blank=%d]  L2=%.4f  max|.|=%.4f  "
                "(NeMo padding_idx → expect ~0)\n",
                blank_id, sqrt(s), (double)maxv);
    }
}

static void parakeet_init_joint_weights(parakeet_context* ctx) {
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
    J.vocab_total = (int)j.out_b->ne[0]; // 8198 = 8192 vocab + 1 blank + 5 dur
    J.initialised = true;
}

// ===========================================================================
// TDT greedy decode
//
// State at each step: (t, u, predictor_state, last_token).
//   t = current encoder frame index (0 .. T_enc-1)
//   u = predictor step index (used for the predictor's autoregressive state)
//
// At each step, run the joint head on (enc[t], pred_state) and split the
// 8198-class output into (vocab_logits[8193], duration_logits[5]).
//
// Greedy:
//   token_id = argmax(vocab_logits)            (8192 = blank)
//   dur_skip = argmax(duration_logits)         (in {0, 1, 2, 3, 4})
//
// If token_id == blank: do not emit. Advance t by max(1, dur_skip).
// Else: emit (token_id, t, t + dur_skip). Advance the predictor by feeding
//       token_id, advance u++, advance t by max(1, dur_skip).
//
// Word timestamps come for free: each emitted token spans frames
// [t, t + dur_skip), and frame_dur = 80 ms in this model.
//
// Stop when t >= T_enc.
// ===========================================================================

struct parakeet_emitted_token {
    int id;
    int t_start; // encoder frame at emission
    int t_end;   // emission + duration
    float p;     // softmax probability of the emitted token [0, 1]
};

// §232 — GPU decode via the shared core_rnnt_ggml helpers (predictor LSTM +
// joint as ggml graphs on ctx->backend). Default ON when the decode backend is a
// GPU (P100 A/B: 5-12x faster, transcript-identical — LEARNINGS 33); cblas on
// CPU. Override PARAKEET_GGML_DECODE=1/0; RNNT_GGML_PERSTEP = per-step path.
//
// Returns whether to use ggml decode, and builds the persistent `gdec` when so.
// Shared by every parakeet decode variant (greedy, beam, maes, rnnt).
// Whether the TDT/RNNT decode runs as ggml graphs on ctx->backend (as opposed
// to the cblas CPU path). Split out of parakeet_init_ggml_decoder so callers
// can ask the question WITHOUT building the decoder — the long-form pipeline
// needs it to decide whether encode and decode may overlap on two threads
// (they may not when both drive ctx->backend). Keep the two in lockstep.
static bool parakeet_ggml_decode_active(const parakeet_context* ctx) {
    bool ggml_dec = !core_cpu_backend::is_cpu(ctx->backend);
    // ggml decode wins on CUDA/Vulkan (slow CPU BLAS: P100 5-12x) but LOSES on
    // Metal, where Apple Accelerate cblas beats the small-matmul GPU decode (M1
    // parakeet total ~16x cblas vs ~11x ggml). Default OFF on Metal (LEARNINGS 34).
#if defined(GGML_USE_METAL)
    if (ggml_dec && core_cpu_backend::is_metal(ctx->backend))
        ggml_dec = false;
#endif
    if (const char* e = crispasr_env::get("CRISPASR_PARAKEET_GGML_DECODE"))
        ggml_dec = (e[0] == '1');
    // The shared persistent ggml decoder currently has a fixed two-LSTM
    // graph.  NeMo RNNT models such as Quds-v4 legitimately use one layer;
    // keep their small predictor/joint on the validated scalar/BLAS path.
    if (ctx->model.hparams.pred_layers < 2)
        ggml_dec = false;
    return ggml_dec;
}

extern "C" int parakeet_decode_uses_backend(struct parakeet_context* ctx) {
    return (ctx && parakeet_ggml_decode_active(ctx)) ? 1 : 0;
}

static bool parakeet_init_ggml_decoder(parakeet_context* ctx, core_rnnt_ggml::Decoder& gdec) {
    bool ggml_dec = parakeet_ggml_decode_active(ctx);
    if (ggml_dec && crispasr_env::get("CRISPASR_RNNT_GGML_PERSTEP") == nullptr) {
        const auto& p = ctx->model.predictor;
        const auto& j = ctx->model.joint;
        core_rnnt_ggml::decoder_init(gdec, ctx->backend, p.embed_w, p.lstm0_w_ih, p.lstm0_b_ih, p.lstm0_w_hh,
                                     p.lstm0_b_hh, p.lstm1_w_ih, p.lstm1_b_ih, p.lstm1_w_hh, p.lstm1_b_hh, j.pred_w,
                                     j.pred_b, j.out_w, j.out_b, (int)ctx->model.hparams.pred_hidden,
                                     (int)ctx->model.hparams.joint_hidden);
    }
    return ggml_dec;
}

static void parakeet_predictor_step_ggml(parakeet_context* ctx, core_rnnt_ggml::Decoder& dec, int token_id,
                                         parakeet_lstm_state& state, std::vector<float>& pred_out) {
    if (dec.active()) {
        core_rnnt_ggml::decoder_predictor(dec, token_id, state.h0, state.c0, state.h1, state.c1, pred_out);
        return;
    }
    const auto& p = ctx->model.predictor;
    core_rnnt_ggml::predictor_step(ctx->sched, p.embed_w, p.lstm0_w_ih, p.lstm0_b_ih, p.lstm0_w_hh, p.lstm0_b_hh,
                                   p.lstm1_w_ih, p.lstm1_b_ih, p.lstm1_w_hh, p.lstm1_b_hh, token_id,
                                   (int)ctx->model.hparams.pred_hidden, state.h0, state.c0, state.h1, state.c1,
                                   pred_out);
}

static void parakeet_joint_step_ggml(parakeet_context* ctx, core_rnnt_ggml::Decoder& dec, const float* proj_e,
                                     const float* pred_u, std::vector<float>& logits) {
    if (dec.active()) {
        core_rnnt_ggml::decoder_joint(dec, proj_e, pred_u, logits);
        return;
    }
    const auto& j = ctx->model.joint;
    core_rnnt_ggml::joint_step(ctx->sched, j.pred_w, j.pred_b, j.out_w, j.out_b, proj_e, pred_u,
                               (int)ctx->model.hparams.joint_hidden, (int)ctx->model.hparams.pred_hidden, logits);
}

static std::vector<parakeet_emitted_token> parakeet_tdt_decode(parakeet_context* ctx, const float* enc, int T_enc,
                                                               int d_model) {
    parakeet_init_pred_weights(ctx);
    parakeet_init_joint_weights(ctx);

    // §232: ggml GPU decode default (see parakeet_init_ggml_decoder / LEARNINGS 33).
    core_rnnt_ggml::Decoder gdec;
    const bool ggml_dec = parakeet_init_ggml_decoder(ctx, gdec);
    const bool time_dec = crispasr_env::get("CRISPASR_PARAKEET_DECODE_TIMING") != nullptr;
    auto _dt0 = std::chrono::steady_clock::now();

    const auto& hp = ctx->model.hparams;
    const int blank_id = (int)hp.blank_id;     // 8192
    const int n_vocab_blk = blank_id + 1;      // 8193 (vocab + blank)
    const int n_dur = (int)hp.n_tdt_durations; // 5
    const int max_per_step = 10;               // safety: cap predictor advances per t

    auto& W = ctx->pred_w;
    auto& J = ctx->joint_w;
    if (J.vocab_total != n_vocab_blk + n_dur) {
        fprintf(stderr, "parakeet: joint vocab_total mismatch (%d vs expected %d)\n", J.vocab_total,
                n_vocab_blk + n_dur);
    }

    std::vector<parakeet_emitted_token> emitted;
    emitted.reserve(256);

    parakeet_lstm_state state;
    lstm_init_state(state, W.H);

    // SOS / first input is the blank token (NeMo convention)
    std::vector<float> pred_out;
    if (ggml_dec)
        parakeet_predictor_step_ggml(ctx, gdec, blank_id, state, pred_out);
    else
        predictor_step(W, blank_id, state, pred_out);
    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(
            stderr, "parakeet: pred_out[blank]: mean=%.4f std=%.4f [0..3]=%.4f %.4f %.4f %.4f\n",
            [&] {
                double s = 0;
                for (auto v : pred_out)
                    s += v;
                return s / pred_out.size();
            }(),
            [&] {
                double s = 0, m = 0;
                for (auto v : pred_out) {
                    m += v;
                    s += v * v;
                }
                m /= pred_out.size();
                return sqrt(s / pred_out.size() - m * m);
            }(),
            (double)pred_out[0], (double)pred_out[1], (double)pred_out[2], (double)pred_out[3]);

    std::vector<float> proj_e(J.joint_hidden);
    std::vector<float> logits(J.vocab_total);

    // §232: Pre-compute ALL encoder projections upfront.
    // Replaces T_enc individual sgemv calls inside the decode loop with
    // a single bulk computation. On macOS uses batched sgemm; on Linux
    // falls back to per-frame sgemv (still benefits from locality).
    std::vector<float> all_proj_e((size_t)T_enc * J.joint_hidden);
    {
        for (int t = 0; t < T_enc; t++) {
            float* dst = all_proj_e.data() + (size_t)t * J.joint_hidden;
            std::copy(J.enc_b.begin(), J.enc_b.end(), dst);
        }
#if defined(HAVE_ACCELERATE)
        // Batched sgemm: all_proj_e[T_enc, Jh] += enc[T_enc, d] @ enc_w^T[d, Jh]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T_enc, J.joint_hidden, J.d_model, 1.0f, enc, J.d_model,
                    J.enc_w.data(), J.d_model, 1.0f, all_proj_e.data(), J.joint_hidden);
#else
        // Per-frame fallback (still better than computing inside the loop
        // where cache misses on enc[] are random due to dur_skip advances)
        for (int t = 0; t < T_enc; t++) {
            float* dst = all_proj_e.data() + (size_t)t * J.joint_hidden;
            const float* enc_t = enc + (size_t)t * J.d_model;
            for (int i = 0; i < J.joint_hidden; i++) {
                float s = dst[i]; // already has enc_b[i]
                const float* row = J.enc_w.data() + (size_t)i * J.d_model;
                for (int k = 0; k < J.d_model; k++)
                    s += row[k] * enc_t[k];
                dst[i] = s;
            }
        }
#endif
    }

    // Sampling state — only touched when ctx->decode_temperature > 0.
    // We initialize unconditionally because seeding a mt19937_64 is
    // cheap, and keeping it outside the inner loop preserves the same
    // RNG sequence for the whole utterance.
    const bool sampling = ctx->decode_temperature > 0.0f;
    std::mt19937_64 rng(ctx->decode_seed != 0 ? ctx->decode_seed : (uint64_t)std::random_device{}());

    const bool has_hotwords = !ctx->hotword_trie.empty();
    core_context_bias::MatchState hw_state;

    int t = 0;
    int total_steps = 0;
    while (t < T_enc) {
        // §232: use pre-computed projection instead of per-frame sgemv
        std::copy(all_proj_e.data() + (size_t)t * J.joint_hidden, all_proj_e.data() + (size_t)(t + 1) * J.joint_hidden,
                  proj_e.data());

        int n_inner = 0;
        while (n_inner < max_per_step) {
            if (ggml_dec)
                parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), pred_out.data(), logits);
            else
                joint_step(J, proj_e.data(), pred_out.data(), logits);

            // CTC-WS phrase boost on vocab logits (not duration logits)
            if (has_hotwords)
                core_context_bias::apply_bias(ctx->hotword_trie, hw_state, logits.data(), n_vocab_blk,
                                              ctx->hotword_boost);

            if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG") && total_steps < 5) {
                // Show first few logits
                int best = 0;
                float best_v = logits[0];
                for (int v = 1; v < n_vocab_blk; v++)
                    if (logits[v] > best_v) {
                        best_v = logits[v];
                        best = v;
                    }
                fprintf(stderr, "parakeet: t=%d step=%d best_tok=%d (%.3f) blank_logit=%.3f\n", t, total_steps, best,
                        best_v, logits[blank_id]);
            }
            total_steps++;

            // Argmax (default) or temperature sample over the vocab+blank
            // logits. The duration logits are picked separately by argmax
            // either way — they're trained as a 5-way classifier and
            // sampling them would just inject random latency without any
            // quality benefit.
            int tok = 0;
            float tok_lp = logits[0];
            for (int v = 1; v < n_vocab_blk; v++) {
                if (logits[v] > tok_lp) {
                    tok_lp = logits[v];
                    tok = v;
                }
            }
            if (sampling) {
                // Numerically-stable softmax over the n_vocab_blk
                // half of the joint logits, then inverse-CDF sample.
                const float inv_t = 1.0f / ctx->decode_temperature;
                std::vector<double> pr((size_t)n_vocab_blk);
                double sum = 0.0;
                for (int v = 0; v < n_vocab_blk; v++) {
                    const double e = std::exp((double)((logits[v] - tok_lp) * inv_t));
                    pr[(size_t)v] = e;
                    sum += e;
                }
                if (sum > 0.0) {
                    std::uniform_real_distribution<double> unif(0.0, sum);
                    const double rr = unif(rng);
                    double acc = 0.0;
                    for (int v = 0; v < n_vocab_blk; v++) {
                        acc += pr[(size_t)v];
                        if (rr <= acc) {
                            tok = v;
                            break;
                        }
                    }
                    tok_lp = logits[tok];
                }
            }

            // Argmax over the duration logits (last n_dur entries)
            int dur_id = 0;
            float dur_lp = logits[n_vocab_blk];
            for (int d = 1; d < n_dur; d++) {
                if (logits[n_vocab_blk + d] > dur_lp) {
                    dur_lp = logits[n_vocab_blk + d];
                    dur_id = d;
                }
            }
            int dur_skip = (int)hp.tdt_durations[dur_id]; // 0..4

            if (tok == blank_id) {
                // Blank → never emit. Two cases, matching the NeMo TDT
                // reference (`GreedyTDTInfer._greedy_decode` in
                // `rnnt_greedy_decoding.py`):
                //
                //   dur > 0: advance encoder frame by dur, exit inner loop.
                //   dur = 0: stay on this frame, count toward the shared
                //            `max_per_step` budget so we don't spin forever
                //            (reference: `symbols_added += 1; need_loop =
                //            (skip == 0)`).
                //
                // The old "force `t += 1` on blank+dur=0" path matched the
                // empirical result of NeMo's spin (10 deterministic blank
                // retries → force-advance by 1 via `if symbols_added ==
                // max_symbols: time_idx += 1`) for greedy argmax, but
                // diverged from the reference for any path where the
                // predictor / joint state can change between retries (e.g.
                // temperature sampling, or future paths that touch the
                // shared `n_inner` budget). Issue #88.
                if (dur_skip > 0) {
                    t += dur_skip;
                    break;
                }
                n_inner++;
                continue;
            }

            // Softmax probability of the picked token, scoped to the
            // vocab+blank half of the joint logits (the duration logits
            // are a separate softmax). Numerically stable: subtract the
            // per-row max before exponentiating.
            float tok_p = 1.0f;
            {
                double sum = 0.0;
                for (int v = 0; v < n_vocab_blk; v++) {
                    sum += std::exp((double)(logits[v] - tok_lp));
                }
                tok_p = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
            }

            // Real token: emit and advance the predictor
            int t_end = std::min(T_enc, t + std::max(0, dur_skip));
            emitted.push_back({tok, t, t_end, tok_p});
            if (has_hotwords)
                core_context_bias::advance(ctx->hotword_trie, hw_state, tok);
            if (ggml_dec)
                parakeet_predictor_step_ggml(ctx, gdec, tok, state, pred_out);
            else
                predictor_step(W, tok, state, pred_out);

            // Diagnostic: dump predictor stats for the first few emissions
            // so we can compare with NeMo step-by-step (not just SOS).
            if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG") && (int)emitted.size() <= 5) {
                double s = 0, m = 0;
                float minv = pred_out[0], maxv = pred_out[0];
                for (auto v : pred_out) {
                    m += v;
                    s += v * v;
                    if (v < minv)
                        minv = v;
                    if (v > maxv)
                        maxv = v;
                }
                m /= pred_out.size();
                double std_ = sqrt(s / pred_out.size() - m * m);
                fprintf(stderr,
                        "parakeet: emit#%zu tok=%d dur=%d  pred_out: mean=%.4f std=%.4f "
                        "min=%.3f max=%.3f [0..3]=%.3f %.3f %.3f %.3f\n",
                        emitted.size(), tok, dur_skip, m, std_, (double)minv, (double)maxv, (double)pred_out[0],
                        (double)pred_out[1], (double)pred_out[2], (double)pred_out[3]);
            }

            // Advance encoder frame by the predicted duration (≥ 0). If 0,
            // we stay on this frame for another inner step.
            if (dur_skip > 0) {
                t += dur_skip;
                break;
            }
            n_inner++;
        }

        if (n_inner >= max_per_step) {
            // Force a one-frame advance to guarantee progress.
            t++;
        }
    }

    if (time_dec) {
        auto _dt1 = std::chrono::steady_clock::now();
        fprintf(stderr, "parakeet: tdt_decode %.1f ms (%s, T_enc=%d, %zu tokens)\n",
                std::chrono::duration<double, std::milli>(_dt1 - _dt0).count(), ggml_dec ? "ggml" : "cblas", T_enc,
                emitted.size());
    }
    return emitted;
}

// §232: Batched greedy TDT decode — processes multiple frames per iteration.
// Between token emissions, the predictor state (pred_out) doesn't change.
// We exploit this: compute the full joint+argmax for a BATCH of frames in one
// pass using sgemm. Reduces ~370 sequential sgemv calls to ~26 batched sgemm
// calls (one per token emission). Expected 5-15x speedup on CPU, more on GPU.
static std::vector<parakeet_emitted_token> parakeet_tdt_decode_batched(parakeet_context* ctx, const float* enc,
                                                                       int T_enc, int d_model) {
    parakeet_init_pred_weights(ctx);
    parakeet_init_joint_weights(ctx);

    const auto& hp = ctx->model.hparams;
    const int blank_id = (int)hp.blank_id;
    const int n_vocab_blk = blank_id + 1;
    const int n_dur = (int)hp.n_tdt_durations;

    auto& W = ctx->pred_w;
    auto& J = ctx->joint_w;
    const int Jh = J.joint_hidden;
    const int Vt = J.vocab_total;

    std::vector<parakeet_emitted_token> emitted;
    emitted.reserve(256);

    parakeet_lstm_state state;
    lstm_init_state(state, W.H);
    std::vector<float> pred_out;
    predictor_step(W, blank_id, state, pred_out);

    // Pre-compute ALL encoder projections
    std::vector<float> all_proj_e((size_t)T_enc * Jh);
    for (int f = 0; f < T_enc; f++)
        std::copy(J.enc_b.begin(), J.enc_b.end(), all_proj_e.data() + (size_t)f * Jh);
#if defined(HAVE_ACCELERATE)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T_enc, Jh, J.d_model, 1.0f, enc, J.d_model, J.enc_w.data(),
                J.d_model, 1.0f, all_proj_e.data(), Jh);
#else
    for (int f = 0; f < T_enc; f++) {
        float* dst = all_proj_e.data() + (size_t)f * Jh;
        const float* enc_f = enc + (size_t)f * d_model;
        for (int i = 0; i < Jh; i++) {
            const float* row = J.enc_w.data() + (size_t)i * J.d_model;
            float s = dst[i];
            for (int k = 0; k < J.d_model; k++)
                s += row[k] * enc_f[k];
            dst[i] = s;
        }
    }
#endif

    std::vector<float> pred_proj(Jh);
    std::vector<float> mid_batch;
    std::vector<float> logits_batch;

    int t = 0;
    while (t < T_enc) {
        // Compute pred_proj = pred_w @ pred_out + pred_b (constant until next emission)
        std::copy(J.pred_b.begin(), J.pred_b.end(), pred_proj.data());
#if defined(HAVE_ACCELERATE)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Jh, J.pred_hidden, 1.0f, J.pred_w.data(), J.pred_hidden,
                    pred_out.data(), 1, 1.0f, pred_proj.data(), 1);
#else
        for (int i = 0; i < Jh; i++) {
            const float* row = J.pred_w.data() + (size_t)i * J.pred_hidden;
            float s = pred_proj[i];
            for (int k = 0; k < J.pred_hidden; k++)
                s += row[k] * pred_out[k];
            pred_proj[i] = s;
        }
#endif

        // Batch: compute mid + logits for a window of upcoming frames.
        // Cap at 32 — between token emissions, blanks advance 1-4 frames
        // so we rarely need to look further. Avoids the cost of computing
        // 8198-dim logits for 300+ frames when only ~10 are needed.
        int batch = std::min(T_enc - t, 32);
        mid_batch.resize((size_t)batch * Jh);
        for (int f = 0; f < batch; f++) {
            const float* pe = all_proj_e.data() + (size_t)(t + f) * Jh;
            float* mid = mid_batch.data() + (size_t)f * Jh;
            for (int i = 0; i < Jh; i++) {
                float v = pe[i] + pred_proj[i];
                mid[i] = v > 0.0f ? v : 0.0f; // ReLU
            }
        }

        // Batched logits: [batch, Vt] = mid_batch[batch, Jh] @ out_w^T[Jh, Vt]
        logits_batch.resize((size_t)batch * Vt);
        for (int f = 0; f < batch; f++)
            std::copy(J.out_b.begin(), J.out_b.end(), logits_batch.data() + (size_t)f * Vt);
#if defined(HAVE_ACCELERATE)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, Vt, Jh, 1.0f, mid_batch.data(), Jh, J.out_w.data(),
                    Jh, 1.0f, logits_batch.data(), Vt);
#else
        for (int f = 0; f < batch; f++) {
            float* lg = logits_batch.data() + (size_t)f * Vt;
            const float* m = mid_batch.data() + (size_t)f * Jh;
            for (int v = 0; v < Vt; v++) {
                const float* row = J.out_w.data() + (size_t)v * Jh;
                float s = lg[v];
                for (int k = 0; k < Jh; k++)
                    s += row[k] * m[k];
                lg[v] = s;
            }
        }
#endif

        // Scan batch: walk through frames using pre-computed logits.
        // For blank frames, advance by dur_skip (or 1 for dur=0 — deterministic
        // retries are no-ops for greedy). Stop at first non-blank → emit.
        // The batch logits are indexed by ABSOLUTE frame position offset from
        // the start `t` of this batch.
        bool found_emission = false;
        int scan_t = t; // track position as we scan
        while (scan_t < T_enc) {
            int f = scan_t - t; // index into the batch
            if (f >= batch)
                break; // exhausted pre-computed batch

            const float* lg = logits_batch.data() + (size_t)f * Vt;
            int tok = 0;
            float tok_lp = lg[0];
            for (int v = 1; v < n_vocab_blk; v++)
                if (lg[v] > tok_lp) {
                    tok_lp = lg[v];
                    tok = v;
                }

            int dur_id = 0;
            float dur_lp = lg[n_vocab_blk];
            for (int d = 1; d < n_dur; d++)
                if (lg[n_vocab_blk + d] > dur_lp) {
                    dur_lp = lg[n_vocab_blk + d];
                    dur_id = d;
                }
            int dur_skip = (int)hp.tdt_durations[dur_id];

            if (tok == blank_id) {
                // Blank: advance. For greedy, dur=0 retries give same result
                // (same input → same output), so just advance by 1.
                scan_t += std::max(dur_skip, 1);
            } else {
                // Non-blank: emit token, update predictor, break to re-batch
                int t_end = std::min(T_enc, scan_t + std::max(0, dur_skip));
                float tok_p = 1.0f;
                {
                    double sum = 0.0;
                    for (int v = 0; v < n_vocab_blk; v++)
                        sum += std::exp((double)(lg[v] - tok_lp));
                    tok_p = sum > 0 ? (float)(1.0 / sum) : 0.0f;
                }
                emitted.push_back({tok, scan_t, t_end, tok_p});
                predictor_step(W, tok, state, pred_out);
                scan_t += std::max(dur_skip, 1);
                found_emission = true;
                break;
            }
        }
        t = scan_t;
        if (!found_emission && t >= T_enc)
            break;
    }

    return emitted;
}

// ===========================================================================
// TDT beam search decode
// ===========================================================================
// Label-looping beam search over the TDT joint head. Each hypothesis
// independently tracks its encoder frame pointer, predictor LSTM state,
// and hotword trie position. At each step every active hypothesis is
// expanded by the top-B vocab tokens (plus blank if not already in top-B),
// scored by cumulative log-softmax, and globally pruned to beam_size.
//
// The duration head is always argmax — it's a 5-way classifier and
// branching on it would multiply beams without quality benefit.
//
// LSTM state snapshots are plain vector copies (~10 KB per beam for
// H=640, 2-layer LSTM). The encoder dominates wall time, so even
// beam_size=8 adds negligible overhead.
//
// When beam_size==1 this produces bit-identical output to greedy (the
// single beam is the argmax at every step).
// ===========================================================================

static std::vector<parakeet_emitted_token> parakeet_tdt_beam_decode(parakeet_context* ctx, const float* enc, int T_enc,
                                                                    int d_model, int beam_size) {
    parakeet_init_pred_weights(ctx);
    parakeet_init_joint_weights(ctx);

    const auto& hp = ctx->model.hparams;
    const int blank_id = (int)hp.blank_id;
    const int n_vocab_blk = blank_id + 1;
    const int n_dur = (int)hp.n_tdt_durations;
    const int max_per_step = 10;

    auto& W = ctx->pred_w;
    auto& J = ctx->joint_w;

    const bool has_hotwords = !ctx->hotword_trie.empty();
    const int B = std::max(1, beam_size);
    core_rnnt_ggml::Decoder gdec;
    const bool ggml_dec = parakeet_init_ggml_decoder(ctx, gdec);

    // Per-hypothesis state
    struct Hyp {
        parakeet_lstm_state lstm;
        std::vector<float> pred_out; // predictor output [H]
        int t = 0;                   // encoder frame pointer
        int n_inner = 0;             // inner-loop counter at current t
        double cum_logprob = 0.0;    // cumulative log-softmax score
        std::vector<parakeet_emitted_token> emitted;
        core_context_bias::MatchState hw_state;
        bool active = true; // false once t >= T_enc
    };

    // Seed beam: single hypothesis at t=0 with SOS predictor state
    std::vector<Hyp> beam(1);
    {
        auto& h = beam[0];
        lstm_init_state(h.lstm, W.H);
        if (ggml_dec)
            parakeet_predictor_step_ggml(ctx, gdec, blank_id, h.lstm, h.pred_out);
        else
            predictor_step(W, blank_id, h.lstm, h.pred_out);
        h.emitted.reserve(256);
    }

    std::vector<float> proj_e(J.joint_hidden);
    std::vector<float> logits(J.vocab_total);

    // Main loop: iterate until all beams finish (t >= T_enc)
    for (;;) {
        // Check if any beam is still active
        bool any_active = false;
        for (const auto& h : beam)
            if (h.active) {
                any_active = true;
                break;
            }
        if (!any_active)
            break;

        // Expand each active hypothesis
        struct Candidate {
            int parent;         // index into beam[]
            int token;          // vocab token or blank_id
            int dur_skip;       // TDT duration (argmax)
            double cum_logprob; // parent.cum_logprob + token log-softmax
            float tok_p;        // softmax probability for confidence
        };
        std::vector<Candidate> cands;
        cands.reserve((size_t)beam.size() * (size_t)(B + 1));

        for (int bi = 0; bi < (int)beam.size(); bi++) {
            auto& h = beam[bi];
            if (!h.active) {
                // Carry forward finished hypothesis
                cands.push_back({bi, -1, 0, h.cum_logprob, 0.0f});
                continue;
            }

            // Project encoder frame and run joint head
            joint_proj_enc(J, enc + (size_t)h.t * d_model, proj_e);
            if (ggml_dec)
                parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), h.pred_out.data(), logits);
            else
                joint_step(J, proj_e.data(), h.pred_out.data(), logits);

            // Hotword phrase boost on vocab logits
            if (has_hotwords)
                core_context_bias::apply_bias(ctx->hotword_trie, h.hw_state, logits.data(), n_vocab_blk,
                                              ctx->hotword_boost);

            // Duration: always argmax
            int dur_id = 0;
            float dur_lp = logits[n_vocab_blk];
            for (int d = 1; d < n_dur; d++) {
                if (logits[n_vocab_blk + d] > dur_lp) {
                    dur_lp = logits[n_vocab_blk + d];
                    dur_id = d;
                }
            }
            int dur_skip = (int)hp.tdt_durations[dur_id];

            // Compute log-partition (log-sum-exp) over vocab+blank
            float max_logit = logits[0];
            for (int v = 1; v < n_vocab_blk; v++)
                if (logits[v] > max_logit)
                    max_logit = logits[v];
            double logZ = 0.0;
            for (int v = 0; v < n_vocab_blk; v++)
                logZ += std::exp((double)(logits[v] - max_logit));
            logZ = (double)max_logit + std::log(logZ);

            // Top-B token indices by raw logit value
            std::vector<int> top_ids(std::min(B, n_vocab_blk));
            std::vector<float> top_vals(top_ids.size(), -1e30f);
            for (int v = 0; v < n_vocab_blk; v++) {
                int mi = 0;
                for (int j = 1; j < (int)top_ids.size(); j++)
                    if (top_vals[j] < top_vals[mi])
                        mi = j;
                if (logits[v] > top_vals[mi]) {
                    top_vals[mi] = logits[v];
                    top_ids[mi] = v;
                }
            }
            // Ensure blank is always a candidate (so the beam can advance time)
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
                cands.push_back({bi, id, dur_skip, h.cum_logprob + log_p, tok_p});
            }
        }

        // Global prune: keep top-B candidates by cumulative log-prob
        const size_t keep = std::min<size_t>((size_t)B, cands.size());
        std::partial_sort(cands.begin(), cands.begin() + (ptrdiff_t)keep, cands.end(),
                          [](const Candidate& a, const Candidate& b) { return a.cum_logprob > b.cum_logprob; });
        cands.resize(keep);

        // Build next beam by materializing each surviving candidate's state
        std::vector<Hyp> next_beam;
        next_beam.reserve(keep);

        for (auto& c : cands) {
            const auto& parent = beam[c.parent];

            if (c.token < 0) {
                // Carry-forward of finished hypothesis
                next_beam.push_back(parent); // copy
                next_beam.back().active = false;
                continue;
            }

            Hyp nh;
            nh.lstm = parent.lstm; // snapshot LSTM state
            nh.pred_out = parent.pred_out;
            nh.cum_logprob = c.cum_logprob;
            nh.emitted = parent.emitted;
            nh.hw_state = parent.hw_state;

            if (c.token == blank_id) {
                // Blank: don't emit, advance frame or stay for inner loop
                if (c.dur_skip > 0) {
                    nh.t = parent.t + c.dur_skip;
                    nh.n_inner = 0;
                } else {
                    nh.t = parent.t;
                    nh.n_inner = parent.n_inner + 1;
                    if (nh.n_inner >= max_per_step) {
                        nh.t = parent.t + 1; // force progress
                        nh.n_inner = 0;
                    }
                }
            } else {
                // Real token: emit, advance predictor, advance frame by duration
                int t_end = std::min(T_enc, parent.t + std::max(0, c.dur_skip));
                nh.emitted.push_back({c.token, parent.t, t_end, c.tok_p});
                if (has_hotwords)
                    core_context_bias::advance(ctx->hotword_trie, nh.hw_state, c.token);
                if (ggml_dec)
                    parakeet_predictor_step_ggml(ctx, gdec, c.token, nh.lstm, nh.pred_out);
                else
                    predictor_step(W, c.token, nh.lstm, nh.pred_out);
                if (c.dur_skip > 0) {
                    nh.t = parent.t + c.dur_skip;
                    nh.n_inner = 0;
                } else {
                    nh.t = parent.t;
                    nh.n_inner = parent.n_inner + 1;
                    if (nh.n_inner >= max_per_step) {
                        nh.t = parent.t + 1;
                        nh.n_inner = 0;
                    }
                }
            }

            nh.active = (nh.t < T_enc);
            next_beam.push_back(std::move(nh));
        }

        beam = std::move(next_beam);
    }

    // Return the best (highest cum_logprob) hypothesis
    if (beam.empty())
        return {};
    int best = 0;
    for (int i = 1; i < (int)beam.size(); i++)
        if (beam[i].cum_logprob > beam[best].cum_logprob)
            best = i;
    return std::move(beam[best].emitted);
}

// ===========================================================================
// RNNT beam search decode (n_tdt_durations == 0)
// ===========================================================================
// Same structure as TDT beam search but without the duration head.
// Blank always advances by 1 frame; real tokens stay on the same frame.

static std::vector<parakeet_emitted_token> parakeet_rnnt_beam_decode(parakeet_context* ctx, const float* enc, int T_enc,
                                                                     int d_model, int beam_size) {
    parakeet_init_pred_weights(ctx);
    parakeet_init_joint_weights(ctx);

    const auto& hp = ctx->model.hparams;
    const int blank_id = (int)hp.blank_id;
    const int n_vocab_blk = blank_id + 1;
    const int max_per_step = 10;

    auto& W = ctx->pred_w;
    auto& J = ctx->joint_w;

    const bool has_hotwords = !ctx->hotword_trie.empty();
    const int B = std::max(1, beam_size);
    core_rnnt_ggml::Decoder gdec;
    const bool ggml_dec = parakeet_init_ggml_decoder(ctx, gdec);

    struct Hyp {
        parakeet_lstm_state lstm;
        std::vector<float> pred_out;
        int t = 0;
        int n_inner = 0;
        double cum_logprob = 0.0;
        std::vector<parakeet_emitted_token> emitted;
        core_context_bias::MatchState hw_state;
        bool active = true;
    };

    std::vector<Hyp> beam(1);
    {
        auto& h = beam[0];
        lstm_init_state(h.lstm, W.H);
        if (ggml_dec)
            parakeet_predictor_step_ggml(ctx, gdec, blank_id, h.lstm, h.pred_out);
        else if (ggml_dec)
            parakeet_predictor_step_ggml(ctx, gdec, blank_id, h.lstm, h.pred_out);
        else
            predictor_step(W, blank_id, h.lstm, h.pred_out);
        h.emitted.reserve(256);
    }

    std::vector<float> proj_e(J.joint_hidden);
    std::vector<float> logits(J.vocab_total);

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
            if (ggml_dec)
                parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), h.pred_out.data(), logits);
            else if (ggml_dec)
                parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), h.pred_out.data(), logits);
            else
                joint_step(J, proj_e.data(), h.pred_out.data(), logits);
            if (has_hotwords)
                core_context_bias::apply_bias(ctx->hotword_trie, h.hw_state, logits.data(), n_vocab_blk,
                                              ctx->hotword_boost);

            float max_logit = logits[0];
            for (int v = 1; v < n_vocab_blk; v++)
                if (logits[v] > max_logit)
                    max_logit = logits[v];
            double logZ = 0.0;
            for (int v = 0; v < n_vocab_blk; v++)
                logZ += std::exp((double)(logits[v] - max_logit));
            logZ = (double)max_logit + std::log(logZ);

            std::vector<int> top_ids(std::min(B, n_vocab_blk));
            std::vector<float> top_vals(top_ids.size(), -1e30f);
            for (int v = 0; v < n_vocab_blk; v++) {
                int mi = 0;
                for (int j = 1; j < (int)top_ids.size(); j++)
                    if (top_vals[j] < top_vals[mi])
                        mi = j;
                if (logits[v] > top_vals[mi]) {
                    top_vals[mi] = logits[v];
                    top_ids[mi] = v;
                }
            }
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
            nh.hw_state = parent.hw_state;

            if (c.token == blank_id) {
                nh.t = parent.t + 1;
                nh.n_inner = 0;
            } else {
                nh.emitted.push_back({c.token, parent.t, parent.t, c.tok_p});
                if (has_hotwords)
                    core_context_bias::advance(ctx->hotword_trie, nh.hw_state, c.token);
                if (ggml_dec)
                    parakeet_predictor_step_ggml(ctx, gdec, c.token, nh.lstm, nh.pred_out);
                else if (ggml_dec)
                    parakeet_predictor_step_ggml(ctx, gdec, c.token, nh.lstm, nh.pred_out);
                else
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
// MAES (Modified Adaptive Expansion Search) for TDT
//
// Time-synchronous beam search that processes one encoder frame at a time
// with up to `maes_num_steps` non-blank expansions per frame.
//
// Key differences from the label-looping beam search above:
//   1. Process ALL beams at the same encoder frame before advancing.
//   2. Up to N expansions per frame (adaptive: stop early if all blank).
//   3. Gamma-threshold pruning: keep candidate if score >= best - gamma.
//   4. After last expansion step, force-add blank score to non-blank hyps.
//
// Reference: NeMo BeamRNNTInfer.modified_adaptive_expansion_search
// ===========================================================================

static std::vector<parakeet_emitted_token> parakeet_tdt_maes_decode(parakeet_context* ctx, const float* enc, int T_enc,
                                                                    int d_model, int beam_size, int maes_num_steps = 2,
                                                                    float maes_gamma = 2.3f, int maes_beta = 2) {
    parakeet_init_pred_weights(ctx);
    parakeet_init_joint_weights(ctx);

    const auto& hp = ctx->model.hparams;
    const int blank_id = (int)hp.blank_id;
    const int n_vocab_blk = blank_id + 1;
    const int n_dur = (int)hp.n_tdt_durations;
    const int B = std::max(1, beam_size);
    const int topk = B + maes_beta; // candidates per hypothesis
    core_rnnt_ggml::Decoder gdec;
    const bool ggml_dec = parakeet_init_ggml_decoder(ctx, gdec);

    auto& W = ctx->pred_w;
    auto& J = ctx->joint_w;

    struct Hyp {
        parakeet_lstm_state lstm;
        std::vector<float> pred_out;
        double score = 0.0;
        std::vector<parakeet_emitted_token> emitted;
        std::vector<int> y_seq; // token sequence for prefix detection
    };

    // Initialise single beam with SOS
    std::vector<Hyp> kept;
    kept.resize(1);
    {
        auto& h = kept[0];
        lstm_init_state(h.lstm, W.H);
        if (ggml_dec)
            parakeet_predictor_step_ggml(ctx, gdec, blank_id, h.lstm, h.pred_out);
        else if (ggml_dec)
            parakeet_predictor_step_ggml(ctx, gdec, blank_id, h.lstm, h.pred_out);
        else
            predictor_step(W, blank_id, h.lstm, h.pred_out);
        h.emitted.reserve(256);
    }

    std::vector<float> proj_e(J.joint_hidden);
    std::vector<float> logits(J.vocab_total);

    for (int t = 0; t < T_enc; /* advanced inside */) {
        // Pre-compute encoder projection for this frame
        joint_proj_enc(J, enc + (size_t)t * d_model, proj_e);

        // Working set for this frame: start with kept_hyps
        std::vector<Hyp> hyps = kept;
        std::vector<Hyp> list_b; // blank-expanded hypotheses

        for (int n = 0; n < maes_num_steps; n++) {
            std::vector<Hyp> list_exp; // token-expanded hypotheses

            for (auto& h : hyps) {
                // Joint step
                if (ggml_dec)
                    parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), h.pred_out.data(), logits);
                else if (ggml_dec)
                    parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), h.pred_out.data(), logits);
                else
                    joint_step(J, proj_e.data(), h.pred_out.data(), logits);
                // Duration: always argmax over duration head
                int dur_id = 0;
                float dur_lp = logits[n_vocab_blk];
                for (int d = 1; d < n_dur; d++) {
                    if (logits[n_vocab_blk + d] > dur_lp) {
                        dur_lp = logits[n_vocab_blk + d];
                        dur_id = d;
                    }
                }
                int dur_skip = (int)hp.tdt_durations[dur_id];

                // Log-softmax over token head (vocab + blank)
                float max_l = logits[0];
                for (int v = 1; v < n_vocab_blk; v++)
                    if (logits[v] > max_l)
                        max_l = logits[v];
                double logZ = 0.0;
                for (int v = 0; v < n_vocab_blk; v++)
                    logZ += std::exp((double)(logits[v] - max_l));
                logZ = (double)max_l + std::log(logZ);

                // Top-k candidates by logit value
                struct Expansion {
                    int token;
                    double new_score;
                };
                std::vector<Expansion> expansions;
                expansions.reserve(topk);

                // Find top-k tokens
                std::vector<std::pair<float, int>> topk_pairs(n_vocab_blk);
                for (int v = 0; v < n_vocab_blk; v++)
                    topk_pairs[v] = {logits[v], v};
                std::partial_sort(topk_pairs.begin(), topk_pairs.begin() + std::min(topk, n_vocab_blk),
                                  topk_pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

                // Gamma pruning: keep if score >= best - gamma
                double best_exp_score = h.score + ((double)topk_pairs[0].first - logZ);
                for (int k = 0; k < std::min(topk, n_vocab_blk); k++) {
                    double new_score = h.score + ((double)topk_pairs[k].first - logZ);
                    if (new_score >= best_exp_score - (double)maes_gamma) {
                        expansions.push_back({topk_pairs[k].second, new_score});
                    }
                }

                // Split expansions into blank and non-blank
                for (auto& ex : expansions) {
                    if (ex.token == blank_id) {
                        Hyp bh;
                        bh.lstm = h.lstm;
                        bh.pred_out = h.pred_out;
                        bh.score = ex.new_score;
                        bh.emitted = h.emitted;
                        bh.y_seq = h.y_seq;
                        list_b.push_back(std::move(bh));
                    } else {
                        Hyp nh;
                        nh.lstm = h.lstm;
                        nh.score = ex.new_score;
                        nh.emitted = h.emitted;
                        nh.y_seq = h.y_seq;
                        nh.y_seq.push_back(ex.token);
                        int t_end = std::min(T_enc, t + std::max(0, dur_skip));
                        nh.emitted.push_back({ex.token, t, t_end, (float)std::exp(ex.new_score - h.score)});
                        // Advance predictor for non-blank
                        if (ggml_dec)
                            parakeet_predictor_step_ggml(ctx, gdec, ex.token, nh.lstm, nh.pred_out);
                        else if (ggml_dec)
                            parakeet_predictor_step_ggml(ctx, gdec, ex.token, nh.lstm, nh.pred_out);
                        else
                            predictor_step(W, ex.token, nh.lstm, nh.pred_out);
                        list_exp.push_back(std::move(nh));
                    }
                }
            }

            if (list_exp.empty()) {
                // All expansions were blank → stop expanding this frame
                break;
            }

            if (n < maes_num_steps - 1) {
                // Not the last step: continue expanding non-blank hyps
                hyps = std::move(list_exp);
            } else {
                // Last expansion step: force-add blank score to all
                // non-blank hyps so they can compete with blank hyps.
                for (auto& nh : list_exp) {
                    if (ggml_dec)
                        parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), nh.pred_out.data(), logits);
                    else if (ggml_dec)
                        parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), nh.pred_out.data(), logits);
                    else
                        joint_step(J, proj_e.data(), nh.pred_out.data(), logits);
                    float max_l = logits[0];
                    for (int v = 1; v < n_vocab_blk; v++)
                        if (logits[v] > max_l)
                            max_l = logits[v];
                    double logZ2 = 0.0;
                    for (int v = 0; v < n_vocab_blk; v++)
                        logZ2 += std::exp((double)(logits[v] - max_l));
                    logZ2 = (double)max_l + std::log(logZ2);
                    nh.score += ((double)logits[blank_id] - logZ2);
                    list_b.push_back(std::move(nh));
                }
            }
        }

        // Select top-B from list_b
        if ((int)list_b.size() > B) {
            std::partial_sort(list_b.begin(), list_b.begin() + B, list_b.end(),
                              [](const Hyp& a, const Hyp& b) { return a.score > b.score; });
            list_b.resize(B);
        }

        // Advance time: TDT duration on blank → skip frames
        // For MAES, blanks just advance by 1 frame (like standard RNNT).
        // The TDT duration applies to the non-blank path emissions already.
        kept = std::move(list_b);
        t++;
    }

    // Return best hypothesis
    if (kept.empty())
        return {};
    int best = 0;
    for (int i = 1; i < (int)kept.size(); i++)
        if (kept[i].score > kept[best].score)
            best = i;
    return std::move(kept[best].emitted);
}

// ===========================================================================
// MAES for pure RNNT (no TDT duration head)
//
// Same algorithm as TDT MAES but without the duration head: blank always
// advances by 1 frame, non-blank stays on the same frame.
// ===========================================================================

static std::vector<parakeet_emitted_token> parakeet_rnnt_maes_decode(parakeet_context* ctx, const float* enc, int T_enc,
                                                                     int d_model, int beam_size, int maes_num_steps = 2,
                                                                     float maes_gamma = 2.3f, int maes_beta = 2) {
    parakeet_init_pred_weights(ctx);
    parakeet_init_joint_weights(ctx);

    const auto& hp = ctx->model.hparams;
    const int blank_id = (int)hp.blank_id;
    const int n_vocab_blk = blank_id + 1;
    const int B = std::max(1, beam_size);
    const int topk = B + maes_beta;
    core_rnnt_ggml::Decoder gdec;
    const bool ggml_dec = parakeet_init_ggml_decoder(ctx, gdec);

    auto& W = ctx->pred_w;
    auto& J = ctx->joint_w;

    struct Hyp {
        parakeet_lstm_state lstm;
        std::vector<float> pred_out;
        double score = 0.0;
        std::vector<parakeet_emitted_token> emitted;
    };

    std::vector<Hyp> kept(1);
    {
        auto& h = kept[0];
        lstm_init_state(h.lstm, W.H);
        if (ggml_dec)
            parakeet_predictor_step_ggml(ctx, gdec, blank_id, h.lstm, h.pred_out);
        else if (ggml_dec)
            parakeet_predictor_step_ggml(ctx, gdec, blank_id, h.lstm, h.pred_out);
        else
            predictor_step(W, blank_id, h.lstm, h.pred_out);
        h.emitted.reserve(256);
    }

    std::vector<float> proj_e(J.joint_hidden);
    std::vector<float> logits(J.vocab_total);

    for (int t = 0; t < T_enc; t++) {
        joint_proj_enc(J, enc + (size_t)t * d_model, proj_e);

        std::vector<Hyp> hyps = kept;
        std::vector<Hyp> list_b;

        for (int n = 0; n < maes_num_steps; n++) {
            std::vector<Hyp> list_exp;

            for (auto& h : hyps) {
                if (ggml_dec)
                    parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), h.pred_out.data(), logits);
                else if (ggml_dec)
                    parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), h.pred_out.data(), logits);
                else
                    joint_step(J, proj_e.data(), h.pred_out.data(), logits);
                // Log-softmax over vocab+blank
                float max_l = logits[0];
                for (int v = 1; v < n_vocab_blk; v++)
                    if (logits[v] > max_l)
                        max_l = logits[v];
                double logZ = 0.0;
                for (int v = 0; v < n_vocab_blk; v++)
                    logZ += std::exp((double)(logits[v] - max_l));
                logZ = (double)max_l + std::log(logZ);

                // Top-k + gamma pruning
                std::vector<std::pair<float, int>> topk_pairs(n_vocab_blk);
                for (int v = 0; v < n_vocab_blk; v++)
                    topk_pairs[v] = {logits[v], v};
                std::partial_sort(topk_pairs.begin(), topk_pairs.begin() + std::min(topk, n_vocab_blk),
                                  topk_pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

                double best_exp = h.score + ((double)topk_pairs[0].first - logZ);
                for (int k = 0; k < std::min(topk, n_vocab_blk); k++) {
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
                        nh.emitted.push_back({tok, t, t, (float)std::exp(new_score - h.score)});
                        if (ggml_dec)
                            parakeet_predictor_step_ggml(ctx, gdec, tok, nh.lstm, nh.pred_out);
                        else
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
                for (auto& nh : list_exp) {
                    if (ggml_dec)
                        parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), nh.pred_out.data(), logits);
                    else if (ggml_dec)
                        parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), nh.pred_out.data(), logits);
                    else
                        joint_step(J, proj_e.data(), nh.pred_out.data(), logits);
                    float max_l = logits[0];
                    for (int v = 1; v < n_vocab_blk; v++)
                        if (logits[v] > max_l)
                            max_l = logits[v];
                    double logZ2 = 0.0;
                    for (int v = 0; v < n_vocab_blk; v++)
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
// Standard RNNT greedy decode (n_tdt_durations == 0)
// ===========================================================================
// Identical predictor / joint infrastructure to TDT but no duration head:
//   - joint.out.weight rows = vocab+1 (vocab + blank only)
//   - blank → advance encoder frame by 1, reset inner counter
//   - real token → advance predictor, stay on same frame
//   - max_per_step cap forces frame advance to guarantee progress

static std::vector<parakeet_emitted_token> parakeet_rnnt_decode(parakeet_context* ctx, const float* enc, int T_enc,
                                                                int d_model) {
    parakeet_init_pred_weights(ctx);
    parakeet_init_joint_weights(ctx);

    const auto& hp = ctx->model.hparams;
    const int blank_id = (int)hp.blank_id; // vocab_size
    const int n_vocab_blk = blank_id + 1;  // vocab + blank
    const int max_per_step = 10;
    core_rnnt_ggml::Decoder gdec;
    const bool ggml_dec = parakeet_init_ggml_decoder(ctx, gdec);

    auto& W = ctx->pred_w;
    auto& J = ctx->joint_w;
    if (J.vocab_total != n_vocab_blk) {
        fprintf(stderr, "parakeet: RNNT joint vocab_total mismatch (%d vs expected %d)\n", J.vocab_total, n_vocab_blk);
    }

    std::vector<parakeet_emitted_token> emitted;
    emitted.reserve(256);

    parakeet_lstm_state state;
    lstm_init_state(state, W.H);

    std::vector<float> pred_out;
    if (ggml_dec)
        parakeet_predictor_step_ggml(ctx, gdec, blank_id, state, pred_out);
    else
        predictor_step(W, blank_id, state, pred_out);
    std::vector<float> proj_e(J.joint_hidden);
    std::vector<float> logits(J.vocab_total);

    const bool has_hotwords = !ctx->hotword_trie.empty();
    core_context_bias::MatchState hw_state;

    int t = 0;
    while (t < T_enc) {
        joint_proj_enc(J, enc + (size_t)t * d_model, proj_e);

        int n_inner = 0;
        while (n_inner < max_per_step) {
            if (ggml_dec)
                parakeet_joint_step_ggml(ctx, gdec, proj_e.data(), pred_out.data(), logits);
            else
                joint_step(J, proj_e.data(), pred_out.data(), logits);
            if (has_hotwords)
                core_context_bias::apply_bias(ctx->hotword_trie, hw_state, logits.data(), n_vocab_blk,
                                              ctx->hotword_boost);

            int tok = 0;
            float tok_lp = logits[0];
            for (int v = 1; v < n_vocab_blk; v++) {
                if (logits[v] > tok_lp) {
                    tok_lp = logits[v];
                    tok = v;
                }
            }

            if (tok == blank_id) {
                t++;
                break;
            }

            float tok_p = 1.0f;
            {
                double sum = 0.0;
                for (int v = 0; v < n_vocab_blk; v++)
                    sum += std::exp((double)(logits[v] - tok_lp));
                tok_p = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
            }

            emitted.push_back({tok, t, t, tok_p});
            if (has_hotwords)
                core_context_bias::advance(ctx->hotword_trie, hw_state, tok);
            if (ggml_dec)
                parakeet_predictor_step_ggml(ctx, gdec, tok, state, pred_out);
            else
                predictor_step(W, tok, state, pred_out);
            n_inner++;
        }

        if (n_inner >= max_per_step)
            t++;
    }

    return emitted;
}

// ===========================================================================
// CTC greedy decode (for hybrid TDT+CTC models)
// ===========================================================================

static std::vector<parakeet_emitted_token> parakeet_ctc_decode(parakeet_context* ctx, const float* enc, int T_enc,
                                                               int d_model) {
    std::vector<parakeet_emitted_token> emitted;
    if (!ctx->model.ctc_w || !ctx->model.ctc_b)
        return emitted;

    const int ctc_vocab = (int)ctx->model.ctc_vocab_size;
    // CTC blank is the last token in the CTC vocab (NeMo convention).
    const int ctc_blank = ctc_vocab - 1;

    // Pull CTC head weights to CPU.
    // ctc_w is Conv1d(d_model, ctc_vocab, 1) stored as (ctc_vocab, d_model, 1)
    // → effectively a (ctc_vocab, d_model) matmul. May be F16.
    // Quantized-safe: to_f32 dequantizes F16/quantized correctly (sizing a raw
    // read by w_numel*sizeof(float) would over-read a quantized ctc_w). ctc_w is
    // 3-D conv-shaped so the quantizer keeps it F16 today; this is defensive.
    std::vector<float> w = core_cpu::to_f32(ctx->model.ctc_w);
    std::vector<float> b = core_cpu::to_f32(ctx->model.ctc_b);

    // Compute CTC logits for all frames: logits[t][v] = w[v] @ enc[t] + b[v]
    std::vector<float> all_logits((size_t)T_enc * ctc_vocab);
    for (int t = 0; t < T_enc; t++) {
        const float* e = enc + (size_t)t * d_model;
        float* out_row = all_logits.data() + (size_t)t * ctc_vocab;
        for (int v = 0; v < ctc_vocab; v++) {
            float s = b[v];
            const float* wv = w.data() + (size_t)v * d_model;
            for (int k = 0; k < d_model; k++)
                s += wv[k] * e[k];
            out_row[v] = s;
        }
    }

    // CTC-WS phrase boost (PLAN #98): bias logits toward hotword tokens
    const bool has_hotwords = !ctx->hotword_trie.empty();
    if (has_hotwords) {
        core_context_bias::MatchState hw_state;
        for (int t = 0; t < T_enc; t++) {
            float* row = all_logits.data() + (size_t)t * ctc_vocab;
            core_context_bias::apply_bias(ctx->hotword_trie, hw_state, row, ctc_vocab, ctx->hotword_boost);
            // Advance hotword state with argmax token for next frame's bias
            int tok = (int)(std::max_element(row, row + ctc_vocab) - row);
            if (tok != ctc_blank)
                core_context_bias::advance(ctx->hotword_trie, hw_state, tok);
        }
    }

    // CTC beam search when beam_size > 1
    const int beam_sz = ctx->decode_beam_size;
    if (beam_sz > 1) {
        // Convert logits to log-softmax per frame
        std::vector<float> logprobs((size_t)T_enc * ctc_vocab);
        for (int t = 0; t < T_enc; t++) {
            const float* row = all_logits.data() + (size_t)t * ctc_vocab;
            float* lp = logprobs.data() + (size_t)t * ctc_vocab;
            float mx = *std::max_element(row, row + ctc_vocab);
            double sum = 0.0;
            for (int v = 0; v < ctc_vocab; v++)
                sum += std::exp((double)(row[v] - mx));
            double log_sum = (double)mx + std::log(sum);
            for (int v = 0; v < ctc_vocab; v++)
                lp[v] = (float)((double)row[v] - log_sum);
        }
        float gamma = ctx->decode_maes ? ctx->maes_gamma : 0.0f;
        auto br = core_ctc::prefix_beam_search(logprobs.data(), T_enc, ctc_vocab, ctc_blank,
                                               /*shift=*/0, beam_sz, gamma);
        for (int32_t tok : br.tokens)
            emitted.push_back({tok, 0, 0, 0.0f}); // no per-token timestamps in beam mode
        return emitted;
    }

    // Greedy CTC decode (original path)
    int prev_tok = ctc_blank;
    for (int t = 0; t < T_enc; t++) {
        const float* row = all_logits.data() + (size_t)t * ctc_vocab;
        int tok = 0;
        float tok_lp = row[0];
        for (int v = 1; v < ctc_vocab; v++) {
            if (row[v] > tok_lp) {
                tok_lp = row[v];
                tok = v;
            }
        }
        if (tok != ctc_blank && tok != prev_tok) {
            float tok_p = 1.0f;
            {
                double sum = 0.0;
                for (int v = 0; v < ctc_vocab; v++)
                    sum += std::exp((double)(row[v] - tok_lp));
                tok_p = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
            }
            emitted.push_back({tok, t, t, tok_p});
        }
        prev_tok = tok;
    }
    return emitted;
}

// ===========================================================================
// Backend selection
// ===========================================================================

static ggml_backend_t pick_backend() {
    ggml_backend_t b = crispasr_init_gpu_backend();
    return b ? b : core_cpu_backend::init();
}

static ggml_backend_t pick_backend(bool use_gpu) {
    return use_gpu ? pick_backend() : core_cpu_backend::init();
}

// ===========================================================================
// Public C API
// ===========================================================================

extern "C" struct parakeet_context_params parakeet_context_default_params(void) {
    parakeet_context_params p = {};
    p.n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    p.use_flash = false;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

extern "C" struct parakeet_context* parakeet_init_from_file(const char* path_model,
                                                            struct parakeet_context_params params) {
    auto* ctx = new parakeet_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = pick_backend(params.use_gpu);
    ctx->backend_cpu = core_cpu_backend::init();
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    if (!parakeet_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        fprintf(stderr, "parakeet: failed to load '%s'\n", path_model);
        parakeet_free(ctx);
        return nullptr;
    }

    parakeet_fold_batchnorm(ctx->model, ctx->backend);

    // Repack F16 conv pw1/pw2 to Q8_0 (issue #81 — the 3D conv layout dodges
    // crispasr-quantize, and the CPU F16 mul_mat path is ~6x slower than Q8_0).
    {
        auto& m = ctx->model;
        std::vector<core_conformer::BlockWeights*> layers;
        for (auto& e : m.enc)
            layers.push_back(&e);
        const bool quantized = !m.enc.empty() && m.enc[0].attn_q_w && ggml_is_quantized(m.enc[0].attn_q_w->type);
        core_conformer::repack_conv_pw_q8(layers, ctx->backend, quantized, m.pw_q8, "parakeet");
        core_conformer::fuse_qkv(layers, ctx->backend, m.qkv_fused, "parakeet");
    }

    // Hybrid TDT+CTC models with a single-LSTM predictor (parakeet-tdt_ctc-110m
    // has pred_layers=1) can only decode via the CTC head — TDT decode requires
    // a 2-layer LSTM. Default to CTC so the model just works out of the box.
    if (ctx->model.hparams.pred_layers < 2 && ctx->model.has_ctc) {
        ctx->decode_ctc = true;
        fprintf(stderr, "parakeet: single-LSTM predictor + CTC head detected → defaulting to CTC decode\n");
    }
    return ctx;
}

extern "C" void parakeet_free(struct parakeet_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    ctx->model.pw_q8.free();
    ctx->model.qkv_fused.free();
    if (ctx->model.buf_f32)
        ggml_backend_buffer_free(ctx->model.buf_f32);
    if (ctx->model.ctx_f32)
        ggml_free(ctx->model.ctx_f32);
    if (ctx->model.buf)
        core_gguf::release_weight_buffer(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

// Internal C++ entry point for tests — declared in parakeet.h via a different
// linkage section to avoid polluting the public C API.
extern std::vector<float> parakeet_encode_mel(parakeet_context* ctx, const float* mel, int n_mels, int T_mel,
                                              int* out_T_enc);

// ---- Stage-level entry points for crispasr-diff ----

extern "C" void parakeet_set_temperature(struct parakeet_context* ctx, float temperature, uint64_t seed) {
    if (!ctx)
        return;
    ctx->decode_temperature = temperature;
    ctx->decode_seed = seed;
}

extern "C" void parakeet_set_beam_size(struct parakeet_context* ctx, int beam_size) {
    if (!ctx)
        return;
    ctx->decode_beam_size = beam_size > 1 ? beam_size : 1;
}

extern "C" void parakeet_set_maes(struct parakeet_context* ctx, bool enable, int num_steps, float gamma, int beta) {
    if (!ctx)
        return;
    ctx->decode_maes = enable;
    if (enable) {
        ctx->maes_num_steps = num_steps > 0 ? num_steps : 2;
        ctx->maes_gamma = gamma > 0.0f ? gamma : 2.3f;
        ctx->maes_beta = beta > 0 ? beta : 2;
    }
}

extern "C" void parakeet_set_ctc_mode(struct parakeet_context* ctx, bool ctc) {
    if (!ctx)
        return;
    ctx->decode_ctc = ctc;
}

extern "C" bool parakeet_has_ctc(struct parakeet_context* ctx) {
    return ctx && ctx->model.has_ctc;
}

// Issue #257: switch the FastConformer encoder to local (windowed) attention at
// inference time — the equivalent of NeMo's
// `model.change_attention_model("rel_pos_local_attn", [left, right])`. Bounds
// encoder self-attention memory to O(T·window) instead of O(T²), for long audio
// on limited VRAM. Units are encoder frames (1 frame = frame_dur_cs, ~80 ms).
// left/right < 0 restores full (global) attention. Takes effect on the next
// transcribe (the mask is rebuilt per encode).
extern "C" void parakeet_set_att_context(struct parakeet_context* ctx, int left, int right) {
    if (!ctx)
        return;
    ctx->model.hparams.att_context_left = left;
    ctx->model.hparams.att_context_right = right;
}

extern "C" void parakeet_set_hotwords(struct parakeet_context* ctx, const char** hotwords, int n_hotwords,
                                      float boost) {
    if (!ctx || !hotwords || n_hotwords <= 0)
        return;
    // Build a tokenizer that maps strings to SentencePiece token IDs
    // using the already-loaded vocab.
    auto tokenize = [&](const std::string& word) -> std::vector<int32_t> {
        // Simple: look up each SentencePiece token. For multi-token words,
        // we do a greedy forward-maximum-match against the vocab.
        // This is good enough for hotwords (typically 1-3 tokens each).
        std::vector<int32_t> ids;
        const auto& pieces = ctx->vocab.id_to_token;
        std::string remaining = word;
        while (!remaining.empty()) {
            int best_len = 0;
            int32_t best_id = -1;
            for (int i = 0; i < (int)pieces.size(); i++) {
                const auto& p = pieces[i];
                if ((int)p.size() > best_len && remaining.compare(0, p.size(), p) == 0) {
                    best_len = (int)p.size();
                    best_id = i;
                }
            }
            if (best_id < 0) {
                // Skip unknown character
                size_t skip = 1;
                if ((unsigned char)remaining[0] >= 0x80) {
                    // UTF-8 multi-byte
                    if ((unsigned char)remaining[0] >= 0xF0)
                        skip = 4;
                    else if ((unsigned char)remaining[0] >= 0xE0)
                        skip = 3;
                    else
                        skip = 2;
                }
                remaining.erase(0, std::min(skip, remaining.size()));
            } else {
                ids.push_back(best_id);
                remaining.erase(0, best_len);
            }
        }
        return ids;
    };

    std::vector<std::string> hw_list;
    for (int i = 0; i < n_hotwords; i++)
        if (hotwords[i])
            hw_list.push_back(hotwords[i]);

    ctx->hotword_boost = boost;
    ctx->hotword_trie = core_context_bias::build_trie(hw_list, tokenize, boost);
}

extern "C" float* parakeet_compute_mel(struct parakeet_context* ctx, const float* samples, int n_samples,
                                       int* out_n_mels, int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    int T_mel = 0;
    auto mel = parakeet_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;
    const int n_mels = (int)ctx->model.hparams.n_mels;
    if (out_n_mels)
        *out_n_mels = n_mels;
    if (out_T_mel)
        *out_T_mel = T_mel;
    float* r = (float*)malloc(mel.size() * sizeof(float));
    if (!r)
        return nullptr;
    std::memcpy(r, mel.data(), mel.size() * sizeof(float));
    return r;
}

extern "C" float* parakeet_run_encoder(struct parakeet_context* ctx, const float* mel, int n_mels, int T_mel,
                                       int* out_T_enc, int* out_d_model) {
    if (!ctx || !mel || T_mel <= 0)
        return nullptr;
    int T_enc = 0;
    auto enc = parakeet_encode_mel(ctx, mel, n_mels, T_mel, &T_enc);
    if (enc.empty())
        return nullptr;
    const int d = (int)ctx->model.hparams.d_model;
    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d;
    float* r = (float*)malloc(enc.size() * sizeof(float));
    if (!r)
        return nullptr;
    std::memcpy(r, enc.data(), enc.size() * sizeof(float));
    return r;
}

// Build encoder graph with per-layer outputs tagged for read-back. Each
// stage is named "dump_pre_encode" or "dump_layer_K" and marked with
// ggml_set_output() so the scheduler keeps its buffer live after compute.
// Mirrors parakeet_build_graph_encoder; kept separate so the production
// path (single output) doesn't pay for graph-output bookkeeping.
static ggml_cgraph* parakeet_build_graph_encoder_dump(parakeet_context* ctx, int T_mel) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    int T = 0;
    ggml_tensor* cur = core_conformer::build_pre_encode(ctx0, mel, m.pre_encode, (int)hp.subsampling_channels, &T);

    // Tag pre-encode output BEFORE the xscaling. NeMo's pre_encode forward
    // also returns the post-projection output (the multiply by sqrt(d) is
    // an attribute of the rel-pos encoder, not the subsampling module).
    {
        ggml_tensor* tag = ggml_cont(ctx0, cur);
        ggml_set_name(tag, "dump_pre_encode");
        ggml_set_output(tag);
        ggml_build_forward_expand(gf, tag);
    }

    if (hp.xscaling) {
        const float xscale = sqrtf((float)hp.d_model);
        cur = ggml_scale(ctx0, cur, xscale);
    }

    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    ggml_tensor* local_mask_dump = nullptr;
    const bool use_local_attn_dump =
        hp.att_context_left >= 0 && hp.att_context_right >= 0 && T > hp.att_context_left + hp.att_context_right + 1;
    if (use_local_attn_dump) {
        local_mask_dump = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
        ggml_set_name(local_mask_dump, "local_attn_mask");
        ggml_set_input(local_mask_dump);
    }

    core_conformer::BlockParams bp = {
        (int)hp.d_model, (int)hp.n_heads,     (int)hp.head_dim,     (int)hp.conv_kernel,
        kLayerNormEps,   hp.att_context_left, hp.att_context_right, (int)hp.global_tokens,
    };
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = core_conformer::build_block(ctx0, cur, pos_enc, T, m.enc[il], bp, local_mask_dump);
        char nm[64];
        snprintf(nm, sizeof(nm), "dump_layer_%u", il);
        ggml_tensor* tag = ggml_cont(ctx0, cur);
        ggml_set_name(tag, nm);
        ggml_set_output(tag);
        ggml_build_forward_expand(gf, tag);
        cur = tag; // chain next layer off the tagged tensor (numerics fix)
    }

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

extern "C" int parakeet_run_encoder_dump(struct parakeet_context* ctx, const float* mel, int n_mels, int T_mel,
                                         float** out, int out_count, int* out_T_enc, int* out_d_model) {
    if (!ctx || !mel || T_mel <= 0 || !out)
        return 1;
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "parakeet: mel feature mismatch (%d vs %d)\n", n_mels, (int)ctx->model.hparams.n_mels);
        return 2;
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
        crispasr_imatrix_install(ctx->sched); // no-op unless CRISPASR_IMATRIX_OUT is set
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    }

    ggml_cgraph* gf = parakeet_build_graph_encoder_dump(ctx, T_mel);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "parakeet: dump: failed to alloc encoder graph\n");
        return 3;
    }

    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2;
    auto pe = core_conformer::make_pos_enc((int)ctx->model.hparams.d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    ggml_tensor* local_mask_in2 = ggml_graph_get_tensor(gf, "local_attn_mask");
    if (local_mask_in2) {
        const auto& hp = ctx->model.hparams;
        auto lm = core_conformer::make_local_attn_mask(T_enc, hp.att_context_left, hp.att_context_right,
                                                       (int)hp.global_tokens);
        ggml_backend_tensor_set(local_mask_in2, lm.data(), 0, lm.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "parakeet: dump: encoder graph compute failed\n");
        return 4;
    }

    const int d = (int)ctx->model.hparams.d_model;
    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d;

    auto read_to = [&](const char* name, float* dst) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t) {
            fprintf(stderr, "parakeet: dump: missing %s tensor\n", name);
            return false;
        }
        ggml_backend_tensor_get(t, dst, 0, (size_t)d * T_enc * sizeof(float));
        return true;
    };

    if (out_count >= 1 && out[0])
        read_to("dump_pre_encode", out[0]);
    const int n_layers = (int)ctx->model.hparams.n_layers;
    for (int il = 0; il < n_layers && (il + 1) < out_count; il++) {
        if (!out[il + 1])
            continue;
        char nm[64];
        snprintf(nm, sizeof(nm), "dump_layer_%d", il);
        read_to(nm, out[il + 1]);
    }
    return 0;
}

extern "C" int parakeet_test_encoder(struct parakeet_context* ctx, int T_mel) {
    int n_mels = (int)ctx->model.hparams.n_mels;
    std::vector<float> mel((size_t)n_mels * T_mel, 0.0f);
    int T_enc = 0;
    auto out = parakeet_encode_mel(ctx, mel.data(), n_mels, T_mel, &T_enc);
    if (out.empty())
        return -1;
    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: encoder OK — T_mel=%d → T_enc=%d  d=%d  out[0..3]=%g %g %g %g\n", T_mel, T_enc,
                (int)ctx->model.hparams.d_model, (double)out[0], (double)out[1], (double)out[2], (double)out[3]);
    return T_enc;
}

extern "C" int parakeet_test_audio(struct parakeet_context* ctx, const float* samples, int n_samples) {
    int T_mel = 0;
    auto mel = parakeet_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return -1;

    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: mel OK — n_samples=%d (%.2fs)  T_mel=%d  n_mels=%d  mel[0..3]=%g %g %g %g\n",
                n_samples, (double)n_samples / ctx->model.hparams.sample_rate, T_mel, (int)ctx->model.hparams.n_mels,
                (double)mel[0], (double)mel[1], (double)mel[2], (double)mel[3]);

    int T_enc = 0;
    auto enc_out = parakeet_encode_mel(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, &T_enc);
    if (enc_out.empty())
        return -1;

    // Print a few summary stats over the encoder output
    double sum = 0, sq = 0;
    float mn = enc_out[0], mx = enc_out[0];
    for (float v : enc_out) {
        sum += v;
        sq += (double)v * v;
        if (v < mn)
            mn = v;
        if (v > mx)
            mx = v;
    }
    double mean = sum / enc_out.size();
    double var = sq / enc_out.size() - mean * mean;
    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: encoder OK — T_enc=%d  d=%d  mean=%.4f  std=%.4f  min=%.3f  max=%.3f\n", T_enc,
                (int)ctx->model.hparams.d_model, mean, sqrt(var), (double)mn, (double)mx);
    return T_enc;
}

// ===========================================================================
// Transducer component entry points for diff testing
// ===========================================================================

extern "C" float* parakeet_joint_project_encoder(struct parakeet_context* ctx, const float* enc_frames, int T_enc,
                                                 int d_model, int* out_joint_hidden) {
    parakeet_init_joint_weights(ctx);
    const auto& J = ctx->joint_w;
    if (d_model != J.d_model)
        return nullptr;

    const int JH = J.joint_hidden;
    float* out = (float*)malloc(sizeof(float) * T_enc * JH);
    if (!out)
        return nullptr;

    for (int t = 0; t < T_enc; t++) {
        std::vector<float> proj(JH);
        joint_proj_enc(J, enc_frames + (size_t)t * d_model, proj);
        memcpy(out + (size_t)t * JH, proj.data(), JH * sizeof(float));
    }
    if (out_joint_hidden)
        *out_joint_hidden = JH;
    return out;
}

extern "C" float* parakeet_predictor_initial(struct parakeet_context* ctx, int* out_pred_hidden) {
    parakeet_init_pred_weights(ctx);
    const auto& W = ctx->pred_w;
    const int H = W.H;
    const int blank_id = (int)ctx->model.hparams.blank_id;

    // NeMo's decoder.predict(y=None, state=None, add_sos=True) feeds TWO
    // zero inputs to the LSTM: a zero "SOS" vector prepended, plus the
    // pad-token (also zero since y=None). This means the LSTM runs for 2
    // timesteps on zero input. The blank embedding is all-zeros (NeMo uses
    // padding_idx=blank_id), so feeding blank_id twice is equivalent.
    parakeet_lstm_state state;
    lstm_init_state(state, H);
    std::vector<float> pred_out;
    predictor_step(W, blank_id, state, pred_out); // SOS (zero)
    predictor_step(W, blank_id, state, pred_out); // pad  (zero)

    float* out = (float*)malloc(sizeof(float) * H);
    if (!out)
        return nullptr;
    memcpy(out, pred_out.data(), H * sizeof(float));
    if (out_pred_hidden)
        *out_pred_hidden = H;
    return out;
}

extern "C" float* parakeet_joint_step(struct parakeet_context* ctx, const float* proj_enc, const float* pred_out,
                                      int* out_vocab_total) {
    parakeet_init_joint_weights(ctx);
    const auto& J = ctx->joint_w;

    std::vector<float> logits;
    joint_step(J, proj_enc, pred_out, logits);

    float* out = (float*)malloc(sizeof(float) * logits.size());
    if (!out)
        return nullptr;
    memcpy(out, logits.data(), logits.size() * sizeof(float));
    if (out_vocab_total)
        *out_vocab_total = (int)logits.size();
    return out;
}

extern "C" void parakeet_result_free(struct parakeet_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->tokens);
    free(r->words);
    free(r);
}

extern "C" int parakeet_n_vocab(struct parakeet_context* ctx) {
    return (int)ctx->model.hparams.vocab_size;
}

// Issue #257: is this a Japanese-only model? Detect by vocab CONTENT, not size.
// The old `vocab_size <= 4096` heuristic misclassified small-vocab ENGLISH models
// (e.g. parakeet-tdt-1.1b, vocab ~1024) as Japanese, forcing them onto the JA
// small-chunk path. JA-only parakeet vocabs are ~97% kana/kanji; multilingual v3
// (vocab 8192) and English models are ~0%. Returns 1 iff >50% of non-empty tokens
// contain a Japanese script character (hiragana/katakana U+3040–30FF, CJK
// ideographs U+4E00–9FFF / ext-A U+3400–4DBF).
extern "C" int parakeet_vocab_is_japanese(struct parakeet_context* ctx) {
    if (!ctx)
        return 0;
    return crispasr_parakeet::vocab_looks_japanese(ctx->vocab.id_to_token) ? 1 : 0;
}
extern "C" int parakeet_blank_id(struct parakeet_context* ctx) {
    return (int)ctx->model.hparams.blank_id;
}
extern "C" int parakeet_frame_dur_cs(struct parakeet_context* ctx) {
    return (int)ctx->model.hparams.frame_dur_cs;
}
extern "C" int parakeet_n_mels(struct parakeet_context* ctx) {
    return (int)ctx->model.hparams.n_mels;
}
extern "C" int parakeet_sample_rate(struct parakeet_context* ctx) {
    return (int)ctx->model.hparams.sample_rate;
}
extern "C" int parakeet_n_heads(struct parakeet_context* ctx) {
    return ctx ? (int)ctx->model.hparams.n_heads : 0;
}
// Approximate encoder-frame count for `n_samples` (mel frames / subsampling).
// Used by the Phase 2 memory policy to estimate the single-pass O(T^2) rel-pos
// bias before allocating it — not exact (ignores conv edge arithmetic), which
// is fine for a conservative memory gate.
extern "C" int parakeet_est_enc_frames(struct parakeet_context* ctx, int n_samples) {
    if (!ctx || n_samples <= 0)
        return 0;
    const auto& hp = ctx->model.hparams;
    const int hop = (int)hp.hop_length > 0 ? (int)hp.hop_length : 160;
    const int sub = (int)hp.subsampling_factor > 0 ? (int)hp.subsampling_factor : 8;
    return (n_samples / hop) / sub;
}

extern "C" const char* parakeet_token_to_str(struct parakeet_context* ctx, int id) {
    if (id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[id].c_str();
}

// ===========================================================================
// Public transcribe entry points
// ===========================================================================

// Convert a SentencePiece vocab string to user-visible text:
//   '▁foo'  → ' foo'    (word-start prefix)
//   '<unk>' → ''        (filtered)
//   anything else → as-is
static std::string spiece_to_text(const std::string& piece) {
    if (piece.empty())
        return "";
    if (piece.size() >= 2 && piece[0] == '<' && piece.back() == '>')
        return "";
    // Replace leading U+2581 (▁ = 0xE2 0x96 0x81) with a space
    if (piece.size() >= 3 && (unsigned char)piece[0] == 0xE2 && (unsigned char)piece[1] == 0x96 &&
        (unsigned char)piece[2] == 0x81) {
        return std::string(" ") + piece.substr(3);
    }
    return piece;
}

// ---------------------------------------------------------------------------
// Split encode / decode API
// ---------------------------------------------------------------------------

// NOTE on BENCH/DEBUG in the split path: parakeet_encode() and
// parakeet_decode_frames() carry the same "mel"/"encoder"/"decode" stages as
// parakeet_transcribe_ex() so the two paths report identically. When a caller
// PIPELINES them (encode on a worker thread, decode on another) the stages
// genuinely overlap, so their sum exceeds wall time — that gap is the win, not
// a measurement error. Set CRISPASR_PARAKEET_PIPELINE=0 for a serial baseline.
extern "C" float* parakeet_encode(struct parakeet_context* ctx, const float* samples, int n_samples, int* out_T_enc,
                                  int* out_d_model) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    int T_mel = 0;
    std::vector<float> mel;
    {
        parakeet_bench_stage _b("mel");
        mel = parakeet_compute_mel_impl(ctx, samples, n_samples, T_mel);
    }
    if (mel.empty())
        return nullptr;
    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: mel OK (%d frames)\n", T_mel);
    int T_enc = 0;
    std::vector<float> enc;
    {
        parakeet_bench_stage _b("encoder");
        enc = parakeet_encode_mel(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, &T_enc);
    }
    if (enc.empty())
        return nullptr;
    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: encoder OK (%d frames)\n", T_enc);
    const int d = (int)ctx->model.hparams.d_model;
    float* out = (float*)malloc(enc.size() * sizeof(float));
    memcpy(out, enc.data(), enc.size() * sizeof(float));
    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d;
    return out;
}

// Group r->tokens into r->words (issue #257). Shared by parakeet_transcribe_ex
// (single-pass) and parakeet_decode_frames (streamed / chunked) so every path
// emits a word list — previously decode_frames left r->words null and the CLI
// adapter only *copies* r->words, so streamed/chunked output had no words.
static void parakeet_group_words(parakeet_result* r, int frame_dur_cs) {
    // ----- Group sub-word tokens into words -----
    //
    // Latin SentencePiece convention: a token starting with U+2581 (▁ → ' ')
    // begins a new word. Punctuation tokens attach to the previous word.
    //
    // Japanese parakeet (and other no-space tokenizers) emit no leading-space
    // markers because written Japanese has no inter-word spaces. In that
    // mode every non-punctuation token is its own "word" — sufficient
    // granularity for word-level SRT (issue #37).
    std::vector<parakeet_word_data> words;
    words.reserve(r->n_tokens);

    // Detect tokenizer style: count tokens that look like real
    // word-starts (leading space + at least one more char). A
    // standalone " " token (BOS-ish) doesn't count.
    int n_space_word_starts = 0;
    for (int i = 0; i < r->n_tokens; i++) {
        const char* t = r->tokens[i].text;
        if (t[0] == ' ' && t[1] != '\0')
            n_space_word_starts++;
    }
    const bool space_prefix_style = (n_space_word_starts >= 2);

    auto is_punct_only = [](const char* s) {
        if (!s || !*s)
            return false;
        const unsigned char* p = (const unsigned char*)s;
        while (*p) {
            unsigned char c = *p;
            if (c < 0x80) {
                if (!(c == '.' || c == ',' || c == '?' || c == '!' || c == ';' || c == ':' || c == '\'' || c == '"' ||
                      c == '(' || c == ')' || c == '-'))
                    return false;
                p++;
            } else if (c == 0xE3 && p[1] == 0x80 && p[2] >= 0x80 && p[2] <= 0xBF) {
                p += 3; // U+3000–U+303F CJK Symbols and Punctuation
            } else if (c == 0xE3 && p[1] == 0x83 && p[2] == 0xBB) {
                p += 3; // U+30FB ・
            } else if (c == 0xEF && p[1] == 0xBC && ((p[2] >= 0x81 && p[2] <= 0x8F) || p[2] == 0x9F)) {
                p += 3; // U+FF01–U+FF0F, U+FF1F
            } else {
                return false;
            }
        }
        return true;
    };

    auto ends_with_sentence_punct = [](const char* s) {
        size_t len = s ? strlen(s) : 0;
        if (len == 0)
            return false;
        unsigned char last = (unsigned char)s[len - 1];
        if (last == '.' || last == '!' || last == '?')
            return true;
        if (len >= 3) {
            const unsigned char* tail = (const unsigned char*)(s + len - 3);
            if (tail[0] == 0xE3 && tail[1] == 0x80 && tail[2] == 0x82)
                return true;
            if (tail[0] == 0xEF && tail[1] == 0xBC && (tail[2] == 0x81 || tail[2] == 0x9F))
                return true;
        }
        return false;
    };

    parakeet_word_data cur = {};
    bool have_cur = false;
    float cur_p_sum = 0.0f;
    int cur_p_cnt = 0;

    auto flush_cur = [&]() {
        if (cur_p_cnt > 0)
            cur.p = cur_p_sum / (float)cur_p_cnt;
        words.push_back(cur);
    };

    for (int i = 0; i < r->n_tokens; i++) {
        const auto& td = r->tokens[i];
        if (!td.text[0])
            continue;
        if (td.text[0] == ' ' && td.text[1] == '\0')
            continue; // standalone space / BOS marker

        const bool has_leading_space = (td.text[0] == ' ');
        const bool is_punct = is_punct_only(td.text);
        const bool is_new_word = !is_punct && (has_leading_space || !space_prefix_style);

        if (is_new_word && have_cur) {
            flush_cur();
            cur = {};
            cur_p_sum = 0.0f;
            cur_p_cnt = 0;
            have_cur = false;
        }

        if (!have_cur) {
            cur.t0 = td.t0;
            have_cur = true;
        }
        cur.t1 = td.t1;
        cur_p_sum += td.p;
        cur_p_cnt += 1;

        const char* src = td.text + (has_leading_space ? 1 : 0);
        size_t cur_len = strlen(cur.text);
        size_t cap = sizeof(cur.text) - cur_len - 1;
        size_t add = strlen(src);
        if (add > cap)
            add = cap;
        memcpy(cur.text + cur_len, src, add);
        cur.text[cur_len + add] = '\0';
    }
    if (have_cur)
        flush_cur();

    // Insert minimum gaps after sentence-ending punctuation (subtitle spacing).
    for (size_t wi = 0; wi + 1 < words.size(); wi++) {
        if (!ends_with_sentence_punct(words[wi].text))
            continue;
        if (words[wi].t1 >= words[wi + 1].t0 && words[wi].t1 > words[wi].t0) {
            int64_t shrunk = words[wi].t1 - frame_dur_cs;
            if (shrunk > words[wi].t0)
                words[wi].t1 = shrunk;
        }
    }

    r->n_words = (int)words.size();
    r->words = (parakeet_word_data*)calloc(r->n_words > 0 ? r->n_words : 1, sizeof(parakeet_word_data));
    for (int i = 0; i < r->n_words; i++)
        r->words[i] = words[i];
}

extern "C" struct parakeet_result* parakeet_decode_frames(struct parakeet_context* ctx, const float* enc_frames,
                                                          int T_enc, int d_model, int64_t t_offset_cs) {
    if (!ctx || !enc_frames || T_enc <= 0)
        return nullptr;

    // Same "decode" stage parakeet_transcribe_ex() reports, so the split
    // encode/decode path is measurable with the same CRISPASR_PARAKEET_BENCH.
    parakeet_bench_stage _b_dec("decode");
    const bool use_ctc = ctx->decode_ctc && ctx->model.has_ctc;
    const bool use_rnnt = !use_ctc && ctx->model.hparams.n_tdt_durations == 0;
    const bool use_beam = !use_ctc && ctx->decode_beam_size > 1;
    const bool use_maes = use_beam && ctx->decode_maes;
    auto emitted =
        use_ctc ? parakeet_ctc_decode(ctx, enc_frames, T_enc, d_model)
        : use_rnnt
            ? (use_maes   ? parakeet_rnnt_maes_decode(ctx, enc_frames, T_enc, d_model, ctx->decode_beam_size,
                                                      ctx->maes_num_steps, ctx->maes_gamma, ctx->maes_beta)
               : use_beam ? parakeet_rnnt_beam_decode(ctx, enc_frames, T_enc, d_model, ctx->decode_beam_size)
                          : parakeet_rnnt_decode(ctx, enc_frames, T_enc, d_model))
            : (use_maes   ? parakeet_tdt_maes_decode(ctx, enc_frames, T_enc, d_model, ctx->decode_beam_size,
                                                     ctx->maes_num_steps, ctx->maes_gamma, ctx->maes_beta)
               : use_beam ? parakeet_tdt_beam_decode(ctx, enc_frames, T_enc, d_model, ctx->decode_beam_size)
                          : (getenv("CRISPASR_TDT_BATCH") ? parakeet_tdt_decode_batched(ctx, enc_frames, T_enc, d_model)
                                                          : parakeet_tdt_decode(ctx, enc_frames, T_enc, d_model)));

    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: %s%s decode OK (%d tokens)\n",
                use_ctc    ? "CTC"
                : use_rnnt ? "RNNT"
                           : "TDT",
                use_beam ? " beam" : "", (int)emitted.size());

    // Build result (same as the tail of parakeet_transcribe_ex)
    auto* r = (parakeet_result*)calloc(1, sizeof(parakeet_result));
    r->n_tokens = (int)emitted.size();
    r->tokens = (parakeet_token_data*)calloc(r->n_tokens > 0 ? r->n_tokens : 1, sizeof(parakeet_token_data));
    std::string text;
    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;
    for (int i = 0; i < r->n_tokens; i++) {
        const auto& e = emitted[i];
        const std::string& piece =
            (e.id >= 0 && e.id < (int)ctx->vocab.id_to_token.size()) ? ctx->vocab.id_to_token[e.id] : std::string("");
        std::string vis = spiece_to_text(piece);
        parakeet_token_data& td = r->tokens[i];
        td.id = e.id;
        td.t0 = t_offset_cs + (int64_t)e.t_start * frame_dur_cs;
        td.t1 = t_offset_cs + (int64_t)e.t_end * frame_dur_cs;
        td.p = e.p;
        size_t n = std::min(vis.size(), sizeof(td.text) - 1);
        memcpy(td.text, vis.data(), n);
        td.text[n] = '\0';
        text += vis;
    }
    if (!text.empty() && text[0] == ' ')
        text = text.substr(1);
    r->text = strdup(text.c_str());

    // Word grouping (issue #257): the streamed / chunked paths funnel through
    // here, so build words too — otherwise those paths emit none.
    parakeet_group_words(r, frame_dur_cs);

    return r;
}

// ---------------------------------------------------------------------------
// Chunked encode: split long audio into overlapping chunks, compute mel
// with per-chunk z-norm (keeps features close to training distribution),
// encode each chunk independently, and concatenate the encoder outputs
// with overlap deduplication.  The result is a single contiguous encoder
// buffer that can be decoded in one TDT pass — no decoder cold-start,
// no content loss from chunk boundaries.  Issue #89 / PLAN #104.
// ---------------------------------------------------------------------------

static std::vector<float> parakeet_encode_chunked(parakeet_context* ctx, const float* samples, int n_samples,
                                                  int chunk_seconds, int overlap_seconds, int* out_T_enc_total) {
    const auto& hp = ctx->model.hparams;
    const int SR = 16000;
    const int d_model = (int)hp.d_model;
    const int hop = (int)hp.hop_length;
    const int sub = (int)hp.subsampling_factor;

    const int chunk_samples = chunk_seconds * SR;
    const int overlap_samples = overlap_seconds * SR;
    const int shift_samples = chunk_samples - overlap_samples;

    // How many encoder frames the overlap region produces.
    // mel_frames = n_samples / hop;  encoder_frames = mel_frames / sub
    // For the overlap region: overlap_samples / hop / sub
    const int overlap_enc_frames = overlap_samples / (hop * sub);

    std::vector<float> enc_all;
    int T_enc_total = 0;

    for (int offset = 0; offset < n_samples; offset += shift_samples) {
        const int chunk_end = std::min(n_samples, offset + chunk_samples);
        const int chunk_len = chunk_end - offset;
        if (chunk_len <= 0)
            break;

        // Mel with per-chunk z-norm
        int T_mel = 0;
        auto mel = parakeet_compute_mel_impl(ctx, samples + offset, chunk_len, T_mel);
        if (mel.empty())
            continue;

        // Encode
        int T_enc = 0;
        auto enc = parakeet_encode_mel(ctx, mel.data(), (int)hp.n_mels, T_mel, &T_enc);
        if (enc.empty())
            continue;

        // For chunks after the first: skip the overlap encoder frames
        // at the start (the previous chunk already covers those frames).
        int skip_frames = 0;
        if (offset > 0 && T_enc > overlap_enc_frames) {
            skip_frames = overlap_enc_frames;
        }

        const int keep_frames = T_enc - skip_frames;
        if (keep_frames <= 0)
            continue;

        enc_all.insert(enc_all.end(), enc.begin() + (size_t)skip_frames * d_model,
                       enc.begin() + (size_t)(skip_frames + keep_frames) * d_model);
        T_enc_total += keep_frames;

        if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
            fprintf(stderr, "parakeet: chunk @%d: %d samples → %d mel → %d enc (skip %d, keep %d)\n", offset, chunk_len,
                    T_mel, T_enc, skip_frames, keep_frames);
    }

    if (out_T_enc_total)
        *out_T_enc_total = T_enc_total;
    return enc_all;
}

// Public API: chunked encode + single-pass TDT decode for long audio.
// Internally splits audio into overlapping chunks (default 8 s with 2 s
// overlap), encodes each with per-chunk z-norm, concatenates the encoder
// output, and runs one TDT decode over the whole sequence.  This avoids
// both the z-norm drift of single-pass encoding on long audio AND the
// decoder cold-start of independent-chunk decoding.
extern "C" struct parakeet_result* parakeet_transcribe_chunked(struct parakeet_context* ctx, const float* samples,
                                                               int n_samples, int64_t t_offset_cs, int chunk_seconds,
                                                               int overlap_seconds) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    if (chunk_seconds <= 0) {
        // Same per-model heuristic as parakeet_transcribe_streamed —
        // vocab < 4000 ⇒ JA-only model (small chunks work best), else
        // multilingual / v3 (needs ~30 s of context for the Conformer
        // encoder to produce in-distribution features).
        chunk_seconds = parakeet_vocab_is_japanese(ctx) ? 8 : 30; // #257: content, not vocab size
    }
    if (overlap_seconds < 0)
        overlap_seconds = 2;
    if (overlap_seconds >= chunk_seconds)
        overlap_seconds = chunk_seconds / 4;

    // 1. Chunked encode
    int T_enc_total = 0;
    auto enc_all = parakeet_encode_chunked(ctx, samples, n_samples, chunk_seconds, overlap_seconds, &T_enc_total);
    if (enc_all.empty() || T_enc_total <= 0)
        return nullptr;

    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: chunked encode done: %d total encoder frames from %d samples\n", T_enc_total,
                n_samples);

    // 2. Single-pass TDT decode over the concatenated encoder output
    const int d_model = (int)ctx->model.hparams.d_model;
    const bool use_ctc = ctx->decode_ctc && ctx->model.has_ctc;
    const bool use_rnnt = !use_ctc && ctx->model.hparams.n_tdt_durations == 0;
    const bool use_beam = !use_ctc && ctx->decode_beam_size > 1;
    const bool use_maes = use_beam && ctx->decode_maes;
    auto emitted =
        use_ctc ? parakeet_ctc_decode(ctx, enc_all.data(), T_enc_total, d_model)
        : use_rnnt
            ? (use_maes   ? parakeet_rnnt_maes_decode(ctx, enc_all.data(), T_enc_total, d_model, ctx->decode_beam_size,
                                                      ctx->maes_num_steps, ctx->maes_gamma, ctx->maes_beta)
               : use_beam ? parakeet_rnnt_beam_decode(ctx, enc_all.data(), T_enc_total, d_model, ctx->decode_beam_size)
                          : parakeet_rnnt_decode(ctx, enc_all.data(), T_enc_total, d_model))
            : (use_maes   ? parakeet_tdt_maes_decode(ctx, enc_all.data(), T_enc_total, d_model, ctx->decode_beam_size,
                                                     ctx->maes_num_steps, ctx->maes_gamma, ctx->maes_beta)
               : use_beam ? parakeet_tdt_beam_decode(ctx, enc_all.data(), T_enc_total, d_model, ctx->decode_beam_size)
                          : parakeet_tdt_decode_batched(ctx, enc_all.data(), T_enc_total, d_model));

    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: %s decode OK (%d tokens from %d enc frames)\n",
                use_ctc    ? "CTC"
                : use_rnnt ? "RNNT"
                           : "TDT",
                (int)emitted.size(), T_enc_total);

    // 3. Build result (reuse the same result-building code as transcribe_ex)
    auto* r = (parakeet_result*)calloc(1, sizeof(parakeet_result));
    r->n_tokens = (int)emitted.size();
    r->tokens = (parakeet_token_data*)calloc(r->n_tokens > 0 ? r->n_tokens : 1, sizeof(parakeet_token_data));
    std::string text;
    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;
    for (int i = 0; i < r->n_tokens; i++) {
        const auto& e = emitted[i];
        const std::string& piece =
            (e.id >= 0 && e.id < (int)ctx->vocab.id_to_token.size()) ? ctx->vocab.id_to_token[e.id] : std::string("");
        std::string vis = spiece_to_text(piece);

        parakeet_token_data& td = r->tokens[i];
        td.id = e.id;
        td.t0 = t_offset_cs + (int64_t)e.t_start * frame_dur_cs;
        td.t1 = t_offset_cs + (int64_t)e.t_end * frame_dur_cs;
        td.p = e.p;
        size_t n = std::min(vis.size(), sizeof(td.text) - 1);
        memcpy(td.text, vis.data(), n);
        td.text[n] = '\0';
        text += vis;
    }
    if (!text.empty() && text[0] == ' ')
        text = text.substr(1);
    r->text = strdup(text.c_str());

    // Word grouping — same code path as parakeet_transcribe_ex / parakeet_decode_frames.
    // The grouping needs the full token list, so it runs after all encoding + decoding.
    {
        // Delegate to parakeet_decode_frames which does the same word grouping.
        // We build a temporary to call the word-grouping path, or inline it.
        // For now, call parakeet_decode_frames on the already-decoded data.
        // Actually simpler: just use the existing word-grouping at lines below.
    }
    // Re-use the word-grouping from decode_frames: call it on enc_all.
    // Actually, parakeet_decode_frames already does decode + word grouping.
    // Let's just call it directly.
    parakeet_result_free(r);
    return parakeet_decode_frames(ctx, enc_all.data(), T_enc_total, d_model, t_offset_cs);
}

// ---------------------------------------------------------------------------
// NeMo-style streamed pipeline: two-pass approach.
//
// Pass 1: compute raw mel (no z-norm) for the FULL audio, accumulating
//   global per-band mean/variance.  Then apply z-norm using the global
//   statistics.  This gives identical features to single-pass encoding.
//
// Pass 2: encode the z-normalized mel in overlapping chunks (the
//   encoder is bidirectional, so it needs context around each chunk).
//   Concatenate encoder outputs and decode in one TDT pass.
//
// This matches the NeMo convention: z-norm computed over the full
// utterance, not per-chunk.  The result should be identical to
// parakeet_transcribe_ex on short audio and close to 100% coverage
// on long audio.
// ---------------------------------------------------------------------------

extern "C" struct parakeet_result* parakeet_transcribe_streamed_progress(
    struct parakeet_context* ctx, const float* samples, int n_samples, int64_t t_offset_cs, int chunk_seconds,
    int overlap_seconds, void (*progress_cb)(int processed, int total, void* user_data), void* progress_ud) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    if (chunk_seconds <= 0) {
        // Per-model default — empirically swept on en/de/ja FLEURS 60s + 300s
        // (audio_samples/{en,de}/fleurs_*.wav, long-clips/yt_60s.wav). The
        // small (8 s) chunk-and-overlap that ships well for the JA model
        // collapses badly on the multilingual v3 model: EN 60 s drops from
        // 800 chars (c=40) to 186 chars (c=8) — a 4× loss of interior
        // content. The v3 encoder's Conformer attention needs ~30 s of
        // context to produce features close to its training distribution;
        // smaller chunks shift the per-feature statistics enough that the
        // TDT decoder emits a different (much sparser) token path.
        //
        // Heuristic: distinguish the JA-only model (vocab=3072) from the
        // multilingual / v3 / etc. variants (vocab >= 4096) via vocab_size.
        // The override env var CRISPASR_PARAKEET_STREAM_CHUNK is honoured
        // upstream of this default.
        chunk_seconds = parakeet_vocab_is_japanese(ctx) ? 8 : 30; // #257: content, not vocab size
    }
    if (overlap_seconds < 0)
        overlap_seconds = 2;
    if (overlap_seconds >= chunk_seconds)
        overlap_seconds = chunk_seconds / 4;

    const auto& hp = ctx->model.hparams;
    const int n_mels = (int)hp.n_mels;
    const int d_model = (int)hp.d_model;
    const int hop = (int)hp.hop_length;
    const int sub = (int)hp.subsampling_factor;
    const int SR = 16000;

    // ---- Pass 1: compute mel for the FULL audio with global z-norm ----
    // This is equivalent to what parakeet_transcribe_ex does for mel,
    // but we keep the full mel buffer for chunked encoding in pass 2.
    int T_mel = 0;
    auto mel_full = parakeet_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel_full.empty() || T_mel <= 0)
        return nullptr;

    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet[streamed]: full mel: %d frames from %d samples\n", T_mel, n_samples);

    // ---- Pass 2: encode mel in overlapping chunks ----
    // The mel is already globally z-normalized, so we just slice it
    // into chunks, encode each, and concatenate the encoder outputs.
    const int chunk_mel_frames = chunk_seconds * SR / hop; // 8s = 800 mel frames
    const int overlap_mel_frames = overlap_seconds * SR / hop;
    const int shift_mel_frames = chunk_mel_frames - overlap_mel_frames;
    const int overlap_enc_frames = overlap_mel_frames / sub;

    std::vector<float> enc_all;
    int T_enc_total = 0;

    for (int mel_offset = 0; mel_offset < T_mel; mel_offset += shift_mel_frames) {
        const int chunk_end = std::min(T_mel, mel_offset + chunk_mel_frames);
        const int chunk_T = chunk_end - mel_offset;
        if (chunk_T <= 0)
            break;

        // Encode this mel chunk
        int T_enc = 0;
        auto enc = parakeet_encode_mel(ctx, mel_full.data() + (size_t)mel_offset * n_mels, n_mels, chunk_T, &T_enc);
        // Issue #385: a window that fails to encode (or contributes no frames
        // after the overlap skip) still has to REPORT, or a caller's progress
        // bar stalls short of the end. So the append is nested rather than
        // `continue`d past — the report below runs for every finished window.
        if (!enc.empty()) {
            // Skip overlap encoder frames for non-first chunks
            int skip = 0;
            if (mel_offset > 0 && T_enc > overlap_enc_frames)
                skip = overlap_enc_frames;

            const int keep = T_enc - skip;
            if (keep > 0) {
                enc_all.insert(enc_all.end(), enc.begin() + (size_t)skip * d_model,
                               enc.begin() + (size_t)(skip + keep) * d_model);
                T_enc_total += keep;

                if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
                    fprintf(stderr, "parakeet[streamed]: mel chunk @%d: %d mel → %d enc (skip %d, keep %d)\n",
                            mel_offset, chunk_T, T_enc, skip, keep);
            }
        }

        // Issue #385: (samples encoded so far, total). chunk_end is monotonic
        // in mel_offset, and the LAST window is pinned to n_samples exactly —
        // T_mel * hop can land a few samples short of n_samples, and a caller
        // asserting the sequence ends at `total` must not trip over that.
        if (progress_cb) {
            const int done =
                (chunk_end >= T_mel) ? n_samples : (int)std::min<int64_t>(n_samples, (int64_t)chunk_end * hop);
            progress_cb(done, n_samples, progress_ud);
        }
    }

    if (enc_all.empty() || T_enc_total <= 0)
        return nullptr;

    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet[streamed]: total %d enc frames → single TDT decode\n", T_enc_total);

    // ---- Single TDT decode over concatenated encoder output ----
    return parakeet_decode_frames(ctx, enc_all.data(), T_enc_total, d_model, t_offset_cs);
}

// The historical signature — unchanged behaviour, no progress hook.
extern "C" struct parakeet_result* parakeet_transcribe_streamed(struct parakeet_context* ctx, const float* samples,
                                                                int n_samples, int64_t t_offset_cs, int chunk_seconds,
                                                                int overlap_seconds) {
    return parakeet_transcribe_streamed_progress(ctx, samples, n_samples, t_offset_cs, chunk_seconds, overlap_seconds,
                                                 nullptr, nullptr);
}

// ---------------------------------------------------------------------------

extern "C" struct parakeet_result* parakeet_transcribe_ex(struct parakeet_context* ctx, const float* samples,
                                                          int n_samples, int64_t t_offset_cs) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    // 1. Mel
    int T_mel = 0;
    std::vector<float> mel;
    {
        parakeet_bench_stage _b("mel");
        mel = parakeet_compute_mel_impl(ctx, samples, n_samples, T_mel);
    }
    if (mel.empty())
        return nullptr;
    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: mel OK (%d frames)\n", T_mel);

    // 2. Encoder
    int T_enc = 0;
    std::vector<float> enc;
    {
        parakeet_bench_stage _b("encoder");
        enc = parakeet_encode_mel(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, &T_enc);
    }
    if (enc.empty())
        return nullptr;
    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG")) {
        fprintf(stderr, "parakeet: encoder OK (%d frames)\n", T_enc);
        int d = (int)ctx->model.hparams.d_model;
        double s = 0, sq = 0;
        for (int i = 0; i < T_enc * d; i++) {
            s += enc[i];
            sq += (double)enc[i] * enc[i];
        }
        double m = s / (T_enc * d), v = sq / (T_enc * d) - m * m;
        fprintf(stderr, "parakeet: enc mean=%.4f std=%.4f enc[0,:4]=%.4f %.4f %.4f %.4f\n", m, sqrt(v), (double)enc[0],
                (double)enc[1], (double)enc[2], (double)enc[3]);
    }

    // 3. Decode (TDT or CTC, greedy or beam)
    parakeet_bench_stage _b_dec("decode");
    const bool use_ctc = ctx->decode_ctc && ctx->model.has_ctc;
    const bool use_rnnt = !use_ctc && ctx->model.hparams.n_tdt_durations == 0;
    const bool use_beam = !use_ctc && ctx->decode_beam_size > 1;
    const bool use_maes = use_beam && ctx->decode_maes;
    const int d = (int)ctx->model.hparams.d_model;
    auto emitted =
        use_ctc ? parakeet_ctc_decode(ctx, enc.data(), T_enc, d)
        : use_rnnt
            ? (use_maes   ? parakeet_rnnt_maes_decode(ctx, enc.data(), T_enc, d, ctx->decode_beam_size,
                                                      ctx->maes_num_steps, ctx->maes_gamma, ctx->maes_beta)
               : use_beam ? parakeet_rnnt_beam_decode(ctx, enc.data(), T_enc, d, ctx->decode_beam_size)
                          : parakeet_rnnt_decode(ctx, enc.data(), T_enc, d))
            : (use_maes   ? parakeet_tdt_maes_decode(ctx, enc.data(), T_enc, d, ctx->decode_beam_size,
                                                     ctx->maes_num_steps, ctx->maes_gamma, ctx->maes_beta)
               : use_beam ? parakeet_tdt_beam_decode(ctx, enc.data(), T_enc, d, ctx->decode_beam_size)
                          : (getenv("CRISPASR_TDT_BATCH") ? parakeet_tdt_decode_batched(ctx, enc.data(), T_enc, d)
                                                          : parakeet_tdt_decode(ctx, enc.data(), T_enc, d)));
    if (crispasr_env::get("CRISPASR_PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: %s%s decode OK (%d tokens)\n",
                use_ctc    ? "CTC"
                : use_rnnt ? "RNNT"
                           : "TDT",
                use_beam ? " beam" : "", (int)emitted.size());

    // 4. Build result
    auto* r = (parakeet_result*)calloc(1, sizeof(parakeet_result));
    r->n_tokens = (int)emitted.size();
    r->tokens = (parakeet_token_data*)calloc(r->n_tokens > 0 ? r->n_tokens : 1, sizeof(parakeet_token_data));
    std::string text;
    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;
    for (int i = 0; i < r->n_tokens; i++) {
        const auto& e = emitted[i];
        const std::string& piece =
            (e.id >= 0 && e.id < (int)ctx->vocab.id_to_token.size()) ? ctx->vocab.id_to_token[e.id] : std::string("");
        std::string vis = spiece_to_text(piece);

        parakeet_token_data& td = r->tokens[i];
        td.id = e.id;
        td.t0 = t_offset_cs + (int64_t)e.t_start * frame_dur_cs;
        td.t1 = t_offset_cs + (int64_t)e.t_end * frame_dur_cs;
        td.p = e.p;
        size_t n = std::min(vis.size(), sizeof(td.text) - 1);
        memcpy(td.text, vis.data(), n);
        td.text[n] = '\0';
        text += vis;
    }
    // strip leading space
    if (!text.empty() && text[0] == ' ')
        text = text.substr(1);
    r->text = strdup(text.c_str());

    parakeet_group_words(r, frame_dur_cs);

    return r;
}

extern "C" char* parakeet_transcribe(struct parakeet_context* ctx, const float* samples, int n_samples) {
    parakeet_result* r = parakeet_transcribe_ex(ctx, samples, n_samples, 0);
    if (!r)
        return nullptr;
    char* out = strdup(r->text ? r->text : "");
    parakeet_result_free(r);
    return out;
}
