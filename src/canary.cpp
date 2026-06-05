// canary.cpp — nvidia/canary-1b-v2 ggml runtime
//
// First iteration: loader + public C API skeleton.
// Encoder forward (FastConformer with biases), Transformer decoder
// (self-attn + cross-attn + FFN with KV cache), task-token prompt and
// greedy decode loop will land in subsequent commits.
//
// Architecture:
//   Mel:           128 mels @ 16 kHz, n_fft=512, win=400, hop=160 (Hann)
//   Encoder:       32× FastConformer block (use_bias=True), d_model=1024,
//                  8 heads, head_dim=128, ff_dim=4096, conv kernel=9,
//                  8× temporal subsampling via dw_striding
//   Decoder:       8× pre-LN Transformer block (SA + CA + FFN),
//                  d_model=1024, 8 heads, head_dim=128, ff_dim=4096
//   Embedding:     token (16384, 1024) + learned pos_enc (1024, 1024) + LN
//   Output head:   linear (1024 → 16384)
//
// Decoder prompt format (mirrors Cohere — same vocab):
//   <|startoftranscript|> <|src|> <|tgt|> <|pnc|> <|notimestamp|> <|nodiarize|> ...

#include "canary.h"

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
#include "core/attention.h"
#include "core/beam_decode.h"
#include "core/crispasr_lcs.h"
#include "core/fastconformer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <algorithm>
#include <cmath>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
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

struct canary_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 512;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;
    uint32_t d_model = 1024;
    uint32_t enc_n_layers = 32;
    uint32_t dec_n_layers = 8;
    uint32_t n_heads = 8;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 4096;
    uint32_t subsampling_factor = 8;
    uint32_t subsampling_channels = 256;
    uint32_t conv_kernel = 9;
    uint32_t vocab_size = 16384;
    uint32_t max_dec_ctx = 1024;
    uint32_t frame_dur_cs = 8;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

// Pre-encode weights: exactly the shared FastConformer layout.
using canary_pre_encode = core_conformer::PreEncodeWeights;

// Per-layer tensor container: inherits the shared Conformer block weights
// (all biases populated for canary) and adds the BN tensors used only at
// load time (BN folding).
struct canary_enc_layer : core_conformer::BlockWeights {
    ggml_tensor *conv_bn_w = nullptr, *conv_bn_b = nullptr;
    ggml_tensor *conv_bn_rm = nullptr, *conv_bn_rv = nullptr;
};

struct canary_dec_layer {
    // Pre-LN block layout:
    //   x = x + sa_out @ SA(norm_sa(x))
    //   x = x + ca_out @ CA(norm_ca(x), enc_kv)
    //   x = x + ff_out @ activation(ff_in @ norm_ff(x))
    ggml_tensor *norm_sa_w = nullptr, *norm_sa_b = nullptr;
    ggml_tensor *sa_q_w = nullptr, *sa_q_b = nullptr;
    ggml_tensor *sa_k_w = nullptr, *sa_k_b = nullptr;
    ggml_tensor *sa_v_w = nullptr, *sa_v_b = nullptr;
    ggml_tensor *sa_out_w = nullptr, *sa_out_b = nullptr;

    ggml_tensor *norm_ca_w = nullptr, *norm_ca_b = nullptr;
    ggml_tensor *ca_q_w = nullptr, *ca_q_b = nullptr;
    ggml_tensor *ca_k_w = nullptr, *ca_k_b = nullptr;
    ggml_tensor *ca_v_w = nullptr, *ca_v_b = nullptr;
    ggml_tensor *ca_out_w = nullptr, *ca_out_b = nullptr;

    ggml_tensor *norm_ff_w = nullptr, *norm_ff_b = nullptr;
    ggml_tensor *ff_in_w = nullptr, *ff_in_b = nullptr;
    ggml_tensor *ff_out_w = nullptr, *ff_out_b = nullptr;
};

// ===========================================================================
// Model
// ===========================================================================

struct canary_model {
    canary_hparams hparams;

    ggml_tensor* mel_fb = nullptr;
    ggml_tensor* mel_window = nullptr;

    canary_pre_encode pre_encode;
    std::vector<canary_enc_layer> enc;
    std::vector<canary_dec_layer> dec;

    // Decoder embeddings + final norm + output head
    ggml_tensor* dec_embed_w = nullptr; // (vocab, d_model)
    ggml_tensor* dec_pos_enc = nullptr; // (max_ctx, d_model) — learned
    ggml_tensor* dec_embed_ln_w = nullptr;
    ggml_tensor* dec_embed_ln_b = nullptr;
    ggml_tensor* dec_final_ln_w = nullptr;
    ggml_tensor* dec_final_ln_b = nullptr;
    ggml_tensor* dec_head_w = nullptr; // (vocab, d_model)
    ggml_tensor* dec_head_b = nullptr;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::map<std::string, ggml_tensor*> tensors;
};

struct canary_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    // PLAN #114 P3 polish — case-insensitive LCS support.
    // `id_to_canonical_lc[i]` is the smallest token id whose
    // lowercase-ASCII text matches token `i`. Used by
    // canary_transcribe_streamed when looking for boundary-overlap
    // matches across chunks where the AED re-emits the same audio with
    // different capitalization (e.g. "world's say for" in chunk 1 vs
    // "World's Save for" in chunk 2 — different raw ids but the LCS
    // should still match them). Populated once at vocab load; ASCII-only
    // (UTF-8 case folding would need libicu and DE/FR/ES chunk-boundary
    // capitalization is rare enough that ASCII is sufficient for the
    // present polish).
    std::vector<int32_t> id_to_canonical_lc;
};

struct canary_context {
    canary_context_params params;

    canary_model model;
    canary_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // Self-attention KV cache for the decoder
    // shape: [head_dim, max_ctx, n_heads, dec_n_layers]
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;

    // Cross-attention K/V — pre-computed once per slice from encoder output.
    // One tensor per decoder layer, shape [head_dim, T_enc, n_heads].
    ggml_context* cross_ctx = nullptr;
    ggml_backend_buffer_t cross_buf = nullptr;
    std::vector<ggml_tensor*> cross_k;
    std::vector<ggml_tensor*> cross_v;

    // Per-step cross-attention weights from the last decoder layer, captured
    // when collect_attn=true and n_tokens==1. Used for DTW timestamp alignment.
    // step_attn[step_idx] has T_enc * n_heads floats (head h occupies the
    // contiguous slice [h*T_enc, h*T_enc + T_enc)).
    bool collect_attn = false;
    int attn_T_enc = 0;
    int attn_n_heads = 0;
    std::vector<std::vector<float>> step_attn;

    int n_threads = 4;

    // Sticky decode-time sampling controls. temperature == 0 keeps the
    // bit-identical greedy path; > 0 switches to numerically-stable
    // softmax sampling. Set via canary_set_temperature().
    float decode_temperature = 0.0f;
    uint64_t decode_seed = 0;

    // §90 beam-search width. 1 = greedy (default).
    int beam_size = 1;
};

// ===========================================================================
// Loader helpers — thin wrappers around core_gguf::.
// ===========================================================================

#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static ggml_tensor* try_get(canary_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(canary_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "canary");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool canary_load_model(canary_model& model, canary_vocab& vocab, const char* path, ggml_backend_t backend) {
    // ---- pass 1: hparams + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "canary.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "canary.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "canary.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "canary.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "canary.hop_length", hp.hop_length);
        hp.d_model = core_gguf::kv_u32(gctx, "canary.d_model", hp.d_model);
        hp.enc_n_layers = core_gguf::kv_u32(gctx, "canary.enc_n_layers", hp.enc_n_layers);
        hp.dec_n_layers = core_gguf::kv_u32(gctx, "canary.dec_n_layers", hp.dec_n_layers);
        hp.n_heads = core_gguf::kv_u32(gctx, "canary.n_heads", hp.n_heads);
        hp.head_dim = core_gguf::kv_u32(gctx, "canary.head_dim", hp.head_dim);
        hp.ff_dim = core_gguf::kv_u32(gctx, "canary.ff_dim", hp.ff_dim);
        hp.subsampling_factor = core_gguf::kv_u32(gctx, "canary.subsampling_factor", hp.subsampling_factor);
        hp.subsampling_channels = core_gguf::kv_u32(gctx, "canary.subsampling_channels", hp.subsampling_channels);
        hp.conv_kernel = core_gguf::kv_u32(gctx, "canary.conv_kernel", hp.conv_kernel);
        hp.vocab_size = core_gguf::kv_u32(gctx, "canary.vocab_size", hp.vocab_size);
        hp.max_dec_ctx = core_gguf::kv_u32(gctx, "canary.max_dec_ctx", hp.max_dec_ctx);
        hp.frame_dur_cs = core_gguf::kv_u32(gctx, "canary.frame_dur_cs", hp.frame_dur_cs);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
            // PLAN #114 P3 polish — build the case-insensitive canonical
            // mapping. ASCII lowercase only; non-ASCII bytes (UTF-8) pass
            // through unchanged, so DE-umlaut tokens still match their
            // own variants but not cross-case. Sufficient for the
            // observed EN FLEURS chunk-boundary capitalization artifacts.
            const int n_vocab = (int)vocab.id_to_token.size();
            vocab.id_to_canonical_lc.assign((size_t)n_vocab, 0);
            std::unordered_map<std::string, int> first_id_for_lc;
            first_id_for_lc.reserve((size_t)n_vocab);
            for (int i = 0; i < n_vocab; i++) {
                std::string lc = vocab.id_to_token[i];
                for (char& c : lc) {
                    if (c >= 'A' && c <= 'Z')
                        c = (char)(c + ('a' - 'A'));
                }
                auto it = first_id_for_lc.find(lc);
                if (it == first_id_for_lc.end()) {
                    first_id_for_lc.emplace(std::move(lc), i);
                    vocab.id_to_canonical_lc[(size_t)i] = i;
                } else {
                    vocab.id_to_canonical_lc[(size_t)i] = it->second;
                }
            }
        }

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: tensor data via shared helper ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "canary", wl)) {
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
    model.enc.resize(model.hparams.enc_n_layers);
    for (uint32_t i = 0; i < model.hparams.enc_n_layers; i++) {
        char buf[128];
        auto& e = model.enc[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return require(model, buf);
        };

        e.norm_ff1_w = get("norm_ff1.weight");
        e.norm_ff1_b = get("norm_ff1.bias");
        e.ff1_l1_w = get("ff1.linear1.weight");
        e.ff1_l1_b = get("ff1.linear1.bias");
        e.ff1_l2_w = get("ff1.linear2.weight");
        e.ff1_l2_b = get("ff1.linear2.bias");

        e.norm_attn_w = get("norm_attn.weight");
        e.norm_attn_b = get("norm_attn.bias");
        e.attn_q_w = get("attn.q.weight");
        e.attn_q_b = get("attn.q.bias");
        e.attn_k_w = get("attn.k.weight");
        e.attn_k_b = get("attn.k.bias");
        e.attn_v_w = get("attn.v.weight");
        e.attn_v_b = get("attn.v.bias");
        e.attn_out_w = get("attn.out.weight");
        e.attn_out_b = get("attn.out.bias");
        e.attn_pos_w = get("attn.pos.weight");
        e.pos_bias_u = get("attn.pos_bias_u");
        e.pos_bias_v = get("attn.pos_bias_v");

        e.norm_conv_w = get("norm_conv.weight");
        e.norm_conv_b = get("norm_conv.bias");
        e.conv_pw1_w = get("conv.pw1.weight");
        e.conv_pw1_b = get("conv.pw1.bias");
        e.conv_dw_w = get("conv.dw.weight");
        e.conv_dw_b = get("conv.dw.bias");
        e.conv_pw2_w = get("conv.pw2.weight");
        e.conv_pw2_b = get("conv.pw2.bias");
        e.conv_bn_w = get("conv.bn.weight");
        e.conv_bn_b = get("conv.bn.bias");
        e.conv_bn_rm = get("conv.bn.running_mean");
        e.conv_bn_rv = get("conv.bn.running_var");

        e.norm_ff2_w = get("norm_ff2.weight");
        e.norm_ff2_b = get("norm_ff2.bias");
        e.ff2_l1_w = get("ff2.linear1.weight");
        e.ff2_l1_b = get("ff2.linear1.bias");
        e.ff2_l2_w = get("ff2.linear2.weight");
        e.ff2_l2_b = get("ff2.linear2.bias");

        e.norm_out_w = get("norm_out.weight");
        e.norm_out_b = get("norm_out.bias");
    }

    // Decoder
    model.dec.resize(model.hparams.dec_n_layers);
    for (uint32_t i = 0; i < model.hparams.dec_n_layers; i++) {
        char buf[128];
        auto& d = model.dec[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "decoder.layers.%u.%s", i, suf);
            return require(model, buf);
        };

        d.norm_sa_w = get("norm_sa.weight");
        d.norm_sa_b = get("norm_sa.bias");
        d.sa_q_w = get("sa_q.weight");
        d.sa_q_b = get("sa_q.bias");
        d.sa_k_w = get("sa_k.weight");
        d.sa_k_b = get("sa_k.bias");
        d.sa_v_w = get("sa_v.weight");
        d.sa_v_b = get("sa_v.bias");
        d.sa_out_w = get("sa_out.weight");
        d.sa_out_b = get("sa_out.bias");

        d.norm_ca_w = get("norm_ca.weight");
        d.norm_ca_b = get("norm_ca.bias");
        d.ca_q_w = get("ca_q.weight");
        d.ca_q_b = get("ca_q.bias");
        d.ca_k_w = get("ca_k.weight");
        d.ca_k_b = get("ca_k.bias");
        d.ca_v_w = get("ca_v.weight");
        d.ca_v_b = get("ca_v.bias");
        d.ca_out_w = get("ca_out.weight");
        d.ca_out_b = get("ca_out.bias");

        d.norm_ff_w = get("norm_ff.weight");
        d.norm_ff_b = get("norm_ff.bias");
        d.ff_in_w = get("ff_in.weight");
        d.ff_in_b = get("ff_in.bias");
        d.ff_out_w = get("ff_out.weight");
        d.ff_out_b = get("ff_out.bias");
    }

    // Decoder embeddings + output head
    model.dec_embed_w = require(model, "decoder.embed.weight");
    model.dec_pos_enc = require(model, "decoder.pos_enc");
    model.dec_embed_ln_w = require(model, "decoder.embed_ln.weight");
    model.dec_embed_ln_b = require(model, "decoder.embed_ln.bias");
    model.dec_final_ln_w = require(model, "decoder.final_norm.weight");
    model.dec_final_ln_b = require(model, "decoder.final_norm.bias");
    model.dec_head_w = require(model, "decoder.head.weight");
    model.dec_head_b = require(model, "decoder.head.bias");

    fprintf(stderr, "canary: vocab=%u  d_model=%u  enc_layers=%u  dec_layers=%u  heads=%u  ff=%u  max_ctx=%u\n",
            model.hparams.vocab_size, model.hparams.d_model, model.hparams.enc_n_layers, model.hparams.dec_n_layers,
            model.hparams.n_heads, model.hparams.ff_dim, model.hparams.max_dec_ctx);
    return true;
}

// ===========================================================================
// FFT (iterative Cooley-Tukey, real-input, N must be a power of 2)
// ===========================================================================

static void canary_fft_r2c(const float* in, int N, float* out) {
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

// NeMo-style mel: same as parakeet (128 mel, 16 kHz, n_fft=512, win=400, hop=160).
// Delegates to core_mel::compute() with the NeMo cluster's parameters; only
// the FFT function pointer differs between parakeet/canary/canary_ctc/cohere.
#include "core/mel.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static std::vector<float> canary_compute_mel_impl(canary_context* ctx, const float* samples, int n_samples,
                                                  int& T_out) {
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int win = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels = (int)hp.n_mels;

    if (!ctx->model.mel_fb || !ctx->model.mel_window) {
        fprintf(stderr, "canary: missing preprocessor.fb / window\n");
        return {};
    }

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
    p.norm = core_mel::Normalization::PerFeatureZ;
    p.layout = core_mel::Layout::TimeMels;
    p.log_eps = (float)(1.0 / (1 << 24));
    p.center_pad = true;
    p.drop_last_frame = true; // NeMo returns feat_len = floor(n_samples/hop) frames
    p.preemph = 0.97f;        // NeMo AudioToMelSpectrogramPreprocessor default (#37)

    return core_mel::compute(samples, n_samples, window_raw.data(), win, mel_fb.data(), n_freqs, canary_fft_r2c, p,
                             T_out);
}

// ===========================================================================
// rel-pos shift (Transformer-XL trick): single zero-cost ggml_view_3d
// ===========================================================================

// rel_shift and make_pos_enc moved to core_conformer in src/core/fastconformer.h.

// ===========================================================================
// Encoder graph
//
// Same structure as parakeet but with biases on every linear/conv. The
// encoder block is the standard pre-LN Conformer (FFN1/2 macaron, MHA
// with rel-pos untied biases, depthwise sep conv with GLU, final block LN).
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

static ggml_cgraph* canary_build_graph_encoder(canary_context* ctx, int T_mel) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // ----- Inputs -----
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // ----- Pre-encode (dw_striding 8×) -----
    int T = 0;
    ggml_tensor* cur = core_conformer::build_pre_encode(ctx0, mel, m.pre_encode, (int)hp.subsampling_channels, &T);

    // ----- Sinusoidal rel-pos table -----
    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    // ----- 32× FastConformer block (with biases) -----
    core_conformer::BlockParams bp = {
        (int)hp.d_model, (int)hp.n_heads, (int)hp.head_dim, (int)hp.conv_kernel, kLayerNormEps,
    };
    for (uint32_t il = 0; il < hp.enc_n_layers; il++) {
        cur = core_conformer::build_block(ctx0, cur, pos_enc, T, m.enc[il], bp);
    }

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Staged graph: same as above but snapshots intermediate tensors via ggml_dup
// so they survive after the allocator reclaims intermediate buffers.
// Names: "pre_enc_out", "enc_L00".."enc_L31", "enc_out".
// Graph size is larger (~33 extra dup nodes), so we use 24576 max nodes.
static ggml_cgraph* canary_build_graph_encoder_staged(canary_context* ctx, int T_mel) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 24576, false);

    ggml_tensor* mel_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel_t, "mel");
    ggml_set_input(mel_t);

    int T = 0;
    ggml_tensor* cur =
        core_conformer::build_pre_encode(ctx0, mel_t, m.pre_encode, (int)hp.subsampling_channels, &T, gf);

    {
        ggml_tensor* snap = ggml_dup(ctx0, cur);
        ggml_set_name(snap, "pre_enc_out");
        ggml_build_forward_expand(gf, snap);
    }

    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    core_conformer::BlockParams bp = {
        (int)hp.d_model, (int)hp.n_heads, (int)hp.head_dim, (int)hp.conv_kernel, kLayerNormEps,
    };
    char lbuf[32];
    for (uint32_t il = 0; il < hp.enc_n_layers; il++) {
        cur = core_conformer::build_block(ctx0, cur, pos_enc, T, m.enc[il], bp);
        snprintf(lbuf, sizeof(lbuf), "enc_L%02u", il);
        ggml_tensor* snap = ggml_dup(ctx0, cur);
        ggml_set_name(snap, lbuf);
        ggml_build_forward_expand(gf, snap);
    }

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run the encoder once. Returns enc_out as a flat row-major [T_enc * d_model].
static std::vector<float> canary_encode_mel(canary_context* ctx, const float* mel, int n_mels, int T_mel,
                                            int* out_T_enc) {
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "canary: mel feature mismatch (%d vs %d)\n", n_mels, (int)ctx->model.hparams.n_mels);
        return {};
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    }

    ggml_cgraph* gf = canary_build_graph_encoder(ctx, T_mel);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "canary: failed to alloc encoder graph\n");
        return {};
    }

    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2;
    auto pe = core_conformer::make_pos_enc((int)ctx->model.hparams.d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "canary: encoder compute failed\n");
        return {};
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "enc_out");
    if (!out)
        return {};
    const int d = (int)out->ne[0];
    const int Te = (int)out->ne[1];
    if (out_T_enc)
        *out_T_enc = Te;

    std::vector<float> result((size_t)d * Te);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Decoder KV cache + cross-KV allocation
// ===========================================================================

static void canary_alloc_kv(canary_context* ctx) {
    if (ctx->kv_buf)
        return;
    const auto& hp = ctx->model.hparams;
    const int head_dim = (int)hp.head_dim;
    const int n_heads = (int)hp.n_heads;
    const int max_ctx = (int)hp.max_dec_ctx;
    const int n_layers = (int)hp.dec_n_layers;

    ggml_init_params p = {
        /*mem_size=*/ggml_tensor_overhead() * 4,
        /*mem_buffer=*/nullptr,
        /*no_alloc=*/true,
    };
    ctx->kv_ctx = ggml_init(p);

    // PLAN #60e + #69e: per-half KV dtype. Canary's per-step write
    // goes through core_attn::kv_cache_write (PLAN #73), and the
    // read path adds a ggml_cast(F32) before the permute+cont chain
    // when the cache is quant — keeps memory savings, gives up some
    // read-bandwidth savings.
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("canary");
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, head_dim, max_ctx, n_heads, n_layers);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, head_dim, max_ctx, n_heads, n_layers);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    // PLAN #69b: optional KV-on-CPU spill for VRAM-tight users.
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "canary");
    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, kv_backend);
}

// Build the cross-K/V tensors from encoder output. Called once per slice.
// enc: flat [T_enc * d_model] (row-major, ne[0]=d_model fastest)
static void canary_build_cross_kv(canary_context* ctx, const float* enc_data, int T_enc) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.d_model;
    const int head_dim = (int)hp.head_dim;
    const int n_heads = (int)hp.n_heads;
    const int n_layers = (int)hp.dec_n_layers;

    // Allocate cross_kv tensors on a CPU buffer (small, ~1 MB per layer for T_enc≈100)
    if (ctx->cross_ctx) {
        ggml_backend_buffer_free(ctx->cross_buf);
        ggml_free(ctx->cross_ctx);
        ctx->cross_ctx = nullptr;
        ctx->cross_buf = nullptr;
        ctx->cross_k.clear();
        ctx->cross_v.clear();
    }

    ggml_init_params p = {
        /*mem_size=*/ggml_tensor_overhead() * (n_layers * 2 + 8),
        /*mem_buffer=*/nullptr,
        /*no_alloc=*/true,
    };
    ctx->cross_ctx = ggml_init(p);
    ctx->cross_k.resize(n_layers);
    ctx->cross_v.resize(n_layers);

    for (int il = 0; il < n_layers; il++) {
        // ggml_flash_attn_ext requires F16 K/V on the CPU backend (the F32 path
        // is not implemented and falls through to broken behaviour). Match the
        // pattern cohere.cpp uses.
        ctx->cross_k[il] = ggml_new_tensor_3d(ctx->cross_ctx, GGML_TYPE_F16, head_dim, T_enc, n_heads);
        ctx->cross_v[il] = ggml_new_tensor_3d(ctx->cross_ctx, GGML_TYPE_F16, head_dim, T_enc, n_heads);
    }
    // Allocate on the SAME backend the decoder graph runs on. Allocating on
    // a separate CPU backend instance breaks the scheduler's cross-graph
    // tensor references — even when both backends happen to be CPU.
    ctx->cross_buf = ggml_backend_alloc_ctx_tensors(ctx->cross_ctx, ctx->backend);

    // Compute K/V projections per layer using a tiny graph
    for (int il = 0; il < n_layers; il++) {
        const auto& dl = m.dec[il];

        ggml_init_params gp = {
            /*mem_size=*/ctx->compute_meta.size(),
            /*mem_buffer=*/ctx->compute_meta.data(),
            /*no_alloc=*/true,
        };
        ggml_context* gctx = ggml_init(gp);
        ggml_cgraph* gf = ggml_new_graph_custom(gctx, 64, false);

        ggml_tensor* enc = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d, T_enc);
        ggml_set_name(enc, "enc_in");
        ggml_set_input(enc);

        // CK: linear(enc) → bias → reshape [hd, n_heads, T_enc] → permute [hd, T_enc, n_heads]
        ggml_tensor* CK = ggml_add(gctx, ggml_mul_mat(gctx, dl.ca_k_w, enc), dl.ca_k_b);
        CK = ggml_cont(gctx, ggml_permute(gctx, ggml_reshape_3d(gctx, CK, head_dim, n_heads, T_enc), 0, 2, 1, 3));
        ggml_set_name(CK, "CK");

        ggml_tensor* CV = ggml_add(gctx, ggml_mul_mat(gctx, dl.ca_v_w, enc), dl.ca_v_b);
        CV = ggml_cont(gctx, ggml_permute(gctx, ggml_reshape_3d(gctx, CV, head_dim, n_heads, T_enc), 0, 2, 1, 3));
        ggml_set_name(CV, "CV");

        ggml_build_forward_expand(gf, CK);
        ggml_build_forward_expand(gf, CV);

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "canary: cross-kv alloc failed\n");
            ggml_free(gctx);
            return;
        }

        ggml_tensor* enc_in = ggml_graph_get_tensor(gf, "enc_in");
        ggml_backend_tensor_set(enc_in, enc_data, 0, (size_t)d * T_enc * sizeof(float));

        ggml_backend_sched_graph_compute(ctx->sched, gf);

        ggml_tensor* CK_out = ggml_graph_get_tensor(gf, "CK");
        ggml_tensor* CV_out = ggml_graph_get_tensor(gf, "CV");
        std::vector<float> kbuf((size_t)head_dim * T_enc * n_heads);
        std::vector<float> vbuf((size_t)head_dim * T_enc * n_heads);
        ggml_backend_tensor_get(CK_out, kbuf.data(), 0, kbuf.size() * sizeof(float));
        ggml_backend_tensor_get(CV_out, vbuf.data(), 0, vbuf.size() * sizeof(float));

        // Convert F32 → F16 before uploading into the F16 cross-KV slots.
        std::vector<ggml_fp16_t> kbuf16(kbuf.size());
        std::vector<ggml_fp16_t> vbuf16(vbuf.size());
        ggml_fp32_to_fp16_row(kbuf.data(), kbuf16.data(), kbuf.size());
        ggml_fp32_to_fp16_row(vbuf.data(), vbuf16.data(), vbuf.size());
        ggml_backend_tensor_set(ctx->cross_k[il], kbuf16.data(), 0, kbuf16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(ctx->cross_v[il], vbuf16.data(), 0, vbuf16.size() * sizeof(ggml_fp16_t));

        ggml_free(gctx);
    }
}

// ===========================================================================
// Decoder per-step graph (autoregressive, with self-attn KV cache + pre-built cross-KV)
// ===========================================================================

static ggml_cgraph* canary_build_graph_decoder(canary_context* ctx, int n_tokens, int offset) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.d_model;
    const int n_heads = (int)hp.n_heads;
    const int head_dim = (int)hp.head_dim;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // PLAN #73: causal mask input for ggml_flash_attn_ext on the
    // self-attention path. Created here once and reused by every layer.
    // Shape [L, n_tokens] F16: 0 = unmasked, -INF = masked. Only needed
    // for prefill (n_tokens > 1); decode steps (n_tokens=1) pass nullptr.
    const int L = offset + n_tokens;
    ggml_tensor* sa_mask = nullptr;
    if (n_tokens > 1) {
        sa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, L, n_tokens);
        ggml_set_name(sa_mask, "sa_mask");
        ggml_set_input(sa_mask);
    }
    ggml_tensor* embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(embd, "embd");
    ggml_set_input(embd);

    ggml_tensor* position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    // Token embed + positional embed (learned)
    ggml_tensor* cur =
        ggml_add(ctx0, ggml_get_rows(ctx0, m.dec_embed_w, embd), ggml_get_rows(ctx0, m.dec_pos_enc, position));

    // Embedding LayerNorm
    cur = ggml_norm(ctx0, cur, kLayerNormEps);
    cur = ggml_mul(ctx0, cur, m.dec_embed_ln_w);
    cur = ggml_add(ctx0, cur, m.dec_embed_ln_b);

    for (uint32_t il = 0; il < hp.dec_n_layers; il++) {
        const auto& dl = m.dec[il];
        ggml_tensor* inpL = cur;

        // ---- Self-attention (PRE-LN: x = x + SA(LN1(x)), causal, with KV cache) ----
        // Confirmed via NeMo source (transformer_decoders.py forward_preln):
        //   residual = decoder_query
        //   decoder_query = layer_norm_1(decoder_query)
        //   decoder_keys  = layer_norm_1(decoder_keys)
        //   self_attn_output = first_sub_layer(decoder_query, decoder_keys, decoder_keys, mask)
        //   self_attn_output += residual
        cur = ggml_norm(ctx0, inpL, kLayerNormEps);
        cur = ggml_mul(ctx0, cur, dl.norm_sa_w);
        cur = ggml_add(ctx0, cur, dl.norm_sa_b);

        ggml_tensor* Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_q_w, cur), dl.sa_q_b);
        ggml_tensor* Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_k_w, cur), dl.sa_k_b);
        ggml_tensor* Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_v_w, cur), dl.sa_v_b);

        // PLAN #73: cache write via core_attn helper. F16/F32 caches go
        // the legacy ggml_cpy(view) path (bit-identical to before);
        // Q8_0/Q4_0 caches go ggml_set_rows(position). The `position`
        // tensor is already populated with [offset, offset+1, …,
        // offset+n_tokens-1] for the embedding-table lookup, so it
        // doubles as the row-index input.
        {
            ggml_tensor* K_perm =
                ggml_permute(ctx0, ggml_reshape_3d(ctx0, Kcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);
            ggml_tensor* V_perm =
                ggml_permute(ctx0, ggml_reshape_3d(ctx0, Vcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);
            core_attn::kv_cache_write(ctx0, gf, K_perm, V_perm, ctx->kv_k, ctx->kv_v, il, offset, n_tokens, position);
        }

        // PLAN #73: ggml_flash_attn_ext fuses K-mul-Q + softmax + V-mul
        // into one op and natively handles quant K/V (no cast tax). Mask
        // is the shared `sa_mask` graph input populated host-side per
        // call. For decode steps (n_tokens=1) mask is nullptr — the
        // causal constraint is trivially satisfied since there's only
        // one query position.
        ggml_tensor* Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Qcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);
        ggml_tensor* K_full = ggml_view_3d(ctx0, ctx->kv_k, head_dim, L, n_heads, ctx->kv_k->nb[1], ctx->kv_k->nb[2],
                                           il * ctx->kv_k->nb[3]);
        ggml_tensor* V_full = ggml_view_3d(ctx0, ctx->kv_v, head_dim, L, n_heads, ctx->kv_v->nb[1], ctx->kv_v->nb[2],
                                           il * ctx->kv_v->nb[3]);
        ggml_tensor* sa_out = ggml_flash_attn_ext(ctx0, ggml_cont(ctx0, Q), K_full, V_full, sa_mask,
                                                  1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
        // Output: [hd, n_heads, n_tokens] — already in the layout the
        // legacy code produced after its trailing permute(0,2,1,3).
        sa_out = ggml_reshape_2d(ctx0, sa_out, d, n_tokens);
        cur = sa_out;

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_out_w, cur), dl.sa_out_b);
        cur = ggml_add(ctx0, cur, inpL);

        // ---- Cross-attention (PRE-LN: x = x + CA(LN2(x)), no causal mask) ----
        ggml_tensor* inpCA = cur;
        cur = ggml_norm(ctx0, cur, kLayerNormEps);
        cur = ggml_mul(ctx0, cur, dl.norm_ca_w);
        cur = ggml_add(ctx0, cur, dl.norm_ca_b);

        ggml_tensor* CQ = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ca_q_w, cur), dl.ca_q_b);
        CQ = ggml_reshape_3d(ctx0, CQ, head_dim, n_heads, n_tokens);
        CQ = ggml_permute(ctx0, CQ, 0, 2, 1, 3);

        ggml_tensor* CK = ctx->cross_k[il];
        ggml_tensor* CV = ctx->cross_v[il];

        ggml_tensor* ca_out = ggml_flash_attn_ext(ctx0, CQ, CK, CV, nullptr, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);

        // Capture cross-attention weights for the last decoder layer in
        // single-token generation mode. Used by the DTW timestamp pass.
        // CK shape: [head_dim, T_enc, n_heads] (F16)
        // CQ shape: [head_dim, n_tokens=1, n_heads]
        // ca_w = softmax( CK^T @ CQ / sqrt(head_dim) ) → [T_enc, 1, n_heads]
        if (ctx->collect_attn && (int)il == (int)hp.dec_n_layers - 1 && n_tokens == 1) {
            ggml_tensor* ca_w = ggml_mul_mat(ctx0, CK, CQ); // [T_enc, 1, n_heads]
            ca_w = ggml_soft_max_ext(ctx0, ca_w, nullptr, 1.0f / sqrtf((float)head_dim), 0.0f);
            ggml_set_name(ca_w, "ca_attn_w");
            ggml_build_forward_expand(gf, ca_w);
        }

        cur = ggml_reshape_2d(ctx0, ca_out, d, n_tokens);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ca_out_w, cur), dl.ca_out_b);
        cur = ggml_add(ctx0, cur, inpCA);

        // ---- FFN (PRE-LN: x = x + FFN(LN3(x)), ReLU activation) ----
        ggml_tensor* inpFF = cur;
        cur = ggml_norm(ctx0, cur, kLayerNormEps);
        cur = ggml_mul(ctx0, cur, dl.norm_ff_w);
        cur = ggml_add(ctx0, cur, dl.norm_ff_b);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ff_in_w, cur), dl.ff_in_b);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ff_out_w, cur), dl.ff_out_b);
        cur = ggml_add(ctx0, cur, inpFF);
    }

    // Final layer norm + output head
    cur = ggml_norm(ctx0, cur, kLayerNormEps);
    cur = ggml_mul(ctx0, cur, m.dec_final_ln_w);
    cur = ggml_add(ctx0, cur, m.dec_final_ln_b);

    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.dec_head_w, cur), m.dec_head_b);
    ggml_set_name(cur, "logits");

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run one decoder step (or batch). Returns logits for the LAST token.
static std::vector<float> canary_decode_step(canary_context* ctx, const int* tokens, int n_tokens, int offset) {
    canary_alloc_kv(ctx);
    ggml_cgraph* gf = canary_build_graph_decoder(ctx, n_tokens, offset);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "canary: decoder alloc failed\n");
        return {};
    }

    ggml_tensor* embd = ggml_graph_get_tensor(gf, "embd");
    ggml_backend_tensor_set(embd, tokens, 0, n_tokens * sizeof(int));

    std::vector<int> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos[i] = offset + i;
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "position");
    ggml_backend_tensor_set(pos_in, pos.data(), 0, n_tokens * sizeof(int));

    // PLAN #73: populate causal mask for ggml_flash_attn_ext. Only built
    // for prefill (n_tokens > 1); decode steps pass nullptr mask.
    if (n_tokens > 1) {
        const int L = offset + n_tokens;
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask((size_t)n_tokens * L, zero);
        for (int q = 0; q < n_tokens; q++) {
            // q-th query at absolute position offset+q. Mask out keys
            // with absolute index > offset+q (i.e. future positions).
            for (int k = offset + q + 1; k < L; k++)
                mask[(size_t)q * L + k] = neg_inf;
        }
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "sa_mask");
        if (mask_in)
            ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "canary: decoder compute failed\n");
        return {};
    }

    // Capture cross-attention for DTW (only on single-token generation steps).
    if (ctx->collect_attn && n_tokens == 1) {
        ggml_tensor* ca_w_t = ggml_graph_get_tensor(gf, "ca_attn_w");
        if (ca_w_t) {
            const int T_enc = (int)ca_w_t->ne[0];
            const int n_heads = (int)ca_w_t->ne[2];
            ctx->attn_T_enc = T_enc;
            ctx->attn_n_heads = n_heads;
            std::vector<float> w((size_t)T_enc * n_heads);
            ggml_backend_tensor_get(ca_w_t, w.data(), 0, w.size() * sizeof(float));
            ctx->step_attn.push_back(std::move(w));
        }
    }

    ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    const int V = (int)logits->ne[0];
    std::vector<float> all((size_t)V * n_tokens);
    ggml_backend_tensor_get(logits, all.data(), 0, all.size() * sizeof(float));

    // Return logits for the last position only
    return std::vector<float>(all.begin() + (size_t)(n_tokens - 1) * V, all.end());
}

// ===========================================================================
// Greedy decode: build prompt, run encoder, run decoder steps until EOS
// ===========================================================================

static std::vector<int> canary_build_prompt(canary_context* ctx, const std::string& src, const std::string& tgt,
                                            bool punctuation) {
    auto tok = [&](const char* s) { return canary_str_to_token(ctx, s); };
    std::string src_tok = "<|" + src + "|>";
    std::string tgt_tok = "<|" + tgt + "|>";

    // NeMo CanaryBPETokenizer (shared with Cohere) prompt format:
    //   <|startofcontext|>
    //   <|startoftranscript|>
    //   <|emo:undefined|>
    //   <|src_lang|>
    //   <|target_lang|>
    //   <|pnc|>  or  <|nopnc|>
    //   <|notimestamp|>
    //   <|nodiarize|>
    std::vector<int> p;
    p.push_back(tok("<|startofcontext|>"));
    p.push_back(tok("<|startoftranscript|>"));
    p.push_back(tok("<|emo:undefined|>"));
    p.push_back(canary_str_to_token(ctx, src_tok.c_str()));
    p.push_back(canary_str_to_token(ctx, tgt_tok.c_str()));
    p.push_back(tok(punctuation ? "<|pnc|>" : "<|nopnc|>"));
    p.push_back(tok("<|notimestamp|>"));
    p.push_back(tok("<|nodiarize|>"));

    static const char* names[] = {"<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
                                  src_tok.c_str(),      tgt_tok.c_str(),         punctuation ? "<|pnc|>" : "<|nopnc|>",
                                  "<|notimestamp|>",    "<|nodiarize|>"};
    for (size_t i = 0; i < p.size(); i++) {
        if (p[i] < 0) {
            fprintf(stderr,
                    "canary: prompt contains an unknown token: '%s' "
                    "(vocab_size=%d, src='%s', tgt='%s', punc=%d)\n",
                    names[i], (int)ctx->vocab.id_to_token.size(), src_tok.c_str(), tgt_tok.c_str(), (int)punctuation);
            return {};
        }
    }
    return p;
}

static std::string spiece_to_text(const std::string& piece) {
    if (piece.empty())
        return "";
    if (piece.size() >= 2 && piece[0] == '<' && piece.back() == '>')
        return "";
    if (piece.size() >= 3 && (unsigned char)piece[0] == 0xE2 && (unsigned char)piece[1] == 0x96 &&
        (unsigned char)piece[2] == 0x81) {
        return std::string(" ") + piece.substr(3);
    }
    return piece;
}

// ===========================================================================
// BatchNorm folding (load-time, once) — same trick as parakeet/cohere
// ===========================================================================

static void canary_fold_batchnorm(canary_model& model) {
    const int d = (int)model.hparams.d_model;
    const int K = (int)model.hparams.conv_kernel;
    const float eps = 1e-5f;

    for (uint32_t il = 0; il < model.hparams.enc_n_layers; il++) {
        auto& e = model.enc[il];
        if (!e.conv_dw_w || !e.conv_dw_b || !e.conv_bn_w || !e.conv_bn_b || !e.conv_bn_rm || !e.conv_bn_rv)
            continue;

        std::vector<float> bn_mean(d), bn_var(d), bn_w(d), bn_b(d), dw_b(d);
        ggml_backend_tensor_get(e.conv_bn_rm, bn_mean.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_rv, bn_var.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_w, bn_w.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_b, bn_b.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));

        std::vector<float> s(d);
        for (int c = 0; c < d; c++)
            s[c] = bn_w[c] / sqrtf(bn_var[c] + eps);

        std::vector<ggml_fp16_t> w_f16((size_t)K * d);
        ggml_backend_tensor_get(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
        std::vector<float> w_f32((size_t)K * d);
        for (size_t i = 0; i < w_f16.size(); i++)
            w_f32[i] = ggml_fp16_to_fp32(w_f16[i]);
        for (int c = 0; c < d; c++)
            for (int ki = 0; ki < K; ki++)
                w_f32[ki + c * K] *= s[c];
        for (size_t i = 0; i < w_f16.size(); i++)
            w_f16[i] = ggml_fp32_to_fp16(w_f32[i]);
        ggml_backend_tensor_set(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));

        // Fold into existing dw_b: b'[c] = (dw_b[c] - mean[c]) * s[c] + bn_b[c]
        for (int c = 0; c < d; c++)
            dw_b[c] = (dw_b[c] - bn_mean[c]) * s[c] + bn_b[c];
        ggml_backend_tensor_set(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
    }

    fprintf(stderr, "canary: BN folded into conv_dw weights for %u layers\n", model.hparams.enc_n_layers);
}

// ===========================================================================
// Backend selection
// ===========================================================================

static ggml_backend_t pick_backend() {
    // ggml_backend_init_best() tries all compiled backends in priority
    // order (CUDA > Metal > Vulkan > CPU) and returns the first one
    // that initialises. This replaces the old Metal/CUDA-specific
    // #ifdef chain and adds Vulkan support for free.
    ggml_backend_t b = ggml_backend_init_best();
    return b ? b : ggml_backend_cpu_init();
}

static ggml_backend_t pick_backend(bool use_gpu) {
    return use_gpu ? pick_backend() : ggml_backend_cpu_init();
}

// ===========================================================================
// Public C API
// ===========================================================================

// ---- Stage-level entry points for crispasr-diff ----

extern "C" float* canary_compute_mel(struct canary_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                     int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    int T_mel = 0;
    auto mel = canary_compute_mel_impl(ctx, samples, n_samples, T_mel);
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

extern "C" float* canary_run_encoder(struct canary_context* ctx, const float* mel, int n_mels, int T_mel,
                                     int* out_T_enc, int* out_d_model) {
    if (!ctx || !mel || T_mel <= 0)
        return nullptr;
    int T_enc = 0;
    auto enc = canary_encode_mel(ctx, mel, n_mels, T_mel, &T_enc);
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

extern "C" int canary_run_encoder_staged(struct canary_context* ctx, const float* mel, int n_mels, int T_mel,
                                         canary_stage_cb cb, void* userdata) {
    if (!ctx || !mel || T_mel <= 0 || !cb)
        return -1;
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "canary: mel feature mismatch (%d vs %d)\n", n_mels, (int)ctx->model.hparams.n_mels);
        return -1;
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 24576, false, false);
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(ggml_tensor_overhead() * 24576 + ggml_graph_overhead_custom(24576, false));
    }

    ggml_cgraph* gf = canary_build_graph_encoder_staged(ctx, T_mel);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "canary: failed to alloc staged encoder graph\n");
        return -1;
    }

    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2;
    auto pe = core_conformer::make_pos_enc((int)ctx->model.hparams.d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "canary: staged encoder compute failed\n");
        return -1;
    }

    const int d = (int)ctx->model.hparams.d_model;

    // Retrieve and deliver each named snapshot in order.
    // deliver() assumes ne[0]=d_model (standard encoder output format).
    auto deliver = [&](const char* name) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t)
            return;
        const int t_steps = (int)t->ne[1];
        std::vector<float> buf((size_t)d * t_steps);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        cb(name, buf.data(), t_steps, d, userdata);
    };
    // deliver_dyn() uses the tensor's own ne[0] as the feature dim.
    // Used for intermediate conv snaps where the feature count differs.
    auto deliver_dyn = [&](const char* name) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t)
            return;
        const int feat = (int)t->ne[0];
        const int t_steps = (int)t->ne[1];
        std::vector<float> buf((size_t)feat * t_steps);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        cb(name, buf.data(), t_steps, feat, userdata);
    };

    deliver_dyn("pre_enc_c0");
    deliver_dyn("pre_enc_c2");
    deliver_dyn("pre_enc_c3");
    deliver_dyn("pre_enc_c5");
    deliver_dyn("pre_enc_c6");
    deliver("pre_enc_out");

    const int n_layers = (int)ctx->model.hparams.enc_n_layers;
    char lbuf[32];
    for (int il = 0; il < n_layers; il++) {
        snprintf(lbuf, sizeof(lbuf), "enc_L%02d", il);
        deliver(lbuf);
    }

    deliver("enc_out");

    return 0;
}

extern "C" struct canary_context_params canary_context_default_params(void) {
    canary_context_params p = {};
    p.n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    p.use_flash = false;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

extern "C" struct canary_context* canary_init_from_file(const char* path_model, struct canary_context_params params) {
    auto* ctx = new canary_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = pick_backend(params.use_gpu);
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    if (!canary_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        canary_free(ctx);
        return nullptr;
    }
    canary_fold_batchnorm(ctx->model);
    return ctx;
}

extern "C" void canary_free(struct canary_context* ctx) {
    if (!ctx)
        return;
    if (ctx->cross_buf)
        ggml_backend_buffer_free(ctx->cross_buf);
    if (ctx->cross_ctx)
        ggml_free(ctx->cross_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void canary_result_free(struct canary_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->tokens);
    free(r->words);
    free(r);
}

extern "C" void canary_set_temperature(struct canary_context* ctx, float temperature, uint64_t seed) {
    if (!ctx)
        return;
    ctx->decode_temperature = temperature;
    ctx->decode_seed = seed;
}

extern "C" void canary_set_beam_size(struct canary_context* ctx, int n) {
    if (!ctx)
        return;
    ctx->beam_size = n > 0 ? n : 1;
}

extern "C" int canary_n_vocab(struct canary_context* ctx) {
    return (int)ctx->model.hparams.vocab_size;
}
extern "C" int canary_n_mels(struct canary_context* ctx) {
    return (int)ctx->model.hparams.n_mels;
}
extern "C" int canary_sample_rate(struct canary_context* ctx) {
    return (int)ctx->model.hparams.sample_rate;
}
extern "C" int canary_frame_dur_cs(struct canary_context* ctx) {
    return (int)ctx->model.hparams.frame_dur_cs;
}

extern "C" const char* canary_token_to_str(struct canary_context* ctx, int id) {
    if (id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[id].c_str();
}

extern "C" int canary_str_to_token(struct canary_context* ctx, const char* str) {
    auto it = ctx->vocab.token_to_id.find(str);
    return it != ctx->vocab.token_to_id.end() ? it->second : -1;
}

extern "C" int canary_test_load(struct canary_context* ctx) {
    fprintf(stderr,
            "canary: load test OK\n"
            "  vocab_size  = %d\n"
            "  d_model     = %d\n"
            "  enc_layers  = %d\n"
            "  dec_layers  = %d\n"
            "  n_heads     = %d\n"
            "  head_dim    = %d\n"
            "  ff_dim      = %d\n"
            "  max_dec_ctx = %d\n"
            "  n_mels      = %d\n"
            "  sample_rate = %d\n"
            "  frame_dur_cs= %d\n",
            (int)ctx->model.hparams.vocab_size, (int)ctx->model.hparams.d_model, (int)ctx->model.hparams.enc_n_layers,
            (int)ctx->model.hparams.dec_n_layers, (int)ctx->model.hparams.n_heads, (int)ctx->model.hparams.head_dim,
            (int)ctx->model.hparams.ff_dim, (int)ctx->model.hparams.max_dec_ctx, (int)ctx->model.hparams.n_mels,
            (int)ctx->model.hparams.sample_rate, (int)ctx->model.hparams.frame_dur_cs);

    // Confirm a few special tokens resolve
    const char* specials[] = {
        "<|startoftranscript|>", "<|endoftext|>", "<|en|>", "<|de|>", "<|fr|>", "<|es|>", "<|nopnc|>",
        "<|notimestamp|>",       "<|nodiarize|>",
    };
    for (const char* s : specials) {
        int id = canary_str_to_token(ctx, s);
        fprintf(stderr, "  token %-22s = %d\n", s, id);
    }
    return 0;
}

extern "C" int canary_test_encoder(struct canary_context* ctx, int T_mel) {
    int n_mels = (int)ctx->model.hparams.n_mels;
    std::vector<float> mel((size_t)n_mels * T_mel, 0.0f);
    int T_enc = 0;
    auto out = canary_encode_mel(ctx, mel.data(), n_mels, T_mel, &T_enc);
    if (out.empty())
        return -1;
    fprintf(stderr, "canary: encoder OK — T_mel=%d → T_enc=%d  d=%d  out[0..3]=%g %g %g %g\n", T_mel, T_enc,
            (int)ctx->model.hparams.d_model, (double)out[0], (double)out[1], (double)out[2], (double)out[3]);
    return T_enc;
}

// PLAN #114 P3 second half — post-encode pipeline shared by
// canary_transcribe_ex (single-pass) and canary_transcribe_streamed
// (parakeet-style chunked-encode + concat for long audio).
static canary_result* canary_finish_from_encoder(canary_context* ctx, const float* enc_data, int T_enc,
                                                 const char* source_lang, const char* target_lang, bool punctuation,
                                                 int64_t t_offset_cs);

extern "C" struct canary_result* canary_transcribe_ex(struct canary_context* ctx, const float* samples, int n_samples,
                                                      const char* source_lang, const char* target_lang,
                                                      bool punctuation, int64_t t_offset_cs) {
    if (!ctx || !samples || n_samples <= 0 || !source_lang || !target_lang)
        return nullptr;

    // 1. Mel
    int T_mel = 0;
    auto mel = canary_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;

    // 2. Encoder
    int T_enc = 0;
    auto enc = canary_encode_mel(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, &T_enc);
    if (enc.empty())
        return nullptr;

    return canary_finish_from_encoder(ctx, enc.data(), T_enc, source_lang, target_lang, punctuation, t_offset_cs);
}

// PLAN #114 P3 second half — parakeet-pattern long-audio port. Computes
// mel for the FULL audio (so PerFeatureZ uses global statistics; this is
// the NeMo convention and matches what canary_transcribe_ex does on short
// audio), then encodes in overlapping mel chunks and concatenates the
// encoder outputs with `overlap_enc` trimming. Runs the existing AED
// decode + cross-attention K/V over the concat. Works for the 4 trained
// languages (en/de/fr/es) — the whitelist in
// crispasr_backend_canary.cpp::transcribe() refuses other langs before
// either code path.
extern "C" struct canary_result* canary_transcribe_streamed(struct canary_context* ctx, const float* samples,
                                                            int n_samples, const char* source_lang,
                                                            const char* target_lang, bool punctuation,
                                                            int64_t t_offset_cs, int chunk_seconds,
                                                            int overlap_seconds) {
    if (!ctx || !samples || n_samples <= 0 || !source_lang || !target_lang)
        return nullptr;
    if (chunk_seconds <= 0)
        chunk_seconds = 8;
    if (overlap_seconds < 0)
        overlap_seconds = 2;
    if (overlap_seconds >= chunk_seconds)
        overlap_seconds = chunk_seconds / 4;

    const auto& hp = ctx->model.hparams;
    const int n_mels = (int)hp.n_mels;
    const int hop = (int)hp.hop_length;
    const int SR = (int)hp.sample_rate;

    // ---- Pass 1: mel for the full audio (global PerFeatureZ) ----
    int T_mel = 0;
    auto mel_full = canary_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel_full.empty() || T_mel <= 0)
        return nullptr;

    // ---- Pass 2: per-chunk encode + per-chunk AED decode (NeMo
    // FrameBatchMultiTaskAED analogon). Each chunk re-injects the
    // <lang> / <task> / <pnc> prompt prefix, so the AED decoder
    // doesn't treat the chunk boundary as <eos>. Per-chunk results
    // are concatenated text-wise; token/word timestamps land in the
    // global time window via per-chunk t_offset arithmetic.
    const int chunk_mel_frames = chunk_seconds * SR / hop;
    const int overlap_mel_frames = overlap_seconds * SR / hop;
    const int shift_mel_frames = chunk_mel_frames - overlap_mel_frames;

    std::string full_text;
    std::vector<canary_token_data> all_tokens;
    std::vector<canary_word_data> all_words;
    int chunks_processed = 0;

    // LCS dedup window: bound how far into the previous chunk we look for
    // boundary duplication. Conservative — overlap_seconds at ~3 tok/s
    // (canary's average emission rate on en/de/fr/es) gives ~6 tokens for
    // a 2 s overlap; cap at 30 to leave headroom for slower regimes.
    const int delay_tokens = std::min(30, std::max(8, overlap_seconds * 5));

    for (int mel_offset = 0; mel_offset < T_mel; mel_offset += shift_mel_frames) {
        const int chunk_end = std::min(T_mel, mel_offset + chunk_mel_frames);
        const int chunk_T = chunk_end - mel_offset;
        if (chunk_T <= 0)
            break;
        int T_enc = 0;
        auto enc = canary_encode_mel(ctx, mel_full.data() + (size_t)mel_offset * n_mels, n_mels, chunk_T, &T_enc);
        if (enc.empty() || T_enc <= 0)
            continue;

        const int64_t chunk_t_offset_cs = t_offset_cs + (int64_t)mel_offset * hop * 100 / SR;
        canary_result* part = canary_finish_from_encoder(ctx, enc.data(), T_enc, source_lang, target_lang, punctuation,
                                                         chunk_t_offset_cs);
        if (!part)
            continue;

        // PLAN #114 P3 polish: boundary-overlap dedup via LCS-merge across
        // adjacent chunks. NeMo's `streaming_utils.longest_common_subsequence_merge`
        // (the same primitive `core/crispasr_lcs::lcs_dedup_prefix_count`
        // wraps) finds the LCS between the tail of accumulated tokens and
        // the head of the current chunk's tokens. If the LCS length >=
        // kMinMergeSubsequenceLen (default 3), the matched prefix of the
        // current chunk is dropped before accumulation.
        int n_skip = 0;
        if (chunks_processed > 0 && part->n_tokens > 0 && !all_tokens.empty()) {
            const int tail_size = std::min(delay_tokens, (int)all_tokens.size());
            // PLAN #114 P3 polish — case-insensitive LCS. The AED can
            // re-emit the same audio with different capitalization at
            // chunk boundaries (e.g. "world's say for" in chunk 1 vs
            // "World's Save for" in chunk 2 — different raw ids but
            // semantically duplicate). Look up each id via
            // vocab.id_to_canonical_lc[id] (smallest id whose
            // lowercase-ASCII text matches) before running LCS. Indices
            // line up with the raw token vectors, so the returned
            // slice_count is still the right number of leading raw
            // tokens to drop.
            auto canon = [&](int32_t id) -> int32_t {
                if (id < 0 || id >= (int32_t)ctx->vocab.id_to_canonical_lc.size())
                    return id;
                return ctx->vocab.id_to_canonical_lc[(size_t)id];
            };
            std::vector<int32_t> prev_tail;
            prev_tail.reserve((size_t)tail_size);
            for (int i = (int)all_tokens.size() - tail_size; i < (int)all_tokens.size(); i++)
                prev_tail.push_back(canon(all_tokens[(size_t)i].id));
            std::vector<int32_t> curr_ids;
            curr_ids.reserve((size_t)part->n_tokens);
            for (int i = 0; i < part->n_tokens; i++)
                curr_ids.push_back(canon(part->tokens[i].id));
            n_skip = crispasr_lcs::lcs_dedup_prefix_count(prev_tail, curr_ids);
        }

        // PLAN #114 P3 polish — word-snap heuristic. The LCS operates on
        // token ids and may drop a prefix that ends mid-word, because
        // BPE-split points inside a duplicated word region don't always
        // align between chunks (chunk 1 emits `[▁Geschirrt, uches]` for
        // "Geschirrtuches"; chunk 2 emits `[▁tuch, ▁umfassen, ...]` for
        // the same audio segment — only `▁umfassen` matches the LCS, so
        // we drop 1 token and leave `▁tuch` as a stray word-fragment in
        // the output). Extend the drop forward until the next token
        // whose text starts with a space (sentencepiece convention:
        // `▁X` decodes to ` X`, so a leading space marks a word start).
        // Trades a few extra tokens for a clean prefix; bounded by
        // n_tokens so we don't drop the whole chunk on a pathological
        // mid-word run.
        if (n_skip > 0 && n_skip < part->n_tokens) {
            while (n_skip < part->n_tokens) {
                const char* t = part->tokens[n_skip].text;
                if (!t || !t[0])
                    break;
                if (t[0] == ' ')
                    break;
                n_skip++;
            }
        }

        // Rebuild text for this chunk from surviving tokens (so the
        // dedupe matches the token vector exactly; trusting part->text
        // would leak the dropped prefix as a string).
        std::string part_text;
        for (int i = n_skip; i < part->n_tokens; i++)
            part_text += part->tokens[i].text;
        if (!part_text.empty() && part_text[0] == ' ')
            part_text.erase(0, 1);

        if (!part_text.empty()) {
            // PLAN #114 P3 polish — splice-punctuation cleanup. After LCS
            // dedup the chunk boundary can land between a mid-sentence
            // punctuation in the previous chunk (`,` `;` `:`) and a
            // sentence-end punctuation in the surviving prefix of the new
            // chunk (`.` `?` `!`), producing e.g. "for you, . Ask...".
            // The chunk-1 comma was the model's best guess for the
            // continuation point; chunk-2 produced a period because the
            // LCS-dropped middle ended that sentence. Collapse the
            // mid-sentence punctuation in favour of the sentence-end
            // punctuation that survived LCS.
            bool attach_directly = false;
            if (!full_text.empty()) {
                const char last_punct = full_text.back();
                size_t first_nws = 0;
                while (first_nws < part_text.size() && (part_text[first_nws] == ' ' || part_text[first_nws] == '\t'))
                    first_nws++;
                if (first_nws < part_text.size()) {
                    const char first_punct = part_text[first_nws];
                    if ((last_punct == ',' || last_punct == ';' || last_punct == ':') &&
                        (first_punct == '.' || first_punct == '?' || first_punct == '!')) {
                        full_text.pop_back();
                        while (!full_text.empty() && full_text.back() == ' ')
                            full_text.pop_back();
                        // Strip leading whitespace from part_text so the
                        // surviving punctuation attaches directly:
                        //   "for you" + ". Ask..."  →  "for you. Ask..."
                        // (instead of "for you . Ask..." with a stray
                        // space before the period).
                        part_text.erase(0, first_nws);
                        attach_directly = true;
                    }
                }
            }
            if (!attach_directly && !full_text.empty() && full_text.back() != ' ' && part_text[0] != ' ')
                full_text += ' ';
            full_text += part_text;
        }
        for (int i = n_skip; i < part->n_tokens; i++)
            all_tokens.push_back(part->tokens[i]);

        // Words: drop leading ones whose t1 is at or before the last
        // surviving token's t0 (best-effort — words and tokens don't have
        // a strict 1:1 mapping but adjacent boundary-overlap words should
        // share timing with the dropped tokens).
        if (n_skip > 0 && part->n_tokens > 0) {
            const int64_t cutoff_t0 =
                (n_skip < part->n_tokens) ? part->tokens[(size_t)n_skip].t0 : part->tokens[(size_t)n_skip - 1].t1;
            for (int i = 0; i < part->n_words; i++) {
                if (part->words[i].t1 <= cutoff_t0)
                    continue;
                all_words.push_back(part->words[i]);
            }
        } else {
            for (int i = 0; i < part->n_words; i++)
                all_words.push_back(part->words[i]);
        }

        canary_result_free(part);
        chunks_processed++;
    }

    if (chunks_processed == 0 && all_tokens.empty() && full_text.empty())
        return nullptr;

    canary_result* r = (canary_result*)calloc(1, sizeof(canary_result));
    if (!r)
        return nullptr;
    r->text = strdup(full_text.c_str());
    if (!r->text) {
        canary_result_free(r);
        return nullptr;
    }
    r->n_tokens = (int)all_tokens.size();
    r->tokens = (canary_token_data*)calloc(r->n_tokens > 0 ? (size_t)r->n_tokens : 1, sizeof(canary_token_data));
    if (!r->tokens) {
        canary_result_free(r);
        return nullptr;
    }
    for (int i = 0; i < r->n_tokens; i++)
        r->tokens[i] = all_tokens[(size_t)i];

    r->n_words = (int)all_words.size();
    r->words = (canary_word_data*)calloc(r->n_words > 0 ? (size_t)r->n_words : 1, sizeof(canary_word_data));
    if (!r->words) {
        canary_result_free(r);
        return nullptr;
    }
    for (int i = 0; i < r->n_words; i++)
        r->words[i] = all_words[(size_t)i];

    return r;
}

static canary_result* canary_finish_from_encoder(canary_context* ctx, const float* enc_data, int T_enc,
                                                 const char* source_lang, const char* target_lang, bool punctuation,
                                                 int64_t t_offset_cs) {
    // 3. Pre-compute cross-attention K/V
    canary_build_cross_kv(ctx, enc_data, T_enc);

    // 4. Build prompt
    std::vector<int> prompt = canary_build_prompt(ctx, source_lang, target_lang, punctuation);
    if (prompt.empty())
        return nullptr;

    // Reset DTW state and enable cross-attn capture for the upcoming greedy loop.
    // Each per-step decode call will append one entry to ctx->step_attn (one
    // T_enc * n_heads vector per emitted token).
    ctx->collect_attn = true;
    ctx->step_attn.clear();

    // 5. Greedy decode
    const int eos = canary_str_to_token(ctx, "<|endoftext|>");
    const int max_steps = 256;
    const int max_ctx = (int)ctx->model.hparams.max_dec_ctx;

    std::vector<int> generated = prompt;
    std::vector<float> emitted_p; // softmax prob per generated non-prompt token

    const bool sampling = ctx->decode_temperature > 0.0f;
    std::mt19937_64 rng(ctx->decode_seed != 0 ? ctx->decode_seed : (uint64_t)std::random_device{}());
    int offset = 0;

    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "canary: prompt =");
        for (int t : prompt)
            fprintf(stderr, " %d(%s)", t, ctx->vocab.id_to_token[t].c_str());
        fprintf(stderr, "\n");
    }


    // First call: feed the entire prompt at once
    auto logits = canary_decode_step(ctx, prompt.data(), (int)prompt.size(), 0);
    if (logits.empty())
        return nullptr;
    offset = (int)prompt.size();

    // §90 beam search — run_with_probs_branched when beam_size > 1.
    // Cross-attention KV (cross_k/v) is shared across beams; only self-attention
    // KV (kv_k/kv_v) is snapshotted per beam.
    if (ctx->beam_size > 1) {
        struct canary_kv_snap {
            std::vector<uint8_t> k_data;
            std::vector<uint8_t> v_data;
        };

        auto save_fn = [](canary_context* c) -> canary_kv_snap* {
            auto* s = new canary_kv_snap();
            size_t kb = ggml_nbytes(c->kv_k);
            size_t vb = ggml_nbytes(c->kv_v);
            s->k_data.resize(kb);
            s->v_data.resize(vb);
            ggml_backend_tensor_get(c->kv_k, s->k_data.data(), 0, kb);
            ggml_backend_tensor_get(c->kv_v, s->v_data.data(), 0, vb);
            return s;
        };

        auto restore_fn = [](canary_context* c, canary_kv_snap* s) {
            ggml_backend_tensor_set(c->kv_k, s->k_data.data(), 0, s->k_data.size());
            ggml_backend_tensor_set(c->kv_v, s->v_data.data(), 0, s->v_data.size());
        };

        auto snap_free_fn = [](canary_kv_snap* s) { delete s; };

        auto step_fn = [](canary_context* c, int32_t tok, int n_past) -> float* {
            auto lg = canary_decode_step(c, &tok, 1, n_past);
            if (lg.empty())
                return nullptr;
            float* out = (float*)std::malloc(lg.size() * sizeof(float));
            std::memcpy(out, lg.data(), lg.size() * sizeof(float));
            return out;
        };

        const int vocab = (int)logits.size();
        core_beam_decode::Config bcfg;
        bcfg.max_new_tokens = max_steps;
        bcfg.eos_id = eos;
        bcfg.vocab_size = vocab;
        bcfg.beam_size = ctx->beam_size;
        bcfg.prompt_len = (int)prompt.size();

        auto br = core_beam_decode::run_with_probs_branched(ctx, logits.data(), save_fn, restore_fn, snap_free_fn,
                                                            step_fn, bcfg);

        // Build result from beam output (skip EOS if present at the end)
        int n_beam = (int)br.tokens.size();
        if (n_beam > 0 && br.tokens.back() == eos)
            n_beam--;

        auto* r = (canary_result*)calloc(1, sizeof(canary_result));
        if (!r)
            return nullptr;

        r->n_tokens = n_beam;
        r->tokens = (canary_token_data*)calloc((size_t)n_beam, sizeof(canary_token_data));
        std::string full_text;
        for (int i = 0; i < n_beam; i++) {
            r->tokens[i].id = br.tokens[i];
            r->tokens[i].p = br.probs[i];
            r->tokens[i].t0 = 0;
            r->tokens[i].t1 = 0;
            const char* txt = canary_token_to_str(ctx, br.tokens[i]);
            if (txt) {
                std::string vis = spiece_to_text(txt);
                snprintf(r->tokens[i].text, sizeof(r->tokens[i].text), "%s", vis.c_str());
                full_text += vis;
            }
        }
        r->text = strdup(full_text.c_str());
        r->n_words = 0;
        r->words = nullptr;
        return r;
    }

    // PLAN #114 P3 polish — degenerate-loop guard. The AED has no
    // repetition penalty; on out-of-distribution audio (or a chunk
    // boundary where the encoder embeddings collapse) the decoder can
    // lock on a short cycle and emit ~100+ copies before <eos>. Funasr
    // (PLAN #125 P1) had a single-id repeat (`!`); canary's BPE often
    // emits a two-token cycle like `▁yeah` + `,` alternating, so the
    // funasr-style consecutive-id guard misses it. Detect by
    // small-vocabulary window: if the last kWindow generated tokens
    // contain ≤ kMaxDistinct distinct ids, the decoder is in a loop.
    // Window/threshold picked so normal speech (~25-30 unique ids per
    // 40-token window) stays well above the trigger and degenerate
    // 1/2/3-cycles all fire.
    constexpr int kWindow = 40;
    constexpr int kMaxDistinct = 3;

    for (int step = 0; step < max_steps && offset < max_ctx - 1; step++) {
        // Argmax (default) or temperature sample
        int best = 0;
        float best_lp = logits[0];
        for (int v = 1; v < (int)logits.size(); v++) {
            if (logits[v] > best_lp) {
                best_lp = logits[v];
                best = v;
            }
        }
        if (sampling) {
            const int V = (int)logits.size();
            const float inv_t = 1.0f / ctx->decode_temperature;
            std::vector<double> pr((size_t)V);
            double sum = 0.0;
            for (int v = 0; v < V; v++) {
                const double e = std::exp((double)((logits[v] - best_lp) * inv_t));
                pr[(size_t)v] = e;
                sum += e;
            }
            if (sum > 0.0) {
                std::uniform_real_distribution<double> unif(0.0, sum);
                const double rr = unif(rng);
                double acc = 0.0;
                for (int v = 0; v < V; v++) {
                    acc += pr[(size_t)v];
                    if (rr <= acc) {
                        best = v;
                        break;
                    }
                }
                best_lp = logits[best];
            }
        }
        if (ctx->params.verbosity >= 2 && step < 30) {
            const char* pc =
                (best >= 0 && best < (int)ctx->vocab.id_to_token.size()) ? ctx->vocab.id_to_token[best].c_str() : "?";
            fprintf(stderr, "  step %3d  tok=%5d  '%s'  logp=%.3f\n", step, best, pc, (double)best_lp);
        }
        if (best == eos)
            break;

        // Degenerate-loop guard: count distinct ids in the last
        // kWindow generated tokens (including this one). Cheap O(W)
        // per step on a fixed-size window — for kWindow=40 that's
        // ~40 comparisons per step, negligible next to the decoder
        // forward.
        if ((int)generated.size() >= (int)prompt.size() + kWindow) {
            const int start = (int)generated.size() - kWindow + 1;
            std::set<int> distinct;
            for (int i = start; i < (int)generated.size(); i++)
                distinct.insert(generated[(size_t)i]);
            distinct.insert(best);
            if ((int)distinct.size() <= kMaxDistinct) {
                if (ctx->params.verbosity >= 1) {
                    fprintf(stderr,
                            "canary: greedy decode degenerated (%d distinct ids in last %d tokens). "
                            "Aborting at step %d.\n",
                            (int)distinct.size(), kWindow, step);
                }
                break;
            }
        }

        // Softmax probability of the picked token. Numerically stable:
        // subtract the max log-prob before exponentiating.
        float best_p = 1.0f;
        {
            double sum = 0.0;
            for (int v = 0; v < (int)logits.size(); v++) {
                sum += std::exp((double)(logits[v] - best_lp));
            }
            best_p = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
        }

        generated.push_back(best);
        emitted_p.push_back(best_p);

        // Decode next step with the new token
        int tok = best;
        logits = canary_decode_step(ctx, &tok, 1, offset);
        if (logits.empty())
            break;
        offset++;
    }

    // 6. Build result (skip the prompt prefix)
    int n_emitted = (int)generated.size() - (int)prompt.size();
    auto* r = (canary_result*)calloc(1, sizeof(canary_result));
    if (!r)
        return nullptr;
    r->n_tokens = n_emitted;
    r->tokens = (canary_token_data*)calloc(n_emitted > 0 ? n_emitted : 1, sizeof(canary_token_data));
    if (!r->tokens) {
        canary_result_free(r);
        return nullptr;
    }

    // ----- Per-token timestamps via cross-attention DTW -----
    //
    // For each emitted token we have ctx->step_attn[i] = cross-attn weights from
    // the LAST decoder layer, layout [T_enc * n_heads]. We:
    //   1. Average across heads → A[n_emitted × T_enc]
    //   2. Subtract per-frame mean (column normalisation) so globally "hot"
    //      frames don't dominate
    //   3. Run prefix-max DTW: D[i][j] = A[i][j] + max_{k≤j} D[i-1][k]
    //   4. Traceback from argmax of last row → path[i] = best frame for token i
    //   5. t0/t1 from path[i] × frame_dur_cs
    //
    // Same approach as cohere-main, gives ~360 ms MAE on word boundaries.
    // (Approach copied from src/cohere.cpp; see that file for the discussion of
    // hot-frame collapse and the column-normalisation fix.)
    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;
    const int64_t total_cs = (int64_t)T_enc * frame_dur_cs;
    const int64_t seg_end_cs = t_offset_cs + total_cs;

    const bool have_attn =
        n_emitted > 0 && ctx->step_attn.size() == (size_t)n_emitted && ctx->attn_T_enc > 0 && ctx->attn_n_heads > 0;

    std::vector<int> path;
    if (have_attn) {
        const int T_e = ctx->attn_T_enc;
        const int n_h = ctx->attn_n_heads;

        // Step 1 + temporal offset correction (same trick as cohere): use
        // step_attn[i-1] for token i because the attention is collected while
        // the decoder processes generated[i] as input to predict generated[i+1].
        std::vector<float> A((size_t)n_emitted * T_e, 0.0f);
        for (int i = 0; i < n_emitted; i++) {
            int src = (i > 0) ? i - 1 : 0;
            const float* w = ctx->step_attn[src].data();
            float* Ai = A.data() + (size_t)i * T_e;
            for (int h = 0; h < n_h; h++)
                for (int t = 0; t < T_e; t++)
                    Ai[t] += w[h * T_e + t];
            for (int t = 0; t < T_e; t++)
                Ai[t] /= n_h;
        }

        // Step 2: column normalisation
        {
            std::vector<float> col_mean(T_e, 0.f);
            for (int i = 0; i < n_emitted; i++)
                for (int t = 0; t < T_e; t++)
                    col_mean[t] += A[(size_t)i * T_e + t];
            for (int t = 0; t < T_e; t++)
                col_mean[t] /= n_emitted;
            for (int i = 0; i < n_emitted; i++)
                for (int t = 0; t < T_e; t++)
                    A[(size_t)i * T_e + t] -= col_mean[t];
        }

        // Step 3: forward DP with prefix-max predecessors
        std::vector<float> D((size_t)n_emitted * T_e);
        std::vector<int> P((size_t)n_emitted * T_e);
        for (int j = 0; j < T_e; j++)
            D[j] = A[j];
        for (int i = 1; i < n_emitted; i++) {
            const float* Di_1 = D.data() + (size_t)(i - 1) * T_e;
            const float* Ai = A.data() + (size_t)i * T_e;
            float* Di = D.data() + (size_t)i * T_e;
            int* Pi = P.data() + (size_t)i * T_e;
            float pm_val = Di_1[0];
            int pm_idx = 0;
            for (int j = 0; j < T_e; j++) {
                if (Di_1[j] > pm_val) {
                    pm_val = Di_1[j];
                    pm_idx = j;
                }
                Di[j] = Ai[j] + pm_val;
                Pi[j] = pm_idx;
            }
        }

        // Step 4: traceback
        path.resize(n_emitted);
        const float* Dlast = D.data() + (size_t)(n_emitted - 1) * T_e;
        path[n_emitted - 1] = (int)(std::max_element(Dlast, Dlast + T_e) - Dlast);
        for (int i = n_emitted - 2; i >= 0; i--)
            path[i] = P[(size_t)(i + 1) * T_e + path[i + 1]];
    }

    std::string text;
    for (int i = 0; i < n_emitted; i++) {
        int tid = generated[(int)prompt.size() + i];
        const std::string& piece =
            (tid >= 0 && tid < (int)ctx->vocab.id_to_token.size()) ? ctx->vocab.id_to_token[tid] : std::string("");
        std::string vis = spiece_to_text(piece);

        canary_token_data& td = r->tokens[i];
        td.id = tid;
        td.p = (i < (int)emitted_p.size()) ? emitted_p[i] : -1.0f;
        if (have_attn) {
            int64_t a = t_offset_cs + (int64_t)path[i] * frame_dur_cs;
            int64_t b = (i + 1 < n_emitted) ? (t_offset_cs + (int64_t)path[i + 1] * frame_dur_cs) : seg_end_cs;
            // Guarantee at least one frame of duration so adjacent tokens that
            // collapse to the same DTW frame still have a non-zero span.
            if (b <= a)
                b = a + frame_dur_cs;
            if (b > seg_end_cs)
                b = seg_end_cs;
            td.t0 = a;
            td.t1 = b;
        } else {
            // Fallback: linear interpolation
            td.t0 = t_offset_cs + (int64_t)((double)i / std::max(1, n_emitted) * total_cs);
            td.t1 = t_offset_cs + (int64_t)((double)(i + 1) / std::max(1, n_emitted) * total_cs);
        }
        size_t n = std::min(vis.size(), sizeof(td.text) - 1);
        memcpy(td.text, vis.data(), n);
        td.text[n] = '\0';
        text += vis;
    }
    if (!text.empty() && text[0] == ' ')
        text = text.substr(1);
    r->text = strdup(text.c_str());
    if (!r->text) {
        canary_result_free(r);
        return nullptr;
    }

    // Disable attn collection so the next call starts fresh
    ctx->collect_attn = false;

    // Word grouping (same as parakeet's)
    {
        std::vector<canary_word_data> words;
        canary_word_data cur = {};
        bool have_cur = false;
        auto is_punct = [](const char* s) {
            if (!s || !*s)
                return false;
            for (const char* p = s; *p; p++) {
                unsigned char c = (unsigned char)*p;
                if (!(c == '.' || c == ',' || c == '?' || c == '!' || c == ';' || c == ':' || c == '\'' || c == '"' ||
                      c == '(' || c == ')' || c == '-'))
                    return false;
            }
            return true;
        };
        for (int i = 0; i < r->n_tokens; i++) {
            const auto& td = r->tokens[i];
            if (!td.text[0])
                continue;
            const bool is_word_start = (td.text[0] == ' ');
            const bool is_p = is_punct(td.text);
            if (is_word_start && !is_p && have_cur) {
                words.push_back(cur);
                cur = {};
                have_cur = false;
            }
            if (!have_cur) {
                cur.t0 = td.t0;
                have_cur = true;
            }
            cur.t1 = td.t1;
            const char* src = td.text + (is_word_start ? 1 : 0);
            size_t cl = strlen(cur.text);
            size_t cap = sizeof(cur.text) - cl - 1;
            size_t add = strlen(src);
            if (add > cap)
                add = cap;
            memcpy(cur.text + cl, src, add);
            cur.text[cl + add] = '\0';
        }
        if (have_cur)
            words.push_back(cur);
        r->n_words = (int)words.size();
        r->words = (canary_word_data*)calloc(r->n_words > 0 ? r->n_words : 1, sizeof(canary_word_data));
        if (!r->words) {
            canary_result_free(r);
            return nullptr;
        }
        for (int i = 0; i < r->n_words; i++)
            r->words[i] = words[i];
    }

    return r;
}

extern "C" char* canary_transcribe(struct canary_context* ctx, const float* samples, int n_samples,
                                   const char* source_lang, const char* target_lang, bool punctuation) {
    canary_result* r = canary_transcribe_ex(ctx, samples, n_samples, source_lang, target_lang, punctuation, 0);
    if (!r)
        return nullptr;
    char* out = strdup(r->text ? r->text : "");
    if (!out) {
        canary_result_free(r);
        return nullptr;
    }
    canary_result_free(r);
    return out;
}
