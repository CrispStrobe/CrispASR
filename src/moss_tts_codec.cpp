// moss_tts_codec.cpp — MOSS-Audio-Tokenizer transformer RVQ codec (decode).
//
// Faithful port of pwilkin/openmoss src/codec.cpp (decoder path) onto CrispASR's
// ggml conventions. See moss_tts_codec.h for the pipeline overview and
// docs/moss-tts/STUDY.md §6 for the full spec + gotchas.
//
// NOT yet parity-checked: per-stage cos >= 0.999 vs the ONNX/PyTorch tokenizer
// is a Phase 6 (Kaggle) gate. The graph math mirrors openmoss (validated there,
// envelope correlation 1.000) but must be diffed before trusting it.

#include "moss_tts_codec.h"

#include "core/gguf_loader.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace moss_tts_codec {

namespace {

constexpr float NORM_EPS = 1e-5f;
constexpr float ROPE_FREQ_BASE = 10000.0f;
constexpr int NUM_VQ = 32;
constexpr int CB_DIM = 8;
constexpr int RVQ_DIM = 512;
constexpr int OUT_DIM = 768;
constexpr int CB_SIZE = 1024;

struct StageSpec {
    int input_dim;
    int d_model;
    int n_heads;
    int dim_ff;
    int n_layers;
    int output_dim;
    int patch_after; // 0 if none
    int gguf_idx;    // dec.<this>
    int context;     // sliding-window causal attention span, in keys
};

// dec.0 @12.5Hz, dec.2 @25Hz, dec.4 @50Hz, dec.6 @100Hz (see STUDY.md §6).
constexpr std::array<StageSpec, 4> DECODER_STAGES = {{
    {768, 1280, 20, 5120, 32, 1280, 2, 0, 125},
    {640, 768, 12, 3072, 12, 768, 2, 2, 250},
    {384, 768, 12, 3072, 12, 768, 2, 4, 500},
    {384, 768, 12, 3072, 12, 240, 240, 6, 1000},
}};

// Encoder (voice cloning: waveform -> codes). Mirror of the decoder; the initial
// patch=240 downsample happens BEFORE enc.1 (CODEC_PRE_PATCH); patch_after here
// is the downsample that FOLLOWS the stage. RVQ dims: cb_dim 8, rvq 512, out 768.
constexpr int CODEC_PRE_PATCH = 240;
constexpr std::array<StageSpec, 4> ENCODER_STAGES = {{
    {240, 768, 12, 3072, 12, 384, 2, 1, 1000},  // enc.1 @100Hz + enc.2 patch
    {768, 768, 12, 3072, 12, 384, 2, 3, 500},   // enc.3 @50Hz  + enc.4 patch
    {768, 768, 12, 3072, 12, 640, 2, 5, 250},   // enc.5 @25Hz  + enc.6 patch
    {1280, 1280, 20, 5120, 32, 768, 0, 7, 125}, // enc.7 @12.5Hz (no patch after)
}};

} // namespace

// ===========================================================================
// Codec
// ===========================================================================

struct Codec {
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;

    // Loaded weights.
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Reconstructed (post-weight-norm) projection weights, own buffer.
    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;
    std::array<ggml_tensor*, NUM_VQ> q_oproj_w{};
    std::array<ggml_tensor*, NUM_VQ> q_oproj_b{};
    ggml_tensor* quant_oproj_w = nullptr;
    ggml_tensor* quant_oproj_b = nullptr;

    // Encoder-only reconstructed weights + L2-normalized codebooks (voice cloning).
    std::array<ggml_tensor*, NUM_VQ> q_iproj_w{};
    std::array<ggml_tensor*, NUM_VQ> q_iproj_b{};
    ggml_tensor* quant_iproj_w = nullptr;
    ggml_tensor* quant_iproj_b = nullptr;
    std::array<ggml_tensor*, NUM_VQ> codebook_normed{};
    bool encoder_ready = false;

    std::array<ggml_tensor*, NUM_VQ> codebook{};

    struct Layer {
        ggml_tensor *norm1_w, *norm1_b, *norm2_w, *norm2_b;
        ggml_tensor *attn_in, *attn_out, *linear1, *linear2;
        ggml_tensor *layer_scale_1, *layer_scale_2;
    };
    struct Stage {
        StageSpec spec;
        ggml_tensor* iproj = nullptr;
        ggml_tensor* oproj = nullptr;
        std::vector<Layer> layers;
    };
    std::array<Stage, 4> stages;
    std::array<Stage, 4> enc_stages; // encoder transformer stages (voice cloning)

    std::vector<uint8_t> compute_meta;

    ggml_tensor* get(const std::string& name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : it->second;
    }
    ggml_tensor* req(const std::string& name) const {
        ggml_tensor* t = get(name);
        if (!t)
            fprintf(stderr, "moss_tts_codec: missing tensor %s\n", name.c_str());
        return t;
    }
};

namespace {

// f16 device tensor -> host f32 vector.
std::vector<float> read_f32(ggml_tensor* t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else { // F16
        std::vector<ggml_fp16_t> h(n);
        ggml_backend_tensor_get(t, h.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++)
            out[i] = ggml_fp16_to_fp32(h[i]);
    }
    return out;
}

// Reconstruct effective Conv1d(k=1) weight from weight-norm params wp0/wp1:
//   w[o,i] = wp0[o] * wp1[o,i] / sqrt(Sum_i wp1[o,i]^2)
// Uploads dst_w as f16 in (in, out) layout, dst_b as f16 (out,).
bool reconstruct(const Codec& c, const std::string& base, ggml_tensor* dst_w, ggml_tensor* dst_b, int in_dim,
                 int out_dim) {
    ggml_tensor* wp0_t = c.get(base + "wp0");
    ggml_tensor* wp1_t = c.get(base + "wp1");
    ggml_tensor* bias_t = c.get(base + "bias");
    if (!wp0_t || !wp1_t)
        return false;
    if ((int)ggml_nelements(wp0_t) != out_dim || (int)ggml_nelements(wp1_t) != in_dim * out_dim)
        return false;
    std::vector<float> wp0 = read_f32(wp0_t);
    std::vector<float> wp1 = read_f32(wp1_t);
    std::vector<ggml_fp16_t> w_eff((size_t)in_dim * out_dim);
    for (int o = 0; o < out_dim; o++) {
        float g = wp0[(size_t)o];
        float ssq = 0.f;
        for (int i = 0; i < in_dim; i++) {
            float v = wp1[(size_t)o * in_dim + i];
            ssq += v * v;
        }
        float scale = ssq > 0.f ? g / std::sqrt(ssq) : 0.f;
        for (int i = 0; i < in_dim; i++)
            w_eff[(size_t)o * in_dim + i] = ggml_fp32_to_fp16(scale * wp1[(size_t)o * in_dim + i]);
    }
    ggml_backend_tensor_set(dst_w, w_eff.data(), 0, w_eff.size() * sizeof(ggml_fp16_t));
    std::vector<ggml_fp16_t> bias((size_t)out_dim, ggml_fp32_to_fp16(0.f));
    if (bias_t) {
        std::vector<float> b = read_f32(bias_t);
        for (int o = 0; o < out_dim && o < (int)b.size(); o++)
            bias[(size_t)o] = ggml_fp32_to_fp16(b[(size_t)o]);
    }
    ggml_backend_tensor_set(dst_b, bias.data(), 0, bias.size() * sizeof(ggml_fp16_t));
    return true;
}

ggml_tensor* to_f32(ggml_context* g, ggml_tensor* t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(g, t, GGML_TYPE_F32);
}

ggml_tensor* build_layer_norm(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* y = ggml_norm(g, x, NORM_EPS);
    y = ggml_mul(g, y, to_f32(g, w));
    y = ggml_add(g, y, to_f32(g, b));
    return y;
}

// x: (d_model, T). Fused QKV -> split -> RoPE (adjacent-pair, base 10000) ->
// manual masked SDPA -> out proj. mask: (T, T) F32 additive sliding-window mask.
ggml_tensor* build_attention(ggml_context* g, ggml_tensor* x, const Codec::Layer& L, int d_model, int n_heads,
                             ggml_tensor* pos, ggml_tensor* mask) {
    const int head_dim = d_model / n_heads;
    const int T = (int)x->ne[1];
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    ggml_tensor* qkv = ggml_mul_mat(g, L.attn_in, x); // (3*d_model, T)
    const size_t e = ggml_type_size(qkv->type);
    const size_t row = (size_t)head_dim * e;
    ggml_tensor* Q = ggml_view_3d(g, qkv, head_dim, n_heads, T, row, qkv->nb[1], 0);
    ggml_tensor* K = ggml_view_3d(g, qkv, head_dim, n_heads, T, row, qkv->nb[1], (size_t)d_model * e);
    ggml_tensor* V = ggml_view_3d(g, qkv, head_dim, n_heads, T, row, qkv->nb[1], (size_t)2 * d_model * e);
    Q = ggml_cont(g, Q);
    K = ggml_cont(g, K);
    V = ggml_cont(g, V);

    Q = ggml_rope_ext(g, Q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, T, ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    K = ggml_rope_ext(g, K, pos, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, T, ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3)); // (head_dim, T, n_heads)
    K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
    V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

    ggml_tensor* scores = ggml_mul_mat(g, K, Q); // (T, T, n_heads)
    scores = ggml_soft_max_ext(g, scores, mask, attn_scale, 0.0f);

    ggml_tensor* V2 = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
    ggml_tensor* attn = ggml_mul_mat(g, V2, scores);        // (head_dim, T, n_heads)
    attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3)); // (head_dim, n_heads, T)
    attn = ggml_reshape_2d(g, attn, d_model, T);
    attn = ggml_mul_mat(g, L.attn_out, attn); // (d_model, T)
    return attn;
}

ggml_tensor* build_layer(ggml_context* g, ggml_tensor* x, const Codec::Layer& L, int d_model, int n_heads,
                         ggml_tensor* pos, ggml_tensor* mask) {
    ggml_tensor* y = build_layer_norm(g, x, L.norm1_w, L.norm1_b);
    y = build_attention(g, y, L, d_model, n_heads, pos, mask);
    y = ggml_mul(g, y, to_f32(g, L.layer_scale_1));
    x = ggml_add(g, x, y);

    y = build_layer_norm(g, x, L.norm2_w, L.norm2_b);
    y = ggml_mul_mat(g, L.linear1, y);
    y = ggml_gelu(g, y);
    y = ggml_mul_mat(g, L.linear2, y);
    y = ggml_mul(g, y, to_f32(g, L.layer_scale_2));
    x = ggml_add(g, x, y);
    return x;
}

ggml_tensor* build_stage(ggml_context* g, ggml_tensor* x, const Codec::Stage& S, ggml_tensor* pos, ggml_tensor* mask) {
    if (S.iproj)
        x = ggml_mul_mat(g, S.iproj, x);
    for (const auto& L : S.layers)
        x = build_layer(g, x, L, S.spec.d_model, S.spec.n_heads, pos, mask);
    if (S.oproj)
        x = ggml_mul_mat(g, S.oproj, x);
    return x;
}

// PyTorch: x.reshape(d, patch, T_in).permute(...).reshape(d/patch? ...) — see
// openmoss patch_upsample_: (d*h, T_in) -> (d, T_in*h). Here dh=x->ne[0].
ggml_tensor* patch_upsample(ggml_context* g, ggml_tensor* x, int patch) {
    const int64_t dh = x->ne[0];
    const int64_t T_in = x->ne[1];
    const int64_t d = dh / patch;
    ggml_tensor* y = ggml_reshape_3d(g, x, patch, d, T_in);
    y = ggml_permute(g, y, 1, 0, 2, 3); // (d, patch, T_in)
    y = ggml_cont(g, y);
    y = ggml_reshape_2d(g, y, d, T_in * patch);
    return y;
}

// Inverse of patch_upsample (encoder downsample): (D, T_in) -> (D*patch, T_in/patch).
// out[d*patch + h, t] = in[d, t*patch + h].
ggml_tensor* patch_downsample(ggml_context* g, ggml_tensor* x, int patch) {
    const int64_t D = x->ne[0];
    const int64_t T_out = x->ne[1] / patch;
    ggml_tensor* y = ggml_reshape_3d(g, x, D, patch, T_out);
    y = ggml_permute(g, y, 1, 0, 2, 3); // (patch, D, T_out)
    y = ggml_cont(g, y);
    y = ggml_reshape_2d(g, y, D * patch, T_out);
    return y;
}

// (T, T) F32 additive sliding-window causal mask: 0 if kv<=q && q-kv<ctx else -inf.
void fill_mask(std::vector<float>& buf, int T, int context) {
    buf.assign((size_t)T * T, 0.f);
    for (int q = 0; q < T; q++)
        for (int kv = 0; kv < T; kv++)
            buf[(size_t)q * T + kv] = (kv <= q && (q - kv) < context) ? 0.f : -INFINITY;
}

} // namespace

// ===========================================================================
// Load
// ===========================================================================

Codec* load(const char* path, ggml_backend_t backend, ggml_backend_sched_t sched, int verbosity) {
    if (!path || !backend)
        return nullptr;
    auto* c = new Codec();
    c->backend = backend;
    c->sched = sched;

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "moss_tts_codec", wl)) {
        fprintf(stderr, "moss_tts_codec: failed to load weights from %s\n", path);
        delete c;
        return nullptr;
    }
    c->ctx = wl.ctx;
    c->buf = wl.buf;
    c->tensors = std::move(wl.tensors);

    // Codebooks + decoder stage weights.
    bool ok = true;
    for (int i = 0; i < NUM_VQ; i++) {
        c->codebook[i] = c->req("moss.codec.quantizer.q." + std::to_string(i) + ".codebook.weight");
        ok = ok && c->codebook[i];
    }
    for (size_t s = 0; s < c->stages.size(); s++) {
        auto& S = c->stages[s];
        S.spec = DECODER_STAGES[s];
        const std::string base = "moss.codec.dec." + std::to_string(S.spec.gguf_idx) + ".";
        S.iproj = (S.spec.d_model != S.spec.input_dim) ? c->req(base + "iproj.weight") : nullptr;
        S.oproj = (S.spec.d_model != S.spec.output_dim) ? c->req(base + "oproj.weight") : nullptr;
        S.layers.resize((size_t)S.spec.n_layers);
        for (int li = 0; li < S.spec.n_layers; li++) {
            const std::string lb = base + "tr.l." + std::to_string(li) + ".";
            auto& L = S.layers[(size_t)li];
            L.norm1_w = c->req(lb + "norm1.weight");
            L.norm1_b = c->req(lb + "norm1.bias");
            L.norm2_w = c->req(lb + "norm2.weight");
            L.norm2_b = c->req(lb + "norm2.bias");
            L.attn_in = c->req(lb + "attn.inp.0.weight");
            L.attn_out = c->req(lb + "attn.outp.0.weight");
            L.linear1 = c->req(lb + "linear1.weight");
            L.linear2 = c->req(lb + "linear2.weight");
            L.layer_scale_1 = c->req(lb + "layer_scale_1.scale");
            L.layer_scale_2 = c->req(lb + "layer_scale_2.scale");
            ok = ok && L.norm1_w && L.attn_in && L.linear1 && L.layer_scale_1;
        }
    }
    if (!ok) {
        fprintf(stderr, "moss_tts_codec: missing decoder tensors\n");
        free(c);
        return nullptr;
    }

    // Reconstruct weight-normed projections into a dedicated buffer.
    {
        ggml_init_params ip = {ggml_tensor_overhead() * (5 * NUM_VQ + 16), nullptr, true};
        c->w_ctx = ggml_init(ip);
        auto mk_w = [&](int in, int out) { return ggml_new_tensor_2d(c->w_ctx, GGML_TYPE_F16, in, out); };
        auto mk_b = [&](int out) { return ggml_new_tensor_1d(c->w_ctx, GGML_TYPE_F16, out); };
        for (int i = 0; i < NUM_VQ; i++) {
            c->q_oproj_w[i] = mk_w(CB_DIM, RVQ_DIM);
            c->q_oproj_b[i] = mk_b(RVQ_DIM);
        }
        c->quant_oproj_w = mk_w(RVQ_DIM, OUT_DIM);
        c->quant_oproj_b = mk_b(OUT_DIM);
        // Encoder-only (voice cloning): q.iproj (512->8), quantizer.iproj (768->512),
        // and per-row-L2-normalized codebooks (8, 1024) for cosine-similarity argmax.
        for (int i = 0; i < NUM_VQ; i++) {
            c->q_iproj_w[i] = mk_w(RVQ_DIM, CB_DIM);
            c->q_iproj_b[i] = mk_b(CB_DIM);
            c->codebook_normed[i] = ggml_new_tensor_2d(c->w_ctx, GGML_TYPE_F16, CB_DIM, CB_SIZE);
        }
        c->quant_iproj_w = mk_w(OUT_DIM, RVQ_DIM);
        c->quant_iproj_b = mk_b(RVQ_DIM);
        c->w_buf = ggml_backend_alloc_ctx_tensors(c->w_ctx, backend);
        if (!c->w_buf) {
            fprintf(stderr, "moss_tts_codec: alloc weight buffer failed\n");
            free(c);
            return nullptr;
        }
        bool rok = true;
        for (int i = 0; i < NUM_VQ; i++) {
            const std::string b = "moss.codec.quantizer.q." + std::to_string(i) + ".oproj.";
            rok = rok && reconstruct(*c, b, c->q_oproj_w[i], c->q_oproj_b[i], CB_DIM, RVQ_DIM);
        }
        rok =
            rok && reconstruct(*c, "moss.codec.quantizer.oproj.", c->quant_oproj_w, c->quant_oproj_b, RVQ_DIM, OUT_DIM);
        if (!rok) {
            fprintf(stderr, "moss_tts_codec: weight-norm reconstruction failed\n");
            free(c);
            return nullptr;
        }

        // Encoder is best-effort: if the codec GGUF lacks enc/iproj tensors, decode
        // still works; only voice cloning (encode) is unavailable.
        bool eok = c->get("moss.codec.quantizer.iproj.wp0") != nullptr;
        for (int i = 0; eok && i < NUM_VQ; i++) {
            const std::string b = "moss.codec.quantizer.q." + std::to_string(i) + ".iproj.";
            eok = reconstruct(*c, b, c->q_iproj_w[i], c->q_iproj_b[i], RVQ_DIM, CB_DIM);
        }
        eok =
            eok && reconstruct(*c, "moss.codec.quantizer.iproj.", c->quant_iproj_w, c->quant_iproj_b, OUT_DIM, RVQ_DIM);
        // L2-normalize each codebook row (8-dim) -> codebook_normed (for cosine sim).
        for (int i = 0; eok && i < NUM_VQ; i++) {
            std::vector<float> cb = read_f32(c->codebook[i]); // (CB_SIZE, CB_DIM) row-major
            if ((int)cb.size() != CB_SIZE * CB_DIM) {
                eok = false;
                break;
            }
            std::vector<ggml_fp16_t> nrm((size_t)CB_SIZE * CB_DIM);
            for (int r = 0; r < CB_SIZE; r++) {
                float ss = 0.f;
                for (int d = 0; d < CB_DIM; d++)
                    ss += cb[(size_t)r * CB_DIM + d] * cb[(size_t)r * CB_DIM + d];
                float inv = ss > 0.f ? 1.0f / std::sqrt(ss) : 0.f;
                for (int d = 0; d < CB_DIM; d++)
                    nrm[(size_t)r * CB_DIM + d] = ggml_fp32_to_fp16(cb[(size_t)r * CB_DIM + d] * inv);
            }
            ggml_backend_tensor_set(c->codebook_normed[i], nrm.data(), 0, nrm.size() * sizeof(ggml_fp16_t));
        }
        // Resolve encoder transformer stages (enc.1/3/5/7), same layout as decoder.
        for (size_t s = 0; eok && s < c->enc_stages.size(); s++) {
            auto& S = c->enc_stages[s];
            S.spec = ENCODER_STAGES[s];
            const std::string base = "moss.codec.enc." + std::to_string(S.spec.gguf_idx) + ".";
            S.iproj = (S.spec.d_model != S.spec.input_dim) ? c->get(base + "iproj.weight") : nullptr;
            S.oproj = (S.spec.d_model != S.spec.output_dim) ? c->get(base + "oproj.weight") : nullptr;
            S.layers.resize((size_t)S.spec.n_layers);
            for (int li = 0; li < S.spec.n_layers; li++) {
                const std::string lb = base + "tr.l." + std::to_string(li) + ".";
                auto& L = S.layers[(size_t)li];
                L.norm1_w = c->get(lb + "norm1.weight");
                L.norm1_b = c->get(lb + "norm1.bias");
                L.norm2_w = c->get(lb + "norm2.weight");
                L.norm2_b = c->get(lb + "norm2.bias");
                L.attn_in = c->get(lb + "attn.inp.0.weight");
                L.attn_out = c->get(lb + "attn.outp.0.weight");
                L.linear1 = c->get(lb + "linear1.weight");
                L.linear2 = c->get(lb + "linear2.weight");
                L.layer_scale_1 = c->get(lb + "layer_scale_1.scale");
                L.layer_scale_2 = c->get(lb + "layer_scale_2.scale");
                eok = eok && L.norm1_w && L.attn_in && L.linear1 && L.layer_scale_1;
            }
            if ((S.iproj == nullptr) != (S.spec.d_model == S.spec.input_dim))
                eok = false;
        }
        c->encoder_ready = eok;
        if (!eok && verbosity >= 1)
            fprintf(stderr, "moss_tts_codec: encoder tensors absent/incomplete — voice cloning disabled\n");
    }

    c->compute_meta.resize(ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false));
    if (verbosity >= 1)
        fprintf(stderr, "moss_tts_codec: loaded %s (%zu tensors)\n", path, c->tensors.size());
    return c;
}

void free(Codec* c) {
    if (!c)
        return;
    if (c->w_buf)
        ggml_backend_buffer_free(c->w_buf);
    if (c->w_ctx)
        ggml_free(c->w_ctx);
    if (c->buf)
        core_gguf::release_weight_buffer(c->buf);
    if (c->ctx)
        ggml_free(c->ctx);
    delete c;
}

// ===========================================================================
// Decode
// ===========================================================================

std::vector<float> decode(Codec* c, const int32_t* codes, int n_vq, int t_audio) {
    if (!c || !codes || t_audio <= 0 || n_vq < 1 || n_vq > NUM_VQ)
        return {};
    const int n_samples = t_audio * 1920;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* g = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 65536, false);

    // Inputs: per-quantizer code vectors + per-stage positions + masks.
    std::vector<ggml_tensor*> codes_in(n_vq);
    for (int i = 0; i < n_vq; i++) {
        codes_in[i] = ggml_new_tensor_1d(g, GGML_TYPE_I32, t_audio);
        char nm[32];
        snprintf(nm, sizeof(nm), "codes_%d", i);
        ggml_set_name(codes_in[i], nm);
        ggml_set_input(codes_in[i]);
    }
    int T_at[4];
    T_at[0] = t_audio;
    T_at[1] = T_at[0] * DECODER_STAGES[0].patch_after;
    T_at[2] = T_at[1] * DECODER_STAGES[1].patch_after;
    T_at[3] = T_at[2] * DECODER_STAGES[2].patch_after;
    std::array<ggml_tensor*, 4> pos_T{}, mask_T{};
    for (int s = 0; s < 4; s++) {
        pos_T[s] = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_at[s]);
        char nm[16];
        snprintf(nm, sizeof(nm), "pos_%d", s);
        ggml_set_name(pos_T[s], nm);
        ggml_set_input(pos_T[s]);
        mask_T[s] = ggml_new_tensor_2d(g, GGML_TYPE_F32, T_at[s], T_at[s]);
        snprintf(nm, sizeof(nm), "mask_%d", s);
        ggml_set_name(mask_T[s], nm);
        ggml_set_input(mask_T[s]);
    }

    // Quantizer.decode_codes: Sum_i q.oproj(codebook[i][codes_i]) -> global oproj.
    ggml_tensor* sum = nullptr;
    for (int i = 0; i < n_vq; i++) {
        ggml_tensor* z = ggml_get_rows(g, c->codebook[i], codes_in[i]); // (8, T)
        z = ggml_mul_mat(g, c->q_oproj_w[i], z);                        // (512, T)
        z = ggml_add(g, z, to_f32(g, c->q_oproj_b[i]));
        sum = sum ? ggml_add(g, sum, z) : z;
    }
    ggml_tensor* x = ggml_mul_mat(g, c->quant_oproj_w, sum); // (768, T)
    x = ggml_add(g, x, to_f32(g, c->quant_oproj_b));

    for (int s = 0; s < 4; s++) {
        x = build_stage(g, x, c->stages[s], pos_T[s], mask_T[s]);
        if (c->stages[s].spec.patch_after > 0)
            x = patch_upsample(g, x, c->stages[s].spec.patch_after);
    }

    ggml_tensor* waveform = ggml_reshape_1d(g, x, n_samples);
    ggml_set_name(waveform, "waveform");
    ggml_set_output(waveform);
    ggml_build_forward_expand(gf, waveform);

    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "moss_tts_codec: alloc decode graph failed\n");
        ggml_free(g);
        return {};
    }

    for (int i = 0; i < n_vq; i++) {
        std::vector<int32_t> col(t_audio);
        for (int t = 0; t < t_audio; t++)
            col[(size_t)t] = codes[(size_t)i * t_audio + t];
        ggml_backend_tensor_set(codes_in[i], col.data(), 0, (size_t)t_audio * sizeof(int32_t));
    }
    for (int s = 0; s < 4; s++) {
        std::vector<int32_t> p((size_t)T_at[s]);
        for (int t = 0; t < T_at[s]; t++)
            p[(size_t)t] = t;
        ggml_backend_tensor_set(pos_T[s], p.data(), 0, p.size() * sizeof(int32_t));
        std::vector<float> mbuf;
        fill_mask(mbuf, T_at[s], DECODER_STAGES[s].context);
        ggml_backend_tensor_set(mask_T[s], mbuf.data(), 0, mbuf.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_tts_codec: decode compute failed\n");
        ggml_free(g);
        return {};
    }

    std::vector<float> wav((size_t)n_samples);
    ggml_tensor* out = ggml_graph_get_tensor(gf, "waveform");
    ggml_backend_tensor_get(out, wav.data(), 0, wav.size() * sizeof(float));
    ggml_free(g);
    return wav;
}

bool encoder_ready(const Codec* c) {
    return c && c->encoder_ready;
}

// Encode a 24 kHz mono waveform -> (n_vq, T_audio) RVQ codes (voice cloning).
// Ported from openmoss codec.cpp encode(): pad -> patch_downsample(240) -> 4 enc
// stages (each + patch_downsample) -> quantizer iproj -> 32-step residual LFQ
// (L2-normalize z, cosine-sim argmax vs the normalized codebook, subtract
// oproj(codebook[idx])).
std::vector<int32_t> encode(Codec* c, const float* waveform, int64_t n_samples, int& n_vq_out, int& t_audio_out) {
    n_vq_out = 0;
    t_audio_out = 0;
    if (!c || !c->encoder_ready || !waveform || n_samples <= 0)
        return {};
    const int64_t hop = 1920;
    const int64_t T_padded = n_samples + ((hop - (n_samples % hop)) % hop);
    const int t_audio = (int)(T_padded / hop);
    if (t_audio <= 0)
        return {};

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* g = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 65536, false);

    ggml_tensor* wav = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, T_padded);
    ggml_set_name(wav, "waveform");
    ggml_set_input(wav);

    int T_at[4];
    T_at[0] = (int)(T_padded / CODEC_PRE_PATCH);
    T_at[1] = T_at[0] / ENCODER_STAGES[0].patch_after;
    T_at[2] = T_at[1] / ENCODER_STAGES[1].patch_after;
    T_at[3] = T_at[2] / ENCODER_STAGES[2].patch_after;
    std::array<ggml_tensor*, 4> pos_T{}, mask_T{};
    for (int s = 0; s < 4; s++) {
        pos_T[s] = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_at[s]);
        char nm[16];
        snprintf(nm, sizeof(nm), "enc_pos_%d", s);
        ggml_set_name(pos_T[s], nm);
        ggml_set_input(pos_T[s]);
        mask_T[s] = ggml_new_tensor_2d(g, GGML_TYPE_F32, T_at[s], T_at[s]);
        snprintf(nm, sizeof(nm), "enc_mask_%d", s);
        ggml_set_name(mask_T[s], nm);
        ggml_set_input(mask_T[s]);
    }

    ggml_tensor* x = patch_downsample(g, wav, CODEC_PRE_PATCH); // (240, T0)
    for (int s = 0; s < 4; s++) {
        x = build_stage(g, x, c->enc_stages[s], pos_T[s], mask_T[s]);
        if (ENCODER_STAGES[s].patch_after > 0)
            x = patch_downsample(g, x, ENCODER_STAGES[s].patch_after);
    }

    ggml_tensor* residual = ggml_mul_mat(g, c->quant_iproj_w, x); // (512, T)
    residual = ggml_add(g, residual, to_f32(g, c->quant_iproj_b));

    std::array<ggml_tensor*, NUM_VQ> idx_t{};
    for (int i = 0; i < NUM_VQ; i++) {
        ggml_tensor* z_e = ggml_mul_mat(g, c->q_iproj_w[i], residual); // (8, T)
        z_e = ggml_add(g, z_e, to_f32(g, c->q_iproj_b[i]));
        ggml_tensor* z_n = ggml_l2_norm(g, z_e, 1e-12f);
        ggml_tensor* sim = ggml_mul_mat(g, c->codebook_normed[i], z_n); // (1024, T)
        ggml_prec_set_acc(sim, GGML_PREC_F32);
        ggml_tensor* idx = ggml_argmax(g, sim); // (T,) i32
        char nm[16];
        snprintf(nm, sizeof(nm), "idx_%d", i);
        ggml_set_name(idx, nm);
        ggml_set_output(idx);
        idx_t[i] = idx;
        ggml_tensor* z_q = ggml_get_rows(g, c->codebook[i], idx); // (8, T)
        z_q = ggml_mul_mat(g, c->q_oproj_w[i], z_q);
        z_q = ggml_add(g, z_q, to_f32(g, c->q_oproj_b[i]));
        residual = ggml_sub(g, residual, z_q);
    }

    for (int i = 0; i < NUM_VQ; i++)
        ggml_build_forward_expand(gf, idx_t[i]);

    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "moss_tts_codec: alloc encode graph failed\n");
        ggml_free(g);
        return {};
    }

    {
        std::vector<float> wpad((size_t)T_padded, 0.0f);
        std::memcpy(wpad.data(), waveform, (size_t)n_samples * sizeof(float));
        ggml_backend_tensor_set(wav, wpad.data(), 0, wpad.size() * sizeof(float));
    }
    for (int s = 0; s < 4; s++) {
        std::vector<int32_t> p((size_t)T_at[s]);
        for (int t = 0; t < T_at[s]; t++)
            p[(size_t)t] = t;
        ggml_backend_tensor_set(pos_T[s], p.data(), 0, p.size() * sizeof(int32_t));
        std::vector<float> mbuf;
        fill_mask(mbuf, T_at[s], ENCODER_STAGES[s].context);
        ggml_backend_tensor_set(mask_T[s], mbuf.data(), 0, mbuf.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_tts_codec: encode compute failed\n");
        ggml_free(g);
        return {};
    }

    std::vector<int32_t> out((size_t)NUM_VQ * (size_t)t_audio, 0);
    for (int i = 0; i < NUM_VQ; i++) {
        ggml_tensor* it = ggml_graph_get_tensor(gf, ("idx_" + std::to_string(i)).c_str());
        ggml_backend_tensor_get(it, out.data() + (size_t)i * t_audio, 0, (size_t)t_audio * sizeof(int32_t));
    }
    ggml_free(g);
    n_vq_out = NUM_VQ;
    t_audio_out = t_audio;
    return out;
}

} // namespace moss_tts_codec
