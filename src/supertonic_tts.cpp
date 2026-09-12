// src/supertonic_tts.cpp — Supertonic-3 TTS runtime (issue #434).
//
// Native ggml port of the ONNX-only Supertone/supertonic-3 release
// (OpenRAIL-M weights; the MIT sample code in supertone-inc/supertonic
// py/helper.py is the driving blueprint and every stage here mirrors it).
//
// Division of labour:
//   * text preprocessing, duration predictor and text encoder run on the
//     CPU (tiny: 64/256-dim over ~10^2 tokens; the VITS relative-position
//     attention and GST cross-attention are finicky and cheap — same
//     approach as melotts.cpp);
//   * the vector estimator (flow matching, 2 CFG passes/step fused into
//     one graph) and the vocoder run as ggml graphs on the chosen backend.
//
// Verified-from-graph facts this file encodes (docs/supertonic/PLAN.md):
//   * CFG inside each flow step: v = 4*v_cond - 3*v_uncond,
//     xt = xt + v/total_steps (plain Euler, t = step/total_steps).
//   * Every ConvNeXt depthwise conv (all four nets, plus the vocoder embed
//     and head convs) uses EDGE/replicate padding, symmetric k//2*dil.
//   * Rotary cross-attn positions are LENGTH-NORMALISED: angle =
//     (t/len)*theta_i, theta_i = 10*10000^(-i/32); rotate-half 32|32 on
//     64-dim heads; score scale 1/16 (not 1/sqrt(64)).
//   * Time embedding: sin/cos(t*1000*f_i), f_i = 10000^(-i/31), then a
//     64->256->64 MLP with Mish.
//   * GST attention: keys pass through tanh AFTER bias; softmax over the
//     50 style tokens; 2 heads split on the channel axis.
//   * speech_prompted_text_encoder: x1 = text + attn1(text); output =
//     LN(text + attn2(q=x1)) — attention2's residual adds the ORIGINAL
//     text, not x1.
//   * duration predictor: learned sentence token PREPENDED (CLS at 0),
//     encoder output sliced at 0, 1x1 proj, concat flattened style_dp,
//     Linear->PReLU->Linear->exp.

#include "supertonic_tts.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "core/gguf_loader.h"
#include "core/ggml_cpu_backend.h"
#include "core/gpu_backend_pref.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

// ── bench instrumentation (SUPERTONIC_BENCH=1) ──────────────────────────
static bool supertonic_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("SUPERTONIC_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}
struct supertonic_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit supertonic_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~supertonic_bench_stage() {
        if (!supertonic_bench_enabled())
            return;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "  supertonic_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ── model structs ───────────────────────────────────────────────────────

struct st_convnext_cpu {
    std::vector<float> dw_w, dw_b;   // (C*K), (C)
    std::vector<float> ln_w, ln_b;   // (C)
    std::vector<float> pw1_w, pw1_b; // (I*C), (I)
    std::vector<float> pw2_w, pw2_b; // (C*I)  (C)
    std::vector<float> gamma;        // (C)
    int dil = 1;
};

struct st_vits_layer {
    std::vector<float> qw, qb, kw, kb, vw, vb, ow, ob; // 1x1 convs (C*C),(C)
    std::vector<float> rel_k, rel_v;                   // (9*D)
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;     // (C)
    std::vector<float> ffn1_w, ffn1_b;                 // (F*C),(F)
    std::vector<float> ffn2_w, ffn2_b;                 // (C*F),(C)
};

struct st_gst_cpu {                                    // spte attention (CPU)
    std::vector<float> qw, qb, kw, kb, vw, vb, ow, ob; // (256*256),(256)
};

struct st_vf_block {       // graph-side tensors
    ggml_tensor* cn[6][9]; // 6 convnext: dw_w,dw_b,ln_w,ln_b,pw1_w,pw1_b,pw2_w,pw2_b,gamma
    int cn_dil[6];
    ggml_tensor* time_w; // (64,512) ne
    ggml_tensor* time_b;
    ggml_tensor *attn_qw, *attn_qb, *attn_kw, *attn_kb, *attn_vw, *attn_vb, *attn_ow, *attn_ob;
    ggml_tensor *attn_ln_w, *attn_ln_b;
    ggml_tensor *st_qw, *st_qb, *st_kw, *st_kb, *st_vw, *st_vb, *st_ow, *st_ob;
    ggml_tensor *st_ln_w, *st_ln_b;
};

struct supertonic_context {
    supertonic_context_params params{};
    ggml_backend_t backend = nullptr; // graph backend (vf + vocoder)
    ggml_backend_t backend_cpu = nullptr;
    core_gguf::WeightLoad weights;

    // hparams
    int sample_rate = 44100;
    int base_chunk = 512;
    int ccf = 6;
    int ldim = 24;
    float cfg_cond = 4.0f, cfg_uncond = 3.0f;
    float normalizer_scale = 0.25f;
    float vf_text_attn_scale = 16.0f;
    float vf_style_attn_scale = 16.0f;
    float te_spte_attn_scale = 16.0f;
    float dp_convnext_eps = 1e-6f, dp_attn_eps = 1e-5f;
    float te_convnext_eps = 1e-6f, te_attn_eps = 1e-5f, te_spte_eps = 1e-5f;
    float vf_convnext_eps = 1e-6f, vf_block_eps = 1e-6f;
    float voc_convnext_eps = 1e-6f, voc_bn_eps = 1e-5f;

    // text tables
    std::vector<int32_t> indexer; // 65536
    std::map<uint32_t, std::vector<uint32_t>> nfkd;
    std::map<uint32_t, int> ccc;

    // voices
    struct Voice {
        std::vector<float> ttl; // 50*256, [s*256+c]
        std::vector<float> dp;  // 8*16
    };
    std::map<std::string, Voice> voices;

    // ── duration predictor (CPU) ──
    std::vector<float> dp_char_emb;       // 8322*64
    std::vector<float> dp_sentence_token; // 64
    st_convnext_cpu dp_cn[6];
    st_vits_layer dp_attn[2];
    std::vector<float> dp_proj_w;        // 64*64 (no bias)
    std::vector<float> dp_l0_w, dp_l0_b; // 128*192, 128
    std::vector<float> dp_l1_w, dp_l1_b; // 1*128, 1
    float dp_prelu_a = 0.25f;

    // ── text encoder (CPU) ──
    std::vector<float> te_char_emb; // 8322*256
    st_convnext_cpu te_cn[6];
    st_vits_layer te_attn[4];
    std::vector<float> te_style_key; // 50*256
    st_gst_cpu te_att1, te_att2;
    std::vector<float> te_norm_w, te_norm_b; // 256

    // ── vector field (graph) ──
    ggml_tensor* vf_proj_in = nullptr;  // (144,512) ne
    ggml_tensor* vf_proj_out = nullptr; // (512,144) ne
    st_vf_block vf_blocks[4];
    ggml_tensor* vf_last_cn[4][9];
    // CPU-side time encoder
    std::vector<float> vf_time_freqs;        // 32
    std::vector<float> vf_mlp0_w, vf_mlp0_b; // 256*64, 256
    std::vector<float> vf_mlp2_w, vf_mlp2_b; // 64*256, 64
    std::vector<float> vf_rotary_theta;      // 32
    std::vector<float> vf_uncond_text;       // 256
    std::vector<float> vf_uncond_sk;         // 50*256
    std::vector<float> vf_uncond_sv;         // 50*256

    // ── vocoder (graph) ──
    ggml_tensor* voc_embed_w = nullptr; // (7,24,512)
    ggml_tensor* voc_embed_b = nullptr;
    ggml_tensor* voc_cn[10][9];
    int voc_cn_dil[10] = {1, 2, 4, 1, 2, 4, 1, 1, 1, 1};
    ggml_tensor* voc_head1_w = nullptr; // (3,512,2048)
    ggml_tensor* voc_head1_b = nullptr;
    ggml_tensor* voc_head2_w = nullptr;                 // (2048,512) matmul
    std::vector<float> voc_bn_scale, voc_bn_shift;      // folded BatchNorm (512)
    std::vector<float> voc_latent_mean, voc_latent_std; // 24
    float voc_prelu_a = 0.25f;

    std::mt19937_64 rng;
    bool rng_seeded = false;
};

// ── small helpers ───────────────────────────────────────────────────────

static bool read_f32(core_gguf::tensor_map& tm, const std::string& name, std::vector<float>& out) {
    auto it = tm.find(name);
    if (it == tm.end()) {
        std::fprintf(stderr, "supertonic: missing tensor %s\n", name.c_str());
        return false;
    }
    ggml_tensor* t = it->second;
    size_t n = ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++)
            out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        std::fprintf(stderr, "supertonic: %s has unsupported type %d\n", name.c_str(), (int)t->type);
        return false;
    }
    return true;
}

static bool read_i32(core_gguf::tensor_map& tm, const std::string& name, std::vector<int32_t>& out) {
    auto it = tm.find(name);
    if (it == tm.end())
        return false;
    ggml_tensor* t = it->second;
    size_t n = ggml_nelements(t);
    out.resize(n);
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(int32_t));
    return true;
}

static ggml_tensor* st_req(core_gguf::tensor_map& tm, const std::string& name) {
    return core_gguf::require(tm, name.c_str(), "supertonic");
}

// ── CPU math ────────────────────────────────────────────────────────────

// y[T,O] = x[T,I] @ w(O,I)^T + b
static void cpu_linear(const float* x, const float* w, const float* b, int T, int I, int O, float* y) {
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < O; o++) {
            float s = b ? b[o] : 0.0f;
            const float* wr = w + (size_t)o * I;
            const float* xr = x + (size_t)t * I;
            for (int i = 0; i < I; i++)
                s += xr[i] * wr[i];
            y[(size_t)t * O + o] = s;
        }
    }
}

static inline float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

static void cpu_layernorm(float* x, const float* w, const float* b, int T, int C, float eps) {
    for (int t = 0; t < T; t++) {
        float* r = x + (size_t)t * C;
        float mean = 0;
        for (int c = 0; c < C; c++)
            mean += r[c];
        mean /= C;
        float var = 0;
        for (int c = 0; c < C; c++) {
            float d = r[c] - mean;
            var += d * d;
        }
        var /= C;
        float inv = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < C; c++)
            r[c] = (r[c] - mean) * inv * w[c] + b[c];
    }
}

// ConvNeXt block on (T,C) row-major, k=5, EDGE padding, dilation d.
static void cpu_convnext(std::vector<float>& x, const st_convnext_cpu& cn, int T, int C, float eps) {
    const int K = (int)(cn.dw_w.size() / C);
    const int pad = (K / 2) * cn.dil;
    std::vector<float> y((size_t)T * C);
    // depthwise conv with edge padding
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            float s = cn.dw_b[c];
            for (int k = 0; k < K; k++) {
                int tt = t + k * cn.dil - pad;
                if (tt < 0)
                    tt = 0;
                if (tt >= T)
                    tt = T - 1;
                s += cn.dw_w[(size_t)c * K + k] * x[(size_t)tt * C + c];
            }
            y[(size_t)t * C + c] = s;
        }
    }
    cpu_layernorm(y.data(), cn.ln_w.data(), cn.ln_b.data(), T, C, eps);
    const int I = (int)(cn.pw1_b.size());
    std::vector<float> h((size_t)T * I);
    cpu_linear(y.data(), cn.pw1_w.data(), cn.pw1_b.data(), T, C, I, h.data());
    for (auto& v : h)
        v = gelu_erf(v);
    cpu_linear(h.data(), cn.pw2_w.data(), cn.pw2_b.data(), T, I, C, y.data());
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            x[(size_t)t * C + c] += cn.gamma[c] * y[(size_t)t * C + c];
}

// VITS MultiHeadAttention with relative-position window (W=4), (T,C).
static void cpu_vits_attn(const std::vector<float>& x, const st_vits_layer& L, int T, int C, int H,
                          std::vector<float>& out) {
    const int D = C / H;
    const int W = 4;
    std::vector<float> Q((size_t)T * C), K((size_t)T * C), V((size_t)T * C);
    cpu_linear(x.data(), L.qw.data(), L.qb.data(), T, C, C, Q.data());
    cpu_linear(x.data(), L.kw.data(), L.kb.data(), T, C, C, K.data());
    cpu_linear(x.data(), L.vw.data(), L.vb.data(), T, C, C, V.data());
    const float inv = 1.0f / sqrtf((float)D);
    std::vector<float> attn_out((size_t)T * C, 0.0f);
    std::vector<float> scores((size_t)T * T);
    for (int h = 0; h < H; h++) {
        const int off = h * D;
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                float s = 0;
                for (int d = 0; d < D; d++)
                    s += Q[(size_t)i * C + off + d] * K[(size_t)j * C + off + d];
                scores[(size_t)i * T + j] = s * inv;
            }
            const int j0 = std::max(0, i - W), j1 = std::min(T, i + W + 1);
            for (int j = j0; j < j1; j++) {
                const int r = j - i + W;
                float s = 0;
                for (int d = 0; d < D; d++)
                    s += Q[(size_t)i * C + off + d] * L.rel_k[(size_t)r * D + d];
                scores[(size_t)i * T + j] += s * inv;
            }
        }
        for (int i = 0; i < T; i++) {
            float mx = scores[(size_t)i * T];
            for (int j = 1; j < T; j++)
                mx = std::max(mx, scores[(size_t)i * T + j]);
            float sum = 0;
            for (int j = 0; j < T; j++) {
                float e = expf(scores[(size_t)i * T + j] - mx);
                scores[(size_t)i * T + j] = e;
                sum += e;
            }
            for (int j = 0; j < T; j++)
                scores[(size_t)i * T + j] /= sum;
        }
        for (int i = 0; i < T; i++) {
            for (int d = 0; d < D; d++) {
                float s = 0;
                for (int j = 0; j < T; j++)
                    s += scores[(size_t)i * T + j] * V[(size_t)j * C + off + d];
                attn_out[(size_t)i * C + off + d] = s;
            }
            const int j0 = std::max(0, i - W), j1 = std::min(T, i + W + 1);
            for (int j = j0; j < j1; j++) {
                const int r = j - i + W;
                const float wij = scores[(size_t)i * T + j];
                for (int d = 0; d < D; d++)
                    attn_out[(size_t)i * C + off + d] += wij * L.rel_v[(size_t)r * D + d];
            }
        }
    }
    out.resize((size_t)T * C);
    cpu_linear(attn_out.data(), L.ow.data(), L.ob.data(), T, C, C, out.data());
}

// VITS attention encoder layer: x = LN1(x + attn(x)); x = LN2(x + ffn(x)).
static void cpu_vits_layer_fwd(std::vector<float>& x, const st_vits_layer& L, int T, int C, int H, float eps) {
    std::vector<float> y;
    cpu_vits_attn(x, L, T, C, H, y);
    for (size_t i = 0; i < x.size(); i++)
        x[i] += y[i];
    cpu_layernorm(x.data(), L.ln1_w.data(), L.ln1_b.data(), T, C, eps);
    const int F = (int)L.ffn1_b.size();
    std::vector<float> h((size_t)T * F);
    cpu_linear(x.data(), L.ffn1_w.data(), L.ffn1_b.data(), T, C, F, h.data());
    for (auto& v : h)
        v = std::max(0.0f, v);
    y.resize((size_t)T * C);
    cpu_linear(h.data(), L.ffn2_w.data(), L.ffn2_b.data(), T, F, C, y.data());
    for (size_t i = 0; i < x.size(); i++)
        x[i] += y[i];
    cpu_layernorm(x.data(), L.ln2_w.data(), L.ln2_b.data(), T, C, eps);
}

// GST cross-attention (queries (T,Cq) -> 256, keys/values from styles),
// H heads on the channel split; keys pass through tanh AFTER bias.
static void cpu_gst_attn(const float* q_in, int T, int Cq, const float* keys_in, const float* vals_in, int S,
                         const st_gst_cpu& A, float scale, int H, int Co, std::vector<float>& out) {
    const int Ci = (int)(A.qw.size() / Cq); // inner dim (256)
    const int D = Ci / H;
    std::vector<float> Q((size_t)T * Ci), K((size_t)S * Ci), V((size_t)S * Ci);
    cpu_linear(q_in, A.qw.data(), A.qb.data(), T, Cq, Ci, Q.data());
    cpu_linear(keys_in, A.kw.data(), A.kb.data(), S, 256, Ci, K.data());
    cpu_linear(vals_in, A.vw.data(), A.vb.data(), S, 256, Ci, V.data());
    for (auto& v : K)
        v = tanhf(v);
    std::vector<float> merged((size_t)T * Ci);
    std::vector<float> sc(S);
    for (int h = 0; h < H; h++) {
        const int off = h * D;
        for (int t = 0; t < T; t++) {
            float mx = -1e30f;
            for (int s = 0; s < S; s++) {
                float v = 0;
                for (int d = 0; d < D; d++)
                    v += Q[(size_t)t * Ci + off + d] * K[(size_t)s * Ci + off + d];
                v /= scale;
                sc[s] = v;
                mx = std::max(mx, v);
            }
            float sum = 0;
            for (int s = 0; s < S; s++) {
                sc[s] = expf(sc[s] - mx);
                sum += sc[s];
            }
            for (int s = 0; s < S; s++)
                sc[s] /= sum;
            for (int d = 0; d < D; d++) {
                float v = 0;
                for (int s = 0; s < S; s++)
                    v += sc[s] * V[(size_t)s * Ci + off + d];
                merged[(size_t)t * Ci + off + d] = v;
            }
        }
    }
    out.resize((size_t)T * Co);
    cpu_linear(merged.data(), A.ow.data(), A.ob.data(), T, Ci, Co, out.data());
}

// ── text preprocessing ──────────────────────────────────────────────────

static std::vector<uint32_t> utf8_decode(const char* s) {
    std::vector<uint32_t> cps;
    const unsigned char* p = (const unsigned char*)s;
    while (*p) {
        uint32_t cp = 0;
        int n = 0;
        if (*p < 0x80) {
            cp = *p;
            n = 1;
        } else if ((*p >> 5) == 0x6) {
            cp = *p & 0x1F;
            n = 2;
        } else if ((*p >> 4) == 0xE) {
            cp = *p & 0x0F;
            n = 3;
        } else if ((*p >> 3) == 0x1E) {
            cp = *p & 0x07;
            n = 4;
        } else {
            p++;
            continue;
        }
        bool ok = true;
        for (int i = 1; i < n; i++) {
            if ((p[i] & 0xC0) != 0x80) {
                ok = false;
                break;
            }
            cp = (cp << 6) | (p[i] & 0x3F);
        }
        if (!ok) {
            p++;
            continue;
        }
        cps.push_back(cp);
        p += n;
    }
    return cps;
}

static bool st_is_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v' || c == 0xA0 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

static bool st_is_emoji(uint32_t c) {
    return (c >= 0x1F600 && c <= 0x1F64F) || (c >= 0x1F300 && c <= 0x1F5FF) || (c >= 0x1F680 && c <= 0x1F6FF) ||
           (c >= 0x1F700 && c <= 0x1F77F) || (c >= 0x1F780 && c <= 0x1F7FF) || (c >= 0x1F800 && c <= 0x1F8FF) ||
           (c >= 0x1F900 && c <= 0x1F9FF) || (c >= 0x1FA00 && c <= 0x1FA6F) || (c >= 0x1FA70 && c <= 0x1FAFF) ||
           (c >= 0x2600 && c <= 0x26FF) || (c >= 0x2700 && c <= 0x27BF) || (c >= 0x1F1E6 && c <= 0x1F1FF);
}

// NFKD via GGUF tables + canonical reordering of combining marks.
static std::vector<uint32_t> st_nfkd(const supertonic_context* ctx, const std::vector<uint32_t>& in) {
    std::vector<uint32_t> out;
    out.reserve(in.size() + 8);
    for (uint32_t c : in) {
        auto it = ctx->nfkd.find(c);
        if (it != ctx->nfkd.end())
            out.insert(out.end(), it->second.begin(), it->second.end());
        else
            out.push_back(c);
    }
    // canonical ordering: stable-sort maximal runs of nonzero-ccc marks
    auto cc = [&](uint32_t c) {
        auto it = ctx->ccc.find(c);
        return it == ctx->ccc.end() ? 0 : it->second;
    };
    size_t i = 0;
    while (i < out.size()) {
        if (cc(out[i]) == 0) {
            i++;
            continue;
        }
        size_t j = i;
        while (j < out.size() && cc(out[j]) != 0)
            j++;
        std::stable_sort(out.begin() + i, out.begin() + j, [&](uint32_t a, uint32_t b) { return cc(a) < cc(b); });
        i = j;
    }
    return out;
}

// Full upstream preprocessing: returns the "<lang>...</lang>" codepoints.
static std::vector<uint32_t> st_preprocess(const supertonic_context* ctx, const char* text, const char* lang) {
    std::vector<uint32_t> cps = st_nfkd(ctx, utf8_decode(text));
    std::vector<uint32_t> t;
    t.reserve(cps.size());
    for (uint32_t c : cps) {
        if (st_is_emoji(c))
            continue;
        switch (c) {
        case 0x2013:
        case 0x2011:
        case 0x2014:
            c = '-';
            break;
        case '_':
            c = ' ';
            break;
        case 0x201C:
        case 0x201D:
            c = '"';
            break;
        case 0x2018:
        case 0x2019:
        case 0xB4:
        case '`':
            c = '\'';
            break;
        case '[':
        case ']':
        case '|':
        case '/':
        case '#':
        case 0x2192:
        case 0x2190:
            c = ' ';
            break;
        case 0x2665:
        case 0x2606:
        case 0x2661:
        case 0xA9:
        case '\\':
            continue;
        default:
            break;
        }
        t.push_back(c);
    }
    // expression replacements (@ / e.g., / i.e.,) — ASCII, safe on codepoints
    std::vector<uint32_t> r;
    r.reserve(t.size() + 16);
    auto match_at = [&](size_t i, const char* pat) {
        size_t n = strlen(pat);
        if (i + n > t.size())
            return false;
        for (size_t k = 0; k < n; k++)
            if (t[i + k] != (uint32_t)pat[k])
                return false;
        return true;
    };
    auto push_str = [&](const char* s) {
        for (; *s; s++)
            r.push_back((uint32_t)*s);
    };
    for (size_t i = 0; i < t.size();) {
        if (t[i] == '@') {
            push_str(" at ");
            i++;
        } else if (match_at(i, "e.g.,")) {
            push_str("for example, ");
            i += 5;
        } else if (match_at(i, "i.e.,")) {
            push_str("that is, ");
            i += 5;
        } else {
            r.push_back(t[i]);
            i++;
        }
    }
    // " X" -> "X" for punctuation
    std::vector<uint32_t> f;
    f.reserve(r.size());
    for (size_t i = 0; i < r.size(); i++) {
        if (r[i] == ' ' && i + 1 < r.size()) {
            uint32_t n = r[i + 1];
            if (n == ',' || n == '.' || n == '!' || n == '?' || n == ';' || n == ':' || n == '\'')
                continue;
        }
        f.push_back(r[i]);
    }
    // duplicate-quote collapse
    std::vector<uint32_t> g;
    g.reserve(f.size());
    for (uint32_t c : f) {
        if (!g.empty() && g.back() == c && (c == '"' || c == '\'' || c == '`'))
            continue;
        g.push_back(c);
    }
    // whitespace collapse + trim
    std::vector<uint32_t> w;
    w.reserve(g.size());
    bool prev_sp = true; // trims leading
    for (uint32_t c : g) {
        if (st_is_space(c)) {
            if (!prev_sp)
                w.push_back(' ');
            prev_sp = true;
        } else {
            w.push_back(c);
            prev_sp = false;
        }
    }
    while (!w.empty() && w.back() == ' ')
        w.pop_back();
    // ensure trailing punctuation
    static const uint32_t enders[] = {'.', '!',    '?',    ';',    ':',    ',',    '\'',   '"',    ')',    ']',
                                      '}', 0x2026, 0x3002, 0x300D, 0x300F, 0x3011, 0x3009, 0x300B, 0x203A, 0xBB};
    bool ended = false;
    if (!w.empty())
        for (uint32_t e : enders)
            if (w.back() == e) {
                ended = true;
                break;
            }
    if (!ended)
        w.push_back('.');
    // wrap <lang> ... </lang>
    std::vector<uint32_t> outv;
    outv.push_back('<');
    for (const char* p = lang; *p; p++)
        outv.push_back((uint32_t)*p);
    outv.push_back('>');
    outv.insert(outv.end(), w.begin(), w.end());
    outv.push_back('<');
    outv.push_back('/');
    for (const char* p = lang; *p; p++)
        outv.push_back((uint32_t)*p);
    outv.push_back('>');
    return outv;
}

static std::vector<int32_t> st_ids_from_cps(const supertonic_context* ctx, const std::vector<uint32_t>& cps) {
    std::vector<int32_t> ids;
    ids.reserve(cps.size());
    const int32_t vocab = 8322;
    for (uint32_t c : cps) {
        int32_t id = ctx->indexer[c & 0xFFFF]; // upstream casts ord() to uint16
        if (id < 0)
            id += vocab; // ONNX Gather negative-index wrap (indexer holds -1 for unknown)
        ids.push_back(id);
    }
    return ids;
}

// ── duration predictor forward (CPU) ────────────────────────────────────

static float st_dp_forward(supertonic_context* ctx, const int32_t* ids, int n, const std::vector<float>& style_dp) {
    supertonic_bench_stage _b("duration_predictor");
    const int C = 64;
    const int T = n + 1; // CLS prepended
    std::vector<float> x((size_t)T * C);
    for (int c = 0; c < C; c++)
        x[c] = ctx->dp_sentence_token[c];
    for (int t = 0; t < n; t++)
        memcpy(&x[(size_t)(t + 1) * C], &ctx->dp_char_emb[(size_t)ids[t] * C], C * sizeof(float));
    std::vector<float> conv_out = x;
    for (int i = 0; i < 6; i++)
        cpu_convnext(conv_out, ctx->dp_cn[i], T, C, ctx->dp_convnext_eps);
    std::vector<float> a = conv_out;
    for (int i = 0; i < 2; i++)
        cpu_vits_layer_fwd(a, ctx->dp_attn[i], T, C, /*H=*/2, ctx->dp_attn_eps);
    // global residual, then CLS slice + 1x1 proj (no bias)
    std::vector<float> cls(C);
    for (int c = 0; c < C; c++)
        cls[c] = a[c] + conv_out[c];
    std::vector<float> proj(C);
    cpu_linear(cls.data(), ctx->dp_proj_w.data(), nullptr, 1, C, C, proj.data());
    std::vector<float> cat(192);
    memcpy(cat.data(), proj.data(), 64 * sizeof(float));
    memcpy(cat.data() + 64, style_dp.data(), 128 * sizeof(float));
    std::vector<float> h(128), o(1);
    cpu_linear(cat.data(), ctx->dp_l0_w.data(), ctx->dp_l0_b.data(), 1, 192, 128, h.data());
    for (auto& v : h)
        v = v >= 0 ? v : ctx->dp_prelu_a * v;
    cpu_linear(h.data(), ctx->dp_l1_w.data(), ctx->dp_l1_b.data(), 1, 128, 1, o.data());
    return expf(o[0]);
}

// ── text encoder forward (CPU) ──────────────────────────────────────────
// out: (L,256) row-major == [256,L] with C fastest.

static void st_te_forward(supertonic_context* ctx, const int32_t* ids, int n, const std::vector<float>& style_ttl,
                          std::vector<float>& out, std::vector<float>* dbg_convnext, std::vector<float>* dbg_pre_spte) {
    supertonic_bench_stage _b("text_encoder");
    const int C = 256;
    const int T = n;
    std::vector<float> x((size_t)T * C);
    for (int t = 0; t < T; t++)
        memcpy(&x[(size_t)t * C], &ctx->te_char_emb[(size_t)ids[t] * C], C * sizeof(float));
    std::vector<float> conv_out = x;
    for (int i = 0; i < 6; i++)
        cpu_convnext(conv_out, ctx->te_cn[i], T, C, ctx->te_convnext_eps);
    if (dbg_convnext)
        *dbg_convnext = conv_out;
    std::vector<float> a = conv_out;
    for (int i = 0; i < 4; i++)
        cpu_vits_layer_fwd(a, ctx->te_attn[i], T, C, /*H=*/4, ctx->te_attn_eps);
    for (size_t i = 0; i < a.size(); i++)
        a[i] += conv_out[i];
    if (dbg_pre_spte)
        *dbg_pre_spte = a;
    // speech-prompted: x1 = a + attn1(a); out = LN(a + attn2(q=x1))
    std::vector<float> y;
    cpu_gst_attn(a.data(), T, C, ctx->te_style_key.data(), style_ttl.data(), 50, ctx->te_att1, ctx->te_spte_attn_scale,
                 /*H=*/2, C, y);
    std::vector<float> x1 = a;
    for (size_t i = 0; i < x1.size(); i++)
        x1[i] += y[i];
    cpu_gst_attn(x1.data(), T, C, ctx->te_style_key.data(), style_ttl.data(), 50, ctx->te_att2, ctx->te_spte_attn_scale,
                 /*H=*/2, C, y);
    out = a;
    for (size_t i = 0; i < out.size(); i++)
        out[i] += y[i];
    cpu_layernorm(out.data(), ctx->te_norm_w.data(), ctx->te_norm_b.data(), T, C, ctx->te_spte_eps);
}

// ── ggml graph helpers ──────────────────────────────────────────────────

// EDGE (replicate) pad along ne0 (time) of x (T,C): returns (T+2p, C).
static ggml_tensor* g_edge_pad(ggml_context* g, ggml_tensor* x, int p) {
    if (p <= 0)
        return x;
    const int64_t T = x->ne[0], C = x->ne[1];
    ggml_tensor* l = ggml_cont(g, ggml_view_2d(g, x, 1, C, x->nb[1], 0));
    ggml_tensor* r = ggml_cont(g, ggml_view_2d(g, x, 1, C, x->nb[1], (T - 1) * x->nb[0]));
    ggml_tensor* lp = ggml_repeat(g, l, ggml_new_tensor_2d(g, GGML_TYPE_F32, p, C));
    ggml_tensor* rp = ggml_repeat(g, r, ggml_new_tensor_2d(g, GGML_TYPE_F32, p, C));
    return ggml_concat(g, ggml_concat(g, lp, x, 0), rp, 0);
}

// ConvNeXt block in-graph. x: (C,T). w[9]: dw_w,dw_b,ln_w,ln_b,pw1_w,pw1_b,
// pw2_w,pw2_b,gamma. dw_w ne (K,1,C); pw ne (in,out). EDGE pad.
static ggml_tensor* g_convnext(ggml_context* g, ggml_tensor* x, ggml_tensor* const w[9], int dil, float eps) {
    const int K = (int)w[0]->ne[0];
    const int pad = (K / 2) * dil;
    ggml_tensor* xt = ggml_cont(g, ggml_transpose(g, x)); // (T,C)
    xt = g_edge_pad(g, xt, pad);
    ggml_tensor* dw = ggml_conv_1d_dw(g, w[0], xt, 1, 0, dil); // (T,C)
    dw = ggml_add(g, dw, ggml_reshape_2d(g, w[1], 1, w[1]->ne[0]));
    ggml_tensor* y = ggml_cont(g, ggml_transpose(g, dw)); // (C,T)
    y = ggml_norm(g, y, eps);
    y = ggml_add(g, ggml_mul(g, y, w[2]), w[3]);
    y = ggml_mul_mat(g, w[4], y); // (I,T)
    if (w[5])
        y = ggml_add(g, y, w[5]);
    y = ggml_gelu_erf(g, y);
    y = ggml_mul_mat(g, w[6], y); // (C,T)
    if (w[7])
        y = ggml_add(g, y, w[7]);
    y = ggml_mul(g, y, w[8]);
    return ggml_add(g, x, y);
}

// rotate-half: x (64, T, H) with cos/sin (32, T, 1) -> same shape.
static ggml_tensor* g_rotate(ggml_context* g, ggml_tensor* x, ggml_tensor* cosv, ggml_tensor* sinv) {
    const int64_t hd = x->ne[0], T = x->ne[1], H = x->ne[2];
    ggml_tensor* x1 = ggml_cont(g, ggml_view_3d(g, x, hd / 2, T, H, x->nb[1], x->nb[2], 0));
    ggml_tensor* x2 = ggml_cont(g, ggml_view_3d(g, x, hd / 2, T, H, x->nb[1], x->nb[2], (hd / 2) * x->nb[0]));
    ggml_tensor* r1 = ggml_sub(g, ggml_mul(g, x1, cosv), ggml_mul(g, x2, sinv));
    ggml_tensor* r2 = ggml_add(g, ggml_mul(g, x1, sinv), ggml_mul(g, x2, cosv));
    return ggml_concat(g, r1, r2, 0);
}

struct st_vf_inputs {
    ggml_tensor* xt;            // (144,N)
    ggml_tensor* t_emb;         // (64,1)
    ggml_tensor *cos_q, *sin_q; // (32,N,1)
    ggml_tensor *cos_k, *sin_k; // (32,L,1)
    ggml_tensor* text_c;        // (256,L)
    ggml_tensor* text_u;        // (256,L)
    ggml_tensor* sk;            // (256,50)  cond style keys (prototype)
    ggml_tensor* sv;            // (256,50)  cond style values (voice ttl)
    ggml_tensor *usk, *usv;     // uncond variants
    ggml_tensor* out;           // (144,N)
    ggml_tensor* projin_c;      // (512,N) debug probe
};

// One CFG half of the vector field. x (144,N) -> v (144,N).
static ggml_tensor* g_vf_half(supertonic_context* ctx, ggml_context* g, ggml_tensor* xt, ggml_tensor* t_emb,
                              ggml_tensor* text, ggml_tensor* sk, ggml_tensor* sv, ggml_tensor* cos_q,
                              ggml_tensor* sin_q, ggml_tensor* cos_k, ggml_tensor* sin_k, ggml_tensor** projin_out) {
    const int64_t N = xt->ne[1];
    ggml_tensor* x = ggml_mul_mat(g, ctx->vf_proj_in, xt); // (512,N)
    if (projin_out)
        *projin_out = x;
    for (int b = 0; b < 4; b++) {
        const st_vf_block& B = ctx->vf_blocks[b];
        // convnext_0 x4
        for (int i = 0; i < 4; i++)
            x = g_convnext(g, x, B.cn[i], B.cn_dil[i], ctx->vf_convnext_eps);
        // time cond
        ggml_tensor* tc = ggml_mul_mat(g, B.time_w, t_emb); // (512,1)
        tc = ggml_add(g, tc, B.time_b);
        x = ggml_add(g, x, tc);
        // convnext_1
        x = g_convnext(g, x, B.cn[4], B.cn_dil[4], ctx->vf_convnext_eps);
        // rotary text cross-attn (8 heads x 64)
        {
            ggml_tensor* q = ggml_add(g, ggml_mul_mat(g, B.attn_qw, x), B.attn_qb);    // (512,N)
            ggml_tensor* k = ggml_add(g, ggml_mul_mat(g, B.attn_kw, text), B.attn_kb); // (512,L)
            ggml_tensor* v = ggml_add(g, ggml_mul_mat(g, B.attn_vw, text), B.attn_vb); // (512,L)
            const int64_t L = text->ne[1];
            q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, q, 64, 8, N), 0, 2, 1, 3)); // (64,N,8)
            k = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, k, 64, 8, L), 0, 2, 1, 3)); // (64,L,8)
            v = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, v, 64, 8, L), 0, 2, 1, 3)); // (64,L,8)
            q = g_rotate(g, q, cos_q, sin_q);
            k = g_rotate(g, k, cos_k, sin_k);
            ggml_tensor* sc = ggml_mul_mat(g, k, q); // (L,N,8)
            sc = ggml_scale(g, sc, 1.0f / ctx->vf_text_attn_scale);
            sc = ggml_soft_max(g, sc);
            ggml_tensor* vp = ggml_cont(g, ggml_permute(g, v, 1, 0, 2, 3)); // (L,64,8)
            ggml_tensor* o = ggml_mul_mat(g, vp, sc);                       // (64,N,8)
            o = ggml_cont(g, ggml_permute(g, o, 0, 2, 1, 3));               // (64,8,N)
            o = ggml_reshape_2d(g, o, 512, N);
            o = ggml_add(g, ggml_mul_mat(g, B.attn_ow, o), B.attn_ob);
            x = ggml_add(g, x, o);
            x = ggml_norm(g, x, ctx->vf_block_eps);
            x = ggml_add(g, ggml_mul(g, x, B.attn_ln_w), B.attn_ln_b);
        }
        // convnext_2
        x = g_convnext(g, x, B.cn[5], B.cn_dil[5], ctx->vf_convnext_eps);
        // GST style cross-attn (2 heads x 128, keys tanh'd)
        {
            ggml_tensor* q = ggml_add(g, ggml_mul_mat(g, B.st_qw, x), B.st_qb);  // (256,N)
            ggml_tensor* k = ggml_add(g, ggml_mul_mat(g, B.st_kw, sk), B.st_kb); // (256,50)
            k = ggml_tanh(g, k);
            ggml_tensor* v = ggml_add(g, ggml_mul_mat(g, B.st_vw, sv), B.st_vb);              // (256,50)
            q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, q, 128, 2, N), 0, 2, 1, 3));  // (128,N,2)
            k = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, k, 128, 2, 50), 0, 2, 1, 3)); // (128,50,2)
            v = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, v, 128, 2, 50), 0, 2, 1, 3)); // (128,50,2)
            ggml_tensor* sc = ggml_mul_mat(g, k, q);                                          // (50,N,2)
            sc = ggml_scale(g, sc, 1.0f / ctx->vf_style_attn_scale);
            sc = ggml_soft_max(g, sc);
            ggml_tensor* vp = ggml_cont(g, ggml_permute(g, v, 1, 0, 2, 3)); // (50,128,2)
            ggml_tensor* o = ggml_mul_mat(g, vp, sc);                       // (128,N,2)
            o = ggml_cont(g, ggml_permute(g, o, 0, 2, 1, 3));               // (128,2,N)
            o = ggml_reshape_2d(g, o, 256, N);
            o = ggml_add(g, ggml_mul_mat(g, B.st_ow, o), B.st_ob); // (512,N)
            x = ggml_add(g, x, o);
            x = ggml_norm(g, x, ctx->vf_block_eps);
            x = ggml_add(g, ggml_mul(g, x, B.st_ln_w), B.st_ln_b);
        }
    }
    for (int i = 0; i < 4; i++)
        x = g_convnext(g, x, ctx->vf_last_cn[i], 1, ctx->vf_convnext_eps);
    return ggml_mul_mat(g, ctx->vf_proj_out, x); // (144,N)
}

// ── model load ──────────────────────────────────────────────────────────

static bool load_convnext_cpu(core_gguf::tensor_map& tm, const std::string& p, st_convnext_cpu& cn, int dil) {
    cn.dil = dil;
    return read_f32(tm, p + ".dwconv.weight", cn.dw_w) && read_f32(tm, p + ".dwconv.bias", cn.dw_b) &&
           read_f32(tm, p + ".norm.norm.weight", cn.ln_w) && read_f32(tm, p + ".norm.norm.bias", cn.ln_b) &&
           read_f32(tm, p + ".pwconv1.weight", cn.pw1_w) && read_f32(tm, p + ".pwconv1.bias", cn.pw1_b) &&
           read_f32(tm, p + ".pwconv2.weight", cn.pw2_w) && read_f32(tm, p + ".pwconv2.bias", cn.pw2_b) &&
           read_f32(tm, p + ".gamma", cn.gamma);
}

static bool load_vits_layer(core_gguf::tensor_map& tm, const std::string& enc, int idx, st_vits_layer& L) {
    const std::string a = enc + ".attn_layers." + std::to_string(idx);
    const std::string s = std::to_string(idx);
    return read_f32(tm, a + ".conv_q.weight", L.qw) && read_f32(tm, a + ".conv_q.bias", L.qb) &&
           read_f32(tm, a + ".conv_k.weight", L.kw) && read_f32(tm, a + ".conv_k.bias", L.kb) &&
           read_f32(tm, a + ".conv_v.weight", L.vw) && read_f32(tm, a + ".conv_v.bias", L.vb) &&
           read_f32(tm, a + ".conv_o.weight", L.ow) && read_f32(tm, a + ".conv_o.bias", L.ob) &&
           read_f32(tm, a + ".emb_rel_k", L.rel_k) && read_f32(tm, a + ".emb_rel_v", L.rel_v) &&
           read_f32(tm, enc + ".norm_layers_1." + s + ".norm.weight", L.ln1_w) &&
           read_f32(tm, enc + ".norm_layers_1." + s + ".norm.bias", L.ln1_b) &&
           read_f32(tm, enc + ".norm_layers_2." + s + ".norm.weight", L.ln2_w) &&
           read_f32(tm, enc + ".norm_layers_2." + s + ".norm.bias", L.ln2_b) &&
           read_f32(tm, enc + ".ffn_layers." + s + ".conv_1.weight", L.ffn1_w) &&
           read_f32(tm, enc + ".ffn_layers." + s + ".conv_1.bias", L.ffn1_b) &&
           read_f32(tm, enc + ".ffn_layers." + s + ".conv_2.weight", L.ffn2_w) &&
           read_f32(tm, enc + ".ffn_layers." + s + ".conv_2.bias", L.ffn2_b);
}

static bool load_gst_cpu(core_gguf::tensor_map& tm, const std::string& p, st_gst_cpu& A) {
    return read_f32(tm, p + ".W_query.linear.weight", A.qw) && read_f32(tm, p + ".W_query.linear.bias", A.qb) &&
           read_f32(tm, p + ".W_key.linear.weight", A.kw) && read_f32(tm, p + ".W_key.linear.bias", A.kb) &&
           read_f32(tm, p + ".W_value.linear.weight", A.vw) && read_f32(tm, p + ".W_value.linear.bias", A.vb) &&
           read_f32(tm, p + ".out_fc.linear.weight", A.ow) && read_f32(tm, p + ".out_fc.linear.bias", A.ob);
}

static ggml_tensor* st_try(core_gguf::tensor_map& tm, const std::string& name) {
    auto it = tm.find(name);
    return it == tm.end() ? nullptr : it->second;
}

static bool bind_convnext_graph(core_gguf::tensor_map& tm, const std::string& p, ggml_tensor* w[9]) {
    w[0] = st_try(tm, p + ".dwconv.weight");
    if (!w[0]) { // vocoder names its dwconv "dwconv.net.*"
        w[0] = st_req(tm, p + ".dwconv.net.weight");
        w[1] = st_req(tm, p + ".dwconv.net.bias");
    } else {
        w[1] = st_req(tm, p + ".dwconv.bias");
    }
    w[2] = st_req(tm, p + ".norm.norm.weight");
    w[3] = st_req(tm, p + ".norm.norm.bias");
    w[4] = st_req(tm, p + ".pwconv1.weight");
    w[5] = st_req(tm, p + ".pwconv1.bias");
    w[6] = st_req(tm, p + ".pwconv2.weight");
    w[7] = st_req(tm, p + ".pwconv2.bias");
    w[8] = st_req(tm, p + ".gamma");
    for (int i = 0; i < 9; i++)
        if (!w[i])
            return false;
    return true;
}

struct supertonic_context_params supertonic_context_default_params(void) {
    supertonic_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.total_steps = 8;
    p.speed = 1.05f;
    snprintf(p.voice, sizeof(p.voice), "M1");
    snprintf(p.language, sizeof(p.language), "en");
    p.seed = 0;
    return p;
}

struct supertonic_context* supertonic_init_from_file(const char* path_model, supertonic_context_params params) {
    if (!path_model || !*path_model)
        return nullptr;
    auto* ctx = new supertonic_context();
    ctx->params = params;

    gguf_context* meta = core_gguf::open_metadata(path_model);
    if (!meta) {
        delete ctx;
        return nullptr;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    if (arch != "supertonic-tts") {
        std::fprintf(stderr, "supertonic: '%s' has arch '%s'\n", path_model, arch.c_str());
        core_gguf::free_metadata(meta);
        delete ctx;
        return nullptr;
    }
    ctx->sample_rate = (int)core_gguf::kv_u32(meta, "supertonic.sample_rate", 44100);
    ctx->base_chunk = (int)core_gguf::kv_u32(meta, "supertonic.base_chunk_size", 512);
    ctx->ccf = (int)core_gguf::kv_u32(meta, "supertonic.chunk_compress_factor", 6);
    ctx->ldim = (int)core_gguf::kv_u32(meta, "supertonic.latent_dim", 24);
    ctx->cfg_cond = core_gguf::kv_f32(meta, "supertonic.vf.cfg_cond", 4.0f);
    ctx->cfg_uncond = core_gguf::kv_f32(meta, "supertonic.vf.cfg_uncond", 3.0f);
    ctx->normalizer_scale = core_gguf::kv_f32(meta, "supertonic.vf.normalizer_scale", 0.25f);
    ctx->vf_text_attn_scale = core_gguf::kv_f32(meta, "supertonic.vf.text_attn_scale", 16.0f);
    ctx->vf_style_attn_scale = core_gguf::kv_f32(meta, "supertonic.vf.style_attn_scale", 16.0f);
    ctx->te_spte_attn_scale = core_gguf::kv_f32(meta, "supertonic.te.spte_attn_scale", 16.0f);
    ctx->dp_convnext_eps = core_gguf::kv_f32(meta, "supertonic.dp.convnext_ln_eps", 1e-6f);
    ctx->dp_attn_eps = core_gguf::kv_f32(meta, "supertonic.dp.attn_ln_eps", 1e-5f);
    ctx->te_convnext_eps = core_gguf::kv_f32(meta, "supertonic.te.convnext_ln_eps", 1e-6f);
    ctx->te_attn_eps = core_gguf::kv_f32(meta, "supertonic.te.attn_ln_eps", 1e-5f);
    ctx->te_spte_eps = core_gguf::kv_f32(meta, "supertonic.te.spte_ln_eps", 1e-5f);
    ctx->vf_convnext_eps = core_gguf::kv_f32(meta, "supertonic.vf.convnext_ln_eps", 1e-6f);
    ctx->vf_block_eps = core_gguf::kv_f32(meta, "supertonic.vf.block_ln_eps", 1e-6f);
    ctx->voc_convnext_eps = core_gguf::kv_f32(meta, "supertonic.voc.convnext_ln_eps", 1e-6f);
    ctx->voc_bn_eps = core_gguf::kv_f32(meta, "supertonic.voc.final_bn_eps", 1e-5f);
    core_gguf::free_metadata(meta);

    // backend
    ctx->backend_cpu = core_cpu_backend::init();
    ggml_backend_t gpu = nullptr;
    if (params.use_gpu) {
        gpu = crispasr_init_gpu_backend();
        if (gpu && core_cpu_backend::is_cpu(gpu)) {
            ggml_backend_free(gpu);
            gpu = nullptr;
        }
    }
    ctx->backend = gpu ? gpu : ctx->backend_cpu;
    if (!ctx->backend) {
        std::fprintf(stderr, "supertonic: backend init failed\n");
        delete ctx;
        return nullptr;
    }
    if (core_cpu_backend::is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, params.n_threads > 0 ? params.n_threads : 4);

    if (!core_gguf::load_weights(path_model, ctx->backend, "supertonic", ctx->weights)) {
        supertonic_free(ctx);
        return nullptr;
    }
    auto& tm = ctx->weights.tensors;

    // text tables
    std::vector<int32_t> keys, offs, cps, ccck, cccv;
    if (!read_i32(tm, "text.unicode_indexer", ctx->indexer) || !read_i32(tm, "text.nfkd_keys", keys) ||
        !read_i32(tm, "text.nfkd_offsets", offs) || !read_i32(tm, "text.nfkd_codepoints", cps) ||
        !read_i32(tm, "text.ccc_keys", ccck) || !read_i32(tm, "text.ccc_vals", cccv)) {
        std::fprintf(stderr, "supertonic: GGUF missing text tables\n");
        supertonic_free(ctx);
        return nullptr;
    }
    for (size_t i = 0; i < keys.size(); i++) {
        std::vector<uint32_t> dec;
        for (int32_t j = offs[i]; j < offs[i + 1]; j++)
            dec.push_back((uint32_t)cps[j]);
        ctx->nfkd[(uint32_t)keys[i]] = std::move(dec);
    }
    for (size_t i = 0; i < ccck.size(); i++)
        ctx->ccc[(uint32_t)ccck[i]] = cccv[i];

    // voices
    static const char* NAMES[] = {"F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"};
    for (const char* v : NAMES) {
        supertonic_context::Voice vc;
        if (read_f32(tm, std::string("voice.") + v + ".ttl", vc.ttl) &&
            read_f32(tm, std::string("voice.") + v + ".dp", vc.dp))
            ctx->voices[v] = std::move(vc);
    }
    if (ctx->voices.empty()) {
        std::fprintf(stderr, "supertonic: no voices in GGUF\n");
        supertonic_free(ctx);
        return nullptr;
    }

    bool ok = true;
    // duration predictor (CPU)
    ok = ok && read_f32(tm, "dp.sentence_encoder.text_embedder.char_embedder.weight", ctx->dp_char_emb);
    ok = ok && read_f32(tm, "dp.sentence_encoder.sentence_token", ctx->dp_sentence_token);
    for (int i = 0; i < 6 && ok; i++)
        ok = load_convnext_cpu(tm, "dp.sentence_encoder.convnext.convnext." + std::to_string(i), ctx->dp_cn[i], 1);
    for (int i = 0; i < 2 && ok; i++)
        ok = load_vits_layer(tm, "dp.sentence_encoder.attn_encoder", i, ctx->dp_attn[i]);
    ok = ok && read_f32(tm, "dp.sentence_encoder.proj_out.net.weight", ctx->dp_proj_w);
    ok = ok && read_f32(tm, "dp.predictor.layers.0.weight", ctx->dp_l0_w) &&
         read_f32(tm, "dp.predictor.layers.0.bias", ctx->dp_l0_b) &&
         read_f32(tm, "dp.predictor.layers.1.weight", ctx->dp_l1_w) &&
         read_f32(tm, "dp.predictor.layers.1.bias", ctx->dp_l1_b);
    {
        std::vector<float> a;
        ok = ok && read_f32(tm, "dp.predictor.activation.weight", a);
        if (ok)
            ctx->dp_prelu_a = a[0];
    }

    // text encoder (CPU)
    static const int TE_DIL[6] = {1, 1, 2, 2, 4, 4};
    ok = ok && read_f32(tm, "te.text_embedder.char_embedder.weight", ctx->te_char_emb);
    for (int i = 0; i < 6 && ok; i++)
        ok = load_convnext_cpu(tm, "te.convnext.convnext." + std::to_string(i), ctx->te_cn[i], TE_DIL[i]);
    for (int i = 0; i < 4 && ok; i++)
        ok = load_vits_layer(tm, "te.attn_encoder", i, ctx->te_attn[i]);
    ok = ok && read_f32(tm, "te.style_key", ctx->te_style_key);
    ok = ok && load_gst_cpu(tm, "te.spte.attention1", ctx->te_att1) &&
         load_gst_cpu(tm, "te.spte.attention2", ctx->te_att2);
    ok = ok && read_f32(tm, "te.spte.norm.norm.weight", ctx->te_norm_w) &&
         read_f32(tm, "te.spte.norm.norm.bias", ctx->te_norm_b);

    // vector field (graph tensors)
    ctx->vf_proj_in = st_req(tm, "vf.proj_in.net.weight");
    ctx->vf_proj_out = st_req(tm, "vf.proj_out.net.weight");
    ok = ok && ctx->vf_proj_in && ctx->vf_proj_out;
    static const int VF_DIL[6] = {1, 2, 4, 8, 1, 1};
    for (int b = 0; b < 4 && ok; b++) {
        st_vf_block& B = ctx->vf_blocks[b];
        const int base = b * 6;
        // module list per block: [cn(4)@base+0, linear@base+1, cn(1)@base+2,
        // attn@base+3, cn(1)@base+4, attention@base+5]
        for (int i = 0; i < 4 && ok; i++) {
            ok = bind_convnext_graph(tm, "vf.main_blocks." + std::to_string(base) + ".convnext." + std::to_string(i),
                                     B.cn[i]);
            B.cn_dil[i] = VF_DIL[i];
        }
        ok = ok && bind_convnext_graph(tm, "vf.main_blocks." + std::to_string(base + 2) + ".convnext.0", B.cn[4]);
        B.cn_dil[4] = 1;
        ok = ok && bind_convnext_graph(tm, "vf.main_blocks." + std::to_string(base + 4) + ".convnext.0", B.cn[5]);
        B.cn_dil[5] = 1;
        const std::string lp = "vf.main_blocks." + std::to_string(base + 1) + ".linear.linear.";
        B.time_w = st_req(tm, lp + "weight");
        B.time_b = st_req(tm, lp + "bias");
        const std::string ap = "vf.main_blocks." + std::to_string(base + 3) + ".attn.";
        B.attn_qw = st_req(tm, ap + "W_query.linear.weight");
        B.attn_qb = st_req(tm, ap + "W_query.linear.bias");
        B.attn_kw = st_req(tm, ap + "W_key.linear.weight");
        B.attn_kb = st_req(tm, ap + "W_key.linear.bias");
        B.attn_vw = st_req(tm, ap + "W_value.linear.weight");
        B.attn_vb = st_req(tm, ap + "W_value.linear.bias");
        B.attn_ow = st_req(tm, ap + "out_fc.linear.weight");
        B.attn_ob = st_req(tm, ap + "out_fc.linear.bias");
        const std::string an = "vf.main_blocks." + std::to_string(base + 3) + ".norm.norm.";
        B.attn_ln_w = st_req(tm, an + "weight");
        B.attn_ln_b = st_req(tm, an + "bias");
        const std::string sp = "vf.main_blocks." + std::to_string(base + 5) + ".attention.";
        B.st_qw = st_req(tm, sp + "W_query.linear.weight");
        B.st_qb = st_req(tm, sp + "W_query.linear.bias");
        B.st_kw = st_req(tm, sp + "W_key.linear.weight");
        B.st_kb = st_req(tm, sp + "W_key.linear.bias");
        B.st_vw = st_req(tm, sp + "W_value.linear.weight");
        B.st_vb = st_req(tm, sp + "W_value.linear.bias");
        B.st_ow = st_req(tm, sp + "out_fc.linear.weight");
        B.st_ob = st_req(tm, sp + "out_fc.linear.bias");
        const std::string sn = "vf.main_blocks." + std::to_string(base + 5) + ".norm.norm.";
        B.st_ln_w = st_req(tm, sn + "weight");
        B.st_ln_b = st_req(tm, sn + "bias");
        ok = ok && B.time_w && B.time_b && B.attn_qw && B.attn_qb && B.attn_kw && B.attn_kb && B.attn_vw && B.attn_vb &&
             B.attn_ow && B.attn_ob && B.attn_ln_w && B.attn_ln_b && B.st_qw && B.st_qb && B.st_kw && B.st_kb &&
             B.st_vw && B.st_vb && B.st_ow && B.st_ob && B.st_ln_w && B.st_ln_b;
    }
    for (int i = 0; i < 4 && ok; i++)
        ok = bind_convnext_graph(tm, "vf.last_convnext.convnext." + std::to_string(i), ctx->vf_last_cn[i]);
    ok = ok && read_f32(tm, "vf.time_freqs", ctx->vf_time_freqs) &&
         read_f32(tm, "vf.rotary_theta", ctx->vf_rotary_theta) &&
         read_f32(tm, "vf.time_encoder.mlp.0.linear.weight", ctx->vf_mlp0_w) &&
         read_f32(tm, "vf.time_encoder.mlp.0.linear.bias", ctx->vf_mlp0_b) &&
         read_f32(tm, "vf.time_encoder.mlp.2.linear.weight", ctx->vf_mlp2_w) &&
         read_f32(tm, "vf.time_encoder.mlp.2.linear.bias", ctx->vf_mlp2_b) &&
         read_f32(tm, "vf.uncond.text", ctx->vf_uncond_text) &&
         read_f32(tm, "vf.uncond.style_key", ctx->vf_uncond_sk) &&
         read_f32(tm, "vf.uncond.style_value", ctx->vf_uncond_sv);

    // vocoder
    ctx->voc_embed_w = st_req(tm, "voc.embed.net.weight");
    ctx->voc_embed_b = st_req(tm, "voc.embed.net.bias");
    for (int i = 0; i < 10 && ok; i++)
        ok = bind_convnext_graph(tm, "voc.convnext." + std::to_string(i), ctx->voc_cn[i]);
    ctx->voc_head1_w = st_req(tm, "voc.head.layer1.net.weight");
    ctx->voc_head1_b = st_req(tm, "voc.head.layer1.net.bias");
    ctx->voc_head2_w = st_req(tm, "voc.head.layer2.weight");
    ok = ok && ctx->voc_embed_w && ctx->voc_embed_b && ctx->voc_head1_w && ctx->voc_head1_b && ctx->voc_head2_w;
    {
        std::vector<float> bw, bb, bm, bv, pa;
        ok = ok && read_f32(tm, "voc.final_norm.norm.weight", bw) && read_f32(tm, "voc.final_norm.norm.bias", bb) &&
             read_f32(tm, "voc.final_norm.norm.running_mean", bm) &&
             read_f32(tm, "voc.final_norm.norm.running_var", bv) && read_f32(tm, "voc.head.act.weight", pa) &&
             read_f32(tm, "voc.latent_mean", ctx->voc_latent_mean) &&
             read_f32(tm, "voc.latent_std", ctx->voc_latent_std);
        if (ok) {
            ctx->voc_prelu_a = pa[0];
            ctx->voc_bn_scale.resize(bw.size());
            ctx->voc_bn_shift.resize(bw.size());
            for (size_t i = 0; i < bw.size(); i++) {
                ctx->voc_bn_scale[i] = bw[i] / sqrtf(bv[i] + ctx->voc_bn_eps);
                ctx->voc_bn_shift[i] = bb[i] - bm[i] * ctx->voc_bn_scale[i];
            }
        }
    }

    if (!ok) {
        std::fprintf(stderr, "supertonic: weight binding failed\n");
        supertonic_free(ctx);
        return nullptr;
    }

    if (params.seed) {
        ctx->rng.seed(params.seed);
        ctx->rng_seeded = true;
    } else {
        ctx->rng.seed(std::random_device{}());
    }
    if (params.verbosity >= 1)
        std::fprintf(stderr, "supertonic: loaded %s (backend %s, %d voices)\n", path_model,
                     ggml_backend_name(ctx->backend), (int)ctx->voices.size());
    return ctx;
}

void supertonic_free(struct supertonic_context* ctx) {
    if (!ctx)
        return;
    if (ctx->weights.buf)
        ggml_backend_buffer_free(ctx->weights.buf);
    if (ctx->weights.ctx)
        ggml_free(ctx->weights.ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

void supertonic_set_n_threads(struct supertonic_context* ctx, int n_threads) {
    if (ctx && core_cpu_backend::is_cpu(ctx->backend) && n_threads > 0)
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
}

int supertonic_set_voice(struct supertonic_context* ctx, const char* voice) {
    if (!ctx || !voice || ctx->voices.find(voice) == ctx->voices.end())
        return -1;
    snprintf(ctx->params.voice, sizeof(ctx->params.voice), "%s", voice);
    return 0;
}

int supertonic_set_language(struct supertonic_context* ctx, const char* lang) {
    static const char* LANGS[] = {"en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et",
                                  "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl",
                                  "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na"};
    if (!ctx || !lang)
        return -1;
    for (const char* l : LANGS)
        if (strcmp(l, lang) == 0) {
            snprintf(ctx->params.language, sizeof(ctx->params.language), "%s", lang);
            return 0;
        }
    return -1;
}

void supertonic_set_speed(struct supertonic_context* ctx, float speed) {
    if (ctx && speed > 0.1f && speed < 10.0f)
        ctx->params.speed = speed;
}

void supertonic_set_seed(struct supertonic_context* ctx, uint64_t seed) {
    if (!ctx)
        return;
    ctx->rng.seed(seed);
    ctx->rng_seeded = seed != 0;
}

void supertonic_set_total_steps(struct supertonic_context* ctx, int steps) {
    if (ctx && steps >= 1 && steps <= 128)
        ctx->params.total_steps = steps;
}

int supertonic_sample_rate(const struct supertonic_context* ctx) {
    return ctx ? ctx->sample_rate : 0;
}

// ── time embedding (CPU) ────────────────────────────────────────────────
static void st_time_emb(supertonic_context* ctx, float t, std::vector<float>& out64) {
    const float x = t * 1000.0f;
    float e[64];
    for (int i = 0; i < 32; i++) {
        float a = x * ctx->vf_time_freqs[i];
        e[i] = sinf(a);
        e[32 + i] = cosf(a);
    }
    std::vector<float> h(256);
    cpu_linear(e, ctx->vf_mlp0_w.data(), ctx->vf_mlp0_b.data(), 1, 64, 256, h.data());
    for (auto& v : h) { // Mish = x * tanh(softplus(x))
        float sp = v > 20.f ? v : log1pf(expf(v));
        v = v * tanhf(sp);
    }
    out64.resize(64);
    cpu_linear(h.data(), ctx->vf_mlp2_w.data(), ctx->vf_mlp2_b.data(), 1, 256, 64, out64.data());
}

// ── flow (graph) ────────────────────────────────────────────────────────
// text_emb (L,256) row-major; xt in/out (N,144) row-major == (144,N) ne.

static bool st_run_flow(supertonic_context* ctx, const std::vector<float>& text_emb, int L, std::vector<float>& xt,
                        int N, int steps, float* out_all_steps, float* out_projin_s0) {
    supertonic_bench_stage _b("vector_field");
    const size_t n_ctx_tensors = 8192;
    ggml_init_params ip = {n_ctx_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false), nullptr,
                           true};
    ggml_context* g = ggml_init(ip);
    st_vf_inputs in{};
    in.xt = ggml_new_tensor_2d(g, GGML_TYPE_F32, 144, N);
    in.t_emb = ggml_new_tensor_2d(g, GGML_TYPE_F32, 64, 1);
    in.cos_q = ggml_new_tensor_3d(g, GGML_TYPE_F32, 32, N, 1);
    in.sin_q = ggml_new_tensor_3d(g, GGML_TYPE_F32, 32, N, 1);
    in.cos_k = ggml_new_tensor_3d(g, GGML_TYPE_F32, 32, L, 1);
    in.sin_k = ggml_new_tensor_3d(g, GGML_TYPE_F32, 32, L, 1);
    in.text_c = ggml_new_tensor_2d(g, GGML_TYPE_F32, 256, L);
    in.text_u = ggml_new_tensor_2d(g, GGML_TYPE_F32, 256, L);
    in.sk = ggml_new_tensor_2d(g, GGML_TYPE_F32, 256, 50);
    in.sv = ggml_new_tensor_2d(g, GGML_TYPE_F32, 256, 50);
    in.usk = ggml_new_tensor_2d(g, GGML_TYPE_F32, 256, 50);
    in.usv = ggml_new_tensor_2d(g, GGML_TYPE_F32, 256, 50);
    ggml_tensor* all[] = {in.xt,     in.t_emb,  in.cos_q, in.sin_q, in.cos_k, in.sin_k,
                          in.text_c, in.text_u, in.sk,    in.sv,    in.usk,   in.usv};
    for (auto* t : all)
        ggml_set_input(t);

    ggml_tensor* projin = nullptr;
    ggml_tensor* v_c = g_vf_half(ctx, g, in.xt, in.t_emb, in.text_c, in.sk, in.sv, in.cos_q, in.sin_q, in.cos_k,
                                 in.sin_k, out_projin_s0 ? &projin : nullptr);
    ggml_tensor* v_u =
        g_vf_half(ctx, g, in.xt, in.t_emb, in.text_u, in.usk, in.usv, in.cos_q, in.sin_q, in.cos_k, in.sin_k, nullptr);
    ggml_tensor* v = ggml_sub(g, ggml_scale(g, v_c, ctx->cfg_cond), ggml_scale(g, v_u, ctx->cfg_uncond));
    ggml_tensor* out = ggml_add(g, in.xt, ggml_scale(g, v, 1.0f / (float)steps));
    ggml_set_name(out, "xt_next");
    ggml_set_output(out);
    if (projin) {
        ggml_set_name(projin, "projin_c");
        ggml_set_output(projin);
    }
    ggml_cgraph* gf = ggml_new_graph_custom(g, 8192, false);
    ggml_build_forward_expand(gf, out);
    if (projin)
        ggml_build_forward_expand(gf, projin);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "supertonic: vf graph alloc failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        return false;
    }

    // constant-per-synthesis host buffers
    std::vector<float> cq((size_t)32 * N), sq((size_t)32 * N), ck((size_t)32 * L), sk_((size_t)32 * L);
    for (int t = 0; t < N; t++)
        for (int i = 0; i < 32; i++) {
            float a = ((float)t / (float)N) * ctx->vf_rotary_theta[i];
            cq[(size_t)t * 32 + i] = cosf(a);
            sq[(size_t)t * 32 + i] = sinf(a);
        }
    for (int t = 0; t < L; t++)
        for (int i = 0; i < 32; i++) {
            float a = ((float)t / (float)L) * ctx->vf_rotary_theta[i];
            ck[(size_t)t * 32 + i] = cosf(a);
            sk_[(size_t)t * 32 + i] = sinf(a);
        }
    std::vector<float> text_u((size_t)L * 256);
    for (int t = 0; t < L; t++)
        memcpy(&text_u[(size_t)t * 256], ctx->vf_uncond_text.data(), 256 * sizeof(float));
    const auto& voice = ctx->voices[ctx->params.voice];

    for (int step = 0; step < steps; step++) {
        std::vector<float> temb;
        st_time_emb(ctx, (float)step / (float)steps, temb);
        // §234: re-set ALL inputs on every compute — gallocr reuses input
        // buffers as scratch after their last read.
        ggml_backend_tensor_set(in.xt, xt.data(), 0, xt.size() * sizeof(float));
        ggml_backend_tensor_set(in.t_emb, temb.data(), 0, 64 * sizeof(float));
        ggml_backend_tensor_set(in.cos_q, cq.data(), 0, cq.size() * sizeof(float));
        ggml_backend_tensor_set(in.sin_q, sq.data(), 0, sq.size() * sizeof(float));
        ggml_backend_tensor_set(in.cos_k, ck.data(), 0, ck.size() * sizeof(float));
        ggml_backend_tensor_set(in.sin_k, sk_.data(), 0, sk_.size() * sizeof(float));
        ggml_backend_tensor_set(in.text_c, text_emb.data(), 0, text_emb.size() * sizeof(float));
        ggml_backend_tensor_set(in.text_u, text_u.data(), 0, text_u.size() * sizeof(float));
        ggml_backend_tensor_set(in.sk, ctx->te_style_key.data(), 0, ctx->te_style_key.size() * sizeof(float));
        ggml_backend_tensor_set(in.sv, voice.ttl.data(), 0, voice.ttl.size() * sizeof(float));
        ggml_backend_tensor_set(in.usk, ctx->vf_uncond_sk.data(), 0, ctx->vf_uncond_sk.size() * sizeof(float));
        ggml_backend_tensor_set(in.usv, ctx->vf_uncond_sv.data(), 0, ctx->vf_uncond_sv.size() * sizeof(float));
        if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "supertonic: vf compute failed\n");
            ggml_gallocr_free(alloc);
            ggml_free(g);
            return false;
        }
        if (step == 0 && projin && out_projin_s0)
            ggml_backend_tensor_get(projin, out_projin_s0, 0, (size_t)512 * N * sizeof(float));
        ggml_backend_tensor_get(out, xt.data(), 0, xt.size() * sizeof(float));
        if (out_all_steps)
            memcpy(out_all_steps + (size_t)step * 144 * N, xt.data(), (size_t)144 * N * sizeof(float));
    }
    ggml_gallocr_free(alloc);
    ggml_free(g);
    return true;
}

// ── vocoder (graph) ─────────────────────────────────────────────────────
// latent (N,144) row-major -> wav (N*6*512).

static bool st_run_vocoder(supertonic_context* ctx, const std::vector<float>& latent, int N, std::vector<float>& wav) {
    supertonic_bench_stage _b("vocoder");
    const int T6 = N * ctx->ccf;
    ggml_init_params ip = {4096 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true};
    ggml_context* g = ggml_init(ip);
    ggml_tensor* lat = ggml_new_tensor_2d(g, GGML_TYPE_F32, 144, N);
    ggml_tensor* mean = ggml_new_tensor_2d(g, GGML_TYPE_F32, 24, 1);
    ggml_tensor* stdv = ggml_new_tensor_2d(g, GGML_TYPE_F32, 24, 1);
    ggml_tensor* bnsc = ggml_new_tensor_2d(g, GGML_TYPE_F32, 512, 1);
    ggml_tensor* bnsh = ggml_new_tensor_2d(g, GGML_TYPE_F32, 512, 1);
    for (auto* t : {lat, mean, stdv, bnsc, bnsh})
        ggml_set_input(t);

    // latent / normalizer_scale, decompress (144,N) -> (24, 6N)
    ggml_tensor* x = ggml_scale(g, lat, 1.0f / ctx->normalizer_scale);
    x = ggml_reshape_3d(g, x, 6, 24, N);              // (sub, ch, l)
    x = ggml_cont(g, ggml_permute(g, x, 1, 0, 2, 3)); // (ch, sub, l)
    x = ggml_reshape_2d(g, x, 24, T6);                // (24, 6N)
    x = ggml_add(g, ggml_mul(g, x, stdv), mean);
    // embed conv k7 EDGE pad 3: (T,C) layout
    ggml_tensor* xt = ggml_cont(g, ggml_transpose(g, x)); // (T6,24)
    xt = g_edge_pad(g, xt, 3);
    ggml_tensor* h = ggml_conv_1d(g, ctx->voc_embed_w, xt, 1, 0, 1); // (T6,512)
    h = ggml_add(g, h, ggml_reshape_2d(g, ctx->voc_embed_b, 1, 512));
    h = ggml_cont(g, ggml_transpose(g, h)); // (512,T6)
    for (int i = 0; i < 10; i++)
        h = g_convnext(g, h, ctx->voc_cn[i], ctx->voc_cn_dil[i], ctx->voc_convnext_eps);
    // folded BatchNorm
    h = ggml_add(g, ggml_mul(g, h, bnsc), bnsh);
    // head layer1 k3 EDGE pad 1
    ggml_tensor* ht = ggml_cont(g, ggml_transpose(g, h)); // (T6,512)
    ht = g_edge_pad(g, ht, 1);
    ggml_tensor* y = ggml_conv_1d(g, ctx->voc_head1_w, ht, 1, 0, 1); // (T6,2048)
    y = ggml_add(g, y, ggml_reshape_2d(g, ctx->voc_head1_b, 1, 2048));
    y = ggml_cont(g, ggml_transpose(g, y)); // (2048,T6)
    y = ggml_leaky_relu(g, y, ctx->voc_prelu_a, false);
    y = ggml_mul_mat(g, ctx->voc_head2_w, y); // (512,T6)
    ggml_tensor* out = ggml_reshape_1d(g, y, (int64_t)512 * T6);
    ggml_set_output(out);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 4096, false);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "supertonic: vocoder graph alloc failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        return false;
    }
    ggml_backend_tensor_set(lat, latent.data(), 0, latent.size() * sizeof(float));
    ggml_backend_tensor_set(mean, ctx->voc_latent_mean.data(), 0, 24 * sizeof(float));
    ggml_backend_tensor_set(stdv, ctx->voc_latent_std.data(), 0, 24 * sizeof(float));
    ggml_backend_tensor_set(bnsc, ctx->voc_bn_scale.data(), 0, 512 * sizeof(float));
    ggml_backend_tensor_set(bnsh, ctx->voc_bn_shift.data(), 0, 512 * sizeof(float));
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "supertonic: vocoder compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        return false;
    }
    wav.resize((size_t)512 * T6);
    ggml_backend_tensor_get(out, wav.data(), 0, wav.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(g);
    return true;
}

// ── chunking + synthesis ────────────────────────────────────────────────

static bool st_is_abbrev_before(const std::vector<uint32_t>& w, size_t dot) {
    // mirrors the upstream lookbehind list (plus single capital letter)
    static const char* ABBR[] = {"Mr.",  "Mrs.", "Ms.",  "Dr.",  "Prof.", "Sr.",   "Jr.", "Ph.D.", "etc.", "e.g.",
                                 "i.e.", "vs.",  "Inc.", "Ltd.", "Co.",   "Corp.", "St.", "Ave.",  "Blvd."};
    for (const char* a : ABBR) {
        size_t n = strlen(a);
        if (dot + 1 >= n) {
            bool m = true;
            for (size_t k = 0; k < n; k++)
                if (w[dot + 1 - n + k] != (uint32_t)a[k]) {
                    m = false;
                    break;
                }
            if (m)
                return true;
        }
    }
    // single capital letter like "F."
    if (dot >= 1 && w[dot - 1] >= 'A' && w[dot - 1] <= 'Z' && (dot == 1 || !isalnum((int)w[dot - 2])))
        return true;
    return false;
}

static std::vector<std::string> st_chunk_text(const char* text, int max_len) {
    std::vector<uint32_t> cps = utf8_decode(text);
    // split into sentences at [.!?] + space (with abbreviation guard)
    std::vector<std::vector<uint32_t>> sentences;
    std::vector<uint32_t> cur;
    for (size_t i = 0; i < cps.size(); i++) {
        cur.push_back(cps[i]);
        if ((cps[i] == '.' || cps[i] == '!' || cps[i] == '?') && i + 1 < cps.size() && st_is_space(cps[i + 1]) &&
            !(cps[i] == '.' && st_is_abbrev_before(cur, cur.size() - 1))) {
            sentences.push_back(cur);
            cur.clear();
            while (i + 1 < cps.size() && st_is_space(cps[i + 1]))
                i++;
        }
    }
    if (!cur.empty())
        sentences.push_back(cur);
    // greedy pack
    std::vector<std::string> chunks;
    std::string acc;
    auto cp_len = [](const std::vector<uint32_t>& v) { return (int)v.size(); };
    auto encode = [](const std::vector<uint32_t>& v) {
        std::string s;
        for (uint32_t c : v) {
            if (c < 0x80)
                s += (char)c;
            else if (c < 0x800) {
                s += (char)(0xC0 | (c >> 6));
                s += (char)(0x80 | (c & 0x3F));
            } else if (c < 0x10000) {
                s += (char)(0xE0 | (c >> 12));
                s += (char)(0x80 | ((c >> 6) & 0x3F));
                s += (char)(0x80 | (c & 0x3F));
            } else {
                s += (char)(0xF0 | (c >> 18));
                s += (char)(0x80 | ((c >> 12) & 0x3F));
                s += (char)(0x80 | ((c >> 6) & 0x3F));
                s += (char)(0x80 | (c & 0x3F));
            }
        }
        return s;
    };
    int acc_len = 0;
    for (auto& s : sentences) {
        if (acc_len > 0 && acc_len + 1 + cp_len(s) > max_len) {
            chunks.push_back(acc);
            acc.clear();
            acc_len = 0;
        }
        if (acc_len > 0) {
            acc += " ";
            acc_len += 1;
        }
        acc += encode(s);
        acc_len += cp_len(s);
    }
    if (!acc.empty())
        chunks.push_back(acc);
    if (chunks.empty())
        chunks.push_back(text);
    return chunks;
}

// one chunk end-to-end; appends trimmed samples to out.
static bool st_synthesize_chunk(supertonic_context* ctx, const char* text, double* out_dur, std::vector<float>& out) {
    const auto it = ctx->voices.find(ctx->params.voice);
    if (it == ctx->voices.end())
        return false;
    const auto& voice = it->second;
    std::vector<uint32_t> cps = st_preprocess(ctx, text, ctx->params.language);
    std::vector<int32_t> ids = st_ids_from_cps(ctx, cps);
    const int L = (int)ids.size();
    if (L == 0)
        return false;
    float dur = st_dp_forward(ctx, ids.data(), L, voice.dp) / ctx->params.speed;
    std::vector<float> text_emb;
    st_te_forward(ctx, ids.data(), L, voice.ttl, text_emb, nullptr, nullptr);
    const int chunk = ctx->base_chunk * ctx->ccf;
    const double wav_len = (double)dur * ctx->sample_rate;
    const int N = (int)((wav_len + chunk - 1) / chunk);
    if (N <= 0)
        return false;
    std::vector<float> xt((size_t)144 * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : xt)
        v = nd(ctx->rng);
    if (!st_run_flow(ctx, text_emb, L, xt, N, ctx->params.total_steps, nullptr, nullptr))
        return false;
    std::vector<float> wav;
    if (!st_run_vocoder(ctx, xt, N, wav))
        return false;
    int n = (int)(ctx->sample_rate * dur);
    n = std::min(n, (int)wav.size());
    out.insert(out.end(), wav.begin(), wav.begin() + n);
    *out_dur = dur;
    return true;
}

float* supertonic_synthesize(struct supertonic_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;
    const bool ko_ja = strcmp(ctx->params.language, "ko") == 0 || strcmp(ctx->params.language, "ja") == 0;
    std::vector<std::string> chunks = st_chunk_text(text, ko_ja ? 120 : 300);
    std::vector<float> pcm;
    const int silence = (int)(0.3 * ctx->sample_rate);
    for (size_t i = 0; i < chunks.size(); i++) {
        if (i > 0)
            pcm.insert(pcm.end(), silence, 0.0f);
        double dur = 0;
        if (!st_synthesize_chunk(ctx, chunks[i].c_str(), &dur, pcm))
            return nullptr;
        if (ctx->params.verbosity >= 2)
            std::fprintf(stderr, "supertonic: chunk %zu/%zu dur %.2fs\n", i + 1, chunks.size(), dur);
    }
    float* res = (float*)malloc(pcm.size() * sizeof(float));
    if (!res)
        return nullptr;
    memcpy(res, pcm.data(), pcm.size() * sizeof(float));
    *out_n_samples = (int)pcm.size();
    return res;
}

// ── diff-harness stage APIs ─────────────────────────────────────────────

int supertonic_debug_text_ids(struct supertonic_context* ctx, const char* text, const char* lang, int32_t* out_ids,
                              int cap) {
    if (!ctx || !text || !lang)
        return -1;
    std::vector<uint32_t> cps = st_preprocess(ctx, text, lang);
    std::vector<int32_t> ids = st_ids_from_cps(ctx, cps);
    const int n = (int)ids.size();
    if (out_ids)
        memcpy(out_ids, ids.data(), (size_t)std::min(n, cap) * sizeof(int32_t));
    return n;
}

float supertonic_debug_duration(struct supertonic_context* ctx, const int32_t* ids, int n) {
    if (!ctx || !ids || n <= 0)
        return -1.0f;
    const auto& voice = ctx->voices[ctx->params.voice];
    return st_dp_forward(ctx, ids, n, voice.dp) / ctx->params.speed;
}

int supertonic_debug_text_encoder(struct supertonic_context* ctx, const int32_t* ids, int n, float* out_text_emb,
                                  float* out_convnext, float* out_pre_spte) {
    if (!ctx || !ids || n <= 0 || !out_text_emb)
        return -1;
    const auto& voice = ctx->voices[ctx->params.voice];
    std::vector<float> emb, cn, pre;
    st_te_forward(ctx, ids, n, voice.ttl, emb, out_convnext ? &cn : nullptr, out_pre_spte ? &pre : nullptr);
    memcpy(out_text_emb, emb.data(), emb.size() * sizeof(float));
    if (out_convnext)
        memcpy(out_convnext, cn.data(), cn.size() * sizeof(float));
    if (out_pre_spte)
        memcpy(out_pre_spte, pre.data(), pre.size() * sizeof(float));
    return 0;
}

int supertonic_debug_flow(struct supertonic_context* ctx, const float* text_emb, int n_text, const float* xt0,
                          int n_latent, float* out_steps, float* out_projin_s0) {
    if (!ctx || !text_emb || !xt0 || !out_steps)
        return -1;
    std::vector<float> emb(text_emb, text_emb + (size_t)n_text * 256);
    std::vector<float> xt(xt0, xt0 + (size_t)n_latent * 144);
    if (!st_run_flow(ctx, emb, n_text, xt, n_latent, ctx->params.total_steps, out_steps, out_projin_s0))
        return -1;
    return 0;
}

int supertonic_debug_vocoder(struct supertonic_context* ctx, const float* latent, int n_latent, float* out_wav,
                             int cap) {
    if (!ctx || !latent || !out_wav)
        return -1;
    std::vector<float> lat(latent, latent + (size_t)n_latent * 144);
    std::vector<float> wav;
    if (!st_run_vocoder(ctx, lat, n_latent, wav))
        return -1;
    const int n = std::min((int)wav.size(), cap);
    memcpy(out_wav, wav.data(), (size_t)n * sizeof(float));
    return n;
}

// ── crispasr-diff: self-contained per-stage parity vs the ONNX reference ─
// ref GGUF from tools/reference_backends/supertonic_tts.py. Compares every
// stage with cosine AND |mine|/|ref| magnitudes (HARD RULE #2b).

static void st_stage_stats(const char* name, const float* a, const float* b, size_t n, bool& all_ok) {
    double dot = 0, na = 0, nb = 0, max_abs = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
        max_abs = std::max(max_abs, (double)fabsf(a[i] - b[i]));
    }
    const double cosv = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : (na == nb ? 1.0 : 0.0);
    const double ra = sqrt(na / (double)n), rb = sqrt(nb / (double)n);
    const bool ok = cosv > 0.995 && (ra < 1e-6 ? rb < 1e-6 : (rb / ra > 0.98 && rb / ra < 1.02));
    all_ok = all_ok && ok;
    std::printf("  %-14s n=%8zu cos=%.6f |mine|=%.6f |ref|=%.6f max_abs=%.5f  %s\n", name, n, cosv, ra, rb, max_abs,
                ok ? "PASS" : "FAIL");
}

extern "C" int supertonic_tts_diff(const char* model_path, const char* ref_path, int verbosity) {
    // load reference stages
    gguf_context* rm = core_gguf::open_metadata(ref_path);
    if (!rm)
        return 1;
    const std::string text = core_gguf::kv_str(rm, "supertonic_ref.text", "");
    const std::string lang = core_gguf::kv_str(rm, "supertonic_ref.lang", "en");
    const std::string voice = core_gguf::kv_str(rm, "supertonic_ref.voice", "M1");
    const int steps = (int)core_gguf::kv_u32(rm, "supertonic_ref.steps", 8);
    const float speed = core_gguf::kv_f32(rm, "supertonic_ref.speed", 1.05f);
    core_gguf::free_metadata(rm);

    ggml_backend_t rbe = core_cpu_backend::init();
    core_gguf::WeightLoad ref;
    if (!core_gguf::load_weights(ref_path, rbe, "supertonic-ref", ref))
        return 1;
    auto rget = [&](const char* n, std::vector<float>& v) { return read_f32(ref.tensors, n, v); };
    std::vector<int32_t> ref_ids;
    read_i32(ref.tensors, "text_ids", ref_ids);

    supertonic_context_params p = supertonic_context_default_params();
    p.total_steps = steps;
    p.speed = speed;
    p.verbosity = verbosity;
    supertonic_context* ctx = supertonic_init_from_file(model_path, p);
    if (!ctx)
        return 1;
    supertonic_set_voice(ctx, voice.c_str());
    supertonic_set_language(ctx, lang.c_str());

    bool all_ok = true;
    std::printf("supertonic-tts diff: text=\"%s\" lang=%s voice=%s steps=%d\n", text.c_str(), lang.c_str(),
                voice.c_str(), steps);

    // 1. text ids (must be BYTE-identical — a text-processing mismatch makes
    //    every later stage incomparable)
    std::vector<int32_t> ids(4096);
    int L = supertonic_debug_text_ids(ctx, text.c_str(), lang.c_str(), ids.data(), (int)ids.size());
    ids.resize(L > 0 ? L : 0);
    {
        bool same = (L == (int)ref_ids.size());
        int first_bad = -1;
        if (same)
            for (int i = 0; i < L; i++)
                if (ids[i] != ref_ids[i]) {
                    same = false;
                    first_bad = i;
                    break;
                }
        std::printf("  %-14s n=%8d %s%s\n", "text_ids", L, same ? "byte-identical  PASS" : "MISMATCH  FAIL",
                    first_bad >= 0 ? (" (first at " + std::to_string(first_bad) + ")").c_str() : "");
        if (!same) {
            std::printf("    mine n=%d ref n=%d\n", L, (int)ref_ids.size());
            all_ok = false;
        }
    }
    // continue with the REFERENCE ids so later stages stay comparable
    const std::vector<int32_t>& use_ids = ref_ids.empty() ? ids : ref_ids;
    const int n_ids = (int)use_ids.size();

    // 2. duration
    std::vector<float> ref_dur;
    if (rget("dur", ref_dur)) {
        float d = supertonic_debug_duration(ctx, use_ids.data(), n_ids);
        bool ok = fabsf(d - ref_dur[0]) < 0.005f * std::max(1.0f, ref_dur[0]);
        all_ok = all_ok && ok;
        std::printf("  %-14s mine=%.5fs ref=%.5fs  %s\n", "dur", d, ref_dur[0], ok ? "PASS" : "FAIL");
    }

    // 3. text encoder (+ interior probes)
    std::vector<float> te((size_t)n_ids * 256), te_cn((size_t)n_ids * 256), te_pre((size_t)n_ids * 256);
    supertonic_debug_text_encoder(ctx, use_ids.data(), n_ids, te.data(), te_cn.data(), te_pre.data());
    std::vector<float> r;
    if (rget("te_convnext", r))
        st_stage_stats("te_convnext", te_cn.data(), r.data(), r.size(), all_ok);
    if (rget("te_pre_spte", r))
        st_stage_stats("te_pre_spte", te_pre.data(), r.data(), r.size(), all_ok);
    std::vector<float> ref_te;
    if (rget("text_emb", ref_te))
        st_stage_stats("text_emb", te.data(), ref_te.data(), ref_te.size(), all_ok);

    // 4. flow with injected noise — driven from the REFERENCE text_emb so a
    //    text-encoder drift doesn't compound into the flow verdict
    std::vector<float> xt0;
    if (rget("xt0", xt0)) {
        const int N = (int)(xt0.size() / 144);
        std::vector<float> allsteps((size_t)steps * 144 * N);
        std::vector<float> projin((size_t)512 * N);
        const std::vector<float>& drive = ref_te.empty() ? te : ref_te;
        if (supertonic_debug_flow(ctx, drive.data(), n_ids, xt0.data(), N, allsteps.data(), projin.data()) == 0) {
            if (rget("vf_projin_s0", r))
                st_stage_stats("vf_projin_s0", projin.data(), r.data(), r.size(), all_ok);
            for (int s = 1; s <= steps; s++) {
                const std::string nm = "xt_" + std::to_string(s);
                if (rget(nm.c_str(), r))
                    st_stage_stats(nm.c_str(), allsteps.data() + (size_t)(s - 1) * 144 * N, r.data(), r.size(), all_ok);
            }
            // 5. vocoder — driven from the REFERENCE final latent
            std::vector<float> lat_ref;
            const std::string last = "xt_" + std::to_string(steps);
            if (rget(last.c_str(), lat_ref) && rget("audio", r)) {
                std::vector<float> wav((size_t)512 * 6 * N);
                int n = supertonic_debug_vocoder(ctx, lat_ref.data(), N, wav.data(), (int)wav.size());
                if (n > 0)
                    st_stage_stats("audio", wav.data(), r.data(), std::min((size_t)n, r.size()), all_ok);
                else
                    all_ok = false;
            }
        } else {
            std::printf("  flow FAILED to run\n");
            all_ok = false;
        }
    }

    supertonic_free(ctx);
    if (ref.buf)
        ggml_backend_buffer_free(ref.buf);
    if (ref.ctx)
        ggml_free(ref.ctx);
    ggml_backend_free(rbe);
    std::printf("supertonic-tts diff: %s\n", all_ok ? "ALL PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
