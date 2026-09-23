// xasr.cpp — X-ASR (icefall streaming Zipformer2 transducer) runtime, #436.
// See xasr.h and docs/xasr/PLAN.md. Graph layout: activations are (C, T) with
// C the fast axis (ggml ne[0]); conv feature maps are (freq, time, channel).
//
// One encoder graph is built per context (its shapes depend only on the chunk
// configuration) and re-run for every chunk; the Zipformer caches go in as
// graph inputs and come back as outputs. The decoder and joiner are a handful
// of mat-vecs per frame and run on the host from F32 copies of their weights.

#include "xasr.h"

#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/kaldi_fbank.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kGraphNodes = 32768;

struct xasr_hparams {
    std::vector<int> n_layers, downsample, ffn_dim, n_heads, dims, qd, vd, pd, kernel;
    int pos_dim = 48, decoder_dim = 512, joiner_dim = 512, context_size = 2, vocab = 0, blank_id = 0, feature_dim = 80;
    int unk_id = -1;
    std::vector<int> chunk_ms_table, left_ctx_table;
};

struct conv_mod {
    ggml_tensor *in_w, *in_b, *causal_w, *causal_b, *chunk_w, *chunk_b, *chunk_scale, *out_w, *out_b;
};

struct zlayer {
    ggml_tensor *aw_in_w, *aw_in_b, *aw_pos_w;
    ggml_tensor *sa1_in_w, *sa1_in_b, *sa1_out_w, *sa1_out_b, *sa2_in_w, *sa2_in_b, *sa2_out_w, *sa2_out_b;
    ggml_tensor *ff_in_w[3], *ff_in_b[3], *ff_out_w[3], *ff_out_b[3];
    ggml_tensor *na_in_w, *na_in_b, *na_out_w, *na_out_b;
    conv_mod cv[2];
    ggml_tensor *norm_bias, *bypass, *bypass_mid;
    float norm_scale = 1.0f; // exp(norm.log_scale)
};

struct zstack {
    std::vector<zlayer> layers;
    std::vector<float> ds_w; // softmax(ds_bias); empty when downsample == 1
    ggml_tensor* combiner = nullptr;
};

struct xasr_model {
    xasr_hparams hp;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    core_gguf::tensor_map tensors;
    std::vector<std::string> vocab;
    // encoder_embed
    ggml_tensor *conv_w[3], *conv_b[3], *cnx_dw_w, *cnx_dw_b, *cnx_pw1_w, *cnx_pw1_b, *cnx_pw2_w, *cnx_pw2_b;
    ggml_tensor *emb_out_w, *emb_out_b, *emb_norm_bias;
    float emb_norm_scale = 1.0f;
    std::vector<zstack> stacks;
    std::vector<float> out_ds_w;
    ggml_tensor *enc_proj_w, *enc_proj_b;
    // host copies: decoder + joiner
    std::vector<float> dec_emb, dec_conv, dec_proj_w, dec_proj_b, join_out_w, join_out_b;
};

// Chunk geometry, fixed per context.
struct geom {
    int chunk_ms = 480, decode_chunk_len = 48, T = 61, chunk = 24, left = 256, tail_pad_ms = 1610;
};

} // namespace

struct xasr_context {
    xasr_context_params params;
    xasr_model model;
    geom g;
    ggml_backend_t backend = nullptr, backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

namespace {

ggml_tensor* req(xasr_model& m, const std::string& n) {
    return core_gguf::require(m.tensors, n.c_str(), "xasr");
}

std::vector<float> to_f32(ggml_tensor* t) {
    std::vector<float> out((size_t)ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        ggml_get_type_traits(t->type)->to_float(raw.data(), out.data(), (int64_t)out.size());
    }
    return out;
}

std::vector<int> kv_int_array(gguf_context* g, const char* key) {
    std::vector<int> out;
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY)
        return out;
    const size_t n = gguf_get_arr_n(g, id);
    const enum gguf_type t = gguf_get_arr_type(g, id);
    const void* d = gguf_get_arr_data(g, id);
    for (size_t i = 0; i < n; i++) {
        if (t == GGUF_TYPE_INT32)
            out.push_back(((const int32_t*)d)[i]);
        else if (t == GGUF_TYPE_UINT32)
            out.push_back((int)((const uint32_t*)d)[i]);
        else if (t == GGUF_TYPE_INT64)
            out.push_back((int)((const int64_t*)d)[i]);
        else
            return {};
    }
    return out;
}

std::vector<float> softmax_host(const std::vector<float>& b) {
    std::vector<float> w(b.size());
    const float mx = *std::max_element(b.begin(), b.end());
    float s = 0.0f;
    for (size_t i = 0; i < b.size(); i++)
        s += (w[i] = std::exp(b[i] - mx));
    for (auto& v : w)
        v /= s;
    return w;
}

bool load(xasr_model& m, const char* path, ggml_backend_t backend) {
    gguf_context* g = core_gguf::open_metadata(path);
    if (!g)
        return false;
    auto& hp = m.hp;
    hp.n_layers = kv_int_array(g, "xasr.n_layers");
    hp.downsample = kv_int_array(g, "xasr.downsample");
    hp.ffn_dim = kv_int_array(g, "xasr.ffn_dim");
    hp.n_heads = kv_int_array(g, "xasr.n_heads");
    hp.dims = kv_int_array(g, "xasr.dims");
    hp.qd = kv_int_array(g, "xasr.query_head_dim");
    hp.vd = kv_int_array(g, "xasr.value_head_dim");
    hp.pd = kv_int_array(g, "xasr.pos_head_dim");
    hp.kernel = kv_int_array(g, "xasr.conv_kernel");
    hp.chunk_ms_table = kv_int_array(g, "xasr.chunk_ms");
    hp.left_ctx_table = kv_int_array(g, "xasr.left_context_frames");
    hp.pos_dim = (int)core_gguf::kv_u32(g, "xasr.pos_dim", 48);
    hp.decoder_dim = (int)core_gguf::kv_u32(g, "xasr.decoder_dim", 512);
    hp.joiner_dim = (int)core_gguf::kv_u32(g, "xasr.joiner_dim", 512);
    hp.context_size = (int)core_gguf::kv_u32(g, "xasr.context_size", 2);
    hp.vocab = (int)core_gguf::kv_u32(g, "xasr.vocab_size", 0);
    hp.blank_id = (int)core_gguf::kv_u32(g, "xasr.blank_id", 0);
    hp.feature_dim = (int)core_gguf::kv_u32(g, "xasr.feature_dim", 80);
    const uint32_t unk = core_gguf::kv_u32(g, "xasr.unk_id", 0xFFFFFFFFu);
    hp.unk_id = unk == 0xFFFFFFFFu ? -1 : (int)unk;
    m.vocab = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    core_gguf::free_metadata(g);
    const size_t S = hp.dims.size();
    if (S == 0 || hp.n_layers.size() != S || hp.downsample.size() != S || hp.n_heads.size() != S || hp.qd.size() != S ||
        hp.vd.size() != S || hp.pd.size() != S || hp.kernel.size() != S || (int)m.vocab.size() != hp.vocab) {
        fprintf(stderr, "xasr: inconsistent hyper-parameters in '%s'\n", path);
        return false;
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "xasr", wl))
        return false;
    m.ctx = wl.ctx;
    m.buf = wl.buf;
    m.tensors = std::move(wl.tensors);

    for (int i = 0; i < 3; i++) {
        m.conv_w[i] = req(m, "emb.conv" + std::to_string(i) + ".weight");
        m.conv_b[i] = req(m, "emb.conv" + std::to_string(i) + ".bias");
    }
    m.cnx_dw_w = req(m, "emb.cnx.dw.weight");
    m.cnx_dw_b = req(m, "emb.cnx.dw.bias");
    m.cnx_pw1_w = req(m, "emb.cnx.pw1.weight");
    m.cnx_pw1_b = req(m, "emb.cnx.pw1.bias");
    m.cnx_pw2_w = req(m, "emb.cnx.pw2.weight");
    m.cnx_pw2_b = req(m, "emb.cnx.pw2.bias");
    m.emb_out_w = req(m, "emb.out.weight");
    m.emb_out_b = req(m, "emb.out.bias");
    m.emb_norm_bias = req(m, "emb.norm.bias");
    m.emb_norm_scale = std::exp(to_f32(req(m, "emb.norm.log_scale"))[0]);

    m.stacks.resize(S);
    for (size_t s = 0; s < S; s++) {
        auto& st = m.stacks[s];
        const std::string sp = "z." + std::to_string(s) + ".";
        if (hp.downsample[s] > 1) {
            st.ds_w = softmax_host(to_f32(req(m, sp + "ds_bias")));
            st.combiner = req(m, sp + "combiner");
        }
        st.layers.resize((size_t)hp.n_layers[s]);
        for (int l = 0; l < hp.n_layers[s]; l++) {
            auto& L = st.layers[(size_t)l];
            const std::string p = sp + std::to_string(l) + ".";
            L.aw_in_w = req(m, p + "aw.in.weight");
            L.aw_in_b = req(m, p + "aw.in.bias");
            L.aw_pos_w = req(m, p + "aw.pos.weight");
            L.sa1_in_w = req(m, p + "sa1.in.weight");
            L.sa1_in_b = req(m, p + "sa1.in.bias");
            L.sa1_out_w = req(m, p + "sa1.out.weight");
            L.sa1_out_b = req(m, p + "sa1.out.bias");
            L.sa2_in_w = req(m, p + "sa2.in.weight");
            L.sa2_in_b = req(m, p + "sa2.in.bias");
            L.sa2_out_w = req(m, p + "sa2.out.weight");
            L.sa2_out_b = req(m, p + "sa2.out.bias");
            for (int f = 0; f < 3; f++) {
                const std::string fp = p + "ff" + std::to_string(f + 1) + ".";
                L.ff_in_w[f] = req(m, fp + "in.weight");
                L.ff_in_b[f] = req(m, fp + "in.bias");
                L.ff_out_w[f] = req(m, fp + "out.weight");
                L.ff_out_b[f] = req(m, fp + "out.bias");
            }
            L.na_in_w = req(m, p + "na.in.weight");
            L.na_in_b = req(m, p + "na.in.bias");
            L.na_out_w = req(m, p + "na.out.weight");
            L.na_out_b = req(m, p + "na.out.bias");
            for (int c = 0; c < 2; c++) {
                const std::string cp = p + "cv" + std::to_string(c + 1) + ".";
                auto& C = L.cv[c];
                C.in_w = req(m, cp + "in.weight");
                C.in_b = req(m, cp + "in.bias");
                C.causal_w = req(m, cp + "causal.weight");
                C.causal_b = req(m, cp + "causal.bias");
                C.chunk_w = req(m, cp + "chunk.weight");
                C.chunk_b = req(m, cp + "chunk.bias");
                C.chunk_scale = req(m, cp + "chunk_scale");
                C.out_w = req(m, cp + "out.weight");
                C.out_b = req(m, cp + "out.bias");
            }
            L.norm_bias = req(m, p + "norm.bias");
            L.norm_scale = std::exp(to_f32(req(m, p + "norm.log_scale"))[0]);
            L.bypass = req(m, p + "bypass");
            L.bypass_mid = req(m, p + "bypass_mid");
        }
    }
    m.out_ds_w = softmax_host(to_f32(req(m, "z.out_ds_bias")));
    m.enc_proj_w = req(m, "join.encoder_proj.weight");
    m.enc_proj_b = req(m, "join.encoder_proj.bias");
    m.dec_emb = to_f32(req(m, "dec.emb.weight"));
    m.dec_conv = to_f32(req(m, "dec.conv.weight"));
    m.dec_proj_w = to_f32(req(m, "join.decoder_proj.weight"));
    m.dec_proj_b = to_f32(req(m, "join.decoder_proj.bias"));
    m.join_out_w = to_f32(req(m, "join.output_linear.weight"));
    m.join_out_b = to_f32(req(m, "join.output_linear.bias"));
    return true;
}

// ---- graph helpers ---------------------------------------------------------

ggml_tensor* lin(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(g, w, x);
    return b ? ggml_add(g, y, b) : y;
}

// Swoosh-L(x) = log(1 + e^(x-4)) - 0.08x - 0.035; Swoosh-R: offset 1, constant 0.313261687.
ggml_tensor* swoosh(ggml_context* g, ggml_tensor* x, float off, float c) {
    ggml_tensor* sp = ggml_softplus(g, ggml_scale_bias(g, x, 1.0f, -off));
    return ggml_sub(g, sp, ggml_scale_bias(g, x, 0.08f, c));
}
ggml_tensor* swoosh_l(ggml_context* g, ggml_tensor* x) {
    return swoosh(g, x, 4.0f, 0.035f);
}
ggml_tensor* swoosh_r(ggml_context* g, ggml_tensor* x) {
    return swoosh(g, x, 1.0f, 0.313261687f);
}

// BiasNorm: x * mean((x - bias)^2)^-1/2 * exp(log_scale), over channels (ne0).
ggml_tensor* bias_norm(ggml_context* g, ggml_tensor* x, ggml_tensor* bias, float scale) {
    ggml_tensor* ms = ggml_mean(g, ggml_sqr(g, ggml_sub(g, x, bias))); // (1, T)
    return ggml_scale(g, ggml_div(g, x, ggml_sqrt(g, ms)), scale);
}

// orig + (x - orig) * scale
ggml_tensor* bypass(ggml_context* g, ggml_tensor* orig, ggml_tensor* x, ggml_tensor* scale) {
    return ggml_add(g, orig, ggml_mul(g, ggml_sub(g, x, orig), scale));
}

ggml_tensor* ffn(ggml_context* g, ggml_tensor* x, ggml_tensor* w1, ggml_tensor* b1, ggml_tensor* w2, ggml_tensor* b2) {
    return lin(g, swoosh_l(g, lin(g, x, w1, b1)), w2, b2);
}

// Depthwise correlation over time of (C, Tin) with a (K, C) kernel: out (C, Tin-K+1).
ggml_tensor* dwconv_valid(ggml_context* g, ggml_tensor* x, ggml_tensor* w) {
    const int C = (int)x->ne[0], Tin = (int)x->ne[1], K = (int)w->ne[0];
    ggml_tensor* xt = ggml_cont(g, ggml_transpose(g, x)); // (Tin, C)
    ggml_tensor* y = ggml_ssm_conv(g, ggml_reshape_3d(g, xt, Tin, C, 1), ggml_reshape_2d(g, w, K, C));
    return ggml_reshape_2d(g, y, C, Tin - K + 1);
}

// Last n columns of a (R, T) tensor, contiguous.
ggml_tensor* last_cols(ggml_context* g, ggml_tensor* x, int n) {
    return ggml_cont(g, ggml_view_2d(g, x, x->ne[0], n, x->nb[1], (size_t)(x->ne[1] - n) * x->nb[1]));
}

ggml_tensor* rows(ggml_context* g, ggml_tensor* x, int r0, int n) {
    return ggml_cont(g, ggml_view_2d(g, x, n, x->ne[1], x->nb[1], (size_t)r0 * ggml_element_size(x)));
}

// CompactRelPositionalEncoding rows for relative positions -(seq+left-1) .. seq-1.
std::vector<float> compact_pe(int pos_dim, int seq, int left) {
    const int L2 = left + 2 * seq - 1;
    std::vector<float> pe((size_t)L2 * pos_dim);
    const float c = std::sqrt((float)pos_dim);
    const float length_scale = (float)pos_dim / (2.0f * (float)M_PI);
    for (int n = 0; n < L2; n++) {
        const float x = (float)(n - (seq + left - 1));
        const float sgn = x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f);
        const float xc = c * sgn * (std::log(std::fabs(x) + c) - std::log(c));
        const float xa = std::atan(xc / length_scale);
        float* r = pe.data() + (size_t)n * pos_dim;
        for (int f = 0; f < pos_dim / 2; f++) {
            r[2 * f] = std::cos(xa * (float)(f + 1));
            r[2 * f + 1] = std::sin(xa * (float)(f + 1));
        }
        r[pos_dim - 1] = 1.0f;
    }
    return pe;
}

std::string lname(const char* what, int li) {
    return std::string(what) + "_" + std::to_string(li);
}

void in_tensor(ggml_tensor* t, const std::string& name) {
    ggml_set_name(t, name.c_str());
    ggml_set_input(t);
}

void out_tensor(ggml_cgraph* gf, ggml_tensor* t, const std::string& name) {
    ggml_set_name(t, name.c_str());
    ggml_set_output(t);
    ggml_build_forward_expand(gf, t);
}

// One Zipformer2 layer, streaming_forward (docs/xasr/PLAN.md, steps 1-12).
ggml_tensor* zipformer_layer(ggml_context* g, ggml_cgraph* gf, const xasr_hparams& hp, int s, const zlayer& L, int li,
                             ggml_tensor* x, ggml_tensor* pe, ggml_tensor* mask, int seq, int left) {
    const int d = hp.dims[s], H = hp.n_heads[s], qd = hp.qd[s], vd = hp.vd[s], pd = hp.pd[s], K = hp.kernel[s];
    const int hid = 3 * d / 4, pad = K / 2, Tk = left + seq;
    ggml_tensor* src_orig = x;

    // 1. attention weights
    ggml_tensor* in = lin(g, x, L.aw_in_w, L.aw_in_b); // (H*(2qd+pd), seq)
    ggml_tensor* q = ggml_reshape_3d(g, rows(g, in, 0, H * qd), qd, H, seq);
    ggml_tensor* k = rows(g, in, H * qd, H * qd);
    ggml_tensor* p = ggml_reshape_3d(g, rows(g, in, 2 * H * qd, H * pd), pd, H, seq);
    ggml_tensor* ck = ggml_new_tensor_2d(g, GGML_TYPE_F32, H * qd, left);
    in_tensor(ck, lname("ck", li));
    ggml_tensor* kfull = ggml_concat(g, ck, k, 1); // (H*qd, Tk)
    out_tensor(gf, last_cols(g, kfull, left), lname("o_ck", li));
    ggml_tensor* Q = ggml_cont(g, ggml_permute(g, q, 0, 2, 1, 3));                                     // (qd, seq, H)
    ggml_tensor* Kh = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, kfull, qd, H, Tk), 0, 2, 1, 3)); // (qd, Tk, H)
    ggml_tensor* sc = ggml_mul_mat(g, Kh, Q);                                                          // (Tk, seq, H)
    const int L2 = (int)pe->ne[1];
    ggml_tensor* PE = ggml_reshape_3d(g, lin(g, pe, L.aw_pos_w, nullptr), pd, H, L2);
    PE = ggml_cont(g, ggml_permute(g, PE, 0, 2, 1, 3));            // (pd, L2, H)
    ggml_tensor* P = ggml_cont(g, ggml_permute(g, p, 0, 2, 1, 3)); // (pd, seq, H)
    ggml_tensor* ps = ggml_mul_mat(g, PE, P);                      // (L2, seq, H)
    // rel -> abs: out[j, t] = ps[j + seq-1-t, t] (icefall's as_strided)
    ggml_tensor* shifted =
        ggml_view_3d(g, ps, Tk, seq, H, ps->nb[1] - ps->nb[0], ps->nb[2], (size_t)(seq - 1) * ps->nb[0]);
    sc = ggml_add(g, sc, ggml_cont(g, shifted));
    sc = ggml_add(g, sc, mask);            // (Tk) broadcast: -inf on not-yet-filled left context
    ggml_tensor* W = ggml_soft_max(g, sc); // (Tk, seq, H)

    // 2. FF1
    x = ggml_add(g, x, ffn(g, x, L.ff_in_w[0], L.ff_in_b[0], L.ff_out_w[0], L.ff_out_b[0]));

    // 3. NonlinAttention with head 0's weights
    {
        ggml_tensor* t = lin(g, x, L.na_in_w, L.na_in_b); // (3*hid, seq)
        ggml_tensor* sgate = ggml_tanh(g, rows(g, t, 0, hid));
        ggml_tensor* xx = ggml_mul(g, rows(g, t, hid, hid), sgate);
        ggml_tensor* y = rows(g, t, 2 * hid, hid);
        ggml_tensor* cn = ggml_new_tensor_2d(g, GGML_TYPE_F32, hid, left);
        in_tensor(cn, lname("cn", li));
        ggml_tensor* xp = ggml_concat(g, cn, xx, 1); // (hid, Tk)
        out_tensor(gf, last_cols(g, xp, left), lname("o_cn", li));
        ggml_tensor* W0 = ggml_view_2d(g, W, Tk, seq, W->nb[1], 0);
        ggml_tensor* o = ggml_mul_mat(g, ggml_cont(g, ggml_transpose(g, xp)), W0); // (hid, seq)
        x = ggml_add(g, x, lin(g, ggml_mul(g, o, y), L.na_out_w, L.na_out_b));
    }

    auto self_attn = [&](ggml_tensor* xin, ggml_tensor* wi, ggml_tensor* bi, ggml_tensor* wo, ggml_tensor* bo,
                         const char* cname) {
        ggml_tensor* v = lin(g, xin, wi, bi); // (H*vd, seq)
        ggml_tensor* cv = ggml_new_tensor_2d(g, GGML_TYPE_F32, H * vd, left);
        in_tensor(cv, lname(cname, li));
        ggml_tensor* vf = ggml_concat(g, cv, v, 1); // (H*vd, Tk)
        out_tensor(gf, last_cols(g, vf, left), lname((std::string("o_") + cname).c_str(), li));
        ggml_tensor* Vt = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, vf, vd, H, Tk), 1, 2, 0, 3)); // (Tk, vd, H)
        ggml_tensor* o = ggml_mul_mat(g, Vt, W);                                                        // (vd, seq, H)
        o = ggml_cont(g, ggml_permute(g, o, 0, 2, 1, 3));                                               // (vd, H, seq)
        return lin(g, ggml_reshape_2d(g, o, H * vd, seq), wo, bo);
    };

    auto conv = [&](ggml_tensor* xin, const conv_mod& C, const char* cname) {
        ggml_tensor* t = lin(g, xin, C.in_w, C.in_b); // (2d, seq)
        ggml_tensor* xg = ggml_mul(g, rows(g, t, 0, d), ggml_sigmoid(g, rows(g, t, d, d)));
        ggml_tensor* cc = ggml_new_tensor_2d(g, GGML_TYPE_F32, d, pad);
        in_tensor(cc, lname(cname, li));
        ggml_tensor* xc = ggml_concat(g, cc, xg, 1); // (d, pad+seq)
        out_tensor(gf, last_cols(g, xc, pad), lname((std::string("o_") + cname).c_str(), li));
        ggml_tensor* causal = ggml_add(g, dwconv_valid(g, xc, C.causal_w), C.causal_b);
        ggml_tensor* chunk = dwconv_valid(g, ggml_pad_ext(g, xg, 0, 0, pad, pad, 0, 0, 0, 0), C.chunk_w);
        chunk = ggml_add(g, chunk, C.chunk_b);
        // chunk scale: 1 + left_edge + right_edge, each (K, d), sliced / zero-padded to seq
        ggml_tensor* le = ggml_view_2d(g, C.chunk_scale, K, d, C.chunk_scale->nb[1], 0);
        ggml_tensor* re = ggml_view_2d(g, C.chunk_scale, K, d, C.chunk_scale->nb[1], C.chunk_scale->nb[2]);
        if (seq < K) {
            le = ggml_view_2d(g, le, seq, d, le->nb[1], 0);
            re = ggml_view_2d(g, re, seq, d, re->nb[1], (size_t)(K - seq) * ggml_element_size(re));
            le = ggml_cont(g, le);
            re = ggml_cont(g, re);
        } else {
            le = ggml_pad_ext(g, ggml_cont(g, le), 0, seq - K, 0, 0, 0, 0, 0, 0);
            re = ggml_pad_ext(g, ggml_cont(g, re), seq - K, 0, 0, 0, 0, 0, 0, 0);
        }
        ggml_tensor* scale = ggml_scale_bias(g, ggml_add(g, le, re), 1.0f, 1.0f); // (seq, d)
        chunk = ggml_mul(g, chunk, ggml_cont(g, ggml_transpose(g, scale)));
        return lin(g, swoosh_r(g, ggml_add(g, chunk, causal)), C.out_w, C.out_b);
    };

    // 4.-10.
    x = ggml_add(g, x, self_attn(x, L.sa1_in_w, L.sa1_in_b, L.sa1_out_w, L.sa1_out_b, "cv1"));
    x = ggml_add(g, x, conv(x, L.cv[0], "cc1"));
    x = ggml_add(g, x, ffn(g, x, L.ff_in_w[1], L.ff_in_b[1], L.ff_out_w[1], L.ff_out_b[1]));
    x = bypass(g, src_orig, x, L.bypass_mid);
    x = ggml_add(g, x, self_attn(x, L.sa2_in_w, L.sa2_in_b, L.sa2_out_w, L.sa2_out_b, "cv2"));
    x = ggml_add(g, x, conv(x, L.cv[1], "cc2"));
    x = ggml_add(g, x, ffn(g, x, L.ff_in_w[2], L.ff_in_b[2], L.ff_out_w[2], L.ff_out_b[2]));
    // 11.-12.
    x = bias_norm(g, x, L.norm_bias, L.norm_scale);
    return bypass(g, src_orig, x, L.bypass);
}

// Weighted mean over each group of `ds` frames of (C, T): (C, T/ds).
ggml_tensor* simple_downsample(ggml_context* g, ggml_tensor* x, const std::vector<float>& w) {
    const int ds = (int)w.size(), C = (int)x->ne[0], Tout = (int)x->ne[1] / ds;
    ggml_tensor* acc = nullptr;
    for (int j = 0; j < ds; j++) {
        ggml_tensor* v = ggml_view_2d(g, x, C, Tout, x->nb[1] * ds, (size_t)j * x->nb[1]);
        ggml_tensor* t = ggml_scale(g, ggml_cont(g, v), w[(size_t)j]);
        acc = acc ? ggml_add(g, acc, t) : t;
    }
    return acc;
}

ggml_cgraph* build_chunk_graph(xasr_context* c, std::vector<uint8_t>& meta, bool dump) {
    auto& m = c->model;
    const auto& hp = m.hp;
    const geom& G = c->g;
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* g = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(g, kGraphNodes, false);

    ggml_tensor* feats = ggml_new_tensor_4d(g, GGML_TYPE_F32, hp.feature_dim, G.T, 1, 1);
    in_tensor(feats, "feats");
    // Conv2dSubsampling: ggml (W=freq, H=time, C, N)
    auto cbias = [&](ggml_tensor* b) { return ggml_reshape_3d(g, b, 1, 1, b->ne[0]); };
    ggml_tensor* x = ggml_conv_2d(g, m.conv_w[0], feats, 1, 1, 1, 0, 1, 1); // pad (freq 1, time 0)
    x = swoosh_r(g, ggml_add(g, x, cbias(m.conv_b[0])));
    x = ggml_conv_2d(g, m.conv_w[1], x, 2, 2, 0, 0, 1, 1);
    x = swoosh_r(g, ggml_add(g, x, cbias(m.conv_b[1])));
    x = ggml_conv_2d(g, m.conv_w[2], x, 2, 1, 0, 0, 1, 1); // stride (freq 2, time 1)
    x = swoosh_r(g, ggml_add(g, x, cbias(m.conv_b[2])));   // (F', chunk+3, 128)
    const int Fp = (int)x->ne[0], Tc = (int)x->ne[1], Ch = (int)x->ne[2];
    GGML_ASSERT(Tc == G.chunk + 3);
    // ConvNeXt, streaming: 3 cached time frames on the left, no time padding
    ggml_tensor* ecache = ggml_new_tensor_3d(g, GGML_TYPE_F32, Fp, 3, Ch);
    in_tensor(ecache, "emb_cache");
    ggml_tensor* xcat = ggml_concat(g, ecache, x, 1); // (F', chunk+6, 128)
    out_tensor(gf,
               ggml_cont(g, ggml_view_3d(g, xcat, Fp, 3, Ch, xcat->nb[1], xcat->nb[2], (size_t)G.chunk * xcat->nb[1])),
               "o_emb_cache");
    ggml_tensor* y = ggml_conv_2d_dw(g, m.cnx_dw_w, xcat, 1, 1, 3, 0, 1, 1); // (F', chunk, 128)
    y = ggml_add(g, y, cbias(m.cnx_dw_b));
    y = ggml_cont(g, ggml_permute(g, y, 1, 2, 0, 3)); // (128, F', chunk)
    y = ggml_reshape_2d(g, y, Ch, Fp * G.chunk);
    y = lin(g, y, ggml_reshape_2d(g, m.cnx_pw1_w, Ch, m.cnx_pw1_w->ne[3]), m.cnx_pw1_b);
    y = swoosh_l(g, y);
    y = lin(g, y, ggml_reshape_2d(g, m.cnx_pw2_w, m.cnx_pw2_w->ne[2], Ch), m.cnx_pw2_b); // (128, F'*chunk)
    ggml_tensor* byp = ggml_view_3d(g, x, Fp, G.chunk, Ch, x->nb[1], x->nb[2], 0);
    byp = ggml_cont(g, ggml_permute(g, byp, 1, 2, 0, 3)); // (128, F', chunk)
    y = ggml_add(g, ggml_reshape_3d(g, y, Ch, Fp, G.chunk), byp);
    // (b, c, t, f) -> (b, t, c*f): feature index c*F' + f
    y = ggml_cont(g, ggml_permute(g, y, 1, 0, 2, 3)); // (F', 128, chunk)
    y = ggml_reshape_2d(g, y, Fp * Ch, G.chunk);
    x = lin(g, y, m.emb_out_w, m.emb_out_b);
    x = bias_norm(g, x, m.emb_norm_bias, m.emb_norm_scale); // (dims[0], chunk)
    if (dump)
        out_tensor(gf, ggml_cont(g, x), "d_embed");

    const int S = (int)hp.dims.size();
    std::vector<ggml_tensor*> outs((size_t)S);
    int li = 0;
    for (int s = 0; s < S; s++) {
        const int dim = hp.dims[s], ds = hp.downsample[s];
        const int cur = (int)x->ne[0];
        if (dim <= cur)
            x = rows(g, x, 0, dim);
        else
            x = ggml_pad_ext(g, x, 0, dim - cur, 0, 0, 0, 0, 0, 0);
        ggml_tensor* src_orig = x;
        const int seq = G.chunk / ds, left = G.left / ds;
        if (ds > 1)
            x = simple_downsample(g, x, m.stacks[(size_t)s].ds_w);
        ggml_tensor* pe = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp.pos_dim, left + 2 * seq - 1);
        in_tensor(pe, "pe_" + std::to_string(s));
        ggml_tensor* mask = ggml_new_tensor_1d(g, GGML_TYPE_F32, left + seq);
        in_tensor(mask, "mask_" + std::to_string(s));
        for (auto& L : m.stacks[(size_t)s].layers)
            x = zipformer_layer(g, gf, hp, s, L, li++, x, pe, mask, seq, left);
        if (ds > 1) {
            x = ggml_repeat(g, ggml_reshape_3d(g, x, dim, 1, seq), ggml_new_tensor_3d(g, GGML_TYPE_F32, dim, ds, seq));
            x = ggml_reshape_2d(g, x, dim, G.chunk);
            x = bypass(g, src_orig, x, m.stacks[(size_t)s].combiner);
        }
        outs[(size_t)s] = x;
        if (dump)
            out_tensor(gf, ggml_cont(g, x), "d_stack_" + std::to_string(s));
    }
    // _get_full_dim_output: the last stack, plus the higher channels of earlier, wider stacks
    ggml_tensor* full = outs[(size_t)S - 1];
    int cur = hp.dims[(size_t)S - 1];
    for (int s = S - 2; s >= 0; s--) {
        if (hp.dims[s] > cur) {
            full = ggml_concat(g, full, rows(g, outs[(size_t)s], cur, hp.dims[s] - cur), 0);
            cur = hp.dims[s];
        }
    }
    full = simple_downsample(g, full, m.out_ds_w); // (max dim, chunk/2)
    if (dump)
        out_tensor(gf, ggml_cont(g, full), "d_enc_full");
    out_tensor(gf, lin(g, full, m.enc_proj_w, m.enc_proj_b), "enc_out");
    ggml_free(g);
    return gf;
}

bool ensure_sched(xasr_context* c) {
    if (c->sched)
        return true;
    ggml_backend_t be[2] = {c->backend, c->backend_cpu};
    c->sched = ggml_backend_sched_new(be, nullptr, c->backend != c->backend_cpu ? 2 : 1, kGraphNodes, false, false);
    return c->sched != nullptr;
}

// Encoder state carried from chunk to chunk: every cache the chunk graph
// consumes (by input name) and the 50 Hz frames through encoder_embed so far.
struct enc_state {
    std::vector<std::pair<std::string, std::vector<float>>> caches;
    int processed = 0;
};

void set_input(ggml_cgraph* gf, const std::string& name, const std::vector<float>& v) {
    ggml_tensor* t = ggml_graph_get_tensor(gf, name.c_str());
    GGML_ASSERT(t && (size_t)ggml_nelements(t) == v.size());
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

void set_input(ggml_cgraph* gf, const std::string& name, const float* v, size_t n) {
    ggml_tensor* t = ggml_graph_get_tensor(gf, name.c_str());
    GGML_ASSERT(t && (size_t)ggml_nelements(t) == n);
    ggml_backend_tensor_set(t, v, 0, n * sizeof(float));
}

std::vector<float> get_output(ggml_cgraph* gf, const std::string& name) {
    ggml_tensor* t = ggml_graph_get_tensor(gf, name.c_str());
    GGML_ASSERT(t);
    std::vector<float> v((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

// Zero caches, sized from the graph's inputs.
void init_caches(xasr_context* c, ggml_cgraph* gf, enc_state& st) {
    auto add = [&](const std::string& n) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, n.c_str());
        GGML_ASSERT(t);
        st.caches.emplace_back(n, std::vector<float>((size_t)ggml_nelements(t), 0.0f));
    };
    add("emb_cache");
    int li = 0;
    for (const auto& s : c->model.stacks)
        for (size_t l = 0; l < s.layers.size(); l++, li++)
            for (const char* n : {"ck", "cn", "cv1", "cv2", "cc1", "cc2"})
                add(lname(n, li));
}

core_kaldi::FbankParams fbank_params(const xasr_context* c) {
    core_kaldi::FbankParams p; // sherpa-onnx FeatureExtractorConfig
    p.n_mels = c->model.hp.feature_dim;
    p.high_freq = -400.0f;
    p.snip_edges = false;
    p.mel_domain_triangles = true;
    return p;
}

std::vector<float> fbank_of(xasr_context* c, const float* samples, int n, int& T) {
    std::vector<float> pcm(samples, samples + n);
    pcm.resize((size_t)n + (size_t)c->g.tail_pad_ms * 16, 0.0f);
    return core_kaldi::compute_fbank(pcm.data(), (int)pcm.size(), fbank_params(c), T);
}

// The chunk loop over feature rows [*pos, n_rows): sherpa's window of T frames
// shifted by decode_chunk_len, run while pos + T < n_rows (strictly, as
// OnlineRecognizer::IsReady). Appends (chunk/2, joiner_dim) rows to enc_out.
bool run_chunks(xasr_context* c, std::vector<uint8_t>& meta, enc_state& st, const float* feats, int n_rows, int* pos,
                std::vector<float>& enc_out, std::vector<std::vector<float>>* dumps) {
    const auto& hp = c->model.hp;
    const geom& G = c->g;
    const int S = (int)hp.dims.size();
    if (*pos + G.T >= n_rows)
        return true;
    const bool dump = dumps != nullptr;
    ggml_cgraph* gf = build_chunk_graph(c, meta, dump);
    if (!ensure_sched(c))
        return false;
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "xasr: chunk graph alloc failed\n");
        return false;
    }
    if (st.caches.empty())
        init_caches(c, gf, st);
    std::vector<std::vector<float>> pes((size_t)S);
    for (int s = 0; s < S; s++)
        pes[(size_t)s] = compact_pe(hp.pos_dim, G.chunk / hp.downsample[s], G.left / hp.downsample[s]);
    if (dump && dumps->size() != (size_t)S + 2)
        dumps->assign((size_t)S + 2, {});
    for (; *pos + G.T < n_rows; *pos += G.decode_chunk_len) {
        set_input(gf, "feats", feats + (size_t)*pos * hp.feature_dim, (size_t)G.T * hp.feature_dim);
        for (auto& kv : st.caches)
            set_input(gf, kv.first, kv.second);
        for (int s = 0; s < S; s++) {
            set_input(gf, "pe_" + std::to_string(s), pes[(size_t)s]);
            const int ds = hp.downsample[s], left = G.left / ds, seq = G.chunk / ds;
            std::vector<float> mask((size_t)(left + seq), 0.0f);
            for (int j = 0; j < left; j++)
                if (j * ds < G.left - st.processed) // left context not filled yet
                    mask[(size_t)j] = -INFINITY;
            set_input(gf, "mask_" + std::to_string(s), mask);
        }
        if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "xasr: chunk compute failed\n");
            return false;
        }
        for (auto& kv : st.caches)
            kv.second = get_output(gf, "o_" + kv.first);
        st.processed += G.chunk;
        const auto e = get_output(gf, "enc_out");
        enc_out.insert(enc_out.end(), e.begin(), e.end());
        if (dump) {
            auto app = [&](size_t slot, const char* name) {
                const auto v = get_output(gf, name);
                (*dumps)[slot].insert((*dumps)[slot].end(), v.begin(), v.end());
            };
            app(0, "d_embed");
            for (int s = 0; s < S; s++)
                app((size_t)s + 1, ("d_stack_" + std::to_string(s)).c_str());
            app((size_t)S + 1, "d_enc_full");
        }
    }
    return true;
}

// ---- decoder / joiner (host) ------------------------------------------------

// decoder_proj(relu(grouped_conv(embedding(ctx tokens)))) for the last context_size tokens.
std::vector<float> decoder_out(const xasr_model& m, const std::vector<int>& hyp) {
    const auto& hp = m.hp;
    const int D = hp.decoder_dim, Kc = hp.context_size,
              gs = (int)(m.dec_conv.size() / ((size_t)D * Kc)); // in-ch per group
    std::vector<float> e((size_t)Kc * D, 0.0f);
    for (int k = 0; k < Kc; k++) {
        const int tok = hyp[hyp.size() - (size_t)Kc + (size_t)k];
        if (tok >= 0)
            std::memcpy(e.data() + (size_t)k * D, m.dec_emb.data() + (size_t)tok * D, (size_t)D * sizeof(float));
    }
    std::vector<float> h((size_t)D);
    for (int o = 0; o < D; o++) {
        const int g0 = (o / gs) * gs;
        float acc = 0.0f;
        for (int i = 0; i < gs; i++)
            for (int k = 0; k < Kc; k++)
                acc += m.dec_conv[((size_t)o * gs + i) * Kc + k] * e[(size_t)k * D + g0 + i];
        h[(size_t)o] = acc > 0.0f ? acc : 0.0f;
    }
    const int J = hp.joiner_dim;
    std::vector<float> out((size_t)J);
    for (int j = 0; j < J; j++) {
        float acc = m.dec_proj_b[(size_t)j];
        const float* w = m.dec_proj_w.data() + (size_t)j * D;
        for (int i = 0; i < D; i++)
            acc += w[i] * h[(size_t)i];
        out[(size_t)j] = acc;
    }
    return out;
}

// sherpa-onnx greedy search, resumable across chunks: tokens start as
// [-1]*(ctx-1) + [blank]; at most one symbol per encoder frame; blank and
// <unk> are never emitted and never re-run the decoder.
struct greedy_state {
    std::vector<int> hyp;
    std::vector<float> d; // decoder_proj output for the current context
    std::vector<int32_t> toks;
    bool first = true;
};

void greedy_init(const xasr_model& m, greedy_state& gs) {
    gs.hyp.assign((size_t)m.hp.context_size, -1);
    gs.hyp.back() = m.hp.blank_id;
    gs.d = decoder_out(m, gs.hyp);
    gs.toks.clear();
    gs.first = true;
}

void greedy_run(const xasr_model& m, greedy_state& gs, const float* enc, int n_enc, float* first_logits) {
    const auto& hp = m.hp;
    const int J = hp.joiner_dim, V = hp.vocab;
    std::vector<float> t((size_t)J), logits((size_t)V);
    for (int f = 0; f < n_enc; f++) {
        const float* e = enc + (size_t)f * J;
        for (int j = 0; j < J; j++)
            t[(size_t)j] = std::tanh(e[j] + gs.d[(size_t)j]);
        int best = 0;
        for (int v = 0; v < V; v++) {
            float acc = m.join_out_b[(size_t)v];
            const float* w = m.join_out_w.data() + (size_t)v * J;
            for (int j = 0; j < J; j++)
                acc += w[j] * t[(size_t)j];
            logits[(size_t)v] = acc;
            if (acc > logits[(size_t)best])
                best = v;
        }
        if (gs.first && first_logits)
            std::memcpy(first_logits, logits.data(), (size_t)V * sizeof(float));
        gs.first = false;
        if (best != hp.blank_id && best != hp.unk_id) {
            gs.toks.push_back(best);
            gs.hyp.push_back(best);
            gs.d = decoder_out(m, gs.hyp);
        }
    }
}

// sherpa-onnx text-utils: IsCJK / IsPunct (code-point ranges) for RemoveSpaceBetweenCjk.
bool is_cjk(char32_t cp) {
    return (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x2E80 && cp <= 0xA4CF) || (cp >= 0xA840 && cp <= 0xD7AF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFF65 && cp <= 0xFFDC) ||
           (cp >= 0x20000 && cp <= 0x2FFFF);
}

bool is_punct(char32_t cp) {
    return (cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) || (cp >= 0x5B && cp <= 0x60) ||
           (cp >= 0x7B && cp <= 0x7E) || (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0xFF01 && cp <= 0xFF0F) ||
           (cp >= 0xFF1A && cp <= 0xFF20) || (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65);
}

std::u32string utf8_to_u32(const std::string& s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        const uint8_t c = (uint8_t)s[i];
        const int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        char32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
        for (int k = 1; k < n && i + (size_t)k < s.size(); k++)
            cp = (cp << 6) | ((uint8_t)s[i + (size_t)k] & 0x3F);
        out.push_back(cp);
        i += (size_t)n;
    }
    return out;
}

std::string u32_to_utf8(const std::u32string& u) {
    std::string out;
    for (char32_t cp : u) {
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

// sherpa-onnx Convert(): join the symbols ('▁' prefix -> space, <unk> dropped),
// RemoveSpaceBetweenCjk (a space between two CJK characters or before
// punctuation goes), then drop leading spaces.
std::string detok(const xasr_model& m, const int32_t* toks, int n) {
    std::string s;
    for (int i = 0; i < n; i++) {
        std::string sym = m.vocab[(size_t)toks[i]];
        if (sym == "<unk>")
            continue;
        if (sym.size() >= 3 && (uint8_t)sym[0] == 0xe2 && (uint8_t)sym[1] == 0x96 && (uint8_t)sym[2] == 0x81)
            sym.replace(0, 3, " ");
        s += sym;
    }
    const std::u32string u = utf8_to_u32(s);
    std::u32string out;
    for (size_t i = 0; i < u.size(); i++) {
        if (i > 0 && u[i] == U' ' && i + 1 < u.size() && ((is_cjk(u[i - 1]) && is_cjk(u[i + 1])) || is_punct(u[i + 1])))
            continue;
        out.push_back(u[i]);
    }
    s = u32_to_utf8(out);
    const size_t b = s.find_first_not_of(' ');
    return b == std::string::npos ? std::string() : s.substr(b);
}

char* dup_str(const std::string& s) {
    char* r = (char*)malloc(s.size() + 1);
    std::memcpy(r, s.c_str(), s.size() + 1);
    return r;
}

} // namespace

extern "C" struct xasr_context_params xasr_context_default_params(void) {
    xasr_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.chunk_ms = 480;
    p.tail_pad_ms = -1;
    return p;
}

extern "C" struct xasr_context* xasr_init_from_file(const char* path, struct xasr_context_params params) {
    auto* c = new xasr_context();
    c->params = params;
    c->backend_cpu = core_cpu_backend::init();
    core_cpu_backend::set_n_threads(c->backend_cpu, params.n_threads > 0 ? params.n_threads : 4);
    c->backend = params.use_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!c->backend)
        c->backend = c->backend_cpu;
    c->compute_meta.resize(ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false));
    if (!load(c->model, path, c->backend)) {
        fprintf(stderr, "xasr: failed to load '%s'\n", path);
        xasr_free(c);
        return nullptr;
    }
    const auto& hp = c->model.hp;
    geom& G = c->g;
    G.chunk_ms = params.chunk_ms > 0 ? params.chunk_ms : 480;
    int left = -1;
    for (size_t i = 0; i < hp.chunk_ms_table.size() && i < hp.left_ctx_table.size(); i++)
        if (hp.chunk_ms_table[i] == G.chunk_ms)
            left = hp.left_ctx_table[i];
    int max_ds = 1;
    for (int d : hp.downsample)
        max_ds = std::max(max_ds, d);
    G.decode_chunk_len = G.chunk_ms / 10;
    G.chunk = G.decode_chunk_len / 2;
    // every stack's SimpleDownsample and the output downsample-by-2 need exact groups
    if (left < 0 || G.chunk % max_ds != 0 || G.chunk % 2 != 0) {
        fprintf(stderr, "xasr: unsupported chunk size %d ms\n", G.chunk_ms);
        xasr_free(c);
        return nullptr;
    }
    G.left = left;
    G.T = G.decode_chunk_len + 13;
    G.tail_pad_ms = params.tail_pad_ms >= 0 ? params.tail_pad_ms : G.T * 10 + 1000;
    if (params.verbosity > 0)
        fprintf(stderr, "xasr: %zu stacks, vocab %d, chunk %d ms (left %d), backend %s\n", hp.dims.size(), hp.vocab,
                G.chunk_ms, G.left, ggml_backend_name(c->backend));
    return c;
}

extern "C" void xasr_free(struct xasr_context* c) {
    if (!c)
        return;
    if (c->sched)
        ggml_backend_sched_free(c->sched);
    if (c->model.buf)
        core_gguf::release_weight_buffer(c->model.buf);
    if (c->model.ctx)
        ggml_free(c->model.ctx);
    if (c->backend_cpu && c->backend_cpu != c->backend)
        ggml_backend_free(c->backend_cpu);
    if (c->backend)
        ggml_backend_free(c->backend);
    delete c;
}

extern "C" float* xasr_compute_fbank(struct xasr_context* c, const float* samples, int n, int* out_T) {
    int T = 0;
    const auto fb = fbank_of(c, samples, n, T);
    *out_T = T;
    float* r = (float*)malloc(std::max<size_t>(1, fb.size()) * sizeof(float));
    std::memcpy(r, fb.data(), fb.size() * sizeof(float));
    return r;
}

extern "C" float* xasr_run_encoder(struct xasr_context* c, const float* fbank, int T, int* out_n_enc, int* out_dim,
                                   float** dumps, int n_dumps) {
    std::vector<float> enc;
    std::vector<std::vector<float>> d;
    enc_state st;
    int pos = 0;
    if (!run_chunks(c, c->compute_meta, st, fbank, T, &pos, enc, dumps && n_dumps > 0 ? &d : nullptr))
        return nullptr;
    for (int i = 0; dumps && i < n_dumps && i < (int)d.size(); i++)
        if (dumps[i])
            std::memcpy(dumps[i], d[(size_t)i].data(), d[(size_t)i].size() * sizeof(float));
    *out_dim = c->model.hp.joiner_dim;
    *out_n_enc = (int)(enc.size() / (size_t)*out_dim);
    float* r = (float*)malloc(std::max<size_t>(1, enc.size()) * sizeof(float));
    std::memcpy(r, enc.data(), enc.size() * sizeof(float));
    return r;
}

extern "C" int32_t* xasr_greedy(struct xasr_context* c, const float* enc, int n_enc, int* out_n, float* first_logits) {
    greedy_state gs;
    greedy_init(c->model, gs);
    greedy_run(c->model, gs, enc, n_enc, first_logits);
    *out_n = (int)gs.toks.size();
    int32_t* r = (int32_t*)malloc(std::max<size_t>(1, gs.toks.size()) * sizeof(int32_t));
    std::memcpy(r, gs.toks.data(), gs.toks.size() * sizeof(int32_t));
    return r;
}

extern "C" char* xasr_tokens_to_text(struct xasr_context* c, const int32_t* toks, int n) {
    return dup_str(detok(c->model, toks, n));
}

// ---- streaming -------------------------------------------------------------
//
// Feature frames are computed as audio arrives, and are bit-identical to one
// fbank over the whole signal. With snip_edges=false frame i covers samples
// [160i - 120, 160i + 280), so it is final once 160i + 280 samples exist
// (frame 0 mirrors its left edge; later frames only read real samples). The
// tail, including the mirrored right edge, is computed on flush after the tail
// padding is appended.

struct xasr_stream {
    xasr_context* c = nullptr;
    std::vector<float> pcm, feats;
    int n_frames = 0, pos = 0;
    enc_state st;
    greedy_state gs;
    std::vector<uint8_t> meta;
    bool finished = false;
};

namespace {

void stream_reset(xasr_stream* s) {
    s->pcm.clear();
    s->feats.clear();
    s->n_frames = s->pos = 0;
    s->st = enc_state();
    greedy_init(s->c->model, s->gs);
    s->finished = false;
}

void stream_extend_frames(xasr_stream* s, bool flush) {
    const auto p = fbank_params(s->c);
    const int F = p.n_mels, hop = 160, win = 400, lead = win / 2 - hop / 2; // 120
    const int n = (int)s->pcm.size();
    if (flush) {
        int T = 0;
        const auto fb = core_kaldi::compute_fbank(s->pcm.data(), n, p, T);
        if (T > s->n_frames) {
            s->feats.insert(s->feats.end(), fb.begin() + (size_t)s->n_frames * F, fb.begin() + (size_t)T * F);
            s->n_frames = T;
        }
        return;
    }
    const int ready = n >= win - lead ? (n - (win - lead)) / hop + 1 : 0;
    if (ready <= s->n_frames)
        return;
    if (s->n_frames == 0) { // frame 0: mirrored left edge
        int T0 = 0;
        const auto f0 = core_kaldi::compute_fbank(s->pcm.data(), win - lead, p, T0);
        s->feats.insert(s->feats.end(), f0.begin(), f0.begin() + F);
        s->n_frames = 1;
    }
    if (ready > s->n_frames) { // frames that only read real samples: snip_edges framing over a slice
        auto q = p;
        q.snip_edges = true;
        const int start = s->n_frames * hop - lead, len = (ready - 1 - s->n_frames) * hop + win;
        int T = 0;
        const auto fb = core_kaldi::compute_fbank(s->pcm.data() + start, len, q, T);
        GGML_ASSERT(T == ready - s->n_frames);
        s->feats.insert(s->feats.end(), fb.begin(), fb.end());
        s->n_frames = ready;
    }
}

bool stream_accept(xasr_stream* s, const float* pcm, int n, bool flush) {
    if (s->finished)
        return true;
    s->pcm.insert(s->pcm.end(), pcm, pcm + n);
    if (flush)
        s->pcm.resize(s->pcm.size() + (size_t)s->c->g.tail_pad_ms * 16, 0.0f);
    stream_extend_frames(s, flush);
    std::vector<float> enc;
    if (!run_chunks(s->c, s->meta, s->st, s->feats.data(), s->n_frames, &s->pos, enc, nullptr))
        return false;
    greedy_run(s->c->model, s->gs, enc.data(), (int)(enc.size() / (size_t)s->c->model.hp.joiner_dim), nullptr);
    s->finished = flush;
    return true;
}

std::string stream_text(const xasr_stream* s) {
    return detok(s->c->model, s->gs.toks.data(), (int)s->gs.toks.size());
}

} // namespace

extern "C" struct xasr_stream* xasr_stream_init(struct xasr_context* c) {
    auto* s = new xasr_stream();
    s->c = c;
    s->meta.resize(c->compute_meta.size());
    stream_reset(s);
    return s;
}

extern "C" char* xasr_stream_accept(struct xasr_stream* s, const float* samples, int n_samples, bool flush) {
    if (!stream_accept(s, samples, n_samples, flush))
        return nullptr;
    return dup_str(stream_text(s));
}

extern "C" void xasr_stream_reset(struct xasr_stream* s) {
    stream_reset(s);
}

extern "C" void xasr_stream_free(struct xasr_stream* s) {
    delete s;
}

extern "C" char* xasr_transcribe(struct xasr_context* c, const float* samples, int n) {
    xasr_stream* s = xasr_stream_init(c);
    char* r = xasr_stream_accept(s, samples, n, true);
    xasr_stream_free(s);
    return r;
}

extern "C" int xasr_n_stacks(struct xasr_context* c) {
    return (int)c->model.hp.dims.size();
}
extern "C" int xasr_stack_dim(struct xasr_context* c, int s) {
    return c->model.hp.dims[(size_t)s];
}
extern "C" int xasr_chunk_frames(struct xasr_context* c) {
    return c->g.chunk;
}
extern "C" int xasr_vocab(struct xasr_context* c) {
    return c->model.hp.vocab;
}
