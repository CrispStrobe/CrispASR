// dolphin.cpp — Dolphin (DataoceanAI) ASR runtime, #436. See dolphin.h and
// docs/dolphin/PLAN.md. Graph layout convention: activations are (d, T) with
// d the fast axis (ggml ne[0]).

#include "dolphin.h"

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
#include <map>
#include <string>
#include <vector>

namespace {

constexpr float kLnEps = 1e-5f;

struct dolphin_hparams {
    uint32_t n_mels = 80, d_model = 768, enc_heads = 12, enc_layers = 12, enc_ffn = 3072;
    uint32_t cgmlp_units = 3072, cgmlp_kernel = 31, merge_kernel = 31;
    uint32_t dec_heads = 12, dec_layers = 12, dec_ffn = 3072, vocab = 0;
    uint32_t blank_id = 0, sos_id = 2, eos_id = 3, asr_id = 4, notimestamp_id = 109;
    uint32_t frame_length_ms = 25, frame_shift_ms = 10;
    bool causal = true;
};

struct enc_block {
    ggml_tensor *ffm_norm_w, *ffm_norm_b, *ffm_w1, *ffm_b1, *ffm_w2, *ffm_b2;
    ggml_tensor *mha_norm_w, *mha_norm_b, *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b, *pos_w, *pos_u, *pos_v;
    ggml_tensor *mlp_norm_w, *mlp_norm_b, *proj1_w, *proj1_b, *csgu_norm_w, *csgu_norm_b, *csgu_conv_w, *csgu_conv_b;
    ggml_tensor *proj2_w, *proj2_b, *merge_conv_w, *merge_conv_b, *merge_proj_w, *merge_proj_b;
    ggml_tensor *ff_norm_w, *ff_norm_b, *ff_w1, *ff_b1, *ff_w2, *ff_b2, *final_norm_w, *final_norm_b;
};

struct dec_block {
    ggml_tensor *n1_w, *n1_b, *sq_w, *sq_b, *sk_w, *sk_b, *sv_w, *sv_b, *so_w, *so_b;
    ggml_tensor *n2_w, *n2_b, *cq_w, *cq_b, *ck_w, *ck_b, *cv_w, *cv_b, *co_w, *co_b;
    ggml_tensor *n3_w, *n3_b, *w1, *b1, *w2, *b2;
};

struct dolphin_model {
    dolphin_hparams hp;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    core_gguf::tensor_map tensors;
    std::vector<float> cmvn_mean, cmvn_istd;
    ggml_tensor *sub_conv0_w, *sub_conv0_b, *sub_conv1_w, *sub_conv1_b, *sub_out_w, *sub_out_b, *enc_pe;
    ggml_tensor *enc_norm_w, *enc_norm_b, *ctc_w, *ctc_b;
    ggml_tensor *dec_embed, *dec_pe, *dec_norm_w, *dec_norm_b, *dec_out_w, *dec_out_b;
    std::vector<enc_block> enc;
    std::vector<dec_block> dec;
    std::vector<std::string> vocab;
    std::map<std::string, int> tok2id;
};

} // namespace

struct dolphin_context {
    dolphin_context_params params;
    dolphin_model model;
    ggml_backend_t backend = nullptr, backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

namespace {

ggml_tensor* req(dolphin_model& m, const std::string& n) {
    return core_gguf::require(m.tensors, n.c_str(), "dolphin");
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

bool load(dolphin_model& m, const char* path, ggml_backend_t backend) {
    gguf_context* g = core_gguf::open_metadata(path);
    if (!g)
        return false;
    auto& hp = m.hp;
#define KV(name) hp.name = core_gguf::kv_u32(g, "dolphin." #name, hp.name)
    KV(n_mels);
    KV(d_model);
    KV(enc_heads);
    KV(enc_layers);
    KV(enc_ffn);
    KV(cgmlp_units);
    KV(cgmlp_kernel);
    KV(merge_kernel);
    KV(dec_heads);
    KV(dec_layers);
    KV(dec_ffn);
    KV(vocab);
    KV(blank_id);
    KV(sos_id);
    KV(eos_id);
    KV(asr_id);
    KV(notimestamp_id);
    KV(frame_length_ms);
    KV(frame_shift_ms);
#undef KV
    hp.causal = core_gguf::kv_bool(g, "dolphin.causal", hp.causal);
    m.vocab = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    core_gguf::free_metadata(g);
    for (size_t i = 0; i < m.vocab.size(); i++)
        m.tok2id[m.vocab[i]] = (int)i;

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "dolphin", wl))
        return false;
    m.ctx = wl.ctx;
    m.buf = wl.buf;
    m.tensors = std::move(wl.tensors);

    m.cmvn_mean = to_f32(req(m, "enc.cmvn.mean"));
    m.cmvn_istd = to_f32(req(m, "enc.cmvn.istd"));
    m.sub_conv0_w = req(m, "enc.sub.conv0.weight");
    m.sub_conv0_b = req(m, "enc.sub.conv0.bias");
    m.sub_conv1_w = req(m, "enc.sub.conv1.weight");
    m.sub_conv1_b = req(m, "enc.sub.conv1.bias");
    m.sub_out_w = req(m, "enc.sub.out.weight");
    m.sub_out_b = req(m, "enc.sub.out.bias");
    m.enc_pe = req(m, "enc.pe");
    m.enc_norm_w = req(m, "enc.norm_out.weight");
    m.enc_norm_b = req(m, "enc.norm_out.bias");
    m.ctc_w = req(m, "ctc.weight");
    m.ctc_b = req(m, "ctc.bias");
    m.dec_embed = req(m, "dec.embed.weight");
    m.dec_pe = req(m, "dec.pe");
    m.dec_norm_w = req(m, "dec.norm_out.weight");
    m.dec_norm_b = req(m, "dec.norm_out.bias");
    m.dec_out_w = req(m, "dec.out.weight");
    m.dec_out_b = req(m, "dec.out.bias");

    m.enc.resize(hp.enc_layers);
    for (uint32_t i = 0; i < hp.enc_layers; i++) {
        const std::string p = "enc.blk." + std::to_string(i) + ".";
        auto& b = m.enc[i];
        b.ffm_norm_w = req(m, p + "norm_ff_macaron.weight");
        b.ffm_norm_b = req(m, p + "norm_ff_macaron.bias");
        b.ffm_w1 = req(m, p + "ffm.w1.weight");
        b.ffm_b1 = req(m, p + "ffm.w1.bias");
        b.ffm_w2 = req(m, p + "ffm.w2.weight");
        b.ffm_b2 = req(m, p + "ffm.w2.bias");
        b.mha_norm_w = req(m, p + "norm_mha.weight");
        b.mha_norm_b = req(m, p + "norm_mha.bias");
        b.q_w = req(m, p + "attn.q.weight");
        b.q_b = req(m, p + "attn.q.bias");
        b.k_w = req(m, p + "attn.k.weight");
        b.k_b = req(m, p + "attn.k.bias");
        b.v_w = req(m, p + "attn.v.weight");
        b.v_b = req(m, p + "attn.v.bias");
        b.o_w = req(m, p + "attn.o.weight");
        b.o_b = req(m, p + "attn.o.bias");
        b.pos_w = req(m, p + "attn.pos.weight");
        b.pos_u = req(m, p + "attn.pos_bias_u");
        b.pos_v = req(m, p + "attn.pos_bias_v");
        b.mlp_norm_w = req(m, p + "norm_mlp.weight");
        b.mlp_norm_b = req(m, p + "norm_mlp.bias");
        b.proj1_w = req(m, p + "cgmlp.proj1.weight");
        b.proj1_b = req(m, p + "cgmlp.proj1.bias");
        b.csgu_norm_w = req(m, p + "cgmlp.csgu_norm.weight");
        b.csgu_norm_b = req(m, p + "cgmlp.csgu_norm.bias");
        b.csgu_conv_w = req(m, p + "cgmlp.csgu_conv.weight");
        b.csgu_conv_b = req(m, p + "cgmlp.csgu_conv.bias");
        b.proj2_w = req(m, p + "cgmlp.proj2.weight");
        b.proj2_b = req(m, p + "cgmlp.proj2.bias");
        b.merge_conv_w = req(m, p + "merge_conv.weight");
        b.merge_conv_b = req(m, p + "merge_conv.bias");
        b.merge_proj_w = req(m, p + "merge_proj.weight");
        b.merge_proj_b = req(m, p + "merge_proj.bias");
        b.ff_norm_w = req(m, p + "norm_ff.weight");
        b.ff_norm_b = req(m, p + "norm_ff.bias");
        b.ff_w1 = req(m, p + "ff.w1.weight");
        b.ff_b1 = req(m, p + "ff.w1.bias");
        b.ff_w2 = req(m, p + "ff.w2.weight");
        b.ff_b2 = req(m, p + "ff.w2.bias");
        b.final_norm_w = req(m, p + "norm_final.weight");
        b.final_norm_b = req(m, p + "norm_final.bias");
    }
    m.dec.resize(hp.dec_layers);
    for (uint32_t i = 0; i < hp.dec_layers; i++) {
        const std::string p = "dec.blk." + std::to_string(i) + ".";
        auto& b = m.dec[i];
        b.n1_w = req(m, p + "norm1.weight");
        b.n1_b = req(m, p + "norm1.bias");
        b.sq_w = req(m, p + "self_attn.q.weight");
        b.sq_b = req(m, p + "self_attn.q.bias");
        b.sk_w = req(m, p + "self_attn.k.weight");
        b.sk_b = req(m, p + "self_attn.k.bias");
        b.sv_w = req(m, p + "self_attn.v.weight");
        b.sv_b = req(m, p + "self_attn.v.bias");
        b.so_w = req(m, p + "self_attn.o.weight");
        b.so_b = req(m, p + "self_attn.o.bias");
        b.n2_w = req(m, p + "norm2.weight");
        b.n2_b = req(m, p + "norm2.bias");
        b.cq_w = req(m, p + "src_attn.q.weight");
        b.cq_b = req(m, p + "src_attn.q.bias");
        b.ck_w = req(m, p + "src_attn.k.weight");
        b.ck_b = req(m, p + "src_attn.k.bias");
        b.cv_w = req(m, p + "src_attn.v.weight");
        b.cv_b = req(m, p + "src_attn.v.bias");
        b.co_w = req(m, p + "src_attn.o.weight");
        b.co_b = req(m, p + "src_attn.o.bias");
        b.n3_w = req(m, p + "norm3.weight");
        b.n3_b = req(m, p + "norm3.bias");
        b.w1 = req(m, p + "ff.w1.weight");
        b.b1 = req(m, p + "ff.w1.bias");
        b.w2 = req(m, p + "ff.w2.weight");
        b.b2 = req(m, p + "ff.w2.bias");
    }
    return true;
}

// ---- graph helpers ---------------------------------------------------------

ggml_tensor* ln(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    return ggml_add(g, ggml_mul(g, ggml_norm(g, x, kLnEps), w), b);
}

ggml_tensor* lin(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(g, w, x);
    return b ? ggml_add(g, y, b) : y;
}

// Scaled dot-product attention over (hd, H, Tq) queries and (hd, H, Tk)
// keys/values, with an optional additive (Tk, Tq) bias per head (ne2 = H or 1).
ggml_tensor* sdpa(ggml_context* g, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* bias, float scale) {
    const int hd = (int)q->ne[0], H = (int)q->ne[1], Tq = (int)q->ne[2];
    ggml_tensor* Q = ggml_cont(g, ggml_permute(g, q, 0, 2, 1, 3)); // (hd, Tq, H)
    ggml_tensor* K = ggml_cont(g, ggml_permute(g, k, 0, 2, 1, 3)); // (hd, Tk, H)
    ggml_tensor* s = ggml_mul_mat(g, K, Q);                        // (Tk, Tq, H)
    if (bias)
        s = ggml_add(g, s, bias);
    s = ggml_soft_max(g, ggml_scale(g, s, scale));
    ggml_tensor* Vt = ggml_cont(g, ggml_permute(g, v, 1, 2, 0, 3)); // (Tk, hd, H)
    ggml_tensor* o = ggml_mul_mat(g, Vt, s);                        // (hd, Tq, H)
    o = ggml_cont(g, ggml_permute(g, o, 0, 2, 1, 3));               // (hd, H, Tq)
    return ggml_reshape_2d(g, o, hd * H, Tq);
}

// Causal depthwise conv over time on (C, T): left-pad K-1 zero frames, then
// ggml_ssm_conv (out[c,t] = sum_k x[t+k, c] * w[k, c]) — PyTorch Conv1d with
// groups=C over the padded input. `pre` runs on the padded (C, T+K-1) tensor
// before the conv (the CSGU's LayerNorm, which upstream applies AFTER padding).
template <typename F>
ggml_tensor* causal_dwconv(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int K, F pre) {
    const int C = (int)x->ne[0], T = (int)x->ne[1];
    ggml_tensor* xp = ggml_pad_ext(g, ggml_cont(g, x), 0, 0, K - 1, 0, 0, 0, 0, 0); // (C, T+K-1)
    xp = pre(xp);
    ggml_tensor* xt = ggml_cont(g, ggml_transpose(g, xp));                         // (T+K-1, C)
    ggml_tensor* y = ggml_ssm_conv(g, ggml_reshape_3d(g, xt, T + K - 1, C, 1), w); // (C, T, 1)
    return ggml_add(g, ggml_reshape_2d(g, y, C, T), b);
}

// ---- encoder ---------------------------------------------------------------

ggml_cgraph* build_encoder(dolphin_context* c, int T, bool dump, int* out_T_enc) {
    auto& m = c->model;
    const auto& hp = m.hp;
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* g = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 16384, false);

    ggml_tensor* feats = ggml_new_tensor_4d(g, GGML_TYPE_F32, hp.n_mels, T, 1, 1); // (F, T, 1, 1), post-CMVN
    ggml_set_name(feats, "feats");
    ggml_set_input(feats);

    // Conv2dSubsampling4: (b, 1, t, f) convs; ggml (W=f, H=t, C, N).
    ggml_tensor* x = ggml_conv_2d(g, m.sub_conv0_w, feats, 2, 2, 0, 0, 1, 1);
    x = ggml_relu(g, ggml_add(g, x, ggml_reshape_3d(g, m.sub_conv0_b, 1, 1, m.sub_conv0_b->ne[0])));
    x = ggml_conv_2d(g, m.sub_conv1_w, x, 2, 2, 0, 0, 1, 1);
    x = ggml_relu(g, ggml_add(g, x, ggml_reshape_3d(g, m.sub_conv1_b, 1, 1, m.sub_conv1_b->ne[0])));
    const int F2 = (int)x->ne[0], T2 = (int)x->ne[1], C2 = (int)x->ne[2];
    // (b, c, t, f) -> transpose(1,2) -> (b, t, c*f): feature index c*F2 + f.
    x = ggml_cont(g, ggml_permute(g, x, 0, 2, 1, 3)); // (F2, C2, T2)
    x = ggml_reshape_2d(g, x, F2 * C2, T2);
    x = lin(g, x, m.sub_out_w, m.sub_out_b); // (d, T2)
    // RelPositionalEncoding (WeNet legacy): x * sqrt(d); pos = pe[0:T2] (absolute).
    x = ggml_scale(g, x, std::sqrt((float)hp.d_model));
    ggml_tensor* pos = ggml_view_2d(g, m.enc_pe, hp.d_model, T2, m.enc_pe->nb[1], 0);
    if (dump) {
        ggml_tensor* s = ggml_cont(g, x);
        ggml_set_name(s, "dump_sub");
        ggml_set_output(s);
        ggml_build_forward_expand(gf, s);
    }

    const int d = (int)hp.d_model, H = (int)hp.enc_heads, hd = d / H;
    const float att_scale = 1.0f / std::sqrt((float)hd);
    for (uint32_t il = 0; il < hp.enc_layers; il++) {
        const auto& b = m.enc[il];
        // macaron FFN, half step
        ggml_tensor* h = ln(g, x, b.ffm_norm_w, b.ffm_norm_b);
        h = lin(g, ggml_silu(g, lin(g, h, b.ffm_w1, b.ffm_b1)), b.ffm_w2, b.ffm_b2);
        x = ggml_add(g, x, ggml_scale(g, h, 0.5f));

        // branch 1: "rel_pos" attention with NO rel_shift (SDPA branch).
        ggml_tensor* x1 = ln(g, x, b.mha_norm_w, b.mha_norm_b);
        ggml_tensor* q = ggml_reshape_3d(g, lin(g, x1, b.q_w, b.q_b), hd, H, T2);
        ggml_tensor* k = ggml_reshape_3d(g, lin(g, x1, b.k_w, b.k_b), hd, H, T2);
        ggml_tensor* v = ggml_reshape_3d(g, lin(g, x1, b.v_w, b.v_b), hd, H, T2);
        ggml_tensor* p = ggml_reshape_3d(g, lin(g, pos, b.pos_w, nullptr), hd, H, T2);
        ggml_tensor* qu = ggml_add(g, q, ggml_reshape_3d(g, b.pos_u, hd, H, 1));
        ggml_tensor* qv = ggml_add(g, q, ggml_reshape_3d(g, b.pos_v, hd, H, 1));
        // bd = (q + v)·pᵀ per head, folded in as an additive bias (same 1/√d scale).
        ggml_tensor* P = ggml_cont(g, ggml_permute(g, p, 0, 2, 1, 3));
        ggml_tensor* QV = ggml_cont(g, ggml_permute(g, qv, 0, 2, 1, 3));
        ggml_tensor* bd = ggml_mul_mat(g, P, QV); // (Tk, Tq, H)
        ggml_tensor* a = sdpa(g, qu, k, v, bd, att_scale);
        x1 = lin(g, a, b.o_w, b.o_b);

        // branch 2: cgMLP
        ggml_tensor* x2 = ln(g, x, b.mlp_norm_w, b.mlp_norm_b);
        x2 = ggml_gelu_erf(g, lin(g, x2, b.proj1_w, b.proj1_b)); // (U, T2)
        const int U2 = (int)hp.cgmlp_units / 2;
        ggml_tensor* xr = ggml_cont(g, ggml_view_2d(g, x2, U2, T2, x2->nb[1], 0));
        ggml_tensor* xg = ggml_view_2d(g, x2, U2, T2, x2->nb[1], (size_t)U2 * ggml_element_size(x2));
        xg = causal_dwconv(g, xg, b.csgu_conv_w, b.csgu_conv_b, (int)hp.cgmlp_kernel,
                           [&](ggml_tensor* t) { return ln(g, t, b.csgu_norm_w, b.csgu_norm_b); });
        x2 = lin(g, ggml_mul(g, xr, xg), b.proj2_w, b.proj2_b);

        // merge: merge_proj(concat + dwconv(concat)), causal, no norm
        ggml_tensor* cat = ggml_concat(g, x1, x2, 0); // (2d, T2)
        ggml_tensor* conv = causal_dwconv(g, cat, b.merge_conv_w, b.merge_conv_b, (int)hp.merge_kernel,
                                          [](ggml_tensor* t) { return t; });
        x = ggml_add(g, x, lin(g, ggml_add(g, cat, conv), b.merge_proj_w, b.merge_proj_b));

        // FFN, half step, then the per-layer final norm
        h = ln(g, x, b.ff_norm_w, b.ff_norm_b);
        h = lin(g, ggml_silu(g, lin(g, h, b.ff_w1, b.ff_b1)), b.ff_w2, b.ff_b2);
        x = ggml_add(g, x, ggml_scale(g, h, 0.5f));
        x = ln(g, x, b.final_norm_w, b.final_norm_b);
        if (dump) {
            char nm[32];
            snprintf(nm, sizeof(nm), "dump_blk_%u", il);
            ggml_tensor* s = ggml_cont(g, x);
            ggml_set_name(s, nm);
            ggml_set_output(s);
            ggml_build_forward_expand(gf, s);
        }
    }
    x = ln(g, x, m.enc_norm_w, m.enc_norm_b);
    ggml_set_name(x, "enc_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    if (out_T_enc)
        *out_T_enc = T2;
    ggml_free(g);
    return gf;
}

bool ensure_sched(dolphin_context* c) {
    if (c->sched)
        return true;
    ggml_backend_t be[2] = {c->backend, c->backend_cpu};
    c->sched = ggml_backend_sched_new(be, nullptr, c->backend != c->backend_cpu ? 2 : 1, 16384, false, false);
    return c->sched != nullptr;
}

std::vector<float> cmvn(const dolphin_model& m, const float* fb, int T) {
    const int F = (int)m.hp.n_mels;
    std::vector<float> out((size_t)T * F);
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++)
            out[(size_t)t * F + f] = (fb[(size_t)t * F + f] - m.cmvn_mean[f]) * m.cmvn_istd[f];
    return out;
}

bool run_encoder(dolphin_context* c, const float* fbank, int T, std::vector<float>& enc, int& T_enc, float** dumps,
                 int n_dumps) {
    const auto& hp = c->model.hp;
    const bool dump = dumps && n_dumps > 0;
    ggml_cgraph* gf = build_encoder(c, T, dump, &T_enc);
    if (!ensure_sched(c))
        return false;
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "dolphin: encoder graph alloc failed\n");
        return false;
    }
    const auto feats = cmvn(c->model, fbank, T);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "feats"), feats.data(), 0, feats.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "dolphin: encoder compute failed\n");
        return false;
    }
    const size_t n = (size_t)T_enc * hp.d_model;
    enc.resize(n);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "enc_out"), enc.data(), 0, n * sizeof(float));
    if (dump) {
        if (dumps[0])
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "dump_sub"), dumps[0], 0, n * sizeof(float));
        for (int i = 0; i + 1 < n_dumps && i < (int)hp.enc_layers; i++) {
            char nm[32];
            snprintf(nm, sizeof(nm), "dump_blk_%d", i);
            if (dumps[i + 1])
                ggml_backend_tensor_get(ggml_graph_get_tensor(gf, nm), dumps[i + 1], 0, n * sizeof(float));
        }
    }
    return true;
}

// Rows of `logits` (V, L) -> log-softmax in place.
void log_softmax_rows(float* x, int V, int L) {
    for (int l = 0; l < L; l++) {
        float* r = x + (size_t)l * V;
        float mx = *std::max_element(r, r + V);
        double s = 0.0;
        for (int v = 0; v < V; v++)
            s += std::exp((double)(r[v] - mx));
        const float lse = mx + (float)std::log(s);
        for (int v = 0; v < V; v++)
            r[v] -= lse;
    }
}

// Generic single-output graph runner for small graphs (CTC head, decoder).
struct graph_run {
    ggml_context* g;
    ggml_cgraph* gf;
};

graph_run new_graph(dolphin_context* c) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    graph_run r;
    r.g = ggml_init(ip);
    r.gf = ggml_new_graph_custom(r.g, 16384, false);
    return r;
}

std::vector<float> ctc_logprobs(dolphin_context* c, const float* enc, int T_enc) {
    auto& m = c->model;
    graph_run r = new_graph(c);
    ggml_tensor* e = ggml_new_tensor_2d(r.g, GGML_TYPE_F32, m.hp.d_model, T_enc);
    ggml_set_name(e, "enc");
    ggml_set_input(e);
    ggml_tensor* y = lin(r.g, e, m.ctc_w, m.ctc_b);
    ggml_set_name(y, "ctc");
    ggml_set_output(y);
    ggml_build_forward_expand(r.gf, y);
    ggml_free(r.g);
    std::vector<float> out;
    if (!ensure_sched(c))
        return out;
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, r.gf))
        return out;
    ggml_backend_tensor_set(ggml_graph_get_tensor(r.gf, "enc"), enc, 0, (size_t)m.hp.d_model * T_enc * sizeof(float));
    if (ggml_backend_sched_graph_compute(c->sched, r.gf) != GGML_STATUS_SUCCESS)
        return out;
    out.resize((size_t)m.hp.vocab * T_enc);
    ggml_backend_tensor_get(ggml_graph_get_tensor(r.gf, "ctc"), out.data(), 0, out.size() * sizeof(float));
    log_softmax_rows(out.data(), (int)m.hp.vocab, T_enc);
    return out;
}

// Decoder teacher-forcing pass: tokens (L) over encoder memory (d, T_enc).
// Returns log-softmax (V, L) — decoder_out[i][pos] in upstream terms.
std::vector<float> decoder_logprobs(dolphin_context* c, const std::vector<int32_t>& toks, const float* enc, int T_enc) {
    auto& m = c->model;
    const auto& hp = m.hp;
    const int L = (int)toks.size(), d = (int)hp.d_model, H = (int)hp.dec_heads, hd = d / H;
    graph_run r = new_graph(c);
    ggml_context* g = r.g;
    ggml_tensor* ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, L);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);
    ggml_tensor* mem = ggml_new_tensor_2d(g, GGML_TYPE_F32, d, T_enc);
    ggml_set_name(mem, "mem");
    ggml_set_input(mem);
    ggml_tensor* mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, L, L); // causal (Tk, Tq)
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    // Embedding * sqrt(d) + absolute interleaved sinusoids (PositionalEncoding).
    ggml_tensor* x = ggml_get_rows(g, m.dec_embed, ids);
    x = ggml_add(g, ggml_scale(g, x, std::sqrt((float)d)), ggml_view_2d(g, m.dec_pe, d, L, m.dec_pe->nb[1], 0));
    const float sc = 1.0f / std::sqrt((float)hd);
    for (uint32_t il = 0; il < hp.dec_layers; il++) {
        const auto& b = m.dec[il];
        ggml_tensor* h = ln(g, x, b.n1_w, b.n1_b);
        ggml_tensor* q = ggml_reshape_3d(g, lin(g, h, b.sq_w, b.sq_b), hd, H, L);
        ggml_tensor* k = ggml_reshape_3d(g, lin(g, h, b.sk_w, b.sk_b), hd, H, L);
        ggml_tensor* v = ggml_reshape_3d(g, lin(g, h, b.sv_w, b.sv_b), hd, H, L);
        x = ggml_add(g, x, lin(g, sdpa(g, q, k, v, mask, sc), b.so_w, b.so_b));

        h = ln(g, x, b.n2_w, b.n2_b);
        q = ggml_reshape_3d(g, lin(g, h, b.cq_w, b.cq_b), hd, H, L);
        k = ggml_reshape_3d(g, lin(g, mem, b.ck_w, b.ck_b), hd, H, T_enc);
        v = ggml_reshape_3d(g, lin(g, mem, b.cv_w, b.cv_b), hd, H, T_enc);
        x = ggml_add(g, x, lin(g, sdpa(g, q, k, v, nullptr, sc), b.co_w, b.co_b));

        h = ln(g, x, b.n3_w, b.n3_b);
        x = ggml_add(g, x, lin(g, ggml_relu(g, lin(g, h, b.w1, b.b1)), b.w2, b.b2));
    }
    x = ln(g, x, m.dec_norm_w, m.dec_norm_b);
    ggml_tensor* y = lin(g, x, m.dec_out_w, m.dec_out_b); // (V, L)
    ggml_set_name(y, "logits");
    ggml_set_output(y);
    ggml_build_forward_expand(r.gf, y);
    ggml_free(g);

    std::vector<float> out;
    if (!ensure_sched(c))
        return out;
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, r.gf))
        return out;
    ggml_backend_tensor_set(ggml_graph_get_tensor(r.gf, "ids"), toks.data(), 0, toks.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(r.gf, "mem"), enc, 0, (size_t)d * T_enc * sizeof(float));
    std::vector<float> mk((size_t)L * L, 0.0f);
    for (int iq = 0; iq < L; iq++)
        for (int ik = iq + 1; ik < L; ik++)
            mk[(size_t)iq * L + ik] = -INFINITY;
    ggml_backend_tensor_set(ggml_graph_get_tensor(r.gf, "mask"), mk.data(), 0, mk.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(c->sched, r.gf) != GGML_STATUS_SUCCESS)
        return out;
    out.resize((size_t)hp.vocab * L);
    ggml_backend_tensor_get(ggml_graph_get_tensor(r.gf, "logits"), out.data(), 0, out.size() * sizeof(float));
    log_softmax_rows(out.data(), (int)hp.vocab, L);
    return out;
}

// ---- WeNet CTC prefix beam search (dolphin/search.py) -----------------------

double log_add(double a, double b) {
    if (a == -INFINITY && b == -INFINITY)
        return -INFINITY;
    const double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

struct nbest_hyp {
    std::vector<int32_t> prefix;
    double score;
};

std::vector<nbest_hyp> ctc_prefix_beam(const float* logp, int T, int V, int beam, int blank) {
    struct PS {
        double s = -INFINITY, ns = -INFINITY;
        double score() const { return log_add(s, ns); }
    };
    // cur_hyps as an ordered list (Python keeps sorted order); next as an
    // insertion-ordered map (Python defaultdict) so ties sort as upstream.
    std::vector<std::pair<std::vector<int32_t>, PS>> cur;
    cur.push_back({{}, PS{0.0, -INFINITY}});
    std::vector<int> idx(V);
    for (int t = 0; t < T; t++) {
        const float* lp = logp + (size_t)t * V;
        for (int v = 0; v < V; v++)
            idx[v] = v;
        std::partial_sort(idx.begin(), idx.begin() + beam, idx.end(), [&](int a, int b) {
            // Highest log-prob first, lower index on a tie. The values
            // are read first so the tie-break on the indices does not
            // read as a bounds check after use (cppcheck
            // arrayIndexThenCheck).
            const float la = lp[a], lb = lp[b];
            return la > lb || (la == lb && a < b);
        });
        std::vector<std::pair<std::vector<int32_t>, PS>> next;
        std::map<std::vector<int32_t>, size_t> where;
        auto at = [&](const std::vector<int32_t>& k) -> PS& {
            auto it = where.find(k);
            if (it != where.end())
                return next[it->second].second;
            where[k] = next.size();
            next.push_back({k, PS{}});
            return next.back().second;
        };
        for (int j = 0; j < beam; j++) {
            const int u = idx[j];
            const double prob = lp[u];
            for (const auto& [prefix, ps] : cur) {
                const int last = prefix.empty() ? -1 : prefix.back();
                if (u == blank) {
                    PS& n = at(prefix);
                    n.s = log_add(n.s, ps.score() + prob);
                } else if (u == last) {
                    PS& n1 = at(prefix);
                    n1.ns = log_add(n1.ns, ps.ns + prob);
                    std::vector<int32_t> np = prefix;
                    np.push_back(u);
                    PS& n2 = at(np);
                    n2.ns = log_add(n2.ns, ps.s + prob);
                } else {
                    std::vector<int32_t> np = prefix;
                    np.push_back(u);
                    PS& n = at(np);
                    n.ns = log_add(n.ns, ps.score() + prob);
                }
            }
        }
        std::stable_sort(next.begin(), next.end(),
                         [](const auto& a, const auto& b) { return a.second.score() > b.second.score(); });
        if ((int)next.size() > beam)
            next.resize(beam);
        cur.swap(next);
    }
    std::vector<nbest_hyp> out;
    for (auto& [p, ps] : cur)
        out.push_back({p, ps.score()});
    return out;
}

int argmax_row(const float* r, int V) {
    return (int)(std::max_element(r, r + V) - r);
}

std::string replace_all(std::string s, const std::string& a, const std::string& b) {
    size_t p = 0;
    while ((p = s.find(a, p)) != std::string::npos) {
        s.replace(p, a.size(), b);
        p += b.size();
    }
    return s;
}

bool is_special(const std::string& t) {
    return t.size() >= 2 && t.front() == '<' && t.back() == '>';
}

} // namespace

extern "C" struct dolphin_context_params dolphin_context_default_params(void) {
    dolphin_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.beam_size = 10;
    return p;
}

extern "C" struct dolphin_context* dolphin_init_from_file(const char* path, struct dolphin_context_params params) {
    auto* c = new dolphin_context();
    c->params = params;
    c->backend_cpu = core_cpu_backend::init();
    core_cpu_backend::set_n_threads(c->backend_cpu, params.n_threads > 0 ? params.n_threads : 4);
    c->backend = params.use_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!c->backend)
        c->backend = c->backend_cpu;
    c->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    if (!load(c->model, path, c->backend)) {
        fprintf(stderr, "dolphin: failed to load '%s'\n", path);
        dolphin_free(c);
        return nullptr;
    }
    if (params.verbosity > 0)
        fprintf(stderr, "dolphin: %u enc / %u dec layers, d=%u, vocab %u, backend %s\n", c->model.hp.enc_layers,
                c->model.hp.dec_layers, c->model.hp.d_model, c->model.hp.vocab, ggml_backend_name(c->backend));
    return c;
}

extern "C" void dolphin_free(struct dolphin_context* c) {
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

extern "C" float* dolphin_compute_fbank(struct dolphin_context* c, const float* samples, int n, int* out_T) {
    core_kaldi::FbankParams p;
    p.n_mels = (int)c->model.hp.n_mels;
    p.frame_length_ms = (int)c->model.hp.frame_length_ms;
    p.frame_shift_ms = (int)c->model.hp.frame_shift_ms;
    p.int16_scale = true; // waveform * (1 << 15), as dolphin/processor.py
    int T = 0;
    std::vector<float> fb = core_kaldi::compute_fbank(samples, n, p, T);
    *out_T = T;
    float* r = (float*)malloc(fb.size() * sizeof(float));
    std::memcpy(r, fb.data(), fb.size() * sizeof(float));
    return r;
}

extern "C" float* dolphin_run_encoder(struct dolphin_context* c, const float* fbank, int T, int* out_T_enc, int* out_d,
                                      float** dumps, int n_dumps) {
    std::vector<float> enc;
    int T_enc = 0;
    if (!run_encoder(c, fbank, T, enc, T_enc, dumps, n_dumps))
        return nullptr;
    *out_T_enc = T_enc;
    *out_d = (int)c->model.hp.d_model;
    float* r = (float*)malloc(enc.size() * sizeof(float));
    std::memcpy(r, enc.data(), enc.size() * sizeof(float));
    return r;
}

extern "C" float* dolphin_ctc_logprobs(struct dolphin_context* c, const float* enc, int T_enc, int* out_V) {
    auto lp = ctc_logprobs(c, enc, T_enc);
    if (lp.empty())
        return nullptr;
    *out_V = (int)c->model.hp.vocab;
    float* r = (float*)malloc(lp.size() * sizeof(float));
    std::memcpy(r, lp.data(), lp.size() * sizeof(float));
    return r;
}

extern "C" int dolphin_n_layers(struct dolphin_context* c) {
    return (int)c->model.hp.enc_layers;
}
extern "C" int dolphin_n_mels(struct dolphin_context* c) {
    return (int)c->model.hp.n_mels;
}
extern "C" const char* dolphin_token_text(struct dolphin_context* c, int id) {
    return (id >= 0 && id < (int)c->model.vocab.size()) ? c->model.vocab[id].c_str() : "";
}

extern "C" struct dolphin_result* dolphin_transcribe_ex(struct dolphin_context* c, const float* samples, int n,
                                                        const char* lang, const char* region) {
    auto& m = c->model;
    const auto& hp = m.hp;
    int T = 0;
    float* fb = dolphin_compute_fbank(c, samples, n, &T);
    if (!fb || T <= 0) {
        free(fb);
        return nullptr;
    }
    std::vector<float> enc;
    int T_enc = 0;
    const bool ok = run_encoder(c, fb, T, enc, T_enc, nullptr, 0);
    free(fb);
    if (!ok || T_enc <= 0)
        return nullptr;
    const int V = (int)hp.vocab;
    auto ctc = ctc_logprobs(c, enc.data(), T_enc);
    if (ctc.empty())
        return nullptr;
    const int beam = c->params.beam_size > 0 ? c->params.beam_size : 10;
    auto hyps = ctc_prefix_beam(ctc.data(), T_enc, V, beam, (int)hp.blank_id);

    // Language / region: forced, or predicted greedily from [sos] (then
    // [sos, lang]) — predict_lang_region_timestamp's first two steps.
    auto tok_id = [&](const char* s) -> int {
        if (!s || !*s)
            return -1;
        auto it = m.tok2id.find(std::string("<") + s + ">");
        return it == m.tok2id.end() ? -1 : it->second;
    };
    int lang_id = tok_id(lang), region_id = tok_id(region);
    if (lang_id < 0)
        region_id = -1; // upstream only honours a region together with a language
    if (region_id < 0) {
        std::vector<int32_t> pre = {(int32_t)hp.sos_id};
        if (lang_id >= 0)
            pre.push_back(lang_id);
        while (pre.size() < 3) {
            auto lp = decoder_logprobs(c, pre, enc.data(), T_enc);
            if (lp.empty())
                return nullptr;
            pre.push_back(argmax_row(lp.data() + (size_t)(pre.size() - 1) * V, V));
        }
        lang_id = pre[1];
        region_id = pre[2];
    }

    // Attention rescoring: [sos, lang, region, <asr>, <notimestamp>, hyp...],
    // score = sum of decoder log-p over hyp tokens + eos (ctc_weight 0).
    const std::vector<int32_t> prefix = {(int32_t)hp.sos_id, (int32_t)lang_id, (int32_t)region_id, (int32_t)hp.asr_id,
                                         (int32_t)hp.notimestamp_id};
    const int off = (int)prefix.size() - 1;
    double best = -INFINITY;
    size_t best_i = 0;
    for (size_t i = 0; i < hyps.size(); i++) {
        std::vector<int32_t> in = prefix;
        in.insert(in.end(), hyps[i].prefix.begin(), hyps[i].prefix.end());
        auto lp = decoder_logprobs(c, in, enc.data(), T_enc);
        if (lp.empty())
            return nullptr;
        double s = 0.0;
        const auto& h = hyps[i].prefix;
        for (size_t j = 0; j < h.size(); j++)
            s += lp[(size_t)(j + off) * V + h[j]];
        s += lp[(size_t)(h.size() + off) * V + hp.eos_id];
        if (s > best) {
            best = s;
            best_i = i;
        }
        if (c->params.verbosity > 1)
            fprintf(stderr, "dolphin: hyp %zu ctc=%.4f att=%.4f len=%zu\n", i, hyps[i].score, s, h.size());
    }

    std::vector<int32_t> toks(prefix.begin() + 1, prefix.end());
    if (!hyps.empty())
        toks.insert(toks.end(), hyps[best_i].prefix.begin(), hyps[best_i].prefix.end());
    // _filter_nonspecial_tokens: a token is text iff its id is above <30.00>
    // (the last timestamp token); fall back to the <...> shape if absent.
    const auto last_time = m.tok2id.find("<30.00>");
    std::string raw, clean;
    for (int32_t t : toks) {
        const std::string& s = m.vocab[t];
        raw += s;
        const bool special = last_time != m.tok2id.end() ? t <= last_time->second : is_special(s);
        if (!special)
            clean += s;
    }
    // Upstream's CharTokenizer joins pieces as-is, leaving SentencePiece's
    // word marker in English output; render it as a space for display.
    clean = replace_all(clean, "\xE2\x96\x81", " ");
    while (!clean.empty() && clean.front() == ' ')
        clean.erase(clean.begin());

    auto* r = (dolphin_result*)calloc(1, sizeof(dolphin_result));
    r->text = strdup(clean.c_str());
    r->raw_text = strdup(raw.c_str());
    auto strip = [](const std::string& s) { return s.size() >= 2 ? s.substr(1, s.size() - 2) : s; };
    snprintf(r->language, sizeof(r->language), "%s", strip(m.vocab[lang_id]).c_str());
    snprintf(r->region, sizeof(r->region), "%s", strip(m.vocab[region_id]).c_str());
    r->n_tokens = (int)toks.size();
    r->tokens = (int32_t*)malloc(toks.size() * sizeof(int32_t));
    std::memcpy(r->tokens, toks.data(), toks.size() * sizeof(int32_t));
    return r;
}

extern "C" void dolphin_result_free(struct dolphin_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->raw_text);
    free(r->tokens);
    free(r);
}

extern "C" char* dolphin_transcribe(struct dolphin_context* c, const float* samples, int n) {
    dolphin_result* r = dolphin_transcribe_ex(c, samples, n, nullptr, nullptr);
    if (!r)
        return nullptr;
    char* t = strdup(r->text);
    dolphin_result_free(r);
    return t;
}
