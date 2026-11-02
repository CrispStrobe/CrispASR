// See mossformer2_blocks.h.  Every function cites the upstream module it ports
// (models/mossformer2_ss/*.py, identical in models/mossformer2_se/ apart from
// the FSMN variant; ClearerVoice-Studio @6b3774dc).
#include "mossformer2_blocks.h"

#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace mf2 {

namespace {

void stage(Ctx& c, const std::string& name, ggml_tensor* x) {
    if (c.capture && !name.empty())
        c.capture(name, x);
}

// Broadcast affine weights shaped [1, C] for a time-major [T, C] operand.
ggml_tensor* col(Ctx& c, const std::string& name) {
    ggml_tensor* w = as_f32(c.g, c.weight(name));
    return ggml_reshape_2d(c.g, w, 1, ggml_nelements(w));
}

// nn.InstanceNorm2d(C, affine=True) on a [B, C, T, 1] map == per-channel
// normalisation over time; here on a time-major [T, C] tensor (rows = time).
ggml_tensor* instance_norm_tmajor(Ctx& c, ggml_tensor* xt, const std::string& prefix) {
    ggml_tensor* y = ggml_norm(c.g, xt, 1e-5f);
    y = ggml_mul(c.g, y, col(c, prefix + ".weight"));
    return ggml_add(c.g, y, col(c, prefix + ".bias"));
}

ggml_tensor* cont(ggml_context* g, ggml_tensor* x) {
    return ggml_is_contiguous(x) ? x : ggml_cont(g, x);
}

ggml_tensor* ssm_conv_2d(ggml_context* g, ggml_tensor* xp, ggml_tensor* k) {
    const int64_t tp = xp->ne[0], ch = xp->ne[1];
    ggml_tensor* y = ggml_ssm_conv(g, ggml_reshape_3d(g, xp, tp, ch, 1), k);
    return ggml_reshape_2d(g, y, ch, tp - k->ne[0] + 1);
}

} // namespace

ggml_tensor* as_f32(ggml_context* g, ggml_tensor* t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(g, t, GGML_TYPE_F32);
}

ggml_tensor* linear(Ctx& c, ggml_tensor* x, const std::string& prefix, bool bias) {
    ggml_tensor* y = ggml_mul_mat(c.g, c.weight(prefix + ".weight"), x);
    if (bias)
        y = ggml_add(c.g, y, as_f32(c.g, c.weight(prefix + ".bias")));
    return y;
}

ggml_tensor* layer_norm(Ctx& c, ggml_tensor* x, const std::string& prefix, float eps) {
    ggml_tensor* y = ggml_norm(c.g, x, eps);
    y = ggml_mul(c.g, y, as_f32(c.g, c.weight(prefix + ".weight")));
    return ggml_add(c.g, y, as_f32(c.g, c.weight(prefix + ".bias")));
}

ggml_tensor* global_group_norm(Ctx& c, ggml_tensor* x, const std::string& prefix, float eps) {
    const int64_t ch = x->ne[0], t = x->ne[1];
    ggml_tensor* y = ggml_norm(c.g, ggml_reshape_1d(c.g, ggml_cont(c.g, x), ch * t), eps);
    y = ggml_reshape_2d(c.g, y, ch, t);
    y = ggml_mul(c.g, y, as_f32(c.g, c.weight(prefix + ".weight")));
    return ggml_add(c.g, y, as_f32(c.g, c.weight(prefix + ".bias")));
}

ggml_tensor* scale_norm(Ctx& c, ggml_tensor* x, const std::string& g_name, float eps) {
    // torch.norm(x, dim=-1) * dim**-0.5 == sqrt(mean(x^2)); clamp(min=eps).
    ggml_tensor* rms = ggml_sqrt(c.g, ggml_mean(c.g, ggml_sqr(c.g, x)));
    rms = ggml_clamp(c.g, rms, eps, FLT_MAX);
    ggml_tensor* y = ggml_div(c.g, x, rms);
    return ggml_mul(c.g, y, as_f32(c.g, c.weight(g_name)));
}

ggml_tensor* prelu(ggml_context* g, ggml_tensor* x, ggml_tensor* a) {
    ggml_tensor* pos = ggml_relu(g, x);
    ggml_tensor* neg = ggml_sub(g, x, pos);
    return ggml_add(g, pos, ggml_mul(g, neg, as_f32(g, a)));
}

std::vector<int> portable_kernel_split(int K) {
    static const int sizes[] = {15, 9, 5, 4, 3};
    if (K < 3)
        return {K};
    // Minimal number of pieces (coin change over the supported widths).
    std::vector<int> best(K + 1, 1 << 20), pick(K + 1, 0);
    best[0] = 0;
    for (int k = 1; k <= K; ++k)
        for (int s : sizes)
            if (s <= k && best[k - s] + 1 < best[k]) {
                best[k] = best[k - s] + 1;
                pick[k] = s;
            }
    if (best[K] >= (1 << 20))
        return {K};
    std::vector<int> out;
    for (int k = K; k > 0; k -= pick[k])
        out.push_back(pick[k]);
    return out;
}

namespace {
// CPU dwconv: dst [C, Tout] (channels innermost), x [C * gin, T], w [C * gin, K]
// tap-major.  Threads split output time; the channel loop vectorises.
struct DwParams {
    int gin, dil, pad_l;
};
void dwconv_cpu_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const intptr_t enc = (intptr_t)userdata;
    const DwParams p{(int)(enc & 0xff), (int)((enc >> 8) & 0xff), (int)(enc >> 16)};
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* w = dst->src[1];
    const int64_t CI = x->ne[0], T = x->ne[1], K = w->ne[0] == CI ? w->ne[1] : 0, C = dst->ne[0], Tout = dst->ne[1];
    GGML_ASSERT(K > 0 && C * p.gin == CI && x->nb[0] == sizeof(float) && w->nb[0] == sizeof(float));
    const int64_t per = (Tout + nth - 1) / nth, t0 = per * ith, t1 = std::min(Tout, t0 + per);
    std::vector<float> acc((size_t)CI);
    for (int64_t t = t0; t < t1; ++t) {
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int64_t k = 0; k < K; ++k) {
            const int64_t src = t + k * p.dil - p.pad_l;
            if (src < 0 || src >= T)
                continue;
            const float* xr = (const float*)((const char*)x->data + src * x->nb[1]);
            const float* wr = (const float*)((const char*)w->data + k * w->nb[1]);
            float* a = acc.data();
            for (int64_t i = 0; i < CI; ++i)
                a[i] += wr[i] * xr[i];
        }
        float* y = (float*)((char*)dst->data + t * dst->nb[1]);
        if (p.gin == 1) {
            std::copy(acc.begin(), acc.end(), y);
        } else {
            for (int64_t ch = 0; ch < C; ++ch) {
                float v = 0.0f;
                for (int j = 0; j < p.gin; ++j)
                    v += acc[(size_t)(ch * p.gin + j)];
                y[ch] = v;
            }
        }
    }
}

const std::map<std::string, ggml_tensor*>::const_iterator find_derived(Ctx& c, const std::string& name) {
    if (!c.derived)
        throw std::runtime_error("mossformer2: derived kernels were not prepared");
    const auto it = c.derived->find(name);
    if (it == c.derived->end())
        throw std::runtime_error("mossformer2: missing derived kernel " + name);
    return it;
}

// x_t [T', C] time-major, already padded; one ssm_conv per supported width.
ggml_tensor* ssm_conv_split(Ctx& c, ggml_tensor* xp, ggml_tensor* kernel) {
    ggml_context* g = c.g;
    const int K = (int)kernel->ne[0];
    const int64_t ch = xp->ne[1], n_out = xp->ne[0] - K + 1;
    ggml_tensor* y = nullptr;
    int off = 0;
    for (int k : portable_kernel_split(K)) {
        ggml_tensor* xs = ggml_cont(g, ggml_view_2d(g, xp, n_out + k - 1, ch, xp->nb[1], (size_t)off * sizeof(float)));
        ggml_tensor* ks = ggml_cont(g, ggml_view_2d(g, kernel, k, ch, kernel->nb[1], (size_t)off * sizeof(float)));
        ggml_tensor* part = ssm_conv_2d(g, xs, ks);
        y = y ? ggml_add(g, y, part) : part;
        off += k;
    }
    return y;
}
} // namespace

ggml_tensor* dwconv(Ctx& c, ggml_tensor* x, const std::string& kernel, int gin, int dil, int pad_l, int pad_r) {
    ggml_context* g = c.g;
    const int64_t CI = x->ne[0], T = x->ne[1], C = CI / gin;
    if (!c.portable_dwconv) {
        ggml_tensor* w = find_derived(c, tap_name(kernel))->second; // [CI, K]
        if (w->ne[0] != CI || gin < 1 || gin > 255 || dil < 1 || dil > 255 || pad_l < 0)
            throw std::runtime_error("mossformer2: dwconv shape mismatch for " + kernel);
        const int64_t Tout = T + pad_l + pad_r - (w->ne[1] - 1) * dil;
        ggml_tensor* args[] = {cont(g, x), w};
        const intptr_t enc = ((intptr_t)pad_l << 16) | ((intptr_t)dil << 8) | (intptr_t)gin;
        return ggml_custom_4d(g, GGML_TYPE_F32, C, Tout, 1, 1, args, 2, dwconv_cpu_op, GGML_N_TASKS_MAX, (void*)enc);
    }
    // Portable: time-major padded copy per grouped input, ggml_ssm_conv pieces.
    ggml_tensor* xt = ggml_cont(g, ggml_transpose(g, x)); // [T, CI]
    ggml_tensor* y = nullptr;
    for (int j = 0; j < gin; ++j) {
        ggml_tensor* in =
            gin == 1 ? xt : ggml_cont(g, ggml_view_2d(g, xt, T, C, gin * xt->nb[1], (size_t)j * xt->nb[1]));
        ggml_tensor* k =
            (gin == 1 && dil == 1) ? as_f32(g, c.weight(kernel)) : find_derived(c, dilated_name(kernel, j))->second;
        if (k->ne[1] != C)
            throw std::runtime_error("mossformer2: depthwise kernel/channel mismatch for " + kernel);
        ggml_tensor* xp = ggml_pad_ext(g, in, pad_l, pad_r, 0, 0, 0, 0, 0, 0);
        ggml_tensor* part = ssm_conv_split(c, xp, k);
        y = y ? ggml_add(g, y, part) : part;
    }
    return y;
}

ggml_tensor* ffconvm(Ctx& c, ggml_tensor* x, const std::string& prefix, Norm norm, const BlockHParams& hp) {
    // FFConvM.mdl = [norm, Linear, SiLU, ConvModule, Dropout]
    ggml_tensor* h =
        norm == Norm::Scale ? scale_norm(c, x, prefix + ".norm.g") : layer_norm(c, x, prefix + ".norm", 1e-5f);
    h = ggml_silu(c.g, linear(c, h, prefix + ".linear"));
    // ConvModule: inputs + DepthwiseConv1d(k=17, padding=8)(inputs^T)^T
    const int pad = (hp.conv_kernel - 1) / 2;
    return ggml_add(c.g, h, dwconv(c, h, prefix + ".dw.weight", 1, 1, pad, pad));
}

ggml_tensor* flash_layer(Ctx& c, ggml_tensor* x, const std::string& prefix, const BlockHParams& hp,
                         const std::string& st) {
    ggml_context* g = c.g;
    const int64_t D = x->ne[0], T = x->ne[1], half = D / 2;
    // Token shift: the first half of the channels moves one step later in time.
    ggml_tensor* shifted;
    if (T > 1) {
        shifted = ggml_view_2d(g, x, half, T - 1, x->nb[1], 0);
        shifted = ggml_pad_ext(g, shifted, 0, 0, 1, 0, 0, 0, 0, 0);
    } else {
        shifted = ggml_scale(g, ggml_cont(g, ggml_view_2d(g, x, half, T, x->nb[1], 0)), 0.0f);
    }
    ggml_tensor* pass = ggml_view_2d(g, x, D - half, T, x->nb[1], (size_t)half * ggml_element_size(x));
    ggml_tensor* xs = ggml_concat(g, shifted, pass, 0);

    ggml_tensor* hidden = ffconvm(c, xs, prefix + ".hidden", Norm::Scale, hp); // [hidden, T] = cat(v, u)
    ggml_tensor* qk = ffconvm(c, xs, prefix + ".qk", Norm::Scale, hp);         // [qk, T]
    stage(c, st.empty() ? "" : st + "_to_hidden", hidden);
    stage(c, st.empty() ? "" : st + "_to_qk", qk);

    // OffsetScale(heads=4) -> quad_q, lin_q, quad_k, lin_k; then RoPE on each
    // (rotary_embedding_torch: interleaved pairs over the first rotary_dim).
    const int64_t QK = qk->ne[0];
    ggml_tensor* q4 = ggml_repeat_4d(g, ggml_reshape_3d(g, qk, QK, 1, T), QK, 4, T, 1);
    q4 = ggml_mul(g, q4, ggml_reshape_3d(g, as_f32(g, c.weight(prefix + ".qk_os.gamma")), QK, 4, 1));
    q4 = ggml_add(g, q4, ggml_reshape_3d(g, as_f32(g, c.weight(prefix + ".qk_os.beta")), QK, 4, 1));
    q4 = ggml_rope_ext(g, q4, c.positions, nullptr, hp.rotary_dim, GGML_ROPE_TYPE_NORMAL, 0, hp.rotary_theta, 1.0f,
                       0.0f, 1.0f, 0.0f, 0.0f);
    auto head = [&](int h) { return ggml_view_2d(g, q4, QK, T, q4->nb[2], (size_t)h * q4->nb[1]); };
    ggml_tensor *quad_q = head(0), *lin_q = head(1), *quad_k = head(2), *lin_k = head(3);

    const int64_t G = (T + hp.group - 1) / hp.group, Tp = G * hp.group, pad = Tp - T;
    auto grouped = [&](ggml_tensor* t) { // [C, T] -> [C, group, G], zero padded
        if (pad == 0)
            return ggml_reshape_3d(g, cont(g, t), t->ne[0], hp.group, G);
        // CPU pad reads strided views directly; other backends get a dense copy.
        t = ggml_pad_ext(g, c.portable_dwconv ? cont(g, t) : t, 0, 0, 0, (int)pad, 0, 0, 0, 0);
        return ggml_reshape_3d(g, t, t->ne[0], hp.group, G);
    };
    // Quadratic attention within groups: relu(q.k / group)^2.
    ggml_tensor* sim = ggml_mul_mat(g, grouped(quad_k), grouped(quad_q)); // [j, i, G]
    ggml_tensor* attn = ggml_sqr(g, ggml_relu(g, ggml_scale(g, sim, 1.0f / (float)hp.group)));
    // Values (v and u share the attention): hidden with time innermost per group.
    ggml_tensor* hT = ggml_cont(g, ggml_permute(g, grouped(hidden), 1, 0, 2, 3)); // [group, H, G]
    ggml_tensor* quad = ggml_mul_mat(g, hT, attn);                                // [H, group(i), G]
    quad = ggml_reshape_2d(g, quad, hidden->ne[0], Tp);
    // Non-causal linear attention over the whole sequence: (lin_k^T [v|u]) / n.
    ggml_tensor* lkT = ggml_cont(g, ggml_permute(g, grouped(lin_k), 1, 0, 2, 3)); // [group, QK, G]
    ggml_tensor* kv_g = ggml_mul_mat(g, lkT, hT);                                 // [QK, H, G]
    ggml_tensor* kv = nullptr;
    for (int64_t i = 0; i < G; ++i) {
        ggml_tensor* part = ggml_view_2d(g, kv_g, QK, hidden->ne[0], kv_g->nb[1], (size_t)i * kv_g->nb[2]);
        kv = kv ? ggml_add(g, kv, part) : ggml_cont(g, part);
    }
    kv = ggml_scale(g, kv, 1.0f / (float)T);
    ggml_tensor* lin = ggml_mul_mat(g, kv, ggml_cont(g, lin_q)); // [H, T]
    ggml_tensor* att = ggml_add(g, ggml_view_2d(g, quad, hidden->ne[0], T, quad->nb[1], 0), lin);

    const int64_t H2 = hidden->ne[0] / 2;
    auto rows = [&](ggml_tensor* t, int64_t r0) {
        return ggml_view_2d(g, t, H2, T, t->nb[1], (size_t)r0 * sizeof(float));
    };
    ggml_tensor *att_v = rows(att, 0), *att_u = rows(att, H2), *v = rows(hidden, 0), *u = rows(hidden, H2);
    // out = (att_u * v) * sigmoid(att_v * u)
    ggml_tensor* out = ggml_mul(g, ggml_mul(g, att_u, v), ggml_sigmoid(g, ggml_mul(g, att_v, u)));
    return ggml_add(g, x, ffconvm(c, out, prefix + ".out", Norm::Scale, hp));
}

ggml_tensor* gated_fsmn_block(Ctx& c, ggml_tensor* x, const std::string& prefix, const BlockHParams& hp,
                              const std::string& st) {
    ggml_context* g = c.g;
    // conv1 = Conv1d(dim, inner, 1) + PReLU(); norm1 = CLayerNorm(inner) (eps 1e-5)
    ggml_tensor* c1 = prelu(g, linear(c, x, prefix + ".conv1"), c.weight(prefix + ".conv1.prelu"));
    stage(c, st.empty() ? "" : st + "_fsmn_conv1", c1);
    ggml_tensor* n1 = layer_norm(c, c1, prefix + ".norm1", 1e-5f);

    // Gated_FSMN(_dilated): x_u = to_u(x), x_v = to_v(x), x_u = fsmn(x_u), x_v * x_u + x
    ggml_tensor* xu = ffconvm(c, n1, prefix + ".to_u", Norm::Layer, hp);
    ggml_tensor* xv = ffconvm(c, n1, prefix + ".to_v", Norm::Layer, hp);
    ggml_tensor* p1 = ggml_relu(g, linear(c, xu, prefix + ".fsmn.linear"));
    p1 = linear(c, p1, prefix + ".fsmn.project", /*bias=*/false); // [C, T]
    const int lo = hp.fsmn_lorder;
    if (!hp.fsmn_dilated) {
        // UniDeepFsmn: input + (p1 + Conv2d([2*lorder-1, 1], groups=C)(pad(p1, lorder-1)))
        ggml_tensor* mem = ggml_add(g, p1, dwconv(c, p1, prefix + ".mem.weight", 1, 1, lo - 1, lo - 1));
        ggml_tensor* uni = ggml_add(g, xu, mem);
        stage(c, st.empty() ? "" : st + "_fsmn_uni", uni);
        ggml_tensor* gated = ggml_add(g, ggml_mul(g, xv, uni), n1);
        stage(c, st.empty() ? "" : st + "_fsmn_gated", gated);
        ggml_tensor* n2 = layer_norm(c, gated, prefix + ".norm2", 1e-5f);
        return ggml_add(g, linear(c, n2, prefix + ".conv2"), x);
    }
    // UniDeepFsmn_dilated: input + DilatedDenseNet(project(relu(linear(input))))
    // DilatedDenseNet(depth=2): conv1 (k=2*lorder-1, dil 1) -> IN -> PReLU,
    // conv2 over cat([out1, x]) (groups=C, 2 inputs each, dil 2) -> IN -> PReLU.
    // InstanceNorm2d statistics run over time, so they use a time-major copy.
    auto in_prelu = [&](ggml_tensor* y, const std::string& norm, const std::string& slope) {
        ggml_tensor* yt = instance_norm_tmajor(c, ggml_cont(g, ggml_transpose(g, y)), prefix + norm);
        yt = prelu(g, yt, col(c, prefix + slope));
        return ggml_cont(g, ggml_transpose(g, yt)); // back to [C, T]
    };
    ggml_tensor* o1 =
        in_prelu(dwconv(c, p1, prefix + ".ddn.conv1.weight", 1, 1, lo - 1, lo - 1), ".ddn.norm1", ".ddn.prelu1");
    const int pad2 = lo + (2 - 1) * (lo - 1) - 1; // pad_length for dil = 2
    ggml_tensor* cat = ggml_concat(g, o1, p1, 0); // [2C, T]; output channel i reads 2i, 2i + 1
    ggml_tensor* ddn =
        in_prelu(dwconv(c, cat, prefix + ".ddn.conv2.weight", 2, 2, pad2, pad2), ".ddn.norm2", ".ddn.prelu2");
    stage(c, st.empty() ? "" : st + "_fsmn_ddn", ddn);
    ggml_tensor* gated = ggml_add(g, ggml_mul(g, xv, ggml_add(g, xu, ddn)), n1);
    stage(c, st.empty() ? "" : st + "_fsmn_gated", gated);
    // norm2 = CLayerNorm(inner); conv2 = Conv1d(inner, dim, 1); + input
    ggml_tensor* n2 = layer_norm(c, gated, prefix + ".norm2", 1e-5f);
    return ggml_add(g, linear(c, n2, prefix + ".conv2"), x);
}

ggml_tensor* mossformer_m(Ctx& c, ggml_tensor* x, const BlockHParams& hp, const std::string& final_norm) {
    for (int i = 0; i < hp.n_layers; ++i) {
        const std::string l = "l" + std::to_string(i);
        const std::string b = "blk." + std::to_string(i);
        // Sub-stage captures are only exposed for layer 0.
        x = flash_layer(c, x, b + ".flash", hp, i == 0 ? l : "");
        stage(c, l + "_flash", x);
        x = gated_fsmn_block(c, x, b + ".fsmn", hp, i == 0 ? l : "");
        stage(c, l + "_fsmn", x);
    }
    return layer_norm(c, x, final_norm, 1e-6f);
}

ggml_tensor* computation_block(Ctx& c, ggml_tensor* x, const BlockHParams& hp, const std::string& final_norm,
                               const std::string& group_norm) {
    ggml_tensor* intra = mossformer_m(c, x, hp, final_norm);
    stage(c, "intra_out", intra);
    return ggml_add(c.g, global_group_norm(c, intra, group_norm, 1e-8f), x);
}

std::vector<float> sinu_embedding(int T, const std::vector<float>& inv_freq, float scale) {
    const size_t half = inv_freq.size(), dim = 2 * half;
    std::vector<float> out((size_t)T * dim);
    for (int t = 0; t < T; ++t) {
        float* row = out.data() + (size_t)t * dim;
        for (size_t j = 0; j < half; ++j) {
            const float a = (float)t * inv_freq[j]; // float32 product, as torch einsum
            row[j] = std::sin(a) * scale;
            row[half + j] = std::cos(a) * scale;
        }
    }
    return out;
}

std::vector<float> tap_kernel(const float* w, int C, int gin, int K) {
    std::vector<float> out((size_t)C * gin * K);
    for (int ch = 0; ch < C; ++ch)
        for (int j = 0; j < gin; ++j)
            for (int k = 0; k < K; ++k)
                out[(size_t)k * C * gin + (size_t)ch * gin + j] = w[((size_t)ch * gin + j) * K + k];
    return out;
}

std::vector<std::vector<float>> dilate_kernel(const float* w, int C, int gin, int K, int dil) {
    const int Kd = (K - 1) * dil + 1;
    std::vector<std::vector<float>> out(gin, std::vector<float>((size_t)C * Kd, 0.0f));
    for (int ch = 0; ch < C; ++ch)
        for (int j = 0; j < gin; ++j)
            for (int k = 0; k < K; ++k)
                out[j][(size_t)ch * Kd + (size_t)k * dil] = w[((size_t)ch * gin + j) * K + k];
    return out;
}

std::string tap_name(const std::string& kernel) {
    return kernel + ".tap";
}

std::string dilated_name(const std::string& kernel, int input) {
    return kernel + ".dil" + std::to_string(input);
}

std::vector<DwKernel> dw_kernels(const BlockHParams& hp) {
    std::vector<DwKernel> out;
    for (int i = 0; i < hp.n_layers; ++i) {
        const std::string b = "blk." + std::to_string(i);
        for (const char* m : {".flash.hidden", ".flash.qk", ".flash.out", ".fsmn.to_u", ".fsmn.to_v"})
            out.push_back({b + m + ".dw.weight", 1, 1});
        if (!hp.fsmn_dilated) {
            out.push_back({b + ".fsmn.mem.weight", 1, 1});
            continue;
        }
        out.push_back({b + ".fsmn.ddn.conv1.weight", 1, 1});
        out.push_back({b + ".fsmn.ddn.conv2.weight", 2, 2});
    }
    return out;
}

} // namespace mf2
