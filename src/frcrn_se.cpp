// frcrn_se.cpp — ClearerVoice-Studio FRCRN_SE_16K, native ggml forward.
//
// Upstream: modelscope/ClearerVoice-Studio clearvoice/models/frcrn_se
// (frcrn.py DCCRN, unet.py, complex_nn.py, se_layer.py, conv_stft.py) and
// clearvoice/utils/decode.py decode_one_audio_frcrn_se_16k.
//
// Activation layout inside the graph: ggml ne = [T, F, C, 2] (frame fastest,
// then frequency bin, channel, re/im part) — numpy (2, C, F, T), which is
// PyTorch's [1, C, F, T, 2] with the complex axis moved to the front.
//
// * Complex conv: both re and im parts go through the same real kernel as a
//   batch of 2 (im2col + mul_mat), then real = Wre*xr - Wim*xi and
//   imag = Wre*xi + Wim*xr. Conv bias + eval BatchNorm are a folded affine.
// * Complex transposed conv (decoders): the converter rewrote each one as a
//   stride-1 polyphase conv; here the two frequency phases are interleaved.
// * FSMN memories: Linear -> ReLU -> Linear(project) -> causal depthwise conv
//   (ggml_ssm_conv), plus the two residuals. The per-level FSMNs run along
//   frequency (per frame); the bottleneck FSMN runs along time.
// * SE: global mean over (F, T) per channel and part, complex gate.
// * Large per-layer work is split into time chunks (all those ops are local
//   in time) and written into the full tensor with ggml_set_inplace, which
//   bounds the im2col workspace for long one-shot inputs (upstream decodes up
//   to 120 s in one pass; SE pooling makes that pass non-separable).

#include "frcrn_se.h"

#include "crispasr_enhancer.h"

#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kLevels = 7;

struct FsmnUnit {
    ggml_tensor* lin_w = nullptr;
    ggml_tensor* lin_b = nullptr;
    ggml_tensor* proj_w = nullptr;
    ggml_tensor* mem = nullptr; // [order, C]
};

struct ComplexFsmn {
    FsmnUnit re, im;
};

struct SeLayer {
    ggml_tensor *r1w = nullptr, *r1b = nullptr, *r2w = nullptr, *r2b = nullptr;
    ggml_tensor *i1w = nullptr, *i1b = nullptr, *i2w = nullptr, *i2b = nullptr;
};

struct ComplexConv {
    ggml_tensor* w_re = nullptr; // [K, Cout] (decoder: [K, 2*Cout])
    ggml_tensor* w_im = nullptr;
    ggml_tensor* scale = nullptr; // [C, 2]
    ggml_tensor* shift = nullptr;
    int kf = 0;
    int cin = 0;
    int cout = 0;
};

struct Unet {
    ComplexConv enc[kLevels], dec[kLevels];
    SeLayer se_enc[kLevels], se_dec[kLevels];
    ComplexFsmn fsmn_enc[kLevels], fsmn_dec[kLevels];
    FsmnUnit mid_re1, mid_im1, mid_re2, mid_im2;
    float lin[4] = {1, 0, 0, 0}; // w_re, b_re, w_im, b_im
};

struct Hparams {
    int sample_rate = 16000;
    int win_len = 640;
    int win_inc = 320;
    int fft_len = 640;
    int channels = 128;
    int fsmn_order = 20;
    float leaky = 0.01f;
    float decode_window_s = 1.0f;
    float decode_stride_ratio = 0.75f;
    float one_time_s = 120.0f;
    float norm_db = -25.0f;
    std::vector<int> enc_kf, dec_kf;
};

bool env_true(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

void write_npy(const std::string& path, const std::vector<float>& data, const int64_t* ne, int n_dims) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f)
        return;
    std::string shape = "(";
    for (int i = n_dims - 1; i >= 0; i--) {
        shape += std::to_string(ne[i]);
        shape += (n_dims == 1 || i > 0) ? "," : "";
        if (i > 0)
            shape += " ";
    }
    shape += ")";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }";
    size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header += "\n";
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    std::fwrite(magic, 1, 8, f);
    const uint16_t hl = (uint16_t)header.size();
    std::fwrite(&hl, 2, 1, f);
    std::fwrite(header.data(), 1, header.size(), f);
    std::fwrite(data.data(), sizeof(float), data.size(), f);
    std::fclose(f);
}

// CRISPASR_FRCRN_SE_PROFILE=1: per-op wall time (forces node-by-node sync).
struct OpProfile {
    std::map<std::string, std::pair<double, int>> by_op;
    std::chrono::steady_clock::time_point last;
};

bool profile_cb(ggml_tensor* t, bool ask, void* ud) {
    if (ask)
        return true;
    auto* p = (OpProfile*)ud;
    const auto now = std::chrono::steady_clock::now();
    std::string key = ggml_op_desc(t);
    if (t->op == GGML_OP_MUL_MAT)
        key += std::string("[") + ggml_type_name(t->src[0]->type) + "]";
    auto& e = p->by_op[key];
    e.first += std::chrono::duration<double>(now - p->last).count();
    e.second++;
    p->last = now;
    return true;
}

} // namespace

struct frcrn_se_context {
    Hparams hp;
    frcrn_se_params params;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu = nullptr;
    core_gguf::WeightLoad weights;
    ggml_tensor* stft_fwd = nullptr;
    ggml_tensor* stft_inv = nullptr;
    std::vector<float> window;
    Unet unet[2];

    // Graph cache (keyed on frame count).
    int graph_frames = -1;
    bool graph_dump = false;
    ggml_context* gctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_backend_sched_t sched = nullptr;
    size_t sched_capacity = 0;
    ggml_tensor* in_frames = nullptr;
    ggml_tensor* out_frames = nullptr;
    std::vector<std::pair<std::string, ggml_tensor*>> stages;
    size_t chunk_bytes = 96u << 20;
    std::string dump_dir;
    bool dumped = false;

    ~frcrn_se_context() {
        if (sched)
            ggml_backend_sched_free(sched);
        if (gctx)
            ggml_free(gctx);
        core_gguf::free_weights(weights);
        if (backend && backend != cpu)
            ggml_backend_free(backend);
        if (cpu)
            ggml_backend_free(cpu);
    }
};

namespace {

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

struct Builder {
    frcrn_se_context& m;
    ggml_context* ctx;
    bool dump;

    void stage(const std::string& name, ggml_tensor* t) {
        if (!dump)
            return;
        ggml_set_name(t, name.c_str());
        ggml_set_output(t);
        m.stages.emplace_back(name, t);
    }

    // Part p (0 = re, 1 = im) of a [a, b, c, 2] tensor as a contiguous 3-D view.
    ggml_tensor* part(ggml_tensor* t, int p) {
        return ggml_view_3d(ctx, t, t->ne[0], t->ne[1], t->ne[2], t->nb[1], t->nb[2], (size_t)p * t->nb[3]);
    }

    // [C, 2] parameter -> [1, 1, C, 2] broadcast operand.
    ggml_tensor* bcast_tfc(ggml_tensor* p) { return ggml_reshape_4d(ctx, p, 1, 1, p->ne[0], 2); }

    // Frames [t0, t0+n) of x [T, F, C, 2], contiguous.
    ggml_tensor* frames(ggml_tensor* x, int64_t t0, int64_t n) {
        if (t0 == 0 && n == x->ne[0])
            return x;
        return ggml_cont(ctx, ggml_view_4d(ctx, x, n, x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3],
                                           (size_t)t0 * x->nb[0]));
    }

    // Run `fn(t0, n)` (producing [n, F, C, 2]) over [0, T) in chunks of
    // `chunk` frames and assemble the result.
    template <class Fn> ggml_tensor* chunked(int64_t T, int64_t F, int64_t C, int64_t chunk, Fn fn) {
        if (chunk >= T)
            return fn(0, T);
        // The first chunk zero-padded to T frames is the destination. It must
        // be a graph node, not a leaf: ggml-alloc keeps leafs alive for the
        // whole graph, which would pin every chunked layer output at once.
        ggml_tensor* cur = ggml_pad_ext(ctx, fn(0, chunk), 0, (int)(T - chunk), 0, 0, 0, 0, 0, 0);
        GGML_ASSERT(cur->ne[1] == F && cur->ne[2] == C);
        for (int64_t t0 = chunk; t0 < T; t0 += chunk) {
            const int64_t n = std::min(chunk, T - t0);
            ggml_tensor* piece = fn(t0, n);
            cur = ggml_set_inplace(ctx, cur, piece, cur->nb[1], cur->nb[2], cur->nb[3], (size_t)t0 * cur->nb[0]);
        }
        return cur;
    }

    int64_t chunk_for(size_t bytes_per_frame) const {
        const size_t c = m.chunk_bytes / std::max<size_t>(1, bytes_per_frame);
        return (int64_t)std::max<size_t>(16, c);
    }

    // Complex 2-D conv of xc [Tin, Fin, Cin, 2] (contiguous). Returns the
    // real and imaginary products, each [Cw, OW, OH] channel-first.
    std::pair<ggml_tensor*, ggml_tensor*> complex_mm(ggml_tensor* xc, const ComplexConv& c, int kh, int kw, int s1,
                                                     int p1) {
        // im2col only reads the kernel's shape.
        ggml_tensor* shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kw, kh, xc->ne[2]);
        ggml_tensor* col = ggml_im2col(ctx, shape, xc, 1, s1, 0, p1, 1, 1, true, GGML_TYPE_F32);
        // col: [K, OW, OH, 2]
        const int64_t K = col->ne[0], OW = col->ne[1], OH = col->ne[2];
        ggml_tensor* col2 = ggml_reshape_2d(ctx, col, K, OW * OH * 2);
        ggml_tensor* A = ggml_mul_mat(ctx, c.w_re, col2); // [Cw, OW*OH*2]
        ggml_tensor* B = ggml_mul_mat(ctx, c.w_im, col2);
        const int64_t Cw = A->ne[0];
        A = ggml_reshape_4d(ctx, A, Cw, OW, OH, 2);
        B = ggml_reshape_4d(ctx, B, Cw, OW, OH, 2);
        ggml_tensor* re = ggml_sub(ctx, part(A, 0), part(B, 1));
        ggml_tensor* im = ggml_add(ctx, part(A, 1), part(B, 0));
        return {re, im};
    }

    ggml_tensor* affine_lrelu(ggml_tensor* x, const ComplexConv& c) {
        x = ggml_mul(ctx, x, bcast_tfc(c.scale));
        x = ggml_add(ctx, x, bcast_tfc(c.shift));
        return ggml_leaky_relu(ctx, x, m.hp.leaky, false);
    }

    // Encoder: Conv2d k=(kf,2) stride (2,1) pad (0,1) + BN + LeakyReLU.
    ggml_tensor* encoder(ggml_tensor* x, const ComplexConv& c) {
        const int64_t T = x->ne[0], F = x->ne[1], Cin = x->ne[2];
        ggml_tensor* xp = ggml_pad_ext(ctx, x, 1, 1, 0, 0, 0, 0, 0, 0); // time padding
        const int64_t Tout = T + 1;
        const int64_t Fout = (F - c.kf) / 2 + 1;
        const size_t per_frame = (size_t)Cin * c.kf * 2 * Fout * 2 * sizeof(float) * 2;
        return chunked(Tout, Fout, c.cout, chunk_for(per_frame), [&](int64_t t0, int64_t n) {
            ggml_tensor* xc = frames(xp, t0, n + 1);
            auto [re, im] = complex_mm(xc, c, c.kf, 2, 2, 0); // [Cout, n, Fout]
            ggml_tensor* y = ggml_concat(ctx, ggml_reshape_4d(ctx, re, re->ne[0], re->ne[1], re->ne[2], 1),
                                         ggml_reshape_4d(ctx, im, im->ne[0], im->ne[1], im->ne[2], 1), 3);
            y = ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3)); // [n, Fout, Cout, 2]
            return affine_lrelu(y, c);
        });
    }

    // Decoder: ConvTranspose2d k=(kf,2) stride (2,1) pad (0,1) as a polyphase
    // stride-1 conv, + BN + LeakyReLU.
    ggml_tensor* decoder(ggml_tensor* x, const ComplexConv& c) {
        const int64_t T = x->ne[0], F = x->ne[1], Cin = x->ne[2];
        const int q = (c.kf + 1) / 2;
        const int64_t Tout = T - 1;
        const int64_t M = F + q - 1;
        const int64_t Fout = 2 * (F - 1) + c.kf;
        const size_t per_frame = (size_t)Cin * q * 2 * M * 2 * sizeof(float) * 2;
        return chunked(Tout, Fout, c.cout, chunk_for(per_frame), [&](int64_t t0, int64_t n) {
            ggml_tensor* xc = frames(x, t0, n + 1);
            auto [re, im] = complex_mm(xc, c, q, 2, 1, q - 1); // [2*Cout, n, M]
            auto interleave = [&](ggml_tensor* t) {
                t = ggml_reshape_4d(ctx, t, c.cout, 2, n, M);
                t = ggml_cont(ctx, ggml_permute(ctx, t, 3, 1, 0, 2)); // [n, 2, M, Cout]
                return ggml_reshape_4d(ctx, t, n, 2 * M, c.cout, 1);
            };
            ggml_tensor* y = ggml_concat(ctx, interleave(re), interleave(im), 3); // [n, 2M, Cout, 2]
            if (2 * M != Fout)
                y = ggml_view_4d(ctx, y, n, Fout, c.cout, 2, y->nb[1], y->nb[2], y->nb[3], 0);
            return affine_lrelu(y, c);
        });
    }

    // UniDeepFsmn on xc [C, S, B1, B2] along S: x + p + causal_dwconv_S(p).
    ggml_tensor* fsmn_unit(ggml_tensor* xc, const FsmnUnit& u) {
        ggml_tensor* h = ggml_mul_mat(ctx, u.lin_w, xc);
        h = ggml_relu(ctx, ggml_add(ctx, h, u.lin_b));
        ggml_tensor* p = ggml_mul_mat(ctx, u.proj_w, h); // [C, S, B1, B2]
        const int64_t C = p->ne[0], S = p->ne[1], B = p->ne[2] * p->ne[3];
        const int order = (int)u.mem->ne[0];
        ggml_tensor* pt = ggml_cont(ctx, ggml_permute(ctx, p, 1, 0, 2, 3)); // [S, C, B1, B2]
        pt = ggml_pad_ext(ctx, pt, order - 1, 0, 0, 0, 0, 0, 0, 0);
        pt = ggml_reshape_3d(ctx, pt, S + order - 1, C, B);
        ggml_tensor* conv = ggml_ssm_conv(ctx, pt, u.mem); // [C, S, B]
        conv = ggml_reshape_4d(ctx, conv, C, S, p->ne[2], p->ne[3]);
        return ggml_add(ctx, ggml_add(ctx, xc, p), conv);
    }

    // complex: real = re(xr) - im(xi), imag = re(xi) + im(xr); xc [C, S, B, 2]
    ggml_tensor* complex_fsmn(ggml_tensor* xc, const FsmnUnit& re_u, const FsmnUnit& im_u) {
        ggml_tensor* R = fsmn_unit(xc, re_u);
        ggml_tensor* I = fsmn_unit(xc, im_u);
        ggml_tensor* re = ggml_sub(ctx, part(R, 0), part(I, 1));
        ggml_tensor* im = ggml_add(ctx, part(R, 1), part(I, 0));
        return ggml_concat(ctx, ggml_reshape_4d(ctx, re, re->ne[0], re->ne[1], re->ne[2], 1),
                           ggml_reshape_4d(ctx, im, im->ne[0], im->ne[1], im->ne[2], 1), 3);
    }

    // ComplexUniDeepFsmn_L1 along frequency, independently per frame.
    ggml_tensor* fsmn_freq(ggml_tensor* x, const ComplexFsmn& f) {
        const int64_t T = x->ne[0], F = x->ne[1], C = x->ne[2];
        const size_t per_frame = (size_t)F * C * 2 * sizeof(float) * 8;
        return chunked(T, F, C, chunk_for(per_frame), [&](int64_t t0, int64_t n) {
            ggml_tensor* v = ggml_view_4d(ctx, x, n, F, C, 2, x->nb[1], x->nb[2], x->nb[3], (size_t)t0 * x->nb[0]);
            ggml_tensor* xc = ggml_cont(ctx, ggml_permute(ctx, v, 2, 1, 0, 3)); // [C, F, n, 2]
            ggml_tensor* y = complex_fsmn(xc, f.re, f.im);
            return ggml_cont(ctx, ggml_permute(ctx, y, 2, 1, 0, 3)); // [n, F, C, 2]
        });
    }

    // Bottleneck ComplexUniDeepFsmn (two layers) along time; x [T, 1, C, 2].
    ggml_tensor* fsmn_time(ggml_tensor* x, const Unet& u) {
        const int64_t T = x->ne[0], C = x->ne[2];
        GGML_ASSERT(x->ne[1] == 1);
        ggml_tensor* xc = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); // [C, T, 1, 2]
        ggml_tensor* y = complex_fsmn(xc, u.mid_re1, u.mid_im1);
        y = complex_fsmn(y, u.mid_re2, u.mid_im2);            // [C, T, 1, 2]
        y = ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3)); // [T, 1, C, 2]
        return ggml_reshape_4d(ctx, y, T, 1, C, 2);
    }

    // SELayer: x * [fc_r(xr)-fc_i(xi), fc_r(xi)+fc_i(xr)] (part-wise).
    ggml_tensor* se_layer(ggml_tensor* x, const SeLayer& s) {
        const int64_t T = x->ne[0], F = x->ne[1], C = x->ne[2];
        ggml_tensor* mean = ggml_mean(ctx, ggml_reshape_3d(ctx, x, T * F, C, 2)); // [1, C, 2]
        mean = ggml_reshape_2d(ctx, mean, C, 2);                                  // cols: xr, xi
        auto fc = [&](ggml_tensor* w1, ggml_tensor* b1, ggml_tensor* w2, ggml_tensor* b2) {
            ggml_tensor* h = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w1, mean), b1));
            return ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w2, h), b2)); // [C, 2]
        };
        ggml_tensor* R = fc(s.r1w, s.r1b, s.r2w, s.r2b);
        ggml_tensor* I = fc(s.i1w, s.i1b, s.i2w, s.i2b);
        auto col = [&](ggml_tensor* t, int j) { return ggml_view_2d(ctx, t, C, 1, t->nb[1], (size_t)j * t->nb[1]); };
        ggml_tensor* yr = ggml_sub(ctx, col(R, 0), col(I, 1));
        ggml_tensor* yi = ggml_add(ctx, col(R, 1), col(I, 0));
        ggml_tensor* y = ggml_concat(ctx, yr, yi, 1); // [C, 2]
        return ggml_mul(ctx, x, ggml_reshape_4d(ctx, y, 1, 1, C, 2));
    }

    ggml_tensor* unet(int ui, ggml_tensor* x) {
        const Unet& u = m.unet[ui];
        const std::string tag = ui == 0 ? "u1." : "u2.";
        ggml_tensor* skips[kLevels + 1] = {x};
        ggml_tensor* cur = x;
        for (int i = 0; i < kLevels; i++) {
            if (i > 0) {
                cur = fsmn_freq(cur, u.fsmn_enc[i]);
                stage(tag + "fsmn_enc" + std::to_string(i), cur);
            }
            cur = encoder(cur, u.enc[i]);
            stage(tag + "enc" + std::to_string(i), cur);
            skips[i + 1] = se_layer(cur, u.se_enc[i]);
            stage(tag + "se_enc" + std::to_string(i), skips[i + 1]);
        }
        cur = fsmn_time(cur, u);
        stage(tag + "fsmn", cur);
        for (int i = 0; i < kLevels; i++) {
            cur = decoder(cur, u.dec[i]);
            stage(tag + "dec" + std::to_string(i), cur);
            if (i == kLevels - 1)
                break;
            cur = fsmn_freq(cur, u.fsmn_dec[i]);
            stage(tag + "fsmn_dec" + std::to_string(i), cur);
            if (i < kLevels - 2) {
                cur = se_layer(cur, u.se_dec[i]);
                stage(tag + "se_dec" + std::to_string(i), cur);
            }
            cur = ggml_concat(ctx, cur, skips[kLevels - 1 - i], 2);
        }
        // Final ComplexConv2d(1, 1, 1): scalar complex affine.
        ggml_tensor* xr = part(cur, 0);
        ggml_tensor* xi = part(cur, 1);
        const float wr = u.lin[0], br = u.lin[1], wi = u.lin[2], bi = u.lin[3];
        ggml_tensor* re = ggml_sub(ctx, ggml_scale_bias(ctx, xr, wr, br - bi), ggml_scale(ctx, xi, wi));
        ggml_tensor* im = ggml_add(ctx, ggml_scale_bias(ctx, xi, wr, br + bi), ggml_scale(ctx, xr, wi));
        ggml_tensor* out = ggml_concat(ctx, ggml_reshape_4d(ctx, re, re->ne[0], re->ne[1], re->ne[2], 1),
                                       ggml_reshape_4d(ctx, im, im->ne[0], im->ne[1], im->ne[2], 1), 3);
        stage(tag + "out", out);
        return out;
    }
};

bool build_graph(frcrn_se_context& m, int T, bool dump) {
    if (m.graph && m.graph_frames == T && m.graph_dump == dump)
        return true;
    if (m.gctx) {
        ggml_free(m.gctx);
        m.gctx = nullptr;
    }
    m.graph = nullptr;
    m.stages.clear();
    // Chunking makes the node count grow with T; the context is only
    // reserved virtual memory until nodes are created.
    const size_t max_nodes = 16384 + (size_t)64 * T;
    ggml_init_params ip = {ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false), nullptr,
                           true};
    m.gctx = ggml_init(ip);
    if (!m.gctx)
        return false;
    ggml_context* ctx = m.gctx;
    Builder b{m, ctx, dump};
    const int D = m.hp.fft_len / 2 + 1;

    m.in_frames = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.hp.win_len, T);
    ggml_set_name(m.in_frames, "frames");
    ggml_set_input(m.in_frames);

    ggml_tensor* spec = ggml_mul_mat(ctx, m.stft_fwd, m.in_frames); // [2D, T]
    b.stage("stft", spec);
    ggml_tensor* x0 = ggml_cont(ctx, ggml_transpose(ctx, spec)); // [T, 2D]
    x0 = ggml_reshape_4d(ctx, x0, T, D, 1, 2);

    ggml_tensor* u1 = b.unet(0, x0);
    ggml_tensor* u2 = b.unet(1, u1);
    ggml_tensor* mask = ggml_add(ctx, ggml_tanh(ctx, u2), ggml_tanh(ctx, u1));
    b.stage("mask", mask);

    ggml_tensor* sr = b.part(x0, 0);
    ggml_tensor* si = b.part(x0, 1);
    ggml_tensor* mr = b.part(mask, 0);
    ggml_tensor* mi = b.part(mask, 1);
    ggml_tensor* er = ggml_sub(ctx, ggml_mul(ctx, sr, mr), ggml_mul(ctx, si, mi)); // [T, D, 1]
    ggml_tensor* ei = ggml_add(ctx, ggml_mul(ctx, sr, mi), ggml_mul(ctx, si, mr));
    ggml_tensor* est = ggml_concat(ctx, er, ei, 1); // [T, 2D, 1]
    est = ggml_reshape_2d(ctx, est, T, 2 * D);
    ggml_tensor* estT = ggml_cont(ctx, ggml_transpose(ctx, est)); // [2D, T]
    b.stage("est_spec", estT);
    m.out_frames = ggml_mul_mat(ctx, m.stft_inv, estT); // [win, T]
    ggml_set_name(m.out_frames, "out_frames");
    ggml_set_output(m.out_frames);

    m.graph = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(m.graph, m.out_frames);
    for (auto& s : m.stages)
        ggml_build_forward_expand(m.graph, s.second);

    if (m.sched && m.sched_capacity < max_nodes) {
        ggml_backend_sched_free(m.sched);
        m.sched = nullptr;
    }
    if (!m.sched) {
        m.sched_capacity = max_nodes;
        ggml_backend_t backends[2] = {m.backend, m.cpu};
        const int nb = m.backend == m.cpu ? 1 : 2;
        m.sched = ggml_backend_sched_new(backends, nullptr, nb, max_nodes, false, true);
        if (!m.sched)
            return false;
    } else {
        ggml_backend_sched_reset(m.sched);
    }
    if (!ggml_backend_sched_alloc_graph(m.sched, m.graph)) {
        std::fprintf(stderr, "frcrn_se: graph allocation failed (T=%d frames)\n", T);
        return false;
    }
    m.graph_frames = T;
    m.graph_dump = dump;
    if (m.params.verbose) {
        size_t total = 0;
        ggml_backend_t bs[2] = {m.backend, m.cpu};
        for (int i = 0; i < (m.backend == m.cpu ? 1 : 2); i++)
            total += ggml_backend_sched_get_buffer_size(m.sched, bs[i]);
        std::fprintf(stderr, "frcrn_se: graph T=%d frames, %d nodes, compute buffer %.1f MiB\n", T,
                     ggml_graph_n_nodes(m.graph), total / 1048576.0);
    }
    return true;
}

// One `model.inference(seg)` call: ConvSTFT -> UNets -> mask -> ConviSTFT.
// Returns (frames-1)*inc + win samples like upstream's conv_transpose1d.
bool infer_segment(frcrn_se_context& m, const float* x, int64_t L, std::vector<float>& y) {
    const int win = m.hp.win_len, inc = m.hp.win_inc;
    if (L < win)
        return false;
    const int T = (int)((L - win) / inc + 1);
    const bool dump = !m.dump_dir.empty() && !m.dumped;
    if (!build_graph(m, T, dump))
        return false;
    std::vector<float> frames((size_t)win * T);
    for (int t = 0; t < T; t++)
        std::memcpy(&frames[(size_t)t * win], x + (size_t)t * inc, sizeof(float) * win);
    // build_graph may reuse a cached allocation; re-alloc is a no-op then.
    ggml_backend_sched_reset(m.sched);
    if (!ggml_backend_sched_alloc_graph(m.sched, m.graph))
        return false;
    ggml_backend_tensor_set(m.in_frames, frames.data(), 0, frames.size() * sizeof(float));
    OpProfile prof;
    const bool profile = env_true("CRISPASR_FRCRN_SE_PROFILE");
    if (profile) {
        prof.last = std::chrono::steady_clock::now();
        ggml_backend_sched_set_eval_callback(m.sched, profile_cb, &prof);
    }
    const bool computed = ggml_backend_sched_graph_compute(m.sched, m.graph) == GGML_STATUS_SUCCESS;
    if (profile) {
        ggml_backend_sched_set_eval_callback(m.sched, nullptr, nullptr);
        std::vector<std::pair<double, std::string>> rows;
        for (auto& kv : prof.by_op)
            rows.emplace_back(kv.second.first, kv.first + " x" + std::to_string(kv.second.second));
        std::sort(rows.rbegin(), rows.rend());
        for (auto& r : rows)
            std::fprintf(stderr, "frcrn_se: profile %8.3f s  %s\n", r.first, r.second.c_str());
    }
    if (!computed)
        return false;
    std::vector<float> of((size_t)win * T);
    ggml_backend_tensor_get(m.out_frames, of.data(), 0, of.size() * sizeof(float));

    // Overlap-add + window-power normalisation (ConviSTFT.forward).
    const int64_t n_out = (int64_t)(T - 1) * inc + win;
    std::vector<float> acc(n_out, 0.0f), coff(n_out, 0.0f);
    for (int t = 0; t < T; t++) {
        const float* f = &of[(size_t)t * win];
        float* a = &acc[(size_t)t * inc];
        float* c = &coff[(size_t)t * inc];
        for (int k = 0; k < win; k++) {
            a[k] += f[k];
            c[k] += m.window[k] * m.window[k];
        }
    }
    y.resize(n_out);
    for (int64_t i = 0; i < n_out; i++)
        y[i] = acc[i] / (coff[i] + 1e-8f);

    if (dump) {
        m.dumped = true;
        for (auto& s : m.stages) {
            ggml_tensor* t = s.second;
            std::vector<float> d(ggml_nelements(t));
            if (ggml_is_contiguous(t)) {
                ggml_backend_tensor_get(t, d.data(), 0, d.size() * sizeof(float));
            } else {
                continue;
            }
            write_npy(m.dump_dir + "/" + s.first + ".npy", d, t->ne, ggml_n_dims(t) < 2 ? 2 : ggml_n_dims(t));
        }
        int64_t ne_in[1] = {L};
        write_npy(m.dump_dir + "/input.npy", std::vector<float>(x, x + L), ne_in, 1);
        int64_t ne_w[1] = {n_out};
        write_npy(m.dump_dir + "/wav.npy", y, ne_w, 1);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

std::vector<int> kv_i32_array(gguf_context* g, const char* key) {
    std::vector<int> v;
    const int64_t k = gguf_find_key(g, key);
    if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_ARRAY)
        return v;
    const enum gguf_type at = gguf_get_arr_type(g, k);
    const size_t n = gguf_get_arr_n(g, k);
    const void* data = gguf_get_arr_data(g, k);
    for (size_t i = 0; i < n; i++) {
        if (at == GGUF_TYPE_INT32)
            v.push_back(((const int32_t*)data)[i]);
        else if (at == GGUF_TYPE_UINT32)
            v.push_back((int)((const uint32_t*)data)[i]);
        else if (at == GGUF_TYPE_INT64)
            v.push_back((int)((const int64_t*)data)[i]);
    }
    return v;
}

bool bind_weights(frcrn_se_context& m) {
    const auto& T = m.weights.tensors;
    bool ok = true;
    auto req = [&](const std::string& name) {
        ggml_tensor* t = core_gguf::require(T, name.c_str(), "frcrn_se");
        if (!t)
            ok = false;
        return t;
    };
    m.stft_fwd = req("stft.fwd");
    m.stft_inv = req("stft.inv");
    ggml_tensor* win = req("stft.window");
    if (!ok)
        return false;
    m.window.resize(ggml_nelements(win));
    ggml_backend_tensor_get(win, m.window.data(), 0, m.window.size() * sizeof(float));
    if ((int)m.window.size() != m.hp.win_len || m.stft_fwd->ne[0] != m.hp.win_len ||
        m.stft_fwd->ne[1] != m.hp.fft_len + 2 || m.stft_inv->ne[0] != m.hp.fft_len + 2)
        return false;

    auto fsmn = [&](const std::string& p) {
        FsmnUnit u;
        u.lin_w = req(p + ".lin.w");
        u.lin_b = req(p + ".lin.b");
        u.proj_w = req(p + ".proj.w");
        u.mem = req(p + ".mem");
        return u;
    };
    auto se = [&](const std::string& p) {
        SeLayer s;
        s.r1w = req(p + ".r1.w");
        s.r1b = req(p + ".r1.b");
        s.r2w = req(p + ".r2.w");
        s.r2b = req(p + ".r2.b");
        s.i1w = req(p + ".i1.w");
        s.i1b = req(p + ".i1.b");
        s.i2w = req(p + ".i2.w");
        s.i2b = req(p + ".i2.b");
        return s;
    };
    auto conv = [&](const std::string& p, int kf, int cin, int cout) {
        ComplexConv c;
        c.w_re = req(p + ".w_re");
        c.w_im = req(p + ".w_im");
        c.scale = req(p + ".scale");
        c.shift = req(p + ".shift");
        c.kf = kf;
        c.cin = cin;
        c.cout = cout;
        return c;
    };
    const int C = m.hp.channels;
    for (int ui = 0; ui < 2; ui++) {
        Unet& u = m.unet[ui];
        const std::string tag = ui == 0 ? "u1" : "u2";
        for (int i = 0; i < kLevels; i++) {
            const std::string s = std::to_string(i);
            u.enc[i] = conv(tag + ".enc." + s, m.hp.enc_kf[i], i == 0 ? 1 : C, C);
            u.se_enc[i] = se(tag + ".se_enc." + s);
            if (i > 0)
                u.fsmn_enc[i] = {fsmn(tag + ".fsmn_enc." + s + ".re"), fsmn(tag + ".fsmn_enc." + s + ".im")};
            const int dec_cout = i == kLevels - 1 ? 1 : C;
            u.dec[i] = conv(tag + ".dec." + s, m.hp.dec_kf[i], i == 0 ? C : 2 * C, dec_cout);
            if (i < kLevels - 1)
                u.fsmn_dec[i] = {fsmn(tag + ".fsmn_dec." + s + ".re"), fsmn(tag + ".fsmn_dec." + s + ".im")};
            if (i < kLevels - 2)
                u.se_dec[i] = se(tag + ".se_dec." + s);
        }
        u.mid_re1 = fsmn(tag + ".fsmn.rel1");
        u.mid_im1 = fsmn(tag + ".fsmn.iml1");
        u.mid_re2 = fsmn(tag + ".fsmn.rel2");
        u.mid_im2 = fsmn(tag + ".fsmn.iml2");
        ggml_tensor* lin = req(tag + ".out.w");
        if (!ok)
            return false;
        if (ggml_nelements(lin) != 4 || lin->type != GGML_TYPE_F32)
            return false;
        ggml_backend_tensor_get(lin, u.lin, 0, sizeof(u.lin));
        // Shape checks for the conv weights (flattened [K, Cout]).
        for (int i = 0; i < kLevels; i++) {
            const ComplexConv& e = u.enc[i];
            if (e.w_re->ne[0] != (int64_t)e.cin * e.kf * 2 || e.w_re->ne[1] != e.cout)
                return false;
            const ComplexConv& d = u.dec[i];
            if (d.w_re->ne[0] != (int64_t)d.cin * ((d.kf + 1) / 2) * 2 || d.w_re->ne[1] != 2 * d.cout)
                return false;
        }
    }
    return ok;
}

} // namespace

frcrn_se_context* frcrn_se_init_from_file(const char* path, frcrn_se_params params) {
    if (!path || !*path)
        return nullptr;
    auto m = std::make_unique<frcrn_se_context>();
    m->params = params;
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta)
        return nullptr;
    if (core_gguf::kv_str(meta, "general.architecture", "") != "frcrn-se") {
        std::fprintf(stderr, "frcrn_se: '%s' is not an frcrn-se GGUF\n", path);
        core_gguf::free_metadata(meta);
        return nullptr;
    }
    Hparams& hp = m->hp;
    hp.sample_rate = (int)core_gguf::kv_u32(meta, "frcrn_se.sample_rate", 16000);
    hp.win_len = (int)core_gguf::kv_u32(meta, "frcrn_se.win_len", 640);
    hp.win_inc = (int)core_gguf::kv_u32(meta, "frcrn_se.win_inc", 320);
    hp.fft_len = (int)core_gguf::kv_u32(meta, "frcrn_se.fft_len", 640);
    hp.channels = (int)core_gguf::kv_u32(meta, "frcrn_se.channels", 128);
    hp.fsmn_order = (int)core_gguf::kv_u32(meta, "frcrn_se.fsmn_order", 20);
    hp.leaky = core_gguf::kv_f32(meta, "frcrn_se.leaky_slope", 0.01f);
    hp.decode_window_s = core_gguf::kv_f32(meta, "frcrn_se.decode_window_s", 1.0f);
    hp.decode_stride_ratio = core_gguf::kv_f32(meta, "frcrn_se.decode_stride_ratio", 0.75f);
    hp.one_time_s = core_gguf::kv_f32(meta, "frcrn_se.one_time_decode_length_s", 120.0f);
    hp.norm_db = core_gguf::kv_f32(meta, "frcrn_se.norm_target_db", -25.0f);
    hp.enc_kf = kv_i32_array(meta, "frcrn_se.enc_kernel_f");
    hp.dec_kf = kv_i32_array(meta, "frcrn_se.dec_kernel_f");
    const int n_levels = (int)core_gguf::kv_u32(meta, "frcrn_se.n_levels", 0);
    core_gguf::free_metadata(meta);
    if (n_levels != kLevels || (int)hp.enc_kf.size() != kLevels || (int)hp.dec_kf.size() != kLevels ||
        hp.win_len != 2 * hp.win_inc || hp.sample_rate <= 0) {
        std::fprintf(stderr, "frcrn_se: unsupported or incomplete metadata in '%s'\n", path);
        return nullptr;
    }

    ggml_backend_load_all();
    m->cpu = core_cpu_backend::init();
    if (!m->cpu)
        return nullptr;
    const int n_threads = params.n_threads > 0 ? params.n_threads : 4;
    core_cpu_backend::set_n_threads(m->cpu, n_threads);
    if (params.use_gpu || env_true("CRISPASR_FRCRN_SE_GPU")) {
        m->backend = crispasr_init_gpu_backend();
        if (m->backend && core_cpu_backend::is_cpu(m->backend)) {
            ggml_backend_free(m->backend);
            m->backend = nullptr;
        }
    }
    if (!m->backend)
        m->backend = m->cpu;
    bool loaded = m->backend == m->cpu ? core_gguf::load_weights(path, m->cpu, "frcrn_se", m->weights)
                                       : core_gguf::load_weights_fit(path, m->backend, m->cpu, "frcrn_se", m->weights);
    if (!loaded)
        return nullptr;
    if (!bind_weights(*m)) {
        std::fprintf(stderr, "frcrn_se: missing or mis-shaped tensors in '%s'\n", path);
        return nullptr;
    }
    if (const char* mb = std::getenv("CRISPASR_FRCRN_SE_CHUNK_MB")) {
        const long v = std::atol(mb);
        if (v > 0)
            m->chunk_bytes = (size_t)v << 20;
    }
    // Upstream decodes inputs up to 120 s in one pass (its SE layers pool over
    // the whole clip) and longer ones in 1 s windows. The one-pass graph needs
    // roughly 1 MB of compute buffer per frame (20 ms); lower this to trade
    // fidelity to upstream for memory.
    if (const char* v = std::getenv("CRISPASR_FRCRN_SE_ONE_PASS_S")) {
        const double s = std::atof(v);
        if (s > 0)
            m->hp.one_time_s = (float)s;
    }
    if (const char* d = std::getenv("CRISPASR_FRCRN_SE_DUMP_DIR"))
        m->dump_dir = d;
    if (params.verbose)
        std::fprintf(stderr, "frcrn_se: loaded '%s' (backend=%s, threads=%d)\n", path, ggml_backend_name(m->backend),
                     n_threads);
    return m.release();
}

int frcrn_se_sample_rate(const frcrn_se_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 0;
}

void frcrn_se_free(frcrn_se_context* ctx) {
    delete ctx;
}

bool frcrn_se_enhance(frcrn_se_context* ctx, const float* in, int n, std::vector<float>& out) {
    if (!ctx || !in || n <= 0)
        return false;
    frcrn_se_context& m = *ctx;
    const auto t_start = std::chrono::steady_clock::now();

    // DataReader audio_norm (float64), then float32 like upstream's astype.
    const double EPS = 1e-6;
    const double target = std::pow(10.0, m.hp.norm_db / 20.0);
    double ss = 0.0;
    for (int i = 0; i < n; i++)
        ss += (double)in[i] * in[i];
    const double rms = std::sqrt(ss / n);
    const double scalar = target / (rms + EPS);
    double avg_pow = 0.0;
    for (int i = 0; i < n; i++) {
        const double v = in[i] * scalar;
        avg_pow += v * v;
    }
    avg_pow /= n;
    double hi_sum = 0.0;
    int64_t hi_n = 0;
    for (int i = 0; i < n; i++) {
        const double v = in[i] * scalar;
        if (v * v > avg_pow) {
            hi_sum += v * v;
            hi_n++;
        }
    }
    // An all-equal-power input (e.g. digital silence) has no sample above
    // the mean; numpy would yield NaN there. Treat it as rmsx = 0.
    const double rmsx = hi_n > 0 ? std::sqrt(hi_sum / hi_n) : 0.0;
    const double scalarx = target / (rmsx + EPS);
    const double denorm = 1.0 / (scalar * scalarx + EPS);

    // decode_one_audio_frcrn_se_16k padding (kept bug-for-bug).
    const int64_t window = (int64_t)(m.hp.sample_rate * m.hp.decode_window_s);
    const int64_t stride = (int64_t)(window * m.hp.decode_stride_ratio);
    int64_t t = n;
    const bool segment =
        (double)t > m.hp.sample_rate * (double)m.hp.one_time_s || env_true("CRISPASR_FRCRN_SE_FORCE_SEGMENT");
    int64_t padded = t;
    if (t < window)
        padded = window;
    else if (t < window + stride)
        padded = window + stride;
    else if ((t - window) % stride != 0)
        padded = t + (t - (t - window) / stride * stride);
    std::vector<float> x(padded, 0.0f);
    for (int i = 0; i < n; i++)
        x[i] = (float)(in[i] * scalar * scalarx);

    std::vector<float> res(padded, 0.0f);
    std::vector<float> seg;
    if (segment) {
        const int64_t give_up = (window - stride) / 2;
        for (int64_t idx = 0; idx + window <= padded; idx += stride) {
            if (!infer_segment(m, x.data() + idx, window, seg))
                return false;
            // upstream slices tmp_output[:-give_up] / [give_up:-give_up];
            // tmp_output has `window` samples when (window-win) % inc == 0.
            const int64_t len = (int64_t)seg.size();
            if (idx == 0) {
                for (int64_t i = 0; i < window - give_up && i < len - give_up; i++)
                    res[i] = seg[i];
            } else {
                for (int64_t i = give_up; i < window - give_up && i < len - give_up; i++)
                    res[idx + i] = seg[i];
            }
        }
    } else {
        if (!infer_segment(m, x.data(), padded, seg))
            return false;
        for (int64_t i = 0; i < padded && i < (int64_t)seg.size(); i++)
            res[i] = seg[i];
    }
    out.resize(n);
    for (int i = 0; i < n; i++)
        out[i] = (float)(res[i] * denorm);
    if (m.params.verbose) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        std::fprintf(stderr, "frcrn_se: enhanced %.2f s of audio in %.2f s (%s)\n", n / (double)m.hp.sample_rate, s,
                     segment ? "segmented" : "one pass");
    }
    return true;
}

// ---------------------------------------------------------------------------
// Enhancer registration
// ---------------------------------------------------------------------------

namespace {
void* ops_init(const char* path, int n_threads, bool verbose) {
    frcrn_se_params p;
    p.n_threads = n_threads;
    p.verbose = verbose;
    return frcrn_se_init_from_file(path, p);
}
int ops_rate(void* ctx) {
    return frcrn_se_sample_rate((frcrn_se_context*)ctx);
}
bool ops_process(void* ctx, const float* in, int n, std::vector<float>& out) {
    return frcrn_se_enhance((frcrn_se_context*)ctx, in, n, out);
}
void ops_free(void* ctx) {
    frcrn_se_free((frcrn_se_context*)ctx);
}
} // namespace

const CrispasrEnhanceModelOps* frcrn_se_enhance_ops() {
    static const CrispasrEnhanceModelOps ops = {"frcrn-se", "frcrn-se", ops_init, ops_rate, ops_process, ops_free};
    return &ops;
}
