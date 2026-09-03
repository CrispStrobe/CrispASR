// beat_this.cpp — Beat This! beat/downbeat tracking runtime (§251b).
//
// Status: front end + weight loading. The ggml graph is the remaining piece;
// see docs/music-transcription/PLAN.md §251b-1 for the traced blueprint.
//
// The front end is implemented and validated FIRST, deliberately: it is the
// part most likely to drift silently (a wrong window, normalization or mel
// layout produces a plausible spectrogram and wrong beats), and it can be
// checked against torchaudio without any of the network existing yet.

#include "beat_this.h"

#include "core/quant_bcast.h"
#include "core/crispasr_env.h"
#include "core/fft.h"
#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <array>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include "core/ggml_cpu_backend.h"

// One roformer Attention sub-block (§251b-1). NOTE the bias asymmetry, which
// is not a transcription slip: to_qkv and to_out are bias-free, to_gates has a
// bias. `n_head` is dim/32 — 1, 2, 4 for frontend blocks 0/1/2.
struct bt_attn {
    ggml_tensor* gamma = nullptr;   // norm.gamma — RMSNorm scale, no bias
    ggml_tensor* qkv_w = nullptr;   // (dim, 3*n_head*head_dim)
    ggml_tensor* gates_w = nullptr; // (dim, n_head)  — ONE scalar per head
    ggml_tensor* gates_b = nullptr; // (n_head)
    ggml_tensor* out_w = nullptr;   // (n_head*head_dim, dim)
    int n_head = 1;
    int head_dim = 32;
};

// One roformer FeedForward: RMSNorm -> Linear -> GELU(erf) -> Linear.
struct bt_ff {
    ggml_tensor* gamma = nullptr; // net.0.gamma
    ggml_tensor* w1 = nullptr;    // net.1 (dim, 4*dim)
    ggml_tensor* b1 = nullptr;
    ggml_tensor* w2 = nullptr; // net.4 (4*dim, dim)
    ggml_tensor* b2 = nullptr;
};

struct bt_partial {
    bt_attn attnF;
    bt_ff ffF;
    bt_attn attnT;
    bt_ff ffT;
};

// A frontend block: PartialFTTransformer, then conv2d(k=(2,3), s=(2,1),
// p=(0,1)) + BN + GELU. The BN is already folded into the conv by the
// converter, which is why a bias exists here where torch has bias=False.
struct bt_block {
    bt_partial partial;
    ggml_tensor* conv_w = nullptr;
    ggml_tensor* conv_b = nullptr;
};

struct beat_this_context {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad wl;
    int n_threads = 4;

    int sample_rate = BEAT_THIS_SAMPLE_RATE;
    int n_fft = BEAT_THIS_N_FFT;
    int hop = BEAT_THIS_HOP;
    int mel_bins = BEAT_THIS_MEL_BINS;
    float log_multiplier = 1000.0f;

    // Baked [513, 128] row-major mel filterbank from the reference export.
    // Never re-derived: slaney-vs-htk and the freq/mel layout are classic
    // silent-drift sources, and the export ships the exact matrix the model
    // was trained and exported with.
    std::vector<float> mel_fb;
    int fb_freqs = 0;

    std::vector<float> hann; // periodic, length n_fft
    bool debug = false;

    // --- graph weights (bound in beat_this_init) -------------------------
    ggml_tensor* bn1d_scale = nullptr; // (128) per-FREQUENCY, pre-conv
    ggml_tensor* bn1d_offset = nullptr;
    ggml_tensor* stem_w = nullptr; // ne (kt=3, kf=4, ic=1, oc=32)
    ggml_tensor* stem_b = nullptr; // (32)

    bt_block blocks[3];

    ggml_tensor* linear_w = nullptr; // frontend.linear, 1024 -> 512
    ggml_tensor* linear_b = nullptr;

    // The 6 main roformer layers: dim 512, 16 heads, same Attention/FeedForward
    // as the frontend's partial transformers with only the dims changed. The
    // checkpoint numbers them positionally, .0 = attention and .1 = ff.
    struct bt_layer {
        bt_attn attn;
        bt_ff ff;
    } layers[BEAT_THIS_N_LAYERS];
    ggml_tensor* out_norm = nullptr; // transformer_blocks.norm.gamma
    ggml_tensor* head_w = nullptr;   // task_heads.beat_downbeat_lin, 512 -> 2
    ggml_tensor* head_b = nullptr;
};

namespace {

// Periodic Hann (torch.hann_window default), NOT the symmetric np.hanning
// variant — they differ by ~2.4e-7/sample, which is enough to move a cos_min
// into the 0.95 band (see the mel pitfalls note in docs/contributing.md).
void build_hann_periodic(int n, std::vector<float>& w) {
    w.resize((size_t)n);
    const double two_pi = 6.283185307179586476925286766559;
    for (int i = 0; i < n; i++)
        w[(size_t)i] = (float)(0.5 - 0.5 * std::cos(two_pi * (double)i / (double)n));
}

// Reflect-pad index, matching torch's pad_mode="reflect" (no edge repeat).
inline int reflect_index(int i, int n) {
    if (n <= 1)
        return 0;
    while (i < 0 || i >= n) {
        if (i < 0)
            i = -i;
        if (i >= n)
            i = 2 * (n - 1) - i;
    }
    return i;
}

// --- graph pieces -------------------------------------------------------
//
// LAYOUT CONVENTION for everything below. The reference works in torch
// (b, c, f, t); ggml reverses that, so the block-level tensor here is
// ne = (t, f, c). Inside a partial transformer the sequence axis moves to
// ne[1] and the *other* spatial axis becomes the batch, folded into ne[2]
// exactly as einops folds it into the batch dim upstream:
//
//   frequency phase: ne = (c, f, t)   -> 't' independent sequences of length f
//   time phase:      ne = (c, t, f)   -> 'f' independent sequences of length t
//
// RMSNorm eps. Upstream is `F.normalize(x, dim=-1) * sqrt(size) * gamma`,
// which divides by the L2 norm with an eps of 1e-12 *outside* the sqrt. Since
// RMS = L2/sqrt(size), that is standard RMSNorm x gamma; 1e-12 is used here so
// the guard term stays far below the f16 weight noise floor rather than
// contributing its own drift.
ggml_tensor* bt_norm(ggml_context* c, ggml_tensor* x, ggml_tensor* gamma) {
    return ggml_mul(c, ggml_rms_norm(c, x, 1e-12f), gamma);
}

// #416: these matmuls pair a quantized weight (ne2=1) with an activation
// whose ne2 is the folded time axis (see the einops note above), so ggml
// broadcasts src0 across it. Gated OFF — see src/core/quant_bcast.h.
static inline ggml_tensor* MM(ggml_context* c, ggml_tensor* w, ggml_tensor* x) {
    // Default ON: verified on a locally-quantized q8_0 build — the detector
    // reports 29 broadcasting quantized matmuls without the fold and 0 with it,
    // and the emitted beats are byte-identical either way. Set
    // CRISPASR_BEATTHIS_FOLD_BCAST=0 to restore the legacy path.
    // Read per call, NOT cached in a function-local static. A cached static is
    // immutable for the process, so any in-process A/B of this gate silently
    // compares one path against ITSELF and passes by construction — the exact
    // bug found in the sidon RPE test, where parse_rpe_mode() ran once at init
    // while the test flipped the env per iteration. This is a getenv on a
    // graph-build path, not a hot loop, so the cost is not worth the hazard.
    const bool fold = core_quant_bcast::fold_enabled("CRISPASR_BEATTHIS_FOLD_BCAST", true);
    return fold ? core_quant_bcast::mul_mat_fold_batch(c, w, x) : ggml_mul_mat(c, w, x);
}

// Attention branch. `x` is ne (C, N, B); returns the BRANCH, ne (C, N, B) —
// the caller adds the residual. `pos` is an I32 vector 0..N-1.
ggml_tensor* bt_attention(ggml_context* c, const bt_attn& a, ggml_tensor* x, ggml_tensor* pos) {
    const int64_t N = x->ne[1], B = x->ne[2];
    const int64_t H = a.n_head, D = a.head_dim;

    // The gates read the NORMED x, not the raw input: upstream rebinds
    // `x = self.norm(x)` before computing both qkv and gates.
    ggml_tensor* xn = bt_norm(c, x, a.gamma);

    ggml_tensor* qkv = MM(c, a.qkv_w, xn); // (3*H*D, N, B)
    const size_t esz = ggml_element_size(qkv);

    // to_qkv's output is packed "(qkv h d)" with d fastest, so q/k/v are three
    // contiguous H*D slabs and each slab is already (d, h) — no interleaving.
    auto slab = [&](int64_t which) {
        return ggml_cont(c, ggml_view_4d(c, qkv, D, H, N, B, D * esz, qkv->nb[1], qkv->nb[2], which * H * D * esz));
    };

    // RoPE wants (head_dim, heads, seq, batch) and mode NORMAL rotates
    // ADJACENT PAIRS — which is exactly rotary-embedding-torch's convention
    // (rotate_half regroups '... (d r) -> ... d r' with r=2, and freqs are
    // repeated pairwise). GGML_ROPE_TYPE_NEOX would rotate the split halves
    // instead and is silently wrong here. The checkpoint's rotary_embed.freqs
    // were verified bit-identical to the analytic 10000^(-2i/32) schedule, so
    // ggml's internal theta table needs no override.
    ggml_tensor* q = ggml_rope(c, slab(0), pos, (int)D, GGML_ROPE_TYPE_NORMAL);
    ggml_tensor* k = ggml_rope(c, slab(1), pos, (int)D, GGML_ROPE_TYPE_NORMAL);
    ggml_tensor* v = slab(2);

    // (D, H, N, B) -> (D, N, H, B), the layout flash_attn_ext wants.
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
    v = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));

    // FLASH ATTENTION, NOT AN EXPLICIT mul_mat + softmax. Upstream's Attend()
    // is F.scaled_dot_product_attention, which never materialises the N x N
    // score matrix — and here N is the CHUNK LENGTH, 1500 frames. Materialised,
    // the attnT scores are (1500, 1500, heads, freqs) = 288 MB for ONE
    // sub-block and the graph's compute buffer measured 322 MB; with flash
    // attention the same graph measures 47.6 MB. That 6.8x matters for the
    // downstream consumer, which is a mobile app.
    //
    // Attend() passes no `scale`, so SDPA's default 1/sqrt(head_dim) applies;
    // Attention.scale is computed in __init__ and never used.
    ggml_tensor* out = ggml_flash_attn_ext(c, q, k, v, /*mask*/ nullptr, 1.0f / std::sqrt((float)D),
                                           /*max_bias*/ 0.0f, /*logit_softcap*/ 0.0f);
    // Accumulate in F32: the default permits F16 accumulation, which is a
    // silent precision loss over a 1500-key softmax.
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    // out ne = (D, H, N, B) — flash_attn_ext already emits the permuted layout
    // that "b h n d -> b n (h d)" wants, so no permute+cont is needed below.

    // PER-HEAD SIGMOID GATING. to_gates is Linear(dim -> heads): one scalar
    // per head, broadcast over that head's head_dim and over the sequence, and
    // applied BEFORE to_out. Dropping it leaves output that is plausible and
    // wrong — it scales rather than reshapes the activation.
    ggml_tensor* g = ggml_add(c, MM(c, a.gates_w, xn), a.gates_b); // (H, N, B)
    g = ggml_sigmoid(c, g);
    out = ggml_mul(c, out, ggml_reshape_4d(c, g, 1, H, N, B)); // broadcasts over head_dim

    // "b h n d -> b n (h d)": D is fastest and H next, so the flattened row is
    // already (h d) per token and this is a pure reshape.
    out = ggml_reshape_3d(c, out, D * H, N, B);
    return MM(c, a.out_w, out);
}

// FeedForward branch. GELU is the EXACT erf form: torch nn.GELU defaults to
// approximate='none'. ggml_gelu is the tanh approximation and drifts.
ggml_tensor* bt_feedforward(ggml_context* c, const bt_ff& f, ggml_tensor* x) {
    ggml_tensor* h = bt_norm(c, x, f.gamma);
    h = ggml_add(c, MM(c, f.w1, h), f.b1);
    h = ggml_gelu_erf(c, h);
    return ggml_add(c, MM(c, f.w2, h), f.b2);
}

// Named intermediates, so one graph can serve every parity stage without the
// stage list and the forward drifting apart.
struct bt_stages {
    std::vector<std::pair<std::string, ggml_tensor*>> v;
    void add(const std::string& n, ggml_tensor* t) { v.emplace_back(n, t); }
    ggml_tensor* get(const std::string& n) const {
        for (const auto& p : v)
            if (p.first == n)
                return p.second;
        return nullptr;
    }
};

// PartialFTTransformer. `x` is ne (t, f, c); returns the same shape.
ggml_tensor* bt_partial_ft(ggml_context* c, const bt_partial& p, ggml_tensor* x, ggml_tensor* pos_f, ggml_tensor* pos_t,
                           int blk, bt_stages& sm) {
    const std::string pre = "blk" + std::to_string(blk) + "_";

    // (t, f, c) -> (c, f, t): FREQUENCY is the sequence axis, time the batch.
    ggml_tensor* h = ggml_cont(c, ggml_permute(c, x, 2, 1, 0, 3));

    ggml_tensor* br = bt_attention(c, p.attnF, h, pos_f);
    sm.add(pre + "attnF", br); // the BRANCH, matching the reference hook
    h = ggml_add(c, h, br);
    br = bt_feedforward(c, p.ffF, h);
    sm.add(pre + "ffF", br);
    h = ggml_add(c, h, br);

    // (c, f, t) -> (c, t, f): TIME is now the sequence axis.
    h = ggml_cont(c, ggml_permute(c, h, 0, 2, 1, 3));

    br = bt_attention(c, p.attnT, h, pos_t);
    sm.add(pre + "attnT", br);
    h = ggml_add(c, h, br);
    br = bt_feedforward(c, p.ffT, h);
    sm.add(pre + "ffT", br);
    h = ggml_add(c, h, br);

    // (c, t, f) -> (t, f, c)
    h = ggml_cont(c, ggml_permute(c, h, 2, 0, 1, 3));
    sm.add(pre + "partial", h);
    return h;
}

// Bind one attention sub-block's weights, e.g.
// prefix = "frontend.blocks.0.partial.attnF".
bool bt_bind_attn(core_gguf::tensor_map& tm, const std::string& prefix, int dim, bt_attn& a) {
    a.head_dim = 32; // head_dim is fixed at 32 model-wide (BeatThis(head_dim=32))
    a.n_head = dim / a.head_dim;
    a.gamma = core_gguf::require(tm, (prefix + ".norm.gamma").c_str(), "beat-this");
    a.qkv_w = core_gguf::require(tm, (prefix + ".to_qkv.weight").c_str(), "beat-this");
    a.gates_w = core_gguf::require(tm, (prefix + ".to_gates.weight").c_str(), "beat-this");
    a.gates_b = core_gguf::require(tm, (prefix + ".to_gates.bias").c_str(), "beat-this");
    a.out_w = core_gguf::require(tm, (prefix + ".to_out.0.weight").c_str(), "beat-this");
    return a.gamma && a.qkv_w && a.gates_w && a.gates_b && a.out_w;
}

bool bt_bind_ff(core_gguf::tensor_map& tm, const std::string& prefix, bt_ff& f) {
    f.gamma = core_gguf::require(tm, (prefix + ".net.0.gamma").c_str(), "beat-this");
    f.w1 = core_gguf::require(tm, (prefix + ".net.1.weight").c_str(), "beat-this");
    f.b1 = core_gguf::require(tm, (prefix + ".net.1.bias").c_str(), "beat-this");
    f.w2 = core_gguf::require(tm, (prefix + ".net.4.weight").c_str(), "beat-this");
    f.b2 = core_gguf::require(tm, (prefix + ".net.4.bias").c_str(), "beat-this");
    return f.gamma && f.w1 && f.b1 && f.w2 && f.b2;
}

} // namespace

extern "C" int beat_this_n_frames(int n_samples) {
    if (n_samples <= 0)
        return 0;
    // torchaudio center=true: 1 + n // hop
    return 1 + n_samples / BEAT_THIS_HOP;
}

extern "C" int beat_this_logmel(struct beat_this_context* ctx, const float* pcm, int n_samples, float* out) {
    if (!ctx || !pcm || n_samples <= 0 || !out)
        return 0;
    const int T = beat_this_n_frames(n_samples);
    const int n_fft = ctx->n_fft, hop = ctx->hop, M = ctx->mel_bins;
    const int n_freqs = n_fft / 2 + 1;
    if (ctx->fb_freqs != n_freqs || (int)ctx->mel_fb.size() != n_freqs * M) {
        fprintf(stderr, "beat_this: filterbank is %d x %d, expected %d x %d\n", ctx->fb_freqs,
                ctx->fb_freqs ? (int)(ctx->mel_fb.size() / ctx->fb_freqs) : 0, n_freqs, M);
        return 0;
    }

    // torchaudio's `normalized="frame_length"` divides by SQRT(n_fft), not
    // n_fft — the name is misleading. Verified empirically: the ratio between
    // normalized=False and normalized="frame_length" is exactly 32.0 at
    // n_fft=1024. Getting this wrong scales every mel bin and, because the
    // features are log1p(1000*x), does NOT cancel downstream.
    const float norm = 1.0f / std::sqrt((float)n_fft);

    // `spec` MUST hold 2*n_fft, not 2*n_freqs. fft_radix2_wrapper emits the
    // FULL complex spectrum — interleaved re/im for all N bins, so 2*N floats —
    // even though only the first n_freqs = N/2+1 are the non-redundant half we
    // go on to read. Sizing it 2*n_freqs overflows the heap by ~4 KB PER FRAME
    // while still producing a numerically perfect log-mel, because the bytes it
    // corrupts are past the ones read back: the front-end fixture scored
    // cos = 1.00000000 with this bug live, and it only surfaced as an
    // intermittent wild-pointer crash once a 45 s file made it 2251 frames of
    // corruption instead of 101. Every other core_fft caller in the tree sizes
    // this 2*fft_size; this was the one that did not.
    std::vector<float> frame((size_t)n_fft), spec((size_t)2 * n_fft), mag((size_t)n_freqs);
    const int half = n_fft / 2;

    for (int t = 0; t < T; t++) {
        // center=true: frame t is centred on sample t*hop, reflect-padded.
        const int start = t * hop - half;
        for (int i = 0; i < n_fft; i++) {
            const int s = start + i;
            const float v = (s >= 0 && s < n_samples) ? pcm[s] : pcm[reflect_index(s, n_samples)];
            frame[(size_t)i] = v * ctx->hann[(size_t)i];
        }
        core_fft::fft_radix2_wrapper(frame.data(), n_fft, spec.data());

        // power=1 => MAGNITUDE, not power. Squaring here is the single easiest
        // way to produce a spectrogram that looks right and tracks wrong.
        for (int k = 0; k < n_freqs; k++) {
            const float re = spec[(size_t)2 * k], im = spec[(size_t)2 * k + 1];
            mag[(size_t)k] = std::sqrt(re * re + im * im) * norm;
        }

        // Project onto the mel filterbank: fb is [n_freqs, n_mels] row-major,
        // so mel[m] = sum_k mag[k] * fb[k * n_mels + m].
        float* dst = out + (size_t)t * M;
        for (int m = 0; m < M; m++)
            dst[m] = 0.0f;
        for (int k = 0; k < n_freqs; k++) {
            const float v = mag[(size_t)k];
            if (v == 0.0f)
                continue;
            const float* frow = ctx->mel_fb.data() + (size_t)k * M;
            for (int m = 0; m < M; m++)
                dst[m] += v * frow[m];
        }
        for (int m = 0; m < M; m++)
            dst[m] = std::log1p(ctx->log_multiplier * dst[m]);
    }
    return T;
}

extern "C" struct beat_this_context* beat_this_init(const char* model_path, int n_threads) {
    if (!model_path)
        return nullptr;
    auto* ctx = new beat_this_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    ctx->debug = crispasr_env::get("CRISPASR_BEAT_THIS_DEBUG") != nullptr;

    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        fprintf(stderr, "beat_this: cannot open %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    ctx->sample_rate = (int)core_gguf::kv_u32(gctx, "beat-this.sample_rate", BEAT_THIS_SAMPLE_RATE);
    ctx->n_fft = (int)core_gguf::kv_u32(gctx, "beat-this.n_fft", BEAT_THIS_N_FFT);
    ctx->hop = (int)core_gguf::kv_u32(gctx, "beat-this.hop_length", BEAT_THIS_HOP);
    ctx->mel_bins = (int)core_gguf::kv_u32(gctx, "beat-this.mel_bins", BEAT_THIS_MEL_BINS);
    ctx->log_multiplier = core_gguf::kv_f32(gctx, "beat-this.log_multiplier", 1000.0f);
    core_gguf::free_metadata(gctx);

    ctx->backend = core_cpu_backend::init();
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }
    if (!core_gguf::load_weights(model_path, ctx->backend, "beat-this", ctx->wl)) {
        beat_this_free(ctx);
        return nullptr;
    }

    ggml_tensor* fb = core_gguf::try_get(ctx->wl.tensors, "aux.mel_filterbank");
    if (!fb) {
        fprintf(stderr, "beat_this: model has no aux.mel_filterbank — reconvert with --filterbank\n");
        beat_this_free(ctx);
        return nullptr;
    }
    // ggml ne is reversed vs the [513,128] row-major source: ne[0]=128, ne[1]=513.
    ctx->fb_freqs = (int)fb->ne[1];
    const size_t n = (size_t)ggml_nelements(fb);
    ctx->mel_fb.resize(n);
    ggml_backend_tensor_get(fb, ctx->mel_fb.data(), 0, n * sizeof(float));

    build_hann_periodic(ctx->n_fft, ctx->hann);

    // Bind the stem weights. require() is loud on a miss: a silently-absent
    // weight would compute a plausible-looking activation from zeros.
    ctx->bn1d_scale = core_gguf::require(ctx->wl.tensors, "frontend.stem.bn1d.scale", "beat-this");
    ctx->bn1d_offset = core_gguf::require(ctx->wl.tensors, "frontend.stem.bn1d.offset", "beat-this");
    ctx->stem_w = core_gguf::require(ctx->wl.tensors, "frontend.stem.conv2d.weight", "beat-this");
    ctx->stem_b = core_gguf::require(ctx->wl.tensors, "frontend.stem.conv2d.bias", "beat-this");
    if (!ctx->bn1d_scale || !ctx->bn1d_offset || !ctx->stem_w || !ctx->stem_b) {
        beat_this_free(ctx);
        return nullptr;
    }

    // Three frontend blocks, dims 32 -> 64 -> 128 (heads 1 -> 2 -> 4).
    for (int i = 0; i < 3; i++) {
        const int dim = 32 << i;
        const std::string bp = "frontend.blocks." + std::to_string(i);
        bt_block& b = ctx->blocks[i];
        const bool ok = bt_bind_attn(ctx->wl.tensors, bp + ".partial.attnF", dim, b.partial.attnF) &&
                        bt_bind_ff(ctx->wl.tensors, bp + ".partial.ffF", b.partial.ffF) &&
                        bt_bind_attn(ctx->wl.tensors, bp + ".partial.attnT", dim, b.partial.attnT) &&
                        bt_bind_ff(ctx->wl.tensors, bp + ".partial.ffT", b.partial.ffT);
        b.conv_w = core_gguf::require(ctx->wl.tensors, (bp + ".conv2d.weight").c_str(), "beat-this");
        b.conv_b = core_gguf::require(ctx->wl.tensors, (bp + ".conv2d.bias").c_str(), "beat-this");
        if (!ok || !b.conv_w || !b.conv_b) {
            beat_this_free(ctx);
            return nullptr;
        }
    }

    ctx->linear_w = core_gguf::require(ctx->wl.tensors, "frontend.linear.weight", "beat-this");
    ctx->linear_b = core_gguf::require(ctx->wl.tensors, "frontend.linear.bias", "beat-this");
    ctx->out_norm = core_gguf::require(ctx->wl.tensors, "transformer_blocks.norm.gamma", "beat-this");
    ctx->head_w = core_gguf::require(ctx->wl.tensors, "task_heads.beat_downbeat_lin.weight", "beat-this");
    ctx->head_b = core_gguf::require(ctx->wl.tensors, "task_heads.beat_downbeat_lin.bias", "beat-this");
    if (!ctx->linear_w || !ctx->linear_b || !ctx->out_norm || !ctx->head_w || !ctx->head_b) {
        beat_this_free(ctx);
        return nullptr;
    }

    for (int i = 0; i < BEAT_THIS_N_LAYERS; i++) {
        const std::string lp = "transformer_blocks.layers." + std::to_string(i);
        if (!bt_bind_attn(ctx->wl.tensors, lp + ".0", BEAT_THIS_DIM, ctx->layers[i].attn) ||
            !bt_bind_ff(ctx->wl.tensors, lp + ".1", ctx->layers[i].ff)) {
            beat_this_free(ctx);
            return nullptr;
        }
    }

    if (ctx->debug)
        fprintf(stderr, "beat_this: sr=%d n_fft=%d hop=%d mel=%d fb=%dx%d\n", ctx->sample_rate, ctx->n_fft, ctx->hop,
                ctx->mel_bins, ctx->fb_freqs, (int)(n / (size_t)std::max(1, ctx->fb_freqs)));
    return ctx;
}

extern "C" void beat_this_free(struct beat_this_context* ctx) {
    if (!ctx)
        return;
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" int beat_this_sample_rate(const struct beat_this_context* ctx) {
    return ctx ? ctx->sample_rate : 0;
}

extern "C" float beat_this_tempo_bpm(const struct beat_this_event* ev, int n) {
    if (!ev || n < 2)
        return 0.0f;
    std::vector<float> iois;
    iois.reserve((size_t)n - 1);
    for (int i = 1; i < n; i++) {
        const float d = ev[i].time_s - ev[i - 1].time_s;
        if (d > 0.0f)
            iois.push_back(d);
    }
    if (iois.empty())
        return 0.0f;
    // Median, not mean: a single missed or doubled beat skews a mean badly,
    // and beat sequences routinely have both.
    std::sort(iois.begin(), iois.end());
    const float med = iois[iois.size() / 2];
    return med > 0.0f ? 60.0f / med : 0.0f;
}

// Stem forward: log-mel (T,128) -> (T, 32 freq, 32 ch).
//
// LAYOUT. The reference is torch (b, c, f, t); ggml is the reverse, so the
// stem output here is ne = (t, f, c). The log-mel arrives row-major (T,128),
// i.e. ne = (128, T) with mel fastest, and conv2d wants ne[0] = W = TIME — so
// it is transposed first.
//
// The conv kernel is stored as torch (oc, ic, kf, kt) row-major, which is
// already ggml ne = (kt, kf, ic, oc) — exactly ggml_conv_2d's expectation,
// with s0/p0 acting along time and s1/p1 along frequency. No permute needed.
//
// GELU IS THE EXACT (erf) FORM. torch nn.GELU defaults to approximate='none';
// ggml_gelu is the TANH approximation and would drift. See the GELU-variant
// entry in the common-divergence list in docs/contributing.md.
//
// Builds the whole forward once and reads back any set of named stages from a
// single compute. `track()` asks for out_beat and out_downbeat together, which
// is why this takes a list rather than one stage: the two logits share the
// entire network and computing them in separate passes would double the work.
static bool bt_run(struct beat_this_context* ctx, const float* logmel, int T, const std::vector<std::string>& names,
                   std::vector<std::vector<float>>& outs, std::vector<std::array<int64_t, 4>>* nes) {
    if (!ctx || !logmel || T <= 0 || names.empty())
        return false;
    const int M = ctx->mel_bins;

    const size_t nodes = 1024;
    ggml_init_params ip = {nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true};
    ggml_context* c0 = ggml_init(ip);
    if (!c0)
        return 0;
    ggml_cgraph* gf = ggml_new_graph_custom(c0, nodes, false);

    // (128, T) as stored; ne[0] = mel.
    ggml_tensor* x = ggml_new_tensor_2d(c0, GGML_TYPE_F32, M, T);
    ggml_set_name(x, "logmel");
    ggml_set_input(x);

    // bn1d is per-FREQUENCY and applies BEFORE any conv, so it acts on the mel
    // axis while that axis is still ne[0].
    ggml_tensor* h = ggml_add(c0, ggml_mul(c0, x, ctx->bn1d_scale), ctx->bn1d_offset);

    // -> ne (T, 128, 1, 1): time fastest, as conv2d wants.
    h = ggml_cont(c0, ggml_transpose(c0, h));
    h = ggml_reshape_4d(c0, h, T, M, 1, 1);

    // stride (time=1, freq=4), pad (time=1, freq=0) -> (T, 32, 32, 1)
    h = ggml_conv_2d(c0, ctx->stem_w, h, /*s0*/ 1, /*s1*/ 4, /*p0*/ 1, /*p1*/ 0, /*d0*/ 1, /*d1*/ 1);
    h = ggml_add(c0, h, ggml_reshape_3d(c0, ctx->stem_b, 1, 1, ctx->stem_b->ne[0]));
    h = ggml_gelu_erf(c0, h);

    bt_stages sm;
    sm.add("stem", h);

    // Positions are just 0..N-1; the sequence length differs per axis and per
    // block (frequency halves each time, time never changes).
    ggml_tensor* pos_t = ggml_new_tensor_1d(c0, GGML_TYPE_I32, T);
    ggml_set_input(pos_t);
    ggml_tensor* pos_f[3];

    for (int i = 0; i < 3; i++) {
        const int64_t F = h->ne[1];
        pos_f[i] = ggml_new_tensor_1d(c0, GGML_TYPE_I32, F);
        ggml_set_input(pos_f[i]);

        h = ggml_reshape_3d(c0, h, h->ne[0], F, h->ne[2]);
        h = bt_partial_ft(c0, ctx->blocks[i].partial, h, pos_f[i], pos_t, i, sm);

        // conv (kt=3, kf=2) stride (t=1, f=2) pad (t=1, f=0): halves frequency,
        // doubles channels. Kernel is stored torch (oc, ic, kf, kt) row-major,
        // which is already ggml ne = (kt, kf, ic, oc) — no permute.
        h = ggml_reshape_4d(c0, h, h->ne[0], h->ne[1], h->ne[2], 1);
        h = ggml_conv_2d(c0, ctx->blocks[i].conv_w, h, 1, 2, 1, 0, 1, 1);
        h = ggml_add(c0, h, ggml_reshape_3d(c0, ctx->blocks[i].conv_b, 1, 1, ctx->blocks[i].conv_b->ne[0]));
        h = ggml_gelu_erf(c0, h);
        sm.add("blk" + std::to_string(i), h);
    }

    // Rearrange "b c f t -> b t (c f)". The flattened index is c*F + f, i.e.
    // FREQUENCY varies fastest inside each channel — so in ggml, whose ne[0] is
    // the fastest axis, the target is ne = (f, c, t) before the reshape. The
    // opposite order transposes the 1024-vector and produces a plausible but
    // wrong projection, and this is the only genuinely new risk in the tail:
    // the layers below are the frontend's code at different dims.
    h = ggml_cont(c0, ggml_permute(c0, h, 2, 0, 1, 3)); // (t,f,c) -> (f,c,t)
    h = ggml_reshape_2d(c0, h, h->ne[0] * h->ne[1], T);
    h = ggml_add(c0, ggml_mul_mat(c0, ctx->linear_w, h), ctx->linear_b);
    h = ggml_reshape_3d(c0, h, BEAT_THIS_DIM, T, 1);
    sm.add("linear", h);

    for (int i = 0; i < BEAT_THIS_N_LAYERS; i++) {
        h = ggml_add(c0, h, bt_attention(c0, ctx->layers[i].attn, h, pos_t));
        h = ggml_add(c0, h, bt_feedforward(c0, ctx->layers[i].ff, h));
    }
    // norm_output: the reference hook sits on transformer_blocks as a whole, so
    // its `transformer` capture is AFTER this final RMSNorm, not before it.
    h = bt_norm(c0, h, ctx->out_norm);
    sm.add("transformer", h);

    // SumHead. beat is the SUM of both logits, downbeat is the second alone —
    // upstream's stated reason is to suppress downbeats that are not beats. It
    // is not a two-way softmax and the two outputs are not independent.
    ggml_tensor* bd = ggml_add(c0, ggml_mul_mat(c0, ctx->head_w, h), ctx->head_b); // (2, T, 1)
    const size_t bsz = ggml_element_size(bd);
    // Each logit row is a strided column of `bd`; cont then reshape to ne (T,1)
    // so the dump reverses to the reference's (1, T).
    auto logit = [&](size_t off) {
        return ggml_reshape_2d(c0, ggml_cont(c0, ggml_view_2d(c0, bd, 1, T, bd->nb[1], off)), T, 1);
    };
    ggml_tensor *lb = logit(0), *ld = logit(bsz);
    sm.add("out_downbeat", ld);
    sm.add("out_beat", ggml_add(c0, lb, ld));

    std::vector<ggml_tensor*> want;
    want.reserve(names.size());
    for (const std::string& n : names) {
        ggml_tensor* t = sm.get(n);
        if (!t) {
            fprintf(stderr, "beat_this: unknown stage '%s'\n", n.c_str());
            ggml_free(c0);
            return false;
        }
        want.push_back(t);
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }

    if (ctx->debug)
        fprintf(stderr, "beat_this: T=%d graph nodes=%d/%zu ctx_mem=%zu/%zu\n", T, ggml_graph_n_nodes(gf), nodes,
                ggml_used_mem(c0), ip.mem_size);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    bool ok = ga && ggml_gallocr_alloc_graph(ga, gf);
    if (ctx->debug && ga)
        fprintf(stderr, "beat_this: compute buffer %.1f MB\n", ggml_gallocr_get_buffer_size(ga, 0) / 1048576.0);
    if (ok) {
        ggml_backend_tensor_set(x, logmel, 0, (size_t)T * M * sizeof(float));
        // Only the stages upstream of the requested ones are in the graph, so
        // later blocks' position inputs may be unallocated — writing to those
        // would be a null-deref, not a no-op.
        // Sized for the LONGEST position vector, not for T: the frequency axis
        // is 32 entries regardless of how short the audio is, so sizing this T
        // would overrun for anything under ~0.64 s.
        std::vector<int32_t> p((size_t)std::max(T, 32));
        auto fill_pos = [&](ggml_tensor* t) {
            if (!t->data)
                return;
            const int n_p = (int)t->ne[0];
            for (int j = 0; j < n_p; j++)
                p[(size_t)j] = j;
            ggml_backend_tensor_set(t, p.data(), 0, (size_t)n_p * sizeof(int32_t));
        };
        fill_pos(pos_t);
        for (int i = 0; i < 3; i++)
            fill_pos(pos_f[i]);
        core_quant_bcast::audit(gf, "beat-this");
        ok = ggml_backend_graph_compute(ctx->backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        outs.resize(want.size());
        if (nes)
            nes->resize(want.size());
        for (size_t i = 0; i < want.size(); i++) {
            const size_t n = (size_t)ggml_nelements(want[i]);
            outs[i].resize(n);
            ggml_backend_tensor_get(want[i], outs[i].data(), 0, n * sizeof(float));
            if (nes)
                for (int d = 0; d < 4; d++)
                    (*nes)[i][(size_t)d] = want[i]->ne[d];
            if (ctx->debug)
                fprintf(stderr, "beat_this: %s ne=(%lld,%lld,%lld,%lld)\n", names[i].c_str(), (long long)want[i]->ne[0],
                        (long long)want[i]->ne[1], (long long)want[i]->ne[2], (long long)want[i]->ne[3]);
        }
    }
    if (ga)
        ggml_gallocr_free(ga);
    ggml_free(c0);
    return ok;
}

extern "C" int beat_this_debug_stage(struct beat_this_context* ctx, const float* logmel, int T, const char* stage,
                                     float* out, int max_out, int64_t* ne_out) {
    if (!ctx || !logmel || T <= 0 || !out || !stage)
        return 0;
    std::vector<std::vector<float>> outs;
    std::vector<std::array<int64_t, 4>> nes;
    if (!bt_run(ctx, logmel, T, {std::string(stage)}, outs, &nes) || outs.empty())
        return 0;
    if (ne_out)
        for (int i = 0; i < 4; i++)
            ne_out[i] = nes[0][(size_t)i];
    if ((int)outs[0].size() > max_out) {
        fprintf(stderr, "beat_this: stage '%s' needs %d floats, buffer holds %d\n", stage, (int)outs[0].size(),
                max_out);
        return 0;
    }
    std::memcpy(out, outs[0].data(), outs[0].size() * sizeof(float));
    return (int)outs[0].size();
}

extern "C" int beat_this_debug_stem(struct beat_this_context* ctx, const float* logmel, int T, float* out) {
    return beat_this_debug_stage(ctx, logmel, T, "stem", out, T * 32 * 32, nullptr);
}

// Framewise logits for a whole piece, reproducing upstream `split_piece` /
// `aggregate_prediction` (beat_this/inference.py) exactly.
//
// WHY THE WINDOWING IS NOT OPTIONAL. The model was not trained on its input
// edges — the max-pool in the training loss discards `border_size` frames on
// each side — so the first and last 6 frames of any chunk's prediction are
// untrusted. Chunks therefore OVERLAP by 2*border and each chunk's border is
// cut before it is written. Getting this wrong does not fail loudly; it
// produces drifting or doubled beats at every 30-second seam.
extern "C" int beat_this_logits(struct beat_this_context* ctx, const float* logmel, int T, float* beat,
                                float* downbeat) {
    if (!ctx || !logmel || T <= 0 || !beat || !downbeat)
        return 0;
    const int chunk = BEAT_THIS_CHUNK_FRAMES, border = BEAT_THIS_BORDER_FRAMES;
    const int stride = chunk - 2 * border;

    // starts = arange(-border, T - border, stride)
    std::vector<int> starts;
    for (int s = -border; s < T - border; s += stride)
        starts.push_back(s);
    if (starts.empty())
        starts.push_back(-border);
    // avoid_short_end: pull the last chunk left so it ends at the piece end,
    // rather than running a short (and therefore differently-normalised) chunk.
    if (T > stride)
        starts.back() = T - (chunk - border);

    // -1000 is upstream's "no prediction here" sentinel: it is a logit, so it
    // is probability zero even in float64 and can never win a peak-pick. Any
    // frame left at -1000 would mean a coverage gap in the chunking.
    for (int i = 0; i < T; i++)
        beat[i] = downbeat[i] = -1000.0f;

    // keep_first: walk in REVERSE so earlier chunks overwrite later ones, which
    // is what upstream's reversed() achieves.
    std::vector<float> pad((size_t)chunk * BEAT_THIS_MEL_BINS);
    for (int ci = (int)starts.size() - 1; ci >= 0; ci--) {
        const int start = starts[(size_t)ci];
        const int lo = std::max(start, 0), hi = std::min(start + chunk, T);
        const int left = std::max(0, -start);
        const int len = hi - lo;
        const int right = std::max(0, std::min(border, start + chunk - T));
        const int clen = left + len + right;

        std::fill(pad.begin(), pad.begin() + (size_t)clen * BEAT_THIS_MEL_BINS, 0.0f);
        std::memcpy(pad.data() + (size_t)left * BEAT_THIS_MEL_BINS, logmel + (size_t)lo * BEAT_THIS_MEL_BINS,
                    (size_t)len * BEAT_THIS_MEL_BINS * sizeof(float));

        std::vector<std::vector<float>> outs;
        if (!bt_run(ctx, pad.data(), clen, {"out_beat", "out_downbeat"}, outs, nullptr))
            return 0;

        // Discard `border` frames on each side, then write the remainder at
        // [start+border, start+chunk-border) — clamped, because for a piece
        // shorter than a chunk that range runs past the end (upstream relies on
        // Python slice clamping doing this implicitly).
        const int keep = clen - 2 * border;
        for (int j = 0; j < keep; j++) {
            const int dst = start + border + j;
            if (dst < 0 || dst >= T)
                continue;
            beat[dst] = outs[0][(size_t)(border + j)];
            downbeat[dst] = outs[1][(size_t)(border + j)];
        }
    }
    return T;
}

namespace {

// Peak-pick: keep frames that equal the max over a +/-3 frame window and are
// above logit 0 (probability 0.5). NOTE the comparison upstream is `!=` against
// the max-pool, not `>` against the neighbours — a plateau of equal values
// therefore keeps EVERY frame in the plateau, and deduplicate_peaks below is
// what collapses it. Using a strict local maximum would silently drop beats
// that land exactly between two frames.
void bt_pick_peaks(const float* logit, int T, std::vector<int>& peaks) {
    peaks.clear();
    for (int i = 0; i < T; i++) {
        if (!(logit[i] > 0.0f))
            continue;
        float mx = logit[i];
        for (int j = std::max(0, i - 3); j <= std::min(T - 1, i + 3); j++)
            mx = std::max(mx, logit[j]);
        if (logit[i] == mx)
            peaks.push_back(i);
    }
}

// Collapse runs of peaks no more than `width` frames apart into their MEAN
// frame index — which is fractional, and deliberately so: a beat whose energy
// straddles two frames lands between them, and rounding to the first frame
// would bias every such beat 10 ms early. Upstream's condition compares the
// next peak against the RUNNING MEAN, not against the previous peak.
void bt_deduplicate_peaks(const std::vector<int>& peaks, int width, std::vector<double>& out) {
    out.clear();
    if (peaks.empty())
        return;
    double p = peaks[0];
    int c = 1;
    for (size_t i = 1; i < peaks.size(); i++) {
        const double p2 = peaks[i];
        if (p2 - p <= (double)width) {
            c++;
            p += (p2 - p) / c; // incremental mean
        } else {
            out.push_back(p);
            p = p2;
            c = 1;
        }
    }
    out.push_back(p);
}

} // namespace

extern "C" int beat_this_events_from_logits(const float* lb, const float* ld, int T, struct beat_this_event* out,
                                            int max_events) {
    if (!lb || !ld || T <= 0 || !out || max_events <= 0)
        return -1;

    std::vector<int> bp, dp;
    bt_pick_peaks(lb, T, bp);
    bt_pick_peaks(ld, T, dp);
    std::vector<double> bf, df;
    bt_deduplicate_peaks(bp, 1, bf);
    bt_deduplicate_peaks(dp, 1, df);

    const double fps = (double)BEAT_THIS_FPS;
    std::vector<double> bt_s, dt_s;
    for (double f : bf)
        bt_s.push_back(f / fps);
    for (double f : df)
        dt_s.push_back(f / fps);

    // Snap each downbeat to its NEAREST BEAT, in seconds and after dedup — a
    // downbeat that is not also a beat is not representable downstream, and the
    // reference makes the same guarantee. Then drop downbeats that collapsed
    // onto the same beat.
    for (double& d : dt_s) {
        if (bt_s.empty())
            break;
        double best = bt_s[0];
        for (double b : bt_s)
            if (std::fabs(b - d) < std::fabs(best - d))
                best = b;
        d = best;
    }
    std::sort(dt_s.begin(), dt_s.end());
    dt_s.erase(std::unique(dt_s.begin(), dt_s.end()), dt_s.end());

    // Emit every beat once, flagging those that are also downbeats. Downbeats
    // are a subset of beats by construction after the snap above.
    int n = 0;
    size_t di = 0;
    for (double b : bt_s) {
        if (n >= max_events)
            break;
        while (di < dt_s.size() && dt_s[di] < b)
            di++;
        const bool is_db = di < dt_s.size() && dt_s[di] == b;
        out[n].time_s = (float)b;
        out[n].is_downbeat = is_db ? 1 : 0;
        n++;
    }
    return n;
}

extern "C" int beat_this_track(struct beat_this_context* ctx, const float* pcm, int n_samples,
                               struct beat_this_event* out, int max_events) {
    if (!ctx || !pcm || n_samples <= 0 || !out || max_events <= 0)
        return -1;

    const int T = beat_this_n_frames(n_samples);
    std::vector<float> lm((size_t)T * BEAT_THIS_MEL_BINS);
    if (beat_this_logmel(ctx, pcm, n_samples, lm.data()) != T)
        return -1;

    std::vector<float> lb((size_t)T), ld((size_t)T);
    if (beat_this_logits(ctx, lm.data(), T, lb.data(), ld.data()) != T)
        return -1;

    return beat_this_events_from_logits(lb.data(), ld.data(), T, out, max_events);
}
