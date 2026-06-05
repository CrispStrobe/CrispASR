// orpheus_snac.cpp — SNAC 24 kHz decoder (hubertsiuzdak/snac_24khz).
//
// Architecture (verbatim from the live model + snac/{snac.py,layers.py}
// at the canonical 24 kHz config — sampling_rate=24000, encoder_dim=48,
// encoder_rates=[2,4,8,8] → latent_dim=48·2^4=768, decoder_dim=1024,
// decoder_rates=[8,8,4,2], depthwise=True, noise=True, vq_strides=[4,2,1]):
//
//   quantizer.from_codes(codes):   one entry per codebook, then sum.
//     for k in 0..2:
//         z = codebook[k][codes[k]]                       (codebook_dim=8, T_gk)
//         z = out_proj_k(z)                               (latent_dim=768, T_gk)
//         z = repeat_interleave(z, vq_strides[k], axis=T) (768, T_q)  T_q=4·T_super
//     z_q = sum_k z                                       (768, T_q)
//
//   decoder.model:
//     [0] depthwise Conv1d(768, 768, k=7, p=3, groups=768)    (z_q)
//     [1] pointwise Conv1d(768, 1024, k=1)                    → "snac_dec_pre"
//     [2] DecoderBlock(1024 →  512, stride 8, k=16, p=4)      → "snac_dec_blk0"
//     [3] DecoderBlock( 512 →  256, stride 8, k=16, p=4)      → "snac_dec_blk1"
//     [4] DecoderBlock( 256 →  128, stride 4, k= 8, p=2)      → "snac_dec_blk2"
//     [5] DecoderBlock( 128 →   64, stride 2, k= 4, p=1)      → "snac_dec_blk3"
//     [6] Snake1d(64)
//     [7] Conv1d(64, 1, k=7, p=3)
//     [8] Tanh                                                → "snac_pcm"
//
//   DecoderBlock layout (input_dim → output_dim, stride s):
//     0. Snake1d(input_dim)
//     1. ConvTranspose1d(input_dim, output_dim, k=2s, s, p=ceil(s/2), op=s%2)
//        For SNAC strides [8,8,4,2], all s%2=0, so op=0, p=s/2.
//     2. NoiseBlock(output_dim) — at inference time the noise term is
//        zero (training-only injection), reducing it to identity. The
//        Python reference monkey-patches NoiseBlock.forward to match.
//     3. ResidualUnit(output_dim, dilation=1)
//     4. ResidualUnit(output_dim, dilation=3)
//     5. ResidualUnit(output_dim, dilation=9)
//
//   ResidualUnit (depthwise=True, output_dim → output_dim):
//     y = Snake1d → depthwise Conv1d(d, d, k=7, p=3·dil, dil, groups=d)
//                 → Snake1d → pointwise Conv1d(d, d, k=1)
//     return x + y
//
//   Snake1d:  y = x + (1/(α + 1e-9)) · sin²(α · x), α per-channel (1, C, 1)
//
// Tensor layout convention used internally: (C, T) channels-innermost,
// i.e. ggml `ne = [C, T]`. Conv1d / ConvTranspose1d helpers transpose
// to `[T, C]` for the ggml call and back.
//
// Critical gotcha (LEARNINGS.md Lesson 2): PyTorch's ConvTranspose1d
// kernel index k satisfies `j·s + k − p = i`, NOT reversed. Mirroring
// the kernel produces near-correct-energy output that fails cosine.
// `ggml_conv_transpose_1d` follows the same convention, so the converter
// passes weights through unchanged — do NOT add a "fix" here.

#include "orpheus_snac.h"
#include "core/activation.h"
#include "core/conv.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct snac_quant {
    ggml_tensor* codebook = nullptr;   // (codebook_dim=8, codebook_size=4096) F16
    ggml_tensor* out_proj_w = nullptr; // (1, codebook_dim=8, latent_dim=768)  F16
    ggml_tensor* out_proj_b = nullptr; // (latent_dim=768,) F32
};

struct snac_res_unit {
    ggml_tensor* alpha0 = nullptr;  // (dim,) F32
    ggml_tensor* conv0_w = nullptr; // (7, 1, dim) F16  — depthwise k=7 dilation=d
    ggml_tensor* conv0_b = nullptr; // (dim,) F32
    ggml_tensor* alpha1 = nullptr;  // (dim,) F32
    ggml_tensor* conv1_w = nullptr; // (1, dim, dim) F16 — pointwise k=1
    ggml_tensor* conv1_b = nullptr; // (dim,) F32
};

struct snac_block {
    ggml_tensor* alpha = nullptr;   // (input_dim,) F32
    ggml_tensor* up_w = nullptr;    // (K=2s, output_dim, input_dim) F16
    ggml_tensor* up_b = nullptr;    // (output_dim,) F32
    ggml_tensor* noise_w = nullptr; // (1, output_dim, output_dim) F16 — bound but unused
    std::array<snac_res_unit, 3> res;
};

struct snac_hparams {
    uint32_t sample_rate = 24000;
    uint32_t n_codebooks = 3;
    uint32_t codebook_size = 4096;
    uint32_t codebook_dim = 8;
    uint32_t latent_dim = 768;   // not stored in GGUF — derived from snac.dec.in0.weight
    uint32_t decoder_dim = 1024; // derived from snac.dec.in1.weight
    uint32_t hop_length = 512;
    std::vector<uint32_t> vq_strides;         // [4, 2, 1]
    std::vector<uint32_t> decoder_strides;    // [8, 8, 4, 2]
    std::vector<uint32_t> residual_dilations; // [1, 3, 9]
};

} // namespace

struct snac_decoder_ctx {
    snac_decoder_params params{};
    int n_threads = 4;

    snac_hparams hp;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    std::array<snac_quant, 3> quantizers;
    ggml_tensor* dec_in0_w = nullptr; // depthwise (7, 1, 768)
    ggml_tensor* dec_in0_b = nullptr;
    ggml_tensor* dec_in1_w = nullptr; // pointwise (1, 768, 1024)
    ggml_tensor* dec_in1_b = nullptr;
    std::array<snac_block, 4> blocks;
    ggml_tensor* out_alpha = nullptr; // (64,)
    ggml_tensor* out_w = nullptr;     // (7, 64, 1)
    ggml_tensor* out_b = nullptr;     // (1,)

    // Per-graph compute meta scratch — re-used across decode calls. Sized to
    // hold a graph for T_super up to ~16 (orpheus streaming windows are 4).
    std::vector<uint8_t> compute_meta;

    ~snac_decoder_ctx() {
        if (ctx_w) {
            ggml_free(ctx_w);
        }
        if (buf_w) {
            ggml_backend_buffer_free(buf_w);
        }
        if (backend && backend != backend_cpu) {
            ggml_backend_free(backend);
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

namespace {

// ---------------------------------------------------------------------------
// Metadata + tensor binding.
// ---------------------------------------------------------------------------

static std::vector<uint32_t> kv_u32_array(gguf_context* g, const char* key) {
    std::vector<uint32_t> out;
    const int k = gguf_find_key(g, key);
    if (k < 0) {
        return out;
    }
    if (gguf_get_kv_type(g, k) != GGUF_TYPE_ARRAY) {
        return out;
    }
    const enum gguf_type at = gguf_get_arr_type(g, k);
    const int n = gguf_get_arr_n(g, k);
    out.resize((size_t)n);
    if (at == GGUF_TYPE_UINT32 || at == GGUF_TYPE_INT32) {
        const auto* d = (const uint32_t*)gguf_get_arr_data(g, k);
        for (int i = 0; i < n; i++) {
            out[i] = d[i];
        }
    } else if (at == GGUF_TYPE_UINT64 || at == GGUF_TYPE_INT64) {
        const auto* d = (const uint64_t*)gguf_get_arr_data(g, k);
        for (int i = 0; i < n; i++) {
            out[i] = (uint32_t)d[i];
        }
    } else {
        out.clear();
    }
    return out;
}

static void load_metadata(snac_decoder_ctx* c, gguf_context* g) {
    auto& hp = c->hp;
    hp.sample_rate = core_gguf::kv_u32(g, "snac.sample_rate", hp.sample_rate);
    hp.n_codebooks = core_gguf::kv_u32(g, "snac.n_codebooks", hp.n_codebooks);
    hp.codebook_size = core_gguf::kv_u32(g, "snac.codebook_size", hp.codebook_size);
    hp.codebook_dim = core_gguf::kv_u32(g, "snac.codebook_dim", hp.codebook_dim);
    hp.hop_length = core_gguf::kv_u32(g, "snac.hop_length", hp.hop_length);

    hp.vq_strides = kv_u32_array(g, "snac.vq_strides");
    if (hp.vq_strides.empty()) {
        hp.vq_strides = {4, 2, 1};
    }
    hp.decoder_strides = kv_u32_array(g, "snac.decoder_strides");
    if (hp.decoder_strides.empty()) {
        hp.decoder_strides = {8, 8, 4, 2};
    }
    hp.residual_dilations = kv_u32_array(g, "snac.residual_dilations");
    if (hp.residual_dilations.empty()) {
        hp.residual_dilations = {1, 3, 9};
    }
}

static bool bind_tensors(snac_decoder_ctx* c) {
    auto& t = c->tensors;
    const char* tag = "orpheus_snac";

    // Quantizers (we only need codebook + out_proj for decode — in_proj is
    // bound implicitly by the GGUF tensor map but unused).
    for (int k = 0; k < (int)c->hp.n_codebooks; k++) {
        char key[64];
        std::snprintf(key, sizeof(key), "snac.q.%d.codebook", k);
        c->quantizers[k].codebook = core_gguf::require(t, key, tag);
        std::snprintf(key, sizeof(key), "snac.q.%d.out_proj.weight", k);
        c->quantizers[k].out_proj_w = core_gguf::require(t, key, tag);
        std::snprintf(key, sizeof(key), "snac.q.%d.out_proj.bias", k);
        c->quantizers[k].out_proj_b = core_gguf::require(t, key, tag);
        if (!c->quantizers[k].codebook || !c->quantizers[k].out_proj_w || !c->quantizers[k].out_proj_b) {
            return false;
        }
    }

    c->dec_in0_w = core_gguf::require(t, "snac.dec.in0.weight", tag);
    c->dec_in0_b = core_gguf::require(t, "snac.dec.in0.bias", tag);
    c->dec_in1_w = core_gguf::require(t, "snac.dec.in1.weight", tag);
    c->dec_in1_b = core_gguf::require(t, "snac.dec.in1.bias", tag);
    if (!c->dec_in0_w || !c->dec_in0_b || !c->dec_in1_w || !c->dec_in1_b) {
        return false;
    }

    // Derive latent_dim from in0.weight (ne=[7, 1, latent_dim]) and
    // decoder_dim from in1.weight (ne=[1, latent_dim, decoder_dim]).
    c->hp.latent_dim = (uint32_t)c->dec_in0_w->ne[2];
    c->hp.decoder_dim = (uint32_t)c->dec_in1_w->ne[2];

    for (int b = 0; b < (int)c->hp.decoder_strides.size() && b < 4; b++) {
        char key[64];
        auto& blk = c->blocks[b];
        std::snprintf(key, sizeof(key), "snac.dec.%d.alpha", b);
        blk.alpha = core_gguf::require(t, key, tag);
        std::snprintf(key, sizeof(key), "snac.dec.%d.up.weight", b);
        blk.up_w = core_gguf::require(t, key, tag);
        std::snprintf(key, sizeof(key), "snac.dec.%d.up.bias", b);
        blk.up_b = core_gguf::require(t, key, tag);
        std::snprintf(key, sizeof(key), "snac.dec.%d.noise.weight", b);
        blk.noise_w = core_gguf::try_get(t, key); // unused at inference
        if (!blk.alpha || !blk.up_w || !blk.up_b) {
            return false;
        }
        for (int r = 0; r < 3; r++) {
            auto& u = blk.res[r];
            std::snprintf(key, sizeof(key), "snac.dec.%d.res.%d.alpha0", b, r);
            u.alpha0 = core_gguf::require(t, key, tag);
            std::snprintf(key, sizeof(key), "snac.dec.%d.res.%d.conv0.weight", b, r);
            u.conv0_w = core_gguf::require(t, key, tag);
            std::snprintf(key, sizeof(key), "snac.dec.%d.res.%d.conv0.bias", b, r);
            u.conv0_b = core_gguf::require(t, key, tag);
            std::snprintf(key, sizeof(key), "snac.dec.%d.res.%d.alpha1", b, r);
            u.alpha1 = core_gguf::require(t, key, tag);
            std::snprintf(key, sizeof(key), "snac.dec.%d.res.%d.conv1.weight", b, r);
            u.conv1_w = core_gguf::require(t, key, tag);
            std::snprintf(key, sizeof(key), "snac.dec.%d.res.%d.conv1.bias", b, r);
            u.conv1_b = core_gguf::require(t, key, tag);
            if (!u.alpha0 || !u.conv0_w || !u.conv0_b || !u.alpha1 || !u.conv1_w || !u.conv1_b) {
                return false;
            }
        }
    }

    c->out_alpha = core_gguf::require(t, "snac.dec.out.alpha", tag);
    c->out_w = core_gguf::require(t, "snac.dec.out.weight", tag);
    c->out_b = core_gguf::require(t, "snac.dec.out.bias", tag);
    return c->out_alpha && c->out_w && c->out_b;
}

// ---------------------------------------------------------------------------
// Graph helpers (all operate in (C, T) channels-innermost layout).
// ---------------------------------------------------------------------------

// Pointwise conv1d k=1: weight ne=[1, Cin, Cout] reshapes to (Cin, Cout)
// matrix. y = W @ x  where  x.ne=[Cin, T] → y.ne=[Cout, T].
// Bias is per-output-channel (ne=[Cout]) and broadcasts over T.
static ggml_tensor* pw_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    const int Cin = (int)w->ne[1];
    const int Cout = (int)w->ne[2];
    ggml_tensor* W = ggml_reshape_2d(ctx, w, Cin, Cout); // (Cin, Cout) F16/F32
    ggml_tensor* y = ggml_mul_mat(ctx, W, x);            // (Cout, T)
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// Depthwise conv1d (groups = C). Weight ne=[K, 1, C]. Same-padding
// (centered): pads `dil*(K-1)/2` on each side along time. Stride=1.
// Input/output: (C, T).
static ggml_tensor* dw_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int K, int dil) {
    const int C = (int)x->ne[0];
    const int T = (int)x->ne[1];
    const int p = dil * (K - 1) / 2;

    ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C)
    y = ggml_conv_1d_dw(ctx, w, y, /*s*/ 1, p, dil);         // (T_out, 1, C)
    // ggml_conv_1d_dw returns ne=[T_out, ne[2]_of_input=1 wait..., C, 1] —
    // squeeze the 1 dim and reshape to (T, C).
    y = ggml_reshape_2d(ctx, y, T, C);          // (T, C)
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (C, T)
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// Standard conv1d k=K, same-padding (p=K/2 for odd K). groups=1.
// Weight ne=[K, Cin, Cout]. Input (Cin, T) → output (Cout, T).
static ggml_tensor* conv1d_k(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int K, int p) {
    (void)K; // kernel size is implicit from w->ne[0]; explicit param kept for readability
    const int T = (int)x->ne[1];
    const int Cout = (int)w->ne[2];
    ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, Cin)
    y = ggml_conv_1d(ctx, w, y, /*s*/ 1, p, /*d*/ 1);        // (T, Cout, 1)
    y = ggml_reshape_2d(ctx, y, T, Cout);                    // (T, Cout)
    y = ggml_cont(ctx, ggml_transpose(ctx, y));              // (Cout, T)
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// PyTorch ConvTranspose1d wrapper (groups=1) with symmetric cropping —
// thin wrapper around core_convt::convt1d_crop. For SNAC strides
// [8,8,4,2]: k = 2s, p = s/2, op = 0 → T_out = T_in · s.
static inline ggml_tensor* convt1d_pad(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride,
                                       int pad) {
    return core_convt::convt1d_crop(ctx, x, w, b, stride, /*crop_left=*/pad, /*crop_right=*/pad);
}

// SNAC Snake1d: y = x + (1/(α + 1e-9)) · sin²(α · x). α stored as (1,C,1)
// in pytorch but the converter flattens to (C,). We pad on the right
// with a unit dim so core_act::snake_alpha (which expects ne[0]=C) gets
// a (C,1) F32 broadcast over T. The 1e-9 clamp is canonical to SNAC's
// snake() (snac/layers.py); core_act::snake_alpha uses 1/α directly.
// For α≥1 (the SNAC checkpoint init) the 1e-9 difference is below F32
// precision in the cosine, so we use the simpler form.
static ggml_tensor* snake1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha) {
    return core_act::snake_alpha(ctx, x, alpha);
}

// Repeat each entry of x along the time axis `factor` times (analogue of
// torch.repeat_interleave(x, factor, dim=-1)). Input (C, T) → (C, T·f).
//
// Implementation: reshape x to (C, 1, T), repeat along the new axis to
// (C, f, T), then reshape to (C, T·f). The (C, f, T) → (C, T·f)
// flattening puts the f copies adjacent in the time axis, matching
// repeat_interleave semantics (a[c, 0..f-1, t] all equal x[c, t]).
static ggml_tensor* repeat_interleave_time(ggml_context* ctx, ggml_tensor* x, int factor) {
    if (factor <= 1) {
        return x;
    }
    const int C = (int)x->ne[0];
    const int T = (int)x->ne[1];
    // Reshape (C, T) → (C, 1, T)
    ggml_tensor* x3 = ggml_reshape_3d(ctx, x, C, 1, T);
    // Tile to (C, factor, T) by allocating a target then ggml_repeat.
    ggml_tensor* target = ggml_new_tensor_3d(ctx, x->type, C, factor, T);
    ggml_tensor* tiled = ggml_repeat(ctx, x3, target); // (C, factor, T)
    // Flatten to (C, T*factor): ne[0]=C, ne[1]=factor, ne[2]=T → ne[0]=C, ne[1]=factor*T
    // After ggml_repeat the result is contiguous, so reshape is safe.
    return ggml_reshape_2d(ctx, tiled, C, T * factor);
}

// One ResidualUnit: y = Snake1d → dw_conv(K=7, dil=d) → Snake1d → pw_conv(k=1).
// Returns x + y.
static ggml_tensor* residual_unit(ggml_context* ctx, ggml_tensor* x, const snac_res_unit& u, int dil) {
    ggml_tensor* y = snake1d(ctx, x, u.alpha0);
    y = dw_conv1d(ctx, y, u.conv0_w, u.conv0_b, /*K*/ 7, /*dil*/ dil);
    y = snake1d(ctx, y, u.alpha1);
    y = pw_conv1d(ctx, y, u.conv1_w, u.conv1_b);
    return ggml_add(ctx, x, y);
}

// One DecoderBlock: Snake1d → ConvT1d(stride s, k=2s, p=s/2)
//                  → NoiseBlock(identity at inference) → 3 × ResidualUnit.
static ggml_tensor* decoder_block(ggml_context* ctx, ggml_tensor* x, const snac_block& blk, int stride,
                                  const std::vector<uint32_t>& dilations) {
    x = snake1d(ctx, x, blk.alpha);
    x = convt1d_pad(ctx, x, blk.up_w, blk.up_b, stride, /*pad*/ stride / 2);
    // NoiseBlock: training-only; at inference we treat it as identity.
    // (Python reference monkey-patches to match. blk.noise_w stays bound
    // for round-trip safety but is not on the graph.)
    for (int r = 0; r < 3 && r < (int)dilations.size(); r++) {
        x = residual_unit(ctx, x, blk.res[r], (int)dilations[r]);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Build the full decode graph for a given super-frame count.
// ---------------------------------------------------------------------------

// Stage output convention: the Python reference dumps tensors as numpy
// (C, T) arrays, which become ggml ne=[T, C] (T innermost) when loaded
// by the diff harness. Our internal layout is ne=[C, T] (C innermost),
// so every named output stage transposes to (T, C) before naming. The
// linear byte order must match for the element-wise diff to be valid.
static ggml_tensor* stage_out(ggml_context* ctx, ggml_tensor* x, const char* name, ggml_cgraph* gf) {
    if (ggml_n_dims(x) >= 2) {
        x = ggml_cont(ctx, ggml_transpose(ctx, x)); // (C, T) → (T, C)
    } else {
        x = ggml_cont(ctx, x);
    }
    ggml_set_name(x, name);
    ggml_set_output(x);
    if (gf) {
        ggml_build_forward_expand(gf, x);
    }
    return x;
}

static ggml_cgraph* build_decode_graph(snac_decoder_ctx* c, ggml_context* ctx0, int T_super) {
    const auto& hp = c->hp;
    const int n_q = (int)hp.n_codebooks;
    const int latent_dim = (int)hp.latent_dim;
    const int T_q = T_super * (int)hp.vq_strides[0]; // 4·T_super for vq_strides=[4,2,1]

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // ── Inputs: codes_0/1/2 as i32 1D tensors. ─────────────────────────────
    // codes_k has T_q / vq_strides[k] entries; vq_strides=[4,2,1] for SNAC
    // 24 kHz, so codes_0 has T_super entries (1 per super-frame), codes_1
    // has 2·T_super (2/sf), codes_2 has 4·T_super (4/sf).
    std::array<ggml_tensor*, 3> codes_in{};
    for (int k = 0; k < n_q; k++) {
        const int T_gk = T_q / (int)hp.vq_strides[k];
        char name[32];
        std::snprintf(name, sizeof(name), "snac_codes_%d_in", k);
        codes_in[k] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_gk);
        ggml_set_name(codes_in[k], name);
        ggml_set_input(codes_in[k]);
    }

    // ── Quantizer.from_codes ────────────────────────────────────────────────
    ggml_tensor* z_q = nullptr;
    for (int k = 0; k < n_q; k++) {
        const auto& q = c->quantizers[k];
        // codebook lookup: (codebook_dim=8, T_gk).
        ggml_tensor* z = ggml_get_rows(ctx0, q.codebook, codes_in[k]);
        z = ggml_cont(ctx0, ggml_cast(ctx0, z, GGML_TYPE_F32));
        // out_proj 1×1: (8, T_gk) → (latent_dim, T_gk).
        z = pw_conv1d(ctx0, z, q.out_proj_w, q.out_proj_b);
        // Upsample to T_q time steps (vq_strides[k] frames per code).
        const int factor = (int)(hp.vq_strides[k]);
        z = repeat_interleave_time(ctx0, z, factor);
        if (k == 0) {
            z_q = z;
        } else {
            z_q = ggml_add(ctx0, z_q, z);
        }
    }
    z_q = ggml_cont(ctx0, z_q);
    stage_out(ctx0, z_q, "snac_quant_out", gf);
    (void)T_q;
    (void)latent_dim;

    // ── decoder.model[0..1] : depthwise conv (k=7) + pointwise conv (k=1) ──
    ggml_tensor* h = dw_conv1d(ctx0, z_q, c->dec_in0_w, c->dec_in0_b, /*K*/ 7, /*dil*/ 1);
    h = pw_conv1d(ctx0, h, c->dec_in1_w, c->dec_in1_b); // (decoder_dim, T_q)
    h = ggml_cont(ctx0, h);
    stage_out(ctx0, h, "snac_dec_pre", gf);

    // ── decoder.model[2..5] : 4 DecoderBlocks ──────────────────────────────
    for (int b = 0; b < 4 && b < (int)hp.decoder_strides.size(); b++) {
        h = decoder_block(ctx0, h, c->blocks[b], (int)hp.decoder_strides[b], hp.residual_dilations);
        h = ggml_cont(ctx0, h);
        char name[32];
        std::snprintf(name, sizeof(name), "snac_dec_blk%d", b);
        stage_out(ctx0, h, name, gf);
    }

    // ── decoder.model[6..8] : Snake1d → Conv1d(64,1,k=7) → Tanh ────────────
    h = snake1d(ctx0, h, c->out_alpha);
    h = conv1d_k(ctx0, h, c->out_w, c->out_b, /*K*/ 7, /*p*/ 3);
    h = ggml_tanh(ctx0, h);
    // Output shape: (1, T_q · 512). Reshape to a 1D (T_q · 512,) for clarity.
    // Single-channel — no transpose needed since it's effectively 1D.
    const int T_pcm = (int)h->ne[1];
    h = ggml_reshape_1d(ctx0, h, (int64_t)T_pcm);
    h = ggml_cont(ctx0, h);
    ggml_set_name(h, "snac_pcm");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);
    return gf;
}

static float* run_graph_and_extract(snac_decoder_ctx* c, const int32_t* c0, int n0, const int32_t* c1, int n1,
                                    const int32_t* c2, int n2, const char* stage_name, int* out_n) {
    if (out_n) {
        *out_n = 0;
    }
    if (!c || !c0 || !c1 || !c2 || n0 <= 0) {
        return nullptr;
    }
    // Shape check: n_k · vq_strides[k] = T_q (= T_super · vq_strides[0]).
    // With vq_strides [4,2,1], n0 = T_super, n1 = 2·T_super, n2 = 4·T_super.
    const int T_q = n0 * (int)c->hp.vq_strides[0];
    if (n1 * (int)c->hp.vq_strides[1] != T_q || n2 * (int)c->hp.vq_strides[2] != T_q ||
        n0 * (int)c->hp.vq_strides[0] != T_q) {
        fprintf(stderr,
                "orpheus_snac: bad code lengths n0=%d n1=%d n2=%d (each n_k must satisfy n_k·vq_strides[k] "
                "= T_q for vq_strides=[%u,%u,%u])\n",
                n0, n1, n2, c->hp.vq_strides[0], c->hp.vq_strides[1], c->hp.vq_strides[2]);
        return nullptr;
    }
    const int T_super = n0;

    // Compute meta scratch — sized generously; ggml will tell us if too small.
    if (c->compute_meta.empty()) {
        c->compute_meta.resize(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 8 +
                               ggml_graph_overhead_custom(8192, false));
    }
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) {
        return nullptr;
    }

    ggml_cgraph* gf = build_decode_graph(c, ctx0, T_super);

    // Allocate + compute on the chosen backend.
    ggml_backend_t backend = c->backend ? c->backend : c->backend_cpu;
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc) {
        ggml_free(ctx0);
        return nullptr;
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "snac_codes_0_in"), c0, 0, (size_t)n0 * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "snac_codes_1_in"), c1, 0, (size_t)n1 * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "snac_codes_2_in"), c2, 0, (size_t)n2 * sizeof(int32_t));

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "orpheus_snac: graph compute failed (status=%d)\n", (int)st);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, stage_name);
    if (!out) {
        fprintf(stderr, "orpheus_snac: stage '%s' not in graph\n", stage_name);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }
    const size_t n = ggml_nelements(out);
    float* buf = (float*)std::malloc(n * sizeof(float));
    if (!buf) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }
    ggml_backend_tensor_get(out, buf, 0, n * sizeof(float));
    if (out_n) {
        *out_n = (int)n;
    }
    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return buf;
}

} // namespace

// ===========================================================================
// Public C ABI
// ===========================================================================

extern "C" struct snac_decoder_params snac_decoder_default_params(void) {
    snac_decoder_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    return p;
}

extern "C" struct snac_decoder_ctx* snac_decoder_init_from_file(const char* path, struct snac_decoder_params params) {
    auto* c = new snac_decoder_ctx();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Pass 1: metadata.
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g) {
            delete c;
            return nullptr;
        }
        load_metadata(c, g);
        core_gguf::free_metadata(g);
    }

    // Backend.
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "orpheus_snac: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(c->backend_cpu, c->n_threads);
    c->backend = params.use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend) {
        c->backend = c->backend_cpu;
    }

    // Pass 2: weights.
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c->backend, "orpheus_snac", wl)) {
        fprintf(stderr, "orpheus_snac: failed to load weights from '%s'\n", path);
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!bind_tensors(c)) {
        fprintf(stderr, "orpheus_snac: tensor binding failed\n");
        delete c;
        return nullptr;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr,
                "orpheus_snac: sample_rate=%u  n_codebooks=%u  codebook=%u×%u  latent_dim=%u  decoder_dim=%u  hop=%u\n",
                c->hp.sample_rate, c->hp.n_codebooks, c->hp.codebook_size, c->hp.codebook_dim, c->hp.latent_dim,
                c->hp.decoder_dim, c->hp.hop_length);
    }
    return c;
}

extern "C" void snac_decoder_free(struct snac_decoder_ctx* ctx) {
    delete ctx;
}

extern "C" uint32_t snac_decoder_sample_rate(const struct snac_decoder_ctx* ctx) {
    return ctx ? ctx->hp.sample_rate : 0;
}

extern "C" uint32_t snac_decoder_n_codebooks(const struct snac_decoder_ctx* ctx) {
    return ctx ? ctx->hp.n_codebooks : 0;
}

extern "C" uint32_t snac_decoder_hop_length(const struct snac_decoder_ctx* ctx) {
    return ctx ? ctx->hp.hop_length : 0;
}

extern "C" void snac_decoder_vq_strides(const struct snac_decoder_ctx* ctx, uint32_t out[3]) {
    if (!ctx || !out) {
        return;
    }
    for (int i = 0; i < 3 && i < (int)ctx->hp.vq_strides.size(); i++) {
        out[i] = ctx->hp.vq_strides[i];
    }
}

extern "C" float* snac_decoder_decode(struct snac_decoder_ctx* ctx, const int32_t* c0, int n0, const int32_t* c1,
                                      int n1, const int32_t* c2, int n2, int* out_n_samples) {
    return run_graph_and_extract(ctx, c0, n0, c1, n1, c2, n2, "snac_pcm", out_n_samples);
}

extern "C" float* snac_decoder_extract_stage(struct snac_decoder_ctx* ctx, const int32_t* c0, int n0, const int32_t* c1,
                                             int n1, const int32_t* c2, int n2, const char* stage_name, int* out_n) {
    return run_graph_and_extract(ctx, c0, n0, c1, n1, c2, n2, stage_name, out_n);
}
