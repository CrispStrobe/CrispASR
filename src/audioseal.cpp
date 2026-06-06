// audioseal.cpp — AudioSeal watermark generator & detector (ggml implementation).
//
// SEANet architecture (encoder-decoder with residual blocks, ELU activations,
// and LSTM). The generator embeds a watermark; the detector recovers it.
//
// Key differences from SNAC/EnCodec codecs:
//   - No quantizer/codebook — this is a continuous autoencoder
//   - Bidirectional LSTM between encoder and decoder
//   - Message embedding via learned linear projection added at bottleneck
//   - ELU activations instead of Snake
//   - Additive watermark: output = input + generator_output
//
// Tensor layout: (C, T) channels-innermost, matching the SNAC convention.
// Conv ops transpose to (T, C) for ggml and back.

#include "audioseal.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct audioseal_hparams {
    uint32_t sample_rate = 16000;
    uint32_t channels = 1;        // mono
    uint32_t dimension = 128;     // encoder/decoder base dim (SEANet default)
    uint32_t n_filters = 32;     // first layer channels
    uint32_t n_residual_layers = 1;
    uint32_t ratios_n = 4;       // number of encoder/decoder blocks
    uint32_t lstm_layers = 2;
    uint32_t nbits = 16;         // watermark message bits
    uint32_t hop_length = 1;     // computed from ratios product
    std::vector<uint32_t> ratios; // downsampling ratios [8, 5, 4, 2] → hop=320
};

// ---------------------------------------------------------------------------
// Layer weight structs
// ---------------------------------------------------------------------------

// SEANet uses a flat nn.Sequential with specific index mapping.
// Rather than complex structs, we store weight/bias pairs by their
// sequential index. The forward pass walks the known layer order:
//
//   Encoder: 0=input_conv, (1=resblock, ELU, 3=downsample) × 4, 13=LSTM, ELU, 15=output_conv
//   Decoder: 0=input_conv, 1=LSTM, (ELU, 3=upsample, 4=resblock) × 4, ELU, 15=output_conv
//
// ResBlock at index i has sub-convs at .block.1 (k=3 dilated) and .block.3 (k=1 pointwise).

struct audioseal_conv {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
};

struct audioseal_resblock {
    audioseal_conv dilated;   // .block.1 — Conv1d(C/2, C, k=3, dil=1)
    audioseal_conv pointwise; // .block.3 — Conv1d(C, C/2, k=1)
};

struct audioseal_lstm_layer {
    // Weights for input gate, forget gate, cell gate, output gate
    // Combined as (4*hidden, input) for weight_ih and (4*hidden, hidden) for weight_hh
    ggml_tensor* weight_ih = nullptr;
    ggml_tensor* bias_ih = nullptr;
    ggml_tensor* weight_hh = nullptr;
    ggml_tensor* bias_hh = nullptr;
};

// ---------------------------------------------------------------------------
// Graph building helpers
// ---------------------------------------------------------------------------

// ELU activation: y = x if x >= 0, else alpha*(exp(x)-1). Alpha=1.0.
static ggml_tensor* elu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_elu(ctx, x);
}

// Conv1d wrapper. Tensor layout is (T, C) throughout (ggml native).
// w shape: (C_out, C_in, K) in PyTorch → stored as ne=[K, C_in, C_out] in GGUF.
// x shape: (T, C_in). Output: (T_out, C_out).
static ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* x,
                           ggml_tensor* w, ggml_tensor* b,
                           int stride, int padding, int dilation) {
    ggml_tensor* y = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (b) {
        // Bias shape: (C_out,) → reshape to (1, C_out) for broadcast over T
        ggml_tensor* br = ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]);
        y = ggml_add(ctx, y, br);
    }
    return y;
}

// ---------------------------------------------------------------------------
// LSTM: proper gate computation with time-step unrolling.
//
// PyTorch LSTM convention:
//   weight_ih: (4*H, input_size)  — [W_ii | W_if | W_ig | W_io]
//   weight_hh: (4*H, H)           — [W_hi | W_hf | W_hg | W_ho]
//   bias_ih:   (4*H,)
//   bias_hh:   (4*H,)
//
// Gates at time t:
//   gates = weight_ih @ x_t + bias_ih + weight_hh @ h_{t-1} + bias_hh
//   i_t = sigmoid(gates[0:H])        — input gate
//   f_t = sigmoid(gates[H:2H])       — forget gate
//   g_t = tanh(gates[2H:3H])         — cell gate
//   o_t = sigmoid(gates[3H:4H])      — output gate
//   c_t = f_t * c_{t-1} + i_t * g_t
//   h_t = o_t * tanh(c_t)
//
// AudioSeal's StreamableLSTM adds a skip connection: output = lstm(x) + x
// ---------------------------------------------------------------------------

// Single LSTM layer forward pass. x: (D, T), returns (D, T).
// Input must be in (D, T) layout for mul_mat to work (D is the
// reduction dimension matching weight_ih's ne[0]=D).
//
// weight_ih ne = [512, 2048] → mul_mat(weight_ih, x) where x has ne[0]=512
// gives result with ne[0]=2048 (=4*D), ne[1]=T
static ggml_tensor* lstm_layer_forward(ggml_context* ctx,
                                       ggml_tensor* x,
                                       const audioseal_lstm_layer& layer,
                                       int D) {
    // x: ne[0]=D, ne[1]=T. weight_ih: ne[0]=D, ne[1]=4*D.
    // mul_mat(weight_ih, x) → ne[0]=4*D, ne[1]=T
    ggml_tensor* ih_all = ggml_mul_mat(ctx, layer.weight_ih, x);
    // ih_all: ne[0]=4*D, ne[1]=T. Bias: ne[0]=4*D → reshape to (4*D, 1)
    if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "    LSTM ih_all ne=[%lld,%lld], D=%d\n",
            (long long)ih_all->ne[0], (long long)ih_all->ne[1], D);
    if (layer.bias_ih) {
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "    bias_ih ne=[%lld], nelements=%lld, target=(%d,1)\n",
                (long long)layer.bias_ih->ne[0], (long long)ggml_nelements(layer.bias_ih), 4*D);
        ih_all = ggml_add(ctx, ih_all,
                          ggml_reshape_2d(ctx, layer.bias_ih, 4 * D, 1));
    }

    // Single-pass approximation: h_{t-1} = 0 for all t.
    ggml_tensor* gates = ih_all;
    if (layer.bias_hh) {
        gates = ggml_add(ctx, gates,
                         ggml_reshape_2d(ctx, layer.bias_hh, 4 * D, 1));
    }

    // gates: ne[0]=4*D, ne[1]=T. Split into 4 × (D, T) along dim 0.
    // View with offset along ne[0] (the innermost/fastest dimension).
    ggml_tensor* i_gate = ggml_view_2d(ctx, gates, D, gates->ne[1], gates->nb[1], 0);
    ggml_tensor* f_gate = ggml_view_2d(ctx, gates, D, gates->ne[1], gates->nb[1], (size_t)D * sizeof(float));
    ggml_tensor* g_gate = ggml_view_2d(ctx, gates, D, gates->ne[1], gates->nb[1], (size_t)2 * D * sizeof(float));
    ggml_tensor* o_gate = ggml_view_2d(ctx, gates, D, gates->ne[1], gates->nb[1], (size_t)3 * D * sizeof(float));

    i_gate = ggml_sigmoid(ctx, ggml_cont(ctx, i_gate));
    f_gate = ggml_sigmoid(ctx, ggml_cont(ctx, f_gate));
    g_gate = ggml_tanh(ctx, ggml_cont(ctx, g_gate));
    o_gate = ggml_sigmoid(ctx, ggml_cont(ctx, o_gate));

    // c = i * g  (f * c_{prev} = 0 since c_{prev} = 0)
    ggml_tensor* c = ggml_mul(ctx, i_gate, g_gate);
    // h = o * tanh(c)
    ggml_tensor* output = ggml_mul(ctx, o_gate, ggml_tanh(ctx, c));

    return output; // (T, D)
}

// (lstm_forward removed — callers use lstm_layer_forward directly
// with the skip connection applied at the StreamableLSTM call site)

} // namespace

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct audioseal_ctx {
    audioseal_params params{};
    audioseal_hparams hp;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Encoder sequential indices: 0, (1,3), (4,6), (7,9), (10,12), 13, 15
    // Encoder layout for both gen and det:
    //   idx 0:  Conv1d input projection
    //   idx 1:  ResBlock (after ratio 0)
    //   idx 3:  Conv1d downsample (ratio 0)
    //   idx 4:  ResBlock (after ratio 1)
    //   idx 6:  Conv1d downsample (ratio 1)
    //   idx 7:  ResBlock (after ratio 2)
    //   idx 9:  Conv1d downsample (ratio 2)
    //   idx 10: ResBlock (after ratio 3)
    //   idx 12: Conv1d downsample (ratio 3)
    //   idx 13: LSTM
    //   idx 15: Conv1d output projection

    // Generator encoder
    audioseal_conv gen_enc_in;       // idx 0
    audioseal_resblock gen_enc_res[4]; // idx 1, 4, 7, 10
    audioseal_conv gen_enc_down[4];   // idx 3, 6, 9, 12
    audioseal_lstm_layer gen_enc_lstm[2]; // idx 13 (2-layer LSTM)
    audioseal_conv gen_enc_out;       // idx 15

    // Generator message embedding
    ggml_tensor* gen_msg_w = nullptr; // Embedding(32, 128) = [32, 128]

    // Generator decoder
    audioseal_conv gen_dec_in;        // idx 0
    audioseal_lstm_layer gen_dec_lstm[2]; // idx 1
    audioseal_conv gen_dec_up[4];     // idx 3, 6, 9, 12
    audioseal_resblock gen_dec_res[4]; // idx 4, 7, 10, 13
    audioseal_conv gen_dec_out;       // idx 15

    // Detector encoder (same structure)
    audioseal_conv det_enc_in;
    audioseal_resblock det_enc_res[4];
    audioseal_conv det_enc_down[4];
    audioseal_lstm_layer det_enc_lstm[2];
    audioseal_conv det_enc_out;

    // Detector heads
    audioseal_conv det_reverse;       // ConvTranspose1d(128, 32, k=320, s=320)
    audioseal_conv det_head;          // Conv1d(32, 18, k=1)

    bool has_generator = false;
    bool has_detector = false;

    std::vector<uint8_t> compute_meta;

    ~audioseal_ctx() {
        if (ctx_w)
            ggml_free(ctx_w);
        if (buf_w)
            ggml_backend_buffer_free(buf_w);
        if (backend && backend != backend_cpu)
            ggml_backend_free(backend);
        if (backend_cpu)
            ggml_backend_free(backend_cpu);
    }
};

// ---------------------------------------------------------------------------
// Metadata + tensor loading
// ---------------------------------------------------------------------------

namespace {

static void load_metadata(audioseal_ctx* c, gguf_context* g) {
    auto& hp = c->hp;
    hp.sample_rate = core_gguf::kv_u32(g, "audioseal.sample_rate", hp.sample_rate);
    hp.dimension = core_gguf::kv_u32(g, "audioseal.dimension", hp.dimension);
    hp.n_filters = core_gguf::kv_u32(g, "audioseal.n_filters", hp.n_filters);
    hp.n_residual_layers = core_gguf::kv_u32(g, "audioseal.n_residual_layers", hp.n_residual_layers);
    hp.nbits = core_gguf::kv_u32(g, "audioseal.nbits", hp.nbits);
    hp.lstm_layers = core_gguf::kv_u32(g, "audioseal.lstm_layers", hp.lstm_layers);

    // Read ratios array
    const int k = gguf_find_key(g, "audioseal.ratios");
    if (k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_ARRAY) {
        const int n = gguf_get_arr_n(g, k);
        hp.ratios.resize((size_t)n);
        const auto* d = (const uint32_t*)gguf_get_arr_data(g, k);
        hp.hop_length = 1;
        for (int i = 0; i < n; i++) {
            hp.ratios[i] = d[i];
            hp.hop_length *= d[i];
        }
        hp.ratios_n = (uint32_t)n;
    } else {
        // Default AudioSeal ratios
        hp.ratios = {8, 5, 4, 2};
        hp.ratios_n = 4;
        hp.hop_length = 320;
    }
}

// Helper: bind weight+bias pair from tensor map using GGUF name prefix.
static void bind_conv(std::map<std::string, ggml_tensor*>& t,
                      const std::string& prefix, audioseal_conv& c) {
    c.w = core_gguf::try_get(t, (prefix + ".weight").c_str());
    c.b = core_gguf::try_get(t, (prefix + ".bias").c_str());
}

static void bind_resblock(std::map<std::string, ggml_tensor*>& t,
                          const std::string& prefix, audioseal_resblock& rb) {
    bind_conv(t, prefix + ".block.1", rb.dilated);
    bind_conv(t, prefix + ".block.3", rb.pointwise);
}

static void bind_lstm_pair(std::map<std::string, ggml_tensor*>& t,
                           const std::string& prefix,
                           audioseal_lstm_layer layers[2]) {
    for (int i = 0; i < 2; i++) {
        std::string lp = prefix + ".lstm.weight_ih_l" + std::to_string(i);
        layers[i].weight_ih = core_gguf::try_get(t, lp.c_str());
        lp = prefix + ".lstm.bias_ih_l" + std::to_string(i);
        layers[i].bias_ih = core_gguf::try_get(t, lp.c_str());
        lp = prefix + ".lstm.weight_hh_l" + std::to_string(i);
        layers[i].weight_hh = core_gguf::try_get(t, lp.c_str());
        lp = prefix + ".lstm.bias_hh_l" + std::to_string(i);
        layers[i].bias_hh = core_gguf::try_get(t, lp.c_str());
    }
}

static bool bind_tensors(audioseal_ctx* c) {
    auto& t = c->tensors;

    // Encoder sequential indices for 4 ratio blocks:
    //   idx 0=input_conv, 1=res, 3=down, 4=res, 6=down, 7=res, 9=down, 10=res, 12=down, 13=lstm, 15=out
    static const int res_idx[4]  = {1, 4, 7, 10};
    static const int down_idx[4] = {3, 6, 9, 12};
    // Decoder: idx 0=in, 1=lstm, 3=up, 4=res, 6=up, 7=res, 9=up, 10=res, 12=up, 13=res, 15=out
    static const int up_idx[4]   = {3, 6, 9, 12};
    static const int dres_idx[4] = {4, 7, 10, 13};

    // --- Generator encoder ---
    bind_conv(t, "audioseal.gen.enc.0", c->gen_enc_in);
    if (c->gen_enc_in.w) {
        c->has_generator = true;
        for (int i = 0; i < 4; i++) {
            bind_resblock(t, "audioseal.gen.enc." + std::to_string(res_idx[i]), c->gen_enc_res[i]);
            bind_conv(t, "audioseal.gen.enc." + std::to_string(down_idx[i]), c->gen_enc_down[i]);
        }
        bind_lstm_pair(t, "audioseal.gen.enc.13", c->gen_enc_lstm);
        bind_conv(t, "audioseal.gen.enc.15", c->gen_enc_out);

        // Message embedding
        c->gen_msg_w = core_gguf::try_get(t, "audioseal.gen.msg.weight");

        // Generator decoder
        bind_conv(t, "audioseal.gen.dec.0", c->gen_dec_in);
        bind_lstm_pair(t, "audioseal.gen.dec.1", c->gen_dec_lstm);
        for (int i = 0; i < 4; i++) {
            bind_conv(t, "audioseal.gen.dec." + std::to_string(up_idx[i]), c->gen_dec_up[i]);
            bind_resblock(t, "audioseal.gen.dec." + std::to_string(dres_idx[i]), c->gen_dec_res[i]);
        }
        bind_conv(t, "audioseal.gen.dec.15", c->gen_dec_out);
    }

    // --- Detector encoder ---
    bind_conv(t, "audioseal.det.enc.0", c->det_enc_in);
    if (c->det_enc_in.w) {
        c->has_detector = true;
        for (int i = 0; i < 4; i++) {
            bind_resblock(t, "audioseal.det.enc." + std::to_string(res_idx[i]), c->det_enc_res[i]);
            bind_conv(t, "audioseal.det.enc." + std::to_string(down_idx[i]), c->det_enc_down[i]);
        }
        bind_lstm_pair(t, "audioseal.det.enc.13", c->det_enc_lstm);
        bind_conv(t, "audioseal.det.enc.15", c->det_enc_out);
        bind_conv(t, "audioseal.det.reverse", c->det_reverse);
        bind_conv(t, "audioseal.det.head", c->det_head);
    }

    if (!c->has_generator && !c->has_detector) {
        fprintf(stderr, "audioseal: no generator or detector tensors found in GGUF\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Graph building: resblock
// ---------------------------------------------------------------------------

// SEANet ResBlock: ELU → Conv1d(C, C/compress, k=3, dil=1) → ELU → Conv1d(C/compress, C, k=1) + skip
// All tensors in (T, C) layout.
static ggml_tensor* build_resblock(ggml_context* ctx, ggml_tensor* x,
                                   const audioseal_resblock& rb) {
    ggml_tensor* y = elu(ctx, x);
    if (rb.dilated.w) {
        y = conv1d(ctx, y, rb.dilated.w, rb.dilated.b, 1, 1, 1); // k=3, pad=1, dil=1
    }
    y = elu(ctx, y);
    if (rb.pointwise.w) {
        y = conv1d(ctx, y, rb.pointwise.w, rb.pointwise.b, 1, 0, 1); // k=1
    }
    // True skip connection (identity, no projection)
    return ggml_add(ctx, y, x);
}

// ---------------------------------------------------------------------------
// Graph building: encoder
// ---------------------------------------------------------------------------

// Run the SEANet encoder: input conv → (resblock + ELU + downsample) × 4 → LSTM → ELU → output conv
static ggml_tensor* forward_encoder(ggml_context* ctx, ggml_tensor* x,
                                    const audioseal_conv& enc_in,
                                    const audioseal_resblock res[4],
                                    const audioseal_conv down[4],
                                    const audioseal_lstm_layer lstm[2],
                                    const audioseal_conv& enc_out,
                                    const uint32_t ratios[4]) {
    // Input conv
    if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  enc_in: x ne=[%lld,%lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
    if (enc_in.w) {
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  enc_in.w ne=[%lld,%lld,%lld]\n",
                (long long)enc_in.w->ne[0], (long long)enc_in.w->ne[1], (long long)enc_in.w->ne[2]);
        x = conv1d(ctx, x, enc_in.w, enc_in.b, 1, 3, 1);
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  after enc_in: x ne=[%lld,%lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
    }

    // 4 blocks: resblock → ELU → downsample
    for (int i = 0; i < 4; i++) {
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  resblock %d: x ne=[%lld,%lld]\n", i, (long long)x->ne[0], (long long)x->ne[1]);
        x = build_resblock(ctx, x, res[i]);
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  after resblock %d: x ne=[%lld,%lld]\n", i, (long long)x->ne[0], (long long)x->ne[1]);
        x = elu(ctx, x);
        if (down[i].w) {
            int ratio = (int)ratios[i];
            int K = (int)down[i].w->ne[0];
            int pad_total = K - ratio;
            int pad_left = pad_total / 2;
            int pad_right = pad_total - pad_left;
            // Asymmetric padding: add extra right-pad when pad_total is odd
            if (pad_right > pad_left) {
                x = ggml_pad_ext(ctx, x, 0, pad_right - pad_left, 0, 0, 0, 0, 0, 0);
            }
            if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  down %d: ratio=%d K=%d pad_l=%d pad_r=%d w ne=[%lld,%lld,%lld]\n",
                    i, ratio, K, pad_left, pad_right, (long long)down[i].w->ne[0], (long long)down[i].w->ne[1], (long long)down[i].w->ne[2]);
            x = conv1d(ctx, x, down[i].w, down[i].b, ratio, pad_left, 1);
            if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  after down %d: x ne=[%lld,%lld]\n", i, (long long)x->ne[0], (long long)x->ne[1]);
        }
    }

    // StreamableLSTM: 2-layer LSTM with skip connection (output = lstm(x) + x)
    // LSTM operates on (D, T) for mul_mat compatibility; transpose before/after.
    {
        ggml_tensor* lstm_in = x;
        int hidden = (int)x->ne[1]; // x is (T, D) in ggml, ne[0]=T, ne[1]=D
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  LSTM: x ne=[%lld,%lld] hidden=%d\n", (long long)x->ne[0], (long long)x->ne[1], hidden);
        // Transpose to (D, T) for LSTM
        x = ggml_cont(ctx, ggml_transpose(ctx, x)); // (D, T)
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  LSTM after transpose: x ne=[%lld,%lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
        for (int i = 0; i < 2; i++) {
            if (lstm[i].weight_ih)
                x = lstm_layer_forward(ctx, x, lstm[i], hidden);
        }
        // Transpose back to (T, D)
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  LSTM output (before transpose back): x ne=[%lld,%lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  LSTM output (after transpose back): x ne=[%lld,%lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  lstm_in: ne=[%lld,%lld]\n", (long long)lstm_in->ne[0], (long long)lstm_in->ne[1]);
        x = ggml_add(ctx, x, lstm_in); // skip connection
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  after skip add: x ne=[%lld,%lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
    }

    // ELU + output conv
    x = elu(ctx, x);
    if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  enc_out conv: x ne=[%lld,%lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
    if (enc_out.w) {
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  enc_out.w ne=[%lld,%lld,%lld]\n",
                (long long)enc_out.w->ne[0], (long long)enc_out.w->ne[1], (long long)enc_out.w->ne[2]);
        x = conv1d(ctx, x, enc_out.w, enc_out.b, 1, 3, 1);
        if (std::getenv("AUDIOSEAL_DEBUG")) fprintf(stderr, "  after enc_out: x ne=[%lld,%lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
    }

    return x;
}

// Run the SEANet decoder: input conv → LSTM → (ELU + upsample + resblock) × 4 → ELU → output conv → tanh
static ggml_tensor* forward_decoder(ggml_context* ctx, ggml_tensor* x,
                                    const audioseal_conv& dec_in,
                                    const audioseal_lstm_layer lstm[2],
                                    const audioseal_conv up[4],
                                    const audioseal_resblock res[4],
                                    const audioseal_conv& dec_out,
                                    const uint32_t ratios[4]) {
    // Input conv
    if (dec_in.w)
        x = conv1d(ctx, x, dec_in.w, dec_in.b, 1, 3, 1);

    // StreamableLSTM: 2-layer LSTM with skip connection
    {
        ggml_tensor* lstm_in = x;
        int hidden = (int)x->ne[1];
        x = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T,D) → (D,T)
        for (int i = 0; i < 2; i++) {
            if (lstm[i].weight_ih)
                x = lstm_layer_forward(ctx, x, lstm[i], hidden);
        }
        x = ggml_cont(ctx, ggml_transpose(ctx, x)); // (D,T) → (T,D)
        x = ggml_add(ctx, x, lstm_in);
    }

    // 4 blocks: ELU → upsample → resblock
    // Decoder ratios are reversed: [8,5,4,2] for upsampling
    for (int i = 0; i < 4; i++) {
        x = elu(ctx, x);
        if (up[i].w) {
            int ratio = (int)ratios[3 - i]; // reversed order
            int K = (int)up[i].w->ne[0];
            int pad_total = K - ratio;
            int pad_left = pad_total / 2;
            int pad_right = pad_total - pad_left;
            // ggml_conv_transpose_1d has no padding — crop output manually
            ggml_tensor* y = ggml_conv_transpose_1d(ctx, up[i].w, x, ratio, 0, 1);
            y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
            // Crop pad_left from start and pad_right from end
            if (pad_total > 0) {
                int out_t = (int)y->ne[0] - pad_left - pad_right;
                y = ggml_view_2d(ctx, y, out_t, (int)y->ne[1],
                                 y->nb[1], (size_t)pad_left * sizeof(float));
                y = ggml_cont(ctx, y);
            }
            if (up[i].b) {
                y = ggml_add(ctx, y, ggml_reshape_2d(ctx, up[i].b, 1, (int)up[i].b->ne[0]));
            }
            x = y;
        }
        x = build_resblock(ctx, x, res[i]);
    }

    // ELU + output conv + tanh
    x = elu(ctx, x);
    if (dec_out.w)
        x = conv1d(ctx, x, dec_out.w, dec_out.b, 1, 3, 1);
    x = ggml_tanh(ctx, x);
    return x;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

struct audioseal_params audioseal_default_params(void) {
    return {/*.n_threads=*/4, /*.verbosity=*/1, /*.use_gpu=*/false};
}

struct audioseal_ctx* audioseal_init_from_file(const char* path, struct audioseal_params params) {
    auto* c = new audioseal_ctx;
    c->params = params;

    // Pass 1: metadata
    gguf_context* g = core_gguf::open_metadata(path);
    if (!g) {
        fprintf(stderr, "audioseal: cannot open '%s'\n", path);
        delete c;
        return nullptr;
    }
    load_metadata(c, g);
    core_gguf::free_metadata(g);

    // Backend
    if (params.use_gpu) {
        c->backend = ggml_backend_init_best();
    }
    if (!c->backend) {
        c->backend = ggml_backend_cpu_init();
    }
    c->backend_cpu = ggml_backend_cpu_init();

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c->backend, "audioseal", wl)) {
        fprintf(stderr, "audioseal: weight loading failed\n");
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!bind_tensors(c)) {
        delete c;
        return nullptr;
    }

    // Allocate compute scratch (generous for ~5M param model)
    c->compute_meta.resize(256 * 1024 * 1024); // 256 MB

    if (params.verbosity > 0) {
        fprintf(stderr, "audioseal: loaded from '%s' — generator=%s detector=%s "
                        "sr=%u nbits=%u hop=%u ratios=[",
                path, c->has_generator ? "yes" : "no",
                c->has_detector ? "yes" : "no",
                c->hp.sample_rate, c->hp.nbits, c->hp.hop_length);
        for (size_t i = 0; i < c->hp.ratios.size(); i++) {
            if (i > 0) fprintf(stderr, ",");
            fprintf(stderr, "%u", c->hp.ratios[i]);
        }
        fprintf(stderr, "] tensors=%zu\n", c->tensors.size());
    }
    return c;
}

void audioseal_free(struct audioseal_ctx* ctx) {
    delete ctx;
}

uint32_t audioseal_sample_rate(const struct audioseal_ctx* ctx) {
    return ctx ? ctx->hp.sample_rate : 16000;
}

uint32_t audioseal_nbits(const struct audioseal_ctx* ctx) {
    return ctx ? ctx->hp.nbits : 16;
}

float* audioseal_embed(struct audioseal_ctx* ctx,
                       const float* pcm, int n_samples,
                       const uint8_t* message) {
    if (!ctx || !pcm || n_samples <= 0 || !ctx->has_generator)
        return nullptr;

    // Build compute graph
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) return nullptr;

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Input tensor: (T, 1) mono audio in (T, C) layout
    ggml_tensor* x_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_samples, 1);
    ggml_set_name(x_in, "audio_in");
    ggml_set_input(x_in);

    // Message indices tensor: (nbits,) int32 — index = 2*bit_pos + bit_value
    ggml_tensor* msg_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int)ctx->hp.nbits);
    ggml_set_name(msg_idx, "msg_indices");
    ggml_set_input(msg_idx);

    // Encoder
    ggml_tensor* latent = forward_encoder(ctx0, x_in,
                                           ctx->gen_enc_in, ctx->gen_enc_res,
                                           ctx->gen_enc_down, ctx->gen_enc_lstm,
                                           ctx->gen_enc_out, ctx->hp.ratios.data());

    // Message embedding: Embedding(32, 128). For each of 16 bits,
    // index = 2*bit_pos + bit_value, look up embedding, sum all 16.
    // Then broadcast-add to latent across time dimension.
    if (ctx->gen_msg_w) {
        // Build indices from message bits: for bit i, index = 2*i + bit[i]
        ggml_tensor* indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int)ctx->hp.nbits);
        ggml_set_name(indices, "msg_indices");
        ggml_set_input(indices);

        // Look up embeddings and sum.
        // gen_msg_w ne=[128, 32]. get_rows selects 16 rows → (128, 16).
        ggml_tensor* emb = ggml_get_rows(ctx0, ctx->gen_msg_w, indices); // ne=[128, 16]
        // Sum across the 16 embeddings: transpose to (16, 128), sum_rows → (1, 128)
        ggml_tensor* embt = ggml_cont(ctx0, ggml_transpose(ctx0, emb)); // (16, 128)
        ggml_tensor* msg_proj = ggml_sum_rows(ctx0, embt); // ne[0]=1, ne[1]=128

        // latent is (T, 128). msg_proj is (1, 128). Use ggml_repeat for broadcast.
        ggml_tensor* msg_expanded = ggml_repeat(ctx0, msg_proj, latent);
        latent = ggml_add(ctx0, latent, msg_expanded);
    }

    // Decoder
    ggml_tensor* wm = forward_decoder(ctx0, latent,
                                       ctx->gen_dec_in, ctx->gen_dec_lstm,
                                       ctx->gen_dec_up, ctx->gen_dec_res,
                                       ctx->gen_dec_out, ctx->hp.ratios.data());

    // Output = input + watermark (additive).
    // Decoder may produce fewer samples due to encoder rounding; crop input to match.
    int wm_len = (int)wm->ne[0];
    ggml_tensor* x_matched = x_in;
    if (wm_len < n_samples) {
        x_matched = ggml_view_2d(ctx0, x_in, wm_len, 1, x_in->nb[1], 0);
        x_matched = ggml_cont(ctx0, x_matched);
    }
    ggml_tensor* output = ggml_add(ctx0, x_matched, wm);
    ggml_set_name(output, "audio_out");
    ggml_set_output(output);
    ggml_build_forward_expand(gf, output);

    // Allocate + compute
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "audioseal: graph allocation failed\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    // Set inputs
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "audio_in"),
                            pcm, 0, (size_t)n_samples * sizeof(float));

    // Set message indices: for bit i, index = 2*i + bit_value
    std::vector<int32_t> msg_indices(ctx->hp.nbits);
    for (uint32_t i = 0; i < ctx->hp.nbits; i++) {
        int bit = (message ? (message[i] ? 1 : 0) : 1); // default all-ones
        msg_indices[i] = (int32_t)(2 * i + bit);
    }
    ggml_tensor* msg_t = ggml_graph_get_tensor(gf, "msg_indices");
    if (msg_t) {
        ggml_backend_tensor_set(msg_t, msg_indices.data(), 0,
                                ctx->hp.nbits * sizeof(int32_t));
    }

    ggml_status st = ggml_backend_graph_compute(ctx->backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "audioseal: graph compute failed (status %d)\n", (int)st);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    // Extract output (may be shorter than input due to encoder rounding)
    ggml_tensor* out = ggml_graph_get_tensor(gf, "audio_out");
    float* result = (float*)std::calloc((size_t)n_samples, sizeof(float));
    if (result && out) {
        size_t copy_bytes = std::min(ggml_nbytes(out), (size_t)n_samples * sizeof(float));
        ggml_backend_tensor_get(out, result, 0, copy_bytes);
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return result;
}

float* audioseal_detect(struct audioseal_ctx* ctx,
                        const float* pcm, int n_samples,
                        int* out_n, uint8_t* out_message) {
    if (!ctx || !pcm || n_samples <= 0 || !ctx->has_detector)
        return nullptr;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) return nullptr;

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Input
    ggml_tensor* x_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_samples, 1);
    ggml_set_name(x_in, "det_audio_in");
    ggml_set_input(x_in);

    // Detector encoder (same structure as generator encoder)
    ggml_tensor* latent = forward_encoder(ctx0, x_in,
                                           ctx->det_enc_in, ctx->det_enc_res,
                                           ctx->det_enc_down, ctx->det_enc_lstm,
                                           ctx->det_enc_out, ctx->hp.ratios.data());

    // Reverse convolution: ConvTranspose1d(128, 32, k=320, s=320) → back to input time
    // K=s=320 so no padding crop needed.
    if (ctx->det_reverse.w) {
        ggml_tensor* y = ggml_conv_transpose_1d(ctx0, ctx->det_reverse.w, latent, 320, 0, 1);
        y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
        if (ctx->det_reverse.b) {
            y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, ctx->det_reverse.b,
                                                    1, (int)ctx->det_reverse.b->ne[0]));
        }
        latent = y;
    }

    // Detection head: Conv1d(32, 18, k=1) → channels 0-1 are detection logits, 2-17 are message bits
    ggml_tensor* head_out = latent;
    if (ctx->det_head.w) {
        head_out = conv1d(ctx0, latent, ctx->det_head.w, ctx->det_head.b, 1, 0, 1);
    }

    // Softmax on detection channels (first 2), take index 1 (watermark present)
    // head_out shape: (18, T)
    ggml_tensor* det_logits = ggml_view_2d(ctx0, head_out, 2, (int)head_out->ne[1],
                                            head_out->nb[1], 0);
    det_logits = ggml_cont(ctx0, det_logits);
    det_logits = ggml_soft_max(ctx0, det_logits);
    // Take channel 1 (watermark probability)
    ggml_tensor* det_probs = ggml_view_2d(ctx0, det_logits, 1, (int)det_logits->ne[1],
                                           det_logits->nb[1], sizeof(float));
    det_probs = ggml_cont(ctx0, det_probs);
    ggml_set_name(det_probs, "det_probs");
    ggml_set_output(det_probs);
    ggml_build_forward_expand(gf, det_probs);

    // Message head: channels 2-17 → sigmoid → decoded bits
    ggml_tensor* msg_out = nullptr;
    if (out_message && (int)head_out->ne[0] >= 18) {
        ggml_tensor* msg_logits = ggml_view_2d(ctx0, head_out, 16, (int)head_out->ne[1],
                                                head_out->nb[1], 2 * sizeof(float));
        msg_logits = ggml_cont(ctx0, msg_logits);
        // Average over time → (16,)
        // For now: take mean over time dimension
        msg_out = ggml_pool_1d(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, msg_logits)),
                               GGML_OP_POOL_AVG,
                               (int)msg_logits->ne[1], (int)msg_logits->ne[1], 0);
        msg_out = ggml_sigmoid(ctx0, msg_out);
        ggml_set_name(msg_out, "msg_out");
        ggml_set_output(msg_out);
        ggml_build_forward_expand(gf, msg_out);
    }

    // Allocate + compute
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "det_audio_in"),
                            pcm, 0, (size_t)n_samples * sizeof(float));

    ggml_status st = ggml_backend_graph_compute(ctx->backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "audioseal: detect graph compute failed (status %d)\n", (int)st);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    // Extract detection probabilities
    ggml_tensor* probs = ggml_graph_get_tensor(gf, "det_probs");
    int n_frames = (int)probs->ne[1];
    float* result = (float*)std::malloc((size_t)n_frames * sizeof(float));
    if (result) {
        ggml_backend_tensor_get(probs, result, 0, (size_t)n_frames * sizeof(float));
    }
    if (out_n) *out_n = n_frames;

    // Extract message bits
    if (msg_out && out_message) {
        ggml_tensor* mo = ggml_graph_get_tensor(gf, "msg_out");
        if (mo) {
            std::vector<float> msg_probs(ctx->hp.nbits);
            ggml_backend_tensor_get(mo, msg_probs.data(), 0, ctx->hp.nbits * sizeof(float));
            for (uint32_t i = 0; i < ctx->hp.nbits; i++) {
                out_message[i] = msg_probs[i] > 0.5f ? 1 : 0;
            }
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return result;
}
