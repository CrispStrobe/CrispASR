#include "sidon.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include "core/quant_bcast.h"
#include "core/cpu_ops.h"
#include "core/dac_decoder.h"
#include "core/fft.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "sidon_rpe_gates.h" // #416 follow-up: length-aware RPE formulation choice

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

struct sidon_hparams {
    int layers = 8;
    int hidden = 1024;
    int intermediate = 4096;
    int heads = 16;
    int conv_kernel = 31;
    int rel_left = 64;
    int rel_right = 8;
    int feature_dim = 160;
    int mel_bins = 80;
    int input_rate = 16000;
    int output_rate = 48000;
    float eps = 1e-5f;
    // Encoder-only GGUF (plain w2v-BERT layers, no Sidon LoRA, no DAC
    // decoder) — serves feature extraction for other backends
    // (confucius4-tts uses layer-17 hidden states as speaker conditioning).
    bool encoder_only = false;
};

struct sidon_ffn {
    ggml_tensor *norm_w = nullptr, *norm_b = nullptr;
    ggml_tensor *up_w = nullptr, *up_b = nullptr;
    ggml_tensor *down_w = nullptr, *down_b = nullptr;
};

struct sidon_layer {
    sidon_ffn ffn1, ffn2;
    ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    ggml_tensor *q_w = nullptr, *q_b = nullptr;
    ggml_tensor *k_w = nullptr, *k_b = nullptr;
    ggml_tensor *v_w = nullptr, *v_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    ggml_tensor* distance_w = nullptr;
    ggml_tensor *conv_norm_w = nullptr, *conv_norm_b = nullptr;
    ggml_tensor *conv_pw1_w = nullptr, *conv_pw2_w = nullptr;
    ggml_tensor* conv_dw_w = nullptr;
    ggml_tensor *conv_dw_norm_w = nullptr, *conv_dw_norm_b = nullptr;
    ggml_tensor *final_norm_w = nullptr, *final_norm_b = nullptr;
};

struct sidon_model {
    sidon_hparams hp;
    bool valid = true;
    ggml_tensor *feature_norm_w = nullptr, *feature_norm_b = nullptr;
    ggml_tensor *feature_proj_w = nullptr, *feature_proj_b = nullptr;
    ggml_tensor *frontend_window = nullptr, *frontend_mels = nullptr;
    std::vector<sidon_layer> layers;
    core_dac::DacWeights dac;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr, buf_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<float> window, mel_filters;
};

// Frontend framing constants, shared by make_features() and the inference
// padding so the two cannot drift. One predictor frame spans
// sidon_frontend_decim STFT hops of the log-mel front end.
static constexpr int sidon_frontend_hop = 160;
static constexpr int sidon_frontend_decim = 2;

// Relative-position-bias formulation. All three are numerically equivalent;
// they differ only in what the graph materialises. Selected by
// CRISPASR_SIDON_RPE (expand | bucket | bucket-direct).
enum class sidon_rpe_mode {
    expand,       // legacy: expand distance_w to [head_dim, T, T] before the dot product
    bucket,       // dot per bucket, head dimension of the gather index built via in-graph REPEAT
    bucket_direct // dot per bucket, gather index supplied host-side already shaped [T, T, H]
};

// Resolve the RPE formulation for a graph of THIS length. Decided per graph
// build, not once at init: the choice depends on T (see sidon_rpe_gates.h), and
// reading it at init also made CRISPASR_SIDON_RPE inert for anything that set
// it after loading the model — which is exactly how the mode-equivalence check
// in tests/test_sidon_live.cpp ended up comparing bucket-direct against itself.
static sidon_rpe_mode resolve_rpe_mode(int T, int head_dim, int heads, bool is_gpu, int verbosity) {
    namespace g = sidon_rpe_gates;
    const char* e = getenv("CRISPASR_SIDON_RPE");

    static bool warned_unknown = false;
    if (g::env_mode_is_unknown(e) && !warned_unknown) {
        warned_unknown = true;
        std::fprintf(stderr,
                     "sidon: unknown CRISPASR_SIDON_RPE='%s' (expand|bucket|bucket-direct); using bucket-direct\n", e);
    }

    const g::Resolved r = g::resolve(e, getenv("CRISPASR_SIDON_RPE_BUDGET_MB"), T, head_dim, heads, is_gpu);
    if (r.auto_expand && verbosity)
        std::fprintf(stderr, "sidon: RPE auto-selected 'expand' (T=%d, +%.1f MiB transient)\n", T,
                     (double)g::expand_extra_bytes(T, head_dim, heads) / (1024.0 * 1024.0));

    switch (r.m) {
    case g::mode::expand:
        return sidon_rpe_mode::expand;
    case g::mode::bucket:
        return sidon_rpe_mode::bucket;
    case g::mode::bucket_direct:
        break;
    }
    return sidon_rpe_mode::bucket_direct;
}

struct sidon_context {
    sidon_context_params params{};
    sidon_rpe_mode rpe_mode = sidon_rpe_mode::bucket_direct;
    // #416 bisection gate: keep the quantized relative-position multiply.
    bool quant_rpe = false;
    sidon_model model;
    core_dac::fastconv_cache decoder_fc;
    ggml_tensor* bucket_ids = nullptr;
    ggml_backend_t backend = nullptr, decoder_backend = nullptr, backend_cpu = nullptr;
    ggml_backend_sched_t predictor_sched = nullptr, decoder_sched = nullptr;
    bool predictor_vulkan = false;

    std::vector<uint8_t> predictor_meta, decoder_meta;
    ggml_context *predictor_ctx = nullptr, *decoder_ctx = nullptr;
    ggml_cgraph *predictor_graph = nullptr, *decoder_graph = nullptr;
    ggml_tensor *predictor_input = nullptr, *relative_indices = nullptr;
    ggml_tensor *predictor_output = nullptr, *decoder_input = nullptr, *decoder_output = nullptr;
};

// One-sided receptive field of the DAC decoder, expressed in LATENT frames.
//
// Derived from the architecture rather than tuned on a signal: a threshold
// picked by ear ("10 works, 9 audibly does not") encodes the property the
// almost-broken build happens to satisfy, and gives no margin when the config
// changes. Walk the decoder and accumulate each layer's one-sided radius,
// divided by the cumulative upsampling at that layer:
//
//   input Conv1d k=7            -> 3 samples at the latent rate
//   per block, ConvTranspose1d(k=2s, stride=s): each output depends on
//                                  ceil(k/s)=2 inputs -> ~1 frame at the
//                                  block's INPUT rate
//   per block, 3x ResidualUnit  -> Conv1d k=7 dilated by d -> 3*d samples at
//                                  the block's OUTPUT rate
//   output Conv1d k=7           -> 3 samples at the audio rate
//
// For Sidon's [8,5,4,3,2] / hop 960 decoder this sums to ~10.4 frames, which
// is why 10 was marginal and 9 audibly broke the joins.
static int dac_receptive_frames(const core_dac::DacWeights& dac) {
    const core_dac::DacConfig& cfg = dac.config;
    double radius = 3.0; // input Conv1d k=7
    double upsample = 1.0;
    for (int b = 0; b < cfg.n_decoder_blocks; ++b) {
        const int stride = cfg.upsampling_ratios[b];
        if (stride <= 0)
            break;
        radius += 1.0 / upsample; // ConvTranspose1d, at this block's input rate
        upsample *= stride;
        for (const int dilation : cfg.residual_dilations)
            radius += 3.0 * dilation / upsample; // ResidualUnit Conv1d k=7, dilated
    }
    radius += 3.0 / upsample; // output Conv1d k=7
    // Round up, then one frame of margin so an exactly-integral radius is still
    // strictly covered.
    return (int)std::ceil(radius) + 1;
}

static ggml_tensor* req(sidon_model& m, const std::string& name) {
    ggml_tensor* tensor = core_gguf::require(m.tensors, name.c_str(), "sidon");
    m.valid = m.valid && tensor != nullptr;
    return tensor;
}

static bool load_model(sidon_model& m, const char* path, ggml_backend_t backend) {
    gguf_context* gctx = core_gguf::open_metadata(path);
    if (!gctx)
        return false;
    const std::string architecture = core_gguf::kv_str(gctx, "general.architecture", "");
    m.hp.layers = (int)core_gguf::kv_u32(gctx, "sidon.predictor.layers", m.hp.layers);
    m.hp.hidden = (int)core_gguf::kv_u32(gctx, "sidon.hidden_size", m.hp.hidden);
    m.hp.intermediate = (int)core_gguf::kv_u32(gctx, "sidon.intermediate_size", m.hp.intermediate);
    m.hp.heads = (int)core_gguf::kv_u32(gctx, "sidon.attention_heads", m.hp.heads);
    m.hp.conv_kernel = (int)core_gguf::kv_u32(gctx, "sidon.conv_kernel", m.hp.conv_kernel);
    m.hp.rel_left = (int)core_gguf::kv_u32(gctx, "sidon.relative_left", m.hp.rel_left);
    m.hp.rel_right = (int)core_gguf::kv_u32(gctx, "sidon.relative_right", m.hp.rel_right);
    m.hp.feature_dim = (int)core_gguf::kv_u32(gctx, "sidon.feature_dim", m.hp.feature_dim);
    m.hp.mel_bins = (int)core_gguf::kv_u32(gctx, "sidon.mel_bins", m.hp.mel_bins);
    m.hp.input_rate = (int)core_gguf::kv_u32(gctx, "sidon.input_sample_rate", m.hp.input_rate);
    m.hp.output_rate = (int)core_gguf::kv_u32(gctx, "sidon.output_sample_rate", m.hp.output_rate);
    m.hp.eps = core_gguf::kv_f32(gctx, "sidon.layer_norm_eps", m.hp.eps);
    m.hp.encoder_only = core_gguf::kv_u32(gctx, "sidon.encoder_only", 0) != 0;
    const int decoder_blocks = (int)core_gguf::kv_u32(gctx, "sidon.decoder_blocks", 5);
    const int decoder_hop = (int)core_gguf::kv_u32(gctx, "sidon.decoder_hop", 960);
    const int expected_rates[5] = {8, 5, 4, 3, 2};
    int decoder_rates[5] = {8, 5, 4, 3, 2};
    bool decoder_rates_valid = true;
    for (int i = 0; i < 5; ++i) {
        const std::string key = "sidon.decoder_rate." + std::to_string(i);
        const uint32_t rate = core_gguf::kv_u32(gctx, key.c_str(), (uint32_t)decoder_rates[i]);
        decoder_rates_valid = decoder_rates_valid && rate == (uint32_t)expected_rates[i];
        decoder_rates[i] = (int)rate;
    }
    core_gguf::free_metadata(gctx);

    if (architecture != "sidon" || m.hp.layers <= 0 || m.hp.layers > 64 || m.hp.hidden <= 0 || m.hp.hidden > 16384 ||
        m.hp.intermediate <= 0 || m.hp.intermediate > 131072 || m.hp.heads <= 0 || m.hp.heads > 256 ||
        m.hp.hidden % m.hp.heads != 0 || m.hp.conv_kernel <= 0 || m.hp.conv_kernel > 1024 ||
        m.hp.conv_kernel % 2 == 0 || m.hp.rel_left < 0 || m.hp.rel_left > 4096 || m.hp.rel_right < 0 ||
        m.hp.rel_right > 4096 || m.hp.mel_bins <= 0 || m.hp.mel_bins > 1024 || m.hp.feature_dim != 2 * m.hp.mel_bins ||
        !std::isfinite(m.hp.eps) || m.hp.eps <= 0.0f || m.hp.eps > 1.0f || m.hp.input_rate != 16000 ||
        (!m.hp.encoder_only &&
         (m.hp.output_rate != 48000 || decoder_blocks != 5 || decoder_hop != 960 || !decoder_rates_valid))) {
        std::fprintf(stderr, "sidon: unsupported or invalid GGUF metadata\n");
        return false;
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "sidon", wl))
        return false;
    m.ctx = wl.ctx;
    m.buf = wl.buf;
    m.buf_cpu = wl.buf_cpu;
    m.tensors = std::move(wl.tensors);

    m.frontend_window = req(m, "frontend.window");
    m.frontend_mels = req(m, "frontend.mel_filters");
    m.feature_norm_w = req(m, "predictor.feature_projection.layer_norm.weight");
    m.feature_norm_b = req(m, "predictor.feature_projection.layer_norm.bias");
    m.feature_proj_w = req(m, "predictor.feature_projection.projection.weight");
    m.feature_proj_b = req(m, "predictor.feature_projection.projection.bias");

    m.layers.resize((size_t)m.hp.layers);
    for (int i = 0; i < m.hp.layers; ++i) {
        auto& l = m.layers[(size_t)i];
        const std::string p = "predictor.encoder.layers." + std::to_string(i) + ".";
        auto ffn = [&](sidon_ffn& f, const char* n) {
            const std::string q = p + n;
            f.norm_w = req(m, q + "_layer_norm.weight");
            f.norm_b = req(m, q + "_layer_norm.bias");
            f.up_w = req(m, q + ".intermediate_dense.weight");
            f.up_b = req(m, q + ".intermediate_dense.bias");
            f.down_w = req(m, q + ".output_dense.weight");
            f.down_b = req(m, q + ".output_dense.bias");
        };
        ffn(l.ffn1, "ffn1");
        ffn(l.ffn2, "ffn2");
        l.attn_norm_w = req(m, p + "self_attn_layer_norm.weight");
        l.attn_norm_b = req(m, p + "self_attn_layer_norm.bias");
        l.q_w = req(m, p + "self_attn.linear_q.weight");
        l.q_b = req(m, p + "self_attn.linear_q.bias");
        l.k_w = req(m, p + "self_attn.linear_k.weight");
        l.k_b = req(m, p + "self_attn.linear_k.bias");
        l.v_w = req(m, p + "self_attn.linear_v.weight");
        l.v_b = req(m, p + "self_attn.linear_v.bias");
        l.attn_out_w = req(m, p + "self_attn.linear_out.weight");
        l.attn_out_b = req(m, p + "self_attn.linear_out.bias");
        l.distance_w = req(m, p + "self_attn.distance_embedding.weight");
        l.conv_norm_w = req(m, p + "conv_module.layer_norm.weight");
        l.conv_norm_b = req(m, p + "conv_module.layer_norm.bias");
        l.conv_pw1_w = req(m, p + "conv_module.pointwise_conv1.weight");
        l.conv_pw2_w = req(m, p + "conv_module.pointwise_conv2.weight");
        l.conv_dw_w = req(m, p + "conv_module.depthwise_conv.weight");
        l.conv_dw_norm_w = req(m, p + "conv_module.depthwise_layer_norm.weight");
        l.conv_dw_norm_b = req(m, p + "conv_module.depthwise_layer_norm.bias");
        l.final_norm_w = req(m, p + "final_layer_norm.weight");
        l.final_norm_b = req(m, p + "final_layer_norm.bias");
    }

    if (m.hp.encoder_only) {
        if (!m.valid)
            return false;
        m.window = core_cpu::to_f32(m.frontend_window);
        m.mel_filters = core_cpu::to_f32(m.frontend_mels);
        return !m.window.empty() && !m.mel_filters.empty();
    }

    auto& d = m.dac;
    d.config.n_codebooks = 0;
    d.config.hidden_size = m.hp.hidden;
    d.config.decoder_hidden_size = 1536;
    d.config.sample_rate = m.hp.output_rate;
    d.config.hop_length = decoder_hop;
    d.config.n_decoder_blocks = decoder_blocks;
    const int channels[6] = {1536, 768, 384, 192, 96, 48};
    std::copy(decoder_rates, decoder_rates + 5, d.config.upsampling_ratios);
    std::copy(channels, channels + 6, d.config.decoder_channels);
    d.in_conv_w = req(m, "decoder.model.0.weight");
    d.in_conv_b = req(m, "decoder.model.0.bias");
    for (int i = 0; i < decoder_blocks; ++i) {
        auto& b = d.blocks[i];
        const std::string p = "decoder.model." + std::to_string(i + 1) + ".block.";
        b.snake_alpha = req(m, p + "0.alpha");
        b.up_w = req(m, p + "1.weight");
        b.up_b = req(m, p + "1.bias");
        for (int j = 0; j < 3; ++j) {
            auto& u = b.res[j];
            const std::string q = p + std::to_string(j + 2) + ".block.";
            u.alpha0 = req(m, q + "0.alpha");
            u.conv0_w = req(m, q + "1.weight");
            u.conv0_b = req(m, q + "1.bias");
            u.alpha1 = req(m, q + "2.alpha");
            u.conv1_w = req(m, q + "3.weight");
            u.conv1_b = req(m, q + "3.bias");
        }
    }
    d.out_snake_alpha = req(m, "decoder.model.6.alpha");
    d.out_conv_w = req(m, "decoder.model.7.weight");
    d.out_conv_b = req(m, "decoder.model.7.bias");

    if (!m.valid)
        return false;
    m.window = core_cpu::to_f32(m.frontend_window);
    m.mel_filters = core_cpu::to_f32(m.frontend_mels);
    return !m.window.empty() && !m.mel_filters.empty();
}

// Exact SeamlessM4T feature frontend used by w2v-BERT 2.0.
static std::vector<float> make_features(const sidon_model& m, const float* pcm, int n, int& T) {
    constexpr int win = 400, hop = sidon_frontend_hop, nfft = 512, bins = 257;
    const int M = m.hp.mel_bins;
    if (n < win) {
        T = 0;
        return {};
    }
    const int raw_T = 1 + (n - win) / hop;
    std::vector<float> raw((size_t)raw_T * M), re(nfft), im(nfft);
    for (int t = 0; t < raw_T; ++t) {
        const float* s = pcm + (size_t)t * hop;
        double mean = 0.0;
        for (int i = 0; i < win; ++i)
            mean += s[i] * 32768.0;
        mean /= win;
        std::fill(re.begin(), re.end(), 0.0f);
        std::fill(im.begin(), im.end(), 0.0f);
        float prev = (float)(s[0] * 32768.0 - mean);
        // transformers.audio_utils.spectrogram applies the first-sample
        // pre-emphasis as x[0] *= (1 - coefficient), rather than leaving it
        // unchanged as Kaldi's older in-place loop does.
        re[0] = (1.0f - 0.97f) * prev * m.window[0];
        for (int i = 1; i < win; ++i) {
            const float cur = (float)(s[i] * 32768.0 - mean);
            re[i] = (cur - 0.97f * prev) * m.window[(size_t)i];
            prev = cur;
        }
        core_fft::fft_radix2_inplace(re.data(), im.data(), nfft);
        for (int mel = 0; mel < M; ++mel) {
            double e = 0.0;
            for (int k = 0; k < bins; ++k) {
                const double power = (double)re[k] * re[k] + (double)im[k] * im[k];
                e += power * m.mel_filters[(size_t)k * M + mel];
            }
            raw[(size_t)t * M + mel] = std::log(std::max((float)e, 1.1920928955078125e-7f));
        }
    }
    // Upstream normalizes each mel bin over time using sample variance.
    for (int mel = 0; mel < M; ++mel) {
        double mean = 0.0;
        for (int t = 0; t < raw_T; ++t)
            mean += raw[(size_t)t * M + mel];
        mean /= raw_T;
        double ss = 0.0;
        for (int t = 0; t < raw_T; ++t) {
            double z = raw[(size_t)t * M + mel] - mean;
            ss += z * z;
        }
        const double var = raw_T > 1 ? ss / (raw_T - 1) : 0.0;
        const float inv = (float)(1.0 / std::sqrt(var + 1e-7));
        for (int t = 0; t < raw_T; ++t)
            raw[(size_t)t * M + mel] = (raw[(size_t)t * M + mel] - (float)mean) * inv;
    }
    T = raw_T / sidon_frontend_decim;
    std::vector<float> out((size_t)T * m.hp.feature_dim);
    for (int t = 0; t < T; ++t)
        std::memcpy(out.data() + (size_t)t * m.hp.feature_dim, raw.data() + (size_t)(2 * t) * M,
                    (size_t)m.hp.feature_dim * sizeof(float));
    return out;
}

static ggml_tensor* linear(ggml_context* c, ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(c, w, x);
    return b ? ggml_add(c, y, b) : y;
}

static ggml_tensor* predictor_norm(sidon_context* ctx, ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                   float eps) {
    if (!ctx->predictor_vulkan)
        return ggml_norm_affine(c, x, w, b, eps);

    // GGML's Vulkan backend does not currently implement the fused
    // NORM_AFFINE op.  Its three constituent ops are native Vulkan kernels,
    // so use those only for the Vulkan predictor and avoid a CPU split (and
    // two device transfers) at every layer norm.  CUDA keeps its fused kernel.
    return ggml_add(c, ggml_mul(c, ggml_norm(c, x, eps), w), b);
}

static ggml_tensor* ffn(sidon_context* ctx, ggml_context* c, ggml_tensor* x, const sidon_ffn& f, float eps) {
    ggml_tensor* y = predictor_norm(ctx, c, x, f.norm_w, f.norm_b, eps);
    y = ggml_silu(c, linear(c, f.up_w, y, f.up_b));
    y = linear(c, f.down_w, y, f.down_b);
    return ggml_add(c, x, ggml_scale(c, y, 0.5f));
}

static ggml_cgraph* build_predictor_graph(sidon_context* ctx, ggml_context* c, int T) {
    auto& m = ctx->model;
    const int D = m.hp.hidden, H = m.hp.heads, hd = D / H, Kc = m.hp.conv_kernel;
    ggml_cgraph* gf = ggml_new_graph_custom(c, 32768, false);
    ggml_tensor* in = ggml_new_tensor_2d(c, GGML_TYPE_F32, m.hp.feature_dim, T);
    ggml_set_name(in, "sidon_features");
    ggml_set_input(in);
    // Relative-position index layout depends on the RPE formulation (see
    // sidon_rpe_mode): the expand path indexes a flat [T*T] gather, the bucket
    // paths index per (key, query) with heads either repeated in-graph or
    // supplied directly.
    ggml_tensor* rel_idx = nullptr;
    switch (ctx->rpe_mode) {
    case sidon_rpe_mode::expand:
        rel_idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t)T * T);
        break;
    case sidon_rpe_mode::bucket:
        rel_idx = ggml_new_tensor_2d(c, GGML_TYPE_I32, T, T);
        break;
    case sidon_rpe_mode::bucket_direct:
        rel_idx = ggml_new_tensor_3d(c, GGML_TYPE_I32, T, T, H);
        break;
    }
    ggml_set_name(rel_idx, "sidon_rel_indices");
    ggml_set_input(rel_idx);
    // Identity gather (0..n_buckets-1) used to dequantize the relative-position
    // table in-graph; see the distance_w note in the bucket branch below.
    ggml_tensor* bucket_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, m.hp.rel_left + m.hp.rel_right + 1);
    ggml_set_name(bucket_ids, "sidon_bucket_ids");
    ggml_set_input(bucket_ids);

    ggml_tensor* cur = predictor_norm(ctx, c, in, m.feature_norm_w, m.feature_norm_b, m.hp.eps);
    cur = linear(c, m.feature_proj_w, cur, m.feature_proj_b);
    const float scale = 1.0f / std::sqrt((float)hd);
    for (int il = 0; il < m.hp.layers; ++il) {
        const auto& l = m.layers[(size_t)il];
        cur = ffn(ctx, c, cur, l.ffn1, m.hp.eps);

        ggml_tensor* x = predictor_norm(ctx, c, cur, l.attn_norm_w, l.attn_norm_b, m.hp.eps);
        ggml_tensor* Q = linear(c, l.q_w, x, l.q_b);
        ggml_tensor* K = linear(c, l.k_w, x, l.k_b);
        ggml_tensor* V = linear(c, l.v_w, x, l.v_b);
        Q = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, Q, hd, H, T), 0, 2, 1, 3));
        K = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, K, hd, H, T), 0, 2, 1, 3));
        V = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, V, hd, H, T), 0, 2, 1, 3));
        ggml_tensor* scores = ggml_mul_mat(c, K, Q);

        // Relative-position bias. All three formulations compute the same
        // quantity: bias[key, query, head] = dot(distance_w[:, bucket(key,
        // query)], Q[:, query, head]).
        ggml_tensor* bias = nullptr;
        if (ctx->rpe_mode == sidon_rpe_mode::expand) {
            // Materialise the distance table per (key, query) first. Costs
            // 4*head_dim*T^2 bytes and repeats the same dot product for every
            // key that falls in the same clipped bucket.
            ggml_tensor* rpe = ggml_get_rows(c, l.distance_w, rel_idx);
            rpe = ggml_reshape_4d(c, rpe, hd, T, T, 1);
            ggml_tensor* q4 = ggml_reshape_4d(c, Q, hd, 1, T, H);
            if (ctx->predictor_vulkan) {
                // Vulkan MUL_MAT requires equal ne[3] batch dimensions and does
                // not implement rpe[..., 1] broadcasting over H heads.  Fold
                // (T,H) into ne[2] instead, with heads as the inner batch because
                // GGML maps repeated ne[2] batches with i02 = i12/H.  This avoids
                // materialising an H-times-larger positional tensor.
                ggml_tensor* q_th = ggml_cont(c, ggml_permute(c, q4, 0, 1, 3, 2));
                ggml_tensor* q_flat = ggml_reshape_3d(c, q_th, hd, 1, (int64_t)T * H);
                ggml_tensor* bias_ht = ggml_reshape_3d(c, ggml_mul_mat(c, rpe, q_flat), T, H, T);
                bias = ggml_cont(c, ggml_permute(c, bias_ht, 0, 2, 1, 3));
            } else {
                bias = ggml_reshape_3d(c, ggml_mul_mat(c, rpe, q4), T, T, H);
            }
        } else {
            // The distance table has n_buckets rows, and n_buckets << T. Take
            // the query/table dot products once per (bucket, query, head) —
            // a [n_buckets, T, H] tensor — then gather the bucket each key
            // selects. Never materialises the [head_dim, T, T] expansion.
            //
            // distance_w must NOT stay quantized here (issue #416). This is the
            // only mul_mat in the model whose src0 is a quantized weight AND is
            // broadcast (ne2=1) across the H attention heads. On a Vulkan device
            // advertising integer dot product, ggml routes quantized src0 through
            // the MMQ/Q8_1 path (`quantize_y` in ggml-vulkan.cpp), and that
            // combination produced an all-zero predictor output -> a full-length
            // file of silence, while the f16 model (never quantized, never on
            // that path) was fine. A device WITHOUT integer dot product takes the
            // dequant-to-F16 fallback and is unaffected, which is why software
            // Vulkan (lavapipe, `int dot: 0`) could not reproduce it.
            //
            // Dequantizing costs nothing worth measuring: the table is 73x64, so
            // ~9 KB per layer, and it is gathered once per graph rather than per
            // frame. Use get_rows, not ggml_cast: get_rows works for every quant
            // type and returns F32, while ggml_cast on a k-quant aborts on Metal
            // (no k-quant CPY kernel).
            //
            // CRISPASR_SIDON_QUANT_RPE=1 restores the quantized multiply as a
            // bisection gate for when the Vulkan MMQ path is fixed upstream.
            ggml_tensor* dist_w = l.distance_w;
            if (ggml_is_quantized(dist_w->type) && !ctx->quant_rpe)
                dist_w = ggml_get_rows(c, dist_w, bucket_ids);       // [hd, n_buckets] F32
            ggml_tensor* bucket_logits = ggml_mul_mat(c, dist_w, Q); // [bucket, query, head]
            bucket_logits = ggml_reshape_4d(c, bucket_logits, 1, bucket_logits->ne[0], T, H);
            // ggml_get_rows batches on (ne2, ne3), so the index must carry the
            // head dimension. bucket_direct supplies it as a plain input (the
            // host fill is a memcpy per head); bucket materialises it with an
            // in-graph REPEAT, which costs an extra 4*H*T^2-byte I32 tensor per
            // layer and, having no Metal I32 kernel, lands on the CPU backend.
            ggml_tensor* idx =
                ctx->rpe_mode == sidon_rpe_mode::bucket_direct ? rel_idx : ggml_repeat_4d(c, rel_idx, T, T, H, 1);
            bias = ggml_reshape_3d(c, ggml_get_rows(c, bucket_logits, idx), T, T, H);
        }
        scores = ggml_soft_max_ext(c, ggml_add(c, scores, bias), nullptr, scale, 0.0f);
        ggml_tensor* vt = ggml_cont(c, ggml_permute(c, V, 1, 0, 2, 3));
        x = ggml_mul_mat(c, vt, scores);
        x = ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3));
        x = ggml_reshape_2d(c, x, D, T);
        cur = ggml_add(c, cur, linear(c, l.attn_out_w, x, l.attn_out_b));

        ggml_tensor* residual = cur;
        x = predictor_norm(ctx, c, cur, l.conv_norm_w, l.conv_norm_b, m.hp.eps);
        x = ggml_siglu_swapped(c, linear(c, ggml_reshape_2d(c, l.conv_pw1_w, D, 2 * D), x, nullptr));
        ggml_tensor* dw = ggml_reshape_4d(c, ggml_cast(c, l.conv_dw_w, GGML_TYPE_F32), Kc, 1, 1, D);
        x = ggml_reshape_4d(c, ggml_cont(c, ggml_transpose(c, x)), T, 1, D, 1);
        x = ggml_conv_2d_dw_direct(c, dw, x, 1, 1, Kc - 1, 0, 1, 1);
        x = ggml_cont(c, ggml_view_4d(c, x, T, 1, D, 1, x->nb[1], x->nb[2], x->nb[3], 0));
        x = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3)), D, T);
        x = ggml_silu(c, predictor_norm(ctx, c, x, l.conv_dw_norm_w, l.conv_dw_norm_b, m.hp.eps));
        x = linear(c, ggml_reshape_2d(c, l.conv_pw2_w, D, D), x, nullptr);
        cur = ggml_add(c, residual, x);
        cur = ffn(ctx, c, cur, l.ffn2, m.hp.eps);
        cur = predictor_norm(ctx, c, cur, l.final_norm_w, l.final_norm_b, m.hp.eps);
    }
    ggml_set_name(cur, "sidon_predictor_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

static ggml_cgraph* build_decoder_graph(sidon_context* ctx, ggml_context* c, int T) {
    ggml_cgraph* gf = ggml_new_graph_custom(c, 32768, false);
    ggml_tensor* in = ggml_new_tensor_2d(c, GGML_TYPE_F32, ctx->model.hp.hidden, T);
    ggml_set_name(in, "sidon_decoder_features");
    ggml_set_input(in);
    core_dac::build_decode_features_graph(c, ctx->model.dac, in, gf, &ctx->decoder_fc);
    return gf;
}

static void prepare_fastconv(sidon_context* ctx) {
    const char* env = std::getenv("CRISPASR_SIDON_FASTCONV");
    enum class mode { off, k1_f16, k1_f32, full } selected = mode::off;
    if (!env || !env[0]) {
        // CUDA A/B on a 5070 Ti: routing only the 15 pointwise residual
        // convolutions through F16-weight mul_mat cuts DAC time ~3.8% with no
        // extra weight cache. Full F32 baking is slower and adds substantial
        // startup/VRAM cost. Other backends retain the legacy path until
        // independently benchmarked.
        selected = ci_starts_with(ggml_backend_name(ctx->decoder_backend), "CUDA") ? mode::k1_f16 : mode::off;
    } else if (env[0] == '0' || std::strcmp(env, "off") == 0) {
        selected = mode::off;
    } else if (std::strcmp(env, "k1") == 0 || std::strcmp(env, "k1-f16") == 0) {
        selected = mode::k1_f16;
    } else if (std::strcmp(env, "k1-f32") == 0) {
        selected = mode::k1_f32;
    } else if (env[0] == '1' || std::strcmp(env, "full") == 0) {
        selected = mode::full;
    } else {
        std::fprintf(stderr, "sidon: unknown CRISPASR_SIDON_FASTCONV='%s'; using off\n", env);
    }
    if (selected == mode::off) {
        if (ctx->params.verbosity)
            std::fprintf(stderr, "sidon: FASTCONV off\n");
        return;
    }

    if (selected == mode::k1_f16) {
        // Enabling the helper without baking leaves every kernel in its GGUF
        // type, but still takes the K=1 -> mul_mat branch.
        ctx->decoder_fc.enabled = true;
        if (ctx->params.verbosity)
            std::fprintf(stderr, "sidon: FASTCONV k1-f16 (no baked kernels)\n");
        return;
    }

    const bool k1_only = selected == mode::k1_f32;
    auto& d = ctx->model.dac;
    std::vector<ggml_tensor*> kernels;
    if (!k1_only) {
        kernels.push_back(d.in_conv_w);
        kernels.push_back(d.out_conv_w);
    }
    for (int b = 0; b < d.config.n_decoder_blocks; ++b) {
        if (!k1_only)
            kernels.push_back(d.blocks[b].up_w);
        for (int r = 0; r < 3; ++r) {
            if (!k1_only)
                kernels.push_back(d.blocks[b].res[r].conv0_w);
            kernels.push_back(d.blocks[b].res[r].conv1_w);
        }
    }
    ctx->decoder_fc.bake(ctx->decoder_backend, kernels, true);
    if (ctx->params.verbosity)
        std::fprintf(stderr, "sidon: FASTCONV %s (%zu baked kernels)\n", k1_only ? "k1" : "full", kernels.size());
}

static void clear_predictor_graph(sidon_context* ctx) {
    if (ctx->predictor_sched)
        ggml_backend_sched_reset(ctx->predictor_sched);
    if (ctx->predictor_ctx)
        ggml_free(ctx->predictor_ctx);
    ctx->predictor_ctx = nullptr;
    ctx->predictor_graph = nullptr;
    ctx->predictor_input = nullptr;
    ctx->relative_indices = nullptr;
    ctx->predictor_output = nullptr;
}

static void clear_decoder_graph(sidon_context* ctx) {
    if (ctx->decoder_sched)
        ggml_backend_sched_reset(ctx->decoder_sched);
    if (ctx->decoder_ctx)
        ggml_free(ctx->decoder_ctx);
    ctx->decoder_ctx = nullptr;
    ctx->decoder_graph = nullptr;
    ctx->decoder_input = nullptr;
    ctx->decoder_output = nullptr;
}

static void clear_graphs(sidon_context* ctx) {
    clear_predictor_graph(ctx);
    clear_decoder_graph(ctx);
}

static ggml_backend_sched_t make_stage_scheduler(ggml_backend_t primary, ggml_backend_t cpu) {
    ggml_backend_t backends[2];
    int n_backends = 0;
    backends[n_backends++] = primary;
    if (cpu && cpu != primary)
        backends[n_backends++] = cpu;
    return ggml_backend_sched_new(backends, nullptr, n_backends, 32768, false, false);
}

// Report the compute-buffer bytes a stage's scheduler actually reserved, per
// backend. Process-level RSS / "peak memory footprint" is NOT a usable proxy
// here: on Metal the compute buffers are MTLBuffers that the macOS footprint
// accounting does not attribute to the process, so a graph-shape change can
// look like a memory win or loss purely from where the bytes live. Gated by
// CRISPASR_SIDON_DEBUG so it can be A/B'd without a rebuild.
static void report_workspace(sidon_context* ctx, ggml_backend_sched_t sched, const char* stage) {
    const char* e = getenv("CRISPASR_SIDON_DEBUG");
    if (!(e && e[0]) || !sched)
        return;
    ggml_backend_t backends[2] = {stage[0] == 'p' ? ctx->backend : ctx->decoder_backend, ctx->backend_cpu};
    size_t total = 0;
    for (int i = 0; i < 2; ++i) {
        if (!backends[i] || (i == 1 && backends[1] == backends[0]))
            continue;
        const size_t sz = ggml_backend_sched_get_buffer_size(sched, backends[i]);
        total += sz;
        std::fprintf(stderr, "sidon: workspace %s [%s] = %.1f MiB\n", stage, ggml_backend_name(backends[i]),
                     (double)sz / (1024.0 * 1024.0));
    }
    std::fprintf(stderr, "sidon: workspace %s TOTAL = %.1f MiB\n", stage, (double)total / (1024.0 * 1024.0));
}

static bool release_predictor_workspace(sidon_context* ctx) {
    clear_predictor_graph(ctx);
    if (ctx->predictor_sched)
        ggml_backend_sched_free(ctx->predictor_sched);
    ctx->predictor_sched = make_stage_scheduler(ctx->backend, ctx->backend_cpu);
    return ctx->predictor_sched != nullptr;
}

static bool release_decoder_workspace(sidon_context* ctx) {
    clear_decoder_graph(ctx);
    if (ctx->decoder_sched)
        ggml_backend_sched_free(ctx->decoder_sched);
    ctx->decoder_sched = make_stage_scheduler(ctx->decoder_backend, ctx->backend_cpu);
    return ctx->decoder_sched != nullptr;
}

static int report_unsupported_vulkan_ops(sidon_context* ctx, ggml_backend_t backend, ggml_cgraph* graph,
                                         const char* stage) {
    int unsupported = 0;
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(backend, node)) {
            if (unsupported < 8) {
                std::fprintf(stderr, "sidon: Vulkan %s fallback op: %s (%s)\n", stage, ggml_op_name(node->op),
                             node->name);
            }
            ++unsupported;
        }
    }
    if (unsupported) {
        std::fprintf(stderr, "sidon: WARNING: Vulkan %s has %d unsupported op(s); CPU fallback remains enabled\n",
                     stage, unsupported);
    } else if (ctx->params.verbosity) {
        std::fprintf(stderr, "sidon: Vulkan %s is fully native (%d graph nodes)\n", stage, n_nodes);
    }
    return unsupported;
}

static bool prepare_predictor_graph(sidon_context* ctx, int T) {
    clear_graphs(ctx);
    if (!ctx->predictor_sched)
        ctx->predictor_sched = make_stage_scheduler(ctx->backend, ctx->backend_cpu);
    if (!ctx->predictor_sched) {
        std::fprintf(stderr, "sidon: predictor scheduler allocation failed\n");
        return false;
    }
    const size_t meta_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false);
    ctx->predictor_meta.assign(meta_size, 0);

    ggml_init_params pred_ip = {ctx->predictor_meta.size(), ctx->predictor_meta.data(), true};
    ctx->predictor_ctx = ggml_init(pred_ip);
    if (!ctx->predictor_ctx) {
        std::fprintf(stderr, "sidon: predictor graph context allocation failed\n");
        clear_graphs(ctx);
        return false;
    }

    // Length-aware RPE choice (#416 follow-up). Must precede the build: it
    // changes which tensors the graph materialises.
    {
        const int H = ctx->model.hp.heads;
        const int hd = H > 0 ? ctx->model.hp.hidden / H : 0;
        const bool is_gpu = ctx->backend && !core_cpu_backend::is_cpu(ctx->backend);
        ctx->rpe_mode = resolve_rpe_mode(T, hd, H, is_gpu, ctx->params.verbosity);
    }

    ctx->predictor_graph = build_predictor_graph(ctx, ctx->predictor_ctx, T);
    if (ctx->predictor_vulkan)
        report_unsupported_vulkan_ops(ctx, ctx->backend, ctx->predictor_graph, "predictor");
    if (!ggml_backend_sched_alloc_graph(ctx->predictor_sched, ctx->predictor_graph)) {
        std::fprintf(stderr, "sidon: predictor graph allocation failed\n");
        clear_graphs(ctx);
        return false;
    }
    report_workspace(ctx, ctx->predictor_sched, "predictor");

    ctx->predictor_input = ggml_graph_get_tensor(ctx->predictor_graph, "sidon_features");
    ctx->relative_indices = ggml_graph_get_tensor(ctx->predictor_graph, "sidon_rel_indices");
    ctx->bucket_ids = ggml_graph_get_tensor(ctx->predictor_graph, "sidon_bucket_ids");
    ctx->predictor_output = ggml_graph_get_tensor(ctx->predictor_graph, "sidon_predictor_output");
    if (!ctx->predictor_input || !ctx->relative_indices || !ctx->predictor_output) {
        std::fprintf(stderr, "sidon: predictor graph is missing a required tensor\n");
        clear_graphs(ctx);
        return false;
    }
    return true;
}

static bool prepare_decoder_graph(sidon_context* ctx, int T) {
    clear_decoder_graph(ctx);
    if (!ctx->decoder_sched)
        ctx->decoder_sched = make_stage_scheduler(ctx->decoder_backend, ctx->backend_cpu);
    if (!ctx->decoder_sched) {
        std::fprintf(stderr, "sidon: decoder scheduler allocation failed\n");
        return false;
    }
    const size_t meta_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false);
    ctx->decoder_meta.assign(meta_size, 0);
    ggml_init_params dec_ip = {ctx->decoder_meta.size(), ctx->decoder_meta.data(), true};
    ctx->decoder_ctx = ggml_init(dec_ip);
    if (!ctx->decoder_ctx) {
        std::fprintf(stderr, "sidon: decoder graph context allocation failed\n");
        clear_decoder_graph(ctx);
        return false;
    }
    ctx->decoder_graph = build_decoder_graph(ctx, ctx->decoder_ctx, T);
    if (ctx->predictor_vulkan)
        report_unsupported_vulkan_ops(ctx, ctx->decoder_backend, ctx->decoder_graph, "DAC");
    if (!ggml_backend_sched_alloc_graph(ctx->decoder_sched, ctx->decoder_graph)) {
        std::fprintf(stderr, "sidon: decoder graph allocation failed\n");
        clear_decoder_graph(ctx);
        return false;
    }
    report_workspace(ctx, ctx->decoder_sched, "decoder");
    ctx->decoder_input = ggml_graph_get_tensor(ctx->decoder_graph, "sidon_decoder_features");
    ctx->decoder_output = ggml_graph_get_tensor(ctx->decoder_graph, "dac_pcm");
    if (!ctx->decoder_input || !ctx->decoder_output) {
        std::fprintf(stderr, "sidon: decoder graph is missing a required tensor\n");
        clear_decoder_graph(ctx);
        return false;
    }
    return true;
}

sidon_context_params sidon_context_default_params() {
    return {4, 1, true};
}

sidon_context* sidon_init_from_file(const char* path, sidon_context_params params) {
    sidon_context* ctx = new sidon_context();
    ctx->params = params;
    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : core_cpu_backend::init();
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();
    ctx->predictor_vulkan = ci_starts_with(ggml_backend_name(ctx->backend), "Vulkan");
    // Keep stage execution and synchronization independent so predictor and
    // DAC timings describe their own CUDA work rather than a shared queue.
    // Vulkan shares one backend instance so decoder weights and the predictor
    // output never cross Vulkan queues/devices merely because the stages have
    // separate schedulers.
    ctx->decoder_backend = ctx->predictor_vulkan
                               ? ctx->backend
                               : (params.use_gpu ? crispasr_init_gpu_backend() : core_cpu_backend::init());
    if (!ctx->decoder_backend)
        ctx->decoder_backend = core_cpu_backend::init();
    ctx->backend_cpu = core_cpu_backend::init();
    const int nt = params.n_threads > 0 ? params.n_threads : 4;
    if (core_cpu_backend::is_cpu(ctx->backend))
        core_cpu_backend::set_n_threads(ctx->backend, nt);
    if (core_cpu_backend::is_cpu(ctx->decoder_backend))
        core_cpu_backend::set_n_threads(ctx->decoder_backend, nt);
    if (ctx->backend_cpu)
        core_cpu_backend::set_n_threads(ctx->backend_cpu, nt);
    if (!load_model(ctx->model, path, ctx->backend)) {
        sidon_free(ctx);
        return nullptr;
    }
    prepare_fastconv(ctx);
    // ctx->rpe_mode is resolved per graph build (it depends on T) — see
    // resolve_rpe_mode() and prepare_predictor_graph().
    if (const char* e = getenv("CRISPASR_SIDON_QUANT_RPE"); e && e[0] && e[0] != '0')
        ctx->quant_rpe = true;
    ctx->predictor_sched = make_stage_scheduler(ctx->backend, ctx->backend_cpu);
    ctx->decoder_sched = make_stage_scheduler(ctx->decoder_backend, ctx->backend_cpu);
    if (!ctx->predictor_sched || !ctx->decoder_sched) {
        sidon_free(ctx);
        return nullptr;
    }
    if (params.verbosity)
        std::fprintf(stderr, "sidon: loaded %s (%d predictor layers, 48 kHz DAC)\n", path, ctx->model.hp.layers);
    return ctx;
}

void sidon_free(sidon_context* ctx) {
    if (!ctx)
        return;
    clear_graphs(ctx);
    if (ctx->predictor_sched)
        ggml_backend_sched_free(ctx->predictor_sched);
    if (ctx->decoder_sched)
        ggml_backend_sched_free(ctx->decoder_sched);
    ctx->decoder_fc.free();
    if (ctx->model.buf)
        core_gguf::release_weight_buffer(ctx->model.buf);
    if (ctx->model.buf_cpu)
        core_gguf::release_weight_buffer(ctx->model.buf_cpu);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    if (ctx->decoder_backend && ctx->decoder_backend != ctx->backend)
        ggml_backend_free(ctx->decoder_backend);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}


// Upload the frontend features and the clipped relative-distance bucket table
// per (key, query). The table is identical for every head; bucket_direct wants
// it replicated H times because ggml_get_rows batches the index on (ne2, ne3).
static void set_predictor_inputs(sidon_context* ctx, const std::vector<float>& feats, int T) {
    ggml_backend_tensor_set(ctx->predictor_input, feats.data(), 0, feats.size() * sizeof(float));
    const int n_heads = ctx->model.hp.heads;
    const size_t plane = (size_t)T * T;
    const size_t n_planes = ctx->rpe_mode == sidon_rpe_mode::bucket_direct ? (size_t)n_heads : 1;
    std::vector<int32_t> indices(plane * n_planes);
    for (int q = 0; q < T; ++q)
        for (int k = 0; k < T; ++k)
            indices[(size_t)q * T + k] =
                std::max(-ctx->model.hp.rel_left, std::min(ctx->model.hp.rel_right, k - q)) + ctx->model.hp.rel_left;
    for (size_t h = 1; h < n_planes; ++h)
        std::memcpy(indices.data() + h * plane, indices.data(), plane * sizeof(int32_t));
    ggml_backend_tensor_set(ctx->relative_indices, indices.data(), 0, indices.size() * sizeof(int32_t));
    // Re-set on EVERY call even though the contents never change: gallocr may
    // hand an input's buffer to a later intermediate as scratch once its last
    // use has passed, so a constant written only at build time is silently
    // corrupted from the second eval onward.
    if (ctx->bucket_ids) {
        const int n_buckets = (int)ggml_nelements(ctx->bucket_ids);
        std::vector<int32_t> ids((size_t)n_buckets);
        for (int i = 0; i < n_buckets; ++i)
            ids[(size_t)i] = i;
        ggml_backend_tensor_set(ctx->bucket_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    }
}

// Encoder-only feature extraction: SeamlessM4T frontend + the loaded conformer
// layers, returning the raw hidden states (T x hidden, row-major). This is the
// PLAIN w2v-BERT forward -- none of sidon_restore's peak normalization or
// boundary padding, which are restoration-recipe specifics. For a 17-layer
// encoder-only GGUF the result equals HF `output.hidden_states[17]`.
std::vector<float> sidon_extract_hidden(sidon_context* ctx, const float* pcm_16k, int n_samples, int* n_frames_out) {
    if (n_frames_out)
        *n_frames_out = 0;
    if (!ctx || !pcm_16k || n_samples < 400)
        return {};
    int T = 0;
    std::vector<float> feats = make_features(ctx->model, pcm_16k, n_samples, T);
    if (T <= 0)
        return {};
    int max_frames = 3000; // same O(T^2) attention guard as sidon_restore
    if (const char* e = getenv("CRISPASR_SIDON_MAX_FRAMES"); e && e[0]) {
        const int v = atoi(e);
        if (v > 0)
            max_frames = v;
    }
    if (T > max_frames) {
        std::fprintf(stderr, "sidon: extract_hidden input too long (%d frames > %d cap)\n", T, max_frames);
        return {};
    }
    if (!prepare_predictor_graph(ctx, T)) {
        release_predictor_workspace(ctx);
        return {};
    }
    set_predictor_inputs(ctx, feats, T);
    core_quant_bcast::audit(ctx->predictor_graph, "sidon");
    if (ggml_backend_sched_graph_compute(ctx->predictor_sched, ctx->predictor_graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "sidon: predictor graph compute failed\n");
        release_predictor_workspace(ctx);
        return {};
    }
    ggml_backend_sched_synchronize(ctx->predictor_sched);
    std::vector<float> hidden((size_t)ggml_nelements(ctx->predictor_output));
    ggml_backend_tensor_get(ctx->predictor_output, hidden.data(), 0, hidden.size() * sizeof(float));
    release_predictor_workspace(ctx);
    if (n_frames_out)
        *n_frames_out = T;
    return hidden;
}

// Degenerate-output detection (issue #416). Sidon's failure mode when a stage
// miscomputes is not a crash and not an error return — it is a full-length
// buffer of zeros or NaNs, which reaches the WAV writer as a perfectly valid
// file of pure silence. The reporter of #416 had literally nothing to go on:
// rc=0, correct duration, correct sample rate, no message.
//
// NaN and all-zero are reported separately on purpose. They look identical
// downstream (float->int16 turns NaN into 0), but they mean different things:
// NaN is arithmetic that went wrong, all-zero is a stage that produced nothing.
// Checking both stages says whether the predictor or the DAC decoder is at
// fault, which is the first question any such report has to answer, and it
// costs one O(n) pass against multi-second graph compute.
namespace {
struct SignalStats {
    size_t n_nonfinite = 0;
    float peak = 0.0f;
};

SignalStats scan_signal(const std::vector<float>& v) {
    SignalStats s;
    for (const float x : v) {
        if (!std::isfinite(x)) {
            ++s.n_nonfinite;
            continue;
        }
        s.peak = std::max(s.peak, std::fabs(x));
    }
    return s;
}

// `stage` is the producer being judged; `next` names what it feeds, so the
// message points at the boundary rather than just the symptom.
void warn_if_degenerate(const SignalStats& s, size_t n, const char* stage, const char* next) {
    if (s.n_nonfinite) {
        std::fprintf(stderr,
                     "sidon: WARNING: %s produced %zu non-finite value(s) of %zu — %s will be silent. "
                     "This is a compute fault, not a quantization-precision effect; please report the model "
                     "file and the backend in use (run with -ng to compare against CPU).\n",
                     stage, s.n_nonfinite, n, next);
    } else if (s.peak == 0.0f && n > 0) {
        std::fprintf(stderr,
                     "sidon: WARNING: %s output is identically zero — %s will be silent. "
                     "Please report the model file and the backend in use (run with -ng to compare against CPU).\n",
                     stage, next);
    }
}
} // namespace

std::vector<float> sidon_restore(sidon_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples < 400)
        return {};
    if (ctx->model.hp.encoder_only) {
        std::fprintf(stderr, "sidon: encoder-only GGUF has no DAC decoder; restoration unavailable\n");
        return {};
    }
    using clock = std::chrono::steady_clock;
    const auto total_start = clock::now();
    // Match the reference inference recipe's peak normalization.
    float peak = 0.0f;
    for (int i = 0; i < n_samples; ++i)
        peak = std::max(peak, std::fabs(samples[i]));
    // Pad the input with a whole predictor frame of leading context and 1.5 s
    // of right-side lookahead, then crop both back off after decoding.
    //
    // Without the lookahead the predictor's boundary response reaches the DAC
    // directly and the last ~12 ms of every clip is a full-scale click (on
    // samples/jfk.wav it is the peak sample of the whole file). The leading pad
    // is a WHOLE predictor frame — sidon_frontend_decim STFT hops, not one —
    // because make_features decimates raw mel frames by taking even indices, so
    // a half-frame shift would change WHICH mel frames the predictor sees
    // rather than just delaying them.
    //
    // Gated so the unpadded path stays available for A/B and for reproducing
    // the pre-padding reference dumps.
    int lead_frames = 1;
    int lookahead_samples = ctx->model.hp.input_rate * 3 / 2; // 1.5 s
    if (const char* e = getenv("CRISPASR_SIDON_LOOKAHEAD"); e && e[0] && atoi(e) == 0) {
        lead_frames = 0;
        lookahead_samples = 0;
    }
    const int frontend_pad_samples = lead_frames * sidon_frontend_decim * sidon_frontend_hop;
    std::vector<float> normalized((size_t)n_samples + frontend_pad_samples + lookahead_samples, 0.0f);
    const float gain = peak > 1e-9f ? 0.9f / peak : 1.0f;
    for (int i = 0; i < n_samples; ++i)
        normalized[(size_t)frontend_pad_samples + i] = samples[i] * gain;
    int T = 0;
    std::vector<float> feats = make_features(ctx->model, normalized.data(), (int)normalized.size(), T);
    if (T <= 0)
        return {};

    // Guard against O(T^2) attention blowup. The predictor materializes
    // (heads, T, T) relative indices and attention scores, so cost grows
    // quadratically in the feature-frame count T (~50 frames/sec of input).
    // Restoration is utterance-scale; cap T and fail cleanly rather than let a
    // multi-minute clip exhaust memory. After the required 1.5 s lookahead,
    // the default ~3000-frame cap permits ~58.5 s of user audio; override it
    // only when the selected backend has sufficient memory.
    int max_frames = 3000;
    if (const char* e = getenv("CRISPASR_SIDON_MAX_FRAMES"); e && e[0]) {
        const int v = atoi(e);
        if (v > 0)
            max_frames = v;
    }
    if (T > max_frames) {
        fprintf(stderr,
                "sidon: input too long — %d feature frames (~%.1f s) exceeds the %d-frame cap; "
                "O(T^2) attention would OOM. Split the audio or raise CRISPASR_SIDON_MAX_FRAMES.\n",
                T, (double)T / 50.0, max_frames);
        return {};
    }
    const auto frontend_done = clock::now();

    if (!prepare_predictor_graph(ctx, T)) {
        release_predictor_workspace(ctx);
        release_decoder_workspace(ctx);
        return {};
    }
    const auto graph_done = clock::now();

    set_predictor_inputs(ctx, feats, T);
    const auto predictor_start = clock::now();
    core_quant_bcast::audit(ctx->predictor_graph, "sidon");
    if (ggml_backend_sched_graph_compute(ctx->predictor_sched, ctx->predictor_graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "sidon: predictor graph compute failed\n");
        release_predictor_workspace(ctx);
        release_decoder_workspace(ctx);
        return {};
    }
    ggml_backend_sched_synchronize(ctx->predictor_sched);
    const auto predictor_done = clock::now();

    std::vector<float> predictor_features((size_t)ggml_nelements(ctx->predictor_output));
    ggml_backend_tensor_get(ctx->predictor_output, predictor_features.data(), 0,
                            predictor_features.size() * sizeof(float));

    // Judge the predictor BEFORE the DAC consumes it, so a bad handoff is
    // attributed to the predictor rather than to the decoder it poisons.
    const SignalStats predictor_stats = scan_signal(predictor_features);
    warn_if_degenerate(predictor_stats, predictor_features.size(), "the w2v-BERT predictor", "the restored audio");

    // Diff-harness hook: dump the predictor handoff (raw f32 + ne dims on a
    // header line to stderr) so it can be compared against the upstream
    // reference (sidon-ref.gguf: predictor_feats) to localize any port
    // divergence to the predictor vs the DAC decoder. See
    // tools/reference_backends/sidon_ref_dump.py.
    if (const char* dp = getenv("CRISPASR_SIDON_DUMP_HANDOFF"); dp && dp[0]) {
        const int64_t ne0 = ctx->predictor_output->ne[0], ne1 = ctx->predictor_output->ne[1];
        if (FILE* f = fopen(dp, "wb")) {
            fwrite(predictor_features.data(), sizeof(float), predictor_features.size(), f);
            fclose(f);
            fprintf(stderr, "sidon: dumped predictor handoff ne=[%lld,%lld] (%zu floats) -> %s\n", (long long)ne0,
                    (long long)ne1, predictor_features.size(), dp);
        }
    }

    if (!release_predictor_workspace(ctx)) {
        std::fprintf(stderr, "sidon: failed to release predictor workspace\n");
        release_decoder_workspace(ctx);
        return {};
    }
    const auto handoff_done = clock::now();

    // DAC is fully convolutional, so a bounded core plus the decoder's latent
    // receptive field reconstructs exactly the samples that core owns. Decode
    // core-sized pieces with context on both sides and crop the overlap away.
    // This bounds ggml_conv_1d's explicit IM2COL workspace independently of
    // utterance length: at T≈2825 the whole-utterance graph reserves ~4.5 GiB
    // against ~0.44 GiB chunked, a 10x reduction.
    //
    // decoder_context_frames is derived, not tuned: see dac_receptive_frames().
    //
    // The knob is the MAXIMUM core size (a workspace budget: the decoder graph
    // costs roughly 1.6 MiB per window frame). 0 means "one graph for the whole
    // utterance", which is what the parity test uses.
    int max_core_frames = 512;
    if (const char* e = getenv("CRISPASR_SIDON_DECODER_CHUNK_FRAMES"); e && e[0]) {
        const int v = atoi(e);
        max_core_frames = v > 0 ? v : T;
    }
    const int decoder_context_frames = dac_receptive_frames(ctx->model.dac);
    const int decoder_hop = ctx->model.dac.config.hop_length;
    const int hidden = ctx->model.hp.hidden;

    // Split into the FEWEST chunks that respect the budget, then spread the
    // cores evenly over them. Even cores matter: with a fixed core size, a T
    // just past a multiple of the core leaves a final window that is nearly all
    // context and almost no payload — at T=625 / core=256 that decoded 840
    // frames' worth of graph for 625 frames of audio (+34%). Balancing gives
    // ceil(625/512)=2 chunks of 313 (+7.8%).
    const int n_chunks = (T + max_core_frames - 1) / max_core_frames;
    const int core_frames = (T + n_chunks - 1) / n_chunks;

    // Every chunk uses the SAME window length, so at the edges the window slides
    // inward rather than shrinking. The cropped core is identical either way.
    //
    // The graph is rebuilt per chunk on purpose. Building it once and reusing it
    // across chunks looks like an obvious win and is NOT correct here: the
    // scheduler hands the decoder input buffer to a later intermediate as
    // scratch, so from the second chunk on the decode runs on clobbered input.
    // Measured on samples/jfk.wav that cost max|diff| 0.476 vs the
    // whole-utterance decode and dropped spectral correlation to the source from
    // 0.928 to 0.795 — and it bought nothing: with balanced cores the rebuild is
    // free (DAC 42234 ms rebuilding vs 42253 ms reusing). Do not "optimize" this
    // back without a bit-exactness check against CRISPASR_SIDON_DECODER_CHUNK_FRAMES=0.
    const int window_frames = std::min(T, core_frames + 2 * decoder_context_frames);

    std::vector<float> pcm;
    pcm.reserve((size_t)T * decoder_hop);
    std::vector<float> chunk_pcm;
    for (int core_start = 0; core_start < T; core_start += core_frames) {
        const int core_end = std::min(T, core_start + core_frames);
        int chunk_start = std::max(0, core_start - decoder_context_frames);
        if (chunk_start + window_frames > T)
            chunk_start = T - window_frames;
        const int chunk_frames = window_frames;
        const int chunk_end = chunk_start + chunk_frames;
        if (!prepare_decoder_graph(ctx, chunk_frames)) {
            release_decoder_workspace(ctx);
            return {};
        }
        const float* chunk_features = predictor_features.data() + (size_t)chunk_start * hidden;
        ggml_backend_tensor_set(ctx->decoder_input, chunk_features, 0, (size_t)chunk_frames * hidden * sizeof(float));
        core_quant_bcast::audit(ctx->decoder_graph, "sidon");
        if (ggml_backend_sched_graph_compute(ctx->decoder_sched, ctx->decoder_graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "sidon: decoder graph compute failed\n");
            release_decoder_workspace(ctx);
            return {};
        }
        ggml_backend_sched_synchronize(ctx->decoder_sched);

        chunk_pcm.resize((size_t)ggml_nelements(ctx->decoder_output));
        ggml_backend_tensor_get(ctx->decoder_output, chunk_pcm.data(), 0, chunk_pcm.size() * sizeof(float));
        // Crop to the samples this CORE owns. Keyed on core_end, not chunk_end:
        // once the window starts sliding inward several consecutive chunks share
        // chunk_end == T, and only the final core may run to the buffer end.
        const size_t crop_begin = (size_t)(core_start - chunk_start) * decoder_hop;
        const size_t crop_end =
            core_end == T ? chunk_pcm.size() : crop_begin + (size_t)(core_end - core_start) * decoder_hop;
        if (crop_begin > crop_end || crop_end > chunk_pcm.size()) {
            std::fprintf(stderr, "sidon: invalid decoder chunk crop [%zu,%zu) of %zu\n", crop_begin, crop_end,
                         chunk_pcm.size());
            release_decoder_workspace(ctx);
            return {};
        }
        pcm.insert(pcm.end(), chunk_pcm.begin() + (std::ptrdiff_t)crop_begin,
                   chunk_pcm.begin() + (std::ptrdiff_t)crop_end);
    }
    const auto decoder_done = clock::now();

    // Crop the padding back off. The leading pad is a whole predictor frame, so
    // it maps to exactly lead_frames * decoder_hop output samples; dropping only
    // the tail (as the original padding change did) would leave the result
    // delayed by that much AND truncate the same amount of real audio off the
    // end.
    const size_t target_samples = (size_t)n_samples * ctx->model.hp.output_rate / ctx->model.hp.input_rate;
    const size_t lead_samples = (size_t)lead_frames * decoder_hop;
    if (pcm.size() < lead_samples + target_samples) {
        std::fprintf(stderr, "sidon: padded decoder output is shorter than the requested duration (%zu < %zu)\n",
                     pcm.size(), lead_samples + target_samples);
        release_decoder_workspace(ctx);
        return {};
    }
    pcm.erase(pcm.begin(), pcm.begin() + (std::ptrdiff_t)lead_samples);
    pcm.resize(target_samples);

    // Only worth reporting if the predictor was healthy — otherwise the warning
    // above already named the real culprit and this would just be its echo.
    if (predictor_stats.n_nonfinite == 0 && predictor_stats.peak > 0.0f)
        warn_if_degenerate(scan_signal(pcm), pcm.size(), "the DAC decoder", "the written file");

    const auto download_done = clock::now();
    if (!release_decoder_workspace(ctx)) {
        std::fprintf(stderr, "sidon: failed to release decoder workspace\n");
        return {};
    }
    const auto total_done = clock::now();
    if (ctx->params.verbosity) {
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::fprintf(stderr,
                     "sidon: timings T=%d frontend=%.2f graph=%.2f predictor=%.2f handoff=%.2f "
                     "dac=%.2f download=%.2f cleanup=%.2f total=%.2f ms\n",
                     T, ms(total_start, frontend_done), ms(frontend_done, graph_done),
                     ms(predictor_start, predictor_done), ms(predictor_done, handoff_done),
                     ms(handoff_done, decoder_done), ms(decoder_done, download_done), ms(download_done, total_done),
                     ms(total_start, total_done));
    }
    return pcm;
}
