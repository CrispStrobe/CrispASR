#include "sidon.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include "core/cpu_ops.h"
#include "core/dac_decoder.h"
#include "core/fft.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

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

struct sidon_context {
    sidon_context_params params{};
    sidon_model model;
    core_dac::fastconv_cache decoder_fc;
    ggml_backend_t backend = nullptr, decoder_backend = nullptr, backend_cpu = nullptr;
    ggml_backend_sched_t predictor_sched = nullptr, decoder_sched = nullptr;
    bool predictor_vulkan = false;

    std::vector<uint8_t> predictor_meta, decoder_meta;
    ggml_context *predictor_ctx = nullptr, *decoder_ctx = nullptr;
    ggml_cgraph *predictor_graph = nullptr, *decoder_graph = nullptr;
    ggml_tensor *predictor_input = nullptr, *relative_indices = nullptr;
    ggml_tensor *predictor_output = nullptr, *decoder_input = nullptr, *decoder_output = nullptr;
};

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
        m.hp.output_rate != 48000 || decoder_blocks != 5 || decoder_hop != 960 || !decoder_rates_valid) {
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
    constexpr int win = 400, hop = 160, nfft = 512, bins = 257;
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
    T = raw_T / 2;
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
    ggml_tensor* rel_idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t)T * T);
    ggml_set_name(rel_idx, "sidon_rel_indices");
    ggml_set_input(rel_idx);

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
        ggml_tensor* rpe = ggml_get_rows(c, l.distance_w, rel_idx);
        rpe = ggml_reshape_4d(c, rpe, hd, T, T, 1);
        ggml_tensor* q4 = ggml_reshape_4d(c, Q, hd, 1, T, H);
        ggml_tensor* bias = nullptr;
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

static void clear_graphs(sidon_context* ctx) {
    if (ctx->predictor_sched)
        ggml_backend_sched_reset(ctx->predictor_sched);
    if (ctx->decoder_sched)
        ggml_backend_sched_reset(ctx->decoder_sched);
    if (ctx->predictor_ctx)
        ggml_free(ctx->predictor_ctx);
    if (ctx->decoder_ctx)
        ggml_free(ctx->decoder_ctx);
    ctx->predictor_ctx = nullptr;
    ctx->decoder_ctx = nullptr;
    ctx->predictor_graph = nullptr;
    ctx->decoder_graph = nullptr;
    ctx->predictor_input = nullptr;
    ctx->relative_indices = nullptr;
    ctx->predictor_output = nullptr;
    ctx->decoder_input = nullptr;
    ctx->decoder_output = nullptr;
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

static bool prepare_graphs(sidon_context* ctx, int T) {
    clear_graphs(ctx);
    const size_t meta_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false);
    ctx->predictor_meta.assign(meta_size, 0);
    ctx->decoder_meta.assign(meta_size, 0);

    ggml_init_params pred_ip = {ctx->predictor_meta.size(), ctx->predictor_meta.data(), true};
    ggml_init_params dec_ip = {ctx->decoder_meta.size(), ctx->decoder_meta.data(), true};
    ctx->predictor_ctx = ggml_init(pred_ip);
    ctx->decoder_ctx = ggml_init(dec_ip);
    if (!ctx->predictor_ctx || !ctx->decoder_ctx) {
        std::fprintf(stderr, "sidon: graph context allocation failed\n");
        clear_graphs(ctx);
        return false;
    }

    ctx->predictor_graph = build_predictor_graph(ctx, ctx->predictor_ctx, T);
    ctx->decoder_graph = build_decoder_graph(ctx, ctx->decoder_ctx, T);
    if (ctx->predictor_vulkan) {
        report_unsupported_vulkan_ops(ctx, ctx->backend, ctx->predictor_graph, "predictor");
        report_unsupported_vulkan_ops(ctx, ctx->decoder_backend, ctx->decoder_graph, "DAC");
    }
    if (!ggml_backend_sched_alloc_graph(ctx->predictor_sched, ctx->predictor_graph) ||
        !ggml_backend_sched_alloc_graph(ctx->decoder_sched, ctx->decoder_graph)) {
        std::fprintf(stderr, "sidon: graph allocation failed\n");
        clear_graphs(ctx);
        return false;
    }

    ctx->predictor_input = ggml_graph_get_tensor(ctx->predictor_graph, "sidon_features");
    ctx->relative_indices = ggml_graph_get_tensor(ctx->predictor_graph, "sidon_rel_indices");
    ctx->predictor_output = ggml_graph_get_tensor(ctx->predictor_graph, "sidon_predictor_output");
    ctx->decoder_input = ggml_graph_get_tensor(ctx->decoder_graph, "sidon_decoder_features");
    ctx->decoder_output = ggml_graph_get_tensor(ctx->decoder_graph, "dac_pcm");
    if (!ctx->predictor_input || !ctx->relative_indices || !ctx->predictor_output || !ctx->decoder_input ||
        !ctx->decoder_output) {
        std::fprintf(stderr, "sidon: stage graph is missing a required tensor\n");
        clear_graphs(ctx);
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
    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->predictor_vulkan = ci_starts_with(ggml_backend_name(ctx->backend), "Vulkan");
    // Keep stage execution and synchronization independent so predictor and
    // DAC timings describe their own CUDA work rather than a shared queue.
    // Vulkan shares one backend instance so decoder weights and the predictor
    // output never cross Vulkan queues/devices merely because the stages have
    // separate schedulers.
    ctx->decoder_backend =
        ctx->predictor_vulkan ? ctx->backend : (params.use_gpu ? crispasr_init_gpu_backend() : ggml_backend_cpu_init());
    if (!ctx->decoder_backend)
        ctx->decoder_backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    const int nt = params.n_threads > 0 ? params.n_threads : 4;
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, nt);
    if (ggml_backend_is_cpu(ctx->decoder_backend))
        ggml_backend_cpu_set_n_threads(ctx->decoder_backend, nt);
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, nt);
    if (!load_model(ctx->model, path, ctx->backend)) {
        sidon_free(ctx);
        return nullptr;
    }
    prepare_fastconv(ctx);
    ggml_backend_t predictor_bes[2];
    int predictor_nbe = 0;
    predictor_bes[predictor_nbe++] = ctx->backend;
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        predictor_bes[predictor_nbe++] = ctx->backend_cpu;
    ggml_backend_t decoder_bes[2];
    int decoder_nbe = 0;
    decoder_bes[decoder_nbe++] = ctx->decoder_backend;
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->decoder_backend)
        decoder_bes[decoder_nbe++] = ctx->backend_cpu;
    ctx->predictor_sched = ggml_backend_sched_new(predictor_bes, nullptr, predictor_nbe, 32768, false, false);
    ctx->decoder_sched = ggml_backend_sched_new(decoder_bes, nullptr, decoder_nbe, 32768, false, false);
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
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.buf_cpu)
        ggml_backend_buffer_free(ctx->model.buf_cpu);
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

std::vector<float> sidon_restore(sidon_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples < 400)
        return {};
    using clock = std::chrono::steady_clock;
    const auto total_start = clock::now();
    // Match the reference inference recipe's peak normalization.
    float peak = 0.0f;
    for (int i = 0; i < n_samples; ++i)
        peak = std::max(peak, std::fabs(samples[i]));
    std::vector<float> normalized((size_t)n_samples);
    const float gain = peak > 1e-9f ? 0.9f / peak : 1.0f;
    for (int i = 0; i < n_samples; ++i)
        normalized[(size_t)i] = samples[i] * gain;
    int T = 0;
    std::vector<float> feats = make_features(ctx->model, normalized.data(), n_samples, T);
    if (T <= 0)
        return {};
    const auto frontend_done = clock::now();

    if (!prepare_graphs(ctx, T))
        return {};
    const auto graph_done = clock::now();

    ggml_backend_tensor_set(ctx->predictor_input, feats.data(), 0, feats.size() * sizeof(float));
    std::vector<int32_t> indices((size_t)T * T);
    for (int q = 0; q < T; ++q)
        for (int k = 0; k < T; ++k)
            indices[(size_t)q * T + k] =
                std::max(-ctx->model.hp.rel_left, std::min(ctx->model.hp.rel_right, k - q)) + ctx->model.hp.rel_left;
    ggml_backend_tensor_set(ctx->relative_indices, indices.data(), 0, indices.size() * sizeof(int32_t));
    const auto predictor_start = clock::now();
    if (ggml_backend_sched_graph_compute(ctx->predictor_sched, ctx->predictor_graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "sidon: predictor graph compute failed\n");
        clear_graphs(ctx);
        return {};
    }
    ggml_backend_sched_synchronize(ctx->predictor_sched);
    const auto predictor_done = clock::now();

    ggml_backend_tensor_copy(ctx->predictor_output, ctx->decoder_input);
    const auto handoff_done = clock::now();
    if (ggml_backend_sched_graph_compute(ctx->decoder_sched, ctx->decoder_graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "sidon: decoder graph compute failed\n");
        clear_graphs(ctx);
        return {};
    }
    ggml_backend_sched_synchronize(ctx->decoder_sched);
    const auto decoder_done = clock::now();

    std::vector<float> pcm((size_t)ggml_nelements(ctx->decoder_output));
    ggml_backend_tensor_get(ctx->decoder_output, pcm.data(), 0, pcm.size() * sizeof(float));
    const auto download_done = clock::now();
    clear_graphs(ctx);
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
