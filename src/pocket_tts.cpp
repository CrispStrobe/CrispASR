// pocket_tts.cpp -- Kyutai Pocket TTS (100M, continuous-latent AR) runtime.
//
// Architecture overview (see pocket_tts.h for full description):
//   1. Tokenize text with SentencePiece -> embed via learned LUT (4001 x 1024)
//   2. Optionally prepend voice conditioning (Mimi VAE encode + project)
//   3. AR loop at 12.5 Hz:
//      a. Feed current latent (32-dim, NaN for BOS) through input_linear (32->1024)
//      b. Concatenate text embeddings + audio input, run through 6-layer
//         causal transformer (1024D, 16H, RoPE, pre-norm LN, GELU FFN)
//      c. Take last position output, check EOS via out_eos linear
//      d. Consistency head (SimpleMLPAdaLN):
//         - Sample noise from N(0, temp^0.5), optionally clamped
//         - LSD decode: iteratively apply flow_net(backbone_out, s, t, x)
//           where s,t are timestep scalars, x is current noise/latent
//         - flow_net = input_proj + ResBlocks(AdaLN) + FinalLayer(AdaLN)
//         - Each ResBlock: LN(x)*scale+shift -> MLP -> gate
//         - Conditioning: sum of two TimestepEmbedders + cond_embed(backbone_out)
//      e. Output: 32-dim continuous float vector (the next latent)
//   4. Denormalize latents: x * emb_std + emb_mean
//   5. Mimi VAE decoder:
//      a. DummyQuantizer Conv1d projection (32 -> 512)
//      b. Upsample (stride-16 transposed conv, 32->512)
//      c. Decoder transformer (2L, 512D, 8H, LayerScale, context=250)
//      d. SEANet decoder (512 -> 1ch, ratios [6,5,4], hop=120)
//   6. Output: 24 kHz mono PCM

#include "pocket_tts.h"

#include "core/gguf_loader.h"
#include "core/sentencepiece.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ── SentencePiece tokenizer ───────────────────────────────────────
// The GGUF stores tokenizer.ggml.tokens + tokenizer.ggml.scores arrays
// (unigram vocab extracted by the converter from the .model protobuf).
// At init we load these into token_to_id / scores and use the shared
// core_spm::tokenize Viterbi segmenter.

namespace {

// ── Hyperparameters ────────────────────────────────────────────────

struct pocket_tts_flow_lm_hp {
    uint32_t d_model = 1024;
    uint32_t num_heads = 16;
    uint32_t num_layers = 6;
    uint32_t hidden_scale = 4; // ff_dim = d_model * hidden_scale
    uint32_t max_period = 10000;
    uint32_t latent_dim = 32;
    uint32_t n_bins = 4000; // vocab size
    uint32_t lut_dim = 1024;
    uint32_t insert_bos_before_voice = 1;

    uint32_t head_dim() const { return d_model / num_heads; }
    uint32_t ff_dim() const { return d_model * hidden_scale; }
};

struct pocket_tts_flow_head_hp {
    uint32_t flow_dim = 512;
    uint32_t flow_depth = 6;
    uint32_t num_time_conds = 2;
    uint32_t freq_embed_size = 256;
};

struct pocket_tts_mimi_hp {
    uint32_t sample_rate = 24000;
    uint32_t frame_rate_num = 25;
    uint32_t frame_rate_den = 2;
    uint32_t inner_dim = 32;
    uint32_t outer_dim = 512;
    uint32_t channels = 1;

    // SEANet
    uint32_t seanet_dimension = 512;
    uint32_t seanet_n_filters = 64;
    uint32_t seanet_n_residual_layers = 1;
    uint32_t seanet_kernel_size = 7;
    uint32_t seanet_residual_kernel_size = 3;
    uint32_t seanet_last_kernel_size = 3;
    uint32_t seanet_dilation_base = 2;
    uint32_t seanet_compress = 2;
    std::vector<int> seanet_ratios = {6, 5, 4};

    // Transformer
    uint32_t xfmr_d_model = 512;
    uint32_t xfmr_num_heads = 8;
    uint32_t xfmr_num_layers = 2;
    uint32_t xfmr_dim_feedforward = 2048;
    uint32_t xfmr_context = 250;
    float xfmr_layer_scale_init = 0.01f;

    // Quantizer
    uint32_t quant_in_dim = 32;
    uint32_t quant_out_dim = 512;

    float frame_rate() const { return (float)frame_rate_num / frame_rate_den; }
    uint32_t hop_length() const {
        uint32_t h = 1;
        for (int r : seanet_ratios)
            h *= r;
        return h;
    }
    float encoder_frame_rate() const { return (float)sample_rate / hop_length(); }
    uint32_t downsample_stride() const { return (uint32_t)(encoder_frame_rate() / frame_rate()); }
};

// ── Transformer layer weights ──────────────────────────────────────

struct pocket_tts_transformer_layer {
    ggml_tensor* attn_norm_w = nullptr;   // LayerNorm weight
    ggml_tensor* attn_norm_b = nullptr;   // LayerNorm bias
    ggml_tensor* attn_in_proj = nullptr;  // (3*d_model, d_model) fused QKV
    ggml_tensor* attn_out_proj = nullptr; // (d_model, d_model)
    ggml_tensor* ffn_norm_w = nullptr;    // LayerNorm weight
    ggml_tensor* ffn_norm_b = nullptr;    // LayerNorm bias
    ggml_tensor* ffn_linear1 = nullptr;   // (ff_dim, d_model)
    ggml_tensor* ffn_linear2 = nullptr;   // (d_model, ff_dim)

    // Mimi transformer layers also have LayerScale
    ggml_tensor* layer_scale_1 = nullptr; // (d_model,) or null
    ggml_tensor* layer_scale_2 = nullptr; // (d_model,) or null
};

// ── Flow network (consistency head) weights ────────────────────────

struct pocket_tts_flow_resblock {
    ggml_tensor* ln_w = nullptr;          // LayerNorm weight
    ggml_tensor* ln_b = nullptr;          // LayerNorm bias
    ggml_tensor* mlp_linear1 = nullptr;   // (flow_dim, flow_dim)
    ggml_tensor* mlp_linear1_b = nullptr; // (flow_dim,)
    ggml_tensor* mlp_linear2 = nullptr;   // (flow_dim, flow_dim)
    ggml_tensor* mlp_linear2_b = nullptr; // (flow_dim,)
    ggml_tensor* ada_linear = nullptr;    // (3*flow_dim, flow_dim)
    ggml_tensor* ada_bias = nullptr;      // (3*flow_dim,)
};

struct pocket_tts_flow_net {
    ggml_tensor* input_proj = nullptr; // (flow_dim, latent_dim)
    ggml_tensor* input_proj_b = nullptr;
    ggml_tensor* cond_embed = nullptr; // (flow_dim, d_model)
    ggml_tensor* cond_embed_b = nullptr;

    // TimestepEmbedders (num_time_conds=2)
    // Each: linear1(freq_embed_size->flow_dim) + SiLU + linear2(flow_dim->flow_dim) + RMSNorm
    struct timestep_embedder {
        ggml_tensor* linear1_w = nullptr; // (flow_dim, freq_embed_size)
        ggml_tensor* linear1_b = nullptr;
        ggml_tensor* linear2_w = nullptr; // (flow_dim, flow_dim)
        ggml_tensor* linear2_b = nullptr;
        ggml_tensor* rms_alpha = nullptr; // (flow_dim,)
        ggml_tensor* freqs = nullptr;     // (freq_embed_size/2,)
    };
    std::vector<timestep_embedder> time_embeds;

    std::vector<pocket_tts_flow_resblock> res_blocks;

    // FinalLayer
    // norm_final is elementwise_affine=False, so no params
    ggml_tensor* final_linear = nullptr; // (latent_dim, flow_dim)
    ggml_tensor* final_linear_b = nullptr;
    ggml_tensor* final_ada = nullptr; // (2*flow_dim, flow_dim)
    ggml_tensor* final_ada_b = nullptr;
};

// ── SEANet decoder weights ─────────────────────────────────────────

struct seanet_resblock_weights {
    // Two conv layers per block: conv0 (dim/compress input, dim/compress out)
    //                           conv1 (dim/compress input, dim out)
    ggml_tensor* conv0_w = nullptr; // (out_ch, in_ch, kernel)
    ggml_tensor* conv0_b = nullptr;
    ggml_tensor* conv1_w = nullptr;
    ggml_tensor* conv1_b = nullptr;
};

struct seanet_decoder_stage {
    // Transposed conv for upsampling
    ggml_tensor* convtr_w = nullptr;
    ggml_tensor* convtr_b = nullptr;
    // Residual blocks
    std::vector<seanet_resblock_weights> resblocks;
};

struct seanet_decoder_weights {
    ggml_tensor* initial_conv_w = nullptr; // (mult*n_filters, dimension, kernel)
    ggml_tensor* initial_conv_b = nullptr;
    std::vector<seanet_decoder_stage> stages;
    ggml_tensor* final_conv_w = nullptr; // (channels, n_filters, kernel)
    ggml_tensor* final_conv_b = nullptr;
};

// ── SEANet encoder weights (for voice cloning) ─────────────────────

struct seanet_encoder_stage {
    std::vector<seanet_resblock_weights> resblocks;
    ggml_tensor* conv_w = nullptr; // downsampling conv
    ggml_tensor* conv_b = nullptr;
};

struct seanet_encoder_weights {
    ggml_tensor* initial_conv_w = nullptr;
    ggml_tensor* initial_conv_b = nullptr;
    std::vector<seanet_encoder_stage> stages;
    ggml_tensor* final_conv_w = nullptr;
    ggml_tensor* final_conv_b = nullptr;
};

// ── Full model ─────────────────────────────────────────────────────

struct pocket_tts_model {
    pocket_tts_flow_lm_hp flow_lm_hp;
    pocket_tts_flow_head_hp flow_head_hp;
    pocket_tts_mimi_hp mimi_hp;

    bool has_voice_cloning = false;

    // FlowLM
    ggml_tensor* conditioner_embed = nullptr; // (n_bins+1, lut_dim)
    ggml_tensor* input_linear = nullptr;      // (d_model, latent_dim)
    ggml_tensor* out_norm_w = nullptr;        // (d_model,)
    ggml_tensor* out_norm_b = nullptr;        // (d_model,)
    ggml_tensor* out_eos_w = nullptr;         // (1, d_model)
    ggml_tensor* out_eos_b = nullptr;         // (1,)
    ggml_tensor* bos_emb = nullptr;           // (latent_dim,)
    ggml_tensor* bos_before_voice = nullptr;  // (1, 1, d_model)
    ggml_tensor* emb_std = nullptr;           // (latent_dim,)
    ggml_tensor* emb_mean = nullptr;          // (latent_dim,)

    // Speaker projection (voice cloning)
    ggml_tensor* speaker_proj = nullptr; // (d_model, inner_dim)

    std::vector<pocket_tts_transformer_layer> backbone_layers;
    pocket_tts_flow_net flow_net;

    // Mimi decoder
    ggml_tensor* quant_proj_w = nullptr;    // (outer_dim, inner_dim, 1)
    ggml_tensor* upsample_conv_w = nullptr; // transposed conv for 32->512 upsample
    ggml_tensor* upsample_conv_b = nullptr;
    std::vector<pocket_tts_transformer_layer> dec_transformer_layers;
    ggml_tensor* dec_xfmr_input_proj = nullptr;  // if d_model != input_dim
    ggml_tensor* dec_xfmr_output_proj = nullptr; // if d_model != output_dim
    seanet_decoder_weights seanet_dec;

    // Mimi encoder (only if has_voice_cloning)
    ggml_tensor* downsample_conv_w = nullptr;
    ggml_tensor* downsample_conv_b = nullptr;
    std::vector<pocket_tts_transformer_layer> enc_transformer_layers;
    ggml_tensor* enc_xfmr_input_proj = nullptr;
    ggml_tensor* enc_xfmr_output_proj = nullptr;
    seanet_encoder_weights seanet_enc;

    // SentencePiece tokenizer (loaded from tokenizer.ggml.tokens/scores)
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<std::string> id_to_token;
    std::vector<float> spm_scores;
    int32_t unk_id = 0;
};

// ── KV cache for transformer ───────────────────────────────────────

struct pocket_tts_kv_cache {
    std::vector<float> k; // (n_layers, max_seq, n_heads, head_dim)
    std::vector<float> v;
    int max_seq = 0;
    int n_layers = 0;
    int n_heads = 0;
    int head_dim = 0;
    int offset = 0; // current write position
};

} // namespace

// ── Context ────────────────────────────────────────────────────────

struct pocket_tts_context {
    pocket_tts_context_params params;
    pocket_tts_model model;

    // GGML
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_type_t buf_type = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;

    // KV caches
    pocket_tts_kv_cache backbone_kv;
    pocket_tts_kv_cache dec_xfmr_kv; // Mimi decoder transformer

    // Voice conditioning state
    bool has_voice_state = false;
    std::vector<float> voice_conditioning; // (n_voice_frames, d_model)
    int n_voice_frames = 0;

    // RNG
    std::mt19937 rng;

    int verbosity = 1;
};

namespace { // reopen anonymous namespace for internal helpers

// ── Helpers ────────────────────────────────────────────────────────

static uint32_t gguf_get_u32(struct gguf_context* meta, const char* key, uint32_t def) {
    int idx = gguf_find_key(meta, key);
    if (idx < 0)
        return def;
    return (uint32_t)gguf_get_val_u32(meta, idx);
}

static float gguf_get_f32(struct gguf_context* meta, const char* key, float def) {
    int idx = gguf_find_key(meta, key);
    if (idx < 0)
        return def;
    return gguf_get_val_f32(meta, idx);
}

static std::string gguf_get_str(struct gguf_context* meta, const char* key, const char* def) {
    int idx = gguf_find_key(meta, key);
    if (idx < 0)
        return def;
    return gguf_get_val_str(meta, idx);
}

// ── Load model from GGUF ───────────────────────────────────────────

static bool load_hparams(struct gguf_context* meta, pocket_tts_model& m) {
    auto& h = m.flow_lm_hp;
    h.d_model = gguf_get_u32(meta, "pocket_tts.flow_lm.d_model", h.d_model);
    h.num_heads = gguf_get_u32(meta, "pocket_tts.flow_lm.num_heads", h.num_heads);
    h.num_layers = gguf_get_u32(meta, "pocket_tts.flow_lm.num_layers", h.num_layers);
    h.hidden_scale = gguf_get_u32(meta, "pocket_tts.flow_lm.hidden_scale", h.hidden_scale);
    h.max_period = gguf_get_u32(meta, "pocket_tts.flow_lm.max_period", h.max_period);
    h.latent_dim = gguf_get_u32(meta, "pocket_tts.flow_lm.latent_dim", h.latent_dim);
    h.n_bins = gguf_get_u32(meta, "pocket_tts.flow_lm.n_bins", h.n_bins);
    h.lut_dim = gguf_get_u32(meta, "pocket_tts.flow_lm.lut_dim", h.lut_dim);
    h.insert_bos_before_voice =
        gguf_get_u32(meta, "pocket_tts.flow_lm.insert_bos_before_voice", h.insert_bos_before_voice);

    auto& fh = m.flow_head_hp;
    fh.flow_dim = gguf_get_u32(meta, "pocket_tts.flow_head.flow_dim", fh.flow_dim);
    fh.flow_depth = gguf_get_u32(meta, "pocket_tts.flow_head.flow_depth", fh.flow_depth);
    fh.num_time_conds = gguf_get_u32(meta, "pocket_tts.flow_head.num_time_conds", fh.num_time_conds);
    fh.freq_embed_size = gguf_get_u32(meta, "pocket_tts.flow_head.freq_embed_size", fh.freq_embed_size);

    auto& mi = m.mimi_hp;
    mi.sample_rate = gguf_get_u32(meta, "pocket_tts.mimi.sample_rate", mi.sample_rate);
    mi.frame_rate_num = gguf_get_u32(meta, "pocket_tts.mimi.frame_rate_num", mi.frame_rate_num);
    mi.frame_rate_den = gguf_get_u32(meta, "pocket_tts.mimi.frame_rate_den", mi.frame_rate_den);
    mi.inner_dim = gguf_get_u32(meta, "pocket_tts.mimi.inner_dim", mi.inner_dim);
    mi.outer_dim = gguf_get_u32(meta, "pocket_tts.mimi.outer_dim", mi.outer_dim);
    mi.channels = gguf_get_u32(meta, "pocket_tts.mimi.channels", mi.channels);

    mi.seanet_dimension = gguf_get_u32(meta, "pocket_tts.mimi.seanet_dimension", mi.seanet_dimension);
    mi.seanet_n_filters = gguf_get_u32(meta, "pocket_tts.mimi.seanet_n_filters", mi.seanet_n_filters);
    mi.seanet_n_residual_layers =
        gguf_get_u32(meta, "pocket_tts.mimi.seanet_n_residual_layers", mi.seanet_n_residual_layers);
    mi.seanet_kernel_size = gguf_get_u32(meta, "pocket_tts.mimi.seanet_kernel_size", mi.seanet_kernel_size);
    mi.seanet_residual_kernel_size =
        gguf_get_u32(meta, "pocket_tts.mimi.seanet_residual_kernel_size", mi.seanet_residual_kernel_size);
    mi.seanet_last_kernel_size =
        gguf_get_u32(meta, "pocket_tts.mimi.seanet_last_kernel_size", mi.seanet_last_kernel_size);
    mi.seanet_dilation_base = gguf_get_u32(meta, "pocket_tts.mimi.seanet_dilation_base", mi.seanet_dilation_base);
    mi.seanet_compress = gguf_get_u32(meta, "pocket_tts.mimi.seanet_compress", mi.seanet_compress);

    mi.xfmr_d_model = gguf_get_u32(meta, "pocket_tts.mimi.xfmr_d_model", mi.xfmr_d_model);
    mi.xfmr_num_heads = gguf_get_u32(meta, "pocket_tts.mimi.xfmr_num_heads", mi.xfmr_num_heads);
    mi.xfmr_num_layers = gguf_get_u32(meta, "pocket_tts.mimi.xfmr_num_layers", mi.xfmr_num_layers);
    mi.xfmr_dim_feedforward = gguf_get_u32(meta, "pocket_tts.mimi.xfmr_dim_feedforward", mi.xfmr_dim_feedforward);
    mi.xfmr_context = gguf_get_u32(meta, "pocket_tts.mimi.xfmr_context", mi.xfmr_context);
    mi.xfmr_layer_scale_init = gguf_get_f32(meta, "pocket_tts.mimi.xfmr_layer_scale_init", mi.xfmr_layer_scale_init);

    mi.quant_in_dim = gguf_get_u32(meta, "pocket_tts.mimi.quant_in_dim", mi.quant_in_dim);
    mi.quant_out_dim = gguf_get_u32(meta, "pocket_tts.mimi.quant_out_dim", mi.quant_out_dim);

    // SEANet ratios from array KV
    int ratios_idx = gguf_find_key(meta, "pocket_tts.mimi.seanet_ratios");
    if (ratios_idx >= 0) {
        enum gguf_type arr_type = gguf_get_arr_type(meta, ratios_idx);
        size_t arr_n = gguf_get_arr_n(meta, ratios_idx);
        if (arr_type == GGUF_TYPE_UINT32 && arr_n > 0) {
            mi.seanet_ratios.clear();
            for (size_t i = 0; i < arr_n; i++) {
                mi.seanet_ratios.push_back((int)((const uint32_t*)gguf_get_arr_data(meta, ratios_idx))[i]);
            }
        } else if (arr_type == GGUF_TYPE_INT32 && arr_n > 0) {
            mi.seanet_ratios.clear();
            for (size_t i = 0; i < arr_n; i++) {
                mi.seanet_ratios.push_back((int)((const int32_t*)gguf_get_arr_data(meta, ratios_idx))[i]);
            }
        }
        // Otherwise keep default [6, 5, 4]
    }

    m.has_voice_cloning = gguf_get_u32(meta, "pocket_tts.has_voice_cloning", 0) != 0;

    return true;
}

// ── Tensor loading ─────────────────────────────────────────────────

static ggml_tensor* try_get_tensor(struct ggml_context* ctx, const char* name) {
    return ggml_get_tensor(ctx, name);
}

static bool load_flow_lm_tensors(struct ggml_context* ctx, pocket_tts_model& m) {
    const auto& h = m.flow_lm_hp;

    m.conditioner_embed = try_get_tensor(ctx, "flow_lm.conditioner.embed.weight");
    m.input_linear = try_get_tensor(ctx, "flow_lm.input_linear.weight");
    m.out_norm_w = try_get_tensor(ctx, "flow_lm.out_norm.weight");
    m.out_norm_b = try_get_tensor(ctx, "flow_lm.out_norm.bias");
    m.out_eos_w = try_get_tensor(ctx, "flow_lm.out_eos.weight");
    m.out_eos_b = try_get_tensor(ctx, "flow_lm.out_eos.bias");
    m.bos_emb = try_get_tensor(ctx, "flow_lm.bos_emb");
    m.bos_before_voice = try_get_tensor(ctx, "flow_lm.bos_before_voice");
    m.emb_std = try_get_tensor(ctx, "flow_lm.emb_std");
    m.emb_mean = try_get_tensor(ctx, "flow_lm.emb_mean");
    m.speaker_proj = try_get_tensor(ctx, "flow_lm.speaker_proj.weight");

    // Backbone transformer layers
    m.backbone_layers.resize(h.num_layers);
    for (uint32_t i = 0; i < h.num_layers; i++) {
        auto& L = m.backbone_layers[i];
        char buf[256];

        snprintf(buf, sizeof(buf), "flow_lm.transformer.%u.norm1.weight", i);
        L.attn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.%u.norm1.bias", i);
        L.attn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.%u.self_attn.in_proj.weight", i);
        L.attn_in_proj = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.%u.self_attn.out_proj.weight", i);
        L.attn_out_proj = try_get_tensor(ctx, buf);

        snprintf(buf, sizeof(buf), "flow_lm.transformer.%u.norm2.weight", i);
        L.ffn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.%u.norm2.bias", i);
        L.ffn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.%u.linear1.weight", i);
        L.ffn_linear1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.%u.linear2.weight", i);
        L.ffn_linear2 = try_get_tensor(ctx, buf);
    }

    // Flow network (consistency head)
    auto& fn = m.flow_net;
    fn.input_proj = try_get_tensor(ctx, "flow_lm.flow_net.input_proj.weight");
    fn.input_proj_b = try_get_tensor(ctx, "flow_lm.flow_net.input_proj.bias");
    fn.cond_embed = try_get_tensor(ctx, "flow_lm.flow_net.cond_embed.weight");
    fn.cond_embed_b = try_get_tensor(ctx, "flow_lm.flow_net.cond_embed.bias");

    fn.time_embeds.resize(m.flow_head_hp.num_time_conds);
    for (uint32_t t = 0; t < m.flow_head_hp.num_time_conds; t++) {
        auto& te = fn.time_embeds[t];
        char buf[256];
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.mlp.0.weight", t);
        te.linear1_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.mlp.0.bias", t);
        te.linear1_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.mlp.2.weight", t);
        te.linear2_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.mlp.2.bias", t);
        te.linear2_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.mlp.3.alpha", t);
        te.rms_alpha = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.freqs", t);
        te.freqs = try_get_tensor(ctx, buf);
    }

    fn.res_blocks.resize(m.flow_head_hp.flow_depth);
    for (uint32_t i = 0; i < m.flow_head_hp.flow_depth; i++) {
        auto& rb = fn.res_blocks[i];
        char buf[256];
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.in_ln.weight", i);
        rb.ln_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.in_ln.bias", i);
        rb.ln_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.mlp.0.weight", i);
        rb.mlp_linear1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.mlp.0.bias", i);
        rb.mlp_linear1_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.mlp.2.weight", i);
        rb.mlp_linear2 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.mlp.2.bias", i);
        rb.mlp_linear2_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.adaLN_modulation.1.weight", i);
        rb.ada_linear = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.adaLN_modulation.1.bias", i);
        rb.ada_bias = try_get_tensor(ctx, buf);
    }

    fn.final_linear = try_get_tensor(ctx, "flow_lm.flow_net.final_layer.linear.weight");
    fn.final_linear_b = try_get_tensor(ctx, "flow_lm.flow_net.final_layer.linear.bias");
    fn.final_ada = try_get_tensor(ctx, "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight");
    fn.final_ada_b = try_get_tensor(ctx, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias");

    return m.conditioner_embed != nullptr && m.input_linear != nullptr;
}

static bool load_mimi_decoder_tensors(struct ggml_context* ctx, pocket_tts_model& m) {
    const auto& mi = m.mimi_hp;

    // Quantizer projection (Conv1d, kernel=1)
    m.quant_proj_w = try_get_tensor(ctx, "mimi.quantizer.output_proj.weight");

    // Upsample conv (transposed, stride=downsample_stride)
    m.upsample_conv_w = try_get_tensor(ctx, "mimi.upsample.convtr.weight");
    if (!m.upsample_conv_w)
        m.upsample_conv_w = try_get_tensor(ctx, "mimi.upsample.conv.weight");
    m.upsample_conv_b = try_get_tensor(ctx, "mimi.upsample.convtr.bias");
    if (!m.upsample_conv_b)
        m.upsample_conv_b = try_get_tensor(ctx, "mimi.upsample.conv.bias");

    // Decoder transformer
    m.dec_xfmr_input_proj = try_get_tensor(ctx, "mimi.decoder_transformer.input_proj.weight");
    m.dec_xfmr_output_proj = try_get_tensor(ctx, "mimi.decoder_transformer.output_projs.0.weight");

    m.dec_transformer_layers.resize(mi.xfmr_num_layers);
    for (uint32_t i = 0; i < mi.xfmr_num_layers; i++) {
        auto& L = m.dec_transformer_layers[i];
        char buf[256];
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.norm1.weight", i);
        L.attn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.norm1.bias", i);
        L.attn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.self_attn.in_proj.weight", i);
        L.attn_in_proj = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.self_attn.out_proj.weight", i);
        L.attn_out_proj = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.norm2.weight", i);
        L.ffn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.norm2.bias", i);
        L.ffn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.linear1.weight", i);
        L.ffn_linear1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.linear2.weight", i);
        L.ffn_linear2 = try_get_tensor(ctx, buf);

        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.layer_scale_1.scale", i);
        L.layer_scale_1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.layer_scale_2.scale", i);
        L.layer_scale_2 = try_get_tensor(ctx, buf);
    }

    // SEANet decoder
    auto& sd = m.seanet_dec;
    sd.initial_conv_w = try_get_tensor(ctx, "mimi.decoder.model.0.conv.weight");
    sd.initial_conv_b = try_get_tensor(ctx, "mimi.decoder.model.0.conv.bias");

    // Decoder stages: for each ratio, there's an upsample + residual blocks
    // Layout in model list: [initial_conv, (ELU, ConvTr, ResBlock*n_res)*n_ratios, ELU, final_conv]
    const auto& ratios = mi.seanet_ratios;
    sd.stages.resize(ratios.size());

    // The decoder model list has:
    // [0] = initial conv
    // Then for each ratio i (0..n_ratios-1):
    //   [1 + i*(1+n_res+1)] = ELU
    //   [1 + i*(1+n_res+1) + 1] = ConvTranspose (upsample)
    //   [1 + i*(1+n_res+1) + 2 .. + 2+n_res-1] = ResBlocks
    // [last-1] = ELU
    // [last]   = final conv
    //
    // In practice with n_res=1:
    //   model[0] = conv, model[1] = ELU, model[2] = ConvTr,
    //   model[3] = ResBlock, model[4] = ELU, model[5] = ConvTr, ...
    // etc. Let's compute indices.

    uint32_t n_res = mi.seanet_n_residual_layers;
    uint32_t idx = 1; // after initial conv
    for (size_t s = 0; s < ratios.size(); s++) {
        auto& stage = sd.stages[s];
        // ELU at idx, skip
        idx++; // ELU
        // ConvTranspose at idx
        char buf[256];
        snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.convtr.weight", idx);
        stage.convtr_w = try_get_tensor(ctx, buf);
        if (!stage.convtr_w) {
            snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.conv.weight", idx);
            stage.convtr_w = try_get_tensor(ctx, buf);
        }
        snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.convtr.bias", idx);
        stage.convtr_b = try_get_tensor(ctx, buf);
        if (!stage.convtr_b) {
            snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.conv.bias", idx);
            stage.convtr_b = try_get_tensor(ctx, buf);
        }
        idx++; // ConvTr

        // Residual blocks
        stage.resblocks.resize(n_res);
        for (uint32_t r = 0; r < n_res; r++) {
            auto& rb = stage.resblocks[r];
            // Each resblock has a model list: [ELU, Conv, ELU, Conv]
            // block[0] = ELU, block[1] = Conv (dim->hidden), block[2] = ELU, block[3] = Conv (hidden->dim)
            snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.block.1.conv.weight", idx);
            rb.conv0_w = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.block.1.conv.bias", idx);
            rb.conv0_b = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.block.3.conv.weight", idx);
            rb.conv1_w = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.block.3.conv.bias", idx);
            rb.conv1_b = try_get_tensor(ctx, buf);
            idx++; // ResBlock
        }
    }

    // ELU + final conv
    idx++; // ELU
    char buf[256];
    snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.conv.weight", idx);
    sd.final_conv_w = try_get_tensor(ctx, buf);
    snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.conv.bias", idx);
    sd.final_conv_b = try_get_tensor(ctx, buf);

    return true;
}

static bool load_mimi_encoder_tensors(struct ggml_context* ctx, pocket_tts_model& m) {
    if (!m.has_voice_cloning)
        return true;

    const auto& mi = m.mimi_hp;

    m.downsample_conv_w = try_get_tensor(ctx, "mimi.downsample.conv.weight");
    m.downsample_conv_b = try_get_tensor(ctx, "mimi.downsample.conv.bias");

    m.enc_xfmr_input_proj = try_get_tensor(ctx, "mimi.encoder_transformer.input_proj.weight");
    m.enc_xfmr_output_proj = try_get_tensor(ctx, "mimi.encoder_transformer.output_projs.0.weight");

    m.enc_transformer_layers.resize(mi.xfmr_num_layers);
    for (uint32_t i = 0; i < mi.xfmr_num_layers; i++) {
        auto& L = m.enc_transformer_layers[i];
        char buf[256];
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.norm1.weight", i);
        L.attn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.norm1.bias", i);
        L.attn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.self_attn.in_proj.weight", i);
        L.attn_in_proj = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.self_attn.out_proj.weight", i);
        L.attn_out_proj = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.norm2.weight", i);
        L.ffn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.norm2.bias", i);
        L.ffn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.linear1.weight", i);
        L.ffn_linear1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.linear2.weight", i);
        L.ffn_linear2 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.layer_scale_1.scale", i);
        L.layer_scale_1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.layer_scale_2.scale", i);
        L.layer_scale_2 = try_get_tensor(ctx, buf);
    }

    // SEANet encoder
    auto& se = m.seanet_enc;
    se.initial_conv_w = try_get_tensor(ctx, "mimi.encoder.model.0.conv.weight");
    se.initial_conv_b = try_get_tensor(ctx, "mimi.encoder.model.0.conv.bias");

    // Encoder: ratios are reversed (compared to decoder)
    std::vector<int> enc_ratios(mi.seanet_ratios.rbegin(), mi.seanet_ratios.rend());
    uint32_t n_res = mi.seanet_n_residual_layers;
    se.stages.resize(enc_ratios.size());

    uint32_t idx = 1; // after initial conv
    for (size_t s = 0; s < enc_ratios.size(); s++) {
        auto& stage = se.stages[s];

        // Residual blocks first in encoder
        stage.resblocks.resize(n_res);
        for (uint32_t r = 0; r < n_res; r++) {
            auto& rb = stage.resblocks[r];
            char buf[256];
            snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.block.1.conv.weight", idx);
            rb.conv0_w = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.block.1.conv.bias", idx);
            rb.conv0_b = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.block.3.conv.weight", idx);
            rb.conv1_w = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.block.3.conv.bias", idx);
            rb.conv1_b = try_get_tensor(ctx, buf);
            idx++;
        }

        // ELU + downsampling conv
        idx++; // ELU
        char buf[256];
        snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.conv.weight", idx);
        stage.conv_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.conv.bias", idx);
        stage.conv_b = try_get_tensor(ctx, buf);
        idx++;
    }

    // ELU + final conv
    idx++; // ELU
    char buf[256];
    snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.conv.weight", idx);
    se.final_conv_w = try_get_tensor(ctx, buf);
    snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.conv.bias", idx);
    se.final_conv_b = try_get_tensor(ctx, buf);

    return true;
}

// ── SentencePiece tokenizer loading ───────────────────────────────
// Load tokenizer.ggml.tokens + tokenizer.ggml.scores arrays from GGUF
// (standard GGUF vocab pattern, written by the converter).

static bool load_tokenizer(struct gguf_context* meta, struct ggml_context* /*ggml_ctx*/, pocket_tts_model& m) {
    int tidx = gguf_find_key(meta, "tokenizer.ggml.tokens");
    if (tidx < 0) {
        fprintf(stderr, "pocket_tts: warning: no tokenizer.ggml.tokens in GGUF\n");
        return false;
    }
    int n = (int)gguf_get_arr_n(meta, tidx);
    m.id_to_token.resize(n);
    for (int i = 0; i < n; i++) {
        m.id_to_token[i] = gguf_get_arr_str(meta, tidx, i);
        m.token_to_id[m.id_to_token[i]] = i;
    }

    int sidx = gguf_find_key(meta, "tokenizer.ggml.scores");
    if (sidx >= 0) {
        int ns = (int)gguf_get_arr_n(meta, sidx);
        m.spm_scores.resize(ns);
        const float* sp = (const float*)gguf_get_arr_data(meta, sidx);
        for (int i = 0; i < ns; i++)
            m.spm_scores[i] = sp[i];
    }

    // Find <unk> token ID
    auto it = m.token_to_id.find("<unk>");
    m.unk_id = (it != m.token_to_id.end()) ? it->second : 0;

    return n > 0;
}

// ── Forward pass building blocks ───────────────────────────────────
//
// Eager CPU implementation for the AR loop (backbone + flow head) since
// each step depends on the previous output. The Mimi decoder uses a
// single ggml graph for the full sequence.

// ── Eager math helpers ────────────────────────────────────────────

// F16-to-F32 dequantization cache: avoids repeated conversion.
// We store dequantized buffers keyed by tensor data pointer.
static std::unordered_map<const void*, std::vector<float>> g_f16_cache;

static inline float* tensor_f32_data(ggml_tensor* t) {
    if (!t)
        return nullptr;
    if (t->type == GGML_TYPE_F32) {
        return (float*)t->data;
    }
    // F16: dequantize to cache
    auto it = g_f16_cache.find(t->data);
    if (it != g_f16_cache.end()) {
        return it->second.data();
    }
    int64_t n = ggml_nelements(t);
    auto& buf = g_f16_cache[t->data];
    buf.resize(n);
    if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)t->data, buf.data(), n);
    } else {
        // Unsupported type — zero-fill as fallback
        memset(buf.data(), 0, n * sizeof(float));
    }
    return buf.data();
}

static inline void vec_copy(float* dst, const float* src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

static inline void vec_add(float* dst, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = a[i] + b[i];
}

static inline void vec_mul(float* dst, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = a[i] * b[i];
}

static inline void vec_scale(float* dst, const float* src, float s, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = src[i] * s;
}

static inline void vec_fma(float* dst, const float* x, float a, int n) {
    for (int i = 0; i < n; i++)
        dst[i] += x[i] * a;
}

static inline float vec_dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}

static inline void vec_zero(float* dst, int n) {
    memset(dst, 0, n * sizeof(float));
}

static inline float silu_f(float x) {
    return x / (1.0f + std::exp(-x));
}

static inline float gelu_f(float x) {
    // tanh approximation GELU (matches PyTorch nn.GELU() default)
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// LayerNorm: affine or non-affine
static void layer_norm(float* out, const float* x, int dim, const float* w, const float* b, float eps = 1e-5f) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++)
        mean += x[i];
    mean /= dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= dim;
    float inv_std = 1.0f / std::sqrt(var + eps);
    if (w && b) {
        for (int i = 0; i < dim; i++)
            out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
    } else if (w) {
        for (int i = 0; i < dim; i++)
            out[i] = (x[i] - mean) * inv_std * w[i];
    } else {
        for (int i = 0; i < dim; i++)
            out[i] = (x[i] - mean) * inv_std;
    }
}

// Pocket-tts "RMSNorm": x * alpha / sqrt(var(x) + eps).
// NOTE: despite the name, this uses variance (mean-subtracted), NOT mean(x²),
// matching the reference _rms_norm which calls torch.var(dim=-1, correction=1).
static void rms_norm(float* out, const float* x, const float* alpha, int dim, float eps = 1e-5f) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++)
        mean += x[i];
    mean /= dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (dim - 1); // Bessel's correction (torch default)
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < dim; i++)
        out[i] = x[i] * inv_std * alpha[i];
}

// Linear: out = W @ x + b   (W is row-major: [out_dim, in_dim])
static void linear_f32(float* out, const float* x, ggml_tensor* W, ggml_tensor* bias, int out_dim, int in_dim) {
    const float* w_data = tensor_f32_data(W);
    for (int o = 0; o < out_dim; o++) {
        out[o] = vec_dot(&w_data[o * in_dim], x, in_dim);
    }
    if (bias) {
        const float* b_data = tensor_f32_data(bias);
        for (int o = 0; o < out_dim; o++)
            out[o] += b_data[o];
    }
}

// RoPE: apply rotary embedding to a single vector (head_dim,) at position pos.
// Uses interleaved complex pairs: vec[2i]=real, vec[2i+1]=imag
// (matching the Kyutai/Moshi convention used by both FlowLM and Mimi).
// freq[i] = exp(-i / (D/2) * ln(max_period)) = max_period^(-2i/D)
static void apply_rope_inplace(float* vec, int head_dim, int pos, float max_period) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        double freq = std::exp(-(double)i / half * std::log((double)max_period));
        double angle = (double)pos * freq;
        float cos_val = (float)std::cos(angle);
        float sin_val = (float)std::sin(angle);
        float xr = vec[2 * i];     // real
        float xi = vec[2 * i + 1]; // imag
        vec[2 * i] = xr * cos_val - xi * sin_val;
        vec[2 * i + 1] = xr * sin_val + xi * cos_val;
    }
}

// ── KV cache management ───────────────────────────────────────────

static void kv_cache_init(pocket_tts_kv_cache& kv, int n_layers, int max_seq, int n_heads, int head_dim) {
    kv.n_layers = n_layers;
    kv.max_seq = max_seq;
    kv.n_heads = n_heads;
    kv.head_dim = head_dim;
    kv.offset = 0;
    size_t total = (size_t)n_layers * max_seq * n_heads * head_dim;
    kv.k.resize(total, 0.0f);
    kv.v.resize(total, 0.0f);
}

static void kv_cache_reset(pocket_tts_kv_cache& kv) {
    kv.offset = 0;
}

// Get pointer to K[layer][pos][head][0]
static float* kv_k_ptr(pocket_tts_kv_cache& kv, int layer, int pos) {
    size_t stride_layer = (size_t)kv.max_seq * kv.n_heads * kv.head_dim;
    size_t stride_pos = (size_t)kv.n_heads * kv.head_dim;
    return &kv.k[layer * stride_layer + pos * stride_pos];
}

static float* kv_v_ptr(pocket_tts_kv_cache& kv, int layer, int pos) {
    size_t stride_layer = (size_t)kv.max_seq * kv.n_heads * kv.head_dim;
    size_t stride_pos = (size_t)kv.n_heads * kv.head_dim;
    return &kv.v[layer * stride_layer + pos * stride_pos];
}

// ── Backbone forward (ggml graph, single AR step) ────────────────

// Build and compute a ggml graph for one backbone step.
// Takes a single input embedding x_in (D,), attends to all past KV entries,
// writes new K/V to the host-side cache, and returns the LayerNormed output.
static void backbone_forward_step_ggml(pocket_tts_context* pctx, const float* x_in, float* out) {
    const auto& m = pctx->model;
    const auto& hp = m.flow_lm_hp;
    const int D = (int)hp.d_model;
    const int NH = (int)hp.num_heads;
    const int HD = (int)hp.head_dim();
    const int FF = (int)hp.ff_dim();
    const int NL = (int)hp.num_layers;
    const int pos = pctx->backbone_kv.offset;
    const int seq_len = pos + 1; // attend over [0..pos] (including current)

    // Build graph
    const size_t graph_nodes = 4096;
    const size_t buf_size = ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false);
    std::vector<uint8_t> compute_meta(buf_size);

    struct ggml_init_params gp = {compute_meta.size(), compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gp);

    // Input: single token embedding (D,) -> (D, 1) for matmul
    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, 1);
    ggml_set_name(inp, "x_in");
    ggml_set_input(inp);

    // Position tensor for RoPE (single position)
    ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(pos_tensor, "pos");
    ggml_set_input(pos_tensor);

    // Past K/V tensors per layer: (HD, pos, NH) — only if pos > 0
    // We'll create input tensors for past K/V and output tensors for new K/V
    struct layer_kv_io {
        ggml_tensor* past_k; // (HD, pos, NH) or nullptr if pos==0
        ggml_tensor* past_v;
        ggml_tensor* new_k; // (HD, 1, NH) — output
        ggml_tensor* new_v;
    };
    std::vector<layer_kv_io> kv_io(NL);

    for (int l = 0; l < NL; l++) {
        char name[64];
        if (pos > 0) {
            snprintf(name, sizeof(name), "past_k_%d", l);
            kv_io[l].past_k = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, HD, pos, NH);
            ggml_set_name(kv_io[l].past_k, name);
            ggml_set_input(kv_io[l].past_k);

            snprintf(name, sizeof(name), "past_v_%d", l);
            kv_io[l].past_v = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, HD, pos, NH);
            ggml_set_name(kv_io[l].past_v, name);
            ggml_set_input(kv_io[l].past_v);
        } else {
            kv_io[l].past_k = nullptr;
            kv_io[l].past_v = nullptr;
        }
    }

    ggml_tensor* x = inp; // (D, 1)

    for (int l = 0; l < NL; l++) {
        const auto& L = m.backbone_layers[l];
        ggml_tensor* residual = x;

        // Pre-norm LayerNorm
        ggml_tensor* h = ggml_norm(ctx0, x, 1e-5f);
        if (L.attn_norm_w)
            h = ggml_mul(ctx0, h, L.attn_norm_w);
        if (L.attn_norm_b)
            h = ggml_add(ctx0, h, L.attn_norm_b);

        // QKV: (3*D, D) @ (D, 1) -> (3*D, 1)
        ggml_tensor* qkv = ggml_mul_mat(ctx0, L.attn_in_proj, h);

        // Split Q, K, V: each (D, 1)
        ggml_tensor* Q = ggml_view_2d(ctx0, qkv, D, 1, qkv->nb[1], 0);
        ggml_tensor* K_cur = ggml_view_2d(ctx0, qkv, D, 1, qkv->nb[1], (size_t)D * ggml_type_size(qkv->type));
        ggml_tensor* V_cur = ggml_view_2d(ctx0, qkv, D, 1, qkv->nb[1], (size_t)2 * D * ggml_type_size(qkv->type));

        // Reshape to (HD, NH, 1) for RoPE
        Q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Q), HD, NH, 1);
        K_cur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, K_cur), HD, NH, 1);
        V_cur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, V_cur), HD, NH, 1);

        // RoPE (backbone uses max_period which may differ from 10000)
        Q = ggml_rope_ext(ctx0, Q, pos_tensor, nullptr, HD, GGML_ROPE_TYPE_NORMAL, 0, (float)hp.max_period, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
        K_cur = ggml_rope_ext(ctx0, K_cur, pos_tensor, nullptr, HD, GGML_ROPE_TYPE_NORMAL, 0, (float)hp.max_period,
                              1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // After RoPE: Q, K_cur, V_cur are (HD, NH, 1).
        // Permute to (HD, 1, NH) for concat with past KV (HD, pos, NH).
        K_cur = ggml_cont(ctx0, ggml_permute(ctx0, K_cur, 0, 2, 1, 3)); // (HD, 1, NH)
        V_cur = ggml_cont(ctx0, ggml_permute(ctx0, V_cur, 0, 2, 1, 3));

        // Mark new K/V as outputs (after permute, layout = (HD, 1, NH))
        char name[64];
        snprintf(name, sizeof(name), "new_k_%d", l);
        kv_io[l].new_k = ggml_cont(ctx0, K_cur);
        ggml_set_name(kv_io[l].new_k, name);
        ggml_set_output(kv_io[l].new_k);

        snprintf(name, sizeof(name), "new_v_%d", l);
        kv_io[l].new_v = ggml_cont(ctx0, V_cur);
        ggml_set_name(kv_io[l].new_v, name);
        ggml_set_output(kv_io[l].new_v);

        // Concat past K/V with current for attention
        ggml_tensor* K_full;
        ggml_tensor* V_full;
        if (pos > 0) {
            K_full = ggml_concat(ctx0, kv_io[l].past_k, K_cur, 1); // (HD, seq_len, NH)
            V_full = ggml_concat(ctx0, kv_io[l].past_v, V_cur, 1);
        } else {
            K_full = K_cur; // (HD, 1, NH)
            V_full = V_cur;
        }

        // Permute Q for flash_attn: (HD, NH, 1) -> (HD, 1, NH)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // (HD, 1, NH)
        // K_full, V_full already in (HD, T, NH) layout

        // Causal attention (no mask needed — Q has T=1, so all KV positions are attended)
        float scale = 1.0f / sqrtf((float)HD);
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K_full, V_full, nullptr, scale, 0.0f, 0.0f);

        // Output: (HD, NH, 1) -> (D, 1)
        attn = ggml_reshape_2d(ctx0, attn, D, 1);

        // Output projection
        attn = ggml_mul_mat(ctx0, L.attn_out_proj, attn);

        x = ggml_add(ctx0, residual, attn);

        // FFN
        residual = x;
        h = ggml_norm(ctx0, x, 1e-5f);
        if (L.ffn_norm_w)
            h = ggml_mul(ctx0, h, L.ffn_norm_w);
        if (L.ffn_norm_b)
            h = ggml_add(ctx0, h, L.ffn_norm_b);

        h = ggml_mul_mat(ctx0, L.ffn_linear1, h); // (FF, 1)
        h = ggml_gelu(ctx0, h);
        h = ggml_mul_mat(ctx0, L.ffn_linear2, h); // (D, 1)

        x = ggml_add(ctx0, residual, h);
    }

    // Final LayerNorm
    x = ggml_norm(ctx0, x, 1e-5f);
    if (m.out_norm_w)
        x = ggml_mul(ctx0, x, m.out_norm_w);
    if (m.out_norm_b)
        x = ggml_add(ctx0, x, m.out_norm_b);

    ggml_set_name(x, "backbone_out");
    ggml_set_output(x);

    // Build graph — expand main output + all new_k/new_v outputs
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, graph_nodes, false);
    ggml_build_forward_expand(gf, x);
    for (int l = 0; l < NL; l++) {
        ggml_build_forward_expand(gf, kv_io[l].new_k);
        ggml_build_forward_expand(gf, kv_io[l].new_v);
    }

    // Allocate
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(pctx->backend_cpu));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "pocket_tts: backbone_forward_step_ggml: alloc failed\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        memset(out, 0, D * sizeof(float));
        return;
    }

    // Set inputs
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), x_in, 0, D * sizeof(float));
    int32_t pos_val = pos;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), &pos_val, 0, sizeof(int32_t));

    // Upload past K/V from host cache
    auto& kv = pctx->backbone_kv;
    for (int l = 0; l < NL; l++) {
        if (pos > 0) {
            // Host cache layout: k[layer * max_seq * NH * HD + pos * NH * HD + head * HD + d]
            // We need (HD, pos, NH) for ggml: ne[0]=HD contiguous, then pos, then NH.
            // Host cache has: for each pos, NH*HD contiguous = (NH, HD) per pos.
            // ggml wants: for each head, pos*HD contiguous, then next head.
            // So we need to transpose from (pos, NH, HD) to (HD, pos, NH).
            // Actually host layout per layer is: [pos][head][d] = stride pos = NH*HD.
            // ggml (HD, pos, NH): ne[0]=HD, ne[1]=pos, ne[2]=NH.
            // Data layout: for h in NH, for p in pos, for d in HD: data[h*pos*HD + p*HD + d]
            // Host layout: for p in pos, for h in NH, for d in HD: data[p*NH*HD + h*HD + d]
            // These differ — need to reorder.
            size_t stride_layer = (size_t)kv.max_seq * NH * HD;
            std::vector<float> k_reorder((size_t)NH * pos * HD);
            std::vector<float> v_reorder((size_t)NH * pos * HD);
            for (int h = 0; h < NH; h++) {
                for (int p = 0; p < pos; p++) {
                    for (int d = 0; d < HD; d++) {
                        k_reorder[(size_t)h * pos * HD + p * HD + d] =
                            kv.k[l * stride_layer + (size_t)p * NH * HD + h * HD + d];
                        v_reorder[(size_t)h * pos * HD + p * HD + d] =
                            kv.v[l * stride_layer + (size_t)p * NH * HD + h * HD + d];
                    }
                }
            }
            char name[64];
            snprintf(name, sizeof(name), "past_k_%d", l);
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, name), k_reorder.data(), 0,
                                    k_reorder.size() * sizeof(float));
            snprintf(name, sizeof(name), "past_v_%d", l);
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, name), v_reorder.data(), 0,
                                    v_reorder.size() * sizeof(float));
        }
    }

    // Compute
    int n_threads = pctx->params.n_threads > 0 ? pctx->params.n_threads : 4;
    ggml_backend_cpu_set_n_threads(pctx->backend_cpu, n_threads);
    ggml_backend_graph_compute(pctx->backend_cpu, gf);

    // Read backbone output
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "backbone_out"), out, 0, D * sizeof(float));

    // Read new K/V and store in host cache
    for (int l = 0; l < NL; l++) {
        // new_k/new_v are (HD, 1, NH) — read and store at cache[layer][pos]
        std::vector<float> new_k(NH * HD), new_v(NH * HD);
        char name[64];
        snprintf(name, sizeof(name), "new_k_%d", l);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, name), new_k.data(), 0, new_k.size() * sizeof(float));
        snprintf(name, sizeof(name), "new_v_%d", l);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, name), new_v.data(), 0, new_v.size() * sizeof(float));

        // ggml layout: (HD, 1, NH) = for h in NH: HD values. = [h0d0..h0d63, h1d0..h1d63, ...]
        // Host layout at cache[layer][pos]: (NH, HD) = [h0d0..h0d63, h1d0..h1d63, ...]
        // Same layout! Just copy directly.
        float* k_slot = kv_k_ptr(kv, l, pos);
        float* v_slot = kv_v_ptr(kv, l, pos);
        memcpy(k_slot, new_k.data(), new_k.size() * sizeof(float));
        memcpy(v_slot, new_v.data(), new_v.size() * sizeof(float));
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
}

// ── Backbone forward: single position (AR step, legacy manual) ────

// Process a single input position through the 6-layer transformer,
// using the KV cache for attending to all previous positions.
// Input: x_in (d_model,) — the input embedding for this position.
// Output: out (d_model,) — the LayerNormed hidden state.
static void backbone_forward_step(pocket_tts_context* pctx, const float* x_in, float* out) {
    const auto& m = pctx->model;
    const auto& hp = m.flow_lm_hp;
    const int D = (int)hp.d_model;
    const int NH = (int)hp.num_heads;
    const int HD = (int)hp.head_dim();
    const int FF = (int)hp.ff_dim();
    const int NL = (int)hp.num_layers;
    const int pos = pctx->backbone_kv.offset;

    // Working buffers (stack-allocated for typical sizes: D=1024, FF=4096)
    std::vector<float> residual(D), normed(D), qkv(3 * D);
    std::vector<float> attn_out(D), ff_h(FF), ff_out(D);
    std::vector<float> cur(D);

    vec_copy(cur.data(), x_in, D);

    for (int l = 0; l < NL; l++) {
        const auto& L = m.backbone_layers[l];

        // ─ Self-attention with pre-norm ─
        vec_copy(residual.data(), cur.data(), D);

        // LayerNorm (pre-norm)
        layer_norm(normed.data(), cur.data(), D, tensor_f32_data(L.attn_norm_w), tensor_f32_data(L.attn_norm_b));

        // QKV projection (fused: in_proj is (3*D, D))
        linear_f32(qkv.data(), normed.data(), L.attn_in_proj, nullptr, 3 * D, D);

        float* Q = qkv.data();
        float* K = qkv.data() + D;
        float* V = qkv.data() + 2 * D;

        // Apply RoPE to Q and K (per head)
        for (int h = 0; h < NH; h++) {
            apply_rope_inplace(&Q[h * HD], HD, pos, (float)hp.max_period);
            apply_rope_inplace(&K[h * HD], HD, pos, (float)hp.max_period);
        }

        // Store K, V in cache
        float* k_slot = kv_k_ptr(pctx->backbone_kv, l, pos);
        float* v_slot = kv_v_ptr(pctx->backbone_kv, l, pos);
        vec_copy(k_slot, K, D);
        vec_copy(v_slot, V, D);

        // Causal attention: attend over positions [0..pos]
        int seq_len = pos + 1;
        vec_zero(attn_out.data(), D);

        for (int h = 0; h < NH; h++) {
            // Compute attention scores for this head
            std::vector<float> scores(seq_len);
            float scale = 1.0f / std::sqrt((float)HD);
            for (int p = 0; p < seq_len; p++) {
                float* kp = kv_k_ptr(pctx->backbone_kv, l, p);
                scores[p] = vec_dot(&Q[h * HD], &kp[h * HD], HD) * scale;
            }
            // Softmax
            float max_s = *std::max_element(scores.begin(), scores.end());
            float sum_exp = 0.0f;
            for (int p = 0; p < seq_len; p++) {
                scores[p] = std::exp(scores[p] - max_s);
                sum_exp += scores[p];
            }
            for (int p = 0; p < seq_len; p++)
                scores[p] /= sum_exp;

            // Weighted sum of V
            for (int p = 0; p < seq_len; p++) {
                float* vp = kv_v_ptr(pctx->backbone_kv, l, p);
                vec_fma(&attn_out[h * HD], &vp[h * HD], scores[p], HD);
            }
        }

        // Output projection (out_proj is (D, D))
        std::vector<float> proj_out(D);
        linear_f32(proj_out.data(), attn_out.data(), L.attn_out_proj, nullptr, D, D);

        // Residual
        vec_add(cur.data(), residual.data(), proj_out.data(), D);

        // ─ FFN with pre-norm ─
        vec_copy(residual.data(), cur.data(), D);

        // LayerNorm
        layer_norm(normed.data(), cur.data(), D, tensor_f32_data(L.ffn_norm_w), tensor_f32_data(L.ffn_norm_b));

        // FFN: Linear(D -> FF) -> GELU -> Linear(FF -> D)
        linear_f32(ff_h.data(), normed.data(), L.ffn_linear1, nullptr, FF, D);
        for (int i = 0; i < FF; i++)
            ff_h[i] = gelu_f(ff_h[i]);
        linear_f32(ff_out.data(), ff_h.data(), L.ffn_linear2, nullptr, D, FF);

        // Residual
        vec_add(cur.data(), residual.data(), ff_out.data(), D);
    }

    // Final LayerNorm (out_norm)
    layer_norm(out, cur.data(), D, tensor_f32_data(m.out_norm_w), tensor_f32_data(m.out_norm_b));
}

// Dispatch: ggml or manual backbone forward step.
// --no-gpu (use_gpu=false) forces legacy manual path; env var POCKET_MANUAL_BACKBONE=1 also works.
static void backbone_step(pocket_tts_context* pctx, const float* x_in, float* out) {
    if (!pctx->params.use_gpu || getenv("POCKET_MANUAL_BACKBONE")) {
        backbone_forward_step(pctx, x_in, out);
    } else {
        backbone_forward_step_ggml(pctx, x_in, out);
    }
}

// ── Flow net (consistency head) forward ───────────────────────────

// Compute timestep embedding for a single scalar t using learned MLP.
// freqs: (half_dim,), linear1: (flow_dim, freq_embed_size), etc.
static void compute_timestep_emb(float* out, float t_val, const pocket_tts_flow_net::timestep_embedder& te,
                                 int freq_embed_size, int flow_dim) {
    int half = freq_embed_size / 2;
    const float* freqs = tensor_f32_data(te.freqs);

    // Compute sinusoidal: args = t * freqs, then cos(args), sin(args), concat
    std::vector<float> sincos(freq_embed_size);
    for (int i = 0; i < half; i++) {
        float angle = t_val * freqs[i];
        sincos[i] = (float)std::cos(angle);
        sincos[i + half] = (float)std::sin(angle);
    }

    // MLP: linear1 -> SiLU -> linear2 -> RMSNorm
    std::vector<float> h1(flow_dim), h2(flow_dim);
    linear_f32(h1.data(), sincos.data(), te.linear1_w, te.linear1_b, flow_dim, freq_embed_size);
    for (int i = 0; i < flow_dim; i++)
        h1[i] = silu_f(h1[i]);
    linear_f32(h2.data(), h1.data(), te.linear2_w, te.linear2_b, flow_dim, flow_dim);
    rms_norm(out, h2.data(), tensor_f32_data(te.rms_alpha), flow_dim);
}

// Single evaluation of the flow network: given conditioning c, timesteps s and t,
// and current latent x, compute the flow direction.
static void flow_net_eval(pocket_tts_context* pctx, const float* cond, // (d_model,)
                          float s_val, float t_val,
                          const float* x_in, // (latent_dim,) current noise/latent
                          float* flow_out    // (latent_dim,) flow direction
) {
    const auto& m = pctx->model;
    const auto& fh = m.flow_head_hp;
    const int FD = (int)fh.flow_dim;
    const int LD = (int)m.flow_lm_hp.latent_dim;
    const int D = (int)m.flow_lm_hp.d_model;
    const int FES = (int)fh.freq_embed_size;
    const auto& fn = m.flow_net;

    // 1. Input projection: x -> flow_dim
    std::vector<float> x(FD);
    linear_f32(x.data(), x_in, fn.input_proj, fn.input_proj_b, FD, LD);

    // 2. Timestep embeddings
    std::vector<float> te0(FD), te1(FD);
    compute_timestep_emb(te0.data(), s_val, fn.time_embeds[0], FES, FD);
    compute_timestep_emb(te1.data(), t_val, fn.time_embeds[1], FES, FD);

    // Average the two timestep embeddings
    std::vector<float> t_combined(FD);
    for (int i = 0; i < FD; i++)
        t_combined[i] = (te0[i] + te1[i]) * 0.5f;

    // 3. Conditioning embedding
    std::vector<float> c_emb(FD);
    linear_f32(c_emb.data(), cond, fn.cond_embed, fn.cond_embed_b, FD, D);

    // 4. Combined modulation signal: y = t_combined + c_emb
    std::vector<float> y(FD);
    vec_add(y.data(), t_combined.data(), c_emb.data(), FD);

    // Dump y for diff
    if (getenv("POCKET_DUMP_DIR")) {
        float yn = 0, tcn = 0, cn = 0;
        for (int i = 0; i < FD; i++) {
            yn += y[i] * y[i];
            tcn += t_combined[i] * t_combined[i];
            cn += c_emb[i] * c_emb[i];
        }
        fprintf(stderr, "  flow_net y_norm=%.4f t_combined_norm=%.4f c_emb_norm=%.4f\n", std::sqrt(yn), std::sqrt(tcn),
                std::sqrt(cn));
    }

    // 5. ResBlocks with AdaLN
    std::vector<float> normed(FD), ada_out(3 * FD), h_mlp(FD), h_out(FD);
    for (int r = 0; r < (int)fh.flow_depth; r++) {
        const auto& rb = fn.res_blocks[r];

        // adaLN_modulation: SiLU(y) -> Linear -> (shift, scale, gate)
        std::vector<float> y_silu(FD);
        for (int i = 0; i < FD; i++)
            y_silu[i] = silu_f(y[i]);
        linear_f32(ada_out.data(), y_silu.data(), rb.ada_linear, rb.ada_bias, 3 * FD, FD);
        float* shift = ada_out.data();
        float* scale = ada_out.data() + FD;
        float* gate = ada_out.data() + 2 * FD;

        // LayerNorm(x) * (1 + scale) + shift  (ResBlock uses eps=1e-6)
        layer_norm(normed.data(), x.data(), FD, tensor_f32_data(rb.ln_w), tensor_f32_data(rb.ln_b), 1e-6f);
        for (int i = 0; i < FD; i++)
            normed[i] = normed[i] * (1.0f + scale[i]) + shift[i];

        // MLP: Linear -> SiLU -> Linear (both with bias)
        linear_f32(h_mlp.data(), normed.data(), rb.mlp_linear1, rb.mlp_linear1_b, FD, FD);
        for (int i = 0; i < FD; i++)
            h_mlp[i] = silu_f(h_mlp[i]);
        linear_f32(h_out.data(), h_mlp.data(), rb.mlp_linear2, rb.mlp_linear2_b, FD, FD);

        // x = x + gate * h_out
        for (int i = 0; i < FD; i++)
            x[i] += gate[i] * h_out[i];

        if (getenv("POCKET_DUMP_DIR")) {
            float xn = 0, gn = 0, hn = 0;
            for (int i = 0; i < FD; i++) {
                xn += x[i] * x[i];
                gn += gate[i] * gate[i];
                hn += h_out[i] * h_out[i];
            }
            fprintf(stderr, "  flow_net ResBlock %d: x_norm=%.4f gate_norm=%.4f h_norm=%.4f\n", r, std::sqrt(xn),
                    std::sqrt(gn), std::sqrt(hn));
        }
    }

    // 6. FinalLayer: AdaLN (2-way) + linear
    std::vector<float> y_silu(FD);
    for (int i = 0; i < FD; i++)
        y_silu[i] = silu_f(y[i]);
    std::vector<float> final_ada_out(2 * FD);
    linear_f32(final_ada_out.data(), y_silu.data(), fn.final_ada, fn.final_ada_b, 2 * FD, FD);
    float* f_shift = final_ada_out.data();
    float* f_scale = final_ada_out.data() + FD;

    // LN without affine + modulate
    layer_norm(normed.data(), x.data(), FD, nullptr, nullptr, 1e-6f);

    // Dump pre-modulation x norm and post-modulation for debugging
    if (getenv("POCKET_DUMP_DIR")) {
        float x_norm = 0, normed_norm = 0;
        for (int i = 0; i < FD; i++) {
            x_norm += x[i] * x[i];
            normed_norm += normed[i] * normed[i];
        }
        fprintf(stderr, "  flow_net pre-final x_norm=%.4f normed_norm=%.4f\n", std::sqrt(x_norm),
                std::sqrt(normed_norm));
        // Dump scale/shift stats
        float scale_mean = 0, shift_mean = 0;
        for (int i = 0; i < FD; i++) {
            scale_mean += f_scale[i];
            shift_mean += f_shift[i];
        }
        fprintf(stderr, "  flow_net final scale_mean=%.4f shift_mean=%.4f\n", scale_mean / FD, shift_mean / FD);
        fprintf(stderr, "  flow_net final 1+scale first4: %.4f %.4f %.4f %.4f\n", 1.0f + f_scale[0], 1.0f + f_scale[1],
                1.0f + f_scale[2], 1.0f + f_scale[3]);
    }

    for (int i = 0; i < FD; i++)
        normed[i] = normed[i] * (1.0f + f_scale[i]) + f_shift[i];

    // Final linear: flow_dim -> latent_dim
    linear_f32(flow_out, normed.data(), fn.final_linear, fn.final_linear_b, LD, FD);
}

// LSD decode: iterative flow integration
static void flow_net_forward(pocket_tts_context* pctx, const float* backbone_out, const float* noise, int lsd_steps,
                             float* latent_out) {
    const int LD = (int)pctx->model.flow_lm_hp.latent_dim;

    // Start from noise
    std::vector<float> current(LD);
    vec_copy(current.data(), noise, LD);

    std::vector<float> flow_dir(LD);
    float dt = 1.0f / lsd_steps;
    for (int step = 0; step < lsd_steps; step++) {
        float s = (float)step / lsd_steps;
        float t = (float)(step + 1) / lsd_steps;
        flow_net_eval(pctx, backbone_out, s, t, current.data(), flow_dir.data());
        for (int i = 0; i < LD; i++)
            current[i] += flow_dir[i] * dt;
    }

    vec_copy(latent_out, current.data(), LD);
}

// ── EOS check ─────────────────────────────────────────────────────

static bool check_eos(pocket_tts_context* pctx, const float* backbone_out) {
    const auto& m = pctx->model;
    const int D = (int)m.flow_lm_hp.d_model;

    // Linear(d_model -> 1) + bias. The reference compares the raw linear
    // output against eos_threshold (no sigmoid).
    float logit = vec_dot(tensor_f32_data(m.out_eos_w), backbone_out, D);
    if (m.out_eos_b)
        logit += tensor_f32_data(m.out_eos_b)[0];

    return logit > pctx->params.eos_threshold;
}

// ── Mimi VAE decoder (ggml graph version) ─────────────────────────

// Helper: causal conv1d with left-only padding.
// Input x: (T, Cin), weight w: ggml (K, Cin, Cout), bias b: (Cout,) or nullptr.
// Returns (T_out, Cout) — same T for stride=1 due to causal padding.
static ggml_tensor* pocket_conv1d_causal(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride,
                                         int dilation = 1) {
    int K = (int)w->ne[0];
    int eff_K = (K - 1) * dilation + 1;
    int pad_left = eff_K - stride;
    if (pad_left > 0) {
        x = ggml_pad_ext(ctx, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
    }
    ggml_tensor* out = ggml_conv_1d(ctx, w, x, stride, 0, dilation);
    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);
    if (b) {
        ggml_tensor* bb = ggml_reshape_2d(ctx, b, 1, ggml_nelements(b));
        out = ggml_add(ctx, out, bb);
    }
    return out;
}

// Helper: causal ConvTranspose1d — full conv then trim right to T_in * stride.
// Input x: (T, Cin), weight w: ggml (K, Cout, Cin), bias b: (Cout,) or nullptr.
// Returns (T_in * stride, Cout).
static ggml_tensor* pocket_convtr1d_causal(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                           int stride) {
    int T_in = (int)x->ne[0];
    int Cout = (int)w->ne[1];
    ggml_tensor* out = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    int T_full = (int)out->ne[0];
    int T_want = T_in * stride;
    if (T_full > T_want) {
        // Trim from the right — keep first T_want samples
        out = ggml_view_2d(ctx, out, T_want, Cout, out->nb[1], 0);
        out = ggml_cont(ctx, out);
    }
    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);
    if (b) {
        ggml_tensor* bb = ggml_reshape_2d(ctx, b, 1, ggml_nelements(b));
        out = ggml_add(ctx, out, bb);
    }
    return out;
}

// Build Mimi decoder transformer graph: 2 layers, pre-norm LN, causal attention
// with limited context window, LayerScale, GELU FFN, RoPE.
// Input x: (T, D) = (T, 512). Returns (T, D).
static ggml_tensor* build_pocket_mimi_xfmr(ggml_context* ctx, const pocket_tts_model& m, ggml_tensor* x,
                                           ggml_tensor* positions, ggml_tensor* causal_mask) {
    const auto& mi = m.mimi_hp;
    int D = (int)mi.xfmr_d_model;
    int NH = (int)mi.xfmr_num_heads;
    int HD = D / NH;
    int FF = (int)mi.xfmr_dim_feedforward;
    int T = (int)x->ne[0]; // time axis is ne[0] in (T, D) layout
    // Note: ggml_norm/ggml_mul_mat operate on ne[0], which is the feature dim.
    // We need x in (D, T) layout for matmul, but (T, D) for conv.
    // Actually, ggml_mul_mat(W, x) where W is (out, in) and x is (in, T) gives (out, T).
    // So x should be (D, T) for matmul. Let's transpose.

    // Work in (D, T) layout for matmul operations
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // (D, T)

    // Apply input projection if present
    if (m.dec_xfmr_input_proj) {
        x = ggml_mul_mat(ctx, m.dec_xfmr_input_proj, x); // (D, T)
    }

    for (int l = 0; l < (int)mi.xfmr_num_layers; l++) {
        const auto& L = m.dec_transformer_layers[l];
        ggml_tensor* residual = x;

        // Pre-norm LayerNorm
        ggml_tensor* h = ggml_norm(ctx, x, 1e-5f);
        if (L.attn_norm_w)
            h = ggml_mul(ctx, h, L.attn_norm_w);
        if (L.attn_norm_b)
            h = ggml_add(ctx, h, L.attn_norm_b);

        // Fused QKV projection: (3*D, D) @ (D, T) -> (3*D, T)
        ggml_tensor* qkv = ggml_mul_mat(ctx, L.attn_in_proj, h);

        // Split Q, K, V: each (D, T)
        ggml_tensor* Q = ggml_view_2d(ctx, qkv, D, T, qkv->nb[1], 0);
        ggml_tensor* K_t = ggml_view_2d(ctx, qkv, D, T, qkv->nb[1], (size_t)D * ggml_type_size(qkv->type));
        ggml_tensor* V = ggml_view_2d(ctx, qkv, D, T, qkv->nb[1], (size_t)2 * D * ggml_type_size(qkv->type));

        // Reshape to (HD, NH, T) for RoPE
        Q = ggml_reshape_3d(ctx, ggml_cont(ctx, Q), HD, NH, T);
        K_t = ggml_reshape_3d(ctx, ggml_cont(ctx, K_t), HD, NH, T);
        V = ggml_reshape_3d(ctx, ggml_cont(ctx, V), HD, NH, T);

        // RoPE (Mimi decoder uses theta=10000)
        Q = ggml_rope_ext(ctx, Q, positions, nullptr, HD, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);
        K_t = ggml_rope_ext(ctx, K_t, positions, nullptr, HD, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f,
                            0.0f, 0.0f);

        // Permute for flash_attn_ext: (HD, NH, T) -> (HD, T, NH)
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K_t = ggml_cont(ctx, ggml_permute(ctx, K_t, 0, 2, 1, 3));
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        // Causal attention with context window mask
        float scale = 1.0f / sqrtf((float)HD);
        ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K_t, V, causal_mask, scale, 0.0f, 0.0f);

        // Output: (HD, NH, T) -> (D, T)
        attn = ggml_reshape_2d(ctx, attn, D, T);

        // Output projection
        attn = ggml_mul_mat(ctx, L.attn_out_proj, attn);

        // LayerScale
        if (L.layer_scale_1) {
            ggml_tensor* ls = ggml_reshape_2d(ctx, L.layer_scale_1, D, 1);
            attn = ggml_mul(ctx, attn, ls);
        }

        x = ggml_add(ctx, residual, attn);

        // FFN
        residual = x;
        h = ggml_norm(ctx, x, 1e-5f);
        if (L.ffn_norm_w)
            h = ggml_mul(ctx, h, L.ffn_norm_w);
        if (L.ffn_norm_b)
            h = ggml_add(ctx, h, L.ffn_norm_b);

        h = ggml_mul_mat(ctx, L.ffn_linear1, h); // (FF, T)
        h = ggml_gelu(ctx, h);
        h = ggml_mul_mat(ctx, L.ffn_linear2, h); // (D, T)

        if (L.layer_scale_2) {
            ggml_tensor* ls = ggml_reshape_2d(ctx, L.layer_scale_2, D, 1);
            h = ggml_mul(ctx, h, ls);
        }

        x = ggml_add(ctx, residual, h);
    }

    // Apply output projection if present
    if (m.dec_xfmr_output_proj) {
        x = ggml_mul_mat(ctx, m.dec_xfmr_output_proj, x); // (D, T)
    }

    // Transpose back to (T, D) for conv operations
    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    return x;
}

// Build SEANet decoder graph.
// Input x: (T, Cin) where Cin = outer_dim (512). Returns (T_pcm, 1) flattened.
static ggml_tensor* build_pocket_seanet_dec(ggml_context* ctx, const pocket_tts_model& m, ggml_tensor* x) {
    const auto& mi = m.mimi_hp;
    const auto& sd = m.seanet_dec;
    const auto& ratios = mi.seanet_ratios;

    // Initial conv: causal, stride=1
    if (sd.initial_conv_w) {
        x = pocket_conv1d_causal(ctx, x, sd.initial_conv_w, sd.initial_conv_b, 1);
    }

    // Stages: ELU -> ConvTranspose (upsample) -> ResBlocks
    for (size_t s = 0; s < ratios.size(); s++) {
        int ratio = ratios[s];
        const auto& stage = sd.stages[s];

        // ELU
        x = ggml_elu(ctx, x);

        // ConvTranspose1d (upsample by ratio) — causal trim
        if (stage.convtr_w) {
            x = pocket_convtr1d_causal(ctx, x, stage.convtr_w, stage.convtr_b, ratio);
        }

        // Residual blocks
        int res_K = (int)mi.seanet_residual_kernel_size;
        for (size_t r = 0; r < stage.resblocks.size(); r++) {
            const auto& rb = stage.resblocks[r];
            if (!rb.conv0_w || !rb.conv1_w)
                continue;

            int dilation = 1;
            for (size_t d = 0; d < r; d++)
                dilation *= (int)mi.seanet_dilation_base;

            ggml_tensor* h = ggml_elu(ctx, x);

            // Dilated causal conv (Cin -> Cin/compress)
            h = pocket_conv1d_causal(ctx, h, rb.conv0_w, rb.conv0_b, 1, dilation);

            // ELU
            h = ggml_elu(ctx, h);

            // Second conv (Cin/compress -> Cin, kernel from weight)
            int K1 = (int)rb.conv1_w->ne[0];
            (void)K1;
            h = pocket_conv1d_causal(ctx, h, rb.conv1_w, rb.conv1_b, 1);

            // Residual add
            x = ggml_add(ctx, x, h);
        }
    }

    // Final: ELU -> conv (C_cur -> 1)
    x = ggml_elu(ctx, x);
    if (sd.final_conv_w) {
        x = pocket_conv1d_causal(ctx, x, sd.final_conv_w, sd.final_conv_b, 1);
    }

    // Flatten to 1D: (T_pcm, 1) -> (T_pcm,)
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));

    return x;
}

// Mimi decode using ggml compute graph (replaces manual mimi_decode).
// CPU-side: denormalize, quantizer projection, depthwise upsample.
// ggml graph: transformer + SEANet.
static void mimi_decode_ggml(pocket_tts_context* pctx, const float* latent_seq, int n_frames, float** pcm_out,
                             int* n_samples_out) {
    const auto& m = pctx->model;
    const auto& mi = m.mimi_hp;
    const int LD = (int)mi.inner_dim;           // 32
    const int OD = (int)mi.outer_dim;           // 512
    const int DS = (int)mi.downsample_stride(); // 16

    if (n_frames <= 0) {
        *pcm_out = nullptr;
        *n_samples_out = 0;
        return;
    }

    if (!m.quant_proj_w || !m.upsample_conv_w) {
        if (pctx->verbosity >= 1)
            fprintf(stderr, "pocket_tts: mimi_decode_ggml: missing tensors\n");
        *pcm_out = nullptr;
        *n_samples_out = 0;
        return;
    }

    // ── CPU-side: denormalize + quantizer projection + depthwise upsample ──

    // 1. Denormalize latents: x * emb_std + emb_mean
    std::vector<float> denorm(n_frames * LD);
    const float* std_data = tensor_f32_data(m.emb_std);
    const float* mean_data = tensor_f32_data(m.emb_mean);
    for (int f = 0; f < n_frames; f++) {
        for (int d = 0; d < LD; d++) {
            denorm[f * LD + d] = latent_seq[f * LD + d] * std_data[d] + mean_data[d];
        }
    }

    // 2. Quantizer projection: Conv1d(inner_dim -> outer_dim, kernel=1) = matmul
    std::vector<float> projected(n_frames * OD);
    const float* qp_w = tensor_f32_data(m.quant_proj_w);
    for (int f = 0; f < n_frames; f++) {
        for (int o = 0; o < OD; o++) {
            float val = 0.0f;
            for (int d = 0; d < LD; d++) {
                val += qp_w[o * LD + d] * denorm[f * LD + d];
            }
            projected[f * OD + o] = val;
        }
    }

    // 3. Depthwise upsample: ConvTranspose1d(OD, OD, K, stride=DS, groups=OD)
    int K_up = (int)m.upsample_conv_w->ne[0];
    int Cout_pg = (int)m.upsample_conv_w->ne[1];
    int T_up_full = (n_frames - 1) * DS + K_up;
    int T_up_causal = n_frames * DS;

    // Layout: (T, OD) — time-major for ggml graph input
    std::vector<float> upsampled(T_up_full * OD, 0.0f);

    if (Cout_pg == 1) {
        // Depthwise: each channel ci uses w[ci, 0, :] = data[ci * K + k]
        const float* w = tensor_f32_data(m.upsample_conv_w);
        for (int ci = 0; ci < OD; ci++) {
            for (int t = 0; t < n_frames; t++) {
                float x_val = projected[t * OD + ci]; // time-major input
                for (int k = 0; k < K_up; k++) {
                    int t_out = t * DS + k;
                    // Output in (T, OD) layout: [t_out * OD + ci]
                    upsampled[t_out * OD + ci] += w[ci * K_up + k] * x_val;
                }
            }
        }
    } else {
        // Full conv transpose — fallback (shouldn't happen for pocket-tts)
        // TODO: implement if needed
        fprintf(stderr, "pocket_tts: mimi_decode_ggml: non-depthwise upsample not implemented\n");
        *pcm_out = nullptr;
        *n_samples_out = 0;
        return;
    }

    // Add bias for upsample
    if (m.upsample_conv_b) {
        const float* bias = tensor_f32_data(m.upsample_conv_b);
        for (int t = 0; t < T_up_full; t++) {
            for (int c = 0; c < OD; c++) {
                upsampled[t * OD + c] += bias[c];
            }
        }
    }

    // Trim to causal length: keep first T_up_causal time steps
    int T_xfmr = T_up_causal;
    if (T_up_causal < T_up_full) {
        std::vector<float> trimmed(T_up_causal * OD);
        for (int t = 0; t < T_up_causal; t++) {
            memcpy(&trimmed[t * OD], &upsampled[t * OD], OD * sizeof(float));
        }
        upsampled = std::move(trimmed);
    }

    // Dump post-upsample (same as manual path for comparison)
    if (pctx->verbosity >= 2 || getenv("POCKET_DUMP_DIR")) {
        const char* dd = getenv("POCKET_DUMP_DIR");
        if (dd) {
            // Convert to channels-first for dump compatibility with manual path
            std::vector<float> dump_cf(OD * T_xfmr);
            for (int t = 0; t < T_xfmr; t++)
                for (int c = 0; c < OD; c++)
                    dump_cf[c * T_xfmr + t] = upsampled[t * OD + c];
            std::string p = std::string(dd) + "/cpp_mimi_upsample.f32";
            FILE* f = fopen(p.c_str(), "wb");
            if (f) {
                fwrite(dump_cf.data(), sizeof(float), dump_cf.size(), f);
                fclose(f);
            }
            fprintf(stderr, "POCKET_DUMP_DIR: wrote cpp_mimi_upsample.f32 (%d x %d)\n", OD, T_xfmr);
        }
    }

    // ── Build ggml graph for transformer + SEANet ──

    // Compute metadata arena size
    const size_t graph_nodes = 16384;
    const size_t buf_size = ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false);
    std::vector<uint8_t> compute_meta(buf_size);

    struct ggml_init_params gp = {
        /*.mem_size   =*/compute_meta.size(),
        /*.mem_buffer =*/compute_meta.data(),
        /*.no_alloc   =*/true,
    };
    ggml_context* ctx0 = ggml_init(gp);
    if (!ctx0) {
        fprintf(stderr, "pocket_tts: mimi_decode_ggml: failed to init compute context\n");
        *pcm_out = nullptr;
        *n_samples_out = 0;
        return;
    }

    // Input tensor: upsampled data in (T, OD) layout
    // ggml ne[0] = first dim = OD (features), ne[1] = T (time)
    // Wait — ggml tensors are column-major. For (T, OD) time-major data:
    // ne[0] = OD (contiguous in memory), ne[1] = T.
    // But ggml_conv_1d expects data b = [T, IC] where ne[0]=T.
    // So we need ne[0]=T, ne[1]=OD => ggml_new_tensor_2d(ctx, type, T, OD).
    // But then data layout is T values contiguous per row, OD rows = channels-last = (OD, T).
    // Hmm, ggml is column-major: ne[0] is the innermost contiguous dim.
    // ggml_new_tensor_2d(ctx, type, A, B) creates ne[0]=A, ne[1]=B, with A contiguous.
    // For conv1d, the input layout should be ne[0]=T (time), ne[1]=Cin (channels).
    // So data in memory is: for each channel, T values contiguous = channels-first (Cin, T).
    // But our upsampled data is time-major: (T, OD) = for each time step, OD values contiguous.
    // So ne[0]=OD, ne[1]=T. We need to transpose for conv1d.

    // Create input as (OD, T) — matching our memory layout (T steps of OD features)
    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, OD, T_xfmr);
    ggml_set_name(inp, "mimi_input");
    ggml_set_input(inp);

    // For conv1d: need (T, Cin) where ne[0]=T. Transpose.
    ggml_tensor* x = ggml_cont(ctx0, ggml_transpose(ctx0, inp)); // (T, OD) for conv

    // Build causal mask for transformer (context-limited causal)
    int context_size = (int)mi.xfmr_context;
    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T_xfmr, T_xfmr);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    // Positions tensor for RoPE
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_xfmr);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Transpose x to (OD, T) for matmul-based transformer
    // Actually, build_pocket_mimi_xfmr expects (T, OD) input — it transposes internally
    x = build_pocket_mimi_xfmr(ctx0, m, x, positions, causal_mask);

    // Dump hook: mark post-transformer output
    ggml_tensor* post_xfmr = x;
    if (getenv("POCKET_DUMP_DIR")) {
        ggml_set_name(post_xfmr, "post_xfmr");
        ggml_set_output(post_xfmr);
    }

    // x is (T, OD) — feed directly into SEANet
    x = build_pocket_seanet_dec(ctx0, m, x);

    ggml_set_name(x, "pcm_output");
    ggml_set_output(x);

    // Build and allocate graph
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, graph_nodes, false);
    ggml_build_forward_expand(gf, x);

    ggml_backend_t backend = pctx->backend_cpu;
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "pocket_tts: mimi_decode_ggml: failed to alloc graph\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        *pcm_out = nullptr;
        *n_samples_out = 0;
        return;
    }

    // Set input data
    ggml_tensor* inp_t = ggml_graph_get_tensor(gf, "mimi_input");
    ggml_backend_tensor_set(inp_t, upsampled.data(), 0, upsampled.size() * sizeof(float));

    // Set positions
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> pos_data(T_xfmr);
    for (int i = 0; i < T_xfmr; i++)
        pos_data[i] = i;
    ggml_backend_tensor_set(pos_t, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));

    // Set causal mask (context-limited causal: 0 = attend, -inf = block)
    ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "causal_mask");
    {
        std::vector<ggml_fp16_t> mask_data((size_t)T_xfmr * T_xfmr);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < T_xfmr; q++) {
            int start = std::max(0, q - context_size + 1);
            for (int k = 0; k < T_xfmr; k++) {
                // Causal: block future (k > q) and outside context window (k < start)
                bool attend = (k <= q) && (k >= start);
                mask_data[(size_t)q * T_xfmr + k] = attend ? zero : neg_inf;
            }
        }
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }

    // Set number of threads
    int n_threads = pctx->params.n_threads > 0 ? pctx->params.n_threads : 4;
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    // Compute
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "pocket_tts: mimi_decode_ggml: graph compute failed (status=%d)\n", (int)st);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        *pcm_out = nullptr;
        *n_samples_out = 0;
        return;
    }

    // Dump post-transformer if requested
    if (getenv("POCKET_DUMP_DIR")) {
        ggml_tensor* pxfmr = ggml_graph_get_tensor(gf, "post_xfmr");
        if (pxfmr) {
            int n_elem = (int)ggml_nelements(pxfmr);
            std::vector<float> xfmr_data(n_elem);
            ggml_backend_tensor_get(pxfmr, xfmr_data.data(), 0, n_elem * sizeof(float));
            const char* dd = getenv("POCKET_DUMP_DIR");
            std::string p = std::string(dd) + "/cpp_mimi_post_xfmr.f32";
            FILE* f = fopen(p.c_str(), "wb");
            if (f) {
                fwrite(xfmr_data.data(), sizeof(float), xfmr_data.size(), f);
                fclose(f);
            }
            fprintf(stderr, "POCKET_DUMP_DIR: wrote cpp_mimi_post_xfmr.f32 (%d x %d)\n", T_xfmr, OD);
        }
    }

    // Read PCM output
    ggml_tensor* pcm_t = ggml_graph_get_tensor(gf, "pcm_output");
    int n_pcm = (int)ggml_nelements(pcm_t);
    *pcm_out = (float*)malloc((size_t)n_pcm * sizeof(float));
    ggml_backend_tensor_get(pcm_t, *pcm_out, 0, (size_t)n_pcm * sizeof(float));
    *n_samples_out = n_pcm;

    // Cleanup
    ggml_gallocr_free(galloc);
    ggml_free(ctx0);

    if (pctx->verbosity >= 2)
        fprintf(stderr, "pocket_tts: mimi_decode_ggml: %d PCM samples\n", n_pcm);
}

// ── Mimi VAE decoder (legacy manual compute) ─────────────────────────

// Mimi decoder transformer: 2 layers with LayerScale, causal attention.
// Operates on the full sequence (non-AR batch). We reuse the same eager
// approach since sequence lengths are modest (< 2000 frames).
// Mimi transformer: shared between encoder and decoder (same architecture, different weights).
// Processes seq in-place: (T, D) with causal attention, RoPE, LayerScale, GELU FFN.
static void mimi_transformer_forward(const std::vector<pocket_tts_transformer_layer>& layers, float* seq, int T, int D,
                                     int num_heads, int dim_feedforward, int context_size) {
    const int NH = num_heads;
    const int HD = D / NH;
    const int FF = dim_feedforward;
    const int NL = (int)layers.size();

    std::vector<float> normed(D), qkv(3 * D), attn_out(D), proj_out(D);
    std::vector<float> ff_h(FF), ff_out(D), residual(D), scaled(D);

    // Per-layer KV cache: compute K,V once per position, attend from cache.
    // Reduces complexity from O(T²·D²) to O(T²·D + T·D²).
    std::vector<float> k_cache((size_t)NL * T * D);
    std::vector<float> v_cache((size_t)NL * T * D);

    for (int l = 0; l < NL; l++) {
        const auto& L = layers[l];
        float* k_layer = &k_cache[(size_t)l * T * D];
        float* v_layer = &v_cache[(size_t)l * T * D];

        for (int t = 0; t < T; t++) {
            float* x_t = &seq[t * D];

            // ─ Self-attention with pre-norm + LayerScale ─
            vec_copy(residual.data(), x_t, D);

            layer_norm(normed.data(), x_t, D, tensor_f32_data(L.attn_norm_w), tensor_f32_data(L.attn_norm_b));
            linear_f32(qkv.data(), normed.data(), L.attn_in_proj, nullptr, 3 * D, D);

            float* Q = qkv.data();
            float* K = qkv.data() + D;
            float* V = qkv.data() + 2 * D;

            // Apply RoPE to Q and K (per head)
            for (int h = 0; h < NH; h++) {
                apply_rope_inplace(&Q[h * HD], HD, t, 10000.0f);
                apply_rope_inplace(&K[h * HD], HD, t, 10000.0f);
            }

            // Store K,V in cache for this position
            vec_copy(&k_layer[t * D], K, D);
            vec_copy(&v_layer[t * D], V, D);

            // Causal attention with limited context
            int start_pos = std::max(0, t - context_size + 1);
            int att_len = t - start_pos + 1;

            vec_zero(attn_out.data(), D);
            for (int h = 0; h < NH; h++) {
                std::vector<float> scores(att_len);
                float scale = 1.0f / std::sqrt((float)HD);
                for (int p = 0; p < att_len; p++) {
                    int abs_p = start_pos + p;
                    float* Kp = &k_layer[abs_p * D];
                    scores[p] = vec_dot(&Q[h * HD], &Kp[h * HD], HD) * scale;
                }
                float max_s = *std::max_element(scores.begin(), scores.begin() + att_len);
                float sum_exp = 0.0f;
                for (int p = 0; p < att_len; p++) {
                    scores[p] = std::exp(scores[p] - max_s);
                    sum_exp += scores[p];
                }
                for (int p = 0; p < att_len; p++)
                    scores[p] /= sum_exp;

                for (int p = 0; p < att_len; p++) {
                    int abs_p = start_pos + p;
                    float* Vp = &v_layer[abs_p * D];
                    vec_fma(&attn_out[h * HD], &Vp[h * HD], scores[p], HD);
                }
            }

            linear_f32(proj_out.data(), attn_out.data(), L.attn_out_proj, nullptr, D, D);

            // LayerScale
            if (L.layer_scale_1) {
                const float* ls = tensor_f32_data(L.layer_scale_1);
                for (int i = 0; i < D; i++)
                    proj_out[i] *= ls[i];
            }

            vec_add(x_t, residual.data(), proj_out.data(), D);

            // ─ FFN with pre-norm + LayerScale ─
            vec_copy(residual.data(), x_t, D);
            layer_norm(normed.data(), x_t, D, tensor_f32_data(L.ffn_norm_w), tensor_f32_data(L.ffn_norm_b));

            linear_f32(ff_h.data(), normed.data(), L.ffn_linear1, nullptr, FF, D);
            for (int i = 0; i < FF; i++)
                ff_h[i] = gelu_f(ff_h[i]);
            linear_f32(ff_out.data(), ff_h.data(), L.ffn_linear2, nullptr, D, FF);

            // LayerScale
            if (L.layer_scale_2) {
                const float* ls = tensor_f32_data(L.layer_scale_2);
                for (int i = 0; i < D; i++)
                    ff_out[i] *= ls[i];
            }

            vec_add(x_t, residual.data(), ff_out.data(), D);
        }
    }
}

// ELU activation
static inline float elu_f(float x, float alpha = 1.0f) {
    return x >= 0.0f ? x : alpha * (std::exp(x) - 1.0f);
}

// Conv1d (naive, valid padding handled by caller with explicit padding)
// w: (Cout, Cin, K), b: (Cout,) or nullptr
// Input: (Cin, T_in), Output: (Cout, T_out) where T_out depends on stride/pad
static void conv1d_eager(float* out, const float* in, int Cin, int T_in, ggml_tensor* w_tensor, ggml_tensor* b_tensor,
                         int stride, int pad_left, int pad_right) {
    const float* w = tensor_f32_data(w_tensor);
    const int K = (int)w_tensor->ne[0];
    const int Cin_w = (int)w_tensor->ne[1];
    const int Cout = (int)w_tensor->ne[2];
    (void)Cin_w; // should equal Cin

    int T_padded = T_in + pad_left + pad_right;
    int T_out = (T_padded - K) / stride + 1;

    for (int co = 0; co < Cout; co++) {
        float bias_val = b_tensor ? tensor_f32_data(b_tensor)[co] : 0.0f;
        for (int t = 0; t < T_out; t++) {
            float val = bias_val;
            int t_start = t * stride - pad_left;
            for (int ci = 0; ci < Cin; ci++) {
                for (int k = 0; k < K; k++) {
                    int t_in = t_start + k;
                    if (t_in >= 0 && t_in < T_in) {
                        val += w[(co * Cin + ci) * K + k] * in[ci * T_in + t_in];
                    }
                }
            }
            out[co * T_out + t] = val;
        }
    }
}

// ConvTranspose1d (naive)
// GGUF ne = [K, Cout, Cin]: ggml row-major data[cin * Cout * K + cout * K + k].
// PyTorch ConvTranspose1d weight is (Cin, Cout, K).
// Input: (Cin, T_in) -> Output: (Cout, T_out) where T_out = (T_in-1)*stride + K - 2*pad
static void conv_transpose1d_eager(float* out, const float* in, int Cin, int T_in, ggml_tensor* w_tensor,
                                   ggml_tensor* b_tensor, int stride, int pad) {
    const float* w = tensor_f32_data(w_tensor);
    const int K = (int)w_tensor->ne[0];
    const int Cout = (int)w_tensor->ne[1];
    const int Cin_w = (int)w_tensor->ne[2];
    (void)Cin_w;

    int T_out = (T_in - 1) * stride + K;
    int T_out_cropped = T_out - 2 * pad;

    // Zero output
    memset(out, 0, (size_t)Cout * T_out_cropped * sizeof(float));

    // Accumulate: for each input position and kernel tap
    for (int ci = 0; ci < Cin; ci++) {
        for (int t = 0; t < T_in; t++) {
            float x_val = in[ci * T_in + t];
            for (int k = 0; k < K; k++) {
                int t_out_raw = t * stride + k;
                int t_out_pos = t_out_raw - pad;
                if (t_out_pos >= 0 && t_out_pos < T_out_cropped) {
                    for (int co = 0; co < Cout; co++) {
                        // ggml row-major: data[cin * Cout * K + cout * K + k]
                        out[co * T_out_cropped + t_out_pos] += w[ci * Cout * K + co * K + k] * x_val;
                    }
                }
            }
        }
    }

    // Add bias
    if (b_tensor) {
        const float* b = tensor_f32_data(b_tensor);
        for (int co = 0; co < Cout; co++) {
            for (int t = 0; t < T_out_cropped; t++) {
                out[co * T_out_cropped + t] += b[co];
            }
        }
    }
}

// Mimi VAE decoder: latent sequence -> 24 kHz PCM
static void mimi_decode(pocket_tts_context* pctx, const float* latent_seq, int n_frames, float** pcm_out,
                        int* n_samples_out) {
    const auto& m = pctx->model;
    const auto& mi = m.mimi_hp;
    const int LD = (int)mi.inner_dim;           // 32
    const int OD = (int)mi.outer_dim;           // 512
    const int DS = (int)mi.downsample_stride(); // 16

    if (n_frames <= 0) {
        *pcm_out = nullptr;
        *n_samples_out = 0;
        return;
    }

    // Verify essential tensors are loaded
    if (!m.quant_proj_w || !m.upsample_conv_w) {
        if (pctx->verbosity >= 1)
            fprintf(stderr, "pocket_tts: mimi_decode: missing tensors (quant_proj=%p, upsample=%p)\n",
                    (void*)m.quant_proj_w, (void*)m.upsample_conv_w);
        *pcm_out = nullptr;
        *n_samples_out = 0;
        return;
    }

    // 1. Denormalize latents: x * emb_std + emb_mean
    std::vector<float> denorm(n_frames * LD);
    const float* std_data = tensor_f32_data(m.emb_std);
    const float* mean_data = tensor_f32_data(m.emb_mean);
    for (int f = 0; f < n_frames; f++) {
        for (int d = 0; d < LD; d++) {
            denorm[f * LD + d] = latent_seq[f * LD + d] * std_data[d] + mean_data[d];
        }
    }

    // 2. Quantizer projection: Conv1d(inner_dim -> outer_dim, kernel=1)
    // quant_proj_w: Conv1d(inner_dim -> outer_dim, K=1), stored in GGUF as
    // ne=[1, 32, 512] — ggml row-major: data[cout * Cin + cin], i.e. (Cout, Cin).
    std::vector<float> projected(n_frames * OD);
    const float* qp_w = tensor_f32_data(m.quant_proj_w);
    for (int f = 0; f < n_frames; f++) {
        for (int o = 0; o < OD; o++) {
            float val = 0.0f;
            for (int d = 0; d < LD; d++) {
                val += qp_w[o * LD + d] * denorm[f * LD + d];
            }
            projected[f * OD + o] = val;
        }
    }

    // 3. Upsample: ConvTranspose1d(outer_dim -> outer_dim, stride=DS)
    // Input layout: channels-first (OD, n_frames) — transpose projected
    std::vector<float> channels_first(n_frames * OD);
    for (int f = 0; f < n_frames; f++) {
        for (int c = 0; c < OD; c++) {
            channels_first[c * n_frames + f] = projected[f * OD + c];
        }
    }

    // Upsample: depthwise ConvTranspose1d (groups=Cin=512), causal.
    // Full convtr (no padding), then trim right by K-S to get T*S output.
    int K_up = (int)m.upsample_conv_w->ne[0];
    int Cout_pg = (int)m.upsample_conv_w->ne[1]; // 1 (per-group output channels)
    int T_up_full = (n_frames - 1) * DS + K_up;  // full transposed conv length
    int T_up_causal = n_frames * DS;             // causal: keep first T*S samples
    std::vector<float> upsampled(OD * T_up_full, 0.0f);

    if (Cout_pg == 1) {
        // Depthwise: each channel ci uses w[ci, 0, :] (= data[ci * K + k])
        const float* w = tensor_f32_data(m.upsample_conv_w);
        for (int ci = 0; ci < OD; ci++) {
            for (int t = 0; t < n_frames; t++) {
                float x_val = channels_first[ci * n_frames + t];
                for (int k = 0; k < K_up; k++) {
                    int t_out = t * DS + k; // no padding offset
                    upsampled[ci * T_up_full + t_out] += w[ci * K_up + k] * x_val;
                }
            }
        }
    } else {
        conv_transpose1d_eager(upsampled.data(), channels_first.data(), OD, n_frames, m.upsample_conv_w,
                               m.upsample_conv_b, DS, 0);
    }
    // Trim to causal length: keep first T*S samples per channel
    if (T_up_causal < T_up_full) {
        std::vector<float> trimmed(OD * T_up_causal);
        for (int c = 0; c < OD; c++)
            memcpy(&trimmed[c * T_up_causal], &upsampled[c * T_up_full], T_up_causal * sizeof(float));
        upsampled = std::move(trimmed);
    }
    int T_xfmr = T_up_causal;

    // Dump post-upsample
    if (pctx->verbosity >= 2 || getenv("POCKET_DUMP_DIR")) {
        const char* dd = getenv("POCKET_DUMP_DIR");
        if (dd) {
            std::string p = std::string(dd) + "/cpp_mimi_upsample.f32";
            FILE* f = fopen(p.c_str(), "wb");
            if (f) {
                fwrite(upsampled.data(), sizeof(float), upsampled.size(), f);
                fclose(f);
            }
            fprintf(stderr, "POCKET_DUMP_DIR: wrote cpp_mimi_upsample.f32 (%d x %d)\n", OD, T_xfmr);
        }
    }

    // 4. Decoder transformer (2 layers, 512D, context=250, LayerScale)
    // Convert to seq-first layout: (T, D)
    std::vector<float> xfmr_seq(T_xfmr * OD);
    for (int t = 0; t < T_xfmr; t++) {
        for (int c = 0; c < OD; c++) {
            xfmr_seq[t * OD + c] = upsampled[c * T_xfmr + t];
        }
    }

    // Apply input projection if present
    if (m.dec_xfmr_input_proj) {
        std::vector<float> proj_seq(T_xfmr * OD);
        for (int t = 0; t < T_xfmr; t++) {
            linear_f32(&proj_seq[t * OD], &xfmr_seq[t * OD], m.dec_xfmr_input_proj, nullptr, OD, OD);
        }
        xfmr_seq = std::move(proj_seq);
    }

    mimi_transformer_forward(m.dec_transformer_layers, xfmr_seq.data(), T_xfmr, OD, (int)mi.xfmr_num_heads,
                             (int)mi.xfmr_dim_feedforward, (int)mi.xfmr_context);

    // Apply output projection if present
    if (m.dec_xfmr_output_proj) {
        std::vector<float> proj_seq(T_xfmr * OD);
        for (int t = 0; t < T_xfmr; t++) {
            linear_f32(&proj_seq[t * OD], &xfmr_seq[t * OD], m.dec_xfmr_output_proj, nullptr, OD, OD);
        }
        xfmr_seq = std::move(proj_seq);
    }

    // Dump post-transformer
    if (getenv("POCKET_DUMP_DIR")) {
        const char* dd = getenv("POCKET_DUMP_DIR");
        std::string p = std::string(dd) + "/cpp_mimi_post_xfmr.f32";
        FILE* f = fopen(p.c_str(), "wb");
        if (f) {
            fwrite(xfmr_seq.data(), sizeof(float), xfmr_seq.size(), f);
            fclose(f);
        }
        fprintf(stderr, "POCKET_DUMP_DIR: wrote cpp_mimi_post_xfmr.f32 (%d x %d)\n", T_xfmr, OD);
    }

    // Convert back to channels-first: (OD, T_xfmr)
    std::vector<float> dec_in(OD * T_xfmr);
    for (int t = 0; t < T_xfmr; t++) {
        for (int c = 0; c < OD; c++) {
            dec_in[c * T_xfmr + t] = xfmr_seq[t * OD + c];
        }
    }

    // 5. SEANet decoder
    const auto& sd = m.seanet_dec;
    const auto& ratios = mi.seanet_ratios;
    int n_filters = (int)mi.seanet_n_filters;
    int mult = 1;
    for (int r : ratios)
        (void)r, mult *= 2; // mult = 2^n_ratios at start for initial conv
    // Actually: initial conv maps from seanet_dimension -> mult*n_filters
    // where mult = 2^(n_ratios) for the first stage, halving each time.

    // Initial conv: (seanet_dimension -> mult*n_filters, kernel=seanet_kernel_size)
    int seanet_K = (int)mi.seanet_kernel_size;
    int T_cur = T_xfmr;
    int C_cur = OD;

    // Compute initial mult
    mult = 1;
    for (size_t i = 0; i < ratios.size(); i++)
        mult *= 2;
    int C_init = mult * n_filters; // e.g., 8 * 64 = 512 for 3 ratios

    // Initial conv with causal padding (left-only, matching SEANet streaming convention)
    int pad_init_left = seanet_K - 1; // effective_kernel - stride, stride=1
    std::vector<float> seanet_buf(C_init * T_cur);
    if (sd.initial_conv_w) {
        conv1d_eager(seanet_buf.data(), dec_in.data(), C_cur, T_cur, sd.initial_conv_w, sd.initial_conv_b, 1,
                     pad_init_left, 0);
        C_cur = C_init;
    } else {
        seanet_buf = dec_in;
    }

    // Stages: ELU -> ConvTranspose (upsample) -> ResBlocks
    for (size_t s = 0; s < ratios.size(); s++) {
        int ratio = ratios[s];
        mult /= 2;
        int C_out = mult * n_filters;
        const auto& stage = sd.stages[s];

        // ELU activation
        for (int i = 0; i < C_cur * T_cur; i++)
            seanet_buf[i] = elu_f(seanet_buf[i]);

        // ConvTranspose1d (upsample by ratio) — causal: no padding, trim right by K-S
        if (stage.convtr_w) {
            int K_tr = (int)stage.convtr_w->ne[0];
            int T_full = (T_cur - 1) * ratio + K_tr; // full transposed conv output
            int T_new = T_cur * ratio;               // causal: keep first T*S samples
            std::vector<float> up_buf(C_out * T_full);
            conv_transpose1d_eager(up_buf.data(), seanet_buf.data(), C_cur, T_cur, stage.convtr_w, stage.convtr_b,
                                   ratio, 0); // pad=0
            // Trim right by (K_tr - ratio) to get causal output
            if (T_new < T_full) {
                std::vector<float> trimmed(C_out * T_new);
                for (int c = 0; c < C_out; c++)
                    memcpy(&trimmed[c * T_new], &up_buf[c * T_full], T_new * sizeof(float));
                seanet_buf = std::move(trimmed);
            } else {
                seanet_buf = std::move(up_buf);
            }
            T_cur = T_new;
            C_cur = C_out;
        }

        // Residual blocks
        int res_K = (int)mi.seanet_residual_kernel_size;
        for (size_t r = 0; r < stage.resblocks.size(); r++) {
            const auto& rb = stage.resblocks[r];
            int C_hidden = C_cur / (int)mi.seanet_compress;
            int dilation = 1;
            for (uint32_t d = 0; d < r; d++)
                dilation *= (int)mi.seanet_dilation_base;

            if (rb.conv0_w && rb.conv1_w) {
                // ELU -> dilated conv -> ELU -> conv1x1
                std::vector<float> h(C_hidden * T_cur);
                // First: ELU on input
                std::vector<float> act_buf(C_cur * T_cur);
                for (int i = 0; i < C_cur * T_cur; i++)
                    act_buf[i] = elu_f(seanet_buf[i]);

                // Dilated conv — causal: pad_left = (K-1)*dilation, pad_right = 0
                int eff_K = (res_K - 1) * dilation + 1; // effective kernel size
                int pad_d_left = eff_K - 1;             // causal left padding
                conv1d_eager(h.data(), act_buf.data(), C_cur, T_cur, rb.conv0_w, rb.conv0_b, 1, pad_d_left, 0);
                int C_h = (int)rb.conv0_w->ne[2];

                // ELU
                for (int i = 0; i < C_h * T_cur; i++)
                    h[i] = elu_f(h[i]);

                // Second conv — causal: pad_left = K-1, pad_right = 0
                int K1 = (int)rb.conv1_w->ne[0];
                int pad1_left = K1 - 1;
                std::vector<float> h2(C_cur * T_cur);
                conv1d_eager(h2.data(), h.data(), C_h, T_cur, rb.conv1_w, rb.conv1_b, 1, pad1_left, 0);

                // Residual add
                for (int i = 0; i < C_cur * T_cur; i++)
                    seanet_buf[i] += h2[i];
            }
        }
    }

    // Final: ELU -> conv (C_cur -> 1, kernel=last_kernel_size)
    for (int i = 0; i < C_cur * T_cur; i++)
        seanet_buf[i] = elu_f(seanet_buf[i]);

    int last_K = (int)mi.seanet_last_kernel_size;
    int pad_last_left = last_K - 1; // causal
    int n_out_samples = T_cur;      // causal pad preserves length
    std::vector<float> pcm_buf(n_out_samples);
    if (sd.final_conv_w) {
        conv1d_eager(pcm_buf.data(), seanet_buf.data(), C_cur, T_cur, sd.final_conv_w, sd.final_conv_b, 1,
                     pad_last_left, 0);
    }

    // Allocate output
    *n_samples_out = n_out_samples;
    *pcm_out = (float*)malloc((size_t)n_out_samples * sizeof(float));
    memcpy(*pcm_out, pcm_buf.data(), (size_t)n_out_samples * sizeof(float));
}

// Mimi VAE encoder: 24 kHz PCM -> continuous latents (voice cloning only)
// Mimi VAE encoder: 24 kHz PCM -> continuous 32-dim latents (for voice cloning)
// Pipeline: pad → SEANet encoder → encoder transformer → downsample → latents
static void mimi_encode(pocket_tts_context* pctx, const float* pcm, int n_samples, float** latent_out,
                        int* n_frames_out) {
    const auto& m = pctx->model;
    const auto& mi = m.mimi_hp;
    const auto& se = m.seanet_enc;
    const int OD = (int)mi.outer_dim; // 512
    const int LD = (int)mi.inner_dim; // 32

    *latent_out = nullptr;
    *n_frames_out = 0;

    if (!m.has_voice_cloning || !se.initial_conv_w) {
        fprintf(stderr, "pocket_tts: mimi_encode requires voice-cloning encoder weights\n");
        return;
    }

    // 1. Pad PCM to hop_length multiple
    int hop = (int)mi.hop_length(); // 120
    int n_padded = ((n_samples + hop - 1) / hop) * hop;
    std::vector<float> pcm_padded(n_padded, 0.0f);
    memcpy(pcm_padded.data(), pcm, n_samples * sizeof(float));

    // 2. SEANet encoder (all convolutions are causal)
    int T_cur = n_padded;
    int C_cur = 1; // mono PCM

    // Encoder ratios are reversed from decoder: [4, 5, 6]
    std::vector<int> enc_ratios(mi.seanet_ratios.rbegin(), mi.seanet_ratios.rend());
    int n_filters = (int)mi.seanet_n_filters;        // 64
    int seanet_K = (int)mi.seanet_kernel_size;       // 7
    int res_K = (int)mi.seanet_residual_kernel_size; // 3
    uint32_t n_res = mi.seanet_n_residual_layers;    // 1

    // Initial conv: (1 -> n_filters, kernel=7, causal)
    int pad_init = seanet_K - 1; // causal left padding
    int C_init = n_filters;
    std::vector<float> enc_buf(C_init * T_cur);

    // Input is single-channel: (1, T_cur)
    conv1d_eager(enc_buf.data(), pcm_padded.data(), C_cur, T_cur, se.initial_conv_w, se.initial_conv_b, 1, pad_init, 0);
    C_cur = C_init;

    // Encoder stages: {ResBlock → ELU → stride_conv} × n_stages
    int mult = 1;
    for (size_t s = 0; s < enc_ratios.size(); s++) {
        int ratio = enc_ratios[s];
        const auto& stage = se.stages[s];

        // Residual blocks
        for (size_t r = 0; r < stage.resblocks.size(); r++) {
            const auto& rb = stage.resblocks[r];
            int dilation = 1;
            for (uint32_t d = 0; d < r; d++)
                dilation *= (int)mi.seanet_dilation_base;

            if (rb.conv0_w && rb.conv1_w) {
                int C_hidden = C_cur / (int)mi.seanet_compress;
                // ELU → dilated conv → ELU → conv1 + residual
                std::vector<float> act_buf(C_cur * T_cur);
                for (int i = 0; i < C_cur * T_cur; i++)
                    act_buf[i] = elu_f(enc_buf[i]);

                int eff_K = (res_K - 1) * dilation + 1;
                int pad_d = eff_K - 1; // causal
                std::vector<float> h(C_hidden * T_cur);
                conv1d_eager(h.data(), act_buf.data(), C_cur, T_cur, rb.conv0_w, rb.conv0_b, 1, pad_d, 0);
                int C_h = (int)rb.conv0_w->ne[2];

                for (int i = 0; i < C_h * T_cur; i++)
                    h[i] = elu_f(h[i]);

                int K1 = (int)rb.conv1_w->ne[0];
                int pad1 = K1 - 1; // causal
                std::vector<float> h2(C_cur * T_cur);
                conv1d_eager(h2.data(), h.data(), C_h, T_cur, rb.conv1_w, rb.conv1_b, 1, pad1, 0);

                // Residual add
                for (int i = 0; i < C_cur * T_cur; i++)
                    enc_buf[i] += h2[i];
            }
        }

        // ELU
        for (int i = 0; i < C_cur * T_cur; i++)
            enc_buf[i] = elu_f(enc_buf[i]);

        // Stride conv (downsample): causal padding
        if (stage.conv_w) {
            mult += 1;
            int C_out = n_filters * (1 << (s + 1)); // 128, 256, 512
            int K_s = (int)stage.conv_w->ne[0];
            int pad_s = K_s - ratio; // causal pad for strided conv: kernel - stride
            int T_new = (T_cur + pad_s - K_s) / ratio + 1;
            std::vector<float> down_buf(C_out * T_new);
            conv1d_eager(down_buf.data(), enc_buf.data(), C_cur, T_cur, stage.conv_w, stage.conv_b, ratio, pad_s, 0);
            enc_buf = std::move(down_buf);
            T_cur = T_new;
            C_cur = C_out;
        }
    }

    // Final: ELU → conv (C_cur → OD=512, kernel=last_kernel_size=3, causal)
    for (int i = 0; i < C_cur * T_cur; i++)
        enc_buf[i] = elu_f(enc_buf[i]);

    int last_K = (int)mi.seanet_last_kernel_size;
    int pad_last = last_K - 1; // causal
    std::vector<float> enc_out(OD * T_cur);
    if (se.final_conv_w) {
        conv1d_eager(enc_out.data(), enc_buf.data(), C_cur, T_cur, se.final_conv_w, se.final_conv_b, 1, pad_last, 0);
    }
    int T_enc = T_cur; // encoder frame rate (200 Hz for 24kHz / 120 hop)

    if (pctx->verbosity >= 2)
        fprintf(stderr, "pocket_tts: seanet encode: %d samples → %d frames (C=%d)\n", n_samples, T_enc, OD);

    // 3. Encoder transformer (2L, causal, RoPE, LayerScale)
    // Convert to seq-first: (T_enc, OD)
    std::vector<float> xfmr_seq(T_enc * OD);
    for (int t = 0; t < T_enc; t++)
        for (int c = 0; c < OD; c++)
            xfmr_seq[t * OD + c] = enc_out[c * T_enc + t];

    // Apply input projection if present
    if (m.enc_xfmr_input_proj) {
        std::vector<float> proj(T_enc * OD);
        for (int t = 0; t < T_enc; t++)
            linear_f32(&proj[t * OD], &xfmr_seq[t * OD], m.enc_xfmr_input_proj, nullptr, OD, OD);
        xfmr_seq = std::move(proj);
    }

    mimi_transformer_forward(m.enc_transformer_layers, xfmr_seq.data(), T_enc, OD, (int)mi.xfmr_num_heads,
                             (int)mi.xfmr_dim_feedforward, (int)mi.xfmr_context);

    // Apply output projection if present
    if (m.enc_xfmr_output_proj) {
        std::vector<float> proj(T_enc * OD);
        for (int t = 0; t < T_enc; t++)
            linear_f32(&proj[t * OD], &xfmr_seq[t * OD], m.enc_xfmr_output_proj, nullptr, OD, OD);
        xfmr_seq = std::move(proj);
    }

    // Convert back to channels-first: (OD, T_enc)
    std::vector<float> enc_cf(OD * T_enc);
    for (int t = 0; t < T_enc; t++)
        for (int c = 0; c < OD; c++)
            enc_cf[c * T_enc + t] = xfmr_seq[t * OD + c];

    // 4. Downsample: Conv1d(OD→LD, kernel=downsample_stride*2, stride=downsample_stride)
    // Causal: pad_left = kernel - stride, pad_right = 0
    if (m.downsample_conv_w) {
        int DS = (int)mi.downsample_stride();
        int K_ds = (int)m.downsample_conv_w->ne[0];
        int pad_ds = K_ds - DS; // causal
        int T_down = (T_enc + pad_ds - K_ds) / DS + 1;
        std::vector<float> down_buf(LD * T_down);
        conv1d_eager(down_buf.data(), enc_cf.data(), OD, T_enc, m.downsample_conv_w, m.downsample_conv_b, DS, pad_ds,
                     0);

        // Output: (T_down, LD) — allocate and return
        *n_frames_out = T_down;
        *latent_out = (float*)malloc((size_t)T_down * LD * sizeof(float));
        // Convert from channels-first (LD, T_down) to frame-major (T_down, LD)
        for (int t = 0; t < T_down; t++)
            for (int d = 0; d < LD; d++)
                (*latent_out)[t * LD + d] = down_buf[d * T_down + t];

        if (pctx->verbosity >= 1)
            fprintf(stderr, "pocket_tts: mimi encode: %d samples → %d latent frames\n", n_samples, T_down);
    }
}

// ── SentencePiece tokenizer ──────────────────────────────────────
// Uses core_spm::tokenize (Viterbi unigram segmenter) with the vocab
// and scores loaded from the GGUF tokenizer.ggml.tokens/scores arrays.
static std::vector<int32_t> tokenize_text(pocket_tts_context* pctx, const char* text) {
    const auto& m = pctx->model;
    if (m.token_to_id.empty()) {
        fprintf(stderr, "pocket_tts: tokenizer not loaded, cannot tokenize\n");
        return {};
    }

    core_spm::Config cfg;
    cfg.unk_id = m.unk_id;
    cfg.merge_consecutive_unk = false; // pocket-tts SPM does not merge
    cfg.unk_penalty = -100.0f;

    return core_spm::tokenize(std::string(text), m.token_to_id, m.spm_scores, cfg,
                              /*prepend_space=*/true);
}

} // namespace

// ── Public C API ───────────────────────────────────────────────────

struct pocket_tts_context_params pocket_tts_context_default_params(void) {
    return {
        /* n_threads       */ 4,
        /* verbosity       */ 1,
        /* use_gpu         */ true,
        /* temperature     */ 0.7f,
        /* seed            */ 0,
        /* lsd_decode_steps */ 1,
        /* noise_clamp     */ 3.0f,
        /* eos_threshold   */ 0.5f,
        /* max_audio_frames */ 0,
    };
}

struct pocket_tts_context* pocket_tts_init_from_file(const char* path_model, struct pocket_tts_context_params params) {
    if (!path_model)
        return nullptr;

    auto* ctx = new pocket_tts_context();
    ctx->params = params;
    ctx->verbosity = params.verbosity;

    // Seed RNG
    if (params.seed != 0) {
        ctx->rng.seed((uint32_t)params.seed);
    } else {
        std::random_device rd;
        ctx->rng.seed(rd());
    }

    // Load GGUF
    struct ggml_context* ggml_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ggml_ctx,
    };

    struct gguf_context* meta = gguf_init_from_file(path_model, gguf_params);
    if (!meta) {
        fprintf(stderr, "pocket_tts: failed to load GGUF: %s\n", path_model);
        delete ctx;
        return nullptr;
    }

    // Verify architecture
    std::string arch = gguf_get_str(meta, "general.architecture", "");
    if (arch != "pocket-tts") {
        fprintf(stderr, "pocket_tts: expected arch 'pocket-tts', got '%s'\n", arch.c_str());
        gguf_free(meta);
        delete ctx;
        return nullptr;
    }

    // Load hyperparameters
    if (!load_hparams(meta, ctx->model)) {
        fprintf(stderr, "pocket_tts: failed to load hyperparameters\n");
        gguf_free(meta);
        delete ctx;
        return nullptr;
    }

    // Load tokenizer
    load_tokenizer(meta, ggml_ctx, ctx->model);

    // Load tensors from the GGML context that was allocated by gguf_init
    if (!load_flow_lm_tensors(ggml_ctx, ctx->model)) {
        fprintf(stderr, "pocket_tts: failed to load FlowLM tensors\n");
        gguf_free(meta);
        delete ctx;
        return nullptr;
    }

    load_mimi_decoder_tensors(ggml_ctx, ctx->model);
    load_mimi_encoder_tensors(ggml_ctx, ctx->model);

    if (ctx->verbosity >= 1) {
        const auto& h = ctx->model.flow_lm_hp;
        const auto& mi = ctx->model.mimi_hp;
        fprintf(stderr, "pocket_tts: loaded model\n");
        fprintf(stderr, "  FlowLM: %u layers, %u heads, %u dim, latent=%u\n", h.num_layers, h.num_heads, h.d_model,
                h.latent_dim);
        fprintf(stderr, "  Flow head: %u dim, %u depth\n", ctx->model.flow_head_hp.flow_dim,
                ctx->model.flow_head_hp.flow_depth);
        fprintf(stderr, "  Mimi: %u Hz, frame_rate=%.1f, hop=%u\n", mi.sample_rate, mi.frame_rate(), mi.hop_length());
        fprintf(stderr, "  Voice cloning: %s\n", ctx->model.has_voice_cloning ? "yes" : "no");
        fprintf(stderr, "  Tokenizer: %zu pieces (unk_id=%d)\n", ctx->model.id_to_token.size(), ctx->model.unk_id);
        fprintf(stderr, "  Mimi tensors: quant_proj=%s upsample=%s seanet_init=%s seanet_final=%s\n",
                ctx->model.quant_proj_w ? "ok" : "MISSING", ctx->model.upsample_conv_w ? "ok" : "MISSING",
                ctx->model.seanet_dec.initial_conv_w ? "ok" : "MISSING",
                ctx->model.seanet_dec.final_conv_w ? "ok" : "MISSING");
        fprintf(stderr, "  Backbone layers[0]: in_proj=%s out_proj=%s ffn1=%s ffn2=%s\n",
                ctx->model.backbone_layers[0].attn_in_proj ? "ok" : "MISSING",
                ctx->model.backbone_layers[0].attn_out_proj ? "ok" : "MISSING",
                ctx->model.backbone_layers[0].ffn_linear1 ? "ok" : "MISSING",
                ctx->model.backbone_layers[0].ffn_linear2 ? "ok" : "MISSING");
    }

    gguf_free(meta);
    // Note: ggml_ctx and tensor data are still alive via the buffer

    // Initialize CPU backend for ggml graph compute
    ctx->backend_cpu = ggml_backend_cpu_init();

    return ctx;
}

void pocket_tts_free(struct pocket_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    // TODO: free ggml weight buffers
    delete ctx;
}

float* pocket_tts_synthesize(struct pocket_tts_context* ctx, const char* text, int* n_samples) {
    if (!ctx || !text || !n_samples)
        return nullptr;
    *n_samples = 0;

    const auto& m = ctx->model;
    const auto& hp = m.flow_lm_hp;
    const int D = (int)hp.d_model;
    const int LD = (int)hp.latent_dim;
    const int lsd_steps = ctx->params.lsd_decode_steps;

    // --- diff/debug hooks (env-gated; paths from env value, never hardcoded) ---
    // POCKET_DUMP_DIR=d : write stage dumps under dir d
    //   cpp_token_ids.i32    — token ID sequence
    //   cpp_text_emb.f32     — (n_tokens, d_model) text embeddings
    //   cpp_latents.f32      — (n_frames, latent_dim) AR latent sequence
    //   cpp_pcm.f32          — final PCM
    // POCKET_FORCE_LATENTS=f : teacher-force AR latents from file f
    const char* pocket_dump_dir = getenv("POCKET_DUMP_DIR");
    const char* pocket_force_lat = getenv("POCKET_FORCE_LATENTS");
    auto dump_path = [&](const char* name) -> std::string {
        return std::string(pocket_dump_dir ? pocket_dump_dir : ".") + "/" + name;
    };

    // 1. Tokenize text
    std::vector<int32_t> tokens = tokenize_text(ctx, text);
    if (tokens.empty()) {
        if (ctx->verbosity >= 1)
            fprintf(stderr, "pocket_tts: empty text after tokenization\n");
        return nullptr;
    }

    if (ctx->verbosity >= 2)
        fprintf(stderr, "pocket_tts: tokenized %zu tokens\n", tokens.size());

    // Dump token IDs
    if (pocket_dump_dir) {
        FILE* f = fopen(dump_path("cpp_token_ids.i32").c_str(), "wb");
        if (f) {
            fwrite(tokens.data(), sizeof(int32_t), tokens.size(), f);
            fclose(f);
        }
        fprintf(stderr, "POCKET_DUMP_DIR: wrote cpp_token_ids.i32 (%zu tokens)\n", tokens.size());
        // Also print IDs for quick inspection
        fprintf(stderr, "  tokens:");
        for (size_t i = 0; i < tokens.size() && i < 30; i++)
            fprintf(stderr, " %d", tokens[i]);
        if (tokens.size() > 30)
            fprintf(stderr, " ...");
        fprintf(stderr, "\n");
    }

    // Determine max audio frames
    int max_frames = ctx->params.max_audio_frames;
    if (max_frames <= 0) {
        // Heuristic: ~12.5 frames per second, ~0.5s per token, min 25 frames (2s)
        max_frames = std::max(25, (int)(tokens.size() * 6));
        max_frames = std::min(max_frames, 500); // ~40s max
    }
    // Allow env override for testing
    if (const char* mf_env = getenv("POCKET_MAX_FRAMES")) {
        max_frames = std::atoi(mf_env);
        if (max_frames <= 0)
            max_frames = 25;
    }

    // 2. Initialize KV cache for backbone
    int n_voice = ctx->has_voice_state ? ctx->n_voice_frames : 0;
    int max_seq = n_voice + (int)tokens.size() + max_frames + 2;
    kv_cache_init(ctx->backbone_kv, (int)hp.num_layers, max_seq, (int)hp.num_heads, (int)hp.head_dim());
    kv_cache_reset(ctx->backbone_kv);

    // 2b. Voice conditioning prefill (if set via set_voice)
    if (ctx->has_voice_state && n_voice > 0) {
        for (int f = 0; f < n_voice; f++) {
            const float* vc = &ctx->voice_conditioning[f * D];
            std::vector<float> backbone_out(D);
            backbone_step(ctx, vc, backbone_out.data());
            ctx->backbone_kv.offset++;
        }
        if (ctx->verbosity >= 1)
            fprintf(stderr, "pocket_tts: voice conditioning prefill: %d frames\n", n_voice);
    }

    // 3. Prefill: embed text tokens and run through backbone
    const float* embed_data = tensor_f32_data(m.conditioner_embed);
    std::vector<float> text_emb_dump; // for dump
    if (pocket_dump_dir)
        text_emb_dump.reserve(tokens.size() * D);

    for (int i = 0; i < (int)tokens.size(); i++) {
        int32_t tok = tokens[i];
        if (tok < 0 || tok > (int32_t)hp.n_bins)
            tok = 0; // clamp to valid range

        // Look up embedding (conditioner_embed is (n_bins+1, lut_dim))
        const float* emb = &embed_data[tok * D];

        if (pocket_dump_dir)
            text_emb_dump.insert(text_emb_dump.end(), emb, emb + D);

        std::vector<float> backbone_out(D);
        backbone_step(ctx, emb, backbone_out.data());
        ctx->backbone_kv.offset++;
    }

    if (ctx->verbosity >= 2)
        fprintf(stderr, "pocket_tts: prefill done, %zu positions cached\n", tokens.size());

    // Dump text embeddings
    if (pocket_dump_dir && !text_emb_dump.empty()) {
        FILE* f = fopen(dump_path("cpp_text_emb.f32").c_str(), "wb");
        if (f) {
            fwrite(text_emb_dump.data(), sizeof(float), text_emb_dump.size(), f);
            fclose(f);
        }
        fprintf(stderr, "POCKET_DUMP_DIR: wrote cpp_text_emb.f32 (%zu x %d)\n", tokens.size(), D);
    }

    // 4. AR loop: generate audio latents
    std::vector<float> latent_sequence; // (n_gen_frames * LD)
    latent_sequence.reserve(max_frames * LD);

    // Teacher-force latents if requested (for diff-testing)
    std::vector<float> forced_latents;
    if (pocket_force_lat) {
        FILE* ff = fopen(pocket_force_lat, "rb");
        if (ff) {
            fseek(ff, 0, SEEK_END);
            long sz = ftell(ff);
            fseek(ff, 0, SEEK_SET);
            forced_latents.resize(sz / sizeof(float));
            if (fread(forced_latents.data(), sizeof(float), forced_latents.size(), ff) != forced_latents.size())
                forced_latents.clear();
            fclose(ff);
            int forced_frames = (int)(forced_latents.size() / LD);
            max_frames = forced_frames;
            fprintf(stderr, "POCKET_FORCE_LATENTS: teacher-forcing %d frames from %s\n", forced_frames,
                    pocket_force_lat);
        }
    }

    // BOS: use the learned bos_emb projected through input_linear
    std::vector<float> bos_proj(D);
    const float* bos_data = tensor_f32_data(m.bos_emb);
    linear_f32(bos_proj.data(), bos_data, m.input_linear, nullptr, D, LD);

    // First AR step uses BOS embedding
    std::vector<float> prev_input(D);
    vec_copy(prev_input.data(), bos_proj.data(), D);

    int frames_after_eos = 0;
    const int eos_grace_frames = 3; // continue a few frames after EOS for tail

    std::normal_distribution<float> noise_dist(0.0f, 1.0f);

    for (int frame = 0; frame < max_frames; frame++) {
        // Run backbone on current input
        std::vector<float> backbone_out(D);
        backbone_step(ctx, prev_input.data(), backbone_out.data());
        ctx->backbone_kv.offset++;

        // Dump step-0 backbone output for diff testing
        if (frame == 0 && pocket_dump_dir) {
            FILE* f = fopen(dump_path("cpp_backbone_out0.f32").c_str(), "wb");
            if (f) {
                fwrite(backbone_out.data(), sizeof(float), D, f);
                fclose(f);
            }
            fprintf(stderr, "POCKET_DUMP_DIR: wrote cpp_backbone_out0.f32 (%d floats)\n", D);
        }

        // Check EOS (skip if teacher-forcing)
        if (forced_latents.empty() && check_eos(ctx, backbone_out.data())) {
            frames_after_eos++;
            if (frames_after_eos >= eos_grace_frames) {
                if (ctx->verbosity >= 2)
                    fprintf(stderr, "pocket_tts: EOS at frame %d\n", frame);
                break;
            }
        }

        std::vector<float> latent(LD);
        if (!forced_latents.empty()) {
            // Teacher-force: use reference latent
            vec_copy(latent.data(), &forced_latents[frame * LD], LD);
        } else {
            // Sample noise
            std::vector<float> noise(LD);
            // Allow forcing step-0 noise from file for diff testing
            bool noise_forced = false;
            if (frame == 0) {
                const char* nf = getenv("POCKET_FORCE_NOISE");
                if (nf) {
                    FILE* fn = fopen(nf, "rb");
                    if (fn) {
                        if (fread(noise.data(), sizeof(float), LD, fn) == (size_t)LD)
                            noise_forced = true;
                        fclose(fn);
                        fprintf(stderr, "POCKET_FORCE_NOISE: loaded %d floats from %s\n", LD, nf);
                    }
                }
            }
            if (!noise_forced) {
                float temp = ctx->params.temperature;
                float noise_clamp = ctx->params.noise_clamp;
                for (int i = 0; i < LD; i++) {
                    float n = noise_dist(ctx->rng) * std::sqrt(temp);
                    if (noise_clamp > 0.0f)
                        n = std::max(-noise_clamp, std::min(noise_clamp, n));
                    noise[i] = n;
                }
            }

            // Dump step-0 noise
            if (frame == 0 && pocket_dump_dir) {
                std::string p = dump_path("cpp_noise0.f32");
                FILE* f = fopen(p.c_str(), "wb");
                if (f) {
                    fwrite(noise.data(), sizeof(float), LD, f);
                    fclose(f);
                }
            }

            // Flow net: predict latent from backbone output + noise
            flow_net_forward(ctx, backbone_out.data(), noise.data(), lsd_steps, latent.data());
        }

        // Append to sequence
        latent_sequence.insert(latent_sequence.end(), latent.begin(), latent.end());

        // Project latent back to d_model for next step input
        linear_f32(prev_input.data(), latent.data(), m.input_linear, nullptr, D, LD);

        if (ctx->verbosity >= 2 && (frame + 1) % 50 == 0)
            fprintf(stderr, "pocket_tts: generated %d frames\n", frame + 1);
    }

    int n_gen_frames = (int)(latent_sequence.size() / LD);
    if (ctx->verbosity >= 1)
        fprintf(stderr, "pocket_tts: generated %d audio frames (%.1f s at 12.5 Hz)\n", n_gen_frames,
                n_gen_frames / 12.5f);

    // Dump latent sequence
    if (pocket_dump_dir && n_gen_frames > 0) {
        FILE* f = fopen(dump_path("cpp_latents.f32").c_str(), "wb");
        if (f) {
            fwrite(latent_sequence.data(), sizeof(float), latent_sequence.size(), f);
            fclose(f);
        }
        fprintf(stderr, "POCKET_DUMP_DIR: wrote cpp_latents.f32 (%d x %d)\n", n_gen_frames, LD);
    }

    if (n_gen_frames == 0)
        return nullptr;

    // 5. Mimi decode: latent sequence -> PCM
    float* pcm = nullptr;
    int pcm_samples = 0;
    // --no-gpu (use_gpu=false) forces legacy manual Mimi path; env var POCKET_MANUAL_MIMI=1 also works.
    if (!ctx->params.use_gpu || getenv("POCKET_MANUAL_MIMI")) {
        mimi_decode(ctx, latent_sequence.data(), n_gen_frames, &pcm, &pcm_samples);
    } else {
        mimi_decode_ggml(ctx, latent_sequence.data(), n_gen_frames, &pcm, &pcm_samples);
    }

    if (!pcm || pcm_samples <= 0) {
        if (ctx->verbosity >= 1)
            fprintf(stderr, "pocket_tts: mimi_decode failed\n");
        return nullptr;
    }

    if (ctx->verbosity >= 1)
        fprintf(stderr, "pocket_tts: decoded %d PCM samples (%.2f s at %u Hz)\n", pcm_samples,
                (float)pcm_samples / m.mimi_hp.sample_rate, m.mimi_hp.sample_rate);

    // Dump PCM
    if (pocket_dump_dir && pcm_samples > 0) {
        FILE* f = fopen(dump_path("cpp_pcm.f32").c_str(), "wb");
        if (f) {
            fwrite(pcm, sizeof(float), pcm_samples, f);
            fclose(f);
        }
        fprintf(stderr, "POCKET_DUMP_DIR: wrote cpp_pcm.f32 (%d samples)\n", pcm_samples);
    }

    *n_samples = pcm_samples;
    return pcm;
}

void pocket_tts_pcm_free(float* pcm) {
    free(pcm);
}

int pocket_tts_set_voice(struct pocket_tts_context* ctx, const float* ref_pcm_24khz, int n_ref_samples) {
    if (!ctx || !ref_pcm_24khz || n_ref_samples <= 0)
        return -1;
    if (!ctx->model.has_voice_cloning) {
        fprintf(stderr, "pocket_tts: voice cloning requires encoder weights in GGUF\n");
        return -1;
    }

    const auto& m = ctx->model;
    const int D = (int)m.flow_lm_hp.d_model;
    const int LD = (int)m.mimi_hp.inner_dim;

    // 1. Encode reference audio -> latents
    float* ref_latents = nullptr;
    int n_ref_frames = 0;
    mimi_encode(ctx, ref_pcm_24khz, n_ref_samples, &ref_latents, &n_ref_frames);
    if (!ref_latents || n_ref_frames <= 0) {
        fprintf(stderr, "pocket_tts: mimi_encode failed for voice reference\n");
        return -1;
    }

    // 2. Project through speaker_proj: (n_frames, inner_dim) -> (n_frames, d_model)
    ctx->voice_conditioning.resize(n_ref_frames * D);
    for (int f = 0; f < n_ref_frames; f++) {
        linear_f32(&ctx->voice_conditioning[f * D], &ref_latents[f * LD], m.speaker_proj, nullptr, D, LD);
    }
    free(ref_latents);

    // 3. If insert_bos_before_voice, prepend bos_before_voice
    if (m.flow_lm_hp.insert_bos_before_voice && m.bos_before_voice) {
        std::vector<float> new_cond(D + n_ref_frames * D);
        const float* bos_v = tensor_f32_data(m.bos_before_voice);
        memcpy(new_cond.data(), bos_v, D * sizeof(float));
        memcpy(new_cond.data() + D, ctx->voice_conditioning.data(), n_ref_frames * D * sizeof(float));
        ctx->voice_conditioning = std::move(new_cond);
        n_ref_frames += 1;
    }

    ctx->n_voice_frames = n_ref_frames;
    ctx->has_voice_state = true;

    if (ctx->verbosity >= 1)
        fprintf(stderr, "pocket_tts: voice conditioning set (%d frames)\n", n_ref_frames);

    return 0;
}

void pocket_tts_clear_voice(struct pocket_tts_context* ctx) {
    if (!ctx)
        return;
    ctx->has_voice_state = false;
    ctx->voice_conditioning.clear();
    ctx->n_voice_frames = 0;
}

void pocket_tts_set_temperature(struct pocket_tts_context* ctx, float temp) {
    if (ctx)
        ctx->params.temperature = temp;
}

void pocket_tts_set_seed(struct pocket_tts_context* ctx, uint64_t seed) {
    if (!ctx)
        return;
    ctx->params.seed = seed;
    if (seed != 0) {
        ctx->rng.seed((uint32_t)seed);
    }
}

int pocket_tts_sample_rate(struct pocket_tts_context* ctx) {
    if (!ctx)
        return 24000;
    return (int)ctx->model.mimi_hp.sample_rate;
}
