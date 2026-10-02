// miotts.cpp — MioTTS backend (Aratako/MioTTS-{0.6B,1.7B} + MioCodec).
//
// LLM: standard Qwen3 forward with KV cache (reuses core_attn::kv_self_attn).
// Codec: FSQ dequant → wave_prenet → conv_upsample → ResNet → wave_decoder
//        (AdaLN-Zero) → ResNet → iSTFT → model-rate waveform.
//
// The LLM generates speech tokens from text; the codec converts them to audio.
// Voice cloning injects a 128-d global embedding at codec decode time.

#include "miotts.h"

#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/istft.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/ggml_cpu_backend.h"

// ── Hyperparameters ─────────────────────────────────────────────────

struct miotts_hparams {
    // LLM
    uint32_t n_layers = 28;
    uint32_t n_heads = 16;
    uint32_t n_kv_heads = 8;
    uint32_t d_model = 1024;
    uint32_t d_ff = 3072;
    uint32_t vocab_size = 164480;
    uint32_t head_dim = 64;
    uint32_t max_pos = 32768;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;

    // Speech token range in the vocabulary
    uint32_t speech_token_start = 151669;
    uint32_t speech_token_end = 164469;
    uint32_t eos_token_id = 151645; // <|endoftext|>

    // Codec
    uint32_t codec_sample_rate = 24000;
    uint32_t codec_frame_rate = 25;
    uint32_t codec_n_fft = 1920;
    uint32_t codec_hop_length = 480;
    uint32_t codec_codebook_size = 12800;
    int32_t fsq_levels[5] = {8, 8, 8, 5, 5};
    uint32_t codec_global_dim = 128;
    uint32_t codec_wave_dim = 512;
    uint32_t codec_wave_prenet_layers = 6;
    uint32_t codec_wave_decoder_layers = 8;
};

// ── LLM weight block ────────────────────────────────────────────────

struct miotts_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr;
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

// ── Context ─────────────────────────────────────────────────────────

// ── Codec layer weight structures (before context for lambda visibility) ────

struct miotts_codec_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    ggml_tensor* wq = nullptr;
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
    ggml_tensor* ffn_w1 = nullptr;
    ggml_tensor* ffn_w2 = nullptr;
    ggml_tensor* ffn_w3 = nullptr;
    // AdaLN-Zero (wave_decoder only)
    ggml_tensor* adaln_attn_w = nullptr;
    ggml_tensor* adaln_attn_b = nullptr;
    ggml_tensor* adaln_ffn_w = nullptr;
    ggml_tensor* adaln_ffn_b = nullptr;
};

struct miotts_resnet_block {
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* conv1_w = nullptr;
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* conv2_w = nullptr;
    ggml_tensor* conv2_b = nullptr;
};

struct miotts_context {
    miotts_context_params params;
    miotts_hparams hp;

    // Backend
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;

    // LLM weights
    ggml_tensor* token_embd = nullptr;
    ggml_tensor* output_norm = nullptr;
    ggml_tensor* output_w = nullptr; // lm_head (may be tied to token_embd)
    std::vector<miotts_layer> layers;

    // KV cache
    ggml_tensor* kv_k = nullptr; // [head_dim, n_kv_heads, max_ctx, n_layers]
    ggml_tensor* kv_v = nullptr;
    ggml_backend_buffer_t buf_kv = nullptr;

    // Codec weights (FSQ dequant projection)
    ggml_tensor* fsq_proj_out_w = nullptr; // (768, 5)
    ggml_tensor* fsq_proj_out_b = nullptr; // (768,)

    // Codec: wave_prenet (6 layers, 768d → 512d)
    std::vector<miotts_codec_layer> wave_prenet_layers;
    ggml_tensor* wave_prenet_norm_w = nullptr; // final LayerNorm before output_proj
    ggml_tensor* wave_prenet_norm_b = nullptr;
    ggml_tensor* wave_prenet_out_proj_w = nullptr;
    ggml_tensor* wave_prenet_out_proj_b = nullptr;

    // Codec: wave_decoder (8 layers, 512d, AdaLN-Zero conditioned on 128-d)
    std::vector<miotts_codec_layer> wave_decoder_layers;
    // wave_decoder final norm (AdaLN-Zero without gate)
    ggml_tensor* wave_decoder_norm_w = nullptr;
    ggml_tensor* wave_decoder_norm_b = nullptr;

    // Codec: conv_upsample (ConvTranspose1d, 512→512, k=2, s=2)
    ggml_tensor* conv_upsample_w = nullptr;
    ggml_tensor* conv_upsample_b = nullptr;

    // Codec: ResNet stacks (prior_net + post_net, 2 blocks each)
    std::vector<miotts_resnet_block> wave_prior_net;
    std::vector<miotts_resnet_block> wave_post_net;

    // Codec: UpSampler (v2 44.1kHz — ConvTranspose1d + SnakeBeta + ResNet, 2 stages)
    struct UpsampleStage {
        ggml_tensor* conv_w = nullptr; // fused weight_norm weight
        ggml_tensor* conv_b = nullptr;
        ggml_tensor* snake_alpha = nullptr;
        ggml_tensor* snake_beta = nullptr;
    };
    std::vector<UpsampleStage> upsampler_stages;
    std::vector<miotts_resnet_block> upsampler_resnets;
    ggml_tensor* upsampler_out_proj_w = nullptr;
    ggml_tensor* upsampler_out_proj_b = nullptr;
    ggml_tensor* upsampler_out_snake_alpha = nullptr;
    ggml_tensor* upsampler_out_snake_beta = nullptr;
    bool has_upsampler = false;

    // Codec: iSTFT head
    ggml_tensor* istft_out_w = nullptr;
    ggml_tensor* istft_out_b = nullptr;

    // Global embedding for voice cloning (128-d, set by miotts_set_reference)
    std::vector<float> global_embedding;

    // Tokenizer (Qwen3 BPE loaded from GGUF)
    struct {
        std::vector<std::string> id_to_token;
        std::unordered_map<std::string, int32_t> token_to_id;
        std::unordered_map<std::string, int32_t> merge_rank;
        bool loaded = false;
    } vocab;

    // Scheduler (handles weight buffer + compute buffer together)
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_t backend_cpu = nullptr; // CPU fallback for split graphs

    // Compute scratch (metadata arena for graph building)
    std::vector<uint8_t> compute_meta;

    // GGUF contexts (kept alive for weight buffer lifetime)
    ggml_context* ctx_weights = nullptr;
    ggml_context* ctx_kv = nullptr;

    ~miotts_context() {
        if (sched)
            ggml_backend_sched_free(sched);
        if (buf_kv)
            ggml_backend_buffer_free(buf_kv);
        if (buf_weights)
            ggml_backend_buffer_free(buf_weights);
        if (ctx_kv)
            ggml_free(ctx_kv);
        if (ctx_weights)
            ggml_free(ctx_weights);
        if (backend_cpu && backend_cpu != backend)
            ggml_backend_free(backend_cpu);
        if (backend)
            ggml_backend_free(backend);
    }
};

// ── Default params ──────────────────────────────────────────────────

miotts_context_params miotts_context_default_params(void) {
    return {
        /*n_threads*/ 4,
        /*verbosity*/ 1,
        /*use_gpu*/ false,
        /*temperature*/ 0.8f,
        /*seed*/ 0,
        /*max_tokens*/ 750, // 30s at 25Hz
        /*flash_attn*/ false,
    };
}

// ── Load model ──────────────────────────────────────────────────────

static bool load_hparams(gguf_context* meta, miotts_hparams& hp) {
    auto get_u32 = [&](const char* key, uint32_t def) -> uint32_t {
        int idx = gguf_find_key(meta, key);
        return idx >= 0 ? (uint32_t)gguf_get_val_u32(meta, idx) : def;
    };
    auto get_f32 = [&](const char* key, float def) -> float {
        int idx = gguf_find_key(meta, key);
        return idx >= 0 ? gguf_get_val_f32(meta, idx) : def;
    };

    hp.n_layers = get_u32("miotts.block_count", 28);
    hp.n_heads = get_u32("miotts.attention.head_count", 16);
    hp.n_kv_heads = get_u32("miotts.attention.head_count_kv", 8);
    hp.d_model = get_u32("miotts.embedding_length", 1024);
    hp.d_ff = get_u32("miotts.feed_forward_length", 3072);
    hp.vocab_size = get_u32("miotts.vocab_size", 164480);
    hp.head_dim = get_u32("miotts.head_dim", hp.d_model / hp.n_heads);
    hp.max_pos = get_u32("miotts.context_length", 32768);
    hp.rope_theta = get_f32("miotts.rope_theta", 1000000.0f);
    hp.rms_norm_eps = get_f32("miotts.rms_norm_eps", 1e-6f);

    hp.speech_token_start = get_u32("miotts.speech_token_start", 151669);
    hp.speech_token_end = get_u32("miotts.speech_token_end", 164469);
    hp.eos_token_id = get_u32("miotts.eos_token_id", 151645);

    hp.codec_sample_rate = get_u32("miotts.codec.sample_rate", 24000);
    hp.codec_frame_rate = get_u32("miotts.codec.frame_rate", 25);
    hp.codec_n_fft = get_u32("miotts.codec.n_fft", 1920);
    hp.codec_hop_length = get_u32("miotts.codec.hop_length", 480);
    hp.codec_codebook_size = get_u32("miotts.codec.codebook_size", 12800);
    hp.codec_global_dim = get_u32("miotts.codec.global_dim", 128);
    hp.codec_wave_dim = get_u32("miotts.codec.wave_dim", 512);
    hp.codec_wave_prenet_layers = get_u32("miotts.codec.wave_prenet_layers", 6);
    hp.codec_wave_decoder_layers = get_u32("miotts.codec.wave_decoder_layers", 8);

    return true;
}

miotts_context* miotts_init_from_file(const char* path_model, miotts_context_params params) {
    auto c = std::make_unique<miotts_context>();
    c->params = params;

    // Init backend — via the process-global preference helper so
    // --gpu-backend (incl. the "cpu" short-circuit) is honoured here too;
    // this was the one direct init_best() bypass left in the tree.
    c->backend = crispasr_init_gpu_backend();
    if (!c->backend) {
        fprintf(stderr, "miotts: failed to init backend\n");
        return nullptr;
    }

    // Load GGUF
    gguf_init_params gip = {/*.no_alloc=*/true, /*.ctx=*/&c->ctx_weights};
    gguf_context* meta = gguf_init_from_file(path_model, gip);
    if (!meta) {
        fprintf(stderr, "miotts: failed to load '%s'\n", path_model);
        return nullptr;
    }

    if (!load_hparams(meta, c->hp)) {
        fprintf(stderr, "miotts: failed to read hyperparameters\n");
        gguf_free(meta);
        return nullptr;
    }

    const auto& hp = c->hp;
    if (params.verbosity >= 1) {
        fprintf(stderr, "miotts: %u layers, %u heads (%u KV), d=%u, vocab=%u\n", hp.n_layers, hp.n_heads, hp.n_kv_heads,
                hp.d_model, hp.vocab_size);
        fprintf(stderr, "miotts: codec=%uHz, FSQ codebook=%u, n_fft=%u\n", hp.codec_frame_rate, hp.codec_codebook_size,
                hp.codec_n_fft);
    }

    // Allocate weight buffer
    c->buf_weights = ggml_backend_alloc_ctx_tensors(c->ctx_weights, c->backend);
    if (!c->buf_weights) {
        fprintf(stderr, "miotts: failed to allocate weight buffer\n");
        gguf_free(meta);
        return nullptr;
    }

    // Load weights from GGUF into the allocated buffer
    {
        FILE* f = fopen(path_model, "rb");
        if (!f) {
            fprintf(stderr, "miotts: cannot open '%s'\n", path_model);
            gguf_free(meta);
            return nullptr;
        }
        const size_t data_offset = gguf_get_data_offset(meta);
        const int n_tensors = gguf_get_n_tensors(meta);
        for (int i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(meta, i);
            ggml_tensor* t = ggml_get_tensor(c->ctx_weights, name);
            if (!t)
                continue;
            const size_t offset = data_offset + gguf_get_tensor_offset(meta, i);
            const size_t nbytes = ggml_nbytes(t);
            fseek(f, (long)offset, SEEK_SET);
            std::vector<uint8_t> buf(nbytes);
            if (fread(buf.data(), 1, nbytes, f) != nbytes) {
                fprintf(stderr, "miotts: short read on tensor '%s'\n", name);
            }
            ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
        }
        fclose(f);
    }

    // Resolve LLM weight pointers
    auto T = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(c->ctx_weights, name); };

    c->token_embd = T("token_embd.weight");
    c->output_norm = T("output_norm.weight");
    c->output_w = T("output.weight");
    // Tied embeddings: if output.weight is missing, use token_embd
    if (!c->output_w)
        c->output_w = c->token_embd;

    c->layers.resize(hp.n_layers);
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        auto& b = c->layers[il];
        char buf[128];
        auto tn = [&](const char* suffix) {
            snprintf(buf, sizeof(buf), "blk.%u.%s", il, suffix);
            return T(buf);
        };
        b.attn_norm_w = tn("attn_norm.weight");
        b.attn_q_w = tn("attn_q.weight");
        b.attn_k_w = tn("attn_k.weight");
        b.attn_v_w = tn("attn_v.weight");
        b.attn_output_w = tn("attn_output.weight");
        b.attn_q_norm_w = tn("attn_q_norm.weight");
        b.attn_k_norm_w = tn("attn_k_norm.weight");
        b.ffn_norm_w = tn("ffn_norm.weight");
        b.ffn_gate_w = tn("ffn_gate.weight");
        b.ffn_up_w = tn("ffn_up.weight");
        b.ffn_down_w = tn("ffn_down.weight");
    }

    // Resolve codec FSQ weights
    c->fsq_proj_out_w = T("codec.local_quantizer.proj_out.weight");
    c->fsq_proj_out_b = T("codec.local_quantizer.proj_out.bias");

    // Resolve codec wave_prenet weights (6 layers)
    c->wave_prenet_layers.resize(hp.codec_wave_prenet_layers);
    for (uint32_t il = 0; il < hp.codec_wave_prenet_layers; il++) {
        auto& b = c->wave_prenet_layers[il];
        char buf[128];
        auto cn = [&](const char* suffix) {
            snprintf(buf, sizeof(buf), "codec.wave_prenet.layers.%u.%s", il, suffix);
            return T(buf);
        };
        b.attn_norm_w = cn("attention_norm.weight");
        b.attn_norm_b = cn("attention_norm.bias");
        b.wq = cn("attention.wq.weight");
        b.wk = cn("attention.wk.weight");
        b.wv = cn("attention.wv.weight");
        b.wo = cn("attention.wo.weight");
        b.ffn_norm_w = cn("ffn_norm.weight");
        b.ffn_norm_b = cn("ffn_norm.bias");
        b.ffn_w1 = cn("feed_forward.w1.weight");
        b.ffn_w2 = cn("feed_forward.w2.weight");
        b.ffn_w3 = cn("feed_forward.w3.weight");
    }
    c->wave_prenet_norm_w = T("codec.wave_prenet.norm.weight");
    c->wave_prenet_norm_b = T("codec.wave_prenet.norm.bias");
    c->wave_prenet_out_proj_w = T("codec.wave_prenet.output_proj.weight");
    c->wave_prenet_out_proj_b = T("codec.wave_prenet.output_proj.bias");

    // Resolve codec wave_decoder weights (8 layers, with AdaLN-Zero)
    c->wave_decoder_layers.resize(hp.codec_wave_decoder_layers);
    for (uint32_t il = 0; il < hp.codec_wave_decoder_layers; il++) {
        auto& b = c->wave_decoder_layers[il];
        char buf[128];
        auto cn = [&](const char* suffix) {
            snprintf(buf, sizeof(buf), "codec.wave_decoder.layers.%u.%s", il, suffix);
            return T(buf);
        };
        b.wq = cn("attention.wq.weight");
        b.wk = cn("attention.wk.weight");
        b.wv = cn("attention.wv.weight");
        b.wo = cn("attention.wo.weight");
        b.ffn_w1 = cn("feed_forward.w1.weight");
        b.ffn_w2 = cn("feed_forward.w2.weight");
        b.ffn_w3 = cn("feed_forward.w3.weight");
        // AdaLN-Zero conditioning (produces shift, scale, gate from 128-d embedding)
        b.adaln_attn_w = cn("attention_norm.condition_proj.1.weight");
        b.adaln_attn_b = cn("attention_norm.condition_proj.1.bias");
        b.adaln_ffn_w = cn("ffn_norm.condition_proj.1.weight");
        b.adaln_ffn_b = cn("ffn_norm.condition_proj.1.bias");
    }

    // Decoder final norm (AdaLN-Zero: condition_proj → shift, scale, no gate)
    c->wave_decoder_norm_w = T("codec.wave_decoder.norm.condition_proj.1.weight");
    c->wave_decoder_norm_b = T("codec.wave_decoder.norm.condition_proj.1.bias");

    // Resolve ResNet stacks
    auto load_resnet = [&](std::vector<miotts_resnet_block>& blocks, const char* prefix, int n) {
        blocks.resize(n);
        for (int i = 0; i < n; i++) {
            auto& b = blocks[i];
            char buf[128];
            auto rn = [&](const char* suffix) {
                snprintf(buf, sizeof(buf), "codec.%s.blocks.%d.%s", prefix, i, suffix);
                return T(buf);
            };
            b.norm1_w = rn("norm1.weight");
            b.norm1_b = rn("norm1.bias");
            b.conv1_w = rn("conv1.weight");
            b.conv1_b = rn("conv1.bias");
            b.norm2_w = rn("norm2.weight");
            b.norm2_b = rn("norm2.bias");
            b.conv2_w = rn("conv2.weight");
            b.conv2_b = rn("conv2.bias");
        }
    };
    load_resnet(c->wave_prior_net, "wave_prior_net", 2);
    load_resnet(c->wave_post_net, "wave_post_net", 2);

    // Resolve conv_upsample + iSTFT head
    c->conv_upsample_w = T("codec.wave_conv_upsample.weight");
    c->conv_upsample_b = T("codec.wave_conv_upsample.bias");
    c->istft_out_w = T("codec.istft_head.out.weight");
    c->istft_out_b = T("codec.istft_head.out.bias");

    // UpSampler (v2 44.1kHz codec — optional, not present in 24kHz codec)
    {
        ggml_tensor* us0_w = T("codec.wave_upsampler.upsample_layers.0.weight");
        if (us0_w) {
            c->has_upsampler = true;
            c->upsampler_stages.resize(2);
            for (int i = 0; i < 2; i++) {
                auto& s = c->upsampler_stages[i];
                char buf[128];
                snprintf(buf, sizeof(buf), "codec.wave_upsampler.upsample_layers.%d.weight", i);
                s.conv_w = T(buf);
                snprintf(buf, sizeof(buf), "codec.wave_upsampler.upsample_layers.%d.bias", i);
                s.conv_b = T(buf);
                snprintf(buf, sizeof(buf), "codec.wave_upsampler.snake_activations.%d.alpha", i);
                s.snake_alpha = T(buf);
                snprintf(buf, sizeof(buf), "codec.wave_upsampler.snake_activations.%d.beta", i);
                s.snake_beta = T(buf);
            }
            c->upsampler_resnets.resize(2);
            for (int i = 0; i < 2; i++) {
                auto& b = c->upsampler_resnets[i];
                char buf[128];
                auto rn = [&](const char* suffix) {
                    snprintf(buf, sizeof(buf), "codec.wave_upsampler.resnet_blocks.%d.%s", i, suffix);
                    return T(buf);
                };
                b.norm1_w = rn("norm1.weight");
                b.norm1_b = rn("norm1.bias");
                b.conv1_w = rn("conv1.weight");
                b.conv1_b = rn("conv1.bias");
                b.norm2_w = rn("norm2.weight");
                b.norm2_b = rn("norm2.bias");
                b.conv2_w = rn("conv2.weight");
                b.conv2_b = rn("conv2.bias");
            }
            c->upsampler_out_proj_w = T("codec.wave_upsampler.out_proj.weight");
            c->upsampler_out_proj_b = T("codec.wave_upsampler.out_proj.bias");
            c->upsampler_out_snake_alpha = T("codec.wave_upsampler.out_snake.alpha");
            c->upsampler_out_snake_beta = T("codec.wave_upsampler.out_snake.beta");
            if (params.verbosity >= 1)
                fprintf(stderr, "miotts: UpSampler loaded (44.1kHz v2 codec)\n");
        }
    }

    // Allocate KV cache
    {
        const int max_ctx = 4096; // sufficient for TTS (30s @ 25Hz = 750 codec tokens + prompt)
        const size_t n_mem = (size_t)max_ctx * hp.n_layers;
        const size_t kv_overhead = 2 * n_mem * ggml_tensor_overhead() + ggml_graph_overhead();
        ggml_init_params kv_ip = {kv_overhead, nullptr, true};
        c->ctx_kv = ggml_init(kv_ip);
        c->kv_k = ggml_new_tensor_4d(c->ctx_kv, GGML_TYPE_F16, hp.head_dim, max_ctx, hp.n_kv_heads, hp.n_layers);
        c->kv_v = ggml_new_tensor_4d(c->ctx_kv, GGML_TYPE_F16, hp.head_dim, max_ctx, hp.n_kv_heads, hp.n_layers);
        ggml_set_name(c->kv_k, "kv_k");
        ggml_set_name(c->kv_v, "kv_v");
        c->buf_kv = ggml_backend_alloc_ctx_tensors(c->ctx_kv, c->backend);
        if (!c->buf_kv) {
            fprintf(stderr, "miotts: failed to allocate KV cache\n");
            gguf_free(meta);
            return nullptr;
        }
        // Zero-init KV cache
        ggml_backend_tensor_set(c->kv_k, std::vector<uint8_t>(ggml_nbytes(c->kv_k), 0).data(), 0, ggml_nbytes(c->kv_k));
        ggml_backend_tensor_set(c->kv_v, std::vector<uint8_t>(ggml_nbytes(c->kv_v), 0).data(), 0, ggml_nbytes(c->kv_v));
    }

    // Compute scratch — metadata arena for graph building (no_alloc=true).
    // Size = enough tensor overhead entries for the graph.
    c->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    // Create scheduler. CPU-only on this VPS.
    c->backend_cpu = core_cpu_backend::init();
    {
        ggml_backend_t backends[2] = {c->backend, c->backend_cpu};
        int n_be = (c->backend && c->backend != c->backend_cpu) ? 2 : 1;
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
        if (!c->sched) {
            fprintf(stderr, "miotts: failed to create scheduler\n");
            gguf_free(meta);
            return nullptr;
        }
    }

    // Init global embedding to zeros (no voice cloning by default)
    c->global_embedding.resize(hp.codec_global_dim, 0.0f);

    // Try to load tokenizer from GGUF (if embedded with compatible format)
    {
        auto tokens = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            c->vocab.id_to_token = std::move(tokens);
            c->vocab.token_to_id.reserve(c->vocab.id_to_token.size());
            for (int i = 0; i < (int)c->vocab.id_to_token.size(); i++)
                c->vocab.token_to_id[c->vocab.id_to_token[i]] = i;
            auto merges = core_gguf::kv_str_array(meta, "tokenizer.ggml.merges");
            for (int i = 0; i < (int)merges.size(); i++)
                c->vocab.merge_rank[merges[i]] = i;
            c->vocab.loaded = true;
            if (params.verbosity >= 1)
                fprintf(stderr, "miotts: tokenizer (GGUF): %zu tokens, %zu merges\n", c->vocab.id_to_token.size(),
                        c->vocab.merge_rank.size());
        }
    }
    // Auto-load tokenizer.json from same directory as model if not embedded
    if (!c->vocab.loaded) {
        std::string model_dir(path_model);
        auto slash = model_dir.find_last_of("/\\");
        if (slash != std::string::npos)
            model_dir.resize(slash); // not substr(0, slash) — that builds a temporary to assign a
                                     // prefix of the string to itself (cppcheck uselessCallsSubstr)
        else
            model_dir = ".";
        std::string tok_path = model_dir + "/tokenizer.json";
        if (miotts_load_tokenizer(c.get(), tok_path.c_str()) == 0) {
            if (params.verbosity >= 1)
                fprintf(stderr, "miotts: tokenizer (auto): loaded from %s\n", tok_path.c_str());
        }
    }

    gguf_free(meta);

    if (params.verbosity >= 1) {
        fprintf(stderr, "miotts: loaded OK (%zu LLM layers, FSQ proj %s, prenet[0].wq %s, prenet_out_proj %s)\n",
                c->layers.size(), c->fsq_proj_out_w ? "yes" : "NO",
                c->wave_prenet_layers.empty() ? "EMPTY" : (c->wave_prenet_layers[0].wq ? "yes" : "NO"),
                c->wave_prenet_out_proj_w ? "yes" : "NO");
    }

    return c.release();
}

// ── LLM graph build ─────────────────────────────────────────────────

static ggml_cgraph* build_graph_llm(miotts_context* c, int n_past, int n_tokens) {
    const auto& hp = c->hp;
    const int d = (int)hp.d_model;
    const int n_q = (int)hp.n_heads;
    const int n_kv = (int)hp.n_kv_heads;
    const int hd = (int)hp.head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float theta = hp.rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: token IDs
    ggml_tensor* input_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(input_ids, "input_ids");
    ggml_set_input(input_ids);

    // Embedding lookup
    ggml_tensor* embeds = ggml_get_rows(ctx0, c->token_embd, input_ids);
    ggml_set_name(embeds, "token_embed");
    ggml_set_output(embeds); // expose for diff harness

    // Position IDs
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Causal mask (F16 for flash_attn_ext in kv_self_attn)
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        const int Lk = n_past + T;
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.max_pos,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 32.0f,
        /*rope_beta_slow*/ 1.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_NATIVE,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->layers[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // Use kv_self_attn which handles KV cache read/write + flash attention.
        // Cache layout: (hd, max_ctx, n_kv, n_layers) — matches kv_self_attn expectations.
        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, b.attn_q_norm_w, b.attn_k_norm_w,
            positions, T > 1 ? causal_mask : nullptr, c->kv_k, c->kv_v, (int)il, n_past, kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, c->output_norm);

    // For T > 1, take only the last position's hidden state
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }

    // Project to vocab logits
    ggml_tensor* logits = ggml_mul_mat(ctx0, c->output_w, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    return gf;
}

// ── Codec: wave_prenet graph build ──────────────────────────────────
// Bidirectional windowed-attention transformer (6 layers, 768d→512d).
// No KV cache (full attention computed each call). Window=125 (62 each side).
// LayerNorm + SwiGLU FFN + RoPE (theta=10000).

static ggml_cgraph* build_graph_wave_prenet(miotts_context* c, int T_in) {
    const int d_in = 768;          // input dim (FSQ embedding)
    const int d_out = 512;         // output dim (after output_proj)
    const int n_heads = 12;        // prenet heads
    const int hd = d_in / n_heads; // 64
    (void)0;                       // window_size (62 on each side) — used for mask setup
    const float eps = 1e-5f;
    const int T = T_in;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: FSQ embeddings (d_in, T)
    ggml_tensor* input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_in, T);
    ggml_set_name(input, "prenet_input");
    ggml_set_input(input);

    // Window mask + positions — needed for windowed attention
    ggml_tensor* win_mask = nullptr;
    ggml_tensor* positions = nullptr;
    const size_t n_prenet_layers = c->wave_prenet_layers.size();
    if (n_prenet_layers > 0) {
        win_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
        ggml_set_name(win_mask, "win_mask");
        ggml_set_input(win_mask);
        positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);
    }

    ggml_tensor* cur = input;

    for (size_t il = 0; il < n_prenet_layers; il++) {
        const auto& b = c->wave_prenet_layers[il];
        ggml_tensor* residual = cur;

        // Pre-attention LayerNorm (affine)
        ggml_tensor* x = ggml_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        x = ggml_add(ctx0, x, b.attn_norm_b);
        // Q/K/V projections
        ggml_tensor* Q = ggml_mul_mat(ctx0, b.wq, x); // (d_in, T)
        ggml_tensor* K = ggml_mul_mat(ctx0, b.wk, x);
        ggml_tensor* V = ggml_mul_mat(ctx0, b.wv, x);

        // Reshape to (hd, n_heads, T)
        Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_heads, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_heads, T);

        // RoPE (NORMAL = adjacent-pair complex rotation, theta=10000)
        Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 512, 10000.0f, 1.0f, 0.0f, 1.0f,
                          32.0f, 1.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 512, 10000.0f, 1.0f, 0.0f, 1.0f,
                          32.0f, 1.0f);

        // Permute to flash-attn layout (hd, T, n_heads)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        // Flash attention with window mask
        float scale = 1.0f / std::sqrt((float)hd);
        // Manual attention (avoid flash_attn_ext for debugging):
        // scores = Q @ K^T * scale → softmax → @ V
        // Q, K, V are (hd=64, T=29, n_heads=12)
        // Scores: (T, T, n_heads)
        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q); // (T, T, n_heads)
        scores = ggml_scale(ctx0, scores, scale);
        scores = ggml_soft_max(ctx0, scores); // softmax over dim 0 (K positions)
        // V needs to be (T, hd, n_heads) for: output = V^T @ scores → (hd, T, n_heads)
        // Actually: ggml_mul_mat(V_permuted, scores) where V_permuted is (hd, T, n_heads)
        // ggml_mul_mat(A, B) = B @ A^T over last 2 dims
        // A = V with shape (hd, T, n_heads), B = scores with shape (T, T, n_heads)
        // Result: (hd, T, n_heads) — each head: scores^T(T,T) @ V(T,hd) → but that's wrong dims
        // Let me use the standard approach:
        // Transpose V to (T, hd, n_heads) then mul_mat(V_T, scores) → (hd, T, n_heads)
        ggml_tensor* V_t = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3)); // (T, hd, n_heads)
        ggml_tensor* attn = ggml_mul_mat(ctx0, V_t, scores);                   // (hd, T, n_heads)

        // Reshape to (d_in, T): (64, 12, 29) → output is (hd, n_heads, T)?
        // No — ggml_mul_mat on 3D: result ne = [V_t.ne[1], scores.ne[1], n_heads]
        // = [hd, T, n_heads] = (64, 29, 12)
        // We need (hd*n_heads, T) = (768, 29). Permute (64,29,12) → (64,12,29) → reshape
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3)); // (64, 12, 29)
        attn = ggml_reshape_2d(ctx0, attn, d_in, T);                  // (768, 29)

        // Output projection
        attn = ggml_mul_mat(ctx0, b.wo, attn);

        // Residual
        cur = ggml_add(ctx0, residual, attn);

        // FFN: LayerNorm + SwiGLU
        residual = cur;
        x = ggml_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        x = ggml_add(ctx0, x, b.ffn_norm_b);

        // SwiGLU: gate = silu(x @ w1.T), up = x @ w3.T, out = (gate * up) @ w2.T
        ggml_tensor* gate = ggml_silu(ctx0, ggml_mul_mat(ctx0, b.ffn_w1, x));
        ggml_tensor* up = ggml_mul_mat(ctx0, b.ffn_w3, x);
        ggml_tensor* ffn_out = ggml_mul_mat(ctx0, b.ffn_w2, ggml_mul(ctx0, gate, up));

        cur = ggml_add(ctx0, residual, ffn_out);
    }

    // Final LayerNorm (before output_proj)
    if (c->wave_prenet_norm_w) {
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul(ctx0, cur, c->wave_prenet_norm_w);
        cur = ggml_add(ctx0, cur, c->wave_prenet_norm_b);
    }
    // Output projection (768 → 512)
    cur = ggml_mul_mat(ctx0, c->wave_prenet_out_proj_w, cur);
    cur = ggml_add(ctx0, cur, c->wave_prenet_out_proj_b);

    ggml_set_name(cur, "wave_prenet_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ── Codec: full decode graph (conv_upsample → ResNet → decoder → ResNet → iSTFT) ──

static ggml_cgraph* build_graph_codec_decode(miotts_context* c, int T_prenet) {
    const int d = 512;
    const int n_fft = (int)c->hp.codec_n_fft;    // 1920
    const int hop = (int)c->hp.codec_hop_length; // 480
    // After conv_upsample (stride=2): T_up = T_prenet * 2
    const int T_up = T_prenet * 2;
    // Target STFT length: audio_length / hop (T_codec * 960 / 480 = T_codec * 2 = T_up already)
    const int T_stft = T_up;                          // for 25Hz codec, T_prenet=29 → T_up=58=T_stft
    const int n_bins = n_fft / 2 + 1;                 // 961
    const int cond_dim = (int)c->hp.codec_global_dim; // 128

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // Input: wave_prenet output (d=512, T_prenet)
    ggml_tensor* input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_prenet);
    ggml_set_name(input, "codec_input");
    ggml_set_input(input);

    // Global embedding input (128-d speaker conditioning)
    ggml_tensor* global_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, cond_dim);
    ggml_set_name(global_emb, "global_emb");
    ggml_set_input(global_emb);

    // ── conv_upsample (ConvTranspose1d, 512→512, k=2, s=2) ─────────
    // ggml doesn't have conv_transpose1d directly, so we implement it
    // as a matrix operation: for stride=2, kernel=2, each input position
    // produces 2 output positions.
    // conv_transpose1d(x, w, b, stride=2) where x is (1, 512, T_prenet)
    // w shape: (in_ch=512, out_ch=512, k=2) → PyTorch convention
    // For stride=2 k=2: output[t] = x[t//2] @ w[:, :, t%2] + bias
    // We'll compute this directly as two matmuls interleaved.
    // ── conv_upsample: ConvTranspose1d(512→512, k=2, s=2) ─────────
    // Input (d=512, T_prenet) → transpose to (T_prenet, d) for ggml_conv_transpose_1d
    ggml_tensor* cur = ggml_cont(ctx0, ggml_transpose(ctx0, input)); // (T_prenet, d)
    cur = ggml_conv_transpose_1d(ctx0, c->conv_upsample_w, cur, /*s=*/2, /*p=*/0, /*d=*/1);
    // Output shape: (T_up, d, 1) — add bias, reshape, transpose back
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, c->conv_upsample_b, 1, d, 1));
    cur = ggml_reshape_2d(ctx0, cur, T_up, d);        // (T_up, d)
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (d, T_up)
    ggml_set_name(cur, "conv_upsample_out");
    ggml_set_output(cur);

    // NOTE: for T_up == T_stft (which it is for 25Hz codec), interpolation is identity.

    // ── Proper GroupNorm via reshape ──────────────────────────────────
    // PyTorch GroupNorm(32, 512) on (1, C=512, T): normalises each group of
    // 16 channels independently over T. ggml_norm normalises over ne[0].
    // Reshape (C=512, T) → (ch_per_group=16, n_groups=32, T), apply ggml_norm
    // over ne[0]=16, then reshape back. The affine weight/bias (512,) is
    // reshaped to (16, 32, 1) for broadcasting.
    const int n_groups = 32;
    const int cpg = d / n_groups; // channels_per_group = 16

    // GroupNorm: normalise each group of 16 channels over ALL time steps.
    // Reshape (C=512, T) → (cpg*T, n_groups) = (16*T, 32), ggml_norm over ne[0],
    // then reshape back. This matches PyTorch GroupNorm(32, 512) on (1, 512, T).
    auto group_norm_op = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int ng = 32) -> ggml_tensor* {
        const int64_t C = x->ne[0];
        const int64_t T_cur = x->ne[1];
        const int ch_pg = (int)(C / ng);
        x = ggml_reshape_3d(ctx0, x, ch_pg, ng, T_cur);
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
        x = ggml_reshape_2d(ctx0, x, (int64_t)ch_pg * T_cur, ng);
        x = ggml_norm(ctx0, x, 1e-6f);
        x = ggml_reshape_3d(ctx0, x, ch_pg, T_cur, ng);
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
        x = ggml_mul(ctx0, x, ggml_reshape_3d(ctx0, w, ch_pg, ng, 1));
        x = ggml_add(ctx0, x, ggml_reshape_3d(ctx0, b, ch_pg, ng, 1));
        return ggml_reshape_2d(ctx0, x, C, T_cur);
    };

    // ── ResNet block ────────────────────────────────────────────────
    auto resnet_block = [&](ggml_tensor* x, const miotts_resnet_block& blk, int ng = 32) -> ggml_tensor* {
        const int64_t ch = x->ne[0]; // channel dim
        ggml_tensor* res = x;
        x = group_norm_op(x, blk.norm1_w, blk.norm1_b, ng);
        x = ggml_silu(ctx0, x);
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        x = ggml_conv_1d(ctx0, blk.conv1_w, x, 1, 1, 1);
        x = ggml_add(ctx0, x, ggml_reshape_3d(ctx0, blk.conv1_b, 1, ch, 1));
        x = ggml_reshape_2d(ctx0, x, (int64_t)x->ne[0], ch);
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        x = group_norm_op(x, blk.norm2_w, blk.norm2_b, ng);
        x = ggml_silu(ctx0, x);
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        x = ggml_conv_1d(ctx0, blk.conv2_w, x, 1, 1, 1);
        x = ggml_add(ctx0, x, ggml_reshape_3d(ctx0, blk.conv2_b, 1, ch, 1));
        x = ggml_reshape_2d(ctx0, x, (int64_t)x->ne[0], ch);
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        return ggml_add(ctx0, x, res);
    };

    for (size_t i = 0; i < c->wave_prior_net.size(); i++)
        cur = resnet_block(cur, c->wave_prior_net[i]);

    ggml_set_name(cur, "wave_prior_net_out");
    ggml_set_output(cur);

    // ── wave_decoder (8 layers, AdaLN-Zero) ─────────────────────────
    const int dec_heads = 8;
    const int dec_hd = d / dec_heads; // 64
    const float dec_scale = 1.0f / std::sqrt((float)dec_hd);

    // Positions for RoPE in decoder
    ggml_tensor* dec_positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_stft);
    ggml_set_name(dec_positions, "dec_positions");
    ggml_set_input(dec_positions);

    for (size_t il = 0; il < c->wave_decoder_layers.size(); il++) {
        const auto& b = c->wave_decoder_layers[il];
        ggml_tensor* residual = cur;

        // AdaLN-Zero attention: SiLU(global_emb) → Linear → (shift, scale, gate)
        ggml_tensor* cond = ggml_silu(ctx0, global_emb);
        ggml_tensor* adaln_out = ggml_mul_mat(ctx0, b.adaln_attn_w, cond);
        adaln_out = ggml_add(ctx0, adaln_out, b.adaln_attn_b);
        // Split into shift, scale, gate (each dim=512)
        ggml_tensor* shift_a = ggml_view_1d(ctx0, adaln_out, d, 0);
        ggml_tensor* scale_a = ggml_view_1d(ctx0, adaln_out, d, (size_t)d * sizeof(float));
        ggml_tensor* gate_a = ggml_view_1d(ctx0, adaln_out, d, 2 * (size_t)d * sizeof(float));

        // LayerNorm (no affine) + modulate
        ggml_tensor* x = ggml_norm(ctx0, cur, 1e-5f);
        x = ggml_add(ctx0, ggml_add(ctx0, x, ggml_mul(ctx0, x, scale_a)), shift_a);

        // Attention (same pattern as wave_prenet but 8 heads)
        ggml_tensor* Q = ggml_mul_mat(ctx0, b.wq, x);
        ggml_tensor* K = ggml_mul_mat(ctx0, b.wk, x);
        ggml_tensor* V = ggml_mul_mat(ctx0, b.wv, x);
        Q = ggml_reshape_3d(ctx0, Q, dec_hd, dec_heads, T_stft);
        K = ggml_reshape_3d(ctx0, K, dec_hd, dec_heads, T_stft);
        V = ggml_reshape_3d(ctx0, V, dec_hd, dec_heads, T_stft);
        // RoPE
        Q = ggml_rope_ext(ctx0, Q, dec_positions, nullptr, dec_hd, GGML_ROPE_TYPE_NORMAL, 512, 10000.0f, 1.0f, 0.0f,
                          1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx0, K, dec_positions, nullptr, dec_hd, GGML_ROPE_TYPE_NORMAL, 512, 10000.0f, 1.0f, 0.0f,
                          1.0f, 32.0f, 1.0f);
        // Permute for attention
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // (hd, T, n_heads)
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
        // Manual attention (scores → softmax → V)
        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);
        scores = ggml_scale(ctx0, scores, dec_scale);
        scores = ggml_soft_max(ctx0, scores);
        ggml_tensor* V_t = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_mul_mat(ctx0, V_t, scores);
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, d, T_stft);
        attn = ggml_mul_mat(ctx0, b.wo, attn);

        // Gated residual: x = residual + gate * attn
        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, attn, gate_a));

        // AdaLN-Zero FFN
        residual = cur;
        cond = ggml_silu(ctx0, global_emb);
        ggml_tensor* adaln_ffn = ggml_mul_mat(ctx0, b.adaln_ffn_w, cond);
        adaln_ffn = ggml_add(ctx0, adaln_ffn, b.adaln_ffn_b);
        ggml_tensor* shift_f = ggml_view_1d(ctx0, adaln_ffn, d, 0);
        ggml_tensor* scale_f = ggml_view_1d(ctx0, adaln_ffn, d, (size_t)d * sizeof(float));
        ggml_tensor* gate_f = ggml_view_1d(ctx0, adaln_ffn, d, 2 * (size_t)d * sizeof(float));

        x = ggml_norm(ctx0, cur, 1e-5f);
        x = ggml_add(ctx0, ggml_add(ctx0, x, ggml_mul(ctx0, x, scale_f)), shift_f);

        // SwiGLU FFN
        ggml_tensor* gate_ffn = ggml_silu(ctx0, ggml_mul_mat(ctx0, b.ffn_w1, x));
        ggml_tensor* up_ffn = ggml_mul_mat(ctx0, b.ffn_w3, x);
        ggml_tensor* ffn_out = ggml_mul_mat(ctx0, b.ffn_w2, ggml_mul(ctx0, gate_ffn, up_ffn));

        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, ffn_out, gate_f));
    }

    // Decoder final norm: AdaLN-Zero without gate
    // SiLU(cond) → Linear → (shift, scale) — 2-way, no gate
    if (c->wave_decoder_norm_w) {
        ggml_tensor* cond = ggml_silu(ctx0, global_emb);
        ggml_tensor* adaln_final = ggml_mul_mat(ctx0, c->wave_decoder_norm_w, cond);
        adaln_final = ggml_add(ctx0, adaln_final, c->wave_decoder_norm_b);
        // Split into shift, scale (each d=512) — output dim is 1024 = 2*512
        const int d_dec = d;
        ggml_tensor* shift_fn = ggml_view_1d(ctx0, adaln_final, d_dec, 0);
        ggml_tensor* scale_fn = ggml_view_1d(ctx0, adaln_final, d_dec, (size_t)d_dec * sizeof(float));
        ggml_tensor* x = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_add(ctx0, ggml_add(ctx0, x, ggml_mul(ctx0, x, scale_fn)), shift_fn);
    }

    ggml_set_name(cur, "wave_decoder_out");
    ggml_set_output(cur);

    // ── wave_post_net (2 ResNet blocks) ─────────────────────────────
    for (size_t i = 0; i < c->wave_post_net.size(); i++)
        cur = resnet_block(cur, c->wave_post_net[i]);

    ggml_set_name(cur, "wave_post_net_out");
    ggml_set_output(cur);

    // ── UpSampler (v2 44.1kHz codec: 9× temporal upsampling) ─────────
    // Two stages of ConvTranspose1d(s=3) + SnakeBeta + ResNet → total 3×3=9× upsample.
    // Then out_proj(128→512) + out_snake, making output (512, T*9).
    if (c->has_upsampler) {
        // SnakeBeta: x + (1/exp(β)) * sin²(exp(α) * x) — per-channel, α/β in log scale
        // cur is (d, T) channel-first. ConvTranspose needs (T, d) input.
        for (size_t si = 0; si < c->upsampler_stages.size(); si++) {
            const auto& stage = c->upsampler_stages[si];
            // ConvTranspose1d(stride=3, k=9, pad=(9-3)/2=3)
            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (T, d)
            // ConvTranspose1d(s=3, k=9, p=3): ggml needs p=0, then trim 3 from each end
            cur = ggml_conv_transpose_1d(ctx0, stage.conv_w, cur, /*s=*/3, /*p=*/0, /*d=*/1);
            cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, stage.conv_b, 1, (int64_t)stage.conv_b->ne[0], 1));
            // Output is 3D (T_raw, d_up, 1). Reshape to 2D, trim padding, transpose.
            const int64_t d_up = (int64_t)stage.conv_b->ne[0];
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], d_up); // (T_raw, d_up)
            const int64_t T_raw = cur->ne[0];
            const int64_t T_up = T_raw - 6; // trim 3 from each end
            // View: skip first 3 elements along T dim
            cur = ggml_view_2d(ctx0, cur, T_up, d_up, cur->nb[1], 3 * cur->nb[0]);
            cur = ggml_cont(ctx0, cur);
            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (d_up, T_up)
            {
                char nm[32];
                snprintf(nm, sizeof(nm), "after_convt_%zu", si);
                ggml_set_name(cur, nm);
                ggml_set_output(cur);
            }

            // SnakeBeta: x + (1/exp(β)) * sin²(exp(α) * x)
            // α,β are (d_up,) in log scale; broadcast over T
            {
                ggml_tensor* alpha = ggml_exp(ctx0, stage.snake_alpha);
                ggml_tensor* neg_beta = ggml_scale(ctx0, stage.snake_beta, -1.0f);
                ggml_tensor* beta_inv = ggml_exp(ctx0, neg_beta);
                ggml_tensor* ax = ggml_mul(ctx0, cur, alpha); // (d_up, T_up)
                ggml_tensor* sinax = ggml_sin(ctx0, ax);
                ggml_tensor* sin2 = ggml_mul(ctx0, sinax, sinax); // sin²(α*x)
                cur = ggml_add(ctx0, cur, ggml_mul(ctx0, sin2, beta_inv));
            }

            {
                char nm[32];
                snprintf(nm, sizeof(nm), "after_snake_%zu", si);
                ggml_set_name(cur, nm);
                ggml_set_output(cur);
            }

            // ResNet block at this channel width
            if (si < c->upsampler_resnets.size()) {
                const auto& blk = c->upsampler_resnets[si];
                ggml_tensor* res = cur;
                // GroupNorm + SiLU + Conv1d + GroupNorm + SiLU + Conv1d + residual
                const int ng = std::min(32, (int)d_up); // adjust groups for smaller channel count
                cur = group_norm_op(cur, blk.norm1_w, blk.norm1_b);
                cur = ggml_silu(ctx0, cur);
                cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
                cur = ggml_conv_1d(ctx0, blk.conv1_w, cur, 1, 1, 1);
                cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, blk.conv1_b, 1, d_up, 1));
                cur = ggml_reshape_2d(ctx0, cur, (int64_t)cur->ne[0], d_up);
                cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
                cur = group_norm_op(cur, blk.norm2_w, blk.norm2_b);
                cur = ggml_silu(ctx0, cur);
                cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
                cur = ggml_conv_1d(ctx0, blk.conv2_w, cur, 1, 1, 1);
                cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, blk.conv2_b, 1, d_up, 1));
                cur = ggml_reshape_2d(ctx0, cur, (int64_t)cur->ne[0], d_up);
                cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
                cur = ggml_add(ctx0, cur, res);
            }
        }

        // out_proj: Linear(128→512) — cur is (128, T) channel-first
        cur = ggml_mul_mat(ctx0, c->upsampler_out_proj_w, cur); // → (512, T)
        cur = ggml_add(ctx0, cur, c->upsampler_out_proj_b);

        // out_snake: SnakeBeta(512)
        {
            ggml_tensor* alpha = ggml_exp(ctx0, c->upsampler_out_snake_alpha);
            ggml_tensor* neg_beta_out = ggml_scale(ctx0, c->upsampler_out_snake_beta, -1.0f);
            ggml_tensor* beta_inv = ggml_exp(ctx0, neg_beta_out);
            ggml_tensor* ax = ggml_mul(ctx0, cur, alpha);
            ggml_tensor* sinax = ggml_sin(ctx0, ax);
            ggml_tensor* sin2 = ggml_mul(ctx0, sinax, sinax);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, sin2, beta_inv));
        }
    }

    // ── iSTFT head ──────────────────────────────────────────────────
    // Linear d→(n_fft+2): 512→394 for 44.1kHz or 512→1922 for 24kHz
    cur = ggml_mul_mat(ctx0, c->istft_out_w, cur);
    cur = ggml_add(ctx0, cur, c->istft_out_b);

    ggml_set_name(cur, "istft_linear_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ── Sampling ────────────────────────────────────────────────────────

static int32_t sample_token(const float* logits, int vocab_size, float temperature, std::mt19937& rng) {
    if (temperature <= 0.0f) {
        // Greedy
        return (int32_t)(std::max_element(logits, logits + vocab_size) - logits);
    }
    // Temperature sampling
    std::vector<float> probs(vocab_size);
    float max_val = *std::max_element(logits, logits + vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp((logits[i] - max_val) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++)
        probs[i] /= sum;

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return (int32_t)dist(rng);
}

// ── FSQ dequantize ──────────────────────────────────────────────────

float* miotts_fsq_dequant(miotts_context* ctx, const int32_t* indices, int n_indices, int* out_dim) {
    if (!ctx || !indices || n_indices <= 0)
        return nullptr;

    // FSQ levels [8, 8, 8, 5, 5], basis [1, 8, 64, 512, 2560]
    const int levels[5] = {8, 8, 8, 5, 5};
    const int basis[5] = {1, 8, 64, 512, 2560};
    const int fsq_dim = 5;
    const int embed_dim = 768; // proj_out output dim

    // Dequantize: index → codes → normalized codes → projected embedding
    std::vector<float> codes(n_indices * fsq_dim);
    for (int t = 0; t < n_indices; t++) {
        int idx = indices[t];
        for (int d = 0; d < fsq_dim; d++) {
            int code = (idx / basis[d]) % levels[d];
            int half = levels[d] / 2;
            codes[t * fsq_dim + d] = (float)(code - half) / (float)half;
        }
    }

    // Project with proj_out: (T, 5) @ (768, 5)^T + bias → (T, 768)
    if (!ctx->fsq_proj_out_w) {
        // No projection available — return raw codes
        if (out_dim)
            *out_dim = fsq_dim;
        float* result = (float*)malloc(n_indices * fsq_dim * sizeof(float));
        memcpy(result, codes.data(), n_indices * fsq_dim * sizeof(float));
        return result;
    }

    // Read projection weights — handle F16 storage by reading raw bytes
    // then converting if needed.
    const size_t w_nelem = (size_t)embed_dim * fsq_dim;
    std::vector<float> proj_w(w_nelem);
    std::vector<float> proj_b(embed_dim);

    // proj_out.weight
    {
        const size_t nbytes = ggml_nbytes(ctx->fsq_proj_out_w);
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(ctx->fsq_proj_out_w, raw.data(), 0, nbytes);
        if (ctx->fsq_proj_out_w->type == GGML_TYPE_F16) {
            const ggml_fp16_t* src = (const ggml_fp16_t*)raw.data();
            for (size_t i = 0; i < w_nelem; i++)
                proj_w[i] = ggml_fp16_to_fp32(src[i]);
        } else {
            memcpy(proj_w.data(), raw.data(), w_nelem * sizeof(float));
        }
    }
    // proj_out.bias (always F32 — norms/biases are kept F32 by convention)
    ggml_backend_tensor_get(ctx->fsq_proj_out_b, proj_b.data(), 0, embed_dim * sizeof(float));

    float* result = (float*)malloc(n_indices * embed_dim * sizeof(float));
    // Matrix multiply: result[t, d] = sum_k(codes[t, k] * proj_w[d, k]) + proj_b[d]
    // proj_w is (768, 5) row-major = weight[out_dim, in_dim]
    for (int t = 0; t < n_indices; t++) {
        for (int d = 0; d < embed_dim; d++) {
            float val = proj_b[d];
            for (int k = 0; k < fsq_dim; k++) {
                val += codes[t * fsq_dim + k] * proj_w[d * fsq_dim + k];
            }
            result[t * embed_dim + d] = val;
        }
    }

    if (out_dim)
        *out_dim = embed_dim;
    return result;
}

// ── Forward logits (diff harness) ───────────────────────────────────

float* miotts_forward_logits(miotts_context* ctx, const int32_t* token_ids, int n_tokens, int* out_vocab) {
    if (!ctx || !token_ids || n_tokens <= 0)
        return nullptr;

    // Zero KV cache before each forward to avoid stale data from prior calls.
    ggml_backend_tensor_set(ctx->kv_k, std::vector<uint8_t>(ggml_nbytes(ctx->kv_k), 0).data(), 0,
                            ggml_nbytes(ctx->kv_k));
    ggml_backend_tensor_set(ctx->kv_v, std::vector<uint8_t>(ggml_nbytes(ctx->kv_v), 0).data(), 0,
                            ggml_nbytes(ctx->kv_v));

    ggml_cgraph* gf = build_graph_llm(ctx, /*n_past=*/0, n_tokens);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "miotts: sched alloc failed\n");
        return nullptr;
    }

    // Set inputs
    ggml_tensor* input_ids_t = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(input_ids_t, token_ids, 0, n_tokens * sizeof(int32_t));

    ggml_tensor* positions_t = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos[i] = i;
    ggml_backend_tensor_set(positions_t, pos.data(), 0, n_tokens * sizeof(int32_t));

    // Causal mask (F16 for flash_attn_ext)
    if (n_tokens > 1) {
        ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "causal_mask");
        const int Lk = n_tokens;
        std::vector<ggml_fp16_t> mask(Lk * n_tokens);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = 0; k < Lk; k++) {
                mask[q * Lk + k] = (k <= q) ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
            }
        }
        ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    // Read token_embed for diff harness debugging
    ggml_tensor* embed_t = ggml_graph_get_tensor(gf, "token_embed");
    if (embed_t) {
        const size_t ne = ggml_nelements(embed_t);
        std::vector<float> emb(ne);
        ggml_backend_tensor_get(embed_t, emb.data(), 0, ne * sizeof(float));
        fprintf(stderr, "miotts: C++ token_embed[0..7] = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f (ne=%zu type=%d)\n",
                emb[0], emb[1], emb[2], emb[3], emb[4], emb[5], emb[6], emb[7], ne, (int)embed_t->type);
    }

    // Read logits
    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    const int vocab = (int)logits_t->ne[0];
    float* result = (float*)malloc(vocab * sizeof(float));
    ggml_backend_tensor_get(logits_t, result, 0, vocab * sizeof(float));

    if (out_vocab)
        *out_vocab = vocab;

    return result;
}

// ── Synthesize ──────────────────────────────────────────────────────

int miotts_load_tokenizer(miotts_context* ctx, const char* tokenizer_json_path) {
    if (!ctx || !tokenizer_json_path)
        return -1;

    FILE* f = fopen(tokenizer_json_path, "rb");
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string json_str(sz, '\0');
    if ((long)fread(&json_str[0], 1, sz, f) != sz) {
        fclose(f);
        return -1;
    }
    fclose(f);

    // Minimal JSON parsing for tokenizer.json: extract model.vocab and model.merges.
    // This is NOT a full JSON parser — it handles the specific structure of HF tokenizer.json.
    auto find_key = [&](const std::string& key) -> size_t {
        std::string search = "\"" + key + "\"";
        return json_str.find(search);
    };

    // Parse vocab: {"model": {"vocab": {"token": id, ...}}}
    size_t vocab_pos = find_key("vocab");
    if (vocab_pos == std::string::npos)
        return -1;

    // Find the opening { after "vocab":
    size_t brace = json_str.find('{', vocab_pos + 7);
    if (brace == std::string::npos)
        return -1;

    // Parse key-value pairs until matching }
    std::unordered_map<std::string, int32_t> token_map;
    int max_id = 0;
    size_t pos = brace + 1;
    while (pos < json_str.size()) {
        // Skip whitespace
        while (pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\n' || json_str[pos] == '\r' ||
                                         json_str[pos] == '\t' || json_str[pos] == ','))
            pos++;
        if (pos >= json_str.size() || json_str[pos] == '}')
            break;
        // Parse key string
        if (json_str[pos] != '"')
            break;
        size_t key_start = pos + 1;
        // Escape-aware string end finder
        size_t key_end = std::string::npos;
        for (size_t p = key_start; p < json_str.size(); p++) {
            if (json_str[p] == '\\' && p + 1 < json_str.size()) {
                p++;
                continue;
            }
            if (json_str[p] == '"') {
                key_end = p;
                break;
            }
        }
        if (key_end == std::string::npos)
            break;
        std::string key = json_str.substr(key_start, key_end - key_start);
        // Handle escape sequences in key
        std::string unescaped;
        for (size_t k = 0; k < key.size(); k++) {
            if (key[k] == '\\' && k + 1 < key.size()) {
                k++;
                switch (key[k]) {
                case 'n':
                    unescaped += '\n';
                    break;
                case 'r':
                    unescaped += '\r';
                    break;
                case 't':
                    unescaped += '\t';
                    break;
                case '\\':
                    unescaped += '\\';
                    break;
                case '"':
                    unescaped += '"';
                    break;
                case 'u': {
                    if (k + 4 < key.size()) {
                        char hex[5] = {key[k + 1], key[k + 2], key[k + 3], key[k + 4], 0};
                        uint32_t cp = (uint32_t)strtoul(hex, nullptr, 16);
                        core_bpe::utf8_encode(cp, unescaped);
                        k += 4;
                    }
                    break;
                }
                default:
                    unescaped += '\\';
                    unescaped += key[k];
                    break;
                }
            } else {
                unescaped += key[k];
            }
        }
        // Skip ": "
        pos = key_end + 1;
        while (pos < json_str.size() && (json_str[pos] == ':' || json_str[pos] == ' '))
            pos++;
        // Parse integer value
        int32_t id = 0;
        bool neg = false;
        if (pos < json_str.size() && json_str[pos] == '-') {
            neg = true;
            pos++;
        }
        while (pos < json_str.size() && json_str[pos] >= '0' && json_str[pos] <= '9') {
            id = id * 10 + (json_str[pos] - '0');
            pos++;
        }
        if (neg)
            id = -id;
        token_map[unescaped] = id;
        if (id > max_id)
            max_id = id;
    }

    if (token_map.empty())
        return -1;

    // Build id_to_token
    ctx->vocab.id_to_token.resize(max_id + 1);
    ctx->vocab.token_to_id.clear();
    ctx->vocab.token_to_id.reserve(token_map.size());
    for (auto& [tok, id] : token_map) {
        if (id >= 0 && id <= max_id) {
            ctx->vocab.id_to_token[id] = tok;
            ctx->vocab.token_to_id[tok] = id;
        }
    }

    // Parse merges: {"model": {"merges": ["a b", "c d", ...]}}
    size_t merges_pos = find_key("merges");
    if (merges_pos != std::string::npos) {
        size_t arr_start = json_str.find('[', merges_pos);
        if (arr_start != std::string::npos) {
            pos = arr_start + 1;
            // Merges are stored as arrays: [["left","right"], ...]
            // core_bpe expects merge_rank keyed by "left right" (space-joined)
            int merge_idx = 0;
            while (pos < json_str.size()) {
                // Skip whitespace/comma
                while (pos < json_str.size() &&
                       (json_str[pos] == ' ' || json_str[pos] == '\n' || json_str[pos] == '\r' ||
                        json_str[pos] == ',' || json_str[pos] == '\t'))
                    pos++;
                if (pos >= json_str.size() || json_str[pos] == ']')
                    break;
                if (json_str[pos] == '[') {
                    // Parse ["left", "right"] — handle escaped quotes
                    auto find_str_end = [&](size_t start) -> size_t {
                        for (size_t p = start; p < json_str.size(); p++) {
                            if (json_str[p] == '\\' && p + 1 < json_str.size()) {
                                p++; // skip escaped char
                                continue;
                            }
                            if (json_str[p] == '"')
                                return p;
                        }
                        return std::string::npos;
                    };
                    auto unescape = [](const std::string& s) -> std::string {
                        std::string r;
                        for (size_t i = 0; i < s.size(); i++) {
                            if (s[i] == '\\' && i + 1 < s.size()) {
                                i++;
                                if (s[i] == '"')
                                    r += '"';
                                else if (s[i] == '\\')
                                    r += '\\';
                                else if (s[i] == 'n')
                                    r += '\n';
                                else {
                                    r += '\\';
                                    r += s[i];
                                }
                            } else {
                                r += s[i];
                            }
                        }
                        return r;
                    };
                    pos++; // skip [
                    while (pos < json_str.size() && json_str[pos] != '"')
                        pos++;
                    if (pos >= json_str.size())
                        break;
                    size_t s1 = pos + 1;
                    size_t e1 = find_str_end(s1);
                    if (e1 == std::string::npos)
                        break;
                    std::string left = unescape(json_str.substr(s1, e1 - s1));
                    pos = e1 + 1;
                    while (pos < json_str.size() && json_str[pos] != '"')
                        pos++;
                    if (pos >= json_str.size())
                        break;
                    size_t s2 = pos + 1;
                    size_t e2 = find_str_end(s2);
                    if (e2 == std::string::npos)
                        break;
                    std::string right = unescape(json_str.substr(s2, e2 - s2));
                    pos = e2 + 1;
                    // Skip to ]
                    while (pos < json_str.size() && json_str[pos] != ']')
                        pos++;
                    if (pos < json_str.size())
                        pos++; // skip ]
                    ctx->vocab.merge_rank[left + " " + right] = merge_idx++;
                } else {
                    pos++; // skip unexpected char
                }
            }
        }
    }

    // Also add added_tokens (speech tokens etc.)
    size_t added_pos = find_key("added_tokens");
    if (added_pos != std::string::npos) {
        // Parse array of objects with "id" and "content" fields
        size_t arr = json_str.find('[', added_pos);
        if (arr != std::string::npos) {
            pos = arr;
            while (pos < json_str.size()) {
                size_t obj = json_str.find('{', pos);
                if (obj == std::string::npos)
                    break;
                size_t obj_end = json_str.find('}', obj);
                if (obj_end == std::string::npos)
                    break;
                std::string chunk = json_str.substr(obj, obj_end - obj + 1);
                // Extract "id" and "content"
                size_t id_pos = chunk.find("\"id\"");
                size_t ct_pos = chunk.find("\"content\"");
                if (id_pos != std::string::npos && ct_pos != std::string::npos) {
                    // Parse id
                    size_t id_val_start = chunk.find(':', id_pos + 4);
                    int32_t at_id = 0;
                    if (id_val_start != std::string::npos) {
                        id_val_start++;
                        while (id_val_start < chunk.size() && chunk[id_val_start] == ' ')
                            id_val_start++;
                        while (id_val_start < chunk.size() && chunk[id_val_start] >= '0' &&
                               chunk[id_val_start] <= '9') {
                            at_id = at_id * 10 + (chunk[id_val_start] - '0');
                            id_val_start++;
                        }
                    }
                    // Parse content
                    size_t ct_val = chunk.find('"', ct_pos + 9);
                    if (ct_val != std::string::npos) {
                        ct_val++;
                        size_t ct_end = chunk.find('"', ct_val);
                        if (ct_end != std::string::npos) {
                            std::string content = chunk.substr(ct_val, ct_end - ct_val);
                            if (at_id >= (int32_t)ctx->vocab.id_to_token.size())
                                ctx->vocab.id_to_token.resize(at_id + 1);
                            ctx->vocab.id_to_token[at_id] = content;
                            ctx->vocab.token_to_id[content] = at_id;
                        }
                    }
                }
                pos = obj_end + 1;
            }
        }
    }

    ctx->vocab.loaded = true;

    // Verify special tokens loaded
    auto check = [&](const char* name, int32_t expected) {
        auto it = ctx->vocab.token_to_id.find(name);
        if (it == ctx->vocab.token_to_id.end())
            fprintf(stderr, "miotts: WARNING: special token '%s' not found in vocab\n", name);
        else if (it->second != expected)
            fprintf(stderr, "miotts: WARNING: '%s' has id %d, expected %d\n", name, it->second, expected);
    };
    check("<|im_start|>", 151644);
    check("<|im_end|>", 151645);
    check("<|endoftext|>", 151643);

    fprintf(stderr, "miotts: tokenizer: %zu tokens, %zu merges, %zu token_to_id entries\n",
            ctx->vocab.id_to_token.size(), ctx->vocab.merge_rank.size(), ctx->vocab.token_to_id.size());

    return 0;
}

int miotts_set_reference(miotts_context* ctx, const float* audio_24k, int n_samples) {
    if (!ctx)
        return -1;
    if (!audio_24k || n_samples <= 0) {
        std::fill(ctx->global_embedding.begin(), ctx->global_embedding.end(), 0.0f);
        return 0;
    }
    fprintf(stderr, "miotts: warning: reference audio encoding not yet implemented, using default voice\n");
    return 0;
}

int miotts_load_preset_embedding(miotts_context* ctx, const char* emb_path) {
    if (!ctx || !emb_path)
        return -1;
    // Try raw binary first (128 floats = 512 bytes)
    FILE* f = fopen(emb_path, "rb");
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz == (long)(ctx->hp.codec_global_dim * sizeof(float))) {
        if (fread(ctx->global_embedding.data(), sizeof(float), ctx->hp.codec_global_dim, f) ==
            ctx->hp.codec_global_dim) {
            fclose(f);
            return 0;
        }
    }
    fclose(f);
    // Try GGUF format (.emb.gguf with tensor "mio.global_embedding")
    gguf_init_params gip = {/*.no_alloc=*/false, /*.ctx=*/nullptr};
    ggml_context* emb_ctx = nullptr;
    gguf_init_params gip2 = {false, &emb_ctx};
    gguf_context* meta = gguf_init_from_file(emb_path, gip2);
    if (meta && emb_ctx) {
        ggml_tensor* t = ggml_get_tensor(emb_ctx, "mio.global_embedding");
        if (t && ggml_nelements(t) == (int64_t)ctx->hp.codec_global_dim) {
            memcpy(ctx->global_embedding.data(), t->data, ctx->hp.codec_global_dim * sizeof(float));
            gguf_free(meta);
            ggml_free(emb_ctx);
            return 0;
        }
        gguf_free(meta);
        ggml_free(emb_ctx);
    }
    return -1;
}

// ── Wave prenet forward (diff harness) ──────────────────────────────

float* miotts_wave_prenet_forward(miotts_context* ctx, const float* fsq_emb, int T, int* out_dim) {
    if (!ctx || !fsq_emb || T <= 0)
        return nullptr;

    const int d_in = 768;
    const int d_out = 512;

    ggml_cgraph* gf = build_graph_wave_prenet(ctx, T);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "miotts: wave_prenet sched alloc failed\n");
        return nullptr;
    }

    // Set input: FSQ embeddings (d_in=768, T) — ggml layout is column-major
    ggml_tensor* input_t = ggml_graph_get_tensor(gf, "prenet_input");
    ggml_backend_tensor_set(input_t, fsq_emb, 0, (size_t)T * d_in * sizeof(float));

    // Verify input was set correctly
    {
        float check[4];
        ggml_backend_tensor_get(input_t, check, 0, 4 * sizeof(float));
        fprintf(stderr, "miotts: prenet input verify[0:4] = %.6f %.6f %.6f %.6f\n", check[0], check[1], check[2],
                check[3]);
    }

    // Set positions [0, 1, 2, ..., T-1]
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "positions");
    if (pos_t) {
        std::vector<int32_t> pos(T);
        for (int i = 0; i < T; i++)
            pos[i] = i;
        ggml_backend_tensor_set(pos_t, pos.data(), 0, T * sizeof(int32_t));
    }

    // Build window mask: bidirectional, window=125 (62 on each side)
    ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "win_mask");
    if (mask_t) {
        const int w = 62; // window_size // 2
        std::vector<ggml_fp16_t> mask(T * T);
        for (int q = 0; q < T; q++) {
            for (int k = 0; k < T; k++) {
                bool in_window = (k >= q - w) && (k <= q + w);
                mask[q * T + k] = in_window ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
            }
        }
        ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    // Read output
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "wave_prenet_out");
    const size_t n_out = (size_t)T * d_out;
    float* result = (float*)malloc(n_out * sizeof(float));
    ggml_backend_tensor_get(out_t, result, 0, n_out * sizeof(float));

    fprintf(stderr, "miotts: C++ prenet_out[0..7] = %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", result[0], result[1],
            result[2], result[3], result[4], result[5], result[6], result[7]);

    if (out_dim)
        *out_dim = d_out;
    return result;
}

// ── Codec decode (wave_prenet output → audio) ──────────────────────

float* miotts_codec_decode(miotts_context* ctx, const float* prenet_out, int T_prenet, int* out_n) {
    if (!ctx || !prenet_out || T_prenet <= 0)
        return nullptr;

    const int d = 512;
    const int n_fft = (int)ctx->hp.codec_n_fft;
    const int hop = (int)ctx->hp.codec_hop_length;
    const int n_bins = n_fft / 2 + 1; // 961

    ggml_cgraph* gf = build_graph_codec_decode(ctx, T_prenet);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "miotts: codec decode sched alloc failed\n");
        return nullptr;
    }

    // Set inputs
    ggml_tensor* input_t = ggml_graph_get_tensor(gf, "codec_input");
    ggml_backend_tensor_set(input_t, prenet_out, 0, (size_t)T_prenet * d * sizeof(float));

    ggml_tensor* emb_t = ggml_graph_get_tensor(gf, "global_emb");
    ggml_backend_tensor_set(emb_t, ctx->global_embedding.data(), 0, ctx->global_embedding.size() * sizeof(float));

    // Positions for decoder RoPE
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "dec_positions");
    if (pos_t) {
        const int T_stft = T_prenet * 2;
        std::vector<int32_t> pos(T_stft);
        for (int i = 0; i < T_stft; i++)
            pos[i] = i;
        ggml_backend_tensor_set(pos_t, pos.data(), 0, T_stft * sizeof(int32_t));
    }

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    // Debug: dump intermediate shapes and first values
    for (const char* sn : {"conv_upsample_out", "wave_prior_net_out", "wave_decoder_out", "wave_post_net_out",
                           "after_convt_0", "after_snake_0", "after_convt_1", "after_snake_1", "istft_linear_out"}) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, sn);
        if (t) {
            float buf[4];
            ggml_backend_tensor_get(t, buf, 0, 4 * sizeof(float));
            fprintf(stderr, "miotts: %s[0:4] = %.4f %.4f %.4f %.4f (ne0=%lld ne1=%lld)\n", sn, buf[0], buf[1], buf[2],
                    buf[3], (long long)t->ne[0], (long long)t->ne[1]);
        }
    }

    // Read iSTFT linear output: (1922, T_stft)
    ggml_tensor* stft_t = ggml_graph_get_tensor(gf, "istft_linear_out");
    const int T_stft = (int)stft_t->ne[1];
    const int out_dim = (int)stft_t->ne[0]; // 1922
    std::vector<float> stft_data((size_t)T_stft * out_dim);
    ggml_backend_tensor_get(stft_t, stft_data.data(), 0, stft_data.size() * sizeof(float));

    // Split into magnitude (log) and phase
    std::vector<float> mag(T_stft * n_bins);
    std::vector<float> phase(T_stft * n_bins);
    for (int t = 0; t < T_stft; t++) {
        for (int b = 0; b < n_bins; b++) {
            float log_mag = stft_data[t * out_dim + b];
            float ph = stft_data[t * out_dim + n_bins + b];
            float m = std::exp(log_mag);
            if (m > 100.0f)
                m = 100.0f;
            mag[t * n_bins + b] = m;
            phase[t * n_bins + b] = ph;
        }
    }

    // CPU iSTFT using core_istft (Hann window, "same" padding trim)
    auto pcm = core_istft::istft(mag.data(), phase.data(), n_fft, hop, T_stft,
                                 /*window=*/nullptr, core_istft::TRIM_SAME);

    if (out_n)
        *out_n = (int)pcm.size();

    float* result = (float*)malloc(pcm.size() * sizeof(float));
    memcpy(result, pcm.data(), pcm.size() * sizeof(float));
    return result;
}

// ── Qwen3 BPE tokenizer (reused from moss_tts.cpp pattern) ─────────

static std::vector<std::string> miotts_qwen_pretokenize(const std::string& s) {
    std::vector<std::string> out;
    const size_t n = s.size();
    auto is_letter = [](unsigned char c) { return std::isalpha(c) != 0 || c >= 0x80; };
    auto is_digit = [](unsigned char c) { return std::isdigit(c) != 0; };
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    auto is_nl = [](unsigned char c) { return c == '\r' || c == '\n'; };
    size_t i = 0;
    while (i < n) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '\'' && i + 1 < n) {
            static const char* cons[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
            bool matched = false;
            for (const char* cc : cons) {
                const size_t len = std::strlen(cc);
                if (i + len > n)
                    continue;
                bool eq = true;
                for (size_t k = 0; k < len; k++)
                    if (std::tolower((unsigned char)s[i + k]) != std::tolower((unsigned char)cc[k])) {
                        eq = false;
                        break;
                    }
                if (eq) {
                    out.push_back(s.substr(i, len));
                    i += len;
                    matched = true;
                    break;
                }
            }
            if (matched)
                continue;
        }
        {
            size_t j = i;
            if (j < n && !is_nl((unsigned char)s[j]) && !is_letter((unsigned char)s[j]) &&
                !is_digit((unsigned char)s[j]))
                j++;
            if (j < n && is_letter((unsigned char)s[j])) {
                while (j < n && is_letter((unsigned char)s[j]))
                    j++;
                out.push_back(s.substr(i, j - i));
                i = j;
                continue;
            }
        }
        if (is_digit(c)) {
            out.push_back(s.substr(i, 1));
            i++;
            continue;
        }
        if (is_nl(c)) {
            size_t j = i;
            while (j < n && is_nl((unsigned char)s[j]))
                j++;
            out.push_back(s.substr(i, j - i));
            i = j;
            continue;
        }
        if (is_space(c)) {
            size_t j = i;
            while (j < n && is_space((unsigned char)s[j]))
                j++;
            out.push_back(s.substr(i, j - i));
            i = j;
            continue;
        }
        out.push_back(s.substr(i, 1));
        i++;
    }
    return out;
}

static std::vector<int32_t> miotts_tokenize(miotts_context* ctx, const std::string& text) {
    const auto& v = ctx->vocab;
    std::vector<int32_t> result;

    auto match_special = [&](const std::string& str, size_t pos, int32_t& id) -> size_t {
        if (str[pos] != '<')
            return 0;
        const bool pipe_form = pos + 1 < str.size() && str[pos + 1] == '|';
        const size_t close = pipe_form ? str.find("|>", pos + 2) : str.find('>', pos + 1);
        if (close == std::string::npos)
            return 0;
        const size_t len = close + (pipe_form ? 2 : 1) - pos;
        if (!pipe_form && len > 24)
            return 0;
        auto it = v.token_to_id.find(str.substr(pos, len));
        if (it == v.token_to_id.end())
            return 0;
        id = it->second;
        return len;
    };

    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '<') {
            int32_t sp_id = 0;
            const size_t sp_len = match_special(text, i, sp_id);
            if (sp_len > 0) {
                result.push_back(sp_id);
                i += sp_len;
                continue;
            }
        }
        size_t j = i;
        if (text[j] == '<')
            j++;
        while (j < text.size()) {
            if (text[j] == '<') {
                int32_t sp_id = 0;
                if (match_special(text, j, sp_id) > 0)
                    break;
            }
            j++;
        }
        std::string chunk = text.substr(i, j - i);
        i = j;
        if (chunk.empty())
            continue;
        for (const std::string& pre : miotts_qwen_pretokenize(chunk)) {
            std::string encoded = core_bpe::bytes_to_unicode(pre.data(), pre.size());
            core_bpe::bpe_one(v.token_to_id, v.merge_rank, encoded, result);
        }
    }
    return result;
}

// ── Synthesize (full pipeline) ──────────────────────────────────────

float* miotts_synthesize(miotts_context* ctx, const char* text, int* out_n) {
    if (!ctx || !text || !out_n)
        return nullptr;
    *out_n = 0;

    if (!ctx->vocab.loaded) {
        fprintf(stderr, "miotts: error: no tokenizer in GGUF (re-convert with updated converter)\n");
        return nullptr;
    }

    const auto& hp = ctx->hp;

    // 1. Build ChatML prompt: <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    std::string prompt = "<|im_start|>user\n";
    prompt += text;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";

    // 2. Tokenize
    std::vector<int32_t> input_ids = miotts_tokenize(ctx, prompt);
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "miotts: prompt tokens: %zu = {", input_ids.size());
    for (size_t ii = 0; ii < input_ids.size(); ii++)
        fprintf(stderr, "%s%d", ii ? "," : "", input_ids[ii]);
    fprintf(stderr, "}\n");

    // 3. Autoregressive LLM generation
    const int max_new = ctx->params.max_tokens > 0 ? ctx->params.max_tokens : 750;
    const uint64_t seed = ctx->params.seed > 0 ? ctx->params.seed : 42;
    std::mt19937 rng(seed);

    // Prefill
    ggml_backend_tensor_set(ctx->kv_k, std::vector<uint8_t>(ggml_nbytes(ctx->kv_k), 0).data(), 0,
                            ggml_nbytes(ctx->kv_k));
    ggml_backend_tensor_set(ctx->kv_v, std::vector<uint8_t>(ggml_nbytes(ctx->kv_v), 0).data(), 0,
                            ggml_nbytes(ctx->kv_v));

    {
        ggml_cgraph* gf = build_graph_llm(ctx, 0, (int)input_ids.size());
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "miotts: prefill alloc failed\n");
            return nullptr;
        }
        ggml_tensor* ids_t = ggml_graph_get_tensor(gf, "input_ids");
        ggml_backend_tensor_set(ids_t, input_ids.data(), 0, input_ids.size() * sizeof(int32_t));
        ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "positions");
        std::vector<int32_t> pos(input_ids.size());
        for (size_t i = 0; i < input_ids.size(); i++)
            pos[i] = (int32_t)i;
        ggml_backend_tensor_set(pos_t, pos.data(), 0, pos.size() * sizeof(int32_t));
        // Causal mask (F16 for flash_attn_ext)
        ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "causal_mask");
        if (mask_t) {
            const int T = (int)input_ids.size();
            const int Lk = T; // n_past=0 so Lk=T
            std::vector<ggml_fp16_t> mask(Lk * T);
            for (int q = 0; q < T; q++)
                for (int k = 0; k < Lk; k++)
                    mask[q * Lk + k] = (k <= q) ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
            ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
        ggml_backend_sched_graph_compute(ctx->sched, gf);
        // Sample first token
        ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
        std::vector<float> logits(hp.vocab_size);
        ggml_backend_tensor_get(logits_t, logits.data(), 0, hp.vocab_size * sizeof(float));
        int32_t next_id = sample_token(logits.data(), (int)hp.vocab_size, ctx->params.temperature, rng);
        {
            int32_t argmax = (int32_t)(std::max_element(logits.data(), logits.data() + hp.vocab_size) - logits.data());
            fprintf(stderr, "miotts: prefill argmax=%d (%.4f) sampled=%d\n", argmax, logits[argmax], next_id);
        }
        input_ids.push_back(next_id);
    }

    // Decode loop — KV-cached T=1 steps
    int n_past = (int)input_ids.size() - 1;
    for (int step = 1; step < max_new; step++) {
        int32_t last_id = input_ids.back();
        if (last_id == (int32_t)hp.eos_token_id || last_id == 151643)
            break;
        ggml_cgraph* gf = build_graph_llm(ctx, n_past, 1);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
            break;
        ggml_tensor* ids_t = ggml_graph_get_tensor(gf, "input_ids");
        ggml_backend_tensor_set(ids_t, &last_id, 0, sizeof(int32_t));
        ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "positions");
        int32_t pos_val = n_past;
        ggml_backend_tensor_set(pos_t, &pos_val, 0, sizeof(int32_t));
        ggml_backend_sched_graph_compute(ctx->sched, gf);
        ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
        std::vector<float> logits(hp.vocab_size);
        ggml_backend_tensor_get(logits_t, logits.data(), 0, hp.vocab_size * sizeof(float));
        int32_t next_id = sample_token(logits.data(), (int)hp.vocab_size, ctx->params.temperature, rng);
        if (step <= 3) {
            int32_t argmax = (int32_t)(std::max_element(logits.data(), logits.data() + hp.vocab_size) - logits.data());
            fprintf(stderr, "miotts: decode step %d: argmax=%d sampled=%d\n", step, argmax, next_id);
        }
        input_ids.push_back(next_id);
        n_past++;
    }

    // 4. Extract speech token indices
    const int prompt_len = (int)(input_ids.size() - (input_ids.size() - n_past - 1));
    // Actually: first prompt_len tokens are the prompt, rest are generated
    // Let me just find speech tokens in the generated portion
    std::vector<int32_t> speech_indices;
    for (size_t i = 0; i < input_ids.size(); i++) {
        int32_t id = input_ids[i];
        if (id >= (int32_t)hp.speech_token_start && id < (int32_t)hp.speech_token_end)
            speech_indices.push_back(id - (int32_t)hp.speech_token_start);
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "miotts: generated %zu tokens, %zu speech tokens\n", input_ids.size(), speech_indices.size());

    if (speech_indices.empty()) {
        fprintf(stderr, "miotts: no speech tokens generated\n");
        return nullptr;
    }

    // 5. FSQ dequant
    int fsq_dim = 0;
    float* fsq_emb = miotts_fsq_dequant(ctx, speech_indices.data(), (int)speech_indices.size(), &fsq_dim);
    if (!fsq_emb)
        return nullptr;

    // 6. Wave prenet
    int prenet_dim = 0;
    float* prenet_out = miotts_wave_prenet_forward(ctx, fsq_emb, (int)speech_indices.size(), &prenet_dim);
    free(fsq_emb);
    if (!prenet_out)
        return nullptr;

    // 7. Codec decode (conv → ResNet → decoder → ResNet → iSTFT)
    int n_samples = 0;
    float* audio = miotts_codec_decode(ctx, prenet_out, (int)speech_indices.size(), &n_samples);
    free(prenet_out);
    if (!audio)
        return nullptr;

    *out_n = n_samples;
    return audio;
}

int miotts_get_sample_rate(const miotts_context* ctx) {
    return ctx ? static_cast<int>(ctx->hp.codec_sample_rate) : 24000;
}

void miotts_free_audio(float* pcm) {
    free(pcm);
}

void miotts_free(miotts_context* ctx) {
    delete ctx;
}
