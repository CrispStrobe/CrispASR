// mimo_tokenizer.cpp — MiMo-Audio-Tokenizer encoder runtime.
//
// PLAN #51 step 1: scaffold + tensor binding. Loads cstr/mimo-tokenizer-GGUF
// and exposes the C ABI; the forward path (mel → conv stem → 32-layer xfmr
// → down-sample → 8-stage RVQ encode) is implemented in follow-up commits.
//
// Architecture is documented in src/mimo_tokenizer.h and mirrored in the
// Python reference dumper at tools/reference_backends/mimo_tokenizer.py.

#include "mimo_tokenizer.h"

#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct mimo_tok_hp {
    uint32_t d_model = 1280;
    uint32_t encoder_layers = 32;
    uint32_t encoder_heads = 20;
    uint32_t encoder_ffn_dim = 5120;
    uint32_t num_quantizers = 20;     // total stages on disk; ASR uses first 8
    uint32_t sampling_rate = 24000;
    uint32_t hop_length = 240;
    uint32_t stride_size = 2;         // conv2 stride
    uint32_t avg_pooler = 2;          // down_sample stride
    uint32_t kernel_size = 3;         // conv stem kernel
    // Defaults below come from MiMo-Audio-Tokenizer/config.json — they are
    // not yet written into the GGUF by the converter, so we baked them in.
    uint32_t encoder_skip_layer_id = 3; // skip-connection saved AFTER layer 2
    uint32_t n_mels = 128;
    uint32_t nfft = 960;
    uint32_t window_size = 960;
    float rope_theta = 10000.0f;
    // Per-stage codebook sizes (1024,1024,128×6 for the first 8 ASR stages).
    std::vector<uint32_t> codebook_size;
};

struct mimo_tok_layer {
    // LayerNorm (pre-attn): self_attn_layer_norm
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    // Attention: q/v/o have biases, k has NO bias (upstream-defined).
    ggml_tensor* q_w = nullptr;
    ggml_tensor* q_b = nullptr;
    ggml_tensor* k_w = nullptr; // no bias
    ggml_tensor* v_w = nullptr;
    ggml_tensor* v_b = nullptr;
    ggml_tensor* o_w = nullptr;
    ggml_tensor* o_b = nullptr;
    // LayerNorm (pre-FFN): final_layer_norm
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
    // FFN: fc1 (d → ffn) + GELU + fc2 (ffn → d), both with biases.
    ggml_tensor* fc1_w = nullptr;
    ggml_tensor* fc1_b = nullptr;
    ggml_tensor* fc2_w = nullptr;
    ggml_tensor* fc2_b = nullptr;
};

struct mimo_tok_codebook {
    // Per-stage: F16 [d_model, codebook_size] in GGUF on-disk order
    // (innermost dim = d_model). Loaded as F16 then promoted to F32 at
    // distance-compute time (matches `quantizer.float()` in upstream).
    ggml_tensor* embed = nullptr;
    uint32_t codebook_size = 0;
};

} // namespace

struct mimo_tokenizer_context {
    mimo_tokenizer_context_params params{};
    int n_threads = 4;

    mimo_tok_hp hp;

    // Backends + weights
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Bound tensor handles (for the encoder forward path).
    ggml_tensor* conv1_w = nullptr;
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* conv2_w = nullptr;
    ggml_tensor* conv2_b = nullptr;
    ggml_tensor* final_norm_w = nullptr;
    ggml_tensor* final_norm_b = nullptr;
    ggml_tensor* down_w = nullptr;       // Conv1d (1280, 1280, k=2, s=2, no bias)
    ggml_tensor* down_norm_w = nullptr;
    ggml_tensor* down_norm_b = nullptr;
    std::vector<mimo_tok_layer> layers;
    std::vector<mimo_tok_codebook> codebooks; // first 8 used for ASR
};

extern "C" struct mimo_tokenizer_context_params mimo_tokenizer_context_default_params(void) {
    mimo_tokenizer_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

extern "C" struct mimo_tokenizer_context* mimo_tokenizer_init_from_file(const char* path_model,
                                                                        struct mimo_tokenizer_context_params params) {
    auto* ctx = new mimo_tokenizer_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // ---- Pass 1: metadata + hparams ----
    gguf_context* gctx = core_gguf::open_metadata(path_model);
    if (!gctx) {
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hp;
    hp.d_model = core_gguf::kv_u32(gctx, "mimo_tok.d_model", hp.d_model);
    hp.encoder_layers = core_gguf::kv_u32(gctx, "mimo_tok.encoder_layers", hp.encoder_layers);
    hp.encoder_heads = core_gguf::kv_u32(gctx, "mimo_tok.encoder_heads", hp.encoder_heads);
    hp.encoder_ffn_dim = core_gguf::kv_u32(gctx, "mimo_tok.encoder_ffn_dim", hp.encoder_ffn_dim);
    hp.num_quantizers = core_gguf::kv_u32(gctx, "mimo_tok.num_quantizers", hp.num_quantizers);
    hp.sampling_rate = core_gguf::kv_u32(gctx, "mimo_tok.sampling_rate", hp.sampling_rate);
    hp.hop_length = core_gguf::kv_u32(gctx, "mimo_tok.hop_length", hp.hop_length);
    hp.stride_size = core_gguf::kv_u32(gctx, "mimo_tok.stride_size", hp.stride_size);
    hp.avg_pooler = core_gguf::kv_u32(gctx, "mimo_tok.avg_pooler", hp.avg_pooler);
    hp.kernel_size = core_gguf::kv_u32(gctx, "mimo_tok.kernel_size", hp.kernel_size);
    // Pull codebook sizes (the converter writes them per index).
    hp.codebook_size.clear();
    for (uint32_t i = 0; i < hp.num_quantizers; i++) {
        char key[64];
        snprintf(key, sizeof(key), "mimo_tok.codebook_size.%u", i);
        hp.codebook_size.push_back(core_gguf::kv_u32(gctx, key, 0));
    }
    core_gguf::free_metadata(gctx);

    // ---- Backends ----
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "mimo_tokenizer: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    // ---- Pass 2: weights (load to CPU; encoder forward graph picks per-op) ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend_cpu, "mimo_tokenizer", wl)) {
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    // ---- Bind named tensors. Conv stem and final norm use the upstream
    // `encoder.*` prefix (the converter's rename rules don't touch them
    // because the upstream uses `conv1`/`conv2`/`down_sample_layer`, which
    // the rename list doesn't match). Per-layer tensors land at `enc.blk.*`.
    auto& T = ctx->tensors;
    auto bind = [&](const char* name) -> ggml_tensor* {
        return core_gguf::require(T, name, "mimo_tokenizer");
    };

    ctx->conv1_w = bind("encoder.conv1.weight");
    ctx->conv1_b = bind("encoder.conv1.bias");
    ctx->conv2_w = bind("encoder.conv2.weight");
    ctx->conv2_b = bind("encoder.conv2.bias");
    ctx->final_norm_w = bind("enc.norm.weight");
    ctx->final_norm_b = bind("enc.norm.bias");
    ctx->down_w = bind("encoder.down_sample_layer.0.weight");
    ctx->down_norm_w = bind("encoder.down_sample_norm.weight");
    ctx->down_norm_b = bind("encoder.down_sample_norm.bias");

    ctx->layers.resize(hp.encoder_layers);
    char buf[128];
    for (uint32_t i = 0; i < hp.encoder_layers; i++) {
        auto& L = ctx->layers[i];
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn_norm.weight", i); L.attn_norm_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn_norm.bias", i);   L.attn_norm_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.q.weight", i);    L.q_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.q.bias", i);      L.q_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.k.weight", i);    L.k_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.v.weight", i);    L.v_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.v.bias", i);      L.v_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.o.weight", i);    L.o_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.attn.o.bias", i);      L.o_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.ffn_norm.weight", i);  L.ffn_norm_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.ffn_norm.bias", i);    L.ffn_norm_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.fc1.weight", i);       L.fc1_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.fc1.bias", i);         L.fc1_b = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.fc2.weight", i);       L.fc2_w = bind(buf);
        snprintf(buf, sizeof(buf), "enc.blk.%u.fc2.bias", i);         L.fc2_b = bind(buf);
    }

    // RVQ codebooks. Only the first 8 are needed for ASR; bind any present.
    ctx->codebooks.resize(hp.num_quantizers);
    for (uint32_t i = 0; i < hp.num_quantizers; i++) {
        snprintf(buf, sizeof(buf), "encoder.quant.vq.layers.%u._codebook.embed", i);
        ggml_tensor* e = core_gguf::try_get(T, buf);
        if (e) {
            ctx->codebooks[i].embed = e;
            // GGUF prints shape [d_model, codebook_size] (innermost first);
            // ne[0] = d_model, ne[1] = codebook_size.
            ctx->codebooks[i].codebook_size = (uint32_t)e->ne[1];
        }
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "mimo_tokenizer: loaded %zu tensors  encoder=%uL/%u  rvq=%u stages\n", ctx->tensors.size(),
                hp.encoder_layers, hp.d_model, hp.num_quantizers);
    }
    return ctx;
}

extern "C" int32_t* mimo_tokenizer_encode_pcm16k(struct mimo_tokenizer_context* /*ctx*/, const float* /*pcm*/,
                                                 int /*n_samples*/, int* n_frames_out) {
    // PLAN #51 step 1: forward path not yet implemented.
    if (n_frames_out)
        *n_frames_out = 0;
    fprintf(stderr, "mimo_tokenizer_encode_pcm16k: forward path not yet implemented (PLAN #51)\n");
    return nullptr;
}

extern "C" float* mimo_tokenizer_extract_stage(struct mimo_tokenizer_context* /*ctx*/, const float* /*pcm*/,
                                               int /*n_samples*/, const char* stage, int* n_out) {
    // PLAN #51 step 1: forward path not yet implemented. Stage names are
    // exposed through this API so the diff-harness wiring in
    // examples/cli/crispasr_diff_main.cpp can be authored against the
    // final symbol set even before the implementation lands.
    if (n_out)
        *n_out = 0;
    fprintf(stderr, "mimo_tokenizer_extract_stage('%s'): forward path not yet implemented (PLAN #51)\n",
            stage ? stage : "(null)");
    return nullptr;
}

extern "C" void mimo_tokenizer_free(struct mimo_tokenizer_context* ctx) {
    if (!ctx)
        return;
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void mimo_tokenizer_set_n_threads(struct mimo_tokenizer_context* ctx, int n_threads) {
    if (!ctx || n_threads <= 0)
        return;
    ctx->n_threads = n_threads;
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}
