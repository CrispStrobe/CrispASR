// mimo_asr.cpp — runtime for XiaomiMiMo/MiMo-V2.5-ASR
//
// SCAFFOLD ONLY (April 2026): the model loads, the tensors are
// populated from the GGUF, and the transcribe entry point returns
// an explicit "not implemented" string so callers can integrate
// against the C ABI today and the encoder/LLM forward passes can
// land incrementally without breaking the build.
//
// Architecture (from the converter docstring + config.json):
//
//   audio side:
//     - 8-channel RVQ codes (12.5 fps after group_size=4 → 25 Hz
//       in the .nemo's frame definition; see `audio.frame_rate`)
//     - speech_embeddings: per-channel embedding tables (codebook
//       sized) → group_proj that fuses the 8 channels into a single
//       1024-d sequence
//     - 6-layer input_local_transformer (1024d, 64h × 16d head,
//       SiLU MLP, RoPE) processes that sequence
//     - hidden_proj projects to LM hidden size
//
//   LLM side:
//     - 36-layer Qwen2 (3584d, 32 heads, 8 KV heads, head_dim 112,
//       SiLU SwiGLU, RoPE theta 640000, RMSNorm)
//     - tied embedding for embed_tokens / lm_head
//     - codebook_head: per-channel head used during ASR-direction
//       inference is the LLM's own lm_head; the per-codebook heads
//       are present in the checkpoint but only used in TTS-direction
//       generation. We can ignore them for ASR.
//
// Known follow-ups (PLAN #51):
//   - Audio: load the mimo-tokenizer GGUF (cstr/mimo-tokenizer-GGUF)
//     and run its encoder over PCM → 8-channel RVQ code stream.
//   - Wire the codes through speech_embeddings → input_local_transformer
//     → hidden_proj → LLM.embed augmentation as the prefill prompt.
//   - LLM forward: standard Qwen2 — fully covered by
//     core_attn::kv_self_attn + core_ffn::swiglu, same call site as
//     qwen3-asr / gemma4-e2b.

#include "mimo_asr.h"

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

struct mimo_asr_hp {
    // LLM (Qwen2)
    uint32_t llm_hidden = 3584;
    uint32_t llm_layers = 36;
    uint32_t llm_heads = 32;
    uint32_t llm_kv_heads = 8;
    uint32_t llm_intermediate = 18944;
    uint32_t llm_vocab = 152064;
    uint32_t llm_max_pos = 8192;
    float llm_rope_theta = 640000.0f;
    float llm_rms_eps = 1e-6f;

    // Audio (input_local_transformer)
    uint32_t audio_channels = 8;
    uint32_t audio_group_size = 4;
    uint32_t audio_layers = 6;
    uint32_t audio_dim = 1024;
    uint32_t audio_heads = 64;
    uint32_t audio_head_dim = 16;
    uint32_t audio_intermediate = 4096;
};

} // namespace

struct mimo_asr_context {
    mimo_asr_context_params params{};
    int n_threads = 4;

    mimo_asr_hp hp;

    // Backends + weights
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Tokeniser GGUF path — set via mimo_asr_set_tokenizer_path before
    // the first transcribe call. Empty until set.
    std::string tokenizer_path;

    std::vector<std::string> vocab;
};

static uint32_t mimo_kv_u32(gguf_context* ctx, const char* key, uint32_t def) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_u32(ctx, id) : def;
}
static float mimo_kv_f32(gguf_context* ctx, const char* key, float def) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_f32(ctx, id) : def;
}

extern "C" struct mimo_asr_context_params mimo_asr_context_default_params(void) {
    mimo_asr_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.temperature = 0.0f;
    return p;
}

extern "C" struct mimo_asr_context* mimo_asr_init_from_file(const char* path_model,
                                                            struct mimo_asr_context_params params) {
    auto* ctx = new mimo_asr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Read GGUF metadata
    ggml_context* gctx_dummy = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, &gctx_dummy};
    gguf_context* gctx = gguf_init_from_file(path_model, gp);
    if (!gctx) {
        fprintf(stderr, "mimo_asr: failed to read GGUF '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hp;
    hp.llm_hidden       = mimo_kv_u32(gctx, "mimo_asr.llm.hidden_size", hp.llm_hidden);
    hp.llm_layers       = mimo_kv_u32(gctx, "mimo_asr.llm.num_layers", hp.llm_layers);
    hp.llm_heads        = mimo_kv_u32(gctx, "mimo_asr.llm.num_heads", hp.llm_heads);
    hp.llm_kv_heads     = mimo_kv_u32(gctx, "mimo_asr.llm.num_kv_heads", hp.llm_kv_heads);
    hp.llm_intermediate = mimo_kv_u32(gctx, "mimo_asr.llm.intermediate_size", hp.llm_intermediate);
    hp.llm_vocab        = mimo_kv_u32(gctx, "mimo_asr.llm.vocab_size", hp.llm_vocab);
    hp.llm_max_pos      = mimo_kv_u32(gctx, "mimo_asr.llm.max_position_embeddings", hp.llm_max_pos);
    hp.llm_rope_theta   = mimo_kv_f32(gctx, "mimo_asr.llm.rope_theta", hp.llm_rope_theta);
    hp.llm_rms_eps      = mimo_kv_f32(gctx, "mimo_asr.llm.rms_norm_eps", hp.llm_rms_eps);

    hp.audio_channels     = mimo_kv_u32(gctx, "mimo_asr.audio.channels", hp.audio_channels);
    hp.audio_group_size   = mimo_kv_u32(gctx, "mimo_asr.audio.group_size", hp.audio_group_size);
    hp.audio_layers       = mimo_kv_u32(gctx, "mimo_asr.audio.input_layers", hp.audio_layers);
    hp.audio_dim          = mimo_kv_u32(gctx, "mimo_asr.audio.input_dim", hp.audio_dim);
    hp.audio_heads        = mimo_kv_u32(gctx, "mimo_asr.audio.input_heads", hp.audio_heads);
    hp.audio_head_dim     = mimo_kv_u32(gctx, "mimo_asr.audio.input_head_dim", hp.audio_head_dim);
    hp.audio_intermediate = mimo_kv_u32(gctx, "mimo_asr.audio.input_intermediate", hp.audio_intermediate);

    // Vocab
    int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tok_key >= 0) {
        int n = gguf_get_arr_n(gctx, tok_key);
        ctx->vocab.resize(n);
        for (int i = 0; i < n; i++) {
            const char* s = gguf_get_arr_str(gctx, tok_key, i);
            if (s) ctx->vocab[i] = s;
        }
    }
    gguf_free(gctx);

    // Backends
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "mimo_asr: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend) ctx->backend = ctx->backend_cpu;

    // Load weights to CPU (Q4_K SIMD path; mirror gemma4_e2b)
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend_cpu, "mimo_asr", wl)) {
        fprintf(stderr, "mimo_asr: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    if (params.verbosity >= 1) {
        fprintf(stderr,
                "mimo_asr: loaded %zu tensors  llm=%uL/%u  audio=%uL/%u (×%u channels)\n",
                ctx->tensors.size(), hp.llm_layers, hp.llm_hidden,
                hp.audio_layers, hp.audio_dim, hp.audio_channels);
    }
    return ctx;
}

extern "C" int mimo_asr_set_tokenizer_path(struct mimo_asr_context* ctx, const char* path) {
    if (!ctx || !path) return -1;
    ctx->tokenizer_path = path;
    return 0;
}

extern "C" char* mimo_asr_transcribe(struct mimo_asr_context* /*ctx*/,
                                     const float* /*pcm*/, int /*n_samples*/) {
    // PLAN #51 — full forward pass not yet implemented. The model
    // loads cleanly; we just don't run the encoder + LLM yet.
    static const char kStub[] =
        "[mimo_asr: forward pass not yet implemented — PLAN #51]";
    char* out = (char*)malloc(sizeof(kStub));
    if (out) std::memcpy(out, kStub, sizeof(kStub));
    return out;
}

extern "C" void mimo_asr_free(struct mimo_asr_context* ctx) {
    if (!ctx) return;
    if (ctx->buf_w) ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w) ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu) ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void mimo_asr_set_n_threads(struct mimo_asr_context* ctx, int n_threads) {
    if (!ctx || n_threads <= 0) return;
    ctx->n_threads = n_threads;
    if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}
