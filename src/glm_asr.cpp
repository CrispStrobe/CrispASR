// glm_asr.cpp — GLM-ASR-Nano runtime (zai-org/GLM-ASR-Nano-2512).
//
// Architecture: Whisper encoder (1280d, 32L, partial RoPE, LayerNorm+bias)
//             + 4-frame-stack projector (5120→4096,GELU → 4096→2048)
//             + Llama LLM (2048d, 28L, GQA 16/4, SwiGLU, RMSNorm)
//
// Closely follows voxtral.cpp — same building blocks, different sizes.

#include "glm_asr.h"

#include "core/attention.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/mel.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ===========================================================================
// Model structures
// ===========================================================================

struct glm_asr_hparams {
    // Audio encoder
    int enc_hidden = 1280;
    int enc_n_layers = 32;
    int enc_n_heads = 20;
    int enc_n_kv_heads = 20;
    int enc_ff = 5120;
    int n_mels = 128;
    int enc_max_pos = 1500;
    float partial_rotary = 0.5f;
    // LLM decoder
    int llm_hidden = 2048;
    int llm_n_layers = 28;
    int llm_n_heads = 16;
    int llm_n_kv_heads = 4;
    int llm_ff = 6144;
    int llm_vocab = 59264;
    int llm_max_pos = 8192;
    float rms_eps = 1e-5f;
    // Special tokens
    int audio_token_id = 59260;
    int bos_token_id = 1;
    int eos_token_ids[4] = {59246, 59253, 59255, -1};
    int n_eos = 3;
    // Derived
    int enc_head_dim = 64;  // enc_hidden / enc_n_heads
    int llm_head_dim = 128; // llm_hidden / llm_n_heads
};

struct glm_enc_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
};

struct glm_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct glm_asr_model {
    glm_asr_hparams hp;

    // Audio encoder
    struct {
        ggml_tensor* conv1_w = nullptr;
        ggml_tensor* conv1_b = nullptr;
        ggml_tensor* conv2_w = nullptr;
        ggml_tensor* conv2_b = nullptr;
        ggml_tensor* ln_post_w = nullptr;
        ggml_tensor* ln_post_b = nullptr;
        std::vector<glm_enc_block> blocks;
    } audio;

    // Projector (4-frame stack → 2 linears)
    struct {
        ggml_tensor* linear1_w = nullptr;
        ggml_tensor* linear1_b = nullptr;
        ggml_tensor* linear2_w = nullptr;
        ggml_tensor* linear2_b = nullptr;
    } proj;

    // LLM
    struct {
        ggml_tensor* token_embd_w = nullptr;
        ggml_tensor* output_norm_w = nullptr;
        ggml_tensor* lm_head_w = nullptr;
        std::vector<glm_llm_block> blocks;
    } llm;

    // Tokenizer
    std::vector<std::string> vocab;

    // GGUF context (owns the weight memory)
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

struct glm_asr_context {
    glm_asr_context_params params;
    glm_asr_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;

    int n_threads = 4;
};

// ===========================================================================
// Implementation
// ===========================================================================

extern "C" struct glm_asr_context_params glm_asr_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1};
}

extern "C" struct glm_asr_context* glm_asr_init_from_file(const char* path_model,
                                                          struct glm_asr_context_params params) {
    auto* ctx = new glm_asr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    // Load GGUF
    auto& m = ctx->model;
    auto& hp = m.hp;

    // ---- pass 1: read hparams + vocab via metadata-only context ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path_model);
        if (!gctx) {
            fprintf(stderr, "glm_asr: failed to open '%s'\n", path_model);
            delete ctx;
            return nullptr;
        }
        hp.enc_hidden = core_gguf::kv_u32(gctx, "glmasr.audio.hidden_size", hp.enc_hidden);
        hp.enc_n_layers = core_gguf::kv_u32(gctx, "glmasr.audio.num_layers", hp.enc_n_layers);
        hp.enc_n_heads = core_gguf::kv_u32(gctx, "glmasr.audio.num_heads", hp.enc_n_heads);
        hp.enc_n_kv_heads = core_gguf::kv_u32(gctx, "glmasr.audio.num_kv_heads", hp.enc_n_kv_heads);
        hp.enc_ff = core_gguf::kv_u32(gctx, "glmasr.audio.intermediate_size", hp.enc_ff);
        hp.n_mels = core_gguf::kv_u32(gctx, "glmasr.audio.num_mel_bins", hp.n_mels);
        hp.enc_max_pos = core_gguf::kv_u32(gctx, "glmasr.audio.max_position_embeddings", hp.enc_max_pos);
        hp.partial_rotary = core_gguf::kv_f32(gctx, "glmasr.audio.partial_rotary_factor", hp.partial_rotary);

        hp.llm_hidden = core_gguf::kv_u32(gctx, "glmasr.llm.hidden_size", hp.llm_hidden);
        hp.llm_n_layers = core_gguf::kv_u32(gctx, "glmasr.llm.num_layers", hp.llm_n_layers);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "glmasr.llm.num_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "glmasr.llm.num_kv_heads", hp.llm_n_kv_heads);
        hp.llm_ff = core_gguf::kv_u32(gctx, "glmasr.llm.intermediate_size", hp.llm_ff);
        hp.llm_vocab = core_gguf::kv_u32(gctx, "glmasr.llm.vocab_size", hp.llm_vocab);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "glmasr.llm.max_position_embeddings", hp.llm_max_pos);
        hp.rms_eps = core_gguf::kv_f32(gctx, "glmasr.llm.rms_norm_eps", hp.rms_eps);

        hp.audio_token_id = core_gguf::kv_u32(gctx, "glmasr.audio_token_id", hp.audio_token_id);
        hp.bos_token_id = core_gguf::kv_u32(gctx, "glmasr.bos_token_id", hp.bos_token_id);
        hp.n_eos = core_gguf::kv_u32(gctx, "glmasr.num_eos_tokens", hp.n_eos);
        for (int i = 0; i < hp.n_eos && i < 4; i++) {
            char key[64];
            snprintf(key, sizeof(key), "glmasr.eos_token_id_%d", i);
            hp.eos_token_ids[i] = core_gguf::kv_u32(gctx, key, hp.eos_token_ids[i]);
        }

        hp.enc_head_dim = hp.enc_hidden / hp.enc_n_heads;
        hp.llm_head_dim = hp.llm_hidden / hp.llm_n_heads;

        // Tokenizer
        m.vocab.resize(hp.llm_vocab);
        const int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (tok_key >= 0) {
            const int n = gguf_get_arr_n(gctx, tok_key);
            for (int i = 0; i < n && i < hp.llm_vocab; i++) {
                const char* s = gguf_get_arr_str(gctx, tok_key, i);
                if (s)
                    m.vocab[i] = s;
            }
        }

        gguf_free(gctx);
    }

    // ---- pass 2: load tensor data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "glm_asr", wl)) {
        fprintf(stderr, "glm_asr: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    m.ctx = wl.ctx;
    m.buf = wl.buf;

    // Map tensors
    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        if (it == wl.tensors.end()) {
            fprintf(stderr, "glm_asr: tensor '%s' not found\n", name);
            return nullptr;
        }
        return it->second;
    };
    auto try_get = [&](const char* name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };

    // Audio encoder tensors
    m.audio.conv1_w = get("audio.conv1.weight");
    m.audio.conv1_b = get("audio.conv1.bias");
    m.audio.conv2_w = get("audio.conv2.weight");
    m.audio.conv2_b = get("audio.conv2.bias");
    m.audio.ln_post_w = get("audio.norm.weight");
    m.audio.ln_post_b = try_get("audio.norm.bias");
    m.audio.blocks.resize(hp.enc_n_layers);
    for (int i = 0; i < hp.enc_n_layers; i++) {
        char buf[128];
        auto& b = m.audio.blocks[i];
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_norm.weight", i);
        b.attn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_norm.bias", i);
        b.attn_norm_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_q.weight", i);
        b.attn_q_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_q.bias", i);
        b.attn_q_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_k.weight", i);
        b.attn_k_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_v.weight", i);
        b.attn_v_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_out.weight", i);
        b.attn_out_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_out.bias", i);
        b.attn_out_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn_norm.weight", i);
        b.ffn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn_norm.bias", i);
        b.ffn_norm_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.up.weight", i);
        b.ffn_up_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.up.bias", i);
        b.ffn_up_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.down.weight", i);
        b.ffn_down_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.down.bias", i);
        b.ffn_down_b = try_get(buf);
    }

    // Projector
    m.proj.linear1_w = get("proj.linear_1.weight");
    m.proj.linear1_b = try_get("proj.linear_1.bias");
    m.proj.linear2_w = get("proj.linear_2.weight");
    m.proj.linear2_b = try_get("proj.linear_2.bias");

    // LLM tensors
    m.llm.token_embd_w = get("llm.token_embd.weight");
    m.llm.output_norm_w = get("llm.norm.weight");
    m.llm.lm_head_w = get("lm_head.weight");
    m.llm.blocks.resize(hp.llm_n_layers);
    for (int i = 0; i < hp.llm_n_layers; i++) {
        char buf[128];
        auto& b = m.llm.blocks[i];
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_norm.weight", i);
        b.attn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn_norm.weight", i);
        b.ffn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_q.weight", i);
        b.attn_q_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_k.weight", i);
        b.attn_k_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_v.weight", i);
        b.attn_v_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_out.weight", i);
        b.attn_out_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn.gate.weight", i);
        b.ffn_gate_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn.up.weight", i);
        b.ffn_up_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn.down.weight", i);
        b.ffn_down_w = get(buf);
    }

    // Scheduler
    int n_be = 1;
    ggml_backend_t backends[2] = {ctx->backend, nullptr};
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
        backends[n_be++] = ctx->backend_cpu;
    }
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    int n_audio_t = 0, n_llm_t = 0;
    for (int i = 0; i < hp.enc_n_layers; i++) {
        if (m.audio.blocks[i].attn_q_w)
            n_audio_t++;
    }
    for (int i = 0; i < hp.llm_n_layers; i++) {
        if (m.llm.blocks[i].attn_q_w)
            n_llm_t++;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "glm_asr: loaded %d audio + %d LLM layers, vocab %d\n", n_audio_t, n_llm_t, hp.llm_vocab);
    }

    return ctx;
}

extern "C" void glm_asr_free(struct glm_asr_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" const char* glm_asr_token_text(struct glm_asr_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->model.vocab.size())
        return "";
    return ctx->model.vocab[id].c_str();
}

extern "C" int32_t* glm_asr_tokenize(struct glm_asr_context* ctx, const char* text, int* out_n_tokens) {
    // Simple tokenizer: look up each known special token, then byte-fallback.
    // For a production implementation, this would use the full tokenizer.
    if (!ctx || !text || !out_n_tokens)
        return nullptr;

    std::vector<int32_t> ids;
    std::string s(text);

    // Map special tokens
    struct {
        const char* text;
        int id;
    } specials[] = {
        {"<|begin_of_audio|>", 59261},
        {"<|end_of_audio|>", 59262},
        {"<|pad|>", 59260},
        {"<|user|>", 59253},
        {"<|assistant|>", 59254},
        {"<|system|>", 59252},
        {"<|endoftext|>", 59246},
        {"\n", -1}, // handled below
    };

    size_t pos = 0;
    while (pos < s.size()) {
        bool found = false;
        for (const auto& sp : specials) {
            size_t len = strlen(sp.text);
            if (s.compare(pos, len, sp.text) == 0) {
                if (sp.id >= 0)
                    ids.push_back(sp.id);
                pos += len;
                found = true;
                break;
            }
        }
        if (!found) {
            // Byte fallback — find the vocab entry for this character/substring
            // For now, skip unknown bytes
            pos++;
        }
    }

    *out_n_tokens = (int)ids.size();
    if (ids.empty())
        return nullptr;
    auto* result = (int32_t*)malloc(ids.size() * sizeof(int32_t));
    memcpy(result, ids.data(), ids.size() * sizeof(int32_t));
    return result;
}

// Stub implementations — the full pipeline will be implemented iteratively
// following the voxtral.cpp pattern. For now, expose the API surface.

extern "C" char* glm_asr_transcribe(struct glm_asr_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    const auto& hp = ctx->model.hp;

    // 1. Compute mel
    int n_mels = 0, T_mel = 0;
    float* mel = glm_asr_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel)
        return nullptr;

    // 2. Run encoder + projector
    int N_enc = 0, enc_dim = 0;
    float* audio_embeds = glm_asr_run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &enc_dim);
    free(mel);
    if (!audio_embeds)
        return nullptr;

    // 3. Build prompt: <|user|>\n<|begin_of_audio|><|pad|>×N<|end_of_audio|><|user|>\nPlease transcribe...<|assistant|>\n
    std::vector<int32_t> ids;
    ids.push_back(59253); // <|user|>
    ids.push_back(59261); // <|begin_of_audio|>
    for (int i = 0; i < N_enc; i++)
        ids.push_back(59260); // <|pad|> (audio placeholder)
    ids.push_back(59262);     // <|end_of_audio|>
    ids.push_back(59253);     // <|user|>

    // Tokenize the instruction — for now hardcode the token IDs for
    // "Please transcribe this audio into text"
    // TODO: proper tokenization
    ids.push_back(59254); // <|assistant|>

    // 4. Embed tokens
    float* text_embeds = glm_asr_embed_tokens(ctx, ids.data(), (int)ids.size());
    if (!text_embeds) {
        free(audio_embeds);
        return nullptr;
    }

    // 5. Splice audio embeddings into the <|pad|> positions
    const int pdim = enc_dim; // 2048
    int spliced = 0;
    for (size_t i = 0; i < ids.size() && spliced < N_enc; i++) {
        if (ids[i] == 59260) { // <|pad|>
            memcpy(text_embeds + i * pdim, audio_embeds + (size_t)spliced * pdim, pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);

    // 6. KV cache + prefill + greedy decode
    if (!glm_asr_kv_init(ctx, 4096)) {
        free(text_embeds);
        return nullptr;
    }
    glm_asr_kv_reset(ctx);

    int n_t = 0, vocab = 0;
    float* logits = glm_asr_run_llm_kv(ctx, text_embeds, (int)ids.size(), 0, &n_t, &vocab);
    free(text_embeds);
    if (!logits)
        return nullptr;

    // First token: argmax
    int next = 0;
    float mx = logits[0];
    for (int k = 1; k < vocab; k++) {
        if (logits[k] > mx) {
            mx = logits[k];
            next = k;
        }
    }
    free(logits);

    // Greedy decode loop
    std::string result;
    int n_past = (int)ids.size();
    const int max_tokens = 512;

    auto is_eos = [&](int id) {
        for (int i = 0; i < hp.n_eos; i++)
            if (id == hp.eos_token_ids[i])
                return true;
        return false;
    };

    for (int step = 0; step < max_tokens; step++) {
        if (is_eos(next))
            break;

        // Append token text
        if (next >= 0 && next < (int)ctx->model.vocab.size()) {
            const auto& tok = ctx->model.vocab[next];
            // Skip special tokens
            if (tok.size() < 2 || tok[0] != '<' || tok[1] != '|')
                result += tok;
        }

        // Forward one token
        float* emb = glm_asr_embed_tokens(ctx, &next, 1);
        if (!emb)
            break;
        float* lg = glm_asr_run_llm_kv(ctx, emb, 1, n_past, nullptr, nullptr);
        free(emb);
        if (!lg)
            break;
        n_past++;

        // Argmax
        next = 0;
        mx = lg[0];
        for (int k = 1; k < vocab; k++) {
            if (lg[k] > mx) {
                mx = lg[k];
                next = k;
            }
        }
        free(lg);
    }

    return strdup(result.c_str());
}

extern "C" float* glm_asr_compute_mel(struct glm_asr_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                      int* out_T_mel) {
    if (!ctx || !samples)
        return nullptr;
    const auto& hp = ctx->model.hp;

    // Get mel_filters and mel_window from model tensors
    ggml_tensor* mel_fb_t = nullptr;
    ggml_tensor* mel_win_t = nullptr;
    auto it = std::find_if(ctx->model.audio.blocks.begin(), ctx->model.audio.blocks.end(),
                           [](const glm_enc_block&) { return false; }); // dummy
    // Look up directly from the weight map — they were loaded via core_gguf
    // but not stored in a named field. Use ggml_get_tensor on the context.
    mel_fb_t = ggml_get_tensor(ctx->model.ctx, "audio.mel_filters");
    mel_win_t = ggml_get_tensor(ctx->model.ctx, "audio.mel_window");

    if (!mel_fb_t || !mel_win_t) {
        fprintf(stderr, "glm_asr: mel_filters or mel_window not found in model\n");
        return nullptr;
    }

    // Read mel data from backend
    int n_freqs = (int)mel_fb_t->ne[0];
    std::vector<float> mel_fb((size_t)n_freqs * hp.n_mels);
    ggml_backend_tensor_get(mel_fb_t, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    int win_len = (int)mel_win_t->ne[0];
    std::vector<float> mel_win(win_len);
    ggml_backend_tensor_get(mel_win_t, mel_win.data(), 0, mel_win.size() * sizeof(float));

    // Compute mel using core_mel
    core_mel::Params p;
    p.n_mels = hp.n_mels;
    p.n_fft = 400;
    p.hop_length = 160;
    p.center_pad = true;

    int T_mel = 0;
    auto mel = core_mel::compute(samples, n_samples, mel_win.data(), win_len, mel_fb.data(), n_freqs,
                                 nullptr, // use default FFT
                                 p, T_mel);
    if (mel.empty())
        return nullptr;

    if (out_n_mels)
        *out_n_mels = hp.n_mels;
    if (out_T_mel)
        *out_T_mel = T_mel;

    float* result = (float*)malloc(mel.size() * sizeof(float));
    memcpy(result, mel.data(), mel.size() * sizeof(float));
    return result;
}

extern "C" float* glm_asr_embed_tokens(struct glm_asr_context* ctx, const int32_t* ids, int n_ids) {
    if (!ctx || !ids || n_ids <= 0)
        return nullptr;

    const auto& m = ctx->model;
    const int d = m.hp.llm_hidden;

    // Build tiny graph: token_embd lookup
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ids);
    ggml_set_name(inp, "token_ids");
    ggml_set_input(inp);

    ggml_tensor* emb = ggml_get_rows(ctx0, m.llm.token_embd_w, inp);
    ggml_set_name(emb, "embeddings");
    ggml_build_forward_expand(gf, emb);
    ggml_free(ctx0);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "token_ids"), ids, 0, n_ids * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeddings");
    float* result = (float*)malloc((size_t)n_ids * d * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)n_ids * d * sizeof(float));
    return result;
}

extern "C" bool glm_asr_kv_init(struct glm_asr_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    const auto& hp = ctx->model.hp;
    const int hd = hp.llm_head_dim;
    const int n_kv = hp.llm_n_kv_heads;
    const int nl = hp.llm_n_layers;
    size_t k_size = (size_t)ggml_type_size(GGML_TYPE_F16) * hd * max_ctx * n_kv * nl;
    size_t total = k_size * 2 + ggml_tensor_overhead() * 2;

    ggml_init_params ip = {total, nullptr, false};
    ctx->kv_ctx = ggml_init(ip);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
        return false;
    }
    glm_asr_kv_reset(ctx);
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "glm_asr: kv cache %zu MiB\n", (k_size * 2) >> 20);
    }
    return true;
}

extern "C" void glm_asr_kv_reset(struct glm_asr_context* ctx) {
    if (!ctx || !ctx->kv_buf)
        return;
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
}

static ggml_cgraph* glm_build_encoder(glm_asr_context* ctx, int T_mel) {
    const auto& m = ctx->model;
    const auto& hp = m.hp;
    const int d = hp.enc_hidden;          // 1280
    const int n_heads = hp.enc_n_heads;   // 20
    const int hd = hp.enc_head_dim;       // 64
    const int n_layers = hp.enc_n_layers; // 32
    const int n_mels = hp.n_mels;         // 128
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input mel (n_mels, T_mel)
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // Conv stem: conv1(128→1280, k=3, s=1, p=1) + GELU + conv2(1280→1280, k=3, s=2, p=1) + GELU
    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };

    ggml_tensor* cur = ggml_conv_1d(ctx0, m.audio.conv1_w, mel, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);

    cur = ggml_conv_1d(ctx0, m.audio.conv2_w, cur, 2, 1, 1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);

    const int T_enc = (T_mel + 1) / 2; // stride 2
    cur = ggml_reshape_2d(ctx0, cur, T_enc, d);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (d, T_enc)

    // Positions for partial RoPE
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_enc);
    ggml_set_name(positions, "enc_positions");
    ggml_set_input(positions);

    // 32 × encoder blocks (pre-norm, LayerNorm with bias, partial RoPE)
    for (int il = 0; il < n_layers; il++) {
        const auto& b = m.audio.blocks[il];
        ggml_tensor* residual = cur;

        // Pre-attention LayerNorm
        ggml_tensor* x = ggml_norm(ctx0, cur, 1e-5f);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        if (b.attn_norm_b)
            x = ggml_add(ctx0, x, b.attn_norm_b);

        // Self-attention with partial RoPE
        // For simplicity, use encoder_self_attn which already handles biases.
        // Partial RoPE (factor=0.5) means RoPE on first hd/2 dims only.
        // For now, apply full RoPE — partial would need splitting Q/K.
        // TODO: implement partial RoPE for better accuracy.
        core_attn::EncoderSelfAttnParams eap = {};
        eap.n_heads = n_heads;
        eap.n_kv_heads = n_heads; // MHA
        eap.head_dim = hd;
        eap.n_kv_grp = 1;
        eap.attn_scale = scale;
        eap.n_ctx_orig = hp.enc_max_pos;
        eap.rope_theta = 10000.0f;
        eap.permute_cont = true;

        ggml_tensor* attn = core_attn::encoder_self_attn(
            ctx0, x, b.attn_q_w, b.attn_q_b, b.attn_k_w, nullptr, // K has no bias in GLM-ASR encoder
            b.attn_v_w, nullptr,                                  // V has no bias
            b.attn_out_w, b.attn_out_b, positions, nullptr,       // no mask (bidirectional)
            eap);
        cur = ggml_add(ctx0, residual, attn);

        // Post-attention LayerNorm + FFN (fc1 → GELU → fc2)
        residual = cur;
        x = ggml_norm(ctx0, cur, 1e-5f);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        if (b.ffn_norm_b)
            x = ggml_add(ctx0, x, b.ffn_norm_b);
        x = ggml_mul_mat(ctx0, b.ffn_up_w, x);
        if (b.ffn_up_b)
            x = ggml_add(ctx0, x, b.ffn_up_b);
        x = ggml_gelu_erf(ctx0, x);
        x = ggml_mul_mat(ctx0, b.ffn_down_w, x);
        if (b.ffn_down_b)
            x = ggml_add(ctx0, x, b.ffn_down_b);
        cur = ggml_add(ctx0, residual, x);
    }

    // Final LayerNorm
    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_mul(ctx0, cur, m.audio.ln_post_w);
    if (m.audio.ln_post_b)
        cur = ggml_add(ctx0, cur, m.audio.ln_post_b);

    // Projector: 4-frame stacking → linear1(5120→4096,GELU) → linear2(4096→2048)
    // Stack 4 consecutive frames: (d, T_enc) → (4*d, T_enc/4)
    const int T_proj = T_enc / 4;
    const int proj_in = 4 * d; // 5120
    cur = ggml_reshape_2d(ctx0, cur, proj_in, T_proj);

    cur = ggml_mul_mat(ctx0, m.proj.linear1_w, cur);
    if (m.proj.linear1_b)
        cur = ggml_add(ctx0, cur, m.proj.linear1_b);
    cur = ggml_gelu_erf(ctx0, cur);

    cur = ggml_mul_mat(ctx0, m.proj.linear2_w, cur);
    if (m.proj.linear2_b)
        cur = ggml_add(ctx0, cur, m.proj.linear2_b);

    ggml_set_name(cur, "encoder_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

extern "C" float* glm_asr_run_encoder(struct glm_asr_context* ctx, const float* mel, int n_mels, int T_mel, int* out_N,
                                      int* out_dim) {
    if (!ctx || !mel)
        return nullptr;

    const auto& hp = ctx->model.hp;
    const int T_enc = (T_mel + 1) / 2;
    const int T_proj = T_enc / 4;
    const int llm_d = hp.llm_hidden; // 2048

    ggml_cgraph* gf = glm_build_encoder(ctx, T_mel);
    if (!gf)
        return nullptr;

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    // Set mel input
    ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_t, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    // Set encoder positions
    ggml_tensor* pos = ggml_graph_get_tensor(gf, "enc_positions");
    std::vector<int32_t> pos_data(T_enc);
    for (int i = 0; i < T_enc; i++)
        pos_data[i] = i;
    ggml_backend_tensor_set(pos, pos_data.data(), 0, T_enc * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* out = ggml_graph_get_tensor(gf, "encoder_out");
    float* result = (float*)malloc((size_t)T_proj * llm_d * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)T_proj * llm_d * sizeof(float));

    if (out_N)
        *out_N = T_proj;
    if (out_dim)
        *out_dim = llm_d;
    return result;
}

static ggml_cgraph* glm_build_llm_kv(glm_asr_context* ctx, int n_past, int T, bool last_token_only) {
    const auto& m = ctx->model;
    const auto& hp = m.hp;
    const int H = hp.llm_hidden;
    const int n_heads = hp.llm_n_heads;
    const int n_kv = hp.llm_n_kv_heads;
    const int hd = hp.llm_head_dim;
    const int grp = n_heads / n_kv;
    const int L = hp.llm_n_layers;
    const int V = hp.llm_vocab;
    const float eps = hp.rms_eps;
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, H, T);
    ggml_set_name(cur, "llm_input");
    ggml_set_input(cur);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Causal mask for prefill (T > 1). For single-token decode, nullptr.
    const int Lk = n_past + T;
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    // L × Llama layers with KV cache
    for (int il = 0; il < L; il++) {
        const auto& b = m.llm.blocks[il];

        core_attn::KvSelfAttnParams ap = {};
        ap.n_heads = n_heads;
        ap.n_kv_heads = n_kv;
        ap.head_dim = hd;
        ap.n_kv_grp = grp;
        ap.n_ctx_orig = hp.llm_max_pos;
        ap.rope_theta = 10000.0f;
        ap.rope_beta_fast = 0.0f;
        ap.rope_beta_slow = 0.0f;
        ap.attn_scale = scale;
        ap.qk_norm_eps = 0.0f;
        ap.gqa_mode = core_attn::GQA_MANUAL_CONT;

        ggml_tensor* residual = cur;

        // Pre-attention RMSNorm
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, b.attn_norm_w);

        // KV-cached self-attention
        cur = core_attn::kv_self_attn(ctx0, gf, cur, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_out_w, nullptr,
                                      nullptr, // no Q/K norm
                                      positions, causal_mask, ctx->kv_k, ctx->kv_v, il, n_past, ap);

        cur = ggml_add(ctx0, residual, cur);

        // Pre-FFN RMSNorm + SwiGLU
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, b.ffn_norm_w);
        cur = core_ffn::swiglu(ctx0, cur, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, cur);
    }

    // Final RMSNorm
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);

    // Last-token-only lm_head (for decode steps)
    if (last_token_only && T > 1) {
        cur = ggml_view_2d(ctx0, cur, H, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }

    // LM head
    cur = ggml_mul_mat(ctx0, m.llm.lm_head_w, cur);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

extern "C" float* glm_asr_run_llm_kv(struct glm_asr_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                                     int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;

    const auto& hp = ctx->model.hp;
    const int H = hp.llm_hidden;
    const int V = hp.llm_vocab;

    ggml_cgraph* gf = glm_build_llm_kv(ctx, n_past, n_tokens, true);
    if (!gf)
        return nullptr;

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    // Set inputs
    ggml_tensor* inp = ggml_graph_get_tensor(gf, "llm_input");
    ggml_backend_tensor_set(inp, inputs_embeds, 0, (size_t)H * n_tokens * sizeof(float));

    ggml_tensor* pos = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> pos_data(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos_data[i] = n_past + i;
    ggml_backend_tensor_set(pos, pos_data.data(), 0, n_tokens * sizeof(int32_t));

    // Causal mask (prefill only)
    if (n_tokens > 1) {
        const int Lk = n_past + n_tokens;
        ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "causal_mask");
        if (mask_t) {
            std::vector<ggml_fp16_t> mask_data((size_t)Lk * n_tokens);
            const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q < n_tokens; q++)
                for (int k = 0; k < Lk; k++)
                    mask_data[(size_t)q * Lk + k] = (k <= n_past + q) ? zero : neg_inf;
            ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
        }
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    int n_out = (int)out->ne[1]; // 1 for last-token-only
    float* result = (float*)malloc((size_t)V * n_out * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)V * n_out * sizeof(float));

    if (out_n_tokens)
        *out_n_tokens = n_out;
    if (out_vocab_size)
        *out_vocab_size = V;
    return result;
}
