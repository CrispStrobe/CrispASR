// confucius4_tts.cpp — Confucius4-TTS backend (§377).
//
// T2S: GPT-2 (24L/1280d/20h) causal LM with custom embedding concatenation:
//   [condition_emb(1,1,1280) | text_emb(1,T,1280) | semantic_emb(1,T',1280)]
// Generates semantic codes (vocab 8194) autoregressively.
//
// S2A: Flow-matching DiT(13L) + WaveNet(8L) → 80-band mel.
// Vocoder: BigVGAN → 22050 Hz PCM (via indextts_voc with 22kHz hparams).

#include "confucius4_tts.h"
#include "indextts_voc.h"

#include "core/attention.h"
#include "core/ecapa_tdnn.h"
#include "core/sdpa.h" // manual-F32-default SDPA / flash opt-in (#nemotron flash class)
#include "core/spm_bpe.h"
#include "core/audio_resample.h"
#include "core/wav_reader.h"
#include "chatterbox_campplus.h"
#include "sidon.h"
#include "core/ggml_cpu_backend.h"
#include "core/gpu_backend_pref.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

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

// ---------------------------------------------------------------------------
// Model hparams
// ---------------------------------------------------------------------------

struct confucius4_t2s_hparams {
    int num_layers = 24;
    int model_dim = 1280;
    int num_heads = 20;
    int max_text_seq_lens = 520;
    int max_semantic_seq_lens = 1520;
    int vocab_size = 32000;
    int semantic_vocab_size = 8194;
    int text_embedding_dim = 4096;
    int speaker_embedding_dim = 1024;
    int start_semantic_token = 8192;
    int stop_semantic_token = 8193;
    int sample_rate = 22050;
    int head_dim() const { return model_dim / num_heads; }
    int ffn_dim() const { return model_dim * 4; } // GPT-2 default: 4x
};

// ---------------------------------------------------------------------------
// T2S model weights
// ---------------------------------------------------------------------------

struct confucius4_t2s_layer {
    // Pre-attention LayerNorm
    struct ggml_tensor* ln_1_w = nullptr;
    struct ggml_tensor* ln_1_b = nullptr;
    // QKV fused + output projection (GPT-2 Conv1D style)
    struct ggml_tensor* attn_qkv_w = nullptr;  // [3*d, d]
    struct ggml_tensor* attn_qkv_b = nullptr;  // [3*d]
    struct ggml_tensor* attn_proj_w = nullptr; // [d, d]
    struct ggml_tensor* attn_proj_b = nullptr;
    // Pre-FFN LayerNorm
    struct ggml_tensor* ln_2_w = nullptr;
    struct ggml_tensor* ln_2_b = nullptr;
    // FFN (GPT-2 MLP: c_fc + c_proj)
    struct ggml_tensor* ffn_fc_w = nullptr; // [4d, d]
    struct ggml_tensor* ffn_fc_b = nullptr;
    struct ggml_tensor* ffn_proj_w = nullptr; // [d, 4d]
    struct ggml_tensor* ffn_proj_b = nullptr;
};

struct confucius4_t2s_model {
    confucius4_t2s_hparams hp;

    // Text projector: Embedding(vocab,4096) → Linear(4096,4096) → SiLU → Linear(4096,d)
    struct ggml_tensor* text_embed_w = nullptr;    // [4096, vocab]
    struct ggml_tensor* text_proj_fc1_w = nullptr; // [4096, 4096]
    struct ggml_tensor* text_proj_fc1_b = nullptr;
    struct ggml_tensor* text_proj_fc2_w = nullptr; // [4096, 1280]  (note: ggml col-major)
    struct ggml_tensor* text_proj_fc2_b = nullptr;

    // Semantic embedding + position embeddings
    struct ggml_tensor* semantic_embed_w = nullptr; // [d, semantic_vocab]
    struct ggml_tensor* text_pos_embed_w = nullptr; // [d, max_text]
    struct ggml_tensor* sem_pos_embed_w = nullptr;  // [d, max_semantic]

    // GPT-2 transformer layers
    std::vector<confucius4_t2s_layer> layers;

    // Final norm + semantic head
    struct ggml_tensor* ln_f_w = nullptr; // GPT-2's own transformer.ln_f
    struct ggml_tensor* ln_f_b = nullptr;
    struct ggml_tensor* final_norm_w = nullptr;
    struct ggml_tensor* final_norm_b = nullptr;
    struct ggml_tensor* semantic_head_w = nullptr; // [d, semantic_vocab]
    struct ggml_tensor* semantic_head_b = nullptr;

    // Speaker encoder (Qwen3TTSSpeakerEncoder / ECAPA-TDNN): 76 tensors that
    // already ship in the T2S GGUF but were never bound.  The graph itself is
    // shared with qwen3-tts via core/ecapa_tdnn.h.
    core_ecapa::model spk_enc;
};

// ---------------------------------------------------------------------------
// S2A model weights (Semantic-to-Acoustic: flow-matching DiT + WaveNet)
// ---------------------------------------------------------------------------

struct confucius4_s2a_hparams {
    int input_size = 512;
    int output_size = 80; // mel bands
    int spk_embed_dim = 192;
    int semantic_embed_dim = 1024;
    int lm_latent_dim = 1280;
    int estimator_depth = 13; // DiT layers
    int estimator_num_heads = 8;
    int estimator_hidden_dim = 512;
    int wavenet_num_layers = 8;
    int estimator_mel_dim = 0; // detected from weights; 0 = use output_size
};

struct confucius4_s2a_model {
    confucius4_s2a_hparams hp;

    // Semantic token embedding: Embedding(8192,8) → Linear(8,1024)
    struct ggml_tensor* input_embed_w = nullptr;
    struct ggml_tensor* input_proj_w = nullptr;
    struct ggml_tensor* input_proj_b = nullptr;

    // Encoder projection: Linear(lm_latent_dim + semantic_embed_dim, lr_in_channels)
    struct ggml_tensor* encoder_proj_w = nullptr;
    struct ggml_tensor* encoder_proj_b = nullptr;

    // Learned prompt condition
    struct ggml_tensor* prompt_cond = nullptr;

    // Weight context + buffer (separate from T2S)
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    core_gguf::tensor_map tensors; // all tensors by name for DiT/WaveNet/LR access

    // Fused WaveNet weights (weight_norm folded at load time for ggml graph use)
    ggml_context* ctx_wn = nullptr;
    ggml_backend_buffer_t buf_wn = nullptr;

    bool loaded = false;
};

// DiT graph cache for S2A flow-matching estimator (cached across ODE steps).
struct confucius4_dit_cache {
    ggml_context* gctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* hidden_in = nullptr; // (dim, T)   — input embedding
    ggml_tensor* t_emb_in = nullptr;  // (dim,)     — timestep embedding (t1 for DiT)
    ggml_tensor* t2_emb_in = nullptr; // (dim,)     — timestep embedding (t2 for WaveNet)
    ggml_tensor* x_mel_in = nullptr;  // (mel_dim, T) — original x for skip_linear
    ggml_tensor* pos_in = nullptr;    // (T,) I32   — RoPE position indices
    ggml_tensor* attn_mask = nullptr; // (2T, 2T) F16 — block-diagonal mask (fused CFG only)
    ggml_tensor* output = nullptr;    // (mel_dim, T) — velocity output
    int T_cached = 0;
    int mel_dim_cached = 0;
    bool fused_cfg = false; // graph runs cond+uncond seq-concat with in-graph blend
    float cfg_rate_cached = 0.0f;
    std::vector<ggml_fp16_t> mask_host; // host copy of the block-diagonal mask

    void reset() {
        if (galloc) {
            ggml_gallocr_free(galloc);
            galloc = nullptr;
        }
        if (gctx) {
            ggml_free(gctx);
            gctx = nullptr;
        }
        gf = nullptr;
        hidden_in = t_emb_in = t2_emb_in = x_mel_in = pos_in = attn_mask = output = nullptr;
        T_cached = mel_dim_cached = 0;
        fused_cfg = false;
        cfg_rate_cached = 0.0f;
        mask_host.clear();
    }
    ~confucius4_dit_cache() { reset(); }
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

// KV cache for the GPT-2 T2S model.
struct confucius4_kv_cache {
    ggml_tensor* k = nullptr; // (hd, max_seq, n_heads, n_layers) F16
    ggml_tensor* v = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int max_seq_len = 0;

    // Persistent single-token decode graph (CRISPASR_CONFUCIUS4_PERSIST=1).
    // Built ONCE per cache with fixed_kv_len = max_seq_len + kv_indices, so
    // the topology is invariant across n_past — the per-step ggml context
    // build + gallocr alloc/free (x3 beams x ~170 steps) disappears. ALL
    // inputs (x, mask, index) are re-set on EVERY compute (gallocr reuses
    // input buffers as scratch — the §234 aliasing gotcha).
    ggml_context* step_ctx = nullptr;
    ggml_cgraph* step_gf = nullptr;
    ggml_gallocr_t step_galloc = nullptr;
    ggml_tensor* step_x = nullptr;      // (D, 1) F32 input embed
    ggml_tensor* step_mask = nullptr;   // (max_seq, 1) F16: 0 for j<=n_past, -inf beyond
    ggml_tensor* step_idx = nullptr;    // (1,) I32 = n_past (cache write slot)
    ggml_tensor* step_logits = nullptr; // (V, 1)
    ggml_tensor* step_hidden = nullptr; // (D, 1) lm_hidden
};

struct confucius4_tts_context {
    confucius4_tts_params params;
    confucius4_t2s_model t2s;
    confucius4_s2a_model s2a;
    confucius4_kv_cache kv;
    confucius4_dit_cache dit_cache;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    struct ggml_context* ctx_w = nullptr;

    // Pre-computed speaker conditioning (from external Wav2Vec2-BERT + CAMPPlus)
    std::vector<float> speaker_semantic_features; // (n_frames, 1024)
    int speaker_n_frames = 0;
    std::vector<float> speaker_style_embedding; // (192,)
    bool has_speaker = false;

    // Pre-computed T2S condition embedding: speaker_encoder(w2v_bert_layer17)
    // in the reference, i.e. the (1, model_dim) vector prepended to the GPT-2
    // prefix.  Injecting it directly avoids porting the 600M w2v-BERT and the
    // ECAPA-TDNN speaker encoder just to prove the pipeline.
    std::vector<float> condition_embedding; // (model_dim,)
    bool has_condition_emb = false;

    // Reference mel prompt for the S2A stage (prompt_feat in the blueprint):
    // (n_prompt_frames, mel_dim), prepended to the flow-matching state.
    std::vector<float> prompt_mel;
    int prompt_n_frames = 0;

    // Host-side F32 copies of the semantic token + position embedding tables.
    // embed_semantic_token used to build a 2-op ggml graph PER TOKEN (x3 beams
    // per decode step); a dequantized table lookup is byte-identical (same
    // ggml to_float dequant) and free. CRISPASR_CONFUCIUS4_GRAPH_EMBED=1
    // restores the graph path for A/B.
    std::vector<float> sem_embed_tab; // (semantic_vocab, model_dim)
    std::vector<float> sem_pos_tab;   // (max_semantic, model_dim)

    // w2v-BERT 2.0 encoder-only (17L sidon-layout GGUF) for the T2S
    // condition_emb — layer-17 hidden states, z-normalised with the baked
    // stats, fed to the native ECAPA.
    sidon_context* w2v_ctx = nullptr;

    // CAMPPlus style encoder (baked into the S2A GGUF under campplus.*)
    cb_campplus_model campplus_model{};
    chatterbox_campplus::cb_campplus_runtime campplus_rt{};
    bool has_campplus = false;

    // BPE tokenizer (loaded from GGUF tokenizer.ggml.tokens + merges)
    std::vector<std::string> bpe_id_to_token;
    std::unordered_map<std::string, int32_t> bpe_token_to_id;
    std::unordered_map<std::string, int32_t> bpe_merge_rank;
    bool has_bpe_tokenizer = false;

    // BigVGAN vocoder (loaded from separate companion GGUF)
    indextts_voc_context* vocoder = nullptr;

    // w2v-bert normalisation stats
    std::vector<float> w2v_mean; // (1024,)
    std::vector<float> w2v_var;  // (1024,)
};

// ---------------------------------------------------------------------------
// Default params
// ---------------------------------------------------------------------------

confucius4_tts_params confucius4_tts_default_params(void) {
    confucius4_tts_params p{};
    p.n_threads = 0;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.8f;
    p.top_p = 0.8f;
    p.top_k = 30;
    p.repetition_penalty = 10.0f;
    p.max_semantic_tokens = 0; // 0 = use hparams default (1520)
    p.ode_steps = 25;          // S2A flow-matching ODE steps (reference default)
    p.cfg_rate = 0.7f;         // S2A classifier-free guidance (reference default)
    p.seed = 0;
    return p;
}

// Forward declarations for helpers used during loading
static std::vector<float> s2a_read_f32(ggml_tensor* t);
static ggml_tensor* s2a_find(const confucius4_s2a_model& s, const std::string& name);
static bool bind_speaker_encoder(confucius4_tts_context* ctx, const core_gguf::tensor_map& tensors);
static bool confucius4_bind_campplus(confucius4_tts_context* ctx);
static bool compute_condition_embedding(confucius4_tts_context* ctx);
// S2A stage-dump helpers (defined further down, next to the flow-matching code)
static const char* s2a_dump_dir();
static void s2a_dump_raw(const char* name, const void* data, size_t nbytes, const char* shape);

// ---------------------------------------------------------------------------
// Load T2S model from GGUF
// ---------------------------------------------------------------------------

static bool load_t2s(confucius4_tts_context* ctx, const char* path) {
    auto& m = ctx->t2s;
    auto& hp = m.hp;

    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta) {
        fprintf(stderr, "confucius4: cannot open T2S GGUF '%s'\n", path);
        return false;
    }

    // Read hparams from GGUF KV
    hp.num_layers = core_gguf::kv_u32(meta, "confucius4.t2s.num_layers", 24);
    hp.model_dim = core_gguf::kv_u32(meta, "confucius4.t2s.model_dim", 1280);
    hp.num_heads = core_gguf::kv_u32(meta, "confucius4.t2s.num_heads", 20);
    hp.max_text_seq_lens = core_gguf::kv_u32(meta, "confucius4.t2s.max_text_seq_lens", 520);
    hp.max_semantic_seq_lens = core_gguf::kv_u32(meta, "confucius4.t2s.max_semantic_seq_lens", 1520);
    hp.vocab_size = core_gguf::kv_u32(meta, "confucius4.t2s.vocab_size", 32000);
    hp.semantic_vocab_size = core_gguf::kv_u32(meta, "confucius4.t2s.semantic_vocab_size", 8194);
    hp.text_embedding_dim = core_gguf::kv_u32(meta, "confucius4.t2s.text_embedding_dim", 4096);
    hp.speaker_embedding_dim = core_gguf::kv_u32(meta, "confucius4.t2s.speaker_embedding_dim", 1024);
    hp.start_semantic_token = core_gguf::kv_u32(meta, "confucius4.t2s.start_semantic_token", 8192);
    hp.stop_semantic_token = core_gguf::kv_u32(meta, "confucius4.t2s.stop_semantic_token", 8193);
    hp.sample_rate = core_gguf::kv_u32(meta, "confucius4.sample_rate", 22050);

    // Read w2v-bert normalisation stats
    ctx->w2v_mean = core_gguf::kv_f32_array(meta, "confucius4.w2v_bert.mean");
    ctx->w2v_var = core_gguf::kv_f32_array(meta, "confucius4.w2v_bert.var");

    // BPE tokenizer from GGUF (baked by the converter or vocab-baking kernel)
    auto tok = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
    if (!tok.empty()) {
        ctx->bpe_id_to_token = std::move(tok);
        ctx->bpe_token_to_id.reserve(ctx->bpe_id_to_token.size());
        for (int i = 0; i < (int)ctx->bpe_id_to_token.size(); i++)
            ctx->bpe_token_to_id[ctx->bpe_id_to_token[i]] = i;
        auto merges = core_gguf::kv_str_array(meta, "tokenizer.ggml.merges");
        for (size_t i = 0; i < merges.size(); i++)
            ctx->bpe_merge_rank[merges[i]] = (int32_t)i;
        ctx->has_bpe_tokenizer = true;
        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "confucius4: BPE tokenizer loaded: %zu tokens, %zu merges\n", ctx->bpe_id_to_token.size(),
                    ctx->bpe_merge_rank.size());
    }

    core_gguf::free_metadata(meta);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "confucius4: T2S hparams: %dL/%dd/%dh, text_vocab=%d, semantic_vocab=%d\n", hp.num_layers,
                hp.model_dim, hp.num_heads, hp.vocab_size, hp.semantic_vocab_size);
    }

    // Two-pass GGUF load via core_gguf::load_weights
    m.layers.resize(hp.num_layers);

    ctx->backend_cpu = core_cpu_backend::init();
    core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->params.n_threads);
    if (ctx->params.use_gpu) {
        ctx->backend = crispasr_init_gpu_backend();
        if (!ctx->backend)
            ctx->backend = ctx->backend_cpu;
    } else {
        ctx->backend = ctx->backend_cpu;
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "confucius4-t2s", wl)) {
        fprintf(stderr, "confucius4: failed to load T2S weights\n");
        return false;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;

    auto find = [&](const char* name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };

    // Text projector
    m.text_embed_w = find("text_projector.embed.weight");
    m.text_proj_fc1_w = find("text_projector.text_projection_fc1.weight");
    m.text_proj_fc1_b = find("text_projector.text_projection_fc1.bias");
    m.text_proj_fc2_w = find("text_projector.text_projection_fc2.weight");
    m.text_proj_fc2_b = find("text_projector.text_projection_fc2.bias");

    // Semantic + positional embeddings
    m.semantic_embed_w = find("semantic_embedding.weight");
    m.text_pos_embed_w = find("text_position_embedding.embedding.weight");
    m.sem_pos_embed_w = find("semantic_position_embedding.embedding.weight");

    // GPT-2 transformer layers
    for (int i = 0; i < hp.num_layers; i++) {
        auto& L = m.layers[i];
        char buf[128];
        auto tn = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "transformer.h.%d.%s", i, suffix);
            return find(buf);
        };
        L.ln_1_w = tn("ln_1.weight");
        L.ln_1_b = tn("ln_1.bias");
        L.attn_qkv_w = tn("attn.c_attn.weight");
        L.attn_qkv_b = tn("attn.c_attn.bias");
        L.attn_proj_w = tn("attn.c_proj.weight");
        L.attn_proj_b = tn("attn.c_proj.bias");
        L.ln_2_w = tn("ln_2.weight");
        L.ln_2_b = tn("ln_2.bias");
        L.ffn_fc_w = tn("mlp.c_fc.weight");
        L.ffn_fc_b = tn("mlp.c_fc.bias");
        L.ffn_proj_w = tn("mlp.c_proj.weight");
        L.ffn_proj_b = tn("mlp.c_proj.bias");
    }

    // Final norms + head. TWO LayerNorms stack at the top of the reference:
    // GPT2Model's internal transformer.ln_f (its output is the lm_latent the
    // S2A conditioning consumes), then the model's own final_norm feeding
    // semantic_head. Skipping ln_f wrecks both the logits and the latents.
    m.ln_f_w = find("transformer.ln_f.weight");
    m.ln_f_b = find("transformer.ln_f.bias");
    m.final_norm_w = find("final_norm.weight");
    m.final_norm_b = find("final_norm.bias");
    m.semantic_head_w = find("semantic_head.weight");
    m.semantic_head_b = find("semantic_head.bias");

    // Verify critical tensors
    if (!m.ln_f_w || !m.ln_f_b || !m.final_norm_w || !m.final_norm_b) {
        fprintf(stderr, "confucius4: missing transformer.ln_f / final_norm tensors\n");
        return false;
    }
    if (!m.text_embed_w || !m.semantic_embed_w || !m.semantic_head_w) {
        fprintf(stderr, "confucius4: missing critical T2S tensors\n");
        return false;
    }
    if (!m.layers[0].attn_qkv_w || !m.layers[hp.num_layers - 1].attn_qkv_w) {
        fprintf(stderr, "confucius4: missing transformer layer tensors\n");
        return false;
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "confucius4: T2S loaded %zu tensors OK\n", wl.tensors.size());

    bind_speaker_encoder(ctx, wl.tensors);

    // Bake the host-side embedding tables (see sem_embed_tab comment).
    if (std::getenv("CRISPASR_CONFUCIUS4_GRAPH_EMBED") == nullptr) {
        ctx->sem_embed_tab = s2a_read_f32(m.semantic_embed_w);
        ctx->sem_pos_tab = s2a_read_f32(m.sem_pos_embed_w);
        if ((int)ctx->sem_embed_tab.size() != hp.semantic_vocab_size * hp.model_dim ||
            (int)ctx->sem_pos_tab.size() != hp.max_semantic_seq_lens * hp.model_dim) {
            ctx->sem_embed_tab.clear();
            ctx->sem_pos_tab.clear();
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// GPT-2 transformer forward: one step (prefill or single-token decode)
// ---------------------------------------------------------------------------

// Build a single GPT-2 forward pass graph for T input tokens.
// Returns the logits tensor (T, semantic_vocab_size).
// `kv_k` / `kv_v` are the KV cache tensors per layer (pre-allocated).
// `n_past` is the KV cache position (0 for prefill).
static ggml_tensor* gpt2_forward(confucius4_tts_context* ctx, ggml_context* ctx0, ggml_cgraph* gf,
                                 ggml_tensor* input_emb, // (model_dim, T)
                                 ggml_tensor* kv_k, ggml_tensor* kv_v, int n_past,
                                 ggml_tensor* causal_mask, // [Lk, T] F16 or nullptr (T==1 only)
                                 int fixed_kv_len = 0, ggml_tensor* kv_indices = nullptr) {
    const auto& m = ctx->t2s;
    const auto& hp = m.hp;

    // Value-initialize: KvSelfAttnParams has no default member initializers
    // for the leading fields, and `soft_max_ext` uses attn_scale VERBATIM —
    // the old `KvSelfAttnParams ap;` left attn_scale (and n_ctx_orig,
    // gqa_mode, ...) as indeterminate stack garbage, which is exactly the
    // "plausible activations, wrong numbers" T2S signature.
    core_attn::KvSelfAttnParams ap{};
    ap.n_heads = hp.num_heads;
    ap.n_kv_heads = hp.num_heads; // GPT-2: no GQA
    ap.n_kv_grp = 1;
    ap.head_dim = hp.head_dim();
    ap.rope_theta = 0.0f; // GPT-2 uses learned positional embeddings, not RoPE
    ap.attn_scale = 1.0f / std::sqrt((float)hp.head_dim());
    ap.gqa_mode = core_attn::GQA_NATIVE; // moot at n_kv_grp==1, but defined

    // CRISPASR_CONFUCIUS4_MAX_LAYERS=N overrides num_layers for debugging.
    int n_layers = hp.num_layers;
    if (const char* env = std::getenv("CRISPASR_CONFUCIUS4_MAX_LAYERS"))
        n_layers = std::min(n_layers, std::atoi(env));

    ggml_tensor* x = input_emb;

    for (int il = 0; il < n_layers; il++) {
        const auto& L = m.layers[il];

        // Pre-attention LayerNorm
        ggml_tensor* ln1 = ggml_norm(ctx0, x, 1e-5f);
        ln1 = ggml_add(ctx0, ggml_mul(ctx0, ln1, L.ln_1_w), L.ln_1_b);

        // Self-attention with fused QKV (GPT-2 style)
        ggml_tensor* attn_out =
            core_attn::kv_self_attn(ctx0, gf, ln1,
                                    /*q_w=*/nullptr, /*k_w=*/nullptr, /*v_w=*/nullptr, L.attn_proj_w,
                                    /*q_norm_w=*/nullptr, /*k_norm_w=*/nullptr,
                                    /*positions=*/nullptr, causal_mask, kv_k, kv_v, il, n_past, ap,
                                    /*qkv_w=*/L.attn_qkv_w, fixed_kv_len, kv_indices,
                                    /*q_b=*/nullptr, /*k_b=*/nullptr, /*v_b=*/nullptr,
                                    /*o_b=*/L.attn_proj_b, /*qkv_b=*/L.attn_qkv_b);

        // Residual
        x = ggml_add(ctx0, x, attn_out);

        // Pre-FFN LayerNorm
        ggml_tensor* ln2 = ggml_norm(ctx0, x, 1e-5f);
        ln2 = ggml_add(ctx0, ggml_mul(ctx0, ln2, L.ln_2_w), L.ln_2_b);

        // FFN: Linear → GELU → Linear (GPT-2 MLP)
        ggml_tensor* ff = ggml_mul_mat(ctx0, L.ffn_fc_w, ln2);
        ff = ggml_add(ctx0, ff, L.ffn_fc_b);
        ff = ggml_gelu(ctx0, ff);
        ff = ggml_mul_mat(ctx0, L.ffn_proj_w, ff);
        ff = ggml_add(ctx0, ff, L.ffn_proj_b);

        // Residual
        x = ggml_add(ctx0, x, ff);
    }

    // GPT-2's own final LayerNorm (transformer.ln_f). Its output is the
    // reference's last_hidden_state — the LM latent the S2A conditioning uses.
    x = ggml_norm(ctx0, x, 1e-5f);
    x = ggml_add(ctx0, ggml_mul(ctx0, x, m.ln_f_w), m.ln_f_b);
    ggml_set_name(x, "lm_hidden");
    ggml_set_output(x);

    // Separate head norm: logits = semantic_head(final_norm(hidden))
    ggml_tensor* hn = ggml_norm(ctx0, x, 1e-5f);
    hn = ggml_add(ctx0, ggml_mul(ctx0, hn, m.final_norm_w), m.final_norm_b);

    // Semantic head: Linear(model_dim, semantic_vocab_size)
    ggml_tensor* logits = ggml_mul_mat(ctx0, m.semantic_head_w, hn);
    logits = ggml_add(ctx0, logits, m.semantic_head_b);

    return logits;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

confucius4_tts_context* confucius4_tts_init_from_file(const char* path_t2s, confucius4_tts_params params) {
    if (!path_t2s || !*path_t2s)
        return nullptr;
    auto* ctx = new confucius4_tts_context();
    ctx->params = params;

    if (!load_t2s(ctx, path_t2s)) {
        delete ctx;
        return nullptr;
    }

    return ctx;
}

int confucius4_tts_set_s2a_path(confucius4_tts_context* ctx, const char* path_s2a) {
    if (!ctx || !path_s2a)
        return -1;
    auto& s = ctx->s2a;
    auto& hp = s.hp;

    gguf_context* meta = core_gguf::open_metadata(path_s2a);
    if (!meta) {
        fprintf(stderr, "confucius4: cannot open S2A GGUF '%s'\n", path_s2a);
        return -1;
    }

    hp.input_size = core_gguf::kv_u32(meta, "confucius4.s2a.input_size", 512);
    hp.output_size = core_gguf::kv_u32(meta, "confucius4.s2a.output_size", 80);
    hp.spk_embed_dim = core_gguf::kv_u32(meta, "confucius4.s2a.spk_embed_dim", 192);
    hp.semantic_embed_dim = core_gguf::kv_u32(meta, "confucius4.s2a.semantic_embed_dim", 1024);
    hp.lm_latent_dim = core_gguf::kv_u32(meta, "confucius4.s2a.lm_latent_dim", 1280);
    hp.estimator_depth = core_gguf::kv_u32(meta, "confucius4.s2a.estimator_depth", 13);
    hp.estimator_num_heads = core_gguf::kv_u32(meta, "confucius4.s2a.estimator_num_heads", 8);
    hp.estimator_hidden_dim = core_gguf::kv_u32(meta, "confucius4.s2a.estimator_hidden_dim", 512);
    hp.wavenet_num_layers = core_gguf::kv_u32(meta, "confucius4.s2a.wavenet_num_layers", 8);
    core_gguf::free_metadata(meta);

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "confucius4: S2A hparams: DiT %dL/%dd/%dh, WaveNet %dL, mel=%d\n", hp.estimator_depth,
                hp.estimator_hidden_dim, hp.estimator_num_heads, hp.wavenet_num_layers, hp.output_size);

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_s2a, ctx->backend, "confucius4-s2a", wl)) {
        fprintf(stderr, "confucius4: failed to load S2A weights\n");
        return -1;
    }
    s.ctx_w = wl.ctx;
    s.buf_w = wl.buf;
    s.tensors = std::move(wl.tensors);

    auto find = [&](const char* name) -> ggml_tensor* {
        auto it = s.tensors.find(name);
        return it != s.tensors.end() ? it->second : nullptr;
    };

    s.input_embed_w = find("input_embedding.embedding.weight");
    s.input_proj_w = find("input_embedding.out_project.weight");
    s.input_proj_b = find("input_embedding.out_project.bias");
    s.encoder_proj_w = find("encoder_proj.weight");
    s.encoder_proj_b = find("encoder_proj.bias");
    s.prompt_cond = find("prompt_cond");

    if (!s.input_embed_w || !s.encoder_proj_w || !s.prompt_cond) {
        fprintf(stderr, "confucius4: missing critical S2A tensors\n");
        return -1;
    }

    // Detect DiT mel_dim from input_embed.proj weight: proj_in = dim + 2*mel_dim + spk_dim
    auto proj_w = find("decoder.estimator.input_embed.proj.weight");
    if (proj_w) {
        int proj_in = (int)proj_w->ne[0];
        int det = (proj_in - hp.estimator_hidden_dim - hp.spk_embed_dim) / 2;
        if (det > 0) {
            hp.estimator_mel_dim = det;
            if (ctx->params.verbosity >= 1)
                fprintf(stderr, "confucius4: detected DiT mel_dim=%d from proj weight (proj_in=%d)\n", det, proj_in);
        }
    }
    if (hp.estimator_mel_dim <= 0)
        hp.estimator_mel_dim = hp.output_size;

    // Fold WaveNet weight_norm tensors (weight_g + weight_v → fused weight)
    // into new ggml tensors on the backend, so the ggml graph can use them.
    {
        // Collect weight_norm pairs: cond_layer + 8×in_layers + 8×res_skip_layers
        struct wn_pair {
            std::string base; // tensor name base (without .weight_g/.weight_v)
        };
        std::vector<wn_pair> wn_pairs;
        wn_pairs.push_back({"decoder.estimator.wavenet.cond_layer.conv"});
        for (int i = 0; i < hp.wavenet_num_layers; i++) {
            char buf[256];
            snprintf(buf, sizeof(buf), "decoder.estimator.wavenet.in_layers.%d.conv", i);
            wn_pairs.push_back({buf});
            snprintf(buf, sizeof(buf), "decoder.estimator.wavenet.res_skip_layers.%d.conv", i);
            wn_pairs.push_back({buf});
        }

        // Count how many tensors we need to create
        size_t total_bytes = 0;
        int n_wn = 0;
        for (const auto& p : wn_pairs) {
            auto wv = find((p.base + ".weight_v").c_str());
            if (wv) {
                total_bytes += ggml_nelements(wv) * sizeof(float);
                n_wn++;
            }
        }

        if (n_wn > 0) {
            // Create a ggml context for the fused weight tensors
            size_t ctx_size = (size_t)n_wn * ggml_tensor_overhead() + 64;
            ggml_init_params ip = {ctx_size, nullptr, true};
            s.ctx_wn = ggml_init(ip);

            // Allocate backend buffer for the fused weights
            s.buf_wn = ggml_backend_alloc_buffer(ctx->backend, total_bytes + 256);

            ggml_tallocr talloc = ggml_tallocr_new(s.buf_wn);
            int n_folded = 0;

            for (const auto& p : wn_pairs) {
                auto wg = find((p.base + ".weight_g").c_str());
                auto wv = find((p.base + ".weight_v").c_str());
                if (!wg || !wv)
                    continue;

                // Create fused tensor with same shape as weight_v
                std::string fused_name = p.base + ".weight";
                ggml_tensor* fused;
                if (ggml_n_dims(wv) == 3)
                    fused = ggml_new_tensor_3d(s.ctx_wn, GGML_TYPE_F32, wv->ne[0], wv->ne[1], wv->ne[2]);
                else
                    fused = ggml_new_tensor_2d(s.ctx_wn, GGML_TYPE_F32, wv->ne[0], wv->ne[1]);
                ggml_set_name(fused, fused_name.c_str());
                ggml_tallocr_alloc(&talloc, fused);

                // Read weight_v and weight_g, fold, write to backend
                auto v = s2a_read_f32(wv);
                auto g = s2a_read_f32(wg);
                int64_t ne2 = ggml_n_dims(wv) >= 3 ? wv->ne[2] : wv->ne[1];
                int64_t vec_len = (int64_t)v.size() / ne2;
                for (int64_t co = 0; co < ne2; co++) {
                    float norm_sq = 0.0f;
                    size_t off = (size_t)(co * vec_len);
                    for (int64_t i = 0; i < vec_len; i++)
                        norm_sq += v[off + (size_t)i] * v[off + (size_t)i];
                    float scale = g[(size_t)co] / (std::sqrt(norm_sq) + 1e-12f);
                    for (int64_t i = 0; i < vec_len; i++)
                        v[off + (size_t)i] *= scale;
                }
                ggml_backend_tensor_set(fused, v.data(), 0, v.size() * sizeof(float));

                // Register in tensor map under the fused name
                s.tensors[fused_name] = fused;
                n_folded++;
            }
            if (ctx->params.verbosity >= 1)
                fprintf(stderr, "confucius4: folded %d WaveNet weight_norm pairs\n", n_folded);
        }
    }

    s.loaded = true;
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "confucius4: S2A loaded %zu tensors OK\n", s.tensors.size());
    confucius4_bind_campplus(ctx); // optional: older GGUFs lack the campplus bake
    return 0;
}


// Bind the CAMPPlus style encoder baked into the S2A GGUF under `campplus.*`
// (same backbone as chatterbox/dots; funasr campplus_cn_common, 192-d).
// Mirrors dots_bind_campplus with the S2A tensor-map lookup.
static bool confucius4_bind_campplus(confucius4_tts_context* ctx) {
    auto& s = ctx->s2a;
    auto& m = ctx->campplus_model;
    char k[192];
    const char* P = "campplus";
    auto T = [&](const char* name) -> ggml_tensor* { return s2a_find(s, name); };

    auto bind_unit = [&](cb_campplus_unit& u, const std::string& base) {
        u.lin_w = T((base + ".linear.weight").c_str());
        u.lin_b = T((base + ".linear.bias").c_str());
        u.bn_w = T((base + ".nonlinear.batchnorm.weight").c_str());
        u.bn_b = T((base + ".nonlinear.batchnorm.bias").c_str());
        u.bn_m = T((base + ".nonlinear.batchnorm.running_mean").c_str());
        u.bn_v = T((base + ".nonlinear.batchnorm.running_var").c_str());
    };
    auto bind_resblock = [&](cb_campplus_resblock& b, const std::string& base) {
        b.conv1_w = T((base + ".conv1.weight").c_str());
        b.conv1_b = T((base + ".conv1.bias").c_str());
        b.bn1_w = T((base + ".bn1.weight").c_str());
        b.bn1_b = T((base + ".bn1.bias").c_str());
        b.bn1_m = T((base + ".bn1.running_mean").c_str());
        b.bn1_v = T((base + ".bn1.running_var").c_str());
        b.conv2_w = T((base + ".conv2.weight").c_str());
        b.conv2_b = T((base + ".conv2.bias").c_str());
        b.bn2_w = T((base + ".bn2.weight").c_str());
        b.bn2_b = T((base + ".bn2.bias").c_str());
        b.bn2_m = T((base + ".bn2.running_mean").c_str());
        b.bn2_v = T((base + ".bn2.running_var").c_str());
        b.sc_w = T((base + ".shortcut.0.weight").c_str());
        b.sc_b = T((base + ".shortcut.0.bias").c_str());
        b.sc_bn_w = T((base + ".shortcut.1.weight").c_str());
        b.sc_bn_b = T((base + ".shortcut.1.bias").c_str());
        b.sc_bn_m = T((base + ".shortcut.1.running_mean").c_str());
        b.sc_bn_v = T((base + ".shortcut.1.running_var").c_str());
    };
    auto bind_dense_layer = [&](cb_campplus_dense_layer& l, const std::string& base) {
        l.nonl1_bn_w = T((base + ".nonlinear1.batchnorm.weight").c_str());
        l.nonl1_bn_b = T((base + ".nonlinear1.batchnorm.bias").c_str());
        l.nonl1_bn_m = T((base + ".nonlinear1.batchnorm.running_mean").c_str());
        l.nonl1_bn_v = T((base + ".nonlinear1.batchnorm.running_var").c_str());
        l.l1_w = T((base + ".linear1.weight").c_str());
        l.l1_b = T((base + ".linear1.bias").c_str());
        l.nonl2_bn_w = T((base + ".nonlinear2.batchnorm.weight").c_str());
        l.nonl2_bn_b = T((base + ".nonlinear2.batchnorm.bias").c_str());
        l.nonl2_bn_m = T((base + ".nonlinear2.batchnorm.running_mean").c_str());
        l.nonl2_bn_v = T((base + ".nonlinear2.batchnorm.running_var").c_str());
        l.cam_ll_w = T((base + ".cam_layer.linear_local.weight").c_str());
        l.cam_l1_w = T((base + ".cam_layer.linear1.weight").c_str());
        l.cam_l1_b = T((base + ".cam_layer.linear1.bias").c_str());
        l.cam_l2_w = T((base + ".cam_layer.linear2.weight").c_str());
        l.cam_l2_b = T((base + ".cam_layer.linear2.bias").c_str());
    };

    // FCM head.
    auto& head = m.head;
    snprintf(k, sizeof(k), "%s.head.conv1.weight", P), head.conv1_w = T(k);
    if (!head.conv1_w)
        return false; // GGUF predates the campplus bake
    snprintf(k, sizeof(k), "%s.head.bn1.weight", P), head.bn1_w = T(k);
    snprintf(k, sizeof(k), "%s.head.bn1.bias", P), head.bn1_b = T(k);
    snprintf(k, sizeof(k), "%s.head.bn1.running_mean", P), head.bn1_m = T(k);
    snprintf(k, sizeof(k), "%s.head.bn1.running_var", P), head.bn1_v = T(k);
    snprintf(k, sizeof(k), "%s.head.conv2.weight", P), head.conv2_w = T(k);
    snprintf(k, sizeof(k), "%s.head.bn2.weight", P), head.bn2_w = T(k);
    snprintf(k, sizeof(k), "%s.head.bn2.bias", P), head.bn2_b = T(k);
    snprintf(k, sizeof(k), "%s.head.bn2.running_mean", P), head.bn2_m = T(k);
    snprintf(k, sizeof(k), "%s.head.bn2.running_var", P), head.bn2_v = T(k);
    head.layer1.assign(2, cb_campplus_resblock{});
    head.layer2.assign(2, cb_campplus_resblock{});
    for (int i = 0; i < 2; i++) {
        snprintf(k, sizeof(k), "%s.head.layer1.%d", P, i), bind_resblock(head.layer1[i], k);
        snprintf(k, sizeof(k), "%s.head.layer2.%d", P, i), bind_resblock(head.layer2[i], k);
    }
    head.layer1[0].stride = 2;
    head.layer2[0].stride = 2;

    // xvector chain.
    snprintf(k, sizeof(k), "%s.xvector.tdnn", P), bind_unit(m.tdnn, k);
    snprintf(k, sizeof(k), "%s.xvector.transit1", P), bind_unit(m.transit1, k);
    snprintf(k, sizeof(k), "%s.xvector.transit2", P), bind_unit(m.transit2, k);
    snprintf(k, sizeof(k), "%s.xvector.transit3", P), bind_unit(m.transit3, k);
    m.out_nl.lin_w = nullptr;
    m.out_nl.lin_b = nullptr;
    snprintf(k, sizeof(k), "%s.xvector.out_nonlinear.batchnorm.weight", P), m.out_nl.bn_w = T(k);
    snprintf(k, sizeof(k), "%s.xvector.out_nonlinear.batchnorm.bias", P), m.out_nl.bn_b = T(k);
    snprintf(k, sizeof(k), "%s.xvector.out_nonlinear.batchnorm.running_mean", P), m.out_nl.bn_m = T(k);
    snprintf(k, sizeof(k), "%s.xvector.out_nonlinear.batchnorm.running_var", P), m.out_nl.bn_v = T(k);
    snprintf(k, sizeof(k), "%s.xvector.dense.linear.weight", P), m.dense.lin_w = T(k);
    snprintf(k, sizeof(k), "%s.xvector.dense.nonlinear.batchnorm.running_mean", P), m.dense.bn_m = T(k);
    snprintf(k, sizeof(k), "%s.xvector.dense.nonlinear.batchnorm.running_var", P), m.dense.bn_v = T(k);

    // Dense blocks: 12 / 24 / 16 layers, dilation 1 / 2 / 2, tdnnd1..N (1-indexed).
    const int nlayers[3] = {12, 24, 16};
    const int dils[3] = {1, 2, 2};
    cb_campplus_dense_block* blocks[3] = {&m.block1, &m.block2, &m.block3};
    for (int bi = 0; bi < 3; bi++) {
        auto& blk = *blocks[bi];
        blk.num_layers = nlayers[bi];
        blk.dilation = dils[bi];
        blk.layers.assign(nlayers[bi], cb_campplus_dense_layer{});
        for (int li = 0; li < nlayers[bi]; li++) {
            snprintf(k, sizeof(k), "%s.xvector.block%d.tdnnd%d", P, bi + 1, li + 1);
            bind_dense_layer(blk.layers[li], k);
        }
    }

    if (!m.tdnn.lin_w || !m.dense.lin_w || m.block1.layers.empty() || !m.block1.layers[0].l1_w) {
        fprintf(stderr, "confucius4: CAMPPlus bind failed (missing core tensors)\n");
        return false;
    }
    ctx->has_campplus = true;
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "confucius4: CAMPPlus style encoder bound from S2A GGUF\n");
    return true;
}

int confucius4_tts_set_vocoder_path(confucius4_tts_context* ctx, const char* path_vocoder) {
    if (!ctx || !path_vocoder)
        return -1;
    if (ctx->vocoder) {
        indextts_voc_free(ctx->vocoder);
        ctx->vocoder = nullptr;
    }
    ctx->vocoder = indextts_voc_init(path_vocoder, ctx->params.n_threads, ctx->params.use_gpu);
    if (!ctx->vocoder) {
        fprintf(stderr, "confucius4: failed to load BigVGAN vocoder '%s'\n", path_vocoder);
        return -1;
    }
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "confucius4: BigVGAN vocoder loaded OK\n");
    return 0;
}

// Bind the ECAPA speaker encoder from the T2S GGUF.  Names follow the upstream
// module layout: blocks.0 is the initial TDNN, blocks.1..3 the SE-Res2Net
// stack, then mfa / asp / fc.  Returns false if any tensor is missing, in which
// case condition_emb stays unavailable and the prefix slot stays zero.
static bool bind_speaker_encoder(confucius4_tts_context* ctx, const core_gguf::tensor_map& tensors) {
    auto& t2s = ctx->t2s;
    auto find = [&](const std::string& n) -> ggml_tensor* {
        auto it = tensors.find(n);
        return it != tensors.end() ? it->second : nullptr;
    };
    auto bind_conv = [&](core_ecapa::tdnn_w& t, const std::string& base) {
        t.w = find(base + ".weight");
        t.b = find(base + ".bias");
        return t.w != nullptr;
    };

    auto& m = t2s.spk_enc;
    bool ok = bind_conv(m.blk0, "speaker_encoder.blocks.0.conv");
    for (int bi = 1; bi <= 3; bi++) {
        auto& blk = m.blk[bi - 1];
        char base[128];
        snprintf(base, sizeof(base), "speaker_encoder.blocks.%d", bi);
        ok &= bind_conv(blk.tdnn1, std::string(base) + ".tdnn1.conv");
        ok &= bind_conv(blk.tdnn2, std::string(base) + ".tdnn2.conv");
        for (int r = 0; r < 7; r++) {
            char rb[192];
            snprintf(rb, sizeof(rb), "%s.res2net_block.blocks.%d.conv", base, r);
            ok &= bind_conv(blk.res2net.blocks[r], rb);
        }
        blk.se.c1w = find(std::string(base) + ".se_block.conv1.weight");
        blk.se.c1b = find(std::string(base) + ".se_block.conv1.bias");
        blk.se.c2w = find(std::string(base) + ".se_block.conv2.weight");
        blk.se.c2b = find(std::string(base) + ".se_block.conv2.bias");
        ok &= blk.se.c1w && blk.se.c2w;
    }
    ok &= bind_conv(m.mfa, "speaker_encoder.mfa.conv");
    ok &= bind_conv(m.asp.tdnn, "speaker_encoder.asp.tdnn.conv");
    m.asp.conv_w = find("speaker_encoder.asp.conv.weight");
    m.asp.conv_b = find("speaker_encoder.asp.conv.bias");
    m.fc_w = find("speaker_encoder.fc.weight");
    m.fc_b = find("speaker_encoder.fc.bias");
    ok &= m.asp.conv_w && m.fc_w;

    m.loaded = ok;
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "confucius4: speaker encoder (ECAPA): %s\n", ok ? "bound" : "MISSING TENSORS");
    return ok;
}

// Run the ECAPA speaker encoder over the (n_frames, feat_dim) Wav2Vec2-BERT
// features to produce the (model_dim,) condition embedding the GPT-2 prefix
// wants.  This is what the reference's speaker_encoder(condition_vector) does.
static bool compute_condition_embedding(confucius4_tts_context* ctx) {
    auto& m = ctx->t2s.spk_enc;
    if (!m.loaded || ctx->speaker_semantic_features.empty() || ctx->speaker_n_frames <= 0)
        return false;

    const int T = ctx->speaker_n_frames;
    const int in_dim = (int)(ctx->speaker_semantic_features.size() / (size_t)T);
    const int enc_dim = (int)m.fc_w->ne[2];

    ggml_init_params ip = {(size_t)6 * 1024 * 1024, nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return false;
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, in_dim, T);
    ggml_set_name(x, "spk_in");
    ggml_set_input(x);

    ggml_tensor* out = core_ecapa::forward(ctx0, m, x);
    ggml_set_name(out, "spk_emb");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    bool ok = ga && ggml_gallocr_alloc_graph(ga, gf);
    if (ok) {
        // features arrive (T, in_dim) row-major, which IS ggml (in_dim, T)
        ggml_backend_tensor_set(x, ctx->speaker_semantic_features.data(), 0,
                                ctx->speaker_semantic_features.size() * sizeof(float));
        ok = ggml_backend_graph_compute(ctx->backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        ctx->condition_embedding.resize(enc_dim);
        ggml_backend_tensor_get(out, ctx->condition_embedding.data(), 0, (size_t)enc_dim * sizeof(float));
        ctx->has_condition_emb = true;
        if (s2a_dump_dir()) {
            char shp[32];
            snprintf(shp, sizeof(shp), "%d", enc_dim);
            s2a_dump_raw("condition_emb", ctx->condition_embedding.data(), (size_t)enc_dim * sizeof(float), shp);
        }
        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "confucius4: condition_emb computed from %d w2v-BERT frames (%d → %d)\n", T, in_dim,
                    enc_dim);
    }
    if (ga)
        ggml_gallocr_free(ga);
    ggml_free(ctx0);
    return ok;
}

int confucius4_tts_set_conditioning(confucius4_tts_context* ctx, const float* condition_embedding, int cond_dim,
                                    const float* style_embedding, int style_dim, const float* prompt_mel,
                                    int n_prompt_frames, int mel_dim) {
    if (!ctx)
        return -1;

    if (condition_embedding && cond_dim > 0) {
        if (cond_dim != ctx->t2s.hp.model_dim) {
            fprintf(stderr, "confucius4: condition embedding dim %d != model_dim %d\n", cond_dim,
                    ctx->t2s.hp.model_dim);
            return -1;
        }
        ctx->condition_embedding.assign(condition_embedding, condition_embedding + cond_dim);
        ctx->has_condition_emb = true;
    }

    if (style_embedding && style_dim > 0) {
        if (style_dim != ctx->s2a.hp.spk_embed_dim) {
            fprintf(stderr, "confucius4: style embedding dim %d != spk_embed_dim %d\n", style_dim,
                    ctx->s2a.hp.spk_embed_dim);
            return -1;
        }
        ctx->speaker_style_embedding.assign(style_embedding, style_embedding + style_dim);
        ctx->has_speaker = true;
    }

    if (prompt_mel && n_prompt_frames > 0 && mel_dim > 0) {
        if (mel_dim != ctx->s2a.hp.estimator_mel_dim) {
            fprintf(stderr, "confucius4: prompt mel dim %d != estimator mel_dim %d\n", mel_dim,
                    ctx->s2a.hp.estimator_mel_dim);
            return -1;
        }
        ctx->prompt_mel.assign(prompt_mel, prompt_mel + (size_t)n_prompt_frames * mel_dim);
        ctx->prompt_n_frames = n_prompt_frames;
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "confucius4: conditioning set: cond_emb=%s style=%s prompt_mel=%d frames\n",
                ctx->has_condition_emb ? "yes" : "no", ctx->has_speaker ? "yes" : "no", ctx->prompt_n_frames);
    return 0;
}

// Testing path: load pre-computed conditioning from raw float32 files in <dir>
// (condition_emb.bin, style_embedding.bin, prompt_mel.bin), mirroring the
// CRISPASR_CONFUCIUS4_TEXT_IDS escape hatch.  Sizes are inferred from the file
// lengths, so a wrong-sized dump is reported rather than silently reshaped.
static void load_conditioning_from_env(confucius4_tts_context* ctx) {
    const char* dir = std::getenv("CRISPASR_CONFUCIUS4_COND_DIR");
    if (!dir || !*dir)
        return;

    auto read_f32 = [&](const char* fname, std::vector<float>& out) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, fname);
        FILE* f = fopen(path, "rb");
        if (!f)
            return false;
        fseek(f, 0, SEEK_END);
        long nbytes = ftell(f);
        fseek(f, 0, SEEK_SET);
        out.resize((size_t)nbytes / sizeof(float));
        size_t got = fread(out.data(), sizeof(float), out.size(), f);
        fclose(f);
        if (got != out.size()) {
            out.clear();
            return false;
        }
        return true;
    };

    std::vector<float> cond_emb, style, pmel;
    read_f32("condition_emb.bin", cond_emb);
    read_f32("style_embedding.bin", style);
    read_f32("prompt_mel.bin", pmel);

    // Prefer computing condition_emb NATIVELY from the raw w2v-BERT features
    // via the ggml ECAPA port (exercises + dumps the port so t2s_parity can
    // check it). CRISPASR_CONFUCIUS4_COND_PYEMB=1 forces the pre-computed
    // Python embedding instead.
    std::vector<float> w2v;
    const bool force_pyemb = std::getenv("CRISPASR_CONFUCIUS4_COND_PYEMB") != nullptr;
    if (!force_pyemb && ctx->t2s.spk_enc.loaded && read_f32("w2v_features.bin", w2v) && w2v.size() >= 1024) {
        ctx->speaker_semantic_features = w2v;
        ctx->speaker_n_frames = (int)(w2v.size() / 1024);
        if (compute_condition_embedding(ctx))
            cond_emb.clear(); // native embedding wins; don't overwrite it below
        else
            fprintf(stderr, "confucius4: native ECAPA condition_emb failed; using condition_emb.bin\n");
    }

    const int mel_dim = ctx->s2a.hp.estimator_mel_dim > 0 ? ctx->s2a.hp.estimator_mel_dim : 80;
    const int n_prompt = pmel.empty() ? 0 : (int)(pmel.size() / mel_dim);

    confucius4_tts_set_conditioning(ctx, cond_emb.empty() ? nullptr : cond_emb.data(), (int)cond_emb.size(),
                                    style.empty() ? nullptr : style.data(), (int)style.size(),
                                    pmel.empty() ? nullptr : pmel.data(), n_prompt, mel_dim);
}

int confucius4_tts_set_w2v_path(confucius4_tts_context* ctx, const char* path_w2v) {
    if (!ctx || !path_w2v)
        return -1;
    if (ctx->w2v_ctx) {
        sidon_free(ctx->w2v_ctx);
        ctx->w2v_ctx = nullptr;
    }
    sidon_context_params sp = sidon_context_default_params();
    sp.n_threads = ctx->params.n_threads;
    sp.verbosity = ctx->params.verbosity;
    sp.use_gpu = ctx->params.use_gpu;
    ctx->w2v_ctx = sidon_init_from_file(path_w2v, sp);
    if (!ctx->w2v_ctx) {
        fprintf(stderr, "confucius4: failed to load w2v-BERT GGUF '%s'\n", path_w2v);
        return -1;
    }
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "confucius4: w2v-BERT encoder loaded (native T2S conditioning)\n");
    return 0;
}

// Native voice conditioning from a reference WAV: CAMPPlus style embedding
// (16 kHz kaldi fbank + CMN, unbiased stats pool — the exact reference
// recipe) and the 22.05 kHz HiFi-GAN-style prompt mel. The T2S
// condition_emb needs w2v-BERT layer-17 features, which are not yet native —
// it stays absent unless CRISPASR_CONFUCIUS4_COND_DIR supplies them, and the
// zero-shot voice then comes from the S2A conditioning alone.
int confucius4_tts_set_voice_path(confucius4_tts_context* ctx, const char* wav_path) {
    if (!ctx || !wav_path)
        return -1;
    if (!ctx->has_campplus) {
        fprintf(stderr, "confucius4: --voice needs the CAMPPlus bake in the S2A GGUF (re-converted model)\n");
        return -1;
    }
    std::vector<float> pcm;
    int sr = 0;
    if (!crispasr::core::read_wav_mono_pcm16(wav_path, pcm, sr) || pcm.empty()) {
        fprintf(stderr, "confucius4: failed to read reference WAV %s\n", wav_path);
        return -1;
    }

    std::vector<float> pcm16 =
        sr == 16000 ? pcm : core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 16000);
    auto style = chatterbox_campplus::embed_speaker(ctx->campplus_model, ctx->campplus_rt, pcm16.data(),
                                                    (int)pcm16.size(), /*stats_var_floor=*/0.0f);
    const int want_spk = ctx->s2a.hp.spk_embed_dim > 0 ? ctx->s2a.hp.spk_embed_dim : 192;
    if ((int)style.size() != want_spk) {
        fprintf(stderr, "confucius4: CAMPPlus produced %zu dims, expected %d\n", style.size(), want_spk);
        return -1;
    }
    ctx->speaker_style_embedding = std::move(style);
    ctx->has_speaker = true;

    const int tgt_sr = ctx->t2s.hp.sample_rate > 0 ? ctx->t2s.hp.sample_rate : 22050;
    std::vector<float> pcm_tgt =
        sr == tgt_sr ? pcm : core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, tgt_sr);
    const int mel_dim = ctx->s2a.hp.estimator_mel_dim > 0 ? ctx->s2a.hp.estimator_mel_dim : 80;
    int T_mel = 0;
    auto pmel = chatterbox_campplus::compute_prompt_feat(pcm_tgt.data(), (int)pcm_tgt.size(), tgt_sr,
                                                         /*n_fft=*/1024, /*hop=*/256, /*win=*/1024, mel_dim,
                                                         /*fmin=*/0.0f, /*fmax=*/tgt_sr / 2.0f,
                                                         /*max_samples=*/0, T_mel);
    if (pmel.empty() || T_mel <= 0) {
        fprintf(stderr, "confucius4: prompt mel computation failed\n");
        return -1;
    }
    ctx->prompt_mel = std::move(pmel);
    ctx->prompt_n_frames = T_mel;

    // T2S condition_emb: w2v-BERT layer-17 hidden states, z-normalised with
    // the stats baked in the T2S GGUF, through the native ECAPA. Runs when the
    // encoder-only w2v GGUF is loaded (confucius4_tts_set_w2v_path).
    if (ctx->w2v_ctx && !ctx->w2v_mean.empty() && ctx->w2v_mean.size() == ctx->w2v_var.size()) {
        int T_w2v = 0;
        std::vector<float> feats = sidon_extract_hidden(ctx->w2v_ctx, pcm16.data(), (int)pcm16.size(), &T_w2v);
        const int fdim = (int)ctx->w2v_mean.size();
        if (!feats.empty() && T_w2v > 0 && (int)feats.size() == T_w2v * fdim) {
            for (int t = 0; t < T_w2v; t++)
                for (int d = 0; d < fdim; d++) {
                    float& v = feats[(size_t)t * fdim + d];
                    v = (v - ctx->w2v_mean[d]) / std::sqrt(ctx->w2v_var[d]);
                }
            ctx->speaker_semantic_features = std::move(feats);
            ctx->speaker_n_frames = T_w2v;
            compute_condition_embedding(ctx);
        } else {
            fprintf(stderr, "confucius4: w2v-BERT feature extraction failed (%d frames)\n", T_w2v);
        }
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "confucius4: voice set: style=%d-d, prompt_mel=%d frames @%d Hz%s\n", want_spk, T_mel, tgt_sr,
                ctx->has_condition_emb ? "" : " (no T2S condition_emb — supply w2v-BERT via set_w2v_path/COND_DIR)");
    return 0;
}

int confucius4_tts_set_speaker(confucius4_tts_context* ctx, const float* semantic_features, int n_frames,
                               const float* style_embedding) {
    if (!ctx || !semantic_features || !style_embedding || n_frames <= 0)
        return -1;

    ctx->speaker_semantic_features.assign(semantic_features, semantic_features + n_frames * 1024);
    ctx->speaker_n_frames = n_frames;
    ctx->speaker_style_embedding.assign(style_embedding, style_embedding + 192);
    ctx->has_speaker = true;

    // Run the ECAPA encoder now: the weights are in the T2S GGUF, so the only
    // external input needed is the w2v-BERT feature block just supplied.
    compute_condition_embedding(ctx);
    return 0;
}

// ---------------------------------------------------------------------------
// KV cache helpers
// ---------------------------------------------------------------------------

static bool kv_init(confucius4_kv_cache& kv, const confucius4_t2s_hparams& hp, int max_seq, ggml_backend_t backend) {
    kv.max_seq_len = max_seq;
    size_t ctx_size = 2 * ggml_tensor_overhead() + 64;
    ggml_init_params ip = {ctx_size, nullptr, true};
    kv.ctx = ggml_init(ip);
    if (!kv.ctx)
        return false;
    kv.k = ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F16, hp.head_dim(), max_seq, hp.num_heads, hp.num_layers);
    kv.v = ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F16, hp.head_dim(), max_seq, hp.num_heads, hp.num_layers);
    ggml_set_name(kv.k, "kv_k");
    ggml_set_name(kv.v, "kv_v");
    kv.buf = ggml_backend_alloc_ctx_tensors(kv.ctx, backend);
    if (!kv.buf)
        return false;
    ggml_backend_tensor_memset(kv.k, 0, 0, ggml_nbytes(kv.k));
    ggml_backend_tensor_memset(kv.v, 0, 0, ggml_nbytes(kv.v));
    return true;
}

static void kv_free(confucius4_kv_cache& kv) {
    if (kv.step_galloc)
        ggml_gallocr_free(kv.step_galloc);
    if (kv.step_ctx)
        ggml_free(kv.step_ctx);
    if (kv.buf)
        ggml_backend_buffer_free(kv.buf);
    if (kv.ctx)
        ggml_free(kv.ctx);
    kv = {};
}

// Copy the first n_cols cached positions from src into dst (identical shapes
// [hd, max_seq, heads, layers]). Beam search reorders hypotheses each step;
// the surviving beams inherit their parent's cache prefix.
static void kv_copy_prefix(confucius4_kv_cache& dst, const confucius4_kv_cache& src, int n_cols) {
    if (n_cols <= 0)
        return;
    for (int which = 0; which < 2; which++) {
        ggml_tensor* s = which ? src.v : src.k;
        ggml_tensor* d = which ? dst.v : dst.k;
        const size_t nb2 = s->nb[2], nb3 = s->nb[3];
        const size_t chunk = (size_t)n_cols * s->nb[1];
        const int heads = (int)s->ne[2], layers = (int)s->ne[3];
        std::vector<uint8_t> buf(chunk);
        for (int l = 0; l < layers; l++) {
            for (int h = 0; h < heads; h++) {
                const size_t off = (size_t)l * nb3 + (size_t)h * nb2;
                ggml_backend_tensor_get(s, buf.data(), off, chunk);
                ggml_backend_tensor_set(d, buf.data(), off, chunk);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Top-p sampling
// ---------------------------------------------------------------------------

static int sample_top_p(const float* logits, int n_vocab, float temperature, float top_p, int top_k,
                        std::mt19937& rng) {
    std::vector<std::pair<float, int>> candidates(n_vocab);
    for (int i = 0; i < n_vocab; i++)
        candidates[i] = {logits[i] / temperature, i};

    // Top-k filter
    if (top_k > 0 && top_k < n_vocab) {
        std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        candidates.resize(top_k);
    } else {
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // Softmax
    float max_val = candidates[0].first;
    float sum = 0.0f;
    for (auto& c : candidates) {
        c.first = expf(c.first - max_val);
        sum += c.first;
    }
    for (auto& c : candidates)
        c.first /= sum;

    // Top-p nucleus filter
    float cumsum = 0.0f;
    int last = (int)candidates.size();
    for (int i = 0; i < (int)candidates.size(); i++) {
        cumsum += candidates[i].first;
        if (cumsum >= top_p) {
            last = i + 1;
            break;
        }
    }
    candidates.resize(last);

    // Re-normalise and sample
    sum = 0.0f;
    for (auto& c : candidates)
        sum += c.first;
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    cumsum = 0.0f;
    for (auto& c : candidates) {
        cumsum += c.first;
        if (cumsum >= r)
            return c.second;
    }
    return candidates.back().second;
}

// ---------------------------------------------------------------------------
// T2S decode: generate semantic codes from text token IDs
// ---------------------------------------------------------------------------

// Build and run the prefix embedding graph:
//   text_ids → Embedding(32k,4096) → Linear(4096,4096) → SiLU → Linear(4096,1280) + pos_emb
//   condition: zero vector (1, 1280) — speaker encoder not yet wired
//   BOS: semantic_embedding[start_semantic_token]
// Returns the concatenated prefix (prefix_len, model_dim) as float32.
static std::vector<float> build_prefix_embedding(confucius4_tts_context* ctx, const std::vector<int32_t>& text_ids) {
    const auto& m = ctx->t2s;
    const auto& hp = m.hp;
    const int D = hp.model_dim;
    const int T_text = (int)text_ids.size();
    const int prefix_len = 1 + T_text + 1; // condition(1) + text(T) + BOS(1)

    // Helper: run a small embedding graph via gallocr
    auto run_embed_graph = [&](auto build_fn) -> bool {
        const int nt = 32;
        size_t cs = ggml_tensor_overhead() * nt + ggml_graph_overhead_custom(64, false);
        ggml_init_params ip2 = {cs, nullptr, true};
        ggml_context* c = ggml_init(ip2);
        if (!c)
            return false;
        ggml_cgraph* g = ggml_new_graph_custom(c, 64, false);
        build_fn(c, g);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        bool ok = ggml_gallocr_alloc_graph(ga, g);
        if (!ok) {
            ggml_gallocr_free(ga);
            ggml_free(c);
            return false;
        }
        return true; // caller must set inputs, compute, read outputs, then free ga+c
    };
    (void)run_embed_graph; // suppress unused warning — used below

    std::vector<float> result(prefix_len * D, 0.0f);

    // ── Sub-graph 1: text projector MLP ──
    {
        const int nt = 32;
        size_t cs = ggml_tensor_overhead() * nt + ggml_graph_overhead_custom(64, false);
        ggml_init_params ip2 = {cs, nullptr, true};
        ggml_context* c = ggml_init(ip2);
        ggml_cgraph* g = ggml_new_graph_custom(c, 64, false);

        ggml_tensor* ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, T_text);
        ggml_set_name(ids, "ids");
        ggml_set_input(ids);

        ggml_tensor* emb = ggml_get_rows(c, m.text_embed_w, ids);
        ggml_tensor* h = ggml_mul_mat(c, m.text_proj_fc1_w, emb);
        h = ggml_add(c, h, m.text_proj_fc1_b);
        h = ggml_silu(c, h);
        h = ggml_mul_mat(c, m.text_proj_fc2_w, h);
        h = ggml_add(c, h, m.text_proj_fc2_b);

        ggml_tensor* pos_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, T_text);
        ggml_set_name(pos_ids, "pos");
        ggml_set_input(pos_ids);

        ggml_tensor* pos_emb = ggml_get_rows(c, m.text_pos_embed_w, pos_ids);
        ggml_tensor* out = ggml_add(c, h, pos_emb);
        ggml_set_name(out, "text_emb");
        ggml_set_output(out);
        ggml_build_forward_expand(g, out);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ggml_gallocr_alloc_graph(ga, g)) {
            fprintf(stderr, "confucius4: text projector graph alloc failed\n");
            ggml_gallocr_free(ga);
            ggml_free(c);
            return {};
        }

        ggml_backend_tensor_set(ids, text_ids.data(), 0, T_text * sizeof(int32_t));
        std::vector<int32_t> pos_data(T_text);
        for (int i = 0; i < T_text; i++)
            pos_data[i] = i;
        ggml_backend_tensor_set(pos_ids, pos_data.data(), 0, T_text * sizeof(int32_t));

        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "confucius4: text projector: computing %d tokens...\n", T_text);

        ggml_backend_graph_compute(ctx->backend, g);
        ggml_backend_tensor_get(out, result.data() + D, 0, (size_t)T_text * D * sizeof(float));

        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "confucius4: text projector: OK\n");

        ggml_gallocr_free(ga);
        ggml_free(c);
    }

    // ── Sub-graph 2: BOS semantic embedding ──
    {
        const int nt = 16;
        size_t cs = ggml_tensor_overhead() * nt + ggml_graph_overhead_custom(32, false);
        ggml_init_params ip2 = {cs, nullptr, true};
        ggml_context* c = ggml_init(ip2);
        ggml_cgraph* g = ggml_new_graph_custom(c, 32, false);

        ggml_tensor* bos_id = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
        ggml_set_name(bos_id, "bos");
        ggml_set_input(bos_id);
        ggml_tensor* bos_emb = ggml_get_rows(c, m.semantic_embed_w, bos_id);

        ggml_tensor* sem_pos_id = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
        ggml_set_name(sem_pos_id, "spos");
        ggml_set_input(sem_pos_id);
        ggml_tensor* sem_pos = ggml_get_rows(c, m.sem_pos_embed_w, sem_pos_id);

        ggml_tensor* out = ggml_add(c, bos_emb, sem_pos);
        ggml_set_name(out, "bos_emb");
        ggml_set_output(out);
        ggml_build_forward_expand(g, out);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ggml_gallocr_alloc_graph(ga, g)) {
            fprintf(stderr, "confucius4: BOS graph alloc failed\n");
            ggml_gallocr_free(ga);
            ggml_free(c);
            return {};
        }

        int32_t bos_val = hp.start_semantic_token;
        ggml_backend_tensor_set(bos_id, &bos_val, 0, sizeof(int32_t));
        int32_t sem_pos0_val = 0;
        ggml_backend_tensor_set(sem_pos_id, &sem_pos0_val, 0, sizeof(int32_t));

        ggml_backend_graph_compute(ctx->backend, g);
        ggml_backend_tensor_get(out, result.data() + (1 + T_text) * D, 0, (size_t)D * sizeof(float));

        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "confucius4: BOS embed: OK\n");

        ggml_gallocr_free(ga);
        ggml_free(c);
    }

    // Slot 0: condition_emb.  In the reference this is
    // speaker_encoder(w2v_bert_layer17), a (1, model_dim) vector prepended to
    // the prefix.  Without a reference wav it stays zero -- note that is NOT
    // what the reference computes for "no speaker" (the ECAPA-TDNN has biases),
    // it simply has no such mode.
    if (ctx->has_condition_emb && (int)ctx->condition_embedding.size() == D)
        std::memcpy(result.data(), ctx->condition_embedding.data(), (size_t)D * sizeof(float));

    return result;
}

// Build and run a single-token semantic embedding graph for decode step.
// Returns (D,) float32.
static std::vector<float> embed_semantic_token(confucius4_tts_context* ctx, int32_t token_id, int sem_pos) {
    const auto& m = ctx->t2s;
    const int D = m.hp.model_dim;

    // Bounds check
    if (token_id < 0 || token_id >= (int32_t)m.semantic_embed_w->ne[1]) {
        fprintf(stderr, "confucius4: embed_semantic_token: token_id=%d OUT OF RANGE [0,%lld)\n", token_id,
                (long long)m.semantic_embed_w->ne[1]);
        return {};
    }
    if (sem_pos < 0 || sem_pos >= (int)m.sem_pos_embed_w->ne[1]) {
        fprintf(stderr, "confucius4: embed_semantic_token: sem_pos=%d OUT OF RANGE [0,%lld)\n", sem_pos,
                (long long)m.sem_pos_embed_w->ne[1]);
        return {};
    }

    // Host-table fast path: identical dequant as ggml_get_rows, no graph.
    if (!ctx->sem_embed_tab.empty() && !ctx->sem_pos_tab.empty()) {
        std::vector<float> out(D);
        const float* e = ctx->sem_embed_tab.data() + (size_t)token_id * D;
        const float* p = ctx->sem_pos_tab.data() + (size_t)sem_pos * D;
        for (int i = 0; i < D; i++)
            out[i] = e[i] + p[i];
        return out;
    }

    size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(64, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return {};

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor* tok = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(tok, "sem_tok");
    ggml_set_input(tok);

    ggml_tensor* emb = ggml_get_rows(ctx0, m.semantic_embed_w, tok);

    ggml_tensor* pos_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(pos_id, "sem_pos");
    ggml_set_input(pos_id);

    ggml_tensor* pos = ggml_get_rows(ctx0, m.sem_pos_embed_w, pos_id);
    ggml_tensor* out = ggml_add(ctx0, emb, pos);
    ggml_set_name(out, "sem_emb_out");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return {};
    }

    ggml_backend_tensor_set(tok, &token_id, 0, sizeof(int32_t));
    int32_t pos_val = sem_pos;
    ggml_backend_tensor_set(pos_id, &pos_val, 0, sizeof(int32_t));

    ggml_backend_graph_compute(ctx->backend, gf);

    std::vector<float> result(D);
    ggml_backend_tensor_get(out, result.data(), 0, D * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return result;
}

// Persistent single-token decode step (see confucius4_kv_cache doc). Builds
// the fixed-topology graph lazily on first use for this cache, then reuses it
// for every subsequent step: only the inputs change. Output-equivalence vs the
// rebuild-per-step path was byte-compared on the Kaggle roundtrip before this
// became reachable (CRISPASR_CONFUCIUS4_PERSIST gate).
static std::vector<float> run_gpt2_step_persistent(confucius4_tts_context* ctx, confucius4_kv_cache& kv,
                                                   const float* input_emb, int n_past, std::vector<float>* out_hidden) {
    const auto& hp = ctx->t2s.hp;
    const int D = hp.model_dim;
    const int V = hp.semantic_vocab_size;
    const int Lk = kv.max_seq_len;

    if (!kv.step_gf) {
        const int n_tensors = hp.num_layers * 50 + 64;
        size_t ctx_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(8192, false);
        ggml_init_params ip = {ctx_size, nullptr, true};
        kv.step_ctx = ggml_init(ip);
        if (!kv.step_ctx)
            return {};
        kv.step_gf = ggml_new_graph_custom(kv.step_ctx, 8192, false);

        kv.step_x = ggml_new_tensor_2d(kv.step_ctx, GGML_TYPE_F32, D, 1);
        ggml_set_name(kv.step_x, "gpt2_input");
        ggml_set_input(kv.step_x);
        kv.step_mask = ggml_new_tensor_2d(kv.step_ctx, GGML_TYPE_F16, Lk, 1);
        ggml_set_name(kv.step_mask, "step_mask");
        ggml_set_input(kv.step_mask);
        kv.step_idx = ggml_new_tensor_1d(kv.step_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(kv.step_idx, "step_idx");
        ggml_set_input(kv.step_idx);

        ggml_tensor* logits = gpt2_forward(ctx, kv.step_ctx, kv.step_gf, kv.step_x, kv.k, kv.v,
                                           /*n_past=*/0, kv.step_mask, /*fixed_kv_len=*/Lk, kv.step_idx);
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);
        ggml_build_forward_expand(kv.step_gf, logits);
        kv.step_logits = logits;
        kv.step_hidden = ggml_graph_get_tensor(kv.step_gf, "lm_hidden");

        kv.step_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ggml_gallocr_alloc_graph(kv.step_galloc, kv.step_gf)) {
            fprintf(stderr, "confucius4: persistent step graph alloc failed\n");
            ggml_gallocr_free(kv.step_galloc);
            ggml_free(kv.step_ctx);
            kv.step_galloc = nullptr;
            kv.step_ctx = nullptr;
            kv.step_gf = nullptr;
            return {};
        }
    }

    // Re-set EVERY input each step (gallocr input-buffer reuse).
    ggml_backend_tensor_set(kv.step_x, input_emb, 0, (size_t)D * sizeof(float));
    std::vector<ggml_fp16_t> mask((size_t)Lk);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int j = 0; j < Lk; j++)
        mask[(size_t)j] = j <= n_past ? zero : ninf;
    ggml_backend_tensor_set(kv.step_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    int32_t idx = n_past;
    ggml_backend_tensor_set(kv.step_idx, &idx, 0, sizeof(int32_t));

    if (ggml_backend_graph_compute(ctx->backend, kv.step_gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "confucius4: persistent step compute failed\n");
        return {};
    }

    std::vector<float> out_logits(V);
    ggml_backend_tensor_get(kv.step_logits, out_logits.data(), 0, (size_t)V * sizeof(float));
    if (out_hidden && kv.step_hidden) {
        size_t old = out_hidden->size();
        out_hidden->resize(old + D);
        ggml_backend_tensor_get(kv.step_hidden, out_hidden->data() + old, 0, (size_t)D * sizeof(float));
    }
    return out_logits;
}

// Run a single GPT-2 forward step via ggml_backend_sched.
// Input: (D, T) float embeddings. Output: last-token logits (semantic_vocab_size,).
// Run one GPT-2 step. Returns logits for the last token.
// If `out_hidden` is non-null, appends the hidden state (D floats) for the last token.
// If `kvc` is non-null, that KV cache is used instead of ctx->kv (beam decode).
// If `out_hidden_all` is non-null, it receives ALL T hidden rows (T*D floats).
static std::vector<float> run_gpt2_step(confucius4_tts_context* ctx, const float* input_emb, int T, int n_past,
                                        std::vector<float>* out_hidden = nullptr, confucius4_kv_cache* kvc = nullptr,
                                        std::vector<float>* out_hidden_all = nullptr) {
    const auto& hp = ctx->t2s.hp;
    const int D = hp.model_dim;

    // Opt-in persistent-graph decode (A/B: byte-identical output, no per-step
    // graph build/alloc). Applies to T==1 steps only; prefill and the
    // teacher-forced latent pass keep the one-off graphs.
    static const bool persist = [] {
        const char* e = std::getenv("CRISPASR_CONFUCIUS4_PERSIST");
        return e && *e && *e != '0';
    }();
    if (persist && T == 1 && !out_hidden_all) {
        confucius4_kv_cache& kv_use0 = kvc ? *kvc : ctx->kv;
        return run_gpt2_step_persistent(ctx, kv_use0, input_emb, n_past, out_hidden);
    }

    // kv_self_attn creates ~40 intermediates per layer (QKV split, views, permutes,
    // cache writes, softmax, output permute, reshape) + FFN adds ~10.
    const int n_tensors = hp.num_layers * 50 + 64;
    size_t ctx_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(8192, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return {};

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Input embedding tensor
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "gpt2_input");
    ggml_set_input(x);

    // Causal mask for multi-token (prefill) steps. Without it, soft_max_ext
    // runs FULL attention over the whole prefix — every position saw the
    // future, poisoning the KV cache for the entire decode (HARD RULE #5;
    // prefill_logits sat at cos≈0.2 vs the reference until this landed).
    // T==1 decode steps attend only the n_past cached positions → no mask.
    const int Lk = n_past + T;
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    // Forward pass
    confucius4_kv_cache& kv_use = kvc ? *kvc : ctx->kv;
    ggml_tensor* logits = gpt2_forward(ctx, ctx0, gf, x, kv_use.k, kv_use.v, n_past, causal_mask);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    // Gate: CRISPASR_CONFUCIUS4_GALLOCR=1 uses gallocr (avoids sched index
    // corruption seen with quantized weight tensors); gallocr is the validated
    // working path. CRISPASR_CONFUCIUS4_SCHED=1 restores the old sched path.
    const bool use_gallocr = (std::getenv("CRISPASR_CONFUCIUS4_SCHED") == nullptr);

    const int V = hp.semantic_vocab_size;
    std::vector<float> out_logits(V);

    if (use_gallocr) {
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            fprintf(stderr, "confucius4: GPT-2 gallocr alloc failed\n");
            ggml_gallocr_free(galloc);
            ggml_free(ctx0);
            return {};
        }
        ggml_backend_tensor_set(x, input_emb, 0, (size_t)D * T * sizeof(float));
        if (causal_mask) {
            std::vector<ggml_fp16_t> mask_data((size_t)Lk * T, ggml_fp32_to_fp16(-INFINITY));
            for (int q = 0; q < T; q++)
                for (int k = 0; k < n_past + q + 1; k++)
                    mask_data[(size_t)q * Lk + k] = ggml_fp32_to_fp16(0.0f);
            ggml_backend_tensor_set(causal_mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
        }
        ggml_backend_graph_compute(ctx->backend, gf);
        size_t offset = (T > 1) ? (size_t)(T - 1) * V * sizeof(float) : 0;
        ggml_backend_tensor_get(logits, out_logits.data(), offset, V * sizeof(float));
        if (out_hidden) {
            ggml_tensor* h = ggml_graph_get_tensor(gf, "lm_hidden");
            if (h) {
                size_t h_off = (T > 1) ? (size_t)(T - 1) * D * sizeof(float) : 0;
                size_t old = out_hidden->size();
                out_hidden->resize(old + D);
                ggml_backend_tensor_get(h, out_hidden->data() + old, h_off, D * sizeof(float));
            }
        }
        if (out_hidden_all) {
            ggml_tensor* h = ggml_graph_get_tensor(gf, "lm_hidden");
            if (h) {
                out_hidden_all->resize((size_t)T * D);
                ggml_backend_tensor_get(h, out_hidden_all->data(), 0, (size_t)T * D * sizeof(float));
            }
        }
        ggml_gallocr_free(galloc);
    } else {
        ggml_backend_sched_t sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, n_tensors, false, false);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "confucius4: GPT-2 sched alloc failed\n");
            ggml_backend_sched_free(sched);
            ggml_free(ctx0);
            return {};
        }
        ggml_backend_tensor_set(x, input_emb, 0, (size_t)D * T * sizeof(float));
        if (causal_mask) {
            std::vector<ggml_fp16_t> mask_data((size_t)Lk * T, ggml_fp32_to_fp16(-INFINITY));
            for (int q = 0; q < T; q++)
                for (int k = 0; k < n_past + q + 1; k++)
                    mask_data[(size_t)q * Lk + k] = ggml_fp32_to_fp16(0.0f);
            ggml_backend_tensor_set(causal_mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
        }
        ggml_backend_sched_graph_compute(sched, gf);
        size_t offset = (T > 1) ? (size_t)(T - 1) * V * sizeof(float) : 0;
        ggml_backend_tensor_get(logits, out_logits.data(), offset, V * sizeof(float));
        if (out_hidden) {
            ggml_tensor* h = ggml_graph_get_tensor(gf, "lm_hidden");
            if (h) {
                size_t h_off = (T > 1) ? (size_t)(T - 1) * D * sizeof(float) : 0;
                size_t old = out_hidden->size();
                out_hidden->resize(old + D);
                ggml_backend_tensor_get(h, out_hidden->data() + old, h_off, D * sizeof(float));
            }
        }
        if (out_hidden_all) {
            ggml_tensor* h = ggml_graph_get_tensor(gf, "lm_hidden");
            if (h) {
                out_hidden_all->resize((size_t)T * D);
                ggml_backend_tensor_get(h, out_hidden_all->data(), 0, (size_t)T * D * sizeof(float));
            }
        }
        ggml_backend_sched_free(sched);
    }

    ggml_free(ctx0);
    return out_logits;
}

// ---------------------------------------------------------------------------
// T2S beam-sample decode -- mirrors transformers 4.52.4 `_beam_search` with
// do_sample=True, the exact path the reference hits with num_beams=3:
//   log_softmax(logits) -> processors+warpers ON LOG-PROBS in order
//   [repetition_penalty, temperature, top_k, top_p] -> + running beam score ->
//   joint softmax over (n_beams * vocab) -> multinomial 2*n_beams WITHOUT
//   replacement -> first n_beams candidates may finish on EOS (score
//   length-penalized by generated length, length_penalty=1.0) -> running
//   beams = top n_beams continuations by score; early_stopping=True stops
//   once n_beams hypotheses have finished.
// ---------------------------------------------------------------------------

static std::vector<int32_t> t2s_decode_beam(confucius4_tts_context* ctx, const std::vector<int32_t>& text_token_ids,
                                            int n_beams, std::vector<float>* out_lm_latent) {
    const auto& hp = ctx->t2s.hp;
    const int T_text = (int)text_token_ids.size();
    const int prefix_len = 1 + T_text + 1; // condition(1) + text(T) + BOS(1)
    // Reference max_length=1520 is the TOTAL sequence length passed to HF
    // generate() (prefix + BOS + generated), not a new-token count.
    const int max_new = ctx->params.max_semantic_tokens > 0 ? ctx->params.max_semantic_tokens
                                                            : std::max(1, hp.max_semantic_seq_lens - prefix_len);
    const int max_seq = prefix_len + max_new + 1;
    const int vb = ctx->params.verbosity;
    const int V = hp.semantic_vocab_size;
    const int eos = hp.stop_semantic_token;
    const int n_cand = 2 * n_beams;

    float rep_pen = ctx->params.repetition_penalty;
    if (const char* e = std::getenv("CRISPASR_CONFUCIUS4_REP_PEN"))
        rep_pen = (float)atof(e);
    const float temp = ctx->params.temperature > 0.0f ? ctx->params.temperature : 1.0f;
    const float top_p = ctx->params.top_p;
    const int top_k = ctx->params.top_k;

    if (vb >= 1)
        fprintf(stderr, "confucius4: T2S beam decode: text_len=%d, prefix_len=%d, beams=%d, max_new=%d\n", T_text,
                prefix_len, n_beams, max_new);

    // Two banks of per-beam KV caches (beam reorder copies parent -> child).
    std::vector<confucius4_kv_cache> bank[2];
    bank[0].resize(n_beams);
    bank[1].resize(n_beams);
    for (int i = 0; i < n_beams; i++) {
        if (!kv_init(bank[0][i], hp, max_seq, ctx->backend) || !kv_init(bank[1][i], hp, max_seq, ctx->backend)) {
            fprintf(stderr, "confucius4: beam KV cache allocation failed\n");
            for (auto& b : bank[0])
                kv_free(b);
            for (auto& b : bank[1])
                kv_free(b);
            return {};
        }
    }
    int cur_bank = 0;

    std::mt19937 rng(ctx->params.seed ? ctx->params.seed : 42);

    // Prefill beam 0, then clone the prefix into every beam of the bank.
    std::vector<float> prefix_emb = build_prefix_embedding(ctx, text_token_ids);
    if (prefix_emb.empty()) {
        for (int w = 0; w < 2; w++)
            for (auto& b : bank[w])
                kv_free(b);
        return {};
    }
    std::vector<std::vector<float>> beam_logits(n_beams);
    beam_logits[0] = run_gpt2_step(ctx, prefix_emb.data(), prefix_len, 0, nullptr, &bank[0][0]);
    if (beam_logits[0].empty()) {
        for (int w = 0; w < 2; w++)
            for (auto& b : bank[w])
                kv_free(b);
        return {};
    }
    for (int i = 1; i < n_beams; i++) {
        kv_copy_prefix(bank[0][i], bank[0][0], prefix_len);
        beam_logits[i] = beam_logits[0];
    }

    if (s2a_dump_dir()) {
        char shp[64];
        snprintf(shp, sizeof(shp), "%d", (int)text_token_ids.size());
        s2a_dump_raw("text_ids_i32", text_token_ids.data(), text_token_ids.size() * sizeof(int32_t), shp);
        snprintf(shp, sizeof(shp), "%d", (int)beam_logits[0].size());
        s2a_dump_raw("prefill_logits", beam_logits[0].data(), beam_logits[0].size() * sizeof(float), shp);
        snprintf(shp, sizeof(shp), "%d,%d", prefix_len, hp.model_dim);
        s2a_dump_raw("prefix_emb", prefix_emb.data(), prefix_emb.size() * sizeof(float), shp);
    }

    struct Beam {
        std::vector<int32_t> codes;
        double score = 0.0;
        std::vector<uint8_t> seen; // per-unique-token repetition set (BOS pre-seeded)
    };
    std::vector<Beam> beams(n_beams);
    for (int i = 0; i < n_beams; i++) {
        beams[i].score = i == 0 ? 0.0 : -1e9;
        beams[i].seen.assign(V, 0);
        beams[i].seen[hp.start_semantic_token] = 1;
    }

    struct Finished {
        std::vector<int32_t> codes;
        double score = -1e18;
    };
    std::vector<Finished> finished; // kept sorted desc, size <= n_beams

    std::vector<double> acc((size_t)n_beams * V);
    std::vector<double> warped(V);
    int n_past = prefix_len;

    for (int step = 0; step < max_new; step++) {
        const bool full = (int)finished.size() >= n_beams; // early_stopping guard
        for (int b = 0; b < n_beams; b++) {
            // log_softmax in double
            const auto& lg = beam_logits[b];
            double mx = -1e30;
            for (int v = 0; v < V; v++)
                mx = std::max(mx, (double)lg[v]);
            double se = 0.0;
            for (int v = 0; v < V; v++)
                se += std::exp((double)lg[v] - mx);
            const double lse = mx + std::log(se);
            for (int v = 0; v < V; v++)
                warped[v] = (double)lg[v] - lse;
            // repetition penalty ON LOG-PROBS (HF applies the processor list
            // to log_softmax output in _beam_search; log-probs are <= 0, so
            // penalized tokens multiply)
            if (rep_pen != 1.0f) {
                for (int v = 0; v < V; v++)
                    if (beams[b].seen[v])
                        warped[v] = warped[v] < 0 ? warped[v] * rep_pen : warped[v] / rep_pen;
            }
            // temperature
            for (int v = 0; v < V; v++)
                warped[v] /= temp;
            // top-k: keep the k largest, others -inf
            if (top_k > 0 && top_k < V) {
                std::vector<double> tmp(warped.begin(), warped.end());
                std::nth_element(tmp.begin(), tmp.begin() + (top_k - 1), tmp.end(), std::greater<double>());
                const double thresh = tmp[top_k - 1];
                for (int v = 0; v < V; v++)
                    if (warped[v] < thresh)
                        warped[v] = -INFINITY;
            }
            // top-p on softmax of the current scores (keep the smallest
            // descending prefix with mass >= top_p, crossing token included)
            if (top_p > 0.0f && top_p < 1.0f) {
                std::vector<std::pair<double, int>> order;
                order.reserve(top_k > 0 ? top_k : V);
                for (int v = 0; v < V; v++)
                    if (warped[v] != -INFINITY)
                        order.push_back({warped[v], v});
                std::sort(order.begin(), order.end(), std::greater<std::pair<double, int>>());
                double m2 = order.empty() ? 0.0 : order[0].first, s2 = 0.0;
                for (auto& pr : order)
                    s2 += std::exp(pr.first - m2);
                double cum = 0.0;
                size_t keep = order.size();
                for (size_t oi = 0; oi < order.size(); oi++) {
                    cum += std::exp(order[oi].first - m2) / s2;
                    if (cum >= (double)top_p) {
                        keep = oi + 1;
                        break;
                    }
                }
                for (size_t oi = keep; oi < order.size(); oi++)
                    warped[order[oi].second] = -INFINITY;
            }
            for (int v = 0; v < V; v++)
                acc[(size_t)b * V + v] = warped[v] + beams[b].score;
        }

        // joint softmax over n_beams*V, multinomial n_cand WITHOUT replacement
        double mx = -1e30;
        for (auto d : acc)
            if (d > mx)
                mx = d;
        std::vector<double> probs(acc.size());
        double se = 0.0;
        for (size_t i = 0; i < acc.size(); i++) {
            probs[i] = acc[i] == -INFINITY || acc[i] < mx - 700.0 ? 0.0 : std::exp(acc[i] - mx);
            se += probs[i];
        }
        std::vector<size_t> cand_idx;
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (int j = 0; j < n_cand && se > 0.0; j++) {
            double r = uni(rng) * se, cum = 0.0;
            size_t pick = 0;
            bool found = false;
            for (size_t i = 0; i < probs.size(); i++) {
                cum += probs[i];
                if (cum >= r && probs[i] > 0.0) {
                    pick = i;
                    found = true;
                    break;
                }
            }
            if (!found) { // numeric tail: last nonzero
                for (size_t i = probs.size(); i-- > 0;)
                    if (probs[i] > 0.0) {
                        pick = i;
                        found = true;
                        break;
                    }
            }
            if (!found)
                break;
            cand_idx.push_back(pick);
            se -= probs[pick];
            probs[pick] = 0.0;
        }
        if (cand_idx.empty())
            break;

        // finished-hypothesis update: only the first n_beams sampled
        // candidates are eligible (HF top_num_beam_mask)
        struct Cand {
            int parent, token;
            double score, cont_score;
        };
        std::vector<Cand> cands;
        for (size_t j = 0; j < cand_idx.size(); j++) {
            Cand c;
            c.parent = (int)(cand_idx[j] / V);
            c.token = (int)(cand_idx[j] % V);
            c.score = acc[cand_idx[j]];
            c.cont_score = c.token == eos ? c.score - 1e9 : c.score;
            cands.push_back(c);
            if (c.token == eos && (int)j < n_beams && !full) {
                const double L = (double)beams[c.parent].codes.size() + 1.0; // generated incl. EOS
                Finished f;
                f.codes = beams[c.parent].codes;
                f.score = c.score / L;
                finished.push_back(std::move(f));
                std::sort(finished.begin(), finished.end(),
                          [](const Finished& a, const Finished& b) { return a.score > b.score; });
                if ((int)finished.size() > n_beams)
                    finished.resize(n_beams);
            }
        }
        if ((int)finished.size() >= n_beams) {
            if (vb >= 1)
                fprintf(stderr, "confucius4: beam decode: %d finished at step %d\n", n_beams, step);
            break;
        }

        // running beams for the next iteration: top n_beams by cont_score
        std::vector<int> sel(cands.size());
        for (size_t j = 0; j < sel.size(); j++)
            sel[j] = (int)j;
        std::stable_sort(sel.begin(), sel.end(),
                         [&](int a, int b) { return cands[a].cont_score > cands[b].cont_score; });
        sel.resize(std::min((size_t)n_beams, sel.size()));

        const int next_bank = 1 - cur_bank;
        std::vector<Beam> next_beams(n_beams);
        for (int i = 0; i < (int)sel.size(); i++) {
            const Cand& c = cands[sel[i]];
            next_beams[i] = beams[c.parent];
            next_beams[i].codes.push_back(c.token);
            next_beams[i].seen[c.token] = 1;
            next_beams[i].score = c.cont_score;
            kv_copy_prefix(bank[next_bank][i], bank[cur_bank][c.parent], n_past);
        }
        for (int i = (int)sel.size(); i < n_beams; i++) {
            next_beams[i] = beams[0];
            next_beams[i].score = -1e18;
            kv_copy_prefix(bank[next_bank][i], bank[cur_bank][0], n_past);
        }
        beams = std::move(next_beams);
        cur_bank = next_bank;

        // feed each beam its new token
        bool fail = false;
        for (int i = 0; i < n_beams; i++) {
            const int sem_pos = std::min((int)beams[i].codes.size(), hp.max_semantic_seq_lens - 1);
            std::vector<float> emb =
                embed_semantic_token(ctx, beams[i].codes.empty() ? eos : beams[i].codes.back(), sem_pos);
            if (emb.empty()) {
                fail = true;
                break;
            }
            beam_logits[i] = run_gpt2_step(ctx, emb.data(), 1, n_past, nullptr, &bank[cur_bank][i]);
            if (beam_logits[i].empty()) {
                fail = true;
                break;
            }
        }
        n_past++;
        if (fail)
            break;
        if (vb >= 2 && step < 6)
            fprintf(stderr, "confucius4: beam step %d: tokens [%d %d %d]\n", step,
                    beams[0].codes.empty() ? -1 : beams[0].codes.back(),
                    n_beams > 1 && !beams[1].codes.empty() ? beams[1].codes.back() : -1,
                    n_beams > 2 && !beams[2].codes.empty() ? beams[2].codes.back() : -1);
    }

    // pick the winner: best finished, else best running (length-penalized)
    std::vector<int32_t> codes;
    if (!finished.empty()) {
        codes = finished[0].codes;
    } else {
        int best = 0;
        double best_s = -1e30;
        for (int i = 0; i < n_beams; i++) {
            const double L = std::max<double>(1.0, (double)beams[i].codes.size());
            if (beams[i].score / L > best_s) {
                best_s = beams[i].score / L;
                best = i;
            }
        }
        codes = beams[best].codes;
    }

    for (int w = 0; w < 2; w++)
        for (auto& b : bank[w])
            kv_free(b);

    if (vb >= 1)
        fprintf(stderr, "confucius4: beam decode: %zu semantic codes (%zu finished hyps)\n", codes.size(),
                finished.size());

    // Teacher-forced latent pass over the winning sequence -- mirrors the
    // reference return_latent path, which re-runs the transformer on
    // [cond | text | BOS | codes] and slices the hidden states.
    if (out_lm_latent && !codes.empty()) {
        const int T_all = prefix_len + (int)codes.size();
        std::vector<float> embeds((size_t)T_all * hp.model_dim);
        std::memcpy(embeds.data(), prefix_emb.data(), prefix_emb.size() * sizeof(float));
        for (size_t i = 0; i < codes.size(); i++) {
            const int sem_pos = std::min((int)i + 1, hp.max_semantic_seq_lens - 1);
            std::vector<float> emb = embed_semantic_token(ctx, codes[i], sem_pos);
            if (emb.empty())
                return codes;
            std::memcpy(embeds.data() + (prefix_len + i) * hp.model_dim, emb.data(), emb.size() * sizeof(float));
        }
        confucius4_kv_cache lat_kv{};
        if (!kv_init(lat_kv, hp, T_all + 1, ctx->backend))
            return codes;
        std::vector<float> hidden_all;
        run_gpt2_step(ctx, embeds.data(), T_all, 0, nullptr, &lat_kv, &hidden_all);
        kv_free(lat_kv);
        if ((int)hidden_all.size() == T_all * hp.model_dim) {
            // hidden at position (prefix_len-1+i) -- BOS for i=0 -- predicts code i
            out_lm_latent->assign(hidden_all.begin() + (size_t)(prefix_len - 1) * hp.model_dim,
                                  hidden_all.begin() + (size_t)(prefix_len - 1 + codes.size()) * hp.model_dim);
        }
    }

    return codes;
}

// Decode text tokens to semantic codes. If `out_lm_latent` is non-null,
// collects the GPT-2 hidden state (model_dim floats) for each generated semantic token.
static std::vector<int32_t> t2s_decode(confucius4_tts_context* ctx, const std::vector<int32_t>& text_token_ids,
                                       std::vector<float>* out_lm_latent = nullptr) {
    // Reference decode is beam-sample with num_beams=3 (inference.py). The
    // pure-sampling path below stays for A/B via CRISPASR_CONFUCIUS4_BEAMS=1.
    {
        int n_beams = 3;
        if (const char* e = std::getenv("CRISPASR_CONFUCIUS4_BEAMS"))
            n_beams = std::atoi(e);
        if (n_beams > 1)
            return t2s_decode_beam(ctx, text_token_ids, n_beams, out_lm_latent);
    }
    const auto& hp = ctx->t2s.hp;
    const int T_text = (int)text_token_ids.size();
    const int prefix_len = 1 + T_text + 1; // condition(1) + text(T) + BOS(1)
    const int max_new = ctx->params.max_semantic_tokens > 0 ? ctx->params.max_semantic_tokens
                                                            : std::max(1, hp.max_semantic_seq_lens - prefix_len);
    const int max_seq = prefix_len + max_new;
    const int vb = ctx->params.verbosity;

    if (vb >= 1)
        fprintf(stderr, "confucius4: T2S decode: text_len=%d, prefix_len=%d, max_new=%d\n", T_text, prefix_len,
                max_new);

    // Allocate KV cache
    kv_free(ctx->kv);
    if (!kv_init(ctx->kv, hp, max_seq, ctx->backend)) {
        fprintf(stderr, "confucius4: KV cache allocation failed\n");
        return {};
    }

    // Seed RNG
    std::mt19937 rng(ctx->params.seed ? ctx->params.seed : 42);

    // ── Step 1: Build prefix embedding via ggml graph ──
    std::vector<float> prefix_emb = build_prefix_embedding(ctx, text_token_ids);
    if (prefix_emb.empty()) {
        fprintf(stderr, "confucius4: prefix embedding failed\n");
        kv_free(ctx->kv);
        return {};
    }

    // ── Step 2: Prefill — run GPT-2 on the full prefix ──
    std::vector<float> logits = run_gpt2_step(ctx, prefix_emb.data(), prefix_len, 0, out_lm_latent);
    if (logits.empty()) {
        fprintf(stderr, "confucius4: prefill failed\n");
        kv_free(ctx->kv);
        return {};
    }

    if (vb >= 1)
        fprintf(stderr, "confucius4: prefill done, logits[0..3] = %.3f %.3f %.3f %.3f\n", logits[0], logits[1],
                logits[2], logits[3]);

    if (s2a_dump_dir()) {
        char shp[64];
        snprintf(shp, sizeof(shp), "%d", (int)text_token_ids.size());
        s2a_dump_raw("text_ids_i32", text_token_ids.data(), text_token_ids.size() * sizeof(int32_t), shp);
        snprintf(shp, sizeof(shp), "%d", (int)logits.size());
        s2a_dump_raw("prefill_logits", logits.data(), logits.size() * sizeof(float), shp);
        // Pre-block-0 structural gate: the full prefix embedding
        // [cond | text+pos | BOS+sem_pos0] as fed to the GPT-2.
        snprintf(shp, sizeof(shp), "%d,%d", prefix_len, hp.model_dim);
        s2a_dump_raw("prefix_emb", prefix_emb.data(), prefix_emb.size() * sizeof(float), shp);
    }

    // ── Step 3: Autoregressive decode ──
    std::vector<int32_t> semantic_codes;
    int n_past = prefix_len;

    // HF RepetitionPenaltyLogitsProcessor: every token already present in
    // input_ids gets its logit divided by the penalty when positive, multiplied
    // when negative — per UNIQUE token, applied BEFORE temperature/top-k/top-p.
    // The reference generates with repetition_penalty=10.0, and its input_ids
    // start as a BOS-filled block, so BOS is penalized from step 0.
    float rep_pen = ctx->params.repetition_penalty;
    if (const char* e = std::getenv("CRISPASR_CONFUCIUS4_REP_PEN"))
        rep_pen = (float)atof(e);
    std::vector<uint8_t> seen(hp.semantic_vocab_size, 0);
    seen[hp.start_semantic_token] = 1;
    auto apply_rep_penalty = [&](float* lg) {
        if (rep_pen == 1.0f)
            return;
        for (int v = 0; v < hp.semantic_vocab_size; v++)
            if (seen[v])
                lg[v] = lg[v] > 0.0f ? lg[v] / rep_pen : lg[v] * rep_pen;
    };

    for (int step = 0; step < max_new; step++) {
        // Sample from logits
        apply_rep_penalty(logits.data());
        int token = sample_top_p(logits.data(), hp.semantic_vocab_size, ctx->params.temperature, ctx->params.top_p,
                                 ctx->params.top_k, rng);

        // Check for EOS
        if (token == hp.stop_semantic_token) {
            if (vb >= 1)
                fprintf(stderr, "confucius4: EOS at step %d\n", step);
            break;
        }

        semantic_codes.push_back(token);
        seen[token] = 1;

        // Embed the new token (semantic_embed + position)
        const int sem_pos = std::min(step + 1, hp.max_semantic_seq_lens - 1);
        std::vector<float> tok_emb = embed_semantic_token(ctx, token, sem_pos);
        if (tok_emb.empty()) {
            fprintf(stderr, "confucius4: token embedding failed at step %d\n", step);
            break;
        }

        // Run one GPT-2 step (also collects hidden state for LM latent)
        logits = run_gpt2_step(ctx, tok_emb.data(), 1, n_past, out_lm_latent);
        if (logits.empty()) {
            fprintf(stderr, "confucius4: decode step %d failed\n", step);
            break;
        }
        n_past++;

        if (vb >= 2 && step < 10)
            fprintf(stderr, "confucius4: step %d: token=%d, n_past=%d\n", step, token, n_past);
    }

    if (vb >= 1)
        fprintf(stderr, "confucius4: generated %zu semantic codes\n", semantic_codes.size());

    kv_free(ctx->kv);
    return semantic_codes;
}

// ---------------------------------------------------------------------------
// S2A CPU helpers — dequantize + matmul on F32 (following the f5-tts pattern)
// ---------------------------------------------------------------------------

// Read a tensor's data as F32 from the backend. For quantized tensors, this
// dequantizes via ggml's type traits. Returns (ne[0] * ne[1]) floats for 2D,
// (ne[0]) for 1D. Caller owns the returned vector.
static std::vector<float> s2a_read_f32(ggml_tensor* t) {
    if (!t)
        return {};
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else {
        // Dequantize: read raw bytes, convert via type traits
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        const auto* tt = ggml_get_type_traits(t->type);
        // Dequant in blocks of ne[0] (one row at a time)
        const int64_t row_size = t->ne[0];
        const int64_t n_rows = n / row_size;
        for (int64_t r = 0; r < n_rows; r++) {
            tt->to_float(raw.data() + r * t->nb[1], out.data() + r * row_size, row_size);
        }
    }
    return out;
}

// y[T,N] = x[T,K] @ W[K,N] + bias[N]  (row-major, W already transposed for matmul)
// This mirrors f5_linear but without BLAS — pure scalar for correctness first.
static void s2a_linear(const float* x, const float* W, const float* bias, float* y, int T, int K, int N) {
    for (int t = 0; t < T; t++) {
        const float* xr = x + (size_t)t * K;
        float* yr = y + (size_t)t * N;
        for (int o = 0; o < N; o++) {
            const float* wr = W + (size_t)o * K;
            float s = bias ? bias[o] : 0.0f;
            for (int k = 0; k < K; k++)
                s += xr[k] * wr[k];
            yr[o] = s;
        }
    }
}

// RMSNorm: y = x / rms(x) * weight
static void s2a_rms_norm(const float* x, const float* weight, float* y, int T, int D, float eps = 1e-6f) {
    for (int t = 0; t < T; t++) {
        const float* xr = x + (size_t)t * D;
        float* yr = y + (size_t)t * D;
        float ss = 0.0f;
        for (int d = 0; d < D; d++)
            ss += xr[d] * xr[d];
        float rms = 1.0f / sqrtf(ss / D + eps);
        for (int d = 0; d < D; d++)
            yr[d] = xr[d] * rms * weight[d];
    }
}

// AdaLN: norm(x) * (1 + scale) + shift, where [scale, shift] = Linear(cond)
static void s2a_adaln(const float* x, const float* norm_w, const float* mod_w, const float* mod_b, const float* cond,
                      float* y, int T, int D) {
    // modulation: (D,) → (2*D,) via Linear
    std::vector<float> mod(2 * D);
    s2a_linear(cond, mod_w, mod_b, mod.data(), 1, D, 2 * D);

    // RMSNorm then modulate
    std::vector<float> normed(T * D);
    s2a_rms_norm(x, norm_w, normed.data(), T, D);

    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) {
            float scale = mod[d];
            float shift = mod[D + d];
            y[t * D + d] = normed[t * D + d] * scale + shift;
        }
    }
}

// ---------------------------------------------------------------------------
// S2A DiT estimator — ggml graph for transformer, CPU for I/O layers
// ---------------------------------------------------------------------------

// Look up an S2A tensor by name, returns nullptr if missing.
static ggml_tensor* s2a_find(const confucius4_s2a_model& s, const std::string& name) {
    auto it = s.tensors.find(name);
    return it != s.tensors.end() ? it->second : nullptr;
}

// SiLU activation in-place: x *= sigmoid(x)
static void s2a_silu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

// Sinusoidal position embedding for a scalar timestep (SinusPositionEmbedding).
// Reference: confuciustts/flow/DiT/modules.py
//   half_dim = dim // 2
//   emb = exp(arange(half_dim) * -(log(10000) / half_dim))
//   emb = scale * t * emb                 with scale = 1000
//   emb = cat((emb.cos(), emb.sin()), -1) -- cosine block FIRST
// The scale=1000 matters: t runs over [0, 1], so without it every frequency
// term collapses towards cos=1 / sin=0 and the ODE sees an almost constant
// timestep signal.
static void s2a_sinusoidal_embed(float t, float* out, int freq_dim) {
    const int half = freq_dim / 2;
    const float scale = 1000.0f;
    for (int i = 0; i < half; i++) {
        float freq = expf(-logf(10000.0f) * (float)i / (float)half);
        float arg = scale * t * freq;
        out[i] = cosf(arg);
        out[half + i] = sinf(arg);
    }
}

// Compute timestep embedding on CPU: sinusoidal → Linear → SiLU → Linear.
// `prefix` is "decoder.estimator.t_embedder" or "decoder.estimator.t_embedder2".
static std::vector<float> s2a_timestep_embed_cpu(const confucius4_s2a_model& s, float t, const std::string& prefix,
                                                 int dim) {
    const int freq_dim = 256;
    std::vector<float> sin_emb(freq_dim);
    s2a_sinusoidal_embed(t, sin_emb.data(), freq_dim);

    std::string n0 = prefix + ".time_mlp.0.weight";
    auto mlp0_w = s2a_read_f32(s2a_find(s, n0));
    auto mlp0_b = s2a_read_f32(s2a_find(s, prefix + ".time_mlp.0.bias"));
    if (mlp0_w.empty()) {
        fprintf(stderr, "confucius4: missing S2A tensor '%s'\n", n0.c_str());
        return {};
    }

    std::vector<float> h(dim);
    s2a_linear(sin_emb.data(), mlp0_w.data(), mlp0_b.data(), h.data(), 1, freq_dim, dim);
    s2a_silu_inplace(h.data(), dim);

    auto mlp2_w = s2a_read_f32(s2a_find(s, prefix + ".time_mlp.2.weight"));
    auto mlp2_b = s2a_read_f32(s2a_find(s, prefix + ".time_mlp.2.bias"));

    std::vector<float> out(dim);
    s2a_linear(h.data(), mlp2_w.data(), mlp2_b.data(), out.data(), 1, dim, dim);
    return out;
}

// Compute input embedding on CPU:
//   mu_proj = Linear(mu, cond_dim→hidden_dim)
//   cat(x, cond_ref, mu_proj, spks) → Linear → hidden
// x: (T, mel_dim) row-major   cond_ref: (T, mel_dim) row-major (zeros if no ref)
// mu: (T, cond_dim) row-major  — conditioning from s2a_build_conditioning
// use_spk=false zeroes the speaker slice, for the unconditioned CFG pass.
static std::vector<float> s2a_input_embed_cpu(confucius4_tts_context* ctx, const float* x, const float* cond_ref,
                                              const float* mu, int T, int mel_dim, bool use_spk = true) {
    const auto& s = ctx->s2a;
    const int dim = s.hp.estimator_hidden_dim;
    const int cond_dim = dim; // mu_projection: Linear(cond_dim=dim, dim)
    const int spk_dim = s.hp.spk_embed_dim;

    auto mu_w = s2a_read_f32(s2a_find(s, "decoder.estimator.input_embed.mu_projection.weight"));
    auto mu_b = s2a_read_f32(s2a_find(s, "decoder.estimator.input_embed.mu_projection.bias"));
    if (mu_w.empty())
        return {};

    std::vector<float> mu_proj((size_t)T * dim);
    s2a_linear(mu, mu_w.data(), mu_b.data(), mu_proj.data(), T, cond_dim, dim);

    // cat(x, cond_ref, mu_proj, spks) → (T, cat_dim)
    const int cat_dim = mel_dim * 2 + dim + spk_dim;
    std::vector<float> cat_in((size_t)T * cat_dim, 0.0f);
    for (int t = 0; t < T; t++) {
        float* row = cat_in.data() + (size_t)t * cat_dim;
        std::memcpy(row, x + (size_t)t * mel_dim, mel_dim * sizeof(float));
        std::memcpy(row + mel_dim, cond_ref + (size_t)t * mel_dim, mel_dim * sizeof(float));
        std::memcpy(row + mel_dim * 2, mu_proj.data() + (size_t)t * dim, dim * sizeof(float));
        if (use_spk && ctx->has_speaker && spk_dim > 0)
            std::memcpy(row + mel_dim * 2 + dim, ctx->speaker_style_embedding.data(), spk_dim * sizeof(float));
    }

    auto proj_w = s2a_read_f32(s2a_find(s, "decoder.estimator.input_embed.proj.weight"));
    auto proj_b = s2a_read_f32(s2a_find(s, "decoder.estimator.input_embed.proj.bias"));
    if (proj_w.empty())
        return {};

    std::vector<float> out((size_t)T * dim);
    s2a_linear(cat_in.data(), proj_w.data(), proj_b.data(), out.data(), T, cat_dim, dim);
    return out; // (T * dim) row-major = ggml (dim, T) data layout
}

// Build the DiT ggml graph for transformer blocks + final output.
// Produces velocity (mel_dim, T) from hidden input + timestep embedding.
//
// fused=true builds the CFG-fused variant (CRISPASR_CONFUCIUS4_CFG_FUSE=1):
// cond and uncond are seq-concatenated along time ([0,T) = cond, [T,2T) =
// uncond) and separated by a block-diagonal flash-attn mask — ONE graph eval
// per ODE step instead of two (dev-guide: seq-concat beats a 4D batch dim).
// Every op in the graph is per-frame except attention (masked) and the
// WaveNet k=5 convs, which would leak ±2 frames/layer across the seam — so
// the WaveNet runs per-arm on split halves inside the same graph.  The CFG
// blend (1+cfg)*v_cond - cfg*v_uncond happens in-graph; output is (mel, T).
static bool s2a_dit_cache_build(confucius4_tts_context* ctx, int T, int mel_dim, bool fused = false,
                                float cfg_rate = 0.0f) {
    auto& cache = ctx->dit_cache;
    if (cache.T_cached == T && cache.mel_dim_cached == mel_dim && cache.fused_cfg == fused &&
        (!fused || cache.cfg_rate_cached == cfg_rate) && cache.galloc)
        return true;
    cache.reset();

    const auto& s = ctx->s2a;
    const int dim = s.hp.estimator_hidden_dim;
    const int n_heads = s.hp.estimator_num_heads;
    const int head_dim = dim / n_heads;
    const int depth = s.hp.estimator_depth;
    const int half = depth / 2;
    const int vb = ctx->params.verbosity;
    const int Tg = fused ? 2 * T : T; // graph-side sequence length

    // Debug taps: when CRISPASR_CONFUCIUS4_DUMP_S2A is set, mark selected
    // intermediates as graph outputs so gallocr cannot elide them, and read
    // them back after compute to bisect the estimator against the reference.
    const bool dbg_taps = s2a_dump_dir() != nullptr;
    auto tap = [&](ggml_tensor* t, const char* name) {
        if (dbg_taps && t) {
            ggml_set_name(t, name);
            ggml_set_output(t);
        }
        return t;
    };

    // Determine FFN intermediate size from w1 weight shape
    auto w1_0 = s2a_find(s, "decoder.estimator.transformer_blocks.0.feed_forward.w1.weight");
    const int inter_dim = w1_0 ? (int)w1_0->ne[1] : dim * 4;

    if (vb >= 1)
        fprintf(stderr, "confucius4: DiT graph: depth=%d dim=%d heads=%d inter=%d mel=%d T=%d%s\n", depth, dim, n_heads,
                inter_dim, mel_dim, T, fused ? " (CFG-fused, 2T)" : "");

    // Allocate graph context: ~35 ops per block × depth + ~20 finals
    struct ggml_init_params p = {4 * 1024 * 1024, nullptr, true};
    cache.gctx = ggml_init(p);
    if (!cache.gctx)
        return false;

    // Graph inputs
    cache.hidden_in = ggml_new_tensor_2d(cache.gctx, GGML_TYPE_F32, dim, Tg);
    ggml_set_name(cache.hidden_in, "dit_in");
    ggml_set_input(cache.hidden_in);

    cache.t_emb_in = ggml_new_tensor_1d(cache.gctx, GGML_TYPE_F32, dim);
    ggml_set_name(cache.t_emb_in, "dit_t");
    ggml_set_input(cache.t_emb_in);

    cache.x_mel_in = ggml_new_tensor_2d(cache.gctx, GGML_TYPE_F32, mel_dim, Tg);
    ggml_set_name(cache.x_mel_in, "dit_xmel");
    ggml_set_input(cache.x_mel_in);

    cache.pos_in = ggml_new_tensor_1d(cache.gctx, GGML_TYPE_I32, Tg);
    ggml_set_name(cache.pos_in, "dit_pos");
    ggml_set_input(cache.pos_in);

    if (fused) {
        // Block-diagonal attention mask (n_kv, n_q) F16: each arm attends only
        // to its own frames.  Re-set on EVERY compute (gallocr may reuse input
        // buffers as scratch — the §234 aliasing gotcha).
        cache.attn_mask = ggml_new_tensor_2d(cache.gctx, GGML_TYPE_F16, Tg, Tg);
        ggml_set_name(cache.attn_mask, "dit_mask");
        ggml_set_input(cache.attn_mask);
    }

    // Helper to find a DiT tensor by formatted name
    char nbuf[256];
    auto tn = [&](const char* fmt, int layer) -> ggml_tensor* {
        snprintf(nbuf, sizeof(nbuf), fmt, layer);
        return s2a_find(s, nbuf);
    };

    // U-Net skip connection stack (LIFO): emit layers 0..half-1, receive half+1..depth-1
    std::vector<ggml_tensor*> skip_stack;
    ggml_tensor* x = cache.hidden_in;

    for (int i = 0; i < depth; i++) {
        // Receive skip connection
        if (i > half && !skip_stack.empty()) {
            ggml_tensor* skip = skip_stack.back();
            skip_stack.pop_back();
            ggml_tensor* cat = ggml_concat(cache.gctx, x, skip, 0);
            auto sw = tn("decoder.estimator.transformer_blocks.%d.skip_in_linear.weight", i);
            auto sb = tn("decoder.estimator.transformer_blocks.%d.skip_in_linear.bias", i);
            if (sw) {
                x = ggml_mul_mat(cache.gctx, sw, cat);
                if (sb)
                    x = ggml_add(cache.gctx, x, sb);
            }
        }

        // AdaLN attention norm: modulation(t_emb) → [weight, bias]; RMSNorm(x)*w + b
        {
            auto nw = tn("decoder.estimator.transformer_blocks.%d.attention_norm.norm.weight", i);
            auto mw = tn("decoder.estimator.transformer_blocks.%d.attention_norm.modulation.weight", i);
            auto mb = tn("decoder.estimator.transformer_blocks.%d.attention_norm.modulation.bias", i);

            ggml_tensor* mod = ggml_mul_mat(cache.gctx, mw, cache.t_emb_in);
            if (mb)
                mod = ggml_add(cache.gctx, mod, mb);
            ggml_tensor* aw = ggml_view_1d(cache.gctx, mod, dim, 0);
            ggml_tensor* ab = ggml_view_1d(cache.gctx, mod, dim, dim * sizeof(float));

            ggml_tensor* normed = ggml_rms_norm(cache.gctx, x, 1e-5f);
            if (nw)
                normed = ggml_mul(cache.gctx, normed, nw);
            normed = ggml_mul(cache.gctx, normed, aw);
            normed = ggml_add(cache.gctx, normed, ab);

            // Attention: fused wqkv → Q,K,V → RoPE → flash_attn → wo
            auto wqkv = tn("decoder.estimator.transformer_blocks.%d.attention.wqkv.weight", i);
            auto wo = tn("decoder.estimator.transformer_blocks.%d.attention.wo.weight", i);

            ggml_tensor* qkv = ggml_mul_mat(cache.gctx, wqkv, normed);
            int esz = (int)ggml_element_size(qkv);
            ggml_tensor* q = ggml_cont(cache.gctx, ggml_view_2d(cache.gctx, qkv, dim, Tg, 3 * dim * esz, 0));
            ggml_tensor* k = ggml_cont(cache.gctx, ggml_view_2d(cache.gctx, qkv, dim, Tg, 3 * dim * esz, dim * esz));
            ggml_tensor* v =
                ggml_cont(cache.gctx, ggml_view_2d(cache.gctx, qkv, dim, Tg, 3 * dim * esz, 2 * dim * esz));

            q = ggml_reshape_3d(cache.gctx, q, head_dim, n_heads, Tg);
            k = ggml_reshape_3d(cache.gctx, k, head_dim, n_heads, Tg);
            v = ggml_reshape_3d(cache.gctx, v, head_dim, n_heads, Tg);

            q = ggml_rope_ext(cache.gctx, q, cache.pos_in, nullptr, head_dim, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                              0.0f);
            k = ggml_rope_ext(cache.gctx, k, cache.pos_in, nullptr, head_dim, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                              0.0f);

            q = ggml_permute(cache.gctx, q, 0, 2, 1, 3);
            k = ggml_permute(cache.gctx, k, 0, 2, 1, 3);
            v = ggml_permute(cache.gctx, v, 0, 2, 1, 3);

            float scale = 1.0f / sqrtf((float)head_dim);
            // Flow-matching estimator (f5-class): default to manual F32 SDPA —
            // fused flash accumulates KQ in F16 and set_prec is ignored on some
            // GPUs (P100/sm_60), which corrupts the ODE. Fused is opt-in via
            // CRISPASR_CONFUCIUS4_FLASH=1. cache.attn_mask is the (Tg,Tg) F16
            // block-diagonal CFG mask, or nullptr when not fused (full attn).
            ggml_tensor* attn = core_sdpa::attn(cache.gctx, q, k, v, cache.attn_mask, dim, Tg, scale,
                                                core_sdpa::flash_enabled("CRISPASR_CONFUCIUS4_FLASH"));

            ggml_tensor* attn_proj = ggml_mul_mat(cache.gctx, wo, attn);
            x = ggml_add(cache.gctx, x, attn_proj);
        }

        // AdaLN FFN norm → SwiGLU FFN
        {
            auto nw = tn("decoder.estimator.transformer_blocks.%d.ffn_norm.norm.weight", i);
            auto mw = tn("decoder.estimator.transformer_blocks.%d.ffn_norm.modulation.weight", i);
            auto mb = tn("decoder.estimator.transformer_blocks.%d.ffn_norm.modulation.bias", i);

            ggml_tensor* mod = ggml_mul_mat(cache.gctx, mw, cache.t_emb_in);
            if (mb)
                mod = ggml_add(cache.gctx, mod, mb);
            ggml_tensor* fw = ggml_view_1d(cache.gctx, mod, dim, 0);
            ggml_tensor* fb = ggml_view_1d(cache.gctx, mod, dim, dim * sizeof(float));

            ggml_tensor* fn = ggml_rms_norm(cache.gctx, x, 1e-5f);
            if (nw)
                fn = ggml_mul(cache.gctx, fn, nw);
            fn = ggml_mul(cache.gctx, fn, fw);
            fn = ggml_add(cache.gctx, fn, fb);

            // SwiGLU: w2(silu(w1(x)) * w3(x))
            auto w1 = tn("decoder.estimator.transformer_blocks.%d.feed_forward.w1.weight", i);
            auto w2 = tn("decoder.estimator.transformer_blocks.%d.feed_forward.w2.weight", i);
            auto w3 = tn("decoder.estimator.transformer_blocks.%d.feed_forward.w3.weight", i);

            ggml_tensor* gate = ggml_silu(cache.gctx, ggml_mul_mat(cache.gctx, w1, fn));
            ggml_tensor* up = ggml_mul_mat(cache.gctx, w3, fn);
            ggml_tensor* ff = ggml_mul_mat(cache.gctx, w2, ggml_mul(cache.gctx, gate, up));

            x = ggml_add(cache.gctx, x, ff);
        }

        if (i == 0 || i == half || i == depth - 1) {
            char tn_dbg[32];
            snprintf(tn_dbg, sizeof(tn_dbg), "dbg_blk%02d", i);
            tap(x, tn_dbg);
        }

        // Emit skip
        if (i < half)
            skip_stack.push_back(x);
    }

    // Final AdaLN (transformer_norm)
    {
        auto nw = s2a_find(s, "decoder.estimator.transformer_norm.norm.weight");
        auto mw = s2a_find(s, "decoder.estimator.transformer_norm.modulation.weight");
        auto mb = s2a_find(s, "decoder.estimator.transformer_norm.modulation.bias");

        ggml_tensor* mod = ggml_mul_mat(cache.gctx, mw, cache.t_emb_in);
        if (mb)
            mod = ggml_add(cache.gctx, mod, mb);
        ggml_tensor* tw = ggml_view_1d(cache.gctx, mod, dim, 0);
        ggml_tensor* tb = ggml_view_1d(cache.gctx, mod, dim, dim * sizeof(float));

        x = ggml_rms_norm(cache.gctx, x, 1e-5f);
        if (nw)
            x = ggml_mul(cache.gctx, x, nw);
        x = ggml_mul(cache.gctx, x, tw);
        x = ggml_add(cache.gctx, x, tb);
    }

    tap(x, "dbg_xres");

    // skip_linear: cat(x_res, x_mel) → Linear(dim + mel_dim → dim)
    auto sl_w = s2a_find(s, "decoder.estimator.skip_linear.weight");
    auto sl_b = s2a_find(s, "decoder.estimator.skip_linear.bias");
    if (sl_w) {
        ggml_tensor* cat = ggml_concat(cache.gctx, x, cache.x_mel_in, 0);
        x = ggml_mul_mat(cache.gctx, sl_w, cat);
        if (sl_b)
            x = ggml_add(cache.gctx, x, sl_b);
    }

    tap(x, "dbg_skip");

    // Full output path in ggml graph (WaveNet weight_norm folded at load time).
    ggml_tensor* x_res = x; // save for res_projection

    // conv1: Linear(dim→dim) — operates in (dim, T) space
    auto c1_w = s2a_find(s, "decoder.estimator.conv1.weight");
    auto c1_b = s2a_find(s, "decoder.estimator.conv1.bias");
    ggml_tensor* x_conv1 = c1_w ? ggml_mul_mat(cache.gctx, c1_w, x_res) : x_res;
    if (c1_b)
        x_conv1 = ggml_add(cache.gctx, x_conv1, c1_b);

    // Transpose to time-first (T, dim) for ggml_conv_1d in WaveNet
    x_conv1 = ggml_cont(cache.gctx, ggml_transpose(cache.gctx, x_conv1));

    // WaveNet: gated dilated residual network with timestep conditioning (t2)
    // Uses fused weight_norm tensors (folded at load time)
    cache.t2_emb_in = ggml_new_tensor_1d(cache.gctx, GGML_TYPE_F32, dim);
    ggml_set_name(cache.t2_emb_in, "dit_t2");
    ggml_set_input(cache.t2_emb_in);

    auto cond_w = s2a_find(s, "decoder.estimator.wavenet.cond_layer.conv.weight");
    auto cond_b = s2a_find(s, "decoder.estimator.wavenet.cond_layer.conv.bias");

    ggml_tensor* wn_output = nullptr;

    if (cond_w) {
        const int n_wn_layers = ctx->s2a.hp.wavenet_num_layers;
        const int hidden = dim;
        // cond_layer: Conv1d(dim, 2*dim*n_layers, 1) — kernel=1 so use mul_mat
        // cond_w is 3D after fold: (1, dim, 2*dim*n_layers); reshape to 2D
        ggml_tensor* cw2d = ggml_reshape_2d(cache.gctx, cond_w, cond_w->ne[0] * cond_w->ne[1], cond_w->ne[2]);
        ggml_tensor* g_all = ggml_mul_mat(cache.gctx, cw2d, cache.t2_emb_in); // (2*dim*n_layers,)
        if (cond_b)
            g_all = ggml_add(cache.gctx, g_all, cond_b);

        // One WaveNet stack over a (T_w, dim) time-first input.  Runs once in
        // the plain graph; in the fused graph it runs once PER ARM — the k=5
        // convs would otherwise smear cond and uncond into each other across
        // the concat seam (±2 frames per layer, compounding through the
        // residual chain).
        auto build_wavenet = [&](ggml_tensor* x_wn, int T_w) -> ggml_tensor* {
            ggml_tensor* wn_out = ggml_dup(cache.gctx, ggml_scale(cache.gctx, x_wn, 0.0f)); // zeros (T_w, dim)

            char nbuf[256];
            for (int il = 0; il < n_wn_layers; il++) {
                // in_layers[il]: Conv1d(dim, 2*dim, k=5, pad=2) — fused weight
                snprintf(nbuf, sizeof(nbuf), "decoder.estimator.wavenet.in_layers.%d.conv.weight", il);
                auto il_w = s2a_find(s, nbuf);
                snprintf(nbuf, sizeof(nbuf), "decoder.estimator.wavenet.in_layers.%d.conv.bias", il);
                auto il_b = s2a_find(s, nbuf);

                ggml_tensor* x_in_l = il_w ? ggml_conv_1d(cache.gctx, il_w, x_wn, 1, 2, 1) : x_wn;
                if (il_b) {
                    ggml_tensor* b2d = ggml_reshape_2d(cache.gctx, il_b, 1, il_b->ne[0]);
                    x_in_l = ggml_add(cache.gctx, x_in_l, b2d);
                }

                // Slice g_all for this layer: g_l = g_all[il*2*hidden .. (il+1)*2*hidden]
                ggml_tensor* g_l = ggml_view_1d(cache.gctx, g_all, 2 * hidden, (size_t)il * 2 * hidden * sizeof(float));

                // x_in_l is (T, 2*hidden) time-first from conv1d. Split into two (T, hidden).
                // g_l is (2*hidden,) — broadcast over T.
                ggml_tensor* g_a = ggml_view_1d(cache.gctx, g_l, hidden, 0);
                ggml_tensor* g_b = ggml_view_1d(cache.gctx, g_l, hidden, hidden * sizeof(float));

                // ne[0]=T, ne[1]=2*hidden. Split along ne[1]:
                int64_t stride1 = x_in_l->nb[1];
                // x_in_l is time-first (T, 2*hidden): ne[0]=T, ne[1]=channel.  The
                // split is along the CHANNEL axis, so the second half starts
                // `hidden` ROWS in -- hidden * nb[1], not hidden * element_size.
                // Using the element size offsets by `hidden` samples along TIME
                // instead, which makes the sigmoid half overlap the tanh half.
                ggml_tensor* xa = ggml_cont(cache.gctx, ggml_view_2d(cache.gctx, x_in_l, T_w, hidden, stride1, 0));
                ggml_tensor* xb = ggml_cont(
                    cache.gctx, ggml_view_2d(cache.gctx, x_in_l, T_w, hidden, stride1, (size_t)hidden * stride1));
                // Reshape g_a/g_b from (hidden,) to (1, hidden) for time-first broadcast
                ggml_tensor* ga2 = ggml_reshape_2d(cache.gctx, g_a, 1, hidden);
                ggml_tensor* gb2 = ggml_reshape_2d(cache.gctx, g_b, 1, hidden);
                xa = ggml_add(cache.gctx, xa, ga2);
                xb = ggml_add(cache.gctx, xb, gb2);

                // Gated activation: tanh(xa) * sigmoid(xb)
                ggml_tensor* acts = ggml_mul(cache.gctx, ggml_tanh(cache.gctx, xa), ggml_sigmoid(cache.gctx, xb));

                // res_skip_layers[il]: Conv1d(hidden, rs_ch, 1) — fused weight
                snprintf(nbuf, sizeof(nbuf), "decoder.estimator.wavenet.res_skip_layers.%d.conv.weight", il);
                auto rs_w = s2a_find(s, nbuf);
                snprintf(nbuf, sizeof(nbuf), "decoder.estimator.wavenet.res_skip_layers.%d.conv.bias", il);
                auto rs_b = s2a_find(s, nbuf);

                ggml_tensor* rs_out = rs_w ? ggml_conv_1d(cache.gctx, rs_w, acts, 1, 0, 1) : acts;
                if (rs_b) {
                    ggml_tensor* b2d = ggml_reshape_2d(cache.gctx, rs_b, 1, rs_b->ne[0]);
                    rs_out = ggml_add(cache.gctx, rs_out, b2d);
                }

                int rs_ch = rs_w ? (int)rs_w->ne[2] : hidden;
                if (il < n_wn_layers - 1 && rs_ch >= 2 * hidden) {
                    // rs_out is (T, rs_ch) time-first. Split: res=(T,hidden), skip=(T,hidden)
                    int64_t rs_stride = rs_out->nb[1];
                    // Same channel-axis split as above: offset by rows, not elements.
                    ggml_tensor* res =
                        ggml_cont(cache.gctx, ggml_view_2d(cache.gctx, rs_out, T_w, hidden, rs_stride, 0));
                    ggml_tensor* skip = ggml_cont(cache.gctx, ggml_view_2d(cache.gctx, rs_out, T_w, hidden, rs_stride,
                                                                           (size_t)hidden * rs_stride));
                    x_wn = ggml_add(cache.gctx, x_wn, res);
                    wn_out = ggml_add(cache.gctx, wn_out, skip);
                } else {
                    // Last layer: output += rs_out
                    wn_out = ggml_add(cache.gctx, wn_out, rs_out);
                }
            }
            return wn_out;
        };

        if (!fused) {
            wn_output = build_wavenet(x_conv1, T);
        } else {
            // Split the fused (2T, dim) time-first tensor into per-arm halves,
            // run the WaveNet on each, and re-concat along time.
            ggml_tensor* h_cond = ggml_cont(cache.gctx, ggml_view_2d(cache.gctx, x_conv1, T, dim, x_conv1->nb[1], 0));
            ggml_tensor* h_unc = ggml_cont(
                cache.gctx, ggml_view_2d(cache.gctx, x_conv1, T, dim, x_conv1->nb[1], (size_t)T * sizeof(float)));
            wn_output = ggml_concat(cache.gctx, build_wavenet(h_cond, T), build_wavenet(h_unc, T), 0);
        }
    }

    // Combine: wn_output (T, dim) time-first + res_projection(x_res) (dim, T) channel-first
    // Transpose wn_output back to (dim, T) to match the rest of the graph
    if (wn_output)
        wn_output = ggml_cont(cache.gctx, ggml_transpose(cache.gctx, wn_output));
    tap(wn_output, "dbg_wn");

    auto res_w = s2a_find(s, "decoder.estimator.res_projection.weight");
    auto res_b = s2a_find(s, "decoder.estimator.res_projection.bias");
    ggml_tensor* res_proj = res_w ? ggml_mul_mat(cache.gctx, res_w, x_res) : x_res;
    if (res_b)
        res_proj = ggml_add(cache.gctx, res_proj, res_b);

    x = wn_output ? ggml_add(cache.gctx, wn_output, res_proj) : res_proj;

    // FinalLayer: LayerNorm(no affine) → (1+scale)*x + shift → Linear
    auto fl_w = s2a_find(s, "decoder.estimator.final_layer.linear.weight");
    auto fl_b = s2a_find(s, "decoder.estimator.final_layer.linear.bias");
    auto fm_w = s2a_find(s, "decoder.estimator.final_layer.adaLN_modulation.1.weight");
    auto fm_b = s2a_find(s, "decoder.estimator.final_layer.adaLN_modulation.1.bias");
    if (fl_w && fm_w) {
        ggml_tensor* silu_t = ggml_silu(cache.gctx, cache.t_emb_in);
        ggml_tensor* fmod = ggml_mul_mat(cache.gctx, fm_w, silu_t);
        if (fm_b)
            fmod = ggml_add(cache.gctx, fmod, fm_b);

        int wn_dim = (int)fl_w->ne[0];
        ggml_tensor* fsh = ggml_view_1d(cache.gctx, fmod, wn_dim, 0);
        ggml_tensor* fsc = ggml_view_1d(cache.gctx, fmod, wn_dim, wn_dim * sizeof(float));

        ggml_tensor* xn = ggml_norm(cache.gctx, x, 1e-6f);
        ggml_tensor* xs = ggml_mul(cache.gctx, xn, fsc);
        xn = ggml_add(cache.gctx, xn, xs);
        xn = ggml_add(cache.gctx, xn, fsh);

        x = ggml_mul_mat(cache.gctx, fl_w, xn);
        if (fl_b)
            x = ggml_add(cache.gctx, x, fl_b);
    }

    tap(x, "dbg_fin");

    // conv2: Conv1d(wn_dim, mel_dim, 1) = Linear projection
    auto c2_w = s2a_find(s, "decoder.estimator.conv2.weight");
    auto c2_b = s2a_find(s, "decoder.estimator.conv2.bias");
    if (c2_w) {
        ggml_tensor* c2_2d = ggml_reshape_2d(cache.gctx, c2_w, c2_w->ne[0] * c2_w->ne[1], c2_w->ne[2]);
        x = ggml_mul_mat(cache.gctx, c2_2d, x);
        if (c2_b)
            x = ggml_add(cache.gctx, x, c2_b);
    }

    if (fused) {
        // CFG blend in-graph: rows [0,T) are the cond arm, [T,2T) the uncond
        // arm.  v = (1 + cfg) * v_cond - cfg * v_uncond, output (out_dim, T).
        ggml_tensor* v_cond = ggml_view_2d(cache.gctx, x, x->ne[0], T, x->nb[1], 0);
        ggml_tensor* v_unc = ggml_view_2d(cache.gctx, x, x->ne[0], T, x->nb[1], (size_t)T * x->nb[1]);
        x = ggml_add(cache.gctx, ggml_scale(cache.gctx, v_cond, 1.0f + cfg_rate),
                     ggml_scale(cache.gctx, v_unc, -cfg_rate));
    }

    ggml_set_name(x, "velocity");
    ggml_set_output(x);
    cache.output = x;

    // Build and allocate graph
    cache.gf = ggml_new_graph_custom(cache.gctx, 8192, false);
    ggml_build_forward_expand(cache.gf, cache.output);

    cache.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_reserve(cache.galloc, cache.gf) || !ggml_gallocr_alloc_graph(cache.galloc, cache.gf)) {
        fprintf(stderr, "confucius4: DiT graph allocation failed\n");
        cache.reset();
        return false;
    }

    cache.T_cached = T;
    cache.mel_dim_cached = mel_dim;
    cache.fused_cfg = fused;
    cache.cfg_rate_cached = cfg_rate;

    if (vb >= 1)
        fprintf(stderr, "confucius4: DiT graph built: %d nodes\n", ggml_graph_n_nodes(cache.gf));

    return true;
}

// Run one DiT forward pass. hidden: (T*dim), t1/t2: (dim), x_mel: (T*mel_dim).
// Returns velocity as (T*out_dim) row-major, or empty on failure.
// fused=true runs the CFG-fused graph: hidden and x_mel then hold BOTH arms
// (2T frames, cond first), and the returned velocity is already CFG-blended
// over T frames.
static std::vector<float> s2a_dit_run(confucius4_tts_context* ctx, const float* hidden, const float* t1_emb,
                                      const float* t2_emb, const float* x_mel, int T, int mel_dim, bool fused = false,
                                      float cfg_rate = 0.0f) {
    auto& cache = ctx->dit_cache;
    if (!s2a_dit_cache_build(ctx, T, mel_dim, fused, cfg_rate))
        return {};

    if (!ggml_gallocr_alloc_graph(cache.galloc, cache.gf))
        return {};

    const int dim = ctx->s2a.hp.estimator_hidden_dim;
    const int Tg = fused ? 2 * T : T;

    ggml_backend_tensor_set(cache.hidden_in, hidden, 0, (size_t)Tg * dim * sizeof(float));
    ggml_backend_tensor_set(cache.t_emb_in, t1_emb, 0, (size_t)dim * sizeof(float));
    if (cache.t2_emb_in)
        ggml_backend_tensor_set(cache.t2_emb_in, t2_emb, 0, (size_t)dim * sizeof(float));
    ggml_backend_tensor_set(cache.x_mel_in, x_mel, 0, (size_t)Tg * mel_dim * sizeof(float));

    // Position indices must be re-set each call (gallocr may alias input
    // buffers).  In the fused graph both arms use positions [0, T).
    std::vector<int32_t> pos(Tg);
    for (int i = 0; i < Tg; i++)
        pos[i] = i % T;
    ggml_backend_tensor_set(cache.pos_in, pos.data(), 0, (size_t)Tg * sizeof(int32_t));

    if (fused && cache.attn_mask) {
        // Block-diagonal mask, re-set EVERY call (gallocr reuses input buffers
        // as scratch after their last read — the §234 aliasing gotcha).
        if ((int64_t)cache.mask_host.size() != (int64_t)Tg * Tg) {
            cache.mask_host.assign((size_t)Tg * Tg, ggml_fp32_to_fp16(0.0f));
            const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
            for (int qi = 0; qi < Tg; qi++) {
                ggml_fp16_t* row = cache.mask_host.data() + (size_t)qi * Tg;
                // keys of the OTHER arm are masked out
                if (qi < T)
                    std::fill(row + T, row + Tg, ninf);
                else
                    std::fill(row, row + T, ninf);
            }
        }
        ggml_backend_tensor_set(cache.attn_mask, cache.mask_host.data(), 0,
                                cache.mask_host.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_graph_compute(ctx->backend, cache.gf) != GGML_STATUS_SUCCESS)
        return {};

    // Read back the debug taps once, on the first call, so the parity harness
    // can localize where inside the graph the estimator starts to diverge.
    if (s2a_dump_dir()) {
        static bool tapped_once = false;
        if (!tapped_once) {
            tapped_once = true;
            for (const char* nm :
                 {"dbg_blk00", "dbg_blk06", "dbg_blk12", "dbg_xres", "dbg_skip", "dbg_wn", "dbg_fin"}) {
                ggml_tensor* t = ggml_graph_get_tensor(cache.gf, nm);
                if (!t)
                    continue;
                std::vector<float> buf(ggml_nelements(t));
                ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
                char shp[64];
                // taps are (C, T) channel-first except dbg_wn, which the graph
                // has already transposed back to (C, T) as well
                snprintf(shp, sizeof(shp), "%d,%d", (int)t->ne[1], (int)t->ne[0]);
                s2a_dump_raw(nm, buf.data(), buf.size() * sizeof(float), shp);
            }
        }
    }

    int out_dim = (int)cache.output->ne[0];
    std::vector<float> vel((size_t)T * out_dim);
    ggml_backend_tensor_get(cache.output, vel.data(), 0, vel.size() * sizeof(float));
    return vel;
}

// Full DiT estimator forward: timestep embed + input embed (CPU) → DiT graph → output path (CPU).
// x_flat: (T*mel_dim) row-major (noisy data), cond: (T*cond_dim) row-major (semantic cond),
// cond_ref: (T*mel_dim) row-major (reference mel, zeros if none).
// Returns velocity (T*mel_dim) row-major.
static std::vector<float> s2a_dit_forward(confucius4_tts_context* ctx, const float* x_flat, int T, int mel_dim,
                                          const float* cond, const float* cond_ref, float timestep,
                                          bool use_spk = true) {
    const auto& s = ctx->s2a;
    const int dim = s.hp.estimator_hidden_dim;

    // t1 (transformer + final_layer) and t2 (WaveNet) timestep embeddings
    auto t1 = s2a_timestep_embed_cpu(s, timestep, "decoder.estimator.t_embedder", dim);
    auto t2 = s2a_timestep_embed_cpu(s, timestep, "decoder.estimator.t_embedder2", dim);
    if (t1.empty()) {
        fprintf(stderr, "confucius4: timestep embedding failed\n");
        return {};
    }
    if (t2.empty())
        t2 = t1; // fallback: share t1 if t2 MLP not found

    // Input embedding: cat(x, cond_ref, mu_proj(cond), spks) → proj (CPU)
    auto hidden = s2a_input_embed_cpu(ctx, x_flat, cond_ref, cond, T, mel_dim, use_spk);
    if (hidden.empty()) {
        fprintf(stderr, "confucius4: input embedding failed\n");
        return {};
    }

    // Bisect the estimator: dump the CPU-side stages once, on the first
    // conditioned call, so the harness can tell an input_embed / timestep bug
    // from one inside the ggml graph.
    if (s2a_dump_dir() && use_spk) {
        static bool dumped_once = false;
        if (!dumped_once) {
            dumped_once = true;
            char shp[64];
            snprintf(shp, sizeof(shp), "%d,%d", T, dim);
            s2a_dump_raw("dit_x_in", hidden.data(), hidden.size() * sizeof(float), shp);
            snprintf(shp, sizeof(shp), "%d", dim);
            s2a_dump_raw("dit_t1", t1.data(), t1.size() * sizeof(float), shp);
            s2a_dump_raw("dit_t2", t2.data(), t2.size() * sizeof(float), shp);
        }
    }

    // Full ggml graph: DiT blocks + WaveNet + final_layer + conv2 → velocity
    return s2a_dit_run(ctx, hidden.data(), t1.data(), t2.data(), x_flat, T, mel_dim);
}

// CFG-fused DiT forward (CRISPASR_CONFUCIUS4_CFG_FUSE=1): the cond and uncond
// estimator passes run as ONE graph eval per ODE step — seq-concat along time
// with a block-diagonal attention mask, per-arm WaveNet, in-graph blend.
// Returns the CFG-blended velocity (T*mel_dim) row-major, or empty on failure.
static std::vector<float> s2a_dit_forward_cfg_fused(confucius4_tts_context* ctx, const float* x_flat, int T,
                                                    int mel_dim, const float* cond, const float* cond_ref,
                                                    const float* cond_zeros, const float* cond_ref_zeros,
                                                    float timestep, float cfg_rate) {
    const auto& s = ctx->s2a;
    const int dim = s.hp.estimator_hidden_dim;

    // Timestep embeddings are shared by both arms (same t).
    auto t1 = s2a_timestep_embed_cpu(s, timestep, "decoder.estimator.t_embedder", dim);
    auto t2 = s2a_timestep_embed_cpu(s, timestep, "decoder.estimator.t_embedder2", dim);
    if (t1.empty()) {
        fprintf(stderr, "confucius4: timestep embedding failed\n");
        return {};
    }
    if (t2.empty())
        t2 = t1;

    // Per-arm input embeddings, concatenated cond-first along time.
    auto h_cond = s2a_input_embed_cpu(ctx, x_flat, cond_ref, cond, T, mel_dim, /*use_spk=*/true);
    auto h_unc = s2a_input_embed_cpu(ctx, x_flat, cond_ref_zeros, cond_zeros, T, mel_dim, /*use_spk=*/false);
    if (h_cond.empty() || h_unc.empty()) {
        fprintf(stderr, "confucius4: input embedding failed\n");
        return {};
    }

    std::vector<float> hidden2((size_t)2 * T * dim);
    std::memcpy(hidden2.data(), h_cond.data(), h_cond.size() * sizeof(float));
    std::memcpy(hidden2.data() + h_cond.size(), h_unc.data(), h_unc.size() * sizeof(float));

    // Both arms see the same noisy x for the skip_linear input.
    std::vector<float> x_mel2((size_t)2 * T * mel_dim);
    std::memcpy(x_mel2.data(), x_flat, (size_t)T * mel_dim * sizeof(float));
    std::memcpy(x_mel2.data() + (size_t)T * mel_dim, x_flat, (size_t)T * mel_dim * sizeof(float));

    return s2a_dit_run(ctx, hidden2.data(), t1.data(), t2.data(), x_mel2.data(), T, mel_dim, /*fused=*/true, cfg_rate);
}

// ---------------------------------------------------------------------------
// S2A: InterpolateRegulator (length regulator)
// ---------------------------------------------------------------------------
// Python reference: confuciustts/flow/length_regulator.py
//   content_in_proj: Linear(in_channels=1024, channels=512)
//   F.interpolate(x.transpose(1,2), size=target_len, mode="nearest")
//   model = 4 x [Conv1d(512,512,k=3,pad=1), GroupNorm(1,512), Mish]
//           + Conv1d(512, out_channels=512, k=1)
// Weights are named length_regulator.model.{0,3,6,9} (conv k=3),
// {1,4,7,10} (GroupNorm) and 12 (conv k=1); the Mish layers carry no weights.

// Mish activation: x * tanh(softplus(x)).
static void s2a_mish_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) {
        // softplus with the usual large-input guard so exp() cannot overflow
        const float v = x[i];
        const float sp = v > 20.0f ? v : logf(1.0f + expf(v));
        x[i] = v * tanhf(sp);
    }
}

// GroupNorm with num_groups == 1 over a (C, T) channel-first buffer: the mean
// and variance are taken jointly over every channel AND every time step, then
// a per-channel affine is applied.  Matches torch nn.GroupNorm(1, C).
static void s2a_group_norm1_ct(float* x, const float* weight, const float* bias, int C, int T, float eps = 1e-5f) {
    const size_t n = (size_t)C * T;
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += x[i];
        sum_sq += (double)x[i] * x[i];
    }
    const float mean = (float)(sum / (double)n);
    const float var = (float)(sum_sq / (double)n - (double)mean * mean);
    const float inv = 1.0f / sqrtf(var + eps);
    for (int c = 0; c < C; c++) {
        const float w = weight ? weight[c] : 1.0f;
        const float b = bias ? bias[c] : 0.0f;
        float* row = x + (size_t)c * T;
        for (int t = 0; t < T; t++)
            row[t] = (row[t] - mean) * inv * w + b;
    }
}

// Conv1d over a channel-first (C_in, T) buffer with symmetric zero padding.
// `w` is the torch weight (C_out, C_in, K) in row-major order, so the flat
// index of tap k of input channel ci for output channel co is
// co*(C_in*K) + ci*K + k.  Output is (C_out, T) — stride 1, padding K/2.
static void s2a_conv1d_ct(const float* x, const float* w, const float* bias, float* y, int C_in, int C_out, int T,
                          int K) {
    const int pad = K / 2;
    for (int co = 0; co < C_out; co++) {
        float* yr = y + (size_t)co * T;
        const float b = bias ? bias[co] : 0.0f;
        for (int t = 0; t < T; t++)
            yr[t] = b;
        for (int ci = 0; ci < C_in; ci++) {
            const float* xr = x + (size_t)ci * T;
            const float* wr = w + (size_t)co * C_in * K + (size_t)ci * K;
            for (int k = 0; k < K; k++) {
                const float wk = wr[k];
                if (wk == 0.0f)
                    continue;
                // output t reads input (t + k - pad)
                const int lo = std::max(0, pad - k);
                const int hi = std::min(T, T + pad - k);
                for (int t = lo; t < hi; t++)
                    yr[t] += xr[t + k - pad] * wk;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// S2A stage dump (parity harness) — gated by CRISPASR_CONFUCIUS4_DUMP_S2A=<dir>
// ---------------------------------------------------------------------------
// Writes raw little-endian buffers plus a shapes.txt manifest so the Python
// reference (confuciustts/flow) can be driven on exactly the same inputs and
// the per-stage outputs compared.  Off unless the env var is set.

static const char* s2a_dump_dir() {
    const char* d = std::getenv("CRISPASR_CONFUCIUS4_DUMP_S2A");
    return (d && *d) ? d : nullptr;
}

static void s2a_dump_raw(const char* name, const void* data, size_t nbytes, const char* shape) {
    const char* dir = s2a_dump_dir();
    if (!dir)
        return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "confucius4: cannot write dump '%s'\n", path);
        return;
    }
    fwrite(data, 1, nbytes, f);
    fclose(f);

    snprintf(path, sizeof(path), "%s/shapes.txt", dir);
    if (FILE* m = fopen(path, "a")) {
        fprintf(m, "%s\t%s\n", name, shape);
        fclose(m);
    }
    fprintf(stderr, "confucius4: dumped %s %s\n", name, shape);
}

// ---------------------------------------------------------------------------
// S2A: run flow-matching to produce mel from semantic codes
// ---------------------------------------------------------------------------

// Run the S2A conditioning pipeline: semantic codes → cond vector for DiT.
// Returns (lr_out_channels, T_mel) conditioning, or empty on failure.
static std::vector<float> s2a_build_conditioning(confucius4_tts_context* ctx,
                                                 const std::vector<int32_t>& semantic_codes, int T_mel,
                                                 const std::vector<float>& lm_latent = {}) {
    const auto& s = ctx->s2a;
    const auto& hp = s.hp;
    const int T_sem = (int)semantic_codes.size();
    const int vb = ctx->params.verbosity;

    const int sem_emb_dim = hp.semantic_embed_dim; // 1024
    const int lm_dim = hp.lm_latent_dim;           // 1280
    const int lr_in = sem_emb_dim + lm_dim;        // 2304
    const int lr_out = hp.input_size;              // 512 — DiT mu dimension

    // encoder_proj is Linear(2304, lr_in_channels); lr_in_channels is 1024 in the
    // reference config, NOT input_size.  Read it off the weight so a re-converted
    // GGUF with different dims still works.
    const int enc_out = s.encoder_proj_w ? (int)s.encoder_proj_w->ne[1] : lr_out;

    if (vb >= 1)
        fprintf(stderr, "confucius4: S2A conditioning: T_sem=%d, T_mel=%d\n", T_sem, T_mel);

    // Step 1: embed semantic codes → (T_sem, sem_emb_dim)
    // input_embedding: Embedding(8192, 8) → Linear(8, sem_emb_dim) via conv1d
    auto embed_w = s2a_read_f32(s.input_embed_w); // (8192, 8)
    auto proj_w = s2a_read_f32(s.input_proj_w);   // (1, 8, sem_emb_dim) conv1d
    auto proj_b = s2a_read_f32(s.input_proj_b);   // (sem_emb_dim,)

    std::vector<float> sem_emb((size_t)T_sem * sem_emb_dim, 0.0f);
    for (int t = 0; t < T_sem; t++) {
        int code = semantic_codes[t];
        if (code < 0 || code >= 8192)
            code = 0;
        // Lookup embedding: (8,)
        const float* emb_row = embed_w.data() + code * 8;
        // Project 8 → sem_emb_dim via conv1d(kernel=1) = Linear
        for (int d = 0; d < sem_emb_dim; d++) {
            float v = proj_b.empty() ? 0.0f : proj_b[d];
            for (int k = 0; k < 8; k++)
                v += emb_row[k] * proj_w[d * 8 + k];
            sem_emb[t * sem_emb_dim + d] = v;
        }
    }

    // Step 2: concat LM latent + semantic embedding → (T_sem, lr_in=2304)
    // lm_latent is (T_sem * lm_dim) row-major from T2S hidden states.
    // If empty, use zeros (no T2S hidden states available).
    const bool have_lm = !lm_latent.empty() && (int)lm_latent.size() >= T_sem * lm_dim;
    std::vector<float> concat_in((size_t)T_sem * lr_in, 0.0f);
    for (int t = 0; t < T_sem; t++) {
        if (have_lm)
            std::memcpy(concat_in.data() + t * lr_in, lm_latent.data() + t * lm_dim, lm_dim * sizeof(float));
        std::memcpy(concat_in.data() + t * lr_in + lm_dim, sem_emb.data() + t * sem_emb_dim,
                    sem_emb_dim * sizeof(float));
    }
    if (vb >= 1)
        fprintf(stderr, "confucius4: S2A conditioning: LM latent %s\n", have_lm ? "OK" : "zeros (not available)");

    // Step 3: project → (T_sem, enc_out=1024)
    auto enc_w = s2a_read_f32(s.encoder_proj_w); // (enc_out, lr_in)
    auto enc_b = s2a_read_f32(s.encoder_proj_b); // (enc_out,)
    std::vector<float> projected((size_t)T_sem * enc_out);
    s2a_linear(concat_in.data(), enc_w.data(), enc_b.data(), projected.data(), T_sem, lr_in, enc_out);

    // Step 4: length regulator.  The legacy path (linear interpolation of the
    // encoder_proj output, no learned convolutions) is kept behind an env gate
    // for A/B; the default now runs the real InterpolateRegulator.
    const char* env_legacy = std::getenv("CRISPASR_CONFUCIUS4_LR_LEGACY");
    const bool legacy_lr = env_legacy && *env_legacy && *env_legacy != '0';

    if (legacy_lr) {
        std::vector<float> cond((size_t)T_mel * lr_out, 0.0f);
        for (int t = 0; t < T_mel; t++) {
            float src = (float)t * T_sem / T_mel;
            int s0 = std::min((int)src, T_sem - 1);
            int s1 = std::min(s0 + 1, T_sem - 1);
            float frac = src - s0;
            for (int d = 0; d < lr_out; d++)
                cond[t * lr_out + d] = (1.0f - frac) * projected[s0 * enc_out + d] + frac * projected[s1 * enc_out + d];
        }
        if (vb >= 1)
            fprintf(stderr, "confucius4: S2A conditioning: legacy linear-interp regulator\n");
        return cond;
    }

    // 4a. content_in_proj: Linear(enc_out=1024, channels=512) → (T_sem, ch)
    auto cip_w = s2a_read_f32(s2a_find(s, "length_regulator.content_in_proj.weight"));
    auto cip_b = s2a_read_f32(s2a_find(s, "length_regulator.content_in_proj.bias"));
    if (cip_w.empty()) {
        fprintf(stderr, "confucius4: missing length_regulator.content_in_proj.weight\n");
        return {};
    }
    const int ch = (int)(cip_w.size() / (size_t)enc_out); // 512
    std::vector<float> content((size_t)T_sem * ch);
    s2a_linear(projected.data(), cip_w.data(), cip_b.data(), content.data(), T_sem, enc_out, ch);

    // 4b. F.interpolate(..., size=T_mel, mode="nearest") on the channel-first
    //     view.  torch maps output index i to input floor(i * T_sem / T_mel).
    //     `h` is (ch, T_mel) channel-first, which is what the convolutions want.
    std::vector<float> h((size_t)ch * T_mel);
    for (int t = 0; t < T_mel; t++) {
        int src = (int)((float)t * (float)T_sem / (float)T_mel);
        src = std::min(std::max(src, 0), T_sem - 1);
        for (int c = 0; c < ch; c++)
            h[(size_t)c * T_mel + t] = content[(size_t)src * ch + c];
    }

    // 4c. model = 4 x [Conv1d(ch, ch, k=3, pad=1) → GroupNorm(1, ch) → Mish],
    //     then Conv1d(ch, lr_out, k=1).  Weight indices are fixed by the
    //     nn.Sequential layout (Mish carries no parameters).
    const int conv_idx[4] = {0, 3, 6, 9};
    const int gn_idx[4] = {1, 4, 7, 10};
    std::vector<float> tmp((size_t)ch * T_mel);
    for (int b = 0; b < 4; b++) {
        char nm[128];
        snprintf(nm, sizeof(nm), "length_regulator.model.%d.weight", conv_idx[b]);
        auto cw = s2a_read_f32(s2a_find(s, nm));
        snprintf(nm, sizeof(nm), "length_regulator.model.%d.bias", conv_idx[b]);
        auto cb = s2a_read_f32(s2a_find(s, nm));
        if (cw.empty()) {
            fprintf(stderr, "confucius4: missing length_regulator.model.%d.weight\n", conv_idx[b]);
            return {};
        }
        s2a_conv1d_ct(h.data(), cw.data(), cb.empty() ? nullptr : cb.data(), tmp.data(), ch, ch, T_mel, 3);
        h.swap(tmp);

        snprintf(nm, sizeof(nm), "length_regulator.model.%d.weight", gn_idx[b]);
        auto gw = s2a_read_f32(s2a_find(s, nm));
        snprintf(nm, sizeof(nm), "length_regulator.model.%d.bias", gn_idx[b]);
        auto gb = s2a_read_f32(s2a_find(s, nm));
        s2a_group_norm1_ct(h.data(), gw.empty() ? nullptr : gw.data(), gb.empty() ? nullptr : gb.data(), ch, T_mel);

        s2a_mish_inplace(h.data(), (int)h.size());
    }

    auto ow = s2a_read_f32(s2a_find(s, "length_regulator.model.12.weight"));
    auto ob = s2a_read_f32(s2a_find(s, "length_regulator.model.12.bias"));
    if (ow.empty()) {
        fprintf(stderr, "confucius4: missing length_regulator.model.12.weight\n");
        return {};
    }
    std::vector<float> out_ct((size_t)lr_out * T_mel);
    s2a_conv1d_ct(h.data(), ow.data(), ob.empty() ? nullptr : ob.data(), out_ct.data(), ch, lr_out, T_mel, 1);

    // Back to (T_mel, lr_out) row-major — the layout s2a_input_embed_cpu wants.
    std::vector<float> cond((size_t)T_mel * lr_out);
    for (int t = 0; t < T_mel; t++)
        for (int d = 0; d < lr_out; d++)
            cond[(size_t)t * lr_out + d] = out_ct[(size_t)d * T_mel + t];

    if (vb >= 1)
        fprintf(stderr, "confucius4: S2A conditioning: embed+project+regulator OK (%d→%d→%d dims)\n", lr_in, enc_out,
                lr_out);

    if (s2a_dump_dir()) {
        char shp[64];
        snprintf(shp, sizeof(shp), "%d,%d", T_mel, lr_out);
        s2a_dump_raw("cond", cond.data(), cond.size() * sizeof(float), shp);
    }

    return cond;
}

// Run the full S2A flow-matching ODE to produce mel.
// Returns (mel_dim, T_mel) mel spectrogram as float32, or empty on failure.
static std::vector<float> s2a_flow_matching(confucius4_tts_context* ctx, const std::vector<int32_t>& semantic_codes,
                                            const std::vector<float>& lm_latent = {}) {
    const auto& s = ctx->s2a;
    const auto& hp = s.hp;
    const int T_sem = (int)semantic_codes.size();
    const int T_mel = (int)(T_sem * 1.72);    // heuristic from Python (target length)
    const int mel_dim = hp.estimator_mel_dim; // detected from weights (may be 80 or 512)
    // Reference-mel prompt.  MaskedDiffWithXvec.inference prepends T_ref frames
    // of the learned prompt_cond to the conditioning, solves over T_ref + T_mel,
    // and strips the prompt span off the result.
    const int T_ref = (int)ctx->prompt_mel.size() / (mel_dim > 0 ? mel_dim : 1);
    const int T_total = T_ref + T_mel;
    const int vb = ctx->params.verbosity;
    const int n_steps = ctx->params.ode_steps > 0 ? ctx->params.ode_steps : 25;

    // Classifier-free guidance rate.  A zero-initialised params struct must still
    // get the reference default (0.7), so 0 means "unset"; pass a negative rate
    // (or CRISPASR_CONFUCIUS4_CFG_RATE=0) to turn CFG off explicitly.
    float cfg_rate = ctx->params.cfg_rate > 0.0f ? ctx->params.cfg_rate : (ctx->params.cfg_rate < 0.0f ? 0.0f : 0.7f);
    if (const char* env_cfg = std::getenv("CRISPASR_CONFUCIUS4_CFG_RATE"))
        if (*env_cfg)
            cfg_rate = strtof(env_cfg, nullptr);

    if (vb >= 1)
        fprintf(stderr,
                "confucius4: S2A flow-matching: T_sem=%d → T_mel=%d (+%d prompt), mel_dim=%d, %d ODE steps, "
                "cfg=%.2f\n",
                T_sem, T_mel, T_ref, mel_dim, n_steps, cfg_rate);

    if (s2a_dump_dir()) {
        char shp[64];
        snprintf(shp, sizeof(shp), "%d", T_sem);
        s2a_dump_raw("semantic_codes_i32", semantic_codes.data(), semantic_codes.size() * sizeof(int32_t), shp);
        if (!lm_latent.empty()) {
            snprintf(shp, sizeof(shp), "%d,%d", (int)(lm_latent.size() / hp.lm_latent_dim), hp.lm_latent_dim);
            s2a_dump_raw("lm_latent", lm_latent.data(), lm_latent.size() * sizeof(float), shp);
        }
    }

    // Build semantic conditioning → (T_mel, cond_dim=512) row-major
    std::vector<float> cond_target = s2a_build_conditioning(ctx, semantic_codes, T_mel, lm_latent);
    if (cond_target.empty())
        return {};

    const int cond_dim = (int)(cond_target.size() / (size_t)T_mel);

    // cat([prompt_cond expanded to T_ref, cond_target]) → (T_total, cond_dim)
    std::vector<float> cond;
    if (T_ref > 0) {
        auto pc = s2a_read_f32(s.prompt_cond); // learned (1, 1, cond_dim) parameter
        cond.assign((size_t)T_total * cond_dim, 0.0f);
        for (int t = 0; t < T_ref; t++)
            if ((int)pc.size() >= cond_dim)
                std::memcpy(cond.data() + (size_t)t * cond_dim, pc.data(), (size_t)cond_dim * sizeof(float));
        std::memcpy(cond.data() + (size_t)T_ref * cond_dim, cond_target.data(), cond_target.size() * sizeof(float));
    } else {
        cond = std::move(cond_target);
    }

    // Initialize noise: z ~ N(0, 1) of shape (T_mel, mel_dim) row-major
    std::mt19937 rng(ctx->params.seed ? ctx->params.seed : 42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    // MaskedDiffWithXvec.inference passes temperature=1.0 to the CFM decoder.
    // params.temperature is the T2S *sampling* temperature, a different knob.
    float temperature = 1.0f;
    if (const char* env_t = std::getenv("CRISPASR_CONFUCIUS4_S2A_TEMP"))
        if (*env_t)
            temperature = strtof(env_t, nullptr);
    std::vector<float> z((size_t)T_total * mel_dim);
    for (auto& v : z)
        v = dist(rng) * temperature;
    // solve_euler zeroes x over the prompt span before the loop and again after
    // every step.
    std::fill(z.begin(), z.begin() + (size_t)T_ref * mel_dim, 0.0f);

    if (s2a_dump_dir()) {
        char shp[64];
        snprintf(shp, sizeof(shp), "%d,%d", T_total, mel_dim);
        s2a_dump_raw("z_init", z.data(), z.size() * sizeof(float), shp);
    }

    // prompt_x: the reference mel over the prompt span, zero elsewhere.
    std::vector<float> cond_ref((size_t)T_total * mel_dim, 0.0f);
    if (T_ref > 0)
        std::memcpy(cond_ref.data(), ctx->prompt_mel.data(), ctx->prompt_mel.size() * sizeof(float));

    // Zeroed conditioning for the unconditioned CFG pass.  cond_ref is already
    // zero while there is no reference mel, but keep the two distinct so the
    // unconditioned pass stays correct once a speaker prompt is wired in.
    std::vector<float> cond_zeros;
    std::vector<float> cond_ref_zeros;
    if (cfg_rate > 0.0f) {
        cond_zeros.assign(cond.size(), 0.0f);
        cond_ref_zeros.assign(cond_ref.size(), 0.0f);
    }

    // CFG fusion: run cond+uncond as ONE DiT graph eval per ODE step
    // (seq-concat + block-diagonal mask, s2a_dit_forward_cfg_fused).
    // Measured A/B on quiet boxes, roundtrip verbatim in every arm
    // (2026-08-25): Metal q4_k s2a 181→158 s (−12.7% — the 2T matmuls
    // amortize the k-quant dequant that dominates at T); Metal f16 s2a
    // 130→131 s (neutral); CUDA P100 neutral (±1%, gpu-verify run 2, f16 AND
    // q4_k); CPU 658→903 s (loses — the masked 2T attention costs more than
    // the saved graph evals).  Default is therefore ON only on Metal (wins
    // on q4_k, harmless on f16); CRISPASR_CONFUCIUS4_CFG_FUSE=0/1 forces
    // either way.  The parity dumps force it off because the harness expects
    // per-pass semantics.
    // NB: ggml names the Metal DEVICE "MTL0" — match on the registry name via
    // core_cpu_backend::is_metal (the ggml_backend_name("Metal") form never
    // fires; red-verified 2026-08-25).
    bool cfg_fuse = ctx->backend && !core_cpu_backend::is_cpu(ctx->backend) && core_cpu_backend::is_metal(ctx->backend);
    if (const char* env_fuse = std::getenv("CRISPASR_CONFUCIUS4_CFG_FUSE"))
        if (*env_fuse)
            cfg_fuse = atoi(env_fuse) != 0;
    if (s2a_dump_dir())
        cfg_fuse = false;
    if (vb >= 1 && cfg_fuse && cfg_rate > 0.0f)
        fprintf(stderr, "confucius4: S2A CFG fusion ON (1 fused DiT pass per step)\n");

    // Time schedule.  MaskedDiffWithXvecConfig.cfm_t_scheduler defaults to
    // "linear" and config/inference_config.yaml does not override it, so the
    // shipped model solves over a plain linspace(0, 1, n_steps + 1).  (The
    // ConditionalCFM.__init__ signature says "cosine", but flow.py passes the
    // config value, not the signature default.)  The cosine schedule stays
    // available for A/B behind CRISPASR_CONFUCIUS4_T_SCHEDULE=cosine.
    bool cosine_schedule = false;
    if (const char* env_sched = std::getenv("CRISPASR_CONFUCIUS4_T_SCHEDULE"))
        cosine_schedule = (std::strcmp(env_sched, "cosine") == 0);

    std::vector<float> t_span(n_steps + 1);
    for (int i = 0; i <= n_steps; i++) {
        float lin = (float)i / (float)n_steps;
        t_span[i] = cosine_schedule ? 1.0f - cosf(lin * 1.5707963f) : lin;
    }
    if (vb >= 1)
        fprintf(stderr, "confucius4: S2A time schedule: %s\n", cosine_schedule ? "cosine" : "linear");

    // Euler ODE: z = z + dt * velocity_from_DiT
    for (int step = 1; step <= n_steps; step++) {
        float dt = t_span[step] - t_span[step - 1];
        float t = t_span[step - 1];

        std::vector<float> velocity;
        if (cfg_rate > 0.0f && cfg_fuse) {
            // Fused CFG: both arms in one graph eval, blend done in-graph.
            velocity = s2a_dit_forward_cfg_fused(ctx, z.data(), T_total, mel_dim, cond.data(), cond_ref.data(),
                                                 cond_zeros.data(), cond_ref_zeros.data(), t, cfg_rate);
            if (velocity.empty()) {
                fprintf(stderr, "confucius4: fused DiT forward failed at step %d\n", step);
                return {};
            }
        } else {
            // Conditioned pass.
            velocity = s2a_dit_forward(ctx, z.data(), T_total, mel_dim, cond.data(), cond_ref.data(), t);
            if (velocity.empty()) {
                fprintf(stderr, "confucius4: DiT forward failed at step %d\n", step);
                return {};
            }

            // Classifier-free guidance: a second pass with mu, the reference mel and
            // the speaker embedding all zeroed, blended as
            //   v = (1 + cfg) * v_cond - cfg * v_uncond
            // (confuciustts/flow/flow_matching.py, solve_euler).
            if (cfg_rate > 0.0f) {
                // The uncond arm runs over the SAME T_total frames (solve_euler
                // feeds x to both halves and zeroes only mu/prompt_x/spks). This
                // used to pass T_mel — harmless without a prompt (T_total==T_mel),
                // but with one the blend read v_uncond OUT OF BOUNDS past the
                // first T_mel frames and every conditioned run decayed to silence.
                auto v_uncond = s2a_dit_forward(ctx, z.data(), T_total, mel_dim, cond_zeros.data(),
                                                cond_ref_zeros.data(), t, /*use_spk=*/false);
                if (v_uncond.empty()) {
                    fprintf(stderr, "confucius4: DiT uncond forward failed at step %d\n", step);
                    return {};
                }
                for (size_t i = 0; i < velocity.size(); i++)
                    velocity[i] = (1.0f + cfg_rate) * velocity[i] - cfg_rate * v_uncond[i];
            }
        }

        // Per-step dump: the state the velocity was computed FROM, and the
        // (post-CFG) velocity itself.  Lets the parity harness recompute each
        // step's velocity from the identical input and find the FIRST
        // divergence, instead of only seeing accumulated error in the mel.
        if (s2a_dump_dir()) {
            char nm[64], shp[64];
            snprintf(shp, sizeof(shp), "%d,%d", T_total, mel_dim);
            snprintf(nm, sizeof(nm), "z_step_%02d", step);
            s2a_dump_raw(nm, z.data(), z.size() * sizeof(float), shp);
            snprintf(nm, sizeof(nm), "v_step_%02d", step);
            s2a_dump_raw(nm, velocity.data(), velocity.size() * sizeof(float), shp);
        }

        for (size_t i = 0; i < z.size(); i++)
            z[i] += dt * velocity[i];

        // solve_euler: x[:, :, :prompt_len] = 0 after every step
        if (T_ref > 0)
            std::fill(z.begin(), z.begin() + (size_t)T_ref * mel_dim, 0.0f);

        if (vb >= 2 && (step <= 3 || step == n_steps)) {
            // Log first few velocity magnitudes for debugging
            float vmax = 0.0f;
            for (size_t i = 0; i < std::min(velocity.size(), (size_t)1000); i++)
                vmax = std::max(vmax, fabsf(velocity[i]));
            fprintf(stderr, "confucius4: ODE step %d/%d: t=%.4f dt=%.4f |v|_max=%.4f\n", step, n_steps, t, dt, vmax);
        }
    }

    if (vb >= 1)
        fprintf(stderr, "confucius4: S2A flow-matching done (%d steps, mel_dim=%d)\n", n_steps, mel_dim);

    // Strip the prompt span: generated_mel = generated_mel[:, :, T_ref:]
    if (T_ref > 0)
        z.erase(z.begin(), z.begin() + (size_t)T_ref * mel_dim);

    if (s2a_dump_dir()) {
        char shp[64];
        snprintf(shp, sizeof(shp), "%d,%d", T_mel, mel_dim);
        s2a_dump_raw("mel", z.data(), z.size() * sizeof(float), shp);
    }

    return z; // (T_mel * mel_dim) row-major
}

float* confucius4_tts_synthesize(confucius4_tts_context* ctx, const char* text, const char* lang, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;

    const int vb = ctx->params.verbosity;

    load_conditioning_from_env(ctx);

    // Step 1: Tokenize text
    std::vector<int32_t> text_ids;
    const char* env_ids = std::getenv("CRISPASR_CONFUCIUS4_TEXT_IDS");
    if (env_ids && *env_ids) {
        // Env var override: comma-separated pre-tokenized IDs (testing path)
        std::string s(env_ids);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            if (comma == std::string::npos)
                comma = s.size();
            text_ids.push_back(std::stoi(s.substr(pos, comma - pos)));
            pos = comma + 1;
        }
        if (vb >= 1)
            fprintf(stderr, "confucius4: using %zu text IDs from CRISPASR_CONFUCIUS4_TEXT_IDS\n", text_ids.size());
    } else if (ctx->has_bpe_tokenizer) {
        // BPE tokenizer from GGUF vocab (tokenizer.ggml.tokens + merges)
        // Format: "You are a helpful assistant. {lang_token}:{text}"
        // Reference LANGUAGE_TOKEN_MAP (confuciustts/utils/text_utils.py):
        // the prompts are CHINESE for every language — the model was trained on
        // these exact strings, an invented English prompt derails the T2S.
        static const std::unordered_map<std::string, const char*> k_lang_token_map = {
            {"zh", "请用中文朗读接下来的文字"},         {"ja", "请用日语朗读接下来的文字"},
            {"ko", "请用韩语朗读接下来的文字"},         {"vi", "请用越南语朗读接下来的文字"},
            {"th", "请用泰语朗读接下来的文字"},         {"id", "请用印尼语朗读接下来的文字"},
            {"ms", "请用马来语朗读接下来的文字"},       {"tl", "请用菲律宾语朗读接下来的文字"},
            {"my", "请用缅甸语朗读接下来的文字"},       {"km", "请用高棉语朗读接下来的文字"},
            {"lo", "请用老挝语朗读接下来的文字"},       {"hi", "请用印地语朗读接下来的文字"},
            {"bn", "请用孟加拉语朗读接下来的文字"},     {"ta", "请用泰米尔语朗读接下来的文字"},
            {"te", "请用泰卢固语朗读接下来的文字"},     {"mr", "请用马拉地语朗读接下来的文字"},
            {"gu", "请用古吉拉特语朗读接下来的文字"},   {"kn", "请用卡纳达语朗读接下来的文字"},
            {"ml", "请用马拉雅拉姆语朗读接下来的文字"}, {"pa", "请用旁遮普语朗读接下来的文字"},
            {"ur", "请用乌尔都语朗读接下来的文字"},     {"ne", "请用尼泊尔语朗读接下来的文字"},
            {"si", "请用僧伽罗语朗读接下来的文字"},     {"en", "请用英文朗读接下来的文字"},
            {"de", "请用德语朗读接下来的文字"},         {"nl", "请用荷兰语朗读接下来的文字"},
            {"sv", "请用瑞典语朗读接下来的文字"},       {"da", "请用丹麦语朗读接下来的文字"},
            {"no", "请用挪威语朗读接下来的文字"},       {"nb", "请用挪威语朗读接下来的文字"},
            {"nn", "请用挪威语朗读接下来的文字"},       {"is", "请用冰岛语朗读接下来的文字"},
            {"af", "请用南非荷兰语朗读接下来的文字"},   {"lb", "请用卢森堡语朗读接下来的文字"},
            {"fy", "请用弗里斯兰语朗读接下来的文字"},   {"fr", "请用法语朗读接下来的文字"},
            {"es", "请用西班牙语朗读接下来的文字"},     {"pt", "请用葡萄牙语朗读接下来的文字"},
            {"it", "请用意大利语朗读接下来的文字"},     {"ro", "请用罗马尼亚语朗读接下来的文字"},
            {"ca", "请用加泰罗尼亚语朗读接下来的文字"}, {"gl", "请用加利西亚语朗读接下来的文字"},
            {"oc", "请用奥克语朗读接下来的文字"},       {"la", "请用拉丁语朗读接下来的文字"},
            {"ru", "请用俄语朗读接下来的文字"},         {"uk", "请用乌克兰语朗读接下来的文字"},
            {"pl", "请用波兰语朗读接下来的文字"},       {"cs", "请用捷克语朗读接下来的文字"},
            {"sk", "请用斯洛伐克语朗读接下来的文字"},   {"bg", "请用保加利亚语朗读接下来的文字"},
            {"sr", "请用塞尔维亚语朗读接下来的文字"},   {"hr", "请用克罗地亚语朗读接下来的文字"},
            {"sl", "请用斯洛文尼亚语朗读接下来的文字"}, {"mk", "请用马其顿语朗读接下来的文字"},
            {"bs", "请用波斯尼亚语朗读接下来的文字"},   {"be", "请用白俄罗斯语朗读接下来的文字"},
            {"lt", "请用立陶宛语朗读接下来的文字"},     {"lv", "请用拉脱维亚语朗读接下来的文字"},
            {"fi", "请用芬兰语朗读接下来的文字"},       {"et", "请用爱沙尼亚语朗读接下来的文字"},
            {"hu", "请用匈牙利语朗读接下来的文字"},     {"ga", "请用爱尔兰语朗读接下来的文字"},
            {"cy", "请用威尔士语朗读接下来的文字"},     {"gd", "请用苏格兰盖尔语朗读接下来的文字"},
            {"br", "请用布列塔尼语朗读接下来的文字"},   {"el", "请用希腊语朗读接下来的文字"},
            {"sq", "请用阿尔巴尼亚语朗读接下来的文字"}, {"eu", "请用巴斯克语朗读接下来的文字"},
            {"mt", "请用马耳他语朗读接下来的文字"},     {"tr", "请用土耳其语朗读接下来的文字"},
            {"az", "请用阿塞拜疆语朗读接下来的文字"},   {"kk", "请用哈萨克语朗读接下来的文字"},
            {"uz", "请用乌兹别克语朗读接下来的文字"},   {"tk", "请用土库曼语朗读接下来的文字"},
            {"ky", "请用吉尔吉斯语朗读接下来的文字"},   {"tt", "请用鞑靼语朗读接下来的文字"},
            {"ar", "请用阿拉伯语朗读接下来的文字"},     {"he", "请用希伯来语朗读接下来的文字"},
            {"am", "请用阿姆哈拉语朗读接下来的文字"},   {"fa", "请用波斯语朗读接下来的文字"},
            {"ps", "请用普什图语朗读接下来的文字"},     {"ku", "请用库尔德语朗读接下来的文字"},
            {"tg", "请用塔吉克语朗读接下来的文字"},     {"ka", "请用格鲁吉亚语朗读接下来的文字"},
            {"hy", "请用亚美尼亚语朗读接下来的文字"},   {"sw", "请用斯瓦希里语朗读接下来的文字"},
            {"yo", "请用约鲁巴语朗读接下来的文字"},     {"ha", "请用豪萨语朗读接下来的文字"},
            {"ig", "请用伊博语朗读接下来的文字"},       {"zu", "请用祖鲁语朗读接下来的文字"},
            {"xh", "请用科萨语朗读接下来的文字"},       {"mn", "请用蒙古语朗读接下来的文字"},
            {"eo", "请用世界语朗读接下来的文字"},
        };
        std::string lang_token;
        {
            std::string lc = (lang && *lang) ? lang : "zh"; // reference CLI default
            auto it = k_lang_token_map.find(lc);
            // Reference fallback: f"请用{lang}朗读接下来的文字"
            lang_token = it != k_lang_token_map.end() ? it->second : ("请用" + lc + "朗读接下来的文字");
        }
        std::string formatted = std::string("You are a helpful assistant. ") + lang_token + ":" + text;
        // LlamaTokenizerFast recipe (checkpoints/tokenizer.json + config):
        // Metaspace (replace ' '→▁, prepend ▁ with prepend_scheme="first"),
        // SP-BPE merges with byte_fallback, then <s>...</s> because the
        // tokenizer_config sets add_bos_token=True AND add_eos_token=True.
        std::string processed = "\xe2\x96\x81"; // leading ▁
        for (const char* p = formatted.c_str(); *p; p++)
            processed += (*p == ' ') ? std::string("\xe2\x96\x81") : std::string(1, *p);
        auto find_id = [&](const char* t, int32_t fallback) {
            auto it = ctx->bpe_token_to_id.find(t);
            return it != ctx->bpe_token_to_id.end() ? it->second : fallback;
        };
        text_ids.push_back(find_id("<s>", 1));
        auto body = core_spm_bpe::bpe_merge(processed, ctx->bpe_token_to_id, ctx->bpe_merge_rank);
        text_ids.insert(text_ids.end(), body.begin(), body.end());
        text_ids.push_back(find_id("</s>", 2));
        if (vb >= 1)
            fprintf(stderr, "confucius4: tokenized '%s' → %zu tokens (SP-BPE + bos/eos)\n", text, text_ids.size());
    } else {
        fprintf(stderr, "confucius4: no tokenizer. Set CRISPASR_CONFUCIUS4_TEXT_IDS=id1,id2,... "
                        "or bake vocab into the GGUF.\n");
        *out_n_samples = 0;
        return nullptr;
    }

    // Step 2: T2S decode → semantic codes + LM latent (hidden states for S2A)
    std::vector<float> lm_latent;
    std::vector<int32_t> semantic_codes = t2s_decode(ctx, text_ids, &lm_latent);
    if (semantic_codes.empty()) {
        if (vb >= 1)
            fprintf(stderr, "confucius4: T2S produced no semantic codes\n");
        *out_n_samples = 0;
        return nullptr;
    }

    // Step 3: S2A flow-matching → mel
    // The S2A pipeline (when implemented) will:
    //   1. Embed semantic codes: Embedding(8192,8) → Linear(8,1024)
    //   2. Concat with LM latent: cat([lm_latent(1280), sem_emb(1024)]) → Linear(2304,1024)
    //   3. Length regulate: conv upsample to target_mel_len = int(T_semantic * 1.72)
    //   4. Prepend reference mel prompt condition
    //   5. Run 25-step Euler ODE through DiT (13L, AdaLN, RoPE, U-Net skips)
    //      with CFG (2× forward per step: cond + uncond)
    //   6. WaveNet final layer (8L dilated conv, gated activation)
    //   7. Strip prompt portion → mel (80, T_target)
    //
    // Step 4: BigVGAN vocoder → PCM @ 22050 Hz (external companion GGUF)
    if (!ctx->s2a.loaded) {
        fprintf(stderr, "confucius4: generated %zu semantic codes (S2A not loaded — pass --codec-model)\n",
                semantic_codes.size());
        *out_n_samples = 0;
        return nullptr;
    }

    // Run S2A flow-matching → mel
    if (vb >= 1 && !lm_latent.empty())
        fprintf(stderr, "confucius4: LM latent: %zu floats (%zu tokens × %d dim)\n", lm_latent.size(),
                lm_latent.size() / ctx->t2s.hp.model_dim, ctx->t2s.hp.model_dim);
    std::vector<float> mel = s2a_flow_matching(ctx, semantic_codes, lm_latent);
    if (mel.empty()) {
        fprintf(stderr, "confucius4: S2A flow-matching failed\n");
        *out_n_samples = 0;
        return nullptr;
    }

    const int mel_dim = ctx->s2a.hp.estimator_mel_dim;
    const int T_mel = (int)(mel.size() / mel_dim);

    // Step 4: BigVGAN vocoder → PCM @ 22050 Hz
    const int sr = ctx->t2s.hp.sample_rate; // 22050
    float* pcm = nullptr;
    int n_pcm = 0;

    if (ctx->vocoder) {
        // mel is (T_mel * mel_dim) row-major = (T_mel, mel_dim).
        // indextts_voc_generate expects [T, gpt_dim] row-major, where gpt_dim=80 (mel_dim).
        pcm = indextts_voc_generate(ctx->vocoder, mel.data(), T_mel, nullptr, &n_pcm);
        if (pcm && vb >= 1)
            fprintf(stderr, "confucius4: BigVGAN: %d mel frames → %d PCM samples @ %d Hz (%.2fs)\n", T_mel, n_pcm, sr,
                    (float)n_pcm / sr);
    }

    if (!pcm) {
        // Fallback: output silence at the right duration
        const int hop = 256;
        n_pcm = T_mel * hop;
        pcm = (float*)calloc(n_pcm, sizeof(float));
        if (!pcm) {
            *out_n_samples = 0;
            return nullptr;
        }
        if (vb >= 1)
            fprintf(stderr,
                    "confucius4: output: %d mel frames → %d PCM samples @ %d Hz (%.2fs, silence — no vocoder)\n", T_mel,
                    n_pcm, sr, (float)n_pcm / sr);
    }

    *out_n_samples = n_pcm;
    return pcm;
}

void confucius4_tts_pcm_free(float* pcm) {
    free(pcm);
}

void confucius4_tts_free(confucius4_tts_context* ctx) {
    if (!ctx)
        return;

    kv_free(ctx->kv);
    if (ctx->w2v_ctx)
        sidon_free(ctx->w2v_ctx);
    // Free vocoder
    if (ctx->vocoder)
        indextts_voc_free(ctx->vocoder);
    // Free S2A
    if (ctx->s2a.buf_wn)
        ggml_backend_buffer_free(ctx->s2a.buf_wn);
    if (ctx->s2a.ctx_wn)
        ggml_free(ctx->s2a.ctx_wn);
    if (ctx->s2a.buf_w)
        ggml_backend_buffer_free(ctx->s2a.buf_w);
    if (ctx->s2a.ctx_w)
        ggml_free(ctx->s2a.ctx_w);
    // Free T2S
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

int confucius4_tts_sample_rate(const confucius4_tts_context* ctx) {
    return ctx ? ctx->t2s.hp.sample_rate : 22050;
}
