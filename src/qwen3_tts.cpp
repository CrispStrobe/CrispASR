// qwen3_tts.cpp — runtime for Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base.
//
// Status (April 2026):
//
//   ✓ talker forward (28L Qwen3 with Q/K-norm, single-axis NEOX RoPE):
//     prefill on a text prompt + autoregressive decode of codebook-0
//     via the talker's `codec_head`. Reuses core_attn::kv_self_attn
//     and core_ffn::swiglu — same backbone as qwen3_asr's LLM tower.
//
//   ✗ code_predictor (5L Qwen3, 15 separate codec_embedding / lm_head
//     pairs for codebooks 1..15): weights are loaded into the model
//     struct but the AR loop that fills in codebooks 1..15 each step
//     is not yet wired. Without it, the rendered audio has only 1 of
//     16 codebooks active and will sound noisy. PLAN #52 step 2.
//
//   ✗ codec decoder (8L sliding-window transformer + 1D-conv up-sample
//     stack to 24 kHz waveform, in the separate Tokenizer-12Hz repo):
//     not yet loaded. `qwen3_tts_synthesize` therefore returns
//     nullptr; use `qwen3_tts_synthesize_codes` to get the codebook-0
//     stream you can render via the HF python codec. PLAN #52 step 3.
//
//   ✗ speaker_encoder (ECAPA-style TDNN + Res2Net + ASP for voice
//     cloning): weights are loaded into the model struct but the
//     forward pass + voice-prompt → 1024-d embedding splice into the
//     prefill is not yet wired. PLAN #52 step 4.
//
// Architecture (from Qwen3-TTS-12Hz-{0.6B,1.7B}-Base config.json,
// confirmed against the safetensors keys):
//
//   Talker (autoregressive LM, generates codebook-0 of 16-CB RVQ):
//     - 28-layer Qwen3 (1024d / 2048d for 0.6B / 1.7B)
//     - 16Q / 8KV / head_dim 128, SiLU SwiGLU, RoPE theta 1e6
//     - mrope_section [24, 20, 20]; for the text-only / pure-AR
//       prefill we run today, the 3 mrope axes collapse to a single
//       stream so single-axis NEOX RoPE is mathematically equivalent.
//       Multi-axis RoPE is wired in via core_attn once the diff
//       harness flags a discrepancy (see CrispEmbed decoder_embed.cpp
//       for the same observation on BidirLM-Omni).
//     - dual embedding: text via `text_embedding` (151936×2048) +
//       `text_proj` resize MLP (2048→2048→1024) for prefill; audio
//       codes via `codec_embedding` (3072×1024) for decode.
//     - output head: `codec_head` (1024→3072) → codebook-0 logits.
//
//   Code predictor (small AR LM, fills codebooks 1..15 per step):
//     - 5-layer Qwen3 (1024d), max-length 20 codes per group
//     - 15 separate codec_embedding tables and 15 separate lm_heads
//     - vocab 2048 per codebook (codec uses 2048-entry codebooks)
//
//   Codec (Qwen/Qwen3-TTS-Tokenizer-12Hz, separate repo):
//     - 8L encoder + 8L decoder transformer (hidden 512)
//     - acoustic RVQ (32 layers) + semantic RVQ (1 layer)
//     - encoder/decoder up-/downsample = 1920, 12.5 fps @ 24 kHz

#include "qwen3_tts.h"

#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Debug / regression knobs (PLAN #52 step 4 methodology)
//
//   QWEN3_TTS_BENCH=1     — print per-stage wall-clock timings on stderr.
//   QWEN3_TTS_DEBUG=1     — verbose per-step trace (prompt ids, sampled
//                           code at every AR step, stop reason).
//   QWEN3_TTS_DUMP_DIR=/d — dump key intermediate tensors to /d as
//                           binary float32 files: text_proj_out.bin,
//                           talker_prefill_logits.bin, talker_codes.bin.
//                           Only set when investigating a specific run;
//                           the dump itself is non-zero overhead.
//
// These knobs follow the existing CrispASR pattern (GEMMA4_E2B_BENCH,
// VIBEVOICE_TTS_DUMP, OMNIASR_DUMP_DIR ...) so the regression harness
// can flip them per-call without rebuilding.
// ---------------------------------------------------------------------------

bool env_bool(const char* k) {
    const char* v = std::getenv(k);
    return v && *v && std::strcmp(v, "0") != 0;
}
const char* env_str(const char* k) {
    const char* v = std::getenv(k);
    return (v && *v) ? v : nullptr;
}
double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
}
void dump_f32(const char* dir, const char* name, const float* data, size_t n) {
    if (!dir || !data || !n)
        return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "qwen3_tts: dump open '%s' failed\n", path);
        return;
    }
    std::fwrite(data, sizeof(float), n, f);
    std::fclose(f);
    fprintf(stderr, "qwen3_tts: dumped %s (%zu floats)\n", path, n);
}
void dump_i32(const char* dir, const char* name, const int32_t* data, size_t n) {
    if (!dir || !data || !n)
        return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "qwen3_tts: dump open '%s' failed\n", path);
        return;
    }
    std::fwrite(data, sizeof(int32_t), n, f);
    std::fclose(f);
    fprintf(stderr, "qwen3_tts: dumped %s (%zu i32)\n", path, n);
}

struct g3t_hp {
    // Talker (Qwen3 backbone)
    uint32_t n_layers = 28;
    uint32_t d_model = 1024; // 0.6B; 2048 for 1.7B
    uint32_t n_heads = 16;
    uint32_t n_kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 3072;     // 0.6B; 6144 for 1.7B
    uint32_t vocab_size = 3072; // audio code vocabulary
    uint32_t text_vocab_size = 151936;
    uint32_t text_hidden_size = 2048;
    uint32_t n_code_groups = 16;
    uint32_t max_pos = 32768;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;
    bool rope_interleaved = true;
    std::vector<uint32_t> mrope_section; // [24, 20, 20]

    // Code predictor
    uint32_t cp_n_layers = 5;
    uint32_t cp_d_model = 1024;
    uint32_t cp_n_heads = 16;
    uint32_t cp_n_kv_heads = 8;
    uint32_t cp_vocab_size = 2048;
    uint32_t cp_max_length = 20;
    uint32_t cp_n_code_groups = 16;

    // Speaker encoder
    uint32_t spk_enc_dim = 1024;
    uint32_t spk_sample_rate = 24000;

    // Token sentinels (text-side)
    uint32_t tts_bos_id = 151672;
    uint32_t tts_eos_id = 151673;
    uint32_t tts_pad_id = 151671;
    uint32_t im_start_id = 151644;
    uint32_t im_end_id = 151645;
    uint32_t assistant_id = 77091;

    // Audio-code sentinels
    uint32_t codec_bos_id = 2149;
    uint32_t codec_eos_id = 2150;
    uint32_t codec_pad_id = 2148;
    uint32_t codec_think_id = 2154;
    uint32_t codec_nothink_id = 2155;
    uint32_t codec_think_bos_id = 2156;
    uint32_t codec_think_eos_id = 2157;
};

struct g3t_layer {
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

struct g3t_talker {
    // Embeddings
    ggml_tensor* token_embd_w = nullptr;      // (3072, 1024) audio codes
    ggml_tensor* token_embd_text_w = nullptr; // (151936, 2048) text

    // text_projection = TalkerResizeMLP: 2048 → 2048 (with bias) → SiLU → 1024 (with bias)
    ggml_tensor* text_proj_fc1_w = nullptr;
    ggml_tensor* text_proj_fc1_b = nullptr;
    ggml_tensor* text_proj_fc2_w = nullptr;
    ggml_tensor* text_proj_fc2_b = nullptr;

    std::vector<g3t_layer> blocks;

    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* codec_head_w = nullptr; // (1024, 3072) — codebook-0 logits
};

struct g3t_code_predictor {
    // 15 codec_embedding tables (codebooks 1..15) and 15 lm_heads.
    // Layout: index i corresponds to codebook (i+1) since codebook-0
    // comes from the talker's codec_head.
    std::vector<ggml_tensor*> codec_embd; // size 15
    std::vector<ggml_tensor*> lm_head;    // size 15
    std::vector<g3t_layer> blocks;        // 5 layers
    ggml_tensor* output_norm_w = nullptr;
};

struct g3t_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" → rank
};

} // namespace

struct qwen3_tts_context {
    qwen3_tts_context_params params{};
    int n_threads = 4;

    g3t_hp hp;
    g3t_talker talker;
    g3t_code_predictor code_pred;
    g3t_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<uint8_t> compute_meta;

    // Talker KV cache: (head_dim, max_ctx, n_kv_heads, n_layers).
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;

    std::string codec_path;
    std::string voice_prompt_path;
};

// ---------------------------------------------------------------------------
// Loader helpers
// ---------------------------------------------------------------------------

namespace {

ggml_tensor* try_get(qwen3_tts_context* c, const char* name) {
    auto it = c->tensors.find(name);
    return it == c->tensors.end() ? nullptr : it->second;
}

ggml_tensor* require(qwen3_tts_context* c, const char* name) {
    auto* t = try_get(c, name);
    if (!t)
        fprintf(stderr, "qwen3_tts: required tensor missing: %s\n", name);
    return t;
}

bool load_talker(qwen3_tts_context* c) {
    auto& t = c->talker;
    t.token_embd_w = require(c, "talker.token_embd.weight");
    t.token_embd_text_w = require(c, "talker.token_embd_text.weight");
    t.text_proj_fc1_w = require(c, "talker.text_proj.fc1.weight");
    t.text_proj_fc1_b = require(c, "talker.text_proj.fc1.bias");
    t.text_proj_fc2_w = require(c, "talker.text_proj.fc2.weight");
    t.text_proj_fc2_b = require(c, "talker.text_proj.fc2.bias");
    t.output_norm_w = require(c, "talker.output_norm.weight");
    t.codec_head_w = require(c, "talker.output.weight");
    if (!t.token_embd_w || !t.token_embd_text_w || !t.text_proj_fc1_w || !t.output_norm_w || !t.codec_head_w)
        return false;
    t.blocks.resize(c->hp.n_layers);
    char buf[128];
    for (uint32_t i = 0; i < c->hp.n_layers; i++) {
        auto& b = t.blocks[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "talker.blk.%u.%s", i, suf);
            return require(c, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_q_w = get("attn_q.weight");
        b.attn_k_w = get("attn_k.weight");
        b.attn_v_w = get("attn_v.weight");
        b.attn_output_w = get("attn_output.weight");
        b.attn_q_norm_w = get("attn_q_norm.weight");
        b.attn_k_norm_w = get("attn_k_norm.weight");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn_gate.weight");
        b.ffn_up_w = get("ffn_up.weight");
        b.ffn_down_w = get("ffn_down.weight");
        if (!b.attn_q_w || !b.ffn_gate_w)
            return false;
    }
    return true;
}

bool load_code_predictor(qwen3_tts_context* c) {
    auto& p = c->code_pred;
    p.output_norm_w = try_get(c, "code_pred.output_norm.weight");
    p.codec_embd.resize(c->hp.cp_n_code_groups - 1);
    p.lm_head.resize(c->hp.cp_n_code_groups - 1);
    char buf[128];
    for (uint32_t j = 0; j + 1 < c->hp.cp_n_code_groups; j++) {
        snprintf(buf, sizeof(buf), "code_pred.token_embd.%u.weight", j);
        p.codec_embd[j] = try_get(c, buf);
        snprintf(buf, sizeof(buf), "code_pred.output.%u.weight", j);
        p.lm_head[j] = try_get(c, buf);
    }
    p.blocks.resize(c->hp.cp_n_layers);
    for (uint32_t i = 0; i < c->hp.cp_n_layers; i++) {
        auto& b = p.blocks[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "code_pred.blk.%u.%s", i, suf);
            return try_get(c, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_q_w = get("attn_q.weight");
        b.attn_k_w = get("attn_k.weight");
        b.attn_v_w = get("attn_v.weight");
        b.attn_output_w = get("attn_output.weight");
        b.attn_q_norm_w = get("attn_q_norm.weight");
        b.attn_k_norm_w = get("attn_k_norm.weight");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn_gate.weight");
        b.ffn_up_w = get("ffn_up.weight");
        b.ffn_down_w = get("ffn_down.weight");
    }
    // Code predictor weights are optional for the talker-only path
    // we expose today; report as a debug line, not an error.
    return true;
}

uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d) {
    int64_t i = gguf_find_key(g, k);
    return i >= 0 ? gguf_get_val_u32(g, i) : d;
}
float kv_f32(gguf_context* g, const char* k, float d) {
    int64_t i = gguf_find_key(g, k);
    return i >= 0 ? gguf_get_val_f32(g, i) : d;
}
bool kv_bool(gguf_context* g, const char* k, bool d) {
    int64_t i = gguf_find_key(g, k);
    return i >= 0 ? gguf_get_val_bool(g, i) : d;
}

void register_qwen_specials(g3t_vocab& v) {
    // Same family as qwen3_asr — the converter writes vocab.json (151 643
    // regular tokens) but the Qwen2/Qwen3 special-token block (151 644+)
    // lives in tokenizer_config.json's added_tokens which the converter
    // doesn't propagate. Patch the canonical strings in so the prompt
    // builder can look up <|im_start|> etc.
    struct SP {
        int id;
        const char* text;
    };
    static const SP specials[] = {
        {151643, "<|endoftext|>"},        {151644, "<|im_start|>"},       {151645, "<|im_end|>"},
        {151646, "<|object_ref_start|>"}, {151647, "<|object_ref_end|>"}, {151648, "<|box_start|>"},
        {151649, "<|box_end|>"},          {151650, "<|quad_start|>"},     {151651, "<|quad_end|>"},
        {151652, "<|vision_start|>"},     {151653, "<|vision_end|>"},     {151654, "<|vision_pad|>"},
        {151655, "<|image_pad|>"},        {151656, "<|video_pad|>"},      {151669, "<|audio_start|>"},
        {151670, "<|audio_end|>"},        {151671, "<|tts_pad|>"},        {151672, "<|tts_bos|>"},
        {151673, "<|tts_eos|>"},          {151676, "<|audio_pad|>"},
    };
    for (const auto& sp : specials) {
        if (sp.id < (int)v.id_to_token.size()) {
            auto old_it = v.token_to_id.find(v.id_to_token[sp.id]);
            if (old_it != v.token_to_id.end() && old_it->second == sp.id)
                v.token_to_id.erase(old_it);
            v.id_to_token[sp.id] = sp.text;
            v.token_to_id[sp.text] = sp.id;
        }
    }
}

bool kv_alloc(qwen3_tts_context* c, int max_ctx) {
    if (c->kv_k)
        return true;
    const auto& hp = c->hp;
    const int hd = (int)hp.head_dim;
    const int n_kv = (int)hp.n_kv_heads;
    const int n_lay = (int)hp.n_layers;
    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    c->kv_ctx = ggml_init(kp);
    c->kv_k = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    c->kv_v = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(c->kv_k, "kv_k");
    ggml_set_name(c->kv_v, "kv_v");
    const size_t kb = ggml_nbytes(c->kv_k), vb = ggml_nbytes(c->kv_v);
    c->kv_buf = ggml_backend_alloc_buffer(c->backend, kb + vb);
    if (!c->kv_buf)
        return false;
    char* base = (char*)ggml_backend_buffer_get_base(c->kv_buf);
    ggml_backend_tensor_alloc(c->kv_buf, c->kv_k, base);
    ggml_backend_tensor_alloc(c->kv_buf, c->kv_v, base + kb);
    c->kv_max_ctx = max_ctx;
    if (c->params.verbosity >= 1) {
        fprintf(stderr, "qwen3_tts: kv cache %d MiB (head_dim=%d max_ctx=%d n_kv=%d n_layers=%d)\n",
                (int)((kb + vb) / 1048576), hd, max_ctx, n_kv, n_lay);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

// Embed text input ids using `talker.token_embd_text` (151936, 2048) and
// project them down to `d_model` (1024) via the TalkerResizeMLP — two
// linear layers with biases and a SiLU in between.
ggml_cgraph* build_graph_embed_text(qwen3_tts_context* c, int n_tokens) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);
    ggml_tensor* h = ggml_get_rows(ctx0, c->talker.token_embd_text_w, ids); // (T, 2048)
    h = ggml_mul_mat(ctx0, c->talker.text_proj_fc1_w, h);                    // (T, 2048)
    h = ggml_add(ctx0, h, c->talker.text_proj_fc1_b);
    h = ggml_silu(ctx0, h);
    h = ggml_mul_mat(ctx0, c->talker.text_proj_fc2_w, h); // (T, 1024)
    h = ggml_add(ctx0, h, c->talker.text_proj_fc2_b);
    ggml_set_name(h, "embeds");
    ggml_build_forward_expand(gf, h);
    ggml_free(ctx0);
    return gf;
}

// Embed audio code ids using `talker.token_embd` (3072, 1024).
ggml_cgraph* build_graph_embed_audio(qwen3_tts_context* c, int n_tokens) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16, false);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);
    ggml_tensor* out = ggml_get_rows(ctx0, c->talker.token_embd_w, ids); // (T, 1024)
    ggml_set_name(out, "embeds");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);
    return gf;
}

// Talker forward with persistent KV cache.
ggml_cgraph* build_graph_talker_kv(qwen3_tts_context* c, int n_past, int n_tokens) {
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
    const int Lk = n_past + T;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    // mRoPE note (PLAN #52): for text-only / pure-AR-codec inputs the
    // 3 mrope axes collapse to a single stream, so single-axis NEOX
    // RoPE matches HF apply_interleaved_mrope element-for-element.
    // The diff harness will flag this if the assumption is wrong on
    // a real prompt — at that point swap to ggml_rope_multi.
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
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->talker.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w,
                                                    b.attn_q_norm_w, b.attn_k_norm_w, positions,
                                                    (T == 1) ? nullptr : causal_mask, c->kv_k, c->kv_v, (int)il, n_past,
                                                    kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, c->talker.output_norm_w);

    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    ggml_tensor* logits = ggml_mul_mat(ctx0, c->talker.codec_head_w, cur);
    ggml_set_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);
    ggml_free(ctx0);
    return gf;
}

// ---------------------------------------------------------------------------
// Compute helpers
// ---------------------------------------------------------------------------

float* run_embed_text(qwen3_tts_context* c, const int32_t* ids, int n) {
    const int d = (int)c->hp.d_model;
    ggml_cgraph* gf = build_graph_embed_text(c, n);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return nullptr;
    ggml_tensor* in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(in, ids, 0, (size_t)n * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* r = (float*)malloc((size_t)d * n * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)d * n * sizeof(float));
    return r;
}

float* run_embed_audio(qwen3_tts_context* c, const int32_t* ids, int n) {
    const int d = (int)c->hp.d_model;
    ggml_cgraph* gf = build_graph_embed_audio(c, n);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return nullptr;
    ggml_tensor* in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(in, ids, 0, (size_t)n * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* r = (float*)malloc((size_t)d * n * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)d * n * sizeof(float));
    return r;
}

float* run_talker_kv(qwen3_tts_context* c, const float* embeds, int n_tokens, int n_past) {
    if (n_past + n_tokens > c->kv_max_ctx) {
        fprintf(stderr, "qwen3_tts: kv overflow (%d+%d > %d)\n", n_past, n_tokens, c->kv_max_ctx);
        return nullptr;
    }
    const auto& hp = c->hp;
    const int d = (int)hp.d_model;
    const int vocab = (int)hp.vocab_size;
    const int Lk = n_past + n_tokens;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * n_tokens, zero_h);
        for (int q = 0; q < n_tokens; q++)
            for (int k = n_past + q + 1; k < Lk; k++)
                mask[(size_t)q * Lk + k] = neginf_h;
    }

    ggml_cgraph* gf = build_graph_talker_kv(c, n_past, n_tokens);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1)
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_tts: talker compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    float* r = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)vocab * sizeof(float));
    return r;
}

int argmax(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > bv) {
            bv = logits[i];
            best = i;
        }
    return best;
}

// ---------------------------------------------------------------------------
// Prompt building — match the Qwen3-TTS chat template loosely:
//   <|im_start|>user\n<text><|im_end|>\n<|im_start|>assistant\n<|tts_bos|>
// then we generate audio codes until <|tts_eos|> (for the talker) or the
// codec-side <eos> code.
// ---------------------------------------------------------------------------

std::vector<int32_t> build_prompt_ids(qwen3_tts_context* c, const std::string& text) {
    std::vector<int32_t> ids;
    const auto& v = c->vocab;
    auto push_special = [&](const char* tok) {
        auto it = v.token_to_id.find(tok);
        if (it != v.token_to_id.end())
            ids.push_back(it->second);
    };
    push_special("<|im_start|>");
    {
        std::string user = "user\n" + text;
        std::string enc = core_bpe::bytes_to_unicode(user.data(), user.size());
        core_bpe::bpe_one(v.token_to_id, v.merge_rank, enc, ids);
    }
    push_special("<|im_end|>");
    {
        std::string nl = "\n";
        std::string enc = core_bpe::bytes_to_unicode(nl.data(), nl.size());
        core_bpe::bpe_one(v.token_to_id, v.merge_rank, enc, ids);
    }
    push_special("<|im_start|>");
    {
        std::string asst = "assistant\n";
        std::string enc = core_bpe::bytes_to_unicode(asst.data(), asst.size());
        core_bpe::bpe_one(v.token_to_id, v.merge_rank, enc, ids);
    }
    ids.push_back((int32_t)c->hp.tts_bos_id);
    return ids;
}

} // namespace

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

extern "C" struct qwen3_tts_context_params qwen3_tts_context_default_params(void) {
    qwen3_tts_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.temperature = 0.0f;
    p.max_codec_steps = 0;
    return p;
}

extern "C" struct qwen3_tts_context* qwen3_tts_init_from_file(const char* path_model,
                                                              struct qwen3_tts_context_params params) {
    auto* c = new qwen3_tts_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // ---- pass 1: read hparams + vocab via metadata-only context ----
    {
        ggml_context* dummy = nullptr;
        gguf_init_params gp = {/*no_alloc=*/true, &dummy};
        gguf_context* g = gguf_init_from_file(path_model, gp);
        if (!g) {
            fprintf(stderr, "qwen3_tts: failed to read GGUF '%s'\n", path_model);
            delete c;
            return nullptr;
        }
        auto& hp = c->hp;
        hp.n_layers = kv_u32(g, "qwen3tts.talker.n_layers", hp.n_layers);
        hp.d_model = kv_u32(g, "qwen3tts.talker.d_model", hp.d_model);
        hp.n_heads = kv_u32(g, "qwen3tts.talker.n_heads", hp.n_heads);
        hp.n_kv_heads = kv_u32(g, "qwen3tts.talker.n_kv_heads", hp.n_kv_heads);
        hp.head_dim = kv_u32(g, "qwen3tts.talker.head_dim", hp.head_dim);
        hp.ff_dim = kv_u32(g, "qwen3tts.talker.ff_dim", hp.ff_dim);
        hp.vocab_size = kv_u32(g, "qwen3tts.talker.vocab_size", hp.vocab_size);
        hp.text_vocab_size = kv_u32(g, "qwen3tts.talker.text_vocab_size", hp.text_vocab_size);
        hp.text_hidden_size = kv_u32(g, "qwen3tts.talker.text_hidden_size", hp.text_hidden_size);
        hp.n_code_groups = kv_u32(g, "qwen3tts.talker.n_code_groups", hp.n_code_groups);
        hp.max_pos = kv_u32(g, "qwen3tts.talker.max_pos", hp.max_pos);
        hp.rope_theta = kv_f32(g, "qwen3tts.talker.rope_theta", hp.rope_theta);
        hp.rms_norm_eps = kv_f32(g, "qwen3tts.talker.rms_norm_eps", hp.rms_norm_eps);
        hp.rope_interleaved = kv_bool(g, "qwen3tts.talker.rope_interleaved", hp.rope_interleaved);
        {
            int mr = gguf_find_key(g, "qwen3tts.talker.mrope_section");
            if (mr >= 0) {
                int n = gguf_get_arr_n(g, mr);
                const auto* d = (const uint32_t*)gguf_get_arr_data(g, mr);
                hp.mrope_section.assign(d, d + n);
            }
        }
        hp.cp_n_layers = kv_u32(g, "qwen3tts.code_pred.n_layers", hp.cp_n_layers);
        hp.cp_d_model = kv_u32(g, "qwen3tts.code_pred.d_model", hp.cp_d_model);
        hp.cp_n_heads = kv_u32(g, "qwen3tts.code_pred.n_heads", hp.cp_n_heads);
        hp.cp_n_kv_heads = kv_u32(g, "qwen3tts.code_pred.n_kv_heads", hp.cp_n_kv_heads);
        hp.cp_n_code_groups = kv_u32(g, "qwen3tts.code_pred.n_code_groups", hp.cp_n_code_groups);
        hp.cp_vocab_size = kv_u32(g, "qwen3tts.code_pred.vocab_size", hp.cp_vocab_size);
        hp.cp_max_length = kv_u32(g, "qwen3tts.code_pred.max_length", hp.cp_max_length);
        hp.spk_enc_dim = kv_u32(g, "qwen3tts.speaker.enc_dim", hp.spk_enc_dim);
        hp.spk_sample_rate = kv_u32(g, "qwen3tts.speaker.sample_rate", hp.spk_sample_rate);
        hp.tts_bos_id = kv_u32(g, "qwen3tts.tts_bos_token_id", hp.tts_bos_id);
        hp.tts_eos_id = kv_u32(g, "qwen3tts.tts_eos_token_id", hp.tts_eos_id);
        hp.tts_pad_id = kv_u32(g, "qwen3tts.tts_pad_token_id", hp.tts_pad_id);
        hp.im_start_id = kv_u32(g, "qwen3tts.im_start_token_id", hp.im_start_id);
        hp.im_end_id = kv_u32(g, "qwen3tts.im_end_token_id", hp.im_end_id);
        hp.assistant_id = kv_u32(g, "qwen3tts.assistant_token_id", hp.assistant_id);
        hp.codec_bos_id = kv_u32(g, "qwen3tts.talker.codec_bos_id", hp.codec_bos_id);
        hp.codec_eos_id = kv_u32(g, "qwen3tts.talker.codec_eos_token_id", hp.codec_eos_id);
        hp.codec_pad_id = kv_u32(g, "qwen3tts.talker.codec_pad_id", hp.codec_pad_id);
        hp.codec_think_id = kv_u32(g, "qwen3tts.talker.codec_think_id", hp.codec_think_id);
        hp.codec_nothink_id = kv_u32(g, "qwen3tts.talker.codec_nothink_id", hp.codec_nothink_id);
        hp.codec_think_bos_id = kv_u32(g, "qwen3tts.talker.codec_think_bos_id", hp.codec_think_bos_id);
        hp.codec_think_eos_id = kv_u32(g, "qwen3tts.talker.codec_think_eos_id", hp.codec_think_eos_id);

        auto tok = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
        if (!tok.empty()) {
            c->vocab.id_to_token = std::move(tok);
            c->vocab.token_to_id.reserve(c->vocab.id_to_token.size());
            for (int i = 0; i < (int)c->vocab.id_to_token.size(); i++)
                c->vocab.token_to_id[c->vocab.id_to_token[i]] = i;
        }
        register_qwen_specials(c->vocab);

        auto merges = core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
        for (size_t i = 0; i < merges.size(); i++)
            c->vocab.merge_rank[merges[i]] = (int32_t)i;

        gguf_free(g);
    }

    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "qwen3_tts: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(c->backend_cpu, c->n_threads);
    c->backend = params.use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend)
        c->backend = c->backend_cpu;

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, c->backend, "qwen3_tts", wl)) {
        fprintf(stderr, "qwen3_tts: failed to load weights from '%s'\n", path_model);
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!load_talker(c)) {
        fprintf(stderr, "qwen3_tts: talker weights incomplete\n");
        qwen3_tts_free(c);
        return nullptr;
    }
    load_code_predictor(c); // soft

    // Scheduler
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = c->backend;
        if (c->backend_cpu && c->backend_cpu != c->backend)
            backends[n_be++] = c->backend_cpu;
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    c->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (!kv_alloc(c, /*max_ctx=*/4096)) {
        fprintf(stderr, "qwen3_tts: kv allocation failed\n");
        qwen3_tts_free(c);
        return nullptr;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "qwen3_tts: loaded %s  (talker %uL/%u  code_pred %uL  vocab %u)\n", path_model, c->hp.n_layers,
                c->hp.d_model, c->hp.cp_n_layers, c->hp.vocab_size);
    }
    return c;
}

extern "C" int qwen3_tts_set_codec_path(struct qwen3_tts_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;
    ctx->codec_path = path;
    return 0;
}

extern "C" int qwen3_tts_set_voice_prompt(struct qwen3_tts_context* ctx, const char* wav_path) {
    if (!ctx)
        return -1;
    ctx->voice_prompt_path = wav_path ? wav_path : "";
    return 0;
}

extern "C" float* qwen3_tts_run_text_proj(struct qwen3_tts_context* ctx, const int32_t* ids, int n_tokens, int* out_T,
                                          int* out_d) {
    if (out_T)
        *out_T = 0;
    if (out_d)
        *out_d = 0;
    if (!ctx || !ids || n_tokens <= 0)
        return nullptr;
    float* r = run_embed_text(ctx, ids, n_tokens);
    if (!r)
        return nullptr;
    if (out_T)
        *out_T = n_tokens;
    if (out_d)
        *out_d = (int)ctx->hp.d_model;
    return r;
}

extern "C" float* qwen3_tts_run_talker_with_embeds(struct qwen3_tts_context* ctx, const float* embeds, int n_tokens,
                                                   int* out_vocab) {
    if (out_vocab)
        *out_vocab = 0;
    if (!ctx || !embeds || n_tokens <= 0)
        return nullptr;
    // Guarantee a clean cache: this is a one-shot diff entry point, so
    // n_past=0 always.
    float* logits = run_talker_kv(ctx, embeds, n_tokens, /*n_past=*/0);
    if (!logits)
        return nullptr;
    if (out_vocab)
        *out_vocab = (int)ctx->hp.vocab_size;
    return logits;
}

extern "C" int32_t* qwen3_tts_synthesize_codes(struct qwen3_tts_context* ctx, const char* text, int* out_n_codes) {
    if (out_n_codes)
        *out_n_codes = 0;
    if (!ctx || !text)
        return nullptr;
    if (ctx->vocab.id_to_token.empty()) {
        fprintf(stderr, "qwen3_tts: vocab empty — re-convert with the updated converter\n");
        return nullptr;
    }

    const bool bench = env_bool("QWEN3_TTS_BENCH");
    const bool dbg = env_bool("QWEN3_TTS_DEBUG");
    const char* dump_dir = env_str("QWEN3_TTS_DUMP_DIR");

    auto prompt_ids = build_prompt_ids(ctx, text);
    if (ctx->params.verbosity >= 1 || dbg) {
        fprintf(stderr, "qwen3_tts: prompt %zu tokens\n", prompt_ids.size());
        if (ctx->params.verbosity >= 2 || dbg) {
            fprintf(stderr, "  ids:");
            for (auto id : prompt_ids)
                fprintf(stderr, " %d", id);
            fprintf(stderr, "\n");
        }
    }
    if (dump_dir)
        dump_i32(dump_dir, "prompt_ids", prompt_ids.data(), prompt_ids.size());

    // Prefill: embed text via text_embedding + text_proj (the prompt is
    // text-only; the prompt builder ends with <|tts_bos|> which is also
    // a text-vocab token), then run the talker over the whole prefix.
    double t0 = bench ? now_ms() : 0.0;
    float* embeds = run_embed_text(ctx, prompt_ids.data(), (int)prompt_ids.size());
    if (!embeds)
        return nullptr;
    if (bench)
        fprintf(stderr, "qwen3_tts: text_proj  %7.1f ms (T=%zu)\n", now_ms() - t0, prompt_ids.size());
    if (dump_dir)
        dump_f32(dump_dir, "text_proj_out", embeds, (size_t)ctx->hp.d_model * prompt_ids.size());
    double t1 = bench ? now_ms() : 0.0;
    float* logits = run_talker_kv(ctx, embeds, (int)prompt_ids.size(), /*n_past=*/0);
    free(embeds);
    if (!logits)
        return nullptr;
    if (bench)
        fprintf(stderr, "qwen3_tts: prefill    %7.1f ms\n", now_ms() - t1);
    if (dump_dir)
        dump_f32(dump_dir, "talker_prefill_logits", logits, ctx->hp.vocab_size);
    int n_past = (int)prompt_ids.size();

    const int max_steps = ctx->params.max_codec_steps > 0 ? ctx->params.max_codec_steps : 1500;
    const int eos = (int)ctx->hp.codec_eos_id;

    std::vector<int32_t> codes;
    codes.reserve(max_steps);

    for (int step = 0; step < max_steps; step++) {
        // Greedy sample codebook-0 from the talker's logits (vocab=3072).
        int code = argmax(logits, (int)ctx->hp.vocab_size);
        free(logits);
        logits = nullptr;
        if (code == eos)
            break;
        codes.push_back(code);

        // Decode step: embed via codec_embedding + run one talker step.
        float* e = run_embed_audio(ctx, &codes.back(), 1);
        if (!e) {
            fprintf(stderr, "qwen3_tts: decode embed failed at step %d\n", step);
            break;
        }
        logits = run_talker_kv(ctx, e, 1, n_past);
        free(e);
        if (!logits) {
            fprintf(stderr, "qwen3_tts: decode talker failed at step %d\n", step);
            break;
        }
        n_past += 1;
        if (n_past >= ctx->kv_max_ctx - 1) {
            fprintf(stderr, "qwen3_tts: kv cache full at %d\n", n_past);
            break;
        }
    }
    if (logits)
        free(logits);

    // The KV cache holds layer-by-layer keys/values for `n_past`
    // tokens; the next call would attend to those stale entries
    // unless we explicitly say "n_past=0 again". Our API today is
    // single-shot, so the call sites always start at n_past=0 — no
    // explicit reset needed because each synthesise call rewrites
    // every cache slot it uses. (If we add a partial-decode API
    // later, expose a kv_reset.)
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "qwen3_tts: produced %zu codes (steps used %d / max %d)\n", codes.size(),
                (int)codes.size(), max_steps);

    *out_n_codes = (int)codes.size();
    int32_t* out = (int32_t*)malloc(codes.size() * sizeof(int32_t));
    memcpy(out, codes.data(), codes.size() * sizeof(int32_t));
    return out;
}

extern "C" void qwen3_tts_codes_free(int32_t* codes) {
    free(codes);
}

extern "C" float* qwen3_tts_synthesize(struct qwen3_tts_context* /*ctx*/, const char* /*text*/, int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    fprintf(stderr,
            "qwen3_tts: synthesize() needs the codec decoder (PLAN #52 step 3). "
            "Use qwen3_tts_synthesize_codes() + the python codec helper for now.\n");
    return nullptr;
}

extern "C" void qwen3_tts_pcm_free(float* pcm) {
    free(pcm);
}

extern "C" void qwen3_tts_free(struct qwen3_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
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

extern "C" void qwen3_tts_set_n_threads(struct qwen3_tts_context* ctx, int n_threads) {
    if (!ctx || n_threads <= 0)
        return;
    ctx->n_threads = n_threads;
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}
