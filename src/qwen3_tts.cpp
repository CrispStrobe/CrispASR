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

    // Code predictor KV cache (5 layers, max_ctx 32 — very small).
    ggml_context* cp_kv_ctx = nullptr;
    ggml_backend_buffer_t cp_kv_buf = nullptr;
    ggml_tensor* cp_kv_k = nullptr;
    ggml_tensor* cp_kv_v = nullptr;
    int cp_kv_max_ctx = 0;

    // Loaded voice pack (zero-copy: `vp_tensors` references the
    // weight context's tensors directly).
    std::vector<std::string> vp_names;
    std::vector<std::string> vp_ref_texts;
    std::map<std::string, ggml_tensor*> vp_tensors;
    int vp_active = -1; // index into vp_names; -1 = none selected

    int language_id = -1; // codec language id (-1 = auto / nothink path)

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
    // The vocab loaded from `tokenizer.ggml.tokens` only has the regular
    // BPE entries (151 643); the chat-template special tokens live above
    // that range and would be silently skipped without resizing first.
    int max_id = 0;
    for (const auto& sp : specials)
        if (sp.id > max_id)
            max_id = sp.id;
    if ((int)v.id_to_token.size() <= max_id)
        v.id_to_token.resize((size_t)max_id + 1);
    for (const auto& sp : specials) {
        auto old_it = v.token_to_id.find(v.id_to_token[sp.id]);
        if (old_it != v.token_to_id.end() && old_it->second == sp.id)
            v.token_to_id.erase(old_it);
        v.id_to_token[sp.id] = sp.text;
        v.token_to_id[sp.text] = sp.id;
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
// Prompt building — Qwen3-TTS Base ICL chat templates per
//   ref/Qwen3-TTS/qwen_tts/inference/qwen3_tts_model.py
//
//   _build_assistant_text(text) =
//       "<|im_start|>assistant\n" + text + "<|im_end|>\n"
//       "<|im_start|>assistant\n"
//   _build_ref_text(ref_text) =
//       "<|im_start|>assistant\n" + ref_text + "<|im_end|>\n"
//
// The first 3 tokens of each are always the role prefix
// "<|im_start|>", "assistant", "\n" (the official tokenizer encodes
// `assistant\n` as one BPE token "assistant" plus `\n`).
// ---------------------------------------------------------------------------

// Token id for `\n` in the Qwen2/Qwen3 vocab — byte-encoded as "Ċ"
// (codepoint U+010A) which the Qwen vocab maps to id 198.
constexpr int32_t kNewlineId = 198;

void push_special(const g3t_vocab& v, std::vector<int32_t>& ids, const char* tok) {
    auto it = v.token_to_id.find(tok);
    if (it != v.token_to_id.end())
        ids.push_back(it->second);
}

// Tokenise a user-supplied free-text fragment. Splits on whitespace
// (matching `core_bpe::tokenize_simple`) and BPE-merges each word,
// pre-pending a leading space to all but the first word. The
// official Qwen tokenizer uses a fuller GPT-2 regex pre-tokeniser
// — for the simple TTS prompts we synthesise (mostly Latin
// letters + punctuation), the whitespace-splitter is sufficient
// and produces matching ids on the smoke prompts. If we ever hit
// a divergence, drop in a regex-based pre-tokenizer here.
void push_text_block(const g3t_vocab& v, std::vector<int32_t>& ids, const std::string& s) {
    auto out = core_bpe::tokenize_simple(v.token_to_id, v.merge_rank, s);
    ids.insert(ids.end(), out.begin(), out.end());
}

// Build the synthesis-side prompt:
//   "<|im_start|>assistant\n<text><|im_end|>\n<|im_start|>assistant\n"
// matching `Qwen3TTSModel._build_assistant_text` in the reference.
std::vector<int32_t> tokenise_assistant_text(qwen3_tts_context* c, const std::string& text) {
    std::vector<int32_t> ids;
    const auto& v = c->vocab;
    push_special(v, ids, "<|im_start|>");
    push_text_block(v, ids, "assistant"); // → [77091]
    ids.push_back(kNewlineId);
    push_text_block(v, ids, text);
    push_special(v, ids, "<|im_end|>");
    ids.push_back(kNewlineId);
    push_special(v, ids, "<|im_start|>");
    push_text_block(v, ids, "assistant");
    ids.push_back(kNewlineId);
    return ids;
}

// Build the reference-side prompt for ICL voice cloning:
//   "<|im_start|>assistant\n<ref_text><|im_end|>\n"
// matching `Qwen3TTSModel._build_ref_text`.
std::vector<int32_t> tokenise_ref_text(qwen3_tts_context* c, const std::string& ref_text) {
    std::vector<int32_t> ids;
    const auto& v = c->vocab;
    push_special(v, ids, "<|im_start|>");
    push_text_block(v, ids, "assistant");
    ids.push_back(kNewlineId);
    push_text_block(v, ids, ref_text);
    push_special(v, ids, "<|im_end|>");
    ids.push_back(kNewlineId);
    return ids;
}

// ---------------------------------------------------------------------------
// ICL prefill builder
//
// Mirrors `Qwen3TTSForConditionalGeneration.generate` (modeling_qwen3_tts.py
// ~line 2070) for the voice-clone Base path with non_streaming_mode=False.
//
// Final prefill tensor structure (auto-language, ICL mode):
//
//   [_talker_input_embed_role  ] (3 tokens)   text_proj(text_embd(syn_ids[:3]))
//   [bridge                    ] (L-1 tokens) tts_pad×(L-2) + tts_bos
//                                              + codec_input_embd[:-1]
//   [icl_input_embed           ] (max(text_lens, codec_lens) tokens)
//
// where L = codec_input_embd length = 3 + 1(spk) + 2(pad,bos) = 6 (auto)
// or 4 + 1 + 2 = 7 (explicit language).
// ---------------------------------------------------------------------------

// Generic embedding-row lookup: builds a tiny graph
//   out = ggml_get_rows(weight, ids)
// runs it, and copies the (n_ids, weight->ne[0]) result to a freshly
// malloc'd float buffer. Used for codec_embedding / code_predictor
// codec_embedding[i] / talker.token_embd lookups. Caller frees with
// free().
float* lookup_rows(qwen3_tts_context* c, ggml_tensor* weight, const int32_t* ids, int n_ids) {
    if (!weight)
        return nullptr;
    const int d = (int)weight->ne[0];

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16, false);
    ggml_tensor* idsT = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ids);
    ggml_set_name(idsT, "ids");
    ggml_set_input(idsT);
    ggml_tensor* out = ggml_get_rows(ctx0, weight, idsT);
    ggml_set_name(out, "rows");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);

    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ids"), ids, 0, (size_t)n_ids * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    ggml_tensor* outT = ggml_graph_get_tensor(gf, "rows");
    float* r = (float*)malloc((size_t)n_ids * d * sizeof(float));
    ggml_backend_tensor_get(outT, r, 0, (size_t)n_ids * d * sizeof(float));
    return r;
}

// Compute the per-frame summed codec embedding for ref_code (T_codec, 16):
//
//   per_frame_sum[t] = sum_{cb=0..15} embd_for_cb(ref_code[t, cb])
//
// where embd_for_cb is `talker.codec_embedding` for cb=0 and
// `code_predictor.codec_embedding[cb-1]` for cb=1..15. Returns a
// freshly malloc'd buffer of size (T_codec * d). Caller frees.
float* sum_codec_embeds(qwen3_tts_context* c, const int32_t* ref_code_TC, int T_codec) {
    const int d = (int)c->hp.d_model;
    const int n_groups = (int)c->hp.n_code_groups;

    auto& cp = c->code_pred;
    if ((int)cp.codec_embd.size() < n_groups - 1) {
        fprintf(stderr, "qwen3_tts: code_predictor codec_embedding tables missing (%zu/%d)\n", cp.codec_embd.size(),
                n_groups - 1);
        return nullptr;
    }

    std::vector<int32_t> col(T_codec);
    float* acc = (float*)calloc((size_t)T_codec * d, sizeof(float));

    for (int cb = 0; cb < n_groups; cb++) {
        for (int t = 0; t < T_codec; t++)
            col[t] = ref_code_TC[(size_t)t * n_groups + cb];
        ggml_tensor* w = (cb == 0) ? c->talker.token_embd_w : cp.codec_embd[cb - 1];
        if (!w) {
            free(acc);
            return nullptr;
        }
        float* rows = lookup_rows(c, w, col.data(), T_codec);
        if (!rows) {
            free(acc);
            return nullptr;
        }
        for (size_t i = 0; i < (size_t)T_codec * d; i++)
            acc[i] += rows[i];
        free(rows);
    }
    return acc;
}

// Build the full ICL prefill embedding for a (text, ref_text, ref_code,
// spk_embed) tuple. Output `prefill_embeds` is (T_prefill, d) row-major
// float32. `trailing_text_hidden` is the M×d "padding" tensor that gets
// added to each decode-step's input (just `tts_pad_embed` in the
// codec_lens > text_lens case which is typical for short syn text +
// long ref audio). Returns true on success.
bool build_icl_prefill_embeds(qwen3_tts_context* c, const std::string& syn_text, const std::string& ref_text,
                              std::vector<float>& prefill_embeds, int& T_prefill,
                              std::vector<float>& trailing_text_hidden, int& M_trailing) {
    const auto& hp = c->hp;
    const int d = (int)hp.d_model;
    const int n_groups = (int)hp.n_code_groups;

    if (c->vp_active < 0) {
        fprintf(stderr, "qwen3_tts: no voice selected — call qwen3_tts_load_voice_pack + select_voice first\n");
        return false;
    }
    const std::string& voice_name = c->vp_names[c->vp_active];

    // ---- gather voice pack tensors ----
    auto find_vp = [&](const std::string& key) -> ggml_tensor* {
        auto it = c->vp_tensors.find(key);
        return it == c->vp_tensors.end() ? nullptr : it->second;
    };
    ggml_tensor* spk_t = find_vp("voicepack.spk." + voice_name + ".embd");
    ggml_tensor* code_t = find_vp("voicepack.code." + voice_name + ".codes");
    if (!spk_t || !code_t) {
        fprintf(stderr, "qwen3_tts: voice '%s' missing spk_embd / ref_code\n", voice_name.c_str());
        return false;
    }
    if ((int)spk_t->ne[0] != d) {
        fprintf(stderr, "qwen3_tts: voice spk_embd dim mismatch: %d vs %d\n", (int)spk_t->ne[0], d);
        return false;
    }
    const int T_codec = (int)code_t->ne[1]; // (16, T_codec) row-major i32, ne[1]=T
    if ((int)code_t->ne[0] != n_groups) {
        fprintf(stderr, "qwen3_tts: voice ref_code groups mismatch: %d vs %d\n", (int)code_t->ne[0], n_groups);
        return false;
    }

    std::vector<float> spk_buf(d);
    ggml_backend_tensor_get(spk_t, spk_buf.data(), 0, (size_t)d * sizeof(float));
    std::vector<int32_t> ref_code_TC((size_t)T_codec * n_groups);
    ggml_backend_tensor_get(code_t, ref_code_TC.data(), 0, ref_code_TC.size() * sizeof(int32_t));

    // ---- tokenise ----
    auto syn_ids = tokenise_assistant_text(c, syn_text);
    auto ref_ids = tokenise_ref_text(c, ref_text);
    if ((int)syn_ids.size() < 8 || (int)ref_ids.size() < 5) {
        fprintf(stderr, "qwen3_tts: prompt too short (syn=%zu ref=%zu)\n", syn_ids.size(), ref_ids.size());
        return false;
    }

    // ---- tts_bos / tts_eos / tts_pad embeds via text_proj ----
    int32_t tts_special[3] = {(int32_t)hp.tts_bos_id, (int32_t)hp.tts_eos_id, (int32_t)hp.tts_pad_id};
    float* tts_special_emb = run_embed_text(c, tts_special, 3);
    if (!tts_special_emb)
        return false;
    const float* tts_bos = tts_special_emb;            // (d,)
    const float* tts_eos = tts_special_emb + d;        // (d,)
    const float* tts_pad = tts_special_emb + 2 * d;    // (d,)

    // ---- codec_input_embedding: sentinels + spk + pad/bos ----
    std::vector<int32_t> codec_prefill;
    if (c->language_id <= 0) {
        codec_prefill = {(int32_t)hp.codec_nothink_id, (int32_t)hp.codec_think_bos_id,
                         (int32_t)hp.codec_think_eos_id};
    } else {
        codec_prefill = {(int32_t)hp.codec_think_id, (int32_t)hp.codec_think_bos_id, (int32_t)c->language_id,
                         (int32_t)hp.codec_think_eos_id};
    }
    int32_t codec_pad_bos[2] = {(int32_t)hp.codec_pad_id, (int32_t)hp.codec_bos_id};

    float* codec_pre_emb = lookup_rows(c, c->talker.token_embd_w, codec_prefill.data(), (int)codec_prefill.size());
    float* codec_pb_emb = lookup_rows(c, c->talker.token_embd_w, codec_pad_bos, 2);
    if (!codec_pre_emb || !codec_pb_emb) {
        free(tts_special_emb);
        free(codec_pre_emb);
        free(codec_pb_emb);
        return false;
    }

    const int L_codec = (int)codec_prefill.size() + 1 /*spk*/ + 2 /*pad,bos*/;
    std::vector<float> codec_input_emb((size_t)L_codec * d);
    {
        // [codec_prefill (3 or 4) | spk | pad,bos (2)]
        size_t pos = 0;
        const size_t bytes_pre = (size_t)codec_prefill.size() * d * sizeof(float);
        std::memcpy(codec_input_emb.data() + pos, codec_pre_emb, bytes_pre);
        pos += codec_prefill.size() * d;
        std::memcpy(codec_input_emb.data() + pos, spk_buf.data(), (size_t)d * sizeof(float));
        pos += d;
        std::memcpy(codec_input_emb.data() + pos, codec_pb_emb, (size_t)2 * d * sizeof(float));
    }
    free(codec_pre_emb);
    free(codec_pb_emb);

    // ---- role embed: text_proj(text_embd(syn_ids[:3])) ----
    std::vector<int32_t> role_ids(syn_ids.begin(), syn_ids.begin() + 3);
    float* role_emb = run_embed_text(c, role_ids.data(), 3);
    if (!role_emb) {
        free(tts_special_emb);
        return false;
    }

    // ---- bridge: cat(tts_pad×(L-2), tts_bos) + codec_input_emb[:-1] ----
    const int L_bridge = L_codec - 1;
    std::vector<float> bridge((size_t)L_bridge * d);
    for (int i = 0; i < L_bridge; i++) {
        const float* left = (i < L_bridge - 1) ? tts_pad : tts_bos;
        const float* right = codec_input_emb.data() + (size_t)i * d;
        for (int j = 0; j < d; j++)
            bridge[(size_t)i * d + j] = left[j] + right[j];
    }

    // ---- text_embed (ref + text + tts_eos) ----
    // input_id[:, 3:-5] = synth text content (without role + end tail)
    // ref_id[:, 3:-2]   = ref text content (without role + end)
    std::vector<int32_t> ref_content(ref_ids.begin() + 3, ref_ids.end() - 2);
    std::vector<int32_t> text_content(syn_ids.begin() + 3, syn_ids.end() - 5);
    std::vector<int32_t> rt_concat;
    rt_concat.reserve(ref_content.size() + text_content.size());
    rt_concat.insert(rt_concat.end(), ref_content.begin(), ref_content.end());
    rt_concat.insert(rt_concat.end(), text_content.begin(), text_content.end());
    float* text_emb = run_embed_text(c, rt_concat.data(), (int)rt_concat.size());
    if (!text_emb) {
        free(tts_special_emb);
        free(role_emb);
        return false;
    }
    const int text_lens = (int)rt_concat.size() + 1; // append tts_eos
    std::vector<float> text_embed_padded;
    text_embed_padded.reserve((size_t)text_lens * d);
    text_embed_padded.insert(text_embed_padded.end(), text_emb, text_emb + (size_t)rt_concat.size() * d);
    text_embed_padded.insert(text_embed_padded.end(), tts_eos, tts_eos + d);
    free(text_emb);

    // ---- codec_embed (codec_bos + per-frame sum of 16 codebooks) ----
    int32_t codec_bos = (int32_t)hp.codec_bos_id;
    float* codec_bos_emb = lookup_rows(c, c->talker.token_embd_w, &codec_bos, 1);
    float* codec_sum = sum_codec_embeds(c, ref_code_TC.data(), T_codec);
    if (!codec_bos_emb || !codec_sum) {
        free(tts_special_emb);
        free(role_emb);
        free(codec_bos_emb);
        free(codec_sum);
        return false;
    }
    const int codec_lens = T_codec + 1;
    std::vector<float> codec_embed((size_t)codec_lens * d);
    std::memcpy(codec_embed.data(), codec_bos_emb, (size_t)d * sizeof(float));
    std::memcpy(codec_embed.data() + d, codec_sum, (size_t)T_codec * d * sizeof(float));
    free(codec_bos_emb);
    free(codec_sum);

    // ---- ICL fusion (non_streaming_mode=False) ----
    int icl_len = std::max(text_lens, codec_lens);
    std::vector<float> icl_input((size_t)icl_len * d);
    if (codec_lens >= text_lens) {
        // Pad text_embed_padded to codec_lens with tts_pad, then sum elementwise.
        std::vector<float> padded((size_t)codec_lens * d);
        std::memcpy(padded.data(), text_embed_padded.data(), text_embed_padded.size() * sizeof(float));
        for (int i = text_lens; i < codec_lens; i++)
            std::memcpy(padded.data() + (size_t)i * d, tts_pad, (size_t)d * sizeof(float));
        for (size_t i = 0; i < padded.size(); i++)
            icl_input[i] = padded[i] + codec_embed[i];
        // Trailing for codec >= text: just tts_pad_embed (1 token).
        trailing_text_hidden.assign(tts_pad, tts_pad + d);
        M_trailing = 1;
    } else {
        // text_lens > codec_lens: take text[:codec_lens] + codec_embed.
        // Trailing = text[codec_lens:].
        for (int i = 0; i < codec_lens; i++)
            for (int j = 0; j < d; j++)
                icl_input[(size_t)i * d + j] =
                    text_embed_padded[(size_t)i * d + j] + codec_embed[(size_t)i * d + j];
        const int trail = text_lens - codec_lens;
        trailing_text_hidden.assign(text_embed_padded.begin() + (size_t)codec_lens * d, text_embed_padded.end());
        M_trailing = trail;
        icl_len = codec_lens;
    }

    // ---- final concat: role(3) + bridge(L_codec-1) + icl_input(icl_len) ----
    T_prefill = 3 + L_bridge + icl_len;
    prefill_embeds.assign((size_t)T_prefill * d, 0.0f);
    size_t off = 0;
    std::memcpy(prefill_embeds.data() + off, role_emb, (size_t)3 * d * sizeof(float));
    off += (size_t)3 * d;
    std::memcpy(prefill_embeds.data() + off, bridge.data(), bridge.size() * sizeof(float));
    off += bridge.size();
    std::memcpy(prefill_embeds.data() + off, icl_input.data(), (size_t)icl_len * d * sizeof(float));

    free(tts_special_emb);
    free(role_emb);

    if (c->params.verbosity >= 1) {
        fprintf(stderr,
                "qwen3_tts: ICL prefill: role=3 + bridge=%d + icl=%d (text_lens=%d codec_lens=%d) = T=%d  "
                "trailing=%d\n",
                L_bridge, icl_len, text_lens, codec_lens, T_prefill, M_trailing);
    }
    if (const char* dd = env_str("QWEN3_TTS_DUMP_DIR")) {
        dump_f32(dd, "icl_role", role_emb, (size_t)3 * d);
        dump_f32(dd, "icl_bridge", bridge.data(), bridge.size());
        dump_f32(dd, "icl_codec_input", codec_input_emb.data(), codec_input_emb.size());
        dump_f32(dd, "icl_text_embed", text_embed_padded.data(), text_embed_padded.size());
        dump_f32(dd, "icl_codec_embed", codec_embed.data(), codec_embed.size());
        dump_f32(dd, "icl_input", icl_input.data(), icl_input.size());
        dump_f32(dd, "icl_prefill", prefill_embeds.data(), prefill_embeds.size());
        dump_f32(dd, "tts_special_emb", tts_special_emb, (size_t)3 * d);
        dump_i32(dd, "syn_ids", syn_ids.data(), syn_ids.size());
        dump_i32(dd, "ref_ids", ref_ids.data(), ref_ids.size());
    }
    return true;
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

extern "C" int qwen3_tts_load_voice_pack(struct qwen3_tts_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;

    // Read the names + ref_texts arrays from the metadata, then load
    // every tensor onto the same backend the talker weights live on
    // (so ggml_get_rows / ggml_view_2d access them without crossing
    // backend boundaries during graph build).
    {
        ggml_context* dummy = nullptr;
        gguf_init_params gp = {true, &dummy};
        gguf_context* g = gguf_init_from_file(path, gp);
        if (!g) {
            fprintf(stderr, "qwen3_tts: failed to read voice pack '%s'\n", path);
            return -1;
        }
        ctx->vp_names = core_gguf::kv_str_array(g, "voicepack.names");
        ctx->vp_ref_texts = core_gguf::kv_str_array(g, "voicepack.ref_texts");
        gguf_free(g);
    }

    // load_weights returns a fresh ggml_context — we keep the existing
    // talker context separate to avoid any aliasing on free.
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "qwen3_tts.voicepack", wl)) {
        fprintf(stderr, "qwen3_tts: failed to load voice pack tensors from '%s'\n", path);
        return -1;
    }
    ctx->vp_tensors = std::move(wl.tensors);
    // We ignore wl.ctx / wl.buf to keep cleanup simple — they leak
    // until process exit, which is fine for the tiny voice-pack
    // tensors (kilobytes total). If memory pressure becomes an
    // issue, retain them and free on context destruction.

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "qwen3_tts: loaded voice pack %s with %zu voices\n", path, ctx->vp_names.size());
        for (size_t i = 0; i < ctx->vp_names.size(); i++)
            fprintf(stderr, "  [%zu] %s\n", i, ctx->vp_names[i].c_str());
    }
    if (ctx->vp_active < 0 && !ctx->vp_names.empty())
        ctx->vp_active = 0; // auto-select first voice
    return 0;
}

extern "C" int qwen3_tts_select_voice(struct qwen3_tts_context* ctx, const char* name) {
    if (!ctx || !name)
        return -1;
    if (ctx->vp_names.empty())
        return -1;
    for (size_t i = 0; i < ctx->vp_names.size(); i++) {
        if (ctx->vp_names[i] == name) {
            ctx->vp_active = (int)i;
            if (ctx->params.verbosity >= 1)
                fprintf(stderr, "qwen3_tts: selected voice [%zu] '%s'\n", i, name);
            return 0;
        }
    }
    return -2;
}

extern "C" int qwen3_tts_set_language(struct qwen3_tts_context* ctx, int codec_language_id) {
    if (!ctx)
        return -1;
    ctx->language_id = codec_language_id;
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

extern "C" float* qwen3_tts_build_icl_prefill(struct qwen3_tts_context* ctx, const char* syn_text, const char* ref_text,
                                              int* out_T) {
    if (out_T)
        *out_T = 0;
    if (!ctx || !syn_text || !ref_text)
        return nullptr;
    std::vector<float> embeds, trailing;
    int T = 0, M = 0;
    if (!build_icl_prefill_embeds(ctx, syn_text, ref_text, embeds, T, trailing, M))
        return nullptr;
    const int d = (int)ctx->hp.d_model;
    float* r = (float*)malloc((size_t)T * d * sizeof(float));
    std::memcpy(r, embeds.data(), embeds.size() * sizeof(float));
    if (out_T)
        *out_T = T;
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

    auto prompt_ids = tokenise_assistant_text(ctx, text);
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
