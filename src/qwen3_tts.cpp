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
//   ✓ codec decoder (8L sliding-window transformer + 1D-conv upsample
//     stack to 24 kHz waveform, in Qwen3-TTS-Tokenizer-12Hz):
//     loaded via `qwen3_tts_set_codec_path`. `qwen3_tts_synthesize`
//     produces PCM end-to-end when a voice pack is also loaded.
//     CPU path verified (T=5 frames → 9600 samples, all finite, correct
//     range). Metal path hangs on M1 — the GPU scheduler conflicts with
//     ggml_conv_1d_dw + SnakeBeta ops; investigate separately.
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

// ---------------------------------------------------------------------------
// Codec decoder structs (PLAN #52 step 3)
// ---------------------------------------------------------------------------

struct g3t_codec_hp {
    uint32_t n_layers = 8;
    uint32_t d_model = 512;
    uint32_t n_heads = 16;
    uint32_t head_dim = 64;
    uint32_t ff_dim = 1024;
    uint32_t n_q = 16;
    uint32_t codebook_size = 2048;
    uint32_t latent_dim = 1024;
    uint32_t decoder_dim = 1536;
    uint32_t sliding_window = 72;
    uint32_t max_pos = 8000;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
    int upsample_rates[4] = {8, 5, 4, 3};
    int upsampling_ratios[2] = {2, 2};
};

struct g3t_codec_xfmr_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_ls_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_ls_w = nullptr;
};

struct g3t_codec_up_stage {
    ggml_tensor* tconv_w = nullptr;
    ggml_tensor* tconv_b = nullptr;
    ggml_tensor* dw_w = nullptr;
    ggml_tensor* dw_b = nullptr;
    ggml_tensor* norm_w = nullptr;
    ggml_tensor* norm_b = nullptr;
    ggml_tensor* pw1_w = nullptr;
    ggml_tensor* pw1_b = nullptr;
    ggml_tensor* pw2_w = nullptr;
    ggml_tensor* pw2_b = nullptr;
    ggml_tensor* gamma = nullptr;
};

struct g3t_codec_res_unit {
    ggml_tensor* act1_a = nullptr;
    ggml_tensor* act1_b = nullptr;
    ggml_tensor* act2_a = nullptr;
    ggml_tensor* act2_b = nullptr;
    ggml_tensor* conv1_w = nullptr;
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* conv2_w = nullptr;
    ggml_tensor* conv2_b = nullptr;
};

struct g3t_codec_dec_block {
    ggml_tensor* snake_a = nullptr;
    ggml_tensor* snake_b = nullptr;
    ggml_tensor* tconv_w = nullptr;
    ggml_tensor* tconv_b = nullptr;
    g3t_codec_res_unit res[3];
};

struct g3t_codec {
    g3t_codec_hp hp;

    ggml_tensor* rvq_first_cb = nullptr;
    ggml_tensor* rvq_first_out_w = nullptr;
    ggml_tensor* rvq_rest_cb[15] = {};
    ggml_tensor* rvq_rest_out_w = nullptr;

    ggml_tensor* pre_conv_w = nullptr;
    ggml_tensor* pre_conv_b = nullptr;

    ggml_tensor* xfmr_in_proj_w = nullptr;
    ggml_tensor* xfmr_in_proj_b = nullptr;
    ggml_tensor* xfmr_norm_w = nullptr;
    ggml_tensor* xfmr_out_proj_w = nullptr;
    ggml_tensor* xfmr_out_proj_b = nullptr;
    std::vector<g3t_codec_xfmr_layer> xfmr_layers;

    g3t_codec_up_stage up[2];

    ggml_tensor* in_conv_w = nullptr;
    ggml_tensor* in_conv_b = nullptr;
    g3t_codec_dec_block blocks[4];
    ggml_tensor* out_snake_a = nullptr;
    ggml_tensor* out_snake_b = nullptr;
    ggml_tensor* out_conv_w = nullptr;
    ggml_tensor* out_conv_b = nullptr;

    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    bool loaded = false;
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

    g3t_codec codec;
    std::vector<uint8_t> codec_compute_meta;
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

// Code-predictor forward with persistent KV cache.
//
// Same Qwen3 backbone as the talker (Q/K-norm + SwiGLU + flash-attn +
// NEOX RoPE + GQA), just narrower (5 layers, hidden 1024, 16Q/8KV, ff
// 3072 — same dims as the 0.6B talker). The lm_head is **per-step** —
// at AR step i in [0..14], we apply `code_pred.lm_head[i]` to project
// the last hidden state to the 2048-codebook vocab. The graph builder
// takes the lm_head tensor as a parameter so we can rebuild a fresh
// graph per step without conditionals inside.
ggml_cgraph* build_graph_code_pred_kv(qwen3_tts_context* c, int n_past, int n_tokens, ggml_tensor* lm_head) {
    const auto& hp = c->hp;
    const int d = (int)hp.cp_d_model;
    const int n_q = (int)hp.cp_n_heads;
    const int n_kv = (int)hp.cp_n_kv_heads;
    const int hd = (int)hp.head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float theta = hp.rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = n_past + T;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

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

    const core_attn::KvSelfAttnParams kvp = {
        n_q, n_kv, hd, n_kv_grp, (int)hp.max_pos, theta, 32.0f, 1.0f, attn_scale, eps, core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.cp_n_layers; il++) {
        const auto& b = c->code_pred.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w,
                                                    b.attn_q_norm_w, b.attn_k_norm_w, positions,
                                                    (T == 1) ? nullptr : causal_mask, c->cp_kv_k, c->cp_kv_v, (int)il,
                                                    n_past, kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, c->code_pred.output_norm_w);

    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    ggml_tensor* logits = ggml_mul_mat(ctx0, lm_head, cur);
    ggml_set_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);
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
    // Expose the last-position hidden state separately — the code
    // predictor's first input is `cat(past_hidden, last_id_hidden)`,
    // and `past_hidden` is exactly this tensor. ggml_cont() so the
    // backend persists it (otherwise it gets folded into the codec_head
    // matmul).
    ggml_tensor* hidden_last = ggml_cont(ctx0, cur);
    ggml_set_name(hidden_last, "hidden_last");
    ggml_build_forward_expand(gf, hidden_last);

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

// Run the talker prefill / decode step, returning logits at position[-1]
// and (optionally) the corresponding last hidden state. Pass `out_hidden_d`
// non-null to receive a malloc'd float buffer of length `hp.d_model`
// — caller frees with free().
float* run_talker_kv(qwen3_tts_context* c, const float* embeds, int n_tokens, int n_past, float** out_hidden_d) {
    if (out_hidden_d)
        *out_hidden_d = nullptr;
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
    if (out_hidden_d) {
        ggml_tensor* hid = ggml_graph_get_tensor(gf, "hidden_last");
        if (hid) {
            float* h = (float*)malloc((size_t)d * sizeof(float));
            ggml_backend_tensor_get(hid, h, 0, (size_t)d * sizeof(float));
            *out_hidden_d = h;
        }
    }
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

// Top-k + temperature sampler. Required for the code_predictor —
// `subtalker_dosample=True` is the official default and greedy
// argmax for codebooks 1..15 produces a degenerate / silent output
// (verified empirically against the qwen-tts reference).
//
// The PyTorch defaults are top_k=50, top_p=1.0, temperature=0.9.
// We implement top_k + temperature; top_p=1.0 is a no-op so omit it.
int top_k_sample(const float* logits, int n, int top_k, float temperature, uint64_t* rng_state) {
    if (top_k <= 0 || top_k >= n)
        top_k = n;
    // Find top-k indices via partial sort. For n=2048, top_k=50,
    // O(n log k) is fine (~50 µs).
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++)
        idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    // Softmax over the top-k logits with temperature.
    const float t = temperature > 0 ? temperature : 1.0f;
    float max_l = logits[idx[0]];
    for (int i = 1; i < top_k; i++)
        if (logits[idx[i]] > max_l)
            max_l = logits[idx[i]];
    std::vector<float> probs(top_k);
    double sum = 0.0;
    for (int i = 0; i < top_k; i++) {
        double p = std::exp((logits[idx[i]] - max_l) / t);
        probs[i] = (float)p;
        sum += p;
    }
    if (sum <= 0)
        return idx[0];
    for (int i = 0; i < top_k; i++)
        probs[i] = (float)(probs[i] / sum);

    // xorshift64* — fast deterministic PRNG, seeded by caller. Given a
    // fixed seed the synthesis is reproducible.
    uint64_t x = *rng_state ? *rng_state : 0xdeadbeefcafebabeULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *rng_state = x;
    double r = (double)((x * 0x2545f4914f6cdd1dULL) >> 11) / (double)(1ULL << 53);

    double cum = 0.0;
    for (int i = 0; i < top_k; i++) {
        cum += probs[i];
        if (r < cum)
            return idx[i];
    }
    return idx[top_k - 1];
}

// Forward-declared: defined in the prefill section below.
float* lookup_rows(qwen3_tts_context* c, ggml_tensor* weight, const int32_t* ids, int n_ids);

// One step of the code-predictor AR loop. Builds + runs the graph
// against a caller-supplied (T, d) embedding tensor and the lm_head
// for the current generation step. Returns logits (cp_vocab,).
float* run_code_pred_kv(qwen3_tts_context* c, const float* embeds, int n_tokens, int n_past,
                        ggml_tensor* lm_head) {
    if (!lm_head)
        return nullptr;
    if (n_past + n_tokens > c->cp_kv_max_ctx) {
        fprintf(stderr, "qwen3_tts: cp_kv overflow (%d+%d > %d)\n", n_past, n_tokens, c->cp_kv_max_ctx);
        return nullptr;
    }
    const auto& hp = c->hp;
    const int d = (int)hp.cp_d_model;
    const int vocab = (int)hp.cp_vocab_size;
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

    ggml_cgraph* gf = build_graph_code_pred_kv(c, n_past, n_tokens, lm_head);
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
        fprintf(stderr, "qwen3_tts: code_pred compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    float* r = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)vocab * sizeof(float));
    return r;
}

// Run the 15-step code-predictor AR loop given the talker's
// past_hidden (the talker's last hidden state, (d,)) and last_id_hidden
// (talker.codec_embedding(codebook-0 sample), (d,)). Writes the 15
// codebook ids (codebooks 1..15) into out_codes. Returns true on
// success.
//
// Always uses sampling (subtalker_dosample=True equivalent) — greedy
// argmax produces a degenerate silent codec output, verified against
// the official qwen-tts reference. Defaults: top_k=50, temperature=0.9.
bool code_pred_generate_15(qwen3_tts_context* c, const float* past_hidden_d, const float* last_id_hidden_d,
                           int32_t* out_codes15, uint64_t* rng_state) {
    auto& cp = c->code_pred;
    const auto& hp = c->hp;
    const int d = (int)hp.cp_d_model;
    const int n_groups = (int)hp.cp_n_code_groups; // 16
    const int top_k = 50;
    const float temperature = 0.9f;

    // ---- step 0: inputs_embeds = (past_hidden, last_id_hidden), n_past=0 ----
    std::vector<float> step0((size_t)2 * d);
    std::memcpy(step0.data(), past_hidden_d, (size_t)d * sizeof(float));
    std::memcpy(step0.data() + d, last_id_hidden_d, (size_t)d * sizeof(float));

    if (!cp.lm_head[0]) {
        fprintf(stderr, "qwen3_tts: code_pred.lm_head[0] missing\n");
        return false;
    }
    float* logits0 = run_code_pred_kv(c, step0.data(), 2, /*n_past=*/0, cp.lm_head[0]);
    if (!logits0)
        return false;
    out_codes15[0] = top_k_sample(logits0, (int)hp.cp_vocab_size, top_k, temperature, rng_state);
    free(logits0);

    int n_past = 2;

    // ---- steps 1..14: input = codec_embedding[i-1](codes[i-1]), apply lm_head[i] ----
    for (int i = 1; i < n_groups - 1; i++) {
        if (!cp.codec_embd[i - 1] || !cp.lm_head[i]) {
            fprintf(stderr, "qwen3_tts: code_pred missing codec_embd[%d] or lm_head[%d]\n", i - 1, i);
            return false;
        }
        int32_t prev = out_codes15[i - 1];
        float* emb = lookup_rows(c, cp.codec_embd[i - 1], &prev, 1);
        if (!emb)
            return false;
        float* logits = run_code_pred_kv(c, emb, 1, n_past, cp.lm_head[i]);
        free(emb);
        if (!logits)
            return false;
        out_codes15[i] = top_k_sample(logits, (int)hp.cp_vocab_size, top_k, temperature, rng_state);
        free(logits);
        n_past += 1;
    }
    return true;
}

// Allocate the code_predictor KV cache: (head_dim, max_ctx, n_kv, cp_n_layers).
// max_ctx is small — at most 2 + 14 = 16 positions per frame.
bool cp_kv_alloc(qwen3_tts_context* c) {
    if (c->cp_kv_k)
        return true;
    const auto& hp = c->hp;
    const int hd = (int)hp.head_dim;
    const int n_kv = (int)hp.cp_n_kv_heads;
    const int n_lay = (int)hp.cp_n_layers;
    const int max_ctx = 32;

    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    c->cp_kv_ctx = ggml_init(kp);
    c->cp_kv_k = ggml_new_tensor_4d(c->cp_kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    c->cp_kv_v = ggml_new_tensor_4d(c->cp_kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(c->cp_kv_k, "cp_kv_k");
    ggml_set_name(c->cp_kv_v, "cp_kv_v");
    const size_t kb = ggml_nbytes(c->cp_kv_k), vb = ggml_nbytes(c->cp_kv_v);
    c->cp_kv_buf = ggml_backend_alloc_buffer(c->backend, kb + vb);
    if (!c->cp_kv_buf)
        return false;
    char* base = (char*)ggml_backend_buffer_get_base(c->cp_kv_buf);
    ggml_backend_tensor_alloc(c->cp_kv_buf, c->cp_kv_k, base);
    ggml_backend_tensor_alloc(c->cp_kv_buf, c->cp_kv_v, base + kb);
    c->cp_kv_max_ctx = max_ctx;
    return true;
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

// ============================================================================
// Codec decoder implementation (PLAN #52 step 3)
// ============================================================================

// ---------------------------------------------------------------------------
// Causal conv1d with explicit dilation — needed for ResidualUnit conv1
// where dilations cycle through 1, 3, 9.
// Input/output: [C, T] channels-first.
// ---------------------------------------------------------------------------
static ggml_tensor* codec_causal_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                        int stride, int dilation) {
    const int K = (int)w->ne[0];
    int pad_left = (K - 1) * dilation;
    if (stride > 1)
        pad_left -= (stride - 1);
    if (pad_left < 0)
        pad_left = 0;
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [T, C]
    if (pad_left > 0) {
        x = ggml_pad_ext(ctx, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
        // ggml_pad_ext always emits a 4D tensor; reshape back to 2D so
        // ggml_conv_1d's internal im2col step sees a standard 2D input.
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
    }
    x = ggml_conv_1d(ctx, w, x, stride, 0, dilation);
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C_out, T_out]
    if (b)
        x = ggml_add(ctx, x, b);
    return x;
}

// ---------------------------------------------------------------------------
// Causal depthwise conv1d for ConvNeXt.
// Input/output: [C, T] channels-first. w shape: [K, 1, C] (ggml ne).
// ---------------------------------------------------------------------------
static ggml_tensor* codec_dw_causal_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    const int K = (int)w->ne[0];
    const int pad_left = K - 1;
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [T, C]
    if (pad_left > 0) {
        x = ggml_pad_ext(ctx, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
    }
    x = ggml_conv_1d_dw(ctx, w, x, 1, 0, 1);
    if (ggml_n_dims(x) > 2)
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C, T]
    if (b)
        x = ggml_add(ctx, x, b);
    return x;
}

// ---------------------------------------------------------------------------
// Causal transposed conv1d for upsampling.
// Input/output: [C, T] channels-first.
// Trims right by (K - stride) samples for causality.
// ---------------------------------------------------------------------------
static ggml_tensor* codec_transposed_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                            int stride) {
    const int K = (int)w->ne[0];
    const int T_in = (int)x->ne[1];
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [T, C_in]
    x = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]); // [T_raw, C_out]
    const int T_raw = (T_in - 1) * stride + K;
    const int T_target = T_in * stride;
    if (T_raw > T_target) {
        const int C_out = (int)x->ne[1];
        x = ggml_view_2d(ctx, x, T_target, C_out, x->nb[1], 0);
        x = ggml_cont(ctx, x);
    }
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C_out, T_out]
    if (b)
        x = ggml_add(ctx, x, b);
    return x;
}

// ---------------------------------------------------------------------------
// SnakeBeta activation: x + exp(-beta) * sin²(x * exp(alpha))
// Input/output: [C, T]. alpha, beta: [C].
// ---------------------------------------------------------------------------
static ggml_tensor* codec_snake_beta(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha, ggml_tensor* beta) {
    ggml_tensor* ea     = ggml_exp(ctx, alpha);           // [C]
    ggml_tensor* neg_eb = ggml_neg(ctx, beta);            // [C]
    ggml_tensor* inv_eb = ggml_exp(ctx, neg_eb);          // [C] = 1/exp(beta)
    ggml_tensor* xa     = ggml_mul(ctx, x, ea);           // [C, T]
    ggml_tensor* s      = ggml_sin(ctx, xa);              // [C, T]
    ggml_tensor* s2     = ggml_mul(ctx, s, s);            // [C, T]
    ggml_tensor* scaled = ggml_mul(ctx, s2, inv_eb);      // [C, T]
    return ggml_add(ctx, x, scaled);
}

// ---------------------------------------------------------------------------
// ConvNeXt block.
// Input/output: [C, T]. dw kernel: [K, 1, C]. pw weights: [4C, C] / [C, 4C].
// ---------------------------------------------------------------------------
static ggml_tensor* codec_convnext_block(ggml_context* ctx, ggml_tensor* x, const g3t_codec_up_stage& up) {
    const float ln_eps = 1e-5f;
    ggml_tensor* residual = x;

    // Depthwise causal conv
    x = codec_dw_causal_conv1d(ctx, x, up.dw_w, up.dw_b); // [C, T]

    // LayerNorm over channels (ggml_norm normalises over ne[0] = C for [C,T])
    x = ggml_norm(ctx, x, ln_eps);
    x = ggml_mul(ctx, x, up.norm_w);
    x = ggml_add(ctx, x, up.norm_b);

    // pwconv1: C → 4C
    x = ggml_add(ctx, ggml_mul_mat(ctx, up.pw1_w, x), up.pw1_b);
    // GELU
    x = ggml_gelu(ctx, x);
    // pwconv2: 4C → C
    x = ggml_add(ctx, ggml_mul_mat(ctx, up.pw2_w, x), up.pw2_b);
    // LayerScale: elementwise [C] × [C, T]
    x = ggml_mul(ctx, x, up.gamma);

    return ggml_add(ctx, residual, x);
}

// ---------------------------------------------------------------------------
// One residual unit: snake1 → dilated_conv(k=7) → snake2 → conv(k=1) → add.
// ---------------------------------------------------------------------------
static ggml_tensor* codec_res_unit(ggml_context* ctx, ggml_tensor* x, const g3t_codec_res_unit& ru, int dilation) {
    ggml_tensor* residual = x;
    x = codec_snake_beta(ctx, x, ru.act1_a, ru.act1_b);
    x = codec_causal_conv1d(ctx, x, ru.conv1_w, ru.conv1_b, 1, dilation);
    x = codec_snake_beta(ctx, x, ru.act2_a, ru.act2_b);
    x = codec_causal_conv1d(ctx, x, ru.conv2_w, ru.conv2_b, 1, 1);
    return ggml_add(ctx, residual, x);
}

// ---------------------------------------------------------------------------
// One decoder block: snake → tconv(stride) → 3× residual_unit(dilations 1,3,9).
// ---------------------------------------------------------------------------
static ggml_tensor* codec_dec_block(ggml_context* ctx, ggml_tensor* x, const g3t_codec_dec_block& blk, int stride) {
    x = codec_snake_beta(ctx, x, blk.snake_a, blk.snake_b);
    x = codec_transposed_conv1d(ctx, x, blk.tconv_w, blk.tconv_b, stride);
    static const int dilations[3] = {1, 3, 9};
    for (int u = 0; u < 3; u++)
        x = codec_res_unit(ctx, x, blk.res[u], dilations[u]);
    return x;
}

// ---------------------------------------------------------------------------
// Build the codec decode compute graph.
//   codes_inp: I32 [T, n_q] input tensor (must be pre-set as ggml_set_input).
//   positions: I32 [T] tensor (0..T-1).
//   attn_mask: F16 [T, T] sliding-window causal mask (nullptr iff T==1).
// Returns the graph output tensor name "pcm" of shape [T_out] F32.
// ---------------------------------------------------------------------------
static ggml_cgraph* build_graph_codec_decode(qwen3_tts_context* c, int T) {
    const auto& codec = c->codec;
    const auto& hp = codec.hp;
    const int n_q = (int)hp.n_q;
    const int n_heads = (int)hp.n_heads;
    const int hd = (int)hp.head_dim;
    const int n_layers = (int)hp.n_layers;
    const float eps = hp.rms_norm_eps;
    const float theta = hp.rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    size_t mem = c->codec_compute_meta.size();
    ggml_init_params ip = {mem, c->codec_compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Input: codes [T, n_q] int32 (inner dim = n_q per frame).
    // Pre-transposed at runtime to [n_q, T] layout so we view each row.
    ggml_tensor* codes_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, T, n_q);
    ggml_set_name(codes_inp, "codec_codes");
    ggml_set_input(codes_inp);

    // ── Step 1: RVQ decode ──────────────────────────────────────────────────
    // rvq_first: lookup codebook 0 → [256, T], apply output_proj → [512, T]
    ggml_tensor* cb0_ids = ggml_view_1d(ctx0, codes_inp, T, 0);
    ggml_tensor* emb_first = ggml_get_rows(ctx0, codec.rvq_first_cb, cb0_ids); // [256, T]
    emb_first = codec_causal_conv1d(ctx0, emb_first, codec.rvq_first_out_w, nullptr, 1, 1); // [512, T]

    // rvq_rest: sum 15 codebook lookups → [256, T], apply output_proj → [512, T]
    ggml_tensor* emb_rest = ggml_get_rows(ctx0, codec.rvq_rest_cb[0],
                                          ggml_view_1d(ctx0, codes_inp, T, (size_t)T * sizeof(int32_t)));
    for (int q = 1; q < 15; q++) {
        ggml_tensor* ids_q = ggml_view_1d(ctx0, codes_inp, T, (size_t)(q + 1) * T * sizeof(int32_t));
        emb_rest = ggml_add(ctx0, emb_rest, ggml_get_rows(ctx0, codec.rvq_rest_cb[q], ids_q));
    }
    emb_rest = codec_causal_conv1d(ctx0, emb_rest, codec.rvq_rest_out_w, nullptr, 1, 1); // [512, T]

    ggml_tensor* h = ggml_add(ctx0, emb_first, emb_rest); // [512, T]

    // ── Step 2: pre_conv ────────────────────────────────────────────────────
    h = codec_causal_conv1d(ctx0, h, codec.pre_conv_w, codec.pre_conv_b, 1, 1); // [1024, T]

    // ── Step 3: transformer ─────────────────────────────────────────────────
    // input_proj: [1024, T] → [512, T]
    h = ggml_add(ctx0, ggml_mul_mat(ctx0, codec.xfmr_in_proj_w, h), codec.xfmr_in_proj_b);

    // Positions [0..T-1]
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "codec_positions");
    ggml_set_input(positions);

    // Causal mask [T, T] (nullptr → pass as tensor of shape [1,T] if T==1)
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
        ggml_set_name(causal_mask, "codec_mask");
        ggml_set_input(causal_mask);
    }

    const core_attn::LlamaSelfAttnParams asp = {
        n_heads, n_heads, hd, /*n_kv_grp*/ 1,
        (int)hp.max_pos, theta, attn_scale,
    };

    for (int il = 0; il < n_layers; il++) {
        const auto& bl = codec.xfmr_layers[il];
        ggml_tensor* residual = h;

        // Pre-attention RMSNorm
        ggml_tensor* x = ggml_rms_norm(ctx0, h, eps);
        x = ggml_mul(ctx0, x, bl.attn_norm_w);

        // Self-attention (full-sequence, no KV cache)
        ggml_tensor* attn = core_attn::llama_self_attn(ctx0, x, bl.attn_q_w, bl.attn_k_w, bl.attn_v_w,
                                                       bl.attn_o_w, positions, causal_mask, asp);
        // LayerScale + residual
        attn = ggml_mul(ctx0, attn, bl.attn_ls_w);
        h = ggml_add(ctx0, residual, attn);

        residual = h;
        // Pre-FFN RMSNorm
        x = ggml_rms_norm(ctx0, h, eps);
        x = ggml_mul(ctx0, x, bl.ffn_norm_w);

        // SwiGLU FFN
        ggml_tensor* ffn = core_ffn::swiglu(ctx0, x, bl.ffn_gate_w, bl.ffn_up_w, bl.ffn_down_w);
        // LayerScale + residual
        ffn = ggml_mul(ctx0, ffn, bl.ffn_ls_w);
        h = ggml_add(ctx0, residual, ffn);
    }

    // Final norm + output_proj: [512, T] → [1024, T]
    h = ggml_rms_norm(ctx0, h, eps);
    h = ggml_mul(ctx0, h, codec.xfmr_norm_w);
    h = ggml_add(ctx0, ggml_mul_mat(ctx0, codec.xfmr_out_proj_w, h), codec.xfmr_out_proj_b);

    // ── Step 4: ConvNeXt upsample (2 stages, each 2×) ──────────────────────
    for (int s = 0; s < 2; s++) {
        h = codec_transposed_conv1d(ctx0, h, codec.up[s].tconv_w, codec.up[s].tconv_b, 2);
        h = codec_convnext_block(ctx0, h, codec.up[s]);
    }

    // ── Step 5: Decoder blocks ──────────────────────────────────────────────
    h = codec_causal_conv1d(ctx0, h, codec.in_conv_w, codec.in_conv_b, 1, 1); // [1536, 4T]
    for (int b = 0; b < 4; b++)
        h = codec_dec_block(ctx0, h, codec.blocks[b], hp.upsample_rates[b]);

    // ── Step 6: Final conv and clamp ────────────────────────────────────────
    h = codec_snake_beta(ctx0, h, codec.out_snake_a, codec.out_snake_b);
    h = codec_causal_conv1d(ctx0, h, codec.out_conv_w, codec.out_conv_b, 1, 1); // [1, 1920T]
    h = ggml_clamp(ctx0, h, -1.0f, 1.0f);

    // Reshape to 1D [1920T]
    const int T_pcm = (int)h->ne[0] * (int)h->ne[1];
    h = ggml_reshape_1d(ctx0, h, T_pcm);
    ggml_set_name(h, "pcm");
    ggml_build_forward_expand(gf, h);
    ggml_free(ctx0);
    return gf;
}

// ---------------------------------------------------------------------------
// Load codec GGUF into g3t_codec.
// ---------------------------------------------------------------------------
static bool load_codec(qwen3_tts_context* c, const char* path) {
    auto& codec = c->codec;
    auto& hp = codec.hp;

    // Pass 1: read hyperparameters
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta) {
        fprintf(stderr, "qwen3_tts: codec: cannot open '%s'\n", path);
        return false;
    }
    hp.n_layers       = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.n_layers",     hp.n_layers);
    hp.d_model        = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.d_model",       hp.d_model);
    hp.n_heads        = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.n_heads",       hp.n_heads);
    hp.head_dim       = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.head_dim",      hp.head_dim);
    hp.ff_dim         = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.ff_dim",        hp.ff_dim);
    hp.n_q            = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.n_quantizers",  hp.n_q);
    hp.codebook_size  = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.codebook_size", hp.codebook_size);
    hp.latent_dim     = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.latent_dim",    hp.latent_dim);
    hp.decoder_dim    = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.decoder_dim",   hp.decoder_dim);
    hp.sliding_window = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.sliding_window",hp.sliding_window);
    hp.max_pos        = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.max_pos",       hp.max_pos);
    hp.rope_theta     = core_gguf::kv_f32(meta, "qwen3tts_codec.dec.rope_theta",    hp.rope_theta);
    hp.rms_norm_eps   = core_gguf::kv_f32(meta, "qwen3tts_codec.dec.rms_norm_eps",  hp.rms_norm_eps);
    core_gguf::free_metadata(meta);

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c->backend, "codec", wl)) {
        fprintf(stderr, "qwen3_tts: codec: failed to load weights from '%s'\n", path);
        return false;
    }
    codec.ctx_w  = wl.ctx;
    codec.buf_w  = wl.buf;
    codec.tensors = std::move(wl.tensors);

    auto req = [&](const char* n) -> ggml_tensor* {
        return core_gguf::require(codec.tensors, n, "codec");
    };
    auto fmt = [](const char* f, auto... a) -> std::string {
        char buf[256];
        snprintf(buf, sizeof(buf), f, a...);
        return buf;
    };

    // RVQ
    codec.rvq_first_cb    = req("codec.dec.rvq_first.codebook");
    codec.rvq_first_out_w = req("codec.dec.rvq_first.out_proj_w");
    for (int q = 0; q < 15; q++)
        codec.rvq_rest_cb[q] = req(fmt("codec.dec.rvq_rest.%d.codebook", q).c_str());
    codec.rvq_rest_out_w  = req("codec.dec.rvq_rest.out_proj_w");

    // pre_conv
    codec.pre_conv_w = req("codec.dec.pre_conv_w");
    codec.pre_conv_b = req("codec.dec.pre_conv_b");

    // Transformer
    codec.xfmr_in_proj_w  = req("codec.dec.xfmr.in_proj_w");
    codec.xfmr_in_proj_b  = req("codec.dec.xfmr.in_proj_b");
    codec.xfmr_norm_w     = req("codec.dec.xfmr.norm_w");
    codec.xfmr_out_proj_w = req("codec.dec.xfmr.out_proj_w");
    codec.xfmr_out_proj_b = req("codec.dec.xfmr.out_proj_b");

    codec.xfmr_layers.resize(hp.n_layers);
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        auto& bl = codec.xfmr_layers[il];
        bl.attn_norm_w = req(fmt("codec.dec.xfmr.blk.%u.attn_norm_w", il).c_str());
        bl.ffn_norm_w  = req(fmt("codec.dec.xfmr.blk.%u.ffn_norm_w",  il).c_str());
        bl.attn_q_w    = req(fmt("codec.dec.xfmr.blk.%u.attn_q_w",    il).c_str());
        bl.attn_k_w    = req(fmt("codec.dec.xfmr.blk.%u.attn_k_w",    il).c_str());
        bl.attn_v_w    = req(fmt("codec.dec.xfmr.blk.%u.attn_v_w",    il).c_str());
        bl.attn_o_w    = req(fmt("codec.dec.xfmr.blk.%u.attn_o_w",    il).c_str());
        bl.attn_ls_w   = req(fmt("codec.dec.xfmr.blk.%u.attn_ls_w",   il).c_str());
        bl.ffn_gate_w  = req(fmt("codec.dec.xfmr.blk.%u.ffn_gate_w",  il).c_str());
        bl.ffn_up_w    = req(fmt("codec.dec.xfmr.blk.%u.ffn_up_w",    il).c_str());
        bl.ffn_down_w  = req(fmt("codec.dec.xfmr.blk.%u.ffn_down_w",  il).c_str());
        bl.ffn_ls_w    = req(fmt("codec.dec.xfmr.blk.%u.ffn_ls_w",    il).c_str());
    }

    // Upsample stages
    for (int s = 0; s < 2; s++) {
        auto& up = codec.up[s];
        up.tconv_w = req(fmt("codec.dec.up.%d.tconv_w",    s).c_str());
        up.tconv_b = req(fmt("codec.dec.up.%d.tconv_b",    s).c_str());
        up.dw_w    = req(fmt("codec.dec.up.%d.cnx.dw_w",   s).c_str());
        up.dw_b    = req(fmt("codec.dec.up.%d.cnx.dw_b",   s).c_str());
        up.norm_w  = req(fmt("codec.dec.up.%d.cnx.norm_w", s).c_str());
        up.norm_b  = req(fmt("codec.dec.up.%d.cnx.norm_b", s).c_str());
        up.pw1_w   = req(fmt("codec.dec.up.%d.cnx.pw1_w",  s).c_str());
        up.pw1_b   = req(fmt("codec.dec.up.%d.cnx.pw1_b",  s).c_str());
        up.pw2_w   = req(fmt("codec.dec.up.%d.cnx.pw2_w",  s).c_str());
        up.pw2_b   = req(fmt("codec.dec.up.%d.cnx.pw2_b",  s).c_str());
        up.gamma   = req(fmt("codec.dec.up.%d.cnx.gamma",  s).c_str());
    }

    // Decoder in_conv
    codec.in_conv_w = req("codec.dec.in_conv_w");
    codec.in_conv_b = req("codec.dec.in_conv_b");

    // Decoder blocks
    for (int b = 0; b < 4; b++) {
        auto& blk = codec.blocks[b];
        blk.snake_a = req(fmt("codec.dec.blk.%d.snake_a", b).c_str());
        blk.snake_b = req(fmt("codec.dec.blk.%d.snake_b", b).c_str());
        blk.tconv_w = req(fmt("codec.dec.blk.%d.tconv_w", b).c_str());
        blk.tconv_b = req(fmt("codec.dec.blk.%d.tconv_b", b).c_str());
        for (int u = 0; u < 3; u++) {
            auto& ru = blk.res[u];
            ru.act1_a  = req(fmt("codec.dec.blk.%d.res.%d.act1_a",  b, u).c_str());
            ru.act1_b  = req(fmt("codec.dec.blk.%d.res.%d.act1_b",  b, u).c_str());
            ru.act2_a  = req(fmt("codec.dec.blk.%d.res.%d.act2_a",  b, u).c_str());
            ru.act2_b  = req(fmt("codec.dec.blk.%d.res.%d.act2_b",  b, u).c_str());
            ru.conv1_w = req(fmt("codec.dec.blk.%d.res.%d.conv1_w", b, u).c_str());
            ru.conv1_b = req(fmt("codec.dec.blk.%d.res.%d.conv1_b", b, u).c_str());
            ru.conv2_w = req(fmt("codec.dec.blk.%d.res.%d.conv2_w", b, u).c_str());
            ru.conv2_b = req(fmt("codec.dec.blk.%d.res.%d.conv2_b", b, u).c_str());
        }
    }

    // Final snake and output conv
    codec.out_snake_a = req("codec.dec.out_snake_a");
    codec.out_snake_b = req("codec.dec.out_snake_b");
    codec.out_conv_w  = req("codec.dec.out_conv_w");
    codec.out_conv_b  = req("codec.dec.out_conv_b");

    // Codec compute metadata
    c->codec_compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));

    codec.loaded = true;
    if (c->params.verbosity >= 1)
        fprintf(stderr, "qwen3_tts: codec loaded from '%s'  (%uL d=%u/%u  rvq=%u)\n",
                path, hp.n_layers, hp.d_model, hp.latent_dim, hp.n_q);
    return true;
}

// ---------------------------------------------------------------------------
// Execute codec decode: codes[T_codec × n_q] → malloc'd float32 PCM.
// Input codes layout: [T_codec, n_q] row-major (T frames, each with n_q codes).
// Output: [T_pcm] float32 @ 24 kHz, caller frees with free().
// ---------------------------------------------------------------------------
static float* codec_decode_codes(qwen3_tts_context* c, const int32_t* codes, int T_codec, int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    const auto& codec = c->codec;
    const auto& hp = codec.hp;
    const int n_q = (int)hp.n_q;

    // Transpose [T, n_q] → [n_q, T] so each codebook is a contiguous row.
    std::vector<int32_t> codes_t((size_t)T_codec * n_q);
    for (int q = 0; q < n_q; q++)
        for (int t = 0; t < T_codec; t++)
            codes_t[(size_t)q * T_codec + t] = codes[(size_t)t * n_q + q];

    // Build sliding-window causal mask
    const int window = (int)hp.sliding_window;
    std::vector<ggml_fp16_t> mask_data;
    if (T_codec > 1) {
        const ggml_fp16_t zero_h   = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask_data.assign((size_t)T_codec * T_codec, neginf_h);
        for (int q = 0; q < T_codec; q++) {
            for (int k = 0; k <= q; k++) {
                if ((q - k) < window)
                    mask_data[(size_t)q * T_codec + k] = zero_h;
            }
        }
    }

    // Build positions [0..T-1]
    std::vector<int32_t> pos(T_codec);
    for (int i = 0; i < T_codec; i++)
        pos[i] = i;

    ggml_cgraph* gf = build_graph_codec_decode(c, T_codec);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "qwen3_tts: codec: graph alloc failed\n");
        return nullptr;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codec_codes"),
                            codes_t.data(), 0, codes_t.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codec_positions"),
                            pos.data(), 0, pos.size() * sizeof(int32_t));
    if (T_codec > 1)
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codec_mask"),
                                mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_tts: codec: compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "pcm");
    const int n_samples = (int)ggml_nelements(out);
    float* pcm = (float*)malloc((size_t)n_samples * sizeof(float));
    if (!pcm)
        return nullptr;
    ggml_backend_tensor_get(out, pcm, 0, (size_t)n_samples * sizeof(float));
    if (out_n_samples)
        *out_n_samples = n_samples;
    return pcm;
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
    if (!load_codec(ctx, path))
        return -1;
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
    float* logits = run_talker_kv(ctx, embeds, n_tokens, /*n_past=*/0, /*out_hidden_d=*/nullptr);
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
    if (ctx->vp_active < 0) {
        fprintf(stderr, "qwen3_tts: no voice loaded — call qwen3_tts_load_voice_pack + select_voice first\n");
        return nullptr;
    }

    const bool bench = env_bool("QWEN3_TTS_BENCH");
    const bool dbg = env_bool("QWEN3_TTS_DEBUG");
    const char* dump_dir = env_str("QWEN3_TTS_DUMP_DIR");
    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;
    const int n_groups = (int)hp.n_code_groups; // 16
    const int max_frames = ctx->params.max_codec_steps > 0 ? ctx->params.max_codec_steps : 1500;
    const int eos = (int)hp.codec_eos_id;

    // PRNG seed — env-overridable for reproducibility. Default seed
    // matches PyTorch's behaviour with `torch.manual_seed(42)`.
    uint64_t rng = 42;
    if (const char* s = env_str("QWEN3_TTS_SEED"))
        rng = (uint64_t)std::strtoull(s, nullptr, 10);

    // ---- ref_text comes from the active voice pack ----
    const std::string& ref_text = ctx->vp_ref_texts[ctx->vp_active];
    if (ref_text.empty()) {
        fprintf(stderr, "qwen3_tts: voice '%s' has no ref_text\n", ctx->vp_names[ctx->vp_active].c_str());
        return nullptr;
    }

    // ---- ICL prefill builder (matches PyTorch's generate_icl_prompt) ----
    double t0 = bench ? now_ms() : 0.0;
    std::vector<float> prefill, trailing;
    int T_pre = 0, M_trail = 0;
    if (!build_icl_prefill_embeds(ctx, text, ref_text, prefill, T_pre, trailing, M_trail))
        return nullptr;
    if (bench)
        fprintf(stderr, "qwen3_tts: icl_prefill %7.1f ms (T=%d)\n", now_ms() - t0, T_pre);
    if (dump_dir)
        dump_f32(dump_dir, "icl_prefill", prefill.data(), prefill.size());

    // ---- talker prefill: get logits + hidden_last ----
    double t1 = bench ? now_ms() : 0.0;
    float* past_hidden = nullptr;
    float* logits = run_talker_kv(ctx, prefill.data(), T_pre, /*n_past=*/0, &past_hidden);
    if (!logits || !past_hidden) {
        free(logits);
        free(past_hidden);
        return nullptr;
    }
    if (bench)
        fprintf(stderr, "qwen3_tts: talker_pre %7.1f ms\n", now_ms() - t1);
    if (dump_dir)
        dump_f32(dump_dir, "talker_prefill_logits", logits, hp.vocab_size);

    int n_past = T_pre;

    // ---- AR loop: 1 talker step + 15 code_predictor steps per frame ----
    if (!cp_kv_alloc(ctx)) {
        fprintf(stderr, "qwen3_tts: cp_kv allocation failed\n");
        free(logits);
        free(past_hidden);
        return nullptr;
    }

    std::vector<int32_t> all_codes; // flattened (T_frames, 16)
    all_codes.reserve((size_t)max_frames * n_groups);

    double t_loop = bench ? now_ms() : 0.0;
    int frame = 0;
    for (frame = 0; frame < max_frames; frame++) {
        // 1. Sample codebook-0 from talker logits.
        int cb0 = argmax(logits, (int)hp.vocab_size);
        free(logits);
        logits = nullptr;
        if (cb0 == eos) {
            if (dbg)
                fprintf(stderr, "qwen3_tts: codec_eos at frame %d\n", frame);
            free(past_hidden);
            past_hidden = nullptr;
            break;
        }
        // 2. Embed cb0 via talker.codec_embedding → last_id_hidden (d,)
        float* last_id_hidden = lookup_rows(ctx, ctx->talker.token_embd_w, &cb0, 1);
        if (!last_id_hidden) {
            free(past_hidden);
            return nullptr;
        }
        // 3. Code predictor AR loop → 15 more codebook ids (sampled).
        int32_t cb1_15[15];
        if (!code_pred_generate_15(ctx, past_hidden, last_id_hidden, cb1_15, &rng)) {
            free(past_hidden);
            free(last_id_hidden);
            return nullptr;
        }
        free(past_hidden);
        past_hidden = nullptr;

        // 4. Append the full 16-codebook frame.
        all_codes.push_back(cb0);
        for (int i = 0; i < 15; i++)
            all_codes.push_back(cb1_15[i]);

        // 5. Build next talker input:
        //    sum_{cb=0..15}(codec_embd_for_cb(frame[cb])) + trailing[step]
        //    where trailing[step] = trailing_text_hidden[gen_step] if gen_step
        //    < M else tts_pad_embed (only the latter when codec_lens > text_lens).
        std::vector<float> next_emb(d, 0.0f);
        for (int cb = 0; cb < n_groups; cb++) {
            int32_t code = (cb == 0) ? cb0 : cb1_15[cb - 1];
            ggml_tensor* w = (cb == 0) ? ctx->talker.token_embd_w : ctx->code_pred.codec_embd[cb - 1];
            float* row = lookup_rows(ctx, w, &code, 1);
            if (!row) {
                free(last_id_hidden);
                return nullptr;
            }
            for (int j = 0; j < d; j++)
                next_emb[j] += row[j];
            free(row);
        }
        free(last_id_hidden);

        // Add trailing_text_hidden[gen_step] (or last row if past M).
        const int trail_idx = std::min(frame, M_trail - 1);
        const float* trail = trailing.data() + (size_t)trail_idx * d;
        for (int j = 0; j < d; j++)
            next_emb[j] += trail[j];

        // 6. Talker forward on the (1, d) input → next logits + hidden_last.
        if (n_past >= ctx->kv_max_ctx - 1) {
            fprintf(stderr, "qwen3_tts: talker kv cache full at frame %d (n_past=%d)\n", frame, n_past);
            break;
        }
        logits = run_talker_kv(ctx, next_emb.data(), 1, n_past, &past_hidden);
        if (!logits || !past_hidden) {
            free(logits);
            free(past_hidden);
            return nullptr;
        }
        n_past += 1;
    }
    free(logits);
    free(past_hidden);

    if (bench)
        fprintf(stderr, "qwen3_tts: ar_loop    %7.1f ms (%d frames, %.1f ms/frame)\n", now_ms() - t_loop, frame,
                frame > 0 ? (now_ms() - t_loop) / frame : 0.0);
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "qwen3_tts: produced %d frames × 16 codebooks = %zu codes\n", frame, all_codes.size());

    *out_n_codes = (int)all_codes.size();
    int32_t* out = (int32_t*)malloc(all_codes.size() * sizeof(int32_t));
    std::memcpy(out, all_codes.data(), all_codes.size() * sizeof(int32_t));
    return out;
}

extern "C" void qwen3_tts_codes_free(int32_t* codes) {
    free(codes);
}

extern "C" float* qwen3_tts_decode_codes(struct qwen3_tts_context* ctx, const int32_t* codes, int n_codes,
                                         int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!ctx || !codes || n_codes <= 0)
        return nullptr;
    if (!ctx->codec.loaded) {
        fprintf(stderr, "qwen3_tts: decode_codes() requires codec — call qwen3_tts_set_codec_path() first.\n");
        return nullptr;
    }
    const int n_q = (int)ctx->codec.hp.n_q;
    if (n_codes % n_q != 0) {
        fprintf(stderr, "qwen3_tts: decode_codes: n_codes %d not divisible by n_q %d\n", n_codes, n_q);
        return nullptr;
    }
    const int T_codec = n_codes / n_q;
    return codec_decode_codes(ctx, codes, T_codec, out_n_samples);
}

extern "C" float* qwen3_tts_synthesize(struct qwen3_tts_context* ctx, const char* text, int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!ctx)
        return nullptr;
    if (!ctx->codec.loaded) {
        fprintf(stderr, "qwen3_tts: synthesize() requires the codec — call qwen3_tts_set_codec_path() first.\n");
        return nullptr;
    }
    int n_codes = 0;
    int32_t* codes = qwen3_tts_synthesize_codes(ctx, text, &n_codes);
    if (!codes || n_codes <= 0) {
        free(codes);
        return nullptr;
    }
    const int n_q = (int)ctx->codec.hp.n_q;
    if (n_codes % n_q != 0) {
        fprintf(stderr, "qwen3_tts: synthesize: unexpected code count %d (n_q=%d)\n", n_codes, n_q);
        free(codes);
        return nullptr;
    }
    const int T_codec = n_codes / n_q;
    float* pcm = codec_decode_codes(ctx, codes, T_codec, out_n_samples);
    free(codes);
    return pcm;
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
    if (ctx->cp_kv_buf)
        ggml_backend_buffer_free(ctx->cp_kv_buf);
    if (ctx->cp_kv_ctx)
        ggml_free(ctx->cp_kv_ctx);
    if (ctx->codec.buf_w)
        ggml_backend_buffer_free(ctx->codec.buf_w);
    if (ctx->codec.ctx_w)
        ggml_free(ctx->codec.ctx_w);
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
