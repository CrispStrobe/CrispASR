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
//     range). Pinned to backend_cpu via a dedicated `codec_sched`:
//     Metal hangs on M1 (kIOGPUCommandBufferCallbackErrorImpactingInteractivity).
//     Root cause isolated via QWEN3_TTS_CODEC_FORCE_METAL=1 + per-op
//     trace: ggml-Metal hangs on GGML_OP_CONV_TRANSPOSE_1D in decoder
//     block 1 (stride=5, output [1605, 384]); block 0 (stride=8,
//     output [320, 768]) is fine. Likely a ggml-Metal kernel bug at
//     large transposed-conv output sizes (file upstream). Talker /
//     code_predictor still run on Metal when use_gpu=true.
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/mel.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
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
    if (!dir || !data || !n) {
        return;
    }
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
    if (!dir || !data || !n) {
        return;
    }
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

struct qwen3_op_prof {
    int64_t t_us = 0;
    int count = 0;
};

struct qwen3_prof_state {
    qwen3_op_prof mul_mat;
    qwen3_op_prof flash_attn;
    qwen3_op_prof norm;
    qwen3_op_prof rope;
    qwen3_op_prof add;
    qwen3_op_prof unary;
    qwen3_op_prof cpy;
    qwen3_op_prof cont;
    qwen3_op_prof get_rows;
    qwen3_op_prof repeat;
    qwen3_op_prof other;
    int64_t t_node_start = 0;
};

static bool qwen3_prof_eval_cb(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* ps = (qwen3_prof_state*)user_data;
    if (ask) {
        ps->t_node_start = ggml_time_us();
        return true;
    }
    const int64_t dt = ggml_time_us() - ps->t_node_start;
    switch (t->op) {
    case GGML_OP_MUL_MAT:
        ps->mul_mat.t_us += dt;
        ps->mul_mat.count++;
        break;
    case GGML_OP_FLASH_ATTN_EXT:
        ps->flash_attn.t_us += dt;
        ps->flash_attn.count++;
        break;
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM:
        ps->norm.t_us += dt;
        ps->norm.count++;
        break;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
        ps->rope.t_us += dt;
        ps->rope.count++;
        break;
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_SCALE:
        ps->add.t_us += dt;
        ps->add.count++;
        break;
    case GGML_OP_UNARY:
        ps->unary.t_us += dt;
        ps->unary.count++;
        break;
    case GGML_OP_CPY:
        ps->cpy.t_us += dt;
        ps->cpy.count++;
        break;
    case GGML_OP_CONT:
        ps->cont.t_us += dt;
        ps->cont.count++;
        break;
    case GGML_OP_GET_ROWS:
        ps->get_rows.t_us += dt;
        ps->get_rows.count++;
        break;
    case GGML_OP_REPEAT:
        ps->repeat.t_us += dt;
        ps->repeat.count++;
        break;
    default:
        ps->other.t_us += dt;
        ps->other.count++;
        break;
    }
    return true;
}

static void qwen3_prof_add(qwen3_prof_state& dst, const qwen3_prof_state& src) {
    dst.mul_mat.t_us += src.mul_mat.t_us;
    dst.mul_mat.count += src.mul_mat.count;
    dst.flash_attn.t_us += src.flash_attn.t_us;
    dst.flash_attn.count += src.flash_attn.count;
    dst.norm.t_us += src.norm.t_us;
    dst.norm.count += src.norm.count;
    dst.rope.t_us += src.rope.t_us;
    dst.rope.count += src.rope.count;
    dst.add.t_us += src.add.t_us;
    dst.add.count += src.add.count;
    dst.unary.t_us += src.unary.t_us;
    dst.unary.count += src.unary.count;
    dst.cpy.t_us += src.cpy.t_us;
    dst.cpy.count += src.cpy.count;
    dst.cont.t_us += src.cont.t_us;
    dst.cont.count += src.cont.count;
    dst.get_rows.t_us += src.get_rows.t_us;
    dst.get_rows.count += src.get_rows.count;
    dst.repeat.t_us += src.repeat.t_us;
    dst.repeat.count += src.repeat.count;
    dst.other.t_us += src.other.t_us;
    dst.other.count += src.other.count;
}

static void qwen3_prof_print(const char* tag, const qwen3_prof_state& ps, int n_calls) {
    auto pct = [&](int64_t v, int64_t total) { return total > 0 ? 100.0 * v / total : 0.0; };
    const int64_t total = ps.mul_mat.t_us + ps.flash_attn.t_us + ps.norm.t_us + ps.rope.t_us + ps.add.t_us +
                          ps.unary.t_us + ps.cpy.t_us + ps.cont.t_us + ps.get_rows.t_us + ps.repeat.t_us +
                          ps.other.t_us;
    fprintf(stderr, "qwen3_tts: ---- %s op profile (QWEN3_TTS_PROF, %d calls) ----\n", tag, n_calls);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms %5.1f%% n=%d\n", "mul_mat", ps.mul_mat.t_us / 1e3,
            pct(ps.mul_mat.t_us, total), ps.mul_mat.count);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms %5.1f%% n=%d\n", "flash_attn", ps.flash_attn.t_us / 1e3,
            pct(ps.flash_attn.t_us, total), ps.flash_attn.count);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms %5.1f%% n=%d\n", "norm", ps.norm.t_us / 1e3,
            pct(ps.norm.t_us, total), ps.norm.count);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms %5.1f%% n=%d\n", "rope", ps.rope.t_us / 1e3,
            pct(ps.rope.t_us, total), ps.rope.count);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms %5.1f%% n=%d\n", "add/mul/sc", ps.add.t_us / 1e3,
            pct(ps.add.t_us, total), ps.add.count);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms %5.1f%% n=%d\n", "unary", ps.unary.t_us / 1e3,
            pct(ps.unary.t_us, total), ps.unary.count);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms %5.1f%% n=%d\n", "cpy", ps.cpy.t_us / 1e3, pct(ps.cpy.t_us, total),
            ps.cpy.count);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms %5.1f%% n=%d\n", "cont", ps.cont.t_us / 1e3,
            pct(ps.cont.t_us, total), ps.cont.count);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms %5.1f%% n=%d\n", "get_rows", ps.get_rows.t_us / 1e3,
            pct(ps.get_rows.t_us, total), ps.get_rows.count);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms %5.1f%% n=%d\n", "repeat", ps.repeat.t_us / 1e3,
            pct(ps.repeat.t_us, total), ps.repeat.count);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms %5.1f%% n=%d\n", "other", ps.other.t_us / 1e3,
            pct(ps.other.t_us, total), ps.other.count);
    fprintf(stderr, "qwen3_tts:  %-12s %7.1f ms\n", "TOTAL", total / 1e3);
    fprintf(stderr, "qwen3_tts: -----------------------------------------------\n");
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
    ggml_tensor* attn_qkv_w = nullptr; // fused Q+K+V (F16/F32 talker only)
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
// Codec encoder structs (SEANet + encoder transformer + RVQ encode)
// ---------------------------------------------------------------------------

struct g3t_cenc_conv {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
};
struct g3t_cenc_resblk {
    g3t_cenc_conv shortcut, expand;
};

struct g3t_cenc_seanet {
    g3t_cenc_conv init;        // layers.0 — 1→64, k=7
    g3t_cenc_resblk resblk[4]; // stages 0..3
    g3t_cenc_conv ds[4];       // stride convs: strides 4,5,6,8
    g3t_cenc_conv final;       // layers.14 — 1024→512, k=3
};

struct g3t_cenc_xfmr_layer {
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_ls = nullptr; // layer_scale
    ggml_tensor* fc1_w = nullptr;   // FFN up 512→2048
    ggml_tensor* fc2_w = nullptr;   // FFN down 2048→512
    ggml_tensor* ffn_ls = nullptr;  // layer_scale
};

struct g3t_cenc_rvq_cb {
    std::vector<float> data;
}; // [2048, 256] row-major

struct g3t_cenc {
    g3t_cenc_seanet seanet;
    std::vector<g3t_cenc_xfmr_layer> xfmr_layers; // 8 layers
    g3t_cenc_conv downsample;                     // stride-2, k=4
    // Quantizer weights (CPU-side for iterative encode)
    g3t_cenc_conv sem_in_proj;  // 512→256 (k=1 conv stored as matrix)
    g3t_cenc_conv sem_out_proj; // 256→512
    g3t_cenc_conv ac_in_proj;   // 512→256
    g3t_cenc_conv ac_out_proj;  // 256→512
    g3t_cenc_rvq_cb sem_cb;     // semantic codebook [2048×256]
    g3t_cenc_rvq_cb ac_cb[15];  // acoustic codebooks [2048×256] ×15
    bool loaded = false;
};

// ---------------------------------------------------------------------------
// Speaker encoder structs (PLAN #52 step 4 — ECAPA-TDNN)
// ---------------------------------------------------------------------------

struct g3t_spk_tdnn_w {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
};
struct g3t_spk_res2net {
    g3t_spk_tdnn_w blocks[7];
}; // scale=8 → 7 inner TDNNs
struct g3t_spk_se {
    ggml_tensor* c1w = nullptr;
    ggml_tensor* c1b = nullptr;
    ggml_tensor* c2w = nullptr;
    ggml_tensor* c2b = nullptr;
};
struct g3t_spk_se_res2net {
    g3t_spk_tdnn_w tdnn1, tdnn2;
    g3t_spk_res2net res2net;
    g3t_spk_se se;
};
struct g3t_spk_asp {
    g3t_spk_tdnn_w tdnn; // (3C, 128, k=1)
    ggml_tensor* conv_w = nullptr;
    ggml_tensor* conv_b = nullptr; // (C, 128, k=1)
};
struct g3t_spk_enc {
    g3t_spk_tdnn_w blk0;       // initial TDNN: 128→512, k=5, d=1
    g3t_spk_se_res2net blk[3]; // 3 SE-Res2Net blocks, d=2/3/4
    g3t_spk_tdnn_w mfa;        // 1536→1536, k=1
    g3t_spk_asp asp;           // attentive-statistics pooling
    ggml_tensor* fc_w = nullptr;
    ggml_tensor* fc_b = nullptr; // 3072→1024
    bool loaded = false;
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
    ggml_backend_sched_t cp_sched = nullptr;
    bool cp_sched_reserved = false;
    // When cp_cpu_pinned=true, code_pred transformer weights (lm_head + blocks)
    // live in cp_cpu_buf/cp_cpu_ctx on the CPU backend, and cp_sched (CPU-only)
    // is used for code_pred forward passes.  This eliminates 15 Metal command-
    // buffer round-trips per AR frame.  talker_embd_cpu is a CPU copy of
    // talker.token_embd_w used for the AR-loop embed lookup (the original Metal
    // tensor is untouched and still used by the ICL prefill path).
    ggml_context* cp_cpu_ctx = nullptr;
    ggml_backend_buffer_t cp_cpu_buf = nullptr;
    ggml_tensor* talker_embd_cpu = nullptr;
    bool cp_cpu_pinned = false;

    // CPU-side embedding caches: raw quantized bytes copied from the Metal
    // weight tensors at init.  Used by the AR loop to dequantize embedding
    // rows directly on CPU, eliminating ~17 Metal command-buffer round-trips
    // per AR frame (~81 ms saved).  ICL prefill / sum_codec_embeds still use
    // the original Metal tensors (they run once per synthesis, not per frame).
    struct CpuEmbdCache {
        ggml_type type = GGML_TYPE_F32;
        int n_rows = 0;
        int d = 0;
        size_t row_bytes = 0;
        std::vector<uint8_t> data;

        bool init(ggml_tensor* src) {
            if (!src) {
                return false;
            }
            type = src->type;
            n_rows = (int)src->ne[1];
            d = (int)src->ne[0];
            row_bytes = ggml_row_size(type, d);
            data.resize((size_t)n_rows * row_bytes);
            ggml_backend_tensor_get(src, data.data(), 0, data.size());
            return true;
        }

        // Dequantize row `id` into dst[0..d-1].  Returns false on bad id.
        bool get_row_into(int32_t id, float* dst) const {
            if (id < 0 || id >= n_rows || data.empty()) {
                return false;
            }
            const struct ggml_type_traits* tr = ggml_get_type_traits(type);
            if (!tr || !tr->to_float) {
                return false;
            }
            tr->to_float(data.data() + (size_t)id * row_bytes, dst, d);
            return true;
        }

        explicit operator bool() const { return !data.empty(); }
    };
    CpuEmbdCache token_embd_cache;              // mirrors talker.token_embd_w
    std::vector<CpuEmbdCache> codec_embd_cache; // mirrors code_pred.codec_embd[0..14]
    // Separate CPU-only scheduler for the codec decode graph. Metal hangs
    // on M1 with the codec's ggml_conv_1d_dw + SnakeBeta (sin/exp) chain
    // (kIOGPUCommandBufferCallbackErrorImpactingInteractivity), so codec
    // weights are pinned to backend_cpu and run through this scheduler
    // independently of params.use_gpu. Talker / code_predictor still use
    // `sched` (GPU-accelerated when available).
    ggml_backend_sched_t codec_sched = nullptr;

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

    // O15: persistent T=1 code_pred graph. cp_lm_head_slot is a writable Metal
    // buffer into which we blit lm_head[i] before each T=1 step, allowing all
    // 13 skip steps per frame to reuse the cached graph (no reset/alloc).
    ggml_context*         cp_lm_slot_ctx  = nullptr;
    ggml_backend_buffer_t cp_lm_slot_buf  = nullptr;
    ggml_tensor*          cp_lm_head_slot = nullptr;
    ggml_cgraph*          cp_t1_gf        = nullptr;

    // Fused Q+K+V weights for the talker (F16/F32 only, runtime-built from
    // the unfused tensors so Q4_K/Q8_0 talkers fall back to the 3-matmul path).
    ggml_context*         fused_ctx = nullptr;
    ggml_backend_buffer_t fused_buf = nullptr;

    // Loaded voice pack (zero-copy: `vp_tensors` references the
    // weight context's tensors directly).
    std::vector<std::string> vp_names;
    std::vector<std::string> vp_ref_texts;
    std::map<std::string, ggml_tensor*> vp_tensors;
    ggml_context* vp_ctx_w = nullptr;
    ggml_backend_buffer_t vp_buf_w = nullptr;
    int vp_active = -1; // index into vp_names; -1 = none selected

    int language_id = -1; // codec language id (-1 = auto / nothink path)

    std::string codec_path;
    std::string voice_prompt_path;

    g3t_codec codec;
    std::vector<uint8_t> codec_compute_meta;

    g3t_spk_enc spk_enc;
    std::vector<float> runtime_spk_emb; // set by qwen3_tts_set_voice_prompt
    std::vector<uint8_t> spk_compute_meta;

    g3t_cenc cenc;                          // codec encoder (SEANet → enc transformer → downsample → RVQ)
    std::vector<int32_t> runtime_ref_codes; // [T_codec * 16] set by set_voice_prompt
    std::string runtime_ref_text;           // ref text for ICL prefill
    std::vector<uint8_t> cenc_compute_meta;
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
    if (!t) {
        fprintf(stderr, "qwen3_tts: required tensor missing: %s\n", name);
    }
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
    if (!t.token_embd_w || !t.token_embd_text_w || !t.text_proj_fc1_w || !t.output_norm_w || !t.codec_head_w) {
        return false;
    }
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
        if (!b.attn_q_w || !b.ffn_gate_w) {
            return false;
        }
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
    for (const auto& sp : specials) {
        if (sp.id > max_id) {
            max_id = sp.id;
        }
    }
    if ((int)v.id_to_token.size() <= max_id) {
        v.id_to_token.resize((size_t)max_id + 1);
    }
    for (const auto& sp : specials) {
        auto old_it = v.token_to_id.find(v.id_to_token[sp.id]);
        if (old_it != v.token_to_id.end() && old_it->second == sp.id) {
            v.token_to_id.erase(old_it);
        }
        v.id_to_token[sp.id] = sp.text;
        v.token_to_id[sp.text] = sp.id;
    }
}

bool kv_alloc(qwen3_tts_context* c, int max_ctx) {
    if (c->kv_k) {
        return true;
    }
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
    if (!c->kv_buf) {
        return false;
    }
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
    h = ggml_mul_mat(ctx0, c->talker.text_proj_fc1_w, h);                   // (T, 2048)
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

ggml_cgraph* build_graph_talker_kv(qwen3_tts_context* c, int n_past, int n_tokens);

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
    // Use a fixed Lk = cp_kv_max_ctx for all code_pred graphs so that all
    // T=1 steps share the same tensor topology.  The gallocr plan is computed
    // once (at init reservation) and reused for every subsequent alloc,
    // eliminating per-step re-planning overhead (~7 ms/frame for Q8_0).
    const int Lk = c->cp_kv_max_ctx;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Always provide a causal mask (even for T=1 decode steps) so that
    // positions beyond the actual context window are masked to -inf.
    // This keeps the (Lk, T) mask shape fixed across all calls with the
    // same T, enabling gallocr plan reuse.
    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    const core_attn::KvSelfAttnParams kvp = {
        n_q, n_kv, hd, n_kv_grp, (int)hp.max_pos, theta, 32.0f, 1.0f, attn_scale, eps, core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.cp_n_layers; il++) {
        const auto& b = c->code_pred.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // For T=1 graphs, fix the KV-read length to cp_kv_max_ctx so that all
        // T=1 steps share the same graph topology (enabling graph reuse in O15).
        const int fixed_kv = (T == 1) ? c->cp_kv_max_ctx : 0;
        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, b.attn_q_norm_w, b.attn_k_norm_w,
            positions, causal_mask, c->cp_kv_k, c->cp_kv_v, (int)il, n_past, kvp,
            /*qkv_w=*/nullptr, /*fixed_kv_len=*/fixed_kv);
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
    // For T=1 graphs, use the writable slot so we can reuse the graph across
    // steps by blitting different lm_head weights in before each compute.
    ggml_tensor* eff_lm = (T == 1 && c->cp_lm_head_slot) ? c->cp_lm_head_slot : lm_head;
    ggml_tensor* logits = ggml_mul_mat(ctx0, eff_lm, cur);
    ggml_set_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);
    ggml_free(ctx0);
    return gf;
}

static ggml_backend_sched_t code_pred_pick_sched(qwen3_tts_context* c) {
    const char* cp_be = env_str("QWEN3_TTS_CP_BACKEND");
    if (cp_be && std::strncmp(cp_be, "cpu", 3) == 0 && c->cp_cpu_pinned && c->cp_sched) {
        return c->cp_sched;
    }
    return c->sched;
}

static bool code_pred_reserve_sched(qwen3_tts_context* c, ggml_backend_sched_t sched) {
    if (c->cp_sched_reserved || !c->code_pred.lm_head[0]) {
        return true;
    }
    ggml_cgraph* gf = build_graph_code_pred_kv(c, /*n_past=*/0, /*n_tokens=*/2, c->code_pred.lm_head[0]);
    if (!gf) {
        return false;
    }
    if (!ggml_backend_sched_reserve(sched, gf)) {
        fprintf(stderr, "qwen3_tts: code_pred reserve failed\n");
        return false;
    }
    c->cp_sched_reserved = true;
    return true;
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

        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, b.attn_q_norm_w, b.attn_k_norm_w,
            positions, (T == 1) ? nullptr : causal_mask, c->kv_k, c->kv_v, (int)il, n_past, kvp, b.attn_qkv_w);
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
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        return nullptr;
    }
    ggml_tensor* in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(in, ids, 0, (size_t)n * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* r = (float*)malloc((size_t)d * n * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)d * n * sizeof(float));
    return r;
}

float* run_embed_audio(qwen3_tts_context* c, const int32_t* ids, int n) {
    const int d = (int)c->hp.d_model;
    ggml_cgraph* gf = build_graph_embed_audio(c, n);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        return nullptr;
    }
    ggml_tensor* in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(in, ids, 0, (size_t)n * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        return nullptr;
    }
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
    if (out_hidden_d) {
        *out_hidden_d = nullptr;
    }
    if (n_past + n_tokens > c->kv_max_ctx) {
        fprintf(stderr, "qwen3_tts: kv overflow (%d+%d > %d)\n", n_past, n_tokens, c->kv_max_ctx);
        return nullptr;
    }
    const auto& hp = c->hp;
    const int d = (int)hp.d_model;
    const int vocab = (int)hp.vocab_size;
    const int Lk = n_past + n_tokens;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        positions[i] = n_past + i;
    }

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * n_tokens, zero_h);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = n_past + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neginf_h;
            }
        }
    }

    // Invalidate any cached T=1 code_pred graph — build_graph_talker_kv will
    // overwrite compute_meta, making cp_t1_gf stale.
    c->cp_t1_gf = nullptr;

    const bool bench = env_bool("QWEN3_TTS_BENCH");
    const bool prof = env_bool("QWEN3_TTS_PROF");
    const double t_build0 = bench ? now_ms() : 0.0;
    ggml_cgraph* gf = build_graph_talker_kv(c, n_past, n_tokens);
    const double t_build1 = bench ? now_ms() : 0.0;
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        return nullptr;
    }
    const double t_alloc1 = bench ? now_ms() : 0.0;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    qwen3_prof_state prof_state;
    if (prof) {
        ggml_backend_sched_set_eval_callback(c->sched, qwen3_prof_eval_cb, &prof_state);
    }
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        if (prof) {
            ggml_backend_sched_set_eval_callback(c->sched, nullptr, nullptr);
        }
        fprintf(stderr, "qwen3_tts: talker compute failed\n");
        return nullptr;
    }
    if (prof) {
        ggml_backend_sched_set_eval_callback(c->sched, nullptr, nullptr);
    }
    const double t_compute1 = bench ? now_ms() : 0.0;
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
    if (bench) {
        static double sum_build = 0.0, sum_alloc = 0.0, sum_compute = 0.0, sum_read = 0.0;
        static int count = 0;
        sum_build += t_build1 - t_build0;
        sum_alloc += t_alloc1 - t_build1;
        sum_compute += t_compute1 - t_alloc1;
        sum_read += now_ms() - t_compute1;
        count++;
        if (count == 14) {
            fprintf(stderr,
                    "qwen3_tts: talker_kv    (%d calls): build=%.1f ms  alloc=%.1f ms  compute=%.1f ms  read=%.1f ms\n",
                    count, sum_build, sum_alloc, sum_compute, sum_read);
            sum_build = sum_alloc = sum_compute = sum_read = 0.0;
            count = 0;
        }
    }
    if (prof) {
        static qwen3_prof_state sum_prof;
        static int count = 0;
        qwen3_prof_add(sum_prof, prof_state);
        count++;
        if (count == 14) {
            qwen3_prof_print("talker_kv", sum_prof, count);
            sum_prof = {};
            count = 0;
        }
    }
    return r;
}

int argmax(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = i;
        }
    }
    return best;
}

// Hugging Face repetition penalty processor:
//   - positive logits are divided by the penalty
//   - negative logits are multiplied by the penalty
// Applied once per distinct previously generated token id.
static void apply_repetition_penalty(float* logits, int n, const std::vector<int32_t>& prev_tokens, float penalty) {
    if (!logits || n <= 0 || penalty <= 1.0f || prev_tokens.empty()) {
        return;
    }
    std::vector<uint8_t> seen((size_t)n, 0);
    for (int32_t tok : prev_tokens) {
        if (tok < 0 || tok >= n || seen[(size_t)tok]) {
            continue;
        }
        seen[(size_t)tok] = 1;
        if (logits[tok] < 0.0f) {
            logits[tok] *= penalty;
        } else {
            logits[tok] /= penalty;
        }
    }
}

// Top-k + temperature sampler. Required for the code_predictor —
// `subtalker_dosample=True` is the official default and greedy
// argmax for codebooks 1..15 produces a degenerate / silent output
// (verified empirically against the qwen-tts reference).
//
// The PyTorch defaults are top_k=50, top_p=1.0, temperature=0.9.
// We implement top_k + temperature; top_p=1.0 is a no-op so omit it.
int top_k_sample(const float* logits, int n, int top_k, float temperature, uint64_t* rng_state) {
    if (top_k <= 0 || top_k >= n) {
        top_k = n;
    }
    // Find top-k indices via partial sort. For n=2048, top_k=50,
    // O(n log k) is fine (~50 µs).
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) {
        idx[i] = i;
    }
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });

    // Softmax over the top-k logits with temperature.
    const float t = temperature > 0 ? temperature : 1.0f;
    float max_l = logits[idx[0]];
    for (int i = 1; i < top_k; i++) {
        if (logits[idx[i]] > max_l) {
            max_l = logits[idx[i]];
        }
    }
    std::vector<float> probs(top_k);
    double sum = 0.0;
    for (int i = 0; i < top_k; i++) {
        double p = std::exp((logits[idx[i]] - max_l) / t);
        probs[i] = (float)p;
        sum += p;
    }
    if (sum <= 0) {
        return idx[0];
    }
    for (int i = 0; i < top_k; i++) {
        probs[i] = (float)(probs[i] / sum);
    }

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
        if (r < cum) {
            return idx[i];
        }
    }
    return idx[top_k - 1];
}

// Forward-declared: defined in the prefill section below.
float* lookup_rows(qwen3_tts_context* c, ggml_tensor* weight, const int32_t* ids, int n_ids);

// One step of the code-predictor AR loop. Builds + runs the graph
// against a caller-supplied (T, d) embedding tensor and the lm_head
// for the current generation step. Returns logits (cp_vocab,).
float* run_code_pred_kv(qwen3_tts_context* c, const float* embeds, int n_tokens, int n_past, ggml_tensor* lm_head,
                        bool skip_plan = false) {
    if (!lm_head) {
        return nullptr;
    }
    if (n_past + n_tokens > c->cp_kv_max_ctx) {
        fprintf(stderr, "qwen3_tts: cp_kv overflow (%d+%d > %d)\n", n_past, n_tokens, c->cp_kv_max_ctx);
        return nullptr;
    }
    const auto& hp = c->hp;
    const int d = (int)hp.cp_d_model;
    const int vocab = (int)hp.cp_vocab_size;
    const int actual_Lk = n_past + n_tokens;
    const int fixed_Lk = c->cp_kv_max_ctx;  // matches build_graph_code_pred_kv

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        positions[i] = n_past + i;
    }

    // Build causal mask for every call (T=1 or T=2).  Size is (fixed_Lk, T).
    // Positions [0 .. n_past+q] are 0; positions [n_past+q+1 .. fixed_Lk-1] are -inf.
    const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> mask((size_t)fixed_Lk * n_tokens, neginf_h);
    for (int q = 0; q < n_tokens; q++) {
        for (int k = 0; k <= n_past + q; k++) {
            mask[(size_t)q * fixed_Lk + k] = zero_h;
        }
    }
    (void)actual_Lk;

    // For T=1 steps: blit current lm_head[i] into the slot so the persistent
    // graph always reads from cp_lm_head_slot regardless of which step i we're on.
    const bool use_slot = (n_tokens == 1 && c->cp_lm_head_slot && !c->cp_cpu_pinned);
    if (use_slot) {
        ggml_backend_tensor_copy(lm_head, c->cp_lm_head_slot);
    }

    const bool bench = env_bool("QWEN3_TTS_BENCH");
    const bool prof = env_bool("QWEN3_TTS_PROF");
    const double t_build0 = bench ? now_ms() : 0.0;

    // skip_plan=true: reuse the cached T=1 graph — no rebuild, no reset, no alloc.
    // The graph is valid because compute_meta wasn't touched since it was built.
    const bool can_skip = skip_plan && use_slot && (c->cp_t1_gf != nullptr);
    ggml_cgraph* gf;
    if (can_skip) {
        gf = c->cp_t1_gf;
    } else {
        gf = build_graph_code_pred_kv(c, n_past, n_tokens, lm_head);
        if (!gf) {
            return nullptr;
        }
        if (use_slot) {
            c->cp_t1_gf = gf;  // cache for future skip_plan calls within this frame
        }
    }

    const double t_build1 = bench ? now_ms() : 0.0;
    ggml_backend_sched_t sched = code_pred_pick_sched(c);
    if (!code_pred_reserve_sched(c, sched)) {
        return nullptr;
    }
    const double t_reset0 = bench ? now_ms() : 0.0;
    if (!can_skip) {
        ggml_backend_sched_reset(sched);
    }
    const double t_reset1 = bench ? now_ms() : 0.0;
    if (!can_skip) {
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            return nullptr;
        }
    }
    const double t_alloc1 = bench ? now_ms() : 0.0;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                            mask.size() * sizeof(ggml_fp16_t));
    qwen3_prof_state prof_state;
    if (prof) {
        ggml_backend_sched_set_eval_callback(sched, qwen3_prof_eval_cb, &prof_state);
    }
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        if (prof) {
            ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
        }
        fprintf(stderr, "qwen3_tts: code_pred compute failed\n");
        return nullptr;
    }
    if (prof) {
        ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
    }
    const double t_compute1 = bench ? now_ms() : 0.0;
    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    float* r = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)vocab * sizeof(float));
    if (bench) {
        static double sum_build = 0.0, sum_reset = 0.0, sum_alloc = 0.0, sum_compute = 0.0, sum_read = 0.0;
        static int count = 0;
        sum_build += t_build1 - t_build0;
        sum_reset += t_reset1 - t_reset0;
        sum_alloc += t_alloc1 - t_reset1;   // alloc_graph only (excludes reset)
        sum_compute += t_compute1 - t_alloc1;
        sum_read += now_ms() - t_compute1;
        count++;
        if (count == 15) {
            fprintf(stderr,
                    "qwen3_tts: code_pred_kv bench (%d calls): build=%.2f ms  reset=%.2f ms  alloc=%.2f ms  "
                    "compute=%.1f ms  read=%.2f ms\n",
                    count, sum_build, sum_reset, sum_alloc, sum_compute, sum_read);
            sum_build = sum_reset = sum_alloc = sum_compute = sum_read = 0.0;
            count = 0;
        }
    }
    if (prof) {
        static qwen3_prof_state sum_prof;
        static int count = 0;
        qwen3_prof_add(sum_prof, prof_state);
        count++;
        if (count == 15) {
            qwen3_prof_print("code_pred_kv", sum_prof, count);
            sum_prof = {};
            count = 0;
        }
    }
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
                           int32_t* out_codes15, uint64_t* rng_state, int frame_idx = -1) {
    auto& cp = c->code_pred;
    const auto& hp = c->hp;
    const int d = (int)hp.cp_d_model;
    const int n_groups = (int)hp.cp_n_code_groups; // 16
    const int top_k = 50;
    const float temperature = 0.9f;
    const char* dump_dir = env_str("QWEN3_TTS_DUMP_DIR");

    // ---- step 0: inputs_embeds = (past_hidden, last_id_hidden), n_past=0 ----
    std::vector<float> step0((size_t)2 * d);
    std::memcpy(step0.data(), past_hidden_d, (size_t)d * sizeof(float));
    std::memcpy(step0.data() + d, last_id_hidden_d, (size_t)d * sizeof(float));
    if (getenv("QWEN3_TTS_EMBD_CHECK")) {
        float ph_l1 = 0.0f, lih_l1 = 0.0f;
        for (int j = 0; j < d; j++)
            ph_l1 += std::abs(past_hidden_d[j]);
        for (int j = 0; j < d; j++)
            lih_l1 += std::abs(last_id_hidden_d[j]);
        fprintf(stderr, "qwen3_tts: cp_step0 past_hidden_l1=%.4f last_id_hidden_l1=%.4f\n", ph_l1, lih_l1);
        fprintf(stderr, "qwen3_tts: cp_step0 ph[0..3]=%.8f %.8f %.8f %.8f\n", past_hidden_d[0], past_hidden_d[1],
                past_hidden_d[2], past_hidden_d[3]);
        fprintf(stderr, "qwen3_tts: cp_step0 lih[0..3]=%.8f %.8f %.8f %.8f\n", last_id_hidden_d[0], last_id_hidden_d[1],
                last_id_hidden_d[2], last_id_hidden_d[3]);
    }

    if (!cp.lm_head[0]) {
        fprintf(stderr, "qwen3_tts: code_pred.lm_head[0] missing\n");
        return false;
    }
    float* logits0 = run_code_pred_kv(c, step0.data(), 2, /*n_past=*/0, cp.lm_head[0]);
    if (!logits0) {
        return false;
    }
    if (dump_dir && frame_idx >= 0) {
        char name[64];
        snprintf(name, sizeof(name), "cp_f%03d_step00_embed", frame_idx);
        dump_f32(dump_dir, name, step0.data(), step0.size());
        snprintf(name, sizeof(name), "cp_f%03d_step00_logits", frame_idx);
        dump_f32(dump_dir, name, logits0, hp.cp_vocab_size);
    }
    if (getenv("QWEN3_TTS_EMBD_CHECK")) {
        float l1 = 0.0f;
        int top_idx = 0;
        float top_val = logits0[0];
        for (int j = 0; j < (int)hp.cp_vocab_size; j++) {
            l1 += std::abs(logits0[j]);
            if (logits0[j] > top_val) {
                top_val = logits0[j];
                top_idx = j;
            }
        }
        fprintf(stderr, "qwen3_tts: cp logits0 l1=%.4f top=%d(%.4f) rng=%llu\n", l1, top_idx, top_val,
                (unsigned long long)*rng_state);
    }
    out_codes15[0] = top_k_sample(logits0, (int)hp.cp_vocab_size, top_k, temperature, rng_state);
    if (dump_dir && frame_idx >= 0) {
        char name[64];
        snprintf(name, sizeof(name), "cp_f%03d_step00_id", frame_idx);
        dump_i32(dump_dir, name, &out_codes15[0], 1);
    }
    free(logits0);

    int n_past = 2;

    // ---- steps 1..14: input = codec_embedding[i-1](codes[i-1]), apply lm_head[i] ----
    std::vector<float> emb_buf(d);
    for (int i = 1; i < n_groups - 1; i++) {
        if (!cp.lm_head[i]) {
            fprintf(stderr, "qwen3_tts: code_pred missing lm_head[%d]\n", i);
            return false;
        }
        int32_t prev = out_codes15[i - 1];
        bool ok = false;
        const bool use_cache = !env_bool("QWEN3_TTS_NO_EMBD_CACHE");
        if (use_cache && i - 1 < (int)c->codec_embd_cache.size() && c->codec_embd_cache[i - 1]) {
            ok = c->codec_embd_cache[i - 1].get_row_into(prev, emb_buf.data());
        } else {
            if (!cp.codec_embd[i - 1]) {
                fprintf(stderr, "qwen3_tts: code_pred missing codec_embd[%d]\n", i - 1);
                return false;
            }
            float* emb = lookup_rows(c, cp.codec_embd[i - 1], &prev, 1);
            if (!emb) {
                return false;
            }
            std::memcpy(emb_buf.data(), emb, (size_t)d * sizeof(float));
            free(emb);
            ok = true;
        }
        if (!ok) {
            return false;
        }
        if (dump_dir && frame_idx >= 0) {
            char name[64];
            snprintf(name, sizeof(name), "cp_f%03d_step%02d_embed", frame_idx, i);
            dump_f32(dump_dir, name, emb_buf.data(), d);
        }
        float* logits = run_code_pred_kv(c, emb_buf.data(), 1, n_past, cp.lm_head[i], /*skip_plan=*/i >= 2);
        if (!logits) {
            return false;
        }
        if (dump_dir && frame_idx >= 0) {
            char name[64];
            snprintf(name, sizeof(name), "cp_f%03d_step%02d_logits", frame_idx, i);
            dump_f32(dump_dir, name, logits, hp.cp_vocab_size);
        }
        out_codes15[i] = top_k_sample(logits, (int)hp.cp_vocab_size, top_k, temperature, rng_state);
        if (dump_dir && frame_idx >= 0) {
            char name[64];
            snprintf(name, sizeof(name), "cp_f%03d_step%02d_id", frame_idx, i);
            dump_i32(dump_dir, name, &out_codes15[i], 1);
        }
        free(logits);
        n_past += 1;
    }
    return true;
}

// Allocate the code_predictor KV cache: (head_dim, max_ctx, n_kv, cp_n_layers).
// max_ctx is small — at most 2 + 14 = 16 positions per frame.
bool cp_kv_alloc(qwen3_tts_context* c) {
    if (c->cp_kv_k) {
        return true;
    }
    const auto& hp = c->hp;
    const int hd = (int)hp.head_dim;
    const int n_kv = (int)hp.cp_n_kv_heads;
    const int n_lay = (int)hp.cp_n_layers;
    const int max_ctx = 16; // 2 prefill + 14 code steps; exactly fills one frame

    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    c->cp_kv_ctx = ggml_init(kp);
    c->cp_kv_k = ggml_new_tensor_4d(c->cp_kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    c->cp_kv_v = ggml_new_tensor_4d(c->cp_kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(c->cp_kv_k, "cp_kv_k");
    ggml_set_name(c->cp_kv_v, "cp_kv_v");
    const size_t kb = ggml_nbytes(c->cp_kv_k), vb = ggml_nbytes(c->cp_kv_v);
    ggml_backend_t cp_backend = c->cp_cpu_pinned ? c->backend_cpu : c->backend;
    c->cp_kv_buf = ggml_backend_alloc_buffer(cp_backend, kb + vb);
    if (!c->cp_kv_buf) {
        return false;
    }
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
    if (it != v.token_to_id.end()) {
        ids.push_back(it->second);
    }
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
    if (!weight) {
        return nullptr;
    }
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
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ids"), ids, 0, (size_t)n_ids * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        return nullptr;
    }
    ggml_tensor* outT = ggml_graph_get_tensor(gf, "rows");
    float* r = (float*)malloc((size_t)n_ids * d * sizeof(float));
    ggml_backend_tensor_get(outT, r, 0, (size_t)n_ids * d * sizeof(float));
    return r;
}

// Like lookup_rows but uses a caller-supplied scheduler.
static float* lookup_rows_sched(qwen3_tts_context* c, ggml_backend_sched_t sched, ggml_tensor* weight,
                                const int32_t* ids, int n_ids) {
    if (!weight) {
        return nullptr;
    }
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
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ids"), ids, 0, (size_t)n_ids * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        return nullptr;
    }
    ggml_tensor* outT = ggml_graph_get_tensor(gf, "rows");
    float* r = (float*)malloc((size_t)n_ids * d * sizeof(float));
    ggml_backend_tensor_get(outT, r, 0, (size_t)n_ids * d * sizeof(float));
    return r;
}

// Build and run a single graph that looks up one row from each of the 16
// codec embedding tables (talker.token_embd_w for cb=0, code_pred.codec_embd
// for cb=1..15) and returns their element-wise sum.  Replaces 16 separate
// lookup_rows round-trips with one Metal command buffer.
// codes16[i] = codec code for codebook i (i in 0..15).
// Returns malloc'd float[d], or nullptr on failure.  Also writes the cb=0
// embedding (talker.token_embd_w[codes16[0]]) into *out_cb0_emb if non-null
// (used as last_id_hidden for code_pred step 0).
static float* run_codec_embed_sum_ar(qwen3_tts_context* c, const int32_t* codes16, float* out_cb0_emb) {
    const int d = (int)c->hp.d_model;
    const int n_groups = (int)c->hp.n_code_groups; // 16

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    // 16 ids tensors + 16 get_rows + 15 adds + 16 input marks = ~64 nodes
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 128, false);

    // One I32 input per codebook (simpler than view slices for Metal compat)
    ggml_tensor* sum = nullptr;
    for (int cb = 0; cb < n_groups; cb++) {
        ggml_tensor* w = (cb == 0) ? c->talker.token_embd_w : c->code_pred.codec_embd[cb - 1];
        if (!w) {
            ggml_free(ctx0);
            return nullptr;
        }

        char nm[32];
        snprintf(nm, sizeof(nm), "id%d", cb);
        ggml_tensor* id_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_name(id_t, nm);
        ggml_set_input(id_t);

        ggml_tensor* row = ggml_get_rows(ctx0, w, id_t); // [d, 1]
        // reshape to [d] so add works cleanly
        row = ggml_reshape_1d(ctx0, row, d);

        if (cb == 0) {
            // Also expose the cb=0 row separately for last_id_hidden
            if (out_cb0_emb) {
                ggml_set_name(row, "cb0_row");
                ggml_build_forward_expand(gf, row);
            }
            sum = row;
        } else {
            sum = ggml_add(ctx0, sum, row);
        }
    }
    ggml_set_name(sum, "codec_sum");
    ggml_build_forward_expand(gf, sum);
    ggml_free(ctx0);

    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        return nullptr;
    }

    // Set the 16 index inputs
    for (int cb = 0; cb < n_groups; cb++) {
        char nm[32];
        snprintf(nm, sizeof(nm), "id%d", cb);
        ggml_tensor* id_t = ggml_graph_get_tensor(gf, nm);
        if (!id_t) {
            return nullptr;
        }
        ggml_backend_tensor_set(id_t, &codes16[cb], 0, sizeof(int32_t));
    }

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        return nullptr;
    }

    float* result = (float*)malloc((size_t)d * sizeof(float));
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "codec_sum"), result, 0, (size_t)d * sizeof(float));

    if (out_cb0_emb) {
        ggml_tensor* cb0_t = ggml_graph_get_tensor(gf, "cb0_row");
        if (cb0_t) {
            ggml_backend_tensor_get(cb0_t, out_cb0_emb, 0, (size_t)d * sizeof(float));
        }
    }

    return result;
}

static bool materialize_tensor_copy(ggml_tensor* dst, ggml_tensor* src, std::vector<uint8_t>& src_buf,
                                    std::vector<uint8_t>& dst_buf, std::vector<float>& row_f32) {
    if (!dst || !src) {
        return false;
    }

    const size_t src_bytes = ggml_nbytes(src);
    if (src_buf.size() < src_bytes) {
        src_buf.resize(src_bytes);
    }
    ggml_backend_tensor_get(src, src_buf.data(), 0, src_bytes);

    if (dst->type == src->type) {
        ggml_backend_tensor_set(dst, src_buf.data(), 0, src_bytes);
        return true;
    }

    const int64_t row_len = src->ne[0];
    if (row_len <= 0) {
        return false;
    }
    const int64_t n_rows = ggml_nelements(src) / row_len;
    const size_t src_row_bytes = ggml_row_size(src->type, row_len);
    const size_t dst_row_bytes = ggml_row_size(dst->type, row_len);
    if ((size_t)n_rows * src_row_bytes != src_bytes) {
        return false;
    }

    const ggml_to_float_t to_float = ggml_get_type_traits(src->type)->to_float;
    ggml_from_float_t from_float = nullptr;
    const auto* dst_cpu_traits = ggml_get_type_traits_cpu(dst->type);
    if (dst_cpu_traits) {
        from_float = dst_cpu_traits->from_float;
    }
    if (!from_float) {
        from_float = ggml_get_type_traits(dst->type)->from_float_ref;
    }
    if ((src->type != GGML_TYPE_F32 && !to_float) || (dst->type != GGML_TYPE_F32 && !from_float)) {
        return false;
    }

    if (row_f32.size() < (size_t)row_len) {
        row_f32.resize((size_t)row_len);
    }
    const size_t dst_bytes = ggml_nbytes(dst);
    if (dst_buf.size() < dst_bytes) {
        dst_buf.resize(dst_bytes);
    }

    for (int64_t row = 0; row < n_rows; row++) {
        const uint8_t* src_row = src_buf.data() + (size_t)row * src_row_bytes;
        uint8_t* dst_row = dst_buf.data() + (size_t)row * dst_row_bytes;
        if (src->type == GGML_TYPE_F32) {
            std::memcpy(row_f32.data(), src_row, (size_t)row_len * sizeof(float));
        } else {
            to_float(src_row, row_f32.data(), row_len);
        }
        if (dst->type == GGML_TYPE_F32) {
            std::memcpy(dst_row, row_f32.data(), (size_t)row_len * sizeof(float));
        } else {
            from_float(row_f32.data(), dst_row, row_len);
        }
    }

    ggml_backend_tensor_set(dst, dst_buf.data(), 0, dst_bytes);
    return true;
}

static enum ggml_type code_pred_cpu_copy_type_from_env(const char* cp_be) {
    if (!cp_be || std::strncmp(cp_be, "cpu", 3) != 0) {
        return GGML_TYPE_COUNT;
    }
    if (std::strcmp(cp_be, "cpu-f32") == 0) {
        return GGML_TYPE_F32;
    }
    if (std::strcmp(cp_be, "cpu-f16") == 0) {
        return GGML_TYPE_F16;
    }
    return GGML_TYPE_COUNT; // "cpu" = keep original tensor types
}

// Copy code_pred transformer weights (lm_head + blocks + output_norm) and a
// copy of talker.token_embd_w to the CPU backend so that code_pred forward
// passes run on c->cp_sched (CPU-only) instead of Metal.  codec_embd tables
// are left on Metal because sum_codec_embeds() uses them on c->sched.
static bool copy_cp_weights_to_cpu(qwen3_tts_context* c, enum ggml_type dst_type) {
    if (!c->backend_cpu || !c->cp_sched) {
        return false;
    }

    auto& cp = c->code_pred;

    // Build the list of (src, dst**) pairs to copy.
    // We update the pointer in the struct so the graph builder picks up
    // the CPU tensor automatically.
    std::vector<ggml_tensor**> ptrs;
    ptrs.push_back(&cp.output_norm_w);
    for (auto& t : cp.lm_head) {
        ptrs.push_back(&t);
    }
    for (auto& b : cp.blocks) {
        ptrs.push_back(&b.attn_norm_w);
        ptrs.push_back(&b.attn_q_w);
        ptrs.push_back(&b.attn_k_w);
        ptrs.push_back(&b.attn_v_w);
        ptrs.push_back(&b.attn_output_w);
        ptrs.push_back(&b.attn_q_norm_w);
        ptrs.push_back(&b.attn_k_norm_w);
        ptrs.push_back(&b.ffn_norm_w);
        ptrs.push_back(&b.ffn_gate_w);
        ptrs.push_back(&b.ffn_up_w);
        ptrs.push_back(&b.ffn_down_w);
    }

    auto copy_type_for = [&](ggml_tensor* orig) -> enum ggml_type {
        return dst_type == GGML_TYPE_COUNT ? orig->type : dst_type;
    };
    auto tensor_nbytes_for_type = [&](ggml_tensor* orig, enum ggml_type type) -> size_t {
        const int64_t row_len = orig->ne[0];
        const int64_t n_rows = ggml_nelements(orig) / row_len;
        return ggml_row_size(type, row_len) * (size_t)n_rows;
    };

    // Calculate total bytes for the destination tensors, not the source
    // tensors. For cpu-f16 / cpu-f32 the CPU copies are larger than the
    // original q8/qk weights.
    size_t total_bytes = 0;
    size_t n_tensors = 0;
    for (auto* pp : ptrs) {
        if (*pp) {
            ggml_tensor* orig = *pp;
            total_bytes += tensor_nbytes_for_type(orig, copy_type_for(orig));
            n_tensors++;
        }
    }
    // talker.token_embd_w copy (kept separately, doesn't overwrite original)
    const size_t embd_bytes = c->talker.token_embd_w
                                  ? tensor_nbytes_for_type(c->talker.token_embd_w,
                                                           copy_type_for(c->talker.token_embd_w))
                                  : 0;
    total_bytes += embd_bytes;
    n_tensors += (embd_bytes > 0) ? 1 : 0;

    c->cp_cpu_buf = ggml_backend_alloc_buffer(c->backend_cpu, total_bytes + 512);
    if (!c->cp_cpu_buf) {
        fprintf(stderr, "qwen3_tts: copy_cp_to_cpu: alloc failed (%zu MB)\n", total_bytes >> 20);
        return false;
    }
    c->cp_cpu_ctx = ggml_init({n_tensors * ggml_tensor_overhead() + 1024, nullptr, true});
    if (!c->cp_cpu_ctx) {
        ggml_backend_buffer_free(c->cp_cpu_buf);
        c->cp_cpu_buf = nullptr;
        return false;
    }

    ggml_tallocr talloc = ggml_tallocr_new(c->cp_cpu_buf);
    std::vector<uint8_t> tmp_src;
    std::vector<uint8_t> tmp_dst;
    std::vector<float> row_f32;

    // Copy transformer weights and update pointers
    for (auto* pp : ptrs) {
        if (!*pp) {
            continue;
        }
        ggml_tensor* orig = *pp;
        const enum ggml_type cpu_type = copy_type_for(orig);
        ggml_tensor* cpu_t = ggml_new_tensor(c->cp_cpu_ctx, cpu_type, GGML_MAX_DIMS, orig->ne);
        if (ggml_tallocr_alloc(&talloc, cpu_t) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "qwen3_tts: copy_cp_to_cpu: alloc failed for tensor %s\n",
                    orig->name[0] ? orig->name : "(unnamed)");
            return false;
        }
        if (!materialize_tensor_copy(cpu_t, orig, tmp_src, tmp_dst, row_f32)) {
            fprintf(stderr, "qwen3_tts: copy_cp_to_cpu: copy failed for tensor %s (%s -> %s)\n",
                    orig->name[0] ? orig->name : "(unnamed)", ggml_type_name(orig->type), ggml_type_name(cpu_type));
            return false;
        }
        *pp = cpu_t;
    }
    // Copy talker.token_embd_w without overwriting the original (Metal) pointer
    if (c->talker.token_embd_w && embd_bytes > 0) {
        ggml_tensor* orig = c->talker.token_embd_w;
        const enum ggml_type cpu_type = copy_type_for(orig);
        ggml_tensor* cpu_t = ggml_new_tensor(c->cp_cpu_ctx, cpu_type, GGML_MAX_DIMS, orig->ne);
        if (ggml_tallocr_alloc(&talloc, cpu_t) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "qwen3_tts: copy_cp_to_cpu: alloc failed for talker.token_embd_w\n");
            return false;
        }
        if (!materialize_tensor_copy(cpu_t, orig, tmp_src, tmp_dst, row_f32)) {
            fprintf(stderr, "qwen3_tts: copy_cp_to_cpu: copy failed for talker.token_embd_w (%s -> %s)\n",
                    ggml_type_name(orig->type), ggml_type_name(cpu_type));
            return false;
        }
        c->talker_embd_cpu = cpu_t;
    }

    c->cp_cpu_pinned = true;
    if (c->params.verbosity >= 1) {
        const char* type_name = (dst_type == GGML_TYPE_F16) ? "f16" : (dst_type == GGML_TYPE_F32) ? "f32" : "native";
        fprintf(stderr, "qwen3_tts: code_pred CPU-pinned (%zu tensors, %zu MB, copy=%s)\n", n_tensors,
                total_bytes >> 20, type_name);
    }
    return true;
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
        for (int t = 0; t < T_codec; t++) {
            col[t] = ref_code_TC[(size_t)t * n_groups + cb];
        }
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
        for (size_t i = 0; i < (size_t)T_codec * d; i++) {
            acc[i] += rows[i];
        }
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

    // Two voice sources: runtime (from set_voice_prompt) or baked voice pack.
    std::vector<float> spk_buf(d);
    std::vector<int32_t> ref_code_TC;
    int T_codec = 0;

    if (!c->runtime_ref_codes.empty() && (int)c->runtime_spk_emb.size() == d) {
        // Runtime path: set_voice_prompt computed both embedding and codes.
        spk_buf = c->runtime_spk_emb;
        const int n_total = (int)c->runtime_ref_codes.size();
        T_codec = n_total / n_groups;
        ref_code_TC = c->runtime_ref_codes; // already [T, n_q] row-major
    } else {
        // Voice pack path (original behavior).
        if (c->vp_active < 0) {
            fprintf(stderr, "qwen3_tts: no voice — call qwen3_tts_load_voice_pack or qwen3_tts_set_voice_prompt\n");
            return false;
        }
        const std::string& voice_name = c->vp_names[c->vp_active];
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
        T_codec = (int)code_t->ne[1];
        if ((int)code_t->ne[0] != n_groups) {
            fprintf(stderr, "qwen3_tts: ref_code groups mismatch: %d vs %d\n", (int)code_t->ne[0], n_groups);
            return false;
        }
        if ((int)c->runtime_spk_emb.size() == d) {
            spk_buf = c->runtime_spk_emb; // override spk from set_voice_prompt
        } else {
            ggml_backend_tensor_get(spk_t, spk_buf.data(), 0, (size_t)d * sizeof(float));
        }
        ref_code_TC.resize((size_t)T_codec * n_groups);
        ggml_backend_tensor_get(code_t, ref_code_TC.data(), 0, ref_code_TC.size() * sizeof(int32_t));
    }

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
    if (!tts_special_emb) {
        return false;
    }
    const float* tts_bos = tts_special_emb;                 // (d,)
    const float* tts_eos = tts_special_emb + d;             // (d,)
    const float* tts_pad = tts_special_emb + (size_t)2 * d; // (d,)

    // ---- codec_input_embedding: sentinels + spk + pad/bos ----
    std::vector<int32_t> codec_prefill;
    if (c->language_id <= 0) {
        codec_prefill = {(int32_t)hp.codec_nothink_id, (int32_t)hp.codec_think_bos_id, (int32_t)hp.codec_think_eos_id};
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
        for (int j = 0; j < d; j++) {
            bridge[(size_t)i * d + j] = left[j] + right[j];
        }
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
        for (int i = text_lens; i < codec_lens; i++) {
            std::memcpy(padded.data() + (size_t)i * d, tts_pad, (size_t)d * sizeof(float));
        }
        for (size_t i = 0; i < padded.size(); i++) {
            icl_input[i] = padded[i] + codec_embed[i];
        }
        // Trailing for codec >= text: just tts_pad_embed (1 token).
        trailing_text_hidden.assign(tts_pad, tts_pad + d);
        M_trailing = 1;
    } else {
        // text_lens > codec_lens: take text[:codec_lens] + codec_embed.
        // Trailing = text[codec_lens:].
        for (int i = 0; i < codec_lens; i++) {
            for (int j = 0; j < d; j++) {
                icl_input[(size_t)i * d + j] = text_embed_padded[(size_t)i * d + j] + codec_embed[(size_t)i * d + j];
            }
        }
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
    if (c->params.verbosity >= 1) {
        fprintf(stderr,
                "qwen3_tts: ICL prefill: role=3 + bridge=%d + icl=%d (text_lens=%d codec_lens=%d) = T=%d  "
                "trailing=%d\n",
                L_bridge, icl_len, text_lens, codec_lens, T_prefill, M_trailing);
    }
    free(tts_special_emb);
    free(role_emb);
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
static ggml_tensor* codec_causal_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride,
                                        int dilation) {
    const int K = (int)w->ne[0];
    int pad_left = (K - 1) * dilation;
    if (stride > 1) {
        pad_left -= (stride - 1);
    }
    if (pad_left < 0) {
        pad_left = 0;
    }
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [T, C]
    if (pad_left > 0) {
        x = ggml_pad_ext(ctx, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
        // ggml_pad_ext always emits a 4D tensor; reshape back to 2D so
        // ggml_conv_1d's internal im2col step sees a standard 2D input.
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
    }
    x = ggml_conv_1d(ctx, w, x, stride, 0, dilation);
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C_out, T_out]
    if (b) {
        x = ggml_add(ctx, x, b);
    }
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
    if (ggml_n_dims(x) > 2) {
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
    }
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C, T]
    if (b) {
        x = ggml_add(ctx, x, b);
    }
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
    if (b) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}

// ---------------------------------------------------------------------------
// SnakeBeta activation: x + exp(-beta) * sin²(x * exp(alpha))
// Input/output: [C, T]. alpha, beta: [C].
// ---------------------------------------------------------------------------
static ggml_tensor* codec_snake_beta(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha, ggml_tensor* beta) {
    ggml_tensor* ea = ggml_exp(ctx, alpha);          // [C]
    ggml_tensor* neg_eb = ggml_neg(ctx, beta);       // [C]
    ggml_tensor* inv_eb = ggml_exp(ctx, neg_eb);     // [C] = 1/exp(beta)
    ggml_tensor* xa = ggml_mul(ctx, x, ea);          // [C, T]
    ggml_tensor* s = ggml_sin(ctx, xa);              // [C, T]
    ggml_tensor* s2 = ggml_mul(ctx, s, s);           // [C, T]
    ggml_tensor* scaled = ggml_mul(ctx, s2, inv_eb); // [C, T]
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
    for (int u = 0; u < 3; u++) {
        x = codec_res_unit(ctx, x, blk.res[u], dilations[u]);
    }
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
    ggml_tensor* emb_first = ggml_get_rows(ctx0, codec.rvq_first_cb, cb0_ids);              // [256, T]
    emb_first = codec_causal_conv1d(ctx0, emb_first, codec.rvq_first_out_w, nullptr, 1, 1); // [512, T]

    // rvq_rest: sum 15 codebook lookups → [256, T], apply output_proj → [512, T]
    ggml_tensor* emb_rest =
        ggml_get_rows(ctx0, codec.rvq_rest_cb[0], ggml_view_1d(ctx0, codes_inp, T, (size_t)T * sizeof(int32_t)));
    for (int q = 1; q < 15; q++) {
        ggml_tensor* ids_q = ggml_view_1d(ctx0, codes_inp, T, (size_t)(q + 1) * T * sizeof(int32_t));
        emb_rest = ggml_add(ctx0, emb_rest, ggml_get_rows(ctx0, codec.rvq_rest_cb[q], ids_q));
    }
    emb_rest = codec_causal_conv1d(ctx0, emb_rest, codec.rvq_rest_out_w, nullptr, 1, 1); // [512, T]

    ggml_tensor* h = ggml_add(ctx0, emb_first, emb_rest); // [512, T]
    ggml_set_name(h, "codec_rvq_out");
    ggml_set_output(h); // prevent gallocr from reusing this buffer

    // ── Step 2: pre_conv ────────────────────────────────────────────────────
    h = codec_causal_conv1d(ctx0, h, codec.pre_conv_w, codec.pre_conv_b, 1, 1); // [1024, T]
    ggml_set_name(h, "codec_pre_conv_out");
    ggml_set_output(h);

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
        n_heads, n_heads, hd, /*n_kv_grp*/ 1, (int)hp.max_pos, theta, attn_scale,
    };

    for (int il = 0; il < n_layers; il++) {
        const auto& bl = codec.xfmr_layers[il];
        ggml_tensor* residual = h;

        // Pre-attention RMSNorm
        ggml_tensor* x = ggml_rms_norm(ctx0, h, eps);
        x = ggml_mul(ctx0, x, bl.attn_norm_w);

        // Self-attention (full-sequence, no KV cache)
        ggml_tensor* attn = core_attn::llama_self_attn(ctx0, x, bl.attn_q_w, bl.attn_k_w, bl.attn_v_w, bl.attn_o_w,
                                                       positions, causal_mask, asp);
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
    ggml_set_name(h, "codec_xfmr_out");
    ggml_set_output(h);

    // ── Step 4: ConvNeXt upsample (2 stages, each 2×) ──────────────────────
    for (int s = 0; s < 2; s++) {
        h = codec_transposed_conv1d(ctx0, h, codec.up[s].tconv_w, codec.up[s].tconv_b, 2);
        h = codec_convnext_block(ctx0, h, codec.up[s]);
        char uname[32];
        snprintf(uname, sizeof(uname), "codec_up%d_out", s);
        ggml_set_name(h, uname);
        ggml_set_output(h);
    }

    // ── Step 5: Decoder blocks ──────────────────────────────────────────────
    h = codec_causal_conv1d(ctx0, h, codec.in_conv_w, codec.in_conv_b, 1, 1); // [1536, 4T]
    ggml_set_name(h, "codec_in_conv_out");
    ggml_set_output(h);
    for (int b = 0; b < 4; b++) {
        h = codec_dec_block(ctx0, h, codec.blocks[b], hp.upsample_rates[b]);
        if (b == 0) {
            ggml_set_name(h, "codec_blk0_out");
            ggml_set_output(h);
        }
    }

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

// Forward declaration — defined after the SEANet implementation section.
static bool load_cenc(qwen3_tts_context* c);

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
    hp.n_layers = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.n_layers", hp.n_layers);
    hp.d_model = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.d_model", hp.d_model);
    hp.n_heads = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.n_heads", hp.n_heads);
    hp.head_dim = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.head_dim", hp.head_dim);
    hp.ff_dim = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.ff_dim", hp.ff_dim);
    hp.n_q = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.n_quantizers", hp.n_q);
    hp.codebook_size = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.codebook_size", hp.codebook_size);
    hp.latent_dim = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.latent_dim", hp.latent_dim);
    hp.decoder_dim = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.decoder_dim", hp.decoder_dim);
    hp.sliding_window = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.sliding_window", hp.sliding_window);
    hp.max_pos = core_gguf::kv_u32(meta, "qwen3tts_codec.dec.max_pos", hp.max_pos);
    hp.rope_theta = core_gguf::kv_f32(meta, "qwen3tts_codec.dec.rope_theta", hp.rope_theta);
    hp.rms_norm_eps = core_gguf::kv_f32(meta, "qwen3tts_codec.dec.rms_norm_eps", hp.rms_norm_eps);
    core_gguf::free_metadata(meta);

    // Pass 2: weights — pinned to CPU backend (see codec_sched comment in
    // qwen3_tts_context). Override with QWEN3_TTS_CODEC_FORCE_METAL=1 to
    // reproduce the M1 Metal crash for instrumentation.
    const bool force_metal = std::getenv("QWEN3_TTS_CODEC_FORCE_METAL") != nullptr;
    ggml_backend_t weight_backend = force_metal ? c->backend : c->backend_cpu;
    if (force_metal && c->params.verbosity >= 0) {
        fprintf(stderr, "qwen3_tts: codec: QWEN3_TTS_CODEC_FORCE_METAL=1 — loading weights onto %s\n",
                ggml_backend_name(weight_backend));
    }
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, weight_backend, "codec", wl)) {
        fprintf(stderr, "qwen3_tts: codec: failed to load weights from '%s'\n", path);
        return false;
    }
    codec.ctx_w = wl.ctx;
    codec.buf_w = wl.buf;
    codec.tensors = std::move(wl.tensors);

    auto req = [&](const char* n) -> ggml_tensor* { return core_gguf::require(codec.tensors, n, "codec"); };
    auto fmt = [](const char* f, auto... a) -> std::string {
        char buf[256];
        snprintf(buf, sizeof(buf), f, a...);
        return buf;
    };

    // RVQ
    codec.rvq_first_cb = req("codec.dec.rvq_first.codebook");
    codec.rvq_first_out_w = req("codec.dec.rvq_first.out_proj_w");
    for (int q = 0; q < 15; q++) {
        codec.rvq_rest_cb[q] = req(fmt("codec.dec.rvq_rest.%d.codebook", q).c_str());
    }
    codec.rvq_rest_out_w = req("codec.dec.rvq_rest.out_proj_w");

    // pre_conv
    codec.pre_conv_w = req("codec.dec.pre_conv_w");
    codec.pre_conv_b = req("codec.dec.pre_conv_b");

    // Transformer
    codec.xfmr_in_proj_w = req("codec.dec.xfmr.in_proj_w");
    codec.xfmr_in_proj_b = req("codec.dec.xfmr.in_proj_b");
    codec.xfmr_norm_w = req("codec.dec.xfmr.norm_w");
    codec.xfmr_out_proj_w = req("codec.dec.xfmr.out_proj_w");
    codec.xfmr_out_proj_b = req("codec.dec.xfmr.out_proj_b");

    codec.xfmr_layers.resize(hp.n_layers);
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        auto& bl = codec.xfmr_layers[il];
        bl.attn_norm_w = req(fmt("codec.dec.xfmr.blk.%u.attn_norm_w", il).c_str());
        bl.ffn_norm_w = req(fmt("codec.dec.xfmr.blk.%u.ffn_norm_w", il).c_str());
        bl.attn_q_w = req(fmt("codec.dec.xfmr.blk.%u.attn_q_w", il).c_str());
        bl.attn_k_w = req(fmt("codec.dec.xfmr.blk.%u.attn_k_w", il).c_str());
        bl.attn_v_w = req(fmt("codec.dec.xfmr.blk.%u.attn_v_w", il).c_str());
        bl.attn_o_w = req(fmt("codec.dec.xfmr.blk.%u.attn_o_w", il).c_str());
        bl.attn_ls_w = req(fmt("codec.dec.xfmr.blk.%u.attn_ls_w", il).c_str());
        bl.ffn_gate_w = req(fmt("codec.dec.xfmr.blk.%u.ffn_gate_w", il).c_str());
        bl.ffn_up_w = req(fmt("codec.dec.xfmr.blk.%u.ffn_up_w", il).c_str());
        bl.ffn_down_w = req(fmt("codec.dec.xfmr.blk.%u.ffn_down_w", il).c_str());
        bl.ffn_ls_w = req(fmt("codec.dec.xfmr.blk.%u.ffn_ls_w", il).c_str());
    }

    // Upsample stages
    for (int s = 0; s < 2; s++) {
        auto& up = codec.up[s];
        up.tconv_w = req(fmt("codec.dec.up.%d.tconv_w", s).c_str());
        up.tconv_b = req(fmt("codec.dec.up.%d.tconv_b", s).c_str());
        up.dw_w = req(fmt("codec.dec.up.%d.cnx.dw_w", s).c_str());
        up.dw_b = req(fmt("codec.dec.up.%d.cnx.dw_b", s).c_str());
        up.norm_w = req(fmt("codec.dec.up.%d.cnx.norm_w", s).c_str());
        up.norm_b = req(fmt("codec.dec.up.%d.cnx.norm_b", s).c_str());
        up.pw1_w = req(fmt("codec.dec.up.%d.cnx.pw1_w", s).c_str());
        up.pw1_b = req(fmt("codec.dec.up.%d.cnx.pw1_b", s).c_str());
        up.pw2_w = req(fmt("codec.dec.up.%d.cnx.pw2_w", s).c_str());
        up.pw2_b = req(fmt("codec.dec.up.%d.cnx.pw2_b", s).c_str());
        up.gamma = req(fmt("codec.dec.up.%d.cnx.gamma", s).c_str());
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
            ru.act1_a = req(fmt("codec.dec.blk.%d.res.%d.act1_a", b, u).c_str());
            ru.act1_b = req(fmt("codec.dec.blk.%d.res.%d.act1_b", b, u).c_str());
            ru.act2_a = req(fmt("codec.dec.blk.%d.res.%d.act2_a", b, u).c_str());
            ru.act2_b = req(fmt("codec.dec.blk.%d.res.%d.act2_b", b, u).c_str());
            ru.conv1_w = req(fmt("codec.dec.blk.%d.res.%d.conv1_w", b, u).c_str());
            ru.conv1_b = req(fmt("codec.dec.blk.%d.res.%d.conv1_b", b, u).c_str());
            ru.conv2_w = req(fmt("codec.dec.blk.%d.res.%d.conv2_w", b, u).c_str());
            ru.conv2_b = req(fmt("codec.dec.blk.%d.res.%d.conv2_b", b, u).c_str());
        }
    }

    // Final snake and output conv
    codec.out_snake_a = req("codec.dec.out_snake_a");
    codec.out_snake_b = req("codec.dec.out_snake_b");
    codec.out_conv_w = req("codec.dec.out_conv_w");
    codec.out_conv_b = req("codec.dec.out_conv_b");

    // Codec compute metadata
    c->codec_compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));

    // CPU-only scheduler for the codec graph (see qwen3_tts_context).
    if (!c->codec_sched) {
        ggml_backend_t cpu_backends[1] = {c->backend_cpu};
        c->codec_sched = ggml_backend_sched_new(cpu_backends, nullptr, 1, 8192, false, false);
        if (!c->codec_sched) {
            fprintf(stderr, "qwen3_tts: codec: failed to create CPU scheduler\n");
            return false;
        }
    }

    codec.loaded = true;
    if (c->params.verbosity >= 1) {
        fprintf(stderr, "qwen3_tts: codec loaded from '%s'  (%uL d=%u/%u  rvq=%u)\n", path, hp.n_layers, hp.d_model,
                hp.latent_dim, hp.n_q);
    }

    // Also load the encoder (soft — no fatal error if encoder tensors are missing)
    load_cenc(c);

    return true;
}

// ---------------------------------------------------------------------------
// Per-op trace callback for codec graph debugging on Metal. Enabled when
// QWEN3_TTS_CODEC_TRACE=1 (or QWEN3_TTS_CODEC_FORCE_METAL=1, which implies
// trace). Prints "[idx/N] op_name(tensor_name) shape -> backend" before
// each node, then synchronizes the assigned backend after, so when the
// GPU crashes the last printed line names the offending op.
// ---------------------------------------------------------------------------
struct codec_trace_state {
    ggml_backend_sched_t sched = nullptr;
    int idx = 0;
    int total = 0;
};

static bool codec_trace_eval_cb(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* s = (codec_trace_state*)user_data;
    if (ask) {
        ggml_backend_t be = ggml_backend_sched_get_tensor_backend(s->sched, t);
        const char* be_name = be ? ggml_backend_name(be) : "?";
        char shape[64];
        snprintf(shape, sizeof(shape), "[%lld,%lld,%lld,%lld]", (long long)t->ne[0], (long long)t->ne[1],
                 (long long)t->ne[2], (long long)t->ne[3]);
        fprintf(stderr, "  [%4d/%4d] %-22s %-32s %-22s -> %s\n", s->idx, s->total, ggml_op_name(t->op),
                t->name[0] ? t->name : "(unnamed)", shape, be_name);
        fflush(stderr);
        return true;
    }
    // Post-execute: synchronize so a Metal hang is attributed to *this* op.
    ggml_backend_t be = ggml_backend_sched_get_tensor_backend(s->sched, t);
    if (be) {
        ggml_backend_synchronize(be);
    }
    s->idx++;
    return true;
}

// Returns the codec scheduler to use for compute. Defaults to the CPU-only
// codec_sched; QWEN3_TTS_CODEC_FORCE_METAL=1 routes through the main
// Metal-capable c->sched to reproduce the M1 crash for instrumentation.
static ggml_backend_sched_t codec_pick_sched(qwen3_tts_context* c) {
    if (std::getenv("QWEN3_TTS_CODEC_FORCE_METAL")) {
        return c->sched;
    }
    return c->codec_sched;
}

// ---------------------------------------------------------------------------
// Execute codec decode: codes[T_codec × n_q] → malloc'd float32 PCM.
// Input codes layout: [T_codec, n_q] row-major (T frames, each with n_q codes).
// Output: [T_pcm] float32 @ 24 kHz, caller frees with free().
// ---------------------------------------------------------------------------
static float* codec_decode_codes(qwen3_tts_context* c, const int32_t* codes, int T_codec, int* out_n_samples) {
    if (out_n_samples) {
        *out_n_samples = 0;
    }
    const auto& codec = c->codec;
    const auto& hp = codec.hp;
    const int n_q = (int)hp.n_q;

    // Transpose [T, n_q] → [n_q, T] so each codebook is a contiguous row.
    std::vector<int32_t> codes_t((size_t)T_codec * n_q);
    for (int q = 0; q < n_q; q++) {
        for (int t = 0; t < T_codec; t++) {
            codes_t[(size_t)q * T_codec + t] = codes[(size_t)t * n_q + q];
        }
    }

    // Build sliding-window causal mask
    const int window = (int)hp.sliding_window;
    std::vector<ggml_fp16_t> mask_data;
    if (T_codec > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask_data.assign((size_t)T_codec * T_codec, neginf_h);
        for (int q = 0; q < T_codec; q++) {
            for (int k = 0; k <= q; k++) {
                if ((q - k) < window) {
                    mask_data[(size_t)q * T_codec + k] = zero_h;
                }
            }
        }
    }

    // Build positions [0..T-1]
    std::vector<int32_t> pos(T_codec);
    for (int i = 0; i < T_codec; i++) {
        pos[i] = i;
    }

    ggml_cgraph* gf = build_graph_codec_decode(c, T_codec);
    ggml_backend_sched_t sched = codec_pick_sched(c);
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "qwen3_tts: codec: graph alloc failed\n");
        return nullptr;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codec_codes"), codes_t.data(), 0,
                            codes_t.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codec_positions"), pos.data(), 0, pos.size() * sizeof(int32_t));
    if (T_codec > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codec_mask"), mask_data.data(), 0,
                                mask_data.size() * sizeof(ggml_fp16_t));
    }

    codec_trace_state ts{sched, 0, ggml_graph_n_nodes(gf)};
    const bool trace = std::getenv("QWEN3_TTS_CODEC_TRACE") || std::getenv("QWEN3_TTS_CODEC_FORCE_METAL");
    if (trace) {
        fprintf(stderr, "qwen3_tts: codec: tracing %d nodes (sched=%s)\n", ts.total,
                sched == c->codec_sched ? "codec_cpu" : "main");
        ggml_backend_sched_set_eval_callback(sched, codec_trace_eval_cb, &ts);
    }

    ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    if (trace) {
        ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
    }
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_tts: codec: compute failed (status=%d, last_op_idx=%d)\n", (int)st, ts.idx);
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "pcm");
    const int n_samples = (int)ggml_nelements(out);
    float* pcm = (float*)malloc((size_t)n_samples * sizeof(float));
    if (!pcm) {
        return nullptr;
    }
    ggml_backend_tensor_get(out, pcm, 0, (size_t)n_samples * sizeof(float));
    if (out_n_samples) {
        *out_n_samples = n_samples;
    }
    return pcm;
}

// ---------------------------------------------------------------------------
// Run codec graph and extract a named intermediate tensor.
// Used by the diff harness to compare stage outputs against PyTorch.
// Returns a malloc'd float array of *out_n elements, or nullptr on failure.
// ---------------------------------------------------------------------------
static float* codec_extract_stage(qwen3_tts_context* c, const int32_t* codes, int T_codec, const char* stage_name,
                                  int* out_n) {
    if (out_n) {
        *out_n = 0;
    }
    const auto& codec = c->codec;
    const auto& hp = codec.hp;
    const int n_q = (int)hp.n_q;

    std::vector<int32_t> codes_t((size_t)T_codec * n_q);
    for (int q = 0; q < n_q; q++) {
        for (int t = 0; t < T_codec; t++) {
            codes_t[(size_t)q * T_codec + t] = codes[(size_t)t * n_q + q];
        }
    }

    const int window = (int)hp.sliding_window;
    std::vector<ggml_fp16_t> mask_data;
    if (T_codec > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask_data.assign((size_t)T_codec * T_codec, neginf_h);
        for (int q = 0; q < T_codec; q++) {
            for (int k = 0; k <= q; k++) {
                if ((q - k) < window) {
                    mask_data[(size_t)q * T_codec + k] = zero_h;
                }
            }
        }
    }
    std::vector<int32_t> pos(T_codec);
    for (int i = 0; i < T_codec; i++) {
        pos[i] = i;
    }

    ggml_cgraph* gf = build_graph_codec_decode(c, T_codec);
    ggml_backend_sched_t sched = codec_pick_sched(c);
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        return nullptr;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codec_codes"), codes_t.data(), 0,
                            codes_t.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codec_positions"), pos.data(), 0, pos.size() * sizeof(int32_t));
    if (T_codec > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codec_mask"), mask_data.data(), 0,
                                mask_data.size() * sizeof(ggml_fp16_t));
    }

    codec_trace_state ts{sched, 0, ggml_graph_n_nodes(gf)};
    const bool trace = std::getenv("QWEN3_TTS_CODEC_TRACE") || std::getenv("QWEN3_TTS_CODEC_FORCE_METAL");
    if (trace) {
        fprintf(stderr, "qwen3_tts: codec: tracing %d nodes (sched=%s, stage=%s)\n", ts.total,
                sched == c->codec_sched ? "codec_cpu" : "main", stage_name);
        ggml_backend_sched_set_eval_callback(sched, codec_trace_eval_cb, &ts);
    }

    ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    if (trace) {
        ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
    }
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_tts: codec: compute failed (status=%d, last_op_idx=%d)\n", (int)st, ts.idx);
        return nullptr;
    }

    ggml_tensor* t = ggml_graph_get_tensor(gf, stage_name);
    if (!t) {
        fprintf(stderr, "qwen3_tts: codec: stage '%s' not found in graph\n", stage_name);
        return nullptr;
    }
    const size_t n = ggml_nelements(t);
    float* buf = (float*)malloc(n * sizeof(float));
    if (!buf) {
        return nullptr;
    }
    ggml_backend_tensor_get(t, buf, 0, n * sizeof(float));
    if (out_n) {
        *out_n = (int)n;
    }
    return buf;
}

// ============================================================================
// Codec encoder implementation (SEANet + enc transformer + RVQ encode)
// ============================================================================

// ---------------------------------------------------------------------------
// Helpers shared with kyutai_stt: causal conv1d in [T, C] format.
// For the SEANet encoder, tensors are in [T, C] format (ne[0]=T innermost).
// ---------------------------------------------------------------------------

// Replicate-pad a [T, C] tensor along the time axis: pad_left copies of x[0]
// on the left, pad_right copies of x[T-1] on the right. Output is contiguous.
static ggml_tensor* cenc_replicate_pad(ggml_context* ctx, ggml_tensor* x, int pad_left, int pad_right) {
    if (pad_left == 0 && pad_right == 0) {
        return x;
    }
    const int T = (int)x->ne[0];
    const int C = (int)x->ne[1];
    ggml_tensor* result = x;
    if (pad_left > 0) {
        // first frame: ggml_view_2d(x, ne0=1, ne1=C, nb1=x->nb[1], offset=0)
        ggml_tensor* first = ggml_view_2d(ctx, x, 1, C, x->nb[1], 0);
        // tile to [pad_left, C]
        ggml_tensor* target = ggml_new_tensor_2d(ctx, x->type, pad_left, C);
        ggml_tensor* left = ggml_repeat(ctx, first, target);
        result = ggml_concat(ctx, left, result, 0); // concat along ne[0]=T
    }
    if (pad_right > 0) {
        const int T_now = (int)result->ne[0];
        ggml_tensor* last = ggml_view_2d(ctx, result, 1, C, result->nb[1], (size_t)(T_now - 1) * result->nb[0]);
        ggml_tensor* target = ggml_new_tensor_2d(ctx, x->type, pad_right, C);
        ggml_tensor* right = ggml_repeat(ctx, last, target);
        result = ggml_concat(ctx, result, right, 0);
    }
    (void)T; // suppress unused warning
    return result;
}

// SEANet/Mimi causal conv1d.
// Input: [T, C_in] → output [T_out, C_out].
// Causal padding (all on the left), plus optional extra-right padding
// for stride alignment (matches MimiConv1d._get_extra_padding_for_conv1d).
// pad_replicate: if true, use replicate (edge-value) padding instead of zero.
static ggml_tensor* cenc_conv1d_ext(ggml_context* ctx, ggml_tensor* x, const g3t_cenc_conv& c, int stride,
                                    bool pad_replicate) {
    const int K = (int)c.w->ne[0];
    const int pad_total = K - stride; // causal: all goes left
    const int T_in = (int)x->ne[0];

    // PyTorch's exact formula:
    //   n_frames = ceil((T - K + pad_total)/stride + 1) - 1
    //   ideal    = n_frames * stride + K - pad_total
    //   extra    = max(0, ideal - T_in)
    int extra_right = 0;
    if (stride >= 1) {
        // floats to mirror PyTorch's float ceil
        double n_frames_f = (double)(T_in - K + pad_total) / (double)stride + 1.0;
        int n_frames = (int)std::ceil(n_frames_f) - 1;
        int ideal = n_frames * stride + K - pad_total;
        extra_right = ideal - T_in;
        if (extra_right < 0) {
            extra_right = 0;
        }
    }
    if (pad_total > 0 || extra_right > 0) {
        if (pad_replicate) {
            x = cenc_replicate_pad(ctx, x, pad_total, extra_right);
        } else {
            x = ggml_pad_ext(ctx, x, pad_total, extra_right, 0, 0, 0, 0, 0, 0);
        }
    }
    x = ggml_conv_1d(ctx, c.w, x, stride, 0, 1);
    if (ggml_n_dims(x) > 2) {
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
    }
    if (c.b) {
        ggml_tensor* b = ggml_reshape_2d(ctx, c.b, 1, ggml_nelements(c.b));
        x = ggml_add(ctx, x, b);
    }
    return x;
}

// Convenience: zero-pad version (used by all SEANet convs).
static ggml_tensor* cenc_conv1d(ggml_context* ctx, ggml_tensor* x, const g3t_cenc_conv& c, int stride) {
    return cenc_conv1d_ext(ctx, x, c, stride, /*pad_replicate*/ false);
}

static ggml_tensor* cenc_elu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_elu(ctx, x);
}

// SEANet residual block: ELU → shortcut(k=3) → ELU → expand(k=1) → add(skip).
// Matches kyutai_stt's resblock_fwd (Mimi/SEANet convention).
static ggml_tensor* cenc_resblk(ggml_context* ctx, ggml_tensor* x, const g3t_cenc_resblk& b) {
    ggml_tensor* h = cenc_elu(ctx, x);
    h = cenc_conv1d(ctx, h, b.shortcut, 1);
    h = cenc_elu(ctx, h);
    h = cenc_conv1d(ctx, h, b.expand, 1);
    return ggml_add(ctx, x, h); // skip connection uses original x
}

// Full SEANet encoder: audio [1, T] → features [T/960, 512].
// Input: pcm in [T_audio, 1] format (1-channel raw PCM).
// graph parameter: if non-null, intra-SEANet checkpoints are added to the
// graph as outputs (cenc_se_init, cenc_se_stride{0..3}, cenc_se_final).
static ggml_tensor* build_cenc_seanet(ggml_context* ctx, const g3t_cenc_seanet& se, ggml_tensor* pcm,
                                      ggml_cgraph* graph_for_checkpoints = nullptr) {
    static const int strides[] = {4, 5, 6, 8};
    ggml_tensor* x = pcm;                // [T, 1]
    x = cenc_conv1d(ctx, x, se.init, 1); // → [T, 64]
    if (graph_for_checkpoints) {
        ggml_tensor* dump = ggml_cont(ctx, ggml_transpose(ctx, x));
        ggml_set_name(dump, "cenc_se_init");
        ggml_set_output(dump);
        ggml_build_forward_expand(graph_for_checkpoints, dump);
    }
    for (int i = 0; i < 4; i++) {
        x = cenc_resblk(ctx, x, se.resblk[i]);
        x = cenc_elu(ctx, x);
        x = cenc_conv1d(ctx, x, se.ds[i], strides[i]);
        if (graph_for_checkpoints) {
            ggml_tensor* dump = ggml_cont(ctx, ggml_transpose(ctx, x));
            char name[32];
            snprintf(name, sizeof(name), "cenc_se_s%d", i);
            ggml_set_name(dump, name);
            ggml_set_output(dump);
            ggml_build_forward_expand(graph_for_checkpoints, dump);
        }
    }
    x = cenc_elu(ctx, x);
    x = cenc_conv1d(ctx, x, se.final, 1); // → [T/960, 512]
    return x;                             // [T_enc, 512] in [T, C] format
}

// Encoder transformer (8L): [d,T] channels-first so ggml_norm normalizes over d.
// Input: [T_enc, 512] from SEANet → converts to [512, T] internally.
// Output: [T_enc, 512] (back-converted).
// Mimi's encoder transformer uses CAUSAL attention with sliding window=250.
static ggml_tensor* build_cenc_transformer(ggml_context* ctx, const std::vector<g3t_cenc_xfmr_layer>& layers,
                                           ggml_tensor* x) {
    // Transpose from SEANet [T, d] → [d, T] for ggml_norm (normalizes over ne[0]=d)
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [d=512, T]

    const int n_heads = 8, head_dim = 64;
    const int sliding_window = 250;
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);
    const int T = (int)x->ne[1];
    const int d = (int)x->ne[0]; // 512

    ggml_tensor* pos = ggml_cast(ctx, ggml_arange(ctx, 0.0f, (float)T, 1.0f), GGML_TYPE_I32);

    // Causal sliding-window mask [T, T] F16 — set as input, filled at compute.
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);
        ggml_set_name(causal_mask, "cenc_mask");
        ggml_set_input(causal_mask);
    }
    (void)sliding_window; // unused — encoder T is always < 250

    for (const auto& L : layers) {
        ggml_tensor* residual = x;

        // LayerNorm over d for each T (ggml_norm normalizes ne[0]=d) + bias
        ggml_tensor* h = ggml_norm(ctx, x, 1e-5f);
        h = ggml_mul(ctx, h, L.norm1_w);
        h = ggml_add(ctx, h, L.norm1_b);

        // Q/K/V: h=[d,T], weight=[d,d] → mul_mat → [d,T]
        ggml_tensor* Q = ggml_mul_mat(ctx, L.attn_q_w, h);
        ggml_tensor* K = ggml_mul_mat(ctx, L.attn_k_w, h);
        ggml_tensor* V = ggml_mul_mat(ctx, L.attn_v_w, h);

        Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, T);
        K = ggml_reshape_3d(ctx, K, head_dim, n_heads, T);
        V = ggml_reshape_3d(ctx, V, head_dim, n_heads, T);

        Q = ggml_rope_ext(ctx, Q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);
        K = ggml_rope_ext(ctx, K, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);

        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, causal_mask, attn_scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx, attn, d, T);    // [d, T]
        attn = ggml_mul_mat(ctx, L.attn_o_w, attn); // [d, T]
        if (L.attn_ls) {
            ggml_tensor* ls = ggml_reshape_2d(ctx, L.attn_ls, d, 1);
            attn = ggml_mul(ctx, attn, ls); // [d,T] * [d,1] → broadcasts ✓
        }
        x = ggml_add(ctx, residual, attn);

        // FFN
        residual = x;
        h = ggml_norm(ctx, x, 1e-5f);
        h = ggml_mul(ctx, h, L.norm2_w);
        h = ggml_add(ctx, h, L.norm2_b);

        h = ggml_mul_mat(ctx, L.fc1_w, h); // [4d, T]
        h = ggml_gelu(ctx, h);
        h = ggml_mul_mat(ctx, L.fc2_w, h); // [d, T]
        if (L.ffn_ls) {
            ggml_tensor* ls = ggml_reshape_2d(ctx, L.ffn_ls, d, 1);
            h = ggml_mul(ctx, h, ls);
        }
        x = ggml_add(ctx, residual, h);
    }
    // Back to [T, d] for the downsample conv
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

// Build the SEANet→transformer→downsample graph and return downsampled embeddings.
// pcm_input: named F32 1D tensor [n_samples] (set as input before compute).
static ggml_cgraph* build_cenc_graph(qwen3_tts_context* c, int n_samples) {
    const auto& ce = c->cenc;
    size_t mem = c->cenc_compute_meta.size();
    ggml_init_params ip = {mem, c->cenc_compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor* pcm = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_samples);
    ggml_set_name(pcm, "pcm_input");
    ggml_set_input(pcm);

    // Reshape to [n_samples, 1] for conv1d (b=[T, IC])
    ggml_tensor* x = ggml_reshape_2d(ctx0, pcm, n_samples, 1); // [T, IC=1]
    x = build_cenc_seanet(ctx0, ce.seanet, x, gf);             // intra-SEANet checkpoints attached to gf
    // For diff harness: transpose to [C, T_enc] = ne=[512, T_enc] so the flat
    // memory layout matches Python's (T_enc, 512) numpy stored to GGUF.
    {
        ggml_tensor* dump = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // ne=[512, T_enc]
        ggml_set_name(dump, "cenc_seanet_out");
        ggml_set_output(dump);
        ggml_build_forward_expand(gf, dump);
    }

    x = build_cenc_transformer(ctx0, ce.xfmr_layers, x); // [T_enc, 512]
    {
        ggml_tensor* dump = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // ne=[512, T_enc]
        ggml_set_name(dump, "cenc_xfmr_out");
        ggml_set_output(dump);
        ggml_build_forward_expand(gf, dump);
    }

    // Downsample uses replicate padding (Mimi pad_mode='replicate' for this
    // single layer, while all other encoder convs use 'constant' zero pad).
    g3t_cenc_conv ds_conv = {ce.downsample.w, ce.downsample.b};
    ggml_tensor* tmp = cenc_conv1d_ext(ctx0, x, ds_conv, 2, /*replicate*/ true); // ne=[T_frames, 512]
    {
        ggml_tensor* dump = ggml_cont(ctx0, ggml_transpose(ctx0, tmp)); // ne=[512, T_frames]
        ggml_set_name(dump, "cenc_ds_out");
        ggml_set_output(dump);
        ggml_build_forward_expand(gf, dump);
    }

    // Transpose to channels-first [512, T_frames] for the RVQ encode step
    tmp = ggml_cont(ctx0, ggml_transpose(ctx0, tmp)); // ne=[512, T_frames]
    ggml_set_name(tmp, "enc_emb");
    ggml_set_output(tmp);
    ggml_build_forward_expand(gf, tmp);
    ggml_free(ctx0);
    return gf;
}

// ---------------------------------------------------------------------------
// CPU-side RVQ encode: k=1 proj matmul + nearest-neighbor search.
// ---------------------------------------------------------------------------

// Apply k=1 conv weight as matrix multiply on [IC, T] → [OC, T].
// FIXED: Expects x as [T, IC] row-major, outputs out as [T, OC] row-major.
static void cpu_k1_proj(const float* x, int T, const float* w, int IC, int OC, float* out) {
    for (int t = 0; t < T; t++) {
        for (int co = 0; co < OC; co++) {
            float s = 0.0f;
            const float* w_co = w + (size_t)co * IC; // row co of (OC, IC) matrix
            for (int ci = 0; ci < IC; ci++) {
                s += x[(size_t)t * IC + ci] * w_co[ci];
            }
            out[(size_t)t * OC + co] = s;
        }
    }
}

// Find nearest codebook entry for each T frame. cb: [2048, 256] row-major.
// FIXED: Expects x as [T, cdim] row-major.
static void rvq_nearest_neighbor(const float* x, int T, const float* cb, int n_cb, int cdim,
                                 std::vector<int32_t>& codes) {
    codes.resize(T);
    for (int t = 0; t < T; t++) {
        float best = INFINITY;
        int best_k = 0;
        for (int k = 0; k < n_cb; k++) {
            float d = 0;
            for (int c = 0; c < cdim; c++) {
                float diff = x[(size_t)t * cdim + c] - cb[(size_t)k * cdim + c];
                d += diff * diff;
            }
            if (d < best) {
                best = d;
                best_k = k;
            }
        }
        codes[t] = best_k;
    }
}

// Subtract quantized entries from residual: residual -= cb[codes[t]] for each t.
// FIXED: Expects residual as [T, cdim] row-major.
static void rvq_subtract(float* residual, int T, const float* cb, int cdim, const std::vector<int32_t>& codes) {
    for (int t = 0; t < T; t++) {
        const float* entry = cb + (size_t)codes[t] * cdim;
        for (int c = 0; c < cdim; c++) {
            residual[(size_t)t * cdim + c] -= entry[c];
        }
    }
}

// Full RVQ encode: embeddings [512, T_frames] → codes [T_frames, 16] (row-major).
static bool cenc_rvq_encode(qwen3_tts_context* c, const float* emb, int T, std::vector<int32_t>& out_codes) {
    const auto& ce = c->cenc;
    const int dim = 512, cdim = 256, n_cb = 2048;
    const int n_q = 16; // 1 semantic + 15 acoustic
    out_codes.assign((size_t)T * n_q, 0);

    // Load k=1 projection weights from GPU tensors to CPU.
    // The encoder has two independent RVQ branches:
    //   semantic: emb[512,T] -> sem_in -> [256,T] -> 1 residual codebook
    //   acoustic: emb[512,T] -> ac_in  -> [256,T] -> 15 residual codebooks
    // Residual subtraction happens inside each branch's 256-d space only.
    std::vector<float> sem_in((size_t)dim * cdim);
    std::vector<float> ac_in((size_t)dim * cdim);
    ggml_backend_tensor_get(ce.sem_in_proj.w, sem_in.data(), 0, sem_in.size() * sizeof(float));
    ggml_backend_tensor_get(ce.ac_in_proj.w, ac_in.data(), 0, ac_in.size() * sizeof(float));

    // Step 1: Semantic branch (1 codebook).
    std::vector<float> z_sem((size_t)cdim * T);
    cpu_k1_proj(emb, T, sem_in.data(), dim, cdim, z_sem.data());
    std::vector<int32_t> sem_codes;
    rvq_nearest_neighbor(z_sem.data(), T, ce.sem_cb.data.data(), n_cb, cdim, sem_codes);
    for (int t = 0; t < T; t++) {
        out_codes[(size_t)t * n_q + 0] = sem_codes[t];
    }

    // Step 2: Acoustic branch (15 residual codebooks in 256-dim space).
    std::vector<float> z_ac((size_t)cdim * T);
    cpu_k1_proj(emb, T, ac_in.data(), dim, cdim, z_ac.data());
    for (int q = 0; q < 15; q++) {
        std::vector<int32_t> ac_codes_q;
        rvq_nearest_neighbor(z_ac.data(), T, ce.ac_cb[q].data.data(), n_cb, cdim, ac_codes_q);
        for (int t = 0; t < T; t++) {
            out_codes[(size_t)t * n_q + q + 1] = ac_codes_q[t];
        }
        rvq_subtract(z_ac.data(), T, ce.ac_cb[q].data.data(), cdim, ac_codes_q);
    }
    return true;
}

// Build a causal mask [T_enc, T_enc] for the encoder transformer.
// T_enc = T_audio / 960 (post-SEANet, pre-downsample).
static std::vector<ggml_fp16_t> build_cenc_mask(int T_enc) {
    std::vector<ggml_fp16_t> mask;
    if (T_enc <= 1) {
        return mask;
    }
    const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
    mask.assign((size_t)T_enc * T_enc, zero_h);
    for (int q = 0; q < T_enc; q++) {
        for (int k = q + 1; k < T_enc; k++) {
            mask[(size_t)q * T_enc + k] = neginf_h;
        }
    }
    return mask;
}

// Run codec encoder: 24kHz PCM → codes [T_frames × 16] row-major.
static bool run_cenc(qwen3_tts_context* c, const float* audio, int n_samples, std::vector<int32_t>& codes,
                     int& T_frames) {
    // Build and execute the SEANet+transformer+downsample graph
    ggml_cgraph* gf = build_cenc_graph(c, n_samples);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "qwen3_tts: cenc: graph alloc failed\n");
        return false;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pcm_input"), audio, 0, (size_t)n_samples * sizeof(float));
    // Set causal mask. T_enc = T_audio/960 (4*5*6*8 SEANet stride product)
    {
        const int T_enc = n_samples / 960;
        ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "cenc_mask");
        if (mask_t && T_enc > 1) {
            auto mask = build_cenc_mask(T_enc);
            ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
    }
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_tts: cenc: compute failed\n");
        return false;
    }
    ggml_tensor* emb_t = ggml_graph_get_tensor(gf, "enc_emb");
    T_frames = (int)emb_t->ne[1];
    const int dim = 512;
    std::vector<float> emb((size_t)dim * T_frames);
    ggml_backend_tensor_get(emb_t, emb.data(), 0, emb.size() * sizeof(float));
    return cenc_rvq_encode(c, emb.data(), T_frames, codes);
}

// ---------------------------------------------------------------------------
// Load codec encoder tensors from the codec GGUF (tensors already in ctx->codec.tensors).
// ---------------------------------------------------------------------------
static bool load_cenc(qwen3_tts_context* c) {
    auto& ce = c->cenc;
    auto& ts = c->codec.tensors; // codec tensors already loaded
    auto G = [&](const char* n) -> ggml_tensor* {
        auto it = ts.find(n);
        return it == ts.end() ? nullptr : it->second;
    };
    auto need = [&](const char* n) -> ggml_tensor* {
        auto* t = G(n);
        if (!t) {
            fprintf(stderr, "qwen3_tts: cenc: missing '%s'\n", n);
        }
        return t;
    };

    // SEANet
    ce.seanet.init.w = need("codec.enc.seanet.init_w");
    ce.seanet.init.b = need("codec.enc.seanet.init_b");
    ce.seanet.final.w = need("codec.enc.seanet.final_w");
    ce.seanet.final.b = need("codec.enc.seanet.final_b");
    char buf[128];
    for (int s = 0; s < 4; s++) {
        snprintf(buf, sizeof(buf), "codec.enc.seanet.blk.%d.short_w", s);
        ce.seanet.resblk[s].shortcut.w = need(buf);
        snprintf(buf, sizeof(buf), "codec.enc.seanet.blk.%d.short_b", s);
        ce.seanet.resblk[s].shortcut.b = need(buf);
        snprintf(buf, sizeof(buf), "codec.enc.seanet.blk.%d.exp_w", s);
        ce.seanet.resblk[s].expand.w = need(buf);
        snprintf(buf, sizeof(buf), "codec.enc.seanet.blk.%d.exp_b", s);
        ce.seanet.resblk[s].expand.b = need(buf);
        snprintf(buf, sizeof(buf), "codec.enc.seanet.ds.%d.w", s);
        ce.seanet.ds[s].w = need(buf);
        snprintf(buf, sizeof(buf), "codec.enc.seanet.ds.%d.b", s);
        ce.seanet.ds[s].b = need(buf);
        if (!ce.seanet.resblk[s].shortcut.w || !ce.seanet.ds[s].w) {
            return false;
        }
    }

    // Encoder transformer
    ce.xfmr_layers.resize(8);
    for (int li = 0; li < 8; li++) {
        auto& L = ce.xfmr_layers[li];
        snprintf(buf, sizeof(buf), "codec.enc.xfmr.blk.%d.", li);
        std::string p(buf);
        L.norm1_w = need((p + "norm1_w").c_str());
        L.norm1_b = need((p + "norm1_b").c_str());
        L.norm2_w = need((p + "norm2_w").c_str());
        L.norm2_b = need((p + "norm2_b").c_str());
        L.attn_q_w = need((p + "attn_q_w").c_str());
        L.attn_k_w = need((p + "attn_k_w").c_str());
        L.attn_v_w = need((p + "attn_v_w").c_str());
        L.attn_o_w = need((p + "attn_o_w").c_str());
        L.attn_ls = G((p + "attn_ls").c_str());
        L.fc1_w = need((p + "fc1_w").c_str());
        L.fc2_w = need((p + "fc2_w").c_str());
        L.ffn_ls = G((p + "ffn_ls").c_str());
        if (!L.norm1_w || !L.attn_q_w || !L.fc1_w) {
            return false;
        }
    }

    // Downsample
    ce.downsample.w = need("codec.enc.ds_w");

    // RVQ projection weights (stay in GPU tensors; CPU read at encode time)
    ce.sem_in_proj.w = need("codec.enc.rvq.sem.in_w");
    ce.sem_out_proj.w = need("codec.enc.rvq.sem.out_w");
    ce.ac_in_proj.w = need("codec.enc.rvq.ac.in_w");
    ce.ac_out_proj.w = need("codec.enc.rvq.ac.out_w");
    if (!ce.sem_in_proj.w || !ce.ac_in_proj.w) {
        return false;
    }

    // Load codebooks to CPU (they're stored as F32 in the GGUF)
    const int cdim = 256, n_cb = 2048;
    auto load_cb = [&](const char* name, g3t_cenc_rvq_cb& cb) -> bool {
        auto* t = need(name);
        if (!t) {
            return false;
        }
        cb.data.resize((size_t)n_cb * cdim);
        ggml_backend_tensor_get(t, cb.data.data(), 0, cb.data.size() * sizeof(float));
        return true;
    };
    if (!load_cb("codec.enc.rvq.sem.cb", ce.sem_cb)) {
        return false;
    }
    for (int i = 0; i < 15; i++) {
        snprintf(buf, sizeof(buf), "codec.enc.rvq.ac.%d.cb", i);
        if (!load_cb(buf, ce.ac_cb[i])) {
            return false;
        }
    }

    c->cenc_compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ce.loaded = true;
    if (c->params.verbosity >= 1) {
        fprintf(stderr, "qwen3_tts: codec encoder loaded (SEANet+xfmr+RVQ)\n");
    }
    return true;
}

// ============================================================================
// Speaker encoder implementation (PLAN #52 step 4 — ECAPA-TDNN)
// ============================================================================

static void spk_fft_r2c(const float* in, int N, float* out) {
    int bits = 0;
    for (int n = N; n > 1; n >>= 1) {
        bits++;
    }
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int b = 0; b < bits; b++) {
            rev = (rev << 1) | ((i >> b) & 1);
        }
        out[(size_t)2 * rev] = in[i];
        out[(size_t)2 * rev + 1] = 0.0f;
    }
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, bj = i + j + len / 2;
                float are = out[(size_t)2 * a], aim = out[(size_t)2 * a + 1], bre = out[(size_t)2 * bj],
                      bim = out[(size_t)2 * bj + 1];
                float tre = ure * bre - uim * bim, tim = ure * bim + uim * bre;
                out[(size_t)2 * a] = are + tre;
                out[(size_t)2 * a + 1] = aim + tim;
                out[(size_t)2 * bj] = are - tre;
                out[(size_t)2 * bj + 1] = aim - tim;
                float nr = ure * wre - uim * wim;
                uim = ure * wim + uim * wre;
                ure = nr;
            }
        }
    }
}

// Slaney-normalized librosa-default mel filterbank (Slaney scale, htk=False).
// Layout: [n_mels, n_freqs] row-major (FbLayout::MelsFreqs).
// The Slaney mel scale is LINEAR below 1 kHz and LOG above, matching librosa's default.
static std::vector<float> build_slaney_mel_fb(int n_mels, int n_fft, int sr, float fmin, float fmax) {
    const int n_freqs = n_fft / 2 + 1;
    // Slaney mel scale (librosa htk=False default)
    const float f_sp = 200.0f / 3.0f; // linear spacing: ~66.67 Hz
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = logf(6.4f) / 27.0f;
    auto hz_to_mel = [&](float hz) -> float {
        if (hz >= min_log_hz) {
            return min_log_mel + logf(hz / min_log_hz) / logstep;
        }
        return hz / f_sp;
    };
    auto mel_to_hz = [&](float mel) -> float {
        if (mel >= min_log_mel) {
            return min_log_hz * expf(logstep * (mel - min_log_mel));
        }
        return mel * f_sp;
    };
    const float mel_lo = hz_to_mel(fmin);
    const float mel_hi = hz_to_mel(fmax > 0.0f ? fmax : (float)sr / 2.0f);
    std::vector<float> mel_pts(n_mels + 2), hz_pts(n_mels + 2);
    for (int i = 0; i <= n_mels + 1; i++) {
        hz_pts[i] = mel_to_hz(mel_lo + (mel_hi - mel_lo) * i / (n_mels + 1));
    }
    std::vector<float> freq_bins(n_freqs);
    for (int k = 0; k < n_freqs; k++) {
        freq_bins[k] = (float)sr * k / (float)n_fft;
    }
    std::vector<float> fb((size_t)n_mels * n_freqs, 0.0f);
    for (int m = 0; m < n_mels; m++) {
        float lo = hz_pts[m], ctr = hz_pts[m + 1], hi = hz_pts[m + 2];
        float enorm = (hi > lo) ? 2.0f / (hi - lo) : 1.0f;
        for (int k = 0; k < n_freqs; k++) {
            float f = freq_bins[k], w = 0.0f;
            if (f >= lo && f < ctr && ctr > lo) {
                w = (f - lo) / (ctr - lo);
            } else if (f >= ctr && f <= hi && hi > ctr) {
                w = (hi - f) / (hi - ctr);
            }
            fb[(size_t)m * n_freqs + k] = w * enorm;
        }
    }
    return fb;
}

// 128-mel spectrogram for the speaker encoder: reflect-pad + Hann STFT + slaney mel.
// Returns (T, 128) row-major float32.
static std::vector<float> compute_spk_mel(const float* audio, int n_samples, int* T_out) {
    const int n_fft = 1024, hop = 256, n_mels = 128, sr = 24000;
    const float fmin = 0.0f, fmax = 12000.0f;
    const int pad = (n_fft - hop) / 2; // 384 reflect-pad on each side

    // Reflect-pad audio: matches PyTorch F.pad(mode='reflect') which excludes
    // the boundary element (e.g. [a,b,c] padded by 2 → [c,b,a,b,c,b,a]).
    std::vector<float> audio_p(n_samples + 2 * pad, 0.0f);
    for (int i = 0; i < pad; i++) {
        audio_p[i] = audio[std::min(pad - i, n_samples - 1)]; // left: audio[pad-i..1]
    }
    for (int i = 0; i < n_samples; i++) {
        audio_p[pad + i] = audio[i];
    }
    for (int i = 0; i < pad; i++) {
        audio_p[pad + n_samples + i] = audio[std::max(n_samples - 2 - i, 0)]; // right
    }

    // Periodic Hann window
    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; i++) {
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)n_fft));
    }

    const int n_freqs = n_fft / 2 + 1;
    auto mel_fb = build_slaney_mel_fb(n_mels, n_fft, sr, fmin, fmax);

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.log_eps = 1e-5f;
    // PyTorch reference computes magnitude spectrum: sqrt(re²+im²+1e-9)
    // before applying the mel filterbank, not power spectrum.
    p.spec_kind = core_mel::SpecKind::Magnitude;
    p.norm = core_mel::Normalization::None;
    p.layout = core_mel::Layout::TimeMels; // (T, n_mels) output
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.matmul = core_mel::MatmulPrecision::Double;
    p.center_pad = false; // already reflect-padded

    int T = 0;
    auto mel = core_mel::compute(audio_p.data(), (int)audio_p.size(), hann.data(), n_fft, mel_fb.data(), n_freqs,
                                 spk_fft_r2c, p, T);
    if (T_out) {
        *T_out = T;
    }
    return mel; // (T, 128) row-major
}

// ---------------------------------------------------------------------------
// ECAPA graph builder helpers
// ---------------------------------------------------------------------------

// Conv1d with symmetric REFLECT padding ("same", matching PyTorch padding_mode='reflect').
// Input/output: [C, T] channels-first.
static ggml_tensor* spk_same_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int dilation) {
    const int K = (int)w->ne[0];
    const int pad = (K - 1) * dilation / 2;
    // Transpose [C, T] → [T, C] for ggml_pad_reflect_1d and ggml_conv_1d
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [T, C]
    if (pad > 0) {
        x = ggml_pad_reflect_1d(ctx, x, pad, pad); // reflect-pad T dimension
    }
    x = ggml_conv_1d(ctx, w, x, 1, 0, dilation); // [T_out, C_out] (p0=0, already padded)
    x = ggml_cont(ctx, ggml_transpose(ctx, x));  // [C_out, T_out]
    if (b) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}

static ggml_tensor* spk_tdnn_block(ggml_context* ctx, ggml_tensor* x, const g3t_spk_tdnn_w& t, int dilation) {
    return ggml_relu(ctx, spk_same_conv1d(ctx, x, t.w, t.b, dilation));
}

static ggml_tensor* spk_se_block(ggml_context* ctx, ggml_tensor* x, const g3t_spk_se& se) {
    const int T = (int)x->ne[1];
    // Global mean over T for each C channel
    ggml_tensor* m =
        ggml_cont(ctx, ggml_transpose(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, x))),
                                                      1.0f / T))); // [C, 1]
    auto w1 = ggml_reshape_2d(ctx, se.c1w, se.c1w->ne[1], se.c1w->ne[2]);
    ggml_tensor* h = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w1, m), se.c1b));
    auto w2 = ggml_reshape_2d(ctx, se.c2w, se.c2w->ne[1], se.c2w->ne[2]);
    ggml_tensor* sc = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w2, h), se.c2b));
    return ggml_mul(ctx, x, ggml_repeat(ctx, sc, x));
}

static ggml_tensor* spk_res2net_block(ggml_context* ctx, ggml_tensor* x, const g3t_spk_res2net& r, int dilation) {
    const int T = (int)x->ne[1];
    const int chunk = 64; // C/8
    ggml_tensor* outs[8];
    for (int i = 0; i < 8; i++) {
        ggml_tensor* ci = ggml_cont(ctx, ggml_view_2d(ctx, x, chunk, T, x->nb[1], (size_t)i * chunk * sizeof(float)));
        if (i == 0) {
            outs[i] = ci;
            continue;
        }
        ggml_tensor* in = (i == 1) ? ci : ggml_add(ctx, ci, outs[i - 1]);
        outs[i] = spk_tdnn_block(ctx, in, r.blocks[i - 1], dilation);
    }
    ggml_tensor* out = outs[0];
    for (int i = 1; i < 8; i++) {
        out = ggml_concat(ctx, out, outs[i], 0);
    }
    return out;
}

static ggml_tensor* spk_se_res2net(ggml_context* ctx, ggml_tensor* x, const g3t_spk_se_res2net& blk, int d) {
    ggml_tensor* res = x;
    x = spk_tdnn_block(ctx, x, blk.tdnn1, 1);
    x = spk_res2net_block(ctx, x, blk.res2net, d);
    x = spk_tdnn_block(ctx, x, blk.tdnn2, 1);
    x = spk_se_block(ctx, x, blk.se);
    return ggml_add(ctx, x, res);
}

static ggml_tensor* spk_asp_block(ggml_context* ctx, ggml_tensor* x, const g3t_spk_asp& asp) {
    const int T = (int)x->ne[1];
    // Global statistics for attention input
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* m1C = ggml_scale(ctx, ggml_sum_rows(ctx, xT), 1.0f / T); // [1, C]
    ggml_tensor* mC1 = ggml_cont(ctx, ggml_transpose(ctx, m1C));          // [C, 1]
    ggml_tensor* mCT = ggml_repeat(ctx, mC1, x);                          // [C, T]
    ggml_tensor* d2 = ggml_mul(ctx, ggml_sub(ctx, x, mCT), ggml_sub(ctx, x, mCT));
    ggml_tensor* s1C = ggml_sqrt(
        ctx, ggml_scale(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, d2))), 1.0f / T)); // [1, C]
    ggml_tensor* sCT = ggml_repeat(ctx, ggml_cont(ctx, ggml_transpose(ctx, s1C)), x);                 // [C, T]
    // [x, mean, std] → TDNN → tanh → k=1-conv → softmax over T
    ggml_tensor* att = ggml_concat(ctx, ggml_concat(ctx, x, mCT, 0), sCT, 0);
    att = spk_tdnn_block(ctx, att, asp.tdnn, 1);
    att = ggml_tanh(ctx, att);
    auto cw = ggml_reshape_2d(ctx, asp.conv_w, asp.conv_w->ne[1], asp.conv_w->ne[2]);
    att = ggml_add(ctx, ggml_mul_mat(ctx, cw, att), asp.conv_b); // [C, T]
    att = ggml_cont(ctx, ggml_transpose(ctx, att));              // [T, C]
    att = ggml_soft_max(ctx, att);                               // softmax over T (ne[0])
    att = ggml_cont(ctx, ggml_transpose(ctx, att));              // [C, T]
    // Weighted mean and std → [2C, 1]
    ggml_tensor* wx = ggml_mul(ctx, att, x);
    ggml_tensor* wm =
        ggml_cont(ctx, ggml_transpose(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, wx))))); // [C,1]
    ggml_tensor* wmCT = ggml_repeat(ctx, wm, x);
    ggml_tensor* dd = ggml_sub(ctx, x, wmCT);
    ggml_tensor* ws = ggml_sqrt(
        ctx,
        ggml_cont(ctx,
                  ggml_transpose(
                      ctx, ggml_sum_rows(
                               ctx, ggml_cont(ctx, ggml_transpose(
                                                       ctx, ggml_mul(ctx, att, ggml_mul(ctx, dd, dd)))))))); // [C,1]
    return ggml_concat(ctx, wm, ws, 0);                                                                      // [2C, 1]
}

static ggml_cgraph* build_graph_spk_enc(qwen3_tts_context* c, int T_mel) {
    const auto& spk = c->spk_enc;
    ggml_init_params ip = {c->spk_compute_meta.size(), c->spk_compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 128, T_mel); // [128, T]
    ggml_set_name(h, "spk_mel");
    ggml_set_input(h);

    h = spk_tdnn_block(ctx0, h, spk.blk0, 1);
    ggml_set_name(h, "spk_blk0_out");
    ggml_set_output(h);

    static const int dilations[3] = {2, 3, 4};
    ggml_tensor* blk_outs[3];
    for (int i = 0; i < 3; i++) {
        h = spk_se_res2net(ctx0, h, spk.blk[i], dilations[i]);
        blk_outs[i] = h;
    }

    ggml_tensor* mfa_in = ggml_concat(ctx0, ggml_concat(ctx0, blk_outs[0], blk_outs[1], 0), blk_outs[2], 0);
    h = spk_tdnn_block(ctx0, mfa_in, spk.mfa, 1);
    ggml_set_name(h, "spk_mfa_out");
    ggml_set_output(h);
    h = spk_asp_block(ctx0, h, spk.asp); // [3072, 1]

    auto fcw = ggml_reshape_2d(ctx0, spk.fc_w, spk.fc_w->ne[1], spk.fc_w->ne[2]);
    h = ggml_add(ctx0, ggml_mul_mat(ctx0, fcw, h), spk.fc_b); // [1024, 1]
    h = ggml_reshape_1d(ctx0, h, 1024);
    ggml_set_name(h, "spk_emb");
    ggml_build_forward_expand(gf, h);
    ggml_free(ctx0);
    return gf;
}

static bool load_spk_enc(qwen3_tts_context* c) {
    auto& spk = c->spk_enc;
    char buf[256];
    auto req = [&](const char* n) { return require(c, n); };
    auto fmt = [&](const char* f, auto... a) -> std::string {
        snprintf(buf, sizeof(buf), f, a...);
        return buf;
    };

    spk.blk0.w = req("speaker.blocks.0.conv.weight");
    spk.blk0.b = req("speaker.blocks.0.conv.bias");
    if (!spk.blk0.w) {
        return false;
    }

    for (int i = 0; i < 3; i++) {
        int bi = i + 1;
        auto& blk = spk.blk[i];
        blk.tdnn1.w = req(fmt("speaker.blocks.%d.tdnn1.conv.weight", bi).c_str());
        blk.tdnn1.b = req(fmt("speaker.blocks.%d.tdnn1.conv.bias", bi).c_str());
        blk.tdnn2.w = req(fmt("speaker.blocks.%d.tdnn2.conv.weight", bi).c_str());
        blk.tdnn2.b = req(fmt("speaker.blocks.%d.tdnn2.conv.bias", bi).c_str());
        blk.se.c1w = req(fmt("speaker.blocks.%d.se_block.conv1.weight", bi).c_str());
        blk.se.c1b = req(fmt("speaker.blocks.%d.se_block.conv1.bias", bi).c_str());
        blk.se.c2w = req(fmt("speaker.blocks.%d.se_block.conv2.weight", bi).c_str());
        blk.se.c2b = req(fmt("speaker.blocks.%d.se_block.conv2.bias", bi).c_str());
        for (int j = 0; j < 7; j++) {
            blk.res2net.blocks[j].w = req(fmt("speaker.blocks.%d.res2net_block.blocks.%d.conv.weight", bi, j).c_str());
            blk.res2net.blocks[j].b = req(fmt("speaker.blocks.%d.res2net_block.blocks.%d.conv.bias", bi, j).c_str());
            if (!blk.res2net.blocks[j].w) {
                return false;
            }
        }
    }
    spk.mfa.w = req("speaker.mfa.conv.weight");
    spk.mfa.b = req("speaker.mfa.conv.bias");
    spk.asp.tdnn.w = req("speaker.asp.tdnn.conv.weight");
    spk.asp.tdnn.b = req("speaker.asp.tdnn.conv.bias");
    spk.asp.conv_w = req("speaker.asp.conv.weight");
    spk.asp.conv_b = req("speaker.asp.conv.bias");
    spk.fc_w = req("speaker.fc.weight");
    spk.fc_b = req("speaker.fc.bias");
    if (!spk.mfa.w || !spk.asp.tdnn.w || !spk.fc_w) {
        return false;
    }

    c->spk_compute_meta.resize(ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false));
    spk.loaded = true;
    if (c->params.verbosity >= 1) {
        fprintf(stderr, "qwen3_tts: speaker encoder loaded (ECAPA-TDNN 128→1024)\n");
    }
    return true;
}

// Run speaker encoder on mel (T, 128) row-major → (1024,) embedding.
static std::vector<float> run_spk_enc(qwen3_tts_context* c, const float* mel_TC, int T_mel) {
    // Convert mel (T, 128) row-major → ggml [C=128, T] flat layout.
    // ggml ne[0]=128 (C innermost), ne[1]=T → position of (c, t) is c + t*128.
    std::vector<float> mel_CT((size_t)128 * T_mel);
    for (int t = 0; t < T_mel; t++) {
        for (int c = 0; c < 128; c++) {
            mel_CT[(size_t)c + (size_t)t * 128] = mel_TC[(size_t)t * 128 + c];
        }
    }

    ggml_cgraph* gf = build_graph_spk_enc(c, T_mel);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        return {};
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "spk_mel"), mel_CT.data(), 0, mel_CT.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        return {};
    }
    std::vector<float> emb(1024);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "spk_emb"), emb.data(), 0, 1024 * sizeof(float));
    return emb;
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

static void build_embd_caches(qwen3_tts_context* c) {
    c->token_embd_cache.init(c->talker.token_embd_w);
    auto& cp = c->code_pred;
    c->codec_embd_cache.resize(cp.codec_embd.size());
    for (size_t i = 0; i < cp.codec_embd.size(); i++)
        c->codec_embd_cache[i].init(cp.codec_embd[i]);
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
            for (int i = 0; i < (int)c->vocab.id_to_token.size(); i++) {
                c->vocab.token_to_id[c->vocab.id_to_token[i]] = i;
            }
        }
        register_qwen_specials(c->vocab);

        auto merges = core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
        for (size_t i = 0; i < merges.size(); i++) {
            c->vocab.merge_rank[merges[i]] = (int32_t)i;
        }

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
    if (!c->backend) {
        c->backend = c->backend_cpu;
    }

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
    load_spk_enc(c);        // soft — logs a warning if tensors are missing but doesn't fail
    build_embd_caches(c);   // CPU copies for AR-loop embed lookups

    // Fuse Q+K+V weights for the talker so each layer's attention does a
    // single mul_mat instead of three. Only applies to F16/F32 talkers —
    // quantized formats (Q8_0/Q4_K) keep the 3-matmul path because their
    // block layout would need a converter-side fuse to be safe.
    // Set QWEN3_TTS_NO_FUSED_QKV=1 to skip (for A/B benchmarking).
    if (!env_bool("QWEN3_TTS_NO_FUSED_QKV")) {
        auto& blocks = c->talker.blocks;
        if (!blocks.empty() && blocks[0].attn_q_w && blocks[0].attn_k_w && blocks[0].attn_v_w &&
            (blocks[0].attn_q_w->type == GGML_TYPE_F32 || blocks[0].attn_q_w->type == GGML_TYPE_F16)) {
            const int q_out = (int)blocks[0].attn_q_w->ne[1];
            const int k_out = (int)blocks[0].attn_k_w->ne[1];
            const int hidden = (int)blocks[0].attn_q_w->ne[0];
            const int qkv_out = q_out + 2 * k_out;
            ggml_init_params fgp = {ggml_tensor_overhead() * blocks.size() + 256, nullptr, true};
            c->fused_ctx = ggml_init(fgp);
            if (c->fused_ctx) {
                for (auto& b : blocks) {
                    b.attn_qkv_w = ggml_new_tensor_2d(c->fused_ctx, b.attn_q_w->type, hidden, qkv_out);
                }
                c->fused_buf =
                    ggml_backend_alloc_ctx_tensors_from_buft(c->fused_ctx, ggml_backend_cpu_buffer_type());
                if (c->fused_buf) {
                    for (auto& b : blocks) {
                        const size_t qb = ggml_nbytes(b.attn_q_w);
                        const size_t kb = ggml_nbytes(b.attn_k_w);
                        std::vector<uint8_t> tmp(qb + 2 * kb);
                        ggml_backend_tensor_get(b.attn_q_w, tmp.data(), 0, qb);
                        ggml_backend_tensor_get(b.attn_k_w, tmp.data() + qb, 0, kb);
                        ggml_backend_tensor_get(b.attn_v_w, tmp.data() + qb + kb, 0, kb);
                        ggml_backend_tensor_set(b.attn_qkv_w, tmp.data(), 0, tmp.size());
                    }
                    if (params.verbosity >= 1) {
                        fprintf(stderr, "qwen3_tts: fused QKV for %zu talker layers (%d+%d+%d→%d)\n",
                                blocks.size(), q_out, k_out, k_out, qkv_out);
                    }
                } else {
                    for (auto& b : blocks) {
                        b.attn_qkv_w = nullptr;
                    }
                    ggml_free(c->fused_ctx);
                    c->fused_ctx = nullptr;
                }
            }
        }
    }

    // Scheduler
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = c->backend;
        if (c->backend_cpu && c->backend_cpu != c->backend) {
            backends[n_be++] = c->backend_cpu;
        }
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
        if (c->backend_cpu && c->backend_cpu != c->backend) {
            ggml_backend_t cp_backends[1] = {c->backend_cpu};
            c->cp_sched = ggml_backend_sched_new(cp_backends, nullptr, 1, 4096, false, false);
        }
    }
    c->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    const char* cp_be = env_str("QWEN3_TTS_CP_BACKEND");
    if (cp_be && std::strncmp(cp_be, "cpu", 3) == 0) {
        if (!copy_cp_weights_to_cpu(c, code_pred_cpu_copy_type_from_env(cp_be))) {
            fprintf(stderr, "qwen3_tts: code_pred CPU pin requested but copy failed; using main backend\n");
        }
    }

    // Eagerly reserve the code_pred scheduler with a clean (empty) scheduler
    // state so the memory layout is deterministic regardless of subsequent
    // lookup_rows calls.  Must happen after compute_meta is sized, cp_kv is
    // allocated, and code_pred weights are loaded.
    if (!cp_kv_alloc(c)) {
        fprintf(stderr, "qwen3_tts: cp_kv pre-alloc failed\n");
        qwen3_tts_free(c);
        return nullptr;
    }
    ggml_backend_sched_t cp_sched = code_pred_pick_sched(c);
    if (!code_pred_reserve_sched(c, cp_sched)) {
        fprintf(stderr, "qwen3_tts: code_pred sched reserve failed\n");
        qwen3_tts_free(c);
        return nullptr;
    }
    ggml_backend_sched_reset(cp_sched);

    // O15: allocate the T=1 lm_head slot on the same backend as the model
    // weights so we can blit different lm_head[i] into it before each step,
    // enabling graph reuse without reset/alloc for steps i=2..14.
    if (!c->code_pred.lm_head.empty() && c->code_pred.lm_head[0] && !c->cp_cpu_pinned) {
        ggml_tensor* proto = c->code_pred.lm_head[0];
        const size_t slot_bytes = ggml_nbytes(proto);
        c->cp_lm_slot_buf = ggml_backend_alloc_buffer(c->backend, slot_bytes);
        if (c->cp_lm_slot_buf) {
            ggml_init_params sp = {ggml_tensor_overhead(), nullptr, /*no_alloc=*/true};
            c->cp_lm_slot_ctx = ggml_init(sp);
            c->cp_lm_head_slot = ggml_new_tensor_2d(c->cp_lm_slot_ctx, proto->type,
                                                     proto->ne[0], proto->ne[1]);
            void* base = ggml_backend_buffer_get_base(c->cp_lm_slot_buf);
            ggml_backend_tensor_alloc(c->cp_lm_slot_buf, c->cp_lm_head_slot, base);
        }
    }

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
    if (!ctx || !path) {
        return -1;
    }
    ctx->codec_path = path;
    if (!load_codec(ctx, path)) {
        return -1;
    }
    return 0;
}

extern "C" int qwen3_tts_set_voice_prompt(struct qwen3_tts_context* ctx, const char* wav_path) {
    if (!ctx) {
        return -1;
    }
    ctx->voice_prompt_path = wav_path ? wav_path : "";
    if (!wav_path || !*wav_path) {
        return 0;
    }
    if (!ctx->spk_enc.loaded) {
        fprintf(stderr, "qwen3_tts: set_voice_prompt: speaker encoder not loaded\n");
        return -1;
    }

    // Read WAV file: expect 24 kHz mono 16-bit or 32-bit float PCM.
    FILE* f = std::fopen(wav_path, "rb");
    if (!f) {
        fprintf(stderr, "qwen3_tts: cannot open wav '%s'\n", wav_path);
        return -1;
    }
    // RIFF header
    char riff[4];
    std::fread(riff, 1, 4, f);
    if (riff[0] != 'R' || riff[1] != 'I' || riff[2] != 'F' || riff[3] != 'F') {
        std::fclose(f);
        fprintf(stderr, "qwen3_tts: not a RIFF WAV\n");
        return -1;
    }
    std::fseek(f, 4, SEEK_CUR); // chunk size
    char wave[4];
    std::fread(wave, 1, 4, f);
    if (wave[0] != 'W' || wave[1] != 'A' || wave[2] != 'V' || wave[3] != 'E') {
        std::fclose(f);
        return -1;
    }

    int sr = 0, channels = 0, bits = 0;
    bool is_float = false;
    bool found_data = false;
    std::vector<float> samples;

    while (!found_data) {
        char id[4];
        uint32_t sz;
        if (std::fread(id, 1, 4, f) != 4 || std::fread(&sz, 4, 1, f) != 1) {
            break;
        }
        if (id[0] == 'f' && id[1] == 'm' && id[2] == 't' && id[3] == ' ') {
            uint16_t fmt, ch;
            uint32_t srate;
            uint32_t brate;
            uint16_t bal, bps;
            std::fread(&fmt, 2, 1, f);
            std::fread(&ch, 2, 1, f);
            std::fread(&srate, 4, 1, f);
            std::fread(&brate, 4, 1, f);
            std::fread(&bal, 2, 1, f);
            std::fread(&bps, 2, 1, f);
            if (sz > 16) {
                std::fseek(f, sz - 16, SEEK_CUR);
            }
            sr = (int)srate;
            channels = (int)ch;
            bits = (int)bps;
            is_float = (fmt == 3); // IEEE float
        } else if (id[0] == 'd' && id[1] == 'a' && id[2] == 't' && id[3] == 'a') {
            if (sr != 24000) {
                std::fclose(f);
                fprintf(stderr, "qwen3_tts: voice prompt must be 24kHz, got %d Hz\n", sr);
                return -1;
            }
            const int n_frames = (int)(sz / (channels * (bits / 8)));
            samples.reserve((size_t)n_frames);
            if (bits == 16 && !is_float) {
                std::vector<int16_t> raw((size_t)n_frames * channels);
                std::fread(raw.data(), 2, raw.size(), f);
                for (int i = 0; i < n_frames; i++) {
                    float s = 0.0f;
                    for (int c = 0; c < channels; c++) {
                        s += raw[(size_t)i * channels + c] / 32768.0f;
                    }
                    samples.push_back(s / channels);
                }
            } else if (bits == 32 && is_float) {
                std::vector<float> raw((size_t)n_frames * channels);
                std::fread(raw.data(), 4, raw.size(), f);
                for (int i = 0; i < n_frames; i++) {
                    float s = 0.0f;
                    for (int c = 0; c < channels; c++) {
                        s += raw[(size_t)i * channels + c];
                    }
                    samples.push_back(s / channels);
                }
            } else {
                std::fclose(f);
                fprintf(stderr, "qwen3_tts: unsupported WAV format: %d bits, float=%d\n", bits, (int)is_float);
                return -1;
            }
            found_data = true;
        } else {
            std::fseek(f, sz, SEEK_CUR);
        }
    }
    std::fclose(f);
    if (samples.empty()) {
        fprintf(stderr, "qwen3_tts: no audio in wav\n");
        return -1;
    }

    // Compute mel + speaker embedding
    int T_mel = 0;
    auto mel = compute_spk_mel(samples.data(), (int)samples.size(), &T_mel);
    if (mel.empty()) {
        fprintf(stderr, "qwen3_tts: mel computation failed\n");
        return -1;
    }

    auto emb = run_spk_enc(ctx, mel.data(), T_mel);
    if (emb.empty()) {
        fprintf(stderr, "qwen3_tts: ECAPA forward failed\n");
        return -1;
    }

    ctx->runtime_spk_emb = std::move(emb);

    // Codec encode: produce ref_codes for ICL prefill
    if (ctx->cenc.loaded) {
        int T_frames = 0;
        if (!run_cenc(ctx, samples.data(), (int)samples.size(), ctx->runtime_ref_codes, T_frames)) {
            fprintf(stderr, "qwen3_tts: voice prompt codec encode failed\n");
            return -1;
        }
        if (ctx->params.verbosity >= 1) {
            fprintf(stderr, "qwen3_tts: voice prompt encoded: %d codec frames\n", T_frames);
        }
    } else {
        fprintf(stderr, "qwen3_tts: codec encoder not loaded — ref_codes not available\n");
        return -1;
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "qwen3_tts: voice prompt set from '%s'  (%d samples, T_mel=%d)\n", wav_path,
                (int)samples.size(), T_mel);
    }
    return 0;
}

extern "C" int qwen3_tts_set_voice_prompt_with_text(struct qwen3_tts_context* ctx, const char* wav_path,
                                                    const char* ref_text) {
    if (!ctx) {
        return -1;
    }
    if (ref_text) {
        ctx->runtime_ref_text = ref_text;
    }
    return qwen3_tts_set_voice_prompt(ctx, wav_path);
}

// Debug: get the runtime ref codes (for diff harness / comparison).
// Returns pointer to internal int32 buffer of *out_n elements (T*16 codes
// in [T, 16] row-major layout). Do NOT free — the buffer is owned by ctx.
extern "C" const int32_t* qwen3_tts_get_runtime_ref_codes(struct qwen3_tts_context* ctx, int* out_n) {
    if (out_n) {
        *out_n = 0;
    }
    if (!ctx || ctx->runtime_ref_codes.empty()) {
        return nullptr;
    }
    if (out_n) {
        *out_n = (int)ctx->runtime_ref_codes.size();
    }
    return ctx->runtime_ref_codes.data();
}

extern "C" const float* qwen3_tts_get_runtime_spk_emb(struct qwen3_tts_context* ctx, int* out_n) {
    if (out_n) {
        *out_n = 0;
    }
    if (!ctx || ctx->runtime_spk_emb.empty()) {
        return nullptr;
    }
    if (out_n) {
        *out_n = (int)ctx->runtime_spk_emb.size();
    }
    return ctx->runtime_spk_emb.data();
}

// Run the codec encoder graph and extract a named intermediate tensor.
extern "C" float* qwen3_tts_cenc_extract_stage(struct qwen3_tts_context* ctx, const float* audio, int n_samples,
                                               const char* stage_name, int* out_n) {
    if (out_n) {
        *out_n = 0;
    }
    if (!ctx || !audio || n_samples <= 0 || !stage_name) {
        return nullptr;
    }
    if (!ctx->cenc.loaded) {
        fprintf(stderr, "qwen3_tts: cenc_extract_stage: encoder not loaded\n");
        return nullptr;
    }
    if (std::strcmp(stage_name, "cenc_codes") == 0) {
        std::vector<int32_t> codes_i32;
        int T_frames = 0;
        if (!run_cenc(ctx, audio, n_samples, codes_i32, T_frames)) {
            fprintf(stderr, "qwen3_tts: cenc_extract_stage: run_cenc failed for cenc_codes\n");
            return nullptr;
        }
        float* out = (float*)malloc(codes_i32.size() * sizeof(float));
        if (!out) {
            return nullptr;
        }

        for (size_t i = 0; i < codes_i32.size(); ++i) {
            out[i] = (float)codes_i32[i];
        }

        if (out_n) {
            *out_n = (int)codes_i32.size();
        }
        return out;
    }
    ggml_cgraph* gf = build_cenc_graph(ctx, n_samples);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pcm_input"), audio, 0, (size_t)n_samples * sizeof(float));
    {
        const int T_enc = n_samples / 960;
        ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "cenc_mask");
        if (mask_t && T_enc > 1) {
            auto mask = build_cenc_mask(T_enc);
            ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        return nullptr;
    }
    ggml_tensor* t = ggml_graph_get_tensor(gf, stage_name);
    if (!t) {
        fprintf(stderr, "qwen3_tts: cenc_extract_stage: '%s' not in graph\n", stage_name);
        return nullptr;
    }
    const int n = (int)ggml_nelements(t);
    float* buf = (float*)malloc((size_t)n * sizeof(float));
    if (!buf) {
        return nullptr;
    }
    ggml_backend_tensor_get(t, buf, 0, (size_t)n * sizeof(float));
    if (out_n) {
        *out_n = n;
    }
    return buf;
}

extern "C" int qwen3_tts_load_voice_pack(struct qwen3_tts_context* ctx, const char* path) {
    if (!ctx || !path) {
        return -1;
    }

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

    if (ctx->vp_buf_w) {
        ggml_backend_buffer_free(ctx->vp_buf_w);
    }
    if (ctx->vp_ctx_w) {
        ggml_free(ctx->vp_ctx_w);
    }
    ctx->vp_tensors.clear();

    ctx->vp_tensors = std::move(wl.tensors);
    ctx->vp_ctx_w = wl.ctx;
    ctx->vp_buf_w = wl.buf;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "qwen3_tts: loaded voice pack %s with %zu voices\n", path, ctx->vp_names.size());
        for (size_t i = 0; i < ctx->vp_names.size(); i++) {
            fprintf(stderr, "  [%zu] %s\n", i, ctx->vp_names[i].c_str());
        }
    }
    if (ctx->vp_active < 0 && !ctx->vp_names.empty()) {
        ctx->vp_active = 0; // auto-select first voice
    }
    return 0;
}

extern "C" int qwen3_tts_select_voice(struct qwen3_tts_context* ctx, const char* name) {
    if (!ctx || !name) {
        return -1;
    }
    if (ctx->vp_names.empty()) {
        return -1;
    }
    for (size_t i = 0; i < ctx->vp_names.size(); i++) {
        if (ctx->vp_names[i] == name) {
            ctx->vp_active = (int)i;
            if (ctx->params.verbosity >= 1) {
                fprintf(stderr, "qwen3_tts: selected voice [%zu] '%s'\n", i, name);
            }
            return 0;
        }
    }
    return -2;
}

extern "C" int qwen3_tts_set_language(struct qwen3_tts_context* ctx, int codec_language_id) {
    if (!ctx) {
        return -1;
    }
    ctx->language_id = codec_language_id;
    return 0;
}

extern "C" float* qwen3_tts_run_text_proj(struct qwen3_tts_context* ctx, const int32_t* ids, int n_tokens, int* out_T,
                                          int* out_d) {
    if (out_T) {
        *out_T = 0;
    }
    if (out_d) {
        *out_d = 0;
    }
    if (!ctx || !ids || n_tokens <= 0) {
        return nullptr;
    }
    float* r = run_embed_text(ctx, ids, n_tokens);
    if (!r) {
        return nullptr;
    }
    if (out_T) {
        *out_T = n_tokens;
    }
    if (out_d) {
        *out_d = (int)ctx->hp.d_model;
    }
    return r;
}

extern "C" float* qwen3_tts_build_icl_prefill(struct qwen3_tts_context* ctx, const char* syn_text, const char* ref_text,
                                              int* out_T) {
    if (out_T) {
        *out_T = 0;
    }
    if (!ctx || !syn_text || !ref_text) {
        return nullptr;
    }
    std::vector<float> embeds, trailing;
    int T = 0, M = 0;
    if (!build_icl_prefill_embeds(ctx, syn_text, ref_text, embeds, T, trailing, M)) {
        return nullptr;
    }
    const int d = (int)ctx->hp.d_model;
    float* r = (float*)malloc((size_t)T * d * sizeof(float));
    std::memcpy(r, embeds.data(), embeds.size() * sizeof(float));
    if (out_T) {
        *out_T = T;
    }
    return r;
}

extern "C" float* qwen3_tts_run_talker_with_embeds(struct qwen3_tts_context* ctx, const float* embeds, int n_tokens,
                                                   int* out_vocab) {
    if (out_vocab) {
        *out_vocab = 0;
    }
    if (!ctx || !embeds || n_tokens <= 0) {
        return nullptr;
    }
    // Guarantee a clean cache: this is a one-shot diff entry point, so
    // n_past=0 always.
    float* logits = run_talker_kv(ctx, embeds, n_tokens, /*n_past=*/0, /*out_hidden_d=*/nullptr);
    if (!logits) {
        return nullptr;
    }
    if (out_vocab) {
        *out_vocab = (int)ctx->hp.vocab_size;
    }
    return logits;
}

extern "C" int32_t* qwen3_tts_synthesize_codes(struct qwen3_tts_context* ctx, const char* text, int* out_n_codes) {
    if (out_n_codes) {
        *out_n_codes = 0;
    }
    if (!ctx || !text) {
        return nullptr;
    }
    if (ctx->vocab.id_to_token.empty()) {
        fprintf(stderr, "qwen3_tts: vocab empty — re-convert with the updated converter\n");
        return nullptr;
    }
    // Need either a voice pack OR a runtime voice prompt (set_voice_prompt).
    if (ctx->vp_active < 0 && ctx->runtime_ref_codes.empty()) {
        fprintf(stderr, "qwen3_tts: no voice — call qwen3_tts_load_voice_pack or qwen3_tts_set_voice_prompt\n");
        return nullptr;
    }

    const bool bench = env_bool("QWEN3_TTS_BENCH");
    const bool dbg = env_bool("QWEN3_TTS_DEBUG");
    const char* dump_dir = env_str("QWEN3_TTS_DUMP_DIR");
    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;
    const int n_groups = (int)hp.n_code_groups; // 16
    int max_frames = ctx->params.max_codec_steps > 0 ? ctx->params.max_codec_steps : 1500;
    if (const char* mf = getenv("QWEN3_TTS_MAX_FRAMES")) {
        const int v = std::atoi(mf);
        if (v > 0) max_frames = v;
    }
    const int eos = (int)hp.codec_eos_id;

    // PRNG seed — env-overridable for reproducibility. Default seed
    // matches PyTorch's behaviour with `torch.manual_seed(42)`.
    uint64_t rng = 42;
    if (const char* s = env_str("QWEN3_TTS_SEED")) {
        rng = (uint64_t)std::strtoull(s, nullptr, 10);
    }

    // ---- ref_text: runtime voice prompt overrides voice pack ----
    std::string ref_text;
    if (!ctx->runtime_ref_text.empty()) {
        ref_text = ctx->runtime_ref_text;
    } else if (ctx->vp_active >= 0) {
        ref_text = ctx->vp_ref_texts[ctx->vp_active];
    }
    if (ref_text.empty()) {
        fprintf(stderr,
                "qwen3_tts: no ref_text — call qwen3_tts_set_voice_prompt with ref_text or load a voice pack\n");
        return nullptr;
    }

    // ---- ICL prefill builder (matches PyTorch's generate_icl_prompt) ----
    double t0 = bench ? now_ms() : 0.0;
    std::vector<float> prefill, trailing;
    int T_pre = 0, M_trail = 0;
    if (!build_icl_prefill_embeds(ctx, text, ref_text, prefill, T_pre, trailing, M_trail)) {
        return nullptr;
    }
    if (bench) {
        fprintf(stderr, "qwen3_tts: icl_prefill %7.1f ms (T=%d)\n", now_ms() - t0, T_pre);
    }
    if (dump_dir) {
        dump_f32(dump_dir, "icl_prefill", prefill.data(), prefill.size());
    }

    // ---- talker prefill: get logits + hidden_last ----
    double t1 = bench ? now_ms() : 0.0;
    float* past_hidden = nullptr;
    float* logits = run_talker_kv(ctx, prefill.data(), T_pre, /*n_past=*/0, &past_hidden);
    if (!logits || !past_hidden) {
        free(logits);
        free(past_hidden);
        return nullptr;
    }
    if (bench) {
        fprintf(stderr, "qwen3_tts: talker_pre %7.1f ms\n", now_ms() - t1);
    }
    if (dump_dir) {
        dump_f32(dump_dir, "talker_prefill_logits", logits, hp.vocab_size);
    }
    if (dbg) {
        const char* cp_be = env_str("QWEN3_TTS_CP_BACKEND");
        fprintf(stderr, "qwen3_tts: code_pred backend=%s\n", (cp_be && *cp_be) ? cp_be : "main");
    }

    int n_past = T_pre;

    const bool embd_cache_enabled = !env_bool("QWEN3_TTS_NO_EMBD_CACHE");

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
    double t_loop_lookup = 0.0;
    double t_loop_code_pred = 0.0;
    double t_loop_next_emb = 0.0;
    double t_loop_talker = 0.0;
    int frame = 0;
    // Mirror PyTorch generate(): top-k=50, temperature=0.9 sampling for
    // cb0, with the official suppress_tokens mask (last 1024 logits
    // except codec_eos_id are -inf — those are pad/silence/dialect
    // semantic ids that produce silence frames if argmax lands on them)
    // and min_new_tokens=2 (no codec_eos for the first two frames).
    // Without these the talker can argmax-attract to a silence token at
    // frame 0 and emit ~5 s of leading silence.
    const int talker_top_k = 50;
    const float talker_temp = 0.9f;
    const float talker_repetition_penalty = 1.05f;
    const int min_new_frames = 2;
    const int suppress_lo = (int)hp.vocab_size - 1024; // 2048 with default config
    std::vector<int32_t> talker_history;
    talker_history.reserve((size_t)max_frames);
    // Pre-allocated scratch buffers for the per-frame embed lookups.
    std::vector<float> last_id_hidden_buf(d);
    std::vector<float> next_emb_row_buf(d);
    std::vector<float> next_emb(d, 0.0f);
    for (frame = 0; frame < max_frames; frame++) {
        apply_repetition_penalty(logits, (int)hp.vocab_size, talker_history, talker_repetition_penalty);
        // 1. Sample codebook-0 from talker logits.
        for (int i = suppress_lo; i < (int)hp.vocab_size; i++) {
            if (i != eos) {
                logits[i] = -INFINITY;
            }
        }
        if (frame < min_new_frames) {
            logits[eos] = -INFINITY;
        }
        int cb0 = top_k_sample(logits, (int)hp.vocab_size, talker_top_k, talker_temp, &rng);
        if (getenv("QWEN3_TTS_EMBD_CHECK"))
            fprintf(stderr, "qwen3_tts: frame=%d cb0=%d rng=%llu\n", frame, cb0, (unsigned long long)rng);
        free(logits);
        logits = nullptr;
        if (cb0 == eos) {
            if (dbg) {
                fprintf(stderr, "qwen3_tts: codec_eos at frame %d\n", frame);
            }
            free(past_hidden);
            past_hidden = nullptr;
            break;
        }
        talker_history.push_back(cb0);
        // 2. Embed cb0 via talker.codec_embedding → last_id_hidden (d,)
        const double t_lookup0 = bench ? now_ms() : 0.0;
        if (embd_cache_enabled && ctx->token_embd_cache) {
            if (!ctx->token_embd_cache.get_row_into(cb0, last_id_hidden_buf.data())) {
                free(past_hidden);
                return nullptr;
            }
        } else {
            float* tmp = lookup_rows(ctx, ctx->talker.token_embd_w, &cb0, 1);
            if (!tmp) {
                free(past_hidden);
                return nullptr;
            }
            std::memcpy(last_id_hidden_buf.data(), tmp, (size_t)d * sizeof(float));
            free(tmp);
        }
        if (bench) {
            t_loop_lookup += now_ms() - t_lookup0;
        }
        // 3. Code predictor AR loop → 15 more codebook ids (sampled).
        int32_t cb1_15[15];
        const double t_cp = bench ? now_ms() : 0.0;
        if (!code_pred_generate_15(ctx, past_hidden, last_id_hidden_buf.data(), cb1_15, &rng, frame)) {
            free(past_hidden);
            return nullptr;
        }
        if (bench) {
            t_loop_code_pred += now_ms() - t_cp;
        }
        free(past_hidden);
        past_hidden = nullptr;

        // 4. Append the full 16-codebook frame.
        all_codes.push_back(cb0);
        for (int i = 0; i < 15; i++) {
            all_codes.push_back(cb1_15[i]);
        }

        // 5. Build next talker input:
        //    sum_{cb=0..15}(codec_embd_for_cb(frame[cb])) + trailing[step]
        //    where trailing[step] = trailing_text_hidden[gen_step] if gen_step
        //    < M else tts_pad_embed (only the latter when codec_lens > text_lens).
        std::fill(next_emb.data(), next_emb.data() + d, 0.0f);
        const double t_next = bench ? now_ms() : 0.0;
        {
            for (int cb = 0; cb < n_groups; cb++) {
                int32_t code = (cb == 0) ? cb0 : cb1_15[cb - 1];
                bool ok = false;
                if (embd_cache_enabled && cb == 0 && ctx->token_embd_cache) {
                    ok = ctx->token_embd_cache.get_row_into(code, next_emb_row_buf.data());
                } else if (embd_cache_enabled && cb > 0 && cb - 1 < (int)ctx->codec_embd_cache.size() &&
                           ctx->codec_embd_cache[cb - 1]) {
                    ok = ctx->codec_embd_cache[cb - 1].get_row_into(code, next_emb_row_buf.data());
                } else {
                    ggml_tensor* w = (cb == 0) ? ctx->talker.token_embd_w : ctx->code_pred.codec_embd[cb - 1];
                    float* row = lookup_rows(ctx, w, &code, 1);
                    if (row) {
                        std::memcpy(next_emb_row_buf.data(), row, (size_t)d * sizeof(float));
                        free(row);
                        ok = true;
                    }
                }
                if (!ok) {
                    return nullptr;
                }
                for (int j = 0; j < d; j++) {
                    next_emb[j] += next_emb_row_buf[j];
                }
            }
        }

        // Add trailing_text_hidden[gen_step] (or last row if past M).
        const int trail_idx = std::min(frame, M_trail - 1);
        const float* trail = trailing.data() + (size_t)trail_idx * d;
        for (int j = 0; j < d; j++) {
            next_emb[j] += trail[j];
        }
        if (bench) {
            t_loop_next_emb += now_ms() - t_next;
        }

        if (getenv("QWEN3_TTS_EMBD_CHECK") && frame < 1) {
            fprintf(stderr, "qwen3_tts: frame=%d codes: cb0=%d cb1..4=%d %d %d %d\n", frame, cb0, cb1_15[0], cb1_15[1],
                    cb1_15[2], cb1_15[3]);
            float s = 0.0f;
            for (int j = 0; j < d; j++)
                s += std::abs(next_emb[j]);
            fprintf(stderr, "qwen3_tts: frame=%d next_emb[0..3]=%.6f %.6f %.6f %.6f  l1=%.4f\n", frame, next_emb[0],
                    next_emb[1], next_emb[2], next_emb[3], s);
        }

        // 6. Talker forward on the (1, d) input → next logits + hidden_last.
        if (n_past >= ctx->kv_max_ctx - 1) {
            fprintf(stderr, "qwen3_tts: talker kv cache full at frame %d (n_past=%d)\n", frame, n_past);
            break;
        }
        const double t_talker = bench ? now_ms() : 0.0;
        logits = run_talker_kv(ctx, next_emb.data(), 1, n_past, &past_hidden);
        if (bench) {
            t_loop_talker += now_ms() - t_talker;
        }
        if (!logits || !past_hidden) {
            free(logits);
            free(past_hidden);
            return nullptr;
        }
        n_past += 1;
    }
    free(logits);
    free(past_hidden);

    if (bench) {
        fprintf(stderr, "qwen3_tts: ar_loop    %7.1f ms (%d frames, %.1f ms/frame)\n", now_ms() - t_loop, frame,
                frame > 0 ? (now_ms() - t_loop) / frame : 0.0);
    }
    if (bench) {
        fprintf(stderr,
                "qwen3_tts: ar_breakdown lookup=%7.1f ms  code_pred=%7.1f ms  next_emb=%7.1f ms  talker=%7.1f ms\n",
                t_loop_lookup, t_loop_code_pred, t_loop_next_emb, t_loop_talker);
    }
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "qwen3_tts: produced %d frames × 16 codebooks = %zu codes\n", frame, all_codes.size());
    }
    if (dump_dir && !all_codes.empty()) {
        dump_i32(dump_dir, "generated_codes", all_codes.data(), all_codes.size());
    }

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
    if (out_n_samples) {
        *out_n_samples = 0;
    }
    if (!ctx || !codes || n_codes <= 0) {
        return nullptr;
    }
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

static bool qwen3_tts_get_active_ref_codes(qwen3_tts_context* ctx, std::vector<int32_t>& ref_codes, int& T_ref) {
    T_ref = 0;
    ref_codes.clear();

    const int n_q = (int)ctx->codec.hp.n_q;

    if (!ctx->runtime_ref_codes.empty()) {
        if ((int)ctx->runtime_ref_codes.size() % n_q != 0) {
            fprintf(stderr, "qwen3_tts: runtime_ref_codes size %zu not divisible by n_q=%d\n",
                    ctx->runtime_ref_codes.size(), n_q);
            return false;
        }
        ref_codes = ctx->runtime_ref_codes;
        T_ref = (int)ref_codes.size() / n_q;
        return true;
    }

    if (ctx->vp_active < 0) {
        return true;
    }

    const std::string& voice_name = ctx->vp_names[ctx->vp_active];
    auto it = ctx->vp_tensors.find("voicepack.code." + voice_name + ".codes");
    if (it == ctx->vp_tensors.end() || !it->second) {
        fprintf(stderr, "qwen3_tts: active voice '%s' missing ref_code tensor\n", voice_name.c_str());
        return false;
    }

    ggml_tensor* code_t = it->second;
    if ((int)code_t->ne[0] != n_q) {
        fprintf(stderr, "qwen3_tts: voicepack ref_code groups mismatch: %d vs %d\n", (int)code_t->ne[0], n_q);
        return false;
    }

    T_ref = (int)code_t->ne[1];
    ref_codes.resize((size_t)T_ref * n_q);
    ggml_backend_tensor_get(code_t, ref_codes.data(), 0, ref_codes.size() * sizeof(int32_t));
    return true;
}

extern "C" float* qwen3_tts_codec_extract_stage(struct qwen3_tts_context* ctx, const int32_t* codes, int n_codes,
                                                const char* stage_name, int* out_n) {
    if (out_n) {
        *out_n = 0;
    }
    if (!ctx || !codes || n_codes <= 0 || !stage_name) {
        return nullptr;
    }
    if (!ctx->codec.loaded) {
        fprintf(stderr, "qwen3_tts: codec_extract_stage() requires codec.\n");
        return nullptr;
    }
    const int n_q = (int)ctx->codec.hp.n_q;
    if (n_codes % n_q != 0) {
        return nullptr;
    }
    const int T_codec = n_codes / n_q;
    return codec_extract_stage(ctx, codes, T_codec, stage_name, out_n);
}

extern "C" float* qwen3_tts_synthesize(struct qwen3_tts_context* ctx, const char* text, int* out_n_samples) {
    if (out_n_samples) {
        *out_n_samples = 0;
    }
    if (!ctx) {
        return nullptr;
    }
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

    std::vector<int32_t> ref_codes;
    int T_ref = 0;
    if (!qwen3_tts_get_active_ref_codes(ctx, ref_codes, T_ref)) {
        free(codes);
        return nullptr;
    }

    const int T_gen = n_codes / n_q;
    float* pcm = nullptr;

    if (!ref_codes.empty()) {
        std::vector<int32_t> codes_for_decode;
        codes_for_decode.reserve(ref_codes.size() + (size_t)n_codes);
        codes_for_decode.insert(codes_for_decode.end(), ref_codes.begin(), ref_codes.end());
        codes_for_decode.insert(codes_for_decode.end(), codes, codes + n_codes);

        int n_pcm_full = 0;
        pcm = codec_decode_codes(ctx, codes_for_decode.data(), T_ref + T_gen, &n_pcm_full);
        if (!pcm || n_pcm_full <= 0) {
            free(codes);
            free(pcm);
            return nullptr;
        }

        const int cut = (int)((double)T_ref / (double)std::max(T_ref + T_gen, 1) * n_pcm_full);
        const int n_pcm_trim = std::max(0, n_pcm_full - cut);
        float* trimmed = (float*)malloc((size_t)n_pcm_trim * sizeof(float));
        if (!trimmed) {
            free(codes);
            free(pcm);
            return nullptr;
        }
        if (n_pcm_trim > 0) {
            std::memcpy(trimmed, pcm + cut, (size_t)n_pcm_trim * sizeof(float));
        }
        free(pcm);
        pcm = trimmed;
        if (out_n_samples) {
            *out_n_samples = n_pcm_trim;
        }
    } else {
        pcm = codec_decode_codes(ctx, codes, T_gen, out_n_samples);
    }

    free(codes);
    return pcm;
}

extern "C" void qwen3_tts_pcm_free(float* pcm) {
    free(pcm);
}

extern "C" void qwen3_tts_free(struct qwen3_tts_context* ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->codec_sched) {
        ggml_backend_sched_free(ctx->codec_sched);
    }
    if (ctx->cp_sched) {
        ggml_backend_sched_free(ctx->cp_sched);
    }
    if (ctx->sched) {
        ggml_backend_sched_free(ctx->sched);
    }
    if (ctx->kv_buf) {
        ggml_backend_buffer_free(ctx->kv_buf);
    }
    if (ctx->kv_ctx) {
        ggml_free(ctx->kv_ctx);
    }
    if (ctx->cp_lm_slot_buf) {
        ggml_backend_buffer_free(ctx->cp_lm_slot_buf);
    }
    if (ctx->cp_lm_slot_ctx) {
        ggml_free(ctx->cp_lm_slot_ctx);
    }
    if (ctx->fused_buf) {
        ggml_backend_buffer_free(ctx->fused_buf);
    }
    if (ctx->fused_ctx) {
        ggml_free(ctx->fused_ctx);
    }
    if (ctx->cp_kv_buf) {
        ggml_backend_buffer_free(ctx->cp_kv_buf);
    }
    if (ctx->cp_kv_ctx) {
        ggml_free(ctx->cp_kv_ctx);
    }
    if (ctx->cp_cpu_buf) {
        ggml_backend_buffer_free(ctx->cp_cpu_buf);
    }
    if (ctx->cp_cpu_ctx) {
        ggml_free(ctx->cp_cpu_ctx);
    }
    if (ctx->codec.buf_w) {
        ggml_backend_buffer_free(ctx->codec.buf_w);
    }
    if (ctx->codec.ctx_w) {
        ggml_free(ctx->codec.ctx_w);
    }
    if (ctx->vp_buf_w) {
        ggml_backend_buffer_free(ctx->vp_buf_w);
    }
    if (ctx->vp_ctx_w) {
        ggml_free(ctx->vp_ctx_w);
    }
    if (ctx->buf_w) {
        ggml_backend_buffer_free(ctx->buf_w);
    }
    if (ctx->ctx_w) {
        ggml_free(ctx->ctx_w);
    }
    if (ctx->backend && ctx->backend != ctx->backend_cpu) {
        ggml_backend_free(ctx->backend);
    }
    if (ctx->backend_cpu) {
        ggml_backend_free(ctx->backend_cpu);
    }
    delete ctx;
}

extern "C" void qwen3_tts_set_n_threads(struct qwen3_tts_context* ctx, int n_threads) {
    if (!ctx || n_threads <= 0) {
        return;
    }
    ctx->n_threads = n_threads;
    if (ctx->backend_cpu) {
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    }
}

extern "C" float* qwen3_tts_compute_speaker_mel(struct qwen3_tts_context* /*ctx*/, const float* audio, int n_samples,
                                                int* out_T_mel, int* out_n_mels) {
    if (out_T_mel) {
        *out_T_mel = 0;
    }
    if (out_n_mels) {
        *out_n_mels = 0;
    }
    if (!audio || n_samples <= 0) {
        return nullptr;
    }
    int T = 0;
    auto mel = compute_spk_mel(audio, n_samples, &T);
    if (mel.empty()) {
        return nullptr;
    }
    const int n_mels = 128;
    float* buf = (float*)malloc(mel.size() * sizeof(float));
    if (!buf) {
        return nullptr;
    }
    std::memcpy(buf, mel.data(), mel.size() * sizeof(float));
    if (out_T_mel) {
        *out_T_mel = T;
    }
    if (out_n_mels) {
        *out_n_mels = n_mels;
    }
    return buf;
}

extern "C" float* qwen3_tts_run_speaker_enc_on_mel(struct qwen3_tts_context* ctx, const float* mel, int T_mel,
                                                   int* out_dim) {
    if (out_dim) {
        *out_dim = 0;
    }
    if (!ctx || !mel || T_mel <= 0) {
        return nullptr;
    }
    if (!ctx->spk_enc.loaded) {
        return nullptr;
    }
    auto emb = run_spk_enc(ctx, mel, T_mel);
    if (emb.empty()) {
        return nullptr;
    }
    float* buf = (float*)malloc(1024 * sizeof(float));
    if (!buf) {
        return nullptr;
    }
    std::memcpy(buf, emb.data(), 1024 * sizeof(float));
    if (out_dim) {
        *out_dim = 1024;
    }
    return buf;
}

extern "C" float* qwen3_tts_compute_speaker_embedding(struct qwen3_tts_context* ctx, const float* audio, int n_samples,
                                                      int* out_dim) {
    if (out_dim) {
        *out_dim = 0;
    }
    if (!ctx || !audio || n_samples <= 0) {
        return nullptr;
    }
    if (!ctx->spk_enc.loaded) {
        fprintf(stderr, "qwen3_tts: compute_speaker_embedding: ECAPA not loaded\n");
        return nullptr;
    }
    int T_mel = 0;
    auto mel = compute_spk_mel(audio, n_samples, &T_mel);
    if (mel.empty()) {
        return nullptr;
    }
    auto emb = run_spk_enc(ctx, mel.data(), T_mel);
    if (emb.empty()) {
        return nullptr;
    }
    float* buf = (float*)malloc(1024 * sizeof(float));
    if (!buf) {
        return nullptr;
    }
    std::memcpy(buf, emb.data(), 1024 * sizeof(float));
    if (out_dim) {
        *out_dim = 1024;
    }
    return buf;
}
