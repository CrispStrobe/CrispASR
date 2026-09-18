// hojo_asr.cpp — HojoAI/Hojo-ASR-Multi-V1 ggml runtime (issue #438).
//
// See hojo_asr.h for the architecture summary and where each fact came from.
//
// Relationship to the sibling moss_transcribe.cpp: the audio encoder is the
// SAME stock Qwen3-Omni AuT tower, so the mel front end, the conv-stem graph,
// the block-diagonal windowed transformer and the Qwen3 KV-cache decode are
// structurally identical and were adapted from it. What is new here is the
// WeNet Conformer adapter (2 macaron blocks with rel-pos attention, a GLU +
// depthwise-conv module and a folded BatchNorm) and the conditioning: Hojo
// feeds the LM `[embed(<|im_start|>)] ++ speech` with no prompt at all, where
// MOSS scatters audio embeddings into a ChatML template.

#include "hojo_asr.h"
#include "core/win_compat.h"

#include "core/beam_decode.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "crispasr_imatrix.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/attention.h"
#include "core/bpe.h"
#include "core/crispasr_env.h"
#include "core/ffn.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/hojo_asr_frames.h"
#include "core/mel.h"
#include "core/ngram_loop_fix.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation — CRISPASR_HOJO_ASR_BENCH=1 for per-stage timings.
// ===========================================================================

static bool hojo_asr_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_HOJO_ASR_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct hojo_asr_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit hojo_asr_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~hojo_asr_bench_stage() {
        if (!hojo_asr_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  hojo_asr_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct hojo_asr_hparams {
    // Mel front end (WhisperFeatureExtractor, 128 bins, n_fft 400, hop 160).
    uint32_t n_mels = 128;
    uint32_t n_fft = 400;
    uint32_t hop_length = 160;
    uint32_t sample_rate = 16000;

    // Audio encoder (qwen3_omni_moe_audio_encoder with Hojo's window override).
    uint32_t enc_layers = 32;
    uint32_t enc_d_model = 1280;
    uint32_t enc_n_heads = 20;
    uint32_t enc_head_dim = 64; // 1280 / 20
    uint32_t enc_ffn_dim = 5120;
    uint32_t enc_ds_hidden = 480;
    uint32_t enc_max_pos = 1500;
    uint32_t enc_output_dim = 2048;
    uint32_t n_window = 1500;       // conv chunk = n_window * 2 = 3000 mel frames
    uint32_t n_window_infer = 3000; // attention block = the conv chunk itself
    float enc_ln_eps = 1e-5f;

    // WeNet Conformer adapter.
    uint32_t adp_blocks = 2;
    uint32_t adp_hidden = 2560;
    uint32_t adp_units = 640; // conformer FFN inner width, NOT a stacking factor
    uint32_t adp_heads = 4;
    uint32_t adp_head_dim = 640; // adp_hidden / adp_heads
    uint32_t adp_kernel = 15;
    float adp_ln_eps = 1e-5f;
    float adp_ff_scale = 0.5f;

    // LLM (Qwen3-4B-Instruct-2507).
    uint32_t llm_hidden = 2560;
    uint32_t llm_layers = 36;
    uint32_t llm_n_heads = 32;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 9728;
    uint32_t llm_vocab_size = 151670;
    uint32_t llm_max_pos = 262144;
    float llm_rope_theta = 5000000.0f;
    float llm_rms_eps = 1e-6f;

    // Decode recipe (config.yaml `generate:`).
    uint32_t bos_token_id = 151644; // <|im_start|>
    uint32_t eos_token_id = 151645; // <|im_end|>
    uint32_t gen_max_new_tokens = 200;
    uint32_t gen_num_beams = 4;
    float gen_repetition_penalty = 2.0f;
    float gen_length_penalty = 1.0f;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct hojo_asr_enc_block {
    ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_o_w = nullptr, *attn_o_b = nullptr;
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor *ffn_fc1_w = nullptr, *ffn_fc1_b = nullptr;
    ggml_tensor *ffn_fc2_w = nullptr, *ffn_fc2_b = nullptr;
};

struct hojo_asr_encoder {
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;
    ggml_tensor *conv3_w = nullptr, *conv3_b = nullptr;
    ggml_tensor* conv_out_w = nullptr; // Linear(7680 → 1280), NO bias
    ggml_tensor *ln_post_w = nullptr, *ln_post_b = nullptr;
    ggml_tensor *proj1_w = nullptr, *proj1_b = nullptr;
    ggml_tensor *proj2_w = nullptr, *proj2_b = nullptr;
    std::vector<hojo_asr_enc_block> blocks;
};

// One WeNet ConformerEncoderLayer: macaron FFN → rel-pos MHA → conv module →
// FFN → final LayerNorm, all pre-norm.
struct hojo_asr_adp_block {
    ggml_tensor *norm_ff_macaron_w = nullptr, *norm_ff_macaron_b = nullptr;
    ggml_tensor *ffm_w1_w = nullptr, *ffm_w1_b = nullptr;
    ggml_tensor *ffm_w2_w = nullptr, *ffm_w2_b = nullptr;

    ggml_tensor *norm_mha_w = nullptr, *norm_mha_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_o_w = nullptr, *attn_o_b = nullptr;
    ggml_tensor* attn_pos_w = nullptr; // linear_pos, no bias
    ggml_tensor* pos_bias_u = nullptr; // (head_dim, n_heads)
    ggml_tensor* pos_bias_v = nullptr;

    ggml_tensor *norm_conv_w = nullptr, *norm_conv_b = nullptr;
    ggml_tensor *conv_pw1_w = nullptr, *conv_pw1_b = nullptr;
    ggml_tensor *conv_dw_w = nullptr, *conv_dw_b = nullptr;
    ggml_tensor *conv_bn_scale = nullptr, *conv_bn_shift = nullptr; // folded BatchNorm1d
    ggml_tensor *conv_pw2_w = nullptr, *conv_pw2_b = nullptr;

    ggml_tensor *norm_ff_w = nullptr, *norm_ff_b = nullptr;
    ggml_tensor *ff_w1_w = nullptr, *ff_w1_b = nullptr;
    ggml_tensor *ff_w2_w = nullptr, *ff_w2_b = nullptr;

    ggml_tensor *norm_final_w = nullptr, *norm_final_b = nullptr;
};

struct hojo_asr_adapter {
    // LinearNoSubsampling: Linear(2048 → 2560) → LayerNorm → ×sqrt(2560)
    ggml_tensor *embed_proj_w = nullptr, *embed_proj_b = nullptr;
    ggml_tensor *embed_norm_w = nullptr, *embed_norm_b = nullptr;
    ggml_tensor* pe = nullptr; // (adp_hidden, max_len) — RelPositionalEncoding table
    std::vector<hojo_asr_adp_block> blocks;
    ggml_tensor *after_norm_w = nullptr, *after_norm_b = nullptr;
    ggml_tensor *ln_speech_w = nullptr, *ln_speech_b = nullptr;
};

struct hojo_asr_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr;
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct hojo_asr_llm {
    ggml_tensor* embed_w = nullptr;
    std::vector<hojo_asr_llm_block> blocks;
    ggml_tensor* final_norm_w = nullptr;
    ggml_tensor* lm_head_w = nullptr; // tied → == embed_w
};

struct hojo_asr_model {
    hojo_asr_hparams hparams;
    hojo_asr_encoder enc;
    hojo_asr_adapter adp;
    hojo_asr_llm llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Encoder sinusoidal position table (enc_max_pos × enc_d_model), CONCAT
    // halves (sin block then cos block) — the audio tower's convention. The
    // adapter's table (adp.pe) is INTERLEAVED and comes from the checkpoint.
    std::vector<float> audio_pe;

    // Mel front end, copied verbatim out of the checkpoint when the converter
    // baked it (HARD RULE 2: a shipped filterbank makes every
    // htk-vs-slaney / norm= / periodic-vs-symmetric question unaskable).
    std::vector<float> mel_filters; // (n_mels, n_freqs), MelsFreqs layout
    std::vector<float> mel_window;  // (n_fft,), periodic Hann
    bool mel_from_checkpoint = false;
};

struct hojo_asr_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

struct hojo_asr_context {
    hojo_asr_context_params params;
    hojo_asr_model model;
    hojo_asr_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;

    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;

    // Conv-stem graph, rebuilt per encoder invocation (see the note in
    // hojo_asr_run_encoder — caching it across invocations is a use-after-free).
    ggml_cgraph* conv_gf = nullptr;
    ggml_context* conv_ctx = nullptr;
    std::vector<uint8_t> conv_meta;
    int conv_win = 0; // mel-frame width the cached conv graph was built for

    int n_threads = 4;
    std::string model_path;
    int beam_size = 0;      // 0 = use hparams.gen_num_beams
    int max_new_tokens = 0; // 0 = use hparams.gen_max_new_tokens

    bool enc_use_flash = true;
};

// ===========================================================================
// Helpers
// ===========================================================================

static ggml_tensor* try_get(hojo_asr_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}
static ggml_tensor* require(hojo_asr_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "hojo_asr");
}

static bool backend_is_vulkan(ggml_backend_t b) {
    if (!b)
        return false;
    const char* name = ggml_backend_name(b);
    return name && std::strncmp(name, "Vulkan", 6) == 0;
}

// Frame arithmetic lives in core/hojo_asr_frames.h so it can be unit-tested
// without a model; it is also where Python's floor-division semantics for the
// `leave == 0` case (an exact multiple of n_window_infer, i.e. exactly 30 s of
// audio) are handled — C's truncating `/` gets that case one frame too long.
using core_hojo_frames::conv_out_len;
using core_hojo_frames::conv_stem_out_len;
using core_hojo_frames::feat_output_len;

// ===========================================================================
// Model loading
// ===========================================================================

static bool hojo_asr_load_model(hojo_asr_model& model, hojo_asr_vocab& vocab, const char* path,
                                ggml_backend_t backend) {
    // ---- pass 1: metadata + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.n_mels = core_gguf::kv_u32(gctx, "hojo_asr.enc.num_mel_bins", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "hojo_asr.mel.n_fft", hp.n_fft);
        hp.hop_length = core_gguf::kv_u32(gctx, "hojo_asr.mel.hop_length", hp.hop_length);
        hp.sample_rate = core_gguf::kv_u32(gctx, "hojo_asr.mel.sample_rate", hp.sample_rate);

        hp.enc_layers = core_gguf::kv_u32(gctx, "hojo_asr.enc.encoder_layers", hp.enc_layers);
        hp.enc_d_model = core_gguf::kv_u32(gctx, "hojo_asr.enc.d_model", hp.enc_d_model);
        hp.enc_n_heads = core_gguf::kv_u32(gctx, "hojo_asr.enc.encoder_attention_heads", hp.enc_n_heads);
        hp.enc_ffn_dim = core_gguf::kv_u32(gctx, "hojo_asr.enc.encoder_ffn_dim", hp.enc_ffn_dim);
        hp.enc_ds_hidden = core_gguf::kv_u32(gctx, "hojo_asr.enc.downsample_hidden_size", hp.enc_ds_hidden);
        hp.enc_max_pos = core_gguf::kv_u32(gctx, "hojo_asr.enc.max_source_positions", hp.enc_max_pos);
        hp.enc_output_dim = core_gguf::kv_u32(gctx, "hojo_asr.enc.output_dim", hp.enc_output_dim);
        hp.n_window = core_gguf::kv_u32(gctx, "hojo_asr.enc.n_window", hp.n_window);
        hp.n_window_infer = core_gguf::kv_u32(gctx, "hojo_asr.enc.n_window_infer", hp.n_window_infer);
        hp.enc_ln_eps = core_gguf::kv_f32(gctx, "hojo_asr.enc.layer_norm_eps", hp.enc_ln_eps);
        hp.enc_head_dim = hp.enc_d_model / hp.enc_n_heads;

        hp.adp_blocks = core_gguf::kv_u32(gctx, "hojo_asr.adapter.num_blocks", hp.adp_blocks);
        hp.adp_hidden = core_gguf::kv_u32(gctx, "hojo_asr.adapter.hidden_size", hp.adp_hidden);
        hp.adp_units = core_gguf::kv_u32(gctx, "hojo_asr.adapter.linear_units", hp.adp_units);
        hp.adp_heads = core_gguf::kv_u32(gctx, "hojo_asr.adapter.attention_heads", hp.adp_heads);
        hp.adp_kernel = core_gguf::kv_u32(gctx, "hojo_asr.adapter.cnn_module_kernel", hp.adp_kernel);
        hp.adp_ln_eps = core_gguf::kv_f32(gctx, "hojo_asr.adapter.layer_norm_eps", hp.adp_ln_eps);
        hp.adp_ff_scale = core_gguf::kv_f32(gctx, "hojo_asr.adapter.ff_scale", hp.adp_ff_scale);
        hp.adp_head_dim = hp.adp_heads ? hp.adp_hidden / hp.adp_heads : hp.adp_hidden;

        hp.llm_hidden = core_gguf::kv_u32(gctx, "hojo_asr.llm.hidden_size", hp.llm_hidden);
        hp.llm_layers = core_gguf::kv_u32(gctx, "hojo_asr.llm.num_layers", hp.llm_layers);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "hojo_asr.llm.num_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "hojo_asr.llm.num_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "hojo_asr.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "hojo_asr.llm.intermediate_size", hp.llm_ff_dim);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "hojo_asr.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "hojo_asr.llm.max_position_embeddings", hp.llm_max_pos);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "hojo_asr.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "hojo_asr.llm.rms_norm_eps", hp.llm_rms_eps);

        hp.bos_token_id = core_gguf::kv_u32(gctx, "hojo_asr.bos_token_id", hp.bos_token_id);
        hp.eos_token_id = core_gguf::kv_u32(gctx, "hojo_asr.eos_token_id", hp.eos_token_id);
        hp.gen_max_new_tokens = core_gguf::kv_u32(gctx, "hojo_asr.gen.max_new_tokens", hp.gen_max_new_tokens);
        hp.gen_num_beams = core_gguf::kv_u32(gctx, "hojo_asr.gen.num_beams", hp.gen_num_beams);
        hp.gen_repetition_penalty =
            core_gguf::kv_f32(gctx, "hojo_asr.gen.repetition_penalty", hp.gen_repetition_penalty);
        hp.gen_length_penalty = core_gguf::kv_f32(gctx, "hojo_asr.gen.length_penalty", hp.gen_length_penalty);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            vocab.token_to_id.reserve(vocab.id_to_token.size());
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++)
                vocab.token_to_id[vocab.id_to_token[i]] = i;
        }

        auto merges = core_gguf::kv_str_array(gctx, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++)
            vocab.merge_rank[merges[i]] = i;

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: tensor data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "hojo_asr", wl))
        return false;
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // ---- audio encoder ----
    auto& enc = model.enc;
    enc.conv1_w = require(model, "enc.conv1.weight");
    enc.conv1_b = require(model, "enc.conv1.bias");
    enc.conv2_w = require(model, "enc.conv2.weight");
    enc.conv2_b = require(model, "enc.conv2.bias");
    enc.conv3_w = require(model, "enc.conv3.weight");
    enc.conv3_b = require(model, "enc.conv3.bias");
    enc.conv_out_w = require(model, "enc.conv_out.weight");
    enc.ln_post_w = require(model, "enc.ln_post.weight");
    enc.ln_post_b = require(model, "enc.ln_post.bias");
    enc.proj1_w = require(model, "enc.proj1.weight");
    enc.proj1_b = require(model, "enc.proj1.bias");
    enc.proj2_w = require(model, "enc.proj2.weight");
    enc.proj2_b = require(model, "enc.proj2.bias");

    enc.blocks.resize(model.hparams.enc_layers);
    for (uint32_t i = 0; i < model.hparams.enc_layers; i++) {
        char buf[128];
        auto& b = enc.blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "enc.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_norm_b = get("attn_norm.bias");
        b.attn_q_w = get("attn.q.weight");
        b.attn_q_b = get("attn.q.bias");
        b.attn_k_w = get("attn.k.weight");
        b.attn_k_b = get("attn.k.bias");
        b.attn_v_w = get("attn.v.weight");
        b.attn_v_b = get("attn.v.bias");
        b.attn_o_w = get("attn.o.weight");
        b.attn_o_b = get("attn.o.bias");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_norm_b = get("ffn_norm.bias");
        b.ffn_fc1_w = get("ffn.fc1.weight");
        b.ffn_fc1_b = get("ffn.fc1.bias");
        b.ffn_fc2_w = get("ffn.fc2.weight");
        b.ffn_fc2_b = get("ffn.fc2.bias");
    }

    // ---- conformer adapter ----
    auto& adp = model.adp;
    adp.embed_proj_w = require(model, "adapter.embed.proj.weight");
    adp.embed_proj_b = require(model, "adapter.embed.proj.bias");
    adp.embed_norm_w = require(model, "adapter.embed.norm.weight");
    adp.embed_norm_b = require(model, "adapter.embed.norm.bias");
    adp.pe = require(model, "adapter.pe");
    adp.after_norm_w = require(model, "adapter.after_norm.weight");
    adp.after_norm_b = require(model, "adapter.after_norm.bias");
    adp.ln_speech_w = require(model, "ln_speech.weight");
    adp.ln_speech_b = require(model, "ln_speech.bias");

    adp.blocks.resize(model.hparams.adp_blocks);
    for (uint32_t i = 0; i < model.hparams.adp_blocks; i++) {
        char buf[128];
        auto& b = adp.blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "adapter.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.norm_ff_macaron_w = get("norm_ff_macaron.weight");
        b.norm_ff_macaron_b = get("norm_ff_macaron.bias");
        b.ffm_w1_w = get("ff_macaron.w1.weight");
        b.ffm_w1_b = get("ff_macaron.w1.bias");
        b.ffm_w2_w = get("ff_macaron.w2.weight");
        b.ffm_w2_b = get("ff_macaron.w2.bias");

        b.norm_mha_w = get("norm_mha.weight");
        b.norm_mha_b = get("norm_mha.bias");
        b.attn_q_w = get("attn.q.weight");
        b.attn_q_b = get("attn.q.bias");
        b.attn_k_w = get("attn.k.weight");
        b.attn_k_b = get("attn.k.bias");
        b.attn_v_w = get("attn.v.weight");
        b.attn_v_b = get("attn.v.bias");
        b.attn_o_w = get("attn.o.weight");
        b.attn_o_b = get("attn.o.bias");
        b.attn_pos_w = get("attn.pos.weight");
        b.pos_bias_u = get("attn.pos_bias_u");
        b.pos_bias_v = get("attn.pos_bias_v");

        b.norm_conv_w = get("norm_conv.weight");
        b.norm_conv_b = get("norm_conv.bias");
        b.conv_pw1_w = get("conv.pw1.weight");
        b.conv_pw1_b = get("conv.pw1.bias");
        b.conv_dw_w = get("conv.dw.weight");
        b.conv_dw_b = get("conv.dw.bias");
        b.conv_bn_scale = get("conv.norm.scale");
        b.conv_bn_shift = get("conv.norm.shift");
        b.conv_pw2_w = get("conv.pw2.weight");
        b.conv_pw2_b = get("conv.pw2.bias");

        b.norm_ff_w = get("norm_ff.weight");
        b.norm_ff_b = get("norm_ff.bias");
        b.ff_w1_w = get("ff.w1.weight");
        b.ff_w1_b = get("ff.w1.bias");
        b.ff_w2_w = get("ff.w2.weight");
        b.ff_w2_b = get("ff.w2.bias");

        b.norm_final_w = get("norm_final.weight");
        b.norm_final_b = get("norm_final.bias");
    }

    // ---- LLM ----
    auto& llm = model.llm;
    llm.embed_w = require(model, "llm.embed.weight");
    llm.final_norm_w = require(model, "llm.final_norm.weight");
    llm.lm_head_w = try_get(model, "llm.lm_head.weight");
    if (!llm.lm_head_w)
        llm.lm_head_w = llm.embed_w; // tie_word_embeddings=True

    llm.blocks.resize(model.hparams.llm_layers);
    for (uint32_t i = 0; i < model.hparams.llm_layers; i++) {
        char buf[128];
        auto& b = llm.blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "llm.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_q_w = get("attn.q.weight");
        b.attn_k_w = get("attn.k.weight");
        b.attn_v_w = get("attn.v.weight");
        b.attn_o_w = get("attn.o.weight");
        b.attn_q_norm_w = get("attn.q_norm.weight");
        b.attn_k_norm_w = get("attn.k_norm.weight");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn.gate.weight");
        b.ffn_up_w = get("ffn.up.weight");
        b.ffn_down_w = get("ffn.down.weight");
    }

    // ---- mel filterbank + window, straight out of the checkpoint ----
    {
        const int n_freqs = (int)model.hparams.n_fft / 2 + 1;
        const int n_mels = (int)model.hparams.n_mels;
        ggml_tensor* fb = try_get(model, "audio.mel_filters");
        ggml_tensor* win = try_get(model, "audio.mel_window");
        const bool fb_ok = fb && fb->type == GGML_TYPE_F32 && fb->ne[0] == n_freqs && fb->ne[1] == n_mels;
        const bool win_ok = win && win->type == GGML_TYPE_F32 && win->ne[0] == (int)model.hparams.n_fft;
        if (fb_ok && win_ok) {
            model.mel_filters.resize((size_t)n_mels * n_freqs);
            ggml_backend_tensor_get(fb, model.mel_filters.data(), 0, model.mel_filters.size() * sizeof(float));
            model.mel_window.resize(model.hparams.n_fft);
            ggml_backend_tensor_get(win, model.mel_window.data(), 0, model.mel_window.size() * sizeof(float));
            model.mel_from_checkpoint = true;
            fprintf(stderr, "hojo_asr: mel filterbank (%d x %d) + window (%u) taken from the GGUF\n", n_mels, n_freqs,
                    model.hparams.n_fft);
        } else {
            // Say WHICH state we are in: a rebuilt filterbank is a different
            // (and weaker) configuration from a copied one, and a readout that
            // renders both identically is useless.
            fprintf(stderr,
                    "hojo_asr: WARNING no baked mel filterbank in the GGUF (%s) — rebuilding a Slaney bank; "
                    "reconvert with a transformers-capable converter for exact parity\n",
                    fb ? "present but wrong shape/type" : "absent");
        }
    }

    // ---- encoder sinusoidal position table ----
    // SinusoidsPositionEmbedding: CONCAT halves — pe[p] = [sin(p·w) | cos(p·w)].
    // (The adapter's own table is INTERLEAVED and ships in the GGUF; the two
    // conventions are 40 lines apart in the reference forward pass.)
    {
        const int C = (int)model.hparams.enc_d_model;
        const int L = (int)model.hparams.enc_max_pos;
        const int half = C / 2;
        const float log_inc = std::log(10000.0f) / (float)(half - 1);
        std::vector<float> inv_t(half);
        for (int i = 0; i < half; i++)
            inv_t[i] = std::exp(-log_inc * (float)i);
        model.audio_pe.assign((size_t)L * C, 0.0f);
        for (int p = 0; p < L; p++) {
            float* row = model.audio_pe.data() + (size_t)p * C;
            for (int i = 0; i < half; i++) {
                const float angle = (float)p * inv_t[i];
                row[i] = std::sin(angle);
                row[half + i] = std::cos(angle);
            }
        }
    }

    const auto& hp = model.hparams;
    fprintf(stderr,
            "hojo_asr: loaded %u enc layers (d=%u, out=%u), %u conformer adapter blocks "
            "(d=%u, units=%u), %u LLM layers (d=%u), vocab=%u\n",
            hp.enc_layers, hp.enc_d_model, hp.enc_output_dim, hp.adp_blocks, hp.adp_hidden, hp.adp_units, hp.llm_layers,
            hp.llm_hidden, hp.llm_vocab_size);
    return true;
}

// ===========================================================================
// FFT (radix-2 with DFT fallback) — for the Whisper-style mel
// ===========================================================================

static void hojo_asr_dft(const float* in, int N, float* out) {
    for (int k = 0; k < N; k++) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; n++) {
            const float ang = -2.0f * (float)M_PI * (float)k * (float)n / (float)N;
            re += in[n] * std::cos(ang);
            im += in[n] * std::sin(ang);
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

static void hojo_asr_fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    const int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        hojo_asr_dft(in, N, out);
        return;
    }
    std::vector<float> even_in(half_N), odd_in(half_N);
    for (int i = 0; i < half_N; i++) {
        even_in[i] = in[2 * i];
        odd_in[i] = in[2 * i + 1];
    }
    std::vector<float> even_out(2 * half_N), odd_out(2 * half_N);
    hojo_asr_fft(even_in.data(), half_N, even_out.data());
    hojo_asr_fft(odd_in.data(), half_N, odd_out.data());
    for (int k = 0; k < half_N; k++) {
        const float ang = -2.0f * (float)M_PI * (float)k / (float)N;
        const float cos_a = std::cos(ang), sin_a = std::sin(ang);
        const float tre = odd_out[2 * k] * cos_a - odd_out[2 * k + 1] * sin_a;
        const float tim = odd_out[2 * k] * sin_a + odd_out[2 * k + 1] * cos_a;
        out[2 * k] = even_out[2 * k] + tre;
        out[2 * k + 1] = even_out[2 * k + 1] + tim;
        out[2 * (k + half_N)] = even_out[2 * k] - tre;
        out[2 * (k + half_N) + 1] = even_out[2 * k + 1] - tim;
    }
}

// ===========================================================================
// Mel spectrogram — WhisperFeatureExtractor with padding=False, so the mel is
// the audio's natural length (no 30 s / 40 s padding anywhere in the pipeline).
// ===========================================================================

extern "C" float* hojo_asr_compute_mel(struct hojo_asr_context* ctx, const float* samples, int n_samples,
                                       int* out_n_mels, int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int n_mels_val = (int)hp.n_mels;
    const int n_freqs = n_fft / 2 + 1;

    const auto& model = ctx->model;
    std::vector<float> hann;
    std::vector<float> mel_filters;
    if (model.mel_from_checkpoint) {
        hann = model.mel_window;
        mel_filters = model.mel_filters;
    } else {
        hann.resize(n_fft);
        for (int i = 0; i < n_fft; i++)
            hann[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)n_fft));
        mel_filters = core_mel::build_slaney_fb((int)hp.sample_rate, n_fft, n_mels_val, 0.0f, -1.0f,
                                                core_mel::FbLayout::MelsFreqs);
    }

    core_mel::FftR2C fft_fn = [](const float* in, int N, float* out) { hojo_asr_fft(const_cast<float*>(in), N, out); };
    core_mel::Params mel_params;
    mel_params.n_fft = n_fft;
    mel_params.hop_length = hop;
    mel_params.win_length = n_fft;
    mel_params.n_mels = n_mels_val;
    mel_params.log_base = core_mel::LogBase::Log10;
    mel_params.spec_kind = core_mel::SpecKind::Power;
    mel_params.norm = core_mel::Normalization::GlobalClipMax;
    mel_params.layout = core_mel::Layout::MelsTime;
    mel_params.log_guard = core_mel::LogGuard::MaxClip;
    mel_params.log_eps = 1e-10f;
    mel_params.center_pad = true;
    mel_params.center_pad_reflect = true;

    int T_mel_actual = 0;
    std::vector<float> mel_out = core_mel::compute(samples, n_samples, hann.data(), n_fft, mel_filters.data(), n_freqs,
                                                   fft_fn, mel_params, T_mel_actual);

    // WhisperFeatureExtractor drops the trailing STFT frame (stft[..., :-1]),
    // yielding exactly n_samples/hop frames. core_mel's center padding produces
    // one extra; truncate to match (else every downstream frame count drifts).
    const int T_target = n_samples / hop;
    if (T_mel_actual > T_target && T_target > 0) {
        std::vector<float> trunc((size_t)n_mels_val * T_target);
        for (int f = 0; f < n_mels_val; f++)
            memcpy(trunc.data() + (size_t)f * T_target, mel_out.data() + (size_t)f * T_mel_actual,
                   (size_t)T_target * sizeof(float));
        mel_out = std::move(trunc);
        T_mel_actual = T_target;
    }

    float* result = (float*)malloc(mel_out.size() * sizeof(float));
    if (!result)
        return nullptr;
    memcpy(result, mel_out.data(), mel_out.size() * sizeof(float));
    if (out_n_mels)
        *out_n_mels = n_mels_val;
    if (out_T_mel)
        *out_T_mel = T_mel_actual;
    return result;
}

// ===========================================================================
// Audio encoder — conv stem
// ===========================================================================
//
// The reference cuts the mel into chunks of `n_window * 2` (= 3000) frames,
// pads the batch to the longest chunk and runs ONE Conv2d stack over the
// padded batch. A 3000-frame chunk makes ggml's im2col for conv2 a
// 4320 x 96000 F16 matrix (~830 MB) — far too much transient memory for a
// model that already needs ~4 GB of weights.
//
// So the conv is TILED along time. The stack is 3 x (k=3, stride=2, pad=1),
// so output frame `o` reads input frames [8o-7, 8o+7] and nothing else. A
// tile that starts its input at S = 8*o0 - 8 therefore has every input that
// its kept outputs [o0, o1) need, and the conv's own zero padding at the
// tile edge is only ever read by the one throw-away frame on each side.
// That is an exact identity, not an approximation — the output is
// bit-comparable with the untiled path, which is still available via
// CRISPASR_HOJO_ASR_CONV_TILE=0 so the two can be A/B'd.

static int hojo_asr_conv_tile_frames() {
    static int v = -2;
    if (v == -2) {
        const char* e = crispasr_env::get("CRISPASR_HOJO_ASR_CONV_TILE");
        v = (e && *e) ? atoi(e) : 64;
        if (v < 0)
            v = 64;
    }
    return v;
}

// Builds the conv front end for a mel window of exactly `win_T` frames:
//   mel(n_mels, win_T) -> conv1/2/3(gelu) -> conv_out(no bias) -> (d, win_T/8)
// Output tensor "conv_stem_out".
static ggml_cgraph* hojo_asr_build_conv_graph(hojo_asr_context* ctx, int win_T, ggml_context* arena_ctx) {
    const auto& hp = ctx->model.hparams;
    const auto& enc = ctx->model.enc;
    const int n_mels = (int)hp.n_mels;

    ggml_context* ctx0 = arena_ctx;
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // mel input as ggml ne=(n_mels, win_T) — freq fastest.
    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, win_T);
    ggml_set_name(mel_in, "mel_input");
    ggml_set_input(mel_in);
    ggml_tensor* x = ggml_reshape_4d(ctx0, mel_in, n_mels, win_T, 1, 1);

    // Kernel stored (KW,KH,IC,OC); with data ne[0]=freq we need (KH,KW,IC,OC).
    auto kt = [&](ggml_tensor* w) -> ggml_tensor* { return ggml_cont(ctx0, ggml_permute(ctx0, w, 1, 0, 2, 3)); };
    auto bias4 = [&](ggml_tensor* b) -> ggml_tensor* { return ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1); };

    x = ggml_conv_2d(ctx0, kt(enc.conv1_w), x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(ctx0, x, bias4(enc.conv1_b));
    x = ggml_gelu_erf(ctx0, x);
    x = ggml_conv_2d(ctx0, kt(enc.conv2_w), x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(ctx0, x, bias4(enc.conv2_b));
    x = ggml_gelu_erf(ctx0, x);
    x = ggml_conv_2d(ctx0, kt(enc.conv3_w), x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(ctx0, x, bias4(enc.conv3_b));
    x = ggml_gelu_erf(ctx0, x);

    // After conv3: ggml ne=(F_down, T_down, C, 1). PyTorch flattens
    // (b,c,f,t) -> (b,t,c*f) with f fastest → merge ne[0]=F_down and ne[2]=C.
    const int F_down = conv_out_len(conv_out_len(conv_out_len(n_mels)));
    const int T_down = conv_out_len(conv_out_len(conv_out_len(win_T)));
    const int C_out = (int)hp.enc_ds_hidden;
    const int conv_in = F_down * C_out;
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3)); // (F, C, T, 1)
    x = ggml_reshape_2d(ctx0, x, conv_in, T_down);

    x = ggml_mul_mat(ctx0, enc.conv_out_w, x); // (d, T_down)

    ggml_set_name(x, "conv_stem_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// Run the conv stem over one mel chunk, writing `out` as (d, n_out) with d
// fastest. `mel` is the FULL (n_mels, T_mel) buffer (mel-major); the chunk is
// [t0, t0 + chunk_len). Frames outside the chunk read as zero — exactly the
// reference's pad_sequence behaviour.
static bool hojo_asr_conv_chunk(hojo_asr_context* ctx, const float* mel, int n_mels, int T_mel, int t0, int chunk_len,
                                int win_T, int n_out, float* out) {
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.enc_d_model;

    auto run_window = [&](int mel_start, int width, int keep_from, int keep_count, int out_off) -> bool {
        // (Re)build the graph when the width changes. Building fresh per
        // encoder invocation is deliberate: caching a conv graph across
        // invocations is a use-after-free once the larger LLM graphs regrow
        // the shared sched's gallocr (the #215 Vulkan crash in moss).
        if (!ctx->conv_gf || ctx->conv_win != width) {
            if (ctx->conv_ctx) {
                ggml_free(ctx->conv_ctx);
                ctx->conv_ctx = nullptr;
                ctx->conv_gf = nullptr;
            }
            ctx->conv_meta.assign(ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false), 0);
            ggml_init_params aip = {ctx->conv_meta.size(), ctx->conv_meta.data(), true};
            ctx->conv_ctx = ggml_init(aip);
            ctx->conv_gf = hojo_asr_build_conv_graph(ctx, width, ctx->conv_ctx);
            ctx->conv_win = width;
        }

        std::vector<float> win_mel((size_t)n_mels * width, 0.0f);
        for (int t = 0; t < width; t++) {
            const int src = mel_start + t;
            // Outside the chunk (and outside the mel) is zero — pad_sequence.
            if (src < t0 || src >= t0 + chunk_len || src < 0 || src >= T_mel)
                continue;
            for (int f = 0; f < n_mels; f++)
                win_mel[(size_t)f + (size_t)n_mels * t] = mel[(size_t)f * T_mel + src];
        }

        ggml_cgraph* gf = ctx->conv_gf;
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "hojo_asr: conv graph alloc failed (width %d)\n", width);
            return false;
        }
        ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel_input");
        ggml_backend_tensor_set(mel_in, win_mel.data(), 0, win_mel.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "hojo_asr: conv graph compute failed (width %d)\n", width);
            return false;
        }
        ggml_tensor* cs = ggml_graph_get_tensor(gf, "conv_stem_out");
        const int T_down = (int)cs->ne[1];
        if (keep_from + keep_count > T_down) {
            fprintf(stderr, "hojo_asr: conv tile overrun (%d+%d > %d)\n", keep_from, keep_count, T_down);
            return false;
        }
        ggml_backend_tensor_get(cs, out + (size_t)out_off * d, (size_t)keep_from * d * sizeof(float),
                                (size_t)keep_count * d * sizeof(float));
        return true;
    };

    const int tile = hojo_asr_conv_tile_frames();
    if (tile <= 0)
        return run_window(t0, win_T, 0, n_out, 0);

    for (int o0 = 0; o0 < n_out; o0 += tile) {
        const int o1 = std::min(o0 + tile, n_out);
        const core_hojo_frames::TileWindow w = core_hojo_frames::tile_window(o0, o1, win_T);
        if (!run_window(t0 + w.mel_offset, w.width, w.keep_from, w.keep_count, w.out_offset))
            return false;
    }
    return true;
}

// ===========================================================================
// Audio encoder — transformer (32 layers + ln_post + proj1/proj2)
// ===========================================================================
//
// The reference builds a block-diagonal attention mask so frames only attend
// inside their own `n_window * 2` mel-frame chunk, then runs all chunks as one
// flat sequence. Every other op in the tower is per-frame, so that is exactly
// equivalent to running each chunk independently with full attention — which
// is what this does. It also avoids materialising a T_enc x T_enc mask that
// would be 112 MB for ten minutes of audio.

static ggml_cgraph* hojo_asr_build_encoder_xf_graph(hojo_asr_context* ctx, int T_enc) {
    const auto& hp = ctx->model.hparams;
    const auto& enc = ctx->model.enc;
    const int d = (int)hp.enc_d_model;
    const int n_heads = (int)hp.enc_n_heads;
    const int head_dim = (int)hp.enc_head_dim;

    struct ggml_init_params gparams = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_enc);
    ggml_set_name(x, "xf_input");
    ggml_set_input(x);

    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    for (uint32_t il = 0; il < hp.enc_layers; il++) {
        const auto& blk = enc.blocks[il];
        ggml_tensor* residual = x;

        ggml_tensor* h = ggml_norm(ctx0, x, hp.enc_ln_eps);
        h = ggml_add(ctx0, ggml_mul(ctx0, h, blk.attn_norm_w), blk.attn_norm_b);

        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_q_w, h), blk.attn_q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_k_w, h), blk.attn_k_b);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_v_w, h), blk.attn_v_b);

        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, T_enc);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_heads, T_enc);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T_enc);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // (hd, T, n_h)
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        ggml_tensor* attn;
        if (ctx->enc_use_flash) {
            attn = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
            attn = ggml_reshape_2d(ctx0, attn, d, T_enc);
        } else {
            // Manual masked-softmax attention (Vulkan-safe, issue #215).
            ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q); // (T_k, T_q, n_h)
            scores = ggml_soft_max_ext(ctx0, scores, nullptr, attn_scale, 0.0f);
            ggml_tensor* V_perm = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3)); // (T_k, hd, n_h)
            attn = ggml_mul_mat(ctx0, V_perm, scores);                                // (hd, T_q, n_h)
            attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));             // (hd, n_h, T_q)
            attn = ggml_reshape_2d(ctx0, attn, d, T_enc);
        }

        ggml_tensor* attn_out = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_o_w, attn), blk.attn_o_b);
        x = ggml_add(ctx0, residual, attn_out);

        residual = x;
        h = ggml_norm(ctx0, x, hp.enc_ln_eps);
        h = ggml_add(ctx0, ggml_mul(ctx0, h, blk.ffn_norm_w), blk.ffn_norm_b);
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.ffn_fc1_w, h), blk.ffn_fc1_b);
        h = ggml_gelu_erf(ctx0, h);
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.ffn_fc2_w, h), blk.ffn_fc2_b);
        x = ggml_add(ctx0, residual, h);

        if (il == 0) {
            ggml_tensor* l0 = ggml_cont(ctx0, x);
            ggml_set_name(l0, "enc_layer_0");
            ggml_set_output(l0);
            ggml_build_forward_expand(gf, l0);
        }
    }

    x = ggml_norm(ctx0, x, hp.enc_ln_eps);
    x = ggml_add(ctx0, ggml_mul(ctx0, x, enc.ln_post_w), enc.ln_post_b);

    x = ggml_add(ctx0, ggml_mul_mat(ctx0, enc.proj1_w, x), enc.proj1_b);
    x = ggml_gelu_erf(ctx0, x);
    x = ggml_add(ctx0, ggml_mul_mat(ctx0, enc.proj2_w, x), enc.proj2_b);

    ggml_set_name(x, "encoder_output");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

extern "C" float* hojo_asr_run_encoder(struct hojo_asr_context* ctx, const float* mel, int n_mels, int T_mel,
                                       int* out_T_enc, int* out_d) {
    if (!ctx || !mel || T_mel <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.enc_d_model;
    const int out_dim = (int)hp.enc_output_dim;
    const int chunk_T = (int)hp.n_window * 2; // 3000 mel frames
    const int nwi = (int)hp.n_window_infer;   // 3000

    // pad_sequence pads the batch of chunks to the LONGEST chunk, so a single
    // short utterance is convolved at its own width, not padded out to 3000.
    const core_hojo_frames::ChunkPlan plan = core_hojo_frames::plan_chunks(T_mel, chunk_T, nwi);
    const int num_chunks = (int)plan.chunks.size();
    const int win_T = plan.win_T;
    const int T_enc = plan.T_enc;
    if (num_chunks <= 0 || T_enc <= 0)
        return nullptr;
    for (const auto& ch : plan.chunks) {
        if (ch.valid > (int)hp.enc_max_pos) {
            fprintf(stderr, "hojo_asr: chunk of %d encoder frames exceeds the %u-row position table\n", ch.valid,
                    hp.enc_max_pos);
            return nullptr;
        }
    }

    // Drop any conv graph left over from a previous invocation (see the note in
    // hojo_asr_conv_chunk).
    if (ctx->conv_ctx) {
        ggml_free(ctx->conv_ctx);
        ctx->conv_ctx = nullptr;
        ctx->conv_gf = nullptr;
        ctx->conv_win = 0;
    }

    float* result = (float*)malloc((size_t)out_dim * T_enc * sizeof(float));
    if (!result)
        return nullptr;

    // PHASE 1 — every chunk's conv stem, BEFORE any transformer graph runs.
    //
    // The conv graph is reused across chunks (and across tiles) of the same
    // width, which is only safe while no LARGER graph passes through the same
    // sched in between: a bigger graph regrows the sched's gallocr, frees the
    // ggml_backend_buffer structs the cached graph's tensors still point at,
    // and the next sched_alloc_graph reads them. That is the #215 Vulkan
    // segfault / ASan heap-use-after-free, and interleaving conv and
    // transformer per chunk walks straight into it.
    std::vector<float> hidden((size_t)d * T_enc, 0.0f);
    {
        int off = 0;
        for (int c = 0; c < num_chunks; c++) {
            const int valid = plan.chunks[c].valid;
            if (!hojo_asr_conv_chunk(ctx, mel, n_mels, T_mel, plan.chunks[c].t0, plan.chunks[c].len, win_T, valid,
                                     hidden.data() + (size_t)off * d)) {
                free(result);
                return nullptr;
            }
            // Per-chunk sinusoidal positions restart at 0.
            for (int t = 0; t < valid; t++) {
                const float* pe = ctx->model.audio_pe.data() + (size_t)t * d;
                float* dst = hidden.data() + (size_t)(off + t) * d;
                for (int j = 0; j < d; j++)
                    dst[j] += pe[j];
            }
            off += valid;
        }
    }
    // The conv graph must not outlive phase 1 for the same reason.
    if (ctx->conv_ctx) {
        ggml_free(ctx->conv_ctx);
        ctx->conv_ctx = nullptr;
        ctx->conv_gf = nullptr;
        ctx->conv_win = 0;
    }

    // PHASE 2 — the 32-layer tower, once per chunk. Attention is
    // block-diagonal over chunks in the reference and every other op is
    // per-frame, so per-chunk full attention is the same computation.
    int out_off = 0;
    for (int c = 0; c < num_chunks; c++) {
        const int valid = plan.chunks[c].valid;

        ggml_cgraph* gf = hojo_asr_build_encoder_xf_graph(ctx, valid);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "hojo_asr: encoder xf alloc failed (chunk %d)\n", c);
            free(result);
            return nullptr;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "xf_input"), hidden.data() + (size_t)out_off * d, 0,
                                (size_t)d * valid * sizeof(float));
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "hojo_asr: encoder xf compute failed (chunk %d)\n", c);
            free(result);
            return nullptr;
        }
        ggml_tensor* eo = ggml_graph_get_tensor(gf, "encoder_output");
        ggml_backend_tensor_get(eo, result + (size_t)out_off * out_dim, 0, (size_t)out_dim * valid * sizeof(float));

        if (c == 0) {
            if (const char* dp = crispasr_env::get("CRISPASR_HOJO_ASR_L0_DUMP")) {
                ggml_tensor* l0 = ggml_graph_get_tensor(gf, "enc_layer_0");
                if (l0) {
                    std::vector<float> b((size_t)d * valid);
                    ggml_backend_tensor_get(l0, b.data(), 0, b.size() * sizeof(float));
                    FILE* f = fopen(dp, "wb");
                    if (f) {
                        fwrite(b.data(), sizeof(float), b.size(), f);
                        fclose(f);
                        fprintf(stderr, "hojo_asr: dumped enc_layer_0 (%d,%d)\n", d, valid);
                    }
                }
            }
        }

        out_off += valid;
    }

    // Always name the conv schedule that produced this output. The tiled and
    // untiled paths must agree exactly, and an A/B that cannot tell "both arms
    // tiled" from "tiling is exact" would report a broken instrument as a fact
    // about the code — so the arm identifies itself in the log.
    {
        const int tile_now = hojo_asr_conv_tile_frames();
        fprintf(stderr, "hojo_asr: conv_schedule=%s tile=%d chunks=%d T_enc=%d\n", tile_now > 0 ? "tiled" : "untiled",
                tile_now, num_chunks, T_enc);
    }
    if (const char* dp = crispasr_env::get("CRISPASR_HOJO_ASR_ENC_DUMP")) {
        FILE* f = fopen(dp, "wb");
        if (f) {
            fwrite(result, sizeof(float), (size_t)out_dim * T_enc, f);
            fclose(f);
            fprintf(stderr, "hojo_asr: dumped encoder_output (%d,%d)\n", out_dim, T_enc);
        }
    }

    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d)
        *out_d = out_dim;
    return result;
}

// ===========================================================================
// Conformer adapter (WeNet ConformerEncoder, 2 macaron blocks) + ln_speech
// ===========================================================================
//
// Every structural choice below comes from WeNet's defaults, because
// hojo_asr_model.py passes only (input_size, output_size, linear_units,
// num_blocks, input_layer) and takes the rest:
//   pos_enc_layer_type="rel_pos"  selfattention_layer_type="rel_selfattn"
//   attention_heads=4  macaron_style=True  activation="swish" (SiLU)
//   use_cnn_module=True  cnn_module_kernel=15  causal=False
//   cnn_module_norm="batch_norm"  normalize_before=True  norm_eps=1e-5
//
// Two details that are invisible in the shapes and wrong by default:
//   1. LinearNoSubsampling feeds RelPositionalEncoding, whose forward does
//      `x = x * sqrt(d_model)` — a 50.6x scale on the residual stream — and
//      returns the position table SEPARATELY. The table is never added to x.
//   2. WeNet comments out `rel_shift`, so matrix_bd is
//      (q + pos_bias_v) . (W_pos . pe[0:T])^T with NO shift. Implementing the
//      Transformer-XL shift here would be plausible, compile, and be wrong.

static ggml_cgraph* hojo_asr_build_adapter_graph(hojo_asr_context* ctx, int T) {
    const auto& hp = ctx->model.hparams;
    const auto& adp = ctx->model.adp;
    const int d_enc = (int)hp.enc_output_dim;
    const int d = (int)hp.adp_hidden;
    const int nh = (int)hp.adp_heads;
    const int hd = (int)hp.adp_head_dim;
    const int units = (int)hp.adp_units;
    const int K = (int)hp.adp_kernel;
    const float eps = hp.adp_ln_eps;
    const float ff_scale = hp.adp_ff_scale;
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    struct ggml_init_params gparams = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* enc_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_enc, T);
    ggml_set_name(enc_in, "adapter_in");
    ggml_set_input(enc_in);

    auto layer_norm = [&](ggml_tensor* v, ggml_tensor* w, ggml_tensor* b) {
        ggml_tensor* n = ggml_norm(ctx0, v, eps);
        return ggml_add(ctx0, ggml_mul(ctx0, n, w), b);
    };

    // ---- LinearNoSubsampling: Linear -> LayerNorm -> x * sqrt(d_model) ----
    ggml_tensor* x = ggml_add(ctx0, ggml_mul_mat(ctx0, adp.embed_proj_w, enc_in), adp.embed_proj_b);
    x = layer_norm(x, adp.embed_norm_w, adp.embed_norm_b);
    x = ggml_scale(ctx0, x, std::sqrt((float)d));
    {
        ggml_tensor* dbg = ggml_cont(ctx0, x);
        ggml_set_name(dbg, "adapter_embed");
        ggml_set_output(dbg);
        ggml_build_forward_expand(gf, dbg);
    }

    // ---- RelPositionalEncoding table, rows [0, T) ----
    ggml_tensor* pos = ggml_cont(ctx0, ggml_view_2d(ctx0, adp.pe, d, T, adp.pe->nb[1], 0));

    for (uint32_t il = 0; il < hp.adp_blocks; il++) {
        const auto& blk = adp.blocks[il];

        // ---- macaron feed-forward (half step) ----
        {
            ggml_tensor* h = layer_norm(x, blk.norm_ff_macaron_w, blk.norm_ff_macaron_b);
            h = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.ffm_w1_w, h), blk.ffm_w1_b);
            h = ggml_silu(ctx0, h);
            h = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.ffm_w2_w, h), blk.ffm_w2_b);
            x = ggml_add(ctx0, x, ggml_scale(ctx0, h, ff_scale));
        }

        // ---- rel-pos multi-head self-attention ----
        {
            ggml_tensor* h = layer_norm(x, blk.norm_mha_w, blk.norm_mha_b);

            ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_q_w, h), blk.attn_q_b);
            ggml_tensor* Kt = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_k_w, h), blk.attn_k_b);
            ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_v_w, h), blk.attn_v_b);
            ggml_tensor* P = ggml_mul_mat(ctx0, blk.attn_pos_w, pos); // linear_pos, no bias

            Q = ggml_reshape_3d(ctx0, Q, hd, nh, T);
            // pos_bias_* are added to an F32 Q; ggml's binbcast rejects
            // F32 (+) F16 on every backend, so cast defensively even though
            // the converter already emits these two F32.
            ggml_tensor* bu_src =
                blk.pos_bias_u->type == GGML_TYPE_F32 ? blk.pos_bias_u : ggml_cast(ctx0, blk.pos_bias_u, GGML_TYPE_F32);
            ggml_tensor* bv_src =
                blk.pos_bias_v->type == GGML_TYPE_F32 ? blk.pos_bias_v : ggml_cast(ctx0, blk.pos_bias_v, GGML_TYPE_F32);
            ggml_tensor* bu = ggml_reshape_3d(ctx0, bu_src, hd, nh, 1);
            ggml_tensor* bv = ggml_reshape_3d(ctx0, bv_src, hd, nh, 1);
            ggml_tensor* Qu = ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, Q, bu), 0, 2, 1, 3)); // (hd,T,nh)
            ggml_tensor* Qv = ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, Q, bv), 0, 2, 1, 3));

            Kt = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Kt, hd, nh, T), 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, T), 0, 2, 1, 3));
            P = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, P, hd, nh, T), 0, 2, 1, 3));

            ggml_tensor* ac = ggml_mul_mat(ctx0, Kt, Qu); // (T_k, T_q, nh)
            ggml_tensor* bd = ggml_mul_mat(ctx0, P, Qv);  // (T_k, T_q, nh) — no rel_shift
            ggml_tensor* scores = ggml_add(ctx0, ac, bd);
            scores = ggml_soft_max_ext(ctx0, scores, nullptr, attn_scale, 0.0f);

            ggml_tensor* V_perm = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3)); // (T_k, hd, nh)
            ggml_tensor* attn = ggml_mul_mat(ctx0, V_perm, scores);                   // (hd, T_q, nh)
            attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));             // (hd, nh, T_q)
            attn = ggml_reshape_2d(ctx0, attn, d, T);
            attn = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_o_w, attn), blk.attn_o_b);
            x = ggml_add(ctx0, x, attn);
        }

        // ---- convolution module ----
        {
            ggml_tensor* h = layer_norm(x, blk.norm_conv_w, blk.norm_conv_b);

            // pointwise_conv1: Conv1d(d -> 2d, k=1) stored (2d, d, 1) → ggml (1, d, 2d)
            h = ggml_mul_mat(
                ctx0, ggml_reshape_2d(ctx0, blk.conv_pw1_w, (int)blk.conv_pw1_w->ne[1], (int)blk.conv_pw1_w->ne[2]), h);
            h = ggml_add(ctx0, h, blk.conv_pw1_b);

            // GLU over the channel axis: first half * sigmoid(second half).
            ggml_tensor* g1 = ggml_cont(ctx0, ggml_view_2d(ctx0, h, d, T, h->nb[1], 0));
            ggml_tensor* g2 = ggml_cont(ctx0, ggml_view_2d(ctx0, h, d, T, h->nb[1], (size_t)d * sizeof(float)));
            h = ggml_mul(ctx0, g1, ggml_sigmoid(ctx0, g2));

            // depthwise Conv1d(d, d, k=15, pad=7, groups=d) as a batched 2-D dw conv
            ggml_tensor* dw = ggml_cast(ctx0, blk.conv_dw_w, GGML_TYPE_F32);
            dw = ggml_reshape_4d(ctx0, dw, K, 1, 1, d);
            ggml_tensor* ht = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // (T, d)
            ht = ggml_reshape_4d(ctx0, ht, T, 1, d, 1);
            ht = ggml_conv_2d_dw_direct(ctx0, dw, ht, 1, 1, K / 2, 0, 1, 1);
            h = ggml_cont(ctx0, ggml_permute(ctx0, ht, 1, 2, 0, 3));
            h = ggml_reshape_2d(ctx0, h, d, T);
            h = ggml_add(ctx0, h, blk.conv_dw_b);

            // eval-mode BatchNorm1d, folded to a per-channel affine at convert time
            h = ggml_add(ctx0, ggml_mul(ctx0, h, blk.conv_bn_scale), blk.conv_bn_shift);
            h = ggml_silu(ctx0, h);

            h = ggml_mul_mat(
                ctx0, ggml_reshape_2d(ctx0, blk.conv_pw2_w, (int)blk.conv_pw2_w->ne[1], (int)blk.conv_pw2_w->ne[2]), h);
            h = ggml_add(ctx0, h, blk.conv_pw2_b);
            x = ggml_add(ctx0, x, h);
        }

        // ---- feed-forward (half step) ----
        {
            ggml_tensor* h = layer_norm(x, blk.norm_ff_w, blk.norm_ff_b);
            h = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.ff_w1_w, h), blk.ff_w1_b);
            h = ggml_silu(ctx0, h);
            h = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.ff_w2_w, h), blk.ff_w2_b);
            x = ggml_add(ctx0, x, ggml_scale(ctx0, h, ff_scale));
        }

        // ---- per-block final LayerNorm (only present with a conv module) ----
        x = layer_norm(x, blk.norm_final_w, blk.norm_final_b);

        if (il == 0) {
            ggml_tensor* dbg = ggml_cont(ctx0, x);
            ggml_set_name(dbg, "adapter_blk0");
            ggml_set_output(dbg);
            ggml_build_forward_expand(gf, dbg);
        }
    }

    // ---- BaseEncoder.after_norm (normalize_before=True) ----
    x = layer_norm(x, adp.after_norm_w, adp.after_norm_b);
    {
        ggml_tensor* dbg = ggml_cont(ctx0, x);
        ggml_set_name(dbg, "adapter_output");
        ggml_set_output(dbg);
        ggml_build_forward_expand(gf, dbg);
    }

    // ---- ln_speech (applied by encode_speech, outside the bottleneck) ----
    x = layer_norm(x, adp.ln_speech_w, adp.ln_speech_b);
    ggml_set_name(x, "speech_embeds");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    (void)units;
    return gf;
}

extern "C" float* hojo_asr_run_adapter(struct hojo_asr_context* ctx, const float* encoder_out, int T_enc, int d_enc,
                                       int* out_T, int* out_d, float** out_pre_ln_speech) {
    if (out_pre_ln_speech)
        *out_pre_ln_speech = nullptr;
    if (!ctx || !encoder_out || T_enc <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d_llm = (int)hp.adp_hidden;
    if (T_enc > (int)ctx->model.adp.pe->ne[1]) {
        fprintf(stderr, "hojo_asr: %d adapter frames exceed the %d-row position table\n", T_enc,
                (int)ctx->model.adp.pe->ne[1]);
        return nullptr;
    }

    ggml_cgraph* gf = hojo_asr_build_adapter_graph(ctx, T_enc);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "hojo_asr: adapter alloc failed\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "adapter_in"), encoder_out, 0,
                            (size_t)d_enc * T_enc * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "hojo_asr: adapter compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "speech_embeds");
    if (!out)
        return nullptr;
    float* result = (float*)malloc((size_t)d_llm * T_enc * sizeof(float));
    if (!result)
        return nullptr;
    ggml_backend_tensor_get(out, result, 0, (size_t)d_llm * T_enc * sizeof(float));

    if (out_pre_ln_speech) {
        ggml_tensor* pre = ggml_graph_get_tensor(gf, "adapter_output");
        if (pre) {
            float* buf = (float*)malloc((size_t)d_llm * T_enc * sizeof(float));
            if (buf) {
                ggml_backend_tensor_get(pre, buf, 0, (size_t)d_llm * T_enc * sizeof(float));
                *out_pre_ln_speech = buf;
            }
        }
    }

    if (const char* dp = crispasr_env::get("CRISPASR_HOJO_ASR_ADAPTER_DUMP")) {
        FILE* f = fopen(dp, "wb");
        if (f) {
            fwrite(result, sizeof(float), (size_t)d_llm * T_enc, f);
            fclose(f);
            fprintf(stderr, "hojo_asr: dumped speech_embeds (%d,%d)\n", d_llm, T_enc);
        }
    }

    if (out_T)
        *out_T = T_enc;
    if (out_d)
        *out_d = d_llm;
    return result;
}

// ===========================================================================
// LLM graph with KV cache (Qwen3-4B)
// ===========================================================================

static ggml_cgraph* hojo_asr_build_llm_kv_graph(hojo_asr_context* ctx, int n_tokens, int n_past, bool last_token_only) {
    const auto& hp = ctx->model.hparams;
    const auto& llm = ctx->model.llm;
    const int d = (int)hp.llm_hidden;
    const int n_heads = (int)hp.llm_n_heads;
    const int n_kv_heads = (int)hp.llm_n_kv_heads;
    const int head_dim = (int)hp.llm_head_dim;
    const int Lk = n_past + n_tokens;

    struct ggml_init_params gparams = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_tokens);
    ggml_set_name(embeds_in, "inputs_embeds");
    ggml_set_input(embeds_in);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    // Value-initialised: an uninitialised attn_scale is stack garbage.
    core_attn::KvSelfAttnParams attn_p{};
    attn_p.n_heads = n_heads;
    attn_p.n_kv_heads = n_kv_heads;
    attn_p.head_dim = head_dim;
    attn_p.n_kv_grp = n_heads / n_kv_heads;
    attn_p.n_ctx_orig = 0;
    attn_p.rope_theta = hp.llm_rope_theta;
    attn_p.rope_beta_fast = 32.0f;
    attn_p.rope_beta_slow = 1.0f;
    attn_p.attn_scale = 1.0f / std::sqrt((float)head_dim);
    attn_p.qk_norm_eps = hp.llm_rms_eps;
    attn_p.gqa_mode = core_attn::GQA_MANUAL_CONT;
    attn_p.rope_type = GGML_ROPE_TYPE_NEOX;

    ggml_tensor* cur = embeds_in;
    for (uint32_t il = 0; il < hp.llm_layers; il++) {
        const auto& blk = llm.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* h = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        h = ggml_mul(ctx0, h, blk.attn_norm_w);

        ggml_tensor* attn_out = core_attn::kv_self_attn(ctx0, gf, h, blk.attn_q_w, blk.attn_k_w, blk.attn_v_w,
                                                        blk.attn_o_w, blk.attn_q_norm_w, blk.attn_k_norm_w, positions,
                                                        causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, attn_p);
        cur = ggml_add(ctx0, residual, attn_out);

        residual = cur;
        h = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        h = ggml_mul(ctx0, h, blk.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, h, blk.ffn_gate_w, blk.ffn_up_w, blk.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
    cur = ggml_mul(ctx0, cur, llm.final_norm_w);

    if (last_token_only && n_tokens > 1)
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(n_tokens - 1) * cur->nb[1]);
    cur = ggml_mul_mat(ctx0, llm.lm_head_w, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ===========================================================================
// KV cache management
// ===========================================================================

extern "C" bool hojo_asr_kv_init(struct hojo_asr_context* ctx, int max_ctx) {
    if (!ctx)
        return false;
    const auto& hp = ctx->model.hparams;
    const int n_layers = (int)hp.llm_layers;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;

    ggml_type kv_type = core_attn::kv_dtype_from_env("hojo_asr");

    struct ggml_init_params kv_params = {2 * ggml_tensor_overhead(), nullptr, true};
    ctx->kv_ctx = ggml_init(kv_params);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, n_layers);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, n_layers);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf) {
        fprintf(stderr, "hojo_asr: kv alloc failed for max_ctx=%d\n", max_ctx);
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
        return false;
    }
    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used = 0;
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    return true;
}

extern "C" void hojo_asr_kv_reset(struct hojo_asr_context* ctx) {
    if (ctx && ctx->kv_buf) {
        ggml_backend_buffer_clear(ctx->kv_buf, 0);
        ctx->kv_n_used = 0;
    }
}

extern "C" float* hojo_asr_run_llm_kv(struct hojo_asr_context* ctx, const float* inputs_embeds, int n_tokens,
                                      int n_past, int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0 || !ctx->kv_k)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_hidden;
    const int vocab = (int)hp.llm_vocab_size;
    const int Lk = n_past + n_tokens;

    ggml_cgraph* gf = hojo_asr_build_llm_kv_graph(ctx, n_tokens, n_past, /*last_token_only=*/true);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "hojo_asr: llm alloc failed\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), inputs_embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            (size_t)n_tokens * sizeof(int32_t));

    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        std::vector<ggml_fp16_t> mask((size_t)Lk * n_tokens);
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            const int abs_q = n_past + q;
            for (int k = 0; k < Lk; k++)
                mask[(size_t)q * Lk + k] = (k <= abs_q) ? zero_h : neginf_h;
        }
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "hojo_asr: llm compute failed\n");
        return nullptr;
    }
    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    if (!logits_t)
        return nullptr;
    if (out_n_tokens)
        *out_n_tokens = 1;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    if (!result)
        return nullptr;
    ggml_backend_tensor_get(logits_t, result, 0, (size_t)vocab * sizeof(float));
    ctx->kv_n_used = Lk;
    return result;
}

// ===========================================================================
// Tokenizer (GPT-2 byte-level BPE via core_bpe)
// ===========================================================================

extern "C" int hojo_asr_tokenize(struct hojo_asr_context* ctx, const char* text, int32_t* out_tokens, int max_tokens) {
    if (!ctx || !text || !out_tokens || max_tokens <= 0)
        return 0;
    const auto& v = ctx->vocab;
    std::vector<int32_t> result;
    const std::string s = text;
    size_t i = 0;
    while (i < s.size()) {
        size_t k = i;
        const size_t start = k;
        if (s[k] == ' ' || s[k] == '\t' || s[k] == '\n')
            k++;
        while (k < s.size() && s[k] != ' ' && s[k] != '\t' && s[k] != '\n')
            k++;
        if (k == start)
            k++;
        const std::string pre(s, start, k - start);
        i = k;
        const std::string encoded = core_bpe::bytes_to_unicode(pre.data(), pre.size());
        core_bpe::bpe_one(v.token_to_id, v.merge_rank, encoded, result);
    }
    const int n = std::min((int)result.size(), max_tokens);
    std::memcpy(out_tokens, result.data(), (size_t)n * sizeof(int32_t));
    return n;
}

extern "C" const char* hojo_asr_token_text(struct hojo_asr_context* ctx, int token_id) {
    if (!ctx || token_id < 0 || token_id >= (int)ctx->vocab.id_to_token.size())
        return nullptr;
    return ctx->vocab.id_to_token[token_id].c_str();
}

extern "C" int hojo_asr_bos_token_id(struct hojo_asr_context* ctx) {
    return ctx ? (int)ctx->model.hparams.bos_token_id : -1;
}

// HOJO_ASR.infer: max_new_tokens = max(10, min(cfg.max_new_tokens, T_enc*2 + 10)).
extern "C" int hojo_asr_max_new_for_frames(struct hojo_asr_context* ctx, int T_enc) {
    if (!ctx)
        return 0;
    const int cap = ctx->max_new_tokens > 0 ? ctx->max_new_tokens : (int)ctx->model.hparams.gen_max_new_tokens;
    return core_hojo_frames::max_new_for_frames(cap, T_enc);
}

// ===========================================================================
// Embed tokens
// ===========================================================================

extern "C" float* hojo_asr_embed_tokens(struct hojo_asr_context* ctx, const int32_t* token_ids, int n_tokens) {
    if (!ctx || !token_ids || n_tokens <= 0)
        return nullptr;
    const int d = (int)ctx->model.hparams.llm_hidden;

    if (n_tokens == 1 && ctx->model.llm.embed_w) {
        const ggml_tensor* w = ctx->model.llm.embed_w;
        const size_t row_bytes = ggml_row_size(w->type, d);
        float* result = (float*)malloc((size_t)d * sizeof(float));
        if (!result)
            return nullptr;
        std::vector<uint8_t> raw(row_bytes);
        ggml_backend_tensor_get(w, raw.data(), (size_t)token_ids[0] * row_bytes, row_bytes);
        if (w->type == GGML_TYPE_F32)
            std::memcpy(result, raw.data(), (size_t)d * sizeof(float));
        else
            ggml_get_type_traits(w->type)->to_float(raw.data(), result, d);
        return result;
    }

    struct ggml_init_params gp = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gp);
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "token_ids");
    ggml_set_input(ids);
    ggml_tensor* emb = ggml_get_rows(ctx0, ctx->model.llm.embed_w, ids);
    ggml_set_name(emb, "embeds");
    ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        ggml_free(ctx0);
        return nullptr;
    }
    ggml_backend_tensor_set(ids, token_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* result = (float*)malloc((size_t)d * n_tokens * sizeof(float));
    if (result)
        ggml_backend_tensor_get(out, result, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_free(ctx0);
    return result;
}

// ===========================================================================
// Decode
// ===========================================================================

// transformers' RepetitionPenaltyLogitsProcessor, in place:
//   s = logits[t] for every ALREADY-GENERATED token t
//   logits[t] = (s < 0) ? s * penalty : s / penalty
// Note it is applied to `input_ids`, which — because HOJO_ASR generates from
// `inputs_embeds` with no prompt ids — contains ONLY the generated suffix. The
// speech frames and the BOS embedding are not tokens and are never penalised.
static void hojo_asr_apply_repetition_penalty(float* logits, int vocab, const int32_t* generated, int n_generated,
                                              float penalty) {
    core_hojo_frames::apply_repetition_penalty(logits, vocab, generated, n_generated, penalty);
}

static char* hojo_asr_impl(struct hojo_asr_context* ctx, const float* samples, int n_samples, hojo_asr_token_cb on_tok,
                           void* userdata) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d_llm = (int)hp.llm_hidden;
    hojo_asr_bench_stage _b_total("total");

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "hojo_asr: %d samples (%.1f s)\n", n_samples, (float)n_samples / (float)hp.sample_rate);

    // 1. Mel
    int n_mels = 0, T_mel = 0;
    float* mel = nullptr;
    {
        hojo_asr_bench_stage _b("mel");
        mel = hojo_asr_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    }
    if (!mel) {
        fprintf(stderr, "hojo_asr: mel failed\n");
        return nullptr;
    }

    // 2. Encoder
    int T_enc = 0, enc_d = 0;
    float* encoder_out = nullptr;
    {
        hojo_asr_bench_stage _b("encoder");
        encoder_out = hojo_asr_run_encoder(ctx, mel, n_mels, T_mel, &T_enc, &enc_d);
    }
    free(mel);
    if (!encoder_out) {
        fprintf(stderr, "hojo_asr: encoder failed\n");
        return nullptr;
    }
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "hojo_asr: encoder %d frames x %d dims\n", T_enc, enc_d);

    // 3. Conformer adapter (+ ln_speech)
    int adapt_T = 0, adapt_d = 0;
    float* speech = nullptr;
    {
        hojo_asr_bench_stage _b("adapter");
        speech = hojo_asr_run_adapter(ctx, encoder_out, T_enc, enc_d, &adapt_T, &adapt_d, nullptr);
    }
    free(encoder_out);
    if (!speech) {
        fprintf(stderr, "hojo_asr: adapter failed\n");
        return nullptr;
    }

    // 4. inputs_embeds = [embed(<|im_start|>)] ++ speech. No prompt, no template.
    const int n_prompt = adapt_T + 1;
    std::vector<float> inputs_embeds((size_t)d_llm * n_prompt, 0.0f);
    {
        int32_t bos = (int32_t)hp.bos_token_id;
        float* bos_emb = hojo_asr_embed_tokens(ctx, &bos, 1);
        if (!bos_emb) {
            free(speech);
            return nullptr;
        }
        memcpy(inputs_embeds.data(), bos_emb, (size_t)d_llm * sizeof(float));
        free(bos_emb);
        memcpy(inputs_embeds.data() + (size_t)d_llm, speech, (size_t)d_llm * adapt_T * sizeof(float));
    }
    free(speech);

    // 5. KV cache + prefill
    const int max_new = hojo_asr_max_new_for_frames(ctx, adapt_T);
    const int max_ctx = n_prompt + max_new + 1;
    if (ctx->kv_k) {
        if (ctx->kv_max_ctx < max_ctx) {
            if (ctx->kv_buf)
                ggml_backend_buffer_free(ctx->kv_buf);
            if (ctx->kv_ctx)
                ggml_free(ctx->kv_ctx);
            ctx->kv_buf = nullptr;
            ctx->kv_ctx = nullptr;
            ctx->kv_k = nullptr;
            ctx->kv_v = nullptr;
            if (!hojo_asr_kv_init(ctx, max_ctx))
                return nullptr;
        } else {
            hojo_asr_kv_reset(ctx);
        }
    } else if (!hojo_asr_kv_init(ctx, max_ctx)) {
        return nullptr;
    }

    int vocab = 0;
    float* logits = nullptr;
    {
        hojo_asr_bench_stage _b("prefill");
        logits = hojo_asr_run_llm_kv(ctx, inputs_embeds.data(), n_prompt, 0, nullptr, &vocab);
    }
    if (!logits)
        return nullptr;

    const float rep_penalty = hp.gen_repetition_penalty;
    // The checkpoint's recipe is num_beams=4, but `core_beam_decode` rebuilds
    // each beam's KV by REPLAYING its whole suffix every step, so beam search
    // costs O(B*T^2) token-forwards against greedy's O(T). On this 4.4 B
    // decoder that is 80,400 forwards vs 200 for a 9-second clip -- hours
    // rather than minutes. A default nobody can afford to run is not a
    // faithful default, so greedy is the default and `-bs 4` selects the
    // checkpoint's recipe explicitly.
    const int beam = ctx->beam_size > 0 ? ctx->beam_size : 1;

    // 6. Decode
    std::vector<int32_t> generated;
    {
        hojo_asr_bench_stage _b("decode");
        if (beam > 1) {
            // Say what this is about to cost before spending it.
            const long long fwd = (long long)beam * max_new * (max_new + 1) / 2;
            fprintf(stderr,
                    "hojo_asr: beam=%d over <=%d tokens -> %lld token-forwards "
                    "(replay-from-prefix, O(B*T^2)); greedy would be %d. Use -bs 1 "
                    "if this is too slow.\n",
                    beam, max_new, fwd, max_new);
            // The repetition penalty is applied INSIDE replay_fn, over that
            // beam's own suffix — which is exactly the per-beam `input_ids`
            // transformers penalises. Step 0 is unpenalised in both (the
            // sequence is still empty), so the prefill logits pass through.
            auto replay = [&vocab, rep_penalty](hojo_asr_context* c, const int32_t* toks, int n,
                                                int prompt_len) -> float* {
                float* emb = hojo_asr_embed_tokens(c, toks, n);
                if (!emb)
                    return nullptr;
                int dummy = 0;
                float* lg = hojo_asr_run_llm_kv(c, emb, n, prompt_len, &dummy, &vocab);
                std::free(emb);
                if (lg)
                    hojo_asr_apply_repetition_penalty(lg, vocab, toks, n, rep_penalty);
                return lg;
            };
            core_beam_decode::Config cfg;
            cfg.max_new_tokens = max_new;
            cfg.eos_id = (int)hp.eos_token_id;
            cfg.vocab_size = vocab;
            cfg.beam_size = beam;
            cfg.prompt_len = n_prompt;
            auto br = core_beam_decode::run_with_probs(ctx, logits, replay, cfg);
            generated = std::move(br.tokens);
            free(logits);
            logits = nullptr;
            if (on_tok)
                for (size_t i = 0; i < generated.size() && i < br.probs.size(); i++)
                    if (generated[i] != (int)hp.eos_token_id)
                        on_tok(generated[i], br.probs[i], userdata);
            if (!generated.empty() && generated.back() == (int)hp.eos_token_id)
                generated.pop_back();
        } else {
            const bool trace = [] {
                const char* e = crispasr_env::get("CRISPASR_HOJO_ASR_LOGIT_TRACE");
                return e && *e && *e != '0';
            }();
            for (int step = 0; step < max_new; step++) {
                hojo_asr_apply_repetition_penalty(logits, vocab, generated.data(), (int)generated.size(), rep_penalty);

                // Per-step top-3 with the top1-top2 margin. A greedy transcript
                // that differs from the reference by one token is either a real
                // defect or a near-tie that quantisation tipped; only the margin
                // at the divergent step tells the two apart, and nothing else in
                // the pipeline records it.
                if (trace) {
                    int t1 = -1, t2 = -1, t3 = -1;
                    float v1 = -INFINITY, v2 = -INFINITY, v3 = -INFINITY;
                    for (int i = 0; i < vocab; i++) {
                        const float v = logits[i];
                        if (!std::isfinite(v))
                            continue;
                        if (v > v1) {
                            v3 = v2;
                            t3 = t2;
                            v2 = v1;
                            t2 = t1;
                            v1 = v;
                            t1 = i;
                        } else if (v > v2) {
                            v3 = v2;
                            t3 = t2;
                            v2 = v;
                            t2 = i;
                        } else if (v > v3) {
                            v3 = v;
                            t3 = i;
                        }
                    }
                    auto txt = [&](int id) {
                        const char* t = hojo_asr_token_text(ctx, id);
                        return t ? core_bpe::token_bytes_to_utf8(t) : std::string("?");
                    };
                    fprintf(stderr,
                            "hojo_asr_trace: step=%3d margin=%.6f  top1=%d '%s' %.6f  "
                            "top2=%d '%s' %.6f  top3=%d '%s' %.6f\n",
                            step, v1 - v2, t1, txt(t1).c_str(), v1, t2, txt(t2).c_str(), v2, t3, txt(t3).c_str(), v3);
                }
                // NaN-robust argmax: seed -inf, skip non-finite, abort if the
                // whole row is non-finite (a silent all-NaN row would otherwise
                // decode as token 0 forever).
                int best_id = -1;
                float best_val = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < vocab; i++)
                    if (std::isfinite(logits[i]) && logits[i] > best_val) {
                        best_val = logits[i];
                        best_id = i;
                    }
                if (best_id < 0) {
                    free(logits);
                    logits = nullptr;
                    fprintf(stderr, "hojo_asr: non-finite logits at step %d — aborting decode\n", step);
                    break;
                }
                float tok_prob = 0.0f;
                if (on_tok && best_id != (int)hp.eos_token_id) {
                    float s = 0.0f;
                    for (int i = 0; i < vocab; i++)
                        if (std::isfinite(logits[i]))
                            s += expf(logits[i] - best_val);
                    tok_prob = (s > 0.0f) ? (1.0f / s) : 0.0f;
                }
                free(logits);
                logits = nullptr;
                if (ctx->params.verbosity >= 2 && step < 5)
                    fprintf(stderr, "hojo_asr: step %d argmax=%d (%.4f)\n", step, best_id, best_val);
                if (best_id == (int)hp.eos_token_id)
                    break;
                generated.push_back(best_id);
                if (on_tok)
                    on_tok(best_id, tok_prob, userdata);
                float* next_emb = hojo_asr_embed_tokens(ctx, &best_id, 1);
                if (!next_emb)
                    break;
                int dummy = 0;
                logits = hojo_asr_run_llm_kv(ctx, next_emb, 1, n_prompt + (int)generated.size() - 1, &dummy, &vocab);
                free(next_emb);
                if (!logits)
                    break;
            }
            if (logits)
                free(logits);
        }
    }

    // 7. Detokenize
    std::string result;
    for (int id : generated) {
        const char* t = hojo_asr_token_text(ctx, id);
        if (t)
            result += core_bpe::token_bytes_to_utf8(t);
    }
    // HOJO_ASR.run_infer strips these two before returning.
    for (const char* marker : {"<|im_end|>", "<|endoftext|>"}) {
        size_t p;
        while ((p = result.find(marker)) != std::string::npos)
            result.erase(p, strlen(marker));
    }
    {
        size_t b = result.find_first_not_of(" \t\r\n");
        size_t e = result.find_last_not_of(" \t\r\n");
        result = (b == std::string::npos) ? std::string() : result.substr(b, e - b + 1);
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "hojo_asr: %zu tokens: \"%s\"\n", generated.size(), result.substr(0, 120).c_str());

    // Collapse degenerate n-gram loops (issue #218). No-op on clean output; set
    // CRISPASR_HOJO_ASR_NO_LOOPFIX=1 for raw upstream-parity text.
    {
        const char* no_fix = crispasr_env::get("CRISPASR_HOJO_ASR_NO_LOOPFIX");
        if (!(no_fix && no_fix[0] == '1')) {
            std::string fixed = core_ngram::fix_loops(result);
            if (fixed != result && ctx->params.verbosity >= 1)
                fprintf(stderr, "hojo_asr: collapsed n-gram loop(s) (%zu -> %zu chars)\n", result.size(), fixed.size());
            result = std::move(fixed);
        }
    }

    char* out = (char*)malloc(result.size() + 1);
    if (!out)
        return nullptr;
    memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

extern "C" char* hojo_asr_transcribe(struct hojo_asr_context* ctx, const float* samples, int n_samples) {
    return hojo_asr_impl(ctx, samples, n_samples, nullptr, nullptr);
}

extern "C" void hojo_asr_transcribe_cb(struct hojo_asr_context* ctx, const float* samples, int n_samples,
                                       hojo_asr_token_cb cb, void* userdata) {
    char* s = hojo_asr_impl(ctx, samples, n_samples, cb, userdata);
    free(s);
}

// ===========================================================================
// Init / Free
// ===========================================================================

extern "C" struct hojo_asr_context_params hojo_asr_context_default_params(void) {
    hojo_asr_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.flash_attn = false;
    return p;
}

extern "C" struct hojo_asr_context* hojo_asr_init_from_file(const char* path_model,
                                                            struct hojo_asr_context_params params) {
    auto* ctx = new hojo_asr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads;
    ctx->model_path = path_model ? path_model : "";

    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : core_cpu_backend::init();
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();
    ctx->backend_cpu = core_cpu_backend::init();
    if (ctx->backend_cpu)
        core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->n_threads);

    {
        const char* force_cpu = crispasr_env::get("CRISPASR_HOJO_ASR_FORCE_CPU");
        if (force_cpu && force_cpu[0] == '1')
            ctx->backend = ctx->backend_cpu;
    }
    if (core_cpu_backend::is_cpu(ctx->backend))
        core_cpu_backend::set_n_threads(ctx->backend, ctx->n_threads);

    // Encoder attention path (issue #215): flash_attn_ext mismanages command
    // buffers on Vulkan, so that backend uses the manual soft_max_ext path —
    // the same op sequence the LM decode already runs safely there.
    {
        const char* force_flash = crispasr_env::get("CRISPASR_HOJO_ASR_ENC_FLASH");
        const char* force_manual = crispasr_env::get("CRISPASR_HOJO_ASR_ENC_MANUAL");
        if (force_flash && force_flash[0] == '1') {
            ctx->enc_use_flash = true;
        } else if (force_manual && force_manual[0] == '1') {
            ctx->enc_use_flash = false;
        } else {
            ctx->enc_use_flash = !backend_is_vulkan(ctx->backend);
        }
        if (!ctx->enc_use_flash && backend_is_vulkan(ctx->backend))
            fprintf(stderr, "hojo_asr: Vulkan backend detected — encoder using manual soft_max_ext "
                            "attention (issue #215; CRISPASR_HOJO_ASR_ENC_FLASH=1 overrides)\n");
    }

    if (!hojo_asr_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        fprintf(stderr, "hojo_asr: failed to load model from %s\n", path_model ? path_model : "(null)");
        hojo_asr_free(ctx);
        return nullptr;
    }

    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
        crispasr_imatrix_install(ctx->sched); // no-op unless CRISPASR_IMATRIX_OUT is set
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    return ctx;
}

extern "C" void hojo_asr_free(struct hojo_asr_context* ctx) {
    if (!ctx)
        return;
    if (ctx->conv_ctx)
        ggml_free(ctx->conv_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        core_gguf::release_weight_buffer(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// <= 0 means greedy. The checkpoint's recipe is num_beams=4 and it is still one
// flag away (-bs 4), but replay-from-prefix makes it O(B*T^2) — see hojo_asr.h.
extern "C" void hojo_asr_set_beam_size(struct hojo_asr_context* ctx, int beam_size) {
    if (ctx)
        ctx->beam_size = beam_size > 0 ? beam_size : 0;
}

// #292: forward the user's --max-new-tokens. <= 0 keeps the checkpoint's cap.
extern "C" void hojo_asr_set_max_new_tokens(struct hojo_asr_context* ctx, int max_new_tokens) {
    if (ctx && max_new_tokens > 0)
        ctx->max_new_tokens = max_new_tokens;
}
