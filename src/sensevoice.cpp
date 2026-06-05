// sensevoice.cpp — FunAudioLLM/SenseVoiceSmall ggml runtime.
//
// See sensevoice.h for the public contract and pipeline overview. The
// stage names exposed via sensevoice_extract_stage match the reference
// dumper at tools/reference_backends/sensevoice.py exactly.

#include "sensevoice.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include "core/ctc.h"
#include "core/gguf_loader.h"
#include "core/kaldi_fbank.h"
#include "core/lfr.h"
#include "core/sanm.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation — `SENSEVOICE_BENCH=1` for per-stage timings.
// ===========================================================================

static bool sensevoice_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("SENSEVOICE_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct sensevoice_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit sensevoice_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~sensevoice_bench_stage() {
        if (!sensevoice_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  sensevoice_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct sensevoice_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 80;
    uint32_t frame_length_ms = 25;
    uint32_t frame_shift_ms = 10;
    uint32_t lfr_m = 7;
    uint32_t lfr_n = 6;

    // Encoder (SenseVoiceEncoderSmall)
    uint32_t input_size = 560; // n_mels * lfr_m
    uint32_t d_model = 512;
    uint32_t n_heads = 4;
    uint32_t head_dim = 128;
    uint32_t ffn_dim = 2048;
    uint32_t n_blocks_base = 50;
    uint32_t n_blocks_tp = 20;
    uint32_t sanm_kernel = 11;
    float enc_ln_eps = 1e-5f;

    // CTC head
    uint32_t vocab_size = 25055;
    uint32_t blank_id = 0;
    uint32_t sos_id = 1;
    uint32_t eos_id = 2;

    // Query token IDs (indices into the 16-row query_embed table).
    // Match funasr/models/sense_voice/model.py SenseVoiceSmall.__init__.
    uint32_t lang_auto = 0;
    uint32_t lang_zh = 3;
    uint32_t lang_en = 4;
    uint32_t lang_yue = 7;
    uint32_t lang_ja = 11;
    uint32_t lang_ko = 12;
    uint32_t lang_nospeech = 13;
    uint32_t event_query = 1;
    uint32_t emo_query = 2;
    uint32_t textnorm_withitn = 14;
    uint32_t textnorm_woitn = 15;
};

// ===========================================================================
// Tensors
// ===========================================================================

struct sensevoice_enc_block {
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    ggml_tensor *attn_qkv_w = nullptr, *attn_qkv_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    ggml_tensor* attn_fsmn_w = nullptr;
    ggml_tensor *ffn_l1_w = nullptr, *ffn_l1_b = nullptr;
    ggml_tensor *ffn_l2_w = nullptr, *ffn_l2_b = nullptr;
};

struct sensevoice_encoder {
    std::vector<sensevoice_enc_block> blocks;
    ggml_tensor *after_norm_w = nullptr, *after_norm_b = nullptr;
    ggml_tensor *tp_norm_w = nullptr, *tp_norm_b = nullptr;
};

struct sensevoice_model {
    sensevoice_hparams hparams;
    sensevoice_encoder enc;
    ggml_tensor* query_embed_w = nullptr; // (16, 560)
    ggml_tensor* ctc_w = nullptr;         // (25055, 512)
    ggml_tensor* ctc_b = nullptr;         // (25055,)

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_buffer_t buf_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Sinusoidal PE precomputed at load time. Depth = input_size = 560.
    std::vector<float> enc_pe;
    int enc_pe_max_T = 0;
};

struct sensevoice_vocab {
    std::vector<std::string> id_to_token; // 25055 SentencePiece pieces
};

struct sensevoice_context {
    sensevoice_context_params params;
    sensevoice_model model;
    sensevoice_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;
    int n_threads = 4;

    bool enc_flash_attn = true;
    std::string requested_stage;

    int beam_size = 1;       // CTC beam search (1 = greedy)
    float beam_gamma = 0.0f; // gamma-threshold pruning (0 = off)
};

// ===========================================================================
// Loader
// ===========================================================================

static ggml_tensor* require(sensevoice_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "sensevoice");
}

static void compute_encoder_pe(sensevoice_model& m, int max_T) {
    // SenseVoice's SinusoidalPositionEncoder: positions = arange(1, T+1),
    // depth = input_size = 560. pe[t, :half] = sin((t+1)*inv_t),
    // pe[t, half:] = cos((t+1)*inv_t).
    const int D = (int)m.hparams.input_size;
    const int half = D / 2;
    const float log_inc = std::log(10000.0f) / (float)(half - 1);
    std::vector<float> inv_t((size_t)half);
    for (int i = 0; i < half; i++)
        inv_t[(size_t)i] = std::exp(-log_inc * (float)i);
    m.enc_pe.assign((size_t)max_T * (size_t)D, 0.0f);
    for (int t = 0; t < max_T; t++) {
        const float pos = (float)(t + 1);
        float* row = m.enc_pe.data() + (size_t)t * (size_t)D;
        for (int i = 0; i < half; i++) {
            const float a = pos * inv_t[(size_t)i];
            row[i] = std::sin(a);
            row[half + i] = std::cos(a);
        }
    }
    m.enc_pe_max_T = max_T;
}

static bool sensevoice_load_model(sensevoice_model& model, sensevoice_vocab& vocab, const char* path,
                                  ggml_backend_t backend, ggml_backend_t /*backend_cpu*/) {
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;
        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "sensevoice.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "sensevoice.n_mels", hp.n_mels);
        hp.frame_length_ms = core_gguf::kv_u32(gctx, "sensevoice.frame_length_ms", hp.frame_length_ms);
        hp.frame_shift_ms = core_gguf::kv_u32(gctx, "sensevoice.frame_shift_ms", hp.frame_shift_ms);
        hp.lfr_m = core_gguf::kv_u32(gctx, "sensevoice.lfr_m", hp.lfr_m);
        hp.lfr_n = core_gguf::kv_u32(gctx, "sensevoice.lfr_n", hp.lfr_n);
        hp.d_model = core_gguf::kv_u32(gctx, "sensevoice.d_model", hp.d_model);
        hp.n_heads = core_gguf::kv_u32(gctx, "sensevoice.n_heads", hp.n_heads);
        hp.ffn_dim = core_gguf::kv_u32(gctx, "sensevoice.ffn_dim", hp.ffn_dim);
        hp.n_blocks_base = core_gguf::kv_u32(gctx, "sensevoice.n_blocks_base", hp.n_blocks_base);
        hp.n_blocks_tp = core_gguf::kv_u32(gctx, "sensevoice.n_blocks_tp", hp.n_blocks_tp);
        hp.sanm_kernel = core_gguf::kv_u32(gctx, "sensevoice.sanm_kernel", hp.sanm_kernel);
        hp.vocab_size = core_gguf::kv_u32(gctx, "sensevoice.vocab_size", hp.vocab_size);
        hp.blank_id = core_gguf::kv_u32(gctx, "sensevoice.blank_id", hp.blank_id);
        hp.sos_id = core_gguf::kv_u32(gctx, "sensevoice.sos_id", hp.sos_id);
        hp.eos_id = core_gguf::kv_u32(gctx, "sensevoice.eos_id", hp.eos_id);
        hp.input_size = hp.n_mels * hp.lfr_m;
        hp.head_dim = hp.d_model / hp.n_heads;
        hp.lang_zh = core_gguf::kv_u32(gctx, "sensevoice.lang_zh", hp.lang_zh);
        hp.lang_en = core_gguf::kv_u32(gctx, "sensevoice.lang_en", hp.lang_en);
        hp.lang_yue = core_gguf::kv_u32(gctx, "sensevoice.lang_yue", hp.lang_yue);
        hp.lang_ja = core_gguf::kv_u32(gctx, "sensevoice.lang_ja", hp.lang_ja);
        hp.lang_ko = core_gguf::kv_u32(gctx, "sensevoice.lang_ko", hp.lang_ko);
        hp.event_query = core_gguf::kv_u32(gctx, "sensevoice.event_query", hp.event_query);
        hp.emo_query = core_gguf::kv_u32(gctx, "sensevoice.emo_query", hp.emo_query);
        hp.textnorm_withitn = core_gguf::kv_u32(gctx, "sensevoice.textnorm_withitn", hp.textnorm_withitn);
        hp.textnorm_woitn = core_gguf::kv_u32(gctx, "sensevoice.textnorm_woitn", hp.textnorm_woitn);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty())
            vocab.id_to_token = std::move(tokens);

        core_gguf::free_metadata(gctx);
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "sensevoice", wl))
        return false;
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.buf_cpu = wl.buf_cpu;
    model.tensors = std::move(wl.tensors);

    const auto& hp = model.hparams;
    const int n_enc = (int)(hp.n_blocks_base + hp.n_blocks_tp);
    model.enc.blocks.resize((size_t)n_enc);
    for (int i = 0; i < n_enc; i++) {
        char buf[128];
        auto& b = model.enc.blocks[(size_t)i];
        auto get = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "sensevoice.enc.blk.%d.%s", i, suf);
            return require(model, buf);
        };
        b.norm1_w = get("norm1.w");
        b.norm1_b = get("norm1.b");
        b.norm2_w = get("norm2.w");
        b.norm2_b = get("norm2.b");
        b.attn_qkv_w = get("attn.qkv.w");
        b.attn_qkv_b = get("attn.qkv.b");
        b.attn_out_w = get("attn.out.w");
        b.attn_out_b = get("attn.out.b");
        b.attn_fsmn_w = get("attn.fsmn.w");
        b.ffn_l1_w = get("ffn.l1.w");
        b.ffn_l1_b = get("ffn.l1.b");
        b.ffn_l2_w = get("ffn.l2.w");
        b.ffn_l2_b = get("ffn.l2.b");
    }
    model.enc.after_norm_w = require(model, "sensevoice.enc.after_norm.w");
    model.enc.after_norm_b = require(model, "sensevoice.enc.after_norm.b");
    model.enc.tp_norm_w = require(model, "sensevoice.enc.tp_norm.w");
    model.enc.tp_norm_b = require(model, "sensevoice.enc.tp_norm.b");
    model.query_embed_w = require(model, "sensevoice.query_embed.w");
    model.ctc_w = require(model, "sensevoice.ctc.w");
    model.ctc_b = require(model, "sensevoice.ctc.b");

    compute_encoder_pe(model, 8192);
    return true;
}

// ===========================================================================
// Frontend — kaldi-fbank + LFR
// ===========================================================================

static std::vector<float> sensevoice_compute_features(sensevoice_context* ctx, const float* pcm, int n_samples,
                                                      int& T_lfr_out, int& D_lfr_out) {
    const auto& hp = ctx->model.hparams;
    core_kaldi::FbankParams fp;
    fp.sample_rate = (int)hp.sample_rate;
    fp.n_mels = (int)hp.n_mels;
    fp.frame_length_ms = (int)hp.frame_length_ms;
    fp.frame_shift_ms = (int)hp.frame_shift_ms;
    fp.int16_scale = true;
    fp.window_type = core_kaldi::WindowType::Hamming;

    int T = 0;
    std::vector<float> mel = core_kaldi::compute_fbank(pcm, n_samples, fp, T);
    if (T == 0)
        return {};
    int T_lfr = 0;
    std::vector<float> lfr = core_lfr::stack(mel.data(), T, (int)hp.n_mels, (int)hp.lfr_m, (int)hp.lfr_n, T_lfr);
    T_lfr_out = T_lfr;
    D_lfr_out = (int)hp.input_size;
    return lfr;
}

// ===========================================================================
// Build encoder graph: prepend 4 query embeds, run 70 SANM blocks, CTC head.
// ===========================================================================

static ggml_tensor* maybe_snap(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* t, const char* name) {
    ggml_tensor* s = ggml_dup(ctx0, t);
    ggml_set_name(s, name);
    ggml_build_forward_expand(gf, s);
    return t;
}

static ggml_cgraph* sensevoice_build_graph(sensevoice_context* ctx, int T_lfr, int T_total) {
    // T_total = T_lfr + 4 (the four prepended query embeds).
    const auto& hp = ctx->model.hparams;
    const int D_in = (int)hp.input_size;
    const int D = (int)hp.d_model;
    const int vocab = (int)hp.vocab_size;
    const int K = (int)hp.sanm_kernel;
    const int n_heads = (int)hp.n_heads;
    const int hd = (int)hp.head_dim;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Inputs:
    //   query_embeds : (D_in, 4) — already-looked-up rows from query_embed.w
    //   mel_in       : (D_in, T_lfr) — LFR fbank features
    //   pe_in        : (D_in, T_total) — sinusoidal PE for the concatenated sequence
    ggml_tensor* query_embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D_in, 4);
    ggml_set_name(query_embeds, "query_embeds");
    ggml_set_input(query_embeds);

    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D_in, T_lfr);
    ggml_set_name(mel_in, "mel_features");
    ggml_set_input(mel_in);

    ggml_tensor* pe_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D_in, T_total);
    ggml_set_name(pe_in, "enc_pe");
    ggml_set_input(pe_in);

    // Concat: [query_embeds, mel_in] → (D_in, T_total).
    ggml_tensor* cur = ggml_concat(ctx0, query_embeds, mel_in, /*dim*/ 1);
    cur = maybe_snap(ctx0, gf, cur, "encoder_input");

    // SenseVoiceEncoderSmall.forward: xs *= sqrt(d_model), then add PE.
    cur = ggml_scale(ctx0, cur, std::sqrt((float)D));
    cur = ggml_add(ctx0, cur, pe_in);

    // 70 SANM blocks (1 entry @ in_size=D_in, 49 main + 20 tp @ in_size=D).
    const int n_base = (int)hp.n_blocks_base;
    const int n_tp = (int)hp.n_blocks_tp;
    for (int i = 0; i < n_base + n_tp; i++) {
        const auto& src = ctx->model.enc.blocks[(size_t)i];
        core_sanm::BlockWeights w;
        w.norm1_w = src.norm1_w;
        w.norm1_b = src.norm1_b;
        w.norm2_w = src.norm2_w;
        w.norm2_b = src.norm2_b;
        w.attn_qkv_w = src.attn_qkv_w;
        w.attn_qkv_b = src.attn_qkv_b;
        w.attn_out_w = src.attn_out_w;
        w.attn_out_b = src.attn_out_b;
        w.attn_fsmn_w = src.attn_fsmn_w;
        w.ffn_l1_w = src.ffn_l1_w;
        w.ffn_l1_b = src.ffn_l1_b;
        w.ffn_l2_w = src.ffn_l2_w;
        w.ffn_l2_b = src.ffn_l2_b;
        core_sanm::BlockParams p;
        p.in_size = (i == 0) ? D_in : D;
        p.size = D;
        p.n_heads = n_heads;
        p.head_dim = hd;
        p.kernel = K;
        p.ln_eps = hp.enc_ln_eps;
        p.flash_attn = ctx->enc_flash_attn;
        const bool attn_residual = (p.in_size == p.size);
        cur = core_sanm::build_block(ctx0, cur, T_total, w, p, attn_residual);

        char nm[32];
        std::snprintf(nm, sizeof(nm), "encoder_layer_%d", i);
        cur = maybe_snap(ctx0, gf, cur, nm);

        if (i == n_base - 1) {
            cur = ggml_norm_affine(ctx0, cur, ctx->model.enc.after_norm_w, ctx->model.enc.after_norm_b, hp.enc_ln_eps);
            cur = maybe_snap(ctx0, gf, cur, "encoder_main_out");
        }
    }
    cur = ggml_norm_affine(ctx0, cur, ctx->model.enc.tp_norm_w, ctx->model.enc.tp_norm_b, hp.enc_ln_eps);
    cur = maybe_snap(ctx0, gf, cur, "encoder_output");

    // CTC head: logits = ctc_w @ enc_out + ctc_b. Result (vocab, T_total).
    ggml_tensor* logits = ggml_mul_mat(ctx0, ctx->model.ctc_w, cur);
    logits = ggml_add(ctx0, logits, ctx->model.ctc_b);
    ggml_set_name(logits, "ctc_logits");
    ggml_build_forward_expand(gf, logits);

    (void)vocab;
    ggml_free(ctx0);
    return gf;
}

// Map a language string to the query-embed row id.
static int sensevoice_lang_id(const sensevoice_hparams& hp, const char* language) {
    if (!language)
        return (int)hp.lang_auto;
    std::string s(language);
    if (s == "auto" || s.empty())
        return (int)hp.lang_auto;
    if (s == "zh")
        return (int)hp.lang_zh;
    if (s == "en")
        return (int)hp.lang_en;
    if (s == "yue")
        return (int)hp.lang_yue;
    if (s == "ja")
        return (int)hp.lang_ja;
    if (s == "ko")
        return (int)hp.lang_ko;
    if (s == "nospeech")
        return (int)hp.lang_nospeech;
    return (int)hp.lang_auto;
}

// Compute the 4-row (D_in, 4) query embedding buffer by gathering rows
// from model.query_embed_w. The PyTorch table is (16, D_in), stored in
// GGUF with ne[0]=D_in, ne[1]=16 — so each "row" in the original sense
// is one column in our F16 buffer at offset row_id * D_in * sizeof(F16).
static std::vector<float> sensevoice_gather_query_rows(sensevoice_context* ctx, const char* language, bool use_itn) {
    const auto& hp = ctx->model.hparams;
    const int D_in = (int)hp.input_size;
    const int lang_id = sensevoice_lang_id(hp, language);
    const int textnorm_id = use_itn ? (int)hp.textnorm_withitn : (int)hp.textnorm_woitn;
    const int row_ids[4] = {lang_id, (int)hp.event_query, (int)hp.emo_query, textnorm_id};

    std::vector<float> out((size_t)4 * (size_t)D_in, 0.0f);
    ggml_tensor* emb = ctx->model.query_embed_w;
    // emb->type is F16, layout (D_in, 16). Read row by row.
    const size_t row_bytes = (size_t)D_in * ggml_type_size(emb->type);
    std::vector<uint8_t> tmp((size_t)D_in * sizeof(uint16_t));
    for (int r = 0; r < 4; r++) {
        const int rid = row_ids[r];
        ggml_backend_tensor_get(emb, tmp.data(), (size_t)rid * row_bytes, row_bytes);
        if (emb->type == GGML_TYPE_F16) {
            const ggml_fp16_t* src = (const ggml_fp16_t*)tmp.data();
            for (int i = 0; i < D_in; i++)
                out[(size_t)r * D_in + i] = ggml_fp16_to_fp32(src[i]);
        } else if (emb->type == GGML_TYPE_F32) {
            std::memcpy(out.data() + (size_t)r * D_in, tmp.data(), row_bytes);
        }
    }
    return out;
}

// ===========================================================================
// CTC greedy decode + SentencePiece-piece detokenize
// ===========================================================================

static std::vector<int32_t> sensevoice_ctc_greedy(const float* logits, int T_total, int vocab, int blank_id) {
    std::vector<int32_t> raw((size_t)T_total);
    for (int t = 0; t < T_total; t++) {
        const float* row = logits + (size_t)t * vocab;
        int best = 0;
        float bv = row[0];
        for (int v = 1; v < vocab; v++) {
            if (row[v] > bv) {
                bv = row[v];
                best = v;
            }
        }
        raw[(size_t)t] = best;
    }
    // unique_consecutive — drop adjacent duplicates first
    std::vector<int32_t> uc;
    uc.reserve(raw.size());
    int32_t prev = -1;
    for (int32_t id : raw) {
        if (id != prev)
            uc.push_back(id);
        prev = id;
    }
    // then drop blank_id
    std::vector<int32_t> kept;
    kept.reserve(uc.size());
    for (int32_t id : uc)
        if (id != (int32_t)blank_id)
            kept.push_back(id);
    return kept;
}

// SentencePiece detokenise: each piece is a UTF-8 string that may start
// with the leading-space marker `▁` (U+2581, "\xE2\x96\x81"). Convert
// `▁` → space and concatenate. Special control tokens (the CJK / emoji
// rich-annotation prefix) come through as their literal piece strings,
// which we keep verbatim — same shape as the upstream `tokenizer.decode`
// output.
static std::string sensevoice_detokenize(const sensevoice_vocab& vocab, const std::vector<int32_t>& ids) {
    static const std::string SPM_BOUNDARY = "\xE2\x96\x81";
    std::string out;
    out.reserve(ids.size() * 4);
    for (int32_t id : ids) {
        if (id < 0 || id >= (int)vocab.id_to_token.size())
            continue;
        const std::string& p = vocab.id_to_token[(size_t)id];
        size_t pos = 0;
        while (true) {
            size_t found = p.find(SPM_BOUNDARY, pos);
            if (found == std::string::npos) {
                out.append(p, pos, std::string::npos);
                break;
            }
            out.append(p, pos, found - pos);
            out.push_back(' ');
            pos = found + SPM_BOUNDARY.size();
        }
    }
    // Trim leading space (artefact of the first piece being ▁-prefixed).
    size_t start = out.find_first_not_of(' ');
    if (start == std::string::npos)
        return "";
    return out.substr(start);
}

// ===========================================================================
// High-level pipeline
// ===========================================================================

static std::string sensevoice_transcribe_impl(sensevoice_context* ctx, const float* pcm, int n_samples,
                                              const char* language, bool use_itn, std::vector<float>* stage_out,
                                              const char* stage_name) {
    const auto& hp = ctx->model.hparams;

    int T_lfr = 0, D_lfr = 0;
    std::vector<float> lfr;
    {
        sensevoice_bench_stage s("fbank+lfr");
        lfr = sensevoice_compute_features(ctx, pcm, n_samples, T_lfr, D_lfr);
    }
    if (T_lfr <= 0)
        return "";
    if (stage_out && stage_name && std::strcmp(stage_name, "mel_features") == 0) {
        stage_out->assign(lfr.begin(), lfr.begin() + (ptrdiff_t)D_lfr * T_lfr);
        return "";
    }

    const int T_total = T_lfr + 4;
    if (T_total > ctx->model.enc_pe_max_T) {
        compute_encoder_pe(ctx->model, T_total + 256);
    }

    std::vector<float> query_rows;
    {
        sensevoice_bench_stage s("query_embed");
        query_rows = sensevoice_gather_query_rows(ctx, language, use_itn);
    }

    std::vector<float> logits;
    {
        sensevoice_bench_stage s("encoder+ctc");
        ggml_cgraph* gf = sensevoice_build_graph(ctx, T_lfr, T_total);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            std::fprintf(stderr, "sensevoice: failed to alloc encoder graph\n");
            return "";
        }
        ggml_tensor* q_in = ggml_graph_get_tensor(gf, "query_embeds");
        ggml_backend_tensor_set(q_in, query_rows.data(), 0, query_rows.size() * sizeof(float));
        ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel_features");
        ggml_backend_tensor_set(mel_in, lfr.data(), 0, (size_t)D_lfr * (size_t)T_lfr * sizeof(float));
        ggml_tensor* pe_in = ggml_graph_get_tensor(gf, "enc_pe");
        ggml_backend_tensor_set(pe_in, ctx->model.enc_pe.data(), 0, (size_t)D_lfr * (size_t)T_total * sizeof(float));

        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "sensevoice: encoder graph compute failed\n");
            return "";
        }

        if (stage_out && stage_name) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, stage_name);
            if (t) {
                const size_t n = ggml_nelements(t);
                stage_out->assign(n, 0.0f);
                ggml_backend_tensor_get(t, stage_out->data(), 0, n * sizeof(float));
                if (std::strncmp(stage_name, "encoder_", 8) == 0)
                    return "";
            }
        }

        ggml_tensor* out = ggml_graph_get_tensor(gf, "ctc_logits");
        const int vocab = (int)hp.vocab_size;
        logits.assign((size_t)vocab * (size_t)T_total, 0.0f);
        ggml_backend_tensor_get(out, logits.data(), 0, logits.size() * sizeof(float));

        if (stage_out && stage_name && std::strcmp(stage_name, "ctc_logits") == 0) {
            *stage_out = logits;
            return "";
        }
    }

    std::vector<int32_t> ids;
    if (ctx->beam_size > 1) {
        // CTC prefix beam search with optional gamma pruning.
        // Convert raw logits to log-softmax first.
        const int V = (int)hp.vocab_size;
        std::vector<float> logprobs((size_t)T_total * V);
        for (int t = 0; t < T_total; t++) {
            const float* row = logits.data() + (size_t)t * V;
            float* lp = logprobs.data() + (size_t)t * V;
            float mx = *std::max_element(row, row + V);
            double sum = 0.0;
            for (int v = 0; v < V; v++)
                sum += std::exp((double)(row[v] - mx));
            double log_sum = (double)mx + std::log(sum);
            for (int v = 0; v < V; v++)
                lp[v] = (float)((double)row[v] - log_sum);
        }
        auto br = core_ctc::prefix_beam_search(logprobs.data(), T_total, V, (int)hp.blank_id, /*shift=*/0,
                                               ctx->beam_size, ctx->beam_gamma);
        ids = std::move(br.tokens);
    } else {
        ids = sensevoice_ctc_greedy(logits.data(), T_total, (int)hp.vocab_size, (int)hp.blank_id);
    }
    return sensevoice_detokenize(ctx->vocab, ids);
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" sensevoice_context_params sensevoice_context_default_params(void) {
    sensevoice_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

extern "C" sensevoice_context* sensevoice_init_from_file(const char* path, sensevoice_context_params params) {
    sensevoice_context* ctx = new sensevoice_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!sensevoice_load_model(ctx->model, ctx->vocab, path, ctx->backend, ctx->backend_cpu)) {
        delete ctx;
        return nullptr;
    }

    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (const char* s = std::getenv("SENSEVOICE_NO_FA")) {
        if (*s && *s != '0')
            ctx->enc_flash_attn = false;
    }

    if (params.verbosity >= 1) {
        std::fprintf(stderr, "sensevoice: loaded %s  (enc %u + tp %u blocks, vocab %u, fa=%s)\n", path,
                     ctx->model.hparams.n_blocks_base, ctx->model.hparams.n_blocks_tp, ctx->model.hparams.vocab_size,
                     ctx->enc_flash_attn ? "on" : "off");
    }
    return ctx;
}

extern "C" void sensevoice_set_beam_size(sensevoice_context* ctx, int beam_size, float gamma) {
    if (!ctx)
        return;
    ctx->beam_size = beam_size > 1 ? beam_size : 1;
    ctx->beam_gamma = gamma > 0.0f ? gamma : 0.0f;
}

extern "C" void sensevoice_free(sensevoice_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.buf_cpu)
        ggml_backend_buffer_free(ctx->model.buf_cpu);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" char* sensevoice_transcribe(sensevoice_context* ctx, const float* samples, int n_samples,
                                       const char* language, bool use_itn) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    std::string s = sensevoice_transcribe_impl(ctx, samples, n_samples, language, use_itn, nullptr, nullptr);
    char* out = (char*)std::malloc(s.size() + 1);
    if (!out)
        return nullptr;
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

namespace {

// Peel the leading `<|...|>` markers off a SenseVoice transcript and
// classify them by content. Upstream emits four markers (language,
// emotion, audio-event, itn) but their *order* is determined by training,
// not by the input query-embed order. Content classification is robust
// to ordering surprises and to the model dropping a marker on degenerate
// audio.
struct sv_prefix {
    std::string language;
    std::string emotion;
    std::string audio_event;
    std::string itn;
    size_t consumed = 0;
};

static sv_prefix sv_parse_prefix(const std::string& s) {
    sv_prefix p;
    size_t pos = 0;
    std::vector<std::string> markers;
    while (pos < s.size() && markers.size() < 4) {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t'))
            ++pos;
        if (pos + 2 > s.size() || s[pos] != '<' || s[pos + 1] != '|')
            break;
        size_t end = s.find("|>", pos + 2);
        if (end == std::string::npos)
            break;
        markers.emplace_back(s.substr(pos + 2, end - (pos + 2)));
        pos = end + 2;
    }
    p.consumed = pos;

    static const std::unordered_set<std::string> LANG{"auto", "zh", "en", "yue",      "ja",
                                                      "ko",   "nl", "de", "nospeech", "unk"};
    static const std::unordered_set<std::string> EMO{"HAPPY",   "SAD",       "ANGRY",     "NEUTRAL",
                                                     "FEARFUL", "DISGUSTED", "SURPRISED", "EMO_UNKNOWN"};
    static const std::unordered_set<std::string> ITN{"withitn", "woitn"};

    for (const auto& m : markers) {
        if (LANG.count(m))
            p.language = m;
        else if (EMO.count(m))
            p.emotion = m;
        else if (ITN.count(m))
            p.itn = m;
        else
            // Anything else falls into audio_event (Speech, Music, BGM,
            // Laughter, Cough, ... — the upstream set is open-ended).
            p.audio_event = m;
    }
    return p;
}

static char* sv_dup(const std::string& s) {
    char* out = (char*)std::malloc(s.size() + 1);
    if (!out)
        return nullptr;
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

} // namespace

extern "C" sensevoice_result* sensevoice_transcribe_structured(sensevoice_context* ctx, const float* samples,
                                                               int n_samples, const char* language, bool use_itn) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    std::string raw = sensevoice_transcribe_impl(ctx, samples, n_samples, language, use_itn, nullptr, nullptr);
    sv_prefix pre = sv_parse_prefix(raw);

    std::string text = raw.substr(pre.consumed);
    // Trim leading whitespace artefact from SentencePiece ▁ on the first
    // post-prefix token.
    size_t lead = text.find_first_not_of(" \t");
    if (lead != std::string::npos && lead > 0)
        text.erase(0, lead);

    auto* r = (sensevoice_result*)std::calloc(1, sizeof(sensevoice_result));
    if (!r)
        return nullptr;
    r->language = sv_dup(pre.language);
    r->emotion = sv_dup(pre.emotion);
    r->audio_event = sv_dup(pre.audio_event);
    r->itn = sv_dup(pre.itn);
    r->text = sv_dup(text);
    r->raw = sv_dup(raw);
    if (!r->language || !r->emotion || !r->audio_event || !r->itn || !r->text || !r->raw) {
        sensevoice_result_free(r);
        return nullptr;
    }
    return r;
}

extern "C" void sensevoice_result_free(sensevoice_result* r) {
    if (!r)
        return;
    std::free(r->language);
    std::free(r->emotion);
    std::free(r->audio_event);
    std::free(r->itn);
    std::free(r->text);
    std::free(r->raw);
    std::free(r);
}

extern "C" float* sensevoice_extract_stage(sensevoice_context* ctx, const float* samples, int n_samples,
                                           const char* language, bool use_itn, const char* stage_name, int* n_out) {
    if (n_out)
        *n_out = 0;
    if (!ctx || !samples || n_samples <= 0 || !stage_name)
        return nullptr;
    ctx->requested_stage = stage_name;

    if (std::strcmp(stage_name, "generated_text") == 0) {
        std::string txt = sensevoice_transcribe_impl(ctx, samples, n_samples, language, use_itn, nullptr, nullptr);
        char* buf = (char*)std::malloc(txt.size() + 1);
        if (!buf)
            return nullptr;
        std::memcpy(buf, txt.data(), txt.size());
        buf[txt.size()] = '\0';
        if (n_out)
            *n_out = (int)txt.size();
        // cppcheck-suppress invalidPointerCast
        return (float*)buf; // caller casts back to char* (generated_text path)
    }

    std::vector<float> staged;
    (void)sensevoice_transcribe_impl(ctx, samples, n_samples, language, use_itn, &staged, stage_name);
    if (staged.empty())
        return nullptr;
    float* out = (float*)std::malloc(staged.size() * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, staged.data(), staged.size() * sizeof(float));
    if (n_out)
        *n_out = (int)staged.size();
    return out;
}
