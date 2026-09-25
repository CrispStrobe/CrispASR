// canary.cpp — NVIDIA Canary AED ggml runtime
//
// Supports the legacy cstr/canary-1b-v2 layout and the metadata-driven
// transcribe.cpp Canary GGUF schema, including Canary 180M Flash.
//
// Shared architecture:
//   Mel:           128 mels @ 16 kHz, n_fft=512, win=400, hop=160 (Hann)
//   Encoder:       FastConformer with biased linears and 8× dw_striding
//                  subsampling; dimensions/layer count come from GGUF metadata
//   Bridge:        optional trained encoder→decoder projection (180M: 512→1024)
//   Decoder:       pre-LN Transformer (self-attn + cross-attn + FFN) with
//                  self/cross KV caches and an untied output head
//
// Prompt formats:
//   legacy cstr: fixed Canary-1B-v2 control-token sequence
//   canary2: metadata token ids for context, transcript, emotion, source,
//            target, PNC, no-ITN, no-timestamp and no-diarization

#include "canary.h"
#include "core/crispasr_env.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "crispasr_imatrix.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "core/attention.h"
#include "core/cpu_ops.h" // core_cpu::to_f32 (quantized-safe weight read)
#include "core/beam_decode.h"
#include "core/canary_chunk_merge.h"
#include "core/crispasr_lcs.h"
#include "core/asr_overlap_trim.h"
#include "core/fastconformer.h"
#include "core/mel.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation — `CANARY_BENCH=1` for per-stage timings.
// ===========================================================================

static bool canary_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_CANARY_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct canary_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit canary_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~canary_bench_stage() {
        if (!canary_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  canary_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct canary_hparams {
    bool new_schema = false;
    bool has_encoder_decoder_proj = false;
    bool tokenizer_single_sp = false;
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 512;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;
    uint32_t enc_n_layers = 32;
    uint32_t enc_d_model = 1024;
    uint32_t enc_n_heads = 8;
    uint32_t enc_head_dim = 128;
    uint32_t enc_ff_dim = 4096;
    uint32_t subsampling_factor = 8;
    uint32_t subsampling_channels = 256;
    uint32_t conv_kernel = 9;
    uint32_t enc_pos_emb_max_len = 5000;
    uint32_t dec_n_layers = 8;
    uint32_t dec_d_model = 1024;
    uint32_t dec_n_heads = 8;
    uint32_t dec_head_dim = 128;
    uint32_t dec_ff_dim = 4096;
    uint32_t vocab_size = 16384;
    uint32_t max_dec_ctx = 1024;
    uint32_t frame_dur_cs = 8;
    float frontend_dither = 0.0f;
    float frontend_preemph = 0.97f;
    float frontend_f_min = 0.0f;
    float frontend_f_max = 8000.0f;
    std::string variant;
    std::string decoder_activation = "relu";
    std::string prompt_format;
    std::vector<std::string> languages;
    std::vector<int> language_ids;
    std::vector<std::string> translation_pairs;
    int startofcontext_id = -1;
    int startoftranscript_id = -1;
    int endoftext_id = -1;
    int pad_id = -1;
    int pnc_id = -1;
    int nopnc_id = -1;
    int noitn_id = -1;
    int notimestamp_id = -1;
    int nodiarize_id = -1;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

// Pre-encode weights: exactly the shared FastConformer layout.
using canary_pre_encode = core_conformer::PreEncodeWeights;

// Per-layer tensor container: inherits the shared Conformer block weights
// (all biases populated for canary) and adds the BN tensors used only at
// load time (BN folding).
struct canary_enc_layer : core_conformer::BlockWeights {
    ggml_tensor *conv_bn_w = nullptr, *conv_bn_b = nullptr;
    ggml_tensor *conv_bn_rm = nullptr, *conv_bn_rv = nullptr;
};

struct canary_dec_layer {
    // Pre-LN block layout:
    //   x = x + sa_out @ SA(norm_sa(x))
    //   x = x + ca_out @ CA(norm_ca(x), enc_kv)
    //   x = x + ff_out @ activation(ff_in @ norm_ff(x))
    ggml_tensor *norm_sa_w = nullptr, *norm_sa_b = nullptr;
    ggml_tensor *sa_q_w = nullptr, *sa_q_b = nullptr;
    ggml_tensor *sa_k_w = nullptr, *sa_k_b = nullptr;
    ggml_tensor *sa_v_w = nullptr, *sa_v_b = nullptr;
    ggml_tensor *sa_out_w = nullptr, *sa_out_b = nullptr;

    ggml_tensor *norm_ca_w = nullptr, *norm_ca_b = nullptr;
    ggml_tensor *ca_q_w = nullptr, *ca_q_b = nullptr;
    ggml_tensor *ca_k_w = nullptr, *ca_k_b = nullptr;
    ggml_tensor *ca_v_w = nullptr, *ca_v_b = nullptr;
    ggml_tensor *ca_out_w = nullptr, *ca_out_b = nullptr;

    ggml_tensor *norm_ff_w = nullptr, *norm_ff_b = nullptr;
    ggml_tensor *ff_in_w = nullptr, *ff_in_b = nullptr;
    ggml_tensor *ff_out_w = nullptr, *ff_out_b = nullptr;
};

// ===========================================================================
// Model
// ===========================================================================

struct canary_model {
    canary_hparams hparams;

    ggml_tensor* mel_fb = nullptr;
    ggml_tensor* mel_window = nullptr;
    std::vector<float> generated_mel_fb;
    std::vector<float> generated_mel_window;

    canary_pre_encode pre_encode;
    std::vector<canary_enc_layer> enc;
    std::vector<canary_dec_layer> dec;
    ggml_tensor* enc_proj_w = nullptr;
    ggml_tensor* enc_proj_b = nullptr;

    // Decoder embeddings + final norm + output head
    ggml_tensor* dec_embed_w = nullptr; // (vocab, d_model)
    ggml_tensor* dec_pos_enc = nullptr; // (max_ctx, d_model) — learned
    ggml_tensor* dec_embed_ln_w = nullptr;
    ggml_tensor* dec_embed_ln_b = nullptr;
    ggml_tensor* dec_final_ln_w = nullptr;
    ggml_tensor* dec_final_ln_b = nullptr;
    ggml_tensor* dec_head_w = nullptr; // (vocab, d_model)
    ggml_tensor* dec_head_b = nullptr;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // Q8_0 repack of the F16 conv pw1/pw2 weights (issue #81, CRISPASR_FC_PW_Q8)
    core_conformer::PwRepackBuf pw_q8;
    // Fused Q/K/V weight concat (issue #81, CRISPASR_FC_FUSED_QKV)
    core_conformer::PwRepackBuf qkv_fused;

    std::map<std::string, ggml_tensor*> tensors;
};

struct canary_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    // PLAN #114 P3 polish — case-insensitive LCS support.
    // `id_to_canonical_lc[i]` is the smallest token id whose
    // lowercase-ASCII text matches token `i`. Used by
    // canary_transcribe_streamed when looking for boundary-overlap
    // matches across chunks where the AED re-emits the same audio with
    // different capitalization (e.g. "world's say for" in chunk 1 vs
    // "World's Save for" in chunk 2 — different raw ids but the LCS
    // should still match them). Populated once at vocab load; ASCII-only
    // (UTF-8 case folding would need libicu and DE/FR/ES chunk-boundary
    // capitalization is rare enough that ASCII is sufficient for the
    // present polish).
    std::vector<int32_t> id_to_canonical_lc;
};

struct canary_context {
    canary_context_params params;

    canary_model model;
    canary_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // Self-attention KV cache for the decoder
    // shape: [head_dim, max_ctx, n_heads, dec_n_layers]
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;

    // Cross-attention K/V — pre-computed once per slice from encoder output.
    // One tensor per decoder layer, shape [head_dim, T_enc, n_heads].
    ggml_context* cross_ctx = nullptr;
    ggml_backend_buffer_t cross_buf = nullptr;
    std::vector<ggml_tensor*> cross_k;
    std::vector<ggml_tensor*> cross_v;

    // Per-step cross-attention weights from the last decoder layer, captured
    // when collect_attn=true and n_tokens==1. Used for DTW timestamp alignment.
    // step_attn[step_idx] has T_enc * n_heads floats (head h occupies the
    // contiguous slice [h*T_enc, h*T_enc + T_enc)).
    bool collect_attn = false;
    int attn_T_enc = 0;
    int attn_n_heads = 0;
    std::vector<std::vector<float>> step_attn;

    int n_threads = 4;

    // §176s: cached encoder graph — reused when T_mel matches.
    ggml_cgraph* cached_enc_gf = nullptr;
    ggml_context* cached_enc_ctx = nullptr;
    std::vector<uint8_t> cached_enc_meta;
    int cached_enc_T_mel = 0;

    // Sticky decode-time sampling controls. temperature == 0 keeps the
    // bit-identical greedy path; > 0 switches to numerically-stable
    // softmax sampling. Set via canary_set_temperature().
    float decode_temperature = 0.0f;
    uint64_t decode_seed = 0;

    // §90 beam-search width. 1 = greedy (default).
    int beam_size = 1;
    // #292: decode cap; forwarded from --max-new-tokens when the user set it,
    // else this default. Clamped to the model's decoder context at use.
    int max_new_tokens = 256;
};

// ===========================================================================
// Loader helpers — thin wrappers around core_gguf::.
// ===========================================================================

#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static ggml_tensor* try_get(canary_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(canary_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "canary");
}

static bool canary_metadata_error(const char* key, const char* detail) {
    fprintf(stderr, "canary: invalid new-schema metadata '%s': %s\n", key, detail);
    return false;
}

static bool canary_read_required_u32(gguf_context* gctx, const char* key, uint32_t& out) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return canary_metadata_error(key, "required key is missing");
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_UINT32)
        return canary_metadata_error(key, "expected uint32");
    out = gguf_get_val_u32(gctx, k);
    return true;
}

static bool canary_read_required_token_id(gguf_context* gctx, const char* key, int& out) {
    uint32_t value = 0;
    if (!canary_read_required_u32(gctx, key, value))
        return false;
    if (value > INT32_MAX)
        return canary_metadata_error(key, "token id exceeds int32 range");
    out = (int)value;
    return true;
}

static bool canary_read_required_f32(gguf_context* gctx, const char* key, float& out) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return canary_metadata_error(key, "required key is missing");
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_FLOAT32)
        return canary_metadata_error(key, "expected float32");
    out = gguf_get_val_f32(gctx, k);
    return true;
}

static bool canary_read_required_bool(gguf_context* gctx, const char* key, bool& out) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return canary_metadata_error(key, "required key is missing");
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_BOOL)
        return canary_metadata_error(key, "expected bool");
    out = gguf_get_val_bool(gctx, k);
    return true;
}

static bool canary_read_required_string(gguf_context* gctx, const char* key, std::string& out) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return canary_metadata_error(key, "required key is missing");
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_STRING)
        return canary_metadata_error(key, "expected string");
    const char* value = gguf_get_val_str(gctx, k);
    if (!value || !*value)
        return canary_metadata_error(key, "value must not be empty");
    out = value;
    return true;
}

static bool canary_read_required_string_array(gguf_context* gctx, const char* key, std::vector<std::string>& out,
                                              bool allow_empty) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return canary_metadata_error(key, "required key is missing");
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gctx, k) != GGUF_TYPE_STRING)
        return canary_metadata_error(key, "expected string array");
    const size_t n = gguf_get_arr_n(gctx, k);
    if (!allow_empty && n == 0)
        return canary_metadata_error(key, "array must not be empty");
    out.clear();
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const char* value = gguf_get_arr_str(gctx, k, i);
        if (!value || !*value)
            return canary_metadata_error(key, "array contains an empty string");
        out.emplace_back(value);
    }
    return true;
}

static bool canary_has_partial_new_schema(gguf_context* gctx) {
    for (int64_t i = 0; i < gguf_get_n_kv(gctx); i++) {
        const char* key = gguf_get_key(gctx, i);
        if (!key)
            continue;
        if (strcmp(key, "stt.variant") == 0 || strncmp(key, "stt.canary.", 11) == 0 ||
            strncmp(key, "stt.frontend.", 13) == 0)
            return true;
    }
    return false;
}

static bool canary_load_new_hparams(gguf_context* gctx, canary_hparams& hp) {
    hp.new_schema = true;

    bool enc_use_bias = false;
    bool dec_pre_ln = false;
    bool dec_learn_pos = false;
    bool frontend_log = false;
    std::string frontend_type;
    std::string frontend_window;
    std::string frontend_normalize;

    if (!canary_read_required_string(gctx, "stt.variant", hp.variant) ||
        !canary_read_required_u32(gctx, "stt.canary.encoder.n_layers", hp.enc_n_layers) ||
        !canary_read_required_u32(gctx, "stt.canary.encoder.d_model", hp.enc_d_model) ||
        !canary_read_required_u32(gctx, "stt.canary.encoder.n_heads", hp.enc_n_heads) ||
        !canary_read_required_u32(gctx, "stt.canary.encoder.d_ff", hp.enc_ff_dim) ||
        !canary_read_required_u32(gctx, "stt.canary.encoder.conv_kernel", hp.conv_kernel) ||
        !canary_read_required_u32(gctx, "stt.canary.encoder.subsampling_factor", hp.subsampling_factor) ||
        !canary_read_required_u32(gctx, "stt.canary.encoder.subsampling_channels", hp.subsampling_channels) ||
        !canary_read_required_u32(gctx, "stt.canary.encoder.pos_emb_max_len", hp.enc_pos_emb_max_len) ||
        !canary_read_required_bool(gctx, "stt.canary.encoder.use_bias", enc_use_bias) ||
        !canary_read_required_u32(gctx, "stt.canary.decoder.n_layers", hp.dec_n_layers) ||
        !canary_read_required_u32(gctx, "stt.canary.decoder.d_model", hp.dec_d_model) ||
        !canary_read_required_u32(gctx, "stt.canary.decoder.n_heads", hp.dec_n_heads) ||
        !canary_read_required_u32(gctx, "stt.canary.decoder.d_ff", hp.dec_ff_dim) ||
        !canary_read_required_u32(gctx, "stt.canary.decoder.max_position", hp.max_dec_ctx) ||
        !canary_read_required_u32(gctx, "stt.canary.decoder.vocab_size", hp.vocab_size) ||
        !canary_read_required_string(gctx, "stt.canary.decoder.activation", hp.decoder_activation) ||
        !canary_read_required_bool(gctx, "stt.canary.decoder.pre_ln", dec_pre_ln) ||
        !canary_read_required_bool(gctx, "stt.canary.decoder.learn_positional_encodings", dec_learn_pos) ||
        !canary_read_required_bool(gctx, "stt.canary.decoder.encoder_decoder_proj", hp.has_encoder_decoder_proj) ||
        !canary_read_required_string(gctx, "stt.canary.tokenizer.prompt_format", hp.prompt_format) ||
        !canary_read_required_bool(gctx, "stt.canary.tokenizer.single_sp", hp.tokenizer_single_sp) ||
        !canary_read_required_string(gctx, "stt.frontend.type", frontend_type) ||
        !canary_read_required_u32(gctx, "stt.frontend.num_mels", hp.n_mels) ||
        !canary_read_required_u32(gctx, "stt.frontend.sample_rate", hp.sample_rate) ||
        !canary_read_required_u32(gctx, "stt.frontend.n_fft", hp.n_fft) ||
        !canary_read_required_u32(gctx, "stt.frontend.win_length", hp.win_length) ||
        !canary_read_required_u32(gctx, "stt.frontend.hop_length", hp.hop_length) ||
        !canary_read_required_string(gctx, "stt.frontend.window", frontend_window) ||
        !canary_read_required_string(gctx, "stt.frontend.normalize", frontend_normalize) ||
        !canary_read_required_f32(gctx, "stt.frontend.dither", hp.frontend_dither) ||
        !canary_read_required_f32(gctx, "stt.frontend.pre_emphasis", hp.frontend_preemph) ||
        !canary_read_required_f32(gctx, "stt.frontend.f_min", hp.frontend_f_min) ||
        !canary_read_required_f32(gctx, "stt.frontend.f_max", hp.frontend_f_max) ||
        !canary_read_required_bool(gctx, "stt.frontend.log", frontend_log) ||
        !canary_read_required_string_array(gctx, "general.languages", hp.languages, false) ||
        !canary_read_required_string_array(gctx, "stt.translation.pairs", hp.translation_pairs, true) ||
        !canary_read_required_token_id(gctx, "stt.canary.special.startofcontext_id", hp.startofcontext_id) ||
        !canary_read_required_token_id(gctx, "stt.canary.special.startoftranscript_id", hp.startoftranscript_id) ||
        !canary_read_required_token_id(gctx, "stt.canary.special.endoftext_id", hp.endoftext_id) ||
        !canary_read_required_token_id(gctx, "stt.canary.special.pad_id", hp.pad_id) ||
        !canary_read_required_token_id(gctx, "stt.canary.special.pnc_id", hp.pnc_id) ||
        !canary_read_required_token_id(gctx, "stt.canary.special.nopnc_id", hp.nopnc_id) ||
        !canary_read_required_token_id(gctx, "stt.canary.special.noitn_id", hp.noitn_id) ||
        !canary_read_required_token_id(gctx, "stt.canary.special.notimestamp_id", hp.notimestamp_id) ||
        !canary_read_required_token_id(gctx, "stt.canary.special.nodiarize_id", hp.nodiarize_id)) {
        return false;
    }

    hp.language_ids.clear();
    hp.language_ids.reserve(hp.languages.size());
    for (const std::string& language : hp.languages) {
        const std::string key = "stt.canary.special.lang." + language + "_id";
        int id = -1;
        if (!canary_read_required_token_id(gctx, key.c_str(), id))
            return false;
        hp.language_ids.push_back(id);
    }

    hp.enc_head_dim = hp.enc_n_heads > 0 ? hp.enc_d_model / hp.enc_n_heads : 0;
    hp.dec_head_dim = hp.dec_n_heads > 0 ? hp.dec_d_model / hp.dec_n_heads : 0;
    if (hp.sample_rate > 0)
        hp.frame_dur_cs = hp.subsampling_factor * hp.hop_length * 100 / hp.sample_rate;

    if (!enc_use_bias)
        return canary_metadata_error("stt.canary.encoder.use_bias", "bias-free Canary is not supported");
    if (!dec_pre_ln)
        return canary_metadata_error("stt.canary.decoder.pre_ln", "only pre-LN decoders are supported");
    if (frontend_type != "mel")
        return canary_metadata_error("stt.frontend.type", "only 'mel' is supported");
    if (frontend_window != "hann")
        return canary_metadata_error("stt.frontend.window", "only 'hann' is supported");
    if (frontend_normalize != "per_feature")
        return canary_metadata_error("stt.frontend.normalize", "only 'per_feature' is supported");
    if (!frontend_log)
        return canary_metadata_error("stt.frontend.log", "log-mel frontend is required");
    if (hp.prompt_format != "canary2")
        return canary_metadata_error("stt.canary.tokenizer.prompt_format", "only 'canary2' is supported");
    if (hp.decoder_activation != "relu" && hp.decoder_activation != "silu" && hp.decoder_activation != "swish")
        return canary_metadata_error("stt.canary.decoder.activation", "expected relu, silu, or swish");
    if (hp.enc_n_layers == 0 || hp.enc_d_model == 0 || hp.enc_n_heads == 0 || hp.enc_ff_dim == 0 ||
        hp.enc_d_model % hp.enc_n_heads != 0 || hp.dec_n_layers == 0 || hp.dec_d_model == 0 || hp.dec_n_heads == 0 ||
        hp.dec_ff_dim == 0 || hp.dec_d_model % hp.dec_n_heads != 0 || hp.vocab_size == 0 || hp.max_dec_ctx == 0)
        return canary_metadata_error("stt.canary", "invalid zero/divisibility invariant");
    if (hp.n_fft == 0 || (hp.n_fft & (hp.n_fft - 1)) != 0 || hp.win_length == 0 || hp.win_length > hp.n_fft ||
        hp.hop_length == 0 || hp.n_mels == 0 || hp.sample_rate == 0)
        return canary_metadata_error("stt.frontend", "invalid FFT/window/rate invariant");
    if (hp.subsampling_factor != 8 || hp.n_mels != 128)
        return canary_metadata_error("stt.canary.encoder", "runtime requires subsampling_factor=8 and num_mels=128");
    if ((hp.enc_d_model != hp.dec_d_model) != hp.has_encoder_decoder_proj)
        return canary_metadata_error("stt.canary.decoder.encoder_decoder_proj",
                                     "projection flag must match encoder/decoder width split");
    (void)dec_learn_pos;
    return true;
}

static bool canary_tensor_type_supported(const ggml_tensor* tensor, bool require_f32) {
    if (!tensor)
        return false;
    if (require_f32)
        return tensor->type == GGML_TYPE_F32;
    return tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16 ||
           ggml_is_quantized(tensor->type);
}

static bool canary_validate_tensor(const ggml_tensor* tensor, const char* name, std::initializer_list<int64_t> shape,
                                   bool require_f32) {
    if (!tensor)
        return false;
    if (!canary_tensor_type_supported(tensor, require_f32)) {
        fprintf(stderr, "canary: tensor '%s' has unsupported type %s%s\n", name, ggml_type_name(tensor->type),
                require_f32 ? " (expected F32)" : "");
        return false;
    }
    size_t dim = 0;
    bool shape_ok = true;
    for (int64_t expected : shape) {
        if (dim >= GGML_MAX_DIMS || tensor->ne[dim] != expected)
            shape_ok = false;
        dim++;
    }
    for (; dim < GGML_MAX_DIMS; dim++) {
        if (tensor->ne[dim] != 1)
            shape_ok = false;
    }
    if (!shape_ok) {
        fprintf(stderr, "canary: tensor '%s' has shape [%lld,%lld,%lld,%lld], expected [", name,
                (long long)tensor->ne[0], (long long)tensor->ne[1], (long long)tensor->ne[2], (long long)tensor->ne[3]);
        size_t i = 0;
        for (int64_t expected : shape)
            fprintf(stderr, "%s%lld", i++ ? "," : "", (long long)expected);
        fprintf(stderr, "]\n");
        return false;
    }
    return true;
}

static ggml_tensor* canary_bind_tensor(canary_model& model, const char* legacy_name, const char* new_name,
                                       std::initializer_list<int64_t> shape, bool require_f32, bool& ok) {
    const char* name = model.hparams.new_schema ? new_name : legacy_name;
    ggml_tensor* tensor = require(model, name);
    if (!tensor || !canary_validate_tensor(tensor, name, shape, require_f32))
        ok = false;
    return tensor;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool canary_load_model(canary_model& model, canary_vocab& vocab, const char* path, ggml_backend_t backend) {
    // ---- pass 1: hparams + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        const bool has_sentinel = gguf_find_key(gctx, "stt.canary.encoder.d_model") >= 0;
        if (has_sentinel) {
            if (!canary_load_new_hparams(gctx, hp)) {
                core_gguf::free_metadata(gctx);
                return false;
            }
        } else {
            if (canary_has_partial_new_schema(gctx)) {
                fprintf(stderr, "canary: partial/mixed transcribe.cpp schema detected without "
                                "stt.canary.encoder.d_model; refusing legacy defaults\n");
                core_gguf::free_metadata(gctx);
                return false;
            }
            hp.sample_rate = core_gguf::kv_u32(gctx, "canary.sample_rate", hp.sample_rate);
            hp.n_mels = core_gguf::kv_u32(gctx, "canary.n_mels", hp.n_mels);
            hp.n_fft = core_gguf::kv_u32(gctx, "canary.n_fft", hp.n_fft);
            hp.win_length = core_gguf::kv_u32(gctx, "canary.win_length", hp.win_length);
            hp.hop_length = core_gguf::kv_u32(gctx, "canary.hop_length", hp.hop_length);
            hp.enc_d_model = core_gguf::kv_u32(gctx, "canary.d_model", hp.enc_d_model);
            hp.dec_d_model = hp.enc_d_model;
            hp.enc_n_layers = core_gguf::kv_u32(gctx, "canary.enc_n_layers", hp.enc_n_layers);
            hp.dec_n_layers = core_gguf::kv_u32(gctx, "canary.dec_n_layers", hp.dec_n_layers);
            hp.enc_n_heads = core_gguf::kv_u32(gctx, "canary.n_heads", hp.enc_n_heads);
            hp.dec_n_heads = hp.enc_n_heads;
            hp.enc_head_dim = core_gguf::kv_u32(gctx, "canary.head_dim", hp.enc_head_dim);
            hp.dec_head_dim = hp.enc_head_dim;
            hp.enc_ff_dim = core_gguf::kv_u32(gctx, "canary.ff_dim", hp.enc_ff_dim);
            hp.dec_ff_dim = hp.enc_ff_dim;
            hp.subsampling_factor = core_gguf::kv_u32(gctx, "canary.subsampling_factor", hp.subsampling_factor);
            hp.subsampling_channels = core_gguf::kv_u32(gctx, "canary.subsampling_channels", hp.subsampling_channels);
            hp.conv_kernel = core_gguf::kv_u32(gctx, "canary.conv_kernel", hp.conv_kernel);
            hp.vocab_size = core_gguf::kv_u32(gctx, "canary.vocab_size", hp.vocab_size);
            hp.max_dec_ctx = core_gguf::kv_u32(gctx, "canary.max_dec_ctx", hp.max_dec_ctx);
            hp.frame_dur_cs = core_gguf::kv_u32(gctx, "canary.frame_dur_cs", hp.frame_dur_cs);
        }

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
            // PLAN #114 P3 polish — build the case-insensitive canonical
            // mapping. ASCII lowercase only; non-ASCII bytes (UTF-8) pass
            // through unchanged, so DE-umlaut tokens still match their
            // own variants but not cross-case. Sufficient for the
            // observed EN FLEURS chunk-boundary capitalization artifacts.
            const int n_vocab = (int)vocab.id_to_token.size();
            vocab.id_to_canonical_lc.assign((size_t)n_vocab, 0);
            std::unordered_map<std::string, int> first_id_for_lc;
            first_id_for_lc.reserve((size_t)n_vocab);
            for (int i = 0; i < n_vocab; i++) {
                std::string lc = vocab.id_to_token[i];
                for (char& c : lc) {
                    if (c >= 'A' && c <= 'Z')
                        c = (char)(c + ('a' - 'A'));
                }
                auto it = first_id_for_lc.find(lc);
                if (it == first_id_for_lc.end()) {
                    first_id_for_lc.emplace(std::move(lc), i);
                    vocab.id_to_canonical_lc[(size_t)i] = i;
                } else {
                    vocab.id_to_canonical_lc[(size_t)i] = it->second;
                }
            }
        }
        if (hp.new_schema && vocab.id_to_token.size() != hp.vocab_size) {
            fprintf(stderr, "canary: tokenizer vocab has %zu entries; metadata declares %u\n", vocab.id_to_token.size(),
                    hp.vocab_size);
            core_gguf::free_metadata(gctx);
            return false;
        }
        if (hp.new_schema) {
            auto token_id_valid = [&](int id, const char* key) {
                if (id >= 0 && id < (int)vocab.id_to_token.size())
                    return true;
                fprintf(stderr, "canary: metadata token id '%s'=%d is outside vocab size %zu\n", key, id,
                        vocab.id_to_token.size());
                return false;
            };
            if (!token_id_valid(hp.startofcontext_id, "startofcontext") ||
                !token_id_valid(hp.startoftranscript_id, "startoftranscript") ||
                !token_id_valid(hp.endoftext_id, "endoftext") || !token_id_valid(hp.pad_id, "pad") ||
                !token_id_valid(hp.pnc_id, "pnc") || !token_id_valid(hp.nopnc_id, "nopnc") ||
                !token_id_valid(hp.noitn_id, "noitn") || !token_id_valid(hp.notimestamp_id, "notimestamp") ||
                !token_id_valid(hp.nodiarize_id, "nodiarize")) {
                core_gguf::free_metadata(gctx);
                return false;
            }
            for (size_t i = 0; i < hp.language_ids.size(); i++) {
                if (!token_id_valid(hp.language_ids[i], hp.languages[i].c_str())) {
                    core_gguf::free_metadata(gctx);
                    return false;
                }
            }
        }

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: tensor data via shared helper ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "canary", wl)) {
        return false;
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // ---- bind named tensors ----
    bool bindings_ok = true;
    const auto& hp = model.hparams;
    const int64_t enc_d = hp.enc_d_model;
    const int64_t dec_d = hp.dec_d_model;
    const int64_t enc_ff = hp.enc_ff_dim;
    const int64_t dec_ff = hp.dec_ff_dim;
    const int64_t enc_heads = hp.enc_n_heads;
    const int64_t enc_head_dim = hp.enc_head_dim;
    const int64_t channels = hp.subsampling_channels;
    const int64_t pre_encode_in = channels * (hp.n_mels / hp.subsampling_factor);

    // Mel preprocessor
    if (hp.new_schema) {
        model.mel_fb = try_get(model, "frontend.mel_filterbank");
        model.mel_window = try_get(model, "frontend.window");
        if ((model.mel_fb == nullptr) != (model.mel_window == nullptr)) {
            fprintf(stderr, "canary: new-schema frontend tensors are incomplete; "
                            "frontend.mel_filterbank and frontend.window must appear together\n");
            bindings_ok = false;
        } else if (model.mel_fb) {
            const size_t fb_elems = (size_t)hp.n_mels * (hp.n_fft / 2 + 1);
            if (model.mel_fb->type != GGML_TYPE_F32 || (size_t)ggml_nelements(model.mel_fb) != fb_elems) {
                fprintf(stderr, "canary: frontend.mel_filterbank must be F32 with %zu elements (got %s, %lld)\n",
                        fb_elems, ggml_type_name(model.mel_fb->type), (long long)ggml_nelements(model.mel_fb));
                bindings_ok = false;
            }
            if (model.mel_window->type != GGML_TYPE_F32 || (size_t)ggml_nelements(model.mel_window) != hp.win_length) {
                fprintf(stderr, "canary: frontend.window must be F32 with %u elements (got %s, %lld)\n", hp.win_length,
                        ggml_type_name(model.mel_window->type), (long long)ggml_nelements(model.mel_window));
                bindings_ok = false;
            }
        } else {
            model.generated_mel_fb =
                core_mel::build_slaney_fb((int)hp.sample_rate, (int)hp.n_fft, (int)hp.n_mels, hp.frontend_f_min,
                                          hp.frontend_f_max, core_mel::FbLayout::MelsFreqs);
            model.generated_mel_window.resize(hp.win_length);
            const float denom = hp.win_length > 1 ? (float)(hp.win_length - 1) : 1.0f;
            for (uint32_t i = 0; i < hp.win_length; i++)
                model.generated_mel_window[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / denom);
        }
    } else {
        model.mel_fb = try_get(model, "preprocessor.fb");
        model.mel_window = try_get(model, "preprocessor.window");
        if (!model.mel_fb || !model.mel_window) {
            fprintf(stderr, "canary: legacy GGUF is missing preprocessor.fb / preprocessor.window\n");
            bindings_ok = false;
        }
    }

    // Pre-encode
    model.pre_encode.conv0_w = canary_bind_tensor(model, "encoder.pre.conv.0.weight", "enc.pre_encode.conv.0.weight",
                                                  {3, 3, 1, channels}, false, bindings_ok);
    model.pre_encode.conv0_b = canary_bind_tensor(model, "encoder.pre.conv.0.bias", "enc.pre_encode.conv.0.bias",
                                                  {channels}, true, bindings_ok);
    model.pre_encode.conv2_w = canary_bind_tensor(model, "encoder.pre.conv.2.weight", "enc.pre_encode.conv.2.weight",
                                                  {3, 3, 1, channels}, false, bindings_ok);
    model.pre_encode.conv2_b = canary_bind_tensor(model, "encoder.pre.conv.2.bias", "enc.pre_encode.conv.2.bias",
                                                  {channels}, true, bindings_ok);
    model.pre_encode.conv3_w = canary_bind_tensor(model, "encoder.pre.conv.3.weight", "enc.pre_encode.conv.3.weight",
                                                  {1, 1, channels, channels}, false, bindings_ok);
    model.pre_encode.conv3_b = canary_bind_tensor(model, "encoder.pre.conv.3.bias", "enc.pre_encode.conv.3.bias",
                                                  {channels}, true, bindings_ok);
    model.pre_encode.conv5_w = canary_bind_tensor(model, "encoder.pre.conv.5.weight", "enc.pre_encode.conv.5.weight",
                                                  {3, 3, 1, channels}, false, bindings_ok);
    model.pre_encode.conv5_b = canary_bind_tensor(model, "encoder.pre.conv.5.bias", "enc.pre_encode.conv.5.bias",
                                                  {channels}, true, bindings_ok);
    model.pre_encode.conv6_w = canary_bind_tensor(model, "encoder.pre.conv.6.weight", "enc.pre_encode.conv.6.weight",
                                                  {1, 1, channels, channels}, false, bindings_ok);
    model.pre_encode.conv6_b = canary_bind_tensor(model, "encoder.pre.conv.6.bias", "enc.pre_encode.conv.6.bias",
                                                  {channels}, true, bindings_ok);
    model.pre_encode.out_w = canary_bind_tensor(model, "encoder.pre.out.weight", "enc.pre_encode.out.weight",
                                                {pre_encode_in, enc_d}, false, bindings_ok);
    model.pre_encode.out_b =
        canary_bind_tensor(model, "encoder.pre.out.bias", "enc.pre_encode.out.bias", {enc_d}, true, bindings_ok);

    // Encoder layers
    model.enc.resize(model.hparams.enc_n_layers);
    for (uint32_t i = 0; i < model.hparams.enc_n_layers; i++) {
        char legacy[128];
        char current[128];
        auto& e = model.enc[i];
        auto get = [&](const char* legacy_suffix, const char* current_suffix, std::initializer_list<int64_t> shape,
                       bool require_f32) {
            snprintf(legacy, sizeof(legacy), "encoder.layers.%u.%s", i, legacy_suffix);
            snprintf(current, sizeof(current), "enc.blocks.%u.%s", i, current_suffix);
            return canary_bind_tensor(model, legacy, current, shape, require_f32, bindings_ok);
        };

        e.norm_ff1_w = get("norm_ff1.weight", "norm_ff1.weight", {enc_d}, true);
        e.norm_ff1_b = get("norm_ff1.bias", "norm_ff1.bias", {enc_d}, true);
        e.ff1_l1_w = get("ff1.linear1.weight", "ff1.linear1.weight", {enc_d, enc_ff}, false);
        e.ff1_l1_b = get("ff1.linear1.bias", "ff1.linear1.bias", {enc_ff}, true);
        e.ff1_l2_w = get("ff1.linear2.weight", "ff1.linear2.weight", {enc_ff, enc_d}, false);
        e.ff1_l2_b = get("ff1.linear2.bias", "ff1.linear2.bias", {enc_d}, true);

        e.norm_attn_w = get("norm_attn.weight", "norm_attn.weight", {enc_d}, true);
        e.norm_attn_b = get("norm_attn.bias", "norm_attn.bias", {enc_d}, true);
        e.attn_q_w = get("attn.q.weight", "attn.linear_q.weight", {enc_d, enc_d}, false);
        e.attn_q_b = get("attn.q.bias", "attn.linear_q.bias", {enc_d}, true);
        e.attn_k_w = get("attn.k.weight", "attn.linear_k.weight", {enc_d, enc_d}, false);
        e.attn_k_b = get("attn.k.bias", "attn.linear_k.bias", {enc_d}, true);
        e.attn_v_w = get("attn.v.weight", "attn.linear_v.weight", {enc_d, enc_d}, false);
        e.attn_v_b = get("attn.v.bias", "attn.linear_v.bias", {enc_d}, true);
        e.attn_out_w = get("attn.out.weight", "attn.linear_out.weight", {enc_d, enc_d}, false);
        e.attn_out_b = get("attn.out.bias", "attn.linear_out.bias", {enc_d}, true);
        e.attn_pos_w = get("attn.pos.weight", "attn.linear_pos.weight", {enc_d, enc_d}, false);
        e.pos_bias_u = get("attn.pos_bias_u", "attn.pos_bias_u", {enc_head_dim, enc_heads}, true);
        e.pos_bias_v = get("attn.pos_bias_v", "attn.pos_bias_v", {enc_head_dim, enc_heads}, true);

        e.norm_conv_w = get("norm_conv.weight", "norm_conv.weight", {enc_d}, true);
        e.norm_conv_b = get("norm_conv.bias", "norm_conv.bias", {enc_d}, true);
        // The converters preserve different harmless singleton-axis layouts:
        // legacy cstr stores pointwise Conv1d as [in,out,1,1], while the
        // transcribe.cpp schema stores [1,in,out]. Both feed the existing
        // reshape-based compute path unchanged; validate each exact catalog
        // shape instead of weakening validation for unrelated tensors.
        e.conv_pw1_w = hp.new_schema ? get("conv.pw1.weight", "conv.pointwise1.weight", {1, enc_d, 2 * enc_d}, false)
                                     : get("conv.pw1.weight", "conv.pointwise1.weight", {enc_d, 2 * enc_d}, false);
        e.conv_pw1_b = get("conv.pw1.bias", "conv.pointwise1.bias", {2 * enc_d}, true);
        e.conv_dw_w = get("conv.dw.weight", "conv.depthwise.weight", {hp.conv_kernel, 1, enc_d}, false);
        e.conv_dw_b = get("conv.dw.bias", "conv.depthwise.bias", {enc_d}, true);
        e.conv_pw2_w = hp.new_schema ? get("conv.pw2.weight", "conv.pointwise2.weight", {1, enc_d, enc_d}, false)
                                     : get("conv.pw2.weight", "conv.pointwise2.weight", {enc_d, enc_d}, false);
        e.conv_pw2_b = get("conv.pw2.bias", "conv.pointwise2.bias", {enc_d}, true);
        e.conv_bn_w = get("conv.bn.weight", "conv.bn.weight", {enc_d}, true);
        e.conv_bn_b = get("conv.bn.bias", "conv.bn.bias", {enc_d}, true);
        e.conv_bn_rm = get("conv.bn.running_mean", "conv.bn.running_mean", {enc_d}, true);
        e.conv_bn_rv = get("conv.bn.running_var", "conv.bn.running_var", {enc_d}, true);

        e.norm_ff2_w = get("norm_ff2.weight", "norm_ff2.weight", {enc_d}, true);
        e.norm_ff2_b = get("norm_ff2.bias", "norm_ff2.bias", {enc_d}, true);
        e.ff2_l1_w = get("ff2.linear1.weight", "ff2.linear1.weight", {enc_d, enc_ff}, false);
        e.ff2_l1_b = get("ff2.linear1.bias", "ff2.linear1.bias", {enc_ff}, true);
        e.ff2_l2_w = get("ff2.linear2.weight", "ff2.linear2.weight", {enc_ff, enc_d}, false);
        e.ff2_l2_b = get("ff2.linear2.bias", "ff2.linear2.bias", {enc_d}, true);

        e.norm_out_w = get("norm_out.weight", "norm_out.weight", {enc_d}, true);
        e.norm_out_b = get("norm_out.bias", "norm_out.bias", {enc_d}, true);
    }

    if (hp.has_encoder_decoder_proj) {
        model.enc_proj_w = canary_bind_tensor(model, nullptr, "enc.proj.weight", {enc_d, dec_d}, false, bindings_ok);
        model.enc_proj_b = canary_bind_tensor(model, nullptr, "enc.proj.bias", {dec_d}, true, bindings_ok);
    }

    // Decoder
    model.dec.resize(model.hparams.dec_n_layers);
    for (uint32_t i = 0; i < model.hparams.dec_n_layers; i++) {
        char legacy[128];
        char current[128];
        auto& d = model.dec[i];
        auto get = [&](const char* legacy_suffix, const char* current_suffix, std::initializer_list<int64_t> shape,
                       bool require_f32) {
            snprintf(legacy, sizeof(legacy), "decoder.layers.%u.%s", i, legacy_suffix);
            snprintf(current, sizeof(current), "dec.layer.%u.%s", i, current_suffix);
            return canary_bind_tensor(model, legacy, current, shape, require_f32, bindings_ok);
        };

        d.norm_sa_w = get("norm_sa.weight", "norm1.weight", {dec_d}, true);
        d.norm_sa_b = get("norm_sa.bias", "norm1.bias", {dec_d}, true);
        d.sa_q_w = get("sa_q.weight", "self_attn.q.weight", {dec_d, dec_d}, false);
        d.sa_q_b = get("sa_q.bias", "self_attn.q.bias", {dec_d}, true);
        d.sa_k_w = get("sa_k.weight", "self_attn.k.weight", {dec_d, dec_d}, false);
        d.sa_k_b = get("sa_k.bias", "self_attn.k.bias", {dec_d}, true);
        d.sa_v_w = get("sa_v.weight", "self_attn.v.weight", {dec_d, dec_d}, false);
        d.sa_v_b = get("sa_v.bias", "self_attn.v.bias", {dec_d}, true);
        d.sa_out_w = get("sa_out.weight", "self_attn.o.weight", {dec_d, dec_d}, false);
        d.sa_out_b = get("sa_out.bias", "self_attn.o.bias", {dec_d}, true);

        d.norm_ca_w = get("norm_ca.weight", "norm2.weight", {dec_d}, true);
        d.norm_ca_b = get("norm_ca.bias", "norm2.bias", {dec_d}, true);
        d.ca_q_w = get("ca_q.weight", "cross_attn.q.weight", {dec_d, dec_d}, false);
        d.ca_q_b = get("ca_q.bias", "cross_attn.q.bias", {dec_d}, true);
        d.ca_k_w = get("ca_k.weight", "cross_attn.k.weight", {dec_d, dec_d}, false);
        d.ca_k_b = get("ca_k.bias", "cross_attn.k.bias", {dec_d}, true);
        d.ca_v_w = get("ca_v.weight", "cross_attn.v.weight", {dec_d, dec_d}, false);
        d.ca_v_b = get("ca_v.bias", "cross_attn.v.bias", {dec_d}, true);
        d.ca_out_w = get("ca_out.weight", "cross_attn.o.weight", {dec_d, dec_d}, false);
        d.ca_out_b = get("ca_out.bias", "cross_attn.o.bias", {dec_d}, true);

        d.norm_ff_w = get("norm_ff.weight", "norm3.weight", {dec_d}, true);
        d.norm_ff_b = get("norm_ff.bias", "norm3.bias", {dec_d}, true);
        d.ff_in_w = get("ff_in.weight", "ffn.up.weight", {dec_d, dec_ff}, false);
        d.ff_in_b = get("ff_in.bias", "ffn.up.bias", {dec_ff}, true);
        d.ff_out_w = get("ff_out.weight", "ffn.down.weight", {dec_ff, dec_d}, false);
        d.ff_out_b = get("ff_out.bias", "ffn.down.bias", {dec_d}, true);
    }

    // Decoder embeddings + output head
    model.dec_embed_w = canary_bind_tensor(model, "decoder.embed.weight", "dec.embed.token.weight",
                                           {dec_d, hp.vocab_size}, false, bindings_ok);
    model.dec_pos_enc =
        canary_bind_tensor(model, "decoder.pos_enc", "dec.embed.pos_enc", {dec_d, hp.max_dec_ctx}, true, bindings_ok);
    model.dec_embed_ln_w =
        canary_bind_tensor(model, "decoder.embed_ln.weight", "dec.embed.norm.weight", {dec_d}, true, bindings_ok);
    model.dec_embed_ln_b =
        canary_bind_tensor(model, "decoder.embed_ln.bias", "dec.embed.norm.bias", {dec_d}, true, bindings_ok);
    model.dec_final_ln_w =
        canary_bind_tensor(model, "decoder.final_norm.weight", "dec.norm.weight", {dec_d}, true, bindings_ok);
    model.dec_final_ln_b =
        canary_bind_tensor(model, "decoder.final_norm.bias", "dec.norm.bias", {dec_d}, true, bindings_ok);
    model.dec_head_w =
        canary_bind_tensor(model, "decoder.head.weight", "dec.head.weight", {dec_d, hp.vocab_size}, false, bindings_ok);
    model.dec_head_b =
        canary_bind_tensor(model, "decoder.head.bias", "dec.head.bias", {hp.vocab_size}, true, bindings_ok);

    if (!bindings_ok)
        return false;

    fprintf(stderr,
            "canary: schema=%s variant=%s vocab=%u encoder=%uL/%u/%uH/%uFF "
            "decoder=%uL/%u/%uH/%uFF max_ctx=%u projection=%s\n",
            hp.new_schema ? "transcribe.cpp" : "legacy", hp.variant.empty() ? "canary-1b-v2" : hp.variant.c_str(),
            hp.vocab_size, hp.enc_n_layers, hp.enc_d_model, hp.enc_n_heads, hp.enc_ff_dim, hp.dec_n_layers,
            hp.dec_d_model, hp.dec_n_heads, hp.dec_ff_dim, hp.max_dec_ctx, hp.has_encoder_decoder_proj ? "yes" : "no");
    return true;
}

// ===========================================================================
// FFT (iterative Cooley-Tukey, real-input, N must be a power of 2)
// ===========================================================================

static void canary_fft_r2c(const float* in, int N, float* out) {
    int bits = 0;
    for (int n = N; n > 1; n >>= 1)
        bits++;
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int b = 0; b < bits; b++)
            rev = (rev << 1) | ((i >> b) & 1);
        out[2 * rev] = in[i];
        out[2 * rev + 1] = 0.0f;
    }
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, b = i + j + len / 2;
                float are = out[2 * a], aim = out[2 * a + 1];
                float bre = out[2 * b], bim = out[2 * b + 1];
                float tre = ure * bre - uim * bim, tim = ure * bim + uim * bre;
                out[2 * a] = are + tre;
                out[2 * a + 1] = aim + tim;
                out[2 * b] = are - tre;
                out[2 * b + 1] = aim - tim;
                float new_ure = ure * wre - uim * wim;
                uim = ure * wim + uim * wre;
                ure = new_ure;
            }
        }
    }
}

// NeMo-style mel: same as parakeet (128 mel, 16 kHz, n_fft=512, win=400, hop=160).
// Delegates to core_mel::compute() with the NeMo cluster's parameters; only
// the FFT function pointer differs between parakeet/canary/canary_ctc/cohere.
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)
#include "core/ggml_cpu_backend.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static std::vector<float> canary_compute_mel_impl(canary_context* ctx, const float* samples, int n_samples,
                                                  int& T_out) {
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int win = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels = (int)hp.n_mels;

    if ((!ctx->model.mel_fb || !ctx->model.mel_window) &&
        (ctx->model.generated_mel_fb.empty() || ctx->model.generated_mel_window.empty())) {
        fprintf(stderr, "canary: missing frontend filterbank/window\n");
        return {};
    }

    std::vector<float> window_raw;
    std::vector<float> mel_fb;
    if (ctx->model.mel_window) {
        window_raw.resize((size_t)win);
        ggml_backend_tensor_get(ctx->model.mel_window, window_raw.data(), 0, window_raw.size() * sizeof(float));
        mel_fb.resize((size_t)n_mels * n_freqs);
        ggml_backend_tensor_get(ctx->model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));
    } else {
        window_raw = ctx->model.generated_mel_window;
        mel_fb = ctx->model.generated_mel_fb;
    }

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = win;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.norm = core_mel::Normalization::PerFeatureZ;
    p.layout = core_mel::Layout::TimeMels;
    p.log_eps = (float)(1.0 / (1 << 24));
    p.center_pad = true;
    p.center_pad_reflect = hp.new_schema;
    p.drop_last_frame = !hp.new_schema; // Preserve the baked cstr frontend's historical frame contract.
    p.preemph = hp.new_schema ? hp.frontend_preemph : 0.97f;
    // Dither is a training/preprocessing knob in the published metadata.
    // Inference stays deterministic and intentionally does not apply it.

    return core_mel::compute(samples, n_samples, window_raw.data(), win, mel_fb.data(), n_freqs, canary_fft_r2c, p,
                             T_out);
}

// ===========================================================================
// rel-pos shift (Transformer-XL trick): single zero-cost ggml_view_3d
// ===========================================================================

// rel_shift and make_pos_enc moved to core_conformer in src/core/fastconformer.h.

// ===========================================================================
// Encoder graph
//
// Same structure as parakeet but with biases on every linear/conv. The
// encoder block is the standard pre-LN Conformer (FFN1/2 macaron, MHA
// with rel-pos untied biases, depthwise sep conv with GLU, final block LN).
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

static ggml_cgraph* canary_build_graph_encoder(canary_context* ctx, int T_mel, ggml_context* arena_ctx = nullptr) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = arena_ctx ? arena_ctx : ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // ----- Inputs -----
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // ----- Pre-encode (dw_striding 8×) -----
    int T = 0;
    ggml_tensor* cur = core_conformer::build_pre_encode(ctx0, mel, m.pre_encode, (int)hp.subsampling_channels, &T);

    // ----- Sinusoidal rel-pos table -----
    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.enc_d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    // ----- 32× FastConformer block (with biases) -----
    core_conformer::BlockParams bp = {
        (int)hp.enc_d_model, (int)hp.enc_n_heads, (int)hp.enc_head_dim, (int)hp.conv_kernel, kLayerNormEps,
    };
    for (uint32_t il = 0; il < hp.enc_n_layers; il++) {
        cur = core_conformer::build_block(ctx0, cur, pos_enc, T, m.enc[il], bp);
    }

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    if (!arena_ctx)
        ggml_free(ctx0);
    return gf;
}

// Staged graph: same as above but snapshots intermediate tensors via ggml_dup
// so they survive after the allocator reclaims intermediate buffers.
// Names: "pre_enc_out", "enc_L00".."enc_L31", "enc_out".
// Graph size is larger (~33 extra dup nodes), so we use 24576 max nodes.
static ggml_cgraph* canary_build_graph_encoder_staged(canary_context* ctx, int T_mel) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 24576, false);

    ggml_tensor* mel_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel_t, "mel");
    ggml_set_input(mel_t);

    int T = 0;
    ggml_tensor* cur =
        core_conformer::build_pre_encode(ctx0, mel_t, m.pre_encode, (int)hp.subsampling_channels, &T, gf);

    {
        ggml_tensor* snap = ggml_dup(ctx0, cur);
        ggml_set_name(snap, "pre_enc_out");
        ggml_build_forward_expand(gf, snap);
    }

    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.enc_d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    core_conformer::BlockParams bp = {
        (int)hp.enc_d_model, (int)hp.enc_n_heads, (int)hp.enc_head_dim, (int)hp.conv_kernel, kLayerNormEps,
    };
    char lbuf[32];
    for (uint32_t il = 0; il < hp.enc_n_layers; il++) {
        cur = core_conformer::build_block(ctx0, cur, pos_enc, T, m.enc[il], bp);
        snprintf(lbuf, sizeof(lbuf), "enc_L%02u", il);
        ggml_tensor* snap = ggml_dup(ctx0, cur);
        ggml_set_name(snap, lbuf);
        ggml_build_forward_expand(gf, snap);
    }

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run the encoder once. Returns enc_out as a flat row-major [T_enc * d_model].
static std::vector<float> canary_encode_mel(canary_context* ctx, const float* mel, int n_mels, int T_mel,
                                            int* out_T_enc) {
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "canary: mel feature mismatch (%d vs %d)\n", n_mels, (int)ctx->model.hparams.n_mels);
        return {};
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
        crispasr_imatrix_install(ctx->sched); // no-op unless CRISPASR_IMATRIX_OUT is set
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    }

    // #215e UAF fix: rebuild the encoder graph every invocation. A cached graph's
    // tensor->buffer pointers go stale when the sched's gallocr regrows for a
    // larger interleaved graph (decoder/adapter). The graph-build is fast (~0.1 ms);
    // the sched alloc + compute dominate. Same fix as moss_transcribe (#215).
    if (ctx->cached_enc_ctx) {
        ggml_free(ctx->cached_enc_ctx);
        ctx->cached_enc_ctx = nullptr;
        ctx->cached_enc_gf = nullptr;
    }
    ctx->cached_enc_meta.assign(ctx->compute_meta.size(), 0);
    ggml_init_params aip = {ctx->cached_enc_meta.size(), ctx->cached_enc_meta.data(), true};
    ctx->cached_enc_ctx = ggml_init(aip);
    ggml_cgraph* gf = canary_build_graph_encoder(ctx, T_mel, ctx->cached_enc_ctx);
    ctx->cached_enc_gf = gf;
    ctx->cached_enc_T_mel = T_mel;

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "canary: failed to alloc encoder graph\n");
        return {};
    }

    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2;
    auto pe = core_conformer::make_pos_enc((int)ctx->model.hparams.enc_d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "canary: encoder compute failed\n");
        return {};
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "enc_out");
    if (!out)
        return {};
    const int d = (int)out->ne[0];
    const int Te = (int)out->ne[1];
    if (out_T_enc)
        *out_T_enc = Te;

    std::vector<float> result((size_t)d * Te);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

static std::vector<float> canary_project_encoder(canary_context* ctx, const float* enc_data, int T_enc) {
    const auto& model = ctx->model;
    const auto& hp = model.hparams;
    if (!hp.has_encoder_decoder_proj)
        return {};

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor* enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.enc_d_model, T_enc);
    ggml_set_name(enc, "enc_native");
    ggml_set_input(enc);
    ggml_tensor* projected = ggml_add(ctx0, ggml_mul_mat(ctx0, model.enc_proj_w, enc), model.enc_proj_b);
    ggml_set_name(projected, "enc_projected");
    ggml_build_forward_expand(gf, projected);
    ggml_free(ctx0);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "canary: encoder projection alloc failed\n");
        return {};
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_native"), enc_data, 0,
                            (size_t)hp.enc_d_model * T_enc * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "canary: encoder projection compute failed\n");
        return {};
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "enc_projected");
    std::vector<float> result((size_t)hp.dec_d_model * T_enc);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Decoder KV cache + cross-KV allocation
// ===========================================================================

static void canary_alloc_kv(canary_context* ctx) {
    if (ctx->kv_buf)
        return;
    const auto& hp = ctx->model.hparams;
    const int head_dim = (int)hp.dec_head_dim;
    const int n_heads = (int)hp.dec_n_heads;
    const int max_ctx = (int)hp.max_dec_ctx;
    const int n_layers = (int)hp.dec_n_layers;

    ggml_init_params p = {
        /*mem_size=*/ggml_tensor_overhead() * 4,
        /*mem_buffer=*/nullptr,
        /*no_alloc=*/true,
    };
    ctx->kv_ctx = ggml_init(p);

    // PLAN #60e + #69e: per-half KV dtype. Canary's per-step write
    // goes through core_attn::kv_cache_write (PLAN #73), and the
    // read path adds a ggml_cast(F32) before the permute+cont chain
    // when the cache is quant — keeps memory savings, gives up some
    // read-bandwidth savings.
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("canary");
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, head_dim, max_ctx, n_heads, n_layers);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, head_dim, max_ctx, n_heads, n_layers);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    // PLAN #69b: optional KV-on-CPU spill for VRAM-tight users.
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "canary");
    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, kv_backend);
}

// Build the cross-K/V tensors from encoder output. Called once per slice.
// enc: flat [T_enc * d_model] (row-major, ne[0]=d_model fastest)
static void canary_build_cross_kv(canary_context* ctx, const float* enc_data, int T_enc) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.dec_d_model;
    const int head_dim = (int)hp.dec_head_dim;
    const int n_heads = (int)hp.dec_n_heads;
    const int n_layers = (int)hp.dec_n_layers;

    // Allocate cross_kv tensors on a CPU buffer (small, ~1 MB per layer for T_enc≈100)
    if (ctx->cross_ctx) {
        ggml_backend_buffer_free(ctx->cross_buf);
        ggml_free(ctx->cross_ctx);
        ctx->cross_ctx = nullptr;
        ctx->cross_buf = nullptr;
        ctx->cross_k.clear();
        ctx->cross_v.clear();
    }

    ggml_init_params p = {
        /*mem_size=*/ggml_tensor_overhead() * (n_layers * 2 + 8),
        /*mem_buffer=*/nullptr,
        /*no_alloc=*/true,
    };
    ctx->cross_ctx = ggml_init(p);
    ctx->cross_k.resize(n_layers);
    ctx->cross_v.resize(n_layers);

    for (int il = 0; il < n_layers; il++) {
        // ggml_flash_attn_ext requires F16 K/V on the CPU backend (the F32 path
        // is not implemented and falls through to broken behaviour). Match the
        // pattern cohere.cpp uses.
        ctx->cross_k[il] = ggml_new_tensor_3d(ctx->cross_ctx, GGML_TYPE_F16, head_dim, T_enc, n_heads);
        ctx->cross_v[il] = ggml_new_tensor_3d(ctx->cross_ctx, GGML_TYPE_F16, head_dim, T_enc, n_heads);
    }
    // Allocate on the SAME backend the decoder graph runs on. Allocating on
    // a separate CPU backend instance breaks the scheduler's cross-graph
    // tensor references — even when both backends happen to be CPU.
    ctx->cross_buf = ggml_backend_alloc_ctx_tensors(ctx->cross_ctx, ctx->backend);

    // Compute K/V projections per layer using a tiny graph
    for (int il = 0; il < n_layers; il++) {
        const auto& dl = m.dec[il];

        ggml_init_params gp = {
            /*mem_size=*/ctx->compute_meta.size(),
            /*mem_buffer=*/ctx->compute_meta.data(),
            /*no_alloc=*/true,
        };
        ggml_context* gctx = ggml_init(gp);
        ggml_cgraph* gf = ggml_new_graph_custom(gctx, 64, false);

        ggml_tensor* enc = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d, T_enc);
        ggml_set_name(enc, "enc_in");
        ggml_set_input(enc);

        // CK: linear(enc) → bias → reshape [hd, n_heads, T_enc] → permute [hd, T_enc, n_heads]
        ggml_tensor* CK = ggml_add(gctx, ggml_mul_mat(gctx, dl.ca_k_w, enc), dl.ca_k_b);
        CK = ggml_cont(gctx, ggml_permute(gctx, ggml_reshape_3d(gctx, CK, head_dim, n_heads, T_enc), 0, 2, 1, 3));
        ggml_set_name(CK, "CK");

        ggml_tensor* CV = ggml_add(gctx, ggml_mul_mat(gctx, dl.ca_v_w, enc), dl.ca_v_b);
        CV = ggml_cont(gctx, ggml_permute(gctx, ggml_reshape_3d(gctx, CV, head_dim, n_heads, T_enc), 0, 2, 1, 3));
        ggml_set_name(CV, "CV");

        ggml_build_forward_expand(gf, CK);
        ggml_build_forward_expand(gf, CV);

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "canary: cross-kv alloc failed\n");
            ggml_free(gctx);
            return;
        }

        ggml_tensor* enc_in = ggml_graph_get_tensor(gf, "enc_in");
        ggml_backend_tensor_set(enc_in, enc_data, 0, (size_t)d * T_enc * sizeof(float));

        ggml_backend_sched_graph_compute(ctx->sched, gf);

        ggml_tensor* CK_out = ggml_graph_get_tensor(gf, "CK");
        ggml_tensor* CV_out = ggml_graph_get_tensor(gf, "CV");
        std::vector<float> kbuf((size_t)head_dim * T_enc * n_heads);
        std::vector<float> vbuf((size_t)head_dim * T_enc * n_heads);
        ggml_backend_tensor_get(CK_out, kbuf.data(), 0, kbuf.size() * sizeof(float));
        ggml_backend_tensor_get(CV_out, vbuf.data(), 0, vbuf.size() * sizeof(float));

        // Convert F32 → F16 before uploading into the F16 cross-KV slots.
        std::vector<ggml_fp16_t> kbuf16(kbuf.size());
        std::vector<ggml_fp16_t> vbuf16(vbuf.size());
        ggml_fp32_to_fp16_row(kbuf.data(), kbuf16.data(), kbuf.size());
        ggml_fp32_to_fp16_row(vbuf.data(), vbuf16.data(), vbuf.size());
        ggml_backend_tensor_set(ctx->cross_k[il], kbuf16.data(), 0, kbuf16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(ctx->cross_v[il], vbuf16.data(), 0, vbuf16.size() * sizeof(ggml_fp16_t));

        ggml_free(gctx);
    }
}

// ===========================================================================
// Decoder per-step graph (autoregressive, with self-attn KV cache + pre-built cross-KV)
// ===========================================================================

static ggml_cgraph* canary_build_graph_decoder(canary_context* ctx, int n_tokens, int offset) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.dec_d_model;
    const int n_heads = (int)hp.dec_n_heads;
    const int head_dim = (int)hp.dec_head_dim;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // PLAN #73: causal mask input for ggml_flash_attn_ext on the
    // self-attention path. Created here once and reused by every layer.
    // Shape [L, n_tokens] F16: 0 = unmasked, -INF = masked. Only needed
    // for prefill (n_tokens > 1); decode steps (n_tokens=1) pass nullptr.
    const int L = offset + n_tokens;
    ggml_tensor* sa_mask = nullptr;
    if (n_tokens > 1) {
        sa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, L, n_tokens);
        ggml_set_name(sa_mask, "sa_mask");
        ggml_set_input(sa_mask);
    }
    ggml_tensor* embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(embd, "embd");
    ggml_set_input(embd);

    ggml_tensor* position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    // Token embed + positional embed (learned)
    ggml_tensor* cur =
        ggml_add(ctx0, ggml_get_rows(ctx0, m.dec_embed_w, embd), ggml_get_rows(ctx0, m.dec_pos_enc, position));

    // Embedding LayerNorm
    cur = ggml_norm(ctx0, cur, kLayerNormEps);
    cur = ggml_mul(ctx0, cur, m.dec_embed_ln_w);
    cur = ggml_add(ctx0, cur, m.dec_embed_ln_b);

    for (uint32_t il = 0; il < hp.dec_n_layers; il++) {
        const auto& dl = m.dec[il];
        ggml_tensor* inpL = cur;

        // ---- Self-attention (PRE-LN: x = x + SA(LN1(x)), causal, with KV cache) ----
        // Confirmed via NeMo source (transformer_decoders.py forward_preln):
        //   residual = decoder_query
        //   decoder_query = layer_norm_1(decoder_query)
        //   decoder_keys  = layer_norm_1(decoder_keys)
        //   self_attn_output = first_sub_layer(decoder_query, decoder_keys, decoder_keys, mask)
        //   self_attn_output += residual
        cur = ggml_norm(ctx0, inpL, kLayerNormEps);
        cur = ggml_mul(ctx0, cur, dl.norm_sa_w);
        cur = ggml_add(ctx0, cur, dl.norm_sa_b);

        ggml_tensor* Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_q_w, cur), dl.sa_q_b);
        ggml_tensor* Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_k_w, cur), dl.sa_k_b);
        ggml_tensor* Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_v_w, cur), dl.sa_v_b);

        // PLAN #73: cache write via core_attn helper. F16/F32 caches go
        // the legacy ggml_cpy(view) path (bit-identical to before);
        // Q8_0/Q4_0 caches go ggml_set_rows(position). The `position`
        // tensor is already populated with [offset, offset+1, …,
        // offset+n_tokens-1] for the embedding-table lookup, so it
        // doubles as the row-index input.
        {
            ggml_tensor* K_perm =
                ggml_permute(ctx0, ggml_reshape_3d(ctx0, Kcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);
            ggml_tensor* V_perm =
                ggml_permute(ctx0, ggml_reshape_3d(ctx0, Vcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);
            core_attn::kv_cache_write(ctx0, gf, K_perm, V_perm, ctx->kv_k, ctx->kv_v, il, offset, n_tokens, position);
        }

        // PLAN #73: ggml_flash_attn_ext fuses K-mul-Q + softmax + V-mul
        // into one op and natively handles quant K/V (no cast tax). Mask
        // is the shared `sa_mask` graph input populated host-side per
        // call. For decode steps (n_tokens=1) mask is nullptr — the
        // causal constraint is trivially satisfied since there's only
        // one query position.
        ggml_tensor* Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Qcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);
        ggml_tensor* K_full = ggml_view_3d(ctx0, ctx->kv_k, head_dim, L, n_heads, ctx->kv_k->nb[1], ctx->kv_k->nb[2],
                                           il * ctx->kv_k->nb[3]);
        ggml_tensor* V_full = ggml_view_3d(ctx0, ctx->kv_v, head_dim, L, n_heads, ctx->kv_v->nb[1], ctx->kv_v->nb[2],
                                           il * ctx->kv_v->nb[3]);
        ggml_tensor* sa_out = ggml_flash_attn_ext(ctx0, ggml_cont(ctx0, Q), K_full, V_full, sa_mask,
                                                  1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
        // Output: [hd, n_heads, n_tokens] — already in the layout the
        // legacy code produced after its trailing permute(0,2,1,3).
        sa_out = ggml_reshape_2d(ctx0, sa_out, d, n_tokens);
        cur = sa_out;

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_out_w, cur), dl.sa_out_b);
        cur = ggml_add(ctx0, cur, inpL);

        // ---- Cross-attention (PRE-LN: x = x + CA(LN2(x)), no causal mask) ----
        ggml_tensor* inpCA = cur;
        cur = ggml_norm(ctx0, cur, kLayerNormEps);
        cur = ggml_mul(ctx0, cur, dl.norm_ca_w);
        cur = ggml_add(ctx0, cur, dl.norm_ca_b);

        ggml_tensor* CQ = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ca_q_w, cur), dl.ca_q_b);
        CQ = ggml_reshape_3d(ctx0, CQ, head_dim, n_heads, n_tokens);
        CQ = ggml_permute(ctx0, CQ, 0, 2, 1, 3);

        ggml_tensor* CK = ctx->cross_k[il];
        ggml_tensor* CV = ctx->cross_v[il];

        ggml_tensor* ca_out = ggml_flash_attn_ext(ctx0, CQ, CK, CV, nullptr, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);

        // Capture cross-attention weights for the last decoder layer in
        // single-token generation mode. Used by the DTW timestamp pass.
        // CK shape: [head_dim, T_enc, n_heads] (F16)
        // CQ shape: [head_dim, n_tokens=1, n_heads]
        // ca_w = softmax( CK^T @ CQ / sqrt(head_dim) ) → [T_enc, 1, n_heads]
        if (ctx->collect_attn && (int)il == (int)hp.dec_n_layers - 1 && n_tokens == 1) {
            ggml_tensor* ca_w = ggml_mul_mat(ctx0, CK, CQ); // [T_enc, 1, n_heads]
            ca_w = ggml_soft_max_ext(ctx0, ca_w, nullptr, 1.0f / sqrtf((float)head_dim), 0.0f);
            ggml_set_name(ca_w, "ca_attn_w");
            ggml_build_forward_expand(gf, ca_w);
        }

        cur = ggml_reshape_2d(ctx0, ca_out, d, n_tokens);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ca_out_w, cur), dl.ca_out_b);
        cur = ggml_add(ctx0, cur, inpCA);

        // ---- FFN (PRE-LN: x = x + FFN(LN3(x)), ReLU activation) ----
        ggml_tensor* inpFF = cur;
        cur = ggml_norm(ctx0, cur, kLayerNormEps);
        cur = ggml_mul(ctx0, cur, dl.norm_ff_w);
        cur = ggml_add(ctx0, cur, dl.norm_ff_b);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ff_in_w, cur), dl.ff_in_b);
        cur = hp.decoder_activation == "relu" ? ggml_relu(ctx0, cur) : ggml_silu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ff_out_w, cur), dl.ff_out_b);
        cur = ggml_add(ctx0, cur, inpFF);
    }

    // Final layer norm + output head
    cur = ggml_norm(ctx0, cur, kLayerNormEps);
    cur = ggml_mul(ctx0, cur, m.dec_final_ln_w);
    cur = ggml_add(ctx0, cur, m.dec_final_ln_b);

    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.dec_head_w, cur), m.dec_head_b);
    ggml_set_name(cur, "logits");

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run one decoder step (or batch). Returns logits for the LAST token.
static std::vector<float> canary_decode_step(canary_context* ctx, const int* tokens, int n_tokens, int offset) {
    canary_alloc_kv(ctx);
    ggml_cgraph* gf = canary_build_graph_decoder(ctx, n_tokens, offset);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "canary: decoder alloc failed\n");
        return {};
    }

    ggml_tensor* embd = ggml_graph_get_tensor(gf, "embd");
    ggml_backend_tensor_set(embd, tokens, 0, n_tokens * sizeof(int));

    std::vector<int> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos[i] = offset + i;
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "position");
    ggml_backend_tensor_set(pos_in, pos.data(), 0, n_tokens * sizeof(int));

    // PLAN #73: populate causal mask for ggml_flash_attn_ext. Only built
    // for prefill (n_tokens > 1); decode steps pass nullptr mask.
    if (n_tokens > 1) {
        const int L = offset + n_tokens;
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask((size_t)n_tokens * L, zero);
        for (int q = 0; q < n_tokens; q++) {
            // q-th query at absolute position offset+q. Mask out keys
            // with absolute index > offset+q (i.e. future positions).
            for (int k = offset + q + 1; k < L; k++)
                mask[(size_t)q * L + k] = neg_inf;
        }
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "sa_mask");
        if (mask_in)
            ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "canary: decoder compute failed\n");
        return {};
    }

    // Capture cross-attention for DTW (only on single-token generation steps).
    if (ctx->collect_attn && n_tokens == 1) {
        ggml_tensor* ca_w_t = ggml_graph_get_tensor(gf, "ca_attn_w");
        if (ca_w_t) {
            const int T_enc = (int)ca_w_t->ne[0];
            const int n_heads = (int)ca_w_t->ne[2];
            ctx->attn_T_enc = T_enc;
            ctx->attn_n_heads = n_heads;
            std::vector<float> w((size_t)T_enc * n_heads);
            ggml_backend_tensor_get(ca_w_t, w.data(), 0, w.size() * sizeof(float));
            ctx->step_attn.push_back(std::move(w));
        }
    }

    ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    const int V = (int)logits->ne[0];
    std::vector<float> all((size_t)V * n_tokens);
    ggml_backend_tensor_get(logits, all.data(), 0, all.size() * sizeof(float));

    // Return logits for the last position only
    return std::vector<float>(all.begin() + (size_t)(n_tokens - 1) * V, all.end());
}

// ===========================================================================
// Greedy decode: build prompt, run encoder, run decoder steps until EOS
// ===========================================================================

static std::vector<int> canary_build_prompt(canary_context* ctx, const std::string& src, const std::string& tgt,
                                            bool punctuation) {
    const auto& hp = ctx->model.hparams;
    if (hp.new_schema) {
        auto language_id = [&](const std::string& language) {
            for (size_t i = 0; i < hp.languages.size(); i++)
                if (hp.languages[i] == language)
                    return hp.language_ids[i];
            return -1;
        };

        std::vector<int> prompt;
        prompt.reserve(hp.tokenizer_single_sp ? 10 : 9);
        if (hp.tokenizer_single_sp)
            prompt.push_back(canary_str_to_token(ctx, "\xE2\x96\x81"));
        prompt.push_back(hp.startofcontext_id);
        prompt.push_back(hp.startoftranscript_id);
        prompt.push_back(canary_str_to_token(ctx, "<|emo:undefined|>"));
        prompt.push_back(language_id(src));
        prompt.push_back(language_id(tgt));
        prompt.push_back(punctuation ? hp.pnc_id : hp.nopnc_id);
        prompt.push_back(hp.noitn_id);
        prompt.push_back(hp.notimestamp_id);
        prompt.push_back(hp.nodiarize_id);

        for (int id : prompt) {
            if (id < 0 || id >= (int)ctx->vocab.id_to_token.size()) {
                fprintf(stderr,
                        "canary: failed to build metadata-driven canary2 prompt "
                        "(src='%s', tgt='%s', pnc=%d, single_sp=%d)\n",
                        src.c_str(), tgt.c_str(), (int)punctuation, (int)hp.tokenizer_single_sp);
                return {};
            }
        }
        return prompt;
    }

    auto tok = [&](const char* s) { return canary_str_to_token(ctx, s); };
    std::string src_tok = "<|" + src + "|>";
    std::string tgt_tok = "<|" + tgt + "|>";

    // NeMo CanaryBPETokenizer (shared with Cohere) prompt format:
    //   <|startofcontext|>
    //   <|startoftranscript|>
    //   <|emo:undefined|>
    //   <|src_lang|>
    //   <|target_lang|>
    //   <|pnc|>  or  <|nopnc|>
    //   <|notimestamp|>
    //   <|nodiarize|>
    std::vector<int> p;
    p.push_back(tok("<|startofcontext|>"));
    p.push_back(tok("<|startoftranscript|>"));
    p.push_back(tok("<|emo:undefined|>"));
    p.push_back(canary_str_to_token(ctx, src_tok.c_str()));
    p.push_back(canary_str_to_token(ctx, tgt_tok.c_str()));
    p.push_back(tok(punctuation ? "<|pnc|>" : "<|nopnc|>"));
    p.push_back(tok("<|notimestamp|>"));
    p.push_back(tok("<|nodiarize|>"));

    static const char* names[] = {"<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
                                  src_tok.c_str(),      tgt_tok.c_str(),         punctuation ? "<|pnc|>" : "<|nopnc|>",
                                  "<|notimestamp|>",    "<|nodiarize|>"};
    for (size_t i = 0; i < p.size(); i++) {
        if (p[i] < 0) {
            fprintf(stderr,
                    "canary: prompt contains an unknown token: '%s' "
                    "(vocab_size=%d, src='%s', tgt='%s', punc=%d)\n",
                    names[i], (int)ctx->vocab.id_to_token.size(), src_tok.c_str(), tgt_tok.c_str(), (int)punctuation);
            return {};
        }
    }
    return p;
}

static std::string spiece_to_text(const std::string& piece) {
    if (piece.empty())
        return "";
    if (piece.size() >= 2 && piece[0] == '<' && piece.back() == '>')
        return "";
    if (piece.size() >= 3 && (unsigned char)piece[0] == 0xE2 && (unsigned char)piece[1] == 0x96 &&
        (unsigned char)piece[2] == 0x81) {
        return std::string(" ") + piece.substr(3);
    }
    return piece;
}

// ===========================================================================
// BatchNorm folding (load-time, once) — same trick as parakeet/cohere
// ===========================================================================

static void canary_fold_batchnorm(canary_model& model) {
    const int d = (int)model.hparams.enc_d_model;
    const int K = (int)model.hparams.conv_kernel;
    const float eps = 1e-5f;

    for (uint32_t il = 0; il < model.hparams.enc_n_layers; il++) {
        auto& e = model.enc[il];
        if (!e.conv_dw_w || !e.conv_dw_b || !e.conv_bn_w || !e.conv_bn_b || !e.conv_bn_rm || !e.conv_bn_rv)
            continue;

        std::vector<float> bn_mean(d), bn_var(d), bn_w(d), bn_b(d), dw_b(d);
        ggml_backend_tensor_get(e.conv_bn_rm, bn_mean.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_rv, bn_var.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_w, bn_w.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_b, bn_b.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));

        std::vector<float> s(d);
        for (int c = 0; c < d; c++)
            s[c] = bn_w[c] / sqrtf(bn_var[c] + eps);

        std::vector<float> w_f32 = core_cpu::to_f32(e.conv_dw_w); // F32/F16/quantized-safe read
        for (int c = 0; c < d; c++)
            for (int ki = 0; ki < K; ki++)
                w_f32[ki + c * K] *= s[c];
        // Write back in the tensor's native dtype — an F32 conv_dw_w (--f32-encoder
        // GGUFs / diff-harness validation) must NOT be clobbered with F16 (c9a4de65).
        if (e.conv_dw_w->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(e.conv_dw_w, w_f32.data(), 0, w_f32.size() * sizeof(float));
        } else {
            std::vector<ggml_fp16_t> w_f16(w_f32.size());
            for (size_t i = 0; i < w_f16.size(); i++)
                w_f16[i] = ggml_fp32_to_fp16(w_f32[i]);
            ggml_backend_tensor_set(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
        }

        // Fold into existing dw_b: b'[c] = (dw_b[c] - mean[c]) * s[c] + bn_b[c]
        for (int c = 0; c < d; c++)
            dw_b[c] = (dw_b[c] - bn_mean[c]) * s[c] + bn_b[c];
        ggml_backend_tensor_set(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
    }

    fprintf(stderr, "canary: BN folded into conv_dw weights for %u layers\n", model.hparams.enc_n_layers);
}

// ===========================================================================
// Backend selection
// ===========================================================================

static ggml_backend_t pick_backend() {
    // crispasr_init_gpu_backend() tries all compiled backends in priority
    // order (CUDA > Metal > Vulkan > CPU) and returns the first one
    // that initialises. This replaces the old Metal/CUDA-specific
    // #ifdef chain and adds Vulkan support for free.
    ggml_backend_t b = crispasr_init_gpu_backend();
    return b ? b : core_cpu_backend::init();
}

static ggml_backend_t pick_backend(bool use_gpu) {
    return use_gpu ? pick_backend() : core_cpu_backend::init();
}

// ===========================================================================
// Public C API
// ===========================================================================

// ---- Stage-level entry points for crispasr-diff ----

extern "C" float* canary_compute_mel(struct canary_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                     int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    int T_mel = 0;
    auto mel = canary_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;
    const int n_mels = (int)ctx->model.hparams.n_mels;
    if (out_n_mels)
        *out_n_mels = n_mels;
    if (out_T_mel)
        *out_T_mel = T_mel;
    float* r = (float*)malloc(mel.size() * sizeof(float));
    if (!r)
        return nullptr;
    std::memcpy(r, mel.data(), mel.size() * sizeof(float));
    return r;
}

extern "C" float* canary_run_encoder(struct canary_context* ctx, const float* mel, int n_mels, int T_mel,
                                     int* out_T_enc, int* out_d_model) {
    if (!ctx || !mel || T_mel <= 0)
        return nullptr;
    int T_enc = 0;
    auto enc = canary_encode_mel(ctx, mel, n_mels, T_mel, &T_enc);
    if (enc.empty())
        return nullptr;
    const int d = (int)ctx->model.hparams.enc_d_model;
    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d;
    float* r = (float*)malloc(enc.size() * sizeof(float));
    if (!r)
        return nullptr;
    std::memcpy(r, enc.data(), enc.size() * sizeof(float));
    return r;
}

extern "C" int canary_run_encoder_staged(struct canary_context* ctx, const float* mel, int n_mels, int T_mel,
                                         canary_stage_cb cb, void* userdata) {
    if (!ctx || !mel || T_mel <= 0 || !cb)
        return -1;
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "canary: mel feature mismatch (%d vs %d)\n", n_mels, (int)ctx->model.hparams.n_mels);
        return -1;
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 24576, false, false);
        crispasr_imatrix_install(ctx->sched); // no-op unless CRISPASR_IMATRIX_OUT is set
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(ggml_tensor_overhead() * 24576 + ggml_graph_overhead_custom(24576, false));
    }

    ggml_cgraph* gf = canary_build_graph_encoder_staged(ctx, T_mel);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "canary: failed to alloc staged encoder graph\n");
        return -1;
    }

    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2;
    auto pe = core_conformer::make_pos_enc((int)ctx->model.hparams.enc_d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "canary: staged encoder compute failed\n");
        return -1;
    }

    const int d = (int)ctx->model.hparams.enc_d_model;

    // Retrieve and deliver each named snapshot in order.
    // deliver() assumes ne[0]=d_model (standard encoder output format).
    auto deliver = [&](const char* name) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t)
            return;
        const int t_steps = (int)t->ne[1];
        std::vector<float> buf((size_t)d * t_steps);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        cb(name, buf.data(), t_steps, d, userdata);
    };
    // deliver_dyn() uses the tensor's own ne[0] as the feature dim.
    // Used for intermediate conv snaps where the feature count differs.
    auto deliver_dyn = [&](const char* name) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t)
            return;
        const int feat = (int)t->ne[0];
        const int t_steps = (int)t->ne[1];
        std::vector<float> buf((size_t)feat * t_steps);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        cb(name, buf.data(), t_steps, feat, userdata);
    };

    deliver_dyn("pre_enc_c0");
    deliver_dyn("pre_enc_c2");
    deliver_dyn("pre_enc_c3");
    deliver_dyn("pre_enc_c5");
    deliver_dyn("pre_enc_c6");
    deliver("pre_enc_out");

    const int n_layers = (int)ctx->model.hparams.enc_n_layers;
    char lbuf[32];
    for (int il = 0; il < n_layers; il++) {
        snprintf(lbuf, sizeof(lbuf), "enc_L%02d", il);
        deliver(lbuf);
    }

    deliver("enc_out");

    return 0;
}

extern "C" struct canary_context_params canary_context_default_params(void) {
    canary_context_params p = {};
    p.n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    p.use_flash = false;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

extern "C" struct canary_context* canary_init_from_file(const char* path_model, struct canary_context_params params) {
    auto* ctx = new canary_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = pick_backend(params.use_gpu);
    ctx->backend_cpu = core_cpu_backend::init();
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    if (!canary_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        canary_free(ctx);
        return nullptr;
    }
    canary_fold_batchnorm(ctx->model);

    // Repack F16 conv pw1/pw2 to Q8_0 (issue #81 — the 3D conv layout dodges
    // crispasr-quantize, and the CPU F16 mul_mat path is ~6x slower than Q8_0).
    {
        auto& m = ctx->model;
        std::vector<core_conformer::BlockWeights*> layers;
        for (auto& e : m.enc)
            layers.push_back(&e);
        const bool quantized = !m.enc.empty() && m.enc[0].attn_q_w && ggml_is_quantized(m.enc[0].attn_q_w->type);
        core_conformer::repack_conv_pw_q8(layers, ctx->backend, quantized, m.pw_q8, "canary");
        core_conformer::fuse_qkv(layers, ctx->backend, m.qkv_fused, "canary");
    }
    return ctx;
}

extern "C" void canary_free(struct canary_context* ctx) {
    if (!ctx)
        return;
    if (ctx->cached_enc_ctx)
        ggml_free(ctx->cached_enc_ctx);
    if (ctx->cross_buf)
        ggml_backend_buffer_free(ctx->cross_buf);
    if (ctx->cross_ctx)
        ggml_free(ctx->cross_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    ctx->model.pw_q8.free();
    ctx->model.qkv_fused.free();
    if (ctx->model.buf)
        core_gguf::release_weight_buffer(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void canary_result_free(struct canary_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->tokens);
    free(r->words);
    free(r);
}

extern "C" void canary_set_temperature(struct canary_context* ctx, float temperature, uint64_t seed) {
    if (!ctx)
        return;
    ctx->decode_temperature = temperature;
    ctx->decode_seed = seed;
}

extern "C" void canary_set_beam_size(struct canary_context* ctx, int n) {
    if (!ctx)
        return;
    ctx->beam_size = n > 0 ? n : 1;
}

// #292: forward --max-new-tokens. <= 0 keeps the 256 default.
extern "C" void canary_set_max_new_tokens(struct canary_context* ctx, int n) {
    if (ctx && n > 0)
        ctx->max_new_tokens = n;
}

extern "C" int canary_n_vocab(struct canary_context* ctx) {
    return (int)ctx->model.hparams.vocab_size;
}
extern "C" int canary_n_mels(struct canary_context* ctx) {
    return (int)ctx->model.hparams.n_mels;
}
extern "C" int canary_sample_rate(struct canary_context* ctx) {
    return (int)ctx->model.hparams.sample_rate;
}
extern "C" int canary_frame_dur_cs(struct canary_context* ctx) {
    return (int)ctx->model.hparams.frame_dur_cs;
}

extern "C" const char* canary_token_to_str(struct canary_context* ctx, int id) {
    if (id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[id].c_str();
}

extern "C" int canary_str_to_token(struct canary_context* ctx, const char* str) {
    auto it = ctx->vocab.token_to_id.find(str);
    return it != ctx->vocab.token_to_id.end() ? it->second : -1;
}

extern "C" int canary_test_load(struct canary_context* ctx) {
    fprintf(
        stderr,
        "canary: load test OK\n"
        "  vocab_size  = %d\n"
        "  enc_layers  = %d\n"
        "  enc_d_model = %d\n"
        "  enc_heads   = %d\n"
        "  enc_head_dim= %d\n"
        "  enc_ff_dim  = %d\n"
        "  dec_layers  = %d\n"
        "  dec_d_model = %d\n"
        "  dec_heads   = %d\n"
        "  dec_head_dim= %d\n"
        "  dec_ff_dim  = %d\n"
        "  max_dec_ctx = %d\n"
        "  n_mels      = %d\n"
        "  sample_rate = %d\n"
        "  frame_dur_cs= %d\n",
        (int)ctx->model.hparams.vocab_size, (int)ctx->model.hparams.enc_n_layers, (int)ctx->model.hparams.enc_d_model,
        (int)ctx->model.hparams.enc_n_heads, (int)ctx->model.hparams.enc_head_dim, (int)ctx->model.hparams.enc_ff_dim,
        (int)ctx->model.hparams.dec_n_layers, (int)ctx->model.hparams.dec_d_model, (int)ctx->model.hparams.dec_n_heads,
        (int)ctx->model.hparams.dec_head_dim, (int)ctx->model.hparams.dec_ff_dim, (int)ctx->model.hparams.max_dec_ctx,
        (int)ctx->model.hparams.n_mels, (int)ctx->model.hparams.sample_rate, (int)ctx->model.hparams.frame_dur_cs);

    // Confirm a few special tokens resolve
    const char* specials[] = {
        "<|startoftranscript|>", "<|endoftext|>", "<|en|>", "<|de|>", "<|fr|>", "<|es|>", "<|nopnc|>",
        "<|notimestamp|>",       "<|nodiarize|>",
    };
    for (const char* s : specials) {
        int id = canary_str_to_token(ctx, s);
        fprintf(stderr, "  token %-22s = %d\n", s, id);
    }
    return 0;
}

extern "C" int canary_test_encoder(struct canary_context* ctx, int T_mel) {
    int n_mels = (int)ctx->model.hparams.n_mels;
    std::vector<float> mel((size_t)n_mels * T_mel, 0.0f);
    int T_enc = 0;
    auto out = canary_encode_mel(ctx, mel.data(), n_mels, T_mel, &T_enc);
    if (out.empty())
        return -1;
    fprintf(stderr, "canary: encoder OK — T_mel=%d → T_enc=%d  d=%d  out[0..3]=%g %g %g %g\n", T_mel, T_enc,
            (int)ctx->model.hparams.enc_d_model, (double)out[0], (double)out[1], (double)out[2], (double)out[3]);
    return T_enc;
}

// PLAN #114 P3 second half — post-encode pipeline shared by
// canary_transcribe_ex (single-pass) and canary_transcribe_streamed
// (parakeet-style chunked-encode + concat for long audio).
static canary_result* canary_finish_from_encoder(canary_context* ctx, const float* enc_data, int T_enc,
                                                 const char* source_lang, const char* target_lang, bool punctuation,
                                                 int64_t t_offset_cs);

extern "C" struct canary_result* canary_transcribe_ex(struct canary_context* ctx, const float* samples, int n_samples,
                                                      const char* source_lang, const char* target_lang,
                                                      bool punctuation, int64_t t_offset_cs) {
    if (!ctx || !samples || n_samples <= 0 || !source_lang || !target_lang)
        return nullptr;

    // 1. Mel
    int T_mel = 0;
    std::vector<float> mel;
    {
        canary_bench_stage _b("mel");
        mel = canary_compute_mel_impl(ctx, samples, n_samples, T_mel);
    }
    if (mel.empty())
        return nullptr;

    // 2. Encoder
    int T_enc = 0;
    std::vector<float> enc;
    {
        canary_bench_stage _b("encoder");
        enc = canary_encode_mel(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, &T_enc);
    }
    if (enc.empty())
        return nullptr;

    canary_bench_stage _b("decoder");
    return canary_finish_from_encoder(ctx, enc.data(), T_enc, source_lang, target_lang, punctuation, t_offset_cs);
}

// ---------------------------------------------------------------------------
// NeMo canary-1b-v2 dynamic-chunking blueprint (long-form audio).
//
// Ports, faithfully:
//   - PromptedAudioToTextLhotseDataset._find_optimal_chunk_size / _chunk_waveform
//     (nemo/collections/asr/data/audio_to_text_lhotse_prompted.py)
//   - merge_parallel_chunks (nemo/collections/asr/parts/utils/chunking_utils.py)
//   - lcs_alignment_merge_buffer + longest_common_subsequence_merge
//     (nemo/collections/asr/parts/utils/streaming_utils.py)
//
// The reference splits the RAW WAVEFORM into chunks of a dynamically chosen
// size (30..40 s, picked to maximize the last chunk's length), with a fixed
// 1 s overlap, decodes each chunk independently (so each chunk gets its own
// PerFeatureZ normalization, exactly like a standalone utterance), and merges
// the token streams with an LCS alignment anchored in the 1 s overlap: the
// accumulated buffer keeps everything up to the END of the match and the new
// chunk contributes everything AFTER it — the duplicated acoustic event is
// kept exactly once, from both sides. Audio under 40 s is a single pass.
// ---------------------------------------------------------------------------

// The exact ports of longest_common_subsequence_merge and
// _find_optimal_chunk_size live in core/canary_chunk_merge.h (weight-free, so
// tests pin them against vectors generated from the actual Python functions).

// Word grouping (same as parakeet's; shared by the single-pass decode and the
// blueprint chunk merge, whose token trims invalidate per-chunk words).
static std::vector<canary_word_data> canary_words_from_tokens(const canary_token_data* toks, int n_tokens) {
    std::vector<canary_word_data> words;
    canary_word_data cur = {};
    bool have_cur = false;
    auto is_punct = [](const char* s) {
        if (!s || !*s)
            return false;
        for (const char* p = s; *p; p++) {
            unsigned char c = (unsigned char)*p;
            if (!(c == '.' || c == ',' || c == '?' || c == '!' || c == ';' || c == ':' || c == '\'' || c == '"' ||
                  c == '(' || c == ')' || c == '-'))
                return false;
        }
        return true;
    };
    for (int i = 0; i < n_tokens; i++) {
        const auto& td = toks[i];
        if (!td.text[0])
            continue;
        const bool is_word_start = (td.text[0] == ' ');
        const bool is_p = is_punct(td.text);
        if (is_word_start && !is_p && have_cur) {
            words.push_back(cur);
            cur = {};
            have_cur = false;
        }
        if (!have_cur) {
            cur.t0 = td.t0;
            have_cur = true;
        }
        cur.t1 = td.t1;
        const char* src = td.text + (is_word_start ? 1 : 0);
        size_t cl = strlen(cur.text);
        size_t cap = sizeof(cur.text) - cl - 1;
        size_t add = strlen(src);
        if (add > cap)
            add = cap;
        memcpy(cur.text + cl, src, add);
        cur.text[cl + add] = '\0';
    }
    if (have_cur)
        words.push_back(cur);
    return words;
}

// LEGACY pre-blueprint long-audio path (PLAN #114 P3): global-mel 8 s / 2 s
// overlapping chunks with LCS prefix-drop + word-snap + splice-punctuation
// cleanup. This is NOT what NeMo does for canary — kept, gated behind
// CRISPASR_CANARY_LEGACY_STREAM=1, purely as the regression-bisection arm
// (never delete the working path). CRISPASR_CANARY_SEAM_DEDUP only applies
// here.
static struct canary_result* canary_transcribe_streamed_legacy(struct canary_context* ctx, const float* samples,
                                                               int n_samples, const char* source_lang,
                                                               const char* target_lang, bool punctuation,
                                                               int64_t t_offset_cs, int chunk_seconds,
                                                               int overlap_seconds) {
    if (!ctx || !samples || n_samples <= 0 || !source_lang || !target_lang)
        return nullptr;
    if (chunk_seconds <= 0)
        chunk_seconds = 8;
    if (overlap_seconds < 0)
        overlap_seconds = 2;
    if (overlap_seconds >= chunk_seconds)
        overlap_seconds = chunk_seconds / 4;

    const auto& hp = ctx->model.hparams;
    const int n_mels = (int)hp.n_mels;
    const int hop = (int)hp.hop_length;
    const int SR = (int)hp.sample_rate;

    // ---- Pass 1: mel for the full audio (global PerFeatureZ) ----
    int T_mel = 0;
    auto mel_full = canary_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel_full.empty() || T_mel <= 0)
        return nullptr;

    // ---- Pass 2: per-chunk encode + per-chunk AED decode (NeMo
    // FrameBatchMultiTaskAED analogon). Each chunk re-injects the
    // <lang> / <task> / <pnc> prompt prefix, so the AED decoder
    // doesn't treat the chunk boundary as <eos>. Per-chunk results
    // are concatenated text-wise; token/word timestamps land in the
    // global time window via per-chunk t_offset arithmetic.
    const int chunk_mel_frames = chunk_seconds * SR / hop;
    const int overlap_mel_frames = overlap_seconds * SR / hop;
    const int shift_mel_frames = chunk_mel_frames - overlap_mel_frames;

    std::string full_text;
    std::vector<canary_token_data> all_tokens;
    std::vector<canary_word_data> all_words;
    int chunks_processed = 0;

    // LCS dedup window: bound how far into the previous chunk we look for
    // boundary duplication. Conservative — overlap_seconds at ~3 tok/s
    // (canary's average emission rate on en/de/fr/es) gives ~6 tokens for
    // a 2 s overlap; cap at 30 to leave headroom for slower regimes.
    const int delay_tokens = std::min(30, std::max(8, overlap_seconds * 5));

    for (int mel_offset = 0; mel_offset < T_mel; mel_offset += shift_mel_frames) {
        const int chunk_end = std::min(T_mel, mel_offset + chunk_mel_frames);
        const int chunk_T = chunk_end - mel_offset;
        if (chunk_T <= 0)
            break;
        int T_enc = 0;
        auto enc = canary_encode_mel(ctx, mel_full.data() + (size_t)mel_offset * n_mels, n_mels, chunk_T, &T_enc);
        if (enc.empty() || T_enc <= 0)
            continue;

        const int64_t chunk_t_offset_cs = t_offset_cs + (int64_t)mel_offset * hop * 100 / SR;
        canary_result* part = canary_finish_from_encoder(ctx, enc.data(), T_enc, source_lang, target_lang, punctuation,
                                                         chunk_t_offset_cs);
        if (!part)
            continue;

        // PLAN #114 P3 polish: boundary-overlap dedup via LCS-merge across
        // adjacent chunks. NeMo's `streaming_utils.longest_common_subsequence_merge`
        // (the same primitive `core/crispasr_lcs::lcs_dedup_prefix_count`
        // wraps) finds the LCS between the tail of accumulated tokens and
        // the head of the current chunk's tokens. If the LCS length >=
        // kMinMergeSubsequenceLen (default 3), the matched prefix of the
        // current chunk is dropped before accumulation.
        int n_skip = 0;
        if (chunks_processed > 0 && part->n_tokens > 0 && !all_tokens.empty()) {
            const int tail_size = std::min(delay_tokens, (int)all_tokens.size());
            // PLAN #114 P3 polish — case-insensitive LCS. The AED can
            // re-emit the same audio with different capitalization at
            // chunk boundaries (e.g. "world's say for" in chunk 1 vs
            // "World's Save for" in chunk 2 — different raw ids but
            // semantically duplicate). Look up each id via
            // vocab.id_to_canonical_lc[id] (smallest id whose
            // lowercase-ASCII text matches) before running LCS. Indices
            // line up with the raw token vectors, so the returned
            // slice_count is still the right number of leading raw
            // tokens to drop.
            auto canon = [&](int32_t id) -> int32_t {
                if (id < 0 || id >= (int32_t)ctx->vocab.id_to_canonical_lc.size())
                    return id;
                return ctx->vocab.id_to_canonical_lc[(size_t)id];
            };
            std::vector<int32_t> prev_tail;
            prev_tail.reserve((size_t)tail_size);
            for (int i = (int)all_tokens.size() - tail_size; i < (int)all_tokens.size(); i++)
                prev_tail.push_back(canon(all_tokens[(size_t)i].id));
            std::vector<int32_t> curr_ids;
            curr_ids.reserve((size_t)part->n_tokens);
            for (int i = 0; i < part->n_tokens; i++)
                curr_ids.push_back(canon(part->tokens[i].id));
            n_skip = crispasr_lcs::lcs_dedup_prefix_count(prev_tail, curr_ids);
        }

        // PLAN #114 P3 polish — word-snap heuristic. The LCS operates on
        // token ids and may drop a prefix that ends mid-word, because
        // BPE-split points inside a duplicated word region don't always
        // align between chunks (chunk 1 emits `[▁Geschirrt, uches]` for
        // "Geschirrtuches"; chunk 2 emits `[▁tuch, ▁umfassen, ...]` for
        // the same audio segment — only `▁umfassen` matches the LCS, so
        // we drop 1 token and leave `▁tuch` as a stray word-fragment in
        // the output). Extend the drop forward until the next token
        // whose text starts with a space (sentencepiece convention:
        // `▁X` decodes to ` X`, so a leading space marks a word start).
        // Trades a few extra tokens for a clean prefix; bounded by
        // n_tokens so we don't drop the whole chunk on a pathological
        // mid-word run.
        if (n_skip > 0 && n_skip < part->n_tokens) {
            while (n_skip < part->n_tokens) {
                const char* t = part->tokens[n_skip].text;
                if (!t || !t[0])
                    break;
                if (t[0] == ' ')
                    break;
                n_skip++;
            }
        }

        // #365: fuzzy seam dedup, UNDER the LCS result.
        //
        // The LCS above compares token ids, so it only fires when the decoder
        // said the same thing twice. Given a different amount of right-context
        // an AED re-words the overlap instead — "s'est rendue compte" in one
        // chunk, "rendu compte" in the next for the same audio — the LCS
        // matches nothing, the whole re-transcription is appended, and the
        // timestamps run backwards because the new chunk carries its own offset.
        //
        // Comparing NORMALISED words with a small edit tolerance sees through
        // the re-wording. Restricted to the nominal overlap window, because only
        // that much of a chunk can be duplicate by construction; and requiring a
        // run of at least two words, so a shared "the" cannot trigger a drop.
        //
        // Deliberately NOT a plain time floor. Trimming the overlap by
        // timestamp alone was measured removing real speech — the leading
        // "Many" from "Many people don't think about them as dinosaurs" — since
        // an AED's boundary timings overshoot and a chunk's first word can
        // legitimately carry an early timestamp.
        // OPT-IN (CRISPASR_CANARY_SEAM_DEDUP=1) until the net effect is proven.
        //
        // Measured on a 600 s clip it removes real duplication — a stray " is."
        // cue, and the doubled "its splendors" and "versions are kept on
        // versions are kept online" — but it also dropped a leading "Many" that
        // I could not confirm was duplicate. Trading one defect for another is
        // not an improvement, and the reporting file (#365, 20+ min French)
        // could not be reproduced here, so this ships gated rather than on.
        static const bool seam_dedup = [] {
            const char* e = crispasr_env::get("CRISPASR_CANARY_SEAM_DEDUP");
            return e && e[0] == '1';
        }();
        if (seam_dedup && !all_tokens.empty() && part->n_tokens > n_skip) {
            const int64_t overlap_end_cs = chunk_t_offset_cs + (int64_t)overlap_seconds * 100;
            auto split_words = [](const std::string& t) {
                std::vector<std::string> w;
                std::string cur;
                for (char c : t) {
                    if (c == ' ' || c == '\t' || c == '\n') {
                        if (!cur.empty())
                            w.push_back(cur), cur.clear();
                    } else {
                        cur.push_back(c);
                    }
                }
                if (!cur.empty())
                    w.push_back(cur);
                return w;
            };
            // Accepted tail: words spoken within overlap_seconds of the seam.
            std::string tail_text;
            for (size_t i = all_tokens.size(); i-- > 0;) {
                if (all_tokens[i].t1 + (int64_t)overlap_seconds * 100 < all_tokens.back().t1)
                    break;
                tail_text.insert(0, all_tokens[i].text);
            }
            // Head: this chunk's tokens that fall inside the overlap window.
            std::string head_text;
            int head_tokens = n_skip;
            for (int i = n_skip; i < part->n_tokens; i++) {
                if (part->tokens[i].t0 > overlap_end_cs)
                    break;
                head_text += part->tokens[i].text;
                head_tokens++;
            }
            const auto tw = split_words(tail_text);
            const auto hw = split_words(head_text);
            const int rep = core_overlap_trim::leading_repeat_len(tw, hw);
            if (rep > 0) {
                // Advance past `rep` whole words of this chunk.
                int words_seen = 0;
                int i = n_skip;
                for (; i < head_tokens && words_seen < rep; i++)
                    if (part->tokens[i].text[0] == ' ' || i == n_skip)
                        words_seen++;
                while (i < head_tokens && part->tokens[i].text[0] != ' ')
                    i++;
                n_skip = std::min(i, part->n_tokens - 1);
            }
        }

        // Rebuild text for this chunk from surviving tokens (so the
        // dedupe matches the token vector exactly; trusting part->text
        // would leak the dropped prefix as a string).
        std::string part_text;
        for (int i = n_skip; i < part->n_tokens; i++)
            part_text += part->tokens[i].text;
        if (!part_text.empty() && part_text[0] == ' ')
            part_text.erase(0, 1);

        if (!part_text.empty()) {
            // PLAN #114 P3 polish — splice-punctuation cleanup. After LCS
            // dedup the chunk boundary can land between a mid-sentence
            // punctuation in the previous chunk (`,` `;` `:`) and a
            // sentence-end punctuation in the surviving prefix of the new
            // chunk (`.` `?` `!`), producing e.g. "for you, . Ask...".
            // The chunk-1 comma was the model's best guess for the
            // continuation point; chunk-2 produced a period because the
            // LCS-dropped middle ended that sentence. Collapse the
            // mid-sentence punctuation in favour of the sentence-end
            // punctuation that survived LCS.
            bool attach_directly = false;
            if (!full_text.empty()) {
                const char last_punct = full_text.back();
                size_t first_nws = 0;
                while (first_nws < part_text.size() && (part_text[first_nws] == ' ' || part_text[first_nws] == '\t'))
                    first_nws++;
                if (first_nws < part_text.size()) {
                    const char first_punct = part_text[first_nws];
                    if ((last_punct == ',' || last_punct == ';' || last_punct == ':') &&
                        (first_punct == '.' || first_punct == '?' || first_punct == '!')) {
                        full_text.pop_back();
                        while (!full_text.empty() && full_text.back() == ' ')
                            full_text.pop_back();
                        // Strip leading whitespace from part_text so the
                        // surviving punctuation attaches directly:
                        //   "for you" + ". Ask..."  →  "for you. Ask..."
                        // (instead of "for you . Ask..." with a stray
                        // space before the period).
                        part_text.erase(0, first_nws);
                        attach_directly = true;
                    }
                }
            }
            if (!attach_directly && !full_text.empty() && full_text.back() != ' ' && part_text[0] != ' ')
                full_text += ' ';
            full_text += part_text;
        }
        for (int i = n_skip; i < part->n_tokens; i++)
            all_tokens.push_back(part->tokens[i]);

        // Words: drop leading ones whose t1 is at or before the last
        // surviving token's t0 (best-effort — words and tokens don't have
        // a strict 1:1 mapping but adjacent boundary-overlap words should
        // share timing with the dropped tokens).
        // Same rule for words. Under CRISPASR_CANARY_SEAM_DEDUP this also runs
        // when the LCS found nothing (n_skip == 0) and clamps to the last
        // accepted word, which is where the visible SRT change comes from —
        // cues are built from WORDS, not tokens. Off by default; see the gate
        // above for why (#365).
        if (part->n_tokens > 0 && (seam_dedup || n_skip > 0)) {
            int64_t cutoff_t0 =
                (n_skip < part->n_tokens) ? part->tokens[(size_t)n_skip].t0 : part->tokens[(size_t)n_skip - 1].t1;
            if (seam_dedup && !all_words.empty())
                cutoff_t0 = std::max(cutoff_t0, all_words.back().t1);
            for (int i = 0; i < part->n_words; i++) {
                if (part->words[i].t1 <= cutoff_t0)
                    continue;
                all_words.push_back(part->words[i]);
            }
        } else {
            for (int i = 0; i < part->n_words; i++)
                all_words.push_back(part->words[i]);
        }

        canary_result_free(part);
        chunks_processed++;
    }

    if (chunks_processed == 0 && all_tokens.empty() && full_text.empty())
        return nullptr;

    canary_result* r = (canary_result*)calloc(1, sizeof(canary_result));
    if (!r)
        return nullptr;
    r->text = strdup(full_text.c_str());
    if (!r->text) {
        canary_result_free(r);
        return nullptr;
    }
    r->n_tokens = (int)all_tokens.size();
    r->tokens = (canary_token_data*)calloc(r->n_tokens > 0 ? (size_t)r->n_tokens : 1, sizeof(canary_token_data));
    if (!r->tokens) {
        canary_result_free(r);
        return nullptr;
    }
    for (int i = 0; i < r->n_tokens; i++)
        r->tokens[i] = all_tokens[(size_t)i];

    r->n_words = (int)all_words.size();
    r->words = (canary_word_data*)calloc(r->n_words > 0 ? (size_t)r->n_words : 1, sizeof(canary_word_data));
    if (!r->words) {
        canary_result_free(r);
        return nullptr;
    }
    for (int i = 0; i < r->n_words; i++)
        r->words[i] = all_words[(size_t)i];

    return r;
}

static canary_result* canary_result_from_merged_tokens(std::vector<canary_token_data>& merged, bool clear_timings,
                                                       int64_t audio_start_cs, int64_t audio_end_cs) {
    if (clear_timings) {
        for (auto& token : merged)
            token.t0 = token.t1 = 0;
    } else {
        // Chunk-local DTW is only a runtime estimate (not Canary's upstream
        // CTC timestamp model). Clamp overlap-boundary spans into one monotonic
        // sequence before exposing them through the existing result ABI.
        int64_t cursor = audio_start_cs;
        for (auto& token : merged) {
            token.t0 = std::min(audio_end_cs, std::max(cursor, std::max(audio_start_cs, token.t0)));
            token.t1 = std::max(token.t0, std::min(audio_end_cs, token.t1));
            cursor = token.t1;
        }
    }

    std::string full_text;
    for (const auto& token : merged)
        full_text += token.text;
    if (!full_text.empty() && full_text[0] == ' ')
        full_text.erase(0, 1);
    auto words = canary_words_from_tokens(merged.data(), (int)merged.size());

    canary_result* result = (canary_result*)calloc(1, sizeof(canary_result));
    if (!result)
        return nullptr;
    result->text = strdup(full_text.c_str());
    if (!result->text) {
        canary_result_free(result);
        return nullptr;
    }
    result->n_tokens = (int)merged.size();
    result->tokens =
        (canary_token_data*)calloc(result->n_tokens > 0 ? (size_t)result->n_tokens : 1, sizeof(canary_token_data));
    if (!result->tokens) {
        canary_result_free(result);
        return nullptr;
    }
    for (int i = 0; i < result->n_tokens; i++)
        result->tokens[i] = merged[(size_t)i];

    result->n_words = (int)words.size();
    result->words =
        (canary_word_data*)calloc(result->n_words > 0 ? (size_t)result->n_words : 1, sizeof(canary_word_data));
    if (!result->words) {
        canary_result_free(result);
        return nullptr;
    }
    for (int i = 0; i < result->n_words; i++)
        result->words[i] = words[(size_t)i];
    return result;
}

static bool canary_chunk_times_valid(const canary_result* part, int64_t chunk_start_cs, int64_t chunk_end_cs,
                                     int frame_dur_cs) {
    int64_t prev_t0 = chunk_start_cs - frame_dur_cs;
    int64_t prev_t1 = chunk_start_cs - frame_dur_cs;
    for (int i = 0; i < part->n_tokens; i++) {
        const auto& token = part->tokens[i];
        if (token.t0 < chunk_start_cs - frame_dur_cs || token.t1 > chunk_end_cs + frame_dur_cs || token.t1 < token.t0 ||
            token.t0 < prev_t0 || token.t1 < prev_t1)
            return false;
        prev_t0 = token.t0;
        prev_t1 = token.t1;
    }
    return true;
}

// Canary 180M Flash is a hard-cap/offline model: NVIDIA documents direct
// inference only up to 40 s and an external chunker above that. The
// canary-1b-v2 dynamic 30..40 s / 1 s-overlap blueprint is a different
// variant contract, and quantized 180M decoders can EOS before those long
// windows end. Use shorter fixed windows with enough overlap to keep each
// stitch seam away from both chunk edges. Chunks still decode independently
// (and therefore each receive the complete metadata-driven prompt).
//
// Defaults were selected by measured Q4_K_M/Q5_K_M A/B on four repeated JFK
// utterances: 20 s windows / 6 s overlap recovered all four repetitions on
// both quants. The old dynamic path recovered only 2 / 3 respectively.
// We keep the centered non-overlapping core from each decoded window:
// overlap supplies acoustic context, while no token-LCS can collapse valid
// repeated speech. This is offline chunked inference, not streaming.
static canary_result* canary_transcribe_streamed_180m(canary_context* ctx, const float* samples, int n_samples,
                                                      const char* source_lang, const char* target_lang,
                                                      bool punctuation, int64_t t_offset_cs, int chunk_seconds,
                                                      int overlap_seconds) {
    const auto& hp = ctx->model.hparams;
    const int sample_rate = (int)hp.sample_rate;
    constexpr int kDirectLimitSeconds = 40;
    constexpr int kDefaultChunkSeconds = 20;
    constexpr int kDefaultOverlapSeconds = 6;

    if (n_samples <= kDirectLimitSeconds * sample_rate)
        return canary_transcribe_ex(ctx, samples, n_samples, source_lang, target_lang, punctuation, t_offset_cs);

    if (chunk_seconds <= 0)
        chunk_seconds = kDefaultChunkSeconds;
    if (overlap_seconds < 0)
        overlap_seconds = kDefaultOverlapSeconds;
    if (overlap_seconds >= chunk_seconds)
        overlap_seconds = std::max(1, chunk_seconds / 3);

    const int chunk_samples = chunk_seconds * sample_rate;
    const int overlap_samples = overlap_seconds * sample_rate;
    const int step_samples = chunk_samples - overlap_samples;
    if (chunk_samples <= 0 || step_samples <= 0)
        return nullptr;

    const int64_t audio_end_cs = t_offset_cs + (int64_t)n_samples * 100 / sample_rate;
    const int64_t half_overlap_cs = (int64_t)overlap_samples * 50 / sample_rate;
    std::vector<canary_token_data> merged;
    bool clear_timings = false;
    bool warned_bad_times = false;
    int chunks_decoded = 0;

    for (int start = 0; start + overlap_samples < n_samples; start += step_samples) {
        const int end = std::min(start + chunk_samples, n_samples);
        const bool first_chunk = start == 0;
        const bool last_chunk = end == n_samples;
        const int64_t chunk_start_cs = t_offset_cs + (int64_t)start * 100 / sample_rate;
        const int64_t chunk_end_cs = t_offset_cs + (int64_t)end * 100 / sample_rate;
        const int64_t core_start_cs = first_chunk ? t_offset_cs : chunk_start_cs + half_overlap_cs;
        const int64_t core_end_cs =
            last_chunk ? audio_end_cs
                       : t_offset_cs + (int64_t)(start + step_samples) * 100 / sample_rate + half_overlap_cs;

        canary_result* part = canary_transcribe_ex(ctx, samples + start, end - start, source_lang, target_lang,
                                                   punctuation, chunk_start_cs);
        if (!part) {
            // Do not silently skip a failed overlap window. Retry exactly its
            // assigned core (at most 17 s with the defaults), which removes
            // overlap/merge dependence and gives the AED a shorter input.
            const int core_start = (int)((core_start_cs - t_offset_cs) * sample_rate / 100);
            const int core_end = (int)((core_end_cs - t_offset_cs) * sample_rate / 100);
            fprintf(stderr,
                    "canary: WARNING: 180M long-form chunk %.2f..%.2f s failed; "
                    "retrying its %.2f..%.2f s core\n",
                    (double)(chunk_start_cs - t_offset_cs) / 100.0, (double)(chunk_end_cs - t_offset_cs) / 100.0,
                    (double)(core_start_cs - t_offset_cs) / 100.0, (double)(core_end_cs - t_offset_cs) / 100.0);
            if (core_end > core_start) {
                part = canary_transcribe_ex(ctx, samples + core_start, core_end - core_start, source_lang, target_lang,
                                            punctuation, core_start_cs);
            }
            if (!part) {
                fprintf(stderr,
                        "canary: ERROR: 180M long-form core %.2f..%.2f s also failed; "
                        "that interval could not be decoded\n",
                        (double)(core_start_cs - t_offset_cs) / 100.0, (double)(core_end_cs - t_offset_cs) / 100.0);
                if (last_chunk)
                    break;
                continue;
            }
            for (int i = 0; i < part->n_tokens; i++)
                merged.push_back(part->tokens[i]);
            chunks_decoded++;
            canary_result_free(part);
            if (last_chunk)
                break;
            continue;
        }

        chunks_decoded++;
        const bool valid_times = canary_chunk_times_valid(part, chunk_start_cs, chunk_end_cs, (int)hp.frame_dur_cs);
        if (!valid_times) {
            clear_timings = true;
            if (!warned_bad_times) {
                warned_bad_times = true;
                fprintf(stderr, "canary: WARNING: non-monotonic 180M chunk DTW timings; "
                                "using deterministic token-position stitching and clearing merged timings\n");
            }
        }

        size_t before = merged.size();
        for (int i = 0; i < part->n_tokens; i++) {
            int64_t midpoint_cs;
            if (valid_times) {
                midpoint_cs = part->tokens[i].t0 + (part->tokens[i].t1 - part->tokens[i].t0) / 2;
            } else {
                midpoint_cs = chunk_start_cs + (int64_t)(2 * i + 1) * (chunk_end_cs - chunk_start_cs) /
                                                   (2 * std::max(1, part->n_tokens));
            }
            if (midpoint_cs >= core_start_cs && midpoint_cs < core_end_cs)
                merged.push_back(part->tokens[i]);
        }

        // Valid DTW can still be unusable for a pathological chunk (all token
        // mass outside its assigned core). Fall back to deterministic token
        // positions rather than silently dropping the whole interval.
        if (merged.size() == before && part->n_tokens > 0) {
            clear_timings = true;
            fprintf(stderr,
                    "canary: WARNING: 180M chunk %.2f..%.2f s had no tokens in its "
                    "assigned core; using token-position fallback and clearing merged timings\n",
                    (double)(chunk_start_cs - t_offset_cs) / 100.0, (double)(chunk_end_cs - t_offset_cs) / 100.0);
            for (int i = 0; i < part->n_tokens; i++) {
                const int64_t midpoint_cs = chunk_start_cs + (int64_t)(2 * i + 1) * (chunk_end_cs - chunk_start_cs) /
                                                                 (2 * std::max(1, part->n_tokens));
                if (midpoint_cs >= core_start_cs && midpoint_cs < core_end_cs)
                    merged.push_back(part->tokens[i]);
            }
        }

        canary_result_free(part);
        if (last_chunk)
            break;
    }

    if (chunks_decoded == 0)
        return nullptr;
    return canary_result_from_merged_tokens(merged, clear_timings, t_offset_cs, audio_end_cs);
}

// Long-form transcription, following canary-1b-v2's own `.transcribe()`
// dynamic chunking (see the blueprint block above). chunk_seconds <= 0 picks
// the reference's dynamic 30..40 s size; overlap_seconds < 0 uses the
// reference's 1 s. Audio that fits one chunk is a single pass — identical to
// canary_transcribe_ex.
extern "C" struct canary_result* canary_transcribe_streamed(struct canary_context* ctx, const float* samples,
                                                            int n_samples, const char* source_lang,
                                                            const char* target_lang, bool punctuation,
                                                            int64_t t_offset_cs, int chunk_seconds,
                                                            int overlap_seconds) {
    if (!ctx || !samples || n_samples <= 0 || !source_lang || !target_lang)
        return nullptr;
    if (ctx->model.hparams.new_schema && ctx->model.hparams.variant == "canary-180m-flash") {
        return canary_transcribe_streamed_180m(ctx, samples, n_samples, source_lang, target_lang, punctuation,
                                               t_offset_cs, chunk_seconds, overlap_seconds);
    }
    {
        static const bool legacy = [] {
            const char* e = crispasr_env::get("CRISPASR_CANARY_LEGACY_STREAM");
            return e && e[0] == '1';
        }();
        if (legacy)
            return canary_transcribe_streamed_legacy(ctx, samples, n_samples, source_lang, target_lang, punctuation,
                                                     t_offset_cs, chunk_seconds > 0 ? chunk_seconds : 8,
                                                     overlap_seconds >= 0 ? overlap_seconds : 2);
    }

    const auto& hp = ctx->model.hparams;
    const int SR = (int)hp.sample_rate;
    const double overlap_sec = overlap_seconds >= 0 ? (double)overlap_seconds : 1.0;
    const int kMinSec = 30, kMaxSec = 40; // _find_optimal_chunk_size defaults

    int chunk_samples;
    if (chunk_seconds > 0) {
        chunk_samples = chunk_seconds * SR;
    } else {
        chunk_samples = core_canary_chunk::optimal_chunk_samples(n_samples, SR, kMinSec, kMaxSec, overlap_sec);
    }
    if (chunk_samples >= n_samples)
        return canary_transcribe_ex(ctx, samples, n_samples, source_lang, target_lang, punctuation, t_offset_cs);

    const int overlap_samples = (int)(overlap_sec * SR);
    const int step = chunk_samples - overlap_samples;
    if (step <= 0)
        return canary_transcribe_ex(ctx, samples, n_samples, source_lang, target_lang, punctuation, t_offset_cs);

    // merge_parallel_chunks: the overlap is 1 s; delay = that second's worth
    // of encoder frames, and only ~60% of them carry non-blank tokens.
    const int sub = hp.subsampling_factor > 0 ? (int)hp.subsampling_factor : 8;
    const int delay = (int)(1.0 / ((double)sub / 100.0)); // 12 for 8x subsampling
    const int head_len_cfg = (int)(delay * 0.6);          // 7
    const int max_steps_per_timestep = 2;                 // search window = delay * 2

    std::vector<canary_token_data> merged;
    int chunks_processed = 0;
    for (int start = 0; start + overlap_samples < n_samples; start += step) {
        const int end = std::min(start + chunk_samples, n_samples);
        const int64_t chunk_t_offset_cs = t_offset_cs + (int64_t)start * 100 / SR;
        canary_result* part = canary_transcribe_ex(ctx, samples + start, end - start, source_lang, target_lang,
                                                   punctuation, chunk_t_offset_cs);
        if (!part)
            continue;

        if (chunks_processed == 0 || merged.empty() || delay < 1) {
            for (int i = 0; i < part->n_tokens; i++)
                merged.push_back(part->tokens[i]);
        } else {
            // lcs_alignment_merge_buffer(buffer, data=head, delay,
            // max_steps_per_timestep=2, min_lcs_length=1, parallel_chunking).
            const int head_len = std::min(head_len_cfg, part->n_tokens);
            const int search = std::min(delay * max_steps_per_timestep, (int)merged.size());
            std::vector<int> X, Y;
            X.reserve((size_t)search);
            Y.reserve((size_t)head_len);
            for (int i = (int)merged.size() - search; i < (int)merged.size(); i++)
                X.push_back(merged[(size_t)i].id);
            for (int i = 0; i < head_len; i++)
                Y.push_back(part->tokens[i].id);
            int li = 0, lj = 0, llen = 0;
            core_canary_chunk::nemo_lcs_merge(X, Y, li, lj, llen);
            int append_from = 0;
            if (llen >= 1) { // min_lcs_length
                // merged = buffer[:i_abs_end] + data[j_after:] — keep the
                // matched acoustic event once, trimming BOTH sides' overlap.
                const int base = (int)merged.size() - search;
                const int i_abs_end = base + li + llen;
                if (i_abs_end >= 0 && i_abs_end <= (int)merged.size())
                    merged.resize((size_t)i_abs_end);
                append_from = std::min(lj + llen, head_len);
            }
            for (int i = append_from; i < part->n_tokens; i++)
                merged.push_back(part->tokens[i]);
        }
        canary_result_free(part);
        chunks_processed++;
    }
    if (chunks_processed == 0)
        return nullptr;

    std::string full_text;
    for (const auto& t : merged)
        full_text += t.text;
    if (!full_text.empty() && full_text[0] == ' ')
        full_text.erase(0, 1);
    auto words = canary_words_from_tokens(merged.data(), (int)merged.size());

    canary_result* r = (canary_result*)calloc(1, sizeof(canary_result));
    if (!r)
        return nullptr;
    r->text = strdup(full_text.c_str());
    if (!r->text) {
        canary_result_free(r);
        return nullptr;
    }
    r->n_tokens = (int)merged.size();
    r->tokens = (canary_token_data*)calloc(r->n_tokens > 0 ? (size_t)r->n_tokens : 1, sizeof(canary_token_data));
    if (!r->tokens) {
        canary_result_free(r);
        return nullptr;
    }
    for (int i = 0; i < r->n_tokens; i++)
        r->tokens[i] = merged[(size_t)i];
    r->n_words = (int)words.size();
    r->words = (canary_word_data*)calloc(r->n_words > 0 ? (size_t)r->n_words : 1, sizeof(canary_word_data));
    if (!r->words) {
        canary_result_free(r);
        return nullptr;
    }
    for (int i = 0; i < r->n_words; i++)
        r->words[i] = words[(size_t)i];
    return r;
}

static canary_result* canary_finish_from_encoder(canary_context* ctx, const float* enc_data, int T_enc,
                                                 const char* source_lang, const char* target_lang, bool punctuation,
                                                 int64_t t_offset_cs) {
    const auto& hp = ctx->model.hparams;
    if (hp.new_schema) {
        auto supports_language = [&](const char* language) {
            return std::find(hp.languages.begin(), hp.languages.end(), language) != hp.languages.end();
        };
        if (!supports_language(source_lang)) {
            fprintf(stderr, "canary: source language '%s' is not advertised by variant '%s'\n", source_lang,
                    hp.variant.c_str());
            return nullptr;
        }
        if (!supports_language(target_lang)) {
            fprintf(stderr, "canary: target language '%s' is not advertised by variant '%s'\n", target_lang,
                    hp.variant.c_str());
            return nullptr;
        }
        if (strcmp(source_lang, target_lang) != 0) {
            const std::string pair = std::string(source_lang) + ">" + target_lang;
            if (std::find(hp.translation_pairs.begin(), hp.translation_pairs.end(), pair) ==
                hp.translation_pairs.end()) {
                fprintf(stderr, "canary: translation pair '%s' is not advertised by variant '%s'\n", pair.c_str(),
                        hp.variant.c_str());
                return nullptr;
            }
        }
    }

    // 3. Build the per-chunk prompt before any decoder-side work.
    std::vector<int> prompt = canary_build_prompt(ctx, source_lang, target_lang, punctuation);
    if (prompt.empty())
        return nullptr;

    // 4. Project the native encoder width to the decoder width when the model
    // carries a trained bridge (Canary 180M Flash: 512 -> 1024).
    const float* decoder_enc_data = enc_data;
    std::vector<float> projected;
    if (hp.has_encoder_decoder_proj) {
        projected = canary_project_encoder(ctx, enc_data, T_enc);
        if (projected.empty())
            return nullptr;
        decoder_enc_data = projected.data();
    }

    // 5. Pre-compute decoder-width cross-attention K/V.
    canary_build_cross_kv(ctx, decoder_enc_data, T_enc);

    // Reset DTW state and enable cross-attn capture for the upcoming greedy loop.
    // Each per-step decode call will append one entry to ctx->step_attn (one
    // T_enc * n_heads vector per emitted token).
    ctx->collect_attn = true;
    ctx->step_attn.clear();

    // 6. Greedy decode
    const int eos = hp.new_schema ? hp.endoftext_id : canary_str_to_token(ctx, "<|endoftext|>");
    const int max_ctx = (int)ctx->model.hparams.max_dec_ctx;
    // #292: honor --max-new-tokens, clamped to the model's decoder context so a
    // large value can't run past the trained window / KV allocation.
    int max_steps = ctx->max_new_tokens > 0 ? ctx->max_new_tokens : 256;
    if (max_ctx > 0 && max_steps > max_ctx)
        max_steps = max_ctx;

    std::vector<int> generated = prompt;
    std::vector<float> emitted_p; // softmax prob per generated non-prompt token

    const bool sampling = ctx->decode_temperature > 0.0f;
    std::mt19937_64 rng(ctx->decode_seed != 0 ? ctx->decode_seed : (uint64_t)std::random_device{}());
    int offset = 0;

    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "canary: prompt =");
        for (int t : prompt)
            fprintf(stderr, " %d(%s)", t, ctx->vocab.id_to_token[t].c_str());
        fprintf(stderr, "\n");
    }


    // First call: feed the entire prompt at once
    auto logits = canary_decode_step(ctx, prompt.data(), (int)prompt.size(), 0);
    if (logits.empty())
        return nullptr;
    offset = (int)prompt.size();

    // §90 beam search — run_with_probs_branched when beam_size > 1.
    // Cross-attention KV (cross_k/v) is shared across beams; only self-attention
    // KV (kv_k/kv_v) is snapshotted per beam.
    if (ctx->beam_size > 1) {
        // GH #161: snapshot/restore self-attention KV on-device via a recycled
        // buffer pool (no PCIe round-trip + sync per beam per step).
        core_attn::kv_snapshot_pool kv_pool(ctx->kv_k, ctx->kv_v);

        auto save_fn = [&kv_pool](canary_context*) -> core_attn::kv_snapshot* { return kv_pool.save(); };

        auto restore_fn = [&kv_pool](canary_context*, core_attn::kv_snapshot* s) { kv_pool.restore(s); };

        auto snap_free_fn = [&kv_pool](core_attn::kv_snapshot* s) { kv_pool.release(s); };

        auto step_fn = [](canary_context* c, int32_t tok, int n_past) -> float* {
            auto lg = canary_decode_step(c, &tok, 1, n_past);
            if (lg.empty())
                return nullptr;
            float* out = (float*)std::malloc(lg.size() * sizeof(float));
            std::memcpy(out, lg.data(), lg.size() * sizeof(float));
            return out;
        };

        const int vocab = (int)logits.size();
        core_beam_decode::Config bcfg;
        bcfg.max_new_tokens = max_steps;
        bcfg.eos_id = eos;
        bcfg.vocab_size = vocab;
        bcfg.beam_size = ctx->beam_size;
        bcfg.prompt_len = (int)prompt.size();

        auto br = core_beam_decode::run_with_probs_branched(ctx, logits.data(), save_fn, restore_fn, snap_free_fn,
                                                            step_fn, bcfg);

        // Build result from beam output (skip EOS if present at the end)
        int n_beam = (int)br.tokens.size();
        if (n_beam > 0 && br.tokens.back() == eos)
            n_beam--;

        auto* r = (canary_result*)calloc(1, sizeof(canary_result));
        if (!r)
            return nullptr;

        r->n_tokens = n_beam;
        r->tokens = (canary_token_data*)calloc((size_t)n_beam, sizeof(canary_token_data));
        std::string full_text;
        for (int i = 0; i < n_beam; i++) {
            r->tokens[i].id = br.tokens[i];
            r->tokens[i].p = br.probs[i];
            r->tokens[i].t0 = 0;
            r->tokens[i].t1 = 0;
            const char* txt = canary_token_to_str(ctx, br.tokens[i]);
            if (txt) {
                std::string vis = spiece_to_text(txt);
                snprintf(r->tokens[i].text, sizeof(r->tokens[i].text), "%s", vis.c_str());
                full_text += vis;
            }
        }
        r->text = strdup(full_text.c_str());
        r->n_words = 0;
        r->words = nullptr;
        return r;
    }

    // PLAN #114 P3 polish — degenerate-loop guard. The AED has no
    // repetition penalty; on out-of-distribution audio (or a chunk
    // boundary where the encoder embeddings collapse) the decoder can
    // lock on a short cycle and emit ~100+ copies before <eos>. Funasr
    // (PLAN #125 P1) had a single-id repeat (`!`); canary's BPE often
    // emits a two-token cycle like `▁yeah` + `,` alternating, so the
    // funasr-style consecutive-id guard misses it. Detect by
    // small-vocabulary window: if the last kWindow generated tokens
    // contain ≤ kMaxDistinct distinct ids, the decoder is in a loop.
    // Window/threshold picked so normal speech (~25-30 unique ids per
    // 40-token window) stays well above the trigger and degenerate
    // 1/2/3-cycles all fire.
    constexpr int kWindow = 40;
    constexpr int kMaxDistinct = 3;

    for (int step = 0; step < max_steps && offset < max_ctx - 1; step++) {
        // Argmax (default) or temperature sample
        int best = 0;
        float best_lp = logits[0];
        for (int v = 1; v < (int)logits.size(); v++) {
            if (logits[v] > best_lp) {
                best_lp = logits[v];
                best = v;
            }
        }
        if (sampling) {
            const int V = (int)logits.size();
            const float inv_t = 1.0f / ctx->decode_temperature;
            std::vector<double> pr((size_t)V);
            double sum = 0.0;
            for (int v = 0; v < V; v++) {
                const double e = std::exp((double)((logits[v] - best_lp) * inv_t));
                pr[(size_t)v] = e;
                sum += e;
            }
            if (sum > 0.0) {
                std::uniform_real_distribution<double> unif(0.0, sum);
                const double rr = unif(rng);
                double acc = 0.0;
                for (int v = 0; v < V; v++) {
                    acc += pr[(size_t)v];
                    if (rr <= acc) {
                        best = v;
                        break;
                    }
                }
                best_lp = logits[best];
            }
        }
        if (ctx->params.verbosity >= 2 && step < 30) {
            const char* pc =
                (best >= 0 && best < (int)ctx->vocab.id_to_token.size()) ? ctx->vocab.id_to_token[best].c_str() : "?";
            fprintf(stderr, "  step %3d  tok=%5d  '%s'  logp=%.3f\n", step, best, pc, (double)best_lp);
        }
        if (best == eos)
            break;

        // Degenerate-loop guard: count distinct ids in the last
        // kWindow generated tokens (including this one). Cheap O(W)
        // per step on a fixed-size window — for kWindow=40 that's
        // ~40 comparisons per step, negligible next to the decoder
        // forward.
        if ((int)generated.size() >= (int)prompt.size() + kWindow) {
            const int start = (int)generated.size() - kWindow + 1;
            std::set<int> distinct;
            for (int i = start; i < (int)generated.size(); i++)
                distinct.insert(generated[(size_t)i]);
            distinct.insert(best);
            if ((int)distinct.size() <= kMaxDistinct) {
                if (ctx->params.verbosity >= 1) {
                    fprintf(stderr,
                            "canary: greedy decode degenerated (%d distinct ids in last %d tokens). "
                            "Aborting at step %d.\n",
                            (int)distinct.size(), kWindow, step);
                }
                break;
            }
        }

        // Softmax probability of the picked token. Numerically stable:
        // subtract the max log-prob before exponentiating.
        float best_p = 1.0f;
        {
            double sum = 0.0;
            for (int v = 0; v < (int)logits.size(); v++) {
                sum += std::exp((double)(logits[v] - best_lp));
            }
            best_p = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
        }

        generated.push_back(best);
        emitted_p.push_back(best_p);

        // Decode next step with the new token
        int tok = best;
        logits = canary_decode_step(ctx, &tok, 1, offset);
        if (logits.empty())
            break;
        offset++;
    }

    // 6. Build result (skip the prompt prefix)
    int n_emitted = (int)generated.size() - (int)prompt.size();
    auto* r = (canary_result*)calloc(1, sizeof(canary_result));
    if (!r)
        return nullptr;
    r->n_tokens = n_emitted;
    r->tokens = (canary_token_data*)calloc(n_emitted > 0 ? n_emitted : 1, sizeof(canary_token_data));
    if (!r->tokens) {
        canary_result_free(r);
        return nullptr;
    }

    // ----- Per-token timestamps via cross-attention DTW -----
    //
    // For each emitted token we have ctx->step_attn[i] = cross-attn weights from
    // the LAST decoder layer, layout [T_enc * n_heads]. We:
    //   1. Average across heads → A[n_emitted × T_enc]
    //   2. Subtract per-frame mean (column normalisation) so globally "hot"
    //      frames don't dominate
    //   3. Run prefix-max DTW: D[i][j] = A[i][j] + max_{k≤j} D[i-1][k]
    //   4. Traceback from argmax of last row → path[i] = best frame for token i
    //   5. t0/t1 from path[i] × frame_dur_cs
    //
    // Same approach as cohere-main, gives ~360 ms MAE on word boundaries.
    // (Approach copied from src/cohere.cpp; see that file for the discussion of
    // hot-frame collapse and the column-normalisation fix.)
    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;
    const int64_t total_cs = (int64_t)T_enc * frame_dur_cs;
    const int64_t seg_end_cs = t_offset_cs + total_cs;

    const bool have_attn =
        n_emitted > 0 && ctx->step_attn.size() == (size_t)n_emitted && ctx->attn_T_enc > 0 && ctx->attn_n_heads > 0;

    std::vector<int> path;
    if (have_attn) {
        const int T_e = ctx->attn_T_enc;
        const int n_h = ctx->attn_n_heads;

        // Step 1 + temporal offset correction (same trick as cohere): use
        // step_attn[i-1] for token i because the attention is collected while
        // the decoder processes generated[i] as input to predict generated[i+1].
        std::vector<float> A((size_t)n_emitted * T_e, 0.0f);
        for (int i = 0; i < n_emitted; i++) {
            int src = (i > 0) ? i - 1 : 0;
            const float* w = ctx->step_attn[src].data();
            float* Ai = A.data() + (size_t)i * T_e;
            for (int h = 0; h < n_h; h++)
                for (int t = 0; t < T_e; t++)
                    Ai[t] += w[h * T_e + t];
            for (int t = 0; t < T_e; t++)
                Ai[t] /= n_h;
        }

        // Step 2: column normalisation
        {
            std::vector<float> col_mean(T_e, 0.f);
            for (int i = 0; i < n_emitted; i++)
                for (int t = 0; t < T_e; t++)
                    col_mean[t] += A[(size_t)i * T_e + t];
            for (int t = 0; t < T_e; t++)
                col_mean[t] /= n_emitted;
            for (int i = 0; i < n_emitted; i++)
                for (int t = 0; t < T_e; t++)
                    A[(size_t)i * T_e + t] -= col_mean[t];
        }

        // Step 3: forward DP with prefix-max predecessors
        std::vector<float> D((size_t)n_emitted * T_e);
        std::vector<int> P((size_t)n_emitted * T_e);
        for (int j = 0; j < T_e; j++)
            D[j] = A[j];
        for (int i = 1; i < n_emitted; i++) {
            const float* Di_1 = D.data() + (size_t)(i - 1) * T_e;
            const float* Ai = A.data() + (size_t)i * T_e;
            float* Di = D.data() + (size_t)i * T_e;
            int* Pi = P.data() + (size_t)i * T_e;
            float pm_val = Di_1[0];
            int pm_idx = 0;
            for (int j = 0; j < T_e; j++) {
                if (Di_1[j] > pm_val) {
                    pm_val = Di_1[j];
                    pm_idx = j;
                }
                Di[j] = Ai[j] + pm_val;
                Pi[j] = pm_idx;
            }
        }

        // Step 4: traceback
        path.resize(n_emitted);
        const float* Dlast = D.data() + (size_t)(n_emitted - 1) * T_e;
        path[n_emitted - 1] = (int)(std::max_element(Dlast, Dlast + T_e) - Dlast);
        for (int i = n_emitted - 2; i >= 0; i--)
            path[i] = P[(size_t)(i + 1) * T_e + path[i + 1]];
    }

    std::string text;
    for (int i = 0; i < n_emitted; i++) {
        int tid = generated[(int)prompt.size() + i];
        const std::string& piece =
            (tid >= 0 && tid < (int)ctx->vocab.id_to_token.size()) ? ctx->vocab.id_to_token[tid] : std::string("");
        std::string vis = spiece_to_text(piece);

        canary_token_data& td = r->tokens[i];
        td.id = tid;
        td.p = (i < (int)emitted_p.size()) ? emitted_p[i] : -1.0f;
        if (have_attn) {
            int64_t a = t_offset_cs + (int64_t)path[i] * frame_dur_cs;
            int64_t b = (i + 1 < n_emitted) ? (t_offset_cs + (int64_t)path[i + 1] * frame_dur_cs) : seg_end_cs;
            // Guarantee at least one frame of duration so adjacent tokens that
            // collapse to the same DTW frame still have a non-zero span.
            if (b <= a)
                b = a + frame_dur_cs;
            if (b > seg_end_cs)
                b = seg_end_cs;
            td.t0 = a;
            td.t1 = b;
        } else {
            // Fallback: linear interpolation
            td.t0 = t_offset_cs + (int64_t)((double)i / std::max(1, n_emitted) * total_cs);
            td.t1 = t_offset_cs + (int64_t)((double)(i + 1) / std::max(1, n_emitted) * total_cs);
        }
        size_t n = std::min(vis.size(), sizeof(td.text) - 1);
        memcpy(td.text, vis.data(), n);
        td.text[n] = '\0';
        text += vis;
    }
    if (!text.empty() && text[0] == ' ')
        text = text.substr(1);
    r->text = strdup(text.c_str());
    if (!r->text) {
        canary_result_free(r);
        return nullptr;
    }

    // Disable attn collection so the next call starts fresh
    ctx->collect_attn = false;

    // Word grouping (same as parakeet's)
    {
        std::vector<canary_word_data> words = canary_words_from_tokens(r->tokens, r->n_tokens);
        r->n_words = (int)words.size();
        r->words = (canary_word_data*)calloc(r->n_words > 0 ? r->n_words : 1, sizeof(canary_word_data));
        if (!r->words) {
            canary_result_free(r);
            return nullptr;
        }
        for (int i = 0; i < r->n_words; i++)
            r->words[i] = words[i];
    }

    return r;
}

extern "C" char* canary_transcribe(struct canary_context* ctx, const float* samples, int n_samples,
                                   const char* source_lang, const char* target_lang, bool punctuation) {
    canary_result* r = canary_transcribe_ex(ctx, samples, n_samples, source_lang, target_lang, punctuation, 0);
    if (!r)
        return nullptr;
    char* out = strdup(r->text ? r->text : "");
    if (!out) {
        canary_result_free(r);
        return nullptr;
    }
    canary_result_free(r);
    return out;
}
