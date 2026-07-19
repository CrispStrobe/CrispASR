// crispasr-quantize — GGUF tensor re-quantization tool.
//
// Takes any GGUF model (whisper, parakeet, canary, cohere, voxtral, qwen3,
// granite, wav2vec2, …) and re-quantizes all eligible tensors to the
// target ggml_ftype, preserving metadata and non-quantizable tensors
// (norms, positional embeddings, biases, small tables) in their
// original types. The logic is model-agnostic — it just iterates the
// GGUF tensor list and calls ggml_quantize_chunk on each float tensor.
//
// Historically lived in examples/cohere-main/cohere-quantize.cpp; moved
// here when the per-model CLIs were consolidated into crispasr.

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "common.h"
#include "common-ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <cmath>

// Per-tensor importance vectors loaded from an importance-matrix (imatrix)
// GGUF file. Keyed by weight name; value length == n_cols (ne[0]).
// importance[c] = sum_of_squares[c] / count, i.e. the mean activation energy
// seen in column c over a calibration run. Passed straight to
// ggml_quantize_chunk, which uses it to minimise *activation-weighted* error
// for k-quants / IQ-quants instead of plain L2 — the same mechanism as
// llama.cpp's `llama-imatrix`. Ported from CrispEmbed tools/quantize.cpp.
//
// File format (a GGUF produced by a calibration run): one F32 tensor per
// weight, name == the weight's tensor name, holding the per-column
// sum-of-squares, plus a `count.<name>` u64 metadata key holding the sample
// count. importance[c] = sum_of_squares[c] / count. When empty /
// shape-mismatched we fall back to unweighted quantization, so `--imatrix` is
// always safe to pass. (An importance-matrix *producer* — a calibration pass
// that emits this file — is not yet in CrispASR; see PLAN.md §llama.cpp
// comparison Tier-1. This is the consumer side, ported from CrispEmbed.)
static std::map<std::string, std::vector<float>> g_imatrix;

static bool crispasr_load_imatrix(const std::string& path) {
    struct ggml_context* ctx = nullptr;
    struct gguf_init_params p = {/*no_alloc*/ false, /*ctx*/ &ctx};
    struct gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        fprintf(stderr, "imatrix: failed to open '%s'\n", path.c_str());
        return false;
    }
    const int64_t nt = gguf_get_n_tensors(g);
    int loaded = 0;
    for (int64_t i = 0; i < nt; i++) {
        const char* name = gguf_get_tensor_name(g, i);
        struct ggml_tensor* t = ggml_get_tensor(ctx, name);
        if (!t || t->type != GGML_TYPE_F32)
            continue;
        const int64_t ne0 = t->ne[0];
        const float* d = (const float*)t->data;
        std::string ck = std::string("count.") + name;
        int64_t kid = gguf_find_key(g, ck.c_str());
        uint64_t count = (kid >= 0) ? gguf_get_val_u64(g, kid) : 0;
        if (count == 0)
            continue;
        std::vector<float> imp((size_t)ne0);
        const double inv = 1.0 / (double)count;
        for (int64_t c = 0; c < ne0; c++)
            imp[c] = (float)((double)d[c] * inv);
        g_imatrix[name] = std::move(imp);
        loaded++;
    }
    gguf_free(g);
    ggml_free(ctx);
    fprintf(stderr, "imatrix: loaded importance vectors for %d tensors from '%s'\n", loaded, path.c_str());
    return loaded > 0;
}

// Per-tensor type overrides from --tensor-type <regex>=<type> (repeatable).
// First matching rule wins; applied AFTER the arch guards, so it can force a
// guarded tensor to quantize, pin a body tensor higher, or keep something at
// F16 — the llama.cpp `--tensor-type` mechanism. Value may be f16/f32 or any
// quant type below.
static std::vector<std::pair<std::regex, ggml_type>> g_type_overrides;
static std::vector<std::string> g_type_override_src; // original "regex=type" text, for logs

static ggml_type crispasr_parse_type_name(const std::string& s) {
    static const std::map<std::string, ggml_type> M = {
        {"f32", GGML_TYPE_F32},       {"f16", GGML_TYPE_F16},       {"q4_0", GGML_TYPE_Q4_0}, {"q4_1", GGML_TYPE_Q4_1},
        {"q5_0", GGML_TYPE_Q5_0},     {"q5_1", GGML_TYPE_Q5_1},     {"q8_0", GGML_TYPE_Q8_0}, {"q2_k", GGML_TYPE_Q2_K},
        {"q3_k", GGML_TYPE_Q3_K},     {"q4_k", GGML_TYPE_Q4_K},     {"q5_k", GGML_TYPE_Q5_K}, {"q6_k", GGML_TYPE_Q6_K},
        {"iq4_nl", GGML_TYPE_IQ4_NL}, {"iq4_xs", GGML_TYPE_IQ4_XS},
    };
    auto it = M.find(s);
    return it == M.end() ? GGML_TYPE_COUNT : it->second;
}

// Return `want` if it tiles `ncols`, else a compatible smaller-block fallback,
// else GGML_TYPE_COUNT (can't quantize this row width).
static ggml_type crispasr_row_fit(ggml_type want, int64_t ncols) {
    if (ncols % ggml_blck_size(want) == 0)
        return want;
    ggml_type fb = GGML_TYPE_COUNT;
    switch (want) {
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
        fb = GGML_TYPE_Q4_0;
        break;
    case GGML_TYPE_Q5_K:
        fb = GGML_TYPE_Q5_0;
        break;
    case GGML_TYPE_Q6_K:
        fb = GGML_TYPE_Q8_0;
        break;
    case GGML_TYPE_IQ4_XS:
        fb = GGML_TYPE_IQ4_NL;
        break;
    case GGML_TYPE_IQ4_NL:
        fb = GGML_TYPE_Q4_0;
        break;
    default:
        break;
    }
    return (fb != GGML_TYPE_COUNT && ncols % ggml_blck_size(fb) == 0) ? fb : GGML_TYPE_COUNT;
}

static bool crispasr_model_quantize(const std::string& fname_inp, const std::string& fname_out, ggml_ftype ftype) {
    ggml_type qtype = GGML_TYPE_F32;

    switch (ftype) {
    case GGML_FTYPE_MOSTLY_Q4_0:
        qtype = GGML_TYPE_Q4_0;
        break;
    case GGML_FTYPE_MOSTLY_Q4_1:
        qtype = GGML_TYPE_Q4_1;
        break;
    case GGML_FTYPE_MOSTLY_Q5_0:
        qtype = GGML_TYPE_Q5_0;
        break;
    case GGML_FTYPE_MOSTLY_Q5_1:
        qtype = GGML_TYPE_Q5_1;
        break;
    case GGML_FTYPE_MOSTLY_Q8_0:
        qtype = GGML_TYPE_Q8_0;
        break;
    case GGML_FTYPE_MOSTLY_Q2_K:
        qtype = GGML_TYPE_Q2_K;
        break;
    case GGML_FTYPE_MOSTLY_Q3_K:
        qtype = GGML_TYPE_Q3_K;
        break;
    case GGML_FTYPE_MOSTLY_Q4_K:
        qtype = GGML_TYPE_Q4_K;
        break;
    case GGML_FTYPE_MOSTLY_Q5_K:
        qtype = GGML_TYPE_Q5_K;
        break;
    case GGML_FTYPE_MOSTLY_Q6_K:
        qtype = GGML_TYPE_Q6_K;
        break;
    // IQ4_NL / IQ4_XS: 4-bit non-linear codebook quants. A/B on CrispEmbed
    // beat Q4_K on both quality and size for the same bit budget (they use a
    // shared non-linear value map + super-block scale). IQ4_XS uses 256-wide
    // super-blocks; IQ4_NL is the 32-wide legacy-block variant used as the
    // row-width fallback below. Both benefit most with an --imatrix.
    case GGML_FTYPE_MOSTLY_IQ4_NL:
        qtype = GGML_TYPE_IQ4_NL;
        break;
    case GGML_FTYPE_MOSTLY_IQ4_XS:
        qtype = GGML_TYPE_IQ4_XS;
        break;
    default:
        fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, ftype);
        return false;
    }

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    struct ggml_context* ctx_in_ggml = nullptr;
    struct gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = &ctx_in_ggml;
    struct gguf_context* ctx_in = gguf_init_from_file(fname_inp.c_str(), params);
    if (!ctx_in || !ctx_in_ggml) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, fname_inp.c_str());
        return false;
    }

    struct gguf_context* ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", ftype);

    // Detect architecture for arch-specific quantization rules
    std::string arch;
    {
        int key = gguf_find_key(ctx_in, "general.architecture");
        if (key >= 0 && gguf_get_kv_type(ctx_in, key) == GGUF_TYPE_STRING)
            arch = gguf_get_val_str(ctx_in, key);
    }
    const bool is_firered = (arch.find("firered") != std::string::npos);
    const bool is_ecapa = (arch.find("ecapa") != std::string::npos);
    // DAC-44kHz: pure convolutional audio codec (Descript Audio Codec).
    // ALL weight tensors store the kernel_size as ne[0] (ggml conv layout:
    // [K, IC, OC] where K ≤ 16) and the codebook embeddings have ne[0]=8.
    // Neither satisfies the minimum block-size requirement (Q8_0: 32,
    // Q4_K: 256). This model cannot be compressed via block quantization.
    const bool is_dac = (arch.find("dac") != std::string::npos);
    if (is_dac) {
        fprintf(stderr,
                "%s: WARNING — architecture '%s' is a convolutional audio codec.\n"
                "  All weight tensors use kernel-size as ne[0] (≤16 elements), which\n"
                "  is below the minimum block size for any GGUF quant type (Q8_0: 32,\n"
                "  Q4_K: 256). Zero tensors will be quantized; the output file will be\n"
                "  the same size as the input. This model cannot be meaningfully\n"
                "  compressed via GGUF block quantization.\n",
                __func__, arch.c_str());
    }
    // CREPE (pitch/F0, 6× conv1d + one dense classifier). Only `conv*.weight`
    // and `classifier.weight` are quantizable. Everything else is a per-channel
    // affine parameter — `conv*.bias`, `conv*_BN.scale`, `conv*_BN.offset`,
    // `classifier.bias` — and MUST stay F32: the relu sits BEFORE the BN, so
    // the BN ships as a standalone affine that is applied once per channel per
    // frame. Rounding those is a direct multiplicative error on the whole
    // channel, which is exactly what corrupts BN-style folds.
    //
    // The conv weights are 3D in ggml conv layout (K, IC, OC), so the 2D-only
    // `ok_dims` rule would otherwise skip them entirely.
    //
    // Block-size geometry — the thing to get right here. `src/crepe.cpp`
    // consumes each kernel as `ggml_reshape_2d(w, ne[0]*ne[1], ne[2])`, i.e.
    // (K*IC, OC), so it is tempting to quantize at row = K*IC. Don't: the GGUF
    // loader validates the DECLARED ne[0] against the block size
    // (ggml/src/gguf.cpp:671), so a file quantized at K*IC but still declaring
    // ne=(K, IC, OC) is rejected at load. We therefore quantize at ne[0] = K.
    // That is layout-safe — blocks are contiguous and K is a multiple of 32, so
    // the runtime reshape to (K*IC, OC) reinterprets exactly the same block
    // stream — and it makes the row-fit fallback below the load-bearing rule:
    //
    //   conv1       ne[0] = 512  -> Q8_0 ok, Q4_K ok (2 super-blocks)
    //   conv2..6    ne[0] =  64  -> Q8_0 ok, Q4_K NO -> falls back to Q4_0
    //   classifier  ne[0] = 256 (tiny) / 2048 (full), 2D -> both ok
    //
    // So conv1 is NOT the problem tensor despite IC=1: K=512 divides both block
    // sizes cleanly. The 64-tap conv2..6 kernels are what Q4_K cannot tile, and
    // a "q4_k" CREPE is really Q4_K on conv1+classifier and Q4_0 on conv2..6.
    // Getting true Q4_K on conv2..6 would mean writing them reshaped to 2D
    // (K*IC, OC) the way `pw_conv3d` does below, which also needs src/crepe.cpp
    // to stop deriving the im2col geometry from the kernel's own shape.
    const bool is_crepe = (arch == "crepe");

    const bool is_chatterbox =
        (arch.find("chatterbox") != std::string::npos || arch.find("kartoffelbox") != std::string::npos);
    // CosyVoice3: the three sub-models live in separate GGUFs but share the
    // `cosyvoice3-` arch prefix (llm / flow / hift). For the LLM sub-model
    // we skip the speech-token embedding + LM-head tensors — they're small
    // (6761 × 896) and quantising them adds noise to the AR sampling
    // logits (same reasoning as llama.cpp's Q4_K_M keeping `output.weight`
    // off the Q4_K path). For the flow sub-model the `input_embd.w` and
    // the `spk_affine` projection stay at full precision too. HiFT is too
    // small to bother quantising (42 MB F16) — the tool will still run on
    // it but the gains are negligible.
    const bool is_cosyvoice3 = (arch.find("cosyvoice3") != std::string::npos);
    // F5-TTS: DiT flow-matching with 32-step Euler ODE. The conditioning
    // pathway (AdaLN modulation, timestep MLP, input/output projections,
    // conv-pos embeddings) must stay at F16 — quantization noise compounds
    // through 22 layers × 32 steps × 2 (CFG) = 1408 forward passes.
    // DiT bulk weight matrices (QKV, O-proj, FFN) can be quantized. Text
    // encoder, Vocos, and the AdaLN/timestep/input/final projections are
    // kept at original precision. Previously this backend was skipped
    // entirely because read_tensor_f32 couldn't dequantize — that's fixed.
    const bool is_f5tts = (arch.find("f5-tts") != std::string::npos || arch.find("f5tts") != std::string::npos);
    // The granite-speech 4.1 family ("granite_speech" base + plus, "granite_nle"
    // for the non-autoregressive variant) all share the same 16-layer Conformer
    // encoder + Q-Former projector + Granite-1B LLM, so the same quantization
    // rules apply: skip enc.* / proj.* unless explicitly overridden.
    const bool is_granite_family =
        (arch.find("granite_speech") != std::string::npos) || (arch.find("granite_nle") != std::string::npos);
    // Optional: downcast granite-family encoder F32 weights to F16 instead of
    // preserving F32. Halves the encoder footprint (~960 MB on 4.1-2b) at
    // negligible quality cost — F16 is what every Whisper / Llama / parakeet
    // GGUF in the wild uses for encoder weights. Off by default to keep the
    // canonical Q4K bit-identical to F16 reference; opt in with the env var.
    const char* env_enc_f16 = std::getenv("CRISPASR_GRANITE_ENC_F16");
    const bool granite_enc_to_f16 = is_granite_family && env_enc_f16 && *env_enc_f16 && *env_enc_f16 != '0';
    // Optional: quantize EVERYTHING for the granite family — including the
    // 16-layer Conformer encoder and the Q-Former projector that we
    // normally pin at F32/F16. Produces the published `-mini` variant
    // (~1.7 GB on 4.1-2b) at the cost of ~0.93 cosine parity instead
    // of ~0.999. Off by default; opt in with the env var.
    const char* env_quant_all = std::getenv("CRISPASR_GRANITE_QUANT_ALL");
    const bool granite_quant_all = is_granite_family && env_quant_all && *env_quant_all && *env_quant_all != '0';

    // OmniASR-CTC: 48-layer wav2vec2-style encoder + CTC head. Per-layer
    // activation cosine analysis on JFK (Q4_K vs Q8_0 dumps via
    // OMNIASR_DUMP_DIR) shows drift accumulates: layers 0–35 stay at
    // cos ≥ 0.995, layers 36–47 drop to ≈0.98. CTC argmax is structurally
    // sensitive to compounded drift (no internal LM smoothing), so the
    // tail-layer drop is enough to flip frames into the blank token,
    // producing single-character drops on JFK. See LEARNINGS "Q4_K is
    // too lossy as the default for CTC-decoded ASR" for the full
    // diagnosis.
    //
    // Default: keep the last 12 encoder layers (cutoff = n_enc - 12) at
    // F16; quantize earlier layers normally. Override the cutoff via env
    // (count of tail layers to keep at F16; 0 = full quant, n_enc =
    // skip whole encoder). Opt out entirely with
    // CRISPASR_OMNIASR_QUANT_ALL=1 to ship a smaller variant at the
    // documented ~22% WER cost.
    const bool is_omniasr_ctc =
        (arch.find("omniasr-ctc") != std::string::npos) || (arch.find("omniasr_ctc") != std::string::npos);
    int omniasr_n_enc = 0;
    // Default: keep first 4 encoder layers at F16. Empirically determined
    // by sweeping CRISPASR_OMNIASR_KEEP_F16_HEAD ∈ {0, 4, 8, 12, 16} on
    // JFK (Q4_K + head=N → 5% WER) vs uniform Q4_K (22.7% WER) vs Q8_0
    // (0% WER). head=4 is the smallest cutoff that prevents noise from
    // compounding through the residual stream — it adds ~107 MB to the
    // Q4_K size (551→658 MB) for ~17 percentage points of WER recovery.
    //
    // Counter-intuitive finding: tail-skip was WORSE than uniform Q4_K
    // (preserves accumulated upstream noise more faithfully through F16
    // math). Don't try to "save" the late layers; stop noise at entry.
    int omniasr_keep_head = 4;
    int omniasr_keep_tail = 0;
    if (is_omniasr_ctc) {
        int key = gguf_find_key(ctx_in, "omniasr.n_enc_layers");
        if (key >= 0)
            omniasr_n_enc = (int)gguf_get_val_u32(ctx_in, key);
        if (const char* env_h = std::getenv("CRISPASR_OMNIASR_KEEP_F16_HEAD"))
            omniasr_keep_head = std::max(0, atoi(env_h));
        if (const char* env_t = std::getenv("CRISPASR_OMNIASR_KEEP_F16_TAIL"))
            omniasr_keep_tail = std::max(0, atoi(env_t));
    }
    const char* env_omniasr_all = std::getenv("CRISPASR_OMNIASR_QUANT_ALL");
    const bool omniasr_quant_all = is_omniasr_ctc && env_omniasr_all && *env_omniasr_all && *env_omniasr_all != '0';
    // Layers in [0, head_cutoff) stay F16; layers in [tail_cutoff, n_enc) stay F16.
    const int omniasr_head_cutoff = is_omniasr_ctc && !omniasr_quant_all ? omniasr_keep_head : 0;
    const int omniasr_tail_cutoff =
        is_omniasr_ctc && !omniasr_quant_all ? std::max(0, omniasr_n_enc - omniasr_keep_tail) : omniasr_n_enc;
    if (is_omniasr_ctc && !omniasr_quant_all && (omniasr_keep_head + omniasr_keep_tail) > 0) {
        if (omniasr_keep_head > 0 && omniasr_keep_tail == 0) {
            printf("%s: omniasr-ctc — keeping enc.0-%d (head) at F16 to "
                   "prevent CTC drift (CRISPASR_OMNIASR_QUANT_ALL=1 to override)\n",
                   __func__, omniasr_head_cutoff - 1);
        } else {
            printf("%s: omniasr-ctc — keeping enc.0-%d (head) + enc.%d-%d (tail) at F16\n", __func__,
                   omniasr_head_cutoff - 1, omniasr_tail_cutoff, omniasr_n_enc - 1);
        }
    }

    // Qwen3-TTS: the talker block weights (attn, ffn) are safe to quantize,
    // but several tensor groups are read via ggml_backend_tensor_get /
    // lookup_rows and are precision-sensitive:
    //   - speaker.* — ECAPA speaker encoder (small 1D/3D convs)
    //   - code_pred.token_embd.* — codec embedding lookups
    //   - code_pred.output.* — per-codebook lm_head (small, sampling-critical)
    //   - talker.token_embd.* — text/audio token embedding lookup
    //   - talker.text_proj.* — text projection (small)
    //   - talker.codec_bridge.* — codec bridge projection (small)
    //   - code_pred.small_to_mtp.* — 1.7B dimension projection (small)
    // The bulk weights (talker.blk.*.attn_*, talker.blk.*.ffn_*,
    // code_pred.blk.*) are safe to quantize.
    const bool is_qwen3_tts = (arch.find("qwen3tts") != std::string::npos);

    // MOSS-TTS-v1.5 (arch "moss-tts"): quantize the Qwen3-8B backbone bulk
    // (llm.blk.*.attn_*, llm.blk.*.ffn_*) but keep the precision-sensitive
    // lookups/heads at F16 — the token embedding (llm.embed), the text lm_head
    // (llm.lm_head), and the 32 audio embedding tables + 32 audio LM heads
    // (moss.audio_embed.*, moss.audio_head.*) that map codebook indices to/from
    // hidden space and drive per-codebook sampling. (The transformer codec ships
    // as a separate F16 GGUF and is never quantized here.)
    const bool is_moss_tts = (arch.find("moss-tts") != std::string::npos || arch.find("moss_tts") != std::string::npos);

    // OmniVoice: Qwen3 backbone + audio_embeddings + audio_heads.
    // Keep audio_embd (8200×1024) and audio_output (8200×1024) at original
    // precision — they map codebook indices to/from hidden space, and
    // quantization noise in the output head corrupts the masked iterative
    // decode. LLM backbone weights (llm.blk.*) are safe to quantize.
    const bool is_omnivoice = (arch.find("omnivoice") != std::string::npos);

    // Parler TTS: DAC audio codec weights are precision-sensitive. Audio
    // codecs reconstruct waveforms from codebook embeddings and small
    // conv stacks — quantization noise in the decoder produces audible
    // artefacts (same reasoning as chatterbox vocoder skip). Keep all
    // dac.* tensors at original precision; the T5 encoder and MusicGen
    // decoder weights are safe to quantize.
    const bool is_parler = (arch.find("parler") != std::string::npos);

    // Dia TTS: 1.6B Llama-style encoder + AR decoder with DAC codec.
    // Dia uses scale=1.0 attention (no 1/sqrt(d)) making it sensitive
    // to quantization noise — similar to the OmniASR CTC drift issue.
    // Quantize Q/K/V/O projections + MLP (gate/up/wo) + decoder heads.
    // Keep embeddings, norms, and DAC codec at original precision.
    const bool is_dia = (arch.find("dia") != std::string::npos);

    // VibeVoice TTS (arch "vibevoice-tts"): two Qwen2 backbones (lm.*, tts_lm.*)
    // drive a diffusion prediction head (pred.*) that emits acoustic latents,
    // which an acoustic connector (at_conn.*, semantic se_conn.*) feeds back
    // into the LM, an EOS classifier (tts_eos.*) stops, and a VAE decoder
    // (at_dec.*, st_dec.*) renders to waveform. The prediction head runs under
    // Classifier-Free Guidance (cfg_scale=3.0 for the Realtime model) over 20
    // DPM-Solver++ steps, so a small q8_0/q4 error in pred.* is amplified ~3×
    // per step and compounds across steps and across the AR feedback loop —
    // enough to push the first few frames onto a wrong diffusion trajectory
    // that decodes as a hallucinated non-speech "music" onset before the voice
    // (issue #171, seen on q8_0). The default (no carve-out) quantizes 22/26
    // pred.* tensors plus at_conn/tts_eos. Keep that trajectory/control stack —
    // pred.*, at_conn.*, se_conn.*, tts_eos.*, tts_types.* — at source precision
    // (all small: q8_0 grows ~40 MB, q4_k ~75 MB). The VAE decoder (at_dec.*,
    // st_dec.*) is deliberately NOT protected: it is a large deterministic
    // renderer, it commonly runs on CPU (VIBEVOICE_VAE_BACKEND), and the
    // published q4_k that quantizes it round-trips ASR perfectly — protecting it
    // would nearly double the q4_k size (its whole point is to be small) for no
    // quality gain and without touching the music-onset cause. Force full quant
    // with CRISPASR_VIBEVOICE_QUANT_ALL=1.
    const bool is_vibevoice = (arch.find("vibevoice") != std::string::npos);
    const char* env_vv_all = std::getenv("CRISPASR_VIBEVOICE_QUANT_ALL");
    const bool vibevoice_quant_all = is_vibevoice && env_vv_all && *env_vv_all && *env_vv_all != '0';

    // Zonos TTS: 26-layer GQA transformer + 9-codebook DAC heads.
    // Uniformly quantizing all tensors inflates the EOS logit at prefill
    // by ~0.9 units (−1.125 → −0.21 in Q4_K), pushing P(EOS) from ~38 %
    // to ~60 %+ and causing every seed to emit EOS at step 0. The error
    // accumulates from two sources: backbone hidden-state drift AND
    // per-codebook head weight noise. Keeping the output heads + input
    // embeddings + prefix-conditioner at F16 adds only ~82 MB overhead
    // (36 + 36 + 10) but eliminates the EOS boundary instability.
    // Only the 210 backbone projection tensors are quantized.
    const bool is_zonos = (arch.find("zonos") != std::string::npos);

    // LFM2-Audio: hybrid conv+attention backbone. The text/audio embedding
    // (lfm.embed_tokens — also serves as LM head via tied weights), the
    // audio adapter MLP, and the Mimi codec are all precision-sensitive.
    // The FastConformer encoder's FFN and attention projection weights can
    // be quantized; the depthwise conv weights are too small. Only the
    // LFM backbone layers (lfm.layers.*) and depthformer layers
    // (depth.layers.*) have bulk weights safe for Q4_K. Keep:
    //   - lfm.embed_tokens (sampling-critical, like llama output.weight)
    //   - audio_embd.* (codebook embedding lookups)
    //   - adapter.* (small, precision-sensitive)
    //   - mimi.* (audio codec — small convs, codebook lookups)
    //   - encoder.* (FastConformer — conformer drift issue, same as
    //     canary/omniasr CTC)
    //   - depth.codebook.* (codebook embedding lookups for TTS)
    //   - preprocessor.* (mel filterbank)
    const bool is_lfm2_audio = (arch.find("lfm2-audio") != std::string::npos);

    // Mini-Omni2: Whisper-small encoder + whisperMLP adapter + Qwen2-0.5B LLM.
    // Only the LLM layers (llm.blk.*) should be quantized. Keep:
    //   - audio.* (Whisper encoder — conformer drift, same issue as canary)
    //   - adapter.* (small SwiGLU adapter, precision-sensitive)
    //   - llm.token_embd.weight (tied with lm_head, sampling-critical)
    //   - llm.output_norm.weight (small, F32 anyway)
    const bool is_mini_omni2 = (arch.find("mini-omni2") != std::string::npos);

    // MioTTS: Qwen3 LLM + MioCodec vocoder. The codec (wave_prenet,
    // wave_decoder, wave_prior/post_net, wave_upsampler, istft_head) is
    // precision-sensitive — SnakeBeta's sin²(exp(α)*x) amplifies F16
    // rounding through the UpSampler ConvTranspose chain. Keep all
    // codec.* weights at original precision; only the LLM layers (blk.*)
    // and output/embedding can be quantized.
    const bool is_miotts = (arch.find("miotts") != std::string::npos);

    // Canary-Qwen: FastConformer encoder + linear projection + Qwen3-1.7B LLM.
    // The encoder is precision-sensitive (conformer drift), keep at source.
    // Only LLM block projections (blk.*.attn_*, blk.*.ffn_*) should be quantized.
    // Keep: encoder.*, preprocessor.*, proj.*, token_embd.*, output_norm.*
    const bool is_canary_qwen =
        (arch.find("canary_qwen") != std::string::npos || arch.find("canary-qwen") != std::string::npos);

    // Qwen3-ASR: Whisper-style 18-layer full-attention audio tower + Qwen3
    // LLM. Sub-8-bit audio.* compounds per-block drift (#218 diff harness:
    // blk00 cos_mean 0.9996 → blk17 0.973 on a 145 s clip) until greedy
    // decode flips to "language none" / empty output on long audio. Floor
    // the tower at Q8_0 (~lossless, cos ≥ 0.9999/blk) instead of keeping it
    // F16 outright — same drift class as the canary/mini-omni2 encoder
    // carve-outs at ~half their size cost. CRISPASR_QWEN3ASR_QUANT_AUDIO=1
    // quantizes audio.* at the target type anyway (A/B gate).
    const bool is_qwen3_asr = (arch == "qwen3asr" || arch == "qwen3-asr");
    const bool qwen3asr_quant_audio = []() {
        const char* e = std::getenv("CRISPASR_QWEN3ASR_QUANT_AUDIO");
        return e && *e && *e != '0';
    }();

    // Bark TTS: 3 GPT-2 sub-models + EnCodec decoder.
    // Embeddings (token_embd, pos_embd), output heads, and the entire
    // EnCodec decoder are read via CPU tensor_get_row_f32 / tensor_get_all_f32
    // and are precision-sensitive. Only attn/ffn projection weights
    // (attn_qkv, attn_output, ffn_up, ffn_down) should be quantized.
    // Verified: Q4_K of all tensors produces near-zero audio (peak 0.001);
    // Q8_0 works fine; selective Q4_K (projections only) is safe.
    const bool is_bark = (arch.find("bark") != std::string::npos);

    // Orpheus TTS: Llama-3.2-3B with SNAC 24 kHz codec. The token embedding
    // is tied with the LM head (no separate output.weight). The talker emits
    // peaked SNAC codec distributions — quantizing the embedding/head breaks
    // the super-frame slot pattern and produces gibberish. Keep
    // talker.token_embd.weight at F16; block projections are safe to quantize.
    const bool is_orpheus = (arch.find("orpheus") != std::string::npos);

    // TADA TTS: Llama-3.2-3B talker + per-token flow-matching head. The
    // conditioning path and FM head are unusually precision-sensitive:
    // small errors in acoustic/time embeddings and FM velocity predictions
    // compound over Euler steps and CFG, changing predicted durations and
    // codec-frame placement. Keep token embeddings, all TADA conditioning,
    // and the FM head at source precision; quantize only the large talker
    // block projection matrices.
    const bool is_tada = (arch.find("tada-tts") != std::string::npos || arch.find("tada_tts") != std::string::npos);
    const char* env_tada_all = std::getenv("CRISPASR_TADA_QUANT_ALL");
    const bool tada_quant_all = is_tada && env_tada_all && *env_tada_all && *env_tada_all != '0';
    if (is_tada && tada_quant_all) {
        printf("%s: tada-tts - quantizing precision-sensitive tada.* tensors (experimental override)\n", __func__);
    }
    int tada_n_layers = 0;
    int tada_keep_head = 0;
    int tada_keep_tail = 0;
    if (is_tada) {
        int key = gguf_find_key(ctx_in, "tada.talker.n_layers");
        if (key >= 0)
            tada_n_layers = (int)gguf_get_val_u32(ctx_in, key);
        if (const char* env_h = std::getenv("CRISPASR_TADA_KEEP_F16_HEAD"))
            tada_keep_head = std::max(0, atoi(env_h));
        if (const char* env_t = std::getenv("CRISPASR_TADA_KEEP_F16_TAIL"))
            tada_keep_tail = std::max(0, atoi(env_t));
        if (tada_quant_all) {
            tada_keep_head = 0;
            tada_keep_tail = 0;
        }
        if (!tada_quant_all && (tada_keep_head > 0 || tada_keep_tail > 0)) {
            const int tail_start = std::max(0, tada_n_layers - tada_keep_tail);
            if (tada_keep_head > 0 && tada_keep_tail > 0) {
                printf("%s: tada-tts - keeping talker.blk.0-%d (head) + talker.blk.%d-%d (tail) at source precision "
                       "(override with CRISPASR_TADA_KEEP_F16_HEAD/TAIL, full quant with CRISPASR_TADA_QUANT_ALL=1)\n",
                       __func__, tada_keep_head - 1, tail_start, tada_n_layers - 1);
            } else if (tada_keep_head > 0) {
                printf("%s: tada-tts - keeping talker.blk.0-%d (head) at source precision "
                       "(override with CRISPASR_TADA_KEEP_F16_HEAD/TAIL, full quant with CRISPASR_TADA_QUANT_ALL=1)\n",
                       __func__, tada_keep_head - 1);
            } else {
                printf("%s: tada-tts - keeping talker.blk.%d-%d (tail) at source precision "
                       "(override with CRISPASR_TADA_KEEP_F16_HEAD/TAIL, full quant with CRISPASR_TADA_QUANT_ALL=1)\n",
                       __func__, tail_start, tada_n_layers - 1);
            }
        }
    }

    // dots.tts: Qwen2.5-1.5B LLM + 18L DiT flow-matching + 24L PatchEncoder.
    // The ENTIRE DiT (velocity field predictor) must stay at source precision:
    // it runs in a CFG flow-matching loop (16 Euler ODE steps × 18 layers × 2
    // CFG = 576 forwards) where per-step q8 noise compounds and DERAILS
    // generation (validated: q8 DiT blocks → flow-match cos 0.994 → no-EOS
    // runaway / garbled audio). So keep ALL dots.dit.* at F16, not just the
    // conditioning pathway — the attn/ffn block weights matter too. The LLM
    // (cos 0.999 on q8, dots-tts-llm diff) and PatchEncoder (cos 0.9999 on q8)
    // blocks quantize safely, so q8 LLM + q8 penc + F16 DiT gives a ~2 GB core
    // with an accurate flow-match (mixed-quant footprint default). Keep at
    // source precision:
    //   - dots.dit.* — the whole flow-matching head (blocks + AdaLN + in/final/time)
    //   - hidden_proj.* — LLM→DiT condition projection
    //   - latent_proj.* — latent→DiT input
    //   - coordinate_proj.* — noise coordinate projection
    //   - xvec_proj.* — speaker embedding projection
    //   - eos_proj.* — EOS detection head
    //   - llm.tok_emb.* — token embedding (sampling-critical)
    //   - latent_stats.* — denormalization constants
    //   - penc.in_proj/out_proj/ds_conv — PatchEncoder I/O
    // Vocoder and speaker encoder are in separate GGUFs — quantize normally.
    const bool is_dots_tts = (arch.find("dots-tts") != std::string::npos || arch.find("dots_tts") != std::string::npos);
    // ARK-ASR-3B: keep the tied embedding/lm_head (dec.embed.weight) and the
    // whole Whisper encoder + adapter (mel-sensitive, small vs the 36L decoder)
    // at F16; quantize only the decoder attn/ffn projections.
    const bool is_arkasr = (arch.find("arkasr") != std::string::npos);

    // higgs-audio-v3-stt: Whisper encoder + Qwen3-1.7B decoder with TIED
    // input/output embeddings (token_embd.weight == output.weight). Both the
    // embedding lookup and the lm_head share these rows, so quantization noise
    // there directly perturbs every logit — keep them at source precision (the
    // attention/FFN blocks quantize normally). The learned audio.embed_positions
    // has no "weight" suffix so it is already skipped by the is_weight test, and
    // the conv stacks are 3-D (skipped by ok_dims).
    const bool is_higgs = (arch == "higgs-stt");

    // Parakeet RNNT: the transducer joint network (joint.{enc,pred,out}.weight)
    // and decoder embedding are structurally sensitive to quantization noise.
    // The joint network's blank/non-blank decision is a ~3001-way argmax where
    // the blank token competes with every real token — quantization noise can
    // flip frames from blank to non-blank, causing the RNNT greedy decoder to
    // enter a non-terminating emission loop. TDT models are less affected
    // because the duration head provides an independent advance mechanism.
    // Default: keep joint.* and decoder.embed.* at source precision for RNNT
    // models (n_tdt_durations==0). Override with CRISPASR_PARAKEET_QUANT_ALL=1.
    const bool is_parakeet = (arch == "parakeet");
    bool parakeet_is_rnnt = false;
    if (is_parakeet) {
        int key = gguf_find_key(ctx_in, "parakeet.n_tdt_durations");
        if (key >= 0)
            parakeet_is_rnnt = (gguf_get_val_u32(ctx_in, key) == 0);
    }
    const char* env_parakeet_all = std::getenv("CRISPASR_PARAKEET_QUANT_ALL");
    const bool parakeet_quant_all = is_parakeet && env_parakeet_all && *env_parakeet_all && *env_parakeet_all != '0';
    if (is_parakeet && parakeet_is_rnnt && !parakeet_quant_all) {
        printf("%s: parakeet RNNT — keeping joint.* and decoder.embed.* at source precision "
               "(override with CRISPASR_PARAKEET_QUANT_ALL=1)\n",
               __func__);
    }

    // First pass: determine which tensors will be quantized and compute
    // their target types. We need this BEFORE adding tensors to ctx_out
    // so that gguf_add_tensor computes correct offsets for the quantized
    // sizes.
    const int n_tensors = gguf_get_n_tensors(ctx_in);
    std::vector<ggml_type> target_types(n_tensors);
    // Effective quantization row length per tensor — t->ne[0] except for the
    // FastConformer 3D conv pointwise carve-out below, which quantizes as 2D.
    std::vector<int64_t> row_lens(n_tensors, 0);
    // Per-rule --tensor-type match counters (summarised after the first pass).
    std::vector<int> override_hits(g_type_overrides.size(), 0);
    std::vector<int> override_skips(g_type_overrides.size(), 0);

    // Allocate a scratch ggml context for creating modified tensor descriptors.
    ggml_init_params scratch_params = {ggml_tensor_overhead() * (size_t)n_tensors + 1024, nullptr, true};
    ggml_context* ctx_scratch = ggml_init(scratch_params);

    // Normally lm_head / tok_emb / lang_emb are kept at input precision (they are
    // sampling/argmax-critical). CRISPASR_QUANT_LMHEAD=1 lets the caller quantize
    // lm_head too, for measuring its impact (e.g. the TADA aligner, whose forced
    // alignment is robust to logit rounding).
    const bool allow_lmhead = []() {
        const char* e = std::getenv("CRISPASR_QUANT_LMHEAD");
        return e && *e && *e != '0';
    }();

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor* t = ggml_get_tensor(ctx_in_ggml, name);

        std::string sname(name);
        bool is_weight = (sname.find("weight") != std::string::npos) ||
                         (sname.size() >= 2 && sname.substr(sname.size() - 2) == "_w") ||
                         (sname.size() >= 2 && sname.substr(sname.size() - 2) == ".w") ||
                         (sname.find("_proj") != std::string::npos) || (sname.find(".gate") != std::string::npos) ||
                         (sname.find(".up") != std::string::npos) || (sname.find(".wo") != std::string::npos) ||
                         (sname.find(".heads.") != std::string::npos) ||
                         // SwiGLU MLP (.w1/.w2/.w3) and attention QKV (.wq/.wk/.wv) — irodori-tts, f5-tts
                         (sname.size() >= 3 && sname[sname.size() - 3] == '.' && sname[sname.size() - 2] == 'w' &&
                          sname[sname.size() - 1] >= '1' && sname[sname.size() - 1] <= '9') ||
                         (sname.find(".wq") != std::string::npos) || (sname.find(".wk") != std::string::npos) ||
                         (sname.find(".wv") != std::string::npos);
        // FastConformer conv pointwise weights ship as 3D conv tensors
        // (1, d, N), which the 2D-only rule skips — so they stayed F16 even
        // in Q8_0/Q4_K models and ran on the slow CPU F16 mul_mat path
        // (issue #81). Every consumer does reshape_2d + mul_mat, so quantize
        // them as their 2D (d, N) equivalent. Runtimes repack pre-existing
        // GGUFs at load (core_conformer::repack_conv_pw_q8).
        auto ends_with = [&sname](const char* suf) {
            const size_t n = strlen(suf);
            return sname.size() >= n && sname.compare(sname.size() - n, n, suf) == 0;
        };
        const bool pw_conv3d =
            ggml_n_dims(t) == 3 && t->ne[0] == 1 && (ends_with("conv.pw1.weight") || ends_with("conv.pw2.weight"));
        // CREPE conv1d kernels are 3D (K, IC, OC) — allow them through the
        // 2D-only gate, but quantize at the DECLARED ne[0] = K, not the K*IC
        // matmul row. See the is_crepe note above for why.
        const bool crepe_conv3d =
            is_crepe && ggml_n_dims(t) == 3 && sname.rfind("conv", 0) == 0 && ends_with(".weight");
        const bool ok_dims =
            (ggml_n_dims(t) == 2) || pw_conv3d || crepe_conv3d || ((is_firered || is_ecapa) && ggml_n_dims(t) >= 2);
        const int64_t ncols = pw_conv3d ? t->ne[1] : t->ne[0];
        row_lens[i] = ncols;

        // Source may be F32/F16 OR already quantized. Accepting a quantized
        // source lets us re-quantize a big model straight from its q8_0 GGUF
        // (dequantized to F32 in the write loop, then re-quantized to the
        // target) instead of needing the full F16/F32 base on disk — q8_0 is
        // ~lossless (cos ~0.9998) so q8→f32→q4 ≈ f32→q4, and the q8_0 is a
        // fraction of the download. Matters here because several backends
        // (ark-asr 3B, …) have F16 bases too large for the local disks.
        const bool src_ok = (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || ggml_is_quantized(t->type));
        bool should_quantize =
            ggml_is_quantized(qtype) && src_ok && ok_dims && is_weight && (sname.find("norm") == std::string::npos) &&
            (granite_quant_all || sname.find("proj.") != 0) &&
            // CREPE: allow-list, not a deny-list — only the conv kernels and the
            // dense classifier matrix. conv*.bias / conv*_BN.scale / .offset /
            // classifier.bias stay F32 (per-channel affines; see is_crepe note).
            !(is_crepe && !(crepe_conv3d || sname == "classifier.weight")) &&
            !(is_granite_family && !granite_quant_all && sname.find("enc.") == 0) &&
            // MOSS-Audio: keep encoder + adapter + deepstack at F16
            !(arch == "moss_audio" &&
              (sname.find("enc.") == 0 || sname.find("adapter.") == 0 || sname.find("deepstack.") == 0)) &&
            // MOSS-Transcribe: keep encoder + adapter at F16
            // MOSS-Transcribe: keep the audio encoder + adapter at F16, and the
            // TIED token embedding at F16 — `llm.embed.weight` doubles as the
            // output head (lm_head = embed), so quantizing it to q4_k corrupts
            // both the input embeddings and every output logit.
            !(arch == "moss_transcribe" &&
              (sname.find("enc.") == 0 || sname.find("adapter.") == 0 || sname == "llm.embed.weight")) &&
            // MOSS-Transcribe-Diarize: keep encoder + adaptor at F16, and the
            // tied token embedding (token_embd.weight = lm_head).
            !(arch == "moss_transcribe_diarize" &&
              (sname.find("enc.") == 0 || sname.find("adaptor.") == 0 || sname == "token_embd.weight")) &&
            // Voxtral-TTS: keep the semantic VQ codebook (256-d rows — quantizing
            // corrupts the codec's per-frame lookup), the preset voice conditioning
            // embeddings, and the tiny FM input projection at F16/F32.
            !(arch == "voxtral_tts" &&
              (sname == "codec.semantic_cb.weight" || sname.find("voice.") == 0 || sname.find("fm.input_proj") == 0)) &&
            !(sname.find("cls.") == 0 && ggml_nelements(t) < 65536) && (sname.find("enc_proj.") != 0) &&
            (allow_lmhead || (sname.find("lm_head.") != 0)) && (sname.find("tok_emb.") != 0) &&
            (sname.find("lang_emb.") != 0) &&
            !(is_chatterbox && (sname.find("s3.v.") == 0 || sname.find("conds.") == 0 || sname.find("ve.") == 0 ||
                                sname.find("t3.text_emb") == 0 || sname.find("t3.speech_emb") == 0 ||
                                sname.find("t3.wpe") == 0 || sname.find("t3.text_pos_emb") == 0 ||
                                sname.find("t3.speech_pos_emb") == 0 || sname.find("t3.cond.") == 0)) &&
            !(is_cosyvoice3 &&
              (sname == "cosyvoice3.speech_embd.weight" || sname == "cosyvoice3.speech_lm_head.weight" ||
               sname == "cosyvoice3.flow.input_embd.w" || sname == "cosyvoice3.flow.spk_affine.w" ||
               sname == "cosyvoice3.s3tok.fsq.proj.w")) &&
            !is_f5tts &&
            !(is_dots_tts && (sname.find("dots.dit.") == 0 || sname.find(".adaln.") != std::string::npos ||
                              sname.find("dots.hidden_proj.") == 0 || sname.find("dots.latent_proj.") == 0 ||
                              sname.find("dots.coordinate_proj.") == 0 || sname.find("dots.xvec_proj.") == 0 ||
                              sname.find("dots.eos_proj.") == 0 || sname.find("dots.llm.tok_emb.") == 0 ||
                              sname.find("dots.latent_stats.") == 0 || sname.find("dots.penc.in_proj.") == 0 ||
                              sname.find("dots.penc.out_proj.") == 0 || sname.find("dots.penc.ds_conv.") == 0)) &&
            !(is_qwen3_tts && (sname.find("speaker.") == 0 || sname.find("code_pred.token_embd") == 0 ||
                               sname.find("code_pred.output") == 0 || sname.find("code_pred.small_to_mtp") == 0 ||
                               sname.find("talker.token_embd") == 0 || sname.find("talker.text_proj") == 0 ||
                               sname.find("talker.codec_bridge") == 0)) &&
            // moss-tts-local (4B) additionally keeps the 1-layer local/depth
            // transformer (local.*) and its binary stop head (moss.local_text_head)
            // at F16 — precision-sensitive, run n_vq times per frame. (8B has no
            // such tensors, so these terms are inert there.)
            !(is_moss_tts && (sname.find("llm.embed") == 0 || sname.find("llm.lm_head") == 0 ||
                              sname.find("moss.audio_embed") == 0 || sname.find("moss.audio_head") == 0 ||
                              sname.find("local.") == 0 || sname.find("moss.local_text_head") == 0)) &&
            !(is_parler && sname.find("dac.") == 0) &&
            !(is_dia && (sname.find("embedding") != std::string::npos || sname.find("audio_encoder") == 0)) &&
            // VibeVoice: keep the trajectory/control stack — diffusion head
            // (pred.*), acoustic/semantic connectors (at_conn.*, se_conn.*), EOS
            // classifier (tts_eos.*), speech-type table (tts_types.*) — at source
            // precision (see is_vibevoice note). The VAE decoder (at_dec.*/st_dec.*)
            // is intentionally left quantizable.
            !(is_vibevoice && !vibevoice_quant_all &&
              (sname.find("pred.") == 0 || sname.find("at_conn.") == 0 || sname.find("se_conn.") == 0 ||
               sname.find("tts_eos.") == 0 || sname.find("tts_types.") == 0)) &&
            !(is_zonos && (sname.find("heads.") == 0 || sname.find("embeddings.") == 0 ||
                           sname.find("prefix_conditioner.") == 0)) &&
            !(is_bark &&
              (sname.find("token_embd") != std::string::npos || sname.find("pos_embd") != std::string::npos ||
               (sname.find("output") != std::string::npos && sname.find("attn_output") == std::string::npos) ||
               sname.find("encodec.") == 0)) &&
            !(is_lfm2_audio && (sname.find("lfm.embed_tokens") == 0 || sname.find("lfm.embedding_norm") == 0 ||
                                sname.find("audio_embd.") == 0 || sname.find("adapter.") == 0 ||
                                sname.find("mimi.") == 0 || sname.find("encoder.") == 0 ||
                                sname.find("depth.codebook.") == 0 || sname.find("preprocessor.") == 0)) &&
            !(is_mini_omni2 &&
              (sname.find("audio.") == 0 || sname.find("adapter.") == 0 || sname.find("llm.token_embd") == 0)) &&
            !(is_omnivoice && (sname.find("audio_embd") == 0 || sname.find("audio_output") == 0 ||
                               sname.find("llm.token_embd") == 0)) &&
            !(is_canary_qwen &&
              (sname.find("encoder.") == 0 || sname.find("preprocessor.") == 0 || sname.find("proj.") == 0 ||
               sname == "token_embd.weight" || sname == "output.weight")) &&
            !(is_orpheus && sname.find("talker.token_embd") == 0) &&
            !(is_arkasr && (sname.find("dec.embed.") == 0 || sname.find("enc.") == 0 || sname.find("adapter.") == 0)) &&
            !(is_higgs && (sname == "token_embd.weight" || sname == "output.weight")) &&
            !(is_miotts && sname.find("codec.") == 0) &&
            !(is_parakeet && parakeet_is_rnnt && !parakeet_quant_all &&
              (sname.find("joint.") == 0 || sname.find("decoder.embed") == 0)) &&
            !(is_tada && !tada_quant_all && (sname.find("talker.token_embd") == 0 || sname.find("tada.") == 0)) &&
            ([&]() {
                if (!is_tada || tada_quant_all || (tada_keep_head == 0 && tada_keep_tail == 0))
                    return true;
                if (sname.rfind("talker.blk.", 0) != 0)
                    return true;
                int idx = 0;
                size_t p = strlen("talker.blk.");
                while (p < sname.size() && sname[p] >= '0' && sname[p] <= '9') {
                    idx = idx * 10 + (sname[p] - '0');
                    p++;
                }
                if (p == strlen("talker.blk."))
                    return true;
                const bool in_head = idx < tada_keep_head;
                const bool in_tail = tada_keep_tail > 0 && idx >= std::max(0, tada_n_layers - tada_keep_tail);
                return !(in_head || in_tail);
            }()) &&
            ([&]() {
                if (!is_omniasr_ctc || omniasr_quant_all ||
                    (omniasr_head_cutoff == 0 && omniasr_tail_cutoff >= omniasr_n_enc))
                    return true;
                if (sname.size() < 5 || sname.compare(0, 4, "enc.") != 0)
                    return true;
                int idx = 0;
                size_t p = 4;
                while (p < sname.size() && sname[p] >= '0' && sname[p] <= '9') {
                    idx = idx * 10 + (sname[p] - '0');
                    p++;
                }
                if (p == 4)
                    return true;
                const bool in_head = idx < omniasr_head_cutoff;
                const bool in_tail = idx >= omniasr_tail_cutoff;
                return !(in_head || in_tail);
            }());

        // Determine actual quant type with row-size fallback
        ggml_type qt = qtype;
        if (should_quantize && ncols % ggml_blck_size(qt) != 0) {
            ggml_type fallback = GGML_TYPE_COUNT;
            switch (qtype) {
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                fallback = GGML_TYPE_Q4_0;
                break;
            case GGML_TYPE_Q5_K:
                fallback = GGML_TYPE_Q5_0;
                break;
            case GGML_TYPE_Q6_K:
                fallback = GGML_TYPE_Q8_0;
                break;
            // IQ4_XS uses 256-wide super-blocks; fall back to IQ4_NL (32-wide,
            // same 4-bit non-linear codebook) when the row isn't 256-aligned,
            // then to legacy Q4_0 if it isn't 32-aligned either.
            case GGML_TYPE_IQ4_XS:
                fallback = GGML_TYPE_IQ4_NL;
                break;
            case GGML_TYPE_IQ4_NL:
                fallback = GGML_TYPE_Q4_0;
                break;
            default:
                break;
            }
            if (fallback != GGML_TYPE_COUNT && ncols % ggml_blck_size(fallback) == 0) {
                qt = fallback;
            } else {
                should_quantize = false;
            }
        }

        // Also handle granite enc F32→F16 downcast
        bool granite_f16 =
            !should_quantize && granite_enc_to_f16 && t->type == GGML_TYPE_F32 && sname.find("enc.") == 0 &&
            sname.find("norm") == std::string::npos && sname.find("running_mean") == std::string::npos &&
            sname.find("running_var") == std::string::npos && sname.find("rel_pos") == std::string::npos &&
            sname.find("conv_bn") == std::string::npos && ggml_n_dims(t) == 2;

        if (should_quantize) {
            target_types[i] = qt;
        } else if (granite_f16) {
            target_types[i] = GGML_TYPE_F16;
        } else {
            target_types[i] = t->type;
        }

        // Qwen3-ASR audio-tower Q8_0 floor (see is_qwen3_asr note above).
        // Applies only when the chosen type is a sub-8-bit quant; Q8_0 and
        // unquantized targets pass through. Rows not 32-aligned keep source.
        if (is_qwen3_asr && !qwen3asr_quant_audio && sname.find("audio.") == 0 && should_quantize &&
            ggml_is_quantized(target_types[i]) && target_types[i] != GGML_TYPE_Q8_0) {
            target_types[i] = (ncols % ggml_blck_size(GGML_TYPE_Q8_0) == 0) ? GGML_TYPE_Q8_0 : t->type;
        }

        // FastConformer conv pointwise Q8_0 floor: sub-8-bit quants are both
        // lossier and slower than Q8_0 on CPU for these matmul shapes, and
        // Q8_0 here makes newly-quantized GGUFs byte-match what
        // core_conformer::repack_conv_pw_q8 produces at load from old files.
        if (pw_conv3d && should_quantize && ggml_is_quantized(target_types[i]) && target_types[i] != GGML_TYPE_Q8_0 &&
            ncols % ggml_blck_size(GGML_TYPE_Q8_0) == 0) {
            target_types[i] = GGML_TYPE_Q8_0;
        }
        // Idempotency for already-fixed GGUFs: their conv pw is 2D Q8_0, so
        // pw_conv3d no longer matches and a sub-8-bit re-run would silently
        // down-quantize the floor above to a k-quant. Keep Q8_0.
        if (!pw_conv3d && ggml_n_dims(t) == 2 && t->type == GGML_TYPE_Q8_0 && should_quantize &&
            ggml_is_quantized(target_types[i]) && target_types[i] != GGML_TYPE_Q8_0 &&
            (ends_with("conv.pw1.weight") || ends_with("conv.pw2.weight"))) {
            target_types[i] = GGML_TYPE_Q8_0;
        }

        // User per-tensor override (--tensor-type <regex>=<type>). First match
        // wins; overrides the arch guards above. A quant override on a <2-D or
        // ill-tiled row is skipped (with a note) rather than corrupting output.
        for (size_t r = 0; r < g_type_overrides.size(); r++) {
            if (!std::regex_search(sname, g_type_overrides[r].first))
                continue;
            ggml_type ov = g_type_overrides[r].second;
            if (ggml_is_quantized(ov)) {
                if (ggml_n_dims(t) < 2) {
                    override_skips[r]++;
                    break;
                }
                ggml_type fit = crispasr_row_fit(ov, ncols);
                if (fit == GGML_TYPE_COUNT) {
                    override_skips[r]++;
                    break;
                }
                ov = fit;
            }
            target_types[i] = ov;
            override_hits[r]++;
            break;
        }

        // Create a tensor descriptor with the target type for ctx_out
        if (target_types[i] != t->type) {
            struct ggml_tensor* t_out;
            if (pw_conv3d && ggml_is_quantized(target_types[i])) {
                // Quantized as the 2D (d, N) matmul view (see carve-out above).
                t_out = ggml_new_tensor_2d(ctx_scratch, target_types[i], t->ne[1], t->ne[2]);
            } else {
                t_out = ggml_new_tensor(ctx_scratch, target_types[i], ggml_n_dims(t), t->ne);
            }
            ggml_set_name(t_out, name);
            gguf_add_tensor(ctx_out, t_out);
        } else {
            gguf_add_tensor(ctx_out, t);
        }
    }

    for (size_t r = 0; r < g_type_overrides.size(); r++) {
        printf("%s: tensor-type override '%s' → %d tensors", __func__, g_type_override_src[r].c_str(),
               override_hits[r]);
        if (override_skips[r])
            printf(" (%d skipped: <2-D or row width)", override_skips[r]);
        printf("\n");
    }

    // Allocate output file
    printf("%s: writing quantized model to '%s'\n", __func__, fname_out.c_str());
    FILE* fout = fopen(fname_out.c_str(), "w+b");
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        gguf_free(ctx_in);
        gguf_free(ctx_out);
        if (ctx_in_ggml)
            ggml_free(ctx_in_ggml);
        return false;
    }

    // Write metadata placeholder
    const size_t meta_size = gguf_get_meta_size(ctx_out);
    std::vector<uint8_t> meta_data(meta_size, 0);
    fwrite(meta_data.data(), 1, meta_size, fout);

    // Open input file for data reading
    FILE* fin = fopen(fname_inp.c_str(), "rb");
    const size_t data_offset_in = gguf_get_data_offset(ctx_in);

    std::vector<float> f32_data;
    std::vector<uint8_t> q_data;
    int n_quantized = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor* t = ggml_get_tensor(ctx_in_ggml, name);

        enum ggml_type type = t->type;
        size_t size = ggml_nbytes(t);
        size_t offset = data_offset_in + gguf_get_tensor_offset(ctx_in, i);

        printf("[%3d/%3d] %-40s - %10s, ", i + 1, n_tensors, name, ggml_type_name(type));

        // Use pre-computed target type from first pass
        ggml_type qtype_used = target_types[i];
        bool quantize = ggml_is_quantized(qtype_used) && (qtype_used != type);

        // Use 64-bit seek to avoid overflow on files > 2 GB (Windows
        // long is 32-bit even on x86_64, wrapping at 2^31).
#ifdef _WIN32
        _fseeki64(fin, (__int64)offset, SEEK_SET);
#else
        fseeko(fin, (off_t)offset, SEEK_SET);
#endif

        if (quantize) {
            n_quantized++;
            printf("quantizing to %s... ", ggml_type_name(qtype_used));

            const int64_t nelements = ggml_nelements(t);
            f32_data.resize(nelements);

            if (type == GGML_TYPE_F32) {
                if (fread(f32_data.data(), sizeof(float), nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f32 data\n");
                    return false;
                }
            } else if (type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16_data(nelements);
                if (fread(f16_data.data(), sizeof(ggml_fp16_t), nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f16 data\n");
                    return false;
                }
                for (int j = 0; j < nelements; j++)
                    f32_data[j] = ggml_fp16_to_fp32(f16_data[j]);
            } else {
                // Quantized source: read the raw quantized bytes and dequantize
                // to F32 via the type's traits, then re-quantize to the target.
                const size_t src_bytes = ggml_nbytes(t);
                std::vector<uint8_t> qbuf(src_bytes);
                if (fread(qbuf.data(), 1, src_bytes, fin) != src_bytes) {
                    fprintf(stderr, "failed to read quantized source data\n");
                    return false;
                }
                const ggml_type_traits* tr = ggml_get_type_traits(type);
                if (!tr || !tr->to_float) {
                    fprintf(stderr, "no dequantizer for source type %s\n", ggml_type_name(type));
                    return false;
                }
                tr->to_float(qbuf.data(), f32_data.data(), nelements);
            }

            const int64_t n_per_row = row_lens[i] > 0 ? row_lens[i] : t->ne[0];
            const size_t max_q_size = ggml_row_size(qtype_used, n_per_row) * (nelements / n_per_row);
            q_data.resize(max_q_size);

            // Importance matrix (if loaded and shape-matched): steers k-quant/IQ
            // precision toward the columns the calibration data actually used.
            const float* imatrix = nullptr;
            if (!g_imatrix.empty()) {
                auto it = g_imatrix.find(name);
                if (it != g_imatrix.end()) {
                    if ((int64_t)it->second.size() == n_per_row) {
                        imatrix = it->second.data();
                        printf("(imatrix) ");
                    } else {
                        printf("(imatrix shape %zu!=%lld, skipped) ", it->second.size(), (long long)n_per_row);
                    }
                }
            }

            size_t q_size = ggml_quantize_chunk(qtype_used, f32_data.data(), q_data.data(), 0, nelements / n_per_row,
                                                n_per_row, imatrix);

            fwrite(q_data.data(), 1, q_size, fout);

            // Padding
            size_t pad = GGML_PAD(q_size, GGUF_DEFAULT_ALIGNMENT) - q_size;
            for (size_t j = 0; j < pad; j++)
                fputc(0, fout);

            printf("done\n");
        } else if ((qtype_used == GGML_TYPE_F16 || qtype_used == GGML_TYPE_F32) && qtype_used != type) {
            // Up/down-cast to F16 or F32 from ANY source (F32/F16/quantized):
            // e.g. the granite encoder F32→F16 downcast, or a --tensor-type
            // override pinning a quantized tensor to F16. Read → F32 → write.
            printf("%s -> %s... ", ggml_type_name(type), ggml_type_name(qtype_used));
            const int64_t nelements = ggml_nelements(t);
            std::vector<float> f32(nelements);
            if (type == GGML_TYPE_F32) {
                if (fread(f32.data(), sizeof(float), nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f32 data\n");
                    return false;
                }
            } else if (type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16in(nelements);
                if (fread(f16in.data(), sizeof(ggml_fp16_t), nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f16 data\n");
                    return false;
                }
                for (int64_t j = 0; j < nelements; j++)
                    f32[j] = ggml_fp16_to_fp32(f16in[j]);
            } else {
                std::vector<uint8_t> qbuf(size);
                if (fread(qbuf.data(), 1, size, fin) != size) {
                    fprintf(stderr, "failed to read quantized source data\n");
                    return false;
                }
                const ggml_type_traits* tr = ggml_get_type_traits(type);
                if (!tr || !tr->to_float) {
                    fprintf(stderr, "no dequantizer for source type %s\n", ggml_type_name(type));
                    return false;
                }
                tr->to_float(qbuf.data(), f32.data(), nelements);
            }
            size_t out_bytes;
            if (qtype_used == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16(nelements);
                for (int64_t j = 0; j < nelements; j++)
                    f16[j] = ggml_fp32_to_fp16(f32[j]);
                out_bytes = (size_t)nelements * sizeof(ggml_fp16_t);
                fwrite(f16.data(), 1, out_bytes, fout);
            } else {
                out_bytes = (size_t)nelements * sizeof(float);
                fwrite(f32.data(), 1, out_bytes, fout);
            }
            size_t pad = GGML_PAD(out_bytes, GGUF_DEFAULT_ALIGNMENT) - out_bytes;
            for (size_t j = 0; j < pad; j++)
                fputc(0, fout);
            printf("done\n");
        } else {
            printf("copying... ");
            std::vector<uint8_t> raw_data(size);
            if (fread(raw_data.data(), 1, size, fin) != size) {
                fprintf(stderr, "failed to read raw data\n");
                return false;
            }
            fwrite(raw_data.data(), 1, size, fout);

            // Padding
            size_t pad = GGML_PAD(size, GGUF_DEFAULT_ALIGNMENT) - size;
            for (size_t j = 0; j < pad; j++)
                fputc(0, fout);
            printf("done\n");
        }
    }

    if (n_quantized == 0) {
        fprintf(stderr,
                "%s: WARNING — 0 of %d tensors were quantized. The output file is the\n"
                "  same size as the input. Check that the architecture supports block\n"
                "  quantization (ne[0] must be ≥32 for Q8_0, ≥256 for Q4_K) and that\n"
                "  weight tensors are 2-D (or a supported conv architecture).\n",
                __func__, n_tensors);
    } else {
        printf("%s: quantized %d / %d tensors\n", __func__, n_quantized, n_tensors);
    }

    // Rewrite metadata header. Since tensor types were set correctly in
    // the first pass (before gguf_add_tensor), meta_size should be stable.
    fflush(fout);
    fseek(fout, 0, SEEK_SET);
    gguf_get_meta_data(ctx_out, meta_data.data());
    size_t written = fwrite(meta_data.data(), 1, meta_size, fout);
    fflush(fout);
    printf("%s: metadata rewrite: %zu / %zu bytes at offset 0, magic=0x%08x\n", __func__, written, meta_size,
           *(uint32_t*)meta_data.data());

    fclose(fin);
    fclose(fout);
    gguf_free(ctx_in);
    gguf_free(ctx_out);
    ggml_free(ctx_in_ggml);
    ggml_free(ctx_scratch);

    return true;
}

int main(int argc, char** argv) {
    // Collect positional args, allowing an optional --imatrix flag anywhere.
    std::vector<std::string> pos;
    std::string imatrix_path;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--imatrix") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--imatrix requires a file path\n");
                return 1;
            }
            imatrix_path = argv[++i];
        } else if (a == "--tensor-type") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--tensor-type requires <regex>=<type>\n");
                return 1;
            }
            std::string rule = argv[++i];
            size_t eq = rule.rfind('=');
            if (eq == std::string::npos) {
                fprintf(stderr, "--tensor-type expects <regex>=<type>, got '%s'\n", rule.c_str());
                return 1;
            }
            std::string pat = rule.substr(0, eq), tn = rule.substr(eq + 1);
            ggml_type ot = crispasr_parse_type_name(tn);
            if (ot == GGML_TYPE_COUNT) {
                fprintf(stderr, "--tensor-type: unknown type '%s'\n", tn.c_str());
                return 1;
            }
            try {
                g_type_overrides.emplace_back(std::regex(pat), ot);
                g_type_override_src.push_back(rule);
            } catch (const std::regex_error& e) {
                fprintf(stderr, "--tensor-type: bad regex '%s': %s\n", pat.c_str(), e.what());
                return 1;
            }
        } else {
            pos.push_back(a);
        }
    }

    if (pos.size() != 3) {
        fprintf(stderr, "usage: %s model-f16.gguf model-quant.gguf type [--imatrix <file>]\n", argv[0]);
        fprintf(stderr, "             [--tensor-type <regex>=<type> ...]\n");
        fprintf(stderr, "  input may be F16/F32 OR an already-quantized (e.g. q8_0) GGUF — a\n");
        fprintf(stderr, "  quantized source is dequantized then re-quantized to the target.\n");
        fprintf(stderr, "  --imatrix <f>  use an importance-matrix GGUF to steer k-quant/IQ\n");
        fprintf(stderr, "                 precision (improves quality, esp. for iq4_* / low-bit).\n");
        fprintf(stderr, "  --tensor-type <regex>=<type>  per-tensor precision override (repeatable,\n");
        fprintf(stderr, "                 first match wins, overrides the built-in arch guards).\n");
        fprintf(stderr, "                 e.g. --tensor-type 'output\\.weight=q8_0' --tensor-type '\\.ffn=q6_k'\n");
        ggml_print_ftypes(stderr);
        return 1;
    }

    const std::string fname_inp = pos[0];
    const std::string fname_out = pos[1];
    const ggml_ftype ftype = ggml_parse_ftype(pos[2].c_str());

    if (!imatrix_path.empty()) {
        crispasr_load_imatrix(imatrix_path); // non-fatal: falls back to unweighted if empty
    }

    if (!crispasr_model_quantize(fname_inp, fname_out, ftype)) {
        fprintf(stderr, "failed to quantize model\n");
        return 1;
    }

    return 0;
}
