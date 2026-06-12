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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <cmath>

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

    // Bark TTS: 3 GPT-2 sub-models + EnCodec decoder.
    // Embeddings (token_embd, pos_embd), output heads, and the entire
    // EnCodec decoder are read via CPU tensor_get_row_f32 / tensor_get_all_f32
    // and are precision-sensitive. Only attn/ffn projection weights
    // (attn_qkv, attn_output, ffn_up, ffn_down) should be quantized.
    // Verified: Q4_K of all tensors produces near-zero audio (peak 0.001);
    // Q8_0 works fine; selective Q4_K (projections only) is safe.
    const bool is_bark = (arch.find("bark") != std::string::npos);

    // First pass: determine which tensors will be quantized and compute
    // their target types. We need this BEFORE adding tensors to ctx_out
    // so that gguf_add_tensor computes correct offsets for the quantized
    // sizes.
    const int n_tensors = gguf_get_n_tensors(ctx_in);
    std::vector<ggml_type> target_types(n_tensors);

    // Allocate a scratch ggml context for creating modified tensor descriptors.
    ggml_init_params scratch_params = {ggml_tensor_overhead() * (size_t)n_tensors + 1024, nullptr, true};
    ggml_context* ctx_scratch = ggml_init(scratch_params);

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor* t = ggml_get_tensor(ctx_in_ggml, name);

        std::string sname(name);
        bool is_weight = (sname.find("weight") != std::string::npos) ||
                         (sname.size() >= 2 && sname.substr(sname.size() - 2) == "_w") ||
                         (sname.size() >= 2 && sname.substr(sname.size() - 2) == ".w") ||
                         (sname.find("_proj") != std::string::npos) || (sname.find(".gate") != std::string::npos) ||
                         (sname.find(".up") != std::string::npos) || (sname.find(".wo") != std::string::npos) ||
                         (sname.find(".heads.") != std::string::npos);
        const bool ok_dims = (ggml_n_dims(t) == 2) || ((is_firered || is_ecapa) && ggml_n_dims(t) >= 2);
        const int64_t ncols = t->ne[0];

        bool should_quantize =
            ggml_is_quantized(qtype) && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && ok_dims &&
            is_weight && (sname.find("norm") == std::string::npos) && (granite_quant_all || sname.find("proj.") != 0) &&
            !(is_granite_family && !granite_quant_all && sname.find("enc.") == 0) &&
            // MOSS-Audio: keep encoder + adapter + deepstack at F16
            !(arch == "moss_audio" &&
              (sname.find("enc.") == 0 || sname.find("adapter.") == 0 || sname.find("deepstack.") == 0)) &&
            !(sname.find("cls.") == 0 && ggml_nelements(t) < 65536) && (sname.find("enc_proj.") != 0) &&
            (sname.find("lm_head.") != 0) && (sname.find("tok_emb.") != 0) && (sname.find("lang_emb.") != 0) &&
            !(is_chatterbox && (sname.find("s3.v.") == 0 || sname.find("conds.") == 0 || sname.find("ve.") == 0 ||
                                sname.find("t3.text_emb") == 0 || sname.find("t3.speech_emb") == 0 ||
                                sname.find("t3.wpe") == 0 || sname.find("t3.text_pos_emb") == 0 ||
                                sname.find("t3.speech_pos_emb") == 0 || sname.find("t3.cond.") == 0)) &&
            !(is_cosyvoice3 &&
              (sname == "cosyvoice3.speech_embd.weight" || sname == "cosyvoice3.speech_lm_head.weight" ||
               sname == "cosyvoice3.flow.input_embd.w" || sname == "cosyvoice3.flow.spk_affine.w" ||
               sname == "cosyvoice3.s3tok.fsq.proj.w")) &&
            !is_f5tts &&
            !(is_qwen3_tts && (sname.find("speaker.") == 0 || sname.find("code_pred.token_embd") == 0 ||
                               sname.find("code_pred.output") == 0 || sname.find("code_pred.small_to_mtp") == 0 ||
                               sname.find("talker.token_embd") == 0 || sname.find("talker.text_proj") == 0 ||
                               sname.find("talker.codec_bridge") == 0)) &&
            !(is_parler && sname.find("dac.") == 0) &&
            !(is_dia && (sname.find("embedding") != std::string::npos || sname.find("audio_encoder") == 0)) &&
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
            !(is_mini_omni2 && (sname.find("audio.") == 0 || sname.find("adapter.") == 0 ||
                                sname.find("llm.token_embd") == 0)) &&
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

        // Create a tensor descriptor with the target type for ctx_out
        if (target_types[i] != t->type) {
            struct ggml_tensor* t_out = ggml_new_tensor(ctx_scratch, target_types[i], ggml_n_dims(t), t->ne);
            ggml_set_name(t_out, name);
            gguf_add_tensor(ctx_out, t_out);
        } else {
            gguf_add_tensor(ctx_out, t);
        }
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
            } else {
                std::vector<ggml_fp16_t> f16_data(nelements);
                if (fread(f16_data.data(), sizeof(ggml_fp16_t), nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f16 data\n");
                    return false;
                }
                for (int j = 0; j < nelements; j++)
                    f32_data[j] = ggml_fp16_to_fp32(f16_data[j]);
            }

            const size_t max_q_size = ggml_row_size(qtype_used, t->ne[0]) * (nelements / t->ne[0]);
            q_data.resize(max_q_size);

            size_t q_size = ggml_quantize_chunk(qtype_used, f32_data.data(), q_data.data(), 0, nelements / t->ne[0],
                                                t->ne[0], nullptr);

            fwrite(q_data.data(), 1, q_size, fout);

            // Padding
            size_t pad = GGML_PAD(q_size, GGUF_DEFAULT_ALIGNMENT) - q_size;
            for (size_t j = 0; j < pad; j++)
                fputc(0, fout);

            printf("done\n");
        } else if (target_types[i] == GGML_TYPE_F16 && type == GGML_TYPE_F32) {
            // Granite encoder F32 → F16 downcast (or any F32→F16 target)
            printf("F32 -> F16... ");
            const int64_t nelements = ggml_nelements(t);
            std::vector<float> f32(nelements);
            if (fread(f32.data(), sizeof(float), nelements, fin) != (size_t)nelements) {
                fprintf(stderr, "failed to read f32 data\n");
                return false;
            }
            std::vector<ggml_fp16_t> f16(nelements);
            for (int64_t j = 0; j < nelements; j++)
                f16[j] = ggml_fp32_to_fp16(f32[j]);
            const size_t out_bytes = (size_t)nelements * sizeof(ggml_fp16_t);
            fwrite(f16.data(), 1, out_bytes, fout);
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
    if (argc != 4) {
        fprintf(stderr, "usage: %s model-f16.gguf model-quant.gguf type\n", argv[0]);
        ggml_print_ftypes(stderr);
        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];
    const ggml_ftype ftype = ggml_parse_ftype(argv[3]);

    if (!crispasr_model_quantize(fname_inp, fname_out, ftype)) {
        fprintf(stderr, "failed to quantize model\n");
        return 1;
    }

    return 0;
}
