// hojo_asr.h — public C API for the HojoAI/Hojo-ASR-Multi-V1 ggml runtime.
//
// Multilingual ASR (de/fr/it/pt/es + more) with the classic
// Encoder → Adapter → LLM layout. Models are produced by:
//   `python models/convert-hojo-asr-to-gguf.py --input HojoAI/Hojo-ASR-Multi-V1 \
//                                             --output hojo-asr-multi-v1-f16.gguf`
//
// Architecture (HojoAI/Hojo-ASR-Multi-V1, Apache-2.0). Every field below was
// read out of the `hojo-asr` PyPI package, which is the driving inference code
// — NOT reconstructed from the config:
//
//   Front end  WhisperFeatureExtractor(feature_size=128, n_fft=400, hop=160)
//              with padding=False → a VARIABLE-length 128-bin log-mel at
//              100 fps. `chunk_length=40` only moves the unused
//              padding="max_length" cap; nothing is padded to 30 s or 40 s.
//
//   Encoder    The STOCK Qwen3-Omni "AuT" audio tower (the same one
//              moss_transcribe.cpp runs) with n_window=1500 and
//              n_window_infer=3000 patched in at construction:
//              3× Conv2d(3×3, stride 2, pad 1, 480 ch, GELU) → conv_out
//              Linear(480·16=7680 → 1280, no bias) → + sinusoidal position
//              (restarting at 0 in every chunk) → 32 pre-LN layers
//              (1280 d, 20 heads, FFN 5120, GELU) → ln_post → proj1 → gelu →
//              proj2 (→ 2048).  Mel is cut into 3000-frame chunks and
//              attention is block-diagonal over them, so each chunk is
//              mathematically independent.  8× time downsample → 12.5 fps.
//
//   Adapter    WeNet ConformerEncoder(2048 → 2560, linear_units=640,
//              num_blocks=2, input_layer="linear"), everything else at WeNet
//              defaults: 4 heads, rel_pos, macaron (ff_scale 0.5), SiLU,
//              cnn kernel 15, non-causal, BatchNorm, pre-LN, eps 1e-5.
//              ⚠ WeNet DELETES `rel_shift`, so the positional term is
//              matrix_bd = (q + pos_bias_v)·(W_pos · pe[0:T])ᵀ with no shift —
//              an absolute-position bias, not Transformer-XL relative.
//              ⚠ The adapter is 1:1 in time. `linear_units: 640` is the
//              conformer FFN's inner width, NOT a frame-stacking factor; the
//              "multi-frame acoustic fusion" the model card names is the
//              encoder's conv stem (8 mel frames + all 128 mel bins fused into
//              one 1280-d frame by conv_out).
//
//   ln_speech  LayerNorm(2560) on the adapter output.
//
//   Decoder    Qwen3-4B-Instruct-2507 (36 L, 2560 d, 32 Q / 8 KV heads,
//              head_dim 128, SwiGLU 9728, QK-norm, RoPE θ=5e6, RMS eps 1e-6),
//              embeddings resized to 151670 (Qwen3's 151669 + an added [PAD]),
//              tied lm_head.
//              inputs_embeds = [embed(<|im_start|>)] ++ speech_embeddings.
//              There is NO text prompt, no chat template and no audio
//              placeholder token — that is the whole conditioning.
//
// Decode recipe from config.yaml `generate:` (passed verbatim to
// `generate()` by HOJO_ASR.infer): num_beams=4, do_sample=False,
// repetition_penalty=2.0, length_penalty=1.0, and
// max_new_tokens = max(10, min(200, T_enc*2 + 10)).
//
// ⚠ ONE KNOWN DIVERGENCE FROM `generate()`, recorded so a transcript diff is
// not misread as a port bug: `core_beam_decode` ranks beams by raw cumulative
// log-probability, while transformers divides by `length ** length_penalty`
// when finalising. With length_penalty=1.0 and beams expanded in lockstep the
// two orderings are identical — every live beam has the same length, so the
// division is a monotone transform. They can differ only when a beam finishes
// EARLY: transformers rewards the shorter finished hypothesis, this helper
// carries its raw score forward. So a transcript mismatch confined to an
// utterance where one beam hit <|im_end|> well before the others is this, not
// the encoder. Greedy (`-bs 1`) has no such divergence and is the cleaner arm
// for localising a parity failure.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct hojo_asr_context;

struct hojo_asr_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    bool flash_attn;
};

struct hojo_asr_context_params hojo_asr_context_default_params(void);

// Load model from GGUF.
struct hojo_asr_context* hojo_asr_init_from_file(const char* path_model, struct hojo_asr_context_params params);

void hojo_asr_free(struct hojo_asr_context* ctx);

// Transcribe raw 16 kHz mono PCM. Returns malloc'd UTF-8 string (caller owns).
char* hojo_asr_transcribe(struct hojo_asr_context* ctx, const float* samples, int n_samples);

// Per-token streaming callback. Fires once per generated token (id, prob, userdata).
typedef void (*hojo_asr_token_cb)(int tok_id, float prob, void* userdata);

// Like hojo_asr_transcribe() but fires cb(tok_id, prob, userdata) per token.
// The assembled text is NOT returned; output is via cb.
void hojo_asr_transcribe_cb(struct hojo_asr_context* ctx, const float* samples, int n_samples, hojo_asr_token_cb cb,
                            void* userdata);

// Beam width. <= 0 means GREEDY (see the cost note below); pass 4 for the
// checkpoint's own `generate.num_beams` recipe.
//
// ⚠ COST: `core_beam_decode` rebuilds each beam's KV by replaying its entire
// generated suffix every step, so beam search is O(B*T^2) token-forwards where
// greedy is O(T). On this 4.4 B decoder and a 9-second clip that is 80,400
// forwards versus 200 — roughly four hours versus three minutes on CPU. That
// is why greedy is the default even though the checkpoint's recipe is beam 4:
// a default nobody can afford to run is not faithfulness, it is a hang.
// The runtime prints the projected forward count whenever beam > 1.
// The real fix is `core_beam_decode::run_with_probs_branched` (per-beam KV
// snapshots, O(B*T)), which needs `hojo_asr_kv_save`/`kv_restore` — not yet
// implemented, and the reason beam 4 is opt-in rather than removed.
void hojo_asr_set_beam_size(struct hojo_asr_context* ctx, int beam_size);

// #292: forward --max-new-tokens. <= 0 keeps the checkpoint's own cap.
void hojo_asr_set_max_new_tokens(struct hojo_asr_context* ctx, int max_new_tokens);

// ---- Stage helpers for differential testing (crispasr-diff) ----

// 128-bin Whisper log-mel (n_fft 400, hop 160, padding=False).
// Output: malloc'd (n_mels, T_mel) F32 row-major. Caller frees.
float* hojo_asr_compute_mel(struct hojo_asr_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                            int* out_T_mel);

// Full audio encoder (conv stem + 32 layers + ln_post + proj1/proj2).
// Returns (T_enc, output_dim=2048) F32 row-major. Caller frees.
float* hojo_asr_run_encoder(struct hojo_asr_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                            int* out_d);

// Conformer adapter + ln_speech on the encoder output.
// Returns (T_enc, llm_dim=2560) F32 row-major — the LM's speech embeddings.
// `out_pre_ln_speech` (optional) additionally receives the bottleneck output
// BEFORE ln_speech, so a parity run can tell a broken conformer apart from a
// broken final LayerNorm without a second forward pass. Both are caller-freed.
float* hojo_asr_run_adapter(struct hojo_asr_context* ctx, const float* encoder_out, int T_enc, int d_enc, int* out_T,
                            int* out_d, float** out_pre_ln_speech);

// Embed tokens via the LM token table. Returns (n_tokens, llm_dim) F32.
float* hojo_asr_embed_tokens(struct hojo_asr_context* ctx, const int32_t* token_ids, int n_tokens);

// KV cache for LLM decode.
bool hojo_asr_kv_init(struct hojo_asr_context* ctx, int max_ctx);
void hojo_asr_kv_reset(struct hojo_asr_context* ctx);

// Run LLM with KV cache. Returns last-token logits (vocab_size,) F32.
float* hojo_asr_run_llm_kv(struct hojo_asr_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                           int* out_n_tokens, int* out_vocab_size);

// Tokenize text using GPT-2 byte-level BPE. Returns count written to out_tokens.
int hojo_asr_tokenize(struct hojo_asr_context* ctx, const char* text, int32_t* out_tokens, int max_tokens);

// Token ID → string (GPT-2 byte-encoded form).
const char* hojo_asr_token_text(struct hojo_asr_context* ctx, int token_id);

// The single BOS token the LM is primed with (<|im_start|>, 151644).
int hojo_asr_bos_token_id(struct hojo_asr_context* ctx);

// Reference cap: max(10, min(gen.max_new_tokens, T_enc*2 + 10)).
int hojo_asr_max_new_for_frames(struct hojo_asr_context* ctx, int T_enc);

#ifdef __cplusplus
}
#endif
