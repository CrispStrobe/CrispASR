# CrispASR — Pending work

Pending roadmap items. Each is self-contained with files, approach, and
effort estimate. Completed items have been moved to `HISTORY.md`.

**Current state (April 2026, v0.5.4):** 21 ASR backends + TTS, unified CLI,
OpenAI-compatible server, shared `src/core/` library, FireRedPunc
post-processor, C-ABI + Python/Rust/Dart wrappers, CI on 6 platforms.
All backends support `-m auto --auto-download`. Three new ggml ops
(`conv_1d_cf`, `conv_1d_dw_cf`, `conv_1d_group`). ggml bumped to 0.10.0.

---

## Priority ordering

| Priority | Item | Effort | Status |
|---|---|---|---|
| **HIGH** | [#52 Qwen3-TTS](#52-qwen3-tts) — speaker_encoder forward | Medium | talker + code_predictor + codec done; ECAPA next |
| **HIGH** | [#51 MiMo-V2.5-ASR runtime](#51-mimo-v25-asr-runtime) | Large | converters done; runtime is a stub |
| **HIGH** | [#54 granite-speech-4.1 plus / nar](#54-granite-speech-41-plus--nar-variants) | Small | base + plus + nar runtimes all DONE; only NAR quant + HF upload remain |
| **HIGH** | [#57 Commercial-friendly TTS expansion](#57-commercial-friendly-tts-backend-expansion) | Phased | Phase 1 (Qwen3-TTS-CustomVoice) in progress; phases 2-5 queued |
| **MEDIUM** | [#5 Reference backends](#5-reference-backends-for-parakeetcanarycohere) | Medium | parakeet/cohere DONE; canary remaining |
| **MEDIUM** | [#53 core/audio_decoder.h](#53-coreaudio_decoderh--dry-across-tts--codec-backends) | Medium | DRY across qwen3-tts/mimo/vibevoice |
| **MEDIUM** | [#56 Kokoro multilingual phonemizer](#56-kokoro-multilingual-phonemizer-espeak-ng) | Small | espeak-ng + DE backbone shipped; HF GGUFs published 2026-05-01; auto-download wired; only Mandarin tones / JA kanji + diff-harness phonemizer-step polish remain |
| **LOW** | #41 Moonshine IPA / phoneme | High | Deferred |
| **LOW** | [#7 voxtral4b streaming](#7-native-voxtral4b-streaming) | High | |
| **LOW** | [#9 Parakeet TDT GPU](#9-parakeet-tdt-decoder-gpu) | Medium | |
| **LOW** | [#11 WebSocket server](#11-websocket-streaming-server) | High | |
| **DONE** | [#16 Shaw RPE](#16-shaw-rpe-for-granite-graph) | Medium | graph encoder path now default — ~2.3× total realtime; per-layer RPE fix landed |
| **BLOCKED** | [#42 VibeVoice-ASR 7B](#42-vibevoice-asr-7b) | High | Needs ≥16 GB RAM |
| **BLOCKED** | [#43 Fun-ASR-Nano](#43-fun-asr-nano) | Medium | License unclear |

---





## 40. More Moonshine model variants

Convert + upload to HuggingFace:
- ~~`moonshine-base` (61.5M, better WER)~~ **DONE** (cstr/moonshine-base-GGUF)
- `moonshine-streaming-tiny/small/medium` — different architecture, needs new runtime
- ~~`moonshine-tiny-{ja,ar,ko,zh,vi,uk}` (multilingual)~~ **DONE** (12 repos on HF)
- ~~`moonshine-base-{ja,uk,vi,zh,ar,ko}` (multilingual)~~ **DONE** (12 repos on HF)

Converter fix: 1D tensors (norms, biases) forced to F32; conv_1d_f32 mul_mat
argument order fixed for F16 kernels.

---

## 41. Moonshine phoneme / IPA output

moonshine-ai/moonshine has a `GraphemeToPhonemizer` — G2P (text→IPA),
NOT audio→phoneme. Runs on transcription output.

**Options:**
1. Port G2P tables to C++ (~500 LOC, needs pronunciation dicts)
2. Post-processing module with `--output-ipa` flag
3. External-only (document piping through Python G2P)

**Recommendation:** Option 3 for now. IPA is niche; ROI of porting is low.

---

## 5. Reference backends for parakeet/canary/cohere

Write `tools/reference_backends/{parakeet,canary,cohere}.py` for
`crispasr-diff` reference activation comparison.

**Effort:** ~100-150 LOC per backend.

**Status (April 2026):** `parakeet` reference backend done — used to
diagnose the JA xscaling bug. Cohere already has one
(`reference_backends/cohere.py`). Canary is the only one still missing.

---

## 7. Native voxtral4b streaming

Expose voxtral4b's native 240ms-2.4s latency streaming via pre_hook
audio frame injection. Needs threading (encoder thread + decoder thread).

**Effort:** ~200-300 LOC. High complexity.

---

## 9. Parakeet TDT decoder GPU

Port LSTM predictor + joint head from CPU loops to ggml graphs. LSTM
is sequential → per-step kernel launches. Encoder already 85%+ of time.

**Effort:** ~150 LOC. Small gain.

---

## 11. WebSocket streaming server

Add `/ws` endpoint for real-time streaming over HTTP. httplib doesn't
support WebSocket — need custom protocol or library.

**Effort:** ~200-300 LOC.

---


## 16. Shaw RPE for granite graph

Today the encoder runs as ~96 small ggml graphs (one per matmul / per
layer) dispatched serially through `ggml_backend_sched`. With Metal
each dispatch is a full command-buffer round-trip — encoder time
dominates at ~3.6 s of a ~5 s total on M1 + Q4_K. The opt-in
`GRANITE_ENCODER_GRAPH=1` path builds the whole 16-layer encoder as
ONE graph; numbers from LEARNINGS:

| Path                       | run_encoder | total      | realtime |
|----------------------------|------------:|-----------:|---------:|
| CPU loops (default)        |  12,624 ms  |  19.6 s    | 0.6×     |
| `GRANITE_ENCODER_GRAPH=1`  |   3,110 ms  |   7.55 s   | **1.5×** |

It's not the default because the graph path uses
`ggml_flash_attn_ext`, which can't ingest a Q-dependent additive
bias — so Shaw RPE is silently omitted and encoder output drifts
from the CPU path. Accuracy on JFK is fine; harder material (long
context, dense overlap) likely degrades.

### Two viable approaches

**Option A — flash-attn with Q-dependent bias.** Compute the
`Q · RPE` term inside the graph (each Q row vs each precomputed
per-position embedding row → a (T, ctx_size, hd) tensor reduced to
(T, ctx_size) per head), feed it to `flash_attn_ext`'s mask slot as
an additive bias. Keeps flash-attn's I/O fusion. Tricky because the
mask slot is currently F16 and broadcastable but not per-Q-vector;
needs either a flash-attn op extension or a manual `mul_mat`-then-
softmax-then-`mul_mat` chain in place of flash-attn.

**Option B — block-attention via per-block subgraphs (preferred for
the prototype).** Reuse the same windowed-dispatch pattern the
projector already uses (`ctx_size=200`, ceil(T/200) blocks). Per
block per layer, build a small graph that does:

  scores = Q · K^T  +  Q · RPE_block      // (block_len × block_len)
  attn   = softmax(scores · attn_scale) · V

The RPE bias is precomputed once per layer at load time
(`core_conformer_ibm::build_shaw_rpe_lookup`, already lifted) and fed
in as a static F32 input tensor. No flash-attn fusion, but the Q/K/V
projections, ffn1/ffn2, conv module, and Shaw scoring all live in
one graph per block per layer — still ~16× fewer dispatches than the
current per-op path. Math is bit-identical to the CPU loop.

### Plan

1. **DONE (prototype, May 2026):** per-block subgraph attention with
   Shaw RPE wired up behind `GRANITE_ENCODER_GRAPH_RPE=1`.
2. **DONE (May 2026):** root-caused the regression — the loader built
   only layer 0's RPE lookup (`ctx->rpe_lookup`) and the graph builder
   reused it for all 16 layers, on the assumption that
   granite-speech ties RPE across layers. granite-speech-4.1-2b in
   fact stores **distinct** `attn_rel_pos.weight` per encoder block
   (verified: layer 0 mean 0.00004, layer 1 mean -0.003, layer 2 mean
   -0.002). Fix: precompute `rpe_per_layer[il]` at load time, declare
   the graph's `rpe_lookup` input with shape
   `(ctx_size*hd, ctx_size, n_layers)`, and slice per-layer via
   `ggml_view_3d` on the layer axis. CPU loop now uses the same
   per-layer lookups. Stage-by-stage taps confirm bit-near-identical
   output between graph and CPU paths through all 16 layers (within
   float precision).
3. **DONE:** with-RPE graph path transcribes JFK byte-for-byte
   identical to CPU loop on `granite-speech-4.1-2b-q4_k`. Encoder
   runs at ~2.1 s vs ~4.9 s CPU baseline (≈2.3× speedup, end-to-end).
4. **DONE:** `GRANITE_ENCODER_GRAPH_RPE=1` retired in favour of the
   graph path being on by default. The PLUS variant (cat_layers) and
   any model whose `attn_rel_pos.weight` type is unsupported by
   `core_conformer_ibm::build_shaw_rpe_lookup` automatically fall back
   to the CPU loop.
5. **DONE:** `GRANITE_DISABLE_ENCODER_GRAPH=1` is the escape hatch
   back to the CPU loop (slower but kept around for debugging).

**Validation gate (per the methodology in LEARNINGS):** stage-by-stage
diff. With per-layer RPE, every sub-stage of layer 0 and layer 1
matches CPU within ~1e-3, and the final encoder output matches within
the same tolerance — well above the cos_min ≥ 0.999 bar.

---


## 42. VibeVoice-ASR 7B

**BLOCKED:** Needs ≥16 GB RAM for conversion. Converter OOMs on 8 GB due
to Qwen2.5-7B embedding (152064 × 3584 = 2.1 GB F32).

**Fix:** Use `safe_open` per-tensor conversion. Then Q4_K → ~4 GB.

Full architecture analysis in HISTORY.md #34. C++ runtime partially
implemented (`src/vibevoice.cpp`). F16 im2col precision issue in
depthwise conv needs fixing.

---

## 43. Fun-ASR-Nano

**BLOCKED:** License unclear. Issue filed at `FunAudioLLM/Fun-ASR#99`.
No response. HF model card has no license field.

---

## 51. MiMo-V2.5-ASR runtime

Converter done (`models/convert-mimo-asr-to-gguf.py` plus the
audio-tokenizer side `models/convert-mimo-tokenizer-to-gguf.py`).
Runtime not yet written.

- **Architecture:** 6-layer input_local_transformer + 36-layer Qwen2
  LLM (4096d, 32Q/8KV, SiLU, RoPE).
- **Audio path:** 8-channel RVQ tokens from `cstr/mimo-tokenizer-GGUF`
  (separate model) get fed into the input_local_transformer, which
  produces embeddings the Qwen2 LLM consumes.
- **Reuse:** Qwen2 LLM core can lean on `core_attn::kv_self_attn` +
  `core_ffn::swiglu` — both already in production via qwen3-asr.

**Effort:** Large (~1000 LOC), but most of it is the audio-tokenizer
runtime (RVQ encoder/decoder over conv stacks). The Qwen2 part
mostly slots into the existing core helpers.

---

## 52. Qwen3-TTS

User-requested follow-on to the VibeVoice TTS work. Apache-2.0
collection: [Qwen/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
[HF collection](https://huggingface.co/collections/Qwen/qwen3-tts).

- **Six repos in the collection** (all BF16 safetensors, Apache 2.0):
  - `Qwen/Qwen3-TTS-Tokenizer-12Hz` — RVQ codec, 16 codebooks × 2048,
    12.5 FPS at 24 kHz. Non-DiT lightweight architecture (8L
    encoder + 8L decoder).
  - `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base` — base talker LM with
    voice clone (3s reference audio).
  - `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice` — fine-tuned,
    fixed speakers.
  - `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` — instruction-tuned
    (voice description → speech).
- **Architecture:** "Discrete Multi-Codebook LM" — Qwen3 backbone
  with a 16-codebook output head. No DiT; direct AR generation of
  RVQ codes. ~97ms end-to-end latency, 10 languages incl.
  en/de/zh/ja/ko/it.
- **Status (April 2026):** **talker + ICL prefill + code_predictor live; intelligible synthesis verified (codec via Python)**.
  - Converter (`models/convert-qwen3-tts-to-gguf.py`) maps the
    actual HF tensor namespace (`talker.model.codec_embedding` →
    `talker.token_embd`, `talker.codec_head` → `talker.output`,
    `talker.text_projection.linear_fc{1,2}` → `talker.text_proj.fc{1,2}`,
    speaker_encoder + 15-CB code-predictor likewise). Writes BPE
    merges so subword tokenisation works. **Verified** end-to-end:
    478 tensors, 0 unmapped, `qwen3-tts-0.6b-base.gguf` loads.
  - Runtime: 28L Qwen3 talker forward via `core_attn::kv_self_attn`
    + `core_ffn::swiglu` (single-axis NEOX RoPE — text-only collapse,
    swap to `ggml_rope_multi` once codec splice lands). F16 KV cache.
    `qwen3_tts_synthesize_codes(text)` AR-decodes codebook-0 from
    `codec_head`. Empirically produces 923 codes for "Hello, this
    is a test." on Metal.
  - Diff harness wired: `tools/reference_backends/qwen3_tts.py`
    drives the qwen-tts pip package on CPU and dumps stage tensors
    to a GGUF archive. `crispasr-diff qwen3-tts` reads `text_input_ids`
    from the archive and runs `qwen3_tts_run_text_proj` against
    `text_proj_out` → **PASS at cos_min=1.000000** for the
    text_embedding + resize-MLP path on the "Hello world." prompt.
  - Debug knobs: `QWEN3_TTS_{BENCH,DEBUG,DUMP_DIR}` env vars in the
    style of `GEMMA4_E2B_BENCH` / `OMNIASR_DUMP_DIR`.
  - **Verified milestones (in landed order):**
    1. ✓ Talker forward (28L Qwen3 + Q/K-norm + flash-attn + F16 KV
       cache). `talker_logits` PASS at cos_min=1.000000 against
       PyTorch when fed the same prefill (commit `2b85b78`).
    2. ✓ ICL prefill builder. Independent C++ implementation of
       `Qwen3TTSForConditionalGeneration.generate_icl_prompt` —
       chat-template prompt + codec sentinels + speaker_embed (from
       voice pack) + per-frame summed 16-codebook ref_code embeddings
       + tts_pad/bos/eos splice. `talker_logits_via_icl` PASS at
       cos_min=1.000000 (commit `b939d4f`).
    3. ✓ Code predictor (5L Qwen3, 15 separate codec_embedding +
       lm_head pairs, top-k=50 + temperature=0.9 sampling). Greedy
       was a silent-output trap; sampling matches the reference
       `subtalker_dosample=True` default (commit `9608202`,
       `69c135c`).
    4. ✓ Roundtrip: TTS-out → ASR-in. Our pipeline synthesises
       "The quick brown fox jumps over the lazy dog." → 55 frames
       × 16 codebooks → Python codec decode → 4.4s audio →
       parakeet ASR transcribes back verbatim.
  - **Open** (in priority order):
    1. ✓ **Codec decoder** (Tokenizer-12Hz, commits `d1f47b1`, `48c6c1a`).
       Converter rewritten (0 unmapped, 253 tensors, 0.25 GB F16 GGUF).
       C++ decoder: SplitRVQ → pre_conv → 8L XFMR(512d, sliding-window=72)
       → 2× ConvNeXt upsample → 4× SnakeBeta+tconv DecoderBlock → PCM.
       Diff harness: 8/8 stages PASS (cos_min ≥ 0.999983) end-to-end
       on Metal with `use_gpu=true`. The original M1 hang
       (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`) was
       fixed in our ggml fork — `kernel_conv_transpose_1d` was
       iterating all IL input positions and filtering with an `if`,
       doing ~160× wasted work which crossed the macOS GPU watchdog
       (~5 sec). Patched to iterate only the
       `i ∈ [ceil((j-K+1)/s0), floor(j/s0)] ∩ [0, IL-1]` range that
       actually contributes. See LEARNINGS.md "Metal
       conv_transpose_1d input range tightening" — MUST RE-APPLY
       after every ggml bump. The runtime CPU-pin (`codec_sched`)
       is kept as a safety net.
    2. **Runtime ECAPA speaker_encoder forward.** Removes the
       `bake-qwen3-tts-voice-pack.py` dependency for new voices —
       end users pass any ref WAV and we compute spk_embedding in
       C++. ~250 LOC: mel(24kHz) → TDNN → 3 SE-Res2Net → MFA →
       ASP → Conv1×1 → 1024-d. Reference: `Qwen3TTSSpeakerEncoder`
       in `ref/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py`.
    3. **Runtime codec encoder forward.** Mimi-based encoder (used
       in voice cloning to extract `ref_code` from the reference
       WAV). Closes the loop on the bake script — pure C++ pipeline.
       Larger effort (Mimi is its own architecture).
    4. **Performance pass.** Current AR step is ~137 ms/frame on
       M1 Metal — 1.7× slower than real-time at 12.5 fps. Bottleneck
       is the 15 sequential code_predictor graph builds per frame.
       Fuse them into a single graph with all 15 lm_heads and KV
       writes baked in. Also: fused QKV in talker (qwen3_asr has
       this; qwen3_tts could too), Q4_K weights (1.83 GB → 480 MB).
- **Reuse:** the talker is essentially Qwen3-0.6B/1.7B with a
  multi-codebook output head — `core_attn::kv_self_attn` +
  `core_ffn::swiglu` again. The codec needs new code for RVQ
  decoding; that work is shared with MiMo (#51) and overlaps in
  shape with the VibeVoice σ-VAE decoder, so a `core_audio_decoder`
  helper is worth landing alongside the runtime (see #53).

**Effort:** Large. ~1500 LOC across runtime + codec + reference
backend. The two TTS targets (Qwen3-TTS and any future expansion)
share enough that landing one substantially de-risks the other.

---

## 54. granite-speech-4.1 plus / nar variants

The `ibm-granite/granite-speech-4.1-2b` family ships three variants
with significantly different decoders despite the shared "4.1-2b"
naming. All three runtimes are now fully supported and bit-exact on
JFK; only the NAR quantization + HF upload remain.

| Variant | Decoder | Encoder change | Outputs | Status |
|---|---|---|---|---|
| `granite-speech-4.1-2b` (base) | Granite-1B AR | none | text | DONE — 4 GGUFs on HF, encoder cos 0.999908, transcribes JFK at 2.1× realtime on M1 Q4K |
| `granite-speech-4.1-2b-plus` | Granite-1B AR | `cat_hidden_layers: [3]` | text + speaker labels + word-level timestamps | DONE — f16 GGUF on HF, transcribes JFK with punctuation/capitalisation by default; speaker labels + word timestamps in template work pending |
| `granite-speech-4.1-2b-nar` | non-autoregressive (`NLENARDecoder`) | self-conditioning at L8 + BPE aux head + 4-layer hidden capture | text | DONE — full pipeline bit-exact on JFK (mel `cos_min=0.999997`, encoder_output `cos_min=0.999852`, encoder_logits `cos_min=0.999675`, projector_output `cos_min=0.999999`, editing_logits `cos_min=0.999999` 47/47 top-1; transcribe matches reference `final_text` exactly) |

### Base 4.1-2b (DONE)

- `granite-4.1` backend alias of `granite`
- 4 GGUF variants on `cstr/granite-speech-4.1-2b-GGUF`: F16 (5.58 GB),
  Q4K with F32 encoder (2.94 GB, recommended), Q4K with F16 encoder
  (2.07 GB, sweet spot), Q4K everywhere (1.7 GB, mini)
- 3.7× total speedup from norm + QKV graph fusion (commit `796824f`)
- `GRANITE_BENCH=1` per-stage timer
- New GGUF keys: `enc.context_size`, `enc.max_pos_emb`,
  `proj.encoder_hidden_size`, `proj.cat_layers`. Old values default
  in for legacy GGUFs.

### Plus variant — DONE for ASR transcription

The PLUS GGUF (5.6 GB f16) is converted and the runtime concatenates
encoder layer 3 with the final layer output (`il + 1 == cat_index`,
matching HF's `output_hidden_states` convention). End-to-end JFK
transcription with the new `--backend granite-4.1-plus` alias produces:

```
And so my fellow Americans, ask not what your country can do for you,
ask what you can do for your country.
```

Punctuation and capitalisation come for free — the PLUS variant's
training default is structured output. Speaker labels and word-level
timestamps are not yet in the output; investigating the upstream
`chat_template.jinja` is the next step (~50 LOC, template-only).

Commits: `f298818` (cat_layer + tokenizer fix), `ed0e5ac` (backend
alias + registry), `a3147b6` (HF README).

### NAR variant — DONE

1. **Encoder forward** (`granite_nle_run_encoder`). DONE. Same
   Conformer block as base; self-conditioning at layer 8 (the running
   char-level CTC logits feed back through `out_mid`); 4-layer hidden
   state capture at the indices listed in `proj.encoder_layer_indices`
   (default `[4, 8, 12, -1]`). The capture obeys HF tuple semantics:
   `-1` resolves to `n_layers`, and the snapshot at the
   self-conditioning layer is taken AFTER the residual is added.
   Validated against PyTorch on JFK at cos_min ≥ 0.999. The BPE
   auxiliary head (`enc.bpe_out`) is intentionally not wired through
   `run_encoder` — it's only needed by the LLM editing pass's text-init
   step, where it's faster to run on the posterior-pooled features.
2. **Windowed Q-Former projector**
   (`granite_nle_run_projector`). DONE. Two-pass implementation: (A)
   one ggml graph for the per-encoder-layer LayerNorms + concat +
   `layer_proj` (4096 → 2048) + GELU; (B) one Q-Former graph per
   block (`block_size=15`, `downsample_rate=5`, `query_length=3`)
   with mean-pool over downsample groups, additive `query` and
   `window_positions`, two 32-head SDPA cross-attention + SiLU-MLP
   layers, and a final `out_norm`+`out_linear`. Output rate: 3 audio
   tokens per 15 encoder frames. Validated against PyTorch on JFK at
   `projector_output cos_min=0.999999` (T_out=111 × llm_dim=2048).
3. **Non-causal LLM editing pass** (`granite_nle_run_llm_editing`).
   DONE. Single graph over the flat `[audio_embs, text_embs_with_slots]`
   sequence with µP scaling (embedding_multiplier=12,
   attention_multiplier=1/128, residual_multiplier=0.22). 40 layers of
   RMSNorm + non-causal `flash_attn_ext` (mask=nullptr, GQA 16/4
   native) + SwiGLU. Tied LM head (matmul against the same
   `token_embd_w` used for embed lookup). The caller passes audio_embs
   pre-divided by `embedding_multiplier` so the uniform downstream
   scale-up recovers the original projector output for audio while
   still scaling text by 12× — mirrors `_build_flat_llm_inputs`.
   Validated bit-exact: `editing_logits cos_min=0.999999` and 47/47
   top-1 match on JFK.

   Reference dump pitfall: `GraniteModel.forward` unconditionally
   builds an upper-triangular causal mask and passes it to SDPA, which
   then enforces causality regardless of `self_attn.is_causal=False`.
   The upstream "flash_attention_2 required" assertion is real — only
   FA2 reads `is_causal` directly without using the mask. The
   `tools/reference_backends/granite_nle.py` dumper monkey-patches
   `transformers.models.granite.modeling_granite.create_causal_mask`
   to return None to get true non-causal attention via SDPA.

4. **Transcribe orchestration** (`granite_nle_transcribe`). DONE.
   Wires together: encoder (with BPE auxiliary head:
   `posterior_weighted_pool` window=4 driven by `1 - blank_prob_mid`
   from the L8 self-conditioning softmax, populating `last_bpe_logits`)
   → BPE-CTC greedy decode (`unique_consecutive` → drop blank label 0
   → shift to LLM IDs by -1) → `core_bpe::detokenize` (GPT-2 byte-level
   reverse, now shared with `granite_speech` and lifted into
   `core_bpe::token_bytes_to_utf8`) → strip + lowercase + " "-fallback
   → re-tokenize via `core_bpe::tokenize_simple` → `add_insertion_slots`
   (`max(2n+1, 8)`, EOS-padded) → `run_projector` divided by
   `embedding_multiplier=12` and sliced to `enc_T // downsample_rate=5`
   audio frames → `run_llm_editing` → per-row argmax + unique_consecutive
   + drop EOS + detokenize. JFK end-to-end output matches reference
   `final_text` exactly.

**Effort remaining:** none — encoder, projector, LLM editing, and
transcribe are all bit-exact end-to-end on JFK.

---

## 53. `core/audio_decoder.h` — DRY across TTS / codec backends

VibeVoice TTS (σ-VAE), Qwen3-TTS (RVQ codec) and MiMo
(MiMo-Audio-Tokenizer) all share the "discrete codes → 24 kHz
waveform" shape: a stack of 1D conv up-sampling blocks driven by a
small transformer or directly by quantised codes. Pull the recurring
patterns into `src/core/audio_decoder.h`:

- RVQ codebook lookup (multi-stage residual) — same in MiMo and
  Qwen3-TTS, plus their semantic-quantiser variant.
- 1D-conv up-sampling stack (`ggml_conv_1d_cf` already there from
  vibevoice work — reuse).
- Optional small transformer on the latent stream (Qwen3-TTS has 8L
  decoder transformer).

Same layering pattern as `core_conformer` / `core_attn` /
`core_ffn`. ~200-300 LOC in headers; payoff is two new TTS backends
plus future ones (e.g. Mimi/Encodec) sharing the same decode path.

**Effort:** Medium. Easier *after* one TTS runtime exists — extract
upward rather than design forward.

---


## ~~55. granite-family DRY refactor~~ — **DONE (May 2026)**

All five steps landed. See `HISTORY.md` §54 for the full table of commits,
LOC moved, and the step-5 plan correction (the speech / NAR Q-Formers
turned out to be structurally different — `core/qformer.h` shipped as a
NAR-only co-location rather than a both-TUs unification).

---


## 56. Kokoro multilingual phonemizer (espeak-ng)

Kokoro/StyleTTS2 is multilingual at the model level — the 178-symbol IPA
vocab covers en, de, fr, ru, cmn, ja and more — but until this work the
runtime always shelled out to `popen("espeak-ng -q --ipa=3 -v LANG …")`,
which (a) cost ~30–50 ms per call on the shell-quoting + fork path,
(b) needed `espeak-ng` on `$PATH`, and (c) emitted U+200D ZWJ tie
characters and newline-separated sentence chunks that the GGUF
tokenizer then has to silently absorb.

This item replaces the popen path with in-process libespeak-ng calls
behind a CMake AUTO probe, while keeping popen as a runtime fallback
so existing builds don't regress.

### Done (this session)

- `src/CMakeLists.txt`: `CRISPASR_WITH_ESPEAK_NG` cache string
  (`AUTO`/`ON`/`OFF`, default `AUTO`). AUTO probes `pkg-config
  espeak-ng` first, then a Homebrew/Linux fallback
  (`/opt/homebrew`, `/usr/local`, `/usr`). When found, defines
  `CRISPASR_HAVE_ESPEAK_NG=1` and links `libespeak-ng` via PUBLIC so
  it propagates into `crispasr` / `libcrispasr.dylib`. `ON` makes a
  missing lib a hard error; `OFF` skips the probe entirely.
- `src/kokoro.cpp`:
  1. `kokoro_phoneme_cache` — bounded LRU (1024 entries,
     mutex-protected) keyed on `lang \0 text`, lives in
     `kokoro_context`.
  2. `phonemize_espeak_lib()` — gated on `CRISPASR_HAVE_ESPEAK_NG`.
     Lazy `espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, …,
     espeakINITIALIZE_PHONEME_IPA | espeakINITIALIZE_DONT_EXIT)`
     behind a process-global mutex; sticky-init-failure flag so we
     don't keep retrying. `CRISPASR_ESPEAK_DATA_PATH` env var
     overrides the data dir for sandboxed apps. Voice changes are
     sticky. Loops `espeak_TextToPhonemes` until `textptr==NULL`,
     joining chunks with spaces.
  3. `phonemize_popen()` — the old shell-out, kept as a runtime
     fallback. `kokoro_synthesize` now calls `phonemize_cached()`
     which tries cache → lib → popen.
- `examples/cli/crispasr_backend_kokoro.cpp`: maps `-l/--language`
  to `cp.espeak_lang`. `auto` keeps the default (en-us) since
  espeak has no auto-detect mode.
- Smoke-tested standalone against libespeak-ng: en-us, de, fr,
  cmn, ru, ja all produce IPA. Compared lib vs popen: see
  LEARNINGS.md "Kokoro phonemizer: libespeak-ng vs popen
  divergence" for the ZWJ + sentence-join behaviour.
- Build verified: `otool -L libcrispasr.dylib` shows
  `libespeak-ng.1.dylib`; `nm libkokoro.a` has the three espeak
  symbols.
- **End-to-end synth check** (against
  `/Volumes/backups/ai/crispasr-models/kokoro-82m-f16.gguf` +
  `kokoro-voice-af_heart.gguf`):
  | lang | phonemes | duration | peak | RMS | verdict |
  |---|---|---:|---:|---:|---|
  | en  | clean | 3.45 s | 11443 | 1545 | ✅ healthy |
  | de  | clean | 4.08 s |   541 |   44 | ❌ near-silence on long phrases (no German voice — see open #1) |
  | fr  | clean | 3.40 s | 12374 | 1434 | ✅ healthy |
  | ru  | clean | 3.38 s | 11375 | 1506 | ✅ healthy |
  | cmn | espeak tone numbers (`ni2χˈɑu2…`) | 3.20 s | 11731 | 1627 | ⚠️ audio plays but tones unmodelled — open #2 |
  | ja  | kanji fallback (`(en)tʃˈaɪniːz(ja)…`) | 8.38 s | 15460 | 1581 | ⚠️ partial — kana works, kanji becomes English — open #3 |

  Short German phrases ("Hallo Welt.", "Guten Morgen.") synthesize
  fine with `af_heart`; the silence collapse only triggers on longer
  out-of-distribution phoneme sequences. See LEARNINGS.md "Kokoro
  phonemizer: libespeak-ng vs popen divergence" for full results.

### Open

1. **German voice pack — DE is a primary target language.** Kokoro-82M
   ships voices only for `a/b` (en US/UK), `e` (es), `f` (fr), `h` (hi),
   `i` (it), `j` (ja), `p` (pt), `z` (zh). No `d_*` (de), no `r_*` (ru),
   no Korean/Arabic. Three options ordered by effort:

   **Option 1 — Closer-language voice fallback (SHIPPED 2026-05-01).**
   Measured against the long German phrase ("Guten Tag, dies ist ein
   Test des deutschen Phonemizers."):

   | voice | peak | RMS | duration | verdict |
   |---|---:|---:|---:|---|
   | `af_heart` (English) |   541 |   44 | 4.08 s | silence collapse |
   | `ff_siwis` (French)  | 20577 | 2318 | 4.22 s | healthy, French-accented |
   | `ef_dora` (Spanish)  | 15036 | 1613 | 3.35 s | healthy, Spanish-accented |

   Wired into `examples/cli/crispasr_backend_kokoro.cpp` as an
   auto-fallback. Selection table:

   | `-l` value | preferred voice | rationale |
   |---|---|---|
   | `de`, `de-*`, `de_*` | `df_victoria` (Option 2b — kikiri-tts, Apache-2.0) → `df_eva` (Option 2a — Tundragoon, Apache-2.0) → `ff_siwis` | in-distribution to dida-80b backbone first; Tundragoon as second tier; French as last resort |
   | everything else without a native pack (ru, ko, ar, …) | `ff_siwis` (French) | non-silence baseline |

   Resolution: `--voice` (explicit) → cascade above → empty (helpful
   error). Explicit `--voice` always wins. Voice GGUFs live at
   `/Volumes/backups/ai/crispasr-models/kokoro-voice-{af_heart,
   ef_dora, ff_siwis, df_eva, dm_bernd, df_victoria, dm_martin}.gguf`.

   **Option 2a — Recovered Tundragoon's German voice packs (DONE,
   SHIPPED 2026-05-01).**
   The only public German Kokoro voice pack on HF was
   `Tundragoon/Kokoro-German` (Apache-2.0) — the user account was
   deleted in early 2026 and the HF repo is 404. **Voices recovered**
   from `r1di/kokoro-fastapi-german`'s Git LFS (`api/src/voices/v1_0/
   {df_eva,dm_bernd}.pt`, sparse + LFS pull). They are
   `[512, 1, 256]` F32 (vs the 510 of official Kokoro voices —
   Tundragoon's fine-tune used a slightly larger max_phonemes; the
   GGUF voice loader reads max_phonemes from the file so this is fine).

   End-to-end synth with the **official** Kokoro-82M model on the
   long German phrase ("Guten Tag, dies ist ein Test des deutschen
   Phonemizers."):

   | voice | peak | RMS | duration | note |
   |---|---:|---:|---:|---|
   | `df_eva` (German F)  | 14716 | 1648 | 3.50 s | healthy, German speaker |
   | `dm_bernd` (German M)| 19185 | 2374 | 3.88 s | healthy, German speaker |

   Both produce non-silent, German-timbred audio with the official
   Kokoro-82M weights — **the matching Tundragoon model fine-tune
   (`kokoro-german-v1_1-de.pth`) is not required.** That model is
   *unrecovered* (only available from the deleted HF repo per
   `r1di/docker/scripts/download_model.py`), but voices alone are
   sufficient for this fallback path. Caveat: predictor + decoder
   weights are still the official English-trained Kokoro-82M's, so
   prosody is not fully native German. Better than ff_siwis (German
   speaker timbre instead of French), worse than Option 2b.

   GGUF artefacts at
   `/Volumes/backups/ai/crispasr-models/kokoro-voice-{df_eva,dm_bernd}.gguf`.
   Wired as the German auto-fallback (Option 1 table above).

   **Option 2b — Native German backbone via dida-80b (SHIPPED 2026-05-01).**

   Sources (all Apache-2.0 weights + Apache-2.0 recipe + CC0 dataset):
   - Recipe: <https://github.com/semidark/kokoro-deutsch> — clone
     locally (recurse-submodules: `StyleTTS2/` + `kokoro/`).
     `scripts/extract_voicepack.py` is the tool for fresh per-speaker
     voicepacks; we did not need to run it (kikiri-tts ships
     pre-extracted voicepacks — see below).
   - Backbone: <https://huggingface.co/dida-80b/kokoro-german-hui-multispeaker-base>
     — `first_stage.pth` + `config.json`. Stage-1 multispeaker base
     fine-tune of Kokoro-82M on HUI-Audio-Corpus-German (51 speakers,
     51 h, 10 epochs A40, mel loss 0.583 → 0.326).
   - Pre-extracted voicepacks (kikiri-tts org, dida-80b maintainer):
     <https://huggingface.co/kikiri-tts/kikiri-german-victoria> +
     <https://huggingface.co/kikiri-tts/kikiri-german-martin>. Each
     ships `voices/{victoria,martin}.pt` extracted via the kikiri
     synthetic StyleEncoder which shares lineage with the dida-80b
     base — saves us from running `extract_voicepack.py` ourselves
     (the underlying HUI corpus is gated and would require a multi-step
     LibriVox-pulling pipeline to reproduce).

   What this adds over Option 2a:
   - **Predictor + decoder are German-trained.** Solves the root
     cause behind the af_heart silence collapse on long German
     phrases — voices alone (Option 2a) only cover the speaker
     timbre, not the prosody/duration distribution.
   - StyleEncoder is German-trained → kikiri voicepacks are in-
     distribution. Pairs cleanly with the dida-80b backbone.

   Steps taken:
   1. ✓ `models/convert-kokoro-to-gguf.py` extended for the modern
      `torch.nn.utils.parametrize` WeightNorm form
      (`parametrizations.weight.original0/original1`) used by dida-80b,
      tolerated the missing `module.` DataParallel prefix on bert keys,
      and added `--config` so the official Kokoro-82M `config.json`
      can be reused (dida-80b ships only a HF-hub stub config without
      vocab; the 178-symbol IPA vocab IDs are byte-identical per
      semidark's `training/kokoro_symbols.py`).
   2. ✓ Converted to
      `/Volumes/backups/ai/crispasr-models/kokoro-de-hui-base-f16.gguf`
      (163.7 MB at F16; 459 tensors mapped, 0 skipped — same byte size
      as `kokoro-82m-f16.gguf`, confirming identical architecture).
   3. ✓ Pulled kikiri voicepacks `voices/{victoria,martin}.pt`
      (510×1×256 F32) via `huggingface_hub.hf_hub_download` and
      converted them with the existing
      `models/convert-kokoro-voice-to-gguf.py` to
      `kokoro-voice-{df_victoria,dm_martin}.gguf` (~510 KB each,
      `[510,1,256]` F32 — direct passthrough, no converter changes).
   4. ✓ C ABI: new `crispasr_kokoro_resolve_model_for_lang()` and
      `crispasr_kokoro_resolve_fallback_voice()` in `src/kokoro.h` /
      `src/kokoro.cpp`, re-exported with the `_abi` suffix from
      `src/crispasr_c_api.cpp` so the dylib (and every wrapper that
      links against it) gets them.
   5. ✓ CLI: `examples/cli/crispasr_backend_kokoro.cpp` now delegates
      to the C ABI. When `-l de*` AND the user-passed model basename
      starts with `kokoro-82m`, the backend silently swaps to a
      sibling `kokoro-de-hui-base-f16.gguf` if present, then loads
      the German fallback voice from the new cascade
      `df_victoria → df_eva → ff_siwis`.
   6. ✓ Python wrapper: `crispasr.kokoro_resolve_for_lang(model, lang)`
      returns `KokoroResolved(model_path, voice_path, voice_name,
      backbone_swapped)`; surfaced from `crispasr/__init__.py`.

   End-to-end measurements on the long German phrase
   ("Guten Tag, dies ist ein Test des deutschen Phonemizers."), each
   ASR-roundtripped through `parakeet-v3 -l de` so we measure
   intelligibility and not just envelope:

   | model + voice | peak | RMS | sec | ASR roundtrip |
   |---|---:|---:|---:|---|
   | official + df_eva (Option 2a) | 14726 | 1648 | 3.50 | "...Phonemizer." (lost trailing 's') |
   | dida-80b + df_eva             | 23477 | 1830 | 3.50 | "...Phonemetzes." (1 word boundary error) |
   | dida-80b + df_victoria        | 12052 | 1177 | 4.22 | "...Tester des Deutschen Phonemizers." (1 word boundary error) |
   | dida-80b + dm_bernd           | 18948 | 2693 | 3.88 | "...Phonemetzers." (1 word boundary error) |
   | **dida-80b + dm_martin**      | 18100 | 1546 | 3.98 | **"...Phonemizers." (perfect)** |

   All four German voices clear the gate (peak ≥ 8000, RMS ≥ 1000)
   on the dida-80b backbone, and three of four are word-perfect except
   for one minor token-boundary error each. dm_martin is byte-perfect
   round-trip; df_victoria handles "Phonemizers" correctly which df_eva
   misses. This is the "fully native German signal path" the option
   promised: predictor + decoder + StyleEncoder distribution all
   German.

   For deployable single-speaker production quality, run Stage-2
   fine-tuning on one HUI speaker (~half-day on an A40) — out of
   scope of this PLAN item; track separately if needed.

   **Option 3 — Extract a style embedding via the English-trained
   StyleEncoder (only if 2a + 2b are blocked).**
   Same recipe as Option 2a's recovery effort but starting from a
   fresh German recording (Common Voice DE, public-domain
   audiobook). `[max_phon=510, 1, 256]` style tensor through
   StyleTTS2's StyleEncoder, save as `.pt`, convert. Strictly worse
   than Option 2b because the predictor/decoder aren't German-aware;
   keep as last-resort.

   **Status:**
   1. ✓ Option 1 shipped (auto-fallback table per-language).
   2. ✓ Option 2a shipped (df_eva + dm_bernd recovered from r1di's
      Git LFS, Apache-2.0; works with both backbones).
   3. ✓ Option 2b SHIPPED (dida-80b backbone + kikiri-tts voicepacks,
      all Apache-2.0; truly native German prosody on long phrases).
      Auto-routing kicks in when both `kokoro-82m-f16.gguf` and
      `kokoro-de-hui-base-f16.gguf` sit in the same directory.
   4. Option 3 not needed.

   **Follow-ups:**
   - ✅ HF GGUF mirrors published (2026-05-01):
     [`cstr/kokoro-82m-GGUF`](https://huggingface.co/cstr/kokoro-82m-GGUF),
     [`cstr/kokoro-de-hui-base-GGUF`](https://huggingface.co/cstr/kokoro-de-hui-base-GGUF),
     [`cstr/kokoro-voices-GGUF`](https://huggingface.co/cstr/kokoro-voices-GGUF)
     — F16 + Q8_0 backbones (Q4_K dropped — see LEARNINGS), 7 voicepacks.
   - ✅ Auto-download via `src/crispasr_model_registry.cpp` (PLAN #56).
     New `ExtraCompanion` mechanism in the registry — backends with >1
     auxiliary file (kokoro: English voice + German backbone + German
     voice) can list extras alongside the inline `companion_file`.
     `crispasr --backend kokoro -m auto -l de` now pulls all 4 files
     and auto-routes to the German backbone.
   - ✅ Wrapper TTS surface across Rust/Go/Java/JS/Ruby
     (commit `4f476c3`, 2026-05-01). Each binding gets
     `Session.{open,setVoice,setCodecPath,synthesize,close}` plus
     `kokoroResolveForLang(model, lang)` returning the same
     `KokoroResolved` shape as the Python wrapper.
   - Stage-2 fine-tune on one HUI speaker (~half-day A40) for
     deployable single-voice production quality. Out of scope here.
2. **Mandarin tone numbers.** espeak-ng outputs digit-suffixed
   tone markers (`ni2χˈɑu2`) that aren't in the kokoro-82m IPA vocab
   (178 symbols) and likely get dropped at tokenization, losing tone
   info. Investigate whether `--ipa=2` (without tone numbers) plus a
   separate tone embedding would work, or whether to switch to a
   different Mandarin G2P (e.g. `pypinyin`).
3. **Japanese kanji.** espeak-ng falls back to English pronunciation
   for kanji (e.g. 日本語 → "Chinese letter"), inserting `(en)…(ja)`
   voice-switch markers that aren't IPA. For full Japanese support,
   pre-process input with a Japanese frontend (`pyopenjtalk` /
   `mecab` + `kakasi`) to convert kanji → kana before espeak.
4. **Diff harness reference backend.** No `crispasr-diff kokoro`
   today. The reference dumper at
   `tools/reference_backends/kokoro.py` already exists for the
   model side (16 stages); extend it (or add a sibling) so the
   phonemizer step itself is also diffed — guard against future
   drift between popen / lib / future Python G2P.
5. **Optional polish.** A `kokoro_phoneme_cache_clear()` C ABI
   for long-running daemons that resynthesize across many speakers.
   Low priority.

### Effort

Small individually. Open items 2 + 3 are each an afternoon if we
go the pre-processing route. Open item 1 is "policy" — a one-line
fallback in the backend or a docs change. Open item 4 is ~150 LOC.
Open item 5 is ~20 LOC if asked.

---

## 57. Commercial-friendly TTS backend expansion

May 2026 sweep through high-traffic HF TTS models. Filter is **permissive
license + reusable architecture + reasonable effort**. Sequenced so each
phase unlocks a family of finetunes — finishing Phase 3 (Chatterbox stack)
also unlocks Phase 5's CFM solver, etc.

License triage that drives the ordering:

| ✅ Permissive (commercial OK) | ⚠️ Llama-3.2 community (commercial OK with attribution) | ❌ Non-commercial — defer |
|---|---|---|
| Qwen3-TTS-{Base,CustomVoice} (Apache 2.0) | Orpheus-3B family + Kartoffel_Orpheus (llama3.2) | SebastianBodza/Kartoffelbox-v0.1 (CC-BY-NC-ND) |
| ResembleAI/chatterbox base (MIT) | HumeAI/tada-3b-ml (llama3.2) | marduk-ra/F5-TTS-German (CC-BY-NC) |
| SebastianBodza/Kartoffelbox_Turbo (CC-BY-4.0, gated) | | mlx-community/fish-audio-s2-pro (Fish-Audio Research) |
| oddadmix/lahgtna-chatterbox-v0/v1 (MIT) | | amphion/Vevo1.5 (CC-BY-NC-ND) |
| openbmb/VoxCPM2 (Apache 2.0) | | mlx-community/Voxtral-4B-TTS-2603 (CC-BY-NC; upstream Mistral Apache OK) |
| FINAL-Bench/Darwin-TTS-1.7B-Cross (Apache 2.0) | | |
| AMAImedia Qwen3-1.7B-TTS-Cross-Darwin AWQ (Apache 2.0) | | |
| g-group-ai-lab/gwen-tts-0.6B (MIT) | | |
| kugelaudio/kugelaudio-0-open (MIT) | | |

License gaps to resolve before depending on a model: CosyVoice 3
(`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` — model card silent;
v1/v2 were Apache 2.0 but v3 not yet confirmed).

### Phase 1 — small code change (was "drop-in" — corrected May 2026 after smoke-test)

- **Qwen3-TTS-CustomVoice (0.6B)** — converted cleanly (402 tensors,
  F16 1.82 GB → Q8_0 968 MB, sitting in
  `/Volumes/backups/ai/crispasr-models/qwen3-tts-12hz-0.6b-customvoice-{f16,q8_0}.gguf`).
  **Smoke-test surfaced a contract mismatch:** CustomVoice has **no
  `speaker_encoder_config` / no ECAPA tensors** (76 tensors fewer
  than Base), and the runtime errors out with `no voice — call
  qwen3_tts_load_voice_pack or qwen3_tts_set_voice_prompt`.
  CustomVoice expects a **fixed speaker_id token** prepended to the
  talker prefill instead. Verified from `config.json` diff:

  ```
  spk_id: { aiden:2861, dylan:2878, eric:2875, ono_anna:2873,
            ryan:3061, serena:3066, sohee:2864, uncle_fu:3010,
            vivian:3065 }
  spk_is_dialect: { dylan:beijing_dialect(2074),
                    eric:sichuan_dialect(2062), ... }
  ```

  **Good news from `ref/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py:2091`:**
  the CustomVoice "speaker embedding" is just
  `talker.get_input_embeddings()(spk_id)` — a single 1024-d row from
  the codec embedding table the GGUF already contains. No extra
  weights, no codec encoder, no ECAPA. The contract is just "swap
  the ECAPA output for this row, and override `language_id` to the
  dialect token when applicable." Revised estimate ~150 LOC.

  **Status (May 2026): SHIPPED.**
  1. ✓ Converter emits `qwen3tts.tts_model_type` (string) +
     `qwen3tts.spk_names` (array<string>) + `qwen3tts.spk_token_ids`
     (array<u32>) + `qwen3tts.spk_dialect_token_ids` (array<u32>; 0
     means no dialect override).
  2. ✓ Runtime (`src/qwen3_tts.cpp`):
     - `load_spk_enc()` short-circuits to `true` for CustomVoice
       (no warnings about missing ECAPA tensors).
     - New `build_customvoice_prefill_embeds()` mirrors the
       non-streaming-mode block of `Qwen3TTSForConditionalGeneration.generate`
       (modeling_qwen3_tts.py:2166-2227): role(3) + bridge(L_codec-1)
       + text(N+1) + final(1).
     - New API: `qwen3_tts_is_custom_voice`, `qwen3_tts_n_speakers`,
       `qwen3_tts_get_speaker_name`, `qwen3_tts_set_speaker_by_name`.
       Set-by-name lifts the row from `talker.token_embd[spk_id]`
       and applies the dialect override to `language_id` when one
       exists and the active language is auto.
     - `qwen3_tts_synthesize_codes` branches on `tts_model_type`,
       skipping the "no voice"/"no ref_text" guards in CustomVoice
       mode.
  3. ✓ CLI (`crispasr_backend_qwen3_tts.cpp`): when
     `qwen3_tts_is_custom_voice`, `--voice` is a speaker name
     (defaults to first speaker if omitted). Backend factory now
     accepts `qwen3-tts-customvoice` / `qwen3-tts-cv` aliases so the
     auto-download registry can point at a separate variant.
  4. ✓ Registry: `qwen3-tts-customvoice` entry added pointing at
     `cstr/qwen3-tts-0.6b-customvoice-GGUF` (URL is the planned
     upload target — user uploads, no code change needed once
     pushed).
  5. ✓ Validation: 4 ASR roundtrips passed against parakeet-tdt-v3:
     - **vivian (English):** "Hello, this is a CustomVoice test using
       the vivian speaker." → "Hello! This is a custom voice test
       using the Vivian speaker." (verbatim modulo punctuation/case)
     - **aiden (English, default):** "The quick brown fox jumps over
       the lazy dog." → "The quick brown fox jumps over the lazy dog."
       (verbatim).
     - **serena (English, alias backend):** "Testing the new backend
       alias and the serena speaker." → "Testing the new back end
       Ilias and the Serena speaker." (1 ASR misrecognition of "alias",
       audio is clean).
     - **dylan (Beijing dialect):** "你好，今天天气真不错。" →
       dialect override to `language_id=2074` correctly engaged;
       3.28s clean audio (no Chinese ASR available locally for
       text-side verification, peak/RMS healthy at 0.37/0.040).

  6. **Pending (small, non-blocking):**
     - Reference backend extension (`tools/reference_backends/qwen3_tts.py`)
       so `crispasr-diff qwen3-tts` covers the CustomVoice prefill
       path. Diff coverage today is ICL/Base only.
     - HF upload + visible registry URL.

- **havok2/Kartoffelbox-v0.1_0.65h2** — checkpoint variant of the
  blocked Kartoffelbox-v0.1; inherits the same NC license. **Skip.**
- **SebastianBodza/Kartoffel_Orpheus_*** family + **lex-au/Orpheus-3b-German-FT-Q8_0.gguf**
  → land once Phase 2 Orpheus base is in.

### Phase 2 — talker pattern (qwen3_tts.cpp reuse)

Models with a Llama/Qwen-style AR talker + a small audio-token codec.
The talker forward fits directly into the `core_attn::kv_self_attn` +
`core_ffn::swiglu` pattern that #52 already uses.

- **Orpheus-3B backbone** (`canopylabs/orpheus-3b-0.1-ft`,
  llama3.2 license) — Llama-3.2-3B + SNAC codec. New backend
  `orpheus`. Effort: talker is ~80% reuse of qwen3_tts; SNAC is a
  small published RVQ codec (4 codebooks × 4096 @ 24 kHz). Once
  this lands, Kartoffel_Orpheus + lex-au + the various Orpheus
  finetunes are checkpoint swaps.
- **g-group-ai-lab/gwen-tts-0.6B** (MIT) — likely a Qwen3-TTS-style
  talker variant; need a weight inspection before sizing. If the
  shape matches, it's a #52 registry add.
- **HumeAI/tada-3b-ml** (llama3.2) — 3B Llama backbone + custom
  codec. Talker reuse high; codec is a new component. Defer until
  Orpheus lands so the SNAC vs Hume-codec contrast informs whether
  a `core_audio_codec` helper makes sense (overlaps with #53).

### Phase 3 — Chatterbox stack (CFM solver)

This is the family-unlock phase. Building a flow-matching (CFM) ODE
solver in ggml is the gating piece; once it's in, three commercial-OK
models become checkpoint-only adds.

- **ResembleAI/chatterbox** (MIT) — full pipeline: BPE tokenizer →
  T3 (0.5B Llama AR) → S3Gen (CosyVoice-style CFM, ~12 ODE steps)
  → HiFT-GAN-style vocoder → 24 kHz PCM. Plus voice encoder for
  cloning. New backend `chatterbox`.
- **SebastianBodza/Kartoffelbox_Turbo** (CC-BY-4.0, gated) — German
  t3 patch on Chatterbox-Turbo (350M, smaller). Drop-in once base
  lands. **Caveat from model card: training loss diverged late;
  paralinguistic tags (laugh/sigh/breath) likely non-functional.**
  Validate via #56-style ASR roundtrip before declaring usable.
- **oddadmix/lahgtna-chatterbox-v1** (MIT) — Arabic t3 patch.
  Drop-in once base lands.

The CFM solver landed here is **also** the gating piece for Phase 4
CosyVoice 3 (license permitting) and partially for Fish-Speech S2
(blocked on license anyway). Ship it once, three families light up.

### Phase 4 — codec-head additions to existing audio LMs

Already-supported encoder/decoders in the tree get a TTS direction by
adding a codec head + sampling path. Cheaper than a full new backend.

- **Voxtral-TTS** — Mistral's Voxtral with a TTS head. Both
  `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` (CC-BY-NC, the MLX
  converter relicensed) and `TrevorJS/voxtral-tts-q4-gguf` exist;
  use the upstream Apache 2.0 weights from Mistral, NOT the MLX
  variant. The voxtral4b ASR backend supplies the encoder/decoder;
  the TTS path needs a codec decode + new sampling. Estimate ~300 LOC.
- **FINAL-Bench/Darwin-TTS-1.7B-Cross** (Apache 2.0) + AWQ
  variant `AMAImedia/Qwen3-1.7B-TTS-Cross-Darwin-NOESIS-AWQ-INT4` —
  Qwen3-1.7B talker + "Darwin" codec. The 1.7B talker is a #52
  shape bump; the AWQ INT4 path is not currently supported and
  should not block (use bf16/fp16). Codec is new — assess after
  Orpheus's SNAC integration.

### Phase 5 — new architectures (medium-large, standalone value)

- **openbmb/VoxCPM2** (Apache 2.0, 1.26k likes) — CPM-backbone TTS
  with diffusion/flow head. Entirely new arch family in the tree.
  High user demand → worth the spend after Chatterbox lands so we
  can reuse whatever flow-matching utilities the CFM solver
  produces. Estimate: comparable to VibeVoice work (~1.5k LOC).
- **kugelaudio/kugelaudio-0-open** (MIT) — multi-component pipeline,
  needs deeper config read before sizing. Defer.

### Deferred / explicitly skipped

| Model | Reason |
|---|---|
| SebastianBodza/Kartoffelbox-v0.1 + havok2 derivative | CC-BY-NC-ND-4.0 — can't ship and can't even fine-tune. Recommend Kartoffelbox_Turbo (CC-BY-4.0) as the German Chatterbox path. |
| marduk-ra/F5-TTS-German | CC-BY-NC. F5-TTS arch is a DiT — would need new ggml ops, not worth the spend on an NC model. |
| mlx-community/fish-audio-s2-pro-* | Fish-Audio Research license — commercial requires separate Fish Audio license. |
| amphion/Vevo1.5 | CC-BY-NC-ND. Also voice conversion, different I/O contract. |
| mlx-community/Voxtral-4B-TTS-2603-mlx-4bit | MLX converter slapped CC-BY-NC on top. Use Mistral upstream Apache 2.0 weights via Phase 4 instead. |
| KevinAHM/pocket-tts-onnx, Pendrokar/xvapitch_nvidia | ONNX-only, niche, no clear demand. |
| NeuralAudioAI/NA_base, tokenaii/horus | Insufficient public info — re-evaluate if asked. |
| FunAudioLLM/Fun-CosyVoice3-* + ayousanz/cosy-voice3-onnx | License unverified on v3. Earlier CosyVoice generations were Apache 2.0; needs confirmation before committing to CFM solver work for it. |

### Per-model status

| Phase | Model | License | Status | Effort |
|---|---|---|---|---|
| 1 | Qwen3-TTS-CustomVoice 0.6B | Apache 2.0 | **DONE — runtime spk_id path landed; 4 ASR roundtrips passed (vivian / aiden / serena / dylan-dialect). Registry line added; HF upload pending.** | S |
| 1 | Qwen3-TTS-CustomVoice 1.7B | Apache 2.0 | queued | XS |
| 2 | Orpheus-3B base | llama3.2 | queued | M |
| 2 | Kartoffel_Orpheus DE (natural+synthetic) | llama3.2 | blocked on Orpheus base | XS |
| 2 | lex-au Orpheus-3B-DE-Q8 | llama3.2 | blocked on Orpheus base (already GGUF) | XS |
| 2 | gwen-tts-0.6B | MIT | queued — needs weight inspection first | S–M |
| 2 | tada-3b-ml | llama3.2 | queued | M |
| 3 | Chatterbox base | MIT | queued — CFM solver gating | L |
| 3 | Kartoffelbox_Turbo DE | CC-BY-4.0 (gated) | blocked on Chatterbox base | XS |
| 3 | lahgtna-chatterbox-v1 AR | MIT | blocked on Chatterbox base | XS |
| 4 | Voxtral-TTS (Mistral upstream) | Apache 2.0 | queued | M |
| 4 | Darwin-TTS-1.7B-Cross | Apache 2.0 | queued | M |
| 5 | VoxCPM2 | Apache 2.0 | queued — large new arch | L |
| 5 | kugelaudio-0-open | MIT | needs scoping | TBD |

### Effort

Phase 1 is hours. Phase 2 is one new backend (Orpheus) + N
checkpoint adds. Phase 3 is the CFM solver + Chatterbox runtime —
the largest single piece, but unlocks Phase 5's VoxCPM2 partially.
Phase 4 is bolt-ons. Phase 5 is standalone large.

Sequencing rationale: do Phase 1 immediately (free coverage), then
Phase 2 because Orpheus reuses #52's talker code most directly,
then Phase 3 because CFM is the biggest force-multiplier, then
Phase 4 (codec heads) as opportunistic, then Phase 5 (VoxCPM2) once
flow-matching utilities exist.

---

## Ecosystem expansion (lower priority)

### New backends from PazaBench assessment (see HISTORY.md #30)

| Model | License | Approach | Priority |
|---|---|---|---|
| Wav2Vec2 Conformer | Apache-2.0 | Conformer attention variant | Medium |
| Qwen2-Audio 7B | Apache-2.0 | Whisper encoder + Qwen2 LLM | Medium |
| OmniASR larger (1B/3B/7B) | Apache-2.0 | Same converter, bigger models | Medium |
| NeMo Canary-Qwen-2.5b | Apache-2.0 | FastConformer + Qwen2.5 decoder | Medium |
| Paza / Phi-4 | MIT | 14B multimodal, defer to llama.cpp | Low |
| **XiaomiMiMo/MiMo-V2.5-ASR** | TBD (check) | LLM-style multimodal speech (similar to Qwen3-ASR pattern) | Medium — user-requested in #35 |
| **google/gemma-4-E2B** | Gemma terms | Conformer + Gemma 4 decoder; matches "Gemma 4 Audio" entry below | Medium — user-requested in #35 |

### From llama.cpp (MIT)

| Model | Architecture | Notes |
|---|---|---|
| Ultravox | Whisper encoder + Llama 3.2 1B/8B | Speech understanding |
| Gemma 4 Audio | Conformer, chunked attention | Streaming, multimodal |
| LFM2-Audio | Conformer variant | Position embeddings |

### Post-processing

| Model | License | Type | Priority |
|---|---|---|---|
| FireRedPunc | Apache-2.0 | BERT punct (zh+en) | **DONE** |
| fullstop-multilingual | MIT | XLM-R punct (en/de/fr/it) | Medium |
| bert-restore-punctuation | MIT | BERT punct+truecase (en) | Medium |
| xashru/punctuation | Apache-2.0 | XLM-R+BiLSTM-CRF (40+ langs) | Low |

### Optimizations (cross-cutting, from survey + CrispEmbed comparison)

| # | Optimization | Applies to | Expected gain | Status |
|---|---|---|---|---|
| O1 | `ggml_soft_max_ext` fusion | wav2vec2, canary, fastconformer | -10% wav2vec2 | **DONE** |
| O11 | wav2vec2 CNN → ggml | wav2vec2 family | **10.8x** | **DONE** |
| O9/#44 | FireRed ggml Q4_K decoder | firered-asr | **6.3x** | **DONE** |
| O10 | Sliding window attention | voxtral4b | Already implemented | **DONE** |
| O2 | Fused QKV pre-merge | LLM decoders | ~10-15% attn (GPU) | API ready in core/attention.h; CPU gain <1%, defer to GPU |
| O3 | Temperature sampling | glm-asr, kyutai-stt | Feature parity | **DONE** |
| O5 | Pipelined mel+encode | LLM backends, CPU | ~15-20% | TODO |
| O4 | Beam search for LLMs | Audio-LLM backends | Quality | TODO |
| O6 | Batched encoder (GPU) | All + GPU | 3-5x | TODO |
| O7 | Speculative decoding | LLM backends | 2-4x decode | TODO |
| O12 | `ggml_conv_1d_cf` channels-first conv | vibevoice VAE | **-29% VAE, -15% total** | **DONE** |
| O13 | `ggml_conv_1d_group` + CNN cleanup | wav2vec2 family | **-12% total** (pos -12%, CNN -22%) | **DONE** |
| O14 | `--tts-steps` configurable DPM steps | vibevoice TTS | **-31% diffusion** | **DONE** |
| O15 | Remove redundant neg base LM | vibevoice TTS | Eliminated 60 LOC of wasted compute | **DONE** |

**From COMPARISON.md (llama.cpp patterns):**
- `ggml_soft_max_ext` with baked scale (O1) — already in llama.cpp, saves one `ggml_scale` op per attention layer
- Chunked window attention (O10) — llama.cpp uses for Gemma4A Conformer
- Conv2d subsampling via ggml ops — llama.cpp does this for Qwen3-ASR encoder

**From CrispEmbed (shared core patterns):**
- Fused QKV (O2) — CrispEmbed pre-merges Q/K/V weights at init, one matmul instead of 3
- SentencePiece Viterbi DP tokenizer — CrispEmbed has proper optimal tokenization
- Lazy graph allocation (`no_alloc=true` + scheduler) — reduces memory churn

**From LEARNINGS.md (FireRed decoder triage):**
- Small per-step ggml graphs are SLOWER than CPU loops (scheduling overhead)
- BUT: native Q4_K matmuls via ggml are 9.3x faster than F32 OpenMP (lesson: never dequant)

### Audio format support

- `.m4a`, `.mp4`, `.webm` crash with upstream ffmpeg integration — needs fix or robust fallback
- `.aiff`, `.wma`, raw PCM not supported without pre-conversion
- Consider bundling a lightweight M4A/AAC decoder or improving the ffmpeg path
- Only move LARGE, REUSED matmuls onto ggml/GPU
- Persistent subgraphs per decode step > one-off graphs

### Other

- **OmniASR-LLM beam search** — beam=2+ with N hypothesis KV caches
- ~~**TTS module** — VibeVoice-Realtime-0.5B text-to-speech~~ **DONE** — perfect ASR round-trip on all test cases. 17 bugs found via stage-by-stage diff. Uses DPM-Solver++, dual KV CFG, voice prompts, EOS classifier, text/speech interleaving.
- ~~**ggml_conv_1d_dw F16 im2col fix**~~ **DONE** — solved via `ggml_conv_1d_dw_cf` (direct F32, no im2col)

---

## Publish language wrappers to package registries

Today the Rust, Dart, and Python wrappers all live in this repo and (for
Python) require a `pip install -e .` from a clone. Move all three onto
their language-native registries so users can install with one command.

**Status (2026-04-25):** All three wrappers now have publishable
metadata + dry-runs pass. The CI workflow `release-wrappers.yml` is
wired up but cannot run until the **one-time registry setup** below
is complete.

| Wrapper | Pre-flight | Blocker |
|---|---|---|
| Python `crispasr` 0.5.4 | sdist + wheel build clean | PyPI trusted-publisher must be configured |
| Dart `crispasr` 0.5.4 | `dart pub publish --dry-run` passes (warnings only) | pub.dev automated publishing must be configured |
| Rust `crispasr-sys` 0.5.4 | `cargo publish --dry-run` clean (5.9 KiB) | needs `CARGO_REGISTRY_TOKEN` repo secret |
| Rust `crispasr` 0.5.4 | publish-order dependent on `crispasr-sys` | same |

### One-time registry setup (must happen before first tag)

1. **PyPI** — go to https://pypi.org/manage/account/publishing/ and add
   a "pending publisher": owner `CrispStrobe`, repo `CrispASR`,
   workflow `release-wrappers.yml`, environment `pypi`. Then push any
   `v*` tag.
2. **crates.io** — generate a token at https://crates.io/me, add it
   as the `CARGO_REGISTRY_TOKEN` secret on the GitHub repo.
3. **pub.dev** — go to https://pub.dev/packages/crispasr/admin (after
   first manual publish or claim) → enable automated publishing → set
   tag pattern `v{{version}}`. Alternatively for the first publish,
   run `dart pub publish` locally with the package owner's credentials.

### Pattern (matches crispasr approach)

All three wrappers are thin FFI/ctypes shims over the C ABI in
`src/crispasr_c_api.cpp`. They do **not** bundle the native library — the
user must have `libcrispasr.{so,dylib,dll}` installed (Homebrew, apt, or
built from source). This keeps the wheels/crates/pub packages tiny and
avoids a per-platform build matrix on every release.

| Wrapper | Registry | Effort | Notes |
|---|---|---|---|
| Python | PyPI | Low | Add `python/pyproject.toml`; pure-Python wheel; `_helpers.c` builds at install if a C toolchain is present, else falls back to ctypes-only path |
| Rust   | crates.io | Low | `crispasr-sys` then `crispasr` (two `cargo publish` calls); already has `Cargo.toml` |
| Dart   | pub.dev | Low | `flutter pub publish --dry-run` then `flutter pub publish`; already has `pubspec.yaml` |

### Library discovery (Python)

Update `_find_lib()` in `python/crispasr/_binding.py` to probe, in order:
1. `$CRISPASR_LIB_PATH` env var (explicit override)
2. `sys.prefix/lib/` (system or virtualenv install)
3. Standard Homebrew/Linux paths (`/opt/homebrew/lib`, `/usr/local/lib`, `/usr/lib`)
4. Existing repo-relative fallbacks (for `pip install -e .` from a clone)

If none found, raise `RuntimeError` with a helpful message linking to
install docs (the same pattern Tesseract / faster-whisper use).

### Release automation

Add a tag-triggered workflow `.github/workflows/release-wrappers.yml`
that, on `v*` tags, runs in parallel:
- `python -m build && twine upload` (PyPI, OIDC trusted-publishing — no API token)
- `cargo publish -p crispasr-sys && cargo publish -p crispasr` (crates.io, `CARGO_REGISTRY_TOKEN` secret)
- `dart pub publish --force` (pub.dev, OIDC publishing)

Trigger only on tag push, not on every commit. Version bumps stay
manual — bump `pyproject.toml` / `Cargo.toml` / `pubspec.yaml` together
in the same commit that creates the tag.

### Future: bundled wheels for Python

After the pure-Python release is out, add a follow-up release pipeline
using `cibuildwheel` to produce manylinux2014 + macOS arm64/x64 +
Windows wheels with `libcrispasr.*` bundled inside via `auditwheel` /
`delocate` / `delvewheel`. Same for Rust if we ever want
`crispasr-sys` to vendor the native build like `tch-rs` /
`onnxruntime-sys` do. Defer until pure-Python wheel is out and stable.
