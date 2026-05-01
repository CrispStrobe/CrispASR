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
| **MEDIUM** | [#5 Reference backends](#5-reference-backends-for-parakeetcanarycohere) | Medium | parakeet/cohere DONE; canary remaining |
| **MEDIUM** | [#53 core/audio_decoder.h](#53-coreaudio_decoderh--dry-across-tts--codec-backends) | Medium | DRY across qwen3-tts/mimo/vibevoice |
| **MEDIUM** | [#55 granite-family DRY refactor](#55-granite-family-dry-refactor) | Medium | ~1500 LOC duplicated between granite_speech and granite_nle; ~5 lifts gated by diff harness |
| **MEDIUM** | [#56 Kokoro multilingual phonemizer](#56-kokoro-multilingual-phonemizer-espeak-ng) | Small | espeak-ng linkage + LRU + `-l/--language` wired; pending: kokoro-82m.gguf for end-to-end synth + diff-harness reference backend |
| **LOW** | #41 Moonshine IPA / phoneme | High | Deferred |
| **LOW** | [#7 voxtral4b streaming](#7-native-voxtral4b-streaming) | High | |
| **LOW** | [#9 Parakeet TDT GPU](#9-parakeet-tdt-decoder-gpu) | Medium | |
| **LOW** | [#11 WebSocket server](#11-websocket-streaming-server) | High | |
| **LOW** | [#16 Shaw RPE](#16-shaw-rpe-for-granite-graph) | Medium | |
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

Add query-dependent Shaw RPE to granite ggml encoder graph (currently
uses flash_attn_ext without RPE). Manual attention: QK^T + RPE bias +
softmax + V matmul.

**Effort:** ~80 LOC.

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

## 55. granite-family DRY refactor

`src/granite_speech.cpp` (2570 LOC) and `src/granite_nle.cpp` (2096
LOC) share roughly 1500 LOC of near-identical math. Both ship the
same 16-layer Macaron Conformer encoder, the same windowed simplified
Q-Former projector, and the same 40-layer Granite-1B LLM with µP
multipliers + RoPE — only the *plumbing* (autoregressive generate vs.
non-autoregressive single-pass; 2-layer hidden-concat in PLUS vs.
4-layer in NAR; BPE auxiliary head only in NAR) differs. PLUS is also
shipped via the same `granite_speech.cpp` runtime + a `cat_hidden_layers`
config knob, so the refactor benefits all three variants at once.

This is a pure refactor — no math changes, no tensor renames, no
behaviour changes. Each step lands behind the diff harness:
`crispasr-diff granite-speech` (base + plus, JFK transcribes ✅) and
`crispasr-diff granite-nle` (transcribe == ref ✅) must both stay
green after every commit. Reference: the granite-speech-4.1 family
just landed in this state in commits `a03c529` (LLM editing bit-exact)
and `d4b892f` (NAR transcribe end-to-end), so the harness gates are
known-good baselines.

### Duplication map (observed during the NAR transcribe work)

| What | Lives in | LOC | Identical? | Where to lift |
|---|---|---|---|---|
| Radix-2 Cooley-Tukey FFT | `granite_speech.cpp` + `granite_nle.cpp:316` (comment notes future cleanup) + `kokoro.cpp` + `mimo_tokenizer.cpp` | ~80 each | Yes (granite/granite_nle); mostly yes (kokoro/mimo) | `core/fft.h` |
| 80-bin log-mel pipeline (FFT → 512 mel filterbank → 8-frame stack → log-clip) | both granite | ~150 each | Yes | extend existing `core/mel.{h,cpp}` (already used by parakeet/canary) |
| `nle_cpu_layernorm` / `nle_run_matmul` (CPU fallbacks) | both granite | ~30 each | Yes | `core/cpu_ops.h` |
| Macaron Conformer block (FFN1 + Shaw-RPE attn + conv module + FFN2 + post-norm) — `nle_run_ffn`, `nle_run_conv_module`, `nle_shaw_block_attention_cpu`, `nle_run_norm_matmul_pair` | both granite | ~600 each | Yes (different from parakeet's FastConformer in `core/fastconformer.h`) | `core/conformer_ibm.h` (sibling of fastconformer.h; do NOT merge — Shaw RPE / fused conv layout / BN folding differ) |
| Granite-1B LLM forward (40 layers, GQA 16/4, µP multipliers, RoPE θ=10000, optional `is_causal=False`) — `nle_llm_attn_noncausal` + `nle_build_llm` | both granite (causal in granite_speech, non-causal in granite_nle) | ~250 each | Same shape, different `is_causal` | `core/granite_llm.h` taking an `is_causal` flag |
| Window-based simplified Q-Former projector (layer_proj + N×{cross-attn + MLP} + out_norm + out_linear) — `nle_proj_layer_proj` + `nle_proj_build_block` | both granite | ~300 each | Yes; only the # of input encoder layer concat'd differs | `core/qformer.h` |
| Posterior-weighted pool (used by NAR's BPE aux head) | granite_nle only today | ~25 | n/a | `core/ctc.h` (next to existing `greedy_decode.h`) — generic CTC trick, useful for any aux-head variant |
| BN-folding-at-load + Shaw RPE lookup-table builder | both granite | ~80 each | Yes | already-marginal; bundle with `conformer_ibm.h` |

GPT-2 byte-level detokenize (`core_bpe::detokenize` /
`token_bytes_to_utf8`) was already lifted in commit `d4b892f`; the
encode side (`core_bpe::tokenize_simple` + `bpe_one`) was lifted
earlier. That work proves the lift pattern works — all three granite
variants ship the same byte-decoder now, with no diff regressions.

### Five-step refactor (in landing order)

Each step is independently committable + reverentible. Diff-harness
green is the gate; do not move to step N+1 until step N's commit
passes both granite-speech (base/plus) and granite-nle.

**Step 1 — `core/fft.h` + `core/cpu_ops.h`. Risk: very low.**

Lift `granite_nle_fft` + `granite_nle_fft_wrapper` into
`core/fft.h` as `core_fft::fft_radix2(...)` and a thread-local-scratch
helper. Lift `nle_cpu_layernorm` + `nle_run_matmul` (and the matching
`granite_speech` copies) into `core/cpu_ops.h` as
`core_cpu::layernorm` and `core_cpu::matmul_F32` (or `matmul` taking
either F32/F16 weights via overload). Update both granite TUs to
include the new headers and call through; delete the old static
helpers. **LOC saved:** ~250. **Validation:** the first lift that
exercises both granite_speech and granite_nle in one PR — both diff
harnesses must report identical numbers (cos values shouldn't even
shift in the LSB; this is a pure rename).

**Step 2 — `core/ctc.h` (posterior-weighted pool + greedy decode helpers). Risk: very low.**

Add `core_ctc::posterior_weighted_pool(hidden, importance, window)`
to a new `core/ctc.h`. Move the BPE-CTC greedy decode from
`granite_nle_transcribe` (the unique_consecutive → drop-blank →
shift-by-K loop) into `core_ctc::greedy_decode_with_blank(logits, T,
V, blank_id, shift)`. parakeet/canary's CTC paths already use
`core/greedy_decode.h`; co-locate. **LOC saved:** ~60.
**Validation:** `crispasr-diff granite-nle` transcribe == ref.

**Step 3 — `core/conformer_ibm.h` (Macaron block forward). Risk: medium.**

Lift `nle_run_ffn`, `nle_run_conv_module`,
`nle_shaw_block_attention_cpu`, `nle_run_norm_matmul_pair`, and the
per-layer Shaw RPE lookup-table builder into `core/conformer_ibm.h`
behind a `core_conformer_ibm::block(ctx, hidden, T, d, n_heads, hd,
ctx_size, ...)` API. The encoder forward in both granite TUs becomes
a 16-iteration loop calling `core_conformer_ibm::block`. Keep
`core/fastconformer.h` (NeMo style) untouched — the Shaw RPE +
fused conv layout + BN folding diverge enough that merging the two
would obscure both. **LOC saved:** ~600. **Validation:** the encoder
forward is the most cosine-sensitive lift; expect drift in the LSB
only. encoder_output cos_min must remain ≥ 0.999852 for granite-nle
and ≥ 0.99 for granite-speech base/plus.

**Step 4 — `core/granite_llm.h` (40-layer forward, causal toggle). Risk: medium.**

Lift `nle_llm_attn_noncausal` + `nle_build_llm` and the matching
causal `granite_speech` LLM forward into `core/granite_llm.h` as
`core_granite_llm::build_graph(ctx, embed_inputs, n_tokens,
position_ids, params, is_causal)`. Both granite TUs now compose
projector_out + (audio + text_with_slots OR audio + prompt) into a
single embedding tensor and hand it to the same builder. The µP
multipliers (`embedding_multiplier=12`, `attention_multiplier=
0.0078125`, `residual_multiplier=0.22`, `logits_scaling=8`) live in
the params struct. The only behavioural difference from the existing
two paths is the `is_causal` flag and whether `lm_head` divides by
`logits_scaling` (NAR bypasses it; base/plus uses it for sampling
correctness). **LOC saved:** ~250. **Validation:** editing_logits
cos_min must stay at 0.999999 for granite-nle; base/plus must keep
transcribing JFK identically.

**Step 5 — `core/qformer.h` (windowed simplified Q-Former). Risk: low-medium.**

Lift `nle_proj_layer_proj` + `nle_proj_build_block` into
`core/qformer.h` as `core_qformer::run_windowed(...)`. Parameters: #
input encoder-layer concat (2 in PLUS, 4 in NAR, 1 in base),
block_size, downsample_rate, n_layers, n_heads, mlp_ratio. The
runtime call sites in both granite TUs collapse to a single function
call. **LOC saved:** ~300. **Validation:** projector_output cos_min
must stay at 0.999999.

### Non-goals

- **No graph-mode encoder.** `GRANITE_ENCODER_GRAPH=1` (the
  experimental Metal-accel encoder graph) is gated by §16 (Shaw RPE
  in graph path) and stays out of this refactor.
- **No FastConformer merge.** parakeet/canary use a different
  Conformer dialect (NeMo's, with conv subsampling and MHA RPE);
  forcing both into one header would muddy both. Keep
  `core/fastconformer.h` and `core/conformer_ibm.h` separate.
- **No tensor renames.** GGUF on-disk names stay identical; existing
  cstr/* HF GGUFs continue to load.
- **No new µP knobs.** Multipliers move to a struct, but their
  values stay hardcoded as defaults from `granite-1b-base` config.

### Acceptance criteria

After all five steps:
- `crispasr-diff granite-speech` (base) — JFK transcribes identically.
- `crispasr-diff granite-speech` (plus, F16 + 3× Q4_K) — same.
- `crispasr-diff granite-nle` — all 6 stages PASS, transcribe matches
  reference `final_text` exactly.
- `granite_speech.cpp` + `granite_nle.cpp` total LOC drops by ~1500
  (from ~4670 to ~3170). New `core/*.h` files add ~600 LOC. Net win:
  ~900 LOC of duplicated math removed.
- No regression in encoder/projector/LLM cosine numbers (allow LSB
  drift only; reject anything that moves cos_min by ≥ 1e-5).

### Effort

Medium. Step 1 is an afternoon. Steps 2–5 are each a session.
Total: ~5 focused sessions. The diff harness gates make each step
zero-risk to land independently — the project never goes through a
"refactor in progress, please don't touch granite" window.

### Why now (this is the optimal moment)

1. All three runtimes (base, plus, nar) just hit bit-exact diff
   parity (commits `a03c529` + `d4b892f`). Any future feature work
   would need to be implemented twice if we don't lift first.
2. The user just asked the DRY question — duplication is recent +
   visible in the working set, easy to see. Six months from now the
   same lift would require re-discovery.
3. `core_bpe` lift in `d4b892f` proved the pattern + the harness
   gate workflow works on the granite family.

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

   **Option 1 — Closer-language voice fallback (DONE measurement,
   SHIP wiring).**
   Tested 2026-05-01 with the same long German phrase ("Guten Tag,
   dies ist ein Test des deutschen Phonemizers."):

   | voice | peak | RMS | duration | verdict |
   |---|---:|---:|---:|---|
   | `af_heart` (English) |   541 |   44 | 4.08 s | silence collapse |
   | `ff_siwis` (French)  | 20577 | 2318 | 4.22 s | healthy, French-accented |
   | `ef_dora` (Spanish)  | 15036 | 1613 | 3.35 s | healthy, Spanish-accented |

   `ff_siwis` is the strongest non-native fallback by RMS. Both are
   converted and live at
   `/Volumes/backups/ai/crispasr-models/kokoro-voice-{ff_siwis,ef_dora}.gguf`.
   **Speaker timbre + prosody will sound French/Spanish, not German**
   — the duration predictor was trained on those distributions, not
   on German. Acceptable as a non-silence baseline; not production
   quality.

   *Wiring task* (~30 LOC): in
   `examples/cli/crispasr_backend_kokoro.cpp`, when `-l` is one of the
   languages without a native voice pack (de, ru, ko, ar, …) and the
   user did not pass `--voice`, auto-select `ff_siwis` and emit one
   warning line ("kokoro: no native voice for 'de', falling back to
   'ff_siwis' (French speaker, French-accented prosody) — see PLAN
   #56"). Document in README. Do **not** override an explicit
   `--voice` argument.

   **Option 2 — Community-trained German voice pack (preferred for
   real production quality).**
   Kokoro has an active fine-tuning community; some have published
   German + other-language voices to HF as `.pt` style packs
   compatible with the official `Kokoro-82M` checkpoint. Search HF
   for tags like `kokoro german`, `kokoro-de`, `kokoro-multilingual`.
   If found:
   - download the `.pt`,
   - run `models/convert-kokoro-voice-to-gguf.py --input X.pt
     --output kokoro-voice-X.gguf` (already shape-validates [510, 1,
     256]),
   - drop into `/Volumes/backups/ai/crispasr-models/`,
   - quality-check with the same long-German phrase + `python -c
     "import wave …"` RMS gate (peak ≥ 8000, RMS ≥ 1000 = passes;
     anything lower means the German didn't actually train well).

   Quality risk: the `.pt` is just a style embedding — it doesn't
   change the model's predictor/decoder weights. If a community
   "voice" was extracted from German speech without any model
   fine-tuning, the duration predictor still sees German-IPA-with-
   English-statistics, and quality will be limited by the base model's
   German latent capacity. Worth trying, but verify with a listening
   test, not just RMS.

   **Option 3 — Extract our own German style embedding from a
   reference recording.**
   Kokoro voice packs are `[max_phonemes=510, 1, 256]` style tensors,
   produced by running a clean reference mel through the StyleEncoder
   in StyleTTS2. Recipe:
   1. Source a clean German speaker recording (Common Voice DE,
      Mozilla TTS dataset, or a public-domain audiobook). Need
      ~10 sec at 24 kHz mono.
   2. In Python with the official `kokoro` package (or its underlying
      `models.py`), call the StyleEncoder forward to get a
      `[1, 1, 256]` style vector. Repeat across 510 rows (or extract
      per-utterance-length variants if we want length-conditioned
      style — the official voice packs vary along axis 0).
   3. Save as a torch `.pt` matching the official voice format, then
      run our existing converter.
   Same caveat as option 2: the model predictor wasn't trained on
   German duration/F0 distributions, so style alone may not unlock
   fluent German. Test with the same RMS gate + listening pass before
   shipping.

   **Recommended order:** ship Option 1's auto-fallback wiring now
   (it's a one-liner in the backend), then in parallel pursue Option 2
   (search HF) — if a usable community pack exists, it's the cheapest
   path to native quality. Option 3 is the fallback if neither of the
   above gives acceptable results.
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
