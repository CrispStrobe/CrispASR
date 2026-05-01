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
   | `de`, `de-*`, `de_*` | `df_eva` (Option 2a — Tundragoon, Apache-2.0) | German speaker timbre, healthy synth |
   | everything else without a native pack (ru, ko, ar, …) | `ff_siwis` (French) | non-silence baseline |

   Resolution: `--voice` (explicit) → preferred per language → ff_siwis
   if preferred missing → empty (helpful error). Explicit `--voice`
   always wins. All three voice GGUFs (`af_heart`, `ff_siwis`,
   `ef_dora`, `df_eva`, `dm_bernd`) live at
   `/Volumes/backups/ai/crispasr-models/`.

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

   **Option 2b — Use `dida-80b/kokoro-german-hui-multispeaker-base`
   as the German backbone (best path to real native quality).**
   <https://huggingface.co/dida-80b/kokoro-german-hui-multispeaker-base>
   — a true Stage-1 multispeaker base fine-tune of Kokoro-82M on
   the HUI-Audio-Corpus-German (51 speakers, 51 h, 10 epochs A40).
   Apache-2.0 weights + CC0 training data. **This is a separate
   model from Kokoro-82M, not a voice pack:** ships only
   `first_stage.pth` + `config.json`. Why it helps:
   - StyleEncoder, predictor, and decoder are all trained on
     German — solves the root cause of the silence collapse.
   - Style embeddings extracted from German speakers will be
     in-distribution.
   - HUI corpus is open, so we can pick any speaker as reference.

   Steps:
   1. Convert `first_stage.pth` to GGUF. Verify the converter at
      `models/convert-kokoro-to-gguf.py` handles dida-80b's tensor
      names — it's a fine-tune of Kokoro-82M so layout *should*
      match, but spot-check first. Output:
      `kokoro-de-hui-base-f16.gguf` (~163 MB at F16).
   2. Extract a German voice pack: pick one speaker from HUI
      (file IDs are public), run a 10 s reference mel through the
      *German-fine-tuned* StyleEncoder, save as a `.pt` shaped
      `[510, 1, 256]`, convert via existing
      `convert-kokoro-voice-to-gguf.py`.
   3. Dispatcher logic: when `-l de` is set, load
      `kokoro-de-hui-base-f16.gguf` instead of `kokoro-82m-f16.gguf`
      (and the matching German voice). Implement as a per-language
      model-routing table in `crispasr_backend_kokoro.cpp` or the
      auto-download manifest.

   For deployable single-voice production quality, run Stage-2
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
      Git LFS, Apache-2.0; sufficient with official Kokoro-82M).
   3. Open: Option 2b (dida-80b) for true native German prosody —
      adds a second Kokoro GGUF variant. Track parallel to this
      under "Native German backbone via dida-80b".
   4. Option 3 only if 2b is blocked (it isn't — model is on HF).
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
