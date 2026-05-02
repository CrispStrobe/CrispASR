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
| **MEDIUM** | [#52 Qwen3-TTS](#52-qwen3-tts) — perf pass | Medium | talker + code_predictor + codec + ECAPA + codec_encoder all done; only step-4 perf pass open (~137 ms/frame → real-time) |
| **DONE** | [#51 MiMo-V2.5-ASR runtime](#51-mimo-v25-asr-runtime--done-may-2026) | Large | end-to-end JFK matches reference; F16+Q4_K on HF; 51b/b' shipped (step-only + cached step graph, 1.46× decode); 51a mmap loader landed behind `CRISPASR_GGUF_MMAP=1` (default-flip + 51c F16 step pending) |
| **DONE** | [#54 granite-speech-4.1 plus / nar](#54-granite-speech-41-plus--nar-variants) | Small | base + plus + nar runtimes all DONE; F16 + 3 Q4_K variants shipped to [`cstr/granite-speech-4.1-2b-nar-GGUF`](https://huggingface.co/cstr/granite-speech-4.1-2b-nar-GGUF); registry wired |
| **HIGH** | [#57 Commercial-friendly TTS expansion](#57-commercial-friendly-tts-backend-expansion) | Phased | Phase 1 (Qwen3-TTS-{CustomVoice 0.6B/1.7B, Base 1.7B, VoiceDesign 1.7B}) DONE; Phase 2 Orpheus-3B base + lex-au-orpheus-de + Kartoffel_Orpheus DE natural DONE; Kartoffel synthetic converted locally, HF upload pending; phases 3-5 queued |
| **DONE** | [#5 Reference backends](#5-reference-backends-for-parakeetcanarycohere) | Medium | parakeet / cohere / canary all DONE — see [HISTORY §63](HISTORY.md) |
| **DONE** | [#53 Two narrow extractions](#53-two-narrow-extractions-for-shared-tts-codec-patterns) | Small | snake_beta + convt1d_crop landed across qwen3-tts + SNAC — see [HISTORY §63](HISTORY.md) |
| **MEDIUM** | [#56 Kokoro multilingual phonemizer](#56-kokoro-multilingual-phonemizer-espeak-ng) | Small | espeak-ng + DE backbone shipped; HF GGUFs published 2026-05-01; auto-download wired; only Mandarin tones / JA kanji + diff-harness phonemizer-step polish remain |
| **MEDIUM** | [#58 MOSS-Audio-4B-Instruct](#58-moss-audio-4b-instruct) | Large | first audio-understanding (not just ASR) backend; introduces DeepStack cross-layer feature injection |
| **MEDIUM** | [#59 Cross-binding C-ABI parity](#59-cross-binding-c-abi-parity) | Medium | TTS surface (incl. qwen3-tts variants) at full parity across all 7 wrappers; align/diarize/VAD/streaming/punctuation/LID/registry still C-ABI-only on Go/Java/Ruby/JS/Dart |
| **MEDIUM** | [#60 llama.cpp/llamafile perf trick ports](#60-cross-backend-perf-tricks-llamacpp--llamafile-ports) | 14 items | 60a/b/c/f/g DONE; 60d (Fused QKV) + 60e (KV quant) OPEN; 60h-n parked/skip |
| **LOW** | #41 Moonshine IPA / phoneme | High | Deferred |
| **LOW** | [#7 voxtral4b streaming](#7-native-voxtral4b-streaming) | High | |
| **LOW** | [#9 Parakeet TDT GPU](#9-parakeet-tdt-decoder-gpu) | Medium | |
| **LOW** | [#11 WebSocket server](#11-websocket-streaming-server) | High | |
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

## ~~5. Reference backends for parakeet/canary/cohere~~ — **DONE (May 2026)**

All three reference dumpers shipped: `parakeet.py` (April 2026, used to
diagnose the JA xscaling bug), `cohere.py` (encoder-decoder NeMo), and
`canary.py` (May 2026, commit `63f708e` — patterned on parakeet, with
TimeMels-layout transpose and 32 per-layer encoder hooks for diagnostic).
See [HISTORY §63](HISTORY.md) for the canary writeup.

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


## ~~16. Shaw RPE for granite graph~~ — **DONE (May 2026)**

Per-block subgraph attention with Shaw RPE landed and is now the
default encoder path for **all three** granite variants —
granite-speech base, PLUS (in-graph cat_layers concat), and NAR
(`granite_nle.cpp` mirror with self-cond residual + snapshot concat
+ final CTC logits + blank-prob softmax tap). Per-layer RPE bug
(layer 0 reused for all 16 blocks) fixed. End-to-end realtime on
M1 + Q4_K: base 2.3× → 4.8×, plus 1.2× → 2.9×, nar 0.6× → 1.7×.
All three transcribe JFK byte-for-byte identical to the CPU loop.
`GRANITE_DISABLE_ENCODER_GRAPH=1` is the unified escape hatch.
Full write-up in [HISTORY.md §55](HISTORY.md). Lesson recorded in
LEARNINGS.

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

## ~~51. MiMo-V2.5-ASR runtime~~ — **DONE (May 2026)**

End-to-end JFK transcription matches the upstream Python
`MimoAudio.asr_sft` reference verbatim. F16 + Q4_K shipped to
[`cstr/mimo-asr-GGUF`](https://huggingface.co/cstr/mimo-asr-GGUF)
with corrected vocab (151680) + merges (151291). See HISTORY entry
56 for the full bug post-mortem. Remaining (low-priority) follow-ups:

### 51a. mmap-style GGUF loader for large F16 models — env-flag SHIPPED

Zero-copy CPU path landed in commit `9710f80` behind
`CRISPASR_GGUF_MMAP=1`: a custom `ggml_backend_buffer_t` wraps the
mmap'd file region, tensors bind directly into mmap offsets via
`ggml_backend_tensor_alloc`, and the buffer's `free_buffer` callback
munmaps when the model is freed. Mmap is `MAP_PRIVATE +
PROT_READ|PROT_WRITE` (Win32 `FILE_MAP_COPY`) so backends that mutate
weights post-load (parakeet's BN-into-conv fold, etc.) get COW pages
instead of SIGBUS. See HISTORY §62 for the implementation walkthrough.

Validated on parakeet Q4_K (Metal default + CPU default + CPU+mmap, all
correct JFK transcripts) and mimo-asr Q4_K (5.5 GB peak RSS baseline →
761 MB working set with mmap, zero-copy mechanism confirmed). Default
remains the legacy copy path; flip is queued for after the F16
motivating case is timed end-to-end and the diff harness on F16 GGUFs
has been exercised. Metal path is untouched (Metal's
`MTLBuffer-newBufferWithBytesNoCopy` plumbing is a separate, larger
follow-up that's not needed for the diff-harness blocker).

**Pending follow-up:** F16 mimo-asr RSS measurement, default-flip
commit, then PLAN #51c can land.

### ~~51b. Step-decode KV cache reuse~~ — **DONE (May 2026)**

Step-only graph variant + cached step graph + diag-capture gate
shipped — JFK transcript still gold byte-for-byte; per-step
decode 1.46× faster (1.15 s → 0.79 s on M1+Metal+Q4_K). Full
write-up in [HISTORY.md §60](HISTORY.md). `MIMO_ASR_BENCH=1`
env exposes prefill/decode timing.

### 51c. F16 step decode

Q4_K dequant on every matmul is the largest single cost at decode
time. F16 weights are ~2× larger but skip the dequant loop
entirely.

**Status (May 2026): code path works, validation deferred to a
larger-RAM box.**

PLAN #51a's CPU mmap loader landed (commit `9710f80`) — Metal
mmap loader landed too (same commit) — and #60a added the
`posix_madvise(WILLNEED)` readahead hint (commit `f1f4bce`).
Together these mean **no code change is needed for 51c** — just
point `crispasr` at the F16 GGUF with `CRISPASR_GGUF_MMAP=1`. We
verified the load path works (no OOM, mmap'd weights at 1.9 GB
RSS on a 16 GB box, prefill compute starts).

What we couldn't validate end-to-end on this box:

- **JFK transcript byte-equality on F16**: prefill compute
  thrashes because the 16 GB F16 working set doesn't fit in 16 GB
  RAM. Pages get evicted as compute walks layers, every
  re-access faults from the disk5 external (99% full, often
  contended by other workers). One bench attempt ran for 51 min
  with 0.1% CPU and never finished prefill.
- **Decode speedup measurement**: same root cause — needs warm
  cache, which we can't achieve.

The ceiling is **hardware, not code**: 16 GB F16 weights need
≥20 GB RAM to comfortably fit + leave headroom for activations +
KV cache + audio tokenizer. On a 32+ GB box this should "just
work" and hit the work order's ≥1× realtime target.

Files **not** touched (no code change required):
- `src/mimo_asr.cpp` — the runtime is dtype-agnostic; F16 weights
  flow through the existing `core_attn::kv_self_attn` matmul kernels
  on Metal without modification.
- `src/core/gguf_loader.cpp` — already wired (60a + #51a).

Validation deferral notes:
- Run `CRISPASR_GGUF_MMAP=1 ./build-ninja-compile/bin/crispasr --backend mimo-asr -m /path/to/mimo-asr-f16.gguf --codec-model /path/to/mimo-tokenizer-q4_k.gguf -f samples/jfk.wav` on a 32+ GB box to validate transcript + bench.
- If F16 prefill hits ≥1× realtime as predicted, ship the F16
  GGUF as the recommended quant and demote Q4_K to a memory-tight
  fallback. Until then both are shipped on `cstr/mimo-asr-GGUF`
  with Q4_K as the default.

Effort: **0 LOC** (validation only). The originally-scoped
"Effort: Small" assumed code work that turned out to be unneeded
once the mmap loader landed.

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
- **Status (May 2026):** **base + CustomVoice + VoiceDesign 0.6B/1.7B all live** — talker forward, ICL prefill, code-predictor sampling, codec decoder, ECAPA speaker_encoder forward, codec encoder forward all DONE. ASR roundtrip word-exact across all variants. Open: only the **performance pass** below.
- **Shipped milestones** (commit references in HISTORY §57/§58 + per-model status table under #57):
  1. ✓ Talker forward (28L Qwen3 + Q/K-norm + flash-attn + F16 KV cache) — `talker_logits` cos=1.000000 (`2b85b78`).
  2. ✓ ICL prefill builder — `talker_logits_via_icl` cos=1.000000 (`b939d4f`).
  3. ✓ Code predictor with sampling — fixed silent-output trap (`9608202`, `69c135c`).
  4. ✓ TTS→ASR roundtrip on parakeet-v3.
  5. ✓ Codec decoder (Tokenizer-12Hz) — diff harness 8/8 PASS at cos≥0.999983 (`d1f47b1`, `48c6c1a`). Required a Metal `kernel_conv_transpose_1d` patch in our ggml fork (input-range tightening — see LEARNINGS, MUST RE-APPLY on every ggml bump).
  6. ✓ ECAPA speaker_encoder runtime forward — cos=0.999999 (`c0a9cb3`, `8a4c49e`, `38040b4`). C ABI: `qwen3_tts_compute_speaker_embedding(audio, n, sr)` + `qwen3_tts_set_voice_prompt[_with_text]`.
  7. ✓ Codec encoder runtime forward — diff 3 stages cos≥0.999 (`ef11c01`, `10302b4`). Closes the bake-script loop.
- **Performance pass (in progress, partial wins shipped).** Quiet-bench Q8_0 0.6B with all defaults: ~96 ms/frame (talker ~49 + cp ~45). Real-time at 12.5 fps = 80 ms/frame, so ~16 ms/frame still over budget; talker compute is the dominant remaining cost. Shipped: **`QWEN3_TTS_O15=1` is default-on** (commit `5e21e4a`) — cp graph reuse saves ~14 ms/frame on cp_pred under contention, ~2-3 ms/frame quiet, bit-identical WAV. Gated, byte-identical, kept default-OFF: `QWEN3_TTS_FUSED_QKV=1` (talker fused QKV, F16/F32 only, no clean quiet bench yet); `QWEN3_TTS_LK_BUCKET=1` (talker Lk bucketing, **net loss on M1 Metal Q8_0** — see LEARNINGS); `QWEN3_TTS_CP_STEP0_CACHE=1` (cp T=2 step-0 graph cache, ~1-3 ms/frame quiet savings, bit-identical). Investigated: Q8_0 KV cache — blocked on Metal `cont(Q8_0)` source (only F32/F16/BF16 sources supported); needs Metal kernel patch or KV layout restructure to land. Still open: F16 FUSED_QKV clean quiet-machine bench (the existing impl + bench harness needs a contention-free run to land a default-flip decision); Q4_K talker fused QKV; the larger lift of fusing 15 cp steps into one graph (needs on-device top-k sampling, ~3 ms/frame upper bound after O15 since most overhead is already gone).
- Debug knobs: `QWEN3_TTS_{BENCH,DEBUG,DUMP_DIR}` env vars; diff harness via `tools/reference_backends/qwen3_tts.py` + `crispasr-diff qwen3-tts`.
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

## ~~54. granite-speech-4.1 plus / nar variants~~ — **DONE (May 2026)**

All three variants (`granite-4.1`, `granite-4.1-plus`, `granite-4.1-nar`)
shipped with bit-exact JFK transcription; full pipeline writeup +
quant table + HF URLs in [HISTORY.md §61](HISTORY.md). Speaker labels
+ word-level timestamps for the `plus` variant remain queued as a
~50 LOC chat-template follow-up.

<details>
<summary>Old per-variant breakdown (kept for reference)</summary>

The `ibm-granite/granite-speech-4.1-2b` family ships three variants
with significantly different decoders despite the shared "4.1-2b"
naming. All three runtimes are now fully supported and bit-exact on
JFK; HF upload landed.

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

</details>

---

## ~~53. Two narrow extractions for shared TTS-codec patterns~~ — **DONE (May 2026)**

`core_act::snake_beta` + `core_convt::convt1d_crop` shipped, qwen3-tts
codec and SNAC both delegate. Bit-equivalent: SNAC `crispasr-diff`
8/8 PASS (cos_min 0.999941 unchanged). Original ambition of a
`core/audio_decoder.h` super-helper rejected because the convergence
across our four TTS decoders (vibevoice σ-VAE, qwen3-tts codec, mimo
encoder-only, SNAC, kokoro istftnet) wasn't there once the code was
read end-to-end. See [HISTORY §63](HISTORY.md) for the full writeup.

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
4. ~~**Diff harness reference backend.**~~ **DONE — phonemizer-step
   diff (May 2026).** The model-side reference dumper at
   `tools/reference_backends/kokoro.py` already covered the 16 model
   stages; the phonemizer step is now covered by a separate sibling
   tool `tools/check_kokoro_phonemizer_parity.py` that exercises the
   newly-exposed `kokoro_phonemize_text_{lib,popen}` C ABI on a fixed
   `(lang, text)` suite (en / de / fr / ru / cmn / ja / it / es / pt)
   and reports drift between the two paths. Default mode normalises
   away the documented benign U+200D ZWJ tie chars (LEARNINGS §6);
   `--strict` does byte-exact comparison. Initial run surfaces 1 real
   substantive divergence in cmn (`ni2χˈɑu2` vs `niɜχˈɑ‍u2`) — that's
   #56 #2's symptom, captured automatically now. No-model unit tests
   in `tests/test_python_session.py` cover the symbol export +
   null-args return path.
5. ~~**Optional polish.**~~ **DONE + CROSS-BINDING.**
   `kokoro_phoneme_cache_clear()` + session-scoped
   `crispasr_session_kokoro_clear_phoneme_cache()` ABI exports for
   long-running daemons that resynthesize across many speakers. Wrappers
   landed across all 7 bindings (Python `Session.clear_phoneme_cache()`,
   Rust `Session::clear_phoneme_cache()`, Dart `clearPhonemeCache()`,
   Go `Session.ClearPhonemeCache()`, Java `clearPhonemeCache()`, JS
   `Module.ttsClearPhonemeCache()`, Ruby `Session.clear_phoneme_cache()`).
   No-model unit tests cover the symbol export + null-handle return path.

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

### Phase 1 — DONE

All four Phase 1 variants shipped to HF and registered as backend
aliases:

| Variant | Backend alias | HF repo | HISTORY |
|---|---|---|---|
| Qwen3-TTS-CustomVoice 0.6B | `qwen3-tts-customvoice` | [`cstr/qwen3-tts-0.6b-customvoice-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-customvoice-GGUF) | per-model status table below |
| Qwen3-TTS-CustomVoice 1.7B | `qwen3-tts-1.7b-customvoice` | [`cstr/qwen3-tts-1.7b-customvoice-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-customvoice-GGUF) | — |
| Qwen3-TTS-Base 1.7B | `qwen3-tts-1.7b-base` | [`cstr/qwen3-tts-1.7b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF) | [§57](HISTORY.md) |
| Qwen3-TTS-VoiceDesign 1.7B | `qwen3-tts-1.7b-voicedesign` | [`cstr/qwen3-tts-1.7b-voicedesign-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF) | [§58](HISTORY.md) |

The CustomVoice contract surfaced from a config.json diff: a fixed
`spk_id` token (e.g. `vivian:3065`, `dylan:2878`) is prepended to the
talker prefill instead of an ECAPA forward; the speaker embedding is
just `talker.get_input_embeddings()(spk_id)`. Dialect override on the
`spk_is_dialect` table swaps `language_id` (e.g. dylan → beijing 2074).
Pending: extend `tools/reference_backends/qwen3_tts.py` so
`crispasr-diff qwen3-tts` covers the CustomVoice prefill path
(today's diff coverage is ICL/Base only).

Skipped: **havok2/Kartoffelbox-v0.1_0.65h2** (checkpoint variant of
CC-BY-NC-ND blocked Kartoffelbox-v0.1).

The Kartoffel_Orpheus DE + lex-au-orpheus-de checkpoints rolled into
Phase 2 (per-model status table).

### Phase 2 — talker pattern (qwen3_tts.cpp reuse)

Models with a Llama/Qwen-style AR talker + a small audio-token codec.
The talker forward fits directly into the `core_attn::kv_self_attn` +
`core_ffn::swiglu` pattern that #52 already uses.

- **Orpheus-3B backbone** (`canopylabs/orpheus-3b-0.1-ft` —
  use `unsloth/orpheus-3b-0.1-ft` non-gated mirror in practice;
  llama3.2 license) — Llama-3.2-3B + SNAC codec. New backend
  `orpheus`. **DONE (May 2026, commit `a0982d3`)** — talker AR
  forward + SNAC C++ decode shipped end-to-end; ASR-roundtrip on
  `"Hello, my name is Tara."` returns the input verbatim through
  parakeet-v3. With Orpheus base in, Kartoffel_Orpheus + lex-au +
  the various Orpheus finetunes are checkpoint swaps. Phase 3+
  follow-ups (out of scope for slice (c)): greedy decoding loops
  (ship-default must pass `--temperature 0.6`); Llama-3 RoPE
  freq scaling unimplemented; no `repetition_penalty`; Metal
  first-load is slow (~10-15 min for 6.6 GB f16 due to kernel
  compilation, fast thereafter); non-streaming AR (sliding-window
  protocol from `orpheus_snac.py` is a follow-up).
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

- ~~**Voxtral-TTS**~~ — **BLOCKED, May 2026 license re-survey.**
  Upstream `mistralai/Voxtral-4B-TTS-2603` is **CC-BY-NC 4.0**, not
  Apache 2.0 as previously assumed. The model card states the license
  is inherited from the voice-reference training datasets (EARS,
  CML-TTS, IndicVoices-R, Arabic Natural Audio) which are themselves
  NC, so the constraint is constitutional and can't be cleansed by
  re-quantization. `TrevorJS/voxtral-tts-q4-gguf` tags itself
  Apache-2.0 but that's incorrect. Same blocker class as F5-TTS-German
  / Vevo1.5 below. Moved to deferred.
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
| mistralai/Voxtral-4B-TTS-2603 + all derivatives (mlx-community 4-bit, TrevorJS Apache-2.0-tagged GGUF) | Upstream weights are CC-BY-NC 4.0 inherited from voice-ref training data (EARS / CML-TTS / IndicVoices-R / Arabic Natural Audio). Constitutional, not cleanable. The "use upstream Apache 2.0 weights" plan turned out to be based on a wrong assumption (May 2026 re-survey). |
| KevinAHM/pocket-tts-onnx, Pendrokar/xvapitch_nvidia | ONNX-only, niche, no clear demand. |
| NeuralAudioAI/NA_base, tokenaii/horus | Insufficient public info — re-evaluate if asked. |
| FunAudioLLM/Fun-CosyVoice3-* + ayousanz/cosy-voice3-onnx | License unverified on v3. Earlier CosyVoice generations were Apache 2.0; needs confirmation before committing to CFM solver work for it. |

### Per-model status

| Phase | Model | License | Status | Effort |
|---|---|---|---|---|
| 1 | Qwen3-TTS-CustomVoice 0.6B | Apache 2.0 | **DONE + SHIPPED — runtime spk_id path; 4 ASR roundtrips passed (vivian / aiden / serena / dylan-dialect); registry alias `qwen3-tts-customvoice`; published as [`cstr/qwen3-tts-0.6b-customvoice-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-customvoice-GGUF) (Q8_0 968 MB).** | S |
| 1 | Qwen3-TTS-CustomVoice 1.7B | Apache 2.0 | **DONE + SHIPPED — `small_to_mtp_projection` applied per-step (steps 1..14), ASR roundtrips word-exact on Q8_0/ryan + F16/vivian. Registry alias `qwen3-tts-1.7b-customvoice`; factory dispatch wired. Published as [`cstr/qwen3-tts-1.7b-customvoice-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-customvoice-GGUF) (F16 3.84 GB + Q8_0 2.04 GB).** | S |
| 1 | Qwen3-TTS-Base 1.7B | Apache 2.0 | **DONE — runtime parameterised `spk_enc_dim` (was hardcoded 1024) so the 1.7B's 2048-d ECAPA output stops getting truncated; registry alias `qwen3-tts-1.7b-base` + HF model card landed. ASR-roundtrip word-exact on F16/Q8_0 (clone.wav English ICL). Published as [`cstr/qwen3-tts-1.7b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF) (F16 3.86 GB + Q8_0 2.07 GB).** | S |
| 1 | Qwen3-TTS-VoiceDesign 1.7B | Apache 2.0 | **DONE (commit `bd3eb71`) — natural-language voice description via `--instruct`. New `build_voicedesign_prefill_embeds` mirrors CustomVoice but omits the speaker frame from the codec bridge and prepends an instruct block tokenised as `<\|im_start\|>user\n{instruct}<\|im_end\|>\n`. New C-ABI: `qwen3_tts_set_instruct` + `qwen3_tts_is_voice_design`. ASR-roundtrip word-exact on F16/Q8_0 (parakeet-v3 verbatim modulo terminal punctuation). Published as [`cstr/qwen3-tts-1.7b-voicedesign-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF) (F16 3.84 GB + Q8_0 2.04 GB). 1.7B-only — no 0.6B-VoiceDesign weight release upstream.** | S |
| 2 | Orpheus-3B base | llama3.2 | **DONE (commits `a0982d3` + `a4f7c49` + `1f62647` + `5025150`) — talker AR forward + SNAC C++ decoder shipped; ASR-roundtrip word-exact on `"Hello, my name is Tara."` (parakeet-v3 verbatim). Published as [`cstr/orpheus-3b-base-GGUF`](https://huggingface.co/cstr/orpheus-3b-base-GGUF) (F16 6.6 GB + Q8_0 3.5 GB) + [`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF) (F32 26 MB). Unified Session API + all 6 wrappers wired (`crispasr_session_set_speaker_name`, `n_speakers`, `get_speaker_name`); orpheus default temperature now 0.6f (was 0.0f / greedy / loops). Phase 3+ gaps tracked in slice prose above.** | M |
| 2 | Kartoffel_Orpheus DE natural | llama3.2 | **DONE + SHIPPED — converted + quantized (F16 6.61 GB / Q8_0 3.5 GB / Q4_K 1.87 GB), ASR-roundtrip word-exact on Q8_0/Julian via parakeet-v3 -l de. Published as [`cstr/kartoffel-orpheus-3b-german-natural-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF). Registry alias `kartoffel-orpheus-de-natural` + factory dispatch live (commit `d5b55a7`). 19 fixed German speakers (Jakob, Anton, Julian, Jan, Alexander, Emil, Ben, Elias, Felix, Jonas, Noah, Maximilian, Sophie, Marie, Mia, Maria, Sophia, Lina, Lea).** | XS |
| 2 | Kartoffel_Orpheus DE synthetic | llama3.2 | **IN FLIGHT — same pipeline, downloading safetensors. 4 speakers (Martin/Luca/Anne/Emma) + 12 emotions + 5 outbursts. Registry + factory + README all pre-staged.** | XS |
| 2 | lex-au Orpheus-3B-DE-Q8 | llama3.2 (HF tags Apache-2.0; underlying Llama-3.2-FT) | **DONE — registry alias `lex-au-orpheus-de` added pointing at the existing `lex-au/Orpheus-3b-German-FT-Q8_0.gguf` (3.52 GB). Factory dispatch wired. SNAC companion shared with the base orpheus row.** | XS |
| 2 | gwen-tts-0.6B | MIT | queued — needs weight inspection first | S–M |
| 2 | tada-3b-ml | llama3.2 | queued | M |
| 3 | Chatterbox base | MIT | queued — CFM solver gating | L |
| 3 | Kartoffelbox_Turbo DE | CC-BY-4.0 (gated) | blocked on Chatterbox base | XS |
| 3 | lahgtna-chatterbox-v1 AR | MIT | blocked on Chatterbox base | XS |
| 4 | Voxtral-TTS (Mistral upstream) | CC-BY-NC 4.0 | **BLOCKED — license inherits from voice-ref training data; moved to Deferred. See Phase 4 prose.** | — |
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

## 58. MOSS-Audio-4B-Instruct

[`OpenMOSS-Team/MOSS-Audio-4B-Instruct`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Instruct)
— Apache-2.0, ~4 B params, released 2026-04. First **audio-
understanding** model in the queue (not just ASR): speech, music,
environmental sounds, scene QA, time-aware ASR, multi-step
reasoning. Mandarin + English. The Instruct variant is the entry
point; the family also has 8B and Thinking (CoT) variants sharing
the same architecture.

### Architecture summary (from `config.json`)

- **Audio encoder** — 32-layer Whisper-style transformer trained
  from scratch (not a stock Whisper checkpoint). 1280 d / 20 heads,
  GELU FFN 5120 d, 128 mel bins, max 1500 source positions, sliding-
  window attention with window=100. Output rate 12.5 Hz after
  downsample (rate=8). The novel bit: **cross-layer feature taps**
  at layers 8, 16, 24 (in addition to the final 32) — these are
  carried through the adapter into the LM via DeepStack injection
  (see below).
- **DeepStack adapter** — adapter MLP (8192 d hidden) projects each
  of the 4 encoder taps into LM-embedding space (2560 d) with
  independent weights. The 4 projections are added as residuals
  into LM block inputs at indices 0, 1, 2, 3 (so the encoder's
  multi-resolution features inject continuously through the LM's
  early layers). This preserves low-level prosody / transients
  alongside high-level semantics in a way single-tap projectors
  (qwen3-asr / voxtral / granite-speech) can't.
- **Time-aware tokens** — explicit time-marker tokens are inserted
  between audio frame embeddings at fixed intervals. The LM learns
  "what happened when" natively; supports word-level + sentence-
  level timestamp ASR + time-based QA without a separate aligner.
- **LM** — 36-layer Qwen3 (hidden=2560, 32 Q / 8 KV head_dim=128,
  SwiGLU, RMSNorm, RoPE θ=1 M, max_pos=40 960, vocab=151 936,
  untied lm_head). No sliding window; full attention.

### Effort breakdown

| Component | LOC | Reuse |
|---|---:|---|
| Audio mel front-end (128-bin) | ~50 | `core_mel` |
| 32-layer Whisper-style encoder | ~150 | ~70 % from `qwen3_asr.cpp` encoder |
| Encoder sliding-window attention | ~50 | reuse pattern from `voxtral4b` |
| **DeepStack 4-tap output capture** | ~80 | **new** — needs encoder builder hooks at L8/16/24/32 |
| **DeepStack 4-projection adapter** | ~60 | **new** — 4× MLP, run once after encoder |
| **DeepStack injection into LM blocks 0–3** | ~120 | **new** — adds a fixed-shape residual at `cur` before block-N's first norm |
| Time-marker tokenization | ~100 | **new** — chat template builder + per-frame interval logic |
| Qwen3 LM body | ~50 | full reuse (`core_attn::kv_self_attn` + `core_ffn::swiglu`) |
| Greedy / sampler decode | ~80 | `core_bpe::tokenize_with_specials` + step builder pattern from `mimo_asr.cpp` |
| Converter (HF → GGUF) | ~250 | `models/convert-mimo-asr-to-gguf.py` template |
| Diff harness reference + 6 stages | ~200 | `tools/reference_backends/mimo_asr.py` template |
| Backend wrapper for main CLI | ~120 | `crispasr_backend_mimo_asr.cpp` template |
| **Total** | ~**1200–1500 LOC** | comparable to PLAN #51 |

Headline new helper: a **DeepStack injection block** (probably
`core_deepstack::inject(ctx, cur, projector_w, projector_b,
encoder_tap)`) that's reusable for any future model adopting this
pattern. The 4 projection heads are independent matmul + bias adds
applied to the captured encoder taps; injection is a residual add
at the input of LM blocks 0..3.

### What we'd need to dump from the Python ref

Stage taps for the diff harness:
- `mel_in` `[T_mel, 128]`
- `enc_l8` / `enc_l16` / `enc_l24` / `enc_l32` `[T_enc, 1280]`
  (the four DeepStack taps)
- `adapter_proj_{0,1,2,3}` `[T_enc, 2560]` (post-projection)
- `lm_inputs_embeds` `[T_total, 2560]` (pre-block-0)
- `lm_block_3_in` `[T_total, 2560]` (after the last DeepStack
  injection — this is where a multi-tap bug would show up)
- `lm_last_hidden` + `lm_logits_step0` (standard tail)

Six-to-eight stages, similar to mimo-asr's prefill captures.

### Risks / open questions

1. **DeepStack injection point semantics** — does the projection
   replace the LM block's input or get added as a residual? Need
   to read `processing_moss_audio.py` + the model's `forward()` to
   confirm. If it's a *replace* (not residual), the injection
   builder is simpler but the math is more sensitive.
2. **Time-marker token vocab** — are these dedicated special tokens
   in the Qwen3 BPE, or are they synthesized in the embedding
   space? The vocab=151 936 has slots beyond Qwen3's 151 643 BPE +
   30 special — likely the extra ~263 are time markers.
3. **Sliding-window encoder attention with mask=100** — already a
   pattern (`voxtral4b`), but interacts non-trivially with the
   12.5 Hz downsample. Confirm causal vs bidirectional via Python
   ref hook.
4. **Family extensibility** — 8B variant has the same architecture
   per the README, just bigger LM hidden + layer count. If we
   parameterize by config, all four (4B/8B × Instruct/Thinking)
   share one runtime. Worth doing up front.

### Why "audio understanding, not just ASR" matters here

The 24 ASR-style backends in CrispASR all map audio → text
transcription. None handle "describe the music in this clip", "is
the speaker happy", "summarise this 10-minute meeting", or
"transcribe with word-level timestamps". MOSS-Audio is the first
candidate that covers that ground with an open license (Apache-2.0)
and a reasonable size (4 B → ~2.5 GB Q4_K). Adding it expands
CrispASR's surface meaningfully — analogous to how qwen3-tts
expanded scope to TTS.

### Sequencing

Don't start until:
- mimo-asr perf follow-ups (51a/b/c) are at least scoped — they'll
  inform DeepStack's KV-reuse strategy.
- Orpheus / Qwen3-TTS-1.7B (PLAN #57 phases 1–2) finish — those are
  active sessions and the parallel-worker contention is high.

Probable kickoff: mid-to-late May 2026 if the queue clears.

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


---

## 59. Cross-binding C-ABI parity

The Session API surface for TTS (incl. qwen3-tts Base / CustomVoice /
VoiceDesign variant routing) is fully wrapped across all 7 bindings as
of commit `65e0a61` + the Dart follow-up. **The non-Session ABI (~80
exports) is still C-ABI-only or partially-wrapped on most bindings.**
This entry tracks closing those gaps.

### Coverage matrix (May 2026)

C-ABI exposes 127+ unique `crispasr_*` exports in
`src/crispasr_c_api.cpp`. Coverage by binding:

| Binding | Symbols wrapped | Approx % | TTS Session | Variant detect | Align | Diarize | LID | VAD | Streaming | Punc | Registry | Cache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rust (`crispasr-sys`) | 56 | ~44% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Python (`_binding.py`) | 53 | ~42% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dart (`flutter/crispasr`) | ~25 | ~20% | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Go (`bindings/go`) | 18 | ~14% | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Java (JNA) | 17 | ~13% | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ruby (C ext) | 19 | ~15% | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| JS (emscripten) | 18 | ~14% | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

Rust + Python are the canonical / "full-coverage" wrappers. The other
five track the high-traffic surface (transcribe + TTS) and were swept
together in `4f476c3` (set_speaker_name) and `65e0a61` (set_instruct +
variant detect).

### Capabilities reachable only from C-ABI / Rust / Python

For each, ~3-12 exports + an idiomatic result type per binding:

- **Forced alignment** — `crispasr_align_words`, `align_words_abi`,
  `align_result_*`. Word-level timestamps from a transcript + audio.
- **Diarization** — `crispasr_diarize_segments[_abi]`. Speaker segment
  spans.
- **Language ID** — `crispasr_detect_language[_pcm]`,
  `crispasr_lid_free_cache`. Pre-transcribe LID for routing.
- **VAD** — `crispasr_vad_segments`, `crispasr_compute_vad_slices`,
  `crispasr_stitch_vad_slices`, `crispasr_vad_remap_timestamp`,
  `crispasr_vad_free`. Standalone VAD + slice stitching.
- **Streaming** — `crispasr_stream_open/feed/get_text/flush/close`,
  `crispasr_stream_run_decode`. Online ASR with a step buffer.
- **Punctuation** — `crispasr_punc_init/process/free/free_text`.
  FireRedPunc post-processor.
- **Model registry** — `crispasr_registry_lookup[_abi]`,
  `registry_lookup_by_filename[_abi]`,
  `crispasr_detect_backend_from_gguf`. Backend / file resolution.
- **Cache** — `crispasr_cache_dir_abi`,
  `crispasr_cache_ensure_file_abi`. Auto-download dir + lookup.

### Effort

Per binding ~150-300 LOC (extern decls + idiomatic methods + result
types + smoke test). Five trailing bindings × 9 capability surfaces ×
~30 LOC each ≈ 1.5 kLOC total. Each capability is independent — can
be staged.

Suggested ordering once a consumer asks:
1. Streaming (Go/Java first — common deployment shapes for ASR servers).
2. VAD + alignment (mobile use cases via Dart).
3. Diarization + LID + punctuation (transcription pipelines).
4. Registry + cache (CLI-style consumers).

### When to do this

Not now. The qwen3-tts sweep was justified because PLAN #57 Phase 2
unblocks needed it. Open this section when a concrete consumer shows
up asking for, say, "Java VAD" or "Go streaming". Reference commits
for the pattern: `4f476c3` (TTS surface sweep) and `65e0a61`
(variant detection sweep). Same shape applies to every other capability.

---

## 60. Cross-backend perf tricks (llama.cpp / llamafile ports)

Catalogue of optimizations worth porting from upstream llama.cpp +
Justine Tunney's llamafile, broken into discrete actionable items
(60a, 60b, …). Prioritized for our specific shape: Apple Silicon M1,
16 GB RAM, 7B-class speech-LLMs (MiMo-ASR, qwen3-asr, voxtral4b),
often-contested external disk.

| Item | Status | Tier | Effort | Notes |
|---|---|---|---|---|
| [60a — madvise WILLNEED](#60a-posix_madvisewillneed-on-mmapd-weights) | **DONE** | T1 | done | Async kernel readahead, both mmap branches |
| [60b — wrap_iface forward-compat](#60b-wrap_iface-forward-compat-set_tensor_2d--get_tensor_2d--reset) | **DONE** | T1 | done | 3 delegations added to mmap_wrap_iface |
| [60c — pre-touch / `--preload` flag](#60c-pre-touch--preload-flag) | **DONE** | T1 | done | `CRISPASR_GGUF_PRELOAD=1` page-walks before return |
| [60d — Fused QKV per LM layer](#60d-fused-qkv-per-lm-layer) | **OPEN** | T2 | M | mimo-asr first; qwen3-asr / voxtral4b later |
| [60e — KV cache quantization](#60e-kv-cache-quantization-f16--q8_0--q4_0) | **OPEN** | T2 | M | Long-form ASR enabler |
| [60f — `--mlock` flag](#60f---mlock-flag) | **DONE** | T3 | done | `CRISPASR_MLOCK=1` pins pages after WILLNEED |
| [60g — `MADV_RANDOM` post-prefill](#60g-madv_random-post-prefill) | **DONE** | T3 | done | `core_gguf::mmap_advise_random()` exposed |
| [60h — Linux huge pages](#60h-linux-huge-pages-map_hugetlb) | PARKED | T3 | S | Linux-only; we don't have Linux production targets |
| [60i — Read-only mmap mode](#60i-read-only-mmap-mode) | PARKED | T3 | XS | Per-backend; safety net not yet needed |
| [60j — Speculative decoding](#60j-speculative-decoding) | PARKED | T3 | L | No obvious draft model in our family |
| [60k — GBNF grammar-constrained decode](#60k-gbnf-grammar-constrained-decoding) | PARKED | T3 | M | No structured-output consumer yet |
| [60l — tinyBLAS x86 kernels](#60l-tinyblas-llamafile-specific) | SKIP | T3 | — | Not relevant on Apple Silicon |
| [60m — APE multi-arch binary](#60m-ape-multi-arch-binary-llamafile-specific) | SKIP | T3 | — | Distribution, not perf |
| [60n — CUDA graphs](#60n-cuda-graphs--cuda-specific) | SKIP | T3 | — | We don't ship a CUDA backend |

**Tiers:** T1 = directly attacks cold-start / loader. T2 = decode-time
speedup, work-order calls out. T3 = situational / parked / skip.

**Suggested order for remaining OPEN items:**
60d (Fused QKV, mimo-asr LM) → 60e (KV cache quantization,
shared `core_attn` surgery). Both want a fresh session — see the
hand-off prompt at the end of the May 2026 perf-wave session.

---

### 60a. `posix_madvise(WILLNEED)` on mmap'd weights — **DONE (May 2026)**

**Status:** DONE. Commit `f1f4bce`.

When `CRISPASR_GGUF_MMAP=1` is set, both the CPU and Metal mmap
branches now call `posix_madvise(POSIX_MADV_WILLNEED)` on the entire
mmap'd weight region. Hints the kernel to start async readahead so
the first access during prefill doesn't synchronously page-fault
(~5-10 ms each on a 99%-full external disk we hit during PLAN #51c
F16 testing). Mirrors llama.cpp's `llama_mmap` populate path.

**Files touched:** `src/core/gguf_loader.cpp` (16 LOC, both mmap branches).
**Open follow-up:** Windows `PrefetchVirtualMemory` (Win8+) — see 60c.

---

### 60b. wrap_iface forward-compat: `set_tensor_2d` / `get_tensor_2d` / `reset` — **DONE (May 2026)**

**Status:** DONE.

Added three delegating shims to `mmap_wrap_iface` in
`src/core/gguf_loader.cpp` so any future ggml dispatch through
`set_tensor_2d` / `get_tensor_2d` / `reset` on a wrapped Metal mmap
buffer routes to the inner buffer's optimized path instead of
silently falling back. Each shim guards on the inner iface method
being non-null (all three are optional in the buffer-iface
contract).

---

### 60c. Pre-touch / `--preload` flag — **DONE (May 2026)**

**Status:** DONE (env-gated; no CLI flag yet).

`CRISPASR_GGUF_PRELOAD=1` triggers a page-walk loop right after
the WILLNEED hint in both mmap branches. Each page is touched
through a `volatile` byte read so the kernel faults it in
synchronously; by the time `load_weights` returns, every weight
page is resident.

Trade-off acknowledged: on a 16 GB box loading a 16 GB F16,
preload just *moves* the wait from compute time to load time and
may evict everything else from cache. Useful for benchmarking and
for users with comfortable RAM headroom — see 60f for the
stronger pin form.

**CLI flag deferred.** Add `--preload` to
`examples/cli/whisper_params.h` later if a user actually asks; for
now the env var is the surface.

Linux `MADV_POPULATE_READ` (single-syscall kernel-walk) would be a
tighter implementation than the per-page volatile read; queued as
a tiny follow-up if anyone runs this path on Linux.

---

### 60d. Fused QKV per LM layer

**Status:** OPEN. **Effort: M (~80 LOC + converter + GGUF re-upload).** **Tier 2.**

The work order's "small wins to consider opportunistically" for
PLAN #51 already lists this. Save 2 matmuls per layer × 36 layers
× N steps by writing one `attn_qkv.weight = concat(Q,K,V)` of
shape `[d, (n_q + 2*n_kv) * hd]` and feeding it to
`core_attn::kv_self_attn` via the existing `qkv_w` parameter
(qwen3-tts already uses this path).

Probably another **1.1-1.2× per-step decode** on top of 51b/b'.

**Files to touch (mimo-asr first):**
- `models/convert-mimo-asr-to-gguf.py` — fuse Q/K/V tensors per
  LM layer at convert time (write one combined tensor per layer
  instead of three).
- `src/mimo_asr.cpp` — load `attn_qkv.weight` per LM layer if
  present; pass through to
  `core_attn::kv_self_attn(..., qkv_w=fused, ...)`.
- Re-upload `cstr/mimo-asr-GGUF` with the new layout. Keep a
  fallback in `mimo_asr_init_from_file` that detects unfused Q/K/V
  in old GGUFs and runs the slow path so existing downloads keep
  working.

**Port pattern to:** qwen3-asr (28L Qwen3), voxtral4b (26L Mistral)
once validated on mimo-asr. Same shape; ~30 LOC each.

---

### 60e. KV cache quantization (F16 → Q8_0 / Q4_0)

**Status:** OPEN. **Effort: M (~200 LOC, shared-code surgery).** **Tier 2.**

llama.cpp ships `--kv-quant-type k Q8_0` etc. Halves (Q8_0) or
quarters (Q4_0) KV memory with near-zero quality loss for ASR.
Currently our `kv_max_ctx` for mimo-asr is capped by the
`mimo_asr_kv_init(prompt_groups + max_new + 16)` budget — with F16
KV at 36 layers × 8 KV heads × 128 head_dim × 369 ctx ≈ **~57 MB**.
Q8_0 would drop that to ~28 MB.

Not load-bearing for JFK (11 s). **Essential for hour-long podcast
ASR** where `max_ctx` balloons past 10k groups (~1.5 GB F16 KV).

**Files to touch:**
- `src/core/attention.h` — `core_attn::kv_self_attn` currently
  hardcodes `GGML_TYPE_F16` for the K/V views. Add a `kv_dtype`
  parameter; the read path needs `ggml_dequantize_row` shims for
  Q8_0 / Q4_0.
- Each backend that allocates KV (`mimo_asr_kv_init`,
  `qwen3_asr_kv_init`, `voxtral4b_kv_init`, etc.) — opt into the
  smaller dtype via env (`CRISPASR_KV_QUANT=q8_0`) or per-backend.

Land behind env flag; flip default per backend after diff-harness
cosine validation. Touches ~10 backends shared via `core_attn` —
needs the same broad-backend gauntlet as PLAN #51a.

---

### 60f. `--mlock` flag — **DONE (May 2026)**

**Status:** DONE (env-gated; no CLI flag yet).

`CRISPASR_MLOCK=1` runs `mlock()` on the mmap region right after
the WILLNEED hint (and after preload, when both are set) in both
mmap branches. Failure (typically `RLIMIT_MEMLOCK` exceeded)
prints an actionable warning and continues — pages are still
mapped, just unpinned.

Risky on RAM-tight hosts — pinning a 16 GB model on a 16 GB box
would starve the rest of the system. The opt-in env makes the
sharp edge explicit.

**CLI flag deferred.** Same shape as 60c — add `--mlock` to
`examples/cli/whisper_params.h` if a user actually asks.

---

### 60g. `MADV_RANDOM` post-prefill — **DONE (May 2026)**

**Status:** DONE (helper exposed; per-backend wiring follow-up).

`core_gguf::mmap_advise_random(ggml_backend_buffer_t)` is now
public in `src/core/gguf_loader.h`. It detects whether the buffer
came from either of our mmap branches (CPU `mmap_buffer_iface` or
Metal `mmap_wrap_iface`) by comparing iface function pointers,
extracts the mmap region, and calls
`posix_madvise(MADV_RANDOM)`. No-op on non-mmap buffers and on
Windows.

**Per-backend wiring deferred.** Backends that want the hint can
add a single call between prefill and the decode loop:

```cpp
core_gguf::mmap_advise_random(ctx->buf_w);
```

mimo-asr would call it from `mimo_asr_transcribe` between
prefill and the decode loop; same shape applies to qwen3-asr,
voxtral, voxtral4b, granite-speech, etc. Skipped from this batch
because the perf delta is marginal on Q4_K (weights fit; readahead
cost is small) and impossible to measure on F16 in the current
RAM-tight environment (see PLAN #51c). Land per-backend when a
benchmark on a larger box demonstrates measurable benefit.

---

### 60h. Linux huge pages (`MAP_HUGETLB`)

**Status:** PARKED. **Tier 3.**

mmap with `MAP_HUGETLB` reduces TLB misses ~10% on big models.
Doesn't exist on macOS. Not currently a Linux production target —
revisit if/when we ship a Linux-first packaging target.

---

### 60i. Read-only mmap mode

**Status:** PARKED. **Tier 3.**

Currently `MappedFile` opens with `MAP_PRIVATE + PROT_READ|PROT_WRITE`
(copy-on-write writable). Some backends (parakeet's BN-into-conv
fold) need the writable path. For backends that don't mutate
weights, `PROT_READ` would catch accidental mutation as a SIGSEGV
instead of silent CoW. Could gate per-backend in
`core_gguf::load_weights` later. Safety net, not perf.

---

### 60j. Speculative decoding

**Status:** PARKED. **Tier 3.** **Reason:** no obvious draft model.

llama.cpp's main throughput trick for autoregressive generation:
small "draft" model proposes 4-8 tokens, large "target" model
verifies in one prefill. Typical 2-3× decode speedup with no
quality loss when draft acceptance rate is high.

For our family, no obvious draft model exists for mimo-asr (7.5B,
36L Qwen2). Could investigate using qwen3-asr-0.6B as a drafter,
but it's a different tokenizer / vocab / audio preprocessing
stack. Not worth the complexity unless someone specifically asks.

---

### 60k. GBNF grammar-constrained decoding

**Status:** PARKED. **Tier 3.** **Reason:** no consumer.

Useful for structured ASR outputs (JSON, fixed-format timestamps,
PII redaction templates). Not for raw transcription. Park until a
consumer asks.

---

### 60l. tinyBLAS (llamafile-specific)

**Status:** SKIP. **Tier 3.** **Reason:** wrong target architecture.

Justine Tunney's bespoke x86 quantized matmul kernels. Faster
than llama.cpp's reference quants on certain CPUs. Apple Silicon
Metal kernels are already fast, and our x86 paths are CI-only
(no production users on x86 yet). Skip.

---

### 60m. APE multi-arch binary (llamafile-specific)

**Status:** SKIP. **Tier 3.** **Reason:** distribution trick, not perf.

Cosmopolitan libc — one binary runs on Linux/macOS/Windows. Skip
until we ship a packaged binary to end users (currently we ship
source + per-platform builds in CI).

---

### 60n. CUDA graphs / CUDA-specific

**Status:** SKIP. **Tier 3.** **Reason:** wrong backend.

Doesn't apply to Metal. If we ever ship a CUDA backend (not
currently a target — we use ggml-cuda for build-time only),
revisit.
