# CrispASR — Performance benchmarks

Test audio: jfk.wav (11.0s), Q4_K quantization, greedy decode (`-bs 1`).

---

## Metal im2col: the v0.17 sync silently dropped the batch-1 occupancy win — restored as kernel_im2col_flat, melotts hifigan back to ~1.85x (Apple M1, 2026-08-06)

The v0.17 ggml sync removed `CRISPASR_METAL_IM2COL_OCC` (upstream reworked
the dispatch; see `tools/upstream-prs/SYNC-v0.17-CONFLICTS.md` "re-derive if
regressions appear") — which un-fixed the melotts/piper "P0 hifigan" row
below: every batch-1 Metal conv went back to KH*KW-thread threadgroups. The
re-derivation now lives in the shared fork as **`kernel_im2col_flat`**
(`89a2039d`, branch `sync/upstream-v0.17`; authored against CrispEmbed's
PP-OCR profile, where the same kernel is 2.3x on the recognizer): one thread
per dst element from a `(ceil(OW*CHW/256), OH, N)` grid, predicate
`N*KH*KW < 128 || IC == 1` (also covers the `conv_2d_dw` lowering the OCC
variant missed), `CRISPASR_METAL_IM2COL_FLAT=0` restores the standard
kernel. A first cut with int64-divmod flat indexing was SLOWER than the
broken dispatch — Apple GPUs emulate int64 division.

This pin bump (`a0f7289d` → `89a2039d`) re-activates the win. Measured, M1,
interleaved same-binary pairs:

- **melotts** (en-q8, 3.2 s utterance): hifigan_decode 2026/2019 ms →
  **1097/1094 ms (1.85x)**, synthesize 2315/2318 → 1364/1365 ms (1.7x) on
  the two quiet pairs; ASR round-trip reproduces the text. WAV byte-compare
  is NOT a valid gate for melotts — the VITS flow is stochastic and two
  legacy-arm runs already differ; the round-trip is the gate.
- **moonshine-tiny** (jfk.wav): transcript **byte-identical** flat vs
  legacy, RTF neutral-to-slightly-better (82.0 vs 80.0x / 73.7 vs 70.4x /
  62.5 vs 62.6x).
- **paraformer-zh-q4_k**: transcript **byte-identical**. Wall pairs were
  noise-dominated (0.85–15.1 s spread; other sessions held load at 14–18,
  and an order-reversal control still disagreed with itself) — the
  attributable quantity is the per-op GPU time of its 2 im2col nodes:
  **8.5–9.1 ms/node legacy → 0.4–1.0 ms/node flat (9–20x)**, ~16 ms saved
  per run, <1.5% of wall either way. No mechanism for a regression.
- 1473/1473 unit tests on the bumped pin.

The historical OCC numbers in the melotts row below (2–3x hifigan, 1.65x
moonshine) were measured on the pre-v0.17 variant on a quiet box; today's
moonshine is closer to neutral because the current decode path spends its
time elsewhere. CrispEmbed-side evidence (2.3x PP-OCR rec, 1.6x
layout_detect, 18-fixture byte-identity) is in their `PERFORMANCE.md`.
`tools/upstream-prs/23` should be re-drafted from this kernel for upstream.

## Backend × Optimization matrix

At-a-glance view of which performance knobs each backend supports today,
and where the gaps are. Last refresh: **2026-05-04** (after PLAN §79 —
14-commit session that shipped #69a / #69b / #69e / #72 / #73).

**Legend**: ✓ = supported, opt-in via env var · `F16` = stuck at F16
(quant cache types unavailable; attention path needs migration) ·
`—` = not applicable (no KV cache or no transformer blocks) ·
`·` = applicable but not yet wired (port deferred).

### LLM-decoder ASR (high VRAM, autoregressive)

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | weight residency |
|---|:-:|:-:|:-:|:-:|:-:|
| voxtral4b (4B) | ✓ | ✓ | ✓ | ✓ | gpu |
| voxtral (3B) | ✓ | ✓ | ✓ | ✓ | gpu |
| granite-speech (1B / 4.0 / 4.1 / 4.1-plus / 4.1-nar) | ✓ | ✓ | ✓ | ✓ | gpu |
| gemma4-e2b (5B effective) | ✓ | ✓ | ✓ | ✓ | gpu (FLIPPED §72) |
| mimo-asr (1.4B) | ✓ | ✓ | ✓ | ✓ | gpu (FLIPPED §72) |
| qwen3-asr (0.6B) | ✓ | ✓ | ✓ | ✓ | gpu |
| glm-asr (1B) | ✓ | ✓ | ✓ | ✓ | gpu |
| omniasr-llm (300M) | ✓ | ✓ | ✓ | ✓ | gpu |
| vibevoice (4B ASR mode) | F16 | F16 | F16 | ✓ | gpu |

### Encoder-decoder ASR (medium VRAM, autoregressive)

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | notes |
|---|:-:|:-:|:-:|:-:|---|
| canary (1B) | ✓ | ✓ | ✓ | · | flash_attn_ext default, -17 % on JFK with q8_0/q4_0 |
| cohere (2B) | ✓ | ✓ | ✓ | · | cast-on-read default (13 % faster on 30 s chunks); `CRISPASR_COHERE_FLASH=1` for unchunked long-form (-26 % win at 300 s) — see §5 |
| kyutai-stt (1B) | ✓ | ✓ | ✓ | · | flash_attn_ext native, quant-safe |
| firered-asr (900M) | — | — | — | — | inline AED, no exposed transformer KV |
| moonshine-tiny / streaming | — | — | — | — | tiny decoder, no exposed KV |

### Encoder-only ASR (low VRAM, single forward)

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | notes |
|---|:-:|:-:|:-:|:-:|---|
| whisper (legacy) | ✓ | ✓ | ✓ | — | upstream loader, separate path |
| parakeet (TDT) | — | — | — | — | RNN-T transducer, no KV cache |
| fastconformer-ctc | — | — | — | — | CTC head |
| wav2vec2 / hubert / data2vec | — | — | — | — | CTC heads |
| omniasr (CTC variant) | — | — | — | — | CTC head |

### TTS

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | notes |
|---|:-:|:-:|:-:|:-:|---|
| orpheus (3B + DE / lex-au variants) | ✓ | ✓ | ✓ | ✓ | shared Llama-3 path |
| chatterbox (T3 + CFG cache) | ✓ | ✓ | ✓ | · | uses kv_self_attn natively |
| qwen3-tts (0.6B + 1.7B variants) | ✓ talker | ✓ talker | ✓ talker | · | code-predictor cache stays F16 (separate path) |
| vibevoice (4B TTS mode) | F16 | F16 | F16 | ✓ | KV migration still pending; layer offload routes `tts_lm.layers.<N>.*` |
| kokoro | — | — | — | — | non-AR vocoder, no transformer KV |

### Batched classifier-free guidance (CFG) — the dispatch/bandwidth lever (§214)

Most TTS backends apply CFG by running the conditioned and unconditioned passes
as **two sequential forwards** and blending the outputs. That's correct and
quant-safe, but on a single-token AR step (or per-diffusion-step) it reads the
full weight set **twice** and doubles the dispatch count. Batching cond+uncond
into **one B=2 forward** reads each weight **once** — the lever for any
decode/step that is weight-bandwidth + dispatch bound. gianni-cor/chatterbox.cpp
measured **−42 % on the chatterbox T3 AR decode** (its largest stage) from
exactly this.

**Who batches CFG, and how (full-tree audit, §214):**

| backend | CFG mechanism | batched on GPU? | quant on GPU |
|---|---|:-:|---|
| **s3gen CFM** (chatterbox) | B=2 (`build_graph_unet1d_b2`) | ✓ | dequant q*→F16 GPU-resident (`dequant_cfm_f16`) |
| **chatterbox T3** (§214) | B=2 (`build_graph_t3_kv_b2`) | ✓ | dequant q*→F16 GPU-resident (`ensure_t3_b2_f16_weights`) |
| dia | B=2 encoder only | ✓ | n/a (F16/F32) |
| zonos | two separate KV caches, sequential | · | n/a |
| tada | two sequential B=1 passes | · | n/a |
| cosyvoice3 | two sequential B=1 (**explicitly chose not to batch**) | · | n/a |
| f5 | two sequential B=1 passes | · | n/a |
| voxcpm2 | two sequential B=1 passes | · | n/a |
| kugelaudio | CFG not implemented (TODO) | — | — |

**The Metal quant gotcha (and the fix).** A B=2 matmul against **quantized**
weights on Metal becomes a mat-vec with `ne[1]=1, ne[2]=2` that does **not**
dispatch the PREC_F32 `mul_mv_q*_K` exact-dot kernel the single-token (`ne==1`)
path hits → it requantizes activations and drifts enough to wreck a greedy
sampler (chatterbox T3: token divergence at step 2 → repetition collapse; same
class as native batched-quant CFM NaN). F16 weights are fine batched. **Fix
(only solution in the tree):** dequantize the batched-against weights q*→F16
GPU-resident **once** at first use (host `to_float` → `ggml_fp32_to_fp16_row` →
upload to the GPU backend), then the B=2 matmuls take the correct `mul_mm_f16`
path. Both GPU-batched paths — s3gen CFM and chatterbox T3 — use this. CPU
batches quantized weights natively (exact dot), no dequant needed.

**Chatterbox T3 B=2 status (§214, `CRISPASR_CHATTERBOX_T3_CFG_B2=1`, default OFF):**
greedy-token bit-identical to the legacy sequential path on CPU (all quants) and
GPU+F16; GPU+quant fixed via the F16 dequant above (ASR-roundtrips verbatim).
Also the **first working T3-on-GPU path on Metal** — it rebuilds the step graph
each step, so it sidesteps the §186 Lk-bucket `buffer is nil` crash that breaks
legacy T3-GPU. CPU floor speedup **~34 % ms/tok** (75→50) measured on a contended
M1 (load 8–12, absolute numbers unreliable).

**Per-step build+alloc is negligible — bucketing the B2 graph is a confirmed DUD**
(`CHATTERBOX_BENCH_B2=1`, 2026-06-21): CPU 0.41 ms/step = **0.8 %** of step time
(compute 53 ms/step); GPU+F16 1.55 ms = **0.7 %** (compute 232 ms/step). Per
§208, a cached graph only helps overhead-bound work — it doesn't here, and a
cached B2 graph would risk the §186 GPU bucket crash. Note GPU compute (232
ms/step) ≫ CPU (53 ms/step): **B2 on GPU is not a speed win over the CPU default**
— its value is enabling T3-on-GPU (sidestepping the bucket crash) + the GPU+quant
F16-dequant path. T3 stays CPU-default. See PLAN §214.

### Where the gaps are

1. **Layer offload (`N_GPU_LAYERS`) on encoder-decoder ASR** (canary,
   cohere, kyutai-stt). Their cross-attention layout doesn't have the
   `blk.<N>.*` block-tagged tensors that the layer-split predicate
   recognises. Encoder-decoder offload is its own design problem —
   probably want to offload only the LLM/decoder side, but the tensor
   names (`<arch>.dec.<N>.*` etc.) need bespoke per-backend predicates.
2. **vibevoice quant K/V (both modes)**. The attention path uses the
   `ggml_cpy(K_perm, view_into_kv_k)` pattern that's incompatible with
   quant K/V (see LEARNINGS.md "ggml_cont(ggml_permute(quant_tensor))
   doesn't move data"). Migration recipe is the canary/cohere
   `ggml_flash_attn_ext` port — ~50-80 LOC + F16 mask graph input.
   Layer offload (`N_GPU_LAYERS`) is independently shipped and works
   on F16 K/V; the migration only unlocks quant K/V on top.
3. **qwen3-tts code-predictor cache**. Talker KV is fully covered via
   `core_attn::kv_self_attn`; the secondary code-predictor path
   doesn't go through that helper, so its cache stays F16. Lower-
   priority since the talker dominates per-frame cost.
4. **Linux/CUDA validation of #72 GPU residency.** mimo-asr 22 % /
   gemma4-e2b 2.2x speedups were measured on Apple Silicon Metal.
   dGPU should be even more favourable; deferred until a CUDA host
   is available. If a platform regresses, gate via env
   (`CRISPASR_FORCE_CPU_WEIGHTS=1`).
5. **Cohere flash_attn_ext: crossover confirmed (PLAN #73 closeout,
   2026-06-04).** Long-form rerun on FLEURS EN, VPS x86 CPU, 2
   threads, `cohere-transcribe-q4_k.gguf`:

   | audio | flash-attn (s) | cast-on-read (s) | delta |
   |---|---:|---:|---|
   | 60 s | 202.72 | 179.13 | flash **+13% slower** |
   | 300 s | 820.37 | 1114.96 | flash **-26% faster** |

   Crossover is between 60 s and 300 s. Flash wins decisively on
   unchunked long-form (5+ min) due to O(n) vs O(n²) attention scaling.
   But with default 30 s auto-chunking, each decode pass is short-form.
   **Recommendation:** cast-on-read is now the cohere default (13%
   faster on the chunked path that all normal users hit). For unchunked
   long-form, set `CRISPASR_COHERE_FLASH=1`.

### Stacking the four knobs

Each addresses an independent bottleneck:

| knob | addresses | when to use |
|---|---|---|
| `CRISPASR_KV_QUANT_K=q8_0 / _V=q4_0` | KV size in VRAM | always reasonable for LLM-decode ASR; quartered V cache on long context |
| `CRISPASR_KV_ON_CPU=1` | KV doesn't fit in VRAM at all | very long context with a tight VRAM budget |
| `CRISPASR_N_GPU_LAYERS=N` | model itself doesn't fit in VRAM | model size > VRAM; spill the last (total-N) layers |
| `CRISPASR_FORCE_CPU_WEIGHTS=1` (proposed) | platform regressed on §72 GPU residency | not yet wired — none seen on Apple Silicon |

```bash
# Maximum-memory-savings combo for a VRAM-tight host
CRISPASR_N_GPU_LAYERS=10 \
  CRISPASR_KV_ON_CPU=1 \
  CRISPASR_KV_QUANT_K=q8_0 \
  CRISPASR_KV_QUANT_V=q4_0 \
  ./build/bin/crispasr --backend voxtral4b -m auto -f long.wav
```

See [`docs/cli.md`](docs/cli.md) "Memory footprint" for the full env-
var reference and the llama.cpp parity comparison table; HISTORY §79
for the implementation write-up.

---

## Runtime Optimization Audit — Re-verification (2026-07-11)

Full re-sweep of the ASR + TTS + codec + pipeline paths against the matrix and
the §232 campaign. Verified against current code, not carried from this doc.
**The matrix and per-model TTS section are stale on several flash/CFG claims**
— corrections below.

### Stale claims corrected

- **Flash attention is NOT missing across AR decoders.** The per-model TTS
  section lists Orpheus/OuteTTS/Zonos/TADA/Chatterbox/CSM as flash-"stub" or
  "not wired". In fact they route through `core_attn::kv_self_attn`
  (`src/core/attention.h:665,903`), whose body **unconditionally** calls
  `ggml_flash_attn_ext` (no soft_max fallback). The per-file grep-count of 0 is
  because the call lives in the shared helper. The Tier-1 "wire flash in all AR
  decoders" backlog item is largely obsolete.
- **The real remaining flash gaps** are only the manual-`ggml_soft_max`
  backends — **dia, speecht5, parler** — and the **structurally-can't-flash**
  additive-relpos models **melotts, piper** (same constraint as T5).
- **chatterbox flash "stub"** — false: `chatterbox.cpp:1806,2662`, default ON.
- **tada "two sequential B=1 FM passes"** — false: opt-in B=2 graph exists
  (`tada_tts.cpp:238`).
- **Batched TDT/RNNT decode (`CRISPASR_TDT_BATCH` / `CRISPASR_RNNT_BATCH`,
  default OFF) is a MEASURED CPU LOSS for parakeet TDT — keep it OFF, do NOT
  flip.** The "370 sgemv → 26 sgemm" framing predicted a CPU win; the #81 A/B on
  a clean Kaggle CPU (P100 box, `chr1str/crispasr-issue81-onnx-bench`,
  2026-07-18) measured the opposite on parakeet-tdt-0.6b, transcript
  byte-identical in both arms:

  | clip | default decode | `TDT_BATCH=1` | ratio |
  |------|---------------|---------------|-------|
  | jfk 11 s   | **4.0×** RT | 3.2× RT | 0.80× (20% slower) |
  | long 134 s | **3.0×** RT | 2.2× RT | 0.73× (27% slower) |

  Why: batching the joint over ALL T encoder frames does strictly more work than
  TDT's duration-skipping greedy decode, which visits far fewer frames. One big
  sgemm does not pay for the extra frames. (parakeet-ctc is unaffected, 0.99× —
  it never enters the TDT decode.) The batched fn stays available as an opt-in
  and is still used by the long-form streamed path; the short-form default
  (parakeet.cpp:3607/3984) must stay per-frame greedy.

### Verified state

- **Fleet FASTCONV campaign (2026-07-16, `docs/perf-sweep/PLAN.md`)** — shared
  `core_dac::fastconv_cache` + fc-aware `conv1d`/`res_unit`/`dec_block`/
  `build_decode_graph`, and a `core_hifigan::forward`/`conv1d` overload, so
  codec/vocoder backends kill the per-graph F16→F32 conv cast from ONE
  implementation. Wired + A/B-verified (byte-identical, seed-isolated), all
  default ON: omnivoice, irodori (`CRISPASR_IRODORI_FASTCONV`), zonos
  (`CRISPASR_ZONOS_FASTCONV`), **speecht5 (`CRISPASR_SPEECHT5_FASTCONV`, 74 F16
  kernels)**, **chatterbox_s3gen (`CRISPASR_S3GEN_FASTCONV`, 275 F16 kernels,
  split-load-aware pointer-swap; ON vs OFF @seed42 = 0/32768)**. Model-free unit
  test `test-fastconv` (210 assertions).
  - ⚠ **Coverage triage (GGUF-parsed, don't trust the doc):** FASTCONV only
    engages on **F16** conv kernels (the cast it kills). The HiFi-GAN overload
    "sets up 3 backends" was over-optimistic — only **speecht5** ships F16 by
    default. **fastpitch** default is q8_0 (F32 kernels → no-op; only its
    non-default `-f16` variant is F16, and that GGUF hits a pre-existing loader
    bug). **bananamind** ships only q8_0 + f32 (no F16 variant at all → can never
    engage; not wired). Grep audit found only 2/26 codec backends had FASTCONV
    before this campaign — the earlier "landed for 5 backends" was an overclaim.
  - **Not-yet:** chatterbox's 175 K=1 kernels could also take the im2col→matmul
    trick (a further win beyond cast-kill), but it changes reduction order so it
    needs its own A/B on the drift-prone GPU path — deferred.
- **§232 TTS campaign** (persistent sched-free graph + batched CFG cond+uncond +
  device KV + FASTCONV codec) has landed for qwen3-tts, voxtral-tts, omnivoice,
  tada, chatterbox. Un-migrated: f5, dots, kugelaudio, pocket (natural next
  targets — same levers).
- **omnivoice codec FASTCONV actually landed 2026-07-15** (`6a1b1903b`) — its DAC
  decode had NOT been migrated (it ran `core_dac::conv1d` with F16 kernels, paying
  the per-graph F16→F32 cast + im2col-for-k=1). Profiling caught it: decode was the
  RTF wall (18.2 s for the reporter's paragraph on M1). Baking F32 decode kernels at
  load + k=1→matmul (`OMNIVOICE_CODEC_FASTCONV`, default ON) → **decode 10.6 s → 3.6 s
  ≈ 2.9×**, ASR-identical. Corroborated by `rockerritesh/omnivoice-tts.cpp` (their fast
  Metal decode is the same f32-no-cast path). GPU codec decode measured slower on M1
  Metal (dispatch-bound) → gated `OMNIVOICE_CODEC_GPU`, default OFF, pending Kaggle CUDA.
  `--tts-steps` now drives omnivoice's stage0 step count (default 32).
- **omnivoice single-shot synthesis (2026-07-16):** omnivoice was sentence-chunked by
  the generic TTS path; a reporter (#254, CUDA) showed that cost 15–20% vs omnivoice.cpp
  (3 chunks → 3 graph builds + 3 CUDA-graph warmups + 3×num_steps). Added omnivoice to
  the single-shot whitelist (masked-iterative → whole span in one pass, like the
  reference). M1 A/B: net faster even here (decode 9.1 s vs 23.7 s from 1 vs 3 decode
  graph builds outweighs +6% gen O(T²)); on CUDA the gen graph-reuse is a bigger win.
  Escape hatch `CRISPASR_OMNIVOICE_CHUNK=1` restores chunking. Interval-CFG
  (`OMNIVOICE_CFG_INTERVAL=K`) recomputes uncond every K steps — K=2 ≈ −30% stage0,
  opt-in/approximate.
- **omnivoice fused stage0 step graph (2026-07-16, default ON):** the residual #254
  gap vs omnivoice.cpp was per-step HOST overhead (≈18 MB embed readback + CPU
  codebook sum + 5 MB re-upload, full-seq 39 MB logits readback, single-threaded
  ~13M-exp CFG scoring). Fused graph: ids-only upload (~140 KB), in-graph embeds
  (get_rows + cb-order adds, bitwise == host), target-slice-only logits, threaded
  scoring (rng-order preserved). Codes byte-identical to legacy on M1 Metal (5 config
  classes) and CUDA (reporter cmp). **Reporter's RTX 5070 Ti: gen 3.55 s → 1.53 s
  (2.3×), RTF 0.17 → 0.07 for the 21.8 s paragraph — ~2× faster than omnivoice.cpp
  (0.144), single CUDA-graph warmup.** Per-step: fwd 27.4 ms + score 11.6 + read 5.2
  + sample 1.5 = 45.9 ms. `OMNIVOICE_FUSED_STEP=0` restores the legacy path.
- **Codec decoders are already ggml-graph** (snac/dac/seanet/hifigan/adaln/
  qformer). Scalar survivors: `core/rvq.cpp` encode-search and `core/istft.h`
  O(N²) IRFFT (both run once/synthesis).
- **Encoder-graph caching is a closed chapter** — disabled in 8 backends for a
  GPU heap-use-after-free (#235) *and* independently a measured dud on
  compute-bound encoders. The one working cache is nemotron's dimension-keyed
  streaming cache.
- **#218 quant floor**: audio towers floored to Q8_0 (cos 0.97→0.9998 fixes
  long-form loops/empty output); quantizing the **tied LM head** re-introduces
  loops even with a Q8 tower — plain `-q4_k`/`-q8_0` are the long-form
  recommendation, imatrix variants for short clips only. No inference-time cost.
- **CUDA-graph capture (§210)** gives ~9–13× on RTX decode; Metal lacks the
  equivalent (no ICB in ggml-metal).

### True remaining gaps (2026-07-11)

| P | Area | Gap | Impact |
|---|---|---|---|
| **P0** | firered_asr | Decoder self-attention has **no KV cache** — growing vector, O(T²) recompute (`firered_asr.cpp:2697`) | Highest-impact ASR gap |
| ~~P0 relpos~~ → ~~P0 hifigan~~ → **RESOLVED ~2× (im2col batch-1 occupancy)** | melotts / piper | relpos is a MEASURED DUD (`2026-07-11`): already GEMM'd, ~1.5% of synthesis. **hifigan_decode is 73–92%**. The prior "99.7% GPU-compute-bound, no cheap lever, do not fund without a new kernel insight" verdict was based on the IDEAL roofline (im2col ≈ 1%) — **wrong**. Per-node profiling (new `CRISPASR_METAL_PROFILE=2`) on a **quiet** M1 (`2026-07-11`) showed **IM2COL = 58%**, CONT 22%, MUL_MAT only 12% (the earlier 4.02e6 gpu_us was ~2× load-inflated; true quiet ≈ 1.75e6 = 13× off roofline). **Root cause:** `ggml-metal-ops.cpp` sized im2col thread-dim0 from batch N — at inference **N=1**, threadgroups ran only KH·KW (3–11) threads = ~10–34% of one simdgroup → im2col ~40× below BW. **Fix (ggml fork, `CRISPASR_METAL_IM2COL_OCC`, auto-on N==1):** block OW across thread-dim0 → threadgroups fill. **Bit-exact** (moonshine ASR occ-on==occ-off byte-identical; melotts ASR roundtrip valid), **~2–3× on hifigan** (gpu 1.88→0.83 s quiet) + **1.65× moonshine, no regression across paraformer/melotts**. A general batch-1 Metal conv win (whisper/all vocoders); CUDA unaffected (parallelizes threads over IC·KH·KW, not N). The existing `ggml_conv_2d_direct` is a **dud** (2.25× slower, naive scalar). Teardown assert FIXED (`119ec75a`). | Was "compute floor"; the floor was a Metal occupancy bug — fixed |
| **P0** | voxcpm2_tts | CPU-only (Metal SIGSEGV), manual per-step host KV re-upload (`voxcpm2_tts.cpp:106-111`) | GPU-locked-out |
| ~~P0~~ | openvoice2 | STFT scalar O(bins·win) DFT → shared radix-2 FFT: **1182 ms → 10.7 ms** (110×, ~26% of convert) `2026-07-11`. WaveNet already GEMM'd (§176d); ref-enc is 4% one-time. **Remaining: hifigan_decode is 67% of convert** — next target | STFT fixed; vocoder dominant |
| **P1** | voxtral/voxtral4b enc, mimo LLM decoder | Attention not on flash_attn_ext (O(T²) manual softmax) | Enc mem+dispatch; mimo dispatch-bound |
| **P1** | dia / speecht5 / parler | Manual soft_max + host KV re-uploaded per step | Long outputs |
| **P1** | firered/glm/funasr/qwen3/omniasr/mimo | Beam search is replay (no KV snapshot pool; canary/moonshine/kyutai have one) | beam≥2 quadratic |
| **P1** | f5/dots/kugelaudio/pocket | CFG serial / no persistent graph — un-migrated §232 targets | ~halves DiT time |
| **P1** | granite/moss Metal decode | Per-op dispatch ~100ms/step; ggml-metal has no ICB replay | Dominant Metal decode cost |
| **P2** | Scalar CPU hotpaths | RNN-T LSTM pred+joint; granite cpu_linear+depthwise; paraformer CIF; rvq encode; istft IRFFT; titanet mel front-end; diarize `apply_xcorr` | Per-token/frame scalar loops |
| ~~P2~~ **CLOSED** | parakeet | Batched TDT decode validated on Kaggle CPU (#81, 2026-07-18): **0.73–0.80× = SLOWER**, byte-identical → keep default OFF, do NOT flip | Predicted win, measured a loss |
| **OPEN** | parakeet CPU vs onnx-asr | crispasr parakeet-tdt CPU 4.0×/3.0× vs onnx-asr int8 8.6×/5.8× ⇒ **~2.1× slower on CPU** (the real #81 residual). Needs a CPU BLAS/kernel lever — batched decode is ruled out | GPU is fine (P100 36.9×/50.9×) |
| **P2** | align_wav2vec2_ctc | **Reloads the 300MB–1GB model every call** (`crispasr_aligner.cpp:315`) — missing the §176e ctx-cache | Concrete single-file win |
| **P2** | paraformer / voxcpm2 | CPU-only, no GPU backend (paraformer leaks 256MB buffer) | GPU offload available |
| **P3** | Threading | Hardcoded default 4 threads in ~90 sites; only whisper-core caps to `min(4, hw)` | Idle cores on big hosts |
| **P3** | Misc | pyannote per-slice not once-over-audio (#107); RNNoise recreates state+resamplers/call; glm mel padded to 3000 always | Localized |

### Highest-ceiling paths forward

### dots.tts baseline and optimization queue (#319, 2026-07-28)

The first local low-memory benchmark used an Apple M1, Metal, a mixed `Q4_K`
core (about 2.2 GiB; DiT tensors intentionally left in F16) and a `Q8_0`
vocoder (about 330 MiB). With an eight-patch cap, synthesis produced valid,
recognizable audio and measured:

| ODE steps | Total | Flow matching | Vocoder decode |
|---:|---:|---:|---:|
| 16 | 24.2 s | 13.7 s | 10.4 s |
| 8 | 16.2 s | 5.8 s | 10.5 s |

Model loading is excluded. The patch cap makes this a bounded smoke benchmark,
not a real-time claim for full utterances. Eight steps is about 1.49x faster in
this run, but the vocoder becomes the floor once DiT steps are reduced. The
current Metal path already uses the persistent fused DiT graph by default;
remaining high-value work is:

1. Make vocoder decode reuse a fixed-shape graph and benchmark the batch-1
   convolution occupancy fix against the existing HiFi-GAN measurements.
2. Batch CFG conditional/unconditional work with a device-resident dequantized
   weight copy, preserving the quantization safety rule established by
   chatterbox.
3. Add length-bucketed graph reuse for autoregressive patch counts, then measure
   the existing approximate CFG-interval mode separately from exact decoding.

Every dots.tts speed change must retain the audio roundtrip gate (TTS WAV → ASR)
and compare decoded output, not only process exit status. Do not quantize the
DiT tensors to F16 or increase the memory footprint as an optimization.

1. **Lk-bucketed decode-step graph caching** generalized to the 30+ decoders
   that rebuild per step — templates: qwen3-tts (5 buckets), granite §210
   gallocr, mimo `step_t1_gf`. Cache the *decode-step* graph, not the encoder.
2. **ggml-metal ICB replay** — the Apple-side equivalent of CUDA-graph capture;
   decode is per-op-dispatch bound.
3. **BLAS/ggml the scalar hotpaths** (~~melotts/piper relpos~~ — measured DUD,
   already GEMM'd + ~1.5%; ~~openvoice2 WaveNet~~ — GEMM'd §176d; openvoice2 STFT
   DFT→FFT DONE `2026-07-11`; rvq, titanet mel, RNN-T LSTM/joint still open) and
   extend §232's CFG-batch + device-KV playbook to f5/dots/kugelaudio/pocket/dia/
   speecht5/parler. **The real cross-cutting TTS cost is HiFi-GAN decode**
   (openvoice2 67%, melotts 73–92%) — GPU-resident but cold-JIT-confounded on Metal.

Do **not** re-enable encoder-graph caching (#235 UAF + measured dud), and do
**not** CPU-batch decode that feeds a GPU pipeline (item 24).

---

## Kaggle GPU — full backend sweep — 2026-06-20

Platform: Kaggle GPU worker (CUDA), `tools/kaggle-benchmark-all-backends.py`
(kernel `chr1s4/crispasr-full-backend-sweep`). Commit: latest `main`. **First
full-coverage sweep** — every ASR + TTS backend plus the two text-MT backends,
59 entries — with **per-backend results streamed live to an HF dataset**
(`cstr/crispasr-kaggle-progress/full-backend-sweep/latest/`, resumable). Audio:
jfk.wav (11 s). TTS phrase: "The quick brown fox…".

**Headline: ASR 33/35 pass · TTS 14/22 pass · MT 2/2 pass.** (Per-backend JSON +
`summary.json` in the dataset; model-size column is unreliable this run — the
benchmark mis-detected several as ~55 MB — so it is omitted below.)

### ASR — 33/35 pass (RTx, WER on JFK)

| Backend | RTx | WER | Backend | RTx | WER |
|---|---|---|---|---|---|
| SenseVoice Small | 17.9x | 0.0% | Voxtral Mini 3B | 3.3x | 0.0% |
| Canary 1B | 8.3x | 0.0% | Granite Speech 4.1 2B | 3.1x | 0.0% |
| Data2Vec Base | 7.2x | 4.5% | Granite Speech 1B | 3.0x | 0.0% |
| FastConformer CTC | 7.2x | 0.0% | Voxtral 4B Realtime | 2.2x | 0.0% |
| Moonshine Tiny | 7.5x | 9.1% | Kyutai STT 1B | 2.3x | 0.0% |
| Cohere Transcribe | 7.0x | 0.0% | VibeVoice ASR | 1.9x | 4.5% |
| HuBERT Large | 6.8x | 0.0% | Nemotron Streaming | 1.6x | 9.1% |
| Wav2Vec2 XLSR-EN | 6.9x | 0.0% | OmniASR LLM 300M | 1.4x | 4.5% |
| OmniASR CTC 1B | 6.2x | 22.7% | Fun-ASR MLT Nano | 1.1x | 0.0% |
| FunASR Nano | 6.1x | 0.0% | Gemma-4-E2B | 0.9x | 9.1% |
| Parakeet TDT 0.6B | 5.9x | 0.0% | Granite 4.1 NAR | 0.8x | 0.0% |
| Whisper (base) | 5.5x | 0.0% | Granite 4.1 2B+ | 0.7x | 0.0% |
| Qwen3 ASR 0.6B | 5.3x | 0.0% | MOSS Audio | 0.6x | 0.0% |
| GLM ASR Nano | 5.1x | 0.0% | FireRed ASR2 AED | 0.6x | 0.0% |
| Paraformer-zh | 4.5x | 0.0% | MiMo-ASR (CPU #115) | 0.5x | 0.0% |
| Mega-ASR 1.7B | 3.9x | 0.0% | mini-omni2 | 0.8x | 0.0% |
| Granite Speech 4.1 2B | 3.1x | 0.0% | moonshine-streaming | 2.8x | 0.0% |

**ASR failures (2):** `lfm2-audio` — **CRASH** mid-run (~9.8 s) — **FIXED §206**
(embed device-ptr deref + a `ggml_backend_sched` weight-less-first-op cross-backend
copy bug in the backbone; now computes directly on `ctx->backend` via gallocr, GPU
transcribes verbatim); `vibevoice-1.5b` — ran (~17 s) but produced **EMPTY**
transcript. Both are newly-covered backends.

### TTS — 14/22 first pass → 16/22 after the voicefix re-test

First pass marked 8 TTS "fails", but an audit found most were **missing-args, not
bugs**: f5-tts/chatterbox/cosyvoice3/vibevoice-tts are voice-**cloning** models
that need a reference voice, and `vibevoice-1.5b` had been mis-listed as ASR. The
benchmark now passes the right voice per backend (`--voice <ref.wav>
--i-have-rights`, fastpitch `--voice 0`) and `vibevoice-1.5b` moved to TTS. A
fixed-subset re-test (run tag `voicefix-retest`) gives:

| ✓ pass (first pass) | ✓ recovered by voicefix | ✗ genuine fail (with correct args) |
|---|---|---|
| piper, kokoro, pocket-tts, bark, csm, parler-tts, dia, qwen3-tts-customvoice, indextts, zonos, melotts, outetts, tada, voxcpm2-tts | **vibevoice-1.5b** (was run as ASR), **vibevoice-tts** (was no-voice) | speecht5, fastpitch, orpheus, chatterbox, cosyvoice3, kugelaudio · **f5-tts** = runs-but-timeout (pending ≥240 s confirm) |

### MT — 2/2 pass

m2m100 (3.7 s), madlad (12.5 s) — en→de translation produced output.

### Notes

- **Streaming + resume validated end-to-end.** All per-backend JSONs landed in the
  HF dataset as each backend finished; a re-push skips already-streamed backends.
  Root cause of the earlier streaming failures (token unresolved on the chr1s4
  nested mount path) fixed in `81826457`.
- **Genuine CUDA failures: ~7** (was reported as 10), of which `fastpitch` +
  `speecht5` (§204), `chatterbox` (§205), `lfm2-audio` (§206), and `kugelaudio` (§209) are now FIXED; `orpheus` and `cosyvoice3` (dies 0.1 s) remain — tracked
  in PLAN.md §201. Several pass on M1
  Metal, so they are CUDA-path-specific. The 2 vibevoice entries were benchmark
  bugs (now fixed + passing); `f5-tts` needs one more timeout-bumped run.

## Kaggle T4 GPU — 2026-06-03

Platform: Tesla T4 (16 GB VRAM), 4 CPU threads, CUDA. Commit: latest
`main` (post `b102060a`). Run via `tools/kaggle-benchmark-all-backends.py`.
**30 backends tested, 30 pass.** First run to cover all backends added in
the June 2-3 script completeness audit: granite-4.1-plus, granite-4.1-nar,
fun-asr-mlt-nano, voxtral4b.

### Speed ranking (11.0 s JFK, Q4_K unless noted, greedy)

| Rank | Backend | RTx | WER | Architecture |
|---|---|---|---|---|
| 1 | SenseVoice Small | **17.3x** | 0.0% | Encoder (multitask) |
| 2 | FastConformer CTC Large | 7.7x | 0.0% | Encoder-CTC |
| 3 | Data2Vec Base | 6.9x | 4.5% | Encoder-CTC |
| 4 | Canary 1B | 6.8x | 0.0% | Encoder-AED |
| 5 | Moonshine Tiny | 6.7x | 9.1% | Encoder-Decoder |
| 6 | Wav2Vec2 XLSR-EN | 6.4x | 0.0% | Encoder-CTC |
| 7 | HuBERT Large | 6.2x | 0.0% | Encoder-CTC |
| 8 | OmniASR CTC 1B v2 | 6.2x | 4.5% | Encoder-CTC |
| 9 | Cohere Transcribe | 6.1x | 0.0% | Encoder-AED |
| 10 | Fun-ASR Nano 2512 | 5.5x | 0.0% | Encoder-LLM (enc GPU, LLM CPU) |
| 11 | Parakeet TDT 0.6B | 5.3x | 0.0% | Encoder-TDT |
| 12 | Paraformer-zh NAR | 4.3x | 0.0% | Encoder (NAR) |
| 13 | Qwen3 ASR 0.6B | 4.0x | 0.0% | Encoder-LLM |
| 14 | GLM ASR Nano | 3.9x | 0.0% | Encoder-LLM |
| 15 | Mega-ASR 1.7B | 2.6x | 0.0% | Encoder-LLM (qwen3) |
| 16 | Moonshine Streaming Tiny | 2.5x | 0.0% | Encoder-Decoder |
| 17 | Granite Speech 1B | 2.5x | 0.0% | Encoder-LLM |
| 18 | Granite Speech 4.1 2B | 2.4x | 0.0% | Encoder-LLM |
| 19 | Voxtral Mini 3B | 2.3x | 0.0% | Encoder-LLM |
| 20 | OmniASR LLM 300M | 1.4x | 4.5% | Encoder-LLM |
| 21 | Kyutai STT 1B | 1.3x | 0.0% | Encoder-AED |
| 22 | VibeVoice ASR | 1.2x | 4.5% | Encoder-LLM |
| 23 | Voxtral 4B Realtime | 0.9x | 0.0% | Encoder-LLM (streaming) |
| 24 | Granite Speech 4.1 2B+ | 0.8x | 0.0% | Encoder-LLM |
| 25 | Gemma-4-E2B 2.3B | 0.8x | 9.1% | Encoder-LLM |
| 26 | Granite Speech 4.1 NAR | 0.6x | 0.0% | Encoder-CTC (non-AR) |
| 27 | FireRed ASR2 AED | 0.5x | 0.0% | Encoder-AED |
| 28 | Whisper (base) | 0.4x | 0.0% | Encoder-Decoder |
| 29 | MiMo-ASR | 0.2x | 0.0% | Encoder-LLM (CPU-forced, #115) |
| 30 | Fun-ASR MLT Nano 2512 | 0.1x | 0.0% | Encoder-LLM (F16 on CPU) |

### Notes

- **30/30 pass, 24/30 at WER 0.0%** on JFK. The 4.5%/9.1% WER backends
  have minor word-boundary differences (e.g. "americans" → "americas").
- **mimo-asr (0.2x RT):** still CPU-forced (PLAN #115 option A). GPU fix
  landed in `3ef9f87e` (June 2) — needs validation run to flip default.
- **fun-asr-mlt-nano (0.1x RT):** running F16 (~2 GB) on CPU. Q8_0 quant
  exists on HF (`cstr/funasr-mlt-nano-GGUF/funasr-mlt-nano-2512-q8_0.gguf`)
  and should be GPU-safe; switching would recover 5-10x speed.
### TTS benchmark (same run, P100 GPU)

First comprehensive TTS benchmark across 11 backends. Phrase: "The quick
brown fox jumps over the lazy dog." Each model auto-downloaded, synthesised,
output WAV checked for >1 KB, then cleaned up.

| Rank | Backend | Status | Wall (s) | WAV size | Notes |
|---|---|---|---|---|---|
| 1 | Piper LessAC Medium | PASS | 3.7 | 103 KB | Fastest TTS, 22 kHz VITS |
| 2 | SpeechT5 TTS | PASS | 4.6 | 56 KB | 16 kHz, deterministic |
| 3 | Bark Small | PASS | 16.9 | 245 KB | 3-stage GPT-2, 24 kHz |
| 4 | Kokoro 82M | PASS | 17.5 | 152 KB | Needs espeak-ng (installed) |
| 5 | Pocket TTS 100M | PASS | 72.1 | 181 KB | Continuous-latent AR, 24 kHz |
| 6 | CSM 1B | PASS | 213.4 | 165 KB | Llama-3.2 + Mimi, 24 kHz |
| 7 | Orpheus 3B-FT | PASS | 269.7 | 205 KB | Llama-3.2 + SNAC, 24 kHz |
| — | FastPitch 60M | TIMEOUT | >30 | 0 | Timeout too short (30s) |
| — | F5-TTS v1 Base | FAIL | 4.1 | 0 | Needs `--voice <ref.wav>` |
| — | Parler TTS Mini v1.1 | TIMEOUT | >180 | 0 | Too slow on CPU path |
| — | Dia 1.6B | TIMEOUT | >240 | 0 | Too slow on CPU path |

**7/11 pass.** FastPitch needs a longer timeout (model download is slow, not
inference). F5-TTS requires a reference audio for voice cloning. Parler-TTS
and Dia are AR LLM-based and need GPU acceleration or longer timeouts.

---

## Kaggle P100 GPU — 2026-05-31

Platform: Tesla P100-PCIE (16 GB VRAM), 4 CPU threads, CUDA **sm_60**
(auto-detected — the kernel pins `CMAKE_CUDA_ARCHITECTURES` from
`nvidia-smi compute_cap`, so the build is correct whether the box is a
T4/P100/A100/L4). Commit: `7bc3ef5b`. Run via
`tools/kaggle-benchmark-all-backends.py` (now on the shared
`tools/kaggle/kaggle_harness.py`). **27 backends tested, 26 pass.**

First run to cover all six newly-registered ASR backends (marked †):
sensevoice, paraformer, mega-asr, granite-4.1, funasr, mimo-asr.

### Speed ranking (11.0 s JFK, Q4_K unless noted, greedy)

| Rank | Backend | RTx | Time | WER | Architecture |
|---|---|---|---|---|---|
| 1 | SenseVoice Small † | **19.8x** | 0.6s | 0.0% | Encoder (multitask) |
| 2 | FastConformer CTC | 8.9x | 1.2s | 0.0% | Encoder-CTC |
| 3 | Moonshine Tiny | 8.6x | 1.3s | 9.1% | Encoder-Decoder |
| 4 | Canary 1B | 8.0x | 1.4s | 9.1% | Encoder-AED |
| 5 | Data2Vec Base | 7.5x | 1.5s | 4.5% | Encoder-CTC |
| 6 | OmniASR CTC 1B | 7.0x | 1.6s | 9.1% | Encoder-CTC |
| 7 | HuBERT Large | 6.9x | 1.6s | 0.0% | Encoder-CTC |
| 8 | Wav2Vec2 XLSR-EN | 6.9x | 1.6s | 0.0% | Encoder-CTC |
| 9 | Cohere Transcribe | 6.6x | 1.7s | 0.0% | Encoder-AED |
| 10 | Parakeet TDT 0.6B | 6.0x | 1.8s | 0.0% | Encoder-TDT |
| 11 | Whisper base | 5.6x | 2.0s | 0.0% | Encoder-Decoder |
| 12 | Paraformer-zh NAR † | 5.0x | 2.2s | 0.0% | Encoder (NAR) |
| 13 | GLM ASR Nano | 4.7x | 2.4s | 0.0% | Encoder-LLM |
| 14 | Qwen3 ASR 0.6B | 4.4x | 2.5s | 0.0% | Encoder-LLM |
| 15 | Moonshine Streaming Tiny | 2.9x | 3.8s | 0.0% | Encoder-Decoder |
| 16 | Mega-ASR 1.7B † | 2.9x | 3.9s | 0.0% | Encoder-LLM (qwen3) |
| 17 | Granite Speech 1B | 2.8x | 3.9s | 0.0% | Encoder-LLM |
| 18 | Granite Speech 4.1 2B † | 2.7x | 4.1s | 0.0% | Encoder-LLM |
| 19 | Voxtral Mini 3B | 2.5x | 4.4s | 0.0% | Encoder-LLM |
| 20 | OmniASR LLM 300M | 1.6x | 7.0s | 4.5% | Encoder-LLM |
| 21 | Kyutai STT 1B | 1.5x | 7.5s | 0.0% | Encoder-AED |
| 22 | VibeVoice ASR | 1.3x | 8.6s | 4.5% | Encoder-LLM |
| 23 | Voxtral 4B Realtime | 0.9x | 11.9s | 0.0% | Encoder-LLM |
| 24 | Gemma-4-E2B 2.3B | 0.8x | 13.6s | 0.0% | Encoder-LLM |
| 25 | FireRed ASR2 AED | 0.6x | 19.2s | 0.0% | Encoder-AED |
| 26 | MiMo-ASR † | 0.3x | 38.0s | 0.0% | Encoder-LLM (CPU-forced, #115) |
| 27 | FunASR Nano † | ~1.0x | ~10.6s | 0.0% | Encoder-LLM (enc GPU, LLM CPU) |

### Notes

- **funasr is FIXED (§136, 2026-06-01).** The original benchmark showed
  6.0× RT / 1.8s but **100% WER** — the `ggml_backend_sched` produced
  all-NaN logits on CUDA (issue #125). The fix (`f94fec90`) splits
  weights: encoder on GPU, LLM+KV on CPU. Now ~1.0× RT / 10.6s wall-clock
  (including model load) with **0% WER**. Slower than the broken all-GPU
  run because the Qwen2-0.6B decode is CPU-bound. The encoder (70 SANM
  blocks) still benefits from GPU — for longer audio the encoder
  dominates and the GPU speedup matters more. `FUNASR_LLM_GPU=1`
  overrides to all-GPU for future testing once the upstream sched bug
  is fixed.
- **SenseVoice debuts at #1** (19.8× RT, 0% WER) — encoder-only multitask
  model, fastest backend now measured.
- **mimo-asr** runs at 0.3× (38 s) because PLAN #115 forces it to CPU; it
  transcribes correctly. The 420 s timeout budgeted for it was ample.
- P100 (sm_60, ~9.3 TFLOPS fp32) lands the LLM-AR tail a touch faster than
  the 2026-04-26 T4 run (e.g. voxtral-3B 2.5× vs 2.4×, granite-1B 2.8× vs
  1.7×); CTC/encoder backends are comparable. Cross-run deltas are also
  affected by Q4_K model refreshes since April.
- Two non-fatal HF pre-download `401`s (the Kaggle Secrets API was
  flaking, so no HF token) fell back to the C++ downloader; all models are
  public `cstr/*` so downloads still succeeded.

---

## Kaggle T4 GPU — 2026-04-26

Platform: 2x Tesla T4 (15 GB VRAM each), 4 CPU threads, CUDA.
Commit: `b9fd8eb`. **All 19 backends pass.**

### By architecture

#### Encoder-CTC (non-autoregressive, single forward pass)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| FastConformer CTC Large | 120M | 83 | 0.0% | **9.6x** | 1.1s | 18 FC layers |
| OmniASR CTC 1B v2 | 975M | 551 | 4.5% | 7.4x | 1.5s | w2v-BERT enc, 276ms GPU |
| Data2Vec Base | 95M | 78 | 0.0% | 5.3x | 2.1s | 12 layers, pos_conv 735ms |
| Wav2Vec2 XLSR-EN | 300M | 212 | 0.0% | 3.6x | 3.1s | 24 layers, pos_conv 1.6s |
| HuBERT Large | 300M | 212 | 0.0% | 3.6x | 3.1s | Same runtime as wav2vec2 |

#### Encoder-TDT (non-autoregressive, transducer)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| Parakeet TDT 0.6B | 600M | 466 | 0.0% | 5.6x | 2.0s | 24 FC layers + joint net |

#### Encoder-Decoder / AED (autoregressive, attention-based)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| Whisper (base) | 74M | 141 | 0.0% | **9.3x** | 1.2s | Full GPU (upstream) |
| Moonshine Tiny | 27M | 20 | 9.1% | 6.7x | 1.6s | CPU-only, tiny |
| Canary 1B | 1B | 672 | 0.0% | 6.2x | 1.8s | GPU enc+dec, 32+8 layers |
| Cohere Transcribe | 2B | 1440 | 0.0% | 5.2x | 2.1s | GPU enc, AED dec |
| Kyutai STT 1B | 1B | 636 | 4.5% | 1.4x | 7.7s | 24-layer Mimi decoder |
| FireRed ASR2 AED | 900M | 918 | 0.0% | 0.6x | 19.0s | CPU Q4_K SIMD dec (60ms/step greedy; beam batched through Q4_K mul_mat, 16.8s→3.1s at beam=3, §224); enc.* split-loaded to GPU by DEFAULT with use_gpu (transcript-identical, enc 2.2-2.3x on Metal/Vulkan/CUDA-P100, §224; CRISPASR_FIRERED_ENC_CPU=1 opts out) |

#### Encoder-LLM (autoregressive, language model decoder)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| Qwen3 ASR 0.6B | 780M | 515 | 0.0% | 4.7x | 2.3s | 0.6B LLM |
| GLM ASR Nano | 1.3B | 1262 | 0.0% | 4.6x | 2.4s | ~1B LLM |
| Voxtral Mini 3B | 3B | 2530 | 0.0% | 2.4x | 4.7s | Mistral 3B LLM |
| OmniASR LLM 300M | 1.6B | 1018 | 4.5% | 1.7x | 6.4s | LLaMA 1.3B dec |
| Granite Speech 1B | 2.9B | 2805 | 0.0% | 1.7x | 6.4s | Granite LLM |
| VibeVoice ASR | 4.5B | 4589 | 4.5% | 1.2x | 8.8s | ~4B LLM, JSON output |
| Voxtral 4B Realtime | 4B | 2407 | 0.0% | 0.9x | 12.8s | Causal streaming arch (PLAN #7 streaming API; 1.6s first-text-token) |

### Speed ranking

| Rank | Backend | RTx | Time | Architecture |
|---|---|---|---|---|
| 1 | FastConformer CTC | 9.6x | 1.1s | Encoder-CTC |
| 2 | Whisper base | 9.3x | 1.2s | Encoder-Decoder |
| 3 | OmniASR CTC 1B | 7.4x | 1.5s | Encoder-CTC |
| 4 | Moonshine Tiny | 6.7x | 1.6s | Encoder-Decoder |
| 5 | Canary 1B | 6.2x | 1.8s | Encoder-AED |
| 6 | Parakeet TDT 0.6B | 5.6x | 2.0s | Encoder-TDT |
| 7 | Data2Vec Base | 5.3x | 2.1s | Encoder-CTC |
| 8 | Cohere Transcribe | 5.2x | 2.1s | Encoder-AED |
| 9 | Qwen3 ASR 0.6B | 4.7x | 2.3s | Encoder-LLM |
| 10 | GLM ASR Nano | 4.6x | 2.4s | Encoder-LLM |
| 11 | Wav2Vec2 XLSR-EN | 3.6x | 3.1s | Encoder-CTC |
| 12 | HuBERT Large | 3.6x | 3.1s | Encoder-CTC |
| 13 | Voxtral Mini 3B | 2.4x | 4.7s | Encoder-LLM |
| 14 | OmniASR LLM 300M | 1.7x | 6.4s | Encoder-LLM |
| 15 | Granite Speech 1B | 1.7x | 6.4s | Encoder-LLM |
| 16 | Kyutai STT 1B | 1.4x | 7.7s | Encoder-AED |
| 17 | VibeVoice ASR | 1.2x | 8.8s | Encoder-LLM |
| 18 | Voxtral 4B Realtime | 0.9x | 12.8s | Encoder-LLM |
| 19 | FireRed ASR2 AED | 0.6x | 19.0s | Encoder-AED |

---

## CPU-only VPS — 2026-04-24

Platform: x86_64, 4 threads, 7.6 GB RAM, AVX2, no GPU.

| Backend | RTx (CPU) | Time (CPU) | RTx (T4) | Speedup |
|---|---|---|---|---|
| FastConformer CTC | 9.4x | 1.2s | 9.6x | 1.1x |
| Moonshine Tiny | 16.8x | 0.7s | 6.7x | 0.4x* |
| Parakeet TDT 0.6B | 2.9x | 3.8s | 5.6x | 1.9x |
| Canary 1B | 2.7x | 4.0s | 6.2x | 2.2x |
| Data2Vec Base | 2.1x | 5.2s | 5.3x | 2.5x |
| Qwen3 ASR 0.6B | 1.7x | 6.5s | 4.7x | 2.8x |
| Wav2Vec2 XLSR-EN | 1.1x | 9.9s | 3.6x | 3.2x |
| Cohere Transcribe | 1.4x | 7.7s | 5.2x | 3.7x |
| FireRed ASR2 AED | 0.1x | 123s | 0.6x | 6.5x |

*Moonshine runs CPU-only on both (tiny model, no GPU benefit).

GPU acceleration is strongest for encoder-heavy models (2-6x). Decoder-bound
models benefit less (FireRed decoder still runs on CPU even with GPU).

---

## Per-phase breakdowns

### wav2vec2 family (Kaggle T4)

| Model | CNN | Pos conv | Encoder | Total |
|---|---|---|---|---|
| wav2vec2-large (24L) | 215ms | 1588ms | 127ms | 1941ms |
| hubert-large (24L) | 227ms | 1595ms | 128ms | 1960ms |
| data2vec-base (12L) | 221ms | 735ms | 57ms | 1023ms |

**Bottleneck:** pos_conv (grouped conv1d on CPU) = 50-80% of total time.
Encoder graph on GPU is only 57-128ms.

### FireRed AED decoder (Kaggle T4)

| Phase | Time | Notes |
|---|---|---|
| Fbank extraction | ~50ms | CPU |
| Conv2d subsampling | ~100ms | CPU |
| Hybrid encoder (16L) | ~17s | GPU matmuls + CPU attention, slow due to CPU weight copies |
| K/V precompute | 433ms | GPU (scheduler auto-copies) |
| Decoder (28 steps) | 1695ms | CPU Q4_K SIMD, 60.5ms/step |
| **Total** | **19.0s** | Encoder dominates |

### OmniASR (Kaggle T4)

| Model | Encoder | Prefill | Decode | Total | RTx |
|---|---|---|---|---|---|
| CTC 1B v2 | 244ms | — | — | 277ms | 39.8x (encoder only) |
| LLM 300M v2 | 97ms | 803ms | 4028ms (103 steps) | 5021ms | 2.2x |

---

## Key observations

1. **CTC models dominate on speed.** No decoder loop = one forward pass.
2. **Small LLM decoders (0.6-1B) are competitive** — Qwen3 and GLM hit 4.5x+
   realtime with 0% WER, close to encoder-only models.
3. **Large LLMs (3-4.5B) are 1-2x realtime** on T4. Usable but not fast.
4. **Most WER=0% on jfk.wav.** The 4.5% models have minor formatting differences,
   not actual transcription errors. Moonshine Tiny (9.1%) has a real word error.
5. **wav2vec2 pos_conv was the bottleneck** — now 4.9x faster with ggml grouped
   conv (im2col + mul_mat SIMD). Was 1.6s (80% of runtime), now 324ms (~3.5%).
6. **FireRed encoder is slow** because CPU weights auto-copy to GPU per-layer.
   Pre-loading encoder weights to GPU would save ~15s.

---

## Optimization history

### wav2vec2 grouped conv — 2026-04-27

| Path | pos_conv | Notes |
|---|---|---|
| Manual C++ (OMP) | 1588ms | 4-thread OMP, plain float loops |
| **ggml im2col + mul_mat** | **324ms** | **4.9x faster**, SIMD kernels |

The grouped positional conv (C=1024, K=128, G=16) is decomposed into G=16
independent `ggml_pad_ext` + `ggml_im2col` + `ggml_mul_mat` calls. The
mul_mat output `[cpg, T]` is transposed to channel-first before reassembly.
Applies to wav2vec2, data2vec, and hubert.

### FireRed decoder — 2026-04-26

| Path | ms/step | 28 tokens | Why |
|---|---|---|---|
| Manual C++ F32 (original) | 4400 | 123s | No SIMD, no parallelism |
| + OpenMP matmuls | 2320 | 58s | 2.1x from OMP |
| + ggml Q4_K CPU native | **70** | **2.0s** | 9.3x from fused SIMD kernel |
| ggml_vecmat on CUDA | 2600 | timeout | CUDA launch overhead kills it |
| F32 dequant + cpu_matmul | 590 | 16.5s | No SIMD, OMP disabled on Kaggle |
| **ggml_vecmat CPU (final)** | **60** | **1.7s** | Weights on CPU, native Q4_K |

### wav2vec2 CNN — 2026-04-24

| Change | CNN | Total | Speedup |
|---|---|---|---|
| Baseline (manual C++) | 95.2s | 108.4s | 1.0x |
| ggml F32 im2col | 2.4s | 15.5s | 7.0x |
| + OpenMP pos_conv | 2.3s | 9.9s | 10.9x |

### voxtral4b streaming — 2026-05 (PLAN #7 phases 1+1.5+2+3+4)

Native incremental encoder + streaming-prompt decode + speculative
prefill + combined-chunk flush + live captions + decoder thread.
M1 Q4_K JFK 11 s baseline, all variants bit-exact-batch:

| Stage / phase | Metric | Before | After | Δ |
|---|---|---|---|---|
| Phase 1 (initial) | first-text-token | n/a | 2674ms | — |
| + 240ms chunks (phase 2) | feed total | 23s | 9.1s | 2.5× faster |
| + default-unification fix | encoder drain | 2064ms | 1016ms | -1.0s |
| + fused QKV (Q4_K) | per-decode-step | 56ms | 50.4ms | -10% |
| + combined-chunk flush (phase 3) | encoder drain | 990ms | 307ms | -683ms |
| + speculative prefill (phase 3) | first-text-token | 921ms | **650ms** | -271ms |

**Final**: first-text-token 2674ms → **650ms (4.1× faster)**;
sequential live decode (phase 3); decoder thread for non-blocking
feed (phase 4, gated on `CRISPASR_VOXTRAL4B_STREAM_DECODER_THREAD=1`).

The remaining ~410ms gap to the model's ≤240ms target is the
architectural floor: 8 streaming-pad warmup steps × 50.4ms + LLM
prefill = 655ms minimum on M1 Q4_K. Cross that floor only via a
faster Q4_K Metal kernel or a model with a different prompt
convention (no streaming-pad warmup).

Cross-backend portability of the fused-QKV Q4_K pattern:
- qwen3-asr Q4_K: default-on (transcript correct; perf within
  noise on JFK's short-decode shape)
- voxtral 3B Q4_K: opt-in (`CRISPASR_VOXTRAL_FUSED_QKV=1`); A/B
  showed no measurable speedup on JFK
- qwen3-tts: opt-in (existing convention)

---

## Reproduce

```bash
# Per-backend timing
CRISPASR_VERBOSE=1 crispasr --backend firered-asr -m auto -f jfk.wav -v -bs 1

# wav2vec2 phase breakdown
WAV2VEC2_VERBOSE=1 crispasr --backend wav2vec2 -m auto -f jfk.wav -v

# Full Kaggle benchmark (all 19 backends)
# See tools/kaggle-benchmark-all-backends.py or gist:
# https://gist.github.com/CrispStrobe/c15f7a64878d93907a8a4a51b193b806
```


## issue #81 / FastConformer family — CURRENT metrics (2026-07-12)

State on main: conv-pw Q8_0 fix (the quantizer skipped the 3D-stored conv
pointwise weights, leaving them F16 in every quantized GGUF — ~35% of CPU
encoder time) + fused QKV + strided flash-attn inputs; all 80 affected HF
GGUFs re-quantized (transcript-verified); CUDA default = manual attention
(flash_attn_ext rejects the per-head rel-pos mask there and bounced all 24
layers to CPU); Metal/CPU keep flash. Rounds 1-3, the May 2026
cross-comparisons, and all pre-fix numbers are superseded (git history).

**M1 CPU-only** (4 threads — optimal on M1; quiet box; JFK 11 s; medians;
transcripts verified identical):

| model | x-RT |
|---|---|
| parakeet-ctc-0.6b Q8_0, re-quantized GGUF | **18.4×** |
| parakeet-ctc-0.6b Q8_0, old GGUF + runtime repack | 15.0× |
| parakeet-ctc-0.6b Q4_K | 10.3× |
| parakeet-tdt-0.6b-v3 Q4_K | 10.7× |
| stt-en-fastconformer-ctc-large Q4_K | 33× |
| onnx-asr parakeet-ctc int8 CPU-EP (same box, 2026-05 reference) | 15.2× |

**M1 Metal** (shared box, load 4-8 → treat ±10%): parakeet-ctc-0.6b Q8_0
**16.2×**; parakeet-tdt-0.6b-v3 Q4_K 13.8× (decode-bound — no encoder gain,
needs a per-stage look).

**P100 CUDA** (transcripts verified):

| measurement | x-RT |
|---|---|
| CrispASR parakeet-ctc Q8_0, manual attn, warm in-process, jfk×5 55 s | **153×** (11 s: 116×) |
| same, flash-with-CPU-fallback (old default) | 48× (11 s: 61×) |
| CrispASR parakeet-ctc Q8_0, manual attn, **134 s varied, load-excl (honest)** | **137×** (tdt 49.5×) |
| onnx-asr parakeet-ctc CUDA fp32, 134 s varied speech, in-process | **214×** (tdt 121×) |
| onnx-asr parakeet-ctc CPU int8, 134 s varied | 5.8× (tdt 5.7×) |

CUDA rows resolved (2026-07-12, kernel `issue81-onnx-bench` v16, real
134 s varied LibriSpeech, load-excluded, 301-word proof-of-work,
same-run onnx head-to-head): the honest-methodology re-run with the
manual-attn default lands **137× ctc / 49.5× tdt** — a 5.4× jump from the
pre-manual-attn 25.3× ctc / 34.7× tdt (superseded). That cuts the warm
gap to onnx-asr CUDA to **~1.56× (ctc) / ~2.4× (tdt)**, confirming the
handover's "~1.4×" claim on independent honest methodology. (The 153×
in-process row is repeat-audio jfk×5, ~6% cache-friendly and warm —
kept as the best-case in-process figure; the 137× row is the fair
varied-audio comparison against onnx.)

**VPS 4-core x86 re-bench (2026-07-12, DONE)**: parakeet-ctc-0.6b q8_0,
jfk 11 s, same-box A/B (load ~2.5) — new defaults **5.43 s (2.0× RT)**
vs legacy (`CRISPASR_FC_PW_Q8=0 CRISPASR_FC_FUSED_QKV=0
CRISPASR_FC_ATTN_CONT=1`) 6.30 s (1.7×); q4_k new 6.24 s. FC-perf gain
on x86 is **~14%** (vs M1's 35%) — OpenBLAS already handles the F16
conv-pw GEMM well, so the Q8 repack recovers less than on ggml's ARM
CPU F16 path. Still beats onnx-CPU int8 (5.8× above). The re-quantized
GGUF would not change this load-excluded inference RTF — it only saves
one-time load (skips the runtime repack).

Gates: `CRISPASR_FC_PW_Q8` (auto-on for quantized models) ·
`CRISPASR_FC_FUSED_QKV` (on) · `CRISPASR_FC_ATTN_CONT=1` legacy conts ·
`CRISPASR_FC_GPU_MANUAL_ATTN` (auto = CUDA only) · `CRISPASR_FC_BUCKET`
(opt-in bucketed persistent graph, output-equivalent) ·
`CRISPASR_FC_PROFILE=1` per-node profiler · `CRISPASR_FC_MAX_LAYERS=N`
bisection. Kernels: `tools/kaggle/fc-unified-graph-ab` (CUDA A/B),
`tools/kaggle/fc-pw-requant` (fleet requant),
`tools/kaggle/issue81-onnx-bench` (onnx head-to-head).

---

## Long-audio coverage — 2026-07-04 final state (issue #89 closed out)

Platform: Apple M1 16 GB, Metal. Commits `ca1e871b` (VAD slice cap 12 s +
per-slice single-pass) + `3a8141e3` (gap-fill second pass). Scoring:
char-bigram recall/precision vs a whisper-large-v3-turbo reference with
hiragana-reading normalization (pykakasi) — the coverage metric that
erases kanji/kana spelling variants; latin stripped (the JA model renders
English in katakana, which a latin reference can't credit).

| clip | old default (streamed) | shipped default | precision | notes |
|---|---|---|---|---|
| yt_60s (reporter's clip) | 64.2 % | **97.2 %** | 90.5 % | auto-VAD + 12 s cap + gap-fill |
| yt_120s | 61.3 % | **96.9 %** | 91.2 % | |
| first300 (reporter's 300 s repro) | — (June main-default.srt: 0.7 %) | **95.9 %** | 93.4 % | |
| session ABI (yt_60s, bindings/server path) | ~58 % | **94.8 %** | 89.2 % | energy slices instead of silero |

Ceiling calibration: an independent SenseVoice-small run scores 97.2/96.8 %
recall against the same reference at 78 % precision — the shipped pipeline
sits AT the inter-model agreement ceiling; the residual is hearing variants,
not missing content. Blueprint comparison on the same audio: NeMo 2.7.3
plain transcribe 11 %, local-attn [128,128] 46 % (both char-identical to our
port — bit-faithfulness check), buffered BatchedFrameASRTDT 15-51 %.
Gap-fill cost: one extra short encode per recovered gap (~1.3-2× wall on
gap-heavy clips); `CRISPASR_GAP_FILL=0` restores single-pass-per-slice.
Reproduce: `tools/asr_coverage_score.py <whisper-ref> <hyp> --strip-latin
--reading` (scorer) and `tools/nemo_parakeet_blueprint.py` (NeMo modes).

## Long-form single-pass — qwen3-asr / glm-asr (#218, 2026-07-10)

Platforms: Apple M1 16 GB Metal + Kaggle T4 CUDA validation
(`tools/kaggle/qwen3-family-rebake/`). Canonical clip: the reporter's
145 s `t32-145s.wav` (noisy dialogue, quiet lead). Full detail in
PLAN "#218 …" sections; the performance-relevant facts:

### Encoder quantization floor (quality, not speed)

Sub-8-bit `audio.*` towers flip greedy LLM decode into loops/empty output
on long audio — a behavioral step function, not gradual WER loss.
`crispasr-quantize` now floors qwen3-asr `audio.*` at Q8_0 (opt-out
`CRISPASR_QWEN3ASR_QUANT_AUDIO=1`):

| model | q4_k size | encoder cos_mean vs F16 | un-chunked 145 s |
|---|---|---|---|
| qwen3-asr-0.6b, old tower Q4 | 540 MB | 0.9716 (cos_min 0.75 @1885 fr) | loops / "language none" |
| qwen3-asr-0.6b, Q8 tower | 631 MB | 0.9997 (cos_min 0.992) | clean+complete, raw (loop-fix off) |
| qwen3-asr-1.7b, old tower Q4 (#240) | 1334 MB | 0.9913 (cos_min 0.963 @jfk 143 fr!) | empty on Metal |
| qwen3-asr-1.7b, Q8 tower (#240) | 1490 MB | 0.9998 (cos_min 0.9989) | fixed |
| 1.7b-ja-anime, Q8 tower | 1421 MB (was 1334) | — | 1024 chars, max run 1 |
| mega-asr, Q8 tower | not shipped | — | still loops at bf16 too — LoRA-inherent, chunked-only |

imatrix caveat: quantizing the tied LM head re-introduces long-form loops
even with a Q8 tower, and a long-form-recalibrated imatrix still drifts
(tested + rejected) — plain `-q4_k`/`-q8_0` are the long-form
recommendation; imatrix variants are for short clips.

### Memory: full vs windowed encoder attention (qwen3-asr)

Default full self-attention is O(T²) in the audio tower: 145 s
(1885 frames) fits easily on 16 GB; beyond ~10 min un-chunked it OOMs.
KV is sized `max(4096, prompt+max_new+16)` + grow-on-demand (the old
fixed 4096 capped un-chunked audio at ~5 min regardless of encoder).

`CRISP_AUDIO_WINDOWED_ATTN=1` (opt-in, 46e08bc4) runs the FA2/cu_seqlens
block-diagonal semantics natively: full 104-frame windows as ONE batched
unmasked attention + ragged tail, no dense mask → **O(N·W) memory**,
removing the encoder length cap. Encoder cos_mean 0.99953 vs a windowed
bf16 reference. **Default flip evaluated and REJECTED (2026-07-10)**,
raw-decode A/B on M1 Metal (phrase-cycle metric, q4_k-v2):

| clip | mode | wall | peak RSS | max uni-run / cycle | complete |
|---|---|---|---|---|---|
| t32-145s (noisy) | full | ~400 s | 1.65 GB | 1 / 0 | yes |
| t32-145s (noisy) | windowed | 658 s | 1.99 GB | **238 / 119** | **no** |
| jfk×12 132 s (clean) | full | 271 s | 1.99 GB | 1 / 0 | 198 words |
| jfk×12 132 s (clean) | windowed | 397 s | 1.97 GB | 1 / 0 | 176 words |

The windowed path attends the quiet/noisy lead and greedy decode
collapses there — on Metal as well as CUDA (the earlier "Metal recovers"
observation was not robust). It is also ~1.5× slower at these lengths
(many 104² matmuls vs one big one), and the O(N·W) memory advantage only
matters past full attention's ~10-min OOM point. Default stays full
attention + 30 s chunks; windowed remains the escape hatch for >10-min
single-pass audio, with fix_loops ON. (jfk×12 note: both modes EOS early
on genuinely 12×-repeated audio — model behaviour, not a loop bug.)

### glm-asr multi-window single-pass

Blueprint path: 30 s sample windows, encode-then-trim, concatenated into
ONE prompt/decode — cap 21 windows = 655 s (processor `max_audio_len`).
The 145 s clip = 5 windows, 1812 audio tokens, 1827-token prompt,
q4_k output 113 tokens vs bf16 blueprint's 115 (near-verbatim). Note the
blueprint (and therefore our single-pass) SKIPS quiet leading audio; the
default 30 s-chunked path covers more content on such clips.

## Long-audio coverage benchmark — 2026-05-21 (issue #89, historical)

Platform: x86_64 VPS, 4 threads, CPU-only, no GPU. Commit `5e16414`
(30 s auto-chunk fallback + PR #116 VAD gate fix).

Test audio: first 60 s of the issue #89 reporter's exact YouTube clip
(`o_9dWkRPYC0`, Japanese podcast, 16 kHz mono PCM). Human estimate:
100-150 words in the first 60 s of continuous speech.

### Issue #89 fix verification — parakeet-tdt-0.6b-ja

**Final state — streamed encoding is always on (default
`CRISPASR_PARAKEET_STREAM_THRESHOLD=0`, see commit "always route
parakeet through streamed encode").**

Global z-norm + overlapping 8 s encoder chunks + single TDT decode pass.

The earlier "single-pass ≤60 s" default produced 99.5 % coverage on the
cached MP3-derived copy of the reporter's clip but only ~33 % on a
fresh `yt-dlp` extract of the same YouTube video (lenhone, issue #89
comment 4529025103). Both extracts are perceptually identical
(duration 60.000 s, 0.998 waveform correlation, ~0.3 % RMS diff from
codec quantization) but the FastConformer encoder's full-clip
bidirectional attention amplified that quantization noise enough
(encoder output std differed by 14 %: 0.2069 vs 0.2415) to drive the
TDT decoder into emit-blank-forever past frame 250 (≈20 s). The
streamed path keeps attention local to 8 s windows, so codec
perturbations don't amplify and the decoder runs to the end.

| path | audio | chars | first_ts | last_ts | coverage% | gaps |
|---|---|---:|---:|---:|---:|---:|
| **streamed (default)** | reporter's MP3-derived `yt_60s.wav` | **309** | **0.00** | **55.84** | **~99** | **0** |
| **streamed (default)** | fresh `yt-dlp` Opus→PCM `o_9dWkRPYC0_60s.wav` | **314** | **0.00** | **55.84** | **~99** | **0** |
| old: single-pass ≤60 s | reporter's MP3-derived `yt_60s.wav` | 309 | 0.16 | 59.84 | 99.5 | 0 |
| old: single-pass ≤60 s | fresh `yt-dlp` Opus→PCM `o_9dWkRPYC0_60s.wav` | 91 | 0.16 | 20.08 | **~33** | 0 |
| `--vad` (silero) | reporter's MP3-derived | 281 | 0.36 | 59.87 | 93.1 | 1 |
| `--vad --vad-model firered` | reporter's MP3-derived | 238 | 0.28 | 58.01 | 85.1 | 1 |
| old: 30 s independent chunks | reporter's MP3-derived | 195 | 0.16 | 58.02 | 59.7 | 2 |

**Key findings:**
- Single-pass encoding over the full clip is **not robust**: a 0.3 %
  RMS difference between two codec-quantized copies of the same speech
  flips the encoder into emit-blank mode at the 20 s mark on one and
  not the other. The streamed path is robust to that perturbation.
- The streamed pipeline gives ~99 % coverage on both audio variants.
- `--vad` (silero) gives 93 % coverage with speech-boundary
  segmentation.  Useful when you want per-utterance SRT entries rather
  than continuous transcription.
- The old 30 s independent-chunk approach (pre-fix) lost content due
  to TDT decoder cold-start on each chunk (each chunk reset the LSTM
  state).
- **Recommendation for Japanese:** just run `crispasr -m
  parakeet-tdt-0.6b-ja.gguf -f audio.wav -osrt` — the default routes
  through streamed and handles any duration on any audio source.

### Robustness validation — 2026-05-23

Full sweep on the reporter's 60 s clip (commit `0c24178e`, CPU-only).
Streamed pipeline output is **byte-identical to single-pass** across
every chunk/overlap combination tested — the global z-norm makes chunk
boundaries transparent to the TDT decoder.

> **Caveat (added 2026-05-24).** The "identical to single-pass" column
> below is only true on the cached MP3-derived audio (`/mnt/storage/
> samples/o_9dWkRPYC0.mp3` → `yt_60s.wav`). On a fresh `yt-dlp` extract
> of the same YouTube video, single-pass collapses to ~20 s of output
> while streamed still covers the whole clip — see the "single-pass
> not robust" finding above. The streamed-vs-streamed numbers across
> chunk and overlap sizes (300+ chars, 99 %+ coverage) hold across
> both audio derivations.

**Chunk-size sweep** (streamed, overlap=2 s):

| chunk | chars | coverage% | gaps | identical to single-pass? |
|---|---:|---:|---:|---|
| 4 s | 294 | 99.5 | 0 | yes |
| 6 s | 294 | 99.5 | 0 | yes |
| 8 s (default) | 294 | 99.5 | 0 | yes |
| 12 s | 294 | 99.5 | 0 | yes |
| 16 s | 294 | 99.5 | 0 | yes |
| 20 s | 294 | 99.5 | 0 | yes |
| 30 s | 294 | 99.5 | 0 | yes |

**Overlap sweep** (streamed, chunk=8 s):

| overlap | chars | coverage% | gaps | identical? |
|---|---:|---:|---:|---|
| 0 s | 294 | 99.5 | 0 | yes |
| 1 s | 294 | 99.5 | 0 | yes |
| 2 s (default) | 294 | 99.5 | 0 | yes |
| 3 s | 294 | 99.5 | 0 | yes |
| 4 s | 294 | 99.5 | 0 | yes |

**300 s Japanese audio** (streamed, default 8 s chunks):
- 655 chars, **98.6 % coverage**, 0 gaps, first=0.00 last=295.84
- Before fix (30 s independent chunks): 636 chars starting at 58 s

**VAD comparison** (60 s):

| mode | chars | coverage% | gaps |
|---|---:|---:|---:|
| auto (streamed) | 294 | 99.5 | 0 |
| `--vad` silero | 281 | 93.1 | 1 |
| `--vad --vad-model firered` | 238 | 85.1 | 1 |

VAD produces fewer characters because it segments on speech boundaries
and transcribes each segment independently. The auto/streamed path
transcribes the full audio continuously and achieves higher coverage.
Use VAD when you need per-utterance SRT entries; use auto when you want
maximum transcription completeness.

### Multi-backend Japanese comparison (60 s)

All backends on the same 60 s Japanese clip. "chars" counts non-space
characters (Japanese has no word spaces; "words" column counts
space-delimited tokens, which undercounts for CJK).

| backend | settings | chars | coverage% | gaps | wall_s | rtf |
|---|---|---:|---:|---:|---:|---:|
| **parakeet-tdt-0.6b-ja** | **auto (streamed)** | **294** | **99.5** | **0** | **~60** | **~1.0×** |
| cohere-transcribe | `--vad` | 296 | 96.8 | 0 | 169.0 | 0.4× |
| parakeet-tdt-0.6b-ja | `--vad` silero | 281 | 93.1 | 1 | 50.7 | 1.2× |
| cohere-transcribe | auto | 242 | 87.4 | 1 | 199.2 | 0.3× |
| parakeet-tdt-0.6b-ja | `--vad` firered | 238 | 85.1 | 1 | 58.3 | 1.0× |

**Quality ranking for 60 s Japanese:**
1. **Parakeet auto (streamed)** — 99.5 % coverage, zero gaps, ~1× RT.
   The NeMo-style pipeline makes this the clear winner.
2. **Cohere + VAD** — 96.8 %, zero gaps, but 3× slower.
3. **Parakeet + VAD silero** — 93.1 %, 1 gap. Useful for per-utterance
   subtitle segmentation.

### Cross-backend CAP_INTERNAL_CHUNKING — 2026-05-23

The 30 s auto-chunk fallback affected all `CAP_UNBOUNDED_INPUT` backends
that use PerFeatureZ mel normalization, not just parakeet.  Adding
`CAP_INTERNAL_CHUNKING` to canary and fastconformer-ctc (commit
`1dd247a7`) lets them skip the auto-chunk and process full audio in a
single encoder pass.

| backend | audio | coverage (old 30 s chunks) | coverage (new, single-pass) |
|---|---|---:|---:|
| parakeet-tdt 0.6b JA | 60 s JA | 59.7 % | **99.5 %** |
| parakeet-ctc 1.1b | 60 s EN | 74.6 % | **98.5 %** |
| canary-1b-v2 Q4_K | 60 s EN | broken (empty) | **96.8 %** |

**Not affected** (different normalization or architecture):
wav2vec2 / hubert / data2vec (GlobalClipMax, no PerFeatureZ drift);
firered-asr (PerFeatureZ but inline AED — needs separate investigation);
granite-nar (different architecture).

### Benchmark framework

Results collected with `tests/benchmark_asr.py`:

```bash
# Quick single-backend triage:
python tests/benchmark_asr.py --audio myfile.wav --backend parakeet

# Full matrix across backends and settings:
python tests/benchmark_asr.py --corpus /mnt/storage/test-audio/corpus.json --all-settings

# Build the test audio corpus (en/de/ja/zh × 4 durations from FLEURS):
python tests/benchmark_corpus.py
```

Results are stored in `/mnt/storage/benchmark-results/runs.jsonl` (JSONL,
one line per run). The framework computes: word count, char count,
first/last timestamp, time coverage %, gap count/size, wall time, and
realtime factor. See `tests/benchmark_metrics.py` for the metric
definitions and `tests/test_benchmark_metrics.py` for 14 pytest unit
tests that validate the computation (including the issue #89 failure
signature: <5 % coverage on 300 s audio).

### Multi-backend long-form Japanese — 120 s sweep (2026-05-24, issue #89 follow-up)

Test audio: first **120 s** of the issue #89 reporter's exact YouTube
clip (`o_9dWkRPYC0`, fresh `yt-dlp` Opus→WAV extract, the file lenhone
actually reports against — md5 `d1f2ef…`, *not* the cached MP3-derived
copy). Apple M1 Metal, default flags unless noted.

Speech runs out around 01:37; the remaining ~22 s is short pause +
follow-on talking. All "covers full speech" rows below land around
01:37 → 02:00.

| backend | mode | segments | first → last ts | coverage | notes |
|---|---|---:|---|---|---|
| **parakeet-tdt-0.6b-ja** (default, streamed TDT) | full | 12 | 0:00 → 1:37.84 | full speech | post-fix `33f9a162` |
| **parakeet-tdt-0.6b-ja** + `--vad` | full | 14 | 0:00 → 1:58.39 | full speech | cleaner per-utterance |
| **parakeet-tdt-0.6b-ja** + `--parakeet-decoder ctc` (hybrid CTC head) | full | 12 | 0:00 → 1:37.84 | full speech | byte-identical to streamed-TDT — confirms encoder is fine |
| **parakeet-tdt-0.6b-ja** + `STREAM_THRESHOLD=999` (forced single-pass) | full | many | 0:00 → ~14 s then kana-by-kana fragmentation | **broken** | the issue #89 bug; reproduces the lenhone complaint |
| **sensevoice-small** (CTC, multilingual) | `--vad` | 13 | 0:00 → 2:00 | full speech | accurate, minor JA glitches (`スピーク**ジャ**プネス**ナチャパ**`); 19.8× RT |
| **voxtral-mini-3b** (LLM AR, multilingual) | default chunking | partial | 0:00 → 0:27 then 1:47 → 2:00 | **drops 0:27 → 1:47 (~80 s)** | LLM decoder loses a middle chunk |
| **cohere-transcribe** (Conformer, multilingual) | default chunking | 4 | 0:00 → 1:53 | **sparse with multi-tens-of-seconds gaps** | only ~0:00, 0:50, 1:18, 1:48 anchors |
| **canary-1b-v2** (NeMo multilingual seq2seq) | default | broken | n/a | hallucinates `"I am not aware of anything"` in English | needs proper language-prompt wiring; out of scope here |

**Best on this 120 s clip:** parakeet-tdt-0.6b-ja (post-fix, streamed
TDT) and sensevoice-small are tied — both produce full speech coverage
with sentence-level segmentation. Parakeet via the CTC head produces
*byte-identical* output to streamed-TDT, which confirms the encoder
isn't the problem: only TDT-decoded-over-the-full-utterance is.

**The new finding (voxtral, cohere):** these aren't parakeet-specific
failures. voxtral and cohere both drop the middle of a 120 s clip on
this audio, with different symptoms:

- **voxtral-mini-3b**: LLM decoder loses one middle chunk entirely
  (segments span 0:00 → 0:27, then skip to 1:47 → 2:00). The chunker
  hands the AR decoder its middle window, the decoder either runs to
  max_new_tokens before catching up to the audio or skips ahead via
  prompt conditioning that misfires on this clip. Not investigated
  deeply yet; this is an open follow-up — see PLAN #114.
- **cohere-transcribe**: at the default chunk size, the energy chunker
  hands the encoder a small number of long slices and the Conformer
  encoder hits a similar long-bidirectional-attention-amplifies-noise
  regime as parakeet single-pass on lenhone's audio. Only ~4 segments
  emitted across 120 s, with gaps of tens of seconds between them.

**Upstream behaviour for the same long-form failure:**

- **NeMo parakeet / canary**: stock `model.transcribe()` does single-
  pass over the full utterance (verified locally — NeMo's own
  `transcribe()` produces 47 chars and stops at ~20 s on the same
  lenhone WAV, *exactly* matching our pre-fix single-pass). For long
  audio NeMo ships `nemo.collections.asr.parts.utils.streaming_utils.
  BatchedFrameASRTDT` / `FrameBatchChunkedCTC` and
  `nemo.collections.asr.parts.utils.transcribe_utils.
  get_buffered_pred_feat_rnnt`, plus the
  `examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_
  infer_rnnt.py` reference script. Our `parakeet_transcribe_streamed`
  is the same shape: global-z-norm + chunked encode + single TDT
  decode, with overlap-skip on chunk boundaries. The difference is
  ours is now the default; upstream `transcribe()` is not.
- **Mistral voxtral**: the reference HuggingFace integration chunks at
  ~30 s with overlap; the long-form failure we see at 120 s is partly
  an artefact of how our energy-chunker hands the slices to the LLM
  decoder (not upstream-identical chunking).
- **Cohere Transcribe**: the released model is intended for
  ≤ a few minutes per call; the hosted product does server-side VAD +
  chunking, the released weights do not.

**Implication.** The "default fine on a single transcribe() call over
the whole file" affordance is fragile across this whole class of
models. Going forward we should probably treat *every* `CAP_UNBOUNDED
_INPUT` backend the way we now treat parakeet: ship a chunked /
streamed default that the user doesn't have to opt into. See PLAN
#114 for the open architectural question and the per-backend ladder.

### Cross-length × cross-backend matrix — 60 / 120 / 300 / 600 s (2026-05-25)

Same audio (lenhone's fresh `yt-dlp` extract), extended to longer
durations. Linux x86 CPU on the issue #89 VPS (`168.119.190.252`),
sequential to avoid memory contention (we paused / split the queue
when the kernel went into thrash territory). `tools/longform_vps.sh`
is the harness; `tools/analyze_longform.py` parses the per-cell
JSON output.

> **Matrix v1 vs v2 (2026-05-25 afternoon recheck).** The numbers
> below were collected in two passes. The first pass (matrix v1)
> ran on the VPS binary `bd8b98cf` (May 24), which **predates** the
> per-backend opt-out fixes for cohere (`dc2295b2`), gemma4-e2b /
> glm-asr (`46f6848d`), kyutai-stt (`eaee2319`), and voxtral
> (`6fef8790`) that landed during the matrix run. Those fixes
> remove an external overlap-save context wrap that the LLM-decoder
> backends couldn't trim back from correctly. Their pre-fix coverage
> of ~9-65 % at 120 s+ was driven by the wrap, not the model
> architecture. **Matrix v2 (post-opt-out, rebuilt VPS binary
> `13059e0c`)** shows the true post-fix behaviour. Both passes are
> kept so the reader can see the cost of the missing opt-out.

**Coverage % (covered span / clip duration, computed from segment
timestamps).** Higher = better. Bold = best at that length.

**Matrix v2 (post-opt-out, what main looks like today):**

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| **parakeet streamed-TDT** (default)       | **93.1** | 81.5 | **96.6** | **99.3** |
| parakeet CTC head (byte-identical)        | **93.1** | 81.5 | **96.6** | **99.3** |
| **voxtral-mini-3b** (default chunking)    | **100.0** | **100.0** | **100.0** | wall-time timeout (`rc=124` at 900 s in 15 min limit; LLM-AR CPU-bound, not a coverage failure) |
| **voxtral-mini-3b** streamed (option A — this PR's pipeline) — single LLM context | **100 %**, 11 segs / 470 chars | **100 %**, 527 chars | **100 %**, 1276 chars / 863 tokens (post-`a5165c84` max_new scaling fix; was 781 chars / 512-tok cap pre-fix) | hung on contended M1 (80 MB free out of 16 GB → Metal allocator stall — see HISTORY 2026-05-25 (late) "Distinguishing slow vs hung run") |
| **cohere-transcribe** (default chunking)  | **96.3** | **97.9** | **98.1** | **97.9** (22 segs, full 0:00 → 10:00, 577 s wall) |
| parakeet single-pass (`STREAM_THRESHOLD=999`, opt-in regression bait) | 33.2 | 81.7 | **1.5** | 99.9 |
| parakeet + `--vad` (silero)               | 86.7 | 82.0 | 76.3 | 84.0 |
| canary-1b-v2                              | (still hallucinates English at every length — separate prompt-wiring bug, PLAN #114 P3) | | | |

**Largest gap (seconds) between consecutive segments, matrix v2:**

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| parakeet streamed-TDT                | 0.0 | 0.0 | 0.0 | 0.0 |
| voxtral default (post-opt-out)       | 0.0 | 0.0 | 0.0 | timeout |
| cohere default (post-opt-out)        | 1.2 | 1.2 | 1.2 | 2.2 |

The pre-fix gap pathologies — voxtral 21.9 / 78.2 / 240.9 / 545.5 s and cohere up to 50 s — are entirely gone with the opt-out fixes. What remains is the cohere baseline ~1.2 s gap between chunks (natural energy-chunker boundaries, well under a sentence pause).

**Wall time (s), matrix v2:**

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| voxtral default (post-opt-out)       | 237 | 393 | 834 | timeout (>900 s) |
| cohere default (post-opt-out)        |  70 | 125 | 290 | 577 |

cohere is consistently ~2× realtime at 300-600 s on VPS x86 CPU. voxtral-mini-3B is ~1.4-2× slower than cohere at the same length (LLM AR decode at 3 B params) and hits the wall around the 10 min mark. Apple Silicon Metal would close most of that gap — the LLM-AR rows on Mac are typically 5-10× faster than x86 CPU on this size class.

**Matrix v1 (pre-opt-out, kept as historical reference for what we
fixed):**

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| **parakeet streamed-TDT** (default)        | **93.1** | 81.5 | **96.6** | **99.3** |
| **parakeet CTC head**                       | **93.1** | 81.5 | **96.6** | **99.3** |
| parakeet single-pass (`STREAM_THRESHOLD=999`) | 33.2 | 81.7 | **1.5** | 99.9 |
| parakeet + `--vad` (silero)                 | 86.7 | **82.0** | 76.3 | 84.0 |
| voxtral-mini-3b (default chunking)          | 63.5 | 34.8 | 19.7 | 9.1 |
| cohere-transcribe (default chunking)        | **95.0** | **91.5** | 58.8 | 61.8 |
| cohere-transcribe + `--vad`                 | **96.8** | 90.8 | **92.5** | **91.4** |
| canary-1b-v2 (default)                      | 99.7* | 99.9* | 99.3* | OOM (rc=137) |

*Canary's coverage% is misleading — the transcribed text is `"I am not aware of anything, I am not aware of…"` (English) at every duration. It's a separate language-prompt-wiring bug, not a long-audio bug. See PLAN #114 P3.

**Largest gap (seconds) between consecutive emitted segments** —
catches the "drops a middle chunk" failure that the coverage% can
under-report when the missing region is bracketed by emitted text on
both sides.

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| parakeet streamed-TDT                       |  0.0 |  0.0 |  0.0 |  0.0 |
| parakeet CTC head                           |  0.0 |  0.0 |  0.0 |  0.0 |
| parakeet single-pass                        |  0.0 |  0.0 |  0.0 |  0.0 |
| parakeet + `--vad`                          |  5.8 | 12.0 | 30.1 | 30.1 |
| **voxtral-mini-3b**                         | **21.9** | **78.2** | **240.9** | **545.5** |
| cohere-transcribe                           |  0.0 |  3.4 | **50.0** | **50.0** |
| cohere-transcribe + `--vad`                 |  1.4 |  4.2 |  4.2 | 19.0 |
| canary-1b-v2                                |  0.0 |  0.0 |  0.0 | n/a |

**Wall time (s) / realtime factor.** Apple Silicon would be 5-10×
faster on the parakeet rows; numbers below are the Linux x86 VPS.

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| parakeet streamed-TDT                       |  55 |  99 |  236 |  463 |
| parakeet CTC head                           |  54 | 102 |  236 |  462 |
| parakeet single-pass                        |  45 |  86 |  235 |  627 |
| parakeet + `--vad`                          |  55 |  97 |  225 |  457 |
| voxtral-mini-3b                             | 166 | 165 |  189 |  193 |
| cohere-transcribe                           |  79 | 144 |  349 |  673 |
| cohere-transcribe + `--vad`                 |  65 | 117 |  279 |  557 |
| cohere-asr-ja (Q4_K, JA audio)             |  31 | 140 |    — |    — |
| cohere-asr-ja (Q4_K, EN audio)             |  39 | 104 |    — |    — |
| canary-1b-v2                                |  68 | 122 |  381 | OOM  |

(voxtral wall time is roughly constant because it silently skips
most of the input — see the gap column.)

### Per-backend take-aways from the matrix

**parakeet (the post-fix default).** streamed-TDT and CTC-head are
byte-identical at every length (CTC head is a frame-synchronous
fallback that bypasses the TDT blank-runaway entirely; streamed-TDT
keeps the TDT decoder but bounds the encoder's bidirectional
attention to 8 s windows so it can't accumulate the codec-noise
amplification). The 120 s coverage dip to 81.5 % is the *audio*, not
the model: the clip's speech runs out at ~01:37 and the next ~22 s
is silence + a sentence-start, so coverage measured against the full
120 s under-counts. Both paths produce the same actual content.

**parakeet single-pass.** Catastrophically non-monotonic: 33 % at
60 s, 82 % at 120 s, **1.5 %** at 300 s, 99.9 % at 600 s. This is the
"per-feature z-norm depends on the full audio's mel statistics"
problem manifesting as random walks across the stable/unstable
boundary. The single-pass path is genuinely unsafe; the "works at
600 s" cell is luck, not a property.

**parakeet + `--vad`.** Coverage drops to 76-87 % across lengths
because VAD trims silence (by design). Larger gaps at longer
durations because the underlying clip has more silence stretches.
Good for "I want per-utterance SRT entries" use cases, less so for
"I want continuous transcription with maximum coverage."

**voxtral-mini-3b.** **In matrix v1: worst long-form behaviour we
measured.** Coverage halved with each length doubling: 64 → 35 → 20 →
9 %. In matrix v2 (post-opt-out, commit `6fef8790` removing the
external overlap-save wrap), coverage jumps to **100 % at 60 / 120 /
300 s**: the LLM AR decoder *was* processing all chunks fine; the
matrix-v1 word-timestamp trim was discarding most of the emitted text
because voxtral's emitted word timestamps don't honour the original
slice frame, so the trim treated almost everything as "outside the
slice range." Two additional fixes shipped together with this matrix:

  * `6fef8790` — voxtral opt-out from the external overlap-save wrap
    (the immediate >90 % fix; default-chunked voxtral is now sound).
  * **PR #114 voxtral_transcribe_streamed** (matching the upstream
    Mistral `apply_transcription_request` pattern): per-30 s encode,
    concatenate audio embeds, **single LLM AR decode** over the whole
    sequence. Result on the 60 s clip is **denser segmentation** (11
    segments / ~470 chars vs 3 segments / 280 chars on default
    chunking) because the LLM doesn't cold-start at every 30 s
    boundary; it sees one continuous audio stream. Both paths produce
    correct content; streamed is the more upstream-faithful default.

**cohere-transcribe.** **In matrix v1: degraded from 95 % at 60 s to
**59 %** at 300 s and **62 %** at 600 s, with 50 s gaps.** In matrix
v2 (post-opt-out, commit `dc2295b2` removing the external overlap-save
wrap for cohere), default chunking jumps to **96-98 %** at 60 / 120 /
300 s with gaps ≤ 1.2 s. The pre-fix gap-growth was driven by the
overlap-save wrap, not the model itself. `--vad` is no longer a
mandatory rescue — it's available for users who want per-utterance
SRT segmentation, but coverage parity is now native.

**canary-1b-v2.** Separate bug. Coverage looks fine because the
decoder emits text for the full duration, but the text is English
`"I am not aware of anything"` in a loop regardless of input
language. Language-prompt wiring problem, not a long-audio problem.
600 s OOM-killed (rc=137) on the 7.6 GB VPS — likely the AED
decoder's hidden-state stack growing past the available memory.

### What's the right default per backend, post-matrix v2

| backend | recommended default | why |
|---|---|---|
| parakeet (any variant)         | streamed-TDT (default since `33f9a162`) | best coverage at all lengths, byte-identical to CTC-head when available |
| voxtral-mini-3b                | streamed (this PR — Mistral `apply_transcription_request` shape) | 100 % coverage at 60-300 s, single LLM context, denser segmentation; default-chunked + opt-out (`6fef8790`) also lands at 100 % |
| cohere-transcribe              | default chunking + opt-out (`dc2295b2`)         | 96-98 % at 60-300 s; `--vad` available but no longer required for coverage |
| canary-1b-v2                   | fix lang-prompt bug first; then streamed-encode port | currently broken at all durations on JA; long-audio fix on hold |
| qwen3-asr / granite-speech / mimo-asr | post-opt-out default chunking (audit pending) | LLM-AR class — opt-out gate is `glm-asr` / `gemma4-e2b` / `kyutai-stt` (`46f6848d`, `eaee2319`); voxtral-style streamed is a follow-up improvement, not a coverage fix |
| fastconformer-ctc / wav2vec2 / firered-asr | current single-pass (CTC is robust) | no observed failure; defer streamed port until reported |
| sensevoice-small               | `--vad`                                          | already the recommendation; matrix v1 confirms 99 %+ at 120 s |
| whisper                        | unchanged                                        | internal 30 s seek handles long audio by design |

### Reproducer

```bash
# Driver — runs all 32 cells sequentially with memory backpressure
bash tools/longform_vps.sh   # outputs to /mnt/akademie_storage/longform_results/

# Parser — JSON outputs → coverage table
python tools/analyze_longform.py /path/to/longform_results/
```

Both scripts in this commit. Audio: `/mnt/akademie_storage/yt_{60,120,300,600}s.wav` on the VPS (PCM s16le, 16 kHz mono, fresh `yt-dlp` extract of `youtube.com/watch?v=o_9dWkRPYC0`).

---

## Beam search — quality vs speed (2026-05-23, PLAN #90)

**Knob:** `--beam-size N` (CLI) / `CRISPASR_BEAM_SIZE=N` (env) /
`crispasr_session_set_beam_size(session, N)` (C API).
Default N=1 (greedy). N > 1 activates beam search on supported backends.
LLM-decoder backends (qwen3-asr, granite-speech, voxtral, gemma4-e2b) use
`core_beam_decode::run_with_probs` (replay-from-prefix). Encoder-decoder
backends (canary, cohere) use `core_beam_decode::run_with_probs_branched`
(KV snapshots). Transducer backends (parakeet) use a dedicated TDT/RNNT
label-looping beam search. CTC-only and NAR backends ignore the flag.

Benchmark script: `tools/benchmark_vitw_beam.py` — runs against
[`zhifeixie/Voices-in-the-Wild-Bench`](https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-Bench)
(5 000 samples, 8 acoustic conditions, streamed — no full download needed).

### Speed cost on JFK (11 s, M1 Metal, post-warmup)

| backend | beam=1 | beam=2 | beam=4 |
|---|---|---|---|
| qwen3-asr 0.6B Q4_K | 3.67 s (1×) | 8.20 s (2.2×) | 14.75 s (4.0×) |
| granite-speech 4.1 2B Q4_K | 18.39 s (1×) | 27.59 s (1.5×) | 33.17 s (1.8×) |
| voxtral mini 3B Q4_K | ~70 s (1×) | ~56 s (0.8×)† | ~77 s (1.1×) |

†voxtral beam=2 < beam=1 is measurement noise — voxtral spends most
of its time in the audio encoder; decoder token count for JFK is
small enough that OS jitter dominates.

### WER by condition (qwen3-asr, Voices-in-the-Wild-Bench, 8 EN samples each)

| condition | beam=1 | beam=2 | beam=4 | beam=2 cost | beam=4 cost |
|---|---|---|---|---|---|
| real_noise | 0.125 | 0.144 | 0.136 | 1.7× | 3.5× |
| syn_noise | 0.167 | 0.167 | 0.167 | 2.6× | 2.7× |
| real_dropout | 0.045 | 0.045 | 0.041 | 1.9× | 4.6× |
| real_obstructed | 0.015 | 0.015 | 0.015 | 1.9× | 3.3× |
| real_mixed | 0.035 | 0.039 | 0.039 | 2.1× | 4.8× |
| syn_mixed | 0.089 | 0.089 | 0.080 | 1.3× | 2.2× |

### Key findings

- **One clear win:** `syn_mixed` — "I called customer service **twice**"
  decoded as "wise" (greedy, WER 0.154), correctly as "twice" at beam=4
  (WER 0.077). Phonetically similar word confusion — exactly the
  scenario beam search is designed to fix.
- **beam search occasionally hurts.** `real_noise` ticks from 0.125 to
  0.144 at beam=2. The beam finds a confident wrong hypothesis that
  greedy would have gotten right — known failure mode on well-trained
  models where the greedy peak is already correct.
- **Heavy hallucination is not rescued.** `real_noise` sample 6
  (WER≈0.42, badly degraded audio) just produces a different confabulation
  at beam=4; the model is guessing regardless of beam width.
- **This dataset skews easy.** Most samples are TTS speech with layered
  acoustic corruption; qwen3-asr is near-ceiling on greedy. Real
  spontaneous noisy speech with disfluencies and rare words would expose
  more beam-search-recoverable errors.

### When to use

| scenario | recommendation |
|---|---|
| Clean / studio speech | greedy (beam=1) — no quality gain, 2-5× cost |
| Noisy real speech, latency-insensitive | beam=2 — marginal gain possible, 2× cost |
| Rare words / phonetic confusion, offline | beam=4 — worth trying |
| Streaming / latency-critical | greedy only — beam adds a full extra decode pass per token step |

Reproduce:

```bash
python tools/benchmark_vitw_beam.py \
    --backends qwen3 \
    --splits real_noise,real_dropout,real_obstructed,real_mixed,syn_mixed \
    --n 8 --beams 1,2,4 \
    --json tools/vitw_beam_results.json
```

### MAES beam search for transducers (2026-06-03, §134)

MAES (Modified Adaptive Expansion Search) is a transducer-specific beam
search that's more efficient than the label-looping beam above. It processes
one encoder frame at a time with up to N adaptive non-blank expansions per
frame, using gamma-threshold pruning to kill low-probability branches.

**Knob:** `CRISPASR_PARAKEET_MAES=1` + `--beam-size N` (CLI), or
`--parakeet-decoder maes` + `--beam-size N`, or
`parakeet_set_maes(ctx, true, num_steps, gamma, beta)` (C API).
Config: `CRISPASR_MAES_NUM_STEPS` (default 2), `CRISPASR_MAES_GAMMA` (2.3),
`CRISPASR_MAES_BETA` (2).

Supports both TDT (Token-and-Duration Transducer) and pure RNNT models.

#### MAES vs greedy on FLEURS English (CPU, Hetzner CCX13, 4 threads)

| Model | Audio | Greedy | MAES beam=4 | Speed cost |
|---|---|---|---|---|
| tdt-0.6b-v2 (1K vocab) | 10s | "...by 25%." | "...by 25 years." | — |
| tdt-0.6b-v3 (8K vocab) | 10s | "...by 25-30 years." | "...by 25 to 30 years." | — |
| tdt-0.6b-v2 | 60s | 5 sentences | 6 sentences (recovered full missing sentence) | +35% |
| tdt-1.1b (8K vocab) | 10s | "...by twenty five to thirty years" | identical | — |
| tdt_ctc-110m (1K vocab) | 10s | garbled | same garble | — |
| rnnt-0.6b (8K vocab) | 10s | "...by twenty five to thirty years" | identical | — |
| rnnt-1.1b (8K vocab) | 60s | truncated at "lettering" | identical | +25% |

#### When to use MAES vs standard beam

| scenario | recommendation |
|---|---|
| Parakeet TDT with small vocab (v2, 1K BPE) | MAES beam=4 — measurable quality gain |
| Parakeet TDT/RNNT with large vocab (8K BPE) | greedy — already strong baseline, MAES matches but costs 25-35% |
| Parakeet with hotwords (CTC-WS) | label-looping beam — hotword trie not yet wired into MAES |
| Tiny model (110M) | neither — model capacity is the bottleneck |

### CTC prefix beam search (2026-06-03, §134)

Shared `core_ctc::prefix_beam_search()` with optional gamma-threshold
pruning. Available for any CTC backend via `--beam-size N`.
Currently wired into: parakeet-CTC, sensevoice, wav2vec2 (16 languages).

CTC beam search has not yet been benchmarked for WER improvement — the
primary benefit is expected to be on character-level CTC models (wav2vec2)
where the small vocab makes greedy more error-prone than BPE models.

### Transducer + encoder-decoder beam search (2026-06-02, issue #136 + §139)

Parakeet TDT/RNNT label-looping beam (`b3cdcebd`), canary + cohere
AED branched-KV beam (§90 runtime, adapter wiring `§139`), gemma4-e2b
replay-from-prefix beam (§139). All on VPS CPU-only (no GPU).

**JFK 11 s — wall time (user time in parentheses)**

| backend | model | beam=1 | beam=2 | beam=4 |
|---|---|---|---|---|
| parakeet | parakeet-tdt-0.6b-v3 F16 | 27 s (30 s) | 26 s (31 s) | 15 s (32 s) |
| canary | canary-1b-v2 Q4_K | 27 s (37 s) | 32 s (50 s) | 42 s (68 s) |
| cohere | cohere-transcribe F16 | 125 s (91 s) | 104 s (107 s) | 120 s (118 s) |

Notes:
- Parakeet beam adds ~7 % user time at beam=4 (LSTM predictor + joint
  head are tiny; encoder dominates). Wall time variance is system load.
- Canary beam=4 costs ~84 % more user time (8-layer decoder × KV
  snapshot/restore per beam step).
- Cohere F16 model is ~3 GB; beam=4 KV snapshots increase peak memory
  substantially. beam=4 was OOM-killed on the FLEURS-10s test.
- All backends produce identical text on JFK at beam=1/2/4.

**FLEURS 10 s — canary beam=4 vs greedy**

| beam | output |
|---|---|
| 1 | "…Styles in the West could lag behind by twenty five percent. 25 to 30 years." |
| 4 | "…styles in the west could lag behind by twenty five percent. 25 to 30 years." |

Minor capitalization difference (proper-noun casing on "Styles"/"West").

**FLEURS 60 s — parakeet beam=4 vs greedy**

| beam | output diff |
|---|---|
| 1 | "…, and which was made famous…" |
| 4 | "… and which was made famous…" (comma dropped) |

Both valid; stylistic punctuation variation.

**Overhead summary (user time)**

| backend | beam=2 | beam=4 | beam=8 |
|---|---|---|---|
| parakeet (TDT LSTM) | ~3 % | ~7 % | ~20 % |
| moonshine-streaming | ~24 % | ~56 % | — |
| canary | ~35 % | ~84 % | — |
| cohere | ~18 % | ~30 % | OOM (F16) |

Parakeet beam search is nearly free because the decoder is a tiny
LSTM (~10 KB state per beam). Canary and cohere have 8-layer
transformer decoders with full KV snapshot/restore, so the cost
scales with decoder depth × sequence length × beam width.

### Translation beam search (m2m100 + madlad/t5, 2026-06-02)

**m2m100-418m Q8_0 — en→de (CPU-only VPS)**

| sentence | beam=1 user | beam=4 user | output |
|---|---|---|---|
| "Hello world, how are you today?" | 6 s | 21 s (3.4×) | "Hallo Welt, wie bist du heute?" |
| "The president said he would not attend…" | 7 s | 45 s (6.4×) | "Der Präsident sagte, er würde wegen der Wetterbedingungen nicht an der Sitzung teilnehmen." |

Translation beam is expensive: the decoder-only replay cost is
O(beam × T²) where T is the output length, and for translation the
decoder does more work per token than for ASR. Identical output on
these clean inputs; benefit is on ambiguous source text.

## Multi-backend long-form comparison — 2026-05-26 (PLAN #114 P3 closeout)

Live runs on M1 Metal with the post-PLAN-#114-P3 binaries. Inputs from
`/Volumes/backups/code/audio_samples/` (mirrored from VPS — see that
dir's CLAUDE.md). All backends invoked with `-l <lang>`, `-np`, `-nt`,
default settings (`CANARY_STREAM_THRESHOLD_S=0` after `10c2fba5`).

### EN 60 s (FLEURS English, narration)

```
audio_samples/en/fleurs_60s.wav  (60 s, 16 kHz mono)
```

| Backend | Chars | Notes |
|---|---|---|
| parakeet-tdt-0.6b-v3 | ~217 → **520** (post `e1904a1e`) | The 217 was with the c=8 chunk default that ships well for the JA-only model. Empirical sweep on EN+DE FLEURS 60s+300s showed c=8 collapses on the multilingual v3 model: EN 60s drops to 23% of the c=40 max. Fix `e1904a1e` adds a per-model chunk default keyed off `vocab_size` (< 4000 ⇒ JA model ⇒ c=8 preserved; ≥ 4000 ⇒ v3 / multilingual ⇒ c=30). v3+EN60 improved 186 → 520 chars (2.8×), v3+EN300 492 → 1550 chars (3.15×), v3+DE60 502 → 679, v3+DE300 2496 → 3064, ja+JA60 1674 unchanged. Root cause is *encoder* context, not decoder cold-start — see the corrected LEARNINGS section on the Independent-chunk failure mode. |
| canary-1b-v2 | ~735 | Full content but visible artifacts: `"twenty five. to thirty"` (model splits a number), `"Save for You"` (AED re-emits with different capitalization → after case-insensitive LCS the dup is dropped but the leftover `"Save for You"` reads as a sentence start), `"Yeah, yeah, ×14"` (degenerate-loop guard fired at the configured 14-token window). |
| voxtral-mini-3b-2507 | **~826** | **Clean.** Includes extra content like `"in which was made famous to foreigners after a glowing account of its splendorous recorded by Lord Byron"` that canary missed entirely. No boundary artifacts. |
| cohere-transcribe | **~864** | **Clean.** Similar coverage to voxtral, `"world's"` instead of `"world"` (model preference), `"Northern Marianas"` instead of `"Northern Mariana's"`. |

### DE 60 s (FLEURS German, narration)

```
audio_samples/de/fleurs_60s.wav  (60 s, 16 kHz mono)
```

| Backend | Notes |
|---|---|
| canary-1b-v2 | Full content but boundary dups: `"Geld-Technologie-Technologie-Technologie-Technologie"` (early-chunk loop, partly caught by the guard at 4 reps before the window opened), `"T-Rex war war"`, `"Rückseite der der Unabhängigkeitserklärung"`, `"Männer und Frauen. Frauen"`, `"Spitze. der Spitze"`. The LCS-merge + word-snap + case-insensitive LCS pipeline caught some but not all — these are exact-token re-emissions across chunks that an LCS strict-prefix match still leaks. |
| voxtral-mini-3b-2507 | **Clean, single-pass-quality.** No boundary artifacts visible. Catches `"Juden und Nicht-Juden gleichermaßen"` (post-segment continuation) that canary missed. |
| cohere-transcribe | **Clean.** `"Tri-Rex"` is a minor model error (not a boundary artifact), otherwise identical-shape transcript to voxtral. |

### Architectural takeaway

The data confirms the design-notes table in PLAN #114 ("Streaming-pattern
design"): the **voxtral-pattern backends** (voxtral, cohere — 30 s
disjoint chunks → audio embeds concat → one LLM AR decode) produce
cleaner long-form output than the **NeMo-pattern backends** (canary —
8 s overlap chunks → per-chunk decode → LCS-merge + word-snap +
case-insensitive LCS dedup). The voxtral pattern's lack of overlap means
no duplication enters the input, so no dedup is needed; the NeMo
pattern's overlap (necessary for bidirectional encoder context) requires
dedup, and any imperfect dedup pass leaves visible artifacts.

This is not a universal win for the voxtral pattern though — it requires
a long-context AR LLM (voxtral's 3 B, cohere's 1.3 B). Canary's AED was
trained on 8–30 s clips and cannot absorb a full 5 min in a single decode
(`<eos>` lands at the first internal utterance boundary). Parakeet's TDT
could in principle use the voxtral pattern but doesn't currently —
something to revisit if parakeet's truncation behaviour on the EN 60 s
clip turns out to be a streamed-path bug rather than a one-off.

### Six-commit canary thread that produced the "full content" column above

| SHA | What |
|---|---|
| `dfe1af3b` | lang-whitelist (en/de/fr/es only) — refused unsupported langs before they could hallucinate |
| `7177c931` | `canary_transcribe_streamed` first cut (concat-then-decode → truncated at AED `<eos>`) |
| `63fdbe46` | NeMo `FrameBatchMultiTaskAED` analogon — per-chunk AED decode with prompt re-injection |
| `62766dae` | LCS boundary dedup |
| `10c2fba5` | splice-punct cleanup + `CANARY_STREAM_THRESHOLD_S=0` default |
| `361df3e2` | window-based degenerate-loop guard |
| `935ffbee` | word-snap heuristic (extend LCS drop to next word-start) |
| `5e402ee9` | case-insensitive LCS (ASCII lowercase canonical id) |

Before this thread canary truncated to ~460 chars on the 1.3 m
De-Abwasch article and ~360 chars on EN FLEURS 60 s; after, full
coverage (~1196 chars and ~735 chars respectively).

## Parakeet long-form option matrix — 2026-05-26 (PLAN #114 follow-up)

Empirical sweep across all the dispatch knobs the parakeet backend
exposes, on the same three 60 s fixtures used elsewhere in this
section. Default mode includes the `e1904a1e` per-model chunk default
(v3 → c=30 internal, ja → c=8 internal).

| Mode | v3 + EN 60s | v3 + DE 60s | v3 + JA 60s | ja + JA 60s |
|---|---|---|---|---|
| **default** (backend streamed, c=auto) | 520 | 679 | 605 | **1674** |
| `CRISPASR_PARAKEET_STREAM_CHUNK=8` forced | 187 | 503 | 375 | **1674** |
| `CRISPASR_PARAKEET_STREAM_CHUNK=30` forced | 520 | 679 | 605 | 508 |
| `CRISPASR_PARAKEET_STREAM_THRESHOLD=999` (single-pass) | **626** | 621 | 599 | 271 |
| `--vad --vad-model silero` | 368 | **709** | 637 | 1627 |
| `--chunk-seconds 30 --chunk-overlap 0` (no LCS) | 713 | 689 | 608 | 1413 |
| `--chunk-seconds 30 --chunk-overlap 3` (LCS) | **755** | 665 | **660** | **1942** |

### Headline finding: dispatcher-side `--chunk-seconds 30 --chunk-overlap 3` wins on 3 of 4 cases — **shipped as the new default in `98381810`**

The internal-streamed-path default that previously shipped was **not** the
quality-optimal long-form mode. The CLI's dispatcher-side chunking +
overlap-save context wrap + LCS-merge dedup recovers more content than
the backend's single-pass-over-concat-encoder design.

**Shipped as the new default 2026-05-26 (`98381810`)** by dropping
`CAP_INTERNAL_CHUNKING` from the parakeet backend's capabilities
declaration. The dispatcher's `should_auto_chunk_long` fallback then
fires for audio > 30 s — chunking at 30 s, overlap-save 3 s, LCS-merge
dedup — exactly the matrix's winning mode. Short audio (< 30 s) is
unaffected: the dispatcher only auto-chunks past the threshold, so the
11 s JFK case still routes through a single backend call.

After-the-fix matrix (the previous matrix was with `CAP_INTERNAL_CHUNKING`
declared, blocking the auto-chunk path):

| case | old default | new default | Δ |
|---|---|---|---|
| JFK 11s | 109 | 109 | unchanged |
| v3 + EN 60s | 520 | **755** | **+45 %** |
| v3 + DE 60s | 679 | 665 | -2 % |
| v3 + JA 60s | 605 | **660** | +9 % |
| ja + JA 60s | 1674 | **1942** | **+16 %** |
| v3 + EN 300s | 1550 | **3865** | **+150 %** |
| v3 + DE 300s | 3064 | **3288** | +7 % |

The longer the audio, the bigger the win — EN 300 s scales from +45 % at
60 s to +150 % at 300 s. The internal-streamed-path's quality
degradation compounds with audio length; the dispatcher chunks scale
linearly.

Wall time on M1 Metal: 300 s EN now takes ~86 s (was ~30 s) — 3.5×
realtime. Acceptable for the quality gains; users can still pass
`CRISPASR_PARAKEET_STREAM_THRESHOLD=99999` to force the older
single-pass path if the wall-time matters more than coverage.

### Headline finding: dispatcher-side `--chunk-seconds 30 --chunk-overlap 3` wins on 3 of 4 cases (original 4-trial sweep)

Why this works: the dispatcher splits the 60 s input into ~30 s chunks
with ±3 s acoustic overlap, calls the backend once per chunk (each
call sees a 33 s window), and LCS-merges the boundary tokens. Inside
each backend call, parakeet's internal streamed path now runs as a
single 30 s encoder window (no further sub-chunking), which is
exactly the encoder context size the v3 model was trained for. The
backend's own "streamed over 60 s with c=30" instead splits the 60 s
mel into two ~30 s chunks internally — but the per-chunk encoder
passes don't see the bidirectional context across the cut as cleanly
as the dispatcher's per-call boundaries do (the dispatcher feeds each
chunk independently with its own mel-norm; the backend's streamed
path applies global mel-norm first then splits).

### When each mode wins

| Audio profile | Recommended mode | Why |
|---|---|---|
| Continuous EN/DE long-form, supported v3 lang | `--chunk-seconds 30 --chunk-overlap 3` | Highest coverage; modest wallclock overhead (12 s for 60 s audio) |
| JA model on JA long-form | `--chunk-seconds 30 --chunk-overlap 3` OR default | Both recover most content; LCS edges default by 16 % on the tested clip |
| Short audio (< 30 s) | default | Single backend call, no dispatcher overhead |
| Speech-with-long-silences | `--vad --vad-model silero` | VAD trims silences and feeds the backend with bounded slices; can outperform chunking when speech density is uneven |
| Reference parity / debugging | `CRISPASR_PARAKEET_STREAM_THRESHOLD=999` | Forces `parakeet_transcribe_ex`, the bit-exact single-pass path |

### Caveats

- The dispatcher's chunk-overlap wrap is what `kBlocked` opts cohere /
  gemma4-e2b / kyutai-stt etc. *out* of (LCS doesn't compose with
  their internal-chunking pipelines). Parakeet is intentionally NOT
  in `kBlocked` — the dispatcher's wrap and LCS dedup are correct for
  TDT's frame-synchronous output.
- The default did NOT change to `--chunk-seconds 30 --chunk-overlap 3`
  because that would force a dispatcher change visible to every
  caller (server API, Python session, …) without their request.
  Users who want the +45 % EN coverage today pass the flags
  explicitly; the option matrix above is the documentation.
- Single-pass occasionally wins (v3 + EN 60s: 626 chars vs 520
  internal-streamed default) — for clips that comfortably fit a single
  encoder forward pass on the model's hardware, the streamed wrapper
  is overhead. `CRISPASR_PARAKEET_STREAM_THRESHOLD=99999` makes
  single-pass the default.

### Reproduce

```
B=build/bin/crispasr
V3=/Volumes/backups/ai/crispasr/parakeet-tdt-0.6b-v3-q4_k.gguf
JA=/Volumes/backups/ai/crispasr/parakeet-tdt-0.6b-ja-q4_k.gguf
EN60=/Volumes/backups/code/audio_samples/en/fleurs_60s.wav
DE60=/Volumes/backups/code/audio_samples/de/fleurs_60s.wav
JA60=/Volumes/backups/ai/long-clips/yt_60s.wav

# default
$B --backend parakeet -m $V3 -f $EN60 -np -nt
# --chunk-seconds 30 --chunk-overlap 3
$B --backend parakeet -m $V3 -f $EN60 -np -nt --chunk-seconds 30 --chunk-overlap 3
# (etc)
```

### Coverage parity check vs cohere / canary on 300 s — 2026-05-26

User direction: "are these after-numbers complete? compare to what
other models deliver". Right — char-count delta vs the previous parakeet
default proves we *improved*, but says nothing about *complete*. Real
test: how does parakeet's new default compare against the best
long-form-capable backends on the same audio.

Ran parakeet (post-`98381810` default) vs cohere vs canary on the
300 s FLEURS clips. Voxtral skipped — its mem-thrash failure mode on
M1 with the 300 s clip is documented in
[`feedback_torch_omp_deadlock`](../memory/feedback_torch_omp_deadlock.md);
sat at 5 s CPU / 30 min wall and was killed.

**EN FLEURS 300 s:**

| Backend | chars | wall (s) | × RT | vs cohere |
|---|---|---|---|---|
| parakeet (default) | **3865** | 66 | 5.0× | -3 % |
| cohere | 3994 | 94 | 3.2× | (ref) |
| canary | 2971 | 74 | 4.1× | -26 % |

**DE FLEURS 300 s:**

| Backend | chars | wall (s) | × RT | vs cohere |
|---|---|---|---|---|
| parakeet (default) | **3288** | 69 | 4.3× | -0.3 % |
| cohere | 3299 | 87 | 3.4× | (ref) |
| canary | 3532 | 273 | 1.1× | +7 %, 3.1× slower |

**Headline.** Parakeet's post-fix default is now within **3 % of cohere
on EN 300 s and within 0.3 % on DE 300 s** at higher throughput
(66 / 69 s wall vs cohere's 94 / 87 s). The numbers are complete in
the same sense as cohere — coverage parity with the best long-form
backend, at faster wallclock. The previous default (CAP_INTERNAL_CHUNKING
set) was 60 % below cohere on EN 300 s; the fix closes that gap.

Canary on DE wins on coverage (3532 chars) but pays 4× the wall time
(273 s) — a different trade-off. For German-only workflows where wall
time is bounded, canary's per-chunk AED decode produces slightly more
content; for general use, parakeet's faster path with coverage-parity
is the recommended default.

---

## Runtime optimization audit — 2026-06-20

Full code-read survey of every runtime in the project: what optimization
tricks each already implements, and where room exists for more. Covers
65+ backends across ASR, TTS, Audio-LLM, VAD, LID, speaker, translation,
enhancement, alignment, punctuation, and diarization — plus the shared
core infrastructure.

### Legend

- **Has** = optimization is implemented and active
- **Partial** = infrastructure exists but disabled by default or incomplete
- **Gap** = applicable but not yet wired

---

### 1. Cross-cutting optimization matrix

#### 1a. AR decode infrastructure (backends with autoregressive token loops)

| Backend | KV cache | Flash attn | Fused QKV | Graph cache | CPU embd cache | Layer offload | Beam search |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **whisper** | Has | Has | — | Gap | — | — | Has (batch) |
| **canary** | Has (quant) | Has | — | Gap | — | Gap | Has (branched) |
| **cohere** | Has (quant) | Has | — | Gap | — | Gap | Has |
| **kyutai-stt** | Has (quant) | Has | Has | Gap | — | Gap | Has |
| **firered-asr** | Gap (vec) | Gap | — | Gap | — | Gap | Has (lazy) |
| **moonshine** | Has | Has | — | Gap | — | — | Has (branched) |
| **moonshine-stream** | Has | Has | — | Gap | — | — | Has |
| **funasr** | Has (quant) | Has (enc) | Has | Partial | — | Has (split) | Has (replay) |
| **voxtral** | Has (quant) | Partial | Has | Gap | — | Has | Has |
| **voxtral4b** | Has (quant) | Partial (enc gap) | Has | Gap | — | Has | Has |
| **glm-asr** | Has (quant) | Has | — | Gap | — | Has (LLM) | Has (replay) |
| **granite-speech** | Has (quant) | Has (LLM) | — | Gap | — | Has | Has |
| **qwen3-asr** | Has (quant) | — | Has | Gap | — | Has | — |
| **omniasr** | Has (quant) | Has | — | Gap | — | Has | Has (replay) |
| **mimo-asr** | Has (quant) | Has (audio) | Has | Has (T=1) | — | Has | Has |
| **moss-audio** | Has | Gap (enc) | Gap | Gap | — | Gap | Has |
| **gemma4-e2b** | Has (dual) | Gap (enc) | — | Gap | — | Has | Has |
| **m2m100** | Has (cross) | Has | — | Gap | — | Gap | Has (replay) |
| **t5-translate** | Has (cross) | Gap (rpe) | — | Gap | — | Gap | — |
| **orpheus** | Has (quant) | Gap | — | Gap | Gap | Has | — |
| **outetts** | Has (quant) | Gap | — | Gap | Gap | Gap | — |
| **kokoro** | — (NAR) | Gap | — | Gap | — | — | — |
| **bark** | Has | Has | Has | Gap | — | — | — |
| **melotts** | — | Gap | — | Gap | — | — | — |
| **parler** | Has (host; device opt-in §176c) | Gap | — | Has (Lk-bucket, opt-in §176b) | — | — | — |
| **speecht5** | Has (host) | Gap | — | Gap | — | — | — |
| **dia** | Has (host) | Gap | — | Gap | — | — | — |
| **csm** | Has (device) | Has (Mimi) | Has | — | — | — | — |
| **indextts** | Has (device) | Has | — | — | — | — | Has (device) |
| **f5-tts** | — (diff) | Has (DiT) | — | Gap | — | — | — |
| **fastpitch** | — (NAR) | Has | Has | — | — | — | — |
| **zonos** | Has (quant) | Gap | Has | Gap | Gap | Gap | — |
| **vibevoice** | Has (F16) | Has | — | Partial | Gap | Has | — |
| **qwen3-tts** | Has (quant) | Has | Has | Has (5-bucket) | Has | Partial | — |
| **cosyvoice3** | Has | Gap | — | Gap | Gap | Gap | — |
| **tada** | Has (quant) | Gap | — | Gap | Gap | Gap | — |
| **voxcpm2** | Has (host) | Gap | — | Gap | — | Gap (CPU only) | — |
| **chatterbox** | Has (quant) | Gap (stub) | — | Gap | Gap | Gap | — |
| **pocket-tts** | Has (host) | Gap | — | Gap | — | — | — |
| **lfm2-audio** | Has (F16) | Has | Has | Gap | — | Has | Gap (stub) |
| **mini-omni2** | Has (quant) | Has | — | Gap | — | — | — |
| **kugelaudio** | Has | Has | — | Gap | — | — | — |
| **pcs** | — (enc) | Has | — | — | — | — | — |

#### 1b. Encoder-only / non-AR backends

| Backend | Flash attn | BN fold | Fused QKV | Graph cache | GPU path |
|---|:-:|:-:|:-:|:-:|:-:|
| **paraformer** | Has | — | Has (cross) | Gap | Gap |
| **parakeet** | Has (opt-in) | Has | — | Gap | Has |
| **sensevoice** | Has | — | Has | Gap | Has |
| **wav2vec2** | Has | — | — | Gap | Has |
| **fireredpunc** | Has | — | — | Gap | Has |

#### 1c. Support runtimes (VAD, LID, speaker, etc.)

| Backend | Context cache | GPU inference | Flash attn | BN fold |
|---|:-:|:-:|:-:|:-:|
| **silero-vad** | Has | — | — | — |
| **encdec-vad** | Gap | Gap (CPU only) | Has | — |
| **marblenet-vad** | Gap | Gap (CPU only) | — | — |
| **firered-vad** | Gap | Gap (loads GPU, runs CPU) | — | — |
| **pyannote-seg** | Gap | — (correct: tiny model) | — | — |
| **ecapa-lid** | Gap | Has (partial) | — | Gap |
| **silero-lid** | — | Gap (loads GPU, runs CPU) | Gap | — |
| **firered-lid** | — | Has (via firered-asr) | Has | — |
| **cld3** | — (correct) | — (correct) | — | — |
| **fasttext-lid** | — | — | — | — |
| **titanet** | Gap | Gap (loads GPU, runs CPU) | — | Has |
| **audioseal** | Has (sched) | Has | — | — |
| **rnnoise** | Gap | — | — | — |

---

### 2. Per-runtime detail

#### ASR — Encoder-only / NAR

**Paraformer** (`paraformer.cpp`):
- Has: flash attn, fused cross-KV, pre-alloc compute meta, sched reuse,
  NAR decode (single pass), LFR 6× frame reduction, CIF on CPU
- Gap: no persistent graph (rebuilt per call), no GPU backend, CIF conv
  is scalar C++ (D=512, K=3 — no SIMD), 256 MB static compute buffer
  never freed, no chunking for long audio

**Parakeet** (`parakeet.cpp`):
- Has: flash attn (opt-in), BN folding at load, lazy weight cache for
  predictor+joint, per-frame encoder projection reuse, 3 transcription
  modes (single/chunked/streamed), TDT duration head, CTC-WS hotword
  trie, MAES beam search, auto thread count
- Gap: LSTM predictor+joint head are scalar C++ (H=640, 2 layers — no
  BLAS), no persistent encoder graph, chunked encoding is serial (could
  parallel-encode), mel window/filterbank re-read from GGUF per call,
  sinusoidal PE recomputed per call, temperature sampling with full
  CDF scan (Gumbel-max would be faster)

**SenseVoice** (`sensevoice.cpp`):
- Has: encoder-only single forward, flash attn (70 SANM blocks), fused
  QKV, sinusoidal PE precomputed at load, single ggml graph for full
  model, GPU backend, CTC prefix beam search with gamma pruning
- Gap: graph rebuilt per call (70 layers each time), no chunking for
  long audio (O(T²) attention per block), log-softmax for beam search
  in C++ loop (could be ggml op), `maybe_snap` emits 70 unconditional
  dup nodes in production

**Wav2Vec2** (`wav2vec2-ggml.cpp`):
- Has: full transformer in ggml, tensor cache for D2H copies, grouped
  conv via im2col+matmul, OMP in CPU pos_conv, quantized weight support,
  MAES gamma-pruned beam decode
- Gap: CNN feature extractor is manual scalar C++ (7-layer, <5% of
  total but no SIMD), fresh gallocr per call, global static tensor
  cache (thread-unsafe), 15× ggml_concat for grouped conv assembly

#### ASR — Encoder-decoder

**Canary** (`canary.cpp`):
- Has: BN folding, KV cache with quant (Q8_0/Q4_0), cross-KV precomputed
  once, flash attn (SA+CA), KV on-CPU spill, on-device beam KV snapshot
  pool, prefill batching, chunked streaming with LCS dedup, degenerate
  loop guard
- Gap: no layer offload, cross-KV freed+rebuilt per call, decode graph
  rebuilt per step, bespoke C++ FFT

**Kyutai STT** (`kyutai_stt.cpp`):
- Has: single ggml graph for Mimi encoder, combined QKV weight, flash
  attn (Mimi+LM), layer scale, ggml_arange in-graph, KV cache (quant),
  frame-aligned timestamps, streaming API, per-token callback, beam
  search, SwiGLU, RoPE
- Gap: RVQ encode is brute-force O(T×32×2048×256 — no FAISS/PQ), linear
  interp resampler (aliasing), no layer offload, LM graph rebuilt per
  frame, Mimi graph rebuilt per call, streaming re-encodes all audio

**FireRed ASR** (`firered_asr.cpp`):
- Has: CPU Q4_K SIMD dispatch, vecmat for quantized matmul, cpu_dot with
  4 accumulators, OMP cpu_matmul_bt, cross-KV precomputed, persistent
  GPU decoder projection graph, hybrid ggml/CPU encoder, beam search
  with lazy full-weight cache, CTC fallback
- Gap: **no KV cache for decoder SA** (growing vector, O(T²) with no
  SIMD — highest-impact gap), 128+ tiny ggml graphs per decode step
  (one per matmul), rel-pos attention scalar C++, Conv2d subsample
  scalar C++, ggml encoder drops position attention term

**Moonshine** (`moonshine.cpp`):
- Has: cross-KV precomputed, self-KV cache with view writes, flash attn
  everywhere, partial RoPE, weight tying detection, xorshift64 RNG,
  on-device beam KV snapshot pool, encoder output freed after cross-KV
- Gap: no persistent graph (rebuild per step), KV re-allocated per call,
  no early exit / length estimation

**Moonshine Streaming** (`moonshine_streaming.cpp`):
- Has: cross-KV precomputed, flash attn with per-layer sliding window
  masks, separate self/cross KV, SiLU-gated FFN, learnable pos embed
  for cross-attn, beam search, rolling-window streaming API, per-token
  callback
- Gap: audio frontend is scalar C++ (2 CausalConv1d, not ggml), decoder
  step graph rebuilt per token, KV allocated fresh per call, sliding
  window mask rebuilt per call per layer (O(T²) × n_layers)

#### ASR — LLM-decoder

**FunASR** (`funasr.cpp`):
- Has: KV cache with zero-init, fused QKV at load (§136), flash attn
  (encoder+adaptor), GQA 16/8, SwiGLU, YaRN RoPE, sinusoidal PE at
  load, last-token-only logit projection, optional step-graph cache
  (disabled), kv_indices/set_rows for graph-cache compat, encoder GPU
  split, per-stage bench, NaN checker, degenerate guard
- Gap: step graph cache disabled (full-window attend), beam search is
  replay O(beam×suffix×full_forward), encoder graph rebuilt per call,
  maybe_snap unconditional in encoder, embed_tokens builds tiny graph
  per step, adaptor QKV not fused

**Voxtral 3B** (`voxtral.cpp`):
- Has: KV cache, fused QKV (7-8% decode speedup), layer offload, 4-frame
  stack projector, Tekken tokenizer with lazy reverse-map, flash attn
  flag
- Gap: flash attn not applied to encoder, encoder graph rebuilt per
  call, no streaming, Tekken vocab stored as F32 tensor

**Voxtral 4B** (`voxtral4b.cpp`):
- Has: fused QKV, adaptive RMSNorm scales precomputed at init, SWA
  (encoder 750, LLM 8192), causal conv1d, native streaming API with
  live decode, tied embeddings, layer offload, thread-local FFT scratch,
  KV cache
- Gap: ada_scale precompute is scalar CPU, SWA mask rebuilt per call,
  FFT memcpy per frame, flash attn not wired for encoder

**GLM ASR** (`glm_asr.cpp`):
- Has: KV cache (quant), KV on-CPU, layer offload (LLM), flash attn,
  last-token-only head, prefill batching, 4-frame stacking, partial
  RoPE, SwiGLU, multi-EOS, zero-fill KV reset
- Gap: mel padded to 3000 frames regardless of audio length, encoder
  graph rebuilt per call, FFT allocates per-window, beam search is
  replay (no KV snapshots)

**Granite Speech** (`granite_speech.cpp`):
- Has: BN folding, Shaw RPE precomputed per layer, KV cache (quant),
  KV on-CPU, layer offload, flash attn (LLM), dual encoder paths
  (CPU/ggml graph), block-local attention (ctx=200), frame stacking 2×,
  Q-Former projector (3 tokens), mid-CTC, GQA 16/4, per-stage bench
- Gap: cpu_linear is naive F32 dequant matmul (no SIMD, no OMP),
  depthwise_conv_1d_cpu is scalar loops, Q-Former 3-token bottleneck
  lossy for long audio, encoder graph rebuilt per call
- §210 bucketed decode (PR #207): single-token LLM decode runs a cached,
  shape-stable graph (fixed `Lk` bucket) so CUDA-graph capture engages
  (~9–13× on RTX 5090, bit-identical). On backends without capture
  (Metal/Vulkan/CPU, and not a CPU layer-split) the same graph is allocated
  **once** via a persistent `ggml_gallocr` on the single GPU backend and reused
  every step (`granite_dec_use_gallocr`), skipping the per-step
  `ggml_backend_sched_reset`+`sched_alloc_graph` that capture requires.
  Default-on; opt out with `CRISPASR_GRANITE_DEC_GALLOCR=0`.
  Measured on M1 Metal (granite-4.1-2b Q4_K, `CRISPASR_GRANITE_DEC_PROFILE=1`):
  per-step sched alloc ≈3 ms quiet (balloons to 68–236 ms under memory
  pressure) → gallocr drops it to ~0.02 ms; n_cb (1/2/4) makes no difference.
  The dominant per-step cost (~100 ms on this box) is **Metal per-op dispatch**
  of the ~280-op graph — the launch-bound cost CUDA capture removes but Metal
  cannot without indirect command buffers (ICB; not yet in ggml-metal). So on
  Metal gallocr is a modest quiet-machine win + a large robustness/variance win
  under load; the real remaining lever is ICB graph replay.

**Qwen3 ASR** (`qwen3_asr.cpp`):
- Has: KV cache, lazy crisp_audio init, optional fused QKV, layer offload,
  sinusoidal PE at load, batched conv for audio chunking, thread-local
  FFT scratch, forced-aligner variant auto-detect, chunked windowed
  attention mask
- Gap: FFT memcpy per frame, encoder graph not cached by shape, dual
  GGUF load for audio tower (could share), no fused QKV for encoder

**OmniASR** (`omniasr.cpp`):
- Has: flash attn, KV cache (quant), KV on-CPU, layer offload, GPU-side
  argmax, token embedding via get_rows, separate enc/dec graphs,
  logit/argmax mode, per-call perf timing, beam search, streaming auto-
  detect, grouped pos_conv
- Gap: encoder graph rebuilt per call, 15× ggml_concat for grouped conv,
  no fused QKV in encoder

**MIMO ASR** (`mimo_asr.cpp`):
- Has: KV cache (quant), cached T=1 step graph (skip_plan), fused QKV,
  GPU embed split, flash attn (audio transformer), layer offload, beam
  search, token streaming callback
- Gap: step graph reset per transcribe call even if params unchanged,
  audio transformer has no KV cache (bidirectional), mimo_tokenizer
  is separate GGUF context, no flash attn in LLM decoder

**MOSS Audio** (`moss_audio.cpp`):
- Has: KV cache, sinusoidal PE at load, DeepStack tap capture, SwiGLU,
  attention windowing mask, beam search, token streaming, seed-controlled
  sampling, conv weight layout fix
- Gap: **encoder flash attn not wired** (32 layers, highest-impact gap),
  encoder graph rebuilt per call, no layer offload, no fused QKV for LLM

**Nemotron** (`nemotron.cpp`):
- Has: per-layer streaming K/V+conv cache, ggml graph cache keyed by
  (layer, T_new, T_cache), flash attn, banded window mask, scalar CPU
  LSTM (weights copied once), lazy sched init, pre-alloc compute meta,
  ggml_siglu_swapped, streaming context presets
- Gap: LSTM/joint head no BLAS/SIMD (H=640), cache eviction O(N) vector
  erase (should be ring buffer), custom FFT no SIMD, sinusoidal PE
  recomputed per chunk, flash attn default off

**Gemma4 E2B** (`gemma4_e2b.cpp`):
- Has: dual KV cache (sliding-window + full-context), KV sharing via
  donor layers, QAT clipping baked per Linear, per_dim_scale baked at
  load, PLE as one large matmul + strided views, logit softcapping,
  layer offload, beam search, token streaming, thread-local FFT scratch,
  partial rotary
- Gap: **conformer uses manual per-block matmul for rel-pos attention**
  (cannot use flash_attn_ext due to additive bias — fundamental
  constraint, but no SIMD either), mel generated at runtime (not cached
  in GGUF), block concat via O(num_chunks) ggml_concat, no audio
  caching between calls

#### TTS

**Kokoro** (`kokoro.cpp`):
- Has: LRU phoneme cache (1024 entries), pre-permuted ConvT weights,
  Metal ConvT CPU pinning workaround, dual backends+schedulers, snake-α
  in graph, iSTFT with Hermitian symmetry, AdaIN/AdaLN
- Gap: BERT encoder graph rebuilt per call, LSTM no SIMD, BERT output
  not cached (only phoneme strings), iSTFT twiddle factors recomputed
  per call, flash attn unused

**Bark** (`bark_tts.cpp`):
- Has: KV cache with growth-amortized realloc, flash attn (causal),
  weight norm folding (EnCodec), fused QKV, min_eos_p early exit,
  CPU context merge
- Gap: EnCodec decoder entirely scalar CPU (SEANet conv+LSTM — should
  use convt1d_decomp), weight norm re-folded per call (should be at
  init), KV freed per call, fine-stage re-embeds overlapping windows

**MeloTTS** (`melotts.cpp`):
- Has: pre-permuted ConvT weights, HiFi-GAN in single ggml graph,
  speaker gin injection
- Gap: **cpu_multihead_attention_relpos** is O(H×T²×D) scalar loops
  (6 layers — highest-impact gap), mini_graph pattern with constant
  malloc/free, scalar dds_conv, read_tensor_f32 re-reads weights per call

**OuteTTS** (`outetts.cpp`):
- Has: KV cache (quant, lazy alloc, persistent), KV backend spill, BPE
  tokenizer, xorshift64* + partial_sort sampling, repetition penalty,
  non-audio token early exit
- Gap: flash_attn declared but not wired, use_gpu=false default, no
  layer offload, embed_tokens builds mini-graph per step

**Orpheus** (`orpheus.cpp`):
- Has: KV cache (quant), layer offload, GQA 3:1, RoPE (theta=500000),
  SNAC lazy-loaded, xorshift64* sampling, n_dropped early exit
- Gap: flash_attn not wired (PLAN #86 — highest priority), embed_tokens
  builds mini-graph per step

**Parler TTS** (`parler_tts.cpp`):
- Has: T5 encoder cached after set_description, cross-KV pre-projected
  once (24 layers, F16 §176i), pre-permuted DAC ConvT weights, prefill
  in single step, delay pattern, top-k+temperature, all_eos early exit
- §176b+c (opt-in `CRISPASR_PARLER_BUCKET=1`): Lk-bucketed decode graph
  cache (no per-step rebuild), device-resident self-attn + cross-attn KV
  (`ggml_set_rows` write, no per-step re-upload, no growing `ggml_concat`),
  dedicated sched reused across steps. ~1.2–1.9× faster than the legacy
  default on M1 Metal (F16); crispasr-diff parity 108/108 vs F32 ground
  truth. Default stays the legacy path (byte-identical); CUDA unvalidated.
- Gap: legacy default path still rebuilds the graph per step / re-uploads
  growing KV; flash_attn unused; read_embed_row per codebook per step.

**SpeechT5** (`speecht5_tts.cpp`):
- Has: host-side KV cache for decoder, cross-KV precomputed once (F16
  §176i, via ggml_cpy auto-conversion), BN fusion in postnet,
  pre-permuted HiFi-GAN ConvT weights, separate mini_graphs per stage,
  early exit via stop probability, sinusoidal PE precomputed
- Gap: **KV re-uploaded every step** (growing O(step×hidden)), new K/V
  readback per step (sync), no flash attn, graph rebuild every step

**F5-TTS** (`f5_tts.cpp`):
- Has: flash attn (22 DiT blocks), RoPE, on-the-fly mel+iSTFT, GPU
  backend, sway sampling
- Gap: **CFG as two serial passes** (should batch B=2), **22 separate
  mini-graphs per ODE step** (1408 round-trips total), **Vocos + text
  encoder entirely scalar C++** (triple-nested loops), mel filterbank
  rebuilt per call

**Dia** (`dia_tts.cpp`):
- Has: pre-allocated KV cache, cross-KV precomputed once (18 layers,
  F16 §176i), encoder runs once, CFG batch B=2 in encoder, GQA with
  repeat_4d, pre-permuted DAC ConvT weights, RoPE
- Gap: past KV re-uploaded every step (18 layers × B=2), cross-KV
  re-uploaded every step (F16 now), new K/V readback per step, graph
  rebuilt per step, no flash attn

**CSM** (`csm_tts.cpp`):
- Has: **device-resident KV cache** (backbone F32, depth F16), KV dtype
  from env, pre-permuted SEANet ConvT weights, fused QKV in Mimi,
  flash attn in Mimi, RoPE, BPE tokenizer, compute meta pre-alloc
- Gap: no flash attn in backbone or depth decoder (16+4 layer Llama),
  RVQ dequant is scalar C++, speaker reference encoding stub (ignored)

**IndexTTS** (`indextts.cpp`):
- Has: **device-resident KV with view writes** (zero host round-trip),
  dynamic KV realloc, flash attn (GPT+Conformer+Perceiver), beam search
  with on-device KV snapshots, prompt prefill in one pass, pre-permuted
  BigVGAN ConvT weights, mel via core/mel.h, repetition penalty,
  return-latent second GPT pass
- Gap: beam search KV copy overhead (host mode = full tensor get/set),
  second GPT forward re-runs full 24L (hidden states not cached from
  AR decode), no quantized KV cache

**FastPitch** (`fastpitch_tts.cpp`):
- Has: NAR single forward, flash attn (enc+dec), fused QKV, pre-permuted
  HiFi-GAN ConvT weights, speaker embedding via get_rows, three-phase
  graph, pitch shift additive, pace control
- Gap: decoder+vocoder graph forces CPU (bypasses scheduler), pitch
  embedding in separate tiny graph, mel transpose done on CPU loop

**OpenVoice2** (`openvoice2.cpp`):
- Has: pre-permuted HiFi-GAN ConvT weights, ref encoder embedding
  reusable, WaveNet speaker cond precomputed, GPU backend for HiFi-GAN,
  WaveNet GEMM'd via Accelerate (§176d), **STFT now radix-2 FFT** (2026-07-11:
  1182→10.7 ms, 110×; was O(bins·win) scalar DFT)
- Measured breakdown (3 s convert, quiet M1): **hifigan_decode 67 %**,
  ~~stft 27 %~~ (fixed), ref_enc 4 % (one-time, not worth caching),
  enc_q+flow×2 3 %. **Real remaining gap: HiFi-GAN decode** — the WaveNet/
  ConvT vocoder dominates; ref-enc Conv2d+GRU scalar is a rounding error.

**Zonos** (`zonos_tts.cpp`):
- Has: dual KV caches (quant), fused gate+up, DAC lazy-loaded,
  build_prefix_cpu avoids GPU round-trip for conditioning
- Gap: no graph caching, espeak via popen (should use C API), no flash
  attn, GPU off by default, no layer offload, no CPU embd cache

**VibeVoice** (`vibevoice.cpp`):
- Has: layer offload, pre-permuted ConvT weights, flash attn (decoder),
  dual CFG KV caches, pre-filled voice prompt KV, PyTorch-exact MT19937
  Gaussian noise. **Streaming ASR 1.5B (2026-09-07, Kaggle P100):** native
  2.93 s chunks + 0.53 s lookahead with persistent decoder KV; Q4_K decoded
  the four-chunk JFK fixture in 6.82 s versus 8.26 s F16. Both emitted the
  official transcript and all 34 official token IDs exactly. F16 prompt-logit
  cosine was 0.9999998 and its four post-delimiter KV-tail cosines were all
  above 0.99999997. Plain Q4_K is 1.86 GB; retaining both acoustic encoders in
  F16 grew it to 2.73 GB without changing the transcript or token IDs, so the
  plain Q4_K arm is the shipped artifact. Kernel:
  `chr1str/crispasr-vibevoice-streaming-426-q4`, commit `f9f96dd2`.
- Gap: acoustic+semantic encoders run serial (could fuse/parallel),
  pred head graph rebuilt per DPM step, no CPU embd cache, DPM schedule
  coefficients recomputed per call

**Qwen3 TTS** (`qwen3_tts.cpp`):
- Has: **CP_DIRECT sched-free code predictor** (§232, default ON GPU /
  OFF CPU — two persistent gallocr graphs, no scheduler; −11 % CUDA,
  ~3× under Metal load, md5-identical; supersedes O15 and fixes its #56
  CUDA crash), **codec FASTCONV** (§232, default ON — K=1-conv→matmul +
  pad-in-im2col + load-time F32 kernel bake; codec 3× Metal / 2.1× CPU,
  0.6B pipeline RTF 2.9→1.25), **CPU embedding cache**, per-op profiler
  + per-node codec trace, fused QKV, dedicated talker scheduler, flash
  attn throughout. Lk-bucketing (opt-in, sched-free since §232 — Metal
  loses ~10 % at short outputs, fastest config on CUDA P100).
- Gap: talker AR step is **GPU-execution-bound** (§232 profiled: encode
  2-3 ms vs GPU 38-42 ms — dispatch tricks can't help; needs kv_self_attn
  node-count slimming, shared 30-backend helper, deprioritized); codec
  residual cost is the legitimate K=7 SEANet im2cols; no speculative
  decoding.

**CosyVoice3** (`cosyvoice3_tts.cpp`):
- Has: RAS sampler, multi-GGUF pipeline, voice packs with precomputed
  conditioning, persistent KV, Euler ODE with cosine schedule, CFG
- Gap: **HiFT source DFT is O(n²)** (should be FFT), no graph caching
  for LLM steps, no CPU embd cache, flash attn unclear, fixed 10 Euler
  steps (DPM-Solver++ would halve), no layer offload

**TADA** (`tada_tts.cpp`):
- Has: dual KV caches (quant), KV zero-init, pre-loaded voice prompt,
  separate graphs for embedding+FM steps
- Gap: no graph caching for AR steps, no FM graph reuse (fixed topology),
  no flash attn, no CPU embd cache, no layer offload

**VoxCPM2** (`voxcpm2_tts.cpp`):
- Has: streaming API (unique), LongRoPE, FSQ bottleneck (no codebook
  lookup)
- Gap: **CPU-only** (Metal SIGSEGV — buffer type mismatch), manual KV
  cache (vector per layer, host upload/download per step), no graph
  caching, no flash attn

**Chatterbox** (`chatterbox.cpp`):
- Has: two T3 variants (Llama/GPT-2), precomputed conditioning in GGUF,
  PyTorch-exact MT19937, multilingual language tokens
- Gap: **flash attn is a stub** (PLAN #86 — 520M Llama AR with up to
  1000 tokens), no Lk-bucketing, no CPU embd cache, no S3Gen CFM graph
  reuse, no layer offload

**Pocket TTS** (`pocket_tts.cpp`):
- Has: pre-permuted Mimi ConvT weights, flow net precomputed sinusoidal
  embed, eos_threshold early exit, F16 dequant cache
- Gap: **per-step graph rebuild is critical bottleneck** (O(pos×NH×HD)
  KV reorder per step), F16 cache is thread-unsafe + unbounded, flow
  net is scalar CPU, no flash attn, Mimi decoder rebuilt per call

#### Audio-LLM / S2S

**LFM2 Audio** (`lfm2_audio.cpp`):
- Has: KV cache (F16 4D), conv state cache per layer, dual compute-meta
  buffers, flash attn (FastConformer+backbone), GQA 32/8, fused QKV,
  NeMo mel with z-norm, lazy detokenizer, streaming TTS API, layer
  offload
- Gap: beam search stub, depthformer KV is manual CPU F32, acoustic
  connector graph rebuilt per call, VAE decoder graph rebuilt per token

**Mini-Omni2** (`mini_omni2.cpp`):
- Has: KV cache (quant), flash attn (Whisper+Qwen2), causal mask only
  for prefill, last_token_only view, 8-stream dual output, SNAC decode
- Gap: 8 separate embed_tokens graph invocations per step (should batch),
  O(N²) DFT (n_fft=400 not power-of-2), rand() unseeded, no beam
  search, adapter graph rebuilt per call, no streaming audio output

**KugelAudio** (`kugelaudio.cpp`):
- Has: KV cache (F16), flash attn GQA 28/4, RoPE NEOX, pre-permuted
  ConvT weights, MT19937 RNG, DPM-Solver++ 2nd-order, cosine beta
  precomputed, constrained argmax (4 tokens)
- Gap: **CFG not implemented** (allocated but unused), acoustic connector
  rebuilt per frame, VAE decoder rebuilt per token, pred head rebuilt per
  diffusion step, no streaming audio output

**PCS** (`pcs.cpp`):
- Has: ggml_backend_sched, flash attn (12 encoder layers), enc+punc heads
  on GPU, chunking at 512 tokens, SentencePiece tokenizer
- Gap: SBD+truecase heads are scalar CPU loops, post_punc weight re-read
  from GPU per call, hidden state readback serializes GPU/CPU, no
  batching

#### VAD

**Silero VAD** (`crispasr_vad.cpp`):
- Has: static cached context (avoids 70× init/free regression),
  energy-minima splitting, binary search timestamp remap
- Gap: caching is Silero-specific (other VAD engines not cached)

**WhisperEncDec VAD** (`crispasr_vad_encdec.cpp`):
- Has: flash attn, fused QKV, thread-local FFT scratch, F16 K/V cast,
  30s windowed processing, auto-threshold lowering
- Gap: no context caching, CPU-only hardcoded, KV rebuilt per call

**MarbleNet VAD** (`marblenet_vad.cpp`):
- Has: depthwise conv via ggml_conv_1d_dw, F16 weight cast, BN pre-fused
- Gap: no context caching, thread count hardcoded to 4, no GPU path

**FireRed VAD** (`firered_vad.cpp`):
- Has: CMVN, hysteresis segmentation
- Gap: **all 8 DFSMN blocks + linear layers are scalar CPU loops** (no
  ggml, no BLAS, no SIMD despite loading via ggml_backend_init_best()),
  mel filterbank recomputed per call

**Pyannote** (`pyannote_seg.cpp`):
- Has: ggml-graph forward (§224, default): SincNet + classifier as CPU-backend
  graphs, LSTM input projections batched per layer/dir as one mul_mat, the
  sequential recurrence dual-threaded (one thread per direction). 4.38 s →
  0.55 s on 31.5 s audio (M1), output frame-identical.
  `CRISPASR_PYANNOTE_LEGACY=1` restores the scalar path (A/B ground truth).
- Gap: no context caching; recurrence R@h still plain (autovec) loops

#### LID

**ECAPA-TDNN LID** (`ecapa_lid.cpp`):
- Has: GPU for Conv1d trunk, mel filterbank in GGUF, 15s audio cap; ASP + FC
  head in-graph (§224, titanet-style). Inverse-default: in-graph WITHOUT
  Accelerate (scalar head was ~3.0 s of a 4.5 s detect → in-graph removes it);
  WITH Accelerate the GEMM head (57 ms) stays default.
  CRISPASR_ECAPA_ASP_GGML=1 / CRISPASR_ECAPA_ASP_CPU=1 force either way.
- Gap: trunk graph ~1.3 s dominates on M1 (profile Metal residency); BN not
  pre-folded

**Silero LID** (`silero_lid.cpp`):
- Has: ggml-graph forward (§224, default): CPU frontend (log-mag precision) +
  sched encoder (GPU on Metal/CUDA; auto-routed to CPU on Vulkan pending an
  upstream mul_mat fix), 30 s slice cap. 103 ms Metal / 183 ms CPU-ggml vs
  241 ms Accelerate / 1014 ms scalar legacy (11 s audio, M1).
  `CRISPASR_SILERO_LID_LEGACY=1` restores the scalar path.
- Gap: Vulkan GPU blocked on the ggml-vulkan FFN MUL_MAT miscompute

**FireRed LID** (`firered_lid.cpp`):
- Has: reuses full firered_asr runtime
- Gap: runs full ASR pass for single language token (extreme overkill)

#### Speaker

**TitaNet** (`titanet.cpp`):
- Has: BN pre-folding at init, pre-emphasis, L2 normalization; ggml-graph
  forward (§224): full encoder + ASP as one CPU-backend graph, embeddings
  cos=1.000000 vs legacy. Inverse-default: ggml is default WITHOUT Accelerate
  (scalar was 106–131 s/embed on M1 → 3.4 s ggml, ~35×); WITH Accelerate the
  legacy AMX GEMM path stays default (0.7 s vs 3.4 s ggml).
  `CRISPASR_TITANET_GGML=1` / `CRISPASR_TITANET_LEGACY=1` force either way.
- Gap: ggml F32 matmul ≪ AMX on Apple (F16 weights would halve bandwidth);
  segments not batched

#### Translation

**M2M-100** (`m2m100.cpp`):
- Has: cross-KV cache, decoder self-KV (F16 4D), flash attn (enc SA,
  dec CA), beam search, prefill batching
- Gap: CPU-only, encoder graph rebuilt per call, KV reallocated per call
- §176i (2026-06-21): cross-KV F32 → F16 (halves cross-KV memory)

**T5 Translation** (`t5_translate.cpp`):
- Has: cross-KV cache, decoder self-KV, Viterbi DP tokenizer, rel-pos
  bias in-graph, RMSNorm, gated-GELU FFN, F32 precision on KQ matmul
- Gap: CPU-only, manual encoder attention (can't use flash_attn_ext with
  additive rel-pos bias), position bucket indices recomputed per step,
  encoder rebuilt per call
- §176i (2026-06-21): cross-KV F32 → F16 (halves cross-KV memory)

#### Other

**AudioSeal** (`audioseal.cpp`):
- Has: pre-permuted ConvT weights, lazy sched, GPU, LSTM ih_all
  precomputed, additive watermark (no full reconstruction)
- Gap: graph rebuilt per call, LSTM concat chain O(T) nodes, no
  streaming/chunking

**RNNoise Enhancement** (`crispasr_enhance.cpp`):
- Has: miniaudio resample, zero-fill guarantee
- Gap: resampler init/destroy per call, DenoiseState created per call,
  linear resampler quality

**FireRedPunc** (`fireredpunc.cpp`):
- Has: flash attn (12 BERT layers), GPU, sched persistent, chunking at
  510, dual tokenizer, CJK/Latin heuristic
- Gap: ggml_context rebuilt per call, WordPiece re-tokenization for
  alignment, sharp chunk boundaries

**CTC Forced Alignment** (`align.cpp`):
- Has: Viterbi DP with rolling vectors (O(S) memory), int8 backpointers,
  UTF-8 codepoint-aware, lowercase fallback
- Gap: log-softmax over entire [T×V], backpointer T separate heap allocs

**Diarization** (`crispasr_diarize.cpp`):
- Has: 4-method dispatch, overlap-class speaker accumulation, silence
  gating, per-slice offset mapping
- Gap: Xcorr is O(seg×2×MAX_LAG) — FFT-based would be O(N log N),
  Pyannote re-initialized per call

---

### 3. Shared core infrastructure

**attention.h**: flash attn, fused QKV, GQA (repeat_4d + native), KV
quant (asymmetric K/V), KV on-CPU, on-device snapshot pool, dynamic
index KV write, forced F32 dequant read. Gap: GQA repeat_4d materializes
copy for non-granite models; Metal HOST snapshot still memcpys.

**ffn.h**: fused gate+up (saves ~30 ms/call per code comment), swiglu/
geglu/silu/gelu variants, no-op bias skip. Gap: no fused SiLU*mul kernel.

**conv.h**: convt1d_decomp (matmul+col2im), batch weight permutation,
depthwise stride-2 k3. Gap: extra ggml_cont calls, single-threaded
permute, F16 cast per call in depthwise.

**mel.cpp**: unified 9-family extractor, optional double precision, frame
stacking zero-copy. Gap: **single-threaded STFT loop** (dominant cost for
long audio), **scalar mel projection** (no BLAS — 31M multiply-adds for
NeMo cluster), per-call power/mel buffer alloc.

**kaldi_fbank.cpp**: thread-local fb/window cache, iterative FFT, in-
place pre-emphasis. Gap: per-call fft scratch alloc, scalar mel projection
inner loop (no SIMD).

**fft.h**: thread-local scratch. Gap: **recursive FFT with O(N log N)
heap allocs** — should use iterative version from kaldi_fbank.

**greedy_decode.h**: reserve, early exit, NoHook eliminated, inverse-CDF
sampling, streaming callback, frequency penalty. Gap: sample_temp allocs
vocab-sized double vector every token.

**beam_decode.h**: branched variant O(B×T) with VRAM snapshot pool,
partial_sort, multi-EOS, RAII snapshot management. Gap: full-vocab idx
alloc per beam per step, Beam deep-copies tokens/probs vectors.

**ctc.h**: gamma pruning, partial_sort. Gap: linear-scan prefix
find_or_insert O(B×V), scalar posterior_pool.

**cpu_ops.h**: OMP layernorm, reused compute_meta. Gap: two-pass
layernorm, matmul rebuilds graph every call.

**gguf_loader.cpp**: zero-copy mmap (CPU+Metal), WILLNEED→RANDOM madvise,
mlock, preload, layer-split loader. Gap: two-pass tensor bind, sequential
split partitions.

---

### 4. Top optimization opportunities by impact

Ordered by estimated breadth × depth of impact across the project:

#### Tier 1 — High impact, broadly applicable

1. **Wire flash_attn_ext in all AR decoders** — Orpheus, OuteTTS,
   Chatterbox, Parler, SpeechT5, Dia, Zonos, TADA, Pocket-TTS, MOSS
   encoder, Voxtral/4B encoder, CSM backbone. The `core_attn::kv_self_attn`
   path already supports it; most backends just never pass the flag.
   Estimated: 2-5× attention bandwidth reduction at long sequences.

2. **Lk-bucketed graph caching for AR decode** — applies to every backend
   that rebuilds its decode graph per step (30+ backends). Qwen3-TTS
   demonstrates the pattern with 5 buckets. The MIMO `step_t1_gf` is a
   simpler single-bucket variant. Graph rebuild overhead is 6-14 ms/step
   (measured in funasr); eliminating it across 100-1000 steps per
   synthesis/transcription is the largest single latency win.

3. **Migrate host-side KV re-upload to device-resident KV** — SpeechT5,
   Dia, Parler, Pocket-TTS, VoxCPM2 all use `std::vector<float>` KV
   caches that grow and re-upload every step. IndexTTS and CSM demonstrate
   the correct pattern: 4D on-device tensor with `ggml_view_4d` + `ggml_cpy`
   writes. Eliminates O(step × layers × hidden) host↔device bandwidth.

4. **BLAS/ggml for scalar CPU matmul hotpaths** — TitaNet ASP TDNN
   (9216×T), Silero LID attention (8 stages × O(T²)), FireRed VAD DFSMN
   (8 blocks), MeloTTS/Piper relpos attention (6 layers × O(T²)),
   OpenVoice2 WaveNet (16 layers), Granite cpu_linear, Parakeet LSTM
   predictor. These are the dominant compute in their respective
   runtimes and run as unvectorized nested loops.

#### Tier 2 — Medium impact, moderate effort

5. **Context caching for support runtimes** — WhisperEncDec VAD, MarbleNet
   VAD, FireRed VAD, Pyannote, ECAPA-TDNN LID, RNNoise enhancement, CTC
   aligner, FireRedPunc graph context. The Silero VAD static-cache pattern
   is the template. Eliminates repeated init/free overhead in pipelines
   that call these per-segment.

6. **Parallel STFT in mel.cpp** — the mel spectrogram is the entry point
   for ~40 backends. The STFT loop is single-threaded. Adding `#pragma omp
   parallel for` with per-thread scratch buffers would scale linearly with
   cores. Similarly, the mel projection matmul should use BLAS (`cblas_sgemm`).

7. **CPU embedding cache** — Qwen3-TTS demonstrates caching quantized
   embedding bytes on CPU to avoid Metal command-buffer round-trips per AR
   step. Applies to Orpheus, OuteTTS, Chatterbox, Zonos, CosyVoice3, TADA,
   VibeVoice, Pocket-TTS.

8. **F5-TTS: collapse 22 per-block mini-graphs into one** — currently 1408
   alloc+compute+get round-trips per synthesis. Batch CFG as B=2 in a
   single graph would halve DiT inference time. Port Vocos + text encoder
   from scalar C++ to ggml.

9. **Cross-KV in F16 (not F32)** — M2M-100, T5, SpeechT5, Dia, Parler all
   store cross-attention K/V as F32 on host. Cross-KV is read-only after
   projection; F16 halves memory with no accuracy impact.

10. **Replace recursive FFT (core/fft.h) with iterative** — recursive
    variant allocates O(N log N) heap per call. kaldi_fbank.cpp already has
    the correct iterative in-place version. Also: CosyVoice3 HiFT DFT is
    O(n²) — must use FFT.

#### Tier 3 — Targeted wins

11. **FireRed ASR: add KV cache for decoder self-attention** — currently
    grows a vector and does O(T²) scalar attention. Highest-impact single-
    backend optimization remaining.

12. **Kyutai STT RVQ encode: vectorized search or PQ** — brute-force
    O(T×32×2048×256) codebook search is the dominant cost.

13. **Nemotron: ring buffer for streaming KV cache** — vector erase from
    front is O(N); a ring buffer is O(1).

14. **VoxCPM2: fix Metal buffer type mismatch** — currently CPU-only due
    to SIGSEGV from buffer placement. Unlocks GPU for the entire pipeline.

15. **embed_tokens micro-graph elimination — MOSTLY DONE.** FunASR,
    GLM-ASR, MOSS-Audio, Qwen3-ASR, Gemma4-E2B shipped with direct CPU
    dequant (`CRISPASR_XXX_EMBED_FAST`). Orpheus/OuteTTS already had it.
    ~1.6× embed step. Only granite-speech remains.

16. **read_tensor_f32 weight pre-cache — DONE (piper 14%, melotts 16%).**
    `CRISPASR_PIPER_WEIGHT_CACHE` / `CRISPASR_MELOTTS_WEIGHT_CACHE`.

### VPS bench data (2026-06-20, 4-core CPU, Q4_K/F16, JFK 11s)

| Backend | Stage | Time (ms) |
|---------|-------|-----------|
| paraformer-zh Q4_K (123 MB) | fbank+lfr | 142 |
| | encoder | 5007 |
| | cif_predict | 461 |
| | decoder | 522 |
| nemotron Q4_K (458 MB) | mel | 143 |
| | encoder | 31411 |
| | rnnt_decode | 7838 |
| canary-ctc Q4_K (434 MB) | encoder+ctc | 43235 |
| piper-en F16 (30 MB) | text_encoder | 939 |
| | flow_inverse | 5187 |
| | hifigan_decode | 5487 |
| | **total** | **11790** |
| melotts-en F16 (98 MB) | text_encoder | 407 |
| | flow_inverse | 2579 |
| | hifigan_decode | 17947 |
| | **total** | **26272** |
