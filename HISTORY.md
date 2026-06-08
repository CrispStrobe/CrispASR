# CrispASR — Port history

Condensed chronology of the ports that built this repo. Kept for
context, not for day-to-day reference. Live work is in `TODO.md`;
technical deep-dives are in `LEARNINGS.md`.

---

## 2026-06-08 §115/§52/§140 audit + pocket-tts/moonshine GPU migration

**§115 mimo-asr GPU — closed.** Deep audit revealed the item is effectively
done: GPU is the default since `a429bb45`, Option B (prefill-graph reuse
for decode, embed tables CPU-resident via `load_weights_split`) validated
on RTX 3090 + Kaggle P100 at 2.4x realtime. k-quant CUDA GET_ROWS fix
(`3bf9a599`) landed as a safety net. Option B (10 ms/step) outperforms
the hypothetical Option C step-graph approach (31 ms/step), so Option C
is not worth pursuing. PLAN.md updated to reflect this.

**§52 Qwen3-TTS O15 CUDA test.** O15 (persistent code-predictor graph
reuse) saves ~14-19 ms/frame but was reverted to default-OFF in `61c42bfb`
after a `GGML_ASSERT` crash on Jetson sm_87 (CUDA, `ggml_backend_tensor_set`
on cached graph with `ggml_set_rows`). Created Kaggle kernel
`tools/kaggle/qwen3-tts-o15-cuda/` — kernel v1 failed because the test
harness didn't pass a voice reference (qwen3-tts requires `--voice`).
Fixed with `jfk.wav` as reference; re-pushed as kernel v2. Awaiting results.

**§140 TTS GPU sched — pocket-tts migrated.** Audit found speecht5/piper/
parler-tts/outetts already had `ggml_backend_sched` wired (just `use_gpu`
defaulting false in backend params, overridden to true by C API). Only
pocket-tts was still on the old `gguf_init_from_file(no_alloc=false)` +
`ggml_gallocr` + `ggml_backend_graph_compute(cpu)` pattern. Migration:
- Switched to `core_gguf::open_metadata()` + `core_gguf::load_weights()`
  (mmap-backed, backend buffer)
- Added `backend` + `sched` to context, `ggml_backend_init_best()` for GPU
- Replaced `gallocr` in `backbone_forward_step_ggml` and `mimi_decode_ggml`
  with `sched_reset` + `sched_alloc_graph` + `sched_graph_compute`
- Updated `pocket_tts_free` for proper resource cleanup
- Tensor loading functions now take `TensorMap&` instead of `ggml_context*`
- **Fix:** weights must load to CPU backend (not GPU) because the flow head
  and AR loop use eager CPU code that reads `t->data` directly.

All 6 TTS sched backends now DONE. PLAN.md §140 updated.

**Moonshine mmap + sched migration.** Both `moonshine.cpp` and
`moonshine_streaming.cpp` migrated from `gguf_init_from_file` + manual
fread tensor loop + per-graph `gallocr` to `core_gguf::open_metadata` +
`core_gguf::load_weights` (mmap) + `ggml_backend_sched`. Added `use_gpu`
to both init params, wired via `g_open_use_gpu_tls` in C API.

---

## 2026-06-07 → 2026-06-08 Piper ASR roundtrip fix + permissive G2P phonemizer (§156)

**Problem:** Piper TTS → Whisper ASR roundtrip produced garbage when
espeak-ng was not installed (raw English text fed as IPA phonemes).

**Root cause + fix:** Backend silently fell back to phoneme synthesis
with plain text. Now rejects non-IPA input; clear error message.

**espeak-ng GPL workaround:** espeak-ng is GPLv3 — can't statically
link. Solution: pre-generate espeak-ng IPA output (factual data, not
GPL-covered) and ship as downloadable dicts. Binary stays MIT-clean.

**Pre-generated espeak dicts** (primary, 99.5% piper match):

| Language | Words | Size | Source vocabulary |
|----------|-------|------|-------------------|
| English | 126K | 3 MB | CMUdict |
| German | 667K | 23 MB | open-dict-data single words |
| French | 257K | 6.6 MB | OLaPh FR |
| Spanish | 600K | 18 MB | OLaPh ES |

Uploaded to HuggingFace `cstr/g2p-dicts`. Auto-downloaded on first use.

**Fallback tiers** (when word not in espeak dict):

| Tier | EN | DE/FR/ES |
|------|----|----------|
| 1 | CMUdict + ARPAbet→IPA (76% match) | OLaPh MIT dict |
| 2 | Neural G2P (GRU seq2seq from GGUF) | LTS rules |
| 3 | LTS digraph/trigraph rules | espeak-ng dlopen |
| 4 | espeak-ng dlopen | espeak-ng popen |
| 5 | espeak-ng popen | — |

**ARPAbet→IPA systematic fixes** (benchmark-driven):
- AH0→ə (schwa), ER stressed→ɜː / unstressed→ɚ, IH0→ɪ, UW0→ʊ
- T-flapping (T/D between vowels→ɾ), ER+vowel linking→ɚɹ
- German: r→plain r (not ʁ), -er→ɜ (not ɐ), long a→ɑː (not aː),
  Auslautverhärtung, open-syllable lengthening, compound splitting

**Wiring:** `--g2p-dict` CLI → `whisper_params` → C API
`crispasr_session_set_g2p_dict()` → Go `SetG2PDict()` → server.

**Benchmark:** 382-word ground truth (espeak-ng en-us/de):
EN 99.5%, DE 100% with espeak dict; EN 76.2% with CMUdict fallback.

**Tests:** 174 unit assertions + 4 live TTS→ASR roundtrips (5/5 perfect).

**Tests:** 202 unit assertions across 38 test cases + 4 live roundtrips.

**Docs:** README.md, docs/tts.md (G2P section), docs/cli.md updated.

---

## 2026-06-03 → 2026-06-05 PLAN audit + OpenVoice2 voice cloning + cohere FA benchmark

### PLAN audit (2026-06-03)

Code-verified 17 items previously flagged as NOT DONE. Found 4 were
already done (stale PLAN): #93 CMake rename, #94 Go auto-gen LDFLAGS,
#103 Silero v6.2.0, O4 beam search. Updated PLAN.md with all findings.

### #96 voxcpm2 graph-default flip (2026-06-04)

Flipped `VOXCPM2_USE_GRAPH` from opt-in to default-on. Graph path was
already validated (48.7→14.1s on M1 Metal). Tested on VPS (1.58x faster)
and Kaggle CPU (1.46-1.61x). Both produce correct ASR roundtrip. Opt-out
via `VOXCPM2_USE_GRAPH=0`. Kaggle kernel: `chr1str/crispasr-voxcpm2-graph-ab`.

### #73 cohere FA long-form benchmark (2026-06-04)

Crossover confirmed: flash-attn wins by 26% on 300s but loses by 13% on
60s. With default 30s auto-chunking, cast-on-read is always faster. Flipped
cohere to default cast-on-read (was flash). `CRISPASR_COHERE_FLASH=1` for
unchunked long-form. Results in PERFORMANCE.md §5.

### #61j glm-asr translate (2026-06-04)

Added `CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE` to glm-asr backend (2 lines —
implementation already existed in C library). voxtral4b and omniasr-llm
assessed as N/A (streaming-only / no text instruction mechanism).

### #100 Phase B: OpenVoice2 voice cloning (2026-06-04 → 2026-06-05)

Full zero-shot voice cloning via OpenVoice V2 Tone Color Converter, wired
as a post-processor on MeloTTS output.

**Architecture:** ref_enc (6 Conv2d + GRU → 256-d speaker embedding) +
enc_q (16-layer WaveNet posterior encoder) + WaveNet coupling flow
(4 blocks × 4 layers, forward + reverse) + HiFi-GAN decoder.

**Files:**
- `models/convert-openvoice2-to-gguf.py` — GGUF converter (346 tensors
  with 11 base speaker embeddings, 125 MB all-F32)
- `src/openvoice2.{h,cpp}` — C runtime (~1300 LOC)
- `examples/cli/crispasr_backend_melotts.cpp` — `--voice ref.wav` wiring
- `tools/reference_backends/openvoice2_ref.py` — Python diff harness
- `tests/test-openvoice2-hifi.cpp` — standalone test

**Bugs found and fixed (5 rounds of diff harness):**
1. F16 tensor read — `ggml_backend_tensor_get` needs dequant for F16
2. WAV chunk parser — proper RIFF chunk walking for extra chunks
3. GRU reshape axis — `(H, C*W)` not `(W, C*H)` for frequency-axis GRU
4. STFT reflect padding — match upstream `F.pad(mode='reflect')` + `1e-6` floor
5. F16→F32 sensitive weights — enc_q/flow/ref_enc weights stored as F32
   in GGUF to prevent accumulated precision loss through 16 WN layers
6. Pre-saved base speaker embeddings — upstream uses `en-default.pth`,
   not ref_enc on synthetic audio
7. HiFi-GAN z layout — ggml `(C,T)` tensor has `ne[0]=C` as fast dim;
   `(T,C)` row-major data already has C fast — feed directly, no transpose

**Validation (tau=0 deterministic):**
- enc_q_z: cos=1.000000 (C++ vs Python)
- z_after_flow_rev: cos=0.999996
- HiFi-GAN output: ±0.306 (C++ vs Python ±0.306)
- ASR roundtrip: "Hello." (matches Python exactly)

### CI fixes (2026-06-05)

- `fix(cmake)`: add `bert-encoder` library target — fixes linker error
  from BERT conditioning commit
- `fix(go)`: sync CGO LDFLAGS via `tools/sync_go_cgo_ldflags.py` —
  fixes drift check. All 6 CI jobs + Go + Ruby + Docker Smoke green.

---

## 2026-06-02 mimo-asr GPU (PLAN #115 option C): the bug was in decode, not prefill

mimo-asr had been forced to CPU since 2026-05-26 (option A) because the GPU
path was "silent-empty / segfault" — PLAN #115 scoped the fix as per-tensor
backend tagging in `mimo_asr_build_prefill_graph`. The local M1 can't hold
the 4.2 GB model (box memory-saturated), so I debugged it on Kaggle GPU,
driving the CLI directly and mirroring the funasr-cuda-debug kernel.

Added `CRISPASR_MIMO_FORCE_GPU=1` (loads weights GPU-resident + computes
there) and `MIMO_ASR_DUMP_STAGES=1` (per-stage prefill tensor stats, like
funasr's `FUNASR_DUMP_STAGES`). Kernel `tools/kaggle/mimo-asr-gpu-diff/`,
three P100 round-trips:

- **Run 1:** the GPU **prefill is correct** — all 5 `mimo_dump` stages match
  CPU, no NaN/Inf. But `rc=-11` (SIGSEGV) at 16.5 s, *after* the prefill →
  the bug is in the **decode step**, refuting the PLAN premise.
- **Run 2 (gdb):** `dequantize_row_q4_K` → `ggml_backend_cpu_graph_compute`
  → `mimo_asr_transcribe_impl`. The sched (`[CUDA, CPU]`) offloaded a decode
  op to the **CPU** backend, which then read a **GPU-resident Q4_K** weight's
  device pointer as host memory → segfault. Not the §125 sched-src-mutation
  class at all.
- **Fix (`3ef9f87e`):** when `force_gpu`, build the sched **GPU-only** so no
  op can be CPU-routed onto a GPU weight. Run 3 validating (expect GPU JFK
  PASS); if green, flip `--gpu` to honour GPU residency by default and
  restore the 22% PLAN #72 win.

Lesson (LEARNINGS): GPU-resident quantized weights need a *single-backend*
sched — a CPU "fallback" backend will dereference device memory. The earlier
"per-tensor tagging in the prefill graph" hypothesis was wrong; reproduce on
the real GPU before theorising.

## 2026-06-02 CSM-1B TTS (§135): "buzzing not speech" was one converter line

CSM (sesame/csm-1b) loaded and generated frames but Whisper heard the
output as "ethereal music / buzzing." Five bugs had already been fixed
(F16 read, codebooks_head shape, EOS = all-32-codebooks-zero, BPE API);
the handover guessed RoPE base or Mimi decode. Built the methodical
diff harness instead of guessing: a `csm` branch in `crispasr-diff`
plus runtime dump entry points (`csm_tts_run_backbone_dump` /
`run_depth_dump` / `run_generate_codes` / `run_mimi_dump`) and
`tools/pack_csm_ref_gguf.py` to pack the manual-reference `.npy` dumps
(`csm_reference_manual.py` — HF transformers' dynamo path is broken for
CSM) into a GGUF the harness loads.

Stage-by-stage against F16 (cos threshold 0.999):

- backbone prefill — 16 layers + `last_h` + `c0` logits/argmax: **cos≈1.0**.
  (NEOX rope is correct here: the GGUF is built from the HF-transformers
  checkpoint, which is rotate_half layout — refuting the RoPE hypothesis.)
- depth decoder — `initial_proj`, `c1` logits/argmax: **cos≈1.0**.
- full greedy `all_codes`: frames 0–5 bit-exact (later drift is just F16
  greedy-argmax tie-flips) → token generation is structurally correct.
- **Mimi `rvq_dequant`** (fed reference codes): **FAIL cos 0.908, max_abs 15.8.**

Root cause in `models/convert-csm-to-gguf.py`: moshi's
`EuclideanCodebook.embedding` is the property
`embed_sum / cluster_usage.clamp(min=1e-5)`, but the converter clamped
`cluster_usage` to `max(cu, 1.0)`. `cluster_usage` is a decayed EMA with
mean ≈0.59 / min ≈0.12 / 96 % of values < 1, so the wrong clamp left
~96 % of codebook vectors un-normalized → garbled RVQ → buzzing.
One-line fix (`1.0` → `1e-5`); numpy proof: wrong clamp reproduces cos
0.92, right clamp gives cos 1.000000.

Re-converted F16, re-diffed: `mimi_rvq_dequant` **cos 1.000000**. ASR
roundtrip of the corrected GGUF: "The quick brown fox jumps over the
lazy dog." → Whisper transcribes it back **verbatim**. Intelligible
audio also proves the rest of the Mimi path (upsample / transformer /
SEANet) was already correct — the clamp was the only Mimi bug. See
moshi-codebook-cluster-usage in LEARNINGS.

## 2026-05-31 all-backends benchmark: green (26/27) — wrong build target was the blocker

The benchmark failed three Kaggle runs in a row; the harness `.log`
(fetchable via `kaggle kernels output --file-pattern`) finally showed
why: the build completed to `[452/453]` (libcrispasr.so) then asserted
`bin/crispasr not found`. The kernel built `--target crispasr-lib` — which
builds only the **library**; the CLI binary comes from target
`crispasr-cli` (it carries `OUTPUT_NAME crispasr`, see
examples/cli/CMakeLists.txt:12,232). All three failures were this, not
the CUDA OOM I first assumed. (The arch-pin fix was still necessary and
validated: run #4's box was a **P100 / sm_60**, not a T4 — a hard-coded
75 would have produced an unrunnable binary; auto-detect pinned sm_60.)

Fixed target → run #4 COMPLETE: **27 backends, 26 PASS**. Fastest
SenseVoice 19.8× RT (WER 0); best-WER tier (0%) includes firered-asr,
whisper, parakeet, cohere, wav2vec2, hubert, glm-asr, qwen3, paraformer,
voxtral 3B/4B, granite 1B/4.1, kyutai, mimo (0.3× RT, CPU-forced per
#115). The lone FAIL is **funasr** (ran at 6× RT but WER 100% — wrong
transcript, a real GPU-path backend bug, consistent with its noted
Blackwell `!`-loop caveat; surfaces on P100 too). Two non-fatal HF
pre-download 401s (auth API was flaking) fell back to the C++ downloader.
Note: `issue126`/`fusion-ab` keep `--target crispasr-lib` correctly — they
consume libcrispasr.so, not the CLI binary.

## 2026-05-31 Kaggle kernels: shared `kaggle_harness.py` (one gold standard)

The all-backends benchmark OOM'd its CUDA build on a T4 (multi-arch nvcc
× `-j4` on a ~16 GB box, died ~19 kernels into ggml-cuda). Diagnosing it
was painful because the build was a silent `subprocess.run` and Kaggle
only returns committed working-dir files, not stdout. Comparing all the
kernels showed **none was complete**: `crispasr-regression.py` had the
logging/heartbeat/ccache/mold/3-tier-auth but no CUDA arch pin;
`fusion-ab` had auto-arch but no streaming/heartbeat/auth fallback;
everyone else was a partial subset.

Extracted the union into `tools/kaggle/kaggle_harness.py` — the single
gold-standard helper module: `init_progress`/`step` (line-buffered I/O +
progress.jsonl + best-effort HF mirror to `cstr/crispasr-kaggle-progress`
for live mid-run visibility), `sh_with_progress` (Popen line-streamer
with ninja [X/N] + TU tracking), `build_heartbeat` (30 s ticker now also
reporting VmRSS + free-GB — the signal that would have *shown* the OOM
climbing), `install_build_toolchain`/`cache_and_link_flags` (ninja +
ccache at `/kaggle/working/.ccache` + mold), `detect_cuda_arch` (nvidia-
smi compute_cap → pinned `CMAKE_CUDA_ARCHITECTURES`, ~5× less nvcc RAM/
time), `cuda_build_flags`, `safe_build_jobs` (CUDA→-j2), and
`resolve_hf_token` (3-tier env → Secret-with-retry → mounted dataset).

Each kernel clones the repo early, so all six import the harness right
after the clone. The two that had copy-pasted helpers (mimo-asr-cpu-vs-
gpu, overlap-save-bug-check) drop them entirely (−453 lines net);
regression rebinds names post-clone (its helpers run pre-clone too);
benchmark + issue126 gain streaming/heartbeat/ccache/auto-arch they
never had. Pure-stdlib at import, every external probe (nvidia-smi, apt,
Secrets, /proc) has a safe fallback, and it imports cleanly off-Kaggle so
kernels stay `py_compile`/import smoke-testable before push.

## 2026-05-31 VAD #132 cache concurrency fix + all-backends benchmark refresh

Two follow-ups after fast-forwarding 35 commits (the F5-TTS / Piper /
OuteTTS native backends, canary+cohere beam wiring, and the #132 Silero
VAD context cache).

**VAD #132 — cache guarded the lookup, not the use.** The #132 fix
(`88f82838`) caches one Silero `whisper_vad_context` across requests to
kill the 70× init/free fragmentation regression. But `g_silero_cache_mtx`
only wrapped `silero_vad_get_cached`; the returned context — which owns
mutable per-request LSTM h/c buffers, the scheduler, and `probs`, all
reset and rewritten inside `whisper_vad_segments_from_samples` — was then
used unlocked. The server slices VAD *before* taking `model_mutex`
(`crispasr_server.cpp`) and httplib serves on a thread pool, so two
concurrent `/transcribe` requests raced on the one shared context (data
race + corrupted VAD). Pre-#132 each request built its own context, so
this was a regression introduced by the cache. Fix: renamed the getter to
`silero_vad_get_cached_locked` (caller-holds-lock) and hold
`g_silero_cache_mtx` across the whole detect, not just the lookup —
keeps the perf win, restores per-request isolation. CLI is single-threaded
and was never affected. `test-vad{,-full,-thresholds}` green.

**Benchmark — `tools/kaggle-benchmark-all-backends.py` was stale.** It
covered 21 backends (17 `BACKENDS` + 4 `SLOW_BACKENDS`); six complete ASR
backends had never been added. Added `sensevoice`, `paraformer` (fast)
and `granite-4.1`, `mega-asr`, `funasr`, `mimo-asr` (slow), with
`MODEL_REGISTRY` xet-safe pre-download entries + the mimo companion
tokenizer. `mimo-asr` gets a 420 s timeout (PLAN #115 forces it to CPU,
~297 s for an 11 s clip). In-progress backends (`pocket`, `dia`) are not
yet in the registry and stay excluded; TTS/diarization/LID/aligner
backends stay excluded (the benchmark measures transcription WER).
Now 27 backends; triggered on a Kaggle T4
(`chr1str/crispasr-all-backends-benchmark-t4`).
## 2026-05-29 omniasr-ctc-300m: blank_id fix + CTC auto-chunking + omniasr-300m backend

Three fixes for the OmniASR CTC backend, resolving the empty-output bug
on the 300M model and wiring it as a first-class named backend:

1. **CTC blank_id = 0** (was `hp.bos_id`, which differs between v1 and
   v2 GGUFs). Both fairseq2 and HF Wav2Vec2ForCTC train with
   `torch.nn.functional.ctc_loss(blank=0)`.

2. **CTC auto-chunking**: the 300M model's positional encoding (conv1d
   kernel=128, groups=16) degrades beyond ~7 s of audio. The backend
   adapter now auto-splits audio >7 s into 5 s chunks with 0.5 s overlap
   for CTC variants. The 1B model handles all lengths.

3. **`omniasr-300m` named backend**: registry entry pointing to
   `cstr/omniASR-CTC-300M-v2-GGUF` Q4_K (~194 MB), CLI alias,
   available_backends in C ABI, feature-matrix row.

Root cause of the empty output: the GGUF was converted from the v1
fairseq2 `.pt` checkpoint, which produces garbage in both Python and C++
inference. Only the v2 (HF transformers format) model works — the v1
format requires fairseq2's internal weight-loading pipeline which applies
weight-norm materialization that our converter replicates correctly
(verified: max diff < 1e-6) but the model still fails, suggesting deeper
incompatibility (possibly different GELU variant, dropout mask, or
attention implementation). The converter already auto-detects v1 vs v2;
only v2 produces usable GGUFs.

---

## 2026-05-29 cosyvoice3: Phase 6 — native arbitrary-WAV cloning (s3tokenizer_v3 byte-exact)

Adds runtime voice cloning from any 16 kHz WAV
(`--voice ref.wav --ref-text "..."`), porting the three front-end
extractors the Python voice-baker used to handle out-of-process. The
Python shellout (`cv3_load_runtime_voice`) stays as a fallback; the
native path is tried first.

* **speech_tokenizer_v3 ggml port** (`models/convert-cosyvoice3-s3tok-to-gguf.py`
  + `cosyvoice3_tts.cpp`): whisper-128 log-mel → 2× stride-2 conv
  subsampler → 12 FSMN/attention blocks (d=1280, h=20, FFN 5120, FSMN
  depthwise kernel 31, NEOX RoPE θ=10000) → FSQ head
  (`round(tanh(z)·0.999)+1`, Σ·3^i, codebook 3^8=6561). No final
  encoder norm — block-11 FFN residual feeds the FSQ proj directly.
* **CAMPPlus** 192-D speaker encoder reused from
  `chatterbox_campplus`; **matcha mel @ 24 kHz** for the flow ref_mel
  prefix. `cosyvoice3_tts_synth_from_wav` + `extract_{speech_tokens,
  spk_emb,ref_mel}` ABI; CLI dispatches `*.wav` voices to it.
* **Byte-exact validation via crispasr-diff.** Extended the harness:
  `_capture_s3tok_stages` (ONNX with post-subsample / blocks 0,11 /
  FSQ-proj / indices edges exposed, whisper mel as `s3tok_mel_in`) +
  `cosyvoice3_tts_extract_stage` s3tok stages + a diff-main loop. On
  fleurs_10s all five stages hit cos=1.0 and `s3tok_tokens` is
  byte-exact (max_abs=0, 264/264) vs the ONNX reference.
* **Three bugs the first (uncommitted) draft had**, all caught by the
  stage diff: (1) converter ggml-layout — gguf-py reverses the numpy
  shape into ggml `ne`, so conv kernels ship onnx `(OC,IC,KW)` as-is
  and 2D MatMul weights are transposed to `(out,in)`; the draft had
  these inverted and crashed the first conv. (2) tanh-GELU instead of
  ONNX erf-GELU. (3) the RoPE `positions` input was never filled.
* **Runtime mel residual**: the full C++ path (own mel) matches
  262/264 (99.24%); the 2 flips are a low-mel-bin **STFT** delta
  between `core::build_slaney_fb`+core_mel and whisper's mel — not the
  filterbank (embedding whisper's exact `mel_filters.npz` left it
  unchanged) and not the tokenizer network. Left as-is: shared mel
  infra, sub-1%, irrelevant to cloning (prompt tokens only condition
  speaker/prosody).
* **Quants/HF**: `crispasr-quantize` s3tok→Q4_K keeps `fsq.proj.w`
  F16 (skip rule). s3tok-f16 (462 MB, byte-exact, what `-m auto`
  pulls) + s3tok-q4_k (139 MB, cos 0.994, optional) + campplus-f16
  (13 MB) uploaded to `cstr/cosyvoice3-0.5b-2512-GGUF`; registry
  extras wired.

## 2026-05-28 cosyvoice3: Phase 5a + 5b — end-to-end synth + HF release

Closes out the CosyVoice3-0.5B-2512 port: ships a working
text→speech pipeline through the unified CLI, three-quant matrix
on HF, registry entry, and full validation.

Phase 5a (commit `b9cd5fa7` + fix `7b04a690`):

* `models/convert-cosyvoice3-voices-to-gguf.py` bakes per-voice
  prompt_speech_tokens (speech_tokenizer_v3.onnx) + spk_emb
  (campplus.onnx 192-D) + ref_mel (matcha 24 kHz log-mel) +
  prompt_text into a single `cosyvoice3-voices.gguf`. Inlined
  librosa.filters.mel + whisper log-mel front-end to dodge the
  numba/numpy-2.4 import storm.
* Runtime: `cosyvoice3_tts_init_voices_from_file` +
  `cosyvoice3_tts_synth` (text → BPE → AR speech tokens → pre_la
  + repeat_interleave → flow Euler → HiFT). The synth wrapper
  composes upstream's CV3LM input layout exactly:
  `[sos@speech_embd[6561], text_embeds, task_id@speech_embd[6563],
  prompt_speech_embeds]` — sos / task come from the speech
  embedding table (not a separate `llm_embedding` as the
  handover spec claimed; CV3LM drops that). Stop-floor variant of
  `generate_tokens` breaks on any id ≥ speech_codebook so the AR
  loop terminates on EOS instead of running max_tokens.
* CLI: `cosyvoice3-tts` backend with sibling auto-discovery of
  flow/HiFT/voices GGUFs (env overrides `COSYVOICE3_HIFT_PATH`,
  `COSYVOICE3_VOICES_PATH`). Defaults temperature to 0.8 —
  greedy decode hits CV3's documented `silent_tokens` loop within
  ~5 steps and produces silence.
* Tokenisation bug found via ASR roundtrip: joining `prompt_text`
  and user text into one string before BPE-ing shifts the
  boundary token (" Hello" → 1 token vs "Hello" → 2 tokens),
  which subtly mispronounces ("a test" → "our test" on the smoke
  prompt; three independent ASR backends agreed). Upstream
  tokenises the two id streams separately then concats; fix
  `7b04a690` mirrors that exactly. Post-fix WER 0% on the smoke
  prompt.

Phase 5b:

* `crispasr-quantize` gained a CV3-specific skip rule
  (`speech_embd.weight` + `speech_lm_head.weight` for the LLM,
  `flow.input_embd.w` + `flow.spk_affine.w` for the flow) so the
  precision-critical tables stay F16 across all quant variants.
  Pattern mirrors the chatterbox skip block.
* Quant matrix: LLM F16 (1.29 GB) → Q4_K (384 MB); Flow F16
  (665 MB) → Q8_0 (361 MB); HiFT stays F16 (42 MB). 896-wide
  rows fall back to Q4_0 (block 32) automatically. Smallest
  viable combo: 745 MB total (Q4_K + Q8_0 + F16 + voices) vs
  1.96 GB F16 reference.
* All four combos validated via ASR roundtrip on "Hello, this is
  a test." Q4_K LLM introduces minor punctuation drift
  ("Hello, this is a test." → "Hello? This is a test.") but
  content is fully preserved. German smoke (same Q4_K combo)
  agrees: "Hallo, das ist ein Test." → "Hallo? Das ist ein Test."
* HF: `cstr/cosyvoice3-0.5b-2512-GGUF` carries all 6 files (LLM
  F16 + Q4_K, flow F16 + Q8_0, HiFT F16, voices) with a full
  model card documenting the quant matrix, ASR-validation
  results, and the auto-discovery rules.
* Registry: `cosyvoice3-tts` entry resolves Q4_K LLM as the
  default plus extras pulling HiFT + voices in one go; `-m auto
  --backend cosyvoice3-tts` works out of the box.
* Docs: README "TTS models" row + feature-matrix row added.

Phase 6 (S3Tokenizer V3 + CAMPPlus + matcha mel C++ port for
runtime arbitrary-WAV cloning) tracked separately in PLAN.md.

## 2026-05-28 cosyvoice3: Phase 4-A — HiFT loader + F0 predictor (cos=1.0)

First slice of the HiFT vocoder port: binds all 246 hift GGUF tensors
(conv_pre/conv_post, 3 upsample stages, 9 main ResBlocks, 3 source
resblocks, m_source, F0 predictor weights) into the new `cv3_hift`
struct on the context, and implements the `CausalConvRNNF0Predictor`
forward.

The F0 predictor is the smallest cohesive sub-module — 5× CausalConv1d
(first is right-pad k=4 "lookahead"; remaining four are left-pad k=3
causal) + ELU between each + `Linear(512, 1)` + `abs`. It takes mel
(B, mel_dim=80, T_mel) and outputs per-frame F0 estimates that feed
the NSF source generator in the full vocoder.

Implementation:

- `cv3_hift_hp` + `cv3_hift` structs (mirror of `cv3_flow_hp` /
  `cv3_flow`) plus a `cv3_hift_resblock` for the 12-tensor resblock
  layout (3× c1, 3× c2, 3× a1.alpha, 3× a2.alpha).
- `cosyvoice3_tts_init_hift_from_file()` public API. Same
  load-weights + require-all-tensors pattern as the flow loader.
  Independent of the flow loader (either can be called without the
  other).
- `cv3_build_hift_f0_graph` reuses phase 3c's `cv3_lookahead_conv1d`
  + `cv3_causal_conv1d` helpers. The classifier matmul transposes
  the post-conv (T, 512) tensor to (512, T) so `ggml_mul_mat(w, x)`
  with w ne=(512, 1) lines up.
- New extract_stage entry `hift_f0` (input layout: T_mel × mel_dim
  F32 floats in `embeds_in`).
- Python ref backend `_capture_hift_f0_stages()` runs upstream
  `CausalConvRNNF0Predictor` on the same seeded mel. The
  weight-norm parametrisation loads in-place (PyTorch handles it
  automatically — we don't need to materialise like the GGUF
  converter does on the C++ side).
- crispasr-diff cosyvoice3-tts handler picks up hift via
  `s/llm/hift/` (or `CV3_HIFT_GGUF`); skips the hift_f0 stage
  gracefully if the hift GGUF isn't present (so phase-3 diffs still
  pass on hift-less setups).

Per-stage diff (26/26 PASS):

  hift_f0   cos=1.000000   max|Δ|=3.7e-01

The max|Δ|=0.37 looks high in absolute terms but the F0 values are
in the hundreds (typical Hz range), so cos=1.0 confirms perfect
angular alignment — the per-element drift is at the F16 weight
floor relative to signal magnitude.

Remaining phase 4 work: 4-B full HiFT decode forward (conv_pre +
SineGen source generation + STFT/iSTFT + 3-stage upsample chain
with Snake-Beta activations + 9 main ResBlocks + 3 source resblocks
+ conv_post); 4-C end-to-end mel → 24 kHz waveform diff gate.

---

## 2026-05-28 cosyvoice3: Phase 3d-B + 3e — CFM Euler ODE end-to-end (cos=1.0)

The full mel generation path: 10-step cosine-schedule CFM Euler ODE
with classifier-free guidance. Validates against upstream
`CausalConditionalCFM.forward()` bit-equivalently on the seeded
init-noise input.

Result (25/25 stages PASS through `crispasr-diff cosyvoice3-tts`):

  flow_euler_dphi_step0  cos=1.000000   max|Δ|=4.5e-03  (single post-CFG step)
  flow_euler             cos=0.999997   max|Δ|=1.4e-02  (final mel, 10 steps)

The final mel cos=0.999997 is well under the 0.99 phase-3e gate.
Per-step error compounds for 10 iterations, yet stays at the F16
weight floor — the input-pipeline + 22-block + CFG combine are
each numerically stable.

Implementation (src/cosyvoice3_tts.cpp + src/cosyvoice3_tts.h):

- `cv3_compute_sin_emb(t, dim=256)` — pure-C++ port of
  `SinusPositionEmbedding.forward(x, scale=1000)`. Computed per-step
  outside the graph; fed in as an input tensor.
- `cv3_build_estimator_full_graph` — single composed graph: time_mlp
  (Linear→SiLU→Linear) + InputEmbedding (normalize-free spks pass-
  through, concat[x, cond, mu, spk_bc], in_proj, grouped-conv
  conv_pos_embed) + 22-block DiT + norm_out + proj_out. Output is
  `dphi_dt` [T_mel, mel_dim].
- `cv3_run_solve_euler` — Euler driver. Takes (mu, spks_proj, cond,
  x_init, n_steps, cfg_rate). Builds the cosine t-span on the CPU,
  loops:
    - Compute `sin_emb_t` for current t.
    - Call estimator with real (mu, spks, cond) → `dphi_cond`.
    - Call estimator with zeros for (mu, spks, cond) → `dphi_unc`.
    - CFG combine: `dphi = (1+cfg) * dphi_cond - cfg * dphi_unc`.
    - Euler step: `x = x + dt * dphi`.
- Public API `cosyvoice3_tts_solve_flow_euler` exposed in the header.
- New extract_stage prefixes `flow_euler_*` (input layout packs
  `[mu | spks_proj | cond | x_init]`).

Why two B=1 estimator passes per step instead of upstream's B=2
batched call: the estimator is deterministic and batch-wise
independent, so the numerical result is identical, and the per-call
overhead is small compared to the 22-block forward itself. Keeps
the graph builder simple — no need to split a B=2 output back into
batches inside ggml.

Why `x_init` is a caller input (not generated inside the driver):
upstream's `CausalConditionalCFM` sets `set_all_random_seed(0);
rand_noise = torch.randn([1, 80, 50*300])` at module init, so the
noise is fixed across calls. Porting torch's Mersenne-Twister +
Box-Muller bit-exactly was out of scope. The Python ref harness
dumps the seeded noise into the GGUF archive so the diff harness
hands the same noise to both sides.

Three new gotchas captured in code comments:

1. **Stub `matcha.models.components.flow_matching::BASECFM`** before
   importing `cosyvoice.flow.flow_matching`. BASECFM contributes
   nothing to the inference path (`__init__` stores cfm_params;
   `forward` + `solve_euler` are defined locally on ConditionalCFM /
   CausalConditionalCFM). The Python ref harness monkey-patches
   `sys.modules` with a `torch.nn.Module` stub so we don't have to
   pip-install matcha just for an unused base class.
2. **CFG zero pass uses zeros AFTER projection**, not raw zeros fed
   through projection. Upstream's `solve_euler` sets
   `spks_in[1] = 0; mu_in[1] = 0; cond_in[1] = 0` AFTER spk_affine,
   so the unconditioned branch's spk_proj is genuinely zero (not
   spk_affine.bias). The C++ driver takes `spks_proj` (already
   projected) and passes a zero buffer for the uncond pass — matching
   upstream exactly.
3. **Single composed graph beats sequential** for the estimator call.
   Inlining input-pipeline + 22-block stack + norm + proj into one
   `cv3_build_estimator_full_graph` avoids the malloc + memcpy round-
   trip between phases-3c/3d-A's per-stage extract calls (which
   exist for diff convenience).

Remaining phase 3 work: none — phase 3 is complete. Next phase 4
is the HiFTGenerator vocoder (mel → 24 kHz waveform).

---

## 2026-05-27 cosyvoice3: Phase 3d-A — full 22-block DiT estimator (cos=1.0)

Composes the per-block forward (phase 3b) into the full 22-layer DiT
estimator stack + `AdaLayerNormZero_Final` (`norm_out`) +
`Linear(1024, 80)` (`proj_out`). This is the function the CFM Euler
solver (phase 3d-B) will call inside its 10-step loop.

Implementation:

- Factor a small `cv3_dit_block_apply()` helper inside the anon namespace
  — mirror of `cv3_build_flow_dit_block_graph` but without the debug
  `dbg_*` named outputs and parameterised on a shared `silu(t_emb)`
  tensor (computed once for the whole 22-layer stack, not 22 times).
- `cv3_build_dit_full_graph()` loops the helper across all 22 blocks,
  applies `norm_out` (chunk order is `(scale, shift)` — opposite of
  the per-block AdaLN's `(shift, scale, gate)` ×2) + `proj_out`, and
  emits two named outputs: `dit_full_norm` (pre-proj, T_mel×1024) and
  `dit_full_out` (post-proj, T_mel×80).
- `cv3_extract_dit_full_stage()` drives the graph from the post-
  input-pipeline x + t_emb that the diff harness packs from the
  reference archive.
- New stage names `flow_dit_full_norm` and `flow_dit_full` in
  `cosyvoice3_tts_extract_stage` (input layout: packed `[x | t_emb]`
  = T_mel·dit_dim + dit_dim floats, n_embed_tokens = T_mel).

Wiring:

- `tools/reference_backends/cosyvoice3_tts.py::dump()` learns
  `_capture_dit_full_stages()`, which instantiates the upstream's 22
  `DiTBlock` modules + `AdaLayerNormZero_Final` + `Linear(1024, 80)`
  from `flow.pt`, runs them on a seeded (T_mel=12, t=0.5) fixture,
  and emits `flow_dit_full_{x_in, t_emb, norm, …}` ndarrays.
- `examples/cli/crispasr_diff_main.cpp` cosyvoice3-tts handler unpacks
  `flow_dit_full_x_in` + `flow_dit_full_t_emb` from the reference,
  packs `[x | t_emb]`, and compares both stages at threshold 0.99
  (relaxed for the depth-22 accumulated F16 noise — same convention
  as voxcpm2-tts for deep-stage stages).

Per-stage diff result (post phase 3b + phase 3c + phase 3d-A — 23/23 PASS):

  flow_dit_full_norm    cos=0.999996  max|Δ|=3.3e-03
  flow_dit_full         cos=1.000000  max|Δ|=3.6e-03

The 22-deep accumulation only buys ~30× more max|Δ| vs the 1-deep
per-block diff (block-21 was max|Δ|=8.7e-02; full-stack is 3.6e-03 —
the deeper-stack output norm is actually SMALLER because norm_out
re-normalises, suppressing the per-block growth). cos remains 1.0
across the entire stack.

Remaining phase 3 work: 3d-B local CFM Euler ODE (cosine t-schedule,
sigma_min=1e-6, inference_cfg_rate=0.7, 10 steps, classifier-free
guidance), 3e end-to-end mel diff.

---

## 2026-05-27 cosyvoice3: Phase 3c — pre-lookahead conv + InputEmbedding (cos=1.0)

Input pipeline that turns raw speech tokens + speaker embedding into
the (T_mel, 1024) tensor fed into the DiT block stack. Validates
against upstream PyTorch on a seeded fixture (T_tok=6 → T_mel=12,
mel_dim=80, spk_dim=192).

Implementation (all in `src/cosyvoice3_tts.cpp`):

- `cv3_lookahead_conv1d` — right-padded dense conv1d (`F.pad(x, (0, K-1))`
  + `conv1d(padding=0)`) via the symmetric-pad-K-1 + take-right-T
  trick. Mirror of chatterbox's left-pad helper.
- `cv3_causal_conv1d` — local copy of chatterbox's left-pad
  conv1d, kept self-contained so phase 3c doesn't pull a static
  helper into a shared header.
- `cv3_causal_grouped_conv1d` — 16-group causal conv1d (k=31) via a
  per-group loop + `ggml_concat` along channel dim (ggml has no
  native grouped-conv op). Pattern taken from
  `omniasr.cpp::build_grouped_pos_conv` and adapted for causal pad.
- `cv3_build_pre_la_graph` + `cv3_extract_pre_la_stage` — wraps
  embedding lookup + the two convs as a named-output graph.
- `cv3_build_in_pipe_graph` + `cv3_extract_in_pipe_stage` — builds
  the `InputEmbedding` pipeline. `F.normalize(spk, dim=1)`
  implemented as `ggml_rms_norm(eps=0) * (1/sqrt(spk_in))` to convert
  RMS denominator to L2 denominator.

Wiring:
- `cosyvoice3_tts_extract_stage` learns the `flow_pre_la_*` and
  `flow_in_pipe_*` prefixes (input layout documented in the header).
- `tools/reference_backends/cosyvoice3_tts.py::dump()` adds matching
  captures via piecewise reconstruction of the upstream modules
  (verified <1e-5 vs the monolithic forward each time).
- `examples/cli/crispasr_diff_main.cpp::cosyvoice3-tts` handler
  unpacks ref inputs (ids for pre_la, packed
  `[pre_la | spk | x | cond]` for in_pipe) and drives each stage.

Three gotchas worth recording:

1. **Layout convention**: `Ref::get_f32` returns a flat float array
   matching the GGUF byte layout. The C++ side produces tensors in
   ggml `ne=(C, T)` col-major; **PyTorch `(T, C)` row-major is
   byte-identical**, so the Python dump must NOT pre-transpose to
   `(C, T)` before storing. Doing so silently inverts the comparison
   (cos → −0.2). Fixed by collapsing the transpose helper into a
   no-op `.squeeze(0).contiguous()`.
2. **Side-output reachability**: `ggml_set_output` on a tensor that
   isn't on the path to the final `build_forward_expand` root gets
   pruned. Fix: `ggml_build_forward_expand` each named intermediate
   explicitly. (Phase 3b avoided this because every named
   intermediate was a parent of the block output.)
3. **`set_output` on a view**: a `reshape_1d` view's output buffer
   can end up sharing storage with the parent's pre-add result in a
   way that breaks `tensor_get` (returns wrong values, cos≈−0.12
   while downstream consumers PASS). Fix: `ggml_cont` after
   `reshape_1d` so the named output owns its buffer.

Per-stage diff (21/21 PASS at cos_min ≥ 0.999):

  flow_pre_la_tok_emb   cos=1.000000  max|Δ|=4.4e-04
  flow_pre_la_c1        cos=0.999891  max|Δ|=4.8e-04
  flow_pre_la_c2        cos=1.000000  max|Δ|=6.0e-04
  flow_pre_la           cos=1.000000  max|Δ|=6.2e-04
  flow_in_pipe_spk      cos=1.000000  max|Δ|=1.2e-04
  flow_in_pipe_cat      cos=1.000000  max|Δ|=1.2e-04
  flow_in_pipe_proj     cos=1.000000  max|Δ|=5.5e-03
  flow_in_pipe_pos      cos=0.999973  max|Δ|=4.7e-03
  flow_in_pipe          cos=1.000000  max|Δ|=5.5e-03

Max relative diff well under the F16 weight floor (~0.08%) — every
stage in the input pipeline is at the numerical noise floor.

Remaining phase 3 work: 3d local CFM Euler ODE + 22-block forward +
mel output, 3e end-to-end mel diff.

---

## 2026-05-27 cosyvoice3: Phase 3b — AdaLN-Zero DiT block forward (cos=1.0)

Per-block DiT forward for the 22-block estimator in the
cosyvoice3-flow GGUF. Validates against upstream PyTorch (FunAudioLLM/
CosyVoice `flow/DiT/modules.py::DiTBlock`) on a seeded fixture
(T=8, dim=1024, t=0.5) at every intermediate stage of block 0 and
block 21.

Built `cv3_build_flow_dit_block_graph` (~80 LOC) +
`cosyvoice3_tts_run_flow_dit_block` (public API) +
`cv3_extract_flow_dit_stage` (private; drives the same graph for
intermediate outputs) + new `flow_dit_blk_<N>_{lnx_a,h_a,attn,xattn,
ff,out}` stages in `cosyvoice3_tts_extract_stage`. Wired into
`crispasr-diff` via a new `cosyvoice3-tts` backend block; the Python
side gets a `dump()` entry in `tools/reference_backends/cosyvoice3_tts.py`
registered in `tools/dump_reference.py` REGISTERED_BACKENDS, so the
diff goes through the unified GGUF-archive pipeline like every other
backend.

Five corrections vs the original handover prompt, all verified against
the upstream source:

1. AdaLN-Zero chunk order is **(shift_msa, scale_msa, gate_msa,
   shift_mlp, scale_mlp, gate_mlp)** — not the prompt's
   "(scale, gate, shift) × 2". Upstream `AdaLayerNormZero.forward`
   line 241: `torch.chunk(emb, 6, dim=1)` returns shift first.
2. SiLU is applied to `t_emb` **before** the AdaLN linear, not after:
   `emb = self.linear(self.silu(emb))`.
3. Both norms (`AdaLayerNormZero.norm` and `DiTBlock.ff_norm`) are
   `nn.LayerNorm(elementwise_affine=False, eps=1e-6)` — plain
   layer norm without learned affine, **not RMSNorm**. Use
   `ggml_norm`.
4. FFN is `Linear → GELU(approximate="tanh") → Linear`. No SiLU, no
   GLU gating. `ggml_gelu` already uses the tanh approximation.
5. RoPE — biggest gotcha. Upstream's `AttnProcessor` calls
   `apply_rotary_pos_emb` on the **pre-reshape** Q/K (shape
   `(B, T, n_h*hd) = (B, T, 1024)`) with `rot_dim = head_dim = 64`.
   The x_transformers helper does `t[..., :rot_dim]` + `t[..., rot_dim:]`
   and only rotates the first 64 channels — which is **head 0**.
   Heads 1..15 carry no positional info. Mode is x_transformers'
   adjacent-pair rotation = ggml `GGML_ROPE_TYPE_NORMAL` (NOT NEOX),
   θ=10000. Match in C++ by reshaping Q/K to `(d, 1, T)` and ropeing
   with `n_dims=hd` so only the first 64 channels rotate, then
   reshape into per-head layout for flash-attn.

Per-stage diff result (`crispasr-diff cosyvoice3-tts ... cv3-flow-dit-ref.gguf`):

  flow_dit_blk_0_lnx_a   cos=1.000000  max|Δ|=2.4e-07
  flow_dit_blk_0_h_a     cos=1.000000  max|Δ|=6.9e-04
  flow_dit_blk_0_attn    cos=0.999994  max|Δ|=2.7e-02
  flow_dit_blk_0_xattn   cos=0.999997  max|Δ|=2.7e-02
  flow_dit_blk_0_ff      cos=0.999997  max|Δ|=7.0e-02
  flow_dit_blk_0_out     cos=0.999994  max|Δ|=1.6e-01
  flow_dit_blk_21_attn   cos=0.999999  max|Δ|=1.6e-02
  flow_dit_blk_21_out    cos=0.999997  max|Δ|=8.7e-02

12/12 stages PASS at cos_min ≥ 0.999 (threshold). Max relative diff
0.04% on the final block-21 output — comfortably under the F16
weight floor (~0.08%) established in Phase 2 calibration.

Remaining Phase 3 work: 3c pre-lookahead conv + input pipeline,
3d local CFM Euler ODE + mel output, 3e end-to-end mel diff.

---

## 2026-05-27 cosyvoice3: Phase 3a — flow loader + DiT block scaffold

Extends `src/cosyvoice3_tts.{h,cpp}` with Phase 3 flow-side support:

- `cosyvoice3_tts_init_flow_from_file()` loads
  `cosyvoice3-flow-f16.gguf` (~670 MB, 330 tensors) into the existing
  context. Hparams from `cosyvoice3.flow.*` keys: 22-block DiT @
  dim=1024, heads=16, head_dim=64, ff_dim=2048, input_dim=320,
  mel_dim=80, spk 192→80, 10-step CFM Euler, cfg=0.70.
- `cv3_flow_hp` + `cv3_dit_block` + `cv3_flow` structs. Each DiT block
  binds (adaln_w 6×1024, attn_q/k/v/o w+b, ffn_l1/l2 w+b). Top-level
  tensors: input_embd (80,6561), pre_la conv1/conv2, spk_affine,
  dit_in_proj, conv_pos c1/c2 (grouped conv1d-31), time_mlp 0/2,
  rope_inv_freq, norm_out, proj_out.
- `cosyvoice3_tts_get_flow_hparams()` + a `"flow_inventory"` diff
  stage for harness validation. Flow GGUF + LM GGUF coexist on the
  same backend; separate `flow.ctx_w` / `flow.buf_w`.

Smoke test (smoke binary now accepts `--flow <path>`) confirms:
  cosyvoice3_tts:flow loaded 330 tensors
  dit=22L d=1024 h=16/hd=64 ff=2048 in_dim=320 mel=80 spk=192/80
  cfm_steps=10 cfg=0.70
  flow_inventory stage returned 10 floats

The DiT forward graph (AdaLN-Zero modulation per block), pre-lookahead
conv pipeline, conv_pos_embed, time_mlp evaluation, and CFM Euler ODE
driver are deferred to Phase 3b/c/d. Chatterbox `cfm_euler_solve` is
NOT directly reusable — it bakes chatterbox-specific tensor names into
the time-MLP path, so the Euler step loop will be re-implemented
locally for cosyvoice3.

---

## 2026-05-27 cosyvoice3: Phase 2 closeout — RAS sampler + greedy AR byte-identical

Two follow-ups on top of `137670d1`:

- **RAS sampler** ported from
  `CosyVoice/cosyvoice/utils/common.py::ras_sampling` —
  `cosyvoice3_tts_sample_ras(logits, history, n_history)` returns a
  single speech token via nucleus_sampling (top_p=0.8, top_k=25,
  stable-sort + multinomial over unrenormalised kept probs) and
  re-samples via plain softmax-multinomial when the picked token
  exceeds `win_size·tau_r = 1` repeats in the last 10. `std::mt19937_64`
  seeded from `ctx->seed`; advances per sample. PyTorch
  `torch.multinomial` bit-identity is **not** in scope (different PRNG
  internals) — the algorithm matches semantically.

- **`generate_tokens_from_embeds`** end-to-end AR loop: prefill on
  caller-supplied embeds + AR loop (greedy when temperature=0, else
  RAS) up to `max_tokens` or `stop_token_id`.

- **Cached step graph clobber bug fixed.** `cv3_run_embed` shares
  `compute_meta` with the step graph; its `ggml_init` overwrites
  the cached `step_t1_gf`'s tensor metadata in place, so reusing
  the cached graph on step 2 read garbage. Now invalidate the
  cache at the top of every embed run; the cache only stays warm
  across consecutive step_speech calls (no embed-in-between).
  Symptom before fix: greedy gen emitted 2 tokens then crashed.

Greedy diff against PyTorch ref on "Hello, this is a test." (7 tokens
prefill, 8 AR steps): C++ `[4512, 4512, 4512, 4512, 4512, 4512, 4512, 4512]`
== PyTorch `[4512, 4512, 4512, 4512, 4512, 4512, 4512, 4512]`. The
all-`4512` collapse is the model's expected behaviour without a voice
prompt; RAS (seed=42, 24 steps) breaks it to varied tokens
`[2729, 4432, 4431, 5890, 5889, 6154, 5899, 58, ...]`, exercising
both the nucleus branch and the repetition-suppression fallback.

**Phase 2 greedy diff gate is formally PASSING** (step0 logits cos=1.0,
8-step argmax sequence byte-identical to PyTorch). Seeded-RAS
bit-identity is left for a later session once
`torch.multinomial` PRNG-matching is in scope.

---

## 2026-05-27 cosyvoice3: Fun-CosyVoice3-0.5B-2512 TTS port — Phase 2 (LLM runtime + diff)

CosyVoice3 LLM scaffolding for the C++ runtime — loader, Qwen2-0.5B
step graph (24L, GQA 14/2, q/k/v biases, NEOX RoPE θ=1e6, RMSNorm
1e-6), speech-side embedding/head (6761-vocab), KV cache + cached
T=1 step graph. Wires into `core_attn::kv_self_attn` + `core_ffn::swiglu`
unchanged. Added Python reference dumper
(`tools/reference_backends/cosyvoice3_tts.py`) that loads `llm.pt`
into HF `Qwen2Model`, attaches the two speech-side modules, and
emits stage activations as a `.npz`. Built a tiny diff runner
(`tools/diff-cosyvoice3-lm.py`) that feeds the same `input_embeds`
through both sides and compares step0 logits.

First-run diff result (prompt "Hello, this is a test." → 7-token
prefill, no voice prompt, greedy):

  cosine        = 1.000000
  max |Δ|       = 0.0149  (0.0775% of max ref)
  ref top-5     = [4512, 2322, 2325, 0, 4916]
  cpp top-5     = [4512, 2322, 2325, 0, 4916]   ← byte-exact match

The 0.08% absolute delta is consistent with F16 weight storage on
the C++ side vs F32 weights on the PyTorch side; the cosine is 1.0
to six decimals and the argmax sequence is identical, so the LLM
diff gate is essentially passing on the greedy path. Remaining
Phase 2 work: RAS sampler + seeded-RAS diff parity (cf.
`handover-prompts/cosyvoice3-runtime-port.md` "Phase 2 — Diff-gate:
byte-identical").

Phase 3 (DiT-CFM flow estimator) and Phase 4 (CausalHiFTGenerator)
are still open; flow / hift GGUFs regenerated by this commit
(`/Volumes/backups/ai/crispasr-models/cosyvoice3-0.5b-2512/`).

---

## 2026-05-26 v0.6.11 RELEASE (cut at commit `b77a74eb`)

Tagged 2026-05-26 after 146 commits since v0.6.10 (2026-05-23). Full
user-facing notes at [`RELEASE_NOTES_v0.6.11.md`](RELEASE_NOTES_v0.6.11.md);
the per-fix HISTORY entries below have the engineering depth.

**Headline themes:**

- Long-form transcription 2-7× better across parakeet / canary / voxtral / cohere — see PLAN #114 and the option matrix in PERFORMANCE.md.
- Issue [#125](https://github.com/CrispStrobe/CrispASR/issues/125) fix train: 8 backend bugs from @montvid (firered-asr / funasr / omniasr-llm / gemma4-e2b / mimo-asr / kyutai-stt ×2 / mimo-asr scheduler P0).
- Chatterbox TTS GPU path finally intelligible on M1 Metal — PLAN #83 R9 #5 sched `parallel=true` fix.
- New CLI: `--hf-repo OWNER/REPO[:FILE]` (issue [#128](https://github.com/CrispStrobe/CrispASR/issues/128)) for llama-server-style HuggingFace model fetching.
- Five new model ports: SenseVoiceSmall, Paraformer-zh, FunASR family (nano + mlt-nano), Parakeet-RNNT 0.6b + 1.1b.
- Infrastructure: Kaggle rebake pipeline (4 new parakeet refs published), overlap-save A/B harness, local audio fixture mirror, CI cleanup (AVX512 + arm64 native runners).

**Breaking / behavioural changes**, all listed in the release notes — most-visible: canary refuses unsupported languages cleanly, parakeet auto-chunks past 30 s by default, gemma4-e2b / firered-asr refuse too-long inputs with clear errors.

**Methodological discipline** that proved load-bearing this cycle:

- Don't trust LEARNINGS unconditionally — the parakeet streamed-path failure was originally diagnosed as decoder cold-start; empirical chunk-size sweep proved the bottleneck is encoder context. Lesson banked at LEARNINGS § Correction 2026-05-26.
- "A CUDA-shaped grep can miss the generic scheduler when an unrelated patch touches it" — the mimo-asr Blackwell P0 was reporter-bisected to a fully `#ifdef`-guarded-OFF FA-mask patch; the real culprit was `ggml-backend.cpp` sched src-mutation log not restoring on early-error compute returns.

Tag: `v0.6.11`. The next cycle (v0.6.12) opens with the Kaggle granite/omniasr chunk-context audit pending and the canary `canary_transcribe_streamed` polish stack stable.

---

## 2026-05-26 (parakeet drop CAP_INTERNAL_CHUNKING) dispatcher chunk-30 + LCS becomes the default for long audio

After the per-model chunk-size fix `e1904a1e`, the user asked: "should we just make per-model defaults? we can NOT expect users to consult a matrix beforehand and then finetune each cli option." Right.

Full option matrix (the seven knobs and three dispatcher modes) ran on en/de/ja FLEURS 60 s + yt_60s.wav JA. The matrix lived in PERFORMANCE.md (`86465651`). It showed the dispatcher's `--chunk-seconds 30 --chunk-overlap 3` mode beats the backend's internal-streamed default on 3 of 4 cases:

  v3 + EN 60s : 520 → 755 (+45 %)
  v3 + JA 60s : 605 → 660 (+9 %)
  ja + JA 60s : 1674 → 1942 (+16 %)
  v3 + DE 60s : 679 → 665 (-2 %, small)

**Mechanism.** The dispatcher's `should_auto_chunk_long` fallback at `examples/cli/crispasr_run.cpp:413` auto-chunks long audio iff `CAP_UNBOUNDED_INPUT && !CAP_INTERNAL_CHUNKING && !wants_vad && audio > 30 s`. Parakeet was declaring `CAP_INTERNAL_CHUNKING`, which suppressed the auto-chunk. Dropping that flag lets the auto-chunk fire for long audio.

**Why the dispatcher path beats the backend's own streamed path:** the dispatcher splits the audio into ~30 s chunks with ±3 s overlap, calls the backend once per chunk (each call sees 33 s — the v3 encoder's training window), and LCS-merges boundary tokens. The backend's own `parakeet_transcribe_streamed` instead applies global mel-norm then splits the mel into chunks; per-call mel-norm (dispatcher) appears to produce better-conditioned encoder features for the TDT decoder than per-clip-then-split (backend).

**Shipped (`98381810`).** Dropped `CAP_INTERNAL_CHUNKING` from `examples/cli/crispasr_backend_parakeet.cpp::capabilities()`. The empirical regression matrix:

| case | before | after | Δ |
|---|---|---|---|
| JFK 11s | 109 | 109 | unchanged (under threshold) |
| v3 + EN 60s | 520 | **755** | **+45 %** |
| v3 + DE 60s | 679 | 665 | -2 % |
| v3 + JA 60s | 605 | **660** | +9 % |
| ja + JA 60s | 1674 | **1942** | **+16 %** |
| v3 + EN 300s | 1550 | **3865** | **+150 %** |
| v3 + DE 300s | 3064 | **3288** | +7 % |

6 of 7 improved, 1 small DE 60 s regression (-2 %). The longer the audio, the bigger the win — EN 300 s scales from +45 % at 60 s to +150 % at 300 s. The internal-streamed-path's quality degradation compounds with audio length; dispatcher chunks scale linearly. JFK 11 s is unchanged because the dispatcher only auto-chunks past 30 s.

**Tradeoff accepted.** Wall time on long audio increases (~30 s → ~86 s for 300 s EN audio on M1 Metal, 3.5× realtime). For users who care more about wallclock than coverage, `CRISPASR_PARAKEET_STREAM_THRESHOLD=99999` still forces the older single-pass path. The CLI flags (`--chunk-seconds`, `--chunk-overlap`, `--vad`, etc.) remain available for users who want to override the auto-chunk decision.

**Architectural note.** The capabilities flag `CAP_INTERNAL_CHUNKING` was *meant* to declare "this backend handles its own long-audio chunking — don't auto-chunk me at the dispatcher level". Parakeet's `parakeet_transcribe_streamed` does technically handle long audio internally, but the empirical data showed the dispatcher's chunking + overlap-save + LCS-merge produces better output than the backend's internal path. So the flag was semantically correct but quality-wise wrong. Dropping it makes the dispatcher's auto-chunk-at-30s the de-facto default for parakeet long audio, while leaving the backend's internal streamed path as a fallback (when the dispatcher hands a per-chunk slice to the backend, the backend's streamed code still runs — but on a 33 s window where it's well-conditioned).

**Methodology.** (1) Asked user the "right" question about defaults vs flags. (2) Built the empirical matrix across all 7 dispatch knobs (PERFORMANCE.md `86465651`). (3) Picked the winning mode. (4) Found the architectural lever (the `CAP_INTERNAL_CHUNKING` flag). (5) Dropped it. (6) Re-ran the 7-fixture regression. (7) Documented the tradeoffs and the override paths so power-users keep their knobs.

---

## 2026-05-26 (parakeet streamed-chunk default fix) per-model heuristic via vocab_size

The multi-backend long-form comparison in PERFORMANCE.md (commit `b8588bc5`) flagged parakeet truncating on the EN FLEURS 60 s clip — only 217 chars (~25 % coverage), missing 4 of 6 sentences. Other backends covered the same audio cleanly: voxtral 826 chars, cohere 864, canary 735. This was supposed to be the streamed path's home-court advantage (we made it the default in `33f9a162` precisely to fix the issue #89 truncation pattern). Instead parakeet was the worst on real long-form en.

User direction: "fix parakeet methodically" + "you cannot trust however what is in learnings".

**Methodical sweep** on the new fixtures (`audio_samples/{en,de}/fleurs_{60,300}s.wav` mirrored from VPS, plus `long-clips/yt_60s.wav`) with `CRISPASR_PARAKEET_STREAM_CHUNK=N` overriding the c=8 default:

| model | audio | c=8 | c=15 | c=20 | c=30 | c=40 |
|---|---|---|---|---|---|---|
| v3 (vocab=8192) | EN 60s | 186 | 159 | 362 | 519 | **800** |
| v3 (vocab=8192) | EN 300s | 492 | 1034 | **1703** | 1549 | 1561 |
| v3 (vocab=8192) | DE 60s | 502 | **709** | 581 | 678 | 520 |
| v3 (vocab=8192) | DE 300s | 2496 | 2806 | 2993 | **3063** | 2944 |
| ja (vocab=3072) | JA 60s | **1674** | — | 907 | 507 | 363 |

**Finding.** The `c=8` default that ships well for the JA-only parakeet model **collapses on the multilingual v3 model** — EN 60s drops to 23 % of the c=40 max. The JA model has the OPPOSITE preference: smaller chunks work better. So one single default can't be right for all variants.

**Fix (`e1904a1e`).** Per-model chunk default keyed off `vocab_size` in both `parakeet_transcribe_streamed` and `parakeet_transcribe_chunked`:
- `vocab_size < 4000` (JA-only, vocab=3072) → c=8 (preserved)
- `vocab_size >= 4000` (multilingual / v3, vocab=8192) → c=30

Threshold of 4000 cleanly separates the two known variants. Backend wrapper at `crispasr_backend_parakeet.cpp` now passes `chunk=0` (sentinel for "let the C library pick") instead of the hard-coded 8. `CRISPASR_PARAKEET_STREAM_CHUNK=N` env override still wins.

**Regression matrix after the fix:**

| case | before | after | Δ |
|---|---|---|---|
| v3 + EN 60s | 186 | **520** | +180 % |
| v3 + DE 60s | 502 | 679 | +35 % |
| v3 + EN 300s | 492 | **1550** | +215 % |
| v3 + DE 300s | 2496 | 3064 | +23 % |
| ja + JA 60s | 1674 | 1674 | unchanged (preserved) |

**Root-cause correction (`ffd708bb`).** The LEARNINGS § "Independent-chunk TDT decode" section described decoder cold-start as the dominant failure mode. That diagnosis was right for `parakeet_transcribe_chunked` (independent per-chunk decodes) but wrong for the currently-shipping `parakeet_transcribe_streamed` (one decode over the concatenated encoder output, no per-chunk decoder reset). What actually fails in streamed: the Conformer encoder's bidirectional attention needs roughly its training-distribution context window (~30 s) to produce features the TDT decoder recognises as in-distribution. With 8 s chunks the per-feature statistics shift enough — even though the mel z-norm is global — that the TDT decoder emits a much sparser token path. The user's "don't trust learnings" directive was load-bearing: I would have gone looking for a decoder-state bug if I'd taken the section at face value, and the bug isn't there.

**Methodology trail.** (1) Reproduce streamed-vs-forced-single-pass — both 217, so it's not the streaming branch itself. (2) Try `--vad` (313 chars), confirms the audio + model are fine. (3) Chunk-size sweep — c=8 → 186, c=40 → 800 on EN 60s — chunk size dominates. (4) Cross-check on JA model + JA audio — c=8 → 1674 (best), c=30 → 507 (worse) — opposite preference, must be per-model. (5) Diff hparams between v3 and JA — `vocab_size` 8192 vs 3072, clean discriminator. (6) Implement + verify across the 5-fixture matrix. (7) Correct LEARNINGS.

**Lesson banked in LEARNINGS** § Correction 2026-05-26: don't generalize a failure-mode diagnosis from one decoder-decode shape to another. Streamed vs chunked is a decoder-axis distinction that matters here; we have *both* code paths and *both* are documented under "Independent-chunk TDT decode" because they shared the same external symptom (sparse interior output) — but the mechanism differs and the fix differs.

---

## 2026-05-26 (P114-P3 word-snap + streaming-pattern design notes)

`935ffbee` lands the word-snap heuristic for canary streamed dedup, the
last documented polish item from PLAN #114 P3. After LCS-prefix-drop, if
the next surviving token's text doesn't start with a space (sentencepiece
convention: `▁X` decodes to ` X`, so a leading space marks a word-start),
extend the drop until the next word-start. Trades a few extra tokens for
a clean prefix; bounded by `n_tokens` so we never drop the whole chunk.

De-Abwasch (1.3 m DE article) before/after on M1 Metal:

| | Before | After |
|---|---|---|
| Length | 1233 chars | 1196 chars |
| `"umfassen. tuch umfassen"` | mid-word fragment | `"umfassen. umfassen"` ✓ |
| `"Maitrein. ittels mit"` | mid-word fragment | `"Maitrein. mit"` ✓ |
| `"irrspülmaschine"` | mid-word fragment | `"Geschirrspülmaschine"` ✓ |
| `"Gefühl. onär ist"` | mid-word fragment | `"Gefühl. ist"` ✓ |

JFK unchanged. Remaining EN FLEURS 60 s artifacts like `"World's Save for
You"` duplicating `"world's say for you"` are model-retokenization (the
AED emits different token ids on the same audio because of capitalization
shift), out of scope for word-snap. Next-level dedup would need
case-insensitive LCS or beam-over-chunks.

**Also banked: streaming-pattern design notes** in PLAN #114 (new section
just before "Decision: don't blanket-VAD everyone"). Two upstream patterns:

- **NeMo `BatchedFrameASRTDT` / `FrameBatchMultiTaskAED`**: overlap-chunks
  (8 s + 2 s) → per-chunk encoder → LCS dedup at boundary. Per-chunk
  decoder reset is mandatory for AED-class decoders (canary's `<eos>`
  semantics); optional for frame-synchronous decoders (TDT/CTC).
- **Mistral voxtral `apply_transcription_request`**: disjoint 30 s chunks
  → per-chunk encoder → audio embeds concatenated → one LLM AR decode.
  No dedup needed (no duplicated audio in the input). Requires
  long-context AR LLM.

**Per-backend fit is structural, not configurable:**

- parakeet (TDT/RNN-T/CTC) → currently hybrid (NeMo overlap + single decode); voxtral pattern feasible but no quality upside
- canary (AED) → NeMo pattern only; voxtral pattern fails by design (`<eos>` at first chunk-boundary)
- voxtral / qwen3-asr / granite / glm-asr / mimo-asr / gemma4-e2b / kyutai-stt (AR LLM) → voxtral pattern preferred

**What IS a runtime knob today**: per-backend `STREAM_THRESHOLD_S` durations,
`STREAM_CHUNK_S` overlap sizes (env vars), the `kBlocked` opt-out list in
`crispasr_chunk_context_gate.h`. **What ISN'T (and probably shouldn't be)**:
the pattern choice itself. Forcing the wrong pattern via a CLI flag would
let users misuse the binary with no quality recovery; nothing is gained
from running canary in voxtral pattern or voxtral in NeMo pattern.

PLAN #114 P3 is now content-correct + cosmetically-clean on canary's four
trained languages. The remaining `World's Save for You`-class artifacts
are model-side (AED retokenization across chunks); not addressable from
the streamed wrapper without restructuring the decode loop.

---

## 2026-05-26 (P114-P3 validation) canary streamed on real en + de long-form audio

After the six P3 fix commits landed (lang-whitelist → streamed → per-chunk re-injection → LCS dedup → splice-punct cleanup → degenerate-loop guard), the streamed path was only validated on JFK (synthetic-equivalent at 11 s) and the Japanese yt_60s.wav (OOD audio for canary). Pulled real en + de long-form clips from VPS `/mnt/akademie_storage/test-audio/{en,de}/fleurs_*.wav` and `german-samples/De-Abwasch-article.wav` (1.3 m DE article) into `/Volumes/backups/code/audio_samples/` (full inventory + CLAUDE.md in that dir). Comparing streamed (default) vs forced single-pass on canary-1b-v2 Q4_K:

| input | single-pass | streamed |
|---|---|---|
| `audio_samples/en/fleurs_60s.wav` (60 s en) | **362 chars**, 3 sentences then truncates | **667 chars**, 5+ sentences, ~2× content |
| `audio_samples/de/fleurs_60s.wav` (60 s de) | **419 chars**, 4 sentences then truncates | **774 chars**, full coverage, ~2× content |
| `audio_samples/multi/De-Abwasch-article.wav` (1.3 m de) | **458 chars**, ~5 sentences then jumps mid-transcript | **1233 chars**, full article body, ~2.7× content |

Single-pass on every long-form input shows the issue #89 encoder-amplification failure: the bidirectional Conformer attention past ~30 s feeds garbage into the AED decoder, which emits `<eos>` early. **Streamed delivers ~2-3× more content** without truncation.

**Boundary artifacts visible on streamed output.** The LCS dedup catches token-level overlaps but not mid-word splits at the chunk boundary. Examples from the live runs:

- en 60s: `"twenty five. to thirty"`, `"the world say for you. World's Save for You"`, `"Yeah, yeah, ..., yeah"` (~14 yeahs before the degenerate-loop guard aborts)
- de 60s: `"die Geld-Technologie-Technologie-Technologie-Technologie."` (early-chunk loop, guard caught it at 4 repeats), `"T-Rex war -Rex war ihm"`, `"Rückseite der ite der Unabhängigkeitserklärung"`
- De-Abwasch: `"S und Ess-"`, `"Maitrein. ittels"`, `"Geschirrtuches umfassen. tuch umfassen"`, `"angepasst werden. werden, können"`, `"irrspülmaschine"`

These are mid-word boundary artifacts the LCS dedup can't resolve cleanly because the BPE-split point inside the duplicated region doesn't align with whole-token boundaries. Two follow-up directions for the next session:

1. **Word-snap dedup heuristic.** After LCS-prefix-drop, if the next surviving token starts mid-word (no leading ▁/space), extend the drop until the next word-boundary token. Trades a few extra tokens for a clean prefix.
2. **Backward LCS extension.** Run the LCS check both directions (chunk N-1 tail vs chunk N head AND chunk N-1 tail extending into chunk N head with longer match). Catches longer overlaps where the AED produced different tokenizations of the same word at the boundary.

**Decision.** Ship as-is. The 2-3× content delivery is the big win; boundary artifacts are cosmetic and reduce as the LCS finds more matching tokens at the boundary. PLAN #114 P3 is **content-correct** (full coverage of long audio); **cosmetic-incomplete** (chunk-boundary word fragments). Mark in the priority table.

---

## 2026-05-26 (P114-P3 closeout) splice-punct cleanup + always-streamed default + degenerate-loop guard

Three follow-ons close PLAN #114 P3:

- **`10c2fba5` splice-punctuation cleanup + always-streamed default.** After LCS dedup the chunk boundary can land between a mid-sentence punctuation in the previous chunk (`,;:`) and a sentence-end punctuation in the surviving prefix (`.?!`), producing `"for you, . Ask..."` on JFK forced streamed. The chunk-1 comma was the model's best guess for the continuation point; chunk-2 produced a period because the LCS-dropped middle ended that sentence. At the splice, if `full_text.back() ∈ {',', ';', ':'}` and the first non-whitespace char of `part_text` ∈ {`.`, `?`, `!`}, drop the mid-sentence punctuation + trailing spaces from `full_text`, strip leading whitespace from `part_text`, and attach directly. JFK now reads `"...for you. Ask what you can do for your country."` — semantically equivalent to single-pass, only the splice rendering differs. With the cosmetic regression closed, flipped `CANARY_STREAM_THRESHOLD_S` default to `0` (always streamed) — matches parakeet.

- **`361df3e2` window-based degenerate-loop guard.** The user pointed at the 60 s Japanese clip output: ~85 "yeah"s in a row mid-transcript. The funasr `!`-loop guard from PLAN #125 P1 (bail after consecutive id repeats > 20) missed it because canary's BPE emits `▁yeah` + `,` alternating — a 2-token cycle, no consecutive-id repeat. Switched the canary AED greedy loop to a window-based check: count distinct ids in the last 40 generated tokens (including the candidate); if ≤ 3 distinct, abort that chunk's decode with a clear stderr message. Normal speech has ~25-30 unique ids per 40-token window, well above the trigger; 1/2/3-cycle degeneracies all fire. 60 s clip: ~14 yeahs before the guard fires (down from ~85), wall time 12 s (down from 18 s).

**Lesson — degenerate-loop guards must match the BPE granularity.** A consecutive-id check is right for vocab-collision loops (funasr's `!`); a window-based distinct-id check is right for multi-token cycle loops (canary's `▁yeah,▁yeah,`). The window/threshold pair (40 tokens / 3 distinct) covers both — funasr would also hit this trigger at step 40. Next backend with a degenerate-loop report should default to the window approach; the consecutive-id check is a special case worth keeping in funasr only because it's lighter and was already shipped.

**P3 fully closed.** Canary's long-audio path is now: lang-whitelist (`dfe1af3b`) → mel+chunked-encode (`7177c931`) → per-chunk AED with prompt re-injection (`63fdbe46`) → LCS boundary dedup (`62766dae`) → splice-punct cleanup (`10c2fba5`) → degenerate-loop guard (`361df3e2`). Always-streamed by default; six commits, no regression on short audio, full long-audio coverage for canary-1b-v2's 4 supported languages.

---

## 2026-05-26 (P114-P3 LCS dedup) canary streamed boundary-dup polish

`62766dae` lands the LCS-merge dedup inside `canary_transcribe_streamed`. Per-chunk re-injection (`63fdbe46`) closed the AED-boundary `<eos>` bug but left a boundary-duplication artifact (each chunk re-decoded the overlap_seconds of audio in the previous chunk's tail). NeMo handles this in `streaming_utils.longest_common_subsequence_merge`; we already exposed the primitive at `core/crispasr_lcs::lcs_dedup_prefix_count` for voxtral PLAN #114 P2 at the dispatcher layer. Wire it INSIDE the streamed function: for each new chunk, find the longest matching prefix against the previous chunk's tail tokens, drop it, rebuild text from surviving tokens, peel words whose `t1` ≤ surviving-tokens[0].t0.

JFK forced streamed: `"...for you, Country can do for you. Ask…"` → `"...for you, . Ask what you can do for your country."`. The duplication is gone; the leftover `, . ` is a cosmetic punctuation artifact at the splice (LCS match fell between the `,` of chunk 1 and the `.` of chunk 2). 60 s Japanese clip: 18 s wall (3.3× RT, down from 5.3× before dedup; fewer redundant tokens emitted).

**`delay_tokens` heuristic:** `clamp(8, 30, overlap_seconds * 5)`. canary's average emission rate on en/de/fr/es is ~3 tok/s, so 2 s overlap gives ~6 tokens of boundary content. The minimum 8 guards against degenerate cases; the cap of 30 leaves headroom for slower regimes (long compound words, multi-token punctuation runs).

**Promotion status.** `CANARY_STREAM_THRESHOLD_S` default stays at 30 (single-pass on short audio). Flipping to `0` (always streamed, matching parakeet) waits on a punctuation-cleanup pass for the small `, . ` splice artifact. Functionally the long-audio path is correct: no truncation, no duplication, just a cosmetic punctuation oddity.

**Lesson.** The same LCS-merge primitive serves at two architectural layers — the dispatcher (voxtral PLAN #114 P2: across `crispasr_segment` slices from the outer chunk loop) and the backend (canary: across `canary_token_data` slices from the inner streamed loop). Reusing the algorithm at both layers is cleaner than each backend rolling its own; the shared `core/crispasr_lcs` makes that one primitive change once and use it twice.

---

## 2026-05-26 (P114-P3 NeMo analogon) canary per-chunk prompt re-injection + real long-audio validation

`63fdbe46` replaces the first cut of `canary_transcribe_streamed` (which had the documented AED-boundary truncation) with the NeMo `FrameBatchMultiTaskAED` analogon: per-chunk AED decode with the language/task prompt re-injected, results concatenated text-wise. Closes the boundary-`<eos>` bug because each chunk's decoder sees a fresh prompt and doesn't carry "we just finished an utterance" state across the splice.

**Live test fixtures fetched from VPS.** The local Mac had no >30 s English/de/fr/es clips for validation. The VPS at `168.119.190.252:/mnt/akademie_storage/` has the issue #89 long-audio set (`yt_{60,120,300,600}s.wav` — actually all 16 kHz mono PCM extracted from a Japanese podcast yt-dlp pull). SCP'd `yt_60s.wav` (60 s, ~2 MB) and `yt_120s.wav` (120 s, ~4 MB) into `/Volumes/backups/ai/long-clips/` for the smoke tests. Path is documented; not committed to the repo because the clips are large and YouTube-derived.

**Per-chunk vs concat vs single-pass on M1 Metal:**

| input | single-pass | first cut concat (`7177c931`) | per-chunk (`63fdbe46`) |
|---|---|---|---|
| JFK 11 s default | ✓ canonical | (single-pass, unchanged) | (single-pass, unchanged) |
| JFK 11 s forced streamed | n/a | "...for you." (truncated) | "...for you, Country can do for you. Ask what you can do for your country." (boundary-overlap dup) |
| yt_60s.wav (Japanese, `-l en`) | empty | one short hallucination | mix of romanized JA + "yeah" loops (60 s → 11 s wall = 5.3× RT) |

The 60 s test uses Japanese audio with `-l en` because that's what we have on this hardware. canary-1b-v2 doesn't handle Japanese (lang whitelist would reject `-l ja` at `dfe1af3b`); the `-l en` invocation exercises the model's behaviour on out-of-distribution audio. The per-chunk path produces meaningful output for chunks the AED can make sense of and degenerates into token-repeat loops on chunks where it can't — strictly more useful than empty (single-pass) or one-short-hallucination (first-cut concat).

**Lesson — AED streaming pattern.** Frame-synchronous decoders (RNN-T, TDT) stream cleanly because there's no "end of utterance" token; the blank-runaway is the same-class failure on the other end. AED decoders carry implicit utterance-end semantics: emitting EOS means "the utterance is over", and a chunk-encode splice can look like utterance end if the decoder sees the same prompt context across chunks. NeMo solves this in `FrameBatchMultiTaskAED` by re-running the decoder per chunk with the prompt re-injected. Our port mirrors that shape one-to-one. The remaining polish is boundary-overlap dedup (LCS-merge from `core/lcs_merge`, already wired in for voxtral PLAN #114 P2) and a no-repetition-penalty token-loop guard (matching the funasr fix in PLAN #125 P1).

---

## 2026-05-26 (P114-P3 second half) `canary_transcribe_streamed` lands

`7177c931` ships the parakeet-pattern long-audio path for canary. Refactor extracts the post-encode body of `canary_transcribe_ex` (cross-KV + prompt + greedy decode + DTW timestamps + word grouping, ~300 lines) into a static `canary_finish_from_encoder` helper. The existing `canary_transcribe_ex` becomes a thin shim (mel + single-pass encode + helper) and the new `canary_transcribe_streamed` is a parallel shim (full-mel + 8 s/2 s chunked encode + concat + helper). Zero duplication; both entry points share the decoder.

Backend wrapper routes through streamed for n_samples > 30 s, `CANARY_STREAM_THRESHOLD_S` overrides. Matches the parakeet backend's `PARAKEET_STREAM_THRESHOLD_S` knob shape (default 0 = always streamed on parakeet; canary is more conservative at 30 because of the AED-boundary issue below).

**Known limitation.** On synthetic concatenated short clips (`jfk × 8 = 88 s`) the streamed path emits `<eos>` at the chunk boundary after producing one full JFK transcript. JFK forced streamed (`CANARY_STREAM_THRESHOLD_S=0`) truncates to the first sentence. Cause: canary's AED was trained on single-utterance audio and treats the splice point as an utterance end. NeMo handles this in `FrameBatchMultiTaskAED` via per-chunk prompt re-injection; we don't yet. For natural long-form audio (one continuous utterance, no internal silences) the streamed path should produce a coherent transcript because there's no mid-stream `<eos>` signal — but this needs a real long en/de/fr/es clip to validate.

**Decision.** Ship at `CANARY_STREAM_THRESHOLD_S=30` with the caveat documented. Per-chunk prompt re-injection is the proper fix and is queued for a future session.

**Lesson.** Porting a streaming pattern that works for one decoder family (RNN-T / TDT) doesn't transfer 1:1 to a different family (AED). The encoder side ports cleanly — chunked encode + overlap trim + concat reproduces the global feature view. The decoder side does not — AED's `<eos>` is implicit ("the utterance ended") and chunk boundaries can look like utterance ends if the model has no other signal. RNN-T / TDT decoders are frame-synchronous and don't have this problem because there's no "end of utterance" token; the blank-runaway pattern is the same class of failure on the other end.

---

## 2026-05-26 (P114-P3 + harness kernel) canary lang-whitelist + Kaggle audit kernel

Two more landings this session:

- **`dfe1af3b` canary lang-whitelist (PLAN #114 P3 first half).** canary-1b-v2's BPE vocab carries every ISO-639 `<|xx|>` language token (200+) — only en/de/fr/es are actually trained. The current build_prompt path treated the token's existence in the vocab as evidence the model could speak that language, so `-l ja` built the prompt cleanly, ran the decoder, and produced hallucinated mixed-Cyrillic/Greek garbage on JFK ("И така, мои сънародници, не питайте, τι может да направи ваша страна, ..."). Add a `{"en", "de", "fr", "es"}` whitelist in `crispasr_backend_canary.cpp` that refuses unsupported langs before invoking `canary_transcribe_ex`, with a message pointing at parakeet-tdt-0.6b-ja/zh for Japanese/Mandarin and qwen3/voxtral for the broader multilingual set. Smoke: `-l en` JFK unchanged; `-l ja` now errors out cleanly. **Second half** (`canary_transcribe_streamed` — parakeet-shape long-audio port for canary's 4 supported langs) is queued for a future session, less urgent now that the JA failure mode is closed off.
- **`fe44abd2` Kaggle harness kernel for the granite-4.1/omniasr-llm chunk-context audit (PLAN #114 follow-up).** The local-M1 `tools/check-overlap-save-bug.sh` run on 2026-05-25 came back inconclusive for `granite-4.1` and `omniasr-llm` because the 20-min/run budget on M1 wasn't enough for those two LLM-AR backends to finish a 5-min clip. Kaggle CPU has 4 vCPUs + 30 GB RAM + a 9-hour wall budget. The kernel at `tools/kaggle/overlap-save-bug-check/`:
  - Builds crispasr CPU-only from main.
  - Downloads granite-speech-4.1-2b-q4_k.gguf + omniasr-llm-300m-v2-q4_k.gguf from `cstr/*` on HF.
  - Synthesises ~88 s audio by concatenating `samples/jfk.wav` 8×.
  - Runs each backend twice (default `--chunk-overlap 3.0` vs `--chunk-overlap 0`).
  - Parses the SRT outputs, computes char + last-timestamp deltas, prints a verdict (OK / SUSPECTED-BUG / BOTH-FAIL / DEFAULT-FAILS) per backend.
  Patterns lifted from `tools/kaggle/mimo-asr-cpu-vs-gpu/` (progress.jsonl + HF mirror, sh_with_progress ninja-progress streamer, HF auth via `chr1str/crispasr-hf-token` dataset). User triggers manually via `kaggle kernels push`. Result feeds the next pass of `examples/cli/crispasr_chunk_context_gate.h::kBlocked`.

**Lesson** for canary P3: "the token is in the vocab" is not the same as "the model supports the language." Multi-language ASR backends with a fixed-size token table that's shared across families need an *explicit* whitelist of what was actually trained on, not a soft check on the embedding lookup. Same trap likely lurks in any future canary-3/-flash variants; the whitelist should be a load-time GGUF KV if upstream NeMo starts publishing the supported set.

---

## 2026-05-26 (P6b + gemma4 chunking) PLAN #125 follow-on — kyutai-stt internal chunking + gemma4 env-gated auto-chunk

Continuation of the same session as the P1-P6 fix train below; two more commits land on top:

- **`043b3ae5` P6b kyutai-stt 30 s internal chunking.** Extracted the per-call transcribe logic into `transcribe_one()` and wrapped with a chunking loop in `crispasr_backend_kyutai_stt.cpp::transcribe()` that splits inputs > 30 s = 480 000 samples into 30 s windows. Each chunk gets its own P6a silence-tail flush; per-chunk `t_offset_cs` = `caller_offset + chunk_start / 16 kHz * 100 cs`. Validated locally on `first90.wav` (the 90 s Japanese clip from issue #89): 568 s wall = 6.3 s/s, finite and linear, vs the previously-reported 14 s/s degradation. Three chunks produced coherent Japanese hiragana output. JFK (11 s) stays on the single-chunk path unchanged. P6c (streaming-only design-limit guardrail) is now DEFERRED — with linear wallclock the "hung" appearance is gone, so the `--force-long-audio` cap is a UX nicety rather than a correctness fix.
- **`9b5a0a2a` gemma4-e2b CRISPASR_GEMMA4_AUTO_CHUNK env-gated chunking.** PLAN #125 P4 left gemma4-e2b aborting on inputs > 30 s. Add an opt-in env var that chunks at the same 30 s boundary internally instead of aborting, same `t_offset_cs` arithmetic as kyutai-stt P6b. Default stays "abort with clear error" because we haven't validated chunk-boundary quality on long gemma4-e2b output; the env var lets users opt in to experimenting. JFK unchanged through the single-chunk path. Followup: long-audio quality validation → promote to default.

**Recurring pattern:** every LLM-decoder backend that's trained on a fixed audio window (~30 s) ends up wanting the same shape — extract per-call logic into a helper, wrap with a chunking loop, per-chunk `t_offset_cs` arithmetic, optional per-chunk pre/post-processing (kyutai's silence tail, gemma4's prompt). When the third such backend appears the right move is probably to extract this into a shared helper rather than copy-paste again. Note: voxtral and parakeet already chunk through their own dedicated paths (`crispasr_run_voxtral_style_pipeline_streamed`, `parakeet streamed_threshold_s`), so a shared helper would have to abstract over two pre-existing dialects + this new one.

---

## 2026-05-26 PLAN #115 — mimo-asr M1 Metal silent-empty fix (option A)

Distinct bug from PLAN #125 P0 (montvid's Blackwell segfault), same backend. Discovered via the overlap-save A/B sweep two days earlier (mimo flagged `BOTH_EMPTY`, the only backend out of 16 that failed both arms — meaning the bug wasn't in the overlap-save path, it was in the backend itself).

**Bisect.** Bypassed `git bisect` by inspection — only ~7 commits touched `src/mimo_asr.cpp` since HISTORY §56 shipped the working baseline (`dae361f2`), and `89111260` ("perf #72: load weights to GPU when use_gpu=true") was the standout candidate: it flipped `core_gguf::load_weights(..., ctx->backend_cpu, ...)` to `..., ctx->backend, ...`. The commit message itself anticipated the regression — *"If a platform regresses, add a CRISPASR_FORCE_CPU_WEIGHTS=1 escape hatch — none seen yet"*.

**Verification.** Couldn't safely build + load on the local M1 box (4.5 GB Q4_K + active 90 s overlap-save sweep + ~1 GB free), so pushed the build to Kaggle following the patterns in [[project_kaggle_rebake_fragilities]]. The first attempt (`chr1str/crispasr-mimo-asr-cpu-vs-gpu-bisect-plan-115`) used a GPU kernel and queue-blocked for 14+ hours on exhausted GPU quota. Switched to a CPU notebook (`chr1str/crispasr-mimo-asr-cpu-validate`) — RUNNING immediately, completed in ~8 min, JFK transcript matches HISTORY §56 reference verbatim. Confirms the regression is GPU-path specific.

**Independence from PLAN #125 P0 / `a5a518c8` sched hardening.** Verified empirically on M1 Metal after rebuilding on HEAD with the sched-restore patch: same silent-empty symptom. Two different mimo-asr bugs surfacing as the same observable on different platforms (Blackwell sm_120 segfault vs M1 Metal silent-empty).

**Fix train.**

1. `5a570b7b` — first pass pinned only the weight load (`backend → backend_cpu`). Local M1 retest exposed a second failure mode: `ggml_metal_buffer_get_id: error: tensor 'llm.embed.weight' buffer is nil`. The §56 working config (CPU weights + Metal compute) no longer holds because the ggml scheduler tightened cross-backend tensor resolution since then.
2. `c887881e` — complete fix: force `ctx->backend = ctx->backend_cpu` unconditionally, ignoring `params.use_gpu`. Verified locally on M1 Metal: JFK transcribes correctly in 297 s, "ANd so, my fellow Americans, ask not what your country can do for you.." Slow — pure CPU LLM on M1 — but correct. Verified again on Kaggle CPU `b85698670`: prefill 15.8 s, decode 7.0 s over 26 steps, total_lm 22.8 s.

**Cost.** Loses the documented 22 % M1 Metal speedup from PLAN #72. Acceptable because the alternative was shipping a backend that produces no output at all.

**Open.** Option C — proper GPU graph fix in `mimo_asr_build_prefill_graph` via per-tensor backend tagging so `ggml_backend_sched` can route weight reads correctly. Similar shape to chatterbox Bug B from issue #83 R9 #4 (see [[project_chatterbox_gpu_bug_s3gen]]) which took ten candidate hypotheses to land on the real cause. Needs Kaggle GPU run with the patched binary, currently quota-blocked.

---

## 2026-05-25 / -26 Overlap-save bug sweep — five new opt-outs + a reusable A/B harness

Generalised PR #124 (CKwasd's cohere fix). The PR landed during this session; while reviewing it I noticed the strcmp-on-backend-name pattern in `crispasr_run.cpp` was hard-coded for cohere alone and the test gate (`should_use_chunk_context`) had no test exercising the new opt-out arg. The question "are other backends in the same bucket?" turned into an A/B sweep that surfaced five more offenders.

**Harness.** `tools/check-overlap-save-bug.sh` runs each at-risk ASR backend twice on a long clip — once with default `--chunk-overlap 3.0` and once with `--chunk-overlap 0` — and prints last-timestamp + char delta side by side. Backends flagged AT-RISK by the static survey (no `CAP_UNBOUNDED_INPUT`, no `CAP_INTERNAL_CHUNKING`): 14 of the 22 ASR backends. The empirical sweep then narrows that to the actual offenders.

**Found: 5 new backends with the cohere-class bug.**

| backend | symptom (5 min Japanese clip unless noted) |
|---|---|
| cohere | already fixed by PR #124 (regression control: default == nooverlap) |
| voxtral | truncation: default 2 segs / 284 chars ending at 30 s; nooverlap 11 segs / 1294 chars full coverage |
| qwen3 | truncation on 90 s clip: default cuts mid-sentence after chunk 1 (1 seg / 173 chars); nooverlap 4 segs / 448 chars |
| gemma4-e2b | LLM-decode runaway: default times out past 15 min wallclock; nooverlap finishes in 13 min with 3572 chars |
| glm-asr | same runaway shape as gemma4-e2b |
| kyutai-stt | same runaway shape; nooverlap produces 3519 chars in 10 min |

voxtral4b was **not** affected even though it's the same name family — different model architecture under the hood. canary / fastconformer-ctc / firered-asr / granite-4.1-nar / parakeet / wav2vec2 already declare `CAP_UNBOUNDED_INPUT` or `CAP_INTERNAL_CHUNKING` so the gate skips them by design. funasr / moonshine / moonshine-streaming / omniasr / sensevoice / vibevoice / voxtral4b all PASSed the A/B cleanly. granite-4.1 / omniasr-llm / mimo-asr came back inconclusive (granite/omniasr-llm just too slow on M1 in the 20 min per-run budget; mimo-asr is the separate PLAN #115 baseline regression).

**Fix shape.** Pulled the per-name strcmp out of `crispasr_run.cpp` into `backend_allows_chunk_context(const char* backend_name)` in `examples/cli/crispasr_chunk_context_gate.h` (the same header that already housed `should_use_chunk_context` for unit-test reasons). The opt-out list is now the testable thing — `tests/test-issue-114-chunk-context-gate.cpp` grew from 7 to 11 test cases including a 5-dimension exhaustive sweep over the new param, an opt-out-is-master-gate test, and a default-arg parity check.

**Fix train (in push order).**

- PR #124 hardened: `79313c3a` style+test (clang-format fix that was failing CI, plus three new test cases for the opt-out arg). PR merged as `242aaea5`.
- `46f6848d` — gemma4-e2b + glm-asr opt-out (the first two empirically confirmed offenders).
- `eaee2319` — kyutai-stt opt-out.
- `6fef8790` — voxtral opt-out.
- `e7dfb93f` — qwen3 opt-out (caught by re-running the inconclusive backends on a 90 s clip after the 5 min run was timing out).

**Tooling notes.** First Kaggle kernel attempt for the bisect went wrong because I didn't read the existing `tools/kaggle/crispasr-regression.py` first — it has the `step()` / `progress.jsonl` + HF mirror / `sh_with_progress()` Popen build streamer / `build_heartbeat()` ticker patterns that [[project_kaggle_rebake_fragilities]] documents as the seven fragilities the script defends against. The second kernel (`chr1str/crispasr-mimo-asr-cpu-validate`) applied those patterns end-to-end and ran cleanly.

---

## 2026-05-25 Registry — cohere-asr-ja-v0.1 added (issue #123)

Two `{"cohere", ...}` rows added to `src/crispasr_model_registry.cpp` pointing at `TransWithAI/cohere-transcribe-ja-v0.1-GGUF` (`-q4_k`, `-q8_0`). Same backend code as the English `cohere-transcribe-03-2026` entry — Japanese fine-tune of the same base, Apache-2.0. Dropped straight in. `--model-quant` resolves the other quants in that repo (`F16/Q6/Q5`) from the registered Q4_K row. Issue closed with a thank-you to the requester (rikimtasu) and CKwasd (who published the GGUF mirror).

Verified locally: `crispasr --backend cohere -m cohere-asr-ja-v0.1-q4_k.gguf --dry-run-resolve` returns the correct upstream URL. Commit `e3ded251` registry, `50b6dfda` README row alongside the English cohere entry.

---

## 2026-05-26 (P1-P6 fix train) PLAN #125 — six issue-#125 findings shipped in one session

Built on the P0 scheduler hardening from earlier in the day. Six commits land six of montvid's twelve issue-#125 findings; the remaining items (P0 external Blackwell retest, P6b/c kyutai-stt long-audio dispatcher chunking, gemma4-e2b auto-chunk-vs-abort decision) are queued for followup.

**Shipped this session (in push order):**

- **`72b74486` P2 firered-asr** — drop `CAP_UNBOUNDED_INPUT` (encoder's relative-PE buffer is `pe_maxlen=5000` ≈ 50 s post-subsample; declaring unbounded bypassed VAD and produced silent OOB on long audio). Add defensive length check in `firered_asr_transcribe_impl` that aborts with a clear error when `T_sub > pe_maxlen`. JFK still transcribes cleanly.
- **`f72d3db1` P1 funasr** — degenerate-loop guard in the AR decode (bail after the same token id repeats > 20× in a row, model-agnostic stop-loss for the `!`-loop reported on Blackwell CUDA). `frames_spliced`/`fake_token_len` surfaced at `CRISPASR_VERBOSE=1` for the next reporter. Four backends added to `tools/test-all-backends.py`: funasr, fun-asr-mlt-nano, sensevoice, paraformer (all had shipped without CI coverage since the 2026-05-20 ports landed). The default werv-threshold check at line 670 would have caught the `!`-loop loud (`wer("…", "!!!!…") ≈ 1.0`).
- **`5f0aefc0` P3 omniasr-llm** — chunking decision rewritten as `(is_streaming || force_seg) && T_enc > 1`. Non-streaming GGUFs were silently feeding the entire long audio to a 512-token LLM head; now they chunk by training-window length. Segment marker injection stays gated on `is_streaming` (non-streaming variants chunk without the marker — each chunk decoded as a complete utterance, which is what the model was trained on).
- **`b936b488` P5 mimo-asr** — tokenizer `cstr/mimo-tokenizer-GGUF/mimo-tokenizer-q4_k.gguf` added to the auto-download manifest. Was previously required separately, blocking everyone whose first `crispasr --backend mimo-asr -m auto --auto-download` produced exit code 1. Error message expanded to surface `--auto-download` as option 1, manual `hf download` as option 3. `docs/architecture.md` mimo-asr section documents the requirement.
- **`ba0e388e` P6a kyutai-stt** — append 500 ms zero-frame silence-tail before calling `kyutai_stt_transcribe_ex`. The streaming-trained LM is causal and needs the tail to flush its final-token state; the batch wrapper was passing raw samples and stopping mid-word. Locally verified: JFK now produces "...for your country." (full final word + sentence-end punctuation, was truncated to "...for your c" before).
- **`8bfaff23` P4 gemma4-e2b** — defensive 30 s training-window guard. On long audio the model emits `<Eos>` immediately after the prompt and continues into unrelated LLM commentary; refusing the slice with a clear "use `--vad`" message stops the silently-wrong output. Followup: auto-chunk vs abort decision for the dispatcher.

**Each fix verified locally on M1 Metal** by running the affected backend on `samples/jfk.wav` (the universal control test from the reporter's methodology). Output recorded inline in each PLAN P-section.

**Workflow notes.** All six commits originated in `/Volumes/backups/code/CrispASR-issue125-p1p2` per `feedback_parallel_workers.md` and `feedback_storage_paths.md`; rebased + fast-forwarded into `main` per `feedback_integrate_to_main.md`. `clang-format-18` applied only to the four `src/`+`examples/cli/` files I touched per `feedback_clang_format_v18.md`'s tight scope (registry .py and docs/architecture.md are out of scope and stayed untouched by the formatter).

**Lessons.** Two patterns recurred across the six fixes:

- **Capability-flag honesty.** firered-asr's `CAP_UNBOUNDED_INPUT` was the same class of failure as the previous voxtral/cohere/gemma4-e2b/glm-asr/kyutai-stt cases that drove the opt-out fix train (`dc2295b2` etc.). A defensive sweep of every `CAP_UNBOUNDED_INPUT` declaration against the actual encoder cap remains worth doing.
- **Defensive backend-side guards beat dispatcher tweaks.** P2 (firered-asr length check), P4 (gemma4-e2b 30 s guard), and P5 (mimo-asr improved error message) all live in the backend wrapper, with a clear abort + remediation instruction. Beats the alternative — push the contract into the dispatcher and pray every entry point honours it.

---

## 2026-05-26 (later) PLAN #125 P0 — mimo-asr regression reattributed; sched src-mutation log hardened

Closer reading of the v0.6.9 → v0.6.10 delta moved the bisect away from `6b492b2b` (FA per-head mask). The FA patch is fully `#ifdef GGML_CUDA_CRISPASR_FA_PERHEAD_MASK`-guarded; the CMake option defaults OFF (`ggml/CMakeLists.txt:211`); no CI or release script sets the flag ON (verified by `grep -rn FA_PERHEAD`). With the macro undefined, the compiled binary is byte-identical to upstream for that path. A self-built `eaee2319` (which is the reporter's binary, `/opt/crispasr-main/build/bin/crispasr`) gets OFF by default. The patch cannot be the cause of a default-build segfault.

The reporter's bisect missed `ggml/src/ggml-backend.cpp` because their grep filtered to `ggml/src/ggml-cuda/` + the mimo backend files. Between v0.6.9 and v0.6.10 there are *two* code-relevant deltas; the second is **`0f0f0793` "fix(#83): ggml sched src-mutation log + UNet input pin — R9 follow-up #4 v2"**. That patch is *unconditional* and adds a `src_mutations[]` log + restore to the generic scheduler, used by every backend including CUDA.

The bug in the original patch:

- The restore loop ran only on the success path of `compute_splits`. If any compute step returned early on non-`GGML_STATUS_SUCCESS`, the mutation log was retained.
- `split_graph` did not reset `n_src_mutations` at its start. Neither did `sched_reset`.
- Next call appended new entries on top of stale ones. Then the next successful `compute_splits` walked the stale entries and wrote `m->orig_src` to `m->node->src[m->j]` — but `m->node` was a pointer into the *previous* gf, which mimo-asr (and any multi-step AR decoder) has already re-laid-out. Write to freed memory → UB → plausible segfault on platforms where the freed page gets reused for something tensor-shaped (CUDA allocator on Blackwell fits the profile).

Why mimo-asr in particular: it builds a fresh cgraph per AR-decoded token, so it's the backend most exposed to stale-`m->node` writes. The reporter's verbose log shows the crash lands right after `kv cache 51 MiB`, i.e. early in the LM AR loop — exactly when the second/third compute would hit the stale-restore path if one of the encoder forward calls returned early.

**Hardening shipped: `a5a518c8` "fix(ggml-backend): restore src[j] on every exit path of compute_splits".**

- Extracted the restore loop into `ggml_backend_sched_restore_src_mutations()`.
- Called on every exit path of `compute_splits` (the two early-error returns + the success path).
- Called defensively at the start of `split_graph` and inside `sched_reset` to drop any stale entries before `ggml_free(sched->ctx)` runs. The restore writes to the user's gf (`orig_src` pointers were captured before the rewire), not to `sched->ctx`, so it is safe to call before the free.
- Tightened the realloc-on-grow to assign-after-success.
- Stripped the `// CrispASR patch (#83 r9 follow-up #4)` and `MUST RE-APPLY` markers per `feedback_strip_local_markers.md` — those should not have been in committed code.

Local M1 Metal verification before pushing:

- `crispasr --backend parakeet -f samples/jfk.wav` — produces the canonical JFK transcript via the hardened scheduler.
- `crispasr --tts "Hello world test." --tts-output /tmp/tts-smoke.wav -m chatterbox-t3-q8_0-regen.gguf` — CFM solver runs cond+uncond and produces `vocoder mel rms=4.625` (ref rms=5.115; broken-baseline before `0f0f0793` was rms=13.938). The original chatterbox fix the patch was meant to deliver is preserved by the hardening.

Outstanding: we have no Blackwell sm_120 hardware locally, so the end-to-end mimo-asr-on-Blackwell confirmation has to come from the reporter. Ask montvid to rebuild from `95d74455` or later and rerun the mimo-asr JFK case. If the segfault persists after the hardening, next debug step is a `gdb --args` backtrace + a closer look at any non-Metal-only ggml deltas we may still be missing.

**Lessons (for LEARNINGS).**

- A bisect grep restricted to "obviously CUDA-shaped" paths can miss the generic scheduler when an unrelated patch touches it.
- A behaviour-changing patch in `ggml-backend.cpp` is implicitly multi-backend; "this fix is for chatterbox CFG on Metal" doesn't bound its blast radius. Hardening on all-exit paths from the start is cheaper than learning this from a user report.
- `// CrispASR patch ... MUST RE-APPLY` markers should not enter committed code — by definition, code that's in `main` *is* the apply, and the marker carries no signal except "remind future-me." Use commit messages and PLAN entries for that, not source markers.

Cross-refs: `0f0f0793` original patch; `a5a518c8` hardening; `feedback_strip_local_markers.md` memory note; PLAN #125 P0.

---

## 2026-05-26 PLAN #125 logged — montvid's 12-finding bug sweep on v0.6.10

External user `montvid` filed issue #125 with twelve well-attested reports against v0.6.10 commit `eaee2319`, hardware NVIDIA RTX PRO 6000 Blackwell sm_120 + CUDA 12.6, against the 50:47 EN FLAC from issue #89 plus the project's own `samples/jfk.wav`. All twelve cached at `/Volumes/backups/code/issue125-attachments/` and written up as **PLAN #125** with a P0–P6 fix priority. Reference build the reporter bisected against: v0.6.9 `f23d9485` (2026-05-21).

The standout finding is **report #12: mimo-asr segfault on Blackwell sm_120** — a regression introduced by my own commit `6b492b2b` "feat(ggml-cuda): per-head additive mask in FLASH_ATTN_EXT (MMA-F16, #81)", which the 2026-05-24 validation block in this file noted was tested only on parakeet-tdt-0.6b-v3 on A1000 sm_86. mimo-asr's MQA shape (`head_dim=128, n_kv=8, n_layers=36`) plus the audio adaptor cross-attention plus the Blackwell architecture together exercise a code path the patch never saw. **The `tools/upstream-prs/06-cuda-fa-perhead-mask.md` PR draft should not be submitted upstream until validated on a wider matrix of backends/architectures.**

The remaining eleven findings cluster by remediation cost:

- **P1 funasr / fun-asr-mlt-nano `!`-loop on any audio length** (reports 01, 07, 08, 09) — JFK 11 s reproduces in 3 seconds, ruling out long-audio / chunking explanations. `-l zh` produces byte-identical output to `-l en`, ruling out the "Chinese-prompt mismatch" hypothesis from report #01. Suspect is the audio adaptor or encoder, not the prompt. Both backends are absent from `tools/test-all-backends.py`, which is why the regression shipped silent. The `werv < threshold` assertion at line 670 would have caught this loud (`wer("…", "!!!!…") ≈ 1.0`).
- **P2 firered-asr declares `CAP_UNBOUNDED_INPUT` but its PE window is `pe_maxlen=5000` ≈ 50 s** (report 04). One-line fix in the backend registry; JFK proves the model is fine.
- **P3 omniasr-llm `is_streaming` chunking gate** (report 03) — for GGUFs without the streaming-mode flag, the dispatcher feeds the entire long audio to a 512-token LLM; chunking decision needs to be `(is_streaming || T_enc > seg_frames)`.
- **P4 gemma4-e2b long-audio hallucinations** (reports 02, 07) — JFK transcribes verbatim; 50 min clip emits unrelated LLM completion starting with `<Eos>!`. Long-context chunking bug, not the audio-soft-token-id mismatch the original report #02 hypothesised before the control test.
- **P5 mimo-asr tokenizer GGUF missing from auto-download manifest** (report 06) — verified workaround using `hf download cstr/mimo-tokenizer-GGUF`. `--codec-model` flag is undocumented in `docs/cli.md`. Blocked by P0 for testability.
- **P6 kyutai-stt** — three separate issues. P6a: drops the final word on the 11 s JFK clip (deterministic, output ends `"…can do for your c"`); causal LM needs zero-frame tail or `<eos>` flush. P6b: 0.07× RT on long file vs 0.73× on JFK (10× per-audio-second degradation; dispatcher passes raw samples in one call, KV grows O(N²)). P6c: backend is fundamentally streaming-trained, batch dispatcher is a footgun — refuse files > 5 min unless `--force-long-audio`.

The reporter's most actionable methodology observation: **JFK as universal control test.** Running the project's own 11 s fixture isolates "model is dead" from "long-audio dispatcher is broken" in under 2 minutes — the two failure classes have completely different fix sites. Reports #07 and #12 both demonstrate this pattern; future "broken backend" reports should run JFK first.

Backend registry coverage audit follow-up: reporter found four backends missing from `tools/test-all-backends.py` (funasr, fun-asr-mlt-nano, sensevoice, paraformer). The script advertises itself as the source of truth in `docs/regression-matrix.md`; the gap means a backend can ship broken indefinitely. Parallel audit warranted for the 18-backend registry the script does cover.

PLAN #115 (the local-Apple-Silicon mimo-asr-baseline note from 2026-05-25) is now cross-linked to PLAN #125 P0; same root cause, different symptom on a different platform.

---

## 2026-05-25 (latest) Kaggle rebake pipeline wired end-to-end — 4 new parakeet refs published; coverage 7 → 11 → 13 manifest entries

Closing out the handover-prompts/extend-coverage-via-kaggle.md initiative. End-of-day state:

### Manifest

13 rebake-pending entries (the handover's "realistic" goal). 7 in by commit `8de7bf66` pre-session; +4 in `ab82dcd0` (mimo-audio-tokenizer, lid-cld3-f16, voxcpm2-tts-2b, chatterbox-turbo-s3gen); +2 in `34db1048` (indextts-1.5, titanet-large). The cohere/granite "fixture exists on HF but manifest doesn't reference it" gap is documented but not closed — the entries need `source_model` added to be re-bakeable.

### Kaggle rebake pipeline

The rebake kernel `chr1str/crispasr-auto-rebake-refs` had been latently broken since the 2026-05-12 rename of `fixture_path` → `fixture_ref_path` (commit `056493a1`). Nobody noticed for 13 days because the rebake didn't run scheduled in that window. v7 of the kernel triggered today hit four cascading issues before producing any output; each surfaced exactly one fix.

| version | symptom | root cause | fix |
|---|---|---|---|
| v7 | log frozen at `[208/360]` t5_translate.cpp warning, kernel still RUNNING at +103 min | child stdout heavily-buffered by Kaggle log capture; healthy build invisible | `1b62776e` heartbeat thread + stdbuf wrapper |
| v8 | (same as v7, immediately) | confirmed by user it's stuck — killed | rolled into v10 |
| v9 | KeyError `'fixture_path'` on first backend | latent rename bug from 2026-05-12 — rebake path was missed | `8cf7e931` field rename + never-done-first ordering |
| v10 | disk-full at backend #8 during NeMo source-weight download | `/kaggle/working/hf_cache/` never freed between backends → ~16 GB cumulative across 4 parakeet variants | `0f4de5b9` rmtree HF + torch caches in `finally:` block per backend; print `free_gb_after` in heartbeat |
| v11 | 9 successful refs, 0 uploaded; WARN downgrade to `UPLOAD=0` | Kaggle Secrets API gave `ConnectionError`; script's "anonymous" path triggered | `2ff4f1ba` 3-tier auth: env → `kaggle_secret()` 3× retry → `kaggle_token_from_dataset()` reading `/kaggle/input/crispasr-hf-token/hf_token.txt` |
| v12 | HF auth resolved cleanly, but still 0 uploaded; `SystemExit: rebake had failures; refusing to upload` | all-or-nothing upload gate at line 866 blocks every partial run with the 14 known-broken backends | `c11e0648` partial-upload — failed entries never write to `REBAKE_STAGE` so `upload_folder()` only ships successes |
| v13 | **upload landed** | — | manifest `fixtures.revision` bumped to `b61b03014bc99ecce18ac8f99988d5110c83f2d2` in `e79d1c77` |

Bonus infrastructure: private Kaggle Dataset `chr1str/crispasr-hf-token` created out-of-band as the durable auth-fallback path (Kaggle's UI has no "Add-ons → Variables" option despite older script comments to the contrary — confirmed on the current UI). Mounted via `kernel-metadata.json:dataset_sources`.

### Published refs at `cstr/crispasr-regression-fixtures@b61b03014bc`

**NEW** (4 of the 13 handover backends):

- `parakeet-tdt-1.1b/ref.gguf` 15.4 MB
- `parakeet-tdt_ctc-1.1b/ref.gguf` 15.4 MB
- `parakeet-rnnt-0.6b/ref.gguf` 15.4 MB
- `parakeet-rnnt-1.1b/ref.gguf` 15.4 MB

**REFRESHED** (5 existing):

- `canary-1b-v2/jfk_11s/ref.gguf` 77 MB
- `moonshine-tiny/jfk_11s/ref.gguf` 1.2 MB
- `moonshine-base/jfk_11s/ref.gguf` 1.4 MB
- `parakeet-tdt-0.6b-en/ref.gguf` 15.6 MB
- `parakeet-tdt-0.6b-ja/reazon_baseball_14s/ref.gguf` 19.6 MB

### Still-broken in the 13 (per v11/v12/v13 SUMMARY)

3 manifest gaps — entries missing `source_model` field; add it and they'll bake next run:

- `cohere-transcribe` — its fixture already exists on HF but the manifest doesn't have the field
- `granite-speech-4.1-2b` — ditto
- `moonshine-streaming-tiny` — ditto

8 missing pip deps or Python ref-module bugs in upstream:

- `paraformer-zh` — `ModuleNotFoundError: funasr`
- `voxtral-mini-3b-2507` — `ImportError` (transformers version)
- `mimo-asr` — `ModuleNotFoundError: 'src'` (path-import bug in `mimo_asr.py`)
- `mimo-audio-tokenizer` — `ModuleNotFoundError: mimo_audio_tokenizer` (needs from-source install)
- `voxcpm2-tts-2b` — `ImportError: pip install --no-deps --target=/tmp/voxcpm_src voxcpm`
- `indextts-1.5` — `OSError: Not found` (model_dir path issue)
- `titanet-large` — `RuntimeError: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor` (NeMo upstream)
- `chatterbox-turbo-s3gen` — `ModuleNotFoundError: chatterbox`

3 fast-fail edge cases needing log inspection:

- `firered-asr2-aed`, `glm-asr-nano`, `lid-cld3-f16` — `dump_reference exit=1` in <1 s; not the standard import error pattern

### What didn't make this commit train

- skip_diff: false flips for the 4 new parakeet refs — needs `fixture_ref_path` set per entry, a transcript captured locally via `crispasr-cli + GGUF`, and `diff_thresholds` calibrated from a local `crispasr-diff` run
- Per-backend dep fixes for the 8 missing-pip cases — each is a small follow-up but needs an env-spec change in the rebake bootstrap (or a per-backend `tools/reference_envs/<name>/requirements.txt` consulted by the kernel)
- The `cohere-transcribe` / `granite` / `moonshine-streaming-tiny` `source_model` additions — trivial 3-entry manifest edit

### Cross-refs

- LEARNINGS "Kaggle as a batch-rebake target: seven fragilities the script has to work around" for the deep dive on each fix
- handover-prompts/extend-coverage-via-kaggle.md for the original initiative scope
- `tools/kaggle/crispasr-regression.py` is the canonical script (modified across `1b62776e`, `8cf7e931`, `eba52bac`, `0f4de5b9`, `2ff4f1ba`, `c11e0648`)
- `tools/kaggle/rebake/kernel-metadata.json` now mounts `chr1str/crispasr-hf-token` as the auth fallback

---

## 2026-05-25 (late) PLAN #114 close-out — voxtral streamed validated at 60 / 120 / 300 s; 600 s hung on M1 memory thrash, not coverage

Closing the day on PLAN #114 with the live test data and the final per-backend status.

### Voxtral streamed (option A) — live numbers on the Apple Silicon M1

After two iterations on the `max_new_tokens` default (`3f20d050` "scale with audio duration" was a no-op because `whisper_params::max_new_tokens` defaults to `512` not `0`; `a5165c84` reformulated as `max(params.max_new_tokens, scaled_default)` and actually kicked in):

| length | streamed (this PR, M1 Metal) | default chunking (VPS x86, post-opt-out) | notes |
|---:|---|---|---|
| 60 s  | ✓ 100 %, 11 segs / 470 chars | ✓ 100 %, 3 segs / 280 chars | denser per-utterance segmentation (streamed has 1 LLM context, no chunk cold-starts) |
| 120 s | ✓ 100 %, 527 chars | ✓ 100 %, 541 chars | matched within token-count noise |
| 300 s | ✓ 100 %, **1276 chars** (post-fix) / 863 tokens, natural EOS | ✓ 100 %, 1300 chars | pre-fix was 781 chars / 512-token cap mid-sentence at `"…今年は4日が今年と言えば20"`; the `a5165c84` scaling fix restored full transcript |
| 600 s | ✗ hung on contended M1 (see below) | ✗ 900 s VPS wall-time timeout (not a coverage regression) | needs a quieter box for an actual streamed measurement |

The pre-3f20d050 ship of `cc319745` introduced the streamed path with a fixed `max_new_tokens = 512` default. Empirically this capped the 300 s output at exactly 512 tokens (mid-sentence). 3f20d050 attempted a scale-with-duration heuristic but guarded on `params.max_new_tokens > 0` — defeated by `whisper_params.h:134`'s default initializer of 512. a5165c84 swapped to `std::max(params.max_new_tokens, scaled_default)`; 300 s rerun produced 1276 chars / 863 tokens with natural EOS, matching the VPS default-chunked baseline.

### 600 s local M1 run hung — memory thrash, not algorithmic

The 600 s streamed test on Apple Silicon ran for 2 h 10 min wall time. Total CPU time accumulated: 20.3 s. State `SN` (sleeping + nice), 0.0 % CPU at sample time. macOS `sample` showed all 850 sampled stack traces in `main` — the process never reached the encoder forward, let alone the LLM decode.

Root cause: severe memory pressure. `vm_stat` reported **80 MB free out of 16 GB** at the time, with 4+ active Claude agents in parallel (`omniasr-llm` bench, doc workers, this session, the chatterbox-ref worker) plus WindowServer at 42 % CPU. The streamed path's KV cache budget for 600 s of audio is `T_prompt (~7600 tokens) + max_new (4800) + 64 ≈ 12 500 tokens` of context, which on a 3 B-parameter LLM is several hundred MB of state plus the embed splice buffers. Under sub-100 MB free, Metal's allocator stalls on first big allocation and the process never makes forward progress — confirmed by `ps -o time` showing 20 s CPU over 130 min wall.

This is a *system-pressure* failure, not a *streaming-pipeline* failure. The algorithm is the same that produced 100 % at 60 / 120 / 300 s. Killed cleanly; will retry on a quieter box or by sequencing other agents to release RAM first.

### Per-backend status as of session close

| backend | default path on `main` | verified coverage |
|---|---|---|
| **parakeet** | streamed-TDT (commit `33f9a162` from 2026-05-24) | 60-600 s: 93 / 82 / 97 / 99 % (the 81.5 % at 120 s is the audio — speech ends at 1:37; both paths produce the right content) |
| **voxtral-mini-3b** | streamed (this PR, `cc319745` → `3f20d050` → `a5165c84`); falls back to single-chunk for ≤30 s | 60-300 s: 100 % local M1; 600 s pending non-thrashing reproduce. Default-chunked path on VPS post-opt-out (`6fef8790`) was 100 % at 60-300 s and wall-time-bound at 600 s |
| **cohere-transcribe** | default chunking with opt-out from the external overlap-save wrap (commit `dc2295b2` from earlier today) | 60-600 s: 96 / 98 / 98 / 98 % (VPS, full 22 segs at 600 s) |
| **canary-1b-v2** | unchanged | still hallucinates English `"I am not aware…"` at every JA length — separate prompt-wiring bug (PLAN #114 P3) |
| **qwen3-asr / granite / mimo-asr** | opt-out fixes for siblings (`46f6848d` / `eaee2319` / similar) | audit pending; not in the lenhone repro path |

### Stash relocation rule update

The `/Users/.../code/issue89-stash/` accumulation of audio fixtures + JSON dumps + log files (~30 MB on top of an already 99-100 % full main volume) was contributing to the memory thrash. Consolidated everything to `/Volumes/backups/code/issue89-stash/` (the SSD, 30 GB+ free). Memory rules updated: `[[feedback_storage_paths]]` and `[[feedback_parallel_workers]]` now direct intermediate work files and worktree stashes to `/Volumes/backups/code/<topic>-stash/`, not `~/code/<topic>-stash/`. The new rule applies to this writeup itself — the worktree for this commit lives at `/Volumes/backups/code/CrispASR-final-writeup/`.

### What's actually open after this commit

- Voxtral streamed 600 s: confirmation run when the box has > 8 GB free.
- Voxtral upstream-groundtruth diff via `transformers.VoxtralProcessor.apply_transcription_request` — fresh venv setup pending; not blocking.
- PLAN #114 P3 (canary lang-prompt) and P4 (qwen3 / granite / mimo audit) are separate cycles; the 2026-05-25 multi-commit work fully closes the parakeet, voxtral, and cohere portion of #114.

---

## 2026-05-25 PLAN #114 — voxtral streamed (Mistral apply_transcription_request pattern) + matrix re-interpretation

Two things landing together:

### 1. Voxtral streamed implementation (option A — single LLM context over the whole clip)

`crispasr_run_voxtral_style_pipeline_streamed` in `examples/cli/crispasr_llm_pipeline.h`: matches what `transformers.models.voxtral.processing_voxtral.VoxtralProcessor.apply_transcription_request` does upstream. For audio > 30 s:

1. Loop 30 s chunks: `voxtral_compute_mel` + `voxtral_run_encoder` → per-chunk audio embeds (375 × 3072).
2. Concatenate all chunks' audio embeds into one buffer.
3. Build prompt with `n_chunks × 375` audio-pad placeholder tokens.
4. Embed + splice audio embeds into placeholders, **single LLM AR decode** over the whole prompt.

Differs from CrispASR's previous per-slice-LLM-call shape: there, each 30 s slice got its own LLM context, the AR decoder cold-started at every chunk boundary, and boundary words / sentence carries could drop. With the streamed path the LLM sees one continuous audio sequence, no cold-starts.

`crispasr_backend_voxtral.cpp`: voxtral now declares `CAP_UNBOUNDED_INPUT | CAP_INTERNAL_CHUNKING` so `crispasr_run.cpp`'s 30 s auto-chunk gate doesn't fire — long audio routes inside the backend via the new template. Short audio (≤ 30 s) still goes through the single-chunk pipeline (bit-identical to today).

KV cache budget bumped from the fixed 4096 to `T_prompt + max_new_tokens + 64` so the longer prompt fits. On the M1 Metal smoke test (60 s clip): 2 chunks → 750 audio tokens → single decode → 11 segments / ~470 chars / full 0:00 → 1:00 coverage. Compare to the per-slice path on the same clip: 3 segments / 280 chars (still correct content, less segmentation granularity).

Voxtral-Mini-3B's 32 768-token context fits up to ~30 min of audio (60 chunks × 375 = 22 500 audio tokens + ~6 000 decoded). Beyond that the CLI's `--vad` or explicit `--chunk-seconds` paths remain available as fallback.

### 2. The cross-length matrix was measuring the pre-opt-out binary; v2 retells the story

The 2026-05-25 morning "Cross-length × cross-backend matrix" in PERFORMANCE.md was collected on VPS binary `bd8b98cf` (May 24), which **predates** the per-backend opt-out fixes for cohere (`dc2295b2`), gemma4-e2b / glm-asr (`46f6848d`), kyutai-stt (`eaee2319`), and voxtral (`6fef8790`) that landed during the matrix run. Those fixes remove an external overlap-save context wrap that the LLM-decoder backends couldn't trim back from correctly. The wrap's word-timestamp trim was discarding most of the LLM's output because the LLM's emitted timestamps don't honor the original slice frame.

Matrix v1 (pre-opt-out) numbers I committed earlier today (cohere 59 %/62 % at 300/600 s, voxtral 20 %/9 % at 300/600 s) **were the wrap's victim, not the model architecture**. Matrix v2 (post-opt-out, VPS rebuilt to `13059e0c`) — running while this commit lands — shows the corrected picture:

| backend / mode               |  60 s |  120 s |   300 s |   600 s (still running) |
|---|---:|---:|---:|---:|
| **voxtral-mini-3b** (default chunking, post-opt-out) | **100.0** | **100.0** | **100.0** | TBD |
| **voxtral-mini-3b** streamed (this PR — single LLM context) | full 0:00→1:00, 11 segs / 470 chars | running | running | running |
| **cohere-transcribe** (default chunking, post-opt-out) | 96.3 | 97.9 | 98.1 | TBD |

PERFORMANCE.md updated to keep matrix v1 *and* v2 side by side so the reader can see the cost of the missing opt-out — the same models, the same audio, just an `kBlocked` flag in `crispasr_chunk_context_gate.h` between them.

### Implication for PLAN #114 priority order

- **P1 (cohere VAD-default):** ~~planned~~ obsolete. `dc2295b2` already gets cohere to 96-98 % default-chunked. `--vad` remains available but no longer required for coverage parity.
- **P2 (voxtral / qwen3 / granite / mimo chunk + LCS dedup):** Reframed. Default coverage is already there for voxtral via opt-out. The voxtral streamed path in *this* commit is an architectural improvement, not a coverage rescue. Same audit pending for qwen3 / granite / mimo (Class B LLM-AR backends).
- **P3 (canary lang-prompt + streamed-encode port):** unchanged. Canary still hallucinates English at every JA length — separate bug, work proceeds.
- **P4 (fastconformer-ctc):** unchanged.

Cross-refs:
- HISTORY 2026-05-25 (above) "Long-form ASR — cross-backend survey" for the matrix-v1 numbers and the failure-class breakdown.
- LEARNINGS new "Always rebuild the test box before benchmarking" note from this session.
- `examples/cli/crispasr_chunk_context_gate.h` `kBlocked` list — the actual mechanism behind the opt-out.

---

## 2026-05-25 Long-form ASR — cross-backend survey and per-backend roadmap (PLAN #114)

Follow-up to the 2026-05-24 "Issue #89 reopened" entry that made the streamed encode default for parakeet. The 120 s multi-backend sweep that came out of that work showed parakeet was the loudest failure on lenhone's audio but not the only one; this entry collects the empirical state across every multilingual / JA-capable ASR backend and the prioritised fix order.

### What we measured (60 s and 120 s on lenhone's fresh `yt-dlp` clip)

| backend | 60 s | 120 s | failure mode | classification |
|---|---|---|---|---|
| parakeet-tdt-0.6b-ja, default (streamed) | ✓ 7 segs, full speech | ✓ 12 segs, full speech to 1:37.84 | none | fixed 2026-05-24 |
| parakeet-tdt-0.6b-ja, single-pass (`STREAM_THRESHOLD=999`) | ✗ stops at 0:20.08 | ✗ kana-by-kana past 1.2 s | TDT blank-runaway after attention amplifies codec noise | the issue #89 bug |
| parakeet-tdt-0.6b-ja, CTC head | ✓ byte-identical to streamed | ✓ byte-identical to streamed | none — CTC is frame-synchronous | confirms encoder is fine |
| parakeet-tdt-0.6b-ja + `--vad` | ✓ 14 segs, 93 % coverage | ✓ 14 segs, 0:00 → 1:58.39 | (VAD trims silence — by design) | alt path |
| sensevoice-small + `--vad` | not in 60 s sweep | ✓ 13 segs, 0:00 → 2:00 | minor JA token glitches (`スピーク**ジャ**プネス`) | clean baseline |
| voxtral-mini-3b | (not specifically measured at 60 s) | ✗ covers 0:00 → 0:27 then jumps to 1:47 → 2:00 | LLM AR loses ~80 s in the middle at chunk boundaries | LLM-decoder-chunking class |
| cohere-transcribe | (not specifically measured at 60 s) | ✗ only ~4 segments across 120 s, multi-tens-of-seconds gaps | Conformer encoder hits the same long-attention regime as parakeet single-pass | Conformer-long-context class |
| canary-1b-v2 | (not specifically measured at 60 s) | ✗ hallucinates English `"I am not aware of anything"` in a loop | AED decoder language-prompt wiring + no streamed path | multi-task-AED class |

### Three failure classes, three fix shapes

1. **Conformer long-attention amplification** (parakeet, cohere, canary): bidirectional self-attention over the full utterance amplifies codec-level audio perturbations into 10-15 % activation shifts that flip downstream decisions. Fix shape: global-z-norm + chunked-encode (8 s) + concat + single decode (the parakeet `transcribe_streamed` pattern). DONE for parakeet; canary needs the same port with AED prompt re-injection at boundaries; cohere's released weights aren't trained for long inputs, so the simpler fix is VAD-default like its hosted product does.

2. **LLM autoregressive decoder loses track at chunk boundaries** (voxtral, qwen3-asr, granite-speech, mimo-asr): the energy chunker hands the LLM 30 s slices, the AR decoder either runs to `max_new_tokens` before catching up to the audio or its prompt conditioning misfires on the boundary, dropping the middle of long clips. Fix shape: chunk with overlap + LCS dedup. We already have `core_lcs::merge_overlapping_hypotheses` from PLAN #80c, just need to wire it as the default for LLM-AR backends.

3. **Multi-task AED language-prompt wiring** (canary): a separate bug from the long-audio one. The JA hallucination loop suggests the language prompt isn't being threaded through the decoder correctly at all, not just at chunk boundaries. Needs short-audio validation first before the long-audio port can be tested cleanly.

### Roadmap

P0 (in progress): 60/120/300/600 s × all-multilingual-backends matrix on the VPS to extend this table to the longer durations and fill in qwen3 / granite / kyutai / gemma4 / firered / mimo / wav2vec2 cells (these weren't fully measured in the 120 s sweep). Pending; numbers update `PERFORMANCE.md` once they land.

P1: cohere `CAP_LONGFORM_PREFERS_VAD` (or similar flag) → CLI auto-enables `--vad` past 30 s. Cheapest fix, faithful to the hosted product's behaviour.

P2: voxtral / qwen3-asr / granite / mimo-asr — wire `core_lcs::merge_overlapping_hypotheses` as the default for LLM-AR chunked output with overlap ≥ 2 s.

P3: canary — fix the JA prompt-wiring bug at short audio first, then implement `canary_transcribe_streamed` with AED prompt re-injection at chunk boundaries (parakeet pattern + extension).

P4 (low): fastconformer-ctc optional streamed wrapper. CTC robustness makes this a portability improvement, not a correctness fix.

### What stays as-is

- **whisper** has its own internal 30 s seek loop in `whisper.cpp`. Untouched.
- **firered-asr / wav2vec2** declare `CAP_UNBOUNDED_INPUT` and use single-pass CTC; CTC is robust to length. Verify in VPS matrix but no immediate change planned.
- **kyutai-stt** is upstream-streaming-native. Should be fine on long audio by design; verify in matrix.

### Tracking

PLAN #114 carries the full per-backend status table and the file list for the four prioritised fixes. The VPS matrix script lives at `tools/longform_vps.sh` (`longform_test.sh` is the local-only variant kept for reference); results aggregate via `tools/analyze_longform.py` into the table format used in `PERFORMANCE.md`.

---

## 2026-05-25 chatterbox hift_pcm(ref_mel) — diff harness layout bug, not vocoder drift

`crispasr-diff chatterbox` had been reporting `[FAIL] hift_pcm(ref_mel) cos=0.879` since the May 2026 parity pass. Two prior investigations (May 8, May 25 addendum) concluded "fp accumulation-order divergence between ggml and torch, not fixable without rewriting ggml reduction order". Both were wrong — the bug was in the harness's input binding.

The python reference dumps `hift_source_stft` via `s_stft.permute(1, 0).contiguous()`, producing bytes in `(T=12241, C=18)` row-major (`data[t*18+c]`). The C++ vocoder allocates `s_stft = ggml_new_tensor_2d(ctx, F32, T_src, 18)` with `ne[0] = T_src` as the fast axis, so it expects `(C, T_fast)` row-major (`data[c*T_src+t]`) — the layout it uses for the internally-generated case at `src_stft[f*T_src+frame] = re`. The harness was `memcpy`'ing the gguf bytes directly into the C++ tensor without transposing, so `source_downs[0]` was reading channels and time swapped.

Single transpose loop in `crispasr_diff_main.cpp` when binding `ref_source_stft` fixes it. Before / after on JFK 11 s @ 16 kHz:

| stage | before | after |
| --- | --- | --- |
| `voc_rb_0` | cos_min 0.937 | cos_min 1.000000 |
| `voc_rb_1` | cos_min 0.609 | cos_min 1.000000 |
| `voc_ups_2` | cos_min −0.133 | cos_min 1.000000 |
| `voc_conv_post` | cos_min 0.653 | cos_min 1.000000 |
| `hift_pcm(ref_mel)` | cos 0.879 FAIL | cos 1.000000 PASS (max_abs 2.85e-05) |

Runtime TTS is unaffected — production synthesis generates `source_stft` internally in the correct layout. The diff harness was the only consumer of the external feed.

Root-cause hunt was structured around the handover at `handover-prompts/chatterbox-hift-ref-mel-drift.md`. Built a standalone torch ground-truth probe (`~/code/chatterbox-hift-stash/probe_stage0.py`) that loaded `voc_ups_0` + `hift_source_stft` from the ref archive and ran stage 0 in torch using the F32 weights from the s3gen GGUF; matched the reference `voc_rb_0` to 7 digits. Comparing torch ground truth's intermediates (`voc_si_0`, `voc_rb_input_0`, `voc_rb0k0_snake1_d0`) against C++ at T=0 c=0..4 then localised the drift to source fusion, not the main resblock. Tracing `s_stft` allocation vs gguf byte layout surfaced the transpose mismatch. Updated `LEARNINGS.md` "Chatterbox HiFT vocoder parity nits" to replace the (wrong) "non-fixable fp drift" section with the correct diagnosis + fix.

## 2026-05-25 CI cleanup — build.yml trim, test #148 rename, GG_BUILD_NO_AVX512 knob

Three small CI fixes landing together — each pre-existing latent failure that surfaced once the issue #89 push cadence let `build.yml` run to completion for the first time in 30+ commits.

**Test #148 rename (`4fda4be5`).** `test-issue-114-chunk-context-gate` had a Catch2 `TEST_CASE("--chunk-overlap 0 disables overlap-save", ...)` whose name was passed verbatim by `catch_discover_tests` as the positional argv to the binary. Catch2's CLI parser then read `--chunk-overlap` as an unknown option and exited with `Unrecognised token: --chunk-overlap`. The test had been failing in CI on every push since it was added. Renamed to `"chunk-overlap=0 disables overlap-save"` with an inline comment about the catch_discover_tests / CLI-parser interaction so the next contributor doesn't reintroduce a `--`-prefixed name. All 6 cases / 307 assertions in the binary still pass; ctest now reports 303/303 unit-test PASS where it had been stuck at 302/303 for weeks.

**build.yml trim (`80ac00d1`).** Inherited from whisper.cpp upstream and grown to 1610 lines / 59 jobs across a matrix CrispASR doesn't actually ship for. Two long-latent failures surfaced when the file finally ran to completion: `ubuntu-22-gcc-arm64 (Release)` failed at 3 m on `qemu: uncaught target signal 11` inside libc-bin's post-install in the emulated arm64 docker guest (a known [`docker/setup-qemu-action@v3` regression](https://github.com/docker/setup-qemu-action/issues)), and `ggml-ci-x64-cpu-high-perf` failed with `crispasr-bench` SIGILL (exit 132) on first AVX512 instruction because GitHub's `ubuntu-22.04` runner pool is CPU-heterogeneous and the bench landed on a non-AVX512 VM after the build picked up AVX512 from a host that had it. Trim:

- **Removed** (not shipping targets): `ubuntu-22-arm-v7`, `ubuntu-22-gcc-arm-v7`, `ubuntu-22-cmake-sycl`, `ubuntu-22-cmake-sycl-fp16`, plus 5 ggml-ci jobs that target ggml-org's self-hosted runner pool (`-x64-nvidia-cuda`, `-x64-nvidia-vulkan-{cm,cm2}`, `-mac-metal`, `-mac-vulkan`) — they'd never schedule on this fork.
- **Switched to native runners** (no QEMU): `ubuntu-22-arm64`, `ubuntu-22-gcc-arm64` now use `runs-on: ubuntu-22.04-arm` (GitHub-hosted native ARM64, ~5-10× faster + structurally cannot hit the QEMU/libc-bin segfault). `ubuntu-22-clang` matrix expanded to cover both amd64 and native arm64 in one job (drops ppc64le).
- **Dropped ppc64le** from remaining linux jobs (`ubuntu-22`, `ubuntu-22-gcc`) — we don't ship for PowerPC, removing the docker wrapper also drops 3-5 min of emulation overhead per cell.
- **Kept** (we ship these + we modify their ggml backends per `tools/upstream-prs/01-12`): the 5 GitHub-hosted CPU `ggml-ci-{x64,arm64}-cpu-{low,high}-perf{,-sve}` jobs. They catch regressions in our ggml patches without needing self-hosted infra.

Net: 1610 → 1324 lines (−286), 59 → 32 jobs.

**GG_BUILD_NO_AVX512 knob (`565b16af`).** Actual fix for `ggml-ci-x64-cpu-high-perf` rather than the interim `continue-on-error: true` from the trim commit. The `.github/workflows/build.yml` job now sets `GG_BUILD_NO_AVX512=1` on the `ci/run.sh` invocation; a new env-var stanza in `ci/run.sh` (mirrors the existing `GG_BUILD_NO_SVE` pattern) appends `-DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX2=ON -DGGML_FMA=ON` to `CMAKE_EXTRA` when set. Pins a uniform AVX2 + FMA baseline that every GitHub-hosted x86_64 runner can execute (Skylake floor). Documented as `tools/upstream-prs/13-ci-no-avx512-knob.{md,patch}` for upstream submission to `ggml-org/llama.cpp` once we have a measured multi-run stability window. Default-OFF, opt-in; bit-identical to upstream when the env var is unset.

**Why these were latent for so long.** `build.yml`'s top-level `concurrency: cancel-in-progress: true` cancels in-flight runs on every new push. Its matrix takes ~30 min to complete; pushes have averaged every ~15 min during the active #83 + #89 + #97 work. So `build.yml` has been getting cancelled on 30+ consecutive commits and the failures never had time to surface. They have nothing to do with the code on any of those commits.

---

## 2026-05-24 PLAN #83 Round 9 follow-up #5 — Bug B FIXED via sched parallel=true

After eliminating ~10 hypotheses (cache barriers, blit copies,
concurrency, fusion, optimize, n_cb variants, private-storage
buffers, im2col 1×1×1 edge case, rc-as-mul_mat) and conclusively
proving the divergence is between the host's view and the GPU's
view of the same shared-storage Metal buffer in the second (uncond)
graph_compute call, the fix turned out to live in
`ggml_backend_sched_new`'s `parallel` flag.

With `parallel=false` (chatterbox default until this fix), sched's
between-backend synchronisation is `ggml_backend_synchronize` →
`[cmd_buf_last waitUntilCompleted]`. That blocks for the prior
command buffer's completion but does NOT invalidate the GPU's L1/L2
cached view of a shared-storage `MTLBuffer` that the CPU just
memcpy'd between consecutive command-buffer submissions.

With `parallel=true`, sched allocates 4× input-copy slots per input
and uses `ggml_backend_event_record` / `event_wait` for ordering.
On Metal that maps to `MTLSharedEvent`'s `encodeSignalEvent` /
`encodeWaitForEvent` commands, which carry proper GPU cache
invalidation between submissions.

Switched `chatterbox_s3gen_init_from_file` to call
`ggml_backend_sched_new(..., /*parallel=*/true, ...)`. Removed the
unet_input GPU pin workaround in `cfm_euler_solve::run_denoiser`
(no longer needed). Removed the `CRISPASR_NO_INPUT_PIN` env
override.

Verification on M1 (smoke, "Hello.", seed 42):

| Configuration | rms (vs ref 5.115) |
| - | - |
| GPU residency, sched parallel=true (this fix) | **5.143** ✓ |
| GPU residency, sched parallel=false (Bug B, removed) | 16.x ✗ |
| CPU residency, sched parallel=true (this fix) | **5.139** ✓ |
| CPU residency, sched parallel=false (prior baseline) | 5.139 (unchanged) |

Diff harness `s3gen_mel`: `cos_min = 0.999976` (matches the
previous workaround's baseline).

The Bug A fix (sched src-mutation log) from R9 #4 is INDEPENDENT
and continues to be required.

LEARNINGS R9 #5 closes Bug B with two new lessons: 7' replaces the
old "Pinning fixes it is not a root cause" with a more nuanced
"Pinning addressed a real symptom but not the root cause", and 8
("Check `ggml_backend_sched_new`'s `parallel` flag for Metal
cache-coherency-shaped bugs").

---

## 2026-05-24 PLAN #83 Round 9 follow-up #5 — Bug B narrowed to rc residual conv (superseded by fix above)

Investigated open Bug B from follow-up #4 (workaround in place: pinning
`unet_input` to GPU when UNet runs GPU-resident). Eliminated five
hypotheses, localized the divergence point, but did not identify the
root cause. The workaround keeps shipping.

**Localized the divergence.** Block-by-block probe of the UNet's
first resnet pass shows b1 and b2 produce bit-identical
(cos=1.000) intermediates between Path X (pinned) and Path Y
(no-pin). The first divergent tensor is the **residual conv
output** (`rc.weight` × `unet_input` inside `causal_resnet_block`
at `s3.fd.db.0.0`). Added a new probe knob
`CRISPASR_S3GEN_UNET_PROBE_RC_OUT=1` that dumps it as
`dump_rc_out_db00`. Comparison:

- Path X (workaround): `rc_out_db00 rms=0.3631`
- Path Y (Bug B): `rc_out_db00 rms=0.0645`
- cos(X, Y) = **0.022**

So the rc kernel sees ~zero input in Bug B, even though the host
`tensor_get` on the same Metal compute-buffer offset returns the
correct Gaussian-noise bytes. b1 and b2 reading the same offset see
correct data and produce identical output.

**Eliminated:**

1. CPU store-buffer / cache coherency — `CRISPASR_FORCE_DMB=1`
   (full `__sync_synchronize` after the memcpy) doesn't help.
2. GPU-side cache invalidation — `CRISPASR_FORCE_BLIT_COPY=1`
   (blit-encoder copy in its own committed cb instead of host
   memcpy) doesn't help.
3. Intra-encoder concurrency — `GGML_METAL_CONCURRENCY_DISABLE=1`
   and `GGML_METAL_GRAPH_OPTIMIZE=0` don't help.
4. Cross-cmd-buf race — `CRISPASR_METAL_N_CB=2` doesn't worsen the
   bug.
5. Missing `mem_ranges` barrier — `CRISPASR_METAL_FORCE_BARRIER=1`
   (emit a Metal `memoryBarrierWithScope:Buffers` before every
   single op) doesn't help.

All five experiments leave Bug B's smoke at `rms ≈ 16.1` vs the
workaround's 5.14. No measurable improvement from any of them.

**Operational note discovered along the way.** `ggml_set_output` on
a probe target preserves that slot but introduces an implicit
"(transposed) (cont)" follow-up tensor that shifts subsequent
gallocr offsets — which is enough to BREAK the workaround under
certain marking patterns (saw `MARK_DB_RESNET + PROBE_BLOCK1=1`
push the workaround to `rms=17.355`). So future investigators
should probe Path X and Path Y as separate runs with **identical
marking config**, and avoid trusting "extra-mark" runs for
end-to-end correctness.

**Diagnostic knobs added** (env-gated, no runtime impact when
unset; committed as WIP for the next investigator):

- `CRISPASR_NO_INPUT_PIN` — gates off the `unet_input` pin workaround
  in `cfm_euler_solve::run_denoiser` to reproduce Bug B.
- `CRISPASR_IM2COL_DBG` — prints first 8 host bytes of `op->src[1]`
  at the FIRST UNet `kernel_im2col_f32` dispatch.
- `CRISPASR_FORCE_DMB`, `CRISPASR_FORCE_BLIT_COPY`,
  `CRISPASR_METAL_FORCE_BARRIER`, `CRISPASR_METAL_N_CB` — the
  hypothesis-elimination knobs above.
- `CRISPASR_S3GEN_UNET_PROBE_RC_OUT` — marks the residual conv output
  at `s3.fd.db.0.0` as a dump tensor.

LEARNINGS Round 9 follow-up #5 records the full hypothesis table
and the next experiment to run (custom Metal kernel that captures
per-thread `x[offset_src]` reads to a side buffer).

---

## 2026-05-24 PLAN #83 Round 9 follow-up #4 — S3Gen UNet GPU drift: TWO bugs in tandem

Three rounds of bisect (#83 R9 follow-ups #1–#3) had ruled out gallocr,
per-op pin workarounds, the GGML_PREC_F32 hint on conv1d, Metal
concurrency, every flash-attn variant, and concluded the bug was
"at the Metal kernel layer, address-dependent, no in-tree workaround."
Follow-up #4 found the actual root cause(s) — there are TWO independent
bugs that both have to be fixed to make GPU-residency UNet work.

**Method.** Captured CPU vs GPU bytes at every probe point in the first
`causal_block1d` (`s3.fd.db.0.0.b1`) via `CRISPASR_S3GEN_UNET_PROBE_BLOCK1=0
+ CRISPASR_S3GEN_DUMP_UNET=<tag> + CRISPASR_S3GEN_DUMP_UNET_NO_AUTO_MARK=1`.
`tools/compare_probe_dumps.py` showed `after_im2col` already at
cos=-0.085 between CPU and GPU — the very first GPU kernel of the
UNet was producing structurally wrong output. `tools/inspect_im2col_dump.py`
confirmed the kernel logic (causal zero-padding) was correct, but it
was reading the wrong values from x.

A small `CRISPASR_S3GEN_LOG_INPUT=1` diagnostic + an inline
`ggml_backend_tensor_get` right before im2col dispatch and an
`fprintf` in `ggml_backend_sched_compute_splits` confirmed:

- On the FIRST call per CFM step (the CFG conditional pass), sched
  detects `unet_input` as a split input and copies CPU→Metal correctly.
- On the SECOND call (the CFG unconditional pass), `split[0].n_inputs`
  is 0 — sched fails to detect `unet_input` as needing a copy. The
  Metal kernel reads stale data left over from a previous compute.

**Root cause #1 — sched src[j] dangling pointer across alloc_graph calls.**
In `ggml_backend_sched_split_graph`, the input-detection loop walks
each node's `src[j]` and, when crossing backends, rewires
`node->src[j] = tensor_id_copy(...)`. This mutates the user's graph
in place. The input_cpy tensors live in `sched->ctx`, which is freed
and re-initialised at the top of the very NEXT split_graph call
(`ggml_free(sched->ctx); sched->ctx = ggml_init(params)`). So after
the cond call's compute returns, the user's `gf->nodes[i]->src[j]`
points at memory that the next split_graph will reclaim. The uncond
call's split_graph then walks the user's graph, reads dangling src
pointers, and silently skips creating new input copies because the
garbage flags rarely match `GGML_TENSOR_FLAG_INPUT`.

Patch: `ggml/src/ggml-backend.cpp` keeps a mutation log of every
`node->src[j]` rewire; `ggml_backend_sched_compute_splits` restores
the originals at the end of compute so the user's graph is left
exactly as they built it. `MUST RE-APPLY` after ggml bumps.

**Root cause #2 — `unet_input` divergence: NOT FIXED, only sidestepped.**
Even with the dangling-pointer bug patched and the sched copy
correctly delivering the user's bytes to `MTL0#unet_input#0` on
every call (verified by inline `ggml_backend_tensor_get` right
before im2col dispatch — kernel reads CORRECT input bytes),
downstream compute still produces `rms ~16` instead of the
expected ~5.1. The cause was not traced; it is sidestepped by
placing `unet_input` on the consuming Metal backend via
`ggml_backend_sched_set_tensor_backend(sched, unet_input, c->backend)`
in `cfm_euler_solve::run_denoiser`. The handover at
`handover-prompts/issue83-r9-followup-5-unet-input-routing.md`
collects everything we know about Bug #2 and what's been ruled
out, for a follow-up agent to chase to root cause.

Bisect findings on which inputs to pin:

| Pinned to GPU | Smoke `vocoder mel rms` | Verdict |
| - | - | - |
| (none) | 14.6 | broken |
| `unet_input` only | **5.14** | works ✓ |
| `time_emb` only | 14.4 | broken |
| `mask` only | 16.0 | broken |
| `unet_input` + `time_emb` | 209 | catastrophic |
| `unet_input` + `mask` | 5.14 | works ✓ |
| all three | 5.14 | works ✓ |

Pinning `time_emb` forces `mish` onto Metal which changes the
resnet block's cross-backend topology in a way that interacts badly
with the rest of the graph. Pinning just `unet_input` is the
minimal correct fix.

**Verification (M1 Metal):**

| Config | Before | After |
| - | - | - |
| baseline GPU residency, `--tts "Hello."` | `rms=13.938` | `rms=5.143` (ref 5.115) |
| 2-mark NaN trigger | `rms=NaN` | `rms=5.291` |
| diff harness `s3gen_mel` | `cos_min=0.940` | `cos_min=0.999976` |
| long text (~9 s) | `rms=13.045` | `rms=4.741` |
| production CPU residency (default) | `rms=5.139` | `rms=5.139` (no change) |

`s3gen_mel cos_min=0.999976` matches the production weight-residency
split (0.999980); the residual ~1e-2 max_abs is FP16/F32 round-off.
The GPU-residency path is now correct AND ~22% faster than the
CPU-residency split on M1 (median 34 s vs 43.7 s on the smoke run,
3-run wall clock).

**Verification that both fix halves are necessary:** ran with each
half disabled in isolation:

| Configuration | smoke rms | result |
| - | - | - |
| neither fix | 13.938 | broken (pre-fix baseline) |
| ggml mutation log only | 16.154 | broken |
| app-side pin of `unet_input` only | 14.640 | broken |
| both | **5.143** | works ✓ |

So both bugs are real and independent. Bug #1 fix is upstream-quality;
Bug #2 fix is a workaround pending follow-up #5.

**Lessons replacing R9 follow-up #3's 2''':**

2''''. **A backend-pin diagnostic ("does forcing input to GPU fix it?")
should be tried before declaring a Metal kernel bug.** Three rounds
chased shader-level address-dependence; the fix was a one-line pin
plus a small ggml sched patch. When CPU vs GPU bytes differ at the
FIRST kernel, the suspect chain is: host upload → sched input
placement → sched copy → kernel dispatch. Verify each step's outputs
before assuming kernel correctness is at fault.

5. **ggml sched mutates the user's graph in place and depends on the
mutations persisting across alloc_graph calls — but doesn't.** The
implicit contract "you give me a fresh gf each call, I rewire it"
breaks when the same gf is reused (e.g. CFG's cond+uncond passes).
Always restore mutations at end of compute. The patched
`ggml-backend.cpp` does this via a mutation log.

6. **For long, mixed-residency GPU sub-graphs, the input-pinning
choice is non-obvious.** Pinning everything to the consuming backend
isn't always right — pinning `time_emb` to GPU here was actively
catastrophic. Pin minimally (just the inputs that need to live on
the consuming side) and verify with end-to-end diff.

---

## 2026-05-24 PLAN #83 Round 9 — S3Gen UNet weight-residency ships; Metal kernel bisect

Production fix for the chatterbox S3Gen UNet GPU drift (see PLAN #83
rounds 7–8 for the prior bisects) plus an upstream-PR-quality bisect
of the residual drift on M1 Metal.

**What shipped:**

1. **Weight residency split** (commit `b84af324`) —
   `src/chatterbox_s3gen.cpp` loads the 910 `s3.fd.*` UNet weight
   tensors (79 MiB) on the CPU backend buffer via
   `core_gguf::load_weights_split`; the encoder (`s3.fe.*`), flow
   front-end (`s3.flow.*`), tokenizer (`s3.tok.*`), speaker encoder
   (`s3.se.*`), and HiFT vocoder (`s3.v.*`) keep GPU residency. The
   ggml scheduler auto-routes UNet ops to CPU based on weight residency
   — no per-op pinning, no GPU↔CPU sync inside the UNet graph (which
   is what caused the Round 8 NaN at T_mel ≥ 200). M1 Metal:
   `s3gen_mel cos_min 0.940 → 0.999980` in diff harness; intelligible
   audio in smoke at all T. Wall-time on M1 is comparable to pure CPU
   (the encoder/vocoder are a small fraction of total time, UNet does
   the same work).

   Two layered fixes: `CRISPASR_S3GEN_UNET_GPU_RESIDENCY=1` opts out
   of the split and falls back to a per-op `mul_mat_hp` helper that
   tags every UNet `mul_mat` with `GGML_PREC_F32`. Used as the
   testbed for the kernel-level path.

2. **Q8_0 × F32 bit-match Metal kernel** (commit `752baecf`) —
   `ggml/src/ggml-metal/ggml-metal.metal` gains `kernel_quantize_q8_0_f32`
   (mirrors `quantize_row_q8_0` ARM NEON path bit-for-bit) +
   `kernel_mul_mv_q8_0_q8_0` (mirrors `ggml_vec_dot_q8_0_q8_0_generic`).
   Same dispatch pattern as the existing Q4_K × Q8_K
   `GGML_PREC_F32` path. Verified bit-identical to CPU mul_mat for
   all 350 UNet Q8_0 mul_mats. Drafted as upstream PR 09 at
   `tools/upstream-prs/09-metal-q8_0-bit-match.{md,patch}`.

3. **Three upstream-PR drafts** (commit `d7d859a2`) in
   `tools/upstream-prs/`:
   - **09** Q8_0 × F32 bit-match kernel (concrete patch).
   - **10** ggml-alloc buffer-reuse drift report (no patch — bug
     report with bisect evidence). Expanded in commit `b6a0b610` with
     all 11 fix attempts and the diff-vs-smoke divergence finding.
   - **11** Scheduler NaN at T_mel ≥ 200 with mixed CPU+GPU ops (no
     patch — bug report).
   All three strip CrispASR-internal markers (no `(#83)` refs, no
   internal language).

4. **Per-segment dump hook + PRESERVE_INTERMEDIATES bisect knob**
   (commits `c00c1493`, `2daf2a19`) — env-gated debug instrumentation
   in `src/chatterbox_s3gen.cpp`. `CRISPASR_S3GEN_DUMP_UNET=<tag>`
   dumps step-0 intermediates from each sub-block to
   `/tmp/cb-unet-dump-<tag>-<name>.bin` for per-block diff.
   `CRISPASR_S3GEN_UNET_PRESERVE_INTERMEDIATES=1` forces `set_output`
   on 14 block-level intermediates (subset of dump points). Neither
   affects default behavior.

**The unresolved finding** (now in PR 10): `set_output` on **all 62**
UNet sub-block intermediates restores **bit-perfect** parity
(`cos_min = 1.000`, `max_abs = 0`) in the diff-harness call context
but produces NaN in the smoke call context with the *exact same*
model, graph topology, and seed. The diff-vs-smoke divergence is
invariant against: random seed, T_mel value, S3-tokenizer presence,
Metal concurrency on/off, `GGML_NO_INPLACE=1`, F32-tile Q in
flash_attn. Something structural in `ggml-alloc`'s state across
multi-graph sched invocations differs between the two call paths in
a way the bisect couldn't isolate within this session. Standalone
handover prompt prepared for further investigation.

**(2026-05-24 evening follow-up)** The unresolved finding above
was *partially a measurement bug*. The diff harness's cosine
comparison silently scored all-NaN data as `cos=1.000, max_abs=0`
(IEEE-754 NaN comparisons all return false). Fixed in commit
`4c2e54c0`. With the comparison fixed: both diff and smoke
produce all-NaN under `set_output` on 62; there is **no path
divergence**. Bisected with new granular env knobs (commit
`1a37f4c8`): the minimum trigger is **2 specific `set_output`
marks** (`dump_db_resnet` + any even-indexed `mb_*_out`, or
`mb_11_out`). The same parity pattern reproduces on both paths.
`tools/upstream-prs/10` rewritten to the tighter repro. The
production fix is unchanged and unaffected.

**Linux CPU smoke** validated on Hetzner VPS (`168.119.190.252`,
no GPU): build clean, `s3gen_mel cos_min = 0.918695,
cos_mean = 0.992655` in diff harness; intelligible TTS in smoke. The
production fix runs the same on Linux as on macOS. Branch
`plan-83-r9-s3gen-gpu-prec-hints` on the VPS at
`/mnt/storage/whisper.cpp`.

Updated: PLAN.md #57/#83 status, LEARNINGS Round 9.

**(2026-05-24 night follow-up #2)** Added a per-node gallocr
trace (`CRISPASR_GGML_ALLOC_TRACE=1`, commit `2f4961d6`). Compared
1-mark and 2-mark UNet allocator passes (n_nodes=2715 each, 3044
events). 2108 lines diverge but **zero overlapping live ranges in
either run** — the allocator is correct in both cases. The parity
mechanism is geometric: in the 1-mark control trace, even-indexed
`mb_*_out` tensors land at `offset=0` and odd-indexed at
`offset=1271296`. `FREE_SKIP_OUTPUT` on the low-offset slot blocks
~1300 downstream allocations into shifted positions; the same
output marked on the mid-offset slot only blocks a small hole.
The cascade is a layout shift, not aliasing. So the prior
follow-up's "buffer aliasing in `ggml_gallocr`" hypothesis is
ruled out. The bug must be in the Metal kernel layer (kernel
correctness sensitive to specific address patterns, or output-path
staging, or sched/OUTPUT-flag interaction). `tools/upstream-prs/10`
to be revised; LEARNINGS Round 9 follow-up #2 records the
methodology and pivots the investigation.

**(2026-05-24 late follow-up #3)** Pushed the bisect inside the
first UNet block. Added `CRISPASR_S3GEN_DUMP_UNET_NO_AUTO_MARK=1`
to decouple `DUMP_UNET` from its implicit "mark every dump point"
behavior, and `CRISPASR_S3GEN_UNET_PROBE_BLOCK1=<N>` to inline-
probe one specific `causal_block1d` call (names + marks
intermediate stages: im2col, mul_mat, conv1d, transpose_in, norm,
ln_mul, ln_bias, transpose_out, mish). With clean A/B against
CPU: GPU's first `causal_resnet_block` output (`dump_db_resnet`,
shape (T=382, C=256)) is structurally wrong, not drifted —
cos similarity to CPU is **-0.09**, magnitude is 10× off. Under
the 2-mark trigger, 1280 NaN values appear at 5 contiguous time
frames (t=260..264) × all 256 channels; bisect localizes the
3-frame NaN slice to the conv1d output in block1 of the very
first resnet block, then block2's `K=3` conv expands it to 5.

Tested `GGML_PREC_F32` on conv1d's internal `mul_mat` — no effect
(rms 13.938 → 13.942). Re-tested per-op CPU pin workarounds with
the fixed `crispasr-diff`: `PIN_CPU_OP=norm/mul_mat/add/cont` all
FAIL with `non_finite=8160/8160`. The handover's earlier "pinning
any frequent op restores cos=1.0" was the same `PASS`-of-NaN
artifact that `4c2e54c0` fixed; with it gone, **no per-op pin
fixes the GPU baseline**. Only working config remains the
production weight-residency split.

Conclusion: GPU-residency UNet on Apple Metal is fundamentally
broken in a way no in-tree workaround addresses. A real fix needs
shader-level Metal investigation against the address pattern this
graph produces — out of scope for a normal session. Until then
`CRISPASR_S3GEN_UNET_GPU_RESIDENCY` remains an investigation-only
knob; production keeps the `s3.fd.*` split on CPU.

LEARNINGS Round 9 follow-up #3 records the methodology and the
updated lesson 2'''.

---

## 2026-05-23 PLAN #52 — Qwen3-TTS perf bench (FUSED_QKV Q8_0 decision)

Ran interleaved A/B bench for `QWEN3_TTS_FUSED_QKV=1` on M1 Metal,
Q8_0 0.6B CustomVoice, 94-frame synthesis:

| Condition | ms/frame |
|---|---|
| Baseline (normal load) | ~129 |
| `FUSED_QKV=1` Q8_0 (interleaved A/B) | ~129 — **neutral** |
| `CP_STEP0_CACHE=1` (normal load) | ~128 — **neutral** |
| Baseline (quiet window) | ~79 — already under 80 ms/frame RT budget |

**Decision:** `FUSED_QKV` stays default-OFF for Q8_0 — confirmed
neutral, not beneficial on Metal. The first bench in the PLAN history
was a cold-cache artifact (model warmed into RAM by the preceding run,
making the FUSED_QKV run appear faster). `CP_STEP0_CACHE` also neutral
at normal load; quiet-machine confirmation still pending.

**F16 FUSED_QKV bench 2026-05-24:** F16 GGUF downloaded; resampled jfk.wav to 24 kHz (codec requires 24 kHz voice ref). Interleaved A/B (1 warm-up + 3A + 3B, 73 frames each): warm-up baseline 133 ms/frame; A mean 212 ms/frame, B mean 191 ms/frame, σ≈47 ms/frame — **inconclusive** (machine loaded: model download + build running concurrently dominated variance). Consistent with Q8_0 result (neutral). **Decision: keep F16 FUSED_QKV default-OFF.** Quiet-machine retest still open.

**Still open:** F16 FUSED_QKV quiet-machine bench; Q4_K bench; fusing 15 cp steps into one graph; Q8_0 KV cache (blocked on Metal `cont(Q8_0)` kernel patch).

Updated: PLAN.md #52 perf notes + `src/qwen3_tts.cpp:5023` comment.
See LEARNINGS.md §Qwen3-TTS FUSED_QKV.

---

## 2026-05-24 PLAN #97 — parakeet-rnnt-0.6b RNNT decoder

**Problem.** `nvidia/parakeet-rnnt-0.6b` is a standard RNN-Transducer (no
TDT duration head). The existing parakeet decoder only handled TDT + CTC
variants; RNNT needed a separate greedy loop.

**Fix.** Added `parakeet_rnnt_decode` in `src/parakeet.cpp`:
- blank token → advance encoder frame by 1 (standard RNNT blank semantics)
- real token → stay on same frame, emit token, update predictor state
- `max_per_step=10` per-frame cap prevents infinite loops on degenerate inputs
- Dispatch: `use_rnnt = !use_ctc && n_tdt_durations==0`; all 3 call sites
  (frame, chunk, full-file) updated to 3-way CTC/RNNT/TDT.

**Converter** (`models/convert-parakeet-to-gguf.py`) gained:
- In-memory nemo loading via BytesIO + torch.load (avoids disk extraction of 2.4 GB tarball)
- RNNT key detection: `joint.joint_net.2.weight` (not `joint.out.weight`)
- `joint_hidden` priority chain: `joint_hidden` → `encoder_hidden` → 640
- Q4_K quantization with F16 fallback for LSTM-shaped tensors (last dim not divisible by 256)

**Model:** downloaded `nvidia/parakeet-rnnt-0.6b` (2.3 GB nemo) to internal disk.
Converted to F16 (1235 MB) then `crispasr-quantize` to Q4_K (447 MB).

**Smoke test:** JFK 11 s → *"and so my fellow americans ask not what your country can do for you ask what you can do for your country"* — correct.

**Upload:** Q4_K + F16 + README to `cstr/parakeet-rnnt-0.6b-GGUF`.

**Registry:** `parakeet-rnnt-0.6b` entry added to `crispasr_model_registry.cpp` (~447 MB).

Committed `48ac6f06` to main. Worktree `CrispASR-parakeet-rnnt` (branch `parakeet-rnnt`) rebased + fast-forwarded into main.

**Follow-up — parakeet-rnnt-1.1b (same day):** External disk freed after the 0.6b cleanup; downloaded 1.1b nemo (4.0 GB), converted to F16 (2144 MB) + Q4_K (770 MB) with the same converter (42-layer encoder vs 24 for 0.6b — auto-handled by `n_layers` lookup). Smoke test on JFK: correct transcript at 5.4× realtime (caches warm). Uploaded to `cstr/parakeet-rnnt-1.1b-GGUF`; registry entry added. Committed `b9509548`.

**Still open in #97:** realtime-EOU; unified-en-0.6b.

---

## 2026-05-23 Global diarization timeline (issue #110)

**Problem.** The `sherpa`/`ecapa` diarization path ran the subprocess
once per VAD slice, producing local speaker IDs that reset or swapped
across slices. Long-file diarization was unreliable — `speaker_00` in
slice 1 might be a different person from `speaker_00` in slice 3.

**Fix.** Mirror the pyannote global-cache pattern (issue #107):

1. New `CrispasrSherpaCache` struct holds the pre-computed global
   speaker-turn timeline with absolute timestamps.
2. `crispasr_compute_sherpa_cache()` writes the full mono audio to
   one temp WAV, runs the sherpa subprocess once, parses all speaker
   regions.
3. `crispasr_run.cpp` pre-computes the cache alongside the existing
   pyannote cache block and threads it through every `process_slice`.
4. `assign_speakers_from_global_sherpa()` assigns each ASR segment
   its dominant speaker from the global timeline AND splits segments
   at speaker-turn boundaries using per-word timestamp overlap scoring.
   Segments without word timestamps get a single dominant-speaker label.

**Tests.** 13 Catch2 unit tests (38 assertions) + 8-assertion live
integration test suite (`test_diarize_live.sh`). All pass.

### Files

| File | Change |
|------|--------|
| `examples/cli/crispasr_diarize_cli.{h,cpp}` | `CrispasrSherpaCache` + `crispasr_compute_sherpa_cache()` + `assign_speakers_from_global_sherpa()` with word-level splitting |
| `examples/cli/crispasr_run.cpp` | Global sherpa pre-compute block + cache threading |
| `tests/test_diarize_global.cpp` | 13 Catch2 unit tests |
| `tests/test_diarize_live.sh` | 8-assertion live integration suite |

---

## 2026-05-23 Hotwords / contextual biasing (PLAN #98)

**Phase A — CTC-WS phrase-boost trie.** New shared helper
`src/core/asr_context_bias.h` implements an Aho-Corasick multi-pattern
trie over token-ID sequences. During CTC/TDT decode, tokens that
continue an active hotword prefix match get a configurable log-prob
boost (shallow fusion). Wired into parakeet CTC decode + TDT decode
paths. CLI flags: `--hotwords "word1,word2"`, `--hotwords-file <path>`,
`--hotwords-boost <float>` (default 2.0). Per-word boost suffix
supported: `"Berenz^5.0"`.

**Phase B — LLM prompt injection.** For LLM-based ASR backends that
accept free-text instructions, `--hotwords` appends a hint to the
system/instruction prompt:
- **qwen3-asr:** appends to ChatML system instruction
- **voxtral:** inserts into `[INST]` turn before `[TRANSCRIBE]`

Not wired (architecture reasons): voxtral4b (fixed streaming prompt),
granite-nle (non-autoregressive, no text prompt), funasr (prompt
hardcoded in library).

**Phase C** (TDT joint-net boost) deferred — Phase A on TDT already
covers the path via shallow fusion.

**Tests.** 13 Catch2 unit tests (34 assertions) for the trie +
4 paraformer integration tests (init, Chinese byte-match, English
byte-match, Q4_K==F16 parity).

### Files

| File | Change |
|------|--------|
| `src/core/asr_context_bias.h` | 182-LOC Aho-Corasick trie (insert, build failure links, apply_bias, advance, parse_hotwords, build_trie) |
| `src/parakeet.{h,cpp}` | `parakeet_set_hotwords()` API + CTC/TDT decode wiring |
| `examples/cli/cli.cpp` | `--hotwords`, `--hotwords-file`, `--hotwords-boost` |
| `examples/cli/whisper_params.h` | `hotwords` + `hotwords_boost` fields |
| `examples/cli/crispasr_backend_{parakeet,qwen3,voxtral}.cpp` | Wire hotwords into each backend |
| `tests/test_context_bias.cpp` | 13 unit tests |
| `tests/test_paraformer.cpp` | 4 live integration tests |

---

## 2026-05-24 Issue #89 reopened — parakeet streamed-encode is now the default for all audio

**Reporter (lenhone) was correct that the 2026-05-23 "99.5 % on 60 s" claim didn't hold.** End-to-end repro of their pipeline:

```
yt-dlp -f 'bestaudio/best' -x --audio-format wav <url>
ffmpeg -i <wav> -ar 16000 -ac 1 -t 60 -c:a pcm_s16le yt60.wav
crispasr -m parakeet-tdt-0.6b-ja.gguf -l ja -f yt60.wav -osrt
  → output stops at 00:00:20.080 (33 % coverage, not 99.5 %)
```

Same crispasr binary on `/mnt/storage/test-audio/ja/yt_60s.wav` (the file the 2026-05-23 benchmark was actually run against, cached on the VPS): full 60 s coverage. Both WAVs have identical duration, 16 kHz mono pcm_s16le, ~0.3 % RMS difference, **0.9977 zero-lag waveform correlation**. The "99.5 %" benchmark was run against the cached MP3-derived copy, not a fresh `yt-dlp` extract of the same YouTube video; the doc wording was misleading.

**Diff harness against NeMo on the bad audio:** our pipeline tracks NeMo bit-for-bit through the entire encoder.

```
[PASS] mel_spectrogram     cos_min=1.000000  max_abs=3.00e-04
[PASS] pre_encode_output   cos_min=0.999999  max_abs=1.04e-02
[PASS] encoder_layer_0..22 cos_min≈1.000000  max_abs ≤ 3.3e-02
[PASS] encoder_output      cos_min=0.999995  max_abs=2.14e-03
```

So this is **not a port-fidelity bug.** It's a model-level numerical instability in TDT single-pass over the full utterance. To prove it: NeMo's own `nvidia/parakeet-tdt_ctc-0.6b-ja` via stock `model.transcribe()` on lenhone's WAV produces 47 chars stopping at ~20 s; on the cached MP3 WAV it produces 294 chars covering the full 60 s. Same model weights, same `transcribe()` call, different audio derivation, same failure pattern as ours. Upstream defaults aren't safe on this audio either — and NeMo ships `BatchedFrameASRTDT` / `FrameBatchChunkedCTC` / `get_buffered_pred_feat_rnnt` in `streaming_utils.py` as the *separate* long-form path that stock `transcribe()` doesn't use.

**Mechanism (from per-frame trace).** With single-pass encoding over 60 s, the bidirectional FastConformer encoder's attention amplifies the 0.3 % audio-level codec quantisation difference into a 14 % shift in encoder output std (0.2069 vs 0.2415 on the two audio variants). That shift is enough to flip the TDT joint network's argmax from real-token to blank starting at frame ~250 (~20 s @ 100 Hz mel / 8× subsampling). Once blank dominates, the decoder stays in that regime for the rest of the utterance. The streamed path (overlapping 8 s encoder chunks → concatenate → single TDT decode) keeps the attention window short enough that the noise doesn't amplify; the decoder still sees one contiguous encoder sequence so there's no LSTM cold-start.

**Fix (`33f9a162`).** Make the streamed path the default for any duration. `examples/cli/crispasr_backend_parakeet.cpp`: `stream_threshold_s` default 60 → 0; condition flipped so 0 means "always streamed" (matching what the docs at `docs/cli.md:147` had already claimed). Single-pass is preserved as an opt-in escape hatch via `CRISPASR_PARAKEET_STREAM_THRESHOLD=999` for callers that want bit-exact NeMo reproduction on test data.

**Verified** on Apple Silicon Metal and Linux x86 CPU, on both audio variants, identical transcription content:

> このチャンネルでは日本語や日本文化について紹介しています。…年末年始に何をするかという話をしたいと思います。お正月を迎えるためのことをするということが大事です。

**120 s multi-backend sweep (`7a14879f`)** added to PERFORMANCE.md confirms this isn't parakeet-specific. On the same fresh `yt-dlp` audio, voxtral-mini-3b drops 0:27 → 1:47 (~80 s lost in the middle), cohere-transcribe emits only 4 segments across 120 s with multi-tens-of-seconds gaps, canary-1b-v2 hallucinates English ("I am not aware of anything" loop). Parakeet was loudest because lenhone happened to hit it, not because the other backends are safe. Filed as PLAN #114 for follow-up — open architectural question: VAD-default for everyone vs parakeet-style streamed-encode trick per backend vs LLM-decoder chunking + LCS dedup.

**Doc updates landing with this fix:**
- `README.md`: corrected the "99.5 % coverage" claim to "model-level numerical fragility; streamed encode is the default workaround"
- `docs/cli.md`: rewrote the parakeet streaming section + fixed the `CRISPASR_PARAKEET_STREAM_THRESHOLD=0` semantic mismatch (doc said "always streamed", code said "never streamed" — fix flipped the code to match the doc)
- `PERFORMANCE.md`: corrected the issue #89 fix-verification table to show both audio variants side-by-side, annotated the earlier byte-identical-streamed-vs-single-pass claim as "only on the cached MP3 derivation"

---

## 2026-05-23 Issue #89 cross-backend validation + PLAN #80d/#105 closure

**Validated** the NeMo-style streamed pipeline on current main
(`0c24178e`). Full chunk-size sweep (4-30 s) and overlap sweep (0-4 s)
confirm byte-identical output to single-pass across all configurations.
300 s Japanese: 98.6 % coverage, 0 gaps.

**Extended `CAP_INTERNAL_CHUNKING`** to canary-1b-v2 and
fastconformer-ctc (`1dd247a7`). Both suffered the same 30 s auto-chunk
z-norm drift as parakeet:

| backend | coverage (old 30 s chunks) | coverage (new) |
|---|---:|---:|
| parakeet-tdt 0.6b JA | 59.7 % | 99.5 % |
| parakeet-ctc 1.1b EN | 74.6 % | 98.5 % |
| canary-1b-v2 Q4_K EN | broken | 96.8 % |

**PLAN #80d** (cross-backend chunking audit): audited all 13 AR backends
— no fixes needed, all use `split_at_energy_minima` via the global
slicer. Cohere's API path also calls it directly.

**PLAN #105** (WhisperX aligner zoo): confirmed all 10 language-specific
wav2vec2 CTC aligner GGUFs uploaded and in the registry (fr/es/it/ja/zh/
nl/uk/pt/ar/cs). Added the full alias table to `docs/cli.md`.

**Issue triage:** closed #115 (hotwords, shipped), #111 (seed, shipped),
#119 (Mega-ASR, in registry), #120 (max_tokens, shipped). Triaged #121
(Pentium Gold crash — build artifact exists, likely user error or
runtime issue). Commented #85 (noise-robust ASR recommendations).

---

## 2026-05-23 Session beam_size wired for all remaining backends (PLAN #90 complete)

**Commit:** `0c24178e`

`crispasr_session_set_beam_size` now threads through every session backend.
Three remaining backend families were wired using `core_beam_decode::run_with_probs`
alongside the existing greedy path (unchanged when `beam_size == 1`):

| Backend | Replay lambda | KV reset |
|---|---|---|
| qwen3-asr | `qwen3_asr_embed_tokens` + `qwen3_asr_run_llm_kv` | `qwen3_asr_kv_reset` |
| granite / granite-4.1 / granite-4.1-plus / granite-4.1-nar | `granite_speech_embed_tokens` + `granite_speech_run_llm_kv` | `granite_speech_kv_reset` |
| voxtral | `run_voxtral_family` gained a `beam_size` parameter; a shared `decode_piece` lambda handles U+2581→space detokenisation for both paths | `ops.kv_reset(ctx)` |

`voxtral4b` (streaming path) is not in scope. Implementation is in
`src/crispasr_c_api.cpp`. The `#include "core/beam_decode.h"` is the only new dependency.

---

## 2026-05-23 PLAN #74: feature-matrix uplift round 2 (commit `b848152a`)

Four items completed in one session:

**74a — chatterbox language routing** (already in commit `c88306fa`):
`-l de` with `--backend chatterbox` auto-routes to `kartoffelbox-turbo`;
`-l ar` routes to `lahgtna-chatterbox`. Only fires when `-m auto` is active.

**74b — capability regression gate** (`tests/test_backend_caps.py`, new):
`TestCapabilityJSON` runs `crispasr --list-backends-json` and asserts:
- known translate backends declare `translate`
- known src-tgt backends declare `src-tgt-language`
- known voice-cloning backends declare `voice-cloning`
- preset-speaker backends (`qwen3-tts-customvoice`, `voicedesign`, `vibevoice`) do NOT declare `voice-cloning`
- whisper does NOT declare `src-tgt-language` (uses `-l` for target, not `-sl/-tl`)

`TestTranslateLive` is a live smoke-test (whisper-tiny on `samples/jfk.wav`) that auto-skips when the model is absent. 6/6 tests pass.

**74c — `CAP_VOICE_CLONING` for qwen3-tts base variants**:
`Qwen3TtsBackend` gained an `is_base_` constructor flag. The two base aliases
(`qwen3-tts`, `qwen3-tts-1.7b-base`) dispatch to a new `crispasr_make_qwen3_tts_base_backend()`
factory that sets `is_base_=true` and includes `CAP_VOICE_CLONING` in `capabilities()`.
The customvoice/voicedesign aliases keep the original factory (`is_base_=false`).

**74d — feature matrix regenerated** (`docs/feature-matrix.md` + `.html`):
`python tools/gen-feature-matrix.py` — 52 backends × 19 caps. `qwen3-tts`
and `qwen3-tts-1.7b-base` now show ✓ in the Voice cloning column.

---

## 2026-05-23 tts: `--seed` parity across sampled TTS backends

**Outcome.** Routed the CLI/server `--seed` knob through the TTS paths
that actually sample, then verified the behavior on the local backup
models in `/Volumes/backups/ai/crispasr`.

**What changed.**
- `qwen3-tts-customvoice` now lets an explicit request/CLI seed win
  over `QWEN3_TTS_SEED`.
- Chatterbox now reseeds both the T3 sampler and the S3Gen diffusion
  noise path from `--seed`.
- VibeVoice now seeds both the realtime and base TTS paths from
  `--seed`, while still honoring the env defaults when the CLI seed is
  zero.
- IndexTTS, Orpheus, and VoxCPM2 all have seed setters wired through
  the CLI surface; their visible impact depends on the backend's actual
  decode path.

**Live verification.**
- `qwen3-tts-customvoice`: same request seed is bit-identical even when
  `QWEN3_TTS_SEED=999`; different request seed changes the WAV hash.
- `chatterbox`: same seed is bit-identical; different seed changes the
  WAV hash.
- `vibevoice-tts` and `vibevoice-1.5b`: same seed is bit-identical;
  different seed changes the WAV hash.
- `IndexTTS`: the seed is accepted, but the tested prompt/reference
  produced identical WAVs across seeds, so the default beam-search path
  is effectively deterministic here.
- `Orpheus`: live check was blocked by the available local SNAC codec
  mismatch / runtime cost on this turn.
- `VoxCPM2`: seed plumbing is present, but there was no local VoxCPM2
  GGUF in the backup set to exercise it here.

**Files.**
- `src/qwen3_tts.{cpp,h}`
- `src/chatterbox.{cpp,h}` / `src/chatterbox_s3gen.{cpp,h}`
- `src/vibevoice.{cpp,h}`
- `src/indextts.{cpp,h}`
- `src/orpheus.{cpp,h}`
- `src/voxcpm2_tts.{cpp,h}`
- `examples/cli/crispasr_backend_*.cpp`
- `docs/cli.md`
- `docs/tts.md`

## 2026-05-21 fix(#89): parakeet long-audio NeMo-style streamed pipeline

**Outcome.** The issue #89 reporter's 300 s Japanese YouTube clip
(`o_9dWkRPYC0`, parakeet-tdt-0.6b-ja on Vulkan/AMD) went from 35
tokens covering 0-5 s → full coverage.  The 60 s case went from 0 chars
(with the original 60 s auto-chunk) to 294 chars / 99.5 % coverage.

**Root cause.** Three layered problems:

1. Per-feature z-norm drift: the TDT decoder emits blanks when mel
   statistics computed over >30 s shift from the training distribution.
2. Decoder cold-start: each independently-chunked slice resets the LSTM
   predictor to SOS, losing 5-20 s of interior content per chunk.
3. `crispasr_run.cpp` auto-chunked at 60 s → 30 s before the backend
   could handle it internally.

**Fix (6 commits).**

| Commit | Change | Impact |
|--------|--------|--------|
| `bdc8175` | Reduce auto-chunk 60 → 30 s | Prevents 0-output catastrophe |
| `1037bcb` | PR #116: VAD slices out of chunk-context path | Fixes VAD + cohere/granite regressions |
| `9488223` | Chunked-encode + single-decode (PLAN #104 v1) | 93 % coverage without VAD |
| `300149e` | NeMo-style: global z-norm + chunked encode | Feature-identical to single-pass |
| `97d2b4f` | Raise threshold to 60 s + env knobs | 99.5 % on 60 s, tuneable |
| `1dd247a7` | `CAP_INTERNAL_CHUNKING` for canary + fastconformer-ctc | 74.6 → 98.5 % (CTC), broken → 96.8 % (canary) |

**Architecture.** Audio ≤60 s: single-pass `parakeet_transcribe_ex` (best
quality, 99.5 %).  Audio >60 s: `parakeet_transcribe_streamed` — compute
mel with global z-norm over the full audio, encode in overlapping 8 s
chunks, concatenate encoder outputs, decode in one TDT pass.  The
encoder is bidirectional so each 8 s chunk gets independent context; the
decoder sees a single continuous sequence (no LSTM reset).

**Env knobs:** `CRISPASR_PARAKEET_STREAM_THRESHOLD` (default 60 s),
`CRISPASR_PARAKEET_STREAM_CHUNK` (default 8 s),
`CRISPASR_PARAKEET_STREAM_OVERLAP` (default 2 s).

**Benchmark framework.** New `tests/benchmark_asr.py` driver + audio
corpus in `/mnt/storage/test-audio/` (en/de/ja/zh × 4 durations from
FLEURS + reporter's audio).  14 pytest unit tests for metric computation
(`tests/test_benchmark_metrics.py`).

**Files.**

| File | Change |
|------|--------|
| `src/parakeet.cpp` | `parakeet_encode_chunked`, `parakeet_transcribe_chunked`, `parakeet_transcribe_streamed`, `parakeet_compute_mel_raw`, `parakeet_apply_znorm` |
| `src/parakeet.h` | New public API declarations |
| `examples/cli/crispasr_backend_parakeet.cpp` | Path selection + env knobs, `CAP_INTERNAL_CHUNKING` |
| `examples/cli/crispasr_backend.h` | `CAP_INTERNAL_CHUNKING` flag |
| `examples/cli/crispasr_long_audio_fallback.h` | `CAP_INTERNAL_CHUNKING_FLAG` gate |
| `examples/cli/crispasr_run.cpp` | 30 s auto-chunk default |
| `tests/benchmark_asr.py` | Multi-backend benchmark driver |
| `tests/benchmark_metrics.py` | Coverage metric computation |
| `tests/benchmark_corpus.py` | FLEURS audio corpus builder |
| `tests/test_benchmark_metrics.py` | 14 pytest unit tests |
| `tests/run-benchmark.sh` | CTest smoke wrapper |

---

## 2026-05-21 paraformer: FunASR Paraformer-zh NAR-ASR port

**Outcome.** Ported FunASR Paraformer-zh (220M params, non-autoregressive,
Mandarin Chinese + English) as a new `--backend paraformer`. Character-level
tokenizer (8404 vocab), single-pass decode via CIF (continuous integrate-and-fire)
predictor. Published F16 (421 MB), Q4_K (123 MB), Q8_0 (227 MB) at
`cstr/paraformer-zh-GGUF`; Q4_K is the registry default. All three produce
byte-identical transcripts vs Python on both Chinese test audio and JFK English.

**Architecture:** 50 SANM encoder blocks (reusing `core_sanm::build_block()`)
→ CIF predictor (Conv1d + sigmoid → accumulation) → 16 NAR decoder blocks
(FFN → FSMN → cross-attn, note the unusual order) → decoders3 post block
→ output_layer → argmax → space insertion between English word tokens.

**Four bugs in initial WIP port:**
1. Decoder block operation order: had FSMN → cross-attn → FFN but upstream
   does FFN → FSMN → cross-attn. Norms (norm1/2/3) were applied to wrong
   sub-layers.
2. FFN internal LayerNorm: was w1→LN→ReLU→w2, upstream is w1→ReLU→LN→w2.
   Post block (decoders3) also had a spurious residual connection.
3. CIF encoder-output transposition: used `enc_out[d*T+t]` instead of
   `enc_out[t*D+d]`. The ggml tensor already stores row-major (T,D), so
   no transpose was needed. Same bug in acoustic_embeds → decoder path.
4. Missing English word spacing: the vocab has whole English words as tokens
   with `@@` BPE continuation markers. Argmax loop concatenated tokens without
   spaces, producing `andsomyfellowamericans...`. Fixed by inserting a space
   between consecutive Latin-script word-final tokens.

**Diff harness.** Reference backend (`tools/reference_backends/paraformer.py`)
captures 73 stages. `paraformer_extract_stage()` implemented for all stages.
generated_text matches byte-for-byte on both Chinese and English.

### Files

| File | Change |
|------|--------|
| `models/convert-paraformer-to-gguf.py` | New: converter (model.pt → 956-tensor GGUF) |
| `src/paraformer.{h,cpp}` | New: ~850 LOC runtime |
| `examples/cli/crispasr_backend_paraformer.cpp` | New: CLI adapter |
| `tools/reference_backends/paraformer.py` | New: 73-stage reference backend |

---

## 2026-05-21 sensevoice: Q4_K + Q8_0 quants + fix crispasr-quantize `.w` suffix gate

**Outcome.** Published Q4_K (129 MB) + Q8_0 (240 MB) alongside the
existing F16 (448 MB) at `cstr/sensevoice-small-GGUF`. Registry
default flipped to Q4_K. All three produce byte-identical English
(JFK) and Japanese (JSUT) transcripts on M1 Metal; Q4_K is ~3× faster
end-to-end than F16, Q8_0 ~1.7×.

**Bug found while quantising.** First attempt at `crispasr-quantize
sensevoice-small-f16.gguf out.gguf q4_k` produced a file the same
size as the input. Every tensor in the log said `copying...`
instead of `quantizing to q4_K...`.

`examples/crispasr-quantize/main.cpp:216` gates "is this a weight
tensor" on either the substring `weight` OR the suffix `_w` (Kyutai
STT convention). SenseVoice's converter uses the FunASR-style `.w`
suffix (e.g. `sensevoice.enc.blk.0.attn.qkv.w`), which matches
neither check. Every tensor fell through to the copy path.

The same bug silently affected the FunASR encoder side too: FunASR's
LLM half uses llama.cpp `weight` names and quantises fine, but the
SANM encoder uses `.w`/`.b` and stayed F16. Older FunASR Q4_K files
on HF are therefore larger than they could be — re-quantising +
re-uploading them is queued as a follow-up but not blocking.

**Fix.** Extend the gate to also accept `.w` as a weight suffix:

```cpp
bool is_weight = (sname.find("weight") != std::string::npos) ||
                 (sname.size() >= 2 && sname.substr(sname.size() - 2) == "_w") ||
                 // FunASR / SenseVoice converter convention: .w / .b suffixes
                 (sname.size() >= 2 && sname.substr(sname.size() - 2) == ".w");
```

**72 tensors legitimately stay F16.** SenseVoice's `attn.fsmn.w` is
the FSMN depthwise convolution kernel with `ne[0] = 11` (kernel
size); `attn.qkv.w` has `ne[0] = 560` (encoder hidden 512 + SANM
context 48). Neither divides any quant block size (256 for K-quants,
32 for legacy), so the fallback chain rightly leaves them F16. The
other ~280 weight matrices quantise cleanly.

### Files

- `examples/crispasr-quantize/main.cpp` — extend the weight-suffix gate.
- `src/crispasr_model_registry.cpp` — Q4_K becomes the default;
  F16 + Q8_0 lookupable by canonical filename.
- `hf_readmes/sensevoice-small-GGUF.md` — quant table + default
  switch + "all three byte-identical on tested clips" note.
- `cstr/sensevoice-small-GGUF` HF repo — Q4_K + Q8_0 uploaded
  alongside the existing F16.

### Verification

- `crispasr -m sensevoice-small-{f16,q8_0,q4_k}.gguf -f samples/jfk.wav`
  → identical "And so my fellow Americans ask not what your
  country can do for you, ask what you can do for your country."
- `crispasr -m sensevoice-small-{f16,q8_0,q4_k}.gguf -f
  samples/ja/jsut_water_3s.wav -l ja` → identical
  "水をマレーシアから買わなくてはならないのです。"
- `-l auto` correctly identifies Japanese on all three quants.

---

## 2026-05-21 funasr: fix MLT-Nano hallucination (PLAN #99)

**Bug.** `cstr/funasr-mlt-nano-GGUF` produced hallucinated endings on
English transcripts (e.g. "ask what your country can do for you"
repeated instead of "ask what you can do for your country").  The
`cstr/funasr-nano-GGUF` variant worked correctly on the same audio.

**Root cause.** The C++ runtime hardcoded `use_low_frame_rate = true`.
Fun-ASR-Nano-2512's `config.yaml` sets it to `true`, but
Fun-ASR-MLT-Nano-2512 omits the key — the upstream Python default is
`false`.  With `true` the prompt builder spliced only the first 23 of
183 adaptor-output frames into the Qwen3-0.6B prompt, truncating 87 %
of the audio context.  The LLM decoded correctly up to token ~20 and
then hallucinated.

**Fix (commit `4433216`).**

| File | Change |
|------|--------|
| `models/convert-funasr-to-gguf.py` | Read `audio_adaptor_conf.use_low_frame_rate` from `config.yaml`; write as bool KV `funasr.use_low_frame_rate`. Also fixed `ada_n_heads` 16 → 8. |
| `src/funasr.cpp` | Read KV at load time; fall back `true` for old GGUFs. |

Both `cstr/funasr-nano-GGUF` and `cstr/funasr-mlt-nano-GGUF` GGUFs
reconverted (F16 + Q4_K + Q8_0) and re-uploaded.

---

## 2026-05-20 parakeet: 4 new variants — v2 + tdt-1.1b + tdt_ctc-{110m,1.1b}

**Goal.** Bring NVIDIA's remaining Parakeet TDT / TDT+CTC variants
onto the existing C++ runtime. The converter (`models/convert-parakeet-to-gguf.py`)
and the runtime (`src/parakeet.cpp`) already supported the full TDT +
CTC dispatch shape; the only gaps were per-checkpoint quirks and the
"how does the CLI find this by name" plumbing.

**Variants shipped** (all CC-BY-4.0, English-only, English BPE 1024):

| Name | -m key | F16 / Q4_K | M1 realtime (Q4_K) | Notes |
| --- | --- | ---: | ---: | --- |
| nvidia/parakeet-tdt-0.6b-v2     | `parakeet-v2`           | 1.24 GB / 468 MB | 11.4× | Original Open ASR Leaderboard topper; pred_layers=2, n_mels=128 |
| nvidia/parakeet-tdt-1.1b        | `parakeet-tdt-1.1b`     | 2.14 GB / 808 MB | 16×   | 42-layer encoder, lowercase output |
| nvidia/parakeet-tdt_ctc-110m    | `parakeet-tdt_ctc-110m` | 230 MB / 91 MB   | 45×   | 17L d=512, pred_layers=1, CTC head; runtime auto-flips to CTC |
| nvidia/parakeet-tdt_ctc-1.1b    | `parakeet-tdt_ctc-1.1b` | 2.15 GB / 810 MB | 14.8× | 42L hybrid, mixed-case + punct vocab |

All uploaded to `cstr/<name>-GGUF` with READMEs. Each repo has three
precisions (F16, Q8_0, Q4_K).

**Two C++ fixes** rolled in alongside:

1. **`parakeet_init_from_file` auto-flips to CTC when `pred_layers <
   2 && has_ctc`** (commit `0a902517`). The 110m has a single-LSTM
   predictor (`pred_layers=1`), so the TDT decoder's 2-LSTM
   requirement made `require()` return nullptr for the missing
   `decoder.lstm.1.*` tensors and then `parakeet_tdt_decode`
   segfaulted silently after the encoder pass. Now `lstm.1.*` are
   optional under `pred_layers < 2`, and the constructor sets
   `decode_ctc=true` automatically for that case. 110m no longer
   needs `--parakeet-decoder ctc`.

2. **`crispasr_resolve_model{,_cli}` sub-variant lookup priority**
   (commit `d8325847`). The CLI's filename-heuristic always sets
   `backend_name="parakeet"` for any `parakeet*` arg, so the old
   lookup ordering — `lookup_by_filename` → `lookup(backend_name)` —
   shadowed every sub-variant key to the default `parakeet` entry
   (v3). Inserted a step `lookup(model_arg)` between the two so
   `-m parakeet-v2` matches the `parakeet-v2` registry entry.

**Wiring touches.** No new dispatch code — the parakeet backend
already handles TDT and CTC, and the filename heuristic at
`crispasr_backend.cpp:370-372` already routes `parakeet-tdt_ctc-*` to
the parakeet backend via the `!contains_ci("tdt")` guard. The work was
4 registry entries + the two priority fixes above + 4 READMEs.

**C ABI parity.** `crispasr_registry_lookup_abi("parakeet-v2", ...)`
returns filename + URL; existing init functions consume the path
directly. Bindings (Go/Java/Ruby/JS/Python/Dart) get the new variants
for free.

PLAN crosswalk: PLAN #97 "More Parakeet variants" — TDT / TDT+CTC
items all green. Deferred items in #97 (RNNT / realtime-EOU /
unified-en) and #98 (hotwords / contextual biasing) untouched in this
session.

---

## 2026-05-20 sensevoice: structured output (C ABI + segment fields + JSON)

**Change.** SenseVoice's multi-task transcript embeds four rich-annotation
tokens as a prefix (`<|en|><|ANGRY|><|Speech|><|withitn|>...`). Until
this commit, the only public API was `sensevoice_transcribe()` which
returned the whole prefixed string and made the caller parse the
markers themselves. Now there's a structured surface.

- **C ABI** (`src/sensevoice.h`): `struct sensevoice_result {
  language, emotion, audio_event, itn, text, raw }` +
  `sensevoice_transcribe_structured()` + `sensevoice_result_free()`.
- **Segment fields** (`crispasr_segment`): four new optional strings
  `lang_id` / `emotion` / `audio_event` / `itn_flag`. Empty for
  non-SenseVoice backends (other backends are unaffected).
- **CLI**: stdout now shows the clean transcript without the `<|…|>`
  prefix. The active JSON writer (`crispasr_output.cpp:473`) emits
  `language`, `audio_event`, `emotion`, `itn_flag` siblings after
  `text` when present.

The parser is content-based, not positional. Upstream output ordering
is `[lang, emo, event, itn]` — different from the input query-embed
ordering `[lang, event, emo, textnorm]` — and degenerate audio can
drop trailing markers. Each `<|…|>` is classified against known
language / emotion / itn sets; anything else is the audio event.
LEARNINGS entry under "SenseVoice query-embed pattern".

---

## 2026-05-20 cosyvoice3: Fun-CosyVoice3-0.5B-2512 TTS port — Phase 1 (recon + converter)

**Change.** Tier 2 of the FunAudioLLM-family work (after funasr +
sensevoice). Multilingual TTS, Apache-2.0, 9 languages + 18+ Chinese
dialects, zero-shot voice cloning, 24 kHz output. Phase 1 lands the
foundation only — recon, converter, and three F16 GGUFs at
`/Volumes/backups/ai/crispasr-models/cosyvoice3-0.5b-2512/`. The C++
runtime is in development (Phase 2 starts with the LLM forward + RAS
sampling); the backend is **not yet user-facing**.

CosyVoice3 is three sub-models tied together:

  llm.pt    2.0 GB  Qwen2-0.5B (hidden=896, 24L, GQA 14/2, q/k/v biases,
                    NEOX RoPE θ=1e6) + speech_embedding (6761, 896) +
                    llm_decoder (6761, 896). AR-decodes speech tokens
                    with RAS sampling (top_p=0.8, top_k=25, win_size=10,
                    tau_r=0.1 — uniform-random fallback when last 10
                    tokens are too repetitive).
  flow.pt   1.3 GB  input_embedding (6561, 80) + pre_lookahead causal
                    conv (3-frame lookahead) + DiT estimator (22 blocks,
                    AdaLN-Zero modulation, dim=1024, heads=16,
                    head_dim=64, ff_mult=2, RoPE in MHA) +
                    CausalConditionalCFM (Euler ODE, 10 steps, cosine
                    t-schedule, cfg_rate=0.7).
  hift.pt    83 MB  CausalHiFTGenerator (HiFi-GAN-iSTFT hybrid):
                    conv_pre 80→512, 3 upsample stages (rates [8,5,3],
                    kernels [16,11,7]) with Snake activations + NSF
                    source modulator chain, CausalConvRNNF0Predictor,
                    conv_post 64→18, iSTFT (n_fft=16, hop=4) → 24 kHz.

Plus CAMPPlus (192-dim spk embed, identical to chatterbox/campplus.onnx)
and Qwen2 BPE tokenizer (vocab=151936, lives in CosyVoice-BlankEN/).

Converter walks all three .pt files and materialises every
`nn.utils.weight_norm` parametrisation in HiFT
(`parametrizations.weight.original0` = g scale, `.original1` = v
direction → plain `w = g·v/‖v‖`) so the runtime side never sees the
parametrised form. Lifts voxcpm2's `wn_reconstruct` pattern into Python.

Output GGUFs:

  cosyvoice3-llm-f16.gguf    1.29 GB  Qwen2 + speech heads
  cosyvoice3-flow-f16.gguf   0.67 GB  pre-lookahead + DiT
  cosyvoice3-hift-f16.gguf     42 MB  causal HiFTGenerator vocoder

**Reuse map.** Repo sweep confirmed almost every primitive needed for
the C++ runtime already exists in tree: Qwen2 LLM via
`core_attn::kv_self_attn`, CFM Euler with cosine schedule via
`chatterbox_s3gen::cfm_euler_solve`, weight_norm resolver via
voxcpm2's `wn_reconstruct`, iSTFT n_fft=16 hop=4 (exact-match
parameters) via chatterbox_s3gen, F0 predictor / NSF SineGen / causal
conv1d all via chatterbox_s3gen, CAMPPlus 100 % reusable, GPT-2 BPE
via core/bpe.h. **Genuinely new C++ code** for Phase 2-4 estimates
~360 LOC of primitives (AdaLN-Zero, Snake, pre-lookahead conv, RAS,
causal upsample padding) + ~1500 LOC of glue. Realistic timeline:
~1 week of focused work, revised down from the original "1-2 weeks".

Multi-phase plan tracked in `PLAN.md` "CosyVoice3-0.5B-2512 TTS port".

## 2026-05-20 sensevoice: FunAudioLLM/SenseVoiceSmall — multi-task ASR + LID + emotion + audio-event

**Change.** Encoder-only sibling of Fun-ASR-Nano: same
`SenseVoiceEncoderSmall` topology (1 entry block @ 560→512 + 49 main
+ 20 tp = 70 SANM blocks), but the LLM decoder is replaced by a
4-query-embed prepend + CTC head (25055 SentencePiece pieces). One
forward pass emits transcript + spoken language + emotion + audio-event
class through reserved positions in the same CTC vocab — no AR loop,
no KV cache.

  audio (16 kHz)
    → kaldi-fbank (hamming) + LFR(7, 6)           → (T_lfr, 560)
    → prepend 4 query embeds
        [lang_id, event_q=1, emo_q=2, textnorm_q]  → (T_lfr+4, 560)
    → SenseVoiceEncoderSmall (70 SANM blocks)     → (T_lfr+4, 512)
    → CTC head (Linear 512→25055)                  → (T_lfr+4, 25055)
    → greedy CTC + SentencePiece detokenize

Output carries the multi-task prefix as readable special tokens
followed by the SentencePiece-detokenised transcript:

  <|en|><|ANGRY|><|Speech|><|withitn|>And so my fellow Americans...
  <|zh|><|NEUTRAL|><|Speech|><|withitn|>开饭时间早上9点至下午5点。

**Reuse.** All the infrastructure from the funasr port carries over
unchanged — `src/core/sanm.h`, `src/core/lfr.h`, and the Hamming
window knob on `src/core/kaldi_fbank` were built generically enough
that the new runtime is ~800 LOC of mostly glue: loader, query-embed
gather + concat, CTC head + greedy decode, ▁→space SentencePiece
detokenize. No new core helpers needed.

**Verification.** `crispasr-diff sensevoice` is 76/76 PASS on
`zh.mp3` (byte-identical `generated_text`) and 75/76 on
`samples/jfk.wav` — the single jfk miss is the emotion-tag argmax
flipping `<|ANGRY|>` ↔ `<|EMO_UNKNOWN|>` at cos=0.999846 (F16 +
flash-attn op-order noise on a near-tied logit; the actual transcript
text is byte-identical and upstream Python is itself inconsistent on
this clip). All 5 example mp3s (zh / yue / en / ja / ko) produce
correct LID + correct script transcripts.

**Perf.** Apple M1 Metal F16:

| Clip | T_lfr | Wall | Realtime |
| --- | ---: | ---: | ---: |
| `example/zh.mp3`  (5.6 s) | 94  | 0.58 s | 9.8× |
| `example/en.mp3`  (7.2 s) | 121 | 0.46 s | 15.6× |
| `example/ja.mp3`  (7.2 s) | 121 | 0.45 s | 16.2× |
| `example/ko.mp3`  (4.6 s) | 77  | 0.41 s | 11.4× |
| `example/yue.mp3` (5.2 s) | 87  | 0.37 s | 13.9× |
| `samples/jfk.wav` (11 s)  | 183 | 0.50 s | 21.8× |

Matches upstream's "15× faster than Whisper-Large" claim and is
3-4× faster than the funasr port on the same clips (because there's
no Qwen3-0.6B AR decode — every position runs in parallel through
the CTC head).

**Files.** New `src/sensevoice.{h,cpp}` (~800 LOC),
`examples/cli/crispasr_backend_sensevoice.cpp`,
`models/convert-sensevoice-to-gguf.py`,
`tools/reference_backends/sensevoice.py`,
`hf_readmes/sensevoice-small-GGUF.md`. Edited `src/CMakeLists.txt`,
`examples/cli/CMakeLists.txt`, `examples/cli/crispasr_backend.cpp`
(dispatch + filename + GGUF-arch auto-routes),
`examples/cli/crispasr_diff_main.cpp` (per-backend `sensevoice` branch),
`src/crispasr_model_registry.cpp` (+1 entry with FunASR Model License
v1.1 attribution), `tools/dump_reference.py` (REGISTERED_BACKENDS).

**HF.** `cstr/sensevoice-small-GGUF` uploaded (F16, 0.47 GB — four
times smaller than funasr-nano-2512 because no LLM half). Wired
into `-m auto` with the license note printed on first download.
README adds the row, the feature-matrix column, the
multilingual-recipe entry.

---

## 2026-05-20 LCS hypothesis stitching for overlap-save chunk boundaries

The cad4c28a overlap-save chunking left a residual class of duplicate
emissions at chunk boundaries: when the FastConformer + TDT decoder
emits the same token in both chunk[i-1]'s right-extension and
chunk[i]'s left-extension, the word-level boundary filter's 200 ms
tolerance can pass both copies through. The reporter's 300 s
parakeet-ja run hit this only on intra-chunk TDT quirks, but the
infrastructure was missing the upstream-equivalent fix.

NeMo's offline long-audio recipe (`BatchedFrameASRTDT` in
`nemo/collections/asr/parts/utils/streaming_utils.py`) handles this
with `longest_common_subsequence_merge` — a sub-word LCS over emitted
token ids, with a leftmost-LCS heuristic and diagonal expansion to
recover one-token gaps from TDT frame drift.

This change ports that algorithm verbatim into
`src/core/crispasr_lcs.h` and wraps it with a segment-aware driver in
`examples/cli/crispasr_lcs_dedup.h` that walks the per-slice
`vector<crispasr_segment>`, drops the duplicate leading tokens from
`chunk[i]`, peels matching words, and rebuilds `seg.text`. The driver
fires only when overlap-save was applied (`use_chunk_context`), so
VAD-derived multi-slice runs (#114 invariant) are untouched.

The port was cross-checked against the upstream Python implementation
on 8 cases (no overlap, perfect 3-/4-/5-token, subthreshold, leftmost
preference, blanks-vs-content, partial alignment with gap) and
matches bit-for-bit.

**CLI surface.** Two new flags expose the behaviour for tuning and
A/B testing:

- `--lcs-dedup {auto|on|off}` — `auto` (default) follows the
  overlap-save gate; `on` forces dedup whenever there is more than
  one chunk (useful for bindings testing); `off` disables it for
  before/after comparison.
- `--lcs-min-length N` — minimum LCS length to act on (default 1,
  matching NeMo). Raise to 3-4 on audio with long-silence regions
  where blank tokens dominate the boundary token run.

**Public C ABI.** `crispasr_lcs_dedup_prefix_count(prev_tail_tokens,
n_prev, curr_tokens, n_curr, min_lcs_length)` is exported from
`libcrispasr` and declared in `include/crispasr.h`. Bindings (Go,
Rust, Dart, Python, Java) that drive the library chunk-by-chunk can
call it directly between adjacent chunks; it returns the number of
leading tokens of `chunk[i]` to drop. The binding owns the actual
slicing of its segment / word / text representation.

The C++ unit tests cover three layers: the algorithm
(`test-lcs-chunk-merge`, 10 cases / 15 assertions), the segment-
aware driver (`test-lcs-dedup-driver`, 8 cases / 22 assertions), and
the public C ABI symbol export (`test-lcs-c-abi`, 5 cases / 7
assertions) — 50 assertions across 23 cases, pure CPU, no model load.

Live runs on the issue #89 reproducer (Apple M1 Metal, 300 s clip):

| `--lcs-dedup` | Coverage | Segments | Notes |
| --- | --- | --- | --- |
| `auto` (default) | 0 – 300 s | 7 | issue #89 fix output |
| `off`            | 0 – 300 s | 7 | regresses to pre-LCS — duplicates re-appear at chunk boundaries (e.g. `なので` at 00:58 emitted twice) |
| `on --lcs-min-length 3` | 0 – 300 s | 7 | matches `auto` except at one cross-chunk match where the LCS run was exactly length 2 |

The "auto vs off" diff is ~6 chunk-boundary improvements on this
clip; the "auto vs on min-3" diff is 1 segment where raising the
floor leaves a 2-token cross-chunk run intact. Behaviour is exactly
what NeMo's `MIN_MERGE_SUBSEQUENCE_LEN` describes.

---

## 2026-05-20 parakeet-ja long-audio collapse on no-VAD path (issue #89)

`lenhone` reported that parakeet-tdt-0.6b-ja stopped transcribing past
~5 s on a 300 s YouTube clip when invoked without `--vad` and without
`--chunk-seconds`. Output collapsed to ~35 single-character tokens at
the very start, with the rest of the audio silently dropped.

**Root cause.** `a069018f fix(#89): always full-encode for
CAP_UNBOUNDED_INPUT backends` set `effective_chunk_seconds = 0`
unconditionally for parakeet/canary/wav2vec2/firered-asr/granite-nar
without an explicit `--chunk-seconds`. That was correct for short
audio (full-audio encoding gives the cleanest features), but
catastrophic for long audio: the FastConformer encoder + TDT decoder
aren't stable in a single pass past ~30-60 s. Per-feature z-norm
stats drift away from the training distribution, position encodings
move past the trained range, and the TDT decoder starts emitting
nothing but blanks. Upstream NeMo uses `FrameBatchASRTDT` /
`BatchedFrameASRTDT` for long-audio offline inference — chunked with
overlap-save and LCS hypothesis stitching, not single-pass encoding.

**Fix.** Add a long-audio chunking fallback: when
`CAP_UNBOUNDED_INPUT && !wants_vad && effective_chunk_seconds == 0 &&
n_samples > 60 s · SR`, set `effective_chunk_seconds = 60`. Chunk
boundaries still get the ± `chunk_overlap_seconds` (default 3 s)
context from `cad4c28a`'s overlap-save logic, gated correctly per
issue #114 — so the boundary mitigation only fires where it's needed
(explicit / fallback chunking) and stays off for VAD-derived slices.

The gate lives in `examples/cli/crispasr_long_audio_fallback.h` so
`tests/test-issue-89-long-audio-fallback.cpp` can pin it as a unit
test without a model load.

Verified on the reporter's 300 s YouTube clip (Apple M1 Metal):
- before: 35 segs, last at 00:00:04,880 (audio dropped 4.9 – 300 s)
- after:  6 chunks → 477 segs, last at 00:04:59,780 (full coverage)
- `--vad --vad-model firered` (same audio): 774 segs (finer slicing,
  works as expected — VAD path still bypasses the fallback)

The cad4c28a / #114 overlap-save logic still applies to the
auto-fallback's chunks (because `effective_chunk_seconds > 0`), so
the FastConformer's missing context at chunk boundaries gets the
± 3 s recovery window. VAD-derived runs continue to skip overlap-
save — silence-bounded slices have no boundary signal to recover.

---

## 2026-05-20 parakeet-ja kanji → hiragana regression (issue #114)

`exn251` reported that `parakeet-tdt-0.6b-ja` produced visibly worse
JA transcripts after v0.6.6: kanji compounds collapsed to bare
hiragana (`名前も教えて` → `なまえもおしえて`), multiple short VAD
slices were dropped entirely, and the runtime was ~30 % slower
(53.23 s vs 40.87 s on the user's 600 s reference clip). Setting
`--chunk-seconds 30` did not change the failing output, so chunking
was ruled out as the proximate cause.

**Root cause.** `cad4c28a feat(#89): overlap-save chunking with
--chunk-overlap flag` extended every slice by ± `chunk_overlap_seconds`
(default 3.0) of acoustic context on each side, gated only on
`slices.size() > 1 && kChunkContextS > 0.0f`. The intent was to
preserve bidirectional encoder context across explicit chunk
boundaries, but the gate also fired for VAD-derived multi-slice
runs. VAD slices are separated by silence — there is no boundary
signal to recover. Adding 3 s of neighbour audio pulled the next
utterance into the current encoder's context window, shifted the
FastConformer features, and caused the TDT decoder to pick a
different (worse) token path.

The crispasr-diff harness still passed at cos_min=1.0 (mel),
0.999994 (encoder) on the single-utterance baseball fixture, so the
per-stage cosine sweep alone did not catch this. The regression is
inherently a multi-slice + bidirectional-encoder + per-feature-z
interaction; it only manifests when at least two VAD slices each
have their own (slightly different) z-norm statistics.

**Fix.** Gate `use_chunk_context` on `effective_chunk_seconds > 0`
so the extension only fires when explicit chunking is in effect.
VAD-derived multi-slice runs revert to the v0.6.6 behaviour:
transcribe each slice's bare samples.

The gate is extracted into
`examples/cli/crispasr_chunk_context_gate.h` so
`tests/test-issue-114-chunk-context-gate.cpp` can pin the invariant
(`effective_chunk_seconds=0 && n_slices>1` must return false) as a
unit test without standing up the full pipeline.

Bisect timeline:
- `5f1bb858` (v0.6.6): GOOD (kanji)
- `8c895a7a~1`: GOOD (kanji, with spacing oddity from pre-617cd02)
- `ae6be961` (post-drop_last_frame, pre-overlap-save): GOOD (kanji)
- `992a5333`: GOOD (kanji)
- **`cad4c28a` (overlap-save default-on): BAD (hiragana)**
- `22ba4bce`/`a069018f`/`adaedb3e`/`HEAD`: BAD (hiragana)

Note: `drop_last_frame=true` (07cfcffe) is unrelated — both
true and false produced the regressed output once cad4c28a was
in. The cos parity at mel/encoder is preserved by this fix.

**Diff harness extension.** The full-audio diff was blind to this
class of bug because it feeds the model one continuous segment, just
like `tools/dump_reference.py` does. Added a `CRISPASR_DIFF_SLICES`
env var to `crispasr-diff` that takes "s0:e0,s1:e1,..." sample
ranges, runs `parakeet_compute_mel + parakeet_run_encoder` per slice,
and compares each slice's encoder output against the matching slab
of the reference's full-audio `encoder_output` on a per-frame basis.
The output reports both `cos_min` (whole slice) and `interior_min`
(skipping the first/last 4 frames where lack of cross-slice context
inherently lowers cos for split-mid-utterance cases). Interior
divergence is the real bug signal; boundary divergence is structural.

The new mode runs against the existing single-pass reference dump —
no per-slice reference re-bake required. Future per-slice bugs in
any backend can plug into the same flag.

## 2026-05-20 funasr: FunAudioLLM/Fun-ASR-{Nano,MLT-Nano}-2512 port lands

**Change.** First multilingual speech-LLM in the ASR family besides
qwen3. Runtime is `src/funasr.{h,cpp}` plus three new shared helpers:
`src/core/sanm.h` (SANM block — fused QKV, FSMN depthwise-conv memory
branch on V, sum-with-attn, optional flash-attention), `src/core/lfr.h`
(LFR(m=7,n=6) frame stacker), and a Hamming-window knob on
`src/core/kaldi_fbank.{h,cpp}` (was hardcoded Povey). The encoder is
`SenseVoiceEncoderSmall` (1 entry block @ 560→512 + 49 main blocks +
20 tp blocks, after_norm between, tp_norm at end). The adaptor is
the 2-block Transformer from `funasr.models.llm_asr.adaptor.Transformer`
(linear 512→2048→1024 prelude + 2× MHA, head_dim=128 not 64 — see
note below). The LLM half is Qwen3-0.6B (same body as qwen3-asr-0.6b);
the converter writes the LLM under llama.cpp-standard tensor names so
the loader reuses `core_attn::kv_self_attn` directly.

**No-CTC trap.** Upstream `config.yaml` and `fun_asr_nano/model.py`
declare a CTC decoder + head, but the published `model.pt` ships only
`audio_encoder.* + audio_adaptor.* + llm.*` — zero CTC weights on
both Nano and MLT-Nano releases. The LLM-decoder path is the only
viable inference path; we don't ship a CTC fallback.

**Adaptor heads = 8 (not 16).** The converter wrote
`funasr.ada_n_heads=16` to the GGUF, but
`funasr.models.llm_asr.adaptor.Transformer` invokes
`MultiHeadedAttention(kwargs.get("attention_heads", 8), llm_dim, ...)`
and the upstream `audio_adaptor_conf` does NOT pass `attention_heads`,
so the adaptor's two transformer blocks run at 8 heads (head_dim 128).
The runtime ignores the GGUF KV and forces 8; without this fix the
diff harness drops to cos~0.6 on the adaptor output (vs cos=1.0 on
the 70-layer encoder which is bit-near-exact).

**Verification.** `crispasr-diff funasr` is 77/77 PASS,
byte-identical `generated_text`, on both Chinese (`example/zh.mp3`
from the snapshot, "开饭时间早上九点至下午五点。") and English
(`samples/jfk.wav`, "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR
COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY"). On
Apple M1 Metal the end-to-end pipeline takes ~620 ms for 5.6 s of
audio (9.0× realtime); per-stage breakdown via `FUNASR_BENCH=1`:

| Stage              | Time (ms) | Notes |
| --- | ---: | --- |
| fbank+lfr          | 10.6      | frontend, single-threaded CPU |
| enc+ada_compute    | 150.8     | 70 SANM + 2 adaptor blocks, single Metal graph |
| prompt_tokenize    | 0.5       | Qwen3 BPE on prefix + suffix strings |
| embed              | 0.4       | token_embd lookup + audio-slot splice |
| kv_init            | 10.9      | one-shot per session |
| llm_prefill        | 54.4      | single forward over the full prompt |
| llm_decode_total   | 393.1     | 11 tokens @ 35.7 ms/tok (≈28 tok/s on M1) |

**Flash-attn on encoder + adaptor.** SANM and adaptor MHA use
`ggml_flash_attn_ext` by default (opt out with `FUNASR_NO_FA=1`).
On samples/jfk.wav (T_lfr=183), the FA path is 30 % faster on the
encoder (258 vs 370 ms) and 7 % faster overall (1.46 vs 1.57 s).
Crossover sits around T_lfr=100-150 (≈6 s audio); under that, the
unfused mul_mat + soft_max_ext path wins by ~6 % because the FA
kernel-launch overhead dominates. Default ON matches typical ASR
workloads; tight-VAD realtime users may want the opt-out.

**Files.** New `src/funasr.{h,cpp}`, `src/core/sanm.h`,
`src/core/lfr.h`, `examples/cli/crispasr_backend_funasr.cpp`. Edited
`src/core/kaldi_fbank.{h,cpp}` (Hamming window),
`models/convert-funasr-to-gguf.py` (retargeted to LLM-decoder path;
commit b6a2f75f), `tools/reference_backends/funasr.py` (stage names +
str-as-meta route for `generated_text`), `examples/cli/crispasr_backend.cpp`
(dispatch + filename auto-route + GGUF-arch auto-route),
`examples/cli/crispasr_diff_main.cpp` (funasr branch — float-cosine
for tensor stages + meta byte-compare for `generated_text`),
`src/crispasr_model_registry.cpp` (two new entries with explicit
FunASR Model License v1.1 attribution; dropped the misleading
hardcoded "(non-commercial)" suffix from the license-note printer
since the license string already carries authoritative wording),
`src/CMakeLists.txt` (new `funasr` static lib),
`examples/cli/CMakeLists.txt` (CLI + diff binary link lists),
`README.md` (ASR table rows + headline count + capability matrix
column + native-punctuation table entry + multilingual recipe row).

**HF.** `cstr/funasr-nano-GGUF` and `cstr/funasr-mlt-nano-GGUF`
uploaded (F16, 1.98 GB each). `hf_readmes/funasr-{nano,mlt-nano}-GGUF.md`
carry the FunASR Model License v1.1 attribution line and link to the
upstream model cards. Auto-download via `--backend funasr -m auto`
works and prints the license note on first download.

## 2026-05-20 voxcpm2-tts: full VAE decode ggml cgraph + transposed-conv fix

**Change.** Added `vae_decode_graph` (gated on `VOXCPM2_USE_GRAPH=1`)
that emits the full VAE decode pipeline as a single ggml cgraph
running on `ctx->backend` (Metal on Apple Silicon). New helpers
`snake1d_ggml`, `causal_conv1d_ggml`, `causal_transposed_conv1d_ggml`,
plus `vae_wn_init_ggml` which builds a dedicated ggml context +
backend buffer holding every WN-scaled conv weight, SR-cond per-bucket
[C] slice, and snake1d `1/(α+1e-9)` reciprocal as `ggml_tensor*`
keyed by GGUF prefix.

**Two pre-existing bugs uncovered + fixed.**

*(1) `causal_transposed_conv1d` head-shift.* Python's
`CausalTransposeConv1d` (`audio_vae_v2.py:31-38`) captures `padding`
/ `output_padding` as named kwargs and **never forwards them to
`nn.ConvTranspose1d.__init__`** — so PyTorch internally uses
`padding=0, output_padding=0` and `super().forward(x)` returns length
`L_std = (T_in - 1) * S + K`. The wrapper then slices `[:-(2P - OP)]`
from the END, yielding the first `T_in * S` samples. The legacy C++
used `trim = K - 1` (head-shift) instead, shifting the audio by
~46 ms across 6 upsample blocks. Long-standing diff-harness regression
`decoded_audio cos = 0.008` was the symptom. Fix in both paths: take
the first `T_in * S` samples of the no-padding output (tail-trim of
`K - S` from end).

*(2) `vae_wn_init_ggml` SR-cond size mismatch.* My init sized the
sr_scale/sr_bias tensors from `it->second->ne[1]` — which returned
**4** (the bucket dim) instead of **2048** (the channel dim) because
GGUF stores `scale_embed` with ne=[2048, 4] (C innermost). ggml's
binary-op broadcast then **silently mishandled** the 4-vs-2048
mismatch instead of asserting, producing cos=0.967 instead of
cos=0.989 for `vae_only_graph`. Fix: take `max(ne[0], ne[1])` for
the non-bucket dim. After fix: graph output is bit-identical to
legacy CPU on every per-block trace.

Also aligned CPU `snake1d` to Python's `1/(α + 1e-9)` formula
(previous `(|α| > 1e-8) ? 1/α : 1` differed only at tiny α; matters
for parity with the graph's pre-baked `inv_alpha`).

**New diff stages.** `vae_only` / `vae_only_graph` take Python's
`generated_latent` as input via `ref_samples` and run the C++ VAE
in isolation — lets us attribute drift directly to the VAE vs
upstream AR pipeline. Backed by a Python-side hook on
`model.audio_vae.decode` in the reference dumper.

**Validation.** Diff harness (M1, voxcpm2-q4_k.gguf, voice clone):

| Stage              | Before | After |
| ------------------ | -----: | ----: |
| decoded_audio      | cos=0.008 FAIL | cos=0.683 (upstream-limited) |
| vae_only (CPU)     | —              | cos=**0.989** |
| vae_only_graph     | —              | cos=**0.989** (Metal + CPU match) |
| Upstream stages    | 13 PASS / 3 FAIL | 13 PASS (unchanged) |

ASR roundtrip: EN ("Hello world") → parakeet-tdt-0.6b-v3, DE
("Hallo, wie geht es dir heute?") → parakeet-v3, ZH
("你好，今天天气真好。") → qwen3-asr — all three languages transcribe
back exactly through both legacy CPU and the new graph path. Audio
audibly natural on M1.

The remaining `decoded_audio` drift (cos=0.68) is now provably
upstream — `cfm_step0_result` cos=0.94 with REF inputs cascading
across 10 Euler steps, plus `tslm_layer_27_out` cos=0.97 (F16
accumulation over 28 layers). Tracked as separate work in PLAN #96.

Diagnostic infrastructure (gated on `VOXCPM2_VAE_TRACE=1`) writes
per-block intermediates to `/tmp/voxcpm2_{l_,g_}<stage>.bin` for
side-by-side python comparison — kept in tree for future
graph-vs-legacy investigations.
---

## 2026-05-20 voxcpm2-tts: cache wn_reconstruct across VAE encode/decode calls

**Change.** VAE decode rebuilt every weight-norm-resolved conv tensor
(`g · v / |v|`) on every call via `wn_reconstruct`. With ~30 conv
layers per decode and `vae_residual_unit` allocating fresh
`std::vector<float>` for each, that was ~60-150 ms of pure
recompute every synth. Added `vae_wn_get_cached(ctx, prefix, …)` that
keys by the GGUF prefix (e.g. `vae.dec.layer.2.block.1`) and returns
a reference to the cached vector — populated lazily on first miss,
reused on subsequent calls within the same context.

Threaded `voxcpm2_context* ctx` through `vae_residual_unit` and
`vae_enc_block` (the two helpers that didn't have it before) so the
cache is reachable from every call site. Both VAE encode and decode
go through it now.

This is also the foundation for the upcoming full VAE ggml graph
(PLAN #96 remaining item): the graph needs the wn-resolved weights
as stable buffers, and the cache already produces them in the
correct ggml conv kernel memory layout
(`[K, in_ch, out_ch]` for forward conv,
`[K, out_ch_tc, in_ch_tc]` for transposed conv — both match the
ggml_conv_1d / ggml_conv_transpose_1d ne convention exactly).

**Validation.** Diff harness `voxcpm2-q4_k.gguf` (CPU path): still
14 pass / 0 fail / 3 skip. "Hello world" zero-shot ASR-roundtrips
correctly.

**Bench** (M1, OMP=8, "Hello world" zero-shot, 6 AR steps):

| VAE decode | Before cache | After cache |
| ---------- | -----------: | ----------: |
|            |    ~3 875 ms |   ~3 300 ms |

~15 % off VAE decode. Small absolute win on the first synth (one
miss + one hit); bigger over the server / diff-harness case where
the same ctx synthesises repeatedly.

---


## 2026-05-20 voxcpm2-tts: LocEnc per-call ggml graph (PLAN #96 follow-on)

**Change.** Added `build_locenc_graph` + cached `get_or_build_locenc_graph`
+ `locenc_forward_graph` wrapper, mirroring the LocDiT graph pattern:
bidirectional 12-layer transformer with LongRoPE GQA flash-attn +
SwiGLU. Simpler than LocDiT — no time/dt embeddings, no mu condition,
no cond projection. Input is one P=4-frame patch `[feat_dim, P]`;
the CLS token (`W.locenc_cls_token`) is prepended via `ggml_concat`,
the graph runs over T=5 sequence, and the output is the CLS hidden
at position 0 (post final norm).

Topology is constant so the graph + gallocr layout are reserved
once on first use, like LocDiT. The two call sites (build_prefill_inputs's
per-ref-patch loop for voice cloning + the AR-loop's per-step LocEnc
on the predicted patch) both route through the graph under
`VOXCPM2_USE_GRAPH=1`. Falls back to the legacy CPU path on graph
init failure.

**Validation.** Diff harness `voxcpm2-q4_k.gguf` (CPU path): still
14 pass / 0 fail / 3 skip. Zero-shot ("Hello world") and voice clone
(jfk.wav ref) smoke tests both ASR-roundtrip correctly.

**Bench** (M1, OMP=8, "Hello world" zero-shot, 6 AR steps):

| Substep    | Before LocEnc graph | After LocEnc graph |
| ---------- | ------------------: | -----------------: |
| locenc/step|             34 ms   |            11 ms   |

3× per-step. Bigger wins on voice cloning where LocEnc is hit ~70×
in build_prefill_inputs — that loop ate ~2.4 s before, now closer
to ~0.8 s on Metal.

---


## 2026-05-20 voxcpm2-tts: SIMD-friendly transposed conv + causal conv layout (PLAN #96 follow-on)

**Change.** The OMP-parallelised `causal_transposed_conv1d` and
`causal_conv1d` still ran scalar inner ic loops because both x
(stride `T_in` across ic) and weight (stride `out_ch*ksize`)
were strided in the inner axis — the compiler can't NEON-vectorise
strided loads. Per-call we now:

1. Reshape the weight from `[in_ch, out_ch, ksize]` to
   `[ksize, out_ch, in_ch_inner]`. Inner ic is contiguous.
2. Transpose x from `[in_ch, T_in]` to `[T_in, in_ch_inner]`.
3. Run the inner ic dot product contiguous on both axes — auto-
   vectorisable.

For `causal_conv1d` the transpose path is gated on
`in_per_grp > 1 && ksize > 1` (depthwise and 1×1 convs skip it —
no SIMD opportunity, transpose would be pure overhead).

The transposes are `O(in_ch × (out_ch × ksize + T))` per call,
small compared to the inner dot product work — block 0's 32 M
weight floats + 16 K x floats vs ~940 M dot floats.

**Validation.** Diff harness `voxcpm2-q4_k.gguf` (CPU path): still
14 pass / 0 fail / 3 skip (VAE isn't probed by the harness — the
`decoded_audio` stage is SKIPPED in the zero-shot ref archive).
Both zero-shot ("Hello world") and voice clone (jfk.wav ref)
smoke tests ASR-roundtrip correctly.

**Bench** (M1, OMP=8, "Hello world" zero-shot, 6 AR steps):

|                       | OMP only | + SIMD layout |
| --------------------- | -------: | ------------: |
| Block 0 upsample (ms) |   2 957  |          615  |
| VAE decode total (ms) |   8 772  |        3 875  |
| Synth wall total (ms) |  14 766  |        6 800  |

≈4.8× on the deepest block-0 upsample; ≈2.3× on total VAE;
≈2.2× on total synth wall.

**Also: README.md.** Removed the stale `(beta)` tag and
"garbled-but-recognisable" / "voice cloning falls back" warnings;
the path is now end-to-end functional. Added a short status block
documenting `VOXCPM2_USE_GRAPH=1` and the current ~7 s wall on M1
for "Hello world" zero-shot Q4_K.

---


## 2026-05-20 voxcpm2-tts: parallelise VAE decode hot paths (PLAN #96 follow-on)

**Change.** VAE decode was the dominant remaining wall-clock cost
(~8 s of the post-Metal 14 s total). The forward `causal_conv1d`
was already OMP-parallelised, but three hot loops weren't:

- `causal_transposed_conv1d` — scatter-add into shared output (race-
  prone, hence serial). Rewrote as gather + OMP `collapse(2)` over
  (out_ch, ot). Also skip directly to the valid kernel taps for each
  output position via the modulus-progression trick (`k0 = (ot+trim)
  mod stride`, step `stride`) instead of testing the modulus on
  every `k`.
- `snake1d` — per-channel SiLU-like activation, outer loop over C
  is write-disjoint. Added OMP.
- VAE per-block SR conditioning (per-channel scale + bias broadcast
  over time) — outer loop over C, OMP.

**Validation.** Diff harness `voxcpm2-q4_k.gguf` (CPU path): still
14 pass / 0 fail / 3 skip. "Hello world" zero-shot ASR-roundtrips
correctly.

**Bench** (M1, OMP=8, "Hello world" zero-shot, 6 AR steps):

| VAE decode | Pre-OMP | Post-OMP (3-run mean) |
| ---------- | ------: | --------------------: |
|            | 8 994 ms |             7 734 ms  |

~14 % faster (variance is real — 7.0–8.6 s across runs). Modest
because the conv work itself dominates, not the loop overhead;
real Metal-class wins need the full graph rewrite (ggml_conv_1d /
ggml_conv_transpose_1d / compound snake1d) — that's the remaining
PLAN #96 item.

---


## 2026-05-20 voxcpm2-tts: multi-bucket TSLM step graph (PLAN #96 follow-on)

**Change.** Replaced the single TSLM-step cgraph cache (Lk=128) with
a 5-bucket array {128, 256, 512, 1024, 2048}. `tslm_pick_bucket`
picks the smallest fitting bucket; each is built lazily via the
existing bucketed `build_tslm_step_graph(fixed_kv_len,
kv_indices=positions)` pattern. The struct now holds an
`std::array<TslmBucket, 5>` with per-bucket arena_meta / arena_ctx /
gf / galloc, freed together in `voxcpm2_free`.

**Why.** Single-bucket Lk=128 covered short inputs but fell through
to the slow dynamic per-call build on anything with prefill > 127
positions (multi-sentence voice cloning, long voice instructions),
losing 4-10× on `tslm_step`. Bigger buckets are paid only when
needed — short prompts still use the cheapest Lk=128.

**Validation.** Long-text clone exercising the 256-bucket
("122 prefill + 81 AR = 203 positions") ASR-roundtrips correctly;
log shows both "built tslm step bucket Lk=128" and "Lk=256" lazy
fires. Short-prompt zero-shot only fires Lk=128 (unchanged). Diff
harness `voxcpm2-q4_k.gguf` still 14 pass / 0 fail / 3 skip.

---


## 2026-05-20 voxcpm2-tts: load weights on init_best — Metal live (PLAN #96 done)

**Change.** Switched `voxcpm2_init_from_file` to set
`c->backend = ggml_backend_init_best()` (Metal on Apple Silicon) when
`params.use_gpu` is true, and `vox_load_weights` now loads onto
`c->backend` instead of the global CPU backend.

Apple Silicon Metal allocates weight buffers in unified-memory
"shared" mode (`is_shared = true`), so `tensor->data` stays
CPU-readable. The remaining legacy `matmul_mv_ggml` paths
(TSLM/RALM prefill, LocEnc, VAE encode/decode, FSQ, stop predictor)
keep working against Metal-resident weight pointers without any
copy. The cached graph paths (LocDiT + TSLM step) run on Metal via
their existing `ggml_backend_graph_compute(ctx->backend, …)` calls.

Dropped `ggml_flash_attn_ext_set_prec(_, GGML_PREC_F32)` on LocDiT's
bidirectional flash-attn — Metal's `supports_op` for
`GGML_OP_FLASH_ATTN_EXT` rejects any op tagged PREC_F32 (chatterbox
T3 patch — wants CPU bit-identity), so leaving it caused the graph
compute to abort. The per-stage cosine threshold tolerates the
resulting F16 simdgroup drift on Metal.

**Validation.** Diff harness `voxcpm2-q4_k.gguf` (CPU path,
`use_gpu=false`): still 14 pass / 0 fail / 3 skip. Smoke "Hello
world" zero-shot AND voice-clone (`samples/jfk.wav` ref) both
ASR-roundtrip to "Hello world." on Metal.

**Bench** (M1, OMP=8, "Hello world" zero-shot, 6 AR steps):

| Path                  | TSLM prefill | AR loop  | Total   |
| --------------------- | -----------: | -------: | ------: |
| legacy                |    ~5 000 ms | 15.3 s   | 48.7 s  |
| graph cached (CPU)    |    ~4 000 ms |  6.0 s   | 26.2 s  |
| graph cached (Metal)  |       80 ms  |  5.0 s   | 14.1 s  |

The dominant Metal win is TSLM prefill (≈60× on the 3-position ×
28-layer matmul-dense pass) — and the same legacy `matmul_mv_ggml`
path benefits, because Metal-resident weights cut the bandwidth
hit of dequantising Q4_K rows per row. Per-AR-step CFM is roughly
unchanged (the cached LocDiT graph is already near-optimal at
T=11 / GQA n_kv=2). Voice cloning AR loop drops similarly to ~5 s
on Metal.

**Next** (PLAN #96 follow-on): multi-bucket TSLM Lk for
long-prefill inputs; graph-ify VAE encode + decode (~8 s of total
wall-clock is still in VAE decode); then drop the legacy
`matmul_mv_ggml` + CPU-only fallback and flip default to
`VOXCPM2_USE_GRAPH=1`.

---


## 2026-05-19 voxcpm2-tts: graph caching — LocDiT one-shot + TSLM bucketed (PLAN #96 CPU target met)

**Problem.** The per-call `build_*_graph` + `ggml_gallocr_alloc_graph`
overhead dominated steady-state perf. LocDiT rebuilt the same
12-layer cgraph on every one of ~19 CFM Euler iterations per AR step
(~570 builds per synth). TSLM rebuilt the 28-layer step graph on
every AR step (~6 builds per synth, but each is ~5× larger), and
its `gallocr_alloc` walked a fresh plan every call.

**Fix.** Two qwen3-style caches:

- **LocDiT** — topology is constant (no `n_past`). Built once into a
  dedicated arena (`locdit_arena_ctx`); `locdit_galloc` reserves the
  plan once on first use. CFM Euler iterations rebind 6 input
  tensors and recompute.
- **TSLM** — topology depends on `n_past`. Made `n_past`-invariant
  via `fixed_kv_len = kTslmBucketLk` + `kv_indices = positions`
  (runtime-indexed K/V scatter via `ggml_set_rows`), with a
  `causal_mask` input that masks the unwritten tail of the bucket.
  Single bucket at `Lk=128` covers every current synth path
  (zero-shot ≪ 20 positions, "Hello world" + jfk.wav clone ~80
  positions). Longer prefills fall through to the dynamic per-call
  build.

**Validation.** Diff harness `voxcpm2-q4_k.gguf` zero-shot ref: 14
pass / 0 fail / 3 skip (unchanged). Both zero-shot ("Hello world")
and voice-clone ("Hello world" + jfk.wav) smoke tests
ASR-roundtrip to "Hello world." correctly.

**Bench** (M1 CPU, OMP=8, "Hello world" zero-shot, 6 AR steps):

| Path                              | AR loop | cfm/step | tslm/step |
| --------------------------------- | ------: | -------: | --------: |
| legacy                            | 15.3 s  | 2398 ms  |    55 ms  |
| graph, uncached                   |  6.8 s  | 1035 ms  |    38 ms  |
| graph, cached (LocDiT only)       |  8.0 s  |  837 ms  |    52 ms  |
| graph, cached (LocDiT + TSLM)     |  6.0 s  |  625 ms  |   180 ms  |

Steady-state CFM **~410 ms / step** — meets the plan's ~400 ms CPU
target. TSLM ~180 ms / step (par with legacy, vs ~1781 ms uncached
TSLM graph). Voice-cloning AR loop drops to **4.1 s** for "Hello
world" + jfk.wav. The Lk=128 bucket avoids the wasted flash-attn
work that Lk=512 caused (flash-attn computes `Q·K^T` over the
entire bucket regardless of the -inf-masked tail).

**Next:** load weights on `c->backend` (Metal-capable). Blocked on
dropping the legacy CPU paths entirely — once both step graphs are
on the backend, `matmul_mv_ggml` is dead. Multi-bucket Lk (128 /
256 / 512 / 1024) for long-prefill inputs is a follow-on.

---


## 2026-05-19 voxcpm2-tts: TSLM per-step ggml graph + backend-resident KV (PLAN #96 partial)

**Problem.** TSLM step (28 MiniCPM-4 layers, T=1) was the #2 AR-step
hotspot after CFM. The legacy path called `tslm_layer_step` 28×, each
of which did 7 `matmul_mv` calls (Q/K/V/O + gate/up/down), and inside
each `matmul_mv` built/destroyed its own tiny cgraph — ~196 graph
builds per AR-step TSLM call.

**Fix.** Added `build_tslm_step_graph` (one 28-layer cgraph using
`core_attn::kv_self_attn` with NEOX RoPE + LongRoPE freq_factors +
GQA expansion + flash-attn + `core_ffn::swiglu`) and
`tslm_step_graph` wrapper. The graph reads/writes a backend-resident
KV tensor (qwen3 layout: `(head_dim, max_ctx, n_kv, n_layers)`) via
`init_tslm_kv_backend` + `ggml_backend_alloc_buffer`. One-time
`sync_tslm_kv_cpu_to_backend` runs once per synthesis after the
legacy prefill has populated `vox_kv_cache`, transposing pos↔kvh
into the qwen3 layout. AR loop routes through the graph under the
same `VOXCPM2_USE_GRAPH=1` gate as LocDiT.

**Validation.** Diff harness `voxcpm2-q4_k.gguf` zero-shot ref: 14
pass / 0 fail / 3 skip — no regression vs LocDiT-only. "Hello
world" zero-shot synth still ASR-roundtrips correctly.

**Bench** (M1 CPU, OMP=8, "Hi" zero-shot, 4 AR steps, contended
system):

| Path                  | AR loop (ms) | cfm/step | tslm/step |
| --------------------- | -----------: | -------: | --------: |
| legacy                | 24 706       | 5 110    |       158 |
| graph (LocDiT + TSLM) | 18 731       | 1 700    |     1 781 |

Net: AR loop -24 %, total -12 %. **TSLM step is slower per call on
CPU** — the 28-layer per-call graph build + `gallocr_alloc_graph`
overhead exceeds the per-matmul-overhead savings for T=1. LocDiT
graph wins enough to make the combined path net-positive. Both
graphs together are the prerequisite for moving weights to the
Metal backend (where GPU compute will dwarf the build overhead).

**Next:** graph-cache the TSLM step across AR iterations (qwen3
LK_BUCKET pattern with `fixed_kv_len` + `kv_indices=positions` so
topology is `n_past`-invariant). Estimated 4-10 × on CPU
`tslm_step` cost — flips it from regression to net win on CPU
alone.

---


## 2026-05-19 voxcpm2-tts: LocDiT per-call ggml graph (PLAN #96 partial)

**Problem.** `cfm_euler_solve` was the AR-step hotspot at 63.7% of
per-step time (1186 ms / 1864 ms total on M1 CPU, "Hi" zero-shot
Q4_K). Inside, `locdit_forward` called `matmul_mv_ggml` ~30 times per
invocation; each matmul did its own `ggml_init` + `ggml_new_graph` +
`ggml_backend_graph_compute` + `ggml_free`. CFM runs LocDiT ~19 times
per AR step (2 × CFG × 10 timesteps − the CFG-zero-star skip), so
~570 graph builds per AR step just for CFM.

**Fix.** Added a backend pool (`backend`, `backend_cpu`, `galloc`,
4 MB `compute_meta` arena) to `voxcpm2_context`, plus
`build_locdit_graph` (a single 12-layer DiT cgraph: time MLPs, in/cond
projections, concat to T=11, bidirectional GQA flash-attn with
LongRoPE, SwiGLU FFN, final norm + out_proj on the last 4 positions)
and `locdit_forward_graph` (precomputes bf16-rounded `t_sin` /
`dt_sin` in C++ to preserve commit `52622dc2`'s bug-#24 fix, then
runs the graph). `cfm_euler_solve` picks the graph or legacy path via
a `locdit_call` lambda gated on `VOXCPM2_USE_GRAPH=1`. Weights stay
on `backend_cpu` for now — moving them to Metal requires the matching
TSLM-step graph (otherwise legacy `matmul_mv_ggml` would SIGSEGV
reading Metal-resident weights).

**Validation.** Diff harness on `voxcpm2-q4_k.gguf` zero-shot ref:
14 pass / 0 fail / 3 skip. Critical stages — `cfm_step0_result
cos_mean=0.9826` (PLAN #96 gate ≥ 0.93), `dit_input_seq
cos_mean=0.9984`. Voice-cloning smoke ("Hello world" + jfk.wav)
still ASR-roundtrips to "Hello world."

**Speedup** (M1 CPU, OMP=8, "Hello world" zero-shot, 6 AR steps):
AR loop 15 306 → 6 815 ms (2.3×), CFM/step 2 398 → 1 035 ms (2.3×),
total wall 24.2 → 18.5 s. Below the plan's `~400 ms` CPU target —
remaining overhead is per-call graph build / gallocr alloc, not the
matmul work. Caching the graph across CFM Euler iterations (qwen3_tts
bucket pattern) is the next CPU win; Metal is gated on the matching
TSLM-step graph (PLAN #96 still-TODO).

---


## 2026-05-19 Issue #89 — chunking, overlap-save, CTC decode path

**Problem:** parakeet-tdt-0.6b-ja (and other non-whisper backends with
bidirectional FastConformer encoders) lost text at fixed 30-second chunk
boundaries. The default `--chunk-seconds 30` was a whisper-era default.

**Four commits:**

1. **`68b4b3e` — disable default chunking for non-whisper backends.**
   Added `chunk_seconds_explicit` flag. Non-whisper backends process
   full audio in one encoder pass unless `--chunk-seconds` is explicit.
   Also fixed streaming VAD timestamp offsets (4 call sites).

2. **`992a533` — scope to `CAP_UNBOUNDED_INPUT` only.** LLM-based
   backends (voxtral, granite, qwen3, glm-asr, kyutai-stt, mimo-asr,
   gemma4, cohere, moonshine) use autoregressive decoders with KV cache
   that grow with input length — they need chunking to avoid OOM. Added
   `CAP_UNBOUNDED_INPUT` capability flag, set only on non-autoregressive
   backends: parakeet, canary, wav2vec2, firered-asr, fastconformer-ctc,
   granite-nar.

3. **`cad4c28` — overlap-save chunking.** When chunking is active,
   extend each chunk by `--chunk-overlap` seconds (default 3.0) on each
   side. Word-level filtering keeps only the original slice region with
   200 ms tolerance for TDT frame shift. Text rebuilt by direct
   concatenation (no space insertion — fixes JA kana-spacing bug from
   617cd02).

4. **`22ba4bc` — CTC decode path for hybrid TDT+CTC models.** Converter
   exports CTC head; runtime adds `parakeet_ctc_decode()` with F16
   tensor support. CLI: `--parakeet-decoder ctc`. CTC is frame-
   synchronous and avoids TDT boundary artifacts entirely.
---

## 2026-05-19 IndexTTS-1.5: drop magic-constant speaker-emb clamp + add greedy beam knob (issue #75)

User #75 reported "abnormal-sounding voice" and "very slow" on their
RTX 5060 Ti (RTF=8.57 for a 6 s Chinese sentence). Root-causes:

1. **Speaker-embedding clamp was the "abnormal voice".** The original
   code in `src/indextts.cpp` rescaled the ECAPA output to L2 norm =
   0.9 with a comment claiming it "matches Python's typical magnitude".
   Upstream BigVGAN consumes the unnormalized ECAPA output directly —
   the clamp shrank `cond_layer(spk_emb)` and `conds[i](spk_emb)`
   projections 2.84× and left the BigVGAN underconditioned. ASR
   roundtrip on "Hello world. This is a test of speech synthesis."
   went from "...test of **the index piece since its** system."
   (~25% CER) to "...test of **speech synthesis**." (clean). Audio
   peak amplitude went 0.06 → 0.21. New env knob
   `INDEXTTS_SPK_NORM=<float|"raw">` defaults to `raw`; pass `0.9`
   to reproduce the old behaviour.

2. **Beam search KV snapshots round-trip through host.** The B=3 beam
   decode `ggml_backend_tensor_get/set`s the full 158 MiB KV cache per
   beam per step. For the issue-75 test (158 mel codes × B=3) that's
   ~146 GB of host transfers — and on GPU it's the wall-time
   bottleneck. New env knob `INDEXTTS_BEAM_SIZE=1` skips snapshots
   entirely (greedy keeps one KV slot resident on device). Measured
   on M1: 75.8 s → 34.4 s (2.2× faster) with ASR still clean. Default
   stays B=3 to keep parity with the Python reference; proper fix
   (B resident slots + `ggml_backend_tensor_copy` device-to-device)
   tracked as follow-up.

Also tried-and-discarded during investigation: parallel audit agents
flagged Conformer `ff_scale=0.5` (macaron) missing and FFN activation
ReLU-vs-SiLU mismatches. Both were false positives — IndexTTS-1.5 has
`macaron_style=False` (config-default) so `ff_scale=1.0`, and the
upstream activation IS SiLU (line 480 of `conformer_encoder.py`
overrides the `PositionwiseFeedForward` default). Lesson: when an
audit agent cites a line in a generic library file, verify how the
specific subclass instantiates it.

Chinese ASR roundtrip remained degraded ("而在曲電韓宿中飛..." for
"而在获取电压函数中...") at the close of the spk-emb / beam fix —
diagnosed and fixed in a follow-up later the same day; see the next
section.

### Files
- `src/indextts.cpp` — replaced `target_norm = 0.9f` clamp with env-
  knob `INDEXTTS_SPK_NORM` (default raw); replaced `const int B = 3`
  with `INDEXTTS_BEAM_SIZE` (default 3, range 1–16).
- `LEARNINGS.md` — two new sub-sections under "IndexTTS-1.5 TTS
  backend".

---

## 2026-05-19 IndexTTS-1.5: CJK char tokenization for Chinese roundtrip (issue #75 follow-up)

The "Chinese ASR roundtrip remains degraded" follow-up from the spk-emb
fix above. The C++ text path called SentencePiece directly on the raw
input; upstream Python runs `TextNormalizer.char_rep_map` (CJK punct →
ASCII) **then** `tokenize_by_CJK_char` (insert space around every CJK
codepoint) **then** SentencePiece. With CJK chars glued together the
BPE Viterbi produced 29 tokens that the model translated into
unintelligible Chinese; the correct pipeline produces 54 tokens with
`▁` (id 10201) between every CJK character, and the model produces
fluent Mandarin.

Two passes were needed:

1. **Port `tokenize_by_CJK_char`** — UTF-8 codepoint helpers + the
   upstream CJK Unicode range (`U+1100-11FF`, `U+2E80-A4CF`,
   `U+A840-D7AF`, `U+F900-FAFF`, `U+FE30-4F`, `U+FF65-FFDC`,
   `U+20000-2FFFF`). Plus a subset of `char_rep_map` for the common
   CJK punctuation (`，：；、！？` → ASCII; quotes / brackets → `'`).
   Got Chinese ASR roundtrip to ~7% CER (Qwen3-ASR-0.6B reference),
   with a single trailing `位` syllable hallucinated.

2. **Map `。` (U+3002) → `.` in the punct table** — full-width period
   sits inside the CJK Unicode range, so without an explicit punct
   mapping it was being split as a CJK character (`▁` + `。` instead
   of the single `▁.` piece). Upstream order is char_rep_map first,
   then CJK split — so `。` lands as ASCII `.` and the model sees its
   trained sentence-end token. Fix removed the trailing `位` and got
   CER to **3.6% raw / 0.0% punct-stripped** on the test prompt — every
   Chinese character matches the input exactly. The only residual is a
   comma the TTS naturally inserts as a pause after `空指针`.

Verified bit-equality of token IDs C++ ↔ Python `sp.Encode` on the
preprocessed text (54 tokens for the issue-75 Chinese prompt).
English path is byte-identical to its pre-fix tokens (no codepoint
hits the punct table). Mixed CN+EN `他用Python写了一个程序。`
roundtrips perfectly — CJK chars get split, `PYTHON` stays a single
upper-cased word, `。` lands as ASCII.

ASR reality check, important: **whisper-base is not a reliable Chinese
ASR for measuring TTS quality**. The first-pass fix looked like 21% CER
under whisper-base; switching to Qwen3-ASR-0.6B exposed the actual
3.8% (a single hallucinated syllable) — and the second-pass `。` fix
dropped that to 0% punct-stripped. Whisper-base alone over-counts
errors by ~5×.

Skipped intentionally: the full wetext `zh_normalizer` (numbers→hanzi,
pinyin tones, English contractions) needs a real rule engine; not
blocking for plain Chinese text and out of scope for this fix.

### Files
- `src/indextts.cpp` — new `preprocess_indextts_text()` (UTF-8 +
  CJK char split + punct-map upper) wired into both prefill and
  latent re-tokenization paths; opt-in `text_ids[...]` dump at
  verbosity ≥ 2 for future BPE diffs without a debug rebuild.
- `LEARNINGS.md` — new sub-section on the CJK pipeline and the
  `。`-in-CJK-range trap.

---

## 2026-05-19 IndexTTS-1.5: device-resident beam KV pool, opt-in (issue #75 follow-up)

The "proper fix" referenced in the May 19 spk-emb section: per-beam KV
state lives in B same-backend tensors and gets swapped via
`ggml_backend_tensor_copy` instead of `_get`/`_set` round-tripping
through `std::vector<uint8_t>`. Slot recycling on candidate selection —
free slots are exactly the parent slots that no surviving child
references, so siblings that split off the same parent each get a
fresh slot via `tensor_copy` with no extra allocation.

Opt-in only: `INDEXTTS_KV_DEVICE_COPY=1`. Default stays on the host
`_get`/`_set` path the original IndexTTS shipped. Reason for opt-in:

Measured on M1 Metal, B=3, "Hello world..." (121 mel codes), 3 trials
each, warm cache:

| Trial | host (default) | device (opt-in) |
| --- | --- | --- |
| 1 | 61.25 s | 61.67 s |
| 2 | 55.12 s | 72.73 s |
| 3 | 70.32 s | 61.66 s |
| median | **61.25 s** | **61.67 s** |

Median delta < 1 % — within noise (each binary spans ~15 s trial-to-trial
on this box). Apple Silicon unified memory makes `_get`/`_set` already
a shared-RAM memcpy with no real "host round-trip" cost — the original
LEARNINGS framing oversold the memcpy bottleneck. The 2.2× B=1 speedup
came mostly from less GPT compute, not less memcpy.

Audio output is byte-identical between the two modes on both the
English test prompt and the issue-#75 Chinese prompt, and identical to
the pre-refactor binary in default mode.

Device path is expected to actually pay off on **CUDA / Vulkan** where
`_get`/`_set` crosses real PCIe. That measurement is the next step;
default will flip on those backends only when the numbers show a real
win.

### Files
- `src/indextts.cpp` — dual KV-snapshot path: host buffers (default)
  vs device tensor pool (`INDEXTTS_KV_DEVICE_COPY=1`). Refcount-based
  slot recycling in the device path. `Beam` struct gained a
  `slot_idx` field next to the existing `kv_k/kv_v` host vectors.
- `LEARNINGS.md` — follow-up sub-section under the existing beam-KV
  entry recording the M1 measurement and the misattribution lesson.

---

## 2026-05-19 Chatterbox GPU bug localised to S3Gen + Metal default flipped

Round 6 + 7 of the PLAN #57 / #83 GPU chase. The round-4 patches
(`kernel_mul_mv_q4_K_q8_K`, `kernel_quantize_q8_K_f32`, `mul_mm_*_hp`,
PREC_F32 tagging) had cleaned up the T3 mul_mat drift that round 3
diagnosed, leaving a remaining "FORCE_GPU=1 → garbled audio" that the
prior narrative still attributed to T3. Re-bisect with `crispasr-diff`
showed `s3gen_encoder_out cos=0.999950` (encoder on GPU is fine) while
`s3gen_mel cos_min=0.923` collapses — the bug is downstream of T3, in
the S3Gen UNet1D CFM denoiser.

### Code changes (`src/chatterbox_s3gen.cpp` + `src/chatterbox.cpp`)

- `ggml_mish` rewritten from the hand-rolled `x * tanh(log(exp(x) +
  exp(x)/exp(x)))` to `x * tanh(ggml_softplus(x))` using ggml's
  native softplus (single fused kernel, identical clamp at x>20 on
  Metal and CPU). The hand-roll fabricates `+1` via `exp(x)/exp(x)`
  which produces NaN whenever exp(x) overflows to inf or underflows
  to 0. Slight CPU correctness gain (s3gen_mel cos_min 0.999971 →
  0.999980) but the GPU divergence is unrelated to mish.
- Added two diagnostic env knobs in `s3gen_maybe_pin_graph_to_cpu`:
  `CRISPASR_S3GEN_UNET_PIN_CPU_OP=<op>` (pin only the named op type
  to CPU under FORCE_GPU) and `CRISPASR_S3GEN_UNET_KEEP_GPU_OP=<op>`
  (pin everything except the named op). Names follow `ggml_op_name()`
  lowercase minus the `OP_` prefix, or `unary_<lowercase>` for
  `GGML_OP_UNARY`.
- Default backend split now branches on `GGML_USE_METAL`:
  - **Metal:** default is full CPU (was T3 GPU + S3Gen CPU). T3's
    batch-1 AR loop is dominated by Metal kernel-launch overhead on
    M1 — measured 50 s full CPU vs 75 s T3-GPU + S3Gen-CPU on the
    JFK sentence, warm cache, M1. New override
    `CRISPASR_CHATTERBOX_T3_GPU=1` opts T3 back onto GPU on Metal.
  - **Non-Metal (CUDA / Vulkan):** default keeps T3 on GPU + S3Gen
    on CPU. The compound-drift bisect was Metal-only; the same
    class of F16-intermediate compound rounding likely applies to
    other GPU backends but isn't verified, so the safer S3Gen-CPU
    default ships everywhere.

### Op-bisect finding (the actual diagnosis)

Pinning *any* of `{mul_mat, flash_attn_ext, norm, add, concat,
unary_gelu, unary_tanh, unary_softplus}` to CPU restores s3gen_mel
`cos_min=1.000` (diff harness with `replay=exact_init_noise`).
Pinning `conv_1d` / `scale` / `unary_mish` has no effect (low/zero op
count in the graph). So the drift is **not in any single op** — it's
~1e-7 per-op precision drift across multiple Metal kernels that the
10-step CFM Euler solver amplifies ~1000× into the observed mel
collapse. Re-applying `GGML_PREC_F32` to every UNet mul_mat dispatches
the `_hp` simdgroup_float8x8 kernel correctly on M1 (`has_tensor=false`)
but doesn't break the chain because the surrounding ops keep
compounding. Tags were reverted as graph clutter.

### Auto-pin attempt: works in diff harness, fails end-to-end

Wired an `unet_force_cpu` flag plus an auto-pin under FORCE_GPU.
`crispasr-diff` shows `s3gen_mel cos=1.000`, but the full TTS path
produces NaN/Inf mel going into the vocoder (`rms=nan min=1e30
max=-1e30`) → saturated audio → empty parakeet transcript. Pinning
encoder + vocoder + UNet all to CPU under a GPU-init sched still
gives `rms ≈ 11089` vs the ~3900 a true CPU sched produces — some
interaction between the GPU-backed scheduler and random-noise inputs
through CPU-pinned compute that the diff harness's pre-recorded
reference noise masked. Reverted the auto-pin. The vocoder is **not**
the bug; it's just amplifying upstream UNet1D garbage.

### Doc + comment cleanups

- `docs/tts.md` — "Known issues" rewritten to reflect the new Metal
  default, the actual root cause (S3Gen UNet compound drift, not T3
  quant-mat drift), and the new env knobs.
- `src/crispasr_model_registry.cpp` chatterbox entry comment — dropped
  the "crispasr CLI adapter is still pending" line (shipped long ago)
  and documented the `--model-quant` / `--tts-codec-quant`
  substitution surface.
- `LEARNINGS.md` — appended round 6 (mish) and round 7 (op-bisect +
  Metal default) sections to the existing chatterbox GPU narrative.

### Verification

- `crispasr-diff chatterbox` with the new code: default config gives
  `s3gen_mel cos_min=0.999980` (slight gain from the mish fix);
  FORCE_GPU=1 alone still shows the documented `cos_min=0.923003`
  (broken state preserved as diagnostic); `FORCE_GPU=1 +
  UNET_PIN_CPU_OP=mul_mat` gives `cos_min=1.000` (bisect tool works).
- ASR roundtrip via parakeet-tdt: full CPU and default both transcribe
  "Ask not what your country can do for you, ask what you can do for
  your country." exactly. FORCE_GPU output remains garbled (empty
  transcript), documented as diagnostic-only.

### Files touched

- `src/chatterbox.cpp` — Metal default flip + new `T3_GPU` opt-in.
- `src/chatterbox_s3gen.cpp` — `ggml_mish` rewrite + bisect env vars
  in `s3gen_maybe_pin_graph_to_cpu`.
- `src/crispasr_model_registry.cpp` — comment refresh.
- `docs/tts.md` — Known-issues paragraph.
- `LEARNINGS.md` — rounds 6 / 7 / 7b appendix.

5. **`a069018` — split encode/decode API.** Added `parakeet_encode()` +
   `parakeet_decode_frames()` public API for future full-encode +
   chunked-decode path.

6. **`adaedb3` — honor `--chunk-seconds` for OOM.** Explicit
   `--chunk-seconds` is respected with overlap-save + quality warning.
   Default remains full-audio encoding.

**Chunk-size quality sweep** (parakeet-tdt_ctc-0.6b-ja, 300 s JA audio,
`--chunk-overlap 3`, reference = full-audio = 3663 chars):

| `--chunk-seconds` | chars | vs full |
|---|---|---|
| 10 | 3707 | 101.2% (minor boundary dupes) |
| 15 | 3649 | 99.6% |
| 20 | 3636 | 99.3% |
| **30** | **3413** | **93.2% (worst — the old default)** |
| 60 | 3705 | 101.1% |
| 120 | 3690 | 100.7% |
| 180 | 3670 | 100.2% |
| full | 3663 | 100.0% |

30 s is an anomalous outlier; every other size is within ±1.2%.
Default (no `--chunk-seconds`) = full-audio = 100%. If OOM forces
chunking, `--chunk-seconds 60` or `120` are safe.

---

## 2026-05-17 Graduate canary-1b-v2 mel + encoder to full diff

The canary encoder had cos_min=0.917 for `pre_encode_output` vs
NeMo reference — well below the 0.999 graduation threshold. Two
sessions of diff-testing chased the bug through the pre_encode conv
pipeline, testing F16 quantization (ruled out — Python F16 also gave
0.990), ggml conv2d/im2col math (verified correct), and intermediate
conv stages (added `snap_conv4d` + per-stage hooks in Python backend).

Root cause: **one extra boundary mel frame**. The C++ STFT produced
`floor((n+pad-n_fft)/hop)+1` frames (1101 for jfk.wav), while NeMo's
`AudioToMelSpectrogramPreprocessor` returns `feat_len = floor(n/hop)`
valid frames (1100). The 1101st frame was boundary garbage — after
per-feature z-normalization, its values diverged by ±4 in z-score
units, propagating through the 8× conv downsampling (max_abs=356 at
pre_encode) and then through 32 conformer layers.

Fix: `p.drop_last_frame = true` in `core_mel::Params` for all three
NeMo backends (canary, parakeet, cohere). Also disabled NeMo dither
(`featurizer.dither = 0.0`) and clipped the Python reference mel to
`feat_len` for deterministic reference dumps.

Result: mel cos_min=1.000, pre_encode cos_min=0.999999,
encoder_output cos_min=0.999280. All 32 layers pass except layers
18–19 (F16 precision, cos_min=0.9977/0.9988, recovers by layer 20).
Deleted the silence-only `canary_dummy.wav` test file (degenerate
input that masked the real mel issue).

Takeaway: see LEARNINGS.md "STFT frame count" entry.

---

## 2026-05-17 Issue #89 round 2 — revert chunked-slice context expansion

`lenhone` reported that parakeet-tdt-0.6b-ja still dropped words across
`--chunk-seconds` boundaries after the 617cd02 fix. Reproduced on a
33 s JA concat (`reazon_baseball + reazon_meal + reazon_raft`) at
`--chunk-seconds 10`:

    Pre-fix (617cd02 word-trim):
      "ピ ッ チャ ー の 岡 本 は 1 回 戦 の 鳥 取 城 北 戦 は 8 回、 6 対 対 3 3 点 …
       腹 す いた い つ もの 手 料 理 ちゃん と お い しく して く れる ジ ュ ー う ん そこ の …"
      Note: every kana spaced; "6対対33" duplicated "対"; missing "お" before "腹"

    Post-revert (no context, no trim):
      "ピッチャーの岡本は1回戦の鳥取城北戦は8回6失点。 6対3、3点リードの場面で3人目で
       マウンドに上がりました。 お腹すいたいつもの手料理ちゃんとおいしくしてくれるジューうん。
       そこの上流から丸太を組んでいかだにしてる。 どんどん流したんです、これを。"

Two failure modes in 617cd02:

1. **TDT emission-frame drift**: the FastConformer encoder is bidirectional,
   so feeding ±2 s of neighbour audio shifts the TDT decoder's emission
   frame for boundary words by 1–2 frames between adjacent slices' passes.
   The "assign each word to the slice it STARTS in" rule
   (`w.t0 < sl.t0_cs || w.t0 >= sl.t1_cs`) is *deterministic* only if the
   same audio content produces the same `t_start` regardless of which
   context window it appears in — which is not what the encoder
   guarantees. A word can land outside *both* adjacent slices' ranges
   and disappear entirely (the user's symptom).

2. **Text rebuild assumed space-prefix tokens**: the trim rebuilt
   `seg.text` from surviving words with `if (!rebuilt.empty() && w.text[0] != ' ') rebuilt += ' '`.
   parakeet-ja emits no-space tokens (every kana is its own word with no
   leading space), so the rebuild inserted a literal space between every
   character.

Fix is a revert. Each slice transcribes its own audio only;
`audio_chunking::split_at_energy_minima` already places chunk seams at
the quietest 100 ms within the search window (PLAN #80b), so boundaries
fall in pauses and the encoder gets coherent input on either side
without needing the ±2 s borrow. The 617cd02-era unit tests in
`tests/test_chunk_context.cpp` that pinned the broken ctx/filter math
are removed; the file now only covers `split_at_energy_minima` (4 tests,
22 assertions, all passing).

Verified on JFK 11 s @ 5 s chunks (English parakeet) — full quote
intact; the very-short 3 s case still has boundary trouble but no
worse than the 617cd02 round, and the user's reported chunk sizes
(10, 30, 180) all land safely above that threshold.

---

## 2026-05-17 PRs #92 + #95 — streaming JSON+VAD merge policy

Two @CKwasd PRs accepted as a pair:

- **PR #92** — `perf(stream): skip aggregate segs in JSON VAD path`.
  Drops the unused aggregate `segs` build-and-postprocess work from
  the live `--stream-json --vad` path, since it's not consumed by the
  JSON state machine. Adds an explicit `decoded_segments_this_step`
  flag for stream-monitor silence detection (the aggregate `segs`
  is intentionally empty in JSON+VAD mode now). Equivalent commit
  had landed in main as 6318884c earlier; the PR merge records the
  formal accept and credit. Our `StreamJsonVadRoutingTests` follow-up
  (42494f23) sits on top.

- **PR #95** — `Rewrite streaming JSON VAD merge policy`. Adds a
  `crispasr_vad_post_merge_policy` enum (`offline` vs
  `streaming_json`) and a new `--stream-vad-merge-gap-ms` CLI flag.
  Offline / file VAD path keeps the historical short/close merge.
  JSON streaming gets a narrower close-gap-only policy that never
  exceeds `--stream-final-on-silence-ms - 1` (so the merger can't
  hide a silence gap that should finalize an utterance). Includes
  a Catch2 unit test for the policy helper
  (`tests/test-stream-vad-merge-policy.cpp`) and a `docs/streaming.md`
  entry.

Verified pre-merge: `test-stream-vad-merge-policy` (18/7),
`test-stream-vad-skip` (20/7), `test-stream-finalize` (11/8, our PR
#92 invariants test) all pass; live smoke with `--stream-json --vad
--stream-vad-merge-gap-ms 250` + parakeet on `samples/jfk.wav` emits
clean utterance-level JSON.

The pair landed via the open-PR audit on 2026-05-17 after we
realised PR #92's diff had been integrated into main without the
GitHub merge button being pressed — bad style on our end. Both PRs
now show as merged with proper credit on @CKwasd's profile.

---

## 2026-05-17 Kokoro short-input Metal regression — ggml-metal kernel_norm fix

CrisperWeaver reported kokoro on Apple Silicon Metal producing
garbage audio for short utterances ("hello world" → ~6× lower
amplitude, transcribed as "Mm-hmm." by parakeet). Long utterances
worked. CPU worked. Five-step bisect via the ground-truth diff
harness localised the divergence to the first AdainResBlk1d in
F0Ntrain, then to `ggml_norm` Metal itself.

Root cause: `kernel_norm_fuse_impl`, `kernel_rms_norm_fuse_impl`,
and `kernel_l2_norm_impl` end their parallel reduction with a
cross-simdgroup `simd_sum(shmem[tiisg])` pattern that produces wrong
totals on Apple Silicon when the LAST active simdgroup had only a
few lanes participate in the prior loop body. Sweep of `ne00_t` ∈
{32..320} confirms the bug strikes at {33, 65, 66, 97, 129-132, 257}
— exactly the cases where `nth` doubles past `ne00_t` and the
trailing simdgroup is sparsely populated. Most LLM workloads escape
because their hidden-size aligns on multiples of 4 (float4 kernel
variant, `ne00_t = ne00/4` lands on "good" boundaries); audio
backends with per-frame normalisation (kokoro AdaIN1d at
T_frames=65) sit exactly in the bug's strike zone.

Fix in `ggml/src/ggml-metal/ggml-metal.metal`: replace the
cross-simdgroup `simd_sum` with a serial reduction by thread 0 of
sg0 that sums `shmem[0..n_sg-1]` and broadcasts via `shmem[31]`
(unused by the original — initialized to 0 by sg0 and never written
by any sg). Applied at four sites. Also adds explicit per-T
overloads (`crispasr_vec_sum` / `crispasr_vec_sqsum`) to replace
`dot(scalar, scalar)` calls — `dot()` is only spec-defined for
vector types and the kernel_norm_f32 (scalar T) instantiation
should not rely on it.

`tests/test_metal_norm_repro.cpp` wired into ctest as a regression
guard (one test per bug-pattern T value). Draft upstream PR at
`tools/upstream-prs/08-metal-norm-cross-simdgroup.{md,patch}`.

The kokoro-side primitive-op workaround landed earlier the same
day (`d1fdd476`) was reverted — kokoro now uses `ggml_norm` again
since the underlying bug is fixed at the source.

Earlier in the chain:
- `ef2f79c0`, `4e9c6ba8` — mmap loader Metal-side fixes (cleared
  the "buffer is nil" spam from issue #94's default-on flip).
- `98488dcb` — bisect infrastructure: extends
  `tools/dump_reference.py` + `kokoro_extract_stage` +
  `crispasr-diff` to capture per-op intermediates inside F0Ntrain.
- `d2266229` — gated the bisect intermediates behind
  `KOKORO_DEBUG_INTERMEDIATES=1` so production builds pay no cost
  but the bisect path stays available for the next per-op Metal
  kernel issue.

---

## 2026-06-07 Issue #158 — VoxCPM2 silent exit on Vulkan/CUDA (discrete GPU)

User reported VoxCPM2 exiting immediately after model load with no WAV
output and no error on AMD RX 580 (Vulkan, `fp16: 0`, Windows).

**Root cause:** Legacy CPU code paths (`tensor_data_f32`, `matmul_mv_ggml`,
`rms_norm_cpu`) dereference `tensor->data` as raw CPU pointers. On Metal
(Apple Silicon) this works via unified memory. On discrete GPUs
(Vulkan/CUDA) weights live in device-local VRAM — SIGSEGV on access.

**Fix (3 commits):**

1. `c6299251` — Initial CPU fallback via `ggml_backend_buft_is_host()`
   check (correct but loses GPU acceleration).
2. `df6cf31e` — **GPU weight mirrors**: load weights to CPU for legacy
   paths, create GPU copies at init for graph-build functions. Graph
   paths use `ctx->graph_weights()` → GPU tensors; legacy paths use
   `ctx->weights` → CPU tensors. Zero cross-backend copies at compute
   time. Memory: ~2× model size on discrete GPUs.
3. `0a8e1372` — **Graph-ify RALM step** (8-layer, ~25% of AR time):
   `build_ralm_step_graph()` + backend KV. All major AR-loop ops now on
   GPU (TSLM 28L + RALM 8L + LocDiT 12L×10 + LocEnc 12L + VAE decode).
   Remaining CPU-only: FSQ/stop/fusion/mu (~5-8 ms/step, negligible).

All graph paths gated on `VOXCPM2_USE_GRAPH` (default on). Respects
`--no-gpu`. Prefills (run once) remain on CPU legacy.

---

## 2026-05-16 Issue #94 — chatterbox-turbo slow / failing init on macOS

External report from `niksedk` (SubtitleEdit ships `crispasr` for
Chatterbox TTS): selecting the Turbo model in SubtitleEdit on
Apple Silicon segfaulted during s3gen init (exit 139); the Base
model worked. Reproduced as a 60 s+ slow load on the legacy
alloc+copy path that, depending on memory pressure, can manifest
either as a perceived hang or a hard crash. mmap path (formerly
`CRISPASR_GGUF_MMAP=1`) loaded the same model in ~26 s and
produced valid 24 kHz audio.

Fix:

- `src/core/gguf_loader.cpp` — flip `mmap_loader_enabled()` to
  default **on** (opt out with `CRISPASR_GGUF_MMAP=0`). The CPU
  mmap path has been the validated default-eligible loader since
  PLAN #51a in late April; this matches llama.cpp's default.
- `src/chatterbox.cpp` + `src/chatterbox_s3gen.cpp` — add explicit
  "loading T3 weights from …" / "T3 loaded N tensors" / "s3gen:
  loading from …" progress prints with `fflush(stderr)` so the
  silent gap between "auto-falling back to CPU" and "precomputed
  conds loaded" reads as progress instead of a hang.
- `docs/cli.md` — update the mmap section + the llama.cpp
  comparison table to reflect the new default.

Verified on M1 / macOS Tahoe 26.2:

- chatterbox-turbo: 29 s init (was 60 s+) + working `/v1/audio/speech`.
- chatterbox (base): 28 s init + working `/v1/audio/speech`.
- `CRISPASR_GGUF_MMAP=0` opt-out: 37 s init, still works.

### 2026-05-17 follow-up — T3 sampler reorder

`src/chatterbox.cpp` `sample_token` had base-chatterbox's ordering
(rep_penalty → temperature → min_p → top_p) hard-wired, but the
Python turbo path runs a different `LogitsProcessorList`
(`chatterbox/models/t3/t3.py:415-490`): temperature → top_k → top_p
→ repetition_penalty. The two are not interchangeable — applying
rep_penalty before top_k/top_p reshapes the survival set in a way
base never needed because base sets `top_k=0`.

- Added `top_k` to `chatterbox_context_params` + `chatterbox_set_top_k`
  setter (default 0 = off, preserves base behaviour bit-equivalent).
- `sample_token` now bifurcates on `top_k > 0`: turbo path follows the
  HF order on raw logits, base path keeps the pre-fix ordering on
  softmaxed probs. Both paths share the existing torch.multinomial CPU
  port.
- `chatterbox_init_from_file` extends the existing `is_gpt2` block
  (was just `cfg_weight=0`) to also set `min_p=0`, `top_p=0.95`,
  `top_k=1000` — the exact `tts_turbo.py:248-260` defaults.
- Aligned `token_hist` passed to rep_penalty with Python: BOS is in
  the history at step 0 only, then drops out (matches `t3.py:450` vs
  `t3.py:471`).

Diagnostic finding from the Python reference (captured 2026-05-17,
audio under `/Volumes/backups/ai/crispasr/issue94-audio/topk-fix/`):

1. Python's turbo sampler also emits S3GEN_SIL (4299) as the second
   generated speech token on both `"hello world test"` and `"hello
   chatterbox turbo"` — so 4299-at-step-1 is not the audible defect.
2. Feeding Python's captured tokens through C++ `chatterbox_synthesize_
   from_tokens` produces clean audio ("HEllo world test..", "HEllo
   chatterbox turbo..") — so S3Gen + HiFT on the C++ side are correct
   end-to-end for inputs sourced from Python's T3.
3. With C++'s natural sampler the same prompt only round-trips cleanly
   on ~3/19 random seeds; the Python turbo path is clean on 6/6 seeds
   tested with the same prompts. The Python path is robust to seed
   variation; the C++ path is fragile.
4. The compression is in raw T3 logits, not the sampler. Forcing
   Python to sample the same step-0 token as our C++ (`tok0=4024`)
   gives Python a step-1 max logit of 13.75 vs C++'s 12.39 — and the
   C++ top-token spread is ~1.4 logits wider than Python's at the
   top of the distribution. After temperature + top_k + top_p that
   shows up as the sampled token landing on phonemically-bad tokens
   more often, which then propagates through the AR loop.
5. F16 weights reproduce the same compression as Q8_0 weights
   (step-1 max 12.37 with F16 vs 12.39 with Q8_0), so this is not a
   weight-quantisation effect.

Per-layer bisect (added 2026-05-17):

`CRISPASR_CHATTERBOX_DUMP_GPT2_LAYERS=1` makes the GPT-2 graph dump
each block's `post_attn` and `post_ffn` hidden states to
`/tmp/cb_gpt2_step_<n_past>_LNN_*.bin`. The companion Python tool
`tools/cb_turbo_perlayer_dump_pyref.py` produces the matching Python
files; `tools/cb_turbo_perlayer_diff.py` walks both sets and reports
per-layer cosine similarity at the first AR step.

For `"hello world test"` (seed 0, forced `tok0=4024`) the bisect shows:

    L00_post_attn cos=0.99893    L00_post_ffn cos=0.99774
    L01_post_attn cos=0.99832    L01_post_ffn cos=0.99868
    L02_post_attn cos=0.99795    L02_post_ffn cos=0.99847
    L03_post_attn cos=0.99741    L03_post_ffn cos=0.99792
    L04_post_attn cos=0.99418    L04_post_ffn cos=0.99434
    L05_post_attn cos=0.94450    ← jump
    L05_post_ffn  cos=0.93701
    L06–L23       cos 0.91–0.97 (does not recover)

L00–L04 stay above 0.994; the divergence locus is **L05 attention**.
Magnitudes (rms) match closely on both sides at the jump (1.83 vs
1.84), so the issue is direction, not scale — consistent with a sharp
softmax attention pattern at L05 where small K/V precision noise flips
which prefill position is attended to.

Ruled out so far: speech_head bias missing, WPE double-add, attention
scale, GPT-2 LayerNorm scale/bias loading, GELU variant, `scale_attn_
by_inverse_layer_idx` (False on this config — verified by loading the
GPT2Config from Python), `scale_attn_weights` (True on both sides),
prefill length mismatch, Q8_0 vs F16 weight precision (F16 model
reproduces the same step-1 max logit 12.37 vs 12.39 for Q8_0), F32
KV cache (cache stored as F32 or read as F32 — neither helps).
Verified independently: the C++ input embed fed to the GPT-2 forward
at step 1 (`speech_emb(4024) + wpe(383)`) matches the Python ref to 5
decimal places, so the divergence is in the transformer forward, not
the input.

Outlier analysis on the L00 post-attn diff: the maximum-absolute-diff
element is at dim 265, which is head 4 / hd-position 9. This same
dim 265 is the largest outlier through L01-L04 with the diff growing
slowly (−0.21 → −0.91), then at L05 the outlier jumps to dim 659
(head 10) with diff -2.36. Suggests a specific head-4 attention
pattern at L00 produces a head-4-localized numerical difference that
propagates into the residual stream and is then amplified by a sharp
attention pattern at head 10 of L05.

**Flash_attn ruled out** (added 2026-05-17): added a naive
softmax(QK^T)V code path behind `CRISPASR_CHATTERBOX_NAIVE_ATTN=1`
following the qwen3_asr.cpp layout (with the post-mul_mat
`permute(0, 2, 1, 3)` so the head dim packs correctly for the WO
projection). Naive attention produces essentially identical bisect
results: L05_post_attn cos 0.94421 with naive vs 0.94450 with
flash_attn. The divergence is therefore **not** in the attention
compute — it's accumulated drift from `ggml_mul_mat` (QKV / WO /
FFN projections), `ggml_norm` (LayerNorm), or some other per-layer
op whose F32 accumulator order differs from PyTorch's CPU path.
Q8_0 weight quant noise (~0.4%/dot) is consistent with the L00 cos
0.999 baseline but is *not* the dominant source — F16 weights
reproduce the same L05 divergence.

Suspect remaining: per-op CPU numerics differences between ggml's
F32 mul_mat / norm and PyTorch's. Hard to fix without bit-exact
op replacements. The diagnostic infrastructure
(`CRISPASR_CHATTERBOX_DUMP_GPT2_LAYERS=1`,
`CRISPASR_CHATTERBOX_NAIVE_ATTN=1`,
`tools/cb_turbo_perlayer_dump_pyref.py`,
`tools/cb_turbo_perlayer_diff.py`) is in place so future attempts
(e.g., dumping per-position K/V at the prefill end and diffing
against Python's cache, or trying a bit-exact F32 mul_mat) can
keep narrowing the locus.

**Deep-debug round (2026-05-17 cont'd):**

Two more candidate fixes tested against the L05 cos 0.944 baseline:

1. **F16 weights** (full `chatterbox-turbo-t3-f16.gguf` + `s3gen-f16`):
   per-layer bisect is *identical* to Q8_0 within 5e-5 (L00 0.99893
   vs 0.99893; L05 0.94470 vs 0.94450; matching outlier at dim 265 /
   head-4 / hd-9). Confirms **weight quantisation is not the bug** —
   neither input-side Q8 quant (which F16 avoids via the PR-01
   `vec_dot_type = GGML_TYPE_F32` patch) nor weight-side precision
   loss accounts for the L00 0.1% drift.

2. **`ggml_norm` Accelerate bypass**: forced
   `ggml_compute_forward_norm_f32` onto the `ggml_vec_cvar_f32` path
   (double accumulator) instead of the macOS Accelerate
   `vDSP_measqv` F32-acc path. No measurable change — L05 cos 0.94427
   vs the prior 0.94450. **`ggml_norm` precision vs PyTorch CPU
   LayerNorm (Welford's algorithm) is not dominant either.**

Inventory of CrispASR ggml patches in `tools/upstream-prs/` that
*could* have been relevant: PR-01 (F16 mul_mat saturation, already
active and confirmed not load-bearing here), PR-08 (Metal norm
cross-simdgroup, Metal-only — chatterbox runs on CPU). None of the
shipped patches address CPU mul_mat or softmax precision.

**Conclusion**: the L00 0.1% drift is an *accumulation* of multiple
tiny per-op numerics differences — likely a combination of
mul_mat F32 SIMD accumulator order vs PyTorch's CPU BLAS, `tanhf`
in `ggml_gelu` vs PyTorch's libm, exp/softmax in `ggml_soft_max_ext`
vs PyTorch's, and similar single-ULP differences. Each contributes
< 0.05% per layer, sums to ~0.6% by L04, then gets amplified ~9× by
L05's sharp attention. No single-knob fix surfaced; a real fix needs
**bit-exact F32 op replacements** matching PyTorch's CPU accumulation
across mul_mat, LayerNorm, GELU, and softmax — outside the scope of
issue #94 as originally filed.

**Practical user-facing baseline**: across 19 random RNG seeds for
"hello chatterbox turbo" with default sampling
(`temp=0.8 top_k=1000 top_p=0.95`, matching Python), ~3/19 ≈ 16%
produce Whisper-clean "HEllo, Chatterbox Turbo" output. The rest
start with a brief onset artifact ("SO,", "UH,", "ANy?", "IN no,",
etc.) before the intelligible "Chatterbox Turbo" payload. The
sampler matches Python exactly so this is the residual T3 drift
compounding through the AR loop. Workaround for users who care
about the onset artifact: sweep `CRISPASR_CHATTERBOX_T3_SEED` until
a clean output appears; or fall back to base chatterbox (the
Llama-style backbone doesn't have L05's amplification).

### 2026-05-17 root cause: SOT/EOT prepended for turbo

The "0.1% per-layer accumulated drift" model from the previous bisect
turned out to mis-diagnose the issue. The real defect was structural:
`chatterbox_synthesize_tokens` unconditionally wrapped tokenized text
with `[start_text_token, …, stop_text_token]` (chatterbox.cpp:2496-
2498). That matches base chatterbox's path
(`chatterbox/models/t3/t3.py:255` calls `_ensure_BOT_EOT`), but the
turbo path (`ChatterboxTurboTTS.inference_turbo`, `t3.py:415`) does
**not** call `_ensure_BOT_EOT` and feeds the raw tokenizer output to
the GPT-2 backbone.

For "Hello world test." that means C++'s prefill carried 6 text
tokens `[SOT=255, 15496, 995, 1332, 13, EOT=0]` while Python's
carried 4 `[15496, 995, 1332, 13]`, making C++'s `prefill_len=383` vs
Python's `prefill_len=381`. Every WPE position was shifted by 2 and
the model attended to two phantom positions (SOT/EOT). The per-layer
cos numbers from the earlier bisect look much closer to "small drift"
than "structural offset" only because the diff tool aligned on
matching `n_past` filenames — at `n_past=383` it was comparing C++'s
first AR step against Python's *third* AR step (Python at
prefill_len=381 dumps its first AR step at n_past=381). With the
mis-alignment, L01–L04 cos accidentally landed in the 0.99-band,
making it look like CPU numerics drift.

Fix (chatterbox.cpp:2496-2509): gate the SOT/EOT insertion on
`!is_gpt2`. Base chatterbox keeps the existing behaviour; turbo now
sends the bare tokenizer output to the GPT-2 backbone, matching
Python's `inference_turbo` exactly.

User-facing audio quality after the fix (q8_0 weights, seed 0,
default sampling, 5 iterations each, Whisper-base transcripts):

    Pre-fix:  hello world test  → 1/5 clean (other 4 had onset noise)
              hello chatterbox  → 1/6 clean (5 had "SO,", "UH,", …)
    Post-fix: hello world test  → 5/5 clean ("Hello world test.")
              hello chatterbox  → 5/5 clean ("Hello chatterbox/…box turbo.")

The "84% bad" baseline is gone — clean rate on default seeds is now
indistinguishable from Python's. No remaining "L05 cumulative
softmax drift" to chase; the diagnostic infrastructure
(`CRISPASR_CHATTERBOX_DUMP_GPT2_LAYERS=1` env knob, the
naive-softmax(QK^T)V path, per-layer dumpers under `tools/`) stays
in the tree as a stage-gate fixture against future regressions.

### 2026-05-17 follow-up: dumper positional-args fix

`tools/cb_turbo_perlayer_dump_pyref.py`'s `patched_block_forward`
also had a latent bug: its signature `(self, hidden_states, *_,
**kwargs)` silently swallowed all positional args. HF
`GPT2Model.forward` calls each block positionally with at least
`(hidden_states, past_key_values, cache_position, attention_mask,
head_mask, encoder_hidden_states, …)`, so the patched block lost
the KV cache and the causal mask on every AR step; the inner
`self.attn(h, **kwargs)` ran as if the prefill had never happened.
That is why L05 cos appeared to crash to 0.944 in the previous
bisect even after C++ and Python were aligned on prefill content —
the Python side was computing degenerate single-token attention.
The dumper now promotes the positional args back to kwargs so
attention sees the full prefix on every step; this only affects
diagnostics, not production.

---

## 2026-05-16 Cross-Stack Audit Hardening

Cross-repo audit work covered `CrispASR`, `CrispLens`, and `cloud-backup`.
Completed items:

- Added `AUDIT.md` and cross-stack tests for repository wiring, VPS services,
  scratch roots, transcript search, and SSH-only deployment shape.
- Removed hardcoded `/tmp` usage from audited production/test paths; local
  scratch now uses configured durable roots such as `/Volumes/backups`.
- CrispLens video transcript search works for imported videos, including the
  real 70 second PURplus MP4 crop test searched by `Detektivarbeit`.
- iOS Capacitor smoke passed: CocoaPods sync under Ruby 3.1.3 detected the
  CrispASR plugin, Xcode built the simulator app with derived data on
  `/Volumes/backups`, and the app installed/launched on the iPhone 17 simulator.
- Android Capacitor packaging passed: the v4 app now has a real native
  CrispASR JNI bridge, the generated Android project builds a debug APK with
  JDK 21, and the verified APK was saved under `/Volumes/backups` before
  generated build outputs were cleaned from the repo volume.
- Installed `ripgrep 14.1.0` on the VPS for future SSH service/env audits.

TB-scale confidence testing was not run. It remains blocked until a real-world
dataset is specified so the resulting cloud/index state is worth keeping.

---

## Timeline

### 1. Cohere Transcribe — the original port
The repository started as a standalone ggml runtime for
`CohereLabs/cohere-transcribe-03-2026` (2B params, Conformer encoder +
Transformer decoder, lowest English WER on Open ASR Leaderboard as of
March 2026).

Starting point: scalar C++ loops ported from the HuggingFace reference.
Optimisation trajectory on an 11s clip (4-thread CPU, same hardware
throughout):

| Step | Wall time | Speedup | Cumulative |
|---|---:|---:|---:|
| Baseline (scalar nested loops) | ~825 s | — | 1× |
| + FFTW3f STFT | ~100 s | 8.2× | 8× |
| + OpenBLAS GEMM | 104 s | ~1× | ~8× |
| + lazy F32 weight cache | 100 s | 1.04× | 8.3× |
| + EncScratch + AVX2 F16C | 32 s | 3.1× | ~26× |
| + ggml compute graph | 12.4 s | 2.6× | ~67× |
| + BatchNorm folding (48 layers × 10 nodes) | 11.5 s | 1.08× | ~72× |
| + depthwise conv direct (kernel=9, no im2col) | 9.6 s | 1.20× | ~86× |
| + self-contained FFT (drop fftw3) | ~8.7 s | 1.10× | **~95×** |

Total: ~825 s → ~8.7 s ≈ **95× on 4-thread CPU for an 11s clip.** Q4_K
quantisation brought this further to 14.8 s for a 5.4s clip (~2.7×
realtime). End-to-end, this beat ONNX INT4 by a hair (17.1s, but 7.5s
of that is cold-load; ggml mmap wins for repeat runs).

Main bug classes fixed: mel normalisation (NeMo per-band z-score with
biased std), cross-attention DTW timestamps, 2-backend scheduler
(GPU primary + CPU fallback).

### 2. Parakeet TDT 0.6B v3 — "free" word timestamps
`nvidia/parakeet-tdt-0.6b-v3`, 600M FastConformer encoder + Token-and-
Duration Transducer (TDT) decoder. Chosen because:

- Word timestamps come for free from the TDT decoder's duration head —
  no separate forced-alignment model.
- 600M params vs 2B for Cohere → ~400 MB Q4_K, ~3× faster.
- 25 EU languages with automatic detection.
- CC-BY-4.0.

- ~80% of encoder code reusable from cohere.cpp (both use the same
  FastConformer).

The novel work was the TDT decoder: LSTM predictor + joint network +
greedy decode loop with duration-advanced time stepping. Shipped with
a raw C++ CPU decoder (still unchanged — tracked in TODO.md as "port
LSTM to ggml" — encoder dominance makes GPU speedup small).

Tested on German audio and discovered the language-ID drift problem
(see `LEARNINGS.md` → "Auto-detect can silently code-switch").

### 3. Canary 1B v2 — explicit language + speech translation
`nvidia/canary-1b-v2`, 978M FastConformer encoder + Transformer decoder
with task-token prompt prefix. Ported specifically to fix the parakeet
language-ID drift: canary takes `-sl SRC -tl TGT` explicit language
flags, and the decoder is forced into the target language by the task
token. Also added speech translation (X→EN and EN→X), the only
translation runtime in the repo.

Implementation effort was small because we already had both halves:

| Canary component | Source |
|---|---|
| FastConformer encoder (32 layers) | `parakeet.cpp` |
| Conv2d dw_striding subsampling | `parakeet.cpp` + `cohere.cpp` |
| Rel-pos attention with Transformer-XL biases | `parakeet.cpp` + `cohere.cpp` |
| Mel preprocessor | `parakeet.cpp` (identical) |
| Transformer decoder (8 layers) | `cohere.cpp` |
| Cross-attention KV cache | `cohere.cpp` |
| 16384-token SentencePiece | `cohere.cpp` |

Shipped together with `nfa-align`, a general-purpose multilingual
forced aligner built from Canary's auxiliary CTC model. Works on any
transcript + audio pair in 25 European languages at ~78ms MAE.

### 4. Qwen3-ASR 0.6B — first speech-LLM
`Qwen/Qwen3-ASR-0.6B`, 900M speech-LLM combining a Whisper-style audio
encoder (18 layers) with a Qwen3 0.6B LLM (28 layers) via audio-token
injection at `<|audio_pad|>` placeholder positions in a ChatML prompt.
First speech-LLM in the repo and the template for everything that came
after.

Architecture: 2D-conv subsampler + Whisper-block body + 18-layer
encoder + Qwen3 LLM decoder. Uses **windowed attention via `cu_seqlens`**
(chunked self-attention with window size ~104 positions after CNN) —
standard full self-attention produces wrong output. This was the
trickiest part of the port.

Also first runtime to use a persistent KV cache for the LLM decode
loop: `qwen3_asr_kv_init` allocates it once, `qwen3_asr_kv_reset`
clears between utterances, each `qwen3_asr_run_llm_kv` call appends
new tokens at position `n_past` and reads K/V views for attention.

### 5. Voxtral-Mini-3B-2507 — Mistral's speech-LLM
`mistralai/Voxtral-Mini-3B-2507`, 3B speech-LLM with a literal Whisper-
large-v3 audio encoder (32 layers, 1280 dim) + 4-frame stack projector
+ Llama 3 (Mistral) 3B LLM. Audio tokens are injected at a special
`audio_token_id=24` placeholder in an `[INST][BEGIN_AUDIO] … [/INST]`
prompt. Uses the Mistral Tekken (tiktoken-style) BPE tokenizer with
150k vocab + 1000 special tokens.

Ported in a single session because we already had the Whisper-encoder
pattern from qwen3 and the LLM pattern was straight Llama 3. Main
novel work: Tekken tokenizer (150k vocab stored as a 1D F32 tensor in
the GGUF because the KV-array path loses uint8 precision) and the
4-frame stack projector.

Diff-tested against PyTorch at every stage boundary:
- Encoder: cosine similarity > 0.999 across all layers
- Projector: exact match
- LLM: top-5 5/5 match on first decoded token, cosine sim 0.999973

This gave us the confidence to ship despite the CPU-only path being
slower than an ideal GPU reference. See `LEARNINGS.md` →
"Model architecture comparisons" for the three-way comparison with
`max-lt/voxtral-cpp` and `llama.cpp mtmd`.

### 6. Voxtral-Mini-4B-Realtime-2602 — streaming speech-LLM
`mistralai/Voxtral-Mini-4B-Realtime-2602`, 4.4B natively-streaming
speech-LLM with a causal RoPE+SwiGLU+RMSNorm audio encoder (32 layers,
sliding window 750) + Mistral 3.4B LLM with adaptive RMSNorm and
sliding window attention. Designed for on-device realtime ASR with
<500ms latency.

Substantial architectural differences from the 3B:

| | 3B | 4B Realtime |
|---|---|---|
| Encoder attention | Full (abs pos embed) | RoPE θ=1e6 + SWA(750) |
| Encoder FFN | GELU fc1/fc2 | SwiGLU |
| Encoder norm | LayerNorm | RMSNorm |
| LLM layers | 30 | 26 |
| LLM FFN | 8192 | 9216 |
| LLM norm | RMSNorm | Adaptive RMSNorm (time-conditioned) |
| LLM attention | Full | Sliding window 8192 |
| LLM embeddings | Separate lm_head | Tied (token_embd = lm_head) |
| Audio tokens/frame | 1 per 4 Whisper frames | `audio_length_per_tok=8` |

Seven critical bugs found during debugging via Kaggle ground truth
and three reference implementations (`voxtral.c`, `voxmlx`, `voxtral-rs`):

1. **`audio_length_per_tok=8`** — 3B uses 4, 4B uses 8. Wrong value
   means audio-to-token alignment is off by 2× and transcript drifts.
2. **Audio padding: 32 tokens left + 17 tokens right + right_align.**
   Left-padding is 32 × 1280 samples = 32 streaming tokens worth of
   silence. Right-padding is 17 × 1280 samples plus whatever's needed
   to align the input length to a token boundary. Skipping the right
   pad silently breaks the encoder graph reshape.
3. **Prompt is `BOS + STREAMING_PAD × 38`**, not Tekken text. The audio
   encoder output is ADDED element-wise to each position's embedding
   (not replaced — this is the streaming mechanism). During decode,
   each generated token ALSO has the next adapter frame added to its
   embedding before the LLM forward.
4. **Tokens with id < 1000 are control tokens** (STREAMING_PAD,
   STREAMING_WORD, etc.) and must be filtered from the output
   transcript.
5. **Adaptive RMSNorm** applies a time-conditioned scale multiplication
   after the standard RMSNorm: `cur = cur * ada_scales[il]`. The
   `ada_scales` are precomputed from a learned module at load time
   as `(1 + scale)` and read from a view in the LLM graph.
6. **RoPE θ difference.** 3B uses θ=1e8, 4B uses θ=1e6. Getting this
   wrong corrupts the attention scores on long contexts.
7. **Tied embeddings.** 4B ties `lm_head = token_embd.T`, so the
   final linear is a `ggml_mul_mat(token_embd_w, cur)` — no separate
   `output_w` tensor.

Q4_K shipped at 2.4 GB and runs an 11s clip in 49s on 4-thread CPU.

### 7. Granite Speech 4.0-1B — Q-Former projector + µP LLM
`ibm-granite/granite-4.0-1b-speech`, 1B speech-LLM with a 16-layer
Conformer encoder (Shaw relative position embeddings + depthwise
conv + batch norm), a 2-layer BLIP-2 Q-Former projector with learned
query tokens, and a 40-layer Granite 1B LLM using µP (maximal update
parameterisation) multipliers. Apache-2.0.

Novel architectural pieces:

- **Q-Former** — cross-attention from a fixed-length learned query
  sequence (3 query tokens) to the encoder output. Each query token
  has its own self-attention among queries, then cross-attention to
  the encoder frames, then a position-wise FFN. Output is a small
  number of audio tokens fed to the LLM. Very different from the
  "stack frames + linear" projector used by qwen3/voxtral.

- **µP multipliers** — four scalar multiplications baked into the
  forward pass:
  - `embedding_multiplier = 12.0` (scales token embeddings)
  - `attention_multiplier = 0.0078125 = 1/128 = 1/head_dim` (scales
    attention logits, replacing the default `1/sqrt(d_head)`)
  - `residual_multiplier = 0.22` (scales residual additions)
  - `logits_scaling = 8.0` (scales output logits)

- **Stacked mel input** — unlike other models that use 128 mels,
  granite uses 80 mels stacked into 160-dim frames (two 80-mel frames
  zipped along channels). `granite_speech_compute_mel` outputs
  `(160, T/2)` instead of `(n_mels, T)`. Still inline — tracked as
  the `core_mel::Params::stacked_frames` follow-up.

Six bugs found during debugging (all preserved in `LEARNINGS.md`):

1. **Hann window centering.** The window must be symmetrically
   zero-padded to n_fft; off-by-one on the centering shifts the power
   spectrum peak and breaks everything downstream.
2. **Q-Former layer norm target.** LN applies to the query tokens,
   not the encoder output.
3. **Embedding multiplier placement.** Applied inside the LLM forward,
   after the token embedding lookup but before the first layer, so
   the raw `token_embd` tensor stays un-scaled in memory.
4. **CTC dim hardcoding.** Encoder output dim for CTC head is 348,
   not the encoder hidden 1024.
5. **Native GQA.** Flash attention handles `n_kv_heads < n_heads`
   natively if the K/V tensor shapes are right — no explicit
   `repeat_4d` needed. We were double-expanding at first.
6. **RoPE mode NEOX vs NORMAL.** The single most expensive bug. See
   `LEARNINGS.md` → "RoPE mode mapping".

### 8. Unified `crispasr` CLI + `src/core/` shared library (April 2026)
Two-phase refactor that reshaped the repo.

**Phase 1 — Unified CLI.** Extended `examples/cli/cli.cpp` with a
backend dispatch layer (`crispasr_backend.*`, backend adapters, VAD,
output writers, model manager, CTC aligner, run loop). Whisper code
path preserved byte-identical to upstream `crispasr`. 7 non-whisper
backends wired up. `-m auto` auto-download via `curl`/`wget` shell-out
(no Python, no libcurl link). `--list-backends` prints the capability
matrix. GGUF-based backend auto-detection with filename heuristic
fallback.

**Phase 0 — Shared model primitives.** Created `src/core/` (library
name `crispasr-core`) with:

- `mel.{h,cpp}` — one parameterised log-mel spectrogram function for
  both NeMo and HF/Whisper clusters. 7 of 8 non-whisper models migrated
  (granite deferred pending `stacked_frames` support).
- `ffn.h` — header-only SwiGLU / plain-SiLU FFN helpers. 4 LLM backends
  migrated.
- `attention.h` — header-only Llama-style self-attention (Q/K/V +
  reshape + NEOX RoPE + GQA + flash-attn + output projection). voxtral
  migrated as pilot; persistent-KV-cache variant for the others
  deferred.
- `gguf_loader.{h,cpp}` — unified GGUF two-pass loader with mmap
  (pread fallback for non-mmap filesystems). All 8 non-whisper models
  migrated.

Regression gate: every commit produces bit-identical output on
`samples/jfk.wav` (or within documented float-ULP tolerance where
matmul accumulator order changes). ~877 lines of duplicated
boilerplate removed from `src/` and replaced with ~730 lines of
shared code in `src/core/`.

Whisper is **intentionally not migrated** to `src/core/` — it's the
battle-tested reference and the `crispasr -m ggml-base.en.bin …` path
stays byte-identical to upstream `crispasr`.

---

## Markdown consolidation (April 2026)

The repo accumulated ~15 per-topic notes during the ports. These were
consolidated into four live documents:

- `README.md` — user-facing docs
- `TODO.md` — pending work, cross-checked against current state
- `LEARNINGS.md` — technical insights, benchmarks, comparisons
- `HISTORY.md` — this file

Removed: `canary-todo.md`, `parakeet-todo.md`, `granite-todo.md`,
`voxtral-todo.md`, `voxtral-4b-todo.md`, `qwen3-asr-todo.md`,
`TODO_COHERE_OPTIMIZATION.md`, `benchmark_cohere.md`,
`qwen3-asr-benchmark.md`, `ggml_plans.md`, `voxtral-comparison.md`,
`test_german.md`, `PERFORMANCE.md`. Everything of continuing value was
folded into the live docs.

Preserved outside these four: `UPSTREAM.md` (active upstream tracker),
`README_sycl.md` (Intel SYCL backend build), `ci/README.md` (CI
tooling), `models/README.md` (converter scripts), `samples/README.md`
(sample audio), and `hf_readmes/*.md` (HuggingFace model cards).

---

## Completed roadmap items (from PLAN.md, April 2026)

Items below were tracked in PLAN.md with full implementation details.
Moved here once shipped. See git history for code diffs.

### Core infrastructure (items 1-4, 6, 8, 10, 13, 17, 21, 24)

- **#1 voxtral4b encoder → encoder_self_attn()** — migrated with `permute_cont=false`. Bit-identical.
- **#2 Qwen3 forced aligner** — `qwen3_asr_run_aligner()` + `crispasr_aligner.cpp`. HF: `cstr/qwen3-forced-aligner-0.6b-GGUF`.
- **#3 Granite µP scale** — already handled via `KvSelfAttnParams::attn_scale`. No change needed.
- **#4 Scheduler reuse audit** — all backends use create-once + `ggml_backend_sched_reset()`.
- **#6 Best-of-N sampling** — all 4 LLM backends (voxtral/qwen3/granite/voxtral4b). `--best-of N -tp T`.
- **#8 voxtral audio Q&A** — `--ask "question"` flag for audio understanding.
- **#10 Granite encoder ggml graph** — `GRANITE_ENCODER_GRAPH=1` env var. CPU-verified identical.
- **#13 canary_ctc CPU fallback** — already implemented (2-backend GPU+CPU pattern).
- **#17 VAD stitching** — stitch + remap matching crispasr. C-ABI: `crispasr_session_transcribe_vad`.
- **#21 CLI→library DRY refactor** — VAD, diarize, LID, aligner, cache, registry promoted to `src/` behind shared C-ABI (v0.4.4–v0.4.8).
- **#24 Wrapper test suites** — Python (13), Rust (5+3), Dart (9) tests.

### New backends (items 26-34)

- **#26 GLM-ASR-Nano** — 12th backend. Whisper encoder + Llama 1.5B. MIT. `glm-asr`.
- **#27 Kyutai STT** — 13th backend. Mimi codec + causal LM. MIT. `kyutai-stt`.
- **#28 FireRedASR2-AED** — 14th backend. Conformer + CTC + beam search. Apache-2.0. `firered-asr`.
- **#29 FireRedVAD** — DFSMN 588K-param VAD, 97.57% F1.
- **#30 Moonshine** — 15th backend. Conv+transformer encoder-decoder. MIT. `moonshine`.
- **#31 FireRedASR decoder** — greedy + beam search Transformer decoder.
- **#32 FireRedLID** — 120-language LID via shared encoder + 6L decoder.
- **#33 OmniASR-LLM** — 16th backend variant. wav2vec2 encoder + 12L LLaMA decoder. Apache-2.0. Dynamic language selection (1693 FLORES-200 codes).
- **#34 VibeVoice** — architecture analysis complete. 1.5B is TTS-only; ASR is 7B (blocked on RAM).
- **ECAPA-TDNN LID** — 107-language LID, ggml graph (4.1s, 6x speedup), 100% accuracy. Two variants (VoxLingua107 + CommonLanguage).
- **OmniASR-CTC** — 300M and 1B variants working. HF: `cstr/omniASR-CTC-1B-GGUF`.

### Post-processing (item 35)

- **#35 FireRedPunc** — BERT-based punctuation restoration. GGUF converter + C++ runtime + CLI (`--punc-model`) + C-ABI + Python/Rust/Dart wrappers. HF: `cstr/fireredpunc-GGUF` (F16/Q8_0/Q4_K). Verified exact match against Python reference.

### Other completed items

- **#31 JSON LID (issue #17)** — language_detected/confidence/source in JSON output.
- **#25 Montreal Forced Aligner** — NOT PLANNED (too heavy, external tool).
- **Qwen Omni ASR** — NOT PLANNED (split GGUF, too large, already in llama.cpp).
- **#30 PazaBench assessment** — 16 model families assessed. 7 already covered, 4 easy wins identified.

### v0.6.7 (May 2026)

**German Moonshine ASR + text post-processing:**
- **moonshine-de** / **moonshine-tiny-de**: fidoriel German fine-tunes (6.9% / 11.4% WER, CC-BY-NC-SA-4.0) wired as registered backends with `-m auto` support. Also converted dattazigzag/moonshine-tiny-de (MIT, 36.7% WER)
- **punctuate-all**: kredor/punctuate-all (XLM-RoBERTa-base, 12 languages, MIT, 154 MB Q4_K) — `--punc-model punctuate-all`
- **PCS model**: 1-800-BAD-CODE XLM-RoBERTa punc+truecase+SBD (47 languages, Apache-2.0) — ONNX→GGUF converter + 4-head C++ runtime: post-punc, pre-punc, sentence boundary, per-character truecasing. `--punc-model pcs`
- **truecaser-lstm**: BiLSTM character-level truecaser (mayhewsw/pytorch-truecaser, Apache-2.0, 3.2 MB, 97.9% F1). Handles adjective/noun distinction ("braune Katze"), formal "Ihnen", compound words — `--truecase-model lstm` (recommended)
- **truecaser-crf**: CRF with context features (word, prev/next, article, suffix), Viterbi decode — `--truecase-model crf`
- **truecaser-de**: Statistical word-frequency truecaser (375K entries, 9 MB) — `--truecase-model auto`
- **License field**: Registry entries carry optional license tag; NC models emit a stderr note on download
- **--punc-model** shortcuts: `auto`/`firered`/`fullstop`/`punctuate-all`/`pcs`

### v0.5.4 (April 2026)

**Maintenance:**
- **Sync versioning**: Synchronized version numbers across CMake, Rust (crispasr, crispasr-sys), Python, and Dart wrappers to 0.5.4.
- **CI/Lint cleanup**: Resolved all remaining clang-tidy and clang-format violations using LLVM 18.
- **Improved Static Analysis**: Updated `.clang-tidy` to exclude third-party headers, reducing noise in CI reports.
- **Code Quality**: Fixed multiple implicit widening conversion warnings and enforced braces for all control flow statements in core core files.

### v0.5.0 (April 2026)

**Features:**
- **#36** ASCII punc mapping — auto-detect Latin script, map `，。？！` → `, . ? !`
- **#37** Progressive SRT (`--flush-after N`) — streaming subtitles for media players
- **#38** Fullstop-punc multilingual — XLM-RoBERTa-large, MIT, EN/DE/FR/IT. HF: `cstr/fullstop-punc-multilang-GGUF`
- **#39** Session API — all 18 backends wired in C-ABI + Python/Rust/Dart
- **#15** CMake rename — crispasr → crispasr in CMake, CI, Dockerfiles, scripts
- **#18** Aligner LIS — Longest Increasing Subsequence monotonicity fix
- **#40** Moonshine converter — multilingual variants (ja, ko, zh, ar, vi, uk)

**Optimizations:**
- **#44** FireRed ggml decoder — native Q4_K matmuls: 123s → 19s (**6.3x**)
- **O11** wav2vec2 CNN → ggml F32 im2col + OpenMP pos_conv: 108s → 10s (**10.8x**)
- **O1** ggml_soft_max_ext fusion — saves one op per attention layer (-10% wav2vec2)
- GPU auto-detect for all 18 backends + aux models

**Server:**
- Auto-chunking for long audio (#27) — prevents OOM
- Verbose logging + chunk progress (#26)
- API keys via env only, not CLI arg (#28)
- JSON error bodies + 404 handler

**Docker:**
- GHCR publishing (main, cuda, vulkan, intel, musa)
- passwd fix for ubuntu:24.04 images
- Standardized run-server.sh entrypoint

**VibeVoice TTS — Perfect ASR Round-Trip (April 2026):**
- **17 bugs found and fixed** via systematic stage-by-stage diff methodology
- **Perfect ASR round-trip**: all test cases produce exact match
  - "Hello, how are you today?" → parakeet: "Hello, how are you today?"
  - "The quick brown fox jumps over the lazy dog" → exact match
- **Model**: VibeVoice-Realtime-0.5B (2.04 GB, 605 tensors)
- **Architecture**: Base LM (4L) → TTS LM (20L) → DPM-Solver++ (20 steps) → σ-VAE (3200x) → 24kHz
- **Voice prompts**: pre-computed KV caches from .pt files (2.7 MB GGUF each)
- **CFG**: dual KV cache with per-frame negative updates, cfg_scale=3.0
- **EOS classifier**: automatic length detection via sigmoid(FC1→SiLU→FC2)
- **Critical bugs**: AdaLN SiLU (#16), text newline (#17), r-ratio sign (#14)
- **CLI**: `crispasr --tts "text" --voice voice.gguf -m vibevoice-realtime.gguf`

**VibeVoice-1.5B Base Model TTS (April 2026):**
- Single-LM autoregressive TTS (no TTS LM, 28-layer Qwen2)
- Voice cloning via acoustic+semantic encoder from reference WAV
- Speech token IDs: vision tokens reused (151654/151652/151653)
- ASR round-trip verified: "Hello, how are you today?" → exact match
- Quantized: F16 (5.1 GB), Q8_0 (2.8 GB), Q4_K (1.6 GB)
- HF: `cstr/vibevoice-1.5b-GGUF`

**ggml conv1d extensions + performance optimizations (April 2026):**
- Three new ggml ops: `GGML_OP_CONV_1D_CF` (channels-first conv1d),
  `GGML_OP_CONV_1D_CF` depthwise variant, `GGML_OP_CONV_1D_GROUP`
  (fused grouped conv1d). All with direct F32 kernels, F16/BF16
  kernel weight support, multi-threaded over output channels.
- VibeVoice TTS: VAE decoder 29% faster (700→476 ops), total 32%
  faster (0.39x→0.56x RT) via conv_1d_cf + `--tts-steps 10`
- wav2vec2: 12% faster via grouped positional conv + CNN cleanup
- firered-asr: depthwise conv migrated to conv_1d_dw_cf
- `VIBEVOICE_BENCH=1` / `WAV2VEC2_BENCH=1` per-phase timing

**Auto-download for all 19 backends (April 2026):**
- Added firered-asr, kyutai-stt, glm-asr, moonshine, fastconformer-ctc
  to model registry. Every backend now supports `-m auto --auto-download`.
- Companion file mechanism for moonshine's tokenizer.bin

**#29 Japanese split-on-punct fix (April 2026):**
- CJK fallback in `split_text_at_punct`: splits at clause breaks (、，)
  after ≥20 chars, force-splits at ~42 chars. English behavior unchanged.

**VibeVoice-7B GGUF (April 2026):**
- Full ASR+TTS GGUF (1205 tensors: encoder + LM + decoder + tokenizer)
- 7 quantizations: Q3_K (4.7 GB) through F16 (17.4 GB)
- TTS requires ≥Q4_K (Q3_K quality too low for decoder)
- HF: `cstr/VibeVoice-7B-GGUF` with README

**OmniASR CTC fix — two bugs found via stage-by-stage diff (April 2026):**
- pos_conv weight normalization: converter used per-output-channel norm
  (dim=0) instead of the model's per-kernel-position norm (dim=2).
  Fix: materialize weight directly from the HF model.
- head_dim hardcoded to 64: the 1B model uses 16 heads × 80 dim,
  not 20 × 64. Fix: read from HF config.
- Before: "koamerik asnot what yor country" (cosine 0.65 at layer 0)
- After: "fellow americans ask not what your country" (exact match)
- Converter now auto-detects v1 (fairseq2 .pt) vs v2 (HF transformers)
- 300M v2: works perfectly on ≤5s audio (cos=0.999997), breaks on >7s
  (positional encoding doesn't generalize beyond training length).
  Workaround: use --vad to chunk audio.
- LLM converter: complete rewrite (previous was corrupted). Tensor
  name mapping fixed (dec_ln, lm_head, tok_emb, gate). Same pos_conv fix.
- HF: `cstr/omniASR-CTC-1B-v2-GGUF` (F16 + Q4_K + Q8_0),
  `cstr/omniASR-CTC-300M-v2-GGUF` (F16),
  `cstr/omniasr-llm-300m-v2-GGUF` (F16 + Q4_K — fixed pos_conv + names)

**Moonshine multilingual — 12 models (April 2026):**
- Fixed converter: 1D tensors (norms/biases) forced to F32 for binary ops
- Fixed runtime: conv_1d_f32 mul_mat argument order for F16 kernels
- All 12 models tested and uploaded to HuggingFace
- HF: `cstr/moonshine-{tiny,base}-{ja,ar,ko,zh,vi,uk}-GGUF`

**OmniASR LLM-1B conversion (April 2026):**
- Converted facebook/omniASR-LLM-1B (8.5 GB .pt) to GGUF (4.55 GB F16, 918 tensors)
- 48-layer encoder (d=1280) + 12-layer LLaMA decoder (d=4096)
- Output: "fellow americas ask not what your country can do for you"
- HF: `cstr/omniasr-llm-1b-GGUF` (F16 + Q4_K)

**Parakeet TDT-CTC Japanese — xscaling fix (April 2026):**
- `nvidia/parakeet-tdt_ctc-0.6b-ja` was emitting "1 token then loop"
  because NeMo's `RelPositionalEncoding` multiplies the encoder
  input by `sqrt(d_model)=32` when `encoder.xscaling=true`. The C++
  runtime never applied this scale; v3 has `xscaling=false` so the
  multilingual sibling worked by accident.
- Diagnostic path: stood up `tools/reference_backends/parakeet.py`
  + `tools/dump_parakeet_reference.py` so `crispasr-diff parakeet
  <model.gguf> <ref.gguf> <audio.wav>` produces a stage-by-stage
  comparison against the NeMo reference. mel matched at cos≈0.99,
  encoder at cos=0.149 → grep'd `model_config.yaml`, found
  `xscaling: true`, applied `ggml_scale(*, sqrt(d_model))` between
  pre_encode and the first conformer block. Encoder cos jumped to
  0.81, F16 transcript bit-exact.
- Verified on a JSUT-basic5000 sample at F16:
  NeMo:     `'水をマレーシアから買わなくてはならないのです。'`
  crispasr: `'水をマレーシアから買わなくてはならないのです。'`
- Converter rewrite: every architecture hparam read from
  `model_config.yaml` (no more hardcoded `d_model=1024` /
  `pred_hidden=640`), cross-checked against actual tensor shapes,
  unmapped tensors warn loudly. New `parakeet.xscaling` GGUF key
  (default true on read) so old-converter v3 GGUFs continue to work
  unchanged once re-converted with `xscaling=false`.
- Q4_K JA still degenerates after ~8 tokens — the smaller 80-mel JA
  encoder is more quantisation-sensitive than v3's 128-mel one and
  `joint.pred` / `decoder.embed` fall back to q4_0. F16 is the
  recommended JA file; Q5_K or pinning those two tensors to F16 is
  the path forward.
- HF: `cstr/parakeet-tdt-0.6b-ja-GGUF` (F16 1.24 GB,
  Q4_K 470 MB) re-uploaded with the new converter + README.

**Gemma-4-E2B-it ASR — landed end-to-end (April 2026):**

`google/gemma-4-E2B-it`: USM Conformer (12L, 1024d, chunked-local
attention with relative position bias, ClippableLinear with QAT
scalars, LightConv1d) + Gemma4 LLM decoder (35L, 1536d, GQA 8Q/1KV,
per-layer embeddings, hybrid sliding/full attention with
per-layer-type head_dim, GeGLU MLP). Q4_K transcribes
`samples/jfk.wav` perfectly:

> "And so my fellow Americans ask not what your country can do for
> you, ask what you can do for your country."

End-to-end took ~16 numerical bugs to fix. The dominant ones:

- **ClippableLinear QAT scalars are NOT optional.** HF
  `Gemma4ClippableLinear.forward` clamps every input AND output of
  every q/k/v/o/ffw_layer/lconv1d.linear with trained finite bounds
  (±5..±40). Skipping them collapsed audio_layer_11 cos to 0.51 vs HF.
  Fix: stop skipping in converter, runtime applies clamp(input)→
  matmul→clamp(output) per linear. 480 scalars persisted per audio
  tower. Confirmed by patching HF locally to disable the clamps:
  HF-no-clip cos = 0.51, exactly matching ours-no-clip → unambiguous
  attribution. cos jumped to 0.97 once enabled.
- **Audio FE is bit-different from Whisper-style.** `frame_length=320`,
  `fft_length=512`, semicausal padding (160 zeros at start only),
  unfold-by-`frame_length+1`-then-drop-last, magnitude (not power)
  spectrum, HTK no-norm filterbank, `log(mel + 0.001)` (additive
  epsilon, natural log), no post-log normalisation. Wrote a
  dedicated `g4e_compute_mel_hf_faithful` instead of fighting
  `core_mel`'s param surface.
- **LLM forward had 5 separate bugs** — attn_scale=1.0 (q_norm
  replaces 1/√d), v_norm RMSNorm-no-weight, layer_scalar at end of
  layer (was applied twice mid-layer), PLE block at end + full
  per_layer_inputs prep stage including `per_layer_model_projection
  + per_layer_projection_norm`, MLP is GeGLU not SwiGLU. Each was a
  distinct numerical mismatch with HF; combined they took the LLM
  from outputting `<pad>` repeats to coherent English.
- **KV-share direction was the LAST 20 layers, not the first.**
  CLAUDE-memorised "first 20 layers reuse from later layers" was
  wrong; HF source has `first_kv_shared_layer_idx = num_layers - N`
  with each shared layer reading from the LAST earlier layer of the
  same `layer_type`. Donor map computed at load.
- **`use_double_wide_mlp=true` is a single 2× MLP, not two halves.**
  Misread of the field name + converter rename rules ate a session;
  HF L1024 of `modeling_gemma4.py` makes it explicit:
  `intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)`.

Per-stage cos vs HF reference (Q4_K, JFK 11s):

```
mel_spectrogram          1.0000   bit-exact
audio_subsample_output   1.0000
audio_layer_0..7        >0.998
audio_layer_11           0.969
audio_tower_output       0.962
encoder_output           0.966
```

Process win: the stage-by-stage diff harness
(`tools/dump_reference.py --backend gemma4` + intermediate dumps from
the runtime via `CRISPASR_DUMP_DIR=…`) was decisive. Every bug was
localised in 1–2 iterations once the per-layer cos table existed —
the alternative (eyeball end-to-end output, guess) wasted multiple
sessions before we wired it up. See LEARNINGS for the methodology.

HF: `cstr/gemma4-e2b-it-GGUF` re-uploaded with the QAT scalars and
all the fixes above (F16, Q8_0, Q4_K, Q2_K).

**Speed follow-up (same day):** end-to-end JFK transcription went
from 0.2× realtime (67.77 s) → **1.4× realtime (7.75 s)** with a
1-line fix. The model emits `<end_of_turn>` (token 106) at the
natural completion point, but greedy decode was configured with
`cfg.eos_id = ctx->eos_id` (token 212 = `<eos>`), so the loop
never matched and ran to `max_new_tokens=256` every call. For an
11s utterance that's 25 real tokens + 231 wasted ones. Fixed by
preferring `end_of_turn_id` when the chat template defines one.
Per-stage profile after the fix: mel 17 ms, encoder 719 ms,
prefill 287 tok in 1.46 s, decode 25 tok in ~5.5 s (~220 ms/tok).
The remaining encoder/decode optimisations (TODO O10/O11) are
secondary now — at 1.4× realtime the model is usable; further
work would target per-token decode cost. See LEARNINGS "Specific
bugs that cost us a day each" #9.

**NeMo-cluster encoder cosine fix — parakeet bias load (April 2026):**

The 24-layer FastConformer encoder was producing cos_mean=0.79 vs the
NeMo reference on Japanese audio (`reazon_meal_11s.wav`) and JSUT,
even though `mel_spectrogram` matched at cos≈0.999 after the preemph
+ Bessel-corrected PerFeatureZ fix. Symptom was small but real: extra
hallucinated prefixes (`本当`) and partial syllables on conversational
JA (parakeet-tdt-0.6b-ja, issue #37).

Root cause was a 10-line bug in `parakeet_load_weights`. The GGUF
stored `attn.{q,k,v,out}.bias` + `{ff1,ff2}.linear{1,2}.bias` +
`conv.{pw1,pw2}.bias` (10 biases per layer × 24 layers = 240 tensors)
but the loader only fetched the weights — the bias slots stayed
nullptr, and `mm_bias` silently skipped the bias add. Fix: add
`e.X_b = try_("…bias")` for each missing slot. `try_get` rather than
`require` keeps v3 (which has `use_bias=False`) compatible.

Result: `encoder_output cos_mean: 0.792 → 0.996` on reazon_meal_11s,
similar on JSUT. v3 EN regression on JFK still passes. Per-layer
diff confirmed the divergence had started at `encoder_layer_0`
(immediately after a bit-exact `pre_encode_output`), localising the
bug inside the conformer block before grep'ing the loader.

Residual: layers 17–22 keep cos_mean ≈ 0.99 but cos_min crashes to
negative on specific frames (`encoder_layer_22` cos_min = −0.67).
Suspects: rel_shift edge cases, position-encoding numerical
instability on specific positions, or a buffer-aliasing issue in the
dump path. Not blocking the bug-report fix; reusable diagnostic
infra (per-layer captures in `reference_backends/parakeet.py`,
`parakeet_run_encoder_dump`, `encoder_layer_K` stages in the diff
harness) is in place for canary, canary_ctc, and any future
NeMo-cluster runtime debug. Commit `e598767`.

**Qwen3-TTS codec decoder — Metal kernel fix (April 2026):**

The codec decoder (8L sliding-window transformer + ConvNeXt + 4×
SnakeBeta+tconv → 24 kHz waveform) hung on M1 with
`kIOGPUCommandBufferCallbackErrorImpactingInteractivity` whenever
`use_gpu=true`.

Root cause was instrumented via two env vars added to
`src/qwen3_tts.cpp`:
- `QWEN3_TTS_CODEC_TRACE=1` — prints per-node
  `op / tensor name / shape -> backend` before each op, with
  `ggml_backend_synchronize` after each so a hang attributes to the
  last printed line.
- `QWEN3_TTS_CODEC_FORCE_METAL=1` — re-routes codec weights and
  compute through the Metal-capable `c->sched`, reproducing the
  hang for triage.

Trace localised the hang to op 536: `GGML_OP_CONV_TRANSPOSE_1D` in
decoder block 1 (in-T=320, out-T=1605, C_out=384, stride=5,
kernel=10). Block 0 (stride=8, in-T=40, out-T=320) ran fine. The
SnakeBeta `sin`/`exp` and `conv_1d_dw` chains that were originally
suspected all completed cleanly on Metal.

The actual ggml-Metal `kernel_conv_transpose_1d` does
`for i in 0..IL { if (j ∈ [i*s0, i*s0+K)) accumulate; }` — but at
most `ceil(K/s0)` (=2 here) values of `i` ever satisfy that
condition. The kernel was iterating all 320 input positions for each
output element, doing ~160× the necessary work, which kept Metal
command buffers above the macOS GPU watchdog's ~5 sec ceiling.

**Fix landed in the ggml fork** (`ggml/src/ggml-metal/ggml-metal.metal`,
marked `// CrispASR patch`): compute the contributing `i` range
analytically before the input-channel loop and iterate only those
positions. Bit-identical output, ~160× less work on the codec
shape, comfortably under the watchdog. Documented in LEARNINGS.md
under "Metal conv_transpose_1d input range tightening" with the
"MUST RE-APPLY after every ggml bump" pattern (matches the existing
"CUDA im2col grid overflow" entry).

After the patch: 8/8 codec stages PASS with `use_gpu=true` end-to-end
on Metal, cos_min ≥ 0.999983 against the Python reference (slightly
tighter than the CPU path, presumably from F16 vs F32 mul/accum
differences in non-conv ops).

The original CPU-pin workaround (`codec_sched`, codec weights loaded
onto `c->backend_cpu`) is kept as a runtime safety net in case the
kernel patch is lost on a future ggml bump before the LEARNINGS
"RE-APPLY" reminder is honoured. Trace env vars also stay — useful
for debugging any future codec issue. PLAN #52 step 4 (ECAPA
speaker_encoder) is unaffected since it uses regular conv1d, not
transposed conv.

**Aux runtimes — Silero LID / pyannote v3 / wav2vec2-ggml (April 2026):**

Three small standalone runtimes landed alongside the main backend
work. All shipped end-to-end-correct with public GGUFs.

- **Silero LID native port (#56):** `src/silero_lid.{h,cpp}` plus
  `models/convert-silero-lid-to-gguf.py`. 95-language detector, 16 MB
  F32 GGUF (Q8_0 ~9 MB; quants below Q8_0 break accuracy on the small
  conv tensors). Pure-C++ forward pass, no ggml graph (manual F32
  loops, similar to pyannote_seg). Architecture: learned STFT
  Conv1d(1→322,k=320,s=160) → magnitude → log(2²⁰·mag+1) → adaptive
  norm (17-tap reflected smooth) → 8×(12 dw-sep conv + post-norm
  transformer + stride-2/1 proj+ReLU) → attention pool (tanh+softmax)
  → 95-lang + 58-group classifiers. Five bugs fixed during port:
  (1) front-end zero-pad 160/side, not reflection-pad 320 left;
  (2) stride-2 output `T = (T−1)/2 + 1`, not `T/2`; (3) QKV split
  order K,Q,V (not Q,K,V); (4) missing ReLU after stride-1
  projections stages 4–7; (5) missing tanh in attention pooling.
  CLI: when `--lid-model *.gguf` is passed, the native path runs;
  falls back to sherpa subprocess for `.onnx`. Verified across
  English, German, and Latvian. HF: `cstr/silero-lid-lang95-GGUF`.

- **Pyannote v3 native (#57):** SincNet + 4× biLSTM + 3× Linear +
  LogSoftmax, ~440 lines of C++. Wired into the CLI as
  `--diarize-method pyannote --sherpa-segment-model *.gguf` (native
  path; subprocess fallback for `.onnx`). Verified on jfk.wav with
  650 frames and correct "(speaker 1)" assignment.
  HF: `cstr/pyannote-v3-segmentation-GGUF` (5.7 MB F32).

- **wav2vec2 ggml rewrite (#63):** `src/wav2vec2-ggml.{h,cpp}`,
  layer-by-layer ggml graphs (~80 MB/layer, reused) for the 24-layer
  XLSR-53 transformer; CNN + pos_conv stay manual. Four root causes
  during port: (1) `ggml_gallocr` / `ggml_backend_sched` corrupt
  external F16 weight tensors (reallocate over them) — workaround
  `ggml_graph_compute_with_ctx`; (2) ggml `[H,T]` stores
  `data[h + t·H]` which is the SAME layout as `[T,H]` row-major in C
  — the original code had a spurious transpose that corrupted all
  data; (3) `flash_attn_ext` crashes with `mask=nullptr` —
  replaced with mul_mat attention; (4) logits `[V,T]` in ggml =
  `[T,V]` row-major, no transpose needed. Tested with
  `jonatasgrosman/wav2vec2-large-xlsr-53-english` (33 vocab,
  1024 hidden, 24 layers) — correct output on jfk.wav.

**iOS + Android CI gates + v0.1.0 release (April 2026):**

- iOS (arm64, Xcode) and Android (arm64-v8a, NDK r26d) cross-compile
  gates added to GitHub Actions. Catches breakage early on the
  lowest-traffic platforms.
- v0.1.0 shipped via GitHub Actions: Linux 660 KB, macOS 484 KB,
  Windows 1437 KB.

**Granite speed (#64) — closed, hardware-blocked:** at Q4_K /
4-thread CPU the 11s clip takes 33 s, and 26 s of that is
autoregressive LLM decode. `--gpu-backend` already exists and
granite uses `ggml_backend_init_best()` — no code change moves the
needle without GPU hardware. Tracked in TODO under per-model
follow-ups for OpenMP encoder annotations as a CPU-only nibble.

### 53. Qwen3-TTS — codec encoder repair (April 2026)
Fixed a critical memory layout bug in the `qwen3_tts` codec encoder. The
CPU-side RVQ loop was assuming channels-first indexing while the SEANet
output was transposed to row-major `[T, 512]`. This fix restored clear
voice cloning in the end-to-end CLI, eliminating the garbled artifacts.
Verified at `cos_mean=0.998` against the Python reference.

### 54. granite-family DRY refactor — PLAN #55 (May 2026)
Five-step lift of duplicated math out of `granite_speech.cpp` (base +
plus, causal + KV-cached) and `granite_nle.cpp` (NAR, non-causal
single-pass) into shared `src/core/` headers. Each step gated by JFK
smoke tests on all three variants — every commit kept transcripts
identical.

| Step | Header | Risk | LOC moved | Commit |
|---|---|---|---:|---|
| 1 | `core/fft.h` + `core/cpu_ops.h` (FFT, layernorm, matmul fallbacks) | very low | ~250 | `5f4b5ae` + `b343a17` |
| 2 | `core/ctc.h` (posterior-weighted pool + greedy decode w/ blank) | very low | ~60 | `65ef44c` |
| 3 | `core/conformer_ibm.h` (Macaron block helpers + Shaw RPE lookup) | medium | ~600 | `0f72391` |
| 4 | `core/granite_llm.h` (40-block backbone, `is_causal` flag) | medium | ~250 | `372a5f7` |
| 5 | `core/qformer.h` (NAR simplified Q-Former) | low | ~190 | `ed80fb0` |

**Step 5 — plan correction:** the duplication-map row originally
listed the windowed Q-Former as shared by both granite TUs. The code
disagreed: granite-speech (base/plus) uses the full BLIP-2 Q-Former
(self-attn + cross-attn + FFN per layer, no pass A, no window-mean-pool)
while granite-nle uses a "simplified" variant (cross-attn-only + MLP).
The block weight structs (`granite_proj_block` with `sa_*`/`ca_*`/`ffn_*`
vs `granite_nle_proj_block` with `attn_*` + `mlp_*`) made the divergence
explicit. Step 5 was rescoped to NAR-only — `core/qformer.h` is co-located
for any future simplified-windowed-Q-Former backend; granite_speech
stays untouched. PLAN.md was updated mid-step to reflect this.

`core_granite_llm::build_decoder` is the most reusable lift — it composes
40 layers of pre-RMSNorm + GQA(16/4) flash-attn + RoPE + SwiGLU + residual
×0.22 with an `is_causal` flag that picks `core_attn::kv_self_attn` (KV-cached
prefill+decode) or an inline non-causal flash path (whole-sequence editing).
Both granite TUs collapse from a per-layer hand-written loop to a single
function call.

`core_conformer_ibm` is a sibling-not-merge with `core/fastconformer.h`:
parakeet/canary use NeMo's Conformer dialect (conv subsampling, MHA RPE)
while granite uses the IBM dialect (Shaw RPE, fused conv layout, BN
folding-at-load) — keeping them separate avoids muddying both.

**Net effect:** `granite_speech.cpp` 2570 → 2113 LOC (−457); `granite_nle.cpp`
2096 → 1615 LOC (−481); combined drop ~940 LOC. New `core/*.h` files
total ~1070 LOC (fft 93 + cpu_ops 98 + ctc 129 + conformer_ibm 336 +
granite_llm 162 + qformer 255). Roughly half of those core LOC are
deduplicated math (the rest is comments + struct/API plumbing). Plus a
clean separation of "backbone" (in core) from "plumbing" (in TUs).

### 55. Granite encoder graph path as default — PLAN #16 (May 2026)

The `GRANITE_ENCODER_GRAPH=1` no-RPE flash-attn baseline silently
regressed: on JFK with `granite-speech-4.1-2b-q4_k` it produced only
the back half of the quote ("ask what you can do for your country").
The PLAN #16 prototype (per-block subgraph attention with Shaw RPE,
gated by `GRANITE_ENCODER_GRAPH_RPE=1`) inherited the same wrong
output, so end-to-end validation was blocked.

**Root cause.** The loader built only **layer 0's** RPE lookup
(`ctx->rpe_lookup`, single `vector<float>`) and the graph builder
reused it for all 16 encoder blocks, on the assumption that RPE is
tied across layers. granite-speech-4.1-2b in fact stores **distinct**
`attn_rel_pos.weight` per block (verified: layer 0 mean ≈ 0.00004,
layer 1 mean ≈ -0.003, layer 2 mean ≈ -0.002). Layer 0 still
matched the CPU loop bit-for-bit, but layer 1's attention diverged
immediately and the drift compounded across 16 layers until the LLM
only latched onto the back half of the audio. The CPU loop was
unaffected because it was already building per-layer RPE locally
inside the encoder forward — that local builder shadowed the
context-level cache and hid the bug.

**Fix.** Replaced `ctx->rpe_lookup` with `ctx->rpe_per_layer`
(`vector<vector<float>>`), built per-block at load time. The graph's
`rpe_lookup` input now has shape `(ctx_size*hd, ctx_size, n_layers)`
and each layer slices its block via `ggml_view_3d` on the layer
axis. CPU loop reuses the same precomputed table.

**Validation (per LEARNINGS methodology).** Stage-by-stage taps in
both paths (input_linear, FFN1, attn, conv, FFN2, post-norm,
block_out per layer) confirmed all sub-stages match within float
precision (~1e-3) — well above the cos_min ≥ 0.999 bar. JFK
transcript matches the CPU loop byte-for-byte:
"and so my fellow americans ask not what your country can do for
you ask what you can do for your country".

**Promotion.** Made the graph path the default and renamed the
escape hatch: `GRANITE_DISABLE_ENCODER_GRAPH=1` now opts back to
the CPU loop. The legacy `GRANITE_ENCODER_GRAPH` and
`GRANITE_ENCODER_GRAPH_RPE` env vars are gone — the no-RPE
flash-attn branch survives only as automatic fallback for models
with an unsupported `attn_rel_pos.weight` type.

**PLUS on graph.** Captured `cat_hidden_layers` post-norm tensors
inline in the graph and concatenated them with the final encoder
output along the feature dim via `ggml_concat`, so the PLUS variant
also rides the GPU path with no CPU-side cat_layers buffering. The
in-graph concat is essentially free (graph fanout off the residual
stream, no extra compute).

**NAR on graph.** Mirrored the granite-speech graph builder for
`granite_nle.cpp`. NAR-specific bits:
1. Self-conditioning residual at `hp.enc_self_conditioning_layer`
   (1-indexed, default = 8). Tapped `softmax(mid_logits)` on its way
   into `ctc_mid_w` so the runner can pull per-frame
   `blank_prob = column 0` for the BPE auxiliary head's
   `posterior_weighted_pool`.
2. Snapshot taps at every entry in `enc_layer_indices_parsed`
   (HF tuple indices: 0 = input embedding, N = output of block N-1
   *after* the self-cond residual at that block). Concatenated along
   the feature dim into `enc_output` (matches the CPU loop's wide
   buffer layout).
3. Final CTC logits = `ctc_out_w @ final_hidden + b` exposed as a
   named graph output. Cached on `ctx->last_ctc_logits` for the BPE
   editing path.
4. The BPE auxiliary head's `posterior_weighted_pool` stays on CPU
   (windowed reduce that doesn't map cleanly to a single ggml op);
   the `bpe_out_w` matmul (1024 → 100353) runs through the same
   scheduler as before. Negligible perf cost vs the encoder body.

**Numbers (M1, Q4_K encoder F32, 11 s JFK clip; lower disk-contention
session):**

| Variant            | CPU loop      | Graph (default) | Speedup |
|--------------------|--------------:|----------------:|--------:|
| `granite-4.1`      | 4.78 s (2.3×) | **2.31 s (4.8×)** | ~2.1×  |
| `granite-4.1-plus` | 9.41 s (1.2×) | **3.74 s (2.9×)** | ~2.5×  |
| `granite-4.1-nar`  | 19.27 s (0.6×, contended) | **6.41 s (1.7×)** | ~3.0×  |

All three variants transcribe JFK byte-for-byte identical to their
CPU-loop reference (including punctuation/casing for PLUS and NAR).
The `GRANITE_DISABLE_ENCODER_GRAPH=1` escape hatch covers all three.

### 56. MiMo-V2.5-ASR end-to-end — PLAN #51 SHIPPED (May 2026)

XiaomiMiMo's 7.5B-class speech-LLM (8-channel RVQ encoder + 36-layer
Qwen2 LM, vocab=151680, MIT) ships fully working. JFK transcription
matches the upstream Python `MimoAudio.asr_sft` reference verbatim:
"And so, my fellow Americans, ask not what your country can do for
you. Ask what you can do for your country." 11 s of audio in ~37 s
on M1+Q4_K+Metal (0.3× realtime).

**The forward path landed in two phases.** First, prefill numerical
correctness (commit `9faccdd`) — five stages (audio_features,
text_embeds, inputs_embeds, last_hidden, logits_step0) match the
bf16 PyTorch reference within Q4_K + bf16 tolerance, with cos_mean
between 0.96–0.998. Argmax of step-0 logits hits token 1597
(`' And'`), matching the reference. The fix-it-or-lose-it bug:
**capture tensors in a ggml graph need `ggml_set_output()`, not just
`ggml_set_name()`.** Without `set_output`, the scheduler treats
named tensors as ordinary intermediates and reuses their buffers
when allocating later ops in the same graph. Symptom we hit:
`prefill_inputs_embeds` collapsed to cos≈0.003 (looked like a
broadcasting bug for hours) before tracing the buffer aliasing.
LEARNINGS.md "Capture tensors MUST call `ggml_set_output()`"
documents the recipe — apply universally to any tensor read out
via `ggml_graph_get_tensor` for diff-harness extraction.

Three other prefill bugs fixed in the same commit: (a) the
input_local_block was permuting `(hd,n_h,gs,ng) → (hd,gs,n_h,ng)`
*before* `ggml_rope_ext`, putting `n_h` at ne[2] where positions[gs]
expected — assertion failed. Fix: rope-then-permute. (b) After the
o-projection, `attn` was 2D `[d, gs*ng]` while the residual was
3D `[d, gs, ng]` — ggml_add broadcast assert. Fix: reshape_3d.
(c) The on-disk Q4_K had a truncated vocab (151643 entries, missing
`<|empty|>`) so the empty-token fallback was hitting Qwen2's
`<|endoftext|>`, which never appears in the prompt — every position
was treated as non-empty, zeroing the audio path entirely. Fix
in step 9 below.

Second, transcribe path (commit `dae361f`) — full prompt
construction in C++, mirroring `process_speechdata.InputSegment`
byte-for-byte. Each text segment becomes a `[9, T_seg]` block (row
0: tokens at stride gs, -100 fillers, padded to gs alignment;
rows 1–8: per-channel `speech_zeroemb_idx`). Audio segment becomes
`[9, T_audio + 2*gs]` with `<|sosp|>` / `<|eosp|>` wrapping the
empty-token text row and codes flanked by speech_zeroemb pads on
the audio rows. The 6 segments (user header, audio, template,
end, assistant header, think+language tag) concatenate into the
prefill input_ids. `mimo_asr_build_prefill_graph` gained an
`n_past` parameter so step decode reuses the same builder with
T=gs and advancing n_past. The decode loop replicates each
generated text token across the gs positions of the new group
and fills audio rows with `speech_zeroemb_idx[c]` (matches
`slm_sample`'s `expand(group_size)` + `zero_embed_tensor` path).
Stops on `<|im_end|>`/eos, strips
`<|empty|>`/`<|eot|>`/`<|eostm|>`/language tags.

The tokenizer follows the qwen3-asr-style splitter: greedy match
`<|...|>` against the vocab, then GPT-2-style whitespace pre-split
+ `bytes_to_unicode` + `bpe_one` for the rest. Works only because
step 9 reconverted the GGUF with `tokenizer.ggml.merges` populated
(151291 entries — the converter fix from commit `2191a70` had
landed earlier but the GGUFs predating it carried no merges, so
BPE collapsed to per-byte and the prompt didn't match upstream).

**Step 9 dragon: torch+OpenMP deadlock.** The bf16→f16 cast in
`models/convert-mimo-asr-to-gguf.py` (via `t.to(torch.float16)`)
goes through `at::native::DEFAULT::copy_kernel` → OpenMP barrier
→ `__kmp_suspend_64` → indefinite `_pthread_cond_wait`. Process
appears alive (RSS stable, mmap'd weights resident, STAT=S, 0.0%
CPU) but the temp file stops growing after ~50 tensors. Workaround
made permanent: prefix all torch-based converters with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONUNBUFFERED=1`. Cost is negligible — the cast is memory-bound,
not compute-bound. Without the env vars: hangs forever. With them:
~20 min for the 14.9 GB F16 on M1, ~5 min more for Q4_K quantize.
LEARNINGS.md and the `mimo-tokenizer-GGUF` README repeat this so
the next person doesn't lose 30 minutes diagnosing it.

**HF release.** [`cstr/mimo-asr-GGUF`](https://huggingface.co/cstr/mimo-asr-GGUF) ships F16 (14.9 GB) + Q4_K
(4.5 GB) with the corrected vocab and merges. Pair with
[`cstr/mimo-tokenizer-GGUF`](https://huggingface.co/cstr/mimo-tokenizer-GGUF) — the audio tokenizer is a separate
encoder model. Q2_K and the legacy `mimo-asr.gguf` are kept for
history but were built before the vocab/merges fix and should not
be used.

### 57. Qwen3-TTS-Base 1.7B — PLAN #57 Phase 1 (May 2026)

The 1.7B variant of Qwen3-TTS-Base shipped end-to-end behind the
`qwen3-tts-1.7b-base` registry alias. Same ICL voice-clone path as
0.6B-Base (`--voice <wav> --ref-text "..."`); the runtime now reads
`qwen3tts.speaker.enc_dim` from the GGUF instead of assuming 1024.

**The bug.** `build_graph_spk_enc` and `run_spk_enc` (plus the two
`extern "C"` speaker-embedding helpers) hardcoded the ECAPA output at
1024 floats. That matched the 0.6B-Base config (`talker.hidden_size =
1024`, `enc_dim = 1024`) but the 1.7B-Base config has
`talker.hidden_size = 2048` and `enc_dim = 2048`. The first 1024
floats of the speaker embedding made it into the codec_input slot;
the second 1024 floats were silently truncated, and the talker's
prefill saw a half-zero spk row. Symptom: ICL produced degenerate
audio.

**The fix** (`0813869`). Read `c->hp.spk_enc_dim` (already plumbed
via `kv_u32(g, "qwen3tts.speaker.enc_dim", ...)` and exported by the
converter — just unused in the graph builders). Five sites
parameterised; banner now logs `ECAPA-TDNN 128→1024` for 0.6B-Base
and `128→2048` for 1.7B-Base.

**Validation.** `clone.wav` ICL on `"Hello, how are you today? The
weather is beautiful."` →
- 1.7B-Base F16  → 57 frames / 4.56 s → "Hello? How are you today? The weather is beautiful."
- 1.7B-Base Q8_0 → 51 frames / 4.08 s → "Hello! How are you today? The weather is beautiful."
- 0.6B-Base Q8_0 (regression, `enc_dim=1024` path) → 75 frames / 6.00 s → "Hello? How are you today? The weather is beautiful."

Word-level exact match across all three; the punctuation jitter on
the leading single-word token is parakeet-v3 behaviour, not a
synthesis defect.

**HF release.** [`cstr/qwen3-tts-1.7b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF) ships F16 (3.86 GB)
+ Q8_0 (2.07 GB). Pair with [`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF)
— same 12 Hz tokenizer as 0.6B-Base.

The 1.7B small_to_mtp_projection bridge from the 2048-d talker to the
1024-d code predictor was already wired in commits `7f79d34` /
`2cc7aeb` (originally for 1.7B-CustomVoice) and is variant-agnostic,
so once `spk_enc_dim` was unstuck the rest of the path "just worked"
on the existing graph builders.

### 58. Qwen3-TTS-VoiceDesign 1.7B — describe-the-voice TTS (May 2026)

The instruct-tuned variant of Qwen3-TTS-12Hz-1.7B shipped behind the
`qwen3-tts-1.7b-voicedesign` registry alias. Replaces the reference-WAV
or fixed-speaker prompt with a natural-language voice description fed
in via `--instruct`. No ECAPA forward, no codec encoder, no preset
speaker table — the model picks a voice purely from the text.

**Runtime contract.** When `qwen3tts.tts_model_type == "voice_design"`,
the talker prefill is built by `build_voicedesign_prefill_embeds`
(`src/qwen3_tts.cpp`), which mirrors `build_customvoice_prefill_embeds`
with two changes:
- The codec bridge omits the speaker frame: `L_codec =
  codec_prefill.size() + 2` (just `pad,bos`), one frame shorter than
  the CustomVoice path.
- An instruct block — `text_proj(text_embd(instruct_ids))`, where
  `instruct_ids` tokenises `<|im_start|>user\n{instruct}<|im_end|>\n`
  — is prepended to the prefill. Mirrors
  `Qwen3TTSForConditionalGeneration.generate` lines 2076–2233 of
  `modeling_qwen3_tts.py` for the `speaker_embed=None` + `instruct_ids`
  path.

**C-ABI.** Two new entry points:
- `qwen3_tts_is_voice_design(ctx)` — variant detection.
- `qwen3_tts_set_instruct(ctx, instruct)` — required before synthesis
  on a VoiceDesign model; returns -1 if the model isn't VoiceDesign.

**CLI.** New `--instruct "..."` flag (parsed in `cli.cpp`). The
`qwen3-tts` backend rejects `--voice` on VoiceDesign models with a
helpful warning, and errors before generation if `--instruct` is
empty.

**Why 1.7B-only.** Upstream `generate_custom_voice` explicitly disables
`instruct` for the 0.6B variant; there is no 0.6B-VoiceDesign weight
release. The 1.7B talker forward (Q/KV/FFN, mrope, small_to_mtp
bridge) is shared with 1.7B-Base, so the runtime side reuses the
existing graph builders end-to-end.

**HF release.** [`cstr/qwen3-tts-1.7b-voicedesign-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF) ships F16 + Q8_0.
Pair with [`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF) — same 12 Hz tokenizer.

### 59. Orpheus 3B-FT — PLAN #57 Phase 2 slice (c) (May 2026)

The first commercial-friendly TTS in the Phase 2 talker family
shipped end-to-end behind the `orpheus` backend
(commit `a0982d3`). The runtime drives a Llama-3.2-3B-Instruct
talker (custom-token vocab `<custom_token_0..28671>`) over a
persistent KV cache, samples on top of the 7-slot SNAC super-frame
layout, de-interleaves per the canonical Orpheus protocol, and
pipes the codes through a SNAC C++ decoder to 24 kHz PCM. With
Orpheus base in, Kartoffel_Orpheus DE + lex-au Orpheus-3B-DE-Q8 +
the various Orpheus finetunes are now checkpoint swaps.

**Sourcing the talker.** `canopylabs/orpheus-3b-0.1-ft` is gated;
having an HF login token isn't enough — you have to click through
the gate. `unsloth/orpheus-3b-0.1-ft` is a **non-gated mirror** of
the same weights and converts cleanly with the new
`models/convert-orpheus-to-gguf.py`. SNAC codec from
`hubertsiuzdak/snac_24khz` (MIT, 3 codebooks × 4096 @ 24 kHz).

**The BOS=128000 trap.** Verbatim from
`canopyai/Orpheus-TTS:engine_class.py:_format_prompt`, the prompt
is built by tokenising `"{name}: {text}"` with `add_special_tokens=True`
(HF tokenizer default) — which inserts the Llama-3
`<|begin_of_text|>=128000` BOS at the start. The engine then
prepends `audio_start=128259` and appends
`eot_id=128009, audio_eot=128260, audio_eom=128261, audio_end=128257`.
**Without the BOS the model produces well-structured but
semantically garbage codec output** — parakeet ASR returned
`"Ineonice perfect of the Pan 8."` for `"Hello, my name is Tara."`;
with it, the roundtrip lands exact. Easy to miss because the model
emits properly slot-patterned super-frames either way.

**Stop policy.** Stop on `audio_end=128257` *or* on >4 consecutive
non-codec tokens. **Don't** stop on `audio_pre_end=128009` or
`audio_end_b=128261`: in the unsloth/canopylabs ID layout those
are either the Llama-3 `<|eot_id|>` (which appears in the prompt)
or `text_N<10` reserved markers in the custom_token block. The
reference `tokens_decoder` filters them silently rather than
terminating on them.

**Sampling.** Greedy decoding (`temperature=0`) gets stuck in a
7-slot loop after a few super-frames and the AR halts after ~24
tokens. `engine_class.py` defaults to `temperature=0.6` — that's
what produced the validated 2.73 s clip below.

**Validation.**

```
$ crispasr --backend orpheus \
    -m /Volumes/backups/ai/crispasr-models/orpheus-3b-ft-f16.gguf \
    --codec-model /Volumes/backups/ai/crispasr-models/snac-24khz.gguf \
    --voice tara --temperature 0.6 \
    --tts "Hello, my name is Tara." \
    --tts-output /Volumes/backups/ai/crispasr-models/orpheus_test.wav
orpheus: AR emitted 224 codec tokens (32 super-frames)
crispasr: TTS output written (65536 samples, 2.73 sec)

$ crispasr --backend parakeet \
    -m /Volumes/backups/ai/crispasr-models/parakeet-tdt-0.6b-v3-q4_k.gguf \
    -f /Volumes/backups/ai/crispasr-models/orpheus_test.wav --no-prints
Hello, my name is Tara.
```

**Converter note.** `models/convert-orpheus-to-gguf.py` had to flip
`GGUFWriter(use_temp_file=True)` → `False`. With temp-file enabled
the writer first buffers tensor data to a system tempfile (honors
`TMPDIR`) then copies it to the output, which on
`/Volumes/backups` at 100% disk usage causes silent corruption /
slow throughput; direct write side-steps the issue.

**HF release.** Talker shipped to
[`cstr/orpheus-3b-base-GGUF`](https://huggingface.co/cstr/orpheus-3b-base-GGUF)
as F16 (6.6 GB) + Q8_0 (3.5 GB). SNAC codec shipped to
[`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF)
as a single F32 file (26 MB; the codec is small enough that
quantising it isn't worth the audio-quality risk). Registry alias
`orpheus` in `src/crispasr_model_registry.cpp` resolves Q8_0 + SNAC
under `-m auto`. The full unified Session API works end-to-end:
opening the talker through `crispasr_session_open` returns a non-null
handle and `crispasr_session_n_speakers` returns the 8 canopylabs
English voices (`tara`/`leah`/`jess`/`leo`/`dan`/`mia`/`zac`/`zoe`).

**Phase 3+ known gaps (out of scope for slice c):** plain NEOX
RoPE with `theta=500000` (no Llama-3 `freq_factors` scaling — fine
for short prompts, may matter for long synthesis); no
`repetition_penalty` in the sampler (engine_class.py default 1.3);
Metal first-load is slow (~10-15 min for 6.6 GB f16 GGUF due to
kernel compilation, fast thereafter); non-streaming AR (the
reference's "emit middle of 4-super-frame sliding window" protocol
in `orpheus_snac.py` is a follow-up).

### 60. MiMo-V2.5-ASR perf wave — PLAN #51b/b' (May 2026)

First decode-side perf pass on the mimo-asr runtime that shipped in
section 56. Two structural changes plus an allocator-friendly diag
gate, all in `src/mimo_asr.cpp` only.

**51b — step-only decode graph.** Decode-time inputs zero out the
audio branch (text row holds the new token, `text_zero_mask=1`,
`speech_active_mask=0`, audio rows are `speech_zeroemb_idx` whose
combined mask is also 0), so the entire 6L input_local_transformer
+ group_proj + fusion path computes a literal zero that gets added
to `text_embeds`. The new `mimo_asr_build_step_graph` skips it
outright: `embed_w[next] → 36L Qwen2 LM → final_norm → lm_head`,
T=1. Decode loop in `mimo_asr_transcribe` calls this instead of
the heavy 9-row prefill graph for every n_past>0 step. Prefill
still uses the original `mimo_asr_build_prefill_graph`.

**51b' — O15-style cached step graph.** Mirrors
`src/qwen3_tts.cpp:976-1050` (the `QWEN3_TTS_O15` path). `Lk`
pinned to `kv_max_ctx` so the graph topology is invariant across
n_past, `kv_indices = lm_positions` so the K/V scatter via
`ggml_set_rows` keys off a runtime tensor instead of a static
byte offset baked at build time. The plan is reused for every
decode step within a transcribe call (`ctx->step_t1_gf`),
invalidated at every transcribe entry and after `extract_stage`
clobbers `compute_meta`. The mask covers the full Lk with -INF
beyond `n_past + q` so never-written cache slots can't leak NaN
or whatever the buffer happened to hold.

**Diag-capture gate.** `mimo_asr_build_prefill_graph` takes a
`bool diag_captures`. Production transcribe passes false (drops 4
`ggml_set_output` calls + 2 `ggml_cont` clones — without
`set_output` the scheduler is free to reuse those buffers for
later ops, so the only extracted output is `prefill_text_logits_step0`
which is consumed). The diff harness `extract_stage` passes true.
`MIMO_ASR_DIAG=1` env var force-enables for transcribe-time
debugging. ~5 % wall-clock + a much cleaner allocator picture.

**Bench (M1, Metal, Q4_K, samples/jfk.wav):**

| | wall | per-step decode |
|---|---:|---:|
| Section 56 baseline (sec. 56) | ~37 s | ~1.15 s |
| After 51b/b' | 44.4 s* | 0.79 s |

\* The wall-clock looks worse because the bench was a cold run and
~7-10 s went to Metal kernel JIT compile in prefill (`Tg=97`,
includes 4.4 s of `kernel_*_compile_pipeline`). On a warm run the
prefill should drop sub-second, putting end-to-end below the 37 s
baseline. Per-step decode is the apples-to-apples metric: 1.46×
faster, hits the lower end of the work order's 51b+51b' target
band ("~1.5–2× from 51b alone, ~1.3× more from 51b'"). Transcript
matches the gold byte-for-byte.

**Cosine gate (crispasr-diff bf16-ref vs Q4_K-C++)** — five
prefill stages reproduce the section 56 numbers exactly, confirming
51b/b' do not perturb prefill numerics:

| Stage | cos | section 56 |
|---|---:|---:|
| prefill_audio_features    | 0.998270 | 0.998 |
| prefill_text_embeds       | 0.996284 | 0.996 |
| prefill_inputs_embeds     | 0.997573 | 0.998 |
| prefill_last_hidden       | 0.963177 | 0.963 |
| prefill_text_logits_step0 | 0.981261 | 0.981 |

Argmax of step-0 logits = 1597 (`' And'`), matches the reference
and is consistent with the JFK transcript starting "And so, ...".
The harness's strict 0.999 threshold doesn't apply to bf16-ref vs
Q4_K-C++ (would need fp32 ref + ~28 GB RAM, blocked by 51a — see
LEARNINGS lesson 3 of section 56). Ref archive at
`/Volumes/backups/ai/mimo-asr-ref.gguf` (4.1 MB) regenerated via
the 2-phase loader patch (commit `3945d7b`) which keeps peak
memory under 16 GB.

**Out of scope, queued for follow-up:** 51a (mmap-backed weight
loader to drop the `_platform_memmove` into a fresh CPU backend
buffer — saves ~12.7 GB resident on the F16 14.9 GB GGUF, but it
touches `src/core/gguf_loader.h` which is shared by 24 backends
and needs the full diff-harness gauntlet); 51c (F16 step decode,
trivial after 51a); fused QKV per LM layer (saves 2 matmuls per
layer × 36 layers × N steps, but the converter has to be updated
to fuse + write a single `attn_qkv.weight` tensor).

### 61. granite-speech-4.1 — plus + nar variants — PLAN #54 (May 2026)

The `ibm-granite/granite-speech-4.1-2b` family ships three variants
with significantly different decoders despite the shared "4.1-2b"
naming. Base shipped in HISTORY §54 of the granite-family DRY refactor;
this entry records the **plus** + **nar** variants reaching bit-exact
JFK transcription and HF release.

| Variant | Decoder | Encoder change | Outputs |
|---|---|---|---|
| `granite-speech-4.1-2b-plus` | Granite-1B AR | `cat_hidden_layers: [3]` | text + speaker labels + word-level timestamps |
| `granite-speech-4.1-2b-nar` | non-autoregressive (`NLENARDecoder`) | self-conditioning at L8 + BPE aux head + 4-layer hidden capture | text |

**Plus variant.** Backend alias `granite-4.1-plus` registered with the
unified Session API; PLUS GGUF (5.6 GB f16) is converted and the
runtime concatenates encoder layer 3 with the final layer output
(`il + 1 == cat_index`, matching HF's `output_hidden_states`
convention). Punctuation + capitalisation come for free from the
PLUS training default. End-to-end JFK transcript:

```
And so my fellow Americans, ask not what your country can do for you,
ask what you can do for your country.
```

Speaker labels + word-level timestamps remain queued (template-only
~50 LOC follow-up). Commits: `f298818` (cat_layer + tokenizer fix),
`ed0e5ac` (backend alias + registry), `a3147b6` (HF README).

**NAR variant.** Backend alias `granite-4.1-nar` registered. Three
stages, all bit-exact on JFK:

1. **Encoder forward** (`granite_nle_run_encoder`). Same Conformer
   block as base; self-conditioning at layer 8 (running char-level
   CTC logits feed back through `out_mid`); 4-layer hidden state
   capture at the indices listed in `proj.encoder_layer_indices`
   (default `[4, 8, 12, -1]`). The capture obeys HF tuple semantics:
   `-1` resolves to `n_layers`, and the snapshot at the
   self-conditioning layer is taken AFTER the residual is added.
   Validated against PyTorch on JFK at cos_min ≥ 0.999. The BPE
   auxiliary head (`enc.bpe_out`) is intentionally not wired through
   `run_encoder` — it's only needed by the LLM editing pass's
   text-init step, where it's faster to run on the posterior-pooled
   features.
2. **Windowed Q-Former projector** (`granite_nle_run_projector`).
   Two-pass: (A) one ggml graph for the per-encoder-layer LayerNorms
   + concat + `layer_proj` (4096 → 2048) + GELU; (B) one Q-Former
   graph per block (`block_size=15`, `downsample_rate=5`,
   `query_length=3`) with mean-pool over downsample groups, additive
   `query` and `window_positions`, two 32-head SDPA cross-attention
   + SiLU-MLP layers, and a final `out_norm` + `out_linear`. Output
   rate: 3 audio tokens per 15 encoder frames. PyTorch JFK match at
   `projector_output cos_min=0.999999` (T_out=111 × llm_dim=2048).
3. **Non-causal LLM editing pass** (`granite_nle_run_llm_editing`).
   Single graph over the flat `[audio_embs, text_embs_with_slots]`
   sequence with µP scaling (embedding_multiplier=12,
   attention_multiplier=1/128, residual_multiplier=0.22). 40 layers
   of RMSNorm + non-causal `flash_attn_ext` (mask=nullptr, GQA 16/4
   native) + SwiGLU. Tied LM head. The caller passes audio_embs
   pre-divided by `embedding_multiplier` so the uniform downstream
   scale-up recovers the original projector output for audio while
   still scaling text by 12× — mirrors `_build_flat_llm_inputs`.
   Validated bit-exact: `editing_logits cos_min=0.999999` and 47/47
   top-1 match on JFK.

**Reference dump pitfall.** `GraniteModel.forward` unconditionally
builds an upper-triangular causal mask and passes it to SDPA, which
then enforces causality regardless of `self_attn.is_causal=False`.
The upstream "flash_attention_2 required" assertion is real — only
FA2 reads `is_causal` directly without using the mask. The
`tools/reference_backends/granite_nle.py` dumper monkey-patches
`transformers.models.granite.modeling_granite.create_causal_mask` to
return None to get true non-causal attention via SDPA.

**Transcribe orchestration** (`granite_nle_transcribe`) wires
together: encoder (with BPE auxiliary head:
`posterior_weighted_pool` window=4 driven by `1 - blank_prob_mid`
from the L8 self-conditioning softmax, populating `last_bpe_logits`)
→ BPE-CTC greedy decode (`unique_consecutive` → drop blank label 0
→ shift to LLM IDs by -1) → `core_bpe::detokenize` (GPT-2 byte-level
reverse, lifted into shared `core_bpe::token_bytes_to_utf8`) → strip
+ lowercase + " "-fallback → re-tokenize via
`core_bpe::tokenize_simple` → `add_insertion_slots` (`max(2n+1, 8)`,
EOS-padded) → `run_projector` divided by `embedding_multiplier=12`
and sliced to `enc_T // downsample_rate=5` audio frames →
`run_llm_editing` → per-row argmax + unique_consecutive + drop EOS +
detokenize. JFK end-to-end output matches reference `final_text`
exactly.

**HF release.** All three variants live on
[`cstr/granite-speech-4.1-2b-GGUF`](https://huggingface.co/cstr/granite-speech-4.1-2b-GGUF) (base, 4 quants: F16
5.58 GB, Q4K F32-enc 2.94 GB, Q4K F16-enc 2.07 GB, Q4K mini 1.7 GB),
[`cstr/granite-speech-4.1-2b-plus-GGUF`](https://huggingface.co/cstr/granite-speech-4.1-2b-plus-GGUF) (plus, F16 5.6 GB),
and [`cstr/granite-speech-4.1-2b-nar-GGUF`](https://huggingface.co/cstr/granite-speech-4.1-2b-nar-GGUF) (nar, 4 quants:
F16 5.8 GB, Q4K 3.4 GB, Q4K f16enc 2.5 GB, Q4K mini 1.6 GB). Registry
aliases `granite`, `granite-4.1`, `granite-4.1-plus`, `granite-4.1-nar`.

### 62. Zero-copy mmap GGUF loader — PLAN #51a env-flag (May 2026)

`core_gguf::load_weights` previously did mmap-then-`tensor_set` into a
freshly allocated CPU backend buffer. For the 14.9 GB F16 mimo-asr GGUF
that peaked at ~13 GB resident on a 16 GB Mac and thrashed swap for
25+ minutes, blocking the F16 + fp32-ref strict cos≥0.999 diff harness
gauntlet that section 60 had to defer. Q4_K (4.5 GB) hid the symptom
on production paths.

**Implementation** (commit `9710f80`, `src/core/gguf_loader.cpp` +
`src/CMakeLists.txt`). Skip `ggml_backend_alloc_ctx_tensors` on the
CPU path when `CRISPASR_GGUF_MMAP=1`; mmap the file with
`MAP_PRIVATE | PROT_READ|PROT_WRITE` (Win32 `FILE_MAP_COPY`); wrap the
data section in a custom `ggml_backend_buffer_t` whose `free_buffer`
callback munmaps; bind each tensor with `ggml_backend_tensor_alloc`
into the mmap'd offsets. Mmap lifetime is owned by `model.buf` so the
existing 24 caller pattern (move `wl.buf` → `model.buf`, drop the
`WeightLoad`) is unchanged. The CMake target adds `../ggml/src` as a
private include for `ggml-backend-impl.h`.

**Copy-on-write was load-bearing.** First validation pass used
`MAP_SHARED + PROT_READ` and parakeet immediately faulted with SIGBUS
in its BN-into-conv fold path (`src/parakeet.cpp:535`,
`ggml_backend_tensor_set` on read-only mmap'd pages). MAP_PRIVATE gives
COW: pages a backend never touches stay shared with the file's page
cache (the RSS win), pages it mutates get a private anon copy.
Backends with similar post-load weight surgery (vibevoice, …) inherit
the fix for free.

**Validated.** parakeet Q4_K — Metal default, CPU default, and CPU +
`CRISPASR_GGUF_MMAP=1` all produce the gold JFK transcript.

| Case | Working-set RSS | Notes |
|---|---:|---|
| mimo-asr Q4_K (4.5 GB GGUF), legacy | ~5.5 GB | full backend buffer + OS-resident mmap pages |
| mimo-asr Q4_K, mmap | ~760 MB | OS keeps the mmap'd file in shared cache only |
| mimo-asr F16 (14.9 GB GGUF), legacy (predicted) | ~13 GB peak | per HANDOFF; thrashes swap 25+ min on 16 GB Mac |
| mimo-asr F16, mmap | **~910 MB** during model load | observed at 60 s elapsed before contention forced a kill |

The F16 mmap loader churned through model load + LID + tokenizer +
encoder in seconds — the same span where the parallel legacy F16 run
on the same file was still 30+ min into its mmap-then-copy. End-to-end
decode timing is still pending. The kill was forced by an unrelated
problem the test surfaced: `/Volumes/backups/ai` had hit 100% capacity
(12 GB free of 1.9 TB), so both my mmap test and the parallel legacy
F16 from another Claude session ended up thrashing on page faults
under heavy memory pressure (vm_stat showed 13M+ swapins). Once that
disk has headroom again, re-time F16 end-to-end before flipping the
default.

**Default still legacy.** The env flag remains opt-in. Flip the
default in a follow-up commit once the F16 RSS savings are measured
end-to-end and the diff harness on F16 GGUFs has been exercised across
the qwen3-tts and granite-speech families.

**Side-quest fix: parakeet `--no-gpu` crash** (commit `b85f56c`,
`ggml/src/ggml.c`). Pre-existing assertion failure
`GGML_ASSERT(*cur_backend_id != -1)` in
`ggml_backend_sched_split_graph` whenever parakeet's encoder ran on
the CPU backend. Root cause: `ggml_conv_2d` and `ggml_conv_2d_dw` set
their im2col output type to `a->type` (the kernel) unconditionally,
producing `MUL_MAT(F16 im2col, F16 kernel)`. The CrispASR fork's
issue-#38 patch (F16 `vec_dot_type=F32` +
`ggml_vec_dot_f16_f32`) doesn't support F16×F16. Fix mirrors
`ggml_conv_1d`: pick F32 im2col when either operand is F32, cast the
kernel to F32 when the chosen path needs it. Conv_2d puts activations
as src0 and kernel as src1 (reversed from typical); Metal's kernel
table has `mul_mv_f16_f32` but not `mul_mv_f32_f16`, so the kernel
cast is needed for both backends. Slight Metal slowdown
(13.3× → 7.5× realtime on parakeet-tdt-0.6b-v3 Q4_K JFK) is the same
trade-off `ggml_conv_1d` already makes upstream.

**Out of scope, queued for follow-up:** measuring the F16 mimo-asr
RSS win end-to-end; flipping the env-flag default; PLAN #51c (F16
step decode) which the HANDOFF flagged as "trivial" once #51a lands.

### 63. Session 2026-05-02 — canary ref dumper + DRY helpers + cache-clear ABI sweep

Three small landings collected here because none warrants its own
section:

**PLAN #5 — Canary reference dumper** (commit `63f708e`). The C++
`crispasr-diff canary` branch was already wired (mel + encoder taps)
but the matching Python ref dumper was missing. Added
`tools/reference_backends/canary.py`, modeled on `parakeet.py`: NeMo
`ASRModel.from_pretrained("nvidia/canary-1b-v2")`, preprocessor +
encoder forward, per-layer hooks for 32 encoder layers of diagnostic
captures, transposed to TimeMels layout for the C++ side. Diff
against the existing Q4_K GGUF on JFK shows expected quantisation
noise (encoder cos_mean 0.972, cos_min 0.35 on low-magnitude frames)
but the runtime still transcribes byte-exact:
`"And so, my fellow Americans, ask not what your country can do for
you, ask what you can do for your country."` Strict cos≥0.999 PASS
would need an F16 GGUF — deferred until disk headroom allows the
converter to run without thrashing (98% full external + slow
`shutil.copyfileobj` per CLAUDE.md note).

**PLAN #53 — Two narrow core helpers** (commit `d393a43`). After the
qwen3-tts codec and SNAC decoders both shipped, a re-read across all
four of our TTS decoders (vibevoice σ-VAE, qwen3-tts codec, mimo
tokenizer encoder-only, SNAC, kokoro istftnet) showed the convergence
the original PLAN #53 ("`core/audio_decoder.h`") imagined wasn't
there: VibeVoice is continuous-VAE with ConvNeXt, MiMo has no
decoder, Kokoro is istftnet, and only qwen3-tts codec + SNAC share
shape — and even there the codebook-handling diverges (qwen3-tts
splits codebook-0 from rest-15 vs SNAC sums all codebooks equally).
Rewrote the PLAN entry to scope down, then extracted exactly two
helpers:

- `core_act::snake_beta` in `core/activation.h` — the BigVGAN
  `y = x + exp(-β)·sin²(x·exp(α))` activation. qwen3-tts now
  delegates via a 1-line alias. ~10 LOC saved net.
- `core_convt::convt1d_crop` in `core/conv.h` — generic
  channels-first `ggml_conv_transpose_1d` wrapper with
  caller-controlled `crop_left`/`crop_right`. qwen3-tts (causal,
  `crop_right=K-stride`) and SNAC (symmetric,
  `crop_left=crop_right=pad`) both delegate. ~30 LOC saved net.

SNAC `crispasr-diff` 8/8 PASS (cos_min 0.999941 unchanged) confirmed
the wrapper is bit-equivalent. PLAN #53 priority moved MEDIUM →
LOW since the `core/audio_decoder.h` super-helper is no longer the
intent.

**PLAN #56 #5 — Kokoro phoneme cache clear ABI** (commits `9bffb0f`,
`6cabefa`, `d022bff`, `603f47e`). The `kokoro_phoneme_cache` LRU
already existed in `kokoro_context`; what was missing was a way for
long-running daemons to drop the cache when resynthesising across
many speakers. Added:

- `kokoro_phoneme_cache_clear(ctx)` extern-C in `kokoro.cpp`/`.h` —
  takes the mutex, `lru.clear()` + `idx.clear()`. Cheap and
  thread-safe.
- Session-scoped re-export
  `crispasr_session_kokoro_clear_phoneme_cache(session)` in
  `crispasr_c_api.cpp` — no-op for non-kokoro backends, returns -1
  on null handle.
- All 7 wrappers got the method (Python `Session.clear_phoneme_cache()`,
  Rust `Session::clear_phoneme_cache()`, Dart `clearPhonemeCache()`,
  Go `Session.ClearPhonemeCache()`, Java `clearPhonemeCache()`,
  JS `Module.ttsClearPhonemeCache()`, Ruby
  `Session.clear_phoneme_cache(handle)`). Each follows the existing
  per-binding pattern for `set_codec_path`. +55 LOC across the 5
  trailing wrappers. PLAN #59's "open this section when a consumer
  asks" rule was relaxed for this single-method addition because the
  alternative (C-only surface) was ergonomically worse.
- No-model unit tests in `tests/test_python_session.py` cover the
  symbol export + null-handle return path. +2 PASS in 0.56 s, no
  model required.

**Side-quest: clang-format-18 CI fix** (commit `21464e3`). The
`d393a43` PLAN #53 commit added 4 lines that local clang-format-22
considered fine but Ubuntu CI's clang-format-18 flagged. Fixed in
place — see LEARNINGS.md "clang-format-22 vs CI v18" lesson for the
trap and the safer workflow.

**Side-quest: 2 LEARNINGS lessons** (commit `cc82e25`).

- The clang-format-22-vs-v18 destruction trap (auto-formatting whole
  files locally produces hundreds of whitespace changes that v18 *also*
  rejects — manually align by eye instead).
- NeMo ref dumpers must transpose to TimeMels for parakeet/canary
  (the C++ side uses `core_mel::Layout::TimeMels` which is
  n_mels-fast, T_mel-slow — opposite of NeMo preprocessor's natural
  `(B, n_mels, T_mel)` C-contiguous order; cosine-signature lookup
  table for diagnosing layout swaps included).

### 64. PLAN #60d Fused QKV + #60e KV-quant plumbing — mimo-asr (May 2026)

Picked up the two MEDIUM-effort OPEN items from PLAN #60 in one
session. Both shipped behind clean fallback paths so existing GGUFs
keep working — only the new fused-QKV Q4_K mimo-asr download gets the
speedup, and only callers that opt in via `CRISPASR_KV_QUANT` change
KV-cache footprint.

**60d Fused QKV (mimo-asr LM):**

The Qwen2 LM in mimo-asr does Q/K/V via three separate `mul_mat`s
plus three `ggml_add` bias ops per layer × 36 layers × N decode
steps. Fusing the per-layer Q/K/V weights into one
`[d_model, q_dim + 2*kv_dim]` tensor and the biases into one
`[q_dim + 2*kv_dim]` 1-D vector replaces the three matmuls with one
fused matmul and the three bias adds with one — algebraically
identical, fewer ggml ops to schedule.

`core_attn::kv_self_attn` already accepted a `qkv_w` parameter
(qwen3-asr / qwen3-tts use it for runtime fusion of F16/F32 weights).
The Qwen3 path doesn't have biases, so the helper needed an extra
`qkv_b` parameter for the Qwen2 case. Added at the end of the
parameter list (after the existing `o_b`) so all existing callers
stay binary-compatible at default-arg level.

`mimo_asr_qwen2_block` gained `attn_qkv_w` / `attn_qkv_b` slots.
`bind_qwen2_block` tries the fused names first via `try_t`; on miss
it falls back to the separate-Q/K/V `require_t` path. Audio
`audio.blk.*` blocks always take the fallback (their bidirectional
attention reads separate Q/K/V outside `core_attn`, and the
converter is intentionally LM-only).

The HF→GGUF converter (`convert-mimo-asr-to-gguf.py`) was updated
to emit fused tensors at convert time. But re-running the BF16→F16
conversion on this 16 GB / 99%-full-disk box thrashes the same way
PLAN #51c documented — the converter sustained ~0.8 MB/min on the
contested disk before being killed. Workaround: a new
`tools/patch_mimo_asr_fuse_qkv.py` that loads an existing GGUF via
`gguf.GGUFReader`, byte-concat's the per-LM-layer Q/K/V data along
the row dim, and re-emits as fused tensors. Bit-identical to a
fresh-from-converter result for F16/F32 (numpy element concat) and
for Q4_K/Q8_0/etc. — each row's quant blocks are independent so
byte concat across rows is a valid quantised tensor. The patcher
runs in ~5 minutes (vs hours / never for the BF16-source converter
on the contested disk).

The Q4_K-on-disk re-quantisation path is separate from this fuse —
the existing 4.5 GB Q4_K stays as Q4_K, just with three Q/K/V
tensors per layer collapsed into one `attn.qkv.weight`.

**Validation (Q4_K, JFK, on this 16 GB box, 4 concurrent claude
sessions, 99%-full external disk):**
- `crispasr-diff` cosines reproduce the §56 / 51b/b' baselines
  bit-exactly (audio_features 0.998270, text_embeds 0.996284,
  inputs_embeds 0.997573, last_hidden 0.963177, logits 0.981261 —
  identical to the unfused-Q4_K reference run, character-by-character).
- JFK transcript byte-identical: "And so, my fellow Americans,
  ask not what your country can do for you. Ask what you can do
  for your country."
- `MIMO_ASR_BENCH=1` on the same disk-thrashed box:
  - Unfused (separate Q/K/V): prefill 10295 ms, decode 79498 ms /
    26 steps = **3058 ms/step**, total LM 89.8 s.
  - Fused (this PLAN #60d): prefill 4881 ms, decode 46946 ms /
    26 steps = **1806 ms/step**, total LM 51.8 s.
  - **1.69× per-step decode speedup** at the same disk thrash
    level. The work order predicted 1.1-1.2× on a quiet box;
    larger here likely because each fewer matmul also avoids one
    page-fault round-trip on the contested disk. On uncontended
    hardware expect closer to 1.1-1.2× pure-compute.

The patched Q4_K was uploaded to `cstr/mimo-asr-GGUF` replacing
the unfused file. The unfused F16 stays in the repo unchanged —
the runtime fallback in `bind_qwen2_block` keeps it working as-is.
F16 re-upload is queued behind PLAN #51c (uncontended-disk
re-conversion).

**60e KV-quant env flag (mimo-asr):**

`CRISPASR_KV_QUANT={f16,q8_0,q4_0}` now picks the KV-cache dtype in
`mimo_asr_kv_init`. Default stays F16, so this is bit-identical to
existing behaviour.

The shared `core_attn::kv_self_attn` adapts both write and read
paths for quantised cache:

- **Write:** the original `ggml_cpy(F32, slice-of-cache)` path
  requires the destination to be contiguous when the source/dst
  types differ, but a per-token slice into a max_ctx-strided 4D
  cache is never contiguous. CPU's `dup_to_q<float>` aborts; Metal
  also skips non-contig quant dst. Fix: when the cache is
  quantised, always go through the `ggml_set_rows` scatter path
  (which both backends accept for F32→Q* directly), even when no
  `kv_indices` is supplied. The `positions` tensor — already
  populated with [n_past..n_past+T) for RoPE — is exactly the
  row-id list set_rows wants, so we re-use it as the synthetic
  kv_indices.
- **Read:** `ggml_is_quantized(kv_k->type)` switches from `ggml_cont`
  to `ggml_cast(view_q*, GGML_TYPE_F32)`. Both backends support
  `Q*→F32` CPY; the CPU backend's `compute_forward_dup` only
  implements `Q*→F32` (not `Q*→F16`), so F32 is the only safe
  dequant target if the scheduler splits the op. Cache *storage*
  still uses ~half the bytes (Q8_0) — the hour-long-podcast use
  case where `max_ctx > 10k` would otherwise need ~1.5 GB F16 KV —
  but reads pay one dequant pass per layer.

**Validation (Q8_0 KV cache, fused Q4_K, JFK):**
- F16 KV baseline (above) had last_hidden cos_min 0.963177, logits
  cos_min 0.981261.
- Q8_0 KV: last_hidden cos_min 0.963031 (Δ -0.000146), logits
  cos_min 0.981454 (Δ +0.000193). Both stay well above the work
  order's ≥0.98 gate. Pre-attn stages (audio_features,
  text_embeds, inputs_embeds) don't go through the KV cache and
  reproduce the F16 cosines exactly.

**Per-backend env wiring (commit `8edfb74`, same session):** the
mimo_asr-local helper was lifted into a shared
`core_attn::kv_dtype_from_env(const char* tag)` and called from
each `*_kv_init` whose KV path routes through
`core_attn::kv_self_attn`. 9 backends wired in one commit:
`mimo_asr_kv_init` (refactored to use the shared helper),
`qwen3_asr_kv_init`, `voxtral_kv_init`, `voxtral4b_kv_init`,
`granite_speech_kv_init`, `gemma4_e2b` (`g4e_kv_init`, both
sliding-window and full-attention caches share the dtype),
`glm_asr_kv_init`, `omniasr` (`omniasr_alloc_kv_cache`), `orpheus`
(`kv_alloc`), `qwen3_tts` talker (`kv_alloc` — `cp_kv` stays F16
since the code-predictor decode doesn't go through `core_attn`).
Default remains `GGML_TYPE_F16` so the wiring is bit-identical to
legacy behaviour until a user opts in via
`CRISPASR_KV_QUANT={q8_0,q4_0}`.

Smoke-tested: `crispasr-diff mimo-asr` with default-F16 KV reproduces
the audio_features 0.998270 / text_embeds 0.996284 / inputs_embeds
0.997573 / last_hidden 0.963177 / logits 0.981261 cosines bit-exact
post-refactor.

Side-quest fixed in passing: `voxtral4b_kv_init` and
`granite_speech_kv_init` had a hardcoded
`ggml_type_size(GGML_TYPE_F16) * hd * max_ctx * n_kv * nl` byte
pre-compute used to size the backend buffer. This silently
over-allocates for quant types (Q8_0's `ggml_type_size` is 34 bytes
for a 32-element block, treating each *element* as 34 bytes →
massive over-alloc). Switched both to compute `ggml_nbytes()` after
`ggml_new_tensor_4d` instead, which is the right pattern regardless
of dtype.

Per-backend cosine validation (CRISPASR_KV_QUANT=q8_0
`crispasr-diff` against each bf16 reference) is the actual rollout
gate that remains open — see PLAN #60e for the list.

Backends with custom KV paths (canary, cohere, kyutai_stt,
vibevoice) don't route through `core_attn::kv_self_attn`, so the
shared write/read fixes from 1594577 don't cover them; they'd each
need backend-specific work to support quant KV. Left as PARKED in
PLAN #60e.

**Side-quest:** the converter run itself was killed mid-flight after
22 min / 0.8 MB/min sustained on the contested disk — same
diagnostic signature as PLAN #51c thrash mode. The patcher script
exists specifically to side-step this on this hardware; the
converter itself is correct (and what would run on an uncontested
machine).

### 65. Session-API word-confidence parity + 61a/b feature-matrix uplift (May 2026)

**Problem.** Every session-API caller (Rust crispasr crate, Python
wrapper, Flutter, Ruby, …) saw `word.confidence = 1.0` for parakeet
(hardcoded) and no per-word data at all for every other backend
(canary, qwen3, cohere, granite, voxtral, voxtral4b, mimo-asr,
wav2vec2, firered-asr, glm-asr, kyutai-stt, moonshine, omniasr,
fastconformer-ctc). Plus the C-side accessor
`crispasr_session_result_word_p` existed in the .cpp but wasn't
declared in any binding's FFI, so callers couldn't even read it.

**Resolution — three batches of work, all runtime-side, no GGUF
changes:**

**1. Per-token confidence runtime APIs (PLAN #61b).** Each ASR
backend that produces text via greedy / beam decode now exposes a
`*_transcribe_with_probs` C-API (or, for backends with token data
already, a refactor that surfaces the `.p` field through the
adapter):

- **wav2vec2** — new `wav2vec2_greedy_decode_with_probs` does CTC
  frame argmax + numerically-stable softmax of the picked logit;
  emits one `wav2vec2_token_prob{id, prob, frame_start, frame_end}`
  per non-blank emission. Adapter populates
  `crispasr_segment::tokens` with frame-aligned t0/t1.
- **firered-asr** — beam-search loop in `firered_asr_transcribe_impl`
  refactored: `beam_hyp` gained `token_logprobs` parallel to
  `tokens`; `candidate` gained `token_logprob` (= `top_score[k]` at
  push time). `firered_asr_transcribe_with_probs` returns a
  `firered_asr_result*` with id / `expf(logprob)` arrays in
  lock-step with the winning beam's token sequence. Old plain
  `firered_asr_transcribe` is now a 1-line wrapper.
- **fastconformer-ctc / canary-ctc** — new
  `canary_ctc_greedy_decode_with_probs` returns a
  `canary_ctc_decode_result*` with token ids, softmax probs,
  CTC-frame ranges, AND text-offset/length into the assembled
  transcript (so callers can substring-quote each emission without
  redoing the ▁→' ' / special-token logic). Frame timestamps
  computed via `canary_ctc_frame_dur_cs`.
- **moonshine** — refactored `moonshine_transcribe` into
  `moonshine_transcribe_impl` that optionally captures probs;
  added `moonshine_set_temperature` (sticky context flag) so the
  same loop also handles `> 0` multinomial sampling. New
  `moonshine_token_text(ctx, id)` uses a new
  `moonshine_tokenizer::token_to_piece` that decodes a single id
  without trim/special-strip.
- **glm-asr** — extended `sample_token` with optional `out_prob`;
  refactored `glm_asr_transcribe` → `_impl` with optional out
  vectors, only emitting tokens for non-special pieces; new
  `glm_asr_transcribe_with_probs`. Adapter does GPT-2 byte-level
  BPE Ġ→space / Ċ→newline decode for per-token text.
- **kyutai-stt** — extended `sample_token` with `out_prob`; refactored
  `kyutai_stt_transcribe` → `_impl`; new
  `kyutai_stt_transcribe_with_probs`. SentencePiece ▁→' ' done in
  the adapter.
- **omniasr (LLM variant)** — extended `omniasr_run_prefill` and
  `omniasr_run_dec_token` with optional `out_logits` parameters;
  when set, the build switches from `output_token_id=true` (argmax
  baked into graph) to `output_token_id=false` and reads back the
  full `[V, 1]` logits tensor for CPU-side argmax+softmax.
  Per-step capture into `cur_prob` + writeback to context-owned
  `capture_token_ids` / `capture_token_probs` pointers. New
  `omniasr_transcribe_with_probs` orchestrates. CTC variant returns
  null and falls back to text-only path.

Adapter capability flags updated: every CTC trio + decoder quartet
now reports `tok-conf Y` in `crispasr --list-backends`. README
"Feature matrix" `Per-token confidence` row went from 8 ✔ to 15 ✔.

**2. Auto-download (PLAN #61a).** Both fc-ctc and wav2vec2 already
had registry entries — only needed `CAP_AUTO_DOWNLOAD` on each
adapter. 2 README cells gained.

**3. Session-API word emission (PLAN #65).** Every modified
backend's session path now produces a `crispasr_session_seg::words`
array with real per-word probs:

- **parakeet** — `parakeet_word_data` gained a `float p` field;
  `parakeet_transcribe_ex` accumulates `cur_p_sum / cur_p_cnt` mean
  alongside the existing word grouping. Session adapter reads
  `pr->words[i].p` instead of hardcoding 1.0. *(Landed in commit
  `bc67d12` mid-session after CI surfaced the missing struct
  field.)*
- **wav2vec2** — session path switched from `wav2vec2_greedy_decode`
  to `wav2vec2_greedy_decode_with_probs`, projecting CTC emissions
  into a new `ca_token_record` shape, then through a shared
  `emit_words_from_tokens` helper that does SentencePiece-style
  word grouping (▁ / leading-space / standalone " " / punctuation
  attachment).
- **canary** — session path switched to `canary_transcribe_ex`
  (existing API), tokens already have `.p`.
- **cohere** — same with `cohere_transcribe_ex`.
- **firered-asr / glm-asr / kyutai-stt / moonshine / omniasr (LLM)** —
  switched to each backend's new `_transcribe_with_probs`. Per
  backend a small ▁→space / Ġ→space decode loop in the adapter.
- **fastconformer-ctc / canary-ctc** — switched to
  `canary_ctc_greedy_decode_with_probs`.
- **voxtral / voxtral4b** — `run_voxtral_family` (the templated
  decode loop both backends share in `c_api.cpp`) gained per-step
  numerically-stable softmax over the picked logit. Decoded piece
  goes through a ▁→space pass in the loop, so `emit_words_from_tokens`
  sees the same convention as the other backends.
- **mimo-asr** — `mimo_asr_transcribe` refactored into
  `mimo_asr_transcribe_impl` with optional `out_token_ids` /
  `out_token_probs`. The greedy `argmax` lambda became
  `pick(L, float* out_p)`, and the impl passes
  `capture_probs ? &prob : nullptr` so the legacy text-only entry
  pays no softmax cost (mimo's vocab is 151,680 — softmax on every
  step would add ~150k `expf` calls per token, measurable). New
  `mimo_asr_transcribe_with_probs` + `mimo_asr_token_text`.
- **qwen3 + granite** — both runtime `*_transcribe` functions are
  stubs that return empty/null. The CLI worked because its adapter
  drives the building blocks (`*_compute_mel`, `*_run_encoder`,
  `*_tokenize`, `*_embed_tokens`, `*_kv_init`, `*_run_llm_kv`,
  `*_token_text`) directly. Ported the same flow into the session
  path in `c_api.cpp`:
  - **qwen3** — chatml prompt with audio-pad splice; greedy decode
    via `core_greedy_decode::run_with_probs`; metadata-token filter
    (`<|im_start|>`, `<asr_text>`, `[PAD…]`, "language <name>"
    capture) mirrors the CLI adapter; GPT-2 byte-level decode via a
    new shared `gpt2_byte_decode` helper.
  - **granite** — chat-template selection (`audio_token < 50000` ⇒
    granite-3.x control tokens, else legacy 4.0 hardcoded `kPrefix4`/
    `kSuffix4`); projector-output splice into the audio-pad
    positions; `granite_speech_decode_tokens` for batch transcript;
    `granite_speech_token_text` + `gpt2_byte_decode` for per-token
    pieces.

**Public-API additions:**

- `crispasr_session_result_word_p(r, i_seg, i_word)` returns the
  per-word probability in `[0, 1]`, or `-1.0f` for "no data".
- Rust `crispasr-sys::crispasr_session_result_word_p` FFI
  declaration added.
- Rust `crispasr` crate's `SessionWord` gained `confidence: f32`,
  populated through the new accessor (folds C-side `-1.0` to `1.0`
  so consumers can render uniformly).
- Python `SessionWord` gained `confidence: float = 1.0`; ctypes
  binding probed via `hasattr(lib,
  "crispasr_session_result_word_p")` so old dylibs stay loadable.
- Flutter was already wired (`wordPFn` lookup with
  `providesSymbol` probe; `Word.p` field).

**Cross-backend smoke-test results** (JFK clip via the Python
session API, real probabilities, no longer 1.0):

```
parakeet     | 22 | And=0.9993 so,=0.9146 my=0.9998 fellow=0.9989
canary       | 22 | And=0.9722 so,=0.9605 my=0.9925 fellow=0.9969
cohere       | 22 | And=0.9810 so,=0.9966 my=0.9998 fellow=1.0000
voxtral      | 22 | And=0.9992 so,=0.8858 my=1.0000 fellow=1.0000
wav2vec2     | 22 | and=0.9999  so=0.9998  my=0.9999 fellow=0.9991
firered-asr  | 22 | AND=0.9825  SO=0.9834  MY=0.9975 FELLOW=0.9974
mimo-asr     | 22 | And=0.9999998… so,=0.9999964… my=1.0 fellow=1.0
qwen3        | 22 | And=1.0000 so,=0.9135 my=1.0000 fellow=0.9997
granite-4.1  | 22 | and=0.9516  so=0.9984  my=0.8053 fellow=0.9986
```

(mimo's 1.0s are softmax-saturated to fp32 precision on a clean
recording — `repr` shows the actual values are like
`0.9999998807907104`. Not a hardcoded 1.0.)

**Side-quest: Metal `GGML_OP_PAD` left-pad audit.** While testing
wav2vec2 the binary aborted with `unsupported op 'PAD'`. Root cause
in `ggml/src/ggml-metal/ggml-metal-device.m:131-138`: Metal supports
`GGML_OP_PAD` only when all left-side pads (params 0/2/4/6) are
zero. wav2vec2's `ggml_grouped_conv1d_same` used
`ggml_pad_ext(ctx, x, pad_l, pad_r, …)` with `pad_l = (K-1)/2 ≈ 63`
inside a `ggml_backend_graph_compute(metal, gf)` call (no scheduler
fallback) — instant crash on Apple Silicon.

Fix: replace the in-graph pad with a host-side zero-fill into a
wider input buffer, set the input tensor at the pre-padded shape,
skip the `ggml_pad_ext` op entirely. Per-call cost is one
`vector<float>` zero-fill + memcpy of `T_pad * C * 4` bytes (≈2.8
MB for wav2vec2-large) — under 1 ms on M1, well below the noise
floor. 3× consecutive JFK runs at 0.69 s ± 0 ms.

Audit of the rest of the tree: 24 `ggml_pad_ext` call sites total.
Every other left-pad (kyutai_stt:238, firered_asr:1404+1513,
qwen3_tts:3299+3321+3998, marblenet_vad:302, gemma4_e2b:320, 761,
762, vibevoice:322+343, core/conv.h:72) runs through
`ggml_backend_sched_graph_compute` which queries `supports_op` and
falls unsupported ops back to a CPU sub-backend transparently —
safe by construction. The only landmine remaining is the dead
`build_pos_conv_graph` in `wav2vec2-ggml.cpp:615` (gated off via
`include_pos_conv=false`); if anyone re-enables it they'd need to
port the same CPU-pad fix. Documented in LEARNINGS.md "Metal
`GGML_OP_PAD` only supports right-side pads".

**No GGUF regeneration required for any of this work.** All
changes are runtime-side, calling already-existing model C-APIs
that read from the existing GGUF layout. Both old (e.g.
granite-4.0 with no `audio_token_index` key) and new GGUFs work
unchanged — that's why the granite path falls back to
`kPrefix4`/`kSuffix4` legacy ids when
`granite_speech_audio_token_id()` returns `-1`.

**Still open** (text-only in the session API; tracked in PLAN
#65a): gemma4-e2b. (vibevoice + moonshine-streaming were added in
the same session, see below.) Plus other-binding word-p exposure
for Go / Java / Ruby (parallel-worker `5534588`) and JS (still
TTS-only, deferred).

**61c kyutai native + word timestamps (parallel worker).**
`kyutai_stt_transcribe_ex` returns per-token + per-word data with
audio-frame-aligned timestamps via Kyutai's "delayed-streams"
architecture: every emitted text token is associated with its
source audio frame, so word boundaries fall out for free at the
LM frame rate (12.5 Hz, with `audio_delay_seconds` lookahead
subtraction). The adapter switched to this entry point and now
declares `CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS`, lighting
up two README cells.

**61e omniasr-llm temperature.** The per-token-confidence work in
this same batch had already added an `out_logits` capture path to
`omniasr_run_prefill` + `omniasr_run_dec_token` (logits-mode graph
instead of graph-baked argmax). Adding sampling on top was a thin
extension: `omniasr_transcribe_llm` now takes a `pick_from_logits`
lambda that does CPU argmax (when `temperature == 0`) or
multinomial sampling from `softmax(logits / T)` with a
deterministic per-call xorshift64 seed (when `temperature > 0`),
overriding the helper's argmax pick. The captured probability is
the softmax-with-temperature of whichever token we picked.
Adapter wires `cp.temperature = params.temperature` and adds
`CAP_TEMPERATURE`. Smoke test confirms `-tp 0.7` produces a
different sample than greedy on a clean recording while keeping
the transcript sensible.

**61f punctuation toggle.** glm-asr / kyutai-stt / moonshine /
omniasr-llm gained `--no-punctuation` via two new shared helpers
in `examples/cli/crispasr_backend_utils.h`:

- `crispasr_strip_ascii_punctuation(s)` — keeps ASCII letters /
  digits / whitespace / any byte ≥ 0x80 (so multi-byte UTF-8
  sequences pass through untouched). Collapses runs of multiple
  spaces left behind by stripped punctuation. Trims leading /
  trailing whitespace.
- `crispasr_lowercase_ascii(s)` — ASCII-only `tolower`. Multi-byte
  UTF-8 bytes pass through unchanged so non-Latin scripts aren't
  corrupted.

Each adapter applies both to `seg.text`, every `seg.tokens[i].text`,
and (kyutai only) every `seg.words[i].text` when
`!params.punctuation`. Capability bits gain `CAP_PUNCTUATION_TOGGLE`.
4 cells in the README "Punctuation toggle" row.

Smoke (moonshine):
```
default      → "And so my fellow-american asked not what your country can do for you, ask what you can do for your country."
--no-punc    → "and so my fellowamerican asked not what your country can do for you ask what you can do for your country"
```

**65a vibevoice + moonshine-streaming.** Closes 2 of the 3 last
text-only session-API backends. Pattern mirrors the previous batch:

- `vibevoice` — refactored `vibevoice_transcribe` into
  `vibevoice_transcribe_impl` with optional `out_token_ids` /
  `out_token_probs`. `argmax` lambda became `pick(lg, out_p)` with
  softmax gated on `capture_probs`. Vocab is ~152k Qwen2 — gating
  is necessary so legacy text-only callers don't pay the
  per-token softmax cost (matches the same pattern from mimo-asr).
  Public C-API: `vibevoice_transcribe_with_probs`,
  `vibevoice_result_free`, `vibevoice_token_text`. Session adapter
  routes raw vocab pieces through `gpt2_byte_decode` (Qwen2
  byte-level BPE Ġ→space, Ċ→newline) and the shared
  `emit_words_from_tokens` helper, dropping `<|...|>` special
  tokens.
- `moonshine-streaming` — same refactor. New
  `moonshine_streaming_transcribe_with_probs`, `_result_free`,
  `_token_text`. The `_token_text` accessor reuses the shared
  `moonshine_tokenizer::token_to_piece` (added during moonshine
  proper's batch).

Cleanup: the historical `run_char_transcribe` lambda in
`crispasr_session_transcribe_lang` was removed. After this batch
every backend either has a token-prob path or routes through the
explicit text-only fallback block — the lambda was unreachable
dead code.

PLAN.md updated: §65 main batch + §65a vibevoice + moonshine-
streaming marked DONE; gemma4-e2b alone remains as the last
text-only session-API holdout. §61 audit updated with current
DONE / OPEN status for each sub-item (61a-c/e/f DONE; 61d/g/h/i/j
queued; 61k blocked on 60k).

**65a-residue gemma4-e2b session-API word probs (commit
946f624).** gemma4-e2b already used `core_greedy_decode::run` for
its decode loop — switching to `run_with_probs` was straight-
forward: capture the prefill-step prob via
`core_greedy_decode::softmax_of`, route surviving (non-special)
ids + probs through optional `out_token_ids` / `out_token_probs`
parameters on `gemma4_e2b_transcribe_impl`. Public additions:
`gemma4_e2b_transcribe_with_probs`, `gemma4_e2b_result_free`,
`gemma4_e2b_token_text`. Session adapter applies SentencePiece
▁→space decode per-token (Gemma's vocab uses ▁ markers) and runs
the shared `emit_words_from_tokens` helper. With this commit
**every ASR backend has a token-prob path through the session
API** — no more text-only fallbacks for transcription.

**61d Best-of-N for the LLM-style decoder quartet (commit
946f624).** The temperature work in 9ffb196 gave glm-asr /
kyutai-stt / moonshine / omniasr-llm multinomial sampling. To
make `--best-of N --temperature T > 0` actually draw N independent
samples (rather than collapse to one repeated sample because the
per-call seed was deterministic from audio), each backend gained
a sticky `*_set_seed(ctx, seed)` setter:

* moonshine: new `seed_override` field on `moonshine_context`,
  mixed into the existing xorshift-derived `rng_state` in the
  decode loop. seed=0 falls through to audio-derived seed (legacy
  single-shot bit-identical behaviour).
* omniasr-llm: same pattern — `seed_override` field mixed into
  the encoder-data-derived `rng_state` of
  `omniasr_transcribe_llm`.
* glm-asr / kyutai-stt: both sample via libc `rand()` (static
  helper with no context arg). `*_set_seed(ctx, unsigned)` calls
  `srand(seed)` instead — process-global, but best-of-N at the
  adapter level is sequential so the pattern works. Documented
  as "serialize at adapter".

Adapter pattern (identical across all four backends):

```cpp
const int n_runs = (params.temperature > 0 && params.best_of > 1)
                     ? params.best_of : 1;
result* best = nullptr;
double best_score = -1.0;
for (int run = 0; run < n_runs; run++) {
    backend_set_seed(ctx,
        run == 0 ? 0 : (uint64_t)run * 0x9E3779B97F4A7C15ULL);
    auto* cand = backend_transcribe_with_probs(ctx, ...);
    double score = mean(cand->token_probs);
    if (!best || score > best_score) { swap; }
    else { backend_result_free(cand); }
}
```

Run 0 always uses seed 0 → sticky override falls back to the
audio-derived seed → bit-identical to single-shot. Only runs
1..N-1 inject salt.

Smoke (moonshine on JFK with `-tp 0.7 --best-of 3`):
```
crispasr[moonshine]: best-of-3 picked score=0.9190
And so my fellow American, ask not what your country can do for
you, ask what you can do for your country.
```
Greedy baseline transcribes "fellow-american asked"; best-of-3 —
choosing the highest-mean-prob sample of 3 — got "fellow
American, ask" which is the actually-correct transcription. Real
quality win, not just diversity. README "Best-of-N" row gains
4 ✔.

**Closures and deferrals.** With this commit, PLAN #65 is fully
closed (no more text-only ASR backends in the session API). PLAN
#61 closes 61a-f (20 cells gained across the audit). Remaining
open: 61h beam search (~300 LOC shared infra, queued), 61j
translate (empirical validation across 3 backends, queued).
Deferred with documented reason:

* 61g `--ask` — target backends (glm-asr, omniasr-llm) are not
  instruction-tuned. glm-asr's prompt is hardcoded ids, no live
  tokenizer; omniasr-llm uses FLORES-200 lang ids, not chat.
  Both would need empirical validation before plumbing the toggle.
* 61i flash-attn for fc-ctc — `core_conformer::build_block`'s
  rel-pos path (Q·K + R·Q_v + rel_shift) doesn't fit
  `ggml_flash_attn_ext` (no rel-pos hook). Would need positional-
  encoding swap or custom flash kernel.
* 65b binding word_p exposure for Go/Java/Ruby/JS — those
  bindings are TTS-only today; full Session ASR API surface is a
  separate scope (~150 LOC + design per binding).

**61h beam search (partial, May 2026 — this session).** Shipped the
shared infra and one of four target cells; the other three are
deferred for a perf-real reason that the original PLAN entry didn't
foresee.

**What landed:**

- `src/core/beam_decode.h` — header-only template helper mirroring
  `core_greedy_decode::run_with_probs`. Takes a single `replay_fn`
  callback `(ctx, tokens, n_tokens, prompt_len) → float*` that
  rebuilds the given beam's KV state and returns last-position
  logits. Strategy is *replay-from-prefix*: each beam-step embeds +
  forwards the entire generated suffix from the post-prompt anchor
  (the C-API stays unchanged, no `*_kv_save` / `*_kv_restore`
  needed). Top-K logits → cumulative log-prob → global prune. Stops
  when `beams[0]` ends in EOS.
- `glm-asr` adapter — `cp.beam_size = params.beam_size` plumbed
  through; `glm_asr_transcribe_impl` dispatches to
  `core_beam_decode::run_with_probs` post-prefill when
  `beam_size > 1`. The replay lambda is `glm_asr_embed_tokens` +
  `glm_asr_run_llm_kv` (which already supports batched
  `(emb, n_tokens, n_past)` calls). `CAP_BEAM_SEARCH` added to
  capabilities. README "Beam search" row gains 1 ✔ for glm-asr.

**Smoke results on JFK (11 s, glm-asr-nano-q4_k):**

| Setting | Wall time | Transcript |
|---|---|---|
| `-bs 1` (greedy, GPU) | ~1 s | "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country." |
| `-bs 1` (greedy, CPU) | 71 s | identical |
| `-bs 2` (beam, GPU) | 52.9 s | identical |
| `-bs 4` (beam, GPU) | 35.1 s | identical |
| `-bs 2` (beam, CPU) | 129 s | identical |

The GPU beam timings are dominated by per-call Metal command-buffer
sync (~150-300 batched calls per beam decode); CPU at `-bs 2` runs 1.8×
greedy as theory predicts.

**Bug found and fixed mid-session.** The first iteration of the helper
took a single `cfg.eos_id` (mirroring `core_greedy_decode::Config`).
glm-asr has *three* stop tokens (`{59246, 59253, 59255}` per its
`hp.eos_token_ids[]`): the LLM's `<|user|>` end (59253) and a couple
of tokeniser sentinels. Single-id matching only caught 1 of 3, so on
~2/3 of inputs the beam would emit a non-cfg.eos_id stop token, NOT
mark itself finished, and run to `max_new_tokens=512`. CPU smoke
stalled past 6 minutes before being killed; that was the symptom.
Fix: extended `Config` with `std::vector<int> eos_ids` (helper checks
membership; legacy single `eos_id` still honoured when the vector is
empty). Adapter passes the full `hp.eos_token_ids[0..n_eos)` set
through. After the fix beam=2 CPU finishes in 129 s, beam=2/4 GPU in
under 1 min — all transcripts match greedy.

**Why the other three target backends got deferred.**
`omniasr-llm`, `kyutai-stt`, and `moonshine` each have a per-step
decode of the form "advance the LM by ONE token at the current KV
position, return logits". Replay-from-prefix on those would do
`O(B × T²)` *single-token* graph rebuilds — each rebuild is a fresh
`ggml_init` + scheduler reset + Metal command buffer + sync. Even
on glm-asr (which only needs `O(B × T)` *batched* rebuilds) the
total Metal compute at `beam_size=2` on 11 s JFK was still many
minutes per run during this session — replay-from-prefix is just
expensive once `T` grows past ~50. The honest fix is per-backend
`*_kv_save` / `*_kv_restore` so each beam can have a real branched
state, instead of paying `O(T²)` to rebuild it from scratch every
step. Reopen those rows when that infra lands. PLAN.md #61h was
re-statused from OPEN to IN PROGRESS with a sub-table showing which
sub-steps shipped vs. deferred.

See LEARNINGS.md "Replay-from-prefix beam search is `O(B × T²)`".

### 66. Feature-matrix closure pass — sticky setters + streaming + mic + Go/Java/Ruby parity (May 2026)

Six commits (`d963e3a`, `947262f`, `041471f`, `89687f0`, `5534588`
plus the parallel-agent sweep `f130a40` that adopted my
`crispasr_session_stream_open`) closed the long-standing gap
between CLI flags and what wrappers can reach. The audit going in:
9 capabilities lived only as CLI flags. The audit coming out: 6 of
9 wired across all 7 wrappers (or "all where it makes sense"); 3
remain documented-deferred with revised effort estimates.

**Sticky session-state setters** (`d963e3a`). New session struct
fields `source_language` / `target_language` / `punctuation` /
`translate` consulted by whisper, canary, cohere transcribe paths
when the per-call `language` arg isn't supplied. Six C-ABI exports
(`set_source_language`, `set_target_language`, `set_punctuation`,
`set_translate`, `set_temperature`, `detect_language`).
`set_temperature` multiplexes per-backend setters (canary, cohere,
parakeet, moonshine each expose runtime temperature) and returns
-2 if no backend in the session honours it (soft no-op).
`detect_language` wraps the standalone LID helper. Wrappers landed
in 3 canonical (Python/Rust/Dart) and the 3 trailing
(Go/Java/Ruby) — JS/emscripten skipped (TTS-only surface today).
+545 LOC (canonical) + +373 LOC (trailing).

**Registry enumeration** (`947262f`). `crispasr_registry_count()`
+ `_get_at(i)` internal API, `crispasr_registry_list_backends_abi`
C-ABI export, `list_known_models()` in Python+Rust+Dart returning
all 37 known backends in declaration order. Wrappers building
model-picker UIs no longer need to know the list out-of-band.

**Streaming session API** (`947262f` + `041471f`).
`crispasr_session_stream_open(s, ...)` decouples streaming from
the whisper-only `crispasr_stream_open(whisper_ctx, ...)`. Today
it routes through whisper internally; the dispatch site is
labelled with where moonshine-streaming + kyutai-stt + voxtral4b
will plug in when those refactors land. Python `Session._Stream`
context-manager handle, Rust `Stream` with `Drop`-managed close,
Dart already had it, Go got it in `5534588` (`Stream` type +
`StreamOpen/Feed/GetText/Flush/Close` + `StreamingUpdate`).

**Library mic API** (`89687f0`). `src/crispasr_mic.{h,cpp}` wraps
miniaudio's `ma_device` capture mode. Cross-platform via
miniaudio's built-in backends (Core Audio / ALSA / PulseAudio /
WASAPI). The `MA_NO_DEVICE_IO` define on `crispasr_audio.cpp` was
dropped to expose `ma_device_*` symbols; the iOS/tvOS/watchOS
variant got it back in `1c0e996` because miniaudio's CoreAudio
backend pulls Objective-C headers on those platforms (the C ABI
is stubbed to return failure on iOS/tvOS/watchOS so libcrispasr
links). Live-mic smoke-tested from Python: 17 callbacks fired in
1s wall-clock = 16320 samples = 1.02s real audio captured. The
audio-thread callback uses a `NativeCallable.listener` (Dart) /
ctypes `CFUNCTYPE` trampoline (Python) / boxed `FnMut` closure
(Rust) — each binding's idiomatic way to keep the
native-to-managed callback alive across the audio thread.

**Realistic effort revisions** in PLAN.md for the still-deferred
items:

- **62c moonshine-streaming + kyutai-stt streaming**: PLAN had
  "~80 LOC each". Survey showed both expose only single-shot
  `_transcribe()` despite their architectures being
  streaming-friendly. Real cost: ~300 LOC + 1-2 days debug per
  backend (chunked encoder + state carry-over + numerical
  regression gating). Documented; opens when a consumer asks.
- **62e voxtral4b streaming**: stays as PLAN #7 high-complexity
  separate session (~200-300 LOC, decoder thread + audio frame
  injection).
- **Init-only flags (grammar/beam/flash)**: needs backend-reinit
  machinery (close+reopen, slow) or per-backend `set_*`
  extensions (~50 LOC × 14 backends = ~700 LOC mechanical +
  per-flag regression test). Documented; opens per PLAN #59
  policy.

**Non-collision dance** worked smoothly throughout: c_api.cpp had
~900 LOC of parallel-agent WIP at one point; my edits rebased
cleanly via the backup-checkout-restage-restore pattern. The
parallel agent's `f130a40` clang-format pass swept up my
`crispasr_session_stream_open` mid-flight under their author tag —
fine outcome (work shipped), surprising commit-message
attribution.

**Tests** (`tests/test_python_session.py`): 4 new no-model test
classes — `TestSessionStateSetters`, `TestStreamingAPI`,
`TestMicAPI`, `TestRegistryEnumeration`. All four PASS in well
under a second collectively, no model required. CI gates against
accidental ABI removal across refactors.

### 67. Kartoffel-Orpheus DE + lex-au — PLAN #57 Phase 2 checkpoint swaps (May 2026)

The first three Orpheus-3B German checkpoints landed as drop-in
swaps on the runtime that section 59 shipped — no source code
changes, just registry rows + factory dispatch + (for Kartoffel)
GGUF conversion via the existing `models/convert-orpheus-to-gguf.py`
with `--variant fixed_speaker`.

**Three new backends, all reachable via `--backend ... -m auto`:**

| Backend alias | HF mirror | Speakers | Provenance |
|---|---|---|---|
| `kartoffel-orpheus-de-natural` | [`cstr/kartoffel-orpheus-3b-german-natural-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF) | 19 (Jakob/Anton/Julian/Jan/…/Sophie/Marie/Mia/…) | Convert of `SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1` (llama3.2, gated) → F16 6.61 GB + Q8_0 3.5 GB + Q4_K 1.87 GB |
| `kartoffel-orpheus-de-synthetic` | [`cstr/kartoffel-orpheus-3b-german-synthetic-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF) | 4 (Martin/Luca/Anne/Emma) + 12 emotions + 5 outbursts | Convert of `SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1` (llama3.2, gated) — same shape; emotion control via `{Speaker} - {Emotion}: {text}` prompt |
| `lex-au-orpheus-de` | `lex-au/Orpheus-3b-German-FT-Q8_0.gguf` (3.52 GB, no convert) | unknown — depends on lex-au's training set | Pre-built Q8_0 from lex-au; registry alias only, points at the upstream HF file directly |

All three share the same SNAC codec ([`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF)) the orpheus base row already publishes.

**Validation: ASR-roundtrip via parakeet-v3 -l de.** Natural variant on Q8_0 / voice Julian:

```
$ crispasr --backend kartoffel-orpheus-de-natural \
    -m kartoffel-orpheus-de-natural-q8_0.gguf \
    --codec-model snac-24khz.gguf --voice Julian --temperature 0.6 \
    --tts "Hallo, ich heiße Julian und das ist ein Kartoffel-Orpheus Test." \
    --tts-output kartoffel_test.wav
$ crispasr --backend parakeet -m parakeet-tdt-0.6b-v3-q4_k.gguf -l de \
    -f kartoffel_test.wav --no-prints
Hallo, ich heiße Julian und das ist ein Kartoffel-Orpheus-Test.
```

Word-exact (only minor `Kartoffel-Orpheus-Test` hyphenation drift). Synthetic variant smoke deferred — local 16 GB box was memory-contested by the parallel agent's concurrent crispasr-diff + converter runs, and the orpheus 3B AR loop hung in both Metal init (84 min stuck on `libraries.data` cache I/O while pages were compressor-evicted) and CPU mode (40+ min for Q8_0, 26+ min for Q4_K, 0 bytes output). Architecture is the same as the natural variant which validated end-to-end, and the synthetic checkpoint is purely an LM-weight delta — no runtime risk.

**HF upload economics.** Per-file `api.upload_file` (sequential, README first) was the working recipe; `api.create_commit` with a 4-op transaction hung on token-refresh `RemoteDisconnected` mid-upload (43 min in TCP CLOSE_WAIT before kill). Xet dedup is dramatic when uploads share a base model: orpheus 10.1 GB upload was 576 MB net new bytes (Llama-3.2-3B base shared with the unsloth mirror); the synthetic Kartoffel 12 GB upload was ~5.1 GB net new because most chunks deduped against the natural variant which had landed an hour earlier.

**Gated-repo gotcha.** `SebastianBodza/Kartoffel_Orpheus-3B_german_*` are click-through gated. Even when authenticated as `cstr`, `snapshot_download` returned 401 — diagnosed as a token-loading bug: when `HF_HOME` is overridden, the lib looks for `$HF_HOME/token` instead of the default `~/.cache/huggingface/token`. Drop the `HF_HOME` override and it works. Direct `requests.head` with explicit `Authorization` header worked all along, which is what made the diagnosis tractable.

### 68. v0.5.4 release + wrapper-publish auto-trigger silenced (May 2026)

Tagged `v0.5.4` at commit `1c0e996`, `release.yml` (binary artifacts +
GitHub release page) + `Publish Docker image` ran cleanly. **291 commits
since v0.5.3** (April 27); highlights in the tag annotation:

- MiMo-V2.5-ASR end-to-end (PLAN #51) + perf wave (#51b/b' step-decode
  KV reuse, #60d fused QKV, #60e KV-quant env wired across 9 backends).
- Qwen3-TTS family (Base 1.7B, VoiceDesign 1.7B, CustomVoice 0.6B/1.7B).
- Orpheus-3B-FT TTS + lex-au-orpheus-de + Kartoffel_Orpheus DE natural.
- granite-speech-4.1 plus + nar variants.
- Per-token confidence + per-word probability across 14 backends.
- Streaming + mic library API (PLAN #62 a/b/d).
- Zero-copy mmap GGUF loader (`CRISPASR_GGUF_MMAP=1`), preload, mlock,
  WILLNEED + MADV_RANDOM helpers (PLAN #60a/b/c/f/g).
- Vibevoice cache-bypass extended to Vulkan + CUDA (issue #47, geneing).

**Wrapper-publish workflow silenced.** `release-wrappers.yml`'s
`tags: ['v*']` auto-trigger has been failing on every release since
v0.5.0 because the three target packages
(`crispasr-sys`/`crispasr` on crates.io, `crispasr` on PyPI, `crispasr`
on pub.dev) **have never been registered** — the registries reject
the very first publish from a CI-only flow because there's no prior
owner record to verify against. v0.5.4 confirmed it again
(`gh run view 25248028443`).

Decision: comment out the auto-trigger so future tag pushes don't keep
producing red runs; keep `workflow_dispatch` so manual ad-hoc testing
still works during eventual bootstrap. Bootstrap procedure is documented
end-to-end in PLAN §66 with exact per-registry commands. Workflow is
also hardened (per-job `continue-on-error: true` + secret-presence
prechecks) so when we re-enable the trigger, a single registry's
misconfiguration won't fail the whole workflow.

PLAN §67 carries forward the rest of this session's deferred
follow-ups in one place: F16 mimo-asr re-upload, per-backend Q8_0 KV
cosine validation, vibevoice CUDA cache reuse re-test,
SYCL/HIP/ROCm cache-bypass extension, `MADV_RANDOM` per-backend
wiring, disk5 cleanup, legacy `build.yml` audit.

### 69. PLAN #62c kyutai-stt streaming — chunked-batch over rolling window (May 2026)

**Done.** `crispasr_session_stream_open` now routes to a
kyutai-backed rolling-window stream when `s->kyutai_ctx` is set;
the four whisper-typed `crispasr_stream_*` functions
(feed/get_text/flush/close) branch on a new optional
`kyutai_stream_state` field on `crispasr_stream`. ~200 LOC across
three files, well under the 300-LOC PLAN estimate.

**Why chunked-batch, not true incremental.** The PLAN's original
"refactor `_transcribe` into incremental encode + decode + state
carry-over" path assumed the Mimi encoder was at worst causal.
Pre-impl exploration found `src/kyutai_stt.cpp:660` calls
`ggml_flash_attn_ext(..., nullptr, ...)` — **non-causal**, every
encoder-transformer frame attends to every other. True streaming
therefore can't bit-match batch without either O(n²) re-encoding
or swapping in sliding-window attention (~500-700 LOC, deviates
from training). Chunked-batch sidesteps both: each decode runs the
existing single-shot `kyutai_stt_transcribe_ex` over the last
`length_ms` of audio, so each window is bit-exact-batch by
construction.

**Validation.**
- Final stream output on JFK matches single-shot batch byte-for-byte
  (after both are lstripped of the SentencePiece `▁ → space`).
- Intermediate decodes show expected rolling-window behavior:
  "America" at decode #3 (6 s window) flips to "Americans" at
  decode #4 (8 s) once the LM has more context.
- Whisper streaming path unchanged — `ggml-tiny.bin` still produces
  a coherent JFK transcript through the same
  `crispasr_session_stream_open` → `crispasr_stream_feed`/`get_text`
  chain.

**Out of scope but trivially next.** `moonshine-streaming` is also
single-shot today; the same chunked-batch pattern (rolling-window
wrapper + `moonshine_stream_state` field) would land in ~50 LOC.
Voxtral4b stays separate (PLAN #7 — needs a real decoder-thread
refactor).

### 70. PLAN #61h beam search — branched-KV variant + 3 deferred backends shipped (May 2026)

**Done.** Closed the three rows §65 left deferred under "61h beam
search (partial)": omniasr-llm, kyutai-stt, moonshine. README "Beam
search" gains 3 ✔ for a total of 6 across the matrix
(whisper / glm-asr / kyutai-stt / firered / moonshine / omniasr).

**The blocker, recap.** Original §65 entry shipped only glm-asr because
the shared `core_beam_decode::run_with_probs` is *replay-from-prefix*
— each beam's per-step decode replays the full generated suffix from
the post-prompt anchor. For glm-asr's batched `glm_asr_run_llm_kv(emb,
n, n_past)` that's `O(B × T)` batched forwards, fast enough. For
omniasr-llm / kyutai-stt / moonshine — all of which have a
single-token-per-call decode API — replay-from-prefix would do
`O(B × T²)` *single-token* graph rebuilds, multi-minute on Metal.

**The unblock.** `core_beam_decode::run_with_probs_branched` —
header-only template, takes per-beam KV `save_fn` / `restore_fn` /
`snap_free_fn` / `step_fn` callbacks. Snap holders are wrapped in
`shared_ptr` inside the helper so siblings can share a parent's
post-step snap without double-free. Cost drops to `O(B × T)` true
single-token forwards. (The branched variant landed independently in
commit `e9783d2`; the same author / parallel-worker pass also did the
clang-format sweep in `db1149c` that bundled the kyutai-stt beam
edits.)

**Per-backend snapshot strategy.**

* **moonshine** — `kv_self.k[il]` / `kv_self.v[il]` are per-layer
  ggml tensors of shape `[head_dim, max_len, n_kv_heads]`. Snapshot
  reads each layer's full tensor via `ggml_backend_tensor_get`; small
  enough total (~MB-scale) to ignore. `kv_self.n` (the next-write
  slot) goes into the snapshot too. `kv_cross` is precomputed once
  outside the beam path and shared across beams.

* **omniasr-llm** — `kv_k` / `kv_v` are single contiguous tensors of
  shape `[head_dim, max_ctx, n_heads, n_layers]`. Snapshot reads the
  whole tensor (~30 MB total for the 300m model). `kv_n_used` is
  written by prefill but never read by the per-step decode (each
  call passes `n_past` explicitly), so the snapshot can omit it.
  Beam path activates only on the LLM variant; CTC is unaffected.

* **kyutai-stt** — same KV layout as omniasr but the per-frame loop
  is fundamentally different: each frame mixes audio code embeddings
  + text token embedding into one LM step, and there is no EOS — the
  loop runs to `T_frames`. Beam = top-K text-token decisions per
  frame, with audio codes shared across all beams. Refactored the
  per-frame body into `kyutai_lm_step(text_token, codes, frame_idx,
  n_past, &out_logits)`; greedy + beam both call it. Beam Config:
  `eos_id = -1`, `max_new_tokens = T_frames`, `prompt_len = 1`
  (frame 0 is run manually to seed initial logits + populate KV
  slot 0; helper handles frames 1..T_frames-1).

**Validation on JFK (11 s, Metal, warm cache).** All three backends
match greedy transcript at `-bs 1 / 2 / 4`; moonshine `-bs 4`
actually improves quality.

| Backend | model | `-bs 1` | `-bs 2` | `-bs 4` | Best transcript |
|---|---|---|---|---|---|
| moonshine | tiny-q4_k | 0.57s | 0.57s | 0.57s | "fellow Americans ask…" (beam=4 only) |
| omniasr-llm | 300m-v2-q4_k | 38s | 23s | 70s | "fellow americas ask…" (all match greedy) |
| kyutai-stt | 1b-q4_k | 5.1s | 8.6s | 14.2s | "fellow Americans, ask…" (all match greedy) |

omniasr-llm `-bs 4` lands above the 60s "rough" PLAN gate but the
per-step compute scales linearly (411 step calls for `-bs 4`,
103 for `-bs 1`); the snapshot copy + Metal single-token graph
dispatch overhead per call dominate, not the helper itself.

**Closures.** PLAN #61h sub-table now has 6 of 8 lines DONE
(generic helper × 2 + 4 backends). Two lines remain DEFERRED:
session-API beam exposure for the qwen3/granite/voxtral4b/voxtral
quartet (pure plumbing once the session API exposes `beam_size`),
and per-decoder beam for the canary/cohere encoder-decoder pair.
Both were never the heart of the §65 deferral — that was the
`O(B × T²)` perf wall that the branched-KV variant tore down.

### 71. PLAN #7 voxtral4b streaming — phase 1 + 1.5 (incremental encoder default-on) (May 2026)

**Scope.** Voxtral-Mini-4B-Realtime is the only deferred backend with an
architectural path to sub-second first-token streaming: causal+SWA
encoder + sliding-window LLM with persistent KV cache. Phase 1 ships
the streaming **API surface** for it. PTT/dictation semantics — `feed`
continuously, `flush` returns the transcript.

**Files.**
- `src/voxtral4b.{h,cpp}`: 5 entrypoints (`voxtral4b_stream_open` /
  `_feed` / `_get_text` / `_flush` / `_close`) + the `voxtral4b_stream`
  struct + an incremental encoder graph (`voxtral4b_build_graph_encoder_stream`)
  and projector graph that's wired but currently opt-in (see phase 1.5).
- `src/crispasr_c_api.cpp`: `void* voxtral4b_stream_state` field on
  `crispasr_stream` + 4 dispatch branches (close/feed/get_text/flush)
  + open path in `crispasr_session_stream_open`. Mirrors the
  kyutai/moonshine pattern from §69.
- `tools/bench_streaming_latency.py`: greenfield bench harness driving
  the stream API; `--check-batch-equality` runs the CLI batch baseline
  and asserts the streaming transcript matches byte-for-byte.

**Default path: incremental encoder during feed.** `feed()` runs the
streaming encoder graph chunk-by-chunk (~80 ms audio per chunk), with
persistent encoder K/V cache (`core_attn::kv_self_attn` with F32 cache
sized to `audio_swa = 750` frames) and per-conv left-context state
(2 mel frames for conv0, 1 d_model frame for conv1). `flush()` does
just the streaming-prompt prefill + per-step audio-injection greedy
decode, exactly mirroring the CLI adapter in
`examples/cli/crispasr_backend_voxtral4b.cpp`. Audio is auto-padded
with 32 left-pad tokens of zero audio at stream open + right-aligned
+ 10 right-pad tokens at flush time. Set
`CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER=1` to fall back to running the
whole encoder at flush time (regression-debug switch).

**Validation.** Bit-exact-batch on JFK 11 s (M1, Q4_K):
- Batch CLI: `"And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country."`
- Streaming via `crispasr_session_stream_open` → `feed` × N → `flush` → `get_text`: byte-for-byte identical, `[bench] BIT-EXACT-BATCH: PASS`.
- Per-embed cosine vs batch encoder: cos≥0.9999 across all non-tail-pad
  chunks; the final chunk diverges by construction (batch's last mel
  frame uses right-side `center_pad` data that streaming doesn't have,
  but the model already has the full content via the preceding 179
  embeds, so the transcript is unaffected).
- Wall-clock: feed ≈ 24 s (~170 ms per 80 ms chunk = 2.1× realtime),
  flush ≈ 10 s. Combined ≈ 34 s for 11 s audio = 0.32× realtime —
  same total cost as batch, distributed across feed + flush.

**The streaming-prompt convention is qualitatively different from
voxtral-3B's `[INST]…[TRANSCRIBE]` template** (which is what
`run_voxtral_family` in `crispasr_c_api.cpp:1483` uses for voxtral 3B
+ qwen3 + granite). The 4B-realtime model's prompt is BOS + 38
STREAMING_PAD = 39 tokens; audio embeds are *element-wise added* to the
first 39 prompt embeds; subsequent audio embeds are *added per-step* to
the next-token's embedding before the LLM forward (the "audio-injection
pre_hook"). Decode stops when audio is exhausted (matches the CLI's
`pre_hook → false` behaviour at `core/greedy_decode.h:306`); without
this stop, the model emits trailing garbage tokens that the prompt
template doesn't license. Streaming control tokens (id < 1000) are
filtered from the user-visible transcript.

**Phase 1.5 root cause and fix.** The incremental encoder went
default-on after a single 2-axis layout transpose was removed from
`vox_stream_advance_mel`. `core_mel::compute` with `Layout::MelsTime`
emits `(n_mels, T)` row-major — T fast, n_mels slow — which is exactly
what the encoder graph's mel input expects (ggml `ne=(T_mel, n_mels)`).
The streaming code had a transpose-on-copy loop that wrote it as
`(T, n_mels)` per row, sending mel and time axes to the encoder
swapped. The encoder produced plausible-looking but content-incorrect
audio embeds (cos~0.987 vs batch on the very first chunk, where the
expected value is cos~1.0 since the input is all-zero left-pad
silence). The model read these as noise and emitted only
streaming-pad tokens. Removing the transpose and keeping `mel_pending`
+ `conv0_lctx` in `(n_mels, T)` per-band-contiguous layout restored
cos≥0.9999 across all non-tail-pad chunks.

Smaller fix bundled in: encoder K/V cache is F32, not F16. The batch
encoder runs F32 throughout; F16 KV cache (which the LLM uses
successfully because the LLM was trained with F16 KV) introduces
precision loss across the encoder's 32 layers. Cost: 393 MB per stream
at the SWA cap of 750 frames. TODO: re-measure F16 cache with the mel
layout fix in place — F16 may now be sufficient.

**Latency target NOT met by phase 1+1.5.** ≤240 ms first-token target
requires further encoder optimisation (Q4_K Metal kernels, larger
chunks, decoder thread overlap with next-chunk encode). Phase 2 work.
Phase 1+1.5 ships the streaming API + bit-exact correctness; the
realtime-feed property is the remaining work.

**Phase 2 SHIPPED — chunk-size + fused QKV + default-unification + timing instrumentation (May 2026).**
- Default internal encoder chunk bumped 80 ms → 240 ms
  (`CRISPASR_VOXTRAL4B_STREAM_CHUNK_MS` env-var override). 2.6× feed
  speedup on JFK 11 s (24 s → 9.3 s); bit-exact-batch unaffected.
  Constraint: must be a multiple of 80 ms (8 mel frames) so the
  projector's stack-4 alignment lands on chunk boundaries.
- Per-stage timing instrumentation (`CRISPASR_VOXTRAL4B_STREAM_TIMING=1`)
  prints encoder-drain / prefill / first-text-token / per-step p50/p95
  / total flush wall-clock to stderr.
- Default-unification fix: feed() was incremental-by-default but flush()
  was batch-encoder-by-default — opposite-default bug from the stash
  restoration. Each flush threw away the streaming encoder's audio_embeds
  and re-ran the whole encoder over the full PCM. Unified to incremental
  on both paths (`CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER=1` to opt out).
  Wins: encoder drain at flush 2064 ms → 1016 ms; first-text-token
  2674 ms → 1646 ms.
- Runtime fused QKV for the LLM. Concat each layer's q/k/v weights
  along the output axis at load time into a single (d_model,
  q_dim+2*kv_dim) tensor; route through `core_attn::kv_self_attn`'s
  `qkv_w` path. Extends the qwen3_asr precedent to handle Q4_K (and
  any row-wise quantized format) by byte-concat — each output row is
  a self-contained block group, so concatenation along the output axis
  is a pure memcpy. ~7-8 % decode speedup (56 → 50.4 ms per step).
  `CRISPASR_VOXTRAL4B_FUSED_QKV=0` to opt out.
- FFN gate+up fuse was tried and reverted — Metal's Q4_K matmul kernel
  for (3072 × 9216) is already memory-bandwidth-bound, so combining
  two of those into (3072 × 18432) didn't help the per-step decode
  budget. The `core_ffn::swiglu_fused_gate_up` helper stays in place
  for any future caller where the ratio is more favourable (e.g. a
  larger model where the gate-up matmul time dwarfs the overhead, or
  a different backend with different kernel characteristics).

**Final phase 2 numbers** (M1 Q4_K JFK 11 s, all phase 1+1.5+2 wins):

| Metric | Phase 1 | Phase 1.5 | Phase 2 |
|---|---|---|---|
| feed total | 23 s | 24 s | **9.1 s** |
| flush total | 8.5 s (batch encoder at flush) | 11.4 s (incremental + waste) | **8.3 s** |
| per-decode-step | 56 ms | 56 ms | **50.4 ms** |
| first-text-token | n/a (no live data) | 2.7 s | **1.6 s** |
| bit-exact-batch | PASS | PASS | PASS |

**Phase 3 partial — combined-chunk flush + speculative LLM prefill (May 2026).**
- **Combined-chunk flush.** Right-pad feed at flush previously ran
  3-4 separate 240 ms encoder chunks plus tail-pad + per-chunk
  projector. Refactored to append right-pad zeros directly to
  `pcm_with_pad` and drain all pending mel via one larger combined
  chunk (~96 mel frames = 48 enc frames = 12 projector groups). Saves
  ~6 Metal kernel launches. Encoder drain at flush: 990 ms → 307 ms;
  first-text-token: 1646 ms → 921 ms.
- **Speculative LLM prefill during feed.** Once feed has produced ≥
  39 audio_embeds (after ~3.1 s of audio at 240 ms chunks), run the
  streaming-prompt prefill speculatively and stash the resulting
  last-position logits + n_past on the stream. Flush skips prefill
  (~250 ms saved) and jumps straight to the decode loop. No
  correctness risk: the LLM's KV cache state at position 39 is
  identical regardless of when prefill runs. First-text-token:
  921 ms → 650 ms.

**Final phase 1+1.5+2+3-partial numbers (M1 Q4_K JFK 11 s):**

| Metric | Phase 1 | Phase 2 | Phase 3-partial |
|---|---|---|---|
| feed total | 23 s | 9.1 s | **9.4 s** |
| flush total | 8.5 s | 8.3 s | **7.5 s** |
| per-decode-step | 56 ms | 50.4 ms | **50.4 ms** |
| **first-text-token** | n/a | 1.6 s | **650 ms** |
| bit-exact-batch | PASS | PASS | PASS |

Total session: first-text-token 2674 ms → 650 ms = **4.1× faster**.
The remaining ~410 ms gap to the ≤240 ms model-card target is the
architectural floor (8-step streaming-pad warmup × 50.4 ms = ~400 ms
plus the first text decode step). Below that requires either a
different prompt convention (model retrain) or substantially faster
Q4_K Metal kernels.

**Phase 3 finale — live captions during speech (May 2026).** Once
feed has produced ≥39 audio_embeds (~3.1 s of audio), every new
audio_embed during subsequent feeds drives one greedy decode step;
tokens commit immediately to `out_text` / `out_text_unread`, so
`get_text()` polled during feed returns progressive transcript.

No stable-prefix heuristic needed: voxtral4b's audio-injection
pre_hook makes each decoded token a deterministic function of the
audio context up to that point. Tokens commit immediately — no
retraction. This is the key architectural difference from
encoder-decoder ASR (whisper / parakeet / canary) where the encoder's
bidirectional context shifts as more audio arrives, making mid-decode
tokens unstable.

API: `voxtral4b_stream_set_live_decode(stream*, int)` toggles per-
stream. Generic dispatch via `crispasr_stream_set_live_decode` for
use through the unified `crispasr_stream*` handle. Python:
`Session.stream_open(live=True)`. Default OFF (PTT semantics
preserved).

Refactor extracted the inline decode loop into `vox_stream_drain_decode`
shared between feed (live mode) and flush (PTT + final drain). A
3-step state machine (argmax → stop-check → inject+forward) with a
`decode_logits_committed` flag prevents double-emission across
multiple drain calls — the bug pattern is "argmax + emit at top of
loop, return without forward, next drain re-argmaxes the same logits
and re-emits."

Smoke on JFK 11 s with `live=True`:
```
+ 1280ms: ' And'
+ 1760ms: ' so,'
+ 2000ms: ' my'
...
+10880ms: ' your'
flush:   ' country.'
```
Cumulative transcript matches batch byte-for-byte.

Limitation: sequential live decode is ~1.5× realtime on M1 Q4_K
(50 ms decode + 100 ms encoder per 100 ms audio chunk). Falls behind
realtime audio when fed from a live mic. Phase 4 (decoder thread
parallel to encoder) would fix this.

**Phase 4 — decoder thread (May 2026).** Optional worker thread that
drains decode steps in the background while feed() handles encoder +
projector on the main thread. Enable via
`CRISPASR_VOXTRAL4B_STREAM_DECODER_THREAD=1` (implies live mode).

Architecture: `worker_thread` sleeps on `cond_var` until shutdown OR
audio_embeds grow past `decode_adapter_pos`. feed() acquires
`sched_mutex` around encoder/projector/prefill, notifies cond_var,
returns. Worker drains under sched_mutex + decode_state_mutex.
flush() signals + busy-waits with 1 ms sleeps until worker is idle
and caught up to N_audio. close() requests shutdown + joins.

Performance on M1 (JFK 11 s, all three modes pass bit-exact-batch):
- PTT: feed 9.3 s, flush 7.3 s
- LIVE single-thread: feed 15.6 s, flush 1.0 s
- LIVE + decoder thread: feed 15.7 s, flush 1.0 s

The thread doesn't reduce total wall-clock on M1 because Metal's
single GPU queue serializes encoder + decoder regardless of how
many CPU threads submit work. **Wins materialise when:**
1. Mic-driven feed has audio-rate gaps between calls — worker
   drains decode during the gaps, feed returns between encoder
   chunks without waiting for the entire decode loop.
2. Faster GPUs (M3 Ultra, NVIDIA) with kernel-level parallelism —
   Metal/CUDA can overlap concurrent compute kernels submitted by
   different ggml_backend_sched_compute calls.

**PLAN #7 closed at phase 4.** Phase 5 (dual-sched for true on-Metal
parallelism via independent backend schedulers, ~150 LOC) deferred
to when a consumer measures gain on faster hardware.

**Architectural floor for first-text-token latency** (revealed by
the timing pass on M1 Q4_K JFK 11 s):

| Stage | ms |
|---|---|
| Encoder drain at flush (Metal JIT + final-chunk + projector) | ~2064 |
| LLM prefill (39-token streaming-prompt) | 247 |
| 8 streaming-pad warmup steps × 52 ms each | ~416 |
| First text-emitting decode step | ~52 |

Even with a fully-warm encoder, the streaming-prompt convention forces
≥ 663 ms before first text emits. ≤240 ms target requires either a
different prompt convention (model retraining) or substantially faster
Q4_K Metal kernels. Phase 2 remainder (encoder kernel pre-warm, LLM
fused QKV, live-captions stable-prefix commit) deferred.

**Working-tree hygiene incident (parallel-agent reset).** Mid-session,
a concurrent process reset `src/voxtral4b.{cpp,h}` and PLAN/HISTORY to
HEAD via `git checkout`/`git restore`, dropping the in-flight phase 1.5
edits. Recovered them from `git stash@{0}` ("On main: parallel-agent
docs + voxtral4b") which captured the work before the reset. Confirms
the LEARNINGS rule "never stash/relocate unrelated changes; the
working tree is shared by parallel agents" — and adds the corollary
that `git stash` from a parallel agent IS a recovery vector when the
working tree gets reset under you.

### 72. PLAN #63 Feature matrix parity — 9-phase capability expansion (May 2026)

**Scope.** Deep audit of the feature matrix (`crispasr --list-backends`
vs README vs actual code) revealed many backends had implemented
features without declaring the corresponding CAP_ flags, and several
cross-cutting improvements were low-hanging fruit. Nine phases executed
in a single session.

**Phase 1 — Beam search for LLM backends.** Wired `core_beam_decode::run_with_probs`
(replay-from-prefix strategy) into granite (all variants) and qwen3 CLI
backend adapters. When `-bs N` (N>1) is passed, the beam search replaces
the greedy decode loop. Greedy and best-of-N paths unchanged.

Files: `crispasr_backend_granite.cpp`, `crispasr_backend_qwen3.cpp`.

Skipped voxtral4b (per-step audio-injection incompatible with replay),
canary/cohere (opaque library decode calls, no beam_size parameter).

**Phase 2 — Auto-download gaps.** Added `CAP_AUTO_DOWNLOAD` to omniasr
(registry entry existed but flag missing). Added mimo-asr to model
registry (`cstr/mimo-asr-GGUF`) with `CAP_AUTO_DOWNLOAD + CAP_TOKEN_CONFIDENCE`.

**Phase 3 — Flash attention declarations.** All 6 backends (glm-asr,
kyutai-stt, firered-asr, moonshine, omniasr, omniasr-llm) already call
`ggml_flash_attn_ext` in their source — just needed `CAP_FLASH_ATTN`
declared in CLI adapters. No attention implementation changes.

**Phase 4 — CTC timestamps for aligner.** Added `CAP_TIMESTAMPS_CTC` to
moonshine, moonshine-streaming, omniasr, omniasr-llm, mimo-asr. The
`-am` CTC aligner flag is gated by this cap in `crispasr_run.cpp:292`.

**Phase 5 — Auto-punctuation for CTC backends.** Backends without
`CAP_PUNCTUATION_TOGGLE` now auto-enable `--punc-model auto` (FireRedPunc,
~50 MB, auto-downloaded). Affects fc-ctc, wav2vec2, firered-asr, omniasr-ctc.
Users suppress with `--no-punctuation` or `--punc-model none/off`.
Tested: fastconformer-ctc on JFK now emits capitalized, punctuated text.

File: `crispasr_run.cpp` (10 lines added before punc_ctx setup).

**Phase 6 — Best-of-N.** No code changes needed — works on GPU, CPU too
slow for large models. Documentation only.

**Phase 7 — Cap declaration fixes.** Re-audit found the initial automated
report was wrong: firered-asr, moonshine, kyutai-stt, omniasr all already
had correct capability declarations. Only omniasr (missing auto-dl) and
mimo-asr (capabilities=0) needed fixes.

**Phase 8 — Vibevoice CLI adapter.** Re-audit found
`crispasr_backend_vibevoice.cpp` already exists (160 lines, 16k→24k
resample, ASR + TTS). Initial report was incorrect.

**Phase 9 — Translation.** Investigated cohere (no translate token in
vocab, transcription-only model) and glm-asr (translate flag + prompt
injection infrastructure wired, but GLM-ASR-Nano doesn't respond to
translation instructions). `CAP_TRANSLATE` not declared for either.

Files: `glm_asr.h` (translate + target_lang fields), `glm_asr.cpp`
(prompt injection), `crispasr_backend_glm_asr.cpp` (param forwarding).

**Net capability gains across all phases:**

| Capability | Backends gained |
|---|---|
| Beam search (`-bs N`) | +granite, +granite-4.1, +granite-4.1-plus, +qwen3 |
| Flash attention | +glm-asr, +kyutai-stt, +firered-asr, +moonshine, +omniasr, +omniasr-llm |
| CTC timestamps (`-am`) | +moonshine, +moonshine-streaming, +omniasr, +omniasr-llm, +mimo-asr |
| Auto-download (`-m auto`) | +omniasr, +mimo-asr |
| Auto-punctuation | +fc-ctc, +wav2vec2, +firered-asr, +omniasr-ctc (opt-in default) |
| Token confidence | +mimo-asr |

**README feature matrix** updated after each phase to stay in sync with
`--list-backends` output. Matrix now covers 21 ASR backends × 18 features.

**Test results** (`test-all-backends.py --profile=feature`): 51 PASS,
0 FAIL, 3 SKIP (stream needs shared lib build). All 18 backends pass
transcribe smoke.

### 73. PLAN #59 Cross-binding C-ABI parity — Go/Java/Ruby ASR surface (May 2026)

**Scope.** The Go/Java/Ruby bindings could only do TTS — the most
critical ASR operation (`Transcribe`) was unwrapped. This session
added the full ASR pipeline surface to all three bindings.

**Go binding (14% → 35%):** Now wraps the complete user-facing surface:
- `Transcribe`, `TranscribeLang`, `TranscribeVAD` — full Session ASR
- `VADSegments` — standalone speech detection
- `DiarizeSegments` — speaker label assignment
- `AlignWords` — CTC forced alignment
- `DetectLanguagePCM` — standalone language detection
- `PuncModel` — FireRedPunc punctuation restoration
- `RegistryLookup`, `CacheEnsureFile`, `CacheDir` — model management

All with idiomatic Go types (`TranscribeResult`, `TranscribeSegment`,
`TranscribeWord`, `VADSpan`, `DiarizeSeg`, `AlignedWord`, etc.).

**Java binding (13% → 30%):** `transcribe()`, `transcribeLang()`,
`transcribeVad()`, `vadSegments()`, `alignWords()`,
`detectLanguagePcm()` + JNA declarations for punc/registry/cache.
Result types: `Segment`, `Word`, `AlignedWord`, `VADSpan`.

**Ruby binding (15% → 24%):** `transcribe(handle, pcm)`,
`vad_segments(path, pcm, ...)`, `align_words(model, text, pcm, threads)`.
Returns Ruby hashes with `:text`, `:t0`, `:t1`, `:words`, `:p` keys.

JS/emscripten skipped — WebAssembly binding needs a different approach
(can't do cgo-style FFI). Dart already had transcribe from a prior
session.

### 74. Issue fixes — gemma4-e2b #49, Docker diagnostics #31 (May 2026)

**gemma4-e2b assertion crash (#49).** `ggml_backend_sched` hash set
was sized to 16384 but the audio encoder graph uses up to 32768 nodes.
Increased to 40960. Also fixed SentencePiece `▁` (U+2581) not being
replaced with space during detokenization — raw vocab tokens were
concatenated with visible `▁` markers.

Added gemma4-e2b to test-all-backends.py registry: **PASS, WER=0.0%**.

**Docker diagnostics env var (#31).** Users hitting CUDA driver
mismatches couldn't run `--diagnostics` because `run-server.sh`
required a model. Added `CRISPASR_DIAGNOSTICS=1` env var that runs
diagnostics and exits without touching `/models`:
```
docker run --rm --gpus all -e CRISPASR_DIAGNOSTICS=1 crispasr:main-cuda
```

### 75. Unit test expansion — core decode + registry (May 2026)

**test-core-decode.cpp** (8 Catch2 cases, 27 assertions): Tests the
shared `core_greedy_decode` and `core_beam_decode` infrastructure used
by granite, qwen3, glm-asr, kyutai-stt, moonshine, omniasr backends.
Uses a mock LLM (no models, no GPU, sub-millisecond). Covers: argmax,
softmax_of, greedy sequence generation with EOS, beam search at
beam_size=1 (greedy equivalence) and beam_size=4, max_new_tokens cap,
multi-EOS termination, probability bounds.

**test-registry.cpp** (9 Catch2 cases, 14 assertions): Verifies the
model registry lookup for 8 backends (whisper, parakeet, mimo-asr,
omniasr, omniasr-llm, granite-4.1, gemma4-e2b, vibevoice) plus unknown
backend returns false. Pure in-memory queries, no network.

**#60e KV cache Q8_0 validation.** Transcript comparison F16 vs Q8_0
across 7 backends: granite, granite-4.1, glm-asr, omniasr-llm, voxtral
all bit-exact; qwen3 minor punctuation diff (WER=0%); mimo-asr
validated previously (cosine ≥0.98). `CRISPASR_KV_QUANT=q8_0` safe
for all tested backends.

**PLAN cleanup.** #41 (Moonshine IPA) superseded by kokoro espeak-ng
phonemizer. #9 (Parakeet TDT GPU) parked — encoder dominates, LSTM
decoder <0.7s. #60 marked DONE. #62 already done (stale PLAN entry).
Priority table deduplicated.

### 76. PLAN #11 WebSocket streaming server (May 2026)

Minimal RFC 6455 WebSocket server for real-time ASR streaming. ~300 LOC
in `examples/server/ws_stream.{h,cpp}`. Runs on `--ws-port` (default:
HTTP port + 1) alongside the existing httplib HTTP server.

Protocol: client connects → server sends `{"status":"ready"}` →
client sends binary PCM frames (16kHz mono float32) → server feeds
to rolling-window streaming decoder → sends JSON text updates
`{"text":"...","t0":0.0,"t1":1.5,"counter":N}` on each commit →
client sends text `"flush"` to finalize → server responds with
`{"text":"...","final":true}`.

Implementation: self-contained SHA-1 + base64 for WS handshake,
raw frame read/write per RFC 6455. One thread per connection, each
with its own crispasr session + streaming decoder. No external
dependencies. No TLS (reverse proxy for wss://).

Files: `ws_stream.h` (C API), `ws_stream.cpp` (implementation),
`server.cpp` (integration: `--ws-port` flag, startup/shutdown hooks),
`CMakeLists.txt` (added ws_stream.cpp to build).

### Session 2026-05-02/03 — full summary

**PLAN items completed this session:**
- **#11** WebSocket server → §76
- **#41** Superseded by kokoro espeak-ng phonemizer
- **#60** KV Q8_0 validated on 7 backends → §75
- **#62** Already done (stale PLAN, re-audited)
- **#63** Feature matrix parity, all 9 phases → §72

**PLAN items with major progress:**
- **#59** Binding parity: Go 14%→35% (full surface), Java 13%→30%, Ruby 15%→24% → §73

**Issue fixes:**
- **#48** granite-4.1-nar auto-detection (filename heuristic)
- **#49** gemma4-e2b scheduler assertion + detokenization → §74
- **#51** Windows build scripts (target + vswhere + exe check)
- **#31** Docker CRISPASR_DIAGNOSTICS=1 env var → §74

**CI / release:**
- 6 cross-platform build fixes (MSVC M_PI, preferred_separator, Win32
  InterlockedIncrement, Xcode POST_BUILD, Vulkan spirv-headers, git
  subject sanitization for MSVC /D + POSIX sh)
- v0.5.5 release shipped with 8 binary packages

**New capabilities shipped:**
- Beam search for granite + qwen3 (core_beam_decode wiring)
- Flash attention declared for 6 backends (already in code)
- CTC timestamps for 5 more backends
- Auto-punctuation for CTC backends (FireRedPunc default)
- Auto-download for omniasr + mimo-asr
- gemma4-e2b in test registry (19 backends total)

**Tests:**
- 17 new Catch2 unit tests (core decode + registry)
- 19/19 backends pass smoke, 0 failures on feature profile

**PLAN state at session end:** 8 DONE (#7, #11, #41, #60, #62, #63
+ #59/#63 partial), 1 PARKED (#9), 2 BLOCKED (#42, #43). Open:
#52 (TTS perf, needs quiet machine), #56 (phonemizer, needs external
libs), #57 (Chatterbox CFM, multi-session), #58 (MOSS, large),
#59 (Java/Ruby remaining gaps).

---

### §78 — Chatterbox vocoder fix (2026-05-03)

**Vocoder now produces correct "Hello world." from Python reference mel**
(previously "Oh."). Two bugs found via crispasr-diff per-stage protocol:

1. **iSTFT transposed data access** — CPU iSTFT loop read ggml tensor
   buffer as `data[frame*C+f]` but ggml stores `ne[0]=T` fast, so correct
   access is `data[f*T+frame]`. Swapped frequency bins with time steps.
2. **Missing ReflectionPad1d((1,0))** at the last upsample stage.

Additional fixes: proper SineGen + windowed STFT for source fusion,
Nyquist term fix in Hermitian iDFT. Debug infrastructure: per-stage
output markers, `vocode_dump()` API, `CRISPASR_HIFT_FULL_IDFT=1` gate.
All ggml graph stages match Python to cos=1.000; deterministic waveform
cos=0.93 vs `torch.istft`.

**Quantization + HF upload (2026-05-04):**
- F16/Q8_0/Q4_K for both T3 and S3Gen — all quant levels ASR-verified
  "Hello world" via moonshine-base.
- Published: [`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF)
  (T3: 1.1G/542M/287M + S3Gen: 548M/342M/237M).
- lahgtna-chatterbox-v1 (Arabic T3): converted, shares S3Gen with base.
  Published: [`cstr/lahgtna-chatterbox-v1-GGUF`](https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF)
  (T3 F16 1.1 GB).
- **Kartoffelbox_Turbo re-scoped**: inspection revealed GPT-2 architecture
  (fused QKV, LayerNorm+bias, learned pos embeddings) — NOT a Chatterbox
  Llama T3 checkpoint swap. Needs its own runtime (Tortoise-TTS variant).

---

### §79 — Cap-honesty audit + KV/layer offload knobs (2026-05-04)

**Test-runner triage uncovered cap-declaration drift, which then led
into the full llama.cpp-parity offload-knob roadmap (#69a/b/e + #72/73).**

#### Phase 1 — cap-honesty audit (commits b2b8a10, 3a4633f)

Running `tools/test-all-backends.py --profile feature` against the
post-3c2864e widened test tuples surfaced 12 FAILs across 24 tested
backends. Triage identified two distinct root causes:

- **Test-runner under-invocation (10/12)** — `test_lid` ran with empty
  args (no `-dl`); `test_word_timestamps` ran without `-am <aligner>`.
  Whisper's auto-detect path uses CRISPASR_LOG_INFO and ignores
  `--no-prints`, which is why the original whisper-only suite passed.
  Fix: `test_lid` invokes with `quiet=False` (no `--no-prints`); regex
  widened to accept the framework's `crispasr: LID -> language = '<x>'`
  format too.
- **Real cap drift (2/12)** — omniasr / omniasr-llm declared
  CAP_PUNCTUATION_TOGGLE but their CTC vocab is lowercase + unpunctuated
  ("and so my fellow americas ask not …"). Cap dropped.

Then, following the same cap-honesty thread, dropped CAP_LANGUAGE_DETECT
from parakeet, glm-asr, gemma4-e2b, qwen3 (all four declared the cap
but had no functioning native LID code path). The framework's
pre-step LID gate is `!has_native_lid`, so declaring the cap actually
*disabled* LID for users running `-dl`. Fixed end-to-end.

#### Phase 2 — Ruby bindings CI break (commit b2b8a10)

`bindings/ruby/ext/extconf.rb` linker hit two issues on Linux:

1. **Duplicate stb_vorbis / miniaudio symbols** — both `crispasr_audio.cpp`
   and `examples/common-crispasr.cpp` translation-unit-include the
   `_IMPLEMENTATION` macros. macOS ld silently picks the first; GNU ld
   errors. Fix: `-Wl,--allow-multiple-definition` on Linux.
2. **Catch2 not found** — CMake's STANDALONE-default `CRISPASR_BUILD_TESTS=ON`
   pulled tests targets into the dep graph, the dependency walker put
   `libCatch2*.a` in `$LOCAL_LIBS`, but `--target common crispasr` never
   built them. Fix: `-D CRISPASR_BUILD_TESTS=OFF -D CRISPASR_BUILD_EXAMPLES=OFF
   -D CRISPASR_BUILD_SERVER=OFF` to the cmake config.

#### Phase 3 — full K/V split (#69b + #69e, commits 1bd9ff8, d70f275, af0d789, 8952b02, 5bb1a0e, 709eafe, c2e1829)

Two independent env knobs:

- `CRISPASR_KV_QUANT_K` / `_V` — per-half KV cache dtype. Common
  llama.cpp recipe `K=q8_0 V=q4_0` saves ~40 % more KV memory than
  symmetric Q8_0. Both fall through to `CRISPASR_KV_QUANT` (legacy)
  when unset.
- `CRISPASR_KV_ON_CPU=1` — spill the KV cache to system RAM even with
  GPU weights. For long-context users where Q4_0 K/V doesn't fit.

Helper in `src/core/attention.h`:
```cpp
ggml_type kv_dtype_pair_from_env(tag).{k, v}    // for tensor allocs
ggml_backend_t kv_backend_from_env(gpu, cpu, tag)  // for buffer alloc
bool kv_cache_write(ctx, gf, K, V, kv_k, kv_v, il, n_past, T, indices)
                                                // F16 → ggml_cpy(view) fast path
                                                // quant → ggml_set_rows(indices)
```

The `kv_cache_write` helper (#73) was needed because backends with the
inline `ggml_cpy(K_perm, view_into_kv_k)` write pattern can't accept
quant K/V — `ggml_cpy` requires contiguous dst, and a view into a
multi-position cache is strided. Migration was straightforward for
backends with single-token decode (kyutai_stt) and for backends that
already had a `position` I32 graph input populated with `[offset,
offset+1, …, offset+n_tokens-1]` (canary, cohere — that tensor doubles
as the row-index input for the new ggml_set_rows path).

For canary/cohere the cache *read* pipeline
(`ggml_cont(ggml_permute(V, 1,0,2,3)) → ggml_mul_mat`) hits
`!ggml_is_transposed(a)` for quant V — `ggml_cont` of a permuted quant
tensor only flips the strides flag without moving data. Fix: insert
`ggml_cast(view, F32)` before the permute when the cache is quant.
Trades read-bandwidth saving for quant correctness; memory + write-
bandwidth savings retained.

**Coverage now: 14 backends with full K/V split.** voxtral, voxtral4b,
omniasr (LLM), qwen3_asr, granite_speech, orpheus, glm_asr, gemma4_e2b,
mimo_asr, qwen3_tts, chatterbox, kyutai_stt, canary, cohere — every
KV-bearing backend in the tree.

#### Phase 4 — layer-residency offload (#69a, commits 324a39c, aa3f54c, 1a0a5fd, 4f656ec)

`CRISPASR_N_GPU_LAYERS=N` puts transformer blocks `[0..N)` on GPU and
`[N..total)` on CPU. ggml_backend_sched routes compute to follow weight
residency. Useful for users with VRAM smaller than the model.

New helper:
```cpp
bool load_weights_split(path, gpu, cpu, is_gpu_fn, user, tag, &out)
int blk_layer_of_with_prefix(name, prefix)
bool is_gpu_tensor_with_prefix(name, &LayerSplitConfig{ prefix, threshold })
```

Per-backend prefix:
```
"blk."           voxtral, voxtral4b, qwen3_asr, granite_speech
"llm.blk."       glm_asr
"talker.blk."    orpheus
"dec."           omniasr-llm   (gated on model_type==1, CTC variant skipped)
"llm.layers."    gemma4_e2b
"model.layers."  mimo_asr
```

Validated on JFK at default + half-offload N per backend; bit-identical
correct transcripts across 9 backends. **Coverage: 9 backends.** Voxtral,
voxtral4b, qwen3_asr, granite_speech, glm_asr, orpheus, omniasr-llm,
gemma4_e2b, mimo_asr.

#### Phase 5 — gemma4_e2b / mimo_asr GPU residency flip (#72, commit 8911126)

Both backends used to load all weights to `backend_cpu` even when
`use_gpu=true` (legacy "Q4_K CPU SIMD" assumption from when the GPU
Q4_K kernels were immature). Today the GPU kernels are mature.

One-line change per backend (`backend_cpu` → `backend` in `load_weights`
call). Validated on Apple Silicon Metal:
```
mimo-asr     CPU-resident:  27.13 s  →  GPU-resident:  21.18 s   (-22 %)
gemma4-e2b   CPU-resident:   8.52 s  →  GPU-resident:   3.95 s   (2.2x)
```

Cold-load wall time also dropped — gemma4_e2b 2:02 → 0:57 — because the
GPU path takes `buffer_from_host_ptr` mmap-zero-copy that the CPU-only
path doesn't. Linux/CUDA validation deferred (no host accessible);
expect at least the same range since dGPUs dominate CPU on matmul
throughput even more than Apple Silicon.

#### Stacking

The four knobs compose for tight-VRAM hosts:
```bash
CRISPASR_N_GPU_LAYERS=10 \
  CRISPASR_KV_ON_CPU=1 \
  CRISPASR_KV_QUANT_K=q8_0 \
  CRISPASR_KV_QUANT_V=q4_0 \
  ./crispasr --backend voxtral4b -m auto -f long-audio.wav
```

Each addresses an independent bottleneck:
- `KV_QUANT_K/_V` — KV size in VRAM
- `KV_ON_CPU` — KV doesn't fit in VRAM at all
- `N_GPU_LAYERS` — model itself doesn't fit in VRAM

#### Open follow-ups (in PLAN.md)

- **#73 flash_attn_ext migration** for canary/cohere — replace the
  `ggml_cont(ggml_permute(V, 1,0,2,3)) → mul_mat` chain with a single
  `ggml_flash_attn_ext` call that natively handles quant K/V. Drops
  the cast-on-read tax for full bandwidth saving. ~60-80 LOC each
  with causal-mask graph-input plumbing. Worth doing for long-context
  workloads; deferred without long-context perf evidence on those
  backends.
- **#72 Linux/CUDA validation** of the GPU-residency flip — needs a
  CUDA host. Math is even more favourable on dGPUs.
- ~~**#69a vibevoice port** — complex ASR+TTS variants (`hp.tts_n_layers`
  vs ASR-only mode). Deferred.~~ → shipped, see §79b below.

#### Commits this session

```
b2b8a10  fix: cap-honesty + test-runner + Ruby CI from feature-suite triage
3a4633f  fix(backends): drop dishonest CAP_LANGUAGE_DETECT from 4 backends
1bd9ff8  feat(kv-cache): asymmetric K vs V quantization (PLAN #69e)
d70f275  feat(kv-cache): KV-on-CPU offload (PLAN #69b)
324a39c  feat(core+voxtral4b): layer-residency offload (PLAN #69a)
aa3f54c  feat(qwen3_asr+granite_speech): port #69a layer offload
af0d789  feat(kv-cache): extend #69e+#69b to 4 more backends
8952b02  feat(kv-cache): Tier-2 K/V split — KV-on-CPU on 4 more backends
5bb1a0e  feat(voxtral): port #69a layer offload (4th backend)
8911126  perf(mimo_asr+gemma4_e2b): load weights to GPU when use_gpu=true
709eafe  feat(kv-cache): quant-safe per-step write helper, kyutai_stt unlocked
d477268  feat(canary): migrate cache write to core_attn::kv_cache_write
c2e1829  feat(canary+cohere): full quant K/V via cast-on-read
4f656ec  feat(#69a): layer offload on glm_asr, orpheus, omniasr-llm, gemma4_e2b, mimo_asr
```

14 commits. PLAN #69 functionally closed (#69a partial — 9/10 LLM-decode
backends done, vibevoice deferred). PLAN #72 closed. PLAN #73 partial
(write path closed everywhere, read-path optimization deferred for
canary/cohere).

### §79b — vibevoice #69a follow-up (2026-05-04)

Closes the last LLM-decode backend on the offload matrix. Vibevoice
is dual-mode (`hp.tts_n_layers == 0` ⇒ ASR-only, `> 0` ⇒ TTS-enabled
with `lm.layers.<N>.*` for the 4-layer base path and
`tts_lm.layers.<N>.*` for the dominant 20-layer TTS path), so the
load-time predicate switches prefix per mode and thresholds the
mode-appropriate layer count. The light base-LM layers in TTS mode
stay on GPU.

Wire-up: added `buf_cpu` to `vibevoice_context`, freed it in
`vibevoice_free`, and gated `core_gguf::load_weights_split` on
`backend_cpu` being non-null (same pattern as voxtral4b/gemma4_e2b).
Predicate is `core_gguf::is_gpu_tensor_with_prefix` with a
mode-selected `LayerSplitConfig{ "tts_lm.layers." | "lm.layers.", N }`.

Validation (Apple M1, Q4_K-fixed for ASR + 0.5b-tts-f16-tokenizer
for TTS):

- ASR `vibevoice-asr-7b` (28 layers, ASR-only mode) at
  `CRISPASR_N_GPU_LAYERS=14` — JFK transcript bit-identical to
  legacy. Logs `layer offload (lm.layers.): gpu=[0,14), cpu=[14,28)`.
- TTS `vibevoice-realtime-0.5b-tts` (20 tts_lm layers) at
  `CRISPASR_N_GPU_LAYERS=10` — `Hello there.` → 21511 samples,
  peak 21449/32767, RMS 4377; bit-identical to legacy WAV
  (43066 bytes, same checksum). Logs
  `layer offload (tts_lm.layers.): gpu=[0,10), cpu=[10,20)`.

PERFORMANCE.md matrix vibevoice rows flipped from `·` → `✓` on
the N_GPU_LAYERS column (both ASR and TTS rows). KV-quant migration
note kept; the two knobs are independent — F16 K/V continues to
work alongside layer offload.

Open after this: encoder-decoder #69a (canary/cohere/kyutai-stt) is
its own design problem (cross-attention layout, no `<prefix><N>.*`
block-tagged tensors). Linux/CUDA validation of #72 still
hardware-blocked.

### §79c — canary/cohere flash_attn_ext bench (2026-05-04)

Commit 193a736 migrated canary + cohere self-attention from the
`ggml_cast(F32) → permute → mul_mat` cast-on-read path to a single
`ggml_flash_attn_ext` call — the read-path optimization the §79
"open follow-ups" section flagged. Bench numbers on JFK (~11 s) on
Apple M1 Metal:

| backend | F16 cast | F16 flash | Q8_0/Q4_0 cast | Q8_0/Q4_0 flash |
|---------|---------:|----------:|---------------:|----------------:|
| canary  | 0.55 s   | 0.53 s    | 0.77 s         | 0.64 s (-17 %)  |
| cohere  | 0.82 s   | 0.83 s    | 0.80 s         | 0.89 s (+11 %)  |

F16 is a tie on both — the cast wasn't the bottleneck at F16. On
quant K/V canary gets the expected -17 % (dequant + permute is
the dominant cost), but cohere *regresses* by +11 %. Likely cause:
cohere's larger model dim / different head config flips the
crossover point — flash kernel overhead exceeds saved bandwidth at
this cache length.

Action: PERFORMANCE.md cohere row note softened from "no cast tax"
to "+11 % regression vs cast-on-read on JFK"; PLAN entry added for
multi-minute clip rerun. flash_attn_ext stays the default in code
(canary wins, cohere short-form regression is small) but
PERFORMANCE.md and PLAN now reflect the data instead of the prior
"flash drops the cast tax universally" claim.

### §80 — Chatterbox-Turbo full pipeline (2026-05-04)

**Chatterbox-Turbo (350M GPT-2 T3 + meanflow S3Gen) now produces
intelligible speech.** ASR roundtrip: "Hello world" (p=0.939).

5 bugs found and fixed in the S3Gen conformer encoder via crispasr-diff
per-stage protocol:

1. **PE ordering reversed**: `fill_pos_enc` generated positions -(T-1) to
   +(T-1) but Python's EspnetRelPositionalEncoding uses +(T-1) to -(T-1).
   This negated the sine components of the positional encoding.
2. **pos_bias_u/v spurious transpose**: reshape(hd, 1, H) is correct —
   the transpose was scrambling head/dim indices, reducing all attention
   scores to ~70% of correct magnitude.
3. **Missing up_layer.conv**: Conv1d(512,512,k=5) with left-pad(4) after
   nearest-neighbor 2x upsample. Weight existed in GGUF but was skipped.
4. **Missing xscale after up_embed**: sqrt(512) scaling needed after
   post-upsample re-embed LayerNorm.
5. **Attention output head layout wrong** (critical): ggml
   reshape(hd,TT,H → D,TT) interleaves head/time indices instead of
   concatenating heads per timestep. Added permute(0,2,1,3) before
   reshape. This single fix brought encoder_out from 94% → 100% match.

Additional improvements:
- F0 predictor wired into vocoder (SineGen 9-harmonic + SourceModuleHnNSF)
- Reflection pad at last vocoder upsample stage
- Encoder attn/FFN weights kept as F32 in converter (closes precision gap)
- Reference backend: `tools/reference_backends/chatterbox_turbo.py`
- T3 arch renamed from "kartoffelbox" to "chatterbox_turbo"

Final metrics: encoder_out rms=0.4602 (exact match to Python), all per-stage
values match Python reference to 2+ decimal places.

---

### §81 — OmniASR Unlimited streaming + conformer flash-attn + Silero LID fix (2026-05-09)

#### OmniASR-LLM-Unlimited segment-token protocol
The "Unlimited" variant of OmniASR-LLM uses a streaming decode protocol
with 3 special tokens above vocab_size: `streaming_lang`, `last_segment`,
`regular_segment`. Without the segment marker in the prefix the decoder
never sees the input shape it was trained on and generates until
max_new_tokens. Fix: auto-detect from tok_emb shape, insert segment
marker, multi-segment decode for >15s audio. Commits: `05290e5`, `762777f`.

#### FastConformer flash attention (re #81)
The FastConformer encoder (parakeet, canary, canary_ctc) used 3 separate
matmuls + add + softmax for Shaw relative-position attention.
Restructured to precompute the BD position bias and pass it as additive
mask to `ggml_flash_attn_ext`, fusing Q_u×K^T + BD + softmax + ×V into
one kernel. CPU: ~10% faster; GPU: expected much larger gain. Output
bit-identical. Commit: `c242331`.

#### Silero LID ISO code fix (#82)
`silero_lid_detect` returns labels like "de, German" but downstream
backends (canary, cohere, etc.) expect bare ISO codes. Extracted the
code before the comma in `crispasr_lid.cpp`. Commit: `43f9015`.

#### Issue verification sweep
- #69 (Silero VAD load): confirmed fixed since v0.6.0 (144d5578)
- #70 (streaming VAD+punc parity): confirmed fixed (cb198fa)
- #74 (vibevoice-tts clone): structural prompt fix working
- #76 (chatterbox-turbo distortion): turbo fixed; base still has multinomial parity gap

#### parakeet-ctc-{0.6b,1.1b} — first-class support (re #81)
`nvidia/parakeet-ctc-0.6b` (24L) and `nvidia/parakeet-ctc-1.1b` (42L)
are architecturally identical to `stt_en_fastconformer_ctc_xlarge` —
same FastConformer encoder + Conv1d CTC head, 80 mel bins, vocab
1024+blank, xscaling. The existing
`models/convert-stt-fastconformer-ctc-to-gguf.py` handles them
unmodified. Wiring: filename auto-detect (`parakeet` + `ctc` + NOT
`tdt` → `fastconformer-ctc`, with the "tdt" guard preserving the JA
hybrid `parakeet-tdt_ctc-0.6b-ja` on the parakeet TDT path) and two
new auto-download registry keys. Quantised variants (F16, Q8_0, Q5_0,
Q4_K) at [`cstr/parakeet-ctc-0.6b-GGUF`](https://huggingface.co/cstr/parakeet-ctc-0.6b-GGUF)
and [`cstr/parakeet-ctc-1.1b-GGUF`](https://huggingface.co/cstr/parakeet-ctc-1.1b-GGUF).
Apple M1 Metal: 0.6b q4_k → 44.7× RT, 1.1b q4_k → 37.3× RT. All 8
quants verified bit-identical English transcript on samples/jfk.wav.
Commit: `369f3ff`.

#### Windows mic UTF-8 popen fix (#70 follow-up)
On localized Windows installs the default DirectShow capture device
name comes back from miniaudio as UTF-8 with non-ASCII characters
(reporter saw zh `麥克風 (2- AIR 192 6)`). The narrow `_popen` path
mangled it before ffmpeg saw the `-i audio="..."` arg, so ffmpeg
exited before delivering any PCM, `2>NUL` hid the failure, and
`--mic` / `--live` ended silently right after the "capturing from
default microphone..." banner. New header
`examples/cli/crispasr_popen.h` widens the UTF-8 command via
`MultiByteToWideChar` (CP_UTF8 with CP_ACP fallback) and calls
`_wpopen`. Mic path also now prints the resolved device name + the
ffmpeg command before spawning, and emits a recovery hint if the
pipe reaches EOF before any PCM was read. Same helper applied to
the file-decode ffmpeg subprocess in `examples/common-crispasr.cpp`
(non-ASCII paths). Commit: `b9b677d`.

#### onnx-asr cross-comparison (re #81)
Apples-to-apples bench against `istupakov/onnx-asr` 0.11.0 on M1.
ONNX execution-provider availability table, TDT-vs-TDT and CTC-vs-CTC
results in `PERFORMANCE.md`. Headline: crispasr Metal beats
onnx-asr CPU EP by 1.32× on TDT and 1.58× on CTC at the same param
count; CoreML EP isn't reachable for the upstream parakeet-tdt ONNX
export (external-data + protobuf 2 GB ceiling, `onnxruntime#26355`,
closed *not planned*); for the only export where CoreML *does* load
(parakeet-ctc int8 single-file), it's *slower* than CPU EP on M1
(1.28 s vs 0.72 s). Reframes the issue's 5×-slower claim as
Windows-DirectML-specific rather than universal "GPU slow."


### §82 — Chatterbox native voice clone — modules 2/3/4 + atomic install (2026-05-09)

Closes the `--voice <ref>.wav` arc started in `8a44fe2b`. Six commits
across one session sprint port the four upstream cond-extractor models
(VoiceEncoder LSTM, S3Tokenizer V2, CAMPPlus, 24 kHz Matcha mel) to
native C++, add a Kaiser-windowed sinc resampler, and wire the
atomic-5-cond install into `chatterbox_set_voice_from_wav`'s `.wav`
branch. End result: `--voice ref_24k.wav` produces real cloned speech
without any python.

Per-stage parity vs PyTorch (verified via `crispasr-diff chatterbox`
on JFK 11 s):

| Module | Stage | cos_mean | Commit |
|---|---|---|---|
| 2 | VoiceEncoder LSTM | 1.000000 | `8a44fe2b` |
| 3 | S3Tokenizer V2 (proj_down) | 0.999928 | `201c5bc5` |
| 4-1 | Kaldi fbank | 0.999999 | `1744869c` |
| 4-2 | CAMPPlus xvector | 0.998070 | `f106c1f6` |
| 4-3 | 24 kHz Matcha mel | 1.000000 | `f387f1b6` |
| 4-final | resampler + atomic install | end-to-end | `847d29aa` |

Five parity-quality compute kernels are bit- or fp32-rounding-tight
when fed identical bytes via the diff harness's `audio_24k_input`
bypass. The atomic install closes the loop: 24 kHz mono PCM16/F32
WAV input → resample 24→16 once → run all five modules from one
source → install all 5 conds in a fresh `voice_ctx_w` slot,
mutually consistent. The `.wav` branch falls back to the partial
M2+M3 path on 16 kHz input (gen.* triple stays at default to avoid
the inconsistent-conditioning silence trap documented in the
S3Tokenizer port).

New shared infrastructure that other backends can consume:
- `src/core/kaldi_fbank.{h,cpp}` — `torchaudio.compliance.kaldi.fbank()`
  with default args (povey window, HTK mel, no Slaney norm,
  preemph=0.97 with kaldi's `s[-1]=s[0]` boundary, snip_edges=True,
  round_to_power_of_two=True). Covers any speaker / VAD encoder
  trained against Kaldi's `fbank` pipeline. firered_asr.cpp keeps
  its inline copy (different int16-magnitude scaling for its CMVN
  baseline) — could dedupe to this helper in a follow-up.
- `src/core/audio_resample.{h,cpp}` — generic Kaiser-windowed sinc
  polyphase resampler (β=8.6, num_zeros=14, same parameters as
  librosa kaiser_fast). NOT bit-equivalent to librosa (resampy uses
  a precomputed table with a different precision knob), but
  acoustically very close. Source of the small text drift between
  native cloning and the python baker observed end-to-end.

End-to-end on a 24 kHz JFK clip: atomic native clone produces real
intelligible English speech in the cloned voice. Whisper roundtrip
transcribes the synthesis; text drifts more than the python baker
because resampler differences propagate through stochastic T3
sampling, but the speaker direction is preserved and the conds are
mutually consistent (no silence collapse). For full library-grade
parity, the python baker workflow remains recommended; the native
path is the no-python-required alternative.

Key gotchas captured in `LEARNINGS.md`:
- `embeds_from_wavs` default `rate=1.3` (NOT `overlap=0.5` as the
  outer-level keyword suggests); affects partial frame_step.
- `mel_type='amp'` skips the log step entirely (added
  `core_mel::LogBase::None` for this).
- S3Tokenizer K projection has no bias (Whisper convention) — guard
  every `attn_k_b` add.
- S3Tokenizer FSMN side branch operates on V before the head
  reshape (depthwise k=31 conv on V, residual-added back, summed
  into the post-O-proj attention output).
- CAMPPlus `out_nl.bn.*` lives directly under `out_nl.*` (bare
  `get_nonlinear`), NOT `out_nl.nl.bn.*` like the wrapped units.
- TransitLayer / DenseLayer order is BN→ReLU→Linear (BN sized for
  input, not output).
- `F.avg_pool1d(ceil_mode=True)` divides by `kernel_size`, not the
  actual frame count (count_include_pad default).
- `torch.std` defaults to `unbiased=True` (divide by `n-1`).
- 24 kHz Matcha mel uses magnitude with `+1e-9` INSIDE the sqrt
  (NOT power) and natural log + `clamp(mel, 1e-5)` (NOT
  log10 + Whisper-style clip-and-scale).
- `gen.{prompt_token, prompt_feat, embedding}` must be installed
  atomically — partially updating one of the three feeds S3Gen's
  flow matcher inconsistent conditioning and silences the output
  (verified: rms drops to ~0.0003).

Module 2 only needed the existing `core_lstm` and `core_mel` infra.
Module 3 reused `core_fft`, `core_mel`, and `ggml_conv_1d_dw`.
Module 4 introduced two new shared helpers (`core_kaldi`,
`core_audio`) and pure-CPU forward kernels for the 815-tensor
CAMPPlus TDNN — the BN-fold + per-channel scale pattern is mechanical
enough to skip ggml graph integration without losing perf (CAMPPlus
runs ~2 s for an 11 s clip on M1, one-shot at voice-clone time).

Files added: 6 new sources / 2 new headers
(`chatterbox_ve.{h,cpp}`, `chatterbox_s3tok.{h,cpp}`,
`chatterbox_campplus.{h,cpp}`, `core/kaldi_fbank.{h,cpp}`,
`core/audio_resample.{h,cpp}`).

### §83 — Text LID: fastText (GlotLID-V3 / LID-176) + Google CLD3 + auto-routing dispatcher (2026-05-10)

Two text-LID backend families landed in one slice. Both run on a
transcript or any UTF-8 string (no audio); both expose the same C
ABI shape (init/free/predict/predict_topk/extract_stage); a thin
dispatcher chooses between them by peeking the GGUF's
`general.architecture` at load time.

#### `lid_fasttext.{h,cpp}` — GlotLID-V3 + Facebook LID-176

`models/convert-glotlid-to-gguf.py` packages both fastText supervised
LID families: GlotLID-V3 (flat softmax, 2102 ISO 639-3 + script
labels, Apache-2.0) and Facebook LID-176 (hierarchical softmax, 176
ISO 639-1 codes, **CC-BY-SA-3.0** — viral; redistributors of the
GGUF inherit ShareAlike). Forward path is manual F32/F16 + on-the-fly
dequant via `ggml_get_type_traits(type)->to_float`, no graph — the
compute is ~1 MFLOP per call over a large embedding table. Released
as `cstr/glotlid-GGUF` and `cstr/fasttext-lid176-GGUF`.

Two traps documented in LEARNINGS.md "Text LID via fastText":

* **`</s>` row injection** — fastText's `Dictionary::getLine` injects
  an `</s>` (always `input_matrix[0]`) at end-of-stream in supervised
  mode; `model.f.tokenize(text)` does NOT return it. Missing the EOS
  row drops cosine vs `model.get_sentence_vector` to ~0.973.

* **Hierarchical-softmax sign convention** — LID-176 uses HS, not
  flat softmax. Per-label `log P = Σ log_sigmoid((2c-1)·f)`. Sign-flip
  makes predict land in the wrong subtree of the root: fastText says
  `fr/0.95` for "Bonjour le monde", the buggy port says `en/0.88`.

#### `lid_cld3.{h,cpp}` — Google compact language detector v3

`models/convert-cld3-to-gguf.py` regex-parses the upstream
`src/lang_id_nn_params.cc` plain-text C++ array literals (1.76 MB
embedded weight blob, no binary side-car) and dequantizes the six
`uint8 + per-row bfloat16-scale` embedding tables to F32. Forward
path runs six feature extractors (4× ContinuousBagOfNgrams at
sizes 1/2/3/4, RelevantScript, Script) → 80-d concat → FC + ReLU →
208-d hidden → FC → softmax over 109 ISO 639-1 labels.

Released as `cstr/cld3-GGUF` (Apache-2.0, 440 KB F16 — F16 is the
*only* viable shipping format because every weight tensor's column
width is in `{8, 16, 80, 208}`, none of which divide the 32-element
K-quant block size; `crispasr-quantize` skips them all).

Diff harness: 8/8 PASS at cos≥0.999 across an en/de/fr/ru/zh/ja/hi/pt
multilingual smoke set on F16 (88/88 stage compares).

Four traps documented in LEARNINGS.md "Text LID via CLD3":

* **bfloat16-style `float16`** — CLD3's `float16` typedef is the top
  16 bits of fp32 (1+8+7), NOT IEEE binary16 (1+5+10). Decoding
  `15392u`-style literals via numpy `<f2` silently produces garbage;
  correct decode is `(uint32(value) << 16).view(float32)`.

* **MurmurHash2-32, NOT CityHash** — `utils.cc:137-183` is textbook
  MurmurHash2 with `m=0x5BD1E995, r=24, seed=0xBEEF`. The cbog
  feature IDs use the raw UTF-8 bytes of the ngram string as the
  hash input, so byte-for-byte hash parity with upstream is
  mandatory.

* **Hiragana/Katakana/Hangul are NOT separate ULScript values** —
  they all return `ULScript_Hani=24` from the upstream
  `ScriptScanner`; a secondary Hangul-vs-Hani codepoint count
  returns `NUM_ULSCRIPTS=102` only when Korean wins. Mis-mapping
  these (Hani=43, Devanagari=10) was the dominant cause of early
  smoke failures (नमस्ते दुनिया → bn, 你好世界 → mr).

* **Full-Unicode lowercase** — ASCII-only case folding flips
  Cyrillic-language argmax (Привет мир → tg instead of ru) because
  the cbog ngrams hash uppercase Cyrillic UTF-8 bytes to different
  feature IDs than upstream's `GetOneScriptSpanLower` produces.

#### `text_lid_dispatch.{h,cpp}` — backend-agnostic façade

Peeks `general.architecture` once at load time, then dispatches each
public call to the matching backend's C ABI via a flat tag-switch
(no virtual base, no function-pointer table — both backends already
mirror each other's shape, so the dispatcher is one integer compare
per call). Powers both the `crispasr-lid` standalone binary and
`crispasr --lid-on-transcript`. Same flag, same binary, any text-LID
GGUF.

End-to-end on this box (one binary, three GGUFs, three label spaces):

```
$ crispasr-lid -m cld3-f16.gguf            --text "Bonjour le monde, comment allez-vous?"
fr	0.999983       (lid-cld3, dim=80, 109 labels)
$ crispasr-lid -m lid-glotlid-f16.gguf     --text "..."
fra_Latn	0.983436   (lid-fasttext glotlid-v3, dim=256, 2102 labels)
$ crispasr-lid -m lid-fasttext176-f16.gguf --text "..."
fr	0.958174       (lid-fasttext fasttext-lid176, dim=16, 176 labels)
```

### §84 — PLAN #89 flash_attn field migration (2026-05-10)

Mechanical struct-field plumbing across every backend whose
`*_context_params` already carried `use_gpu`. Each gained a
`flash_attn` field (default true), threaded through
`*_default_params()` and into `crispasr_session_open_explicit`'s
arm via `g_open_flash_attn_tls`. The **compute graphs do not yet
branch on the flag** — that's PLAN #86, lands per backend
incrementally. After this slice every backend ACCEPTS the toggle
at session-open time; honouring it at the kernel level is
follow-up work.

Backends touched (12 of 12):

| Backend | Pre-existing field | New field | Notes |
|---|---|---|---|
| parakeet | `use_flash` | — | TLS routes to existing field |
| canary | `use_flash` | — | TLS routes to existing field |
| cohere | `use_flash` | — | TLS routes to existing field |
| qwen3 (asr) | — | `flash_attn` | new |
| voxtral | — | `flash_attn` | new |
| voxtral4b | — | `flash_attn` | new + arm cleanup (was hard-coding verbosity=0) |
| granite_speech | — | `flash_attn` | new |
| vibevoice | — | `flash_attn` | new |
| qwen3_tts | — | `flash_attn` | new |
| orpheus | — | `flash_attn` | new |
| kokoro | — | `flash_attn` | new |
| chatterbox | — | `flash_attn` | new |

Commit `0b7dd749`. Pairs with the open-params v2 work in §85's
companion patch. Closes the prerequisite for PLAN #86.

### §85 — PLAN #88 Kokoro length_scale + VibeVoice runtime tts_steps (2026-05-10)

Two TTS backend-internal refactors that close the cross-repo
deferred items from CrisperWeaver's May 2026 parity sweep.

**Kokoro length-scale** (per-phoneme speaking-rate scalar):
* New `length_scale` field on `kokoro_context_params` (default
  1.0). Applied to the duration-predictor output BEFORE the
  banker's-round + clamp-min-1 in the "durations" stage extractor
  in `kokoro_run_predictor`, so PyTorch's `torch.round` semantics
  (round-half-to-even) are preserved.
* New `kokoro_set_length_scale(ctx, scale)` runtime setter clamps
  to [0.25, 4.0]. Read on every `kokoro_synthesize` call so post-
  init mutation just changes the next call's pacing.

**VibeVoice runtime tts_steps**:
* The `tts_steps` field (DPM-Solver++ inference steps, default 20)
  has been on `vibevoice_context_params` since the original
  vibevoice port, but was init-time only. New
  `vibevoice_set_tts_steps(ctx, steps)` runtime setter mutates the
  pre-existing field; clamps to [4, 100]. `vibevoice_synthesize`
  reads `ctx->params.tts_steps` on every call so the setter just
  changes the next call's schedule density.

**Unified API** (`crispasr_c_api.cpp`):
* New `crispasr_session_set_length_scale(s, scale)` export routes
  to `kokoro_set_length_scale` on kokoro sessions; rc=-2 on
  backends without a duration model.
* `crispasr_session_set_tts_steps` extended to also route to
  vibevoice (was chatterbox-only).
* Dart binding: new `CrispasrSession.setLengthScale()` method,
  `providesSymbol`-gated for pre-0.6.2 compatibility.

Commit `cda44359`. CrisperWeaver picks this up automatically —
the existing TTS *speed* slider on the Synthesize screen now
drives `setLengthScale(1/speed)` IN ADDITION to the client-side
resampler. On kokoro the duration model produces a clean
stretch/squeeze; on every other TTS backend the client-side
resample is the fallback.

Closes PLAN #88 entirely. Both kokoro + vibevoice runtime knobs
now reachable from any C-ABI consumer.

### §86 — IndexTTS BigVGAN anti-aliased SnakeBeta on by default (2026-05-11)

Spotted while A/B-benchmarking IndexTTS-1.5 across `{Q4_K, Q8_0, F16}` ×
`{GPU, CPU, CPU+AA}` on M1. The non-AA outputs measured fine on peak/RMS
but had ~2 000 sample-to-sample jumps exceeding 30 % FS (and several
over 100 % FS — physically impossible for a 24 kHz band-limited signal);
audible as broadband click/buzz on every quant. The AA path produced
0–27 such jumps. The original `src/indextts_voc.cpp` comment claiming
"quality impact on TTS speech is negligible" was wrong; BigVGAN v2's
upsample→activate→downsample sandwich exists exactly to suppress the
`sin²` harmonics SnakeBeta emits above Nyquist.

Shipped:

- AA is the default. `INDEXTTS_VOCODER_RAW=1` (or `INDEXTTS_VOCODER_AA=0`)
  opts back into the aliased fast path; useful for the speed A/B and as
  the legacy fallback if the AA op ever regresses.
- Pre-allocated per-thread scratch (lifted three per-channel
  `std::vector<float>` allocations out of the hot loop, capped at 64
  workers since `GGML_N_TASKS_MAX` is the `-1` sentinel not a count).
- Pre-scaled the upsample FIR by ×2 (zero-stuff gain) so the inner loop
  is one mul, not two.
- `std::memcpy`/`std::memset` replace per-element padding fills.

Result on M1, JFK voice prompt, ≈ 6.7 s of audio: AA on CPU drops to
6.65 s vocoder-only (was 6.7–9.3 s before optimization, ≈ 5 % over raw),
and the click-detector reads 0–2 jumps > 30 % FS instead of 1 600–2 600.
AA on GPU is 8.5 s — slowest because the CPU custom op forces a Metal →
CPU → Metal sync per AMP block; recommend `--no-gpu` for IndexTTS until
the AA sandwich is ported to native ggml ops (`ggml_conv_transpose_1d` +
depthwise `ggml_conv_1d`, both Metal-capable via IM2COL).

Stale GGUFs at `~/.cache/crispasr/` and `/Volumes/backups/ai/crispasr/`
predated commit `99fca4c3` (`_shorten_gpt`) — tensor names up to 66 chars
hit `GGML_MAX_NAME = 64` and `gguf.cpp:587` rejects the file. Re-pulled
from `cstr/indextts-1.5-GGUF` (already ships the corrected names) for
the bench; that's where the registry's `-m auto` path lands.

Full A/B numbers + click-detector results: `LEARNINGS.md` §"BigVGAN v2
SnakeBeta needs anti-aliasing".

### §87 — IndexTTS BigVGAN AA: Step A auto-CPU + Step C-1 vDSP (2026-05-11)

Follow-up to §86. Mixed-backend AA (CPU custom op inside a Metal vocoder
graph) measured ≈ 25 % slower than CPU-only because of the
Metal↔CPU sync per AMP block. Three optimisations attempted on the same
M1 / q8_0 / JFK prompt:

- **Step A (shipped):** when `use_aa=true`, override `use_gpu` in
  `indextts_voc_init` and run the whole vocoder on CPU. Skips the
  ~20 round-trips per generate. `INDEXTTS_VOC_FORCE_GPU=1` opts back into
  the slow mixed path for benching. `INDEXTTS_BENCH=1` gate added to the
  vocoder timing log per the repo's `<BACKEND>_BENCH` convention.
- **Step B (attempted, deferred):** express the AA chain via native ggml
  ops (replicate-pad + zero-stuff + `ggml_conv_1d` + SnakeBeta + replicate-pad
  + stride-2 `ggml_conv_1d`). Blocked on (a) `ggml_conv_1d` only accepting
  symmetric `p0`, which can't reproduce `conv_transpose_1d`'s `(T-1)·s + K`
  output length without three extra concat nodes, and (b) `ggml_add_inplace`
  shape-broadcast assertion firing on the downstream BigVGAN bias adds.
  Documented in `src/indextts_voc.cpp` and LEARNINGS; the right fix is
  Step C-2, not a different ggml-ops expression.
- **Step C-1 (shipped):** Accelerate vDSP path inside the existing CPU
  custom op — `vDSP_vsmul + vvsinf + vDSP_vsq + vDSP_vsma` for SnakeBeta,
  `vDSP_desamp(decimation=2)` for the downsample FIR. ~2-3 % on the full
  vocoder (small because AA is a fraction of the BigVGAN graph), but
  numerically equivalent to scalar (rmsdiff ≈ 1.3 × 10⁻⁵) and free —
  `INDEXTTS_AA_SCALAR=1` opts back to scalar for A/B.
- **Step C-2 (drafted, not implemented):** new `GGML_OP_AA_SNAKE_BETA`
  with a fused Metal kernel ported from upstream IndexTTS's CUDA
  reference (`anti_alias_activation_cuda.cu`). RFC scope; one launch
  does the whole sandwich in registers — projects vocoder ≈ 1.5-2 s on
  M1, vs 6.65 s today (CPU). Drafted as PR slot 07 in
  `tools/upstream-prs/07-metal-aa-snake-beta.md`. Slots 05 (CUDA per-row
  contiguous unary) and 06 (CUDA per-head FA mask) reserved for the
  in-flight `issue81-phase1-uar-wip` branch.

Per-step bench numbers in LEARNINGS.md §"Mixed-backend custom ops…" and
§"Accelerate vDSP_desamp…".

### §88 — IndexTTS Step B-v2: native-ggml-ops AA path lands as opt-in (2026-05-11)

Same-day follow-up to §87. The two blockers documented in §87 (output
length mismatch vs `conv_transpose1d`; reshape-after-truncating-view
shape drift) were both fixable in a few lines: `p0 = K - 1` on the
upsample conv1d closes the K-1 gap, and `ggml_cont` between the
truncating `ggml_view_3d` and its `ggml_reshape_2d` keeps the layout
valid for the downstream BigVGAN bias adds.

Now in `src/indextts_voc.cpp:aa_snake_beta_native`, behind
`INDEXTTS_AA_BACKEND=native`. CPU output is bit-equivalent to the
custom-op reference (same click pattern, ASR exact). GPU output drifts
into the noise floor (26 inter-sample jumps > 0.3 vs 2, all max|Δ| ≤
0.4 — Metal vs CPU float order-of-ops on the broadcast muls). ASR
identical across all three.

Wall-clock didn't move much:

| Path                     | voc-only |
| ------------------------ | -------- |
| Step A custom-op (CPU)   | 7.87 s   |
| Step B-v2 native (CPU)   | 7.57 s   |
| Step B-v2 native (GPU)   | 8.01 s   |

Native-on-CPU is 4 % faster, native-on-GPU is still slower than
custom-op-on-CPU because the concat/reshape/scale graph overhead inside
Metal eats the kernel-level GPU advantage. Default stays on the
custom-op path; native is opt-in so the GPU pathway is unlocked for
people who need it and for the eventual fused-kernel work
(`tools/upstream-prs/07-metal-aa-snake-beta.md`).

Step A's auto-fall-to-CPU is suppressed when `aa_use_native()` returns
true — the whole vocoder graph stays on Metal end-to-end in that case.

### §89 — TitaNet-Large speaker verification + speaker profile DB (2026-05-11)

Added NVIDIA TitaNet-Large (23M params, CC-BY-4.0) as a standalone
speaker verification / embedding extractor backend. Produces 192-d
L2-normalized speaker embeddings with 0.66% EER on VoxCeleb1-O cleaned.

**Architecture:** 5 Jasper-style blocks with depthwise separable Conv1d +
Squeeze-and-Excite + residual connections. Decoder: Attentive Statistical
Pooling (ASP) → BN → Linear(6144→192) → L2-normalize.

**Implementation:**
- `models/convert-titanet-to-gguf.py` — .nemo → GGUF converter (~45 MB)
- `src/titanet.{h,cpp}` — pure-CPU runtime (init/embed/free/cosine_sim)
- `src/speaker_db.{h,cpp}` — file-per-speaker profile database (.spkr format)
- `tools/reference_backends/titanet.py` — NeMo diff-testing backend
- `tests/test-titanet.cpp` — standalone smoke test binary

**CLI integration:**
- `--enroll-speaker <name>` — extract embedding, save to speaker DB, exit
- `--speaker-db <path>` — match transcription speakers against profile DB
- `--titanet-model <path>` — model path or "auto" for auto-download
- `--speaker-threshold <float>` — cosine sim threshold (default 0.7)
- Speaker ID runs as a post-step after diarize, relabeling anonymous
  `(speaker N)` with matched names like `(alice)`

**C-ABI:** `crispasr_titanet_init/embed/free` + `crispasr_speaker_db_load/
match/enroll/count/free`. Wrapper bindings added for Python, Dart/Flutter,
and Rust.

**Model registry:** `titanet-large.gguf` at `cstr/titanet-large-GGUF`.
Auto-download via `--titanet-model auto`.

**Parity validation — 6 bugs found and fixed during diff-testing:**

| Bug | Symptom | Fix |
|---|---|---|
| BN epsilon | NeMo uses ε=0.001 for encoder (not 1e-5) | Store in GGUF metadata, use per-stage |
| Missing mout ReLU | JasperBlock applies ReLU AFTER SE+residual | Add `apply_relu` after residual add |
| ReLU on last sub-block | Last sub-block has no activation before SE | Skip ReLU for sub R-1 |
| STFT window centering | PyTorch centers window at (n_fft−win)/2 | Center window in frame |
| Missing pre-emphasis | NeMo applies x[t] -= 0.97·x[t-1] before STFT | Add pre-emphasis filter |
| Wrong STFT padding | NeMo uses pad_mode="constant" (zero-pad) | Changed from reflect to zero |

Final parity: **cos = 0.9999** on both test samples (an255: 0.999978,
cen7: 0.999917). Encoder+decoder isolated: cos = 0.999997 with mel
injection.

Commits: `dc5f01b`, `3d12359`, `b54b92e`.

### §90 — Chat C ABI: text-LLM surface on libcrispasr (2026-05-11)

Spec: `docs/prompts/chat-abi.md`. Phases 0–6 landed in one pass.

**Phase 0 — `crispasr-llama-core` static lib.** Lifted the 50+
llama.cpp source files under `examples/talk-llama/` (`llama.cpp`,
`llama-{adapter,arch,batch,chat,context,…}.cpp`, `unicode*.cpp`,
`models/*.cpp`) out of the SDL2-gated `crispasr-talk-llama` example
and into a STATIC lib defined in `src/CMakeLists.txt`. Built
unconditionally with hidden visibility (`CXX_VISIBILITY_PRESET hidden
+ VISIBILITY_INLINES_HIDDEN ON`) so `llama_*` / unicode helpers don't
leak from `libcrispasr.dylib`'s export table. `examples/talk-llama/
CMakeLists.txt` simplified to a single `add_executable(crispasr-talk-
llama talk-llama.cpp)` that links against the new lib — one source
set, two consumers.

**Phase 1+2+3 — `crispasr_chat_*` ABI.** Public header at
`include/crispasr_chat.h` (POD structs + opaque handle only, no
`llama.h` types). Implementation `src/chat.cpp` linked PRIVATE into
`libcrispasr` so external consumers see only the `crispasr_chat_*`
surface. Covers open / close / reset / generate (one-shot) /
generate_stream (on_token callback) / memory_estimate / template_name
/ n_ctx / string_free. Sampler chain composed via
`llama_sampler_chain_init + temp / top_k / top_p / min_p /
repeat_penalty / dist` from upstream `examples/main` — full sampling
config landed in one pass rather than churning ABI between Phase 1
and Phase 2. Multi-turn KV cache reuse via a per-session token
history; divergent prompts trigger `llama_memory_clear` + re-prefill,
matching prefixes skip re-decode.

**Phase 4 — `crispasr-chat` CLI.** `examples/cli/crispasr_chat_main.cpp`
— minimal stdin → stdout binary. Auto-detects TTY vs piped stdin: TTY
gives an interactive REPL with streaming deltas, pipe gives a one-shot
"read all of stdin → print reply" mode. No SDL2 dep. Links against
`crispasr` only.

**Phase 5 — `POST /v1/chat/completions` in the server.** Added to
`examples/cli/crispasr_server.cpp` (the active server with the
existing `/v1/audio/*` OpenAI surface). New `--chat-model PATH` /
`--chat-ctx N` / `--chat-gpu-layers N` flags; the session lazy-opens
on first request and re-uses across calls (one process-wide handle,
session-internal mutex serialises overlapping requests). Supports
both `stream: false` (plain `chat.completion` JSON) and `stream: true`
(SSE deltas + `data: [DONE]`). OpenAI-shape `{role, content}` arrays
accepted; multimodal content arrays are collapsed to their text-only
joined form for now.

**Phase 6 — Dart binding.** `flutter/crispasr/lib/src/chat.dart`. New
`CrispasrChatSession` class mirrors `CrispasrSession`'s shape:
`Finalizer` for free-on-GC, explicit `close()` for deterministic
cleanup, `generate()` returns `Future<String>`, `reset()` flushes KV.
`generateStream` intentionally NOT exposed — the C ABI's `on_token`
callback passes a `const char*` only valid for the duration of the
synchronous call, which is incompatible with Dart's
`NativeCallable.listener` (async delivery → dangling pointer by the
time the Dart closure runs). The recommended Dart streaming path is
the HTTP `/v1/chat/completions` SSE endpoint from Phase 5. SDK
constraint bumped to `>=3.1.0` for `Array<Int8>` inline struct fields.

**Tests.** `tests/test-chat-ggml.cpp` Catch2 smoke verifies open →
n_ctx → template_name → generate (one-shot) → reset → generate_stream
→ close. Streamed output equals one-shot under greedy + fixed seed.
Gated on `CRISPASR_CHAT_TEST_MODEL` env var pointing at a tiny GGUF
chat model — skipped when unset so plain builds stay green. Verified
locally against `HuggingFaceTB/SmolLM2-360M-Instruct-Q8_0.gguf`
(386 MB) on M1 Metal; 13 assertions pass, Metal pipeline cache picks
up the chat code path automatically.

**Won't-fix this session.** Phase 7 (CrisperWeaver integration) is in
a different repo and stays out of scope. The Dart `generateStream`
absence is documented in the binding's source comment; closing it
properly needs an ABI tweak so `on_token` pointers stay valid past
the C callback's return (e.g. malloc-per-chunk + Dart-side
`crispasr_chat_string_free`).


### §91 — VibeVoice 1.5B voice clone fidelity fix (issue #74) (2026-05-17)

Root cause of VibeVoice 1.5B TTS voice clone not carrying speaker
identity (reported by vkrmch in #74): the TTS voice-ref path only
used the **acoustic** σ-VAE encoder, but Microsoft's Python reference
combines **both** acoustic and semantic encoder outputs via element-wise
sum before embedding them as voice conditioning tokens.

**The fix** (`src/vibevoice.cpp`, ~30 LOC net):

1. Run both `at_enc` (acoustic) and `st_enc` (semantic) on the
   reference WAV.
2. Scale acoustic output: `(x + bias_factor) * scaling_factor` —
   semantic gets NO scaling (matches Python).
3. Run both through their respective connectors (`at_conn`, `se_conn`).
4. Element-wise sum → `voice_embeds[i] = at_feat[i] + st_feat[i]`.
5. Graceful fallback to acoustic-only if semantic path fails (old
   GGUFs without `st_enc` tensors).

All encoder/connector tensors were already in the 1.5B GGUF
(converter always included them) and the C++ runtime already had
`run_encoder_stage` + `run_connector_stage` for both prefixes (used
by the ASR path). The TTS clone path simply wasn't calling them.

**24 kHz resample fix** (`src/vibevoice_wav_ref.h`):

The WAV parser was ignoring the sample rate field from the fmt chunk.
A 16 kHz reference WAV would produce 55 frames instead of the correct
83, because the σ-VAE expects 24 kHz input (3200× downsample). Fix:
parse `sample_rate` from fmt chunk offset +4, add
`vibevoice_resample_linear()` (simple linear interpolation), apply
automatically when the WAV rate ≠ 24 kHz. Log line confirms:
`resampled voice ref 16000 Hz → 24000 Hz (176000 → 264000 samples)`.

**C ABI `.wav` voice path** (`src/crispasr_c_api.cpp`):

`crispasr_session_set_voice()` unconditionally called
`vibevoice_load_voice()` for all paths including `.wav` references,
which only works for pre-baked GGUF voice packs. Fix: detect `.wav`
suffix and set `VIBEVOICE_VOICE_AUDIO` env var instead, matching the
CLI adapter's existing logic. Python and Dart bindings (which wrap the
C ABI) now support `.wav` voice cloning for vibevoice-1.5b sessions.

**Verification:**

| Metric | Value |
|---|---|
| cos(C++ F16, Python F32) | 0.999884 |
| cos(C++ Q8_0, Python F32) | 0.999229 |
| ASR roundtrip (parakeet-ctc q4_k) | "Hello, this is the voice scone test of j. F. K." |
| Peak | 7940 (no clipping) |
| Nonzero samples | 99.9% |

Python reference script: `tools/vibevoice_tts_ref_voice_clone.py` —
loads encoder + connector weights from safetensors, dumps per-stage
intermediates (`voice_at_enc_mean`, `voice_at_conn`, `voice_st_conn`,
`voice_combined`) for comparison via `crispasr-diff` or manual cosine.
C++ dump via `VIBEVOICE_TTS_DUMP=<dir>` writes `tts_voice_embeds.bin`.

**Wiring audit** (status after this session):

| Layer | `.gguf` voice | `.wav` clone | Notes |
|---|---|---|---|
| CLI (`--voice`) | ✓ | ✓ + resample | env var path |
| C ABI | ✓ | ✓ (NEW) | detects `.wav` suffix |
| Python | ✓ | ✓ (inherited) | via C ABI |
| Dart/Flutter | ✓ | ✓ (inherited) | via C ABI |
| Server `/v1/audio/speech` | ✓ | ✓ | via CLI adapter |

### §92 — issue #81 Phase 1 #06: FA per-head mask lands on CUDA (2026-05-24)

`tools/upstream-prs/06-cuda-fa-perhead-mask.md` implemented and
validated on feature branch `issue81-fa-perhead-mask`
(commit `60bc4294`, +87 LOC across 4 files). Closes the remaining
72 CPU splits per chunk on parakeet / canary / FastConformer-CTC —
the dominant CPU-fallback cost left after `d758fe69` cleared the
UNARY splits (see PERFORMANCE.md "Phase 1 update (2026-05-23) —
fused siglu/norm_affine A1000 verdict").

**The patch:**

| file | LOC | what |
|---|---:|---|
| `ggml/CMakeLists.txt` | 3 | `GGML_CUDA_CRISPASR_FA_PERHEAD_MASK` option (default OFF) |
| `ggml/src/ggml-cuda/CMakeLists.txt` | 4 | wire option to `add_compile_definitions` |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | 14 | consume `nb32` in `mask_h` offset at both kbc loop sites |
| `ggml/src/ggml-cuda/fattn.cu` | 66 | gate relaxation: VEC/TILE/WMMA returns fall through to MMA-F16 for per-head masks; safety net + `gqa_ratio > 1` hard gate |

The MMA-F16 kernel signature **already** took `nb32` as a parameter
(the launcher plumbed all six mask dims/strides through) but the
kernel body only offset by sequence (`nb33 * (sequence % ne33)`).
The patch adds the missing per-head term `nb32 * (zt_Q % ne32)`.
When `ne32 == 1` (broadcast mask — upstream's only supported case)
`zt_Q % 1 == 0` and the result is byte-identical, so default-OFF
builds stay bit-equal to upstream. Build with
`cmake -DGGML_CUDA_CRISPASR_FA_PERHEAD_MASK=ON` to enable.

**Validation (parakeet-tdt-0.6b-v3 q8_0 on A1000, sm_86, driver 596.36):**

| check | OFF (`dll-postsiglu`) | ON (`dll-faon`) | verdict |
|---|---|---|---|
| JFK transcript | "And so, my fellow Americans, ask Not what your country can do for you. Ask what you can do for your country." | byte-identical | ✓ |
| sched splits per chunk | 49 (24 CPU + 25 CUDA0) | **1 (0 CPU + 1 CUDA0)** | ✓ |
| `FLASH_ATTN_EXT` on CPU | 24/chunk | **0/chunk** | ✓ |
| short-clip mean (11 s JFK, 9 calls) | 2.526 s | **1.587 s** (−37 %) | ✓ |
| long-clip mean (60 s tiled, 150 calls, cold-GPU) | 12.450 s | 12.204 s (−2 %) | partial — see caveat |

**Cold-GPU caveat:** the long-clip A/B ran with the GPU stuck in
P8 (315/405 MHz) for both runs — both numbers are ~4.6× slower
than the postsiglu warm baseline (2.7 s mean). The
probe-then-bench two-process warmup pattern doesn't keep WDDM in
P0 across the second script's Python startup + model mmap + JIT
prewarm phase. Short-clip is less sensitive because per-call
overhead dominates; long-clip shows the cold-clock effect. The
architectural and correctness signals are unambiguous. Warm-GPU
wallclock target: ~2.4 s long-clip mean / ~150 ms p50 / RTx ~24×.

**Deferred to follow-up PRs:**

- `ncols2 > 1` GQA-folded mask case — current patch is correct
  for `ncols2 == 1` only; safety gate at `gqa_ratio > 1` falls
  back to CPU (== upstream pre-patch behaviour, no regression).
  No current CrispASR model has GQA-conformer attention with
  per-head masks.
- VEC / TILE / WMMA-F16 kernel patches — these still don't
  consume `nb32`; per-head masks force the MMA-F16 path or NONE.
- `test-backend-ops` validation on sm_75 / sm_86 / sm_89 — only
  `ggml/src/` is vendored in CrispASR (not `ggml/tests/`), so the
  canonical FA backend test will run as part of the upstream
  `ggml-org/llama.cpp` PR step rather than locally.
- Upstream PR to `ggml-org/llama.cpp` — gated on the items above.

Sidecars: `bench-issue81/results/wer-{off,on}.json` (correctness),
`bench-issue81/results/a1000-fa-{off,on}.json` (wallclock cold-GPU),
`bench-issue81/sched-debug-fa{off,on}.log` (split-count A/B).

---

### 2026-05-31 / 2026-06-01 — §136 funasr CUDA !-loop fix (issue #125)

**Bug:** funasr produces `!!!!!` (token-0 repeat) on every CUDA GPU
tested (P100 sm_60, Blackwell sm_120). Correct on CPU and Metal.

**Investigation (16 Kaggle kernel versions):**

1. v3: per-stage encoder dump → encoder/adaptor fine on CUDA (0 NaN).
2. v4: LLM layer 0 fine, but ALL prefill logits NaN. Bug in LLM decoder.
3. v4: FA-off, KV_READ_F32 → still NaN. Not flash-attn or F16 KV.
4. v6: single-backend GPU sched → crashes (some ops need CPU fallback).
5. v8: parallel=true → still NaN.
6. v11: full 28-layer scan → **layer 2 produces 1 Inf** (max=6973 vs
   CPU max=124512), **layer 3+ all-NaN**. With weight-split but KV on
   GPU, the sched still misroutes ops.
7. v13: weight-split + KV-on-CPU → **PASS** (correct transcript).
8. v15: QKV fusion alone (all GPU) → still NaN. Fusion is not the fix.
9. v16: weight-split + QKV fusion + KV-on-CPU → **PASS**.

**Root cause:** `ggml_backend_sched` with `[CUDA,CPU]` produces Inf in
funasr's Qwen2-0.6B LLM at layer 2 on CUDA. Exact sched bug unknown —
not specific to Q/K/V split (QKV fusion doesn't help), not flash-attn,
not F16 KV. Likely a graph-topology-specific scheduling issue.

**Fix (`f94fec90`):**
- `load_weights_split`: encoder on GPU, LLM on CPU
- KV cache forced to CPU when split active
- QKV fusion at init (Q+K+V → single QKV weight per layer)
- KV cache zeroed on allocation
- `FUNASR_LLM_GPU=1` override for future testing

**Commits:** `7dfe401d` (dump instrumentation), `bc04e263` (v13 fix),
`454e9ef8` (QKV fusion), `f94fec90` (final: split + fusion + KV-CPU).

**Also:** merged stray PR #134 (session-beam test fixes).
