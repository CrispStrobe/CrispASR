# CrispASR — Implementation plan for remaining work

This document details how each remaining roadmap item would be
implemented. It's written for a fresh session that hasn't seen the
prior conversation — every item is self-contained with file paths,
line numbers, approach, risks, and verification steps.

**Current state (April 2026, v0.3.0):** 11 ASR backends, unified CLI,
OpenAI-compatible server, shared `src/core/` library (mel, ffn,
attention, gguf_loader, greedy_decode, bpe), ground-truth diff infra,
CI on 6 platforms + 3-job lint.

---

## Table of contents

1. [voxtral4b audio encoder → encoder_self_attn()](#1-voxtral4b-audio-encoder-migration)
2. [Qwen3 forced aligner as generic timestamp provider](#2-qwen3-forced-aligner)
3. [Granite µP scale extraction into core/attention.h](#3-granite-µp-scale)
4. [Scheduler reuse audit (stale TODO cleanup)](#4-scheduler-reuse-audit)
5. [Reference backends for parakeet/canary/cohere](#5-reference-backends)
6. [Best-of-N sampling for LLM backends](#6-best-of-n-sampling)
7. [Native voxtral4b streaming protocol](#7-voxtral4b-native-streaming)
8. [Audio understanding / Q&A mode for voxtral 3B](#8-voxtral-audio-qa)
9. [Parakeet TDT decoder ggml graph port](#9-parakeet-tdt-gpu)
10. [Granite encoder ggml graph port](#10-granite-encoder-graph)
11. [WebSocket streaming server](#11-websocket-streaming)
12. [Pipeline template consolidation](#12-pipeline-template)
13. [canary_ctc aligner CPU fallback](#13-canary-ctc-fallback)
14. [Misc cleanup items](#14-misc-cleanup)

---

## 1. voxtral4b audio encoder migration — DONE

**Status:** Completed. Added `bool permute_cont` flag to
`EncoderSelfAttnParams` (default `true`). Replaced the 32-line inline
attention block in voxtral4b.cpp with a single `encoder_self_attn()`
call using `permute_cont = false`. Bit-identical output verified on
jfk.wav.

---

## 2. Qwen3 forced aligner — DONE

**Status:** Fully implemented and verified. All code already exists:

- `qwen3_asr.cpp:372-382` — lm_head shape read from tensor, not asserted
- `qwen3_asr.h:160-190` — `qwen3_asr_lm_head_dim()`, `qwen3_asr_run_aligner()`,
  `qwen3_asr_align_words()` APIs
- `crispasr_aligner.cpp:61-129` — dispatch via filename detection
  ("forced-aligner", "qwen3-fa", "qwen3-forced")
- GGUF converter handles the aligner model out of the box
- HF release: `cstr/qwen3-forced-aligner-0.6b-GGUF`

Verified working:
```bash
crispasr --backend voxtral -m voxtral.gguf -f samples/jfk.wav \
    -am qwen3-forced-aligner-0.6b.gguf -osrt -ml 1
```
Produces per-word SRT timestamps (80ms resolution, 5000 classes).

---

## 3. Granite µP scale

**Goal:** Document and optionally extract granite's µP (maximal update
parameterization) attention and residual scaling into named parameters.

**Files:**
- `src/granite_speech.cpp` — lines using `hp.attention_multiplier`
  and `hp.residual_multiplier`
- `src/core/attention.h` — `KvSelfAttnParams::attn_scale` already
  handles the attention multiplier

**Current state:** Granite already uses `attn_scale = hp.attention_multiplier`
(0.0078125 = 1/128) instead of the standard `1/sqrt(head_dim)`, and this
is passed through `KvSelfAttnParams::attn_scale`. The `residual_multiplier`
(0.22) is applied outside the helper, inline in granite_speech.cpp:
```cpp
cur = ggml_add(ctx0, residual, ggml_scale(ctx0, attn, hp.residual_multiplier));
```

**Assessment:** This is already clean. The `attn_scale` knob covers the
attention side, and the residual multiplier is a one-line inline scale
that doesn't benefit from extraction. **No code change needed** — just
update TODO.md to mark it as "handled via existing knobs."

---

## 4. Scheduler reuse audit

**Goal:** Verify that the TODO item about recreating `ggml_backend_sched`
per call is resolved.

**Current state:** All 11 backends create `ggml_backend_sched_new()` once
at init and use `ggml_backend_sched_reset()` between compute calls:
- qwen3: init at line 1390, reset at lines 1462/1547/1624/1656
- voxtral: init at line 787, reset at lines 996/1028/1068/1311
- voxtral4b: init at line 889, reset pattern matches
- granite: init at line 700, reset pattern matches
- parakeet/canary/cohere/canary_ctc: same pattern

**Action:** Mark the TODO item as done. No code changes needed.

---

## 5. Reference backends for parakeet/canary/cohere

**Goal:** Write `tools/reference_backends/{parakeet,canary,cohere}.py`
so `crispasr-diff` can generate reference activations for these backends.

**Files to create:**
- `tools/reference_backends/parakeet.py`
- `tools/reference_backends/canary.py`
- `tools/reference_backends/cohere.py`

**Approach for each:**

### parakeet.py
- Load the `.nemo` tarball using `tarfile` + `torch.load()`, following
  the pattern in `models/convert-parakeet-to-gguf.py`'s `unpack_nemo()`.
- Run the NeMo model's `forward()` with PyTorch hooks to capture:
  `mel`, `encoder_output` (per-layer), `tdt_joint_logits`, `decoded_text`.
- The model uses `nemo_toolkit` OR can be loaded as raw PyTorch state
  dict with manual forward pass (the converter already does this for
  weight extraction). Prefer the manual path to avoid nemo dependency.

### canary.py
- Similar to parakeet — `.nemo` tarball, `unpack_nemo()`, PyTorch
  state dict.
- Capture: `mel`, `encoder_output`, `decoder_cross_attn_kv`,
  `decoder_output`, `decoded_text`.
- Canary has both encoder (FastConformer) and decoder (Transformer)
  stages, so more capture points.

### cohere.py
- Load via `transformers.AutoModel.from_pretrained("CohereLabs/cohere-transcribe-03-2026")`.
- Capture: `mel` (pre-emphasized), `encoder_output`, `decoder_logits`,
  `decoded_text`.
- Simpler than NeMo models since HF transformers has a clean API.

**Template:** Follow `tools/reference_backends/qwen3.py` for the
registration pattern:
```python
BACKEND_NAME = "parakeet"
DEFAULT_STAGES = ["mel", "encoder", "text"]

def load_model(model_dir, device="cpu"):
    ...
def capture_stages(model, audio_path, stages):
    ...
    return {"mel": mel_np, "encoder": enc_np, "text": text}
```

**Verification:** For each backend, run:
```bash
python tools/dump_reference.py --backend parakeet \
    --model-dir /path/to/parakeet-tdt-0.6b-v3 \
    --audio samples/jfk.wav --output /tmp/parakeet-ref.gguf
./build/bin/crispasr-diff parakeet parakeet.gguf /tmp/parakeet-ref.gguf samples/jfk.wav
```

**Risk:** Medium. NeMo checkpoint loading is non-trivial — the `.nemo`
tarball contains nested `model_weights.ckpt` files with non-standard
key names. The converter scripts already handle this, so the code can
be adapted.

**LOC:** ~100–150 lines per backend.

---

## 6. Best-of-N sampling for LLM backends — DONE

**Status:** Implemented for all four LLM backends (voxtral, qwen3,
granite, voxtral4b). Added `run_with_probs()` overload with PreHook
in `greedy_decode.h` for voxtral4b's streaming audio injection.
Verified: temperature=0 stays bit-identical on all backends;
`--best-of 3 -tp 0.3` works on all (qwen3 score=0.9726, voxtral4b
score=0.9785).

---

## 7. voxtral4b native streaming protocol

**Goal:** Expose voxtral4b's native streaming mode — the model is
designed for realtime ASR with configurable 240ms–2.4s latency.

**Current state:** voxtral4b runs in chunk-and-transcribe mode like
other backends. The `pre_hook` in `crispasr_backend_voxtral4b.cpp`
already implements the streaming audio injection mechanism (adds one
audio frame to the LLM embedding per decode step), but this operates
on pre-segmented chunks, not a continuous stream.

**Design:**
1. **New CLI mode:** `crispasr --backend voxtral4b --stream-native -m model.gguf`
   enters a loop that reads PCM from stdin in small chunks (240ms at
   minimum), runs the audio encoder on each chunk, and feeds encoder
   frames to the LLM one at a time during generation.
2. **Latency control:** `--stream-delay 240` (ms) controls how many
   audio frames to buffer before starting generation. The model
   supports 240ms, 480ms, 960ms, and 2400ms modes.
3. **Output:** Each generated token is printed immediately (partial
   transcript), with periodic newlines at sentence boundaries.

**Implementation:**
- Extend `crispasr_backend_voxtral4b.cpp` with a `transcribe_streaming()`
  method that takes a callback for new PCM data.
- The audio encoder runs on accumulated chunks (e.g. 1 second of audio),
  producing N encoder frames that are queued.
- The LLM decode loop pops frames from the queue via the existing
  `pre_hook` mechanism, blocking if no frames are available yet.
- Thread model: main thread reads PCM and runs encoder, decode thread
  runs the LLM. A mutex-protected frame queue connects them.

**Verification:**
- Pipe audio from ffmpeg: `ffmpeg -i audio.wav -f s16le -ar 16000 -ac 1 - | crispasr --backend voxtral4b --stream-native -m model.gguf`
- Measure time-to-first-token from audio start.
- Compare transcript quality vs chunk mode.

**Risk:** High. This is a significant feature that changes the
threading model. The audio encoder → LLM frame injection timing is
critical. The existing `--stream` mode (generic, works for all backends)
is simpler and may be sufficient for most users.

**LOC:** ~200–300 lines.

---

## 8. voxtral audio Q&A

**Goal:** Support audio understanding / Q&A mode for voxtral 3B, beyond
transcription.

**Current state:** The model supports arbitrary prompts over audio
content (summarization, Q&A, analysis). Currently only the transcription
prompt template is implemented.

**Approach:**
1. Add `--prompt-mode chat` flag (or `--ask "What language is spoken?"`)
   that switches from the transcription template to a chat template.
2. The Tekken chat template for voxtral:
   ```
   <s>[INST][BEGIN_AUDIO]<audio_pad>×N[/INST]<user_question>[/INST]
   ```
3. Output: print the LLM's free-form text response (not structured
   as crispasr_segment with timestamps — this is conversational output).

**Implementation:**
- In `crispasr_backend_voxtral.cpp`, add an `ask()` method alongside
  `transcribe()`.
- The `VoxtralOps::build_suffix()` already takes `whisper_params` and
  can read a new `params.ask_prompt` field.
- In `cli.cpp`, wire `--ask` to set the prompt and call the backend.

**Risk:** Low — the transcription pipeline already works; this just
changes the prompt template.

**LOC:** ~50 lines.

---

## 9. Parakeet TDT decoder ggml graph port

**Goal:** Port parakeet's TDT decoder (LSTM predictor + joint head)
from manual CPU float* loops to ggml graphs for GPU acceleration.

**Files:**
- `src/parakeet.cpp` — the TDT decoder section (~300 lines of manual
  LSTM stepping + joint network evaluation)

**Current state:** The encoder runs as a ggml graph (via core/mel +
FastConformer), but the TDT decoder is hand-written C++ with manual
LSTM cell computation, joint head matrix multiplies, and token-by-token
stepping.

**Challenge:** The LSTM is inherently sequential — each time step
depends on the previous hidden state. On GPU this means per-step
kernel launches with tiny workloads. The encoder is already 85%+ of
total time (FastConformer with O(T²) attention), so GPU-accelerating
the decoder saves at most 15%.

**Approach:**
1. Build a ggml graph for one LSTM step: `x → gate(W_ih, W_hh) → cell update → hidden`.
2. Run in a loop with `ggml_backend_sched_reset()` between steps.
3. The joint head (a single matmul + tanh + linear) goes in the same
   graph as the LSTM step.
4. Use the scheduler's GPU backend if available, CPU otherwise.

**Verification:**
- Bit-identical transcript on samples/jfk.wav before and after.
- Benchmark: time the decoder phase alone (encoder excluded).

**Risk:** Medium. LSTM in ggml is unusual — most ggml models are
transformer-based. The per-step graph is very small (a few matmuls),
so the overhead of graph construction and scheduling may exceed the
compute time. Profile before committing.

**LOC:** ~100–150 lines.

---

## 10. Granite encoder ggml graph port — DONE (CPU verified)

**Status:** The existing `granite_build_encoder()` graph was wired up
with a new `granite_run_encoder_graph()` runner function. Enable with
`GRANITE_ENCODER_GRAPH=1` environment variable; falls back to CPU loops
on failure.

**Results on CPU (q5_0, jfk.wav):**
- Output: identical transcript to CPU loop path
- Timing: ~35.4s graph vs ~36.0s CPU loops (marginal improvement on
  CPU — the LLM decode dominates at ~26s)
- GPU expected to drop encoder from ~10s to <1s

**Known limitation:** Shaw relative position embeddings are omitted
in the graph path (uses flash_attn_ext with block mask only). Output
is identical for this test case despite the approximation. For cases
where RPE matters, implement Q@RPE bias via manual ggml_mul_mat +
ggml_soft_max attention instead of flash_attn_ext.

**Follow-up:** Add Shaw RPE to the graph path for full accuracy parity.

---

## 11. WebSocket streaming server

**Goal:** Add WebSocket support to the server for real-time
transcription over HTTP.

**Current state:** The server uses httplib (HTTP only). Real-time
streaming requires WebSocket for bidirectional audio/text flow.

**Approach:**
1. httplib does not support WebSocket. Two options:
   a. Add a WebSocket library (e.g. `websocketpp`, header-only) as a
      second listener alongside the HTTP server.
   b. Use a simple custom WebSocket handshake on a separate port
      (the protocol is well-documented and the handshake is ~50 lines).
2. Client sends raw PCM audio chunks over the WebSocket.
3. Server processes each chunk through the backend's transcribe() and
   sends back JSON results incrementally.
4. Keep the existing HTTP endpoints unchanged.

**Wire protocol (matching common ASR WebSocket APIs):**
```
Client → Server: binary PCM frames (16 kHz, 16-bit, mono)
Server → Client: {"text": "partial...", "is_final": false}
Server → Client: {"text": "Final result.", "is_final": true}
Client → Server: {"type": "close"}  (or WebSocket close frame)
```

**Risk:** Medium. WebSocket adds a new dependency or custom protocol
code. The httplib library doesn't support it natively.

**LOC:** ~200–300 lines.

---

## 12. Pipeline template consolidation

**Goal:** Evaluate whether qwen3, granite, and voxtral4b backend
adapters should adopt the `crispasr_llm_pipeline.h` template (currently
only used by voxtral 3B).

**Current state:**
- `crispasr_llm_pipeline.h` implements: mel → encoder → prompt build →
  embed → splice → KV init → best-of-N decode → detokenize.
- voxtral uses it via `VoxtralOps` traits struct (~100 lines).
- qwen3/granite/voxtral4b implement the same pipeline inline
  (~100–150 lines each) with minor differences:
  - **qwen3:** GPT-2 byte-encoded token text needs `decode_gpt2_bytes()`.
  - **granite:** Different prompt template, BPE tokenizer.
  - **voxtral4b:** Streaming pre_hook audio injection.

**Assessment:** The template would need these additions to cover all:
1. A `decode_token(ctx, id) → string` trait method (instead of raw
   `token_text → bytes`), to handle qwen3's GPT-2 encoding.
2. voxtral4b's streaming pre_hook is already supported by
   `core_greedy_decode::run()` — the pipeline template just needs to
   accept an optional pre_hook in its Ops traits.

**Recommendation:** Do this only if we add a 5th LLM backend.
Currently 3/4 backends are inline and work fine. The ROI of
templatizing is small.

---

## 13. canary_ctc aligner CPU fallback — DONE

**Status:** Already implemented. Both scheduler init points
(`canary_ctc_compute_logits_from_mel_debug` at line 578 and
`canary_ctc_compute_logits` at line 626) already use the 2-backend
pattern (GPU primary + CPU fallback). No code change needed.

---

## 14. Misc cleanup items

### a. Test target rename
`tests/CMakeLists.txt` uses `whisper-cli` as the test target name. Once
the rename to `crispasr` has propagated fully, change test references to
`$<TARGET_FILE:crispasr>`. Low priority, cosmetic.

### b. Delete empty legacy dirs
`examples/{parakeet,canary,cohere,qwen3-asr,voxtral,voxtral4b,granite}-main/`
may have stale build artifacts. They're untracked (not in git), so this
is just `rm -rf` on local filesystems. Not a code change.

### c. Granite dead code — DONE
`granite_build_encoder` resurrected for the ggml graph encoder path
(`GRANITE_ENCODER_GRAPH=1`).

### d. Remove dead TODO markdown files
The consolidation from 15 per-model markdown files into TODO.md,
LEARNINGS.md, HISTORY.md was tracked but the deletion of the old files
may not have been committed. Check: `ls *-todo.md benchmark_*.md ggml_plans.md`
and remove any that remain.

---

## 15. CMake target rename (whisper-cli → crispasr)

**Goal:** Rename the CMake target from `whisper-cli` to `crispasr` for
consistency with the binary name.

**Scope:** ~50 references across:
- `examples/cli/CMakeLists.txt` — target definition + backward-compat aliases
- `CMakeLists.txt` — MSVC warning suppression
- `.github/workflows/{ci,release,lint}.yml` — 8 `--target whisper-cli` refs
- `tests/CMakeLists.txt` — 15 `$<TARGET_FILE:whisper-cli>` refs
- Shell scripts (8+) — `./build/bin/whisper-cli` paths
- Documentation — README.md, ARCHITECTURE.md, etc.

**Approach:** Single mechanical commit renaming all references. Keep
the backward-compat symlink (whisper-cli → crispasr) in the install
tree so existing scripts don't break.

**Risk:** Low individually but touches many files. Best done as a
standalone commit/PR to keep the diff reviewable.

---

## 16. Shaw RPE for granite encoder graph

**Goal:** Add query-dependent Shaw relative position embeddings to the
granite ggml graph encoder (currently uses flash_attn_ext with block
mask only, omitting RPE).

**Approach:** Replace `ggml_flash_attn_ext` with manual attention:
1. QK^T via `ggml_mul_mat` → (C, C) per head
2. RPE bias via 3D `ggml_mul_mat`: Q(hd, 1, C) × RPE(hd, C, C) → (C, 1, C)
3. Add content + position scores, scale, softmax, V matmul
4. Precompute RPE lookup (200, 200, 128) per layer at init time

**Risk:** Medium. The 3D batched mul_mat for RPE bias needs careful
tensor shape handling. Profile to verify the manual attention path
isn't slower than flash_attn on CPU.

---

## 17. VAD stitching for long audio (whisper.cpp parity) — DONE

**Shipped:** v0.4.3 (CLI-internal stitching) → v0.4.4 (exposed via
C-ABI as `crispasr_session_transcribe_vad`, so Dart/Python/Rust all
get it through one call).

**Goal:** Match whisper.cpp's VAD approach for non-whisper backends:
stitch VAD segments into a contiguous buffer, process as one audio,
remap timestamps back to original positions.

**Current state:** Done end-to-end. The shared library
(`src/crispasr_vad.{h,cpp}`) owns slicing + merge/split + stitch +
remap. The CLI calls it. The C-ABI function
`crispasr_session_transcribe_vad(session, pcm, n, sr, vad_path, opts)`
runs the whole pipeline in one FFI hop and every wrapper binds it.
CrisperWeaver v0.1.7 uses this path for all 9 non-whisper backends.

**Pre-DRY state (for archival context):**
- VAD worked (centisecond bug fixed, segment merging added)
- Each VAD segment was transcribed independently as a separate slice
- This lost cross-segment context at boundaries
- whisper.cpp stitches segments + builds a `vad_mapping_table` for
  timestamp remapping — this gave better results

**How whisper.cpp does it (src/whisper.cpp lines 6895-6980):**
1. Concatenate all VAD segments into one contiguous float buffer
2. Insert 0.1s silence between segments
3. Add 0.1s overlap at segment boundaries
4. Build `vad_mapping_table` (stitched_position → original_position)
5. Process the stitched buffer through whisper's normal pipeline
6. Remap output timestamps via `vad_time_map_get_original()`

**How other projects handle long audio:**
- **ChunkFormer**: Fixed chunks with 128-frame left+right context windows.
  Model architecture supports this natively (trained with chunking).
- **Eve (nexmoe)**: Silero VAD → Qwen3-ASR via sherpa-onnx. Same as us.
- **Conformer-Athena**: Dynamic chunk-based attention (arxiv 2012.05481).
  Model-level solution, not applicable to pre-trained models.

**Our approach:** Infrastructure for stitching is in place
(`crispasr_stitched_audio`, `crispasr_vad_remap_timestamp` in
crispasr_vad.{h,cpp}). What remains is wiring it into the dispatch
loop in `crispasr_run.cpp`:
1. When VAD is active, stitch segments into one buffer
2. Send the stitched buffer as a single `transcribe()` call
3. Remap `seg.t0`/`seg.t1` and word timestamps afterward
4. If the stitched buffer exceeds `chunk_seconds`, split at the
   best VAD boundary within that range (already implemented)

**Risk:** Medium. The stitching itself is simple. The tricky part is
remapping word-level timestamps correctly (linear interpolation
across silence gaps).

**Immediate value:** Current VAD segment merging (min 3s, gap < 1s)
already handles the common case well. The stitching improvement
matters most for audio with many short pauses (lectures, interviews).

---

## 21. CLI→library DRY refactor — DONE (v0.4.4–v0.4.8)

**Goal:** End the duplication between `examples/cli/crispasr_*.{h,cpp}`
and the C-ABI. Each release in this cycle promoted one concern from
CLI-only into `src/` behind the shared C-ABI, leaving the CLI with
nothing but arg parsing, output formatting, and thin UX shims.

**Shipped:**

| Release | Moved to `src/` | New C-ABI |
|---|---|---|
| v0.4.4 | VAD slicing + stitching + remap (`crispasr_vad.{h,cpp}`) + file rename `crispasr_dart_helpers.cpp` → `crispasr_c_api.cpp` | `crispasr_session_transcribe_vad` |
| v0.4.5 | 4 in-process diarizers: energy, xcorr, vad-turns, native pyannote (`crispasr_diarize.{h,cpp}`) | `crispasr_diarize_segments_abi` |
| v0.4.6 | whisper-tiny + silero-native LID with context cache (`crispasr_lid.{h,cpp}`) | `crispasr_detect_language_pcm` |
| v0.4.7 | canary-CTC + qwen3-forced-aligner (`crispasr_aligner.{h,cpp}`) | `crispasr_align_words_abi` + 5 result accessors |
| v0.4.8 | HF download + cache (`crispasr_cache.{h,cpp}`) + model registry (`crispasr_model_registry.{h,cpp}`) | `crispasr_cache_ensure_file_abi`, `crispasr_cache_dir_abi`, `crispasr_registry_lookup_abi`, `_by_filename_abi` |

**What stayed in the CLI:**
- `cli.cpp` — arg parsing + dispatch
- `crispasr_backend.{h,cpp}` + `crispasr_backend_*.cpp` — the
  whisper-cli-shaped `CrispasrBackend` wrapper over each model's
  native C API (the session API in `src/` is the modern path; this
  one is kept for the historical `whisper-cli`-compatible mode)
- `crispasr_output.cpp` — TXT/SRT/VTT/CSV/JSON/LRC writers
- `crispasr_server.cpp` — HTTP server for persistent-model mode
- `crispasr_diff*.cpp` — diff tool for regression runs
- `*_cli.{h,cpp}` shims: `crispasr_vad_cli`, `crispasr_lid_cli`,
  `crispasr_diarize_cli`, `crispasr_model_mgr_cli`, `crispasr_aligner_cli`
  — all purely UX policy (auto-download from `~/.cache/crispasr`,
  sherpa-ONNX subprocess fallback, TTY download prompt, format
  adapter layer)

**Wrapper alignment:** all three bindings bumped to match.
- `package:crispasr` 0.4.8 (Dart)
- `crispasr` 0.4.8 (Python)
- `crispasr` + `crispasr-sys` 0.1.6 (Rust)

Each wrapper gained top-level helpers for every promoted function,
plus a `*_abi` ABI-POD struct for the parameters where that makes
sense (VAD opts, diarize opts). The `crispasr_dart_helpers_version`
C-ABI symbol is kept as a back-compat alias pointing at
`crispasr_c_api_version`, so 0.4.x-era external consumers keep
resolving.

**Lessons:**
- Header basename clashes between `src/crispasr_vad.h` and
  `examples/cli/crispasr_vad.h` break the build because the whisper
  target marks `src/` `PUBLIC` on its include path. Fix: rename the
  CLI header to `crispasr_vad_cli.h` (and same for diarize/lid/
  model_mgr/aligner).
- Function-name collisions between CLI and library (e.g. both had a
  `crispasr_detect_language`) are OK in C++ if argument types
  differ (different mangled names) but fragile. Safer pattern: CLI
  shim exposes `crispasr_*_cli(...)` variants.
- POD ABI structs need explicit `_pad` fields for predictable layout
  on 32-bit vs 64-bit. We use `int32_t` explicitly and check sizes
  in the Dart/Python/Rust bindings.

---

## 18. Qwen3 forced aligner accuracy improvements

**Current state:** Basic aligner works but has known quality issues:
1. Leading silence → timestamps start too early (aligner assigns
   timestamps to silence). Workaround: use `--vad`.
2. Missing `fix_timestamp()` LIS post-processing from the reference
   implementation. We added a simpler forward clamp (monotonicity
   enforcement) which handles most cases.
3. 80ms resolution (5000 classes) is inherently coarser than
   parakeet's native TDT timestamps.

**Reference implementation:** See
`qwen_asr/inference/qwen3_forced_aligner.py` in QwenLM/Qwen3-ASR.
Key differences from our implementation:
- Full LIS (longest increasing subsequence) for timestamp correction
- Language-specific word tokenization (CJK character-level, nagisa
  for Japanese, soynlp for Korean)
- We use simple whitespace splitting for all languages

**Recommendation:** Parakeet is the better choice for timestamp-
critical use cases. The forced aligner is best as a fallback for
backends that lack native timestamps (voxtral, granite, cohere).

---

## 24. Wrapper integration test suites — DONE

Python (13 tests), Rust (5+3 ignored), Dart (9 tests) — all passing.

| Wrapper | Tests | Pass | Notes |
|---|---|---|---|
| Python | 13 | 13 | Session API: whisper + parakeet, transcription, timestamps, edge cases |
| Rust | 8 | 5 (+3 ignored) | Session API works; old CrispASR API crashes (C++ exceptions through FFI) |
| Dart | 9 | 9 | FFI smoke (7) + transcription (2) |

**Known issue:** The old Rust `CrispASR` struct (wrapping `whisper_full()`
directly) crashes because C++ exceptions propagate through Rust's FFI
boundary. The `Session` API is safe (C-ABI wrapper catches exceptions).
`CrispASR` is deprecated in favor of `Session`.

---

## Priority ordering

| Priority | Item | Impact | Effort |
|---|---|---|---|
| **Done** | #2 Qwen3 forced aligner | Already implemented and verified | 0 LOC |
| **Done** | #10 Granite encoder graph | Wired and tested on CPU; enable with GRANITE_ENCODER_GRAPH=1 | ~60 LOC new |
| **Done** | #1 voxtral4b encoder migration | Migrated to encoder_self_attn() | 0 LOC |
| **Done** | #6 Best-of-N for all LLM backends | All 4 backends support --best-of N | 0 LOC |
| **Done** | #13 canary_ctc CPU fallback | Already implemented | 0 LOC |
| **Medium** | #5 Reference backends | Testing infrastructure completeness | ~400 LOC |
| **Done** | #3 Granite µP | Already handled via existing knobs | 0 LOC |
| **Done** | #4 Scheduler audit | Already done | 0 LOC |
| **Done** | #8 voxtral Q&A | --ask flag for audio understanding | ~10 LOC |
| **Done** | #14a granite dead code | Resurrected for graph encoder | 0 LOC |
| **Low** | #7 voxtral4b streaming | Complex, niche | ~300 LOC |
| **Low** | #9 Parakeet TDT GPU | Small gain, encoder dominates | ~150 LOC |
| **Low** | #11 WebSocket streaming | Needs new dependency | ~300 LOC |
| **Low** | #12 Pipeline template | ROI too small with only 4 backends | 0 LOC |
| **Low** | #14 Cleanup | Cosmetic | ~20 LOC |
| **Done** | #17 VAD stitching | Stitch + remap matching whisper.cpp | ~155 LOC |
| **Medium** | #18 Aligner LIS | Full LIS monotonicity fix + language-specific tokenization | ~80 LOC |
| **Medium** | #15 CMake target rename | Rename whisper-cli → crispasr across CI/tests/scripts (~50 refs) | ~50 files |
| **High** | #19 wav2vec2-base support | Fix forward pass for base models (768 hidden, group norm) | ~50 LOC |
| **Medium** | #20 OGG support on Windows | stb_vorbis not linked or read_audio_data path issue (#10) | ~20 LOC |
| **Low** | #16 Shaw RPE for granite graph | Add query-dependent position bias to encoder graph | ~80 LOC |
| **Done** | #21 CLI→library DRY refactor | VAD, diarize, LID, aligner, cache, registry promoted to src/ behind shared C-ABI | ~2000 LOC moved |
| **Pending** | #22 Stream + audio decoder in wrappers | Expose `crispasr_audio_decode_*` + streaming session through Dart/Python/Rust bindings | ~150 LOC each |
| **Pending** | #23 Diarization + LID + align in CrisperWeaver | Swap the MFCC/k-means stopgap for the lib path; wire LID for auto-language; add forced-aligner for LLM backends | ~250 LOC |

## 25. Montreal Forced Aligner evaluation — NOT PLANNED

MFA uses Kaldi + OpenFST + Pynini (heavy C++ dependencies, ~500MB).
The core Viterbi alignment algorithm is simple (~200 LOC), but the
acoustic models (TDNN-F) and pronunciation dictionaries (OpenFST
graph composition) require the full Kaldi stack.

Our existing aligners (canary-CTC, qwen3-forced-aligner) cover
word-level alignment. MFA's advantage is phoneme-level precision,
which is a niche use case. Users can run MFA externally via
`pip install montreal-forced-aligner`.

**Decision:** Not worth the dependency cost. Keep as external tool.

## 26. GLM-ASR-Nano backend — DONE

**Model:** zai-org/GLM-ASR-Nano-2512 (1.5B params, MIT license)
**Architecture:** Whisper encoder (1280d, 32L, partial RoPE factor=0.5)
+ 4-frame-stack projector (5120→4096→2048) + Llama LLM (2048d, 28L,
GQA 16/4, SwiGLU). 17 languages, optimized for Mandarin + English.

**Shipped:**
- GGUF converter (747 tensors, 4.52 GB F16)
- C++ runtime with partial RoPE, GPT-2 BPE decoding
- Backend adapter registered as 'glm-asr', auto-detection for 'glmasr'
- Correct transcription on jfk.wav verified

**Key learnings:**
- Partial RoPE (factor≠1.0) must split Q/K, apply RoPE to first half only
- FFT for n_fft=400 needs zero-padding to 512 (radix-2 requirement)
- KV cache context must be no_alloc=true for ggml_backend_sched

## 27. Kyutai STT evaluation — DEFERRED

**Model:** kyutai/stt-2.6b-en (2.6B params, CC-BY-4.0, English only)
**Architecture:** Decoder-only transformer with Mimi neural audio codec.

Unlike all other CrispASR backends which use mel spectrograms, Kyutai
STT uses a **neural audio codec** (Mimi) to tokenize audio into 32
parallel codebook streams at 12.5 Hz. The STT decoder is a 48-layer
transformer that processes these 32+1 (audio+text) token streams
simultaneously with sliding window attention (375).

**Complexity assessment:**
1. **Mimi codec** — CNN encoder + 8-layer transformer + 32-codebook
   residual VQ. Novel architecture, ~200MB. No mel at all.
2. **Multistream decoder** — 32 audio codebook streams + 1 text stream
   interleaved. 48 layers, 2048 hidden, 32 heads.
3. **24 kHz audio** — our pipeline assumes 16 kHz everywhere.
4. **Two-model inference** — Mimi runs first, then STT decoder.

**Estimated effort:** 2-3 weeks (Mimi codec port is the hard part).

**Key discovery:** [`Codes4Fun/moshi.cpp`](https://github.com/Codes4Fun/moshi.cpp)
already implements Mimi codec + STT + TTS in ggml (MIT license, 83
commits, 28 stars). Works at 93 fps STT on RTX 2070. Supports GGUF,
CUDA, Vulkan, CPU. Integration paths:
1. Adapt their Mimi+STT code into our backend interface (~1 week)
2. Use as subprocess backend like whisper.cpp (~1 day)
3. Use as reference to build our own implementation (~2 weeks)

**Decision:** Deferred but significantly de-risked by moshi.cpp's
existence. When we do this, option 1 (adapt their code) is fastest.
