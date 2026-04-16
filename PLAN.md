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

## 6. Best-of-N sampling for LLM backends — DONE (qwen3 + granite)

**Status:** Implemented for qwen3 and granite backend adapters,
matching the voxtral pipeline pattern (N prefill+decode runs, seed
varied per run, best selected by mean token probability). Verified:
temperature=0 stays bit-identical; `--best-of 3 -tp 0.3` works on
qwen3 (score=0.9726).

**voxtral4b deferred:** Its streaming pre_hook (audio frame injection
per decode step) is incompatible with `run_with_probs()` which doesn't
accept a pre_hook. Would need a `run_with_probs_hooked()` variant in
`greedy_decode.h`, or refactoring the hook into the embed_fn.

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

### c. Granite dead code
`granite_build_encoder` in `granite_speech.cpp` is a dead function from
a previous attempt at ggml graph encoding. Remove it (or revive it for
item #10 above).

### d. Remove dead TODO markdown files
The consolidation from 15 per-model markdown files into TODO.md,
LEARNINGS.md, HISTORY.md was tracked but the deletion of the old files
may not have been committed. Check: `ls *-todo.md benchmark_*.md ggml_plans.md`
and remove any that remain.

---

## Priority ordering

| Priority | Item | Impact | Effort |
|---|---|---|---|
| **Done** | #2 Qwen3 forced aligner | Already implemented and verified | 0 LOC |
| **Done** | #10 Granite encoder graph | Wired and tested on CPU; enable with GRANITE_ENCODER_GRAPH=1 | ~60 LOC new |
| **Done** | #1 voxtral4b encoder migration | Migrated to encoder_self_attn() | 0 LOC |
| **Done** | #6 Best-of-N for qwen3+granite | Implemented; voxtral4b deferred (needs pre_hook probs) | 0 LOC |
| **Done** | #13 canary_ctc CPU fallback | Already implemented | 0 LOC |
| **Medium** | #5 Reference backends | Testing infrastructure completeness | ~400 LOC |
| **Done** | #3 Granite µP | Already handled via existing knobs | 0 LOC |
| **Done** | #4 Scheduler audit | Already done | 0 LOC |
| **Low** | #8 voxtral Q&A | New feature, niche use case | ~50 LOC |
| **Low** | #7 voxtral4b streaming | Complex, niche | ~300 LOC |
| **Low** | #9 Parakeet TDT GPU | Small gain, encoder dominates | ~150 LOC |
| **Low** | #11 WebSocket streaming | Needs new dependency | ~300 LOC |
| **Low** | #12 Pipeline template | ROI too small with only 4 backends | 0 LOC |
| **Low** | #14 Cleanup | Cosmetic | ~20 LOC |
