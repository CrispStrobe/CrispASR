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

## 1. voxtral4b audio encoder migration

**Goal:** Migrate voxtral4b's inline audio encoder attention (32 layers)
to `core_attn::encoder_self_attn()`.

**Files:**
- `src/voxtral4b.cpp` lines 585–626 (the encoder loop)
- `src/core/attention.h` (the `encoder_self_attn()` function)

**Problem:** voxtral4b's encoder uses `ggml_permute()` WITHOUT
`ggml_cont()` before `ggml_flash_attn_ext()` (line 613). The existing
`encoder_self_attn()` uses `ggml_cont()` after permute (matching
voxtral 3B). Adding `ggml_cont` produces the same values but changes
the ggml graph structure, which could cause subtle differences in
buffer allocation or scheduling.

**Approach:**
1. Add a `bool permute_cont` flag to `EncoderSelfAttnParams` (default
   `true`). When `false`, skip the `ggml_cont()` after permute — this
   matches voxtral4b's current behavior.
2. Replace the inline attention block at voxtral4b.cpp:594–625 with a
   single `encoder_self_attn()` call, passing `permute_cont = false`.
3. The SWA mask is already a parameter (`mask`), and RoPE params are
   already in `EncoderSelfAttnParams`.

**Verification:**
- Requires a voxtral4b GGUF model file (`voxtral-mini-4b-realtime-*.gguf`)
- Run: `crispasr --backend voxtral4b -m model.gguf -f samples/jfk.wav -np > before.txt`
- Make the change, rebuild
- Run again: `diff before.txt after.txt && echo BIT-IDENTICAL`
- If not bit-identical: check if `ggml_cont` vs no-cont causes the
  divergence. If it does, the flag approach is correct. If not, there's
  another difference to investigate.

**Risk:** Low. The helper already handles everything; this is just a
contiguity flag.

**LOC:** ~5 lines changed in voxtral4b.cpp, ~3 lines added to attention.h.

---

## 2. Qwen3 forced aligner

**Goal:** Add Qwen3-ForcedAligner-0.6B as a second timestamp provider
alongside canary-ctc-aligner, giving all backends access to
Qwen3-quality word timestamps via `-am qwen3-forced-aligner.gguf`.

**Files:**
- `src/qwen3_asr.cpp` — needs to handle the aligner's
  5000-class lm_head (currently assumes lm_head matches vocab_size)
- `src/qwen3_asr.h` — add `qwen3_asr_run_aligner()` entry point
- `examples/cli/crispasr_aligner.{h,cpp}` — add dispatch branch for
  qwen3 aligner alongside the existing canary-ctc branch
- `models/convert-qwen3-asr-to-gguf.py` — already handles the aligner
  model (verified)

**Approach:**

### Step 1: Fix lm_head shape assumption in loader
In `qwen3_asr_load_model()`, the lm_head tensor is loaded and its
shape is asserted to match `hp.llm_vocab_size`. The forced aligner
has `output.weight (5000, 1024)` instead of `(151936, 1024)`. Fix:
read the actual ne[0] from the GGUF tensor and store it as
`hp.llm_lm_head_dim`. The `run_llm_kv()` function already has a
fallback: `vocab = hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size`
(line 1664).

### Step 2: Add `qwen3_asr_run_aligner()` C API
```c
// One forward pass (no autoregressive decode). Returns per-position
// argmax for every input token where id == 151705 (timestamp placeholder).
// The caller provides the full input sequence (text + timestamp placeholders)
// and gets back an array of frame indices (argmax * 80ms = timestamp).
int qwen3_asr_run_aligner(
    qwen3_asr_context* ctx,
    const float* audio_samples, int n_samples,
    const int32_t* input_ids, int n_ids,
    int64_t* out_timestamps_ms, int* n_timestamps);
```

Implementation: compute mel → run encoder → embed tokens → splice
audio into audio_pad positions → run one forward pass → for each
position where `input_ids[i] == 151705`, take `argmax(logits[i, :5000])`
and multiply by 80ms.

### Step 3: Wire into crispasr_aligner.cpp
The existing `crispasr_ctc_align()` dispatches based on the model path.
Add a branch: if the model filename contains "qwen3" or the GGUF
architecture is "qwen3-asr" with `llm_lm_head_dim == 5000`, use the
Qwen3 aligner path instead of canary-ctc.

### Step 4: Template construction
The Qwen3 forced aligner expects a specific chat template with
`<|timestamp|>` tokens inserted between words. The HF reference code
is at `qwen_asr/inference/qwen3_forced_aligner.py`. The template is:
```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>
<|im_start|>user\nAudio 1: <|audio_bos|><audio_pad>×N<|audio_eos|>
<|timestamp|>word1<|timestamp|>word2...<|timestamp|><|im_end|>
<|im_start|>assistant\n
```

The output at each `<|timestamp|>` position is a 5000-class softmax
where `argmax * 80ms` = the timestamp for that word boundary.

**Verification:**
- Convert the aligner: `python models/convert-qwen3-asr-to-gguf.py --model-dir Qwen3-ForcedAligner-0.6B --output qwen3-fa.gguf`
- Run: `crispasr --backend voxtral -m auto -f samples/jfk.wav -am qwen3-fa.gguf -osrt -ml 1`
- Compare word timestamps against canary-ctc aligner output.
- Cross-reference with HF reference output.

**Risk:** Medium. The template construction is tricky — wrong token
IDs or missing special tokens will produce garbage timestamps. Use
`tools/dump_reference.py --backend qwen3 --stages aligner` to get
ground truth.

**LOC:** ~150 lines total across 3 files.

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

## 6. Best-of-N sampling for LLM backends

**Goal:** When `temperature > 0` and `--best-of N` is set, run N
independent temperature-sampled decodes and pick the highest-scoring
one.

**Current state:** Already implemented for voxtral in
`crispasr_llm_pipeline.h` lines 158–221. The pipeline runs N decode
loops with different RNG seeds and picks the result with the highest
mean per-token probability.

**What's missing:** qwen3, voxtral4b, and granite have their own
inline pipelines in `crispasr_backend_{qwen3,voxtral4b,granite}.cpp`
that don't implement best-of-N. They do single-run decoding.

**Approach:**
1. In each of the three backend adapters, wrap the existing
   prefill + decode section in a `for (int run = 0; run < n_runs; run++)`
   loop, mirroring `crispasr_llm_pipeline.h` lines 170–221.
2. Each run: reset KV cache, re-prefill, decode with a different seed
   (`seed ^ (run * 0x9E3779B97F4A7C15ull)`), score by mean probability.
3. Keep best result.

**Alternative:** Migrate qwen3/granite/voxtral4b to use the
`crispasr_llm_pipeline.h` template (like voxtral does). This would
require making their prompt construction, tokenization, and
detokenization fit the Ops traits pattern. Qwen3 has GPT-2 byte-
encoded token text that needs special handling; granite has a BPE
tokenizer with different special tokens. The template may need
generalization.

**Verification:** Compare with temperature=0 (should be identical to
single-run). With temperature>0 and best_of=5, quality should be
equal or better than single-run.

**Risk:** Low — the pattern is proven in voxtral's pipeline.

**LOC:** ~30 lines per backend adapter (3 backends = ~90 lines).

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

## 10. Granite encoder ggml graph port

**Goal:** Port granite's per-layer CPU encoder from manual float*
loops to a ggml graph, enabling GPU acceleration.

**Files:**
- `src/granite_speech.cpp` — the encoder section (Conformer layers
  with Conv + self-attention + FFN)
- There's a dead `granite_build_encoder` function that was a previous
  attempt. May be useful as a starting point.

**Current state:** The encoder runs as manual float* CPU loops. This
is the dominant cost (~22.5s for 11s audio on Q4_K). With GPU, this
could drop to <1s.

**Approach:**
1. Resurrect or rewrite `granite_build_encoder` to produce a valid
   ggml graph with the Conformer block structure.
2. Use the existing ggml backends (CUDA/Metal/Vulkan) via
   `ggml_backend_sched`.
3. The Q-Former projector is a separate stage — port it as a second
   ggml graph or fold it into the encoder graph.

**Verification:**
- Transcript regression on samples/jfk.wav (allow small float drift
  from GPU accumulation order differences; transcript must match).
- Benchmark: encoder time should drop from ~22s to <2s on GPU.

**Risk:** Medium-high. The Conformer architecture has depthwise
separable convolutions that need `ggml_conv_1d` / `ggml_conv_2d`
with specific stride/padding — these ops are supported on GPU but
less tested than matmul.

**LOC:** ~200–300 lines.

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

## 13. canary_ctc aligner CPU fallback

**Goal:** Fix the aligner's ggml scheduler to include a CPU fallback
backend when the primary backend rejects an op.

**Files:**
- `src/canary_ctc.cpp` lines 585–633 (scheduler init)

**Current state:** The aligner creates a single-backend scheduler. If
the GPU backend rejects an op (e.g. a custom convolution), the compute
fails rather than falling back to CPU.

**Fix:** Mirror the 2-backend pattern from `canary.cpp` / `cohere.cpp`:
```cpp
ggml_backend_t backends[2];
int n = 0;
backends[n++] = ctx->backend; // GPU (or CPU if no GPU)
if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
    backends[n++] = ctx->backend_cpu;
ctx->sched = ggml_backend_sched_new(backends, nullptr, n, 16384, false, false);
```

**Verification:** Run the aligner on a CUDA build. Currently crashes;
after fix, should succeed with some ops on CPU and the rest on GPU.

**Risk:** Very low. ~20 lines, well-understood pattern.

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
| **High** | #2 Qwen3 forced aligner | Unlocks word timestamps for all backends via a second aligner option | ~150 LOC |
| **High** | #10 Granite encoder graph | 20x speedup on GPU (22s → <2s) | ~250 LOC |
| **Medium** | #1 voxtral4b encoder migration | Code cleanliness, ~30 LOC of boilerplate removed | ~10 LOC |
| **Medium** | #6 Best-of-N for all LLM backends | Quality improvement with temperature sampling | ~90 LOC |
| **Medium** | #13 canary_ctc CPU fallback | GPU compatibility fix | ~20 LOC |
| **Medium** | #5 Reference backends | Testing infrastructure completeness | ~400 LOC |
| **Low** | #3 Granite µP | Already handled via existing knobs | 0 LOC |
| **Low** | #4 Scheduler audit | Already done, just docs update | 0 LOC |
| **Low** | #8 voxtral Q&A | New feature, niche use case | ~50 LOC |
| **Low** | #7 voxtral4b streaming | Complex, niche | ~300 LOC |
| **Low** | #9 Parakeet TDT GPU | Small gain, encoder dominates | ~150 LOC |
| **Low** | #11 WebSocket streaming | Needs new dependency | ~300 LOC |
| **Low** | #12 Pipeline template | ROI too small with only 4 backends | 0 LOC |
| **Low** | #14 Cleanup | Cosmetic | ~20 LOC |
