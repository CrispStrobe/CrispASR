# CrispASR — comprehensive TODO

Last updated: 2026-04-09. Covers all 7 (soon 8) runtimes, pending work,
and strategic direction. Written at the end of a long multi-session port
that added Qwen3-ASR and Voxtral to the family.

---

## Immediate (next session)

### 1. Build + commit GPU init changes (already written, not yet compiled)

Files modified but not yet built/committed:
- `src/qwen3_asr.cpp` — switched from `ggml_backend_cpu_init()` to
  `ggml_backend_init_best()` which auto-selects Metal → CUDA → CPU.
  Also added `#ifdef GGML_USE_METAL/CUDA` includes (can be removed
  since `init_best` doesn't need them).
- `src/voxtral.cpp` — same change.
- `src/canary_ctc.cpp` — `cc_pick_backend()` now uses `init_best`.

Need to: `cd /mnt/akademie_storage/whisper.cpp && cmake --build build -j`
then verify the CLI still works, then `git add && commit && push`.

### 2. Retry Voxtral 4B Realtime download

Previous download failed. Run:
```bash
hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir /tmp/voxtral-4b
```

Architecture plan already written at `voxtral-4b-todo.md`. Key new pieces:
RoPE encoder (not absolute pos embed), sliding window attention (encoder
window=750, LLM window=8192), tied embeddings, different dims (26 layers,
FFN=9216, RoPE θ=1e6).

### 3. Upload voxtral HF README

The file `hf_readmes/voxtral-mini-3b-2507-GGUF.md` was uploaded to HF
already but NOT committed to the git repo. Commit it.

Also `voxtral-comparison.md` was written but the user said NOT to commit
it to git — it's a local analysis doc only.

---

## GPU support audit (current state)

| Runtime | GPU status | What's needed |
| --- | --- | --- |
| **parakeet** | ✅ Metal + CUDA (explicit init) | Done |
| **cohere** | ✅ `ggml_backend_init_best()` | Done |
| **canary** | ✅ Metal + CUDA (explicit init) | Done |
| **canary_ctc** (nfa-align) | 🔄 Changed to `init_best` (not yet compiled) | Build + test |
| **wav2vec2** (cohere-align) | ❌ Old ggml pattern, no backend API | Needs refactor (~200 LOC) |
| **qwen3_asr** | 🔄 Changed to `init_best` (not yet compiled) | Build + test |
| **voxtral** | 🔄 Changed to `init_best` (not yet compiled) | Build + test |

After building, all 7 runtimes (except wav2vec2) will auto-detect GPU.
No code changes needed for Metal/CUDA — `ggml_backend_init_best()` handles
runtime dispatch. The CMake `GGML_METAL=ON` / `GGML_CUDA=ON` flags are
already wired in `src/CMakeLists.txt` for all libraries.

---

## Timestamps / timecodes (feature gap)

| Runtime | Current timestamps | Approach |
| --- | --- | --- |
| **parakeet** | ✅ TDT duration head (free, per-token, ~80ms) | Done — best in class |
| **cohere** | ✅ Cross-attention DTW (~360ms MAE) | Done |
| **canary** | ❌ Segment-level only | Use nfa-align as 2nd pass |
| **nfa-align** | ✅ CTC Viterbi forced alignment (~78ms MAE) | Done — can align any model's output |
| **cohere-align** | ✅ char-level CTC (~30ms MAE, English only) | Done |
| **qwen3_asr** | ❌ **None** | Options below |
| **voxtral** | ❌ **None** | Options below |

### Timestamp options for speech-LLMs (qwen3_asr + voxtral)

1. **nfa-align second pass (recommended, ~1 day):** After the LLM generates
   the transcript, run `canary_ctc_align_words()` on the same audio + transcript
   to get word-level timestamps at ~78ms MAE. Already implemented for canary;
   just wire it as an optional `-timestamps` flag in the CLI. Works for any
   language the CTC aligner supports (25 EU languages).

2. **Audio-frame-position timestamps (trivial, ~2 hours):** Each audio_pad
   token in the prompt corresponds to a fixed time span:
   - Qwen3-ASR: 143 frames for 11s → ~77ms per frame
   - Voxtral: 375 frames for 30s → 80ms per frame
   Map each generated text token back to the nearest audio_pad position via
   the self-attention weights. Coarse (~80ms) but free.

3. **Cross-attention extraction (hard, ~3 days):** Extract the attention
   weights between the audio-pad positions and the generated-text positions
   from the LLM's self-attention layers. Apply DTW to align. The "cross-
   attention" isn't a separate module — it's the self-attention between
   audio and text tokens in the same sequence. Requires saving attention
   weights per layer during the forward pass and modifying the graph.

**Recommendation:** Option 1 (nfa-align second pass). It's proven, already
implemented, and gives the best accuracy. The only cost is running the
CTC aligner (~2-3s for an 11s clip), which is small relative to the LLM
decode time.

---

## Performance optimization opportunities

### Already done
- [x] F16 KV cache (qwen3_asr, voxtral)
- [x] Flash attention on prefill + decode (qwen3_asr, voxtral)
- [x] Last-token-only lm_head slice (qwen3_asr, voxtral)
- [x] Q4_K weight quantization with Q4_0 fallback for odd-width tensors
- [x] Baked mel filterbank (no runtime recomputation)

### Next wins
- [ ] **GPU backend** (~5-10× on LLM forward, biggest single win for voxtral 3B)
- [ ] **Speculative decoding** — use a smaller draft model to propose N tokens,
      verify with the big model in one forward pass. Could 2-3× the decode
      throughput for voxtral.
- [ ] **Continuous batching** — pipeline multiple audio files for throughput
      (not latency). Relevant for server/batch workloads.
- [ ] **Chunked long-audio** — for audio >30s, split into overlapping 30s
      chunks, run encoder on each, concatenate. Currently voxtral pads to
      30s; qwen3-asr handles arbitrary length but >30s means >375 audio
      tokens which slows prefill quadratically.
- [ ] **GGML_BLAS=ON** — tested, negligible speedup on CPU for Q4_K (ggml's
      k-quant kernels skip BLAS). Only helps for F16/F32 weights which we
      don't ship. Documented as a negative finding in qwen3-asr-benchmark.md.

---

## Voxtral 4B Realtime port (pending)

Plan at `voxtral-4b-todo.md`. Key differences from the 3B:
- RoPE encoder (not absolute pos embed)
- Sliding window attention (encoder=750, LLM=8192)
- 26 layers, FFN=9216, RoPE θ=1e6
- Tied embeddings
- `VoxtralRealtimeForConditionalGeneration` (different HF class)

Estimated effort: ~3-4 days. The SWA + RoPE encoder is genuinely new.

Weights need to be downloaded:
```bash
hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir /tmp/voxtral-4b
```

---

## Tekken tokenizer (incomplete)

`voxtral_tokenize()` is a stub — the vocab blob is loaded but the
rank-based byte BPE encode logic isn't implemented yet. The CLI
currently hardcodes the transcription prompt token IDs. For audio
understanding (Q&A) mode, a real Tekken encoder would be needed.

The Tekken BPE is structurally similar to tiktoken: start with byte
sequences, greedily merge the lowest-rank pair. The pre-tokenizer regex
uses Unicode property classes (`\p{Lu}` etc.) which needs either:
- A hand-rolled approximation (works for English/German)
- A Unicode regex lib (RE2, onigmo, PCRE2)
- A bundled Unicode character class table

Estimated effort: ~300 LOC, ~1.5 days.

---

## Model-specific pending items

### Qwen3-ASR
- [ ] Timestamps (see above — nfa-align second pass)
- [ ] Streaming support (chunked audio → incremental transcript)
- [ ] Test on more languages (only tested English + German)
- [ ] The Qwen3-ForcedAligner-0.6B companion model (separate from nfa-align)

### Voxtral 3B
- [ ] Timestamps (see above)
- [ ] Audio understanding mode (Q&A about audio content) — needs Tekken tokenizer
- [ ] Function calling from voice — needs Tekken + tool-use prompt format
- [ ] Long audio >30s — needs chunked encoder or padding strategy
- [ ] Test on non-English languages (only English tested)

### Parakeet
- [x] GPU support ✅
- [x] Word timestamps ✅ (TDT)
- [ ] Auto language detection sometimes picks wrong language on accented audio

### Canary
- [x] GPU support ✅
- [ ] Word timestamps via nfa-align integration
- [ ] Speech translation quality validation on more language pairs

### Cohere
- [x] GPU support ✅
- [x] Cross-attention DTW timestamps ✅
- [ ] The upstream ffmpeg-transcode.cpp mp4 bug (tracked in UPSTREAM.md)

---

## HF releases status

| Repo | Status | Files |
| --- | --- | --- |
| `cstr/parakeet-tdt-0.6b-v3-GGUF` | ✅ shipped | F16 + Q8_0 + Q5_0 + Q4_K |
| `cstr/parakeet_de_med-GGUF` | ✅ shipped | F16 + Q8_0 + Q5_0 + Q4_K |
| `cstr/canary-1b-v2-GGUF` | ✅ shipped | F16 + Q8_0 + Q5_0 + Q4_K |
| `cstr/canary-ctc-aligner-GGUF` | ✅ shipped | F16 + Q8_0 + Q5_0 + Q4_K |
| `cstr/cohere-transcribe-03-2026-GGUF` | ✅ shipped | F16 + Q8_0 + Q6_K + Q5_1 + Q5_0 + Q4_K |
| `cstr/qwen3-asr-0.6b-GGUF` | ✅ shipped | F16 + Q8_0 + Q4_K (proper special tokens) |
| `cstr/voxtral-mini-3b-2507-GGUF` | ✅ shipped | Q4_K + Q8_0 + README |
| `cstr/voxtral-mini-4b-realtime-GGUF` | ❌ not started | Pending 4B port |

---

## Code quality / cleanup

- [ ] Remove the `#ifdef GGML_USE_METAL/CUDA` includes from qwen3_asr.cpp
      and voxtral.cpp since `ggml_backend_init_best()` doesn't need them
- [ ] Suppress the remaining -Wunused-variable warnings in voxtral.cpp
      (proj_out, max_pos)
- [ ] Factor out shared mel compute code (currently duplicated between
      qwen3_asr.cpp and voxtral.cpp — identical ~150 LOC)
- [ ] Factor out shared WAV reader (duplicated in every CLI main.cpp)
- [ ] Factor out shared .npy loader (duplicated in every test driver)
- [ ] The voxtral Tekken vocab blob is stored as a 1D F32 tensor (wasteful
      ~5 MB for 1.3 MB of actual bytes) because gguf-py's add_array
      loses uint8 dtype. Could use a custom binary KV type or a better
      gguf-py path.

---

## Strategic notes

### llama.cpp mtmd compatibility

Analysed in `voxtral-comparison.md` (local, not committed). Summary:
- llama.cpp's mtmd support for Voxtral has two unfixed bugs (#17868,
  #18419), worse WER than transformers/vLLM at same precision, and
  Ollama dropped llama.cpp for multimodal citing instability.
- Our standalone ggml approach avoids all these issues.
- Recommendation: keep CrispASR standalone as primary, optionally
  produce llama.cpp-compatible GGUFs for ecosystem users.
- For GPU: use ggml's native Metal/CUDA backends directly (already
  wired via `ggml_backend_init_best()`), not llama.cpp's abstraction.

### The "merkel.wav" lesson

The Wikimedia file `Angela_Merkel_voice.ogg` used in test_german.md
was actually Russian, not German. All three speech-LLMs (Qwen3-ASR,
parakeet, whisper) correctly detected it as Russian. The prior analysis
calling this a "parakeet bug" was wrong. Removed from all docs.

### predict-woo/qwen3-asr.cpp PR

https://github.com/predict-woo/qwen3-asr.cpp/pull/7 — CMake build
fixes for Linux (OpenMP linkage + auto ggml submodule build). Status:
open, awaiting review.

---

## German test audio samples

Saved at `/mnt/storage/german-samples/`:
- `berlin_word.wav` (0.7s) — "Berlin"
- `bundeskanzler_word.wav` (1.6s) — "Bundeskanzler"
- `jazeschann.wav` (4.8s) — "Leider zu spät"
- `De-Abwasch-article.wav` (79.4s) — Wikipedia: Dishwashing
- `De-Afghani-article.wav` (207.6s) — Wikipedia: Afghani currency
- `De-Airbus-A320-Familie_1-article.wav` (2303.3s) — too long for testing

All from Wikimedia Commons, CC-licensed, 16 kHz mono WAV.

---

## Session history (for context recovery)

This project has been built across multiple long sessions:
1. **Cohere Transcribe** — the original port, mel norm bug, DTW timestamps
2. **Parakeet TDT** — FastConformer + TDT decoder, free word timestamps
3. **Canary 1B v2** — speech translation, nfa-align CTC aligner
4. **Qwen3-ASR 0.6B** — first speech-LLM port (Whisper encoder + Qwen3 LLM),
   5 stages across multiple sessions, BPE tokenizer, flash-attn, KV cache
5. **Voxtral-Mini 3B** — second speech-LLM (Whisper-large-v3 encoder +
   Llama 3 LLM), ported from zero to working CLI in one session

The user prefers high autonomy — act first, ask only for remote-touching
/ destructive / architectural decisions. See `feedback_autonomy.md` in
the auto-memory directory.
