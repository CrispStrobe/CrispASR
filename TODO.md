# CrispASR — comprehensive TODO

Last updated: 2026-04-10.

---

## In-progress: GPU/performance correctness fixes

### P1. Parakeet decoder: port LSTM+Joint to ggml graphs
**Problem:** The TDT transducer decoder (LSTM predictor + Joint head) runs
entirely on CPU using raw float* loops with weights explicitly downloaded
from the backend buffer. The encoder is a proper ggml graph, but the
decoder cannot benefit from GPU at all.

**Fix:** Replace the manual LSTM step (`predictor_step`) and joint network
(`joint_step`) with small ggml graphs. Each LSTM step is a small matmul
(hidden_dim × 4*hidden_dim), so the graph overhead may not help for
per-token stepping. Two options:
- (a) Port to ggml matmul ops — gains GPU acceleration but adds per-step
  graph build/compute overhead. Best for batch mode (multiple tokens).
- (b) Keep CPU but use ggml_mul_mat with a CPU backend — at least gets
  the weights in a consistent allocation pattern.

**Risk:** The per-token LSTM stepping is inherently sequential (each step
depends on the previous hidden state). Even on GPU the latency may not
improve much. The encoder is the dominant cost anyway. Consider whether
this is worth the complexity.

**Effort:** ~200 LOC, ~1 day. Medium risk.

### P2. canary_ctc: fix single-backend scheduler (no CPU fallback)
**Problem:** `canary_ctc.cpp` line 713 creates `ggml_backend_sched` with
only 1 backend, and `backend_cpu` is aliased to the primary backend (line
676). If the primary is GPU, any unsupported op will fail instead of
falling back to CPU.

**Fix:** Add a proper 2-backend setup: `backends[0] = primary` (GPU or
CPU), `backends[1] = cpu_fallback`. Match the pattern used by canary.cpp
and cohere.cpp.

**Effort:** ~20 LOC, 15 minutes. Low risk.

### P3. qwen3_asr + voxtral: stop recreating ggml_backend_sched per call
**Problem:** Both runtimes free and recreate `ggml_backend_sched` on every
compute call (encoder, prefill, each decode step). This adds per-call
allocation overhead. The original reason was to handle different graph
sizes across stages (conv, encoder, LLM variants).

**Fix:** Create the scheduler once at init with the worst-case node budget
(the LLM prefill graph is largest). Use `ggml_backend_sched_reset()` between
calls instead of free+recreate. The max graph size can be computed once
during init by building the largest graph variant and measuring its node
count.

**Effort:** ~80 LOC per runtime, ~2 hours. Low risk.

### P4. Cohere: upgrade self-attention KV cache from F32 to F16
**Problem:** Cohere's self-attention KV cache uses F32 while canary and
the speech-LLMs use F16. This wastes 2× GPU memory and bandwidth.

**Fix:** Change `kv_k` and `kv_v` tensor types from `GGML_TYPE_F32` to
`GGML_TYPE_F16` in the KV cache allocation. Update the decoder graph to
use F16 KV reads/writes (ggml_cpy handles the F32↔F16 conversion). Flash
attention already expects F16 K/V on the CPU backend, so this may already
be partially wired.

**Risk:** F16 KV can cause minor precision loss in long decoding sequences.
Cohere's cross-attention KV is already F16 without issues, so self-attention
should be fine too.

**Effort:** ~30 LOC, 30 minutes. Low risk.

---

## GPU support audit (updated)

| Runtime | Encoder | Decoder/LLM | KV Cache | Mel/STFT | Backend sched |
| --- | --- | --- | --- | --- | --- |
| **parakeet** | ggml ✅ | **raw CPU** ❌ (P1) | N/A (LSTM) | raw C++ | 2-backend ✅ |
| **cohere** | ggml ✅ | ggml ✅ | F32 self ⚠️ (P4), F16 cross ✅ | raw C++ (cblas mel) | 2-backend ✅ |
| **canary** | ggml ✅ | ggml ✅ | F32 self, F16 cross ✅ | raw C++ | 2-backend ✅ |
| **canary_ctc** | ggml ✅ | N/A | N/A | raw C++ | **1-backend** ❌ (P2) |
| **qwen3_asr** | ggml ✅ | ggml ✅ | F16 ✅ | raw C++ | recreated ⚠️ (P3) |
| **voxtral** | ggml ✅ | ggml ✅ | F16 ✅ | raw C++ | recreated ⚠️ (P3) |
| **wav2vec2** | old ggml ❌ | N/A | N/A | N/A | no backend API |

**Universal gap:** Mel/STFT/FFT is CPU-only in all runtimes. Each has its
own C++ FFT. This is the biggest remaining GPU opportunity but would require
porting STFT to ggml ops (ggml has no native FFT, would need a custom op
or pre-computed DFT matrix as a matmul).

---

## Feature parity gaps (speech-LLMs vs whisper)

| Feature | whisper | qwen3_asr | voxtral | canary | parakeet | cohere |
| --- | --- | --- | --- | --- | --- | --- |
| Temperature sampling | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Beam search | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Repetition penalty | ✅ (loop detect) | ❌ | ❌ | ❌ | ❌ | ❌ |
| VAD segmentation | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Streaming/callback | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Custom prompt | ✅ | partial | partial | ❌ | ❌ | ❌ |
| Multi-turn (arch) | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Quantize tool | ✅ | shared | ❌ | ❌ | ❌ | ✅ |

---

## Timestamps — ✅ COMPLETE

All runtimes now have word-level timestamps.

| Runtime | Method | Accuracy |
| --- | --- | --- |
| parakeet | TDT duration head (native) | ~80ms |
| cohere | Cross-attention DTW | ~360ms MAE |
| canary | Decoder cross-attn + optional CTC re-align (-am) | ~78ms with CTC |
| nfa-align | CTC Viterbi forced alignment | ~78ms |
| cohere-align | char-level wav2vec2 CTC | ~30ms (English) |
| qwen3_asr | CTC aligner second pass (-am) | ~78ms |
| voxtral | CTC aligner second pass (-am) | ~78ms |

---

## Tekken tokenizer — ✅ COMPLETE

Full `voxtral_tokenize()` implemented and verified.

---

## Performance optimizations done

- [x] F16 KV cache (qwen3_asr, voxtral)
- [x] Flash attention prefill + decode (qwen3_asr, voxtral)
- [x] Last-token-only lm_head slice (qwen3_asr, voxtral)
- [x] Q4_K weight quantization with Q4_0 fallback
- [x] Baked mel filterbank (no runtime recomputation)
- [x] GPU auto-detection via ggml_backend_init_best()

---

## Voxtral 4B Realtime — ✅ COMPLETE

Ported, debugged, quantized, uploaded to HF.

| GGUF | Size | Total (11s audio, CPU) |
| --- | --- | --- |
| F16 | 8.3 GB | 133s |
| Q8_0 | 4.5 GB | 79s |
| **Q4_K** | **2.4 GB** | **49s** |

English + German verified. 13 languages supported.

---

## Model-specific pending

### Qwen3-ASR
- [ ] VAD segmentation for long audio
- [ ] Temperature/sampling controls
- [ ] Streaming support
- [ ] Test more languages

### Voxtral 3B
- [ ] Variable-length mel (currently pads to 3000=30s; needs encoder to handle variable T)
- [ ] Audio understanding mode (Q&A)
- [ ] Long audio >30s chunking
- [ ] Temperature/sampling controls
- [ ] Test non-English languages

### Parakeet
- [ ] Port LSTM decoder to ggml (P1)
- [ ] Auto language detection accuracy on accented audio

### Canary
- [ ] Speech translation quality validation

### Voxtral 4B Realtime
- [ ] SRT/VTT subtitle output
- [ ] Temperature/sampling controls
- [ ] Reduce right padding (17→10 tokens, matching voxtral.c)

### Granite Speech 1B
- [ ] Build CLI (`granite-main` or integrate into consolidated CLI)
- [ ] Quantize Q4_K + Q8_0
- [ ] Upload to HF (`cstr/granite-speech-4.0-1b-GGUF`)
- [ ] Word timestamps via CTC aligner
- [ ] Performance tuning (encoder is slow: per-layer CPU → single ggml graph)
- [ ] Remove dead ggml graph encoder (`granite_build_encoder`)

### Cohere
- [x] F32→F16 self-attention KV ✅ (P4)
- [ ] Upstream ffmpeg mp4 bug (UPSTREAM.md)

---

## HF releases

| Repo | Status |
| --- | --- |
| `cstr/parakeet-tdt-0.6b-v3-GGUF` | ✅ shipped |
| `cstr/parakeet_de_med-GGUF` | ✅ shipped |
| `cstr/canary-1b-v2-GGUF` | ✅ shipped |
| `cstr/canary-ctc-aligner-GGUF` | ✅ shipped |
| `cstr/cohere-transcribe-03-2026-GGUF` | ✅ shipped |
| `cstr/qwen3-asr-0.6b-GGUF` | ✅ shipped |
| `cstr/voxtral-mini-3b-2507-GGUF` | ✅ shipped |
| `cstr/voxtral-mini-4b-realtime-GGUF` | ✅ shipped (Q4_K + Q8_0) |
| `cstr/granite-speech-4.0-1b-GGUF` | ❌ pending quantize + upload |

---

## Code quality (deferred until consolidated CLI)

- [ ] Factor out shared mel compute (~150 LOC)
- [ ] Factor out shared WAV reader
- [ ] Factor out shared .npy loader
- [ ] Voxtral Tekken vocab blob stored as F32 (wasteful)

---

## Session history

1. **Cohere Transcribe** — original port, mel norm bug, DTW timestamps
2. **Parakeet TDT** — FastConformer + TDT decoder, free word timestamps
3. **Canary 1B v2** — speech translation, nfa-align CTC aligner
4. **Qwen3-ASR 0.6B** — speech-LLM port, BPE tokenizer, flash-attn, KV cache
5. **Voxtral-Mini 3B** — speech-LLM, ported from zero in one session
6. **Feature completion** (2026-04-09) — GPU init, timestamps, Tekken,
   --flash, SRT/VTT, Voxtral 4B downloaded
7. **Performance fixes** (2026-04-09) — P1-P4 GPU correctness fixes
8. **Voxtral-Mini 4B Realtime** (2026-04-09/10) — full port from scratch,
   7 critical bugs found via Kaggle ground truth + 3 reference implementations
   (voxtral.c, voxmlx, voxtral-rs). Q4_K quantized: 2.4GB, 49s for 11s audio.
   Uploaded to HF.
9. **Granite Speech 4.0-1B** (2026-04-10) — full port: Conformer encoder
   (Shaw RPE, depthwise conv, batch norm folding) + BLIP-2 Q-Former projector +
   Granite LLM (μP multipliers). 6 bugs found: Hann window centering, Q-Former LN
   target, embedding multiplier placement, CTC dim hardcoding, native GQA, and
   **RoPE mode (NEOX vs NORMAL)** — the critical fix for correct transcription.
