# CrispASR — comprehensive TODO

Last updated: 2026-04-09. Covers all 7 (soon 8) runtimes, pending work,
and strategic direction.

---

## Immediate (next session)

### 1. ~~Build + commit GPU init changes~~ ✅ DONE
Switched canary_ctc, qwen3_asr, voxtral to `ggml_backend_init_best()`.
Removed unnecessary Metal/CUDA ifdef includes. Built and verified.

### 2. Voxtral 4B Realtime — port the model
Weights downloaded to `/mnt/akademie_storage/voxtral-4b-realtime/` (17 GB).
Architecture plan at `voxtral-4b-todo.md`. Key differences from 3B:
RoPE encoder (not absolute pos embed), sliding window attention (encoder
window=750, LLM window=8192), tied embeddings, 26 layers, FFN=9216,
RoPE θ=1e6. Estimated effort: ~3-4 days.

### 3. ~~Upload voxtral HF README~~ ✅ DONE
Committed `hf_readmes/voxtral-mini-3b-2507-GGUF.md` to git.

---

## GPU support audit (current state)

| Runtime | GPU status |
| --- | --- |
| **parakeet** | ✅ Metal + CUDA (explicit init) |
| **cohere** | ✅ `ggml_backend_init_best()` |
| **canary** | ✅ Metal + CUDA (explicit init) |
| **canary_ctc** (nfa-align) | ✅ `ggml_backend_init_best()` with CPU fallback |
| **wav2vec2** (cohere-align) | ❌ Old ggml pattern, no backend API — needs refactor (~200 LOC) |
| **qwen3_asr** | ✅ `ggml_backend_init_best()` with CPU fallback |
| **voxtral** | ✅ `ggml_backend_init_best()` with CPU fallback |

All runtimes except wav2vec2 now auto-detect GPU.

---

## Timestamps / timecodes (current state)

| Runtime | Timestamps | How |
| --- | --- | --- |
| **parakeet** | ✅ TDT duration head (~80ms) | Native, always available |
| **cohere** | ✅ Cross-attention DTW (~360ms MAE) | Native, -ts flag |
| **canary** | ✅ Decoder cross-attn + optional CTC re-align | Native + -am flag for CTC |
| **nfa-align** | ✅ CTC Viterbi forced alignment (~78ms MAE) | Core purpose |
| **cohere-align** | ✅ char-level CTC (~30ms MAE, English only) | Core purpose |
| **qwen3_asr** | ✅ CTC aligner second pass (~78ms MAE) | -am + -timestamps flags |
| **voxtral** | ✅ CTC aligner second pass (~78ms MAE) | -am + -timestamps flags |

All runtimes now have word-level timestamps. SRT/VTT output supported in
parakeet, canary, qwen3-asr, voxtral.

---

## CLI feature matrix

| Feature | parakeet | cohere | canary | qwen3-asr | voxtral |
| --- | --- | --- | --- | --- | --- |
| `-m` model | ✅ | ✅ | ✅ | ✅ | ✅ |
| `-f` audio | ✅ | ✅ | ✅ | ✅ | ✅ |
| `-t` threads | ✅ | ✅ | ✅ | ✅ | ✅ |
| `-l` language | — | ✅ 14 | ✅ 25 (sl/tl) | auto-detect | ✅ (tokenizer) |
| `-am` CTC align | — | — | ✅ | ✅ | ✅ |
| `-timestamps` | native | `-ts` | native + CTC | ✅ | ✅ |
| `-osrt` | ✅ (file) | ✅ (file) | ✅ (file) | ✅ (stdout) | ✅ (stdout) |
| `-ovtt` | ✅ (file) | ✅ (file) | ✅ (file) | ✅ (stdout) | ✅ (stdout) |
| `-np` quiet | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--flash` | ✅ | ✅ | ✅ | always on | always on |
| `-vad-model` | ✅ | ✅ | ✅ | — | — |
| `-ml` max-len | ✅ | ✅ | ✅ | — | — |
| `-ck` chunking | ✅ | internal | ✅ | — | — |

---

## Tekken tokenizer — ✅ DONE

`voxtral_tokenize()` is now fully implemented: rank-based byte BPE with
pre-tokenizer splitting and special token handling. Verified with round-trip
tests on English, German, special tokens, and the full prompt template.
The voxtral CLI now uses the tokenizer instead of hardcoded token IDs,
supporting any language without ID lookup tables.

---

## Performance optimization opportunities

### Already done
- [x] F16 KV cache (qwen3_asr, voxtral)
- [x] Flash attention on prefill + decode (qwen3_asr, voxtral)
- [x] Last-token-only lm_head slice (qwen3_asr, voxtral)
- [x] Q4_K weight quantization with Q4_0 fallback for odd-width tensors
- [x] Baked mel filterbank (no runtime recomputation)
- [x] GPU auto-detection via ggml_backend_init_best()

### Next wins
- [ ] **GPU backend** (~5-10× on LLM forward, biggest single win for voxtral 3B)
- [ ] **Speculative decoding** — use a smaller draft model to propose N tokens
- [ ] **Continuous batching** — pipeline multiple audio files for throughput
- [ ] **Chunked long-audio** — for audio >30s, split into overlapping 30s chunks

---

## Voxtral 4B Realtime port (ready to start)

Weights downloaded to `/mnt/akademie_storage/voxtral-4b-realtime/`.
Plan at `voxtral-4b-todo.md`. Key new pieces:
- RoPE encoder (not absolute pos embed)
- Sliding window attention (encoder=750, LLM=8192)
- 26 layers, FFN=9216, RoPE θ=1e6
- Tied embeddings
- `VoxtralRealtimeForConditionalGeneration` (different HF class)

Estimated effort: ~3-4 days.

---

## Model-specific pending items

### Qwen3-ASR
- [x] Timestamps ✅ (CTC aligner second pass)
- [ ] Streaming support (chunked audio → incremental transcript)
- [ ] Test on more languages (only tested English + German)
- [ ] The Qwen3-ForcedAligner-0.6B companion model

### Voxtral 3B
- [x] Timestamps ✅ (CTC aligner second pass)
- [x] Tekken tokenizer ✅ (full BPE encoder)
- [ ] Audio understanding mode (Q&A about audio content)
- [ ] Function calling from voice — needs tool-use prompt format
- [ ] Long audio >30s — needs chunked encoder or padding strategy
- [ ] Test on non-English languages

### Parakeet
- [x] GPU support ✅
- [x] Word timestamps ✅ (TDT)
- [x] --flash CLI flag ✅
- [ ] Auto language detection sometimes picks wrong language on accented audio

### Canary
- [x] GPU support ✅
- [x] Word timestamps ✅ (decoder + optional CTC re-align)
- [x] --flash CLI flag ✅
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
| `cstr/qwen3-asr-0.6b-GGUF` | ✅ shipped | F16 + Q8_0 + Q4_K |
| `cstr/voxtral-mini-3b-2507-GGUF` | ✅ shipped | Q4_K + Q8_0 + README |
| `cstr/voxtral-mini-4b-realtime-GGUF` | ❌ pending port | Weights downloaded |

---

## Code quality / cleanup

- [x] Remove `#ifdef GGML_USE_METAL/CUDA` includes ✅
- [x] Suppress -Wunused-variable warnings in voxtral.cpp ✅
- [ ] Factor out shared mel compute code (~150 LOC duplicated between
      qwen3_asr.cpp and voxtral.cpp — similar but voxtral pads to 3000)
- [ ] Factor out shared WAV reader (duplicated in qwen3-asr/voxtral CLIs;
      parakeet/canary use common-whisper instead)
- [ ] Factor out shared .npy loader (duplicated in every test driver)
- [ ] The voxtral Tekken vocab blob stored as F32 tensor (wasteful ~5 MB)

---

## Strategic notes

### llama.cpp mtmd compatibility

Analysed in `voxtral-comparison.md` (local, not committed). Summary:
- llama.cpp's mtmd support for Voxtral has two unfixed bugs (#17868,
  #18419), worse WER than transformers/vLLM at same precision
- Our standalone ggml approach avoids all these issues
- Recommendation: keep CrispASR standalone, optionally produce
  llama.cpp-compatible GGUFs for ecosystem users

### predict-woo/qwen3-asr.cpp PR

https://github.com/predict-woo/qwen3-asr.cpp/pull/7 — CMake build
fixes for Linux. Status: open, awaiting review.

---

## German test audio samples

Saved at `/mnt/storage/german-samples/`:
- `berlin_word.wav` (0.7s) — "Berlin"
- `bundeskanzler_word.wav` (1.6s) — "Bundeskanzler"
- `jazeschann.wav` (4.8s) — "Leider zu spät"
- `De-Abwasch-article.wav` (79.4s) — Wikipedia: Dishwashing
- `De-Afghani-article.wav` (207.6s) — Wikipedia: Afghani currency

All from Wikimedia Commons, CC-licensed, 16 kHz mono WAV.

---

## Session history

1. **Cohere Transcribe** — original port, mel norm bug, DTW timestamps
2. **Parakeet TDT** — FastConformer + TDT decoder, free word timestamps
3. **Canary 1B v2** — speech translation, nfa-align CTC aligner
4. **Qwen3-ASR 0.6B** — first speech-LLM port, BPE tokenizer, flash-attn, KV cache
5. **Voxtral-Mini 3B** — second speech-LLM, ported from zero in one session
6. **Feature completion** (2026-04-09) — GPU init for all runtimes, word timestamps
   for qwen3-asr/voxtral/canary via CTC aligner, Tekken tokenizer, --flash for
   parakeet/canary, SRT/VTT/-np for qwen3-asr/voxtral, Voxtral 4B weights downloaded
