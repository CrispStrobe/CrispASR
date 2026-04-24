# CrispASR — Pending work

Pending roadmap items. Each is self-contained with files, approach, and
effort estimate. Completed items have been moved to `HISTORY.md`.

**Current state (April 2026, v0.5.0):** 17 ASR backends, unified CLI,
OpenAI-compatible server, shared `src/core/` library, FireRedPunc
post-processor, C-ABI + Python/Rust/Dart wrappers, CI on 6 platforms.

---

## Priority ordering

| Priority | Item | Impact | Effort | Status |
|---|---|---|---|---|
| ~~HIGH~~ | ~~#36 ASCII punc mapping~~ | ~~Fix English output~~ | ~~Trivial~~ | **DONE** |
| ~~HIGH~~ | ~~#37 Progressive SRT (#24)~~ | ~~Streaming subtitles~~ | ~~Medium~~ | **DONE** |
| ~~MEDIUM~~ | ~~#38 Fullstop-punc multilingual~~ | ~~DE/FR/IT punc~~ | ~~Medium~~ | **DONE** |
| ~~MEDIUM~~ | ~~#39 Session API backends~~ | ~~5 backends wired~~ | ~~Low-Med~~ | **DONE** |
| **MEDIUM** | [#40 Moonshine variants](#40-more-moonshine-model-variants) | Easy model uploads | Trivial | Needs ext converter |
| **MEDIUM** | [#5 Reference backends](#5-reference-backends-for-parakeetcanarycohere) | Test infra completeness | Medium |
| **LOW** | [#41 Moonshine IPA](#41-moonshine-phoneme--ipa-output) | Niche (phoneme output) | Medium-High |
| **LOW** | [#7 voxtral4b streaming](#7-native-voxtral4b-streaming) | Complex, niche | High |
| **LOW** | [#9 Parakeet TDT GPU](#9-parakeet-tdt-decoder-gpu) | Small gain (encoder dominates) | Medium |
| **LOW** | [#11 WebSocket server](#11-websocket-streaming-server) | Needs new dependency | High |
| **LOW** | [#16 Shaw RPE](#16-shaw-rpe-for-granite-graph) | Accuracy edge case | Medium |
| **LOW** | [#15 CMake rename](#15-cmake-target-rename) | Cosmetic | Low |
| **LOW** | [#18 Aligner LIS](#18-qwen3-aligner-accuracy) | Niche improvement | Low |
| **BLOCKED** | [#42 VibeVoice-ASR 7B](#42-vibevoice-asr-7b) | Needs ≥16 GB RAM | High |
| **BLOCKED** | [#43 Fun-ASR-Nano](#43-fun-asr-nano) | License unclear | Medium |

---

## 36. ASCII punctuation mapping

**Problem:** FireRedPunc outputs Chinese full-width marks (`，` `。` `？` `！`)
even for English text.

**Fix:** Auto-detect Latin-script input → map to ASCII (`, . ? !`).

**Files:** `src/fireredpunc.cpp` — 4 string replacements after BERT pass.

**Effort:** Trivial (~10 LOC).

---

## 37. Progressive SRT output (issue #24)

**Problem:** Non-whisper backends buffer all segments and flush stdout at
the end. 30-minute files produce nothing until fully processed. Media
players (PotPlayer) need progressive SRT for real-time subtitle display.

**Approach:** `--flush-after N` flag (default: 0 = all-at-end). N=1 means
print each SRT entry as its VAD slice finishes.

**Implementation:**
- Per-slice loop in `crispasr_run.cpp`: after `transcribe()`, immediately
  format + print SRT entries.
- Post-processing (punc, strip) runs per-slice.
- Diarization: skip or defer when progressive (needs full context).
- Maintain SRT index counter across slices.

**Files:** `examples/cli/crispasr_run.cpp`, `whisper_params.h`, `cli.cpp`.

**Effort:** Medium (~100 LOC).

---

## 38. Fullstop-punctuation-multilingual

**Model:** `oliverguhr/fullstop-punctuation-multilang-large` (MIT)

**Architecture:** XLM-RoBERTa-large (560M params). Token classification
with 6 classes (`. , ? - :` + no-punc). Includes truecasing.

**Languages:** English, German, French, Italian.

**Differences from FireRedPunc:**
- RoBERTa (no token_type_embeddings, different LN order)
- 250K SentencePiece vocab (vs 21K WordPiece)
- 6 classes (vs 5), ASCII punctuation output

**Approach:** Extend `fireredpunc.cpp` to detect model type from GGUF
metadata and handle both BERT and RoBERTa. Or create separate runtime.

**Other candidates:**
- `felflare/bert-restore-punctuation` (MIT, BERT-base, English, truecasing)
- `xashru/punctuation-restoration` (Apache-2.0, XLM-RoBERTa, 40+ langs)

**Size:** ~1.1 GB F16, ~300 MB Q4_K.

**Effort:** Medium (~200 LOC converter + ~100 LOC runtime).

---

## 39. Session API for remaining backends

**Problem:** C-ABI session API missing cases for glm-asr, kyutai-stt,
firered-asr, moonshine, omniasr. These work via CLI but not via
Python/Rust/Dart wrappers.

**Fix:** Add switch cases in `crispasr_c_api.cpp` for open/transcribe/close.
Context struct pointers already exist.

**Files:** `src/crispasr_c_api.cpp` (~30 LOC per backend × 5).

**Effort:** Low-Medium (~150 LOC).

---

## 40. More Moonshine model variants

Convert + upload to HuggingFace:
- `moonshine-base` (61.5M, better WER)
- `moonshine-streaming-tiny/small/medium`
- `moonshine-tiny-{ja,ar,ko,zh,vi,uk}` (multilingual)
- `moonshine-base-{ja,uk,vi,zh,ar,ko}` (multilingual)

Existing converter handles all sizes. Run + quantize + upload.

**Effort:** Trivial per-model.

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

## 15. CMake target rename

Rename `whisper-cli` → `crispasr` across CMake/CI/tests/scripts (~50 refs).
Keep backward-compat symlink.

---

## 16. Shaw RPE for granite graph

Add query-dependent Shaw RPE to granite ggml encoder graph (currently
uses flash_attn_ext without RPE). Manual attention: QK^T + RPE bias +
softmax + V matmul.

**Effort:** ~80 LOC.

---

## 18. Qwen3 aligner accuracy

Add full LIS (longest increasing subsequence) monotonicity fix and
language-specific word tokenization (CJK, nagisa for Japanese).

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

## Ecosystem expansion (lower priority)

### New backends from PazaBench assessment (see HISTORY.md #30)

| Model | License | Approach | Priority |
|---|---|---|---|
| Wav2Vec2 Conformer | Apache-2.0 | Conformer attention variant | Medium |
| Qwen2-Audio 7B | Apache-2.0 | Whisper encoder + Qwen2 LLM | Medium |
| OmniASR larger (1B/3B/7B) | Apache-2.0 | Same converter, bigger models | Medium |
| NeMo Canary-Qwen-2.5b | Apache-2.0 | FastConformer + Qwen2.5 decoder | Medium |
| Paza / Phi-4 | MIT | 14B multimodal, defer to llama.cpp | Low |

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

### Other

- **FireRedASR GPU decoder** — persistent ggml graph for self-attn/cross-attn/MLP per decode step
- **OmniASR-LLM beam search** — beam=2+ with N hypothesis KV caches
- **TTS module** — VibeVoice-1.5B σ-VAE decoder for text-to-speech
- **ggml_conv_1d_dw F16 im2col fix** — CPU depthwise conv without im2col for VibeVoice precision
