# CrispASR v0.8.6

This release adds **one new TTS backend and four new ASR/multimodal backends**,
makes **TADA fully controllable at query time**, hardens the **native-Vulkan TADA
path**, and adds **consent-gated speaker identification** to diarization — plus a
batch of GPU-stability, server, and build/CI fixes.

---

## What's new

### dots.tts — 2B continuous-AR TTS backend (#200)

A new text-to-speech backend for **`rednote-hilab/dots.tts`**, a 2B continuous
autoregressive TTS model (Llama-style LLM + flow-matching DiT + BigVGAN vocoder).

- **End-to-end synthesis** — `"Hello world."` and longer text round-trip to
  **ASR-verbatim** audio. The full pipeline is ported and validated against the
  PyTorch reference (DiT forward cos = 0.9999, LLM forward cos = 0.999).
- **CAM++ voice cloning** — clone a speaker from a reference clip with `--voice`
  (encoder cos 0.9996 / global-cond 0.9999 vs reference).
- **Mixed-quant packaging** — the DiT stays F16 while the LLM and PatchEncoder
  quantize to Q8/Q4_K (quantizing the DiT derails generation). Published as
  `f16` / `q8` (3.1 GB) / `q4_k` (2.3 GB) on
  [`cstr/dots-tts-soar-GGUF`](https://huggingface.co/cstr/dots-tts-soar-GGUF).
- **Incremental PatchEncoder** — streaming `O(N²)→O(N)` patch encoding so long
  text reaches EOS in reasonable time (the earlier `O(N³)` recompute made long
  text too slow to terminate — it was never an EOS bug).
- **Metal GPU backend** — single-backend raw-gallocr compute (no sched hazards),
  **5.5× faster on long text**, ASR-verbatim. Opt out with
  `CRISPASR_DOTS_TTS_CPU=1`. (CUDA path untested.)
- Root-cause fix along the way: the LLM step was missing the attention scale
  (`attn_scale=0` → uniform attention → garbage prefill).

### Higgs-Audio v3 STT — Whisper-LV3 + Qwen3-1.7B ASR backend

A new ASR backend for **`bosonai/higgs-audio-v3-stt`** (Whisper-large-v3 encoder
+ Qwen3-1.7B decoder).

- **Chunked Whisper encoder** — audio is encoded in independent 4 s chunks with
  chunk-local positions and concatenated, matching the upstream blueprint. (A
  single padded 30 s window let global attention over the silence-pad derail the
  decoder.) Transcribes JFK and a 45 s / 12-chunk clip **verbatim** vs the bf16
  reference at F16 / Q8 / Q4.
- **Custom prompt** — `--ask "<prompt>"` steers the decoder (e.g. translation or
  Q&A over the audio), with an `-l <lang>` / `params.language` hint.
- `CAP_UNBOUNDED_INPUT | CAP_INTERNAL_CHUNKING` — the backend chunks long audio
  internally (no CLI window-split duplication).
- Published on
  [`cstr/higgs-audio-v3-stt-GGUF`](https://huggingface.co/cstr/higgs-audio-v3-stt-GGUF).

### ARK-ASR-3B — experimental Whisper-RoPE + Qwen2.5-3B ASR backend

A new **experimental** ASR backend (Whisper-large-v3 encoder with partial RoPE +
MLP adapter + stock Qwen2.5-3B decoder).

- **GPU by default** (Metal-validated; opt out with `CRISPASR_ARKASR_CPU=1`).
- **Single-pass whole-audio** decode — kills the language drift that came from
  *our* 30 s chunking (not the model). Capped by
  `CRISPASR_ARKASR_MAX_SINGLE_PASS_S=300`.
- **Optional language steering** + a live test; diff-harness-validated
  (mel cos 0.999993, logits cos 0.999646).
- Published as `f16` / `q8_0` / `q4_k` on
  [`cstr/ark-asr-3b-GGUF`](https://huggingface.co/cstr/ark-asr-3b-GGUF). Marked
  experimental/WIP.

### Parakeet CTC 1.1B (Japanese) — FastConformer-CTC ASR backend

A new Japanese ASR backend for **`grider-transwithai/parakeet-ctc-1.1b-ja`** — a
42-layer FastConformer with a CTC decoder.

- Published as `f16` / `q8_0` / `q4_k` on
  [`cstr/parakeet-ctc-1.1b-ja-GGUF`](https://huggingface.co/cstr/parakeet-ctc-1.1b-ja-GGUF),
  with a model card and architecture docs.

### Gemma-4 E4B — multimodal ASR model variant (#196)

- Added the **`gemma4-e4b`** model (same `gemma4` architecture as E2B, larger
  decoder) to the registry, with a Kaggle conversion kernel and README. Published
  on [`cstr/gemma4-e4b-it-GGUF`](https://huggingface.co/cstr/gemma4-e4b-it-GGUF).
- The 12B `gemma4_unified` variant is explicitly rejected by the converter with a
  clear message (not yet supported).

### TADA TTS — query-time control + reliability (#197, #201)

- **Single-pass whole-text generation (#197)** — TADA now generates the whole
  text in one pass like the upstream `tada.py`, instead of splitting on
  punctuation. Splitting an isolated `"Hi."` produced a 9.4 s pause + hum; the
  whole-text path removes the spurious pauses and hums.
- **Per-request flow-matching knobs (#197)** — expose the acoustic FM controls at
  query time: `num_fm_steps` (accuracy vs speed), `acoustic_cfg`, and a new
  `noise_temp`. The talker sampler (`top_k` / `do_sample` /
  `num_acoustic_candidates`, temperature, rep-penalty) is now per-request and
  wired through every consumer (CLI, session ABI, server).
- **Switch voice per request without a restart (#201)** — the HTTP/session path
  can change the voice reference between requests (chatterbox-style cached
  last-voice key); previously only a restart picked up a new voice.

### Diarization — consent-gated speaker identification

- New **`--diarize-speakers`** convenience alias (enables diarization with
  session-scoped speaker clustering → transient `(speaker N)` labels).
- A named, persistent **1:N voiceprint database** (`--speaker-db` /
  `--enroll-speaker`) is now **off by default** behind an explicit
  **`--speaker-db-consent`** flag (GDPR Art. 9 / biometric-data scoping). Session
  clustering needs no consent flag; only the persistent named DB does.

### Server — diarized JSON output (#205, #206)

- New **`diarized_json`** response format (#206) — structured per-segment speaker
  + text output over HTTP.
- Fixed **`--max-len`** handling for the granite backend on the server (#205).

---

## Bug fixes & hardening

| Area | Fix |
|------|-----|
| TADA / Vulkan (#192) | Native-Vulkan garbled/empty output was the **codec**, not the FM head — run the DAC codec on **CPU when GPU = Vulkan** (the codec graph miscomputes at length under MoltenVK; the talker/FM keep their native-Vulkan path). Gated `CRISPASR_TADA_VULKAN_NATIVE=1`, default CPU-fallback. |
| TADA / Vulkan (#192) | Force F32 KV read on the native-Vulkan path to unblock the GQA `REPEAT`-f16 abort; direct-on-backend (gallocr) compute for the talker/FM graphs to avoid sched cross-backend corruption; cap codec expansion to prevent a runaway allocation. |
| TADA codec | Dropped **805 MB of dead precomputed attention masks** — the codec GGUF shrank from ~1055 MB to ~250 MB (byte-identical output); re-uploaded to `cstr/tada-tts-{1b,3b-ml}-GGUF` and updated registry size hints. |
| lfm2-audio (#199) | GPU-safe embed + decode — fixes a **CUDA `ACCESS_VIOLATION`** crash. |
| dots.tts | Vocoder compute-graph node budget sized for the 6×3 ResBlocks + MI-LSTM at 1024 frames; numerous PatchEncoder / DiT / KV-cache load + graph-size fixes during the port. |
| Build (#191) | Robust link against system `libopusfile`; `.opus` support is now truly optional. |
| crispasr-sys (#203) | `build.rs` auto-uses Ninja + ccache when available and passes `--parallel` to `cmake --build`. |
| CI | Install hipBLAS / rocBLAS and set the ROCm prefix for HIP release builds. |
| CI | Sync the Go cgo `LDFLAGS` for the new `higgs-stt` / `dots-tts` / `ark-asr` static libs (fixes the Go bindings link + the `cgo-ldflags-drift` check). |
| Static analysis | Guard an `all_times[i]` index in the TADA decode expansion (cppcheck `containerOutOfBounds`). |

---

## Upgrading

No breaking changes to the C ABI, bindings, or CLI flags — all additions.

- **Diarization named speaker DB is now opt-in.** Session-scoped clustering
  (`--diarize` / `--diarize-speakers`) is unchanged. The persistent named
  voiceprint database (`--speaker-db` / `--enroll-speaker`) now requires
  `--speaker-db-consent`; without it those flags are inert. This is a deliberate
  privacy default, not a regression.
- **TADA generates whole text in one pass** by default (#197). If you relied on
  the previous punctuation-split behaviour, note that single-pass matches the
  upstream reference and removes spurious pauses/hums.
- New per-request TADA knobs (`num_fm_steps`, `acoustic_cfg`, `noise_temp`,
  talker sampler) and per-request voice switching are additive; older bindings
  without the setters soft-no-op.

---

## Full changelog

`git log v0.8.5..v0.8.6 --oneline --no-merges`
