# Architecture

CrispASR is structured around three layers on top of whisper.cpp.
The split between `src/` (library) and `examples/cli/` (presentation)
is deliberate: **every algorithm** — VAD, diarization, LID, CTC
alignment, HF download/cache, model registry — lives in `src/` behind
a stable C-ABI (`src/crispasr_c_api.cpp`), and every consumer (CLI,
Dart, Python, Rust, Go, Java, Ruby) reaches it through the same
symbols. The CLI keeps only presentation + UX policy.

```
┌───────────────────────────────────────────────────────────────────┐
│ examples/cli/cli.cpp (the crispasr binary)                        │
│   Parses CLI args, dispatches to backend when --backend           │
│   is set or GGUF arch is non-whisper; otherwise runs whisper_full │
│   unchanged                                                        │
├───────────────────────────────────────────────────────────────────┤
│ examples/cli/crispasr_*_cli.{h,cpp}                               │
│   Thin CLI shims for policy only — auto-download, TTY prompts,    │
│   sherpa-ONNX subprocess fallbacks. Delegate the algorithmic      │
│   work to the shared library below.                                │
├───────────────────────────────────────────────────────────────────┤
│ src/crispasr_c_api.cpp — C-ABI (shared with Dart / Python / Rust) │
│   crispasr_vad.{h,cpp}           Silero VAD + whisper-style       │
│                                  stitching, timestamp remap       │
│   crispasr_diarize.{h,cpp}       energy / xcorr / vad-turns /     │
│                                  native pyannote diarization      │
│   crispasr_lid.{h,cpp}           whisper-tiny + silero-native LID │
│   crispasr_aligner.{h,cpp}       canary-CTC + qwen3-forced-aligner│
│   crispasr_cache.{h,cpp}         HF download + ~/.cache/crispasr  │
│   crispasr_model_registry.{h,cpp} backend → canonical GGUF URL    │
├───────────────────────────────────────────────────────────────────┤
│ src/{whisper,parakeet,canary,canary_ctc,cohere,qwen3_asr,         │
│      voxtral,voxtral4b,granite_speech,silero_lid,pyannote_seg}.cpp│
│   Per-model runtimes (public C APIs)                              │
├───────────────────────────────────────────────────────────────────┤
│ src/core/      — shared model primitives (crispasr-core)          │
│   mel.{h,cpp}          log-mel spectrogram (NeMo + HF clusters)   │
│   ffn.h                SwiGLU + SiLU FFN helpers                  │
│   attention.h          Llama-style self-attention + flash-attn    │
│   gguf_loader.{h,cpp}  Unified GGUF open / weight mmap / lookup   │
├───────────────────────────────────────────────────────────────────┤
│ ggml                                                               │
└───────────────────────────────────────────────────────────────────┘
```

## `src/` — shared library surface

Every algorithm listed below is exposed as `extern "C"` functions
with a `crispasr_` prefix. The CLI, Python, Rust, and Dart bindings
all consume the same symbols.

| File | Role |
|---|---|
| `crispasr_c_api.cpp` | The C-ABI. Exports session open/close/transcribe, VAD, diarize, LID, alignment, cache, registry — everything a wrapper needs. |
| `crispasr_vad.{h,cpp}` | Silero VAD slicing + whisper-style stitching with timestamp remapping. Used by `crispasr_session_transcribe_vad`. |
| `crispasr_diarize.{h,cpp}` | Four diarizers: energy (stereo), xcorr (stereo, TDOA), vad-turns (mono, timing), pyannote (mono, GGUF). |
| `crispasr_lid.{h,cpp}` | whisper-tiny + silero-native language ID with process-wide whisper-context cache. |
| `crispasr_aligner.{h,cpp}` | canary-CTC + Qwen3-ForcedAligner forced alignment behind one entry point; filename-based dispatch. |
| `crispasr_cache.{h,cpp}` | WinHTTP / curl / wget download into `~/.cache/crispasr/`; zombie-file detection. |
| `crispasr_model_registry.{h,cpp}` | Backend → canonical GGUF URL table; fuzzy filename lookup for "did you mean …?" hints. |
| `whisper_params.h` | Shared params struct (extracted from cli.cpp, extended). |

## `examples/cli/` — presentation + policy

| File | Role |
|---|---|
| `cli.cpp` | crispasr entry point, extended with `--backend` dispatch branch. |
| `crispasr_backend.{h,cpp}` | `CrispasrBackend` abstract class, capability bitmask, factory, GGUF auto-detect. |
| `crispasr_backend_{parakeet,canary,cohere,granite,granite_nle,voxtral,voxtral4b,qwen3,fastconformer_ctc,wav2vec2,glm_asr,kyutai_stt,firered_asr,moonshine,moonshine_streaming,omniasr,gemma4_e2b,mimo_asr,vibevoice,qwen3_tts,orpheus,kokoro,chatterbox,m2m100}.cpp` | Per-backend thin wrapper over each model's C API. ASR backends emit `crispasr_segment`s; TTS backends (`vibevoice`, `qwen3_tts`, `orpheus`, `kokoro`, `chatterbox`) implement `synthesize(text)` instead and write 24 kHz mono WAV via `--tts-output`; the translation backend (`m2m100`) implements `translate_text(text, src, tgt)` and writes UTF-8 to stdout. |
| `crispasr_output.{h,cpp}` | TXT / SRT / VTT / CSV / JSON / LRC writers on `crispasr_segment`. |
| `crispasr_vad_cli.{h,cpp}` | Delegates to `src/crispasr_vad`; adds auto-download for the Silero GGUF. |
| `crispasr_lid_cli.{h,cpp}` | Delegates to `src/crispasr_lid`; adds auto-download + sherpa-ONNX subprocess fallback. |
| `crispasr_diarize_cli.{h,cpp}` | Delegates to `src/crispasr_diarize`; adds sherpa subprocess fallback + pyannote GGUF auto-download. |
| `crispasr_model_mgr_cli.{h,cpp}` | Delegates to `src/crispasr_model_registry`; adds "Download now? [Y/n]" prompt on TTY. |
| `crispasr_aligner_cli.{h,cpp}` | Adapter converting `CrispasrAlignedWord` → the CLI's `crispasr_word` shape. |
| `crispasr_server.cpp` | HTTP server for the persistent-model mode + OpenAI-compatible endpoints. |
| `crispasr_llm_pipeline.h` | Templated audio-LLM pipeline (mel → encoder → prompt → KV decode). |
| `crispasr_run.cpp` | Top-level pipeline dispatch: resolve → detect → load → slice → transcribe → write. |

## `src/core/` — the shared model primitives

Duplicated scaffolding is bundled in a single static library,
`crispasr-core`, linked into every non-whisper model target.

| Header | Replaces | Consumers |
|---|---|---|
| `core/mel.{h,cpp}` | 7× copy-pasted STFT + mel filterbank + log + norm | parakeet, canary, canary_ctc, cohere, voxtral, voxtral4b, qwen3 |
| `core/ffn.h` | 4× inline SwiGLU blocks | qwen3, voxtral, voxtral4b, granite |
| `core/attention.h` | Llama-style self-attention with NEOX RoPE + GQA + flash-attn | voxtral, granite (via `core_granite_llm`) |
| `core/gguf_loader.{h,cpp}` | 8× identical two-pass GGUF load + mmap + tensor-map build | all non-whisper models |
| `core/fft.h` | Radix-2 Cooley-Tukey FFT (4× duplicated) | granite_speech, granite_nle (kokoro/mimo can adopt) |
| `core/cpu_ops.h` | CPU LayerNorm + matmul fallbacks (when no GPU sched is available) | granite_speech, granite_nle |
| `core/ctc.h` | `posterior_weighted_pool` + `greedy_decode_with_blank` | granite_nle (any aux-head/CTC variant can adopt) |
| `core/fastconformer.h` | NeMo-style FastConformer block (conv subsampling + MHA RPE) | parakeet, canary, canary_ctc |
| `core/conformer_ibm.h` | IBM Macaron Conformer block (FFN + Shaw RPE attn + conv module + FFN + Shaw lookup) — **sibling of `fastconformer.h`, intentionally not merged** | granite_speech, granite_nle |
| `core/granite_llm.h` | Granite-1B 40-block backbone (RMSNorm + GQA(16/4) flash-attn + RoPE + SwiGLU + µP residual scale); `is_causal` flag picks KV-cached prefill+decode (`core_attn::kv_self_attn`) vs non-causal flash (whole-sequence editing) | granite_speech, granite_nle |
| `core/qformer.h` | Windowed simplified Q-Former: pass A (LayerNorm + concat + linear + GELU) and per-window cross-attn + MLP cgraph builder | granite_nle (NAR-only — granite_speech uses a different full BLIP-2 Q-Former) |
| `core/bpe.h` | GPT-2 byte-level BPE encode + decode | granite_speech, granite_nle, voxtral, qwen3, glm-asr |
| `core/greedy_decode.h` | Autoregressive greedy decode loop with EOS handling | qwen3, voxtral, voxtral4b, granite, glm-asr |

`core_mel::Params` spans both algorithm clusters: the NeMo family
(`ln` + per-mel z-score + `(T, n_mels)` layout) and the HF/Whisper
family (`log10` + global clip normalization + `(n_mels, T)` layout),
with knobs for `LogGuard` (add-epsilon vs max-clip), `MatmulPrecision`
(`Float` vs `Double`), `FbLayout` (`MelsFreqs` vs `FreqsMels`),
`drop_last_frame` / `drop_first_frame_if_odd`, and `pad_to_T`.

`core_gguf::WeightLoad` owns the `ggml_context`, the
`ggml_backend_buffer_t`, and the `std::map<std::string, ggml_tensor*>`
in one struct that models `std::move()` into their own state. The
mmap path has a `pread` / `fseek` fallback for filesystems that don't
support mmap.

## Whisper is the reference implementation

`src/crispasr` is **intentionally not migrated** to `src/core/` (yet)
— it's (for the time being) the battle-tested reference and the
`crispasr -m ggml-base.en.bin …` code path is byte-identical to
upstream `whisper.cpp`. This guarantee is a test gate: every
CrispASR commit that touches the CLI is checked against it.

## Regression discipline

Every `src/core/` migration commit includes a `md5sum`-level
regression test against `samples/jfk.wav`:

- **mel extraction**: bit-identical transcript + SRT on parakeet,
  canary, canary_ctc, voxtral, voxtral4b, qwen3. Cohere transcript
  is bit-identical but a single SRT boundary shifts by 80 ms due to
  the CBLAS → manual-loop matmul accumulator reorder.
- **ffn extraction**: bit-identical on qwen3, voxtral, voxtral4b,
  granite.
- **gguf_loader extraction**: bit-identical on all 8 non-whisper
  models.
- **attention extraction**: bit-identical on voxtral (only consumer
  so far).

## Backend internals

> **Note:** the snapshot below was last hand-edited in early 2026 and
> is not regenerated from the registry — treat it as a sketch, not
> ground truth. The authoritative source for what's compiled in,
> per-backend GPU support, and current capability bits is
> `src/crispasr_model_registry.cpp` and the `capabilities()` returned
> by each adapter in `examples/cli/crispasr_backend_*.cpp`.

| Backend | Arch pattern | ggml graph | Flash attn | KV cache | GPU | Shared core modules |
|---|---|:-:|:-:|:-:|---|---|
| whisper | Enc-dec transformer | ✔ | ✔ | ✔ | CUDA / Metal / Vulkan | (upstream) |
| parakeet | FastConformer + TDT | ✔ | ✔ | partial | CPU | mel, fastconformer |
| canary | FastConformer + Transformer dec | ✔ | ✔ | ✔ | CUDA / Metal | mel, fastconformer |
| cohere | Conformer + Transformer dec | ✔ | ✔ | ✔ | CUDA / Metal | mel |
| granite | Conformer + Q-Former + LLM | ✔ | ✔ | ✔ | CPU | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| voxtral | Whisper enc + Mistral LLM | ✔ | ✔ | ✔ | CUDA / Metal | mel, kv_self_attn, encoder_self_attn, swiglu, greedy_decode, bpe |
| voxtral4b | RoPE enc + 3.4 B LLM | ✔ | ✔ | ✔ | CUDA / Metal | mel, kv_self_attn, encoder_self_attn, swiglu, greedy_decode, bpe |
| qwen3 | Whisper enc + Qwen3 LLM | ✔ | ✔ | ✔ | CUDA / Metal | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| fc-ctc | FastConformer + CTC | ✔ | ✔ | — | CPU | mel, fastconformer |
| wav2vec2 | CNN + Transformer + CTC | ✔ | — | — | CUDA / Metal | gguf_loader |
| glm-asr | Whisper enc + Llama LLM | ✔ | ✔ | ✔ | CPU | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| kyutai-stt | Mimi codec + causal LM | ✔ | ✔ | ✔ | CPU | gguf_loader |
| firered-asr | Conformer + CTC + beam dec | ✔ | ✔ | ✔ | CPU | mel, gguf_loader |
| moonshine | Conv + 6L enc-dec | ✔ | ✔ | ✔ | CPU | (vendored) |
| moonshine-streaming | Sliding-window enc + dec | ✔ | ✔ | ✔ | CPU | (vendored) |
| omniasr | wav2vec2 enc + CTC / LLM | ✔ | ✔ | CTC: — / LLM: ✔ | CPU | gguf_loader, kv_self_attn, swiglu |
| gemma4-e2b | Conformer enc + Gemma4 LLM | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| mimo-asr | wav2vec2 enc + Qwen2 LM | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| vibevoice | σ-VAE + Qwen2 (ASR) / TTS LM (synth) | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader |
| kokoro | StyleTTS2 BERT + ProsodyPredictor + iSTFTNet | ✔ | — | — | CPU | gguf_loader, fft, ffn |
| qwen3-tts | Qwen3 talker + 12 Hz codec + code-predictor | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| orpheus | Llama-3.2 talker + SNAC RVQ codec | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| chatterbox | T3 (Llama / GPT-2) + S3Gen (Conformer + UNet1D CFM + HiFTGen) | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu, fft |
| m2m100 | facebook/m2m100 12L+12L transformer (text-to-text translation) | ✔ | — | ✔ (cross-attn) | CUDA / Metal | gguf_loader, kv_self_attn |

### Architecture families

- **Feedforward CTC** (wav2vec2, omniasr-CTC, fc-ctc, firered-asr):
  No decoder, no KV cache. Fastest. No native punctuation.
- **Encoder-decoder** (whisper, canary, cohere, moonshine,
  moonshine-streaming): cross-attention KV cache, autoregressive
  text decoder.
- **Audio-LLM** (granite, voxtral, voxtral4b, qwen3, glm-asr,
  omniasr-LLM, gemma4-e2b, mimo-asr, vibevoice): audio features
  injected into LLM embedding space, KV-cached autoregressive
  decoding.
- **Transducer** (parakeet): LSTM predictor + joint network,
  frame-synchronous TDT decoding.
- **Codec + LM** (kyutai-stt): neural audio codec (RVQ) →
  token-based LM.
- **TTS — codec / vocoder pipeline**:
  - **Discrete-token codec + vocoder** (qwen3-tts, orpheus): talker
    LM emits codec tokens; a separate decoder GGUF (12 Hz codec /
    SNAC RVQ) renders the audio. Two-GGUF runtime.
  - **Flow-matching mel + iSTFT vocoder** (chatterbox / chatterbox-
    turbo / kartoffelbox-turbo / lahgtna-chatterbox): T3 emits speech
    tokens; S3Gen runs an UpsampleConformerEncoder + UNet1D CFM
    (10-step Euler for base / 2-step meanflow for turbo) producing a
    mel-spectrogram, then HiFTGenerator (conv chains + Snake +
    iSTFT) renders 24 kHz audio. Two-GGUF runtime.
  - **Realtime σ-VAE** (vibevoice in TTS mode): 4L base LM + 20L TTS
    LM + DPM-Solver++ + σ-VAE decoder.
  - **StyleTTS2 / iSTFTNet** (kokoro): BERT + ProsodyPredictor
    + iSTFTNet decoder, single-shot (no AR).
- **Text-to-text translation** (m2m100): SentencePiece BPE
  + transformer encoder + transformer decoder (with cross-attn
  KV cache) + greedy decode. Source/target language codes prefix
  the encoder/decoder input streams. No audio path; driven by
  `--text "..." -sl <src> -tl <tgt>`.

### Optimization opportunities

- **Beam search** for all encoder-decoder and Audio-LLM backends —
  PLAN #63 added it for several LLM backends; whisper + firered-asr
  always had it.
- **Fused QKV** (single matmul for Q / K / V projections) — used in
  CrispEmbed, applicable to all attention layers; landed for
  qwen3-tts talker (Q8_0/Q4_K-skipped) under env flag
  `QWEN3_TTS_FUSED_QKV`.
- **Temperature sampling** for the few backends that don't have it
  (glm-asr, kyutai-stt, firered-asr, moonshine, omniasr-LLM) via
  `core_greedy_decode`.
- **GPU offload** for the still-CPU-only backends — needs
  `ggml_backend_sched` with GPU primary.
