# CrispASR

**One C++ binary, twenty-four ASR backends + three TTS engines, zero Python dependencies.**

CrispASR started as a fork of [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and extends that base into a **unified speech engine** called `crispasr`, backed by full ggml C++ runtimes for major open-weights ASR *and* TTS architectures. One build, one binary, one consistent CLI — pick the backend at the command line or let CrispASR auto-detect it from your GGUF file. See [Text-to-Speech](#text-to-speech-tts) for the TTS side.

```console
$ crispasr -m ggml-base.en.bin          -f samples/jfk.wav                    # OpenAI Whisper
$ crispasr -m parakeet-tdt-0.6b.gguf    -f samples/jfk.wav                    # NVIDIA Parakeet
$ crispasr -m canary-1b-v2.gguf         -f samples/jfk.wav                    # NVIDIA Canary
$ crispasr -m voxtral-mini-3b-2507.gguf -f samples/jfk.wav                    # Mistral Voxtral
$ crispasr --backend qwen3 -m auto      -f samples/jfk.wav                    # -m auto downloads
$ crispasr --backend kokoro -m auto --tts "Hello world" --tts-output out.wav  # TTS
```

No Python. No PyTorch. No separate per-model binary. No `pip install`. Just one C++ binary and a GGUF file.

### Ecosystem

| Project | What it does |
|---|---|
| **[CrispASR](https://github.com/CrispStrobe/CrispASR)** | This repo — C++ speech recognition engine. 24 ASR backends + 4 TTS backends, CLI + HTTP server + C-ABI + Python/Rust/Dart bindings. |
| **[CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver)** | Cross-platform Flutter transcription app built on CrispASR. Desktop + mobile, all 10 backends, model browser with download queue, mic capture, SRT/VTT/JSON export, diarization, batch processing. Fully offline. |
| **[CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)** | Text embedding engine via ggml — same philosophy as CrispASR but for retrieval. 10 architectures (XLM-R, Qwen3-Embed, Gemma3, ModernBERT, ...), dense + sparse + ColBERT + reranking. 9.5x faster than ONNX on CPU, GPU via CUDA/Metal/Vulkan. Python/Rust/Dart bindings. |
| **[Susurrus](https://github.com/CrispStrobe/Susurrus)** | Python ASR GUI with 9 backends (faster-whisper, mlx-whisper, voxtral, insanely-fast-whisper, ...). The Python counterpart to CrispASR's C++ approach. |

---

## Table of contents

- [Supported backends](#supported-backends) — [ASR](#asr-backends) + [TTS](#text-to-speech-models) + [post-processing](#post-processing-models)
- [Feature matrix](#feature-matrix)
- [Install & build](#install--build)
- [Quick start — ASR](#quick-start)
- [**Text-to-Speech (TTS)**](#text-to-speech-tts) — Kokoro, Qwen3-TTS, VibeVoice
- [Streaming & live transcription](#streaming--live-transcription)
- [Server mode (HTTP API)](#server-mode-persistent-model-http-api)
- [CLI reference](#cli-reference)
- [Voice Activity Detection (VAD)](#voice-activity-detection-vad)
- [Word-level timestamps via CTC alignment](#word-level-timestamps-via-ctc-alignment)
- [Output formats](#output-formats)
- [Auto-download (`-m auto`)](#auto-download--m-auto)
- [Audio formats](#audio-formats)
- [Architecture](#architecture)
- [Adding a new backend](#adding-a-new-backend)
- [HOWTO Quantize](#howto-quantize)
- [Branch state & roadmap](#branch-state--roadmap)
- [GPU backend selection](#gpu-backend-selection)
- [Debugging & profiling](#debugging--profiling)
- [Credits](#credits)

---

## Supported backends

CrispASR ships **24 ASR backends** for transcription/translation and
**three TTS engines** for synthesis. Pick at the CLI with `--backend NAME`,
or omit it to let the binary auto-detect from the GGUF metadata. Jump
to the [TTS table](#text-to-speech-models) for the synthesis side.

### ASR backends

| Backend | Model | Architecture | Languages | License |
|---|---|---|---|---|
| **whisper** | [`ggml-base.en.bin`](https://huggingface.co/ggerganov/whisper.cpp/) and all OpenAI Whisper variants | Encoder-decoder transformer | 99 | MIT |
| **whisper** | [`distil-whisper/distil-large-v3`](https://huggingface.co/cstr/distil-large-v3-GGUF) | Distilled Whisper: 32L encoder + 2L decoder (6.3x faster, 513 MB Q5_0) | English | MIT |
| **parakeet** | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | FastConformer + TDT | 25 EU (auto-detect) | CC-BY-4.0 |
| **parakeet** | [`nvidia/parakeet-tdt_ctc-0.6b-ja`](https://huggingface.co/cstr/parakeet-tdt-0.6b-ja-GGUF) | FastConformer-TDT-CTC, xscaling, 80 mels | Japanese | CC-BY-4.0 |
| **canary** | [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) | FastConformer + Transformer decoder | 25 EU (explicit `-sl/-tl`) | CC-BY-4.0 |
| **cohere** | [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Conformer + Transformer | 13 | Apache-2.0 |
| **granite** | [`ibm-granite/granite-speech-{3.2-8b,3.3-2b,3.3-8b}`](https://huggingface.co/ibm-granite/granite-speech-3.3-2b), [`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | Conformer + BLIP-2 Q-Former + Granite LLM (μP) | en fr de es pt ja | Apache-2.0 |
| **granite-4.1** | [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | Same architecture as 4.0 (16-layer Conformer + Q-Former + Granite LLM); "2B" = full system. Encoder runs as a single ggml graph by default with per-layer Shaw RPE in attention (PLAN #16) — bit-near-identical to the per-op CPU loop, ~2.1× faster end-to-end on M1+Q4_K. `GRANITE_DISABLE_ENCODER_GRAPH=1` falls back to the CPU loop. | en fr de es pt ja | Apache-2.0 |
| **granite-4.1-plus** | [`ibm-granite/granite-speech-4.1-2b-plus`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | 4.1 + 2-layer encoder hidden-state concatenation (1024+1024=2048 projector input); emits punctuated / capitalised transcripts by default. `cat_hidden_layers` post-norm tensors are captured inline in the graph and `ggml_concat`-ed with the final encoder output, so PLUS rides the GPU path too (~2.5× end-to-end on M1+Q4_K). | en fr de es pt | Apache-2.0 |
| **granite-4.1-nar** | [`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar) | 4.1 with non-autoregressive decoder — single LLM forward over [audio, text+slots] + slot argmax decode (`is_causal=False` everywhere); 4-layer encoder hidden-state concatenation + posterior-pooled BPE auxiliary CTC head; bit-exact end-to-end on JFK via `crispasr-diff granite-nle`. Wired into the main CLI as `--backend granite-4.1-nar` (alias `granite-nar`). Encoder also runs as a single ggml graph (sibling builder with self-cond residual + snapshot concat + final CTC logits), ~3× faster end-to-end on M1+Q4_K. | en fr de es pt | Apache-2.0 |
| **fastconformer-ctc** | [`nvidia/stt_en_fastconformer_ctc_large`](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large) | FastConformer + CTC (NeMo family, all sizes) | en | CC-BY-4.0 |
| **voxtral** | [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Whisper encoder + Mistral 3B LLM, audio-token injection | 8 | Apache-2.0 |
| **voxtral4b** | [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Causal RoPE+SwiGLU encoder + 3.4B LLM with adaptive RMSNorm + sliding window | 13, realtime streaming | Apache-2.0 |
| **qwen3** | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Whisper-style audio encoder + Qwen3 0.6B LLM | 30 + 22 Chinese dialects | Apache-2.0 |
| **wav2vec2** | [`jonatasgrosman/wav2vec2-large-xlsr-53-english`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) | CNN + 24-layer transformer + CTC head (any Wav2Vec2ForCTC) | per-model (en, de, multilingual available) | Apache-2.0 |
| **wav2vec2** | [`facebook/data2vec-audio-base-960h`](https://huggingface.co/cstr/data2vec-audio-960h-GGUF) | Data2Vec Audio: 5-layer pos_conv, post-norm (79 MB Q4_K) | English | Apache-2.0 |
| **wav2vec2** | [`facebook/hubert-large-ls960-ft`](https://huggingface.co/cstr/hubert-large-ls960-ft-GGUF) | HuBERT Large: pre-norm, single pos_conv (212 MB Q4_K) | English | Apache-2.0 |
| **glm-asr** | [`zai-org/GLM-ASR-Nano-2512`](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) | Whisper encoder (partial RoPE) + 4-frame projector + Llama 1.5B LLM (GQA) | 17 (Mandarin, English, Cantonese, ...) | MIT |
| **kyutai-stt** | [`kyutai/stt-1b-en_fr`](https://huggingface.co/kyutai/stt-1b-en_fr) | Mimi neural audio codec (SEANet + 8L transformer + RVQ) + 16L causal LM (SwiGLU, RMSNorm) | en, fr | MIT |
| **firered-asr** | [`FireRedTeam/FireRedASR2-AED`](https://huggingface.co/FireRedTeam/FireRedASR2-AED) | Conformer encoder + CTC + beam search decoder; also LID (120 languages via FireRedLID GGUF) | Mandarin, English, 20+ Chinese dialects | Apache-2.0 |
| **moonshine** | [`UsefulSensors/moonshine-{tiny,base}`](https://huggingface.co/cstr/moonshine-base-GGUF) | Conv stem + 6L transformer encoder + 6L decoder (288–416d, partial RoPE, SiLU); multilingual variants (ja, ko, zh, ar, vi, uk) | English + 6 langs | MIT |
| **moonshine-streaming** | [`UsefulSensors/moonshine-streaming-{tiny,small,medium}`](https://huggingface.co/cstr/moonshine-streaming-tiny-GGUF) | Streaming ASR: raw-waveform frontend + sliding-window encoder + autoregressive decoder (34–245M, designed for edge/low-latency) | English | MIT |
| **gemma4-e2b** | [`google/gemma-4-E2B-it`](https://huggingface.co/cstr/gemma4-e2b-it-GGUF) | USM Conformer encoder (12L, 1024d) + Gemma4 LLM decoder (35L, 1536d, GQA, PLE); 128-bin log-mel, 30s max | 140+ langs | Apache-2.0 |
| **omniasr** | [`facebook/omniASR-CTC-{300M,1B}`](https://huggingface.co/cstr/omniASR-CTC-1B-GGUF) | wav2vec2-style CNN + 24–48L transformer + CTC head | **1600+** | Apache-2.0 |
| **omniasr** | [`omniASR-LLM-300M-v2`](https://huggingface.co/cstr/omniasr-llm-300m-v2-GGUF) | Same encoder + 12L LLaMA decoder (SwiGLU, RoPE); autoregressive, best quality | **1600+** | Apache-2.0 |
| **vibevoice** | [`microsoft/VibeVoice-ASR`](https://huggingface.co/cstr/vibevoice-asr-GGUF) | σ-VAE ConvNeXt encoders + Qwen2.5-7B decoder; timestamps, diarization, hotwords | 50+ | MIT |
| **mimo-asr** | [`XiaomiMiMo/MiMo-V2.5-ASR`](https://huggingface.co/cstr/mimo-asr-GGUF) | 6L input_local_transformer (1024d) + 36L Qwen2 LM (4096d, 32Q/8KV); 8-channel RVQ codes from separate MiMo-Audio-Tokenizer GGUF (`--codec-model`); JFK matches reference verbatim | Mandarin (Wu/Cantonese/Hokkien/Sichuanese dialects) + English + code-switching | MIT |

### Text-to-Speech models

Synthesis backends, driven by the `--tts` flag and a `--tts-output PATH.wav`.
See the dedicated [Text-to-Speech](#text-to-speech-tts) section below for
quick-start commands and engine selection guidance.

| Backend | Models | Architecture | Languages | License |
|---------|--------|-------------|-----------|---------|
| **vibevoice-tts** | [`VibeVoice-Realtime-0.5B`](https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF) | 4L base + 20L TTS LM + DPM-Solver++ + σ-VAE decoder; pre-computed voice presets | en | MIT |
| **vibevoice-tts** | [`VibeVoice-1.5B`](https://huggingface.co/cstr/vibevoice-1.5b-GGUF) | 28L Qwen2 LM + DPM-Solver++ + σ-VAE decoder; voice cloning from audio | en, zh | MIT |
| **qwen3-tts** | [`Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF), [`Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF), [`Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF) | Qwen3 talker LM + 12 Hz RVQ speech tokenizer; baked voice pack GGUF or runtime WAV + `--ref-text`. Pick `--backend qwen3-tts-1.7b-base` for the larger talker, or `--backend qwen3-tts-1.7b-voicedesign` to describe the voice in natural language via `--instruct`. | multilingual, per base model | Apache-2.0 |
| **kokoro** | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) + [`dida-80b/kokoro-german-hui-multispeaker-base`](https://huggingface.co/dida-80b/kokoro-german-hui-multispeaker-base) (German backbone) + [`kikiri-tts/kikiri-german-{victoria,martin}`](https://huggingface.co/kikiri-tts) (German voicepacks) | StyleTTS2 / iSTFTNet (BERT + ProsodyPredictor + iSTFTNet decoder, 82M params); per-voice GGUF; in-process libespeak-ng phonemizer with LRU cache; auto-routing for `-l de` swaps in the German-trained backbone + cascading voice fallback | en, es, fr, hi, it, ja, pt, zh native + de via Option 2b (PLAN §56) + others through espeak-ng with French/German voice fallback | Apache-2.0 (model + German backbone + kikiri voicepacks); HUI corpus CC0 |
| **orpheus** | [`Orpheus-3B-FT`](https://huggingface.co/cstr/orpheus-3b-base-GGUF) + [`SNAC 24 kHz`](https://huggingface.co/cstr/snac-24khz-GGUF) | Llama-3.2-3B-Instruct talker (28L, 3072 d) + SNAC RVQ codec (3 codebooks × 4096 @ 24 kHz); 8 baked English speakers (`tara`/`leah`/`leo`/...). Pick the speaker with `--voice <name>` and pass `--temperature 0.6` (engine_class.py default — greedy loops). Drop-in DE checkpoint variants shipped: `--backend kartoffel-orpheus-de-natural` ([`cstr/kartoffel-orpheus-3b-german-natural-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF), 19 speakers, ASR-roundtrip word-exact via parakeet-v3 -l de), `--backend kartoffel-orpheus-de-synthetic` ([`cstr/kartoffel-orpheus-3b-german-synthetic-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF), 4 speakers + 12 emotions + 5 outbursts via `{Speaker} - {Emotion}: {text}` syntax), `--backend lex-au-orpheus-de` (`lex-au/Orpheus-3b-German-FT-Q8_0.gguf`). | en (canopylabs); de (Kartoffel_Orpheus + lex-au) | llama3.2 community ("Built with Llama") for talker; MIT for SNAC |

### Post-processing models

Work with all backends.

| Model | Task | Architecture | Languages | License | HuggingFace |
|---|---|---|---|---|---|
| **FireRedPunc** | Punctuation restoration | BERT-base (12L, d=768), 5 classes | Chinese + English | Apache-2.0 | [`cstr/fireredpunc-GGUF`](https://huggingface.co/cstr/fireredpunc-GGUF) |
| **fullstop-punc** | Punctuation restoration | XLM-RoBERTa-large (24L, d=1024), 6 classes | EN, DE, FR, IT | MIT | [`cstr/fullstop-punc-multilang-GGUF`](https://huggingface.co/cstr/fullstop-punc-multilang-GGUF) |

All runtimes share ggml-based inference. The speech-LLM backends (**qwen3**, **voxtral**, **voxtral4b**, **granite**, **glm-asr**, **kyutai-stt**) inject audio encoder frames directly into an autoregressive language model's input embeddings, instead of using a dedicated CTC/transducer/seq2seq decoder. The **fastconformer-ctc** backend hosts the NeMo FastConformer-CTC standalone ASR family (small through xxlarge, same architecture as the canary aligner) with greedy CTC decoding.

## Feature matrix

Run `crispasr --list-backends` to see it live. Each backend declares capabilities at runtime; if you ask for a feature the selected backend does not support, CrispASR prints a warning and silently ignores the flag.

<!-- Generated from `crispasr --list-backends` + cross-cutting features. -->

| Feature | whisper | parakeet | canary | cohere | granite | granite&#8209;4.1 | voxtral | voxtral4b | qwen3 | fc&#8209;ctc | wav2vec2 | glm&#8209;asr | kyutai&#8209;stt | firered | moonshine | moon&#8209;stream | omniasr | omniasr&#8209;llm | vibevoice | gemma4&#8209;e2b | mimo&#8209;asr |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Native timestamps | ✔ | ✔ | ✔ | ✔ | | | | | | | | | ✔ | | | | | | | | |
| CTC timestamps | | | ✔ | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | ✔ |
| Word-level timing | ✔ | ✔ | ✔ | ✔ | `-am` | `-am` | `-am` | `-am` | `-am` | `-am` | `-am` | `-am` | ✔ | `-am` | `-am` | `-am` | `-am` | `-am` | | | `-am` |
| Per-token confidence | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | ✔ | ✔ | | | ✔ |
| Language auto-detect | ✔ | ✔ | LID | LID | LID | LID | LID | LID | ✔ | LID | LID | ✔ | LID | LID | LID | LID | LID | LID | LID | ✔ | LID |
| Speech translation | ✔ | | ✔ | | ✔ | ✔ | ✔ | | ✔ | | | | | | | | | | | | |
| Speaker diarization | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | all | all | all | all | all | all | all | all | all | all | | |
| Grammar (GBNF) | ✔ | | | | | | | | | | | | | | | | | | | | |
| Temperature sampling | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | ✔ | | ✔ | | ✔ | ✔ | ✔ | | |
| Beam search | ✔ | | | | ✔ | ✔ | ✔ | | ✔ | | | ✔ | ✔ | ✔ | ✔ | | ✔ | ✔ | | | |
| Flash attention | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | ✔ | ✔ | ✔ | | ✔ | ✔ | ✔ | | |
| Punctuation toggle | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | ✔ | | ✔ | | ✔ | ✔ | | | |
| Punc restoration | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | | |
| Source / target language | | | ✔ | | ✔ | ✔ | ✔ | | ✔ | | | | | | | | | | | | |
| Audio Q&A (`--ask`) | | | | | * | * | ✔ | | * | | | | | | | | | | | | |
| Streaming | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | |
| Auto-download (`-m auto`) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | ✔ |
| TTS | | | | | | | | | | | | | | | | | | | ✔ | | |

**Key:** ✔ = native/built-in, `-am` = via CTC forced aligner (`-am canary-ctc-aligner.gguf` or `-am qwen3-forced-aligner.gguf`), **LID** = via external language identification pre-step (`-l auto`), **all** = via `--diarize` post-step (not declared by backend but always available), **pp** = via `--punc-model` post-processor (FireRedPunc or fullstop-punc), * = experimental or partial support. granite-4.1 covers both the regular and `-plus` variants; granite-4.1-nar is a non-autoregressive variant with encoder+projector only (no LLM decode features).

**Speaker diarization** is available for all backends as a post-processing step via `--diarize`:
- `--diarize-method energy` / `xcorr` — stereo-only, no extra deps
- `--diarize-method pyannote` — native GGUF runtime (no Python, no sherpa-onnx). Pass `--sherpa-segment-model pyannote-v3-seg.gguf` for the pyannote v3 segmentation model. Falls back to sherpa subprocess for `.onnx` models.
- `--diarize-method sherpa` / `ecapa` — calls an externally-installed [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) subprocess with speaker embedding models.
- `--diarize-method vad-turns` — mono-friendly, assigns speaker labels at gap boundaries

**Language identification** for backends without native LID: `--lid-backend whisper` (default, 75 MB ggml-tiny.bin), `--lid-backend silero` (native GGUF, 16 MB, 95 languages), or `--lid-backend firered` (FireRedLID, 1.7 GB, 120 languages — Conformer encoder + Transformer decoder).

**Voice activity detection**: `--vad` uses the default Silero VAD (~885 KB, auto-downloaded). Each VAD segment is transcribed independently, producing separate SRT/VTT entries with correct timestamps. Use `--vad --split-on-punct` for best subtitle output. Four VAD backends: Silero (default), FireRedVAD (`-vm firered`, recommended), MarbleNet (`-vm marblenet`, 439 KB, 6 languages), Whisper-VAD-EncDec (`-vm whisper-vad`, experimental).

**Punctuation restoration** (`--punc-model`): CTC-based backends output lowercase without punctuation. Add `--punc-model fireredpunc-q8_0.gguf` (or `fullstop-punc-q4_k.gguf` for DE/FR/IT) to restore punctuation and capitalization. See the post-processing table above for model details. Also available via Python/Rust/Dart wrappers (`crispasr.PuncModel`).

<details>
<summary>Which backends produce punctuation natively?</summary>

| Backend | Punctuation | Capitalization | Notes |
|---|:-:|:-:|---|
| whisper | ✔ | ✔ | Full punctuation and casing |
| parakeet | ✔ | ✔ | |
| canary | ✔ | ✔ | |
| cohere | ✔ | ✔ | Toggleable via `--no-punctuation` |
| granite | ✔ | ✔ | LLM output |
| voxtral | ✔ | ✔ | LLM output |
| voxtral4b | ✔ | ✔ | LLM output |
| qwen3 | ✔ | ✔ | LLM output |
| glm-asr | ✔ | ✔ | LLM output |
| kyutai-stt | ✔ | ✔ | LLM output |
| moonshine | ✔ | ✔ | Encoder-decoder output |
| **fastconformer-ctc** | **no** | **no** | CTC — add `--punc-model` |
| **wav2vec2** | **no** | **no** | CTC — add `--punc-model` |
| **firered-asr** | **no** | **no** | CTC — add `--punc-model` |
| **omniasr** (CTC) | **no** | **no** | CTC — add `--punc-model` |
| **omniasr** (LLM) | ✔ | ✔ | Autoregressive decoder |

Other freely-licensed alternatives that could be added: [felflare/bert-restore-punctuation](https://huggingface.co/felflare/bert-restore-punctuation) (MIT, English, includes truecasing), [xashru/punctuation-restoration](https://github.com/xashru/punctuation-restoration) (Apache-2.0, 40+ languages, BiLSTM-CRF).

</details>

**Progressive subtitle output** (`--flush-after`): By default, non-whisper backends buffer all segments and print output at the end. For real-time subtitle consumption (PotPlayer, custom media players), use `--flush-after 1` to print each SRT entry to stdout immediately after its VAD segment is transcribed:

```bash
crispasr --backend parakeet -m parakeet.gguf --vad --flush-after 1 -osrt -f long_audio.wav
# SRT entries appear progressively as each segment finishes
```

**JSON output with language detection**: When using `-l auto -oj`, the JSON output includes detected language info:
```json
{
  "crispasr": {
    "backend": "cohere",
    "language": "en",
    "language_detected": "en",
    "language_confidence": 0.977,
    "language_source": "ecapa"
  },
  "transcription": [...]
}
```

### Which backend should I pick?

| Need | Pick |
|---|---|
| Battle-tested, all features exposed | **whisper** |
| Lowest English WER | **cohere** |
| **Fastest** (16x realtime on CPU) | **moonshine** (tiny), **fc-ctc** (10x) |
| Multilingual + word timestamps + fast | **parakeet** (2.9x RT) |
| Multilingual with **explicit language control** | **canary** |
| **Speech translation** (X→en or en→X) | **canary**, **voxtral**, **qwen3** |
| **30 languages + Chinese dialects** | **qwen3** |
| **1600+ languages** | **omniasr** (CTC or LLM) |
| **Realtime streaming ASR** (native incremental encoder, ~2× RT feed; sub-second-token target deferred to phase 2) | **voxtral4b** |
| Highest-quality offline speech-LLM | **voxtral** |
| Apache-licensed speech-LLM | **granite**, **voxtral**, **qwen3**, **omniasr-llm** |
| **Lightweight CTC-only** (fast, no decoder) | **wav2vec2**, **fc-ctc**, **data2vec**, **omniasr** |
| **Mandarin + Chinese dialects** | **firered-asr**, **qwen3**, **glm-asr** |

### Language detection for backends that don't do it natively

Cohere, canary, granite, voxtral and voxtral4b need an explicit
language code up front. If you don't know the language, pass
`-l auto` and crispasr runs an optional LID pre-step before the main
transcribe() call:

```bash
# Downloads ggml-tiny.bin (75 MB, 99 languages) on first use
crispasr --backend cohere -m $TC/cohere-transcribe-q5_0.gguf \
         -f unknown.wav -l auto
# crispasr[lid]: detected 'en' (p=0.977) via whisper-tiny
# crispasr: LID -> language = 'en' (whisper, p=0.977)
```

These LID providers are available:

- `--lid-backend whisper` (default) — uses a small multilingual ggml-*.bin model via the crispasr C API. Auto-downloads ~75 MB on first use. 99 languages.
- `--lid-backend silero` — native GGUF port of Silero's 95-language classifier. 16 MB F32, pure C++. Faster and smaller than whisper-tiny but slightly less accurate on long audio (>20s).
- `--lid-backend ecapa` — **recommended**: ECAPA-TDNN (Apache-2.0). Purpose-built for language ID. Very high accuracy on TTS benchmark. Two variants via `--lid-model`:
  - [`cstr/ecapa-lid-107-GGUF`](https://huggingface.co/cstr/ecapa-lid-107-GGUF) — VoxLingua107, 43 MB F16, 107 languages, ISO codes (en, de, ...). **Default.**
  - [`cstr/ecapa-lid-commonlanguage-GGUF`](https://huggingface.co/cstr/ecapa-lid-commonlanguage-GGUF) — CommonLanguage, 40 MB F16, 45 languages, full names (English, German, ...).
- `--lid-backend firered` — FireRedLID (Conformer encoder + Transformer decoder). Q4_K (544 MB), 120 languages including Chinese dialects. Slower but covers more languages.

These VAD providers are available:

- **Silero VAD** (default) — ~885 KB, auto-downloaded via `--vad`. Industry-standard, well-tested.
- **FireRedVAD** — DFSMN-based, 2.4 MB, F1=97.57%. Pass `--vad -vm firered` to auto-download. Recommended.
- **MarbleNet** — NVIDIA 1D separable CNN, 439 KB, 6 languages (EN/DE/FR/ES/RU/ZH). Pass `--vad -vm marblenet` to auto-download. Smallest model. ([`cstr/marblenet-vad-GGUF`](https://huggingface.co/cstr/marblenet-vad-GGUF))
- **Whisper-VAD-EncDec** *(experimental)* — Whisper-base encoder + TransformerDecoder head, 22 MB Q4_K. Trained on Japanese ASMR; may not generalise well to all domains. Pass `--vad -vm whisper-vad`. Slower than others (~1s vs ~50ms). ([`cstr/whisper-vad-encdec-asmr-GGUF`](https://huggingface.co/cstr/whisper-vad-encdec-asmr-GGUF))

Pass `--lid-backend off` to skip LID entirely.

---

## Install & build

### Prerequisites

- C++17 compiler (GCC 10+, Clang 12+, MSVC 19.30+)
- CMake 3.14+
- Optional: `libavformat`/`libavcodec`/`libavutil`/`libswresample` for Opus/M4A ingestion (`CRISPASR_FFMPEG=ON`)
- Optional: `libopenblas`/MKL/Accelerate — speeds up CPU-side matmuls for the Conformer-based encoders (parakeet, canary, cohere, granite, fastconformer-ctc). The ggml CPU backend picks up BLAS automatically when it's present at build time; no CrispASR configure flag needed.
- Optional: CUDA/Metal/Vulkan GPU backend — enabled via ggml's standard flags (`GGML_CUDA=ON`, `GGML_METAL=ON`, etc.). On CUDA you can set `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` to allow swapping to system RAM when VRAM is exhausted.
- `curl` or `wget` on `$PATH` if you want to use `-m auto` auto-download
- Optional: `sherpa-onnx` binaries on `$PATH` if you want `--diarize-method sherpa` with ONNX models

No Python, PyTorch, or pip required at runtime.

### Build

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target crispasr
```

The produced binary is `build/bin/crispasr`.

For the repo's default local preset, which already disables tests and ccache-related friction:

```bash
cmake --preset default
cmake --build build -j$(nproc) --target crispasr
```

### Windows (convenience scripts)

Two batch scripts handle the Windows build without requiring a pre-opened Developer Command Prompt. They use `vswhere.exe` to locate Visual Studio 2022 automatically, call `vcvars64.bat`, then drive CMake + Ninja.

#### `build-windows.bat` — CPU build

```cmd
build-windows.bat
```

Produces `build\bin\crispasr.exe`. Extra CMake flags can be appended:

```cmd
build-windows.bat -DCRISPASR_CURL=ON          :: enable libcurl fallback
build-windows.bat -DGGML_CUDA=ON             :: NVIDIA GPU (CUDA must be installed)
```

What it does:
1. Locates `vswhere.exe` under `%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\`
2. Finds the latest VS 2022 installation that includes the VC++ toolchain
3. Calls `vcvars64.bat` to initialize the 64-bit MSVC environment
4. Runs `cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release [extra flags]`
5. Builds the `crispasr` target → `build\bin\crispasr.exe`

#### `build-vulkan.bat` — Vulkan GPU build

```cmd
build-vulkan.bat
```

Produces `build-vulkan\bin\crispasr.exe` with the Vulkan compute backend enabled. In addition to the VS detection above, it:

1. Checks `%VULKAN_SDK%`. If unset, scans `C:\VulkanSDK\` for the newest installed version and sets `VULKAN_SDK` accordingly.
2. Adds `-DGGML_VULKAN=ON -DGGML_CUDA=OFF` so CUDA is not accidentally pulled in if the CUDA toolkit is also installed.
3. Writes the build into a separate `build-vulkan\` directory so it coexists with a CPU build.

Important:
- `build-windows.bat -DGGML_CUDA=ON` produces a CUDA build, not a Vulkan build.
- `--gpu-backend vulkan` only works if the binary was actually built with Vulkan support.
- On hybrid laptops, Vulkan device `0` may be the integrated GPU. Use `-dev N` to pin the discrete GPU if needed.

```cmd
:: Typical usage — VULKAN_SDK is picked up automatically
build-vulkan.bat

:: Override Vulkan SDK location explicitly
set VULKAN_SDK=C:\VulkanSDK\1.4.304.1
build-vulkan.bat

:: Run on Vulkan, pinned to GPU 1 (for example: NVIDIA on a hybrid laptop)
build-vulkan\bin\crispasr.exe --gpu-backend vulkan -dev 1 -m model.gguf -f audio.wav
```

Both scripts exit with a non-zero code and a `[ERROR]` message if any step fails (VS not found, CMake configure error, build error).

### With GPU backends

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON     # NVIDIA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON    # Apple Silicon
```

### With ffmpeg ingestion (Opus, M4A, WebM, …)

```bash
# Install ffmpeg dev libs first:
#   apt install libavformat-dev libavcodec-dev libavutil-dev libswresample-dev

cmake -B build-ffmpeg -DCMAKE_BUILD_TYPE=Release -DCRISPASR_FFMPEG=ON
cmake --build build-ffmpeg -j$(nproc) --target crispasr
```

> **Upstream bug warning.** `.m4a` / `.mp4` / `.webm` containers currently crash `crispasr`'s ffmpeg integration. For those formats, pre-convert to WAV:
> ```bash
> ffmpeg -i input.opus -ar 16000 -ac 1 -c:a pcm_s16le -y /tmp/audio.wav
> ```
> Bare-codec `.opus` files work fine with `CRISPASR_FFMPEG=ON`.

---

## Quick start

ASR examples below; for TTS see the [Text-to-Speech](#text-to-speech-tts) section.

### Whisper (historical path, byte-identical to upstream whisper.cpp)

```bash
# Download a whisper model (same as upstream whisper.cpp)
./models/download-ggml-model.sh base.en

./build/bin/crispasr -m models/ggml-base.en.bin -f samples/jfk.wav
# [00:00:00.000 --> 00:00:07.940]   And so my fellow Americans ask not what your country can do for you
# [00:00:07.940 --> 00:00:10.760]   ask what you can do for your country.
```

### Parakeet (multilingual, free word timestamps, fastest)

```bash
# Grab the quantized model (~467 MB)
curl -L -o parakeet.gguf \
    https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/parakeet-tdt-0.6b-v3-q4_k.gguf

./build/bin/crispasr -m parakeet.gguf -f samples/jfk.wav
# Auto-detected backend 'parakeet' from GGUF metadata.
# And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

# Word-level timestamps (one line per word)
./build/bin/crispasr -m parakeet.gguf -f samples/jfk.wav -ml 1
```

### Canary (explicit language, speech translation)

```bash
# Transcription (source == target)
./build/bin/crispasr --backend canary -m canary-1b-v2-q5_0.gguf -f audio.de.wav -sl de -tl de

# Translation (German speech → English text)
./build/bin/crispasr --backend canary -m canary-1b-v2-q5_0.gguf -f audio.de.wav -sl de -tl en

# ...or use the familiar crispasr flag:
./build/bin/crispasr --backend canary -m canary-1b-v2-q5_0.gguf -f audio.de.wav -l de --translate
```

### Voxtral (speech-LLM with auto-download)

```bash
# First run downloads ~2.5 GB to ~/.cache/crispasr/ via curl, then runs
./build/bin/crispasr --backend voxtral -m auto -f samples/jfk.wav

# Subsequent runs use the cached file
./build/bin/crispasr --backend voxtral -m auto -f samples/jfk.wav -l en
```

### Qwen3-ASR (30 languages + Chinese dialects)

```bash
./build/bin/crispasr --backend qwen3 -m auto -f audio.zh.wav
```

### MiMo-V2.5-ASR (Mandarin + dialects + English, 7.5B Qwen2 LM)

```bash
# Download the LM + audio tokenizer (the tokenizer is a separate model)
huggingface-cli download cstr/mimo-asr-GGUF mimo-asr-q4_k.gguf \
    --local-dir ~/.cache/crispasr
huggingface-cli download cstr/mimo-tokenizer-GGUF mimo-tokenizer-q4_k.gguf \
    --local-dir ~/.cache/crispasr

# Transcribe (auto-discovers tokenizer if it sits next to the LM)
./build/bin/crispasr \
    --backend mimo-asr \
    -m ~/.cache/crispasr/mimo-asr-q4_k.gguf \
    --codec-model ~/.cache/crispasr/mimo-tokenizer-q4_k.gguf \
    -f samples/jfk.wav
# Output: And so, my fellow Americans, ask not what your country can do
# for you. Ask what you can do for your country.
```

The 4.5 GB Q4_K is the recommended quant; F16 (14.9 GB) needs ~16 GB
RAM during inference. JFK matches the upstream Python
`MimoAudio.asr_sft` reference verbatim; performance on M1+Metal is
~0.3× realtime (Q4_K dequant per step is the bottleneck — F16 +
KV-reuse follow-ups are queued under PLAN #51a/b/c).

### Wav2Vec2 (lightweight CTC, any HF Wav2Vec2ForCTC model)

```bash
# English (Q4_K quantized, 212 MB — 6x smaller than F16)
curl -L -o wav2vec2-en-q4k.gguf \
    https://huggingface.co/cstr/wav2vec2-large-xlsr-53-english-GGUF/resolve/main/wav2vec2-xlsr-en-q4_k.gguf

./build/bin/crispasr -m wav2vec2-en-q4k.gguf -f samples/jfk.wav
# and so my fellow americans ask not what your country can do for you ask what you can do for your country

# German
curl -L -o wav2vec2-de-q4k.gguf \
    https://huggingface.co/cstr/wav2vec2-large-xlsr-53-german-GGUF/resolve/main/wav2vec2-xlsr-de-q4_k.gguf

./build/bin/crispasr -m wav2vec2-de-q4k.gguf -f audio.de.wav

# Convert any HuggingFace Wav2Vec2ForCTC model:
python models/convert-wav2vec2-to-gguf.py \
    --model-dir jonatasgrosman/wav2vec2-large-xlsr-53-german \
    --output wav2vec2-de.gguf --dtype f32
# Then optionally quantize:
./build/bin/crispasr-quantize wav2vec2-de.gguf wav2vec2-de-q4k.gguf q4_k
```

---

## Streaming & live transcription

```bash
# Pipe audio from ffmpeg, sox, or any tool that outputs raw PCM:
ffmpeg -i audio.wav -f s16le -ar 16000 -ac 1 - | \
    crispasr --stream -m model.gguf

# Live microphone transcription (auto-detects arecord/sox/ffmpeg):
crispasr --mic -m model.gguf

# Continuous live mode (prints each chunk as a new line, never stops):
crispasr --live -m model.gguf

# With progress monitor symbols (▶ processing, ✓ got text, · silence):
crispasr --live --monitor -m model.gguf

# Per-token confidence and alternative candidates:
crispasr -m model.gguf -f audio.wav --alt
```

Streaming works with all backends. The `--stream-step` (default 3s), `--stream-length` (default 10s), and `--stream-keep` (default 200ms overlap) flags control the sliding window.

---

## Text-to-Speech (TTS)

CrispASR ships **three open-weights TTS engines** behind the same
`crispasr` binary, each with a distinct voice/quality/footprint trade-off:

| Backend | Why pick it | Voice cloning | First-run download |
|---|---|---|---|
| **`kokoro`** | Smallest + fastest. 82M-param StyleTTS2-derived model. Multilingual via espeak-ng + native German backbone. | No (preset voice packs) | Manual `wget` (no `-m auto`) |
| **`qwen3-tts`** | Highest fidelity / strongest cloning. Speech-LLM (talker + code predictor + 12 Hz codec). | Yes (WAV + ref-text or baked voice GGUF) | ~1.3 GB via `-m auto` |
| **`vibevoice-tts`** | Lowest-latency streaming TTS, designed for realtime. | Preset voice packs (and a 1.5B-base WAV cloning path) | ~636 MB via `-m auto` |
| **`orpheus`** | Llama-3.2-3B talker + SNAC 24 kHz codec. 8 baked English speakers; expressive output. Greedy loops — pass `--temperature 0.6`. | Preset names via `--voice tara/leah/...` | ~3.5 GB via `-m auto` (talker Q8 + 26 MB SNAC) |

All three write 24 kHz mono WAV via `--tts-output`.

### Kokoro — multilingual, smallest

Kokoro is the 82M-param StyleTTS2-derived model. It does not currently
support `-m auto`; drop the GGUFs into a directory of your choice
(`~/.cache/crispasr/` works) and pass explicit paths.

```bash
# English — uses the official Kokoro-82M with the bundled af_heart voice.
./build/bin/crispasr \
    --backend kokoro \
    -m ~/.cache/crispasr/kokoro-82m-f16.gguf \
    --voice ~/.cache/crispasr/kokoro-voice-af_heart.gguf \
    --tts "Hello, how are you today?" -l en \
    --tts-output hello.wav

# German — pass `-l de` and the CLI auto-routes:
#   1. If kokoro-de-hui-base-f16.gguf sits next to kokoro-82m-f16.gguf,
#      the German-trained backbone (dida-80b/kokoro-german-hui-
#      multispeaker-base, Apache-2.0; HUI corpus CC0) is loaded instead
#      of the official one.
#   2. If --voice is omitted, a per-language fallback voice is picked
#      from <model_dir>/kokoro-voice-<name>.gguf in the cascade
#      df_victoria → df_eva → ff_siwis. Drop any of these into the
#      model directory; the first that exists wins.
./build/bin/crispasr \
    --backend kokoro \
    -m ~/.cache/crispasr/kokoro-82m-f16.gguf \
    --tts "Guten Tag, dies ist ein Test des deutschen Phonemizers." \
    -l de --tts-output guten_tag.wav
```

| Voice (German) | Source | License | Roundtrip on the test phrase (parakeet-v3, -l de) |
|---|---|---|---|
| `dm_martin` | [`kikiri-tts/kikiri-german-martin`](https://huggingface.co/kikiri-tts/kikiri-german-martin) | Apache-2.0 | "...Phonemizers." (perfect) |
| `df_victoria` | [`kikiri-tts/kikiri-german-victoria`](https://huggingface.co/kikiri-tts/kikiri-german-victoria) | Apache-2.0 | "...Tester des Deutschen Phonemizers." (1 word boundary err) |
| `dm_bernd` | Tundragoon (recovered from `r1di/kokoro-fastapi-german`'s Git LFS) | Apache-2.0 | "...Phonemetzers." (1 word boundary err) |
| `df_eva` | Tundragoon (recovered from `r1di/kokoro-fastapi-german`'s Git LFS) | Apache-2.0 | "...Phonemetzes." (1 word boundary err) |

All four voices clear the gate (peak ≥ 8000, RMS ≥ 1000) on the
dida-80b backbone — see `PLAN.md` §56 Option 2b for the full
methodology. The `crispasr_kokoro_resolve_*_abi` C ABI in `src/kokoro.h`
exposes the same routing logic to wrappers; from Python it surfaces
as `crispasr.kokoro_resolve_for_lang(model_path, lang)` returning a
`KokoroResolved(model_path, voice_path, voice_name, backbone_swapped)`
record.

### Qwen3-TTS — voice cloning, highest fidelity

Speech-LLM (talker + code predictor + 12 Hz codec). Needs both a
talker GGUF and a codec/tokenizer GGUF. With `-m auto` both are pulled
into `~/.cache/crispasr/` on first run (Q8_0 talker + F16 codec by
default).

```bash
# Auto-download, runtime WAV clone (~1.3 GB on first run):
./build/bin/crispasr \
    --backend qwen3-tts -m auto \
    --voice samples/qwen3_tts/clone.wav \
    --ref-text "Okay, yeah. I resent you, I love you, I respect you. But you know what - You blew it, and thanks to you." \
    --tts "Hello there" \
    --tts-output hello.wav

# F16 reference baseline (1.83 GB talker; strict-fidelity):
./build/bin/crispasr \
    --backend qwen3-tts \
    -m ~/.cache/crispasr/qwen3-tts-12hz-0.6b-base.gguf \
    --voice samples/qwen3_tts/clone.wav \
    --ref-text "Okay, yeah. I resent you, I love you, I respect you. But you know what - You blew it, and thanks to you." \
    --tts "Hello there" \
    --tts-output hello.wav

# Baked voice-pack GGUF (skips the WAV+ref-text step):
./build/bin/crispasr \
    --backend qwen3-tts -m auto \
    --voice /tmp/qwen3-tts-voice-pack.gguf \
    --tts "Hello there" \
    --tts-output hello.wav

# Larger 1.7B talker (~2.07 GB Q8_0 / ~3.86 GB F16; same ICL contract):
./build/bin/crispasr \
    --backend qwen3-tts-1.7b-base -m auto \
    --voice samples/qwen3_tts/clone.wav \
    --ref-text "Okay, yeah. I resent you, I love you, I respect you. But you know what - You blew it, and thanks to you." \
    --tts "Hello there" \
    --tts-output hello.wav

# VoiceDesign — describe the voice in natural language. No reference WAV,
# no preset speaker. 1.7B-only (~1.9 GB Q8_0). Pass --instruct instead of
# --voice; the codec bridge omits the speaker frame and the description
# is prepended to the prefill as a `<|im_start|>user\n…<|im_end|>\n`
# block.
./build/bin/crispasr \
    --backend qwen3-tts-1.7b-voicedesign -m auto \
    --instruct "A young female voice with a slight British accent, energetic, slightly fast paced" \
    --tts "Hello, I'm an excited engineer." \
    --tts-output hello.wav
```

Notes:
- When `--voice` points to a `.wav`, `--ref-text` is required. When it
  points to a `.gguf`, it is treated as a baked voice pack and
  `--ref-text` is ignored.
- With an explicit `-m`, the CLI auto-discovers the codec when
  `qwen3-tts-tokenizer-12hz.gguf` sits next to the talker; otherwise
  pass `--codec-model`.
- Quantization is **not** quality-equivalent across variants. The
  reference baseline is `f16` talker + `f16` codec. The recommended
  deployment quant is `q8_0` talker + `f16` codec — used by `-m auto`,
  ~986 MB, audibly indistinguishable from F16 on the test prompts in
  LEARNINGS.md. Lower-bit talker quants (`q6_k`, `q5_k`, `q4_k`) drift
  noticeably in strict tensor diffs. Quantizing the codec hurts
  earlier than quantizing the talker — keep
  `qwen3-tts-tokenizer-12hz.gguf` at `f16`.

### VibeVoice — realtime streaming TTS

Lowest-latency TTS engine. Uses `--voice` for its voice prompt or
preset; the realtime `0.5B` flow is typically driven by a voice GGUF.

```bash
# First run downloads ~636 MB to ~/.cache/crispasr/ (Q4_K talker + emma
# voice from cstr/vibevoice-realtime-0.5b-GGUF), then runs from cache.
./build/bin/crispasr \
    --backend vibevoice-tts -m auto \
    --tts "Hello, how are you today?" \
    --tts-output hello.wav
```

### Orpheus — Llama-3.2-3B + SNAC codec

Llama-3.2-3B-Instruct talker emitting `<custom_token_N>` LM tokens that
SNAC decodes to 24 kHz PCM. 8 baked English speakers (`tara`, `leah`,
`jess`, `leo`, `dan`, `mia`, `zac`, `zoe`). The talker GGUF and the
SNAC codec live in two separate HF repos and download together via
`-m auto`.

```bash
# First run pulls ~3.5 GB (Q8_0 talker) + 26 MB (SNAC codec) into
# ~/.cache/crispasr/.  --temperature 0.6 is the upstream
# engine_class.py default — DO NOT skip it. Greedy (--temperature 0)
# enters a 7-slot loop after a few super-frames and produces unusable
# audio.
./build/bin/crispasr \
    --backend orpheus -m auto \
    --voice tara --temperature 0.6 \
    --tts "Hello, my name is Tara." \
    --tts-output hello.wav
```

Drop-in DE checkpoint variants are shipped: pass
`--backend kartoffel-orpheus-de-natural` for a 19-speaker German
fine-tune trained on natural speech recordings,
`--backend kartoffel-orpheus-de-synthetic` for a 4-speaker variant
with explicit emotion + outburst control (`Martin - Sad: Oh, ich
bin so traurig.`), or `--backend lex-au-orpheus-de` for lex-au's
German Q8_0 mirror. All three reuse the same orpheus runtime +
SNAC codec.

### TTS GGUF downloads

[`cstr/vibevoice-realtime-0.5b-GGUF`](https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF) ·
[`cstr/vibevoice-1.5b-GGUF`](https://huggingface.co/cstr/vibevoice-1.5b-GGUF) ·
[`cstr/qwen3-tts-0.6b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF) ·
[`cstr/qwen3-tts-1.7b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF) ·
[`cstr/qwen3-tts-1.7b-voicedesign-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF) ·
[`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF) ·
[`cstr/orpheus-3b-base-GGUF`](https://huggingface.co/cstr/orpheus-3b-base-GGUF) ·
[`cstr/kartoffel-orpheus-3b-german-natural-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF) ·
[`cstr/kartoffel-orpheus-3b-german-synthetic-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF) ·
[`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF)

### qwen3-tts environment switches

Diagnostic / experimental knobs. Leave them unset for normal use — the
defaults reproduce the validated, end-to-end-tested code path.

| Variable | Default | Effect when set |
| --- | --- | --- |
| `QWEN3_TTS_MAX_FRAMES` | `1500` | Hard cap on AR decode steps. Short prompts that fail to sample `codec_eos` would otherwise run to the 1500-frame ceiling. |
| `QWEN3_TTS_O15` | unset | Pin code-predictor `Lk = cp_kv_max_ctx` and reuse one cached T=1 graph across AR steps 2..14 (saves ~7 ms/frame on Mac/Metal). End-to-end output matches the dynamic-Lk default; flag stays opt-in pending a clean speed A/B. |
| `QWEN3_TTS_FUSED_QKV` | unset | Fuse Q+K+V weights into one matmul per talker layer at load time (F16/F32 talker only; auto-skipped for Q8_0/Q4_K). Bit-identical to the unfused path on M1 Metal; speed effect is machine-dependent. |
| `QWEN3_TTS_BENCH` | unset | Print per-call build/alloc/compute/read timings for `talker_kv` and `code_pred_kv`. |
| `QWEN3_TTS_PROF` | unset | Per-op profiler (more granular than `BENCH`). |
| `QWEN3_TTS_CP_BACKEND` | unset | Pin the code predictor to a chosen backend. `cpu`, `cpu-f16`, `cpu-f32` keep its weights on the CPU backend — useful when isolating bugs to the talker vs. code-predictor or when comparing CPU and Metal end-to-end. |
| `QWEN3_TTS_DUMP_DIR` | unset | Write per-frame intermediate tensors into the named directory. Bulky; intended for diff-harness work (`tools/dump_reference.py --backend qwen3-tts`). |

---

## Server mode (persistent model, HTTP API)

```bash
# Start server with model loaded once
crispasr --server -m model.gguf --port 8080

# Transcribe via HTTP (model stays loaded between requests):
curl -F "file=@audio.wav" http://localhost:8080/inference
# Returns JSON: {"text": "...", "segments": [...], "backend": "parakeet", "duration": 11.0}

# Hot-swap to a different model at runtime:
curl -F "model=path/to/other-model.gguf" http://localhost:8080/load

# Check server status:
curl http://localhost:8080/health
# {"status": "ok", "backend": "parakeet"}

# List available backends:
curl http://localhost:8080/backends
# {"backends": ["whisper","parakeet","canary",...], "active": "parakeet"}
```

The server loads the model once at startup and keeps it in memory. Subsequent `/inference` requests reuse the loaded model with no reload overhead. Requests are mutex-serialized. Use `--host 0.0.0.0` to accept remote connections.

To require API keys, set the `CRISPASR_API_KEYS` env var (comma-separated). **Do not** pass keys as CLI arguments — they would be visible in `ps`/`top`. Protected endpoints accept either `Authorization: Bearer <key>` or `X-API-Key: <key>`. `/health` remains public for container health checks.

```bash
CRISPASR_API_KEYS=key-one,key-two crispasr --server -m model.gguf

curl -H "Authorization: Bearer key-one" \
  -F "file=@audio.wav" \
  http://localhost:8080/v1/audio/transcriptions
```

### Docker Compose

The repo includes a root-level [`docker-compose.yml`](docker-compose.yml) for running the persistent HTTP server against a mounted model directory.

```bash
cp .env.example .env
# Edit CRISPASR_MODEL to point at a file mounted under ./models

docker compose up --build

# Health check
curl http://localhost:8080/health

# OpenAI-compatible transcription API
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $CRISPASR_API_KEY" \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"
```

By default the compose stack:
- builds from `.devops/main.Dockerfile`
- mounts `./models` into `/models`
- stores auto-downloaded models in the Docker-managed `crispasr-cache` volume at `/cache`
- serves on `http://localhost:8080`

If you want `/cache` to be a host directory instead, replace the `crispasr-cache:/cache` volume with `./cache:/cache` and make it writable by the container user before startup:

```bash
mkdir -p cache models
sudo chown -R "$(id -u):$(id -g)" cache models
```

You can cap or raise build parallelism with `CRISPASR_BUILD_JOBS`:

```bash
docker compose build --build-arg CRISPASR_BUILD_JOBS=8
```

For CUDA builds, use the override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

### Prebuilt CUDA images — choosing a tag

We publish two CUDA tags on `ghcr.io/crispstrobe/crispasr`. Pick the one that matches your host driver:

| Tag | CUDA | Min NVIDIA driver | Supported arches | Notes |
|---|---|---|---|---|
| `main-cuda` | 13.0 | **R535+** (R580+ for full features) | sm_75…sm_120 incl. RTX 50xx (Blackwell) | Default. Pull this on modern hosts. |
| `main-cuda-12` | 12.4 | **R510+** | sm_75…sm_90 (RTX 20/30/40-series, Hopper) | Legacy compat — use on RHEL 7/8, older Ubuntu LTS, or any host that hasn't updated drivers in a while. RTX 50xx is **not** supported here. |

Quick check: `nvidia-smi` shows your driver version in the top-right. If it's R535 or higher, pull `main-cuda`. If it's R510–R534, pull `main-cuda-12`. If it's older than R510, update your driver — neither image will work.

```bash
docker pull ghcr.io/crispstrobe/crispasr:main-cuda      # modern hosts
docker pull ghcr.io/crispstrobe/crispasr:main-cuda-12   # legacy driver
```

### Hugging Face Space wrapper

There is also a Gradio-based Hugging Face Space wrapper under [`hf-space/`](hf-space/README.md). It starts the CrispASR HTTP server inside the container and provides a small browser UI on top of the OpenAI-compatible transcription endpoint.

Build it locally with:

```bash
docker build -f hf-space/Dockerfile -t crispasr-hf-space .
docker run --rm -p 7860:7860 -p 8080:8080 \
  -e CRISPASR_MODEL=/models/ggml-base.en.bin \
  -v "$PWD/models:/models" \
  crispasr-hf-space
```

The compose files default to local image tags (`crispasr-local:*`) so they don't depend on pulling a published registry image first.

You can override the loaded model and startup flags through `.env`:
- `CRISPASR_MODEL=/models/parakeet-tdt-0.6b-v2.gguf`
- `CRISPASR_BACKEND=parakeet`
- `CRISPASR_LANGUAGE=auto`
- `CRISPASR_AUTO_DOWNLOAD=1`
- `CRISPASR_CACHE_DIR=/cache`
- `CRISPASR_API_KEYS=key-one,key-two`
- `CRISPASR_EXTRA_ARGS=--no-punctuation`

The service is configured to avoid serving as root by default:
- `user: "${CRISPASR_UID:-1000}:${CRISPASR_GID:-1000}"`
- `security_opt: ["no-new-privileges:true"]`

### OpenAI-compatible API

The server exposes `POST /v1/audio/transcriptions`, a drop-in replacement for the [OpenAI Whisper API](https://platform.openai.com/docs/api-reference/audio/createTranscription). Any tool that speaks the OpenAI transcription protocol (LiteLLM, LangChain, custom clients) can point at CrispASR with zero code changes.

```bash
# Same curl syntax as the OpenAI API:
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $CRISPASR_API_KEY" \
  -F "file=@audio.wav" \
  -F "response_format=json"
# {"text": "And so, my fellow Americans, ask not what your country can do for you..."}

# Verbose JSON with per-segment timestamps (matches OpenAI's format):
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $CRISPASR_API_KEY" \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"
# {"task": "transcribe", "language": "en", "duration": 11.0, "text": "...", "segments": [...]}

# SRT subtitles:
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=srt"

# Plain text:
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=text"
```

**Supported form fields:**

| Field | Description |
|---|---|
| `file` | Audio file (required) |
| `model` | Ignored (uses the loaded model) |
| `language` | ISO-639-1 code (default: server's `-l` setting) |
| `prompt` | Initial prompt / context |
| `response_format` | `json` (default), `verbose_json`, `text`, `srt`, `vtt` |
| `temperature` | Sampling temperature (default: 0.0) |

`GET /v1/models` returns an OpenAI-compatible model list with the currently loaded model.

---

## CLI reference

`crispasr` extends upstream `crispasr`'s argument set with a handful of backend-dispatch flags. Every historical whisper flag still works — when you don't pass `--backend`, whisper is the default.

### Core

| Flag | Meaning |
|---|---|
| `-m FNAME`, `--model FNAME` | Path to a model file, or `auto` to download a default for the selected backend |
| `--backend NAME` | Force a specific backend. Default: auto-detected from GGUF metadata + filename heuristics |
| `-f FNAME`, `--file FNAME` | Input audio (can repeat; also accepts positional filenames) |
| `-t N`, `--threads N` | Thread count (default: `min(4, nproc)`) |
| `-l LANG`, `--language LANG` | ISO-639-1 code (default: `en`) |
| `--list-backends` | Print the capability matrix and exit |

### Output

| Flag | Output |
|---|---|
| `-otxt` | Plain text to `<audio>.txt` |
| `-osrt` | SubRip (SRT) to `<audio>.srt` |
| `-ovtt` | WebVTT to `<audio>.vtt` |
| `-ocsv` | CSV (start, end, text) |
| `-oj`, `-ojf` | JSON (compact or full with word/token arrays) |
| `-olrc` | LRC lyrics format |
| `-of FNAME` | Output file base (no extension) |
| `-np` | No prints (suppress stderr progress) |
| `-pc` | Color-code output by token confidence (where supported) |
| `--no-timestamps` | Plain text only, no timing |
| `-ml N` | Max chars per display segment. `0`=unlimited, `1`=per-word, `N`=split at word boundaries |
| `-sp`, `--split-on-punct` | Split subtitle lines at sentence-ending punctuation (`. ! ?`). Creates readable subtitles even for CTC models that produce long segments |

### Segmentation / chunking

| Flag | Meaning |
|---|---|
| `--vad` | Enable Silero VAD. Auto-downloads `ggml-silero-v5.1.2.bin` (~885 KB) to `~/.cache/crispasr/` on first use |
| `--vad-model FNAME` | Override the VAD model path (default: auto) |
| `-vt F` | VAD threshold (default 0.5) |
| `-vspd N` | VAD min speech duration (ms, default 250) |
| `-vsd N` | VAD min silence duration (ms, default 100) |
| `-ck N`, `--chunk-seconds N` | Fallback chunk size when VAD is off (default 30 s) |

### Sampling / decoding (whisper + LLM backends)

| Flag | Meaning |
|---|---|
| `-tp F`, `--temperature F` | Sampling temperature. `0` = pure argmax (default, bit-identical). `> 0` enables multinomial sampling for whisper, voxtral, voxtral4b, qwen3, granite |
| `-bs N`, `--beam-size N` | Beam search width (whisper only) |
| `-tpi F`, `--temperature-inc F` | Whisper temperature-fallback increment |
| `--grammar FNAME` | GBNF grammar file (whisper only, including `--backend whisper`) |
| `--grammar-rule NAME` | Top-level rule name in the grammar |
| `--prompt STR` | Initial prompt for whisper |

### Language detection (LID)

| Flag | Meaning |
|---|---|
| `-l auto`, `--detect-language` | Auto-detect the input language. Backends without native lang-detect (cohere, canary, granite, voxtral, voxtral4b) get it via the LID pre-step |
| `--lid-backend NAME` | LID provider: `whisper` (default), `silero` (95 langs, 16 MB), `ecapa` (107 or 45 langs, 40-43 MB), `firered` (120 langs, 544 MB), or `off` |
| `--lid-model FNAME` | Override the LID model path (default: auto-downloads `ggml-tiny.bin` ~75 MB on first use) |

### LLM-backend specific

| Flag | Meaning |
|---|---|
| `-am FNAME`, `--aligner-model FNAME` | CTC aligner GGUF for word-level timestamps |
| `-n N`, `--max-new-tokens N` | Max tokens the LLM may generate (default 512) |

### Multi-language / translation

| Flag | Meaning |
|---|---|
| `-sl LANG`, `--source-lang LANG` | Source language (canary) |
| `-tl LANG`, `--target-lang LANG` | Target language (canary; set different from `-sl` for X→Y translation) |
| `-tr`, `--translate` | Translate to English (whisper, canary) |
| `--no-punctuation` | Disable punctuation in the output. Native for cohere/canary, post-processed for everyone else |

### Threading / processors

| Flag | Meaning |
|---|---|
| `-t N`, `--threads N` | Threads per inference call (default `min(4, nproc)`) |
| `-p N`, `--processors N` | Run N parallel decoder states (whisper only — uses `whisper_full_parallel`) |
| `--no-gpu` / `--device N` | Disable GPU or pin to GPU N |

### Whisper-only flags

These work both with the historical default whisper code path AND with `--backend whisper`. The historical path retains a few extras unique to it (`-owts` karaoke, full-mode JSON DTW tokens, `-di` stereo diarize) — pass a `ggml-*.bin` model without `--backend` to get them.

`--diarize`, `-tdrz`/`--tinydiarize`, `--carry-initial-prompt`, `-dtw`, `-fa`/`-nfa`, `-suppress-regex`, `-suppress-nst`, and the full upstream `crispasr --help` list.

---

## Voice Activity Detection (VAD)

Every non-whisper backend uses the Silero VAD model to segment long audio into speech regions, **stitch them into a single contiguous buffer** (with 0.1s silence gaps, matching crispasr's internal approach), transcribe in one pass, and remap timestamps back to original-audio positions. This preserves cross-segment context and avoids boundary artifacts. Short VAD segments (< 3s) are auto-merged, and oversized segments are split at `--chunk-seconds` boundaries. Whisper handles VAD internally via `wparams.vad`.

```bash
# Just pass --vad — the model is auto-downloaded on first use
./build/bin/crispasr --backend parakeet -m parakeet.gguf -f long_audio.wav \
    --vad -osrt

# Or point at an existing GGUF
./build/bin/crispasr --backend parakeet -m parakeet.gguf -f long_audio.wav \
    --vad-model ~/models/ggml-silero-v5.1.2.bin -osrt
```

The cached model lives at `~/.cache/crispasr/ggml-silero-v5.1.2.bin` (~885 KB). If you don't provide `--vad`, CrispASR falls back to fixed 30-second chunking (configurable via `-ck`). Encoder cost is O(T²) in the frame count, so for multi-minute audio you really want VAD.

**Recommended for subtitles:**
```bash
crispasr --backend parakeet -m parakeet.gguf -f long_audio.wav --vad -osrt --split-on-punct
```

**Accurate subtitle timing:**

- **Best timing quality:** use **parakeet**. Its native TDT timestamps are more accurate and more natural than the forced-aligner fallback used by LLM backends.
- **Best default subtitle flags:** use `--vad --split-on-punct`. VAD segments at natural speech pauses, then CrispASR stitches/remaps timestamps back to the original timeline. This avoids the mid-sentence boundary problems of fixed 30-second chunking.
- **For backends without native timestamps** (`cohere`, `granite`, `voxtral`, `voxtral4b`, `qwen3`): use a CTC aligner together with `--vad`. Without VAD, leading silence can throw off sentence starts, especially for the qwen3 forced aligner.
- **If parakeet is too heavy for very long audio:** keep parakeet for timing quality, but cap memory use with fixed chunking:

```bash
./build/bin/crispasr --backend parakeet -m parakeet.gguf -f long_audio.wav \
    --chunk-seconds 180 -osrt --split-on-punct
```

In practice:
- `parakeet --vad -osrt --split-on-punct` is the best default for subtitle generation
- `cohere/canary/qwen3/... --vad -am <aligner.gguf> -osrt --split-on-punct` is the best fallback when you need another backend
- `--chunk-seconds N` is mainly a VRAM-control knob for long audio, not the preferred path for subtitle accuracy

---

## Word-level timestamps via CTC alignment

The LLM-based backends (qwen3, voxtral, voxtral4b, granite) don't emit timestamps natively. CrispASR supports a second-pass forced alignment via NVIDIA's canary-ctc-aligner — a 600M-param FastConformer + CTC head that works on any transcript + audio pair in 25+ European languages.

```bash
# Grab the aligner once (~400 MB)
curl -L -o canary-ctc-aligner.gguf \
    https://huggingface.co/cstr/canary-ctc-aligner-GGUF/resolve/main/canary-ctc-aligner-q5_0.gguf

# Now any LLM backend can produce word-level SRT output
./build/bin/crispasr --backend voxtral -m auto -f samples/jfk.wav \
    -am canary-ctc-aligner.gguf -osrt -ml 1
# [00:00:00.240 --> 00:00:00.640]  And
# [00:00:00.640 --> 00:00:00.880]  so,
# [00:00:00.880 --> 00:00:01.040]  my
# ...
```

Alignment granularity is one encoder frame, ~80 ms.

For subtitle output, prefer adding `--vad --split-on-punct`:

```bash
./build/bin/crispasr --backend cohere -m cohere.gguf -f talk.wav \
    -am canary-ctc-aligner.gguf --vad -osrt --split-on-punct
```

Notes:
- The aligner path is a fallback for backends that lack native timestamps.
- `qwen3-forced-aligner` is more sensitive to leading silence; `--vad` is strongly recommended with it.
- Parakeet remains the better choice when timestamp quality is the top priority.

---

## Output formats

CrispASR writes these formats side-by-side with the input audio (e.g. `jfk.wav` → `jfk.srt`, `jfk.vtt`, `jfk.json`). The JSON layout:

```json
{
  "crispasr": {
    "backend": "parakeet",
    "model":   "parakeet-tdt-0.6b-v3-q4_k.gguf",
    "language":"en"
  },
  "transcription": [
    {
      "timestamps": { "from": "00:00:00,240", "to": "00:00:10,880" },
      "offsets":    { "from": 240, "to": 10880 },
      "text":       "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
    }
  ]
}
```

Add `-ojf` (`--output-json-full`) to include per-word `words[]` and per-token `tokens[]` arrays when the backend populates them.

---

## Language bindings

All three wrappers are thin shells over the same C-ABI surface in
`src/crispasr_c_api.cpp`. Anything the CLI can do — transcribe, VAD,
diarize, LID, align, download — is one function call in every
language.

### Python

```python
from crispasr import (
    Session, diarize_segments, detect_language_pcm,
    align_words, cache_ensure_file, registry_lookup,
)

# Transcribe (any of the 24 ASR backends via one session object)
sess = Session("parakeet-tdt-0.6b-v3-q4_k.gguf")
segs = sess.transcribe_vad(pcm, "silero-v5.1.2.bin")  # stitched VAD pass

# Run each shared post-step standalone
lang = detect_language_pcm(pcm, model_path="ggml-tiny.bin")
diarize_segments(my_segs, pcm, method=DiarizeMethod.VAD_TURNS)
words = align_words("canary-ctc-aligner.gguf", "hello world", pcm)

# Auto-download a canonical model
entry = registry_lookup("parakeet")
path  = cache_ensure_file(entry.filename, entry.url)
```

### Rust

```rust
use crispasr::{
    Session, DiarizeMethod, DiarizeOptions, DiarizeSegment,
    LidMethod, detect_language_pcm, align_words,
    cache_ensure_file, registry_lookup,
};

let sess = Session::open("cohere-transcribe-q4_k.gguf", 4)?;
let segs = sess.transcribe_vad(&pcm, "silero-v5.1.2.bin", None)?;

let entry = registry_lookup("canary")?.unwrap();
let path  = cache_ensure_file(&entry.filename, &entry.url, false, None)?;
```

### Dart / Flutter

```dart
import 'package:crispasr/crispasr.dart' as crispasr;

final sess = crispasr.CrispasrSession.open(modelPath, backend: 'parakeet');
final segs = sess.transcribeVad(pcm, vadModelPath);

final lang = crispasr.detectLanguagePcm(
  pcm: pcm, method: crispasr.LidMethod.whisper, modelPath: tinyPath);
final words = crispasr.alignWords(
  alignerModel: ctcPath, transcript: text, pcm: pcm);
```

Reference application: **[CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver)** — a cross-platform Flutter desktop/mobile transcription app built on `package:crispasr`. Ships with model browser + downloader (all 10 backends + quants), drag-and-drop files, mic capture, SRT/VTT/TXT export, per-run performance metrics, and full en/de i18n. The v0.5.4 release uses the new `transcribeVad` path so every non-whisper backend benefits from stitched Silero VAD with zero CrisperWeaver-side work.

### Mobile

```bash
./build-ios.sh                    # iOS xcframework with Metal
./build-android.sh --vulkan       # Android NDK with Vulkan GPU
```

---

## Auto-download (`-m auto`)

When you pass `-m auto` (or `-m default`), CrispASR downloads the default quantized model for the selected backend into `~/.cache/crispasr/` on first use. The registry (kept in sync with `src/crispasr_model_registry.cpp`):

| Backend | Download | Approx size |
|---|---|---|
| whisper | `ggerganov/whisper.cpp/ggml-base.en.bin` | ~147 MB |
| parakeet | `cstr/parakeet-tdt-0.6b-v3-GGUF` | ~467 MB |
| canary | `cstr/canary-1b-v2-GGUF` | ~600 MB |
| voxtral | `cstr/voxtral-mini-3b-2507-GGUF` | ~2.5 GB |
| voxtral4b | `cstr/voxtral-mini-4b-realtime-GGUF` | ~3.3 GB |
| granite | `cstr/granite-speech-4.0-1b-GGUF` | ~2.94 GB |
| granite-4.1 | `cstr/granite-speech-4.1-2b-GGUF` | ~2.94 GB |
| granite-4.1-plus | `cstr/granite-speech-4.1-2b-plus-GGUF` | ~5.6 GB |
| granite-4.1-nar | `cstr/granite-speech-4.1-2b-nar-GGUF` | ~5.4 GB (F16) / ~3.2 GB (Q4_K) |
| qwen3 | `cstr/qwen3-asr-0.6b-GGUF` | ~500 MB |
| cohere | `cstr/cohere-transcribe-03-2026-GGUF` | ~550 MB |
| wav2vec2 | `cstr/wav2vec2-large-xlsr-53-english-GGUF` | ~212 MB |
| omniasr | `cstr/omniASR-CTC-1B-GGUF` | ~551 MB |
| omniasr-llm | `cstr/omniasr-llm-300m-v2-GGUF` | ~580 MB |
| hubert | `cstr/hubert-large-ls960-ft-GGUF` | ~200 MB |
| data2vec | `cstr/data2vec-audio-960h-GGUF` | ~60 MB |

Downloads go through `curl` (preferred) with a `wget` fallback — **no Python, no libcurl link dependency**. Works identically on Linux, macOS, and Windows 10+ where `curl` ships in the base system. Models are cached by filename; re-running is a single `stat()` check. The same registry + cache helpers are reachable from the wrappers via `crispasr.registry_lookup()` / `crispasr.cache_ensure_file()` so Python/Rust callers can drive `-m auto`-style resolution without re-implementing it.

---

## Audio formats

Every audio path goes through `read_audio_data()` inherited from upstream whisper.cpp. Two single-header decoders are embedded:

- **[miniaudio](https://miniaud.io/)** — WAV (any bit depth: 16/24/32 PCM, IEEE float, A-law, μ-law, ADPCM), FLAC, MP3
- **[stb_vorbis](https://github.com/nothings/stb)** — OGG Vorbis

Out of the box, CrispASR accepts **WAV / FLAC / MP3 / OGG Vorbis** at any bit depth and any sample rate (auto-resampled to 16 kHz), mono or stereo (auto-mixed to mono).

| Format | Default build | `CRISPASR_FFMPEG=ON` |
|---|:---:|:---:|
| WAV / FLAC / MP3 / OGG | ✔ | ✔ |
| `.opus` | ✗ | ✔ |
| `.m4a` / `.mp4` / `.webm` | ✗ | ⚠ upstream crash, pre-convert |
| `.aiff` / `.wma` / raw PCM | ✗ | pre-convert |

For anything in the bottom half, the reliable path is `ffmpeg -i in.X -ar 16000 -ac 1 -c:a pcm_s16le out.wav` then pass the WAV.

---

## Architecture

CrispASR is structured around three layers on top of crispasr. The
split between `src/` (library) and `examples/cli/` (presentation) is
deliberate: **every algorithm** — VAD, diarization, LID, CTC alignment,
HF download/cache, model registry — lives in `src/` behind a stable
C-ABI (`src/crispasr_c_api.cpp`), and every consumer (CLI, Dart, Python,
Rust) reaches it through the same symbols. The CLI keeps only
presentation + UX policy.

```
┌───────────────────────────────────────────────────────────────────┐
│ examples/cli/cli.cpp (the crispasr binary)                        │
│   Parses CLI args, dispatches to backend when --backend   │
│   is set or GGUF arch is non-whisper; otherwise runs whisper_full │
│   unchanged                                                        │
├───────────────────────────────────────────────────────────────────┤
│ examples/cli/crispasr_*_cli.{h,cpp}                               │
│   Thin CLI shims for policy only — auto-download, TTY prompts,    │
│   sherpa-ONNX subprocess fallbacks. Delegate the algorithmic      │
│   work to the shared library below.                                │
├───────────────────────────────────────────────────────────────────┤
│ src/crispasr_c_api.cpp — C-ABI (shared with Dart / Python / Rust) │
│   crispasr_vad.{h,cpp}           Silero VAD + crispasr-style   │
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

### `src/` — shared library surface

Every algorithm listed below is exposed as `extern "C"` functions
with a `crispasr_` prefix. The CLI, Python, Rust, and Dart bindings
all consume the same symbols.

| File | Role |
|---|---|
| `crispasr_c_api.cpp` | The C-ABI. Exports session open/close/transcribe, VAD, diarize, LID, alignment, cache, registry — everything a wrapper needs. |
| `crispasr_vad.{h,cpp}` | Silero VAD slicing + crispasr-style stitching with timestamp remapping. Used by `crispasr_session_transcribe_vad`. |
| `crispasr_diarize.{h,cpp}` | Four diarizers: energy (stereo), xcorr (stereo, TDOA), vad-turns (mono, timing), pyannote (mono, GGUF). |
| `crispasr_lid.{h,cpp}` | whisper-tiny + silero-native language ID with process-wide whisper-context cache. |
| `crispasr_aligner.{h,cpp}` | canary-CTC + Qwen3-ForcedAligner forced alignment behind one entry point; filename-based dispatch. |
| `crispasr_cache.{h,cpp}` | WinHTTP / curl / wget download into `~/.cache/crispasr/`; zombie-file detection. |
| `crispasr_model_registry.{h,cpp}` | Backend → canonical GGUF URL table; fuzzy filename lookup for "did you mean …?" hints. |
| `whisper_params.h` | Shared params struct (extracted from cli.cpp, extended). |

### `examples/cli/` — presentation + policy

| File | Role |
|---|---|
| `cli.cpp` | crispasr entry point, extended with `--backend` dispatch branch. |
| `crispasr_backend.{h,cpp}` | `CrispasrBackend` abstract class, capability bitmask, factory, GGUF auto-detect. |
| `crispasr_backend_{parakeet,canary,cohere,granite,voxtral,voxtral4b,qwen3,fastconformer_ctc,wav2vec2,glm_asr,kyutai_stt,firered_asr,moonshine,omniasr,vibevoice}.cpp` | Per-backend thin wrapper over each model's C API. |
| `crispasr_output.{h,cpp}` | TXT / SRT / VTT / CSV / JSON / LRC writers on `crispasr_segment`. |
| `crispasr_vad_cli.{h,cpp}` | Delegates to `src/crispasr_vad`; adds auto-download for the Silero GGUF. |
| `crispasr_lid_cli.{h,cpp}` | Delegates to `src/crispasr_lid`; adds auto-download + sherpa-ONNX subprocess fallback. |
| `crispasr_diarize_cli.{h,cpp}` | Delegates to `src/crispasr_diarize`; adds sherpa subprocess fallback + pyannote GGUF auto-download. |
| `crispasr_model_mgr_cli.{h,cpp}` | Delegates to `src/crispasr_model_registry`; adds "Download now? [Y/n]" prompt on TTY. |
| `crispasr_aligner_cli.{h,cpp}` | Adapter converting `CrispasrAlignedWord` → the CLI's `crispasr_word` shape. |
| `crispasr_server.cpp` | HTTP server for the persistent-model mode + OpenAI-compatible endpoints. |
| `crispasr_llm_pipeline.h` | Templated audio-LLM pipeline (mel → encoder → prompt → KV decode). |
| `crispasr_run.cpp` | Top-level pipeline dispatch: resolve → detect → load → slice → transcribe → write. |

### `src/core/` — the shared model primitives

Duplicated scaffolding is bundled in a single static library, `crispasr-core`, linked into every non-whisper model target.

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

`core_mel::Params` spans both algorithm clusters: the NeMo family (`ln` + per-mel z-score + `(T, n_mels)` layout) and the HF/Whisper family (`log10` + global clip normalization + `(n_mels, T)` layout), with knobs for `LogGuard` (add-epsilon vs max-clip), `MatmulPrecision` (`Float` vs `Double`), `FbLayout` (`MelsFreqs` vs `FreqsMels`), `drop_last_frame` / `drop_first_frame_if_odd`, and `pad_to_T`.

`core_gguf::WeightLoad` owns the `ggml_context`, the `ggml_backend_buffer_t`, and the `std::map<std::string, ggml_tensor*>` in one struct that models `std::move()` into their own state. The mmap path has a `pread`/`fseek` fallback for filesystems that don't support mmap.

### Whisper is the reference implementation

`src/crispasr` is **intentionally not migrated** to `src/core/` (yet) — it's (for the time being) the battle-tested reference and the `crispasr -m ggml-base.en.bin …` code path is byte-identical to upstream `crispasr`. This guarantee is a test gate: every CrispASR commit that touches the CLI is checked against it.

### Regression discipline

Every `src/core/` migration commit includes a `md5sum`-level regression test against `samples/jfk.wav`:

- **mel extraction**: bit-identical transcript + SRT on parakeet, canary, canary_ctc, voxtral, voxtral4b, qwen3. Cohere transcript is bit-identical but a single SRT boundary shifts by 80 ms due to the CBLAS→manual-loop matmul accumulator reorder.
- **ffn extraction**: bit-identical on qwen3, voxtral, voxtral4b, granite.
- **gguf_loader extraction**: bit-identical on all 8 non-whisper models.
- **attention extraction**: bit-identical on voxtral (only consumer so far).

### Backend internals

<details>
<summary>Optimization and graph survey (all backends)</summary>

| Backend | Arch pattern | ggml graph | Flash attn | KV cache | GPU | Shared core modules |
|---|---|:-:|:-:|:-:|:-:|---|
| whisper | Enc-dec transformer | ✔ | ✔ | ✔ | CUDA/Metal | (upstream) |
| parakeet | FastConformer + TDT | ✔ | ✔ | partial | CPU | mel, fastconformer |
| canary | FastConformer + Transformer dec | ✔ | ✔ | ✔ | CUDA/Metal | mel, fastconformer |
| cohere | Conformer + Transformer dec | ✔ | ✔ | ✔ | CUDA/Metal | mel |
| granite | Conformer + Q-Former + LLM | ✔ | ✔ | ✔ | CPU | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| voxtral | Whisper enc + Mistral LLM | ✔ | ✔ | ✔ | CUDA/Metal | mel, kv_self_attn, encoder_self_attn, swiglu, greedy_decode, bpe |
| voxtral4b | RoPE enc + 3.4B LLM | ✔ | ✔ | ✔ | CPU | mel, kv_self_attn, encoder_self_attn, swiglu, greedy_decode, bpe |
| qwen3 | Whisper enc + Qwen3 LLM | ✔ | ✔ | ✔ | CUDA/Metal | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| fc-ctc | FastConformer + CTC | ✔ | ✔ | — | CPU | mel, fastconformer |
| wav2vec2 | CNN + Transformer + CTC | ✔ | — | — | CUDA/Metal | gguf_loader |
| glm-asr | Whisper enc + Llama LLM | ✔ | ✔ | ✔ | CPU | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| kyutai-stt | Mimi codec + causal LM | ✔ | ✔ | ✔ | CPU | gguf_loader |
| firered-asr | Conformer + CTC + beam dec | ✔ | ✔ | ✔ | CPU | mel, gguf_loader |
| moonshine | Conv + 6L enc-dec | ✔ | ✔ | ✔ | CPU | (vendored) |
| moonshine-streaming | Sliding-window enc + dec | ✔ | ✔ | ✔ | CPU | (vendored) |
| omniasr | wav2vec2 enc + CTC/LLM | ✔ | ✔ | CTC:— LLM:✔ | CPU | gguf_loader, kv_self_attn, swiglu |
| vibevoice | σ-VAE + Qwen2 7B | ✔ | ✔ | ✔ | CUDA/Metal | gguf_loader |

**Architecture families:**
- **Feedforward CTC** (wav2vec2, omniasr-CTC, fc-ctc, firered-asr): No decoder, no KV cache. Fastest. No native punctuation.
- **Encoder-decoder** (whisper, canary, cohere, moonshine, moonshine-streaming): Cross-attention KV cache, autoregressive text decoder.
- **Audio-LLM** (granite, voxtral, voxtral4b, qwen3, glm-asr, omniasr-LLM, vibevoice): Audio features injected into LLM embedding space, KV-cached autoregressive decoding.
- **Transducer** (parakeet): LSTM predictor + joint network, frame-synchronous TDT decoding.
- **Codec + LM** (kyutai-stt): Neural audio codec (RVQ) → token-based LM.

**Optimization opportunities:**
- **Beam search** could be added to all encoder-decoder and Audio-LLM backends (currently only whisper + firered-asr)
- **Fused QKV** (single matmul for Q/K/V projections) — used in CrispEmbed, applicable to all attention layers
- **Temperature sampling** could be added to glm-asr, kyutai-stt, firered-asr, moonshine, omniasr-LLM via `core_greedy_decode`
- **GPU offload** for CPU-only backends (parakeet, granite, voxtral4b, etc.) — needs ggml_backend_sched with GPU primary

</details>

---

## Adding a new backend

Adding a new ASR model to CrispASR is a focused exercise in five files. The worked examples to copy from are the existing `crispasr_backend_*.cpp` adapters.

> **‼️ `clang-format` MUST be v18 — never use v22 (Homebrew's default `clang-format` keg, or Xcode's bundled clang-format, both currently ship v22).** CI pins `clang-format-18` (`.github/workflows/lint.yml`); v22 silently re-wraps lines and fails the lint job with ~80+ violations across the project's C++ files. Anyone formatting with v22 will produce a commit that breaks CI even though the code "looks formatted" locally.
>
> **Use `./tools/format.sh`** — it locates clang-format-18 via Homebrew's `llvm@18` keg or apt's `clang-format-18` and refuses to run any other version. `./tools/format.sh` checks; `./tools/format.sh --fix` rewrites in place. The script's scope mirrors `lint.yml` exactly so local check ≡ CI check.
>
> **Install** clang-format-18:
> - macOS: `brew install llvm@18` (binary lands at `/opt/homebrew/opt/llvm@18/bin/clang-format`)
> - Ubuntu / Debian: `sudo apt install clang-format-18`
> - Cross-platform via pip: `pip install 'clang-format==18.1.8'` (puts `clang-format` v18 on `PATH`; works as long as your pip user-bin precedes Homebrew's bin)
>
> **Do NOT** put `/opt/homebrew/bin/clang-format` (currently v22) on `PATH` for this project; it will silently mis-format. The wrapper script defends against this by hardcoded version-check.

### 1. Land the model's C API in `src/yourmodel.{h,cpp}`

Following the established convention:

```c
struct yourmodel_context * yourmodel_init_from_file(const char * path, yourmodel_context_params p);
void                       yourmodel_free(struct yourmodel_context *);
char *                     yourmodel_transcribe(struct yourmodel_context *, const float * samples, int n);
```

Use `src/core/mel`, `src/core/ffn`, `src/core/attention`, and `src/core/gguf_loader` wherever they fit — they cover ~80% of the boilerplate.

### 2. Write the backend adapter

Create `examples/cli/crispasr_backend_yourmodel.cpp`:

```cpp
#include "crispasr_backend.h"
#include "whisper_params.h"
#include "yourmodel.h"

namespace {
class YourmodelBackend : public CrispasrBackend {
public:
    const char * name() const override { return "yourmodel"; }
    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | /* ... */;
    }
    bool init(const whisper_params & p) override { /* yourmodel_init_from_file(...) */ }
    std::vector<crispasr_segment> transcribe(
        const float * samples, int n, int64_t t_off,
        const whisper_params & p) override { /* call yourmodel_transcribe and return segments */ }
    void shutdown() override { /* yourmodel_free(...) */ }
private:
    yourmodel_context * ctx_ = nullptr;
};
} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_yourmodel_backend() {
    return std::make_unique<YourmodelBackend>();
}
```

### 3. Register with the factory

In `examples/cli/crispasr_backend.cpp`:

```cpp
std::unique_ptr<CrispasrBackend> crispasr_make_yourmodel_backend();
// ...
if (name == "yourmodel") return crispasr_make_yourmodel_backend();
// ...
std::vector<std::string> crispasr_list_backends() {
    return { ..., "yourmodel" };
}
```

Add the architecture string to `crispasr_detect_backend_from_gguf()` so `general.architecture` auto-detection works.

### 4. Wire into CMake

In `examples/cli/CMakeLists.txt`:

```cmake
add_executable(${TARGET}
    # ...
    crispasr_backend_yourmodel.cpp
)

target_link_libraries(${TARGET} PRIVATE
    # ...
    yourmodel_lib
)
```

### 5. Optional: add to the model registry

If your model has a canonical Q4_K HuggingFace release, add it to `crispasr_model_mgr.cpp`'s registry so `-m auto` works.

### Regression-test your backend

For ASR backends, the transcript is the regression target:

```bash
./build/bin/crispasr --backend yourmodel -m model.gguf -f samples/jfk.wav -np > before.txt
# ... make changes ...
cmake --build build --target crispasr
./build/bin/crispasr --backend yourmodel -m model.gguf -f samples/jfk.wav -np > after.txt
diff before.txt after.txt && echo BIT-IDENTICAL
```

For TTS backends the output is audio (not text), and diffusion samplers
are stochastic, so a `diff` of two runs won't compare. The regression
target is "audio cosine similarity vs the official model's output, with
the same Gaussian noise pinned in both runs":

```bash
# 1. Run the official PyTorch model with hooks that capture the
#    per-frame init noise plus the conditions / latents per frame
HF_HOME=/path/to/hf-cache python tools/run_official_vibevoice.py \
    --text "Hello, how are you today?" \
    --voice voices_pt/en-Emma_woman.pt \
    --output-wav /tmp/ref.wav \
    --output-dir /tmp/ref_dump

# 2. Run crispasr with the same noise pinned and per-frame dumps
VIBEVOICE_TTS_NOISE=/tmp/ref_dump/noise.bin \
VIBEVOICE_TTS_DUMP=/tmp/cpp_dump VIBEVOICE_TTS_DUMP_PERFRAME=1 \
./build/bin/crispasr --tts "Hello, how are you today?" \
    -m vibevoice-realtime-0.5b-tts-f16.gguf \
    --voice vibevoice-voice-emma.gguf \
    --tts-output /tmp/cpp.wav -ng

# 3. Audio cos at xcorr peak (accounts for any leading-silence trim)
python -c "import sys; sys.path.insert(0, 'tools'); \
    from _audio_diff import cos_report; \
    print(cos_report('/tmp/ref.wav', '/tmp/cpp.wav'))"
# OFFICIAL: 182400 samples = 7.60s  rms=0.0653
# AFTER_FIX: 171459 samples = 7.14s  rms=0.0672
# cos at zero shift  = 0.0027
# cos at xcorr peak  = 0.9991  lag=7741 samples = 322.5 ms
```

`cos at xcorr peak ≥ 0.999` is "essentially bit-exact modulo F16
quantization". A drop indicates a real divergence — pair the audio
diff with the per-frame stage diff (next section) to localise.

### Debug a new backend against PyTorch ground truth

Bit-identical regression against the previous C++ version proves the
change was neutral, but it doesn't tell you the C++ forward pass is
correct in the first place. For that, use the ground-truth tools:

```bash
# 1. Capture PyTorch reference activations at every named stage
python tools/dump_reference.py --backend voxtral \
    --model-dir /path/to/hf/voxtral-mini-3b-2507 \
    --audio samples/jfk.wav \
    --output /tmp/voxtral-ref.gguf

# 2. Compare your C++ forward pass against the reference, stage by stage
./build/bin/crispasr-diff voxtral \
    voxtral-mini-3b-2507-q4_k.gguf \
    /tmp/voxtral-ref.gguf \
    samples/jfk.wav
#
# [PASS] mel_spectrogram    shape=[128,3000]  cos_min=0.99998  max_abs=3e-5
# [PASS] projector_output   shape=[375,3072]  cos_min=0.99985  max_abs=4e-4
# summary: 2 pass, 0 fail, 0 skip (cos threshold 0.999)
```

The Python dumper uses PyTorch forward hooks to capture intermediate
activations (mel, per-encoder-layer output, projector, LLM block output,
logits, argmax) and writes them to a single **GGUF tensor archive**.
The C++ side loads the archive via `core_gguf::load_weights` and runs
the backend's public stage helpers (`*_compute_mel`, `*_run_encoder`,
etc.) to produce the same tensors, then the shared `crispasr_diff::Ref`
compares them with **cosine similarity per row**, **max-abs error**,
**RMS**, and — for logits — **top-1 argmax match rate**.

Adding a new backend to the dumper is a ~60-line file in
`tools/reference_backends/<name>.py` that registers PyTorch forward
hooks and returns a dict `{stage_name: ndarray}`. Worked examples:

- `tools/reference_backends/qwen3.py`, `voxtral.py`, `cohere.py`,
  `parakeet.py`, `gemma4.py`, `omniasr_llm.py`, `granite.py` —
  encoder-decoder / Audio-LLM ASR backends. Use the `_hooks.py`
  forward_hook helpers (`capture_modules`, `drop_hooks`, `finalize`).
- `tools/reference_backends/qwen3_tts.py`,
  `qwen3_tts_codec.py`, `qwen3_tts_spk.py`, `qwen3_tts_cenc.py` —
  TTS prefill / encoder backends. Use `capture_modules(..., first_call_only=True)`
  for hooks that fire once per stage (e.g. talker prefill called from
  inside `generate()`).
- `tools/reference_backends/vibevoice.py`, `vibevoice_tts.py` —
  VibeVoice σ-VAE encoder + TTS pipeline.

For **autoregressive / diffusion-sampler diffs** the per-stage capture
above isn't enough — bugs that appear only after several AR steps don't
show up in a frame-0 cos diff. Two extra helpers:

- `tools/reference_backends/_iter_capture.py` — companion to `_hooks.py`
  for "monkey-patch a sampler entry point and append one tensor per
  iteration to a `{stage: [...]}` dict". Used to capture
  `pos_cond / neg_cond / noise / v_cfg_step0 / latent` per frame inside
  `sample_speech_tokens` for vibevoice.
- `tools/_audio_diff.py` — sample-wise audio cos at zero shift, cos at
  the cross-correlation peak (so leading-silence trims and causal-padding
  offsets don't tank the score), spectral band-power table, and a
  one-call `cos_report(a_path, b_path)` for CLI use.

`tools/run_official_vibevoice.py` is the worked example combining all
three: it loads the upstream model, monkey-patches `sample_speech_tokens`
via `_iter_capture.patch_method`, captures `acoustic_embed` via a
standard forward_hook, and writes both `perframe_<stage>_f<NNN>.bin`
files (matching the C++ runtime's `VIBEVOICE_TTS_DUMP_PERFRAME=1`
output) and a `noise.bin` for the C++ side to replay.

---

## Regression matrix (all backends)

`tools/test-all-backends.py` is the pass/fail gate for the whole
backend matrix — sister to `tools/macbook-benchmark-all-backends.py`
which is the perf benchmark.

Each backend declares advertised capabilities (`transcribe`,
`json-output`, `stream`, `beam`, `best-of-n`, `temperature`,
`word-timestamps`, `punctuation`, `vad`, `lid`, `tts-roundtrip`).
For each capability, three tiers exist: `ignore` / `smoke` / `full`.
Profiles set defaults; per-capability flags override.

```bash
# Quick smoke (transcribe-only on the JFK reference)
python tools/test-all-backends.py

# Full feature matrix (every advertised capability)
python tools/test-all-backends.py --profile=feature

# Subset by name or capability
python tools/test-all-backends.py --backends parakeet,kyutai-stt
python tools/test-all-backends.py --capabilities stream,beam

# Override one capability tier
python tools/test-all-backends.py --vad=full --transcribe=full
```

### Cache modes — `--cache-mode={keep,ephemeral}`

Two operational profiles for how the script handles model files:

| Mode | When | Behavior |
|---|---|---|
| `keep` (default) | local dev, bandwidth-limited | Downloads persist across runs in `--models PATH`. |
| `ephemeral` | tight-disk runners (Kaggle, cloud), bandwidth-cheap | After each backend's tier tests complete, deletes files we downloaded *this run*. Pre-existing files are never touched. Reference models (`whisper-tiny` for LID, `parakeet` for TTS-roundtrip ground truth) are pinned and kept until end of run. |

Kaggle invocation:

```bash
python tools/test-all-backends.py \
  --models /kaggle/working/models \
  --cache-mode=ephemeral \
  --profile=feature
```

The reference downloads (~500 MB total) stay resident; each big
backend (1–2 GB Q4_K) comes + goes one at a time, keeping peak
disk to ~2.5 GB even though the registry totals ~15 GB.

### Skip-missing — `--skip-missing`

Disables HF downloads entirely. Backends without a local model on
disk are reported as `NO_MODEL` (not `FAIL`). Useful for CI runners
that should fail on regression but not on missing models.

### Pre-download disk-space check

Each `Backend` declares an `approx_size_mb` hint. Before downloading,
the script checks `shutil.disk_usage(models_dir).free` against
`approx_size_mb + 2 GB margin`; if low, it skips the download
(reports `SKIP — need ~X MB, only Y MB free`) instead of half-filling
the disk.

---

## HOWTO Quantize

CrispASR includes a unified GGUF re-quantization tool, `crispasr-quantize`, that works across all supported model families (Whisper, Parakeet, Canary, Cohere, Voxtral, Qwen3, Granite, Wav2Vec2, etc.).

It is a model-agnostic tool that iterates through the GGUF tensor list and re-quantizes eligible 2D weight matrices while preserving metadata and non-quantizable tensors (norms, positional embeddings, biases) in their original types.

### Usage

```bash
./build/bin/crispasr-quantize model-f16.gguf model-quant.gguf <type>
```

### Supported types

| Type | Description |
|---|---|
| `q4_0` | 4-bit (scale only) |
| `q4_1` | 4-bit (scale + minimum; slightly higher accuracy than q4_0) |
| `q5_0` | 5-bit (scale only) |
| `q5_1` | 5-bit (scale + minimum; slightly higher accuracy than q5_0) |
| `q8_0` | 8-bit (scale only) |
| `q2_k` | 2-bit K-quant |
| `q3_k` | 3-bit K-quant |
| `q4_k` | 4-bit K-quant (generally preferred over legacy Q4) |
| `q5_k` | 5-bit K-quant (generally preferred over legacy Q5) |
| `q6_k` | 6-bit K-quant |

### Examples

```bash
# Quantize a Parakeet F16 model to Q4_K
./build/bin/crispasr-quantize parakeet-f16.gguf parakeet-q4_k.gguf q4_k

# Quantize a Voxtral model to Q5_0
./build/bin/crispasr-quantize voxtral-f16.gguf voxtral-q5_0.gguf q5_0
```

> **Note on alignment.** K-quants (`q2_k` through `q6_k`) require tensor row sizes to be multiples of 256. If a tensor does not meet this requirement (e.g., the 896-wide tensors in some Qwen3-ASR layers), the tool automatically falls back to a compatible legacy quantization type (like `q4_0` or `q8_0`) to ensure the entire model is quantized.

---

## Branch state & roadmap

**24 ASR backends** + 3 TTS backends + 2 punctuation models + VAD/LID/diarization/alignment — all through a unified C-ABI with Python/Rust/Dart wrappers. See [PLAN.md](PLAN.md) for the roadmap, [HISTORY.md](HISTORY.md) for completed milestones, and [PERFORMANCE.md](PERFORMANCE.md) for benchmarks on Kaggle T4 GPU (21/21 of the benchmarked backends pass, fastest 9.0x RT — newer backends not yet in the benchmark suite).

---

## GPU backend selection

All backends use `ggml_backend_init_best()` which automatically picks the highest-priority compiled backend: CUDA > Metal > Vulkan > CPU. To force a specific backend:

```bash
# Force Vulkan even when CUDA is available
crispasr --gpu-backend vulkan -m model.gguf -f audio.wav

# Pin a specific GPU (useful on Vulkan systems with iGPU + dGPU)
crispasr --gpu-backend vulkan -dev 1 -m model.gguf -f audio.wav

# Force CPU (useful for benchmarking)
crispasr -ng -m model.gguf -f audio.wav

# CUDA unified memory (swap to RAM when VRAM exhausted)
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 crispasr -m model.gguf -f audio.wav
```

Build flags: `-DGGML_CUDA=ON`, `-DGGML_METAL=ON`, `-DGGML_VULKAN=ON`.

Notes:
- `--gpu-backend vulkan` selects the Vulkan backend, but it does not choose which physical GPU to use. Use `-dev N` to select the Vulkan device index.
- On some Windows laptops, Vulkan device `0` is the Intel iGPU and the NVIDIA GPU is `1`. If Vulkan looks unexpectedly slow, rerun with `-dev 1`.
- The Windows convenience script `build-vulkan.bat` creates a separate Vulkan-capable binary at `build-vulkan\bin\crispasr.exe`.

---

## Debugging & profiling

For most backends, `-v` / `--verbose` surfaces per-stage timings and
device picks. For headless / library use (where the CLI flag isn't
plumbed through), set `CRISPASR_VERBOSE=1` instead.

```bash
# Per-stage timing breakdown (mel / encoder / prefill / decode):
crispasr -v --backend gemma4-e2b -m model.gguf -f audio.wav
# gemma4_e2b: mel 128x1099 (17.2 ms)
# gemma4_e2b: encoder done: 1536x275 (719.0 ms)
# gemma4_e2b: prefill done, first_token=3133 (1464.0 ms)
# gemma4_e2b: decoded 25 tokens (7748.3 ms total)
# crispasr: transcribed 11.0s audio in 7.75s (1.4x realtime)

# Hugging Face access for gated models (Voxtral, Gemma4-E2B, …):
HF_TOKEN=hf_xxx crispasr -m auto --backend gemma4-e2b -f audio.wav
```

The server has its own auth env: `CRISPASR_API_KEYS` (see
[Server mode](#server-mode-persistent-model-http-api)).

<details>
<summary><b>Per-backend debug / bench / dump-dir env vars (developer)</b></summary>

These are useful when porting a new backend or chasing a regression.
The `*_BENCH=1` toggles emit per-stage timings even without `-v`; the
`*_DEBUG=1` toggles emit per-step diagnostic prints; the `*_DUMP_DIR=`
paths write per-stage F32 tensors for diff-testing against a PyTorch
reference (see [Debug a new backend against PyTorch ground truth](#debug-a-new-backend-against-pytorch-ground-truth)).

| Env var | Purpose |
| --- | --- |
| `CRISPASR_VERBOSE=1` | Forces verbose mode for any backend (parallel to the `-v` flag). |
| `CRISPASR_DUMP_DIR=path/` | Generic per-stage F32 tensor dump for the `crispasr-diff` harness. |
| `GEMMA4_E2B_BENCH=1` | Per-stage timings for the Gemma-4-E2B backend. |
| `COHERE_BENCH=1` / `COHERE_DEBUG=1` | Cohere transcribe per-stage timings / per-step diagnostics. |
| `COHERE_PROF=1` | Cohere graph-level profiling (per-op timings). |
| `COHERE_THREADS=N` | Override thread count for the Cohere backend. |
| `COHERE_DEVICE=cpu\|cuda\|metal\|vulkan` | Force the Cohere backend onto a specific device. |
| `COHERE_DUMP_ATTN=path/` | Dump attention activations for Cohere (used by the diff harness). |
| `FIRERED_BENCH=1` | Per-stage timings for the FireRedASR backend. |
| `FIREREDPUNC_DEBUG=1` | Per-step diagnostics for the FireRed punctuation post-step. |
| `MOONSHINE_STREAMING_BENCH=1` | Per-stage timings for moonshine-streaming. |
| `OMNIASR_BENCH=1` / `OMNIASR_DEBUG=1` / `OMNIASR_DUMP_DIR=` | OmniASR per-stage timings, diagnostics, and stage dumps. |
| `PARAKEET_DEBUG=1` | Parakeet TDT per-step diagnostics (joint network, blank-id sanity). |
| `QWEN3_TTS_BENCH=1` / `QWEN3_TTS_DEBUG=1` / `QWEN3_TTS_DUMP_DIR=` | Qwen3-TTS per-stage timings, diagnostics, and stage dumps. |
| `VIBEVOICE_BENCH=1` / `VIBEVOICE_DEBUG=1` / `VIBEVOICE_DUMP_DIR=` | VibeVoice ASR per-stage timings, diagnostics, and stage dumps. |
| `VIBEVOICE_REF_FEATURES=path` | Replace the live encoder with a saved feature tensor (regression harness). |
| `VIBEVOICE_TTS_DUMP=path/` | VibeVoice TTS per-stage dumps (token IDs, base/TTS hidden, neg condition, frame-0 noise/v_cfg/latent/acoustic_embed) for the diff harness. |
| `VIBEVOICE_TTS_DUMP_PERFRAME=1` | Per-frame VibeVoice TTS dumps written as `perframe_<stage>_f<NNN>.bin`. Pair with `VIBEVOICE_TTS_DUMP=path/` and `VIBEVOICE_TTS_NOISE=path` for stage-by-stage AR diff against `tools/run_official_vibevoice.py`. |
| `VIBEVOICE_TTS_TRACE=1` | Extra one-line traces (negative-condition prefill rms, scaling/bias factors loaded). Same effect as `-vv`. |
| `VIBEVOICE_VOICE_AUDIO=path.wav` | Reference voice WAV for 1.5B-base TTS without a `.gguf` voice cache. |
| `VIBEVOICE_TTS_NOISE=path` | Override the per-frame Gaussian init noise. Flat little-endian float32 `[N_frames, vae_dim]` — typically the `noise.bin` written by `tools/run_official_vibevoice.py`. |
| `VIBEVOICE_VAE_BACKEND=cpu\|metal\|cuda\|vulkan` | Pin the VAE decoder onto a specific backend. |
| `WAV2VEC2_BENCH=1` / `WAV2VEC2_VERBOSE=1` / `WAV2VEC2_DUMP_DIR=` | wav2vec2 per-stage timings, verbose graph traces, and stage dumps. |
| `CRISPASR_VOXTRAL4B_STREAM_TIMING=1` | Per-stage timings for the voxtral4b streaming path (encoder drain / prefill / first-text-token / decode-step p50/p95). |
| `CRISPASR_VOXTRAL4B_STREAM_CHUNK_MS=N` | Override the internal encoder chunk size (default 240 ms). Must be a multiple of 80 ms. Larger = faster feed (kernel-launch amortisation), longer live-caption latency floor. |
| `CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER=1` | Regression-debug: ignore the streaming encoder's audio_embeds and re-run the whole batch encoder at flush. |
| `CRISPASR_VOXTRAL4B_STREAM_DEBUG=1` / `CRISPASR_VOXTRAL4B_STREAM_DIFF=1` | Per-step decode prints / side-by-side encoder cosine vs the batch encoder. |
| `CRISPASR_VOXTRAL4B_STREAM_LIVE=1` | Live-captions decode-during-feed (PLAN #7 phase 3). `get_text()` polled during feed returns progressive transcript. Default OFF (PTT semantics). Wrappers: Python `Session.stream_open(live=True)`, Rust `stream_open_ex(.., live: true)`. |
| `CRISPASR_VOXTRAL4B_STREAM_DECODER_THREAD=1` | Decoder worker thread (PLAN #7 phase 4, implies live mode). Lets `feed()` return between encoder chunks without waiting for the decode loop — useful for mic-driven workloads. On M1 the Metal queue serializes encoder and decoder so total wall-clock is unchanged; faster GPUs with kernel-level parallelism see real overlap. |
| `CRISPASR_VOXTRAL4B_FUSED_QKV=0` | Opt out of the runtime fused-QKV LLM path (default-on, ~7-8 % decode speedup on M1 Q4_K, ~500 MB extra memory). |
| `CRISPASR_QWEN3_ASR_FUSED_QKV=0` | Opt out of the runtime fused-QKV LLM path for qwen3-asr (default-on; works on F16/F32/Q4_K/Q8_0/...). |
| `CRISPASR_VOXTRAL_FUSED_QKV=1` | Opt **in** to the runtime fused-QKV LLM path for voxtral 3B. Off by default (no measurable speedup on JFK-shape decodes; useful for long-form workloads where decode dominates). |
| `QWEN3_TTS_FUSED_QKV=1` | Opt in to the runtime fused-QKV talker path. |
| `GRANITE_DISABLE_ENCODER_GRAPH=1` | Force the granite-speech / -plus / -nar encoder back to the per-layer CPU loop (slower but kept around for debugging). The single ggml-graph encoder with per-layer Shaw RPE is the default and is bit-near-identical to the CPU loop while being ~2× faster end-to-end across all three variants. |
| `CRISPASR_NO_REL_POS=1` | Ablate the relative-position bias in the Gemma-4 audio encoder (development only). |
| `ECAPA_REF_FBANK=path` | Reference filterbank tensor for the ECAPA-TDNN LID model (regression harness). |
| `CRISPASR_SHERPA_LID_BIN=path` | Override the auto-detected sherpa-onnx LID binary. |
| `CRISPASR_ARG_DEVICE=N` | Default GPU device index when `-dev` isn't passed. |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` | Let CUDA swap to RAM when VRAM is exhausted. |
| `GGML_VK_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` | Standard ggml/CUDA device-visibility filters. |

`HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` are both honoured for gated-model
downloads (in that order).

</details>

---

## Credits

- **[whisper.cpp](https://github.com/ggml-org/whisper.cpp)** — the original ggml inference engine and Whisper runtime this fork is built on
- **[ggml](https://github.com/ggml-org/ggml)** — the tensor library everything runs on
- **NVIDIA NeMo** — parakeet-tdt, canary-1b-v2, canary-ctc aligner, and the FastConformer-CTC family (stt_en_fastconformer_ctc_{large,xlarge,xxlarge})
- **Cohere** — cohere-transcribe-03-2026
- **Qwen team (Alibaba)** — Qwen3-ASR-0.6B, Qwen3-ASR-1.7B, Qwen3-ForcedAligner-0.6B
- **Mistral AI** — Voxtral Mini 3B and 4B Realtime
- **IBM Granite team** — Granite Speech 3.2-8b, 3.3-2b, 3.3-8b, 4.0-1b
- **Meta / wav2vec2** — wav2vec2 CTC models (XLSR-53 English, German, multilingual via any Wav2Vec2ForCTC checkpoint)
- **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)** — optional diarization via subprocess (ONNX models)
- **[Silero](https://github.com/snakers4/silero-vad)** — VAD (native GGUF) and language identification (native GGUF, 95 languages)
- **[pyannote](https://github.com/pyannote/pyannote-audio)** — speaker diarization segmentation (native GGUF port)
- **[miniaudio](https://miniaud.io/)** and **[stb_vorbis](https://github.com/nothings/stb)** — embedded audio decoders
- **[Claude Code](https://claude.ai/claude-code)** (Anthropic) — significant portions of the crispasr integration layer, all model converters, and the FastConformer/attention/mel/FFN/BPE core helpers were co-authored with Claude

---

## License

Same as upstream whisper.cpp: **MIT**.

Per-model weights are covered by their respective HuggingFace model licenses (see [Supported backends](#supported-backends)). The `crispasr` binary itself links model runtimes that are mostly permissively licensed (MIT / Apache-2.0 / CC-BY-4.0 for weights).
