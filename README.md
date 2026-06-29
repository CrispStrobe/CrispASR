# CrispASR

**One C++ binary, 35 ASR backends + 23 TTS engines + multilingual text translation, zero Python dependencies.**

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

**Browser**: All backends compile to WebAssembly (4.3 MB) via `build-wasm.sh`.
Multithreaded, runs entirely client-side with COOP/COEP headers.

**Demo**: [HuggingFace Space](https://huggingface.co/spaces/cstr/CrispASR) —
live transcription + TTS + language detection, auto-deployed from `hf-space/`.

### What's new (v0.8.3)

- **Audio format expansion (v0.8.3):** `.opus`, `.webm`, `.au`, `.amr`, `.m4a`/`.aac` decode — no ffmpeg. Apple platforms use AudioToolbox natively for AAC/M4A/ALAC/CAF/AIFF.
- **qwen3-tts: GQA_NATIVE + chunked codec (v0.8.3, #183):** O(N²)→O(N) scaling on Vulkan; RTF stays ~0.5 at any text length. Chunked codec decode caps VRAM peak. Scratch scheduler reset prevents cross-request memory bloat.
- **vibevoice TTS on Vulkan/CUDA (v0.8.3, #184):** fixed segfault on AMD RDNA4 (strided view fix) and illegal memory access on CUDA (stale gallocr state in bucket cache).
- **chatterbox: long-text + turbo emotion tags (v0.8.2, #182):** sentence-chunked synthesis for arbitrarily long input; `[laugh]`, `[whispering]`, `[angry]` etc. drive turbo prosody.
- **Parakeet long-audio = NeMo-exact (#89):** non-JA models default to single full-attention pass, byte-for-byte identical to upstream NeMo; clips past 300 s are silence-split with no boundary duplicates.
- **Hotwords (#98):** `--hotwords "Tokyo,CrispASR"` for CTC/TDT contextual biasing + LLM prompt injection.
- **Global diarization (#110):** `--diarize-method sherpa` / `pyannote` runs once on full audio for consistent speaker IDs.
- **Generation controls:** `--seed`, `--beam-size`, `--frequency-penalty`, `--max-new-tokens` wired through all backends.

### Ecosystem

| Project | What it does |
|---|---|
| **[CrispASR](https://github.com/CrispStrobe/CrispASR)** | This repo — C++ speech engine. 35 ASR backends + 23 TTS engines, CLI + HTTP server + C-ABI + Python/Rust/Dart/Go/Ruby bindings. |
| **[CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver)** | Cross-platform Flutter transcription app built on CrispASR. Desktop + mobile, all 10 backends, model browser with download queue, mic capture, SRT/VTT/JSON export, diarization, batch processing. Fully offline. |
| **[CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)** | Text embedding engine via ggml — same philosophy as CrispASR but for retrieval. 10 architectures (XLM-R, Qwen3-Embed, Gemma3, ModernBERT, ...), dense + sparse + ColBERT + reranking. 9.5x faster than ONNX on CPU, GPU via CUDA/Metal/Vulkan. Python/Rust/Dart bindings. |
| **[Susurrus](https://github.com/CrispStrobe/Susurrus)** | Python ASR GUI with 9 backends (faster-whisper, mlx-whisper, voxtral, insanely-fast-whisper, ...). The Python counterpart to CrispASR's C++ approach. |

---

## Table of contents

- [Supported backends](#supported-backends) — [ASR](#asr-backends) + [TTS](#text-to-speech-models) + [translation](#translation) + [post-processing](#post-processing-models)
- [Feature matrix](#feature-matrix)
- [Install & build](#install--build) — quick install (full guide in [docs/install.md](docs/install.md))
- [Quick start — ASR](#quick-start)
- [**Text-to-Speech (TTS)**](docs/tts.md) — Kokoro, Qwen3-TTS, VibeVoice, Orpheus, Chatterbox, IndexTTS, VoxCPM2, CosyVoice3, CSM, Dia, Zonos, Bark, Piper, MeloTTS, and more
- [Streaming & live transcription](docs/streaming.md)
- [Server mode (HTTP API)](docs/server.md)
- [CLI reference](docs/cli.md) — flags, VAD, CTC alignment, output formats, auto-download, audio formats
- [Language bindings](docs/bindings.md) — Python / Rust / Dart / Go / Java / JavaScript / Ruby / mobile
- [Architecture](docs/architecture.md) — layered layout, `src/core/` primitives, regression discipline
- [Contributing — adding a new backend](docs/contributing.md) — 5-file recipe, ground-truth diff workflow
- [Regression matrix](docs/regression-matrix.md) — `tools/test-all-backends.py` capability tiers
- [Quantize models](docs/quantize.md) — `crispasr-quantize` for all backends
- [GPU backend selection](#gpu-backend-selection)
- [Debugging & profiling](#debugging--profiling)
- [Credits](#credits)

---

## Supported backends

CrispASR ships **35 ASR backends** for transcription/translation and
**23 TTS engines** for synthesis. Pick at the CLI with `--backend NAME`,
or omit it to let the binary auto-detect from the GGUF metadata. Jump
to the [TTS table](#text-to-speech-models) for the synthesis side.

### ASR backends

| Backend | Model | Architecture | Languages | License |
|---|---|---|---|---|
| **whisper** | [`ggml-base.en.bin`](https://huggingface.co/ggerganov/whisper.cpp/) and all OpenAI Whisper variants | Encoder-decoder transformer | 99 | MIT |
| **whisper** | [`distil-whisper/distil-large-v3`](https://huggingface.co/cstr/distil-large-v3-GGUF) | Distilled Whisper: 32L encoder + 2L decoder (6.3x faster) | English | MIT |
| **parakeet** | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | FastConformer + TDT | 25 EU (auto-detect) | CC-BY-4.0 |
| **parakeet** | [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v2-GGUF) | FastConformer + TDT, original Open ASR Leaderboard topper | en (mixed-case + punct) | CC-BY-4.0 |
| **parakeet** | [`nvidia/parakeet-tdt-1.1b`](https://huggingface.co/cstr/parakeet-tdt-1.1b-GGUF) | 42L FastConformer + TDT, larger English variant | en (lowercase) | CC-BY-4.0 |
| **parakeet** | [`nvidia/parakeet-tdt_ctc-110m`](https://huggingface.co/cstr/parakeet-tdt_ctc-110m-GGUF) | 17L FastConformer + TDT+CTC hybrid; smallest variant, auto-CTC decode | en | CC-BY-4.0 |
| **parakeet** | [`nvidia/parakeet-tdt_ctc-1.1b`](https://huggingface.co/cstr/parakeet-tdt_ctc-1.1b-GGUF) | 42L FastConformer + TDT+CTC hybrid; largest, mixed-case + punct | en | CC-BY-4.0 |
| **parakeet** | [`nvidia/parakeet-tdt_ctc-0.6b-ja`](https://huggingface.co/cstr/parakeet-tdt-0.6b-ja-GGUF) | FastConformer-TDT-CTC, xscaling, 80 mels | Japanese | CC-BY-4.0 |
| **reazonspeech** | [`reazon-research/reazonspeech-nemo-v2`](https://huggingface.co/cstr/reazonspeech-nemo-v2-GGUF) | FastConformer-RNNT, local attn (w=256), 80 mels, 619M params | Japanese | Apache-2.0 |
| **fastconformer-ctc** | [`nvidia/parakeet-ctc-0.6b`](https://huggingface.co/cstr/parakeet-ctc-0.6b-GGUF) | 24L FastConformer + CTC, 80 mels (same arch as fc-ctc-xlarge) | en | CC-BY-4.0 |
| **fastconformer-ctc** | [`nvidia/parakeet-ctc-1.1b`](https://huggingface.co/cstr/parakeet-ctc-1.1b-GGUF) | 42L FastConformer + CTC, 80 mels | en | CC-BY-4.0 |
| **fastconformer-ctc** | [`grider-transwithai/parakeet-ctc-1.1b-ja`](https://huggingface.co/cstr/parakeet-ctc-1.1b-ja-GGUF) | 42L FastConformer + CTC, 80 mels, Japanese fine-tune | Japanese | Apache-2.0 |
| **canary** | [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) | FastConformer + Transformer decoder | 25 EU (explicit `-sl/-tl`) | CC-BY-4.0 |
| **lfm2-audio** | [`LiquidAI/LFM2.5-Audio-1.5B`](https://huggingface.co/cstr/lfm2-audio-1.5b-GGUF) | FastConformer + LFM2 hybrid conv+attention backbone (ASR+TTS) | en | LFM Open v1.0 |
| **lfm2-audio** | [`LiquidAI/LFM2.5-Audio-1.5B-JP`](https://huggingface.co/cstr/lfm2-audio-1.5b-jp-GGUF) | FastConformer + LFM2 hybrid conv+attention backbone (ASR+TTS) | ja | LFM Open v1.0 |
| **mini-omni2** | [`gpt-omni/mini-omni2`](https://huggingface.co/gpt-omni/mini-omni2) | Whisper-small + Qwen2-0.5B (ASR+TTS+S2S) | en | MIT |
| **cohere** | [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Conformer + Transformer | 13 | Apache-2.0 |
| **cohere** | [`efwkjn/cohere-asr-ja-v0.1`](https://huggingface.co/TransWithAI/cohere-transcribe-ja-v0.1-GGUF) | Japanese fine-tune of cohere-transcribe-03-2026 (TedX/JSUT-tuned) | Japanese | Apache-2.0 |
| **granite** | [`ibm-granite/granite-speech-{3.2-8b,3.3-2b,3.3-8b}`](https://huggingface.co/ibm-granite/granite-speech-3.3-2b), [`granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | Conformer + Q-Former + Granite LLM (μP) ([more](docs/architecture.md#granite--granite-41--granite-41-plus--granite-41-nar)) | en fr de es pt ja | Apache-2.0 |
| **granite-4.1** | [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | 16L Conformer + Q-Former + Granite LLM; single ggml graph ([more](docs/architecture.md#granite--granite-41--granite-41-plus--granite-41-nar)) | en fr de es pt ja | Apache-2.0 |
| **granite-4.1-plus** | [`ibm-granite/granite-speech-4.1-2b-plus`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | 4.1 + hidden-state concat; punctuated output ([more](docs/architecture.md#granite--granite-41--granite-41-plus--granite-41-nar)) | en fr de es pt | Apache-2.0 |
| **granite-4.1-nar** | [`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar) | Non-autoregressive: single LLM forward + slot argmax ([more](docs/architecture.md#granite--granite-41--granite-41-plus--granite-41-nar)) | en fr de es pt | Apache-2.0 |
| **fastconformer-ctc** | [`nvidia/stt_en_fastconformer_ctc_large`](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large) | FastConformer + CTC (NeMo family, all sizes) | en | CC-BY-4.0 |
| **voxtral** | [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Whisper encoder + Mistral 3B LLM | 8 | Apache-2.0 |
| **voxtral4b** | [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Causal encoder + 3.4B LLM, sliding window | 13, realtime streaming | Apache-2.0 |
| **qwen3** | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Whisper-style audio encoder + Qwen3 0.6B LLM | 30 + 22 Chinese dialects | Apache-2.0 |
| **qwen3-1.7b** | [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Whisper-style audio encoder + Qwen3 1.7B LLM | 30 + 22 Chinese dialects | Apache-2.0 |
| **mega-asr** | [`zhifeixie/Mega-ASR`](https://huggingface.co/zhifeixie/Mega-ASR) | Qwen3-ASR-1.7B + merged robustness LoRA; always-on robust path | noisy / degraded speech | Apache-2.0 |
| **wav2vec2** | [`jonatasgrosman/wav2vec2-large-xlsr-53-english`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) | CNN + 24L transformer + CTC head (any Wav2Vec2ForCTC) | per-model | Apache-2.0 |
| **wav2vec2** | [`facebook/data2vec-audio-base-960h`](https://huggingface.co/cstr/data2vec-audio-960h-GGUF) | Data2Vec Audio (79 MB Q4_K) | English | Apache-2.0 |
| **wav2vec2** | [`facebook/hubert-large-ls960-ft`](https://huggingface.co/cstr/hubert-large-ls960-ft-GGUF) | HuBERT Large (212 MB Q4_K) | English | Apache-2.0 |
| **glm-asr** | [`zai-org/GLM-ASR-Nano-2512`](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) | Whisper encoder + 4-frame projector + Llama 1.5B (GQA) | 17 (Mandarin, English, Cantonese, ...) | MIT |
| **kyutai-stt** | [`kyutai/stt-1b-en_fr`](https://huggingface.co/kyutai/stt-1b-en_fr) | Mimi codec (SEANet + RVQ) + 16L causal LM | en, fr | MIT |
| **firered-asr** | [`FireRedTeam/FireRedASR2-AED`](https://huggingface.co/FireRedTeam/FireRedASR2-AED) | Conformer + CTC + beam search; also LID (120 langs) | Mandarin, English, 20+ Chinese dialects | Apache-2.0 |
| **moonshine** | [`UsefulSensors/moonshine-{tiny,base}`](https://huggingface.co/cstr/moonshine-base-GGUF) | Conv + 6L enc + 6L dec; multilingual variants | English + 6 langs | MIT |
| **moonshine&#8209;de** | [`fidoriel/moonshine-base-de`](https://huggingface.co/cstr/moonshine-base-de-fidoriel-GGUF) | German fine-tune of moonshine-base (6.9% WER CV22) | German | CC&#8209;BY&#8209;NC&#8209;SA&#8209;4.0 |
| **moonshine&#8209;tiny&#8209;de** | [`fidoriel/moonshine-tiny-de`](https://huggingface.co/cstr/moonshine-tiny-de-fidoriel-GGUF) | German fine-tune of moonshine-tiny (11.4% WER CV22) | German | CC&#8209;BY&#8209;NC&#8209;SA&#8209;4.0 |
| **moonshine-streaming** | [`UsefulSensors/moonshine-streaming-{tiny,small,medium}`](https://huggingface.co/cstr/moonshine-streaming-tiny-GGUF) | Streaming: sliding-window encoder + AR decoder (34–245M) | English | MIT |
| **gemma4-e2b** | [`google/gemma-4-E2B-it`](https://huggingface.co/cstr/gemma4-e2b-it-GGUF) | USM Conformer 12L + Gemma4 LLM 35L (GQA, PLE) | 140+ langs | Apache-2.0 |
| **gemma4-e4b** | [`google/gemma-4-E4B-it`](https://huggingface.co/cstr/gemma4-e4b-it-GGUF) | Same USM Conformer 12L + larger Gemma4 LLM 42L (GQA, PLE); runs on `--backend gemma4-e2b` | 140+ langs | Apache-2.0 |
| **omniasr** | [`omniASR-CTC-1B-v2`](https://huggingface.co/cstr/omniASR-CTC-1B-v2-GGUF) | wav2vec2 CNN + 48L transformer + CTC ([more](docs/architecture.md#omniasr-ctc--llm--unlimited)) | **1600+** | Apache-2.0 |
| **omniasr&#8209;300m** | [`omniASR-CTC-300M-v2`](https://huggingface.co/cstr/omniASR-CTC-300M-v2-GGUF) | Same arch, 24L, ~194 MB Q4_K; auto-chunks >7 s ([more](docs/architecture.md#omniasr-ctc--llm--unlimited)) | **1600+** | Apache-2.0 |
| **omniasr-llm** | [`omniASR-LLM-300M-v2`](https://huggingface.co/cstr/omniasr-llm-300m-v2-GGUF) | Same encoder + 12L LLaMA decoder ([more](docs/architecture.md#omniasr-ctc--llm--unlimited)) | **1600+** | Apache-2.0 |
| **omniasr-llm** | [`omniASR-LLM-Unlimited-300M-v2`](https://huggingface.co/cstr/omniasr-llm-unlimited-300m-v2-GGUF) | Streaming: 15s segment protocol, unlimited audio ([more](docs/architecture.md#omniasr-ctc--llm--unlimited)) | **1600+** | Apache-2.0 |
| **vibevoice** | [`microsoft/VibeVoice-ASR`](https://huggingface.co/cstr/vibevoice-asr-GGUF) | σ-VAE ConvNeXt + Qwen2.5-7B ([more](docs/architecture.md#vibevoice)) | 50+ | MIT |
| **mimo-asr** | [`XiaomiMiMo/MiMo-V2.5-ASR`](https://huggingface.co/cstr/mimo-asr-GGUF) | 6L transformer + 36L Qwen2 LM + RVQ codec ([more](docs/architecture.md#mimo-asr)) | Mandarin + dialects + English | MIT |
| **ark-asr** ⚠️*experimental/WIP* | [`cstr/ark-asr-3b-GGUF`](https://huggingface.co/cstr/ark-asr-3b-GGUF) (base [`AutoArk-AI/ARK-ASR-3B`](https://huggingface.co/AutoArk-AI/ARK-ASR-3B)) | Whisper-large-v3 enc (partial RoPE) + Qwen2.5-3B LM ([more](docs/architecture.md#ark-asr)) | 19 (zh, en, de, ja, fr, ko, es, pl, it, ro, hu, cs, nl, fi, hr, sk, sl, et, lt) | see base |
| **moss-audio** | [`OpenMOSS-Team/MOSS-Audio-4B-Instruct`](https://huggingface.co/cstr/MOSS-Audio-4B-Instruct-GGUF) | 32L Whisper encoder + DeepStack 3-tap + 36L Qwen3 LM; audio understanding + ASR ([more](docs/architecture.md#moss-audio)) | zh, en | Apache-2.0 |
| **funasr** | [`FunAudioLLM/Fun-ASR-Nano-2512`](https://huggingface.co/cstr/funasr-nano-GGUF) | 70-block SANM encoder + 2-block Transformer adaptor + Qwen3-0.6B LLM | zh, yue, en, ja, ko | FunASR Model License v1.1 (commercial OK w/ attribution) |
| **fun-asr-mlt-nano** | [`FunAudioLLM/Fun-ASR-MLT-Nano-2512`](https://huggingface.co/cstr/funasr-mlt-nano-GGUF) | Same architecture, multilingual decoder | 31 langs incl. de, fr, es, pt, ru, ar, hi, vi, th, ko | FunASR Model License v1.1 |
| **paraformer** | [`funasr/paraformer-zh`](https://huggingface.co/cstr/paraformer-zh-GGUF) | 50-block SANM encoder + CIF predictor + 16-block NAR decoder (single-pass, non-autoregressive); character-level vocab (8404); 220M params | zh, en | FunASR Model License (commercial OK w/ attribution) |
| **sensevoice** | [`FunAudioLLM/SenseVoiceSmall`](https://huggingface.co/cstr/sensevoice-small-GGUF) | 70-block SANM encoder + CTC head; emits transcript + language ID + emotion + audio-event in one forward pass (non-AR, 15× faster than Whisper-Large); structured C ABI + `-oj` JSON expose the four tags as separate fields | 50+ langs; native LID + emotion + audio-event tags | FunASR Model License v1.1 |

### Text-to-Speech models

Synthesis backends, driven by the `--tts` flag and a `--tts-output PATH.wav`.
See the dedicated [Text-to-Speech](#text-to-speech-tts) section below for
quick-start commands and engine selection guidance.

| Backend | Models | Architecture | Languages | License |
|---------|--------|-------------|-----------|---------|
| **vibevoice-tts** | [`VibeVoice-Realtime-0.5B`](https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF), [`VibeVoice-1.5B`](https://huggingface.co/cstr/vibevoice-1.5b-GGUF) | DPM-Solver++ + σ-VAE decoder; voice presets or cloning | en, zh | MIT |
| **kugelaudio** | [`kugelaudio-0-open`](https://huggingface.co/cstr/kugelaudio-0-open-GGUF) | Qwen2.5-7B LM + 4L DiT diffusion + acoustic VAE decoder; voice cloning | multilingual | Apache-2.0 |
| **qwen3-tts** | [`Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF), [`1.7B-Base`](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF), [`1.7B-VoiceDesign`](https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF) | Qwen3 talker LM + 12 Hz RVQ ([more](docs/architecture.md#qwen3-tts)) | multilingual | Apache-2.0 |
| **qwen3-tts-customvoice** | [`1.7B-CustomVoice`](https://huggingface.co/cstr/qwen3-tts-1.7b-customvoice-GGUF) | Same talker + 9 premium built-in speakers (`--voice <name>`); optional style via `--instruct` (e.g. "spoke very slowly") ([more](docs/architecture.md#qwen3-tts)) | multilingual | Apache-2.0 |
| **melotts** | [`myshell-ai/MeloTTS`](https://github.com/myshell-ai/MeloTTS) EN_V2 | VITS2 (6L transformer + SDP/DP + transformer coupling flow + HiFi-GAN); 44.1 kHz, 102 MB + 52 MB BERT Q4_K companion (154 MB total); neural G2P; 4 EN speakers ([more](docs/architecture.md#melotts)) | en | MIT |
| **piper** | [`rhasspy/piper`](https://github.com/rhasspy/piper) community voices | VITS (6L transformer + SDP + 4-block coupling flow + HiFi-GAN); 22 kHz mono, 30 MB F16 per voice; built-in G2P for EN/DE/FR/ES (`--g2p-dict`) | 30+ langs (built-in + espeak dlopen) | MIT |
| **kokoro** | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) + German backbones | StyleTTS2 / iSTFTNet (82M); per-voice GGUF ([more](docs/architecture.md#kokoro)) | en, es, fr, hi, it, ja, pt, zh, de | Apache-2.0 |
| **orpheus** | [`Orpheus-3B-FT`](https://huggingface.co/cstr/orpheus-3b-0.1-ft-GGUF) + [`SNAC 24 kHz`](https://huggingface.co/cstr/snac-24khz-GGUF) | Llama-3.2-3B + SNAC RVQ codec; 8 speakers ([more](docs/architecture.md#orpheus)) | en, de | Llama / MIT |
| **chatterbox** | [`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) + turbo/kartoffelbox/lahgtna variants | T3 AR + S3Gen flow-matching ([more](docs/architecture.md#chatterbox--chatterbox-turbo--kartoffelbox-turbo--lahgtna-chatterbox)) | 23 multilingual; separate Arabic (`lahgtna-chatterbox`) and German turbo (`kartoffelbox-turbo`) fine-tunes | MIT |
| **indextts** | [`cstr/indextts-1.5-GGUF`](https://huggingface.co/cstr/indextts-1.5-GGUF) | GPT-2 AR (24L/1280d) + Conformer conditioning + BigVGAN vocoder; voice cloning via reference audio | zh, en | Apache-2.0 |
| **voxcpm2-tts** | [`cstr/voxcpm2-GGUF`](https://huggingface.co/cstr/voxcpm2-GGUF) | Tokenizer-free CFM diffusion AR (TSLM + RALM + LocDiT) at 48 kHz native; zero-shot + voice cloning via `--voice <wav>` | 30 languages | Apache-2.0 |
| **cosyvoice3-tts** | [`cstr/cosyvoice3-0.5b-2512-GGUF`](https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF) | Qwen2-0.5B AR speech-token LM + DiT-CFM (10-step Euler) + HiFT (NSF + iSTFT) at 24 kHz; baked-voice zero-shot cloning via `--voice <name>` | 9 langs + 18 zh dialects | Apache-2.0 |
| **csm** | [`cstr/csm-1b-GGUF`](https://huggingface.co/cstr/csm-1b-GGUF) | Sesame CSM-1B conversational TTS: Llama-3.2 1B backbone + 100M depth decoder (32-codebook RVQ) + Kyutai Mimi codec at 24 kHz ([more](docs/architecture.md#csm)) | en | Apache-2.0 |
| **lfm2-audio** | [`cstr/lfm2-audio-1.5b-GGUF`](https://huggingface.co/cstr/lfm2-audio-1.5b-GGUF) + [`jp`](https://huggingface.co/cstr/lfm2-audio-1.5b-jp-GGUF) | LFM2.5-Audio ASR+TTS+S2S: FastConformer enc + LFM2 hybrid backbone + depthformer (8-codebook Mimi) + ISTFT detokenizer at 24 kHz; interleaved text+audio generation | en, ja | LFM Open v1.0 |
| **dia** | [`nari-labs/Dia-1.6B`](https://huggingface.co/cstr/dia-1.6b-GGUF) | Byte-level text encoder (12L) + AR audio decoder (18L GQA + CFG) → 9 delayed DAC codebooks + 44.1 kHz DAC codec; dialogue style with `[S1]`/`[S2]` tags (use >100-char prompts) | en | Apache-2.0 |
| **zonos-tts** | [`cstr/zonos-v0.1-transformer-GGUF`](https://huggingface.co/cstr/zonos-v0.1-transformer-GGUF) + [`cstr/dac-44khz-GGUF`](https://huggingface.co/cstr/dac-44khz-GGUF) | Zyphra Zonos-v0.1: 26L GQA AR transformer (2B) + 9-codebook DAC @ 44.1 kHz; CFG-guided; voice cloning via reference WAV ([more](docs/architecture.md#zonos-tts)) | en | Apache-2.0 |
| **bark** | [`cstr/bark-small-GGUF`](https://huggingface.co/cstr/bark-small-GGUF) | Suno Bark 3-stage GPT-2 TTS: text→semantic (12L) → coarse EnCodec (12L, 2 codebooks) → fine (12L, 8 codebooks) → EnCodec 24 kHz decoder; speaker conditioning via `.npz` prompts (`--voice <file.npz>`) | multilingual | MIT |
| **speecht5** | [`cstr/speecht5-tts-GGUF`](https://huggingface.co/cstr/speecht5-tts-GGUF) | SpeechT5 80M: char-level encoder (12L) + AR mel decoder (6L) + 5-layer conv postnet + HiFi-GAN at 16 kHz; speaker via 512-d x-vector (`--voice <xvector.bin>`) | en | MIT |
| **fastpitch** | [`cstr/fastpitch-en-GGUF`](https://huggingface.co/cstr/fastpitch-en-GGUF) | NVIDIA FastPitch 60M: non-autoregressive parallel TTS — 6L encoder + duration/pitch predictors + 6L decoder + HiFi-GAN at 22 kHz; deterministic, single forward pass ([more](docs/architecture.md#fastpitch)) | en | CC-BY-4.0 |
| **parler-tts** | [`cstr/parler-tts-mini-v1.1-GGUF`](https://huggingface.co/cstr/parler-tts-mini-v1.1-GGUF) | Parler TTS Mini v1.1 (~900M): T5 encoder + MusicGen decoder + DAC 44.1 kHz; prompt-conditioned (describe voice in text via `--instruct`) | en | Apache-2.0 |
| **outetts** | [`cstr/outetts-0.3-1b-GGUF`](https://huggingface.co/cstr/outetts-0.3-1b-GGUF) | OLMo-1B talker + WavTokenizer single-codebook VQ-GAN at 24 kHz; voice cloning via speaker profile JSON (`--voice <speaker.json>`) | en | CC-BY-4.0 |
| **pocket-tts** | [`cstr/pocket-tts-GGUF`](https://huggingface.co/cstr/pocket-tts-GGUF) | Kyutai Pocket TTS 100M: continuous-latent AR at 12.5 Hz + one-step LSD flow + Mimi VAE 24 kHz; voice cloning via ref audio ([more](docs/architecture.md#pocket-tts)) | en | MIT / CC-BY-4.0 |
| **tada** | [`cstr/tada-tts-1b-GGUF`](https://huggingface.co/cstr/tada-tts-1b-GGUF) + `HumeAI/tada-3b-ml` | Llama-3.2 1B/3B backbone + per-token FM diffusion head + TADA codec at 24 kHz; 1:1 text-to-acoustic alignment; default prompt via `tada-ref.gguf`, custom voices via `--voice <tada-ref.gguf>` built with `models/convert-tada-ref-to-gguf.py` ([more](docs/architecture.md#tada)) | en | — |

<details>
<summary><b>TTS feature matrix</b></summary>

| Backend | Voice cloning | Sampling | kHz | Auto-download | Flash attn |
|---------|:---:|:---:|:---:|:---:|:---:|
| vibevoice-tts | yes | temp | 24 | yes | yes |
| qwen3-tts | yes* | temp | 24 | yes | yes |
| kokoro | — | — | 24 | yes | — |
| orpheus | — | temp | 24 | yes | yes |
| chatterbox | yes | temp | 24 | yes | yes |
| outetts | yes (JSON) | temp | 24 | yes | yes |
| indextts | yes | temp | 24 | yes | yes |
| voxcpm2-tts | yes | — | 48 | yes | — |
| cosyvoice3-tts | yes | temp | 24 | yes | yes |
| f5-tts | yes | — | 24 | yes | — |
| csm | — | temp | 24 | yes | — |
| dia | — | temp | 44 | yes | — |
| bark | yes (.npz) | temp | 24 | yes | — |
| speecht5 | yes (xvec) | — | 16 | yes | — |
| parler-tts | — | temp | 44 | yes | — |
| fastpitch | — | — | 22 | — | — |
| piper | — | — | 22 | — | — |
| pocket-tts | yes | temp | 24 | yes | — |
| tada | yes | temp | 24 | yes | — |

\* CustomVoice variant only; Base uses baked speakers via `--voice <name>`.

</details>

### Translation

Text-to-text translation, distinct from the audio-side `--translate`
flag (which routes audio → English text on whisper / canary / etc.).
Driven by `--text "..." -sl <src> -tl <tgt>`.

| Backend | Models | Architecture | Languages | License |
|---|---|---|---|---|
| **m2m100** | [`facebook/m2m100_418M`](https://huggingface.co/cstr/m2m100-418m-GGUF) | 12L enc + 12L dec transformer, SentencePiece 128K ([more](docs/architecture.md#m2m100--wmt21)) | 100 langs, any-to-any | MIT |
| **m2m100-wmt21** | [`facebook/wmt21-dense-24-wide-en-x`](https://huggingface.co/cstr/wmt21-dense-24-wide-en-x-GGUF) + [`facebook/wmt21-dense-24-wide-x-en`](https://huggingface.co/cstr/wmt21-dense-24-wide-x-en-GGUF) | Same as m2m100, scaled to 4.7B (24L enc) ([more](docs/architecture.md#m2m100--wmt21)) | English ↔ 7 langs (separate `en-x` / `x-en` checkpoints) | MIT |
| **madlad** | [`google/madlad400-3b-mt`](https://huggingface.co/cstr/madlad400-3b-mt-GGUF) | T5 enc-dec (12L+12L, d=2048, gated-GELU, RMSNorm) ([more](docs/architecture.md#madlad)) | 419 languages | Apache-2.0 |

```bash
# m2m100 base (production-ready)
./build/bin/crispasr --backend m2m100 -m auto \
    --text "Hello world, how are you today?" \
    -sl en -tl de
# → Hallo Welt, wie bist du heute?

# WMT21 dense (English ↔ X, 4.7B — auto-downloads ~2.5 GB).
# Two separate checkpoints: en-x for English-source, x-en for
# English-target. Pick the one matching your `-sl`/`-tl` direction
# (or pass an explicit `-m <path>` to load the other manually).
./build/bin/crispasr --backend m2m100-wmt21 -m auto \
    --text "The president said he would not attend." \
    -sl en -tl de   # uses wmt21-dense-24-wide-en-x

./build/bin/crispasr --backend m2m100-wmt21 \
    -m models/wmt21-dense-24-wide-x-en-q4_k.gguf \
    --text "Le président a dit qu'il ne serait pas présent." \
    -sl fr -tl en   # uses wmt21-dense-24-wide-x-en

# MADLAD-400 3B (419 languages, bit-token-identical to Python SP)
./build/bin/crispasr --backend madlad -m auto \
    --text "Hello world." \
    -sl en -tl ta
```

For 2-stage pipelines (e.g., ASR → m2m100), use the dedicated
`--tr-sl` / `--tr-tl` flags; they fall back to `-sl` / `-tl` when
unset, so single-stage standalone usage is just `-sl/-tl`.

### Post-processing models

Work with all backends.

| Model | Task | Architecture | Languages | License | HuggingFace |
|---|---|---|---|---|---|
| **FireRedPunc** | Punctuation restoration | BERT-base (12L, d=768), 5 classes | Chinese + English | Apache-2.0 | [`cstr/fireredpunc-GGUF`](https://huggingface.co/cstr/fireredpunc-GGUF) |
| **fullstop-punc** | Punctuation restoration | XLM-RoBERTa-large (24L, d=1024), 6 classes | EN, DE, FR, IT | MIT | [`cstr/fullstop-punc-multilang-GGUF`](https://huggingface.co/cstr/fullstop-punc-multilang-GGUF) |
| **punctuate-all** | Punctuation restoration | XLM-RoBERTa-base (12L, d=768), 6 classes | 12 languages | MIT | [`cstr/punctuate-all-GGUF`](https://huggingface.co/cstr/punctuate-all-GGUF) |
| **PCS** | Punc + truecase + SBD | XLM-RoBERTa-base (12L), 4 heads | 47 languages | Apache-2.0 | `--punc-model pcs` |
| **truecaser&#8209;lstm** | German truecasing (best) | BiLSTM char-level (2×150, 3.2 MB, 97.9% F1) | German | Apache-2.0 | `--truecase-model lstm` |
| **truecaser&#8209;crf** | German truecasing | CRF + context features (8.5 MB) | German | MIT | `--truecase-model crf` |
| **truecaser&#8209;de** | German truecasing (simple) | Statistical word-frequency (71K entries, 1.7 MB) | German | MIT | `--truecase-model auto` |
| **CLD3** | Text language ID | Embedding-bag → FC + ReLU → softmax (~1.5 MB F32) | 109 ISO 639-1 | Apache-2.0 | [`cstr/cld3-GGUF`](https://huggingface.co/cstr/cld3-GGUF) |
| **GlotLID-V3** | Text language ID | fastText supervised, flat softmax | 2102 ISO 639-3 + script | Apache-2.0 | [`cstr/glotlid-GGUF`](https://huggingface.co/cstr/glotlid-GGUF) |
| **LID-176** | Text language ID | fastText supervised, hierarchical softmax | 176 ISO 639-1 | CC-BY-SA-3.0 | [`cstr/fasttext-lid176-GGUF`](https://huggingface.co/cstr/fasttext-lid176-GGUF) |

All runtimes share ggml-based inference. The speech-LLM backends (**qwen3**, **voxtral**, **voxtral4b**, **granite**, **glm-asr**, **kyutai-stt**) inject audio encoder frames directly into an autoregressive language model's input embeddings, instead of using a dedicated CTC/transducer/seq2seq decoder. The **fastconformer-ctc** backend hosts the NeMo FastConformer-CTC standalone ASR family — `stt_en_fastconformer_ctc_{large,xlarge,xxlarge}` and the architecturally-identical `parakeet-ctc-{0.6b,1.1b}` (different training data + tokenizer, same encoder + head shape) — with greedy CTC decoding. Same C++ runtime as the canary-ctc aligner.

## Feature matrix

Run `crispasr --list-backends` to see it live. Each backend declares capabilities at runtime; if you ask for a feature the selected backend does not support, CrispASR prints a warning and silently ignores the flag.

**Sortable / filterable view:** [`docs/feature-matrix.html`](docs/feature-matrix.html) — click any column header to sort, type to filter rows, click cap pills to require a capability. Generated from `crispasr --list-backends-json` (single source of truth — drift impossible). Regenerate via `python tools/gen-feature-matrix.py`. A Markdown twin lives at [`docs/feature-matrix.md`](docs/feature-matrix.md).

The static table below is a curated subset focusing on the ASR backends and the cross-cutting features that matter for ASR pipelines. The full 39-backend × 18-cap surface is in the generated views.

<!-- Generated from `crispasr --list-backends` + cross-cutting features. -->

| Feature | whisper | parakeet | canary | cohere | granite | granite&#8209;4.1 | voxtral | voxtral4b | qwen3 | fc&#8209;ctc | wav2vec2 | glm&#8209;asr | kyutai&#8209;stt | firered | moonshine | moon&#8209;stream | omniasr | omniasr&#8209;llm | vibevoice | gemma4&#8209;e2b | mimo&#8209;asr | funasr | paraformer | sensevoice |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Native timestamps | ✔ | ✔ | ✔ | ✔ | | | | | | | | | ✔ | | | | | | | | | | | |
| CTC timestamps | | | ✔ | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Word-level timing | ✔ | ✔ | ✔ | ✔ | `-am` | ✔† | `-am` | `-am` | `-am` | `-am` | `-am` | `-am` | ✔ | `-am` | `-am` | `-am` | `-am` | `-am` | | `-am` | `-am` | `-am` | | `-am` |
| Per-token confidence | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | ✔ | | |
| Language auto-detect | ✔ | ✔ | LID | LID | LID | LID | LID | LID | ✔ | LID | LID | ✔ | LID | LID | LID | LID | LID | LID | LID | ✔ | LID | LID | LID | ✔ |
| Speech translation | ✔ | | ✔ | | ✔ | ✔ | ✔ | | ✔ | | | | | | | | | | | | | | | |
| Speaker diarization | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Grammar (GBNF) | ✔ | | | | | | | | | | | | | | | | | | | | | | | |
| Temperature sampling | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | ✔ | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | |
| Beam search | ✔ | | | | ✔ | ✔ | ✔ | | ✔ | | | ✔ | ✔ | ✔ | ✔ | | ✔ | ✔ | | | | | | |
| Flash attention | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Punctuation toggle | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | ✔ | | ✔ | | ✔ | ✔ | | | | ✔ | | ✔ |
| Punc restoration | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp |
| Source / target language | | | ✔ | | ✔ | ✔ | ✔ | | ✔ | | | | | | | | | | | | | | | |
| Audio Q&A (`--ask`) | | | | | * | * | ✔ | | * | | | * | | | | | | | | * | * | | | |
| Streaming | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Auto-download (`-m auto`) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| KV quant (`CRISPASR_KV_QUANT`, plus per-half `_K` / `_V`) | | | | | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | | | | | | ✔ | | ✔ | ✔ | ✔ | | |
| mmap weights (`CRISPASR_GGUF_MMAP`) | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| TTS | | | | | | | | | | | | | | | | | | | ✔ | | | | | |

The matrix above covers 24 ASR backends. **Additional ASR backends** not shown: `nemotron` (39-lang streaming ASR with cache-aware FastConformer + RNN-T), `lfm2-audio` (ASR + TTS + S2S in one model), `moss-audio` (audio understanding + ASR), `mini-omni2` (ASR + TTS + S2S), `kugelaudio` (7B audio understanding). See [`docs/feature-matrix.md`](docs/feature-matrix.md) for the full 76-backend matrix. **TTS-only backends** (`kokoro`, `qwen3-tts` + variants, `vibevoice-tts`, `orpheus` + DE variants, `chatterbox` / `chatterbox-turbo` / `kartoffelbox-turbo` / `lahgtna-chatterbox`, `dia`, `bark`, `outetts`, `zonos`, `csm`, `f5-tts`, `parler-tts`, `speecht5`, `piper`, `fastpitch`, `pocket-tts`, `melotts`, `cosyvoice3`, `voxcpm2`, `tada-tts`) all carry the TTS, AUTO_DOWNLOAD, TEMPERATURE, and FLASH_ATTN caps; per-backend cloning + voice-pack support is documented in the [Text-to-Speech models](#text-to-speech-models) table above and [`docs/tts.md`](docs/tts.md). The vibevoice and lfm2-audio columns mark dual-mode (ASR + TTS) backends.

**Key:** ✔ = native/built-in, `-am` = via CTC forced aligner (`-am canary-ctc-aligner.gguf` or `-am qwen3-forced-aligner.gguf`), **LID** = via external language identification pre-step (`-l auto`), **pp** = via `--punc-model` post-processor (FireRedPunc or fullstop-punc), * = experimental or partial support, † = PLUS variant only (native `[T:N]` word timestamps with `-owts`; base uses `-am`). granite-4.1 covers both the regular and `-plus` variants; granite-4.1-nar is a non-autoregressive variant with encoder+projector only (no LLM decode features). The **KV quant** row marks backends that honor `CRISPASR_KV_QUANT={f16,q8_0,q4_0}` — CTC-style backends without a KV cache (parakeet, fc-ctc, wav2vec2, kyutai-stt, firered, moonshine variants, omniasr-CTC) don't apply. The same backends also honor the per-half `CRISPASR_KV_QUANT_K` / `CRISPASR_KV_QUANT_V` overrides (llama.cpp `--cache-type-k` / `--cache-type-v` parity) for asymmetric K-vs-V precision; common recipe `K=q8_0 V=q4_0` saves ~40 % more KV memory than symmetric Q8_0. The **mmap weights** row marks backends consuming `core_gguf::load_weights()` and therefore honoring `CRISPASR_GGUF_MMAP=1`; whisper itself uses upstream's loader and is unaffected. See [`docs/cli.md`](docs/cli.md) Memory footprint for usage + recommended combos.

**Speaker diarization** as a post-processing step via `--diarize`:
- `energy` / `xcorr` — stereo-only, no extra deps
- `pyannote` — native GGUF (no Python, no sherpa-onnx); add `--diarize-embedder auto` (TitaNet) or `--diarize-embedder indextts` (ECAPA-TDNN) for globally stable speaker IDs across long files
- `sherpa` / `ecapa` — external [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) subprocess; runs once globally on full audio for consistent speaker IDs (#110)
- `vad-turns` — mono-friendly gap-based proxy

Full reference + tuning knobs (cluster threshold, max speakers, pluggable embedder adapters): see [`docs/cli.md#diarization`](docs/cli.md#diarization).

**Language identification** for backends without native LID: `--lid-backend whisper` (default, 75 MB ggml-tiny.bin), `--lid-backend silero` (native GGUF, 16 MB, 95 languages), or `--lid-backend firered` (FireRedLID, 1.7 GB, 120 languages — Conformer encoder + Transformer decoder).

**Voice activity detection**: `--vad` uses the default Silero VAD (~885 KB, auto-downloaded). Each VAD segment is transcribed independently, producing separate SRT/VTT entries with correct timestamps. Use `--vad --split-on-punct` for best subtitle output. Four VAD backends: Silero (default), FireRedVAD (`-vm firered`, recommended), MarbleNet (`-vm marblenet`, 439 KB, 6 languages), Whisper-VAD-EncDec (`-vm whisper-vad`, experimental).

**Punctuation restoration** (`--punc-model`): CTC-based backends output lowercase without punctuation. Named shortcuts: `auto`/`firered` (Chinese+English), `fullstop` (EN/DE/FR/IT, XLM-R-large), `punctuate-all` (12 languages, XLM-R-base), `pcs` (47 languages, punc + truecasing + sentence boundary detection in one model). Or pass a GGUF path directly. Also available via Python/Rust/Dart wrappers (`crispasr.PuncModel`).

**Truecasing** (`--truecase-model`): Restore German noun/name capitalization in lowercase ASR output. Three options in ascending quality: `auto` (statistical, 9 MB), `crf` (CRF with context, 24 MB), `lstm` (BiLSTM char-level, 3.2 MB, **recommended** — 97.9% F1, handles adjective/noun distinction and formal "Ihnen"). All auto-download from [`cstr/truecaser-de`](https://huggingface.co/cstr/truecaser-de). Or use `--punc-model pcs` for neural punc + truecasing in one pass (47 languages).

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
| funasr | ✔ | ✔ | LLM output (Qwen3-0.6B decoder). Chinese chars carry full-width period; mlt-nano variant adds Latin-script casing + punctuation. |
| sensevoice | ✔ | ✔ | CTC output with native ITN — toggle via `--punctuation` / `--no-punctuation`, controls Arabic-digit vs spelled-out numerals + comma/period emission. |
| paraformer | **no** | **no** | NAR character-level output — add `--punc-model` |
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
| **Mandarin + Chinese dialects** | **firered-asr**, **qwen3**, **glm-asr**, **funasr**, **paraformer**, **sensevoice** |
| **Multilingual (31 langs) speech-LLM** | **fun-asr-mlt-nano**, **qwen3**, **omniasr-llm**, **gemma4-e2b** |
| **Multilingual (50+ langs) + LID + emotion + audio-event in one pass** | **sensevoice** (encoder-only CTC, non-AR, 15× faster than Whisper-Large) |

### CPU performance tips

Audio-LLM backends (`qwen3`, `voxtral`, `granite`, `glm-asr`, etc.) run full
transformer decoder stacks (28+ layers, 2048-dim) and are **dramatically slower
on CPU** than encoder-only backends. On older dual-core hardware they can drop
below 0.01× realtime. If you're on CPU-only hardware:

- Prefer **moonshine** (16× RT), **fc-ctc** (10× RT), **parakeet** (2.9× RT),
  or **whisper** for usable speeds.
- Use `--flush-after 1` to see results as each VAD slice completes instead of
  waiting for the entire file.
- Use `-pp` / `--print-progress` for per-slice progress indicators on all
  backends (unified backends show slice-level progress; whisper shows
  encoder-level progress).
- Quantize models to Q4_K or Q5_K to reduce memory and compute.

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

### Text language identification (post-ASR / standalone)

Audio LID (above) tags **what was spoken**; text LID tags **what was
written**. Text LID runs on a transcript or any UTF-8 string and is
useful for routing post-ASR pipelines (translation, punctuation, sub
selection) without re-running an audio model. Three GGUF families,
one binary — the dispatcher picks by `general.architecture`:

| Backend | Labels | Size (F16) | License | HF repo |
|---|---:|---:|---|---|
| **CLD3** (Google compact language detector v3) | 109 ISO 639-1 | **440 KB** | Apache-2.0 | [`cstr/cld3-GGUF`](https://huggingface.co/cstr/cld3-GGUF) |
| **GlotLID-V3** (cis-lmu fastText) | 2102 ISO 639-3 + script | 250 MB | Apache-2.0 | [`cstr/glotlid-GGUF`](https://huggingface.co/cstr/glotlid-GGUF) |
| **LID-176** (Facebook fastText) | 176 ISO 639-1 | 63 MB | CC-BY-SA-3.0¹ | [`cstr/fasttext-lid176-GGUF`](https://huggingface.co/cstr/fasttext-lid176-GGUF) |

¹ LID-176 is **CC-BY-SA-3.0 (viral)** — redistributors of the GGUF
inherit ShareAlike. CLD3 + GlotLID-V3 are Apache-2.0 with no such
constraint. Pick CLD3 for the smallest, fastest path; GlotLID for
maximum coverage (low-resource languages); LID-176 only if you need
its specific 176-label space and accept the SA obligation.

**Standalone CLI** — auto-routes by GGUF arch, with auto-download:

```bash
crispasr-lid -m auto --text "Bonjour le monde"        # → cstr/cld3-GGUF (default, ~440 KB)
crispasr-lid -m auto:glotlid --text "Bonjour le monde" -k 5
crispasr-lid -m auto:lid-fasttext176 --text "Hallo Welt"
# Or pass an explicit path / canonical filename (looked up in the registry):
crispasr-lid -m cld3-f16.gguf --text "你好世界"
# zh	0.997816
echo "Привет мир" | crispasr-lid -m auto --quiet
# ru	0.907322
```

**Post-ASR pipeline** — `--lid-on-transcript` runs the same dispatcher
on the assembled transcript (also accepts `auto[:variant]`):

```bash
crispasr -m ggml-tiny.bin -f speech.wav --lid-on-transcript auto
# (transcript on stdout)
# lang=de	conf=0.997123	backend=lid-cld3
```

The dispatcher (`src/text_lid_dispatch.{h,cpp}`) is a thin C ABI
façade — one integer compare per call; per-stage diff harness is
green at cos≥0.999 across 8 multilingual smoke samples.

---

## Install & build

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Produces `build/bin/crispasr` (main CLI), `build/bin/crispasr-quantize`,
and `build/bin/crispasr-diff`. No Python, PyTorch, or pip required at
runtime — just a C++17 compiler and CMake 3.14+.

For GPU acceleration, add the matching ggml flag at configure time:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON     # NVIDIA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON    # Apple Silicon
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON   # cross-vendor
```

**See [`docs/install.md`](docs/install.md)** for the full guide:
all GPU backends (CUDA / Metal / Vulkan / MUSA / SYCL), Windows
convenience scripts, ffmpeg ingestion, optional BLAS, glibc notes,
and the `scripts/dev-build.sh` wrapper.

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

## Streaming, TTS, and HTTP server

CrispASR has three feature areas that warrant their own docs pages:

- **[Streaming & live transcription](docs/streaming.md)** — `--stream`,
  `--mic`, `--live`, sliding-window chunking, per-token confidence.
- **[Text-to-Speech (TTS)](docs/tts.md)** — Kokoro (multilingual,
  smallest), Qwen3-TTS (highest fidelity, voice cloning), VibeVoice
  (lowest-latency streaming), Orpheus (3 B Llama + SNAC), Chatterbox
  (flow-matching + HiFT vocoder, German via Kartoffelbox), IndexTTS,
  VoxCPM2, and CosyVoice3 (9 langs + 18 zh dialects; baked-voice bank +
  arbitrary-WAV cloning). Voice packs, language routing, and qwen3-tts
  environment switches. All TTS output is watermarked; post-embed
  verification warns if confidence is low. Use
  `--detect-watermark file.wav` to check any WAV for AI watermarks.
- **[Server mode (HTTP API)](docs/server.md)** — persistent model,
  OpenAI-compatible `/v1/audio/transcriptions` (ASR) and
  `/v1/audio/speech` + `/v1/voices` (TTS, automatic on any loaded
  CAP_TTS backend), per-request voice + speed + instructions, CORS,
  long-form sentence chunking, API keys, Docker Compose, prebuilt
  CUDA images.

Quickest taste of each:

```bash
# Streaming from microphone
crispasr --mic -m model.gguf

# TTS via auto-downloaded VibeVoice (~636 MB on first run)
crispasr --backend vibevoice-tts -m auto --tts "Hello world" --tts-output hello.wav

# CosyVoice3 on GPU; companions auto-download beside the LLM
crispasr --backend cosyvoice3-tts -m auto --tts "Hello world" --tts-output cosy.wav

# CosyVoice3 fast mode: 5 flow steps instead of the quality-default 10
COSYVOICE3_FLOW_STEPS=5 crispasr --backend cosyvoice3-tts -m auto \
  --tts "Hello world" --tts-output cosy-fast.wav

# Persistent HTTP server, OpenAI-compatible
crispasr --server -m model.gguf --port 8080
curl -F "file=@audio.wav" http://localhost:8080/v1/audio/transcriptions

# TTS over HTTP — load a TTS backend, hit /v1/audio/speech
crispasr --server --backend qwen3-tts-customvoice -m auto --voice-dir ./voices --port 8080
curl http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello world","voice":"vivian"}' -o out.wav
```

CosyVoice3 uses batched classifier-free guidance and request-sized KV
caching by default. Baked voices load only the LLM, flow, HiFT, and voice
bank; the larger S3 tokenizer and CAMPPlus companions load lazily when a
`.wav` cloning voice is first requested.

---

## CLI reference

Common flags:

```bash
crispasr -m auto --backend parakeet -f audio.wav --vad -osrt --split-on-punct
```

| Flag | Meaning |
|---|---|
| `-m FNAME` / `--backend NAME` | Model path (or `auto`) and forced backend |
| `-f FNAME` | Input audio (repeatable; positional accepted) |
| `--vad` | Silero VAD chunking — strongly recommended for multi-minute audio |
| `-osrt` / `-ovtt` / `-otxt` / `-oj` / `-ojf` | Output formats (also `-ocsv`, `-olrc`) |
| `-am FNAME` | CTC aligner GGUF for word-level timestamps on LLM backends |
| `-tp F` / `-bs N` | Sampling temperature / beam search width |
| `-n N` / `--frequency-penalty F` | Generated-token cap / opt-in repeated-token penalty for supported autoregressive ASR backends |
| `-l auto` / `--detect-language` | LID pre-step for backends without native lang detect |
| `--hotwords "A,B,C"` | Contextual biasing — boost named terms during CTC/TDT decode or LLM prompt |
| `-ck N` | Fallback chunk size when VAD is off (default 30 s) |
| `--list-backends` | Print the capability matrix and exit |

**See [`docs/cli.md`](docs/cli.md)** for the full reference: every
flag, VAD details, CTC alignment workflow, output JSON layout, the
auto-download registry, and supported audio formats. **See
[`docs/bindings.md`](docs/bindings.md)** for Python / Rust / Dart /
Go / Java / JavaScript / Ruby / mobile.

---

## Architecture, contributing, regression matrix

CrispASR is structured as a stable C-ABI in `src/` (every algorithm:
VAD, diarize, LID, alignment, cache, registry) consumed by all
language wrappers, with thin presentation layers in `examples/cli/`.
Per-model runtimes live in `src/{whisper,parakeet,canary,...}.cpp`,
sharing primitives from `src/core/` (mel, ffn, attention, GGUF
loader, FastConformer / Conformer / Granite-LLM blocks, etc.).

- **[`docs/architecture.md`](docs/architecture.md)** — full layered
  layout, file-by-file tour of `src/` and `examples/cli/`,
  per-backend internals table, regression discipline.
- **[`docs/contributing.md`](docs/contributing.md)** — adding a new
  backend in five files, clang-format-18 setup, the
  `crispasr-diff` PyTorch-ground-truth workflow, and the
  TTS audio-cosine-vs-reference regression target.
- **[`docs/regression-matrix.md`](docs/regression-matrix.md)** —
  `tools/test-all-backends.py` capability tiers, cache modes
  (`keep` / `ephemeral`), `--skip-missing` for CI.

**Shared libraries** (cross-repo with CrispEmbed):
- `crisp_audio/` — Whisper-shape audio encoder (Conv-stem + Transformer)
- `crisp_punc/` — punctuation restoration (FireRedPunc + PCS)
- `crisp_lid/` — text-based language identification (fastText + CLD3)
- `crisp_truecase/` — truecasing (statistical + CRF + BiLSTM)

Both are self-contained static libraries with CMakeLists.txt. CrispEmbed
links them via `add_subdirectory(../CrispASR/crisp_*/)`; CrispASR uses
them directly. If the shared dir is absent, both repos fall back to local
copies of the source files.

For benchmarks see [`PERFORMANCE.md`](PERFORMANCE.md); for the
session-by-session port log and the bug-class lessons, see
[`LEARNINGS.md`](LEARNINGS.md).

---

## Quantize models

`build/bin/crispasr-quantize` is a single, model-agnostic GGUF
re-quantization tool that works across all supported model families
(Whisper, Parakeet, Canary, Cohere, Voxtral, Qwen3, Granite, Wav2Vec2,
MiMo-ASR, GLM-ASR, Moonshine, VibeVoice, Kokoro, Qwen3-TTS, …):

```bash
./build/bin/crispasr-quantize input.gguf output.gguf q4_k
```

**See [`docs/quantize.md`](docs/quantize.md)** for the full guide:
supported quant types, K-quant alignment fallback, recommended quant
per backend, and worked examples for each architecture.

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
- **NVIDIA NeMo** — parakeet-tdt-{0.6b-v2,0.6b-v3,1.1b}, parakeet-tdt_ctc-{110m,1.1b,0.6b-ja}, parakeet-ctc-{0.6b,1.1b}, canary-1b-v2, canary-ctc aligner, and the FastConformer-CTC family (stt_en_fastconformer_ctc_{large,xlarge,xxlarge})
- **Cohere** — cohere-transcribe-03-2026
- **Qwen team (Alibaba)** — Qwen3-ASR-0.6B, Qwen3-ASR-1.7B, Qwen3-ForcedAligner-0.6B
- **Mistral AI** — Voxtral Mini 3B and 4B Realtime
- **IBM Granite team** — Granite Speech 3.2-8b, 3.3-2b, 3.3-8b, 4.0-1b
- **Meta / wav2vec2** — wav2vec2 CTC models (XLSR-53 English, German, multilingual via any Wav2Vec2ForCTC checkpoint)
- **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)** — optional diarization via subprocess (ONNX models)
- **[Silero](https://github.com/snakers4/silero-vad)** — VAD (native GGUF) and language identification (native GGUF, 95 languages)
- **[pyannote](https://github.com/pyannote/pyannote-audio)** — speaker diarization segmentation (native GGUF port)
- **[miniaudio](https://miniaud.io/)** + **[stb_vorbis](https://github.com/nothings/stb)** + **[libopus/opusfile](https://opus-codec.org/)** — embedded/linked audio decoders (WAV/MP3/FLAC/AIFF/OGG/Opus, no ffmpeg; AAC/M4A/ALAC via Apple AudioToolbox)
- **[Claude Code](https://claude.ai/claude-code)** (Anthropic) — significant portions of the crispasr integration layer, all model converters, and the FastConformer/attention/mel/FFN/BPE core helpers were co-authored with Claude

---

## License

Same as upstream whisper.cpp: **MIT**.

Per-model weights are covered by their respective HuggingFace model licenses (see [Supported backends](#supported-backends)). The `crispasr` binary itself links model runtimes that are mostly permissively licensed (MIT / Apache-2.0 / CC-BY-4.0 for weights).
