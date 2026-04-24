# CrispASR

**One C++ binary, twelve ASR model families, zero Python dependencies.**

CrispASR is a fork of [whisper.cpp](https://github.com/ggml-org/whisper.cpp) that extends the familiar `whisper-cli` interface into a **unified speech recognition tool** called `crispasr`, backed by full ggml C++ runtimes for major open-weights ASR architectures. One build, one binary, one consistent CLI — pick the backend at the command line or let CrispASR auto-detect it from your GGUF file.

```console
$ crispasr -m ggml-base.en.bin          -f samples/jfk.wav        # OpenAI Whisper
$ crispasr -m parakeet-tdt-0.6b.gguf    -f samples/jfk.wav        # NVIDIA Parakeet
$ crispasr -m canary-1b-v2.gguf         -f samples/jfk.wav        # NVIDIA Canary
$ crispasr -m voxtral-mini-3b-2507.gguf -f samples/jfk.wav        # Mistral Voxtral
$ crispasr --backend qwen3 -m auto      -f samples/jfk.wav        # -m auto downloads
```

No Python. No PyTorch. No separate per-model binary. No `pip install`. Just one C++ binary and a GGUF file.

### Ecosystem

| Project | What it does |
|---|---|
| **[CrispASR](https://github.com/CrispStrobe/CrispASR)** | This repo — C++ speech recognition engine. 11 backends, CLI + HTTP server + C-ABI + Python/Rust/Dart bindings. |
| **[CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver)** | Cross-platform Flutter transcription app built on CrispASR. Desktop + mobile, all 10 backends, model browser with download queue, mic capture, SRT/VTT/JSON export, diarization, batch processing. Fully offline. |
| **[CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)** | Text embedding engine via ggml — same philosophy as CrispASR but for retrieval. 10 architectures (XLM-R, Qwen3-Embed, Gemma3, ModernBERT, ...), dense + sparse + ColBERT + reranking. 9.5x faster than ONNX on CPU, GPU via CUDA/Metal/Vulkan. Python/Rust/Dart bindings. |
| **[Susurrus](https://github.com/CrispStrobe/Susurrus)** | Python ASR GUI with 9 backends (faster-whisper, mlx-whisper, voxtral, insanely-fast-whisper, ...). The Python counterpart to CrispASR's C++ approach. |

---

## Table of contents

- [Supported backends](#supported-backends)
- [Feature matrix](#feature-matrix)
- [Install & build](#install--build)
- [Quick start](#quick-start)
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
- [Credits](#credits)

---

## Supported backends

| Backend | Model | Architecture | Languages | License |
|---|---|---|---|---|
| **whisper** | [`ggml-base.en.bin`](https://huggingface.co/ggerganov/whisper.cpp/) and all OpenAI Whisper variants | Encoder-decoder transformer | 99 | MIT |
| **whisper** | [`distil-whisper/distil-large-v3`](https://huggingface.co/cstr/distil-large-v3-GGUF) | Distilled Whisper: 32L encoder + 2L decoder (6.3x faster, 513 MB Q5_0) | English | MIT |
| **parakeet** | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | FastConformer + TDT | 25 EU (auto-detect) | CC-BY-4.0 |
| **canary** | [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) | FastConformer + Transformer decoder | 25 EU (explicit `-sl/-tl`) | CC-BY-4.0 |
| **cohere** | [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Conformer + Transformer | 13 | Apache-2.0 |
| **granite** | [`ibm-granite/granite-speech-{3.2-8b,3.3-2b,3.3-8b,4.0-1b}`](https://huggingface.co/ibm-granite/granite-speech-3.3-2b) | Conformer + BLIP-2 Q-Former + Granite LLM (μP) | en fr de es pt ja | Apache-2.0 |
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
| **omniasr** | [`facebook/omniASR-CTC-{300M,1B}`](https://huggingface.co/cstr/omniASR-CTC-1B-GGUF) | wav2vec2-style CNN + 24–48L transformer + CTC head | **1600+** | Apache-2.0 |
| **omniasr** | [`omniASR-LLM-300M-v2`](https://huggingface.co/cstr/omniasr-llm-300m-v2-GGUF) | Same encoder + 12L LLaMA decoder (SwiGLU, RoPE); autoregressive, best quality | **1600+** | Apache-2.0 |
| **vibevoice** | [`microsoft/VibeVoice-ASR`](https://huggingface.co/cstr/vibevoice-asr-GGUF) | σ-VAE ConvNeXt encoders + Qwen2.5-7B decoder; timestamps, diarization, hotwords | 50+ | MIT |

**Post-processing models** (work with all backends):

| Model | Task | Architecture | Languages | License | HuggingFace |
|---|---|---|---|---|---|
| **FireRedPunc** | Punctuation restoration | BERT-base (12L, d=768), 5 classes | Chinese + English | Apache-2.0 | [`cstr/fireredpunc-GGUF`](https://huggingface.co/cstr/fireredpunc-GGUF) |
| **fullstop-punc** | Punctuation restoration | XLM-RoBERTa-large (24L, d=1024), 6 classes | EN, DE, FR, IT | MIT | [`cstr/fullstop-punc-multilang-GGUF`](https://huggingface.co/cstr/fullstop-punc-multilang-GGUF) |

All eighteen runtimes share ggml-based inference. The speech-LLM backends (**qwen3**, **voxtral**, **voxtral4b**, **granite**, **glm-asr**, **kyutai-stt**) inject audio encoder frames directly into an autoregressive language model's input embeddings, instead of using a dedicated CTC/transducer/seq2seq decoder. The **fastconformer-ctc** backend hosts the NeMo FastConformer-CTC standalone ASR family (small through xxlarge, same architecture as the canary aligner) with greedy CTC decoding.

## Feature matrix

Run `crispasr --list-backends` to see it live. Each backend declares capabilities at runtime; if you ask for a feature the selected backend does not support, CrispASR prints a warning and silently ignores the flag.

<!-- Generated from `crispasr --list-backends` + cross-cutting features. -->

| Feature | whisper | parakeet | canary | cohere | granite | voxtral | voxtral4b | qwen3 | fc-ctc | wav2vec2 | glm-asr | kyutai-stt | firered | moonshine | omniasr |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Native timestamps | ✔ | ✔ | ✔ | ✔ | | | | | | | | | | | |
| CTC timestamps | | | ✔ | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | ✔ | | ✔ |
| Word-level timing | ✔ | ✔ | ✔ | ✔ | `-am` | `-am` | `-am` | `-am` | `-am` | `-am` | `-am` | | `-am` | `-am` | `-am` |
| Per-token confidence | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | | | | | |
| Language auto-detect | ✔ | ✔ | LID | LID | LID | LID | LID | ✔ | LID | LID | ✔ | LID | LID | LID | LID |
| Speech translation | ✔ | | ✔ | | ✔ | ✔ | | ✔ | | | | | | | |
| Speaker diarization | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | all | all | all | all | all | all | all |
| Grammar (GBNF) | ✔ | | | | | | | | | | | | | | |
| Temperature sampling | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | | | | |
| Beam search | ✔ | | | | | | | | | | | | ✔ | | |
| Best-of-N (`--best-of`) | ✔ | | | | ✔ | ✔ | ✔ | ✔ | | | | | | | |
| Flash attention | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | ✔ | ✔ | ✔ | ✔ |
| KV cache | ✔ | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | ✔ | ✔ | ✔ | * |
| Punctuation toggle | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | | | | | |
| Punc restoration | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp | pp |
| Source / target language | | | ✔ | | ✔ | ✔ | | ✔ | | | | | | | |
| Audio Q&A (`--ask`) | | | | | * | ✔ | | * | | | | | | | |
| Streaming (`--stream/--mic/--live`) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Auto-download (`-m auto`) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | | ✔ | ✔ | ✔ | ✔ | ✔ |

**Key:** ✔ = native/built-in, `-am` = via CTC forced aligner (`-am canary-ctc-aligner.gguf` or `-am qwen3-forced-aligner.gguf`), **LID** = via external language identification pre-step (`-l auto`), **all** = via `--diarize` post-step (not declared by backend but always available), **pp** = via `--punc-model` post-processor (FireRedPunc or fullstop-punc), * = omniasr-LLM has KV cache (CTC variant does not).

**Speaker diarization** is available for all backends as a post-processing step via `--diarize`:
- `--diarize-method energy` / `xcorr` — stereo-only, no extra deps
- `--diarize-method pyannote` — native GGUF runtime (no Python, no sherpa-onnx). Pass `--sherpa-segment-model pyannote-v3-seg.gguf` for the pyannote v3 segmentation model. Falls back to sherpa subprocess for `.onnx` models.
- `--diarize-method sherpa` / `ecapa` — calls an externally-installed [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) subprocess with speaker embedding models.
- `--diarize-method vad-turns` — mono-friendly, assigns speaker labels at gap boundaries

**Language identification** for backends without native LID: `--lid-backend whisper` (default, 75 MB ggml-tiny.bin), `--lid-backend silero` (native GGUF, 16 MB, 95 languages), or `--lid-backend firered` (FireRedLID, 1.7 GB, 120 languages — Conformer encoder + Transformer decoder).

**Voice activity detection**: `--vad` uses the default Silero VAD (~885 KB, auto-downloaded). Each VAD segment is transcribed independently, producing separate SRT/VTT entries with correct timestamps. Use `--vad --split-on-punct` for best subtitle output. Pass `--vad-model <path>` to use an alternative model — if the filename contains "firered-vad", FireRedVAD (DFSMN, 2.4 MB) is used automatically.

**Punctuation restoration** (`--punc-model`): CTC-based backends (wav2vec2, omniasr, fastconformer-ctc, firered-asr) output **lowercase text without punctuation**. Add `--punc-model <model>.gguf` to restore punctuation and capitalization via a BERT-based post-processor. Works with **all** backends — LLM backends already produce punctuated output, so the punc model is a no-op for those. Two models are supported:

| Model | License | Languages | Size (Q4_K) | HuggingFace |
|---|---|---|---|---|
| **FireRedPunc** | Apache-2.0 | Chinese + English | 56 MB (Q4_K) / 104 MB (Q8_0) | [`cstr/fireredpunc-GGUF`](https://huggingface.co/cstr/fireredpunc-GGUF) |
| **fullstop-punctuation-multilingual** | MIT | English, German, French, Italian | 254 MB (Q4_K) | [`cstr/fullstop-punc-multilang-GGUF`](https://huggingface.co/cstr/fullstop-punc-multilang-GGUF) |

FireRedPunc uses BERT-base (12L, d=768, 5 classes: comma/period/question/exclamation). Fullstop uses XLM-RoBERTa-large (24L, d=1024, 6 classes including dash and colon). Both auto-detect Latin script and output ASCII punctuation for English/European text.

```bash
# CTC backend + punc model → proper output
crispasr --backend wav2vec2 -m wav2vec2.gguf --punc-model fireredpunc-q8_0.gguf -f audio.wav
# Output: "And so my fellow americans, ask not what your country can do for you."

# Multilingual punc (German/French/Italian)
crispasr --backend omniasr -m omniasr.gguf --punc-model fullstop-punc-q4_k.gguf -f audio_de.wav
```

Also available via the Python/Rust/Dart wrappers:
```python
import crispasr
with crispasr.PuncModel("fireredpunc-q8_0.gguf") as punc:
    text = punc.process("and so my fellow americans ask not")
    # -> "And so my fellow americans, ask not..."
```

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
| Multilingual + word timestamps + small + fast | **parakeet** |
| Multilingual with **explicit language control** | **canary** |
| **Speech translation** (X→en or en→X) | **canary** |
| **30 languages + Chinese dialects** | **qwen3** |
| **Realtime streaming ASR** (<500 ms latency) | **voxtral4b** |
| Highest-quality offline speech-LLM | **voxtral** |
| Apache-licensed speech-LLM | **granite**, **voxtral**, **voxtral4b**, **qwen3** |
| **Lightweight CTC-only** (single-language, no LLM) | **wav2vec2**, **fc-ctc** |

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

Three LID providers are available:

- `--lid-backend whisper` (default) — uses a small multilingual ggml-*.bin model via the whisper.cpp C API. Auto-downloads ~75 MB on first use. 99 languages.
- `--lid-backend silero` — native GGUF port of Silero's 95-language classifier. 16 MB F32, pure C++. Faster and smaller than whisper-tiny but slightly less accurate on long audio (>20s).
- `--lid-backend ecapa` — **recommended**: ECAPA-TDNN (Apache-2.0). Purpose-built for language ID. Very high accuracy on TTS benchmark. Two variants via `--lid-model`:
  - [`cstr/ecapa-lid-107-GGUF`](https://huggingface.co/cstr/ecapa-lid-107-GGUF) — VoxLingua107, 43 MB F16, 107 languages, ISO codes (en, de, ...). **Default.**
  - [`cstr/ecapa-lid-commonlanguage-GGUF`](https://huggingface.co/cstr/ecapa-lid-commonlanguage-GGUF) — CommonLanguage, 40 MB F16, 45 languages, full names (English, German, ...).
- `--lid-backend firered` — FireRedLID (Conformer encoder + Transformer decoder). Q4_K (544 MB), 120 languages including Chinese dialects. Slower but covers more languages.

Two VAD providers are available:

- **Silero VAD** (default) — ~885 KB, auto-downloaded via `--vad`. Industry-standard, well-tested.
- **FireRedVAD** — DFSMN-based, 2.4 MB. Pass `--vad-model firered-vad.gguf` to use. Detected automatically from filename.

Pass `--lid-backend off` to skip LID entirely.

---

## Install & build

### Prerequisites

- C++17 compiler (GCC 10+, Clang 12+, MSVC 19.30+)
- CMake 3.14+
- Optional: `libavformat`/`libavcodec`/`libavutil`/`libswresample` for Opus/M4A ingestion (`WHISPER_FFMPEG=ON`)
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
cmake --build build -j$(nproc) --target whisper-cli
```

The target is named `whisper-cli` for CMake compatibility; the produced binary is `build/bin/crispasr` with a `build/bin/whisper-cli` alias next to it. Either name works.

### Windows (convenience scripts)

Two batch scripts handle the Windows build without requiring a pre-opened Developer Command Prompt. They use `vswhere.exe` to locate Visual Studio 2022 automatically, call `vcvars64.bat`, then drive CMake + Ninja.

#### `build-windows.bat` — CPU build

```cmd
build-windows.bat
```

Produces `build\bin\crispasr.exe`. Extra CMake flags can be appended:

```cmd
build-windows.bat -DWHISPER_CURL=ON          :: enable libcurl fallback
build-windows.bat -DGGML_CUDA=ON             :: NVIDIA GPU (CUDA must be installed)
```

What it does:
1. Locates `vswhere.exe` under `%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\`
2. Finds the latest VS 2022 installation that includes the VC++ toolchain
3. Calls `vcvars64.bat` to initialize the 64-bit MSVC environment
4. Runs `cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release [extra flags]`
5. Builds the `whisper-cli` target → `build\bin\crispasr.exe`

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

cmake -B build-ffmpeg -DCMAKE_BUILD_TYPE=Release -DWHISPER_FFMPEG=ON
cmake --build build-ffmpeg -j$(nproc) --target whisper-cli
```

> **Upstream bug warning.** `.m4a` / `.mp4` / `.webm` containers currently crash `whisper.cpp`'s ffmpeg integration. For those formats, pre-convert to WAV:
> ```bash
> ffmpeg -i input.opus -ar 16000 -ac 1 -c:a pcm_s16le -y /tmp/audio.wav
> ```
> Bare-codec `.opus` files work fine with `WHISPER_FFMPEG=ON`.

---

## Quick start

### Whisper (historical path, byte-identical to upstream whisper-cli)

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

# ...or use the familiar whisper-cli flag:
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

### Streaming & live transcription

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

Streaming works with all 11 backends. The `--stream-step` (default 3s), `--stream-length` (default 10s), and `--stream-keep` (default 200ms overlap) flags control the sliding window.

### Server mode (persistent model, HTTP API)

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

To require API keys, pass a comma-separated list with `--api-keys` or set `CRISPASR_API_KEYS`. Protected endpoints accept either `Authorization: Bearer <key>` or `X-API-Key: <key>`. `/health` remains public for container health checks.

```bash
crispasr --server -m model.gguf --api-keys key-one,key-two

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

`crispasr` extends upstream `whisper-cli`'s argument set with a handful of backend-dispatch flags. Every historical whisper flag still works — when you don't pass `--backend`, whisper is the default.

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

`--diarize`, `-tdrz`/`--tinydiarize`, `--carry-initial-prompt`, `-dtw`, `-fa`/`-nfa`, `-suppress-regex`, `-suppress-nst`, and the full upstream `whisper-cli --help` list.

---

## Voice Activity Detection (VAD)

Every non-whisper backend uses the Silero VAD model to segment long audio into speech regions, **stitch them into a single contiguous buffer** (with 0.1s silence gaps, matching whisper.cpp's internal approach), transcribe in one pass, and remap timestamps back to original-audio positions. This preserves cross-segment context and avoids boundary artifacts. Short VAD segments (< 3s) are auto-merged, and oversized segments are split at `--chunk-seconds` boundaries. Whisper handles VAD internally via `wparams.vad`.

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

# Transcribe (all 10 backends via one session object)
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

Reference application: **[CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver)** — a cross-platform Flutter desktop/mobile transcription app built on `package:crispasr`. Ships with model browser + downloader (all 10 backends + quants), drag-and-drop files, mic capture, SRT/VTT/TXT export, per-run performance metrics, and full en/de i18n. The v0.1.7 release uses the new `transcribeVad` path so every non-whisper backend benefits from stitched Silero VAD with zero CrisperWeaver-side work.

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

| Format | Default build | `WHISPER_FFMPEG=ON` |
|---|:---:|:---:|
| WAV / FLAC / MP3 / OGG | ✔ | ✔ |
| `.opus` | ✗ | ✔ |
| `.m4a` / `.mp4` / `.webm` | ✗ | ⚠ upstream crash, pre-convert |
| `.aiff` / `.wma` / raw PCM | ✗ | pre-convert |

For anything in the bottom half, the reliable path is `ffmpeg -i in.X -ar 16000 -ac 1 -c:a pcm_s16le out.wav` then pass the WAV.

---

## Architecture

CrispASR is structured around three layers on top of whisper.cpp. The
split between `src/` (library) and `examples/cli/` (presentation) is
deliberate: **every algorithm** — VAD, diarization, LID, CTC alignment,
HF download/cache, model registry — lives in `src/` behind a stable
C-ABI (`src/crispasr_c_api.cpp`), and every consumer (CLI, Dart, Python,
Rust) reaches it through the same symbols. The CLI keeps only
presentation + UX policy.

```
┌───────────────────────────────────────────────────────────────────┐
│ examples/cli/cli.cpp (the crispasr binary)                        │
│   Parses whisper-cli args, dispatches to backend when --backend   │
│   is set or GGUF arch is non-whisper; otherwise runs whisper_full │
│   unchanged                                                        │
├───────────────────────────────────────────────────────────────────┤
│ examples/cli/crispasr_*_cli.{h,cpp}                               │
│   Thin CLI shims for policy only — auto-download, TTY prompts,    │
│   sherpa-ONNX subprocess fallbacks. Delegate the algorithmic      │
│   work to the shared library below.                                │
├───────────────────────────────────────────────────────────────────┤
│ src/crispasr_c_api.cpp — C-ABI (shared with Dart / Python / Rust) │
│   crispasr_vad.{h,cpp}           Silero VAD + whisper.cpp-style   │
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
| `crispasr_vad.{h,cpp}` | Silero VAD slicing + whisper.cpp-style stitching with timestamp remapping. Used by `crispasr_session_transcribe_vad`. |
| `crispasr_diarize.{h,cpp}` | Four diarizers: energy (stereo), xcorr (stereo, TDOA), vad-turns (mono, timing), pyannote (mono, GGUF). |
| `crispasr_lid.{h,cpp}` | whisper-tiny + silero-native language ID with process-wide whisper-context cache. |
| `crispasr_aligner.{h,cpp}` | canary-CTC + Qwen3-ForcedAligner forced alignment behind one entry point; filename-based dispatch. |
| `crispasr_cache.{h,cpp}` | WinHTTP / curl / wget download into `~/.cache/crispasr/`; zombie-file detection. |
| `crispasr_model_registry.{h,cpp}` | Backend → canonical GGUF URL table; fuzzy filename lookup for "did you mean …?" hints. |
| `whisper_params.h` | Shared params struct (extracted from cli.cpp, extended). |

### `examples/cli/` — presentation + policy

| File | Role |
|---|---|
| `cli.cpp` | whisper-cli entry point, extended with `--backend` dispatch branch. |
| `crispasr_backend.{h,cpp}` | `CrispasrBackend` abstract class, capability bitmask, factory, GGUF auto-detect. |
| `crispasr_backend_{parakeet,canary,cohere,granite,voxtral,voxtral4b,qwen3,fastconformer_ctc,wav2vec2}.cpp` | Per-backend thin wrapper over each model's C API. |
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
| `core/attention.h` | Llama-style self-attention with NEOX RoPE + GQA + flash-attn | voxtral (more coming) |
| `core/gguf_loader.{h,cpp}` | 8× identical two-pass GGUF load + mmap + tensor-map build | all non-whisper models |

`core_mel::Params` spans both algorithm clusters: the NeMo family (`ln` + per-mel z-score + `(T, n_mels)` layout) and the HF/Whisper family (`log10` + global clip normalization + `(n_mels, T)` layout), with knobs for `LogGuard` (add-epsilon vs max-clip), `MatmulPrecision` (`Float` vs `Double`), `FbLayout` (`MelsFreqs` vs `FreqsMels`), `drop_last_frame` / `drop_first_frame_if_odd`, and `pad_to_T`.

`core_gguf::WeightLoad` owns the `ggml_context`, the `ggml_backend_buffer_t`, and the `std::map<std::string, ggml_tensor*>` in one struct that models `std::move()` into their own state. The mmap path has a `pread`/`fseek` fallback for filesystems that don't support mmap.

### Whisper is the reference implementation

`src/whisper.cpp` is **intentionally not migrated** to `src/core/` (yet) — it's (for the time being) the battle-tested reference and the `crispasr -m ggml-base.en.bin …` code path is byte-identical to upstream `whisper-cli`. This guarantee is a test gate: every CrispASR commit that touches the CLI is checked against it.

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
| omniasr | wav2vec2 enc + CTC/LLM | ✔ | ✔ | CTC:— LLM:✔ | CPU | gguf_loader, kv_self_attn, swiglu |
| vibevoice | σ-VAE + Qwen2 7B | ✔ | ✔ | ✔ | CUDA/Metal | gguf_loader |

**Architecture families:**
- **Feedforward CTC** (wav2vec2, omniasr-CTC, fc-ctc, firered-asr): No decoder, no KV cache. Fastest. No native punctuation.
- **Encoder-decoder** (whisper, canary, cohere, moonshine): Cross-attention KV cache, autoregressive text decoder.
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

```bash
./build/bin/crispasr --backend yourmodel -m model.gguf -f samples/jfk.wav -np > before.txt
# ... make changes ...
cmake --build build --target whisper-cli
./build/bin/crispasr --backend yourmodel -m model.gguf -f samples/jfk.wav -np > after.txt
diff before.txt after.txt && echo BIT-IDENTICAL
```

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
hooks and returns a dict `{stage_name: ndarray}`. See
`tools/reference_backends/qwen3.py` and `voxtral.py` for worked
examples; `voxtral4b.py` and `granite.py` are stubs with inline notes
on what to port from the legacy `models/*-dump-*.py` scripts.

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

### What's done

- **Ten backends shipped, all runtime-ready** through the unified `crispasr_session_*` C-ABI: whisper, parakeet, canary, cohere, granite, fastconformer-ctc, canary_ctc, voxtral (3B), voxtral4b, qwen3, wav2vec2.
- **Unified language bindings** — Dart (`package:crispasr` 0.4.x), Python (`crispasr.Session`), Rust (`crispasr::Session`). One C-ABI, three wrappers, all 10 backends reachable from each.
- **Cross-platform audio decoding** via `crispasr_audio_load` (embedded miniaudio + stb_vorbis). WAV / MP3 / FLAC / OGG Vorbis without an ffmpeg dependency; same loader used from CLI, server, and all language bindings.
- **`src/core/` shared library** (`crispasr-core`) —
  - `core/mel` ✅ all non-whisper models migrated (including granite's stacked-2-frame variant)
  - `core/ffn` ✅ all 4 SwiGLU consumers migrated
  - `core/gguf_loader` ✅ all non-whisper models migrated
  - `core/attention` ✅ `encoder_self_attn()` shared; voxtral + voxtral4b migrated, qwen3/granite remaining
  - `core/greedy_decode` ✅ all 4 LLM backends migrated
- **Ground-truth diff infrastructure** — `tools/dump_reference.py` plug-in Python modules (qwen3, voxtral, voxtral4b, granite) + `crispasr_diff::Ref` C++ loader + `crispasr-diff` CLI. Bit-identical regression on `samples/jfk.wav` gates every commit.
- **HTTP server** with persistent model, hot-swap, streaming, mic input, per-token alternatives.
- **OpenAI-compatible API** — `POST /v1/audio/transcriptions` with all five response formats, drop-in replacement for the OpenAI Whisper API.
- **Reference Flutter app** — [CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver), cross-platform desktop/mobile transcription app built on `package:crispasr`.
- **VAD stitching** — Silero VAD segments are stitched into a single buffer with timestamp remapping (matching whisper.cpp's internal approach). Short segments merged, oversized segments split.
- **Best-of-N sampling** — all 4 LLM backends (voxtral, voxtral4b, qwen3, granite) support `--best-of N` with temperature sampling.
- **Audio Q&A** — voxtral 3B supports `--ask "question"` for audio understanding beyond transcription.
- **Integration test suites** — Python (13 tests), Rust (5 tests), Dart (9 tests) covering Session API, transcription, timestamps, edge cases.
- **Performance**: 3.8x faster than [antirez/voxtral.c](https://github.com/antirez/voxtral.c) on the same CPU for Voxtral 4B. See `PERFORMANCE.md` for benchmarks.

### Near-term

- CMake target rename: `whisper-cli` → `crispasr` across CI/tests/scripts (~50 references).
- Shaw relative position embeddings for granite encoder graph (currently uses block mask only).
- Full LIS monotonicity fix for qwen3 forced aligner (currently forward-clamp only).
- Chunk-overlap experiments for fixed chunking (VAD is the recommended path, but the fixed-chunk fallback could benefit from 2s overlap like Susurrus does).

### Long-term

- WebSocket streaming mode for real-time transcription over HTTP.
- Model-agnostic batch processing with a progress bar.
- TTS subcommand (`crispasr speak …`) via voxtral-rs.

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

## Credits

- **[whisper.cpp](https://github.com/ggml-org/whisper.cpp)** — the ggml inference engine and the whisper runtime this fork is built on
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
