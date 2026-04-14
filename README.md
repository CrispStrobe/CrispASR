# CrispASR

**One C++ binary, ten ASR model families, zero Python dependencies.**

CrispASR is a fork of [whisper.cpp](https://github.com/ggml-org/whisper.cpp) that extends the familiar `whisper-cli` interface into a **unified speech recognition tool** called `crispasr`, backed by full ggml C++ runtimes for major open-weights ASR architectures. One build, one binary, one consistent CLI — pick the backend at the command line or let CrispASR auto-detect it from your GGUF file.

```console
$ crispasr -m ggml-base.en.bin          -f samples/jfk.wav        # OpenAI Whisper
$ crispasr -m parakeet-tdt-0.6b.gguf    -f samples/jfk.wav        # NVIDIA Parakeet
$ crispasr -m canary-1b-v2.gguf         -f samples/jfk.wav        # NVIDIA Canary
$ crispasr -m voxtral-mini-3b-2507.gguf -f samples/jfk.wav        # Mistral Voxtral
$ crispasr --backend qwen3 -m auto      -f samples/jfk.wav        # -m auto downloads
```

No Python. No PyTorch. No separate per-model binary. No `pip install`. Just one C++ binary and a GGUF file.

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
| **parakeet** | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | FastConformer + TDT | 25 EU (auto-detect) | CC-BY-4.0 |
| **canary** | [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) | FastConformer + Transformer decoder | 25 EU (explicit `-sl/-tl`) | CC-BY-4.0 |
| **cohere** | [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Conformer + Transformer | 13 | Apache-2.0 |
| **granite** | [`ibm-granite/granite-speech-{3.2-8b,3.3-2b,3.3-8b,4.0-1b}`](https://huggingface.co/ibm-granite/granite-speech-3.3-2b) | Conformer + BLIP-2 Q-Former + Granite LLM (μP) | en fr de es pt ja | Apache-2.0 |
| **fastconformer-ctc** | [`nvidia/stt_en_fastconformer_ctc_large`](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large) | FastConformer + CTC (NeMo family, all sizes) | en | CC-BY-4.0 |
| **voxtral** | [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Whisper encoder + Mistral 3B LLM, audio-token injection | 8 | Apache-2.0 |
| **voxtral4b** | [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Causal RoPE+SwiGLU encoder + 3.4B LLM with adaptive RMSNorm + sliding window | 13, realtime streaming | Apache-2.0 |
| **qwen3** | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Whisper-style audio encoder + Qwen3 0.6B LLM | 30 + 22 Chinese dialects | Apache-2.0 |
| **wav2vec2** | [`jonatasgrosman/wav2vec2-large-xlsr-53-english`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) | CNN + 24-layer transformer + CTC head (any Wav2Vec2ForCTC) | per-model (en, de, multilingual available) | Apache-2.0 |

All eleven runtimes share ggml-based inference. The speech-LLM backends (**qwen3**, **voxtral**, **voxtral4b**, **granite**) inject audio encoder frames directly into an autoregressive language model's input embeddings, instead of using a dedicated CTC/transducer/seq2seq decoder. The **fastconformer-ctc** backend hosts the NeMo FastConformer-CTC standalone ASR family (small through xxlarge, same architecture as the canary aligner) with greedy CTC decoding.

## Feature matrix

Run `crispasr --list-backends` to see it live. Each backend declares capabilities at runtime; if you ask for a feature the selected backend does not support, CrispASR prints a warning and silently ignores the flag.

<!-- Generated from `crispasr --list-backends` + cross-cutting features. -->

| Feature | whisper | parakeet | canary | cohere | granite | voxtral | voxtral4b | qwen3 | fc-ctc | wav2vec2 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Native timestamps | ✔ | ✔ | ✔ | ✔ | | | | | | |
| CTC timestamps | | | ✔ | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Word-level timing | ✔ | ✔ | ✔ | ✔ | `-am` | `-am` | `-am` | `-am` | | |
| Per-token confidence | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | |
| Language auto-detect | ✔ | ✔ | LID | LID | LID | LID | LID | ✔ | LID | LID |
| Speech translation | ✔ | | ✔ | | ✔ | ✔ | | ✔ | | |
| Speaker diarization | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | all | all |
| Grammar (GBNF) | ✔ | | | | | | | | | |
| Temperature sampling | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | |
| Beam search | ✔ | | | | | ✔ | | | | |
| Flash attention | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | |
| Punctuation toggle | | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | |
| Source / target language | | | ✔ | | ✔ | ✔ | | ✔ | | |
| Streaming (`--stream/--mic/--live`) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Auto-download (`-m auto`) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | | |

**Key:** ✔ = native/built-in, `-am` = via CTC forced aligner (`-am canary-ctc-aligner.gguf` or `-am qwen3-forced-aligner.gguf`), **LID** = via external language identification pre-step (`-l auto`), **all** = via `--diarize` post-step (not declared by backend but always available).

**Speaker diarization** is available for all backends as a post-processing step via `--diarize`:
- `--diarize-method energy` / `xcorr` — stereo-only, no extra deps
- `--diarize-method pyannote` — native GGUF runtime (no Python, no sherpa-onnx). Pass `--sherpa-segment-model pyannote-v3-seg.gguf` for the pyannote v3 segmentation model. Falls back to sherpa subprocess for `.onnx` models.
- `--diarize-method sherpa` / `ecapa` — calls an externally-installed [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) subprocess with speaker embedding models.
- `--diarize-method vad-turns` — mono-friendly, assigns speaker labels at gap boundaries

**Language identification** for backends without native LID: `--lid-backend whisper` (default, 75 MB ggml-tiny.bin) or `--lid-backend silero` (native GGUF, 16 MB, 95 languages — pass `--lid-model silero-lid-95.gguf`).

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

Two LID providers are available:

- `--lid-backend whisper` (default) — uses a small multilingual ggml-*.bin model via the whisper.cpp C API. Auto-downloads ~75 MB on first use. 99 languages.
- `--lid-backend silero --lid-model silero-lid-95.gguf` — native GGUF port of Silero's 95-language classifier. 16 MB F32, pure C++ (manual F32 forward pass, no Python). Faster and smaller than whisper-tiny but slightly less accurate on long audio (>20s).

Pass `--lid-backend off` to skip the pre-step entirely.

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

```cmd
:: Typical usage — VULKAN_SDK is picked up automatically
build-vulkan.bat

:: Override Vulkan SDK location explicitly
set VULKAN_SDK=C:\VulkanSDK\1.4.304.1
build-vulkan.bat
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
| `--lid-backend NAME` | LID provider: `whisper` (default, ships ggml-tiny.bin), `silero` (native GGUF, 95 languages, 16 MB), or `off` to disable |
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

Every non-whisper backend uses the Silero VAD model to pre-slice long audio into speech segments, transcribe each slice, and re-stitch the output with absolute timestamps. Whisper handles VAD internally via `wparams.vad`.

```bash
# Just pass --vad — the model is auto-downloaded on first use
./build/bin/crispasr --backend parakeet -m parakeet.gguf -f long_audio.wav \
    --vad -osrt

# Or point at an existing GGUF
./build/bin/crispasr --backend parakeet -m parakeet.gguf -f long_audio.wav \
    --vad-model ~/models/ggml-silero-v5.1.2.bin -osrt
```

The cached model lives at `~/.cache/crispasr/ggml-silero-v5.1.2.bin` (~885 KB). If you don't provide `--vad`, CrispASR falls back to fixed 30-second chunking (configurable via `-ck`). Encoder cost is O(T²) in the frame count, so for multi-minute audio you really want VAD.

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

## Auto-download (`-m auto`)

When you pass `-m auto` (or `-m default`), CrispASR downloads the default quantized model for the selected backend into `~/.cache/crispasr/` on first use. The registry:

| Backend | Download | Approx size |
|---|---|---|
| parakeet | `cstr/parakeet-tdt-0.6b-v3-GGUF` | ~467 MB |
| canary | `cstr/canary-1b-v2-GGUF` | ~600 MB |
| voxtral | `cstr/voxtral-mini-3b-2507-GGUF` | ~2.5 GB |
| voxtral4b | `cstr/voxtral-mini-4b-realtime-GGUF` | ~3.3 GB |
| granite | `cstr/granite-4.0-1b-speech-GGUF` | ~900 MB |

Downloads go through `curl` (preferred) with a `wget` fallback — **no Python, no libcurl link dependency**. Works identically on Linux, macOS, and Windows 10+ where `curl` ships in the base system. Models are cached by filename; re-running is a single `stat()` check.

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

CrispASR is structured around two new layers on top of whisper.cpp:

```
┌───────────────────────────────────────────────────────────────────┐
│ examples/cli/crispasr_*                                           │
│   Backend interface, factory, dispatch, VAD slicing,              │
│   common output writers, CTC aligner, auto-download, model-mgr    │
├───────────────────────────────────────────────────────────────────┤
│ examples/cli/cli.cpp (the crispasr binary)                        │
│   Parses whisper-cli args, dispatches to backend when --backend   │
│   is set or GGUF arch is non-whisper; otherwise runs whisper_full │
│   unchanged                                                        │
├───────────────────────────────────────────────────────────────────┤
│ src/{whisper,parakeet,canary,canary_ctc,cohere,qwen3_asr,         │
│      voxtral,voxtral4b,granite_speech}.cpp                        │
│   Per-model runtimes (public C APIs)                              │
├───────────────────────────────────────────────────────────────────┤
│ src/core/      ← NEW shared library: crispasr-core                │
│   mel.{h,cpp}          log-mel spectrogram (both NeMo + HF clusters)
│   ffn.h                SwiGLU + SiLU FFN helpers (header-only)    │
│   attention.h          Llama-style self-attention + flash-attn    │
│   gguf_loader.{h,cpp}  Unified GGUF open / weight mmap / lookup   │
├───────────────────────────────────────────────────────────────────┤
│ ggml                                                               │
└───────────────────────────────────────────────────────────────────┘
```

### `examples/cli/` — the dispatch layer

| File | Role |
|---|---|
| `cli.cpp` | whisper-cli entry point, extended with `--backend` dispatch branch |
| `whisper_params.h` | Shared params struct (extracted from cli.cpp, extended) |
| `crispasr_backend.{h,cpp}` | `CrispasrBackend` abstract class, capability bitmask, factory, GGUF auto-detect |
| `crispasr_backend_{parakeet,canary,cohere,granite,voxtral,voxtral4b,qwen3}.cpp` | Per-backend thin wrapper over each model's C API |
| `crispasr_vad.{h,cpp}` | Silero VAD slicing |
| `crispasr_output.{h,cpp}` | TXT/SRT/VTT/CSV/JSON/LRC writers on `crispasr_segment` |
| `crispasr_model_mgr.{h,cpp}` | `-m auto` via curl/wget shell-out |
| `crispasr_aligner.{h,cpp}` | canary_ctc forced alignment wrapper |
| `crispasr_llm_pipeline.h` | Templated audio-LLM pipeline (mel→encoder→prompt→KV decode) |
| `crispasr_run.cpp` | Top-level pipeline dispatch: resolve → detect → load → slice → transcribe → write |

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

### What's done (on `integrated_cli`)

- **Phase 1** — Unified CLI with backend dispatch, VAD, common output writers, CTC alignment, auto-download. 7 non-whisper backends wired (parakeet, canary, cohere, granite, voxtral, voxtral4b, qwen3). Whisper code path unchanged and byte-identical to upstream.
- **Phase 0** — `src/core/` shared library (`crispasr-core`):
  - `core/mel` ✅ **all 8** non-whisper models migrated (including granite's stacked-2-frame variant)
  - `core/ffn` ✅ 4 of 4 SwiGLU consumers migrated
  - `core/gguf_loader` ✅ all 8 non-whisper models migrated
  - `core/attention` ✅ 1 of ~6 LLM attention blocks migrated (voxtral)
  - `core/greedy_decode` ✅ **all 4** LLM backends migrated (voxtral, voxtral4b, qwen3, granite)
- **Ground-truth diff infrastructure** — `tools/dump_reference.py` with plug-in per-backend Python modules + `crispasr_diff::Ref` C++ loader + `crispasr-diff` CLI. Runs the C++ forward pass against PyTorch-dumped reference activations and reports cosine-similarity / max-abs / RMS / top-1-argmax at every named stage. Plug-in modules: qwen3, voxtral, voxtral4b, granite (all 4 LLM backends).
- **Bit-identical regression** on `samples/jfk.wav` is the gate for every commit, and the ground-truth gate is the gate for every new backend. ~1,000 lines of duplicated boilerplate removed from `src/`.

### Near-term (next sessions)

- `core/attention.h` variants for persistent-KV-cache models (qwen3, voxtral4b, granite LLM) — needs a separate helper signature
- `core/attention.h` variants for Q/K norm (qwen3), biases + no RoPE (voxtral audio), µP scale tricks (granite)
- `core/greedy_decode.h` — unified LLM decode loop, retires the CLI-side `crispasr_llm_pipeline.h`
- `core/mel::Params::stacked_frames` — for granite's 2-frame stacked output (the last holdout on mel extraction)
- `cli.cpp` `output_json` / `output_wts` refactor to consume `crispasr_segment` — unblocks `backend-whisper.cpp` wrapper so whisper routes through the same dispatch as everything else
- Delete the per-model `examples/*-main/` directories once the unified CLI has shipped and regression-tested in CI

### Long-term

- Microphone input (SDL2-based, pattern exists in `examples/stream/`)
- HTTP / WebSocket server mode (pattern in `examples/server/`)
- Model-agnostic batch processing with a progress bar
- TTS subcommand (`crispasr speak …`) via voxtral-rs

---

## GPU backend selection

All backends use `ggml_backend_init_best()` which automatically picks the highest-priority compiled backend: CUDA > Metal > Vulkan > CPU. To force a specific backend:

```bash
# Force Vulkan even when CUDA is available
crispasr --gpu-backend vulkan -m model.gguf -f audio.wav

# Force CPU (useful for benchmarking)
crispasr -ng -m model.gguf -f audio.wav

# CUDA unified memory (swap to RAM when VRAM exhausted)
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 crispasr -m model.gguf -f audio.wav
```

Build flags: `-DGGML_CUDA=ON`, `-DGGML_METAL=ON`, `-DGGML_VULKAN=ON`.

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
