# CrispASR

> **Note:** this repo was previously called `cohere-whisper.cpp`. GitHub keeps a permanent redirect from the old URL, so existing links/clones still work.

A fork of [whisper.cpp](https://github.com/ggml-org/whisper.cpp) that adds full C++ ggml runtimes for **multiple multilingual ASR models** plus a universal multilingual forced aligner:

| Runtime | What it is | HF release |
| --- | --- | --- |
| **`parakeet-main`** | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — 600M FastConformer + TDT. Fastest multilingual ASR + **free word timestamps** | [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF) |
| `parakeet-main` (DE) | [`johannhartmann/parakeet_de_med`](https://huggingface.co/johannhartmann/parakeet_de_med) — German medical PEFT fine-tune | [`cstr/parakeet_de_med-GGUF`](https://huggingface.co/cstr/parakeet_de_med-GGUF) |
| **`canary-main`** | [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) — 978M FastConformer + Transformer. Multilingual ASR + **speech translation** with explicit `-sl/-tl` flags | [`cstr/canary-1b-v2-GGUF`](https://huggingface.co/cstr/canary-1b-v2-GGUF) |
| **`nfa-align`** | Auxiliary CTC model from canary-1b-v2's `.nemo` — **universal multilingual forced aligner** (25 languages, ~78 ms MAE) | [`cstr/canary-ctc-aligner-GGUF`](https://huggingface.co/cstr/canary-ctc-aligner-GGUF) |
| `cohere-main` | [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) — 2B Conformer + Transformer. Lowest English WER on Open ASR Leaderboard | [`cstr/cohere-transcribe-03-2026-GGUF`](https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF) |
| `cohere-align` | wav2vec2 character CTC forced aligner (English-only, 30-50 ms MAE) | uses [`jonatasgrosman/wav2vec2-large-xlsr-53-english`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) |
| **`qwen3-asr-main`** | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) — 900M speech-LLM (Whisper-style audio encoder + Qwen3 0.6B LLM with audio-token injection). 30 languages + 22 Chinese dialects, Open ASR avg WER 6.42, persistent KV cache | [`cstr/qwen3-asr-0.6b-GGUF`](https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF) |
| **`voxtral-main`** | [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) — 3B speech-LLM (Whisper-large-v3 encoder + Mistral/Llama 3B LLM). ASR + audio understanding + text Q&A, 8 languages, function calling from voice. Best-in-class text performance retained | [`cstr/voxtral-mini-3b-2507-GGUF`](https://huggingface.co/cstr/voxtral-mini-3b-2507-GGUF) |
| **`voxtral4b-main`** | [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) — 4.4B **realtime streaming** speech-LLM (causal RoPE+SwiGLU encoder + Mistral 3.4B LLM with adaptive RMSNorm). 13 languages, <500ms latency, competitive with offline models | `cstr/voxtral-mini-4b-realtime-GGUF` (pending) |
| **`granite-main`** | [`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) — 1B speech-LLM (16-layer Conformer encoder with Shaw RPE + BLIP-2 Q-Former projector + Granite 1B LLM with μP). 6 languages (en, fr, de, es, pt, ja), Apache-2.0 | `cstr/granite-speech-4.0-1b-GGUF` (pending) |

All nine runtimes share ggml-based inference — that's how we ported each new model in days rather than weeks. Qwen3-ASR, Voxtral 3B, and Voxtral 4B Realtime are **speech-LLMs**: instead of dedicated CTC/transducer/seq2seq decoders, the audio encoder output frames are injected into the input embeddings of a stock LLM, and the LLM autoregressively generates the transcript. Voxtral 3B additionally supports audio understanding (Q&A about audio content) and function calling from voice. Voxtral 4B Realtime is a **natively streaming** model with a causal audio encoder and configurable transcription delay (240ms-2.4s), designed for on-device real-time ASR.

> **Branch state.** Everything lives on `main` as of April 2026. The original cohere-only history is preserved at the [`archive/cohere-only`](https://github.com/CrispStrobe/CrispASR/tree/archive/cohere-only) branch as a historical reference.

> **Future direction.** The plan is to fold the per-model `*-main` binaries into a single subcommand-driven `crispasr` binary, and `cohere-align` + `nfa-align` into a single `crispalign` binary. The current per-model binaries will remain as thin shims for backward compat. See [`RENAMING.md`](RENAMING.md) for the plan.

> **Pending upstream fixes** that affect this fork are tracked in [`UPSTREAM.md`](UPSTREAM.md) (currently: `whisper.cpp` `ffmpeg-transcode.cpp` mp4-container bug, ggml VNNI Q8_0 dispatch, NeMo aux model standalone release).

## Which runtime should I use?

| Need | Right tool |
| --- | --- |
| Lowest English WER, model size doesn't matter | `cohere-main` (ggml branch) |
| Word-level timestamps + multilingual + small + fast | **`parakeet-main`** |
| Multilingual + **explicit language control** (no auto-detect) | **`canary-main`** |
| **Speech translation** (X→En or En→X) | **`canary-main`** |
| 30 ms-accurate word stamps via CTC forced alignment | `cohere-align` (ggml branch) |
| **30 languages + 22 Chinese dialects** | **`qwen3-asr-main`** |
| Speech-LLM with persistent KV cache (faster than realtime at Q4_K) | **`qwen3-asr-main`** |
| **Realtime streaming** ASR (low latency, on-device) | **`voxtral4b-main`** |
| Highest-quality offline speech-LLM (3B backbone) | **`voxtral-main`** |

| | parakeet-tdt-0.6b-v3 | canary-1b-v2 |
| --- | --- | --- |
| Parameters | 600M | 978M |
| Architecture | FastConformer encoder + TDT decoder | FastConformer encoder + Transformer decoder |
| Languages | 25 EU (auto-detect) | 25 EU (explicit `-sl` flag) |
| Speech translation | ❌ | ✅ X→En and En→X |
| Word timestamps | ✅ from TDT duration head | ✗ (segment-level via auxiliary CTC) |
| Q4_K size | 467 MB | ~600 MB |
| Open ASR WER (avg) | 6.34% | 7.15% |
| License | CC-BY-4.0 | CC-BY-4.0 |

Both share the same FastConformer encoder code and the same NeMo-style mel preprocessor (128 mels, 16 kHz, n_fft=512, win=400, hop=160).

---

## Audio formats

Every CLI in this repo (`cohere-main`, `parakeet-main`, `canary-main`, `cohere-align`, `nfa-align`) routes input through the same `read_audio_data()` loader inherited from whisper.cpp. By default the loader uses two embedded single-header decoders:

- **[miniaudio](https://miniaud.io/)** — handles **WAV** (any bit depth, including 16/24/32-bit PCM, IEEE float, A-law, μ-law, ADPCM), **FLAC**, and **MP3**
- **[stb_vorbis](https://github.com/nothings/stb)** — handles **OGG Vorbis**

So **out of the box, all five tools accept WAV / FLAC / MP3 / OGG Vorbis** at any bit depth, any sample rate (auto-resampled to 16 kHz), mono or stereo (auto-mixed to mono). No external dependencies needed.

### What's NOT supported in the default build

| Format | Workaround |
| --- | --- |
| `.opus` | Convert to WAV first, OR build with `WHISPER_FFMPEG=ON` (see below) |
| `.m4a`, `.aac`, `.alac` | Same — convert or use ffmpeg build |
| `.webm`, `.mp4`, `.mkv`, `.mov` (video containers with audio tracks) | Same |
| `.aiff`, `.au`, `.wma`, raw PCM | Same |

### Option A — pre-convert with ffmpeg (zero changes to this repo)

Single-line workaround that handles **every audio/video format ffmpeg knows**:

```bash
ffmpeg -i input.opus -ar 16000 -ac 1 -c:a pcm_s16le -y /tmp/audio.wav
./build/bin/parakeet-main -m model.gguf -f /tmp/audio.wav -t 8

# Same for video files — just extract the audio track:
ffmpeg -i input.mp4 -vn -ar 16000 -ac 1 -c:a pcm_s16le -y /tmp/audio.wav
```

Most users running ASR tools already have ffmpeg installed. This is the recommended path for occasional use.

### Option B — build with `WHISPER_FFMPEG=ON` (transparent in-process decoding)

The whisper.cpp loader has a built-in ffmpeg fallback path: if miniaudio refuses a file, it routes the bytes through `libavformat` / `libavcodec` / `libswresample` to extract a 16 kHz mono PCM stream in-process. **No `/tmp/audio.wav` round-trip, no shell invocation, no separate ffmpeg binary needed at runtime** — just the shared libs.

```bash
# Install the ffmpeg dev libraries first (one-time):
#   Debian/Ubuntu:  apt install libavformat-dev libavcodec-dev libavutil-dev libswresample-dev
#   macOS:          brew install ffmpeg
#   Fedora:         dnf install ffmpeg-devel

cmake -B build-ffmpeg -DCMAKE_BUILD_TYPE=Release -DWHISPER_FFMPEG=ON
cmake --build build-ffmpeg -j$(nproc) --target parakeet-main canary-main nfa-align cohere-main cohere-align

# Now every CLI accepts every format ffmpeg supports, transparently:
./build-ffmpeg/bin/parakeet-main -m model.gguf -f input.opus -t 8
./build-ffmpeg/bin/canary-main   -m model.gguf -f input.mp4  -sl en -tl en -t 8
./build-ffmpeg/bin/nfa-align     -m model.gguf -f input.m4a  -tt "transcript text"
```

The runtime then depends on the ffmpeg shared libraries (`libavformat.so`, `libavcodec.so`, `libavutil.so`, `libswresample.so`) — anywhere those are installed, it just works.

### Measured results (this fork, on jfk.wav transcoded to various formats)

| Format | Default build | `WHISPER_FFMPEG=ON` build |
| --- | :---: | :---: |
| `.wav` (any bit depth) | ✅ | ✅ |
| `.flac`                 | ✅ | ✅ |
| `.mp3`                  | ✅ | ✅ |
| `.ogg` (Vorbis)         | ✅ | ✅ |
| `.opus`                 | ❌ "failed to read audio data as wav" | ✅ perfect transcript |
| `.m4a` (AAC)            | ❌ | ⚠ **crashes** (`munmap_chunk()` — upstream `whisper.cpp` `ffmpeg-transcode.cpp` bug on mp4-container files) |
| `.webm` (Opus inside)   | ❌ | ⚠ **hangs** (same upstream bug class) |

The upstream `ffmpeg-transcode.cpp` integration in `whisper.cpp` has known issues with mp4-family container formats. **For these the safe path is still pre-conversion via ffmpeg one-liner.** Bare-codec files like `.opus` work cleanly in the FFmpeg build.

### When to use which option

- **Default build (no ffmpeg dep)** — for clean WAV/FLAC/MP3/OGG pipelines, smallest binary, no system dependencies. **Recommended for most users.**
- **`WHISPER_FFMPEG=ON` build** — adds in-process Opus support and a one-step decode for any other format that doesn't crash the upstream `ffmpeg-transcode.cpp`. Useful but currently NOT a complete substitute for pre-conversion: m4a/mp4/webm containers still crash. Treat it as opt-in convenience for `.opus` ingestion.
- **Pre-conversion via ffmpeg** (`ffmpeg -i in.X -ar 16000 -ac 1 -c:a pcm_s16le out.wav`) — **the universally safe path** for everything not in the default-build column above. No build flags, no upstream bugs, identical results.

Both binaries can coexist — keep `build/` for the lean default build and `build-ffmpeg/` for the Opus-supporting one.

## Quick start — parakeet (fastest, multilingual ASR)

```bash
git clone -b parakeet https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target parakeet-main

huggingface-cli download cstr/parakeet-tdt-0.6b-v3-GGUF \
    parakeet-tdt-0.6b-v3-q4_k.gguf --local-dir .

./build/bin/parakeet-main -m parakeet-tdt-0.6b-v3-q4_k.gguf -f samples/jfk.wav -t 8
# And so, my fellow Americans, ask not what your country can do for you,
# ask what you can do for your country.
```

Word timestamps via `-v`:
```
[ 0.32s →  0.64s]  And
[ 0.64s →  0.88s]  so,
[ 1.04s →  1.28s]  my
[ 1.28s →  1.76s]  fellow
[ 1.76s →  3.28s]  Americans.
```

Each boundary is one encoder frame = **80 ms**. Long audio + VAD + SRT/VTT/TXT all supported via `-vad-model`, `-osrt`, `-ovtt`, `-ot`, `-ml N`. See the [parakeet quick start above](#quick-start--parakeet-fastest-multilingual-asr) for the full CLI reference.

The auto-language detect on parakeet works well for clean speech but can code-switch into English on German clips with technical vocabulary or proper nouns (see [`test_german.md`](test_german.md)). For German production use, prefer canary with `-sl de`.

### Parakeet — German fine-tunes

Any fine-tune of `parakeet-tdt-0.6b-v3` can be loaded with the same `parakeet-main` runtime, since the GGUF converter and C++ loader work on the architecture, not on a specific checkpoint. We've tested:

- **[`johannhartmann/parakeet_de_med`](https://huggingface.co/johannhartmann/parakeet_de_med)** — PEFT decoder+joint fine-tune on German medical documentation, **3.28% WER** on the German medical test set (vs 11.73% for the base model). Encoder is frozen so it inherits the base model's auto-language behaviour, but the German bias on the decoder makes it the right choice for German medical transcription on CPU.

```bash
# Convert + run
python models/convert-parakeet-to-gguf.py \
    --nemo  parakeet_de_med.nemo \
    --output parakeet_de_med.gguf
./build/bin/cohere-quantize parakeet_de_med.gguf parakeet_de_med-q4_k.gguf q4_k
./build/bin/parakeet-main -m parakeet_de_med-q4_k.gguf -f german_audio.wav -t 8
```

Pre-converted GGUFs at **[cstr/parakeet_de_med-GGUF](https://huggingface.co/cstr/parakeet_de_med-GGUF)** (F16 + Q4_K/Q5_0/Q8_0).

---

## Quick start — canary (explicit language + translation)

```bash
cmake --build build -j$(nproc) --target canary-main

# Convert your own from the .nemo (no pre-quantised yet at time of writing):
pip install gguf torch sentencepiece huggingface_hub
python -c "from huggingface_hub import snapshot_download; \
  print(snapshot_download('nvidia/canary-1b-v2'))"
python models/convert-canary-to-gguf.py \
    --nemo  <snapshot-path>/canary-1b-v2.nemo \
    --output canary-1b-v2.gguf

./build/bin/canary-main -m canary-1b-v2.gguf -f samples/jfk.wav -sl en -tl en -t 8
# And so, my fellow Americans, ask not what your country can do for you,
# ask what you can do for your country.
```

### German ASR

```bash
./build/bin/canary-main -m canary-1b-v2.gguf -f german_audio.wav -sl de -tl de
# Ich heiße Amadeus Scharma. Ich bin 1955 in Kassel in Deutschland geboren,
# weitgehend in Indien aufgewachsen. ...
```

### Speech translation (German → English)

```bash
./build/bin/canary-main -m canary-1b-v2.gguf -f german_audio.wav -sl de -tl en
# My name is Amadeo Sharma. I was born in Kassel in Germany in 1955,
# and I grew up largely in India. ...
```

Same `-sl X -tl Y` works for any pair of the 25 supported languages: `bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk`. When `sl == tl` it's ASR; when they differ, it's speech translation.

### CLI reference

```
usage: canary-main [options] -m MODEL -f AUDIO

  -m  FNAME       canary GGUF model
  -f  FNAME       input audio (16 kHz mono WAV)
  -sl LANG        source language ISO-639-1 (en, de, fr, ...)
  -tl LANG        target language. Same as -sl → ASR; differs → translation
  -t  N           threads (default: 4)
  -v              dump per-token decoder steps for debugging
```

The full VAD / chunking / SRT / VTT / TXT plumbing matches `parakeet-main` and is being added incrementally (see `canary-todo.md` for status).

### Architecture

| Component | Details |
| --- | --- |
| Encoder       | 32-layer FastConformer, d=1024, 8 heads, head_dim=128, FFN=4096, conv kernel=9, **biases on every linear/conv** (Canary uses `use_bias: true`) |
| Subsampling   | Conv2d dw_striding stack, 8× temporal (100 → 12.5 fps) |
| Decoder       | 8-layer pre-LN Transformer (self-attn + cross-attn + FFN), d=1024, 8 heads, head_dim=128, FFN=4096, max_ctx=1024 |
| Embedding     | Token (16384 × 1024) + learned positional (1024 × 1024) + LN |
| Output head   | Separate linear (1024 → 16384) |
| Vocab         | 16384 SentencePiece (CanaryBPETokenizer, identical to Cohere Transcribe) |
| Parameters    | ~978M (encoder 811M + decoder 152M + head 17M) |
| Tensors       | 1478 in the GGUF (encoder 1294 + decoder 179 + head 2 + preprocessor 2) |

The Conformer encoder is identical in structure to parakeet's (we share the encoder code). The decoder block is pre-LN with three sub-layers: `LN → SA → +residual → LN → CA → +residual → LN → FFN → +residual`, FFN activation is ReLU (per NeMo's `PositionWiseFF` default). Self-attention KV cache lives on a backend buffer for fast autoregressive generation. Cross-attention K/V is pre-computed once per audio slice from the encoder output.

### `nfa-align` — universal multilingual subword forced alignment

`canary-1b-v2.nemo` actually ships with **two** weight files inside the tarball: the main encoder–decoder model AND a separate 600 M-parameter Parakeet-style FastConformer + CTC head trained with the same SentencePiece vocab. NVIDIA uses the latter inside [NeMo Forced Aligner](https://github.com/NVIDIA-NeMo/NeMo) (NFA) to compute reliable word-level timestamps for canary's transcripts.

We extract that auxiliary model as a standalone GGUF and expose it via `nfa-align` — a **universal multilingual forced aligner** that works on **any transcript text** in the 25 supported languages.

```bash
# Build
cmake --build build -j$(nproc) --target nfa-align

# Download the aligner model
huggingface-cli download cstr/canary-ctc-aligner-GGUF \
    canary-ctc-aligner-q4_k.gguf --local-dir .

# Forced-align an existing transcript (any source: canary, parakeet, cohere, whisper, hand-typed)
./build/bin/nfa-align \
    -m canary-ctc-aligner-q4_k.gguf \
    -f samples/jfk.wav \
    -tt "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
```

Output:
```
[ 0.40 →  0.48]  And
[ 0.64 →  1.04]  so,
[ 1.12 →  1.20]  my
[ 1.36 →  1.60]  fellow
[ 1.84 →  3.20]  Americans,
[ 3.52 →  3.76]  ask
[ 4.08 →  4.16]  not
...
[10.08 → 11.04]  country.
```

**Measured accuracy on JFK (22 words, vs `cohere-align` wav2vec2 ground truth):**

| Method | MAE | Notes |
| --- | ---: | --- |
| `cohere-align` (wav2vec2 char CTC) | ~30-50 ms | English only, 1 model per language |
| **`nfa-align` (subword CTC, this fork)** | **78 ms** | **All 25 EU languages in one model** |
| `canary-main` cross-attention DTW | ~414 ms | Built into canary-main, no extra model |

**5.3× tighter** than canary's built-in DTW path, and the **first multilingual forced aligner** in this fork. Works as a drop-in replacement for `cohere-align` with broader language coverage but slightly looser per-word boundaries (subword vs character granularity). For 24 of the 25 supported languages there is no comparable wav2vec2 model, so this is the only option at this accuracy.

It also doubles as a standalone CTC ASR via `-decode`:

```bash
$ ./build/bin/nfa-align -m canary-ctc-aligner-q4_k.gguf -f samples/jfk.wav -decode
And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
```

Pre-converted GGUFs at **[cstr/canary-ctc-aligner-GGUF](https://huggingface.co/cstr/canary-ctc-aligner-GGUF)** (F16 + Q4_K/Q5_0/Q8_0). Q4_K alignment is byte-identical to F16 on the verification clip.

**License note.** The aligner model is the auxiliary CTC component of `nvidia/canary-1b-v2`, **CC-BY-4.0**, full credit and attribution to NVIDIA's NeMo team. See the HF model card for the full attribution and citation.

### Decoder prompt format

Canary uses task tokens in the decoder prompt prefix to drive ASR vs translation:

```
<|startofcontext|> <|startoftranscript|> <|emo:undefined|>
<|src_lang|> <|target_lang|> <|pnc|>|<|nopnc|>
<|notimestamp|> <|nodiarize|>
... model output starts here ...
```

The src/tgt language tokens explicitly tell the decoder what language to expect and what language to emit. There is no auto-detect — this is the whole point of using canary. If the source language is unknown, you'd need a separate language-ID model.

---

## Quick start — Voxtral 4B Realtime (streaming speech-LLM)

```bash
cmake --build build -j$(nproc) --target voxtral4b-main

# Convert from HuggingFace (or download pre-converted GGUF when available)
python models/convert-voxtral4b-to-gguf.py \
    --input /path/to/Voxtral-Mini-4B-Realtime-2602 \
    --output voxtral-4b-realtime.gguf

./build/bin/voxtral4b-main -m voxtral-4b-realtime.gguf -f samples/jfk.wav
# And so, my fellow Americans, ask not what your country can do for you.
# Ask what you can do for your country.
```

The 4B Realtime model is a **natively streaming** architecture with a causal audio encoder (RoPE + SwiGLU + sliding window attention). It produces text interleaved with streaming control tokens — the CLI filters these automatically for clean output. Key features:

- **13 languages** (en, fr, es, de, ru, zh, ja, it, pt, nl, ar, hi, ko)
- **Configurable delay**: 480ms default (6 tokens), tunable from 240ms to 2.4s
- **4.4B parameters** (970M encoder + 3.4B LLM), ~8.9 GB F16 GGUF
- **Apache 2.0** license
- Optional word timestamps via CTC aligner: `-am aligner.gguf -timestamps`

---

## Quick start — Qwen3-ASR (30+ languages, speech-LLM)

```bash
cmake --build build -j$(nproc) --target qwen3-asr-main

huggingface-cli download cstr/qwen3-asr-0.6b-GGUF \
    qwen3-asr-0.6b-q4_k.gguf --local-dir .

./build/bin/qwen3-asr-main -m qwen3-asr-0.6b-q4_k.gguf -f samples/jfk.wav
# And so, my fellow Americans, ask not what your country can do for you,
# ask what you can do for your country.
```

900M speech-LLM (Whisper encoder + Qwen3 0.6B LLM). Auto-detects language from audio. Supports 30 languages + 22 Chinese dialects. Optional word timestamps via CTC aligner:

```bash
./build/bin/qwen3-asr-main -m qwen3-asr-0.6b-q4_k.gguf -f audio.wav \
    -am canary-ctc-aligner-q4_k.gguf -timestamps
```

---

## Quick start — Voxtral 3B (offline speech-LLM)

```bash
cmake --build build -j$(nproc) --target voxtral-main

huggingface-cli download cstr/voxtral-mini-3b-2507-GGUF \
    voxtral-mini-3b-2507-q4_k.gguf --local-dir .

./build/bin/voxtral-main -m voxtral-mini-3b-2507-q4_k.gguf -f samples/jfk.wav -l en
# And so, my fellow Americans, ask not what your country can do for you,
# ask what you can do for your country.
```

3B speech-LLM (Whisper-large-v3 encoder + Mistral/Llama 3B). Supports 8 languages (`-l en/de/fr/es/it/pt/nl/hi`). Full Tekken tokenizer for audio understanding and function calling.

---

## Current status

| Component | parakeet | canary | cohere | qwen3-asr | voxtral 3B | voxtral 4B | granite 1B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GGUF converter | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Encoder forward | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Decoder/LLM forward | ✅ TDT | ✅ Transformer | ✅ Transformer | ✅ Qwen3 LLM | ✅ Llama 3B | ✅ Llama 3.4B | ✅ Granite 1B |
| Word timestamps | ✅ TDT native | ✅ CTC re-align | ✅ cross-attn DTW | ✅ CTC 2nd pass | ✅ CTC 2nd pass | ✅ CTC 2nd pass | ❌ pending |
| SRT/VTT output | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| `--flash` attention | ✅ | ✅ | ✅ | always on | always on | always on | always on |
| VAD segmentation | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| GPU auto-detect | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Quantization | ✅ Q4_K-Q8_0 | ✅ | ✅ | ✅ | ✅ | ❌ pending | ❌ pending |
| HF release | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ pending | ❌ pending |

See `TODO.md` for the full feature roadmap.

## Repository layout

| Path | Description |
| --- | --- |
| `src/parakeet.{h,cpp}` | Parakeet TDT 0.6B — FastConformer + TDT transducer |
| `src/canary.{h,cpp}` | Canary 1B v2 — FastConformer + Transformer decoder |
| `src/canary_ctc.{h,cpp}` | CTC forced aligner (from canary's auxiliary model) |
| `src/cohere.{h,cpp}` | Cohere Transcribe 2B — Conformer + Transformer |
| `src/qwen3_asr.{h,cpp}` | Qwen3-ASR 0.6B — Whisper encoder + Qwen3 LLM |
| `src/voxtral.{h,cpp}` | Voxtral-Mini 3B — Whisper-large-v3 encoder + Llama 3B |
| `src/voxtral4b.{h,cpp}` | Voxtral-Mini 4B Realtime — causal encoder + Llama 3.4B |
| `src/granite_speech.{h,cpp}` | Granite Speech 4.0-1B — Conformer + Q-Former + Granite LLM |
| `src/wav2vec2-ggml.{h,cpp}` | wav2vec2 CTC for `cohere-align` |
| `examples/*/main.cpp` | CLI entry points for each runtime |
| `models/convert-*-to-gguf.py` | Model converters (HF/NeMo → GGUF) |

## Attribution

- **Parakeet TDT** and **Canary 1B v2**: NVIDIA NeMo team (CC-BY-4.0)
- **Cohere Transcribe**: Cohere Labs (Apache-2.0)
- **Qwen3-ASR**: Qwen team / Alibaba (Apache-2.0)
- **Voxtral-Mini 3B** and **Voxtral-Mini 4B Realtime**: Mistral AI (Apache-2.0)
- **Granite Speech 4.0-1B**: IBM Granite team (Apache-2.0)
- **wav2vec2 weights**: Jonatas Grosman (Apache-2.0)
- **Underlying runtime**: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) / [ggml](https://github.com/ggerganov/ggml) (MIT)

Voxtral 4B Realtime port cross-referenced against [antirez/voxtral.c](https://github.com/antirez/voxtral.c), [awni/voxmlx](https://github.com/awni/voxmlx), and [TrevorS/voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs).

## License

The fork code is MIT (matching whisper.cpp). Individual models have their own licenses — see the links above. Use of GGUF files must comply with each model's upstream license.
