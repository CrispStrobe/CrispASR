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

All five runtimes share the same NeMo-style mel preprocessor and (mostly) the same FastConformer encoder family — that's how we ported each new model in days rather than weeks.

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

The auto-language detect on parakeet works well for clean speech but can misfire on accented or noisy audio (we found it picked Russian on Angela Merkel's German speech, see [`test_german.md`](test_german.md)). For German production use, prefer canary with `-sl de`.

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

## Current status (parakeet branch)

| Component | parakeet | canary |
| --- | --- | --- |
| GGUF converter | ✅ | ✅ |
| Loader | ✅ | ✅ |
| Encoder forward | ✅ | ✅ |
| Decoder forward | ✅ | ✅ |
| Mel STFT | ✅ | ✅ (shared) |
| Greedy decode | ✅ TDT | ✅ Transformer |
| Word timestamps | ✅ from TDT durations | ⏳ scaffold (linear) |
| CLI: -sl/-tl | n/a | ✅ |
| CLI: VAD + chunking | ✅ | ⏳ |
| CLI: SRT/VTT/TXT | ✅ | ⏳ |
| Quantisation | ✅ Q4_K/Q5_0/Q8_0 | ⏳ |
| HF release | ✅ | ⏳ |

`canary-todo.md` tracks the remaining items in detail. The encoder, decoder, prompt builder, and end-to-end ASR + translation are all proven working — what's left is mostly CLI polish and quantisation.

## Key bug fixes in this branch

- **Transposed positional encoding** in `parakeet_make_pos_enc` / `canary_make_pos_enc` — the function wrote `pe[(2*i)*K + j]` (positions fast, dims slow) but the ggml tensor was `(d, 2T-1)` with dims fast. The correct layout is `pe[dim + pos*d]`. Parakeet's TDT decoder was robust enough to mostly recover; canary's encoder-decoder cross-attention exposed the bug immediately. Both runtimes now use the corrected layout. JFK output went from `"And so my fellow Americans. Ask not..."` (periods, parakeet pre-fix) to `"And so, my fellow Americans, ask not..."` (commas, canonical).

## Repository layout

| Path | Description |
| --- | --- |
| `src/parakeet.{h,cpp}`                    | Public C API + ggml runtime for parakeet TDT |
| `src/canary.{h,cpp}`                      | Public C API + ggml runtime for canary 1B v2 |
| `src/cohere.{h,cpp}`                      | Cohere Transcribe runtime (from the ggml branch) |
| `src/wav2vec2-ggml.{h,cpp}` + `src/align.{h,cpp}` | wav2vec2 CTC forced alignment for `cohere-align` |
| `models/convert-parakeet-to-gguf.py`      | `.nemo → GGUF` for parakeet |
| `models/convert-canary-to-gguf.py`        | `.nemo → GGUF` for canary |
| `examples/parakeet-main/main.cpp`         | parakeet CLI (full: VAD, chunking, SRT/VTT/TXT) |
| `examples/canary-main/main.cpp`           | canary CLI (basic: ASR + translation, polish ongoing) |
| `examples/cohere-main/`, `examples/cohere-align/` | cohere CLIs |
| `parakeet-todo.md`                        | parakeet implementation plan (mostly complete) |
| `canary-todo.md`                          | canary implementation plan (encoder + decoder + prompt done) |
| `benchmark_cohere.md`                     | Cross-runtime benchmark numbers |
| `test_german.md`                          | German audio comparison: parakeet's auto-detect failure modes |
| `ggml_plans.md`                           | VNNI Q8_0 plan to close the ONNX inference gap |

## Attribution

- **Parakeet TDT 0.6B v3** and **Canary 1B v2**: NVIDIA NeMo team (CC-BY-4.0). Use must comply with the CC-BY-4.0 license including attribution.
- **Encoder graph patterns**: shared between cohere/parakeet/canary, originally adapted from the CrispASR ggml branch.
- **Decoder pattern (canary)**: cross-checked against NeMo's `transformer_decoders.py` and `transformer_modules.py` source.
- **Underlying runtime**: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) / [ggml](https://github.com/ggerganov/ggml).

## License

The fork code is MIT (matching whisper.cpp). The parakeet and canary models themselves are **CC-BY-4.0**, inherited from NVIDIA. Use of the GGUF files must comply with CC-BY-4.0 including attribution.
