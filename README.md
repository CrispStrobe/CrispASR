# parakeet-whisper.cpp

A fork of [whisper.cpp](https://github.com/ggml-org/whisper.cpp) that adds a full C++ ggml runtime for **[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** — NVIDIA's 600M-parameter multilingual ASR model with **built-in word-level timestamps**.

Pre-converted GGUF weights are on Hugging Face: **[cstr/parakeet-tdt-0.6b-v3-GGUF](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF)** (F16, Q8_0, Q5_0, Q4_K).

> **Looking for the Cohere Transcribe runtime?** It lives on the **[`ggml`](https://github.com/CrispStrobe/cohere-whisper.cpp/tree/ggml)** branch of the same repo. This branch (`parakeet`) is dedicated to the parakeet TDT runtime and shares the underlying ggml + Conformer infrastructure.

## Why parakeet?

| | Whisper-style | Cohere Transcribe | **Parakeet TDT 0.6B v3** |
| --- | --- | --- | --- |
| Architecture | Encoder–decoder | Encoder–decoder | **Encoder–transducer (TDT)** |
| Parameters | 39M – 1.5B | 2B | **600M** |
| Languages | 99 (most thin) | 14 | **25 EU (well covered)** |
| Word timestamps | Native (timestamp tokens) | Cross-attention DTW (~360 ms) | **TDT duration head (~80 ms)** |
| Word-timestamp model needed? | included | + separate CTC aligner | **none — built into the decoder** |
| Open ASR WER (avg) | 7.5 (large-v3) | 5.42 | 6.34 |
| Auto language detect | yes | no (need `-l`) | **yes** |
| Q4_K size on disk | varies | 1.2 GB | **467 MB** |
| Wall time on 5.4 s clip (8 thr CPU, Q4_K) | varies | 14.8 s | **5.3 s** (~1× realtime) |
| Licence | MIT | Apache 2.0 + CC-BY-NC research notes | **CC-BY-4.0** |

The killer feature is that **word timestamps come for free from the TDT decoder's duration head**. Every emitted token already carries `(t_start, t_end)` in encoder frames. No separate CTC alignment model, no DTW post-processing, no quality–speed trade-off.

## Quick start

### 1. Build

```bash
git clone -b parakeet https://github.com/CrispStrobe/cohere-whisper.cpp
cd cohere-whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target parakeet-main
```

On macOS (Apple Silicon), Metal acceleration is enabled automatically. CUDA: `-DGGML_CUDA=ON`.

### 2. Get a model

Download a pre-quantised one:
```bash
huggingface-cli download cstr/parakeet-tdt-0.6b-v3-GGUF \
    parakeet-tdt-0.6b-v3-q4_k.gguf --local-dir .
```

Or convert your own from the original `.nemo` checkpoint:
```bash
pip install gguf torch sentencepiece huggingface_hub

python -c "from huggingface_hub import snapshot_download; \
  print(snapshot_download('nvidia/parakeet-tdt-0.6b-v3'))"

python models/convert-parakeet-to-gguf.py \
    --nemo  <snapshot-path>/parakeet-tdt-0.6b-v3.nemo \
    --output parakeet-tdt-0.6b-v3.gguf

# Optional: quantise
./build/bin/cohere-quantize parakeet-tdt-0.6b-v3.gguf parakeet-tdt-0.6b-v3-q4_k.gguf q4_k
```

Available quantisations:

| File | Size |
| --- | ---: |
| `parakeet-tdt-0.6b-v3.gguf`      | 1.26 GB (F16, full precision) |
| `parakeet-tdt-0.6b-v3-q8_0.gguf` |  711 MB (near-lossless) |
| `parakeet-tdt-0.6b-v3-q5_0.gguf` |  516 MB |
| `parakeet-tdt-0.6b-v3-q4_k.gguf` |  467 MB (recommended default) |

### 3. Transcribe

Basic transcription:
```bash
./build/bin/parakeet-main \
    -m parakeet-tdt-0.6b-v3-q4_k.gguf \
    -f samples/jfk.wav -t 8
# And so my fellow Americans. Ask not what your country can do for you.
# Ask what you can do for your country.
```

No `-l LANG` is needed — parakeet auto-detects from the audio. The 25 supported languages: `bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk`.

### 4. Word timestamps

Pass `-v` for per-word and per-token timestamps from the TDT duration head:

```
$ ./build/bin/parakeet-main -m parakeet-tdt-0.6b-v3-q4_k.gguf -f samples/jfk.wav -t 8 -v
And so my fellow Americans. Ask not what your country can do for you. Ask what you can do for your country.

  --- words (22) for slice [00:00:00.000 --> 00:00:11.000] ---
  [    0.32s →     0.64s]  And
  [    0.64s →     0.96s]  so
  [    1.04s →     1.28s]  my
  [    1.28s →     1.76s]  fellow
  [    1.76s →     3.28s]  Americans.
  [    3.28s →     3.84s]  Ask
  [    4.08s →     4.40s]  not
  [    5.28s →     5.60s]  what
  [    5.60s →     5.92s]  your
  [    5.92s →     6.48s]  country
  [    6.48s →     6.72s]  can
  [    6.72s →     7.04s]  do
  [    7.04s →     7.36s]  for
  [    7.36s →     8.16s]  you.
  ...
```

Each word boundary is one encoder frame = **80 ms**. Sub-word SentencePiece tokens are grouped at the leading-space boundary, with punctuation tokens attaching to the previous word. Compare to cohere-main's cross-attention DTW path: ~360 ms MAE there, ~80 ms here, no extra model.

### 5. Subtitle output

```bash
./build/bin/parakeet-main -m parakeet-tdt-0.6b-v3-q4_k.gguf \
    -f long-audio.wav -t 8 \
    -ml 60 -osrt -ovtt -ot
# writes long-audio.srt, long-audio.vtt, long-audio.txt
```

`-ml N` packs words into segments of at most N characters, splitting only at word boundaries. `-ml 1` produces one SRT cue per word.

### 6. Long audio

The encoder is O(T²) in the number of frames (the model itself supports up to 24 minutes of audio with full attention, but on CPU you hit memory walls before that). Two strategies are available:

**Fixed chunking (default).** Audio longer than 30 s is automatically split into 30 s windows. Chunk size is configurable with `-ck N`:
```bash
./build/bin/parakeet-main -m model.gguf -f hour-long-talk.wav -ck 30
```

**VAD-aware segmentation (recommended for long audio).** Uses Silero VAD to find real speech boundaries — sentence-aligned, with cleaner output and slightly better accuracy than arbitrary 30 s windows:
```bash
# Download silero VAD model once
wget https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin -O ggml-silero-vad.bin

./build/bin/parakeet-main -m model.gguf -f long-audio.wav \
    -vad-model ggml-silero-vad.bin -osrt
```

## CLI reference

```
usage: parakeet-main [options] -m MODEL.gguf -f AUDIO.wav

  -m  FNAME       parakeet GGUF model
  -f  FNAME       input audio (any format common-whisper can decode → 16 kHz mono)
  -t  N           threads (default: min(4, hw_threads))
  -ot             write transcript to <audio>.txt
  -osrt           write SRT subtitle file
  -ovtt           write WebVTT subtitle file
  -ml N           max chars per output segment (0 = unlimited, 1 = per-word)
  -ck N           chunk size in seconds for long audio without VAD (default: 30)
  -vad-model FNAME   Silero VAD model for speech segmentation
  -vad-thold F    VAD threshold (default: 0.5)
  -v, --verbose   dump per-word and per-token timestamps
  -np             suppress all informational output
```

## Architecture

Parakeet TDT is a **transducer**, not an encoder–decoder. Inference is fundamentally different from Whisper / Cohere: there's no autoregressive cross-attention loop, no KV cache, just a small LSTM predictor and a joint head over (encoder_frame × predictor_state).

```
audio
  ↓ NeMo-style mel preprocessor (128 mels, 16 kHz, n_fft=512, hop=160, Hann)
  ↓ Conv2d dw_striding subsampling (8× temporal: 100 fps → 12.5 fps)
  ↓ 24× FastConformer block:
       FFN1 (Macaron, ½ scale)
       MHA  (rel-pos with Transformer-XL untied biases)
       Conv (pw1 + GLU + dw_k=9 + BN + swish + pw2)
       FFN2 (Macaron, ½ scale)
       LN_out
  → encoder_out [T_enc, 1024]
                                ┐
            ┌── joint(enc[t]) + joint(pred[u]) ─→ ReLU ─→ linear(640 → 8198)
            │       (640 hid)        (640 hid)               (8192 vocab + 1 blank + 5 durations)
            ▼
  predictor: embed(8193, 640) + 2-layer LSTM(640, 640)
            ↑
           emitted tokens (autoregressive over u, NOT over t)
```

| Component | Details |
| --- | --- |
| Encoder       | 24-layer FastConformer, d=1024, 8 heads, head_dim=128, FFN=4096, conv kernel=9 |
| Subsampling   | Conv2d dw_striding stack, 8× temporal (100 → 12.5 fps) |
| Predictor     | 2-layer LSTM, hidden 640, embed 8193 × 640 |
| Joint head    | enc(1024 → 640) + pred(640 → 640) → ReLU → linear(640 → 8198) |
| Vocab         | 8192 SentencePiece tokens (multilingual) |
| Audio         | 16 kHz mono, 128 mel bins, n_fft=512, hop=160, win=400 |
| Parameters    | ~600M |

### TDT decoding

The joint head's 8198-class output splits into `[8192 vocab + 1 blank + 5 duration logits]`. The greedy decode loop alternates two cases:

```
while t < T_enc:
    proj_e = joint.enc @ encoder_out[t]
    while True:
        logits = joint.out @ relu(proj_e + joint.pred @ predictor_h)
        token  = argmax(logits[0..8193])           # vocab + blank
        dur    = argmax(logits[8193..8198])        # 0..4 frames

        if token == blank:
            t += max(1, dur)
            break                                   # advance encoder frame
        else:
            emit (token, t, t + dur)
            advance_predictor(token)
            if dur > 0:
                t += dur
                break
```

The duration head is what makes TDT a "skip-frame transducer": instead of always advancing one encoder frame at a time (RNN-T), it predicts how many frames to skip after each emission. That's also why each emitted token already has `(t_start = t, t_end = t + dur)` — word timestamps come out of the decoder loop directly, not from a post-hoc alignment pass.

### Implementation highlights

- **Encoder is a single ggml graph** built once per slice. The dw_striding subsampling, rel-pos attention with Transformer-XL untied biases, and macaron FFN sandwich all run as ggml ops.
- **BatchNorm folding**: NeMo's BN-after-depthwise-conv layers are folded into the depthwise conv weights at load time, so the encoder graph has zero BN nodes.
- **rel_pos_shift** is implemented as a single zero-cost `ggml_view_3d` (same trick as cohere.cpp's encoder).
- **Predictor LSTM and joint head run as manual F32 CPU loops**, not through ggml. They're called per-emitted-token (not per-frame), the work per step is two small (640 → 4·640) GEMMs, and a graph build per token would dwarf the actual compute.
- **Mel filterbank and Hann window are baked into the GGUF** from `preprocessor.featurizer.fb` and `preprocessor.featurizer.window` in the `.nemo` checkpoint — no recomputation at runtime, no librosa dependency.
- **Self-contained Cooley-Tukey FFT** in pure C++ — no FFTW, no MKL.

## Performance

5.4 s clip, 8-thread x86 CPU, voxpopuli demo:

| Path | Wall | RTF | Peak RSS |
| --- | ---: | ---: | ---: |
| **parakeet Q4_K** | **5.3 s** | **0.97×** | **0.96 GB** |
| parakeet F16    | 9.3 s  | 1.71× | 2.5 GB |
| cohere   Q4_K   | 14.8 s | 2.72× | 3.0 GB |
| cohere   F16    | 27.6 s | 5.07× | 7.1 GB |

See [`benchmark_cohere.md`](benchmark_cohere.md) for the full table including ONNX, Python `transformers`, Rust libtorch, and the `cohere-align` CTC pipeline.

## Repository layout

| Path | Description |
| --- | --- |
| `src/parakeet.{h,cpp}`              | Public C API + ggml runtime (loader, encoder graph, mel, LSTM step, joint head, TDT decode) |
| `models/convert-parakeet-to-gguf.py`| `.nemo → GGUF` converter |
| `examples/parakeet-main/main.cpp`   | CLI with VAD + chunking + SRT/VTT/TXT output |
| `parakeet-todo.md`                  | Implementation plan (mostly complete) |
| `benchmark_cohere.md`               | Cross-runtime benchmark numbers |
| `ggml_plans.md`                     | Notes on the gap to ONNX INT8 and a VNNI Q8_0 plan |

The cohere-related files (`src/cohere.{h,cpp}`, `src/wav2vec2-ggml.{h,cpp}`, `src/align.{h,cpp}`, `examples/cohere-main`, `examples/cohere-align`) are still present and continue to build — this branch is a strict superset of the cohere branch. To use just the cohere runtime, switch back to the [`ggml`](https://github.com/CrispStrobe/cohere-whisper.cpp/tree/ggml) branch.

## Attribution

- **Original model:** [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC-BY-4.0). NVIDIA NeMo team.
- **Cross-check for the joint head + TDT greedy loop:** [`istupakov/onnx-asr`](https://github.com/istupakov/onnx-asr).
- **Encoder graph patterns:** the dw_striding subsampling and Conformer block code follows the same shape as the [`ggml` branch](https://github.com/CrispStrobe/cohere-whisper.cpp/tree/ggml) of this fork (cohere.cpp).
- **Underlying runtime:** [whisper.cpp](https://github.com/ggml-org/whisper.cpp) / [ggml](https://github.com/ggerganov/ggml).

## License

The fork code is MIT (matching whisper.cpp). The parakeet model itself is **CC-BY-4.0**, inherited from NVIDIA. Use of the GGUF files must comply with CC-BY-4.0 including attribution.
