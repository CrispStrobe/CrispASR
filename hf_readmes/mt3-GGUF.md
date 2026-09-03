---
license: apache-2.0
base_model:
- google/mt3
pipeline_tag: audio-classification
tags:
- music-transcription
- note-events
- multi-instrument
- mt3
- amt
- midi
- gguf
- crispasr
library_name: ggml
---

# MT3 — GGUF (ggml)

GGUF conversion of Google Magenta's **MT3** multi-instrument music transcription model (Apache-2.0) for **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)** (§250).

MT3 is a T5-style encoder/decoder that reads a log-mel spectrogram and emits MIDI-like event tokens, transcribing **multiple instruments at once** — every note carries a General MIDI program, and drums are a separate class. It is the only per-program note-event model in CrispASR (piano-transcription is 88-key piano, basic-pitch is single-track polyphonic).

## Files

| File | Size | Notes |
|---|---:|---|
| `mt3-f16.gguf` | 96 MB | 190 tensors; encoder+decoder, sinusoidal absolute positions, event codec |

Converted with [`models/convert-mt3-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-mt3-to-gguf.py), which decodes the T5X/zarr checkpoint with stdlib + numpy only (no JAX, t5x or TensorStore). Weights verified **bit-exact** against the reference PyTorch export (189/189 tensors, max abs diff 0.0).

## Usage

```bash
crispasr --piano --backend mt3 -m auto --auto-download -f song.wav
crispasr --piano -m mt3-f16.gguf -f song.wav --piano-format json   # + program/instrument/is_drum
crispasr --piano -m mt3-f16.gguf -f song.wav --piano-format midi   # multi-track SMF
```

`--piano-format midi` writes a format-1 Standard MIDI File with one track per program and drums on GM channel 10.

## Parity

Validated stage-by-stage against a numpy reference implementation of the upstream graph (`tools/reference_backends/mt3.py`):

| stage | result |
|---|---|
| mel front end | cos 1.000000000 |
| encoder output | cos 0.999999879 |
| first-step decoder logits | cos 0.999999999, argmax identical |
| greedy tokens | identical, 22/22 steps |
| **end-to-end note events** | **exact positional match** (88/88, 5/5, 8/8 across three clips) |

The GGUF records `mt3.pos_embed=sinusoidal` and `mt3.use_relative_attention_bias=0`; the runtime hard-fails rather than silently falling back to T5's relative-bucket attention, which is a known failure mode of other ports.
