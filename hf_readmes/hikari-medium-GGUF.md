---
license: mit
language:
- en
- de
- ja
- ru
base_model:
- sbintuitions/hikari-medium
pipeline_tag: automatic-speech-recognition
tags:
- speech-translation
- simultaneous-translation
- streaming-asr
- whisper
- gguf
- crispasr
library_name: ggml
---

# Hikari-medium — GGUF (ggml)

GGUF / ggml conversion of [`sbintuitions/hikari-medium`](https://huggingface.co/sbintuitions/hikari-medium) (revision `178bd876`) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)** (`--backend hikari`).

Hikari is a Whisper-medium encoder-decoder with a **causal** encoder. Every 80 ms of audio it decides whether to emit the next token or wait, so it translates **while the speaker is still talking**: English speech into German, Japanese or Russian text, or English transcription. See the [paper](https://arxiv.org/abs/2603.11578) and the [upstream code](https://github.com/sbintuitions/hikari).

## Licence

MIT, as the original ([LICENSE](https://github.com/sbintuitions/hikari/blob/main/LICENSE)). This repository is a format conversion; the f16 weights are unchanged.

## Files

| File | Size | Against the reference implementation |
|---|---:|---|
| `hikari-medium-f16.gguf` | 1.53 GB | **exact**: every stream step and the text equal on the test clips (below); same output on CPU and Metal. Recommended. |
| `hikari-medium-q8_0.gguf` | 873 MB | same text on jfk (159/161 steps); on Apple Metal it changed one German sentence of a 27 s clip, on CPU not. |
| `ggml-silero-v6.2.0.bin` | 0.9 MB | Silero VAD. Required: its speech probability drives the policy's wait penalty, and without it the model hardly ever emits. |

"Reference" is the PyTorch model driven by a re-implementation of the upstream server's streaming policy (CrispASR `tools/reference_backends/hikari.py`), compared stage by stage with `crispasr-diff hikari`:

| clip (en→de) | mel | encoder cos_min | decoder logits cos_min | stream steps | text |
|---|---|---|---|---|---|
| jfk.wav (11 s) | 1.000000 | 0.99978 | 0.99956 (argmax 164/164) | 161/161 | identical |
| jfk 0–4 s | 1.000000 | 0.99961 | 0.99996 (51/51) | 48/48 | identical |

A q4_k file is not published: it broke the encoder (cos_min 0.36) and changed the text.

## Quick start

```bash
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# English speech -> German text (-m auto fetches the f16 + Silero)
./build/bin/crispasr --backend hikari -m auto -l en -tl de -f talk.wav
# -> Japanese / Russian: -tl ja / -tl ru ; English transcription: omit -tl

# Live: German text appears while the speaker talks
./build/bin/crispasr --stream --backend hikari -m auto -l en --tr-tl de
```

Example (27 s English talk, f16, `-tl de`): *"Guten Morgen, alle zusammen, und danke, dass Sie gekommen sind. Heute möchte ich darüber sprechen, wie kleine Teams zuverlässige Software liefern können. …"*

## Speed

Not real time yet on an Apple M1: 1.8 s per second of audio on Metal (per 80 ms step about 87 ms encoder + 58 ms decoder), slower on CPU. The encoder already runs incrementally; the per-step cost is what remains. Faster GPUs should do better; upstream serves it on A100/H100.

## Conversion

```bash
python models/convert-hikari-to-gguf.py --input <dir>/hikari-medium --output hikari-medium-f16.gguf
./build/bin/crispasr-quantize hikari-medium-f16.gguf hikari-medium-q8_0.gguf q8_0
python tools/dump_reference.py --backend hikari ...   # reference archive for crispasr-diff
```
