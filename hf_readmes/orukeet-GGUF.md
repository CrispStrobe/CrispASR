---
license: cc-by-sa-4.0
language:
- bg
- cs
- da
- de
- el
- en
- es
- et
- fi
- fr
- hr
- hu
- it
- lt
- lv
- mt
- nl
- pl
- pt
- ro
- ru
- sk
- sl
- sv
- uk
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- ggml
- gguf
- parakeet
- tdt
- fastconformer
- multilingual
- crispasr
library_name: ggml
base_model: oruk/orukeet
---

# Orukeet — GGUF

GGUF conversions of [`oruk/orukeet`](https://huggingface.co/oruk/orukeet) (checkpoint **r3**) for
**[CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Orukeet is Oruk AI's fine-tune of [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
Half of the encoder's temporal depthwise convolution kernels (12,288 of 24,576) were replaced with fitted
Gabor functions and frozen, and the model was then re-adapted. Tensor shapes and operators are unchanged, so it
runs on CrispASR's existing Parakeet runtime: 25 European languages, automatic language detection, TDT word
timestamps. See the [technical report](https://huggingface.co/oruk/orukeet/blob/main/docs/technical-report.md).

> **About the benchmark numbers.** Orukeet's final adaptation step trained on all 2,939 LibriSpeech
> test-other recordings (report: "Test-other is used for training, checkpoint selection and
> re-evaluation"). Its LibriSpeech test-other score is therefore not a held-out result. Judge it on
> other test sets.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `orukeet-f16.gguf`  | 1.26 GB | F16 |
| `orukeet-q8_0.gguf` | 674 MB  | Q8_0 |
| `orukeet-q5_0.gguf` | 470 MB  | Q5_0 |
| `orukeet-q4_k.gguf` | 402 MB  | **Q4_K, the default for `-m orukeet`** |

Converted from `orukeet-v0.1.0.nemo` at revision `555136b5` (SHA-256 `031c8dda…73b56`, matching the report)
with `models/convert-parakeet-to-gguf.py`, and quantised with `crispasr-quantize`.

## Verification against NeMo

The same WAVs were run through the original checkpoint in NeMo (greedy TDT, CPU) and through CrispASR:

| clip | NeMo | F16 | Q8_0 | Q4_K |
| --- | --- | :-: | :-: | :-: |
| en, `samples/jfk.wav` | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. | = | = | = |
| de | Guten Morgen. Die Sitzung beginnt heute um neun Uhr im großen Saal. | = | = | punct. |
| fr | Bonjour à tous, la réunion commence demain matin à huit heures. | = | = | = |
| es | Buenas tardes. El tren para Madrid sale a las cinco y media. | = | = | punct. |

`=` means identical text. `punct.` means identical words with one sentence-final period rendered as a comma.

Per-stage comparison with `crispasr-diff` against a NeMo reference dump of the same checkpoint
(`tools/dump_reference.py --backend parakeet`), F16 GGUF, worst frame (`cos_min`):

| stage | en (jfk, 138 frames) | de (61 frames) |
| --- | --- | --- |
| mel spectrogram | 1.000000 | 1.000000 |
| pre-encode | 0.999998 | 0.999986 |
| encoder layers 0–23 (worst layer) | ≥ 0.999981 | ≥ 0.999942 |
| encoder output | 0.999883 | 0.999988 |

The quantised files differ from NeMo by ordinary quantisation error (Q8_0 encoder-output `cos_mean` 0.9996 / 0.9961),
and the transcripts above are the end-to-end check for them.

## Usage

```bash
# auto-download the Q4_K file and transcribe
crispasr -m orukeet -f audio.wav

# or point at a downloaded file; the Parakeet backend is detected from the GGUF
crispasr -m orukeet-q8_0.gguf -f audio.wav -osrt
```

Every Parakeet option applies, including VAD, chunking, word timestamps and subtitle output.

## Licence and attribution

The weights are **CC BY-SA 4.0**, inherited from Orukeet, so these GGUF files are distributed under the same
licence: you must credit the authors and share derivatives under the same terms.

- Base model: [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), CC BY 4.0, NVIDIA.
- Fine-tune: [`oruk/orukeet`](https://huggingface.co/oruk/orukeet), CC BY-SA 4.0, Oruk AI (Roll, Yi, Marşan,
  Grenez, Stein, Mrkaic, Padjin, Zeljkovic, Graham).
- Conversion and runtime: [CrispASR](https://github.com/CrispStrobe/CrispASR). The only change made here is the
  format conversion and quantisation.
