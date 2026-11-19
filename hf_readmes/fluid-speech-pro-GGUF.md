---
license: cc-by-4.0
pipeline_tag: automatic-speech-recognition
language:
- en
- bg
- hr
- cs
- da
- nl
- et
- fi
- fr
- de
- el
- hu
- it
- lv
- lt
- mt
- pl
- pt
- ro
- sk
- sl
- es
- sv
- ru
- uk
tags:
- audio
- speech-recognition
- gguf
- ggml
- crispasr
- parakeet
- tdt
- fastconformer
base_model: FluidVoice-Labs/fluid-speech-pro
library_name: ggml
---

# Fluid Speech Pro — GGUF

GGUF conversions of **Fluid Speech Pro by FluidVoice**
([`FluidVoice-Labs/fluid-speech-pro`](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro), CC-BY-4.0) for
**[CrispASR](https://github.com/CrispStrobe/CrispASR)** (`-m fluid-speech-pro`). The model detects the language
itself and writes punctuation and capitals. It covers English and 24 other European languages.

Fluid Speech Pro is [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) plus one
small residual block between the encoder's subsampling and its first Conformer layer:
LayerNorm → Linear → depthwise Conv1d (kernel 15) → GELU → Linear. These files were converted from the reference
`.nemo` checkpoint, so the block is included. The upstream transformers weights and the upstream parakeet.cpp GGUF
leave it out. CrispASR's parakeet backend runs the block whenever its tensors are present.

## Files

| File | Size | Notes |
| --- | --- | --- |
| `fluid-speech-pro-f16.gguf` | 1.26 GB | F16 |
| `fluid-speech-pro-q8_0.gguf` | 677 MB | Q8_0 (default for `-m fluid-speech-pro`) |
| `fluid-speech-pro-q4_k.gguf` | 403 MB | Q4_K: smallest, some drift on code-switched speech |
| `fluid-speech-pro-clean-f16.gguf` | 1.26 GB | clean mode, F16 |
| `fluid-speech-pro-clean-q8_0.gguf` | 677 MB | clean mode, Q8_0 (default for `-m fluid-speech-pro-clean`) |
| `fluid-speech-pro-clean-q4_k.gguf` | 403 MB | clean mode, Q4_K |

**Clean mode** produces edited text instead of a verbatim record: it drops repeated words and fillers such as
"um" and "uh". Upstream it is an optional low-rank adapter. In the `-clean` files it was folded into the weights
at conversion, exactly as `fluid_speech.py` folds it. Don't use clean mode where you need an exact record.

## Usage

```bash
crispasr -m fluid-speech-pro --auto-download -f audio.wav
crispasr -m fluid-speech-pro-clean --auto-download -f audio.wav     # edited, non-verbatim text
crispasr -m fluid-speech-pro-q8_0.gguf -f audio.wav                 # a local file
```

Conversion: `python models/convert-parakeet-to-gguf.py --nemo fluid-speech-pro.nemo --output fluid-speech-pro-f16.gguf [--clean-mode]`,
then `crispasr-quantize <f16> <out> q8_0|q4_k`.

## Validation

Compared against NeMo running upstream's loader (block attached, every tensor loaded strictly, dither 0):

- **F16 stage diff (`crispasr-diff parakeet`):** all 28 stages pass (mel, block output, 24 encoder layers,
  encoder output) on jfk.wav and multispeaker.wav, for both base and clean mode. Encoder output cosine is
  ≥ 0.99997. Run on heavy-cpu CI with `tools/ci-heavy/fluid_speech_validate.py`.
- **Transcripts:** F16 and Q8_0 match NeMo on jfk.wav, multispeaker.wav and a 30 s German/English
  code-switched clip, for both base and clean mode. On multispeaker.wav, clean-mode Q8_0 differs from NeMo by
  whitespace or punctuation only (word error rate 0). Q4_K matches on the English clips but changes 3–4 words
  in the code-switched opening.
- **Q8_0 / Q4_K stage tolerances:** the quantized files fail the F16-tuned per-stage thresholds, as stock
  parakeet-tdt-0.6b-v3 converted the same way does. Worst-frame encoder cosine for Q8_0 is 0.933 (Fluid) vs
  0.988 (v3) on jfk.wav, and 0.9946 vs 0.9957 on multispeaker.wav.
- Source checkpoint: `fluid-speech-pro.nemo` at revision `46027a3b`, sha256 `f3e377b792ae5a52a3c071adb7eba5504306f2f4bdcf933ae9da2cbcab6f0449`.

## License and attribution

CC BY 4.0. Please credit the model as **Fluid Speech Pro by FluidVoice**.

- Model: [FluidVoice-Labs/fluid-speech-pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) (CC-BY-4.0)
- Built on [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and
  [Moondream Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra), both CC-BY-4.0
- Changes in this repository: format conversion to GGUF, quantization, and (for the `-clean` files) the upstream
  clean-mode adapter folded into the weights. No retraining.
