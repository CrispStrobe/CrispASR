---
license: cc-by-4.0
language:
- ja
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
- japanese
library_name: ggml
base_model: nvidia/parakeet-tdt_ctc-0.6b-ja
---

# Parakeet TDT-CTC 0.6B (Japanese) — GGUF

GGUF / ggml conversions of [`nvidia/parakeet-tdt_ctc-0.6b-ja`](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) for use with the `crispasr` CLI from **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

A 600 M-parameter Japanese ASR model with punctuation:

- **Hybrid FastConformer-TDT-CTC**: TDT (Token-and-Duration Transducer) decoder by default, CTC available as a fallback (Python only — the GGUF runtime exercises the TDT path).
- **Built-in word-level timestamps** from the TDT duration head — no separate CTC alignment.
- **6.4 % CER on JSUT basic5000**.
- **CC-BY-4.0** licence.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `parakeet-tdt-0.6b-ja.gguf`        | 1.24 GB | F16, **bit-exact match with NeMo on JSUT samples** |
| `parakeet-tdt-0.6b-ja-q4_k.gguf`   | ~470 MB | Q4_K — degrades after ~8 tokens on this model, see note below |

## Recommended: F16

Verified on a JSUT-basic5000 sample at F16:

```
NeMo (PyTorch): '水をマレーシアから買わなくてはならないのです。'
crispasr (F16): '水をマレーシアから買わなくてはならないのです。'
```

The F16 GGUF produces an **identical transcript** to the official NeMo Python pipeline.

### About the Q4_K variant

The Japanese model uses an **80-mel** preprocessor (vs. 128 for the
multilingual v3) and a smaller, more sensitive encoder distribution.
With our default Q4_K quantisation, two of the most logit-shaping
tensors (`joint.pred.weight`, `decoder.embed.weight`) fall back to
`q4_0` because their dimensions don't tile cleanly for q4_k blocks.
This is enough quantisation noise that the TDT decoder enters a
fixed-point loop after the first ~8 tokens. **If you don't need a
small file, prefer the F16.** A Q5_K build (or Q4_K with the two
above tensors pinned to F16) is on the roadmap.

## Quick start

```bash
# 1. Build the runtime
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target crispasr

# 2. Download the F16
huggingface-cli download cstr/parakeet-tdt-0.6b-ja-GGUF \
    parakeet-tdt-0.6b-ja.gguf --local-dir .

# 3. Transcribe a 16 kHz mono WAV
./build/bin/crispasr --backend parakeet \
    -m parakeet-tdt-0.6b-ja.gguf -f your-japanese-audio.wav -t 8
```

You can also let crispasr auto-download the model:

```bash
./build/bin/crispasr --backend parakeet -m auto --auto-download \
    -f your-japanese-audio.wav --model-name parakeet-ja
```

## Word-level timestamps for free

Pass `-v` to dump per-token timestamps from the TDT duration head. Each token spans one or more encoder frames; one frame = **80 ms**. No separate alignment model required.

## Model architecture

| Component | Details |
| --- | --- |
| Encoder       | 24-layer FastConformer, d=1024, 8 heads, head_dim=128, FFN=4096, conv kernel=9 |
| Encoder input | **xscaling = `True`** (input × √d_model = 32 before the first block) |
| Subsampling   | Conv2d dw_striding stack, 8× temporal (50 → 12.5 fps) |
| Predictor     | 2-layer LSTM, hidden 640, embed 3073 × 640 (blank padding-idx) |
| Joint head    | enc(1024 → 640) + pred(640 → 640) → ReLU → linear(640 → 3078) |
| Vocab         | 3072 SentencePiece tokens (Japanese, with punctuation) |
| Audio         | 16 kHz mono, **80 mel bins**, n_fft=512, hop=160, win=400 |
| Parameters    | ~600 M |

`xscaling=True` is the most important architectural detail vs. the
multilingual v3 model — `nvidia/parakeet-tdt-0.6b-v3` uses
`xscaling=False`. Both settings are stored in the GGUF metadata
(`parakeet.xscaling`) and read by the runtime, so the same code path
serves both variants without per-model branches.

## How this was made

1. The `.nemo` checkpoint is unpacked; every architecture hyperparameter
   (d_model, n_layers, ff_dim, pred_hidden, joint_hidden, xscaling, …)
   is read from `model_config.yaml` and cross-checked against the actual
   tensor shapes. The mel filterbank and Hann window are baked directly
   into the GGUF (`preprocessor.fb`, `preprocessor.window`).
2. NeMo state-dict keys are remapped to ggml-friendly names. Weights are
   written as F16 for matmul tensors and F32 for norms / biases / mel
   filterbank. A synthetic zero `conv.dw.bias` is added per encoder
   layer when the checkpoint omits it (older NeMo BN-only convs).
3. Inference is implemented in `src/parakeet.{h,cpp}`: the FastConformer
   encoder runs as a single ggml graph (BN folded into the depthwise
   conv weights at load time, xscaling applied between the pre-encode
   and the first block when `parakeet.xscaling=true`), the LSTM
   predictor and joint head run as manual F32 CPU loops, and the TDT
   greedy decode loop alternates "advance encoder frame" / "emit token +
   advance predictor" using the duration head's argmax.

## Attribution

- **Original model:** [`nvidia/parakeet-tdt_ctc-0.6b-ja`](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) (CC-BY-4.0). NVIDIA NeMo team.
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR).

## Related

- Multilingual sibling (25 EU languages): [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF)
- C++ runtime: [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR)

## License

CC-BY-4.0, inherited from the base model. Use of these GGUF files must comply with the CC-BY-4.0 license including attribution.
