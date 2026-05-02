---
license: apache-2.0
language:
- en
- fr
- de
- es
- pt
base_model:
- ibm-granite/granite-speech-4.1-2b-plus
tags:
- asr
- speech
- gguf
- crispasr
- granite-speech-plus
- speaker-attributed
- word-timestamps
---

# granite-speech-4.1-2b-plus — GGUF

GGUF conversion of [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus)
for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

The PLUS variant adds two capabilities over the base 4.1-2b:

- **Punctuated and capitalised transcripts by default** — no special
  prompt required.
- **Speaker labels and word-level timestamps** in the model's structured
  output (full output parsing in CrispASR is the next step; raw text
  works today).

Architecturally PLUS is the base 4.1-2b plus a single change: the
encoder's layer-3 hidden state is concatenated with the final layer
output (config: `cat_hidden_layers: [3]`), producing a 2048-dim
projector input instead of 1024. The Q-Former cross-attention K/V
projection weights are correspondingly `(1024, 2048)`.

## Files

| File | Quantisation | Size | Notes |
|---|---|---|---|
| `granite-speech-4.1-2b-plus-f16.gguf` | F16 | ~5.6 GB | Encoder + projector in F32, LLM weights in F16 — full parity reference |
| `granite-speech-4.1-2b-plus-q4_k.gguf` | Q4_K | ~2.96 GB | **Recommended.** LLM layers Q4_K; encoder + projector kept F32 (precision-sensitive). Bit-identical-quality to F16 on encoder + projector |
| `granite-speech-4.1-2b-plus-q4_k-f16enc.gguf` | Q4_K + F16 encoder | ~2.28 GB | LLM Q4_K, encoder + projector F16 (norms / biases / BN stats stay F32). ~700 MB smaller than the recommended Q4_K with no measurable parity loss on this clip |
| `granite-speech-4.1-2b-plus-q4_k-mini.gguf` | Q4_K (aggressive) | ~1.66 GB | Encoder, projector and LLM all Q4_K. Smallest / fastest to download. **Cosine parity is noticeably worse on PLUS than on the base 4.1 mini** because the layer-3 + final hidden-state concat (the architectural delta in PLUS) doubles the surface for Q4_K rounding error. JFK still transcribes correctly with light punctuation drift, but harder material is more likely to regress than on base-4.1 mini. Use Q4_K or Q4_K-f16enc unless disk size is the binding constraint |

## Cosine parity (vs PyTorch BF16 reference, JFK 11 s clip)

| Stage | F16 cos_min | Q4_K cos_min | Q4_K-f16enc cos_min | Q4_K-mini cos_min |
|---|---|---|---|---|
| mel_spectrogram | 0.999997 | 0.999997 | 0.999997 | 0.999997 |
| encoder_out | 0.999938 | 0.999938 | 0.999938 | 0.622 |
| projector_out | 0.999995 | 0.999995 | 0.999995 | 0.960 |

`encoder_out` is the 2048-dim concatenation of the layer-3 hidden state
and the final encoder layer (the PLUS architectural delta). On the
recommended and `-f16enc` files the encoder weights stay in F32/F16, so
parity is essentially indistinguishable from the F16 reference. On the
`-mini` file the encoder weights are Q4_K — rounding error compounds
across the 16-layer Conformer and shows up amplified after the concat,
which is why `encoder_out` cos_min drops to ~0.62 on PLUS where base-4.1
mini sits at ~0.93. End-to-end JFK transcription is still correct.

_Tested with `crispasr-diff granite-4.1 <model.gguf> <ref.gguf> samples/jfk.wav`_

## Usage with CrispASR

```bash
# auto-download and transcribe
crispasr --backend granite-4.1-plus -m auto samples/audio.wav

# or with explicit path
crispasr --backend granite-4.1-plus \
  -m granite-speech-4.1-2b-plus-f16.gguf \
  samples/audio.wav
```

End-to-end example on the JFK 11s clip:

```
$ crispasr --backend granite-4.1-plus -m auto samples/jfk.wav
And so my fellow Americans, ask not what your country can do for
you, ask what you can do for your country.
```

(Note the punctuation + capitalisation that the base 4.1-2b only
produces with an explicit `--ask "transcribe with proper
punctuation..."` prompt.)

## Architecture

| Stage | Description |
|---|---|
| Encoder | 16-layer Macaron Conformer (1024 dim, 8 heads, 15-tap depthwise conv). Hidden state at layer 3 is captured and concatenated with the final layer output → 2048-dim projector input. |
| Projector | 2-layer BLIP-2 Q-Former. Cross-attention K/V weights are `(1024, 2048)` to consume the wider concatenated encoder feature. 3 learned query tokens per 15-frame window. |
| LLM | Granite 4.0-1B (40 layers, 2048 hidden, GQA 16/4, SwiGLU, RoPE θ=10000, μP multipliers). |

Total ~2.2 B parameters. The "+" capability is encoded entirely in
training — the architectural delta from base is just the layer
concatenation.

## Conversion

```bash
# Convert HF safetensors → GGUF F16
python models/convert-granite-speech-to-gguf.py \
  --input /path/to/granite-speech-4.1-2b-plus \
  --output granite-speech-4.1-2b-plus-f16.gguf

# Quantise F16 → Q4_K (encoder + projector preserved F32, LLM Q4_K)
crispasr-quantize granite-speech-4.1-2b-plus-f16.gguf \
                  granite-speech-4.1-2b-plus-q4_k.gguf q4_k

# Q4_K with F16 encoder/projector (smaller, no measurable parity loss)
CRISPASR_GRANITE_ENC_F16=1 \
crispasr-quantize granite-speech-4.1-2b-plus-f16.gguf \
                  granite-speech-4.1-2b-plus-q4_k-f16enc.gguf q4_k

# Aggressive Q4_K everywhere (encoder + projector + LLM)
CRISPASR_GRANITE_QUANT_ALL=1 \
crispasr-quantize granite-speech-4.1-2b-plus-f16.gguf \
                  granite-speech-4.1-2b-plus-q4_k-mini.gguf q4_k
```

The same converter handles base / 4.1-2b / 4.1-2b-plus from a single
script — variant detection happens via `config.json` keys
(`cat_hidden_layers`, `encoder_hidden_size`).

## Licence

Apache 2.0 — same as the original
[ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus).
