---
license: apache-2.0
language:
- en
- fr
- de
- es
- pt
base_model:
- ibm-granite/granite-speech-4.1-2b-nar
tags:
- asr
- speech
- gguf
- crispasr
- granite-speech-nar
- non-autoregressive
---

# granite-speech-4.1-2b-nar — GGUF

GGUF conversion of [ibm-granite/granite-speech-4.1-2b-nar](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar)
for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

The NAR variant replaces the autoregressive Granite decoder with a
**non-autoregressive** one — the LLM runs **once** over the full
sequence with `is_causal=False` everywhere instead of token-by-token
sampling. The slot positions in the LLM input absorb edit votes; a
per-row argmax + `unique_consecutive` + drop-EOS gives the final
transcript. Throughput is several times higher than the
autoregressive variants on the same audio.

Architecturally NAR differs from base 4.1-2b in three places:

- **Encoder self-conditioning at layer 8** — the layer-8 CTC softmax
  is fed back into the hidden stream as a 1024-dim residual. The
  per-frame blank probability captured here also drives the
  posterior-weighted pool of the BPE auxiliary head.
- **BPE auxiliary CTC head** (100353-vocab) on a posterior-pooled
  (window=4) view of the final hidden states. Its greedy decode
  initialises the LLM input text.
- **4-layer encoder hidden-state concatenation** for the projector
  input (`[layer 4, 8, 12, last]` → 4×1024 = 4096-dim), vs. base's
  single layer and PLUS's 2-layer concat.

## Files

| File | Quantisation | Size | Notes |
|---|---|---|---|
| `granite-speech-4.1-2b-nar-f16.gguf` | F16 | ~5.4 GB | Encoder + projector in F32, LLM weights in F16 — full parity reference |
| `granite-speech-4.1-2b-nar-q4_k.gguf` | Q4_K | ~3.2 GB | **Recommended.** LLM layers Q4_K; encoder + projector kept F32 (precision-sensitive). Bit-identical-quality to F16 on encoder + projector |
| `granite-speech-4.1-2b-nar-q4_k-f16enc.gguf` | Q4_K + F16 encoder | ~2.4 GB | LLM Q4_K, encoder + projector F16 (norms / biases / BN stats stay F32). ~800 MB smaller than the recommended Q4_K with no measurable parity loss on this clip |
| `granite-speech-4.1-2b-nar-q4_k-mini.gguf` | Q4_K (aggressive) | ~1.5 GB | Encoder, projector and LLM all Q4_K. Smallest / fastest to download. **Cosine parity is noticeably worse on NAR than on base or PLUS mini** because the 4-layer hidden-state concat (the architectural delta in NAR) quadruples the surface for Q4_K rounding error. JFK still transcribes correctly because the LLM's argmax recovers the right token; harder material is more likely to regress. Use Q4_K or Q4_K-f16enc unless disk size is the binding constraint |

## Cosine parity (vs PyTorch BF16 reference, JFK 11 s clip)

| Stage | F16 cos_min | Q4_K cos_min | Q4_K-f16enc cos_min | Q4_K-mini cos_min |
|---|---|---|---|---|
| mel_spectrogram | 0.999997 | 0.999997 | 0.999997 | 0.999997 |
| encoder_out | 0.999852 | 0.999852 | 0.999852 | 0.104 |
| encoder_logits (CTC) | 0.999675 | 0.999675 | 0.999675 | 0.864 |
| projector_out | 0.999999 | 0.999999 | 0.999999 | 0.965 |
| editing_logits | 0.999999 | 0.956 | 0.956 | 0.956 |
| editing_logits_top1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| transcribe == ref | ✅ | ✅ | ✅ | ✅ |

`encoder_out` is the 4096-dim concatenation of layer 4, 8, 12 and the
final encoder layer (the NAR architectural delta). The recommended
and `-f16enc` files keep the encoder in F32/F16 so parity matches the
F16 reference exactly. On `-mini` the encoder is fully Q4_K — rounding
error compounds across the 16-layer Conformer and the 4-way concat
amplifies the worst-frame divergence to cos_min ≈ 0.10.

`editing_logits` raw cosine sits at ~0.956 on every Q4_K variant —
that's expected LLM-quantization noise across 100 K vocab logits. The
argmax (`editing_logits_top1`) and the resulting transcript are
unchanged: all four files reproduce the reference final text exactly.

Reference transcript: *"and so, my fellow americans, ask not what your country can do for you. ask what you can do for your country."*

_Tested with `crispasr-diff granite-nle <model.gguf> <ref.gguf> samples/jfk.wav`_

## Usage with CrispASR

NAR uses a separate runtime from the autoregressive granite variants.
Today it is reachable via the `crispasr-diff` harness and the
`granite_nle` library directly; a `granite-4.1-nar` backend in the
main `crispasr` CLI is the next step (see [TODO.md](https://github.com/CrispStrobe/CrispASR/blob/main/TODO.md)).

```bash
# Bit-exact end-to-end transcribe via the diff harness
crispasr-diff granite-nle \
  granite-speech-4.1-2b-nar-q4_k.gguf \
  /path/to/ref.gguf \
  samples/jfk.wav
```

The library entry point is `granite_nle_transcribe(ctx, samples,
n_samples)` in [`src/granite_nle.h`](https://github.com/CrispStrobe/CrispASR/blob/main/src/granite_nle.h);
it returns a malloc'd UTF-8 string with the final transcript. There
are also fine-grained accessors (`compute_mel`, `run_encoder`,
`run_projector`, `run_llm_editing`) for partial-pipeline use.

Supported languages: English, French, German, Spanish, Portuguese.

## Architecture

| Stage | Description |
|---|---|
| Encoder | 16-layer Macaron Conformer (1024 dim, 8 heads, 15-tap depthwise conv). **Self-conditioning at layer 8** (CTC softmax → 1024-dim residual back into the hidden stream). **BPE auxiliary CTC head** (100353-vocab) on a posterior-pooled (window=4) view of the final hidden states, weighted by `1 - blank_prob_mid` from the layer-8 self-conditioning softmax. **4-layer hidden-state concatenation** `[layer 4, 8, 12, last]` → 4096-dim projector input. |
| Projector | 2-layer simplified Q-Former with 32 attention heads. Block size 15, downsample rate 5 → 3 audio tokens per 15-frame window. Projects 4096-dim concatenated encoder feature down to 2048-dim LLM input. |
| LLM (NAR) | Granite 4.0-1B (40 layers, 2048 hidden, GQA 16/4, SwiGLU, RoPE θ=10000, μP multipliers). Every `self_attn` layer runs with `is_causal=False`. Single forward pass over `[audio_embs, text_with_insertion_slots]`; the slot logits decode into the final transcript via argmax + `unique_consecutive` + drop EOS. |

Total ~2.2 B parameters. NAR throughput is several times higher than
the autoregressive variants because there is no token-by-token
sampling loop.

## Conversion

```bash
# Convert HF safetensors → GGUF F16
python models/convert-granite-nle-to-gguf.py \
  --input /path/to/granite-speech-4.1-2b-nar \
  --output granite-speech-4.1-2b-nar-f16.gguf

# Quantise F16 → Q4_K (encoder + projector preserved F32, LLM Q4_K)
crispasr-quantize granite-speech-4.1-2b-nar-f16.gguf \
                  granite-speech-4.1-2b-nar-q4_k.gguf q4_k

# Q4_K with F16 encoder/projector (smaller, no measurable parity loss)
CRISPASR_GRANITE_ENC_F16=1 \
crispasr-quantize granite-speech-4.1-2b-nar-f16.gguf \
                  granite-speech-4.1-2b-nar-q4_k-f16enc.gguf q4_k

# Aggressive Q4_K everywhere (encoder + projector + LLM)
CRISPASR_GRANITE_QUANT_ALL=1 \
crispasr-quantize granite-speech-4.1-2b-nar-f16.gguf \
                  granite-speech-4.1-2b-nar-q4_k-mini.gguf q4_k
```

The NAR converter is a separate script (`convert-granite-nle-to-gguf.py`)
because the GGUF arch (`granite_nle`), tensor naming (BPE auxiliary
head, 4-layer hidden capture indices) and self-conditioning metadata
all differ from the autoregressive variants. The `crispasr-quantize`
binary recognises both `granite_speech` and `granite_nle` archs and
applies identical encoder/projector skip rules to both.

## Licence

Apache 2.0 — same as the original
[ibm-granite/granite-speech-4.1-2b-nar](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar).
