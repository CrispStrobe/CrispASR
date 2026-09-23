---
license: apache-2.0
pipeline_tag: automatic-speech-recognition
language:
- zh
tags:
- audio
- speech-recognition
- chinese-dialects
- gguf
- ggml
- crispasr
base_model: DataoceanAI1/dolphin-cn-dialect-small-streaming
library_name: ggml
---

# Dolphin CN-Dialect small (streaming) — GGUF

GGUF conversions of [`DataoceanAI1/dolphin-cn-dialect-small-streaming`](https://huggingface.co/DataoceanAI1/dolphin-cn-dialect-small-streaming)
for **[CrispASR](https://github.com/CrispStrobe/CrispASR)** (`--backend dolphin`, or `-m dolphin` to auto-download).

Dolphin CN-Dialect, from Dataocean AI and Tsinghua University, recognises Mandarin plus Chinese dialects. It is a
WeNet model: an E-Branchformer encoder, a Transformer attention decoder and a CTC head. CrispASR decodes it
the way upstream `dolphin/transcribe.py` does. It runs a CTC prefix beam search (beam 10) and rescores the n-best
with the attention decoder under the prompt `[sos, <lang>, <region>, <asr>, <notimestamp>]`. The decoder
predicts the language and region, or `-l zh` / `-l zh-SICHUAN` forces them.

## Files

| File | Size | Notes |
| --- | --- | --- |
| `dolphin-cn-dialect-small-streaming-f16.gguf`  | 792 MB | F16 matrices, F32 norms / convs / tables |
| `dolphin-cn-dialect-small-streaming-q8_0.gguf` | 443 MB | Q8_0 |
| `dolphin-cn-dialect-small-streaming-q4_k.gguf` | 258 MB | Q4_K (the default for `-m dolphin`) |

## Verification

The F16 file was checked stage by stage against the upstream `dolphin` package (float32, CPU, dither 0) with
`crispasr-diff dolphin`. The checked stages are the fbank, the subsampling output, every encoder block, the encoder
output, the CTC log-probs and the final text. Every stage passed, and the transcripts are identical to upstream
on a Mandarin clip and on `jfk.wav`.

Quantised transcripts against F16 on 15 in-domain clips: Mandarin, Sichuan, Tianjin and Henan dialects,
Cantonese, and zh-en code-switching.

| | identical to F16 | differences |
| --- | --- | --- |
| Q8_0 | 13 / 15 | one dropped trailing 啊; one character (点 → 哪, Cantonese) |
| Q4_K | 13 / 15 | one dropped trailing 啊; one dropped trailing 的 |

Encoder-output cosines are less smooth than the transcripts suggest. A handful of frames sit near decision
boundaries, so the worst-frame cosine moves a lot under quantisation while the mean stays at ≥ 0.996 for Q8_0.
English (out of domain for this model) drifts more at Q4_K.

## Licence

Apache-2.0, as upstream.
