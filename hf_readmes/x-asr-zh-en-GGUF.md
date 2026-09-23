---
license: apache-2.0
pipeline_tag: automatic-speech-recognition
language:
- zh
- en
tags:
- audio
- speech-recognition
- streaming
- zipformer
- transducer
- gguf
- ggml
- crispasr
base_model: GilgameshWind/X-ASR-zh-en
library_name: ggml
---

# X-ASR zh-en — GGUF

GGUF conversions of [`GilgameshWind/X-ASR-zh-en`](https://huggingface.co/GilgameshWind/X-ASR-zh-en)
for **[CrispASR](https://github.com/CrispStrobe/CrispASR)** (`--backend xasr`, or `-m xasr` to auto-download).

X-ASR is an icefall streaming Zipformer2 transducer for Chinese and English with punctuation and casing, trained
on about a million hours of speech. Upstream ships it as sherpa-onnx exports for four chunk sizes: 160, 480, 960 and
1920 ms. The four exports share every weight bit for bit, so one GGUF serves all of them, and
`CRISPASR_XASR_CHUNK_MS` picks the chunk size (default 480).

CrispASR decodes exactly as sherpa-onnx's `OnlineRecognizer` greedy search does. That covers the same feature
windows and caches carried from chunk to chunk, at most one symbol per frame, and the same text rendering. It also
runs the model as a true streaming recogniser: the realtime WebSocket endpoint
(`crispasr --server --ws-port N`, then `ws://…:N+1/v1/realtime`) emits append-only text as audio arrives.

## Files

| File | Size | Notes |
| --- | --- | --- |
| `x-asr-zh-en-f16.gguf`  | 310 MB | F16 matrices, F32 convs / norms / tables |
| `x-asr-zh-en-q8_0.gguf` | 168 MB | Q8_0 (the default for `-m xasr`) |
| `x-asr-zh-en-q4_k.gguf` |  93 MB | Q4_K |

Converted from the sherpa-onnx exports. The repository's `streaming_exp/pretrained.pt` is a different, earlier
checkpoint: none of its tensors equal the exported weights, and it transcribes poorly.

## Verification

The reference is the icefall PyTorch modules, loaded with the exported weights and driven chunk by chunk like
sherpa-onnx. That reference reproduces onnxruntime's encoder output (cos ≥ 0.9999997) and sherpa-onnx's text exactly
on both clips at 480 ms and 160 ms. Against it, `crispasr-diff xasr` on the F16 file:

| stage | jfk (480 ms) | zh (480 ms) | jfk (160 ms) | zh (160 ms) |
| --- | --- | --- | --- | --- |
| fbank, embed, 6 Zipformer stacks, encoder output: worst-frame cos | ≥ 0.99999 | ≥ 0.99999 | ≥ 0.99999 | ≥ 0.99999 |
| greedy tokens | identical | identical | identical | identical |
| text | identical | identical | identical | identical |
| streamed in 370 ms pieces | identical to one-shot | identical | identical | identical |

Quantised transcripts: the Mandarin clip is identical at Q8_0 and Q4_K. On `jfk.wav`, Q8_0 gives "ask" for "asked", the
same near-tie the 160 ms export shows. Q4_K also drops the English punctuation.

## Licence

Apache-2.0, as upstream.
