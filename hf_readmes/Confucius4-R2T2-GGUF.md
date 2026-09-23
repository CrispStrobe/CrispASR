---
license: other
license_name: netease-youdao-model-use-license-agreement
license_link: https://huggingface.co/cstr/confucius4-r2t2-GGUF/blob/main/MODEL_LICENSE
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- streaming
- gguf
- ggml
- crispasr
- qwen3-asr
base_model: netease-youdao/Confucius4-R2T2
library_name: ggml
---

# Confucius4-R2T2 — GGUF

GGUF conversions of [`netease-youdao/Confucius4-R2T2`](https://huggingface.co/netease-youdao/Confucius4-R2T2)
for **[CrispASR](https://github.com/CrispStrobe/CrispASR)** (`--backend qwen3`).

R2T2 ("Real Real-Time Transcription") is NetEase Youdao's streaming fine-tune of Qwen3-ASR-1.7B. It emits
append-only text, committing only stable prefixes, with 80 ms to 2 s decoding chunks. The architecture is stock
Qwen3-ASR (the configs are identical to `Qwen/Qwen3-ASR-1.7B`), except that R2T2 ties `lm_head` to the token
embedding. CrispASR runs it offline like any Qwen3-ASR model, and also in its realtime WebSocket session with
R2T2's own streaming recipe.

> Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed
> by the original right-holder of the original model, and the original right-holder disclaims all liability
> related to this Derivative Work.

## Files

| File | Notes |
| --- | --- |
| `confucius4-r2t2-f16.gguf`  | F16 |
| `confucius4-r2t2-q8_0.gguf` | Q8_0 |
| `confucius4-r2t2-q4_k.gguf` | Q4_K (default for `-m confucius4-r2t2`) |
| `MODEL_LICENSE`             | the NetEase Youdao Model Use License Agreement, which applies to every file here |

Converted from revision `185ce639` (SHA-256 of `model.safetensors`: `cc4d5324…4610dc`). All files carry
`qwen3asr.streaming_recipe = r2t2`, which turns on the streaming session by default.

## Streaming

The realtime endpoint (`crispasr --server --ws-port N`, then `ws://…:N+1/v1/realtime`) runs R2T2's prefix-rollback
algorithm. It is a port of `R2T2ASRModel.streaming_transcribe` driven by the repository's `example.py` schedule:
160 ms steps, a 160 ms first-chunk lookahead, rollback of 1 token, and adaptive `max_new_tokens`. Deltas are
append-only, following R2T2's own `ws_server.py`. Tunable with `CRISPASR_QWEN3_STREAM_STEP_MS`,
`CRISPASR_QWEN3_STREAM_LOOKAHEAD_MS` and `CRISPASR_QWEN3_STREAM_UNFIXED_TOKENS`.

## Verification

Against the upstream `qwen_asr` package (float32, CPU) on `jfk.wav` (English) and a Mandarin clip:

| check | F16 | Q8_0 | Q4_K |
| --- | --- | --- | --- |
| encoder output (reference mel), worst-frame cos | 1.000000 | 0.9993 | 0.9993 |
| first-step logits cos | 0.999998 | 0.9964 | 0.83 |
| offline transcript vs upstream | identical (en, zh) | identical (en, zh) | same words, punctuation differs |
| streaming final text vs upstream `example.py` | identical (en, zh) | — | identical (zh), punctuation (en) |

The streaming reference ran R2T2's own `example.run_streaming` unmodified, with its vLLM call served by the same
weights through transformers. Call by call, CrispASR reproduces the same audio lengths and token budgets, and
the same generated text on 66 of 68 (en) and 73 of 81 (zh) model calls. The rest are near-tied greedy steps that
re-converge to the identical final transcript.

## Licence

These files are a Derivative Work under the **NetEase Youdao Model Use License Agreement** (see `MODEL_LICENSE`),
not an open-source licence. Obligations include the following:
- Retain the agreement with every copy.
- Pass these terms on to anyone you redistribute to.
- Make the disclaimer above when you distribute a derivative.
- Obtain a separate licence from NetEase if your products exceed 100 M monthly active users or RMB 1 B annual revenue.
- Keep the model out of the high-risk uses listed in §4.2.

The inference code in CrispASR is MIT and unaffected.
