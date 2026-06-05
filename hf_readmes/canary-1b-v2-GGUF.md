---
license: cc-by-4.0
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
- translation
- speech-translation
- ggml
- gguf
- canary
- fastconformer
- multilingual
library_name: ggml
base_model: nvidia/canary-1b-v2
---

# Canary 1B v2 — GGUF (ggml-quantised)

GGUF / ggml conversions of [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) for use with the `canary-main` CLI from **[CrispStrobe/CrispASR@parakeet](https://github.com/CrispStrobe/CrispASR/tree/parakeet)**.

Canary 1B v2 is NVIDIA's 978 M-parameter multilingual ASR + speech translation model:

- **25 European languages** with **explicit `source_lang` / `target_lang` task tokens** (no auto-detect ambiguity)
- **Speech translation** in both directions: X→English (24 languages) and English→X (24 languages)
- **7.15% avg WER** on the HuggingFace Open ASR Leaderboard (English) — competitive with Whisper-large-v3 at 1/1.6× the size
- **CC-BY-4.0** licence

This is the encoder–decoder companion to **[`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF)**, which is the same FastConformer encoder family but with a TDT decoder for ASR-only. Both share the runtime and were ported in the same fork.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `canary-1b-v2.gguf`        | 1.97 GB | F16, full precision |
| `canary-1b-v2-q8_0.gguf`   | 1.1 GB  | Q8_0, near-lossless |
| `canary-1b-v2-q5_0.gguf`   | 777 MB  | Q5_0 |
| `canary-1b-v2-q4_k.gguf`   | 673 MB  | **Q4_K — recommended default** |

## Quick Start

```bash
# 1. Build the runtime
git clone -b parakeet https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target canary-main

# 2. Download a quantisation
huggingface-cli download cstr/canary-1b-v2-GGUF \
    canary-1b-v2-q4_k.gguf --local-dir .

# 3. ASR (English → English)
./build/bin/canary-main \
    -m canary-1b-v2-q4_k.gguf \
    -f your-audio.wav \
    -sl en -tl en -t 8

# 4. ASR (German → German)
./build/bin/canary-main -m canary-1b-v2-q4_k.gguf \
    -f german_audio.wav -sl de -tl de

# 5. Speech translation (German → English)
./build/bin/canary-main -m canary-1b-v2-q4_k.gguf \
    -f german_audio.wav -sl de -tl en
```

## Verified end-to-end output

**English ASR (`samples/jfk.wav`, 11 s):**
> And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

**German ASR (Wikimedia Commons `Amardeo_Sarma_voice_-_de.ogg`, 91 s):**
> Ich heiße Amadeus Scharma. Ich bin 1955 in Kassel in Deutschland geboren, weitgehend in Indien aufgewachsen. Ich hatte meine Ausbildung in Ingenieurwissenschaften in Neu-Delhi und dann später auch in Darmstadt, an der Technischen Universität von Darmstadt. ...

**Speech translation DE → EN (same clip):**
> My name is Amadeo Sharma. I was born in Kassel in Germany in 1955, and I grew up largely in India. I had my education in engineering in New Delhi and then later also in Darmstadt, at the Technical University of Darmstadt. ...

## Why explicit language tokens?

Auto-detect language ID can misfire on accented or noisy speech. We tested parakeet (which has no `-l` flag and relies on auto-detect) on the same German clips and it picked **Russian** for Angela Merkel and **code-switched into English** on Sarma's recording. Canary's `-sl LANG` removes that whole class of failures by telling the decoder explicitly what language to expect — see [`test_german.md`](https://github.com/CrispStrobe/CrispASR/blob/parakeet/test_german.md) in the runtime repo.

## Supported languages

`bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk` (25 European languages).

For each pair (`source_lang`, `target_lang`):
- `sl == tl` → ASR
- `sl != tl` → speech translation

Translation supports any pair from 24 non-English languages → English, and English → any of 24 non-English languages.

## Architecture

| Component | Details |
| --- | --- |
| Encoder       | 32-layer FastConformer, d=1024, 8 heads, head_dim=128, FFN=4096, conv kernel=9, biases on every linear/conv |
| Subsampling   | Conv2d dw_striding stack, 8× temporal (100 → 12.5 fps) |
| Decoder       | 8-layer pre-LN Transformer (self-attn + cross-attn + FFN), d=1024, 8 heads, head_dim=128, FFN=4096, max_ctx=1024 |
| Embedding     | Token (16384 × 1024) + learned positional (1024 × 1024) + LN |
| Output head   | Linear (1024 → 16384) |
| Vocab         | 16384 SentencePiece (NeMo CanaryBPETokenizer) |
| Audio         | 16 kHz mono, 128 mel bins, n_fft=512, hop=160, win=400 |
| Parameters    | ~978 M (encoder 811M + decoder 152M + head 17M) |

The mel filterbank and Hann window are baked into the GGUF (`preprocessor.fb` and `preprocessor.window`), so no recomputation at runtime. BatchNorm in the convolution module is folded into the depthwise conv weights at load time. Cross-attention K/V is pre-computed once per audio slice from the encoder output and then reused across decoder steps.

## How this was made

1. **Inspect** the `.nemo` tarball: 1510 tensors total — encoder (1294), `transf_decoder` (214), `log_softmax` head (2), preprocessor (2). Skipped the auxiliary `timestamps_asr_model_weights.ckpt` which is the separate Parakeet CTC model used by NeMo Forced Aligner for segment-level timestamps.
2. **Convert** with [`models/convert-canary-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/parakeet/models/convert-canary-to-gguf.py): remap NeMo state-dict keys (`transf_decoder._embedding.token_embedding` → `decoder.embed`, `first_sub_layer.query_net` → `sa_q`, etc.) and write 1478 tensors as F16 (matmul) + F32 (norms / biases / mel filterbank). 1.97 GB GGUF.
3. **C++ runtime** in [`src/canary.{h,cpp}`](https://github.com/CrispStrobe/CrispASR/blob/parakeet/src/canary.cpp): mmap the GGUF, fold BN into the depthwise conv at load time, build the encoder graph (32-layer FastConformer with biases), build the decoder graph per step (with self-attention KV cache + pre-computed cross-K/V), greedy decode with task-token prompt, detokenise via SentencePiece.
4. **Quantise** with [`crispasr-quantize`](https://github.com/CrispStrobe/CrispASR/blob/parakeet/examples/cohere-main/crispasr-quantize.cpp): same llama.cpp-style quantiser used for the other GGUFs in this family.

## Comparison with Parakeet TDT 0.6B v3

| | parakeet-tdt-0.6b-v3 | **canary-1b-v2 (this repo)** |
| --- | --- | --- |
| Architecture | FastConformer + TDT (transducer) | FastConformer + Transformer (encoder–decoder) |
| Parameters | 600M | 978M |
| Languages | 25 (auto-detect) | 25 (**explicit `-sl` / `-tl`**) |
| Speech translation | ❌ | ✅ X→En and En→X |
| Word timestamps | ✅ from TDT duration head | ✗ (segment-level via aux CTC) |
| Q4_K size | 467 MB | ~600 MB |
| Open ASR WER (avg) | 6.34% | 7.15% |
| Use case | fastest multilingual ASR with word stamps | best multilingual ASR + translation, language is known |

## Attribution

- **Original model:** [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) (CC-BY-4.0). NVIDIA NeMo team. See the [Canary-1B-v2 & Parakeet-TDT-0.6B-v3 technical report](https://arxiv.org/abs/2509.14128).
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR@parakeet`](https://github.com/CrispStrobe/CrispASR/tree/parakeet). The decoder structure was cross-checked against NeMo's `transformer_decoders.py` and `transformer_modules.py` source.
- **Encoder graph patterns:** shared between cohere/parakeet/canary in the same fork, originally adapted for the [CrispASR ggml branch](https://github.com/CrispStrobe/CrispASR/tree/ggml).

## Related

- C++ runtime: **[CrispStrobe/CrispASR@parakeet](https://github.com/CrispStrobe/CrispASR/tree/parakeet)**
- Sister model (ASR-only, smaller, with word timestamps): [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF)
- Sister runtime (Cohere Transcribe, lowest English WER): [`cstr/cohere-transcribe-03-2026-GGUF`](https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF)
- ONNX INT4 (Cohere): [`cstr/cohere-transcribe-onnx-int4`](https://huggingface.co/cstr/cohere-transcribe-onnx-int4)
- ONNX INT8 (Cohere): [`cstr/cohere-transcribe-onnx-int8`](https://huggingface.co/cstr/cohere-transcribe-onnx-int8)

## License

CC-BY-4.0, inherited from the base model. Use of these GGUF files must comply with the CC-BY-4.0 license including attribution.
