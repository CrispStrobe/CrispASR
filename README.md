# parakeet-whisper.cpp

A fork of [whisper.cpp](https://github.com/ggml-org/whisper.cpp) that adds full C++ ggml runtimes for two NVIDIA NeMo speech models, both released August 2025:

- **[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** — 600M-parameter FastConformer + Token-and-Duration Transducer. Multilingual ASR (25 European languages, auto-detect), **built-in word timestamps**, runs at ~1× realtime CPU.
- **[nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2)** — 978M-parameter FastConformer + Transformer encoder-decoder. Multilingual ASR + **speech translation** (X→En, En→X) across 25 languages, with **explicit `source_lang` / `target_lang` task tokens** (no auto-detect ambiguity).

Both work end-to-end on CPU via ggml. Pre-converted GGUF weights:
- **[cstr/parakeet-tdt-0.6b-v3-GGUF](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF)** — F16, Q8_0, Q5_0, Q4_K
- **[cstr/canary-1b-v2-GGUF](https://huggingface.co/cstr/canary-1b-v2-GGUF)** — coming soon

> **Looking for the Cohere Transcribe runtime?** It lives on the **[`ggml`](https://github.com/CrispStrobe/cohere-whisper.cpp/tree/ggml)** branch of the same repo. This branch (`parakeet`) is a strict superset that adds parakeet + canary on top of the cohere infrastructure.

## Which runtime should I use?

| Need | Right tool |
| --- | --- |
| Lowest English WER, model size doesn't matter | `cohere-main` (ggml branch) |
| Word-level timestamps + multilingual + small + fast | **`parakeet-main`** |
| Multilingual + **explicit language control** (no auto-detect) | **`canary-main`** |
| **Speech translation** (X→En or En→X) | **`canary-main`** |
| 30 ms-accurate word stamps via CTC forced alignment | `cohere-align` (ggml branch) |

| | parakeet-tdt-0.6b-v3 | canary-1b-v2 |
| --- | --- | --- |
| Parameters | 600M | 978M |
| Architecture | FastConformer encoder + TDT decoder | FastConformer encoder + Transformer decoder |
| Languages | 25 EU (auto-detect) | 25 EU (explicit `-sl` flag) |
| Speech translation | ❌ | ✅ X→En and En→X |
| Word timestamps | ✅ from TDT duration head | ✗ (segment-level via auxiliary CTC) |
| Q4_K size | 467 MB | ~600 MB |
| Open ASR WER (avg) | 6.34% | 7.15% |
| License | CC-BY-4.0 | CC-BY-4.0 |

Both share the same FastConformer encoder code and the same NeMo-style mel preprocessor (128 mels, 16 kHz, n_fft=512, win=400, hop=160).

---

## Quick start — parakeet (fastest, multilingual ASR)

```bash
git clone -b parakeet https://github.com/CrispStrobe/cohere-whisper.cpp
cd cohere-whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target parakeet-main

huggingface-cli download cstr/parakeet-tdt-0.6b-v3-GGUF \
    parakeet-tdt-0.6b-v3-q4_k.gguf --local-dir .

./build/bin/parakeet-main -m parakeet-tdt-0.6b-v3-q4_k.gguf -f samples/jfk.wav -t 8
# And so, my fellow Americans, ask not what your country can do for you,
# ask what you can do for your country.
```

Word timestamps via `-v`:
```
[ 0.32s →  0.64s]  And
[ 0.64s →  0.88s]  so,
[ 1.04s →  1.28s]  my
[ 1.28s →  1.76s]  fellow
[ 1.76s →  3.28s]  Americans.
```

Each boundary is one encoder frame = **80 ms**. Long audio + VAD + SRT/VTT/TXT all supported via `-vad-model`, `-osrt`, `-ovtt`, `-ot`, `-ml N`. See the [parakeet quick start above](#quick-start--parakeet-fastest-multilingual-asr) for the full CLI reference.

The auto-language detect on parakeet works well for clean speech but can misfire on accented or noisy audio (we found it picked Russian on Angela Merkel's German speech, see [`test_german.md`](test_german.md)). For German production use, prefer canary with `-sl de`.

---

## Quick start — canary (explicit language + translation)

```bash
cmake --build build -j$(nproc) --target canary-main

# Convert your own from the .nemo (no pre-quantised yet at time of writing):
pip install gguf torch sentencepiece huggingface_hub
python -c "from huggingface_hub import snapshot_download; \
  print(snapshot_download('nvidia/canary-1b-v2'))"
python models/convert-canary-to-gguf.py \
    --nemo  <snapshot-path>/canary-1b-v2.nemo \
    --output canary-1b-v2.gguf

./build/bin/canary-main -m canary-1b-v2.gguf -f samples/jfk.wav -sl en -tl en -t 8
# And so, my fellow Americans, ask not what your country can do for you,
# ask what you can do for your country.
```

### German ASR

```bash
./build/bin/canary-main -m canary-1b-v2.gguf -f german_audio.wav -sl de -tl de
# Ich heiße Amadeus Scharma. Ich bin 1955 in Kassel in Deutschland geboren,
# weitgehend in Indien aufgewachsen. ...
```

### Speech translation (German → English)

```bash
./build/bin/canary-main -m canary-1b-v2.gguf -f german_audio.wav -sl de -tl en
# My name is Amadeo Sharma. I was born in Kassel in Germany in 1955,
# and I grew up largely in India. ...
```

Same `-sl X -tl Y` works for any pair of the 25 supported languages: `bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk`. When `sl == tl` it's ASR; when they differ, it's speech translation.

### CLI reference

```
usage: canary-main [options] -m MODEL -f AUDIO

  -m  FNAME       canary GGUF model
  -f  FNAME       input audio (16 kHz mono WAV)
  -sl LANG        source language ISO-639-1 (en, de, fr, ...)
  -tl LANG        target language. Same as -sl → ASR; differs → translation
  -t  N           threads (default: 4)
  -v              dump per-token decoder steps for debugging
```

The full VAD / chunking / SRT / VTT / TXT plumbing matches `parakeet-main` and is being added incrementally (see `canary-todo.md` for status).

### Architecture

| Component | Details |
| --- | --- |
| Encoder       | 32-layer FastConformer, d=1024, 8 heads, head_dim=128, FFN=4096, conv kernel=9, **biases on every linear/conv** (Canary uses `use_bias: true`) |
| Subsampling   | Conv2d dw_striding stack, 8× temporal (100 → 12.5 fps) |
| Decoder       | 8-layer pre-LN Transformer (self-attn + cross-attn + FFN), d=1024, 8 heads, head_dim=128, FFN=4096, max_ctx=1024 |
| Embedding     | Token (16384 × 1024) + learned positional (1024 × 1024) + LN |
| Output head   | Separate linear (1024 → 16384) |
| Vocab         | 16384 SentencePiece (CanaryBPETokenizer, identical to Cohere Transcribe) |
| Parameters    | ~978M (encoder 811M + decoder 152M + head 17M) |
| Tensors       | 1478 in the GGUF (encoder 1294 + decoder 179 + head 2 + preprocessor 2) |

The Conformer encoder is identical in structure to parakeet's (we share the encoder code). The decoder block is pre-LN with three sub-layers: `LN → SA → +residual → LN → CA → +residual → LN → FFN → +residual`, FFN activation is ReLU (per NeMo's `PositionWiseFF` default). Self-attention KV cache lives on a backend buffer for fast autoregressive generation. Cross-attention K/V is pre-computed once per audio slice from the encoder output.

### Decoder prompt format

Canary uses task tokens in the decoder prompt prefix to drive ASR vs translation:

```
<|startofcontext|> <|startoftranscript|> <|emo:undefined|>
<|src_lang|> <|target_lang|> <|pnc|>|<|nopnc|>
<|notimestamp|> <|nodiarize|>
... model output starts here ...
```

The src/tgt language tokens explicitly tell the decoder what language to expect and what language to emit. There is no auto-detect — this is the whole point of using canary. If the source language is unknown, you'd need a separate language-ID model.

---

## Current status (parakeet branch)

| Component | parakeet | canary |
| --- | --- | --- |
| GGUF converter | ✅ | ✅ |
| Loader | ✅ | ✅ |
| Encoder forward | ✅ | ✅ |
| Decoder forward | ✅ | ✅ |
| Mel STFT | ✅ | ✅ (shared) |
| Greedy decode | ✅ TDT | ✅ Transformer |
| Word timestamps | ✅ from TDT durations | ⏳ scaffold (linear) |
| CLI: -sl/-tl | n/a | ✅ |
| CLI: VAD + chunking | ✅ | ⏳ |
| CLI: SRT/VTT/TXT | ✅ | ⏳ |
| Quantisation | ✅ Q4_K/Q5_0/Q8_0 | ⏳ |
| HF release | ✅ | ⏳ |

`canary-todo.md` tracks the remaining items in detail. The encoder, decoder, prompt builder, and end-to-end ASR + translation are all proven working — what's left is mostly CLI polish and quantisation.

## Key bug fixes in this branch

- **Transposed positional encoding** in `parakeet_make_pos_enc` / `canary_make_pos_enc` — the function wrote `pe[(2*i)*K + j]` (positions fast, dims slow) but the ggml tensor was `(d, 2T-1)` with dims fast. The correct layout is `pe[dim + pos*d]`. Parakeet's TDT decoder was robust enough to mostly recover; canary's encoder-decoder cross-attention exposed the bug immediately. Both runtimes now use the corrected layout. JFK output went from `"And so my fellow Americans. Ask not..."` (periods, parakeet pre-fix) to `"And so, my fellow Americans, ask not..."` (commas, canonical).

## Repository layout

| Path | Description |
| --- | --- |
| `src/parakeet.{h,cpp}`                    | Public C API + ggml runtime for parakeet TDT |
| `src/canary.{h,cpp}`                      | Public C API + ggml runtime for canary 1B v2 |
| `src/cohere.{h,cpp}`                      | Cohere Transcribe runtime (from the ggml branch) |
| `src/wav2vec2-ggml.{h,cpp}` + `src/align.{h,cpp}` | wav2vec2 CTC forced alignment for `cohere-align` |
| `models/convert-parakeet-to-gguf.py`      | `.nemo → GGUF` for parakeet |
| `models/convert-canary-to-gguf.py`        | `.nemo → GGUF` for canary |
| `examples/parakeet-main/main.cpp`         | parakeet CLI (full: VAD, chunking, SRT/VTT/TXT) |
| `examples/canary-main/main.cpp`           | canary CLI (basic: ASR + translation, polish ongoing) |
| `examples/cohere-main/`, `examples/cohere-align/` | cohere CLIs |
| `parakeet-todo.md`                        | parakeet implementation plan (mostly complete) |
| `canary-todo.md`                          | canary implementation plan (encoder + decoder + prompt done) |
| `benchmark_cohere.md`                     | Cross-runtime benchmark numbers |
| `test_german.md`                          | German audio comparison: parakeet's auto-detect failure modes |
| `ggml_plans.md`                           | VNNI Q8_0 plan to close the ONNX inference gap |

## Attribution

- **Parakeet TDT 0.6B v3** and **Canary 1B v2**: NVIDIA NeMo team (CC-BY-4.0). Use must comply with the CC-BY-4.0 license including attribution.
- **Encoder graph patterns**: shared between cohere/parakeet/canary, originally adapted from the cohere-whisper.cpp ggml branch.
- **Decoder pattern (canary)**: cross-checked against NeMo's `transformer_decoders.py` and `transformer_modules.py` source.
- **Underlying runtime**: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) / [ggml](https://github.com/ggerganov/ggml).

## License

The fork code is MIT (matching whisper.cpp). The parakeet and canary models themselves are **CC-BY-4.0**, inherited from NVIDIA. Use of the GGUF files must comply with CC-BY-4.0 including attribution.
