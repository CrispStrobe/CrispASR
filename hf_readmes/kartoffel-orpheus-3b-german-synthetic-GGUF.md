---
license: llama3.2
language:
- de
base_model:
- SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1
- canopylabs/orpheus-3b-0.1-ft
- meta-llama/Llama-3.2-3B-Instruct
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- orpheus
- kartoffel
- llama
- snac
- german
- deutsch
- emotion
- gguf
- crispasr
library_name: ggml
---

# Kartoffel-Orpheus 3B (German, synthetic + emotions) — GGUF (ggml-quantised)

GGUF / ggml conversion of [`SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1`](https://huggingface.co/SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

A German fine-tune of [`canopylabs/orpheus-3b-0.1-ft`](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft), trained on **synthetic German speech with explicit emotion and outburst control**. Drop-in checkpoint swap on the same Orpheus runtime — same Llama-3.2-3B-Instruct talker arch, same SNAC 24 kHz codec, same `<custom_token_N>` super-frame protocol. The natural-data sibling lives at [`cstr/kartoffel-orpheus-3b-german-natural-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF).

## Speakers (4)

| Name | Voice |
|---|---|
| `Martin` | Male |
| `Luca` | Male |
| `Anne` | Female |
| `Emma` | Female |

## Emotions + outbursts

The synthetic variant uses an extended prompt syntax:

```
{Speaker} - {Emotion}: {German text}
```

| Emotions | `Neutral`, `Happy`, `Sad`, `Excited`, `Surprised`, `Humorous`, `Angry`, `Calm`, `Disgust`, `Fear`, `Proud`, `Romantic` |
| Outbursts | `haha`, `ughh`, `wow`, `wuhuuu`, `ohhh` (in-text, surrounded by spaces) |

Example: `"Martin - Sad: Oh, ich bin so traurig."` → mournful Martin voice.
Example: `"Anne - Happy: wow das ist ja großartig."` → cheerful Anne with the `wow` outburst.

Pair this with the SNAC codec at [`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF).

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `kartoffel-orpheus-de-synthetic-f16.gguf`  | F16  | ~6.2 GB | Reference quality |
| `kartoffel-orpheus-de-synthetic-q8_0.gguf` | Q8_0 | ~3.4 GB | **Recommended** |
| `kartoffel-orpheus-de-synthetic-q4_k.gguf` | Q4_K | ~1.8 GB | Smallest |

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target crispasr-lib

# 2. Pull the talker + the SNAC codec
huggingface-cli download cstr/kartoffel-orpheus-3b-german-synthetic-GGUF \
    kartoffel-orpheus-de-synthetic-q8_0.gguf --local-dir .
huggingface-cli download cstr/snac-24khz-GGUF snac-24khz.gguf --local-dir .

# 3. Synthesise — German prompt with emotion control
./build/bin/crispasr --backend kartoffel-orpheus \
    -m kartoffel-orpheus-de-synthetic-q8_0.gguf \
    --codec-model snac-24khz.gguf \
    --voice Martin \
    --temperature 0.6 \
    --tts "Martin - Sad: Oh, ich bin so traurig." \
    --tts-output martin_sad.wav
```

For **auto-download** simply pass `-m auto`:

```bash
./build/bin/crispasr --backend kartoffel-orpheus-de-synthetic -m auto \
    --voice Anne --temperature 0.6 \
    --tts "Anne - Happy: Hallo, wie geht es dir heute?" \
    --tts-output anne_happy.wav
```

## Architecture

Identical to Orpheus 3B — see [`cstr/orpheus-3b-base-GGUF`](https://huggingface.co/cstr/orpheus-3b-base-GGUF) for the full architecture writeup. The CrispASR `orpheus` runtime is checkpoint-agnostic; this GGUF is loaded by the same `orpheus_init_from_file` path with no source-code changes. The `--voice` flag here is just the speaker name string in the prompt; emotion and outburst control happens via the prompt text itself.

## Conversion

```bash
python models/convert-orpheus-to-gguf.py \
    --input SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1 \
    --output kartoffel-orpheus-de-synthetic-f16.gguf \
    --speakers Martin,Luca,Anne,Emma \
    --variant fixed_speaker

build/bin/crispasr-quantize kartoffel-orpheus-de-synthetic-f16.gguf \
    kartoffel-orpheus-de-synthetic-q8_0.gguf q8_0
build/bin/crispasr-quantize kartoffel-orpheus-de-synthetic-f16.gguf \
    kartoffel-orpheus-de-synthetic-q4_k.gguf q4_k
```

## Attribution

- **Talker base:** [`SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1`](https://huggingface.co/SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1).
- **Upstream Orpheus base:** [`canopylabs/orpheus-3b-0.1-ft`](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft).
- **Llama base:** [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) — Llama-3.2 community license.
- **SNAC codec:** [`hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz) (MIT) — see [`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF).
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR) — see `src/orpheus.cpp`, `src/orpheus_snac.cpp`, `models/convert-orpheus-to-gguf.py`.

## License

Llama-3.2 community license (inherited from the talker base). Includes the Acceptable Use Policy and the "Built with Llama" attribution requirement.

The SNAC codec is MIT and ships separately under [`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF).
