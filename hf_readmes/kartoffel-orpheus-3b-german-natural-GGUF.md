---
license: llama3.2
language:
- de
base_model:
- SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1
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
- gguf
- crispasr
library_name: ggml
---

# Kartoffel-Orpheus 3B (German, natural) — GGUF (ggml-quantised)

GGUF / ggml conversion of [`SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1`](https://huggingface.co/SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

A German fine-tune of [`canopylabs/orpheus-3b-0.1-ft`](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft), trained primarily on **natural human speech recordings** from German speakers. Drop-in checkpoint swap on the same Orpheus runtime — same Llama-3.2-3B-Instruct talker arch, same SNAC 24 kHz codec, same `<custom_token_N>` super-frame protocol, just different LM weights and a German speaker roster. The synthetic-data sibling lives at [`cstr/kartoffel-orpheus-3b-german-synthetic-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF).

| Speakers | Names |
|---|---|
| Male | `Jakob`, `Anton`, `Julian`, `Jan`, `Alexander`, `Emil`, `Ben`, `Elias`, `Felix`, `Jonas`, `Noah`, `Maximilian` |
| Female | `Sophie`, `Marie`, `Mia`, `Maria`, `Sophia`, `Lina`, `Lea` |

Pair this with the SNAC codec at [`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF) — the talker outputs codec tokens but doesn't render audio without it.

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `kartoffel-orpheus-de-natural-f16.gguf`  | F16  | ~6.2 GB | Reference quality |
| `kartoffel-orpheus-de-natural-q8_0.gguf` | Q8_0 | ~3.4 GB | **Recommended** |
| `kartoffel-orpheus-de-natural-q4_k.gguf` | Q4_K | ~1.8 GB | Smallest; smoke-tested but Q8_0 preferred |

Like the upstream Orpheus, sub-Q8 quants tend to break the SNAC super-frame slot pattern on rare prompts, so we ship Q4_K with a recommendation to use Q8_0 by default.

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target crispasr

# 2. Pull the talker + the SNAC codec
huggingface-cli download cstr/kartoffel-orpheus-3b-german-natural-GGUF \
    kartoffel-orpheus-de-natural-q8_0.gguf --local-dir .
huggingface-cli download cstr/snac-24khz-GGUF snac-24khz.gguf --local-dir .

# 3. Synthesise — German prompt with a German speaker
./build/bin/crispasr --backend kartoffel-orpheus \
    -m kartoffel-orpheus-de-natural-q8_0.gguf \
    --codec-model snac-24khz.gguf \
    --voice Julian \
    --temperature 0.6 \
    --tts "Hallo, ich heiße Julian und das ist ein Kartoffel-Orpheus Test." \
    --tts-output hallo.wav
```

For **auto-download** simply pass `-m auto`:

```bash
./build/bin/crispasr --backend kartoffel-orpheus-de-natural -m auto \
    --voice Sophie --temperature 0.6 \
    --tts "Auto-download holt beide Dateien." \
    --tts-output out.wav
```

## Quality verification

ASR roundtrip via [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF) on Q8_0, voice `Julian`:

| Synthesised text | parakeet-v3 -l de output |
|---|---|
| `"Hallo, ich heiße Julian und das ist ein Kartoffel-Orpheus Test."` | `"Hallo, ich heiße Julian und das ist ein Kartoffel-Orpheus-Test."` (verbatim, only minor hyphenation drift) |

Validation script:

```bash
crispasr --backend kartoffel-orpheus-de-natural \
    -m kartoffel-orpheus-de-natural-q8_0.gguf \
    --codec-model snac-24khz.gguf --voice Julian --temperature 0.6 \
    --tts "Hallo, ich heiße Julian und das ist ein Kartoffel-Orpheus Test." \
    --tts-output kartoffel_test.wav
crispasr --backend parakeet -m parakeet-tdt-0.6b-v3-q4_k.gguf -l de \
    -f kartoffel_test.wav --no-prints
# → Hallo, ich heiße Julian und das ist ein Kartoffel-Orpheus-Test.
```

## Architecture

Identical to Orpheus 3B — see [`cstr/orpheus-3b-base-GGUF`](https://huggingface.co/cstr/orpheus-3b-base-GGUF) for the full architecture writeup. The CrispASR `orpheus` runtime is checkpoint-agnostic; this GGUF is loaded by the same `orpheus_init_from_file` path with no source-code changes.

Prompt format (verbatim from the upstream): the LM sees `[audio_start=128259, BOS=128000, ...tokenize("{name}: {text}")..., eot_id=128009, audio_eot=128260, audio_eom=128261, audio_end=128257]`. The runtime handles this for you; just pass `--voice <Name>`.

## Stop policy

Stop on `audio_end=128257` **or** on >4 consecutive non-codec tokens. The `audio_pre_end=128009` and `audio_end_b=128261` tokens are **not** termination signals — they overlap with Llama-3 specials in the prompt and get filtered silently by the runtime.

## Conversion

```bash
python models/convert-orpheus-to-gguf.py \
    --input SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1 \
    --output kartoffel-orpheus-de-natural-f16.gguf \
    --speakers Jakob,Anton,Julian,Jan,Alexander,Emil,Ben,Elias,Felix,Jonas,Noah,Maximilian,Sophie,Marie,Mia,Maria,Sophia,Lina,Lea \
    --variant fixed_speaker

build/bin/crispasr-quantize kartoffel-orpheus-de-natural-f16.gguf \
    kartoffel-orpheus-de-natural-q8_0.gguf q8_0
build/bin/crispasr-quantize kartoffel-orpheus-de-natural-f16.gguf \
    kartoffel-orpheus-de-natural-q4_k.gguf q4_k
```

The `--variant fixed_speaker` flag bakes the German speaker roster into `orpheus.fixed_speakers` so the runtime's `--voice <Name>` lookup works without an external mapping.

## Attribution

- **Talker base:** [`SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1`](https://huggingface.co/SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1). German fine-tune by Sebastian Bodza on natural German speech.
- **Upstream Orpheus base:** [`canopylabs/orpheus-3b-0.1-ft`](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft).
- **Llama base:** [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) — Llama-3.2 community license.
- **SNAC codec:** [`hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz) (MIT) — see [`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF).
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR) — see `src/orpheus.cpp`, `src/orpheus_snac.cpp`, `models/convert-orpheus-to-gguf.py`.

## License

Llama-3.2 community license (inherited from the talker base). Includes the Acceptable Use Policy and the "Built with Llama" attribution requirement. Commercial use is permitted under the community license terms.

The SNAC codec is MIT and ships separately under [`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF).
