---
license: llama3.2
language:
- en
base_model:
- canopylabs/orpheus-3b-0.1-ft
- unsloth/orpheus-3b-0.1-ft
- meta-llama/Llama-3.2-3B-Instruct
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- orpheus
- llama
- snac
- gguf
- crispasr
library_name: ggml
---

# Orpheus 3B — GGUF (ggml-quantised)

GGUF / ggml conversion of [`canopylabs/orpheus-3b-0.1-ft`](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) (sourced via the non-gated [`unsloth/orpheus-3b-0.1-ft`](https://huggingface.co/unsloth/orpheus-3b-0.1-ft) mirror) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Orpheus 3B is a Llama-3.2-3B-Instruct talker finetuned to emit `<custom_token_N>` codec tokens that the SNAC 24 kHz codec decodes back to speech. Distributed under the **Llama-3.2 community license** ("Built with Llama"). 8 fixed English speakers (`tara, leah, jess, leo, dan, mia, zac, zoe`).

Pair this with the SNAC codec at [`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF) — the talker outputs codec tokens but doesn't render audio without it.

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `orpheus-3b-base-f16.gguf`  | F16  | 6.2 GB | Reference quality |
| `orpheus-3b-base-q8_0.gguf` | Q8_0 | 3.4 GB | **Recommended** — ASR roundtrip word-exact vs F16 |

The talker LM is sensitive to peaked codec distributions, so we ship F16 + Q8_0 only. Sub-Q8 quants tend to break the SNAC super-frame slot pattern and produce gibberish even when the LM perplexity remains plausible.

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target crispasr-lib

# 2. Pull the talker + the SNAC codec
huggingface-cli download cstr/orpheus-3b-base-GGUF orpheus-3b-base-q8_0.gguf --local-dir .
huggingface-cli download cstr/snac-24khz-GGUF snac-24khz.gguf --local-dir .

# 3. Synthesise
./build/bin/crispasr --backend orpheus \
    -m orpheus-3b-base-q8_0.gguf \
    --codec-model snac-24khz.gguf \
    --voice tara \
    --temperature 0.6 \
    --tts "Hello, my name is Tara." \
    --tts-output hello.wav
```

24 kHz mono WAV. `--voice <name>` picks one of the 8 baked speakers; `--temperature 0.6` is the upstream `engine_class.py` default and is required — greedy decoding (`--temperature 0`) enters a 7-slot loop after a few super-frames and produces unusable audio.

For **auto-download** simply pass `-m auto`:

```bash
./build/bin/crispasr --backend orpheus -m auto \
    --voice leo --temperature 0.6 \
    --tts "Auto-download fetches both files." \
    --tts-output out.wav
```

## Quality verification

ASR roundtrip via [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF) on F16, voice `tara`:

| Synthesised text | Parakeet output |
|---|---|
| `"Hello, my name is Tara."` | `"Hello, my name is Tara."` (verbatim) |

Q8_0 produces ASR-identical output on the same prompt. Validation script:

```bash
crispasr --backend orpheus -m orpheus-3b-base-q8_0.gguf \
    --codec-model snac-24khz.gguf --voice tara --temperature 0.6 \
    --tts "Hello, my name is Tara." --tts-output orpheus_test.wav
crispasr --backend parakeet -m parakeet-tdt-0.6b-v3-q4_k.gguf \
    -f orpheus_test.wav --no-prints
# → Hello, my name is Tara.
```

## Architecture

| Component | Details |
|---|---|
| Talker LM | Llama-3.2-3B-Instruct (28 layers, 3072 hidden, 24 heads, 8 KV heads, head_dim=128, vocab 128256 + 7×4096 codec tokens) |
| RoPE | NEOX, theta=500000 |
| Codec | hubertsiuzdak/snac_24khz (RVQ, 3 codebooks × 4096) — separate GGUF |
| Sampling | temperature=0.6 + top-k by default; greedy is unstable |
| Audio | 24 kHz mono float32 PCM |

The talker emits a stream of `<custom_token_N>` LM tokens; every 7 emitted tokens form one "super-frame" that de-interleaves into 1 codes_0 / 2 codes_1 / 4 codes_2 entries (per `orpheus_tts_pypi/orpheus_tts/decoder.py`). 4 super-frames cover 16 SNAC frames (× 512-sample hop = 8192 PCM samples at 24 kHz).

## Prompt format (verbatim from canopyai/Orpheus-TTS)

```
[audio_start=128259, BOS=128000, ...tokenize("{name}: {text}")...,
 eot_id=128009, audio_eot=128260, audio_eom=128261, audio_end=128257]
```

The Llama-3 BOS at position 1 is **critical**. Without it the talker still emits well-structured super-frames but the audio is semantically garbage. The CrispASR runtime handles this for you — direct callers of `orpheus_synthesize_codes` need to mirror the layout.

## Stop policy

Stop on `audio_end=128257` **or** on >4 consecutive non-codec tokens. Don't stop on `audio_pre_end=128009` or `audio_end_b=128261` — those overlap with Llama-3 specials in the prompt and `text_N<10` reserved markers in the custom_token block; the upstream `tokens_decoder` filters them silently rather than terminating on them.

## Conversion

```bash
python models/convert-orpheus-to-gguf.py \
    --input unsloth/orpheus-3b-0.1-ft \
    --output orpheus-3b-ft-f16.gguf \
    --outtype f16

build/bin/crispasr-quantize orpheus-3b-ft-f16.gguf orpheus-3b-base-q8_0.gguf q8_0
```

The converter sets `GGUFWriter(use_temp_file=False)` because the `True` path buffers tensor data via `tempfile.SpooledTemporaryFile` and collapses throughput on near-full external disks (`/Volumes/backups` at 100% saw multi-MB/s spooling). The direct write holds the full tensor list in RAM during emit but completes in ~30 s on the 6.6 GB f16.

## Drop-in checkpoint variants

The `orpheus` runtime is checkpoint-agnostic — same arch, same prompt format, same SNAC codec. Future GGUF mirrors of:

- `SebastianBodza/Kartoffel_Orpheus_*` (German finetunes, 26 fixed speakers)
- `lex-au/Orpheus-3b-German-FT-Q8_0.gguf`

are checkpoint swaps. They reuse this same SNAC codec.

## Attribution

- **Talker base model:** [`canopylabs/orpheus-3b-0.1-ft`](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) (Llama-3.2 community license). canopylabs / canopyai.
- **Non-gated mirror used for conversion:** [`unsloth/orpheus-3b-0.1-ft`](https://huggingface.co/unsloth/orpheus-3b-0.1-ft).
- **Llama base:** [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) — Llama-3.2 community license.
- **SNAC codec:** [`hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz) (MIT) — see [`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF).
- **Reference TTS engine:** [canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) (`engine_class.py:_format_prompt`, `decoder.py`).
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR) — see `src/orpheus.cpp`, `src/orpheus_snac.cpp`, `models/convert-orpheus-to-gguf.py`.

## License

Llama-3.2 community license (inherited from the base talker). Includes the Acceptable Use Policy and the "Built with Llama" attribution requirement. Commercial use is permitted under the community license terms; review [`canopylabs/orpheus-3b-0.1-ft`](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) and [the Llama-3.2 license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) before redistribution.

The SNAC codec is MIT and ships separately under [`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF).
