---
license: apache-2.0
language:
- en
- es
- fr
- de
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- kokoro
- styletts2
- voice-pack
- gguf
- crispasr
library_name: ggml
---

# Kokoro voices — GGUF bundle

Per-speaker style packs for the Kokoro-82M family, converted to ggml's GGUF voice-pack format (arch=`kokoro-voice`, single F32 tensor `voice.pack[max_phon, 1, 256]`). For use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)** alongside [`cstr/kokoro-82m-GGUF`](https://huggingface.co/cstr/kokoro-82m-GGUF) or the German backbone [`cstr/kokoro-de-hui-base-GGUF`](https://huggingface.co/cstr/kokoro-de-hui-base-GGUF).

Each voicepack is ~510 KB. Loading is direct passthrough — no quantisation needed at this size.

## Voices

| File | Speaker | Language | Source | License |
|---|---|---|---|---|
| `kokoro-voice-af_heart.gguf`     | af_heart      | English (US, F)  | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) `voices/af_heart.pt` | Apache-2.0 |
| `kokoro-voice-ef_dora.gguf`      | ef_dora       | Spanish (F)      | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) `voices/ef_dora.pt`  | Apache-2.0 |
| `kokoro-voice-ff_siwis.gguf`     | ff_siwis      | French (F)       | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) `voices/ff_siwis.pt` | Apache-2.0 |
| `kokoro-voice-df_eva.gguf`       | df_eva        | German (F)       | recovered from [`r1di/kokoro-fastapi-german`](https://huggingface.co/r1di/kokoro-fastapi-german) Git LFS (originally Tundragoon) | Apache-2.0 |
| `kokoro-voice-dm_bernd.gguf`     | dm_bernd      | German (M)       | recovered from [`r1di/kokoro-fastapi-german`](https://huggingface.co/r1di/kokoro-fastapi-german) Git LFS (originally Tundragoon) | Apache-2.0 |
| `kokoro-voice-df_victoria.gguf`  | df_victoria   | German (F)       | [`kikiri-tts/kikiri-german-victoria`](https://huggingface.co/kikiri-tts/kikiri-german-victoria) `voices/victoria.pt` | Apache-2.0 |
| `kokoro-voice-dm_martin.gguf`    | dm_martin     | German (M)       | [`kikiri-tts/kikiri-german-martin`](https://huggingface.co/kikiri-tts/kikiri-german-martin) `voices/martin.pt`     | Apache-2.0 |

The Tundragoon voicepacks are `[512, 1, 256]` F32 (max_phon=512); the official Kokoro and kikiri voicepacks are `[510, 1, 256]` F32 (max_phon=510). The voice loader reads `max_phon` from the file so both layouts work transparently.

## German voice cascade

When CrispASR is invoked with `-l de` (or any `de_*` / `de-*` locale) and no explicit `--voice`, it picks German voicepacks in this order:

1. `df_victoria` — kikiri-tts, in-distribution to the dida-80b German backbone (recommended)
2. `df_eva`      — Tundragoon recovery, second-tier German speaker
3. `ff_siwis`    — French baseline, last-resort non-silence fallback

Languages without a native pack (ru, ko, ar, …) fall back to `ff_siwis`. See [`cstr/kokoro-de-hui-base-GGUF`](https://huggingface.co/cstr/kokoro-de-hui-base-GGUF) for the matching German backbone.

## Quality (ASR roundtrip)

Long German phrase ("Guten Tag, dies ist ein Test des deutschen Phonemizers."), `parakeet-tdt-0.6b-v3 -l de`, dida-80b backbone F16:

| Voice | Parakeet output |
|---|---|
| `dm_martin`   | "...Phonemizers." (perfect) |
| `df_victoria` | "...Tester des Deutschen Phonemizers." (1 word-boundary err) |
| `dm_bernd`    | "...Phonemetzers." (1 phoneme err) |
| `df_eva`      | "...Phonemetzes." (1 phoneme err) |

All four clear the energy gate (peak ≥ 8000, RMS ≥ 1000); two are word-perfect on a phrase the official English-trained Kokoro-82M with `af_heart` collapses to silence on.

## Quick start

```bash
huggingface-cli download cstr/kokoro-voices-GGUF kokoro-voice-af_heart.gguf --local-dir .
huggingface-cli download cstr/kokoro-82m-GGUF    kokoro-82m-q8_0.gguf       --local-dir .

./crispasr --backend kokoro \
    -m kokoro-82m-q8_0.gguf \
    --voice kokoro-voice-af_heart.gguf \
    --tts "Hello world" --tts-output hello.wav
```

## Conversion

```bash
python models/convert-kokoro-voice-to-gguf.py \
    --input voices/af_heart.pt \
    --output kokoro-voice-af_heart.gguf
```

## Attribution

- Official voices (`af_heart`, `ef_dora`, `ff_siwis`): [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0.
- Tundragoon recovery (`df_eva`, `dm_bernd`): the original `Tundragoon/Kokoro-German` HF repo was deleted; voices were recovered from [`r1di/kokoro-fastapi-german`](https://huggingface.co/r1di/kokoro-fastapi-german)'s Git LFS (`api/src/voices/v1_0/{df_eva,dm_bernd}.pt`), retaining the original Apache-2.0 license.
- kikiri-tts (`df_victoria`, `dm_martin`): [`kikiri-tts/kikiri-german-victoria`](https://huggingface.co/kikiri-tts/kikiri-german-victoria) and [`kikiri-tts/kikiri-german-martin`](https://huggingface.co/kikiri-tts/kikiri-german-martin) by the dida-80b maintainer, Apache-2.0.
- GGUF format + runtime: [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR).

## License

Apache-2.0 across the bundle, matching every upstream source.
