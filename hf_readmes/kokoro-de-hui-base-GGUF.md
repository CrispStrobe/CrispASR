---
license: apache-2.0
language: de
base_model:
- dida-80b/kokoro-german-hui-multispeaker-base
- hexgrad/Kokoro-82M
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- kokoro
- styletts2
- german
- multispeaker
- hui-corpus
- gguf
- crispasr
library_name: ggml
---

# Kokoro German — HUI Multispeaker Base — GGUF

GGUF / ggml conversion of [`dida-80b/kokoro-german-hui-multispeaker-base`](https://huggingface.co/dida-80b/kokoro-german-hui-multispeaker-base) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

This is the **Stage-1 multispeaker base** of Kokoro-82M re-trained on the [HUI Audio Corpus German](https://github.com/iisys-hof/HUI-Audio-Corpus-German) (CC0, 51 speakers, ~51 h). Predictor + decoder + StyleEncoder were all trained on German, so prosody and timing on long German phrases sound native — unlike using the official English-trained Kokoro-82M with a foreign-language voice fallback, which collapses to silence on long German utterances.

> **This is a base model, not a voice.** It pairs with a German voice pack (e.g. `df_victoria` or `dm_martin` from the [kikiri-tts](https://huggingface.co/kikiri-tts) lineage). For deployable single-speaker production quality, you can run a Stage-2 fine-tune on one HUI speaker (~half-day on an A40); the base alone gives multispeaker-conditioned output.

Training repo: [semidark/kokoro-deutsch](https://github.com/semidark/kokoro-deutsch) · Discussion: [Issue #9](https://github.com/semidark/kokoro-deutsch/issues/9)

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `kokoro-de-hui-base-f16.gguf`  | F16  | 156 MB | Reference quality — ASR roundtrip byte-perfect on the long test phrase |
| `kokoro-de-hui-base-q8_0.gguf` | Q8_0 | 135 MB | **Recommended** — ASR roundtrip identical to F16 |

Q4_K is **not** published: it loses the trailing "-izers" suffix on "Phonemizers" and degrades into garbled output ("Guten A, dies ist ein S des Worten von links."). Q8_0 is the smallest safe quant for this backbone.

## Auto-routing in CrispASR

When `kokoro-de-hui-base-f16.gguf` (or its Q8_0 sibling) sits in the same directory as `kokoro-82m-f16.gguf`, CrispASR silently swaps to the German backbone whenever the user passes `-l de` (or any `de_*` / `de-*` locale). Voice cascade for German: `df_victoria` → `df_eva` → `ff_siwis`. Explicit `--voice` always wins.

```bash
./crispasr --backend kokoro \
    -m kokoro-82m-q8_0.gguf \
    -l de \
    --tts "Guten Tag, dies ist ein Test des deutschen Phonemizers." \
    --tts-output de.wav
# crispasr silently picks kokoro-de-hui-base-q8_0.gguf + kokoro-voice-df_victoria.gguf
```

The C ABI behind this routing is `crispasr_kokoro_resolve_model_for_lang_abi` and `crispasr_kokoro_resolve_fallback_voice_abi` (see `src/kokoro.h`). The Python wrapper exposes it as `crispasr.kokoro_resolve_for_lang(model, lang)`.

## Quality verification

ASR roundtrip via `parakeet-tdt-0.6b-v3 -l de` on "Guten Tag, dies ist ein Test des deutschen Phonemizers.":

| Backbone × voice | Quant | Parakeet output | verdict |
|---|---|---|---|
| dida-80b + dm_martin | F16  | "Guten Tag, dies ist ein Test des deutschen Phonemizers." | byte-perfect |
| dida-80b + dm_martin | Q8_0 | "Guten Tag, dies ist ein Test des deutschen Phonemizers." | byte-perfect |
| dida-80b + df_victoria | F16 | "Guten Tag, dies ist ein Tester des Deutschen Phonemizers." | 1 word-boundary err on "Test" |
| dida-80b + dm_bernd | F16 | "Guten Tag, dies ist ein Test des deutschen Phonemetzers." | 1 phoneme err on "izers" |

`crispasr-diff kokoro` against the PyTorch reference (after rewriting dida-80b's modern parametrize WeightNorm keys to legacy `weight_g/weight_v` so upstream `KModel` can load them):

| Stage (selection) | F16 cos_min | Q8_0 cos_min |
|---|---:|---:|
| token_ids → durations | 1.000 | 1.000 |
| `text_enc_out` | 1.000 | 0.9999 |
| `dec_decode_3_out` | 0.9999 | 0.999 |
| `audio_out` (RNG-divergent) | 0.91 | 0.03 |

F16 hits 14/16 PASS at cos≥0.999 (the 2 fails are RNG-divergent decoder stages). Q8_0 has weaker `audio_out` cosine but ASR-roundtrips identically — perceptual quality is preserved even where bit-level cosine drops.

## Voice packs

Use the sister repo [`cstr/kokoro-voices-GGUF`](https://huggingface.co/cstr/kokoro-voices-GGUF), which bundles:

| Voice | Source | Tier |
|---|---|---|
| `df_victoria` | [`kikiri-tts/kikiri-german-victoria`](https://huggingface.co/kikiri-tts/kikiri-german-victoria), Apache-2.0 | in-distribution to dida-80b lineage; recommended German default |
| `dm_martin`   | [`kikiri-tts/kikiri-german-martin`](https://huggingface.co/kikiri-tts/kikiri-german-martin), Apache-2.0 | byte-perfect ASR roundtrip on test phrase |
| `df_eva`, `dm_bernd` | recovered via [`r1di/kokoro-fastapi-german`](https://huggingface.co/r1di/kokoro-fastapi-german) Git LFS from the deleted Tundragoon repo, Apache-2.0 | second-tier German fallback |

The kikiri voicepacks are pre-extracted by the dida-80b maintainer using a German-trained StyleEncoder that shares lineage with this base — so they're in-distribution to the predictor and decoder bundled here.

## Architecture

Identical to [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M): same 178-symbol IPA vocab (per `semidark/kokoro-deutsch/training/kokoro_symbols.py`), same component shapes. Only the weight values differ — predictor + decoder + StyleEncoder were retrained on HUI; the BERT prosody encoder uses the same German `dbmdz/bert-base-german-cased` weights as the upstream kokoro-deutsch recipe.

## Conversion

```bash
# dida-80b ships only a stub config.json — pass the official Kokoro-82M
# config so the 178-symbol IPA vocab is preserved (vocab IDs are byte-
# identical between the two by training-recipe construction).
python models/convert-kokoro-to-gguf.py \
    --input dida-80b/kokoro-german-hui-multispeaker-base \
    --output kokoro-de-hui-base-f16.gguf \
    --config /path/to/hexgrad/Kokoro-82M/config.json \
    --outtype f16

build/bin/crispasr-quantize kokoro-de-hui-base-f16.gguf kokoro-de-hui-base-q8_0.gguf q8_0
```

## License

- Weights: Apache-2.0 (inherited from Kokoro-82M).
- Training corpus: HUI-Audio-Corpus-German is CC0; the gated `dida-80b/hui-german-51speakers` HF dataset uses the same source but is access-controlled.

## Attribution

- Re-train: [`dida-80b/kokoro-german-hui-multispeaker-base`](https://huggingface.co/dida-80b/kokoro-german-hui-multispeaker-base) (Apache-2.0).
- Recipe: [semidark/kokoro-deutsch](https://github.com/semidark/kokoro-deutsch) (Apache-2.0).
- Base: [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0).
- GGUF runtime: [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR).
