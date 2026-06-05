---
license: mit
language:
- multilingual
base_model:
- hubertsiuzdak/snac_24khz
pipeline_tag: text-to-speech
tags:
- codec
- audio-codec
- snac
- residual-vector-quantization
- gguf
- crispasr
library_name: ggml
---

# SNAC 24 kHz Codec — GGUF (ggml-quantised)

GGUF / ggml conversion of [`hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz) — the 24 kHz Multi-Scale Neural Audio Codec used by Orpheus-TTS — for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

SNAC is a small (~25 MB) Residual Vector Quantisation codec by Hubert Siuzdak. Pair it with the [`cstr/orpheus-3b-base-GGUF`](https://huggingface.co/cstr/orpheus-3b-base-GGUF) talker (or any other GGUF orpheus checkpoint) — the talker emits codec tokens but doesn't render audio without this codec.

## Files

| File | Type | Size | Notes |
|---|---|---:|---|
| `snac-24khz.gguf` | F32 | 25 MB | Full-precision codec — model is small enough that quantisation is not worth the quality risk |

The codec runs in C++ inside `libcrispasr` once `--codec-model snac-24khz.gguf` is passed; no Python required at inference time.

## Quick start

```bash
huggingface-cli download cstr/orpheus-3b-base-GGUF orpheus-3b-base-q8_0.gguf --local-dir .
huggingface-cli download cstr/snac-24khz-GGUF       snac-24khz.gguf            --local-dir .

./build/bin/crispasr --backend orpheus \
    -m orpheus-3b-base-q8_0.gguf \
    --codec-model snac-24khz.gguf \
    --voice tara --temperature 0.6 \
    --tts "Hello, my name is Tara." \
    --tts-output hello.wav
```

Or pass `-m auto` and CrispASR's auto-download fetches both files from HF.

## Architecture

| Component | Details |
|---|---|
| Sample rate | 24 kHz mono |
| Codebooks | 3 (RVQ — 1, 2, 4 entries per super-frame) |
| Codebook size | 4096 (12 bits each) |
| Frame rate | 12 ms (288 samples / hop 512) |
| Super-frame | 7 codec tokens cover 4 codec frames = 32 ms ≈ 768 PCM samples (deinterleave 1+2+4) |
| Decoder | Snake activations + transposed convolutions, full-precision F32 |

The codec maps 7 talker emissions to 1 super-frame; 4 super-frames give a 32 × 32 ms = 128 ms audio chunk (8192 samples at 24 kHz). The streaming protocol from `orpheus_snac.py` emits the middle 2048 samples of each 4-super-frame sliding window.

## Conversion

```bash
python models/convert-snac-to-gguf.py \
    --input hubertsiuzdak/snac_24khz \
    --output snac-24khz.gguf
```

No quantisation — the model is small enough (~25 MB) that the disk savings would be marginal and the quality risk on the codec output isn't worth it. The Q8_0 path collapses cosine similarity on `audio_out` below the ASR-roundtrip threshold in our diff harness.

## Used by

- [`cstr/orpheus-3b-base-GGUF`](https://huggingface.co/cstr/orpheus-3b-base-GGUF) — canonical English Orpheus 3B-FT.
- (queued) `cstr/orpheus-kartoffel-de-GGUF` — German Kartoffel_Orpheus finetunes.
- (queued) `cstr/orpheus-3b-de-q8_0-GGUF` — lex-au Orpheus-3B-DE Q8 mirror.

All three reuse this exact codec — only the talker GGUF changes.

## Attribution

- **Original codec:** [`hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz) (MIT). Hubert Siuzdak.
- **Reference paper:** [SNAC: Multi-Scale Neural Audio Codec](https://arxiv.org/abs/2410.14411).
- **Reference codec implementation:** [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac).
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR) — see `src/orpheus_snac.cpp` and `models/convert-snac-to-gguf.py`.

## License

MIT, inherited from the base codec.
