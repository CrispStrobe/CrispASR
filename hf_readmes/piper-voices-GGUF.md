---
license: cc-by-4.0
language:
- de
- en
base_model:
- rhasspy/piper-voices
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- piper
- vits
- gguf
- crispasr
library_name: ggml
---

# Piper voices — GGUF bundle

[rhasspy/piper](https://github.com/rhasspy/piper) VITS voices converted to
ggml's GGUF format (arch=`piper`) for the **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**
native runtime. Each voice is a single self-contained file — the
phoneme-id map and the espeak-ng voice are embedded in the GGUF, so no
companion is needed. Output is 22.05 kHz mono (CrispASR resamples to its
24 kHz playback convention).

Tiny and fast: ~15–60 MB per voice, single-digit-ms-per-sentence on CPU —
the best fit for mobile (CrisperWeaver) and quick previews. Converted with
[`models/convert-piper-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-piper-to-gguf.py)
(F16; reads the upstream `.onnx` + `.onnx.json`).

## Voices

Only **permissively-licensed** voices are hosted here — the underlying
training datasets allow redistribution (CC0 / public domain). Restrictive
voices from `rhasspy/piper-voices` (e.g. `en_US-lessac`, Blizzard 2013
research license; the CC BY-NC-SA voices) are **deliberately excluded**.

| File | Voice | Language | Quality | Dataset license |
|---|---|---|---|---|
| `piper-de_DE-thorsten-medium-f16.gguf` | thorsten | German (M) | medium | **CC0** |
| `piper-de_DE-thorsten-high-f16.gguf` | thorsten | German (M) | high | **CC0** |
| `piper-de_DE-thorsten_emotional-medium-f16.gguf` | thorsten (emotional) | German (M) | medium | **CC0** |
| `piper-de_DE-kerstin-low-f16.gguf` | kerstin | German (F) | low | **CC0** |
| `piper-de_DE-mls-medium-f16.gguf` | mls | German | medium | **CC-BY 4.0** † |
| `piper-en_GB-cori-medium-f16.gguf` | cori | English (GB, F) | medium | **public domain** |
| `piper-en_US-libritts_r-medium-f16.gguf` | libritts_r | English (US) | medium | **CC-BY 4.0** † |
| `piper-de_DE-eva_k-x_low-f16.gguf` | eva_k | German (F) | x-low | **BSD-style** ‡ |
| `piper-de_DE-karlsson-low-f16.gguf` | karlsson | German (M) | low | **BSD-style** ‡ |
| `piper-de_DE-ramona-low-f16.gguf` | ramona | German (F) | low | **BSD-style** ‡ |

The Thorsten + kerstin German voices are released **CC0** (public-domain
dedication — [Thorsten-Voice](https://www.thorsten-voice.de/) /
Thorsten Müller; kerstin from [rhasspy/dataset-voice-kerstin](https://github.com/rhasspy/dataset-voice-kerstin)).
`en_GB-cori` is **public domain** (trained on [LibriVox](https://librivox.org)
recordings, which are public domain).

† **CC-BY 4.0 — attribution required.** `mls` is from
[Multilingual LibriSpeech](https://www.openslr.org/94/) (MLS);
`en_US-libritts_r` is from [LibriTTS-R](https://www.openslr.org/141/)
(Koizumi et al., Google). Credit the dataset per CC-BY 4.0 when shipping
their audio.

‡ **M-AILABS — BSD-style license.** `eva_k` / `karlsson` / `ramona` are
from the [M-AILABS Speech Dataset](https://github.com/i-celeste-aurora/m-ailabs-dataset)
(German voices derived from LibriVox + Project Gutenberg public-domain
sources). Its BSD-style license permits commercial use and redistribution
**provided the copyright notice is retained**, and forbids using the
contributors' names to endorse derived products. Retain the M-AILABS
notice when redistributing.

### Excluded (license not redistributable)

Deliberately **not** converted/hosted here:
- `en_US-lessac` — Blizzard 2013 (CSTR Edinburgh), research/non-commercial.
- `en_US-ryan`, `en_US-hfc_female/male`, `de_DE-pavoque` — CC BY-**NC**-SA
  (non-commercial).

## Licensing

Two layers, both permissive here:

- **Runtime + converter** — the Piper architecture, the espeak-ng
  phonemizer integration, and `convert-piper-to-gguf.py` are MIT
  (rhasspy/piper is MIT-licensed; CrispASR's runtime is its own).
- **Voice weights** — each GGUF is a derivative of an upstream Piper voice
  and carries **that voice's dataset license** (the table above). All
  voices here are redistributable and commercial-use-OK, but obligations
  differ: CC0 / public-domain (thorsten, kerstin, cori) need nothing;
  CC-BY 4.0 (mls, libritts_r) need dataset attribution; M-AILABS
  (eva_k, karlsson, ramona) need the copyright notice retained. Honour
  the per-voice terms for any audio you ship. Crediting
  [Thorsten-Voice](https://www.thorsten-voice.de/)
  and the [Piper](https://github.com/rhasspy/piper) project is appreciated.

To add a voice with a different license (e.g. the CC BY 4.0
`en_US-libritts_r`), convert it yourself and honour that voice's terms —
do not assume the repo-level tag applies.

## Usage

```bash
crispasr --backend piper -m piper-de_DE-thorsten-medium-f16.gguf \
  --tts "Guten Tag, dies ist ein Test." --tts-output out.wav
```

In CrisperWeaver these appear in the Synthesize screen's model picker once
downloaded; the backend resamples 22.05 kHz → 24 kHz transparently.
