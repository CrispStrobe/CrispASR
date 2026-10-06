---
license: cc-by-4.0
language:
- tr
- en
base_model:
- Helsinki-NLP/opus-mt-tr-en
pipeline_tag: translation
tags:
- translation
- marian
- opus-mt
- encoder-decoder
- gguf
- crispasr
library_name: ggml
---

# Opus-MT Turkish → English — GGUF (ggml)

GGUF / ggml conversion of [`Helsinki-NLP/opus-mt-tr-en`](https://huggingface.co/Helsinki-NLP/opus-mt-tr-en) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)** (`--backend marian`).

A small (~80M parameters, 6+6 layers, d=512) MarianMT model for one language pair. In CrispASR these are the fastest translators for live transcription + translation (`--live-translate`).

## Licence and attribution

**CC-BY-4.0.** The weights are the `opus-2020-01-16` release of the [OPUS-MT project](https://github.com/Helsinki-NLP/Opus-MT) (Jörg Tiedemann and Santhosh Thottingal, University of Helsinki; *OPUS-MT — Building open translation services for the World*, EAMT 2020), trained on [OPUS](https://opus.nlpl.eu/) data. The project distributes its pre-trained models under CC-BY 4.0 ([Opus-MT README](https://github.com/Helsinki-NLP/Opus-MT), [OPUS-MT-train README](https://github.com/Helsinki-NLP/OPUS-MT-train)); that statement, not the tag on an individual model card, is what this repository follows. This is a format conversion: the weights are unchanged at f16 and quantized at q8_0. Redistribution requires attribution to the OPUS-MT project.

## Files

| File | Size | Output vs. the reference implementation |
|---|---:|---|
| `opus-mt-tr-en-f16.gguf` | 158 MB | 8/8 sentences identical greedy, 8/8 with beam 4 |
| `opus-mt-tr-en-q8_0.gguf` | 87 MB | 8/8 sentences identical greedy — any others differ in wording. Fastest; recommended. |

"Reference" is `MarianMTModel.generate` from Hugging Face `transformers` on the original checkpoint, on 8 test sentences; input token ids were identical for 8/8.

## Quick start

```bash
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Text → text. -bs 1 is greedy; without it the checkpoint's own beam size is used.
./build/bin/crispasr --backend marian -m opus-mt-tr-en -sl tr -tl en -bs 1 \
    --text "Günaydın, bugünkü toplantıya hoş geldiniz."
# Good morning, and welcome to today's meeting.

# Live: microphone in, transcript + translation out, sentence by sentence
./build/bin/crispasr --live-translate -l tr --tr-tl en \
    -m auto --backend parakeet --translate-backend marian
```

`-m opus-mt-tr-en` downloads `opus-mt-tr-en-q8_0.gguf` on first use. The recogniser in the live example must support Turkish.

## Notes

- One model per direction; the opposite direction, where one exists, is `cstr/opus-mt-en-tr-GGUF`.
- Live mode always decodes greedy.
- Literal `</s>`, `<unk>`, `<pad>` in the input are treated as ordinary text here (the reference treats them as special tokens).

## Conversion

```bash
python models/convert-marian-to-gguf.py --input <dir>/opus-mt-tr-en --output opus-mt-tr-en-f16.gguf
./build/bin/crispasr-quantize opus-mt-tr-en-f16.gguf opus-mt-tr-en-q8_0.gguf q8_0
python tools/marian_parity.py --hf-dir <dir>/opus-mt-tr-en --gguf opus-mt-tr-en-f16.gguf --lang en --tgt en --sentences <file>
```
