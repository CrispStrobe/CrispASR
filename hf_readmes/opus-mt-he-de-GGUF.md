---
license: cc-by-4.0
language:
- he
- de
base_model:
- Helsinki-NLP/opus-mt-he-de
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

# Opus-MT Hebrew → German — GGUF (ggml)

GGUF / ggml conversion of [`Helsinki-NLP/opus-mt-he-de`](https://huggingface.co/Helsinki-NLP/opus-mt-he-de) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)** (`--backend marian`).

A small (~80M parameters, 6+6 layers, d=512) MarianMT model for one language pair. In CrispASR these are the fastest translators for live transcription + translation (`--live-translate`).

## Licence and attribution

**CC-BY-4.0.** The weights are the `opus-2020-01-26` release of the [OPUS-MT project](https://github.com/Helsinki-NLP/Opus-MT) (Jörg Tiedemann and Santhosh Thottingal, University of Helsinki; *OPUS-MT — Building open translation services for the World*, EAMT 2020), trained on [OPUS](https://opus.nlpl.eu/) data. The project distributes its pre-trained models under CC-BY 4.0 ([Opus-MT README](https://github.com/Helsinki-NLP/Opus-MT), [OPUS-MT-train README](https://github.com/Helsinki-NLP/OPUS-MT-train)); that statement, not the tag on an individual model card, is what this repository follows. This is a format conversion: the weights are unchanged at f16 and quantized at q8_0. Redistribution requires attribution to the OPUS-MT project.

## Files

| File | Size | Output vs. the reference implementation |
|---|---:|---|
| `opus-mt-he-de-f16.gguf` | 159 MB | 8/8 sentences identical greedy, 8/8 with beam 4 |
| `opus-mt-he-de-q8_0.gguf` | 87 MB | 7/8 sentences identical greedy — any others differ in wording. Fastest; recommended. |

"Reference" is `MarianMTModel.generate` from Hugging Face `transformers` on the original checkpoint, on 8 test sentences; input token ids were identical for 8/8.

## Quick start

```bash
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Text → text. -bs 1 is greedy; without it the checkpoint's own beam size is used.
./build/bin/crispasr --backend marian -m opus-mt-he-de -sl he -tl de -bs 1 \
    --text "בוקר טוב וברוכים הבאים לישיבה של היום."
# Guten Morgen und willkommen zur heutigen Sitzung.

# Live: microphone in, transcript + translation out, sentence by sentence
./build/bin/crispasr --live-translate -l he --tr-tl de \
    -m auto --backend parakeet --translate-backend marian
```

`-m opus-mt-he-de` downloads `opus-mt-he-de-q8_0.gguf` on first use. The recogniser in the live example must support Hebrew.

## Notes

- One model per direction; the opposite direction, where one exists, is `cstr/opus-mt-de-he-GGUF`.
- Live mode always decodes greedy.
- Literal `</s>`, `<unk>`, `<pad>` in the input are treated as ordinary text here (the reference treats them as special tokens).

## Conversion

```bash
python models/convert-marian-to-gguf.py --input <dir>/opus-mt-he-de --output opus-mt-he-de-f16.gguf
./build/bin/crispasr-quantize opus-mt-he-de-f16.gguf opus-mt-he-de-q8_0.gguf q8_0
python tools/marian_parity.py --hf-dir <dir>/opus-mt-he-de --gguf opus-mt-he-de-f16.gguf --lang en --tgt de --sentences <file>
```
