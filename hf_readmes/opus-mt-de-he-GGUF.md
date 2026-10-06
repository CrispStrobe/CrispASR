---
license: cc-by-4.0
language:
- de
- he
base_model:
- Helsinki-NLP/opus-mt-de-he
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

# Opus-MT German → Hebrew — GGUF (ggml)

GGUF / ggml conversion of [`Helsinki-NLP/opus-mt-de-he`](https://huggingface.co/Helsinki-NLP/opus-mt-de-he) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)** (`--backend marian`).

A small (~80M parameters, 6+6 layers, d=512) MarianMT model for one language pair. In CrispASR these are the fastest translators for live transcription + translation (`--live-translate`).

## Licence and attribution

**CC-BY-4.0.** The weights are the `opus-2020-01-29` release of the [OPUS-MT project](https://github.com/Helsinki-NLP/Opus-MT) (Jörg Tiedemann and Santhosh Thottingal, University of Helsinki; *OPUS-MT — Building open translation services for the World*, EAMT 2020), trained on [OPUS](https://opus.nlpl.eu/) data. The project distributes its pre-trained models under CC-BY 4.0 ([Opus-MT README](https://github.com/Helsinki-NLP/Opus-MT), [OPUS-MT-train README](https://github.com/Helsinki-NLP/OPUS-MT-train)); that statement, not the tag on an individual model card, is what this repository follows. This is a format conversion: the weights are unchanged at f16 and quantized at q8_0. Redistribution requires attribution to the OPUS-MT project.

## Files

| File | Size | Output vs. the reference implementation |
|---|---:|---|
| `opus-mt-de-he-f16.gguf` | 159 MB | 8/8 sentences identical greedy, 8/8 with beam 4 |
| `opus-mt-de-he-q8_0.gguf` | 87 MB | 8/8 sentences identical greedy — any others differ in wording. Fastest; recommended. |

"Reference" is `MarianMTModel.generate` from Hugging Face `transformers` on the original checkpoint, on 8 test sentences; input token ids were identical for 8/8.

## Quick start

```bash
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Text → text. -bs 1 is greedy; without it the checkpoint's own beam size is used.
./build/bin/crispasr --backend marian -m opus-mt-de-he -sl de -tl he -bs 1 \
    --text "Guten Morgen und herzlich willkommen zur heutigen Sitzung."
# בוקר טוב, וברוכים הבאים לפגישה של היום.

# Live: microphone in, transcript + translation out, sentence by sentence
./build/bin/crispasr --live-translate -l de --tr-tl he \
    -m auto --backend parakeet --translate-backend marian
```

`-m opus-mt-de-he` downloads `opus-mt-de-he-q8_0.gguf` on first use. The recogniser in the live example must support German.

## Notes

- One model per direction; the opposite direction, where one exists, is `cstr/opus-mt-he-de-GGUF`.
- Live mode always decodes greedy.
- Literal `</s>`, `<unk>`, `<pad>` in the input are treated as ordinary text here (the reference treats them as special tokens).

## Conversion

```bash
python models/convert-marian-to-gguf.py --input <dir>/opus-mt-de-he --output opus-mt-de-he-f16.gguf
./build/bin/crispasr-quantize opus-mt-de-he-f16.gguf opus-mt-de-he-q8_0.gguf q8_0
python tools/marian_parity.py --hf-dir <dir>/opus-mt-de-he --gguf opus-mt-de-he-f16.gguf --lang en --tgt he --sentences <file>
```
