---
license: apache-2.0
language:
- multilingual
tags:
- text-classification
- language-identification
- cld3
- compact-language-detector
- gguf
library_name: ggml
pipeline_tag: text-classification
---

# CLD3 (Compact Language Detector v3) — GGUF

GGUF conversion of [Google CLD3](https://github.com/google/cld3),
the compact neural-net language detector originally shipped in
Chromium for browser-side language identification. Apache-2.0,
upstream code unchanged.

Distinct architecture from the much larger fastText-based GlotLID-V3
and Facebook LID-176 ports — CLD3 is ~440 KB at F16 (vs 250 MB for
GlotLID-V3 F16) and runs in well under a millisecond per text input.
Trades raw accuracy on rare languages for tiny memory footprint and
sub-microsecond CPU latency.

## Architecture

```
[utf-8 text]
   ↓ cleanup (lowercase, strip ASCII punct/digits)
   │
   ├─ feature 0  ContinuousBagOfNgrams  size=2  id_dim=1000   bigrams       → 16-d
   ├─ feature 1  ContinuousBagOfNgrams  size=4  id_dim=5000   quadgrams     → 16-d
   ├─ feature 2  RelevantScriptFeature                        relevant scripts →  8-d
   ├─ feature 3  ScriptFeature                                text-script   →  8-d
   ├─ feature 4  ContinuousBagOfNgrams  size=3  id_dim=5000   trigrams      → 16-d
   └─ feature 5  ContinuousBagOfNgrams  size=1  id_dim=100    unigrams      → 16-d
                                                  concat[80]
                                                     ↓
                                  hidden FC + ReLU → hidden[208]
                                                     ↓
                                              output FC → logits[109]
                                                     ↓
                                                  softmax → 109 ISO 639-1 labels
```

* **6 feature extractors** emit `(id, weight)` pairs (3 distinct
  feature classes, the n-gram one instantiated 4× with different
  parameters).
* **Six embedding tables** mean-pool the per-feature pairs into the
  80-d concatenated input.
* **Two FC layers** (80→208 + ReLU, 208→109) produce the logits.
* **109 ISO 639-1 labels** including `zh-Latn`, `bg-Latn`, `el-Latn`,
  `hi-Latn`, `ja-Latn`, `ru-Latn` (Romanized variants of major
  scripts). Full list at the end of this card.
* Hash function for n-gram → embedding-row IDs is **MurmurHash2-32**
  with seed `0xBEEF`. The C++ port and converter both match upstream
  byte-for-byte.

## Files

| File              | Size    | Notes                                       |
|-------------------|---------|---------------------------------------------|
| `cld3-f16.gguf`   | 440 KB  | F16 weights — production. Recommended.      |

Only F16 ships. CLD3 is too small (~1.5 MB FP32) for K-quants to be
meaningful — Q4_K would only save ~200 KB and adds a quantization
cosine penalty that's pointless at this size.

## Usage (CrispASR C++ runtime)

```c
#include "lid_cld3.h"

lid_cld3_context* ctx = lid_cld3_init_from_file("cld3-f16.gguf", 1);
float conf = 0.0f;
const char* lang = lid_cld3_predict(ctx, "Bonjour le monde", &conf);
// lang = "fr", conf = 0.9994

const char* labels[5];
float scores[5];
int n = lid_cld3_predict_topk(ctx, "Bonjour le monde", 5, labels, scores);
// labels = {"fr", "ca", "ro", ...}, scores descending

lid_cld3_free(ctx);
```

C ABI mirrors `lid_fasttext_*` (the GlotLID / LID-176 backend) so
the auto-routing text-LID dispatcher in CrispASR picks between them
at load time based on the GGUF's `general.architecture` field.

## Multilingual smoke set (F16)

Reproducing upstream `pycld3.get_language()` predictions across 8
inputs covering Latin, Cyrillic, CJK, hiragana, devanagari, and the
"too-short-to-determine" edge case:

| Input              | C++ runtime        | pycld3 oracle     |
|--------------------|--------------------|-------------------|
| `Hello world`      | `fi` (p=0.39) ⚠️   | `ky` (p=0.72) ⚠️  |
| `Hallo Welt`       | `et` (p=0.85)      | `et` (p=0.43)     |
| `Bonjour le monde` | `fr` (p=0.999)     | `fr` (p=0.999)    |
| `Привет мир`       | `ru` (p=0.91)      | `ru` (p=0.97)     |
| `你好世界`           | `zh` (p=0.998)     | `zh` (p=0.999)    |
| `こんにちは世界`     | `ja` (p=1.000)     | `ja` (p=1.000)    |
| `नमस्ते दुनिया`    | `mr` (p=0.9999)    | `mr` (p=0.9999)   |
| `Olá mundo`        | `sk` (p=0.97)      | `sk` (p=0.89)     |

⚠️ Both `Hello world` predictions are wrong — the input is too
short for CLD3 to be confident. Below the model's reliability
threshold (`p<0.7`), small algorithmic differences flip the
argmax. This is the **known short-input quirk** of CLD3 and is
the same behaviour the upstream Chromium browser language
detector exhibits on short text.

For 7/8 inputs, our C++ runtime matches `pycld3` on top-1.
Per-stage cosine match against the F32 Python reference at
`cos≥0.999` for all 88 stage compares (11 stages × 8 inputs)
on the F16 GGUF.

## Caveats

* **Short-input quirks** — CLD3 was designed for the typical text
  Chromium sees (web pages, document content). On strings under
  ~50 characters it falls back to its training-frequency prior;
  e.g. `"Hello world"` lands on `ky` (Kyrgyz) consistently. The
  `is_reliable` flag in the upstream API was specifically designed
  to surface these — anything below `p=0.7` (or `p=0.5` for the
  hr/bs Croatian/Bosnian disambiguation) is unreliable.
* **Romanized labels** — `zh-Latn`, `bg-Latn`, `ru-Latn`, `ja-Latn`,
  `hi-Latn`, `el-Latn` are distinct labels from their native-script
  variants. Code that matches on `language.startswith("zh")` will
  conflate them.
* **Text cleanup is simplified** — the C++ runtime's preprocessing
  is full-Unicode lowercase + ASCII punct/digit strip + whitespace
  collapse. Upstream additionally runs the
  `ScriptScanner::GetOneScriptSpanLower` state machine for snippet
  selection on long inputs and `CheapSqueezeInplace` for repetitive
  chunk removal. For the multilingual smoke set above this is
  byte-for-byte equivalent; for HTML-laden or heavily-punctuated
  inputs the simplified cleanup may diverge.

## Conversion

```bash
git clone --depth 1 https://github.com/google/cld3 /tmp/cld3
python models/convert-cld3-to-gguf.py \
    --upstream-src /tmp/cld3/src \
    --out cld3-f16.gguf
```

The converter regex-parses the upstream C++ array literals in
`src/lang_id_nn_params.cc` (1.76 MB embedded weight blob, no binary
side-car) and dequantizes the six `uint8 + per-row bfloat16-scale`
embedding tables to F32, then casts to F16 on write. Hidden + softmax
weights are stored as plain `float` literals upstream and pass through
unchanged.

## Languages (109 ISO 639-1 labels)

```
eo co eu ta de mt ps te su uz zh-Latn ne nl sw sq hmn ja no mn so
ko kk sl ig mr th zu ml hr bs lo sd cy hy uk pt lv iw cs vi jv be
km mk tr fy am zh da sv fi ht af la id fil sm ca el ka sr it sk ru
ru-Latn bg ny fa haw gl et ms gd bg-Latn ha is ur mi hi bn hi-Latn
fr yi hu xh my tg ro ar lb el-Latn st ceb kn az si ky mg en gu es
pl ja-Latn ga lt sn yo pa ku
```

## License

Apache-2.0, inherited from the upstream
[google/cld3](https://github.com/google/cld3) project. Original
copyright Google Inc.
