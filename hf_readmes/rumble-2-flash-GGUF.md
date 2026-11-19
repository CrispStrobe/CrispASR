---
license: apache-2.0
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- gguf
- ggml
- crispasr
- qwen3-asr
base_model: Reson8/Rumble-2-Flash
library_name: ggml
---

# Rumble-2-Flash — GGUF

GGUF conversions of [`Reson8/Rumble-2-Flash`](https://huggingface.co/Reson8/Rumble-2-Flash)
(Apache-2.0) for **[CrispASR](https://github.com/CrispStrobe/CrispASR)** (`--backend qwen3`, or `-m rumble-2-flash`).

Rumble-2-Flash is a 0.5B Qwen3-ASR fine-tune covering 45 languages. It pairs an 18-layer audio encoder with a
14-layer Qwen3 LLM and a re-indexed 99,630-token vocabulary. Reson8 control tokens
(`<digits><nofiller><verbatim>`, the chat template defaults) open the system turn. The GGUF records them as
`qwen3asr.system_prefix`, and the encoder runs with the model's 8 s attention windows.

## Files

| File | Size | Notes |
| --- | --- | --- |
| `rumble-2-flash-f16.gguf`  | 1.23 GB | F16 |
| `rumble-2-flash-q8_0.gguf` | 657 MB  | Q8_0 |
| `rumble-2-flash-q4_k.gguf` | 445 MB  | Q4_K (default for `-m rumble-2-flash`) |

## Usage

```bash
crispasr -m rumble-2-flash -f audio.wav            # language auto-detected
crispasr -m rumble-2-flash -f audio.wav -l nl      # forced language
CRISPASR_QWEN3_SYSTEM_PREFIX='<nodigits><filler><noverbatim>' crispasr -m rumble-2-flash -f audio.wav
```

`-l` forces the language the way the chat template's `language` option does: the assistant turn is prefilled
with `language <Name><asr_text>`. `CRISPASR_QWEN3_SYSTEM_PREFIX` sets the `digits` / `filler` / `verbatim`
options. The auxiliary CTC head in the original repository is not included.

## Validation

`tools/ci-heavy/rumble2_validate.py` compares these files against fp32 transformers on `jfk.wav` (English),
`ko-369.wav` (Korean) and `paraformer_zh.wav` (Chinese), each with auto-detected and forced language:

- F16 matches transformers at every `crispasr-diff` stage (mel, conv, all 18 encoder blocks, projector, prompt
  ids, first-step logits), with cosine 1.000000 on all three clips.
- All 18 transcripts (3 quants × 6 cases) are byte-identical to the transformers output. JFK WER is 0.
