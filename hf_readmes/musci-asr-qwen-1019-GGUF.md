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
base_model: Musci-research/Musci-ASR-qwen-1019
library_name: ggml
---

# Musci-ASR-qwen-1019 — GGUF

GGUF conversions of [`Musci-research/Musci-ASR-qwen-1019`](https://huggingface.co/Musci-research/Musci-ASR-qwen-1019)
(Apache-2.0) for **[CrispASR](https://github.com/CrispStrobe/CrispASR)** (`--backend qwen3`, or `-m musci-asr`).

Musci-ASR-qwen-1019 is an unofficial research fine-tune of
[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), trained with GRPO on Open ASR Leaderboard training
prompts and curated hard cases. Its config, tokenizer, chat template and preprocessor are identical to the base
model, so it runs on CrispASR's standard qwen3 runtime. It is not an official Qwen release.

## Files

| File | Notes |
| --- | --- |
| `musci-asr-qwen-1019-f16.gguf` | F16, 4.70 GB |
| `musci-asr-qwen-1019-q8_0.gguf` | Q8_0, 2.51 GB |
| `musci-asr-qwen-1019-q4_k.gguf` | Q4_K, 1.49 GB (default for `-m musci-asr`) |

## Usage

```bash
crispasr -m musci-asr --auto-download -f audio.wav            # language auto-detected
crispasr --backend qwen3 -m musci-asr-qwen-1019-q8_0.gguf -f audio.wav -l en
```

## Validation

`tools/ci-heavy/musci_validate.py` converts the safetensors at revision `c376e9c4` (sha256 checked). It compares
the F16, Q8_0 and Q4_K files against the fp32 `qwen_asr` reference on `jfk.wav` (English) and
`paraformer_zh.wav` (Chinese), checking every `crispasr-diff qwen3` stage and the CLI transcripts.

Results (heavy-cpu run 44600040132; these files are that run's byte-verified outputs):

- **F16:** every stage passes on both clips with cosine 1.000000. That covers the mel, conv front end, all 24
  encoder blocks, `ln_post`, the projector, prompt ids and the first-step logits.
- **Transcripts:** all six (3 quants × 2 clips) match the fp32 reference. Q4_K on the Chinese clip changes only
  punctuation (CER 0 after normalisation). JFK WER is 0 for every quant.

## License and attribution

Apache-2.0, as for the upstream model.

- Model: [Musci-research/Musci-ASR-qwen-1019](https://huggingface.co/Musci-research/Musci-ASR-qwen-1019)
- Base model: [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) by the Qwen team, Apache-2.0
- Changes in this repository: format conversion to GGUF and quantization. No retraining.
