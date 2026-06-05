---
license: apache-2.0
language:
- en
tags:
- gguf
- audio
- speech-recognition
- data2vec
- wav2vec2
- ctc
- automatic-speech-recognition
base_model: facebook/data2vec-audio-base-960h
pipeline_tag: automatic-speech-recognition
---

# Data2Vec Audio (GGUF)

GGUF conversion of [facebook/data2vec-audio-base-960h](https://huggingface.co/facebook/data2vec-audio-base-960h) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Model Details

- **Architecture**: Data2Vec Audio — wav2vec2-style CNN (7L, 512-dim) + 12-layer transformer (768-dim, 12 heads) + CTC head
- **Parameters**: ~95M
- **Training**: Self-supervised pre-training on LibriSpeech 960h, fine-tuned with CTC loss
- **Language**: English only
- **License**: Apache 2.0
- **WER**: 1.89% (LibriSpeech test-clean), 4.07% (test-other)

## Usage with CrispASR

```bash
# Uses the wav2vec2 backend (auto-detected from GGUF architecture)
crispasr --backend wav2vec2 -m data2vec-audio-base-960h-q4_k.gguf -f audio.wav
```

## Architecture Notes

Data2Vec Audio differs from standard wav2vec2 in three ways handled by the converter:

1. **5-layer positional convolution** (vs 1 for wav2vec2), each with Conv1d + LayerNorm(no affine) + GELU
2. **Global encoder LayerNorm BEFORE transformer layers** (vs after for wav2vec2)
3. **POST-norm encoder** despite using LayerNorm in CNN (wav2vec2-large uses pre-norm)

All three are auto-detected from the HuggingFace model config and stored as GGUF metadata flags.

## Files

| File | Size | JFK Transcription |
|------|------|-------------------|
| data2vec-audio-base-960h-f16.gguf | 196 MB | perfect |
| data2vec-audio-base-960h-q4_k.gguf | 79 MB | perfect |
| data2vec-audio-base-960h-q8_0.gguf | 120 MB | perfect |

## Accuracy

Tested on JFK inaugural address (11s):

```
AND SO A MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU
ASK WHAT YOU CAN DO FOR YOUR COUNTRY
```

Identical to the Python HuggingFace reference output. All quantized variants produce the same transcription.

## Citation

```bibtex
@inproceedings{baevski2022data2vec,
  title={data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language},
  author={Baevski, Alexei and Hsu, Wei-Ning and Xu, Qiantong and Babu, Arun and Gu, Jiatao and Auli, Michael},
  booktitle={ICML},
  year={2022}
}
```
