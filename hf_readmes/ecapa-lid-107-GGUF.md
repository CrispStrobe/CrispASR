---
license: apache-2.0
language:
- multilingual
tags:
- audio-classification
- language-identification
- ecapa-tdnn
- speechbrain
- gguf
- crispasr
base_model: speechbrain/lang-id-voxlingua107-ecapa
pipeline_tag: audio-classification
---

# ECAPA-TDNN Language Identification (GGUF)

GGUF conversion of [speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Model Details

- **Architecture**: ECAPA-TDNN (SE-Res2Net + Attentive Statistical Pooling)
- **Parameters**: 21M
- **Size**: 43 MB (F16)
- **Languages**: 107 (VoxLingua107 dataset)
- **License**: Apache 2.0
- **Training**: SpeechBrain on VoxLingua107 (6,628 hours YouTube speech)

## Usage with CrispASR

```bash
# As LID pre-step for any ASR backend
crispasr -m whisper-large-v3.gguf --lid-backend ecapa -l auto -f audio.wav

# Model auto-downloads on first use, or specify path:
crispasr -m model.gguf --lid-backend ecapa --lid-model ecapa-lid-107-f16.gguf -l auto -f audio.wav
```

## Accuracy

Tested on 12-language edge-TTS benchmark (3 samples per language):

| Language | Accuracy | Confidence |
|----------|----------|------------|
| English | 3/3 | p≥0.99 |
| German | 3/3 | p≥0.99 |
| French | 3/3 | p≥0.99 |
| Spanish | 3/3 | p≥0.96 |
| Japanese | 3/3 | p≥0.99 |
| Chinese | 3/3 | p≥0.99 |
| Korean | 3/3 | p≥0.99 |
| Russian | 3/3 | p≥0.99 |
| Arabic | 3/3 | p≥0.99 |
| Hindi | 3/3 | p≥0.99 |
| Portuguese | 3/3 | p≥0.99 |
| Italian | 3/3 | p≥0.99 |

## Files

| File | Size | Description |
|------|------|-------------|
| `ecapa-lid-107-f16.gguf` | 43 MB | F16 weights (recommended) |

## Conversion

```bash
python models/convert-ecapa-tdnn-lid-to-gguf.py \
    --input speechbrain/lang-id-voxlingua107-ecapa \
    --output ecapa-lid-107-f16.gguf
```

## Architecture

```
Input: 16kHz PCM → 60-dim mel fbank (SpeechBrain STFT, n_fft=400)
       → Sentence-level mean normalization
       → Block0: Conv1d(60→1024, k=5) + ReLU + BN
       → Block1-3: SE-Res2Net (1024 channels, 8 sub-bands, dilations 2/3/4)
       → MFA: concatenate block1-3 outputs → Conv1d(3072→3072, k=1) + ReLU + BN
       → ASP: Attentive Statistical Pooling → [6144]
       → BN + FC(6144→256) → embedding
       → Classifier: BN → Linear(256→512) + BN + LeakyReLU → Linear(512→107)
```

## Citation

```bibtex
@inproceedings{ravanelli2021speechbrain,
  title={SpeechBrain: A General-Purpose Speech Toolkit},
  author={Ravanelli, Mirco and others},
  booktitle={Proceedings of the 22nd Annual Conference of the International Speech Communication Association (INTERSPEECH)},
  year={2021}
}
```
