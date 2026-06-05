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
base_model: speechbrain/lang-id-commonlanguage_ecapa
pipeline_tag: audio-classification
---

# ECAPA-TDNN CommonLanguage LID (GGUF)

GGUF conversion of [speechbrain/lang-id-commonlanguage_ecapa](https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Model Details

- **Architecture**: ECAPA-TDNN (SE-Res2Net + Attentive Statistical Pooling)
- **Parameters**: 21M
- **Size**: 40 MB (F16)
- **Languages**: 45 (CommonLanguage dataset)
- **Input**: 80-dim mel filterbank (vs 60 for VoxLingua107)
- **Embedding dim**: 192 (vs 256 for VoxLingua107)
- **Classifier**: Cosine similarity (vs DNN for VoxLingua107)
- **License**: Apache 2.0
- **Training**: SpeechBrain on CommonLanguage

## Languages (45)

Arabic, Basque, Breton, Catalan, Chinese_China, Chinese_Hongkong, Chinese_Taiwan, Chuvash, Czech, Dhivehi, Dutch, English, Esperanto, Estonian, French, Frisian, Georgian, German, Greek, Hakha_Chin, Indonesian, Interlingua, Italian, Japanese, Kabyle, Kinyarwanda, Kyrgyz, Latvian, Maltese, Mongolian, Persian, Polish, Portuguese, Romanian, Romansh_Sursilvan, Russian, Sakha, Slovenian, Spanish, Swedish, Tamil, Tatar, Turkish, Ukranian, Welsh

## Usage with CrispASR

```bash
# As LID pre-step for any ASR backend
crispasr -m model.gguf --lid-backend ecapa --lid-model ecapa-lid-commonlanguage-f16.gguf -l auto -f audio.wav
```

## Accuracy

Tested on TTS-generated audio samples:

| Language | Detected | Confidence |
|----------|----------|------------|
| English  | English  | 0.033 |
| German   | German   | 0.035 |
| French   | French   | 0.035 |
| Spanish  | Spanish  | 0.033 |
| Japanese | Japanese | 0.044 |

Note: Cosine similarity scores are naturally lower than softmax probabilities. The model ranks correctly even with low absolute confidence.

## Comparison with VoxLingua107

| Feature | VoxLingua107 | CommonLanguage |
|---------|-------------|----------------|
| Languages | 107 | 45 |
| Input mels | 60 | 80 |
| Embedding | 256 | 192 |
| Classifier | DNN | Cosine |
| File size | 43 MB | 40 MB |
| Labels | ISO codes (en, de, ...) | Full names (English, German, ...) |

**Recommendation**: Use VoxLingua107 (`cstr/ecapa-lid-107-GGUF`) for most use cases — wider language coverage and higher confidence scores. CommonLanguage is useful if you need the specific 45-language set or prefer full language names.

## Files

| File | Size | Description |
|------|------|-------------|
| ecapa-lid-commonlanguage-f16.gguf | 40 MB | F16 (recommended) |

## Technical Notes

- Quantized versions not provided — ECAPA-TDNN's small conv kernels (K=3,5) and cosine classifier are precision-sensitive
- The SpeechBrain filterbank matrix is embedded in the GGUF for exact reproduction
- Runtime uses ggml graph for the heavy SE-Res2Net forward pass with CPU-based ASP + classifier

## Citation

```bibtex
@inproceedings{desplanques2020ecapa,
  title={ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Interspeech},
  year={2020}
}
```
