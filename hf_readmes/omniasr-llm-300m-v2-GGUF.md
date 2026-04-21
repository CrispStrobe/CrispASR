---
license: apache-2.0
language:
- multilingual
tags:
- gguf
- ggml
- audio
- speech-recognition
- transcription
- omniasr
- wav2vec2
- llama
- automatic-speech-recognition
- en
- de
- fr
- es
- ja
- ko
- zh
- ar
- hi
- pt
- ru
base_model: facebook/omniASR-LLM-300M
pipeline_tag: automatic-speech-recognition
---

# OmniASR-LLM-300M-v2 (GGUF)

GGUF conversion of Facebook's [omniASR-LLM-300M-v2](https://github.com/facebookresearch/omnilingual-asr) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Model Details

- **Architecture**: wav2vec2 encoder (24L, d=1024) + 12-layer LLaMA decoder (d=4096, 8 heads, SwiGLU, RMSNorm, RoPE)
- **Total Parameters**: ~1.5B (300M encoder + 1.2B decoder)
- **Encoder**: 7-layer CNN frontend (320x downsampling) + 24L transformer with grouped positional convolution
- **Decoder**: 12-layer LLaMA-style autoregressive transformer with language conditioning
- **Languages**: 1600+ (via language ID token, e.g. `eng_Latn`, `deu_Latn`)
- **Tokenizer**: Character-based SentencePiece v2 (10,288 tokens)
- **License**: Apache 2.0
- **Input**: Raw 16 kHz mono PCM (no mel features)

## Usage with CrispASR

```bash
# Auto-detected from GGUF metadata (omniasr.model_type=1)
crispasr --backend omniasr -m omniasr-llm-300m-v2-f16.gguf -l en -f audio.wav

# With explicit language code (recommended for best quality)
crispasr --backend omniasr -m omniasr-llm-300m-v2-q4_k.gguf -l eng_Latn -f audio.wav
```

## Accuracy

Tested on JFK inaugural address (11s, vintage recording):

| Variant | Output |
|---------|--------|
| **LLM v2 (this model)** | "and so my palamericas is not what your country can do for you is what you can do for your country" |
| CTC 300M | "en so my tonek n what yor campri kand fur yo s watyukandfur yor kontry" |

The LLM variant produces significantly better transcription quality than CTC, with correct word boundaries and English grammar.

## Architecture Details

The model runs as a two-phase pipeline:

1. **Encoder phase** (ggml graph, fast):
   - Input normalization (layer_norm waveform)
   - 7-layer CNN (512-dim, strides [5,2,2,2,2,2,2] = 320x downsampling)
   - Post-extract LayerNorm + linear projection (512 -> 1024)
   - Grouped positional convolution (K=128, groups=16)
   - 24-layer transformer encoder (d=1024, 16 heads, GELU FFN)

2. **Decoder phase** (ggml graph with KV cache):
   - Encoder projection (1024 -> 4096)
   - Prefix: [audio_embeddings, lid_marker, lang_embedding, BOS]
   - 12-layer LLaMA decoder (d=4096, 8 heads, head_dim=512, SwiGLU d_ffn=2816)
   - RoPE: interleaved mode (fairseq2 convention)
   - Greedy autoregressive decoding

## Language Conditioning

The LLM variant requires a language code for optimal quality. Without it, the model may hallucinate or transcribe in the wrong language.

Common language IDs (embedding index from `languges_lookup_table.parquet`):

| Language | Code | Embedding Index |
|----------|------|-----------------|
| English | eng_Latn | 417 |
| German | deu_Latn | 367 |
| French | fra_Latn | 448 |
| Spanish | spa_Latn | 1355 |
| Japanese | jpn_Jpan | 632 |
| Korean | kor_Hang | 734 |

## Files

| File | Size | Description |
|------|------|-------------|
| omniasr-llm-300m-v2-f16.gguf | 3.1 GB | F16 (full precision) |
| omniasr-llm-300m-v2-q4_k.gguf | 1.1 GB | Q4_K (recommended) |
| omniasr-llm-300m-v2-q8_0.gguf | 1.8 GB | Q8_0 (higher quality) |

Bridging tensors (enc_proj, lm_head, tok_emb, lang_emb) are kept at F16 in all
quantized variants for accuracy. Only the encoder and decoder transformer
weights are quantized.

## Performance

On CPU (4 threads):
- Encoder: ~8s for 11s audio (ggml graph)
- Decoder prefill: ~20s for 552 tokens
- Generation: ~0.5s/token (ggml graph with KV cache)
- Total: ~75s for 11s audio

## Citation

```bibtex
@article{auli2024omnilingual,
  title={Omnilingual ASR: Open-Source Multilingual Speech Recognition},
  author={Auli, Michael and others},
  journal={arXiv preprint arXiv:2511.09690},
  year={2024}
}
```
