---
license: apache-2.0
language: [multilingual]
tags: [embeddings, gguf, ggml, text-embeddings, qwen3-bidirectional, crispembed]
pipeline_tag: feature-extraction
base_model: BidirLM/BidirLM-Omni-2.5B-Embedding
---

# bidirlm-omni-2.5b GGUF

GGUF format of [BidirLM/BidirLM-Omni-2.5B-Embedding](https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding) for use with [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).

BidirLM-Omni 2.5B — Qwen3-derived bidirectional encoder, 2048-d shared embedding space, 90+ languages. Includes text + audio + vision paths (audio via the shared CrispAudio library; vision via the BidirLM ViT + DeepStack hierarchy).

## Modalities

The upstream model is omnimodal (text + image + audio). This GGUF includes:

- **Text** — bidirectional Qwen3 body with mean pooling. Validated against the
  upstream reference at cosine ≥ 0.999 across the test set.
- **Audio** — Whisper-shape audio tower (Conv2D stem + 24-layer encoder +
  1024→2048 projection). Encodes raw 16 kHz mono PCM to the same 2048-d
  shared embedding space as text, enabling cross-modal cosine similarity.
- **Vision** — 24-layer ViT with 4-corner bilinear pos interp, 2D rotate-half RoPE, and DeepStack hierarchy (3 hooks at config-listed layers). Encodes preprocessed image patches into the same 2048-d shared space as text, validated at cosine ≥ 0.9999 per-token vs the HF reference.

### CLI usage

```bash
# Text
./crispembed -m bidirlm-omni-2.5b "your query"

# Audio (raw f32le 16 kHz mono PCM)
ffmpeg -i clip.wav -ar 16000 -ac 1 -f f32le clip.raw
./crispembed -m bidirlm-omni-2.5b --audio clip.raw

# Image (Python — preprocessor needs Pillow + transformers)
python -c "from crispembed import CrispEmbed; ce=CrispEmbed('bidirlm-omni-2.5b'); print(ce.encode_image('photo.jpg').shape)"
```

### Build requirements

The audio path is provided by the shared **CrispAudio** library (lives in
[CrispASR/crisp_audio](https://github.com/CrispStrobe/CrispASR/tree/main/crisp_audio)).
CrispEmbed's CMake auto-discovers it at the sibling-repo path
`../CrispASR/crisp_audio` (overridable via `-DCRISP_AUDIO_DIR=...`). If that
directory is not present at configure time, `crispembed_has_audio()` returns
0 and the `--audio` flag fails — text encoding still works.

The vision tower is built unconditionally (no sibling-repo dependency).
Image preprocessing in Python uses HF's `Qwen2VLImageProcessorFast` —
`pip install transformers torchvision pillow`.

## Files

| File | Quantization | Size |
|------|-------------|------|
| [bidirlm-omni-2.5b-f16.gguf](https://huggingface.co/cstr/bidirlm-omni-2.5b-GGUF/resolve/main/bidirlm-omni-2.5b-f16.gguf) | F16 | 5264 MB |
| [bidirlm-omni-2.5b-q4_k.gguf](https://huggingface.co/cstr/bidirlm-omni-2.5b-GGUF/resolve/main/bidirlm-omni-2.5b-q4_k.gguf) | Q4_K | 1475 MB |
| [bidirlm-omni-2.5b-q5_k.gguf](https://huggingface.co/cstr/bidirlm-omni-2.5b-GGUF/resolve/main/bidirlm-omni-2.5b-q5_k.gguf) | Q5_K | 1729 MB |
| [bidirlm-omni-2.5b-q6_k.gguf](https://huggingface.co/cstr/bidirlm-omni-2.5b-GGUF/resolve/main/bidirlm-omni-2.5b-q6_k.gguf) | Q6_K | 1998 MB |
| [bidirlm-omni-2.5b-q8_0.gguf](https://huggingface.co/cstr/bidirlm-omni-2.5b-GGUF/resolve/main/bidirlm-omni-2.5b-q8_0.gguf) | Q8_0 | 3368 MB |


## Parity vs HuggingFace reference

Cosine similarity vs the upstream sentence-transformers reference on a fixed
test set (text + audio (jfk.wav) + vision (cat.jpg)):

| Quant | Text | Audio | Vision |
|------|-------:|-------:|-------:|
| f16 | 0.9998 | 0.9949 | 0.9999 |
| q8_0 | 0.9991 | 0.9952 | 0.9953 |
| q6_k | 0.9939 | 0.9949 | 0.9939 |
| q5_k | 0.9831 | 0.9945 | 0.9884 |
| q4_k | 0.9374 | 0.9915 | 0.9662 |

*Note:* below the 0.99 retrieval-quality bar — text: `q5_k` (0.983), `q4_k` (0.937); vision: `q5_k` (0.988), `q4_k` (0.966). Embeddings are still functionally usable (>0.9 = directionally correct for similarity ranking) but expect small differences in nearest-neighbor results vs the upstream f32 reference.


## Quick Start

```bash
# Download
huggingface-cli download cstr/bidirlm-omni-2.5b-GGUF bidirlm-omni-2.5b-f16.gguf --local-dir .

# Run with CrispEmbed
./crispembed -m bidirlm-omni-2.5b-f16.gguf "Hello world"

# Or with auto-download
./crispembed -m bidirlm-omni-2.5b "Hello world"
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Qwen3-Bidirectional |
| Parameters | 2.5B |
| Embedding Dimension | 2048 |
| Layers | 28 |
| Pooling | mean |
| Tokenizer | BPE |
| Base Model | [BidirLM/BidirLM-Omni-2.5B-Embedding](https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding) |

## Verification

Verified bit-identical to HuggingFace sentence-transformers (cosine similarity >= 0.999 on test texts).

## Usage with CrispEmbed

CrispEmbed is a lightweight C/C++ text embedding inference engine using ggml.
No Python runtime, no ONNX. Supports BERT, XLM-R, Qwen3, and Gemma3 architectures.

```bash
# Build CrispEmbed
git clone https://github.com/CrispStrobe/CrispEmbed
cd CrispEmbed
cmake -S . -B build && cmake --build build -j

# Encode
./build/crispembed -m bidirlm-omni-2.5b-f16.gguf "query text"

# Server mode
./build/crispembed-server -m bidirlm-omni-2.5b-f16.gguf --port 8080
curl -X POST http://localhost:8080/v1/embeddings \
    -d '{"input": ["Hello world"], "model": "bidirlm-omni-2.5b"}'
```

## Credits

- Original model: [BidirLM/BidirLM-Omni-2.5B-Embedding](https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding)
- Inference engine: [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) (ggml-based)
- Conversion: `convert-decoder-embed-to-gguf.py`
