---
license: cc-by-4.0
language:
- bg
- cs
- da
- de
- el
- en
- es
- et
- fi
- fr
- hr
- hu
- it
- lt
- lv
- mt
- nl
- pl
- pt
- ro
- ru
- sk
- sl
- sv
- uk
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- forced-alignment
- ctc
- ggml
- gguf
- canary
- nemo
- nemo-forced-aligner
- nfa
- multilingual
library_name: ggml
base_model: nvidia/canary-1b-v2
---

# Canary-1B-v2 CTC aligner — GGUF (ggml-quantised)

GGUF / ggml conversions of the **auxiliary CTC alignment model** that ships *inside* [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2)'s `.nemo` tarball, repackaged as a standalone forced-aligner for use with the `nfa-align` CLI from **[CrispStrobe/CrispASR@parakeet](https://github.com/CrispStrobe/CrispASR/tree/parakeet)**.

## What is this?

When you download `nvidia/canary-1b-v2` you get a single `canary-1b-v2.nemo` tarball. Inside that tarball there are TWO weight files:

- `model_weights.ckpt` — the main 978 M-parameter Canary encoder–decoder ASR / translation model
- **`timestamps_asr_model_weights.ckpt`** — a separate, fully-trained 600 M-parameter Parakeet-style FastConformer + CTC head, trained with the **same unified SentencePiece tokenizer as Canary-1B-v2** for 250 000 steps on the same Granary dataset (Canary paper §5)

That second model is what NVIDIA calls the "auxiliary CTC model" used by [NeMo Forced Aligner](https://github.com/NVIDIA-NeMo/NeMo/tree/main/tools/nemo_forced_aligner) (NFA) to compute frame-aligned word and segment timestamps for Canary's transcript output. **It's the official NVIDIA-recommended path for getting word-level timestamps from Canary**, since the main encoder–decoder doesn't emit them natively (cross-attention DTW tops out around ~360 ms MAE; CTC forced alignment gives ~30-80 ms).

This repo extracts that auxiliary model as a standalone GGUF and exposes it as a **general-purpose multilingual subword CTC ASR + forced aligner** in 25 European languages. Because the aligner re-tokenises any input transcript through its own SentencePiece vocabulary, it works as a drop-in alignment tool for transcripts produced by **any other ASR system** — Canary, Parakeet, Cohere Transcribe, Whisper, even hand-typed text.

## License & attribution — IMPORTANT

This model is **`nvidia/canary-1b-v2`'s auxiliary CTC component**, redistributed under the same **CC-BY-4.0** license as the original. **All credit and attribution go to NVIDIA's NeMo team**, who:

1. Designed the FastConformer + CTC architecture
2. Trained the model on the Granary dataset (1.7 M hours of multilingual speech)
3. Released `nvidia/canary-1b-v2` (and the auxiliary model bundled inside it) under CC-BY-4.0

**If you use this GGUF you MUST attribute NVIDIA per the CC-BY-4.0 terms:**

> *This work uses the auxiliary CTC alignment model from NVIDIA's [`canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2), released under CC-BY-4.0. Cite the technical report: Sekoyan et al., ["Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST"](https://arxiv.org/abs/2509.14128) (arXiv:2509.14128, 2025).*

The original model card, paper, and license are at:

- Model card: https://huggingface.co/nvidia/canary-1b-v2
- Technical report: https://arxiv.org/abs/2509.14128
- NeMo Forced Aligner (the tool that uses this model): https://github.com/NVIDIA-NeMo/NeMo
- NVIDIA NeMo: https://github.com/NVIDIA-NeMo/NeMo

The `.nemo` extraction + GGUF conversion + ggml runtime in this repo are MIT-licensed (matching the [CrispASR](https://github.com/CrispStrobe/CrispASR) base they extend), but **the model weights themselves are CC-BY-4.0 from NVIDIA**.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `canary-ctc-aligner.gguf`        | 1.25 GB | F16, full precision |
| `canary-ctc-aligner-q8_0.gguf`   | 704 MB  | Q8_0, near-lossless |
| `canary-ctc-aligner-q5_0.gguf`   | 508 MB  | Q5_0 |
| `canary-ctc-aligner-q4_k.gguf`   | 442 MB  | **Q4_K — recommended default, byte-identical alignment to F16** |

## Quick start

```bash
# 1. Build the runtime
git clone -b parakeet https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target nfa-align

# 2. Download the aligner GGUF
huggingface-cli download cstr/canary-ctc-aligner-GGUF \
    canary-ctc-aligner-q4_k.gguf --local-dir .

# 3. Greedy CTC decode (sanity check — works as a standalone ASR too)
./build/bin/nfa-align \
    -m canary-ctc-aligner-q4_k.gguf \
    -f your-audio.wav -decode -t 8

# 4. Forced alignment of an existing transcript
./build/bin/nfa-align \
    -m canary-ctc-aligner-q4_k.gguf \
    -f your-audio.wav \
    -tt "your transcript text here" -t 8

# Or read the transcript from a file
./build/bin/nfa-align \
    -m canary-ctc-aligner-q4_k.gguf \
    -f your-audio.wav \
    -tf transcript.txt -t 8
```

## Verified accuracy

Tested on `samples/jfk.wav` (11 s, 22 words). Ground truth from `cohere-align` (wav2vec2 character-level CTC, the standard reference, ~30-50 ms expected MAE):

| Word | cohere-align (truth) | **nfa-align (this)** | canary-DTW |
| --- | ---: | ---: | ---: |
| And | 0.39 | **0.40** | 1.12 |
| so | 0.62 | **0.64** | 1.12 |
| my | 1.04 | **1.12** | 2.16 |
| fellow | 1.30 | **1.36** | 2.16 |
| Americans | 1.68 | **1.84** | 2.72 |
| ask | 3.52 | **3.52** | 4.08 |
| ... | ... | ... | ... |

**Mean Absolute Error (22 words): 78 ms** for nfa-align vs 414 ms for canary's cross-attention DTW path. **5.3× tighter word boundaries.** That's right at the cohere-align ceiling — for English the wav2vec2 reference is marginally better because its character-level granularity is finer than subword, but the difference is within the encoder frame quantum (80 ms).

For non-English languages where there's no wav2vec2 character-CTC model, **nfa-align is the most accurate forced aligner this fork has.**

## How it works under the hood

```
audio (16 kHz mono PCM)
  ↓ NeMo mel preprocessor (128 mels, n_fft=512, hop=160, Hann + log + per-feature norm)
  ↓ Conv2d dw_striding subsampling (8× temporal: 100 → 12.5 fps)
  ↓ 24× FastConformer block:
       FFN1 (Macaron, ½ scale)
       MHA  (rel-pos with Transformer-XL untied biases)
       Conv (pw1 + GLU + dw_k=9 + BN + swish + pw2)
       FFN2 (Macaron, ½ scale)
       LN_out
  ↓ ctc_head: Linear(1024 → 16385)
  → log-softmax → per-frame logits over 16384 vocab + 1 blank

Forced alignment:
  1. Tokenise the input transcript word-by-word using greedy
     longest-prefix matching against the SentencePiece vocab
     (handles the U+2581 ▁ word-boundary marker).
  2. Build the CTC-expanded label sequence:
     [blank, t0, blank, t1, ..., blank, t_{N-1}, blank]
  3. Run Viterbi DP in log-probability space:
     - At each frame t, for each label position j: stay / advance / skip-blank
  4. Traceback → optimal monotone path → per-token frame index
  5. Map per-token frames back to per-word (t0, t1) in centiseconds
```

The encoder is **architecturally identical to `nvidia/parakeet-tdt-0.6b-v3`** (24-layer FastConformer, no biases on q/k/v/out/ff/conv) — only the head differs (linear CTC vs LSTM TDT). That's why we can reuse the same encoder loading + BN-folding + graph-build code. The only canary-specific piece is the SentencePiece subword Viterbi.

## Comparison with other alignment tools

| Tool | Vocab | Languages | English MAE on JFK | Notes |
| --- | --- | --- | ---: | --- |
| `cohere-align` (wav2vec2-xlsr-en) | 33 chars | 1 (English) | ~30 ms | Char-level CTC, single language per model |
| **`nfa-align` (this)**       | **16384 SP** | **25** | **78 ms** | **Subword CTC, all 25 EU languages in one model** |
| `cohere-main` cross-attn DTW | — | 14 | ~360 ms | Cohere's built-in DTW |
| `canary-main` cross-attn DTW | — | 25 | ~414 ms | Canary's built-in DTW |
| `parakeet-main` TDT durations | 8192 SP | 25 | ~80 ms | Free from the TDT decoder |

For the 25 European languages canary supports, `nfa-align` is the **only forced-alignment tool with frame-level (~80 ms) accuracy that doesn't require a separate per-language model**. The wav2vec2 path (`cohere-align`) is marginally tighter but you'd need 25 separate language-specific GGUFs.

## Compatibility — works with any ASR transcript

Because `nfa-align` re-tokenises words through the auxiliary model's SentencePiece vocab, you can drop in transcripts from **any other ASR system**:

```bash
# Cohere transcript → align with canary CTC
./build/bin/cohere-main -m cohere.gguf -f audio.wav -np > t.txt
./build/bin/nfa-align -m canary-ctc-aligner-q4_k.gguf -f audio.wav -tf t.txt

# Parakeet transcript → align with canary CTC
./build/bin/parakeet-main -m parakeet.gguf -f audio.wav -np > t.txt
./build/bin/nfa-align -m canary-ctc-aligner-q4_k.gguf -f audio.wav -tf t.txt

# Canary transcript → align with canary CTC (the official path)
./build/bin/canary-main -m canary.gguf -f audio.wav -sl en -tl en -np > t.txt
./build/bin/nfa-align -m canary-ctc-aligner-q4_k.gguf -f audio.wav -tf t.txt

# Whisper / OpenAI / hand-typed text → also works
./build/bin/nfa-align -m canary-ctc-aligner-q4_k.gguf -f audio.wav \
    -tt "your transcript here, in any of the 25 supported languages"
```

## Standalone ASR mode

The auxiliary model is a fully-trained CTC ASR in its own right. Pass `-decode` to use it as a transcriber instead of an aligner:

```bash
$ ./build/bin/nfa-align -m canary-ctc-aligner-q4_k.gguf -f samples/jfk.wav -decode
And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
```

It's not as accurate as the main `canary-1b-v2` encoder–decoder for general transcription (CTC has no language model component, and the 600 M parakeet-style backbone is smaller than canary's 978 M AED), but it's a useful sanity check and works as a fast multilingual fallback ASR if you don't need the full Canary stack.

## Supported languages (25)

`bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk`

Same coverage as Canary-1B-v2 and Parakeet-TDT-0.6B-v3, since this is the same Granary-trained backbone.

## Architecture details

| Component | Details |
| --- | --- |
| Encoder       | 24-layer FastConformer (Parakeet-style, no biases on q/k/v/out/ff), d=1024, 8 heads, head_dim=128, FFN=4096, conv kernel=9 |
| Subsampling   | Conv2d dw_striding stack, 8× temporal (100 → 12.5 fps) |
| CTC head      | Linear (1024 → 16385) — squeezed from NeMo's `Conv1d(1024, 16385, kernel=1)` |
| Vocab         | 16384 SentencePiece pieces + 1 blank token (NeMo CanaryBPETokenizer family) |
| Audio         | 16 kHz mono, 128 mel bins, n_fft=512, hop=160, win=400 |
| Frame rate    | 12.5 fps after subsampling = **80 ms / encoder frame** = the alignment quantum |
| Parameters    | ~600 M (encoder 608.9 M + CTC head 16.8 M + preprocessor 0) |
| Tensors       | 712 in the GGUF (24 × 29 encoder + 12 pre-encode + 2 preprocessor + 2 CTC head + 24 synthetic conv.dw.bias) |

## How this was made

1. **Inspect** `canary-1b-v2.nemo`: there's a separate `timestamps_asr_model_weights.ckpt` and `timestamps_asr_model_config.yaml` inside the tarball. The weights file has 712 tensors, ~625 M parameters, structured as `encoder.* + decoder.decoder_layers.0.*`.
2. **Convert** with [`models/convert-canary-ctc-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/parakeet/models/convert-canary-ctc-to-gguf.py): extract just the auxiliary checkpoint (not the main canary), remap NeMo state-dict keys (`encoder.layers.{i}.feed_forward1.linear1.weight` → `encoder.layers.{i}.ff1.linear1.weight`, etc.), squeeze the trailing kernel-1 dim on the CTC weight to make it a plain 2D linear, and write 712 tensors as F16 + F32.
3. **C++ runtime** in [`src/canary_ctc.{h,cpp}`](https://github.com/CrispStrobe/CrispASR/blob/parakeet/src/canary_ctc.cpp): mmap the GGUF, fold BN into the depthwise conv at load time, build the encoder graph (identical to parakeet's), add a final `mul_mat` with the CTC head, return per-frame logits.
4. **Subword Viterbi** in the same file: greedy longest-prefix tokenisation against the vocab, build the CTC-expanded label sequence, run the standard CTC Viterbi DP, traceback to per-token frames.
5. **Quantise** with `cohere-quantize` (the same llama.cpp-style quantiser used for the other GGUFs in this fork). Q4_K alignment is byte-identical to F16 on the verification clip.

## Related

- **C++ runtime:** [`CrispStrobe/CrispASR@parakeet`](https://github.com/CrispStrobe/CrispASR/tree/parakeet)
- **Original NVIDIA model:** [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) — the .nemo tarball this aligner was extracted from
- **Sister GGUF release (canary main model):** [`cstr/canary-1b-v2-GGUF`](https://huggingface.co/cstr/canary-1b-v2-GGUF) — the encoder-decoder transcription + translation model
- **Sister parakeet release:** [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF) — the same encoder family with a TDT decoder for free word timestamps
- **NeMo Forced Aligner (the official tool that uses this same auxiliary model):** [NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo)
- **Canary technical report:** [arXiv:2509.14128](https://arxiv.org/abs/2509.14128)

## License

**CC-BY-4.0**, inherited from `nvidia/canary-1b-v2`. Use of these GGUF files **must** comply with the CC-BY-4.0 license including attribution to NVIDIA's NeMo team. See [the license](https://creativecommons.org/licenses/by/4.0/) for full terms.

The conversion + runtime code is MIT-licensed (matching the crispasr base), but the model weights themselves are NVIDIA's CC-BY-4.0 work.
