# v0.8.49 — Audio8-TTS (ArkTTS), LID Standalone/Aggregation, and Qwen3 Context

## Speech synthesis

- Full backend port and integration for **Audio8-TTS (ArkTTS 0.6B)** (`Edge0/Audio8-TTS-Realistic-0.6b`):
  - Dual-autoregressive text-to-speech architecture coupling a 28-layer slow semantic transformer with an 8-layer fast acoustic subnetwork generating 10 codebooks per frame at 44.1 kHz.
  - Causal neural vocoder with Snake activations and ConvNeXt residual blocks.
  - Precision-preserving GGUF quantization: attention and FFN linear weights quantised to `Q4_K` / `Q8_0` while keeping 1D conv kernels ($K \le 16$) and discrete acoustic centroids ($k=1024$) at FP16 to guarantee bit-exact arithmetic without acoustic hash noise.
  - Verified 1.000000 cosine parity across all 7 stages on `crispasr-diff` against the PyTorch reference implementation.
  - Performance optimizations:
    - FASTCONV $K=1$ bypass eliminating im2col transposition and contiguous memory buffer copies across 12 audio-rate layers.
    - Dual-transformer in-place KV caching.
    - Chunked causal vocoder decoding with causal history overlap, bounding peak RAM to `<2.5 GB` for arbitrary audio lengths (3+ minutes) where monolithic graphs require `>160 GB`.
  - Official GGUF weights published to Hugging Face repository [`cstr/Audio8-TTS-Realistic-0.6b-GGUF`](https://huggingface.co/cstr/Audio8-TTS-Realistic-0.6b-GGUF) and registered in `crispasr_model_registry`.
  - Verified 0.0% Word Error Rate (WER) across TTS-to-ASR roundtrips.

## Recognition and CLI

- Support free-form transcription context in **Qwen3-ASR** (#559, thanks @pszemraj).
- Language Identification (LID) improvements (#560):
  - Standalone `--detect-language` (`-dl`) action performs language detection and exits cleanly without requiring an ASR model to be loaded or executed.
  - Dominant language detection across full recordings.
  - Automatic provider fallback to Whisper when dedicated LID models are not specified.

## Streaming and context

- Adaptive strided audio context gating for Hikari and Parakeet streaming inference.
- Attention mask caching in Moonshine to eliminate redundant mask construction during streaming chunk processing.
- Synchronize `audio_ctx` and `hikari_refresh_stride` across Go, Dart, Rust, Python, JavaScript, and C ABI.
- Formatted Dart session setter lookupFunction to pass strict CI format checks.

## Translation and tools

- Madlad translation: sharded safetensors conversion, low-RAM mmap scratch buffer allocation, and Kaggle quantizations for Madlad-7B and Madlad-10B.
- Cleaned and migrated historical maintenance handoffs and benchmark receipts into `docs/historical/`.
