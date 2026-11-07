# Haas-ja-v1.0 Japanese transcription

Haas-ja-v1.0 is a Japanese speech recognition model from
[third-intelligence](https://huggingface.co/third-intelligence/Haas-ja-v1.0).
CrispASR runs its FastConformer encoder, Kimi Delta Attention predictor and
TDT decoder through the Parakeet implementation. Haas support shipped in
v0.8.50; use a current release for the latest fixes.

## Transcribe audio

```sh
crispasr --backend haas -m auto -f japanese.wav
```

On first use, `-m auto` resolves the registered default and downloads it to the
CrispASR model cache. `haas-ja` and `haas-asr` are also accepted as explicit
backend aliases. To use a model you have already downloaded, pass its path:

```sh
crispasr --backend haas -m /path/to/haas-ja-v1.0-q4_k.gguf -f japanese.wav
```

The [CrispASR GGUF repository](https://huggingface.co/cstr/Haas-ja-v1.0-GGUF)
provides F16 (~1.18 GB), Q8_0 (~626 MB) and Q4_K (~368 MB) files. The model
registry lists Q8_0 first for automatic selection. Sizes are approximate;
choose a specific downloaded file when a particular precision is required.
The model is Apache-2.0; see its upstream model card for its intended use.

For applications using the session C ABI, `haas`, `haas-ja` and `haas-asr`
are supported explicit aliases. Automatic backend detection uses GGUF
architecture metadata; the alias entrypoint and registry are covered by
`tests/test_haas_dispatch.py` and `tests/test-registry.cpp`. These are dispatch
tests, not a general accuracy benchmark for every audio domain or quantization.
