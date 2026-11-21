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

## Long audio and gap-fill

Haas-ja uses the same Japanese long-audio path as parakeet-ja: VAD slices
capped at 12 s, each decoded in one pass. The gap-fill second pass, which
re-transcribes spans a slice left empty, is **off by default** for Haas-ja.
That pass exists because parakeet-ja's encoder blanks utterances inside a
slice (#89). Haas-ja does not: on the #89 podcast it reaches 97 % content
recall without gap-fill, and the pass adds nothing there. On noisy audio such
as a game stream, the pass only re-transcribed pauses and sound effects into
short, wrong subtitles (#605). Set `CRISPASR_GAP_FILL=1` to run it anyway; it
then keeps only recoveries that clear the confidence gate
(`CRISPASR_GAP_FILL_MIN_CONF`, default 0.85).
