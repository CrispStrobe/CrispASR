# Ten-second corpus reference chunking

CPU kernel `crispasr-reuse-corpus-ten-second-20261113` version 1 completed all
four speakers in 1,788 seconds. Three focused functional controls pass locally.
Independent waveform checks reproduce every reported SNR/SI-SDR and replay
the stored windows into the exact saved chunked outputs.

Whole/chunked output SNRs are 12.799/12.748 dB (61), 12.698/12.368 dB (121),
15.670/15.346 dB (237), and 14.797/14.744 dB (260), from 5 dB inputs.
Whole-versus-chunked relRMS differs by 2.08–4.53%; norm ratios are
0.9790–0.9915. These context differences remain visible despite high cosine.
STOI is unavailable in this offline image. Full numbers are in the
[result receipt](../../../docs/historical/reuse-corpus-ten-second-20261113.json).

Original-only CPU evaluation for four real LibriSpeech test-clean utterances,
one per existing speaker. Each is at least ten seconds; the first ten seconds
are used without padding or repeating audio. A single seeded white-noise
mixture at 5 dB SNR is generated for the complete input before any windows.

Compare a fresh original whole-input forward with three fresh original
five-second forwards at starts 0, 2.5 and 5 seconds, stitched by the existing
reference chunking helper. Model equations, frontend and frame suppression
are unchanged. Retain clean/noisy/whole/chunked and per-window WAVs, quality
metrics, scale-sensitive waveform differences, and compressed actual arrays
at eight selected frontend/encoder/final-block/decoder boundaries for each
forward. These selected arrays are not the complete 71-stage native test.
The separate `crispasr-diff` corpus job supplies native stage comparisons.

Inputs use the existing model and LibriSpeech Kaggle datasets; no uploads,
internet, GPU or CI are needed. LibriSpeech is OpenSLR12, CC BY4.0:
https://www.openslr.org/12. Source files are pinned to the official archive.

```sh
python tests/test_reuse_corpus_ten_second.py
python tools/kaggle/reuse-corpus-ten-second-20261113/prepare.py <staging>/quality.py
```

Copy the adjacent metadata to staging and submit with the usual kernel tool.
The generated source embeds the already exercised CPU adapter and chunking
helpers. This evaluation measures context/chunking differences on four inputs;
it does not establish native parity, broad quality or product support.
