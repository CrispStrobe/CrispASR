---
license: apache-2.0
base_model: IndexTeam/Index-Echo-S2TT-2B
pipeline_tag: automatic-speech-recognition
library_name: ggml
language:
  - zh
  - en
  - ja
  - es
tags:
  - gguf
  - crispasr
  - speech-translation
---

# Index-Echo S2TT 2B — GGUF for CrispASR

Conversion of [IndexTeam/Index-Echo-S2TT-2B](https://huggingface.co/IndexTeam/Index-Echo-S2TT-2B)
for the `index-echo` backend in [CrispASR](https://github.com/CrispStrobe/CrispASR).
Weights have passed the independent CPU/CUDA checks below. The backend is
available on CrispASR main, including C ABI loading, companion downloads and
a pinned nightly regression.

The released model transcribes Chinese and translates each subtitle into
English, Japanese or Spanish. CrispASR preserves the three-line generation
recipe and returns bilingual cues with timestamps and punctuation. English
JFK transcription/English translation also matches the released Python model
in the CPU fixtures; this is a port-parity result, not a language-wide accuracy
claim. The 9B checkpoint is not included.

## Files and companions

| Precision | Tower + connector | Qwen3.5 decoder | Combined |
|---|---|---|---|
| F16 | `index-echo-2b-f16.gguf` (1,313,848,896 bytes) | `index-echo-2b-decoder-f16.gguf` (3,897,387,648 bytes) | 4.853 GiB |
| Q8_0 | `index-echo-2b-q8_0.gguf` (702,951,040 bytes) | `index-echo-2b-decoder-q8_0.gguf` (2,076,674,688 bytes) | 2.589 GiB |

Both files of the chosen precision must be in the same directory. The tower's
`index_echo.decoder_file` metadata names its decoder; point CrispASR at the
tower, not the decoder. The decoder omits the absent MTP branch and contains
exactly the 24 text blocks used by the released speech inference path. Vision
weights are omitted.

For speech-boundary windows, also put `ggml-silero-v6.3.0.bin` from
[ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad) beside them,
or pass `--vad-model`. Silero uses its separate MIT license. Without Silero,
CrispASR uses bounded 60-second windows. The source recipe keeps context from
up to five prior windows and resets history for each input file.

## Use

Build CrispASR with the Index-Echo backend, then download a matching pair:

```bash
hf download cstr/index-echo-2b-GGUF \
  index-echo-2b-q8_0.gguf index-echo-2b-decoder-q8_0.gguf \
  --local-dir models/index-echo
hf download ggml-org/whisper-vad ggml-silero-v6.3.0.bin \
  --local-dir models/index-echo
./build/bin/crispasr --backend index-echo \
  -m models/index-echo/index-echo-2b-q8_0.gguf \
  -f audio.wav --target-lang en -osrt
```

`--target-lang en|ja|es` selects the translation. `--prompt` accepts glossary
entries such as `name:translation`; `--ask` overrides the task instruction.
Greedy decoding is the default. Positive temperature enables sampling;
`--seed` and `--max-new-tokens` are forwarded. Custom instructions need the
three-line timestamp/transcript/translation format for parsed bilingual cues;
unformatted custom output is returned as a whole-window text segment.
Word timestamps, beam search and live streaming are not exposed by this recipe.

## Validation and quantization

[CPU validation run 36905507951](https://github.com/CrispStrobe/CrispASR/actions/runs/36905507951)
passes F16 and Q8_0 against independent CPU captures (F32 tower/connector, BF16 decoder) from the released
Python inference class: English JFK, Chinese and a partial-hop JFK clip.
Checks cover mel/convolutions, all 32 audio blocks, the connector, prompt IDs,
all 24 decoder blocks, logits and 16 cached teacher-forced steps. The actual
Python Session/C ABI loads a filename without a backend hint and reproduces
all decoded text and centisecond timestamps exactly.

| Precision | Worst stage-row cosine | Largest stage relative L2 |
|---|---|---|
| F16 | 0.998464 | 1.415% |
| Q8_0 | 0.995173 | 3.561% |

A separately forced F32 decoder diagnostic now proves higher-precision math:
[run 36915483750](https://github.com/CrispStrobe/CrispASR/actions/runs/36915483750)
passes **every F16** stage, magnitude, cache, direct-output and full-file gate,
including English/Japanese/Spanish and two-window context. Minimum cosine is
0.999992; maximum stage relative L2 is 0.239%. Q8 passes stage/cache/direct
checks against that reference too (0.997150 minimum, 3.094% maximum relative L2).
The overall diagnostic run is red because Q8 has the documented full-file
precision differences; this is not an all-cohort exact-parity claim.

Cosine alone is scale-blind, so magnitude is gated separately. F16 stages
require cosine >= .999 (cached logits >= .998) and relative L2 <= 2%;
quantized learned stages require cosine >= .99 and relative L2 <= 5%.
Frontend stages retain strict gates. Prompt IDs, cached greedy IDs and full
direct-window decoded cues are additional mandatory checks. Full-file quality
acceptance separately requires the complete text of an independent source
precision variant, with exact F16 timestamps and at most 20ms Q8 timestamp
drift. No arbitrary text edits are accepted.

Plain Q4_K is rejected: worst cosine 0.781530, relative L2 up to 23.349%, and
changed translations/punctuation. Selective Q4 is also rejected: it fails stage/output checks and saves only
2.8% versus Q8. Neither Q4 recipe is a recommended download.

Full-file checks exposed and fixed a Silero waveform-context discrepancy.
With that fix, F16 full-file English/Japanese/Spanish and two-window outputs
match the separate explicitly F32 decoder source run. The released BF16
decoder differs by 20ms on three Japanese boundaries and chooses a Spanish
synonym (`creamos` versus F32/native F16 `pensemos`). Q8 also shifts one English
boundary by 20ms. Exact mixed-precision parity is therefore claimed only for
the three direct-window fixtures, not all full-file multilingual outputs.

A same-host four-core ARM Neoverse-N2 profile shows Q8 warm speedups of
1.33–1.46× over the actual F32/BF16 Python blueprint, with peak RSS 4.03 versus
7.95GiB. A two-Tesla-T4 CUDA run passes stage/cache checks for both cohorts;
GPU SRTs match native CPU controls byte for byte. Q8 inference is 4.5–5.3×
realtime on those three short clips, excluding load. See CrispASR PERFORMANCE.md
and its JSON receipts for methodology and limitations.

## Reproducibility and license

Source revision: `5d98a34d9685869b11e9c94d01dee22a8b8e53b5`.
The tower converter is `models/convert-index-echo-to-gguf.py`; the standard
Qwen3.5 decoder converter is llama.cpp commit
`42d958167a748f2c04b1f888e84e7a58f609ddcb`, with `--outtype f16 --no-mtp`.
Quantization uses `crispasr-quantize`; frontend constants and small affines
retain floating-point precision. Stage fixtures are in
[cstr/crispasr-regression-fixtures](https://huggingface.co/cstr/crispasr-regression-fixtures)
at `374efe4d7f5c4ff5dce32deffb14e23865fa8493`: released mixed-precision
captures under `index-echo-2b/`, separately forced F32 decoder captures under
`index-echo-2b-f32/`. The nightly uses the F32 fixture; the original golden
is preserved. Both source full-pipeline captures and effective-dtype audit are
included.

The model weights are Apache-2.0; the source license is included as `LICENSE`.
CrispASR runtime code has its own repository license. This conversion is not
an official IndexTeam release.
