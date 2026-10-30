# Dia DAC companion conversion

`models/convert-dia-dac-to-gguf.py` reproduces the tested Dia DAC precision
candidate from the original checkpoint. Its `output-f32` policy uses original
F32 values for `decoder.snake1.alpha`, `decoder.conv2.weight` and
`decoder.conv2.bias`, preserving all 131 unrelated tensor payloads and types.
The bias is already F32 and byte-identical; the other two selected payloads
change. The complete GGUF grows from 108,695,872 to 108,697,184 bytes:
**1,312 additional bytes**, including metadata and alignment.

Use the published `dac-44khz.gguf` from
[`cstr/dia-1.6b-GGUF` at the validated revision](https://huggingface.co/cstr/dia-1.6b-GGUF/tree/3233fbcb32be47761d2e736857b6d1a075b9ba7e).
The converter checks its SHA256 before loading model dependencies:

```text
6f62c6dbb26fd69f0634db3c7f044464cfbd9a5b8806eb621f8be7034d869a3e
```

Run in a separate reference environment with `descript-audio-codec==1.0.0`,
Torch, its matching torchaudio/torchvision packages, NumPy and GGUF installed.
The hosted validation uses Torch/torchaudio 2.6.0 and torchvision 0.21.0.
The original DAC 44 kHz / 8 kbps / 0.0.1 checkpoint is downloaded by DAC,
or supplied through `--checkpoint`. Its required SHA256 is:

```text
a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa
```

```bash
python3 models/convert-dia-dac-to-gguf.py \
  --template /path/to/dac-44khz.gguf \
  --output /path/to/dac-output-f32.gguf \
  --precision output-f32 \
  --receipt /path/to/dac-output-f32-audit.json
```

Output and receipt paths must be new and distinct; their parent directories
must exist. The receipt records all 134 physical tensor hashes/types, original
checkpoint and template hashes, output hash, and exact byte change. Unknown
tensor names, unsupported types, inconsistent shapes, nonfinite values and
overwrites fail the export. Original F32 coefficients come from the original
module parameters after Torch removes weight normalization; converting the
rounded published F16 tensor to F32 cannot recover them.

The validated `output-f32` GGUF SHA256 is:

```text
6db51da2333868918b6d476008d92036a53400bc816ff7e1d2b1e973499e4135
```

`rvq-output-f32` and `f32` remain additional numerical-control policies.
The model registry still downloads the existing published companion. To inspect
a converted companion, select its path explicitly with `--codec-model`.

## Validation and limits

[The standalone hosted proof](https://github.com/CrispStrobe/CrispASR/actions/runs/38306172119)
reproduces the candidate byte for byte and passes strict comparisons against
both retained originals: **48/48 startup stages** and **40/40 full-utterance
stages**, including cosine and magnitude. The
[complete receipt](historical/dia-dac-standalone-export-2026-10-12.json) contains every
stage result, physical audit and retained-file hash.

Those comparisons replay original decoder inputs and original DAC codes.
Native free-running short speech still fails ASR, and Q4_K neural parity remains
failed. This converter does not establish speech quality, sampling parity or
physical GPU acceptance. The original reference and acceptance thresholds are
unchanged.

The `Dia DAC conversion` workflow runs on converter/helper/test changes. It
executes seven tiny actual-original-module/CLI contracts and the real pinned
conversion, requiring the exact validated GGUF bytes and all 134 tensor-audit
rows. Its worker is `tools/ci-heavy/dia_dac_converter_validate.py`; results and
logs are retained as an Actions artifact.

The complete converter and related original/negative proof artifacts are retained in [pinned public proof archives](dia-proof-archives.md), with a verified anonymous retrieval helper.
