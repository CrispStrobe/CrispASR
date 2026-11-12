# Offline CPU speech-quality evaluation

The saved ten-second JFK proxy exposes a quality failure in the original graph,
although native/reference intermediate parity passed. The independent
[saved-waveform receipt](../../../docs/historical/reuse-saved-waveform-quality-20261112.json)
extracts each retained reference window without running a model. For the same
5–7.5 s speech region, the window starting at 2.5 s gives SI-SDR 6.39 dB, while
that starting at 5 s gives −9.03 dB. Their waveform disagreement has RMS 0.699
relative to the clean region. Window 2 as a whole scores −6.14 dB SI-SDR versus
3.75 dB for its noisy input. These observations localize sensitivity to the
processing context; they do not establish its cause.

Aggregate ESTOI improves from 0.456 to 0.543 while SI-SDR falls from 5.00 to
3.48 dB. A STOI improvement alone cannot establish acceptable enhancement.
All these results concern one historical speech proxy and one synthetic noise
realization. No listening-quality or generalization claim follows.

Recompute the saved-waveform analysis with Python, NumPy, SciPy, SoundFile,
pystoi and gguf:

```
python tools/kaggle/reuse-quality-cpu-20261112/audit_saved_waveforms.py \
  --input-dir "$REUSE_TEN_SECOND_ARTIFACTS" --output-dir "$REUSE_QUALITY_OUTPUT"
```

`quality_cpu.py` prepares the next, broader evaluation: LibriSpeech test-clean,
two female and two male speakers, one explicit five-second crop per speaker,
with clean speech, white noise at 0/10 dB, and synthetic pink noise at 5 dB.
Speaker and utterance selection is deterministic and independent of the model
output. The job retains clean/noisy/enhanced FLOAT WAVs and hashes, corpus
metadata/license text, model/source hashes, environment versions, and per-case
receipts. It reports SNR, SI-SDR, projection gain, RMS ratio and spectral
convergence. If pystoi exists on the worker it also reports STOI/ESTOI; otherwise
these can be recomputed from the retained WAVs after download. The original
recording is read speech, not a studio-quality or noiseless reference; noise is
synthetic, not a real noise corpus. This bounded sample cannot estimate a
population-level quality score.

The original graph runs with the previously audited CPU non-fused Mamba adapter.
Each case records shape, finite checks, norm, maximum magnitude and SHA-256 for
all 71 comparison-stage tensors plus raw input. These are fingerprints, not
comparisons against native execution. Existing 142-row gates remain the
reference-input parity evidence; this job establishes no additional native,
fused CUDA or GPU-backend parity.

The single uploaded `code_file` contains exact compressed, SHA-verified copies
of the CPU adapter and progress harness. The model and corpus arrive through
`dataset_sources`; the job performs no clone, pip install, download or upload.
No secrets dataset is attached. Metadata explicitly disables GPU and internet.
Dependencies, metric controls, corpus selection and every pinned model/source
hash are checked before model initialization. The job checkpoints each case to
`/kaggle/working`, so completed cases survive later failures; the two-hour
budget is checked between cases. CPU jobs have no live external progress
mirror, so retrieve terminal logs/output before any later kernel version.

A local dependency/metric preflight is available without model inference:

```
REUSE_QUALITY_OUT="$REUSE_QUALITY_OUTPUT" \
REUSE_QUALITY_TEMP="$REUSE_QUALITY_TEMP" \
python tools/kaggle/reuse-quality-cpu-20261112/quality_cpu.py --preflight-only
```

`--data-preflight-only` additionally checks mounted model/corpus hashes and
deterministic corpus selection, then exits before model initialization.
`REUSE_QUALITY_INPUT` may point to a local fixture mount for that read-only check.

This preparation is not a completed broader evaluation. Read the submitted
kernel's own `quality-report.json`, its exact script hash, all case receipts and
terminal status before reporting results.

LibriSpeech attribution: Panayotov et al., [OpenSLR 12](https://www.openslr.org/12),
CC BY 4.0. The attached Kaggle mirror is `ayoubakanoun/librispeechtest`; retain
its mounted license and source-file hashes. The four selected FLAC SHA-256 values
are pinned from the independently downloaded official OpenSLR archive (verified
MD5 `32fa31d27d2e1cad72775fee3f4849a9`); all four must match on the Kaggle mount
before the model is initialized. The frozen selection is 121-121726-0000,
237-126133-0000, 61-70968-0005 and 260-123286-0000.
