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

`quality_cpu.py` runs a bounded evaluation: LibriSpeech test-clean,
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
of the CPU adapter and progress harness. Their exact bytes also match current
source revision `69b11562aa7b3a174add9a6767098d4eeae680d8`, which is
recorded in each run report. The model and corpus arrive through
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

Version 1 completed and all 16 cases were independently verified. On the 12
synthetic-noise cases SI-SDR improved in every case; condition mean improvements
were 11.50 dB for white 0 dB, 6.74 dB for white 10 dB, and 9.64 dB for pink
5 dB. Clean inputs were also changed by enhancement. The [verified receipt](../../../docs/historical/reuse-quality-cpu-20261112.json)
contains per-case measurements and provenance. This is four read-speech
speakers with synthetic noise; it does not resolve the JFK-proxy result or
establish listening quality, native parity or product readiness.

LibriSpeech attribution: Panayotov et al., [OpenSLR 12](https://www.openslr.org/12),
CC BY 4.0. The attached Kaggle mirror is `ayoubakanoun/librispeechtest`; retain
its mounted license and source-file hashes. The four selected FLAC SHA-256 values
are pinned from the independently downloaded official OpenSLR archive (verified
MD5 `32fa31d27d2e1cad72775fee3f4849a9`); all four must match on the Kaggle mount
before the model is initialized. The frozen selection is 121-121726-0000,
237-126133-0000, 61-70968-0005 and 260-123286-0000.

Model-free controls (no model initialization or inference):

```
python tests/test_reuse_quality_cpu.py -v
```

These exercise deterministic selection and tamper rejection, exact noise SNR,
pink-noise determinism, stage shape/finite/completeness checks, FLOAT waveform
roundtrips, and interruption during aggregate checkpoint replacement.
All four selected files were also downloaded individually from the live Kaggle
mirror and independently matched the official SHA-256 values before submission.
The worker repeats these checks against its actual mounted bytes.

The saved-waveform auditor is import-safe, validates explicit shapes/rates and
matching raw input windows, and atomically replaces extracted WAVs and JSON.
Its optional prior receipt enforces the three complete input-file hashes.
Zero/constant references are rejected; zero-projection estimates have null
SI-SDR. Seeded STOI/ESTOI leaves the caller's NumPy RNG unchanged.
Model-free auditor controls: `python tests/test_reuse_saved_waveform_audit.py -v`.

The submitted Kaggle **version 1** runs script **v2**. Kaggle normalized the title
to actual slug `crispasr-re-use-quality-cpu-20261112`, which differs from the
metadata's requested slug. Use this actual slug for status, terminal logs and
output recovery; do not repush the running kernel.

After terminal completion, `verify_completed.py` independently validates all
16 case receipts and their aggregate, source/corpus pins, exact noise recipe,
FLOAT WAV/file hashes, raw/output tensor hashes and all 72 stage fingerprint
structures. It recomputes SNR/SI-SDR and seeded STOI/ESTOI and groups results by
speaker and noise condition. Original intermediate arrays are not retained;
their fingerprint structure and magnitude invariants cannot establish native
parity. Verification of artifacts does not impose a speech-quality threshold.

Save a normalized terminal receipt from the read-only Kaggle API and retrieved
uploaded source: `kernel_slug`, integer `version`, actual `status` and
`script_sha256`. The verifier requires the actual slug above, version 1,
terminal `COMPLETE`, and uploaded script SHA-256
`e5bb069a930745389e180d393dcbfd00d8903c8ec49ec3011505a1936c0e13e9`.
It also requires the independently verified official LibriSpeech root and
compares every clean crop against its pinned original FLAC:

```
python tools/kaggle/reuse-quality-cpu-20261112/verify_completed.py \
  --artifacts "$REUSE_QUALITY_ARTIFACTS" \
  --official-corpus "$REUSE_OFFICIAL_LIBRISPEECH" \
  --terminal-receipt "$REUSE_QUALITY_TERMINAL_RECEIPT" \
  --output "$REUSE_QUALITY_VERIFICATION"
```

Partial/error states, missing cases, tampered hashes or inconsistent metrics
produce a nonpassing verdict and nonzero exit; a stale passing verdict is
replaced on failure. NumPy, SciPy, SoundFile and pystoi are required; no model
runtime is imported. Model-free verifier controls:
`python tests/test_reuse_quality_verifier.py -v`.

A separate historical Whisper tiny.en CPU proxy compares the saved whole/tail
transcripts against ASR of the matching clean proxy, rather than a human
transcription. The [receipt](../../../docs/historical/reuse-whisper-tiny-asr-proxy-20261112.json)
records raw artifact hashes, sanitized stdout, exact token/error mappings and
binary/model/resolved-library hashes. Whole-clip disagreement changes from
4/22 words (18.18%) for noisy speech to 2/22 (9.09%) for enhanced speech; on the
last five seconds it changes from 1/15 (6.67%) to 2/15 (13.33%). These mixed
results are one recognizer's transcript agreement on one proxy. They establish
neither human-ground-truth WER nor intelligibility or broad quality acceptance.
The ASR binary reports historical version 0.8.43/revision 369029c9; those strings
do not independently establish a clean source identity.

`word_error_rate.py` provides reusable model-free tokenization and unit-cost
edit alignment for any caller-selected transcript reference. Normalization
uses Unicode NFKC, casefolding and letter/digit runs, with punctuation as word
boundaries; it preserves `f` in `F-not` and the difference between `american`
and `americans`. Empty references produce null WER with explicit edit counts.
For example, compare each speaker's clean-ASR transcript against each condition:

```
python tools/kaggle/reuse-quality-cpu-20261112/word_error_rate.py \
  --reference "$CLEAN_ASR_TRANSCRIPT" --hypothesis "$CONDITION_ASR_TRANSCRIPT"
python tests/test_reuse_word_error_rate.py -v
```

`asr_proxy_runner.py` is an optional sequential research runner for a completed
four-speaker/16-case output. It is prepared independently of the Kaggle script.
Supply an ASR binary/model and their expected SHA-256 values; the runner checks
those pins, records dynamically resolved library hashes, and verifies runtime
bytes again after completion. It accepts only complete predetermined cases
whose input/output WAV hashes agree with their receipts. Use the independent
artifact verifier first; this runner does not replace its stage/corpus checks.

```
python tools/kaggle/reuse-quality-cpu-20261112/asr_proxy_runner.py \
  --artifacts "$REUSE_QUALITY_ARTIFACTS" --binary "$ASR_BINARY" \
  --model "$WHISPER_TINY_EN_MODEL" --binary-sha256 "$ASR_BINARY_SHA256" \
  --model-sha256 "$ASR_MODEL_SHA256" --output-dir "$ASR_PROXY_OUTPUT" \
  --threads 2 --timeout 180
```

It first recognizes each clean speaker reference, then each case's noisy/clean
input and enhanced output, with explicit CPU/English/no-timestamp text-output
flags. Content-identical audio may reuse a completed transcript through an
explicit mapping. Each clean-condition input is always recognized afresh,
checked against the clean reference's audio, and required to produce the exact
same transcript bytes; a mismatch is reported as ASR nondeterminism. No clean
identity result is inferred from deduplication. Empty condition transcripts
remain measurable as deletions against the nonempty clean-ASR reference.

Each invocation records flags, timeout, exit code/status, input file/audio
hashes, raw stdout/stderr and transcript hashes. Public report commands use
portable aliases; raw log files may retain local execution paths. Each completed
case is checkpointed before the next starts. Later nonzero exit, timeout, missing
transcript or runtime change produces a failed report while preserving earlier
case/invocation receipts and transcript files; temporary process directories are
removed. Use a fresh output directory for a new run. This optional proxy measures
ASR disagreement with the same-speaker clean-ASR output, not human-reference WER,
intelligibility or product quality. No real ASR/model execution is part of its
model-free controls: `python tests/test_reuse_asr_proxy_runner.py -v`.
