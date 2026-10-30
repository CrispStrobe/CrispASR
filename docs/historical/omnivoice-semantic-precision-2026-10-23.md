# OmniVoice Metal semantic precision — 2026-10-23

Historical experiment below: subsequent listening approval and
[Metal default promotion](omnivoice-metal-defaults-2026-10-23.md) update the
current eligibility/defaults. The original raw measurements are retained.

The station CPU/GPU code difference in the [encoder report](omnivoice-encoder-2026-10-23.md)
is reproducible and can be removed on the tested M1 by requesting F32 matrix
multiplication for semantic encoding. The new independent switch is
`CRISPASR_OMNIVOICE_ENCODER_SEMANTIC_F32=1`. It is off by default, as are GPU
hybrid placement, graph quantization and fused HuBERT. CPU defaults are unchanged.
[Raw evidence](omnivoice-semantic-precision-2026-10-23.json) retains both
unhinted and corrected paths, intermediate norms, hashes and individual timings.

## What changed the audio

Station's first changed reference code is at zero-based quantizer stage3,
frame55. CPU chooses centroid226; unhinted Metal chooses262. CPU's two best
scores are22.13980103 and22.13928986, separated by only0.00051117. The GPU
pipeline instead gives262 the higher score. This first choice changes the
residual, producing three additional changes at later stages4,6,7 of that frame.
The generator receives different reference codes and consequently produces
different target codes and PCM, despite identical target text and seed.
The correct ASR transcripts did not establish voice/prosody parity.

All three semantic boundaries can contribute rounding drift. The isolation
probe runs all eight CPU/GPU combinations of frontend, HuBERT transformer and
semantic bridge, keeping acoustic encoding and FC/RVQ fixed on CPU. Some
unhinted mixed combinations happen to preserve station codes, while others
change four. Selecting a CPU stage solely because one combination passes could
therefore depend on cancellation. The precision option addresses matrix
multiplication throughout the semantic graphs instead.

## First numerical divergence and precision control

The public `tools/ci-heavy/omnivoice_encoder_diff.cpp` includes the actual runtime.
It returns genuine truncated frontend graph outputs, avoiding scratch snapshots.
The first convolution already has tiny CPU/GPU reduction-order differences;
the second convolution's matrix product introduces a much larger discrepancy:

| Station boundary | Unhinted max absolute error | F32-hinted error |
|---|---:|---:|
| First convolution matrix output |0.000000358|0.000000358|
| Second convolution matrix output |0.012154|0.0000458|
| Whole semantic output |0.0035073|0.00000787|

Whole semantic cosine improves from0.9999997663 to0.999999999999; CPU/corrected
norms are74.1202631/74.1202536. The precision-hinted and original CPU stage arrays
are byte-identical on station. Float CPU/GPU arrays remain slightly different;
all station reference codes now match.

The pinned ggml fork already documents and implements the mechanism: on M1–M4,
its ordinary tiled Metal matrix path can reduce operand precision even when
operands are F32. `GGML_PREC_F32` selects its existing precise matrix path.
The new runtime option tags semantic graph matrix multiplications, including
convolution lowering, before allocation/compute. It covers staged/fused HuBERT
and the semantic bridge. Graph FC/RVQ already requests F32 accumulation.
There is no ggml bump, modified weight, epsilon/tie bias, resampling or sampler
change. Reference-code cache identity includes the precision setting.

## Broader parity and timing

Seven synthetic references cover three existing Piper clips and four locally
generated system voices: Samantha and Daniel in English, Anna in German and
Thomas in French. Encoded references span60–315frames, approximately2.4–12.6s
at25Hz. All56 isolated CPU/GPU combinations produce exact CPU reference codes
with precision enabled. The three original CPU arrays additionally match the
separately compiled frozen encoder. System-voice WAVs remain local diagnostic
fixtures; only generation recipes and hashes are recorded publicly.

Final paired measurements compare against the shipping CPU cache/pointwise
configuration. Both arms use CACHE=1 and FASTCONV=1; the candidate enables
HYBRID=1 and SEMANTIC_F32=1. The combined candidate also enables GRAPH/FUSED.
Four CPU threads, one warmup per arm, three measured calls per arm, reversed
interleaved order, disk reference cache off. All48 reference-code arrays match
the frozen baseline exactly, with actual GPU backend `MTL0` verified.

| Reference | CPU → precise hybrid | CPU → precise combined |
|---|---:|---:|
| window |1.569→0.962s|1.482→0.830s|
| station |1.419→0.943s|1.729→0.872s|
| fox |1.804→1.036s|1.595→0.882s|

This is about34–50% less wall time for fresh encoding in these paired samples.
Process CPU medians fall from5.06–5.67s to2.22–2.43s. Desktop contention and
raw timing variation are retained. These are encoding-stage measurements,
not generation, cache-hit or cold-start gains. Precise GPU encoding is slower
than the earlier unhinted GPU path; that path fails the station exact-code gate.
No extra persistent weight copy is introduced by the precision flag; existing
hybrid weight/cache costs and additional backend scratch still apply.

## Full synthesis and limits

A full freshly linked Session/C ABI comparison uses the original Q8_0 main/F16
tokenizer pins, English target text, seed42 reset per call,32steps, duration6,
four CPU threads, guidance2/class temperature0/position temperature5, exact
CFG, split/resident/decomposed decoder and attention views. Reference caching
is off and normal preprocessing/cleanup remains on. The receipt records the
actual library hash, each output hash, exact-code/PCM comparisons and both
frozen ASR judges. Raw local diagnostic artifacts retain the explicit marking
responsibility attestation from the original audit.

All24 renders pass: eight target/reference cases each run on CPU, precise hybrid
and precise combined paths. All16 candidate/CPU pairs have byte-identical
generated codes and float PCM. Both judges score the seven other distinct
waveforms WER0. For the Anna German-reference/English-target case, both omit the
final word “home” (WER1/12) on the unchanged CPU baseline and identical GPU
outputs. This existing output limitation is retained, not hidden by a changed
threshold. Original fox/Linux failures remain open separately.

The local listening folder preserves the original CPU/unhinted-GPU WAV pairs
and adds two `GPU-precise.wav` copies. Those corrected WAVs are byte-identical
to their CPU counterparts after the same unity-gain24kHz PCM16 conversion.
They are clearly named AI-generated diagnostic artifacts, with no normalization
or trimming. Their filenames and source hashes are recorded in the receipt.

The normal marked CLI is also checked with precise hybrid, graph and fused
encoding. Its target code dump is captured before the subsequent required
spoken disclosure overwrites that diagnostic file; the final WAV retains the
disclosure and C2PA. This distinguishes target-code parity from intentional
marking/disclosure changes to the delivered waveform.

The experiment concerns one M1 and one tokenizer precision/artifact. Reference
languages do not certify multilingual generation, and these controls do not
close earlier fox/Linux cloning failures. No independent speaker-embedding
model or completed human listening verdict is claimed. Keep GPU placement
optional pending wider quality/device coverage; CPU remains the normal default.

## Running the corrected A/B

Use the [TTS guide](../tts.md#omnivoice-reproducible-parameters-and-m1-performance-switches)
for builds, model pins, license acceptance and ordinary marking. Add:

```sh
CRISPASR_OMNIVOICE_ENCODER_HYBRID=1 \
CRISPASR_OMNIVOICE_ENCODER_SEMANTIC_F32=1 \
  "$BUILD/bin/crispasr" ...
```

GRAPH=1 and FUSED=1 select the measured combined candidate. Explicit
SEMANTIC_F32=0 retains the unhinted numerical control. `--no-gpu` still prevents
GPU placement. Precision remains independent of placement/fusion and does not
change license acceptance or reference-voice consent requirements.

Compile `omnivoice_encoder_diff.cpp` with the same flags/libraries as the
[encoder benchmark](omnivoice-encoder-2026-10-23.md#reproducing-the-native-ab),
substituting its source and executable name:

```sh
CRISPASR_OMNIVOICE_ENCODER_SEMANTIC_F32=1 \
CRISPASR_OMNIVOICE_ENCODER_DIFF_NODES=1 \
  ./encoder-diff "$OV_CODEC" "$REF" "$OUT_DIR"
```

Run the identical command with SEMANTIC_F32=0 in a separate output directory.
The tool checks the CPU decomposition against actual `set_voice_prompt`,
rejects CPU fallback for its GPU copy, verifies every reported quantizer winner
against runtime RVQ, and saves stage arrays/codes/top-two scores. The node flag
adds ten genuine truncated frontend outputs per backend; omit it for the broader
pipeline sweep. Names fe/hb/sem0 mean CPU,1 means GPU. Compare both norms and
errors; high cosine alone did not prevent the station code flip.

The encoder benchmark adds `precise` and `precise-all` modes for paired timing
against the shipping CPU configuration. Its original `all` mode keeps precision
off and remains the five-experiment numerical control. Use the reverse-order
flag documented in the encoder report; retain warmups and per-call samples.
