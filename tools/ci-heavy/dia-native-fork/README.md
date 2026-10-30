# Reciprocal native sampled-decision experiment

This diagnostic asks whether replacing only native sampled904 with original877
at step978/codebook2 restores speech in the fixed native trajectory. The
original-only reciprocal experiment already established the other direction:
changing877→904 destroys original speech for this prompt/seed. Neither token
substitution is a production repair or justified sampling-policy change.

Runtime8d0ad886/C36, physically verified F32 weights8128132c and F32 cross caches
are fixed. Portable RNG remains enabled, original-source RoPE disabled. Prompt
is `[S1] We are committed to making it secure.`, seed42,3072-token capacity,
temperature1.2, top_p.95, top_k45, CFG3, four CPU threads. The worker never
instantiates the full original neural model. Full generation/builds run on GH;
local checks use tiny actual pinned sampler/header code only.

```sh
gh workflow run heavy-cpu.yml --ref <candidate-branch> \
  -f script=tools/ci-heavy/dia_native_fork.py -f timeout_minutes=300 \
  -f pip="numpy==1.26.4 torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 pydantic>=2.11.3 huggingface_hub==0.36.0 safetensors==0.5.3 descript-audio-codec==1.0.0 gguf"
```

Four separate native C ABI processes run in order:

1. Pristine runtime: no scratch fork patch; actual native DAC audio, all inputs,
   logits, samples and position978 stage captures are retained.
2. Observer-only OFF: patch records the actual completed nine-head sample,
   full serialized MT state and delayed/undelayed code grids, changing no token.
3. Identity904: the same assignment site writes back the actual sampled904.
4. Restore877: change only that entry after the actual unchanged sampler.

Pristine/OFF/identity inputs, full logits, stage captures and native PCM must
be byte-exact. OFF/identity complete code grids, samples and final RNG must be
exact. The fork's earlier inputs/logits and all position978 stages must remain
exact. Input979 must differ only by904→877 and match the original next row;
the first963 undelayed code frames must be exact. All three witnesses must
record the same pre-substitution RNG, and before/after RNG must be unchanged.
The unmodified native prefix must also reproduce the separately accepted native
raw logits and stages; no prefix import or teacher-forced input is allowed.

All native audio is recognized through the same pinned C ABI Nemotron model.
OFF/identity/restore code grids are additionally decoded with the actual pinned
original DAC, then recognized independently. Native8d0ad886's stopping/reversal
conventions remain fixed in every arm; the grids are exactly those passed to
its own native DAC. This does not silently repair or relabel the historical
output-protocol difference. The actual original canonical codes/PCM validate
original DAC relative L2 below1e-4 and both original ASR positives require WER0.
Native and fork ASR outcomes are retained without a quality pass threshold.

The tiny guard calls the actual pinned exponential generator and sampler,
then verifies OFF/identity/restore token and typed MT-state invariants. Nine
invalid/repeated/missing-sample controls reject; repeated source patching also
rejects. No neural arithmetic is copied or reimplemented in the helper. Source,
physical export receipts, pre/post libraries, logs, complete raw arrays and
all actual original DAC source/provenance are retained and hashed. Results
require a separate complete-artifact audit before interpretation or integration.

## Independent auditor

`audit.py` is prepared to validate the official complete artifact. It requires
`raw.zip`, `run.json`, `artifacts.json` and `lint.json` in a bundle directory;
these must describe the explicit expected successful immutable worker/run.
Pass their full SHA/run ID independently of the artifact receipt. It verifies ZIP CRC,
full archive/payload hashes and exact file closure before extracting/loading.
No full neural model runs locally.

```sh
python tools/ci-heavy/dia-native-fork/audit_selftest.py --repo /path/to/CrispASR \
  --scratch /path/to/fresh-local-audit-controls
python tools/ci-heavy/dia-native-fork/audit.py --bundle /path/to/bundle \
  --worker-source <full-worker-sha> --run-id <heavy-run-id> \
  --repo /path/to/CrispASR --original-repo /path/to/original-dia \
  --canonical /path/to/canonical/ref.gguf \
  --terminal-bundle /path/to/accepted-terminal-bundle \
  --output /path/to/fresh-audit.json
```

Use NumPy and gguf. The fresh sampler guard requires g++ and the pinned8d0ad886
git object. Reproduction verifies all retained methods against immutable git
objects, regenerates the exact scratch patch, freshly compiles/runs the actual
sampler guard, checks pristine/patched ELF and unchanged ggml libraries, full
control/prefix/RNG/code/capture invariants, physical export receipts, accepted
native/canonical archives and actual original DAC source provenance. ASR edit
distance, audio duration/RMS and audio/library/model hashes are recalculated
independently; either fork quality outcome is reported without a threshold.
The corrected calibration passes29 rejecting controls plus the actual sampler guard.
Preparation/calibration does not establish a successful full artifact audit.

Bootstrap correction: the first hosted run38851402907 failed before model/build
because its tiny guard attempted to read the pinned source before fetching it
into the shallow checkout. Fetch now precedes the guard. The failed raw artifact
is retained; a fresh shallow repository exercises the corrected order. The
runtime, model, sampler, recipe, controls and quality thresholds stay fixed.

Completed result: [audited reciprocal speech/trajectory recovery](../../../docs/historical/dia-native-fork-results-2026-10-16.md).
The auditor accepts silent capacity logs because that pinned runtime message is
verbosity-gated; any printed capacity must remain3072. Exact immutable worker/
runtime/export provenance remains required. Imports never write bytecode into
evidence. Both earlier failed audits are retained in the public result bundle.
