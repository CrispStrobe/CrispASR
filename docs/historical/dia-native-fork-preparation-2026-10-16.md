# Dia reciprocal native sampled-decision fork — preparation

Follow-up: the [reciprocal native experiment](dia-native-fork-results-2026-10-16.md)
is complete and independently audited. The one-token restoration recovers WER0
and the original input/code trajectory for this prompt/seed. Earlier pending
statements below describe the historical preparation checkpoint. No production
speech/default/GPU/Q4/performance acceptance follows.

Current attempt: corrected source5417ed6d, heavy CPU
[38855559740](https://github.com/CrispStrobe/CrispASR/actions/runs/38855559740)
and lint [38855559850](https://github.com/CrispStrobe/CrispASR/actions/runs/38855559850).
The source fetch now precedes its tiny guard. The actual fresh shallow-clone
test passes; all experiment/runtime/model/recipe/gates remain unchanged.
The independent auditor now requires the expected full worker SHA and run ID
as explicit arguments, then checks every official record/source against them.
Full hosted quality and artifact audit remain pending.

Update: first heavy38851402907 failed before model/build at its tiny guard:
the shallow checkout had not fetched the pinned source yet. The bootstrap now
fetches before checking source. A fresh shallow repository reproduces the
missing-source failure before fetch and passes the actual sampler guard after
fetch; proof is under the owned storage root's local-v1/shallow-bootstrap-v1.
The failed artifact and logs remain under failed-38851402907-v1. Lint38851403072
passes all10 jobs. Relaunch is being prepared with the same runtime, model,
sampler, recipe and acceptance gates. The earlier checkpoints below remain
historical preparation records; no hosted speech outcome is accepted yet.

Original attempt: candidateeae36591 on diag/dia-native-fork-20261016. Heavy CPU
[38851402907](https://github.com/CrispStrobe/CrispASR/actions/runs/38851402907)
and lint [38851403072](https://github.com/CrispStrobe/CrispASR/actions/runs/38851403072)
are dispatched; this checkpoint is not a hosted result. The owned worktree is
${CRISPASR_WORKTREES}/crispasr-dia-native-fork-20261016, with local proof/scratch under
${CRISPASR_STORAGE}/crispasr/dia-native-fork-20261016/local-v1. The detached pinned
syntax-check worktree ${CRISPASR_WORKTREES}/crispasr-dia-native-fork-runtime-20261016
contains only this experiment's scratch source/header patch.

The [accepted original-only fork](dia-original-trace-results-2026-10-16.md)
shows that replacing sampled877→904 at978/codebook2 destroys speech for this
prompt/seed in the original itself. The reciprocal test asks whether changing
only native904→877 restores speech in the fixed native trajectory. Either
outcome is retained; a successful token override is not a production repair.

The worker uses immutable8d0ad886/C36, the same physically verified full F32
neural weights, F32 cross caches, portable RNG and source-RoPE-off configuration
as the accepted native boundary. Prompt, seed42,3072-token capacity, temperature,
top_p/top_k and CFG remain unchanged. It runs the actual native C ABI and DAC in
four separate processes: pristine, observer-only OFF, identity904, restore877.
Pristine generation occurs before the scratch patch and rebuild. Original
sampling/neural code remains unchanged; the patch observes or substitutes only
the completed nine-head sample at the fixed boundary, then saves raw code grids.

Pristine/OFF/identity complete inputs, logits, position978 captures and native
PCM must be byte-exact. OFF/identity full codes, witnesses and final serialized
MT state must also be exact. Restore must preserve all earlier inputs/logits,
all position978 stages and the safe963-frame code prefix; input979 differs in
exactly one entry and matches the original row. Typed and serialized RNG states
before/after the substitution must be identical, with the same pre-fork state
in all instrumented arms. A separate immutable accepted native archive must
reproduce the observed prefix logits/stages; no prefix import or teacher forcing.

Actual native audio is recognized for all four arms. Instrumented arms' complete
code grids are also decoded with the actual pinned original DAC and recognized.
Native8d0ad886's historical stopping/reversal conventions remain fixed; no
unreported output-protocol repair is mixed into the experiment. Original
canonical codes/PCM independently validate DAC reconstruction with relative
L2 below1e-4; both original ASR positives require WER0. All ASR results use the
same final recognizer library and pinned model. Native/fork WER has no relaxed
or newly selected pass threshold: report it honestly.

Local preflight compiles the actual pinned sampler and exponential generator,
then checks OFF/identity/restore tokens, full MT-state equality and9 rejecting
controls. Repeated patching also rejects. The complete patched runtime passes
C++ syntax checking against exact8d0ad886/C36 headers. Clang-format18, Python
compilation and all4 unskipped no-secrets checks pass. These are tiny/control
checks; full model/build/codec/ASR work remains on GH CPU. No local full neural
model is loaded. Source/ELF/library/export receipts, complete arrays and original
DAC source/provenance are retained for independent artifact audit.

No production runtime/default, model pin, sampler filter, seed or acceptance
threshold changes. Native speech, Q4/GPU/performance and release promotion
remain unaccepted. The full worker method is retained on the candidate branch
at tools/ci-heavy/dia-native-fork/README.md until hosted evidence is audited.

## Independent auditor preparation checkpoint

Auditorbc59663e is pushed on the same owned feature branch. The worker remains
pinned toeae36591; its queued heavy/lint runs are unchanged. The auditor verifies
complete official ZIP/CRC/SHA/file closure, every payload and immutable method,
regenerates the exact patch and freshly compiles/runs the actual sampler guard.
It checks full control/prefix/RNG/code/stage invariants, pristine/patched ELF and
unchanged ggml libraries, physical export ledgers, canonical/accepted native
archives and original DAC source provenance. All ASR edit distances, audio
durations/RMS and audio/library/model hashes are recalculated independently.

Calibration passes23 rejecting controls, including changed prefixes, extra token
changes, bad serialized MT states, false ASR scores, audio scale, nonfinite PCM
and unsafe paths. Its fresh actual pinned sampler/patch guard also passes.
Local receipt/logs remain under
${CRISPASR_STORAGE}/crispasr/dia-native-fork-20261016/local-v1/auditor-v2.
All4 unskipped no-secrets checks pass before the auditor push. Scratch audit runs
use unique directories so a failed audit's evidence survives later retries.

As of2026-10-16T04:45Z the heavy worker and lint are queued. Auditor calibration
does not establish successful validation of the future hosted artifact or any
speech outcome. Full artifact audit and integration remain pending.
