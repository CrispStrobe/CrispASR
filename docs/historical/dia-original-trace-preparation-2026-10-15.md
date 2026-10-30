# Dia actual original trace and sampled-decision fork — preparation

Superseded by the [independently audited result](dia-original-trace-results-2026-10-16.md).
The text below preserves the preparation checkpoint.

Candidate9ed6cff6 runs on the owned branch diag/dia-original-trace-20261015.
Heavy CPU [38818571466](https://github.com/CrispStrobe/CrispASR/actions/runs/38818571466)
and lint [38818570959](https://github.com/CrispStrobe/CrispASR/actions/runs/38818570959)
are dispatched; this document records preparation, not accepted hosted results.
The owned worktree is ${CRISPASR_WORKTREES}/crispasr-dia-original-trace-20261015.
Local scratch/proof is under ${CRISPASR_STORAGE}/crispasr/dia-original-trace-20261015/local-v1.

The completed terminal intervention separates inherited hidden-state error,
final normalization and output-head effects, with CFG F64 giving identical
results. It does not establish a new runtime defect or repair native speech.
See [accepted terminal evidence](dia-terminal-replay-results-2026-10-15.md).
The next experiment fills missing actual full-model self-Q provenance and tests
the downstream sensitivity to the first differing native sampled decision.

The worker pins reference8d0ad886 and original4a9e29b1, physical caba weights,
Torch2.6 CPU/four threads, seed42 and the original3072-token recipe. It runs
OFF/ON/OFF in separate processes. ON observes all18 self Q/K/V/attention,
pre-RoPE Q/K/V projections, all16 boundaries per decoder layer and actual
cross-cache valid39-position prefixes. No observer substitutes neural arithmetic.
Every common tensor, complete trajectory, PCM and RNG witness must remain
byte-exact. Original inputs/codes must reproduce the canonical archive, and
original TTS→ASR must have WER0 before the fork runs.

The fourth process calls the original sampler, then clones its step978 result
and changes only codebook2 value877 to904. It checks unchanged RNG consumption,
the complete earlier input/logit prefix, the single changed input979 entry and
the safe963-frame undelayed code prefix. Complete subsequent codes/audio are
retained; fork ASR is reported whether speech succeeds or fails. This tests one
decision's sensitivity in the original, without establishing a unique native
cause or native speech acceptance.

Local preflight passes actual pinned Attention OFF/ON/OFF, cache return/data
identity, actual sampler one-entry fork/RNG invariance and6 rejecting controls.
An expanded test exercises the pinned reference's function-local hook import
through a tiny original neural model, capturing projections, caches and decoder
boundaries simultaneously. All common generation arrays remain byte-exact.
Its DAC is a stub; the hosted run requires the actual original pinned DAC and
positive speech. All4 unskipped no-secrets checks pass before the source push.

Hosted payloads retain complete NPY manifests, lossless exact-array storage
references, source/method snapshots, logs and recognizer libraries. Accepting
findings requires independent full artifact/hash/control validation. No native
arithmetic/default, GPU/Q4/performance or release claim follows from dispatch.
