Completed: heavy38870499727/lint38870502430 and independent655-payload audit
pass. [Final scoped result](dia-serial-head-results-2026-10-16.md) supersedes
pending-status statements in this retained preparation checkpoint.

# Dia all-step serial F32 head — preparation

Immutable candidate3c2fbb5a9c8c73225677ce6c9eabb6b31b21a92b is submitted in
[heavy CPU38870499727](https://github.com/CrispStrobe/CrispASR/actions/runs/38870499727)
and [lint38870502430](https://github.com/CrispStrobe/CrispASR/actions/runs/38870502430).
Heavy is executing; lint remains queued at this checkpoint. No full-generation
result is accepted.

The callback is a CPU node in the actual weighted ggml graph. Serial mode uses
native loaded F32 head weights and current hidden, with separately rounded
products and serial F32 addition at every step/head. Identity mode adds the same
node/dependencies but copies native logits; OFF leaves the graph unchanged.
The original matmul remains a dependency, deliberately paying both paths for
this correctness experiment. No performance/default/GPU claim follows.

Local compiled graph tests match every one of18,504 serial head values on both
captured hidden states, with identical one/four-thread results. Native output
reproduces all captured hosted native head logits; identity is exact. The tiny
fusion witness returns−2^-46 in native FMA mode and0 in serial mode. Three
invalid mode controls and repeated patch rejection pass. The entire patched
runtime passes C++ syntax against pinned8d0ad886/C36 headers. C++ format18,
Python compilation and all4 unskipped secret checks pass. Local retained proof:
${CRISPASR_STORAGE}/crispasr/dia-serial-head-20261016/local-v1/graph-v2.

The worker compiles pristine8d0ad886/C36, runs actual graph replay against the
public original/native hidden fixture, then uses physically verified F32 neural
weights and codec. Pristine generation precedes the scratch patch/rebuild.
Separate OFF, identity and serial C-ABI processes use the same portable RNG,
F32 cross caches, source RoPE off, seed42 and original3072-token recipe. No
external reference tensor or sampled token is injected. Numerical code has no
position/token-specific branch. All raw trajectories, logits, selected stage
captures, code grids, PCM, final MT state, sources and actual libraries remain
retained. Pristine/OFF/identity must be exact; the pristine baseline also must
reproduce the prior accepted complete free-generation inputs/logits/PCM.

Actual original DAC decodes all instrumented native code grids. Every native/
original DAC PCM is recognized by the same final recognizer library/model;
canonical and reconstructed original positives must score WER0. All outcomes
are checkpointed and hashed before the final gate, which fails unless serial
scores WER0 through both decoders. An independent complete artifact/source/
array/recurrence audit is required before accepting even that scoped result.

The preceding fresh original projection replay38868907653 passes on a separate
Xeon Platinum8573C. The official artifact12032740700, all13 payloads and exact
fresh query/both hidden-head outputs are verified; it is not full generation.
[Cause report on its immutable candidate](https://github.com/CrispStrobe/CrispASR/blob/d868644bda0f0bdc6e6288a564d51a8ff545ff87/docs/historical/dia-reduction-cause-2026-10-16.md).

Current main lacks the diagnostic portable-RNG/F32-cross-cache path used by
this pinned experiment. Additional current-main integration, prompts/seeds,
actual GPU/Q4 checks and profiling remain separate before any promotion.
[Candidate method](https://github.com/CrispStrobe/CrispASR/blob/3c2fbb5a9c8c73225677ce6c9eabb6b31b21a92b/tools/ci-heavy/dia-serial-head/README.md).

Independent auditor eadfb145 is prepared and its26 rejecting calibration
controls pass. It checks complete source/artifact/physical identities, all
controls, delay-reversed grids, original-decoder ASR scores and a freshly
compiled head recurrence using retained actual libraries. Fresh seed42
std::mt19937 advanced once per decoder step must match each final state.
Single collector session55605 under the owned storage root/collector-v1
preserves the complete heavy artifact even on failure, then audits it after
the exact-source lint gate completes. No second model run was submitted.

The second-CPU replay also passes its independent full-array audit and all10
lint jobs. Cause integration61f4ed8c preserves all7 tested files byte-exact;
this completed projection evidence remains distinct from pending speech.
