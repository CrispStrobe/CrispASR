# Dia public reference/diff validation

`dia_reference_wire.py` tests actual public entry points and separately checks
whether marking intermediate outputs changes current-main generation. This is
a bounded harness test, not speech acceptance or a performance experiment.

The worker builds pristine main9aa8df20 and the candidate with identical CPU
flags, exports and physically audits all381 F32 neural tensors, and verifies
the known output-F32 DAC export. Original Python is pinned at4a9e29b1; the
weights are pinned at257bc72f. No model runs on the VPS.

It runs the tiny original-module observer tests and compiled archive/parser
controls, then compares complete native inputs/logits/PCM across pristine and
candidate OFF/ON/OFF. ON captures all decoder block boundaries and attention
at step5, plus encoder and DAC stages. Each arm uses the same native sampler,
F16 cross cache, seed42 and32-step capacity; production defaults stay intact.

The actual `tools/dump_reference.py --backend dia` runs without a placeholder
WAV, with attention and block-boundary layers0,17. The actual four-argument
`crispasr-diff dia MODEL REFERENCE` must replay the complete original input
grid exactly before interpreting numbers, compare every requested stage, and
replay original post-reversal codes through the actual DAC. Strict comparison
retains cos_min>=0.9995 and per-row norm error<0.02, with global relative L2
also printed. A doubled-scale vector must fail despite cosine1; integer inputs
must be exact. A missing reference must fail through the Dia dispatch.

Raw reference GGUF, complete native tensor files, sources, libraries, physical
export ledgers, commands/logs, transparency comparisons and payload hashes go
to the official artifact, including on failure. Fresh independent artifact
and array auditing is required before accepting the result. A nonzero parity
exit is retained; the worker never loosens thresholds to obtain a pass.

Free-running speech/ASR, broader prompts/seeds, Q4, GPU and speed are separate
acceptance tasks.32 steps are intentionally too short to establish speech
quality. Successful forced-input replay does not establish sampler parity or
delay reversal; the production protocol test and speech worker cover those.


The composed-source worker also retains the actual diff ELF, head header and
all three licenses. An extra native identity-head arm enables captures and
must remain byte-identical to pristine inputs/logits/PCM. A second public diff
replays original inputs through the serial head with the same stage gates,
retaining a separate complete tensor directory. These checks validate the
composition of the accepted head diagnostic with capture instrumentation;
none enables a production default or establishes unforced speech quality.


Local combined-source preparation passes the actual30-check head unit,
17 original-class observer tests and237 unchanged computation-call guards.
Actual tiny GQA runtime replay passes1194 complete comparisons across capture
ON, identity-head+capture and serial-head+capture modes. OFF/ON/OFF/identity
input/logit files are exact to the earlier accepted candidate. This is local
synthetic composition proof; the combined full-model hosted run remains required.
