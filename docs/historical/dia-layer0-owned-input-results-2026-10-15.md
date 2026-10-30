# Dia layer0: owned inputs and the self-cache key history

The early-layer probe uses the same shared-history step978, original Python
source4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8 and literal native source
8d0ad8862d2cad5d305a48a57f2a99b5024bc39a as the accepted late-layer studies.
All13 original F32 parameter tensors (293,625,856 bytes) and14 native mappings
match physical export hashes. It loads only this layer, not the full model.

Native layer output reproduces the independently captured hosted output
byte-for-byte; repeated one/four-thread native executions are also exact.
There are39 component/hosted comparisons. Final same-input relative L2 is
3.662e-6 on original-owned context and3.663e-6 on native-owned context.
The largest stage error is self-attention: relative L2 2.120e-5,
minimum row cosine0.999999997537 and maximum row norm error3.636e-6.
All unchanged cosine, magnitude and error gates pass. All32 doubled-output
negative controls fail magnitude/error despite cosine near one.

The actual incoming layer0 hidden state is byte-identical between original
and native captures. On the same original module with independently varied
owned contexts, hidden-only substitution is exactly unchanged, self-prefix
substitution changes final output by3.353e-6 relative L2, and cross-cache-only
by1.052e-7. These are nonlinear interventions, not additive percentages.
All eight contexts and their actual serialized inputs are retained; repeated
original execution is exact within this process. Both endpoint contexts also
match the separate main replay byte-for-byte in this experiment. This does
not resolve the earlier layer16 original cross-process exactness failure.

A separate process holds hidden/cross inputs fixed and splits actual self-cache
keys from values. Key-only substitution changes output by3.366e-6, value-only
by1.397e-7, and both by3.353e-6. All64 stage comparison rows, all repeats and
four actual serialized input payloads are retained. The key history itself
has relative L2 1.569e-5; value history has6.788e-7. The experiment localizes
the larger context effect to keys in this fixture; it does not establish that
this is the sole cause of the later discrete token or speech failure.

Actual wrong-cache controls reverse self-cache heads or swap cross-cache
batches; resulting relative L2 errors0.6153 and0.03758 both fail the existing
1% error gate. Independent auditors rehash all57 main payloads and27 coefficient
payloads, recompute39 main rows/32 scale negatives/8 contexts/2 real cache
negatives, then independently recompute64 split-stage rows and verify all
four split input payloads. Every process uses a separate output directory;
no failed process output is overwritten.

There is no new demonstrated runtime defect or accepted optimization here.
Previously rejected source-RoPE full-speech controls remain rejected. Do not
repeat those unchanged, change the seed/filter/reference, infer Q4/GPU parity,
or promote a model/default from this component result. Full speech still fails.
The next hypothesis must address unmeasured behavior or a specific arithmetic
mechanism rather than blindly replaying every remaining decoder layer.

[Structured results](dia-layer0-owned-input-results-2026-10-15.json).

[Complete283-payload proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/abc43fe7c26407fa243b78c200bb837e3930dee0/dia-20261015-decoder-boundaries/complete-owned-layer0.zip) is public and anonymously verified:
804,678,169 bytes, SHA256 `38ade7ae8952507729eaa48ed87cc0523866d923acf0a316fe01cf158cd7eb0c`. The archive includes all
consumed coefficient/input payloads, actual executable/shared libraries,
original/native expressions, both independent audits, complete process outputs,
input-path mapping, notices and the per-file hash manifest.
