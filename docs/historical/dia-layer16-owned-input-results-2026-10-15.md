# Dia layer16 at the first F32 sampling divergence

A bounded CPU replay executes the pinned original Python `DecoderLayer` and
literal native ggml layer expressions at step978. It consumes captured
layer15 outputs and real layer16 self-cache prefixes, with each owner's
cross-cache producer reconstructed from its captured encoder output. All13
original F32 coefficient tensors (293,625,856 bytes) and14 mapped native tensors
match the physical F32 export audit. No full-model load or new generation.

Native final output reproduces the independent hosted capture byte-for-byte;
repeated one/four-thread native runs are also byte-exact. Across39 comparisons,
the final same-input relative L2 errors are3.527e-7 on original-owned inputs
and3.636e-7 on native-owned inputs. The largest stage error is cross-query
relative L2 of2.145e-5, with minimum row cosine0.999999999231 and maximum row
norm error5.631e-7. All unchanged cosine/norm/error gates pass. All32 actual
2x-scale negatives fail the magnitude/error gates.

Within the current original-Python context experiment, hidden-input-only
substitution changes final output by1.880e-6 relative L2, self-prefix-only by
6.286e-7 and cross-cache-only by8.622e-8; all three together give2.043e-6.
These nonlinear interventions are not additive causal percentages. All eight
contexts repeat exactly within the process. Reversing self-cache heads and
swapping cross-cache batches produce actual errors0.2775 and0.01863; both fail
the unchanged1% error gate.

An initial cross-process original-output exactness assertion failed by
9.029e-8 relative L2 (maximum absolute error3.815e-6). Cache values checked
later are byte-identical. The current process matches the main replay, but
that does not resolve the first failure or prove general cross-process Python
byte reproducibility. Its log and method are retained; its initial context
arrays were overwritten during investigation and are unavailable. The complete
current context arrays, repeats and wrong-cache payloads are retained. This
limitation does not alter the independently observed native/hosted byte match.

The independent auditor rehashes93 initial payloads and recalculates all39
comparison rows,32 magnitude controls,8 current contexts and2 wrong-cache
controls. The first audit correctly rejected changed method files; those
original files were restored and the extended methods preserved under new
names before the successful audit. Neither failed check is hidden.

No demonstrated layer16 arithmetic defect, runtime/default change, Q4/GPU/
performance acceptance or speech repair follows from this component replay.
The original/native full-speech acceptance failure remains open.

[Structured results](dia-layer16-owned-input-results-2026-10-15.json).

[Complete layer16 proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/a6d4c51a50463c5913c5c1bbba53acfeb6bd420d/dia-20261015-decoder-boundaries/complete-owned-layer16.zip) contains136 payloads,
592,952,896 bytes, SHA256 `a62cdd1d2c146e427da37577a86a40e554bc507ebf0db30a694b95012cf887f3`. It is public and anonymously
verified, with raw arrays, actual methods, native executable/libraries,
coefficient/input hashes, both failed logs, independent audit and notices.

A separate weight-free scan compares all18 captured layer outputs at step978.
Cumulative relative L2 starts at1.436e-6 in layer0, peaks at3.806e-6 in layer3
and ends at2.347e-6 in layer17. These are inherited cumulative differences,
not isolated per-layer errors. The next useful diagnostic is an early-layer
owned-input replay around that growth, rather than another late-layer replay.
All18 scan rows are independently recalculated from the retained arrays.
[Complete scan proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/0dbb7e184a39d8476c06fe4769282bd83df7e343/dia-20261015-decoder-boundaries/complete-hidden-drift-scan.zip) contains45 payloads,
570,289 bytes, SHA256 `4fc755184fadf7cf74e83a0f2b6c20edefb5b09088244e3c675ae9689c2def32`; anonymously verified.
