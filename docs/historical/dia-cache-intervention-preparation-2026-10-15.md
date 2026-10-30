# Dia full-decoder layer0 key intervention preparation

**Completed:** the repaired hosted run and independent 720-payload audit pass.
All three original-key arms fail to restore the disputed ranking; see the
[complete result](dia-cache-intervention-results-2026-10-15.md). The preparation
and failed first attempt below are retained as historical evidence.

Initial candidate `2f04f4e6f64ab3aa607fe055b9e8c39a8b1a87ea` adds a diagnostic worker
and small runner. Production runtime/defaults remain unchanged. The worker
patches only its scratch checkout at the immutable previously tested boundary
runtime `8d0ad8862d2cad5d305a48a57f2a99b5024bc39a`.

The [GH full-model run38726696058](https://github.com/CrispStrobe/CrispASR/actions/runs/38726696058)
and [lint38726699171](https://github.com/CrispStrobe/CrispASR/actions/runs/38726699171)
have been submitted. Heavy38726696058 failed after byte-exact reproduction
of all979 baseline logit rows. The native replacement control aborts on a
missing tensor upload: its new full-key input prunes the unused past-key leaf.
The new input also needs a retained pointer across capture renaming. No
original-key arm ran, so there is no valid intervention result. That plumbing
is now repaired and both cases pass a small actual ggml graph; repaired retry
38738218813 and all10 lint checks subsequently pass; see the complete result. The failed raw artifact and logs are
retained in full, with publication details below.
The [method](https://github.com/CrispStrobe/CrispASR/blob/2f04f4e6f64ab3aa607fe055b9e8c39a8b1a87ea/tools/ci-heavy/dia-cache-intervention/README.md)
records the exact dispatch and predefined gates.

Local axis-ownership guards pass, including wrong shape/dtype/nonfinite
rejections. The actual compiled C++ file loader accepts valid keys and rejects
missing, truncated, extra-length and nonfinite files. The patch applies to the
exact pinned runtime, and the full patched translation unit passes a C++17
syntax check against its actual headers. All four repository secret guards
pass unskipped. These are preparation checks, not hosted/model acceptance.
The actual worker sampling observer also passes on pinned Torch2.6.0+cpu:
original/native support sets exactly match the retained first-divergence proof,
with46/904 ranks32/33 reversed. Original CFG margin is+4.6253204e-5; native
is−2.6702881e-5. This independently checks the measurement path without a new
full-model inference.


The [immutable input ZIP](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/5e2fc2430413357a21807cd0350960aeeb6fe183/dia-20261015-cache-intervention/fixed-history-inputs-v1.zip)
contains31 payloads,138612077 compressed bytes, SHA256
`a0c7e2e876394469e07a455095cf79f6e004c1d2087f3ce4fb671dedbb567d52`.
An anonymous complete download matches that SHA. It retains979 shared inputs,
complete original/native logits, original endpoint stages and captured original/
native layer0 keys with source/hash manifests and license notices. No weights
or newly reconstructed keys replace these captured inputs.

The baseline must reproduce every retained native input and all979 native
logit rows byte-for-byte. A second native-key replacement arm must preserve
all logits and captured stages, controlling the new graph input. Only then
interpret original prefix0..977, original current key978, and combined arms.
The actual consumed keys must match each requested array exactly. Earlier
logits, layer0 input/norm/Q/V and cross inputs must remain unchanged; downstream
current activations may respond naturally. Past histories stay native.

Compare all18 decoder outputs, final hidden/logits with cosine, both norms and
relative L2, plus actual original Torch2.6 filtering support and head2 CFG46/904
margin/ranks. Keep all arms and failures. This key-only intervention cannot
settle correlated joint Q/K phase effects. Null PCM is deliberate without DAC;
runner exit alone is insufficient. No speech, GPU, Q4, performance or default
promotion follows from this diagnostic. The full model runs on GH, with all
local evidence under ${CRISPASR_STORAGE}/crispasr/dia-cache-intervention-20261015.

## Failed control retained; diagnostic graph repair submitted

The complete raw failed artifact11973635349 is428224673 bytes, SHA256
`196a86087fc6df6d7037d7991f6cc23c67c6cbad727bc5a5aed529d9552e37ac`.
Full digest/CRC/unique safe paths pass. Independent audit verifies all979
baseline inputs/logits and captured keys byte-for-byte, actual executable and
shared-library hashes, all input payload hashes and18 endpoint metrics.
The native control's first978 logit rows are exact; it aborts before the target
step computes. No original-key arm ran. The actual baseline runner SHA256 is
`dbbf9fba971b669f27e8813320a52dbc7da6abc505100c4b8cb67485ca5ef12b`.

Repairfc8f652f retains the replacement input pointer across the existing capture
rename and skips only the unused layer0 past-key upload at step978. A small
actual ggml graph reproduces both old missing lookups and verifies all corrected
consumed values exactly. It passes against the actual failed-run shared libraries
and exact c36dab89 headers; the worker runs this guard before model downloads.
Full patched-runtime syntax and existing input/axis controls still pass.

[Repaired heavy38738218813](https://github.com/CrispStrobe/CrispASR/actions/runs/38738218813)
and [lint38738222779](https://github.com/CrispStrobe/CrispASR/actions/runs/38738222779)
are submitted at `fc8f652fac1a602a2a9e7fbc645dc6795cd8f0e4`. The numerical retry and independent audit are complete. Both native controls
are exact and all three original-key arms are retained in the result receipt.
No acceptance threshold, model coefficient or production default changed.

The [complete failed-run proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/26b52aa34bafd3b50b22c0e62cedda6231f40fdc/dia-20261015-cache-intervention/complete-hosted-intervention-failure-v1.zip)
contains15 payloads,428154431 compressed bytes, SHA256
`f94ffdd5b3379bf553ef24a5d980c7d473845c2fc3cda1a8ee018a9313ce680a`.
An anonymous full download matches. It includes the complete raw hosted ZIP,
API/logs, auditor/result, actual hosted-ggml graph guard executable/source/log,
manifest and notices. Its original-key diagnostic remains incomplete.
