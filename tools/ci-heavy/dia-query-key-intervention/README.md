# Dia fixed-history query/key factorial

This diagnostic extends the accepted key-only experiment at the same pinned
runtime8d0ad886 and step978. It patches only a scratch runtime. Production
inference and defaults remain unchanged. The full F32 model runs on GH.

The original query comes from **actual pinned original module replay**, not a
query captured during the earlier full-model run. `dia_self_query.py` executes
unchanged original self-attention, observing its Q projection and RoPE outputs.
Five physically verified F32 tensors total41,951,232 bytes. Replay reproduces
all retained component stages exactly; full-model pre-SA norm is exact and
self-attention/output reconstruct within the unchanged2e-6 relative-L2/norm
and0.9999999999 cosine gates. Repeated arrays are exact. Scale and head-order
controls fail. An independent raw-input/original-expression replay verifies all
49 payloads and every baseline/negative attention array.

The full-model stages are not byte-exact. That uncertainty remains explicit in
all receipts. This is a conditional intervention using that replayed Q together
with previously captured original keys; it cannot claim an earlier full-model
query or establish a production or speech repair.

The immutable complete producer proof and compact query-input ZIP are published
under `dia-20261015-query-key/` in the public regression-fixture dataset. Revision
and full digests are pinned in `dia_query_key_intervention.py`. Full anonymous
downloads match. The compact ZIP retains native/original query arrays, the
producer receipt, independent audit, manifest and license notices. The complete
proof additionally retains consumed coefficients, source, inputs, every trace,
controls and auditor. The previous immutable shared-history fixture stays pinned.

The predefined arms are:

1. Baseline without replacement inputs.
2. Native Q/native K substitution control.
3. Replayed-original Q/native K.
4. Native Q/captured-original K.
5. Replayed-original Q/captured-original K.

Only layer0 at step978 changes. All four substitutions use the same paired graph
input path. Both native controls must reproduce all979 retained logit rows and
all captured baseline stages byte-for-byte before interpreting any original arm.
Actual consumed Q `(2,1,16,128)` and K `(2,979,4,128)` must equal each requested
F32 array exactly. Earlier978 logits, layer0 input/norm/V and cross-cache inputs
remain exact. Final hidden/logit/layer cosine, both norms and relative error,
plus original Torch2.6 sampler46/904 margin/ranks/support, are retained for every
arm. Original-key-only results can be cross-checked against the accepted prior
key-only experiment. No seed, filter, text or model coefficient changes.

The existing compiled file loader rejects missing, short, extra and nonfinite
Q/K files. The weight-free actual ggml guard tests both renamed replacement
leaves, pruned original query producer, actual consumed arrays and dot products.
Full patched-runtime syntax is checked against its exact ggml headers.

The original audited run used this historical dispatch command. Its feature
branch is retired after integration; the command records provenance rather than
a current ref to rerun.

```sh
gh workflow run heavy-cpu.yml --ref diag/dia-query-key-20261015 \
  -f script=tools/ci-heavy/dia_query_key_intervention.py \
  -f 'pip=numpy==1.26.4 torch==2.6.0 safetensors==0.5.3 gguf==0.18.0 huggingface_hub==1.10.1 requests' \
  -f timeout_minutes=300
```

The [completed hosted run](https://github.com/CrispStrobe/CrispASR/actions/runs/38764924112)
and independent739-payload audit pass; the result does not restore the original
ordering or support. Exact source, terminal CI and public proof are recorded in
[the result receipt](../../../docs/historical/dia-query-key-results-2026-10-15.md). Do not
repeat the unchanged historical experiment.
No new model runs belong on the VPS; local evidence and all large files stay
under ${CRISPASR_STORAGE}/crispasr/dia-query-key-20261015.

## One-step native query producer

`tools/ci-heavy/dia_query_producer.py` builds the literal pinned projection/RoPE
with the actual c36dab89 libraries and headers. It needs the complete original
query proof and previous hosted native artifact extracted locally. Only the
16 MiB native query matrix and small owned arrays are loaded; no full model or
new full-speech run is involved. Large outputs belong on ${CRISPASR_STORAGE}.

```sh
python tools/ci-heavy/dia_query_producer.py \
  --query-proof /path/to/query-replay-v2 \
  --native /path/to/hosted-native-out \
  --headers /path/to/c36-include \
  --output ${CRISPASR_STORAGE}/crispasr/fresh-native-query-probe
```

The producer requires an exact captured-native query control before interpreting
projection versus RoPE. It retains both one/four-thread fresh repeats, consumed
pre-RoPE arrays, native-owned substitution control, actual scale/head-order
negative controls, source, ELF, linked libraries and hashes. Its optional CMake
target is `dia-query-producer` when this method directory is added to a build.
This is conditional one-step attribution; it supplies no speech/GPU acceptance
or additive causal percentages and does not replace the queued full factorial.
