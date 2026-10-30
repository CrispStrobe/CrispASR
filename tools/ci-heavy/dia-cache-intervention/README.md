# Dia fixed-history layer0 key intervention

This CPU diagnostic uses a scratch checkout of runtime
`8d0ad8862d2cad5d305a48a57f2a99b5024bc39a`. It does not patch the running
checkout or change production defaults. The full F32 model must run on GH;
small file/axis guards and syntax checks can run locally.

The immutable public input ZIP is pinned in `dia_cache_intervention.py` by
revision and full SHA256. Its 979 input rows and complete native logits come
from the retained step978 boundary run. Original/native key arrays come from
the independently audited key-history producer fixture. They are captured
arrays, not newly reconstructed keys. The ZIP includes its full per-file hash
manifest and prior provenance/physical audit references.

The worker exports the same physically audited F32 GGUF, verifies its full
SHA256, and builds a small executable linked to the actual pinned Dia runtime.
It invokes shared `dia_tts_synthesize` with the canonical tagged text and fixed
979-row forced history. No DAC is loaded. A null PCM return is expected;
acceptance requires every input and logit, the capture marker, and actual
attention inputs. The executable's exit alone cannot pass the diagnostic.

Run on GH with:

```sh
gh workflow run heavy-cpu.yml --ref main \
  -f script=tools/ci-heavy/dia_cache_intervention.py \
  -f 'pip=numpy==1.26.4 torch==2.6.0 safetensors==0.5.3 gguf==0.18.0 huggingface_hub==1.10.1 requests' \
  -f timeout_minutes=300
```

This is the prepared dispatch recipe, not evidence of a hosted pass. Run IDs
and terminal results belong in `PLAN.md` and the experiment receipt.

The predefined arms are:

1. Baseline: no replacement input; all979 logit rows must reproduce retained
   native logits byte-for-byte, and the captured layer0 keys must match too.
2. Native replacement: feed that same full key array through the new input;
   all logits and retained stages must remain byte-exact.
3. Original prefix: replace positions0..977, keeping native current key978.
4. Original current: replace key978, keeping the native prefix.
5. Original both: replace prefix and current key.

All replacements occur only in layer0 attention at step978. Native producer
outputs and caches remain intact. The actual consumed key tensor must equal
the requested arm. Earlier logits, layer0 Q/V, input/norm and cross inputs must
remain exact. Shape is F32 `[batch=2, time=979, kv_heads=4, head_dim=128]`, with
unconditional batch first. The loader rejects missing, truncated, extra-length
and nonfinite files. The compiled weight-free selftest exercises that same
loader; Python guards reject wrong shape/dtype/nonfinite arrays.

The receipt retains final hidden/layer/logit cosine, both norms, relative L2,
and original Torch2.6 filtering support plus head2 CFG46/904 margin and ranks.
It observes the actual original sampler's probabilities without drawing a new
trajectory. Every arm, source, patch, actual executable and linked-library
hash survives a failure. A key-only result is conditional: correlated Q/K
phase errors are outside this intervention. No arm establishes speech, Q4,
GPU or performance acceptance.

The first hosted attempt38726696058 reproduces all979 baseline logits and keys
exactly but aborts in the native control before any original-key arm runs. Its
full artifact remains retained. The scratch patch now skips only the pruned
layer0 past-key upload at step978 and retains the replacement tensor pointer
across the existing capture rename. The new weight-free `dia-key-graph-guard`
reproduces both missing old lookups and verifies the corrected consumed values.
It passes locally against the actual failed-run ggml libraries and exact
c36dab89 headers; the worker runs it before downloading full model weights.
The repaired full native control is still mandatory before interpreting arms.

The repaired five-arm run and independent audit are complete; key replacement
alone does not restore the disputed ordering. See the
[complete result](../../../docs/historical/dia-cache-intervention-results-2026-10-15.md).
