# Moonshine full hosted CPU fixture — 2026-10-14

[Hosted run38555394740](https://github.com/CrispStrobe/CrispASR/actions/runs/38555394740)
passes at source `a149749ee492c4ea9fa80b682d0248090f9bb6af`, ggml
`0a862076567afd8468822ccc512c97ac6a69b340`. Its returned26-payload archive
has SHA256 `b912d7138ff36c87ee250e5e2cde59d3deb63225ae6df5bb6e3312e591901ff5`
and size103,535,813 bytes. An independent audit rehashes every payload,
checks ZIP integrity, compares all160 freshly converted F32 tensors against
previously verified original checkpoint coefficients, and recomputes complete
encoder comparisons and bad-array controls. [Audit receipt](moonshine-hosted-cpu-fixture-2026-10-14.json).

| All457×288 encoder values | Result |
|---|---:|
| Minimum row cosine | 0.9999999997168613 |
| Maximum row norm error | 5.519607124693238e-6 |
| Relative L2 error | 2.025471488361104e-6 |
| Maximum absolute error | 5.054473876953125e-5 |

The original Python card generation recipe and actual `tools/dump_reference.py`
use the complete pinned original model and unchanged JFK PCM. Original
reference PCM and encoder arrays match the independently captured inputs and
outputs within this run byte for byte. Native encoder repetitions are byte
identical. All four transcripts from automatic/explicit Python Sessions
through the freshly built C ABI equal the original transcript byte for byte;
CLI SRT words match after its requested punctuation normalization. Actual
`crispasr-diff` reports1 pass,0 fail,0 skip. Scale, nonfinite and wrong-shape
arrays are rejected. The four lightweight producer guard tests also pass.

Separate-host original encoder outputs differ at float rounding scale:
relative L2 `6.7663161488964e-7`, minimum cosine `0.9999999999804139`.
They are not byte identical. Full model bytes and original generation token
IDs agree. Actual build logs, compilation commands, CMake cache and shared
library closure are retained. Shared-library ELF bytes are absent from this
artifact, so their hashes remain producer receipts rather than local rehashes.
The first producer submission was cancelled while queued, before any model
work; its cancellation metadata remains retained.

This accepts the full **CPU** fixture and its CLI/C ABI wiring. CANN NPU
execution, scheduler fallback, ACL capture/replay, F16/Q8/Q4 model quality and
performance remain pending. The existing Moonshine diff dispatch initializes
CPU; physical validation must first add strict device routing and trace actual
scheduled NPU work. Heavy evidence remains under
`${CRISPASR_STORAGE}/crispasr/cann-model-fixture-20261014-v2`.

Complete public evidence, verified by anonymous full downloads:

- [complete-hosted-cpu-fixture.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/b5ec8e8a229bd388194237bf216ae086024212a6/cann-20261014-model-fixture/complete-hosted-cpu-fixture.zip): 103,535,813 bytes, SHA256 `b912d7138ff36c87ee250e5e2cde59d3deb63225ae6df5bb6e3312e591901ff5`.
- [complete-hosted-cpu-fixture-audit.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/b5ec8e8a229bd388194237bf216ae086024212a6/cann-20261014-model-fixture/complete-hosted-cpu-fixture-audit.zip): 32,063 bytes, SHA256 `88c2d8cb0e72888e76fd4170747e04bc56c02abd8375a13ed486c9e0a4efd811`.
