# Index-Echo 9B mixed Q4_K investigation

**Preparation is blocked; no Q4 candidate has passed GPU acceptance.** Corrected
CPU run [37046153440](https://github.com/CrispStrobe/CrispASR/actions/runs/37046153440)
produced a 5.05 GB plain decoder, then its upload commit failed with HTTP 400
(private HF storage quota). Run 37044026371 was cancelled to correct source-F32
guards. Private staging is a transfer choice, not a runtime requirement; do not
retry the quota-limited route unchanged. GPU-only v2 has not been pushed. Its
preparation revision/checksum remain unset until a successful producer run and
artifact transfer.

Build the CUDA bundle on GitHub. Run `tools/ci-heavy/index_echo_q4_prepare.py`
through `heavy-cpu.yml` with the existing HF write credential; quantization, physical audits and
explicitly unvalidated public experimental uploads run on that CPU runner.
Pin its preparation revision/checksum as well as the successful CUDA build
SHA, HF dataset revision and SHA256 before pushing with `../kpush.py`.
No GPU-less build counts as hardware acceptance.
The kernel requires two actual SM75 GPUs and exits inconclusive before model
downloads if that hardware is absent. Never repush to fish for hardware.

An immutable canonical F16 pair must first pass the existing independent
F32 stage/cache/magnitude, exact CLI/ABI, five-file and Piper checks. Four
predeclared candidates then receive the same acceptance, without relaxing
punctuation, timestamps, cosine, magnitude or cache/token criteria:

| Candidate | Decoder recipe; acoustic tower/connector always original F16 |
|---|---|
| `q4_k_plain` | Generic Q4_K baseline, with existing small-tensor guards |
| `q4_k_sensitive` | F16 token tables and recurrent weights; Q8 attention and FFN down projections; Q4 FFN gate/up |
| `q4_k_ffn_guarded` | Q4 FFN gate/up, Q8 FFN down; all other matrices F16 |
| `q4_k_middle` | Only FFN gate/up in layers 4–27 at Q4; all other matrices F16 |

All 177 original F32 tensors, including 24 recurrent convolution matrices, must
retain F32. Per-tensor inventory and actual Q4 byte counts prevent a nominal Q4 artifact
from silently remaining F16. Separate candidate directories resolve the
unchanged primary's original companion basename to the tested mixed decoder.
Each decoder is uploaded immediately to the public experimental prefix by the
GitHub producer and local runner weights are released after remote size/SHA
verification. Kaggle downloads one candidate
at a time and releases it after actual GPU acceptance. Its receipt/log is
retained even after rejection. All CLI checks must log actual CUDA layer
assignment; the small CPU VAD companion is part of the real file pipeline.
No compilation, quantization, CPU-only CLI pass, TTS synthesis, weight uploads
or default quantization changes occur in the GPU kernel.

The original v1 GPU F16 control passed every acceptance gate but stopped before
any quantization: its Kaggle token could not create model repositories (403).
All terminal logs and outputs were saved before preparing v2. This is a
preparation/auth failure, not evidence for or against any Q4 recipe.

These guards follow the existing quantizer's acoustic/adapter/output-head
precision floors for Hojo, MOSS, Qwen3-ASR, Canary-Qwen and TTS models, plus
Echo's prior Q8 failure evidence. A recipe is usable only after decoded and
numerical acceptance. A failed recipe is not rescued by a high average cosine.

## Public preparation retry (2026-10-08)

The CPU producer now writes unvalidated candidates under
`cstr/index-echo-9b-GGUF/experiments/q4-guards-20261008/`. It verifies each
immutable remote object's size and LFS SHA256 before removing the local result.
Per-candidate manifests include the weight commit. This does not overwrite
published model files or change model-registry pins. The GPU consumer still
requires a completed preparation receipt and immutable pins before launch;
Q4 runtime acceptance remains pending.

The consumer now uses that public preparation repo and each recipe’s immutable
weight revision, and rejects missing remote verification or paths outside the
experimental prefix. Its launch pins remain deliberately unset while the CPU
producer is queued; the old private route is retained only as historical evidence.
