# Index-Echo 9B mixed Q4_K investigation

Build the CUDA bundle (including `crispasr-quantize`) on GitHub; pin its
successful build SHA, HF dataset revision and SHA256 before pushing this
kernel with `../kpush.py`. No GPU-less build counts as hardware acceptance.
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

Per-tensor inventory and actual Q4 byte counts prevent a nominal Q4 artifact
from silently remaining F16. Separate candidate directories resolve the
unchanged primary's original companion basename to the tested mixed decoder.
Each decoder is uploaded immediately to a private experiment repository;
its receipt/log is retained even after rejection and local weights are
released after testing. No public weights or default quantization changes.

These guards follow the existing quantizer's acoustic/adapter/output-head
precision floors for Hojo, MOSS, Qwen3-ASR, Canary-Qwen and TTS models, plus
Echo's prior Q8 failure evidence. A recipe is usable only after decoded and
numerical acceptance. A failed recipe is not rescued by a high average cosine.
