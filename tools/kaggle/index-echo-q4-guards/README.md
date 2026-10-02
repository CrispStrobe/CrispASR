# Index-Echo 9B mixed Q4_K investigation

Build the CUDA bundle on GitHub. Run `tools/ci-heavy/index_echo_q4_prepare.py`
through `heavy-cpu.yml` with the dedicated Index-Echo staging credential;
quantization, physical audits and private uploads all run on that CPU runner.
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

Per-tensor inventory and actual Q4 byte counts prevent a nominal Q4 artifact
from silently remaining F16. Separate candidate directories resolve the
unchanged primary's original companion basename to the tested mixed decoder.
Each decoder is uploaded immediately to the existing private staging repository
on GitHub and local runner weights are released. Kaggle downloads one candidate
at a time and releases it after actual GPU acceptance. Its receipt/log is
retained even after rejection. All CLI checks must log actual CUDA layer
assignment; the small CPU VAD companion is part of the real file pipeline.
No compilation, quantization, CPU-only CLI pass, TTS synthesis, public weights
or default quantization changes occur in the GPU kernel.

The original v1 GPU F16 control passed every acceptance gate but stopped before
any quantization: its Kaggle token could not create model repositories (403).
All terminal logs and outputs were saved before preparing v2. This is a
preparation/auth failure, not evidence for or against any Q4 recipe.

These guards follow the existing quantizer's acoustic/adapter/output-head
precision floors for Hojo, MOSS, Qwen3-ASR, Canary-Qwen and TTS models, plus
Echo's prior Q8 failure evidence. A recipe is usable only after decoded and
numerical acceptance. A failed recipe is not rescued by a high average cosine.
