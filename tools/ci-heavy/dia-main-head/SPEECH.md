# Current-main native speech A/B

`dia_main_speech.py` tests the current graph candidate's actual production
recipe, independently of the accepted pinned-runtime recovery. It retains F16
cross caches, existing std::discrete_distribution and current top-k/output
protocol. No portable RNG, F32 cache switch, forced token, external hidden state
or seed sweep is introduced. All modes use F32 neural weights and output-F32
codec pins; original prompt, seed42, CPU4 and capacity3072 stay fixed.

The pre-head source9aa8df20 and graph candidated198d7a4 are built separately.
Their compiled source differences are checked explicitly; all other relevant
C++/binding files remain exact. The worker changes no runtime source. Conversion
uses the verified immutable8d0ad886 export tools only, with all381 coefficients
and complete model SHA checked. Those tools do not supply the inference runtime.
Original DAC calls use the actual pinned4a9e29b1 source, checkpoint and Torch2.6.

Pristine/OFF/identity complete native inputs/logits/stages/PCM must be exact.
Serial runs the same current native recipe. Returned grids follow the current
production stop/crop law, unlike the historical extra-frame protocol. A local
control maps all1226 actual original inputs to all1210 actual original code
frames exactly; five invalid-input and three ASR-score controls pass. Each
returned native grid is decoded again through the real current C ABI and must
reproduce its complete generated PCM exactly, then through original DAC.

All native/original DAC outputs and both canonical/original-reconstructed
positives use the same final native recognizer. Both positives must score WER0.
Actual CLI OFF/serial generations must reproduce complete Session inputs and
logits and pass their own ASR checks. Full CLI PCM is retained, including WAV
quantization and its normal watermark; no cross-surface bitwise PCM assertion
is made. A neutral-filename GGUF is also opened through C-ABI auto detection.

Six available step-zero/cache boundaries are compared with actual canonical
Python arrays, reporting cosine, both norms, norm-relative error, relative L2
and max absolute error for OFF/serial. Frozen gates are cosine>=.99,
norm-relative error<=.01 and relative L2<=.02. This is a bounded current-main
capture, not all18 layers or full crispasr-diff integration. Main's legacy Dia
reference/diff wiring still needs reconciliation with the verified observer;
that task remains open regardless of the speech result.

All outcomes and payload hashes are checkpointed before final quality gates.
The run fails unless serial passes these stage gates and WER0 through native
DAC, original DAC and CLI. Baseline is not required to fail: if it also passes,
there is no demonstrated default-quality benefit. The complete artifact still
needs independent auditing. This tests one prompt/seed and F32 neural weights,
not additional prompts, quantized/GPU quality, speed or a production promotion.

```sh
gh workflow run heavy-cpu.yml --ref IMMUTABLE_CANDIDATE \
  -f script=tools/ci-heavy/dia_main_speech.py -f timeout_minutes=300 \
  -f pip="numpy==1.26.4 torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 pydantic>=2.11.3 huggingface_hub==0.36.0 safetensors==0.5.3 descript-audio-codec==1.0.0 gguf"
```
