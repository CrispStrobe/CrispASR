# tools/ci-heavy: heavy CPU jobs on GitHub Actions

Scripts here run on GitHub-hosted runners through `.github/workflows/heavy-cpu.yml`,
triggered by hand. Use this for CPU work that used to go to Kaggle: Python reference
dumps, parity diffs, ASR roundtrips, and download → convert → quantize → upload.

| Runner | CPU / RAM | Disk | Notes |
|---|---|---|---|
| `ubuntu-latest` (default) | 4 vCPU / 16 GB | ~14 GB free on `/`, plus the `/mnt` scratch disk; the workflow also deletes unused toolchains | fp32 references up to ~3B params |
| `ubuntu-22.04` | x86 CPU | as above | Pinned older Ubuntu image for compatibility checks |
| `ubuntu-24.04-arm` | 4 vCPU / 16 GB | as above, minus `/mnt` | aarch64 numerics |
| `macos-14` | 3 vCPU (M1) / 7 GB | ~14 GB | Metal |

Runners are free for public repos and a job can run up to 6 h.

**Keep on Kaggle** (`tools/kaggle/`) only what needs a real GPU: CUDA or Vulkan
speed and parity on hardware, and PyTorch on CUDA. Kaggle allows one account per
person, with GPU sessions used for GPU work.

## Running a job

```bash
gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/<name>.py \
    -f pip="torch torchvision transformers>=4.57 accelerate pillow"
gh run watch "$(gh run list -w heavy-cpu.yml -L 1 --json databaseId -q '.[0].databaseId')"
gh run download <run-id>           # the $HEAVY_OUT artifact
```

## Script contract

`dia_dac_converter_validate.py` checks the isolated Dia DAC conversion tool on
main without needing the diagnostic runtime branch. It runs seven tiny actual
original-module/CLI contracts, then converts the pinned published template using
the original DAC checkpoint. The GGUF SHA/size and all 134 physical audit rows
must match the prior standalone candidate that passed startup/full-history
numerical comparisons. `dia-dac-export.yml` calls this worker automatically on
converter/helper/test changes. Run manually with
`-f pip="numpy torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 pydantic safetensors huggingface_hub descript-audio-codec==1.0.0 gguf"`.
This is conversion integration evidence, not new free-speech or GPU acceptance.

- **`$HEAVY_OUT`**: write results here. The whole directory becomes the run's artifact, kept for 14 days.
- **`$HEAVY_SCRATCH`**: large scratch space for models, builds and caches. Not uploaded. `HF_HOME` points inside it.
- **`$HF_TOKEN`**: the repository secret (fine-grained, CI-scoped); may be empty.
- **`$HEAVY_OUT/summary.md`**: if present, appended to the run's summary page.
- **Exit code**: non-zero when the check fails, so the run shows red. A readout that prints the same thing whether it passed or failed isn't a check.
- **Paths**: the workflow only accepts `.py` / `.sh` files under `tools/ci-heavy/`.
- **Porting a Kaggle kernel**: replace `/kaggle/working` with `$HEAVY_OUT` and `/kaggle/input/...` with a download into `$HEAVY_SCRATCH`. Drop the GPU and ccache datasets.

`miotts_sample_rate.py` checks the pinned MioCodec-v2 model through the native
live test, session rate getter, CLI WAV header and both TTS→ASR roundtrips.
It also checks 24 kHz and missing-key metadata dispatch using temporary copies;
those copies are not claimed to be legacy codec speech models. Run with
`-f pip="numpy gguf huggingface_hub"` on Linux x86 or ARM.

`mimo_cabi_device_params.py` checks four device/verbosity combinations and four
failed-open default restores through the actual shared library and an init hook.
Removing the two forwarding assignments must fail; restoring them must pass.
Run on Linux x86 with no extra pip packages. No model/GPU inference claim.

`data2vec_reference_parity.py` verifies the existing independent Python CTC
transcript using pinned logits/vocabulary/sample hashes, rejects four broken
provenance/target controls, then runs the actual Q4 CLI and F16 stage regression.
Zero-WER upstream parity remains hard; the original model's 4.55% human WER
is reported separately. Run with `-f pip="numpy gguf huggingface_hub"`.

`dia_dac_export_validate.py` validates the explicit original critical-output
DAC conversion against retained startup and complete-original-utterance
references. It requires exact bytes from the prior validated diagnostic and
physically audits original F32 promoted coefficients plus unchanged unrelated
payloads. This does not establish native free speech or GPU execution and
does not update production model pins. Run with
`-f pip="numpy torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 pydantic safetensors huggingface_hub descript-audio-codec==1.0.0 gguf"`.

`dia_canonical_random_control.py` replays the complete accepted original history
without model weights. It executes actual pinned generation, sampling, BOS/EOS
feedback and delay reversal on retained original logits, requiring all 1,226
decoder inputs and the final 1,210-frame code grid to match exactly. It observes
actual Torch exponential noise and RNG states, verifies the source-derived race
against actual multinomial, and retains a fixed row-split RNG negative control.
Use `-f pip="numpy torch==2.6.0 pydantic==2.11.3 gguf"`. Other Torch versions
are recorded and may fail; this never establishes native free speech or PCM
acceptance, changes production defaults, or supplies forced production tokens.


`dia_portable_random_control.py` validates the compiled MCG31/exponential-race
candidate against all 11,034 pinned original draws, using both actual original
probabilities and actual native filtering. Every original decoder transition
and final codec code must agree. Use `-f pip="numpy torch==2.6.0 pydantic==2.11.3 gguf==0.18.0"`.
No model weights are loaded and this does not establish speech acceptance.

`dia_portable_speech_control.py` runs the fixed canonical short prompt through
actual original neural/DAC generation with the gated native candidate, then
actual native F16 C ABI and CLI generation with the validated critical-output
DAC export. It requires canonical original codes and original/native/CLI ASR
roundtrips. `CRISPASR_DIA_PORTABLE_RNG=1` selects the experimental candidate;
the default remains the existing sampler. Use the DAC exporter dependency set
with pinned Torch/torchaudio 2.6.0, torchvision 0.21.0, DAC 1.0.0, Pydantic 2.11.3
and GGUF 0.18.0. This fixed-prompt CPU control does not establish full model,
GPU or Q4 acceptance or change production model pins.


`dia_neural_precision_control.py` isolates the remaining short-speech failure
with two fixed original-neural coefficient policies: nine original F32 output
heads (+37,896,192 bytes), and full original F32. The streaming helper uses the
existing converter's actual parameter mappings, preserves template metadata and
unselected payloads, and physically verifies all 381 tensors against the pinned
original checkpoint before native execution. Five tiny actual original-module
contracts cover coefficient/layout/metadata preservation and reject changed
weights, nonfinite values and bad pins. Sequential hosted exports, strict
canonical `crispasr-diff`, actual native C ABI/CLI generation and ASR preserve
every failed arm. Run with the portable speech worker's pinned dependency set,
`-f timeout_minutes=240`. Diagnostic models stay out of production registries;
no RNG, seed, top-k/top-p or reference/gate changes are made, and this does not
establish GPU, Q4 or complete model acceptance.

`dia_attention_history_control.py` runs actual original and native step-978
self-attention snapshots with full-trajectory OFF/ON/OFF controls. It observes
post-rotation queries, complete K/V histories and pre-output-projection attention
in all 18 layers; preserves canonical input/code grids and the original ASR
positive control; records cosine, magnitudes and complete failed runs. Diagnostic
snapshots are not free-speech acceptance or a new model/default. Run the diagnostic
source branch with `-f pip="numpy torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 pydantic safetensors huggingface_hub descript-audio-codec==1.0.0 gguf"`.
