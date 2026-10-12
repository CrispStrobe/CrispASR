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
