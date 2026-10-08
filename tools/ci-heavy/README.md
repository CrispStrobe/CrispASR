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

`pr492_acceptance.py --quant q4_k|f16|all` compares PR #492 with pinned main
using the same ggml source. It checks 16 shared-mel cases byte-for-byte, five
LM stages against the frozen Python reference with norms/relative L2, tokenizer
stages and RVQ codes against baseline, and English/Chinese CLI/session speech.
Default output must be exact; non-flash output is separately checked. F16 refers
to the LM; both runs use the shipped Q4_K tokenizer. All arms explicitly force
CPU, including the legacy session that predates device-flag forwarding. CLI
and session inference run sequentially to avoid retaining two F16 model/KV
contexts. Run on Linux x86 with
`-f pip="numpy gguf huggingface_hub soundfile"`. CPU results do not establish
CANN/CUDA correctness or speed. See `docs/mimo-pr492-acceptance-2026-10-08.json`
for reference provenance and the initial local mel result.

`pr492_tokenizer_diagnose.py` isolates Q4 activation-rounding effects by
promoting the same quantized matrix weights to F32 and comparing native
flash/eager attention with the official Python transformer on identical native
conv2 inputs. Pooling is checked independently on each arm's own transformer
output. The source and weight hashes, all metrics and failed gates are retained.
This is a diagnostic with dequantized weights, not original-checkpoint or
decoded-output acceptance. Run with
`-f pip="numpy gguf huggingface_hub soundfile torch transformers==4.57.6"`.

`pr492_cabi_params.py` checks the actual MiMo session C ABI using an initializer
interposition hook, without weights or GPU execution. Eight combinations cover
CPU/GPU preference, verbosity and flash attention, followed by default-restoration
checks after failed opens. An incremental rebuild with the device/verbosity
assignments removed must fail, then restoring the source must pass. Run on Linux
x86 with no extra pip packages; this validates parameter forwarding only.
