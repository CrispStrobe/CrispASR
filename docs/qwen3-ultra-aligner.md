# Qwen3-ForcedAligner-Ultra base model

The base [Qwen3-ForcedAligner-Ultra](https://huggingface.co/17slever17/Qwen3-ForcedAligner-Ultra)
checkpoint uses the existing Qwen3 forced-aligner architecture. It can be
converted and supplied to `--align-only`; the GGUF filename must contain
`forced-aligner` so the aligner dispatch selects the Qwen path.

For example, install the converter dependencies, download the pinned checkpoint,
and convert it:

```sh
python -m pip install torch numpy gguf safetensors huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('17slever17/Qwen3-ForcedAligner-Ultra', revision='765dc1d714e4c80196782a5df5bfbcb3e1416736', local_dir='ultra-model', ignore_patterns=['local_head/*', 'companion/*'])"
python models/convert-qwen3-asr-to-gguf.py --input ultra-model \
  --output qwen3-forced-aligner-ultra-f16.gguf
crispasr --align-only -am qwen3-forced-aligner-ultra-f16.gguf \
  -f audio.wav --ref-text 'Words to align' --align-format json
```

This command handles the base model only. For the experimental optional local
boundary refinement head and its separate converter, see the
[local-head guide](qwen3-ultra-local-head-port.md). Do not infer local-head
precision from the base-only mode.

A converted F16 GGUF is published as [`cstr/qwen3-forced-aligner-ultra-GGUF`](https://huggingface.co/cstr/qwen3-forced-aligner-ultra-GGUF) (revision `992f8360e1e99cc5dae6bb162c530cd927e14a30`), and the registry resolves it:

```sh
crispasr --align-only -am qwen3-forced-aligner-ultra --auto-download \
  -f audio.wav --ref-text 'Words to align' --align-format json
```

A GGUF named after the upstream repo (`Qwen3-ForcedAligner-Ultra*.gguf`) also selects the Qwen path.

**Regression coverage:** the nightly regression runs `qwen3-forced-aligner-ultra-f16` and `qwen3-forced-aligner-0.6b-f16`. Each is compared with the upstream `qwen-asr` word timestamps for JFK, with a tolerance of one 80 ms class. Changes to the aligner sources select both entries on pushes and PRs. The manual audit workflow is still available for a fresh upstream comparison.

[Hosted CPU audit 42311362995](https://github.com/CrispStrobe/CrispASR/actions/runs/42311362995)
converted all 708 tensors from the pinned model (source SHA256
`e6fcf5be2c11b265f7217ba6102917a7fe64438506357be6999f6ad5a707cb5f`)
and compared the F16 GGUF against the upstream `qwen-asr` Python implementation.
All 22 word boundaries matched exactly for the English `samples/jfk.wav`
transcript. This is one CPU input; it does not qualify the optional head,
other languages, quantizations, or GPU execution. The manually dispatched
`issue579 Ultra CPU parity` workflow reproduces this narrow comparison (last run 42371592193 passed).
