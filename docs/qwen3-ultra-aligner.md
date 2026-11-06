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

This handles the base model only. The optional local boundary refinement head
is not converted or executed. Do not infer 10 ms local-head precision from
this mode, and do not use a separately downloaded local-head file with it.
There is no built-in Ultra auto-download entry.

[Hosted CPU audit 42311362995](https://github.com/CrispStrobe/CrispASR/actions/runs/42311362995)
converted all 708 tensors from the pinned model (source SHA256
`e6fcf5be2c11b265f7217ba6102917a7fe64438506357be6999f6ad5a707cb5f`)
and compared the F16 GGUF against the upstream `qwen-asr` Python implementation.
All 22 word boundaries matched exactly for the English `samples/jfk.wav`
transcript. This is one CPU input; it does not qualify the optional head,
other languages, quantizations, or GPU execution. The manually dispatched
`issue579 Ultra CPU parity` workflow reproduces this narrow comparison.
