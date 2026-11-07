# Qwen3 Ultra local-head port: CPU stages

The optional local head is not yet available through the CrispASR CLI, session
C ABI or language bindings. The experimental native path implements fixed-center
features, a CPU neural head, hidden-state capture, timestamp repair, and a
strictly pinned F32 sidecar converter/loader. End-to-end CPU parity is being
checked in [run 42319319711](https://github.com/CrispStrobe/CrispASR/actions/runs/42319319711);
its result, public integration and GPU acceptance remain open.

[Reference run 42313568853](https://github.com/CrispStrobe/CrispASR/actions/runs/42313568853)
used the upstream `qwen_fa_ultra` 0.1.1 package and pinned Ultra revision
`765dc1d714e4c80196782a5df5bfbcb3e1416736` on `samples/jfk.wav`. It captured
44 endpoint feature tensors, contexts, masks, logits and the final timings for
22 words. The checked-in fixtures contain captured logits, timings, selected
normalized features and the normalizer; they do not include neural-head weights,
audio or hidden states.

`qwen3-ultra-boundary-reference` checks the C++ stage against all 22 upstream
word intervals to 0.002 ms, plus analytic controls for uniform and extreme
logits, nonfinite inputs, overlap, short clips and invalid duration. This is a
CPU post-processing test; it does not establish model, language or GPU parity.

`qwen3-ultra-features-reference` compares 14,924 normalized feature values for
four captured endpoints against the same upstream run. It also tests silence,
invalid inputs and an analytic FFT impulse. NumPy computes its FFT in double
precision; the equivalent C++ stage does the same before converting magnitudes
to float, because a float FFT produced appreciable error in log-spectrum bands.

The CPU head stage was also compared against all 44 captured endpoint logits
using the pinned 1.6 MB local-head checkpoint (SHA256
`50a70d6ed969446770fb9dc1063f5f4e6e4c86eac05fd3efb637492407d0e2ac`).
The maximum absolute error across 572 logits was 0.00000287 on the local CPU.
The separate oracle test downloads that checkpoint and is not a model-free test.
The implementation is a CPU reference, not a GPU or product integration.

`models/convert-qwen3-ultra-head-to-gguf.py` validates the exact source
revision's configuration, normalizer and model hashes before writing an F32
sidecar. A local conversion round-trip preserved all 28 weight and normalizer
arrays byte-for-byte. Synthetic controls reject invalid inventory, shape,
nonfinite tensors and invalid normalizers.
