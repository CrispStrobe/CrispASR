# Qwen3 Ultra local-head port: post-processing stage

The optional local head is not yet available in CrispASR. This first port
stage implements only its posterior-mean offset decoder and interval repair;
it is not wired into the aligner. Feature extraction, hidden-state capture,
head inference, converter support and end-to-end acceptance remain open.

[Reference run 42313568853](https://github.com/CrispStrobe/CrispASR/actions/runs/42313568853)
used the upstream `qwen_fa_ultra` 0.1.1 package and pinned Ultra revision
`765dc1d714e4c80196782a5df5bfbcb3e1416736` on `samples/jfk.wav`. It captured
44 endpoint feature tensors, contexts, masks, logits and the final timings for
22 words. The checked-in synthetic test fixture contains only the captured
logits and times; it does not include the model weights, audio or hidden states.

`qwen3-ultra-boundary-reference` checks the C++ stage against all 22 upstream
word intervals to 0.002 ms, plus analytic controls for uniform and extreme
logits, nonfinite inputs, overlap, short clips and invalid duration. This is a
CPU post-processing test; it does not establish model, language or GPU parity.
