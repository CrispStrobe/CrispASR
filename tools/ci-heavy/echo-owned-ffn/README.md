# Echo9B FFN attribution on owned native inputs

`echo_owned_ffn_run.py` uses the existing public native stage API to capture
JFK/ZH prefill OFF/ON/OFF with the failed calibrated plain-Q4 decoder. Existing
stages, prompt IDs and logits must be byte-identical across capture modes.
OFF/ON/OFF toggles only FFN capture: the existing block-stage observer remains
enabled identically in every arm, with a prefill setup guard and negative control.
The original F16 primary file keeps its metadata unchanged; its decoder companion
symlink explicitly points to the physically audited Q4 bytes.

After the native child exits, the worker lazily reads one original BF16 MLP
layer at a time, constructs the verified released `Qwen3_5MLP` class and loads
F32 coefficients. It replays that class on the exact captured native `mlp_in`,
with original OFF/ON/OFF output identity. Six matching vectors retain cosine,
both norms, relative L2 and maximum absolute error. Additional original
`down_proj(native_down_in)` and `act_fn(native_gate)*native_up` measurements
separate projection error from upstream activation drift. Tensor ranks match
the original MLP's `[1,1,width]` calls.

This is component attribution. It does not reconstruct the whole Python model,
compare independent native/original histories, or establish decoded speech,
Q4 acceptance, GPU portability or performance. A first cumulative block crossing
is not the responsible FFN projection. Do not change thresholds or promote the
failed decoder from these measurements.

The hosted worker verifies hashes, source/class provenance, physical types,
owned-input ledgers, ABI exports and the actual compiled library. It checkpoints
new arrays and receipts publicly before later phases. Weights and builds remain
in `HEAVY_SCRATCH`; proof files go in `HEAVY_OUT` and the immutable fixture prefix
`index-echo-9b/owned-ffn-20261017/v2`. The original whole 9B Python model is never
constructed. The three source weight files total about24.3GB; a disk guard adds
8GiB headroom before downloading. One original MLP's F32 parameters occupy604MB.

The standalone `selftest.py` uses the actual released class and tiny BF16
safetensors, with scale, changed-input, changed-source and malformed-shape
negatives. It requires the pinned5.6.0 Transformers distribution, Torch,
NumPy and safetensors. It lives with the hosted worker so ordinary repository
test discovery does not require that pinned reference environment.

Run the worker using `heavy-cpu.yml` with script
`tools/ci-heavy/echo_owned_ffn_run.py`, Linux runner and pinned CPU dependencies.
The workflow selects the existing Index-Echo publishing credential for this
script; preflight proves public fixture write access before builds or downloads.
A prepared/dispatched job is not evidence of completed model attribution.

The frozen v1 worker967a62e0 failed on its first OFF arm because both observers
were disabled while block tensors were required. Its failed partial proof at
v1 remains intact. The v2 repair changes only observer setup and retains every
native/source identity and numerical gate; runtime code and recipe are unchanged.
