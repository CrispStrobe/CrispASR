# §438 — Hojo-ASR-Multi-V1 port

## NOW — active work

**Branch** `feat/438-hojo-asr` (pushed; no PR). **Kernel**
`chr1s4/crispasr-hojo-asr-438` (`tools/kaggle/hojo-asr-438/`).

### The two gating questions, answered before any runtime was written

**1. Where do the encoder weights live?** Inside
`merged_full_model.safetensors`. The safetensors header (read by range request,
not inferred from the card) lists 1013 tensors totalling 11,956,785,668 bytes,
which matches the file size exactly:

| group | tensors | bytes | dtype |
|---|---|---|---|
| `speech_encoder.*` | 525 | 2.59 GB | F32 |
| `bottleneck.*` | 87 | 0.55 GB | F32 |
| `decoder_model.*` | 399 | 8.82 GB | BF16 |
| `ln_speech.*` | 2 | ~20 KB | F32 |

The bare `Qwen3-Omni-30B-A3B-Instruct/config.json` beside it is read for audio
hyper-parameters only — `hojo_asr_model.py` does
`AutoConfig.from_pretrained(encoder_path)` and then constructs
`ModifyQwen3OmniMoeAudioEncoder(audio_config)` with fresh weights, which
`load_state_dict(..., assign=True)` immediately overwrites from the merged
file. **No 30 B checkpoint is ever fetched.** The port is viable: ~5.2 B
params total.

**2. Does the adapter stack frames?** No — it is 1:1 in time.
`ConformerEncoder(2048, 2560, linear_units=640, num_blocks=2,
input_layer="linear")`: the input dim 2048 is exactly the encoder's own
`output_dim`, and `linear_units: 640` is the conformer FFN's inner width (the
`feed_forward.w_1` weights are `[640, 2560]`), not a 2:1 stack against the
1280-wide tower.

The "customized multi-frame acoustic fusion" the card advertises is the
encoder's conv stem, and its ordering is the one that matters:

```python
b, c, f, t = padded_embed.size()
padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
```

`(b,c,f,t) -> (b,t,c,f) -> (b,t,c*f)` — **channel-major, frequency-fastest**,
fusing 8 mel frames × all 128 mel bins (down to 16 after three stride-2 convs)
into one 1280-d frame via `conv_out: Linear(480*16=7680 -> 1280, no bias)`. The
C++ reproduces it as `permute(0,2,1,3)` on `(F, T, C)` then
`reshape_2d(F*C, T)`; getting it backwards is the fluent-but-wrong failure.

### Landed on the branch

| what | where |
|---|---|
| converter (BN folded, `pe` shipped verbatim, tied `lm_head` proven) | `models/convert-hojo-asr-to-gguf.py` |
| runtime | `src/hojo_asr.{h,cpp}` |
| frame schedule + decode transforms, unit-tested | `src/core/hojo_asr_frames.h`, `tests/test-hojo-asr-frames.cpp` |
| CLI adapter + full checklist wiring | `examples/cli/crispasr_backend_hojo_asr.cpp` + 10 files |
| `crispasr-diff` arm | `examples/cli/crispasr_diff_main.cpp` |
| Python reference (drives the upstream PyPI package) | `tools/reference_backends/hojo_asr.py` |
| pipeline kernel | `tools/kaggle/hojo-asr-438/` |

### Bugs found before any model ran

* `feat_output_len` — Python's `//` floors, C's `/` truncates; they disagree
  exactly at `leave == 0`, so a direct transcription is one frame too long for
  every exact multiple of `n_window_infer` (30 s, 60 s, …). Found while
  extracting the schedule into a testable header.
* conv tile halo — zero-filling a shifted halo is NOT what the conv's padding
  does (it injects a literal zero vector per level; zeros as input give
  `gelu(bias)` at level 1 and propagate). Windows are now clamped to the real
  array so the tile's boundary sits where the full array's boundary sits.
* `run_encoder` interleaved the cached conv graph with the per-chunk
  transformer graph through one sched — the #215 use-after-free. Conv is now a
  complete phase 1.
* `pos_bias_u/v` are 2-D, so the converter's "2-D goes to F16" rule caught
  them; they are ADDED to an F32 Q and ggml's binbcast rejects F32 ⊕ F16.
* The reference cannot run on CPU as shipped (upstream relies on CUDA autocast
  to reconcile F32 speech embeddings with a BF16 decoder).

### Not done / next

* **Per-stage parity numbers and the ASR roundtrip are NOT in yet.** The port
  is unvalidated until the kernel reports them; nothing here should be read as
  a parity claim.
* `docs/feature-matrix.md` is generated from a built binary and is regenerated
  on the Kaggle worker; the result still has to be committed back.
* No local validation arm exists: q4_k is ~4.4 GB (tied embedding + audio
  tower stay F16) against 8 GB of shared VPS RAM. The diff loop lives on
  Kaggle. A q8_0 audio tower is the obvious size follow-up once parity holds.
