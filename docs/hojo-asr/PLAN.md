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

### The instrument bugs (worth more than the code bugs)

* **A reachable arm is not an affordable arm.** My first CPU fix cast the
  speech embeddings to bf16 so the BF16 decoder would accept them — correct,
  cheap in memory, and it put the whole Qwen3-4B decode on PyTorch's CPU bf16
  path, which has no fast kernel without AMX. Costed against the real shapes:
  1.11 TFLOP prefill + 7.06 TFLOP of beam-4 decode is ~2 min/utterance at
  ~60 GFLOPS (f32/oneDNN) and ~45 min at ~3 GFLOPS. Two arms: 4 min vs 90. A
  Kaggle run sat at 90 minutes and the tempting explanation was a cold ccache.
  Fix: f32 decoder on CPU, plus `decode_rate_probe()`, which times one prefill
  and one step and **prints the projected total before the expensive call**.
* **An A/B whose arms might be identical proves nothing.** The tiled-vs-untiled
  encoder check could not distinguish "tiling is exact" from
  "`CRISPASR_HOJO_ASR_CONV_TILE` did nothing" — both give max|delta| = 0. The
  encoder now prints `conv_schedule=tiled|untiled …` and the kernel refuses to
  score the comparison unless the two arms report different schedules.
* **The default decode was a hang, not a recipe.** I made beam 4 the default
  for fidelity to `config.yaml`. `core_beam_decode` rebuilds each beam's KV by
  replaying its entire suffix every step, so beam search is O(B*T^2)
  token-forwards where greedy is O(T): 80,400 vs 200 for a 9-second clip on
  this 4.4 B decoder, roughly four hours vs three minutes. The Kaggle
  end-to-end step ran four of those and would have blown the 12 h cap. Greedy
  is now the default, `-bs 4` selects the checkpoint's recipe, the runtime
  prints the projected forward count whenever beam > 1, and the reference emits
  a matched `generated_text_greedy` so the roundtrip compares like with like.
  Proper fix (a follow-up): `run_with_probs_branched` + `hojo_asr_kv_save` /
  `kv_restore`, which is O(B*T).
* **A cosine is the wrong instrument for the LM stage.** With an f32 reference
  and F16 C++ weights the logits cosine is precision-bound by construction. The
  arm now also reports `compare_argmax` top-1 agreement, which answers the
  question the cosine cannot.

### Not done / next

* **Per-stage parity numbers and the ASR roundtrip are NOT in yet.** The port
  is unvalidated until the kernel reports them; nothing here should be read as
  a parity claim.
* `docs/feature-matrix.md` must be regenerated from a build of **main after
  this branch merges**, never from this branch. The generator reads a live
  `crispasr --list-backends-json`, so a binary built here — from a base that
  predates breeze-tts-2 (#412) and voxtral-f16 (#437) — would silently delete
  their rows. The kernel only checks whether hojo-asr *appears*; it
  deliberately does not export the file.
* No local validation arm exists: q4_k is ~4.4 GB (tied embedding + audio
  tower stay F16) against 8 GB of shared VPS RAM. The diff loop lives on
  Kaggle. A q8_0 audio tower is the obvious size follow-up once parity holds.
