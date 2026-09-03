# MT3 port notes — phase 1 (converter + reference oracle)

Companion to [`mt3-feasibility.md`](mt3-feasibility.md) (the GO memo). That memo
answered *should we*; this one records *what exactly*, with every constant
line-cited against the upstream sources, so phase 2 (the ggml runtime) can be
written without re-reading Python.

**Status: phase 1 done.** Deliverables:

| File | What |
|---|---|
| `models/convert-mt3-to-gguf.py` | T5X (zarr + msgpack) → GGUF arch `mt3`. stdlib + numpy + msgpack + gguf only. |
| `tools/reference_backends/mt3.py` | numpy-only forward reference: mel → encoder → greedy decode → tie state machine → notes. |

Sources used, both pinned locally during this work:

* `github.com/magenta/mt3` @ depth-1 clone → `/mnt/volume1/tmp-overflow/mt3-src`
  (Apache-2.0). All `mt3/*.py:NNN` citations below refer to that tree.
* `gs://mt3/checkpoints/mt3` → `/mnt/storage/gguf-models/mt3-src/`
  (295 files, 171.6 MB, anonymous HTTPS, Apache-2.0).
* Oracle weights `mt3.pth` (kunato/mt3-pytorch, re-hosted at
  `huggingface.co/gudgud1014/MR-MT3`) → `/mnt/storage/gguf-models/mt3-oracle/`,
  sha256 `b8a3807e…5398f` (matches openmirlab/mt3-infer's
  `config/checkpoints.toml`).

Reproduce:

```bash
# 1. checkpoint (296 objects, anonymous, no gsutil)
curl -s 'https://storage.googleapis.com/storage/v1/b/mt3/o?prefix=checkpoints/mt3/&maxResults=1000'
#    then GET https://storage.googleapis.com/mt3/<name> for each object
#    -> /mnt/storage/gguf-models/mt3-src/mt3/

# 2. convert
python3 models/convert-mt3-to-gguf.py \
    --input  /mnt/storage/gguf-models/mt3-src/mt3 \
    --output /mnt/storage/gguf-models/mt3-f16.gguf

# 3. reference dump + self-checks (numpy only, ~40 s)
python3 tools/reference_backends/mt3.py \
    --audio <16 kHz wav> -o /mnt/volume1/tmp-overflow/mt3-ref \
    --max-segments 2 --decode-steps 48

# 4. optional: weight + encoder parity vs the PyTorch conversion (needs torch)
/mnt/volume1/miniconda/bin/python tools/reference_backends/mt3.py \
    --audio <wav> -o /mnt/volume1/tmp-overflow/mt3-ref \
    --torch-oracle /mnt/storage/gguf-models/mt3-oracle/mt3.pth
```

---

## 1. Extracted constants (line-cited)

### 1.1 Network

| Constant | Value | Source |
|---|---|---|
| `emb_dim` (d_model) | 512 | `mt3/gin/model.gin:51` |
| `num_heads` | 6 | `mt3/gin/model.gin:52` |
| `head_dim` | 64 (→ qkv width 384) | `mt3/gin/model.gin:55` |
| `mlp_dim` | 1024 | `mt3/gin/model.gin:56` |
| `num_encoder_layers` | 8 | `mt3/gin/model.gin:53` |
| `num_decoder_layers` | 8 | `mt3/gin/model.gin:54` |
| `mlp_activations` | `('gelu','linear')` = gated-GELU | `mt3/gin/model.gin:57` |
| `logits_via_embedding` | `False` (untied `logits_dense`) | `mt3/gin/model.gin:59` |
| `dropout_rate` | 0.1 (inference: off) | `mt3/gin/model.gin:58` |
| LayerNorm | **RMSNorm**, no mean, no bias, `eps=1e-6` | `mt3/layers.py:604-621` |
| Attention scale | **1.0 — no `1/sqrt(head_dim)`** | `mt3/layers.py:230-234` |
| Positions | **fixed sinusoidal absolute, added to embeddings** | `mt3/network.py:180` (enc), `:225` (dec) |
| `FixedEmbed.max_length` | 2048 | `mt3/layers.py:565` |
| Encoder input | `DenseGeneral(512)` on the raw mel frame; **no token embedding** | `mt3/network.py:174-179` |
| Encoder padding mask | all-ones — nothing is masked | `mt3/network.py:286-289` |
| GELU flavour | flax default `approximate=True` → tanh | `flax.linen.gelu`, applied at `mt3/layers.py:469` |

Two of these are load-bearing traps.

**(a) No relative attention bias.** `mt3/layers.py:164-355`
(`MultiHeadDotProductAttention`) never constructs one, and `network.py` never
passes `bias=`. The checkpoint contains **zero** `relative_attention_bias`
tensors. `src/t5_translate.cpp:220` (`t5_relative_position_bucket`) and its
`ggml_add(kq, pos_bias)` at line 713 must be branched off entirely for MT3.

**(b) No `1/sqrt(d)` rescale.** `mt3/layers.py:230-234` says so in a comment
("NOTE: T5 does not explicitly rescale the attention logits by
`1/sqrt(depth_kq)`! This is folded into the initializers"). `ggml_soft_max`
must be called with scale `1.0`. The converter writes
`mt3.attn_logit_scale = 1.0` so the runtime cannot get it from stack garbage
(cf. the `KvSelfAttnParams ap{}` lesson).

### 1.2 Sinusoidal position table

`mt3/layers.py:51-82`:

```
half         = features // 2                       # 256
scale_factor = -log(10000.0 / 1.0) / (half - 1)    # note: half - 1 == 255
div_term     = 1.0 * exp(arange(half) * scale_factor)
pe[:,      :half] = sin(position * div_term)
pe[:, half:2*half] = cos(position * div_term)
```

i.e. `div_term[i] = 10000 ** (-i / 255)`, first half sin / second half cos,
**not interleaved**, **not** the `d/2` denominator of the original Transformer
paper. Encoder positions are `arange(T)` (`network.py:171`), decoder positions
are `arange(T)` too (`network.py:214` — the `decoder_positions` argument is
overwritten and ignored). In cached single-step decode the same table is
indexed by an internal counter starting at −1 and pre-incremented
(`layers.py:589-596`), which is just "position = step index".

> **Trap.** kunato/mt3-pytorch (the only widely-used PyTorch MT3, vendored by
> openmirlab/mt3-infer as `mt3_infer/models/mt3_pytorch/t5.py:529-541`) uses
> `inv_freq = 1/(10000 ** (arange(0,d,2)/d))`, i.e. denominator `256`, not
> `255`. Its weights are bit-identical to Magenta's but **its positional
> encoding is not.** Any parity run against it must patch `inv_freq` first;
> `tools/reference_backends/mt3.py --torch-oracle` does, and
> `--torch-oracle-raw` shows the un-patched drift.

### 1.3 Spectrogram front end

| Constant | Value | Source |
|---|---|---|
| `sample_rate` | 16000 | `mt3/spectrograms.py:23` |
| `hop_width` | 128 → **125 fps** | `mt3/spectrograms.py:24,51-52` |
| `num_mel_bins` | 512 | `mt3/spectrograms.py:25` |
| `FFT_SIZE` (= frame_length) | 2048 | `mt3/spectrograms.py:28` |
| `MEL_LO_HZ` | 20.0 | `mt3/spectrograms.py:29` |
| **`hi_hz`** | **7600.0** | `mt3/spectral_ops.py:78` **default**, never overridden — `spectrograms.py:67-73` passes only `bins`/`lo_hz`/`overlap`/`fft_size`/`sample_rate` |
| `overlap` | `1 - 128/2048 = 0.9375` → `frame_step = 128` | `mt3/spectrograms.py:66`, `spectral_ops.py:42-47` |
| window | `tf.signal.stft` default = **Hann, periodic** | `mt3/spectral_ops.py:42-47` |
| `pad_end` | `True`; **no centering, no reflect pad** | `mt3/spectral_ops.py:35-47` |
| magnitude | `abs(stft)` — **magnitude, not power** | `mt3/spectral_ops.py:52-54` |
| mel matrix | `tf.signal.linear_to_mel_weight_matrix` — HTK mel, **no Slaney area norm**, **DC bin zeroed** | `mt3/spectral_ops.py:69-71` |
| log | `safe_log`: `where(x <= 0, 1e-5, x)` then **natural** `log` | `mt3/spectral_ops.py:29-32` |

Four things `src/core/mel.h` does *not* currently do and that a naive port gets
wrong:

1. **`hi_hz = 7600`, not Nyquist.** The memo omitted this; it is a default
   inherited from DDSP.
2. **The DC spectrogram bin is dropped.** TF's `bands_to_zero = 1` — row 0 of
   the (1025 × 512) filterbank is all zeros. librosa keeps it.
3. **Triangles are unnormalised** (peak exactly 1.0). librosa's default
   `norm='slaney'` divides by bandwidth; that is a different filterbank.
4. **`safe_log` clamps only non-positive values.** Magnitudes in `(0, 1e-5)`
   pass through and produce values *below* `log(1e-5) = -11.5129`. A C++ port
   that floors at `eps` is wrong. Observed on the phase-1 test signal:
   `mel.min() = -13.2797`.

TF's HTK mel is `1127.0 * ln(1 + f/700)` (equivalently `2595 * log10(1+f/700)`),
edges from `linspace(mel(lo), mel(hi), num_mel_bins + 2)` framed `(3, 1)`.
`tools/reference_backends/mt3.py:linear_to_mel_weight_matrix` is a direct port;
its self-check confirms shape `(1025, 512)`, zeroed DC row, peak ≤ 1.0, and a
unit partition over bins 3..968 (max deviation 5.6e-4).

### 1.4 Segmentation

`mt3/gin/mt3.gin:4` — `TASK_FEATURE_LENGTHS = {'inputs': 256, 'targets': 1024}`.

```
audio  ──zero-pad to a multiple of 128──►  frames of 128 samples
frames ──chunk by inputs_length=256────►  segments of 32768 samples = 2.048 s
segment ──flatten──► tf.signal.stft ────►  256 × 1025 → mel → 256 × 512
```

**The STFT is computed per segment, on the flattened 32768 samples — not once
over the whole file** (`mt3/preprocessors.py:614-618`
`compute_spectrograms(flatten_frames(ex['inputs']))`, applied after the
`split_tokens_to_inputs_length` chunking). With `pad_end=True` this yields
exactly `ceil(32768/128) = 256` frames per full segment, each frame reading
`[i*128, i*128+2048)` of the segment's own samples and zero beyond its end.
A whole-file STFT would give different values at every segment boundary. The
ismir2021 piano model uses `inputs: 512` = 4.096 s (`mt3/gin/ismir2021.gin:4`).

Segment start time (colab `music_transcription_with_transformers.ipynb`,
`_preprocess`):

```
start_time  = first_frame_index / 125.0
start_time -= start_time % (1 / steps_per_second)     # floor to 10 ms
```

**The final short segment is zero-padded to 256 rows *in the mel domain*.**
seqio pads the `inputs` feature to `sequence_length['inputs']` after the
preprocessors have run, so the padding is literal `0.0` mel rows — which is
log-magnitude 0, i.e. magnitude 1.0, *not* silence — and
`mt3/network.py:286-289` deliberately leaves them unmasked ("we don't actually
mask out any input positions, letting the model potentially attend to the zero
vector used as padding"). The encoder is therefore always run at T = 256.
Observed effect on the phase-1 test file: the 113-frame tail segment goes from
2 tokens / 0 invalid (truncated) to 9 tokens / 5 invalid (padded, correct).

So segment 1 starts at `256/125 = 2.048 → 2.04`, segment 2 at `4.096 → 4.09`,
segment 3 at `6.144 → 6.14`. **This floor is not cosmetic** — it is the
`start_time` that all of that segment's event times are offset by, and it is
also the `max_decode_time` clamp applied to the *previous* segment
(`mt3/metrics_utils.py:106-108`).

### 1.5 Vocabulary and event codec

`mt3/vocabularies.py:119-140` `build_codec()` supplies the ranges;
`mt3/event_codec.py:57-59` forces `shift` to be first and to start at id 0.
For `NUM_VELOCITY_BINS = 1` (`mt3/gin/mt3.gin:6`):

| # | type | value range | codec ids | **token ids** (= codec + 3) | source |
|---|---|---|---|---|---|
| 0 | `shift` | 0 … 1000 | 0 … 1000 | 3 … 1003 | `event_codec.py:57-59`, `vocabularies.py:137-139` |
| 1 | `pitch` | 0 … 127 | 1001 … 1128 | 1004 … 1131 | `vocabularies.py:122-123` |
| 2 | `velocity` | 0 … 1 | 1129 … 1130 | 1132 … 1133 | `vocabularies.py:125` |
| 3 | `tie` | 0 … 0 | 1131 | 1134 | `vocabularies.py:129` |
| 4 | `program` | 0 … 127 | 1132 … 1259 | 1135 … 1262 | `vocabularies.py:130-131` |
| 5 | `drum` | 0 … 127 | 1260 … 1387 | 1263 … 1390 | `vocabularies.py:132-133` |

* `num_classes = 1388` (`event_codec.py:64-66`).
* Specials: `0 = PAD`, `1 = EOS`, `2 = UNK` (`vocabularies.py:153,159,163`),
  prepended → model token id = codec id + 3 (`vocabularies.py:186-194`).
* `extra_ids = 100` (`t5.data.DEFAULT_EXTRA_IDS`, `vocabularies.py:145`).
* Padded embedding table: `128 * ceil((3 + 1388 + 100)/128) = 128 * 12 = 1536`
  (`vocabularies.py:280-282`) — matches `token_embedder.embedding` `[1536,512]`.
* Decoding an id back (`vocabularies.py:211-219`): `EOS → -1`,
  `id < 3 or id >= 1491 → -2 (invalid)`, else `id - 3`. Note **1491, not 1536**
  — the 45 padding slots above the extra_ids are invalid, and so are the 100
  extra_ids themselves.
* `shift` is **absolute within the segment**, not a delta:
  `mt3/run_length_encoding.py:397-408` accumulates `cur_steps` across
  *consecutive* shift tokens and resets it to 0 on any non-shift event, then
  recomputes `cur_time = start_time + cur_steps / 100`.

The ismir2021 piano variant differs: `NUM_VELOCITY_BINS=127`,
`PROGRAM_GRANULARITY='flat'`, `USE_TIES=False` (`mt3/gin/ismir2021.gin:6-9`) →
`num_classes = 1514` → padded vocab **1664**. The converter takes
`--variant ismir2021` for it.

---

## 2. Checkpoint inventory

### 2.1 Physical format (verified, no JAX involved)

`gs://mt3/checkpoints/mt3` = 296 objects = 1 directory marker + 1 `checkpoint`
+ 147 zarr directories × 2 files. **189 parameters total**, in two places:

* **147 zarr v2 directories** — `<param>/.zarray` (JSON:
  `{"chunks":[…],"compressor":{"id":"gzip","level":1},"dtype":"<f4",
  "order":"C","shape":[…],"zarr_format":2}`) plus one whole-array chunk `0.0`
  that is a raw gzip stream of C-order `<f4`.
  `gzip.decompress` + `np.frombuffer` + `reshape` is the entire decoder.
* **42 inline parameters** — every RMSNorm `scale`, shape `(512,)`. They live in
  the 894 205-byte msgpack `checkpoint` under `/optimizer/target/…` as flax
  `ExtType(code=1)` whose payload is itself msgpack `[shape, dtype, bytes]`.
  A 6-line `ext_hook` reads them; no `flax` import.

42 = 8 encoder layers × 2 + 8 decoder layers × 3 + `encoder_norm` +
`decoder_norm`.

> **Correction to `mt3-feasibility.md` §3**: it says "51 LayerNorm scales". The
> real count is **42**, and the totals are 147 zarr + 42 inline = 189
> parameters (not 147). Everything else in §3 held up exactly.

In `/optimizer/target/`, a zarr-backed parameter appears as a TensorStore spec
`dict` (has `driver` + `kvstore` keys) rather than an ndarray — that is how the
converter tells the two storage classes apart while walking the tree.

### 2.2 Tensor map: flax name → GGUF name

Flax `DenseGeneral` kernels are stored **`[in, out]`** (`mt3/layers.py:405-415`
builds `kernel_param_shape = (prod(in_axes), prod(features))`), while
`ggml_mul_mat(W, x)` contracts `W->ne[0]` and therefore wants PyTorch's
**`[out, in]`**. Every kernel is transposed by the converter; `token_embedder`
is an `Embed` table (`layers.py:509-514`), already `[vocab, d_model]`, and is
**not** transposed.

`N` = 0…7 for both stacks. 190 tensors written (189 params + the generated
position table); 147 F16, 43 F32.

| GGUF name | flax source | checkpoint shape | GGUF shape | dtype |
|---|---|---|---|---|
| `pos_embd.weight` | *generated* from `layers.py:51-82` | — | `(2048, 512)` | F32 |
| `token_embd.weight` | `decoder.token_embedder.embedding` | `(1536, 512)` | `(1536, 512)` | F16 |
| `lm_head.weight` | `decoder.logits_dense.kernel` | `(512, 1536)` | `(1536, 512)` | F16 |
| `enc.inp_proj.weight` | `encoder.continuous_inputs_projection.kernel` | `(512, 512)` | `(512, 512)` | F16 |
| `enc.blk.N.attn_rms.weight` | `encoder.layers_N.pre_attention_layer_norm.scale` | `(512,)` | `(512,)` | F32 |
| `enc.blk.N.attn_q.weight` | `encoder.layers_N.attention.query.kernel` | `(512, 384)` | `(384, 512)` | F16 |
| `enc.blk.N.attn_k.weight` | `encoder.layers_N.attention.key.kernel` | `(512, 384)` | `(384, 512)` | F16 |
| `enc.blk.N.attn_v.weight` | `encoder.layers_N.attention.value.kernel` | `(512, 384)` | `(384, 512)` | F16 |
| `enc.blk.N.attn_o.weight` | `encoder.layers_N.attention.out.kernel` | `(384, 512)` | `(512, 384)` | F16 |
| `enc.blk.N.ffn_rms.weight` | `encoder.layers_N.pre_mlp_layer_norm.scale` | `(512,)` | `(512,)` | F32 |
| `enc.blk.N.ffn_gate.weight` | `encoder.layers_N.mlp.wi_0.kernel` | `(512, 1024)` | `(1024, 512)` | F16 |
| `enc.blk.N.ffn_up.weight` | `encoder.layers_N.mlp.wi_1.kernel` | `(512, 1024)` | `(1024, 512)` | F16 |
| `enc.blk.N.ffn_down.weight` | `encoder.layers_N.mlp.wo.kernel` | `(1024, 512)` | `(512, 1024)` | F16 |
| `enc.final_rms.weight` | `encoder.encoder_norm.scale` | `(512,)` | `(512,)` | F32 |
| `dec.blk.N.attn_{rms,q,k,v,o}` | `decoder.layers_N.{pre_self_attention_layer_norm.scale, self_attention.{query,key,value,out}.kernel}` | as encoder | as encoder | F32/F16 |
| `dec.blk.N.cross_{rms,q,k,v,o}` | `decoder.layers_N.{pre_cross_attention_layer_norm.scale, encoder_decoder_attention.{query,key,value,out}.kernel}` | as encoder | as encoder | F32/F16 |
| `dec.blk.N.ffn_{rms,gate,up,down}` | `decoder.layers_N.{pre_mlp_layer_norm.scale, mlp.{wi_0,wi_1,wo}.kernel}` | as encoder | as encoder | F32/F16 |
| `dec.final_rms.weight` | `decoder.decoder_norm.scale` | `(512,)` | `(512,)` | F32 |

The names deliberately match `models/convert-madlad-to-gguf.py` /
`src/t5_translate.cpp`, except:

* `enc.inp_proj.weight` and `pos_embd.weight` are new;
* `enc.rel_bias.weight` / `dec.rel_bias.weight` **do not exist** and must not be
  required by the loader.

### 2.3 Converter run

```
$ python3 models/convert-mt3-to-gguf.py \
    --input  /mnt/storage/gguf-models/mt3-src/mt3 \
    --output /mnt/storage/gguf-models/mt3-f16.gguf
MT3 -> GGUF   variant=mt3
  codec classes : 1388  (+3 special +100 extra_ids -> padded vocab 1536)
  checkpoint    : 147 zarr params + 42 inline msgpack params = 189 total
  tensors: 190  (F16 147, F32 43)
  wrote /mnt/storage/gguf-models/mt3-f16.gguf  (96.0 MB)
```

Zero unmapped parameters, zero missing. **96.0 MB F16** — the memo's ~120 MB
estimate was high because it counted the F32 checkpoint's optimiser slots.

GGUF metadata written (arch `mt3`): the network hparams, the full spectrogram
config, the segmentation lengths, and the codec range layout as three parallel
arrays `mt3.codec.event_{types,min_values,max_values}` so the C++ codec is data
driven rather than a hard-coded table. Explicitly recorded so the runtime
cannot pick the wrong branch:

```
mt3.pos_embed                    = "sinusoidal"
mt3.use_relative_attention_bias  = 0
mt3.attn_logit_scale             = 1.0
mt3.spectrogram.mel_scale        = "htk"
mt3.spectrogram.mel_norm         = "none"
mt3.spectrogram.magnitude        = "magnitude"
mt3.spectrogram.center           = 0
mt3.spectrogram.pad_end          = 1
mt3.spectrogram.log_eps          = 1e-5
```

---

## 3. Reference oracle: what exists, what does not

### 3.1 The oracle problem, precisely

There is **no runnable Magenta MT3**. openmirlab/mt3-infer — the packaged
PyTorch toolkit the feasibility memo flagged as "the ready-made oracle" — states
in its own README (lines 100-101, 116-118) that *"the original Magenta MT3
(JAX/Flax) backend is not supported… excluded due to dependency conflicts
between the JAX/Flax/t5x/seqio stack and the PyTorch ecosystem"*. Its three
backends are MR-MT3, MT3-PyTorch and YourMT3.

That leaves `mt3.pth` (kunato/mt3-pytorch, re-hosted on HF, sha256
`b8a3807e…` — the file openmirlab/mt3-infer pins for its `mt3_pytorch`
profile), which **is** a conversion of this exact Magenta checkpoint.
`tools/reference_backends/mt3.py --torch-oracle` runs the comparison:

```
weights: compared 189 tensors, 0 mismatches, max|abs diff| = 0.000e+00
```

**All 189 parameters bit-exact.** The converter's zarr + msgpack decode path
and its `[in,out] → [out,in]` transposes are therefore proven correct against a
fully independent third-party conversion.

What that `.pth` is **not** is a drop-in forward oracle: its
`FixedPositionalEmbedding` (`t5.py:529-541`) uses the `d/2` denominator instead
of Magenta's `d/2 - 1`. The dumper reports this directly:

```
oracle inv_freq == Magenta 10000**(-i/255): False
oracle inv_freq == kunato  10000**(-i/256): True
```

With the table corrected, a torch re-implementation of the encoder (torch ops,
torch softmax, `F.gelu(approximate="tanh")`) agrees with the numpy reference:

```
enc_out [pos patched to Magenta]  cos = 1.000000000   max|abs diff| = 4.86e-06
enc_out [pos RAW (kunato)      ]  cos = 0.995751164   max|abs diff| = 4.99e+00
```

**Gate passed: cos 1.000000000 ≫ 0.999.** The second line is the size of
kunato's off-by-one: cos 0.9958 — high enough to pass a sloppy parity check,
low enough to change the decoded notes. That is exactly the failure mode
§4.5.3 warns about.

The vendored `mt3_pytorch/t5.py` module itself is **not** importable on this
box: it does `from transformers.models.t5.modeling_t5 import Seq2SeqLMOutput,
BaseModelOutput, …` and the installed transformers is 5.13.0.dev0, which moved
those symbols. Hence the direct-from-state-dict torch forward.

### 3.2 GGUF round-trip

Independently of the oracle, every written tensor was read back out of
`mt3-f16.gguf` and compared against a fresh checkpoint decode:

```
GGUF round-trip: 189 tensors, 0 bad, worst relative err 4.643e-04  (F16 rounding)
pos_embd.weight vs closed form: max err 0.0                        (F32, exact)
```

### 3.3 What the numpy reference does instead

`tools/reference_backends/mt3.py` is a from-source numpy re-implementation:
tf.signal-compatible mel front end, 8-layer encoder, non-cached decoder,
greedy search, and the tie-section note assembly. It dumps
`audio_segment0 / mel / mel_all / enc_input / enc_out / logits_step0 /
logits_prefix` as `.npy` plus a `ref.gguf` for `crispasr-diff`, and
`tokens.json` with per-segment tokens, decoded events and the final note list.

**Functional validation.** On a synthetic 5 s signal — four harmonic-stack
notes at C4/E4/G4/C5 (261.63 / 329.63 / 392.00 / 523.25 Hz) starting at
0.0 / 0.6 / 1.2 / 1.8 s, then a C-major triad at 2.6 s — the reference produced:

```
segment 0  tokens [1139, 1064, 1134, 63, 1133, 1068, 123, 1071, 183, 1076, 1]
  t=0.00  program 4        (Electric Piano 1)
  t=0.00  pitch 60         (in tie section -> rejected, see below)
  t=0.00  tie
  t=0.60  velocity 1 ; pitch 64
  t=1.20  pitch 67
  t=1.80  pitch 72
segment 1  (start_time 2.04)
  t=2.04  program 4 ; tie section: pitch 67, 72
  t=2.59  velocity 0 -> note-offs ; velocity 1 -> pitch 60,64,67,72,76,79
notes after tie assembly: 9
```

Pitches 60/64/67/72 at 0.0/0.6/1.2/1.8 s are the ground truth, exactly. A wrong
mel filterbank, a wrong sinusoidal denominator, a spurious `1/sqrt(d)`, an
erf-GELU, or a transposed kernel would each destroy this. It is a stronger
signal than any cosine number on an encoder activation.

**Self-checks** (23/23 pass; run by default, `--no-self-check` to skip): codec
range endpoints and round-trip, padded vocab = 1536, sinusoidal table against
the `10000**(-i/255)` closed form at positions 0/1/37/255/2047, filterbank
shape / zeroed DC / unnormalised peak / unit partition, mel reproducibility and
`mel_all[0:256] == mel`, and `safe_log`'s exact clamp semantics.

**Known gaps for phase 2** (do not declare parity without closing them):

* No bit-exact comparison against real TensorFlow `tf.signal` — TF is not
  installed and does not fit this box. The mel is a source port validated by
  its algebraic properties and by end-to-end transcription accuracy, not by a
  numeric diff against TF.
* The reference decoder is non-cached (`O(T²)` re-run of the prefix). Correct,
  but it does not exercise the KV-cache semantics the C++ runtime will use.
* Velocity is only 2 bins in the `mt3` variant, so velocity parity is trivially
  satisfied here and will not be exercised until someone converts `ismir2021`.

---

## 4. Phase-2 plan: the ggml runtime

### 4.1 What transfers from `src/t5_translate.cpp` (1526 lines)

Roughly 80 % of the graph is reusable as-is:

| Reusable | Where |
|---|---|
| `t5_rms_norm` | `:268` — identical (RMSNorm, eps from hparams) |
| encoder self-attention block | `:695-740` — `ggml_soft_max` is already unscaled (scale 1.0), which is what MT3 needs |
| gated-GELU FFN | `:734-737` (enc), `:940-943` (dec). `ggml_gelu` is ggml's tanh approximation = flax's `approximate=True`. **Do not** switch to `ggml_gelu_erf`. |
| decoder self/cross-attention, KV cache, cross-KV precompute | `:556-624`, `:760-818`, `:820-976` |
| greedy step loop | `:1009-1045` |
| `#333` diff-capture stage naming | `:625-647` — reuse verbatim for `crispasr-diff mt3` |

### 4.2 What has to change

1. **Encoder input head** — replace
   `cur = ggml_get_rows(ctx0, m.shared_embed, inp)` (`:679`) with
   `cur = ggml_mul_mat(ctx0, m.enc_inp_proj, mel)` where `mel` is an
   `(512, T)` F32 input tensor. `MelsTime` *is* the layout ggml wants here —
   do not transpose (see the `feedback_mel_no_transpose` lesson). Always run at
   `T = mt3.inputs_length`, zero-padding a short final segment (§1.4).

2. **The `FixedEmbed` branch.** Gate on the GGUF key:

   ```cpp
   const bool sinusoidal = (hp.pos_embed == "sinusoidal");   // mt3.pos_embed
   ```

   * **Encoder** (`build_encoder_graph`): if sinusoidal, skip the whole
     `enc_pos_bucket` input, the `ggml_get_rows(enc_rel_bias, …)` and the
     `ggml_add(kq, pos_bias)` at `:713`; instead add
     `ggml_get_rows(m.pos_embd, arange(T))` right after the input projection.
   * **Decoder** (`build_decoder_graph`): same, with row index `offset + i`,
     matching `layers.py:589-596`'s pre-incremented counter.
   * `bind_model` (`:345`) must stop requiring `enc.rel_bias.weight` /
     `dec.rel_bias.weight` — they are absent from the MT3 GGUF. Today it
     silently binds `nullptr`; the graph builders would then dereference it.
   * `run_encoder` (`:978-1007`) and `run_decoder_step` (`:1009-1045`) must not
     compute or upload buckets on the sinusoidal path.

   The `pos_embd.weight` table is shipped in the GGUF (2048 × 512 F32, 4 MB) so
   there is nothing to re-derive and no chance of reproducing kunato's
   off-by-one.

3. **Attention scale.** Read `mt3.attn_logit_scale` and pass it explicitly to
   `ggml_soft_max_ext` / keep `ggml_soft_max`. Never leave it implicit.

4. **Mel front end.** `src/core/mel.h` needs a tf.signal-compatible variant:
   `hi_hz = 7600`, HTK mel, **no** Slaney norm, **DC bin zeroed**, magnitude
   (not power), Hann *periodic*, `pad_end` (right-zero) rather than centered,
   and `safe_log` clamping only `x <= 0`. Gate behind
   `CRISPASR_MT3_*` env vars per the repo's env-gating rule.

5. **New: event codec** (~120 lines). Data-driven from
   `mt3.codec.event_{types,min_values,max_values}`; `id ↔ (type, value)` with
   the `+3` special-token offset and the `>= 1491 → invalid` rule. Unit-test it
   against `tools/reference_backends/mt3.py`'s `Codec` over all 1536 ids.

6. **New: tie state machine + cross-segment assembly** — §4.3.

7. **New: multi-track MIDI writer** (128 programs + drum channel 10). §251
   already wants one in `core/`; `src/piano_transcription.cpp` emits its own
   single track and should eventually be migrated onto it.

8. Registry entry, `--task transcribe-music --model mt3`, 12-point checklist,
   HF upload of F16 + Q4_K.

### 4.3 The segment-stitching state machine (the memo's #1 risk)

This is the part that produces plausible-looking, wrong MIDI when it is subtly
off. Spelled out from `mt3/note_sequences.py` and `mt3/metrics_utils.py`:

**Driver** (`metrics_utils.py:92-116`) — one decoding state for the *whole
file*, never reset between segments:

```
sort segments by start_time
for i, seg in enumerate(segments):
    begin_tied_pitches_section(state)                       # :99
    max_decode_time = segments[i+1].start_time if i+1 < n else None   # :106-108
    decode_events(state, seg.tokens, seg.start_time, max_decode_time) # :110-111
notes = flush_note_decoding_state(state)                    # :117
```

`max_decode_time` is what stops a segment's trailing predictions from bleeding
into the next segment's territory; when exceeded, `decode_events` **breaks out
of the loop** and counts every remaining token as dropped
(`run_length_encoding.py:409-411`).

**Per-segment entry** (`note_sequences.py:390-393`):
`tied_pitches = {}`, `is_tie_section = True`. Note what is *not* reset:
`current_time`, `current_velocity`, `current_program`, `active_pitches`, and the
accumulated note list. `current_program` carrying across a segment boundary is
intentional.

**Time** (`run_length_encoding.py:395-421`): `cur_steps` accumulates over
consecutive `shift` tokens and is zeroed by any non-shift event;
`cur_time = start_time + cur_steps / 100`. A non-monotonic time raises
(`note_sequences.py:320-322`) and the event is counted invalid, **not** fatal.

**Tie semantics** (`note_sequences.py:313-387`):

| state | event | action |
|---|---|---|
| `is_tie_section` | `pitch p` | `(p, current_program)` **must** already be in `active_pitches`, else `ValueError` → invalid. Must not already be in `tied_pitches`, else `ValueError`. Otherwise add to `tied_pitches`. |
| `is_tie_section` | `program v` | `current_program = v` — the tie list is `(program, pitch)` **pairs**, re-declared with interleaved `program` tokens |
| `is_tie_section` | `tie 0` | **end of section**: every `(p, prog)` in `active_pitches` that is *not* in `tied_pitches` is closed at `current_time`; `is_tie_section = False` |
| normal | `velocity 0` then `pitch p` | note-off: pop `(p, prog)` from `active_pitches`, emit note `[onset_time, time)` |
| normal | `velocity ≥ 1` then `pitch p` | note-on. If `(p, prog)` is already active, close the old note at `time` first and start a new one (`:347-355`) |
| normal | `drum p` | one-shot: emit `[time, time + 0.01)`, `is_drum = True`, program ignored. Velocity 0 is an error here (`:360-361`). |
| normal | `program v` | `current_program = v` |

Also: `_add_note_to_sequence` enforces `end = max(end, start + 0.01)`
(`MIN_NOTE_DURATION`, `:306`), and `flush_note_decoding_state` (`:396-408`)
first pushes `current_time` past every dangling onset + 0.01, then closes all
still-active notes there, then calls `assign_instruments` (`:72-84`): programs
get instrument numbers in first-appearance order, **skipping 9**, and every drum
note gets instrument 9 — i.e. the General MIDI channel-10 convention.

**Invalid-token recovery is part of the spec, not an error path.** Upstream
catches `ValueError` per event, increments a counter, logs, and *continues*
(`run_length_encoding.py:416-422`). The phase-1 reference run hits this
legitimately — segment 0's tie section declares a pitch that was never active,
and segment 1 declares two. The C++ must reproduce the same
skip-and-continue behaviour, including the fact that a rejected tie declaration
means that pitch is then **closed** by the `tie` token (it is absent from
`tied_pitches`). Getting this wrong is exactly how you get correct-looking
notes with wrong durations.

### 4.4 Parity gates for phase 2

Per the diff-harness discipline, in order:

1. `crispasr-diff mt3` on `mel` — cos ≥ 0.9999 vs the numpy reference, and
   `max|abs|` reported. The mel is the highest-risk stage; do not average over
   segments, compare segment 0 exactly.
2. `enc_input`, then per-layer encoder stages, then `enc_out` — cos ≥ 0.999.
3. `logits_step0` — cos ≥ 0.999 **and** identical `argmax`.
4. `logits_prefix` — identical greedy token sequence for ≥ 64 steps on segment
   0, over at least three different inputs.
5. **The real gate: note-level F1** — the decoded note list
   (`tokens.json:notes`) must match the C++ output exactly on a multi-segment
   file, including durations and programs. Cosine parity on 1–3 will pass early
   and mean nothing about the state machine.
6. ASR-style round trip: render the emitted MIDI and re-transcribe, per the
   repo's "validate by round trip" rule.

### 4.5 Three riskiest points

1. **Tie-section / cross-segment note assembly** (§4.3). Everything about MT3's
   output quality lives here, it is invisible to cosine metrics, and the
   invalid-event recovery path is a documented, *exercised* behaviour rather
   than an edge case. Budget the debugging here.
2. **The mel front end has no bit-exact oracle.** TF is not installable on this
   box, so the tf.signal port is validated by algebraic properties and by
   end-to-end transcription, not by a numeric diff. The four divergences from
   librosa in §1.3 (hi_hz 7600, zeroed DC bin, unnormalised triangles, one-sided
   `safe_log` clamp) are each individually silent-but-wrong, and a wrong mel
   will still transcribe *something*.
3. **The `FixedEmbed` branch inside `t5_translate.cpp`.** It is invasive: the
   bucket tensor is an *input* to both graphs and is uploaded by `run_encoder` /
   `run_decoder_step`, and `bind_model` currently treats the rel-bias tensors as
   present. A half-done branch yields a model that loads, runs, and is wrong
   only at positions > 0 — which is precisely the failure mode kunato's
   PyTorch port shipped with (§1.2). Add a load-time assertion that
   `mt3.use_relative_attention_bias == 0` implies no `*.rel_bias.weight` tensor
   was bound, and vice versa.

Secondary, non-blocking: greedy decode at up to 1024 tokens × `ceil(len/2.048s)`
segments (~117 segments for a 4-minute track) is inherently serial. Expect
single-digit× real time on CPU.
