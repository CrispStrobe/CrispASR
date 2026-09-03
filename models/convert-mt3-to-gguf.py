#!/usr/bin/env python3
"""Convert the Magenta MT3 T5X checkpoint to GGUF (arch "mt3").

MT3 ("Multi-Task Multitrack Music Transcription", Gardner et al. 2022,
https://arxiv.org/abs/2111.03017) is a T5.1.1-flavoured encoder-decoder that
maps a log-mel spectrogram directly to a stream of MIDI-like note events.
Upstream weights live in a public, anonymously readable GCS bucket:

    gs://mt3/checkpoints/mt3        (multi-instrument, 171.6 MB, 296 objects)
    gs://mt3/checkpoints/ismir2021  (piano-only,       172.0 MB)

Fetch with plain HTTPS, no gsutil / credentials needed:

    curl -s 'https://storage.googleapis.com/storage/v1/b/mt3/o?prefix=checkpoints/mt3/&maxResults=1000'
    curl -s  https://storage.googleapis.com/mt3/checkpoints/mt3/<object>

Checkpoint format (verified, not guessed)
-----------------------------------------
The T5X checkpoint is *not* a monolith and needs neither JAX, t5x, seqio,
TensorStore nor TensorFlow to read:

  * 147 "large" parameters are each a **zarr v2 directory**:
        <param>/.zarray   JSON metadata: {"chunks":[...],"compressor":
                          {"id":"gzip","level":1},"dtype":"<f4",
                          "order":"C","shape":[...],"zarr_format":2}
        <param>/0.0       one gzip-level-1 deflate blob, C-order raw <f4
    -> gzip.decompress + np.frombuffer + reshape. That is the whole decoder.

  * 42 small parameters (all the RMSNorm `scale` vectors, shape (512,)) are
    stored **inline** in the 895 KB msgpack file `checkpoint`, under
    /optimizer/target/..., as flax msgpack ExtType(code=1) whose payload is
    itself msgpack: [shape, dtype_string, raw_bytes].
    -> a ~6-line ext_hook. No flax import needed.

  147 + 42 = 189 parameters total. (The GCS object count 296 =
  1 dir marker + 1 `checkpoint` + 147 zarr dirs x 2 files.)

Weight layout
-------------
Flax `DenseGeneral` kernels are stored **[in, out]** (and flattened for the
multi-axis cases), whereas ggml's `ggml_mul_mat(W, x)` contracts W->ne[0], i.e.
it wants the PyTorch **[out, in]** convention that `src/t5_translate.cpp`
already assumes. Every kernel is therefore transposed on the way out.

Attention has **no 1/sqrt(head_dim) rescale** -- mt3/layers.py:230-234 folds it
into the query initializer ("NOTE: T5 does not explicitly rescale the attention
logits by 1/sqrt(depth_kq)"). The converter records this as
`mt3.attn_logit_scale = 1.0` so the runtime cannot silently reintroduce it.

Positional encoding is **fixed sinusoidal absolute, added to the embeddings**
(mt3/network.py:180 encoder, :225 decoder; mt3/layers.py:51-82 initializer),
*not* T5 relative-attention buckets -- the checkpoint contains zero
`relative_attention_bias` tensors. The table is regenerated here bit-for-bit
from `mt3.layers.sinusoidal()` and written as `pos_embd.weight` (2048, 512) so
the runtime never has to re-derive it.

Usage
-----
    python3 models/convert-mt3-to-gguf.py \
        --input  /mnt/storage/gguf-models/mt3-src/mt3 \
        --output /mnt/storage/gguf-models/mt3-f16.gguf \
        [--outtype f16|f32] [--variant mt3|ismir2021]

Emitted tensors (arch "mt3", 190 total for the mt3 variant)
-----------------------------------------------------------
    token_embd.weight                       F16  (vocab, d_model)
    lm_head.weight                          F16  (vocab, d_model)
    pos_embd.weight                         F32  (2048, d_model)   generated
    enc.inp_proj.weight                     F16  (d_model, n_mel)
    enc.final_rms.weight                    F32  (d_model,)
    enc.blk.N.attn_{q,k,v,o}.weight         F16
    enc.blk.N.attn_rms.weight               F32
    enc.blk.N.ffn_{gate,up,down}.weight     F16
    enc.blk.N.ffn_rms.weight                F32
    dec.final_rms.weight                    F32  (d_model,)
    dec.blk.N.attn_{q,k,v,o}.weight         F16
    dec.blk.N.attn_rms.weight               F32
    dec.blk.N.cross_{q,k,v,o}.weight        F16
    dec.blk.N.cross_rms.weight              F32
    dec.blk.N.ffn_{gate,up,down}.weight     F16
    dec.blk.N.ffn_rms.weight                F32

License: MT3 code and weights are Apache-2.0 (Google Magenta).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

try:
    import msgpack
except ImportError:  # pragma: no cover
    sys.exit("msgpack is required: pip install msgpack")

try:
    import gguf
except ImportError:  # pragma: no cover
    sys.exit("gguf is required: pip install gguf")


# ---------------------------------------------------------------------------
# Constants extracted from the MT3 source tree (github.com/magenta/mt3).
# Every value below is line-cited so a future reader can re-verify it.
# ---------------------------------------------------------------------------

# mt3/gin/model.gin:48-60  (network.T5Config)
D_MODEL = 512          # emb_dim
N_HEADS = 6            # num_heads
HEAD_DIM = 64          # head_dim        -> qkv width 6*64 = 384
D_FF = 1024            # mlp_dim
ENC_LAYERS = 8         # num_encoder_layers
DEC_LAYERS = 8         # num_decoder_layers
FF_PROJ = "gated-gelu"  # mlp_activations = ('gelu', 'linear')
LOGITS_VIA_EMBEDDING = False   # untied logits_dense
LN_EPS = 1e-6          # mt3/layers.py:606  LayerNorm.epsilon (RMSNorm, no mean)
ATTN_LOGIT_SCALE = 1.0  # mt3/layers.py:230-234  no 1/sqrt(head_dim)

# mt3/layers.py:556-567  FixedEmbed
POS_MAX_LENGTH = 2048
POS_MIN_SCALE = 1.0     # mt3/layers.py:51  sinusoidal(min_scale=1.0,
POS_MAX_SCALE = 10000.0  #                              max_scale=10000.0)

# mt3/spectrograms.py:23-29
SAMPLE_RATE = 16000     # DEFAULT_SAMPLE_RATE
HOP_WIDTH = 128         # DEFAULT_HOP_WIDTH        -> 125 frames/s
NUM_MEL_BINS = 512      # DEFAULT_NUM_MEL_BINS
FFT_SIZE = 2048         # FFT_SIZE
MEL_LO_HZ = 20.0        # MEL_LO_HZ
# mt3/spectral_ops.py:77-84  compute_logmel default hi_hz, NOT overridden by
# spectrograms.compute_spectrogram (mt3/spectrograms.py:64-73 passes only
# bins / lo_hz / overlap / fft_size / sample_rate).
MEL_HI_HZ = 7600.0
MEL_LOG_EPS = 1e-5      # mt3/spectral_ops.py:29  safe_log(x, eps=1e-5)

# mt3/vocabularies.py:33-35 defaults
STEPS_PER_SECOND = 100      # DEFAULT_STEPS_PER_SECOND
MAX_SHIFT_SECONDS = 10      # DEFAULT_MAX_SHIFT_SECONDS
NUM_SPECIAL_TOKENS = 3      # mt3/vocabularies.py:153  0=PAD 1=EOS 2=UNK
EOS_ID = 1                  # mt3/vocabularies.py:159
UNK_ID = 2                  # mt3/vocabularies.py:163
PAD_ID = 0
EXTRA_IDS = 100             # t5.data.DEFAULT_EXTRA_IDS (mt3/vocabularies.py:145)

# Per-variant gin (mt3/gin/mt3.gin, mt3/gin/ismir2021.gin)
VARIANTS = {
    # mt3/gin/mt3.gin:4-9
    "mt3": dict(inputs_length=256, targets_length=1024, num_velocity_bins=1,
                program_granularity="full", use_ties=True, onsets_only=False),
    # mt3/gin/ismir2021.gin:4-9
    "ismir2021": dict(inputs_length=512, targets_length=1024,
                      num_velocity_bins=127, program_granularity="flat",
                      use_ties=False, onsets_only=False),
}


def build_event_ranges(num_velocity_bins: int):
    """Replicate mt3.vocabularies.build_codec() range layout, in order.

    mt3/event_codec.py:57-59 forces 'shift' to be the first block starting at
    id 0; mt3/vocabularies.py:121-134 supplies the rest in this exact order.
    note_seq.MIN/MAX_MIDI_PITCH = 0/127, MIN/MAX_MIDI_PROGRAM = 0/127.

    Returns a list of (type, min_value, max_value) tuples.
    """
    return [
        ("shift", 0, STEPS_PER_SECOND * MAX_SHIFT_SECONDS),  # 0..1000
        ("pitch", 0, 127),
        ("velocity", 0, num_velocity_bins),   # bin 0 == note-off
        ("tie", 0, 0),
        ("program", 0, 127),
        ("drum", 0, 127),
    ]


def codec_num_classes(ranges) -> int:
    """mt3/event_codec.py:64-66."""
    return sum(hi - lo + 1 for _, lo, hi in ranges)


def num_embeddings(num_classes: int) -> int:
    """mt3/vocabularies.py:280-282 -- round the vocab up to a multiple of 128."""
    vocab_size = NUM_SPECIAL_TOKENS + num_classes + EXTRA_IDS
    return 128 * -(-vocab_size // 128)


def sinusoidal_table(max_len: int, features: int) -> np.ndarray:
    """Bit-for-bit port of mt3/layers.py:51-82 `sinusoidal()` initializer.

    Note the layout: the first half of the feature axis is sin, the second half
    is cos -- *not* interleaved, and *not* the "attention is all you need"
    even/odd arrangement. `features // 2 - 1` (not `features // 2`) is the
    denominator of the log-space step.
    """
    pe = np.zeros((max_len, features), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    half = features // 2
    scale_factor = -np.log(POS_MAX_SCALE / POS_MIN_SCALE) / (half - 1)
    div_term = POS_MIN_SCALE * np.exp(np.arange(0, half) * scale_factor)
    pe[:, :half] = np.sin(position * div_term)
    pe[:, half:2 * half] = np.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# T5X checkpoint reading: zarr v2 directories + inline flax msgpack
# ---------------------------------------------------------------------------


def read_zarr_param(param_dir: Path) -> np.ndarray:
    """Decode one zarr v2 parameter directory with stdlib gzip + numpy."""
    meta = json.loads((param_dir / ".zarray").read_text())
    if meta.get("zarr_format") != 2:
        raise ValueError(f"{param_dir.name}: unsupported zarr_format "
                         f"{meta.get('zarr_format')}")
    if meta.get("order") != "C":
        raise ValueError(f"{param_dir.name}: unsupported order {meta['order']}")
    if meta.get("filters"):
        raise ValueError(f"{param_dir.name}: filters are not supported")
    comp = meta.get("compressor") or {}
    if comp.get("id") != "gzip":
        raise ValueError(f"{param_dir.name}: unsupported compressor {comp}")

    shape = tuple(meta["shape"])
    chunks = tuple(meta["chunks"])
    sep = meta.get("dimension_separator", ".")
    dtype = np.dtype(meta["dtype"])

    if chunks != shape:
        raise ValueError(
            f"{param_dir.name}: expected a single whole-array chunk, got "
            f"chunks={chunks} for shape={shape}; multi-chunk zarr is not "
            f"implemented (no MT3 checkpoint param needs it)")

    chunk_name = sep.join("0" for _ in shape) if shape else "0"
    raw = gzip.decompress((param_dir / chunk_name).read_bytes())
    arr = np.frombuffer(raw, dtype=dtype)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"{param_dir.name}: decoded {arr.size} elements, "
                         f"expected {int(np.prod(shape))}")
    return arr.reshape(shape).astype(np.float32, copy=True)


def _flax_ext_hook(code: int, data: bytes):
    """flax.serialization msgpack ExtType(code=1) == an ndarray.

    Payload is msgpack [shape, dtype_name, raw_bytes]; see
    flax/serialization.py `_ndarray_from_bytes`.
    """
    if code == 1:
        shape, dtype_name, buf = msgpack.unpackb(data, raw=False)
        if dtype_name == "bfloat16":
            u16 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            return u16.view(np.float32).reshape(tuple(shape)).copy()
        return (np.frombuffer(buf, dtype=np.dtype(dtype_name))
                .reshape(tuple(shape)).astype(np.float32, copy=True))
    return msgpack.ExtType(code, data)


def read_inline_params(checkpoint_file: Path) -> dict[str, np.ndarray]:
    """Return {dotted_name: ndarray} for every param stored inside `checkpoint`.

    Parameters that live in zarr directories appear here as a TensorStore spec
    dict instead of an ndarray, and are skipped.
    """
    blob = checkpoint_file.read_bytes()
    doc = msgpack.unpackb(blob, ext_hook=_flax_ext_hook, raw=False)
    target = doc["optimizer"]["target"]

    out: dict[str, np.ndarray] = {}

    def walk(node, prefix: str) -> None:
        if isinstance(node, np.ndarray):
            out[prefix] = node
            return
        if isinstance(node, dict):
            # A TensorStore spec dict (the zarr-backed params) -- not a subtree.
            if "driver" in node and "kvstore" in node:
                return
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))

    walk(target, "")
    return out


class Checkpoint:
    """Unified accessor over the zarr dirs + the inline msgpack params."""

    def __init__(self, root: Path):
        self.root = root
        ckpt_file = root / "checkpoint"
        if not ckpt_file.is_file():
            raise SystemExit(f"missing {ckpt_file} -- point --input at the "
                             f"directory that contains 'checkpoint' and the "
                             f"'target.*' zarr directories")
        self.inline = read_inline_params(ckpt_file)
        self.zarr_dirs = {
            p.name[len("target."):]: p
            for p in sorted(root.iterdir())
            if p.is_dir() and p.name.startswith("target.")
            and (p / ".zarray").is_file()
        }

    def names(self) -> list[str]:
        return sorted(set(self.zarr_dirs) | set(self.inline))

    def get(self, name: str) -> np.ndarray:
        if name in self.zarr_dirs:
            return read_zarr_param(self.zarr_dirs[name])
        if name in self.inline:
            return self.inline[name]
        raise KeyError(f"parameter not found in checkpoint: {name}")

    def source_of(self, name: str) -> str:
        return "zarr" if name in self.zarr_dirs else "msgpack"


# ---------------------------------------------------------------------------
# Name mapping:  flax param path  ->  GGUF tensor name
# ---------------------------------------------------------------------------


def build_tensor_map(enc_layers: int, dec_layers: int) -> list[tuple[str, str, bool]]:
    """Return [(flax_name, gguf_name, transpose)] in emission order.

    `transpose` is True for every 2-D kernel: flax DenseGeneral stores
    [in, out] (mt3/layers.py:405-415) but ggml_mul_mat contracts ne[0], i.e.
    wants [out, in].
    """
    m: list[tuple[str, str, bool]] = []

    # --- global -----------------------------------------------------------
    # (1536, 512) already [vocab, d_model] -- an Embed table, not a kernel.
    m.append(("decoder.token_embedder.embedding", "token_embd.weight", False))
    # (512, 1536) [in=d_model, out=vocab] -> [vocab, d_model]
    m.append(("decoder.logits_dense.kernel", "lm_head.weight", True))

    # --- encoder ----------------------------------------------------------
    # (512, 512) [in=n_mel, out=d_model] -> [d_model, n_mel]
    m.append(("encoder.continuous_inputs_projection.kernel",
              "enc.inp_proj.weight", True))
    for i in range(enc_layers):
        p = f"encoder.layers_{i}"
        g = f"enc.blk.{i}"
        m += [
            (f"{p}.pre_attention_layer_norm.scale", f"{g}.attn_rms.weight", False),
            (f"{p}.attention.query.kernel", f"{g}.attn_q.weight", True),
            (f"{p}.attention.key.kernel",   f"{g}.attn_k.weight", True),
            (f"{p}.attention.value.kernel", f"{g}.attn_v.weight", True),
            (f"{p}.attention.out.kernel",   f"{g}.attn_o.weight", True),
            (f"{p}.pre_mlp_layer_norm.scale", f"{g}.ffn_rms.weight", False),
            (f"{p}.mlp.wi_0.kernel", f"{g}.ffn_gate.weight", True),
            (f"{p}.mlp.wi_1.kernel", f"{g}.ffn_up.weight",   True),
            (f"{p}.mlp.wo.kernel",   f"{g}.ffn_down.weight", True),
        ]
    m.append(("encoder.encoder_norm.scale", "enc.final_rms.weight", False))

    # --- decoder ----------------------------------------------------------
    for i in range(dec_layers):
        p = f"decoder.layers_{i}"
        g = f"dec.blk.{i}"
        m += [
            (f"{p}.pre_self_attention_layer_norm.scale", f"{g}.attn_rms.weight", False),
            (f"{p}.self_attention.query.kernel", f"{g}.attn_q.weight", True),
            (f"{p}.self_attention.key.kernel",   f"{g}.attn_k.weight", True),
            (f"{p}.self_attention.value.kernel", f"{g}.attn_v.weight", True),
            (f"{p}.self_attention.out.kernel",   f"{g}.attn_o.weight", True),
            (f"{p}.pre_cross_attention_layer_norm.scale", f"{g}.cross_rms.weight", False),
            (f"{p}.encoder_decoder_attention.query.kernel", f"{g}.cross_q.weight", True),
            (f"{p}.encoder_decoder_attention.key.kernel",   f"{g}.cross_k.weight", True),
            (f"{p}.encoder_decoder_attention.value.kernel", f"{g}.cross_v.weight", True),
            (f"{p}.encoder_decoder_attention.out.kernel",   f"{g}.cross_o.weight", True),
            (f"{p}.pre_mlp_layer_norm.scale", f"{g}.ffn_rms.weight", False),
            (f"{p}.mlp.wi_0.kernel", f"{g}.ffn_gate.weight", True),
            (f"{p}.mlp.wi_1.kernel", f"{g}.ffn_up.weight",   True),
            (f"{p}.mlp.wo.kernel",   f"{g}.ffn_down.weight", True),
        ]
    m.append(("decoder.decoder_norm.scale", "dec.final_rms.weight", False))

    return m


def is_f32_tensor(gguf_name: str) -> bool:
    """RMSNorm scales and the generated position table stay F32."""
    return (gguf_name.endswith("_rms.weight")
            or gguf_name == "pos_embd.weight")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, type=Path,
                    help="checkpoint dir (contains 'checkpoint' + target.* dirs)")
    ap.add_argument("--output", required=True, type=Path,
                    help="output .gguf path")
    ap.add_argument("--variant", default="mt3", choices=sorted(VARIANTS),
                    help="which gin config the checkpoint was trained with")
    ap.add_argument("--outtype", default="f16", choices=("f16", "f32"))
    ap.add_argument("--dump-inventory", type=Path, default=None,
                    help="also write a JSON tensor inventory here")
    args = ap.parse_args()

    v = VARIANTS[args.variant]
    ranges = build_event_ranges(v["num_velocity_bins"])
    n_classes = codec_num_classes(ranges)
    vocab_size = num_embeddings(n_classes)

    print(f"MT3 -> GGUF   variant={args.variant}")
    print(f"  codec classes : {n_classes}  (+{NUM_SPECIAL_TOKENS} special "
          f"+{EXTRA_IDS} extra_ids -> padded vocab {vocab_size})")
    for t, lo, hi in ranges:
        # cumulative id offsets, mt3/event_codec.py:93-101
        pass
    off = 0
    for t, lo, hi in ranges:
        print(f"    {t:<9s} value {lo:>4d}..{hi:<4d}  -> codec id "
              f"{off:>4d}..{off + hi - lo:<4d}  -> token id "
              f"{off + NUM_SPECIAL_TOKENS:>4d}..{off + hi - lo + NUM_SPECIAL_TOKENS}")
        off += hi - lo + 1

    ck = Checkpoint(args.input)
    print(f"  checkpoint    : {len(ck.zarr_dirs)} zarr params + "
          f"{len(ck.inline)} inline msgpack params "
          f"= {len(ck.zarr_dirs) + len(ck.inline)} total")

    tmap = build_tensor_map(ENC_LAYERS, DEC_LAYERS)
    missing = [f for f, _, _ in tmap if f not in ck.zarr_dirs and f not in ck.inline]
    if missing:
        for f in missing:
            print(f"  [err] missing checkpoint param: {f}", file=sys.stderr)
        return 1
    unmapped = sorted(set(ck.names()) - {f for f, _, _ in tmap})
    for u in unmapped:
        print(f"  [warn] checkpoint param not mapped: {u}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(args.output), arch="mt3")

    # ---- architecture ----------------------------------------------------
    w.add_string("general.name", f"mt3-{args.variant}")
    w.add_string("general.license", "apache-2.0")
    w.add_string("general.source.url",
                 f"https://storage.googleapis.com/mt3/checkpoints/{args.variant}")

    w.add_uint32("mt3.vocab_size", vocab_size)
    w.add_uint32("mt3.d_model", D_MODEL)
    w.add_uint32("mt3.d_kv", HEAD_DIM)
    w.add_uint32("mt3.d_ff", D_FF)
    w.add_uint32("mt3.n_heads", N_HEADS)
    w.add_uint32("mt3.encoder.n_layers", ENC_LAYERS)
    w.add_uint32("mt3.decoder.n_layers", DEC_LAYERS)
    w.add_float32("mt3.layer_norm_epsilon", LN_EPS)
    w.add_string("mt3.feed_forward_proj", FF_PROJ)
    w.add_uint32("mt3.tie_word_embeddings", int(LOGITS_VIA_EMBEDDING))
    # Explicit, because src/t5_translate.cpp's relative-attention-bucket path
    # must NOT be taken for MT3 (mt3/network.py:180,225).
    w.add_string("mt3.pos_embed", "sinusoidal")
    w.add_uint32("mt3.pos_embed_max_length", POS_MAX_LENGTH)
    w.add_float32("mt3.pos_embed_min_scale", POS_MIN_SCALE)
    w.add_float32("mt3.pos_embed_max_scale", POS_MAX_SCALE)
    w.add_uint32("mt3.use_relative_attention_bias", 0)
    # mt3/layers.py:230-234: T5 folds 1/sqrt(head_dim) into the query init.
    w.add_float32("mt3.attn_logit_scale", ATTN_LOGIT_SCALE)

    w.add_uint32("mt3.pad_token_id", PAD_ID)
    w.add_uint32("mt3.eos_token_id", EOS_ID)
    w.add_uint32("mt3.unk_token_id", UNK_ID)
    w.add_uint32("mt3.decoder_start_token_id", PAD_ID)  # t5x: BOS == 0

    # ---- spectrogram front end -------------------------------------------
    w.add_uint32("mt3.spectrogram.sample_rate", SAMPLE_RATE)
    w.add_uint32("mt3.spectrogram.hop_width", HOP_WIDTH)
    w.add_uint32("mt3.spectrogram.n_fft", FFT_SIZE)
    w.add_uint32("mt3.spectrogram.frame_length", FFT_SIZE)
    w.add_uint32("mt3.spectrogram.num_mel_bins", NUM_MEL_BINS)
    w.add_float32("mt3.spectrogram.mel_lo_hz", MEL_LO_HZ)
    w.add_float32("mt3.spectrogram.mel_hi_hz", MEL_HI_HZ)
    w.add_float32("mt3.spectrogram.frames_per_second",
                  float(SAMPLE_RATE) / HOP_WIDTH)
    w.add_float32("mt3.spectrogram.log_eps", MEL_LOG_EPS)
    w.add_string("mt3.spectrogram.mel_scale", "htk")       # tf.signal
    w.add_string("mt3.spectrogram.mel_norm", "none")       # tf.signal: no slaney norm
    w.add_string("mt3.spectrogram.magnitude", "magnitude")  # not power
    w.add_string("mt3.spectrogram.window", "hann_periodic")
    w.add_uint32("mt3.spectrogram.center", 0)   # tf.signal.stft: no centering
    w.add_uint32("mt3.spectrogram.pad_end", 1)  # ...but zero-pads the tail

    # ---- segmentation ----------------------------------------------------
    w.add_uint32("mt3.inputs_length", v["inputs_length"])
    w.add_uint32("mt3.targets_length", v["targets_length"])
    w.add_float32("mt3.segment_seconds",
                  v["inputs_length"] * HOP_WIDTH / float(SAMPLE_RATE))

    # ---- event codec -----------------------------------------------------
    w.add_uint32("mt3.codec.steps_per_second", STEPS_PER_SECOND)
    w.add_uint32("mt3.codec.max_shift_steps", STEPS_PER_SECOND * MAX_SHIFT_SECONDS)
    w.add_uint32("mt3.codec.num_classes", n_classes)
    w.add_uint32("mt3.codec.num_special_tokens", NUM_SPECIAL_TOKENS)
    w.add_uint32("mt3.codec.extra_ids", EXTRA_IDS)
    w.add_uint32("mt3.codec.num_velocity_bins", v["num_velocity_bins"])
    w.add_string("mt3.codec.program_granularity", v["program_granularity"])
    w.add_uint32("mt3.codec.use_ties", int(v["use_ties"]))
    w.add_uint32("mt3.codec.onsets_only", int(v["onsets_only"]))
    # Range layout, in codec order. Codec id of the i-th range starts at
    # sum_{j<i} (max_j - min_j + 1); token id = codec id + num_special_tokens.
    w.add_array("mt3.codec.event_types", [t for t, _, _ in ranges])
    w.add_array("mt3.codec.event_min_values", [lo for _, lo, _ in ranges])
    w.add_array("mt3.codec.event_max_values", [hi for _, _, hi in ranges])

    # ---- tensors ---------------------------------------------------------
    inventory: list[dict] = []
    n_f16 = n_f32 = 0

    def emit(gguf_name: str, arr: np.ndarray, src: str, src_shape) -> None:
        nonlocal n_f16, n_f32
        if is_f32_tensor(gguf_name) or args.outtype == "f32":
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            n_f32 += 1
        else:
            arr = np.ascontiguousarray(arr.astype(np.float16))
            n_f16 += 1
        w.add_tensor(gguf_name, arr)
        inventory.append(dict(gguf=gguf_name, source=src,
                              src_shape=list(src_shape),
                              shape=list(arr.shape), dtype=str(arr.dtype)))

    # generated sinusoidal position table (identical for encoder & decoder)
    pos = sinusoidal_table(POS_MAX_LENGTH, D_MODEL)
    emit("pos_embd.weight", pos, "generated:mt3/layers.py:sinusoidal()",
         pos.shape)

    for flax_name, gguf_name, transpose in tmap:
        a = ck.get(flax_name)
        src_shape = a.shape
        if transpose:
            if a.ndim != 2:
                return _fail(f"{flax_name}: expected 2-D kernel, got {a.shape}")
            a = np.ascontiguousarray(a.T)
        emit(gguf_name, a, f"{ck.source_of(flax_name)}:{flax_name}", src_shape)

    print(f"\n  tensors: {len(inventory)}  (F16 {n_f16}, F32 {n_f32})")
    for row in inventory[:6]:
        print(f"    {row['gguf']:<28s} {str(tuple(row['src_shape'])):>14s} -> "
              f"{str(tuple(row['shape'])):>14s} {row['dtype']}")
    print("    ...")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    size = args.output.stat().st_size
    print(f"\n  wrote {args.output}  ({size / 1e6:.1f} MB)")

    if args.dump_inventory:
        args.dump_inventory.write_text(json.dumps(inventory, indent=1))
        print(f"  inventory -> {args.dump_inventory}")
    return 0


def _fail(msg: str) -> int:
    print(f"  [err] {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
