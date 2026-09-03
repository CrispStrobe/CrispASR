#!/usr/bin/env python3
"""
MT3 (Magenta "Multi-Task Multitrack Music Transcription") reference dumper.

A **numpy-only** re-implementation of the upstream JAX/Flax forward pass, run
directly on the T5X checkpoint (zarr + msgpack, decoded by
`models/convert-mt3-to-gguf.py`). No JAX, no t5x, no seqio, no TensorFlow, no
torch is required to produce the reference tensors.

Why not just run upstream?  The Magenta MT3 repo pins an old
JAX/Flax/t5x/seqio/TF stack that no longer resolves, and the one packaged
PyTorch oracle (openmirlab/mt3-infer) *explicitly excludes* the Magenta MT3
backend for exactly that reason.  So the oracle for CrispASR's C++ port has to
be written from the published source, line by line -- which is what this file
is.  Every non-obvious constant carries a `mt3/<file>.py:<line>` citation.

Stages dumped (all .npy, plus ref.gguf for crispasr-diff):

  1. audio_segment0    (32768,)     exact float32 samples of segment 0
  2. mel               (256, 512)   log-mel, segment 0   [MelsTime layout:
                                    row = frame, col = mel bin]
  3. mel_all           (S*256, 512) log-mel for every segment, concatenated
                                    (a short final segment is zero-padded to
                                    256 rows, as seqio does)
  4. enc_input         (256, 512)   after continuous_inputs_projection + pos
  5. enc_out           (256, 512)   encoder output (post encoder_norm)
  6. logits_step0      (1536,)      decoder logits for the first greedy step
  7. logits_prefix     (T, 1536)    logits for every greedy step actually taken
  8. tokens.json                    greedy token ids + decoded events per
                                    segment, and the note list after the tie
                                    state machine

Front end -- tf.signal-compatible, re-derived from source:
  mt3/spectrograms.py:55-73 + mt3/spectral_ops.py:35-88
    * audio is zero-padded up to a multiple of hop_width (128), split into
      128-sample frames, then chunked into `inputs_length` (256) frame
      segments; the STFT is computed **per segment**, on the 32768 flattened
      samples, NOT once over the whole file (mt3/preprocessors.py:614-618).
    * tf.signal.stft(frame_length=2048, frame_step=128, fft_length=2048,
      window=hann periodic, pad_end=True) -> 256 frames x 1025 bins.
    * magnitude (not power) -> tf.signal.linear_to_mel_weight_matrix(
      512 bins, 1025, 16000, lo=20.0, hi=7600.0)  [HTK mel, no Slaney norm,
      DC bin zeroed]
    * safe_log: where(x <= 0, 1e-5, x) then natural log.  NOTE the clamp is
      only on non-positive values; small positive values are NOT floored.

Model -- mt3/network.py + mt3/layers.py:
    * RMSNorm (no mean subtraction, no bias), eps 1e-6   [layers.py:604-620]
    * gated-GELU FFN: gelu(x@wi_0) * (x@wi_1) @ wo       [layers.py:454-486]
      with flax's default tanh-approximate GELU.
    * NO 1/sqrt(head_dim) attention rescale               [layers.py:230-234]
    * fixed sinusoidal absolute positions added to the embeddings, first half
      sin / second half cos, log-step denominator (d/2 - 1)  [layers.py:51-82,
      network.py:180 encoder, network.py:225 decoder]
    * encoder input is continuous: Dense(512->512) on the mel frame, there is
      no encoder token embedding                          [network.py:174-180]
    * untied logits_dense                                 [network.py:256-261]

Usage:
    python3 tools/reference_backends/mt3.py \\
        --checkpoint /mnt/storage/gguf-models/mt3-src/mt3 \\
        --audio samples/piano.wav \\
        --output-dir /mnt/volume1/tmp-overflow/mt3-ref \\
        --max-segments 2 --decode-steps 64

Self-checks (--self-check, on by default):
    * shapes and frame counts against the gin config
    * sinusoidal table reproduces the closed form at several positions
    * the mel filterbank rows sum to the expected triangular partition
    * segment 0 mel recomputed from `mel_all` matches `mel` bit-for-bit
    * greedy decode never emits a token outside the codec range without being
      counted as invalid

Optional torch cross-check (--torch-oracle PATH_TO_mt3.pth), against
kunato/mt3-pytorch's conversion of this same Magenta checkpoint (re-hosted as
huggingface.co/gudgud1014/MR-MT3 `mt3.pth`, sha256 b8a3807e...; the same file
openmirlab/mt3-infer pins for its `mt3_pytorch` profile).  Two things are
checked:

  * **weights** -- all 189 parameters decoded from zarr/msgpack, transposed
    flax [in,out] -> torch [out,in], compared elementwise. This validates the
    checkpoint decoder and the transpose convention used by
    models/convert-mt3-to-gguf.py against an independent conversion.
  * **encoder forward** -- recomputed with torch ops and compared by cosine.

Caveat, and the reason the forward check needs a patch: kunato's
`FixedPositionalEmbedding` (t5.py:529-541) uses
`inv_freq = 10000 ** (-i/(d/2))` whereas Magenta uses `10000 ** (-i/(d/2-1))`
(mt3/layers.py:76).  Its weights are Magenta's; its position table is not.
The check runs both ways and prints both numbers; `--torch-oracle-raw` runs
only the un-patched one.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (mt3 source line citations in the module docstring / inline)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000     # mt3/spectrograms.py:23
HOP_WIDTH = 128         # mt3/spectrograms.py:24
NUM_MEL_BINS = 512      # mt3/spectrograms.py:25
FFT_SIZE = 2048         # mt3/spectrograms.py:28
MEL_LO_HZ = 20.0        # mt3/spectrograms.py:29
MEL_HI_HZ = 7600.0      # mt3/spectral_ops.py:78 (default, not overridden)
LOG_EPS = 1e-5          # mt3/spectral_ops.py:29

D_MODEL = 512           # mt3/gin/model.gin:51
N_HEADS = 6             # mt3/gin/model.gin:52
HEAD_DIM = 64           # mt3/gin/model.gin:55
D_FF = 1024             # mt3/gin/model.gin:56
ENC_LAYERS = 8          # mt3/gin/model.gin:53
DEC_LAYERS = 8          # mt3/gin/model.gin:54
LN_EPS = 1e-6           # mt3/layers.py:606
POS_MAX_LENGTH = 2048   # mt3/layers.py:565

INPUTS_LENGTH = 256     # mt3/gin/mt3.gin:4  {'inputs': 256}
TARGETS_LENGTH = 1024   # mt3/gin/mt3.gin:4  {'targets': 1024}

STEPS_PER_SECOND = 100  # mt3/vocabularies.py:33
MAX_SHIFT_SECONDS = 10  # mt3/vocabularies.py:34
NUM_VELOCITY_BINS = 1   # mt3/gin/mt3.gin:6  (multi-instrument model)
NUM_SPECIAL_TOKENS = 3  # mt3/vocabularies.py:153
EOS_ID = 1              # mt3/vocabularies.py:159
PAD_ID = 0
DECODER_START_ID = 0    # t5x autoregressive decode starts from 0

DECODED_EOS = -1        # mt3/vocabularies.py:29
DECODED_INVALID = -2    # mt3/vocabularies.py:30

DEFAULT_VELOCITY = 100  # mt3/note_sequences.py:28
DEFAULT_NOTE_DURATION = 0.01  # mt3/note_sequences.py:29
MIN_NOTE_DURATION = 0.01      # mt3/note_sequences.py:32

# mt3/vocabularies.py:121-134 + mt3/event_codec.py:57-59 ('shift' forced first)
EVENT_RANGES = [
    ("shift", 0, STEPS_PER_SECOND * MAX_SHIFT_SECONDS),
    ("pitch", 0, 127),
    ("velocity", 0, NUM_VELOCITY_BINS),
    ("tie", 0, 0),
    ("program", 0, 127),
    ("drum", 0, 127),
]


# ---------------------------------------------------------------------------
# Event codec (mt3/event_codec.py)
# ---------------------------------------------------------------------------


class Codec:
    """Port of mt3.event_codec.Codec for the ranges above."""

    def __init__(self, ranges=EVENT_RANGES, steps_per_second=STEPS_PER_SECOND):
        self.ranges = list(ranges)
        self.steps_per_second = steps_per_second
        self.offsets = []
        off = 0
        for _, lo, hi in self.ranges:
            self.offsets.append(off)
            off += hi - lo + 1
        self.num_classes = off

    def event_type_range(self, t):
        """mt3/event_codec.py:93-101 -> inclusive [min_id, max_id]."""
        for (name, lo, hi), off in zip(self.ranges, self.offsets):
            if name == t:
                return off, off + (hi - lo)
        raise KeyError(t)

    def encode(self, t, value):
        """mt3/event_codec.py:79-91."""
        for (name, lo, hi), off in zip(self.ranges, self.offsets):
            if name == t:
                if not lo <= value <= hi:
                    raise ValueError(f"{t} value {value} out of [{lo},{hi}]")
                return off + value - lo
        raise KeyError(t)

    def decode(self, index):
        """mt3/event_codec.py:103-112 -> (type, value)."""
        for (name, lo, hi), off in zip(self.ranges, self.offsets):
            if off <= index <= off + hi - lo:
                return name, lo + index - off
        raise ValueError(f"unknown event index {index}")


def vocab_size(codec: Codec, extra_ids: int = 100) -> int:
    """mt3/vocabularies.py:280-282: round up to a multiple of 128."""
    n = NUM_SPECIAL_TOKENS + codec.num_classes + extra_ids
    return 128 * -(-n // 128)


def decode_token_id(tid: int, codec: Codec, base_vocab: int) -> int:
    """mt3/vocabularies.py:211-219  GenericTokenVocabulary._decode_id."""
    if tid == EOS_ID:
        return DECODED_EOS
    if tid < NUM_SPECIAL_TOKENS:
        return DECODED_INVALID
    if tid >= base_vocab:
        return DECODED_INVALID
    return tid - NUM_SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# Front end: tf.signal-compatible log-mel
# ---------------------------------------------------------------------------


def hertz_to_mel(f):
    """tensorflow/python/ops/signal/mel_ops.py `_hertz_to_mel`.

    HTK form with the natural-log constants TF actually uses:
    1127.0 * ln(1 + f / 700.0)  ==  2595 * log10(1 + f / 700).
    """
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def linear_to_mel_weight_matrix(num_mel_bins=NUM_MEL_BINS,
                                num_spectrogram_bins=FFT_SIZE // 2 + 1,
                                sample_rate=SAMPLE_RATE,
                                lower_edge_hertz=MEL_LO_HZ,
                                upper_edge_hertz=MEL_HI_HZ) -> np.ndarray:
    """Port of tf.signal.linear_to_mel_weight_matrix.

    Differences from librosa.filters.mel that matter:
      * the DC spectrogram bin is dropped (zeroed), TF's `bands_to_zero = 1`;
      * no Slaney area normalisation -- triangles peak at exactly 1.0;
      * edges come from a length-(num_mel_bins + 2) linspace in mel space,
        framed with (length 3, step 1).
    """
    nyquist = sample_rate / 2.0
    linear_hz = np.linspace(0.0, nyquist, num_spectrogram_bins,
                            dtype=np.float64)[1:]          # bands_to_zero = 1
    bins_mel = hertz_to_mel(linear_hz)[:, None]            # (B-1, 1)

    edges = np.linspace(hertz_to_mel(lower_edge_hertz),
                        hertz_to_mel(upper_edge_hertz),
                        num_mel_bins + 2, dtype=np.float64)
    lower = edges[0:num_mel_bins][None, :]
    center = edges[1:num_mel_bins + 1][None, :]
    upper = edges[2:num_mel_bins + 2][None, :]

    lower_slopes = (bins_mel - lower) / (center - lower)
    upper_slopes = (upper - bins_mel) / (upper - center)
    w = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.pad(w, [[1, 0], [0, 0]]).astype(np.float32)  # restore DC row = 0


def hann_periodic(n: int) -> np.ndarray:
    """tf.signal.hann_window(periodic=True)."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def stft_magnitude(samples: np.ndarray, frame_length=FFT_SIZE,
                   frame_step=HOP_WIDTH) -> np.ndarray:
    """tf.signal.stft(..., pad_end=True) magnitude, shape (frames, n_fft/2+1).

    pad_end=True (tf.signal.frame): num_frames = ceil(n / frame_step) and the
    signal is zero-padded on the right so the last frame is full. There is no
    centering and no reflection padding.
    """
    n = samples.shape[0]
    num_frames = -(-n // frame_step)
    needed = (num_frames - 1) * frame_step + frame_length
    if needed > n:
        samples = np.pad(samples, (0, needed - n))
    idx = (np.arange(num_frames)[:, None] * frame_step
           + np.arange(frame_length)[None, :])
    frames = samples[idx] * hann_periodic(frame_length)
    return np.abs(np.fft.rfft(frames.astype(np.float64), n=frame_length))


def safe_log(x: np.ndarray) -> np.ndarray:
    """mt3/spectral_ops.py:29-32 -- clamp only NON-POSITIVE values."""
    return np.log(np.where(x <= 0.0, LOG_EPS, x))


_MEL_FB_CACHE: dict = {}


def compute_logmel(samples: np.ndarray) -> np.ndarray:
    """mt3/spectrograms.py:64-73 -> (frames, 512) float32."""
    key = (NUM_MEL_BINS, FFT_SIZE // 2 + 1, SAMPLE_RATE, MEL_LO_HZ, MEL_HI_HZ)
    fb = _MEL_FB_CACHE.get(key)
    if fb is None:
        fb = linear_to_mel_weight_matrix()
        _MEL_FB_CACHE[key] = fb
    mag = stft_magnitude(samples)
    mel = mag.astype(np.float32) @ fb
    return safe_log(mel).astype(np.float32)


def split_into_segments(audio: np.ndarray, inputs_length=INPUTS_LENGTH):
    """mt3 colab `_audio_to_frames` + t5 `split_tokens_to_inputs_length`.

    Pads the audio up to a whole number of 128-sample frames, then chunks the
    frame stream into `inputs_length`-frame segments (the last one short).
    Returns [(start_time_seconds, samples_1d)].
    """
    n = audio.shape[0]
    pad = (-n) % HOP_WIDTH
    if pad:
        audio = np.pad(audio, (0, pad))
    num_frames = audio.shape[0] // HOP_WIDTH
    fps = SAMPLE_RATE / HOP_WIDTH                     # 125.0
    segs = []
    for f0 in range(0, num_frames, inputs_length):
        f1 = min(f0 + inputs_length, num_frames)
        start_time = f0 / fps
        # colab: start_time -= start_time % (1 / steps_per_second)
        start_time -= start_time % (1.0 / STEPS_PER_SECOND)
        segs.append((float(start_time),
                     audio[f0 * HOP_WIDTH:f1 * HOP_WIDTH].astype(np.float32)))
    return segs


# ---------------------------------------------------------------------------
# Checkpoint reading (same decode path as models/convert-mt3-to-gguf.py)
# ---------------------------------------------------------------------------


def _read_zarr(param_dir: Path) -> np.ndarray:
    meta = json.loads((param_dir / ".zarray").read_text())
    shape = tuple(meta["shape"])
    sep = meta.get("dimension_separator", ".")
    name = sep.join("0" for _ in shape)
    raw = gzip.decompress((param_dir / name).read_bytes())
    return (np.frombuffer(raw, dtype=np.dtype(meta["dtype"]))
            .reshape(shape).astype(np.float32, copy=True))


def _flax_ext_hook(code, data):
    import msgpack
    if code == 1:
        shape, dtype_name, buf = msgpack.unpackb(data, raw=False)
        return (np.frombuffer(buf, dtype=np.dtype(dtype_name))
                .reshape(tuple(shape)).astype(np.float32, copy=True))
    return msgpack.ExtType(code, data)


def load_checkpoint(root: Path) -> dict[str, np.ndarray]:
    """Return {flax_dotted_name: float32 ndarray} for all 189 parameters."""
    import msgpack
    doc = msgpack.unpackb((root / "checkpoint").read_bytes(),
                          ext_hook=_flax_ext_hook, raw=False)
    params: dict[str, np.ndarray] = {}

    def walk(node, prefix):
        if isinstance(node, np.ndarray):
            params[prefix] = node
        elif isinstance(node, dict):
            if "driver" in node and "kvstore" in node:
                return           # a TensorStore spec -> lives in a zarr dir
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))

    walk(doc["optimizer"]["target"], "")
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("target.") and (p / ".zarray").is_file():
            params[p.name[len("target."):]] = _read_zarr(p)
    return params


# ---------------------------------------------------------------------------
# Model ops (numpy)
# ---------------------------------------------------------------------------


def sinusoidal_table(max_len=POS_MAX_LENGTH, features=D_MODEL) -> np.ndarray:
    """mt3/layers.py:51-82. First half sin, second half cos; denom d/2 - 1."""
    pe = np.zeros((max_len, features), dtype=np.float32)
    pos = np.arange(0, max_len)[:, None]
    half = features // 2
    scale_factor = -np.log(10000.0 / 1.0) / (half - 1)
    div_term = 1.0 * np.exp(np.arange(0, half) * scale_factor)
    pe[:, :half] = np.sin(pos * div_term)
    pe[:, half:2 * half] = np.cos(pos * div_term)
    return pe


def rms_norm(x: np.ndarray, scale: np.ndarray, eps=LN_EPS) -> np.ndarray:
    """mt3/layers.py:604-621 -- T5 LayerNorm: no mean, no bias, float32."""
    x = x.astype(np.float32)
    mean2 = np.mean(x * x, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(mean2 + eps))) * scale


def gelu(x: np.ndarray) -> np.ndarray:
    """flax.linen.gelu default (approximate=True) -- tanh approximation."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _heads(x: np.ndarray) -> np.ndarray:
    """(T, n_heads*head_dim) -> (n_heads, T, head_dim)."""
    t = x.shape[0]
    return x.reshape(t, N_HEADS, HEAD_DIM).transpose(1, 0, 2)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
              causal: bool = False) -> np.ndarray:
    """mt3/layers.py:85-157.  NO 1/sqrt(head_dim) rescale (layers.py:230-234).

    q/k/v are (T, n_heads*head_dim). Returns (T, n_heads*head_dim).
    The encoder mask is all-ones (network.py:286-289), i.e. bias 0 everywhere,
    so it is omitted; only the decoder's causal mask is materialised.
    """
    qh, kh, vh = _heads(q), _heads(k), _heads(v)
    logits = np.einsum("htd,hsd->hts", qh, kh)
    if causal:
        tq, tk = logits.shape[1], logits.shape[2]
        # mt3/layers.py:319-322 turns a false mask into -1e10, not -inf.
        mask = np.triu(np.ones((tq, tk), dtype=bool), k=1 + (tk - tq))
        logits = np.where(mask, np.float32(-1e10), logits)
    w = softmax(logits, axis=-1)
    out = np.einsum("hts,hsd->htd", w, vh)
    return out.transpose(1, 0, 2).reshape(q.shape[0], N_HEADS * HEAD_DIM)


class MT3Numpy:
    """Forward pass over the raw flax parameter dict."""

    def __init__(self, params: dict[str, np.ndarray]):
        self.p = params
        self.pos = sinusoidal_table()

    def k(self, name: str) -> np.ndarray:
        return self.p[name]

    # -- encoder ----------------------------------------------------------
    def encode(self, mel: np.ndarray, dump: dict | None = None) -> np.ndarray:
        """mel (T, 512) -> encoder output (T, 512)."""
        # network.py:174-179 continuous_inputs_projection, kernel is [in, out]
        x = mel.astype(np.float32) @ self.k("encoder.continuous_inputs_projection.kernel")
        # network.py:180  x = x + FixedEmbed(arange(T))
        x = x + self.pos[:x.shape[0]]
        if dump is not None:
            dump["enc_input"] = x.copy()
        for i in range(ENC_LAYERS):
            pre = f"encoder.layers_{i}"
            h = rms_norm(x, self.k(f"{pre}.pre_attention_layer_norm.scale"))
            q = h @ self.k(f"{pre}.attention.query.kernel")
            kk = h @ self.k(f"{pre}.attention.key.kernel")
            v = h @ self.k(f"{pre}.attention.value.kernel")
            a = attention(q, kk, v) @ self.k(f"{pre}.attention.out.kernel")
            x = x + a
            h = rms_norm(x, self.k(f"{pre}.pre_mlp_layer_norm.scale"))
            g = gelu(h @ self.k(f"{pre}.mlp.wi_0.kernel"))
            u = h @ self.k(f"{pre}.mlp.wi_1.kernel")
            x = x + (g * u) @ self.k(f"{pre}.mlp.wo.kernel")
        return rms_norm(x, self.k("encoder.encoder_norm.scale"))

    # -- decoder ----------------------------------------------------------
    def decode_step(self, enc_out: np.ndarray, tokens: list[int]) -> np.ndarray:
        """Full (non-cached) decoder forward; returns logits for the LAST step.

        Runs the whole prefix each call: O(T^2) but trivially correct, which is
        what a reference wants. `tokens` is the decoder INPUT sequence, i.e.
        [0, tok_0, tok_1, ...] (t5x shifts right with a 0 start token).
        """
        ids = np.asarray(tokens, dtype=np.int64)
        emb = self.k("decoder.token_embedder.embedding")
        y = emb[ids].astype(np.float32)
        # network.py:225 decoder positions are 0..T-1 (arange, network.py:214)
        y = y + self.pos[:y.shape[0]]
        for i in range(DEC_LAYERS):
            pre = f"decoder.layers_{i}"
            h = rms_norm(y, self.k(f"{pre}.pre_self_attention_layer_norm.scale"))
            q = h @ self.k(f"{pre}.self_attention.query.kernel")
            kk = h @ self.k(f"{pre}.self_attention.key.kernel")
            v = h @ self.k(f"{pre}.self_attention.value.kernel")
            a = attention(q, kk, v, causal=True) @ self.k(f"{pre}.self_attention.out.kernel")
            y = y + a

            h = rms_norm(y, self.k(f"{pre}.pre_cross_attention_layer_norm.scale"))
            q = h @ self.k(f"{pre}.encoder_decoder_attention.query.kernel")
            kk = enc_out @ self.k(f"{pre}.encoder_decoder_attention.key.kernel")
            v = enc_out @ self.k(f"{pre}.encoder_decoder_attention.value.kernel")
            a = attention(q, kk, v) @ self.k(f"{pre}.encoder_decoder_attention.out.kernel")
            y = y + a

            h = rms_norm(y, self.k(f"{pre}.pre_mlp_layer_norm.scale"))
            g = gelu(h @ self.k(f"{pre}.mlp.wi_0.kernel"))
            u = h @ self.k(f"{pre}.mlp.wi_1.kernel")
            y = y + (g * u) @ self.k(f"{pre}.mlp.wo.kernel")

        y = rms_norm(y, self.k("decoder.decoder_norm.scale"))
        return (y @ self.k("decoder.logits_dense.kernel")).astype(np.float32)

    def greedy(self, enc_out: np.ndarray, max_steps: int,
               collect_logits: bool = False):
        """Greedy autoregressive decode (paper section 3). Stops at EOS."""
        inp = [DECODER_START_ID]
        out: list[int] = []
        all_logits: list[np.ndarray] = []
        for _ in range(max_steps):
            logits = self.decode_step(enc_out, inp)[-1]
            if collect_logits:
                all_logits.append(logits.copy())
            tok = int(np.argmax(logits))
            out.append(tok)
            if tok == EOS_ID:
                break
            inp.append(tok)
        return out, (np.stack(all_logits) if all_logits else
                     np.zeros((0, 1), dtype=np.float32))


# ---------------------------------------------------------------------------
# Tie-section note assembly (mt3/note_sequences.py + run_length_encoding.py)
# ---------------------------------------------------------------------------


class NoteDecodingState:
    """mt3/note_sequences.py:262-281."""

    def __init__(self):
        self.current_time = 0.0
        self.current_velocity = DEFAULT_VELOCITY
        self.current_program = 0
        self.active_pitches: dict[tuple[int, int], tuple[float, int]] = {}
        self.tied_pitches: set[tuple[int, int]] = set()
        self.is_tie_section = False
        self.notes: list[dict] = []
        self.total_time = 0.0

    def add_note(self, start, end, pitch, velocity, program=0, is_drum=False):
        """mt3/note_sequences.py:301-310."""
        end = max(end, start + MIN_NOTE_DURATION)
        self.notes.append(dict(start_s=float(start), end_s=float(end),
                               pitch=int(pitch), velocity=int(velocity),
                               program=int(program), is_drum=bool(is_drum)))
        self.total_time = max(self.total_time, end)


def bin_to_velocity(b, num_bins=NUM_VELOCITY_BINS):
    """mt3/vocabularies.py:70-74."""
    return 0 if b == 0 else int(127 * b / num_bins)


def decode_note_event(state: NoteDecodingState, time: float, etype: str,
                      value: int) -> None:
    """mt3/note_sequences.py:313-387. Raises ValueError on invalid events."""
    if time < state.current_time:
        raise ValueError(f"event time < current time ({time} < {state.current_time})")
    state.current_time = time
    if etype == "pitch":
        key = (value, state.current_program)
        if state.is_tie_section:
            if key not in state.active_pitches:
                raise ValueError(f"inactive pitch/program in tie section: {key}")
            if key in state.tied_pitches:
                raise ValueError(f"pitch/program already tied: {key}")
            state.tied_pitches.add(key)
        elif state.current_velocity == 0:
            if key not in state.active_pitches:
                raise ValueError(f"note-off for inactive pitch/program: {key}")
            t0, v0 = state.active_pitches.pop(key)
            state.add_note(t0, time, value, v0, state.current_program)
        else:
            if key in state.active_pitches:
                t0, v0 = state.active_pitches.pop(key)
                state.add_note(t0, time, value, v0, state.current_program)
            state.active_pitches[key] = (time, state.current_velocity)
    elif etype == "drum":
        if state.current_velocity == 0:
            raise ValueError("velocity cannot be zero for drum event")
        state.add_note(time, time + DEFAULT_NOTE_DURATION, value,
                       state.current_velocity, 0, True)
    elif etype == "velocity":
        state.current_velocity = bin_to_velocity(value)
    elif etype == "program":
        state.current_program = value
    elif etype == "tie":
        if not state.is_tie_section:
            raise ValueError("tie end event when not in tie section")
        for key in list(state.active_pitches.keys()):
            if key not in state.tied_pitches:
                t0, v0 = state.active_pitches.pop(key)
                state.add_note(t0, state.current_time, key[0], v0, key[1])
        state.is_tie_section = False
    else:
        raise ValueError(f"unexpected event type {etype}")


def decode_segment(state: NoteDecodingState, tokens, start_time: float,
                   max_time, codec: Codec, base_vocab: int):
    """mt3/run_length_encoding.py:371-423 over one segment's tokens.

    NOTE the time semantics: `cur_steps` accumulates across CONSECUTIVE shift
    tokens and is reset to 0 by any non-shift event, and `cur_time` is
    recomputed as `start_time + cur_steps / steps_per_second`. Shifts are
    therefore ABSOLUTE within the segment, not deltas from the previous event.
    """
    invalid = dropped = 0
    cur_steps = 0
    cur_time = start_time
    events = []
    for idx, tok in enumerate(tokens):
        cid = decode_token_id(int(tok), codec, base_vocab)
        if cid == DECODED_EOS:
            break
        if cid == DECODED_INVALID or cid >= codec.num_classes:
            invalid += 1
            continue
        etype, value = codec.decode(cid)
        if etype == "shift":
            cur_steps += value
            cur_time = start_time + cur_steps / codec.steps_per_second
            if max_time is not None and cur_time > max_time:
                dropped = len(tokens) - idx
                break
        else:
            cur_steps = 0
            try:
                decode_note_event(state, cur_time, etype, value)
                events.append(dict(t=round(cur_time, 4), type=etype, value=value))
            except ValueError as exc:
                invalid += 1
                events.append(dict(t=round(cur_time, 4), type=etype,
                                   value=value, error=str(exc)))
    return invalid, dropped, events


def flush(state: NoteDecodingState):
    """mt3/note_sequences.py:396-408."""
    for t0, _ in state.active_pitches.values():
        state.current_time = max(state.current_time, t0 + MIN_NOTE_DURATION)
    for key in list(state.active_pitches.keys()):
        t0, v0 = state.active_pitches.pop(key)
        state.add_note(t0, state.current_time, key[0], v0, key[1])
    state.notes.sort(key=lambda n: (n["start_s"], n["program"], n["pitch"]))
    return state.notes


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------


def self_check(codec: Codec, base_vocab: int, mel: np.ndarray,
               mel_all: np.ndarray, segs, verbose=True) -> list[str]:
    fails: list[str] = []

    def chk(ok, msg):
        (print if verbose else (lambda *_: None))(
            f"    [{'ok ' if ok else 'FAIL'}] {msg}")
        if not ok:
            fails.append(msg)

    # --- vocabulary layout -------------------------------------------------
    chk(codec.num_classes == 1388,
        f"codec.num_classes == 1388 (got {codec.num_classes})")
    chk(base_vocab == 1536, f"padded vocab == 1536 (got {base_vocab})")
    chk(codec.event_type_range("shift") == (0, 1000),
        f"shift ids 0..1000 (got {codec.event_type_range('shift')})")
    chk(codec.event_type_range("pitch") == (1001, 1128),
        f"pitch ids 1001..1128 (got {codec.event_type_range('pitch')})")
    chk(codec.event_type_range("velocity") == (1129, 1130),
        f"velocity ids 1129..1130 (got {codec.event_type_range('velocity')})")
    chk(codec.event_type_range("tie") == (1131, 1131),
        f"tie id 1131 (got {codec.event_type_range('tie')})")
    chk(codec.event_type_range("program") == (1132, 1259),
        f"program ids 1132..1259 (got {codec.event_type_range('program')})")
    chk(codec.event_type_range("drum") == (1260, 1387),
        f"drum ids 1260..1387 (got {codec.event_type_range('drum')})")
    rt = all(codec.decode(codec.encode(t, v)) == (t, v)
             for t, lo, hi in EVENT_RANGES for v in (lo, hi, (lo + hi) // 2))
    chk(rt, "codec encode/decode round-trips on every range endpoint")

    # --- sinusoidal table --------------------------------------------------
    pe = sinusoidal_table()
    half = D_MODEL // 2
    ref_div = 10000.0 ** (-np.arange(half) / (half - 1))
    ok = True
    for pos in (0, 1, 37, 255, 2047):
        ok &= np.allclose(pe[pos, :half], np.sin(pos * ref_div), atol=1e-6)
        ok &= np.allclose(pe[pos, half:], np.cos(pos * ref_div), atol=1e-6)
    chk(ok, "sinusoidal table matches 10000**(-i/(d/2-1)) closed form")
    chk(np.allclose(pe[0, :half], 0.0) and np.allclose(pe[0, half:], 1.0),
        "position 0 is all-sin 0 / all-cos 1")

    # --- mel filterbank ----------------------------------------------------
    fb = linear_to_mel_weight_matrix()
    chk(fb.shape == (1025, 512), f"filterbank shape (1025, 512) (got {fb.shape})")
    chk(np.all(fb[0] == 0.0), "DC spectrogram bin is zeroed (TF bands_to_zero=1)")
    chk(float(fb.max()) <= 1.0 + 1e-6,
        f"triangles are unnormalised, peak <= 1.0 (got {fb.max():.4f})")
    chk(float(fb.min()) >= 0.0, "filterbank is non-negative")
    # Every pair of adjacent triangles partitions its overlap region: rows in
    # the covered band sum to 1 within a half-bin of quantisation slack.
    covered = fb.sum(axis=1)
    lo_bin = int(np.argmax(covered > 0.99))
    hi_bin = 1025 - int(np.argmax(covered[::-1] > 0.99))
    dev = float(np.abs(covered[lo_bin:hi_bin] - 1).max())
    chk(dev < 1e-3,
        f"unit partition over bins {lo_bin}..{hi_bin} (max dev {dev:.2e})")

    # --- segmentation / mel ------------------------------------------------
    chk(mel.shape == (INPUTS_LENGTH, NUM_MEL_BINS) or
        mel.shape[0] == segs[0][1].shape[0] // HOP_WIDTH,
        f"segment 0 mel has one frame per {HOP_WIDTH} samples "
        f"(got {mel.shape})")
    recomputed = compute_logmel(segs[0][1])
    chk(np.array_equal(recomputed, mel),
        "segment 0 mel is reproducible bit-for-bit")
    want = sum(max(INPUTS_LENGTH, s.shape[0] // HOP_WIDTH) for _, s in segs)
    chk(mel_all.shape[0] == want,
        f"mel_all frame count matches the segmentation, short final segment "
        f"zero-padded to {INPUTS_LENGTH} ({mel_all.shape[0]} vs {want})")
    chk(np.array_equal(mel_all[:mel.shape[0]], mel),
        "mel_all[0:256] == segment 0 mel")
    chk(bool(np.isfinite(mel).all()), "mel is finite (safe_log never hits -inf)")
    # safe_log clamps ONLY non-positive values (mt3/spectral_ops.py:29-32);
    # small positive magnitudes pass through, so log(mel) < log(1e-5) is
    # expected and correct. A C++ port that floors at eps instead would be
    # wrong -- pin the exact semantics here.
    probe = np.array([-1.0, 0.0, 1e-12, LOG_EPS, 1.0], dtype=np.float32)
    want = np.array([math.log(LOG_EPS), math.log(LOG_EPS), math.log(1e-12),
                     math.log(LOG_EPS), 0.0], dtype=np.float32)
    chk(np.allclose(safe_log(probe), want, atol=1e-5),
        "safe_log clamps only x <= 0; sub-eps positives pass through")
    chk(float(mel.min()) >= math.log(1e-30),
        f"mel floor is a real magnitude, not a degenerate zero "
        f"(min {mel.min():.4f})")
    return fails


# ---------------------------------------------------------------------------
# Optional torch cross-check
# ---------------------------------------------------------------------------


def compare_weights(params: dict[str, np.ndarray], pth_path: str) -> dict:
    """Bit-compare the zarr/msgpack decode against kunato/mt3-pytorch's mt3.pth.

    This validates the checkpoint decoder and the flax [in,out] -> torch
    [out,in] transposes used by models/convert-mt3-to-gguf.py against a
    completely independent third-party conversion of the same Magenta weights.
    """
    import torch
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)

    pairs = [("decoder.token_embedder.embedding", "decoder_embed_tokens.weight", False),
             ("decoder.logits_dense.kernel", "lm_head.weight", True),
             ("encoder.continuous_inputs_projection.kernel", "proj.weight", True),
             ("encoder.encoder_norm.scale", "encoder.final_layer_norm.weight", False),
             ("decoder.decoder_norm.scale", "decoder.final_layer_norm.weight", False)]
    for i in range(ENC_LAYERS):
        e, b = f"encoder.layers_{i}", f"encoder.block.{i}"
        for a, c in [("query", "q"), ("key", "k"), ("value", "v"), ("out", "o")]:
            pairs.append((f"{e}.attention.{a}.kernel",
                          f"{b}.layer.0.SelfAttention.{c}.weight", True))
        for a in ("wi_0", "wi_1", "wo"):
            pairs.append((f"{e}.mlp.{a}.kernel",
                          f"{b}.layer.1.DenseReluDense.{a}.weight", True))
        pairs += [(f"{e}.pre_attention_layer_norm.scale",
                   f"{b}.layer.0.layer_norm.weight", False),
                  (f"{e}.pre_mlp_layer_norm.scale",
                   f"{b}.layer.1.layer_norm.weight", False)]
    for i in range(DEC_LAYERS):
        e, b = f"decoder.layers_{i}", f"decoder.block.{i}"
        for a, c in [("query", "q"), ("key", "k"), ("value", "v"), ("out", "o")]:
            pairs.append((f"{e}.self_attention.{a}.kernel",
                          f"{b}.layer.0.SelfAttention.{c}.weight", True))
            pairs.append((f"{e}.encoder_decoder_attention.{a}.kernel",
                          f"{b}.layer.1.EncDecAttention.{c}.weight", True))
        for a in ("wi_0", "wi_1", "wo"):
            pairs.append((f"{e}.mlp.{a}.kernel",
                          f"{b}.layer.2.DenseReluDense.{a}.weight", True))
        pairs += [(f"{e}.pre_self_attention_layer_norm.scale",
                   f"{b}.layer.0.layer_norm.weight", False),
                  (f"{e}.pre_cross_attention_layer_norm.scale",
                   f"{b}.layer.1.layer_norm.weight", False),
                  (f"{e}.pre_mlp_layer_norm.scale",
                   f"{b}.layer.2.layer_norm.weight", False)]

    worst = 0.0
    bad: list[str] = []
    for f, t, tr in pairs:
        a = params[f]
        if tr:
            a = a.T
        b = sd[t].numpy()
        if a.shape != b.shape:
            bad.append(f"{f}: shape {a.shape} vs {b.shape}")
            continue
        d = float(np.abs(a - b).max())
        worst = max(worst, d)
        if d > 0.0:
            bad.append(f"{f}: max|diff| {d:.3e}")

    # The oracle's own sinusoidal table, for the record.
    inv = sd["encoder.pos_emb.inv_freq"].numpy()
    half = D_MODEL // 2
    magenta_div = 10000.0 ** (-np.arange(half) / (half - 1))
    kunato_div = 10000.0 ** (-np.arange(half) / half)
    return dict(n_compared=len(pairs), n_mismatch=len(bad),
                max_abs_diff=worst, mismatches=bad[:10],
                oracle_inv_freq_matches_magenta=bool(
                    np.allclose(inv, magenta_div, atol=1e-7)),
                oracle_inv_freq_matches_d_over_2=bool(
                    np.allclose(inv, kunato_div, atol=1e-7)))


def torch_oracle_encoder(pth_path: str, mel: np.ndarray,
                         patch_pos: bool = True) -> np.ndarray:
    """Encoder output computed with **torch** ops from kunato/mt3-pytorch's
    mt3.pth -- an independent framework running the same architecture.

    IMPORTANT: kunato/mt3-pytorch's FixedPositionalEmbedding
    (mt3_infer/models/mt3_pytorch/t5.py:529-541) uses
        inv_freq = 1 / (10000 ** (arange(0, d, 2) / d))     -> 10000**(-i/256)
    while Magenta's mt3/layers.py:76 uses
        exp(arange(d/2) * -log(10000) / (d/2 - 1))          -> 10000**(-i/255)
    With patch_pos=True the table is corrected to Magenta's, which is what
    makes this a valid oracle for the Magenta checkpoint. patch_pos=False
    reproduces kunato's (divergent) behaviour so the size of that bug is
    visible rather than assumed.
    """
    import torch
    import torch.nn.functional as F

    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    T = mel.shape[0]
    x = torch.from_numpy(np.ascontiguousarray(mel)).float()

    def w(name):
        return sd[name].float()

    def tnorm(h, scale, eps=LN_EPS):
        v = h.float().pow(2).mean(-1, keepdim=True)
        return h * torch.rsqrt(v + eps) * scale

    def tattn(q, k, v):
        q = q.view(T, N_HEADS, HEAD_DIM).transpose(0, 1)
        k = k.view(T, N_HEADS, HEAD_DIM).transpose(0, 1)
        v = v.view(T, N_HEADS, HEAD_DIM).transpose(0, 1)
        # No 1/sqrt(head_dim): mt3/layers.py:230-234
        a = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
        return (a @ v).transpose(0, 1).reshape(T, N_HEADS * HEAD_DIM)

    x = x @ w("proj.weight").t()
    if patch_pos:
        pos = torch.from_numpy(sinusoidal_table()[:T]).float()
    else:
        inv = sd["encoder.pos_emb.inv_freq"].float()
        si = torch.arange(T).float()[:, None] * inv[None, :]
        pos = torch.cat([si.sin(), si.cos()], dim=-1)
    x = x + pos
    for i in range(ENC_LAYERS):
        b = f"encoder.block.{i}"
        h = tnorm(x, w(f"{b}.layer.0.layer_norm.weight"))
        q = h @ w(f"{b}.layer.0.SelfAttention.q.weight").t()
        k = h @ w(f"{b}.layer.0.SelfAttention.k.weight").t()
        v = h @ w(f"{b}.layer.0.SelfAttention.v.weight").t()
        x = x + tattn(q, k, v) @ w(f"{b}.layer.0.SelfAttention.o.weight").t()
        h = tnorm(x, w(f"{b}.layer.1.layer_norm.weight"))
        # HF/torch tanh-approximate GELU == flax nn.gelu(approximate=True)
        g = F.gelu(h @ w(f"{b}.layer.1.DenseReluDense.wi_0.weight").t(),
                   approximate="tanh")
        u = h @ w(f"{b}.layer.1.DenseReluDense.wi_1.weight").t()
        x = x + (g * u) @ w(f"{b}.layer.1.DenseReluDense.wo.weight").t()
    x = tnorm(x, w("encoder.final_layer_norm.weight"))
    return x.detach().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def write_ref_gguf(output_dir: str) -> None:
    from gguf import GGMLQuantizationType, GGUFWriter
    path = os.path.join(output_dir, "ref.gguf")
    w = GGUFWriter(path, "mt3-ref")
    for f in sorted(Path(output_dir).glob("*.npy")):
        arr = np.load(f).astype(np.float32)
        w.add_tensor(f.stem, arr, raw_dtype=GGMLQuantizationType.F32)
        print(f"  ref: {f.stem} {list(arr.shape)}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  Reference GGUF: {path}")


def load_audio(path: str) -> np.ndarray:
    try:
        import librosa
        a, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return a.astype(np.float32)
    except ImportError:
        pass
    import wave
    with wave.open(path, "rb") as fh:
        if fh.getsampwidth() != 2:
            raise SystemExit("stdlib wav reader supports 16-bit PCM only; "
                             "install librosa for anything else")
        n, ch, sr = fh.getnframes(), fh.getnchannels(), fh.getframerate()
        raw = np.frombuffer(fh.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"wav is {sr} Hz, need {SAMPLE_RATE}; "
                         f"resample first or install librosa")
    return raw.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="MT3 numpy reference dumper")
    ap.add_argument("--checkpoint", "-c", type=Path,
                    default=Path("/mnt/storage/gguf-models/mt3-src/mt3"),
                    help="T5X checkpoint dir ('checkpoint' + target.* zarr dirs)")
    ap.add_argument("--audio", "-a", required=True)
    ap.add_argument("--output-dir", "-o",
                    default="/mnt/volume1/tmp-overflow/mt3-ref")
    ap.add_argument("--max-segments", type=int, default=2,
                    help="how many 2.048 s segments to run (0 = all)")
    ap.add_argument("--decode-steps", type=int, default=64,
                    help="greedy steps per segment (upstream cap is 1024)")
    ap.add_argument("--no-decode", action="store_true",
                    help="dump the front end and encoder only")
    ap.add_argument("--no-gguf", action="store_true")
    ap.add_argument("--no-self-check", action="store_true")
    ap.add_argument("--torch-oracle", default=None,
                    help="path to mt3.pth (kunato/mt3-pytorch) for a "
                         "cross-framework encoder check")
    ap.add_argument("--torch-oracle-raw", action="store_true",
                    help="do NOT correct the oracle's sinusoidal table; shows "
                         "how far kunato/mt3-pytorch drifts from Magenta")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    codec = Codec()
    base_vocab = vocab_size(codec)

    audio = load_audio(args.audio)
    print(f"Audio: {args.audio}  {audio.shape[0]} samples "
          f"({audio.shape[0] / SAMPLE_RATE:.2f}s @ {SAMPLE_RATE} Hz)")

    segs = split_into_segments(audio)
    n_seg = len(segs) if args.max_segments in (0, None) else min(
        args.max_segments, len(segs))
    print(f"Segments: {len(segs)} total ({INPUTS_LENGTH} frames = "
          f"{INPUTS_LENGTH * HOP_WIDTH / SAMPLE_RATE:.3f}s each); "
          f"running {n_seg}")

    # seqio pads the `inputs` feature to sequence_length['inputs'] = 256 AFTER
    # the preprocessors have run, i.e. it appends ZERO MEL ROWS -- and
    # mt3/network.py:286-289 deliberately does not mask them ("letting the
    # model potentially attend to the zero vector used as padding"). Only the
    # final short segment is affected, but its encoder output changes.
    mels = []
    for _, s in segs[:n_seg]:
        m = compute_logmel(s)
        if m.shape[0] < INPUTS_LENGTH:
            m = np.pad(m, ((0, INPUTS_LENGTH - m.shape[0]), (0, 0)))
        mels.append(m)
    mel0 = mels[0]
    mel_all = np.concatenate(mels, axis=0)
    print(f"  mel: segment 0 {mel0.shape} "
          f"min {mel0.min():.4f} max {mel0.max():.4f} mean {mel0.mean():.4f}")

    print(f"Loading checkpoint: {args.checkpoint}")
    params = load_checkpoint(args.checkpoint)
    print(f"  {len(params)} parameters")
    model = MT3Numpy(params)

    dump: dict[str, np.ndarray] = {}
    enc_outs = []
    for i, m in enumerate(mels):
        d = dump if i == 0 else {}
        e = model.encode(m, dump=d)
        enc_outs.append(e)
        print(f"  segment {i}: enc_out {e.shape} "
              f"norm {np.linalg.norm(e):.3f} mean {e.mean():+.5f}")

    np.save(os.path.join(args.output_dir, "audio_segment0.npy"), segs[0][1])
    np.save(os.path.join(args.output_dir, "mel.npy"), mel0)
    np.save(os.path.join(args.output_dir, "mel_all.npy"), mel_all)
    np.save(os.path.join(args.output_dir, "enc_input.npy"), dump["enc_input"])
    np.save(os.path.join(args.output_dir, "enc_out.npy"), enc_outs[0])

    meta: dict = {
        "audio": os.path.abspath(args.audio),
        "n_samples": int(audio.shape[0]),
        "sample_rate": SAMPLE_RATE,
        "n_segments_total": len(segs),
        "n_segments_run": n_seg,
        "segment_start_times": [float(t) for t, _ in segs[:n_seg]],
        "inputs_length": INPUTS_LENGTH,
        "codec": {
            "num_classes": codec.num_classes,
            "padded_vocab": base_vocab,
            "num_special_tokens": NUM_SPECIAL_TOKENS,
            "ranges": {t: list(codec.event_type_range(t))
                       for t, _, _ in EVENT_RANGES},
        },
    }

    if not args.no_decode:
        state = NoteDecodingState()
        per_segment = []
        first_logits = None
        for i, e in enumerate(enc_outs):
            toks, logits = model.greedy(e, args.decode_steps,
                                        collect_logits=(i == 0))
            if i == 0:
                first_logits = logits
            start_time = segs[i][0]
            max_time = segs[i + 1][0] if i + 1 < n_seg else None
            # mt3/metrics_utils.py:99 begin_segment_fn ->
            # mt3/note_sequences.py:390-393 begin_tied_pitches_section
            state.tied_pitches = set()
            state.is_tie_section = True
            inv, drop, events = decode_segment(state, toks, start_time,
                                               max_time, codec, base_vocab)
            print(f"  segment {i}: {len(toks)} tokens, {len(events)} events, "
                  f"{inv} invalid, {drop} dropped")
            per_segment.append(dict(index=i, start_time=start_time,
                                    max_decode_time=max_time,
                                    tokens=[int(t) for t in toks],
                                    invalid=inv, dropped=drop, events=events))
        notes = flush(state)
        meta["segments"] = per_segment
        meta["n_notes"] = len(notes)
        meta["notes"] = notes
        print(f"  notes after tie assembly: {len(notes)}")
        if first_logits is not None and first_logits.size:
            np.save(os.path.join(args.output_dir, "logits_step0.npy"),
                    first_logits[0])
            np.save(os.path.join(args.output_dir, "logits_prefix.npy"),
                    first_logits)
            print(f"  logits_step0 {first_logits[0].shape} "
                  f"argmax {int(np.argmax(first_logits[0]))}")

    if args.torch_oracle:
        print("Torch cross-check (kunato/mt3-pytorch, mt3.pth):")
        wc = compare_weights(params, args.torch_oracle)
        print(f"  weights: compared {wc['n_compared']} tensors, "
              f"{wc['n_mismatch']} mismatches, "
              f"max|abs diff| = {wc['max_abs_diff']:.3e}")
        for m in wc["mismatches"]:
            print(f"    {m}")
        print(f"  oracle inv_freq == Magenta 10000**(-i/255): "
              f"{wc['oracle_inv_freq_matches_magenta']}")
        print(f"  oracle inv_freq == kunato  10000**(-i/256): "
              f"{wc['oracle_inv_freq_matches_d_over_2']}")

        res = {}
        for patched in ((True, False) if not args.torch_oracle_raw else (False,)):
            ref = torch_oracle_encoder(args.torch_oracle, mel0,
                                       patch_pos=patched)
            cos = cosine(enc_outs[0], ref)
            mad = float(np.abs(enc_outs[0] - ref).max())
            tag = "pos patched to Magenta" if patched else "pos RAW (kunato)"
            print(f"  enc_out [{tag:<22s}] cos = {cos:.9f}  "
                  f"max|abs diff| = {mad:.3e}")
            res["patched" if patched else "raw"] = dict(cos=cos, max_abs_diff=mad)
        meta["torch_oracle"] = dict(weights=wc, encoder=res)

    if not args.no_self_check:
        print("Self-checks:")
        fails = self_check(codec, base_vocab, mel0, mel_all, segs[:n_seg])
        meta["self_check_failures"] = fails
        if fails:
            print(f"  {len(fails)} SELF-CHECK FAILURE(S)", file=sys.stderr)

    with open(os.path.join(args.output_dir, "tokens.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"  tokens.json -> {args.output_dir}/tokens.json")

    if not args.no_gguf:
        write_ref_gguf(args.output_dir)

    return 1 if meta.get("self_check_failures") else 0


if __name__ == "__main__":
    sys.exit(main())
