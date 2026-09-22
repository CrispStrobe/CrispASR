#!/usr/bin/env python3
"""
Convert hFT-Transformer (Toyama et al., ISMIR 2023) -> GGUF.

Source is the PRUNED ONNX export, `hft_transformer.pruned.onnx`, not the full
one. The full export keeps all fifteen forward outputs, two of which nobody
decodes — `enc_vector` at [1, 128, 4, 88, 256] is 11.5 M floats on its own —
and could not be loaded and run at all on the box this port was measured on.
`bench/tool/prune_hft.py` in CrispStrobe/flutter_tuner cuts the graph to
`onset_B`, `offset_B`, `mpe_B`, `velocity_B` (1,621 -> 1,517 nodes). The
weights are identical either way; this script only ever reads initializers.

ARCHITECTURE (read out of the graph, not out of the paper)

  input `spec`: [1, 256 mel bins, 192 frames] — log-mel, BIN-major.
    192 = 128 answered frames + 32 margin frames at each end.

  ENCODER, one sequence per answered frame (batch 128, 256 freq tokens):
      for t in 0..127:  win = spec[:, :, t : t+65]          [256 bins, 65]
      Conv2d(1, 4, kernel (1, 5), no padding)               [256, 4, 61]
      flatten channel-major -> [256, 244]
      Linear(244, 256) * sqrt(256) + pos_embedding_freq[0:256]
      3 x encoder layer:
          x = LN(x + SelfAttn(x));  x = LN(x + FF(x))       (ONE LayerNorm
                                                             module, applied
                                                             twice)

  DECODER-FREQ, 88 pitch tokens attending to the 256 encoder tokens
  (batch 128 frames):
      x = pos_embedding_freq_dec[0:88]                      (no sqrt scale)
      layer_zero_freq : x = LN(x + CrossAttn(x, enc)); x = LN(x + FF(x))
      2 x layers_freq : x = LN(x + SelfAttn(x))
                        x = LN(x + CrossAttn(x, enc))
                        x = LN(x + FF(x))
      -> [128 frames, 88 pitches, 256]  -> transpose -> [88, 128, 256]

  DECODER-TIME, 128 time tokens (batch 88 pitches):
      x = x * sqrt(256) + pos_embedding_time[0:128]
      3 x layers_time : x = LN(x + SelfAttn(x)); x = LN(x + FF(x))
      onset/offset/mpe = Linear(256, 1);  velocity = Linear(256, 128)

  All heads emit LOGITS. All attention is 4 heads x 64, scale 1/8. Every
  `layer_norm` in the checkpoint is ONE module used two or three times inside
  its layer — there is no second set of gains — and eps is 1e-5.

THE FUSED FRONT END.  The convolution and `tok_embedding_freq` are both
linear in the 65-tap window and neither has a nonlinearity between them, so
they collapse into a single Linear(65, 256) applied per (frame, bin):

    K[m, d] = sum_c sum_{j: 0<=j<=60, 0<=m-j<=4} W_tok[c*61 + j, d] * Wc[c, m-j]
    b[d]    = b_tok[d] + sum_c b_conv[c] * sum_j W_tok[c*61 + j, d]

That is exact arithmetic, not an approximation, and it replaces a Conv plus a
244x256 GEMM with a 65x256 one — 16,640 weights where the graph carries
62,464 — and removes two im2col passes from the runtime. `--verify-fusion`
checks it against an explicit conv+matmul on random input and asserts the
residual is at f32 rounding level. The reordered summation is the only
difference and it shows up in tools/hft_parity.py as ~1e-6, alongside
everything else.

QUANTISATION.  --quant q8_0 / q4_0 quantises the 2-D matrices that are
consumed by `ggml_mul_mat` and whose contraction axis is a multiple of 32:
every attention projection, every feed-forward matrix and the velocity head.
The positional embeddings are ADDED, not multiplied, so they stay F32; the
fused front end's contraction axis is 65 and stays F32; the three scalar heads
are [1, 256] and are left F32 because quantising 256 floats saves nothing and
they sit directly under a sigmoid.

Usage:
    python models/convert-hft-transformer-to-gguf.py \\
        --input  /mnt/storage/tuner-bench/onnx/hft_transformer.pruned.onnx \\
        --output /mnt/storage/gguf-models/hft-transformer-f32.gguf
    ... --quant q8_0 --output hft-transformer-q8_0.gguf
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("pip install onnx")

try:
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")


# --- hyperparameters, every one asserted against the graph below -----------

SAMPLE_RATE = 16000
N_FFT = 2048
HOP = 256
N_MELS = 256
F_MIN = 0.0
F_MAX = 8000.0
MEL_EPS = 1e-8            # log(mel + 1e-8), an ADD not a clamp

N_FRAME = 128             # frames answered per window
N_MARGIN = 32             # margin frames each side; window = 128 + 2*32 = 192
CNN_CHANNEL = 4
CNN_KERNEL = 5
CNN_TAPS = 2 * N_MARGIN + 1          # 65
CNN_OUT = CNN_TAPS - CNN_KERNEL + 1  # 61
HID = 256
N_HEADS = 4
PF_DIM = 512
CLASSES = 88
BEGIN_NOTE = 21           # MIDI A0
VELOCITY_BINS = 128
LN_EPS = 1e-5
ENC_LAYERS = 3
DEC_FREQ_LAYERS = 2       # on top of layer_zero_freq
DEC_TIME_LAYERS = 3

QUANT_TYPES = {
    "q4_0": GGMLQuantizationType.Q4_0,
    "q8_0": GGMLQuantizationType.Q8_0,
    "f16": None,          # handled separately
}


# --- front end -------------------------------------------------------------

def hz_to_mel_htk(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz_htk(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """[n_mels, n_freqs] — librosa's `htk=True, norm='slaney'`, fmin 0, fmax 8000.

    hFT's front end is 256 mel bins, n_fft 2048, hop 256, power 2.0, CONSTANT
    (zero) padding and `log(mel + 1e-8)`. Three of those are non-default
    somewhere — a periodic Hann window, the HTK mel scale and slaney filter
    normalisation — and collectively fatal if any is wrong; §35.1 of the
    flutter_tuner benchmark checked this exact formula against librosa to
    6.8e-08 and this is that formula.
    """
    n_freqs = N_FFT // 2 + 1
    all_freqs = np.arange(n_freqs, dtype=np.float64) * SAMPLE_RATE / N_FFT
    m_min, m_max = hz_to_mel_htk(F_MIN), hz_to_mel_htk(F_MAX)
    f_pts = mel_to_hz_htk(m_min + (m_max - m_min) * np.arange(N_MELS + 2) / (N_MELS + 1))

    fb = np.zeros((N_MELS, n_freqs), dtype=np.float64)
    for m in range(N_MELS):
        lower, centre, upper = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        down = (all_freqs - lower) / (centre - lower)
        up = (upper - all_freqs) / (upper - centre)
        enorm = 2.0 / (upper - lower)        # slaney: unit area
        fb[m] = np.maximum(np.minimum(down, up), 0.0) * enorm
    return fb.astype(np.float32)


def hann_periodic() -> np.ndarray:
    """librosa/torch `hann` — PERIODIC (divide by N), not symmetric (N-1)."""
    n = np.arange(N_FFT, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * math.pi * n / N_FFT)).astype(np.float32)


def check_mel_against_librosa() -> None:
    try:
        import librosa
    except ImportError:
        print("  [check-mel] librosa not installed - skipped")
        return
    ref = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                              fmin=F_MIN, fmax=F_MAX, htk=True, norm="slaney")
    d = float(np.abs(ref - mel_filterbank()).max())
    w = float(np.abs(librosa.filters.get_window("hann", N_FFT, fftbins=True)
                     - hann_periodic()).max())
    print(f"  [check-mel] filterbank max abs diff vs librosa: {d:.3e}")
    print(f"  [check-mel] window     max abs diff vs librosa: {w:.3e}")
    assert d < 1e-6 and w < 1e-6, "front end does not match librosa"


# --- graph reading ---------------------------------------------------------

class Graph:
    def __init__(self, path: Path):
        self.model = onnx.load(str(path))
        self.g = self.model.graph
        self.init = {t.name: numpy_helper.to_array(t) for t in self.g.initializer}
        self.by_output = {}
        self.by_name = {}
        for n in self.g.node:
            for o in n.output:
                self.by_output[o] = n
            if n.name:
                self.by_name[n.name] = n

    def matmul_weight(self, node_name: str) -> np.ndarray:
        """The initializer operand of the MatMul the exporter named `node_name`.

        ONNX stores Linear as `MatMul(x, W)` with W already [in, out], so the
        array returned here contracts on its FIRST axis.
        """
        n = self.by_name.get(node_name)
        if n is None or n.op_type != "MatMul":
            sys.exit(f"no MatMul node named {node_name!r} — graph shape changed")
        for i in n.input:
            if i in self.init:
                return self.init[i]
        sys.exit(f"{node_name!r} has no initializer operand")

    def arr(self, name: str) -> np.ndarray:
        if name not in self.init:
            sys.exit(f"no initializer named {name!r} — graph shape changed")
        return self.init[name]


def linear(gr: Graph, mm_node: str, bias_init: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (W as [out, in] for ggml_mul_mat, b as [out])."""
    w = gr.matmul_weight(mm_node)
    b = gr.arr(bias_init)
    assert w.shape[1] == b.shape[0], (mm_node, w.shape, b.shape)
    return np.ascontiguousarray(w.T), np.ascontiguousarray(b)


def fuse_front_end(gr: Graph) -> tuple[np.ndarray, np.ndarray]:
    """Conv2d(1,4,(1,5)) + Linear(244, 256)  ->  Linear(65, 256).

    Returns (K as [256, 65] for ggml_mul_mat, b as [256]).
    """
    wc = gr.arr("encoder.conv.weight")          # [4, 1, 1, 5]
    bc = gr.arr("encoder.conv.bias")            # [4]
    assert wc.shape == (CNN_CHANNEL, 1, 1, CNN_KERNEL), wc.shape
    wt = gr.matmul_weight("/encoder/tok_embedding_freq/MatMul")   # [244, 256]
    bt = gr.arr("encoder.tok_embedding_freq.bias")                # [256]
    assert wt.shape == (CNN_CHANNEL * CNN_OUT, HID), wt.shape

    wc2 = wc.reshape(CNN_CHANNEL, CNN_KERNEL).astype(np.float64)
    wt3 = wt.reshape(CNN_CHANNEL, CNN_OUT, HID).astype(np.float64)

    K = np.zeros((CNN_TAPS, HID), dtype=np.float64)
    for c in range(CNN_CHANNEL):
        for j in range(CNN_OUT):
            for k in range(CNN_KERNEL):
                K[j + k] += wt3[c, j] * wc2[c, k]
    b = bt.astype(np.float64) + np.einsum("c,cjd->d", bc.astype(np.float64), wt3)
    return np.ascontiguousarray(K.T.astype(np.float32)), b.astype(np.float32)


def verify_fusion(gr: Graph) -> None:
    """The fused Linear(65, 256) against an explicit conv + 244x256 matmul."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((37, CNN_TAPS)).astype(np.float32)   # 37 (bin, frame) rows

    wc = gr.arr("encoder.conv.weight").reshape(CNN_CHANNEL, CNN_KERNEL)
    bc = gr.arr("encoder.conv.bias")
    wt = gr.matmul_weight("/encoder/tok_embedding_freq/MatMul")
    bt = gr.arr("encoder.tok_embedding_freq.bias")

    conv = np.zeros((x.shape[0], CNN_CHANNEL, CNN_OUT), dtype=np.float32)
    for c in range(CNN_CHANNEL):
        for j in range(CNN_OUT):
            conv[:, c, j] = x[:, j:j + CNN_KERNEL] @ wc[c] + bc[c]
    ref = conv.reshape(x.shape[0], CNN_CHANNEL * CNN_OUT) @ wt + bt

    K, b = fuse_front_end(gr)
    got = x @ K.T + b
    d = float(np.abs(got - ref).max())
    rel = d / float(np.abs(ref).max())
    print(f"  [verify-fusion] max abs diff {d:.3e}  (relative {rel:.3e})")
    assert rel < 1e-5, "the fused front end is not the conv + tok_embedding"


def collect_attention(gr: Graph, onnx_prefix: str, out_prefix: str,
                      tensors: dict[str, np.ndarray]) -> None:
    for which in ("q", "k", "v", "o"):
        w, b = linear(gr, f"{onnx_prefix}/fc_{which}/MatMul", f"{_dot(onnx_prefix)}.fc_{which}.bias")
        assert w.shape == (HID, HID), (onnx_prefix, which, w.shape)
        tensors[f"{out_prefix}.{which}.weight"] = w
        tensors[f"{out_prefix}.{which}.bias"] = b


def collect_ff(gr: Graph, onnx_prefix: str, out_prefix: str,
               tensors: dict[str, np.ndarray]) -> None:
    w1, b1 = linear(gr, f"{onnx_prefix}/positionwise_feedforward/fc_1/MatMul",
                    f"{_dot(onnx_prefix)}.positionwise_feedforward.fc_1.bias")
    w2, b2 = linear(gr, f"{onnx_prefix}/positionwise_feedforward/fc_2/MatMul",
                    f"{_dot(onnx_prefix)}.positionwise_feedforward.fc_2.bias")
    assert w1.shape == (PF_DIM, HID) and w2.shape == (HID, PF_DIM), (w1.shape, w2.shape)
    tensors[f"{out_prefix}.ff1.weight"] = w1
    tensors[f"{out_prefix}.ff1.bias"] = b1
    tensors[f"{out_prefix}.ff2.weight"] = w2
    tensors[f"{out_prefix}.ff2.bias"] = b2


def collect_ln(gr: Graph, module: str, out_prefix: str,
               tensors: dict[str, np.ndarray]) -> None:
    """ONE LayerNorm per layer, applied twice or three times. The checkpoint
    has a single `layer_norm.weight` per module and the graph reuses it; a
    port that allocated one set of gains per application would run and be
    wrong by however much the second application differs."""
    tensors[f"{out_prefix}.ln.weight"] = gr.arr(f"{module}.layer_norm.weight")
    tensors[f"{out_prefix}.ln.bias"] = gr.arr(f"{module}.layer_norm.bias")


def _dot(onnx_prefix: str) -> str:
    """'/encoder/layers_freq.0/self_attention' -> 'encoder.layers_freq.0.self_attention'"""
    return onnx_prefix.strip("/").replace("/", ".")


# --- writing ---------------------------------------------------------------

QUANT_SKIP_SUFFIX = (".bias",)
QUANT_SKIP_EXACT = ("hft.mel_fb", "hft.window",
                    "hft.encoder.pos_freq", "hft.decoder.pos_freq",
                    "hft.decoder.pos_time",
                    "hft.encoder.front.weight",
                    "hft.head.onset.weight", "hft.head.offset.weight",
                    "hft.head.mpe.weight")


def quantisable(name: str, arr: np.ndarray) -> bool:
    if arr.ndim != 2 or name in QUANT_SKIP_EXACT:
        return False
    if name.endswith(QUANT_SKIP_SUFFIX) or name.endswith(".ln.weight"):
        return False
    return arr.shape[-1] % 32 == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/mnt/storage/tuner-bench/onnx/hft_transformer.pruned.onnx")
    ap.add_argument("--output", required=True)
    ap.add_argument("--quant", default=None, choices=sorted(QUANT_TYPES),
                    help="q4_0 | q8_0 | f16 (default: F32 throughout)")
    ap.add_argument("--check-mel", action="store_true")
    ap.add_argument("--verify-fusion", action="store_true")
    args = ap.parse_args()

    if args.check_mel:
        check_mel_against_librosa()

    gr = Graph(Path(args.input))
    outs = [o.name for o in gr.g.output]
    expected = ["onset_B", "offset_B", "mpe_B", "velocity_B"]
    if outs != expected:
        print(f"  [warn] graph outputs are {outs}, expected the pruned set {expected}")
        if not set(expected).issubset(outs):
            sys.exit("this is not an hFT-Transformer export")

    if args.verify_fusion:
        verify_fusion(gr)

    tensors: dict[str, np.ndarray] = {}

    # Fused conv + tok_embedding_freq.
    K, kb = fuse_front_end(gr)
    tensors["hft.encoder.front.weight"] = K
    tensors["hft.encoder.front.bias"] = kb

    # Positional embeddings. These are Gather'd by index, so they stay in
    # [position, dim] row-major and are consumed by ggml_add.
    tensors["hft.encoder.pos_freq"] = gr.arr("encoder.pos_embedding_freq.weight")
    tensors["hft.decoder.pos_freq"] = gr.arr("decoder.pos_embedding_freq.weight")
    tensors["hft.decoder.pos_time"] = gr.arr("decoder.pos_embedding_time.weight")
    assert tensors["hft.encoder.pos_freq"].shape == (N_MELS, HID)
    assert tensors["hft.decoder.pos_freq"].shape == (CLASSES, HID)
    assert tensors["hft.decoder.pos_time"].shape == (N_FRAME, HID)

    for i in range(ENC_LAYERS):
        p = f"/encoder/layers_freq.{i}"
        collect_attention(gr, f"{p}/self_attention", f"hft.enc.{i}.attn", tensors)
        collect_ff(gr, p, f"hft.enc.{i}", tensors)
        collect_ln(gr, f"encoder.layers_freq.{i}", f"hft.enc.{i}", tensors)

    # Decoder-freq layer 0 is `layer_zero_freq`: cross-attention only.
    collect_attention(gr, "/decoder/layer_zero_freq/encoder_attention",
                      "hft.decfreq.0.xattn", tensors)
    collect_ff(gr, "/decoder/layer_zero_freq", "hft.decfreq.0", tensors)
    collect_ln(gr, "decoder.layer_zero_freq", "hft.decfreq.0", tensors)

    for i in range(DEC_FREQ_LAYERS):
        p = f"/decoder/layers_freq.{i}"
        collect_attention(gr, f"{p}/self_attention", f"hft.decfreq.{i + 1}.attn", tensors)
        collect_attention(gr, f"{p}/encoder_attention", f"hft.decfreq.{i + 1}.xattn", tensors)
        collect_ff(gr, p, f"hft.decfreq.{i + 1}", tensors)
        collect_ln(gr, f"decoder.layers_freq.{i}", f"hft.decfreq.{i + 1}", tensors)

    for i in range(DEC_TIME_LAYERS):
        p = f"/decoder/layers_time.{i}"
        collect_attention(gr, f"{p}/self_attention", f"hft.dectime.{i}.attn", tensors)
        collect_ff(gr, p, f"hft.dectime.{i}", tensors)
        collect_ln(gr, f"decoder.layers_time.{i}", f"hft.dectime.{i}", tensors)

    for head, bias in (("onset", "decoder.fc_onset_time.bias"),
                       ("offset", "decoder.fc_offset_time.bias"),
                       ("mpe", "decoder.fc_mpe_time.bias"),
                       ("velocity", "decoder.fc_velocity_time.bias")):
        w, b = linear(gr, f"/decoder/fc_{head}_time/MatMul", bias)
        tensors[f"hft.head.{head}.weight"] = w
        tensors[f"hft.head.{head}.bias"] = b
    assert tensors["hft.head.onset.weight"].shape == (1, HID)
    assert tensors["hft.head.velocity.weight"].shape == (VELOCITY_BINS, HID)

    tensors["hft.mel_fb"] = mel_filterbank()
    tensors["hft.window"] = hann_periodic()

    quant_type = QUANT_TYPES.get(args.quant) if args.quant else None
    as_f16 = args.quant == "f16"

    writer = GGUFWriter(args.output, "hft-transformer")
    writer.add_uint32("hft.sample_rate", SAMPLE_RATE)
    writer.add_uint32("hft.n_fft", N_FFT)
    writer.add_uint32("hft.hop_size", HOP)
    writer.add_uint32("hft.n_mels", N_MELS)
    writer.add_float32("hft.fmin", F_MIN)
    writer.add_float32("hft.fmax", F_MAX)
    writer.add_float32("hft.mel_eps", MEL_EPS)
    writer.add_uint32("hft.n_frame", N_FRAME)
    writer.add_uint32("hft.n_margin", N_MARGIN)
    writer.add_uint32("hft.classes_num", CLASSES)
    writer.add_uint32("hft.begin_note", BEGIN_NOTE)
    writer.add_uint32("hft.hidden", HID)
    writer.add_uint32("hft.n_heads", N_HEADS)
    writer.add_uint32("hft.pf_dim", PF_DIM)
    writer.add_uint32("hft.velocity_bins", VELOCITY_BINS)
    writer.add_uint32("hft.enc_layers", ENC_LAYERS)
    writer.add_uint32("hft.dec_freq_layers", DEC_FREQ_LAYERS + 1)
    writer.add_uint32("hft.dec_time_layers", DEC_TIME_LAYERS)
    writer.add_float32("hft.ln_eps", LN_EPS)
    writer.add_uint32("hft.front_taps", CNN_TAPS)
    # The runtime must not silently disagree with the converter about whether
    # the conv is folded into the embedding; if it ever needs the unfused form
    # this string is how it finds out.
    writer.add_string("hft.front_end", "fused-conv-tok-embedding")

    n_q = n_f16 = n_f32 = 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name]).astype(np.float32)
        raw_dtype = None
        if quant_type is not None and quantisable(name, arr):
            arr = gguf.quantize(arr, quant_type)
            raw_dtype = quant_type
            n_q += 1
        elif as_f16 and quantisable(name, arr):
            arr = arr.astype(np.float16)
            n_f16 += 1
        else:
            n_f32 += 1
        writer.add_tensor(name, arr, raw_dtype=raw_dtype)
        label = raw_dtype.name if raw_dtype else str(arr.dtype)
        print(f"  {name:34s} {str(tensors[name].shape):16s} {label}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = Path(args.output).stat().st_size
    print(f"\n  {len(tensors)} tensors  (quant: {n_q}, F16: {n_f16}, F32: {n_f32})")
    print(f"  wrote {args.output}  ({size / 1e6:.1f} MB, {size / 1048576.0:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
