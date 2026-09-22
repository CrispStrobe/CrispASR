#!/usr/bin/env python3
"""
Convert Onsets & Frames (Hawthorne et al. 2018) -> GGUF.

Source is the ONNX export rather than a PyTorch checkpoint, for one reason
that matters: `torch.onnx.export` has already folded every BatchNorm into the
Conv that precedes it, so the graph carries 12 biased convolutions where the
checkpoint carries 12 convolutions plus 12 BatchNorms with running statistics.
Folding them here would mean reproducing the eps convention exactly, and
`src/piano_transcription.cpp`'s PIANO_BN2D_EPS comment is this repo's record of
how expensive getting that wrong is. The exporter already did it correctly.

  Default input: /mnt/storage/tuner-bench/onnx/onsets_and_frames.onnx

ARCHITECTURE (verified against the graph, not against a paper)

  input: log-mel, [1, frames, 229]

  ConvStack (x4 - onset, offset, frame, velocity; all four read the SAME mel):
      Conv2d(1, 48, 3x3, pad 1) + ReLU
      Conv2d(48, 48, 3x3, pad 1) + ReLU
      MaxPool2d((1,2))                     229 -> 114
      Conv2d(48, 96, 3x3, pad 1) + ReLU
      MaxPool2d((1,2))                     114 -> 57
      transpose to [B, T, C, F], flatten -> [B, T, 96*57 = 5472]
      Linear(5472, 768)

  onset_stack  : ConvStack -> BiLSTM(768 -> 384) -> Linear(768, 88) => "onset"
  offset_stack : ConvStack -> BiLSTM(768 -> 384) -> Linear(768, 88) => "offset"
  frame_stack  : ConvStack -> Linear(768, 88)                       => activation
  velocity_stack: ConvStack -> Linear(768, 88)                      => velocity
  combined_stack: cat(onset, offset, activation) [264]
                  -> BiLSTM(264 -> 384) -> Linear(768, 88)          => frame

THE OUTPUT-NAME SHIFT.  The export was handed four `output_names` for a
five-output forward, so every name slid down one slot: the graph output called
`frame` is the pre-combination activation and the one called `velocity` is the
real frame head.  Verified structurally here rather than assumed - `--verify`
walks the graph and asserts that the output named `velocity` is the one whose
producer consumes the Concat of `onset`, `offset` and `frame`.  This converter
therefore writes the *meanings*, not the names: what lands in the GGUF as
`oaf.frame.head.*` is the combined stack's head.

LAYOUT NOTES (the things that are wrong-but-runnable if you get them backwards)

  * Linear weights arrive from ONNX MatMul as [in, out] and are transposed to
    [out, in] here, because `ggml_mul_mat(A, B)` wants ne0 of A to be the
    contraction axis.
  * LSTM weights are ONNX's, which means gate order **iofc** - NOT PyTorch's
    ifgo.  They are split per direction and stored as [4H, in] / [4H, H] with
    the ONNX gate order preserved; `src/onsets_and_frames.cpp` reads them in
    that order and the constant OAF_LSTM_GATE_ORDER records it.
  * The mel filterbank and the analysis window are COMPUTED HERE and stored in
    the GGUF (`oaf.mel_fb`, `oaf.window`), following the rule in
    crispasr-crispembed-dev.md 2b: a front end with four non-default
    conventions (periodic Hann, HTK mel scale, slaney filter normalisation,
    power 1.0) is a scale bug waiting to happen, and shipping the basis makes
    the question unaskable.  `--check-mel` diffs them against torchaudio when
    it is installed.

QUANTISATION.  --quant q4_0 / q8_0 quantises the 2-D weight matrices whose
contraction axis is a multiple of 32 (the fc, the LSTM input projections and
the recurrent matrices).  Convolution kernels, biases, the filterbank, the
window and the combined stack's input projection (264 is not a multiple of 32)
stay F32.

Usage:
    python models/convert-onsets-and-frames-to-gguf.py \
        --input  /mnt/storage/tuner-bench/onnx/onsets_and_frames.onnx \
        --output /mnt/storage/gguf-models/onsets-and-frames-f32.gguf
    ... --quant q4_0 --output onsets-and-frames-q4_0.gguf
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


# --- hyperparameters, all read back out of the graph and asserted ----------

SAMPLE_RATE = 16000
N_FFT = 2048
HOP = 512
N_MELS = 229
F_MIN = 30.0
F_MAX = 8000.0
CLASSES = 88
BEGIN_NOTE = 21          # MIDI A0
LSTM_HIDDEN = 384
FC_OUT = 768
MIDFEAT = 5472           # 96 channels * (229 // 4) bins

# ONNX LSTM gate order. Recorded in the GGUF so the runtime cannot silently
# disagree with the converter about it.
LSTM_GATE_ORDER = "iofc"

QUANT_TYPES = {
    "q4_0": GGMLQuantizationType.Q4_0,
    "q8_0": GGMLQuantizationType.Q8_0,
    "f16": None,   # handled separately
}


# --- front end -------------------------------------------------------------

def hz_to_mel_htk(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz_htk(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """[n_mels, n_freqs], `mel_scale='htk'`, `norm='slaney'` - torchaudio's.

    Four settings here are non-default somewhere, and each is individually
    small: HTK rather than Slaney's mel scale, `slaney` filter normalisation
    (unit AREA, not unit peak - a factor of two to three across the band),
    f_min 30 rather than 0, and f_max 8000 rather than sr/2.  bench/lib/mel.dart
    in CrispStrobe/flutter_tuner is the same formula checked against librosa to
    6.1e-08; this is that formula.
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
        tri = np.minimum(down, up)
        enorm = 2.0 / (upper - lower)     # slaney: unit area
        fb[m] = np.maximum(tri, 0.0) * enorm
    return fb.astype(np.float32)


def hann_periodic() -> np.ndarray:
    """`torch.hann_window` - PERIODIC (divide by N), not symmetric (N-1)."""
    n = np.arange(N_FFT, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * math.pi * n / N_FFT)).astype(np.float32)


def check_mel_against_torchaudio() -> None:
    try:
        import torch
        import torchaudio
    except ImportError:
        print("  [check-mel] torchaudio not installed - skipped")
        return
    ref = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
        f_min=F_MIN, f_max=F_MAX, n_mels=N_MELS, power=1.0,
        norm="slaney", mel_scale="htk", center=True, pad_mode="reflect",
    )
    fb_ref = ref.mel_scale.fb.numpy().T           # torchaudio: [n_freqs, n_mels]
    fb_mine = mel_filterbank()
    d = np.abs(fb_ref - fb_mine).max()
    print(f"  [check-mel] filterbank max abs diff vs torchaudio: {d:.3e}")
    w_ref = torch.hann_window(N_FFT).numpy()
    dw = np.abs(w_ref - hann_periodic()).max()
    print(f"  [check-mel] window     max abs diff vs torch:      {dw:.3e}")
    assert d < 1e-6 and dw < 1e-6, "front end does not match torchaudio"


# --- graph reading ---------------------------------------------------------

class Graph:
    def __init__(self, path: Path):
        self.model = onnx.load(str(path))
        self.g = self.model.graph
        self.init = {t.name: numpy_helper.to_array(t) for t in self.g.initializer}
        self.by_output = {}
        for n in self.g.node:
            for o in n.output:
                self.by_output[o] = n

    def arr(self, name: str) -> np.ndarray:
        return self.init[name]


def verify_output_shift(gr: Graph) -> dict[str, str]:
    """Establish, from the graph's own wiring, which output is which head.

    Returns a map meaning -> graph output name.  Any deviation from the known
    shift is fatal: a silently renamed export would otherwise be decoded with
    the activation head standing in for the frame head, which §35.2 of the
    flutter_tuner benchmark measured at 9.1% -> 5.6% F1 with offsets required.
    """
    outs = [o.name for o in gr.g.output]
    if len(outs) != 5:
        sys.exit(f"expected 5 graph outputs, found {len(outs)}: {outs}")

    # The combined stack is the only consumer of a Concat of three other
    # outputs. Find that Concat, then walk forward to the graph output.
    concat = None
    for n in gr.g.node:
        if n.op_type == "Concat" and len(n.input) == 3 and all(i in outs for i in n.input):
            concat = n
            break
    if concat is None:
        sys.exit("no Concat of three graph outputs found - graph shape changed")

    combined_out = None
    frontier = list(concat.output)
    seen = set()
    while frontier:
        t = frontier.pop()
        if t in seen:
            continue
        seen.add(t)
        if t in outs:
            combined_out = t
            break
        for n in gr.g.node:
            if t in n.input:
                frontier.extend(n.output)
    if combined_out is None:
        sys.exit("the Concat does not reach any graph output")

    onset_name, offset_name, activation_name = concat.input
    velocity_name = next(o for o in outs
                         if o not in (onset_name, offset_name, activation_name, combined_out))

    mapping = {
        "onset": onset_name,
        "offset": offset_name,
        "activation": activation_name,
        "frame": combined_out,
        "velocity": velocity_name,
    }
    print("  [verify] head -> graph output name (the export's names are shifted by one):")
    for k, v in mapping.items():
        shifted = " <- NOT what the name says" if k != v else ""
        print(f"    {k:11s} = {v!r}{shifted}")
    if mapping["frame"] != "velocity" or mapping["activation"] != "frame":
        print("  [verify] NOTE: this export does not have the known one-slot shift.")
    return mapping


def producer_chain(gr: Graph, out_name: str) -> list:
    """Nodes feeding `out_name`, nearest first, stopping at graph inputs."""
    chain, frontier, seen = [], [out_name], set()
    while frontier:
        t = frontier.pop(0)
        if t in seen or t in gr.init:
            continue
        seen.add(t)
        n = gr.by_output.get(t)
        if n is None:
            continue
        chain.append(n)
        frontier.extend(n.input)
    return chain


def collect_stack(gr: Graph, prefix: str) -> dict[str, np.ndarray]:
    """Pull one *_stack's weights out by the exporter's node-name prefix."""
    out: dict[str, np.ndarray] = {}
    convs, matmuls, adds, lstms = [], [], [], []
    for n in gr.g.node:
        if not n.name.startswith(prefix):
            continue
        if n.op_type == "Conv":
            convs.append(n)
        elif n.op_type == "MatMul":
            matmuls.append(n)
        elif n.op_type == "Add":
            adds.append(n)
        elif n.op_type == "LSTM":
            lstms.append(n)

    for i, n in enumerate(convs):
        w = gr.arr(n.input[1])
        b = gr.arr(n.input[2])
        out[f"conv{i}.weight"] = w        # [OC, IC, KH, KW] -> ggml ne [KW,KH,IC,OC]
        out[f"conv{i}.bias"] = b
    # MatMuls appear in graph order: fc first (if the stack has a ConvStack),
    # then the head. The head's contraction axis is 768.
    for n in matmuls:
        w = gr.arr(n.input[1])            # [in, out]
        bias_node = next(a for a in adds if n.output[0] in a.input)
        b = gr.arr(next(i for i in bias_node.input if i in gr.init))
        key = "fc" if w.shape[0] == MIDFEAT else "head"
        out[f"{key}.weight"] = np.ascontiguousarray(w.T)   # -> [out, in]
        out[f"{key}.bias"] = b
    for n in lstms:
        W = gr.arr(n.input[1])            # [2, 4H, in]   gate order iofc
        R = gr.arr(n.input[2])            # [2, 4H, H]
        B = gr.arr(n.input[3])            # [2, 8H]       Wb then Rb
        assert W.shape[0] == 2 and W.shape[1] == 4 * LSTM_HIDDEN, W.shape
        for d, tag in ((0, "fwd"), (1, "rev")):
            out[f"lstm.{tag}.W"] = np.ascontiguousarray(W[d])
            out[f"lstm.{tag}.R"] = np.ascontiguousarray(R[d])
            # Fold ONNX's two bias vectors into one: the recurrence adds both
            # every step, so Wb+Rb is arithmetically identical and halves the
            # per-step adds.
            out[f"lstm.{tag}.b"] = np.ascontiguousarray(
                B[d, :4 * LSTM_HIDDEN] + B[d, 4 * LSTM_HIDDEN:])
    return out


# --- writing ---------------------------------------------------------------

def quantisable(name: str, arr: np.ndarray) -> bool:
    """Only 2-D matrices whose contraction axis is a multiple of 32."""
    if arr.ndim != 2:
        return False
    if name.endswith(".bias") or name.endswith(".b"):
        return False
    return arr.shape[-1] % 32 == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/mnt/storage/tuner-bench/onnx/onsets_and_frames.onnx")
    ap.add_argument("--output", required=True)
    ap.add_argument("--quant", default=None, choices=sorted(QUANT_TYPES),
                    help="q4_0 | q8_0 | f16 (default: F32 throughout)")
    ap.add_argument("--check-mel", action="store_true",
                    help="diff the generated filterbank/window against torchaudio")
    args = ap.parse_args()

    if args.check_mel:
        check_mel_against_torchaudio()

    gr = Graph(Path(args.input))
    heads = verify_output_shift(gr)

    tensors: dict[str, np.ndarray] = {}

    # The four ConvStacks. `frame_stack` and `velocity_stack` have no LSTM;
    # `combined_stack` has no ConvStack.
    for stack, gguf_name in (("/onset_stack/", "onset"),
                             ("/offset_stack/", "offset"),
                             ("/frame_stack/", "activation"),
                             ("/velocity_stack/", "velocity"),
                             ("/combined_stack/", "frame")):
        got = collect_stack(gr, stack)
        if not got:
            sys.exit(f"no weights found under node prefix {stack!r}")
        for k, v in got.items():
            tensors[f"oaf.{gguf_name}.{k}"] = v

    # Sanity: shapes the runtime hard-codes.
    assert tensors["oaf.onset.fc.weight"].shape == (FC_OUT, MIDFEAT)
    assert tensors["oaf.onset.head.weight"].shape == (CLASSES, 2 * LSTM_HIDDEN)
    assert tensors["oaf.frame.lstm.fwd.W"].shape == (4 * LSTM_HIDDEN, 3 * CLASSES)
    assert "oaf.frame.fc.weight" not in tensors, "combined stack should have no ConvStack fc"
    assert "oaf.activation.lstm.fwd.W" not in tensors, "frame_stack should have no LSTM"

    tensors["oaf.mel_fb"] = mel_filterbank()
    tensors["oaf.window"] = hann_periodic()

    quant_type = QUANT_TYPES.get(args.quant) if args.quant else None
    as_f16 = args.quant == "f16"

    writer = GGUFWriter(args.output, "onsets-and-frames")
    writer.add_uint32("oaf.sample_rate", SAMPLE_RATE)
    writer.add_uint32("oaf.n_fft", N_FFT)
    writer.add_uint32("oaf.hop_size", HOP)
    writer.add_uint32("oaf.n_mels", N_MELS)
    writer.add_float32("oaf.fmin", F_MIN)
    writer.add_float32("oaf.fmax", F_MAX)
    writer.add_uint32("oaf.classes_num", CLASSES)
    writer.add_uint32("oaf.begin_note", BEGIN_NOTE)
    writer.add_uint32("oaf.lstm_hidden", LSTM_HIDDEN)
    writer.add_uint32("oaf.fc_out", FC_OUT)
    writer.add_uint32("oaf.midfeat", MIDFEAT)
    writer.add_string("oaf.lstm_gate_order", LSTM_GATE_ORDER)
    writer.add_string("oaf.head_names", ",".join(f"{k}={v}" for k, v in heads.items()))

    n_q = n_f16 = n_f32 = 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name]).astype(np.float32)
        raw_dtype = None
        if quant_type is not None and quantisable(name, arr):
            arr = gguf.quantize(arr, quant_type)
            raw_dtype = quant_type
            n_q += 1
        elif as_f16 and not name.endswith(".bias") and name not in ("oaf.mel_fb", "oaf.window"):
            arr = arr.astype(np.float16)
            n_f16 += 1
        else:
            n_f32 += 1
        writer.add_tensor(name, arr, raw_dtype=raw_dtype)
        label = raw_dtype.name if raw_dtype else str(arr.dtype)
        print(f"  {name:38s} {str(tensors[name].shape):22s} {label}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = Path(args.output).stat().st_size
    print(f"\n  {len(tensors)} tensors  (quant: {n_q}, F16: {n_f16}, F32: {n_f32})")
    print(f"  wrote {args.output}  ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
