#!/usr/bin/env python3
"""Onsets & Frames reference dumper — per-stage ONNX ground truth for crispasr-diff.

O&F was the one model of the six note/piano/transcription backends with NO
per-layer parity path at all (docs/ggml-optimisation-playbook.md §5.2). What it
had — tests/oaf_parity_dump.cpp plus tools/oaf_parity.py — compares a mel and
five head activations against native onnxruntime. That is end-of-pipeline: a
regression introduced inside a ConvStack or a BiLSTM shows up as "the onset head
moved", not as a layer, and the script has no PASS/FAIL gate. This dumper plus
the `crispasr-diff onsets-and-frames` arm is the per-layer half.

STANDALONE SCRIPT, not the tools/dump_reference.py `dump()` contract — the same
shape as tools/reference_backends/basic_pitch.py, and for a concrete reason:
the ONNX graph's input is a log-mel, not audio, so a `dump(model_dir, audio,
...)` implementation would have to REIMPLEMENT core_mel in Python. That would
both risk a front-end mismatch of its own and conflate the front end with the
model. Instead this script is handed the mel the C++ runtime actually computed
(oaf-parity-dump writes it to <prefix>.mel.f32), stores it in the reference as
the `mel` stage, and runs ORT on it — so every downstream stage isolates the
model, while the `mel` stage still catches a front-end change as itself. This is
tools/oaf_parity.py:5-11's two-stage design, extended per layer.

    # 1. the C++ runtime's own mel (and the end-of-pipeline heads)
    build/bin/oaf-parity-dump model.gguf audio16k.wav /mnt/storage/oaf-parity/ggml 1

    # 2. the per-stage reference, run on that mel
    MKL_NUM_THREADS=1 $PY tools/reference_backends/onsets_and_frames.py \\
        --onnx   /mnt/storage/tuner-bench/onnx/onsets_and_frames.onnx \\
        --mel    /mnt/storage/oaf-parity/ggml.mel.f32 \\
        --output /mnt/storage/oaf-parity/ref.gguf

    # 3. per-stage cosine, gated at 0.999
    build/bin/crispasr-diff onsets-and-frames model.gguf \\
        /mnt/storage/oaf-parity/ref.gguf audio16k.wav

MKL_NUM_THREADS=1 IS MANDATORY on this box for any parity or timing run of a
mel-front-end model — CrispASR issue #453, "core_mel: threaded MKL silently
multiplies the upper mel bins by the thread count". tools/oaf_parity.py:84-102
already pins it and calls it "a host defect, not a model one". It is pinned in
this script's own environment too, before onnxruntime is imported, so a caller
who forgets is still safe.

THE OUTPUT-NAME SHIFT. The export was handed four `output_names` for a
five-output forward, so every name slid down one slot:

    ONNX output name  ->  what it actually is
    onset                 onset logits
    offset                offset logits
    frame                 ACTIVATION logits (frame_stack, pre-combination)
    velocity              FRAME logits (combined_stack — the real frame head)
    679                   VELOCITY logits

Stage names below use the MEANINGS, matching src/onsets_and_frames.cpp. The map
is asserted structurally by models/convert-onsets-and-frames-to-gguf.py.

LAYOUT. Every stage is written flat, in the element order the C++ side produces:

    mel                       (T, 229)
    <stack>_conv0             (48, T, 229)   after Conv+ReLU
    <stack>_conv1             (48, T, 114)   after Conv+ReLU+MaxPool
    <stack>_conv2             (96, T,  57)   after Conv+ReLU+MaxPool
    <stack>_fc                (T, 768)       after the Linear(5472, 768)
    <x>_bilstm                (T, 768)       BiLSTM output, forward half first
    combined_input            (T, 264)       cat(onset, offset, activation) logits
    <x>_logits                (T, 88)        pre-sigmoid

ONNX gives the conv stages as (1, C, T, F), which flattens to exactly the ggml
(ne0=F, ne1=T, ne2=C) element order — so no transposition is needed, and a
layout bug would show up as a wrecked cosine rather than being papered over.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Before onnxruntime pulls in a threaded BLAS. See the docstring, issue #453.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

N_MELS = 229

# meaning -> the ONNX prefix of its ConvStack
STACK_PREFIX = {
    "onset": "/onset_stack/onset_stack.0",
    "offset": "/offset_stack/offset_stack.0",
    "activation": "/frame_stack/frame_stack.0",
    "velocity": "/velocity_stack/velocity_stack.0",
}

# The three conv stages, as ONNX value-name suffixes under the prefix above.
CONV_SUFFIX = [
    "/cnn/cnn.2/Relu_output_0",      # conv0: Conv + ReLU
    "/cnn/cnn.6/MaxPool_output_0",   # conv1: Conv + ReLU + MaxPool
    "/cnn/cnn.11/MaxPool_output_0",  # conv2: Conv + ReLU + MaxPool
]
FC_SUFFIX = "/fc/fc.0/Add_output_0"

# meaning -> the ONNX value name of that BiLSTM's (1, T, 768) output.
BILSTM = {
    "onset_bilstm": "/onset_stack/onset_stack.1/rnn/Transpose_2_output_0",
    "offset_bilstm": "/offset_stack/offset_stack.1/rnn/Transpose_2_output_0",
    "frame_bilstm": "/combined_stack/combined_stack.0/rnn/Transpose_2_output_0",
}

# meaning -> the ONNX graph output name. Note the shift (see the docstring).
LOGITS = {
    "onset_logits": "onset",
    "offset_logits": "offset",
    "activation_logits": "frame",
    "frame_logits": "velocity",
    "velocity_logits": "679",
}

COMBINED_INPUT = "/Concat_output_0"


def stage_map() -> dict[str, str]:
    """stage name -> ONNX value name, for every stage this dumper exports."""
    m: dict[str, str] = {}
    for meaning, prefix in STACK_PREFIX.items():
        for i, suffix in enumerate(CONV_SUFFIX):
            m[f"{meaning}_conv{i}"] = prefix + suffix
        m[f"{meaning}_fc"] = prefix + FC_SUFFIX
    m.update(BILSTM)
    m["combined_input"] = COMBINED_INPUT
    m.update(LOGITS)
    return m


DEFAULT_STAGES = ["mel"] + list(stage_map().keys())


def expose_intermediates(onnx_path: str, value_names: list[str]) -> str:
    """Write a copy of the graph with `value_names` promoted to graph outputs.

    onnxruntime will only hand back tensors that are declared outputs. The copy
    goes next to the reference so the original export is never mutated — and so
    a stale augmented graph cannot silently outlive an ONNX update.
    """
    import onnx
    from onnx import helper

    model = onnx.load(onnx_path)
    existing = {o.name for o in model.graph.output}
    produced = {o for node in model.graph.node for o in node.output}
    missing = [v for v in value_names if v not in produced and v not in existing]
    if missing:
        raise SystemExit(
            "onsets_and_frames.py: these value names are not produced by the graph — "
            "the export changed and this dumper's node map is stale:\n  " + "\n  ".join(missing)
        )
    for v in value_names:
        if v in existing:
            continue
        # Shape left unspecified on purpose: ORT infers it, and asserting a
        # wrong one here would fail the session rather than the comparison.
        model.graph.output.append(helper.make_empty_tensor_value_info(v))
    out = onnx_path + ".oaf-perlayer.onnx"
    onnx.save(model, out, save_as_external_data=False)
    return out


def write_ref_gguf(path: str, stages: dict[str, np.ndarray]) -> None:
    import gguf

    w = gguf.GGUFWriter(path, "onsets_and_frames_ref")
    for name, arr in sorted(stages.items()):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).reshape(-1))
        w.add_tensor(name, a, raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="/mnt/storage/tuner-bench/onnx/onsets_and_frames.onnx")
    ap.add_argument("--mel", required=True,
                    help="<prefix>.mel.f32 from build/bin/oaf-parity-dump — raw float32, (T, 229)")
    ap.add_argument("--output", required=True, help="reference GGUF to write")
    ap.add_argument("--stages", default="", help="comma-separated subset (default: all)")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    mel = np.fromfile(args.mel, dtype=np.float32)
    if mel.size % N_MELS != 0:
        raise SystemExit(f"onsets_and_frames.py: {args.mel} holds {mel.size} floats, not a multiple of {N_MELS}")
    mel = mel.reshape(-1, N_MELS)
    T = mel.shape[0]
    print(f"mel: T={T} frames x {N_MELS} mels from {args.mel}", file=sys.stderr)

    smap = stage_map()
    want = [s.strip() for s in args.stages.split(",") if s.strip()] or DEFAULT_STAGES
    unknown = [s for s in want if s != "mel" and s not in smap]
    if unknown:
        raise SystemExit("onsets_and_frames.py: unknown stage(s): " + ", ".join(unknown))

    value_names = [smap[s] for s in want if s != "mel"]
    augmented = expose_intermediates(args.onnx, value_names)

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(augmented, so, providers=["CPUExecutionProvider"])
    outs = sess.run(value_names, {"mel": mel[None, :, :].astype(np.float32)})

    stages: dict[str, np.ndarray] = {"mel": mel}
    for stage, arr in zip([s for s in want if s != "mel"], outs):
        stages[stage] = np.asarray(arr, dtype=np.float32)

    for name in sorted(stages):
        print(f"  {name:22s} {tuple(stages[name].shape)!s:>20s}  {stages[name].size} floats", file=sys.stderr)

    write_ref_gguf(args.output, stages)
    total = sum(a.size for a in stages.values())
    print(f"wrote {args.output}  ({len(stages)} stages, {total * 4 / 1e6:.1f} MB)", file=sys.stderr)
    Path(augmented).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
