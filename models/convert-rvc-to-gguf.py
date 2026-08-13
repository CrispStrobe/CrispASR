#!/usr/bin/env python3
"""Convert an RVC voice-conversion generator (SynthesizerTrnMs*NSFsid) to GGUF.

    python models/convert-rvc-to-gguf.py \
        --model f0G40k.pth --config configs/v1/40k.json \
        --output rvc-40k-f32.gguf --dtype f32

Covers the WHOLE inference path `SynthesizerTrnMs768NSFsid.infer()`:
`enc_p` (relative-attention transformer) + `flow` (normalizing flow, reversed)
+ `dec` (NSF vocoder). See docs/music-transcription/RVC_BLUEPRINT.md; the wire
contract is SVC_RECORD_SHAPES.md.

LICENCE. RVC's code is MIT, but CHECKPOINTS are not uniformly so — community
voice models have unclear provenance and some forks add non-commercial terms.
Pass --license to stamp the GGUF; it is deliberately REQUIRED rather than
defaulted, so nobody ships a model whose terms were never considered. CrispASR's
registry gate matches on that tag.

Details that are NOT the obvious default — each one a silent bug if assumed:

  * WEIGHT NORM. `dec` and `flow` store `weight_g`/`weight_v`, not `weight`
    (torch.nn.utils.weight_norm). We FUSE at convert time:
    `w = g * v / ||v||` with the norm over all dims except 0. Some layers in the
    SAME module (conv_pre, noise_convs, conv_post) carry a plain `.weight`, so
    both forms must be handled.
  * `enc_q.*` is the PosteriorEncoder — TRAINING ONLY, never called by infer().
    Skipped, and counted, so "skipped" is a reported number rather than silence.
  * UPSAMPLE RATES ARE NOT RECOVERABLE FROM THE CHECKPOINT. The ConvTranspose1d
    kernels are 16,16,4,4 for 40k while the rates are 10,10,2,2 — assuming
    `kernel == 2*rate` yields 8,8,2,2 and a silently wrong model. They come from
    the config JSON. We then VERIFY them: `noise_convs[i]` must have kernel
    `2*prod(rates[i+1:])` (1 for the last), which pins the schedule exactly.
  * `sr / prod(upsample_rates) == 100` for every shipped config (32k/40k/48k) —
    asserted, because that 100 Hz is the frame rate the wire contract promises.
  * `emb_pitch` is `nn.Embedding(256, hidden)`: the coarse pitch is a TABLE
    INDEX, so an off-by-one is a different learned vector, not a small error.
  * `emb_rel_k/v` shape (1, 2*w+1, d) means RELATIVE positional attention with
    window w — not absolute sinusoidal PE.
  * `harmonic_num` is hardcoded 0 in GeneratorNSF, so SineGen's random initial
    phase is identically zero (`rand_ini[:,0]=0` on a 1-element tensor). The
    only live noise in the source module is the ADDITIVE term.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("error: pip install torch")
try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("error: pip install gguf")

ARCH = "rvc"
SKIP_PREFIXES = ("enc_q.",)  # PosteriorEncoder: training only


def fuse_weight_norm(sd):
    """Return a dict with every weight_g/weight_v pair replaced by a fused weight.

    torch weight_norm (dim=0) stores  w = g * v / ||v||  with ||v|| taken over
    every dim except 0. Fusing here keeps the runtime free of a special case.
    """
    out = {}
    fused = 0
    for k, t in sd.items():
        if k.endswith(".weight_v"):
            base = k[: -len(".weight_v")]
            g = sd.get(base + ".weight_g")
            if g is None:
                sys.exit(f"error: {k} has no matching weight_g — cannot fuse weight_norm")
            v = t.float()
            g = g.float()
            dims = tuple(range(1, v.ndim))
            norm = v.norm(dim=dims, keepdim=True)
            out[base + ".weight"] = (g * v / norm).contiguous()
            fused += 1
        elif k.endswith(".weight_g"):
            continue  # consumed above
        else:
            out[k] = t
    return out, fused


def verify_upsample_schedule(sd, rates):
    """Cross-check config rates against the checkpoint's noise_convs kernels.

    noise_convs[i] is a Conv1d over the source signal with kernel 2*stride and
    stride prod(rates[i+1:]) (kernel 1 for the final stage). This pins the whole
    schedule, so a mismatched config fails here instead of producing a GGUF that
    is silently wrong.
    """
    for i in range(len(rates)):
        key = f"dec.noise_convs.{i}.weight"
        if key not in sd:
            sys.exit(f"error: {key} missing — checkpoint has fewer upsample stages than the config's {len(rates)}")
        k = sd[key].shape[2]
        stride = math.prod(rates[i + 1 :]) if i + 1 < len(rates) else 1
        expect = stride * 2 if stride > 1 else 1
        if k != expect:
            sys.exit(
                f"error: upsample-rate mismatch. {key} kernel={k}, but rates={rates} imply "
                f"2*prod(rates[{i+1}:])={expect}.\n"
                "       The config does not match this checkpoint — using it would produce a "
                "silently wrong model (the rates are NOT recoverable from the weights alone)."
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="RVC generator .pth (e.g. f0G40k.pth)")
    ap.add_argument("--config", required=True, help="matching configs/vN/<sr>.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    ap.add_argument(
        "--license",
        required=True,
        help="SPDX-ish tag for THESE WEIGHTS (e.g. mit, cc-by-nc-sa-4.0, other). "
        "Required on purpose: RVC checkpoints vary and the registry gate matches on this.",
    )
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    m, d = cfg["model"], cfg["data"]
    rates = list(m["upsample_rates"])
    sr = int(d["sampling_rate"])

    upp = math.prod(rates)
    fps = sr / upp
    if abs(fps - 100.0) > 1e-6:
        sys.exit(
            f"error: sr/prod(upsample_rates) = {fps}, expected exactly 100.0.\n"
            "       The wire contract (SVC_RECORD_SHAPES.md) promises a 100 Hz feature/F0 rate, "
            "and this checkpoint would not honour it."
        )

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    if not isinstance(sd, dict):
        sys.exit(f"error: unexpected checkpoint layout: {type(ck)}")

    n_total = len(sd)
    skipped = [k for k in sd if k.startswith(SKIP_PREFIXES)]
    sd = {k: v for k, v in sd.items() if not k.startswith(SKIP_PREFIXES)}

    verify_upsample_schedule(sd, rates)
    sd, n_fused = fuse_weight_norm(sd)

    # Geometry read off the weights, not assumed.
    content_dim = sd["enc_p.emb_phone.weight"].shape[1]  # 256 (v1) or 768 (v2)
    hidden = sd["enc_p.emb_phone.weight"].shape[0]
    n_pitch = sd["enc_p.emb_pitch.weight"].shape[0]
    n_spk, gin = sd["emb_g.weight"].shape
    window = (sd["enc_p.encoder.attn_layers.0.emb_rel_k"].shape[1] - 1) // 2
    if n_pitch != 256:
        sys.exit(f"error: emb_pitch has {n_pitch} rows, expected 256 (coarse pitch 0..255)")

    w = GGUFWriter(str(args.output), ARCH)
    w.add_uint32("rvc.content_dim", int(content_dim))
    w.add_uint32("rvc.hidden_channels", int(hidden))
    w.add_uint32("rvc.n_speakers", int(n_spk))
    w.add_uint32("rvc.gin_channels", int(gin))
    w.add_uint32("rvc.sample_rate", sr)
    w.add_uint32("rvc.rel_attn_window", int(window))
    w.add_array("rvc.upsample_rates", [int(r) for r in rates])
    w.add_array("rvc.upsample_kernel_sizes", [int(k) for k in m["upsample_kernel_sizes"]])
    w.add_array("rvc.resblock_kernel_sizes", [int(k) for k in m["resblock_kernel_sizes"]])
    # Dilations are NOT recoverable from the weights (a dilated conv has the same
    # shape as an undilated one), so they must be carried explicitly. Flattened
    # row-major, n_kernels x n_dilations.
    _dil = m["resblock_dilation_sizes"]
    w.add_array("rvc.resblock_dilations", [int(d) for row in _dil for d in row])
    w.add_uint32("rvc.resblock_n_dilations", int(len(_dil[0])))
    w.add_uint32("rvc.upsample_initial_channel", int(m["upsample_initial_channel"]))
    w.add_string("rvc.resblock", str(m["resblock"]))
    w.add_uint32("rvc.n_layers", int(m["n_layers"]))
    w.add_uint32("rvc.n_heads", int(m["n_heads"]))
    w.add_uint32("rvc.inter_channels", int(m["inter_channels"]))
    # Source module — harmonic_num is hardcoded 0 in GeneratorNSF, which is why
    # SineGen's random initial phase is identically zero. Stored explicitly so
    # the runtime never has to infer it.
    w.add_uint32("rvc.harmonic_num", 0)
    w.add_float32("rvc.sine_amp", 0.1)
    w.add_float32("rvc.add_noise_std", 0.003)
    w.add_float32("rvc.noise_scale", 0.66666)  # z_p = m_p + exp(logs_p)*randn*THIS
    w.add_string("general.license", args.license)
    w.add_string(
        "general.license.description",
        "RVC code is MIT; THESE WEIGHTS carry the tag above. Community RVC "
        "checkpoints vary and some forks are non-commercial — verify before redistributing.",
    )

    emitted = 0
    for k, t in sd.items():
        a = t.detach().cpu().float().numpy()
        # f16 only for the big 2-D+ weights; norms/biases/embeddings stay f32.
        if args.dtype == "f16" and a.ndim >= 2 and not k.endswith(".bias"):
            a = a.astype(np.float16)
        w.add_tensor(k, np.ascontiguousarray(a))
        emitted += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    if emitted + len(skipped) != n_total - n_fused:
        # weight_g is consumed rather than emitted, hence the -n_fused.
        print(
            f"WARNING: tensor accounting does not balance: emitted={emitted} skipped={len(skipped)} "
            f"fused={n_fused} total={n_total}",
            file=sys.stderr,
        )

    print(
        f"rvc: content_dim={content_dim} hidden={hidden} speakers={n_spk} gin={gin} "
        f"sr={sr} rates={rates} upp={upp} ({fps:.0f} fps) rel_window={window}"
    )
    print(
        f"wrote {args.output}: {emitted} tensors emitted, {len(skipped)} skipped "
        f"(enc_q/PosteriorEncoder, training-only), {n_fused} weight_norm pairs fused, dtype {args.dtype}"
    )
    print(f"NOTE: weights licensed '{args.license}' — verify before redistribution.")


if __name__ == "__main__":
    main()
