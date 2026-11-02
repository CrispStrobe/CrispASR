#!/usr/bin/env python3
"""Convert ClearerVoice-Studio MossFormer2_SR_48K to a CrispASR GGUF.

    python models/convert-mossformer2-sr-to-gguf.py \
        --checkpoint-dir MossFormer2_SR_48K \
        --output mossformer2-sr-48k-f16.gguf [--dtype f16|f32]

Source: ModelScope / Hugging Face ``MossFormer2_SR_48K`` checkpoint (rev
39eb1f25, Apache-2.0); network code modelscope/ClearerVoice-Studio@6b3774dc
(``models/mossformer2_sr``).  Two checkpoints, named by ``last_best_checkpoint``:

* ``last_best_checkpoint_m.pt`` (key ``mossformer``): ``Mossformer`` ->
  ``MossFormer_MaskNet(in_channels=80, out_channels=512,
  out_channels_final=80)`` (one speaker), 80 log-mel bins in, 80 channels out.
  Same module layout as MossFormer2_SE_48K, so the names (``mn.*``,
  ``blk.{i}.flash.*``, ``blk.{i}.fsmn.*``, ``rotary.freqs``) come from that
  converter's ``map_name`` and the stack runs through src/mossformer2_blocks.cpp.
* ``last_best_checkpoint_g.pt`` (key ``generator``): a HiFi-GAN V1-style
  ``Generator`` with Snake activations (upsample 8, 8, 2, 2 from 1024
  channels; resblock kernels 3, 7, 11 with dilations 1, 3, 5), stored under
  ``gen.*``.  Weight norm is folded here (``torch._weight_norm``, i.e. what
  upstream's ``remove_weight_norm()`` computes) and every convolution is stored
  as a 2-D matmul matrix the runtime slices per kernel tap:

    gen.<conv>.weight      Conv1d [Cout, Cin, K] -> [K * Cout, Cin] (row k * Cout + o)
    gen.ups.{i}.weight     ConvTranspose1d [Cin, Cout, K] -> [Cout * K, Cin]
                           (row o * K + k; the layout ggml_col2im_1d consumes)

  (NumPy shapes; ggml ne is reversed.)  Snake alphas are stored flat ([C]).

The checkpoints are read with ``torch.load(weights_only=True)``: only tensors
are unpickled and no upstream code runs.

Precision policy: matrices fed to mul_mat (linear layers, k=1 convolutions and
the generator's per-tap / transposed-conv matrices) are F16 in ``--dtype
f16``; every 1-D tensor (norm affines, biases, PReLU slopes, Snake alphas,
scales), the OffsetScale gamma/beta and all depthwise time kernels stay F32.
Use ``crispasr-quantize`` for Q8_0 (see docs/mossformer2-sr.md for the
tensors its arch guard keeps at source precision).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
from pathlib import Path

ARCH = "mossformer2-sr"
PREFIX = "mossformer2_sr"
SOURCE_NAME = "MossFormer2_SR_48K"
SOURCE_REV = "39eb1f25"
UPSTREAM_REPO = "modelscope/ClearerVoice-Studio"
UPSTREAM_REV = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
CHECKPOINT_M_SHA256 = "6cbadb2b6b839e444bb65223c69eea162c8ad08f36e9d0a64144672c4095ab36"
CHECKPOINT_G_SHA256 = "0bdd13c21466f5963d9d1f86a9d84fc6196868318fe22c6b0a750f041805adda"

_SE_CONVERTER = Path(__file__).resolve().parent / "convert-mossformer2-se-to-gguf.py"
_spec = importlib.util.spec_from_file_location("_convert_mossformer2_se", _SE_CONVERTER)
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)

# MossFormer2_SR_48K.yaml/.json + the fixed constructor arguments of Mossformer /
# MossFormer_MaskNet / MossFormerM / Generator.
HPARAMS = {
    "sample_rate": 48000,
    "num_mels": 80,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,
    "mel_fmin": 0,
    "mel_fmax": 8000,
    "num_spks": 1,
    "mossformer_sequence_dim": 512,
    "num_mossformer_layer": 24,
    "group_size": 256,
    "query_key_dim": 128,
    "hidden_dim": 2048,  # int(dim * expansion_factor 4.0)
    "fsmn_inner_channels": 256,
    "fsmn_lorder": 20,
    "conv_module_kernel": 17,
    "rotary_dim": 32,
    "upsample_initial_channel": 1024,
    "gen_conv_kernel": 7,  # conv_pre / conv_post
}
ARRAYS = {
    "upsample_rates": [8, 8, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilations": [1, 3, 5],  # identical for every resblock kernel size
}
FLOATS = {
    "rotary_theta": 10000.0,
    "one_time_decode_length": 20.0,  # seconds (YAML)
    "decode_window": 4.0,  # seconds (YAML)
    "decode_stride_ratio": 0.75,  # decode.py: stride = int(window * 0.75)
    "mel_log_clip": 1e-5,  # dynamic_range_compression_torch clip_val
    "bandwidth_energy_threshold": 0.9996,  # bandwidth_sub.detect_bandwidth
    "bandwidth_crossfade_ms": 100.0,  # bandwidth_sub.smooth_transition transition_band
}

_GEN_CONV = re.compile(r"^(conv_pre|conv_post|resblocks\.(\d+)\.convs([12])\.(\d+))\.(weight_g|weight_v|bias)$")
_GEN_UPS = re.compile(r"^ups\.(\d+)\.(weight_g|weight_v|bias)$")
_GEN_ALPHA = re.compile(r"^(snakes\.(\d+)|snake_post|resblocks\.(\d+)\.convs([12])_activates\.(\d+))\.alpha$")


def gen_base(key: str) -> tuple[str, str] | None:
    """Generator key -> (GGUF base name, kind) with kind conv / ups / alpha."""
    m = _GEN_CONV.match(key)
    if m:
        if m.group(2) is None:
            return f"gen.{m.group(1)}", "conv"
        return f"gen.rb.{int(m.group(2))}.c{m.group(3)}.{int(m.group(4))}", "conv"
    m = _GEN_UPS.match(key)
    if m:
        return f"gen.ups.{int(m.group(1))}", "ups"
    m = _GEN_ALPHA.match(key)
    if m:
        if m.group(2) is not None:
            return f"gen.snakes.{int(m.group(2))}.alpha", "alpha"
        if m.group(3) is not None:
            return f"gen.rb.{int(m.group(3))}.a{m.group(4)}.{int(m.group(5))}", "alpha"
        return "gen.snake_post.alpha", "alpha"
    return None


def fold_weight_norm(v, g):
    """torch.nn.utils.weight_norm(dim=0) with the weight recomputed as remove_weight_norm() does."""
    import torch

    v = torch.as_tensor(v, dtype=torch.float32)
    g = torch.as_tensor(g, dtype=torch.float32)
    return torch._weight_norm(v, g, 0).numpy()


def conv_matrix(w):
    """Conv1d weight [Cout, Cin, K] -> per-tap stacked [K * Cout, Cin]."""
    import numpy as np

    cout, cin, k = w.shape
    return np.ascontiguousarray(np.transpose(w, (2, 0, 1)).reshape(k * cout, cin))


def convt_matrix(w):
    """ConvTranspose1d weight [Cin, Cout, K] -> [Cout * K, Cin] (row o * K + k)."""
    import numpy as np

    cin, cout, k = w.shape
    return np.ascontiguousarray(w.reshape(cin, cout * k).T)


def build_generator(state: dict, dtype: str = "f16") -> dict:
    import numpy as np

    groups: dict = {}
    for key, value in state.items():
        key = key.replace("module.", "")
        arr = value.detach().cpu().float().numpy() if hasattr(value, "detach") else np.asarray(value, np.float32)
        base = gen_base(key)
        if base is None:
            raise KeyError(f"unmapped upstream generator tensor: {key}")
        name, kind = base
        groups.setdefault((name, kind), {})[key.rsplit(".", 1)[1]] = arr
    out = {}
    for (name, kind), parts in groups.items():
        if kind == "alpha":
            out[name] = np.ascontiguousarray(parts["alpha"].reshape(-1), dtype=np.float32)
            continue
        if set(parts) != {"weight_g", "weight_v", "bias"}:
            raise ValueError(f"{name}: expected weight_g, weight_v and bias, got {sorted(parts)}")
        w = fold_weight_norm(parts["weight_v"], parts["weight_g"])
        mat = conv_matrix(w) if kind == "conv" else convt_matrix(w)
        out[name + ".weight"] = mat.astype(np.float16 if dtype == "f16" else np.float32)
        out[name + ".bias"] = np.ascontiguousarray(parts["bias"], dtype=np.float32)
    return out


def build_tensors(state_m: dict, state_g: dict, dtype: str = "f16", hparams: dict | None = None,
                  arrays: dict | None = None) -> dict:
    """Map + validate both state dicts; returns {gguf_name: numpy array}.

    ``hparams`` / ``arrays`` override HPARAMS / ARRAYS for validation (unit
    tests use tiny dims).
    """
    import numpy as np

    out = {}
    rotary = None
    for key, value in state_m.items():
        arr = value.detach().cpu().float().numpy() if hasattr(value, "detach") else np.asarray(value)
        if _se._ROTARY.match(key.replace("module.", "")):
            if rotary is None:
                rotary = arr.astype(np.float32)
            elif not np.array_equal(rotary, arr):
                raise ValueError("per-layer rotary frequencies differ; the shared-buffer assumption is broken")
            continue
        name = _se.map_name(key)
        if name in out:
            raise ValueError(f"duplicate GGUF tensor name {name}")
        a = _se.convert_array(name, arr)
        if dtype == "f16" and _se.is_matmul(name):
            a = a.astype(np.float16)
        out[name] = a
    if rotary is None:
        raise ValueError("mossformer checkpoint has no rotary_pos_emb.freqs buffer")
    out["rotary.freqs"] = rotary
    for name, a in build_generator(state_g, dtype).items():
        if name in out:
            raise ValueError(f"duplicate GGUF tensor name {name}")
        out[name] = a
    validate(out, hparams or HPARAMS, arrays or ARRAYS)
    return out


def validate(t: dict, h: dict, arr: dict) -> None:
    d, c, hid, qk = h["mossformer_sequence_dim"], h["fsmn_inner_channels"], h["hidden_dim"], h["query_key_dim"]
    k, lo, n = h["conv_module_kernel"], 2 * h["fsmn_lorder"] - 1, h["num_mels"]
    if h["num_spks"] != 1:
        raise ValueError("MossFormer2_SR_48K has a single-speaker mask head")
    want = {
        "mn.norm.weight": (n,),
        "mn.conv_enc.weight": (d, n),
        "mn.pos_enc.inv_freq": (d // 2,),
        "mn.conv_out.weight": (d, d),
        "mn.output.weight": (d, d),
        "mn.output_gate.weight": (d, d),
        "mn.conv_dec.weight": (n, d),
        "rotary.freqs": (h["rotary_dim"] // 2,),
    }
    for i in range(h["num_mossformer_layer"]):
        b = f"blk.{i}."
        want.update({
            b + "flash.hidden.linear.weight": (hid, d), b + "flash.hidden.dw.weight": (hid, k),
            b + "flash.qk.linear.weight": (qk, d), b + "flash.qk.dw.weight": (qk, k),
            b + "flash.out.linear.weight": (d, hid // 2), b + "flash.out.dw.weight": (d, k),
            b + "flash.qk_os.gamma": (4, qk), b + "flash.qk_os.beta": (4, qk),
            b + "fsmn.conv1.weight": (c, d), b + "fsmn.conv2.weight": (d, c),
            b + "fsmn.to_u.dw.weight": (c, k), b + "fsmn.to_v.dw.weight": (c, k),
            b + "fsmn.fsmn.project.weight": (c, c), b + "fsmn.mem.weight": (c, lo),
        })
    ch, gk = h["upsample_initial_channel"], h["gen_conv_kernel"]
    want.update({"gen.conv_pre.weight": (gk * ch, n), "gen.conv_pre.bias": (ch,)})
    n_rb = len(arr["resblock_kernel_sizes"])
    for i, (u, uk) in enumerate(zip(arr["upsample_rates"], arr["upsample_kernel_sizes"])):
        cin, cout = ch >> i, ch >> (i + 1)
        if (uk - u) % 2:
            raise ValueError("upsample kernel - rate must be even (symmetric padding)")
        want.update({f"gen.snakes.{i}.alpha": (cin,), f"gen.ups.{i}.weight": (cout * uk, cin),
                     f"gen.ups.{i}.bias": (cout,)})
        for j, rk in enumerate(arr["resblock_kernel_sizes"]):
            r = f"gen.rb.{i * n_rb + j}."
            for di in range(len(arr["resblock_dilations"])):
                for s in ("1", "2"):
                    want.update({f"{r}c{s}.{di}.weight": (rk * cout, cout), f"{r}c{s}.{di}.bias": (cout,),
                                 f"{r}a{s}.{di}": (cout,)})
    last = ch >> len(arr["upsample_rates"])
    want.update({"gen.snake_post.alpha": (last,), "gen.conv_post.weight": (gk, last), "gen.conv_post.bias": (1,)})
    for name, shape in want.items():
        if name not in t:
            raise ValueError(f"missing tensor {name}")
        if tuple(t[name].shape) != shape:
            raise ValueError(f"{name}: shape {tuple(t[name].shape)} != expected {shape}")
    n_mn = 17 + 1 + h["num_mossformer_layer"] * (14 + 23)
    n_gen = 2 + len(arr["upsample_rates"]) * (3 + n_rb * len(arr["resblock_dilations"]) * 6) + 3
    if len(t) != n_mn + n_gen:
        raise ValueError(f"{len(t)} tensors, expected {n_mn + n_gen}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_gguf(tensors: dict, output: Path, dtype: str, source_sha: tuple[str, str] = ("", ""),
               hparams: dict | None = None, arrays: dict | None = None) -> None:
    from gguf import GGMLQuantizationType, GGUFWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(output), ARCH)
    w.add_name(SOURCE_NAME)
    w.add_description("ClearerVoice-Studio MossFormer2 speech super-resolution (bandwidth extension) to 48 kHz")
    w.add_string("general.license", "apache-2.0")
    w.add_string("general.source.revision", SOURCE_REV)
    w.add_string("general.source.code_repository", UPSTREAM_REPO)
    w.add_string("general.source.code_revision", UPSTREAM_REV)
    if source_sha[0]:
        w.add_string("general.source.checkpoint_m_sha256", source_sha[0])
    if source_sha[1]:
        w.add_string("general.source.checkpoint_g_sha256", source_sha[1])
    w.add_file_type(GGMLQuantizationType.F16 if dtype == "f16" else GGMLQuantizationType.F32)
    for key, value in (hparams or HPARAMS).items():
        w.add_uint32(f"{PREFIX}.{key}", value)
    for key, value in (arrays or ARRAYS).items():
        w.add_array(f"{PREFIX}.{key}", [int(v) for v in value])
    for key, value in FLOATS.items():
        w.add_float32(f"{PREFIX}.{key}", value)
    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def checkpoint_paths(ckpt_dir: Path) -> tuple[Path, Path]:
    """``last_best_checkpoint`` names the mossformer file first, the generator second."""
    best = ckpt_dir / "last_best_checkpoint"
    if best.is_file():
        names = [s.strip() for s in best.read_text().splitlines() if s.strip()]
        if len(names) >= 2:
            return ckpt_dir / names[0], ckpt_dir / names[1]
    return ckpt_dir / "last_best_checkpoint_m.pt", ckpt_dir / "last_best_checkpoint_g.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, required=True,
                    help="directory with last_best_checkpoint, last_best_checkpoint_m.pt and _g.pt")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    ap.add_argument("--allow-checkpoint-mismatch", action="store_true",
                    help="convert checkpoints whose sha256 differs from the pinned release")
    args = ap.parse_args()

    import torch

    ckpt_m, ckpt_g = checkpoint_paths(args.checkpoint_dir)
    digests = (sha256(ckpt_m), sha256(ckpt_g))
    if digests != (CHECKPOINT_M_SHA256, CHECKPOINT_G_SHA256):
        msg = (f"checkpoint sha256 {digests[0]} / {digests[1]} != pinned {CHECKPOINT_M_SHA256} / "
               f"{CHECKPOINT_G_SHA256} ({SOURCE_NAME}@{SOURCE_REV})")
        if not args.allow_checkpoint_mismatch:
            raise SystemExit(msg + "; pass --allow-checkpoint-mismatch to convert anyway")
        print("warning:", msg)
    cm = torch.load(str(ckpt_m), map_location="cpu", weights_only=True)
    cg = torch.load(str(ckpt_g), map_location="cpu", weights_only=True)
    state_m = cm["mossformer"] if isinstance(cm, dict) and "mossformer" in cm else cm
    state_g = cg["generator"] if isinstance(cg, dict) and "generator" in cg else cg
    tensors = build_tensors(state_m, state_g, args.dtype)
    write_gguf(tensors, args.output, args.dtype, digests)
    n_params = sum(a.size for a in tensors.values())
    print(f"wrote {args.output}: {len(tensors)} tensors, {n_params / 1e6:.2f} M params, {args.dtype}")


if __name__ == "__main__":
    main()
