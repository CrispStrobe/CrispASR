#!/usr/bin/env python3
"""Convert ClearerVoice-Studio MossFormerGAN_SE_16K to a CrispASR GGUF.

Upstream: modelscope/ClearerVoice-Studio (Apache-2.0), weights
alibabasglab/MossFormerGAN_SE_16K ``last_best_checkpoint.pt`` (the generator;
the ``.disc.pt`` discriminator is training-only and not needed).

    python models/convert-mossformergan-se-to-gguf.py \
        --checkpoint MossFormerGAN_SE_16K/last_best_checkpoint.pt \
        --output mossformergan-se-16k-f16.gguf --outtype f16

The checkpoint is read with ``torch.load(weights_only=True)`` (no pickle code
execution). Tensor names keep the upstream SyncANet state-dict names; only the
memory layout of some tensors is changed so the native graph can use them
directly (``mossformergan_se.layout = 2``):

* ``Conv2d`` weights ``[Cout, Cin, KH, KW]`` -> per-tap matrices
  ``[P*KH*KW, Cout, Cpart]`` (ggml ne ``[Cpart, Cout, taps]``). The
  dilated-dense convs are split into P input parts of 64 channels so the
  runtime never materialises the dense concatenation.
* ``ConvTranspose1d`` ``[Cin, Cout, K]`` -> ``[K, Cout, Cin]``.
* depthwise ``Conv1d``/``Conv2d`` ``[D, 1, K(, 1)]`` -> ``[K, D]`` (channels fastest, as
  ggml_conv_2d_dw_direct's CWHN path reads it; always F32).
* ``Fconv`` (grouped 1x2 conv) ``[2C, 1, 1, 2]`` -> ``[2, 2C]`` (F32).
* ``LayerNormalization4DCF`` gamma/beta ``[1, E, 1, F]`` -> ``[F, E]``.
* ``LayerNormalization4D`` gamma/beta ``[1, C, 1, 1]`` -> ``[C]``.
* attention 1x1 convs ``[E, C, 1, 1]`` -> ``[E, C]``.

``--outtype`` f32 | f16 | q8_0 applies to matmul weights only; norms, PReLU
slopes, biases, depthwise kernels and the 1x1 3-channel input conv stay F32.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np

ARCH = "mossformergan-se"
PREFIX = "mossformergan_se"
HF_REPO = "alibabasglab/MossFormerGAN_SE_16K"
HF_REVISION = "dcb1b20e"
UPSTREAM = "modelscope/ClearerVoice-Studio@6b3774dc"

# clearvoice/config/inference/MossFormerGAN_SE_16K.yaml + generator.py constants
HPARAMS_U32 = {
    "sample_rate": 16000,
    "n_fft": 400,
    "hop_length": 100,
    "win_length": 400,
    "num_channel": 64,
    "num_features": 201,
    "n_layers": 6,
    "n_head": 4,
    "attn_qk_dim": 6,  # ceil(512 / n_freqs(101))
    "mossformer_qk_dim": 128,
    "mossformer_hidden": 256,
    "rotary_dim": 32,
    "decode_window_seconds": 10,
    "one_time_decode_seconds": 10,
    "layout": 2,
}
HPARAMS_F32 = {
    "compress_factor": 0.3,
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


DENSE_CONV = re.compile(r"\.(dilated_dense|dense_block)\.conv(\d)\.weight$")


def transform(name: str, t: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (array in runtime layout, is_matmul_weight)."""
    if name.endswith("rotary_pos_emb.freqs"):
        return t, False
    m = DENSE_CONV.search(name)
    if m:  # [64, 64*P, 2, 3] -> [P*2*3, 64, 64]; part p covers input channels p*64..p*64+63
        cout, cin, kh, kw = t.shape
        parts = cin // 64
        a = t.reshape(cout, parts, 64, kh, kw).transpose(1, 3, 4, 0, 2)  # [P, KH, KW, Cout, 64]
        return a.reshape(parts * kh * kw, cout, 64), True
    if re.search(r"(conv_1\.0|conv_2\.0|sub_pixel\.conv|mask_decoder\.conv_1|complex_decoder\.conv)\.weight$", name):
        cout, cin, kh, kw = t.shape
        a = t.transpose(2, 3, 0, 1).reshape(kh * kw, cout, cin)
        return a, cin % 32 == 0
    if name.endswith("Fconv.weight"):  # [2C, 1, 1, 2] -> [2, 2C]
        return t.reshape(t.shape[0], t.shape[3]).T.copy(), False
    if re.search(r"(intra|inter)_linear\.weight$", name):  # ConvTranspose1d [Cin, Cout, K] -> [K, Cout, Cin]
        return t.transpose(2, 1, 0).copy(), True
    if re.search(r"(\.conv1\.weight|sequential\.1\.conv\.weight)$", name):  # depthwise -> [K, D]
        return t.reshape(t.shape[0], -1).T.copy(), False
    if re.search(r"attn_(conv_[QKV]_\d|concat_proj)\.0\.weight$", name):
        return t.reshape(t.shape[0], t.shape[1]), True
    if re.search(r"attn_(conv_[QKV]_\d|concat_proj)\.2\.(gamma|beta)$", name):  # [1, E, 1, F] -> [F, E]
        return t.reshape(t.shape[1], t.shape[3]).T.copy(), False
    if re.search(r"_norm\.(gamma|beta)$", name):  # LayerNormalization4D
        return t.reshape(-1), False
    if t.ndim == 2 and not name.endswith("qk_offset_scale.gamma") and not name.endswith("qk_offset_scale.beta"):
        return t, True  # nn.Linear
    return t, False


def convert(checkpoint: Path, output: Path, outtype: str, verbose: bool = False) -> None:
    import torch
    from gguf import GGMLQuantizationType, GGUFWriter
    from gguf.quants import quantize

    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    if "dense_encoder.conv_1.0.weight" not in sd or "blocks.5.attn_concat_proj.0.weight" not in sd:
        raise SystemExit(f"{checkpoint}: not a MossFormerGAN_SE_16K generator checkpoint")
    if any(k.startswith("discriminator") for k in sd):
        sd = {k: v for k, v in sd.items() if not k.startswith("discriminator")}

    freqs = sd["blocks.0.intra_mossformer.rotary_pos_emb.freqs"].numpy()
    expect = 1.0 / (10000.0 ** (np.arange(0, 32, 2, dtype=np.float32) / 32.0))
    if not np.allclose(freqs, expect, rtol=1e-6):
        raise SystemExit("unexpected rotary frequencies; the runtime assumes theta=10000, dim=32")

    writer = GGUFWriter(str(output), ARCH)
    writer.add_name("MossFormerGAN_SE_16K")
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("general.source.huggingface.repository", HF_REPO)
    writer.add_string("general.source.revision", HF_REVISION)
    writer.add_string("general.source.upstream", UPSTREAM)
    if checkpoint.is_file():
        writer.add_string("general.source.checkpoint_sha256", sha256(checkpoint))
    for k, v in HPARAMS_U32.items():
        writer.add_uint32(f"{PREFIX}.{k}", v)
    for k, v in HPARAMS_F32.items():
        writer.add_float32(f"{PREFIX}.{k}", v)
    writer.add_string(f"{PREFIX}.window", "hamming_periodic")

    counts: dict[str, int] = {}
    for name, tensor in sd.items():
        t = tensor.detach().float().numpy()
        a, is_mm = transform(name, t)
        a = np.ascontiguousarray(a, dtype=np.float32)
        qtype = GGMLQuantizationType.F32
        if is_mm and outtype == "f16":
            qtype = GGMLQuantizationType.F16
        elif is_mm and outtype == "q8_0" and a.shape[-1] % 32 == 0:
            qtype = GGMLQuantizationType.Q8_0
        if qtype == GGMLQuantizationType.F16:
            writer.add_tensor(name, a.astype(np.float16))
        elif qtype == GGMLQuantizationType.Q8_0:
            writer.add_tensor(name, quantize(a, qtype), raw_dtype=qtype)
        else:
            writer.add_tensor(name, a)
        counts[qtype.name] = counts.get(qtype.name, 0) + 1
        if verbose:
            print(f"{name:70s} {str(tuple(t.shape)):20s} -> {str(a.shape):20s} {qtype.name}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB, {len(sd)} tensors: {counts})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="last_best_checkpoint.pt (or a directory containing it)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--outtype", choices=("f32", "f16", "q8_0"), default="f16")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    ckpt = a.checkpoint
    if ckpt.is_dir():
        ckpt = ckpt / "last_best_checkpoint.pt"
    convert(ckpt, a.output, a.outtype, a.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
