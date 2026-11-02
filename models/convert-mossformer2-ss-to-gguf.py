#!/usr/bin/env python3
"""Convert ClearerVoice-Studio MossFormer2_SS_16K to a CrispASR GGUF.

    python models/convert-mossformer2-ss-to-gguf.py \
        --checkpoint MossFormer2_SS_16K/last_best_checkpoint.pt \
        --output mossformer2-ss-16k-f16.gguf [--dtype f16|f32]

Source: Hugging Face ``alibabasglab/MossFormer2_SS_16K`` (rev 407cb030,
Apache-2.0); network code modelscope/ClearerVoice-Studio@6b3774dc.

The checkpoint is read with ``torch.load(weights_only=True)``: only tensors are
unpickled and no upstream code runs.  Only the ``model`` state dict is
converted (optimizer state is dropped).  Names are shortened to fit
GGML_MAX_NAME (the upstream ``mask_net.mdl.intra_mdl.mossformerM...`` paths
exceed 64 bytes); see ``map_name``.  Layout changes are limited to dropping
singleton kernel axes (k=1 convolutions become plain matrices, depthwise
``[C, 1, K]`` kernels become ``[C, K]``).

Precision policy: matrices fed to mul_mat (linear layers, k=1 convolutions,
encoder/decoder filter banks) are F16 in ``--dtype f16``; every 1-D tensor
(norm affines, biases, PReLU slopes, scales), the OffsetScale gamma/beta and
all depthwise time kernels (consumed by ggml_ssm_conv, which needs F32) stay
F32.  Use ``crispasr-quantize`` for Q8_0: its 2-D/"weight"/block-size rules
quantize only the matmul matrices of this model.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

ARCH = "mossformer2-ss"
PREFIX = "mossformer2_ss"
HF_REPO = "alibabasglab/MossFormer2_SS_16K"
HF_REV = "407cb030"
UPSTREAM_REPO = "modelscope/ClearerVoice-Studio"
UPSTREAM_REV = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
CHECKPOINT_SHA256 = "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a"

# MossFormer2_SS_16K.yaml + the fixed constructor arguments of MossFormer /
# MossFormerM / FLASH_ShareA_FFConvM / Gated_FSMN_Block_Dilated.
HPARAMS = {
    "sample_rate": 16000,
    "num_spks": 2,
    "encoder_kernel_size": 16,
    "encoder_embedding_dim": 512,
    "mossformer_sequence_dim": 512,
    "num_mossformer_layer": 24,
    "group_size": 256,
    "query_key_dim": 128,
    "hidden_dim": 2048,  # int(dim * expansion_factor 4.0)
    "fsmn_inner_channels": 256,
    "fsmn_lorder": 20,
    "conv_module_kernel": 17,
    "rotary_dim": 32,
}
FLOATS = {
    "rotary_theta": 10000.0,
    "one_time_decode_length": 2.0,  # seconds (YAML)
    "decode_window": 2.0,  # seconds (YAML)
    "decode_stride_ratio": 0.75,  # decode.py: stride = int(window * 0.75)
}

MN = "mask_net."
BLK = MN + "mdl.intra_mdl.mossformerM."

# FFConvM (norm, linear, SiLU, ConvModule) sub-paths.
_FFCONV = {
    "mdl.0.g": "norm.g",  # ScaleNorm
    "mdl.0.weight": "norm.weight",  # LayerNorm
    "mdl.0.bias": "norm.bias",
    "mdl.1.weight": "linear.weight",
    "mdl.1.bias": "linear.bias",
    "mdl.3.sequential.1.conv.weight": "dw.weight",
}
_FLASH = {f"{src}.{k}": f"{dst}.{v}" for src, dst in (("to_hidden", "hidden"), ("to_qk", "qk"), ("to_out", "out"))
          for k, v in _FFCONV.items()}
_FLASH.update({"qk_offset_scale.gamma": "qk_os.gamma", "qk_offset_scale.beta": "qk_os.beta"})
_FSMN = {
    "conv1.0.weight": "conv1.weight",
    "conv1.0.bias": "conv1.bias",
    "conv1.1.weight": "conv1.prelu",
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "gated_fsmn.fsmn.linear.weight": "fsmn.linear.weight",
    "gated_fsmn.fsmn.linear.bias": "fsmn.linear.bias",
    "gated_fsmn.fsmn.project.weight": "fsmn.project.weight",
    "gated_fsmn.fsmn.conv.conv1.weight": "ddn.conv1.weight",
    "gated_fsmn.fsmn.conv.norm1.weight": "ddn.norm1.weight",
    "gated_fsmn.fsmn.conv.norm1.bias": "ddn.norm1.bias",
    "gated_fsmn.fsmn.conv.prelu1.weight": "ddn.prelu1",
    "gated_fsmn.fsmn.conv.conv2.weight": "ddn.conv2.weight",
    "gated_fsmn.fsmn.conv.norm2.weight": "ddn.norm2.weight",
    "gated_fsmn.fsmn.conv.norm2.bias": "ddn.norm2.bias",
    "gated_fsmn.fsmn.conv.prelu2.weight": "ddn.prelu2",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
    "conv2.weight": "conv2.weight",
    "conv2.bias": "conv2.bias",
}
_FSMN.update({f"gated_fsmn.{src}.{k}": f"{dst}.{v}" for src, dst in (("to_u", "to_u"), ("to_v", "to_v"))
              for k, v in _FFCONV.items()})
_TOP = {
    "enc.conv1d.weight": "enc.weight",
    "dec.weight": "dec.weight",
    MN + "norm.weight": "mn.norm.weight",
    MN + "norm.bias": "mn.norm.bias",
    MN + "conv1d_encoder.weight": "mn.conv_enc.weight",
    MN + "pos_enc.scale": "mn.pos_enc.scale",
    MN + "pos_enc.inv_freq": "mn.pos_enc.inv_freq",
    MN + "mdl.intra_mdl.norm.weight": "mn.intra_ln_norm.weight",
    MN + "mdl.intra_mdl.norm.bias": "mn.intra_ln_norm.bias",
    MN + "mdl.intra_norm.weight": "mn.intra_norm.weight",
    MN + "mdl.intra_norm.bias": "mn.intra_norm.bias",
    MN + "prelu.weight": "mn.prelu",
    MN + "conv1d_out.weight": "mn.conv_out.weight",
    MN + "conv1d_out.bias": "mn.conv_out.bias",
    MN + "output.0.weight": "mn.output.weight",
    MN + "output.0.bias": "mn.output.bias",
    MN + "output_gate.0.weight": "mn.output_gate.weight",
    MN + "output_gate.0.bias": "mn.output_gate.bias",
    MN + "conv1_decoder.weight": "mn.conv_dec.weight",
}
_ROTARY = re.compile(re.escape(BLK) + r"layers\.(\d+)\.rotary_pos_emb\.freqs$")

# Tensors consumed by mul_mat (F16 in the f16 output, quantizable).
_MATMUL_SUFFIXES = (".linear.weight", ".project.weight", "fsmn.conv1.weight", "fsmn.conv2.weight")
_MATMUL_NAMES = {"enc.weight", "dec.weight", "mn.conv_enc.weight", "mn.conv_out.weight", "mn.output.weight",
                 "mn.output_gate.weight", "mn.conv_dec.weight"}


def map_name(name: str) -> str | None:
    """Upstream state-dict key -> GGUF tensor name (None: dropped)."""
    name = name.replace("module.", "")
    if name in _TOP:
        return _TOP[name]
    if _ROTARY.match(name):
        return None  # identical non-trainable buffer in every layer; stored once as "rotary.freqs"
    for kind, table, dst in (("layers", _FLASH, "flash"), ("fsmn", _FSMN, "fsmn")):
        m = re.match(re.escape(BLK) + kind + r"\.(\d+)\.(.+)$", name)
        if m:
            sub = table.get(m.group(2))
            if sub is None:
                raise KeyError(f"unmapped upstream tensor: {name}")
            return f"blk.{int(m.group(1))}.{dst}.{sub}"
    raise KeyError(f"unmapped upstream tensor: {name}")


def is_matmul(gguf_name: str) -> bool:
    if gguf_name in _MATMUL_NAMES:
        return True
    if ".ddn." in gguf_name:
        return False
    return gguf_name.endswith(_MATMUL_SUFFIXES)


def convert_array(gguf_name: str, array):
    """Drop singleton kernel axes; keep the upstream axis order otherwise."""
    import numpy as np

    a = np.asarray(array, dtype=np.float32)
    if gguf_name.endswith((".dw.weight", "ddn.conv1.weight")):
        a = a.reshape(a.shape[0], -1)  # [C,1,K] / [C,1,K,1] -> [C, K]
    elif gguf_name.endswith("ddn.conv2.weight"):
        a = a.reshape(a.shape[0], a.shape[1], a.shape[2])  # [C,2,K,1] -> [C, 2, K]
    elif gguf_name in ("enc.weight", "dec.weight"):
        a = a.reshape(a.shape[0], a.shape[-1])  # [512,1,16] -> [512, 16]
    elif a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]  # k=1 Conv1d -> [out, in]
    return np.ascontiguousarray(a)


def build_tensors(state: dict, dtype: str = "f16", hparams: dict | None = None) -> dict:
    """Map + validate a state dict; returns {gguf_name: numpy array}.

    ``hparams`` overrides HPARAMS for validation (unit tests use tiny dims).
    """
    import numpy as np

    out = {}
    rotary = None
    for key, value in state.items():
        arr = value.detach().cpu().float().numpy() if hasattr(value, "detach") else np.asarray(value)
        if _ROTARY.match(key.replace("module.", "")):
            if rotary is None:
                rotary = arr.astype(np.float32)
            elif not np.array_equal(rotary, arr):
                raise ValueError("per-layer rotary frequencies differ; the shared-buffer assumption is broken")
            continue
        name = map_name(key)
        if name in out:
            raise ValueError(f"duplicate GGUF tensor name {name}")
        a = convert_array(name, arr)
        if dtype == "f16" and is_matmul(name):
            a = a.astype(np.float16)
        out[name] = a
    if rotary is None:
        raise ValueError("checkpoint has no rotary_pos_emb.freqs buffer")
    out["rotary.freqs"] = rotary
    validate(out, hparams or HPARAMS)
    return out


def validate(t: dict, h: dict) -> None:
    d, c, hid, qk = h["mossformer_sequence_dim"], h["fsmn_inner_channels"], h["hidden_dim"], h["query_key_dim"]
    k, lo = h["conv_module_kernel"], 2 * h["fsmn_lorder"] - 1
    want = {
        "enc.weight": (h["encoder_embedding_dim"], h["encoder_kernel_size"]),
        "dec.weight": (h["encoder_embedding_dim"], h["encoder_kernel_size"]),
        "mn.conv_enc.weight": (d, d),
        "mn.pos_enc.inv_freq": (d // 2,),
        "mn.conv_out.weight": (d * h["num_spks"], d),
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
            b + "fsmn.ddn.conv1.weight": (c, lo), b + "fsmn.ddn.conv2.weight": (c, 2, lo),
        })
    for name, shape in want.items():
        if name not in t:
            raise ValueError(f"missing tensor {name}")
        if tuple(t[name].shape) != shape:
            raise ValueError(f"{name}: shape {tuple(t[name].shape)} != expected {shape}")
    n_expected = 19 + 1 + h["num_mossformer_layer"] * (14 + 30)
    if len(t) != n_expected:
        raise ValueError(f"{len(t)} tensors, expected {n_expected}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_gguf(tensors: dict, output: Path, dtype: str, source_sha: str = "", hparams: dict | None = None) -> None:
    from gguf import GGMLQuantizationType, GGUFWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(output), ARCH)
    w.add_name("MossFormer2_SS_16K")
    w.add_description("ClearerVoice-Studio MossFormer2 two-speaker speech separation, 16 kHz")
    w.add_string("general.license", "apache-2.0")
    w.add_string("general.source.huggingface.repository", HF_REPO)
    w.add_string("general.source.revision", HF_REV)
    w.add_string("general.source.code_repository", UPSTREAM_REPO)
    w.add_string("general.source.code_revision", UPSTREAM_REV)
    if source_sha:
        w.add_string("general.source.checkpoint_sha256", source_sha)
    w.add_file_type(GGMLQuantizationType.F16 if dtype == "f16" else GGMLQuantizationType.F32)
    for key, value in (hparams or HPARAMS).items():
        w.add_uint32(f"{PREFIX}.{key}", value)
    for key, value in FLOATS.items():
        w.add_float32(f"{PREFIX}.{key}", value)
    w.add_array(f"{PREFIX}.source_names", ["spk1", "spk2"])
    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="last_best_checkpoint.pt")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    ap.add_argument("--allow-checkpoint-mismatch", action="store_true",
                    help="convert a checkpoint whose sha256 differs from the pinned release")
    args = ap.parse_args()

    import torch

    digest = sha256(args.checkpoint)
    if digest != CHECKPOINT_SHA256:
        msg = f"checkpoint sha256 {digest} != pinned {CHECKPOINT_SHA256} ({HF_REPO}@{HF_REV})"
        if not args.allow_checkpoint_mismatch:
            raise SystemExit(msg + "; pass --allow-checkpoint-mismatch to convert anyway")
        print("warning:", msg)
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    tensors = build_tensors(state, args.dtype)
    write_gguf(tensors, args.output, args.dtype, digest)
    n_params = sum(a.size for a in tensors.values())
    print(f"wrote {args.output}: {len(tensors)} tensors, {n_params / 1e6:.2f} M params, {args.dtype}")


if __name__ == "__main__":
    main()
