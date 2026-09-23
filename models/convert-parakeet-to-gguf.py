#!/usr/bin/env python3
"""
Convert nvidia/parakeet-tdt-0.6b-v3 (a NeMo .nemo checkpoint) → GGUF F16.

Architecture (from model_config.yaml + tensor inspection):

  preprocessor (mel):     128 mel bins, n_fft=512, win=25ms, stride=10ms (Hann)
  encoder (24× FastConformer):
    pre_encode:           3-stage dw_striding Conv2d (8× time downsample, 128→16 freq)
                          out: linear(4096 → 1024)
    layer i:              FFN1(½) → MHA(rel_pos, untied bias) → conv(dw, k=9, BN) → FFN2(½) → LN
    d_model = 1024  n_heads = 8  ff = 4096  v = 8192
  decoder.prediction:     embed(8193, 640) + 2-layer LSTM(640, 640)
  joint:                  enc(1024→640) + pred(640→640) → tanh → linear(640 → 8198)
                          8198 = 8192 vocab + 1 blank + 5 TDT durations {0,1,2,3,4}

GGUF tensor naming (mirrors what the C++ loader will expect):

  preprocessor.fb                                    F32  (128, 257)
  preprocessor.window                                F32  (400,)

  encoder.pre.conv.{0,2,3,5,6}.{weight,bias}         F16/F32
  encoder.pre.out.{weight,bias}                      F16/F32

  encoder.layers.{i}.norm_ff1.{weight,bias}          F32
  encoder.layers.{i}.ff1.linear1.weight              F16
  encoder.layers.{i}.ff1.linear2.weight              F16
  encoder.layers.{i}.ff1.linear1.bias                F32   (zero in this model — bias-less FF)
  ... (analogous for ff2)

  encoder.layers.{i}.norm_attn.{weight,bias}         F32
  encoder.layers.{i}.attn.{q,k,v,out,pos}.weight     F16
  encoder.layers.{i}.attn.pos_bias_u                 F32  (8, 128)
  encoder.layers.{i}.attn.pos_bias_v                 F32  (8, 128)

  encoder.layers.{i}.norm_conv.{weight,bias}         F32
  encoder.layers.{i}.conv.pw1.weight                 F16  (2048, 1024, 1)
  encoder.layers.{i}.conv.dw.weight                  F16  (1024, 1, 9)
  encoder.layers.{i}.conv.bn.{weight,bias,running_mean,running_var}  F32
  encoder.layers.{i}.conv.pw2.weight                 F16  (1024, 1024, 1)

  encoder.layers.{i}.norm_ff2.{weight,bias}          F32
  encoder.layers.{i}.norm_out.{weight,bias}          F32

  decoder.embed.weight                               F16  (8193, 640)
  decoder.lstm.{0,1}.{w_ih,w_hh,b_ih,b_hh}           F16/F32

  joint.enc.{weight,bias}                            F16/F32
  joint.pred.{weight,bias}                           F16/F32
  joint.out.{weight,bias}                            F16/F32

GGUF metadata keys (under `parakeet.*`):
  parakeet.sample_rate          = 16000
  parakeet.n_mels               = 128
  parakeet.n_fft                = 512
  parakeet.win_length           = 400
  parakeet.hop_length           = 160
  parakeet.frame_dur_ms         = 80   (10 ms × 8× subsampling)
  parakeet.d_model              = 1024
  parakeet.n_layers             = 24
  parakeet.n_heads              = 8
  parakeet.head_dim             = 128
  parakeet.ff_dim               = 4096
  parakeet.subsampling_factor   = 8
  parakeet.subsampling_channels = 256
  parakeet.conv_kernel          = 9
  parakeet.pred_hidden          = 640
  parakeet.pred_layers          = 2
  parakeet.joint_hidden         = 640
  parakeet.vocab_size           = 8192
  parakeet.blank_id             = 8192
  parakeet.n_tdt_durations      = 5
  parakeet.tdt_durations        = [0, 1, 2, 3, 4]

  tokenizer.ggml.tokens         = [<8192 strings from SentencePiece>]
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")
try:
    import torch
except ImportError:
    sys.exit("pip install torch")
try:
    import sentencepiece as spm
except ImportError:
    sys.exit("pip install sentencepiece")


# ---------------------------------------------------------------------------
# .nemo loading — disk-extract path and in-memory path
# ---------------------------------------------------------------------------


def load_nemo_inmem(nemo_path: Path) -> dict:
    """Load .nemo tarball entirely in RAM — avoids writing extracted ckpt to disk.

    Returns dict with keys: 'weights' (state_dict), 'config' (yaml str),
    'spm' (bytes), 'vocab' (bytes or None).
    """
    import io as _io
    result = {}
    with tarfile.open(nemo_path, "r") as tf:
        for m in tf.getmembers():
            n = m.name
            fobj = tf.extractfile(m)
            if fobj is None:
                continue
            data = fobj.read()
            if n.endswith("model_weights.ckpt"):
                buf = _io.BytesIO(data)
                del data
                sd = torch.load(buf, map_location="cpu", weights_only=True)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                result["weights"] = sd
            elif n.endswith("model_config.yaml"):
                result["config_str"] = data.decode()
            elif n.endswith("_tokenizer.model"):
                result["spm_bytes"] = data
            elif n.endswith("_vocab.txt"):
                result["vocab_bytes"] = data
    if "weights" not in result or "spm_bytes" not in result:
        sys.exit(f"could not find weights / tokenizer in {nemo_path}")
    return result


def unpack_nemo(nemo_path: Path, out_dir: Path) -> dict:
    """Extract .nemo tarball, return paths to weights / config / tokenizer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nemo_path, "r") as tf:
        tf.extractall(out_dir)
    paths = {}
    for f in out_dir.iterdir():
        n = f.name
        if n.endswith("model_weights.ckpt"):
            paths["weights"] = f
        elif n.endswith("model_config.yaml"):
            paths["config"] = f
        elif n.endswith("_tokenizer.model"):
            paths["spm"] = f
        elif n.endswith("_vocab.txt"):
            paths["vocab"] = f
    if "weights" not in paths or "spm" not in paths:
        sys.exit(f"could not find weights / tokenizer in {nemo_path}")
    return paths


def load_nemo_disk(nemo_path: Path, extract_dir: Path) -> dict:
    """Extract .nemo to disk and load weights via mmap — low memory for large models.

    Returns dict with keys: 'weights' (state_dict), 'config_str', 'spm_bytes'.
    """
    paths = unpack_nemo(nemo_path, extract_dir)
    sd = torch.load(paths["weights"], map_location="cpu", weights_only=True, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    result = {"weights": sd}
    result["config_str"] = paths["config"].read_text() if "config" in paths else ""
    result["spm_bytes"] = paths["spm"].read_bytes()
    if "vocab" in paths:
        result["vocab_bytes"] = paths["vocab"].read_bytes()
    return result


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# HF-transformers ParakeetForTDT checkpoints (#454: moondream/parakeet-ultra,
# moondream/parakeet-redux). Same architecture as the NeMo v3 model with
# transformers' names, no .nemo: rename to the NeMo keys (so the rest of this
# converter is shared), synthesise the featurizer filterbank + window exactly
# as NeMo / transformers' ParakeetFeatureExtractor build them, and dequantise
# parakeet-redux's base-3 packed ternary weights.
# ---------------------------------------------------------------------------

_HF_LAYER_SUBS = [
    ("self_attn.q_proj.", "self_attn.linear_q."),
    ("self_attn.k_proj.", "self_attn.linear_k."),
    ("self_attn.v_proj.", "self_attn.linear_v."),
    ("self_attn.o_proj.", "self_attn.linear_out."),
    ("self_attn.relative_k_proj.", "self_attn.linear_pos."),
    ("self_attn.bias_u", "self_attn.pos_bias_u"),
    ("self_attn.bias_v", "self_attn.pos_bias_v"),
    ("conv.norm.", "conv.batch_norm."),
]


def hf_to_nemo_name(k: str) -> str | None:
    """transformers ParakeetForTDT key -> NeMo key, or None to drop."""
    if k.startswith("vad_head."):
        return None  # moondream's speech-activity head; transcription does not use it
    if k.startswith("encoder.subsampling.layers."):
        return k.replace("encoder.subsampling.layers.", "encoder.pre_encode.conv.")
    if k.startswith("encoder.subsampling.linear."):
        return k.replace("encoder.subsampling.linear.", "encoder.pre_encode.out.")
    if k.startswith("encoder.layers."):
        for a, b in _HF_LAYER_SUBS:
            k = k.replace(a, b)
        return k
    if k == "decoder.embedding.weight":
        return "decoder.prediction.embed.weight"
    if k.startswith("decoder.lstm."):
        return "decoder.prediction.dec_rnn.lstm." + k[len("decoder.lstm."):]
    if k.startswith("decoder.decoder_projector."):
        return "joint.pred." + k[len("decoder.decoder_projector."):]
    if k.startswith("encoder_projector."):
        return "joint.enc." + k[len("encoder_projector."):]
    if k.startswith("joint.head."):
        return "joint.joint_net.2." + k[len("joint.head."):]
    return k


def dequant_ternary(qweight, scales, in_features: int, group: int):
    """parakeet-redux 'thrush-ternary-v2': element i of a row is base-3 digit i%5
    of byte i//5 (least significant first), w = scales[row, i // group] * (code - 1)."""
    q = qweight.to(torch.int64)
    digits = torch.stack([(q // (3 ** d)) % 3 for d in range(5)], dim=-1)  # (rows, bytes, 5)
    codes = digits.reshape(q.shape[0], -1)[:, :in_features]
    s = scales.to(torch.float32).repeat_interleave(group, dim=1)[:, :in_features]
    return s * (codes.to(torch.float32) - 1.0)


def load_hf(path: str) -> dict:
    """An HF ParakeetForTDT snapshot (dir or repo id) -> the dict convert() expects."""
    import json
    from safetensors.torch import load_file

    p = Path(path)
    if not p.is_dir():
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(path, allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                                         "ternary.json"]))
    cfg = json.loads((p / "config.json").read_text())
    if cfg.get("model_type") != "parakeet_tdt":
        sys.exit(f"{p}: model_type {cfg.get('model_type')!r}, expected parakeet_tdt")
    raw = load_file(str(p / "model.safetensors"))
    tern = None
    if (p / "ternary.json").exists():
        tern = json.loads((p / "ternary.json").read_text())
        if tern.get("format") != "thrush-ternary-v2" or tern.get("quant", {}).get("mode") != "ternary":
            sys.exit(f"unsupported ternary format: {tern.get('format')}")
    group = int(cfg.get("ternary_group_size", 128))
    enc = cfg["encoder_config"]
    d, ffd = enc["hidden_size"], enc["intermediate_size"]

    sd = {}
    for k, t in raw.items():
        if k.endswith(".scales"):
            continue
        if k.endswith(".qweight"):
            base = k[: -len(".qweight")]
            in_f = ffd if base.endswith("linear2") else d
            w = dequant_ternary(t, raw[base + ".scales"], in_f, group)
            if ".conv.pointwise_conv" in base:
                w = w.unsqueeze(-1)  # Conv1d(k=1) layout (out, in, 1)
            k, t = base + ".weight", w
        n = hf_to_nemo_name(k)
        if n is not None:
            sd[n] = t
    if tern is not None:
        n_tern = sum(1 for k in raw if k.endswith(".qweight"))
        print(f"  ternary: dequantised {n_tern} matrices (group {group})")

    # Featurizer, as NeMo's FilterbankFeatures and transformers'
    # ParakeetFeatureExtractor build it: librosa slaney mel over n_fft 512, and a
    # symmetric Hann window of 400 samples.
    import librosa
    sr, n_fft, win, hop, n_mels = 16000, 512, 400, 160, int(enc.get("num_mel_bins", 128))
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2, norm="slaney")
    sd["preprocessor.featurizer.fb"] = torch.from_numpy(np.asarray(fb, dtype=np.float32))[None]
    sd["preprocessor.featurizer.window"] = torch.hann_window(win, periodic=False)

    tok = json.loads((p / "tokenizer.json").read_text())["model"]["vocab"]
    vocab = [None] * len(tok)
    for piece, i in (tok.items() if isinstance(tok, dict) else [(pc, i) for i, (pc, _) in enumerate(tok)]):
        vocab[i] = piece
    if any(v is None for v in vocab):
        sys.exit("tokenizer.json: vocab ids are not contiguous")

    nemo_cfg = {
        "preprocessor": {"sample_rate": sr, "features": n_mels, "n_fft": n_fft, "window_size": win / sr,
                         "window_stride": hop / sr},
        "encoder": {"feat_in": n_mels, "d_model": d, "n_layers": enc["num_hidden_layers"],
                    "n_heads": enc["num_attention_heads"], "ff_expansion_factor": ffd // d,
                    "subsampling_factor": enc["subsampling_factor"],
                    "subsampling_conv_channels": enc["subsampling_conv_channels"],
                    "conv_kernel_size": enc["conv_kernel_size"], "xscaling": bool(enc.get("scale_input", False)),
                    "self_attention_model": "rel_pos"},
        "decoder": {"prednet": {"pred_hidden": cfg["decoder_hidden_size"],
                                "pred_rnn_layers": cfg["num_decoder_layers"]},
                    "durations": cfg["durations"]},
        "joint": {"jointnet": {"joint_hidden": int(sd["joint.pred.weight"].shape[0])}},
    }
    import yaml
    return {"weights": sd, "config_str": yaml.safe_dump(nemo_cfg), "vocab": vocab}


def remap_name(nemo_name: str) -> str | None:
    """
    Map NeMo state-dict keys to GGUF-friendly names.
    Returns None for tensors we deliberately drop (e.g. num_batches_tracked).
    """
    n = nemo_name

    # Skip BatchNorm stats counter
    if n.endswith("num_batches_tracked"):
        return None

    # ---- preprocessor (mel filterbank + Hann window) ----
    if n == "preprocessor.featurizer.fb":
        return "preprocessor.fb"
    if n == "preprocessor.featurizer.window":
        return "preprocessor.window"

    # ---- pre-encoder (subsampling Conv2d stack) ----
    if n.startswith("encoder.pre_encode."):
        # encoder.pre_encode.conv.{0,2,3,5,6}.{weight,bias}
        # encoder.pre_encode.out.{weight,bias}
        return n.replace("encoder.pre_encode.", "encoder.pre.")

    # ---- conformer layers ----
    if n.startswith("encoder.layers."):
        rest = n[len("encoder.layers.") :]
        layer_id, sub = rest.split(".", 1)
        sub = (
            sub.replace("feed_forward1", "ff1")
            .replace("feed_forward2", "ff2")
            .replace("norm_feed_forward1", "norm_ff1")
            .replace("norm_feed_forward2", "norm_ff2")
            .replace("norm_self_att", "norm_attn")
            .replace("self_attn.linear_q", "attn.q")
            .replace("self_attn.linear_k", "attn.k")
            .replace("self_attn.linear_v", "attn.v")
            .replace("self_attn.linear_out", "attn.out")
            .replace("self_attn.linear_pos", "attn.pos")
            .replace("self_attn.pos_bias_u", "attn.pos_bias_u")
            .replace("self_attn.pos_bias_v", "attn.pos_bias_v")
            .replace("conv.pointwise_conv1", "conv.pw1")
            .replace("conv.depthwise_conv", "conv.dw")
            .replace("conv.pointwise_conv2", "conv.pw2")
            .replace("conv.batch_norm", "conv.bn")
        )
        return f"encoder.layers.{layer_id}.{sub}"

    # ---- decoder (predictor) ----
    if n == "decoder.prediction.embed.weight":
        return "decoder.embed.weight"
    if n.startswith("decoder.prediction.dec_rnn.lstm."):
        suf = n[len("decoder.prediction.dec_rnn.lstm.") :]
        # weight_ih_l0 / weight_hh_l0 / bias_ih_l0 / bias_hh_l0
        for key, gguf_key in [
            ("weight_ih_l0", "lstm.0.w_ih"),
            ("weight_hh_l0", "lstm.0.w_hh"),
            ("bias_ih_l0", "lstm.0.b_ih"),
            ("bias_hh_l0", "lstm.0.b_hh"),
            ("weight_ih_l1", "lstm.1.w_ih"),
            ("weight_hh_l1", "lstm.1.w_hh"),
            ("bias_ih_l1", "lstm.1.b_ih"),
            ("bias_hh_l1", "lstm.1.b_hh"),
        ]:
            if suf == key:
                return f"decoder.{gguf_key}"

    # ---- joint ----
    if n == "joint.enc.weight":
        return "joint.enc.weight"
    if n == "joint.enc.bias":
        return "joint.enc.bias"
    if n == "joint.pred.weight":
        return "joint.pred.weight"
    if n == "joint.pred.bias":
        return "joint.pred.bias"
    if n == "joint.joint_net.2.weight":
        return "joint.out.weight"
    if n == "joint.joint_net.2.bias":
        return "joint.out.bias"

    # ---- CTC head (hybrid TDT+CTC models like parakeet-tdt_ctc-*) ----
    if n == "ctc_decoder.decoder_layers.0.weight":
        return "ctc.weight"
    if n == "ctc_decoder.decoder_layers.0.bias":
        return "ctc.bias"

    # Unmapped: print a clear warning so any extra structure (e.g. a
    # t_norm or extra joint Linear that the runtime doesn't know about)
    # cannot be silently dropped during conversion.
    print(f"  [WARN unmapped] {n}", file=sys.stderr)
    return None


# Tensors that should stay F32 even when --quant-linear is set:
# layer norms, batch-norm stats, biases, the mel filterbank, the rel-pos biases.
def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    if gguf_name.startswith("preprocessor."):
        return True
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name:
        return True
    if "bn" in gguf_name:
        return True
    if "pos_bias_u" in gguf_name or "pos_bias_v" in gguf_name:
        return True
    if len(shape) <= 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def model_d_for_layer(sd, layer_id: int) -> int:
    """Look up d_model for the given encoder layer (1024 for parakeet-tdt-0.6b-v3)."""
    key = f"encoder.layers.{layer_id}.conv.depthwise_conv.weight"
    return int(sd[key].shape[0])


_QUANT_TYPE_MAP: dict[str, gguf.GGMLQuantizationType] = {
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}


def convert(nemo_path: Path | None, out_path: Path, quant: str | None = None,
            extract_dir: Path | None = None, hf: str | None = None) -> None:
    quant_type = _QUANT_TYPE_MAP.get(quant.lower()) if quant else None
    if quant and quant_type is None:
        sys.exit(f"Unknown --quant type '{quant}'. Choices: {list(_QUANT_TYPE_MAP)}")

    if hf is not None:
        print(f"Loading: {hf}  (HF-transformers ParakeetForTDT)")
        nemo_data = load_hf(hf)
    elif extract_dir is not None:
        print(f"Loading: {nemo_path}  (disk extract to {extract_dir}, mmap)")
        nemo_data = load_nemo_disk(nemo_path, extract_dir)
    else:
        print(f"Loading: {nemo_path}  (in-memory, no disk extraction)")
        nemo_data = load_nemo_inmem(nemo_path)
    sd = nemo_data["weights"]

    import yaml
    cfg = yaml.safe_load(nemo_data["config_str"])

    # Pure-CTC guard. A NeMo EncDecCTCModelBPE (parakeet-ctc-*,
    # stt_*_fastconformer_ctc_*) has an encoder + a CTC classification head
    # (`decoder.decoder_layers.0.weight`) but no RNN-T prediction network
    # (`decoder.prediction.*`) and no joint. This converter targets the
    # transducer (RNN-T/TDT) family; converting a pure-CTC model here would
    # silently drop the CTC head and emit an unusable encoder-only GGUF.
    # Redirect the user to the FastConformer-CTC converter instead.
    has_rnnt_pred = any(k.startswith("decoder.prediction.") for k in sd)
    has_joint = any(k.startswith("joint.") for k in sd)
    has_ctc_head = "decoder.decoder_layers.0.weight" in sd
    if not has_rnnt_pred and not has_joint and has_ctc_head:
        sys.exit(
            "This is a pure-CTC model (EncDecCTCModelBPE): it has a CTC head but no\n"
            "RNN-T prediction/joint network, so it cannot run on the parakeet\n"
            "(transducer) backend. Convert it with the FastConformer-CTC converter:\n"
            f"    python models/convert-stt-fastconformer-ctc-to-gguf.py \\\n"
            f"        --nemo {nemo_path} --output {out_path}\n"
            "The resulting GGUF (arch 'canary-ctc') runs with '--backend fastconformer-ctc'\n"
            "or auto-detects when you omit --backend."
        )

    import io as _io
    if "vocab" in nemo_data:
        vocab = nemo_data["vocab"]
    else:
        sp = spm.SentencePieceProcessor()
        sp.LoadFromSerializedProto(nemo_data["spm_bytes"])
        vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    print(f"  vocab:  {len(vocab)} pieces")

    # ----- write GGUF -----
    print(f"Writing: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch="parakeet")

    # Hyper-parameters — read every value from model_config.yaml when
    # available, falling back to parakeet-tdt-0.6b-v3 defaults only as a
    # last resort. The first round of JA debugging caught a hardcoded
    # n_mels=128 silently reading 80-mel data; never repeat that pattern.
    prep = cfg.get("preprocessor", {}) if cfg else {}
    enc_cfg = cfg.get("encoder", {}) if cfg else {}
    dec_cfg = cfg.get("decoder", {}) if cfg else {}
    pred_cfg = dec_cfg.get("prednet", {}) if cfg else {}
    joint_cfg = cfg.get("joint", {}) if cfg else {}
    joint_net = joint_cfg.get("jointnet", {}) if cfg else {}

    feat_in = enc_cfg.get("feat_in", prep.get("features", 128))
    sr = prep.get("sample_rate", 16000)
    n_fft = prep.get("n_fft", 512)
    ws = prep.get("window_size", 0.025)
    wst = prep.get("window_stride", 0.01)
    d_model = enc_cfg.get("d_model", 1024)
    n_layers = enc_cfg.get("n_layers", 24)
    n_heads = enc_cfg.get("n_heads", 8)
    head_dim = d_model // n_heads
    ff_dim = enc_cfg.get("ff_expansion_factor", 4) * d_model
    subsampling_factor = enc_cfg.get("subsampling_factor", 8)
    subsampling_channels = enc_cfg.get("subsampling_conv_channels", 256)
    conv_kernel = enc_cfg.get("conv_kernel_size", 9)
    xscaling = bool(enc_cfg.get("xscaling", True))
    pred_hidden = pred_cfg.get("pred_hidden", 640)
    pred_layers = pred_cfg.get("pred_rnn_layers", 2)
    # joint_hidden: RNNT uses jointnet.joint_hidden; TDT has encoder_hidden at
    # joint top-level (joint_cfg) or pred_hidden in jointnet. Cross-check below
    # against joint.pred.weight is the ultimate guard.
    joint_hidden = (joint_net.get("joint_hidden")
                    or joint_cfg.get("encoder_hidden")
                    or joint_net.get("encoder_hidden")
                    or joint_net.get("pred_hidden")
                    or 640)
    n_tdt_durations = len(dec_cfg.get("durations", [0, 1, 2, 3, 4]))
    tdt_durations = list(dec_cfg.get("durations", [0, 1, 2, 3, 4]))

    # RNNT models have no duration head: joint.out.weight rows == vocab+1.
    # TDT models have joint.out.weight rows == vocab+1+n_tdt_durations.
    # Cross-check the actual tensor shape and override n_tdt_durations=0 for RNNT
    # so the runtime detects it correctly (n_tdt_durations==0 → RNNT decode path).
    # Note: RNNT checkpoints may use joint.joint_net.2.weight (NeMo RNNTDecoder)
    # instead of joint.out.weight (NeMo TDTDecoder).
    joint_out_w = sd.get("joint.out.weight") or sd.get("joint.joint_net.2.weight")
    if joint_out_w is not None:
        actual_joint_out = int(joint_out_w.shape[0])
        vocab_plus_blank = len(vocab) + 1
        if actual_joint_out == vocab_plus_blank:
            if n_tdt_durations != 0:
                print(
                    f"  [info] joint.out.weight rows={actual_joint_out} == vocab+blank"
                    f" → standard RNNT (no duration head); overriding n_tdt_durations 0",
                    file=sys.stderr,
                )
            n_tdt_durations = 0
            tdt_durations = []
        elif actual_joint_out != vocab_plus_blank + n_tdt_durations:
            print(
                f"  [warn] joint.out.weight rows={actual_joint_out} doesn't match"
                f" vocab+blank+dur={vocab_plus_blank + n_tdt_durations}",
                file=sys.stderr,
            )

    # Cross-check the reported pred_hidden against the actual LSTM weight
    # shape — the surest defence against another silent hparam mismatch.
    lstm0_w_ih = sd.get("decoder.prediction.dec_rnn.lstm.weight_ih_l0")
    if lstm0_w_ih is not None:
        actual = int(lstm0_w_ih.shape[1])
        if actual != pred_hidden:
            print(
                f"  [warn] config pred_hidden={pred_hidden} disagrees "
                f"with lstm.weight_ih_l0 in_dim={actual}; using {actual}",
                file=sys.stderr,
            )
            pred_hidden = actual
    joint_pred_w = sd.get("joint.pred.weight")
    if joint_pred_w is not None:
        actual = int(joint_pred_w.shape[0])
        if actual != joint_hidden:
            print(
                f"  [warn] config joint_hidden={joint_hidden} disagrees "
                f"with joint.pred.weight rows={actual}; using {actual}",
                file=sys.stderr,
            )
            joint_hidden = actual

    print(
        f"  hparams: d_model={d_model} layers={n_layers} heads={n_heads} "
        f"ff={ff_dim} pred_hidden={pred_hidden} joint_hidden={joint_hidden} "
        f"n_mels={feat_in}"
    )

    writer.add_uint32("parakeet.sample_rate", sr)
    writer.add_uint32("parakeet.n_mels", feat_in)
    writer.add_uint32("parakeet.n_fft", n_fft)
    writer.add_uint32("parakeet.win_length", int(ws * sr))
    writer.add_uint32("parakeet.hop_length", int(wst * sr))
    writer.add_uint32("parakeet.d_model", d_model)
    writer.add_uint32("parakeet.n_layers", n_layers)
    writer.add_uint32("parakeet.n_heads", n_heads)
    writer.add_uint32("parakeet.head_dim", head_dim)
    writer.add_uint32("parakeet.ff_dim", ff_dim)
    writer.add_uint32("parakeet.subsampling_factor", subsampling_factor)
    writer.add_uint32("parakeet.subsampling_channels", subsampling_channels)
    writer.add_uint32("parakeet.conv_kernel", conv_kernel)
    writer.add_bool("parakeet.xscaling", xscaling)
    writer.add_uint32("parakeet.pred_hidden", pred_hidden)
    writer.add_uint32("parakeet.pred_layers", pred_layers)
    writer.add_uint32("parakeet.joint_hidden", joint_hidden)
    writer.add_uint32("parakeet.vocab_size", len(vocab))
    writer.add_uint32("parakeet.blank_id", len(vocab))  # blank is vocab_size
    writer.add_uint32("parakeet.n_tdt_durations", n_tdt_durations)
    writer.add_array("parakeet.tdt_durations", tdt_durations)
    # frame_dur_cs is in centiseconds: 0.01 s stride × 8× subsampling = 8 cs (80 ms)
    writer.add_uint32("parakeet.frame_dur_cs", int(round(wst * subsampling_factor * 100)))

    # Local attention context (rel_pos_local_attn models like reazonspeech-nemo-v2).
    # att_context_size: [left, right] in encoder frames (after 8× subsampling).
    # When absent or [−1, −1], the runtime uses full (global) attention.
    att_model = enc_cfg.get("self_attention_model", "rel_pos")
    att_ctx = enc_cfg.get("att_context_size", None)
    if att_model == "rel_pos_local_attn" and att_ctx:
        att_left = att_ctx[0] if isinstance(att_ctx, list) else att_ctx
        att_right = att_ctx[1] if isinstance(att_ctx, list) and len(att_ctx) > 1 else att_left
        writer.add_int32("parakeet.att_context_left", att_left)
        writer.add_int32("parakeet.att_context_right", att_right)
        n_global_tokens = enc_cfg.get("global_tokens", 0)
        writer.add_uint32("parakeet.global_tokens", n_global_tokens)
        print(f"  local attention: left={att_left} right={att_right} global_tokens={n_global_tokens}")

    # Check if CTC head is present (hybrid TDT+CTC models).
    has_ctc = "ctc_decoder.decoder_layers.0.weight" in sd
    writer.add_bool("parakeet.has_ctc", has_ctc)
    if has_ctc:
        ctc_w = sd["ctc_decoder.decoder_layers.0.weight"]
        ctc_vocab = ctc_w.shape[0]
        writer.add_uint32("parakeet.ctc_vocab_size", ctc_vocab)
        print(f"  CTC head: vocab_size={ctc_vocab}")

    writer.add_array("tokenizer.ggml.tokens", vocab)

    # ----- inventory: highlight any decoder/joint structure we don't
    # know about so a JA-specific extra Linear/LayerNorm/Dropout-with-
    # weights can't slip past unnoticed (the converter has been bitten
    # by silent skips before, see HISTORY/parakeet-ja).
    decoder_keys = sorted(k for k in sd.keys() if k.startswith("decoder.prediction."))
    joint_keys = sorted(k for k in sd.keys() if k.startswith("joint."))
    print("  decoder.prediction.* tensors:")
    for k in decoder_keys:
        print(f"    {k}  shape={tuple(sd[k].shape)}")
    print("  joint.* tensors:")
    for k in joint_keys:
        print(f"    {k}  shape={tuple(sd[k].shape)}")

    # ----- tensors -----
    n_written = 0
    n_f16 = 0
    n_f32 = 0
    n_quant = 0
    n_unmapped = 0
    layers_seen = set()
    layers_with_dw_bias = set()
    for name in sorted(sd.keys()):
        gguf_name = remap_name(name)
        if gguf_name is None:
            # remap_name already printed a warning; track for the summary.
            if not name.endswith("num_batches_tracked"):
                n_unmapped += 1
            continue
        t = sd[name].cpu().numpy()
        if t.dtype == np.float64:
            t = t.astype(np.float32)

        raw_dtype = None
        if is_f32_tensor(gguf_name, t.shape):
            t = t.astype(np.float32)
            n_f32 += 1
        elif quant_type is not None:
            t = t.astype(np.float32)
            try:
                t = gguf.quantize(t, quant_type)
                raw_dtype = quant_type
                n_quant += 1
            except Exception:
                # Shape not quantizable (e.g. last dim not divisible by QK_K=256)
                t = t.astype(np.float16)
                n_f16 += 1
        else:
            t = t.astype(np.float16)
            n_f16 += 1

        writer.add_tensor(gguf_name, t, raw_dtype=raw_dtype)
        n_written += 1
        if n_written <= 30 or n_written % 50 == 0:
            dtype_label = str(quant_type.name) if raw_dtype else str(t.dtype)
            print(f"  {gguf_name:60s}  {str(t.shape):28s}  {dtype_label}")

        # Track encoder layers so we can add a zero conv.dw.bias for each.
        if gguf_name.startswith("encoder.layers.") and ".conv.dw.weight" in gguf_name:
            li = int(gguf_name.split(".")[2])
            layers_seen.add(li)
        # Track layers that already have a dw.bias from the checkpoint.
        if gguf_name.startswith("encoder.layers.") and ".conv.dw.bias" in gguf_name:
            li = int(gguf_name.split(".")[2])
            layers_with_dw_bias.add(li)

    # Inject a zero-valued conv.dw.bias per encoder layer that doesn't already
    # have one. Older NeMo models have bias-less depthwise conv (BN provides the
    # bias term), but newer ones (parakeet-ja) have an explicit bias.
    for li in sorted(layers_seen):
        if li in layers_with_dw_bias:
            continue  # already has a real bias from the checkpoint
        bias = np.zeros(int(model_d_for_layer(sd, li)), dtype=np.float32)
        gguf_name = f"encoder.layers.{li}.conv.dw.bias"
        writer.add_tensor(gguf_name, bias)
        n_written += 1
        n_f32 += 1

    quant_label = f", {quant_type.name}: {n_quant}" if quant_type else ""
    print(
        f"\n  total tensors: {n_written}  (F16: {n_f16}, F32: {n_f32}{quant_label})  "
        f"(+{len(layers_seen) - len(layers_with_dw_bias)} synthetic conv.dw.bias)"
    )
    if n_unmapped:
        print(
            f"\n  WARNING: {n_unmapped} tensor(s) were unmapped — see "
            f"[WARN unmapped] lines above. Re-check remap_name() before "
            f"trusting this GGUF for inference.",
            file=sys.stderr,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Parakeet .nemo → GGUF (F16 or quantized)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--nemo", type=Path, help="path to .nemo file")
    src.add_argument("--hf", help="HF-transformers ParakeetForTDT snapshot dir or repo id "
                                  "(e.g. moondream/parakeet-ultra, moondream/parakeet-redux)")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    p.add_argument("--quant", default=None, help="quantize linear weights (e.g. q4_k, q8_0); default: F16")
    p.add_argument("--extract-dir", default=None, type=Path,
                   help="extract .nemo to this dir and load via mmap (low memory for large models)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.nemo, args.output, quant=args.quant, extract_dir=args.extract_dir, hf=args.hf)
