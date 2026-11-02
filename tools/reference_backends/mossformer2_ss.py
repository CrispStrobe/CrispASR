"""Hooks on the ORIGINAL ClearerVoice-Studio MossFormer2_SS_16K PyTorch model.

The upstream package (``clearvoice/clearvoice/models/mossformer2_ss`` of
modelscope/ClearerVoice-Studio, source hashes pinned below) is imported from an
explicit checkout path; forward methods are not modified, forward hooks only
copy tensors out.  Layouts are NumPy row-major with time OUTER and channels
INNER ([T, C] == ggml ne [C, T]); see tools/dump_mossformer2_ss_reference.py
for the stage list and the standalone CLI (which also writes .npy files).

Shared-harness entry point (tools/dump_reference.py --backend mossformer2-ss):
``model_dir`` is the checkpoint file or the directory holding
``last_best_checkpoint.pt``; the upstream checkout comes from
``$CLEARVOICE_STUDIO_DIR``.  Requires einops, rotary-embedding-torch, torchinfo.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

UPSTREAM_REPO = "modelscope/ClearerVoice-Studio"
UPSTREAM_REV = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
HF_REPO = "alibabasglab/MossFormer2_SS_16K"
HF_REV = "407cb030"
# Paths relative to <ClearerVoice-Studio>/clearvoice/clearvoice.
SOURCE_HASHES = {
    "models/mossformer2_ss/mossformer2.py": "66f14af8efc0758d8c79bd7d1054851558081c5f6f1150cdf133ae4b6ca9c442",
    "models/mossformer2_ss/mossformer2_block.py": "2760c18c1eb68f8ba2f1160e9a46dc6adfbcb397a693dad27ae6e43e8a02faf0",
    "models/mossformer2_ss/fsmn.py": "62c376b2708ac1c59853c13cbe9acb7ea011082c39f1a42e1af0f8213a00bb76",
    "models/mossformer2_ss/conv_module.py": "b413aa0cb0df34742460a1f1af2e2cd1d2005900af8ac17e8de8675b8f80de86",
    "models/mossformer2_ss/layer_norm.py": "d28fea41b3c5a2dd8bb7a1a8c1c97492fcc7eb88cc7b46283fc21e4dbf19f062",
    "utils/decode.py": "e19417ea6b0556af8b002a2260d9ec287ed4d776eb49e9612586f217ee13f6be",
    "config/inference/MossFormer2_SS_16K.yaml": "839fb7cc7eac064552e397172ef2d48fb2762180f3aaba12c04b5899f9edfb00",
}
CHECKPOINT_SHA256 = "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a"
DEFAULT_LAYERS = (0, 1, 2, 5, 11, 17, 23)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_upstream(path: Path) -> Path:
    """Accept the repo root, ``clearvoice/`` or ``clearvoice/clearvoice``."""
    for cand in (path / "clearvoice" / "clearvoice", path / "clearvoice", path):
        if (cand / "models" / "mossformer2_ss" / "mossformer2.py").is_file():
            return cand
    raise SystemExit(f"{path}: not a ClearerVoice-Studio checkout (models/mossformer2_ss missing)")


def verify_sources(root: Path, allow_mismatch: bool) -> None:
    for rel, want in SOURCE_HASHES.items():
        got = sha256_file(root / rel)
        if got != want:
            msg = f"{rel}: sha256 {got} != pinned {want} (upstream {UPSTREAM_REV})"
            if not allow_mismatch:
                raise SystemExit(msg + "; pass --allow-source-mismatch to override")
            print("warning:", msg, file=sys.stderr)


def read_config(root: Path) -> types.SimpleNamespace:
    cfg = {}
    for line in (root / "config/inference/MossFormer2_SS_16K.yaml").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = (s.strip().strip("'\"") for s in line.split(":", 1))
        try:
            cfg[key] = int(value)
        except ValueError:
            try:
                cfg[key] = float(value)
            except ValueError:
                cfg[key] = value
    return types.SimpleNamespace(**cfg)


def load_upstream_model(root: Path, checkpoint: Path, args_ns):
    import torch

    pkg_dir = root / "models" / "mossformer2_ss"
    pkg_name = "_clearvoice_mossformer2_ss"
    spec = importlib.util.spec_from_file_location(pkg_name, pkg_dir / "__init__.py",
                                                  submodule_search_locations=[str(pkg_dir)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)
    mossformer2 = importlib.import_module(pkg_name + ".mossformer2")
    # networks.CLS_MossFormer2_SS_16K: self.model = MossFormer2_SS_16K(args).model
    model = mossformer2.MossFormer2_SS_16K(args_ns).model
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    pretrained = ckpt["model"] if "model" in ckpt else ckpt
    # networks.SpeechModel._load_model key handling (module. prefixes).
    state = model.state_dict()
    missing = []
    for key in state:
        for cand in (key, key.replace("module.", ""), "module." + key):
            if cand in pretrained and state[key].shape == pretrained[cand].shape:
                state[key] = pretrained[cand]
                break
        else:
            missing.append(key)
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} model tensors, e.g. {missing[:3]}")
    model.load_state_dict(state)
    model.eval()
    return model


def load_decode_fn(root: Path):
    """Extract decode_one_audio_mossformer2_ss_16k verbatim from utils/decode.py.

    decode.py imports torchaudio/librosa helpers unrelated to this model, so
    the single function is compiled from its unmodified source instead.
    """
    import torch

    src = (root / "utils" / "decode.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode_one_audio_mossformer2_ss_16k")
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(root / "utils" / "decode.py"), "exec"), namespace)
    return namespace["decode_one_audio_mossformer2_ss_16k"]


def to_tc(t) -> np.ndarray:
    """[1, C, T] torch -> [T, C] numpy."""
    a = t.detach().float().cpu().numpy()
    assert a.ndim == 3 and a.shape[0] == 1, a.shape
    return np.ascontiguousarray(a[0].T)


def to_btc(t) -> np.ndarray:
    """[1, T, C] torch -> [T, C] numpy."""
    a = t.detach().float().cpu().numpy()
    assert a.ndim == 3 and a.shape[0] == 1, a.shape
    return np.ascontiguousarray(a[0])


def capture_stages(model, audio: np.ndarray, layers) -> dict:
    import torch

    caps = {}
    hooks = []
    mn = model.mask_net
    block = mn.mdl.intra_mdl.mossformerM

    def fwd(name, conv):
        def hook(_mod, _inp, out):
            caps[name] = conv(out)
        return hook

    hooks.append(model.enc.register_forward_hook(fwd("enc_out", to_tc)))
    hooks.append(mn.norm.register_forward_hook(fwd("mn_norm", to_tc)))
    hooks.append(mn.mdl.register_forward_pre_hook(lambda _m, inp: caps.__setitem__("mn_in", to_tc(inp[0]))))
    hooks.append(block.layers[0].to_hidden.register_forward_hook(fwd("l0_to_hidden", to_btc)))
    hooks.append(block.layers[0].to_qk.register_forward_hook(fwd("l0_to_qk", to_btc)))
    hooks.append(block.fsmn[0].conv1.register_forward_hook(fwd("l0_fsmn_conv1", to_tc)))
    hooks.append(block.fsmn[0].gated_fsmn.fsmn.conv.register_forward_hook(
        fwd("l0_fsmn_ddn", lambda o: np.ascontiguousarray(o.detach().float().cpu().numpy()[0, :, :, 0].T))))
    hooks.append(block.fsmn[0].gated_fsmn.register_forward_hook(fwd("l0_fsmn_gated", to_btc)))
    for i in layers:
        hooks.append(block.layers[i].register_forward_hook(fwd(f"l{i}_flash", to_btc)))
        hooks.append(block.fsmn[i].register_forward_hook(fwd(f"l{i}_fsmn", to_btc)))
    hooks.append(mn.mdl.intra_mdl.register_forward_hook(fwd("intra_out", to_btc)))
    hooks.append(mn.mdl.register_forward_hook(fwd("mdl_out", to_tc)))
    hooks.append(mn.register_forward_hook(fwd(
        "masks", lambda o: np.ascontiguousarray(o.detach().float().cpu().numpy()[:, 0].transpose(0, 2, 1)))))
    try:
        with torch.no_grad():
            outs = model(torch.from_numpy(audio[None, :].astype(np.float32)))
    finally:
        for h in hooks:
            h.remove()
    for spk, o in enumerate(outs):
        caps[f"dec_spk{spk + 1}"] = np.ascontiguousarray(o[0].detach().float().cpu().numpy())
    return caps



DEFAULT_STAGES = [
    "enc_out", "mn_norm", "mn_in", "l0_to_hidden", "l0_to_qk", "l0_fsmn_conv1", "l0_fsmn_ddn", "l0_fsmn_gated",
    *[f"l{i}_{k}" for i in DEFAULT_LAYERS for k in ("flash", "fsmn")],
    "intra_out", "mdl_out", "masks", "dec_spk1", "dec_spk2", "e2e_spk1", "e2e_spk2",
]


def run(root: Path, checkpoint: Path, audio: np.ndarray, layers=DEFAULT_LAYERS, e2e_only=False,
        allow_mismatch=False) -> dict:
    """Pinned-source model + stage captures + decode-pipeline outputs."""
    import torch

    verify_sources(root, allow_mismatch)
    cfg = read_config(root)
    model = load_upstream_model(root, checkpoint, cfg)
    caps = {"input_audio": np.ascontiguousarray(audio, dtype=np.float32)}
    if not e2e_only:
        caps.update(capture_stages(model, caps["input_audio"], layers))
    decode = load_decode_fn(root)
    with torch.no_grad():
        outs = decode(model, torch.device("cpu"), caps["input_audio"][None, :].copy(), cfg)
    for spk in range(cfg.num_spks):
        caps[f"e2e_spk{spk + 1}"] = np.ascontiguousarray(np.asarray(outs[spk][:len(audio)], dtype=np.float32))
    return caps


def dump(model_dir, audio, stages, max_new_tokens=None, **kwargs):
    """tools/dump_reference.py adapter: 16-kHz mono in, requested stages out."""
    upstream = os.environ.get("CLEARVOICE_STUDIO_DIR")
    if not upstream:
        raise SystemExit("set CLEARVOICE_STUDIO_DIR to a ClearerVoice-Studio checkout @ " + UPSTREAM_REV[:8])
    ckpt = Path(model_dir)
    if ckpt.is_dir():
        ckpt = ckpt / "last_best_checkpoint.pt"
    wanted = set(stages or DEFAULT_STAGES)
    layers = sorted({int(s[1:].split("_")[0]) for s in wanted if s[0] == "l" and s[1].isdigit()} | {0})
    e2e_only = not any(not s.startswith("e2e_") for s in wanted)
    caps = run(resolve_upstream(Path(upstream)), ckpt, np.asarray(audio, dtype=np.float32), layers, e2e_only)
    return {k: v for k, v in caps.items() if k in wanted or k == "input_audio"}
