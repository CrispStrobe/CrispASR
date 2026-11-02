#!/usr/bin/env python3
"""Reference dumps from the ORIGINAL ClearerVoice-Studio MossFormer2_SE_48K model.

The upstream package (``clearvoice/clearvoice/models/mossformer2_se`` of
modelscope/ClearerVoice-Studio, source hashes pinned below) is imported from an
explicit checkout path; forward methods are not modified, forward hooks only
copy tensors out.  The decode wrapper ``decode_one_audio_mossformer2_se_48k``
and its helpers (``compute_fbank``, ``stft``, ``istft`` from utils/misc.py) are
compiled from their unmodified source with ``ast``.

One deliberate deviation: upstream ``compute_fbank`` calls Kaldi fbank with
``dither=1.0`` (Gaussian noise of one int16 LSB added to every frame from the
unseeded global RNG), so two upstream runs never agree bit for bit.  The dump
forces ``dither=0.0`` (the native default) unless ``--upstream-dither`` is
given; see docs/mossformer2-se.md for the measured effect.

Layouts are NumPy row-major with time OUTER and channels INNER ([T, C] ==
ggml ne [C, T]):

  input_audio            [L]        mono 48 kHz in [-1, 1]
  fbank                  [T, 60]    Kaldi log-mel fbank of input * 32768
  feats                  [T, 180]   fbank + compute_deltas + delta-deltas
  mn_norm                [T, 180]   MaskNet GroupNorm(1, 180)
  mn_in                  [T, 512]   conv1d_encoder + scaled sinusoidal PE
  l0_to_hidden           [T, 2048]  layer-0 FLASH to_hidden (FFConvM)
  l0_to_qk               [T, 128]   layer-0 FLASH to_qk (FFConvM)
  l0_fsmn_conv1          [T, 256]   layer-0 Gated-FSMN conv1 + PReLU
  l0_fsmn_uni            [T, 256]   layer-0 UniDeepFsmn output
  l0_fsmn_gated          [T, 256]   layer-0 Gated_FSMN output
  l{i}_flash, l{i}_fsmn  [T, 512]   FLASH layer / Gated-FSMN block outputs
  intra_out              [T, 512]   MossFormerM output (after final LayerNorm)
  mdl_out                [T, 512]   Computation_Block output (GroupNorm + skip)
  mask                   [T, 961]   predicted magnitude mask (speaker 0)
  e2e_wave               [L]        decode_one_audio_mossformer2_se_48k(clip),
                                    truncated to L as ClearVoice.decode() does

Standalone CLI (writes .npy files and/or a GGUF archive for
``crispasr-diff mossformer2-se <model.gguf> <ref.gguf> <audio>``)::

    python tools/reference_backends/mossformer2_se.py \\
        --upstream ClearerVoice-Studio --checkpoint last_best_checkpoint.pt \\
        --audio noisy48k.wav --gguf ref.gguf [--e2e-only] [--seconds 4]

Shared-harness entry point (tools/dump_reference.py --backend mossformer2-se):
``model_dir`` is the checkpoint file or its directory; the upstream checkout
comes from ``$CLEARVOICE_STUDIO_DIR``.  Requires torch, torchaudio (only
``compliance.kaldi`` and ``functional.compute_deltas``, both pure Python),
einops, rotary-embedding-torch, torchinfo, soundfile.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

UPSTREAM_REPO = "modelscope/ClearerVoice-Studio"
UPSTREAM_REV = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
MODELSCOPE_REV = "eff8c979"
# Paths relative to <ClearerVoice-Studio>/clearvoice/clearvoice.
SOURCE_HASHES = {
    "models/mossformer2_se/mossformer2.py": "528b7d6078c8ac99806a2baefe2dc707a2d9ddc6595abe95e4084b92b397b94b",
    "models/mossformer2_se/mossformer2_block.py": "8b4f1714156365d9f1e5d96f750793f47d92f51c14d0afd00d43055cd78bdee7",
    "models/mossformer2_se/mossformer2_se_wrapper.py":
        "85e71167a9025a137f9ca8d314c3da4a34cd3285dc856d51352050ecc56e4dda",
    "models/mossformer2_se/fsmn.py": "62c376b2708ac1c59853c13cbe9acb7ea011082c39f1a42e1af0f8213a00bb76",
    "models/mossformer2_se/conv_module.py": "cd16d84a55ed22a234d91871d685a7ae97fa3fbca3f0d94bd065ce0b4c9a1180",
    "models/mossformer2_se/layer_norm.py": "d28fea41b3c5a2dd8bb7a1a8c1c97492fcc7eb88cc7b46283fc21e4dbf19f062",
    "utils/decode.py": "e19417ea6b0556af8b002a2260d9ec287ed4d776eb49e9612586f217ee13f6be",
    "utils/misc.py": "c8b81a12efd673dd3ead2dd8ca9e09e6be0575482e3a2f2ff0b9a4dc0122ca1b",
    "config/inference/MossFormer2_SE_48K.yaml": "68bb5cef7c8cb2d30bfe3e1b25dceccaff16314335c8d7a68a64e509e463955e",
}
CHECKPOINT_SHA256 = "03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6"
DEFAULT_LAYERS = (0, 1, 2, 5, 11, 17, 23)
MAX_WAV_VALUE = 32768.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_upstream(path: Path) -> Path:
    """Accept the repo root, ``clearvoice/`` or ``clearvoice/clearvoice``."""
    for cand in (path / "clearvoice" / "clearvoice", path / "clearvoice", path):
        if (cand / "models" / "mossformer2_se" / "mossformer2.py").is_file():
            return cand
    raise SystemExit(f"{path}: not a ClearerVoice-Studio checkout (models/mossformer2_se missing)")


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
    for line in (root / "config/inference/MossFormer2_SE_48K.yaml").read_text().splitlines():
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

    pkg_dir = root / "models" / "mossformer2_se"
    pkg_name = "_clearvoice_mossformer2_se"
    spec = importlib.util.spec_from_file_location(pkg_name, pkg_dir / "__init__.py",
                                                  submodule_search_locations=[str(pkg_dir)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)
    wrapper = importlib.import_module(pkg_name + ".mossformer2_se_wrapper")
    # networks.CLS_MossFormer2_SE_48K: self.model = MossFormer2_SE_48K(args).model
    model = wrapper.MossFormer2_SE_48K(args_ns).model
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    pretrained = ckpt.get("model", ckpt)
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


def _functions(path: Path, names) -> list:
    tree = ast.parse(path.read_text())
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(found) != len(names):
        raise SystemExit(f"{path}: expected functions {names}")
    return found


def load_decode_namespace(root: Path, upstream_dither: bool = False) -> dict:
    """decode_one_audio_mossformer2_se_48k + compute_fbank/stft/istft, verbatim.

    decode.py / misc.py import unrelated helpers (librosa, joblib, ...), so the
    needed functions are compiled from their unmodified source into one
    namespace.  ``torchaudio`` there is a proxy whose kaldi.fbank forces
    dither=0 unless ``upstream_dither``.
    """
    import torch
    import torchaudio
    import torchaudio.functional as taf
    from torchaudio.compliance import kaldi

    def fbank(*a, **k):
        if not upstream_dither:
            k["dither"] = 0.0
        return kaldi.fbank(*a, **k)

    proxy = types.SimpleNamespace(compliance=types.SimpleNamespace(kaldi=types.SimpleNamespace(fbank=fbank)),
                                  functional=taf, __version__=torchaudio.__version__)
    ns = {"np": np, "torch": torch, "torchaudio": proxy, "MAX_WAV_VALUE": MAX_WAV_VALUE}
    for rel, names in (("utils/misc.py", ["compute_fbank", "stft", "istft"]),
                       ("utils/decode.py", ["decode_one_audio_mossformer2_se_48k"])):
        body = _functions(root / rel, names)
        # Pinned (sha256-verified) upstream source, compiled verbatim.
        exec(compile(ast.Module(body=body, type_ignores=[]), str(root / rel), "exec"), ns)  # noqa: S102
    return ns


def features(ns: dict, audio: np.ndarray, cfg):
    """The feature block of decode_one_audio_mossformer2_se_48k (one window)."""
    import torch

    x = torch.from_numpy(audio.astype(np.float32) * MAX_WAV_VALUE)
    fb = ns["compute_fbank"](x.unsqueeze(0), cfg)
    taf = ns["torchaudio"].functional
    tr = torch.transpose(fb, 0, 1)
    d1 = taf.compute_deltas(tr)
    d2 = taf.compute_deltas(d1)
    feats = torch.cat([fb, torch.transpose(d1, 0, 1), torch.transpose(d2, 0, 1)], dim=1)
    return fb, feats


def to_tc(t) -> np.ndarray:
    """[1, C, T] torch -> [T, C] numpy."""
    a = t.detach().float().cpu().numpy()
    assert a.ndim == 3 and a.shape[0] == 1, a.shape
    return np.ascontiguousarray(a[0].T)


def to_btc(t) -> np.ndarray:
    """[1, T, C] (or [T, C] after an upstream squeeze) torch -> [T, C] numpy."""
    a = t.detach().float().cpu().numpy()
    if a.ndim == 3:
        assert a.shape[0] == 1, a.shape
        a = a[0]
    return np.ascontiguousarray(a)


def capture_stages(model, feats, layers) -> dict:
    import torch

    caps = {}
    hooks = []
    mn = model.mossformer
    block = mn.mdl.intra_mdl.mossformerM

    def fwd(name, conv):
        def hook(_mod, _inp, out):
            caps[name] = conv(out)
        return hook

    hooks.append(mn.norm.register_forward_hook(fwd("mn_norm", to_tc)))
    hooks.append(mn.mdl.register_forward_pre_hook(lambda _m, inp: caps.__setitem__("mn_in", to_tc(inp[0]))))
    hooks.append(block.layers[0].to_hidden.register_forward_hook(fwd("l0_to_hidden", to_btc)))
    hooks.append(block.layers[0].to_qk.register_forward_hook(fwd("l0_to_qk", to_btc)))
    hooks.append(block.fsmn[0].conv1.register_forward_hook(fwd("l0_fsmn_conv1", to_tc)))
    hooks.append(block.fsmn[0].gated_fsmn.fsmn.register_forward_hook(fwd("l0_fsmn_uni", to_btc)))
    hooks.append(block.fsmn[0].gated_fsmn.register_forward_hook(fwd("l0_fsmn_gated", to_btc)))
    for i in layers:
        hooks.append(block.layers[i].register_forward_hook(fwd(f"l{i}_flash", to_btc)))
        hooks.append(block.fsmn[i].register_forward_hook(fwd(f"l{i}_fsmn", to_btc)))
    hooks.append(mn.mdl.intra_mdl.register_forward_hook(fwd("intra_out", to_btc)))
    hooks.append(mn.mdl.register_forward_hook(fwd("mdl_out", to_tc)))
    try:
        with torch.no_grad():
            out = model(feats.unsqueeze(0))
    finally:
        for h in hooks:
            h.remove()
    caps["mask"] = to_btc(out[-1])
    return caps


DEFAULT_STAGES = [
    "fbank", "feats", "mn_norm", "mn_in", "l0_to_hidden", "l0_to_qk", "l0_fsmn_conv1", "l0_fsmn_uni",
    "l0_fsmn_gated", *[f"l{i}_{k}" for i in DEFAULT_LAYERS for k in ("flash", "fsmn")],
    "intra_out", "mdl_out", "mask", "e2e_wave",
]


def run(root: Path, checkpoint: Path, audio: np.ndarray, layers=DEFAULT_LAYERS, e2e_only=False,
        allow_mismatch=False, upstream_dither=False) -> dict:
    """Pinned-source model + stage captures + decode-pipeline output."""
    import torch

    verify_sources(root, allow_mismatch)
    cfg = read_config(root)
    model = load_upstream_model(root, checkpoint, cfg)
    ns = load_decode_namespace(root, upstream_dither)
    caps = {"input_audio": np.ascontiguousarray(audio, dtype=np.float32)}
    if not e2e_only:
        fb, feats = features(ns, caps["input_audio"], cfg)
        caps["fbank"] = np.ascontiguousarray(fb.numpy())
        caps["feats"] = np.ascontiguousarray(feats.numpy())
        caps.update(capture_stages(model, feats, layers))
    with torch.no_grad():
        out = ns["decode_one_audio_mossformer2_se_48k"](model, torch.device("cpu"),
                                                        caps["input_audio"][None, :].copy(), cfg)
    caps["e2e_wave"] = np.ascontiguousarray(np.asarray(out[:len(audio)], dtype=np.float32))
    return caps


def dump(model_dir, audio, stages, max_new_tokens=None, **kwargs):
    """tools/dump_reference.py adapter: requested stages out.

    The shared harness always hands over 16-kHz mono; it is upsampled 3x
    (scipy resample_poly) to the model's 48 kHz, and the archive's
    ``input_audio`` is that 48-kHz signal, so the native diff replays exactly
    what the reference saw.  Use the standalone CLI for real 48-kHz clips.
    """
    from scipy.signal import resample_poly

    audio = resample_poly(np.asarray(audio, dtype=np.float64), 3, 1)
    upstream = os.environ.get("CLEARVOICE_STUDIO_DIR")
    if not upstream:
        raise SystemExit("set CLEARVOICE_STUDIO_DIR to a ClearerVoice-Studio checkout @ " + UPSTREAM_REV[:8])
    ckpt = Path(model_dir)
    if ckpt.is_dir():
        ckpt = ckpt / "last_best_checkpoint.pt"
    wanted = set(stages or DEFAULT_STAGES)
    layers = sorted({int(s[1:].split("_")[0]) for s in wanted if s[0] == "l" and s[1].isdigit()} | {0})
    e2e_only = not any(not s.startswith("e2e_") for s in wanted)
    caps = run(resolve_upstream(Path(upstream)), ckpt, audio.astype(np.float32), layers, e2e_only)
    return {k: v for k, v in caps.items() if k in wanted or k == "input_audio"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True, help="ClearerVoice-Studio checkout @ " + UPSTREAM_REV[:8])
    ap.add_argument("--checkpoint", type=Path, required=True, help="MossFormer2_SE_48K last_best_checkpoint.pt")
    ap.add_argument("--audio", type=Path, required=True, help="48-kHz mono WAV")
    ap.add_argument("--npy-dir", type=Path)
    ap.add_argument("--gguf", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="clip start (seconds)")
    ap.add_argument("--seconds", type=float, help="clip length (seconds); default: to the end")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)), help="layers whose outputs to capture")
    ap.add_argument("--e2e-only", action="store_true", help="only the decode-pipeline output")
    ap.add_argument("--upstream-dither", action="store_true",
                    help="keep upstream's random Kaldi dither (1.0); not reproducible")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--allow-source-mismatch", action="store_true")
    args = ap.parse_args()
    if not args.npy_dir and not args.gguf:
        ap.error("pass --npy-dir and/or --gguf")

    import soundfile as sf
    import torch

    torch.set_num_threads(args.threads)
    root = resolve_upstream(args.upstream)
    ck_sha = sha256_file(args.checkpoint)
    if ck_sha != CHECKPOINT_SHA256:
        msg = f"checkpoint sha256 {ck_sha} != pinned {CHECKPOINT_SHA256}"
        if not args.allow_source_mismatch:
            raise SystemExit(msg)
        print("warning:", msg, file=sys.stderr)
    cfg = read_config(root)
    audio, sr = sf.read(str(args.audio), dtype="float32", always_2d=True)
    if sr != cfg.sampling_rate:
        raise SystemExit(f"{args.audio}: {sr} Hz, model needs {cfg.sampling_rate} Hz (resample explicitly first)")
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    s0 = round(args.start * sr)
    audio = audio[s0:] if args.seconds is None else audio[s0:s0 + round(args.seconds * sr)]
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if len(audio) < cfg.win_len:
        raise SystemExit(f"clip shorter than one {cfg.win_len}-sample analysis window (upstream cannot decode it)")

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    caps = run(root, args.checkpoint, audio, layers, args.e2e_only, args.allow_source_mismatch,
               args.upstream_dither)
    meta = {
        "backend": "mossformer2-se",
        "upstream_repo": UPSTREAM_REPO,
        "upstream_rev": UPSTREAM_REV,
        "modelscope_rev": MODELSCOPE_REV,
        "checkpoint_sha256": ck_sha,
        "sample_rate": int(sr),
        "n_samples": len(audio),
        "segmented": int(len(audio) > cfg.sampling_rate * cfg.one_time_decode_length),
        "fbank_dither": 1.0 if args.upstream_dither else 0.0,
        "input_pcm_sha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
        "stages_json": json.dumps(sorted(caps)),
    }
    if args.npy_dir:
        args.npy_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in caps.items():
            np.save(args.npy_dir / f"{name}.npy", arr)
        (args.npy_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if args.gguf:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from dump_reference import write_gguf_archive

        write_gguf_archive(caps, meta, args.gguf)
    print(f"{len(audio)} samples ({len(audio) / sr:.2f} s), {len(caps)} stages")
    for name in sorted(caps):
        print(f"  {name:16s} {list(caps[name].shape)}")


if __name__ == "__main__":
    main()
