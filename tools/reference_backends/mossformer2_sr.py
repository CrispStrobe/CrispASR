#!/usr/bin/env python3
"""Reference dumps from the ORIGINAL ClearerVoice-Studio MossFormer2_SR_48K model.

MossFormer2_SR_48K is a two-stage speech super-resolution model: a MossFormer2
MaskNet (``Mossformer`` in models/mossformer2_sr/generator.py, 80 log-mel bins
in, 80 "mel" channels out) feeding a HiFi-GAN-style generator with Snake
activations (``Generator``, x256 upsampling to 48 kHz), followed by
``bandwidth_sub`` (utils/bandwidth_sub.py), which keeps the input's own band
below its detected cut-off and takes only the band above it from the generator.

The upstream package (``clearvoice/clearvoice/models/mossformer2_sr`` of
modelscope/ClearerVoice-Studio, source hashes pinned below) is imported from an
explicit checkout path; forward methods are not modified, forward hooks only
copy tensors out.  ``decode_one_audio_mossformer2_sr_48k`` / ``get_mel``
(utils/decode.py), ``mel_spectrogram`` and its helpers
(dataloader/meldataset.py) and the ``bandwidth_sub`` functions are compiled
from their unmodified source with ``ast``.  As networks.CLS_MossFormer2_SR_48K
does, the generator's weight norm is removed after loading.

Layouts are NumPy row-major with time OUTER and channels INNER ([T, C] ==
ggml ne [C, T]):

  input_audio            [L]          mono 48 kHz in [-1, 1]
  mel                    [T, 80]      get_mel(): log-mel, n_fft 1024, hop 256, 0-8 kHz
  mn_norm                [T, 80]      MaskNet GroupNorm(1, 80)
  mn_in                  [T, 512]     conv1d_encoder + scaled sinusoidal PE
  l0_to_hidden           [T, 2048]    layer-0 FLASH to_hidden (FFConvM)
  l0_to_qk               [T, 128]     layer-0 FLASH to_qk (FFConvM)
  l0_fsmn_conv1          [T, 256]     layer-0 Gated-FSMN conv1 + PReLU
  l0_fsmn_uni            [T, 256]     layer-0 UniDeepFsmn output
  l0_fsmn_gated          [T, 256]     layer-0 Gated_FSMN output
  l{i}_flash, l{i}_fsmn  [T, 512]     FLASH layer / Gated-FSMN block outputs
  intra_out              [T, 512]     MossFormerM output (after final LayerNorm)
  mdl_out                [T, 512]     Computation_Block output (GroupNorm + skip)
  mf_out                 [T, 80]      MaskNet output (generator input)
  g_pre                  [T, 1024]    generator conv_pre                (--gen-stages)
  g_ups{i}               [T_i, C_i]   ConvTranspose1d output of stage i (--gen-stages)
  g_mrf{i}               [T_i, C_i]   stage i after the resblock average (--gen-stages)
  gen_wave               [256 T]      generator output (tanh)
  e2e_wave               [L]          decode_one_audio_mossformer2_sr_48k(clip) incl.
                                      bandwidth_sub; zero-padded to L where upstream
                                      returns fewer samples (one-pass: 256 * (L // 256))

Standalone CLI (writes .npy files and/or a GGUF archive for
``crispasr-diff mossformer2-sr <model.gguf> <ref.gguf> <audio>``)::

    python tools/reference_backends/mossformer2_sr.py \\
        --upstream ClearerVoice-Studio --checkpoint-dir MossFormer2_SR_48K \\
        --audio in48k.wav --gguf ref.gguf [--e2e-only] [--gen-stages] [--seconds 4]

Shared-harness entry point (tools/dump_reference.py --backend mossformer2-sr):
``model_dir`` is the checkpoint directory (last_best_checkpoint_{m,g}.pt); the
upstream checkout comes from ``$CLEARVOICE_STUDIO_DIR``.  Requires torch,
einops, rotary-embedding-torch, librosa (mel basis), scipy, soundfile, and a
torchaudio importable by generator.py (only its module-level imports are used).
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
MODELSCOPE_REV = "39eb1f25"
# Paths relative to <ClearerVoice-Studio>/clearvoice/clearvoice.
SOURCE_HASHES = {
    "models/mossformer2_sr/mossformer2.py": "5aba71d403837b60e291ab963b9c755b5937d1b7b6209bea0d2b59403a543bff",
    "models/mossformer2_sr/mossformer2_block.py": "8b4f1714156365d9f1e5d96f750793f47d92f51c14d0afd00d43055cd78bdee7",
    "models/mossformer2_sr/mossformer2_sr_wrapper.py":
        "6629e9fc48686f61e48420f5f66334a03db60418597f557469f645c865f455d0",
    "models/mossformer2_sr/generator.py": "11e902ff230339d5abe13ba64c92d8d33d64fe1f25293ccd281241c53ad69661",
    "models/mossformer2_sr/snake.py": "ec2649649d787b166a138d5ad9dcd585aeabbcd93670771b30cbbbab731c4b63",
    "models/mossformer2_sr/fsmn.py": "62c376b2708ac1c59853c13cbe9acb7ea011082c39f1a42e1af0f8213a00bb76",
    "models/mossformer2_sr/conv_module.py": "cd16d84a55ed22a234d91871d685a7ae97fa3fbca3f0d94bd065ce0b4c9a1180",
    "models/mossformer2_sr/layer_norm.py": "d28fea41b3c5a2dd8bb7a1a8c1c97492fcc7eb88cc7b46283fc21e4dbf19f062",
    "models/mossformer2_sr/utils.py": "f55cb87c72434d0a7a2d8003b12e82ba29fd53a8f1d016c7cab98124f7d9203e",
    "utils/decode.py": "e19417ea6b0556af8b002a2260d9ec287ed4d776eb49e9612586f217ee13f6be",
    "utils/bandwidth_sub.py": "348731f17524f5fd5481e1f38762d957119cc5a49b36d3b3f536a084c4b7fdba",
    "dataloader/meldataset.py": "76259659f69c38114e95fbe981fd9c59e2bae8f917452d2c3ad64ca7bb08912f",
    "config/inference/MossFormer2_SR_48K.yaml": "faed51a51161d3c22ca89eec69ff62412cd8517bdb28af65febd14e6830561b3",
    "config/inference/MossFormer2_SR_48K.json": "3251e54c6179ec4b4b07d51136ecaa60996b2086ca263a441977c48313a80a47",
}
CHECKPOINT_M_SHA256 = "6cbadb2b6b839e444bb65223c69eea162c8ad08f36e9d0a64144672c4095ab36"
CHECKPOINT_G_SHA256 = "0bdd13c21466f5963d9d1f86a9d84fc6196868318fe22c6b0a750f041805adda"
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
        if (cand / "models" / "mossformer2_sr" / "generator.py").is_file():
            return cand
    raise SystemExit(f"{path}: not a ClearerVoice-Studio checkout (models/mossformer2_sr missing)")


def checkpoint_paths(ckpt_dir: Path) -> tuple[Path, Path]:
    """last_best_checkpoint names the mossformer file first, the generator second."""
    names = [s.strip() for s in (ckpt_dir / "last_best_checkpoint").read_text().splitlines() if s.strip()]
    if len(names) < 2:
        raise SystemExit(f"{ckpt_dir}/last_best_checkpoint: expected two checkpoint names")
    return ckpt_dir / names[0], ckpt_dir / names[1]


def verify_sources(root: Path, allow_mismatch: bool) -> None:
    for rel, want in SOURCE_HASHES.items():
        got = sha256_file(root / rel)
        if got != want:
            msg = f"{rel}: sha256 {got} != pinned {want} (upstream {UPSTREAM_REV})"
            if not allow_mismatch:
                raise SystemExit(msg + "; pass --allow-source-mismatch to override")
            print("warning:", msg, file=sys.stderr)


def read_config(root: Path) -> types.SimpleNamespace:
    """network_wrapper.combine_config_and_args: JSON config, then the YAML on top."""
    cfg = json.loads((root / "config/inference/MossFormer2_SR_48K.json").read_text())
    for line in (root / "config/inference/MossFormer2_SR_48K.yaml").read_text().splitlines():
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


def _load_state(model, pretrained) -> None:
    """networks.SpeechModel._load_model key handling (module. prefixes)."""
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


def load_upstream_models(root: Path, ckpt_m: Path, ckpt_g: Path, args_ns):
    import torch

    pkg_dir = root / "models" / "mossformer2_sr"
    pkg_name = "_clearvoice_mossformer2_sr"
    spec = importlib.util.spec_from_file_location(pkg_name, pkg_dir / "__init__.py",
                                                  submodule_search_locations=[str(pkg_dir)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)
    wrapper = importlib.import_module(pkg_name + ".mossformer2_sr_wrapper")
    # networks.CLS_MossFormer2_SR_48K
    model_m = wrapper.MossFormer2_SR_48K(args_ns).model_m
    model_g = wrapper.MossFormer2_SR_48K(args_ns).model_g
    _load_state(model_m, torch.load(str(ckpt_m), map_location="cpu", weights_only=True)["mossformer"])
    _load_state(model_g, torch.load(str(ckpt_g), map_location="cpu", weights_only=True)["generator"])
    model_m.eval()
    model_g.eval()
    model_g.remove_weight_norm()
    return model_m, model_g


def _functions(path: Path, names) -> list:
    tree = ast.parse(path.read_text())
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(found) != len(names):
        raise SystemExit(f"{path}: expected functions {names}")
    return found


def load_decode_namespace(root: Path) -> dict:
    """decode_one_audio_mossformer2_sr_48k + get_mel + mel_spectrogram + bandwidth_sub, verbatim.

    decode.py / meldataset.py import unrelated helpers (pydub, torchaudio, ...),
    so the needed functions are compiled from their unmodified source into one
    namespace with the module globals they read.
    """
    import librosa
    import torch
    from scipy.signal import butter, filtfilt, stft

    ns = {"np": np, "torch": torch, "librosa": librosa, "butter": butter, "filtfilt": filtfilt, "stft": stft,
          "mel_basis": {}, "hann_window": {}}
    for rel, names in (("dataloader/meldataset.py",
                        ["dynamic_range_compression_torch", "spectral_normalize_torch", "mel_spectrogram"]),
                       ("utils/bandwidth_sub.py", ["detect_bandwidth", "lowpass_filter", "highpass_filter",
                                                   "replace_bandwidth", "smooth_transition", "bandwidth_sub"]),
                       ("utils/decode.py", ["get_mel", "decode_one_audio_mossformer2_sr_48k"])):
        body = _functions(root / rel, names)
        # Pinned (sha256-verified) upstream source, compiled verbatim.
        exec(compile(ast.Module(body=body, type_ignores=[]), str(root / rel), "exec"), ns)  # noqa: S102
    return ns


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


def capture_stages(model_m, model_g, mel, layers, gen_stages) -> dict:
    import torch

    caps = {}
    hooks = []
    mn = model_m.mossformer
    block = mn.mdl.intra_mdl.mossformerM

    def fwd(name, conv):
        def hook(_mod, _inp, out):
            caps[name] = conv(out)
        return hook

    def pre(name, conv):
        def hook(_mod, inp):
            caps[name] = conv(inp[0])
        return hook

    hooks.append(mn.norm.register_forward_hook(fwd("mn_norm", to_tc)))
    hooks.append(mn.mdl.register_forward_pre_hook(pre("mn_in", to_tc)))
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
    if gen_stages:
        hooks.append(model_g.conv_pre.register_forward_hook(fwd("g_pre", to_tc)))
        n_up = len(model_g.ups)
        for i in range(n_up):
            hooks.append(model_g.ups[i].register_forward_hook(fwd(f"g_ups{i}", to_tc)))
            # The resblock average of stage i is the input of the next Snake.
            nxt = model_g.snakes[i + 1] if i + 1 < n_up else model_g.snake_post
            hooks.append(nxt.register_forward_pre_hook(pre(f"g_mrf{i}", to_tc)))
    try:
        with torch.no_grad():
            mf = model_m(mel)
            caps["mf_out"] = to_tc(mf)
            wave = model_g(mf)
    finally:
        for h in hooks:
            h.remove()
    caps["gen_wave"] = np.ascontiguousarray(wave.detach().float().cpu().numpy().reshape(-1))
    return caps


def run(root: Path, ckpt_dir: Path, audio: np.ndarray, layers=DEFAULT_LAYERS, e2e_only=False,
        gen_stages=False, allow_mismatch=False) -> dict:
    """Pinned-source models + stage captures + decode-pipeline output."""
    import torch

    verify_sources(root, allow_mismatch)
    cfg = read_config(root)
    ckpt_m, ckpt_g = checkpoint_paths(ckpt_dir)
    model_m, model_g = load_upstream_models(root, ckpt_m, ckpt_g, cfg)
    ns = load_decode_namespace(root)
    caps = {"input_audio": np.ascontiguousarray(audio, dtype=np.float32)}
    if not e2e_only:
        mel = ns["get_mel"](torch.from_numpy(caps["input_audio"]).unsqueeze(0), cfg)
        caps["mel"] = to_tc(mel)
        caps.update(capture_stages(model_m, model_g, mel, layers, gen_stages))
    with torch.no_grad():
        out = ns["decode_one_audio_mossformer2_sr_48k"]([model_m, model_g], torch.device("cpu"),
                                                        caps["input_audio"][None, :].copy(), cfg)
    out = np.asarray(out[:len(audio)], dtype=np.float32)  # ClearVoice.decode() truncation
    e2e = np.zeros(len(audio), dtype=np.float32)
    e2e[:len(out)] = out
    caps["e2e_wave"] = e2e
    return caps


def dump(model_dir, audio, stages, max_new_tokens=None, **kwargs):
    """tools/dump_reference.py adapter: requested stages out.

    The shared harness always hands over 16-kHz mono; it is upsampled to the
    model's 48 kHz with librosa.resample (what ClearVoice's audioread does),
    and the archive's ``input_audio`` is that 48-kHz signal, so the native diff
    replays exactly what the reference saw.
    """
    import librosa

    audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=16000, target_sr=48000)
    upstream = os.environ.get("CLEARVOICE_STUDIO_DIR")
    if not upstream:
        raise SystemExit("set CLEARVOICE_STUDIO_DIR to a ClearerVoice-Studio checkout @ " + UPSTREAM_REV[:8])
    ckpt = Path(model_dir)
    if not ckpt.is_dir():
        ckpt = ckpt.parent
    wanted = set(stages or [])
    layers = sorted({int(s[1:].split("_")[0]) for s in wanted if s[0] == "l" and s[1].isdigit()} | {0})
    e2e_only = bool(wanted) and not any(not s.startswith("e2e_") for s in wanted)
    gen = any(s.startswith("g_") for s in wanted)
    caps = run(resolve_upstream(Path(upstream)), ckpt, audio.astype(np.float32), layers or DEFAULT_LAYERS,
               e2e_only, gen)
    if not wanted:
        return caps
    return {k: v for k, v in caps.items() if k in wanted or k == "input_audio"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True, help="ClearerVoice-Studio checkout @ " + UPSTREAM_REV[:8])
    ap.add_argument("--checkpoint-dir", type=Path, required=True,
                    help="MossFormer2_SR_48K directory (last_best_checkpoint + _m.pt / _g.pt)")
    ap.add_argument("--audio", type=Path, required=True, help="48-kHz mono WAV (band-limited content)")
    ap.add_argument("--npy-dir", type=Path)
    ap.add_argument("--gguf", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="clip start (seconds)")
    ap.add_argument("--seconds", type=float, help="clip length (seconds); default: to the end")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)), help="layers whose outputs to capture")
    ap.add_argument("--e2e-only", action="store_true", help="only the decode-pipeline output")
    ap.add_argument("--gen-stages", action="store_true",
                    help="also capture generator intermediates (large: 64 x 48000 floats per second)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--allow-source-mismatch", action="store_true")
    args = ap.parse_args()
    if not args.npy_dir and not args.gguf:
        ap.error("pass --npy-dir and/or --gguf")

    import soundfile as sf
    import torch

    torch.set_num_threads(args.threads)
    root = resolve_upstream(args.upstream)
    ckpt_m, ckpt_g = checkpoint_paths(args.checkpoint_dir)
    sha_m, sha_g = sha256_file(ckpt_m), sha256_file(ckpt_g)
    if (sha_m, sha_g) != (CHECKPOINT_M_SHA256, CHECKPOINT_G_SHA256):
        msg = f"checkpoint sha256 {sha_m} / {sha_g} != pinned {CHECKPOINT_M_SHA256} / {CHECKPOINT_G_SHA256}"
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
    if len(audio) < cfg.n_fft:
        raise SystemExit(f"clip shorter than one {cfg.n_fft}-sample analysis window")

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    caps = run(root, args.checkpoint_dir, audio, layers, args.e2e_only, args.gen_stages, args.allow_source_mismatch)
    meta = {
        "backend": "mossformer2-sr",
        "upstream_repo": UPSTREAM_REPO,
        "upstream_rev": UPSTREAM_REV,
        "modelscope_rev": MODELSCOPE_REV,
        "checkpoint_m_sha256": sha_m,
        "checkpoint_g_sha256": sha_g,
        "sample_rate": int(sr),
        "n_samples": len(audio),
        "segmented": int(len(audio) > cfg.sampling_rate * cfg.one_time_decode_length),
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
