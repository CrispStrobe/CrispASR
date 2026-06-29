"""dots.tts reference dump backend for the crispasr-diff harness.

Driven by ``tools/dump_reference.py --backend dots-tts``; the harness
serializes the returned dict to a GGUF the C++ ``crispasr-diff dots-tts``
branch reads.

MEMORY DISCIPLINE (see crispasr-crispembed-dev.md): never load the full
dots.tts model in PyTorch — that is ~11 GB in float32 and OOM-crashes a
16 GB machine. Each stage instead loads ONLY the sub-model under test via
``safe_open`` (lazy, per-tensor bf16->f32) and runs it in isolation:

  PatchEncoder (VAESemanticEncoder, ~128 M / ~0.5 GB f32) — decode_patch on
  a fixed, seeded "patch 0" latent (zero conv tail, empty KV). This is the
  cleanest isolation point and the critical stage to validate first.

Stages dumped (all f32; GGUF column-major == numpy row-major flat):
  penc_in_patch0        (patch_size, latent_dim)   latent fed to decode_patch
  penc_positions_patch0 (out_ds_rate,)             decode positions, as f32
  penc_conv_tail_patch0 (latent_dim, left_pad)     conv tail (zeros for patch 0)
  penc_out_patch0       (1, out_dim=1536)          decode_patch llm_embedding

The ``--model-dir`` is the source safetensors snapshot
(rednote-hilab/dots.tts-soar). ``--audio`` is ignored (dots.tts is TTS).

Env:
  DOTS_SEED   deterministic latent seed (default 0).

Future stages (DiT velocity, BigVGAN) follow the same isolated, lazy pattern.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "penc_in_patch0",
    "penc_positions_patch0",
    "penc_conv_tail_patch0",
    "penc_out_patch0",
    "dit_x",
    "dit_t",
    "dit_gcond",
    "dit_vel",
    "fm_input_seq",
    "fm_cfg_seq",
    "fm_mask",
    "fm_pos",
    "fm_noise",
    "fm_latent_out",
    "fm_meta",
    "voc_latent_in",
    "voc_post_proj",
    "voc_mi_out",
    "voc_conv_pre",
    "voc_ups0",
    "voc_stage0",
    "voc_audio",
    "act_in",
    "act_out",
    "act_alpha",
    "act_beta",
    "act_up_filter",
    "act_down_filter",
]


def required_packages() -> list[str]:
    return ["dots_tts @ git+https://github.com/rednote-hilab/dots.tts.git"]


class _Cfg(dict):
    """dict with attribute access + .get + .to_dict (duck-types ConfigBase)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(k) from e

    def to_dict(self):
        return dict(self)


def _install_import_stubs() -> None:
    """Stub heavy text-frontend deps (lingua / WeTextProcessing-pynini) that
    dots_tts.utils.text imports at module load. We only import the backbone
    module and never run text normalization, so inert stubs suffice."""

    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return self

        def __getattr__(self, _n):
            return _Dummy()

    def _mod(name, **attrs):
        if name in sys.modules:
            return
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    try:
        import lingua  # noqa: F401
    except ImportError:
        _mod("lingua", Language=_Dummy, LanguageDetectorBuilder=_Dummy)
    try:
        import tn  # noqa: F401
    except ImportError:
        _mod("tn")
        _mod("tn.chinese")
        _mod("tn.chinese.normalizer", Normalizer=_Dummy)
        _mod("tn.english")
        _mod("tn.english.normalizer", Normalizer=_Dummy)


def _npf(t) -> np.ndarray:
    return np.ascontiguousarray(t.detach().cpu().float().numpy())


def _dump_patch_encoder(model_dir: Path, seed: int) -> Dict[str, np.ndarray]:
    """Isolated PatchEncoder (VAESemanticEncoder) decode_patch reference."""
    import torch
    from safetensors import safe_open
    from dots_tts.modules.backbone.semantic_encoder import VAESemanticEncoder

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)

    cj = json.loads((model_dir / "config.json").read_text())
    latent_dim = int(cj["latent_dim"])
    patch_size = int(cj["patch_size"])
    top_cfg = _Cfg({
        "patch_size": patch_size,
        "latent_dim": latent_dim,
        "PatchEncoder": _Cfg(cj["PatchEncoder"]),
    })

    st_path = model_dir / "model.safetensors"
    with safe_open(str(st_path), framework="pt") as f:
        out_dim = int(list(f.get_slice("patch_encoder.out_proj.weight").get_shape())[0])
        penc_sd = {
            k[len("patch_encoder."):]: f.get_tensor(k).float()
            for k in f.keys() if k.startswith("patch_encoder.")
        }
    print(f"[dots-ref/penc] out_dim={out_dim} penc tensors={len(penc_sd)}")

    enc = VAESemanticEncoder(in_dim=latent_dim, out_dim=out_dim, config=top_cfg)
    missing, unexpected = enc.load_state_dict(penc_sd, strict=False)
    if missing:
        print(f"[dots-ref/penc] WARNING missing: {missing}")
    if unexpected:
        print(f"[dots-ref/penc] WARNING unexpected: {unexpected}")
    enc.eval().float()

    g = torch.Generator().manual_seed(seed)
    latent_patch = torch.randn(1, patch_size, latent_dim, generator=g, dtype=torch.float32)
    state = enc.init_decode_state(
        max_audio_patch_count=4, batch_size=1,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    positions = torch.arange(enc.out_ds_rate, dtype=torch.long)  # patch 0: [0, 1]
    embedding, _conv_tail = enc.decode_patch(
        latent_patch, state.conv_tail, state.layer_caches, positions,
    )

    return {
        "penc_in_patch0": _npf(latent_patch[0]),               # (patch_size, latent_dim)
        "penc_positions_patch0": positions.float().numpy(),    # (out_ds_rate,)
        "penc_conv_tail_patch0": _npf(state.conv_tail[0]),     # (latent_dim, left_pad)
        "penc_out_patch0": _npf(embedding[0]),                 # (1, out_dim)
    }


def _dump_dit(model_dir: Path, seed: int) -> Dict[str, np.ndarray]:
    """Isolated DiT (velocity_field_predictor) forward reference.

    Loads ONLY the ~256 M DiT and runs one velocity prediction on a seeded
    sequence (attn_mask=None => bidirectional, internal pos=arange, matching
    the C++ dit forward), with a seeded global condition g_cond.
    """
    import torch
    from safetensors import safe_open
    from dots_tts.modules.backbone.dit import DiT

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)

    cj = json.loads((model_dir / "config.json").read_text())
    latent_dim = int(cj["latent_dim"])
    dit_cfg = _Cfg(cj["DiT"])
    in_dim = int(dit_cfg["hidden_size"])  # fm_hidden_size

    dit = DiT(in_dim=in_dim, out_dim=latent_dim, transformer_config=dit_cfg,
              mode="flow_matching")
    st_path = model_dir / "model.safetensors"
    with safe_open(str(st_path), framework="pt") as f:
        sd = {
            k[len("velocity_field_predictor."):]: f.get_tensor(k).float()
            for k in f.keys() if k.startswith("velocity_field_predictor.")
        }
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    if missing:
        print(f"[dots-ref/dit] WARNING missing: {missing}")
    if unexpected:
        print(f"[dots-ref/dit] WARNING unexpected: {unexpected}")
    dit.eval().float()
    print(f"[dots-ref/dit] in_dim={in_dim} out_dim={latent_dim} tensors={len(sd)}")

    g = torch.Generator().manual_seed(seed)
    T = 6
    x = torch.randn(1, T, in_dim, generator=g, dtype=torch.float32)
    timesteps = torch.tensor([0.3], dtype=torch.float32)
    g_cond = torch.randn(1, in_dim, generator=g, dtype=torch.float32)

    if os.environ.get("DOTS_DIT_DEBUG"):
        caps = {}
        dit.input_layer.register_forward_hook(
            lambda m, i, o: caps.__setitem__("x0", o.detach()))
        dit.blocks[0].register_forward_hook(
            lambda m, i, o: caps.__setitem__("b0", o.detach()))
        dit.time_embedder.register_forward_hook(
            lambda m, i, o: caps.__setitem__("temb", o.detach()))
        _ = dit(x=x, timesteps=timesteps, g_cond=g_cond, attn_mask=None)
        c_dbg = caps["temb"] + g_cond
        for nm, t in [("temb", caps["temb"]), ("c", c_dbg),
                      ("x0", caps["x0"]), ("b0", caps["b0"])]:
            v = t.reshape(-1)[:6].tolist()
            print(f"[dit-dbg] {nm}: " + " ".join(f"{x:+.4f}" for x in v))

    vel = dit(x=x, timesteps=timesteps, g_cond=g_cond, attn_mask=None)

    return {
        "dit_x": _npf(x[0]),            # (T, in_dim)
        "dit_t": _npf(timesteps),       # (1,)
        "dit_gcond": _npf(g_cond[0]),   # (in_dim,)
        "dit_vel": _npf(vel[0]),        # (T, latent_dim)
    }


def _dump_flowmatch(model_dir: Path, seed: int) -> Dict[str, np.ndarray]:
    """Isolated flow-matching ODE driver reference (core.fm_solver_step + Euler).

    Builds a patch-1-style FM sequence [h0, l0(×4), h1] + 4 noise slots so the
    block-causal attn_mask AND non-arange pos_ids are exercised (not just the
    trivial patch-0 case). Loads ONLY the DiT + the 3 projections, runs a
    deterministic Euler loop with seeded noise feeding core.fm_solver_step.
    """
    import torch
    from safetensors import safe_open
    from dots_tts.modules.backbone.dit import DiT

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)

    cj = json.loads((model_dir / "config.json").read_text())
    latent_dim = int(cj["latent_dim"])
    patch_size = int(cj["patch_size"])
    dit_cfg = _Cfg(cj["DiT"])
    fm_hidden = int(dit_cfg["hidden_size"])

    st_path = model_dir / "model.safetensors"
    with safe_open(str(st_path), framework="pt") as f:
        llm_dim = int(list(f.get_slice("hidden_proj.weight").get_shape())[1])
        dit_sd = {k[len("velocity_field_predictor."):]: f.get_tensor(k).float()
                  for k in f.keys() if k.startswith("velocity_field_predictor.")}
        proj = {}
        for nm in ("hidden_proj", "latent_proj", "coordinate_proj"):
            proj[nm + "_w"] = f.get_tensor(nm + ".weight").float()
            proj[nm + "_b"] = f.get_tensor(nm + ".bias").float()

    dit = DiT(in_dim=fm_hidden, out_dim=latent_dim, transformer_config=dit_cfg,
              mode="flow_matching")
    dit.load_state_dict(dit_sd, strict=False)
    dit.eval().float()

    def lin(name, x):  # x: (..., in) -> (..., out)
        return torch.nn.functional.linear(x, proj[name + "_w"], proj[name + "_b"])

    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)
    h0 = rn(1, 1, llm_dim)
    h1 = rn(1, 1, llm_dim)
    lat_hist = rn(1, patch_size, latent_dim)          # decoded latent-0 history
    noise = rn(1, patch_size, latent_dim)             # ODE init coordinate
    g_cond = torch.zeros(1, fm_hidden)                # text-only -> null g_cond

    fm_seq_len = 1 + patch_size + 1                   # [h0, l0×4, h1]
    total_len = fm_seq_len + patch_size               # + 4 noise slots
    latent_start = fm_seq_len

    input_seq = torch.zeros(1, total_len, fm_hidden)
    cfg_seq = torch.zeros(1, total_len, fm_hidden)
    h0p, h1p = lin("hidden_proj", h0), lin("hidden_proj", h1)
    hnull = lin("hidden_proj", torch.zeros(1, 1, llm_dim))
    latp = lin("latent_proj", lat_hist)               # (1, patch_size, fm_hidden)
    input_seq[:, 0:1] = h0p
    input_seq[:, 1:1 + patch_size] = latp
    input_seq[:, 1 + patch_size:fm_seq_len] = h1p
    cfg_seq[:, 0:1] = hnull
    cfg_seq[:, 1:1 + patch_size] = latp
    cfg_seq[:, 1 + patch_size:fm_seq_len] = hnull

    # _build_fm_attn_mask (bool, True=attend) for fm_seq_len, latent_start.
    hidden_patch_size = 1
    mask = torch.zeros(1, total_len, total_len, dtype=torch.bool)
    block_start = fm_seq_len - hidden_patch_size
    if block_start > 0:
        causal = torch.ones(block_start, block_start, dtype=torch.bool).triu(1).logical_not()
        mask[:, :block_start, :block_start] = causal
    mask[:, block_start:fm_seq_len, :fm_seq_len] = True
    mask[:, block_start:fm_seq_len, latent_start:] = True
    mask[:, latent_start:, :fm_seq_len] = True
    mask[:, latent_start:, latent_start:] = True

    # _build_fm_pos_ids
    pos = torch.zeros(1, total_len, dtype=torch.float32)
    pos[:, :fm_seq_len] = torch.arange(fm_seq_len, dtype=torch.float32)
    pos[:, latent_start:] = torch.arange(fm_seq_len, fm_seq_len + patch_size, dtype=torch.float32)

    num_steps = 4
    cfg_scale = 3.0

    def solver(t, z):
        zc = lin("coordinate_proj", z)                # (1, patch, fm_hidden)
        sc = input_seq.clone(); sc[:, latent_start:] = zc
        su = cfg_seq.clone();   su[:, latent_start:] = zc
        z_z = torch.cat([sc, su], 0)                  # (2, total_len, fm_hidden)
        t_t = t.reshape(1).repeat(2)
        gg = torch.cat([g_cond, torch.zeros_like(g_cond)], 0)
        vt = dit(x=z_z, timesteps=t_t, attn_mask=mask, pos_ids=pos, g_cond=gg)
        vt = vt[:, latent_start:]
        vc, vu = vt[:1], vt[1:]
        return vc + cfg_scale * (vc - vu)

    z = noise.clone()
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = torch.tensor(float(k) * dt, dtype=torch.float32)
        z = z + dt * solver(t, z)

    # Additive mask [q][k] row-major (0 attend, -inf block) for the C++ side.
    add_mask = torch.where(mask[0], torch.zeros(()), torch.full((), float("-inf")))
    print(f"[dots-ref/fm] total_len={total_len} latent_start={latent_start} "
          f"steps={num_steps} cfg={cfg_scale}")
    return {
        "fm_input_seq": _npf(input_seq[0]),    # (total_len, fm_hidden)
        "fm_cfg_seq": _npf(cfg_seq[0]),        # (total_len, fm_hidden)
        "fm_mask": _npf(add_mask),             # (total_len, total_len) [q][k]
        "fm_pos": _npf(pos[0]),                # (total_len,)
        "fm_noise": _npf(noise[0]),            # (patch_size, latent_dim)
        "fm_latent_out": _npf(z[0]),           # (patch_size, latent_dim)
        "fm_meta": np.array([total_len, latent_start, num_steps, cfg_scale],
                            dtype=np.float32),
    }


def _dump_vocoder(model_dir: Path, seed: int) -> Dict[str, np.ndarray]:
    """Isolated BigVGAN vocoder reference (AudioVAE.inference_from_latents).

    Loads only the vocoder, removes weight-norm (checkpoint decoder is folded),
    runs the do_sample=False TTS path on a short seeded latent, and captures
    intermediate stages so the non-anti-aliased parts (post_proj, dec_mi LSTM,
    conv_pre) can be validated independently of the alias-free resampling.
    """
    import torch
    from safetensors import safe_open
    from dots_tts.modules.vocoder.bigvgan import AudioVAE

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)

    cj = json.loads((model_dir / "config.json").read_text())
    vcfg = _Cfg(cj["vocoder"])
    latent_dim = int(vcfg["latent_dim"])

    vae = AudioVAE(vcfg)
    vae.remove_weight_norm()  # checkpoint decoder weights are weight-norm-folded
    st_path = model_dir / "vocoder.safetensors"
    with safe_open(str(st_path), framework="pt") as f:
        sd = {(k[len("vocoder."):] if k.startswith("vocoder.") else k): f.get_tensor(k).float()
              for k in f.keys()}
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    # audio_encoder is unused here; only decoder/mi/post_proj must be present.
    crit = [m for m in missing if m.startswith(("decoder.", "dec_mi_layer.", "post_proj."))]
    if crit:
        print(f"[dots-ref/voc] WARNING missing critical: {crit[:6]}")
    vae.eval().float()

    g = torch.Generator().manual_seed(seed)
    n_frames = 4
    latent = torch.randn(1, n_frames, latent_dim, generator=g, dtype=torch.float32)
    x_in = latent.transpose(1, 2).contiguous()  # (1, latent_dim, n_frames)

    caps: Dict[str, torch.Tensor] = {}
    vae.post_proj.register_forward_hook(lambda m, i, o: caps.__setitem__("post", o.detach()))
    vae.dec_mi_layer.register_forward_hook(lambda m, i, o: caps.__setitem__("mi", o.detach()))
    vae.decoder.conv_pre.register_forward_hook(lambda m, i, o: caps.__setitem__("convpre", o.detach()))
    # Bisection taps: ups[0] output (after first upsample) and ups[1] input
    # (== stage-0 output, after the 3 averaged resblocks).
    vae.decoder.ups[0][0].register_forward_hook(lambda m, i, o: caps.__setitem__("ups0", o.detach()))
    vae.decoder.ups[1][0].register_forward_pre_hook(lambda m, i: caps.__setitem__("stage0", i[0].detach()))

    audio = vae.inference_from_latents(x_in, do_sample=False)  # (1, 1, n_samples)

    print(f"[dots-ref/voc] n_frames={n_frames} audio={tuple(audio.shape)} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    # Dump sequence-first (T, C) to match the penc/dit/fm GGUF convention
    # (numpy (T,C) -> ggml ne=(C,T), read directly as channel-first in C++).
    return {
        "voc_latent_in": _npf(x_in[0].transpose(0, 1)),   # (n_frames, latent_dim)
        "voc_post_proj": _npf(caps["post"][0].transpose(0, 1)),    # (n_frames, latent_dim)
        "voc_mi_out": _npf(caps["mi"][0]),                # already (n_frames, latent_dim)
        "voc_conv_pre": _npf(caps["convpre"][0].transpose(0, 1)),  # (n_frames, initial_ch)
        "voc_ups0": _npf(caps["ups0"][0].transpose(0, 1)),         # (T0, C0) after first upsample
        "voc_stage0": _npf(caps["stage0"][0].transpose(0, 1)),     # (T0, C0) after stage-0 resblocks
        "voc_audio": _npf(audio[0, 0]),                   # (n_samples,)
    }


def _dump_act(model_dir: Path, seed: int) -> Dict[str, np.ndarray]:
    """Isolated single alias-free Activation1d reference (resblocks.0.activations.0).

    Builds just that Activation1d (upsample 2x -> SnakeBeta -> downsample 2x with
    the checkpoint's kaiser-sinc filters) and runs it on a seeded input, so the
    C++ alias-free op can be debugged on its own. Also dumps the alpha/beta and
    up/down filters so the C++ side reads identical values.
    """
    import torch
    from safetensors import safe_open
    from dots_tts.modules.vocoder.alias_free_act import Activation1d, SnakeBeta

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)

    pfx = "decoder.resblocks.0.activations.0."
    with safe_open(str(model_dir / "vocoder.safetensors"), framework="pt") as f:
        alpha = f.get_tensor(pfx + "act.alpha").float()
        beta = f.get_tensor(pfx + "act.beta").float()
        up_f = f.get_tensor(pfx + "upsample.filter").float()
        dn_f = f.get_tensor(pfx + "downsample.lowpass.filter").float()
    C = int(alpha.shape[0])

    snake = SnakeBeta(C, alpha_logscale=True)
    snake.alpha.copy_(alpha)
    snake.beta.copy_(beta)
    act = Activation1d(activation=snake, causal=True, fixed_filter=True)
    act.upsample.filter.copy_(up_f)
    act.downsample.lowpass.filter.copy_(dn_f)
    act.eval()

    g = torch.Generator().manual_seed(seed)
    T = 8
    x = torch.randn(1, C, T, generator=g, dtype=torch.float32)
    y = act(x)
    print(f"[dots-ref/act] C={C} T={T} -> out T={y.shape[-1]}")
    return {
        "act_in": _npf(x[0].transpose(0, 1)),    # (T, C)
        "act_out": _npf(y[0].transpose(0, 1)),   # (T, C)
        "act_alpha": _npf(alpha),                # (C,)
        "act_beta": _npf(beta),                  # (C,)
        "act_up_filter": _npf(up_f.reshape(-1)),    # (12,)
        "act_down_filter": _npf(dn_f.reshape(-1)),  # (12,)
    }


def dump(
    model_dir: "Path | str | None" = None,
    audio: np.ndarray | None = None,
    stages: Set[str] | None = None,
    max_new_tokens: int = 20,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Return per-stage dots.tts reference activations (isolated, lazy load).

    The harness serializes the returned dict to GGUF. ``audio`` is ignored.
    """
    if stages is None:
        stages = set(DEFAULT_STAGES)
    if model_dir is None:
        model_dir = os.environ.get("DOTS_MODEL_DIR",
                                   "/Volumes/backups/ai/crispasr-gguf/dots-tts-soar-src")
    model_dir = Path(model_dir)
    seed = int(os.environ.get("DOTS_SEED", "0"))

    _install_import_stubs()

    results: Dict[str, np.ndarray] = {}

    if any(s.startswith("penc_") for s in stages):
        results.update(_dump_patch_encoder(model_dir, seed))
    if any(s.startswith("dit_") for s in stages):
        results.update(_dump_dit(model_dir, seed))
    if any(s.startswith("fm_") for s in stages):
        results.update(_dump_flowmatch(model_dir, seed))
    if any(s.startswith("voc_") for s in stages):
        results.update(_dump_vocoder(model_dir, seed))
    if any(s.startswith("act_") for s in stages):
        results.update(_dump_act(model_dir, seed))

    # Keep only requested stages (plus the always-present penc/dit/fm/voc tensors).
    out = {k: np.ascontiguousarray(v.astype(np.float32))
           for k, v in results.items()
           if k in stages or k.startswith("penc_") or k.startswith("dit_")
           or k.startswith("fm_") or k.startswith("voc_") or k.startswith("act_")}
    print(f"[dots-ref] returning {len(out)} tensors: {sorted(out)}")
    return out
