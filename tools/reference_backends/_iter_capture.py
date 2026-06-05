"""Per-iteration capture helpers for AR / diffusion samplers.

Companion to `_hooks.py`. That module hooks `nn.Module.forward` and
captures one tensor per stage — fine for prefill-only diffs (qwen3-tts
talker prefill, parakeet encoder layers, etc.).

Per-frame diffs against an AR generator need to capture state INSIDE
sampler loops that aren't exposed as `nn.Module.forward` calls:

  - VibeVoice-Realtime's `sample_speech_tokens` runs DPM-Solver++
    iterations internally; per-frame `pos_cond`, `neg_cond`, init
    `noise`, `v_cfg` at step 0, and final `latent` are visible only
    inside that method.
  - Qwen3-TTS-talker's greedy AR loop runs `forward(input_ids)` once
    per step inside `generate()`; per-step `talker_logits` and
    `codec_head` outputs are similarly invisible to a single
    forward_hook.

The shared shape is: monkey-patch one entry-point method, append a
tensor per iteration to a `{stage: [tensor, ...]}` dict, restore the
original method when done. This module factors that out.

Usage:

    captured = {}
    handles = []

    @torch.no_grad()
    def hooked_sample(condition, neg_condition, cfg_scale=3.0):
        append(captured, "pos_cond", condition[0])
        append(captured, "neg_cond", neg_condition[0])
        ...
        return final
    handles.append(patch_method(model, "sample_speech_tokens", lambda _orig: hooked_sample))

    # nn.Module hooks via _hooks.py still work side-by-side:
    handles.extend(_hooks.capture_modules(per_module_captured, [("acoustic_embed", model.connector)]))

    out = model.generate(...)

    drop_handles(handles)
    dump_perframe_dir(captured, out_dir)

Filename convention is `perframe_<stage>_f<NNN>.bin` (float32 little-endian),
matching the C++ runtime `VIBEVOICE_TTS_DUMP_PERFRAME=1` output and what
`tools/diff_vibevoice_tts.py` expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np


def append(captured: Dict[str, List[Any]], stage: str, tensor: Any) -> None:
    """Append one frame's tensor to `captured[stage]`.

    Detaches, moves to CPU, casts to float32, and clones — matches the
    `_hooks.py` convention so callers can mix-and-match capture sources.
    Pass a plain `np.ndarray` to skip the torch path (useful for
    backends that already produced numpy).
    """
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().float().clone()
    except ImportError:
        pass
    captured.setdefault(stage, []).append(tensor)


def patch_method(obj: Any, name: str, wrap: Callable[[Callable], Callable]) -> Callable[[], None]:
    """Replace `obj.name` with `wrap(orig_method)`. Returns a restore()
    closure that puts the original method back. Bind both:

        restores = []
        restores.append(patch_method(model, "sample_speech_tokens", lambda orig: hooked_sample))
        ...
        for r in restores: r()
    """
    orig = getattr(obj, name)
    setattr(obj, name, wrap(orig))

    def _restore() -> None:
        setattr(obj, name, orig)

    return _restore


def drop_handles(handles: List[Any]) -> None:
    """Run every restore closure / `RemovableHandle` produced by hook
    helpers. Mixed lists from `patch_method` and `_hooks.capture_modules`
    are both supported (the former returns callables, the latter returns
    objects with `.remove()`)."""
    for h in handles:
        if callable(h):
            h()
        elif hasattr(h, "remove"):
            h.remove()


def stack_frames(captured: Dict[str, List[Any]], stage: str) -> np.ndarray | None:
    """Stack per-frame tensors into a `(N, *)` numpy array. Returns
    None if the stage is missing or empty."""
    frames = captured.get(stage)
    if not frames:
        return None
    try:
        import torch
        if isinstance(frames[0], torch.Tensor):
            return torch.stack(frames).numpy()
    except ImportError:
        pass
    return np.stack([np.asarray(f) for f in frames])


def dump_perframe_dir(captured: Dict[str, List[Any]], out_dir: str | Path,
                      *, prefix: str = "perframe", dtype=np.float32) -> int:
    """Write each captured frame to `<out_dir>/<prefix>_<stage>_f<NNN>.bin`.

    File format: flat little-endian `dtype` (default float32) — same as
    the C++ runtime's `vibevoice_dump_f32` writer. Returns the total
    number of frames written.

    Ints (e.g. token-id captures) should be written by the caller via
    `np.int32` directly; this helper assumes the float-tensor case which
    covers all current diff-harness needs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for stage, frames in captured.items():
        for fi, t in enumerate(frames):
            if hasattr(t, "numpy"):
                arr = t.numpy()
            else:
                arr = np.asarray(t)
            arr = np.ascontiguousarray(arr, dtype=dtype)
            arr.tofile(str(out_dir / f"{prefix}_{stage}_f{fi:03d}.bin"))
            n_written += 1
    # Convenience: if a single stage has a "noise" sequence, also stack
    # and write `noise.bin` so the C++ side can replay via
    # VIBEVOICE_TTS_NOISE / similar env vars.
    if "noise" in captured and captured["noise"]:
        stacked = stack_frames(captured, "noise")
        if stacked is not None:
            np.ascontiguousarray(stacked, dtype=dtype).tofile(str(out_dir / "noise.bin"))
    return n_written
