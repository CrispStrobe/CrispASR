"""Forward-hook helpers for capturing per-stage activations.

Reference backends in this directory dump intermediate tensors from a
PyTorch reference model so `crispasr-diff` can locate where the C++
forward path diverges. The capture set typically includes
`encoder_layer_0..N-1`, `pre_encode_output`, attention/conv submodule
outputs, etc.

The hook plumbing is the same shape for every backend: walk a list of
`(stage_name, nn.Module)` pairs, register a `forward_hook` on each,
collect the output tensors keyed by stage name, then strip the hooks
after the forward pass. This module factors that out so each backend's
`dump()` only declares *which* submodules it cares about.

Output tensors are normalised to `(T, D)` float32 row-major to match
crispasr's flat layout — that's what the diff harness expects from
`ref.compare(name, ...)`. Drop the batch dim, transpose `(B, D, T)`
returns, and slice off any padded suffix (use `T_max=` for that).

Usage:

    captured = {}
    handles = capture_modules(captured, [
        ("pre_encode_output", model.encoder.pre_encode),
        *[(f"encoder_layer_{i}", layer) for i, layer in enumerate(model.encoder.layers)],
    ])
    with torch.no_grad():
        out, _ = model.encoder(audio_signal=feats, length=feat_len)
    drop_hooks(handles)
    captures = finalize(captured, T_max=int(enc_len.item()))
    out_dict.update(captures)
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple


def _hook_factory(captured: Dict, name: str, *, first_call_only: bool = False) -> Callable:
    """Return a forward_hook that stores `output` under `captured[name]`.

    NeMo conformer modules return a Tensor or a tuple whose first element
    is the (B, T, D) hidden state. Both shapes are accepted; downstream
    `finalize()` does the (B, T, D) → (T, D) reshape.

    `first_call_only=True` skips overwrites on subsequent calls. Useful
    when a module is invoked many times inside `generate()` — e.g. the
    Qwen3-TTS talker decoder fires once for the full prefill and then
    again for every AR step with shape (1, 1, *), and we want to keep
    the prefill capture. This option does NOT replace `_iter_capture`,
    which collects ONE tensor PER call instead of skipping later calls.
    """
    def hook(_module, _inp, output):
        import torch
        if first_call_only and name in captured:
            return
        # Three output shapes seen in practice:
        #   - bare torch.Tensor (most modules);
        #   - (tuple|list) — first element is the hidden state (NeMo conformer);
        #   - HF ModelOutput / BaseModelOutputWithPast / dict-like — has
        #     `.last_hidden_state` (transformer modules in HF). We pick that
        #     attribute when present without forcing the caller to write a
        #     custom hook just to peel one layer.
        if hasattr(output, "last_hidden_state"):
            t = output.last_hidden_state
        elif isinstance(output, (tuple, list)):
            t = output[0]
        else:
            t = output
        if isinstance(t, torch.Tensor):
            captured[name] = t.detach().cpu().float()
    return hook


def capture_modules(
    captured: Dict,
    stages: Iterable[Tuple[str, object]],
    *,
    first_call_only: bool = False,
) -> List:
    """Register forward hooks on each (name, module) pair in `stages`.

    Skips entries whose module is None (so callers can pass
    `getattr(model, attr, None)` without pre-checking). Returns the list
    of handles for `drop_hooks()`. Stage names that match the diff
    harness must be lowercase ASCII (no spaces) — they end up as GGUF
    tensor names.

    `first_call_only=True` is the qwen3-tts pattern: the hooked module
    fires repeatedly inside `generate()` and only the prefill call's
    output is meaningful. Set this when the module is invoked more than
    once per `dump()` and you want to keep the first.
    """
    handles = []
    for name, module in stages:
        if module is None:
            continue
        handles.append(module.register_forward_hook(
            _hook_factory(captured, name, first_call_only=first_call_only)))
    return handles


def drop_hooks(handles: List) -> None:
    """Remove every handle from a previous `capture_modules()` call."""
    for h in handles:
        h.remove()


def finalize(captured: Dict, *, T_max: Optional[int] = None) -> Dict:
    """Convert captured PyTorch tensors to numpy `(T, D)` row-major.

    Strips the leading batch dim, accepts either `(B, T, D)` or
    `(B, D, T)` (sniffs by checking which axis is closer to the encoder
    d_model — irrelevant when D >> T which holds for NeMo encoders), and
    truncates to `T_max` rows if provided. Returns a fresh dict —
    callers typically merge it into the backend's main `out` dict.
    """
    import torch
    result = {}
    for name, t in captured.items():
        if not isinstance(t, torch.Tensor):
            continue
        if t.ndim == 3:
            arr = t[0]  # drop batch
        elif t.ndim == 2:
            arr = t
        else:
            continue
        # NeMo encoder layers return (T, D); the conv-subsampling block
        # also returns (T, D) directly. The bare `model.encoder()` final
        # output is (B, D, T) and is captured separately at the dump
        # site, not via this helper. So no dim-sniff needed here.
        if T_max is not None and arr.shape[0] >= T_max:
            arr = arr[:T_max]
        result[name] = arr.contiguous().numpy()
    return result
