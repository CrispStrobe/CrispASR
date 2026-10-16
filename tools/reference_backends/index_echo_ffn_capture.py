"""
Optional Index-Echo FFN capture helper (outside runtime graphs).

Captures original Qwen3_5MLP intermediates for a single requested forward
call.  This helper does not modify production decoding and does not require
native graph edits.

Verification requires caller-supplied provenance (transformers version,
modeling-file SHA256, Qwen3_5MLP class SHA256) which is checked against the
actual installed distribution via importlib.metadata without importing the
modeling module.  The class source is extracted with ast.get_source_segment
over the full ClassDef.

Captures are small (one MLP call per selected layer) and are written only
under the supplied storage directory; never /tmp.  No large weights are
required for the tiny control; full-model capture still needs the original
HF checkpoint and is not part of CI.
"""

import ast
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _locate_modeling_file() -> str:
    """Locate modeling file without importing the modeling module."""
    dist = importlib.metadata.distribution("transformers")
    # locate_file returns a pathlib.Path-like
    located = dist.locate_file("transformers/models/qwen3_5/modeling_qwen3_5.py")
    return str(located)


def _extract_class_source(modeling_path: str) -> str:
    """Extract the full Qwen3_5MLP ClassDef source via AST."""
    src = Path(modeling_path).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5MLP":
            seg = ast.get_source_segment(src, node)
            if seg is None:
                raise RuntimeError("ast.get_source_segment returned None for Qwen3_5MLP")
            return seg
    raise RuntimeError("Qwen3_5MLP class not found in modeling file")


def verify_source(*, expected_version: str, expected_modeling_sha256: str, expected_class_sha256: str,
                  modeling_path: Optional[str] = None) -> Dict:
    """
    Verify caller-supplied provenance against the actual installed files.
    Caller must supply all three values as keyword-only; mismatch raises.
    Returns provenance dict on success.
    """
    if modeling_path is None:
        modeling_path = _locate_modeling_file()
    actual_version = importlib.metadata.version("transformers")
    if actual_version != expected_version:
        raise RuntimeError(f"transformers version mismatch: expected {expected_version}, got {actual_version}")
    actual_file_sha = _file_sha256(modeling_path)
    if actual_file_sha != expected_modeling_sha256:
        raise RuntimeError(f"modeling file SHA mismatch: expected {expected_modeling_sha256}, got {actual_file_sha}")
    class_src = _extract_class_source(modeling_path)
    actual_class_sha = hashlib.sha256(class_src.encode()).hexdigest()
    if actual_class_sha != expected_class_sha256:
        raise RuntimeError(f"Qwen3_5MLP class SHA mismatch: expected {expected_class_sha256}, got {actual_class_sha}")
    return {
        "transformers_version": actual_version,
        "modeling_file": modeling_path,
        "modeling_sha256": actual_file_sha,
        "class_sha256": actual_class_sha,
        "verified": True,
    }


def _load_qwen3_5mlp_class(modeling_path: Optional[str] = None):
    """
    Execute the actual Qwen3_5MLP class AST from the modeling file with real
    torch and ACT2FN.  Provenance notes that only the class AST was executed.
    """
    if modeling_path is None:
        modeling_path = _locate_modeling_file()
    class_src = _extract_class_source(modeling_path)
    import transformers.activations as act_mod
    ACT2FN = act_mod.ACT2FN
    import torch.nn as nn

    class Qwen3_5Config:  # stub for annotation only
        pass

    namespace = {"torch": torch, "nn": nn, "ACT2FN": ACT2FN, "Qwen3_5Config": Qwen3_5Config}
    exec(class_src, namespace)
    Qwen3_5MLP = namespace["Qwen3_5MLP"]
    # Attach provenance as attributes for verification
    Qwen3_5MLP._crispasr_source_hashes = {
        "modeling_sha256": _file_sha256(modeling_path),
        "class_sha256": hashlib.sha256(class_src.encode()).hexdigest(),
    }
    return Qwen3_5MLP, {"method": "ast_exec_actual_class_only", "modeling_path": modeling_path,
                        "class_sha256": hashlib.sha256(class_src.encode()).hexdigest()}


class FFNCapture:
    """
    Context manager that captures MLP input, gate/up outputs, down input/output
    and MLP output for exactly one requested forward call.

    Provenance values are keyword-only and have no defaults; they must be
    supplied by the caller and are verified before any hook is installed.

    On success, `store` contains owned copies for each selected layer:
      blk{idx}_mlp_in, blk{idx}_gate_out, blk{idx}_up_out,
      blk{idx}_down_in, blk{idx}_down_out, blk{idx}_mlp_out

    All arrays are owned copies and are frozen after the selected call.
    """

    def __init__(self, llm, layers: List[int], *, expected_version: str, expected_modeling_sha256: str,
                 expected_class_sha256: str, call_index: int = 0, out_dir: Optional[Path] = None,
                 last_token_only: bool = False):
        # Validate provenance args are present (keyword-only ensures this)
        if expected_version is None or expected_modeling_sha256 is None or expected_class_sha256 is None:
            raise ValueError("expected_version, expected_modeling_sha256 and expected_class_sha256 are required")
        if not isinstance(layers, (list, tuple)):
            raise TypeError("layers must be list/tuple of ints")

        self.llm = llm
        self.layers = list(layers)
        # No int coercion; store exactly as given for validation in __enter__
        self.call_index = call_index
        self.expected_version = expected_version
        self.expected_modeling_sha256 = expected_modeling_sha256
        self.expected_class_sha256 = expected_class_sha256
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.last_token_only = bool(last_token_only)
        self.handles: List = []
        self.store: Dict[str, np.ndarray] = {}
        self.provenance: Optional[Dict] = None
        self._call_counts: Dict[int, int] = {}
        self._entered = False
        self._exited = False
        self._finalized = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("FFNCapture already entered (reuse not allowed)")
        self._entered = True
        # 0. Validate call_index and layers strictly before any other work
        if isinstance(self.call_index, bool) or not isinstance(self.call_index, int):
            raise TypeError("call_index must be int (not bool/float)")
        if self.call_index < 0:
            raise ValueError("call_index must be >=0")
        if len(self.layers) == 0:
            raise ValueError("layers must be non-empty")
        if len(self.layers) != len(set(self.layers)):
            raise ValueError(f"duplicate layer indices in {self.layers}")
        for idx in self.layers:
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise TypeError(f"layer index must be int, got {type(idx)}")
            if idx < 0:
                raise ValueError(f"layer index must be >=0, got {idx}")
        # Initialize call counts after validation
        self._call_counts = {idx: 0 for idx in self.layers}
        # 1. Verify source provenance before touching the model
        self.provenance = verify_source(
            expected_version=self.expected_version,
            expected_modeling_sha256=self.expected_modeling_sha256,
            expected_class_sha256=self.expected_class_sha256,
        )
        # 2. Validate ALL layers before installing any hook
        layers_obj = getattr(getattr(self.llm, "model", self.llm), "layers", None)
        if layers_obj is None:
            raise RuntimeError("llm has no .model.layers or .layers")
        n_layers = len(layers_obj)
        for idx in self.layers:
            if idx >= n_layers:
                raise ValueError(f"invalid layer index {idx}, has {n_layers} layers")
            mlp = getattr(layers_obj[idx], "mlp", None)
            if mlp is None:
                raise AttributeError(f"layer {idx} has no .mlp")
            for attr in ("gate_proj", "up_proj", "down_proj", "act_fn"):
                if not hasattr(mlp, attr):
                    raise AttributeError(f"layer {idx} mlp missing {attr}")
            # Verify supplied model's MLP class corresponds to expected source:
            mlp_type = type(mlp)
            if mlp_type.__name__ != "Qwen3_5MLP":
                raise RuntimeError(f"model MLP class hash mismatch for layer {idx}: unexpected class {mlp_type.__name__}")
            if "_crispasr_source_hashes" in mlp_type.__dict__:
                # AST-loaded test class: verify its hashes match expected
                h = mlp_type._crispasr_source_hashes
                if h.get("modeling_sha256") != self.expected_modeling_sha256:
                    raise RuntimeError(f"model MLP modeling file hash mismatch for layer {idx}: expected {self.expected_modeling_sha256}, got {h.get('modeling_sha256')}")
                if h.get("class_sha256") != self.expected_class_sha256:
                    raise RuntimeError(f"model MLP class hash mismatch for layer {idx}: expected {self.expected_class_sha256}, got {h.get('class_sha256')}")
            else:
                # Direct-import: verify both file and class hashes without swallowing
                src_file = inspect.getsourcefile(mlp_type)
                if src_file is None:
                    raise RuntimeError(f"cannot determine source file for MLP class at layer {idx}")
                # Verify file hash matches expected modeling file hash
                actual_file_sha = _file_sha256(src_file)
                if actual_file_sha != self.expected_modeling_sha256:
                    raise RuntimeError(f"model MLP modeling file hash mismatch for layer {idx}: expected {self.expected_modeling_sha256}, got {actual_file_sha}")
                # Verify class source hash
                class_src = _extract_class_source(src_file)
                actual_class_sha = hashlib.sha256(class_src.encode()).hexdigest()
                if actual_class_sha != self.expected_class_sha256:
                    raise RuntimeError(f"model MLP class hash mismatch for layer {idx}: expected {self.expected_class_sha256}, got {actual_class_sha}")
        # Capture act_fn receipt for provenance (inspect actual selected mlp)
        try:
            first_mlp = layers_obj[self.layers[0]].mlp
            act_fn = getattr(first_mlp, "act_fn", None)
            act_name = getattr(act_fn, "__name__", str(type(act_fn))) if act_fn is not None else "unknown"
            self.provenance["mlp_act_fn"] = act_name
            # Also record model config hidden_act if available
            for idx in self.layers:
                cfg = getattr(getattr(layers_obj[idx].mlp, "config", None), "hidden_act", None)
                if cfg is not None:
                    self.provenance[f"blk{idx}_hidden_act"] = cfg
                    break
        except Exception:
            pass

        # Own a new output directory; never replace a previous proof.
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=False)
        # 3. Install hooks only after full validation
        try:
            for idx in self.layers:
                mlp = layers_obj[idx].mlp
                self.handles.append(mlp.register_forward_hook(self._make_mlp_hook(idx)))
                self.handles.append(mlp.gate_proj.register_forward_hook(self._make_linear_hook(idx, "gate")))
                self.handles.append(mlp.up_proj.register_forward_hook(self._make_linear_hook(idx, "up")))
                self.handles.append(mlp.down_proj.register_forward_hook(self._make_linear_hook(idx, "down")))
        except Exception:
            self._cleanup_handles()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._exited:
            raise RuntimeError("FFNCapture already exited")
        self._exited = True
        self._cleanup_handles()
        if exc_type is not None:
            return False
        try:
            self.finalize()
        except Exception:
            raise
        return False

    def _cleanup_handles(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _make_mlp_hook(self, idx: int):
        def hook(module, inputs, output):
            cnt = self._call_counts[idx]
            if cnt != self.call_index:
                self._call_counts[idx] = cnt + 1
                return
            inp = inputs[0] if isinstance(inputs, tuple) else inputs
            if isinstance(inp, tuple):
                inp = inp[0]
            out = output[0] if isinstance(output, tuple) else output
            # Owned copies; last_token_only handling without modifying originals
            arr_in = self._owned(inp)
            arr_out = self._owned(out)
            # Do not mutate original tensors; just capture
            self.store[f"blk{idx}_mlp_in"] = arr_in
            self.store[f"blk{idx}_mlp_out"] = arr_out
            self._save(f"blk{idx}_mlp_in", arr_in)
            self._save(f"blk{idx}_mlp_out", arr_out)
            self._call_counts[idx] = cnt + 1
        return hook

    def _make_linear_hook(self, idx: int, kind: str):
        def hook(module, inputs, output):
            cnt = self._call_counts.get(idx, 0)
            if cnt != self.call_index:
                return
            inp = inputs[0] if isinstance(inputs, tuple) else inputs
            if isinstance(inp, tuple):
                inp = inp[0]
            out = output[0] if isinstance(output, tuple) else output
            if kind == "down":
                arr_in = self._owned(inp)
                arr_out = self._owned(out)
                self.store[f"blk{idx}_down_in"] = arr_in
                self.store[f"blk{idx}_down_out"] = arr_out
                self._save(f"blk{idx}_down_in", arr_in)
                self._save(f"blk{idx}_down_out", arr_out)
            else:
                arr_out = self._owned(out)
                self.store[f"blk{idx}_{kind}_out"] = arr_out
                self._save(f"blk{idx}_{kind}_out", arr_out)
        return hook

    def _owned(self, tensor):
        value = tensor.detach()
        # Slice before conversion/transfer; a final-token diagnostic must not
        # copy the full prefill matrix from GPU to CPU for each projection.
        if self.last_token_only and value.ndim == 3:
            value = value[:, -1:, :]
        return value.float().cpu().numpy().copy()

    def _save(self, name, array):
        if self.out_dir is not None:
            with (self.out_dir / (name + ".npy")).open("xb") as stream:
                np.save(stream, array, allow_pickle=False)

    def finalize(self):
        if self._finalized:
            return self.store
        missing = []
        for idx in self.layers:
            for key in (f"blk{idx}_mlp_in", f"blk{idx}_gate_out", f"blk{idx}_up_out",
                        f"blk{idx}_down_in", f"blk{idx}_down_out", f"blk{idx}_mlp_out"):
                if key not in self.store:
                    missing.append(key)
        if missing:
            raise RuntimeError(f"incomplete capture for call_index {self.call_index}, missing {missing}; "
                               f"captured keys: {sorted(self.store.keys())}")
        if self.out_dir is not None:
            with (self.out_dir / "provenance.json").open("x") as stream:
                json.dump(self.provenance, stream, indent=2)
                stream.write("\n")
        self._finalized = True
        return self.store

    def captured(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.store.items()}
