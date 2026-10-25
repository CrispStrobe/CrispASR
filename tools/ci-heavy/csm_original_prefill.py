#!/usr/bin/env python3
"""Hosted-only F32 prefill from the pinned, unmodified original HF CSM model.

Install torch==2.6.0 numpy safetensors accelerate and transformers from
798f948e88fd0b93fc515ec6b96e0503b78ad6ba. No hand-written model forward,
sampling, codec inference or voice recording is used. Weights stay in scratch.
"""

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

import nightly_error_ab as ab
import csm_original_rope_reference as original


ROOT = Path(__file__).resolve().parents[2]
TOKENS = [128000, 58, 15, 60, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 128001]
TEXT = "The quick brown fox jumps over the lazy dog"


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true" or not os.environ.get("RUNNER_TEMP"):
        raise RuntimeError("full original model reference must run on hosted Actions")
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "csm-original-prefill"
    scratch.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"complete": False, "scope": __doc__, "cases": {}, "files": {},
               "source_revision": original.REVISION, "config_revision": original.CONFIG_REVISION,
               "prompt_tokens": TOKENS, "text": TEXT,
               "fidelity_scope": "Full original HF model initialization and F32 backbone prefill; no depth or speech acceptance"}

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def headroom(stage, minimum_gib):
        state = ab.snapshot()
        receipt[stage] = state
        save()
        ab.require_headroom(state)
        if int(state["memory"]["MemAvailable"].split()[0]) < minimum_gib * 1024**2:
            raise RuntimeError(f"original F32 model requires {minimum_gib} GiB available before {stage}")

    def retain_array(name, value):
        import numpy as np
        array = value.detach().float().cpu().numpy().astype("<f4", copy=False)
        if not np.isfinite(array).all():
            raise RuntimeError(f"original model returned nonfinite {name}")
        np.save(out / f"{name}.npy", array, allow_pickle=False)
        array.tofile(out / f"{name}.bin")
        receipt["files"][name] = {"shape": list(array.shape), "dtype": "float32",
                                   "raw_sha256": ab.sha256(out / f"{name}.bin"),
                                   "npy_sha256": ab.sha256(out / f"{name}.npy")}
        save()

    save()
    try:
        headroom("resources_before_import", 12)
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        import transformers.models.csm.modeling_csm as modeling
        import transformers.modeling_rope_utils as rope
        from transformers import AutoTokenizer, CsmForConditionalGeneration

        for name, module in (("modeling_csm.py", modeling), ("modeling_rope_utils.py", rope)):
            source = Path(inspect.getfile(module))
            expected = original.SOURCES[name][1]
            if ab.sha256(source) != expected:
                raise RuntimeError(f"installed original module differs from pinned source: {name}")
            shutil.copy2(source, out / name)
        receipt["versions"] = {name: importlib.metadata.version(name)
                                for name in ("torch", "transformers", "numpy", "safetensors", "huggingface_hub")}
        if torch.__version__.split("+")[0] != "2.6.0":
            raise RuntimeError("reference requires the same Torch 2.6.0 as the original operator fixture")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(1234)
        config = ab.regression.hf_download("sesame/csm-1b", "config.json", original.CONFIG_REVISION, scratch)
        card = ab.regression.hf_download("sesame/csm-1b", "README.md", original.CONFIG_REVISION, scratch)
        if ab.sha256(config) != original.CONFIG_SHA256 or ab.sha256(card) != original.CARD_SHA256:
            raise RuntimeError("original config/model license card bytes changed")
        shutil.copy2(config, out / "original-config.json")
        shutil.copy2(card, out / "CSM_MODEL_CARD.md")
        original.fetch(f"https://raw.githubusercontent.com/huggingface/transformers/{original.REVISION}/LICENSE",
                       out / "TRANSFORMERS_LICENSE")
        original.fetch("https://raw.githubusercontent.com/SesameAILabs/csm/836f886515f0dec02c22ed2316cc78904bdc0f36/LICENSE",
                       out / "CSM_LICENSE", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4")
        index = ab.regression.hf_download("sesame/csm-1b", "transformers.safetensors.index.json",
                                          original.CONFIG_REVISION, scratch)
        weights = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if not weights or any(Path(name).name != name or not name.endswith(".safetensors") for name in weights):
            raise RuntimeError("original checkpoint has unsafe or missing shard names")
        headroom("resources_before_download", 12)
        snapshot = Path(snapshot_download("sesame/csm-1b", revision=original.CONFIG_REVISION,
                                          allow_patterns=[*weights, "*.json", "tokenizer*", "*.model", "README.md"]))
        # This repository also has an ORIGINAL-format model.safetensors. The
        # official HF conversion has differently named shards/index. Present
        # ONLY that index under HF's conventional loader name, preserving every
        # coefficient and weight-map entry; never let AutoModel choose the
        # original-format file as though it contained HF-format tensors.
        model_dir = scratch / "hf-loader-view"
        model_dir.mkdir()
        for path in snapshot.iterdir():
            if path.name in weights or path.suffix in (".json", ".model", ".txt"):
                if path.name != "model.safetensors.index.json":
                    (model_dir / path.name).symlink_to(path.resolve())
        (model_dir / "model.safetensors.index.json").symlink_to(index.resolve())
        receipt["loader_index_alias"] = {"source": "transformers.safetensors.index.json",
                                           "alias": "model.safetensors.index.json",
                                           "sha256": ab.sha256(index), "weights_unchanged": True}
        receipt["weight_files"] = {name: {"bytes": (model_dir / name).stat().st_size,
                                          "sha256": ab.sha256(model_dir / name)} for name in weights}
        receipt["weight_index_sha256"] = ab.sha256(index)
        receipt["config_sha256"] = ab.sha256(config)
        # Independently recover the exact prompt; do not silently substitute a
        # different tokenizer sequence to improve parity or speech.
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        actual_tokens = tokenizer.encode("[0]" + TEXT, add_special_tokens=True)
        # C++ explicitly surrounds the encoded body with begin/end-of-text.
        body = tokenizer.encode("[0]" + TEXT, add_special_tokens=False)
        expected_tokens = [128000, *body, 128001]
        receipt["tokenizer_encode_with_special_tokens"] = actual_tokens
        receipt["tokenizer_body_ids"] = body
        if expected_tokens != TOKENS:
            raise RuntimeError("original tokenizer differs from retained native sampled prompt")
        save()
        headroom("resources_before_model_load", 12)
        model, loading = CsmForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, attn_implementation="eager", output_loading_info=True)
        if loading.get("missing_keys") or loading.get("unexpected_keys") or loading.get("mismatched_keys"):
            receipt["loading_info"] = loading
            save()
            raise RuntimeError("original model checkpoint does not exactly populate the pinned class")
        model.eval()
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError("original model parameters are not all CPU F32")
        receipt["parameter_count"] = sum(p.numel() for p in model.parameters())
        handles = []
        handles.append(model.backbone_model.layers[0].input_layernorm.register_forward_hook(
            lambda module, inputs, result: retain_array("backbone_layer0_normed", result[0])))
        for i, layer in enumerate(model.backbone_model.layers):
            handles.append(layer.register_forward_hook(
                lambda module, inputs, result, i=i: retain_array(f"backbone_layer{i}_output", result[0][0])))
        # Parameters remain live here: enforce at least 4 GiB beyond resident
        # model pages before executing the bounded fourteen-token prefill.
        headroom("resources_before_prefill", 4)
        with torch.inference_mode():
            output = model(input_ids=torch.tensor([TOKENS], dtype=torch.long),
                           use_cache=False, output_hidden_states=True, logits_to_keep=1,
                           return_dict=True)
        for handle in handles:
            handle.remove()
        retain_array("backbone_prefill_last_h", output.hidden_states[-1][0, -1])
        retain_array("backbone_prefill_c0_logits", output.logits[0, -1])
        expected = {"backbone_layer0_normed": [14, 2048],
                    **{f"backbone_layer{i}_output": [14, 2048] for i in range(16)},
                    "backbone_prefill_last_h": [2048], "backbone_prefill_c0_logits": [2051]}
        if {key: value["shape"] for key, value in receipt["files"].items()} != expected:
            raise RuntimeError("original model hook shapes/stages differ from native prefill contract")
        receipt["complete"] = True
        save()
        (out / "summary.md").write_text("# Original CSM F32 prefill\n\nFull pinned HF model and checkpoint, "
                                         "exact fourteen-token native prompt, nineteen finite F32 stage arrays. "
                                         "No native parity or speech acceptance is implied.\n")
        return 0
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    sys.exit(main())
