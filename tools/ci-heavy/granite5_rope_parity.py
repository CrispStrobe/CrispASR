#!/usr/bin/env python3
"""Check actual shared Granite attention against the pinned original RoPE dump."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    origin = json.loads(Path(str(args.oracle) + ".json").read_text())
    assert origin["original_module"] and origin["transformers"] == "4.57.3"
    assert origin["source_files"] == {
        "modeling_granite.py": "edb70e0780d16eb5e473df244cb9153e96f01fe4944010e784fcd1cbc5a0b542",
        "modeling_rope_utils.py": "55cc0c8cb76f592ab178b4662adcbdf4ad5012bb0b4249c0a96603141f18e9ac"}
    assert origin["rope_scaling"] == {
        "factor": 4.0, "original_max_position_embeddings": 4096, "rope_type": "yarn"}
    arrays = np.load(args.oracle, allow_pickle=False)
    assert arrays["positions"].tolist() == [[0, 1, 127, 4095, 4096, 8191, 16383]]
    fixture = args.out / "original.bin"
    with fixture.open("wb") as stream:
        stream.write(struct.pack("<ii", 128, 7))
        stream.write(arrays["positions"].astype("<i4").tobytes())
        for key in ("q", "k", "q_rotated", "k_rotated"):
            assert arrays[key].shape == (1, 1, 7, 128) and np.isfinite(arrays[key]).all()
            stream.write(arrays[key].astype("<f4").tobytes())
    receipt = {"oracle_sha256": sha(args.oracle), "probe_sha256": sha(args.probe),
               "fixture_sha256": sha(fixture), "native_model_acceptance": False, "device": args.device,
               "precision_budget": {"maxabs": .005, "low_position_maxabs": 2e-5,
                                    "relative_l2": .001, "norm_error": 1e-6},
               "precision_basis": "Normalized FP32 fixture; CUDA uses fast pow/trig intrinsics, phase roundoff grows with position",
               "complete": False, "cases": []}
    for mode, name in enumerate(("unscaled_rejected", "yarn_matches", "wrong_context_rejected",
                                  "double_magnitude_rejected", "wrong_pairing_rejected")):
        result = subprocess.run([str(args.probe), str(fixture), str(mode), args.device], capture_output=True, text=True)
        (args.out / f"{name}.log").write_text(result.stdout + result.stderr)
        metrics = json.loads(result.stdout)
        if args.device == "cuda":
            assert metrics["backend"] == "CUDA0", metrics
        assert result.returncode == (0 if mode == 1 else 1), (name, result.returncode, metrics)
        receipt["cases"].append({"case": name, "exit_code": result.returncode, **metrics})
    receipt["complete"] = True
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Original-module RoPE parity passed; all four incorrect controls rejected.")


if __name__ == "__main__":
    main()
