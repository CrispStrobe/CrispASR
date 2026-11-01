#!/usr/bin/env python3
"""Collect input-column importance from the pinned original Mini BS RoFormer.

Each original torch.nn.Linear pre-hook accumulates sum(x**2) per input column
in float64 and the number of flattened input rows. Only those small statistics
are retained. The resulting GGUF uses the genuine CrispEmbed/CrispASR imatrix
format: F32 sum-of-squares tensors with original weight names and UINT64
``count.<weight-name>`` metadata. The quantizer divides by count internally.

    python tools/collect_mini_bs_roformer_imatrix.py \
      --model-dir /path/to/pinned-model \
      --audio diagnostic-stereo-0p5s.wav music-8s.wav --samples 22050 \
      --output mini-bs-roformer-imatrix.gguf --threads 2

This collects calibration statistics, not an acceptance result. Q8_0 ignores
importance weights in ggml; Q4_K and other weighted quantizers can use them.
Do not report calibration inputs as independent held-out validation inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reference_backends.mini_bs_roformer import MODEL_ID, MODEL_REVISION, SOURCE_HASHES, _load


class ImportanceCollector:
    """Observe unmodified original linear inputs with bounded reduction blocks."""

    def __init__(self, model, row_block_size: int = 1024):
        import torch

        if row_block_size <= 0:
            raise ValueError("row_block_size must be positive")
        self.row_block_size = row_block_size
        self.sumsq = {}
        self.counts = {}
        self.calls = {}
        self.handles = []
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight_name = f"{name}.weight"
            self.sumsq[weight_name] = np.zeros(module.in_features, dtype=np.float64)
            self.counts[weight_name] = 0
            self.calls[weight_name] = 0
            self.handles.append(module.register_forward_pre_hook(self._hook(weight_name, module.in_features)))
        expected = {name for name, tensor in model.state_dict().items()
                    if name.endswith(".weight") and tensor.ndim == 2}
        if set(self.sumsq) != expected:
            self.close()
            raise ValueError("the original model's linear modules do not cover every matrix weight")

    def _hook(self, name: str, width: int):
        def capture(_module, inputs):
            import torch

            if not inputs or not isinstance(inputs[0], torch.Tensor) or inputs[0].shape[-1] != width:
                raise ValueError(f"unexpected original linear input shape for {name}")
            source = inputs[0].detach().reshape(-1, width)
            rows = int(source.shape[0])
            if rows == 0:
                raise ValueError(f"empty original linear input for {name}")
            for start in range(0, rows, self.row_block_size):
                block = source[start:start + self.row_block_size].to(dtype=torch.float64)
                if not bool(torch.isfinite(block).all()):
                    raise ValueError(f"nonfinite original linear input for {name}")
                # CPU/GPU reductions accumulate in F64. Transfer only one
                # width-sized vector, rather than retaining the activation.
                energy = torch.sum(block * block, dim=0).cpu().numpy()
                self.sumsq[name] += energy
            self.counts[name] += rows
            self.calls[name] += 1

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def inventory(self) -> dict:
        result = {}
        for name, energy in sorted(self.sumsq.items()):
            count = self.counts[name]
            if count <= 0 or not np.isfinite(energy).all() or np.any(energy < 0):
                raise ValueError(f"invalid or uncaptured importance statistics for {name}")
            mean = energy / count
            result[name] = {
                "input_columns": len(energy), "rows": count, "calls": self.calls[name],
                "mean_energy_min": float(mean.min()), "mean_energy_max": float(mean.max()),
                "zero_columns": int(np.count_nonzero(energy == 0)),
            }
        return result


def write_imatrix(path: Path, collector: ImportanceCollector, provenance: dict) -> dict:
    """Write the exact F32-sumsq/UINT64-count format consumed by --imatrix."""
    from gguf import GGUFWriter

    inventory = collector.inventory()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(path), "crispembed-imatrix")
    writer.add_uint32("imatrix.version", 1)
    writer.add_string("imatrix.source.model", MODEL_ID)
    writer.add_string("imatrix.source.revision", MODEL_REVISION)
    writer.add_string("imatrix.provenance_json", json.dumps(provenance, sort_keys=True))
    for name, energy in sorted(collector.sumsq.items()):
        values = energy.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"importance sum exceeds the F32 file format for {name}")
        count = collector.counts[name]
        if count <= 0 or count >= 2**64:
            raise ValueError(f"invalid UINT64 activation-row count for {name}")
        writer.add_uint64("count." + name, count)
        writer.add_tensor(name, values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=MODEL_ID)
    parser.add_argument("--audio", type=Path, nargs="+", required=True, help="native 44100-Hz stereo WAV inputs")
    parser.add_argument("--samples", type=int, default=22050,
                        help="samples from the start of each input (default: 22050 = 0.5 seconds)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--row-block-size", type=int, default=1024)
    args = parser.parse_args()
    if args.samples <= 2048 or args.threads <= 0 or args.row_block_size <= 0:
        parser.error("samples must exceed 2048; threads and row-block-size must be positive")
    import soundfile as sf
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    model = _load(args.model_dir, device=args.device, dtype="float32")
    collector = ImportanceCollector(model, args.row_block_size)
    provenance = {
        "model": MODEL_ID, "revision": MODEL_REVISION, "source_sha256": SOURCE_HASHES,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "reference_dtype": "float32", "accumulator_dtype": "float64", "device": args.device,
        "threads": args.threads, "sample_rate": 44100, "channels": 2,
        "row_block_size": args.row_block_size, "inputs": [],
        "method": "original nn.Linear forward_pre_hook; sum of squares per input column",
        "acceptance": False,
    }
    try:
        for path in args.audio:
            # Read only the requested frames; an entire recording is never
            # retained merely to collect a short calibration window.
            with sf.SoundFile(path) as source:
                if source.samplerate != 44100 or source.channels != 2:
                    raise ValueError(f"{path.name}: expected native 44100-Hz stereo")
                if len(source) < args.samples:
                    raise ValueError(f"{path.name}: shorter than the requested {args.samples} samples")
                audio = source.read(args.samples, dtype="float32", always_2d=True)
            audio = np.ascontiguousarray(audio.T)
            if not np.isfinite(audio).all():
                raise ValueError(f"{path.name}: nonfinite calibration audio")
            provenance["inputs"].append({
                "name": path.name, "samples": args.samples, "offset_samples": 0,
                "input_pcm_sha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
            })
            wave = torch.from_numpy(audio).to(device=args.device).unsqueeze(0)
            with torch.inference_mode():
                output = model(wave)
                if not bool(torch.isfinite(output).all()):
                    raise ValueError(f"{path.name}: nonfinite original separated output")
            del output, wave
            print(f"Captured original inputs: {path.name}, {args.samples} stereo samples", flush=True)
    finally:
        collector.close()
    inventory = write_imatrix(args.output, collector, provenance)
    receipt = {
        "provenance": provenance, "tensor_count": len(inventory), "tensors": inventory,
        "imatrix_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "total_input_samples": sum(item["samples"] for item in provenance["inputs"]),
        "total_activation_rows": sum(value["rows"] for value in inventory.values()),
        "total_columns": sum(value["input_columns"] for value in inventory.values()),
    }
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    sidecar.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {len(inventory)} importance vectors, {receipt['total_columns']} columns")
    print(f"Provenance: {sidecar}")


if __name__ == "__main__":
    main()
