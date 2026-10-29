#!/usr/bin/env python3
"""Original duration predictors on captured native encoder output and noise."""

import argparse
import json
import math
import os
from pathlib import Path
import types
import urllib.request

from melotts_decoder import comparison
from melotts_inputs import SOURCES, UPSTREAM, sha
from melotts_upstream import original_classes

TRANSFORMS_SHA = "e6798989a0d75cbec45e7e24e3697e1c4886b3664c6a80a452645ef1d2a93f9c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" and not (
        args.device == "cuda" and Path("/kaggle/working").is_dir()
    ):
        raise RuntimeError("Hosted-only duration boundary checks")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        complete=False,
        acceptance=False,
        application_acceptance=False,
        full_neural_parity=False,
        scope="Original deterministic/stochastic duration predictors; shared converted weights, native encoder output and noise",
        upstream=UPSTREAM,
        source_sha256=dict(SOURCES, **{"melo/transforms.py": TRANSFORMS_SHA}),
        device=args.device,
        cases=[],
        resource_checks=[],
    )

    def save():
        (args.output / "duration-controls.json").write_text(
            json.dumps(receipt, indent=2)
        )

    # Reuse the same persistent resource protocol as the native input child.
    def guard(label):
        import shutil
        import subprocess

        memory = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        row = dict(
            phase=label,
            available_bytes=int(memory["MemAvailable"].split()[0]) * 1024,
            load=list(os.getloadavg()),
            cpus=os.cpu_count() or 1,
            scratch_free_bytes=shutil.disk_usage(args.output).free,
        )
        if args.device == "cuda":
            row["gpu_free_mib"] = [
                int(x)
                for x in subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=30,
                ).splitlines()
            ]
        receipt["resource_checks"].append(row)
        save()
        if (
            row["available_bytes"] < 3 * 1024**3
            or row["load"][0] > 2 * row["cpus"]
            or row["scratch_free_bytes"] < 2 * 1024**3
        ):
            raise RuntimeError("Insufficient duration-reference resources")
        if args.device == "cuda" and (
            not row["gpu_free_mib"] or min(row["gpu_free_mib"]) < 1024
        ):
            raise RuntimeError("Insufficient actual GPU memory")

    guard("load-original-duration")
    import numpy as np
    import torch
    from gguf import GGUFReader, GGMLQuantizationType

    torch.set_num_threads(2)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Actual CUDA reference required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    source = args.captures / "oracle-source"
    path = source / "melo/transforms.py"
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/myshell-ai/MeloTTS/"
        + UPSTREAM
        + "/melo/transforms.py",
        timeout=60,
    ) as response:
        path.write_bytes(response.read())
    for name, digest in receipt["source_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError("Original duration source checksum mismatch")
    transforms = types.ModuleType("original_melo_transforms")
    exec(compile(path.read_text(), str(path), "exec"), transforms.__dict__)
    ns = dict(
        torch=torch,
        nn=torch.nn,
        F=torch.nn.functional,
        math=math,
        piecewise_rational_quadratic_transform=transforms.piecewise_rational_quadratic_transform,
    )
    original_classes(
        source / "melo/modules.py",
        {"LayerNorm", "DDSConv", "ElementwiseAffine", "ConvFlow", "Flip", "Log"},
        ns,
    )
    ns["modules"] = types.SimpleNamespace(
        **{
            name: ns[name]
            for name in [
                "LayerNorm",
                "DDSConv",
                "ElementwiseAffine",
                "ConvFlow",
                "Flip",
                "Log",
            ]
        }
    )
    original_classes(
        source / "melo/models.py",
        {"DurationPredictor", "StochasticDurationPredictor"},
        ns,
    )
    model = GGUFReader(args.model)
    tensors = {t.name: t for t in model.tensors}
    hidden = int(model.fields["melotts.hidden_channels"].contents())
    gin = int(model.fields["melotts.gin_channels"].contents())
    noise_scale = float(model.fields["melotts.noise_w"].contents())
    trained_flows = len(
        [
            name
            for name in tensors
            if name.startswith("sdp.flows.") and name.endswith(".pre.weight")
        ]
    )
    dp = ns["DurationPredictor"](
        hidden,
        tensors["dp.conv_1.weight"].data.shape[0],
        tensors["dp.conv_1.weight"].data.shape[-1],
        0.5,
        gin_channels=gin,
    )
    sdp = ns["StochasticDurationPredictor"](
        hidden,
        hidden,
        tensors["sdp.convs.convs_sep.0.weight"].data.shape[-1],
        0.5,
        n_flows=trained_flows,
        gin_channels=gin,
    )
    for module, prefix in [(dp, "dp."), (sdp, "sdp.")]:
        weights = {}
        for name, tensor in tensors.items():
            if not name.startswith(prefix):
                continue
            if tensor.tensor_type not in [
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
            ]:
                raise ValueError("Unquantized original duration weights required")
            weights[name[len(prefix) :]] = torch.from_numpy(
                tensor.data.astype(np.float32, copy=True)
            )
        module.load_state_dict(weights, strict=True)
    receipt.update(
        model_sha256=sha(args.model),
        trained_stochastic_flows=trained_flows,
        active_stochastic_flows=trained_flows - 1,
        noise_scale=noise_scale,
        torch_version=torch.__version__,
    )
    del weights, tensors, model
    dp = dp.eval().to(args.device)
    sdp = sdp.eval().to(args.device)
    for index in range(3):
        guard("original-duration-" + str(index))
        case = args.captures / ("case-" + str(index))
        enc = (
            np.fromfile(case / "enc_output.bin", dtype="<f4")
            .reshape(-1, hidden)
            .T.copy()[None]
        )
        g = torch.from_numpy(
            np.fromfile(case / "speaker_emb.bin", dtype="<f4").reshape(1, gin, 1)
        ).to(args.device)
        x = torch.from_numpy(enc).to(args.device)
        mask = torch.ones((1, 1, enc.shape[-1]), device=args.device)
        scaled = (
            np.fromfile(case / "sdp_noise.bin", dtype="<f4")
            .reshape(-1, 2)
            .T.copy()[None]
        )

        # Inject only the common random input, without modifying the original
        # predictor/transform operators or claiming C++/Torch RNG equivalence.
        class SharedNoise:
            calls = 0

            def __getattr__(self, name):
                return getattr(torch, name)

            def randn(self, *shape):
                self.calls += 1
                if tuple(shape) != scaled.shape or self.calls != 1 or noise_scale <= 0:
                    raise ValueError("Original SDP random-input protocol changed")
                return torch.from_numpy(scaled) / noise_scale

        shared = SharedNoise()
        ns["torch"] = shared
        try:
            with torch.inference_mode():
                references = dict(
                    dp_logw=dp(x, mask, g=g),
                    sdp_logw=sdp(x, mask, g=g, reverse=True, noise_scale=noise_scale),
                )
        finally:
            ns["torch"] = torch
        if shared.calls != 1:
            raise ValueError("Original SDP did not consume the shared noise")
        results = {}
        for name, tensor in references.items():
            reference = tensor.flatten().cpu().numpy()
            reference.tofile(args.output / (f"case-{index}-{name}-original.bin"))
            results[name] = comparison(
                np.fromfile(case / (name + ".bin"), dtype="<f4"), reference
            )
        receipt["cases"].append(
            dict(
                case=index,
                stages=results,
                accepted=all(row["accepted"] for row in results.values()),
            )
        )
        save()
    receipt.update(
        complete=True, acceptance=all(row["accepted"] for row in receipt["cases"])
    )
    save()
    if not receipt["acceptance"]:
        raise RuntimeError(
            "Original duration boundary differs; inspect retained comparisons"
        )


if __name__ == "__main__":
    main()
