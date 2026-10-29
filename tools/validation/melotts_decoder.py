#!/usr/bin/env python3
"""Original Melo Generator on captured native inputs and identical GGUF weights.

This isolates the decoder. It does not qualify checkpoint conversion, upstream
conditioning/flow, complete original-model speech, or the failed ASR gate.
"""

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import types
import urllib.request

from melotts_inputs import SOURCES, UPSTREAM, select_functions, sha

COMMONS_SHA = "046943d265635c3d9c99195562e703564a26300981a7db5b87d4cd3fd180ae0d"


def guard(directory):
    memory = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    available = int(memory["MemAvailable"].split()[0]) * 1024
    free = [
        int(x)
        for x in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
    ]
    row = dict(available_bytes=available, load=list(os.getloadavg()), gpu_free_mib=free)
    if (
        available < 3 * 1024**3
        or row["load"][0] > 2 * (os.cpu_count() or 1)
        or not free
        or min(free) < 1024
    ):
        raise RuntimeError(
            "Insufficient hosted resources before original decoder inference"
        )
    if shutil.disk_usage(directory).free < 2 * 1024**3:
        raise RuntimeError("Insufficient diagnostic disk headroom")
    return row


def comparison(actual, reference):
    import numpy as np

    a = np.asarray(actual, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    if (
        a.shape != b.shape
        or not len(a)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        return dict(accepted=False, actual_elements=len(a), reference_elements=len(b))
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0 or bn == 0:
        return dict(accepted=False, actual_norm=an, reference_norm=bn)
    cosine = float(np.dot(a, b) / (an * bn))
    ratio = an / bn
    relative = float(np.linalg.norm(a - b) / bn)
    return dict(
        accepted=bool(cosine >= 0.999 and 0.99 <= ratio <= 1.01 and relative <= 0.02),
        elements=len(a),
        cosine=cosine,
        norm_ratio=ratio,
        relative_l2=relative,
        max_absolute_error=float(np.max(np.abs(a - b))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        not Path("/kaggle/working").is_dir()
        or os.environ.get("CRISPASR_DECODER_READY") != "1"
    ):
        raise RuntimeError("Hosted GPU diagnostic only")
    guard(args.output.parent)
    import numpy as np
    import torch
    from gguf import GGUFReader, GGMLQuantizationType

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("The original decoder must actually use CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.captures / "oracle-source"
    sources = dict(SOURCES, **{"melo/commons.py": COMMONS_SHA})
    path = source / "melo/commons.py"
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/myshell-ai/MeloTTS/"
        + UPSTREAM
        + "/melo/commons.py",
        timeout=60,
    ) as response:
        path.write_bytes(response.read())
    for name, digest in sources.items():
        if sha(source / name) != digest:
            raise ValueError("Original source checksum mismatch: " + name)
    ns = dict(
        torch=torch,
        nn=torch.nn,
        F=torch.nn.functional,
        Conv1d=torch.nn.Conv1d,
        ConvTranspose1d=torch.nn.ConvTranspose1d,
        weight_norm=torch.nn.utils.weight_norm,
        remove_weight_norm=torch.nn.utils.remove_weight_norm,
    )
    select_functions(path, {"init_weights", "get_padding"}, ns)
    tree = ast.parse((source / "melo/modules.py").read_text())
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.ClassDef) and n.name in {"ResBlock1", "ResBlock2"})
        or (
            isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "LRELU_SLOPE" for t in n.targets
            )
        )
    ]
    if len(nodes) != 3:
        raise ValueError("Original residual block selection changed")
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), "original-melo-resblocks", "exec"
        ),
        ns,
    )
    ns["modules"] = types.SimpleNamespace(
        **{k: ns[k] for k in ["ResBlock1", "ResBlock2", "LRELU_SLOPE"]}
    )
    tree = ast.parse((source / "melo/models.py").read_text())
    nodes = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Generator"
    ]
    if len(nodes) != 1:
        raise ValueError("Original Generator selection changed")
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), "original-melo-generator", "exec"
        ),
        ns,
    )
    model = GGUFReader(args.model)
    fields = model.fields

    def integer(name):
        return int(fields["melotts." + name].contents())

    n = integer("n_upsample_stages")
    kernels = []
    dilations = []
    for i in range(10):
        if "melotts.resblock_kernel." + str(i) not in model.fields:
            break
        kernels.append(integer("resblock_kernel." + str(i)))
        dilations.append([integer(f"resblock_dilation.{i}.{j}") for j in range(3)])
    decoder = ns["Generator"](
        integer("inter_channels"),
        "1",
        kernels,
        dilations,
        [integer("upsample_rate." + str(i)) for i in range(n)],
        integer("upsample_initial_channel"),
        [integer("upsample_kernel." + str(i)) for i in range(n)],
        gin_channels=integer("gin_channels"),
    )
    decoder.remove_weight_norm()
    weights = {}
    for tensor in model.tensors:
        if not tensor.name.startswith("dec."):
            continue
        if tensor.tensor_type not in [
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F16,
        ]:
            raise ValueError(
                "This diagnostic requires identical unquantized decoder weights"
            )
        weights[tensor.name[4:]] = torch.from_numpy(
            tensor.data.astype(np.float32, copy=True)
        )
    # Strict loading prevents even one randomly initialized parameter surviving.
    decoder.load_state_dict(weights, strict=True)
    receipt = dict(
        complete=False,
        acceptance=False,
        application_acceptance=False,
        scope="Decoder only; original Generator/ResBlock classes, shared converted weights and native latent inputs",
        full_neural_parity=False,
        upstream=UPSTREAM,
        source_sha256=sources,
        model_sha256=sha(args.model),
        device="cuda",
        torch_version=torch.__version__,
        weight_tensors=len(weights),
        cases=[],
        resource_checks=[],
    )
    del weights, model
    decoder = decoder.eval().cuda()

    def save():
        (args.output / "decoder-controls.json").write_text(
            json.dumps(receipt, indent=2)
        )

    save()
    for index in range(3):
        receipt["resource_checks"].append(guard(args.output))
        case = args.captures / ("case-" + str(index))
        z = np.fromfile(case / "z_dec.bin", dtype="<f4")
        channels = decoder.conv_pre.in_channels
        z = z.reshape(-1, channels).T.copy()[None]
        g = np.fromfile(case / "speaker_emb.bin", dtype="<f4").reshape(1, -1, 1)
        residuals = []
        post = []

        def residual_hook(module, inputs, output):
            residuals.append(output.detach().clone())

        def post_hook(module, inputs):
            post.append(inputs[0].detach().clone())

        hooks = [
            block.register_forward_hook(residual_hook)
            for block in decoder.resblocks[-decoder.num_kernels :]
        ]
        hooks.append(decoder.conv_post.register_forward_pre_hook(post_hook))
        try:
            with torch.inference_mode():
                audio = decoder(torch.from_numpy(z).cuda(), torch.from_numpy(g).cuda())
                before = residuals[0].clone()
                for value in residuals[1:]:
                    before += value
                before /= decoder.num_kernels
                references = dict(
                    dec_before_post=before, dec_post_activation=post[0], audio=audio
                )
                results = {}
                for name, tensor in references.items():
                    reference = tensor.squeeze(0).T.contiguous().cpu().numpy()
                    reference.tofile(
                        args.output / (f"case-{index}-{name}-original.bin")
                    )
                    actual = np.fromfile(case / (name + ".bin"), dtype="<f4")
                    results[name] = comparison(actual, reference)
        finally:
            for hook in hooks:
                hook.remove()
        receipt["cases"].append(
            dict(
                case=index,
                stages=results,
                accepted=all(r["accepted"] for r in results.values()),
            )
        )
        save()
    receipt.update(
        complete=True, acceptance=all(row["accepted"] for row in receipt["cases"])
    )
    save()
    if not receipt["acceptance"]:
        raise RuntimeError(
            "Original decoder boundary differs; all comparisons retained"
        )


if __name__ == "__main__":
    main()
