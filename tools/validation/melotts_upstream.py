#!/usr/bin/env python3
"""Isolate original Melo encoder and flow on actual native boundary inputs.

Converted weights and captured BERT features/noise are shared inputs. This
does not qualify BERT inference, conversion, duration sampling or full speech.
"""

import argparse
import ast
import json
import math
import os
from pathlib import Path
import types
import urllib.request

from melotts_decoder import COMMONS_SHA, comparison, guard
from melotts_inputs import SOURCES, UPSTREAM, sha

ATTENTIONS_SHA = "5261b73e4945651e9f7133375f4f83c1532a30f8227f6249c7006d696a737353"


def original_classes(path, names, namespace):
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name in names
    ]
    if len(nodes) != len(names):
        raise ValueError("Original class selection changed")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


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
        raise RuntimeError("Hosted CUDA diagnostic only")
    guard(args.output.parent)
    import numpy as np
    import torch
    from gguf import GGUFReader, GGMLQuantizationType

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Original boundaries require actual CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.captures / "oracle-source"
    sources = dict(
        SOURCES,
        **{"melo/commons.py": COMMONS_SHA, "melo/attentions.py": ATTENTIONS_SHA},
    )
    for name in ["melo/commons.py", "melo/attentions.py"]:
        with urllib.request.urlopen(
            "https://raw.githubusercontent.com/myshell-ai/MeloTTS/"
            + UPSTREAM
            + "/"
            + name,
            timeout=60,
        ) as response:
            (source / name).write_bytes(response.read())
    for name, digest in sources.items():
        if sha(source / name) != digest:
            raise ValueError("Original source checksum mismatch")
    ns = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, math=math)
    # commons defines convert_pad_shape twice; Python uses its last definition.
    functions = {
        n.name: n
        for n in ast.parse((source / "melo/commons.py").read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"sequence_mask", "convert_pad_shape"}
    }
    if len(functions) != 2:
        raise ValueError("Original common functions changed")
    exec(
        compile(
            ast.Module(body=list(functions.values()), type_ignores=[]),
            "original-melo-commons",
            "exec",
        ),
        ns,
    )
    ns["commons"] = types.SimpleNamespace(**{k: ns[k] for k in functions})
    original_classes(
        source / "melo/attentions.py",
        {"LayerNorm", "Encoder", "MultiHeadAttention", "FFN"},
        ns,
    )
    ns["attentions"] = types.SimpleNamespace(Encoder=ns["Encoder"])
    original_classes(
        source / "melo/modules.py", {"Flip", "TransformerCouplingLayer"}, ns
    )
    ns["modules"] = types.SimpleNamespace(
        Flip=ns["Flip"], TransformerCouplingLayer=ns["TransformerCouplingLayer"]
    )
    original_classes(
        source / "melo/models.py", {"TextEncoder", "TransformerCouplingBlock"}, ns
    )
    model = GGUFReader(args.model)
    fields = model.fields

    def integer(name):
        return int(fields["melotts." + name].contents())

    tensors = {tensor.name: tensor for tensor in model.tensors}
    hidden = integer("hidden_channels")
    inter = integer("inter_channels")
    gin = integer("gin_channels")
    encoder = ns["TextEncoder"](
        integer("num_symbols"),
        inter,
        hidden,
        integer("filter_channels"),
        integer("n_heads"),
        integer("n_layers_enc"),
        tensors["enc_p.encoder.ffn_layers.0.conv_1.weight"].data.shape[-1],
        0,
        gin_channels=gin,
        num_languages=integer("num_languages"),
        num_tones=integer("num_tones"),
    )
    flow = ns["TransformerCouplingBlock"](
        inter,
        hidden,
        integer("filter_channels"),
        integer("n_heads"),
        integer("n_layers_trans_flow"),
        tensors["flow.flows.0.enc.ffn_layers.0.conv_1.weight"].data.shape[-1],
        0,
        n_flows=integer("n_flow_blocks"),
        gin_channels=gin,
        share_parameter=False,
    )
    counts = {}
    for module, prefix in [(encoder, "enc_p."), (flow, "flow.")]:
        weights = {}
        for name, tensor in tensors.items():
            if not name.startswith(prefix):
                continue
            if tensor.tensor_type not in [
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
            ]:
                raise ValueError("Require identical unquantized boundary weights")
            weights[name[len(prefix) :]] = torch.from_numpy(
                tensor.data.astype(np.float32, copy=True)
            )
        module.load_state_dict(weights, strict=True)
        counts[prefix] = len(weights)
    del weights, tensors, model
    encoder = encoder.eval().cuda()
    flow = flow.eval().cuda()
    receipt = dict(
        complete=False,
        acceptance=False,
        application_acceptance=False,
        full_neural_parity=False,
        scope="Original encoder and inverse flow with shared converted weights and actual native BERT/latent inputs",
        upstream=UPSTREAM,
        source_sha256=sources,
        model_sha256=sha(args.model),
        device="cuda",
        torch_version=torch.__version__,
        weight_tensors=counts,
        cases=[],
        resource_checks=[],
    )

    def save():
        (args.output / "upstream-controls.json").write_text(
            json.dumps(receipt, indent=2)
        )

    save()
    for index in range(3):
        receipt["resource_checks"].append(guard(args.output))
        case = args.captures / ("case-" + str(index))

        def ids(name):
            return torch.from_numpy(
                np.fromfile(case / (name + ".bin"), dtype="<i4").astype(np.int64)
            )[None].cuda()

        phones = ids("phoneme_ids")
        tones = ids("tone_ids")
        languages = ids("lang_ids")
        length = phones.shape[1]
        g = torch.from_numpy(
            np.fromfile(case / "speaker_emb.bin", dtype="<f4").reshape(1, gin, 1)
        ).cuda()
        ja_bert = torch.from_numpy(
            np.fromfile(case / "ja_bert_features.bin", dtype="<f4").reshape(
                1, 768, length
            )
        ).cuda()
        bert = torch.zeros((1, 1024, length), device="cuda")
        embeddings = []

        def capture_embedding(module, inputs):
            embeddings.append(inputs[0].detach().clone())

        hook = encoder.encoder.register_forward_pre_hook(capture_embedding)
        try:
            with torch.inference_mode():
                enc, mean, logs, mask = encoder(
                    phones,
                    torch.tensor([length], device="cuda"),
                    tones,
                    languages,
                    bert,
                    ja_bert,
                    g=g,
                )
        finally:
            hook.remove()
        references = dict(
            emb_output=embeddings[0], enc_output=enc, enc_mean=mean, enc_logvar=logs
        )
        # The native sampled latent is the exact shared input. No Torch/C++ RNG equivalence is assumed.
        receipt["resource_checks"].append(guard(args.output))
        z = np.fromfile(case / "z_p.bin", dtype="<f4").reshape(-1, inter).T.copy()[None]
        with torch.inference_mode():
            references["z_dec"] = flow(
                torch.from_numpy(z).cuda(),
                torch.ones((1, 1, z.shape[-1]), device="cuda"),
                g=g,
                reverse=True,
            )
        results = {}
        for name, value in references.items():
            reference = value.squeeze(0).T.contiguous().cpu().numpy()
            reference.tofile(args.output / (f"case-{index}-{name}-original.bin"))
            actual = np.fromfile(case / (name + ".bin"), dtype="<f4")
            results[name] = comparison(actual, reference)
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
            "Original upstream boundary diverges; every comparison is retained"
        )


if __name__ == "__main__":
    main()
