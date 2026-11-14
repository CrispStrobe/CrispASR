#!/usr/bin/env python
"""Optional CPU parity test against Transformers Qwen2; no downloaded weights.

Compile vibevoice-negative-history-probe.cpp against the repository ggml build,
then run this file with --probe PATH --output-dir SCRATCH. Requires torch,
transformers and gguf. The fixture is a seeded two-layer Qwen2 with GQA and
nontrivial rotary positions; the probe calls the production cached TTS graph.
"""
import argparse
import json
import subprocess
from pathlib import Path

def main():
    import numpy as np
    import torch
    from gguf import GGUFWriter
    from transformers import Qwen2Config, Qwen2Model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(589)
    config = Qwen2Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=16,
        max_position_embeddings=32, rope_theta=1000000.0,
        attention_dropout=0.0, rms_norm_eps=1e-6,
    )
    config._attn_implementation = "eager"
    model = Qwen2Model(config).eval()
    # Large enough weights that accidentally reading the positive cache is clear.
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "norm" not in name:
                value.mul_(4)
    inputs = torch.randn(1, 4, 32)
    model_path = args.output_dir / "tiny.gguf"
    writer = GGUFWriter(str(model_path), "vibevoice-tts")
    for name, value in {
        "d_lm": 32, "d_ffn": 64, "n_lm_layers": 2, "n_heads": 4,
        "n_kv_heads": 2, "head_dim": 8, "vocab_size": 16,
    }.items():
        writer.add_uint32("vibevoice." + name, value)
    writer.add_float32("vibevoice.rope_theta", config.rope_theta)
    for name, value in model.state_dict().items():
        key = "lm." + name
        for before, after in [
            ("embed_tokens", "tok_emb"), ("self_attn", "attn"),
            ("input_layernorm", "attn_ln"),
            ("post_attention_layernorm", "ffn_ln"), ("mlp", "ffn"),
            ("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down"),
        ]:
            key = key.replace(before, after)
        writer.add_tensor(key, value.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    input_path = args.output_dir / "inputs.bin"
    output_path = args.output_dir / "outputs.bin"
    inputs.numpy().tofile(input_path)
    subprocess.run([args.probe, str(model_path), str(input_path), str(output_path)], check=True)
    with torch.no_grad():
        normal = model(inputs_embeds=inputs, use_cache=False).last_hidden_state[0]
        restarted = torch.cat([inputs[:, :1], inputs[:, :1], inputs[:, 1:2]], dim=1)
        restart = model(inputs_embeds=restarted, use_cache=False).last_hidden_state[0, 1:]
        reference = torch.cat([normal, restart]).numpy()
    actual = np.fromfile(output_path, np.float32).reshape(6, 32)
    report = {}
    for index, (mine, ref) in enumerate(zip(actual, reference)):
        mine, ref = mine.astype(np.float64), ref.astype(np.float64)
        cosine = float(mine @ ref / np.linalg.norm(mine) / np.linalg.norm(ref))
        relative_l2 = float(np.linalg.norm(mine - ref) / np.linalg.norm(ref))
        report[str(index)] = {
            "cosine": cosine, "relative_l2": relative_l2,
            "mine_norm": float(np.linalg.norm(mine)),
            "reference_norm": float(np.linalg.norm(ref)),
        }
        assert cosine > 0.999999, report
        assert relative_l2 < 0.001, report
    # A frozen initial condition must fail decisively on the same teacher inputs.
    frozen_error = float(np.linalg.norm(actual[0] - reference[1]) / np.linalg.norm(reference[1]))
    assert frozen_error > 0.5, frozen_error
    report["frozen_frame1_relative_l2"] = frozen_error
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("PASS: six upstream negative states, interleaved positive history, restart, and frozen control")


if __name__ == "__main__":
    main()
