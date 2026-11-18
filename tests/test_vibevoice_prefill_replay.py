#!/usr/bin/env python3
"""Tiny39-token actual-upstream streaming/full/native production graph control."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from gguf import GGUFWriter
    from transformers import Qwen2Config, Qwen2Model
    torch.set_num_threads(1)
    torch.manual_seed(589)
    config = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=16,
        max_position_embeddings=64, rope_theta=1000000., rms_norm_eps=1e-6,
        attention_dropout=0.)
    config._attn_implementation = 'eager'
    full = Qwen2Model(config).eval()
    with torch.no_grad():
        for name, parameter in full.named_parameters():
            if 'norm' not in name:
                parameter.mul_(4)
    prompt_ids = np.arange(39, dtype=np.int64) % 16
    inputs = full.embed_tokens(torch.from_numpy(prompt_ids).unsqueeze(0)).detach()
    args.output.mkdir(parents=True, exist_ok=False)
    model = args.output / 'tiny.gguf'
    writer = GGUFWriter(str(model), 'vibevoice-tts')
    for name, value in dict(d_lm=32, d_ffn=64, n_lm_layers=2, n_heads=4, n_kv_heads=2, head_dim=8, vocab_size=16).items():
        writer.add_uint32('vibevoice.' + name, value)
    writer.add_float32('vibevoice.rope_theta', config.rope_theta)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('prefill_replay', root / 'tools/vibevoice_prefill_replay.py')
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    for name, value in full.state_dict().items():
        key = 'lm.tok_emb.weight' if name == 'embed_tokens.weight' else 'lm.' + replay.native_key(name)
        writer.add_tensor(key, value.numpy())
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    captured = []
    hooks = [layer.register_forward_hook(lambda module, values, output: captured.append(output.detach().clone())) for layer in full.layers]
    with torch.no_grad():
        expected = full(inputs_embeds=inputs, use_cache=False).last_hidden_state
    for hook in hooks:
        hook.remove()
    report = replay.replay(model, config, inputs.numpy(), args.output / 'upstream', prompt_ids)
    for index, value in enumerate(captured):
        actual = np.fromfile(args.output / 'upstream' / f'prefill_block_{index:03}.bin', dtype='<f4').reshape(1,39,32)
        assert np.array_equal(actual, value.numpy()), 'streamed/full upstream block mismatch'
    upstream_hidden = np.fromfile(args.output / 'upstream/tts_prefill_hidden.bin', dtype='<f4')
    assert np.array_equal(upstream_hidden, expected[0,-1].numpy()), 'streamed/full final norm mismatch'
    inputs.numpy().astype('<f4').tofile(args.output / 'inputs.bin')
    subprocess.run([str(args.probe.resolve()), str(model), str(args.output/'inputs.bin'), str(args.output/'native.bin')], check=True)
    native = np.fromfile(args.output/'native.bin', dtype='<f4').reshape(2,48)
    comparisons = []
    with torch.no_grad():
        for index, raw in enumerate(captured):
            hidden = full.norm(raw)[0,-1].numpy()
            logits = (full.embed_tokens.weight @ torch.from_numpy(hidden)).numpy()
            for label, reference, mine in [('hidden',hidden,native[index,:32]), ('logits',logits,native[index,32:])]:
                x, y = reference.astype('float64'), mine.astype('float64')
                relative = float(np.linalg.norm(x-y)/np.linalg.norm(x))
                cosine = float(x@y / np.linalg.norm(x) / np.linalg.norm(y))
                comparisons.append(dict(depth=index+1, stage=label, cosine=cosine, relative_l2=relative,
                    upstream_norm=float(np.linalg.norm(x)), native_norm=float(np.linalg.norm(y)), max_abs=float(np.max(np.abs(x-y)))))
                assert cosine > .999999 and relative < .001, comparisons[-1]
    boundary = args.output / 'upstream/prefill_block_000.bin'
    altered = np.fromfile(boundary, dtype='<f4')
    altered[0] += .25
    altered_path = args.output / 'perturbed-block.bin'
    altered.tofile(altered_path)
    perturbation = replay.compare_files(boundary, altered_path)
    assert abs(perturbation['max_abs'] - .25) < 1e-6 and perturbation['relative_l2'] > .001
    receipt = dict(negative_perturbation=dict(stage='prefill_block_000', **perturbation), scope='tiny39-token upstream/full/streamed/native controls only; not7B parity',
        upstream_streamed_raw_blocks_bit_exact=True, native_comparisons=comparisons, replay=report)
    (args.output/'comparison.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    print('PASS: tiny39-token2-layer full upstream and streamed blocks exact; native truncated hidden/logits controls')


if __name__ == '__main__':
    main()
