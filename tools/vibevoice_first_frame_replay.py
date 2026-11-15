#!/usr/bin/env python3
"""Tiny-control-validated prototype: replay a bounded native first frame through actual upstream head and scheduler.

This injects native LM conditions; it does NOT establish upstream LM/prefill parity.
Only prediction-head GGUF tensors are dequantized. No model download or GPU use.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def upstream_head(path, torch):
    class Base(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
    ns = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, math=math,
              Optional=__import__('typing').Optional, Tuple=tuple, Union=__import__('typing').Union,
              ACT2FN={'silu': torch.nn.SiLU()}, PreTrainedModel=Base,
              VibeVoiceDiffusionHeadConfig=object)
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns['VibeVoiceDiffusionHead']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--cfg', type=float, default=1.3)
    args = p.parse_args()
    if sha(args.gguf) != args.expected_sha256:
        p.error('GGUF SHA-256 mismatch')
    if sum(f.stat().st_size for f in args.capture.iterdir() if f.is_file()) > 64 << 20:
        p.error('capture exceeds 64 MiB')
    dims = np.fromfile(args.capture / 'capture_complete.bin', dtype='<i4')
    if len(dims) != 3 or not (0 < dims[2] <= 100):
        p.error('incomplete or invalid bounded capture')
    vae, hidden, steps = map(int, dims)
    import torch
    import gguf
    from gguf.quants import dequantize
    torch.set_num_threads(1)
    head_path = args.source / 'vibevoice/modular/modular_vibevoice_diffusion_head.py'
    scheduler_path = args.source / 'vibevoice/schedule/dpm_solver.py'
    cls = upstream_head(head_path, torch)
    reader = gguf.GGUFReader(str(args.gguf))
    tensors = {t.name: t for t in reader.tensors if t.name.startswith('pred.')}
    size = sum(int(np.prod(t.shape)) * 4 for t in tensors.values())
    if not size or size > 2 << 30:
        p.error('prediction head exceeds 2 GiB dequantized-weight prototype bound')
    layers = len({n.split('.')[2] for n in tensors if n.startswith('pred.layers.')})
    ffn = int(tensors['pred.layers.0.ffn.gate_proj.weight'].shape[1])
    cfg = SimpleNamespace(hidden_size=hidden, latent_size=vae, head_ffn_ratio=ffn / hidden,
                          head_layers=layers, rms_norm_eps=1e-5)
    with torch.device('meta'):
        head = cls(cfg)
    converter = Path(__file__).resolve().parents[1] / 'models/convert-vibevoice-to-gguf.py'
    main_node = next(n for n in ast.parse(converter.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    shorten_node = next(n for n in main_node.body if isinstance(n, ast.FunctionDef) and n.name == 'shorten')
    ns = {}
    exec(compile(ast.Module(body=[shorten_node], type_ignores=[]), str(converter), 'exec'), ns)
    state = {}
    for name in head.state_dict():
        tensor = tensors[ns['shorten']('model.prediction_head.' + name)]
        array = dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape)))
        state[name] = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
    head.load_state_dict(state, assign=True)
    head.eval()
    spec = importlib.util.spec_from_file_location('first_frame_upstream_scheduler', scheduler_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scheduler = module.DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
    scheduler.set_timesteps(steps)
    native_ts = np.fromfile(args.capture / 'tts_frame0_timesteps.bin', dtype='<i4')
    if not np.array_equal(native_ts, scheduler.timesteps.numpy()):
        p.error('native/upstream timestep schedules differ')
    def read(name, count):
        v = np.fromfile(args.capture / (name + '.bin'), dtype='<f4')
        if v.size != count or not np.isfinite(v).all():
            raise ValueError('invalid capture stage ' + name)
        return torch.from_numpy(v.copy())
    cond = torch.stack([read('tts_pos_condition_frame0', hidden), read('tts_neg_condition_frame0', hidden)])
    z = read('tts_noise_frame0', vae).repeat(2, 1)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    def compare(name, value):
        a = value.detach().float().numpy().ravel()
        b = read(name, a.size).numpy()
        a.astype('<f4').tofile(args.output / (name + '.bin'))
        x, y = a.astype('float64'), b.astype('float64')
        rows.append(dict(stage=name, cosine=float(x@y / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-30)),
                         norm_ref=float(np.linalg.norm(x)), norm_native=float(np.linalg.norm(y)),
                         max_abs=float(np.max(np.abs(x-y)))))
    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            both = head(z[:1].repeat(2, 1), t.repeat(2).to(z), cond)
            v = both[1] + args.cfg * (both[0] - both[1])
            compare(f'tts_frame0_v_both_s{i:03}', both)
            compare(f'tts_frame0_v_cfg_s{i:03}', v)
            z = scheduler.step(v.repeat(2, 1), t, z).prev_sample
            compare(f'tts_frame0_z_s{i:03}', z[0])
        compare('tts_latent_frame0', z[0])
    receipt = dict(scope='upstream prediction head and scheduler; native injected LM conditions; no prefill parity',
                   weights_sha256=args.expected_sha256, source_sha256={str(f.relative_to(args.source)): sha(f) for f in (head_path, scheduler_path)},
                   dequantized_weight_bytes=size, stages=rows)
    (args.output / 'comparison.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
