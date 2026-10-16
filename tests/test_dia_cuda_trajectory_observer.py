#!/usr/bin/env python3
"""Tiny actual original Dia/DAC method controls; no full model or GPU acceptance.

Run in an isolated reference Torch environment, with pinned blueprint and DAC
source paths. Neural decode logits and tiny codec weights are synthetic; the
sampler, CFG/masks, delay removal, from_codes and decode bodies are original.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types


def selected(source, name, methods=None):
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        assert {n.name for n in node.body} == set(methods)
    return compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blueprint', type=Path, required=True)
    parser.add_argument('--dac-source', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    kernel = root / 'tools/kaggle/dia-original-cuda-trace/dia_original_cuda_trace.py'
    spec = importlib.util.spec_from_file_location('trajectory_kernel', kernel)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    module = types.ModuleType('tiny_actual_dia')
    sys.modules[module.__name__] = module
    module.__dict__.update(torch=torch, np=np, DecoderInferenceState=object)
    exec(compile((args.blueprint / 'dia/audio.py').read_text(), str(args.blueprint / 'dia/audio.py'), 'exec'), module.__dict__)
    source = args.blueprint / 'dia/model.py'
    exec(selected(source, '_sample_next_token'), module.__dict__)
    exec(selected(source, 'Dia', ['_decoder_step', '_generate_output']), module.__dict__)
    codec = types.ModuleType('tiny_actual_dac')
    codec.__dict__.update(torch=torch, nn=nn, BaseModel=nn.Module,
                          CodecMixin=type('UnusedCodecMixin', (), {}))
    exec(selected(args.dac_source / 'nn/quantize.py', 'ResidualVectorQuantize', ['from_codes']), codec.__dict__)
    exec(selected(args.dac_source / 'model/dac.py', 'DAC', ['decode']), codec.__dict__)
    torch.manual_seed(123)

    class Codebook(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Embedding(1024, 2)
            self.out_proj = nn.utils.weight_norm(nn.Conv1d(2, 4, 1))

        def decode_code(self, codes):
            return self.table(codes).transpose(1, 2)

    quantizer = codec.ResidualVectorQuantize()
    quantizer.quantizers = nn.ModuleList([Codebook() for _ in range(9)])
    dac = codec.DAC()
    dac.quantizer = quantizer
    dac.decoder = nn.Conv1d(4, 1, 1)
    dia = module.Dia()
    dia.config = types.SimpleNamespace(data=types.SimpleNamespace(channels=9,
        delay_pattern=[0, 8, 9, 10, 11, 12, 13, 14, 15], audio_eos_value=1024, audio_pad_value=1025))
    dia.dac_model = dac

    class Decoder:
        def decode_step(self, tokens, state):
            logits = torch.arange(2 * 9 * 1028, dtype=torch.float32).reshape(2, 1, 9, 1028)
            return torch.sin(logits / 17.)

    dia.model = types.SimpleNamespace(decoder=Decoder())
    original_sampler = module._sample_next_token
    original_methods = (module.Dia._decoder_step, module.Dia._generate_output,
                        codec.ResidualVectorQuantize.from_codes, codec.DAC.decode)
    hook_signature = [(name, tuple(m._forward_pre_hooks.items())) for name, m in dac.named_modules()]
    raw = (torch.arange(26 * 9).reshape(26, 9) * 7) % 1024
    raw[17, 2], raw[15, 7] = 1025, -1

    def flow():
        tokens = torch.full((2, 1, 9), 1026, dtype=torch.int64)
        predictions = []
        for _ in range(5):
            pred = dia._decoder_step(tokens, None, 3., 1.3, .95, 35)
            predictions.append(pred.clone())
            pred[0] = 1024  # Actual generate mutates this result after sampling.
            tokens = pred.view(1, 1, 9).expand(2, -1, -1)
        return dia._generate_output(raw.clone()), torch.stack(predictions)

    outputs, states = [], []
    with torch.inference_mode():
        for phase in ['off0', 'on', 'off1']:
            torch.manual_seed(42)
            if phase == 'on':
                with observer.TrajectoryCapture(dia) as capture:
                    output, preds = flow()
            else:
                output, preds = flow()
            outputs.append((output.copy(), preds.clone()))
            states.append(torch.get_rng_state().clone())
            assert module._sample_next_token is original_sampler
            assert not {'_decoder_step', '_generate_output'} & vars(dia).keys()
            assert 'from_codes' not in vars(quantizer) and 'decode' not in vars(dac)
            assert hook_signature == [(name, tuple(m._forward_pre_hooks.items())) for name, m in dac.named_modules()]
    assert all(o.tobytes() == outputs[0][0].tobytes() and torch.equal(p, outputs[0][1]) for o, p in outputs)
    assert all(torch.equal(s, states[0]) for s in states)
    assert np.array_equal(np.stack(capture.steps['predictions']), outputs[1][1].numpy())
    assert np.array_equal(capture.arrays['raw_delayed_codes'], raw.numpy())
    codes = np.stack([raw.numpy()[np.arange(11) + d, c] for c, d in enumerate(dia.config.data.delay_pattern)])[None]
    codes[(codes < 0) | (codes > 1023)] = 0
    assert np.array_equal(capture.arrays['rvq_codes'], codes)
    assert np.array_equal(capture.arrays['rvq_zq'], capture.arrays['decoder_input'])
    assert np.array_equal(capture.arrays['decoder_output'].squeeze(), outputs[1][0])
    assert len(capture.steps['sampler_logits']) == 5
    assert all(np.isneginf(x[:, 1025:]).all() and np.isneginf(x[1:, 1024:]).all()
               for x in capture.steps['sampler_logits'])
    with tempfile.TemporaryDirectory() as folder:
        capture.save(Path(folder) / 'trajectory')
        assert len(list((Path(folder) / 'trajectory').glob('*.npy'))) == 9
    try:
        with observer.TrajectoryCapture(dia):
            raise RuntimeError('forced observer scope failure')
    except RuntimeError:
        pass
    assert module._sample_next_token is original_sampler
    assert not {'_decoder_step', '_generate_output'} & vars(dia).keys()
    assert 'from_codes' not in vars(quantizer) and 'decode' not in vars(dac)
    assert original_methods == (module.Dia._decoder_step, module.Dia._generate_output,
                               codec.ResidualVectorQuantize.from_codes, codec.DAC.decode)
    # Stored snapshots must remain independent of caller-owned tensors.
    old = capture.arrays['raw_delayed_codes'].copy()
    raw.zero_()
    assert np.array_equal(capture.arrays['raw_delayed_codes'], old)
    print(json.dumps(dict(status='PASS', phases=3, steps_per_phase=5,
        actual_original_methods=5, trajectory_arrays=9, pcm_rng_prediction_identity=True,
        invalid_code_clamp_verified=True, exception_restoration=True,
        original_weightnorm_hooks_preserved=9, neural_logits_and_codec_weights='tiny synthetic',
        full_model_acceptance=False)))


if __name__ == '__main__':
    main()
