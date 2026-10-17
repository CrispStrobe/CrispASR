#!/usr/bin/env python3
"""Actual original BOS generation boundary; no neural weights or speech claim."""
import argparse
import ast
from contextlib import contextmanager, nullcontext
import importlib.util
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blueprint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    sys.modules['dac'] = types.ModuleType('dac')
    sys.modules['torchaudio'] = types.ModuleType('torchaudio')
    sys.path.insert(0, str(args.blueprint))
    from dia.model import Dia
    from dia.state import DecoderOutput
    from dia.config import DiaConfig
    path = Path(__file__).resolve().parents[1] / 'tools/kaggle/dia-bos-boundary/dia_bos_boundary.py'
    spec = importlib.util.spec_from_file_location('worker', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    config = DiaConfig.load(str(args.config))
    model = Dia.__new__(Dia)
    model.config = config
    model.device = torch.device('cpu')
    model.model = types.SimpleNamespace(eval=lambda: None)
    outputs = []
    inputs = []
    calls = []
    def prepare(text, audio, verbose):
        prefill, start = model._prepare_audio_prompt(audio)
        assert prefill[15, 8].item() == config.data.audio_bos_value
        output = DecoderOutput.new(config, model.device)
        output.prefill(prefill, start)
        outputs.append(output)
        return types.SimpleNamespace(prepare_step=lambda *args: None), output
    model._prepare_generation = prepare
    model._generate_output = lambda codes: codes.cpu().numpy().copy()
    original = DecoderOutput.update_one
    def run(context, sampled_eos):
        calls.clear()
        inputs.clear()
        def step(tokens, *args):
            inputs.append(tokens[0, 0].clone())
            calls.append(1)
            prediction = torch.full((9,), 568, dtype=torch.int64)
            if sampled_eos and len(calls) == 19:
                prediction[0] = 1024
            return prediction
        model._decoder_step = step
        with context as record:
            codes = model.generate('[S1] tiny', max_tokens=128, use_torch_compile=False)
        assert DecoderOutput.update_one is original
        assert len(calls) == (33 if sampled_eos else 127)
        assert codes.shape == (len(calls) - 1, 9)
        return record
    # The unchanged source already overwrites its prepared final BOS.
    for sampled in [True, False]:
        run(nullcontext({}), sampled)
        plain_inputs = torch.stack(inputs).clone()
        plain_tokens = outputs[-1].generated_tokens.clone()
        baseline = run(worker.final_bos_boundary(torch, DecoderOutput, False, config.data), sampled)
        assert torch.equal(torch.stack(inputs), plain_inputs)
        assert torch.equal(outputs[-1].generated_tokens, plain_tokens)
        assert inputs[15][8].item() == 568
        assert baseline['before'][0, 8].item() == config.data.audio_bos_value
        assert baseline['after'][0, 8].item() == 568
        fixed = run(worker.final_bos_boundary(torch, DecoderOutput, True, config.data), sampled)
        assert fixed['target_calls'] == 1 and inputs[15][8].item() == config.data.audio_bos_value
        assert inputs[16][8].item() == 568
        assert torch.equal(inputs[15][:8], torch.full((8,), 568, dtype=torch.int32))
        # All explicitly prepared BOS slots survive; after their delay, predictions are used.
        table = outputs[-1].generated_tokens
        for book, delay in enumerate(config.data.delay_pattern):
            assert (table[:delay + 1, book] == config.data.audio_bos_value).all().item()
            assert table[delay + 1, book].item() == 568
    try:
        with worker.final_bos_boundary(torch, DecoderOutput, True, config.data):
            raise RuntimeError('intentional restoration')
    except RuntimeError:
        pass
    assert DecoderOutput.update_one is original
    text = path.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'final_bos_boundary')
    source = ast.get_source_segment(text, node)
    rejected = 0
    for before, after in [('bool(enabled) or apply_mask', 'apply_mask'), ('if step != last:', 'if step != last - 1:')]:
        assert before in source
        ns = dict(contextmanager=contextmanager)
        exec('@contextmanager\n' + source.replace(before, after), ns)
        try:
            run(ns['final_bos_boundary'](torch, DecoderOutput, True, config.data), True)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('changed boundary accepted')
        assert DecoderOutput.update_one is original
    assert rejected == 2
    assert list(worker.ARMS) == ['original', 'preserve-final-bos', 'original-anchor']
    assert [a['preserve_bos'] for a in worker.ARMS.values()] == [False, True, False]
    assert all(a['precision'] == 'float32' and a['draw'] == 'input' and a['temperature'] == 1.2 and a['top_k'] == 45 for a in worker.ARMS.values())
    print('PASS actual original generate/prepare/update: original overwrites row15/book8, intervention preserves it')
    print('PASS all prepared BOS slots, next valid feedback, sampled/capacity EOS, exact one-slot change, restoration')
    print('PASS removed-preservation and wrong-step negatives; original/changed/original arm routing')
    print('Neural steps and codec output are stubs. Original source behavior is already documented; no speech fix claimed.')


if __name__ == '__main__':
    main()
