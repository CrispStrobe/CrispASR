#!/usr/bin/env python3
"""Actual original CPU sampler, placement restoration and RNG controls; no weights."""
import argparse
import ast
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
    from dia.model import Dia, _sample_next_token
    from dia.config import DiaConfig
    from dia.state import DecoderOutput
    path = Path(__file__).resolve().parents[1] / 'tools/kaggle/dia-sampler-device/dia_sampler_device.py'
    spec = importlib.util.spec_from_file_location('worker', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    assert worker.ARMS == {
        'cuda-draw': dict(precision='float32', temperature=1.2, top_k=45, draw='input'),
        'cpu-draw': dict(precision='float32', temperature=1.2, top_k=45, draw='cpu'),
        'cuda-draw-anchor': dict(precision='float32', temperature=1.2, top_k=45, draw='input')}
    original = torch.multinomial
    logits = torch.sin(torch.arange(9 * 1028).reshape(9, 1028).float() * .03) * 4
    for draw in ['input', 'cpu']:
        for seed in [42, 123]:
            torch.manual_seed(seed)
            expected = [_sample_next_token(logits.clone(), 1.2, .95, 45) for _ in range(8)]
            expected_state = torch.get_rng_state().clone()
            torch.manual_seed(seed)
            with worker.draw_placement(torch, draw, source_device='cpu') as record:
                actual = [_sample_next_token(logits.clone(), 1.2, .95, 45) for _ in range(8)]
                assert torch.multinomial is not original
            assert torch.multinomial is original and record['calls'] == 8
            assert record['transfer_checks'] == (8 if draw == 'cpu' else 0)
            assert torch.equal(torch.stack(actual), torch.stack(expected))
            assert torch.equal(torch.get_rng_state(), expected_state)
            assert torch.equal(torch.stack(record['ids']).squeeze(-1), torch.stack(actual))
            assert all(torch.equal(p, record['probabilities'][0]) for p in record['probabilities'])
            # Returned IDs are mutated by original generate; retained IDs must own their data.
            actual[0].fill_(1024)
            assert torch.equal(record['ids'][0].squeeze(-1), expected[0])
    try:
        with worker.draw_placement(torch, 'cpu', source_device='cpu'):
            raise RuntimeError('intentional restoration check')
    except RuntimeError:
        pass
    assert torch.multinomial is original
    rejected = 0
    for bad in [torch.ones(8, 1028), torch.ones(9, 1028, dtype=torch.float64)]:
        try:
            with worker.draw_placement(torch, 'cpu', source_device='cpu'):
                torch.multinomial(bad, num_samples=1)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('invalid input accepted')
        assert torch.multinomial is original
    try:
        with worker.draw_placement(torch, 'invalid', source_device='cpu'):
            pass
    except AssertionError:
        rejected += 1
    assert rejected == 3 and torch.multinomial is original
    # Observe deletion of exception restoration failing, rather than only checking source strings.
    text = path.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'draw_placement')
    source = ast.get_source_segment(text, node)
    from contextlib import contextmanager
    ns = dict(contextmanager=contextmanager)
    exec('@contextmanager\n' + source.replace('torch.multinomial = original', 'pass'), ns)
    try:
        with ns['draw_placement'](torch, 'cpu', source_device='cpu'):
            raise RuntimeError('negative')
    except RuntimeError:
        pass
    assert torch.multinomial is not original
    torch.multinomial = original
    class Recorder:
        def generate(self, *args, **kwargs):
            assert args == ('fixed',)
            assert kwargs == dict(max_tokens=3072, use_torch_compile=False, temperature=1.2,
                                  cfg_filter_top_k=45, cfg_scale=3., top_p=.95)
    for arm in worker.ARMS.values():
        worker.generate_case(Recorder(), dict(text='fixed'), arm)
    config = DiaConfig.load(str(args.config))
    model = Dia.__new__(Dia)
    model.config = config
    model.device = torch.device('cpu')
    model.model = types.SimpleNamespace(eval=lambda: None)
    model.dac_model = types.SimpleNamespace(
        quantizer=types.SimpleNamespace(from_codes=lambda codes: (codes, None, None)),
        decode=lambda codes: torch.zeros(1, 1, codes.shape[-1] * 512))
    def prepare(text, audio, verbose):
        prefill, start = model._prepare_audio_prompt(audio)
        output = DecoderOutput.new(config, model.device)
        output.prefill(prefill, start)
        return types.SimpleNamespace(prepare_step=lambda *args: None), output
    model._prepare_generation = prepare
    lengths = []
    for sampled_eos in [True, False]:
        calls = []
        def step(*args):
            calls.append(1)
            ids = torch.full((9,), 568, dtype=torch.int64)
            if sampled_eos and len(calls) == 19:
                ids[0] = 1024
            return ids
        model._decoder_step = step
        pcm = model.generate('[S1] tiny', max_tokens=128, use_torch_compile=False)
        assert len(pcm) % 512 == 0 and len(calls) == len(pcm) // 512 + 16
        lengths.append((len(calls), len(pcm)))
    assert lengths == [(33, 8704), (127, 56832)]
    print('PASS original generate/delay-removal/codec-input frame count: sampled and capacity EOS', lengths)
    print('PASS actual original sampler: 32 draws, exact IDs/RNG, transfer counters, owned snapshots,')
    print('PASS exception restoration, 3 invalid-input negatives, broken-restoration negative, all recipe routes')
    print('NO neural weights or CUDA acceptance: actual device controls run before model downloads on GPU')


if __name__ == '__main__':
    main()
