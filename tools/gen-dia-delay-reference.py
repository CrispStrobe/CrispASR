#!/usr/bin/env python3
"""Capture the pinned official Dia raw-token -> DAC-code boundary without weights."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import types

PIN = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Official Dia checkout at the pinned commit')
    parser.add_argument('--output', type=Path, default=ROOT / 'tests/fixtures/dia-delay-reference.json')
    args = parser.parse_args()
    import torch
    import numpy as np
    assert subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    model_path = args.source / 'dia/model.py'
    audio_path = args.source / 'dia/audio.py'
    for path in (model_path, audio_path):
        assert path.read_bytes() == subprocess.check_output([
            'git', '-C', str(args.source), 'show', f'{PIN}:{path.relative_to(args.source)}'])
    spec = importlib.util.spec_from_file_location('official_dia_audio', audio_path)
    audio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audio)
    model_ast = ast.parse(model_path.read_text())
    dia_class = next(n for n in model_ast.body if isinstance(n, ast.ClassDef) and n.name == 'Dia')
    method = next(n for n in dia_class.body if isinstance(n, ast.FunctionDef) and n.name == '_generate_output')
    captured = []

    def capture_decode(_model, codes):
        # Capture the real codec input, not a simulated waveform or neural output.
        captured.append(codes.detach().cpu().clone())
        return torch.zeros(1, 1, 1)

    env = dict(vars(audio), decode=capture_decode, np=np)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(model_path), 'exec'), env)
    delay = [0, 8, 9, 10, 11, 12, 13, 14, 15]
    instance = types.SimpleNamespace(config=types.SimpleNamespace(data=types.SimpleNamespace(
        channels=9, delay_pattern=delay, audio_pad_value=1025)), dac_model=None)
    ordinary = torch.arange(19 * 9, dtype=torch.int32).reshape(19, 9)
    examples = {'valid': ordinary.clone(), 'eos': ordinary.clone(), 'bos_pad': ordinary.clone(),
                'unwritten': ordinary.clone(), 'all_invalid': torch.full_like(ordinary, 1024),
                'boundary': ordinary.clone(), 'delay_only': ordinary[:15].clone(),
                'one_frame': ordinary[:16].clone()}
    examples['eos'][0, 0] = 1024
    examples['bos_pad'][8, 1] = 1026
    examples['bos_pad'][10, 2] = 1025
    examples['unwritten'][15, 8] = -1
    examples['boundary'][0, 0] = 1023
    examples['boundary'][1, 0] = 1024
    rows = []
    for name, raw in examples.items():
        captured.clear()
        env['_generate_output'](instance, raw)
        assert len(captured) == 1
        expected = captured[0][0].transpose(0, 1)
        assert expected.shape == (max(0, len(raw) - 15), 9)
        rows.append({'name': name, 'raw': raw.tolist(), 'expected': expected.tolist()})
    fixture = {'source_pin': PIN, 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
               'audio_sha256': hashlib.sha256(audio_path.read_bytes()).hexdigest(),
               'boundary': 'Official Dia._generate_output input to decode; no weights, no PCM claim',
               'delay_pattern': delay, 'audio_vocab_size': 1024, 'cases': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + '\n')
    print(f'Captured {len(rows)} official CPU code-boundary cases at {PIN}')


if __name__ == '__main__':
    main()
