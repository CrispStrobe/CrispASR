#!/usr/bin/env python3
"""Calibrate current-main speech grid/scoring/scale gates with actual original data."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    path = Path(__file__).parents[1]/'dia_cache_sampler.py'
    spec = importlib.util.spec_from_file_location('worker', path)
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    assert worker.sha(args.canonical) == '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542'
    wanted = {'decoder_codes_rev', 'decoder_input_tokens'}
    original = {t.name: np.array(t.data).reshape(tuple(reversed(t.shape)))
                for t in GGUFReader(str(args.canonical)).tensors if t.name in wanted}
    inputs = original['decoder_input_tokens'].astype('<i4')
    grid = worker.grid_from_inputs(inputs)
    assert grid.shape == (1210, 9) and np.array_equal(grid, original['decoder_codes_rev'])
    rejects = 0
    bad_token = inputs.copy(); bad_token[15, 0] = 1027
    for bad in [inputs.astype('f4'), inputs[:16], np.tile(inputs[0], (3072, 1)), np.zeros_like(inputs), bad_token]:
        try: worker.grid_from_inputs(bad)
        except AssertionError: rejects += 1
        else: raise AssertionError('Invalid input accepted')
    assert worker.score('We are committed to making it secure.') == 0
    assert worker.score('') == 1
    assert worker.score('We are committed to making it safer.') == 1/7
    vector = np.array([1., 2., 3.])
    assert worker.metric('test', 'identity', vector, vector)['pass']
    scaled = worker.metric('test', 'scaled', vector*2, vector)
    assert np.isclose(scaled['cosine'], 1) and not scaled['pass']
    assert not worker.metric('test', 'reversed', -vector, vector)['pass']
    for bad in [np.zeros(3), np.array([np.nan, 2, 3]), np.ones(2)]:
        try: worker.metric('test', 'invalid', bad, vector)
        except AssertionError: rejects += 1
        else: raise AssertionError('Invalid metric accepted')
    args.output.write_text(json.dumps(dict(worker_sha256=worker.sha(path), canonical_sha256=worker.sha(args.canonical),
        actual_original_inputs=len(inputs), actual_original_code_frames=len(grid), all_original_codes_exact=True,
        invalid_input_and_metric_rejects=rejects, asr_controls=3, scale_blind_cosine_rejected=True,
        sign_reversal_rejected=True, full_model_acceptance=False), indent=2)+'\n')
    print('DIA_MAIN_SPEECH_SELFTEST_PASS: actual1210 original frames, eight invalid inputs/metrics, ASR and scale guards')


if __name__ == '__main__':
    main()
