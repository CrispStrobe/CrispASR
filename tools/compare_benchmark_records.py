#!/usr/bin/env python3
"""Compare retained paired benchmark records without running inference.

Each JSON input supplies identity (model_revision, precision, input_sha256,
settings, device), and samples with seconds and optional steps. Samples must be
interleaved by the producer under comparable load; this tool cannot prove that.
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

IDENTITY = ('model_revision', 'precision', 'input_sha256', 'settings', 'device')


def compare(left, right):
    for key in IDENTITY:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f'Unmatched or missing benchmark identity: {key}')
    if not re.fullmatch(r'[0-9a-f]{64}', left['input_sha256']):
        raise ValueError('input_sha256 must be a lowercase SHA-256 digest')
    for key in ('model_revision', 'precision', 'device'):
        if not isinstance(left[key], str) or not left[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    if not isinstance(left['settings'], dict):
        raise ValueError('settings must be an object')
    result = {}
    for name, record in [('left', left), ('right', right)]:
        samples = record.get('samples', [])
        if len(samples) < 3:
            raise ValueError('At least three measured samples per implementation are required')
        times = [float(row['seconds']) for row in samples]
        if any(not math.isfinite(value) or value <= 0 for value in times):
            raise ValueError('Timings must be finite and positive')
        per_step = []
        for row in samples:
            steps = row.get('steps')
            if steps is not None:
                if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
                    raise ValueError('Step counts must be positive integers')
                per_step.append(1000 * float(row['seconds']) / steps)
        if per_step and len(per_step) != len(samples):
            raise ValueError('Step counts must be present for every sample or none')
        result[name] = {'samples': len(times), 'median_seconds': statistics.median(times),
                        'min_seconds': min(times), 'max_seconds': max(times),
                        'median_ms_per_step': statistics.median(per_step) if per_step else None}
    if (result['left']['median_ms_per_step'] is None) != (result['right']['median_ms_per_step'] is None):
        raise ValueError('Step counts must be available for both implementations or neither')
    result['left_over_right_speed_ratio'] = result['right']['median_seconds'] / result['left']['median_seconds']
    result['limitations'] = 'Identity equality does not establish output quality, identical weights, or controlled hardware load.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(args.left.read_text()), json.loads(args.right.read_text())), indent=2))


if __name__ == '__main__':
    main()
