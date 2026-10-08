#!/usr/bin/env python3
"""Validate data2vec's frozen Python transcript target and actual regression.

No model/decoder/reference activation changes. Zero-WER upstream parity and
F16 stage gates remain hard; human WER is reported separately, not improved.
"""
import contextlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'data2vec-reference-parity'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    sys.path.insert(0, str(ROOT / 'tests/regression'))
    import run_one as regression
    manifest = json.loads((ROOT / 'tests/regression/manifest.json').read_text())
    entry = next(e for e in manifest['backends'] if e['name'] == 'data2vec-base')
    receipt = dict(passed=False, scope=__doc__, negative_controls={},
                   source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   provenance=entry['transcript_reference'])
    def save():
        (out / 'data2vec-reference-parity.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    reference = regression.hf_download(manifest['fixtures']['repo'], entry['fixture_ref_path'],
                                       manifest['fixtures']['revision'], scratch)
    sample = ROOT / entry['sample']
    decoded = regression.verify_transcript_reference(entry, reference, sample, scratch)
    receipt['independent_decoded_transcript'] = decoded
    receipt['human_wer'] = regression.compute_transcript_metrics(entry['human_reference_transcript'], decoded)[1]
    for name in ['wrong-target', 'reference-sha', 'vocab-sha', 'sample-sha']:
        broken = copy.deepcopy(entry)
        if name == 'wrong-target':
            broken['expected_transcript'] = entry['human_reference_transcript']
        elif name == 'reference-sha':
            broken['transcript_reference']['sha256'] = '0' * 64
        elif name == 'vocab-sha':
            broken['transcript_reference']['vocab']['sha256'] = '0' * 64
        else:
            broken['transcript_reference']['sample_sha256'] = '0' * 64
        try:
            regression.verify_transcript_reference(broken, reference, sample, scratch)
        except ValueError as error:
            receipt['negative_controls'][name] = str(error)
        else:
            raise AssertionError('Reference guard accepted ' + name)
        save()
    def run(args, tag):
        with (out / (tag + '.log')).open('wb') as f:
            result = subprocess.run(list(map(str, args)), cwd=ROOT, stdout=f,
                                    stderr=subprocess.STDOUT, timeout=3600)
        print(tag, result.returncode, (out / (tag + '.log')).read_text()[-2000:], flush=True)
        assert result.returncode == 0, tag
    run(['bash', ROOT / 'tools/ci-apt.sh', 'update'], 'apt-update')
    run(['bash', ROOT / 'tools/ci-apt.sh', 'install', '-y', 'libopenblas-dev', 'libespeak-ng-dev', 'espeak-ng', 'pkg-config'], 'apt-install')
    build = scratch / 'build'
    run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
         '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_EXAMPLES=ON', '-DCRISPASR_BUILD_SERVER=OFF'], 'configure')
    run(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-diff', '-j4'], 'build')
    (out / 'CMakeCache.txt').write_bytes((build / 'CMakeCache.txt').read_bytes())
    with (out / 'regression.log').open('w') as log, contextlib.redirect_stdout(log):
        failures = regression.regression_for('data2vec-base', manifest, scratch,
                                              build / 'bin/crispasr', build / 'bin/crispasr-diff')
    print((out / 'regression.log').read_text(), flush=True)
    receipt.update(failures=failures, passed=failures == 0)
    save()
    assert receipt['passed'], 'Actual data2vec CLI/stage regression failed'
    (out / 'summary.md').write_text('data2vec frozen Python transcript provenance, four negative controls, actual Q4 CLI zero-WER upstream parity and F16 stage regression PASS. Human WER remains 4.55%.\n')


if __name__ == '__main__':
    main()
