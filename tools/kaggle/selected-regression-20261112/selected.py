#!/usr/bin/env python3
"""Run three extended regression entries on a pinned CrispASR source."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = "853f04f09ebf8b0b574c79c0b95126eabdf7a2d9"
BACKENDS = ("tada-tts-3b-ml", "parler-tts-mini-v1.1", "kugelaudio-0-open")


def cleanup_workdir(workdir):
    """Keep the results and logs listable without publishing build/cache files."""
    for name in ('CrispASR', '_bootstrap', '.ccache', 'hf_cache', 'build', 'rebake-stage'):
        shutil.rmtree(workdir / name, ignore_errors=True)


def main():
    workdir = Path('/kaggle/working')
    bootstrap = workdir / '_bootstrap'
    try:
        subprocess.check_call(['git', 'clone', '--depth', '1', '--branch', 'main',
                               'https://github.com/CrispStrobe/CrispASR.git', str(bootstrap)])
        sys.path.insert(0, str(bootstrap / 'tools/kaggle'))
        import kaggle_harness as kh

        token = kh.resolve_hf_token()
        if token:
            os.environ['HF_TOKEN'] = token
        os.environ.update(CRISPASR_REGRESSION_MODE='validate', CRISPASR_REGRESSION_BUILD='cpu',
                          CRISPASR_REF=SOURCE, CRISPASR_REGRESSION_BACKENDS=','.join(BACKENDS))
        print('Pinned source:', SOURCE, 'selected:', ','.join(BACKENDS), flush=True)
        rc = subprocess.call([sys.executable, str(bootstrap / 'tools/kaggle/crispasr-regression.py')])
        files = sorted((workdir / 'results').glob('results-validate-*.jsonl'))
        if rc or len(files) != 1:
            raise SystemExit(f'regression failed or result missing: rc={rc}, files={len(files)}')
        rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
        if len(rows) != len(BACKENDS) or {r.get('backend') for r in rows} != set(BACKENDS):
            raise SystemExit('selected regression did not report exactly the three requested backends')
        print('SELECTED_REGRESSION_RESULTS', json.dumps(rows, sort_keys=True), flush=True)
    finally:
        cleanup_workdir(workdir)


if __name__ == '__main__':
    main()
