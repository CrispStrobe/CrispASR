#!/usr/bin/env python3
"""Hosted full CPU acceptance of measured Granite5 precision profiles.

Fresh quantizer output is checked against the controlled diagnostic before
native execution. No CUDA or production release is established by this job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from granite5_execution import run, validate_case
from granite5_native_acceptance import sha

ROOT = Path(__file__).resolve().parents[2]
PRECISION_RUN = 39979238513
PRECISION_SOURCE = '7287199c319f0c6534e2d64732cf385f1c58fbdb'
NATIVE_RUN = 39970747786
NATIVE_SOURCE = 'c433b9f57347e357922770ad2b8f74c019ab70d0'
PRODUCER_RUN = 39964199536
PRODUCER_SOURCE = 'a02a0dd9299e0b759704f520fe3959f50a764f71'
STAGING = 'cstr/crispasr-granite5-validation-public'


def main():
    from gguf import GGUFReader
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', nargs='+', choices=['f16', 'q8_0', 'q6_k'], default=['f16', 'q8_0'])
    args = parser.parse_args()
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'granite5-profiles'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    run_id = int(os.environ['GITHUB_RUN_ID'])
    receipt = dict(source=source, run=run_id, complete=False, cpu_acceptance=False, cuda_acceptance=False,
                   production_release=False, profiles={}, requested_profiles=args.profiles)
    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    environment = dict(os.environ)

    def execute(command, label):
        return run(command, label, out, environment, 'cpu', timeout=7200)

    def artifact(run_id, expected_source, conclusion):
        meta = json.loads(subprocess.check_output(['gh', 'api', f'repos/CrispStrobe/CrispASR/actions/runs/{run_id}']))
        if not (meta['status'] == 'completed' and meta['conclusion'] == conclusion and
                meta['head_sha'] == expected_source and meta['event'] == 'workflow_dispatch'):
            raise ValueError(f'Unexpected input run {run_id}')
        directory = scratch / str(run_id)
        execute(['gh', 'run', 'download', str(run_id), '--repo', 'CrispStrobe/CrispASR',
                 '--name', f'heavy-{run_id}', '--dir', directory], f'input-{run_id}')
        result = json.loads((directory / 'receipt.json').read_text())
        if result['source'] != expected_source:
            raise ValueError('Artifact/source mismatch')
        return result

    precision = artifact(PRECISION_RUN, PRECISION_SOURCE, 'success')
    original = artifact(PRODUCER_RUN, PRODUCER_SOURCE, 'success')
    prior = artifact(NATIVE_RUN, NATIVE_SOURCE, 'failure')
    if not (precision['complete'] and precision['cases']['q8_0-embedding']['accepted'] and original['complete'] and
            not prior['cpu_acceptance'] and prior['producer_source'] == PRODUCER_SOURCE):
        raise ValueError('Input evidence is incomplete or misclassified')
    receipt['original_source'] = original['revision']
    receipt['reference'] = original['outputs']['jfk-ref.gguf']
    receipt['precision_run'] = PRECISION_RUN
    receipt['q4_disposition'] = 'Not accepted: every controlled head/embedding arm fails fixed numerical gates'
    save()

    def fetch(record):
        path = Path(hf_hub_download(record['repo'], record['path'], repo_type=record['repo_type'],
                                   revision=record['revision'], local_dir=scratch / 'inputs'))
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Input content hash mismatch')
        return path

    reference = fetch(receipt['reference'])
    f16_record = prior['models']['granite-speech-5.0-2b-f16']['file']
    f16 = fetch(f16_record)
    if not shutil.which('ccache') or not shutil.which('ninja'):
        execute(['sudo', 'apt-get', 'update'], 'apt-update')
        execute(['sudo', 'apt-get', 'install', '-y', 'ccache', 'ninja-build'], 'apt-tools')
    build = scratch / 'build'
    execute(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
             '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DCRISPASR_BUILD_SERVER=OFF',
             '-DCRISPASR_BUILD_TESTS=ON'], 'configure')
    execute(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-lib', 'crispasr-quantize',
             'granite5-stage-probe', '-j2'], 'build')
    bundle = scratch / 'cpu-runtime'
    bundle.mkdir()
    for name in ('crispasr', 'granite5-stage-probe'):
        (bundle / name).symlink_to(build / 'bin' / name)
    library = next(build.rglob('libcrispasr.so'))
    (bundle / 'libcrispasr.so').symlink_to(library)
    environment['LD_LIBRARY_PATH'] = ':'.join(sorted({str(p.parent) for p in build.rglob('*.so')}))
    instruction = original['reference_recipe']['prompt'].removeprefix('<|audio|>')
    receipt['instruction'] = instruction
    receipt['runtime_sources'] = {
        str(p.relative_to(ROOT)): sha(p) for directory in [ROOT / 'src', ROOT / 'examples/cli']
        for p in directory.rglob('*') if p.is_file() and p.suffix in {'.cpp', '.c', '.h', '.hpp'}}
    api = HfApi()
    card = api.model_info(STAGING).card_data
    if card['license'] != 'apache-2.0' or card['crispasr_native_acceptance'] is not False:
        raise ValueError('Diagnostic staging license/scope mismatch')
    save()
    for quant in args.profiles:
        model = f16 if quant == 'f16' else scratch / f'granite-speech-5.0-2b-{quant}.gguf'
        record = f16_record
        if quant != 'f16':
            execute([build / 'bin/crispasr-quantize', f16, model, quant], f'quantize-{quant}')
            record = dict(repo=STAGING, repo_type='model', path=f'profiles/{source}/{model.name}',
                          sha256=sha(model), bytes=model.stat().st_size)
            # Checkpoint before inference so later failures do not lose a produced model.
            try:
                commit = api.upload_file(path_or_fileobj=model, path_in_repo=record['path'], repo_id=STAGING,
                                         commit_message='Checkpoint unaccepted Granite5 precision profile')
            except Exception:
                fallback = out / 'failed-transfer'
                fallback.mkdir(exist_ok=True)
                shutil.copyfile(model, fallback / model.name)
                raise
            record['revision'] = commit.oid
        row = dict(file=record, accepted=False)
        receipt['profiles'][quant] = row
        save()
        if quant == 'q8_0':
            controlled = fetch(precision['cases']['q8_0-embedding']['checkpoint'])
            before, after = GGUFReader(controlled), GGUFReader(model)
            expected = {t.name: t for t in before.tensors}
            if len(expected) != len(after.tensors):
                raise ValueError('Fresh Q8 tensor count differs from measured controlled profile')
            for tensor in after.tensors:
                ref = expected[tensor.name]
                if not (tensor.tensor_type == ref.tensor_type and np.array_equal(tensor.shape, ref.shape) and
                        hashlib.sha256(tensor.data).digest() == hashlib.sha256(ref.data).digest()):
                    raise ValueError(f'Fresh Q8 tensor differs: {tensor.name}')
            row['controlled_tensor_payloads_exact'] = len(expected)
            del before, after, expected, tensor, ref
            controlled.unlink()
            save()
        row['validation'] = validate_case(model, reference, instruction, bundle, ROOT / 'python',
                                           ROOT / 'samples/jfk.wav', out / quant, 'cpu', environment, quant)
        row['accepted'] = True
        save()
        if quant != 'f16':
            model.unlink()
    receipt.update(complete=True, cpu_acceptance=True)
    save()
    api.upload_folder(folder_path=out, path_in_repo=f'profiles/{source}/runs/{run_id}/cpu-evidence', repo_id=STAGING)
    (out / 'summary.md').write_text('Granite5 requested profiles pass full original-reference CPU gates; CUDA pending.\n')


if __name__ == '__main__':
    main()
