#!/usr/bin/env python3
"""Actual Kaggle CUDA acceptance of source-pinned CPU-approved Granite5 models.

Uses prebuilt binaries and host NVIDIA libraries. No compiler or Torch required.
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys

from granite5_cuda_preflight import run_preflight
from granite5_execution import resources, validate_case
from granite5_native_acceptance import sha

ROOT = Path(__file__).resolve().parents[2]

ORACLE = dict(repo='cstr/crispasr-regression-fixtures', repo_type='dataset',
    path='granite-speech-5.0-2b/a02a0dd9299e0b759704f520fe3959f50a764f71/original-yarn.npz',
    revision='49334e0bb83a5f04c33d5def4251b6efd2f059d9', bytes=24054,
    sha256='4589e1a785fff344f75054530a282f01e4150ceb1d337f43d26f5c226b3859ae')
ORACLE_METADATA = dict(repo='cstr/crispasr-granite5-validation-public', repo_type='model',
    path='granite-speech-5.0-2b/a02a0dd9299e0b759704f520fe3959f50a764f71/evidence/original-yarn.npz.json',
    revision='99bc8384b8ce3807731e1c13a9e96de391fa22a0', bytes=431,
    sha256='6d52e2799d0c0d5e547547d0a5aabff7443ef80b81bae3c46bcfda9537922a71')


def qualify_runtime(bundle, cpu):
    provenance = json.loads((bundle / 'provenance.json').read_text())
    if not (provenance['build_complete'] and not provenance['gpu_executed'] and
            not provenance['model_acceptance'] and provenance['host_provided_nvidia_libraries']):
        raise ValueError('Unexpected runtime provenance scope')
    actual_sources = {str(p.relative_to(ROOT)): sha(p)
                      for directory in [ROOT / 'src', ROOT / 'examples/cli'] for p in directory.rglob('*')
                      if p.is_file() and p.suffix in {'.cpp', '.c', '.h', '.hpp'}}
    if not (cpu['runtime_sources'] == provenance['runtime_sources'] == actual_sources):
        raise ValueError('CPU/GPU/runtime source payloads differ')
    expected_probes = {name: sha(ROOT / 'tools/ci-heavy' / name)
                       for name in ('granite5_stage_probe.cpp', 'granite5_rope_probe.cpp')}
    if provenance.get('probe_sources') != expected_probes:
        raise ValueError('CUDA capsule probe sources differ from the checked-out validators')
    for name, record in provenance['files'].items():
        path = bundle / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file() or
                not path.resolve().is_relative_to(bundle.resolve()) or
                sha(path) != record['sha256'] or path.stat().st_size != record['bytes']):
            raise ValueError(f'Runtime manifest mismatch: {name}')
    cub = provenance.get('compiler_header_components', {}).get('CUB', {})
    if not (cub.get('version') == '2.3.2' and cub.get('license') == 'BSD-3-Clause' and
            cub.get('notice') == 'licenses/CUB-2.3.2.txt' and
            cub.get('license_sha256') == '2e15eba2e130f3e80d9e91724c7dce3781ce99316324681ff54b00c1fc9759b3' and
            sha(bundle / cub['notice']) == cub['license_sha256']):
        raise ValueError('The compiled CUB templates require their verified BSD notice')
    return provenance


def main():
    from huggingface_hub import hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'cpu-receipt', 'scratch', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    cpu = json.loads(args.cpu_receipt.read_text())
    if not (cpu['complete'] and cpu['cpu_acceptance'] and not cpu['cuda_acceptance'] and
            not cpu['production_release'] and set(cpu['profiles']) == {'f16', 'q8_0'} and
            all(row['accepted'] and row['validation']['accepted'] for row in cpu['profiles'].values())):
        raise ValueError('A complete F16/Q8 CPU acceptance receipt is required')
    provenance = qualify_runtime(args.bundle, cpu)
    receipt = dict(complete=False, cuda_acceptance=False, production_release=False,
                   cpu_source=cpu['source'], runtime_source=provenance['source'],
                   cpu_receipt_sha256=sha(args.cpu_receipt),
                   runtime_provenance_sha256=sha(args.bundle / 'provenance.json'),
                   compiler_header_components=provenance['compiler_header_components'], profiles={})
    def save():
        (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    # Select one physical device consistently for loader evidence, backend and
    # per-child residency sampling. No second-GPU fallback is permitted.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    preflight = run_preflight(args.bundle, args.out / 'runtime-preflight.json')
    if not preflight['runtime_preflight_pass']:
        raise ValueError('CUDA loader preflight did not pass')
    environment = dict(os.environ, LD_LIBRARY_PATH=preflight['ld_library_path'],
                       GGML_BACKEND_PATH=str(args.bundle), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    # Inherited graph/quant overrides would invalidate production-path parity.
    forbidden = [name for name in environment if
                 name.startswith(('CRISPASR_GRANITE_', 'GRANITE_')) or name == 'CRISPASR_N_GPU_LAYERS']
    if forbidden:
        raise ValueError(f'Unexpected inherited Granite overrides: {forbidden}')
    cap = float(preflight['gpu0']['compute_capability'])
    receipt['hardware'] = preflight['gpu0']
    receipt['integer_mmq_hardware_capable'] = cap >= 6.1
    receipt['mmq_limit'] = 'Hardware capability only; no isolated integer-MMQ A/B claim'
    save()

    def fetch(record):
        path = Path(hf_hub_download(record['repo'], record['path'], repo_type=record['repo_type'],
                                   revision=record['revision'], local_dir=args.scratch / 'inputs'))
        if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('CPU-approved artifact hash/size mismatch')
        return path

    # Exercise the actual shared Q/K attention path through original position
    # 16383, including four incorrect controls, before loading trained weights.
    oracle = fetch(ORACLE)
    metadata = fetch(ORACLE_METADATA)
    shutil.copyfile(metadata, Path(str(oracle) + '.json'))
    snapshot = resources('cuda')
    receipt['rope_resources'] = snapshot
    save()
    if (snapshot['memory_available_kib'] < 5 * 1024 * 1024 or
            snapshot['load_average'][0] > 2 * snapshot['cpus']):
        raise RuntimeError('Insufficient headroom before CUDA RoPE component inference')
    with (args.out / 'rope-parity.log').open('w') as log:
        subprocess.run([sys.executable, str(ROOT / 'tools/ci-heavy/granite5_rope_parity.py'),
                        '--probe', str(args.bundle / 'granite5-rope-probe'), '--oracle', str(oracle),
                        '--out', str(args.out / 'rope-parity'), '--device', 'cuda'],
                       env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
    receipt['rope_parity'] = json.loads((args.out / 'rope-parity/receipt.json').read_text())
    if not (receipt['rope_parity']['complete'] and receipt['rope_parity']['device'] == 'cuda'):
        raise ValueError('CUDA RoPE controls were not completed')
    save()
    reference = fetch(cpu['reference'])
    for quant in ('f16', 'q8_0'):
        record = cpu['profiles'][quant]['file']
        model = fetch(record)
        receipt['profiles'][quant] = dict(file=record, accepted=False)
        save()
        validation = validate_case(model, reference, cpu['instruction'], args.bundle, args.bundle / 'python',
                                   ROOT / 'samples/jfk.wav', args.out / quant, 'cuda', environment, quant)
        receipt['profiles'][quant].update(accepted=True, validation=validation)
        save()
        model.unlink()
    receipt.update(complete=True, cuda_acceptance=True)
    save()
    print('GRANITE5_CUDA_ACCEPTED: F16/Q8 original stage, decoder, Session and CLI gates passed', flush=True)


if __name__ == '__main__':
    main()
