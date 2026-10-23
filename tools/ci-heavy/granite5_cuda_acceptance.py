#!/usr/bin/env python3
"""Actual Kaggle CUDA acceptance of source-pinned CPU-approved Granite5 models.

Uses prebuilt binaries and host NVIDIA libraries. No compiler or Torch required.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from granite5_cuda_preflight import run_preflight
from granite5_execution import validate_case
from granite5_native_acceptance import sha

ROOT = Path(__file__).resolve().parents[2]


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
    provenance = json.loads((args.bundle / 'provenance.json').read_text())
    if not (provenance['build_complete'] and not provenance['gpu_executed'] and
            not provenance['model_acceptance'] and provenance['host_provided_nvidia_libraries']):
        raise ValueError('Unexpected runtime provenance scope')
    actual_sources = {str(p.relative_to(ROOT)): sha(p)
                      for directory in [ROOT / 'src', ROOT / 'examples/cli'] for p in directory.rglob('*')
                      if p.is_file() and p.suffix in {'.cpp', '.c', '.h', '.hpp'}}
    if not (cpu['runtime_sources'] == provenance['runtime_sources'] == actual_sources):
        raise ValueError('CPU/GPU/runtime source payloads differ')
    for name, record in provenance['files'].items():
        path = args.bundle / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file() or
                not path.resolve().is_relative_to(args.bundle.resolve()) or
                sha(path) != record['sha256'] or path.stat().st_size != record['bytes']):
            raise ValueError(f'Runtime manifest mismatch: {name}')
    receipt = dict(complete=False, cuda_acceptance=False, production_release=False,
                   cpu_source=cpu['source'], runtime_source=provenance['source'],
                   cpu_receipt_sha256=sha(args.cpu_receipt), profiles={})
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
