#!/usr/bin/env python3
"""Prove bare crispasr-quantize reproduces the CPU/CUDA-qualified Granite5 Q4_K.

Hosted CPU only. Exact whole-file equality transfers the existing inference
qualification; this worker neither relaxes its gates nor publishes weights.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from granite5_cuda_acceptance import qualify_runtime
from granite5_execution import run
from granite5_native_acceptance import sha
from granite5_q4_cuda import approved_case
from granite5_q4_recipe import build_quantizer, RUNTIME, STAGING, tensor_inventory

ROOT = Path(__file__).resolve().parents[2]
CPU_SOURCE = '8fbb1794da4eea3851d8384934695d39b3518b72'
CPU_RUN = 40068653973
CPU_REVISION = 'e425e802e3a029edcf15d51cb31e863d59b5b0d2'
CPU_SHA = 'de60017c1e5d443539021124927a6c30d99bfa399d24ea018e274574ae9f53d7'


def main():
    from huggingface_hub import hf_hub_download
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'granite5-q4-default'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    environment = dict(os.environ)
    if any(k.startswith(('CRISPASR_GRANITE_', 'GRANITE_')) or k in
           ('CRISPASR_QUANT_LMHEAD', 'CRISPASR_IMATRIX_OUT', 'CRISPASR_ACTDUMP_OUT') for k in environment):
        raise ValueError('Bare-command proof requires an override-free environment')
    receipt = dict(complete=False, exact_qualified_file=False, production_release=False,
                   source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   run=int(os.environ['GITHUB_RUN_ID']), cpu_source=CPU_SOURCE, cpu_run=CPU_RUN,
                   cpu_revision=CPU_REVISION, cpu_receipt_sha256=CPU_SHA)

    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def fetch(record):
        path = Path(hf_hub_download(record['repo'], record['path'], revision=record['revision'],
                                   repo_type=record.get('repo_type', 'model'), local_dir=scratch / 'inputs'))
        if sha(path) != record['sha256'] or ('bytes' in record and path.stat().st_size != record['bytes']):
            raise ValueError('Immutable input differs')
        return path

    def execute(command, label):
        return run(command, label, out, environment, 'cpu', timeout=7200)

    save()
    cpu_path = fetch(dict(repo=STAGING, revision=CPU_REVISION, sha256=CPU_SHA,
        path=f'q4-recipes/{CPU_SOURCE}/runs/{CPU_RUN}/cpu-evidence/receipt.json'))
    cpu = json.loads(cpu_path.read_text())
    if cpu['source'] != CPU_SOURCE or cpu['run'] != CPU_RUN:
        raise ValueError('Unexpected CPU provenance')
    row = approved_case(cpu, 'value-down-f16')
    archive_path = fetch(RUNTIME)
    with tarfile.open(archive_path) as archive:
        archive.extractall(scratch / 'runtime', filter='data')
    bundle = scratch / 'runtime/bundle'
    provenance = qualify_runtime(bundle, cpu)
    ggml = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'ggml', text=True).strip()
    if ggml != provenance['ggml']:
        raise ValueError('Quantizer headers and qualified GGML source differ')
    receipt['build'] = dict(main_source_sha256=sha(ROOT / 'examples/crispasr-quantize/main.cpp'),
        helper_source_sha256=sha(ROOT / 'examples/common-ggml.cpp'), ggml_source=ggml,
        ggml_base_sha256=sha(bundle / 'libggml-base.so'), tensor_name_abi=128)
    save()
    environment['LD_LIBRARY_PATH'] = str(bundle)
    if not shutil.which('ccache') or not shutil.which('ninja'):
        execute(['sudo', 'apt-get', 'update'], 'apt-update')
        execute(['sudo', 'apt-get', 'install', '-y', 'ccache', 'ninja-build'], 'apt-tools')
    quantizer = build_quantizer(bundle, scratch, execute)
    f16 = fetch(cpu['f16'])
    candidate = scratch / 'granite-speech-5.0-2b-q4_k.gguf'
    execute([quantizer, f16, candidate, 'q4_k'], 'bare-q4-k')
    receipt['file'] = dict(bytes=candidate.stat().st_size, sha256=sha(candidate))
    receipt['expected_file'] = row['file']
    save()
    if receipt['file'] != {k: row['file'][k] for k in ('bytes', 'sha256')}:
        # Preserve unexpected expensive output for diagnosis, with the complete
        # existing license/attribution bundle. The already checkpointed accepted
        # file needs no duplicate transfer when whole-file identity matches.
        fallback = out / 'failed-byte-equivalence'
        fallback.mkdir()
        shutil.copyfile(candidate, fallback / candidate.name)
        legal = json.loads((ROOT / 'docs/historical/granite5-2026-10-23.json').read_text())['license']['assets']
        for name, identity in legal.items():
            companion = fetch(dict(repo=cpu['f16']['repo'], revision=cpu['f16']['revision'],
                                   path=name, **identity))
            shutil.copyfile(companion, fallback / name)
        raise ValueError('Bare quantizer output differs from the CPU/CUDA-qualified file')
    inventory = tensor_inventory(f16, candidate, row['rules'])
    (out / 'tensors.json').write_text(json.dumps(inventory, indent=2) + '\n')
    if inventory['elements_by_type'] != row['elements_by_type']:
        raise ValueError('Qualified precision inventory differs')
    receipt.update(complete=True, exact_qualified_file=True,
                   elements_by_type=inventory['elements_by_type'])
    save()
    print('Bare crispasr-quantize Q4_K exactly reproduces every qualified byte', flush=True)


if __name__ == '__main__':
    main()
