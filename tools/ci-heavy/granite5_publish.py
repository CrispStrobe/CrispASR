#!/usr/bin/env python3
"""Hosted model publication after independently audited CPU and actual CUDA gates.

No inference/build occurs here. Large model transfer stays on the hosted runner.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

import requests

from granite5_native_acceptance import sha
from granite5_execution import gpu_matmul_nodes, gpu_decoder_matmul_nodes

ROOT = Path(__file__).resolve().parents[2]
STAGING = 'cstr/crispasr-granite5-validation-public'
DESTINATION = 'cstr/granite-speech-5.0-2b-GGUF'
ORIGINAL = 'ibm-granite/granite-speech-5.0-2b'
ORIGINAL_REVISION = 'c075cf19e041a2e4d16c5ee24763e5b037d8f479'
CPU_SOURCE = '1bad3b2dbe8b5ce55cc5dfafcec25610d442ee95'
CPU_REVISION = 'b99c15b3719518846a90402447cee58331470ed7'
CPU_SHA = 'bb8e17e1e4fc81fdb85bbdcab84ae91f01833d30d4da9135f2ea2af4874bfdea'
LEGAL_REVISION = '7601437936c5b406af3e00c3a549d913329605ac'


def readback(repo, revision, name, expected_sha, expected_bytes=None):
    from huggingface_hub import hf_hub_url
    # This deliberately sends no HF credentials; public availability is a gate.
    session = requests.Session()
    session.trust_env = False  # Do not inherit netrc credentials or proxy authentication.
    response = session.get(hf_hub_url(repo, name, revision=revision), stream=True, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f'Anonymous readback failed: HTTP {response.status_code}')
    digest, count = hashlib.sha256(), 0
    for block in response.iter_content(1024 * 1024):
        digest.update(block)
        count += len(block)
    response.close()
    session.close()
    if digest.hexdigest() != expected_sha or expected_bytes is not None and count != expected_bytes:
        raise ValueError('Anonymous content readback mismatch')
    return dict(sha256=digest.hexdigest(), bytes=count, anonymous_full_byte_readback=True)


def qualify(cpu, gpu, preflight, gpu_root, profiles=('f16', 'q8_0')):
    """Reject partial/stale records before creating a production repository."""
    if tuple(profiles) not in (('f16',), ('f16', 'q8_0')):
        raise ValueError('Declare either the F16 release or the fully qualified F16/Q8 release')
    if not (cpu['source'] == CPU_SOURCE and cpu['complete'] and cpu['cpu_acceptance'] and
            not cpu['cuda_acceptance'] and
            gpu['cpu_source'] == CPU_SOURCE and gpu['cpu_receipt_sha256'] == CPU_SHA):
        raise ValueError('Linked CPU and actual CUDA approval is required')
    if tuple(profiles) == ('f16', 'q8_0') and not (gpu['complete'] and gpu['cuda_acceptance']):
        raise ValueError('The entire F16/Q8 matrix must pass before publishing both profiles')
    if not (preflight['runtime_preflight_pass'] and preflight['gpu0']['uuid'] == gpu['hardware']['uuid']):
        raise ValueError('Loader/hardware evidence mismatch')
    cub = gpu.get('compiler_header_components', {}).get('CUB', {})
    if not (cub.get('version') == '2.3.2' and cub.get('license') == 'BSD-3-Clause' and
            cub.get('license_sha256') == '2e15eba2e130f3e80d9e91724c7dce3781ce99316324681ff54b00c1fc9759b3' and
            re.fullmatch('[0-9a-f]{64}', gpu.get('runtime_provenance_sha256', ''))):
        raise ValueError('CUDA proof must identify the licensing-complete runtime')
    if set(cpu['profiles']) != {'f16', 'q8_0'} or set(gpu['profiles']) != {'f16', 'q8_0'}:
        raise ValueError('Only the qualified F16/Q8 profiles may be published')
    rope = gpu['rope_parity']
    if not (rope['complete'] and rope['device'] == 'cuda' and len(rope['cases']) == 5):
        raise ValueError('Actual CUDA RoPE positive and all four controls are required')
    for index, case in enumerate(rope['cases']):
        values = [case[k] for k in ('maxabs', 'low_position_maxabs', 'norm_ratio', 'relative_l2')]
        numeric_pass = (all(math.isfinite(v) for v in values) and values[0] < .005 and
                        values[1] < 2e-5 and abs(values[2] - 1) < 1e-6 and values[3] < .001)
        if (case['mode'] != index or case['backend'] != 'CUDA0' or case['exit_code'] != (0 if index == 1 else 1) or
                not all(math.isfinite(v) for v in values) or numeric_pass != (index == 1)):
            raise ValueError('Wrong CUDA RoPE control outcomes')
    for quant in profiles:
        c, g = cpu['profiles'][quant], gpu['profiles'][quant]
        if not (c['accepted'] and c['validation']['complete'] and g['accepted'] and
                g['validation']['complete'] and g['validation']['accepted'] and
                g['validation']['device'] == 'cuda' and c['file'] == g['file']):
            raise ValueError('CPU/GPU weight or acceptance mismatch')
        for label in ('stages', 'session', 'cli'):
            execution = g['validation']['executions'][label]
            log = (gpu_root / quant / (label + '.log')).read_text()
            actual_nodes = gpu_matmul_nodes(log)
            decoder_nodes = gpu_decoder_matmul_nodes(log)
            if not (execution['returncode'] == 0 and execution['peak_gpu_mib'] >= 512 and execution['gpu_matmul_nodes'] > 0 and
                    execution['gpu_matmul_nodes'] == actual_nodes and decoder_nodes > 0 and
                    execution['gpu_decoder_matmul_nodes'] == decoder_nodes):
                raise ValueError('Actual per-process GPU0 work evidence is missing')


def main():
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-revision', required=True)
    parser.add_argument('--cuda-prefix', required=True)
    parser.add_argument('--cuda-receipt-sha', required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--profiles', nargs='+', choices=['f16', 'q8_0'], default=['f16', 'q8_0'])
    args = parser.parse_args()
    if not (re.fullmatch('[0-9a-f]{40}', args.cuda_revision) and
            re.fullmatch('[0-9a-f]{64}', args.cuda_receipt_sha)):
        raise ValueError('Immutable CUDA evidence revision and receipt hash are required')
    prefix = PurePosixPath(args.cuda_prefix)
    if prefix.is_absolute() or '..' in prefix.parts:
        raise ValueError('Invalid CUDA evidence prefix')
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH']) / 'granite5-release'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    cpu_path = Path(hf_hub_download(STAGING, f'profiles/{CPU_SOURCE}/cpu-evidence/receipt.json',
                                  revision=CPU_REVISION, local_dir=scratch / 'cpu'))
    if sha(cpu_path) != CPU_SHA:
        raise ValueError('Reviewed CPU receipt mismatch')
    cpu = json.loads(cpu_path.read_text())
    snapshot_download(STAGING, revision=args.cuda_revision, allow_patterns=[str(prefix) + '/*'],
                      local_dir=scratch / 'gpu')
    gpu_root = scratch / 'gpu' / prefix
    gpu_path = gpu_root / 'receipt.json'
    if sha(gpu_path) != args.cuda_receipt_sha:
        raise ValueError('Reviewed CUDA receipt mismatch')
    gpu = json.loads(gpu_path.read_text())
    preflight = json.loads((gpu_root / 'runtime-preflight.json').read_text())
    qualify(cpu, gpu, preflight, gpu_root, args.profiles)
    actual_sources = {str(p.relative_to(ROOT)): sha(p)
        for folder in [ROOT / 'src', ROOT / 'examples/cli'] for p in folder.rglob('*')
        if p.is_file() and p.suffix in {'.cpp', '.c', '.h', '.hpp'}}
    if actual_sources != cpu['runtime_sources']:
        raise ValueError('Native sources changed since CPU/CUDA validation')

    def fetch(record):
        path = Path(hf_hub_download(record['repo'], record['path'], repo_type=record['repo_type'],
                                   revision=record['revision'], local_dir=scratch / 'inputs'))
        if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('Qualified input bytes differ')
        return path

    reference = fetch(cpu['reference'])
    audit = out / 'independent-gpu-audit.json'
    subprocess.run([sys.executable, ROOT / 'tools/ci-heavy/granite5_result_audit.py', '--reference', reference,
                    '--results-root', gpu_root, '--quants', *args.profiles, '--output', audit], check=True)
    report = json.loads(audit.read_text())
    if not all(c['official_request_text_exact'] and c['original_vs_bucketed']['exact'] and
               c['original_vs_dynamic']['exact'] for c in report['cases'].values()):
        raise ValueError('Independent original transcript/token fidelity checks failed')
    api = HfApi()
    if api.model_info(ORIGINAL, revision=ORIGINAL_REVISION).card_data['license'] != 'apache-2.0':
        raise ValueError('Original pinned model license differs')
    assets = out / 'assets'
    assets.mkdir()
    for name in ('LICENSE', 'LICENSE-RECIPE.txt', 'NOTICE', 'original-publisher-README.md'):
        source = Path(hf_hub_download(STAGING, name, revision=LEGAL_REVISION, local_dir=scratch / 'legal'))
        shutil.copyfile(source, assets / name)
    if (assets / 'LICENSE-RECIPE.txt').read_bytes() != (ROOT / 'LICENSE').read_bytes():
        raise ValueError('Recipe MIT license differs from this source')
    upstream_card = Path(hf_hub_download(ORIGINAL, 'README.md', revision=ORIGINAL_REVISION,
                                       local_dir=scratch / 'upstream'))
    if sha(upstream_card) != sha(assets / 'original-publisher-README.md'):
        raise ValueError('Original publisher attribution/card differs')
    evidence = dict(original_model=ORIGINAL, original_revision=ORIGINAL_REVISION,
                    cpu_run=39984477044, cpu_source=CPU_SOURCE, cpu_revision=CPU_REVISION, cpu_receipt_sha256=CPU_SHA,
                    cuda_revision=args.cuda_revision, cuda_prefix=str(prefix), cuda_receipt_sha256=args.cuda_receipt_sha,
                    runtime_source=gpu['runtime_source'], hardware=gpu['hardware'],
                    runtime_provenance_sha256=gpu['runtime_provenance_sha256'],
                    compiler_header_components=gpu['compiler_header_components'],
                    profiles={q: cpu['profiles'][q]['file'] for q in args.profiles},
                    parent_cuda_complete=gpu['complete'], parent_cuda_acceptance=gpu['cuda_acceptance'],
                    cpu_acceptance=True, cuda_acceptance=True, production_release=False,
                    unaccepted_profiles=[q for q in ('q8_0', 'q4_k', 'q6_k') if q not in args.profiles],
                    qualification_scope='Only the explicitly selected profiles; rejected parent-matrix cases remain rejected',
                    native_context_limit=4096,
                    full_16k_context_qualified=False)
    (assets / 'validation-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')

    def card(complete):
        return f'''---
license: apache-2.0
base_model: {ORIGINAL}
pipeline_tag: automatic-speech-recognition
tags: [gguf, crispasr, granite-speech]
crispasr_native_acceptance: {str(complete).lower()}
---
# IBM Granite Speech 5.0 2B — CrispASR GGUF

Independent conversion of [{ORIGINAL}](https://huggingface.co/{ORIGINAL}/tree/{ORIGINAL_REVISION}).
Requires a CrispASR build with Granite5 YaRN metadata support.

Validated formats: {', '.join(args.profiles)}.
{'Q8_0 retains token embeddings at F16.' if 'q8_0' in args.profiles else 'Q8_0 is not qualified on CUDA: its pipeline-logit magnitude gate failed.'}
Q4_K and Q6_K are not qualified: their reference magnitude gates failed.
Original F32 stage comparisons, exact generated tokens, repeated C ABI Session,
and CLI transcripts pass on CPU and actual CUDA for the JFK English fixture.
The native context limit remains 4096. The separate rotary component was checked
through position 16383; full 16k-context inference is not qualified.
See validation-evidence.json for immutable inputs and test provenance.
Publication complete: {complete}. Qualification applies only to the files listed above.
{'The linked parent CUDA matrix also tested Q8_0; its failure remains in the evidence.' if 'q8_0' not in args.profiles else 'Both profiles passed the complete parent CUDA matrix.'}

Model weights: Apache-2.0 (LICENSE), inherited from the pinned IBM publisher.
Conversion recipes: MIT (LICENSE-RECIPE.txt). Preserve NOTICE and the original
publisher card. These scopes do not relicense upstream or runtime dependencies.
No runtime binaries or NVIDIA libraries are included in this repository.
'''

    (assets / 'README.md').write_text(card(False))
    if not args.publish:
        print('Qualified release assets prepared; no production repository modified')
        return
    api.create_repo(DESTINATION, exist_ok=True, private=False)
    if api.model_info(DESTINATION).private:
        raise ValueError('Existing private repository cannot be used as a public release destination')
    api.upload_folder(folder_path=assets, repo_id=DESTINATION,
                      commit_message='Prepare licensed Granite5 release assets; model transfer pending')
    initial = api.model_info(DESTINATION)
    if initial.card_data['license'] != 'apache-2.0' or initial.card_data['crispasr_native_acceptance'] is not False:
        raise ValueError('Pending-release license/scope card did not land before model transfer')
    published = {}
    for quant in args.profiles:
        record = cpu['profiles'][quant]['file']
        model = fetch(record)
        commit = api.upload_file(path_or_fileobj=model, path_in_repo=model.name, repo_id=DESTINATION,
                                 commit_message=f'Publish qualified Granite5 {quant} bytes')
        verified = readback(DESTINATION, commit.oid, model.name, record['sha256'], record['bytes'])
        published[quant] = dict(filename=model.name, revision=commit.oid, **verified)
        (out / 'transfer-checkpoints.json').write_text(json.dumps(published, indent=2) + '\n')
        model.unlink()  # Destination was independently verified first.
    evidence.update(production_release=True, published_files=published)
    (assets / 'validation-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (assets / 'README.md').write_text(card(True))
    final = api.upload_folder(folder_path=assets, repo_id=DESTINATION,
                              commit_message='Complete qualified Granite5 selected-profile release and provenance')
    info = api.model_info(DESTINATION, revision=final.oid)
    if info.card_data['license'] != 'apache-2.0' or not info.card_data['crispasr_native_acceptance']:
        raise ValueError('Production license/acceptance card did not land')
    for path in assets.iterdir():
        readback(DESTINATION, final.oid, path.name, sha(path), path.stat().st_size)
    (out / 'release.json').write_text(json.dumps(dict(repo=DESTINATION, revision=final.oid,
        complete=True, models=published, evidence=evidence), indent=2) + '\n')
    print('Qualified model release complete; registry changes remain a separate code change')


if __name__ == '__main__':
    main()
