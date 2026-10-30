#!/usr/bin/env python3
"""Publish an independently audited Q4 recipe while preserving the F16 release.

No inference or build; --publish is required for any repository mutation.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

from granite5_execution import gpu_matmul_nodes, gpu_decoder_matmul_nodes
from granite5_native_acceptance import sha
from granite5_publish import readback, STAGING, DESTINATION
from granite5_q4_cuda import approved_case
from granite5_q4_recipe import CALIBRATION, MANIFEST

ROOT = Path(__file__).resolve().parents[2]
BASE = '701f5529d8459412cc1b6053b5880c630125b138'
CANONICAL = 'granite-speech-5.0-2b-q4_k.gguf'


def immutable(value, length):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{' + str(length) + '}', value):
        raise ValueError('Immutable revision/content hash required')


def prefix_path(value):
    path = PurePosixPath(value)
    if (not value or str(path) == '.' or path.is_absolute() or '..' in path.parts or
            any(character in value for character in '*?[]\\')):
        raise ValueError('Invalid evidence prefix')
    return path


def qualify(cpu, gpu, preflight, gpu_root, *, source, run, receipt_sha, case, actual_sources):
    """Pure receipt/log gates; independent array replay is additionally mandatory."""
    row = approved_case(cpu, case)
    immutable(row['file']['revision'], 40)
    immutable(row['file']['sha256'], 64)
    if not (row['file']['repo'] == STAGING and row['file']['repo_type'] == 'model' and row['file']['bytes'] > 0):
        raise ValueError('An immutable staged candidate is required')
    selected = gpu['profiles']['q4_k']
    if not (cpu['source'] == source and cpu['run'] == run and
            cpu['runtime_sources'] == actual_sources and len(actual_sources) == 700 and
            gpu['complete'] and gpu['cuda_acceptance'] and not gpu['production_release'] and
            gpu['selected_recipe'] == case and gpu['cpu_source'] == source and gpu['cpu_run'] == run and
            gpu['cpu_receipt_sha256'] == receipt_sha and set(gpu['profiles']) == {'q4_k'} and
            selected['accepted'] and selected['file'] == row['file'] and selected['rules'] == row['rules'] and
            gpu['profiles']['q4_k']['validation']['complete'] and
            gpu['profiles']['q4_k']['validation']['accepted'] and
            gpu['profiles']['q4_k']['validation']['device'] == 'cuda'):
        raise ValueError('Linked complete CPU/CUDA Q4 approval and actual runtime sources required')
    immutable(gpu['runtime_source'], 40)
    immutable(gpu['runtime_provenance_sha256'], 64)
    cub = gpu['compiler_header_components']['CUB']
    if not (cub['version'] == '2.3.2' and cub['license'] == 'BSD-3-Clause' and
            cub['notice'] == 'licenses/CUB-2.3.2.txt' and
            cub['license_sha256'] == '2e15eba2e130f3e80d9e91724c7dce3781ce99316324681ff54b00c1fc9759b3'):
        raise ValueError('Verified compiled CUB licensing required')
    if not (preflight['runtime_preflight_pass'] and preflight['gpu0']['uuid'] == gpu['hardware']['uuid']):
        raise ValueError('Loader and actual hardware mismatch')
    for validation, device in ((row['validation'], 'cpu'), (selected['validation'], 'cuda')):
        if not (validation['device'] == device and validation['prompt_ids_exact'] and
                validation['bucketed']['exact'] and validation['dynamic']['exact']):
            raise ValueError('Complete exact prompt and both decoder-path gates required')
    rope = gpu['rope_parity']
    if not (rope['complete'] and rope['device'] == 'cuda' and len(rope['cases']) == 5):
        raise ValueError('All five actual CUDA rotary controls required')
    for index, control in enumerate(rope['cases']):
        values = [control[k] for k in ('maxabs', 'low_position_maxabs', 'norm_ratio', 'relative_l2')]
        finite = all(math.isfinite(v) for v in values)
        passed = (finite and values[0] < .005 and values[1] < 2e-5 and
                  abs(values[2] - 1) < 1e-6 and values[3] < .001)
        if not (finite and control['mode'] == index and control['backend'] == 'CUDA0' and
                control['exit_code'] == (0 if index == 1 else 1) and passed == (index == 1)):
            raise ValueError('Incorrect CUDA rotary control outcome')
    for label in ('stages', 'session', 'cli'):
        execution = selected['validation']['executions'][label]
        log = (gpu_root / 'q4_k' / (label + '.log')).read_text()
        nodes, decoder = gpu_matmul_nodes(log), gpu_decoder_matmul_nodes(log)
        if not (execution['returncode'] == 0 and execution['peak_gpu_mib'] >= 512 and
                nodes > 0 and decoder > 0 and execution['gpu_matmul_nodes'] == nodes and
                execution['gpu_decoder_matmul_nodes'] == decoder):
            raise ValueError('Actual per-process CUDA decoder work and residency required')
    return row


def extend_evidence(baseline, acceptance):
    result = copy.deepcopy(baseline)
    if 'q4_acceptance' in result:
        raise ValueError('Q4 already recorded; refuse to replace previous evidence')
    result['parent_unaccepted_profiles'] = copy.deepcopy(result['unaccepted_profiles'])
    result['unaccepted_profiles'] = [q for q in result['unaccepted_profiles'] if q != 'q4_k']
    result['qualified_profiles'] = ['f16', 'q4_k']
    result['q4_acceptance'] = acceptance
    return result


def check_independent(report):
    if not (report['structural_numerical_content_pass'] and set(report['cases']) == {'q4_k'}):
        raise ValueError('Independent Q4 stage and magnitude audit failed')
    case = report['cases']['q4_k']
    if not (case['checks_complete'] and case['official_request_text_exact'] and
            case['original_vs_bucketed']['exact'] and case['original_vs_dynamic']['exact']):
        raise ValueError('Independent original tokens and official transcript mismatch')


def main():
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('cpu-source', 'cpu-revision', 'cpu-receipt-sha', 'case',
                  'cuda-revision', 'cuda-prefix', 'cuda-receipt-sha'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--cpu-run', required=True, type=int)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    for value in (args.cpu_source, args.cpu_revision, args.cuda_revision):
        immutable(value, 40)
    for value in (args.cpu_receipt_sha, args.cuda_receipt_sha):
        immutable(value, 64)
    if args.cpu_run <= 0:
        raise ValueError('Positive immutable CPU run required')
    prefix = prefix_path(args.cuda_prefix)
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'granite5-q4-release'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    api = HfApi()
    cpu_prefix = f'q4-recipes/{args.cpu_source}/runs/{args.cpu_run}/cpu-evidence'
    def download(repo, path, revision, directory, repo_type='model'):
        return Path(hf_hub_download(repo, path, revision=revision, repo_type=repo_type,
                                   local_dir=scratch / directory))
    cpu_path = download(STAGING, cpu_prefix + '/receipt.json', args.cpu_revision, 'cpu')
    if sha(cpu_path) != args.cpu_receipt_sha:
        raise ValueError('Reviewed CPU receipt differs')
    cpu = json.loads(cpu_path.read_text())
    snapshot_download(STAGING, revision=args.cuda_revision, allow_patterns=[str(prefix) + '/*'],
                      local_dir=scratch / 'gpu')
    gpu_root = scratch / 'gpu' / prefix
    if sha(gpu_root / 'receipt.json') != args.cuda_receipt_sha:
        raise ValueError('Reviewed CUDA receipt differs')
    gpu = json.loads((gpu_root / 'receipt.json').read_text())
    actual = {str(p.relative_to(ROOT)): sha(p) for folder in ('src', 'examples/cli')
              for p in (ROOT / folder).rglob('*')
              if p.is_file() and p.suffix in {'.cpp', '.c', '.h', '.hpp'}}
    row = qualify(cpu, gpu, json.loads((gpu_root / 'runtime-preflight.json').read_text()), gpu_root,
                  source=args.cpu_source, run=args.cpu_run, receipt_sha=args.cpu_receipt_sha,
                  case=args.case, actual_sources=actual)
    # Verify the runtime build's Git tree as well as the publisher's actual checkout.
    # Hosted publication checkouts are shallow; fetch this validated immutable
    # object explicitly rather than assuming its ancestors are already present.
    subprocess.run(['git', 'fetch', '--depth=1', 'origin', gpu['runtime_source']], cwd=ROOT, check=True)
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', gpu['runtime_source'],
                                     '--', 'src', 'examples/cli'], cwd=ROOT, text=True).splitlines()
    source_paths = {name for name in paths if Path(name).suffix in {'.cpp', '.c', '.h', '.hpp'}}
    if source_paths != set(actual):
        raise ValueError('CUDA runtime source inventory differs from the actual checkout')
    for name, digest in actual.items():
        payload = subprocess.check_output(['git', 'show', gpu['runtime_source'] + ':' + name], cwd=ROOT)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('CUDA runtime Git source differs from validated CPU source')
    record = cpu['reference']
    reference = download(record['repo'], record['path'], record['revision'], 'reference', record['repo_type'])
    if sha(reference) != record['sha256'] or reference.stat().st_size != record['bytes']:
        raise ValueError('Original reference bytes differ')
    audit = out / 'independent-q4-audit.json'
    subprocess.run([sys.executable, ROOT / 'tools/ci-heavy/granite5_result_audit.py', '--reference', reference,
                    '--results-root', gpu_root, '--quants', 'q4_k', '--output', audit], check=True)
    check_independent(json.loads(audit.read_text()))
    original = json.loads((ROOT / 'docs/historical/granite5-2026-10-23.json').read_text())
    if original['release']['revision'] != BASE or original['release']['repo'] != DESTINATION:
        raise ValueError('Expected accepted F16 release provenance required')
    if cpu['reference'] != original['reference'] or cpu['instruction'] != 'transcribe the speech with proper punctuation and capitalization.':
        raise ValueError('The accepted original reference and official request must remain unchanged')
    info = api.model_info(DESTINATION, revision=BASE, files_metadata=True)
    if info.private or info.card_data['license'] != 'apache-2.0' or info.card_data['crispasr_native_acceptance'] is not True:
        raise ValueError('Accepted public Apache F16 release required')
    f16 = original['release']['models']['f16']
    sibling = next(s for s in info.siblings if s.rfilename == f16['filename'])
    if sibling.size != f16['bytes'] or sibling.lfs.sha256 != f16['sha256']:
        raise ValueError('Existing F16 model identity differs')
    assets = out / 'assets'
    assets.mkdir()
    baseline_files = {}
    for name, identity in original['license']['assets'].items():
        source = download(DESTINATION, name, BASE, 'baseline')
        if sha(source) != identity['sha256'] or source.stat().st_size != identity['bytes']:
            raise ValueError('Accepted F16 legal/provenance asset differs')
        baseline_files[name] = source
        shutil.copyfile(source, assets / name)
    if (assets / 'LICENSE-RECIPE.txt').read_bytes() != (ROOT / 'LICENSE').read_bytes():
        raise ValueError('Recipe MIT license differs')
    shutil.copyfile(audit, assets / audit.name)
    shutil.copyfile(cpu_path, assets / 'q4-cpu-receipt.json')
    shutil.copyfile(gpu_root / 'receipt.json', assets / 'q4-cuda-receipt.json')
    shutil.copyfile(baseline_files['validation-evidence.json'], assets / 'F16-validation-evidence.json')
    acceptance = dict(selected_recipe=args.case, rules=row['rules'], file=row['file'],
        cpu_source=args.cpu_source, cpu_run=args.cpu_run, cpu_revision=args.cpu_revision,
        cpu_receipt_sha256=args.cpu_receipt_sha, cuda_revision=args.cuda_revision,
        cuda_prefix=str(prefix), cuda_receipt_sha256=args.cuda_receipt_sha,
        runtime_source=gpu['runtime_source'], runtime_provenance_sha256=gpu['runtime_provenance_sha256'],
        compiler_header_components=gpu['compiler_header_components'], hardware=gpu['hardware'],
        independent_audit_sha256=sha(audit), cpu_acceptance=True, cuda_acceptance=True,
        production_release=False, native_context_limit=4096, full_16k_context_qualified=False,
        scope='English JFK only; original F32 stages, exact tokens, repeated Session and CLI')
    if args.case == 'body-imatrix':
        calibration = cpu['calibration']
        if not (calibration['manifest'] == MANIFEST and calibration['archive'] == CALIBRATION and
                re.fullmatch('[0-9a-f]{64}', calibration['imatrix_sha256']) and len(calibration['clips']) == 8):
            raise ValueError('Pinned disjoint CC0 calibration evidence required')
        acceptance['calibration'] = calibration
        for name in ('calibration-source-card.md', 'commonvoice-source-card.md'):
            source = download(STAGING, cpu_prefix + '/' + name, args.cpu_revision, 'calibration')
            expected = ('8889b7876209a35cb5ab633cbd8ee0cc5b7d70f01d32b2db87ad8ef0f36bd88c'
                        if name.startswith('calibration-') else
                        '5d7f13a790c3f4de73ec28608570d7a4619bce675f37bdd42af47dfa6bfb0281')
            if sha(source) != expected or 'license: cc0-1.0' not in source.read_text():
                raise ValueError('Calibration CC0 publisher provenance differs')
            shutil.copyfile(source, assets / name)
        acceptance['calibration_source_cards'] = [dict(repo=STAGING, revision=args.cpu_revision,
            path=cpu_prefix + '/' + name, sha256=sha(assets / name))
            for name in ('calibration-source-card.md', 'commonvoice-source-card.md')]
    baseline = json.loads(baseline_files['validation-evidence.json'].read_text())
    def write_assets(published=None):
        current = copy.deepcopy(acceptance)
        if published:
            current.update(production_release=True, published_file=published)
        (assets / 'validation-evidence.json').write_text(json.dumps(extend_evidence(baseline, current), indent=2) + '\n')
        text = baseline_files['README.md'].read_text()
        # Preserve the complete old card/evidence as a separately named immutable asset.
        (assets / 'F16-release-README.md').write_text(text)
        text = text.replace('crispasr_native_acceptance: true',
                            'crispasr_native_acceptance: true\ncrispasr_validated_profiles: [f16, q4_k]')
        text = text.replace('Validated formats: f16.', 'Validated formats: f16, q4_k (selected recipe below).')
        text = text.replace('Q4_K and Q6_K are not qualified: their reference magnitude gates failed.',
                            'Earlier Q4_K recipes and Q6_K failed their reference magnitude gates; '
                            'only the newly qualified Q4_K recipe below supersedes those Q4 results.')
        text += ('\n## Additional Q4_K recipe qualification\n\n'
                 f'Canonical file: `{CANONICAL}`. Recipe: `{args.case}`; overrides: '
                 + ', '.join('`' + r + '`' for r in row['rules']) + '.\n'
                 'This selected recipe passes full CPU and actual CUDA checks against the original F32 '
                 'English JFK reference, including magnitude, exact prompt/generated tokens, repeated C ABI '
                 'Session and CLI. Native context remains 4096; full 16k inference is unqualified. '
                 'Earlier Q4 failures and the rejected Q8/Q6 profiles remain in the archived parent evidence. '
                 'The original F16 qualification and bytes remain unchanged. No runtime binaries, ELF files '
                 'or audio clips are distributed here. See `validation-evidence.json` → `q4_acceptance` '
                 'for immutable recipe, CPU/CUDA and optional CC0 calibration pins.\n')
        if 'calibration' in acceptance:
            text += ('Calibration uses eight fixed, JFK-disjoint Common Voice English clips under CC0-1.0. '
                     'The copied `calibration-source-card.md` and `commonvoice-source-card.md`, and '
                     '`q4_acceptance.calibration_source_cards`, retain the immutable publisher declarations.\n')
        (assets / 'README.md').write_text(text)
    write_assets()
    if not args.publish:
        print('All Q4 gates passed; review assets prepared without repository mutation')
        return
    if api.model_info(DESTINATION).sha != BASE:
        raise ValueError('Production main moved; review before changing the F16 release')
    model_record = row['file']
    model = download(model_record['repo'], model_record['path'], model_record['revision'], 'model', model_record['repo_type'])
    if sha(model) != model_record['sha256'] or model.stat().st_size != model_record['bytes']:
        raise ValueError('CPU/CUDA-approved candidate bytes differ')
    transfer = api.upload_file(repo_id=DESTINATION, path_or_fileobj=model, path_in_repo=CANONICAL,
        parent_commit=BASE, commit_message='Transfer CPU/CUDA-qualified Q4 recipe; final profile declaration pending')
    verified = readback(DESTINATION, transfer.oid, CANONICAL, model_record['sha256'], model_record['bytes'])
    published = dict(filename=CANONICAL, revision=transfer.oid, **verified)
    (out / 'transfer-checkpoint.json').write_text(json.dumps(published, indent=2) + '\n')
    model.unlink()
    write_assets(published)
    final = api.upload_folder(repo_id=DESTINATION, folder_path=assets, parent_commit=transfer.oid,
                              commit_message='Qualify selected Q4 profile while preserving accepted F16 provenance')
    for path in assets.iterdir():
        readback(DESTINATION, final.oid, path.name, sha(path), path.stat().st_size)
    final_info = api.model_info(DESTINATION, revision=final.oid, files_metadata=True)
    if not (final_info.card_data['license'] == 'apache-2.0' and
            final_info.card_data['crispasr_native_acceptance'] is True and
            list(final_info.card_data['crispasr_validated_profiles']) == ['f16', 'q4_k']):
        raise ValueError('Final qualified-profile declaration differs')
    for filename, identity in ((f16['filename'], f16), (CANONICAL, model_record)):
        item = next(s for s in final_info.siblings if s.rfilename == filename)
        if item.size != identity['bytes'] or item.lfs.sha256 != identity['sha256']:
            raise ValueError('Final published model identity differs')
    (out / 'release.json').write_text(json.dumps(dict(repo=DESTINATION, revision=final.oid,
        complete=True, base_f16_revision=BASE, models={'f16': f16, 'q4_k': published}), indent=2) + '\n')


if __name__ == '__main__':
    main()
