#!/usr/bin/env python3
"""Audit retained 9B F16 evidence against a complete independent F32 source.

Keep the original BF16 diagnostic failures visible. Accept complete F32 cases
at the unchanged 5.1 ms bound; never edit text, mix cues or relax timestamps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from index_echo_acceptance import compare_case

REQUIRED = {'jfk-en': 'en', 'zh-en': 'en', 'zh-ja': 'ja', 'zh-es': 'es', 'multi-en': 'en'}
CLIPS = {'jfk', 'zh', 'jfk-tail'}
PARAMETERS = {'tower': 647927168, 'connector': 8388608, 'llm': 8953803264}


def reference_cues(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError('Malformed independent direct-window reference')
    cues = []
    for i in range(0, len(lines), 3):
        m = re.fullmatch(r'\[(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)\]', lines[i])
        if not m:
            raise ValueError('Malformed independent source timestamp')
        cues.append(dict(start=60*int(m[1])+float(m[2]), end=60*int(m[3])+float(m[4]),
                         text=lines[i+1]+'\n'+lines[i+2]))
    return cues


def source_cases(source):
    if source.get('complete') is not True or set(source['cases']) != set(REQUIRED):
        raise ValueError('Incomplete independent F32 file oracle')
    if source['parameter_dtypes'].get('embedding_dtype') != 'torch.float32':
        raise ValueError('Actual decoder embeddings must be F32')
    for module, count in PARAMETERS.items():
        if source['parameter_dtypes'][module]['parameter_elements'] != {'torch.float32': count}:
            raise ValueError('The file oracle must use actual F32 parameters: ' + module)
    if source.get('context_audio') != 'zh-pause':
        raise ValueError('Natural context and rejected repeated-speech stress must stay separate')
    for name, target in REQUIRED.items():
        case = source['cases'][name]
        if case['target'] != target or not case['segments'] or any(row.get('parse_warn', 0) for row in case['rows']):
            raise ValueError('Invalid independent source case: ' + name)
    rows = source['cases']['multi-en']['rows']
    if len(rows) != 3 or not rows[1].get('has_ctx'):
        raise ValueError('Prior-window conditioning was not exercised')
    return source['cases']


def stage_report(text):
    if '[FAIL]' in text or '[SKIP]' in text:
        raise ValueError('Mandatory stage failed or was skipped')
    # stdout rows and stderr teardown diagnostics can interleave mid-row.
    # Bound every match by the next verdict so a missing metric cannot borrow
    # one from a different stage.
    rows = []
    for block in re.split(r'(?=\[(?:PASS|FAIL|SKIP)\])', text):
        name = re.match(r'\[PASS\]\s+(\S+)\s+shape\b', block)
        if not name:
            continue
        cosine = re.search(r'cos_min=([\d.]+)', block)
        relative = re.search(r'relative_l2=([\d.]+)', block)
        if not cosine or not relative:
            raise ValueError('Missing stage magnitude/cosine metric')
        rows.append(dict(stage=name[1], cosine_min=float(cosine[1]), relative_l2=float(relative[1])))
    names = {r['stage'] for r in rows}
    if len(rows) != 75 or len(names) != 75 or any(f'{prefix}_{i}' not in names for prefix in ['encoder_layer', 'llm_block'] for i in range(32)):
        raise ValueError('Every encoder/decoder layer and all 75 numerical rows are mandatory')
    if '[PASS] cached greedy token parity (16/16)' not in text or '[PASS] first greedy token parity (1/1)' not in text or not re.search(r'\[PASS\].*prompt', text):
        raise ValueError('Exact prompt and cached greedy IDs are mandatory')
    for row in rows:
        threshold = .998 if row['stage'] == 'teacherforced_logits' else .999
        if row['cosine_min'] < threshold or row['relative_l2'] > .02:
            raise ValueError('F16 numerical bounds exceeded: ' + row['stage'])
    return dict(passed=True, reported_checks=76, numerical_checks=75, cached_predictions=16,
                cosine_min=min(r['cosine_min'] for r in rows),
                relative_l2_max=max(r['relative_l2'] for r in rows), stages=rows)


def pipeline_report(native, source, gpu=False):
    import numpy as np
    golden = source_cases(source)
    if set(native['cases']) != set(REQUIRED):
        raise ValueError('Missing native file cases')
    results = {}
    for name, expected in golden.items():
        actual = native['cases'][name]
        matched = compare_case(actual['segments'], {'independent-all-f32': expected['segments']}, 'f16')
        probs = np.asarray(actual['vad_probabilities'], dtype=np.float64)
        reference = np.asarray(expected['vad_probabilities'], dtype=np.float64)
        if probs.shape != reference.shape or not probs.size:
            raise ValueError('VAD frame count differs: ' + name)
        delta = float(np.max(np.abs(probs-reference)))
        cosine = float(np.dot(probs,reference)/max(np.linalg.norm(probs)*np.linalg.norm(reference),1e-30))
        if delta > .01 or cosine < .999:
            raise ValueError('Independent VAD probabilities differ: ' + name)
        if gpu and actual.get('vad_gpu_request_equal') is not True:
            raise ValueError('GPU-requested CPU VAD regression failed')
        results[name] = dict(matched, vad_max_abs=delta, vad_cosine=cosine,
                             elapsed_seconds=actual['elapsed_seconds'])
    return results


def audit(cpu, cuda, source_path, model_revision, reference_revision, conversion_path):
    for pin in [model_revision, reference_revision]:
        if not re.fullmatch(r'[0-9a-f]{40}', pin):
            raise ValueError('Immutable model and reference revisions are mandatory')
    evidence = []
    def read(path):
        evidence.append(path)
        return json.loads(path.read_text())
    source = read(source_path)
    source_cases(source)
    provenance = read(cpu / 'validation-provenance.json')
    if provenance['model_revision'] != model_revision:
        raise ValueError('CPU evidence belongs to a different model')
    cpu_stages = read(cpu / 'stage-results-f16.json')
    if cpu_stages['failed']:
        raise ValueError('CPU stages failed')
    direct = read(cpu / 'decoded-f16.json')
    if set(direct) != CLIPS:
        raise ValueError('Missing CPU direct-window cases')
    for capture in direct.values():
        compare_case(capture['segments'], {'independent-generation':reference_cues(capture['reference'])}, 'f16')
    allowed = {'zh-en: full-pipeline decoded mismatch', 'zh-es: full-pipeline decoded mismatch'}
    if set(read(cpu / 'cohort-results.json')['f16']) - allowed:
        raise ValueError('CPU failure beyond the retained BF16 timing diagnostics')
    gpu = read(cuda / 'cuda-validation.json')
    if gpu['model_revision'] != model_revision or gpu['cuda_arch'] != '75':
        raise ValueError('CUDA evidence provenance differs')
    native_gpu = gpu['cohorts']['f16']
    if set(native_gpu['stages']) != CLIPS or any(v['rc'] or not v['cuda_used'] for v in native_gpu['stages'].values()):
        raise ValueError('Real CUDA stage execution is mandatory')
    for field in ['c_abi', 'cli']:
        if set(native_gpu[field]) != CLIPS or any(not v['passed'] for v in native_gpu[field].values()):
            raise ValueError('Exact CUDA direct decoding failed: ' + field)
        for capture in native_gpu[field].values():
            compare_case(capture['actual'], {'independent-generation':capture['expected']}, 'f16')
    if set(native_gpu['pipeline_failures']) - allowed or native_gpu['roundtrip_failures']:
        raise ValueError('CUDA failure beyond retained BF16 timing diagnostics')
    stages = {}
    for device, folder in [('cpu',cpu),('cuda',cuda)]:
        stages[device] = {}
        for clip in sorted(CLIPS):
            path = folder / f'f16-{clip}-diff.log'
            evidence.append(path)
            stages[device][clip] = stage_report(path.read_text())
    pipelines = {}
    for device, folder in [('cpu',cpu),('cuda',cuda)]:
        pipelines[device] = pipeline_report(read(folder / 'pipeline-f16.json'), source, gpu=device=='cuda')
    roundtrip = read(cuda / 'index-echo-9b-f16-roundtrip.json')
    if not roundtrip['roundtrip_passed'] or roundtrip['failed'] or set(roundtrip['cases']) != {'fox','window','station'}:
        raise ValueError('Real Piper roundtrip evidence incomplete')
    for case in roundtrip['cases'].values():
        if not case['passed'] or case['wer'] != 0 or not case['valid_segments'] or not case['cli_abi_text_match']:
            raise ValueError('Piper roundtrip failed')
    conversion = read(conversion_path)
    artifacts = {row['path']: dict(bytes=row['bytes'], sha256=row['sha256']) for row in conversion['artifacts']
                 if row['path'] in {'LICENSE','index-echo-9b-f16.gguf','index-echo-9b-decoder-f16.gguf'}}
    return dict(validated=True, accepted_cohort='f16', source_model=conversion['source'],
        source_revision=conversion['revision'], model_revision=model_revision,
        reference_revision=reference_revision, artifacts=artifacts,
        cpu=dict(passed=True, provenance=provenance, clips=stages['cpu']),
        cuda=dict(passed=True, build_commit=gpu['build_commit'], build_run=gpu['build_run'],
                  hardware=gpu['hardware'], clips=stages['cuda']),
        pipeline=dict(passed=True, source_precision='independently forced all-F32',
                      timestamp_tolerance_seconds=.0051, cases=pipelines,
                      original_bf16_diagnostics=dict(cpu=read(cpu / 'cohort-results.json')['f16'],
                                                    cuda=native_gpu['pipeline_failures'])),
        roundtrip=dict(passed=True, device='CUDA', english_wer=0.0, cases=roundtrip['cases']),
        evidence=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in evidence])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cpu','cuda','source','conversion','output']:
        p.add_argument('--'+name,type=Path,required=True)
    for name in ['model-revision','reference-revision']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    result=audit(a.cpu,a.cuda,a.source,a.model_revision,a.reference_revision,a.conversion)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('9B F16: CPU/CUDA stages/cache/direct decoding, all five complete F32 file cases and real Piper roundtrips PASS')
