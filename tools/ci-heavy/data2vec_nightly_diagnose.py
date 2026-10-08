#!/usr/bin/env python3
"""Diagnose data2vec Q4 word insertion against v0.8.41 and precision controls.

The original transcript gate stays at zero WER. Preserved tensors come from
the original pinned F16 GGUF, never from dequantizing Q4. This one-clip study
does not certify a replacement model or change any published weights.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import gguf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
BASELINE = '340d7085eaa53c40a46dcb73a6d3d0448a480006'
REVISION = '97d798574af55a3b861d323c2222f3d6cbbf1048'
HASHES = {
    'q4_k': '93b6ab01f1f83525157d797a385a3e9e014c6761d3e974351363adc452a86f7e',
    'q8_0': '622ea462be07a10bb4a4ebee4ceeb8b2fc049eb224d588f078582279a02255d3',
    'f16': '4df9cc50b7f5340488fa64f5ead34f93660c6ff8785e7e71a859a8860a13254e',
}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'data2vec-nightly'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    receipt = dict(passed=False, scope=__doc__, baseline=BASELINE, runs={}, variants={},
                   source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    def save():
        (out / 'data2vec-diagnosis.json').write_text(json.dumps(receipt, indent=2) + '\n')
    def run(args, tag, cwd=ROOT):
        with (out / (tag + '.log')).open('w') as f:
            p = subprocess.run(list(map(str, args)), cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        print(tag, p.returncode, (out / (tag + '.log')).read_text()[-1500:], flush=True)
        assert p.returncode == 0, tag
    save()
    run(['bash', ROOT / 'tools/ci-apt.sh', 'update'], 'apt-update')
    run(['bash', ROOT / 'tools/ci-apt.sh', 'install', '-y', 'libopenblas-dev', 'libespeak-ng-dev', 'espeak-ng', 'pkg-config'], 'apt-install')
    run(['git', 'fetch', '--depth=1', 'origin', BASELINE], 'fetch-baseline')
    old = scratch / 'baseline'
    run(['git', 'worktree', 'add', '--detach', old, BASELINE], 'baseline-worktree')
    run(['git', 'submodule', 'update', '--init', '--recursive'], 'baseline-submodules', cwd=old)
    binaries = {}
    receipt['ggml'] = {}
    for name, source in [('baseline', old), ('current', ROOT)]:
        build = scratch / ('build-' + name)
        run(['cmake', '-S', source, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_EXAMPLES=ON', '-DCRISPASR_BUILD_SERVER=OFF'], name + '-configure')
        run(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-diff', '-j4'], name + '-build')
        binaries[name] = build / 'bin/crispasr'
        receipt['ggml'][name] = subprocess.check_output(['git', '-C', source / 'ggml', 'rev-parse', 'HEAD'], text=True).strip()
        (out / (name + '-CMakeCache.txt')).write_bytes((build / 'CMakeCache.txt').read_bytes())
    sys.path.insert(0, str(ROOT / 'tests/regression'))
    from run_one import run_transcript, transcript_gate
    manifest = json.loads((ROOT / 'tests/regression/manifest.json').read_text())
    entry = next(b for b in manifest['backends'] if b['name'] == 'data2vec-base')
    sample = ROOT / entry['sample']
    receipt['audio_sha256'] = digest(sample)
    receipt['expected_transcript'] = entry['expected_transcript']
    models = {}
    for quant, sha in HASHES.items():
        models[quant] = Path(hf_hub_download('cstr/data2vec-audio-960h-GGUF',
            f'data2vec-audio-base-960h-{quant}.gguf', revision=REVISION, local_dir=scratch / 'models'))
        assert digest(models[quant]) == sha
    receipt['models'] = {q: dict(sha256=sha, bytes=models[q].stat().st_size, revision=REVISION) for q, sha in HASHES.items()}
    def transcribe(binary, model):
        text = run_transcript(binary, model, sample)
        ok, report = transcript_gate(entry, text)
        return dict(passed=ok, text=text, report=report)
    for name, binary in binaries.items():
        receipt['runs'][name] = {}
        for quant, model in models.items():
            receipt['runs'][name][quant] = transcribe(binary, model)
            save()
            print(name, quant, json.dumps(receipt['runs'][name][quant]), flush=True)
    q4 = gguf.GGUFReader(str(models['q4_k']))
    full = gguf.GGUFReader(str(models['f16']))
    original = {t.name: t for t in full.tensors}
    for profile in ['original-head', 'head1', 'head2', 'head4']:
        def select(t):
            if profile == 'original-head':
                return t.name.startswith('lm_head.')
            m = re.match(r'enc\.(\d+)\.', t.name)
            return m and int(m[1]) < int(profile[4:]) and gguf.GGMLQuantizationType(t.tensor_type) not in {
                gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}
        model = scratch / (profile + '.gguf')
        writer = gguf.GGUFWriter(str(model), q4.fields['general.architecture'].contents(), use_temp_file=True)
        for key, field in q4.fields.items():
            if key.startswith('GGUF.') or key == 'general.architecture':
                continue
            writer.add_key_value(key, field.contents(), field.types[0],
                                 field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
        preserved = []
        for t in q4.tensors:
            use = original[t.name] if select(t) else t
            writer.add_tensor(t.name, use.data, raw_dtype=use.tensor_type)
            if use is not t:
                preserved.append(dict(name=t.name, old_type=t.tensor_type.name, new_type=use.tensor_type.name,
                                      unchanged_bytes=t.data.tobytes() == use.data.tobytes()))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        del writer
        gc.collect()
        result = dict(preserved=preserved, model_bytes=model.stat().st_size, sha256=digest(model),
                      transcript=transcribe(binaries['current'], model))
        receipt['variants'][profile] = result
        save()
        print(profile, json.dumps(result), flush=True)
        model.unlink()
    fixtures = manifest['fixtures']
    reference = Path(hf_hub_download(fixtures['repo'], entry['fixture_ref_path'],
                    revision=fixtures['revision'], local_dir=scratch / 'fixtures'))
    receipt['reference'] = dict(repo=fixtures['repo'], revision=fixtures['revision'], sha256=digest(reference))
    save()
    for name, binary in binaries.items():
        run([binary.parent / 'crispasr-diff', entry['backend_id'], models['f16'], reference, sample], name + '-f16-diff')
    receipt['passed'] = receipt['runs']['current']['q4_k']['passed']
    save()
    assert receipt['passed'], 'Current published Q4 still fails the zero-WER transcript gate; inspect all controls'


if __name__ == '__main__':
    main()
