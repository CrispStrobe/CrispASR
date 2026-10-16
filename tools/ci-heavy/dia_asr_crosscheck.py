#!/usr/bin/env python3
"""Cross old/new sealed CPU recognizers against retained Dia PCM; no synthesis."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import wave
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = 'CrispStrobe/CrispASR'
ASR_REV = 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'
ASR_SHA = '60a2b074cdf46e81d72bc15d3eb79505dafbb697135f558659d9068f651854a2'
PINS = {
    'old': dict(run=38892690744, artifact=12049198423, source='5e04955e580b798b3cd45d870d407a3dd7600c4e',
        zip_sha='d4753346f6f98088e28d098fe014c12b960a05ff8acbbed1648d90b7854ad25c',
        library_sha='f9184c6bde354332300a989216550dc93eaf7452cfe4781dd48c7769ed2a40c0', prefix='libs/candidate/'),
    'new': dict(run=38913941927, artifact=12057273677, source='17a7aedc508be885cce0024199bd48525d3fc41b',
        zip_sha='75d9f3ec9d4fa96f1e284d2cf602afcf8b288d5cbdaa5bea5bf88532eda26c90',
        library_sha='631a7d89f2e0bcc0e30758e534226f1ea413ef5698ac130271f9d4f2fc4444c5', prefix='libs/'),
}
TEXTS = {'legacy': '[S1] We are committed to making it secure.',
         **{f'hello-{s}': '[S1] Hello there, how are you doing today?' for s in [42, 123]},
         **{f'fox-{s}': '[S1] The quick brown fox jumps over the lazy dog.' for s in [42, 123]},
         'jfk': 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country.'}
sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def score(transcript, expected):
    left = re.findall('[a-z]+', re.sub(r'\[S[12]\]', '', expected).lower())
    right = re.findall('[a-z]+', transcript.lower())
    assert left
    row = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        new = [i]
        for j, other in enumerate(right, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (token != other)))
        row = new
    return row[-1] / len(left)


def child(args):
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    assert sha(args.library) == PINS[args.child]['library_sha'] and sha(args.model) == ASR_SHA
    audio = np.load(args.audio, allow_pickle=False)
    assert audio.dtype == np.float32 and audio.ndim == 1 and audio.size and np.isfinite(audio).all()
    with Session(str(args.model), lib_path=str(args.library), backend='nemotron', n_threads=4) as session:
        assert session.backend == 'nemotron'
        segments = session.transcribe(audio, sample_rate=16000 if args.case == 'jfk' else 44100, language='en')
        text = ' '.join(s.text for s in segments).strip()
    maps = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        path = line.split()[-1]
        if '/libggml' in path or '/libcrispasr' in path:
            file = Path(path).resolve()
            assert file.is_file() and file.parent == args.library.parent.resolve(), file
            maps.add(file)
    assert maps
    row = dict(library=args.child, library_sha256=sha(args.library), audio_sha256=sha(args.audio), asr_sha256=ASR_SHA,
        case=args.case, samples=len(audio), transcript=text, wer=score(text, TEXTS[args.case]),
        mapped_libraries={p.name: sha(p) for p in sorted(maps)})
    args.output.write_text(json.dumps(row, indent=2) + '\n')


def selftest():
    assert score('We are committed to making it secure.', TEXTS['legacy']) == 0
    assert score('', TEXTS['hello-42']) == 1
    assert score('The quick brown fix jumps over the lazy dig.', TEXTS['fox-42']) == 2/9
    assert score('hello THERE how are you doing today?', TEXTS['hello-123']) == 0
    assert set(TEXTS) == {'legacy', 'hello-42', 'hello-123', 'fox-42', 'fox-123', 'jfk'}
    print('ASR_CROSSCHECK_SCORING_AND_CASE_CONTROLS_PASS')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child', choices=['old', 'new'])
    p.add_argument('--case', choices=list(TEXTS))
    p.add_argument('--selftest', action='store_true')
    for name in ['library', 'model', 'audio', 'output']:
        p.add_argument('--' + name, type=Path)
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if args.child:
        child(args)
        return
    selftest()
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE='1')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert source == os.environ['GITHUB_SHA'] and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
    report = dict(source=source, parents=PINS, results={}, complete=False, valid_recognizer_controls=False,
        speech_acceptance=False, gpu_acceptance=False, q4_acceptance=False, production_defaults_changed=False)

    def checkpoint():
        report['payloads'] = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                              for p in sorted(out.rglob('*')) if p.is_file() and not p.is_symlink() and p != out / 'receipt.json']
        (out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')

    def gh(*args):
        return subprocess.check_output(['gh', *map(str, args)])

    try:
        method = out / 'method'
        method.mkdir()
        shutil.copyfile(__file__, method / Path(__file__).name)
        shutil.copyfile(ROOT / 'python/crispasr/_binding.py', method / '_binding.py')
        for name in ['LICENSE', 'ggml/LICENSE']:
            shutil.copyfile(ROOT / name, out / ('GGML-LICENSE' if '/' in name else 'CrispASR-LICENSE'))
        audio_dir = out / 'audio'
        audio_dir.mkdir()
        libraries = {}
        sealed = {}
        for label, pin in PINS.items():
            run = json.loads(gh('run', 'view', pin['run'], '--repo', REPO, '--json', 'databaseId,headSha,conclusion,status'))
            assert run['databaseId'] == pin['run'] and run['headSha'] == pin['source'] and run['status'] == 'completed'
            (out / (label + '-run.json')).write_text(json.dumps(run, indent=2) + '\n')
            artifacts = json.loads(gh('api', f'repos/{REPO}/actions/runs/{pin["run"]}/artifacts'))
            (out / (label + '-artifacts.json')).write_text(json.dumps(artifacts, indent=2) + '\n')
            artifact = next(a for a in artifacts['artifacts'] if a['id'] == pin['artifact'])
            assert artifact['digest'] == 'sha256:' + pin['zip_sha']
            archive = scratch / (label + '.zip')
            with archive.open('xb') as f:
                subprocess.run(['gh', 'api', f'repos/{REPO}/actions/artifacts/{pin["artifact"]}/zip'], stdout=f, check=True)
            assert sha(archive) == pin['zip_sha'] and archive.stat().st_size == artifact['size_in_bytes']
            dest = out / 'libraries' / label
            dest.mkdir(parents=True)
            with zipfile.ZipFile(archive) as z:
                assert len(z.namelist()) == len(set(z.namelist()))
                for entry in z.infolist():
                    path = PurePosixPath(entry.filename)
                    assert not path.is_absolute() and '..' not in path.parts and '\\' not in entry.filename
                    assert not stat.S_ISLNK(entry.external_attr >> 16)
                receipt_data = z.read('receipt.json')
                (out / (label + '-parent-receipt.json')).write_bytes(receipt_data)
                parent = json.loads(receipt_data)
                assert parent['source'] == pin['source'] and parent['asr_sha256'] == ASR_SHA
                assert parent['library_sha256'] == pin['library_sha']
                index = {r['path']: r for r in parent['payloads']}

                def take(name, target):
                    data = z.read(name)
                    assert hashlib.sha256(data).hexdigest() == index[name]['sha256'] and len(data) == index[name]['bytes'], name
                    target.write_bytes(data)

                for name in z.namelist():
                    if name.startswith(pin['prefix']) and not name.endswith('/'):
                        assert len(PurePosixPath(name).relative_to(pin['prefix']).parts) == 1
                        take(name, dest / Path(name).name)
                names = ['legacy'] if label == 'old' else ['hello-42', 'hello-123', 'fox-42', 'fox-123']
                for key in names:
                    name = ('arms/original/' if label == 'old' else f'cases/{key}/arms/original/') + 'canonical-pcm.npy'
                    take(name, audio_dir / (key + '.npy'))
                    data = z.read(name + '.asr.json')
                    assert hashlib.sha256(data).hexdigest() == index[name + '.asr.json']['sha256']
                    row = json.loads(data)
                    sealed[key] = row
                    (audio_dir / (key + '-sealed-asr.json')).write_bytes(data)
            for file in list(dest.iterdir()):
                text = subprocess.check_output(['readelf', '-d', str(file)], text=True)
                match = re.search(r'\(SONAME\).*\[([^\]]+)\]', text)
                if match and not (dest / match[1]).exists():
                    (dest / match[1]).symlink_to(file.name)
            matches = [p for p in dest.iterdir() if p.is_file() and not p.is_symlink() and sha(p) == pin['library_sha']]
            assert len(matches) == 1
            libraries[label] = matches[0]
            checkpoint()
        wav_path = ROOT / 'samples/jfk.wav'
        shutil.copyfile(wav_path, audio_dir / 'jfk.wav')
        with wave.open(str(wav_path), 'rb') as wav:
            assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype('f4') / 32768
        np.save(audio_dir / 'jfk.npy', audio, allow_pickle=False)
        from huggingface_hub import hf_hub_download
        model = Path(hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
            'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf', revision=ASR_REV, local_dir=scratch / 'model'))
        assert sha(model) == ASR_SHA
        for label, library in libraries.items():
            report['results'][label] = {}
            for key in TEXTS:
                output = out / (label + '-' + key + '.json')
                env = {k: v for k, v in os.environ.items() if not k.startswith(('CRISPASR_DIA_', 'DIA_', 'LD_PRELOAD', 'LD_LIBRARY_PATH'))}
                env.update(LD_LIBRARY_PATH=str(library.parent), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                command = [sys.executable, __file__, '--child', label, '--case', key, '--library', library,
                           '--model', model, '--audio', audio_dir / (key + '.npy'), '--output', output]
                with (out / (label + '-' + key + '.log')).open('xb') as log:
                    result = subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                row = json.loads(output.read_text()) if output.exists() else dict(exit=result.returncode)
                row['exit'] = result.returncode
                if key != 'jfk' and result.returncode == 0:
                    row['matches_sealed_transcript'] = row['transcript'] == sealed[key]['transcript']
                report['results'][label][key] = row
                print(label, key, json.dumps(row), flush=True)
                checkpoint()
        rows = report['results']
        report['complete'] = all(set(r) == set(TEXTS) for r in rows.values()) and all(v['exit'] == 0 for r in rows.values() for v in r.values())
        report['valid_recognizer_controls'] = report['complete'] and all(r['legacy']['wer'] == 0 and r['jfk']['transcript'] for r in rows.values())
        report['recognizers_agree'] = report['complete'] and all(rows['old'][k]['transcript'] == rows['new'][k]['transcript'] for k in TEXTS)
        report['new_cases_wer_zero'] = report['complete'] and all(r[k]['wer'] == 0 for r in rows.values() for k in TEXTS if k not in ['legacy', 'jfk'])
        assert report['valid_recognizer_controls'], 'Recognizer controls failed; complete matrix retained'
    finally:
        checkpoint()
    print('DIA_ASR_CROSSED_REPLAY_COMPLETE', report['recognizers_agree'], report['new_cases_wer_zero'], flush=True)


if __name__ == '__main__':
    main()
