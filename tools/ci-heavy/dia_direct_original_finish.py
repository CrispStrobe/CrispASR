#!/usr/bin/env python3
"""Finish recognition of sealed direct-original PCM; never repeat synthesis."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

import dia_direct_original as original

crossed = original.crossed
ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'd41cfb273dc5dda942d11cab20b3285a08585955'
RUN = 38940468611
ARTIFACT = 12065989037
DIGEST = 'a6460cf55be5b71afc083bf43af7a63ad39d32696c820979b68d50680c2d98ed'
ALIASES = {'libcrispasr.so.1': 'libcrispasr.so.0.8.43',
           **{'libggml' + suffix + '.so.0': 'libggml' + suffix + '.so.0.26.0'
              for suffix in ['', '-base', '-cpu']}}


def select_library(directory, expected):
    """Validate all canonical ELF bytes and upload-dereferenced SONAME aliases."""
    assert set(expected) == set(ALIASES.values())
    assert {p.name for p in directory.iterdir()} == set(expected) | set(ALIASES)
    for name, digest in expected.items():
        assert crossed.sha(directory / name) == digest, name
    for alias, canonical in ALIASES.items():
        assert crossed.sha(directory / alias) == expected[canonical], alias
    library = directory / 'libcrispasr.so.0.8.43'
    assert crossed.sha(library) == crossed.PINS['old']['library_sha']
    return library


def materialize_libraries(source, destination, expected):
    select_library(source, expected)
    destination.mkdir()
    for name in expected:
        shutil.copyfile(source / name, destination / name)
    # dlopen must map the canonical files, rather than regular duplicate aliases.
    for alias, canonical in ALIASES.items():
        (destination / alias).symlink_to(canonical)
    return select_library(destination, expected)


def selftest():
    with tempfile.TemporaryDirectory(prefix='dia-library-alias-') as temporary:
        directory = Path(temporary)
        expected = {}
        for i, name in enumerate(ALIASES.values()):
            (directory / name).write_bytes(bytes([i + 1]) * 31)
            expected[name] = crossed.sha(directory / name)
        for alias, canonical in ALIASES.items():
            shutil.copyfile(directory / canonical, directory / alias)
        # Override only the model-independent expected library digest.
        saved = crossed.PINS['old']['library_sha']
        try:
            crossed.PINS['old']['library_sha'] = expected['libcrispasr.so.0.8.43']
            assert select_library(directory, expected).name == 'libcrispasr.so.0.8.43'
            with tempfile.TemporaryDirectory(prefix='dia-canonical-layout-') as layout:
                mapped = materialize_libraries(directory, Path(layout) / 'libraries', expected)
                for alias, canonical in ALIASES.items():
                    assert (mapped.parent / alias).is_symlink()
                    assert (mapped.parent / alias).resolve() == mapped.parent / canonical
            # The old selection rejects this valid dereferenced alias layout.
            assert len([p for p in directory.iterdir() if crossed.sha(p) == expected['libcrispasr.so.0.8.43']]) == 2
            alias = directory / 'libcrispasr.so.1'
            alias.write_bytes(b'corrupted alias')
            try:
                select_library(directory, expected)
            except AssertionError:
                pass
            else:
                raise AssertionError('Corrupted alias accepted')
            shutil.copyfile(directory / 'libcrispasr.so.0.8.43', alias)
            alias.unlink()
            try:
                select_library(directory, expected)
            except AssertionError:
                pass
            else:
                raise AssertionError('Missing library alias accepted')
        finally:
            crossed.PINS['old']['library_sha'] = saved
    print('CANONICAL_LIBRARY_ALIAS_POSITIVE_AND_REJECTION_CONTROLS_PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    selftest()
    if args.selftest:
        return
    from huggingface_hub import hf_hub_download
    import numpy as np
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert source == os.environ['GITHUB_SHA']
    report = dict(source=source, parent_source=SOURCE, parent_run=RUN, parent_artifact=ARTIFACT,
                  parent_digest=DIGEST, results={}, controls={}, complete=False,
                  synthesis_repeated=False, speech_acceptance=False, gpu_acceptance=False,
                  q4_acceptance=False, production_defaults_changed=False)

    def checkpoint():
        report['payloads'] = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=crossed.sha(p))
                              for p in sorted(out.rglob('*')) if p.is_file() and p != out / 'receipt.json']
        (out / 'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    try:
        method = out / 'method'
        method.mkdir()
        for name in ['dia_direct_original_finish.py', 'dia_direct_original.py', 'dia_asr_crosscheck.py']:
            shutil.copyfile(Path(__file__).parent / name, method / name)
        shutil.copyfile(ROOT / 'python/crispasr/_binding.py', method / '_binding.py')
        shutil.copyfile(ROOT / 'LICENSE', out / 'CrispASR-LICENSE')
        metadata = subprocess.check_output(['gh', 'api', f'repos/CrispStrobe/CrispASR/actions/artifacts/{ARTIFACT}'])
        (out / 'parent-artifact.json').write_bytes(metadata)
        meta = json.loads(metadata)
        assert meta['digest'] == 'sha256:' + DIGEST and meta['size_in_bytes'] == 55896329
        assert meta['workflow_run']['id'] == RUN and meta['workflow_run']['head_sha'] == SOURCE
        bundle = scratch / 'direct-original.zip'
        with bundle.open('xb') as stream:
            subprocess.run(['gh', 'api', f'repos/CrispStrobe/CrispASR/actions/artifacts/{ARTIFACT}/zip'], stdout=stream, check=True)
        assert crossed.sha(bundle) == DIGEST
        with zipfile.ZipFile(bundle) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist()))
            for entry in archive.infolist():
                path = PurePosixPath(entry.filename)
                assert not path.is_absolute() and '..' not in path.parts and '\\' not in entry.filename
                assert not stat.S_ISLNK(entry.external_attr >> 16)
            archive.extractall(out / 'parent')
        parent = json.loads((out / 'parent/receipt.json').read_text())
        assert parent['source'] == SOURCE and not parent['complete'] and parent['results'] == {}
        entries = parent['payloads']
        assert len(entries) == len({r['path'] for r in entries})
        assert {p.relative_to(out / 'parent').as_posix() for p in (out / 'parent').rglob('*') if p.is_file()} == {r['path'] for r in entries} | {'receipt.json'}
        for row in entries:
            path = out / 'parent' / row['path']
            assert path.stat().st_size == row['bytes'] and crossed.sha(path) == row['sha256']
        for name in ['dia_direct_original.py', 'dia_asr_crosscheck.py', '_binding.py']:
            assert (method / name).read_bytes() == (out / 'parent/method' / name).read_bytes()
        captured = json.loads((out / 'parent/parent/receipt.json').read_text())
        assert captured['source'] == original.CROSSED_SOURCE and captured['valid_recognizer_controls']
        expected = captured['results']['old']['legacy']['mapped_libraries']
        library = materialize_libraries(out / 'parent/parent/libraries/old', scratch / 'asr-libraries', expected)
        report['recognizer_library_layout'] = dict(canonical=expected, aliases=ALIASES, alias_kind='symlink')
        model = Path(hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
            'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf', revision=crossed.ASR_REV))
        assert crossed.sha(model) == crossed.ASR_SHA
        env = {k: v for k, v in os.environ.items() if not k.startswith(('LD_PRELOAD', 'LD_LIBRARY_PATH', 'DIA_', 'CRISPASR_DIA_'))}
        env.update(LD_LIBRARY_PATH=str(library.parent.resolve()), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        for label, case, audio, control in [
            ('control-legacy', 'legacy', out / 'parent/parent/audio/legacy.npy', True),
            ('control-jfk', 'jfk', out / 'parent/parent/audio/jfk.npy', True),
            *[(key, key, out / 'parent/direct' / (key + '.npy'), False) for key in original.CASES],
        ]:
            destination = out / (label + '-asr.json')
            with (out / (label + '-asr.log')).open('xb') as stream:
                subprocess.run([sys.executable, Path(crossed.__file__), '--child', 'old', '--case', case,
                    '--library', library, '--model', model, '--audio', audio, '--output', destination],
                    env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1200)
            result = json.loads(destination.read_text())
            assert result['mapped_libraries'] == expected and result['audio_sha256'] == crossed.sha(audio)
            if control:
                report['controls'][case] = result
            else:
                row = original.compare_pcm(np.load(audio, allow_pickle=False), np.load(out / 'parent/parent/audio' / (case + '.npy'), allow_pickle=False))
                report['results'][case] = row | dict(asr=result)
            checkpoint()
            print('RETAINED_PCM_ASR_COMPLETE', label, result['wer'], flush=True)
        report['complete'] = True
        report['valid_recognizer_controls'] = all(r['wer'] == 0 for r in report['controls'].values())
        report['all_pcm_exact'] = all(r['exact'] for r in report['results'].values())
        report['all_wer_zero'] = all(r['asr']['wer'] == 0 for r in report['results'].values())
        assert report['valid_recognizer_controls'], 'Sealed positive controls failed'
        # A complete diagnostic may confirm failed identity/quality; retain both.
        checkpoint()
    finally:
        checkpoint()


if __name__ == '__main__':
    main()
