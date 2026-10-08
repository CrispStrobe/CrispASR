#!/usr/bin/env python3
"""Find whether selective same-weight promotion stabilizes MiMo Q4 attention.

Diagnostic only: promoting dequantized Q4 weights does not recover original
weight precision. Remaining matrices stay Q4. Compare attention paths, exact
RVQ codes and a frozen independent same-weight Python encoder from the accepted
diagnostic. A passing profile is not original-checkpoint or full PR acceptance.
"""
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import gguf
import numpy as np
from huggingface_hub import hf_hub_download

from pr492_acceptance import ROOT, PINS, digest, metrics, run

REFERENCE_RUN = 37783601573
REFERENCE_ARTIFACT = 11554395735
REFERENCE_SOURCE = '7551b12de69510e26db6ba22077d0d4ea999674b'
REFERENCE_HASHES = {
    'tok_xfmr_out': 'e577e9ee68fc8ead762b7ab3b27a3f697161904235dbb8193e12aa3d1aab9ddb',
    'tok_pool_out': '6cbab6dc0d27ce645af4b91caed038717ed269cb17645716cf01399313aad1f6',
}


def accepted(m):
    return m['cosine'] >= .9999 and m['relative_l2'] <= .005


def main():
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'pr492-tokenizer-ablation'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                      CRISPASR_MIMO_FORCE_CPU='1', CRISPASR_GGUF_MMAP='1')
    os.environ.pop('CRISPASR_CORE_ATTN_EAGER_F32', None)
    receipt = dict(passed=False, scope=__doc__, profiles={}, reference_run=REFERENCE_RUN,
                   reference_artifact=REFERENCE_ARTIFACT, reference_source=REFERENCE_SOURCE,
                   source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    def save():
        (out / 'tokenizer-ablation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    archive = scratch / 'reference.zip'
    with archive.open('wb') as f:
        subprocess.run(['gh', 'api', f'repos/CrispStrobe/CrispASR/actions/artifacts/{REFERENCE_ARTIFACT}/zip'],
                       stdout=f, check=True, timeout=600)
    with zipfile.ZipFile(archive) as z:
        prior = json.loads(z.read('tokenizer-diagnosis.json'))
        assert prior['passed'] and prior['source'] == REFERENCE_SOURCE
        reference = {}
        for stage, sha in REFERENCE_HASHES.items():
            path = scratch / f'python-{stage}.npy'
            path.write_bytes(z.read(path.name))
            assert digest(path) == sha
            reference[stage] = np.load(path)
    receipt['reference_sha256'] = REFERENCE_HASHES
    receipt['scale_negative_control'] = metrics(reference['tok_pool_out'] * 2, reference['tok_pool_out'])
    assert not accepted(receipt['scale_negative_control'])
    run(['bash', ROOT / 'tools/ci-apt.sh', 'update'], out / 'apt-update.log')
    run(['bash', ROOT / 'tools/ci-apt.sh', 'install', '-y', 'ffmpeg'], out / 'apt-install.log')
    build = scratch / 'build'
    run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
         '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DCRISPASR_MEL_BLAS=OFF',
         '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=OFF'], out / 'configure.log')
    run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], out / 'build.log')
    library = next(build.rglob('libcrispasr.so'))
    repo, rev, filename, sha = PINS['codec']
    codec = Path(hf_hub_download(repo, filename, revision=rev, local_dir=scratch / 'models'))
    assert digest(codec) == sha == prior['codec']['sha256']
    receipt['codec'] = prior['codec']
    audio = scratch / 'en.wav'
    run(['ffmpeg', '-y', '-i', ROOT / 'samples/jfk.mp3', '-ar', '16000', '-ac', '1', audio], out / 'resample.log')
    # Check the exact native input before reusing the frozen independent output.
    receipt['audio_sha256'] = digest(audio)
    reader = gguf.GGUFReader(str(codec))
    quantized = {t.name for t in reader.tensors
                 if t.tensor_type not in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}}
    def head(name, count):
        match = re.match(r'enc\.blk\.(\d+)\.', name)
        return bool(match and int(match[1]) < count)
    profiles = {
        'head4': lambda n: head(n, 4),
        'head8': lambda n: head(n, 8),
        'head16': lambda n: head(n, 16),
        'outputs': lambda n: n.endswith(('.attn.o.weight', '.fc2.weight')),
        'ffn': lambda n: n.endswith(('.fc1.weight', '.fc2.weight')),
    }
    for name, select in profiles.items():
        model = scratch / f'{name}.gguf'
        writer = gguf.GGUFWriter(str(model), 'mimo_tokenizer', use_temp_file=True)
        for key, field in reader.fields.items():
            if key.startswith('GGUF.') or key in {'general.architecture', 'general.file_type', 'general.quantization_version'}:
                continue
            writer.add_key_value(key, field.contents(), field.types[0],
                                 field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
        promoted = []
        for t in reader.tensors:
            if t.name in quantized and select(t.name):
                value = gguf.dequantize(t.data, t.tensor_type).reshape(tuple(t.shape[::-1]))
                writer.add_tensor(t.name, np.ascontiguousarray(value, dtype=np.float32))
                promoted.append(t.name)
            else:
                writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
        assert promoted and len(promoted) < len(quantized)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        del writer
        gc.collect()
        result = dict(promoted=promoted, remaining_q4=len(quantized) - len(promoted),
                      model_sha256=digest(model), model_bytes=model.stat().st_size,
                      source_bytes=codec.stat().st_size)
        receipt['profiles'][name] = result
        save()
        for flash in [1, 0]:
            arm = name + ('-flash' if flash else '-eager')
            run([sys.executable, __file__, '--native', library, model, audio, out / arm, str(flash)],
                out / (arm + '.log'))
        def data(arm, stage):
            return np.load(out / f'{name}-{arm}' / (stage + '.npy'))
        # Stem tensors are unchanged across profiles and the pinned oracle input.
        for arm in ['flash', 'eager']:
            stem = data(arm, 'tok_conv2_out')
            with zipfile.ZipFile(archive) as z:
                import io
                expected = np.load(io.BytesIO(z.read('promoted-flash/tok_conv2_out.npy')))
            assert np.array_equal(stem, expected)
        result['attention_ab'] = {s: metrics(data('eager', s), data('flash', s)) for s in REFERENCE_HASHES}
        result['independent_same_weights'] = {s: {
            arm: metrics(data(arm, s).reshape(ref.shape), ref) for arm in ['flash', 'eager']}
            for s, ref in reference.items()}
        a, b = data('eager', 'tok_codes'), data('flash', 'tok_codes')
        result['codes'] = dict(exact=np.array_equal(a, b), match_fraction=float(np.mean(a == b)))
        result['passed'] = (all(accepted(m) for m in result['attention_ab'].values()) and
                            all(accepted(m) for arms in result['independent_same_weights'].values() for m in arms.values()) and
                            result['codes']['exact'])
        save()
        print(name, json.dumps(result), flush=True)
        model.unlink()
    receipt['passing_profiles'] = [n for n, r in receipt['profiles'].items() if r['passed']]
    receipt['passed'] = bool(receipt['passing_profiles'])
    save()
    assert receipt['passed'], 'No selective promotion profile met the unchanged gates'
    print('MIMO_TOKENIZER_ABLATION_PASS; diagnostic only, not PR acceptance', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--native':
        # Reuse the actual native extractor, selecting the required boundaries.
        import pr492_tokenizer_diagnose as diagnose
        diagnose.TOK_STAGES = ['tok_conv2_out', 'tok_xfmr_out', 'tok_pool_out', 'tok_codes']
        diagnose.native(*sys.argv[2:])
    else:
        main()
