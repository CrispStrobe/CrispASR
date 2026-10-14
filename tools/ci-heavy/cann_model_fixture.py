#!/usr/bin/env python3
"""Prepare and validate a CPU Moonshine fixture; this does not validate CANN."""
import argparse
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = '390624ed33d594443aa4aa221f5b9f283b545b5a'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def compare(reference, actual):
    import numpy as np
    if reference.shape != actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Shape/nonfinite mismatch')
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    if a.ndim != 2 or not a.size:
        raise ValueError('Expected a nonempty encoder matrix')
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if not (na > 0).all() or not (nb > 0).all():
        raise ValueError('Zero-norm encoder row')
    result = dict(cosine_min=float(np.min(np.sum(a*b, axis=1)/(na*nb))),
                  max_row_norm_error=float(np.max(np.abs(nb/na-1))),
                  relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                  max_abs=float(np.max(np.abs(a-b))))
    # Declared before the model run; magnitude checks reject cosine-blind scale defects.
    if result['cosine_min'] < .999 or result['max_row_norm_error'] > .01 or result['relative_l2'] > .01:
        raise ValueError(result)
    return result


def audio(path):
    import numpy as np
    with wave.open(str(path)) as f:
        assert f.getframerate() == 16000 and f.getnchannels() == 1 and f.getsampwidth() == 2
        return np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float32)/32768


def native_case(args):
    import numpy as np
    sys.path.insert(0, str(ROOT/'python'))
    from crispasr import Session
    pcm = audio(args.audio)
    outputs = {}
    for backend in (None, 'moonshine'):
        label = 'auto' if backend is None else 'explicit'
        with Session(args.model, lib_path=args.lib, backend=backend, n_threads=4) as session:
            assert session.backend == 'moonshine'
            outputs[label] = [' '.join(s.text for s in session.transcribe(pcm)) for _ in range(2)]
    lib = ctypes.CDLL(args.lib)
    class Params(ctypes.Structure):
        _fields_ = [('model', ctypes.c_char_p), ('tokenizer', ctypes.c_char_p),
                    ('threads', ctypes.c_int), ('gpu', ctypes.c_bool)]
    lib.moonshine_init_with_params.argtypes = [Params]
    lib.moonshine_init_with_params.restype = ctypes.c_void_p
    lib.moonshine_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.moonshine_encode.restype = ctypes.c_int
    lib.moonshine_free.argtypes = [ctypes.c_void_p]
    libc = ctypes.CDLL(None); libc.free.argtypes = [ctypes.c_void_p]
    ctx = lib.moonshine_init_with_params(Params(args.model.encode(), None, 4, False))
    assert ctx
    try:
        for i in range(2):
            ptr = ctypes.POINTER(ctypes.c_float)(); rows = ctypes.c_int(); dim = ctypes.c_int()
            assert lib.moonshine_encode(ctx, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(pcm),
                                        ctypes.byref(ptr), ctypes.byref(rows), ctypes.byref(dim)) == 0
            assert ptr and rows.value > 0 and dim.value == 288
            try:
                value = np.ctypeslib.as_array(ptr, shape=(rows.value*dim.value,)).copy().reshape(rows.value, dim.value)
            finally:
                libc.free(ptr)
            np.save(Path(args.output)/f'encoder-native-{i}.npy', value)
    finally:
        lib.moonshine_free(ctx)
    (Path(args.output)/'session-outputs.json').write_text(json.dumps(outputs, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', action='store_true')
    for name in ('model','lib','audio','output'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if args.native:
        native_case(args); return
    import numpy as np
    from huggingface_hub import snapshot_download, hf_hub_download
    out = Path(os.environ['HEAVY_OUT']); out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ['HEAVY_SCRATCH'])/'cann-model-fixture'; scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                      MKL_NUM_THREADS='4', USE_TF='0', PYTHONNOUSERSITE='1')
    sys.path.insert(0, str(ROOT/'ggml/gguf-py'))
    from gguf import GGUFReader
    receipt = dict(passed=False, scope='CPU fixture preparation only; no NPU/model-quant/performance acceptance.',
                   source=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                   ggml=subprocess.check_output(['git','-C',str(ROOT/'ggml'),'rev-parse','HEAD'], text=True).strip(),
                   original_revision=ORIGINAL)
    def save():
        (out/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    save()
    def run(cmd, label, env=None):
        with (out/(label+'.log')).open('wb') as log:
            result = subprocess.run(list(map(str,cmd)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        receipt.setdefault('commands',[]).append(dict(label=label,command=list(map(str,cmd)),exit=result.returncode));save()
        assert result.returncode == 0, label
        print(label, 'PASS', flush=True)
    snapshot = Path(snapshot_download('usefulsensors/moonshine-tiny', revision=ORIGINAL))
    receipt['original_files'] = [dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(snapshot.iterdir()) if p.is_file()]
    run(['bash',ROOT/'tools/ci-apt.sh','update'],'apt-update')
    run(['bash',ROOT/'tools/ci-apt.sh','install','-y','libopenblas-dev','libespeak-ng-dev','espeak-ng','pkg-config'],'apt-install')
    build = scratch/'build'
    run(['cmake','-S',ROOT,'-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=ON',
         '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
         '-DGGML_NATIVE=OFF','-DGGML_CANN=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DGGML_BLAS=OFF',
         '-DCRISPASR_BUILD_TESTS=OFF','-DCRISPASR_BUILD_SERVER=OFF','-DCRISPASR_OPUS=OFF','-DCRISPASR_AMR=OFF'],'configure')
    run(['cmake','--build',build,'--target','crispasr-cli','crispasr-lib','crispasr-diff','-j4'],'build')
    (out/'CMakeCache.txt').write_bytes((build/'CMakeCache.txt').read_bytes())
    (out/'compile_commands.json').write_bytes((build/'compile_commands.json').read_bytes())
    manifest = json.loads((ROOT/'tests/regression/manifest.json').read_text())
    entry = next(e for e in manifest['backends'] if e['name']=='moonshine-tiny')
    model_dir = scratch/'model'; model_dir.mkdir(exist_ok=True)
    model = model_dir/'moonshine-tiny.gguf'
    # Fresh conversion from the pinned original; keep published regression weights separately.
    run([sys.executable,ROOT/'models/convert-moonshine-to-gguf.py','--model',snapshot,'--outfile',model,'--outtype','f16'],'convert-f16')
    (out/model.name).write_bytes(model.read_bytes())
    (out/'tokenizer.bin').write_bytes((model_dir/'tokenizer.bin').read_bytes())
    published = Path(hf_hub_download(entry['gguf']['repo'],entry['gguf']['file'],revision=entry['gguf']['revision']))
    fresh={t.name:hashlib.sha256(t.data.tobytes()).hexdigest() for t in GGUFReader(str(model)).tensors}
    existing={t.name:hashlib.sha256(t.data.tobytes()).hexdigest() for t in GGUFReader(str(published)).tensors}
    assert fresh == existing, 'Fresh original conversion differs from pinned regression coefficients'
    receipt['physical_converted_tensors']=fresh
    sample=ROOT/'samples/jfk.wav';pcm=audio(sample)
    run([sys.executable,ROOT/'tools/dump_reference.py','--backend','moonshine','--model-dir',snapshot,'--audio',sample,'--output',out/'reference.gguf'],'dump-reference')
    reference={t.name:t for t in GGUFReader(str(out/'reference.gguf')).tensors}
    assert np.array_equal(reference['raw_audio'].data.reshape(-1),pcm)
    encoder=reference['encoder_output'];ref=encoder.data.reshape(tuple(int(n) for n in reversed(encoder.shape))).copy()
    np.save(out/'encoder-original.npy',ref)
    import torch, transformers
    from transformers import AutoProcessor, MoonshineForConditionalGeneration
    torch.set_num_threads(4)
    original=MoonshineForConditionalGeneration.from_pretrained(snapshot,torch_dtype=torch.float32).eval()
    processor=AutoProcessor.from_pretrained(snapshot)
    inputs=processor(pcm,return_tensors='pt',sampling_rate=processor.feature_extractor.sampling_rate)
    assert np.array_equal(inputs.input_values[0].numpy(), pcm)
    assert inputs.attention_mask.shape == inputs.input_values.shape and inputs.attention_mask.all()
    receipt['original_processor_input_shape']=list(inputs.input_values.shape)
    # Exact model-card driving recipe, including its per-second generation cap.
    max_length=int((inputs.attention_mask.sum(dim=-1)*(6.5/processor.feature_extractor.sampling_rate)).max().item())
    with torch.inference_mode():ids=original.generate(**inputs,max_length=max_length)
    text=processor.decode(ids[0],skip_special_tokens=True)
    assert words(text)==words(entry['expected_transcript']),text
    receipt['original_transcript']=text;receipt['generation_max_length']=max_length
    receipt['versions']=dict(torch=torch.__version__,transformers=transformers.__version__)
    modeling=Path(inspect.getfile(type(original)));(out/'actual-original-modeling.py').write_bytes(modeling.read_bytes())
    (out/'original-generation-config.json').write_text(original.generation_config.to_json_string())
    (out/'original-token-ids.json').write_text(json.dumps(ids.tolist())+'\n')
    del original,processor;save()
    lib=next(build.rglob('libcrispasr.so'));cli=build/'bin/crispasr';diff=build/'bin/crispasr-diff'
    dirs={str(p.parent) for p in build.rglob('*.so')}
    env=dict(os.environ,LD_LIBRARY_PATH=':'.join(sorted(dirs)))
    run(['ldd',lib],'native-library-closure',env)
    run([sys.executable,__file__,'--native','--model',model,'--lib',lib,'--audio',sample,'--output',out],'native-session-encoder',env)
    native=np.load(out/'encoder-native-0.npy');assert native.tobytes()==np.load(out/'encoder-native-1.npy').tobytes()
    receipt['encoder_same_input_metrics']=compare(ref,native)
    for label,value in [('double-scale',native*2),('nonfinite',native*np.nan),('wrong-shape',native[:-1])]:
        try:compare(ref,value)
        except ValueError:receipt.setdefault('negative_controls',[]).append(label)
        else:raise AssertionError(label)
    sessions=json.loads((out/'session-outputs.json').read_text())
    assert all(words(t)==words(text) for ts in sessions.values() for t in ts)
    receipt['session_outputs']=sessions
    run([diff,'moonshine',model,out/'reference.gguf',sample],'crispasr-diff',env)
    diff_text=(out/'crispasr-diff.log').read_text()
    match=re.search(r'\[PASS\]\s+encoder_output\s+.*cos_min=([0-9.]+)',diff_text)
    assert match and float(match[1])>=.999
    prefix=out/'cli'
    run([cli,'--backend','moonshine','-m',model,'-f',sample,'--no-gpu','-t','4','-l','en','--no-punctuation','-osrt','-of',prefix],'cli',env)
    srt=prefix.with_suffix('.srt').read_text()
    decoded=' '.join(line for line in srt.splitlines() if line.strip() and not line.strip().isdigit() and '-->' not in line)
    assert words(decoded)==words(text),decoded
    receipt['cli_transcript']=decoded
    receipt['sample']=dict(bytes=sample.stat().st_size,sha256=digest(sample));(out/'audio.wav').write_bytes(sample.read_bytes())
    receipt['model']=dict(bytes=model.stat().st_size,sha256=digest(model),tokenizer_sha256=digest(model_dir/'tokenizer.bin'))
    receipt['libraries']=[dict(path=str(p.relative_to(build)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(build.rglob('*.so'))]
    receipt['passed']=True;save()
    (out/'summary.md').write_text('CPU-only Moonshine Tiny fresh conversion, original generation/dumper, magnitude-sensitive encoder parity, diff, CLI and auto/explicit Session checks pass. This is not CANN hardware acceptance.\n')


if __name__ == '__main__':
    main()
