#!/usr/bin/env python3
"""#540: convert both pinned checkpoints, original dumps, diff and C ABI speech.

GH CPU only. Preserve every full array, failed stage and transcript; no
quantization or hardware/OS inference beyond the actual runner is accepted.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'dolphin-540'
UPSTREAM = '78ea615194777f91a3970ff010b5d678d0bbf5f0'
MODELS = [
    dict(name='small.cn', repo='DataoceanAI1/dolphi-cn-dialect-small',
         revision='eb6854969b5715cfccf4a9297a75f189343700dc', causal=False,
         checkpoint='1cee2b8d2133cabb36567625a832d4033569e27eaf5f98df9be1139ec6068bbb',
         yaml='617150581aae385f260e5e6fd17bbbee5ddd33db19ab8109d534e1fe1a765a01'),
    dict(name='small.cn.streaming', repo='DataoceanAI1/dolphin-cn-dialect-small-streaming',
         revision='f8e0bf3f0827ec24a2bb60e398e8e49ef2b5fd64', causal=True,
         checkpoint='bba8688ed33841b5f8b4578370be553f4557739c406dc7298f22985f7b061faf',
         yaml='cad5f3cf294f6e782ebbbfbdc6fa915dd28d58747725bccba1eb21e4d0320758')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_audio(path):
    import numpy as np
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768


def reference(args):
    import torch
    import numpy as np
    torch.set_num_threads(4)
    torch.manual_seed(42)
    sys.path.insert(0, str(ROOT))
    from tools.reference_backends.dolphin import dump
    from tools.dump_reference import write_gguf_archive
    captures = dump(model_dir=args.model_dir, audio=read_audio(args.audio), stages=set())
    args.reference_out.mkdir(parents=True, exist_ok=True)
    metadata = dict(text=captures.pop('text'), model_name=os.environ['DOLPHIN_MODEL_NAME'],
                    upstream=UPSTREAM, torch=torch.__version__, dither=0, audio_sha256=sha(args.audio))
    rows = {}
    for name, a in captures.items():
        assert a.size and np.isfinite(a).all(), name
        p = args.reference_out / (name + '.npy')
        np.save(p, a)
        rows[name] = dict(sha256=sha(p), shape=list(a.shape), dtype=str(a.dtype))
    write_gguf_archive(captures, metadata, args.reference_out / 'ref.gguf')
    (args.reference_out / 'reference.json').write_text(json.dumps(dict(metadata, arrays=rows), indent=2) + '\n')


def run(command, label, env=None, required=True):
    with (OUT / (label + '.log')).open('w') as log:
        r = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    print(label, 'rc', r.returncode, flush=True)
    if required:
        assert r.returncode == 0, label
    return r.returncode


class Params(C.Structure):
    _fields_ = [('n_threads', C.c_int), ('verbosity', C.c_int), ('use_gpu', C.c_bool), ('beam_size', C.c_int)]


class Result(C.Structure):
    _fields_ = [('text', C.c_char_p), ('raw_text', C.c_char_p), ('language', C.c_char * 32),
               ('region', C.c_char * 32), ('tokens', C.POINTER(C.c_int32)), ('n_tokens', C.c_int)]


def native(library, model, audio, reference_dir, dest):
    import numpy as np
    dest.mkdir(parents=True, exist_ok=True)
    fp = C.POINTER(C.c_float)
    lib = C.CDLL(str(library))
    lib.dolphin_context_default_params.restype = Params
    lib.dolphin_init_from_file.argtypes = [C.c_char_p, Params]
    lib.dolphin_init_from_file.restype = C.c_void_p
    lib.dolphin_free.argtypes = [C.c_void_p]
    lib.dolphin_compute_fbank.argtypes = [C.c_void_p, fp, C.c_int, C.POINTER(C.c_int)]
    lib.dolphin_compute_fbank.restype = fp
    lib.dolphin_run_encoder.argtypes = [C.c_void_p, fp, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(fp), C.c_int]
    lib.dolphin_run_encoder.restype = fp
    lib.dolphin_ctc_logprobs.argtypes = [C.c_void_p, fp, C.c_int, C.POINTER(C.c_int)]
    lib.dolphin_ctc_logprobs.restype = fp
    lib.dolphin_n_layers.argtypes = [C.c_void_p]
    lib.dolphin_n_layers.restype = C.c_int
    lib.dolphin_transcribe_ex.argtypes = [C.c_void_p, fp, C.c_int, C.c_char_p, C.c_char_p]
    lib.dolphin_transcribe_ex.restype = C.POINTER(Result)
    lib.dolphin_result_free.argtypes = [C.POINTER(Result)]
    libc = C.CDLL(None)
    libc.free.argtypes = [C.c_void_p]
    params = lib.dolphin_context_default_params()
    params.n_threads, params.use_gpu, params.verbosity = 4, False, 0
    ctx = lib.dolphin_init_from_file(str(model).encode(), params)
    assert ctx
    rows = {}
    def capture(name, array):
        original = np.load(reference_dir / (name + '.npy'), allow_pickle=False)
        assert original.shape == array.shape and np.isfinite(array).all(), name
        path = dest / (name + '.npy')
        np.save(path, array)
        a, b = original.astype(np.float64), array.astype(np.float64)
        aa, bb = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        norms = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
        cos = np.divide(np.sum(aa * bb, axis=1), norms, out=np.zeros_like(norms), where=norms > 0)
        cos[(np.linalg.norm(aa, axis=1) == 0) & (np.linalg.norm(bb, axis=1) == 0)] = 1
        ref_norm, cpp_norm = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        ratio = cpp_norm / ref_norm if ref_norm else (1. if cpp_norm == 0 else float('inf'))
        rows[name] = dict(shape=list(array.shape), sha256=sha(path), cosine_min=float(cos.min()),
                          cosine_mean=float(cos.mean()), reference_norm=ref_norm, native_norm=cpp_norm,
                          norm_ratio=ratio, relative_l2=float(np.linalg.norm(a-b) / ref_norm),
                          passed=bool(cos.min() >= .99 and .98 <= ratio <= 1.02))
        if name == 'ctc_logprobs':
            exact = bool(np.array_equal(original.argmax(-1), array.argmax(-1)))
            rows[name].update(argmax_exact=exact, passed=rows[name]['passed'] and exact)
    try:
        pcm = read_audio(audio)
        length = C.c_int()
        fbank = lib.dolphin_compute_fbank(ctx, pcm.ctypes.data_as(fp), len(pcm), C.byref(length))
        assert fbank
        try:
            capture('fbank', np.ctypeslib.as_array(fbank, shape=(length.value * 80,)).copy().reshape(length.value, 80))
        finally:
            libc.free(fbank)
        fb = np.load(reference_dir / 'fbank.npy', allow_pickle=False)
        n_layers = lib.dolphin_n_layers(ctx)
        expected = np.load(reference_dir / 'encoder_output.npy', allow_pickle=False)
        slots = [np.empty((expected.shape[0] + 8) * 1024, dtype=np.float32) for _ in range(n_layers + 1)]
        pointers = (fp * len(slots))(*(a.ctypes.data_as(fp) for a in slots))
        T, d = C.c_int(), C.c_int()
        encoder = lib.dolphin_run_encoder(ctx, fb.ctypes.data_as(fp), len(fb), C.byref(T), C.byref(d), pointers, len(slots))
        assert encoder and T.value == expected.shape[0] and d.value == expected.shape[1]
        try:
            capture('subsample_out', slots[0][:T.value*d.value].reshape(T.value, d.value).copy())
            for i in range(n_layers):
                capture(f'enc_blk_{i:02d}', slots[i+1][:T.value*d.value].reshape(T.value, d.value).copy())
            capture('encoder_output', np.ctypeslib.as_array(encoder, shape=(T.value*d.value,)).copy().reshape(T.value, d.value))
        finally:
            libc.free(encoder)
        V = C.c_int()
        ctc = lib.dolphin_ctc_logprobs(ctx, expected.ctypes.data_as(fp), len(expected), C.byref(V))
        assert ctc
        try:
            capture('ctc_logprobs', np.ctypeslib.as_array(ctc, shape=(len(expected)*V.value,)).copy().reshape(len(expected), V.value))
        finally:
            libc.free(ctc)
        result = lib.dolphin_transcribe_ex(ctx, pcm.ctypes.data_as(fp), len(pcm), None, None)
        assert result
        try:
            want = json.loads((reference_dir/'reference.json').read_text())['text']
            speech = dict(original_raw_text=want, native_raw_text=result.contents.raw_text.decode(),
                          native_text=result.contents.text.decode(), language=result.contents.language.decode(),
                          region=result.contents.region.decode(), tokens=np.ctypeslib.as_array(result.contents.tokens,
                          shape=(result.contents.n_tokens,)).tolist())
            speech['original_raw_exact'] = bool(want and speech['native_raw_text'] == want)
        finally:
            lib.dolphin_result_free(result)
    finally:
        lib.dolphin_free(ctx)
    # Preserve the full low-level result even if the independent binding path fails.
    checkpoint = dict(stages=rows, speech=speech, session_complete=False, passed=False)
    (dest/'validation.json').write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False)+'\n')
    # The automatic architecture path through the actual session binding.
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    with Session(str(model), lib_path=str(library), n_threads=4) as session:
        assert session.backend == 'dolphin'
        speech['session_backend'] = session.backend
        speech['session_text'] = ''.join(s.text for s in session.transcribe(pcm, sample_rate=16000)).strip()
        speech['session_matches_native'] = speech['session_text'] == speech['native_text'].strip()
    passed = all(r['passed'] for r in rows.values()) and speech['original_raw_exact'] and speech['session_matches_native']
    checkpoint.update(speech=speech, session_complete=True, passed=passed)
    (dest/'validation.json').write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False)+'\n')
    return checkpoint


def main():
    import yaml
    from huggingface_hub import hf_hub_download
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   ggml=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'ggml', text=True).strip(),
                   upstream=UPSTREAM, device='CPU; GPU backends OFF', models=MODELS,
                   complete=False, passed=False, cases={}, failures=[])
    def save():
        (OUT/'validation.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    save()
    try:
        run([sys.executable, '-m', 'pip', 'freeze'], 'python-packages')
        run(['uname', '-a'], 'host-platform')
        run(['sudo', 'bash', 'tools/ci-apt.sh', 'update'], 'apt-update')
        run(['sudo', 'bash', 'tools/ci-apt.sh', 'install', '-y', 'cmake', 'ninja-build', 'g++', 'libopenblas-dev', 'ffmpeg'], 'apt-install')
        run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
             'git+https://github.com/DataoceanAI/Dolphin.git@'+UPSTREAM], 'install-original')
        build = SCRATCH/'build'
        run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_OPENBLAS=ON',
             '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF',
             '-DCRISPASR_BUILD_TESTS=ON', '-DCRISPASR_BUILD_SERVER=OFF'], 'configure')
        run(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-lib',
             'crispasr-diff', 'test-dolphin-depthwise', 'test-arch-backend-map', '-j4'], 'build')
        run(['ctest', '--test-dir', build, '-R', 'dolphin-depthwise-checkpoint-padding|every converter-written architecture resolves|every emitted backend name is a name a surface can open|crispasr_detect_backend_from_gguf reads the shared table',
             '--output-on-failure', '--no-tests=error'], 'native-unit')
        receipt['native_unit_passed'] = True
        receipt['library_sha256'] = sha(build/'src/libcrispasr.so')
        save()
        for model in MODELS:
            md = SCRATCH/model['name']
            md.mkdir()
            pins = {model['name']+'.pt':model['checkpoint'], 'train.yaml':model['yaml'],
                    'units.txt':'1325172a696c68215206057f74be3fd527f54413173fad55dca04349e5070699',
                    'global_cmvn':'b1d75d6230fbbd34ae8eefc2f2488cfcb28b9894af7f14f27dc1576f081258f3'}
            for filename, digest in pins.items():
                p = hf_hub_download(model['repo'], filename, revision=model['revision'], local_dir=md)
                assert sha(p) == digest, filename
            assert yaml.safe_load((md/'train.yaml').read_text())['encoder_conf']['causal'] == model['causal']
            gguf = SCRATCH/(model['name']+'-f16.gguf')
            run([sys.executable, ROOT/'models/convert-dolphin-to-gguf.py', '--pt', md/(model['name']+'.pt'),
                 '--yaml', md/'train.yaml', '--units', md/'units.txt', '--output', gguf], 'convert-'+model['name'])
            model['gguf_sha256'], model['gguf_bytes'] = sha(gguf), gguf.stat().st_size
            for name, audio in [('zh', ROOT/'samples/paraformer_zh.wav'), ('jfk', ROOT/'samples/jfk.wav')]:
                label=model['name']+'-'+name
                ref=OUT/'reference'/label
                env=dict(os.environ, DOLPHIN_MODEL_NAME=model['name'], OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
                run([sys.executable, __file__, '--reference-out', ref, '--model-dir', md, '--audio', audio], 'reference-'+label, env)
                rc=run([build/'bin/crispasr-diff','dolphin',gguf,ref/'ref.gguf',audio], 'diff-'+label, required=False)
                row=native(build/'src/libcrispasr.so',gguf,audio,ref,OUT/'native'/label)
                row['diff_returncode'],row['audio_sha256']=rc,sha(audio)
                row['passed']=row['passed'] and rc==0
                receipt['cases'][label]=row
                if not row['passed']:receipt['failures'].append(label)
                save()
        receipt['complete']=True
        receipt['passed']=not receipt['failures']
        save()
        assert receipt['passed'],receipt['failures']
    except BaseException as error:
        receipt['error']=f'{type(error).__name__}: {error}'
        save()
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-out',type=Path)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--audio',type=Path)
    args=parser.parse_args()
    if args.reference_out:
        reference(args)
    else:
        main()
