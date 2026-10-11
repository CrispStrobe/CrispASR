#!/usr/bin/env python3
"""PR537 fused-graph row-copy regression, identical pinned input and models.

A/B the production shared library with only three CONT nodes varied. CPU exact
and physical Metal controls; this does not replace original Python stage parity
or certify the reporter's two-minute music recording.
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'htdemucs-views'
BASE = '7e7094320f1d9f765379a7fcf5f84a03283fb21b'
BASE_HASH = 'ce4017e482b6bbb329096b55cef127218cf1e9716c64fadb0392a0efd9ce7eb0'
MODEL_REV = 'e24b8c69380f69b5ad96d8fa5d0e01b1aa2867db'
MODEL_HASH = {
    'f16': '4e7c4dcb8ac6315115848ff642ff8465f56b96d0a8ac7450be82acfc3534f006',
    'q8_0': 'd378b50ce50608093e5f64416bcefacce3f7bc45c0cd9141c276750cb8662942',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def execute(cmd, label, allow_failure=False):
    path = OUT / (label + '.log')
    with path.open('w') as log:
        result = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, timeout=7200)
    print(label, result.returncode, path.read_text(errors='replace')[-1600:], flush=True)
    if not allow_failure:
        assert result.returncode == 0, label
    return result.returncode


def infer(library, model, gpu, label):
    import numpy as np
    # Force the modified graph even on CPU: legacy BLAS never visits these nodes.
    os.environ.update(CRISPASR_HTDEMUCS_GGML='1', CRISPASR_HTDEMUCS_FUSED='1',
                      CRISPASR_HTDEMUCS_GPU=str(int(gpu)))
    lib = ct.CDLL(str(library))
    class Params(ct.Structure):
        _fields_ = [('threads', ct.c_int), ('gpu', ct.c_bool), ('device', ct.c_int)]
    class Result(ct.Structure):
        _fields_ = [('sources_count', ct.c_int), ('channels', ct.c_int), ('samples', ct.c_int),
                    ('rate', ct.c_int), ('sources', ct.POINTER(ct.POINTER(ct.c_float))),
                    ('names', ct.POINTER(ct.c_char_p))]
    lib.htdemucs_init_from_file.argtypes = [ct.c_char_p, Params]
    lib.htdemucs_init_from_file.restype = ct.c_void_p
    lib.htdemucs_separate.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int]
    lib.htdemucs_separate.restype = ct.POINTER(Result)
    lib.htdemucs_free.argtypes = [ct.c_void_p]
    lib.htdemucs_free.restype = None
    lib.htdemucs_result_free.argtypes = [ct.POINTER(Result)]
    lib.htdemucs_result_free.restype = None
    pcm = np.load(OUT / 'input.npy', allow_pickle=False)
    ctx = lib.htdemucs_init_from_file(str(model).encode(), Params(3, gpu, 0))
    assert ctx, 'Model load failed'
    result = None
    try:
        result = lib.htdemucs_separate(ctx, pcm.ctypes.data_as(ct.POINTER(ct.c_float)), len(pcm))
        assert result, 'Separation failed'
        r = result.contents
        assert (r.sources_count, r.channels, r.samples, r.rate) == (4, 2, len(pcm), 44100)
        stems = np.stack([np.ctypeslib.as_array(r.sources[i], shape=(len(pcm) * 2,)).copy().reshape(-1, 2)
                          for i in range(4)])
        assert np.isfinite(stems).all() and np.linalg.norm(stems) > 0
        np.save(OUT / (label + '.npy'), stems, allow_pickle=False)
        names = [r.names[i].decode('utf8', 'strict') for i in range(4)]
        assert names == ['drums', 'bass', 'other', 'vocals'], names
    finally:
        if result:
            lib.htdemucs_result_free(result)
        lib.htdemucs_free(ctx)


def compare(a, b, exact=False):
    import numpy as np
    x = np.load(OUT / (a + '.npy'), allow_pickle=False)
    y = np.load(OUT / (b + '.npy'), allow_pickle=False)
    assert x.shape == y.shape and x.dtype == y.dtype == np.float32
    rows = []
    for i, name in enumerate(['drums', 'bass', 'other', 'vocals']):
        u, v = x[i].astype(np.float64).ravel(), y[i].astype(np.float64).ravel()
        nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        assert nu > 0 and nv > 0, name
        cos = float(np.dot(u, v) / (nu * nv))
        ratio = nu / nv
        row = dict(stem=name, cosine=cos, mine_norm=nu, reference_norm=nv,
                   magnitude_ratio=ratio, max_absolute_error=float(np.max(np.abs(u-v))),
                   exact=bool(np.array_equal(x[i], y[i])))
        rows.append(row)
        # Existing #398/#414 CPU/GPU acceptance gates, unchanged.
        row['passed'] = row['exact'] if exact else cos > 0.98 and 0.9 < ratio < 1.1
    return dict(candidate=a, reference=b, require_exact=exact, rows=rows,
                passed=all(r['passed'] for r in rows))


def main():
    import numpy as np
    from huggingface_hub import hf_hub_download
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'src/htdemucs.cpp'
    candidate = source.read_bytes()
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   ggml=subprocess.check_output(['git', '-C', ROOT / 'ggml', 'rev-parse', 'HEAD'], text=True).strip(),
                   platform=platform.platform(), passed=False, comparisons=[],
                   scope='row-copy regression A/B, not original Python or music-corpus acceptance')
    try:
        if subprocess.run(['git', 'cat-file', '-e', BASE + '^{commit}'], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            subprocess.run(['git', 'fetch', '--depth=1', 'https://github.com/CrispStrobe/CrispASR.git', BASE],
                           cwd=ROOT, check=True, timeout=300)
        baseline = subprocess.check_output(['git', 'show', BASE + ':src/htdemucs.cpp'], cwd=ROOT)
        assert hashlib.sha256(baseline).hexdigest() == BASE_HASH
        note = b'    // Row slices are already contiguous; ggml_cont would be a Q8_0 -> Q8_0 copy Metal does not support.\n'
        assert candidate.count(note) == 1
        restored = candidate.replace(note, b'')
        for w, x in [('Wq', 'qn'), ('Wk', 'kn'), ('Wv', 'kn')]:
            old = f'ggml_mul_mat(g, {w}, {x})'.encode()
            assert restored.count(old) == 1
            restored = restored.replace(old, f'ggml_mul_mat(g, ggml_cont(g, {w}), {x})'.encode())
        assert restored == baseline, 'A/B must differ only by three CONT nodes'
        receipt.update(baseline_source=BASE, baseline_cpp_sha256=BASE_HASH,
                       candidate_cpp_sha256=hashlib.sha256(candidate).hexdigest())
        # Deterministic nonzero asymmetric stereo input, retained byte-for-byte.
        t = np.arange(44100 * 3, dtype=np.float64) / 44100
        pcm = np.stack([0.2*np.sin(2*np.pi*220*t)+0.08*np.sin(2*np.pi*701*t),
                        0.17*np.sin(2*np.pi*330*t)+0.07*np.sin(2*np.pi*997*t)], axis=1).astype(np.float32)
        np.save(OUT / 'input.npy', pcm, allow_pickle=False)
        receipt['input_sha256'] = digest(OUT / 'input.npy')
        metal = subprocess.check_output(['swift', '-e',
            'import Metal; print(MTLCreateSystemDefaultDevice()?.name ?? "NONE")'], text=True).strip() if sys.platform == 'darwin' else 'NONE'
        receipt['metal_device'] = metal
        models = {}
        for quant, sha in MODEL_HASH.items():
            model = Path(hf_hub_download('cstr/htdemucs-GGUF', f'htdemucs-{quant}.gguf', revision=MODEL_REV))
            assert digest(model) == sha, quant
            models[quant] = model
        receipt['models'] = dict(repo='cstr/htdemucs-GGUF', revision=MODEL_REV, sha256=MODEL_HASH)
        for arm, contents in [('baseline', baseline), ('candidate', candidate)]:
            source.write_bytes(contents)
            build = SCRATCH / arm
            execute(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
                     '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF',
                     '-DGGML_METAL=' + ('ON' if sys.platform == 'darwin' else 'OFF'),
                     '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=OFF'], arm + '-configure')
            execute(['cmake', '--build', build, '--target', 'crispasr-lib', '-j3'], arm + '-build')
            libs = list(build.rglob('libcrispasr.dylib' if sys.platform == 'darwin' else 'libcrispasr.so'))
            assert len(libs) == 1
            receipt[arm + '_library_sha256'] = digest(libs[0])
            for quant, model in models.items():
                for gpu in ([False, True] if metal != 'NONE' else [False]):
                    label = f'{arm}-{quant}-gpu{int(gpu)}'
                    expected_abort = arm == 'baseline' and quant == 'q8_0' and gpu
                    rc = execute([sys.executable, __file__, '--infer', libs[0], model, str(int(gpu)), label],
                                 label, allow_failure=expected_abort)
                    log = (OUT / (label + '.log')).read_text(errors='replace')
                    assert f'gates graph=1 fused=1 gpu={int(gpu)}' in log, 'Target fused graph must engage'
                    if gpu:
                        assert 'ggml_metal' in log and metal in log, 'No CPU fallback hardware claim'
                    if rc:
                        assert expected_abort and "unsupported op 'CONT'" in log, log[-2000:]
                        receipt['baseline_q8_metal_abort_reproduced'] = True
        source.write_bytes(candidate)
        for quant in models:
            receipt['comparisons'].append(compare(f'candidate-{quant}-gpu0', f'baseline-{quant}-gpu0', exact=True))
            if metal != 'NONE':
                receipt['comparisons'].append(compare(f'candidate-{quant}-gpu1', f'candidate-{quant}-gpu0'))
        if metal != 'NONE':
            receipt['comparisons'].append(compare('candidate-f16-gpu1', 'baseline-f16-gpu1', exact=True))
        assert all(c['passed'] for c in receipt['comparisons']), 'Stem numerical gate failed'
        receipt['passed'] = True
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        source.write_bytes(candidate)
        (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (OUT / 'summary.md').write_text(f"HTDemucs row-view A/B passed={receipt['passed']}, "
                                       f"Metal={receipt.get('metal_device', 'not probed')}. "
                                       'Original Python/music-corpus parity is not tested.\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--infer':
        infer(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4] == '1', sys.argv[5])
    else:
        main()
