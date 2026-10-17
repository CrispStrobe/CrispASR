#!/usr/bin/env python3
"""9B native FFN owned-input capture and lazy actual original MLP attribution (CPU)."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import numpy as np
from echo_owned_ffn import (SUFFIXES, TRANSFORMERS, MODELING_SHA, MLP_SHA, array_sha, exact,
                            native_observer_environment, validate_native_observers)

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = 'b8ac6fb7d3dc17cee48a52201bd3d93dc86b0dba'
CONVERTED = '70cbee15fa7fadb85829d9d8d3439a5cb8fbe52b'
GGML = '0a862076567afd8468822ccc512c97ac6a69b340'
DATASET = 'cstr/crispasr-regression-fixtures'
PREFIX = 'index-echo-9b/owned-ffn-20261017/v2'
FILES = {
    'primary': ('cstr/index-echo-9b-GGUF', CONVERTED, 'index-echo-9b-f16.gguf', 1313848736,
                'e4b1dd177211ba5d8eb0ea7140ac2f813ddc68b55381f050ecf7ddab3e83172f'),
    'decoder': ('cstr/index-echo-9b-GGUF', CONVERTED, 'experiments/q4-imatrix-20261009/q4_k_plain/decoder.gguf', 5051130528,
                '85b43a414f6bd8b29b26e58efb4dbd7d7cb489432295f197384f79813021fd2b'),
    'original': ('IndexTeam/Index-Echo-S2TT-9B', ORIGINAL, 'llm/model.safetensors', 17907656616,
                 '42e924f739466a73f4f78f58d5ab0ea58475037dfc4724737ce3d4d68019fc72'),
}
CASES = {'jfk': 'jfk.wav', 'zh': 'paraformer_zh.wav'}
sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Params(C.Structure):
    _fields_ = [('n_threads', C.c_int), ('verbosity', C.c_int), ('use_gpu', C.c_bool), ('flash_attn', C.c_bool)]


def native(args):
    """One actual native context, OFF/ON/OFF; parent loads no native model."""
    import soundfile as sf
    lib = C.CDLL(str(args.lib))
    fp, ip, vp = C.POINTER(C.c_float), C.POINTER(C.c_int), C.c_void_p
    signatures = {'context_default_params': ([], Params), 'init_from_file': ([C.c_char_p, Params], vp),
        'free': ([vp], None), 'compute_mel': ([vp, fp, C.c_int, ip, ip], fp),
        'run_encoder': ([vp, fp, C.c_int, C.c_int, ip, ip], fp),
        'prefill': ([vp, fp, C.c_int, C.c_int, ip], fp), 'stage': ([vp, C.c_char_p, ip], fp),
        'prompt_ids': ([vp, ip], C.POINTER(C.c_int32)), 'decoder_layers': ([vp], C.c_int)}
    for name, (inputs, output) in signatures.items():
        fn = getattr(lib, 'index_echo_'+name)
        fn.argtypes, fn.restype = inputs, output
    free = C.CDLL(None).free
    free.argtypes, free.restype = [vp], None
    def owned(pointer, count):
        assert pointer and count > 0
        try:
            a = np.ctypeslib.as_array(pointer, shape=(count,)).copy()
            assert a.size and np.isfinite(a).all()
            return a
        finally:
            free(C.cast(pointer, vp))
    audio, rate = sf.read(args.audio, dtype='float32')
    assert rate == 16000 and audio.ndim == 1 and np.isfinite(audio).all()
    audio = np.ascontiguousarray(audio)
    data = audio.ctypes.data_as(fp)
    for key in list(os.environ):
        if key.startswith(('INDEX_ECHO_', 'CRISPASR_INDEX_ECHO_', 'CRISP_AUDIO_DUMP_')):
            del os.environ[key]
    params = lib.index_echo_context_default_params()
    params.n_threads, params.verbosity, params.use_gpu = 4, 0, False
    assert params.flash_attn
    ctx = lib.index_echo_init_from_file(str(args.model).encode(), params)
    assert ctx and lib.index_echo_decoder_layers(ctx) == 32
    inputs = {}
    baseline = None
    try:
        for arm in ['off', 'on', 'off2']:
            os.environ.pop('CRISPASR_INDEX_ECHO_DUMP_FFN', None)
            os.environ.update(native_observer_environment(arm))
            validate_native_observers(arm, os.environ)
            folder = args.output/arm
            folder.mkdir(parents=True, exist_ok=False)
            mels, frames, rows, width, vocab = [C.c_int() for _ in range(5)]
            mel = owned(lib.index_echo_compute_mel(ctx, data, len(audio), C.byref(mels), C.byref(frames)), mels.value*frames.value)
            emb = owned(lib.index_echo_run_encoder(ctx, mel.ctypes.data_as(fp), mels.value, frames.value, C.byref(rows), C.byref(width)), rows.value*width.value)
            logits = owned(lib.index_echo_prefill(ctx, emb.ctypes.data_as(fp), rows.value, width.value, C.byref(vocab)), vocab.value)
            count = C.c_int()
            pointer = lib.index_echo_prompt_ids(ctx, C.byref(count))
            assert pointer and count.value > 0
            arrays = dict(mel=mel, connector=emb, logits=logits,
                          prompt_ids=np.ctypeslib.as_array(pointer, shape=(count.value,)).copy())
            for layer in range(32):
                for suffix in ['block', *SUFFIXES]:
                    name = f'llm_block_{layer}' if suffix == 'block' else f'llm_ffn_{layer}_{suffix}'
                    count.value = 0
                    pointer = lib.index_echo_stage(ctx, name.encode(), C.byref(count))
                    if suffix != 'block' and arm != 'on':
                        assert not pointer, 'Observer stages leaked into OFF'
                        continue
                    assert pointer and count.value > 0, name
                    values = np.ctypeslib.as_array(pointer, shape=(count.value,)).copy()
                    assert np.isfinite(values).all()
                    if suffix != 'block':
                        expected = 12288 if suffix in ['gate_out', 'up_out', 'down_in'] else 4096
                        assert values.shape == (expected,), (name, values.shape)
                    arrays[name] = values
            common = {k:v for k,v in arrays.items() if not k.startswith('llm_ffn_')}
            if baseline is None:
                baseline = {k:v.copy() for k,v in common.items()}
            else:
                assert set(common) == set(baseline)
                for key in common:
                    exact(common[key], baseline[key])
            for name, value in arrays.items():
                np.save(folder/(name+'.npy'), value, allow_pickle=False)
                if arm == 'on' and name.startswith('llm_ffn_') and name.endswith('_mlp_in'):
                    inputs[name] = array_sha(value)
            if arm == 'on':
                for i in range(32):
                    exact(arrays[f'llm_ffn_{i}_down_out'], arrays[f'llm_ffn_{i}_mlp_out'])
    finally:
        lib.index_echo_free(ctx)
    (args.output/'identity.json').write_text(json.dumps(dict(native_off_on_off_exact=True,
        decoder_layers=32, inputs=inputs, use_gpu=False, scope='Prefill only; no decoded/model acceptance'), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true')
    for name in ['lib', 'model', 'audio', 'output']:
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if args.native:
        native(args)
        return
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    from gguf import GGUFReader
    from echo_owned_ffn import load_layer, attribute, source_helper
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])/'echo-owned-ffn'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=False)
    os.environ.update(TMPDIR=str(scratch), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert source == os.environ['GITHUB_SHA']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'ggml', text=True).strip() == GGML
    import torch
    torch.set_num_threads(4)
    report = dict(source=source, ggml=GGML, original_revision=ORIGINAL, converted_revision=CONVERTED,
        torch=torch.__version__, scope='CPU same-native-input FFN attribution only; no full original decoder resident',
        runs=[], files={}, layers={}, complete=False, speech_acceptance=False, q4_acceptance=False,
        gpu_acceptance=False, production_defaults_changed=False)
    source_helper()
    method = out/'method'
    method.mkdir()
    for name in ['tools/ci-heavy/echo_owned_ffn.py', 'tools/ci-heavy/echo_owned_ffn_run.py',
        'tools/reference_backends/index_echo_ffn_capture.py', 'src/index_echo.cpp', 'src/index_echo.h',
        'src/core/index_echo_ffn_capture.h', 'examples/talk-llama/llama-graph.cpp',
        'examples/talk-llama/models/qwen35.cpp', 'tools/ci-heavy/echo-owned-ffn/selftest.py', 'tools/ci-heavy/echo-owned-ffn/README.md', '.github/workflows/heavy-cpu.yml', 'LICENSE', 'ggml/LICENSE']:
        target = method/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    from reference_backends.index_echo_ffn_capture import _locate_modeling_file
    shutil.copyfile(_locate_modeling_file(), method/'original-modeling-qwen3_5.py')
    api = HfApi()
    assert not api.repo_info(DATASET, repo_type='dataset').private
    assert not api.file_exists(DATASET, PREFIX+'/receipt.json', repo_type='dataset')
    sent = {}
    def checkpoint(stage):
        payloads = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                    for p in sorted(out.rglob('*')) if p.is_file() and p != out/'receipt.json']
        report['stage'], report['payloads'] = stage, payloads
        report['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt = out/'receipt.json'
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        changed = [row for row in payloads if sent.get(row['path']) != row['sha256']]
        ops = [CommitOperationAdd(path_in_repo=PREFIX+'/'+r['path'], path_or_fileobj=str(out/r['path'])) for r in changed]
        ops.append(CommitOperationAdd(path_in_repo=PREFIX+'/receipt.json', path_or_fileobj=str(receipt)))
        commit = api.create_commit(DATASET, repo_type='dataset', commit_message='Echo9B owned-input FFN: '+stage, operations=ops)
        sent.update({row['path']:row['sha256'] for row in changed})
        print('PUBLIC_CHECKPOINT', stage, commit.oid, flush=True)
    def run(label, argv):
        with (out/(label+'.log')).open('xb') as log:
            result = subprocess.run(list(map(str, argv)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        report['runs'].append(dict(label=label, argv=list(map(str, argv)), exit=result.returncode))
        checkpoint(label)
        assert result.returncode == 0, label
    try:
        checkpoint('source-preflight')  # Prove public write access before builds/downloads.
        run('tiny-controls', [sys.executable, ROOT/'tools/ci-heavy/echo-owned-ffn/selftest.py', '-v'])
        required = sum(row[3] for row in FILES.values()) + 8*1024**3
        assert shutil.disk_usage(scratch).free > required, 'Insufficient disk for exact bounded artifact set'
        if not shutil.which('ccache') or not shutil.which('ninja'):
            run('apt-update', ['bash', ROOT/'tools/ci-apt.sh', 'update'])
            run('install-build-tools', ['bash', ROOT/'tools/ci-apt.sh', 'install', '-y', 'ccache', 'ninja-build'])
        build = scratch/'build'
        run('configure', ['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache', '-DBUILD_SHARED_LIBS=ON',
            '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
            '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'])
        run('build', ['cmake', '--build', build, '--target', 'crispasr-lib', '-j3'])
        library = next(build.rglob('libcrispasr.so'))
        run('abi-exports', ['nm', '-D', '--defined-only', library])
        exports = (out/'abi-exports.log').read_text()
        for name in ['index_echo_context_default_params', 'index_echo_init_from_file', 'index_echo_free',
                     'index_echo_compute_mel', 'index_echo_run_encoder', 'index_echo_prefill',
                     'index_echo_stage', 'index_echo_prompt_ids', 'index_echo_decoder_layers']:
            assert any(line.split()[-1] == name for line in exports.splitlines() if line.split()), name
        report['library_sha256'] = sha(library)
        binaries = out/'libs'
        binaries.mkdir()
        for file in build.rglob('*.so*'):
            if file.is_file():
                target = binaries/file.name
                if target.exists():
                    assert sha(target) == sha(file)
                else:
                    shutil.copyfile(file, target)
        checkpoint('binary-proof')
        paths = {}
        for label, (repo, revision, name, size, digest) in FILES.items():
            path = Path(hf_hub_download(repo, name, revision=revision))
            assert path.stat().st_size == size and sha(path) == digest, label
            paths[label] = path
            report['files'][label] = dict(repo=repo, revision=revision, path=name, bytes=size, sha256=digest)
            checkpoint('download-'+label)
        models = scratch/'models'
        models.mkdir()
        primary = models/'index-echo-9b-f16.gguf'
        primary.symlink_to(paths['primary'])
        # Filename is the unchanged primary metadata companion; physical bytes are Q4.
        (models/'index-echo-9b-decoder-f16.gguf').symlink_to(paths['decoder'])
        physical = GGUFReader(str(paths['decoder']))
        tensors = [dict(name=t.name, type=t.tensor_type.name, shape=list(map(int,t.shape)), bytes=int(t.n_bytes)) for t in physical.tensors]
        assert any(t['type']=='Q4_K' for t in tensors)
        (out/'physical-decoder.json').write_text(json.dumps(tensors,indent=2)+'\n')
        physical.data._mmap.close()
        del physical
        for case, filename in CASES.items():
            folder = out/case
            folder.mkdir()
            shutil.copyfile(ROOT/'samples'/filename, folder/'audio.wav')
            run('native-'+case, [sys.executable, __file__, '--native', '--lib', library, '--model', primary,
                '--audio', folder/'audio.wav', '--output', folder/'native'])
        config_path = Path(hf_hub_download('IndexTeam/Index-Echo-S2TT-9B', 'llm/config.json', revision=ORIGINAL))
        shutil.copyfile(config_path, method/'original-llm-config.json')
        config = json.loads(config_path.read_text())
        config = config.get('text_config', config)
        assert (config['num_hidden_layers'],config['hidden_size'],config['intermediate_size']) == (32,4096,12288)
        from safetensors import safe_open
        with safe_open(str(paths['original']), framework='pt', device='cpu') as reader:
            expected = {f'model.layers.{i}.mlp.{p}_proj.weight' for i in range(32) for p in ['gate','up','down']}
            assert {k for k in reader.keys() if '.mlp.' in k} == expected
            for layer in range(32):
                mlp, ledger = load_layer(reader, layer, config)
                results = {}
                for case in CASES:
                    folder = out/case
                    native_arrays = {s:np.load(folder/'native/on'/f'llm_ffn_{layer}_{s}.npy', allow_pickle=False) for s in SUFFIXES}
                    identity = json.loads((folder/'native/identity.json').read_text())
                    results[case] = attribute(mlp, native_arrays, identity['inputs'][f'llm_ffn_{layer}_mlp_in'], folder/'original'/f'layer-{layer:02d}')
                report['layers'][str(layer)] = dict(coefficients=ledger, results=results)
                del mlp
                checkpoint('layer-'+str(layer))
        assert len(report['layers']) == 32
        report['complete'] = True
        checkpoint('final')
        print('OWNED_FFN_DIAGNOSTIC_COMPLETE_ACCEPTANCE_FALSE', flush=True)
    finally:
        if not report['complete']:
            checkpoint('failed-partial')


if __name__ == '__main__':
    main()
