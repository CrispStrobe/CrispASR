#!/usr/bin/env python3
"""Full-model 2B observer identity; distinct from Q4 attribution or promotion."""
import argparse
import ctypes as C
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GGML = '0a862076567afd8468822ccc512c97ac6a69b340'
SOURCE = '5d98a34d9685869b11e9c94d01dee22a8b8e53b5'
MODEL = 'b34538267099a6769111aada860d57f312894510'
CAPTURE_SOURCE = '0dfccc7d01b94a8975c0cc9c6afac83471660833'
TRANSFORMERS = '5.6.0'
MODELING_SHA = '3e2b6239e4b2c3e512d4f9836a1dba12e975ae652407e84f2d3c2beebf0c9528'
MLP_SHA = 'c4e763e69d5676e71d222060e36258f6b2cc939a3597118c4207d370d120a265'
INFER_SHA = '558b2a7d5f8f2ba9ec7651562b1a40602ef06c40f75ebbb2e690fb82fd5d347e'
WEIGHTS = {
    'audio_tower.safetensors': '0dc5e5ac9c5cdc6704f5565252beb145a3eab0b24ab331a4f9c4d769f42cd074',
    'connector.safetensors': 'ec0d62780536a76b19d306759af086295a9d29ca82eae47b6a6ba3c993162073',
    'llm/model-00000-of-00001.safetensors': '32ecbbf705b6a8c6a850e065a5bfd1a9598620165da6ad8e2a075e01d973958f',
    'index-echo-2b-f16.gguf': 'fcb3f431c5898ce2029b96b04ef01d9bb8ed913ab1aad3d0ee01e1e17f8a0f44',
    'index-echo-2b-decoder-f16.gguf': '3f503cbd23ba45fd3b9f83db4d7d0060cb95ba227b5f133237ed539ad2bdde44',
}
ARMS = ['off', 'on', 'off2']
SUFFIXES = ['mlp_in', 'gate_out', 'up_out', 'down_in', 'down_out', 'mlp_out']
CASES = {'jfk': 'jfk.wav', 'zh': 'paraformer_zh.wav'}
sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def exact(a, b):
    if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
        raise ValueError('Observer changed tensor bytes, shape or dtype')
    if not a.size or not np.isfinite(a).all():
        raise ValueError('Empty/nonfinite tensor')


def ffn_names(layers):
    return {f'llm_ffn_{i}_{suffix}' for i in range(layers) for suffix in SUFFIXES}


def old_names(layers=24, encoder_layers=32):
    return ({'mel_spectrogram', 'encoder_input', 'encoder_output', 'connector_output', 'prompt_ids',
             'llm_logits', 'teacherforced_logits', 'ln_post_out', 'proj1_out', 'conv1_out', 'conv2_out', 'conv3_out'}
            | {f'encoder_layer_{i}' for i in range(encoder_layers)}
            | {f'llm_block_{i}' for i in range(layers)})


def compare_capture(before, enabled, after, layers, required_old=None, widths=None):
    """Exact identity only; cosine alone would miss a uniform rescaling."""
    old = set(before)
    new = ffn_names(layers)
    if not old or any(k.startswith('llm_ffn_') for k in old):
        raise ValueError('OFF capture is empty or contains enabled observer stages')
    if required_old is not None and old != required_old:
        raise ValueError('Incomplete existing stage inventory')
    if set(after) != old or set(enabled) != old | new:
        raise ValueError('Missing, unexpected, or leaked stage')
    for name in sorted(old):
        exact(before[name], enabled[name])
        exact(before[name], after[name])
    for i in range(layers):
        arrays = {s: enabled[f'llm_ffn_{i}_{s}'] for s in SUFFIXES}
        for v in arrays.values():
            if v.ndim != 1 or not v.size or not np.isfinite(v).all():
                raise ValueError('FFN public stages must be complete finite last-token vectors')
        if len({arrays[s].size for s in ['mlp_in', 'down_out', 'mlp_out']}) != 1:
            raise ValueError('Hidden widths disagree')
        if len({arrays[s].size for s in ['gate_out', 'up_out', 'down_in']}) != 1:
            raise ValueError('Intermediate widths disagree')
        exact(arrays['down_out'], arrays['mlp_out'])
        if widths is not None and (arrays['mlp_in'].size, arrays['gate_out'].size) != widths:
            raise ValueError('FFN dimensions differ from the pinned model configuration')
    return dict(old_arrays=len(old), new_arrays=len(new), exact=True)


def read_reference(path):
    from gguf import GGUFReader
    r = GGUFReader(str(path))
    arrays = {t.name: np.array(t.data).reshape(tuple(reversed(t.shape))) for t in r.tensors}
    meta = {k.removeprefix('crispasr.ref.'): v.contents()
            for k, v in r.fields.items() if k.startswith('crispasr.ref.')}
    r.data._mmap.close()
    return arrays, meta


def observer_environment(parent, enabled, directory):
    env = {k: v for k, v in parent.items()
           if not k.startswith(('INDEX_ECHO_', 'CRISPASR_INDEX_ECHO_', 'CRISP_AUDIO_DUMP_'))}
    env.update(CRISP_AUDIO_DUMP_STAGES=str(directory), CRISPASR_INDEX_ECHO_DUMP_STAGES=str(directory),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    if enabled:
        env['CRISPASR_INDEX_ECHO_DUMP_FFN'] = '1'
    return env


def cue_text(cue):
    """CLI and Session append translation only when it is nonempty."""
    return cue['transcript'] + ('\n' + cue['translation'] if cue['translation'] else '')


class Params(C.Structure):
    _fields_ = [('n_threads', C.c_int), ('verbosity', C.c_int), ('use_gpu', C.c_bool), ('flash_attn', C.c_bool)]


class Cue(C.Structure):
    _fields_ = [('start_seconds', C.c_double), ('end_seconds', C.c_double),
               ('transcript', C.c_char_p), ('translation', C.c_char_p)]


class Result(C.Structure):
    _fields_ = [('n_cues', C.c_int), ('cues', C.POINTER(Cue)), ('raw_text', C.c_char_p), ('parse_warnings', C.c_int)]


def native_child(args):
    """Actual backend C interface, same loaded context across OFF/ON/OFF."""
    import soundfile as sf
    lib = C.CDLL(str(args.lib))
    fp, ip, vp = C.POINTER(C.c_float), C.POINTER(C.c_int), C.c_void_p
    signatures = {
        'context_default_params': ([], Params),
        'init_from_file': ([C.c_char_p, Params], vp), 'free': ([vp], None),
        'compute_mel': ([vp, fp, C.c_int, ip, ip], fp),
        'run_encoder': ([vp, fp, C.c_int, C.c_int, ip, ip], fp),
        'prefill': ([vp, fp, C.c_int, C.c_int, ip], fp),
        'decode_token': ([vp, C.c_int32, ip], fp),
        'stage': ([vp, C.c_char_p, ip], fp), 'prompt_ids': ([vp, ip], C.POINTER(C.c_int32)),
        'decoder_layers': ([vp], C.c_int), 'transcribe': ([vp, fp, C.c_int], C.POINTER(Result)),
        'result_free': ([C.POINTER(Result)], None),
    }
    for name, (inputs, output) in signatures.items():
        fn = getattr(lib, 'index_echo_' + name)
        fn.argtypes, fn.restype = inputs, output
    free = C.CDLL(None).free
    free.argtypes, free.restype = [vp], None
    audio, rate = sf.read(args.audio, dtype='float32')
    assert rate == 16000 and audio.ndim == 1 and np.isfinite(audio).all()
    audio = np.ascontiguousarray(audio)
    data = audio.ctypes.data_as(fp)
    reference, _ = read_reference(args.reference)
    ids = reference['generated_ids'].reshape(-1)
    assert len(ids) >= 16 and np.array_equal(ids, np.rint(ids))
    directory = args.destination
    directory.mkdir(parents=True, exist_ok=False)
    clean = dict(os.environ)
    params = lib.index_echo_context_default_params()
    params.n_threads, params.verbosity, params.use_gpu = 4, 0, False
    assert params.flash_attn is True
    ctx = lib.index_echo_init_from_file(str(args.model).encode(), params)
    assert ctx, 'Actual native model initialization failed'

    def owned(pointer, count, shape=None):
        assert pointer and count > 0
        try:
            a = np.ctypeslib.as_array(pointer, shape=(count,)).copy()
            assert np.isfinite(a).all()
            return a.reshape(shape) if shape else a
        finally:
            free(C.cast(pointer, vp))

    try:
        layers = lib.index_echo_decoder_layers(ctx)
        assert layers == 24
        for arm in ARMS:
            dest = directory / arm
            frontend = dest / 'frontend'
            frontend.mkdir(parents=True)
            env = observer_environment(clean, arm == 'on', frontend)
            for key in list(os.environ):
                if key.startswith(('INDEX_ECHO_', 'CRISPASR_INDEX_ECHO_', 'CRISP_AUDIO_DUMP_')):
                    del os.environ[key]
            os.environ.update(env)
            mels, frames, rows, width, vocab = [C.c_int() for _ in range(5)]
            mel = owned(lib.index_echo_compute_mel(ctx, data, len(audio), C.byref(mels), C.byref(frames)),
                        mels.value * frames.value)
            emb = owned(lib.index_echo_run_encoder(ctx, mel.ctypes.data_as(fp), mels.value, frames.value,
                        C.byref(rows), C.byref(width)), rows.value * width.value)
            logits = owned(lib.index_echo_prefill(ctx, emb.ctypes.data_as(fp), rows.value, width.value,
                           C.byref(vocab)), vocab.value)
            arrays = dict(mel_spectrogram=mel, connector_output=emb, llm_logits=logits)
            count = C.c_int()
            pointer = lib.index_echo_prompt_ids(ctx, C.byref(count))
            assert pointer and count.value > 0
            arrays['prompt_ids'] = np.ctypeslib.as_array(pointer, shape=(count.value,)).copy()
            for name in sorted({f'llm_block_{i}' for i in range(layers)} | ffn_names(layers)):
                count.value = 0
                pointer = lib.index_echo_stage(ctx, name.encode(), C.byref(count))
                if pointer:
                    arrays[name] = np.ctypeslib.as_array(pointer, shape=(count.value,)).copy()
                elif not name.startswith('llm_ffn_') or arm == 'on':
                    raise ValueError('Missing native stage ' + name)
            trace = [logits]
            for token in ids[:15]:
                trace.append(owned(lib.index_echo_decode_token(ctx, int(token), C.byref(vocab)), vocab.value))
            arrays['teacherforced_logits'] = np.stack(trace)
            mapping = {k: k for k in ['conv1_out', 'conv2_out', 'conv3_out', 'encoder_input', 'ln_post_out', 'proj1_out']}
            mapping['encoder_out'] = 'encoder_output'
            mapping.update({f'enc_blk{i:02d}_out': f'encoder_layer_{i}' for i in range(32)})
            for filename, name in mapping.items():
                arrays[name] = np.fromfile(frontend / (filename + '.f32'), dtype='<f4')
            for name, array in arrays.items():
                assert array.size and np.isfinite(array).all(), name
                np.save(dest / (name + '.npy'), array, allow_pickle=False)
            transcribe_frontend = dest / 'transcribe-frontend'
            transcribe_frontend.mkdir()
            os.environ['CRISP_AUDIO_DUMP_STAGES'] = str(transcribe_frontend)
            result = lib.index_echo_transcribe(ctx, data, len(audio))
            assert result
            try:
                r = result.contents
                decoded = dict(raw_text=r.raw_text.decode(), parse_warnings=r.parse_warnings,
                    cues=[dict(start=c.start_seconds, end=c.end_seconds, transcript=c.transcript.decode(),
                               translation=c.translation.decode()) for c in r.cues[:r.n_cues]])
                assert decoded['cues'] and decoded['raw_text'] and r.parse_warnings == 0
                (dest / 'decoded.json').write_text(json.dumps(decoded, ensure_ascii=False, indent=2) + '\n')
                # Observe the real generation prefill too, without overwriting its proof.
                for name in sorted(k for k in arrays if k.startswith(('llm_block_', 'llm_ffn_'))):
                    count.value = 0
                    pointer = lib.index_echo_stage(ctx, name.encode(), C.byref(count))
                    assert pointer and count.value > 0, name
                    actual = np.ctypeslib.as_array(pointer, shape=(count.value,)).copy()
                    np.save(dest / ('generation-' + name + '.npy'), actual, allow_pickle=False)
                    exact(arrays[name], actual)
            finally:
                lib.index_echo_result_free(result)
    finally:
        lib.index_echo_free(ctx)
    # Actual automatic Session/C-ABI detection, independent of the CLI filename pass.
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    renamed = args.model.parent / ('metadata-only-' + args.audio.stem + '.gguf')
    assert not renamed.exists()
    renamed.symlink_to(args.model)
    try:
        os.environ.update(observer_environment(clean, True, directory / 'on' / 'transcribe-frontend'))
        with Session(str(renamed), lib_path=str(args.lib), n_threads=4) as session:
            assert session.backend == 'index-echo'
            session.set_target_language('en')
            segments = session.transcribe(audio)
            actual = [dict(start=s.start, end=s.end, text=s.text) for s in segments]
        expected = json.loads((directory / 'on' / 'decoded.json').read_text())
        golden = [dict(start=c['start'], end=c['end'], text=cue_text(c)) for c in expected['cues']]
        assert len(actual) == len(golden) and all(a['text'] == b['text'] and abs(a['start']-b['start']) <= .0051
            and abs(a['end']-b['end']) <= .0051 for a, b in zip(actual, golden))
        (directory / 'session.json').write_text(json.dumps(dict(backend='index-echo', segments=actual), ensure_ascii=False, indent=2) + '\n')
    finally:
        renamed.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    for name in ['lib', 'model', 'audio', 'reference', 'destination']:
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.child:
        native_child(args)
        return
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE='1')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert source == os.environ['GITHUB_SHA']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'ggml', text=True).strip() == GGML
    import torch
    assert torch.__version__ in ['2.6.0', '2.6.0+cpu']
    assert importlib.metadata.version('transformers') == TRANSFORMERS
    sys.path.insert(0, str(ROOT / 'tools'))
    from reference_backends.index_echo_ffn_capture import verify_source
    provenance = verify_source(expected_version=TRANSFORMERS, expected_modeling_sha256=MODELING_SHA, expected_class_sha256=MLP_SHA)
    report = dict(source=source, capture_source=CAPTURE_SOURCE, ggml=GGML, original_revision=SOURCE, model_revision=MODEL,
        provenance=provenance, torch=torch.__version__, platform=sys.platform, runs=[], cases={},
        observer_identity_passed=False, stage_parity_passed=False, decoded_parity_passed=False,
        model_acceptance=False, q4_acceptance=False, gpu_acceptance=False,
        production_defaults_changed=False)
    report['packages'] = {name: importlib.metadata.version(name) for name in
        ['torch', 'torchaudio', 'transformers', 'numpy', 'gguf', 'huggingface-hub', 'safetensors', 'tokenizers', 'soundfile']}
    report['settings'] = dict(threads=4, use_gpu=False, flash_attn=True, temperature=0,
        max_new_tokens=2000, trace_tokens=16, target_lang='en', original_dtype='float32',
        source_decoder_layers=24, source_hidden_size=2048, source_intermediate_size=6144,
        scope='2B direct-window observer identity and numerical/decoded controls; no 9B/Q4 attribution')

    def checkpoint():
        report['payloads'] = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                              for p in sorted(out.rglob('*')) if p.is_file() and p != out / 'receipt.json']
        (out / 'receipt.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    def run(label, argv, env=None, required=True):
        with (out / (label + '.log')).open('xb') as log:
            result = subprocess.run(list(map(str, argv)), cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=7200)
        report['runs'].append(dict(label=label, argv=list(map(str, argv)), exit=result.returncode))
        checkpoint()
        print(label, result.returncode, flush=True)
        if required:
            assert result.returncode == 0, label
        return result.returncode

    try:
        method = out / 'method'
        method.mkdir()
        for name in ['tools/ci-heavy/echo_ffn_transparency.py', 'tools/ci-heavy/echo-ffn-transparency/selftest.py',
                     'tools/dump_reference.py', 'tools/reference_backends/index_echo.py',
                     'tools/reference_backends/index_echo_ffn_capture.py', 'src/index_echo.cpp', 'src/index_echo.h',
                     'src/core/index_echo_ffn_capture.h', 'examples/cli/crispasr_diff_main.cpp',
                     'src/crispasr_c_api.cpp', 'examples/cli/crispasr_backend_index_echo.cpp',
                     'tools/ci-heavy/echo_ffn_literal_graph.py', 'tests/test-index-echo-ffn-capture.cpp', 'tools/ci-apt.sh',
                     'examples/talk-llama/llama-graph.cpp', 'examples/talk-llama/llama-graph.h',
                     'examples/talk-llama/models/qwen35.cpp']:
            dest = method / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        shutil.copyfile(provenance['modeling_file'], method / 'original-modeling-qwen3_5.py')
        dist = importlib.metadata.distribution('transformers')
        for name in ['activations.py', 'models/qwen3_5/configuration_qwen3_5.py',
                     'models/qwen3_omni_moe/modeling_qwen3_omni_moe.py',
                     'models/qwen3_omni_moe/configuration_qwen3_omni_moe.py',
                     'models/whisper/feature_extraction_whisper.py']:
            dest = method / 'transformers' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dist.locate_file('transformers/' + name), dest)
        for i, path in enumerate(p for p in dist.files if Path(p).name in ['LICENSE', 'LICENSE.txt', 'LICENSE.md', 'NOTICE']):
            shutil.copyfile(dist.locate_file(path), out / f'Transformers-LICENSE-{i}')
        for name in ['LICENSE', 'ggml/LICENSE']:
            shutil.copyfile(ROOT / name, out / ('GGML-LICENSE' if '/' in name else 'CrispASR-LICENSE'))
        run('worker-selftest', [sys.executable, ROOT / 'tools/ci-heavy/echo-ffn-transparency/selftest.py'])
        if not shutil.which('ffmpeg'):
            run('apt-update-audio-tools', ['bash', ROOT / 'tools/ci-apt.sh', 'update'])
            run('install-ffmpeg', ['bash', ROOT / 'tools/ci-apt.sh', 'install', '-y', 'ffmpeg'])
        assert shutil.which('ffmpeg'), 'Original encode_audio requires the actual ffmpeg executable'
        run('ffmpeg-version', [shutil.which('ffmpeg'), '-version'])
        run('fetch-capture-source', ['git', 'fetch', '--depth=1', 'origin', CAPTURE_SOURCE])
        run('capture-runtime-identity', ['git', 'diff', '--exit-code', CAPTURE_SOURCE, 'HEAD', '--',
            'src', 'include', 'ggml', 'examples', 'python', 'bindings', 'tools/dump_reference.py', 'tools/reference_backends'])
        # Exercise the real released loader mapping before any large download.
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from transformers import AutoConfig, AutoModelForCausalLM
        config = Path(hf_hub_download('IndexTeam/Index-Echo-S2TT-2B', 'llm/config.json', revision=SOURCE,
                                     local_dir=scratch / 'loader-preflight'))
        assert sha(config) == 'ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4'
        cls = AutoModelForCausalLM._model_mapping[type(AutoConfig.from_pretrained(config.parent))]
        assert cls.__name__ == 'Qwen3_5ForCausalLM'
        report['loader_class'] = cls.__module__ + '.' + cls.__name__
        build = scratch / 'build'
        run('configure', ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
            '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_VULKAN=OFF',
            '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'])
        run('build', ['cmake', '--build', build, '--target', 'crispasr-lib', 'crispasr-cli', 'crispasr-diff', '-j3'])
        library = next(p for pattern in ['libcrispasr.so', 'libcrispasr.dylib'] for p in build.rglob(pattern))
        symbols = C.CDLL(str(library))
        assert all(hasattr(symbols, 'index_echo_' + s) for s in ['init_from_file', 'context_default_params',
            'compute_mel', 'run_encoder', 'prefill', 'decode_token', 'stage', 'prompt_ids', 'decoder_layers',
            'transcribe', 'result_free', 'free'])
        for target in ['crispasr', 'crispasr-diff']:
            shutil.copyfile(build / 'bin' / target, out / target)
        libs = out / 'libs'
        libs.mkdir()
        for pattern in ['*.so*', '*.dylib']:
            for p in build.rglob(pattern):
                if p.is_file():
                    dest = libs / p.name
                    if dest.exists():
                        assert sha(p) == sha(dest)
                    else:
                        shutil.copyfile(p, dest)
        run('fetch-old-observer', ['git', 'fetch', '--depth=1', 'origin', 'd8183cf92abc24eeeb6c969db563e968c5345d46'])
        run('literal-builder-controls', [sys.executable, ROOT / 'tools/ci-heavy/echo_ffn_literal_graph.py',
            '--libraries', libs, '--output', out / 'literal-builder'])
        run('wiring', [sys.executable, ROOT / 'tools/check-backend-wiring.py', '--crispasr', build / 'bin/crispasr',
                       '--lib', library, '--require-lib'])
        api = HfApi()
        pins = {}
        seen_weights = set()
        for repo, revision, directory, patterns in [
            ('IndexTeam/Index-Echo-S2TT-2B', SOURCE, 'original', ['*.py', '*.json', '*.safetensors', '*.txt', '*.jinja', '*.model', '*LICENSE*']),
            ('cstr/index-echo-2b-GGUF', MODEL, 'native', ['index-echo-2b-f16.gguf', 'index-echo-2b-decoder-f16.gguf'])]:
            info = api.model_info(repo, revision=revision, files_metadata=True)
            assert info.sha == revision
            folder = Path(snapshot_download(repo, revision=revision, local_dir=scratch / directory, allow_patterns=patterns))
            records = []
            for sibling in info.siblings:
                path = folder / sibling.rfilename
                if not path.is_file():
                    continue
                raw_sha = sha(path)
                if sibling.rfilename in WEIGHTS:
                    assert raw_sha == WEIGHTS[sibling.rfilename]
                    seen_weights.add(sibling.rfilename)
                if sibling.lfs:
                    assert raw_sha == sibling.lfs.sha256 and path.stat().st_size == sibling.lfs.size
                else:
                    data = path.read_bytes()
                    assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == sibling.blob_id
                records.append(dict(path=sibling.rfilename, bytes=path.stat().st_size, sha256=raw_sha))
            pins[directory] = dict(repo=repo, revision=revision, files=records)
            (out / 'model-provenance.json').write_text(json.dumps(pins, indent=2) + '\n')
        assert seen_weights == set(WEIGHTS)
        original, models = scratch / 'original', scratch / 'native'
        assert sha(original / 'infer.py') == INFER_SHA
        text_config = json.loads((original / 'llm/config.json').read_text())['text_config']
        assert (text_config['num_hidden_layers'], text_config['hidden_size'], text_config['intermediate_size']) == (24, 2048, 6144)
        assert json.loads((original / 'audio_config.json').read_text())['encoder_layers'] == 32
        for name in ['infer.py', 'audio_config.json', 'llm/config.json', 'requirements.txt', 'llm/LICENSE']:
            dest = method / 'original' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original / name, dest)
        from gguf import GGUFReader
        physical = {}
        for path in sorted(models.glob('*.gguf')):
            r = GGUFReader(str(path))
            rows = [dict(name=t.name, type=t.tensor_type.name, shape=list(map(int, t.shape)), bytes=int(t.n_bytes)) for t in r.tensors]
            assert rows and all(t['type'] in ['F16', 'F32'] for t in rows)
            physical[path.name] = rows
            r.data._mmap.close()
        (out / 'physical-model-audit.json').write_text(json.dumps(physical, indent=2) + '\n')
        clean = observer_environment(os.environ, False, scratch)
        for key in ['CRISP_AUDIO_DUMP_STAGES', 'CRISPASR_INDEX_ECHO_DUMP_STAGES']:
            clean.pop(key)
        for case, filename in CASES.items():
            directory = out / case
            directory.mkdir()
            audio = ROOT / 'samples' / filename
            shutil.copyfile(audio, directory / 'audio.wav')
            errors = []
            for arm in ARMS:
                env = clean | dict(INDEX_ECHO_REF_DEVICE='cpu', INDEX_ECHO_REF_DTYPE='float32',
                    INDEX_ECHO_REF_FP32_DECODER='1', INDEX_ECHO_REF_THREADS='4', INDEX_ECHO_TARGET_LANG='en',
                    INDEX_ECHO_REF_MAX_TOKENS='2000', INDEX_ECHO_TRACE_TOKENS='16')
                if arm == 'on':
                    env.update(INDEX_ECHO_REF_CAPTURE_FFN='1', INDEX_ECHO_REF_TRANSFORMERS_VERSION=TRANSFORMERS,
                               INDEX_ECHO_REF_MODELING_SHA256=MODELING_SHA, INDEX_ECHO_REF_MLP_SHA256=MLP_SHA)
                if run(case + '-original-' + arm, [sys.executable, ROOT / 'tools/dump_reference.py', '--backend',
                    'index-echo', '--model-dir', original, '--audio', audio, '--output', directory / (arm + '-ref.gguf')], env, False):
                    errors.append('original-' + arm)
            if not errors:
                try:
                    refs = [read_reference(directory / (a + '-ref.gguf')) for a in ARMS]
                    identity = compare_capture(*(r[0] for r in refs), 24,
                        required_old=old_names() | {'generated_ids'}, widths=(2048, 6144))
                    for key in ['generated_text', 'parameter_dtypes', 'target_lang']:
                        assert refs[0][1][key] == refs[1][1][key] == refs[2][1][key], key
                    assert all(r[1]['aliasing_after_capture'] == 'none' and r[1]['target_lang'] == 'en' for r in refs)
                    precision = json.loads(refs[0][1]['parameter_dtypes'])
                    assert all(set(precision[key]['parameter_elements']) == {'torch.float32'} for key in ['tower', 'connector', 'llm'])
                    assert refs[0][1]['generated_text'] and refs[1][1]['ffn_source_provenance']
                    report['cases'][case] = dict(original=identity)
                except Exception as exc:
                    errors.append('original-identity: ' + repr(exc))
            golden = directory / 'on-ref.gguf'
            if golden.exists():
                code = run(case + '-native', [sys.executable, __file__, '--child', '--lib', library, '--model',
                    models / 'index-echo-2b-f16.gguf', '--audio', audio, '--reference', golden,
                    '--destination', directory / 'native'], clean, False)
                if code:
                    errors.append('native-child')
                else:
                    try:
                        arrays = [{p.stem: np.load(p, allow_pickle=False) for p in (directory / 'native' / arm).glob('*.npy')
                                   if not p.name.startswith('generation-')} for arm in ARMS]
                        identity = compare_capture(*arrays, 24, required_old=old_names(), widths=(2048, 6144))
                        outputs = [json.loads((directory / 'native' / arm / 'decoded.json').read_text()) for arm in ARMS]
                        assert outputs[0] == outputs[1] == outputs[2]
                        report['cases'].setdefault(case, {})['native'] = identity
                        original_text = read_reference(golden)[1]['generated_text']
                        report['cases'][case]['decoded_original_native_exact'] = outputs[0]['raw_text'] == original_text
                        if not report['cases'][case]['decoded_original_native_exact']:
                            errors.append('original-native-decoded-text')
                    except Exception as exc:
                        errors.append('native-identity: ' + repr(exc))
                for arm in ARMS:
                    (directory / 'native' / arm / 'cli-frontend').mkdir(parents=True, exist_ok=False)
                    env = observer_environment(clean, arm == 'on', directory / 'native' / arm / 'cli-frontend')
                    if run(case + '-cli-' + arm, [build / 'bin/crispasr', '-ng', '-m', models / 'index-echo-2b-f16.gguf',
                        '-f', audio, '-l', 'en', '-osrt', '-of', directory / ('cli-' + arm), '-t', '4'], env, False):
                        errors.append('cli-' + arm)
                try:
                    srts = [(directory / ('cli-' + arm + '.srt')).read_bytes() for arm in ARMS]
                    assert srts[0] and srts[0] == srts[1] == srts[2]
                    report['cases'].setdefault(case, {})['cli_identity'] = True
                except Exception as exc:
                    errors.append('cli-identity: ' + repr(exc))
                for arm in ['off', 'on']:
                    if run(case + '-diff-' + arm, [build / 'bin/crispasr-diff', 'index-echo',
                        models / 'index-echo-2b-f16.gguf', directory / (arm + '-ref.gguf'), audio],
                        clean | dict(CRISPASR_DIFF_NO_GPU='1'), False):
                        errors.append('original-native-diff-' + arm)
            report['cases'].setdefault(case, {})['errors'] = errors
            checkpoint()
        complete = set(report['cases']) == set(CASES)
        report['observer_identity_passed'] = complete and all('original' in r and 'native' in r
            and r.get('cli_identity') is True for r in report['cases'].values())
        report['stage_parity_passed'] = complete and all(not any(e.startswith('original-native-diff-') for e in r['errors'])
            and any(v['label'] == case + '-diff-on' and v['exit'] == 0 for v in report['runs']) for case, r in report['cases'].items())
        report['decoded_parity_passed'] = complete and all(r.get('decoded_original_native_exact') is True for r in report['cases'].values())
        assert all(not r['errors'] for r in report['cases'].values()) and report['observer_identity_passed'] \
            and report['stage_parity_passed'] and report['decoded_parity_passed'], report['cases']
    finally:
        checkpoint()
    print('ECHO_FULLMODEL_OBSERVER_IDENTITY_PASS', source, flush=True)


if __name__ == '__main__':
    main()
