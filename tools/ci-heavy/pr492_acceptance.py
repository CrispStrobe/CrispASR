#!/usr/bin/env python3
"""PR #492 CPU A/B: mel exactness, frozen Python LM stages, CLI/session speech.

No CANN/CUDA performance claim. The frozen LM reference has no generated text;
speech acceptance separately uses the previously accepted English/Chinese clips.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import gguf
from huggingface_hub import hf_hub_download
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
BASE = '7afaf2e8559efdb0f16bae8a6bc95e145e9a18d9'  # main before the author-preserving PR merge
MODEL_REV = 'e2d7dfebf0afd8076771903e92958039c5074eab'
PINS = {
    'q4_k': ('cstr/mimo-asr-GGUF', MODEL_REV, 'mimo-asr-q4_k.gguf',
             '12dbc7cc7a20c7add6ff00bf8b12bca1c46304e0100a5c5a6e74bdecfc57a306'),
    'f16': ('cstr/mimo-asr-GGUF', MODEL_REV, 'mimo-asr-f16.gguf',
            '619172cd385561c95f2bc2c5cdcb20ef9bbc25bb542bc41ba528182e8804c0f2'),
    'codec': ('cstr/mimo-tokenizer-GGUF', 'fa380f4c49a8e8c62c02c00d0da5e263fc5b0dcf',
              'mimo-tokenizer-q4_k.gguf', '3f3a903b10294ead4ef6a4afec035639fd2113b1d307d42f649a97cc85670e3f'),
    'reference': ('cstr/mimo-asr-GGUF', MODEL_REV, 'diff-harness-ref/mimo-asr-ref.gguf',
                  '7d4dfedc32a1451751199ed694a887f5091d3131bdefa840f811bc74e6312218'),
}
STAGES = ['prefill_audio_features', 'prefill_text_embeds', 'prefill_inputs_embeds',
          'prefill_last_hidden', 'prefill_text_logits_step0']
TOK_STAGES = ['tok_mel', 'tok_conv1_out', 'tok_conv2_out', 'tok_xfmr_out', 'tok_pool_out', 'tok_codes']


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(command, log, cwd=ROOT, timeout=7200, env=None):
    with Path(log).open('wb') as f:
        result = subprocess.run(list(map(str, command)), cwd=cwd, stdout=f,
                                stderr=subprocess.STDOUT, env=env, timeout=timeout)
    print(Path(log).name, result.returncode, Path(log).read_text()[-1200:], flush=True)
    assert result.returncode == 0, str(log)


def metrics(actual, expected):
    assert actual.shape == expected.shape and np.isfinite(actual).all()
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    assert an > 0 and bn > 0
    return dict(cosine=float(np.sum(a * b) / (an * bn)), mine_norm=an, ref_norm=bn,
                relative_l2=float(np.linalg.norm(a - b) / bn),
                max_abs=float(np.max(np.abs(a - b))), exact=np.array_equal(actual, expected))


def worker(args):
    """Isolate each native model's lifetime, globals, and attention selection."""
    build, out, model, codec, reference, variant = map(Path, args[:6])
    candidate, flash = args[6] == 'candidate', args[7] == 'flash'
    out.mkdir(parents=True, exist_ok=True)
    lib_path = next(build.rglob('libcrispasr.so'))
    lib = C.CDLL(str(lib_path))
    fields = [('n_threads', C.c_int), ('verbosity', C.c_int), ('use_gpu', C.c_bool), ('temperature', C.c_float)]
    if candidate:
        fields.append(('flash_attn', C.c_bool))
    class Params(C.Structure):
        _fields_ = fields
    lib.mimo_asr_context_default_params.restype = Params
    lib.mimo_asr_init_from_file.argtypes = [C.c_char_p, Params]
    lib.mimo_asr_init_from_file.restype = C.c_void_p
    lib.mimo_asr_free.argtypes = [C.c_void_p]
    lib.mimo_asr_extract_stage.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_int,
                                          C.c_char_p, C.POINTER(C.c_int)]
    lib.mimo_asr_extract_stage.restype = C.POINTER(C.c_float)
    libc = C.CDLL(None)
    libc.free.argtypes = [C.c_void_p]
    ref = gguf.GGUFReader(str(reference))
    tensors = {t.name: t for t in ref.tensors}
    ids = np.ascontiguousarray(tensors['prefill_input_ids'].data, dtype=np.int32)
    assert ids.ndim == 2 and ids.shape[0] == 9
    params = lib.mimo_asr_context_default_params()
    params.n_threads, params.verbosity, params.use_gpu = 4, 0, False
    if candidate:
        params.flash_attn = flash
    ctx = lib.mimo_asr_init_from_file(os.fsencode(model), params)
    assert ctx, 'MiMo stage context did not load'
    stage_results = {}
    try:
        for name in STAGES:
            n = C.c_int()
            ptr = lib.mimo_asr_extract_stage(ctx, ids.ctypes.data_as(C.POINTER(C.c_int32)), ids.shape[1],
                                             name.encode(), C.byref(n))
            assert ptr and n.value == tensors[name].data.size, name
            try:
                data = np.ctypeslib.as_array(ptr, (n.value,)).copy().reshape(tensors[name].data.shape)
            finally:
                libc.free(ptr)
            np.save(out / (name + '.npy'), data)
            stage_results[name] = metrics(data, tensors[name].data)
            print(name, stage_results[name], flush=True)
    finally:
        lib.mimo_asr_free(ctx)
    (out / 'stages.json').write_text(json.dumps(stage_results, indent=2) + '\n')

    # The LM reference starts at audio codes. Cover the tokenizer independently
    # so its new eager encoder attention cannot hide behind oracle input IDs.
    tok_fields = [('n_threads', C.c_int), ('verbosity', C.c_int), ('use_gpu', C.c_bool)]
    if candidate:
        tok_fields.append(('flash_attn', C.c_bool))
    class TokParams(C.Structure):
        _fields_ = tok_fields
    lib.mimo_tokenizer_context_default_params.restype = TokParams
    lib.mimo_tokenizer_init_from_file.argtypes = [C.c_char_p, TokParams]
    lib.mimo_tokenizer_init_from_file.restype = C.c_void_p
    lib.mimo_tokenizer_free.argtypes = [C.c_void_p]
    lib.mimo_tokenizer_extract_stage.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int,
                                                C.c_char_p, C.POINTER(C.c_int)]
    lib.mimo_tokenizer_extract_stage.restype = C.POINTER(C.c_float)
    tok_params = lib.mimo_tokenizer_context_default_params()
    tok_params.n_threads, tok_params.verbosity, tok_params.use_gpu = 4, 0, False
    if candidate:
        tok_params.flash_attn = flash
    tok = lib.mimo_tokenizer_init_from_file(os.fsencode(codec), tok_params)
    assert tok
    pcm, sr = sf.read(variant / 'en.wav', dtype='float32')
    assert sr == 16000 and pcm.ndim == 1
    try:
        for name in TOK_STAGES:
            n = C.c_int()
            ptr = lib.mimo_tokenizer_extract_stage(tok, pcm.ctypes.data_as(C.POINTER(C.c_float)), len(pcm),
                                                   name.encode(), C.byref(n))
            assert ptr and n.value > 0, name
            try:
                data = np.ctypeslib.as_array(ptr, (n.value,)).copy()
            finally:
                libc.free(ptr)
            assert np.isfinite(data).all()
            np.save(out / (name + '.npy'), data)
    finally:
        lib.mimo_tokenizer_free(tok)

    # Configure the C ABI explicitly, then use the ordinary Session result reader.
    # Session's public constructor does not expose the open-params flash switch.
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    class OpenParams(C.Structure):
        _fields_ = [(x, C.c_int) for x in ['abi_version', 'n_threads', 'use_gpu', 'verbosity',
                                          'flash_attn', 'n_gpu_layers']] + [('reserved', C.c_int * 6)]
    session = Session.__new__(Session)
    session._lib, session._handle, session._progress_cb_holder = lib, None, None
    session._setup_session_signatures()
    opts = OpenParams(2, 4, 0, 0, int(flash), -1)
    session._handle = lib.crispasr_session_open_with_params(os.fsencode(model), b'mimo-asr', C.byref(opts))
    assert session._handle, 'MiMo C ABI did not load'
    session.backend, session._n_threads = 'mimo-asr', 4
    speech = {}
    with session:
        session.set_codec_path(str(codec))
        session.set_max_new_tokens(128)
        for language in ['en', 'zh']:
            audio = variant / (language + '.wav')
            pcm, sr = sf.read(audio, dtype='float32')
            assert sr == 16000 and pcm.ndim == 1
            session.set_source_language(language)
            text = ' '.join(s.text for s in session.transcribe(pcm)).strip()
            assert text
            if language == 'en':
                assert all(w in re.findall('[a-z]+', text.lower()) for w in ['americans', 'country', 'ask']), text
            else:
                assert len(re.findall('[\u4e00-\u9fff]', text)) >= 5, text
            prefix = out / ('cli-' + language)
            run([build / 'bin/crispasr', '--backend', 'mimo-asr', '-m', model, '--codec-model', codec,
                 '-ng', '-t', '4', '-l', language, '-fa' if flash else '-nfa',
                 '--max-new-tokens', '128', '-f', audio, '-otxt', '-of', prefix], out / ('cli-' + language + '.log'))
            cli_text = prefix.with_suffix('.txt').read_text().strip()
            speech[language] = dict(abi=text, cli=cli_text, audio_sha256=digest(audio))
            (out / 'speech.json').write_text(json.dumps(speech, indent=2, ensure_ascii=False) + '\n')
            assert cli_text == text, (cli_text, text)
    print('MIMO_STAGE_AND_SPEECH_WORKER_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quant', choices=['q4_k', 'f16', 'all'], default='q4_k')
    args = parser.parse_args()
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'pr492'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), HF_HOME=str(scratch / 'hf'), OMP_NUM_THREADS='4',
                      CRISPASR_GGUF_MMAP='1')
    os.environ.pop('CRISPASR_CORE_ATTN_EAGER_F32', None)
    os.environ.pop('CRISPASR_MIMO_FORCE_CPU', None)
    receipt = dict(passed=False, source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   baseline=BASE, pr_source='ac07cf0be3b528cdb035735738098e30f7468a41', pins=PINS,
                   scope=__doc__, results={})
    def save():
        (out / 'acceptance.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    probe = np.array([1., 2., 3.], dtype=np.float32)
    control = metrics(probe * 2, probe)
    assert control['cosine'] > .999999 and control['relative_l2'] == 1
    receipt['scale_negative_control'] = control
    save()
    baseline = scratch / 'baseline'
    run(['git', 'fetch', '--depth', '1', 'origin', BASE], out / 'fetch.log')
    run(['git', 'worktree', 'add', '--detach', baseline, BASE], out / 'worktree.log')
    run(['git', 'submodule', 'update', '--init', '--recursive'], out / 'submodules.log', cwd=baseline)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD:ggml'], cwd=baseline) == subprocess.check_output(['git', 'rev-parse', 'HEAD:ggml'], cwd=ROOT)
    run(['sudo', 'apt-get', 'update'], out / 'apt-update.log')
    run(['sudo', 'apt-get', 'install', '-y', 'ccache', 'ffmpeg'], out / 'apt-install.log')
    builds = {}
    for name, source in [('baseline', baseline), ('candidate', ROOT)]:
        build = scratch / ('build-' + name)
        run(['cmake', '-S', source, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
             '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DCRISPASR_MEL_BLAS=OFF',
             '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=ON'], out / ('configure-' + name + '.log'))
        run(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-lib', 'test-mimoasr-params',
             'test-mel-blas-parity', '-j4'], out / ('build-' + name + '.log'))
        for target in ['test-mimoasr-params', 'test-mel-blas-parity']:
            run([build / 'bin' / target], out / (name + '-' + target + '.log'))
        probe = scratch / ('mel-' + name)
        run(['c++', '-std=c++17', '-O3', '-fopenmp', '-pthread', '-I' + str(source / 'src'),
             ROOT / 'tools/ci-heavy/pr492_mel_probe.cpp', source / 'src/core/mel.cpp', '-o', probe],
            out / ('mel-build-' + name + '.log'))
        with (out / ('mel-' + name + '.bin')).open('wb') as f, (out / ('mel-' + name + '.log')).open('wb') as log:
            result = subprocess.run([probe], stdout=f, stderr=log, timeout=60)
        assert result.returncode == 0
        builds[name] = build
    assert (out / 'mel-baseline.bin').read_bytes() == (out / 'mel-candidate.bin').read_bytes()
    receipt['mel'] = '16 cases: 63/64/65/300 frames, both filterbank layouts and accumulators, exact baseline/candidate and 1/4 threads'
    save()
    def pinned(name):
        repo, revision, filename, sha = PINS[name]
        path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=scratch / 'models' / name))
        assert digest(path) == sha, filename
        return path
    reference, codec = pinned('reference'), pinned('codec')
    ref = gguf.GGUFReader(str(reference))
    receipt['reference_source'] = ref.fields['crispasr.ref.model_dir'].contents()
    receipt['reference_generated_text'] = ref.fields['crispasr.ref.generated_text'].contents()
    audio = scratch / 'audio'
    audio.mkdir(exist_ok=True)
    run(['ffmpeg', '-y', '-i', ROOT / 'samples/jfk.mp3', '-ar', '16000', '-ac', '1', audio / 'en.wav'], out / 'en-resample.log')
    zh = hf_hub_download('FunAudioLLM/SenseVoiceSmall', 'example/zh.mp3',
                         revision='3847d57b6bdf2dd8875cb1508d2af43d80a16bf7', local_dir=audio)
    run(['ffmpeg', '-y', '-i', zh, '-ar', '16000', '-ac', '1', audio / 'zh.wav'], out / 'zh-resample.log')
    save()
    for quant in (['q4_k', 'f16'] if args.quant == 'all' else [args.quant]):
        model = pinned(quant)
        rows = {}
        for variant, build, arm, flash in [('baseline-flash', builds['baseline'], 'baseline', 'flash'),
                                          ('candidate-flash', builds['candidate'], 'candidate', 'flash'),
                                          ('candidate-eager', builds['candidate'], 'candidate', 'eager')]:
            dest = out / quant / variant
            run([sys.executable, __file__, '--worker', build, dest, model, codec, reference, audio, arm, flash],
                out / (quant + '-' + variant + '.log'))
            rows[variant] = json.loads((dest / 'speech.json').read_text())
        comparisons = {}
        for stage in STAGES:
            b = np.load(out / quant / 'baseline-flash' / (stage + '.npy'))
            c = np.load(out / quant / 'candidate-flash' / (stage + '.npy'))
            e = np.load(out / quant / 'candidate-eager' / (stage + '.npy'))
            comparisons[stage] = dict(default=metrics(c, b), nonflash=metrics(e, b))
            assert c.tobytes() == b.tobytes(), (quant, stage, 'default changed')
            assert comparisons[stage]['nonflash']['cosine'] >= .9999 and comparisons[stage]['nonflash']['relative_l2'] <= .005, (quant, stage, comparisons[stage])
        tokenizer_comparisons = {}
        for stage in TOK_STAGES:
            b = np.load(out / quant / 'baseline-flash' / (stage + '.npy'))
            c = np.load(out / quant / 'candidate-flash' / (stage + '.npy'))
            e = np.load(out / quant / 'candidate-eager' / (stage + '.npy'))
            assert b.shape == c.shape == e.shape
            assert c.tobytes() == b.tobytes(), (stage, 'tokenizer default changed')
            if stage == 'tok_codes':
                tokenizer_comparisons[stage] = dict(default_exact=True, nonflash_exact=np.array_equal(e, b),
                                                    nonflash_match_fraction=float(np.mean(e == b)))
                assert np.array_equal(e, b), ('RVQ codes changed', tokenizer_comparisons[stage])
            else:
                tokenizer_comparisons[stage] = dict(default=metrics(c, b), nonflash=metrics(e, b))
                assert tokenizer_comparisons[stage]['nonflash']['cosine'] >= .9999 and tokenizer_comparisons[stage]['nonflash']['relative_l2'] <= .005, (stage, tokenizer_comparisons[stage])
        for language in ['en', 'zh']:
            assert rows['baseline-flash'][language]['abi'] == rows['candidate-flash'][language]['abi'] == rows['candidate-eager'][language]['abi'], (quant, language, rows)
        # Independent reference metrics are always retained, including quant error.
        # F16 must pass the port's >99% cosine and scale-sensitive gate. Q4 is
        # judged against the identical shipped baseline plus decoded output.
        if quant == 'f16':
            for variant in ['baseline-flash', 'candidate-flash', 'candidate-eager']:
                for stage, value in json.loads((out / quant / variant / 'stages.json').read_text()).items():
                    assert value['cosine'] >= .99 and value['relative_l2'] <= .05, (variant, stage, value)
        receipt['results'][quant] = dict(passed=True, comparisons=comparisons, tokenizer_comparisons=tokenizer_comparisons, speech=rows)
        save()
        model.unlink()
    receipt['passed'] = True
    save()
    (out / 'summary.md').write_text('PR #492 CPU acceptance PASS: shared mel exact; five LM stages with norms/relative L2; default and non-flash CLI/session English/Chinese speech. See acceptance.json for quant scope. No GPU/CANN timing claim.\n')
    print('PR492_CPU_ACCEPTANCE_PASS', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2:])
    else:
        main()
