#!/usr/bin/env python3
"""Test original Q4 MiMo weights with a diagnostic CPU BLAS scheduler.

The production runtime is untouched. This script records and applies a temporary
scheduler patch, builds once, compares CPU/BLAS paths on the same runner, and
restores source in finally. The official same-weight encoder and exact RVQ gates
remain mandatory. No lost checkpoint precision is recovered.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from pr492_acceptance import ROOT, TOK_STAGES, digest, metrics, run


def main():
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'pr492-tokenizer-blas'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    os.environ.pop('MIMO_TOKENIZER_DIAG_GPU', None)
    os.environ.pop('CRISPASR_DIAG_MIMO_BLAS', None)
    receipt = dict(passed=False, scope=__doc__,
                   source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    def save():
        (out / 'tokenizer-blas.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    run(['bash', ROOT / 'tools/ci-apt.sh', 'update'], out / 'apt-update.log')
    run(['bash', ROOT / 'tools/ci-apt.sh', 'install', '-y', 'ffmpeg', 'libopenblas-dev'], out / 'apt-install.log')
    source = ROOT / 'src/mimo_tokenizer.cpp'
    original = source.read_text()
    patched = original
    replacements = [
        ('ggml_backend_t backend_cpu = nullptr;',
         'ggml_backend_t backend_blas = nullptr; // diagnostic CPU precision experiment\n    ggml_backend_t backend_cpu = nullptr;'),
        ('        ggml_backend_t backends[2];\n        backends[n_be++] = ctx->backend;',
         '''        ggml_backend_t backends[3];
        if (std::getenv("CRISPASR_DIAG_MIMO_BLAS") && ctx->backend == ctx->backend_cpu) {
            ctx->backend_blas = ggml_backend_init_by_name("BLAS", nullptr);
            if (!ctx->backend_blas) {
                fprintf(stderr, "MIMO_DIAG_BLAS_UNAVAILABLE\\n");
                mimo_tokenizer_free(ctx);
                return nullptr;
            }
            core_cpu_backend::set_n_threads(ctx->backend_blas, ctx->n_threads);
            fprintf(stderr, "MIMO_DIAG_BLAS_ACTIVE backend=%s\\n", ggml_backend_name(ctx->backend_blas));
            backends[n_be++] = ctx->backend_blas;
        }
        backends[n_be++] = ctx->backend;'''),
        ('    if (ctx->backend_cpu)\n        ggml_backend_free(ctx->backend_cpu);',
         '    if (ctx->backend_blas)\n        ggml_backend_free(ctx->backend_blas);\n    if (ctx->backend_cpu)\n        ggml_backend_free(ctx->backend_cpu);'),
        ('    ctx->n_threads = n_threads;\n    if (ctx->backend_cpu)',
         '    ctx->n_threads = n_threads;\n    if (ctx->backend_blas)\n        core_cpu_backend::set_n_threads(ctx->backend_blas, n_threads);\n    if (ctx->backend_cpu)'),
    ]
    for before, after in replacements:
        assert patched.count(before) == 1, before
        patched = patched.replace(before, after)
    (out / 'mimo_tokenizer.cpp.diagnostic').write_text(patched)
    receipt['diagnostic_cpp_sha256'] = digest(out / 'mimo_tokenizer.cpp.diagnostic')
    receipt['temporary_replacements'] = replacements
    save()
    try:
        source.write_text(patched)
        # Link only the actual tokenizer object and its ggml dependencies.
        # Unrelated ASR backends and Rust/C2PA are outside this experiment.
        wrapper = scratch / 'wrapper'
        wrapper.mkdir(exist_ok=True)
        (wrapper / 'entry.cpp').write_text('#include "mimo_tokenizer.h"\nextern "C" int mimo_probe_threads() { return mimo_tokenizer_context_default_params().n_threads; }\n')
        (wrapper / 'add-probe.cmake').write_text(f'function(mimo_diag_add_probe)\n  add_library(mimo_tokenizer_probe SHARED "{wrapper / "entry.cpp"}")\n  target_link_libraries(mimo_tokenizer_probe PRIVATE mimo_tokenizer)\nendfunction()\ncmake_language(DEFER CALL mimo_diag_add_probe)\n')
        build = scratch / 'build'
        run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=ON', '-DGGML_BLAS_VENDOR=OpenBLAS',
             '-DCRISPASR_MEL_BLAS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=OFF',
             '-DCRISPASR_BUILD_EXAMPLES=OFF',
             f"-DCMAKE_PROJECT_crispasr_INCLUDE={wrapper / 'add-probe.cmake'}"], out / 'configure.log')
        run(['cmake', '--build', build, '--target', 'mimo_tokenizer_probe', '-j4'], out / 'build.log')
        library = build / 'libmimo_tokenizer_probe.so'
        assert library.is_file()
        (out / 'CMakeCache.txt').write_bytes((build / 'CMakeCache.txt').read_bytes())
        os.environ.update(MIMO_TOKENIZER_DIAG_LIB=str(library), CRISPASR_DIAG_MIMO_BLAS='1')
        # The complete diagnostic generates an official reference on the native
        # BLAS arm's own conv2 input, rather than borrowing a changed input.
        run([sys.executable, ROOT / 'tools/ci-heavy/pr492_tokenizer_diagnose.py'], out / 'diagnose.log')
        study = json.loads((out / 'tokenizer-diagnosis.json').read_text())
        receipt['codec'] = study['codec']
        receipt['q4_attention_ab'] = study['attention_ab']['q4']
        receipt['q4_independent_same_weights'] = {
            stage: {arm: m for arm, m in arms.items() if arm.startswith('q4-')}
            for stage, arms in study['independent_same_weights'].items()}
        receipt['timings'] = {}
        for arm in ['q4-flash', 'q4-eager']:
            assert 'MIMO_DIAG_BLAS_ACTIVE backend=BLAS' in (out / (arm + '.log')).read_text(), arm
            receipt['timings']['blas-' + arm] = json.loads((out / arm / 'native-execution.json').read_text())
        save()
        # Same binary, same original file, same host/thread count: disabling only
        # the experimental backend is the negative/control arm.
        os.environ.pop('CRISPASR_DIAG_MIMO_BLAS')
        codec = Path(os.environ['HEAVY_SCRATCH']) / 'pr492-tokenizer/models' / study['codec']['filename']
        audio = Path(os.environ['HEAVY_SCRATCH']) / 'pr492-tokenizer/en.wav'
        for flash in [1, 0]:
            arm = 'cpu-q4-' + ('flash' if flash else 'eager')
            run([sys.executable, ROOT / 'tools/ci-heavy/pr492_tokenizer_diagnose.py', '--native',
                 library, codec, audio, out / arm, str(flash)], out / (arm + '.log'))
            assert 'MIMO_DIAG_BLAS_ACTIVE' not in (out / (arm + '.log')).read_text()
            receipt['timings'][arm] = json.loads((out / arm / 'native-execution.json').read_text())
        def data(arm, stage):
            return np.load(out / arm / (stage + '.npy'))
        receipt['cpu_attention_ab'] = {
            stage: (dict(exact=np.array_equal(data('cpu-q4-eager', stage), data('cpu-q4-flash', stage)),
                         match_fraction=float(np.mean(data('cpu-q4-eager', stage) == data('cpu-q4-flash', stage))))
                    if stage == 'tok_codes' else metrics(data('cpu-q4-eager', stage), data('cpu-q4-flash', stage)))
            for stage in TOK_STAGES}
        def accepted(m):
            return m['cosine'] >= .9999 and m['relative_l2'] <= .005
        receipt['passed'] = (study['passed'] and
            all(accepted(m) for s, m in receipt['q4_attention_ab'].items() if s != 'tok_codes') and
            receipt['q4_attention_ab']['tok_codes']['exact'] and
            all(accepted(m) for arms in receipt['q4_independent_same_weights'].values() for m in arms.values()))
        save()
        assert receipt['passed'], 'Original Q4 CPU BLAS path failed unchanged numerical/RVQ gates'
    finally:
        source.write_text(original)
        receipt['source_restored'] = source.read_text() == original
        save()
    (out / 'summary.md').write_text('MiMo original Q4 CPU BLAS numerical diagnostic PASS; inspect paired timings. No production runtime or model changes.\n')


if __name__ == '__main__':
    main()
