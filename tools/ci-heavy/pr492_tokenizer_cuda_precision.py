#!/usr/bin/env python3
"""Original-file MiMo Q4 CUDA precision experiment; no production changes.

Call patch_cuda() before the CUDA build. The temporary dispatch bypasses both
MMQ and MMVQ only when explicitly enabled. Actual cuBLAS calls are traced;
F32 computation is forced separately. Full PR acceptance still requires exact
RVQ codes, independent original-checkpoint and decoded-output validation.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from pr492_acceptance import ROOT, TOK_STAGES, digest, metrics, run

FLAG = 'CRISPASR_DIAG_CUDA_Q4_CUBLAS'
MARKER = 'MIMO_DIAG_CUDA_CUBLAS_F32'


def patch_cuda(repo, out):
    source = Path(repo) / 'ggml/src/ggml-cuda/ggml-cuda.cu'
    original = source.read_text()
    before = '''static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS
'''
    after = before + '''
    // Diagnostic only: retain Q4 storage but avoid Q8 activation rounding.
    if (getenv("CRISPASR_DIAG_CUDA_Q4_CUBLAS") && ggml_is_quantized(src0->type)) {
        static thread_local unsigned trace_count = 0;
        if (trace_count++ < 16) {
            fprintf(stderr, "MIMO_DIAG_CUDA_CUBLAS_F32 device=%d weight=%s type=%s k=%lld m=%lld n=%lld\\n",
                    ctx.device, src0->name, ggml_type_name(src0->type),
                    (long long) src0->ne[0], (long long) src0->ne[1], (long long) src1->ne[1]);
        }
        // Explicit template selection avoids the normal quantized/F16 policy
        // and does not depend on a precision hint ignored by another kernel.
        ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
        return;
    }
'''
    assert original.count(before) == 1
    source.write_text(original.replace(before, after))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cuda-dispatch.patch').write_text(subprocess.check_output(
        ['git', 'diff', '--', 'src/ggml-cuda/ggml-cuda.cu'], cwd=Path(repo) / 'ggml', text=True))
    receipt = dict(original_sha256=__import__('hashlib').sha256(original.encode()).hexdigest(),
                   patched_sha256=digest(source), patch_sha256=digest(out / 'cuda-dispatch.patch'),
                   flag=FLAG, trace_marker=MARKER, scope=__doc__)
    (out / 'cuda-patch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    out = Path(os.environ['HEAVY_OUT'])
    library = Path(os.environ['MIMO_TOKENIZER_DIAG_LIB']).resolve(strict=True)
    assert os.environ.get('MIMO_TOKENIZER_DIAG_GPU') == '1'
    os.environ.update({FLAG: '1', 'NV_TF32_OVERRIDE': '0'})
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   scope=__doc__, continuous_passed=False, exact_rvq_passed=False,
                   tokenizer_acceptance_passed=False, full_pr_acceptance=False)
    def save():
        (out / 'cuda-precision.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    run([sys.executable, ROOT / 'tools/ci-heavy/pr492_tokenizer_diagnose.py'], out / 'diagnose.log')
    study = json.loads((out / 'tokenizer-diagnosis.json').read_text())
    receipt['codec'] = study['codec']
    receipt['q4_attention_ab'] = study['attention_ab']['q4']
    receipt['q4_independent_same_weights'] = {
        s: {a: m for a, m in arms.items() if a.startswith('q4-')}
        for s, arms in study['independent_same_weights'].items()}
    receipt['timings'] = {}
    receipt['dispatch_traces'] = {}
    for arm in ['q4-flash', 'q4-eager']:
        lines = (out / (arm + '.log')).read_text().splitlines()
        traces = [line for line in lines if MARKER in line]
        assert traces, 'No actual quantized cuBLAS dispatch: ' + arm
        receipt['dispatch_traces'][arm] = traces
        receipt['timings'][arm] = json.loads((out / arm / 'native-execution.json').read_text())
    for arm in ['promoted-flash', 'promoted-eager']:
        assert MARKER not in (out / (arm + '.log')).read_text(), arm
    save()
    codec = Path(os.environ['HEAVY_SCRATCH']) / 'pr492-tokenizer/models' / study['codec']['filename']
    audio = Path(os.environ['HEAVY_SCRATCH']) / 'pr492-tokenizer/en.wav'
    receipt['codec_bytes'] = codec.stat().st_size
    # Same binary/GPU/file: disable only the temporary dispatch.
    os.environ.pop(FLAG)
    for flash in [1, 0]:
        arm = 'default-q4-' + ('flash' if flash else 'eager')
        run([sys.executable, ROOT / 'tools/ci-heavy/pr492_tokenizer_diagnose.py', '--native',
             library, codec, audio, out / arm, str(flash)], out / (arm + '.log'))
        assert MARKER not in (out / (arm + '.log')).read_text()
        assert 'mimo_tokenizer: RVQ backend=CUDA' in (out / (arm + '.log')).read_text()
        receipt['timings'][arm] = json.loads((out / arm / 'native-execution.json').read_text())
    def compare(a, b):
        result = {}
        for stage in TOK_STAGES:
            x, y = [np.load(out / arm / (stage + '.npy')) for arm in [a, b]]
            result[stage] = (dict(exact=np.array_equal(x, y), match_fraction=float(np.mean(x == y)),
                                  mismatches=int(np.count_nonzero(x != y)))
                             if stage == 'tok_codes' else metrics(x, y))
        return result
    receipt['default_attention_ab'] = compare('default-q4-eager', 'default-q4-flash')
    receipt['dispatch_ab'] = {mode: compare('q4-' + mode, 'default-q4-' + mode)
                              for mode in ['flash', 'eager']}
    # Shape controls exercise short sequences, including matrix-vector routing.
    # These have attention/code A/B evidence, not an independent encoder oracle.
    import soundfile as sf
    pcm, sr = sf.read(audio, dtype='float32')
    receipt['short_shape_controls'] = {}
    for samples in [4000, 38400]:
        short = out / f'short-{samples}.wav'
        sf.write(short, pcm[:samples], sr, subtype='FLOAT')
        arms = []
        for enabled in [False, True]:
            if enabled:
                os.environ[FLAG] = '1'
            else:
                os.environ.pop(FLAG, None)
            for flash in [1, 0]:
                arm = f'short-{samples}-' + ('cublas' if enabled else 'default') + ('-flash' if flash else '-eager')
                run([sys.executable, ROOT / 'tools/ci-heavy/pr492_tokenizer_diagnose.py', '--native',
                     library, codec, short, out / arm, str(flash)], out / (arm + '.log'))
                log = (out / (arm + '.log')).read_text()
                assert (MARKER in log) == enabled, arm
                arms.append(arm)
        receipt['short_shape_controls'][str(samples)] = {
            'default_attention_ab': compare(arms[1], arms[0]),
            'cublas_attention_ab': compare(arms[3], arms[2]),
            'timings': {a: json.loads((out / a / 'native-execution.json').read_text()) for a in arms}}
        save()
    def accepted(m):
        return m['cosine'] >= .9999 and m['relative_l2'] <= .005
    receipt['continuous_passed'] = bool(study['passed'] and
        all(accepted(m) for s, m in receipt['q4_attention_ab'].items() if s != 'tok_codes') and
        all(accepted(m) for arms in receipt['q4_independent_same_weights'].values() for m in arms.values()))
    receipt['exact_rvq_passed'] = bool(receipt['q4_attention_ab']['tok_codes']['exact'])
    receipt['tokenizer_acceptance_passed'] = receipt['continuous_passed'] and receipt['exact_rvq_passed']
    save()
    print(json.dumps({k: receipt[k] for k in ['continuous_passed', 'exact_rvq_passed',
                                             'tokenizer_acceptance_passed', 'full_pr_acceptance']}), flush=True)
    assert receipt['tokenizer_acceptance_passed'], 'Unchanged CUDA tokenizer numerical/exact-code gates failed; inspect cuda-precision.json'


if __name__ == '__main__':
    main()
