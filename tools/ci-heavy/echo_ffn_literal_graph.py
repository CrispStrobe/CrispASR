#!/usr/bin/env python3
"""Execute the verbatim vendored build_ffn body on tiny actual ggml tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def block(text, signature):
    begin = text.index(signature)
    start = text.index('{', begin)
    depth, end = 1, start + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[begin:end]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libraries', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--ggml-root', type=Path, default=ROOT / 'ggml')
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'examples/talk-llama/llama-graph.cpp'
    header = ROOT / 'examples/talk-llama/llama-graph.h'
    model = ROOT / 'examples/talk-llama/models/qwen35.cpp'
    body = block(source.read_text(), 'ggml_tensor* llm_graph_context::build_ffn(')
    qwen = block(model.read_text(), 'ggml_tensor* llm_build_qwen35::build_layer_ffn(')
    assert 'LLM_FFN_SILU, LLM_FFN_PAR, il' in qwen and 'cb(cur, "ffn_out", il)' in qwen
    declaration = body[:body.index('{')].replace('llm_graph_context::', '').strip() + ';'
    enums = '\n'.join(block(header.read_text(), 'enum ' + name) + ';' for name in ['llm_ffn_op_type', 'llm_ffn_gate_type'])
    # Binding-only shim: no adapters/bias/scales on the pinned Qwen3.5 FFN.
    # The complete actual build_ffn function, including fused branch and names,
    # is inserted unchanged below. No handwritten FFN math substitutes for it.
    prefix = r'''
#include "ggml.h"
#include <cstdio>
#include <array>
enum { LLM_ARCH_QWEN35, LLM_ARCH_STEP35, LLM_ARCH_GLM4, LLM_ARCH_GLM4_MOE, LLM_ARCH_JAIS2 };
'''
    shim = r'''
struct llm_graph_context {
    ggml_context* ctx0;
    int arch = LLM_ARCH_QWEN35;
    struct { std::array<float, 8> swiglu_clamp_shexp{}; } hparams;
    ggml_tensor* build_lora_mm(ggml_tensor* w, ggml_tensor* x) const { return ggml_mul_mat(ctx0, w, x); }
    void cb(ggml_tensor* tensor, const char* name, int layer) const {
        ggml_format_name(tensor, "%s-%d", name, layer);
    }
DECLARATION
};
'''.replace('DECLARATION', declaration)
    tail = r'''
static ggml_tensor* crispasr_literal_ffn(ggml_context* ctx, ggml_tensor* up, ggml_tensor* gate,
                                        ggml_tensor* down, ggml_tensor* x, int layer) {
    const llm_graph_context builder{ctx};
    auto* out = builder.build_ffn(x, up, nullptr, nullptr, gate, nullptr, nullptr, down,
                                  nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, layer);
    builder.cb(out, "ffn_out", layer);
    return out;
}
'''
    generated = out / 'literal-ffn.h'
    generated.write_text(prefix + enums + shim + body + tail)
    (out / 'actual-build-ffn.cpp').write_text(body)
    (out / 'actual-qwen35-call.cpp').write_text(qwen)
    for path in [source, header, model, ROOT / 'src/core/index_echo_ffn_capture.h',
                 ROOT / 'tests/test-index-echo-ffn-capture.cpp', Path(__file__).resolve()]:
        dest = out / 'method' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    libraries = []
    for stem in ['ggml', 'ggml-cpu', 'ggml-base']:
        matches = [args.libraries / name for name in [f'lib{stem}.so', f'lib{stem}.so.0', f'lib{stem}.dylib']]
        match = next((x for x in matches if x.is_file()), None)
        assert match is not None, (stem, args.libraries)
        libraries.append(match)
    compiler = shutil.which('c++')
    assert compiler
    ggml_source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.ggml_root, text=True).strip()
    assert ggml_source == '0a862076567afd8468822ccc512c97ac6a69b340'
    common = [compiler, '-std=c++17', '-O2', '-I' + str(ROOT / 'src'), '-I' + str(args.ggml_root / 'include'),
        '-DCRISPASR_INDEX_ECHO_LITERAL_FFN="' + str(generated) + '"',
        str(ROOT / 'tests/test-index-echo-ffn-capture.cpp'), *map(str, libraries), '-Wl,-rpath,' + str(args.libraries)]
    report = dict(scope='Verbatim actual build_ffn; actual ggml CPU; no LoRA/bias/scales or model weights',
        source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        ggml=ggml_source,
        builder_sha256=hashlib.sha256(body.encode()).hexdigest(), libraries={str(p): sha(p) for p in libraries}, runs=[])
    # The unchanged previous helper is a rejecting control on the REAL builder.
    old = subprocess.check_output(['git', 'show', 'd8183cf9:src/core/index_echo_ffn_capture.h'], cwd=ROOT)
    bad = out / 'old-helper' / 'core'
    bad.mkdir(parents=True)
    (bad / 'index_echo_ffn_capture.h').write_bytes(old)
    for label in ['old-helper', 'fixed-helper']:
        exe = out / label
        if label == 'old-helper':
            exe = out / 'old-helper-control'
        command = common + ['-o', str(exe)]
        if label == 'old-helper':
            command.insert(1, '-I' + str(bad.parent))
        with (out / (label + '-compile.log')).open('xb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, label + '-compile'
        with (out / (label + '.log')).open('xb') as log:
            result = subprocess.run([str(exe)], stdout=log, stderr=subprocess.STDOUT)
        report['runs'].append(dict(label=label, command=command, exit=result.returncode, executable_sha256=sha(exe)))
        (out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
        assert result.returncode == (1 if label == 'old-helper' else 0), label
        print(label, result.returncode, flush=True)
    print('ECHO_VERBATIM_BUILDER_PASS: old observer misses fused product; repaired observer captures all six', flush=True)


if __name__ == '__main__':
    main()
