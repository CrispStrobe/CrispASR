#!/usr/bin/env python3
"""Replay real aligned Dia logits through actual original/native probability code.

No model weights, neural reconstruction or RNG change. The original Torch
multinomial is intercepted only to observe its final probabilities; native
distribution observation is mechanically inserted into the actual header.
This is component evidence, never speech acceptance or a replacement reference.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--proof-folder', type=Path, help='Retained heavy-38299826144 folder; default downloads it')
args = parser.parse_args()
sys.path.insert(0, str(ROOT / 'tools'))
from dia_dac_export import digest
from reference_backends import dia_tts_reference as reference

torch.set_num_threads(1)
folder = args.proof_folder
if folder is None:
    archive = SCRATCH / 'sampler-proof'
    subprocess.run(['gh', 'run', 'download', '38299826144', '-R', 'CrispStrobe/CrispASR', '-D', archive], check=True)
    folder = next(archive.glob('heavy-*'))
previous = json.loads((folder / 'receipt.json').read_text())
assert previous['source'] == '1d8d5bbc2351b74721abb11d0a39d2b8f86f2441'
assert previous['canonical-original']['wer'] == 0 and previous['original-native-sampler']['wer'] == 1
prefix = previous['paired_generation']['aligned_input_prefix_steps']
assert prefix == 37 and previous['paired_generation']['first_pre_override_sample_divergence']['step'] == 36
source = os.environ.get('DIA_SOURCE')
if not source:
    source = SCRATCH / 'official-dia'
    subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', reference.SOURCE_PIN], check=True)
source, _ = reference.verify_source(source)
original_text = (source / 'dia/model.py').read_text()
tree = ast.parse(original_text)
sampler = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_sample_next_token')
dia = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Dia')
step = next(node for node in dia.body if isinstance(node, ast.FunctionDef) and node.name == '_decoder_step')
namespace = dict(torch=torch, DecoderInferenceState=object)
exec(compile(ast.Module(body=[sampler, step], type_ignores=[]), str(source / 'dia/model.py'), 'exec'), namespace)
official_sampler = namespace['_sample_next_token']
header = ROOT / 'src/dia_sampling.h'
assert digest(header) == digest(folder / 'dia_sampling.h'), 'Production sampler changed since retained run'
code = header.read_text()
assert code.count('    return dist(rng);') == 1
instrumented = code.replace('    return dist(rng);', '''    const auto observed = dist.probabilities();
    if (fwrite(observed.data(), sizeof(double), observed.size(), stdout) != observed.size()) std::abort();
    return dist(rng);''')
cpp = OUT / 'actual-sampler-probability-observer.cpp'
cpp.write_text('#include <cstdio>\n#include <cstdlib>\n' + instrumented.replace('#pragma once', '') + '''
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    FILE* input = fopen(argv[1], "rb"); FILE* output = fopen(argv[2], "wb");
    if (!input || !output) return 3;
    std::mt19937 rng(42); std::vector<float> row(1028);
    while (fread(row.data(), sizeof(float), row.size(), input) == row.size()) {
        uint32_t token = dia_sample_token(row.data(), 1028, 1.2f, .95f, 45, rng);
        if (fwrite(&token, sizeof(token), 1, output) != 1) return 4;
    }
    if (ferror(input)) return 5;
    fclose(input); fclose(output); return 0;
}
''')
binary = SCRATCH / 'probability-observer'
with (OUT / 'compile.log').open('w') as log:
    subprocess.run(['c++', '-std=c++17', '-O2', cpp, '-o', binary], stdout=log, stderr=subprocess.STDOUT, check=True)
report = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              probe_script_sha256=digest(Path(__file__)),
              source_worktree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
              original_source_pin=reference.SOURCE_PIN, original_source_sha256=hashlib.sha256(original_text.encode()).hexdigest(),
              native_header_sha256=digest(header), observer_sha256=digest(cpp), torch_version=torch.__version__,
              actual_original_logits_run=38299826144, prefix_steps=prefix, probability_arms=[],
              production_defaults_changed=False, speech_acceptance=False, model_weights_loaded=False)
for label, basename, sample_file in [('original-control', 'original-control', 'original-control-pre-override-samples.npy'),
                                    ('native', 'native', 'native-pre-override-samples.npy')]:
    path = folder / (basename + '-logits.f32')
    steps, remainder = divmod(path.stat().st_size, 2 * 9 * 1028 * 4)
    assert remainder == 0 and steps >= prefix
    logits = np.memmap(path, mode='r', dtype='<f4', shape=(steps, 2, 9, 1028))
    inputs, probabilities = [], []
    def observe_multinomial(probability, num_samples):
        assert num_samples == 1
        probabilities.append(probability.detach().numpy().copy())
        return probability.argmax(dim=-1, keepdim=True)
    def observe_sampler(logits, **kwargs):
        inputs.append(logits.detach().numpy().copy())
        return official_sampler(logits, **kwargs)
    namespace['_sample_next_token'] = observe_sampler
    with mock.patch.object(torch, 'multinomial', observe_multinomial):
        for index in range(prefix):
            raw = torch.from_numpy(np.array(logits[index], copy=True))[:, None]
            fake = types.SimpleNamespace(config=types.SimpleNamespace(data=types.SimpleNamespace(audio_eos_value=1024)),
                model=types.SimpleNamespace(decoder=types.SimpleNamespace(decode_step=lambda *_: raw)))
            namespace['_decoder_step'](fake, None, None, 3., 1.2, .95, 45)
    logits._mmap.close()
    inputs = np.stack(inputs).astype('<f4')
    original = np.stack(probabilities)
    assert inputs.shape == original.shape == (prefix, 9, 1028)
    input_file = OUT / (label + '-actual-original-cfg-inputs.f32')
    inputs.tofile(input_file)
    samples = OUT / (label + '-replayed-samples.u32')
    with (OUT / (label + '-native-probabilities.f64')).open('wb') as output:
        subprocess.run([binary, input_file, samples], stdout=output, check=True)
    native = np.fromfile(OUT / (label + '-native-probabilities.f64'), dtype='<f8').reshape(prefix, 9, 1028)
    replayed = np.fromfile(samples, dtype='<u4').reshape(prefix, 9)
    saved = np.load(folder / sample_file)[:prefix]
    exact = bool(np.array_equal(replayed, saved))
    np.save(OUT / (label + '-original-probabilities.npy'), original)
    support = np.count_nonzero((original > 0) != (native > 0), axis=-1)
    absolute = np.max(np.abs(original - native), axis=-1)
    variation = .5 * np.sum(np.abs(original - native), axis=-1)
    np.save(OUT / (label + '-support-difference.npy'), support)
    np.save(OUT / (label + '-maximum-probability-error.npy'), absolute)
    np.save(OUT / (label + '-total-variation.npy'), variation)
    report['probability_arms'].append(dict(label=label, sampled_rows=prefix * 9, support_mismatching_rows=int(np.count_nonzero(support)),
        maximum_absolute_probability_error=float(absolute.max()), maximum_total_variation=float(variation.max()),
        actual_native_samples_reproduced=exact, first_divergence_row=dict(step=36, codebook=8,
            support_difference=int(support[36, 8]), maximum_absolute_error=float(absolute[36, 8]), total_variation=float(variation[36, 8]))))
    (OUT / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    assert exact, 'Actual native sample replay differs from retained run'
report['scope'] = 'Actual probability/filter component replay on the complete identical input prefix; no speech acceptance, RNG compatibility or model-quality waiver.'
(OUT / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
