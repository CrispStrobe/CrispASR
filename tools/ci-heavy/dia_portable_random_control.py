#!/usr/bin/env python3
"""Test the compiled portable sampler against the canonical original feedback.

No model weights or audio synthesis. Actual original generate/filter/delay
code consumes retained logits and feeds back compiled C++ candidate draws.
Actual Torch multinomial and its state remain the oracle, not forced inputs.
This candidate is not wired into the runtime and is not speech acceptance.
"""
import argparse
import atexit
import ast
from dataclasses import dataclass
import importlib.util
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
sys.path.insert(0, str(ROOT / 'tools'))
from dia_dac_export import digest, mapped_gguf
from reference_backends import dia_tts_reference as reference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--proof-folder', type=Path)
args = parser.parse_args()
folder = args.proof_folder
if folder is None:
    archive = SCRATCH / 'canonical-proof'
    subprocess.run(['gh', 'run', 'download', '38285566623', '-R', 'CrispStrobe/CrispASR', '-D', archive], check=True)
    folder = next(archive.glob('heavy-*'))
ref = folder / 'ref.gguf'
assert digest(ref) == '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542'
previous = json.loads((folder / 'original-reported.json').read_text())
assert previous['wer'] == 0 and previous['seed'] == 42 and previous['capacity'] == 3072
source = os.environ.get('DIA_SOURCE')
if not source:
    source = SCRATCH / 'official-dia'
    subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', reference.SOURCE_PIN], check=True)
source, hashes = reference.verify_source(source)
assert torch.__version__ == "2.6.0+cpu" and torch.backends.mkl.is_available(), "Pinned MKL CPU oracle required"
torch.set_num_threads(1)

# Compile the original functions/classes verbatim, avoiding unused DAC imports
# and neural weights. Only neural computation and PCM decoding are stand-ins.
namespace = dict(torch=torch, np=np, dataclass=dataclass, DiaConfig=object,
                 DecoderInferenceState=object, __name__=__name__)
exec(compile((source / 'dia/audio.py').read_text(), str(source / 'dia/audio.py'), 'exec'), namespace)
tree = ast.parse((source / 'dia/state.py').read_text())
output = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DecoderOutput')
exec(compile(ast.Module(body=[output], type_ignores=[]), str(source / 'dia/state.py'), 'exec'), namespace)
tree = ast.parse((source / 'dia/model.py').read_text())
sampler = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_sample_next_token')
dia = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Dia')
names = {'_prepare_audio_prompt', '_decoder_step', '_generate_output', 'generate'}
functions = [node for node in dia.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(functions) == len(names)
exec(compile(ast.Module(body=[sampler] + functions, type_ignores=[]), str(source / 'dia/model.py'), 'exec'), namespace)
spec = importlib.util.spec_from_file_location('canonical_dia_config', source / 'dia/config.py')
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = types.SimpleNamespace(data=config_module.DataConfig(text_length=128, audio_length=3072))

report = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              source_worktree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
              script_sha256=digest(Path(__file__)), original_source_pin=reference.SOURCE_PIN,
              original_source_files=hashes, reference_sha256=digest(ref), original_run=38285566623,
              torch_version=torch.__version__, mkl_available=torch.backends.mkl.is_available(),
              model_weights_loaded=False, production_defaults_changed=False, speech_acceptance=False,
              scope='Compiled portable RNG candidate with actual original filtering/generate/delay feedback; no neural, PCM or native free-generation acceptance.')
(OUT / 'torch-build-config.txt').write_text(torch.__config__.show())
header = ROOT / 'src/dia_exponential_rng.h'
cpp = OUT / 'portable-sampler-bridge.cpp'
cpp.write_text('#include <cstdio>\n#include <vector>\n#include "' + str(ROOT / 'src/dia_sampling.h') + '"\n' + r"""
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    FILE* output = fopen(argv[1], "wb");
    if (!output) return 3;
    std::mt19937 rng(42), unused(42);
    std::vector<float> probabilities(9 * 1028), logits(9 * 1028), noise(9 * 1028);
    while (fread(probabilities.data(), sizeof(float), probabilities.size(), stdin) == probabilities.size()) {
        if (fread(logits.data(), sizeof(float), logits.size(), stdin) != logits.size()) return 7;
        dia_rng::exponential_batch(noise.data(), noise.size(), rng);
        if (fwrite(noise.data(), sizeof(float), noise.size(), output) != noise.size()) return 4;
        uint32_t tokens[18];
        for (int row = 0; row < 9; ++row) {
            const auto offset = row * 1028;
            tokens[row] = dia_rng::exponential_race(probabilities.data() + offset, noise.data() + offset, 1028);
            tokens[9 + row] = dia_sample_token(logits.data() + offset, 1028, 1.2f, .95f, 45, unused, noise.data() + offset);
        }
        if (fwrite(tokens, sizeof(uint32_t), 18, stdout) != 18 || fflush(stdout)) return 5;
    }
    int error = ferror(stdin);
    return fclose(output) || error ? 6 : 0;
}
""")
binary = SCRATCH / 'portable-sampler'
with (OUT / 'compile.log').open('w') as log:
    subprocess.run(['c++', '-std=c++17', '-O2', cpp, '-o', binary], stdout=log, stderr=subprocess.STDOUT, check=True)
portable_noise = OUT / 'portable-exponential-noise.f32'
bridge = subprocess.Popen([binary, portable_noise], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def close_bridge():
    bridge.stdin.close()
    try:
        bridge.wait(timeout=5)
    except subprocess.TimeoutExpired:
        bridge.kill()
        bridge.wait()
    bridge.stdout.close()

atexit.register(close_bridge)
report.update(candidate_header_sha256=digest(header), candidate_bridge_sha256=digest(cpp),
              candidate_binary_sha256=digest(binary), candidate_filter_header_sha256=digest(ROOT / 'src/dia_sampling.h'),
              production_calls_changed=False)

draws, split_draws, input_rows = [], [], []
actual_multinomial = torch.multinomial
actual_sampler = namespace['_sample_next_token']
current_logits = None

def capture_sampler(logits, **kwargs):
    global current_logits
    current_logits = logits.numpy().astype('<f4')
    return actual_sampler(logits, **kwargs)

namespace['_sample_next_token'] = capture_sampler
with mapped_gguf(ref) as reader:
    tensors = {tensor.name: tensor.data for tensor in reader.tensors}
    inputs = tensors['decoder_input_tokens']
    logits = tensors['decoder_step_logits']
    canonical_codes = tensors['decoder_codes_rev']
    assert inputs.shape == (1226, 9) and logits.shape == (1226, 2, 9, 1028)
    assert canonical_codes.shape == (1210, 9), 'Reference code grid must use captured time/codebook order'
    fake = types.SimpleNamespace(config=config, device=torch.device('cpu'))
    fake.model = types.SimpleNamespace(eval=lambda: None,
        decoder=types.SimpleNamespace(decode_step=None))
    for name in names:
        setattr(fake, name, types.MethodType(namespace[name], fake))

    def prepare(*_):
        prefill, step = fake._prepare_audio_prompt(None)
        result = namespace['DecoderOutput'].new(config, fake.device)
        result.prefill(prefill, step)
        return types.SimpleNamespace(prepare_step=lambda *_: None), result

    def decode_step(tokens, _state):
        index = len(input_rows)
        assert index < len(inputs), 'Original replay exceeded retained history'
        row = tokens[0, 0].numpy().copy()
        input_rows.append(row)
        assert np.array_equal(tokens[0], tokens[1]), 'CFG input batches differ'
        if not np.array_equal(row, inputs[index]):
            report['first_input_divergence'] = dict(step=index, expected=inputs[index].tolist(), actual=row.tolist())
            raise AssertionError(f'Canonical feedback mismatch at step {index}')
        return torch.from_numpy(np.array(logits[index], copy=True))[:, None]

    fake._prepare_generation = prepare
    fake.model.decoder.decode_step = decode_step
    # The actual output function still does delay reversal and invalid-code
    # sanitization. This identity codec returns codes, explicitly not PCM.
    fake.dac_model = types.SimpleNamespace(
        quantizer=types.SimpleNamespace(from_codes=lambda codes: (codes,)),
        decode=lambda codes: codes)
    noise_file = OUT / 'actual-torch-exponential-noise.f32'
    state_file = OUT / 'actual-torch-rng-states.u8'
    with noise_file.open('wb') as noise, state_file.open('wb') as states:
        def observe(probability, num_samples):
            assert num_samples == 1 and probability.shape == (9, 1028)
            before = torch.get_rng_state()
            actual = actual_multinomial(probability, num_samples=1)
            bridge.stdin.write(probability.numpy().astype('<f4').tobytes())
            bridge.stdin.write(current_logits.tobytes())
            bridge.stdin.flush()
            data = bridge.stdout.read(18 * 4)
            assert len(data) == 18 * 4, 'Compiled candidate stopped before returning both nine-draw arms'
            both = np.frombuffer(data, dtype='<u4').astype(np.int64).reshape(2, 9, 1)
            candidate = torch.from_numpy(both[0].copy())
            native_filter = torch.from_numpy(both[1].copy())
            if not torch.equal(native_filter, actual):
                report['first_native_filter_draw_divergence'] = dict(step=len(draws), actual=actual.reshape(9).tolist(), candidate=native_filter.reshape(9).tolist())
                raise AssertionError('Actual native probability/filter plus candidate RNG differs from Torch')
            if not torch.equal(candidate, actual):
                report['first_candidate_draw_divergence'] = dict(step=len(draws), actual=actual.reshape(9).tolist(), candidate=candidate.reshape(9).tolist())
                raise AssertionError('Compiled candidate differs from actual Torch draw')
            after = torch.get_rng_state()
            torch.set_rng_state(before)
            exponential = torch.empty_like(probability).exponential_(1)
            raced = (probability / exponential).argmax(dim=-1, keepdim=True)
            assert torch.equal(raced, actual), 'Source-derived exponential race differs from actual multinomial'
            assert torch.equal(torch.get_rng_state(), after), 'Exponential race consumes a different random stream'
            assert bool(torch.isfinite(exponential).all()) and bool((exponential > 0).all())
            # A row-by-row sampler is a different RNG contract from the
            # original one tensor draw. Keep this fixed negative control,
            # restoring the actual batch state before feedback continues.
            torch.set_rng_state(before)
            split = torch.cat([actual_multinomial(row[None], num_samples=1) for row in probability])
            split_draws.append(split.numpy().reshape(9).copy())
            torch.set_rng_state(after)
            noise.write(exponential.numpy().astype('<f4').tobytes())
            states.write(before.numpy().tobytes())
            draws.append(actual.numpy().reshape(9).copy())
            return candidate

        try:
            torch.manual_seed(42)
            with mock.patch.object(torch, 'multinomial', observe):
                codes = fake.generate('[S1] We are committed to making it secure.', max_tokens=3072,
                    cfg_scale=3., temperature=1.2, top_p=.95, cfg_filter_top_k=45, use_torch_compile=False)
            assert len(input_rows) == len(inputs), 'Canonical replay terminated at a different step'
            assert codes.shape == (9, 1210), 'Identity codec must retain original codebook/time order'
            assert np.array_equal(codes.T, canonical_codes), 'Canonical final codec grid differs'
            report['compiled_candidate_draws_exact'] = True
            report['native_filter_and_candidate_draws_exact'] = True
            report.update(canonical_input_transitions_exact=True, canonical_final_codes_exact=True,
                          exponential_race_samples_and_rng_state_exact=True, steps=len(draws),
                          sampled_rows=len(draws) * 9, rng_state_bytes=torch.get_rng_state().numel(),
                          codec_frames=codes.shape[-1], component_replay_pass=True)
            split_array = np.asarray(split_draws)
            report['split_row_negative'] = dict(sample_mismatches=int(np.count_nonzero(split_array != np.asarray(draws))),
                scope='Fixed row-by-row Torch sampling from each same pre-batch RNG state; diagnostic only, never used for feedback.')
            assert report['split_row_negative']['sample_mismatches'] > 0, 'Row-split negative failed to distinguish batch RNG'
            np.save(OUT / 'replayed-code-grid.npy', codes.T)
        except Exception as error:
            report.update(component_replay_pass=False, failed_after_steps=len(input_rows), error=str(error))
            raise
        finally:
            np.save(OUT / 'pre-override-samples.npy', np.asarray(draws))
            np.save(OUT / 'split-row-negative-samples.npy', np.asarray(split_draws))
            np.save(OUT / 'replayed-decoder-inputs.npy', np.asarray(input_rows))
            (OUT / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
bridge.stdin.close()
assert bridge.wait(timeout=30) == 0, 'Compiled candidate process failed'
bridge.stdout.close()
atexit.unregister(close_bridge)
actual_noise = np.fromfile(noise_file, dtype='<f4')
candidate_noise = np.fromfile(portable_noise, dtype='<f4')
assert actual_noise.shape == candidate_noise.shape == (1226 * 9 * 1028,)
assert bool(np.isfinite(candidate_noise).all()) and bool((candidate_noise > 0).all())
report['noise_comparison'] = dict(values=actual_noise.size, bit_differences=int(np.count_nonzero(actual_noise != candidate_noise)),
    maximum_absolute_error=float(np.max(np.abs(actual_noise-candidate_noise))),
    maximum_relative_error=float(np.max(np.abs(actual_noise-candidate_noise)/actual_noise)),
    norm_relative_error=float(abs(np.linalg.norm(candidate_noise.astype('f8')) / np.linalg.norm(actual_noise.astype('f8')) - 1)),
    cosine=float(np.dot(actual_noise.astype('f8'),candidate_noise.astype('f8')) / np.linalg.norm(actual_noise.astype('f8')) / np.linalg.norm(candidate_noise.astype('f8'))))
for path in (noise_file, state_file, portable_noise):
    report[path.name] = dict(bytes=path.stat().st_size, sha256=digest(path))
(OUT / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
