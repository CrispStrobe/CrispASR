#!/usr/bin/env python3
"""Isolate DAC weight precision using preserved actual Dia reference codes.

Original DAC modules decode both original and published GGUF weights. Native
diff checks each variant against the canonical F32 original and a separately
labelled precision-matched original control. No production default/threshold
changes; narrow F32 exports are diagnostics, not accepted speech models.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-run', default='38278884937')
parser.add_argument('--child', action='store_true')
parser.add_argument('--reference', type=Path)
parser.add_argument('--codec', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(ROOT / 'tools'))
from reference_backends import dia_tts_reference as reference


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


from dia_dac_export import parameter


if args.child:
    import numpy as np
    import torch
    import dac
    import importlib.metadata
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
    import dump_reference
    torch.set_num_threads(4)
    reader = GGUFReader(str(args.reference))
    tensors = {tensor.name: np.asarray(tensor.data).copy() for tensor in reader.tensors}
    def metadata_value(field):
        value = reader.fields['crispasr.ref.' + field]
        return bytes(value.parts[value.data[0]]).decode()
    assert metadata_value('dia_source_pin') == reference.SOURCE_PIN
    assert metadata_value('dia_model_revision') == reference.MODEL_PIN
    assert importlib.metadata.version('descript-audio-codec') == metadata_value('dia_dac_package') == '1.0.0'
    checkpoint = Path(dac.utils.download(model_type='44khz', model_bitrate='8kbps', tag='0.0.1'))
    assert digest(checkpoint) == metadata_value('dia_dac_sha256')
    codec = dac.DAC.load(str(checkpoint)).cpu().eval()
    codes = tensors['decoder_codes_rev']
    assert codes.ndim == 2 and codes.shape[1] == 9
    assert np.isfinite(codes).all() and (codes == np.floor(codes)).all()
    assert codes.min() >= 0 and codes.max() < 1024
    codes = torch.from_numpy(codes.T[None].copy()).long()
    def decode():
        with torch.inference_mode():
            latent = codec.quantizer.from_codes(codes)[0]
            return codec.decode(latent).cpu().float().numpy().reshape(-1).copy()
    original = decode()  # Also materializes every actual weight_norm pre-hook.
    canonical = tensors['decoded_audio'].reshape(-1)
    assert original.shape == canonical.shape and np.isfinite(original).all()
    # Same original checkpoint/modules/input must reproduce the preserved PCM
    # before any weight-precision interpretation. This is tighter than native.
    np.testing.assert_allclose(original, canonical, rtol=0, atol=1e-7)
    dot = float(np.dot(original.astype(np.float64), canonical.astype(np.float64)))
    norm = float(np.linalg.norm(original.astype(np.float64)) / np.linalg.norm(canonical.astype(np.float64)))
    cosine = dot / float(np.linalg.norm(original.astype(np.float64)) * np.linalg.norm(canonical.astype(np.float64)))
    assert cosine >= .999999 and abs(norm - 1) < .0001, (cosine, norm)
    for module in codec.modules():
        if hasattr(module, 'weight_g'):
            torch.nn.utils.remove_weight_norm(module)
    np.testing.assert_array_equal(decode(), original)
    original_trace = reference.capture_codec(codec, codes)
    np.testing.assert_array_equal(original_trace['decoded_audio'], original)
    metadata = json.loads((args.reference.parent / 'reference.json').read_text())
    trace_metadata = dict(metadata, dia_dac_stage_trace='rvq+decoder-v1')
    # Keep the preserved canonical PCM bytes. Newly observed intermediate
    # modules supplement that archive; they never replace its ground truth.
    extended = dict(tensors, **{name: value for name, value in original_trace.items() if name != 'decoded_audio'})
    dump_reference.write_gguf_archive(extended, trace_metadata, OUT / 'canonical-codec-ref.gguf')
    for name, value in original_trace.items():
        np.save(OUT / f'original-{name}.npy', value)
    dump_reference.write_gguf_archive({name: value for name, value in extended.items() if name != 'dac_block_0'},
                                     trace_metadata, OUT / 'missing-codec-stage-ref.gguf')
    scaled = dict(extended, dac_latent=original_trace['dac_latent'] * 2)
    dump_reference.write_gguf_archive(scaled, trace_metadata, OUT / 'scaled-codec-stage-ref.gguf')
    published = GGUFReader(str(args.codec))
    weights = {}
    coefficients = []
    for tensor in published.tensors:
        assert tensor.tensor_type in (GGMLQuantizationType.F16, GGMLQuantizationType.F32)
        value = parameter(codec, tensor.name).detach().cpu().float().numpy().copy()
        assert value.size == tensor.data.size, tensor.name
        value = value.reshape(tensor.data.shape)
        data = np.asarray(tensor.data).copy()
        weights[tensor.name] = (value, data)
        coefficients.append({'name': tensor.name, 'shape': list(data.shape),
                             'published_type': tensor.tensor_type.name,
                             'original_f32_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                             'published_sha256': hashlib.sha256(data.tobytes()).hexdigest(),
                             'rounded_original_matches': bool(np.array_equal(value.astype(data.dtype), data)),
                             'max_absolute_weight_difference': float(np.max(np.abs(value - data.astype(np.float32))))})
    stats = {'checkpoint_sha256': digest(checkpoint), 'published_codec_sha256': digest(args.codec),
             'reference_sha256': digest(args.reference), 'canonical_reproduction_cosine': cosine,
             'canonical_reproduction_norm_ratio': norm, 'weight_norm_removal_pcm_exact': True,
             'coefficients': coefficients, 'variants': []}
    stats["source_files"] = {}
    # Save the actual installed driving sources as part of the proof closure.
    import inspect
    import dac.nn.layers
    import dac.nn.quantize
    import dac.model.dac
    for name, module in [('layers', dac.nn.layers), ('quantize', dac.nn.quantize), ('dac', dac.model.dac)]:
        path = OUT / f'original-dac-{name}.py'
        path.write_text(inspect.getsource(module))
        stats['source_files'][path.name] = digest(path)
    for variant in ('published', 'f32', 'output-f32', 'rvq-output-f32'):
        def promote(name):
            return (variant == 'f32' or variant in ('output-f32', 'rvq-output-f32')
                    and name.startswith(('decoder.snake1.', 'decoder.conv2.'))
                    or variant == 'rvq-output-f32' and name.startswith('quantizer.'))
        path = args.codec if variant == 'published' else OUT / f'dac-{variant}.gguf'
        writer = None if variant == 'published' else GGUFWriter(str(path), 'dac-44khz')
        promoted = []
        for name, (effective, data) in weights.items():
            selected = effective if promote(name) else data
            if promote(name):
                promoted.append(name)
            with torch.no_grad():
                destination = parameter(codec, name)
                destination.copy_(torch.from_numpy(selected.astype(np.float32).reshape(tuple(destination.shape))))
            if writer:
                writer.add_tensor(name, np.ascontiguousarray(selected))
        if writer:
            writer.add_string('crispasr.diagnostic.original_dac_sha256', digest(checkpoint))
            writer.add_string('crispasr.diagnostic.precision_variant', variant)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
        pcm = decode()
        np.save(OUT / f'original-{variant}-pcm.npy', pcm)
        trace = reference.capture_codec(codec, codes)
        np.testing.assert_array_equal(trace['decoded_audio'], pcm)
        control = dict(tensors, **trace)
        control_metadata = dict(trace_metadata, dia_dac_control_mode='precision-matched-original-modules',
                                dia_dac_weight_variant=variant, dia_dac_weight_sha256=digest(path))
        dump_reference.write_gguf_archive(control, control_metadata, OUT / f'{variant}-control-ref.gguf')
        stats['variants'].append({'variant': variant, 'codec': str(path), 'codec_sha256': digest(path),
                                  'codec_bytes': path.stat().st_size, 'promoted': promoted})
        (OUT / 'codec-weight-controls.json').write_text(json.dumps(stats, indent=2) + '\n')
    sys.exit(0)


receipt = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'reference_run': args.reference_run, 'runs': [], 'full_speech_acceptance': False,
           'production_defaults_changed': False}
def run(command, name, allow_failure=False):
    with (OUT / f'{name}.log').open('w') as log:
        process = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=7200)
    data = (OUT / f'{name}.log').read_bytes()
    receipt['runs'].append({'name': name, 'exit': process.returncode,
                            'log_sha256': hashlib.sha256(data).hexdigest()})
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(name, process.returncode, data.decode(errors='replace')[-1200:], flush=True)
    if not allow_failure:
        assert process.returncode == 0, name
    return process.returncode, data.decode(errors='replace')


archive = SCRATCH / 'preserved-reference'
run([sys.executable, ROOT / 'tests/test_dia_codec_parameters.py', '-v'], 'codec-parameter-contracts')
run([sys.executable, ROOT / 'tests/test_dia_reference_capture.py', 'CodecCaptureOwnershipTest', '-v'],
    'codec-capture-contracts')
run(['gh', 'run', 'download', args.reference_run, '-R', 'CrispStrobe/CrispASR', '-D', archive], 'reference-download')
folder = next(archive.glob('heavy-*'))
assert digest(folder / 'ref.gguf') == 'fd2f6ce7a82de51060578665cf5b49d3170d8a95ab0b0f021cc9288212e92767'
from huggingface_hub import hf_hub_download
repo, revision = 'cstr/dia-1.6b-GGUF', '3233fbcb32be47761d2e736857b6d1a075b9ba7e'
codec_path = hf_hub_download(repo, 'dac-44khz.gguf', revision=revision)
run([sys.executable, __file__, '--child', '--reference', folder / 'ref.gguf', '--codec', codec_path], 'original-codec-controls')
build = SCRATCH / 'dia-codec-build'
run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
     '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF',
     '-DCRISPASR_AMR=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-diff', '-j4'], 'build-diff')
model = hf_hub_download(repo, 'dia-1.6b-f16.gguf', revision=revision)
controls = json.loads((OUT / 'codec-weight-controls.json').read_text())
failed = []
for variant in controls['variants']:
    name = variant['variant']
    os.environ['DIA_CODEC'] = variant['codec']
    canonical_code, canonical_log = run([build / 'bin/crispasr-diff', 'dia', model, OUT / 'canonical-codec-ref.gguf'],
                                        name + '-canonical', True)
    control_code, control_log = run([build / 'bin/crispasr-diff', 'dia', model, OUT / f'{name}-control-ref.gguf'],
                                    name + '-precision-control', True)
    if control_code or name == 'f32' and canonical_code:
        failed.append(name)
    if name == 'f32' and '[PASS] encoder_output' in canonical_log:
        for negative, file in [('scaled', 'scaled-ref.gguf'), ('missing', 'missing-ref.gguf')]:
            code, log = run([build / 'bin/crispasr-diff', 'dia', model, folder / file], negative + '-negative', True)
            assert code != 0 and '[FAIL] encoder_output' in log, negative
    if name == 'f32' and '[PASS] dac_latent' in canonical_log:
        for negative, file, stage in [('scaled-codec', 'scaled-codec-stage-ref.gguf', 'dac_latent'),
                                     ('missing-codec', 'missing-codec-stage-ref.gguf', 'dac_block_0')]:
            code, log = run([build / 'bin/crispasr-diff', 'dia', model, OUT / file], negative + '-negative', True)
            assert code != 0 and '[FAIL] ' + stage in log, negative
receipt['codec_diagnostic_pass'] = not failed
receipt['scope'] = 'Preserved code replay with original, published, F32 and narrow F32 codec weights; canonical and precision-matched original PCM are distinct.'
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
assert not failed, failed
print('DIA_CODEC_PRECISION_DIAGNOSTIC_PASS; no full-speech or default-promotion claim', flush=True)
