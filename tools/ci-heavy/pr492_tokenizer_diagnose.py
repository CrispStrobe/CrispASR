#!/usr/bin/env python3
"""Isolate MiMo Q4 activation rounding from the encoder attention calculation.

This is a diagnostic, not PR acceptance or original-checkpoint certification.
Promote the SAME dequantized Q4 matrix weights to F32, without recovering any
lost weight precision. Compare both native attention paths and the official
Python transformer on identical native conv2 inputs. Also check pooling against
PyTorch on each path's own input. No LM weights or GPU are needed.
"""
import ast
import ctypes as C
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import gguf
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

from pr492_acceptance import ROOT, PINS, TOK_STAGES, digest, metrics, run

UPSTREAM = '691ce54144a6844cc641fd96046a6ba20776c8b0'


def native(library, codec, audio, dest, flash):
    class Params(C.Structure):
        _fields_ = [('n_threads', C.c_int), ('verbosity', C.c_int),
                    ('use_gpu', C.c_bool), ('flash_attn', C.c_bool)]
    lib = C.CDLL(library)
    lib.mimo_tokenizer_context_default_params.restype = Params
    lib.mimo_tokenizer_init_from_file.argtypes = [C.c_char_p, Params]
    lib.mimo_tokenizer_init_from_file.restype = C.c_void_p
    lib.mimo_tokenizer_free.argtypes = [C.c_void_p]
    lib.mimo_tokenizer_extract_stage.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int,
                                                C.c_char_p, C.POINTER(C.c_int)]
    lib.mimo_tokenizer_extract_stage.restype = C.POINTER(C.c_float)
    libc = C.CDLL(None)
    libc.free.argtypes = [C.c_void_p]
    p = lib.mimo_tokenizer_context_default_params()
    p.n_threads, p.verbosity, p.use_gpu, p.flash_attn = 4, 0, False, bool(int(flash))
    ctx = lib.mimo_tokenizer_init_from_file(os.fsencode(codec), p)
    assert ctx
    pcm, sr = sf.read(audio, dtype='float32')
    assert sr == 16000 and pcm.ndim == 1
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        for stage in TOK_STAGES:
            n = C.c_int()
            ptr = lib.mimo_tokenizer_extract_stage(ctx, pcm.ctypes.data_as(C.POINTER(C.c_float)),
                                                   len(pcm), stage.encode(), C.byref(n))
            assert ptr and n.value > 0, stage
            try:
                data = np.ctypeslib.as_array(ptr, (n.value,)).copy()
            finally:
                libc.free(ptr)
            assert np.isfinite(data).all()
            np.save(dest / (stage + '.npy'), data)
    finally:
        lib.mimo_tokenizer_free(ctx)


def official_classes(scratch, receipt):
    """Execute the pinned official classes, replacing only the CUDA call.

    AST selection avoids importing the unused vocoder/decoder and flash-attn
    package. Attention.forward, TransformerLayer.forward and RoPE are the
    upstream source itself. Batch=1, unmasked/noncausal full attention only.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F
    from transformers.configuration_utils import PretrainedConfig
    from transformers.utils import is_torch_available, logging
    from functools import wraps
    from typing import Optional

    env = dict(torch=torch, nn=nn, F=F, math=math, Optional=Optional,
               PretrainedConfig=PretrainedConfig, wraps=wraps,
               is_torch_available=is_torch_available, logging=logging,
               logger=logging.get_logger(__name__))
    def flash(q, k, v, cq, ck, mq, mk, causal=False, window_size=(-1, -1)):
        assert not causal and tuple(window_size) == (-1, -1)
        assert cq.tolist() == ck.tolist() == [0, q.shape[0]]
        a = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                          v.transpose(0, 1)[None], dropout_p=0.0, is_causal=False)
        return a[0].transpose(0, 1).contiguous()
    env['flash_attn_varlen_func'] = flash
    for filename, names in [
        ('modeling_rope_utils.py', {'dynamic_rope_update', '_compute_default_rope_parameters',
                                  'rotate_half', 'apply_rotary_pos_emb'}),
        ('modeling_audio_tokenizer.py', {'RotaryEmbedding', 'RMSNorm', 'Attention', 'TransformerLayer'}),
    ]:
        url = f'https://raw.githubusercontent.com/XiaomiMiMo/MiMo-Audio/{UPSTREAM}/src/mimo_audio_tokenizer/{filename}'
        path = scratch / filename
        urllib.request.urlretrieve(url, path)
        receipt.setdefault('upstream_files', {})[filename] = dict(url=url, sha256=digest(path))
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes} == names
        if filename == 'modeling_audio_tokenizer.py':
            env['ROPE_INIT_FUNCTIONS'] = {'default': env['_compute_default_rope_parameters']}
            # TransformerLayer resolves this at instantiation, after RMSNorm
            # has been defined by the source module.
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
        if filename == 'modeling_audio_tokenizer.py':
            env['LAYER_NORM'] = {'LayerNorm': nn.LayerNorm, 'RMSNorm': env['RMSNorm']}
    return env


def main():
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'pr492-tokenizer'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                      CRISPASR_MIMO_FORCE_CPU='1', CRISPASR_GGUF_MMAP='1')
    os.environ.pop('CRISPASR_CORE_ATTN_EAGER_F32', None)
    receipt = dict(passed=False, scope=__doc__, upstream=UPSTREAM,
                   source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    def save():
        (out / 'tokenizer-diagnosis.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    build = scratch / 'build'
    run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
         '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DCRISPASR_MEL_BLAS=OFF',
         '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=OFF'], out / 'configure.log')
    run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], out / 'build.log')
    library = next(build.rglob('libcrispasr.so'))
    repo, rev, filename, sha = PINS['codec']
    codec = Path(hf_hub_download(repo, filename, revision=rev, local_dir=scratch / 'models'))
    assert digest(codec) == sha
    receipt['codec'] = dict(repo=repo, revision=rev, filename=filename, sha256=sha)
    audio = scratch / 'en.wav'
    run(['ffmpeg', '-y', '-i', ROOT / 'samples/jfk.mp3', '-ar', '16000', '-ac', '1', audio], out / 'resample.log')
    reader = gguf.GGUFReader(str(codec))
    tensors = {t.name: t for t in reader.tensors}
    receipt['tensor_types'] = {t.name: t.tensor_type.name for t in reader.tensors}
    promoted = scratch / 'same-weights-f32.gguf'
    writer = gguf.GGUFWriter(str(promoted), 'mimo_tokenizer', use_temp_file=True)
    for name, field in reader.fields.items():
        if name.startswith('GGUF.') or name in {'general.architecture', 'general.file_type', 'general.quantization_version'}:
            continue
        writer.add_key_value(name, field.contents(), field.types[0],
                             field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
    def weight(name):
        t = tensors[name]
        return np.ascontiguousarray(gguf.dequantize(t.data, t.tensor_type).reshape(tuple(t.shape[::-1])), dtype=np.float32)
    promoted_names = []
    for t in reader.tensors:
        if t.tensor_type not in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            writer.add_tensor(t.name, weight(t.name))
            promoted_names.append(t.name)
        else:
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    del writer
    gc.collect()
    receipt['promoted_quantized_weights'] = promoted_names
    receipt['promoted_sha256'] = digest(promoted)
    save()
    for kind, model in [('q4', codec), ('promoted', promoted)]:
        for flash in [1, 0]:
            arm = f'{kind}-' + ('flash' if flash else 'eager')
            run([sys.executable, __file__, '--native', library, model, audio, out / arm, str(flash)], out / (arm + '.log'))
    def data(arm, stage):
        return np.load(out / arm / (stage + '.npy'))
    receipt['attention_ab'] = {}
    for kind in ['q4', 'promoted']:
        receipt['attention_ab'][kind] = {}
        for stage in TOK_STAGES:
            a, b = data(kind + '-eager', stage), data(kind + '-flash', stage)
            receipt['attention_ab'][kind][stage] = (
                dict(match_fraction=float(np.mean(a == b)), exact=np.array_equal(a, b))
                if stage == 'tok_codes' else metrics(a, b))
    save()
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    env = official_classes(scratch, receipt)
    h = torch.from_numpy(data('promoted-flash', 'tok_conv2_out').reshape(-1, 1280))
    assert np.array_equal(h.numpy().ravel(), data('promoted-eager', 'tok_conv2_out'))
    lengths = torch.tensor([len(h)])
    rope = env['RotaryEmbedding'](10000.0, 64, len(h))
    position = rope(h, torch.arange(len(h)))
    def tensor(name):
        return torch.from_numpy(weight(name))
    def pool(h):
        x = h.T[None]
        if h.shape[0] % 2:
            x = F.pad(x, (0, 1))
        x = F.gelu(F.conv1d(x, tensor('encoder.down_sample_layer.0.weight'), stride=2))[0].T
        return F.layer_norm(x, (1280,), tensor('encoder.down_sample_norm.weight'),
                            tensor('encoder.down_sample_norm.bias'), 1e-5)
    receipt['pool_isolated'] = {}
    with torch.inference_mode():
        for arm in ['q4-flash', 'q4-eager', 'promoted-flash', 'promoted-eager']:
            reference = pool(torch.from_numpy(data(arm, 'tok_xfmr_out').reshape(-1, 1280))).numpy()
            receipt['pool_isolated'][arm] = metrics(data(arm, 'tok_pool_out').reshape(-1, 1280), reference)
        save()
        skip = None
        for il in range(32):
            layer = env['TransformerLayer'](F.gelu, 1280, 20, 5120, False).eval()
            replacements = {'.self_attn.q_proj.': '.attn.q.', '.self_attn.k_proj.': '.attn.k.',
                            '.self_attn.v_proj.': '.attn.v.', '.self_attn.out_proj.': '.attn.o.',
                            '.self_attn_layer_norm.': '.attn_norm.', '.final_layer_norm.': '.ffn_norm.'}
            state = {}
            for name in layer.state_dict():
                mapped = '.' + name
                for old, new in replacements.items():
                    mapped = mapped.replace(old, new)
                state[name] = tensor(f'enc.blk.{il}' + mapped)
            layer.load_state_dict(state, strict=True)
            h = layer(h, lengths, position)
            np.save(out / f'python-layer-{il:02d}.npy', h.numpy())
            if il == 2:
                skip = h.clone()
            del layer, state
        h = F.layer_norm(h + skip, (1280,), tensor('enc.norm.weight'), tensor('enc.norm.bias'), 1e-5)
        reference = {'tok_xfmr_out': h.numpy(), 'tok_pool_out': pool(h).numpy()}
        receipt['independent_same_weights'] = {}
        for stage, value in reference.items():
            np.save(out / ('python-' + stage + '.npy'), value)
            receipt['independent_same_weights'][stage] = {
                arm: metrics(data(arm, stage).reshape(value.shape), value)
                for arm in ['q4-flash', 'q4-eager', 'promoted-flash', 'promoted-eager']}
    receipt['scale_negative_control'] = metrics(reference['tok_pool_out'] * 2, reference['tok_pool_out'])
    save()
    failures = []
    for kind, stages in receipt['attention_ab'].items():
        if kind == 'promoted':
            for stage, m in stages.items():
                if stage != 'tok_codes' and not (m['cosine'] >= .9999 and m['relative_l2'] <= .005):
                    failures.append(f'promoted attention A/B {stage}')
    for stage, arms in receipt['independent_same_weights'].items():
        for arm, m in arms.items():
            if arm.startswith('promoted') and not (m['cosine'] >= .9999 and m['relative_l2'] <= .005):
                failures.append(f'independent {arm} {stage}')
    for arm, m in receipt['pool_isolated'].items():
        if not (m['cosine'] >= .9999 and m['relative_l2'] <= .005):
            failures.append(f'isolated pooling {arm}')
    receipt['failures'] = failures
    receipt['passed'] = not failures
    save()
    assert not failures, failures
    print('MIMO_TOKENIZER_DIAGNOSIS_PASS; not PR acceptance', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--native':
        native(*sys.argv[2:])
    else:
        main()
