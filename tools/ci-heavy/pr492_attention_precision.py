#!/usr/bin/env python3
"""Diagnostic-only F32 TILE attention; preserves existing kernels and defaults.

Generate a separately named copy of the pinned TILE implementation using its
existing F32 branches, including matching host/device launch configuration.
Only an explicit flag on unmasked MiMo D=64/H=20 F32 tensors selects it.
K/V are still converted to F16 by launch_fattn; Q/softmax/accumulation use F32.
This is not full-F32 attention or production acceptance.
"""
import hashlib
import json
from pathlib import Path
import subprocess

FLAG = 'CRISPASR_DIAG_MIMO_TILE_F32'
MARKER = 'MIMO_DIAG_TILE_F32'


def patch_attention(repo, out):
    repo, out = Path(repo), Path(out)
    cuda = repo / 'ggml/src/ggml-cuda'
    tile = (cuda / 'fattn-tile.cuh').read_text()
    # Do not copy extern-template declarations: the new D64 specialization
    # must instantiate in fattn.cu rather than refer to an absent object file.
    declarations = '\nvoid ggml_cuda_flash_attn_ext_tile('
    assert tile.count(declarations) == 1
    generated = tile.split(declarations)[0]
    for before, after in [
        ('GGML_CUDA_FATTN_TILE_CONFIG_CASE', 'MIMO_DIAG_FATTN_TILE_CONFIG_CASE'),
        ('ggml_cuda_fattn_tile', 'mimo_diag_fattn_tile'),
        ('ggml_cuda_flash_attn_ext_tile_case', 'mimo_diag_flash_attn_ext_tile_case'),
        ('launch_fattn_tile', 'mimo_diag_launch_fattn_tile'),
        ('flash_attn_tile', 'mimo_diag_flash_attn_tile'),
        ('FAST_FP16_AVAILABLE', 'MIMO_DIAG_DISABLED_FAST_FP16')]:
        generated = generated.replace(before, after)
    # Device F32 branches must use the same F32 configuration on the host.
    before = 'if (fast_fp16_available(cc)) {'
    assert generated.count(before) == 1
    generated = generated.replace(before, 'if (false) { // diagnostic F32 host/device configuration')
    assert '#define MIMO_DIAG_DISABLED_FAST_FP16' not in generated
    header = cuda / 'mimo-diag-fattn-tile-f32.cuh'
    assert not header.exists()
    header.write_text('// Generated diagnostic from pinned fattn-tile.cuh; do not ship.\n' + generated)
    source = cuda / 'fattn.cu'
    original = source.read_text()
    include = '#include "fattn.cuh"'
    assert original.count(include) == 1
    changed = original.replace(include, include + '\n#include "mimo-diag-fattn-tile-f32.cuh"')
    before = '''void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
'''
    after = before + '''    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    if (getenv("CRISPASR_DIAG_MIMO_TILE_F32") && q->type == GGML_TYPE_F32 &&
        k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32 &&
        q->ne[0] == 64 && k->ne[0] == 64 && v->ne[0] == 64 &&
        q->ne[2] == 20 && k->ne[2] == 20 && v->ne[2] == 20 &&
        q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1 &&
        !dst->src[3] && !dst->src[4]) {
        static thread_local unsigned trace_count = 0;
        if (trace_count++ < 16) {
            fprintf(stderr, "MIMO_DIAG_TILE_F32 device=%d q=%lld kv=%lld d=64 heads=20\\n",
                    ctx.device, (long long) q->ne[1], (long long) k->ne[1]);
        }
        mimo_diag_flash_attn_ext_tile_case<64, 64>(ctx, dst);
        return;
    }
'''
    assert changed.count(before) == 1
    source.write_text(changed.replace(before, after))
    out.mkdir(parents=True, exist_ok=True)
    patch = out / 'cuda-attention.patch'
    patch.write_text(subprocess.check_output(['git', 'diff', '--', 'src/ggml-cuda/fattn.cu'],
                                            cwd=repo / 'ggml', text=True))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = dict(scope=__doc__, flag=FLAG, trace_marker=MARKER,
                   original_tile_sha256=digest(cuda / 'fattn-tile.cuh'),
                   generated_tile_sha256=digest(header), patched_dispatch_sha256=digest(source),
                   patch_sha256=digest(patch))
    (out / 'cuda-attention-patch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (out / header.name).write_bytes(header.read_bytes())
    return receipt
