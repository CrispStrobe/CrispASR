#!/usr/bin/env python3
"""Original-only F32 arithmetic/F16 K/V rounding control; preserve all gates."""
import json
import os
from pathlib import Path
import sys


def round_f32_storage(tensor, witness, field):
    import torch
    if tensor.dtype != torch.float32:
        return tensor
    rounded = tensor.half().float()
    assert torch.isfinite(rounded).all(), 'F16 storage must not overflow'
    error = (rounded - tensor).abs()
    witness[field + '_changed_elements'] += int(torch.count_nonzero(error))
    witness[field + '_max_abs_rounding_error'] = max(
        witness[field + '_max_abs_rounding_error'], float(error.max()))
    return rounded


def main():
    os.environ.update(ATEN_CPU_CAPABILITY='avx2', ONEDNN_MAX_CPU_ISA='AVX2',
                      MKL_ENABLE_INSTRUCTIONS='AVX2')
    sys.path.insert(0, str(Path(__file__).parent))
    import torch
    import breeze_guided_reference
    torch.manual_seed(42)
    out = Path(os.environ['HEAVY_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    witness = dict(cache_calls=0, depth_attention_calls=0,
                   cache_changed_elements=0, cache_max_abs_rounding_error=0.0,
                   depth_changed_elements=0, depth_max_abs_rounding_error=0.0)

    def install():
        from transformers.cache_utils import DynamicCache
        import models.breeze as breeze
        original_update = DynamicCache.update
        original_attention = breeze.eager_attention_forward

        def update(cache, key, value, *args, **kwargs):
            if key.dtype == torch.float32:
                assert value.dtype == torch.float32
                witness['cache_calls'] += 1
                key = round_f32_storage(key, witness, 'cache')
                value = round_f32_storage(value, witness, 'cache')
            return original_update(cache, key, value, *args, **kwargs)

        def attention(module, query, key, value, *args, **kwargs):
            if query.dtype == torch.float32:
                assert key.dtype == value.dtype == torch.float32
                witness['depth_attention_calls'] += 1
                # CFG depth forwards use_cache=False in the original. Keep
                # that flow and round their complete K/V before attention.
                key = round_f32_storage(key, witness, 'depth')
                value = round_f32_storage(value, witness, 'depth')
            return original_attention(module, query, key, value, *args, **kwargs)

        DynamicCache.update = update
        breeze.eager_attention_forward = attention

    install.description = 'F16-rounded F32 DynamicCache writes and depth attention K/V; F32 attention math; BF16 text/codec unchanged'
    install.witness = witness
    (out / 'arithmetic-control-provenance.json').write_text(json.dumps(dict(
        torch_cpu_capability=torch.backends.cpu.get_cpu_capability(),
        torch_config=torch.__config__.show(), initialization_seed=42,
        canonical_reference=False, native_parity_accepted=False,
        kv_storage_control=install.description), indent=2) + '\n')
    breeze_guided_reference.main(promote_joint_arithmetic=True, kv_control_installer=install)
    assert witness['cache_calls'] > 0 and witness['depth_attention_calls'] > 0
    assert witness['cache_changed_elements'] > 0 and witness['depth_changed_elements'] > 0
    assert witness['cache_max_abs_rounding_error'] > 0 and witness['depth_max_abs_rounding_error'] > 0


if __name__ == '__main__':
    main()
