#!/usr/bin/env python3
"""Independent model-free attention oracle; never a full-model quality gate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metric(mine, reference):
    assert mine.shape == reference.shape
    assert np.isfinite(mine).all() and np.isfinite(reference).all()
    a, b = mine.astype('f8').ravel(), reference.astype('f8').ravel()
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dict(cosine=float(a @ b / (an * bn)) if an and bn else float(an == bn),
                mine_norm=an, reference_norm=bn,
                relative_l2=float(np.linalg.norm(a-b)/bn) if bn else float(an),
                max_abs=float(np.max(np.abs(a-b))))


def same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def strict(a, b):
    row = metric(a, b)
    assert row['cosine'] > .999999 and row['relative_l2'] < 1e-5


def audit(root):
    rows, inputs = [], {}
    d, keys, heads = 32, 71, 2
    scale = np.float32(1/np.sqrt(np.float32(d)))
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        fields = dict(item for item in [
            ('pattern', int(folder.name.split('-')[0][1:])),
            ('queries', int(folder.name.split('-')[1][1:])),
            ('threads', int(folder.name.split('-')[2][1:])),
            ('half', int(folder.name.split('-')[3][3:])),
            ('reference', int(folder.name.split('-')[4][3:]))])
        p, t, half, ref = (fields[k] for k in ['pattern', 'queries', 'half', 'reference'])
        assert p in [0, 1] and t in [1, 63, 64, 71]
        assert fields['threads'] in [1, 4] and half in [0, 1] and ref in [0, 1]
        dtype, suffix = ('<f2', 'f16') if half else ('<f4', 'f32')
        q = np.fromfile(folder/'q.f32', '<f4').reshape(heads, t, d)
        k = np.fromfile(folder/('k.'+suffix), dtype).astype('f4').reshape(heads, keys, d)
        v = np.fromfile(folder/('v.'+suffix), dtype).astype('f4').reshape(heads, keys, d)
        mask = np.fromfile(folder/'mask.f16', '<f2').reshape(t, keys)
        scores = np.fromfile(folder/'scores.f32', '<f4').reshape(heads, t, keys)
        probabilities = np.fromfile(folder/'probabilities.f32', '<f4').reshape(heads, t, keys)
        eager = np.fromfile(folder/'eager.f32', '<f4').reshape(t, heads, d)
        flash = np.fromfile(folder/'flash.f32', '<f4').reshape(t, heads, d)
        same(q, np.broadcast_to(q[:, :1], q.shape).copy())
        same(mask, np.broadcast_to(mask[:1], mask.shape).copy())
        assert not np.isnan(mask).any() and np.isfinite(k).all() and np.isfinite(v).all()
        identity = [q[:, 0].copy(), k, v, mask[0].copy()]
        if p not in inputs:
            inputs[p] = identity
        else:
            for a, b in zip(identity, inputs[p]):
                same(a, b)
        # F64 oracle uses the exact native F16-representable K/V, never original weights.
        dot = np.einsum('hkd,hqd->hqk', k.astype('f8'), q.astype('f8'))
        scaled = dot * float(scale) + mask.astype('f8')
        exps = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        prob = exps/exps.sum(axis=-1, keepdims=True)
        oracle = np.einsum('hqk,hkd->qhd', prob, v.astype('f8'))
        comparisons = dict(scores=metric(scores, dot), probabilities=metric(probabilities, prob),
                           eager=metric(eager, oracle), flash=metric(flash, oracle),
                           flash_vs_eager=metric(flash, eager))
        assert comparisons['scores']['relative_l2'] < 1e-6
        assert comparisons['probabilities']['relative_l2'] < 1e-6
        assert comparisons['eager']['relative_l2'] < 1e-6
        if not half or (not ref and t >= 64):
            assert comparisons['flash']['relative_l2'] < 1e-6
        if p == 0:
            same(k, np.zeros_like(k))
            same(q, np.zeros_like(q))
            same(v, np.broadcast_to(v[:, :1], v.shape).copy())
            constant = np.broadcast_to(v[:, 0][None], flash.shape).copy()
            # Algebraic identity: equal V rows imply attention output V for any probabilities summing to one.
            strict(eager, constant)
            if half and (ref or t < 64):
                accum = np.zeros((heads, d), dtype='f2')
                for _ in range(keys):
                    accum = (accum.astype('f4') + v[:, 0]).astype('f2')
                half_oracle = accum.astype('f4') * (np.float32(1)/np.float32(keys))
                same(flash, np.broadcast_to(half_oracle[None], flash.shape).copy())
                assert comparisons['flash']['relative_l2'] > 1e-4
                comparisons['half_accumulator_prediction_exact'] = True
            else:
                strict(flash, constant)
        rows.append(dict(cell=folder.name, **fields, metrics=comparisons))
    assert len(rows) == 64
    # Known errors must fail a magnitude-aware gate even when cosine is one.
    control = np.arange(1, 65, dtype='f4').reshape(2, 32)
    strict(control, control)
    rejected = []
    for name, changed in [('scale-times-two', control*2), ('channel-order', control[:, ::-1]),
                          ('shape', control.reshape(32, 2))]:
        try:
            strict(changed, control)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Negative did not reject: '+name)
    owned = control.copy()
    digest = hashlib.sha256(owned.tobytes()).hexdigest()
    owned[0, 0] += 1
    assert hashlib.sha256(owned.tobytes()).hexdigest() != digest
    return dict(operator_attribution_pass=True, cells=rows, negative_controls=rejected+['changed-input'],
                constant_v_half_accumulation_prediction='Byte-exact for all scalar/reference F16 cells',
                scope='Tiny same-input CPU operator diagnosis, not actual MiMo Q/K/V replay or model-quality acceptance',
                explains_71_row_prefill_failure=False, production_changes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve failed audits'
    result = audit(args.inputs)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('INDEPENDENT_ATTENTION_OPERATOR_AUDIT_PASS cells=64 model_acceptance=false')


if __name__ == '__main__':
    main()
