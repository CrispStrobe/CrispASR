"""Actual released MLP on owned native inputs; one layer resident, no network reconstruction."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np

TRANSFORMERS = '5.6.0'
MODELING_SHA = '3e2b6239e4b2c3e512d4f9836a1dba12e975ae652407e84f2d3c2beebf0c9528'
MLP_SHA = 'c4e763e69d5676e71d222060e36258f6b2cc939a3597118c4207d370d120a265'
SUFFIXES = ['mlp_in', 'gate_out', 'up_out', 'down_in', 'down_out', 'mlp_out']


def native_observer_environment(arm):
    assert arm in ['off', 'on', 'off2']
    # OFF refers only to FFN capture. Block outputs remain observed in every arm.
    values = {'CRISPASR_INDEX_ECHO_DUMP_STAGES': '1'}
    if arm == 'on':
        values['CRISPASR_INDEX_ECHO_DUMP_FFN'] = '1'
    return values


def validate_native_observers(arm, environment):
    assert arm in ['off', 'on', 'off2']
    assert environment.get('CRISPASR_INDEX_ECHO_DUMP_STAGES') is not None, 'Block observer must remain enabled'
    assert (environment.get('CRISPASR_INDEX_ECHO_DUMP_FFN') == '1') == (arm == 'on'), 'Wrong FFN observer arm'


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), 'Input/observer identity failure'
    assert a.size and np.isfinite(a).all()


def metric(mine, reference):
    assert mine.shape == reference.shape and mine.size
    assert np.isfinite(mine).all() and np.isfinite(reference).all()
    a, b = mine.astype('f8').ravel(), reference.astype('f8').ravel()
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dict(cosine=float(a @ b / (an * bn)) if an and bn else float(an == bn),
                mine_norm=an, reference_norm=bn, norm_ratio=an / bn if bn else None,
                relative_l2=float(np.linalg.norm(a-b) / bn) if bn else None,
                max_abs=float(np.max(np.abs(a-b))))


def source_helper():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reference_backends import index_echo_ffn_capture as helper
    helper.verify_source(expected_version=TRANSFORMERS, expected_modeling_sha256=MODELING_SHA,
                         expected_class_sha256=MLP_SHA)
    return helper


def capture(mlp, values):
    """Call the actual MLP OFF/ON/OFF and retain the existing verified hooks."""
    import torch
    helper = source_helper()
    x = torch.from_numpy(values.copy()).reshape(1, 1, -1)
    wrapper = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]))
    with torch.inference_mode():
        off = mlp(x).detach().numpy().copy()
        with helper.FFNCapture(wrapper, [0], expected_version=TRANSFORMERS,
                expected_modeling_sha256=MODELING_SHA, expected_class_sha256=MLP_SHA) as hooks:
            on = mlp(x).detach().numpy().copy()
        after = mlp(x).detach().numpy().copy()
    exact(off, on)
    exact(off, after)
    arrays = {s: hooks.store['blk0_' + s].reshape(-1).copy() for s in SUFFIXES}
    exact(arrays['mlp_in'], values)
    exact(arrays['down_out'], arrays['mlp_out'])
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in mlp.modules())
    return arrays


def load_layer(reader, layer, text_config):
    import torch
    helper = source_helper()
    klass, provenance = helper._load_qwen3_5mlp_class()
    hidden, intermediate = text_config['hidden_size'], text_config['intermediate_size']
    config = SimpleNamespace(**text_config)
    with torch.device('meta'):
        mlp = klass(config, intermediate)
    state, ledger = {}, []
    for projection in ['gate', 'up', 'down']:
        key = f'model.layers.{layer}.mlp.{projection}_proj.weight'
        value = reader.get_tensor(key)
        expected = (hidden, intermediate) if projection == 'down' else (intermediate, hidden)
        assert value.dtype == torch.bfloat16 and tuple(value.shape) == expected
        assert torch.isfinite(value).all().item()
        ledger.append(dict(key=key, dtype=str(value.dtype), shape=list(value.shape),
            bytes=value.numel()*value.element_size(),
            sha256=hashlib.sha256(value.view(torch.uint16).numpy().tobytes()).hexdigest()))
        state[projection+'_proj.weight'] = value.float()
        del value
    resident_bytes = sum(p.numel()*p.element_size() for p in state.values())
    assert resident_bytes <= 768*1024**2, 'More than one bounded FFN layer resident'
    mlp.load_state_dict(state, strict=True, assign=True)
    mlp.eval()
    assert all(p.dtype == torch.float32 and not p.is_meta for p in mlp.parameters())
    return mlp, dict(source=provenance, tensors=ledger, resident_parameter_bytes=resident_bytes)


def attribute(mlp, native, expected_input_sha256, destination):
    import torch
    hidden, intermediate = mlp.hidden_size, mlp.intermediate_size
    assert set(native) == set(SUFFIXES)
    for name, value in native.items():
        width = intermediate if name in ['gate_out', 'up_out', 'down_in'] else hidden
        assert value.dtype == np.float32 and value.shape == (width,) and np.isfinite(value).all()
    assert array_sha(native['mlp_in']) == expected_input_sha256, 'Owned input ledger mismatch'
    original = capture(mlp, native['mlp_in'])
    with torch.inference_mode():
        down = mlp.down_proj(torch.from_numpy(native['down_in'].copy()).reshape(1, 1, -1)).numpy().reshape(-1).copy()
        product = (mlp.act_fn(torch.from_numpy(native['gate_out'].copy()).reshape(1, 1, -1)) *
                   torch.from_numpy(native['up_out'].copy()).reshape(1, 1, -1)).numpy().reshape(-1).copy()
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in original.items():
        np.save(destination/(name+'.npy'), value, allow_pickle=False)
    np.save(destination/'down-on-native-down-in.npy', down, allow_pickle=False)
    np.save(destination/'activation-on-native-gate-up.npy', product, allow_pickle=False)
    rows = {name: metric(native[name], original[name]) for name in SUFFIXES}
    rows['down_on_native_down_in'] = metric(native['down_out'], down)
    rows['activation_on_native_gate_up'] = metric(native['down_in'], product)
    return dict(metrics=rows, original_off_on_off_exact=True, input_exact=True,
                input_sha256=expected_input_sha256, scope='Same-input component attribution; no whole-model reference or quant acceptance')
