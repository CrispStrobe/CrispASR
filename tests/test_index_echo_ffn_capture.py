"""
Tiny tests for Index-Echo FFN capture helper (actual-class, no handwritten FFN).

Uses the actual Qwen3_5MLP from the installed transformers distribution via
AST execution of the real ClassDef source.  No large weights are required;
no new recipe or Q4 claim is made.
"""

import hashlib
import pathlib
import re

import numpy as np
import pytest
import torch


def _current_provenance():
    """Return actual version/modeling/class SHAs for the currently installed transformers."""
    import importlib.metadata, hashlib, pathlib, ast
    from tools.reference_backends.index_echo_ffn_capture import _file_sha256, _extract_class_source, _locate_modeling_file
    modeling_path = _locate_modeling_file()
    ver = importlib.metadata.version("transformers")
    file_sha = _file_sha256(modeling_path)
    class_src = _extract_class_source(modeling_path)
    class_sha = hashlib.sha256(class_src.encode()).hexdigest()
    return ver, file_sha, class_sha, modeling_path


def _load_actual_mlp():
    from tools.reference_backends.index_echo_ffn_capture import _load_qwen3_5mlp_class
    Qwen3_5MLP, prov = _load_qwen3_5mlp_class(None)
    return Qwen3_5MLP, prov


def _tiny_config(hidden=8, intermediate=16):
    class C:
        hidden_size = hidden
        intermediate_size = intermediate
        hidden_act = "silu"
    return C()


def _make_llm_with_mlps(n_layers=2, hidden=8, intermediate=16):
    Qwen3_5MLP, _ = _load_actual_mlp()
    class Layer:
        pass
    class Model:
        pass
    class LLM:
        pass
    llm = LLM()
    llm.model = Model()
    llm.model.layers = []
    for _ in range(n_layers):
        layer = Layer()
        cfg = _tiny_config(hidden, intermediate)
        mlp = Qwen3_5MLP(cfg, intermediate)
        layer.mlp = mlp
        llm.model.layers.append(layer)
    return llm


def test_actual_class_forward_off_on_off_identity():
    """OFF/ON/OFF proves hook does not alter output; capture matches real forward with torch.equal."""
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1, hidden=8, intermediate=16)
    x = torch.randn(2, 4, 8)
    x_clone = x.clone()
    mlp = llm.model.layers[0].mlp
    with torch.no_grad():
        off1 = mlp(x).clone()
        # Ensure off1 is not affected by later mutation
        assert torch.equal(off1, mlp(x_clone))
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    with FFNCapture(llm, layers=[0], call_index=0,
                    expected_version=ver,
                    expected_modeling_sha256=fsha,
                    expected_class_sha256=csha,
                    last_token_only=False) as cap:
        with torch.no_grad():
            on = mlp(x)
        store = cap.store
        assert "blk0_mlp_in" in store and "blk0_mlp_out" in store
        assert "blk0_gate_out" in store and "blk0_up_out" in store
        assert "blk0_down_in" in store and "blk0_down_out" in store
        # torch.equal exact, not allclose
        assert torch.equal(torch.from_numpy(store["blk0_mlp_out"]), on.float().cpu()), "captured mlp_out != forward output"
        assert store["blk0_gate_out"].shape == (2, 4, 16)
        assert store["blk0_up_out"].shape == (2, 4, 16)
        from transformers.activations import ACT2FN
        gate = torch.from_numpy(store["blk0_gate_out"])
        up = torch.from_numpy(store["blk0_up_out"])
        hidden = ACT2FN["silu"](gate) * up
        assert np.allclose(hidden.numpy(), store["blk0_down_in"], atol=1e-6)
        # Output observation should not change bytes: off1 vs on still equal
        assert torch.equal(off1, on), "hook changed output bytes"
    with torch.no_grad():
        off2 = mlp(x)
    assert torch.equal(off1, on)
    assert torch.equal(off1, off2)


def test_selected_call_alignment():
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1, hidden=4, intermediate=8)
    mlp = llm.model.layers[0].mlp
    x0 = torch.randn(1, 2, 4)
    x1 = torch.randn(1, 2, 4) * 2
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    with FFNCapture(llm, layers=[0], call_index=1,
                    expected_version=ver,
                    expected_modeling_sha256=fsha,
                    expected_class_sha256=csha,
                    last_token_only=False) as cap:
        with torch.no_grad():
            mlp(x0)
            mlp(x1)
        assert np.allclose(cap.store["blk0_mlp_in"], x1.numpy(), atol=1e-6)
        assert not np.allclose(cap.store["blk0_mlp_in"], x0.numpy(), atol=1e-6)


def test_wrong_source_rejection():
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1)
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    with pytest.raises(RuntimeError, match="version mismatch|SHA mismatch"):
        with FFNCapture(llm, layers=[0], call_index=0,
                        expected_version="0.0.0",
                        expected_modeling_sha256=fsha,
                        expected_class_sha256=csha):
            pass
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        with FFNCapture(llm, layers=[0], call_index=0,
                        expected_version=ver,
                        expected_modeling_sha256="0"*64,
                        expected_class_sha256=csha):
            pass


def test_invalid_layer_cleanup():
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=2, hidden=4, intermediate=8)
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    cap = FFNCapture(llm, layers=[0, 0], call_index=0,
                     expected_version=ver,
                     expected_modeling_sha256=fsha,
                     expected_class_sha256=csha)
    with pytest.raises(ValueError, match="duplicate"):
        with cap:
            pass
    assert cap.handles == []
    cap2 = FFNCapture(llm, layers=[99], call_index=0,
                      expected_version=ver,
                      expected_modeling_sha256=fsha,
                      expected_class_sha256=csha)
    with pytest.raises(ValueError, match="invalid layer"):
        with cap2:
            pass
    assert cap2.handles == []


def test_empty_layers_rejected():
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1)
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    with pytest.raises(ValueError, match="non-empty"):
        with FFNCapture(llm, layers=[], call_index=0,
                        expected_version=ver,
                        expected_modeling_sha256=fsha,
                        expected_class_sha256=csha):
            pass


def test_call_index_validation():
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1)
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    # negative
    with pytest.raises(ValueError, match=">=0"):
        with FFNCapture(llm, layers=[0], call_index=-1,
                        expected_version=ver,
                        expected_modeling_sha256=fsha,
                        expected_class_sha256=csha):
            pass
    # float should not be coerced
    with pytest.raises(TypeError, match="int"):
        with FFNCapture(llm, layers=[0], call_index=1.0,
                        expected_version=ver,
                        expected_modeling_sha256=fsha,
                        expected_class_sha256=csha):
            pass
    # bool is not allowed
    with pytest.raises(TypeError, match="int"):
        with FFNCapture(llm, layers=[0], call_index=True,
                        expected_version=ver,
                        expected_modeling_sha256=fsha,
                        expected_class_sha256=csha):
            pass
    # keyword-only enforcement: positional should fail
    with pytest.raises(TypeError):
        FFNCapture(llm, [0], "5.5.4", fsha, csha)  # type: ignore


def test_repeat_enter_reuse_rejected():
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1, hidden=4, intermediate=8)
    mlp = llm.model.layers[0].mlp
    x = torch.randn(1, 2, 4)
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    cap = FFNCapture(llm, layers=[0], call_index=0,
                     expected_version=ver,
                     expected_modeling_sha256=fsha,
                     expected_class_sha256=csha,
                     last_token_only=False)
    with cap:
        with torch.no_grad():
            mlp(x)
    # Second enter should fail
    with pytest.raises(RuntimeError, match="already entered"):
        with cap:
            pass


def test_owned_array_retention():
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1, hidden=4, intermediate=8)
    mlp = llm.model.layers[0].mlp
    x = torch.randn(1, 2, 4)
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    with FFNCapture(llm, layers=[0], call_index=0,
                    expected_version=ver,
                    expected_modeling_sha256=fsha,
                    expected_class_sha256=csha,
                    last_token_only=False) as cap:
        with torch.no_grad():
            mlp(x)
        captured = cap.captured()
        orig = cap.store["blk0_mlp_out"].copy()
        captured["blk0_mlp_out"][:] = 0
        assert np.allclose(cap.store["blk0_mlp_out"], orig, atol=1e-6)
        x2 = torch.randn(1, 2, 4) * 3
        with torch.no_grad():
            mlp(x2)
        assert np.allclose(cap.store["blk0_mlp_out"], orig, atol=1e-6)


def test_owned_input_snapshot():
    """Mutating ORIGINAL x after capture must not affect stored mlp_in."""
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1, hidden=4, intermediate=8)
    mlp = llm.model.layers[0].mlp
    x = torch.randn(1, 2, 4)
    x_before = x.clone()
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    with FFNCapture(llm, layers=[0], call_index=0,
                    expected_version=ver,
                    expected_modeling_sha256=fsha,
                    expected_class_sha256=csha,
                    last_token_only=False) as cap:
        with torch.no_grad():
            mlp(x)
        # Mutate original x
        x.add_(10)
        assert np.allclose(cap.store["blk0_mlp_in"], x_before.numpy(), atol=1e-6), "stored mlp_in was not owned copy of input"
        assert not np.allclose(cap.store["blk0_mlp_in"], x.numpy(), atol=1e-6)


def test_complete_capture_set_required():
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1, hidden=4, intermediate=8)
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    with pytest.raises(RuntimeError, match="incomplete capture"):
        with FFNCapture(llm, layers=[0], call_index=0,
                        expected_version=ver,
                        expected_modeling_sha256=fsha,
                        expected_class_sha256=csha,
                        last_token_only=False) as cap:
            pass

def test_handwritten_lookalike_rejected_before_hooks():
    """A handwritten lookalike MLP (not from AST) must be rejected before any hook, not just file hash mismatch."""
    ver, fsha, csha, _ = _current_provenance()
    # Create a handwritten lookalike that mimics structure but is not from pinned source
    import torch.nn as nn
    from transformers.activations import ACT2FN

    class HandwrittenMLP(nn.Module):
        def __init__(self, hidden=4, intermediate=8):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
            self.act_fn = ACT2FN["silu"]
        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class Layer:
        pass
    class Model:
        pass
    class LLM:
        pass
    llm = LLM()
    llm.model = Model()
    llm.model.layers = []
    layer = Layer()
    layer.mlp = HandwrittenMLP(hidden=4, intermediate=8)
    llm.model.layers.append(layer)

    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    # Even though file hashes would match (we pass correct expected hashes for real class),
    # the handwritten class lacks _crispasr_source_hashes and its source file is this test file,
    # so its file hash won't match expected modeling file hash -> should raise before hooks
    with pytest.raises(RuntimeError, match="modeling file hash mismatch|class hash mismatch"):
        with FFNCapture(llm, layers=[0], call_index=0,
                        expected_version=ver,
                        expected_modeling_sha256=fsha,
                        expected_class_sha256=csha,
                        last_token_only=False) as cap:
            # Should not reach here; validation happens on enter, before hooks
            import torch
            with torch.no_grad():
                llm.model.layers[0].mlp(torch.randn(1, 2, 4))
    # Ensure no handles leaked after rejection
    # Create a new cap to check that previous failure didn't leave handles
    cap2 = FFNCapture(llm, layers=[0], call_index=0,
                      expected_version=ver,
                      expected_modeling_sha256=fsha,
                      expected_class_sha256=csha)
    # This should still fail for same reason, but not due to leftover handles
    with pytest.raises(RuntimeError):
        with cap2:
            pass
    assert cap2.handles == []

def test_last_token_slicing_shape():
    """last_token_only must slice ALL six arrays to [1,1,width]."""
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1, hidden=8, intermediate=16)
    mlp = llm.model.layers[0].mlp
    x = torch.randn(1, 3, 8)  # batch 1, seq 3, hidden 8
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    with FFNCapture(llm, layers=[0], call_index=0,
                    expected_version=ver,
                    expected_modeling_sha256=fsha,
                    expected_class_sha256=csha,
                    last_token_only=True) as cap:
        with torch.no_grad():
            mlp(x)
        # All six should be [1,1,width]
        assert cap.store["blk0_mlp_in"].shape == (1, 1, 8), f"mlp_in shape {cap.store['blk0_mlp_in'].shape}"
        assert cap.store["blk0_mlp_out"].shape == (1, 1, 8)
        assert cap.store["blk0_gate_out"].shape == (1, 1, 16)
        assert cap.store["blk0_up_out"].shape == (1, 1, 16)
        assert cap.store["blk0_down_in"].shape == (1, 1, 16)
        assert cap.store["blk0_down_out"].shape == (1, 1, 8)
    # Without last_token_only, shapes should be full [1,3,width]
    with FFNCapture(llm, layers=[0], call_index=0,
                    expected_version=ver,
                    expected_modeling_sha256=fsha,
                    expected_class_sha256=csha,
                    last_token_only=False) as cap2:
        with torch.no_grad():
            mlp(x)
        assert cap2.store["blk0_mlp_in"].shape == (1, 3, 8)
        assert cap2.store["blk0_gate_out"].shape == (1, 3, 16)


def test_partial_installation_failure_removes_all_hooks(monkeypatch):
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1)
    mlp = llm.model.layers[0].mlp
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    def fail(*args, **kwargs):
        raise RuntimeError('deliberate hook installation failure')
    monkeypatch.setattr(mlp.up_proj, 'register_forward_hook', fail)
    cap = FFNCapture(llm, [0], expected_version=ver,
                     expected_modeling_sha256=fsha, expected_class_sha256=csha)
    with pytest.raises(RuntimeError, match='deliberate hook'):
        with cap:
            pass
    assert not cap.handles
    assert not mlp._forward_hooks and not mlp.gate_proj._forward_hooks
    assert not mlp.down_proj._forward_hooks


def test_output_directory_is_owned_and_never_overwritten(tmp_path):
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1)
    mlp = llm.model.layers[0].mlp
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    options = dict(expected_version=ver, expected_modeling_sha256=fsha,
                   expected_class_sha256=csha, out_dir=tmp_path/'capture')
    with FFNCapture(llm, [0], **options) as cap:
        with torch.no_grad():
            mlp(torch.randn(1, 3, 8))
        cap.finalize()  # Explicit finalization and context exit are idempotent.
    saved = {p.name: p.read_bytes() for p in (tmp_path/'capture').iterdir()}
    assert len(saved) == 7
    with pytest.raises(FileExistsError):
        with FFNCapture(llm, [0], **options):
            pass
    assert saved == {p.name: p.read_bytes() for p in (tmp_path/'capture').iterdir()}
    assert not mlp._forward_hooks and not mlp.gate_proj._forward_hooks


def test_public_reference_rejects_unpinned_source_before_weights(monkeypatch):
    from tools.reference_backends import index_echo
    def no_weights(*args, **kwargs):
        raise AssertionError('weights must not be loaded for invalid provenance')
    monkeypatch.setattr(index_echo, 'load_blueprint', no_weights)
    monkeypatch.setenv('INDEX_ECHO_REF_CAPTURE_FFN', '1')
    monkeypatch.delenv('INDEX_ECHO_REF_CAPTURE_GENERATION', raising=False)
    for key in ['INDEX_ECHO_REF_TRANSFORMERS_VERSION', 'INDEX_ECHO_REF_MODELING_SHA256',
                'INDEX_ECHO_REF_MLP_SHA256']:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(KeyError, match='INDEX_ECHO_REF_TRANSFORMERS_VERSION'):
        index_echo.dump('unused', None, set())
    ver, fsha, csha, _ = _current_provenance()
    monkeypatch.setenv('INDEX_ECHO_REF_TRANSFORMERS_VERSION', ver)
    monkeypatch.setenv('INDEX_ECHO_REF_MODELING_SHA256', '0'*64)
    monkeypatch.setenv('INDEX_ECHO_REF_MLP_SHA256', csha)
    with pytest.raises(RuntimeError, match='modeling file SHA mismatch'):
        index_echo.dump('unused', None, set())


def test_public_generation_mode_rejects_unsupported_ffn_capture(monkeypatch):
    from tools.reference_backends import index_echo
    monkeypatch.setenv('INDEX_ECHO_REF_CAPTURE_FFN', '1')
    monkeypatch.setenv('INDEX_ECHO_REF_CAPTURE_GENERATION', '1')
    with pytest.raises(ValueError, match='initial-prefill'):
        index_echo.dump('unused', None, set())


def test_subclass_cannot_inherit_verified_source_provenance():
    ver, fsha, csha, _ = _current_provenance()
    actual, _ = _load_actual_mlp()
    class Changed(actual):
        def forward(self, x):
            return super().forward(x) * 2
    llm = _make_llm_with_mlps(n_layers=1)
    llm.model.layers[0].mlp = Changed(_tiny_config(), 16)
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    cap = FFNCapture(llm, [0], expected_version=ver,
                     expected_modeling_sha256=fsha, expected_class_sha256=csha)
    with pytest.raises(RuntimeError, match='model MLP class hash mismatch'):
        with cap:
            pass
    assert not cap.handles and not llm.model.layers[0].mlp._forward_hooks


def test_last_token_capture_transfers_only_requested_rows(monkeypatch):
    ver, fsha, csha, _ = _current_provenance()
    llm = _make_llm_with_mlps(n_layers=1)
    original_cpu = torch.Tensor.cpu
    copied = []
    def observed_cpu(tensor, *args, **kwargs):
        copied.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', observed_cpu)
    from tools.reference_backends.index_echo_ffn_capture import FFNCapture
    with FFNCapture(llm, [0], expected_version=ver,
                    expected_modeling_sha256=fsha, expected_class_sha256=csha,
                    last_token_only=True):
        with torch.no_grad():
            llm.model.layers[0].mlp(torch.randn(1, 37, 8))
    assert len(copied) == 6
    assert sorted(copied) == [8, 8, 8, 16, 16, 16]
