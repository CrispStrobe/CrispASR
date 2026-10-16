#!/usr/bin/env python3
"""Real tiny Dia constructor/capture control; final DAC is expressly a stub."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
import dia_direct_original as worker

spec = importlib.util.spec_from_file_location('tiny_original_fixture', ROOT / 'tests/test_dia_reference_capture.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
fixture.DiaCaptureTest.setUpClass()
torch = fixture.DiaCaptureTest.torch
captured_dia = fixture.DiaCaptureTest.dia
from dia.model import Dia

direct = Dia(captured_dia.config, compute_dtype='float32', device=torch.device('cpu'))
direct.model.load_state_dict(captured_dia.model.state_dict(), strict=True)
direct.model.eval()
direct.dac_model = captured_dia.dac_model
worker.no_hooks(direct.model)
for key, case in worker.CASES.items():
    torch.manual_seed(case['seed'])
    with torch.inference_mode():
        pcm = direct.generate(case['text'], max_tokens=32, cfg_scale=3., temperature=1.2,
                              top_p=.95, cfg_filter_top_k=45, use_torch_compile=False)
    result = fixture.reference.capture(captured_dia, case['text'], max_tokens=32,
        seed=case['seed'], temperature=1.2, top_p=.95, top_k=45, cfg_scale=3.,
        attention_layers=(0, 1), boundary_layers=(0, 1))
    assert worker.compare_pcm(pcm, result['decoded_audio'])['exact'], key
    print('REAL_TINY_CONSTRUCTOR_CAPTURE_IDENTITY', key)
handle = direct.model.encoder.register_forward_hook(lambda *args: None)
try:
    try:
        worker.no_hooks(direct.model)
    except ValueError:
        pass
    else:
        raise AssertionError('Injected observer accepted by no-hook guard')
finally:
    handle.remove()
worker.no_hooks(direct.model)
worker.selftest()
print('DIRECT_ORIGINAL_REAL_TINY_NEURAL_CONTROL_PASS_NO_REAL_DAC_OR_SPEECH_ACCEPTANCE')
