"""Fail closed on SDK interposition and unresolved vendor libraries; no models."""
import ctypes
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('current_replay',
    ROOT / 'tools/kaggle/openvoice-current-replay/replay.py')
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class CurrentReplayContracts(unittest.TestCase):
    def test_embedded_session_phase_compiles(self):
        compile(REPLAY.SESSION_SCRIPT, 'session-child', 'exec')

    def test_unresolved_dependencies_fail_even_when_ldd_returns_zero(self):
        for output in ['libc.so.6 => /lib/libc.so.6',
                       'libcublas.so.12 => not found', 'undefined symbol: ggml_backend_cuda_init']:
            with self.subTest(output=output), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                def run(command, label):
                    (root / (label + '.log')).write_text(output)
                worker = SimpleNamespace(OUT=root, BUNDLE=root, run=Mock(side_effect=run))
                if '=>' in output and 'not found' not in output:
                    REPLAY.check_dependencies(worker)
                    self.assertEqual(worker.run.call_count, 2)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Unresolved host runtime'):
                        REPLAY.check_dependencies(worker)
                    self.assertEqual(worker.run.call_count, 1)

    def test_symbol_owner_rejects_other_library_and_accepts_actual_owner(self):
        # Real dladdr lookup, without loading model weights or compiling a stub.
        libc = ctypes.CDLL('libc.so.6')
        class DlInfo(ctypes.Structure):
            _fields_ = [('filename', ctypes.c_char_p), ('base', ctypes.c_void_p),
                        ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]
        dl = ctypes.CDLL('libdl.so.2')
        info = DlInfo()
        dl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        self.assertTrue(dl.dladdr(ctypes.cast(libc.puts, ctypes.c_void_p), ctypes.byref(info)))
        owner = Path(info.filename.decode()).resolve()
        fixture = SimpleNamespace(**{name: libc.puts for name in [
            'openvoice2_context_default_params', 'openvoice2_init_from_file',
            'openvoice2_convert', 'openvoice2_set_stage_callback']})
        real_cdll = ctypes.CDLL
        def load(name, *args, **kwargs):
            return real_cdll(name, *args, **kwargs) if name == 'libdl.so.2' else fixture
        with patch.object(REPLAY.ct, 'CDLL', side_effect=load):
            self.assertEqual(len(REPLAY.check_symbol_owners(owner)), 4)
            with self.assertRaisesRegex(RuntimeError, 'interposed'):
                REPLAY.check_symbol_owners(Path('/unused/libopenvoice-probe.so'))


if __name__ == '__main__':
    unittest.main()
