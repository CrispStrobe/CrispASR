"""Public reference CLI routing and GGUF serialization, without model weights.

The endpoint is mocked only here; test_dia_reference_capture.py separately runs
actual original Dia classes. This suite never claims neural/speech parity.
"""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import dump_reference as dispatcher


class DiaReferenceDispatch(unittest.TestCase):
    def test_public_backend_list(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/dump_reference.py'),
                                 '--list-backends'], capture_output=True, text=True, check=True)
        self.assertRegex(result.stdout, r'(?m)^  dia\s+-> tools/reference_backends/dia_tts_reference.py$')
        self.assertEqual(dispatcher._resolve_backend('dia').SOURCE_PIN,
                         '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8')

    def test_text_only_entrypoint_writes_real_archive_and_optional_captures(self):
        import gguf
        backend = dispatcher._resolve_backend('dia')
        with tempfile.TemporaryDirectory(prefix='dia-dispatch-') as temp:
            output = Path(temp) / 'ref.gguf'
            captures = dict(decoder_hidden=np.arange(8, dtype=np.float32).reshape(2, 4),
                            decoder_block17_input=np.arange(8, dtype=np.float32).reshape(2, 4),
                            dia_text='[S1] Test.', dia_source_pin=backend.SOURCE_PIN,
                            dia_boundary_capture_layers='17')
            argv = ['dump_reference.py', '--backend', 'dia', '--model-dir', temp,
                    '--output', str(output)]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(backend, 'dump', return_value=captures) as endpoint, \
                    mock.patch.object(dispatcher, 'load_audio_16k_mono', side_effect=AssertionError('unexpected WAV read')), \
                    contextlib.redirect_stdout(io.StringIO()):
                dispatcher.main()
            kwargs = endpoint.call_args.kwargs
            self.assertEqual(kwargs['audio'].shape, (0,))
            self.assertEqual(kwargs['audio'].dtype, np.dtype('float32'))
            self.assertEqual(kwargs['stages'], set(backend.DEFAULT_STAGES))
            reader = gguf.GGUFReader(str(output))
            try:
                tensors = {t.name: t for t in reader.tensors}
                np.testing.assert_array_equal(tensors['decoder_hidden'].data, captures['decoder_hidden'])
                self.assertIn('decoder_block17_input', tensors)
                self.assertEqual(reader.get_field('crispasr.ref.sample_rate').contents(), 0)
                # GGUFWriter omits empty strings rather than storing them.
                audio_field = reader.get_field('crispasr.ref.audio')
                self.assertTrue(audio_field is None or audio_field.contents() == '')
                self.assertEqual(reader.get_field('crispasr.ref.dia_boundary_capture_layers').contents(), '17')
                self.assertEqual(reader.get_field('crispasr.ref.dia_source_pin').contents(), backend.SOURCE_PIN)
            finally:
                reader.data._mmap.close()

    def test_audio_backends_still_require_wav(self):
        with mock.patch.object(sys, 'argv', ['dump_reference.py', '--backend', 'whisper',
                                           '--model-dir', '.', '--output', 'unused.gguf']), \
                mock.patch.object(dispatcher, '_resolve_backend', side_effect=AssertionError('must reject first')), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            dispatcher.main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
