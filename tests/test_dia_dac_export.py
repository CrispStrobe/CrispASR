#!/usr/bin/env python3
"""Physical mixed-precision exports from tiny actual DAC original modules."""
import os
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import test_dia_codec_parameters as fixtures
from dia_dac_export import CHECKPOINT_SHA, OUTPUT_NAMES, export_precision, mapped_gguf


class ConverterCLITest(unittest.TestCase):
    def test_wrong_template_fails_before_loading_model_dependencies(self):
        path = Path(__file__).resolve().parents[1] / 'models/convert-dia-dac-to-gguf.py'
        spec = importlib.util.spec_from_file_location('dia_dac_converter', path)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            template, output, receipt = [Path(directory) / name for name in ('wrong.gguf', 'output.gguf', 'audit.json')]
            template.write_bytes(b'not the pinned published DAC template')
            argv = [str(path), '--template', str(template), '--output', str(output),
                    '--precision', 'output-f32', '--receipt', str(receipt)]
            with mock.patch.object(sys, 'argv', argv), mock.patch.dict(sys.modules, {'torch': None, 'dac': None}):
                with self.assertRaises(SystemExit) as caught:
                    converter.main()
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(output.exists())
            self.assertFalse(receipt.exists())


class CodecExportTest(fixtures.CodecParameterTest):
    def template(self, path, unknown=False):
        from gguf import GGUFWriter
        writer = GGUFWriter(str(path), 'dac-44khz')
        names = sorted(OUTPUT_NAMES) + ['quantizer.quantizers.0.codebook.weight']
        for name in names:
            writer.add_tensor(name, self.parameter(self.codec, name).detach().numpy().astype(np.float16))
        if unknown:
            writer.add_tensor('decoder.unknown.weight', np.ones((2, 2), dtype=np.float16))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_output_preserves_original_coefficients_and_unrelated_bytes(self):
        from gguf import GGMLQuantizationType
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            template, output = Path(directory) / 'published.gguf', Path(directory) / 'output.gguf'
            self.template(template)
            result = export_precision(self.codec, template, output)
            with mapped_gguf(template) as old_reader, mapped_gguf(output) as new_reader:
                old = {tensor.name: tensor for tensor in old_reader.tensors}
                new = {tensor.name: tensor for tensor in new_reader.tensors}
                self.assertEqual(new.keys(), old.keys())
                for name in OUTPUT_NAMES:
                    self.assertEqual(new[name].tensor_type, GGMLQuantizationType.F32)
                    original = self.parameter(self.codec, name).detach().numpy()
                    np.testing.assert_array_equal(new[name].data, original)
                untouched = 'quantizer.quantizers.0.codebook.weight'
                self.assertEqual(new[untouched].tensor_type, GGMLQuantizationType.F16)
                self.assertEqual(new[untouched].data.tobytes(), old[untouched].data.tobytes())
                self.assertEqual(sum(row['promoted'] for row in result['tensors']), 3)
                self.assertFalse(result['speech_acceptance'])
                self.assertFalse(result['production_defaults_changed'])

    def test_f32_control_uses_actual_original_parameters(self):
        from gguf import GGMLQuantizationType
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            template, output = Path(directory) / 'published.gguf', Path(directory) / 'all-f32.gguf'
            self.template(template)
            export_precision(self.codec, template, output, 'f32')
            with mapped_gguf(output) as reader:
                for tensor in reader.tensors:
                    self.assertEqual(tensor.tensor_type, GGMLQuantizationType.F32)
                    np.testing.assert_array_equal(tensor.data, self.parameter(self.codec, tensor.name).detach().numpy())

    def test_unknown_tensor_bad_pin_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            template, output = Path(directory) / 'published.gguf', Path(directory) / 'output.gguf'
            self.template(template, unknown=True)
            with self.assertRaisesRegex(ValueError, 'Unmapped published DAC tensor'):
                export_precision(self.codec, template, output)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                export_precision(self.codec, template, output, checkpoint_sha='0' * len(CHECKPOINT_SHA))
            with self.assertRaisesRegex(ValueError, 'new output path'):
                export_precision(self.codec, template, template)


if __name__ == '__main__':
    unittest.main()
