#!/usr/bin/env python3
"""Physical tiny original-module precision export contracts, no model proof."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from dia_dac_export import mapped_gguf
from dia_model_precision import parameter_catalog, original_array, export_precision, original_timescales


class NeuralPrecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_dia_reference_capture import DiaCaptureTest
        DiaCaptureTest.setUpClass()
        cls.parameters = {name: value.detach().numpy().copy()
                          for name, value in DiaCaptureTest.dia.model.named_parameters()}
        cls.catalog = parameter_catalog(cls.parameters)
        cls.originals = {name: original_array(name, cls.catalog, cls.parameters.__getitem__)
                         for name in cls.catalog}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.template = self.root / 'tiny.gguf'
        from gguf import GGUFWriter
        writer = GGUFWriter(str(self.template), 'dia')
        writer.add_uint32('dia.decoder.output_heads', 9)
        writer.add_uint32('dia.attn_head_size', 4)
        writer.add_array('test.metadata', [1, 2, 3])
        for name, original in self.originals.items():
            writer.add_tensor(name, original.astype(np.float16 if original.ndim > 1 else np.float32))
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()

    def export(self, precision, getter=None, output=None, **kwargs):
        return export_precision(self.parameters, getter or self.parameters.__getitem__, self.template,
                                output or self.root / 'result.gguf', precision, **kwargs)

    def test_heads_are_original_f32_and_other_payloads_unchanged(self):
        result = self.export('heads-f32')
        selected = [row for row in result['tensors'] if row['selected']]
        self.assertEqual(len(selected), 9)
        with mapped_gguf(self.template) as old, mapped_gguf(self.root / 'result.gguf') as new:
            for left, right in zip(old.tensors, new.tensors):
                if right.name.startswith('dia.decoder.heads.'):
                    self.assertEqual(right.tensor_type.name, 'F32')
                    np.testing.assert_array_equal(right.data, self.originals[right.name])
                else:
                    self.assertEqual(left.data.tobytes(), right.data.tobytes())
                    self.assertEqual(left.tensor_type, right.tensor_type)

    def test_all_f32_preserves_actual_original_coefficients_and_metadata(self):
        self.export('f32')
        with mapped_gguf(self.root / 'result.gguf') as new:
            self.assertEqual(new.fields['test.metadata'].contents(), [1, 2, 3])
            for tensor in new.tensors:
                self.assertEqual(tensor.tensor_type.name, 'F32')
                np.testing.assert_array_equal(tensor.data, self.originals[tensor.name])

    def test_bad_checkpoint_unknown_mapping_and_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            self.export('f32', checkpoint_sha='bad')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            parameter_catalog(['bad.weight'])
        with self.assertRaisesRegex(ValueError, 'new output'):
            self.export('f32', output=self.template)
        with self.assertRaisesRegex(ValueError, 'Unknown neural precision'):
            self.export('bad')

    def test_changed_original_coefficients_rejected(self):
        key = next(iter(self.parameters))
        def changed(name):
            array = self.parameters[name].copy()
            if name == key: array.flat[0] += 1
            return array
        with self.assertRaisesRegex(ValueError, 'not rounded original'):
            self.export('f32', getter=changed)

    def test_nonfinite_original_rejected(self):
        def bad(name):
            array = self.parameters[name].copy(); array.flat[0] = np.nan; return array
        with self.assertRaisesRegex(ValueError, 'finite F32'):
            self.export('f32', getter=bad)

    def test_actual_original_timescale_is_auxiliary_and_existing_tensors_are_identical(self):
        scale, meta = original_timescales(os.environ['DIA_SOURCE'], 4)
        baseline = self.root / 'baseline.gguf'; candidate = self.root / 'candidate.gguf'
        self.export('f32', output=baseline)
        audit = self.export('f32', output=candidate, timescales=scale, timescale_metadata=meta)
        self.assertEqual(len(audit['auxiliary_tensors']), 1)
        with mapped_gguf(baseline) as old, mapped_gguf(candidate) as new:
            self.assertEqual(len(new.tensors), len(old.tensors) + 1)
            for left, right in zip(old.tensors, new.tensors):
                self.assertEqual(left.name, right.name)
                self.assertEqual(left.data.tobytes(), right.data.tobytes())
                self.assertEqual(left.tensor_type, right.tensor_type)
            np.testing.assert_array_equal(new.tensors[-1].data, scale)
            self.assertEqual(new.tensors[-1].tensor_type.name, 'F32')
            self.assertEqual(new.tensors[-1].name, 'dia.rope.timescale')

    def test_bad_source_buffer_or_dimension_is_rejected(self):
        scale, meta = original_timescales(os.environ['DIA_SOURCE'], 4)
        for bad in (scale.astype('f2'), scale * np.nan, scale * -1):
            with self.subTest(dtype=bad.dtype):
                with self.assertRaisesRegex(ValueError, 'timescale'):
                    self.export('f32', timescales=bad, timescale_metadata=meta)
        with self.assertRaisesRegex(ValueError, 'dimension'):
            scale, meta = original_timescales(os.environ['DIA_SOURCE'], 8)
            self.export('f32', timescales=scale, timescale_metadata=meta)


if __name__ == '__main__':
    unittest.main()
