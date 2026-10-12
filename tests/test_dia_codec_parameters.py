#!/usr/bin/env python3
"""Exercise diagnostic tensor lookup against tiny actual DAC 1.0.0 modules.

No checkpoint, audio tooling, reconstructed neural math, or native build. Only
original source classes and the shared export lookup are loaded; no heavy
command-line pipeline executes.
"""
import ast
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(ROOT / 'tools'))
from dia_dac_export import parameter


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CodecParameterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch
        distribution = importlib.metadata.distribution('descript-audio-codec')
        assert distribution.version == '1.0.0', distribution.version
        package = Path(distribution.locate_file('dac'))
        # Bypass DAC's unrelated audio/serialization imports. Layer and
        # quantizer modules and selected decoder classes are original source.
        dac, nn = types.ModuleType('dac'), types.ModuleType('dac.nn')
        dac.__path__, nn.__path__ = [str(package)], [str(package / 'nn')]
        with mock.patch.dict(sys.modules, {'dac': dac, 'dac.nn': nn}):
            layers = source_module('dac.nn.layers', package / 'nn/layers.py')
            quantize = source_module('dac.nn.quantize', package / 'nn/quantize.py')
            path = package / 'model/dac.py'
            tree = ast.parse(path.read_text())
            names = {'ResidualUnit', 'DecoderBlock', 'Decoder'}
            nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
            assert len(nodes) == len(names)
            namespace = {'nn': torch.nn, 'math': math, 'Snake1d': layers.Snake1d,
                         'WNConv1d': layers.WNConv1d, 'WNConvTranspose1d': layers.WNConvTranspose1d}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
            cls.quantizer = quantize.ResidualVectorQuantize(input_dim=8, n_codebooks=9,
                                                          codebook_size=16, codebook_dim=2).eval()
            cls.decoder = namespace['Decoder'](8, 32, [2, 2, 2, 2]).eval()
        cls.codec = types.SimpleNamespace(quantizer=cls.quantizer, decoder=cls.decoder)
        # Obtain actual effective Parameters using the original Torch utility,
        # as the heavy worker does after canonical PCM reproduction.
        for module in [*cls.quantizer.modules(), *cls.decoder.modules()]:
            if hasattr(module, 'weight_g'):
                torch.nn.utils.remove_weight_norm(module)
        cls.parameter = staticmethod(parameter)

    def test_complete_quantizer_and_decoder_parameter_identity(self):
        expected = {}
        for index, quantizer in enumerate(self.quantizer.quantizers):
            for name in ('codebook', 'in_proj', 'out_proj'):
                for suffix, value in getattr(quantizer, name).named_parameters(recurse=False):
                    expected[f'quantizer.quantizers.{index}.{name}.{suffix}'] = value
        for name, index in [('conv1', 0), ('snake1', 5), ('conv2', 6)]:
            for suffix, value in self.decoder.model[index].named_parameters(recurse=False):
                expected[f'decoder.{name}.{suffix}'] = value
        for index in range(4):
            block = self.decoder.model[index + 1].block
            for name, module_index in [('snake1', 0), ('conv_t1', 1)]:
                for suffix, value in block[module_index].named_parameters(recurse=False):
                    expected[f'decoder.block.{index}.{name}.{suffix}'] = value
            for residual in range(3):
                unit = block[residual + 2].block
                for name, module_index in [('snake1', 0), ('conv1', 1), ('snake2', 2), ('conv2', 3)]:
                    for suffix, value in unit[module_index].named_parameters(recurse=False):
                        expected[f'decoder.block.{index}.res_unit{residual + 1}.{name}.{suffix}'] = value
        self.assertEqual(len(expected), 134)
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertIs(self.parameter(self.codec, name), value)

    def test_input_projection_is_mapped_but_not_used_by_code_replay(self):
        torch = self.torch
        codes = torch.arange(27).reshape(1, 9, 3) % 16
        with torch.inference_mode():
            original = self.quantizer.from_codes(codes)[0].clone()
            parameter = self.parameter(self.codec, 'quantizer.quantizers.0.in_proj.bias')
            saved = parameter.clone()
            try:
                parameter.fill_(123)
                torch.testing.assert_close(self.quantizer.from_codes(codes)[0], original, rtol=0, atol=0)
            finally:
                parameter.copy_(saved)

    def test_unknown_tensor_is_rejected(self):
        for name in ('quantizer.quantizers.0.invented.bias', 'decoder.block.0.res_unit4.conv1.weight',
                     'decoder.invented.weight', 'encoder.conv1.weight'):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'Unmapped published DAC tensor'):
                    self.parameter(self.codec, name)


if __name__ == '__main__':
    unittest.main()
