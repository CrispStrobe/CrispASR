"""Actual Python generic-speaker methods through the shared native POD oracle."""
import ctypes as C
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import SpeakerEmbedder, PyannoteCache, DiarizeSegment, agglomerative_cluster


@unittest.skipUnless(shutil.which('cc') and sys.platform != 'win32', 'POSIX C compiler required')
class SpeakerPrimitives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='python-speaker-')
        directory = Path(cls.tmp.name)
        runtime = (ROOT / 'src/crispasr_c_api.cpp').read_text()
        layouts = []
        for name in ('crispasr_diarize_seg_abi', 'crispasr_diarize_turn_abi', 'crispasr_diarize_opts_abi'):
            start = runtime.index('struct ' + name + ' {')
            layouts.append(runtime[start:runtime.index('};', start) + 2])
        (directory / 'runtime-abi-layout.h').write_text('\n'.join(layouts))
        cls.library = str(directory / 'probe.so')
        subprocess.run(['cc', '-std=c11', '-Werror', '-fPIC', '-shared', '-I' + str(ROOT / 'include'),
                        '-I' + str(directory), str(ROOT / 'tests/fixtures/wrapper-speaker-primitives.c'),
                        '-o', cls.library], check=True)
        cls.lib = C.CDLL(cls.library)
        cls.lib.probe_speaker_mode.argtypes = [C.c_int]
        cls.lib.probe_speaker_live.restype = C.c_int

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.lib.probe_speaker_mode(0)
        self.resources = []

    def tearDown(self):
        for resource in self.resources:
            resource.close()
        self.assertEqual(self.lib.probe_speaker_live(), 0)

    def embedder(self):
        value = SpeakerEmbedder('embed', 2, 'cache', lib_path=self.library)
        self.resources.append(value)
        return value

    def cache(self):
        value = PyannoteCache([1, 3, 5], 'seg', 2, lib_path=self.library)
        self.resources.append(value)
        return value

    def cluster(self):
        return agglomerative_cluster([[1, 2, 3], [4, 5, 6]], merge_threshold=.6,
                                     max_speakers=3, lib_path=self.library)

    def test_owned_embedding_exact_success_and_failure_codes(self):
        model = self.embedder()
        self.assertEqual((model.dim, model.name), (3, 'generic'))
        output = model.embed(np.arange(1, 7, dtype=np.float32)[::2])
        np.testing.assert_array_equal(output, [.25, .5, .75])
        for mode in (2, 3, 4, 8):
            self.lib.probe_speaker_mode(mode)
            self.assertIsNone(model.embed([1, 3, 5]), str(mode))
        model.close()
        np.testing.assert_array_equal(output, [.25, .5, .75])

    def test_mono_shape_and_closed_guards(self):
        model = self.embedder()
        with self.assertRaises(ValueError):
            model.embed([[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        model.close()
        for call in (lambda: model.dim, lambda: model.name, lambda: model.embed([1, 3, 5])):
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                call()
        model.close()

    def test_array_coercion_cannot_reenter_a_closed_embedder(self):
        model = self.embedder()
        class ClosesModel:
            def __array__(self, dtype=None, copy=None):
                model.close()
                return np.array([1, 3, 5], dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            model.embed(ClosesModel())

    def test_cluster_counts_labels_and_unassigned(self):
        np.testing.assert_array_equal(self.cluster(), [0, 1])
        for mode in (2, 6, 7):
            self.lib.probe_speaker_mode(mode)
            with self.assertRaises(RuntimeError):
                self.cluster()
        self.lib.probe_speaker_mode(9)
        np.testing.assert_array_equal(self.cluster(), [-1, -1])

    def test_cluster_invalid_shape_and_options(self):
        for data in ([1, 2, 3], np.zeros((2, 2, 2))):
            with self.assertRaises(ValueError):
                agglomerative_cluster(data, lib_path=self.library)
        for threshold in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                agglomerative_cluster([[1, 2, 3]], merge_threshold=threshold, lib_path=self.library)
        with self.assertRaises(ValueError):
            agglomerative_cluster([[1, 2, 3]], max_speakers=2**32 + 3, lib_path=self.library)
        self.assertEqual(agglomerative_cluster(np.empty((0, 3)), lib_path=self.library).size, 0)

    def test_cluster_bounds_precede_dtype_and_contiguous_copies(self):
        before = self.lib.probe_cluster_calls()
        for dtype in (np.float32, np.float64):
            for shape in ((2**31, 1), (1, 2**31), (65536, 65536)):
                matrix = np.broadcast_to(np.array([[1]], dtype=dtype), shape)
                with self.subTest(dtype=dtype, shape=shape):
                    def no_dtype_copy(value, *args, **kwargs):
                        if args or kwargs.get('dtype') is not None:
                            raise AssertionError('unexpected dtype copy')
                        return original_asarray(value)
                    original_asarray = np.asarray
                    with patch.object(np, 'asarray', side_effect=no_dtype_copy) as coercion:
                        with patch.object(np, 'ascontiguousarray', side_effect=AssertionError('unexpected copy')):
                            with self.assertRaises(ValueError):
                                agglomerative_cluster(matrix, lib_path=self.library)
                        self.assertTrue(all(not call.kwargs.get('dtype') for call in coercion.call_args_list))
        self.assertEqual(self.lib.probe_cluster_calls(), before)
        for matrix in (np.array([[1, 0, 2, 0, 3, 0], [4, 0, 5, 0, 6, 0]],
                                dtype=np.float64)[:, ::2],
                       np.array([[6, 5, 4], [3, 2, 1]], dtype=np.float32)[::-1, ::-1]):
            np.testing.assert_array_equal(agglomerative_cluster(matrix, merge_threshold=.6,
                                           max_speakers=3, lib_path=self.library), [0, 1])

    def test_cluster_zero_dimension_rejected_and_zero_rows_preserved(self):
        before = self.lib.probe_cluster_calls()
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                with patch.object(np, 'ascontiguousarray', side_effect=AssertionError('unexpected copy')):
                    with self.assertRaises(ValueError):
                        agglomerative_cluster(np.empty((3, 0), dtype=dtype), lib_path=self.library)
                    for shape in ((0, 3), (0, 0)):
                        labels = agglomerative_cluster(np.empty(shape, dtype=dtype), lib_path=self.library)
                        self.assertEqual(labels.shape, (0,))
                        self.assertEqual(labels.dtype, np.int32)
        self.assertEqual(self.lib.probe_cluster_calls(), before)

    def test_constructor_rejects_truncated_strings_and_counts(self):
        for model, cache, threads in [('embed\0bad', 'cache', 2), ('embed', 'cache\0bad', 2),
                                      ('embed', 'cache', 2**32 + 2)]:
            with self.assertRaises(ValueError):
                value = SpeakerEmbedder(model, threads, cache, lib_path=self.library)
                self.resources.append(value)
        with self.assertRaises(ValueError):
            value = PyannoteCache([1, 3, 5], 'seg\0bad', 2, lib_path=self.library)
            self.resources.append(value)

    def test_cache_transport_failures_and_close(self):
        cache = self.cache()
        segments = [DiarizeSegment(1.25, 1.75, -1), DiarizeSegment(2., 2.5, -1)]
        cache.apply(segments, 1.25)
        self.assertEqual([s.speaker for s in segments], [2, 1])
        for mode in (2, 5):
            self.lib.probe_speaker_mode(mode)
            with self.assertRaises(RuntimeError):
                cache.apply(segments, 1.25)
            self.assertEqual([s.speaker for s in segments], [2, 1])
        cache.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            cache.apply(segments, 1.25)


if __name__ == '__main__':
    unittest.main()
