import ctypes
import errno
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import crispasr._binding as binding  # noqa: E402


class _Callable:
    def __init__(self, fn):
        self.fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.fn(*args)


class PythonDiarizeTurnsTest(unittest.TestCase):
    def test_abi_layout_and_capacity_retry(self):
        self.assertEqual(ctypes.sizeof(binding._DiarizeSegAbi), 24)
        self.assertEqual(ctypes.sizeof(binding._DiarizeTurnAbi), 24)
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self.assertEqual(ctypes.sizeof(binding._DiarizeOptsAbi), 48)
            self.assertEqual(binding._DiarizeOptsAbi.foxnose_embedder_path.offset, 24)

        calls = 0

        def old_call(*_args):
            self.fail("the turn-returning wrapper called the legacy symbol")

        def turns_call(_left, _right, _n, _stereo, segs_ptr, n_segs,
                       opts_ptr, turns_ptr, turn_cap, out_n_ptr):
            nonlocal calls
            calls += 1
            opts = ctypes.cast(
                opts_ptr, ctypes.POINTER(binding._DiarizeOptsAbi)
            ).contents
            self.assertEqual(opts.method, binding.DiarizeMethod.FOXNOSE)
            self.assertEqual(opts.foxnose_embedder_path, b"wespeaker.gguf")
            self.assertEqual((opts.min_speakers, opts.max_speakers,
                              opts.num_speakers), (2, 4, 3))

            out_n = ctypes.cast(out_n_ptr, ctypes.POINTER(ctypes.c_int32))
            if calls == 1:
                # Force the wrapper through the ABI's size-and-retry protocol.
                self.assertGreater(turn_cap, 0)
                out_n.contents.value = turn_cap + 2
                return 2

            out_n.contents.value = 2
            segs = ctypes.cast(
                segs_ptr, ctypes.POINTER(binding._DiarizeSegAbi)
            )
            for i in range(n_segs):
                segs[i].speaker = i + 1
            turns = ctypes.cast(
                turns_ptr, ctypes.POINTER(binding._DiarizeTurnAbi)
            )
            turns[0].t0_cs, turns[0].t1_cs, turns[0].speaker = 125, 250, 1
            turns[1].t0_cs, turns[1].t1_cs, turns[1].speaker = 250, 375, 2
            return 0

        fake_lib = type("FakeLib", (), {})()
        fake_lib.crispasr_diarize_segments_abi = _Callable(old_call)
        fake_lib.crispasr_diarize_segments_turns_abi = _Callable(turns_call)

        segs = [
            binding.DiarizeSegment(1.0, 2.5),
            binding.DiarizeSegment(2.5, 4.0),
        ]
        with mock.patch.object(binding.ctypes, "CDLL", return_value=fake_lib):
            ok, turns = binding.diarize_segments_with_turns(
                segs,
                np.zeros(16000, dtype=np.float32),
                method=binding.DiarizeMethod.FOXNOSE,
                foxnose_embedder_path="wespeaker.gguf",
                min_speakers=2,
                max_speakers=4,
                num_speakers=3,
                lib_path="/fake/libcrispasr.so",
            )

        self.assertTrue(ok)
        self.assertEqual(calls, 2)
        self.assertEqual([s.speaker for s in segs], [1, 2])
        self.assertEqual(
            [(t.t0, t.t1, t.speaker) for t in turns],
            [(1.25, 2.5, 1), (2.5, 3.75, 2)],
        )


class AuxiliaryPcmTransportTest(unittest.TestCase):
    """Read the actual pointers in native code, without loading any model."""

    @classmethod
    def setUpClass(cls):
        if not sys.platform.startswith("linux"):
            raise unittest.SkipTest("native recording fixture currently supports Linux")
        compiler = shutil.which(os.environ.get("CC", "cc"))
        if compiler is None:
            raise unittest.SkipTest("native recording fixture requires a C compiler")
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name)
        source = path / "pcm.c"
        source.write_text(r'''
#include <stdint.h>
#include <string.h>
static float recorded[2][32];
static int calls;
static int model_loaded, last_loaded;
int probe_calls(void) { return calls; }
int probe_last_loaded(void) { return last_loaded; }
float probe_sample(int channel, int sample) { return recorded[channel][sample]; }
static void record(const float* l, const float* r, int n) {
    ++calls;
    if (n > 32) n = 32;
    memcpy(recorded[0], l, n * sizeof(float));
    memcpy(recorded[1], r, n * sizeof(float));
}
int crispasr_detect_language_pcm(const float* pcm, int32_t n, int32_t method,
    const char* model, int32_t threads, int32_t gpu, int32_t device,
    int32_t flash, char* lang, int32_t cap, float* confidence) {
    record(pcm, pcm, n);
    if (cap >= 3) memcpy(lang, "en", 3);
    *confidence = 0.75f;
    return 0;
}
int crispasr_diarize_segments_abi(const float* l, const float* r, int32_t n,
    int32_t stereo, void* segs, int32_t nsegs, void* opts) {
    record(l, r, n);
    return 0;
}
int crispasr_diarize_segments_turns_abi(const float* l, const float* r, int32_t n,
    int32_t stereo, void* segs, int32_t nsegs, void* opts,
    void* turns, int32_t cap, int32_t* count) {
    record(l, r, n);
    *count = 0;
    return 0;
}
void crispasr_watermark_embed(float* pcm, int n, float alpha) {
    last_loaded = model_loaded;
    record(pcm, pcm, n);
    for (int i = 0; i < n; ++i) pcm[i] += 0.25f;
}
float crispasr_watermark_detect(const float* pcm, int n) {
    last_loaded = model_loaded;
    record(pcm, pcm, n);
    return 0.75f;
}
int crispasr_watermark_load_model(const char* path) {
    ++calls;
    if (strcmp(path, "unused.gguf") != 0) return -1;
    model_loaded = 1;
    return 0;
}
''')
        cls.lib_path = str(path / "pcm.so")
        subprocess.run([compiler, "-shared", "-fPIC",
                        str(source), "-o", cls.lib_path], check=True)
        cls.library_fd = None
        compiled_path = cls.lib_path
        if hasattr(os, "memfd_create") and Path("/proc/self/fd").is_dir():
            # Keep dlopen's mapping off a network-backed TMPDIR: CIFS leaves
            # delete-pending entries for mapped shared objects until exit.
            fd = None
            try:
                fd = os.memfd_create("crispasr-pcm-recorder", os.MFD_CLOEXEC)
                with os.fdopen(os.dup(fd), "wb") as output:
                    output.write(Path(compiled_path).read_bytes())
                cls.lib_path = "/proc/self/fd/{}".format(fd)
                cls.lib = ctypes.CDLL(cls.lib_path)
                cls.library_fd = fd
            except OSError:
                # Restricted Linux hosts may expose the API but deny the
                # syscall, proc lookup or executable mapping. Use the file.
                if fd is not None:
                    os.close(fd)
                cls.lib_path = compiled_path
        if cls.library_fd is None:
            cls.lib = ctypes.CDLL(cls.lib_path)
        cls.lib.probe_sample.argtypes = [ctypes.c_int, ctypes.c_int]
        cls.lib.probe_sample.restype = ctypes.c_float

    @classmethod
    def tearDownClass(cls):
        if cls.library_fd is not None:
            os.close(cls.library_fd)
            cls.library_fd = None
        # CIFS can acknowledge unlink before its directory entry disappears.
        # Retry only that transient failure; preserve other cleanup errors.
        for attempt in range(5):
            try:
                cls.tmp.cleanup()
                return
            except OSError as exc:
                if exc.errno != errno.ENOTEMPTY or attempt == 4:
                    raise
                time.sleep(0.1 * (2 ** attempt))

    def assert_recorded(self, expected, channel=0):
        got = [self.lib.probe_sample(channel, i) for i in range(len(expected))]
        np.testing.assert_array_equal(got, expected)

    def test_lid_contiguous_and_strided_pcm(self):
        backing = np.arange(32, dtype=np.float32)
        for pcm in (backing[:4], backing[:8:2], backing[6:2:-1],
                    backing[:4].astype(np.float64)):
            with self.subTest(strides=pcm.strides, dtype=pcm.dtype):
                result = binding.detect_language_pcm(
                    pcm, model_path="unused.gguf", lib_path=self.lib_path)
                self.assertEqual((result.lang_code, result.confidence), ("en", 0.75))
                self.assert_recorded(pcm)

    def test_diarize_contiguous_and_strided_channels(self):
        backing = np.arange(32, dtype=np.float32)
        for fn in (binding.diarize_segments, binding.diarize_segments_with_turns):
            for left, right in ((backing[:4], backing[4:8]),
                                (backing[:8:2], backing[12:8:-1])):
                with self.subTest(function=fn.__name__, strides=left.strides):
                    fn([binding.DiarizeSegment(0, 1)], left, right=right,
                       is_stereo=True, lib_path=self.lib_path)
                    self.assert_recorded(left)
                    self.assert_recorded(right, 1)
            fn([binding.DiarizeSegment(0, 1)], backing[:8:2], lib_path=self.lib_path)
            self.assert_recorded(backing[:8:2], 1)

    def test_invalid_dimensions_and_stereo_lengths_do_not_dispatch(self):
        before = self.lib.probe_calls()
        with self.assertRaises(ValueError):
            binding.detect_language_pcm(np.zeros((2, 2)), model_path="unused.gguf",
                                        lib_path=self.lib_path)
        for fn in (binding.diarize_segments, binding.diarize_segments_with_turns):
            for left, right in ((np.zeros((2, 2)), None),
                                (np.zeros(4), np.zeros((2, 2))),
                                (np.zeros(4), np.zeros(3)),
                                (np.zeros(4), np.zeros(5)),
                                (np.zeros(4), np.zeros(0))):
                with self.subTest(function=fn.__name__, left=left.shape,
                                  right=None if right is None else right.shape):
                    with self.assertRaises(ValueError):
                        fn([binding.DiarizeSegment(0, 1)], left, right=right,
                           is_stereo=True, lib_path=self.lib_path)
        self.assertEqual(self.lib.probe_calls(), before)

    def test_watermark_views_and_in_place_copyback(self):
        for selection in (slice(0, 4), slice(0, 8, 2), slice(6, 2, -1)):
            with self.subTest(selection=selection):
                backing = np.arange(32, dtype=np.float32)
                pcm = backing[selection]
                expected = backing.copy()
                expected[selection] += 0.25
                with mock.patch.object(binding, "_find_lib", return_value=self.lib_path):
                    self.assertEqual(binding.watermark_detect(pcm), 0.75)
                    self.assert_recorded(pcm)
                    binding.watermark_embed(pcm)
                np.testing.assert_array_equal(backing, expected)

    def test_watermark_model_loading(self):
        before = self.lib.probe_calls()
        with mock.patch.object(binding, "_find_lib", return_value=self.lib_path):
            binding.watermark_load_model("unused.gguf")
            pcm = np.zeros(4, dtype=np.float32)
            binding.watermark_embed(pcm)
            self.assertEqual(self.lib.probe_last_loaded(), 1)
            self.assertEqual(binding.watermark_detect(pcm), 0.75)
            self.assertEqual(self.lib.probe_last_loaded(), 1)
            with self.assertRaises(RuntimeError):
                binding.watermark_load_model("missing.gguf")
        self.assertEqual(self.lib.probe_calls(), before + 4)

    def test_watermark_invalid_input_does_not_dispatch(self):
        before = self.lib.probe_calls()
        readonly = np.zeros(4, dtype=np.float32)
        readonly.flags.writeable = False
        with mock.patch.object(binding, "_find_lib", return_value=self.lib_path):
            with self.assertRaises(ValueError):
                binding.watermark_embed(readonly)
            for fn in (binding.watermark_embed, binding.watermark_detect):
                with self.assertRaises(ValueError):
                    fn(np.zeros((2, 2), dtype=np.float32))
                with self.assertRaises(TypeError):
                    fn(np.zeros(4, dtype=np.float64))
        self.assertEqual(self.lib.probe_calls(), before)
        with mock.patch.object(binding, "_find_lib", return_value=self.lib_path):
            self.assertEqual(binding.watermark_detect(readonly), 0.75)


if __name__ == "__main__":
    unittest.main()
