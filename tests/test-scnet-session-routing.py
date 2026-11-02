#!/usr/bin/env python3
"""Compile the production SCNet Session routing against a small native oracle.

This isolates wrapper ownership and dispatch from model inference: actual
crispasr_c_api.cpp blocks are compiled unchanged, with a deterministic backend
that records parameters and can fail. Actual-model/CLI tests are separate.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def source():
    text = (ROOT / "src/crispasr_c_api.cpp").read_text()
    blocks = re.findall(r"#ifdef CA_HAVE_SCNET\n.*?\n#endif", text, re.S)
    members = next(block for block in blocks if "scnet_context* scnet_ctx" in block)
    opening = next(block for block in blocks if 's->backend == "scnet"' in block)
    closing = next(block for block in reversed(blocks) if "scnet_free(s->scnet_ctx)" in block)
    start = text.index("CA_EXPORT int crispasr_session_separate(")
    end = text.index("// Pitch (F0) session API", start)
    separation = text[start:end]
    return r'''
#include "scnet.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#define CA_EXPORT extern "C"
#define CA_HAVE_SCNET 1
struct scnet_context { int id = 1; };
struct crispasr_session {
    std::string backend;
    int n_threads = 0;
''' + members + r'''
};
static bool g_open_use_gpu_tls = false;
static uint64_t g_open_seed_tls = 0;
static int selected_device = 0;
static int freed_results = 0, freed_contexts = 0, active_results = 0;
static int last_frames = 0, last_channels = 0;
static scnet_params last_params{};
static const char* names[] = {"drums", "bass", "other", "vocals"};
int crispasr_gpu_device_pref_storage() { return selected_device; }
extern "C" scnet_params scnet_default_params() {
    scnet_params p{};
    p.segment_seconds = 20; p.shifts = 1; p.shift_seed = 0;
    return p;
}
extern "C" scnet_context* scnet_init_from_file(const char* path, scnet_params p) {
    last_params = p;
    return path && std::strcmp(path, "valid") == 0 ? new scnet_context : nullptr;
}
extern "C" void scnet_free(scnet_context* ctx) {
    if (ctx) { ++freed_contexts; delete ctx; }
}
extern "C" int scnet_sample_rate(const scnet_context* ctx) { return ctx ? 44100 : 0; }
extern "C" scnet_result* scnet_separate(scnet_context* ctx, const float* pcm, int n, int channels) {
    last_frames = n; last_channels = channels;
    if (!ctx || !pcm || n == 17) return nullptr;
    auto* r = new scnet_result{};
    r->n_sources = 4; r->n_channels = 2; r->n_samples = n; r->sample_rate = 44100;
    r->source_names = names; r->sources = new float*[4];
    for (int stem = 0; stem < 4; ++stem) {
        r->sources[stem] = new float[2 * n];
        for (int i = 0; i < 2 * n; ++i) r->sources[stem][i] = pcm[i] + stem;
    }
    ++active_results;
    return r;
}
extern "C" void scnet_result_free(scnet_result* r) {
    if (!r) return;
    for (int stem = 0; stem < r->n_sources; ++stem) delete[] r->sources[stem];
    delete[] r->sources; delete r;
    ++freed_results; --active_results;
}
extern "C" crispasr_session* probe_open(const char* model_path, const char* backend,
                                       int threads, bool gpu, int device, uint64_t seed) {
    auto* s = new crispasr_session;
    s->backend = backend; s->n_threads = threads;
    g_open_use_gpu_tls = gpu; selected_device = device; g_open_seed_tls = seed;
''' + opening + r'''
    delete s; return nullptr;
}
extern "C" void probe_close(crispasr_session* s) {
    if (!s) return;
''' + closing + r'''
    delete s;
}
extern "C" const char* probe_backend(crispasr_session* s) { return s->backend.c_str(); }
extern "C" int probe_value(int field) {
    switch (field) {
    case 0: return last_params.n_threads;
    case 1: return last_params.use_gpu;
    case 2: return last_params.gpu_device;
    case 3: return last_params.segment_seconds;
    case 4: return last_params.shifts;
    case 5: return last_params.no_segment;
    case 6: return last_frames;
    case 7: return last_channels;
    case 8: return freed_results;
    case 9: return freed_contexts;
    case 10: return active_results;
    default: return -999;
    }
}
extern "C" uint64_t probe_seed() { return last_params.shift_seed; }
''' + separation


class SCNetSessionRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
        if not compiler:
            raise RuntimeError("C++ compiler is required for production Session routing controls")
        cls.temp = tempfile.TemporaryDirectory(prefix="scnet-session-routing-")
        work = Path(cls.temp.name)
        cpp = work / "routing.cpp"
        cpp.write_text(source())
        binary = work / ("routing.dylib" if os.sys.platform == "darwin" else "routing.so")
        subprocess.run([compiler, "-std=c++17", "-O0", "-shared", "-fPIC", str(cpp),
                        "-I", str(ROOT / "src"), "-o", str(binary)], check=True,
                       capture_output=True, text=True, timeout=60)
        cls.lib = C.CDLL(str(binary))
        for name, args, result in [
            ("probe_open", [C.c_char_p, C.c_char_p, C.c_int, C.c_bool, C.c_int, C.c_uint64], C.c_void_p),
            ("probe_close", [C.c_void_p], None),
            ("probe_backend", [C.c_void_p], C.c_char_p),
            ("probe_value", [C.c_int], C.c_int),
            ("probe_seed", [], C.c_uint64),
            ("crispasr_session_separate", [C.c_void_p, C.POINTER(C.c_float), C.c_int], C.c_int),
            ("crispasr_session_separate_n_stems", [C.c_void_p], C.c_int),
            ("crispasr_session_separate_sample_rate", [C.c_void_p], C.c_int),
            ("crispasr_session_separate_stem_name", [C.c_void_p, C.c_int], C.c_char_p),
            ("crispasr_session_separate_stem", [C.c_void_p, C.c_int, C.POINTER(C.c_int)], C.POINTER(C.c_float)),
        ]:
            function = getattr(cls.lib, name)
            function.argtypes, function.restype = args, result

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def open(self, name=b"scnet", **kwargs):
        return self.lib.probe_open(kwargs.get("model", b"valid"), name, 3, True, 7,
                                   kwargs.get("seed", (1 << 40) + 123))

    def test_parameter_forwarding_and_alias(self):
        for name in (b"scnet", b"scnet-large"):
            session = self.open(name)
            self.assertTrue(session)
            try:
                self.assertEqual(self.lib.probe_backend(session), b"scnet")
                self.assertEqual([self.lib.probe_value(i) for i in range(6)], [3, 1, 7, 20, 1, 0])
                self.assertEqual(self.lib.probe_seed(), (1 << 40) + 123)
                self.assertEqual(self.lib.crispasr_session_separate_sample_rate(session), 44100)
            finally:
                self.lib.probe_close(session)
        self.assertFalse(self.open(model=b"missing"))
        self.assertFalse(self.open(b"unknown"))

    def test_interleaved_shape_replacement_failure_and_close(self):
        session = self.open()
        before = [self.lib.probe_value(i) for i in (8, 9)]
        audio = (C.c_float * 64)(*(i / 64 for i in range(64)))
        try:
            self.assertEqual(self.lib.crispasr_session_separate_n_stems(session), 0)
            for invocation in range(2):
                self.assertEqual(self.lib.crispasr_session_separate(session, audio, 32), 4)
                self.assertEqual([self.lib.probe_value(i) for i in (6, 7, 10)], [32, 2, 1])
                self.assertEqual(self.lib.probe_value(8), before[0] + invocation)
                for index, name in enumerate((b"drums", b"bass", b"other", b"vocals")):
                    self.assertEqual(self.lib.crispasr_session_separate_stem_name(session, index), name)
                    count = C.c_int(-1)
                    ptr = self.lib.crispasr_session_separate_stem(session, index, C.byref(count))
                    self.assertEqual(count.value, 32)
                    self.assertEqual(list(ptr[:64]), [sample + index for sample in audio])
                for invalid in (-1, 4):
                    self.assertIsNone(self.lib.crispasr_session_separate_stem_name(session, invalid))
                    count = C.c_int(999)
                    self.assertFalse(self.lib.crispasr_session_separate_stem(session, invalid, C.byref(count)))
                    self.assertEqual(count.value, 0)
            self.assertEqual(self.lib.crispasr_session_separate(session, None, 32), -1)
            self.assertEqual(self.lib.crispasr_session_separate(session, audio, 0), -1)
            self.assertEqual(self.lib.crispasr_session_separate_n_stems(session), 4)
            self.assertEqual(self.lib.crispasr_session_separate(session, audio, 17), -1)
            self.assertEqual(self.lib.crispasr_session_separate_n_stems(session), 0)
            self.assertEqual(self.lib.probe_value(10), 0)
            self.assertEqual(self.lib.crispasr_session_separate(session, audio, 32), 4)
        finally:
            self.lib.probe_close(session)
        self.assertEqual(self.lib.probe_value(8), before[0] + 3)
        self.assertEqual(self.lib.probe_value(9), before[1] + 1)
        self.assertEqual(self.lib.probe_value(10), 0)

    def test_null_session(self):
        self.assertEqual(self.lib.crispasr_session_separate(None, None, 0), -1)
        self.assertEqual(self.lib.crispasr_session_separate_n_stems(None), 0)
        self.assertEqual(self.lib.crispasr_session_separate_sample_rate(None), 0)
        self.assertIsNone(self.lib.crispasr_session_separate_stem_name(None, 0))


if __name__ == "__main__":
    unittest.main()
