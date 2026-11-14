#!/usr/bin/env python3
"""Weight-free configuration and actual production DPM scheduler controls."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DiffusionConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.temp = tempfile.TemporaryDirectory(prefix="vibevoice-diffusion-")
        root = Path(cls.temp.name)
        native = (ROOT / "src/vibevoice.cpp").read_text()
        # Compile the production scheduler/solver, not a rewritten test version.
        solver = native[native.index("struct ddim_schedule {"):native.index("static ggml_cgraph* build_pred_head_graph_impl")]
        source = root / "probe.cpp"
        source.write_text('''#include "core/vibevoice_tts_generation.h"
#include <vector>
#include <cmath>
''' + solver + '''
extern "C" int steps(int n) { return core_vibevoice_tts::diffusion_steps(n); }
extern "C" float cfg(bool base, float value, const char* env) {
 return core_vibevoice_tts::guidance_scale(base,value,env);
}
extern "C" void schedule(int n,int* out) {
 auto s=make_ddim_schedule(n);for(int i=0;i<n;++i)out[i]=s.timesteps[i];
}
extern "C" float solve(int n,float scale) {
 auto s=make_ddim_schedule(n);float z=.25f,prev=0;
 for(int i=0;i<n;++i){float v=.15f+scale*(.3f-.15f),x0;
 v_to_x0(&z,&v,&x0,1,s.alphas_cumprod[s.timesteps[i]]);
 if(i==0 || i==n-1)dpm_first_order(s,i,&z,&x0,1);
 else dpm_second_order(s,i,&z,&x0,&prev,i-1,1);
 prev=x0;
 }return z;
}
''')
        lib = root / "probe.so"
        subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-I", str(ROOT / "src"),
                        str(source), "-o", str(lib)], check=True, capture_output=True)
        cls.lib = ctypes.CDLL(str(lib))
        cls.lib.steps.argtypes = [ctypes.c_int]
        cls.lib.cfg.argtypes = [ctypes.c_bool, ctypes.c_float, ctypes.c_char_p]
        cls.lib.cfg.restype = ctypes.c_float
        cls.lib.schedule.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        cls.lib.solve.argtypes = [ctypes.c_int, ctypes.c_float]
        cls.lib.solve.restype = ctypes.c_float

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_defaults_preserve_old_solver_bytes(self):
        for value in (0, -1, 20):
            self.assertEqual(self.lib.steps(value), 20)
            self.assertEqual(self.lib.cfg(True, 0., None), ctypes.c_float(1.3).value)
            old = self.lib.solve(20, 1.3)
            new = self.lib.solve(self.lib.steps(value), self.lib.cfg(True, 0., None))
            self.assertEqual(bytes(ctypes.c_float(old)), bytes(ctypes.c_float(new)))
        self.assertEqual(self.lib.cfg(False, 0., None), 3.)

    def test_explicit_cfg_overrides_environment_and_sentinel(self):
        self.assertEqual(self.lib.cfg(True, 2., b"4.5"), 2.)
        self.assertEqual(self.lib.cfg(True, 0., b"4.5"), 4.5)
        for env in (None, b"0", b"-1", b"invalid"):
            self.assertEqual(self.lib.cfg(True, 0., env), ctypes.c_float(1.3).value)
        self.assertNotEqual(self.lib.solve(20, 1.3), self.lib.solve(20, 2.))

    def test_configured_schedule_matches_upstream_linspace_recipe(self):
        # Upstream DPMSolverMultistepScheduler timestep_spacing='linspace'.
        for n in (4, 8, 20, 40, 100):
            self.assertEqual(self.lib.steps(n), n)
            out = (ctypes.c_int * n)()
            self.lib.schedule(n, out)
            expected = [round(999 * (n-i) / n) for i in range(n)]
            self.assertEqual(list(out), expected)
            self.assertNotEqual(out[-1], 0)  # final sigma zero is the last solver update
        self.assertNotEqual(self.lib.solve(4, 1.3), self.lib.solve(20, 1.3))

    def test_base_and_realtime_consume_resolved_controls(self):
        native = (ROOT / "src/vibevoice.cpp").read_text()
        base = native[native.index("if (is_base_model) {"):native.index("// ── Realtime-0.5B Streaming Model TTS Path")]
        self.assertIn("make_ddim_schedule(num_steps)", base)
        self.assertIn("si < num_steps", base)
        self.assertIn("si == num_steps - 1", base)
        self.assertIn("cfg_scale * (v_cond - v_uncond)", base)
        self.assertEqual(native.count("core_vibevoice_tts::diffusion_steps(ctx->params.tts_steps)"), 1)
        self.assertEqual(native.count("core_vibevoice_tts::guidance_scale("), 1)


if __name__ == "__main__":
    unittest.main()
