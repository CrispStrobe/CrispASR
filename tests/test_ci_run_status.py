"""Run the real CI shell with fake tools only: no builds, models or benchmarks."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("CRISPASR_CI_RUN_SCRIPT", ROOT / "ci/run.sh"))
REAL_TEE = shutil.which("tee")

TOOL = r'''#!/bin/bash
name=${0##*/}
mode=${PWD##*/build-ci-}
case "$name" in
  cmake) stage=configure:$mode ;;
  make) stage=build:$mode ;;
  ctest) stage=test:$mode ;;
  download)
    stage=download:$1
    ;;
  mock-bench)
    if [ "$1" = -w ]; then
      if [ "$2" = 1 ]; then stage=bench:memcpy; else stage=bench:mul_mat; fi
    else
      model=${2##*/ggml-}; model=${model%.bin}; stage=bench:$model
    fi
    printf 'bench-args:%s\n' "$*" >> "$MOCK_TRACE"
    ;;
  *) printf 'UNEXPECTED:%s\n' "$name" >> "$MOCK_TRACE"; exit 97 ;;
esac
printf '%s\n' "$stage" >> "$MOCK_TRACE"
if [ "$stage" = "$MOCK_FAIL_STAGE" ]; then
  if [ "$MOCK_SIGNAL" = 1 ]; then kill -SEGV $$; fi
  exit "${MOCK_STATUS:-42}"
fi
if [ "$name" = download ]; then
  : > "ggml-$1.bin"
elif [ "$name" = make ] && [ "$mode" = release ] && [ "$MOCK_MISSING_BENCH" != 1 ]; then
  mkdir -p bin
  printf '#!/bin/bash\nexec mock-bench "$@"\n' > bin/crispasr-bench
  chmod +x bin/crispasr-bench
elif [ "$name" = cmake ] && [ "$MOCK_EXIT_WRITE_FAIL" = 1 ]; then
  mkdir -p "$MOCK_OUT/ctest_debug-cmake.exit"
fi
printf 'MOCK ONLY: %s\n' "$stage"
'''


class CiRunStatus(unittest.TestCase):
    def run_mock(self, **options):
        with tempfile.TemporaryDirectory(prefix="ci-status-mock-") as directory:
            root = Path(directory)
            (root / "ci").mkdir()
            (root / "models").mkdir()
            (root / "tools").mkdir()
            script = root / "ci/run.sh"
            script.write_bytes(SOURCE.read_bytes())
            trace = root / "trace"
            out = root / "out"
            for name in ("cmake", "make", "ctest", "download", "mock-bench", "wget", "curl"):
                tool = root / "tools" / name
                tool.write_text(TOOL)
                tool.chmod(0o755)
            download = root / "models/download-ggml-model.sh"
            download.write_text('#!/bin/bash\nexec download "$@"\n')
            tee = root / "tools/tee"
            tee.write_text("#!/bin/bash\n" + shlex.quote(REAL_TEE) + ' "$@"\n'
                           'status=$?\n'
                           'if [ -n "$MOCK_TEE_FAIL" ] && [[ "${@: -1}" = *"$MOCK_TEE_FAIL" ]]; then exit 23; fi\n'
                           'exit "$status"\n')
            tee.chmod(0o755)
            env = os.environ.copy()
            # Isolate from an owner's GPU/performance build environment.
            for key in list(env):
                if key.startswith(("GG_BUILD_", "MOCK_")):
                    del env[key]
            env.update(PATH=str(root / "tools") + os.pathsep + env["PATH"],
                       GG_BUILD_TEST_MODELS="fake-one,fake-two", MOCK_TRACE=str(trace), MOCK_OUT=str(out))
            env.update(options)
            result = subprocess.run(["bash", "-c", 'ulimit -c 0; exec bash "$@"', "mock",
                                     str(script), str(out), str(root / "mnt")],
                                    cwd=root, env=env, text=True, capture_output=True, timeout=10)
            events = trace.read_text().splitlines() if trace.exists() else []
            exits = {p.name: p.read_text().strip() for p in out.glob("*.exit") if p.is_file()}
            return result, events, exits

    def assert_stopped(self, stage, expected, **options):
        result, events, exits = self.run_mock(MOCK_FAIL_STAGE=stage, MOCK_STATUS=str(expected), **options)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        phases = [e for e in events if not e.startswith("bench-args:")]
        self.assertEqual(phases[-1], stage, events)
        self.assertFalse(any(e.startswith("UNEXPECTED:") for e in events), events)
        return exits

    def test_success_keeps_both_builds_and_all_performance_commands(self):
        result, events, exits = self.run_mock()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        phases = [e for e in events if not e.startswith("bench-args:")]
        self.assertEqual(phases, ["download:fake-one", "download:fake-two", "configure:debug", "build:debug",
                                 "test:debug", "configure:release", "build:release", "test:release",
                                 "bench:memcpy", "bench:mul_mat", "bench:fake-one", "bench:fake-two"])
        self.assertIn("bench-args:-w 1 -t 4", events)
        self.assertIn("bench-args:-w 2 -t 4", events)
        self.assertEqual(sum(e.startswith("bench-args:-m ") and e.endswith(" -t 4 -nfa") for e in events), 2)
        self.assertEqual(exits["bench.exit"], "0")

    def test_download_failure_stops_before_second_download_or_configure(self):
        self.assert_stopped("download:fake-one", 19)

    def test_configure_failure_is_not_hidden_by_tee(self):
        exits = self.assert_stopped("configure:debug", 31)
        self.assertEqual(exits["ctest_debug-cmake.exit"], "31")

    def test_build_failure_stops_before_ctest(self):
        self.assert_stopped("build:debug", 32)

    def test_debug_failure_preserves_8_and_stops_release_and_bench(self):
        exits = self.assert_stopped("test:debug", 8)
        self.assertEqual(exits["ctest_debug.exit"], "8")

    def test_release_failure_stops_bench(self):
        self.assert_stopped("test:release", 8)

    def test_missing_benchmark_preserves_127_and_stops_model_loop(self):
        result, events, exits = self.run_mock(MOCK_MISSING_BENCH="1")
        self.assertEqual(result.returncode, 127, result.stdout + result.stderr)
        self.assertEqual(exits["bench-memcpy.exit"], "127")
        self.assertEqual(exits["bench.exit"], "127")
        self.assertFalse(any(e.startswith("bench:") for e in events))

    def test_memcpy_failure_is_fatal(self):
        exits = self.assert_stopped("bench:memcpy", 42)
        self.assertEqual(exits["bench-memcpy.exit"], "42")

    def test_matrix_failure_stops_before_second_model(self):
        self.assert_stopped("bench:fake-one", 43)

    def test_mock_crash_is_not_informational(self):
        self.assert_stopped("bench:memcpy", 139, MOCK_SIGNAL="1")

    def test_inner_tee_failure_is_fatal(self):
        result, events, exits = self.run_mock(MOCK_TEE_FAIL="ctest_debug-cmake.log")
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertEqual(events[-1], "configure:debug")
        self.assertEqual(exits["ctest_debug-cmake.exit"], "23")

    def test_outer_tee_failure_is_fatal(self):
        result, events, exits = self.run_mock(MOCK_TEE_FAIL="ctest_debug.log")
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertEqual(events[-1], "test:debug")
        self.assertEqual(exits["ctest_debug.exit"], "23")

    def test_producer_failure_wins_over_tee_failure(self):
        exits = self.assert_stopped("configure:debug", 31, MOCK_TEE_FAIL="ctest_debug-cmake.log")
        self.assertEqual(exits["ctest_debug-cmake.exit"], "31")

    def test_exit_file_failure_stops_next_phase(self):
        result, events, _ = self.run_mock(MOCK_EXIT_WRITE_FAIL="1")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(events[-1], "configure:debug")


if __name__ == "__main__":
    unittest.main()
