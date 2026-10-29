"""Model-free caller controls for the issue #560 CLI detect-and-exit path."""

import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "examples/cli/crispasr_run.cpp"
CLI = ROOT / "examples/cli/cli.cpp"
BASELINE = "--baseline" in sys.argv


def assert_headroom():
    if not Path("/proc/meminfo").is_file():
        raise unittest.SkipTest(
            "Guarded compile fixture requires Linux memory telemetry"
        )
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available_kib < 3 * 1024**2 or os.getloadavg()[0] > 2 * (os.cpu_count() or 1):
        raise unittest.SkipTest(
            "Tiny C++ fixture needs at least 3 GiB available and CPU headroom"
        )


def source(path):
    if BASELINE:
        return subprocess.check_output(
            ["git", "show", "HEAD:" + str(path.relative_to(ROOT))], cwd=ROOT, text=True
        )
    return path.read_text()


class LidOnlyCaller(unittest.TestCase):
    def test_route_precedes_asr_resolution(self):
        cli = source(CLI)
        route = cli.index("return crispasr_run_lid_only(params);")
        self.assertLess(route, cli.index("if (!params.hf_repo.empty())"))
        self.assertLess(route, cli.index("return crispasr_run_backend(params);"))
        self.assertLess(
            route, cli.index("whisper_init_from_file_with_params(params.model.c_str()")
        )

    def test_actual_caller(self):
        run = source(RUN)
        start = run.index("int crispasr_run_lid_only(const whisper_params& params) {")
        end = run.index("\nint crispasr_run_backend(", start)
        caller = run[start:end]
        harness = r"""
#include <cstdio>
#include <string>
#include <vector>

struct whisper_params {
    std::vector<std::string> fname_inp;
    std::string lid_backend;
    std::string lid_model;
};
struct crispasr_lid_result {
    std::string lang_code;
    float confidence = -1;
    std::string source;
};
int detect_calls = 0;
int free_calls = 0;
bool read_audio_data(const std::string& path, std::vector<float>& samples,
                     std::vector<std::vector<float>>&, bool) {
    if (path == "bad.wav") return false;
    samples.assign(160, 0.5f);
    return true;
}
bool crispasr_detect_language_cli(const float*, int, const whisper_params& params,
                                  crispasr_lid_result& result) {
    ++detect_calls;
    if (params.lid_backend == "firered") return false;
    if (params.lid_model == "missing.gguf") return false;
    result.lang_code = "de";
    result.confidence = 0.75f;
    result.source = params.lid_backend.empty() ? "whisper" : params.lid_backend;
    return true;
}
void crispasr_lid_free_cache() { ++free_calls; }
"""
        harness += caller
        harness += r"""
int main(int argc, char** argv) {
    whisper_params params;
    params.lid_backend = argv[1];
    params.lid_model = argv[2];
    for (int i = 3; i < argc; ++i) params.fname_inp.emplace_back(argv[i]);
    const int rc = crispasr_run_lid_only(params);
    fprintf(stderr, "CALLS=%d,%d\n", detect_calls, free_calls);
    return rc;
}
"""
        assert_headroom()
        with tempfile.TemporaryDirectory(prefix="issue560-lid-") as scratch:
            binary = Path(scratch) / "caller"
            subprocess.run(
                ["c++", "-std=c++17", "-O0", "-x", "c++", "-", "-o", str(binary)],
                input=harness,
                text=True,
                check=True,
                cwd=ROOT,
            )

            def call(backend, lid_model, *files):
                return subprocess.run(
                    [str(binary), backend, lid_model, *files],
                    capture_output=True,
                    text=True,
                    check=False,
                )

            default = call("", "", "speech.wav")
            self.assertEqual(default.returncode, 0)
            self.assertEqual(
                default.stdout,
                "lang=de\tconfidence=0.750000\tsource=whisper\tfile=speech.wav\n",
            )
            self.assertIn("CALLS=1,1", default.stderr)

            explicit = call("silero", "chosen.gguf", "a.wav", "b.wav")
            self.assertEqual(explicit.returncode, 0)
            self.assertEqual(explicit.stdout.count("source=silero"), 2)
            self.assertIn("CALLS=2,1", explicit.stderr)

            for backend, model, files, expected_calls in [
                ("", "", (), "CALLS=0,0"),
                ("self", "", ("speech.wav",), "CALLS=0,0"),
                ("probe", "", ("speech.wav",), "CALLS=0,0"),
                ("off", "", ("speech.wav",), "CALLS=0,0"),
                ("", "", ("bad.wav",), "CALLS=0,1"),
                ("firered", "", ("speech.wav",), "CALLS=1,1"),
                ("", "missing.gguf", ("speech.wav",), "CALLS=1,1"),
            ]:
                failed = call(backend, model, *files)
                self.assertNotEqual(failed.returncode, 0)
                self.assertEqual(failed.stdout, "")
                self.assertIn(expected_calls, failed.stderr)

            partial = call("silero", "chosen.gguf", "a.wav", "bad.wav")
            self.assertNotEqual(partial.returncode, 0)
            self.assertEqual(partial.stdout.count("lang=de"), 1)

    def test_actual_whisper_model_resolver(self):
        lid = source(ROOT / "examples/cli/crispasr_lid_cli.cpp")
        start = lid.index(
            "std::string resolve_whisper_lid_model(const whisper_params& p) {"
        )
        end = lid.index("\nstd::string resolve_silero_lid_model(", start)
        resolver = lid[start:end]
        harness = r"""
#include <cstdio>
#include <string>
struct whisper_params {
    std::string lid_model;
    bool no_prints = true;
    std::string cache_dir;
};
constexpr const char* kWhisperLidDefaultFile = "ggml-tiny.bin";
constexpr const char* kWhisperLidDefaultUrl = "https://example.test/tiny";
int cache_calls = 0;
std::string expand_home(const std::string& path) { return "EXPLICIT:" + path; }
namespace crispasr_cache {
std::string ensure_cached_file(const std::string& filename, const std::string&, bool,
                               const char*, const std::string&) {
    ++cache_calls;
    return "CACHE:" + filename;
}
}
"""
        harness += resolver
        harness += r"""
int main(int argc, char** argv) {
    whisper_params params;
    if (argc > 1) params.lid_model = argv[1];
    const std::string path = resolve_whisper_lid_model(params);
    fprintf(stdout, "%s\t%d\n", path.c_str(), cache_calls);
}
"""
        assert_headroom()
        with tempfile.TemporaryDirectory(prefix="issue560-lid-") as scratch:
            binary = Path(scratch) / "resolver"
            subprocess.run(
                ["c++", "-std=c++17", "-O0", "-x", "c++", "-", "-o", str(binary)],
                input=harness,
                text=True,
                check=True,
                cwd=ROOT,
            )

            def call(model):
                return subprocess.check_output([str(binary), model], text=True)

            self.assertEqual(call(""), "CACHE:ggml-tiny.bin\t1\n")
            self.assertEqual(call("auto"), "CACHE:ggml-tiny.bin\t1\n")
            self.assertEqual(call("chosen.gguf"), "EXPLICIT:chosen.gguf\t0\n")


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if BASELINE:
        sys.argv.remove("--baseline")
    unittest.main()
