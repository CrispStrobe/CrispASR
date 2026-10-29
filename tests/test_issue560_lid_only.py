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
        resolver_start = lid.index(
            "std::string resolve_whisper_lid_model(const whisper_params& p) {"
        )
        resolver_end = lid.index("\nstd::string resolve_silero_lid_model(", resolver_start)
        detect_start = lid.index("static bool detect_whisper_cli(")
        window_start = lid.index("static bool crispasr_detect_language_cli_window(", detect_start)
        window_end = lid.index("\nbool crispasr_detect_language_cli(", window_start)
        harness = r"""
#include <cstdio>
#include <string>
struct whisper_params {
    std::string lid_backend;
    std::string lid_model;
    bool no_prints = true;
    std::string cache_dir;
    int n_threads = 2;
    bool use_gpu = false;
    int gpu_device = 0;
    bool flash_attn = false;
};
struct crispasr_lid_result {
    std::string lang_code;
    float confidence = 0;
    std::string source;
};
enum class CrispasrLidMethod { Whisper, Silero, Firered, Ecapa };
struct CrispasrLidOptions {
    CrispasrLidMethod method = CrispasrLidMethod::Whisper;
    std::string model_path;
    int n_threads = 0;
    bool use_gpu = false;
    int gpu_device = 0;
    bool flash_attn = false;
    bool verbose = false;
};
struct CrispasrLidResult {
    std::string lang_code;
    float confidence = 0;
    std::string source;
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
std::string resolve_silero_lid_model(const whisper_params& p) { return p.lid_model; }
std::string resolve_firered_lid_model(const whisper_params& p) { return p.lid_model; }
std::string resolve_ecapa_lid_model(const whisper_params& p) { return p.lid_model; }
bool detect_with_sherpa(const float*, int, const whisper_params&, crispasr_lid_result&) { return false; }
bool crispasr_detect_language(const float*, int, const CrispasrLidOptions& opts,
                              CrispasrLidResult& result) {
    if (opts.method == CrispasrLidMethod::Silero) {
        if (opts.model_path != "success.gguf") return false;
        result = {"ko", 0.7f, "silero"};
        return true;
    }
    if (opts.method != CrispasrLidMethod::Whisper) return false;
    fprintf(stdout, "WHISPER_MODEL=%s\n", opts.model_path.c_str());
    result = {"ko", 0.8f, "whisper"};
    return true;
}
"""
        harness += lid[resolver_start:resolver_end]
        harness += lid[detect_start:window_end]
        harness += r"""
int main(int argc, char** argv) {
    whisper_params params;
    if (argc > 1) params.lid_backend = argv[1];
    if (argc > 2) params.lid_model = argv[2];
    const float samples[10] = {};
    crispasr_lid_result result;
    if (!crispasr_detect_language_cli_window(samples, 10, params, result)) return 2;
    fprintf(stdout, "SOURCE=%s\tCACHE_CALLS=%d\n", result.source.c_str(), cache_calls);
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

            def call(backend, model):
                return subprocess.check_output([str(binary), backend, model], text=True)

            self.assertEqual(call("", ""), "WHISPER_MODEL=CACHE:ggml-tiny.bin\nSOURCE=whisper\tCACHE_CALLS=1\n")
            self.assertEqual(call("", "chosen.bin"), "WHISPER_MODEL=EXPLICIT:chosen.bin\nSOURCE=whisper\tCACHE_CALLS=0\n")
            self.assertEqual(call("whisper", "chosen.bin"), "WHISPER_MODEL=EXPLICIT:chosen.bin\nSOURCE=whisper\tCACHE_CALLS=0\n")
            self.assertEqual(call("silero", "chosen.gguf"), "WHISPER_MODEL=CACHE:ggml-tiny.bin\nSOURCE=whisper\tCACHE_CALLS=1\n")
            self.assertEqual(call("probe", "chosen.gguf"), "WHISPER_MODEL=CACHE:ggml-tiny.bin\nSOURCE=whisper\tCACHE_CALLS=1\n")
            self.assertEqual(call("silero", "success.gguf"), "SOURCE=silero\tCACHE_CALLS=0\n")


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if BASELINE:
        sys.argv.remove("--baseline")
    unittest.main()
