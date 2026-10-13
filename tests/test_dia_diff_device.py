"""Compile actual replay request/guard code; no simulated GPU or model claim."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DiaDiffDeviceTest(unittest.TestCase):
    def test_explicit_device_request_survives_guard_and_restores_parent(self):
        text = (ROOT / 'examples/cli/crispasr_dia_diff.h').read_text()
        start = text.index('inline void set_env('); end = text.index('\nstruct Scratch', start)
        driver = r'''
#include <cstdlib>
#include <string>
#include <vector>
''' + text[start:end] + r'''
int main() {
    set_env("CRISPASR_DIA_TTS_GPU", nullptr); set_env("DIA_TTS_GPU", nullptr);
    if (requested_gpu()) return 1;
    set_env("DIA_TTS_GPU", "1");
    if (!requested_gpu()) return 2;
    set_env("CRISPASR_DIA_TTS_GPU", "0");
    if (requested_gpu()) return 3;
    set_env("CRISPASR_DIA_TTS_GPU", "1");
    const bool gpu = requested_gpu();
    {
        Environment guard;
        if (requested_gpu()) return 4;
        apply_replay_device(gpu);
        if (!requested_gpu() || std::string(std::getenv("CRISPASR_DIA_TTS_GPU")) != "1") return 5;
    }
    if (!requested_gpu() || std::string(std::getenv("DIA_TTS_GPU")) != "1") return 6;
    set_env("CRISPASR_DIA_TTS_GPU", "malformed-parent-value");
    {
        const bool request = requested_gpu();
        Environment guard;
        apply_replay_device(request);
        if (requested_gpu() || std::string(std::getenv("CRISPASR_DIA_TTS_GPU")) != "0") return 7;
    }
    if (std::string(std::getenv("CRISPASR_DIA_TTS_GPU")) != "malformed-parent-value") return 8;
    set_env("CRISPASR_DIA_TTS_GPU", nullptr); set_env("DIA_TTS_GPU", nullptr);
    { Environment guard; apply_replay_device(false); if (requested_gpu()) return 9; }
    if (std::getenv("CRISPASR_DIA_TTS_GPU") || std::getenv("DIA_TTS_GPU")) return 10;
    for (const char* suffix : {"DUMP_SELF_ATTN", "DUMP_DECODER_BOUNDARIES"}) {
    const std::string modern = std::string("CRISPASR_DIA_") + suffix;
    const std::string legacy = std::string("DIA_") + suffix;
    set_env(modern.c_str(), "1"); set_env(legacy.c_str(), "parent-value");
    {
        Environment guard;
        if (std::getenv(modern.c_str()) || std::getenv(legacy.c_str())) return 11;
        set_env(modern.c_str(), "1");
    }
    if (std::string(std::getenv(modern.c_str())) != "1" ||
        std::string(std::getenv(legacy.c_str())) != "parent-value") return 12;
    set_env(modern.c_str(), nullptr); set_env(legacy.c_str(), nullptr);
    { Environment guard; set_env(modern.c_str(), "1"); }
    if (std::getenv(modern.c_str()) || std::getenv(legacy.c_str())) return 13;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='dia-diff-device-') as temp:
            path = Path(temp); cpp = path / 'contract.cpp'; binary = path / 'contract'
            cpp.write_text(driver)
            subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
