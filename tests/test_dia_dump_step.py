"""Actual diagnostic index parser and replay environment isolation contracts."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DiaDumpStepTest(unittest.TestCase):
    def test_actual_parser_and_replay_environment_restore(self):
        header = (ROOT / 'examples/cli/crispasr_dia_diff.h').read_text()
        start = header.index('inline void set_env(')
        end = header.index('\nstruct Scratch', start)
        environment = header[start:end]
        driver = r'''
#include "dia_dump_step.h"
#include <cstdlib>
#include <string>
#include <vector>
#include <cstdio>
''' + environment + r'''
int main() {
    size_t value = 99;
    if (!dia_debug::parse_dump_step(nullptr, 1226, value) || value != 0) return 1;
    if (!dia_debug::parse_dump_step("978", 1226, value) || value != 978) return 2;
    if (!dia_debug::parse_dump_step("1225", 1226, value) || value != 1225) return 3;
    const char* bad[] = {"", "-1", "+1", " 1", "1 ", "1x", "1.0", "1226", "99999999999999999999999999999"};
    for (auto input : bad) if (dia_debug::parse_dump_step(input, 1226, value)) return 4;
    if (dia_debug::parse_dump_step(nullptr, 0, value)) return 5;
    set_env("DIA_DUMP_STEP", "17");
    set_env("CRISPASR_DIA_DUMP_STEP", "malformed-parent-value");
    {
        Environment guard;
        if (std::getenv("DIA_DUMP_STEP") || std::getenv("CRISPASR_DIA_DUMP_STEP")) return 6;
        set_env("CRISPASR_DIA_DUMP_STEP", "978");
    }
    if (std::string(std::getenv("DIA_DUMP_STEP")) != "17") return 7;
    if (std::string(std::getenv("CRISPASR_DIA_DUMP_STEP")) != "malformed-parent-value") return 8;
    set_env("DIA_DUMP_STEP", nullptr); set_env("CRISPASR_DIA_DUMP_STEP", nullptr);
    { Environment guard; set_env("DIA_DUMP_STEP", "978"); }
    if (std::getenv("DIA_DUMP_STEP") || std::getenv("CRISPASR_DIA_DUMP_STEP")) return 9;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='dia-dump-step-') as temp:
            path = Path(temp); cpp = path / 'contract.cpp'; binary = path / 'contract'
            cpp.write_text(driver)
            subprocess.run(['c++', '-std=c++17', '-I', str(ROOT / 'src'),
                            str(cpp), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_actual_optional_layer_parser(self):
        text = (ROOT / 'examples/cli/crispasr_dia_diff.h').read_text()
        start = text.index('inline bool parse_layers(')
        end = text.index('\ninline bool write_codes', start)
        driver = r'''
#include "dia_dump_step.h"
#include <algorithm>
#include <string>
#include <vector>
''' + text[start:end] + r'''
int main() {
    std::vector<int> layers;
    if (!parse_layers("", layers) || !layers.empty()) return 1;
    if (!parse_layers("0,17,5", layers) || layers != std::vector<int>({0,17,5})) return 2;
    const char* bad[] = {"18", "-1", "+1", " 1", "1 ", "1x", "1.0", ",1", "1,", "1,,2", "1,1", "99999999999999999999999999999"};
    for (auto input : bad) if (parse_layers(input, layers)) return 3;
    if (!parse_layers("7", layers) || layers != std::vector<int>({7})) return 4;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='dia-layer-parser-') as temp:
            path = Path(temp); cpp = path / 'contract.cpp'; binary = path / 'contract'
            cpp.write_text(driver)
            subprocess.run(['c++', '-std=c++17', '-I', str(ROOT / 'src'),
                            str(cpp), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_actual_scratch_retains_requested_proof_and_rejects_existing_directory(self):
        text = (ROOT / 'examples/cli/crispasr_dia_diff.h').read_text()
        start = text.index('struct Scratch {')
        end = text.index('\ninline bool parse_layers', start)
        driver = """
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cstdio>
""" + text[start:end] + r"""
int main(int argc, char** argv) {
    if (argc != 2) return 1;
    std::filesystem::path ephemeral;
    { Scratch probe; ephemeral = probe.path; std::ofstream(probe.path / "array.f32") << "owned-data"; }
    if (std::filesystem::exists(ephemeral)) return 2;
    const std::string retained = argv[1];
    { Scratch probe(retained); std::ofstream(probe.path / "array.f32") << "owned-data"; }
    if (!std::filesystem::is_regular_file(std::filesystem::path(retained) / "array.f32")) return 3;
    bool rejected = false;
    try { Scratch duplicate(retained); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) return 4;
    std::ifstream file(std::filesystem::path(retained) / "array.f32");
    std::string data; file >> data;
    if (data != "owned-data") return 5;
    return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix='dia-owned-dumps-') as temp:
            path = Path(temp); cpp = path / 'contract.cpp'; binary = path / 'contract'
            cpp.write_text(driver)
            subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(binary)], check=True)
            subprocess.run([str(binary), str(path / 'retained')], check=True)


if __name__ == '__main__':
    unittest.main()
