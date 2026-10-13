#!/usr/bin/env python3
"""Execute the production GPU-selection helper against a fake device registry.

These controls need no GPU SDK/model; they prove routing, not NPU inference.
Use --header to run the same controls against a historical production header.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'src/core/gpu_backend_pref.h'

MOCK = r'''
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <stdexcept>
enum ggml_backend_dev_type { GGML_BACKEND_DEVICE_TYPE_CPU, GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU };
struct Backend { int id; };
struct Device { const char* name; ggml_backend_dev_type type; Backend backend; bool fails = false; };
struct Registry { const char* name; std::vector<Device*> devices; };
using ggml_backend_t = Backend*;
using ggml_backend_dev_t = Device*;
using ggml_backend_reg_t = Registry*;
std::vector<Device*> devices;
std::vector<Registry*> registries;
int enumerations = 0;
std::vector<int> initialized;
Backend cpu{-1}, automatic{-2};
namespace core_metal_cache { inline bool apply() { return false; } }
namespace core_cpu_backend { inline Backend* init() { return &cpu; } }
size_t ggml_backend_dev_count() { ++enumerations; return devices.size(); }
Device* ggml_backend_dev_get(size_t i) { return devices.at(i); }
ggml_backend_dev_type ggml_backend_dev_type(Device* dev) { return dev->type; }
const char* ggml_backend_dev_name(Device* dev) { return dev->name; }
Backend* ggml_backend_dev_init(Device* dev, const char*) {
    initialized.push_back(dev->backend.id);
    return dev->fails ? nullptr : &dev->backend;
}
size_t ggml_backend_reg_count() { return registries.size(); }
Registry* ggml_backend_reg_get(size_t i) { return registries.at(i); }
const char* ggml_backend_reg_name(Registry* reg) { return reg->name; }
size_t ggml_backend_reg_dev_count(Registry* reg) { return reg->devices.size(); }
Device* ggml_backend_reg_dev_get(Registry* reg, size_t i) { return reg->devices.at(i); }
Backend* ggml_backend_init_best() { return &automatic; }
'''

CONTROL = r'''
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::string mode = argv[1];
    Device host{"CPU", GGML_BACKEND_DEVICE_TYPE_CPU, {-1}};
    Device other{"CUDA0", GGML_BACKEND_DEVICE_TYPE_GPU, {9}};
    Device first{"CANN0", GGML_BACKEND_DEVICE_TYPE_GPU, {0}};
    Device second{"CANN1", GGML_BACKEND_DEVICE_TYPE_GPU, {1}};
    Registry reg{"CANN", {&host, &first, &second}};
    devices = {&host, &other, &first, &second};
    registries = {&reg};
    crispasr_set_gpu_backend_pref("cann");
    int expected = 0;
    if (mode == "direct-second" || mode == "registry-second") {
        crispasr_set_gpu_device_pref(1); expected = 1;
    }
    if (mode == "registry-default" || mode == "registry-second") {
        first.name = "Ascend910B0"; second.name = "Ascend910B1";
    }
    if (mode == "case-insensitive") crispasr_set_gpu_backend_pref("CaNn");
    if (mode == "invalid-index") { crispasr_set_gpu_device_pref(2); expected = -2; }
    if (mode == "failed-selected") {
        crispasr_set_gpu_device_pref(1); second.fails = true; expected = -2;
    }
    if (mode == "cpu") { crispasr_set_gpu_backend_pref("cpu"); expected = -1; }
    if (mode == "auto") { crispasr_set_gpu_backend_pref(""); expected = -2; }
    if (mode == "metal-alias") {
        crispasr_set_gpu_backend_pref("metal"); first.name = "MTL0"; expected = 0;
    }
    auto result = crispasr_init_gpu_backend();
    if (!result || result->id != expected) {
        fprintf(stderr, "%s: expected device %d, got %d\n", mode.c_str(), expected, result ? result->id : -99);
        return 1;
    }
    if (mode == "cpu" && enumerations != 0) return 3;
    if ((mode == "invalid-index" || mode == "auto") && !initialized.empty()) return 4;
    if (mode == "failed-selected") {
        for (int id : initialized) if (id != 1) return 5;
    }
    return 0;
}
'''


class GPUBackendSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = HEADER.read_text()
        start = text.index('namespace crispasr_gpu_pref {')
        end = text.index('// ── Can this backend actually run')
        production = text[start:end]
        # The baseline predates the setter. This inert compatibility declaration
        # lets it compile while the second-device controls still expose the bug.
        compatibility = '' if 'inline void crispasr_set_gpu_device_pref(' in production else (
            'inline void crispasr_set_gpu_device_pref(int) {}\n')
        cls.temp = tempfile.TemporaryDirectory(prefix='crispasr-gpu-selection-')
        directory = Path(cls.temp.name)
        cls.binary = directory / 'selection'
        source = directory / 'selection.cpp'
        source.write_text(MOCK + compatibility + production + CONTROL)
        try:
            subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            str(source), '-o', str(cls.binary)], check=True,
                           capture_output=True, text=True, timeout=60)
        except BaseException:
            cls.temp.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_registered_devices(self):
        for scenario in ('direct-default', 'direct-second', 'registry-default',
                         'registry-second', 'case-insensitive', 'invalid-index',
                         'failed-selected', 'cpu', 'auto', 'metal-alias'):
            with self.subTest(scenario=scenario):
                result = subprocess.run([str(self.binary), scenario],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--header', type=Path)
    args, remaining = parser.parse_known_args()
    if args.header is not None:
        HEADER = args.header
    unittest.main(argv=[__file__, *remaining])
