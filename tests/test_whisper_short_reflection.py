"""Compile the production reflection block with ASan and real short buffers."""
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReflectionTest(unittest.TestCase):
    def test_short_and_normal_audio(self):
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(memory['MemAvailable'].split()[0]) * 1024 < 3 * 1024**3 or os.getloadavg()[0] > 2 * (os.cpu_count() or 1):
            self.skipTest('insufficient headroom for small sanitizer compile')
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        text = (ROOT / 'src/crispasr.cpp').read_text()
        start = text.index('    // reflective pad 200 samples at the beginning of audio', text.index('static bool log_mel_spectrogram('))
        end = text.index('    mel.n_mel = n_mel;', start)
        block = text[start:end]
        code = '''#include <algorithm>\n#include <cstdint>\n#include <vector>\n#include <cstdlib>\nint main(int argc,char** argv){\nint n_samples=atoi(argv[1]);int64_t stage_2_pad=200;\nstd::vector<float> input(n_samples);for(int i=0;i<n_samples;i++)input[i]=i;\nconst float* samples=input.data();std::vector<float> samples_padded(400,0);\n''' + block + '''\nif(n_samples==201)for(int i=0;i<200;i++)if(samples_padded[i]!=200-i)return 2;\nif(n_samples==2 && samples_padded[199]!=1)return 3;\nreturn 0;}\n'''
        with tempfile.TemporaryDirectory(prefix='whisper-short-') as tmp:
            root=Path(tmp);(root/'test.cpp').write_text(code)
            subprocess.run(['clang++','-std=c++17','-fsanitize=address','-O0',str(root/'test.cpp'),'-o',str(root/'test')],check=True)
            for length in [0, 1, 2, 200, 201, 250]:
                with self.subTest(length=length):
                    subprocess.run([str(root/'test'),str(length)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)


if __name__ == '__main__': unittest.main()
