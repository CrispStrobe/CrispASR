#!/usr/bin/env python3
"""Base CFG draw cadence; production C++ versus an independently qualified fixture.

The stored paired128 C++ oracle was checked pointwise against actual
Torch2.9.1 CPU torch.randn(2,64) for three frames: max error2.3841858e-7.
It is not mislabeled as raw Torch output. Its first frame is byte-identical
to the actual retained7B capture. No audio quality pass follows from this test.
"""
import base64
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

FIXTURE_B64 = 'KqX2P1Nfvj9nlWY/28AGwNWsLT+RBZ6/h2cwvbplzb/wi0C/WwnTP/byyL5nqbO/bVY6v9E2D7+h0kS/ny9DP3E30j+FbSO+4qr+vtcR4T7iFEK/UAaKP0MBTT+THtc/WbqjPy/xpT+IRxw/sNiqP+0ubb4ADCs9es6AvrAfXD/9PLG/VQlfvwq6ZL6A0ts/SESjPo1a2b50h5w+sktGv4lex78C4n4/pTlhv3PkGb9jF6O/ttsHQB0Jnr/Zz/m+TvBpv697KL/7yp89Z5sGPw3a+b7Ifpg/zmJQvwdqPL+gnbO/lXgTPV8Agr0Z9Sw/+07Ivagb7D/snpe/JRixP0Uw9z/BhIE/K9y3vz+fkL+jTAu+GlXRPxidJz8JdRM/7xySP70UmDyfJOe/S+lsPyMtwL41PIQ/WsgvvzgGIz8oAXm/e111P/ZBzz+Vrbk/fPmJPsxsV772mDu/GprVPZWPsj4+tHc/t27uvv9pzT9Kuh7AS8jVvqEEmb9Q9U8/VEXzv/cPaj7zpcs8ZCCxvsDbkj59GDu/FgQzPuAFjL+XE82/uiutP074pD/HM1Y9Mv/Fv323QT9yiEY/wrIBQH21Ej059/Y9/D9OvwWPVL4hlG6/yKTLv6hnkb/0yAW/VM4Ev9MpwL+dnPa/b+sCPtPugj+XSA6/OEs0P2+6NT+4ey+/Z0UQPx3rwL9OKs6/ZlG9vw5T3T6pBgC+6ThIP2ukzL+Hf9+9UBc3P3NQID1tJqc/z4J8PrMh/b+5mpI8+Yywv5w0ID/UbyXAuJrEvENA+r0YOz+/ocraP1lAbT2Ws5g/Ofn3P/eMOj/bG3s/cUXUPkMKlD+FwYk+wAgWvcgpeT8a7oG/HLsKvwPO4b6DkqC+OFsEvscHN7+L0EK93VMBQKwBgj5vum8/IFU2P5YcAr0WKdA9Y/GrP9aYNj9Dv84+Scc2v0ZvVT85Y3W/wELoPvF/nz/WnRPA37ikvys0OD7hHAjAF00Jvug3hb/hxEO/829ivYw4mj9kg3u/'
FIXTURE_SHA256 = '48ed9c433f5bb1099b007a28645e202299a310fa6b701d3be03583c122e79b63'


class BaseCFGNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which(os.environ.get('CXX','c++'))
        if not compiler:raise unittest.SkipTest('requires C++ compiler')
        source=Path(os.environ.get('CRISPASR_VIBEVOICE_RNG_SOURCE',Path(__file__).resolve().parents[1]/'src/vibevoice.cpp')).read_text()
        begin=source.index('struct mt19937_state {')
        end=source.index("// Upstream's ASR path",begin)
        functions=source[begin:end]
        candidate='fill_base_cfg_gaussian_noise' if 'static void fill_base_cfg_gaussian_noise(' in functions else 'fill_gaussian_noise'
        cls.temporary=tempfile.TemporaryDirectory(prefix='vibevoice-cfg-rng-')
        root=Path(cls.temporary.name)
        cpp=root/'probe.cpp'
        cpp.write_text('#include <cstdint>\n#include <cmath>\n#include <cstring>\n#include <cstdio>\n#include <vector>\n'+functions+"\nint main(){ mt19937_state legacy,candidate;mt19937_seed(legacy,42);mt19937_seed(candidate,42);for(int f=0;f<3;++f){float a[64],b[64];fill_gaussian_noise(a,64,legacy);"+candidate+"(b,64,candidate);fwrite(a,4,64,stdout);fwrite(b,4,64,stdout);}return 0;}\n")
        subprocess.run([compiler,'-std=c++17','-O2',str(cpp),'-o',str(root/'probe')],check=True,capture_output=True,timeout=30)
        raw=subprocess.check_output([str(root/'probe')],timeout=10)
        if len(raw)!=1536:raise ValueError('wrong three-frame payload size')
        values=struct.unpack('<384f',raw)
        cls.legacy=[values[f*128:f*128+64] for f in range(3)]
        cls.candidate=[values[f*128+64:(f+1)*128] for f in range(3)]
        fixture=base64.b64decode(FIXTURE_B64,validate=True)
        if hashlib.sha256(fixture).hexdigest()!=FIXTURE_SHA256:raise ValueError('fixture identity mismatch')
        gold=struct.unpack('<192f',fixture)
        cls.reference=[gold[f*64:(f+1)*64] for f in range(3)]

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def test_candidate_matches_three_upstream_qualified_frames(self):
        for frame in range(3):
            difference=max(abs(x-y) for x,y in zip(self.candidate[frame],self.reference[frame]))
            self.assertLess(difference,1e-6,f'frame{frame}: max_abs={difference}')

    def test_original_first_frame_preserved(self):
        self.assertEqual(self.candidate[0],self.legacy[0])
        self.assertLess(max(abs(x-y) for x,y in zip(self.legacy[0],self.reference[0])),1e-6)

    def test_old_single_row_cadence_is_a_counterexample(self):
        self.assertGreater(max(abs(x-y) for x,y in zip(self.legacy[1],self.reference[1])),3.)
        self.assertGreater(max(abs(x-y) for x,y in zip(self.legacy[2],self.reference[2])),4.)


if __name__=='__main__':unittest.main()
