"""Execute the actual native speaker setter; invalid IDs preserve prior state."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_openvoice_cli_routing import function

ROOT = Path(__file__).resolve().parents[1]


class SpeakerRange(unittest.TestCase):
    def test_invalid_ids_do_not_install_out_of_bounds_indices(self):
        source = Path(os.environ.get('MELOTTS_SOURCE', ROOT/'src/melotts.cpp')).read_text()
        setter = function(source, 'void melotts_set_speaker_id(')
        helper = function(source, 'static bool melotts_speaker_id_valid(') if 'static bool melotts_speaker_id_valid(' in source else ''
        code = '''#include <cstdint>
#include <cassert>
#include <climits>
struct melotts_context {struct {uint32_t n_speakers=2;} hp;int speaker_id=0;};
''' + helper + setter + '''
int main(){melotts_context ctx;
 melotts_set_speaker_id(&ctx,1);assert(ctx.speaker_id==1);
 for(int id:{-1,2,INT_MAX,INT_MIN}){melotts_set_speaker_id(&ctx,id);assert(ctx.speaker_id==1);}
 melotts_set_speaker_id(&ctx,0);assert(ctx.speaker_id==0);
 melotts_set_speaker_id(nullptr,0);
 ctx.hp.n_speakers=0;melotts_set_speaker_id(&ctx,1);assert(ctx.speaker_id==0);
}
'''
        code = '#include <initializer_list>\n' + code
        with tempfile.TemporaryDirectory(prefix='melo-speaker-') as temp:
            root = Path(temp);(root/'test.cpp').write_text(code)
            subprocess.run(['c++','-std=c++17',str(root/'test.cpp'),'-o',str(root/'test')],check=True)
            subprocess.run([str(root/'test')],check=True)


if __name__ == '__main__':
    unittest.main()
