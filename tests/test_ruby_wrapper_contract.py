"""Compile the actual Ruby extension against public-header, model-free probes."""
import os
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUBY = os.environ.get('CRISPASR_RUBY') or shutil.which('ruby')
PROGRAM = (ROOT/'tests/fixtures/ruby-contract-program.rb').read_text()

@unittest.skipUnless(RUBY and shutil.which('cc'), 'Ruby SDK and C compiler required')
class RubyWrapperContract(unittest.TestCase):
    def test_metadata_music_conversion_and_stream_callbacks(self):
        subprocess.run([sys.executable,str(ROOT/'tools/sync_ruby_context_guards.py'),'--check'],check=True)
        env = os.environ.copy()
        config = subprocess.check_output([RUBY, '-rrbconfig', '-e',
            'puts RbConfig::CONFIG.values_at("rubyhdrdir","rubyarchhdrdir")'], env=env, text=True).splitlines()
        headers = os.environ.get('CRISPASR_RUBY_HEADERS')
        if headers:
            config = headers.split(os.pathsep)
        with tempfile.TemporaryDirectory(prefix='ruby-wrapper-abi-') as tmp:
            tmp = Path(tmp)
            source=(ROOT/'src/crispasr_c_api.cpp').read_text();layouts=[]
            for name in ['crispasr_diarize_seg_abi','crispasr_diarize_turn_abi','crispasr_diarize_opts_abi']:
                start=source.index('struct '+name+' {');end=source.index('};',start)+2;layouts.append(source[start:end])
            (tmp/'runtime-abi-layout.h').write_text('\n'.join(layouts))
            start=source.index('struct crispasr_open_params_v1 {')
            (tmp/'runtime-open-layout.h').write_text(source[start:source.index('};',start)+2])
            flags = ['-fPIC', '-DCRISPASR_WRAPPER_PROBE', *['-I'+h for h in config], '-I'+str(ROOT/'include'), '-I'+str(ROOT/'ggml/include')]
            obj = tmp/'wrapper.o'
            subprocess.run(['cc', *flags, '-c', str(ROOT/'bindings/ruby/ext/ruby_crispasr_session.c'), '-o', str(obj)], check=True)
            probe = tmp/'probe.o'
            metadata = tmp/'metadata.o'
            audio = tmp/'audio.o'
            chat = tmp/'chat.o'
            aux = tmp/'aux.o'
            parakeet = tmp/'parakeet.o'
            titanet = tmp/'titanet.o'
            db = tmp/'db.o'
            speaker = tmp/'speaker.o'
            stream = tmp/'stream.o'
            text = tmp/'text.o'
            mic = tmp/'mic.o'
            vad = tmp/'vad.o'
            for target, source in [(probe,'python-wrapper-abi.c'),(metadata,'wrapper-session-metadata.c'),(audio,'wrapper-audio-utilities.c'),(chat,'wrapper-chat-abi.c'),(aux,'ruby-wrapper-aux.c'),(parakeet,'wrapper-parakeet-legacy.c'),(titanet,'wrapper-titanet-legacy.c'),(db,'wrapper-speaker-db.c'),(speaker,'wrapper-speaker-primitives.c'),(stream,'wrapper-stream-lifetime.c'),(text,'wrapper-text-processors.c'),(mic,'wrapper-microphone.c'),(vad,'wrapper-vad-segments.c')]:
                subprocess.run(['cc','-fPIC','-DCRISPASR_PROBE_UNIQUE_SESSIONS','-DCRISPASR_PROBE_RUBY_STREAM_OWNERS','-DCRISPASR_PROBE_REAL_TEXT_PROCESSORS','-DCRISPASR_PROBE_LITERAL_OPEN_LAYOUT','-I'+str(ROOT/'include'),'-I'+str(ROOT/'src'),'-I'+str(ROOT/'ggml/include'),'-I'+str(tmp),'-c',str(ROOT/'tests/fixtures'/source),'-o',str(target)],check=True)
            def symbols(args):
                return {line.split()[-1] for line in subprocess.check_output(['nm',*args],text=True).splitlines() if line.split()}
            needed = symbols(['-u',str(obj)])
            supplied = symbols(['-g','--defined-only',str(probe),str(metadata),str(audio),str(chat),str(aux),str(parakeet),str(titanet),str(db),str(speaker),str(stream),str(text),str(mic),str(vad)])
            unused = sorted(n for n in needed-supplied if n.startswith('crispasr_'))
            stub = tmp/'unused.c'
            stub.write_text('#include <stdlib.h>\n'+''.join('void '+n+'(void){abort();}\n' for n in unused))
            init = tmp/'init.c'
            init.write_text('#include <ruby.h>\nvoid init_ruby_crispasr_session(VALUE*);\nvoid Init_wrapper_probe(void){VALUE m=rb_define_module("Whisper");init_ruby_crispasr_session(&m);}\n')
            lib = tmp/'wrapper_probe.so'
            subprocess.run(['cc','-shared',*flags,str(obj),str(probe),str(metadata),str(audio),str(chat),str(aux),str(parakeet),str(titanet),str(db),str(speaker),str(stream),str(text),str(mic),str(vad),str(stub),str(init),'-lpthread','-o',str(lib)],check=True)
            program = tmp/'probe.rb';program.write_text(PROGRAM)
            result = subprocess.run([RUBY,str(program),str(lib)],text=True,capture_output=True,env=env,
                preexec_fn=(lambda: __import__('resource').setrlimit(__import__('resource').RLIMIT_CORE,(0,0))) if os.name=='posix' else None)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS Ruby ABI:',result.stdout)
            print(result.stdout.strip())

if __name__ == '__main__':
    unittest.main()
