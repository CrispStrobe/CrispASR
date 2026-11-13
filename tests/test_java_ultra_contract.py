"""Execute the production Java Ultra wrapper against a recording native ABI."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
JAR = os.environ.get('CRISPASR_JNA_JAR')
JAVA = os.environ.get('JAVA') or shutil.which('java')
PROBE = r'''
#include "crispasr_session.h"
#include <string.h>
struct crispasr_ultra_align_result { char text[16]; };
static struct crispasr_ultra_align_result result;
static int mode, calls, frees;
void probe_mode(int value) { mode=value; }
int probe_calls(void) { return calls; }
int probe_frees(void) { return frees; }
crispasr_ultra_align_result* crispasr_align_words_ultra_abi(const char* base,const char* head,const char* text,const float* pcm,int32_t n,int32_t threads) {
 ++calls;
 if(mode==2 || strcmp(base,"基础.gguf") || strcmp(head,"头.gguf") || strcmp(text,"你好") || n!=3 || threads!=7 || pcm[0]!=1 || pcm[1]!=3 || pcm[2]!=5) return 0;
 strcpy(result.text,text); return &result;
}
int crispasr_ultra_align_result_n_words(const crispasr_ultra_align_result* r) { return mode==1?0:mode==3?-1:1; }
const char* crispasr_ultra_align_result_word_text(const crispasr_ultra_align_result* r,int i) { return mode==4?0:r->text; }
double crispasr_ultra_align_result_word_start_ms(const crispasr_ultra_align_result* r,int i) { return 12.345; }
double crispasr_ultra_align_result_word_end_ms(const crispasr_ultra_align_result* r,int i) { return 67.891; }
void crispasr_ultra_align_result_free(crispasr_ultra_align_result* r) { ++frees; strcpy(r->text,"freed"); }
'''
PROGRAM = r'''
import io.github.ggerganov.whispercpp.CrispasrSession;
import com.sun.jna.*;
public class JavaUltraProbe {
 public interface Probe extends Library {
  Probe INSTANCE=Native.load("crispasr",Probe.class);
  void probe_mode(int value); int probe_calls(); int probe_frees();
 }
 static int checks;
 static void check(boolean value) { ++checks; if(!value) throw new AssertionError("Check "+checks); }
 static void reject(Class<? extends Throwable> cls,Runnable action) {
  try { action.run(); } catch(Throwable error) { if(cls.isInstance(error)){++checks; return;} throw error; }
  throw new AssertionError("Expected "+cls.getName());
 }
 static CrispasrSession.UltraAlignedWord[] align() {
  return CrispasrSession.alignWordsUltra("基础.gguf","头.gguf","你好",new float[]{1,3,5},7);
 }
 public static void main(String[] args) {
  if(args[0].equals("transport")) {
   CrispasrSession.UltraAlignedWord[] words=align(); check(words.length==1); check(words[0].text.equals("你好"));
   check(Math.abs(words[0].start-.012345)<1e-12); check(Math.abs(words[0].end-.067891)<1e-12);
   check(Probe.INSTANCE.probe_calls()==1); check(Probe.INSTANCE.probe_frees()==1);
  } else if(args[0].equals("results")) {
   Probe.INSTANCE.probe_mode(1); check(align().length==0); check(Probe.INSTANCE.probe_frees()==1);
   Probe.INSTANCE.probe_mode(2); reject(IllegalStateException.class,()->align()); check(Probe.INSTANCE.probe_frees()==1);
   Probe.INSTANCE.probe_mode(3); reject(IllegalStateException.class,()->align()); check(Probe.INSTANCE.probe_frees()==2);
   Probe.INSTANCE.probe_mode(4); check(align()[0].text.equals("")); check(Probe.INSTANCE.probe_frees()==3);
  } else {
   for(String bad:new String[]{null,"","bad\u0000path"}) {
    reject(IllegalArgumentException.class,()->CrispasrSession.alignWordsUltra(bad,"head","text",new float[]{1},1));
    reject(IllegalArgumentException.class,()->CrispasrSession.alignWordsUltra("base",bad,"text",new float[]{1},1));
    reject(IllegalArgumentException.class,()->CrispasrSession.alignWordsUltra("base","head",bad,new float[]{1},1));
   }
   reject(IllegalArgumentException.class,()->CrispasrSession.alignWordsUltra("base","head","text",null,1));
   reject(IllegalArgumentException.class,()->CrispasrSession.alignWordsUltra("base","head","text",new float[0],1));
   reject(IllegalArgumentException.class,()->CrispasrSession.alignWordsUltra("base","head","text",new float[1920001],1));
   check(Probe.INSTANCE.probe_calls()==0); check(Probe.INSTANCE.probe_frees()==0);
  }
  System.out.println("PASS Java Ultra "+args[0]+": "+checks+" checks");
 }
}
'''

@unittest.skipUnless(JAVA and JAR and Path(JAR).is_file(), 'Java runtime/compiler and CRISPASR_JNA_JAR required')
class JavaUltraContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='java-ultra-')
        cls.path = path = Path(cls.temp.name)
        (path/'probe.c').write_text(PROBE)
        (path/'JavaUltraProbe.java').write_text(PROGRAM)
        sources = [ROOT/'bindings/java/src/main/java/io/github/ggerganov/whispercpp'/name
                   for name in ('CrispasrSession.java', 'CrispasrSessionTasks.java', 'CrispasrUtilities.java')]
        javac = shutil.which('javac')
        compiler = [javac] if javac else [JAVA, '-m', 'jdk.compiler/com.sun.tools.javac.Main']
        build = subprocess.run(compiler+['-encoding', 'UTF-8', '-cp', JAR, '-d', str(path), *map(str, sources), str(path/'JavaUltraProbe.java')], capture_output=True, text=True)
        if build.returncode:
            cls.temp.cleanup()
            raise AssertionError(build.stdout+build.stderr)
        subprocess.run(['cc', '-shared', '-fPIC', '-std=c11', '-Wall', '-Werror', '-I'+str(ROOT/'include'), str(path/'probe.c'), '-o', str(path/'libcrispasr.so')], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_probe(self, mode):
        # Prove the wrapper's UTF-8 contract even when JNA defaults to a legacy encoding.
        for encoding in ('UTF-8', 'ISO-8859-1'):
            result = subprocess.run([JAVA, '-Djna.tmpdir='+str(self.path), '-Djna.library.path='+str(self.path), '-Djna.encoding='+encoding, '-cp', str(self.path)+os.pathsep+JAR, 'JavaUltraProbe', mode], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('PASS Java Ultra '+mode, result.stdout)
            print(result.stdout.strip())

    def test_transport(self):
        self.run_probe('transport')

    def test_results(self):
        self.run_probe('results')

    def test_invalid_input(self):
        self.run_probe('invalid')

if __name__ == '__main__':
    unittest.main()
