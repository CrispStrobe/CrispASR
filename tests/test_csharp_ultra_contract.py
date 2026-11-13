"""Execute the production C# Ultra wrapper against a recording native ABI."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
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
 if(mode==2 || strcmp(base,"base.gguf") || strcmp(head,"head.gguf") || strcmp(text,"你好") || n!=3 || threads!=7 || pcm[0]!=1 || pcm[1]!=3 || pcm[2]!=5) return 0;
 strcpy(result.text,text); return &result;
}
int crispasr_ultra_align_result_n_words(const crispasr_ultra_align_result* r) { return mode==1?0:mode==3?-1:1; }
const char* crispasr_ultra_align_result_word_text(const crispasr_ultra_align_result* r,int i) { return mode==4?0:r->text; }
double crispasr_ultra_align_result_word_start_ms(const crispasr_ultra_align_result* r,int i) { return 12.345; }
double crispasr_ultra_align_result_word_end_ms(const crispasr_ultra_align_result* r,int i) { return 67.891; }
void crispasr_ultra_align_result_free(crispasr_ultra_align_result* r) { ++frees; strcpy(r->text,"freed"); }
'''
PROGRAM = r'''
using System;
using System.Runtime.InteropServices;
using CrispASR;
class Probe {
 [DllImport("crispasr")] static extern void probe_mode(int value);
 [DllImport("crispasr")] static extern int probe_calls();
 [DllImport("crispasr")] static extern int probe_frees();
 static int checks;
 static void Check(bool value) { ++checks; if(!value) throw new Exception("Check "+checks); }
 static void Throws<T>(Action action) where T:Exception { try { action(); } catch(T) { ++checks; return; } throw new Exception("Expected "+typeof(T)); }
 static AlignedWord[] Align() => Session.AlignWordsUltra("base.gguf","head.gguf","你好",new float[]{1,3,5},7);
 static void Main(string[] args) {
  NativeLibraryResolver.Install();
  if(args[0]=="transport") {
   var words=Align(); Check(words.Length==1); Check(words[0].Text=="你好");
   Check(Math.Abs(words[0].T0-.012345)<1e-12); Check(Math.Abs(words[0].T1-.067891)<1e-12);
   Check(words[0].Characters.Length==0); Check(probe_calls()==1); Check(probe_frees()==1);
  } else if(args[0]=="results") {
   probe_mode(1); Check(Align().Length==0); Check(probe_frees()==1);
   probe_mode(2); Throws<InvalidOperationException>(()=>Align()); Check(probe_frees()==1);
   probe_mode(3); Throws<InvalidOperationException>(()=>Align()); Check(probe_frees()==2);
   probe_mode(4); Check(Align()[0].Text==""); Check(probe_frees()==3);
  } else {
   foreach(var bad in new string[]{null!,"","bad\0path"}) {
    Throws<ArgumentException>(()=>Session.AlignWordsUltra(bad,"head.gguf","你好",new float[]{1}));
    Throws<ArgumentException>(()=>Session.AlignWordsUltra("base.gguf",bad,"你好",new float[]{1}));
    Throws<ArgumentException>(()=>Session.AlignWordsUltra("base.gguf","head.gguf",bad,new float[]{1}));
   }
   Throws<ArgumentException>(()=>Session.AlignWordsUltra("base.gguf","head.gguf","你好",null!));
   Throws<ArgumentException>(()=>Session.AlignWordsUltra("base.gguf","head.gguf","你好",Array.Empty<float>()));
   Throws<ArgumentException>(()=>Session.AlignWordsUltra("base.gguf","head.gguf","你好",new float[1920001]));
   Check(probe_calls()==0); Check(probe_frees()==0);
  }
  Console.WriteLine("PASS C# Ultra "+args[0]+": "+checks+" checks");
 }
}
'''

@unittest.skipUnless(shutil.which('dotnet') and shutil.which('cc'), '.NET SDK and C compiler required')
class CSharpUltraContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='csharp-ultra-')
        cls.path = path = Path(cls.temp.name)
        (path/'probe.c').write_text(PROBE)
        (path/'Program.cs').write_text(PROGRAM)
        (path/'Probe.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>
<OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><Nullable>enable</Nullable>
<AllowUnsafeBlocks>true</AllowUnsafeBlocks><EnableDefaultCompileItems>false</EnableDefaultCompileItems>
</PropertyGroup><ItemGroup><Compile Include="Program.cs"/><Compile Include="{ROOT/'bindings/csharp/CrispASR'}/*.cs"/></ItemGroup></Project>''')
        cls.env = dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT='1', DOTNET_NOLOGO='1', CRISPASR_LIBRARY_PATH=str(path/'libcrispasr.so'))
        build = subprocess.run(['dotnet','build',str(path/'Probe.csproj'),'-c','Release','--nologo','-v','quiet','-m:1','-p:UseSharedCompilation=false','--disable-build-servers'],env=cls.env,capture_output=True,text=True)
        if build.returncode:
            cls.temp.cleanup()
            raise AssertionError(build.stdout+build.stderr)
        subprocess.run(['cc','-shared','-fPIC','-std=c11','-Wall','-Werror','-I'+str(ROOT/'include'),str(path/'probe.c'),'-o',str(path/'libcrispasr.so')],check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_probe(self, mode):
        result = subprocess.run(['dotnet',str(self.path/'bin/Release/net8.0/Probe.dll'),mode],env=self.env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('PASS C# Ultra '+mode,result.stdout)
        print(result.stdout.strip())

    def test_transport(self):
        self.run_probe('transport')

    def test_results(self):
        self.run_probe('results')

    def test_invalid_input(self):
        self.run_probe('invalid')

if __name__ == '__main__':
    unittest.main()
