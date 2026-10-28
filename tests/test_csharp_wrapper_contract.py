"""Model-free C# operations against a compiled public-header ABI probe."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

PROBE = r'''
#include "crispasr_session.h"
static int calls, mode;
static int setter_rc, setter_calls;
void probe_setter_rc(int rc) { setter_rc=rc; }
int probe_setter_calls(void) { return setter_calls; }
int crispasr_session_set_punc_model(crispasr_session* s,const char* model) { ++setter_calls;return setter_rc; }
int crispasr_session_set_voice(crispasr_session* s,const char* path,const char* text) { ++setter_calls;return setter_rc; }
int crispasr_session_set_speaker_id(crispasr_session* s,int id) { ++setter_calls;return setter_rc; }
int crispasr_session_set_top_k(crispasr_session* s,int k) { ++setter_calls;return setter_rc; }

static float audio[] = {1,2,3,4};
crispasr_session* crispasr_session_open(const char* p, int threads) { return (crispasr_session*)1; }
void crispasr_session_close(crispasr_session* s) {}
int crispasr_session_convert_content_dim(crispasr_session* s) { return 2; }
int crispasr_session_convert_n_speakers(crispasr_session* s) { return 1; }
int crispasr_session_convert_sample_rate(crispasr_session* s) { return 40000; }
#ifndef OLD_PROBE
int crispasr_session_convert_inter_channels(crispasr_session* s) { return mode ? 2147483647 : 2; }
int crispasr_session_convert_upsample_product(crispasr_session* s) { return 4; }
#endif
int crispasr_session_convert(crispasr_session* s, const float* content, int n, const float* f0,
                             int speaker, const float* zp, const float* sine) {
    ++calls;
    return n == 3 && speaker == 0 ? 4 : -1;
}
const float* crispasr_session_convert_audio(crispasr_session* s, int* n) { *n=4;return audio; }
int probe_calls(void) { return calls; }
void probe_mode(int m) { mode=m; }
void probe_mutate(void) { audio[0]=999; }
'''

PROBE += r"""
#include <stdlib.h>
#include <string.h>
#include "wrapper-vad-arguments.h"
static char token0[3],token1[2];
static int frees,owned;
static crispasr_token_callback token_cb;
int probe_frees(void) { return frees; }
int probe_owned(void) { return owned; }
static crispasr_session_result* result(void) {
    token0[0]=(char)0xe6;token0[1]=(char)0x97;token0[2]=0;
    token1[0]=(char)0xa5;token1[1]=0;
    if(token_cb)token_cb(token0,0,NULL);
    return (crispasr_session_result*)1;
}
crispasr_session_result* crispasr_session_transcribe_lang(crispasr_session* s,const float* p,int n,const char* lang) { return result(); }
crispasr_session_result* crispasr_session_transcribe_chunked_lang(crispasr_session* s,const float* p,int n,int c,int overlap,const char* lang) { return result(); }
crispasr_session_result* crispasr_session_transcribe_vad(crispasr_session* s,const float* p,int n,int sr,const char* model,const crispasr_vad_abi_opts* opts) { return probe_vad_arguments(p,n,sr,model,opts,NULL) ? result():NULL; }
crispasr_session_result* crispasr_session_transcribe_vad_lang(crispasr_session* s,const float* p,int n,int sr,const char* model,const crispasr_vad_abi_opts* opts,const char* lang) { return probe_vad_arguments(p,n,sr,model,opts,lang) ? result():NULL; }
int crispasr_session_set_return_logits(crispasr_session* s,int on) { return 0; }
int crispasr_session_result_n_logit_frames(crispasr_session_result* r) { return 0; }
int crispasr_session_result_n_logit_vocab(crispasr_session_result* r) { return 0; }
const float* crispasr_session_result_logits(crispasr_session_result* r) { return NULL; }
int crispasr_session_result_n_segments(crispasr_session_result* r) { return 1; }
const char* crispasr_session_result_segment_text(crispasr_session_result* r,int i) { return "Japanese"; }
int64_t crispasr_session_result_segment_t0(crispasr_session_result* r,int i) { return 123; }
int64_t crispasr_session_result_segment_t1(crispasr_session_result* r,int i) { return 456; }
float crispasr_session_result_segment_no_speech_prob(crispasr_session_result* r,int i) { return .125f; }
const char* crispasr_session_result_segment_speaker(crispasr_session_result* r,int i) { return "speaker"; }
int crispasr_session_result_n_words(crispasr_session_result* r,int i) { return 1; }
const char* crispasr_session_result_word_text(crispasr_session_result* r,int i,int j) { return "word"; }
int64_t crispasr_session_result_word_t0(crispasr_session_result* r,int i,int j) { return -1; }
int64_t crispasr_session_result_word_t1(crispasr_session_result* r,int i,int j) { return -1; }
float crispasr_session_result_word_p(crispasr_session_result* r,int i,int j) { return .75f; }
int crispasr_session_result_word_n_alts(crispasr_session_result* r,int i,int j) { return 1; }
const char* crispasr_session_result_word_alt_text(crispasr_session_result* r,int i,int j,int k) { return "alternative"; }
float crispasr_session_result_word_alt_p(crispasr_session_result* r,int i,int j,int k) { return .25f; }
void crispasr_session_result_free(crispasr_session_result* r) { ++frees;token0[0]='X'; }
#ifndef OLD_PROBE
int crispasr_session_result_n_tokens(crispasr_session_result* r,int i) { return 2; }
const char* crispasr_session_result_token_text(crispasr_session_result* r,int i,int j) { return j ? token1:token0; }
int64_t crispasr_session_result_token_t0(crispasr_session_result* r,int i,int j) { return j ? 123:-1; }
int64_t crispasr_session_result_token_t1(crispasr_session_result* r,int i,int j) { return j ? 124:-1; }
float crispasr_session_result_token_p(crispasr_session_result* r,int i,int j) { return .75f; }
int crispasr_session_set_voice_samples(crispasr_session* s,const float* p,int32_t n,int32_t sr,const char* text) { return n==3 && sr==24000 ? 0:-1; }
int crispasr_session_set_tts_cfg_scale(crispasr_session* s,float scale) { return scale==1.5f ? 0:-2; }
int crispasr_session_tada_set_makeref_models(crispasr_session* s,const char* enc,const char* align) { return enc==NULL && align==NULL ? 0:-1; }
const char* crispasr_session_last_synth_error(crispasr_session* s) { return "probe error"; }
const char* crispasr_session_disclaimer_text(void) { return "AI generated"; }
float* crispasr_session_get_disclaimer_pcm(crispasr_session* s,int* n) { float* p=malloc(8);p[0]=.25f;p[1]=.5f;*n=2;++owned;return p; }
void crispasr_pcm_free(float* p) { if(p){--owned;free(p);} }
int crispasr_session_synthesize_streaming(crispasr_session* s,const char* text,crispasr_pcm_stream_cb cb,void* ud) {
    float pcm[2]={.25f,.5f};cb(pcm,2,0,ud);pcm[0]=.75f;cb(pcm,2,1,ud);return 0;
}
void crispasr_session_set_token_callback(crispasr_session* s,crispasr_token_callback cb,void* ud) { token_cb=cb; }
#endif
"""

PROGRAM = r'''
using System;
using System.Runtime.InteropServices;
using CrispASR;
class Program {
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)] static extern int probe_calls();
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)] static extern void probe_mode(int n);
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)] static extern void probe_mutate();
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)] static extern int probe_frees();
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)] static extern int probe_owned();
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)]static extern void probe_vad_expect(int n);
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)]static extern int probe_vad_calls();
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)]static extern void probe_setter_rc(int rc);
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)]static extern int probe_setter_calls();
    static int assertions;
    static void Equal<T>(T a,T b) { if(!Equals(a,b))throw new Exception($"{a} != {b}");assertions++; }
    static void Reject<T>(Action action) where T:Exception {
        int before=probe_calls();
        try { action(); } catch(T) { Equal(probe_calls(),before);return; }
        throw new Exception("missing "+typeof(T).Name+" before native conversion");
    }
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)]static extern int probe_stream_closes();
    static void Main(string[] args) {
        using var s=Session.Open("probe");
        foreach(int rc in new[]{0,-1,-2,-3,1}) {
            probe_setter_rc(rc);
            foreach(Action setter in new Action[]{()=>s.SetPuncModel("missing.gguf"),
                                                   ()=>s.SetVoice("reference.wav"),()=>s.SetSpeakerId(-1)}) {
                int before=probe_setter_calls();
                if(rc==0) setter();
                else {
                    try { setter();throw new Exception($"setter swallowed rc={rc}"); }
                    catch(InvalidOperationException error) { Equal(error.Message.Contains($"rc={rc}"),true); }
                }
                Equal(probe_setter_calls(),before+1);
            }
        }
        probe_setter_rc(-2);s.SetTopK(10); // unsupported tuning remains a compatible no-op
        probe_setter_rc(-1);Reject<InvalidOperationException>(()=>s.SetTopK(10));
        probe_setter_rc(0);
        var content=new float[6];var f0=new float[3];
        if(args.Length>0 && args[0]=="old") {
            Equal(s.Convert(content,f0).Length,4);
            Reject<NotSupportedException>(()=>s.Convert(content,f0,noiseZp:new float[6]));
            Equal(s.Transcribe(f0)[0].Tokens.Length,0);
        } else {
            Reject<ArgumentException>(()=>s.Convert(content,f0,noiseZp:new float[5]));
            Reject<ArgumentException>(()=>s.Convert(content,f0,noiseSine:new float[11]));
            Reject<ArgumentException>(()=>s.Convert(content,f0,noiseZp:new float[7]));
            Reject<ArgumentOutOfRangeException>(()=>s.Convert(content,f0,speakerId:1));
            var audio=s.Convert(content,f0,noiseZp:new float[6],noiseSine:new float[12]);
            Equal(audio.Length,4);probe_mutate();Equal(audio[0],1f);
            probe_mode(1);
            Reject<OverflowException>(()=>s.Convert(content,f0,noiseZp:new float[6]));
            Equal(s.Convert(content,f0).Length,4); // random draws need no dimension probe
            foreach(var seg in new[]{s.Transcribe(f0)[0],s.TranscribeChunked(f0)[0],
                                      s.TranscribeVad(f0,16000,"vad")[0],s.TranscribeWithLogits(f0).Segments[0]}) {
                Equal(seg.T0,1.23);Equal(seg.Words[0].T0,-1d);Equal(seg.Words[0].HasTimestamps,false);
                Equal(seg.Speaker,"speaker");Equal(seg.Words[0].Alts[0].Text,"alternative");
                Equal(seg.Tokens.Length,2);Equal(seg.Tokens[0].RawText[0],(byte)0xe6);
                Equal(seg.Tokens[0].HasTimestamps,false);Equal(seg.Tokens[1].T0,1.23);
                var bytes=new byte[]{seg.Tokens[0].RawText[0],seg.Tokens[0].RawText[1],seg.Tokens[1].RawText[0]};
                Equal(System.Text.Encoding.UTF8.GetString(bytes),"日");
            }
            var vad = new VadOptions { Threshold=.375f, MinSpeechDurationMs=123, MinSilenceDurationMs=456,
                SpeechPadMs=78, ChunkSeconds=9, NThreads=2 };
            probe_vad_expect(1);Equal(s.TranscribeVad(f0,16000,"vad","ja",vad)[0].Tokens.Length,2);
            probe_vad_expect(2);Equal(s.TranscribeVad(f0,16000,"vad","ja")[0].Tokens.Length,2);
            int vadCalls=probe_vad_calls();
            Reject<ArgumentException>(()=>s.TranscribeVad(Array.Empty<float>(),16000,"vad","ja",vad));
            Reject<ArgumentNullException>(()=>s.TranscribeVad(null!,16000,"vad","ja",vad));
            Reject<ArgumentOutOfRangeException>(()=>s.TranscribeVad(f0,0,"vad","ja",vad));
            vad.Threshold=float.NaN;Reject<ArgumentOutOfRangeException>(()=>s.TranscribeVad(f0,16000,"vad","ja",vad));
            vad.Threshold=.375f;vad.NThreads=0;Reject<ArgumentOutOfRangeException>(()=>s.TranscribeVad(f0,16000,"vad","ja",vad));
            Equal(probe_vad_calls(),vadCalls);vad.NThreads=2;
            probe_vad_expect(3);Reject<InvalidOperationException>(()=>s.TranscribeVad(f0,16000,"vad","ja",vad));
            probe_vad_expect(1);int vadFrees=probe_frees();
            s.SetTokenCallback((_,__)=>throw new ArgumentException("vad callback"));
            Reject<ArgumentException>(()=>s.TranscribeVad(f0,16000,"vad","ja",vad));
            Equal(probe_frees(),vadFrees+1);s.SetTokenCallback(null);probe_vad_expect(0);
            s.SetVoiceSamples(f0,24000);s.SetTtsCfgScale(1.5f);s.TadaSetMakerefModels();
            Reject<InvalidOperationException>(()=>s.SetTtsCfgScale(2));
            Equal(Session.DisclaimerText,"AI generated");Equal(s.GetDisclaimerPcm()[0],.25f);Equal(probe_owned(),0);
            var chunks=new System.Collections.Generic.List<float[]>();
            s.SynthesizeStreaming("one. two.",(pcm,final)=>chunks.Add(pcm));
            Equal(chunks.Count,2);Equal(chunks[0][0],.25f);Equal(chunks[1][0],.75f);
            Reject<InvalidOperationException>(()=>s.SynthesizeStreaming("test",(_,__)=>s.Dispose()));
            int before=probe_frees();
            s.SetTokenCallback((_,__)=>throw new ArgumentException("callback"));
            Reject<ArgumentException>(()=>s.Transcribe(f0));
            Equal(probe_frees(),before+1);s.SetTokenCallback(null);
        }
        var owner=Session.Open("probe");var stream=owner.StreamOpen();
        Equal(stream.Feed(new float[]{1,2,3}),1);stream.SetLiveDecode(true);Equal(stream.GetText().Text,"ok");
        owner.Dispose();Equal(probe_stream_closes(),1);Reject<ObjectDisposedException>(()=>stream.Feed(new float[]{1}));stream.Dispose();
        Console.WriteLine($"PASS C# conversion ABI: {assertions} assertions");
    }
}
'''


@unittest.skipUnless(shutil.which('dotnet') and shutil.which('cc'), 'dotnet SDK and C compiler required')
class CSharpWrapperContract(unittest.TestCase):
    def test_replay_buffers_and_older_library(self):
        with tempfile.TemporaryDirectory(prefix='csharp-contract-') as temporary:
            path = Path(temporary)
            src=(ROOT/'src/crispasr_c_api.cpp').read_text()
            start=src.index('struct crispasr_vad_abi_opts {')
            (path/'runtime-vad-layout.h').write_text(src[start:src.index('};',start)+2])
            (path/'probe.c').write_text(PROBE)
            (path/'Program.cs').write_text(PROGRAM)
            (path/'Probe.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>
<OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><Nullable>enable</Nullable>
<AllowUnsafeBlocks>true</AllowUnsafeBlocks><EnableDefaultCompileItems>false</EnableDefaultCompileItems>
</PropertyGroup><ItemGroup><Compile Include="Program.cs"/>
<Compile Include="{ROOT / 'bindings/csharp/CrispASR'}/*.cs"/></ItemGroup></Project>''')
            env=dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT='1', DOTNET_NOLOGO='1')
            subprocess.run(['dotnet','build',str(path/'Probe.csproj'),'-c','Release','--nologo','-v','quiet','-m:1','-p:UseSharedCompilation=false','--disable-build-servers'],
                           env=env,check=True,capture_output=True,text=True)
            for old in (False,True):
                library=path/('old.so' if old else 'probe.so')
                subprocess.run(['cc','-shared','-fPIC','-std=c11','-Werror',
                                '-I'+str(ROOT/'include'), '-I'+str(ROOT/'tests/fixtures'), '-I'+str(path), *(['-DOLD_PROBE'] if old else []),
                                str(path/'probe.c'),str(ROOT/'tests/fixtures/wrapper-stream-lifetime.c'),'-o',str(library)],check=True)
                result=subprocess.run(['dotnet',str(path/'bin/Release/net8.0/Probe.dll'),
                                       *(['old'] if old else [])],
                                      env=dict(env,CRISPASR_LIBRARY_PATH=str(library)),
                                      capture_output=True,text=True,
                                      preexec_fn=(lambda: __import__('resource').setrlimit(__import__('resource').RLIMIT_CORE,(0,0))) if os.name=='posix' else None)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('PASS C# conversion ABI:',result.stdout)
                print(result.stdout.strip())


AUX_PROGRAM = r'''
using System;
using System.Runtime.InteropServices;
using CrispASR;
class AuxiliaryProbe {
    [DllImport("crispasr")]static extern int probe_utility_live();
    [DllImport("crispasr")]static extern void probe_utility_mode(int mode);
    [DllImport("crispasr")]static extern int probe_live();
    static int checks;
    static void Eq<T>(T a,T b){if(!object.Equals(a,b))throw new Exception($"{a} != {b}");checks++;}
    static void Reject<T>(Action f)where T:Exception{try{f();}catch(T){checks++;return;}throw new Exception("Expected "+typeof(T));}
    static void Main(string[] args){
        string path=args[0];var pcm=new float[]{1,3,5};
        Eq(Audio.Load(path)[0],.25f);Eq(Audio.Load(path,24000,out int rate)[2],.75f);Eq(rate,24000);
        Eq(Audio.LoadStereo(path).Samples.Length,6);probe_utility_mode(3);Eq(Audio.LoadStereo(path).SourceChannels,1);
        Eq(Audio.Resample(Array.Empty<float>(),16000,24000).Length,0);Eq(Audio.Resample(pcm,16000,24000)[0],.25f);
        Eq(Audio.PcmToWav(pcm)[1],(byte)0);Eq(Audio.SignC2pa(new byte[]{1,0,2})[1],(byte)0);
        probe_utility_mode(2);Reject<InvalidOperationException>(()=>Audio.Load(path));Reject<InvalidOperationException>(()=>Audio.LoadStereo(path));Reject<InvalidOperationException>(()=>Audio.Resample(pcm,16000,24000));Reject<InvalidOperationException>(()=>Audio.SignC2pa(new byte[]{1}));
        Eq(probe_utility_live(),0);probe_utility_mode(0);
        Eq(RuntimeInfo.Version,"probe");Eq(RuntimeInfo.BackendCapabilities("foo")[1],"tts");Eq(RuntimeInfo.AllBackendCapabilities().Contains("foo"),true);
        using(var p=new TextProcessor("probe")){Eq(p.Process("x"),"Grüße");}
        using(var p=new TextProcessor("probe",TextProcessorKind.PunctuationCapitalizationSegmentation)){Eq(p.Process("x"),"Hello.");}
        using(var session=Session.Open("probe")){Eq(session.Backend,"probe");Eq(session.ScoreTexts(pcm,new[]{"one","two"}).TokenCounts[1],3);}
        Eq(probe_live(),0);Eq(probe_utility_live(),0);
        Console.WriteLine($"PASS C# utilities: {checks} assertions");
    }
}
'''

@unittest.skipUnless(shutil.which('dotnet') and shutil.which('cc'), '.NET SDK and C compiler required')
class CSharpUtilityContract(unittest.TestCase):
    def test_audio_container_processors_and_scoring(self):
        with tempfile.TemporaryDirectory(prefix='csharp-utility-contract-') as tmp:
            path=Path(tmp)
            sources=ROOT/'bindings/csharp/CrispASR'
            (path/'Probe.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><Nullable>enable</Nullable><EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup><ItemGroup><Compile Include="{sources}/*.cs"/><Compile Include="Program.cs"/></ItemGroup></Project>''')
            (path/'Program.cs').write_text(AUX_PROGRAM)
            subprocess.run(['dotnet','build',str(path/'Probe.csproj'),'-c','Release','--nologo','-v','quiet'],check=True,capture_output=True,text=True)
            library=path/'libcrispasr.so'
            subprocess.run(['cc','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),'-I'+str(ROOT/'ggml/include'),
                *[str(ROOT/'tests/fixtures'/name) for name in ['python-wrapper-abi.c','wrapper-session-metadata.c','wrapper-audio-utilities.c']],'-o',str(library)],check=True)
            audio=path/'audio.wav';audio.write_bytes(b'probe')
            env=os.environ.copy();env['LD_LIBRARY_PATH']=str(path)+os.pathsep+env.get('LD_LIBRARY_PATH','')
            result=subprocess.run(['dotnet',str(path/'bin/Release/net8.0/Probe.dll'),str(audio)],capture_output=True,text=True,env=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('PASS C# utilities:',result.stdout);print(result.stdout.strip())


CHAT_PROGRAM=r'''
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using CrispASR;
class ChatProbe {
    [DllImport("crispasr")]static extern int probe_chat_sessions();
    [DllImport("crispasr")]static extern int probe_chat_strings();
    [DllImport("crispasr")]static extern int probe_chat_resets();
    [DllImport("crispasr")]static extern int probe_chat_open_size();
    [DllImport("crispasr")]static extern int probe_chat_generate_size();
    static int checks;
    static void Eq<T>(T a,T b){if(!object.Equals(a,b))throw new Exception($"{a} != {b}");checks++;}
    static void Reject<T>(Action f)where T:Exception{try{f();}catch(T){checks++;return;}throw new Exception("Expected "+typeof(T));}
    static void Main(){
        Eq(Marshal.SizeOf<ChatNative.OpenParams>(),probe_chat_open_size());Eq(Marshal.SizeOf<ChatNative.GenerateParams>(),probe_chat_generate_size());
        var messages=new[]{new ChatMessage("system","prompt"),new ChatMessage("user","日本語")};
        Reject<ChatException>(()=>ChatSession.Open("fail"));Eq(probe_chat_sessions(),0);
        try{ChatSession.Open("fail");}catch(ChatException e){Eq(e.NativeCode,3);Eq(e.Message,"cannot open");}
        var options=new ChatOptions{ContextSize=321,GpuLayers=0,UseMmap=false,Template="gemma"};
        using(var chat=ChatSession.Open("probe",options)){
            Eq(chat.ContextSize,321);Eq(chat.TemplateName,"probe-template");Eq(chat.CountTokens(messages),9);
            Eq(chat.Generate(messages),"Grüße");Eq(probe_chat_strings(),0);
            Eq(chat.Generate(messages,new ChatGenerateOptions{MaxTokens=7,Seed=uint.MaxValue,Stop=new[]{"STOP"},PrefillOnly=true}),"");
            var chunks=new List<byte[]>();chat.GenerateStreaming(messages,x=>chunks.Add(x));Eq(chunks.Count,2);Eq(chunks[0][0],(byte)0xe6);Eq(System.Text.Encoding.UTF8.GetString(new[]{chunks[0][0],chunks[0][1],chunks[1][0]}),"日");
            Reject<ArgumentException>(()=>chat.GenerateStreaming(messages,x=>throw new ArgumentException("sentinel")));
            Reject<InvalidOperationException>(()=>chat.GenerateStreaming(messages,x=>chat.Dispose()));
            using(var token=new CancellationTokenSource()){token.Cancel();Reject<OperationCanceledException>(()=>chat.Generate(messages,cancellationToken:token.Token));}
            chat.SetAbortCallback(()=>false);Reject<OperationCanceledException>(()=>chat.Generate(messages));
            chat.SetAbortCallback(()=>throw new ArgumentException("abort sentinel"));Reject<ArgumentException>(()=>chat.Generate(messages));chat.SetAbortCallback(null);
            chat.Reset();Eq(probe_chat_resets(),1);
        }
        Eq(probe_chat_sessions(),0);Eq(probe_chat_strings(),0);Eq(ChatSession.MemoryEstimate("probe"),(1UL<<40)+123);Eq(ChatSession.AiDisclosureText,"You are talking to an AI.");
        var closed=ChatSession.Open("probe");closed.Dispose();closed.Dispose();Reject<ObjectDisposedException>(()=>closed.Reset());
        Console.WriteLine($"PASS C# chat: {checks} assertions");
    }
}
'''

@unittest.skipUnless(shutil.which('dotnet') and shutil.which('cc'), '.NET SDK and C compiler required')
class CSharpChatContract(unittest.TestCase):
    def test_chat_layout_lifetime_streaming_and_cancellation(self):
        with tempfile.TemporaryDirectory(prefix='csharp-chat-contract-') as tmp:
            path=Path(tmp);sources=ROOT/'bindings/csharp/CrispASR'
            (path/'Probe.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><Nullable>enable</Nullable><EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup><ItemGroup><Compile Include="{sources}/*.cs"/><Compile Include="Program.cs"/></ItemGroup></Project>''')
            (path/'Program.cs').write_text(CHAT_PROGRAM)
            result=subprocess.run(['dotnet','build',str(path/'Probe.csproj'),'-c','Release','--nologo','-v','quiet'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            subprocess.run(['cc','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),str(ROOT/'tests/fixtures/wrapper-chat-abi.c'),'-o',str(path/'libcrispasr.so')],check=True)
            env=os.environ.copy();env['LD_LIBRARY_PATH']=str(path)+os.pathsep+env.get('LD_LIBRARY_PATH','')
            result=subprocess.run(['dotnet',str(path/'bin/Release/net8.0/Probe.dll')],capture_output=True,text=True,env=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('PASS C# chat:',result.stdout);print(result.stdout.strip())


@unittest.skipUnless(shutil.which('dotnet') and shutil.which('cc'), '.NET SDK and C compiler required')
class CSharpSessionAuxiliaryContract(unittest.TestCase):
    def test_registry_options_speaker_and_diarization_operations(self):
        with tempfile.TemporaryDirectory(prefix='csharp-session-aux-') as tmp:
            path=Path(tmp);sources=ROOT/'bindings/csharp/CrispASR'
            (path/'Probe.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><Nullable>enable</Nullable><EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup><ItemGroup><Compile Include="{sources}/*.cs"/><Compile Include="Program.cs"/></ItemGroup></Project>''')
            (path/'Program.cs').write_text((ROOT/'tests/fixtures/csharp-aux-program.cs').read_text())
            result=subprocess.run(['dotnet','build',str(path/'Probe.csproj'),'-c','Release','--nologo','-v','quiet'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            # Compile the literal native POD layouts, not a second guessed mirror.
            source=(ROOT/'src/crispasr_c_api.cpp').read_text();layouts=[]
            for name in ['crispasr_diarize_seg_abi','crispasr_diarize_turn_abi','crispasr_diarize_opts_abi']:
                start=source.index('struct '+name+' {');end=source.index('};',start)+2;layouts.append(source[start:end])
            (path/'runtime-abi-layout.h').write_text('\n'.join(layouts))
            subprocess.run(['cc','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),'-I'+str(path),
                *[str(ROOT/'tests/fixtures'/name) for name in ['python-wrapper-abi.c','wrapper-session-metadata.c','csharp-wrapper-aux.c']],'-o',str(path/'libcrispasr.so')],check=True)
            env=os.environ.copy();env['LD_LIBRARY_PATH']=str(path)+os.pathsep+env.get('LD_LIBRARY_PATH','')
            result=subprocess.run(['dotnet',str(path/'bin/Release/net8.0/Probe.dll')],capture_output=True,text=True,env=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('PASS C# auxiliary:',result.stdout);print(result.stdout.strip())


@unittest.skipUnless(shutil.which('dotnet') and shutil.which('c++'), '.NET SDK and C++ compiler required')
class CSharpWhisperContract(unittest.TestCase):
    def test_native_pointer_parameters_parallel_dtw_tokens_and_cleanup(self):
        from build_legacy_wrapper_probe import build
        with tempfile.TemporaryDirectory(prefix='csharp-legacy-whisper-') as tmp:
            path=Path(tmp);sources=ROOT/'bindings/csharp/CrispASR'
            (path/'Probe.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><Nullable>enable</Nullable><EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup><ItemGroup><Compile Include="{sources}/*.cs"/><Compile Include="Program.cs"/></ItemGroup></Project>''')
            (path/'Program.cs').write_text((ROOT/'tests/fixtures/csharp-whisper-program.cs').read_text())
            result=subprocess.run(['dotnet','build',str(path/'Probe.csproj'),'-c','Release','--nologo','-v','quiet'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            build(path/'libcrispasr.so')
            env=os.environ.copy();env['LD_LIBRARY_PATH']=str(path)+os.pathsep+env.get('LD_LIBRARY_PATH','')
            result=subprocess.run(['dotnet',str(path/'bin/Release/net8.0/Probe.dll')],capture_output=True,text=True,env=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS C# legacy Whisper:',result.stdout);print(result.stdout.strip())


@unittest.skipUnless(shutil.which('dotnet') and shutil.which('c++'), '.NET SDK and C++ compiler required')
class CSharpParakeetContract(unittest.TestCase):
    def test_legacy_parakeet_offset_copy_and_disposal(self):
        with tempfile.TemporaryDirectory(prefix='csharp-legacy-whisper-') as tmp:
            path=Path(tmp);sources=ROOT/'bindings/csharp/CrispASR'
            (path/'Probe.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><Nullable>enable</Nullable><EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup><ItemGroup><Compile Include="{sources}/*.cs"/><Compile Include="Program.cs"/></ItemGroup></Project>''')
            (path/'Program.cs').write_text((ROOT/'tests/fixtures/csharp-parakeet-program.cs').read_text())
            result=subprocess.run(['dotnet','build',str(path/'Probe.csproj'),'-c','Release','--nologo','-v','quiet'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            subprocess.run(['cc','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),str(ROOT/'tests/fixtures/wrapper-parakeet-legacy.c'),'-o',str(path/'libcrispasr.so')],check=True)
            env=os.environ.copy();env['LD_LIBRARY_PATH']=str(path)+os.pathsep+env.get('LD_LIBRARY_PATH','')
            result=subprocess.run(['dotnet',str(path/'bin/Release/net8.0/Probe.dll')],capture_output=True,text=True,env=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS C# legacy Parakeet:',result.stdout);print(result.stdout.strip())


@unittest.skipUnless(shutil.which('dotnet') and shutil.which('c++'), '.NET SDK and C++ compiler required')
class CSharpTitaNetContract(unittest.TestCase):
    def test_titanet_success_count_failure_and_disposal(self):
        with tempfile.TemporaryDirectory(prefix='csharp-titanet-') as tmp:
            path=Path(tmp);sources=ROOT/'bindings/csharp/CrispASR'
            (path/'Probe.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><Nullable>enable</Nullable><EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup><ItemGroup><Compile Include="{sources}/*.cs"/><Compile Include="Program.cs"/></ItemGroup></Project>''')
            (path/'Program.cs').write_text((ROOT/'tests/fixtures/csharp-titanet-program.cs').read_text())
            result=subprocess.run(['dotnet','build',str(path/'Probe.csproj'),'-c','Release','--nologo','-v','quiet'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            subprocess.run(['cc','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),str(ROOT/'tests/fixtures/wrapper-titanet-legacy.c'),'-o',str(path/'libcrispasr.so')],check=True)
            env=os.environ.copy();env['LD_LIBRARY_PATH']=str(path)+os.pathsep+env.get('LD_LIBRARY_PATH','')
            result=subprocess.run(['dotnet',str(path/'bin/Release/net8.0/Probe.dll')],capture_output=True,text=True,env=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS C# TitaNet:',result.stdout);print(result.stdout.strip())



@unittest.skipUnless(shutil.which('dotnet') and shutil.which('cc'), '.NET SDK and C compiler required')
class CSharpSpeakerDbContract(unittest.TestCase):
    def test_speaker_roster_scores_enrollment_and_disposal(self):
        with tempfile.TemporaryDirectory(prefix='csharp-speaker-db-') as tmp:
            path=Path(tmp);sources=ROOT/'bindings/csharp/CrispASR'
            (path/'Probe.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><Nullable>enable</Nullable><EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup><ItemGroup><Compile Include="{sources}/*.cs"/><Compile Include="Program.cs"/></ItemGroup></Project>''')
            (path/'Program.cs').write_text((ROOT/'tests/fixtures/csharp-speaker-db-program.cs').read_text())
            result=subprocess.run(['dotnet','build',str(path/'Probe.csproj'),'-c','Release','--nologo','-v','quiet'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            subprocess.run(['cc','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),str(ROOT/'tests/fixtures/wrapper-speaker-db.c'),'-o',str(path/'libcrispasr.so')],check=True)
            env=os.environ.copy();env['LD_LIBRARY_PATH']=str(path)+os.pathsep+env.get('LD_LIBRARY_PATH','')
            result=subprocess.run(['dotnet',str(path/'bin/Release/net8.0/Probe.dll')],capture_output=True,text=True,env=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS C# speaker DB:',result.stdout);print(result.stdout.strip())


if __name__=='__main__': unittest.main()
