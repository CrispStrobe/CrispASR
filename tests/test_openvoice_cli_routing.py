"""Execute the actual Melo CLI init body with model-free parameter captures."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(text, signature):
    begin = text.index(signature)
    opening = text.index('{', begin)
    end, depth = opening + 1, 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[begin:end]


class CLIRouting(unittest.TestCase):
    def test_converter_receives_cli_device_choice(self):
        source = Path(os.environ.get('OPENVOICE_CLI_SOURCE',
            ROOT / 'examples/cli/crispasr_backend_melotts.cpp')).read_text()
        init = function(source, 'bool init(const whisper_params& p)').replace(' override', '')
        synth = function(source, 'std::vector<float> synthesize(').replace(' override', '')
        rate = function(source, 'int tts_sample_rate() const').replace(' override', '')
        helper = function((ROOT / 'examples/cli/crispasr_backend_utils.h').read_text(),
                          'inline bool crispasr_backend_should_use_gpu(')
        mock = r'''
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <cassert>
struct whisper_params {
 float tts_speed=1;int n_threads=3,tts_speaker_id=2,seed=42;bool no_prints=true,use_gpu=true;
 std::string gpu_backend="cuda",model,tts_voice,tts_codec_model,cache_dir;
};
struct melotts_context{};struct openvoice2_context{};
struct melotts_params {int n_threads=4,verbosity=1,speaker_id=0;bool use_gpu=false;uint32_t seed=0;};
struct openvoice2_context_params {int n_threads=4,verbosity=1;bool use_gpu=false;};
static melotts_params melo_seen;static openvoice2_context_params ov_seen;
static melotts_context melo;static openvoice2_context ov;
static int conversions=0,speaker=-1;static bool converter_fails=false,reference_ok=true;
void melotts_set_speaker_id(melotts_context*,int id){speaker=id;}
void melotts_set_length_scale(melotts_context*,float){}
int melotts_sample_rate(melotts_context*){return 44100;}
int openvoice2_sample_rate(openvoice2_context*){return 22050;}
int melotts_synthesize(melotts_context*,const char*,float** out,int*){
 *out=(float*)malloc(8*sizeof(float));for(int i=0;i<8;i++)(*out)[i]=i;return 8;}
void melotts_pcm_free(float* pcm){free(pcm);}
const char* melotts_source_speaker(melotts_context*){return "en-us";}
bool openvoice2_set_source_speaker(openvoice2_context*,const char*){return true;}
bool openvoice2_convert(openvoice2_context*,const float*,int,int,const float*,int,int,float** out,int* n){
 ++conversions;if(converter_fails)return false;*out=(float*)malloc(4*sizeof(float));*n=4;return true;}
namespace core_audio {
 std::vector<float> resample_polyphase(const float*,int n,int src,int dst,int=14,float=8.6f){
 assert(src==44100 && dst==22050);return std::vector<float>((n+1)/2);}
}

melotts_params melotts_default_params(){return {};}
openvoice2_context_params openvoice2_context_default_params(){return {};}
melotts_context* melotts_init_from_file(const char*,melotts_params p){melo_seen=p;return &melo;}
openvoice2_context* openvoice2_init_from_file(const char*,openvoice2_context_params p){ov_seen=p;return &ov;}
bool melotts_load_bert(melotts_context*,const char*){return true;}
void openvoice2_set_dump_dir(openvoice2_context*,const char*){}
namespace crispasr_env {const char* get(const char*){return nullptr;}}
bool read_wav_mono(const char*,std::vector<float>& pcm,int& sr){pcm={1,2};sr=22050;return reference_ok;}
'''
        fields = '''melotts_context* ctx_=nullptr;openvoice2_context* ov2_ctx_=nullptr;
std::vector<float> ref_pcm_;int ref_sr_=0,ov2_output_sr_=0;'''
        main = r'''
int main(int argc,char** argv){
 assert(argc==3);std::string mode=argv[2];whisper_params p;p.model=std::string(argv[1])+"/opaque.gguf";p.tts_voice="generated.wav";
 if(mode=="missing-model"){remove((std::string(argv[1])+"/openvoice2-tcc-f16.gguf").c_str());Probe b;return b.init(p)?1:0;}
 if(mode=="invalid-reference"){reference_ok=false;Probe b;return b.init(p)?1:0;}
 if(mode=="f32-companion"){
  auto base=std::string(argv[1]);rename((base+"/openvoice2-tcc-f16.gguf").c_str(),(base+"/openvoice2-tcc-f32sens.gguf").c_str());
  Probe b;return b.init(p)?0:1;}
 if(mode=="unset-speaker"){p.tts_speaker_id=-1;Probe b;if(!b.init(p))return 2;auto out=b.synthesize("test",p);
  return melo_seen.speaker_id==0 && speaker==0 && out.size()==4?0:1;}
 if(mode=="baseline"){p.tts_voice.clear();Probe b;if(!b.init(p))return 2;auto out=b.synthesize("test",p);
  return out.size()==8 && b.tts_sample_rate()==44100 && conversions==0?0:1;}
 if(mode=="clone"){Probe b;if(!b.init(p))return 2;auto out=b.synthesize("test",p);
  return out.size()==4 && b.tts_sample_rate()==22050 && conversions==1 && speaker==2?0:1;}
 if(mode!="routing"){
  Probe b;if(!b.init(p))return 2;
  if(mode=="initial-rate")return b.tts_sample_rate()==22050?0:1;
  if(mode=="neutral"){p.tts_voice.clear();p.tts_speaker_id=0;auto out=b.synthesize("disclosure",p);
   return conversions==0 && speaker==0 && out.size()==4 && b.tts_sample_rate()==22050?0:1;}
  if(mode=="conversion-failure"){converter_fails=true;auto out=b.synthesize("test",p);return out.empty()?0:1;}
  return 3;
 }
 for(bool requested:{false,true})for(const char* backend:{"cpu","auto","cuda"}){
  p.use_gpu=requested;p.gpu_backend=backend;Probe b;assert(b.init(p));
  bool expected=requested && p.gpu_backend!="cpu";
  assert(melo_seen.use_gpu==expected);
  if(ov_seen.use_gpu!=expected){fprintf(stderr,"OpenVoice routing wrong: requested=%d backend=%s actual=%d\n",requested,backend,ov_seen.use_gpu);return 1;}
  assert(ov_seen.n_threads==3 && ov_seen.verbosity==0);
 }
 puts("6 actual CLI init routing controls passed");
}
'''
        with tempfile.TemporaryDirectory(prefix='openvoice-cli-routing-') as temp:
            root = Path(temp)
            (root / 'openvoice2-tcc-f16.gguf').write_bytes(b'fixture: no model is loaded')
            (root / 'probe.cpp').write_text(mock + helper + '\nstruct Probe{\n' + init + synth + rate + fields + '\n};\n' + main)
            subprocess.run(['c++', '-std=c++17', str(root / 'probe.cpp'), '-o', str(root / 'probe')], check=True)
            for mode in ['routing','initial-rate','neutral','conversion-failure',
                         'missing-model','invalid-reference','baseline','clone','f32-companion','unset-speaker']:
                with self.subTest(mode=mode):
                    (root / 'openvoice2-tcc-f16.gguf').write_bytes(b'fixture')
                    subprocess.run([str(root / 'probe'), str(root),mode],check=True)


if __name__ == '__main__':
    unittest.main()
