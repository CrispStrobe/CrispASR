#!/usr/bin/env python3
"""Compile actual conditioning/averaging bodies with tiny embedding inputs."""
from pathlib import Path
import os
import subprocess
import re
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def body(source,signature):
    start=source.index(signature);opening=source.index('{',start);end=opening+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]


class ConditioningContracts(unittest.TestCase):
    def test_owned_atomic_conditioning_and_equal_clip_average(self):
        source=(ROOT/'src/openvoice2.cpp').read_text()
        preamble=r'''
#include "openvoice2.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>
struct openvoice2_context {
 struct {int gin_channels=256;int inter_channels=2;} hp;
 std::vector<float> target_override,posterior_noise;
};
static int calls=0;
extern "C" bool openvoice2_extract_speaker_embedding(openvoice2_context*,const float* pcm,int,int,float* out) {
 ++calls;if(*pcm<0)return false;
 for(int i=0;i<256;++i)out[i]=*pcm;
 return true;
}
'''
        functions='\n'.join(body(source,'extern "C" bool '+name+'(') for name in [
            'openvoice2_set_target_embedding','openvoice2_set_posterior_noise',
            'openvoice2_extract_speaker_embedding_segments'])
        main=r'''
int main(){
 openvoice2_context ctx;float embedding[256];for(float& x:embedding)x=1;
 assert(openvoice2_set_target_embedding(&ctx,embedding));embedding[0]=2;
 assert(ctx.target_override[0]==1);embedding[0]=NAN;
 assert(!openvoice2_set_target_embedding(&ctx,embedding));assert(ctx.target_override[0]==1);
 assert(openvoice2_set_target_embedding(&ctx,nullptr));assert(ctx.target_override.empty());
 float noise[]={1,2,3,4};assert(openvoice2_set_posterior_noise(&ctx,noise,4));noise[0]=5;
 assert(ctx.posterior_noise[0]==1);assert(!openvoice2_set_posterior_noise(&ctx,noise,3));
 noise[0]=INFINITY;assert(!openvoice2_set_posterior_noise(&ctx,noise,4));
 assert(ctx.posterior_noise[0]==1);assert(!openvoice2_set_posterior_noise(&ctx,nullptr,4));
 assert(openvoice2_set_posterior_noise(&ctx,nullptr,0));assert(ctx.posterior_noise.empty());
 float pcm[]={1,1,5,5,5,5};int starts[]={0,2},ends[]={2,6};float out[256];
 for(float& x:out)x=42;
 assert(openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,22050,starts,ends,2,out));
 for(float x:out)assert(x==3);assert(calls==2);
 starts[1]=1;assert(!openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,22050,starts,ends,2,out));
 assert(calls==2);starts[1]=2;pcm[2]=-1;
 assert(!openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,22050,starts,ends,2,out));
 for(float x:out)assert(x==3);
 assert(!openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,0,starts,ends,2,out));
 assert(!openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,22050,nullptr,ends,2,out));
 assert(!openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,22050,starts,ends,0,out));
}
'''
        with tempfile.TemporaryDirectory(prefix='openvoice-conditioning-') as tmp:
            path=Path(tmp);(path/'test.cpp').write_text(preamble+functions+main)
            subprocess.run(['c++','-std=c++17','-I',str(ROOT/'src'),str(path/'test.cpp'),'-o',str(path/'test')],check=True)
            subprocess.run([str(path/'test')],check=True)



class NumericalRecipeContracts(unittest.TestCase):
    """Production CPU bodies; tiny biased affine maps expose recipe errors."""
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='openvoice-recipe-')
        source=(ROOT/'src/openvoice2.cpp').read_text()
        preamble=r"""
#include <vector>
#include <deque>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <string>
struct ggml_tensor {long ne[4];std::vector<float> data;};
static std::deque<ggml_tensor> weights;
static ggml_tensor* tensor(int first,std::initializer_list<float> data){
 weights.push_back({{first,1,1,1},data});return &weights.back();
}
static void read_f32(ggml_tensor* t,std::vector<float>& out){out=t->data;}
struct ov2_wn_layer {ggml_tensor *in_w,*in_b,*res_skip_w,*res_skip_b;};
struct ov2_wn_block {ggml_tensor *cond_w,*cond_b;std::vector<ov2_wn_layer> layers;};
struct ov2_flow_block {ggml_tensor *pre_w,*pre_b,*post_w,*post_b;ov2_wn_block wn;};
struct posterior {ggml_tensor *pre_w,*pre_b,*proj_w,*proj_b;ov2_wn_block wn;};
struct openvoice2_context {
 struct {int hidden_channels=2,inter_channels=2,spec_channels=2,gin_channels=2,
             n_wn_layers_enc_q=1,n_wn_layers_flow=1,n_flow_blocks=1;bool zero_g=true;} hp;
 struct {posterior enc_q;std::vector<ov2_flow_block> flow_blocks;} w;
 float tau=0;std::vector<float> posterior_noise;std::mt19937 rng{42};
};
static void dump_stage(openvoice2_context*,const char*,const float*,size_t){}
static ov2_wn_block wn(){
 return {tensor(1,{1,0,0,1,1,-1,.5f,.5f}),tensor(4,{.25f,-.5f,.1f,-.2f}),
 {{tensor(1,{0,0,0,0,0,0,0,0}),tensor(4,{.1f,.2f,.3f,.4f}),
   tensor(1,{1,0,0,1}),tensor(2,{0,0})}}};
}
"""
        functions='\n'.join(body(source,signature) for signature in [
            'static void wavenet_forward(', 'static void compute_g_cond(',
            'static void enc_q_forward(', 'static void flow_wavenet('])
        main=r"""
static bool close(float a,float b){return std::abs(a-b)<1e-6f;}
int main(int argc,char** argv){
 if(std::string(argv[1])=="bias"){
  openvoice2_context ctx;
  ctx.w.enc_q={tensor(1,{1,0,0,1}),tensor(2,{0,0}),
      tensor(1,{1,0,0,1,0,0,0,0}),tensor(4,{0,0,0,0}),wn()};
  const float spec[]={.2f,.4f,-.3f,.7f,.5f,-.1f};
  for(bool zero:{true,false}){
   ctx.hp.zero_g=zero;
   for(const auto& g:std::vector<std::vector<float>>{{0,0},{1,-2}}){
    std::vector<float> z;enc_q_forward(&ctx,{spec,spec+6},3,g,z);
    // Original Conv1d: zero speaker input retains the learned affine bias.
    float cond[]={.25f,-.5f,.1f,-.2f};
    if(!zero){cond[0]+=g[0];cond[1]+=g[1];cond[2]+=g[0]-g[1];cond[3]+=.5f*(g[0]+g[1]);}
    for(int t=0;t<3;t++)for(int c=0;c<2;c++){
     float expected=std::tanh(.1f*(c+1)+cond[c])/(1+std::exp(-(.1f*(c+3)+cond[c+2])));
     if(!close(z[2*t+c],expected)){std::fprintf(stderr,"zero_g affine bias mismatch: %.9g != %.9g\n",z[2*t+c],expected);return 1;}
    }
   }
  }
 }else{
  for(int blocks:{1,2}){
   openvoice2_context ctx;ctx.hp.inter_channels=4;ctx.hp.n_flow_blocks=blocks;
   for(int i=0;i<blocks;i++)ctx.w.flow_blocks.push_back({tensor(1,{0,0,0,0}),tensor(2,{0,0}),
       tensor(1,{0,0,0,0}),tensor(2,{.25f*(i+1),-.5f*(i+1)}),wn()});
   std::vector<float> input={1,2,3,4,-1,-2,-3,-4},actual=input,expected=input;
   // Original forward module order is coupling THEN full channel flip.
   for(int i=0;i<blocks;i++)for(int t=0;t<2;t++){
    expected[4*t+2]+=.25f*(i+1);expected[4*t+3]-=.5f*(i+1);
    std::reverse(expected.begin()+4*t,expected.begin()+4*t+4);
   }
   flow_wavenet(&ctx,actual,2,{1,-2},false);
   for(size_t i=0;i<actual.size();i++)if(!close(actual[i],expected[i])){
    std::fprintf(stderr,"forward coupling/flip order mismatch at %zu: %.9g != %.9g\n",i,actual[i],expected[i]);return 1;
   }
   flow_wavenet(&ctx,actual,2,{1,-2},true);
   for(size_t i=0;i<actual.size();i++)if(!close(actual[i],input[i]))return 2;
  }
 }
}
"""
        path=Path(cls.temp.name);(path/'test.cpp').write_text(preamble+functions+main)
        cls.binary=path/'test'
        subprocess.run(['c++','-std=c++17','-I',str(ROOT/'src'),str(path/'test.cpp'),'-o',str(cls.binary)],check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_zero_speaker_keeps_learned_conditioning_bias(self):
        result=subprocess.run([str(self.binary),'bias'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_forward_couples_before_flip_and_reverse_is_its_inverse(self):
        result=subprocess.run([str(self.binary),'flow'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)



class DecoderActivationContract(unittest.TestCase):
    def test_final_default_slope_differs_from_upsample_slope(self):
        self.check_decoder('openvoice2')

    def test_melo_final_default_slope_differs_from_upsample_slope(self):
        self.check_decoder('melotts')

    def check_decoder(self, backend):
        source=(ROOT/('src/'+backend+'.cpp')).read_text()
        signature='static bool hifigan_decode_cpu(' if backend=='openvoice2' else 'static bool hifigan_decode('
        decoder=body(source,signature)
        calls=re.findall(r'x = ggml_leaky_relu\(gc, x, [^;]+;',decoder)
        self.assertEqual(len(calls),2)
        # Execute both real graph-builder calls on tiny signed inputs. The
        # original explicitly uses .1 in the upsample loop and F.leaky_relu's
        # default .01 immediately before conv_post. Positive values are fixed.
        program=r"""
#include <array>
#include <cmath>
#include <cstdio>
struct ggml_tensor {std::array<float,5> data;};
ggml_tensor* ggml_leaky_relu(void*,ggml_tensor* x,float slope,bool){
 for(float& v:x->data)if(v<0)v*=slope;return x;
}
int main(){
 void* gc=nullptr;
 for(int stage=0;stage<2;stage++){
  ggml_tensor tensor{{-10,-1,0,1,10}};ggml_tensor* x=&tensor;
  if(stage==0){ UPSAMPLE_CALL }else{ FINAL_CALL }
  const float expected=stage==0?-.1f:-.01f;
  if(std::abs(x->data[1]-expected)>1e-6f){
   std::fprintf(stderr,"decoder activation stage %d: %g != %g\n",stage,x->data[1],expected);return 1;
  }
  if(x->data[2]!=0 || x->data[3]!=1 || x->data[4]!=10)return 2;
 }
}
""".replace('UPSAMPLE_CALL',calls[0]).replace('FINAL_CALL',calls[1])
        with tempfile.TemporaryDirectory(prefix='openvoice-final-activation-') as temp:
            if Path('/proc/meminfo').exists():
                memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                if int(memory['MemAvailable'].split()[0])*1024 < 3*1024**3 or os.getloadavg()[0] > 2*(os.cpu_count() or 1):
                    raise RuntimeError('Insufficient resources before activation fixture compile')
            path=Path(temp);(path/'test.cpp').write_text(program)
            subprocess.run(['c++','-std=c++17',str(path/'test.cpp'),'-o',str(path/'test')],check=True)
            result=subprocess.run([str(path/'test')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
