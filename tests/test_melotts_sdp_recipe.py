"""Run production duration code with tiny noncommuting flow fixtures."""

import ast
import ctypes
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import unittest

from test_openvoice_cli_routing import function

ROOT = Path(__file__).resolve().parents[1]


def guard():
    if Path("/proc/meminfo").exists():
        memory = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        if int(
            memory["MemAvailable"].split()[0]
        ) * 1024 < 3 * 1024**3 or os.getloadavg()[0] > 2 * (os.cpu_count() or 1):
            raise RuntimeError(
                "Insufficient resources before tiny duration fixture compile"
            )
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class SDPRecipe(unittest.TestCase):
    def test_converter_counts_stochastic_namespace(self):
        source = (ROOT / "models/convert-melotts-to-gguf.py").read_text()
        start = source.index("    n_sdp_flows =")
        end = source.index("    n_resblocks =", start)
        import textwrap

        tensors = {f"sdp.flows.{i}.pre.weight": None for i in [1, 3, 5, 7]}
        tensors.update({f"sdp.convs.convs_sep.{i}.weight": None for i in range(3)})
        ns = dict(tensors=tensors)
        exec(
            compile(
                ast.parse(textwrap.dedent(source[start:end])),
                "actual-converter-counts",
                "exec",
            ),
            ns,
        )
        self.assertEqual(ns["n_sdp_flows"], 3)
        self.assertEqual(ns["n_sdp_dds_layers"], 3)

    def test_actual_reverse_flow_order(self):
        source = (ROOT / "src/melotts.cpp").read_text()
        preamble = r"""
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm>
struct tensor {std::vector<float> data;};
struct weights {
 tensor *sdp_pre_w,*sdp_pre_b,*sdp_cond_w,*sdp_cond_b,*sdp_proj_w,*sdp_proj_b;
 tensor *sdp_flow0_m=nullptr,*sdp_flow0_logs=nullptr;
 int sdp_convs=0;std::vector<int> sdp_flows;
};
struct melotts_context {struct {int hidden_channels=2,sdp_num_bins=10,gin_channels=2;} hp;
 weights w;unsigned seed=42;};
void read_tensor_f32(tensor* t,std::vector<float>& out){out=t->data;}
void dds_conv_forward(int,const std::vector<float>& in,int,int,std::vector<float>& out){out=in;}
void dump_stage(melotts_context*,const char*,const float*,size_t){}
static void sdp_convflow_inverse(melotts_context*,int fi,std::vector<float>& z,
 const std::vector<float>& h,int C,int T,int){
 for(int t=0;t<T;t++)z[2*t+1]-=.25f*(fi+1)+.2f*z[2*t]+.1f*h[t*C];
}
"""
        main = r"""
int main(){
 tensor identity{{1,0,0,1}},zeros{{0,0}};melotts_context ctx;
 ctx.w={&identity,&zeros,&identity,&zeros,&identity,&zeros};
 for(int blocks:{1,2,3}){
  ctx.w.sdp_flows.clear();for(int i=0;i<blocks;i++)ctx.w.sdp_flows.push_back(i);
  const std::vector<float> h={1,2,3,4},g={.5f,-.5f};std::vector<float> actual;
  sdp_forward(&ctx,h,g,2,.8f,actual);
  std::vector<float> expected(4,0),conditioned={1.5f,1.5f,3.5f,3.5f};
  std::mt19937 rng(ctx.seed+1);std::normal_distribution<float> dist(0,.8f);
  for(float& value:expected)value=dist(rng);
  // Original reversed ModuleList: Flip, ConvFlow, ..., Flip, ElementwiseAffine.
  for(int fi=blocks-1;fi>=0;fi--){
   for(int t=0;t<2;t++)std::swap(expected[2*t],expected[2*t+1]);
   sdp_convflow_inverse(&ctx,fi,expected,conditioned,2,2,10);
  }
  for(int t=0;t<2;t++)std::swap(expected[2*t],expected[2*t+1]);
  for(int t=0;t<2;t++)if(std::abs(actual[t]-expected[2*t])>1e-6f){
   std::fprintf(stderr,"SDP inverse module order mismatch: %g != %g\n",actual[t],expected[2*t]);return 1;
  }
 }
}
"""
        self.compile_run(preamble + function(source, "static void sdp_forward(") + main)

    def test_legacy_zero_metadata_recovers_real_tensor_layers(self):
        source = (ROOT / "src/melotts.cpp").read_text()
        start = source.index("    w.sdp_pre_w = require_tensor")
        end = source.index("    int sdp_flow_idx[]", start)
        code = (
            r"""
#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <cassert>
#if __has_include("core/melotts_sdp.h")
#include "core/melotts_sdp.h"
#endif
struct tensor{};
struct melotts_context {
 struct {unsigned n_sdp_flows=0,n_sdp_dds_layers=0;} hp;
 struct {tensor *sdp_pre_w,*sdp_pre_b,*sdp_proj_w,*sdp_proj_b,*sdp_cond_w,*sdp_cond_b,
  *sdp_flow0_m,*sdp_flow0_logs;int sdp_convs;std::vector<int> sdp_flows;} w;
};
tensor* require_tensor(const std::map<std::string,tensor*>& ts,const std::string& key){auto it=ts.find(key);return it==ts.end()?nullptr:it->second;}
tensor* try_tensor(const std::map<std::string,tensor*>& ts,const std::string& key){return require_tensor(ts,key);}
bool load_dds_conv(const std::map<std::string,tensor*>&,const char*,int n,int& dds){dds=n;return true;}
bool probe(melotts_context* ctx,const std::map<std::string,tensor*>& tensors){
 auto& hp=ctx->hp;auto& w=ctx->w;
"""
            + source[start:end]
            + r"""
 return true;
}
int main(){
 tensor t;std::map<std::string,tensor*> tensors;
 for(int i:{1,3,5,7})tensors["sdp.flows."+std::to_string(i)+".pre.weight"]=&t;
 for(int i=0;i<3;i++)tensors["sdp.convs.convs_sep."+std::to_string(i)+".weight"]=&t;
 melotts_context ctx;assert(probe(&ctx,tensors));
 assert(ctx.hp.n_sdp_flows==3);assert(ctx.hp.n_sdp_dds_layers==3);
 assert(ctx.w.sdp_convs==3);assert(ctx.w.sdp_flows.size()==3);
 ctx.hp.n_sdp_flows=4;assert(!probe(&ctx,tensors));
 ctx.hp.n_sdp_flows=0;ctx.hp.n_sdp_dds_layers=0;
 tensors.erase("sdp.flows.5.pre.weight");assert(!probe(&ctx,tensors));
 tensors["sdp.flows.5.pre.weight"]=&t;
 tensors.erase("sdp.convs.convs_sep.1.weight");assert(!probe(&ctx,tensors));
}
"""
        )
        self.compile_run(code)

    def test_actual_spline_matches_frozen_original_scalar_oracle(self):
        fixture = json.loads(
            (ROOT / "tests/fixtures/melotts-rqs-original.json").read_text()
        )
        source = (ROOT / "src/melotts.cpp").read_text()
        code = (
            """#include <vector>
#include <cmath>
#include <algorithm>
"""
            + function(source, "static float rqs_inverse(")
            + """
extern "C" float evaluate(float y,const float* w,const float* h,const float* d){
 return rqs_inverse(y,w,h,d,10,-5,5);
}
"""
        )
        guard()
        with tempfile.TemporaryDirectory(prefix="melo-rqs-") as temp:
            path = Path(temp)
            (path / "test.cpp").write_text(code)
            library = path / "test.so"
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    str(path / "test.cpp"),
                    "-o",
                    str(library),
                ],
                check=True,
            )
            lib = ctypes.CDLL(str(library))
            lib.evaluate.restype = ctypes.c_float
            lib.evaluate.argtypes = [ctypes.c_float] + [
                ctypes.POINTER(ctypes.c_float)
            ] * 3
            for row in fixture["cases"]:
                with self.subTest(pattern=row["pattern"], value=row["input"]):
                    values = [
                        (ctypes.c_float * len(row[key]))(*row[key])
                        for key in [
                            "width_logits",
                            "height_logits",
                            "derivative_logits",
                        ]
                    ]
                    actual = lib.evaluate(row["input"], *values)
                    self.assertAlmostEqual(actual, row["expected"], delta=2e-5)

    def compile_run(self, code):
        guard()
        with tempfile.TemporaryDirectory(prefix="melo-sdp-") as temp:
            path = Path(temp)
            (path / "test.cpp").write_text(code)
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-I",
                    str(ROOT / "src"),
                    str(path / "test.cpp"),
                    "-o",
                    str(path / "test"),
                ],
                check=True,
            )
            result = subprocess.run(
                [str(path / "test")], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
