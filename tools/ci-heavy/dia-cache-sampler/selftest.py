#!/usr/bin/env python3
"""Check actual native sampling/noise against the pinned original Python function."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--blueprint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[3]
source = (a.blueprint/'dia/model.py').read_bytes()
assert hashlib.sha256(source).hexdigest() == '6bc51e6e5ff58ca9ed66431773df6cda954afee3d4b7ea515c1363b73ca82926'
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_sample_next_token')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'pinned-original-dia-sampler', 'exec'), namespace)
reference = namespace['_sample_next_token']
assert torch.__version__ == '2.6.0+cpu', torch.__version__
torch.set_num_threads(1)
cpp = r'''
#include "dia_sampling.h"
#include <fstream>
#include <cstdlib>
int main(int argc,char**argv) {
    if (argc!=9) return 2;
    const int heads=std::atoi(argv[1]),vocab=std::atoi(argv[2]),k=std::atoi(argv[3]);
    const float temperature=std::atof(argv[4]),top_p=std::atof(argv[5]);
    std::ifstream f(argv[6],std::ios::binary);
    std::vector<float> logits(heads*vocab),noise(heads*vocab);
    f.read((char*)logits.data(),logits.size()*4); if(!f) return 3;
    std::mt19937 rng(42);
    if(temperature>0) dia_rng::exponential_batch(noise.data(),noise.size(),rng);
    std::ofstream out(argv[7],std::ios::binary);
    out.write((char*)noise.data(),noise.size()*4);
    for(int h=0;h<heads;++h) {
        auto token=dia_sample_token(logits.data()+h*vocab,vocab,temperature,top_p,k,rng,
                                   temperature>0 ? noise.data()+h*vocab : nullptr);
        out.write((char*)&token,4);
    }
    std::ofstream state(argv[8]); state<<rng;
    return !out || !state;
}
'''
(a.output/'probe.cpp').write_text(cpp)
with (a.output/'compile.log').open('w') as log:
    subprocess.run(['c++','-std=c++17','-O2','-I',root/'src',a.output/'probe.cpp','-o',a.output/'probe'],check=True,stdout=log,stderr=subprocess.STDOUT)
rows = []
original_multinomial = torch.multinomial
for case, (heads,vocab,k,temp,top_p) in enumerate([(9,1028,45,1.2,.95), (3,32,7,1.2,.7), (2,17,1,1.,.95), (3,17,7,0.,.5), (9,1028,45,1.2,.95)]):
    folder=a.output/str(case);folder.mkdir()
    x=np.random.default_rng(case).normal(0,3,(heads,vocab)).astype('<f4')
    # Include all masked vocabulary entries in the same batch, just as runtime.
    x[:,-3:]=-np.inf
    if case==4:
        x.fill(-np.inf);x[:,3]=20.;x[:,5]=0.;x[:,7]=-1.
    x.tofile(folder/'logits.f32')
    captured=[]
    def observe(probabilities,num_samples):
        captured.append(probabilities.clone())
        return original_multinomial(probabilities,num_samples)
    torch.manual_seed(42);before=torch.get_rng_state().clone()
    torch.multinomial=observe
    try: expected=reference(torch.from_numpy(x),temp,top_p,k).numpy()
    finally: torch.multinomial=original_multinomial
    after=torch.get_rng_state().clone()
    torch.set_rng_state(before)
    noise=torch.empty((heads,vocab)).exponential_().numpy() if temp else np.zeros((heads,vocab),dtype='<f4')
    assert torch.equal(after,torch.get_rng_state())
    # Original MT state advances exactly once for a non-greedy contiguous batch.
    torch.set_rng_state(before)
    if temp: torch.empty((),dtype=torch.int32).random_()
    assert torch.equal(after,torch.get_rng_state())
    subprocess.run([a.output/'probe',str(heads),str(vocab),str(k),str(temp),str(top_p),folder/'logits.f32',folder/'actual.bin',folder/'rng.txt'],check=True)
    actual=np.fromfile(folder/'actual.bin',dtype='<f4',count=heads*vocab).reshape(heads,vocab)
    tokens=np.fromfile(folder/'actual.bin',dtype='<u4',offset=heads*vocab*4)
    assert np.array_equal(tokens,expected),(case,tokens,expected)
    assert np.allclose(actual,noise,rtol=5e-7,atol=2e-7),(case,np.max(abs(actual-noise)))
    np.save(folder/'original-noise.npy',noise,allow_pickle=False)
    np.save(folder/'original-tokens.npy',expected,allow_pickle=False)
    if captured: np.save(folder/'original-probabilities.npy',captured[0].numpy(),allow_pickle=False)
    # State serialization is equal for every stochastic case irrespective of
    # vocabulary size, number of heads, masks or sampled tokens.
    if temp:
        if any(r['temperature'] for r in rows):
            previous=next(r for r in rows if r['temperature'])
            assert (folder/'rng.txt').read_bytes()==(a.output/str(previous['case'])/'rng.txt').read_bytes()
    rows.append(dict(case=case,heads=heads,vocab=vocab,temperature=temp,top_p=top_p,top_k=k,
        exact_original_tokens=True,one_mt_word_per_batch=bool(temp),max_noise_abs=float(abs(actual-noise).max())))
(a.output/'receipt.json').write_text(json.dumps(dict(torch=torch.__version__,original_sha256=hashlib.sha256(source).hexdigest(),cases=rows),indent=2)+'\n')
print('DIA_CACHE_SAMPLER_ORIGINAL_TINY_PASS',len(rows),'cases',sum(r['heads'] for r in rows),'head decisions')
