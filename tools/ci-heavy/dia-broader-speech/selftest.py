#!/usr/bin/env python3
"""Calibrate matrix gates and actual shared child parameter forwarding; no model."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dia_broader_speech as worker


def main():
    assert list(worker.CASES)==['hello-42','hello-123','fox-42','fox-123']
    for case in worker.CASES.values():
        assert worker.support.score(case['text'][5:],case['text'])==0
        assert worker.support.score('',case['text'])==1
    assert worker.support.score('We are committed to making it secure.')==0
    vector=np.array([[1.,2.,3.],[2.,1.,4.]],dtype='f4')
    assert worker.metrics(vector,vector)['pass']
    assert not worker.metrics(vector*2,vector)['pass']
    assert not worker.metrics(-vector,vector)['pass']
    a=np.ones((1000,3),dtype='f4');b=a.copy();b[3]*=2
    assert not worker.metrics(a,b)['pass']
    assert worker.metrics(np.zeros(3),np.zeros(3))['pass']
    assert not worker.metrics(np.ones(3),np.zeros(3))['pass']
    rejects=0
    for bad in [vector[:1],vector[:,0],np.empty((0,3)),np.full_like(vector,np.nan),np.full_like(vector,np.inf)]:
        try:worker.metrics(bad,vector)
        except ValueError:rejects+=1
        else:raise AssertionError('Bad stage accepted')
    assert rejects==5
    observed=[]
    class Session:
        def __init__(self,*a,**kw):assert kw['n_threads']==4 and kw['backend']=='dia'
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def set_temperature(self,temperature,seed):observed.append(dict(temperature=temperature,seed=seed))
        def synthesize(self,text):observed[-1]['text']=text;return np.ones(512,dtype='f4')
        def output_sample_rate(self):return 44100
    env={k:v for k,v in os.environ.items() if not k.startswith(('DIA_','CRISPASR_DIA_'))}
    # Use the real shared child body, replacing only model loading/computation.
    with tempfile.TemporaryDirectory(prefix='dia-broader-selftest-') as folder:
        ledger=Path(folder)/'ledger';ledger.mkdir();(ledger/'nested').mkdir()
        (ledger/'receipt.json').write_text('{}');(ledger/'nested/receipt.json').write_text('{}')
        assert [row['path'] for row in worker.support.payload_inventory(ledger)]==['nested/receipt.json']
        with patch.dict(os.environ,env,clear=True),patch.dict(sys.modules,{'crispasr':SimpleNamespace(Session=Session)}):
            for key,case in worker.CASES.items():
                args=SimpleNamespace(child='native',arm=worker.ARM,runtime=Path('/unused'),model=Path('/unused'),lib=Path('/unused'),**case)
                worker.support.child(args,Path(folder)/key)
                assert observed[-1]==dict(temperature=1.2,**case)
                for name,value in worker.support.arm_environment(worker.ARM).items():assert os.environ[name]==value
    print('DIA_BROADER_SELFTEST_PASS: four exact text/seed forwards, original scoring and scale/row/invalid gates')


if __name__=='__main__':main()
