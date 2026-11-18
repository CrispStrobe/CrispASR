#!/usr/bin/env python3
"""Tiny retained-fixture A/B of gated raw prefill output preservation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    model=args.fixture/'tiny.gguf';inputs=args.fixture/'inputs.bin'
    if model.stat().st_size>1<<20 or inputs.stat().st_size!=39*32*4:
        raise ValueError('only retained tiny fixture is permitted')
    args.output.mkdir(parents=True,exist_ok=False)
    capture=args.output/'capture';capture.mkdir()
    env={key:value for key,value in os.environ.items() if not key.startswith('CRISPASR_VIBEVOICE_TTS_CAPTURE')}
    env['CRISPASR_VIBEVOICE_TTS_FLASH_ATTN']='1'
    off=args.output/'off.bin';on=args.output/'on.bin'
    subprocess.run([str(args.probe),str(model),str(inputs),str(off)],env=env,check=True,close_fds=False)
    env.update(CRISPASR_VIBEVOICE_TTS_CAPTURE_FIRST_FRAME=str(capture.resolve()),CRISPASR_VIBEVOICE_TTS_CAPTURE_PREFILL_BLOCKS='1')
    subprocess.run([str(args.probe),str(model),str(inputs),str(on)],env=env,check=True,close_fds=False)
    assert off.read_bytes()==on.read_bytes(),'tap preservation changes tiny native hidden/logits'
    assert off.read_bytes()==(args.fixture/'native.bin').read_bytes(),'gate-off differs from retained native default'
    import struct
    assert struct.unpack('<iii',(capture/'tts_prefill_blocks_complete.bin').read_bytes())==(2,39,32)
    files=sorted(capture.glob('tts_prefill_block_*.bin'))
    assert len(files)==2
    stages=[]
    for path in files:
        values=np.fromfile(path,dtype='<f4')
        assert values.size==39*32 and np.isfinite(values).all() and np.linalg.norm(values)>0
        stages.append(dict(stage=path.name,bytes=path.stat().st_size,norm=float(np.linalg.norm(values.astype('float64'))),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    receipt=dict(scope='tiny CPU gate-off/gate-on A/B only; not7B/GPU parity',gate_off_on_bit_exact=True,
        gate_off_retained_bit_exact=True,raw_blocks=stages,raw_payload_bytes=sum(row['bytes'] for row in stages),
        native_output_sha256=hashlib.sha256(on.read_bytes()).hexdigest())
    (args.output/'comparison.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2));print('PASS: tap-off/tap-on/retained native outputs bit-exact; two valid raw39-token blocks')


if __name__=='__main__':main()
