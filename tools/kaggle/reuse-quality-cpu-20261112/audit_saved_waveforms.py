import argparse
import hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from pystoi import stoi
from gguf import GGUFReader
parser=argparse.ArgumentParser(description='Audit saved RE-USE waveform windows without model inference')
parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
SRC=args.input_dir
OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
def metrics(c,y):
 np.random.seed(579) # pystoi ESTOI normalization jitter must be reproducible.
 c=np.asarray(c,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 a=np.dot(c-c.mean(),y-y.mean())/np.dot(c-c.mean(),c-c.mean());p=a*(c-c.mean());e=y-y.mean()-p
 _,_,C=signal.stft(c,16000,nperseg=512,noverlap=384);_,_,Y=signal.stft(y,16000,nperseg=512,noverlap=384)
 return dict(snr_db=float(10*np.log10(np.dot(c,c)/max(np.dot(c-y,c-y),1e-30))),si_sdr_db=float(10*np.log10(np.dot(p,p)/max(np.dot(e,e),1e-30))),gain_projection=float(a),rms_ratio=float(np.linalg.norm(y)/np.linalg.norm(c)),stoi=float(stoi(c,y,16000)),estoi=float(stoi(c,y,16000,extended=True)),spectral_convergence=float(np.linalg.norm(abs(C)-abs(Y))/np.linalg.norm(abs(C))))
clean,clean_rate=sf.read(SRC/'input-proxy.wav');noisy,noisy_rate=sf.read(SRC/'input-noisy.wav');stitched,out_rate=sf.read(SRC/'output-reference-chunked.wav')
assert clean_rate == noisy_rate == out_rate == 16000
assert clean.shape == noisy.shape == stitched.shape == (160000,)
assert all(np.isfinite(x).all() for x in (clean,noisy,stitched))
report={'estoi_normalization_seed':579,'input_hashes': {name:hashlib.sha256((SRC/name).read_bytes()).hexdigest() for name in ('input-proxy.wav','input-noisy.wav','output-reference-chunked.wav')},'scope':'Saved waveform audit; one JFK speech proxy, no new model runs or independent voices','segments':[],'windows':[],'overlaps':[]}
for start,end in [(0,160000),(0,80000),(80000,160000),(0,40000),(40000,80000),(80000,120000),(120000,160000)]:
 report['segments'].append(dict(start=start,end=end,input=metrics(clean[start:end],noisy[start:end]),output=metrics(clean[start:end],stitched[start:end])))
waves=[]
for i,start in enumerate([0,40000,80000]):
 r=GGUFReader(SRC/f'window-{i}-reference.gguf');t={x.name:x for x in r.tensors};y=t['waveform'].data.copy();raw=t['raw_audio'].data.copy();waves.append(y)
 assert np.array_equal(raw,noisy[start:start+80000].astype('float32'))
 sf.write(OUT/f'window-{i}-reference.wav',y,16000,subtype='FLOAT')
 report['windows'].append(dict(index=i,start=start,input=metrics(clean[start:start+80000],raw),output=metrics(clean[start:start+80000],y),raw_input_matches=True,raw_audio_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),waveform_sha256=hashlib.sha256(y.tobytes()).hexdigest()))
 del r,t
for i in range(2):
 a=waves[i][40000:];b=waves[i+1][:40000];c=clean[(i+1)*40000:(i+2)*40000]
 report['overlaps'].append(dict(left_index=i,right_index=i+1,relative_disagreement=float(np.linalg.norm(a-b)/np.linalg.norm(c)),left=metrics(c,a),right=metrics(c,b)))
(OUT/'saved-waveform-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
