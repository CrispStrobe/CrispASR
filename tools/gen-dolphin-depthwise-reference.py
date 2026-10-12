#!/usr/bin/env python3
import ast,pathlib,hashlib,typing,json,torch,argparse
ap=argparse.ArgumentParser(description="Regenerate original Dolphin CSGU/merge fixtures without checkpoints")
ap.add_argument("--source",type=pathlib.Path,required=True,help="model.py from the pinned original revision")
ap.add_argument("--output-dir",type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]/"tests/fixtures")
args=ap.parse_args()
source=args.source;assert hashlib.sha256(source.read_bytes()).hexdigest()=='bad508847e2d341af12d852053f9de4e48ad3fe3b3366ea1a96fad7dcf5163a7', 'Original source hash mismatch';tree=ast.parse(source.read_text());names={'ConvolutionalSpatialGatingUnit','EBranchformerEncoderLayer'};nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names];assert len(nodes)==2
ns=dict(torch=torch,nn=torch.nn,Tuple=typing.Tuple,Optional=typing.Optional,T_CACHE=typing.Tuple[torch.Tensor,torch.Tensor]);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
class Attention(torch.nn.Module):
 def forward(self,x,*args):self.last=x.detach().clone();return x,torch.zeros(0)
class MLP(torch.nn.Module):
 def forward(self,x,*args):self.last=x.detach().clone();return x,torch.zeros(0)
rows=[];torch.set_num_threads(1);torch.manual_seed(42)
for causal in [True,False]:
 for T,K in [(1,3),(5,3),(7,31)]:
  for csgu in [True,False]:
   C=3 if csgu else 6
   if csgu:module=ns['ConvolutionalSpatialGatingUnit'](2*C,K,0.,False,'identity',causal);conv=module.conv;module.norm.weight.data.copy_(torch.linspace(.7,1.3,C));module.norm.bias.data.copy_(torch.linspace(-.3,.4,C));x=torch.arange(T*C,dtype=torch.float32).reshape(1,T,C)/5-.7
   else:
    attn=Attention();mlp=MLP();module=ns['EBranchformerEncoderLayer'](C//2,attn,mlp,None,None,0.,merge_conv_kernel=K,causal=causal);conv=module.depthwise_conv_fusion;x=torch.arange(T*C//2,dtype=torch.float32).reshape(1,T,C//2)/5-.7
   conv.weight.data.copy_(torch.sin(torch.arange(C*K,dtype=torch.float32)).reshape(C,1,K)/10);conv.bias.data.copy_(torch.linspace(.2,.6,C));module.eval()
   with torch.no_grad():
    if csgu:y=module(torch.cat([torch.ones_like(x),x],dim=-1))[0]
    else:
     captured=[];hook=conv.register_forward_hook(lambda m,i,o:captured.append(o.detach().clone()));module._forward(x,torch.ones(1,T,T,dtype=torch.bool),torch.zeros(1,T,C//2));hook.remove();y=captured[0].transpose(1,2);x=torch.cat([attn.last,mlp.last],dim=-1)
   row=dict(C=C,T=T,K=K,causal=causal,csgu=csgu,x=x.flatten().tolist(),w=conv.weight.flatten().tolist(),b=conv.bias.flatten().tolist(),lnw=module.norm.weight.flatten().tolist() if csgu else [],lnb=module.norm.bias.flatten().tolist() if csgu else [],y=y.flatten().tolist());rows.append(row)
folder=args.output_dir;folder.mkdir(parents=True,exist_ok=True)
header=['// Generated from DataoceanAI/Dolphin model.py @ 78ea615194777f91a3970ff010b5d678d0bbf5f0.', '// Both original modules execute on CPU; CSGU has nonzero learned LayerNorm bias.', 'static const ConvCase cases[] = {']
def floats(xs):return '{'+','.join(format(v,'.9g')+'f' if '.' in format(v,'.9g') or 'e' in format(v,'.9g') else format(v,'.9g')+'.0f' for v in xs)+'}'
for r in rows:header.append('  {'+','.join([str(r['C']),str(r['T']),str(r['K']),str(r['causal']).lower(),str(r['csgu']).lower()]+[floats(r[k]) for k in ['x','w','b','lnw','lnb','y']])+'},')
header.append('};');(folder/'dolphin-depthwise-reference.h').write_text('\n'.join(header)+'\n');(folder/'dolphin-depthwise-reference.json').write_text(json.dumps(dict(upstream='78ea615194777f91a3970ff010b5d678d0bbf5f0',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),torch=torch.__version__,seed=42,cases=len(rows),modules=sorted(names),method='Execute original CSGU.forward and EBranchformerEncoderLayer._forward with captured actual merge convolution; no model checkpoints'),indent=2)+'\n');print('Generated',len(rows),'actual original module controls')
