#!/usr/bin/env python3
"""
Convert kyutai/pocket-tts safetensors -> GGUF for the CrispASR
`pocket-tts` backend.

Pocket TTS architecture:
  1. FlowLM backbone — causal transformer (1024D, 16H, 6L, RoPE)
     that autoregressively predicts *continuous* 32-dim float latents
     at 12.5 Hz. NOT discrete tokens — no codebook.
  2. Consistency head (SimpleMLPAdaLN) — MLP mapping backbone hidden
     states + timestep embeddings -> flow direction for one-step
     Lagrangian Self Distillation (LSD) decode.
  3. Mimi VAE codec — SEANet CNN encoder/decoder + small transformer,
     operating in a continuous 32-dim latent space (no RVQ).
     Encoder: causal conv + 2L transformer -> downsample -> 32-dim.
     Decoder: upsample -> 2L transformer -> causal conv -> 24 kHz PCM.
  4. Text conditioner — SentencePiece (4000 vocab) + learned embedding
     (4001 x 1024) lookup table.
  5. Speaker conditioning (gated variant) — ref audio -> Mimi VAE
     encode -> linear project -> prepend to transformer input.

Single GGUF output: contains FlowLM backbone + consistency head +
Mimi decoder (encoder weights optional for voice cloning) +
SentencePiece tokenizer model bytes.

Usage:
    python models/convert-pocket-tts-to-gguf.py \\
        --input kyutai/pocket-tts-without-voice-cloning \\
        --output-dir /mnt/storage/pocket-tts \\
        --language english

    python models/convert-pocket-tts-to-gguf.py \\
        --input /mnt/storage/pocket-tts \\
        --output-dir /mnt/storage/pocket-tts \\
        --language english \\
        --voice-cloning
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    snapshot_download = None
    hf_hub_download = None


# ── Architecture constants (from english.yaml) ─────────────────────

FLOW_LM_HPARAMS = dict(
    d_model=1024,
    num_heads=16,
    num_layers=6,
    hidden_scale=4,      # ff_dim = d_model * hidden_scale = 4096
    max_period=10000,
    latent_dim=32,        # continuous latent dimension
    n_bins=4000,          # SentencePiece vocab size
    lut_dim=1024,         # lookup table embedding dim
    insert_bos_before_voice=1,
)

FLOW_HEAD_HPARAMS = dict(
    flow_dim=512,         # consistency head hidden dim
    flow_depth=6,         # number of ResBlocks
    num_time_conds=2,     # s and t timestep embeddings
    freq_embed_size=256,  # sinusoidal frequency embedding
)

MIMI_HPARAMS = dict(
    sample_rate=24000,
    frame_rate_num=25,    # encoder_frame_rate = 24000 / hop_length
    frame_rate_den=2,     # target frame_rate = 12.5 = 25/2
    inner_dim=32,
    outer_dim=512,
    channels=1,
    # SEANet
    seanet_dimension=512,
    seanet_n_filters=64,
    seanet_n_residual_layers=1,
    seanet_kernel_size=7,
    seanet_residual_kernel_size=3,
    seanet_last_kernel_size=3,
    seanet_dilation_base=2,
    seanet_compress=2,
    # Transformer (encoder + decoder share config)
    xfmr_d_model=512,
    xfmr_num_heads=8,
    xfmr_num_layers=2,
    xfmr_dim_feedforward=2048,
    xfmr_context=250,
    xfmr_layer_scale_init=0.01,
    # Quantizer (really just a Conv1d projection)
    quant_in_dim=32,
    quant_out_dim=512,
)

# SEANet ratios: [6, 5, 4] => hop_length = 6*5*4 = 120
# encoder_frame_rate = 24000/120 = 200
# downsample_stride = 200/12.5 = 16
SEANET_RATIOS = [6, 5, 4]


# ── Helpers ─────────────────────────────────────────────────────────

def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def to_f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().half().cpu().numpy()


def choose_dtype(name: str, shape: list[int], tensor: torch.Tensor):
    """Choose GGUF dtype for a tensor. Large 2D weight matrices get F16,
    small biases/norms/embeddings stay F32."""
    ndim = len(shape)
    numel = int(np.prod(shape))

    # Embeddings, biases, norms, small tensors -> F32
    if ndim <= 1 or numel < 512:
        return to_f32(tensor), GGMLQuantizationType.F32
    # Layer norms
    if "norm" in name or "scale" in name:
        return to_f32(tensor), GGMLQuantizationType.F32
    # Everything 2D with >= 512 elements -> F16
    if ndim == 2 and numel >= 512:
        return to_f16(tensor), GGMLQuantizationType.F16
    # Conv weights (3D) -> F16 if large enough
    if ndim == 3 and numel >= 512:
        return to_f16(tensor), GGMLQuantizationType.F16
    return to_f32(tensor), GGMLQuantizationType.F32


def load_model_dir(model_id: str, allow_patterns=None) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    if snapshot_download is None:
        sys.exit(f"Directory {model_id} not found and huggingface_hub not installed")
    if allow_patterns is None:
        allow_patterns = ["*.safetensors", "*.yaml", "*.model", "tokenizer*"]
    print(f"Downloading {model_id}...", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=allow_patterns))


def denorm_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Denormalize PyTorch weight_norm: weight = g * v / ||v||."""
    return torch._weight_norm(v, g, dim=0)


def load_safetensors_with_weight_norm(path: Path) -> dict:
    """Load safetensors, folding weight_norm pairs into single weight tensors."""
    raw = {}
    with safe_open(str(path), framework='pt') as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)

    # Fold weight_norm: weight_g + weight_v -> weight
    wn_bases = set()
    for k in list(raw.keys()):
        if k.endswith('.weight_g'):
            base = k[:-len('.weight_g')]
            wn_bases.add(base)

    result = {}
    for k, v in raw.items():
        if k.endswith('.weight_g') or k.endswith('.weight_v'):
            base = k.rsplit('.', 1)[0]
            if base in wn_bases and k.endswith('.weight_g'):
                g = raw[f'{base}.weight_g']
                vv = raw[f'{base}.weight_v']
                result[f'{base}.weight'] = denorm_weight_norm(g, vv)
            continue
        result[k] = v
    return result


# ── Weight name mapping ─────────────────────────────────────────────
#
# PyTorch state_dict keys (after weights_loading.py remapping) ->
# GGUF tensor names.
#
# The runtime indexes layers as:
#   backbone.layers.{i}.self_attn.{in_proj,out_proj}.weight
#   backbone.layers.{i}.{norm1,norm2}.{weight,bias}
#   backbone.layers.{i}.{linear1,linear2}.weight

def map_flow_lm_name(key: str) -> str | None:
    """Map FlowLM state_dict key to GGUF tensor name."""
    # Skip keys we don't need
    if key in ('num_ema_updates',):
        return None
    if 'learnt_padding' in key:
        return None

    # Conditioner embedding
    if key == 'conditioner.embed.weight':
        return 'flow_lm.conditioner.embed.weight'

    # Speaker projection (voice cloning)
    if key == 'speaker_proj_weight':
        return 'flow_lm.speaker_proj.weight'

    # BOS embedding
    if key == 'bos_emb':
        return 'flow_lm.bos_emb'
    if key == 'bos_before_voice':
        return 'flow_lm.bos_before_voice'

    # Input linear
    if key == 'input_linear.weight':
        return 'flow_lm.input_linear.weight'

    # Output norm
    if key == 'out_norm.weight':
        return 'flow_lm.out_norm.weight'
    if key == 'out_norm.bias':
        return 'flow_lm.out_norm.bias'

    # EOS head
    if key == 'out_eos.weight':
        return 'flow_lm.out_eos.weight'
    if key == 'out_eos.bias':
        return 'flow_lm.out_eos.bias'

    # Latent stats
    if key == 'emb_std':
        return 'flow_lm.emb_std'
    if key == 'emb_mean':
        return 'flow_lm.emb_mean'

    # Transformer backbone layers
    if key.startswith('transformer.layers.'):
        rest = key[len('transformer.layers.'):]
        return f'flow_lm.transformer.{rest}'

    # Flow network (consistency head)
    if key.startswith('flow_net.'):
        return f'flow_lm.flow_net.{key[len("flow_net."):]}'

    print(f"  WARNING: unmapped FlowLM key: {key}", file=sys.stderr)
    return None


def map_mimi_name(key: str, prefix: str = 'mimi') -> str | None:
    """Map Mimi state_dict key to GGUF tensor name."""
    # Skip VQ/codebook keys (not used for continuous VAE)
    if 'quantizer.vq.' in key or '_codebook' in key:
        return None
    if key == 'quantizer.logvar_proj.weight' or key == 'quantizer.logvar_param':
        return None
    if key.endswith('.weight_v') or key.endswith('.weight_g'):
        return None  # should be folded already
    if 'wavlm' in key:
        return None

    return f'{prefix}.{key}'


def remap_flow_lm_from_original(key: str) -> str:
    """Remap from original pocket-tts safetensors key to our canonical
    FlowLM key (matching what weights_loading.py produces)."""
    if key == 'condition_provider.conditioners.transcript_in_segment.embed.weight':
        return 'conditioner.embed.weight'
    if key == 'condition_provider.conditioners.speaker_wavs.output_proj.weight':
        return 'speaker_proj_weight'
    if key == 'fuser.padding_value':
        return 'bos_before_voice'
    if key.startswith('condition_provider.conditioners.transcript_in_segment.learnt_padding'):
        return '__skip__'
    if key.startswith('condition_provider.conditioners.speaker_wavs.learnt_padding'):
        return '__skip__'
    if key == 'num_ema_updates':
        return '__skip__'
    if key.startswith('flow.w_s_t.'):
        return '__skip__'
    # Remap .self_attn.in_proj_weight -> .self_attn.in_proj.weight
    new = key.replace('.self_attn.in_proj_weight', '.self_attn.in_proj.weight')
    return new


def remap_mimi_from_original(key: str) -> str | None:
    """Remap from original Mimi safetensors key to canonical key."""
    if (key.startswith('model.quantizer.vq.') or '_codebook' in key
            or key.endswith('.weight_v') or key.endswith('.weight_g')
            or key == 'model.quantizer.logvar_proj.weight'
            or key == 'quantizer.logvar_proj.weight'
            or key == 'quantizer.logvar_param'
            or 'wavlm' in key):
        return None
    new = key.removeprefix('model.')
    new = new.replace('.conv.conv.', '.conv.')
    new = new.replace('.convtr.convtr.', '.convtr.')
    new = new.replace('in_proj_weight', 'in_proj.weight')
    return new


# ── Main converter ──────────────────────────────────────────────────

def convert(args):
    model_dir = load_model_dir(args.input)
    lang = args.language

    # Find config YAML
    config_path = None
    for candidate in [
        model_dir / 'pocket_tts' / 'config' / f'{lang}.yaml',
        model_dir / 'config' / f'{lang}.yaml',
        model_dir / f'{lang}.yaml',
    ]:
        if candidate.exists():
            config_path = candidate
            break

    if config_path is None:
        # Try downloading just the config
        if hf_hub_download is not None:
            try:
                config_path = Path(hf_hub_download(
                    args.input, f'pocket_tts/config/{lang}.yaml'))
            except Exception:
                pass

    if config_path:
        print(f"Using config: {config_path}", file=sys.stderr)
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        print(f"No config YAML found, using defaults for '{lang}'", file=sys.stderr)
        config = None

    # Parse hparams from config if available
    hparams = dict(FLOW_LM_HPARAMS)
    flow_head = dict(FLOW_HEAD_HPARAMS)
    mimi = dict(MIMI_HPARAMS)
    seanet_ratios = list(SEANET_RATIOS)

    if config:
        fl = config.get('flow_lm', {})
        xfmr = fl.get('transformer', {})
        if xfmr:
            hparams['d_model'] = xfmr.get('d_model', hparams['d_model'])
            hparams['num_heads'] = xfmr.get('num_heads', hparams['num_heads'])
            hparams['num_layers'] = xfmr.get('num_layers', hparams['num_layers'])
            hparams['hidden_scale'] = xfmr.get('hidden_scale', hparams['hidden_scale'])
            hparams['max_period'] = xfmr.get('max_period', hparams['max_period'])
        lut = fl.get('lookup_table', {})
        if lut:
            hparams['n_bins'] = lut.get('n_bins', hparams['n_bins'])
            hparams['lut_dim'] = lut.get('dim', hparams['lut_dim'])
        flow = fl.get('flow', {})
        if flow:
            flow_head['flow_dim'] = flow.get('dim', flow_head['flow_dim'])
            flow_head['flow_depth'] = flow.get('depth', flow_head['flow_depth'])
        if fl.get('insert_bos_before_voice') is not None:
            hparams['insert_bos_before_voice'] = int(fl['insert_bos_before_voice'])

        mi = config.get('mimi', {})
        if mi:
            mimi['sample_rate'] = mi.get('sample_rate', mimi['sample_rate'])
            mimi['inner_dim'] = mi.get('inner_dim', mimi['inner_dim'])
            mimi['outer_dim'] = mi.get('outer_dim', mimi['outer_dim'])
            mimi['channels'] = mi.get('channels', mimi['channels'])
            fr = mi.get('frame_rate', 12.5)
            # Store as rational: 12.5 = 25/2
            mimi['frame_rate_num'] = int(fr * 2)
            mimi['frame_rate_den'] = 2
            sn = mi.get('seanet', {})
            if sn:
                mimi['seanet_dimension'] = sn.get('dimension', mimi['seanet_dimension'])
                mimi['seanet_n_filters'] = sn.get('n_filters', mimi['seanet_n_filters'])
                mimi['seanet_n_residual_layers'] = sn.get('n_residual_layers', mimi['seanet_n_residual_layers'])
                mimi['seanet_kernel_size'] = sn.get('kernel_size', mimi['seanet_kernel_size'])
                mimi['seanet_residual_kernel_size'] = sn.get('residual_kernel_size', mimi['seanet_residual_kernel_size'])
                mimi['seanet_last_kernel_size'] = sn.get('last_kernel_size', mimi['seanet_last_kernel_size'])
                mimi['seanet_dilation_base'] = sn.get('dilation_base', mimi['seanet_dilation_base'])
                mimi['seanet_compress'] = sn.get('compress', mimi['seanet_compress'])
                if sn.get('ratios'):
                    seanet_ratios = sn['ratios']
            xf = mi.get('transformer', {})
            if xf:
                mimi['xfmr_d_model'] = xf.get('d_model', mimi['xfmr_d_model'])
                mimi['xfmr_num_heads'] = xf.get('num_heads', mimi['xfmr_num_heads'])
                mimi['xfmr_num_layers'] = xf.get('num_layers', mimi['xfmr_num_layers'])
                mimi['xfmr_dim_feedforward'] = xf.get('dim_feedforward', mimi['xfmr_dim_feedforward'])
                mimi['xfmr_context'] = xf.get('context', mimi['xfmr_context'])
                if xf.get('layer_scale') is not None:
                    mimi['xfmr_layer_scale_init'] = xf['layer_scale']
            qt = mi.get('quantizer', {})
            if qt:
                mimi['quant_in_dim'] = qt.get('dimension', mimi['quant_in_dim'])
                mimi['quant_out_dim'] = qt.get('output_dimension', mimi['quant_out_dim'])

    # Find weight files
    # The combined model.safetensors contains everything
    weights_path = None
    for candidate in [
        model_dir / 'languages' / lang / 'model.safetensors',
        model_dir / 'model.safetensors',
    ]:
        if candidate.exists():
            weights_path = candidate
            break

    # Alternatively, separate FlowLM + Mimi files
    flow_lm_path = None
    mimi_path = None
    if weights_path is None:
        for candidate in [
            model_dir / 'languages' / lang / 'flow_lm.safetensors',
            model_dir / 'flow_lm.safetensors',
            model_dir / 'final.safetensors',
        ]:
            if candidate.exists():
                flow_lm_path = candidate
                break
        for candidate in [
            model_dir / 'languages' / lang / 'codec.safetensors',
            model_dir / 'codec.safetensors',
            model_dir / 'mimi.safetensors',
        ]:
            if candidate.exists():
                mimi_path = candidate
                break

    if weights_path is None and flow_lm_path is None:
        sys.exit(f"No model weights found in {model_dir}")

    # Find tokenizer
    tokenizer_path = None
    for candidate in [
        model_dir / 'languages' / lang / 'tokenizer.model',
        model_dir / 'tokenizer.model',
    ]:
        if candidate.exists():
            tokenizer_path = candidate
            break

    # Output path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vc_suffix = '' if args.voice_cloning else '-novc'
    output_path = out_dir / f'pocket-tts-{lang}{vc_suffix}-f16.gguf'

    print(f"\n=== Writing Pocket TTS GGUF: {output_path} ===", file=sys.stderr)

    writer = GGUFWriter(str(output_path), "pocket-tts")

    # ── Write hyperparameters ──
    writer.add_string("pocket_tts.language", lang)
    writer.add_uint32("pocket_tts.has_voice_cloning", 1 if args.voice_cloning else 0)

    for k, v in hparams.items():
        key = f"pocket_tts.flow_lm.{k}"
        if isinstance(v, int):
            writer.add_uint32(key, v)
        elif isinstance(v, float):
            writer.add_float32(key, v)

    for k, v in flow_head.items():
        key = f"pocket_tts.flow_head.{k}"
        if isinstance(v, int):
            writer.add_uint32(key, v)
        elif isinstance(v, float):
            writer.add_float32(key, v)

    for k, v in mimi.items():
        key = f"pocket_tts.mimi.{k}"
        if isinstance(v, int):
            writer.add_uint32(key, v)
        elif isinstance(v, float):
            writer.add_float32(key, v)

    writer.add_array("pocket_tts.mimi.seanet_ratios",
                     [int(r) for r in seanet_ratios])

    # ── Write tokenizer ──
    if tokenizer_path:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(tokenizer_path))
        vocab = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]
        scores = [sp.GetScore(i) for i in range(sp.GetPieceSize())]
        writer.add_array("tokenizer.ggml.tokens", vocab)
        writer.add_array("tokenizer.ggml.scores", scores)
        writer.add_string("tokenizer.ggml.model", "unigram")
        print(f"  Tokenizer: {len(vocab)} pieces from {tokenizer_path}",
              file=sys.stderr)
    else:
        print("  WARNING: no tokenizer.model found!", file=sys.stderr)

    # ── Load and write weights ──
    if weights_path:
        # Combined model: contains both FlowLM + Mimi + speaker_proj
        print(f"  Loading combined weights: {weights_path}", file=sys.stderr)
        all_tensors = {}
        with safe_open(str(weights_path), framework='pt') as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

        # Split into FlowLM and Mimi
        flow_lm_tensors = {}
        mimi_tensors = {}
        for key, tensor in all_tensors.items():
            if key.startswith('mimi.'):
                # This is Mimi (from combined model.safetensors)
                mimi_key = key[len('mimi.'):]
                mimi_key = remap_mimi_from_original(mimi_key)
                if mimi_key:
                    mimi_tensors[mimi_key] = tensor
            elif key.startswith('flow_lm.'):
                # FlowLM
                fl_key = key[len('flow_lm.'):]
                fl_key = remap_flow_lm_from_original(fl_key)
                if fl_key != '__skip__':
                    flow_lm_tensors[fl_key] = tensor
            else:
                # Try as FlowLM key
                fl_key = remap_flow_lm_from_original(key)
                if fl_key != '__skip__':
                    flow_lm_tensors[fl_key] = tensor
    else:
        # Separate files
        if flow_lm_path:
            print(f"  Loading FlowLM weights: {flow_lm_path}", file=sys.stderr)
            flow_lm_raw = {}
            with safe_open(str(flow_lm_path), framework='pt') as f:
                for key in f.keys():
                    flow_lm_raw[key] = f.get_tensor(key)
            flow_lm_tensors = {}
            for key, tensor in flow_lm_raw.items():
                new_key = remap_flow_lm_from_original(key)
                if new_key != '__skip__':
                    flow_lm_tensors[new_key] = tensor
        else:
            flow_lm_tensors = {}

        if mimi_path:
            print(f"  Loading Mimi weights: {mimi_path}", file=sys.stderr)
            mimi_raw = load_safetensors_with_weight_norm(mimi_path)
            mimi_tensors = {}
            for key, tensor in mimi_raw.items():
                new_key = remap_mimi_from_original(key)
                if new_key:
                    mimi_tensors[new_key] = tensor
        else:
            mimi_tensors = {}

    # Write FlowLM tensors
    n_flow = 0
    for key in sorted(flow_lm_tensors.keys()):
        tensor = flow_lm_tensors[key]
        gguf_name = map_flow_lm_name(key)
        if gguf_name is None:
            continue
        data, dtype = choose_dtype(gguf_name, list(tensor.shape), tensor)
        writer.add_tensor(gguf_name, data, raw_dtype=dtype)
        n_flow += 1
    print(f"  FlowLM: {n_flow} tensors", file=sys.stderr)

    # Write Mimi tensors
    # For non-voice-cloning, we only need the decoder + quantizer
    # For voice cloning, we also need the encoder
    n_mimi = 0
    for key in sorted(mimi_tensors.keys()):
        tensor = mimi_tensors[key]
        # Skip encoder weights if not doing voice cloning
        if not args.voice_cloning:
            if key.startswith('encoder.') or key.startswith('encoder_transformer.'):
                continue
            if key == 'downsample.conv.weight':
                continue
        gguf_name = map_mimi_name(key)
        if gguf_name is None:
            continue
        data, dtype = choose_dtype(gguf_name, list(tensor.shape), tensor)
        writer.add_tensor(gguf_name, data, raw_dtype=dtype)
        n_mimi += 1
    print(f"  Mimi: {n_mimi} tensors", file=sys.stderr)

    # ── Finalize ──
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Written: {output_path} ({size_mb:.1f} MB)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Pocket TTS safetensors to GGUF")
    parser.add_argument("--input", required=True,
                        help="HuggingFace model ID or local directory")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for GGUF")
    parser.add_argument("--language", default="english",
                        help="Language config to use (default: english)")
    parser.add_argument("--voice-cloning", action="store_true",
                        help="Include Mimi encoder for voice cloning")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
