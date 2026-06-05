"""TitaNet-Large speaker verification reference dump backend.

Loads `nvidia/speakerverification_en_titanet_large` via NeMo and captures
intermediate activations at every architectural boundary for crispasr-diff
comparison against the C++ titanet runtime.

Stages:

  raw_audio           (N,)               input PCM
  mel_spectrogram     (n_mels, T_mel)    preprocessor output (channel-first)
  encoder_output      (epilog_ch, T_enc) after all 5 Jasper blocks
  block_0_output      (C, T)             after prolog block
  block_1_output      (C, T)             after mega block 1
  block_2_output      (C, T)             after mega block 2
  block_3_output      (C, T)             after mega block 3
  block_4_output      (epi_C, T)         after epilog block
  pool_output         (6144,)            after ASP pooling + BN
  embedding           (192,)             final L2-normalized embedding

Usage:

  python tools/dump_reference.py --backend titanet \\
      --model-dir nvidia/speakerverification_en_titanet_large \\
      --audio speaker.wav \\
      --output /tmp/titanet-ref.gguf
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "block_0_output",
    "block_1_output",
    "block_2_output",
    "block_3_output",
    "block_4_output",
    "encoder_output",
    "pool_output",
    "embedding",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run NeMo TitaNet reference forward and return stage captures."""
    import overrides.signature
    overrides.signature.ensure_all_kwargs_defined_in_sub = lambda *a, **kw: None
    overrides.signature.ensure_all_positional_args_defined_in_sub = lambda *a, **kw: None

    import torch
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        raise SystemExit(
            "NeMo toolkit required.\n"
            "Install: pip install 'nemo_toolkit[asr]'\n"
            f"(import error: {e})")

    pretrained = str(model_dir)
    print(f"  loading NeMo TitaNet model from {pretrained}")
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(pretrained)
    model.eval()
    dev = next(model.parameters()).device

    sig = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(dev)
    sig_len = torch.tensor([audio.shape[0]], device=dev)

    out: Dict[str, np.ndarray] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    # ---- Capture per-block encoder activations via forward hooks ----
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    encoder = model.encoder
    # encoder.encoder is nn.ModuleList of JasperBlock modules
    blocks = encoder.encoder
    for bi in range(len(blocks)):
        stage = f"block_{bi}_output"
        if stage in stages:
            def make_hook(name):
                def hook(module, input, output):
                    # output is (x, length) tuple; x is (B, C, T)
                    x = output[0] if isinstance(output, tuple) else output
                    captured[name] = x[0].detach().cpu().float()  # (C, T)
                return hook
            handles.append(blocks[bi].register_forward_hook(make_hook(stage)))

    with torch.no_grad():
        # Preprocessor: mel spectrogram
        feats, feat_len = model.preprocessor(
            input_signal=sig, length=sig_len)
        # feats: (B=1, n_mels=80, T_mel)
        if "mel_spectrogram" in stages:
            m = feats[0].detach().cpu().float().numpy()  # (80, T)
            out["mel_spectrogram"] = m

        # Encoder forward
        enc_out, enc_len = model.encoder(audio_signal=feats, length=feat_len)
        # enc_out: (B=1, C=3072, T_enc)
        T_enc = int(enc_len.item())
        if "encoder_output" in stages:
            e = enc_out[0, :, :T_enc].detach().cpu().float().numpy()  # (3072, T)
            out["encoder_output"] = e

        # Decoder forward — ASP pooling + FC + L2 norm
        # We need to trace through the decoder manually to get intermediates
        decoder = model.decoder

        # ASP pooling
        pool_in = enc_out  # (B, 3072, T)
        lengths = enc_len.unsqueeze(1)  # (B, 1)

        # The decoder._pooling module handles ASP
        pooled, _ = decoder._pooling(pool_in, lengths)
        # pooled: (B, 3072*2=6144)

        # emb_layers: BN + Linear
        emb_out = pooled
        for layer in decoder.emb_layers:
            emb_out = layer(emb_out)
        # emb_out: (B, 192)

        if "pool_output" in stages:
            # Capture after BN, before FC
            pool_bn_out = decoder.emb_layers[0][0](pooled)  # BN
            out["pool_output"] = pool_bn_out[0].detach().cpu().float().numpy()

        if "embedding" in stages:
            # L2 normalize
            emb_normed = torch.nn.functional.normalize(emb_out, p=2, dim=1)
            out["embedding"] = emb_normed[0].detach().cpu().float().numpy()
            print(f"  embedding norm={float(torch.norm(emb_normed[0])):.6f}")
            print(f"  embedding[0:5]={emb_normed[0,:5].tolist()}")

    for h in handles:
        h.remove()

    # Add captured block outputs
    for name, tensor in captured.items():
        out[name] = tensor.numpy()

    return out
