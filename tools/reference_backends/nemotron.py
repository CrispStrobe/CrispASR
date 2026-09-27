"""
Nemotron-3.5-ASR-Streaming reference backend for crispasr-diff.

model_dir: HF id "nvidia/nemotron-3.5-asr-streaming-0.6b" or local .nemo path.
audio: 16 kHz mono WAV.

Stages:
  mel_spectrogram   — (n_mels, T_mel) log-mel filterbank
  pre_encode_output — (T_enc, d_model) after 4× conv subsampling
  encoder_output    — (T_enc, d_model) after 24L FastConformer + prompt kernel

Requires: nemo_toolkit[asr], torch
"""

import os
import sys
import numpy as np

DEFAULT_STAGES = [
    "mel_spectrogram",
    "pre_encode_output",
    "encoder_output",
]


def dump(model_dir: str, audio, stages=None, max_new_tokens: int = 0, verbose: bool = False, **kwargs):
    """Run NeMo nemotron inference and dump intermediate activations."""
    import torch
    import nemo.collections.asr as nemo_asr

    tensors = {}
    if stages is None:
        stages = {"mel_spectrogram", "pre_encode_output", "encoder_output"}

    # audio is passed as a numpy array, shape (T,)
    # NeMo expects a torch tensor of shape (B, T)
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    audio_len = torch.tensor([audio_t.shape[1]], dtype=torch.long)

    model_dir = str(model_dir)
    print(f"Loading NeMo model from {model_dir} ...")
    if os.path.exists(model_dir) and model_dir.endswith(".nemo"):
        model = nemo_asr.models.EncDecRNNTModel.restore_from(model_dir, map_location="cpu")
    else:
        model = nemo_asr.models.EncDecRNNTModel.from_pretrained(model_dir, map_location="cpu")
    model.eval()

    with torch.no_grad():
        # 1. Mel spectrogram (frontend)
        # processor returns (processed_signal, processed_length)
        processed_signal, processed_length = model.preprocessor(
            input_signal=audio_t, length=audio_len
        )
        if "mel_spectrogram" in stages:
            # NeMo returns (B, n_mels, T)
            tensors["mel_spectrogram"] = processed_signal[0].cpu().numpy()
            if verbose:
                print(f"  mel_spectrogram: {tensors['mel_spectrogram'].shape}")

        # 2. Pre-encoder (causal subsampling)
        if hasattr(model.encoder, "pre_encode"):
            # pre_encode takes (B, T, n_mels) - ConformerEncoder.forward transposes
            # the preprocessor's (B, n_mels, T) before calling it - and returns
            # (B, T_enc, d_model), already the (T_enc, d_model) C++ compares
            # against. Feeding (B, n_mels, T) raised "mat1 and mat2 shapes cannot
            # be multiplied (17x35584 and 4352x1024)" (rebake 2026-09-27).
            pre_enc, pre_len = model.encoder.pre_encode(processed_signal.transpose(1, 2), processed_length)
            if "pre_encode_output" in stages:
                tensors["pre_encode_output"] = pre_enc[0].cpu().numpy()
                if verbose:
                    print(f"  pre_encode_output: {tensors['pre_encode_output'].shape}")
        else:
            pre_enc, pre_len = processed_signal, processed_length

        # 3. Encoder + Prompt MLP
        enc, enc_len = model.encoder(audio_signal=processed_signal, length=processed_length)
        if "encoder_output" in stages:
            tensors["encoder_output"] = enc[0].transpose(0, 1).cpu().numpy()
            if verbose:
                print(f"  encoder_output: {tensors['encoder_output'].shape}")

    return tensors
