#!/usr/bin/env python3
"""
Dump reference activations for Haas-ja-v1.0 on an audio file.

Captures:
  - raw_audio
  - mel_spectrogram (128, T_mel)
  - encoder_output  (T_enc, 1024)
  - decoder_sos     (640,)
  - joint_sos_t0    (3078,)
  - reference text
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import yaml
import sentencepiece as spm
import gguf


def load_extracted(model_dir: Path):
    ckpt_path = model_dir / "model_weights.ckpt"
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    cfg = yaml.safe_load((model_dir / "model_config.yaml").read_text("utf-8"))
    spm_file = next(model_dir.glob("*tokenizer.model"))
    sp = spm.SentencePieceProcessor()
    sp.Load(str(spm_file))
    return sd, cfg, sp


def compute_mel(audio: np.ndarray, fb_tensor: torch.Tensor, win_tensor: torch.Tensor, n_fft=512, hop=160, win_len=400):
    # Match NeMo AudioToMelSpectrogramPreprocessor exactly
    x = torch.from_numpy(audio).float().unsqueeze(0) # [1, N]
    # STFT
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_len,
        window=win_tensor.float(),
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True
    ) # [1, 257, T]
    mag = torch.abs(stft).pow(2.0)
    fb = fb_tensor.float()
    if fb.dim() == 3:
        fb = fb.squeeze(0) # [128, 257]
    mel = torch.matmul(fb, mag) # [1, 128, T]
    mel = torch.log(mel.clamp(min=1e-5))
    # NeMo normalize: per_feature: (mel - mean) / (std + 1e-5)
    mean = mel.mean(dim=-1, keepdim=True)
    std = mel.std(dim=-1, keepdim=True)
    mel = (mel - mean) / (std + 1e-5)
    return mel


def run_kda_step(
    x: torch.Tensor, # [640]
    conv_states: list[dict], # per layer: conv_q, conv_k, conv_v [320, 4]
    recurrent_states: list[torch.Tensor], # per layer: S [5, 64, 64]
    sd: dict,
    eps_norm=1e-5,
    eps_l2=1e-6
):
    H, D = 5, 64
    out_x = x.clone().float()

    for l in range(2):
        norm_w = sd[f"decoder.prediction.dec_rnn.norms.{l}.weight"].float()
        norm_b = sd[f"decoder.prediction.dec_rnn.norms.{l}.bias"].float()
        x_norm = F.layer_norm(out_x, (640,), weight=norm_w, bias=norm_b, eps=eps_norm)

        # Projections
        qw = sd[f"decoder.prediction.dec_rnn.layers.{l}.q_proj.weight"].float()
        kw = sd[f"decoder.prediction.dec_rnn.layers.{l}.k_proj.weight"].float()
        vw = sd[f"decoder.prediction.dec_rnn.layers.{l}.v_proj.weight"].float()
        q_raw = F.linear(x_norm, qw) # [320]
        k_raw = F.linear(x_norm, kw) # [320]
        v_raw = F.linear(x_norm, vw) # [320]

        # Conv1d step
        cs = conv_states[l]
        cs["q"] = torch.cat([cs["q"][:, 1:], q_raw.unsqueeze(-1)], dim=-1)
        cs["k"] = torch.cat([cs["k"][:, 1:], k_raw.unsqueeze(-1)], dim=-1)
        cs["v"] = torch.cat([cs["v"][:, 1:], v_raw.unsqueeze(-1)], dim=-1)

        q_cw = sd[f"decoder.prediction.dec_rnn.layers.{l}.q_conv1d.weight"].float().squeeze(1) # [320, 4]
        k_cw = sd[f"decoder.prediction.dec_rnn.layers.{l}.k_conv1d.weight"].float().squeeze(1) # [320, 4]
        v_cw = sd[f"decoder.prediction.dec_rnn.layers.{l}.v_conv1d.weight"].float().squeeze(1) # [320, 4]

        q_conv = F.silu((cs["q"] * q_cw).sum(-1)) # [320]
        k_conv = F.silu((cs["k"] * k_cw).sum(-1)) # [320]
        v_conv = F.silu((cs["v"] * v_cw).sum(-1)) # [320]

        # Reshape to heads [5, 64]
        q = q_conv.view(H, D)
        k = k_conv.view(H, D)
        v = v_conv.view(H, D)

        # L2 norm over head dim D = 64
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + eps_l2)
        q = q * (1.0 / math.sqrt(D))
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + eps_l2)

        # Gate g
        fw0 = sd[f"decoder.prediction.dec_rnn.layers.{l}.f_proj.0.weight"].float()
        fw1 = sd[f"decoder.prediction.dec_rnn.layers.{l}.f_proj.1.weight"].float()
        f_mid = F.linear(x_norm, fw0)
        f_out = F.linear(f_mid, fw1).view(H, D)
        dt_bias = sd[f"decoder.prediction.dec_rnn.layers.{l}.dt_bias"].float().view(H, D)
        A_log = sd[f"decoder.prediction.dec_rnn.layers.{l}.A_log"].float().view(H, 1)
        g = -A_log.exp() * F.softplus(f_out + dt_bias) # [5, 64]

        # Beta
        bw = sd[f"decoder.prediction.dec_rnn.layers.{l}.b_proj.weight"].float()
        b_out = F.linear(x_norm, bw)
        beta = torch.sigmoid(b_out) # [5]

        # Recurrent state S [5, 64, 64] with state_v_first=True: S[h, v, k]
        S = recurrent_states[l]
        # Decay columns (k dimension)
        S = S * g.exp().unsqueeze(1) # [5, 1, 64] broadcasts over v (dim 1)

        # Projection onto k: p_v = sum_k S[h, v, k] * k[h, k]
        p = torch.einsum('h v k, h k -> h v', S, k) # [5, 64]
        diff = v - p
        u = beta.unsqueeze(-1) * diff # [5, 64]
        S = S + torch.einsum('h v, h k -> h v k', u, k)
        recurrent_states[l] = S

        # Attention output: o_v = sum_k S[h, v, k] * q[h, k]
        o = torch.einsum('h v k, h k -> h v', S, q) # [5, 64]

        # Output norm & gate
        gw0 = sd[f"decoder.prediction.dec_rnn.layers.{l}.g_proj.0.weight"].float()
        gw1 = sd[f"decoder.prediction.dec_rnn.layers.{l}.g_proj.1.weight"].float()
        gb1 = sd[f"decoder.prediction.dec_rnn.layers.{l}.g_proj.1.bias"].float()
        g_mid = F.linear(x_norm, gw0)
        g_out = (F.linear(g_mid, gw1) + gb1).view(H, D)

        o_norm_w = sd[f"decoder.prediction.dec_rnn.layers.{l}.o_norm.weight"].float()
        # RMSNorm per head over D
        rms = torch.sqrt((o * o).mean(-1, keepdim=True) + eps_norm)
        o_normed = (o / rms) * o_norm_w.unsqueeze(0)
        y = o_normed * torch.sigmoid(g_out) # [5, 64]
        y_flat = y.view(320)

        ow = sd[f"decoder.prediction.dec_rnn.layers.{l}.o_proj.weight"].float()
        proj = F.linear(y_flat, ow) # [640]
        out_x = out_x + proj

    fn_w = sd["decoder.prediction.dec_rnn.final_norm.weight"].float()
    fn_b = sd["decoder.prediction.dec_rnn.final_norm.bias"].float()
    out_final = F.layer_norm(out_x, (640,), weight=fn_w, bias=fn_b, eps=eps_norm)
    return out_final


def main():
    parser = argparse.ArgumentParser(description="Dump Haas-ja reference activations")
    parser.add_argument("--model-dir", type=Path, default=Path("/Volumes/backups/ai/cache/haas_extracted"))
    parser.add_argument("--audio", type=Path, default=Path("samples/ja_test.wav"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/haas-ja-ref.gguf"))
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    sd, cfg, sp = load_extracted(args.model_dir)

    print(f"Loading audio from {args.audio}...")
    audio, sr = sf.read(str(args.audio))
    if sr != 16000:
        sys.exit(f"Expected 16kHz audio, got {sr}")
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Preprocessor
    fb = sd["preprocessor.featurizer.fb"]
    win = sd["preprocessor.featurizer.window"]
    mel = compute_mel(audio, fb, win) # [1, 128, T_mel]
    mel_2d = mel.squeeze(0) # [128, T_mel]
    print(f"Computed mel spectrogram: {tuple(mel_2d.shape)}")

    # Predictor at SOS (token = 3072, blank)
    blank_id = 3072
    embed = sd["decoder.prediction.embed.weight"].float() # [3073, 640]
    sos_x = embed[blank_id] # [640]

    conv_states = [{"q": torch.zeros(320, 4), "k": torch.zeros(320, 4), "v": torch.zeros(320, 4)} for _ in range(2)]
    recurrent_states = [torch.zeros(5, 64, 64) for _ in range(2)]

    pred_sos = run_kda_step(sos_x, conv_states, recurrent_states, sd) # [640]
    print(f"Computed SOS predictor output: mean={pred_sos.mean():.4f}, std={pred_sos.std():.4f}")

    # FastConformer encoder
    # Load NeMo ConformerEncoder to run the encoder forward pass
    print("Loading Conformer encoder in PyTorch...")
    import nemo.collections.asr as nemo_asr
    enc_cfg = cfg["encoder"]
    from omegaconf import OmegaConf
    conf = OmegaConf.create(enc_cfg)
    encoder = nemo_asr.modules.ConformerEncoder(
        feat_in=conf.feat_in,
        n_layers=conf.n_layers,
        d_model=conf.d_model,
        subsampling=conf.subsampling,
        subsampling_factor=conf.subsampling_factor,
        subsampling_conv_channels=conf.subsampling_conv_channels,
        causal_downsampling=conf.causal_downsampling,
        ff_expansion_factor=conf.ff_expansion_factor,
        self_attention_model=conf.self_attention_model,
        n_heads=conf.n_heads,
        att_context_size=conf.att_context_size,
        att_context_style=conf.att_context_style,
        xscaling=conf.xscaling,
        untie_biases=conf.untie_biases,
        pos_emb_max_len=conf.pos_emb_max_len,
        conv_kernel_size=conf.conv_kernel_size,
        conv_norm_type=conf.conv_norm_type,
        use_bias=conf.use_bias,
        dropout=conf.dropout,
        dropout_pre_encoder=conf.dropout_pre_encoder,
        dropout_emb=conf.dropout_emb,
        dropout_att=conf.dropout_att,
        stochastic_depth_drop_prob=conf.stochastic_depth_drop_prob,
    ).eval()

    # Load encoder weights
    enc_sd = {k[len("encoder."):]: v.float() for k, v in sd.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_sd, strict=False)

    with torch.no_grad():
        mel_len = torch.tensor([mel.shape[-1]])
        enc_out, enc_len = encoder(audio_signal=mel, length=mel_len)
    enc_2d = enc_out.squeeze(0).transpose(0, 1).contiguous() # [T_enc, 1024]
    print(f"Computed encoder output: {tuple(enc_2d.shape)}")

    # Joint at t=0, u=0
    enc_w = sd["joint.enc.weight"].float() # [640, 1024]
    enc_b = sd["joint.enc.bias"].float() # [640]
    pred_w = sd["joint.pred.weight"].float() # [640, 640]
    pred_b = sd["joint.pred.bias"].float() # [640]
    out_w = sd["joint.joint_net.2.weight"].float() # [3078, 640]
    out_b = sd["joint.joint_net.2.bias"].float() # [3078]

    proj_enc_0 = F.linear(enc_2d[0], enc_w, enc_b) # [640]
    proj_pred_0 = F.linear(pred_sos, pred_w, pred_b) # [640]
    joint_mid_0 = F.relu(proj_enc_0 + proj_pred_0) # [640]
    joint_logits_0 = F.linear(joint_mid_0, out_w, out_b) # [3078]
    print(f"Computed joint logits at (0, 0): {tuple(joint_logits_0.shape)}")

    # Greedy decode loop in PyTorch to obtain reference transcript!
    print("Running greedy TDT decode in PyTorch...")
    T_enc = enc_2d.shape[0]
    t = 0
    emitted_tokens = []
    max_per_step = 10

    # Reset states for full decode
    conv_states = [{"q": torch.zeros(320, 4), "k": torch.zeros(320, 4), "v": torch.zeros(320, 4)} for _ in range(2)]
    recurrent_states = [torch.zeros(5, 64, 64) for _ in range(2)]
    pred_u = run_kda_step(embed[blank_id], conv_states, recurrent_states, sd)

    durations = [0, 1, 2, 3, 4]

    while t < T_enc:
        proj_e = F.linear(enc_2d[t], enc_w, enc_b)
        n_inner = 0
        while n_inner < max_per_step:
            proj_p = F.linear(pred_u, pred_w, pred_b)
            mid = F.relu(proj_e + proj_p)
            logits = F.linear(mid, out_w, out_b) # [3078]

            vocab_logits = logits[:blank_id + 1] # 3073
            dur_logits = logits[blank_id + 1:] # 5

            tok = int(torch.argmax(vocab_logits).item())
            dur_idx = int(torch.argmax(dur_logits).item())
            dur = durations[dur_idx]

            if tok == blank_id:
                if dur > 0:
                    t += dur
                    break
                else:
                    n_inner += 1
                    continue
            else:
                emitted_tokens.append(tok)
                pred_u = run_kda_step(embed[tok], conv_states, recurrent_states, sd)
                t += dur
                break
        if n_inner >= max_per_step:
            t += 1

    ref_text = sp.decode(emitted_tokens)
    print(f"\n=======================================================")
    print(f"Reference Transcript ({len(emitted_tokens)} tokens):")
    print(ref_text)
    print(f"=======================================================\n")

    # Write GGUF reference archive
    print(f"Writing reference GGUF to {args.output}...")
    w = gguf.GGUFWriter(str(args.output), arch="crispasr.reference")
    w.add_string("crispasr.reference.backend", "parakeet")
    w.add_string("crispasr.reference.transcript", ref_text)

    # Stages expected by crispasr-diff:
    w.add_tensor("raw_audio", audio.astype(np.float32))
    w.add_tensor("mel_spectrogram", mel_2d.transpose(0, 1).contiguous().numpy().astype(np.float32))
    w.add_tensor("encoder_output", enc_2d.numpy().astype(np.float32))
    w.add_tensor("decoder_sos", pred_sos.numpy().astype(np.float32))
    w.add_tensor("joint_sos_t0", joint_logits_0.numpy().astype(np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("Done!")


if __name__ == "__main__":
    main()
