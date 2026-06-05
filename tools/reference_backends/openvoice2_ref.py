#!/usr/bin/env python
"""OpenVoice2 TCC reference — dump intermediate values for diff harness."""
import json, struct, sys, os
import numpy as np
import torch
import torch.nn.functional as F

DUMP_DIR = os.environ.get("OV2_DUMP_DIR", "/tmp/ov2-ref")
os.makedirs(DUMP_DIR, exist_ok=True)

def dump(name, t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float().numpy()
    t = np.ascontiguousarray(t.flatten())
    path = f"{DUMP_DIR}/{name}.bin"
    t.tofile(path)
    print(f"  {name}: shape orig, {len(t)} elems, "
          f"min={t.min():.6f} max={t.max():.6f} mean={t.mean():.6f}")

def load_wav(path):
    with open(path, "rb") as f:
        raw = f.read()
    pos = 12
    sr = 16000; bits = 16
    while pos < len(raw) - 8:
        cid = raw[pos:pos+4]; csz = struct.unpack_from("<I", raw, pos+4)[0]
        if cid == b"fmt ":
            sr = struct.unpack_from("<I", raw, pos+12)[0]
            bits = struct.unpack_from("<H", raw, pos+22)[0]
        elif cid == b"data":
            n = csz // (bits // 8)
            samples = np.array(struct.unpack_from(f"<{n}h", raw, pos+8), dtype=np.float32) / 32768.0
            return samples, sr
        pos += 8 + csz
    raise ValueError("no data chunk")

def spectrogram_torch(y, n_fft, hop, win_len):
    pad_size = (n_fft - hop) // 2
    y = F.pad(y.unsqueeze(1), (pad_size, pad_size), mode="reflect").squeeze(1)
    window = torch.hann_window(win_len)
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=win_len, window=window,
                      center=False, pad_mode="reflect", normalized=False,
                      onesided=True, return_complex=True)
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
    return spec

def fuse_wn(sd, base):
    g = sd[base + ".weight_g"]
    v = sd[base + ".weight_v"]
    dims = tuple(range(1, v.ndim))
    v_norm = torch.linalg.vector_norm(v, dim=dims, keepdim=True)
    return v * (g / (v_norm + 1e-12))

def resample_linear(pcm, sr_in, sr_out):
    if sr_in == sr_out:
        return pcm
    n_out = int(len(pcm) * sr_out / sr_in)
    indices = np.arange(n_out) * sr_in / sr_out
    idx = indices.astype(int).clip(0, len(pcm) - 2)
    frac = indices - idx
    return pcm[idx] * (1 - frac) + pcm[idx + 1] * frac

def main():
    ckpt_path = sys.argv[1]  # converter/checkpoint.pth
    config_path = sys.argv[2]  # converter/config.json
    src_wav = sys.argv[3]  # source audio
    ref_wav = sys.argv[4]  # reference audio

    with open(config_path) as f:
        cfg = json.load(f)
    hp = cfg["model"]
    dp = cfg["data"]

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]

    sr = dp["sampling_rate"]  # 22050
    n_fft = dp["filter_length"]  # 1024
    hop = dp["hop_length"]  # 256
    win = dp["win_length"]  # 1024
    hidden = hp["hidden_channels"]  # 192
    inter = hp["inter_channels"]  # 192
    gin = hp["gin_channels"]  # 256

    # Load and resample audio
    src_pcm, src_sr = load_wav(src_wav)
    ref_pcm, ref_sr = load_wav(ref_wav)
    src_22k = resample_linear(src_pcm, src_sr, sr)
    ref_22k = resample_linear(ref_pcm, ref_sr, sr)
    print(f"src: {len(src_pcm)}@{src_sr} → {len(src_22k)}@{sr}")
    print(f"ref: {len(ref_pcm)}@{ref_sr} → {len(ref_22k)}@{sr}")

    # STFT
    src_spec = spectrogram_torch(torch.FloatTensor(src_22k).unsqueeze(0), n_fft, hop, win)
    ref_spec = spectrogram_torch(torch.FloatTensor(ref_22k).unsqueeze(0), n_fft, hop, win)
    print(f"src_spec: {src_spec.shape}")
    print(f"ref_spec: {ref_spec.shape}")
    dump("src_spec", src_spec)
    dump("ref_spec", ref_spec)

    # ── ref_enc ──
    print("\n=== ref_enc ===")
    x = ref_spec.transpose(1, 2)  # (1, T, 513)
    # LayerNorm
    ln_w = sd["ref_enc.layernorm.weight"]
    ln_b = sd["ref_enc.layernorm.bias"]
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    x = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b
    x = x.unsqueeze(1)  # (1, 1, T, 513)

    for i in range(6):
        w = fuse_wn(sd, f"ref_enc.convs.{i}")
        b = sd[f"ref_enc.convs.{i}.bias"]
        x = F.conv2d(x, w, b, stride=2, padding=1)
        x = F.relu(x)
        print(f"  conv2d {i}: {x.shape} mean={x.mean():.6f}")
    dump("ref_enc_conv_out", x)

    # GRU: (1, C=128, H=15, W=9) → (1, H=15, C*W=1152) for batch_first GRU
    # Process along the H (frequency) axis, flatten C and W
    C_out, H_out, W_out = x.shape[1], x.shape[2], x.shape[3]
    # permute to (H, C, W) then reshape to (H, C*W)
    x_gru = x.squeeze(0).permute(1, 0, 2).reshape(H_out, C_out * W_out).unsqueeze(0)
    print(f"  GRU input: {x_gru.shape} (H={H_out}, C*W={C_out}*{W_out}={C_out*W_out})")

    gru = torch.nn.GRU(C_out * W_out, 128, batch_first=True)
    gru.weight_ih_l0.data = sd["ref_enc.gru.weight_ih_l0"]
    gru.weight_hh_l0.data = sd["ref_enc.gru.weight_hh_l0"]
    gru.bias_ih_l0.data = sd["ref_enc.gru.bias_ih_l0"]
    gru.bias_hh_l0.data = sd["ref_enc.gru.bias_hh_l0"]
    _, h = gru(x_gru)
    h = h.squeeze(0)  # (1, 128)
    dump("ref_enc_gru_out", h)

    proj_w = sd["ref_enc.proj.weight"]
    proj_b = sd["ref_enc.proj.bias"]
    target_se = F.linear(h, proj_w, proj_b)
    dump("target_se", target_se)
    print(f"  target_se: mean={target_se.mean():.6f} std={target_se.std():.6f}")
    print(f"  first 8: {target_se[0,:8].tolist()}")

    # ── enc_q ──
    print("\n=== enc_q ===")
    # Pre conv: (513, T) → (192, T)
    pre_w = fuse_wn(sd, "enc_q.pre") if "enc_q.pre.weight_g" in sd else sd["enc_q.pre.weight"]
    pre_b = sd["enc_q.pre.bias"]
    x_enc = F.conv1d(src_spec, pre_w, pre_b)
    dump("enc_q_pre", x_enc)
    print(f"  pre: {x_enc.shape} mean={x_enc.mean():.6f}")

    # WaveNet (16 layers, zero_g so no conditioning)
    h_wn = x_enc  # (1, 192, T)
    T = h_wn.shape[2]
    skip_sum = torch.zeros_like(h_wn)

    for il in range(16):
        # Dilated conv
        in_w = fuse_wn(sd, f"enc_q.enc.in_layers.{il}") if f"enc_q.enc.in_layers.{il}.weight_g" in sd else sd[f"enc_q.enc.in_layers.{il}.weight"]
        in_b = sd[f"enc_q.enc.in_layers.{il}.bias"]
        k = in_w.shape[2]
        pad = (k - 1) // 2  # dilation=1
        conv_out = F.conv1d(h_wn, in_w, in_b, padding=pad)

        # Gated activation
        t_val = torch.tanh(conv_out[:, :hidden, :])
        s_val = torch.sigmoid(conv_out[:, hidden:, :])
        gated = t_val * s_val

        # Res+skip
        rs_w = fuse_wn(sd, f"enc_q.enc.res_skip_layers.{il}") if f"enc_q.enc.res_skip_layers.{il}.weight_g" in sd else sd[f"enc_q.enc.res_skip_layers.{il}.weight"]
        rs_b = sd[f"enc_q.enc.res_skip_layers.{il}.bias"]
        rs = F.conv1d(gated, rs_w, rs_b)

        if rs.shape[1] == 2 * hidden:
            h_wn = h_wn + rs[:, :hidden, :]
            skip_sum = skip_sum + rs[:, hidden:, :]
        else:
            skip_sum = skip_sum + rs

    dump("enc_q_wn_out", skip_sum)
    print(f"  WN out: mean={skip_sum.mean():.6f}")

    # Proj: (192, T) → (384, T) → split mean + logvar
    proj_w2 = fuse_wn(sd, "enc_q.proj") if "enc_q.proj.weight_g" in sd else sd["enc_q.proj.weight"]
    proj_b2 = sd["enc_q.proj.bias"]
    stats = F.conv1d(skip_sum, proj_w2, proj_b2)
    m_enc = stats[:, :inter, :]
    logvar = stats[:, inter:, :]

    # Sample z with tau=0.3
    tau = 0.3
    torch.manual_seed(42)
    noise = torch.randn_like(m_enc)
    z = m_enc + torch.exp(logvar) * noise * tau
    dump("enc_q_z", z)
    print(f"  z: {z.shape} mean={z.mean():.6f} std={z.std():.6f}")

    # ── source speaker embedding (pre-saved, matching upstream demo) ──
    print("\n=== src speaker embedding ===")
    import os as _os
    base_se_path = _os.path.join(_os.path.dirname(sys.argv[1]), "..", "base_speakers", "ses", "en-default.pth")
    if _os.path.exists(base_se_path):
        src_se = torch.load(base_se_path, map_location="cpu", weights_only=False)
        src_se = src_se.squeeze()
        if src_se.ndim == 1:
            src_se = src_se.unsqueeze(0)
        print(f"  loaded pre-saved SE from {base_se_path}: {src_se.shape}")
    else:
        # Fallback: extract from source audio via ref_enc
        print(f"  WARNING: no pre-saved SE at {base_se_path}, extracting from source audio")
        x_s = src_spec.transpose(1, 2)
        mean_s = x_s.mean(-1, keepdim=True)
        var_s = x_s.var(-1, unbiased=False, keepdim=True)
        x_s = (x_s - mean_s) / torch.sqrt(var_s + 1e-5) * ln_w + ln_b
        x_s = x_s.unsqueeze(1)
        for i in range(6):
            w = fuse_wn(sd, f"ref_enc.convs.{i}")
            b = sd[f"ref_enc.convs.{i}.bias"]
            x_s = F.conv2d(x_s, w, b, stride=2, padding=1)
            x_s = F.relu(x_s)
        C2, H2, W2 = x_s.shape[1], x_s.shape[2], x_s.shape[3]
        x_s_gru = x_s.squeeze(0).permute(1, 0, 2).reshape(H2, C2*W2).unsqueeze(0)
        gru2 = torch.nn.GRU(C2*W2, 128, batch_first=True)
        gru2.weight_ih_l0.data = sd["ref_enc.gru.weight_ih_l0"]
        gru2.weight_hh_l0.data = sd["ref_enc.gru.weight_hh_l0"]
        gru2.bias_ih_l0.data = sd["ref_enc.gru.bias_ih_l0"]
        gru2.bias_hh_l0.data = sd["ref_enc.gru.bias_hh_l0"]
        _, h_s = gru2(x_s_gru)
        src_se = F.linear(h_s.squeeze(0), proj_w, proj_b)
    dump("src_se", src_se)
    print(f"  src_se: mean={src_se.mean():.6f}")

    # ── flow forward (z → z_p with src_se) ──
    print("\n=== flow forward ===")
    z_flow = z.clone()
    half = inter // 2

    for fi in range(4):
        idx = fi * 2
        # Flip
        z_flow = z_flow.flip(1)
        z0 = z_flow[:, :half, :]
        z1 = z_flow[:, half:, :]

        # Pre conv
        pre = fuse_wn(sd, f"flow.flows.{idx}.pre") if f"flow.flows.{idx}.pre.weight_g" in sd else sd[f"flow.flows.{idx}.pre.weight"]
        pre_b_ = sd[f"flow.flows.{idx}.pre.bias"]
        h_f = F.conv1d(z0, pre, pre_b_)

        # WN (4 layers with src_se conditioning)
        cond_w = fuse_wn(sd, f"flow.flows.{idx}.enc.cond_layer") if f"flow.flows.{idx}.enc.cond_layer.weight_g" in sd else sd[f"flow.flows.{idx}.enc.cond_layer.weight"]
        cond_b_ = sd[f"flow.flows.{idx}.enc.cond_layer.bias"]
        g_cond = F.conv1d(src_se.unsqueeze(-1), cond_w, cond_b_)  # (1, 2*hidden*4, 1)

        skip_f = torch.zeros(1, hidden, T)
        for il in range(4):
            in_w_ = fuse_wn(sd, f"flow.flows.{idx}.enc.in_layers.{il}") if f"flow.flows.{idx}.enc.in_layers.{il}.weight_g" in sd else sd[f"flow.flows.{idx}.enc.in_layers.{il}.weight"]
            in_b_ = sd[f"flow.flows.{idx}.enc.in_layers.{il}.bias"]
            k = in_w_.shape[2]
            conv_o = F.conv1d(h_f, in_w_, in_b_, padding=(k-1)//2)
            # Add conditioning slice
            offset = il * 2 * hidden
            conv_o = conv_o + g_cond[:, offset:offset + 2*hidden, :]
            t_v = torch.tanh(conv_o[:, :hidden, :])
            s_v = torch.sigmoid(conv_o[:, hidden:, :])
            gated_f = t_v * s_v
            rs_w_ = fuse_wn(sd, f"flow.flows.{idx}.enc.res_skip_layers.{il}") if f"flow.flows.{idx}.enc.res_skip_layers.{il}.weight_g" in sd else sd[f"flow.flows.{idx}.enc.res_skip_layers.{il}.weight"]
            rs_b_ = sd[f"flow.flows.{idx}.enc.res_skip_layers.{il}.bias"]
            rs_f = F.conv1d(gated_f, rs_w_, rs_b_)
            if rs_f.shape[1] == 2 * hidden:
                h_f = h_f + rs_f[:, :hidden, :]
                skip_f = skip_f + rs_f[:, hidden:, :]
            else:
                skip_f = skip_f + rs_f

        # Post conv
        post_w_ = fuse_wn(sd, f"flow.flows.{idx}.post") if f"flow.flows.{idx}.post.weight_g" in sd else sd[f"flow.flows.{idx}.post.weight"]
        post_b_ = sd[f"flow.flows.{idx}.post.bias"]
        m_f = F.conv1d(skip_f, post_w_, post_b_)

        # Forward affine: z1 = z1 + m
        z1 = z1 + m_f
        z_flow = torch.cat([z0, z1], dim=1)

    dump("z_after_flow_fwd", z_flow)
    print(f"  z_p: mean={z_flow.mean():.6f} std={z_flow.std():.6f}")

    # ── flow reverse (z_p → z_hat with target_se) ──
    print("\n=== flow reverse ===")
    for fi in range(3, -1, -1):
        idx = fi * 2
        z_flow = z_flow.flip(1)
        z0 = z_flow[:, :half, :]
        z1 = z_flow[:, half:, :]

        pre = fuse_wn(sd, f"flow.flows.{idx}.pre") if f"flow.flows.{idx}.pre.weight_g" in sd else sd[f"flow.flows.{idx}.pre.weight"]
        pre_b_ = sd[f"flow.flows.{idx}.pre.bias"]
        h_f = F.conv1d(z0, pre, pre_b_)

        cond_w = fuse_wn(sd, f"flow.flows.{idx}.enc.cond_layer") if f"flow.flows.{idx}.enc.cond_layer.weight_g" in sd else sd[f"flow.flows.{idx}.enc.cond_layer.weight"]
        cond_b_ = sd[f"flow.flows.{idx}.enc.cond_layer.bias"]
        g_cond = F.conv1d(target_se.unsqueeze(-1), cond_w, cond_b_)

        skip_f = torch.zeros(1, hidden, T)
        for il in range(4):
            in_w_ = fuse_wn(sd, f"flow.flows.{idx}.enc.in_layers.{il}") if f"flow.flows.{idx}.enc.in_layers.{il}.weight_g" in sd else sd[f"flow.flows.{idx}.enc.in_layers.{il}.weight"]
            in_b_ = sd[f"flow.flows.{idx}.enc.in_layers.{il}.bias"]
            k = in_w_.shape[2]
            conv_o = F.conv1d(h_f, in_w_, in_b_, padding=(k-1)//2)
            offset = il * 2 * hidden
            conv_o = conv_o + g_cond[:, offset:offset + 2*hidden, :]
            t_v = torch.tanh(conv_o[:, :hidden, :])
            s_v = torch.sigmoid(conv_o[:, hidden:, :])
            gated_f = t_v * s_v
            rs_w_ = fuse_wn(sd, f"flow.flows.{idx}.enc.res_skip_layers.{il}") if f"flow.flows.{idx}.enc.res_skip_layers.{il}.weight_g" in sd else sd[f"flow.flows.{idx}.enc.res_skip_layers.{il}.weight"]
            rs_b_ = sd[f"flow.flows.{idx}.enc.res_skip_layers.{il}.bias"]
            rs_f = F.conv1d(gated_f, rs_w_, rs_b_)
            if rs_f.shape[1] == 2 * hidden:
                h_f = h_f + rs_f[:, :hidden, :]
                skip_f = skip_f + rs_f[:, hidden:, :]
            else:
                skip_f = skip_f + rs_f

        post_w_ = fuse_wn(sd, f"flow.flows.{idx}.post") if f"flow.flows.{idx}.post.weight_g" in sd else sd[f"flow.flows.{idx}.post.weight"]
        post_b_ = sd[f"flow.flows.{idx}.post.bias"]
        m_f = F.conv1d(skip_f, post_w_, post_b_)

        # Reverse affine: z1 = z1 - m
        z1 = z1 - m_f
        z_flow = torch.cat([z0, z1], dim=1)

    dump("z_after_flow_rev", z_flow)
    print(f"  z_hat: mean={z_flow.mean():.6f} std={z_flow.std():.6f}")

    # ── HiFi-GAN decode (zero_g → no speaker conditioning) ──
    print("\n=== HiFi-GAN decode ===")
    # Fuse all weight norms in dec
    def get_w(name):
        if name + ".weight_g" in sd:
            return fuse_wn(sd, name)
        return sd.get(name + ".weight", sd.get(name))

    x_dec = F.conv1d(z_flow, get_w("dec.conv_pre"), sd["dec.conv_pre.bias"], padding=3)
    dump("dec_conv_pre", x_dec)
    print(f"  conv_pre: {x_dec.shape} mean={x_dec.mean():.6f}")

    upsample_rates = hp["upsample_rates"]
    upsample_kernels = hp["upsample_kernel_sizes"]
    resblock_kernels = hp["resblock_kernel_sizes"]
    resblock_dilations = hp["resblock_dilation_sizes"]
    n_rk = len(resblock_kernels)
    rb_idx = 0

    for us in range(len(upsample_rates)):
        x_dec = F.leaky_relu(x_dec, 0.1)
        stride = upsample_rates[us]
        kernel = upsample_kernels[us]
        up_w = get_w(f"dec.ups.{us}")
        up_b = sd[f"dec.ups.{us}.bias"]
        x_dec = F.conv_transpose1d(x_dec, up_w, up_b, stride=stride,
                                    padding=(kernel - stride) // 2)

        sum_rb = None
        for ri in range(n_rk):
            rk = resblock_kernels[ri]
            y = x_dec
            for di in range(3):
                d = resblock_dilations[ri][di]
                p = (rk * d - d) // 2
                yt = F.leaky_relu(y, 0.1)
                yt = F.conv1d(yt, get_w(f"dec.resblocks.{rb_idx+ri}.convs1.{di}"),
                              sd[f"dec.resblocks.{rb_idx+ri}.convs1.{di}.bias"], padding=p, dilation=d)
                yt = F.leaky_relu(yt, 0.1)
                yt = F.conv1d(yt, get_w(f"dec.resblocks.{rb_idx+ri}.convs2.{di}"),
                              sd[f"dec.resblocks.{rb_idx+ri}.convs2.{di}.bias"], padding=(rk-1)//2)
                y = y + yt
            sum_rb = y if sum_rb is None else sum_rb + y
        x_dec = sum_rb / n_rk
        rb_idx += n_rk

    x_dec = F.leaky_relu(x_dec, 0.1)
    x_dec = F.conv1d(x_dec, sd["dec.conv_post.weight"], padding=3)
    x_dec = torch.tanh(x_dec)
    dump("dec_output", x_dec)
    print(f"  output: {x_dec.shape} mean={x_dec.mean():.6f} "
          f"min={x_dec.min():.6f} max={x_dec.max():.6f}")

    # Write WAV
    pcm = x_dec.squeeze().numpy()
    pcm_16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    out_path = f"{DUMP_DIR}/ref_output.wav"
    with open(out_path, "wb") as f:
        n_bytes = len(pcm_16) * 2
        f.write(struct.pack("<4sI4s", b"RIFF", 36 + n_bytes, b"WAVE"))
        f.write(struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 22050, 22050*2, 2, 16))
        f.write(struct.pack("<4sI", b"data", n_bytes))
        f.write(pcm_16.tobytes())
    print(f"\nWrote {out_path} ({len(pcm_16)} samples)")

if __name__ == "__main__":
    main()
