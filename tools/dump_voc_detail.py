#!/usr/bin/env python3
"""Dump detailed per-sub-stage vocoder activations for debugging."""

import sys, struct, torch, numpy as np
from pathlib import Path

def main():
    model_dir = Path("/mnt/storage/chatterbox")
    mel_path = model_dir / "ref_mel_80x62.bin"
    out_path = model_dir / "voc-ref-detail.gguf"

    # Load the full ChatterboxTTS model (handles weight_norm correctly)
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_local(str(model_dir), device="cpu")
    voc = model.s3gen.mel2wav  # HiFTGenerator
    print(f"Vocoder loaded: {type(voc).__name__}")
    voc.eval()

    # Load mel
    mel_data = np.fromfile(str(mel_path), dtype=np.float32)
    T_mel = 62
    mel = mel_data.reshape(80, T_mel)  # channel-first
    mel_t = torch.from_numpy(mel).unsqueeze(0)  # (1, 80, 62)
    print(f"Mel shape: {mel_t.shape}, rms={mel_t.pow(2).mean().sqrt():.3f}")

    # Run vocoder step by step, capturing intermediates
    stages = {}

    with torch.no_grad():
        x = mel_t

        # conv_pre
        x = voc.conv_pre(x)
        stages["voc_conv_pre"] = x.squeeze(0).numpy()  # (512, 62)
        print(f"conv_pre: {x.shape}, rms={x.pow(2).mean().sqrt():.3f}")

        # 3 upsample stages
        for i in range(voc.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, 0.1)
            x = voc.ups[i](x)
            stages[f"voc_ups_{i}"] = x.squeeze(0).numpy()
            print(f"ups_{i}: {x.shape}, rms={x.pow(2).mean().sqrt():.3f}")

            # Source fusion DISABLED (to match reference)
            # Skip: si = source_downs[i](s_stft); si = source_resblocks[i](si); x = x + si

            # ResBlocks
            xs = None
            for j in range(voc.num_kernels):
                rb_idx = i * voc.num_kernels + j
                rb_out = voc.resblocks[rb_idx](x)

                # Dump per-resblock and per-dilation outputs for stage 0
                if i == 0:
                    stages[f"voc_rb_0_k{j}"] = rb_out.squeeze(0).numpy()

                    # Also run the resblock step-by-step for the first one
                    if j == 0:
                        rb = voc.resblocks[0]
                        x_rb = x.clone()
                        for d_idx in range(len(rb.convs1)):
                            xt = rb.activations1[d_idx](x_rb)
                            stages[f"voc_rb0k0_snake1_d{d_idx}"] = xt.squeeze(0).numpy()
                            xt = rb.convs1[d_idx](xt)
                            stages[f"voc_rb0k0_conv1_d{d_idx}"] = xt.squeeze(0).numpy()
                            xt = rb.activations2[d_idx](xt)
                            stages[f"voc_rb0k0_snake2_d{d_idx}"] = xt.squeeze(0).numpy()
                            xt = rb.convs2[d_idx](xt)
                            stages[f"voc_rb0k0_conv2_d{d_idx}"] = xt.squeeze(0).numpy()
                            x_rb = xt + x_rb
                            stages[f"voc_rb0k0_res_d{d_idx}"] = x_rb.squeeze(0).numpy()

                if xs is None:
                    xs = rb_out
                else:
                    xs += rb_out

            x = xs / voc.num_kernels
            stages[f"voc_rb_{i}"] = x.squeeze(0).numpy()
            print(f"rb_{i}: {x.shape}, rms={x.pow(2).mean().sqrt():.3f}")

        # conv_post
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = voc.conv_post(x)
        x = torch.clamp(x, -2.0, 2.0)
        stages["voc_conv_post"] = x.squeeze(0).numpy()
        print(f"conv_post: {x.shape}, rms={x.pow(2).mean().sqrt():.3f}")

    # Write to GGUF
    from gguf import GGUFWriter
    writer = GGUFWriter(str(out_path), "chatterbox-voc-detail")
    for name, arr in stages.items():
        arr = arr.astype(np.float32)
        print(f"  Writing {name}: shape={arr.shape}")
        writer.add_tensor(name, arr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nWrote {len(stages)} tensors to {out_path}")

if __name__ == "__main__":
    main()
