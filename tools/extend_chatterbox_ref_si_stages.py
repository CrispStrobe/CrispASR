"""
Extend an existing chatterbox-ref.gguf archive with the new voc_si_{0,1,2}
and voc_rb_input_{0,1,2} stages without re-running the full chatterbox-ref-venv
forward pass.

Why: the python reference backend now dumps these stages (see
tools/reference_backends/chatterbox.py), and the diff harness expects them
to be there. Re-running the full venv-based dump is expensive (requires the
full upstream chatterbox + safetensors checkpoint, ~6 GB of weights, and
the venv is pruned on this machine). But the new stages can be exactly
computed from tensors that ARE already in the existing archive:

  voc_si_i      = source_resblocks[i](source_downs[i](hift_source_stft))
  voc_rb_input_i = voc_ups_i + voc_si_i  (with crop to min(T_x, T_si))

Both source_downs/source_resblocks are pure-F32 in the s3gen GGUF so torch
matches the original venv-dumped reference to ~1e-6 (verified in the
2026-05-25 hift-drift investigation, see LEARNINGS.md).

This script reads the existing chatterbox-ref.gguf and the s3gen GGUF,
computes the 6 new tensors via torch, writes a new archive that contains
all original tensors plus the new ones, and atomically renames into place.

Usage:
    python tools/extend_chatterbox_ref_si_stages.py \\
        --ref /Volumes/backups/ai/chatterbox-ref.gguf \\
        --s3gen /Volumes/backups/ai/crispasr/chatterbox-s3gen-q8_0.gguf

Pass --dry-run to print what would be written without modifying anything.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F


def load_f32(path: Path, prefix: str = "") -> dict[str, torch.Tensor]:
    r = gguf.GGUFReader(str(path))
    out: dict[str, torch.Tensor] = {}
    for t in r.tensors:
        if prefix and not t.name.startswith(prefix):
            continue
        if t.tensor_type.name != "F32":
            continue
        arr = np.array(t.data, dtype=np.float32).copy()
        torch_shape = tuple(int(s) for s in t.shape)[::-1]
        out[t.name] = torch.from_numpy(arr.reshape(torch_shape))
    return out


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a = alpha.view(1, -1, 1)
    return x + (1.0 / (a + 1e-9)) * torch.sin(x * a) ** 2


# Per-stage configuration from chatterbox/models/s3gen/hifigan.py.
# Matches src/chatterbox_s3gen.cpp::hift_vocoder_cpu line-for-line.
SD_STRIDES = [15, 3, 1]
SD_PADS = [7, 1, 0]
SRB_KERNELS = [7, 7, 11]
SRB_DILATIONS = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


def compute_si_stage(stage: int, s_stft: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    sd_w = w[f"s3.v.sd.{stage}.weight"]
    sd_b = w[f"s3.v.sd.{stage}.bias"]
    si = F.conv1d(s_stft, sd_w, bias=sd_b,
                  stride=SD_STRIDES[stage], padding=SD_PADS[stage], dilation=1)
    k = SRB_KERNELS[stage]
    srb_in = si
    for d, dil in enumerate(SRB_DILATIONS[stage]):
        pad = (k * dil - dil) // 2
        a1 = w[f"s3.v.srb.{stage}.a1.{d}.alpha"]
        si = snake(si, a1)
        c1w = w[f"s3.v.srb.{stage}.c1.{d}.weight"]
        c1b = w[f"s3.v.srb.{stage}.c1.{d}.bias"]
        si = F.conv1d(si, c1w, bias=c1b, stride=1, padding=pad, dilation=dil)
        a2 = w[f"s3.v.srb.{stage}.a2.{d}.alpha"]
        si = snake(si, a2)
        c2w = w[f"s3.v.srb.{stage}.c2.{d}.weight"]
        c2b = w[f"s3.v.srb.{stage}.c2.{d}.bias"]
        si = F.conv1d(si, c2w, bias=c2b, stride=1, padding=(k - 1) // 2, dilation=1)
        si = si + srb_in
        srb_in = si
    return si  # shape (1, C, T_si)


def to_tc_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert (1, C, T) torch tensor to (T, C) numpy float32, matching the
    layout the python reference dumper uses (permute(1, 0).contiguous())."""
    return t.squeeze(0).t().contiguous().cpu().numpy().astype(np.float32, copy=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", type=Path, required=True,
                   help="Path to chatterbox-ref.gguf to extend")
    p.add_argument("--s3gen", type=Path, required=True,
                   help="Path to chatterbox-s3gen-q8_0.gguf (provides F32 vocoder weights)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned tensors without writing anything")
    args = p.parse_args()

    if not args.ref.exists():
        print(f"error: ref archive not found: {args.ref}", file=sys.stderr)
        return 1
    if not args.s3gen.exists():
        print(f"error: s3gen GGUF not found: {args.s3gen}", file=sys.stderr)
        return 1

    print(f"reading {args.ref}")
    src = gguf.GGUFReader(str(args.ref))
    src_tensor_names = {t.name for t in src.tensors}

    print(f"reading vocoder weights from {args.s3gen}")
    w = load_f32(args.s3gen, prefix="s3.v.")

    # Pull the existing inputs we need
    ref_tensors = load_f32(args.ref)
    if "hift_source_stft" not in ref_tensors:
        print("error: ref archive missing hift_source_stft (re-dump the python ref first)",
              file=sys.stderr)
        return 1
    s_stft = ref_tensors["hift_source_stft"].t().unsqueeze(0).contiguous()  # (1, 18, T)
    print(f"  hift_source_stft (1, C, T) = {tuple(s_stft.shape)}")

    new_tensors: dict[str, np.ndarray] = {}
    for stage in range(3):
        ups_name = f"voc_ups_{stage}"
        if ups_name not in ref_tensors:
            print(f"warning: ref missing {ups_name}, skipping stage {stage}", file=sys.stderr)
            continue
        ups_t = ref_tensors[ups_name].t().unsqueeze(0).contiguous()  # (1, C, T)

        with torch.no_grad():
            si = compute_si_stage(stage, s_stft, w)

        T_x = ups_t.shape[-1]
        T_si = si.shape[-1]
        T_min = min(T_x, T_si)
        si_crop = si[..., :T_min]
        ups_crop = ups_t[..., :T_min]
        rb_input = ups_crop + si_crop

        si_np = to_tc_numpy(si_crop)
        rb_input_np = to_tc_numpy(rb_input)
        si_key = f"voc_si_{stage}"
        ri_key = f"voc_rb_input_{stage}"
        new_tensors[si_key] = si_np
        new_tensors[ri_key] = rb_input_np
        print(f"  stage {stage}: si shape (T,C)={si_np.shape}  "
              f"rb_input shape (T,C)={rb_input_np.shape}")

    overlap = sorted(set(new_tensors).intersection(src_tensor_names))
    if overlap:
        print(f"  overwriting existing tensors: {overlap}")

    if args.dry_run:
        print("--dry-run: not writing anything")
        return 0

    # Write a fresh GGUF that has every original tensor plus the new ones.
    # gguf has no in-place append, so we copy all bytes.
    tmp_path = args.ref.with_suffix(args.ref.suffix + ".tmp")
    print(f"writing {tmp_path}")
    writer = gguf.GGUFWriter(str(tmp_path), "chatterbox")

    # KV metadata: passthrough from source. Skip GGUF.* internals (managed by
    # the writer's header/tensor accounting) and general.architecture (set in
    # the GGUFWriter constructor).
    SKIP_FIELDS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for field_name, field in src.fields.items():
        if field_name in SKIP_FIELDS:
            continue
        # Determine type and value via gguf field accessors
        try:
            val = field.contents()
        except Exception:
            val = None
        if val is None:
            continue
        gtype = field.types[0] if field.types else None
        if gtype == gguf.GGUFValueType.STRING:
            writer.add_string(field_name, val)
        elif gtype == gguf.GGUFValueType.UINT32:
            writer.add_uint32(field_name, int(val))
        elif gtype == gguf.GGUFValueType.INT32:
            writer.add_int32(field_name, int(val))
        elif gtype == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(field_name, float(val))
        elif gtype == gguf.GGUFValueType.BOOL:
            writer.add_bool(field_name, bool(val))
        elif gtype == gguf.GGUFValueType.ARRAY:
            # arrays of mixed types — skip to keep this simple
            continue
        # Other types fall through

    # Tensors: passthrough every original, replacing any whose name collides
    # with a new entry. Original order is preserved otherwise.
    for t in src.tensors:
        if t.name in new_tensors:
            data = new_tensors.pop(t.name)
        else:
            data = np.array(t.data, dtype=np.float32 if t.tensor_type.name == "F32" else None).copy()
        writer.add_tensor(t.name, data)
    # Append any genuinely-new tensors
    for name, data in new_tensors.items():
        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Backup original, then atomically replace
    backup = args.ref.with_suffix(args.ref.suffix + ".bak")
    if backup.exists():
        backup.unlink()
    shutil.copy2(args.ref, backup)
    tmp_path.replace(args.ref)
    print(f"done. original backed up to {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
