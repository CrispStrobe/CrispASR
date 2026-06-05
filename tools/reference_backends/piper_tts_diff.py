#!/usr/bin/env python3
"""
Piper TTS diff harness — dumps every intermediate stage from ONNX reference
and compares against C++ dumps.

Usage:
  # 1. Dump reference intermediates:
  python tools/reference_backends/piper_tts_diff.py dump \
      --onnx /mnt/storage/piper/en_US-lessac-medium.onnx \
      --ipa "həlˈoʊ wˈɜːld" \
      --out /mnt/storage/piper/ref_stages.npz

  # 2. Dump C++ intermediates (run test binary with --dump-stages):
  build/bin/test-piper-tts <model.gguf> <ipa> <wav> 0 0 --dump /mnt/storage/piper/cpp_stages/

  # 3. Compare:
  python tools/reference_backends/piper_tts_diff.py compare \
      --ref /mnt/storage/piper/ref_stages.npz \
      --cpp /mnt/storage/piper/cpp_stages/
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
    import onnx
except ImportError:
    sys.exit("pip install onnxruntime onnx")


# ── All intermediate node names to capture ──
INTERMEDIATES = {
    # Encoder per-layer
    **{f'enc_layer{i}_post_attn': f'/enc_p/encoder/norm_layers_1.{i}/Transpose_1_output_0'
       for i in range(6)},
    **{f'enc_layer{i}_post_ffn': f'/enc_p/encoder/norm_layers_2.{i}/Transpose_1_output_0'
       for i in range(6)},
    # Encoder final
    'enc_output': '/enc_p/encoder/Mul_2_output_0',
    'enc_proj': '/enc_p/proj/Conv_output_0',
    # SDP
    'sdp_pre': '/dp/pre/Conv_output_0',
    'sdp_ddsconv': '/dp/convs/Mul_15_output_0',
    'sdp_proj': '/dp/proj/Conv_output_0',
    'sdp_flow7_proj': '/dp/flows.7/proj/Conv_output_0',
    'sdp_flow7_z': '/dp/flows.7/Mul_35_output_0',
    'sdp_flow5_proj': '/dp/flows.5/proj/Conv_output_0',
    'sdp_flow5_z': '/dp/flows.5/Mul_35_output_0',
    'sdp_flow3_proj': '/dp/flows.3/proj/Conv_output_0',
    'sdp_flow3_z': '/dp/flows.3/Mul_35_output_0',
    'sdp_logw': '/dp/flows.0/Mul_1_output_0',
    # Durations + latent
    'durations': '/Ceil_output_0',
    'z_p': '/Add_output_0',
    # Flow inverse
    'flow6_m': '/flow/flows.6/post/Conv_output_0',
    'flow4_m': '/flow/flows.4/post/Conv_output_0',
    'flow2_m': '/flow/flows.2/post/Conv_output_0',
    'flow0_m': '/flow/flows.0/post/Conv_output_0',
    'z_dec': '/Mul_7_output_0',
    # Decoder
    'dec_conv_pre': '/dec/conv_pre/Conv_output_0',
    'dec_stage0_mrf': '/dec/Div_output_0',
    'dec_stage1_mrf': '/dec/Div_1_output_0',
    'dec_stage2_mrf': '/dec/Div_2_output_0',
}


def encode_phonemes(ipa: str, phoneme_id_map: dict) -> list[int]:
    ids = [1]  # BOS
    i = 0
    while i < len(ipa):
        found = False
        for length in range(4, 0, -1):
            candidate = ipa[i:i+length]
            if candidate in phoneme_id_map:
                ids.extend(phoneme_id_map[candidate])
                ids.append(0)
                i += length
                found = True
                break
        if not found:
            i += 1
    ids.append(2)  # EOS
    return ids


def cmd_dump(args):
    """Dump all intermediate stages from ONNX reference."""
    with open(args.onnx + '.json') as f:
        config = json.load(f)
    pmap = config.get('phoneme_id_map', {})

    ids = encode_phonemes(args.ipa, pmap)
    print(f"IPA: {args.ipa}")
    print(f"Phoneme IDs ({len(ids)}): {ids}")

    # Load and patch ONNX model to expose intermediates
    model = onnx.load(args.onnx)
    for label, node_name in INTERMEDIATES.items():
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(node_name, onnx.TensorProto.FLOAT, None))

    tmp = args.out + '.tmp.onnx'
    onnx.save(model, tmp)
    sess = ort.InferenceSession(tmp)

    input_ids = np.array([ids], dtype=np.int64)
    input_lengths = np.array([len(ids)], dtype=np.int64)
    scales = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # deterministic

    outputs = sess.run(None, {
        'input': input_ids,
        'input_lengths': input_lengths,
        'scales': scales,
    })
    os.unlink(tmp)

    # First output is audio, rest are intermediates
    output_names = [o.name for o in sess.get_outputs()]
    result = {'phoneme_ids': np.array(ids, dtype=np.int64)}
    result['audio'] = outputs[0].squeeze()

    labels = list(INTERMEDIATES.keys())
    for i, label in enumerate(labels):
        arr = outputs[1 + i].squeeze()
        result[label] = arr
        print(f"  {label:30s} shape={str(arr.shape):20s} max_abs={np.max(np.abs(arr)):10.4f}  "
              f"rms={np.sqrt(np.mean(arr**2)):10.4f}")

    np.savez(args.out, **result)
    print(f"\nSaved {len(result)} tensors to {args.out}")


def cos_sim(a, b):
    """Cosine similarity between flattened arrays."""
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0 if (na > 1e-10 or nb > 1e-10) else 1.0
    return float(np.dot(a, b) / (na * nb))


def cmd_compare(args):
    """Compare Python reference dump vs C++ dump."""
    ref = np.load(args.ref)
    cpp_dir = Path(args.cpp)

    print(f"{'Stage':35s} {'Shape':22s} {'cos':>8s} {'max_abs_err':>12s} {'ref_max':>10s} {'cpp_max':>10s}")
    print("-" * 100)

    first_fail = None
    for label in ['phoneme_ids'] + list(INTERMEDIATES.keys()) + ['audio']:
        if label not in ref:
            continue
        ref_arr = ref[label]

        cpp_file = cpp_dir / f"{label}.bin"
        if not cpp_file.exists():
            print(f"  {label:35s} MISSING from C++ dump")
            continue

        cpp_arr = np.fromfile(str(cpp_file), dtype=np.float32)
        if label == 'phoneme_ids':
            cpp_arr = np.fromfile(str(cpp_file), dtype=np.int64)

        # C++ uses (T, C) layout; ONNX uses (C, T). Try transposing.
        if ref_arr.ndim == 2 and cpp_arr.size == ref_arr.size:
            # Reshape C++ flat as (T, C) then transpose to (C, T)
            T_dim, C_dim = ref_arr.shape[1], ref_arr.shape[0]
            cpp_arr = cpp_arr.reshape(T_dim, C_dim).T
        elif ref_arr.ndim == 1 and cpp_arr.size == ref_arr.size:
            pass  # 1D arrays match directly
        elif cpp_arr.size != ref_arr.size:
            # Genuine size mismatch (different durations)
            ref_flat = ref_arr.flatten()
            cpp_flat = cpp_arr.flatten()
            min_len = min(len(ref_flat), len(cpp_flat))
            # Try to infer dimensions and transpose
            # For 2D ref (C, T_ref) and cpp flat of C*T_cpp: reshape as (T_cpp, C).T
            if ref_arr.ndim == 2:
                C_dim = ref_arr.shape[0]
                T_cpp = cpp_arr.size // C_dim
                if T_cpp * C_dim == cpp_arr.size:
                    cpp_2d = cpp_arr.reshape(T_cpp, C_dim).T  # (C, T_cpp)
                    ref_2d = ref_arr
                    min_T = min(ref_2d.shape[1], cpp_2d.shape[1])
                    ref_sub = ref_2d[:, :min_T].flatten()
                    cpp_sub = cpp_2d[:, :min_T].flatten()
                    c = cos_sim(ref_sub, cpp_sub)
                    max_err = np.max(np.abs(ref_sub - cpp_sub))
                    print(f"  {label:35s} {'T_MISMATCH':22s} {c:8.6f} {max_err:12.6f} "
                          f"ref=({C_dim},{ref_2d.shape[1]}) cpp=({C_dim},{T_cpp})")
                    if first_fail is None and c < 0.999:
                        first_fail = label
                    continue
            c = cos_sim(ref_flat[:min_len], cpp_flat[:min_len])
            max_err = np.max(np.abs(ref_flat[:min_len] - cpp_flat[:min_len]))
            print(f"  {label:35s} {'SIZE_MISMATCH':22s} {c:8.6f} {max_err:12.6f} "
                  f"ref={ref_arr.shape}={ref_arr.size} cpp={cpp_arr.size}")
            if first_fail is None and c < 0.999:
                first_fail = label
            continue

        cpp_arr = cpp_arr.reshape(ref_arr.shape)
        c = cos_sim(ref_arr, cpp_arr)
        max_err = float(np.max(np.abs(ref_arr.astype(np.float64) - cpp_arr.astype(np.float64))))
        ref_max = float(np.max(np.abs(ref_arr)))
        cpp_max = float(np.max(np.abs(cpp_arr)))

        flag = " <<<" if c < 0.999 else ""
        print(f"  {label:35s} {str(ref_arr.shape):22s} {c:8.6f} {max_err:12.6f} "
              f"{ref_max:10.4f} {cpp_max:10.4f}{flag}")
        if first_fail is None and c < 0.999:
            first_fail = label

    if first_fail:
        print(f"\n*** FIRST DIVERGENCE at: {first_fail} ***")
    else:
        print(f"\nAll stages match (cos >= 0.999).")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    p_dump = sub.add_parser('dump', help='Dump ONNX reference intermediates')
    p_dump.add_argument('--onnx', required=True)
    p_dump.add_argument('--ipa', default='həlˈoʊ wˈɜːld')
    p_dump.add_argument('--out', required=True)

    p_cmp = sub.add_parser('compare', help='Compare ref vs C++ dumps')
    p_cmp.add_argument('--ref', required=True)
    p_cmp.add_argument('--cpp', required=True)

    args = parser.parse_args()
    if args.cmd == 'dump':
        cmd_dump(args)
    elif args.cmd == 'compare':
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
