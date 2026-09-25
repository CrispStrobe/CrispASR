#!/usr/bin/env python3
"""Convert nvidia/Nemotron-3-Diarization (transformers format) to GGUF (#466).

The output uses the SAME layout as the `sortformer` GGUF NVIDIA ships in the
model repo (`Nemotron-3-Diarization.q8_0.gguf`, written by NeMo-Speech.cpp's
conversion/diarization.py): general.architecture = "sortformer", tensor names
encoder.* / head.* / subpixel_upsample.* / learnable_sil_emb / preprocessor.fb
and the sortformer.* hyper-parameter keys. CrispASR's nemotron3-diar runtime
therefore loads either file; this script exists for F32/F16 (and, via
crispasr-quantize, Q4_K) artifacts and so the weights come from the reference
the diff harness uses (transformers Nemotron3DiarizationForAudioFrameClassification).

Verified mapping (HF -> GGUF, NVIDIA q8_0 dequantised vs HF: cos >= 0.99997):
  model.audio_tower.embedder.projection        -> encoder.pre_encode.proj
  model.audio_tower.input_layer_norm / layer_norm -> encoder.embed_norm / final_norm
  layers.i.self_attn.{q,k,v}_proj (concat q;k;v) -> encoder.layers.i.attn.w_qkv
  layers.i.self_attn.o_proj                    -> encoder.layers.i.attn.out_proj
  layers.i.layer_norm{1,2}                     -> encoder.layers.i.norm{1,2}
  layers.i.mlp.fc1 / fc2                       -> encoder.layers.i.ffn.net.0 / net.3
  model.proj                                   -> encoder_proj
  model.upsampler.conv                         -> subpixel_upsample
  classifier.dense / out_proj                  -> head.first_hidden_to_hidden / single_hidden_to_spks
  silence_embeds                               -> learnable_sil_emb

Offline chunking (config.json chunk_length / chunk_right_context / fifo_length /
speaker_cache_update_period) goes into nemotron3diar.offline.* - NVIDIA's own
sortformer.streaming.* keys describe a different default schedule, and the
runtime takes transformers' offline schedule as the reference.

Usage:
  python models/convert-nemotron3-diar-to-gguf.py --input <hf dir> --output n3d-f16.gguf [--outtype f16|f32]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    raise SystemExit("pip install gguf")
from safetensors.numpy import load_file


def slaney_mel(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm='slaney') -> (n_mels, n_fft//2+1)."""
    try:
        import librosa

        return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2, norm="slaney").astype(
            np.float32
        )
    except ImportError:
        pass

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        mel = f / (200.0 / 3)
        log_t = f >= 1000.0
        return np.where(log_t, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mel)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f = m * (200.0 / 3)
        return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), f)

    fft_f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_f[None, :]
    w = np.zeros((n_mels, len(fft_f)))
    for i in range(n_mels):
        w[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    w *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--outtype", default="f16", choices=["f16", "f32"])
    a = ap.parse_args()

    cfg = json.loads((a.input / "config.json").read_text())
    pcfg = json.loads((a.input / "processor_config.json").read_text()) if (a.input / "processor_config.json").exists() else {}
    fe = pcfg.get("feature_extractor", {})
    ac, hc, sc = cfg["audio_config"], cfg["head_config"], cfg["streaming_config"]
    sd = load_file(str(a.input / "model.safetensors"))

    w = gguf.GGUFWriter(str(a.output), "sortformer")
    w.add_name("Nemotron-3-Diarization")
    w.add_string("sortformer.version", "v3")
    E = "sortformer.encoder."
    d = ac["hidden_size"]
    w.add_uint32(E + "d_model", d)
    w.add_uint32(E + "n_layers", ac["num_hidden_layers"])
    w.add_uint32(E + "n_heads", ac["num_attention_heads"])
    w.add_uint32(E + "d_ff", ac["intermediate_size"])
    w.add_uint32(E + "subsampling_factor", ac["subsampling_factor"])
    w.add_uint32(E + "feat_in", ac["num_mel_bins"])
    w.add_uint32(E + "pos_emb_max_len", ac.get("max_position_embeddings", 5000))
    w.add_string(E + "type", "transformer_rope")
    w.add_string(E + "subsampling_type", "feature_stacking")
    w.add_float32(E + "rope_base", float(ac["rope_parameters"]["rope_theta"]))
    w.add_float32(E + "rotary_fraction", float(ac["rope_parameters"].get("partial_rotary_factor", 1.0)))
    w.add_uint32("sortformer.transformer.hidden_size", hc["hidden_size"])
    w.add_uint32("sortformer.num_speakers", hc["num_speakers"])
    w.add_uint32("sortformer.upsample_factor", hc["subsampling_factor"])
    P = "sortformer.preprocessor."
    sr = fe.get("sampling_rate", 16000)
    n_fft = fe.get("n_fft", 512)
    w.add_uint32(P + "sample_rate", sr)
    w.add_uint32(P + "n_fft", n_fft)
    w.add_float32(P + "window_size", fe.get("win_length", 400) / sr)
    w.add_float32(P + "window_stride", fe.get("hop_length", 160) / sr)
    w.add_uint32(P + "features", fe.get("feature_size", ac["num_mel_bins"]))
    w.add_string(P + "normalize", "NA")
    w.add_float32(P + "preemph", float(fe.get("preemphasis", 0.97)))
    w.add_float32(P + "log_zero_guard", 2.0**-24)
    S = "sortformer.scoring."
    w.add_uint32(S + "spkcache_sil_frames_per_spk", sc["speaker_cache_silence_frames_per_speaker"])
    w.add_float32(S + "pred_score_threshold", sc["prediction_score_threshold"])
    w.add_float32(S + "scores_boost_latest", sc["latest_frames_score_boost"])
    w.add_float32(S + "strong_boost_rate", sc["strong_boost_rate"])
    w.add_float32(S + "weak_boost_rate", sc["weak_boost_rate"])
    w.add_float32(S + "min_pos_scores_rate", sc["min_positive_scores_rate"])
    w.add_uint32("sortformer.streaming.spkcache_len", sc["speaker_cache_length"])
    w.add_uint32("sortformer.streaming.fifo_len", sc["fifo_length"])
    w.add_uint32("sortformer.streaming.spkcache_update_period", sc["speaker_cache_update_period"])
    O = "nemotron3diar.offline."
    w.add_uint32(O + "chunk_len", cfg["chunk_length"])
    w.add_uint32(O + "chunk_right_context", cfg["chunk_right_context"])
    w.add_uint32(O + "fifo_len", cfg["fifo_length"])
    w.add_uint32(O + "spkcache_update_period", cfg["speaker_cache_update_period"])
    modes = pcfg.get("streaming_modes", {"low_latency": [9, 4], "very_low_latency": [6, 2], "ultra_low_latency": [3, 1]})
    for name, (cl, rc) in modes.items():
        w.add_uint32(f"nemotron3diar.streaming.{name}.chunk_len", cl)
        w.add_uint32(f"nemotron3diar.streaming.{name}.chunk_right_context", rc)

    ftype = np.float16 if a.outtype == "f16" else np.float32

    def put(name: str, arr: np.ndarray, force_f32: bool = False) -> None:
        arr = np.ascontiguousarray(arr)
        big = arr.ndim >= 2 and arr.shape[-1] % 32 == 0 and not force_f32
        w.add_tensor(name, arr.astype(ftype if big else np.float32))

    put("preprocessor.fb", slaney_mel(sr, n_fft, fe.get("feature_size", 128)), force_f32=True)
    A = "model.audio_tower."
    # (512, 1024), kept F32: streaming projects only chunk + look-ahead (13 rows),
    # and CUDA's small-batch F16 matmul rounds the activations to F16 -> embeds
    # off by 5e-2 and a few flipped decisions vs transformers (T4, #466)
    put("encoder.pre_encode.proj.weight", sd[A + "embedder.projection.weight"], force_f32=True)
    put("encoder.embed_norm.weight", sd[A + "input_layer_norm.weight"])
    put("encoder.embed_norm.bias", sd[A + "input_layer_norm.bias"])
    put("encoder.final_norm.weight", sd[A + "layer_norm.weight"])
    put("encoder.final_norm.bias", sd[A + "layer_norm.bias"])
    for i in range(ac["num_hidden_layers"]):
        L, G = f"{A}layers.{i}.", f"encoder.layers.{i}."
        put(G + "norm1.weight", sd[L + "layer_norm1.weight"])
        put(G + "norm1.bias", sd[L + "layer_norm1.bias"])
        qkv = np.concatenate([sd[L + "self_attn.q_proj.weight"], sd[L + "self_attn.k_proj.weight"],
                              sd[L + "self_attn.v_proj.weight"]], axis=0)
        put(G + "attn.w_qkv.weight", qkv)
        put(G + "attn.out_proj.weight", sd[L + "self_attn.o_proj.weight"])
        put(G + "attn.out_proj.bias", sd[L + "self_attn.o_proj.bias"])
        put(G + "norm2.weight", sd[L + "layer_norm2.weight"])
        put(G + "norm2.bias", sd[L + "layer_norm2.bias"])
        put(G + "ffn.net.0.weight", sd[L + "mlp.fc1.weight"])
        put(G + "ffn.net.0.bias", sd[L + "mlp.fc1.bias"])
        put(G + "ffn.net.3.weight", sd[L + "mlp.fc2.weight"])
        put(G + "ffn.net.3.bias", sd[L + "mlp.fc2.bias"])
    put("encoder_proj.weight", sd["model.proj.weight"])
    put("encoder_proj.bias", sd["model.proj.bias"])
    # (out=1536, in=192, k=3): kept F32 - it is small and ggml's conv path is simplest in F32
    put("subpixel_upsample.weight", sd["model.upsampler.conv.weight"], force_f32=True)
    put("subpixel_upsample.bias", sd["model.upsampler.conv.bias"])
    put("head.first_hidden_to_hidden.weight", sd["classifier.dense.weight"], force_f32=True)
    put("head.first_hidden_to_hidden.bias", sd["classifier.dense.bias"])
    put("head.single_hidden_to_spks.weight", sd["classifier.out_proj.weight"], force_f32=True)
    put("head.single_hidden_to_spks.bias", sd["classifier.out_proj.bias"])
    put("learnable_sil_emb", sd["silence_embeds"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
