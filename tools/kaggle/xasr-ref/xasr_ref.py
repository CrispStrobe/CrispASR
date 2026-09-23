#!/usr/bin/env python3
"""CrispASR #436 (X-ASR) — reference fixtures + F16 GGUF, with two control arms.

1. The sherpa-onnx exports: verify every sha256 against HF's LFS oid.
2. (pretrained.pt is a different checkpoint — xasr-onnx-map — so the ONNX
   export is the source for both the GGUF and the reference.)
3. convert-xasr-to-gguf.py --models-dir -> F16; upload.
4. tools/dump_reference.py --backend xasr for jfk + zh at chunk 480 (and 160,
   the export with the shorter 96-frame left context); upload each ref.gguf.
5. Control A: the ONNX encoder (onnxruntime) on the same fbank chunks, states
   carried, vs the torch encoder_out -> proves the reference is what sherpa runs.
6. Control B: sherpa-onnx OnlineRecognizer text with the same tail padding.
"""
import hashlib, json, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; M = Path("/tmp/xasr"); ICE = Path("/tmp/icefall_zipformer")
res = {"errors": [], "refs": {}, "control_onnx": {}, "control_sherpa": {}}
def save(): (OUT / "xasr_ref.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
R = "GilgameshWind/X-ASR-zh-en"
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gguf", "kaldi-native-fbank", "onnx",
                           "onnxruntime", "sherpa-onnx", "soundfile"])
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", "feat/436-xasr", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    ice_sha = json.loads(subprocess.check_output(["curl", "-sf", "https://api.github.com/repos/k2-fsa/icefall/commits/master"]))["sha"]
    ICE.mkdir(parents=True, exist_ok=True)
    for f in ("zipformer.py", "scaling.py", "subsampling.py", "decoder.py", "joiner.py"):
        subprocess.check_call(["curl", "-sfL", "-o", str(ICE / f),
                               f"https://raw.githubusercontent.com/k2-fsa/icefall/{ice_sha}/egs/librispeech/ASR/zipformer/{f}"])
    res["icefall_sha"] = ice_sha; save()
    from huggingface_hub import snapshot_download
    snapshot_download(R, allow_patterns=["deployment/models/*"], local_dir=str(M))
    md = M / "deployment/models"
    for p in md.glob("chunk-*/*.onnx"):
        info = api.get_paths_info(R, [str(p.relative_to(M))])[0]
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
        if info.lfs is not None and h.hexdigest() != info.lfs.sha256:
            raise RuntimeError(f"sha256 mismatch: {p}")
    res["onnx_sha_ok"] = True; save()
    os.environ["XASR_ICEFALL_DIR"] = str(ICE)
    f16 = M / "x-asr-zh-en-f16.gguf"
    cv = subprocess.run([sys.executable, str(REPO / "models/convert-xasr-to-gguf.py"), "--models-dir", str(md),
                         "--output", str(f16)], capture_output=True, text=True)
    res["convert"] = (cv.stdout + cv.stderr).strip()[-3000:]; save()
    if cv.returncode != 0:
        raise RuntimeError("convert failed: " + res["convert"])
    api.create_repo("cstr/x-asr-zh-en-GGUF", repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(f16), path_in_repo=f16.name, repo_id="cstr/x-asr-zh-en-GGUF", repo_type="model")
    res["f16_bytes"] = f16.stat().st_size; save()
    import numpy as np
    wav = {"jfk": str(REPO / "samples/jfk.wav"), "zh": str(REPO / "samples/paraformer_zh.wav")}
    import soundfile as sf
    import sherpa_onnx
    sys.path.insert(0, str(REPO / "tools"))
    from reference_backends import xasr as X
    for ch in (480, 160):
        os.environ["XASR_CHUNK_MS"] = str(ch)
        d = md / f"chunk-{ch}ms-model"
        for c, w in wav.items():
            tag = c if ch == 480 else f"{c}-c{ch}"
            ref = M / f"{tag}.gguf"
            subprocess.check_call([sys.executable, str(REPO / "tools/dump_reference.py"), "--backend", "xasr", "--model-dir", str(md),
                                   "--audio", w, "--output", str(ref)])
            api.upload_file(path_or_fileobj=str(ref), path_in_repo=f"xasr/{tag}/ref.gguf",
                            repo_id="cstr/crispasr-regression-fixtures", repo_type="dataset")
            import gguf
            rd = gguf.GGUFReader(str(ref))
            meta = {f.name.split(".")[-1]: bytes(f.parts[f.data[0]]).decode() for f in rd.fields.values()
                    if f.name in ("crispasr.ref.text", "crispasr.ref.tail_pad_ms")}
            ten = {t.name: np.array(t.data).reshape([int(x) for x in reversed(t.shape)]) for t in rd.tensors}
            res["refs"][tag] = {"text": meta.get("text"), "tail_pad_ms": meta.get("tail_pad_ms"),
                                "shapes": {k: list(v.shape) for k, v in ten.items()}}
            save()
            # Control A: onnxruntime encoder on the same fbank chunks
            import onnxruntime as ort
            sess = ort.InferenceSession(str(d / f"encoder-{ch}ms.onnx"), providers=["CPUExecutionProvider"])
            feats = ten["fbank"]; chunk = ch // 20; T = 2 * chunk + 13
            st = {}
            for i in sess.get_inputs()[1:]:
                shp = [1 if isinstance(s, str) else s for s in i.shape]
                st[i.name] = np.zeros(shp, dtype=np.int64 if "int64" in i.type else np.float32)
            outs = []; p = 0
            while p + T < feats.shape[0]:
                r = sess.run(None, {"x": feats[p:p + T][None], **st})
                names = [o.name for o in sess.get_outputs()]
                outs.append(r[0][0])
                for nm, v in zip(names[1:], r[1:]):
                    st[nm[len("new_"):]] = v
                p += 2 * chunk
            eo = np.concatenate(outs); te = ten["encoder_out"]
            cos = (eo * te).sum(1) / (np.linalg.norm(eo, axis=1) * np.linalg.norm(te, axis=1) + 1e-12)
            res["control_onnx"][tag] = {"shape": list(eo.shape), "ref_shape": list(te.shape), "cos_min": float(cos.min()),
                                        "max_abs": float(np.abs(eo - te).max()), "ref_rms": float(np.sqrt((te ** 2).mean()))}
            save()
            # Control B: sherpa-onnx with identical tail padding
            rec = sherpa_onnx.OnlineRecognizer.from_transducer(tokens=str(d / "tokens.txt"), encoder=str(d / f"encoder-{ch}ms.onnx"),
                  decoder=str(d / f"decoder-{ch}ms.onnx"), joiner=str(d / f"joiner-{ch}ms.onnx"), num_threads=4,
                  feature_dim=80, decoding_method="greedy_search")
            a, sr = sf.read(w, dtype="float32")
            s = rec.create_stream(); s.accept_waveform(sr, a)
            s.accept_waveform(sr, np.zeros(16 * int(meta["tail_pad_ms"]), dtype=np.float32)); s.input_finished()
            while rec.is_ready(s): rec.decode_stream(s)
            sher = rec.get_result(s)
            res["control_sherpa"][tag] = {"sherpa": sher, "ours": meta.get("text"), "same": sher == meta.get("text")}
            save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
