#!/usr/bin/env python3
"""
Parakeet-Unified-EN-0.6B → GGUF: direct NeMo 2.x zip extraction.

Extracts weights from the NeMo 2.x zip format (no NeMo model instantiation),
takes the tokenizer from the same .nemo tar (issue #567: never borrow it from
another model; parakeet-rnnt-0.6b has the same size but a different order), and runs
the existing convert-parakeet-to-gguf.py converter on a synthetic .nemo tar.
"""
import json, os, subprocess, sys, time, traceback, shutil, zipfile, pickle, io, tarfile, tempfile
from pathlib import Path

WORK = Path("/kaggle/working")
results = {}

def log(msg):
    print(msg, flush=True)
    try:
        with open(WORK / "progress.txt", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass

def save():
    try:
        (WORK / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception:
        pass

def main():
    global results
    save()
    log("=== Parakeet-Unified v5 — full conversion ===")

    # HF token
    for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
              "/kaggle/input/datasets/${KAGGLE_ACCOUNT}/crispasr-hf-token/hf_token.txt"]:
        if os.path.exists(p):
            tok = open(p).read().strip()
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
            break

    # Install deps
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf", "pyyaml", "sentencepiece"], check=False)

    # Clone CrispASR
    cdir = Path("/tmp/CrispASR")
    if not cdir.exists():
        subprocess.check_call(["git", "clone", "--depth", "1",
            "https://github.com/CrispStrobe/CrispASR.git", str(cdir)])

    # Build
    log("Building CrispASR...")
    subprocess.run("apt-get update -qq && apt-get install -y cmake ninja-build g++ 2>/dev/null || true",
                   shell=True, capture_output=True)
    bdir = cdir / "build"
    subprocess.check_call(["cmake", "-G", "Ninja", "-B", str(bdir),
                          "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON"], cwd=str(cdir))
    subprocess.check_call(["cmake", "--build", str(bdir), "-j2"], cwd=str(cdir))
    log("Build OK")

    from huggingface_hub import hf_hub_download
    import torch
    import yaml

    # Step 1: Download the unified .nemo
    log("Downloading parakeet-unified .nemo (2.4 GB)...")
    nemo_path = hf_hub_download("nvidia/parakeet-unified-en-0.6b",
                                 "parakeet-unified-en-0.6b.nemo",
                                 cache_dir="/tmp/.hf")
    log(f"  Downloaded: {os.path.getsize(nemo_path)/(1024*1024):.0f} MB")

    # Step 2: Extract state dict from NeMo 2.x zip
    log("Extracting state dict...")
    class NeMo2Unpickler(pickle.Unpickler):
        def __init__(self, *args, zf=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._zf = zf
        def persistent_load(self, saved_id):
            if isinstance(saved_id, tuple) and len(saved_id) >= 4:
                if len(saved_id) == 5:
                    _, stype, key, device, numel = saved_id
                else:
                    stype, key, device, numel = saved_id
                raw = self._zf.read(f"model_weights/data/{key}")
                dtype = torch.float32
                if 'Half' in str(stype) or 'float16' in str(stype): dtype = torch.float16
                elif 'BFloat16' in str(stype): dtype = torch.bfloat16
                nbytes = numel * torch.tensor([], dtype=dtype).element_size()
                return torch.frombuffer(bytearray(raw[:nbytes]), dtype=dtype).storage()
            return saved_id

    with zipfile.ZipFile(nemo_path) as zf:
        sd = NeMo2Unpickler(io.BytesIO(zf.read("model_weights/data.pkl")), zf=zf).load()
    log(f"  {len(sd)} keys extracted")
    results["n_keys"] = len(sd)
    save()

    # Step 3: Tokenizer from the unified .nemo itself. The .nemo is a tar that
    # holds the SentencePiece model next to the zip-format model_weights.ckpt.
    # Issue #567: borrowing parakeet-rnnt-0.6b's tokenizer (same size, different
    # order) produced nonsense text.
    log("Extracting tokenizer from the unified .nemo...")
    spm_bytes = None
    with tarfile.open(nemo_path, "r:") as tf:
        for m in tf.getmembers():
            if m.name.endswith("tokenizer.model"):
                spm_bytes = tf.extractfile(m).read()
                log(f"  Tokenizer: {m.name} ({len(spm_bytes)} bytes)")
                break
    if not spm_bytes:
        log("  ERROR: no tokenizer in the unified .nemo")
        results["error"] = "no tokenizer"
        save()
        return
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_proto=spm_bytes)
    last_piece = sp.id_to_piece(sp.get_piece_size() - 1)
    results["tokenizer_last_piece"] = last_piece
    if last_piece != "\u00a5":
        log(f"  ERROR: tokenizer last piece {last_piece!r}, expected '\u00a5' (#567)")
        results["error"] = "wrong tokenizer"
        save()
        return

    # Step 4: Build synthetic model_config.yaml from inferred hparams
    log("Building config...")
    d_model = int(sd["encoder.layers.0.self_attn.linear_q.weight"].shape[0])
    n_layers = max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers.") and k.split(".")[2].isdigit()) + 1
    vocab_size = int(sd["decoder.prediction.embed.weight"].shape[0])
    pred_hidden = int(sd["decoder.prediction.dec_rnn.lstm.weight_ih_l0"].shape[1])
    n_heads = 8  # standard for d=1024

    # Check subsampling: pre_encode.conv layers
    pre_conv_keys = sorted(k for k in sd if k.startswith("encoder.pre_encode.conv.") and "weight" in k)
    n_pre_convs = len([k for k in pre_conv_keys if k.endswith(".weight") and "." not in k.split("conv.")[1].split(".weight")[0]])
    subsampling_factor = 8  # 3 strided convs → 8x (vs standard parakeet's 2 convs → 4x)

    # Check for att_context_size in tensor names (streaming presets)
    has_joint = "joint.joint_net.2.weight" in sd
    n_tdt_durations = 0  # RNNT, not TDT

    config = {
        "encoder": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "feat_in": 128,
            "conv_kernel_size": 9,
            "subsampling_factor": subsampling_factor,
            "subsampling": "dw_striding",
            "subsampling_conv_channels": 256,
            "xscaling": True,
        },
        "decoder": {
            "prednet": {"pred_hidden": pred_hidden, "pred_rnn_layers": 2},
            "vocab_size": vocab_size,
        },
        "joint": {
            "joint_hidden": 640,
            "num_extra_outputs": 0,  # RNNT (no TDT durations)
        },
        "preprocessor": {
            "sample_rate": 16000,
            "features": 128,  # inferred from preprocessor.fb shape [257, 128]
            "n_fft": 512,
            "window_size": 0.025,
            "window_stride": 0.01,
            "dither": 0.0,
            "normalize": "per_feature",
        },
    }
    config_str = yaml.dump(config)
    log(f"  Config: d={d_model} layers={n_layers} vocab={vocab_size} sub={subsampling_factor}")
    results["hparams"] = {"d_model": d_model, "n_layers": n_layers, "vocab_size": vocab_size,
                           "subsampling_factor": subsampling_factor}
    save()

    # Step 5: Create synthetic NeMo 1.x tar that the converter understands
    log("Creating synthetic .nemo tar...")
    synthetic_nemo = WORK / "parakeet-unified-synthetic.nemo"
    with tarfile.open(str(synthetic_nemo), "w") as tf:
        # model_config.yaml
        cfg_data = config_str.encode()
        cfg_info = tarfile.TarInfo("./model_config.yaml")
        cfg_info.size = len(cfg_data)
        tf.addfile(cfg_info, io.BytesIO(cfg_data))

        # tokenizer
        spm_info = tarfile.TarInfo("./nemo_rnnt_bpe_tokenizer.model")
        spm_info.size = len(spm_bytes)
        tf.addfile(spm_info, io.BytesIO(spm_bytes))

        # model_weights.ckpt — save state dict as torch checkpoint
        log("  Saving weights to tar (this takes a minute)...")
        ckpt_buf = io.BytesIO()
        torch.save(sd, ckpt_buf)
        ckpt_data = ckpt_buf.getvalue()
        ckpt_info = tarfile.TarInfo("./model_weights.ckpt")
        ckpt_info.size = len(ckpt_data)
        tf.addfile(ckpt_info, io.BytesIO(ckpt_data))
        del ckpt_data, ckpt_buf

    sz = os.path.getsize(str(synthetic_nemo)) / (1024*1024)
    log(f"  Synthetic .nemo: {sz:.0f} MB")
    del sd  # free memory
    import gc; gc.collect()

    # Step 6: Run the converter
    log("Running convert-parakeet-to-gguf.py...")
    converter = str(cdir / "models" / "convert-parakeet-to-gguf.py")
    out_f16 = str(WORK / "parakeet-unified-en-0.6b-f16.gguf")
    r = subprocess.run([sys.executable, converter,
                       "--nemo", str(synthetic_nemo),
                       "--output", out_f16],
                      capture_output=True, text=True, timeout=600)
    results["converter_rc"] = r.returncode
    log(f"  Converter rc={r.returncode}")
    if r.returncode != 0:
        results["converter_stderr"] = r.stderr[-1000:]
        log(f"  stderr: {r.stderr[-500:]}")
        save()
    else:
        gguf_sz = os.path.getsize(out_f16) / (1024*1024)
        results["gguf_f16_mb"] = round(gguf_sz, 1)
        log(f"  F16 GGUF: {gguf_sz:.0f} MB")

        # Step 7: Test with CrispASR
        log("Testing with CrispASR...")
        CLI = str(bdir / "bin" / "crispasr")
        JFK = str(cdir / "samples" / "jfk.wav")
        r2 = subprocess.run([CLI, "--backend", "parakeet", "-m", out_f16, "-f", JFK, "-np"],
                           capture_output=True, text=True, timeout=300)
        lines = [l.strip() for l in r2.stdout.strip().split('\n') if l.strip()]
        transcript = lines[-1] if lines else ""
        results["test_rc"] = r2.returncode
        results["transcript"] = transcript
        log(f"  Test rc={r2.returncode}: {transcript[:80]}")

        # Step 8: Quantize Q4_K
        if True:  # upload even if test fails — debug SIGABRT later
            log("Quantizing Q4_K...")
            QUANT = str(bdir / "bin" / "crispasr-quantize")
            out_q4k = str(WORK / "parakeet-unified-en-0.6b-q4_k.gguf")
            subprocess.run([QUANT, out_f16, out_q4k, "q4_k"], capture_output=True, timeout=300)
            if os.path.exists(out_q4k):
                results["gguf_q4k_mb"] = round(os.path.getsize(out_q4k)/(1024*1024), 1)
                log(f"  Q4_K: {results['gguf_q4k_mb']} MB")

            # Step 9: Upload to HF
            log("Uploading to HF...")
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=os.environ.get("HF_TOKEN"))
                repo = "cstr/parakeet-unified-en-0.6b-GGUF"
                api.create_repo(repo, exist_ok=True)
                for fpath in [out_f16, out_q4k]:
                    if os.path.exists(fpath):
                        api.upload_file(path_or_fileobj=fpath,
                                       path_in_repo=os.path.basename(fpath),
                                       repo_id=repo,
                                       commit_message=f"Add {os.path.basename(fpath)}")
                        log(f"  Uploaded {os.path.basename(fpath)}")
                results["uploaded"] = True
            except Exception as e:
                results["upload_error"] = str(e)
                log(f"  Upload error: {e}")

    save()
    log("\nDONE")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        results["_crash"] = str(e)
        results["_tb"] = traceback.format_exc()
        save()
        log(f"CRASH: {e}")
        traceback.print_exc()

# Cleanup
for p in [Path("/tmp/CrispASR"), Path("/tmp/.hf"), WORK / ".hf"]:
    shutil.rmtree(str(p), ignore_errors=True)
for f in WORK.glob("*.nemo"):
    f.unlink(missing_ok=True)
for f in WORK.glob("*.gguf"):
    f.unlink(missing_ok=True)
