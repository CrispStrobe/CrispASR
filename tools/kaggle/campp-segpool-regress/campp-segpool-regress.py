#!/usr/bin/env python3
"""CAM++ seg_pooling regression gate for the four backends that were never
re-validated after the divisor fix (chatterbox, confucius4-tts, cosyvoice3-tts,
dots-tts), with fireredtts3 carried along as a POSITIVE CONTROL.

Background. `campplus_segpool::avg` divided the PARTIAL TAIL window of
`F.avg_pool1d(k=100, stride=100, ceil_mode=True)` by the kernel size instead of
by the frames actually in it. Fixed on main. fireredtts3 was measured against
upstream and went 0.268 -> 0.999452. The other four share the same
`chatterbox_campplus::embed_speaker` and were all accepted END-TO-END, and no
acceptance test any of them has diffs that stage -- so their speaker embeddings
changed and nothing could have noticed.

Two questions, per backend:
  1. does it still work end-to-end?  (voice clone -> wav -> whisper roundtrip)
  2. did the fix move the speaker embedding TOWARD upstream, or away?

Method. `CRISPASR_CAMPP_DUMP_EMB=<path>` (added on this branch) appends the
embedding AND the mean-subtracted fbank that produced it, from inside
embed_speaker() -- the one function all five funnel through. So each backend is
driven through its REAL CLI voice-clone path, twice: once on current main and
once with `CRISPASR_CAMPP_LEGACY_SEGPOOL=1` restoring the known-wrong divisor.
The upstream reference (torch CAMPPlus, each backend's own checkpoint) is then
run ON THE DUMPED FBANK, so the CAMPPlus forward is the only thing that can
differ -- a reference running its own front end would fold any Kaldi-fbank
difference into the same number and the two would not be separable afterwards.

Controls, without which the numbers mean nothing:
  C1  the fbank must be byte-identical across the two arms. The gate lives
      downstream of the front end; if the fbank moves, something else did too.
  C2  the two arms must DIFFER whenever T_cam % 100 != 0. If they agree, the
      gate never reached that backend and every reading through it is vacuous --
      reported as GATE_NOT_REACHED, never as a pass.
  C3  T_cam % 100 == 0 (no partial tail) must give bit-identical arms. Asserted
      hermetically in tests/test-campplus-segpool.cpp rather than bought with a
      run here, because it is arithmetic.
  C4  fireredtts3 must reproduce its known answer (~0.999 fixed, ~0.27 legacy).
      If it does not, the instrument is wrong and NO backend's number stands.
  C5  if BOTH arms score near zero against a reference, that reference is
      broken, not the port. Reported as REFERENCE_UNUSABLE, never as "both wrong".

Magnitude is reported next to every cosine: the original bug was a ~1.67x
magnitude error on the tail, and cosine is scale-blind.

Push (chr1s4):
  export KAGGLE_API_TOKEN=KGAT_8bf612aeb5eb7e3eb52c0ee861871ee5
  python -m kaggle kernels push -p tools/kaggle/campp-segpool-regress
"""

import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_VERSION = "v2-companion-paths"
WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO = TEMP / "CrispASR"
BRANCH = "test/campplus-regress"
TEST_TEXT = "The quick brown fox jumps over the lazy dog."
# jfk.wav's actual words, for fireredtts3's --ref-text (it needs the reference
# transcript for ICL cloning; the others infer or ignore it).
JFK_TEXT = ("And so my fellow Americans ask not what your country can do for you, "
            "ask what you can do for your country.")

print(f"=== campp-segpool-regress {SCRIPT_VERSION} ===", flush=True)

if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
subprocess.check_call(["git", "log", "--oneline", "-1"], cwd=str(REPO))
subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=str(REPO))
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

kh.step("install deps")
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer gguf soundfile safetensors")
tool = kh.install_build_toolchain()
print(f"  toolchain: {tool}")

# Cheap checks before expensive ones: name the EXACT symbols consumed later, so
# a missing package fails now rather than after the build and five CLI runs.
for mod, sym in (("gguf", "GGUFReader"), ("safetensors.torch", "load_file"),
                 ("soundfile", "read"), ("torch", "nn"), ("numpy", "linalg")):
    try:
        m = __import__(mod, fromlist=[sym])
        getattr(m, sym)
    except Exception as e:
        raise SystemExit(f"{mod}.{sym} unavailable: {e!r} -- failing before the build")
print("  all consumed symbols importable")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# ── build (CPU) ─────────────────────────────────────────────────────────────
kh.step("cmake configure")
flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF",
         "-DCRISPASR_BUILD_TESTS=OFF", "-DCRISPASR_BUILD_SERVER=OFF"]
flags += kh.cache_and_link_flags()
gen = ["-G", "Ninja"] if tool.get("ninja") else []
kh.sh_with_progress("cmake -B build " + " ".join(gen + flags), cwd=str(REPO))

kh.step("build crispasr-cli")
with kh.build_heartbeat("build"):
    kh.sh_with_progress(f"cmake --build build --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}",
                        cwd=str(REPO))
BIN = REPO / "build" / "bin" / "crispasr"
assert BIN.exists(), f"missing {BIN} -- the build did not produce the binary"
print(f"  built: {BIN} ({BIN.stat().st_size} bytes)")

# Proof the dump hook is actually IN this binary. Without it every backend would
# report "no record" and that would read like a backend failure rather than a
# stale build.
_strings = subprocess.run(["strings", str(BIN)], capture_output=True, text=True).stdout
assert "CRISPASR_CAMPP_DUMP_EMB" in _strings, \
    "binary has no CRISPASR_CAMPP_DUMP_EMB -- wrong branch or a stale object"
assert "CRISPASR_CAMPP_LEGACY_SEGPOOL" in _strings, \
    "binary has no CRISPASR_CAMPP_LEGACY_SEGPOOL -- the A/B gate is not in this build"
print("  binary carries both env gates")

PROMPT_WAV = REPO / "samples" / "jfk.wav"
assert PROMPT_WAV.exists(), f"missing reference wav {PROMPT_WAV}"

MD = TEMP / "models"
MD.mkdir(parents=True, exist_ok=True)

# ── the record reader ───────────────────────────────────────────────────────
# Mirrors campp_dump_embedding() in src/chatterbox_campplus.cpp.
def read_records(path):
    """[(n_samples, T_fbank, T_cam, emb(np), fbank(np or None)), ...]"""
    out = []
    if not Path(path).exists():
        return out
    blob = Path(path).read_bytes()
    off = 0
    while off + 24 <= len(blob):
        if blob[off:off + 4] != b"CPE2":
            raise SystemExit(f"{path}: bad magic at {off} -- record format drifted from the C++")
        off += 4
        n_samp, t_fb, t_cam, dim, n_mels = struct.unpack_from("<5i", blob, off)
        off += 20
        emb = np.frombuffer(blob, dtype=np.float32, count=dim, offset=off).copy()
        off += 4 * dim
        fb = None
        if n_mels > 0:
            cnt = t_fb * n_mels
            fb = np.frombuffer(blob, dtype=np.float32, count=cnt, offset=off).reshape(t_fb, n_mels).copy()
            off += 4 * cnt
        out.append((n_samp, t_fb, t_cam, emb, fb))
    return out


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ── upstream reference: one 3D-Speaker CAMPPlus, each backend's checkpoint ──
kh.step("fetch reference CAMPPlus implementation")
UP = TEMP / "FireRedTTS3-upstream"
if not UP.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/FireRedTeam/FireRedTTS3.git", str(UP)])
sys.path.insert(0, str(UP))
from fireredtts3.campp.DTDNN import CAMPPlus  # noqa: E402
torch.set_grad_enabled(False)
torch.set_num_threads(2)

_REF_CACHE = {}


def ref_state_dict(kind):
    """Upstream CAMPPlus state dict for a backend, plus its embedding size.

    Every one of these is the ORIGINAL published checkpoint, not something
    reconstructed from our own GGUF -- a reference rebuilt by inverting our own
    converter would agree with the port by construction on exactly the thing
    being measured.  cosyvoice3/confucius4 share one: the cosyvoice3 campplus
    GGUF was verified locally to be funasr/campplus campplus_cn_common.bin with
    BatchNorm folded into the convs (cos = 1.00000000 after re-folding, and the
    unfolded cam.l1.bias matches bit-for-bit).
    """
    if kind in _REF_CACHE:
        return _REF_CACHE[kind]
    if kind == "funasr":  # cosyvoice3-tts + confucius4-tts, 192-d
        p = hf_hub_download("funasr/campplus", "campplus_cn_common.bin", token=hf_token)
        sd, dim = torch.load(p, map_location="cpu", weights_only=True), 192
    elif kind == "voxceleb":  # fireredtts3, 512-d
        p = hf_hub_download("FireRedTeam/FireRedTTS3", "campp/campplus_voxceleb.bin", token=hf_token)
        sd, dim = torch.load(p, map_location="cpu", weights_only=True), 512
    elif kind == "chatterbox":  # inside s3gen_v3, 192-d
        # 937 `speaker_encoder.*` keys, verified by range-reading the
        # safetensors header -- the same count as campplus_cn_common.bin.
        from safetensors.torch import load_file
        p = hf_hub_download("ResembleAI/chatterbox", "s3gen_v3.safetensors", token=hf_token)
        full = load_file(p)
        sd = {k[len("speaker_encoder."):]: v for k, v in full.items()
              if k.startswith("speaker_encoder.")}
        if not sd:
            raise RuntimeError(f"no speaker_encoder.* keys in s3gen_v3 (saw e.g. {sorted(full)[:4]})")
        dim = 192
    elif kind == "dots":  # dots-studio/dots.tts-soar speaker encoder, 512-d
        # Keys are prefixed `model.` (verified from the safetensors header);
        # the converter's `xvector_extractor.` case is handled too rather than
        # assumed away.
        from safetensors.torch import load_file
        p = hf_hub_download("dots-studio/dots.tts-soar", "speaker_encoder.safetensors", token=hf_token)
        full = load_file(p)
        sd = {}
        for k, v in full.items():
            for pre in ("xvector_extractor.", "model."):
                if k.startswith(pre):
                    k = k[len(pre):]
                    break
            sd[k] = v
        dim = 512
    else:
        raise ValueError(kind)
    _REF_CACHE[kind] = (sd, dim)
    return sd, dim


def ref_embed(kind, fbank, var_floor):
    """Run upstream CAMPPlus on the fbank the C++ actually consumed.

    `var_floor` mirrors the stats_var_floor each backend passes to
    embed_speaker (dots.tts uses 3D-Speaker's masked stats pooling, 1e-2; the
    rest use 0). When it is non-zero the tail of the network is recomputed from
    the out_nonlinear hook so the clamp is applied the same way, rather than
    quietly measuring a different pooling rule and calling the gap a regression.
    """
    sd, dim = ref_state_dict(kind)
    m = CAMPPlus(feat_dim=fbank.shape[1], embedding_size=dim)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "num_batches_tracked" not in k]
    unexpected = [k for k in unexpected if "num_batches_tracked" not in k]
    # A silently half-loaded reference produces a plausible-looking vector that
    # nothing matches, and that reads as "the port regressed". Refuse instead.
    if missing:
        raise RuntimeError(f"{kind}: reference checkpoint is missing {len(missing)} keys "
                           f"(e.g. {missing[:3]}) -- refusing to score against a partly "
                           f"initialised model")
    if unexpected:
        print(f"    [{kind}] note: {len(unexpected)} checkpoint keys unused by the reference "
              f"module (e.g. {unexpected[:3]})")
    m.eval()
    x = torch.from_numpy(np.ascontiguousarray(fbank)).unsqueeze(0)
    if var_floor <= 0:
        return m(x)[0].float().numpy()

    cap = {}
    h = m.xvector.out_nonlinear.register_forward_hook(
        lambda mod, i, o: cap.__setitem__("o", o.detach()))
    m(x)
    h.remove()
    feat = cap["o"][0].float().numpy()                      # (512, T)
    mean = feat.mean(axis=1)
    var = np.maximum(feat.var(axis=1), var_floor)
    stats = np.concatenate([mean, np.sqrt(var)]).astype(np.float32)
    w = sd["xvector.dense.linear.weight"].float().numpy().reshape(dim, -1)
    y = w @ stats
    bn = "xvector.dense.nonlinear.batchnorm."
    rm = sd[bn + "running_mean"].float().numpy()
    rv = sd[bn + "running_var"].float().numpy()
    return ((y - rm) / np.sqrt(rv + 1e-5)).astype(np.float32)


# ── per-backend configuration ───────────────────────────────────────────────
# Sizes chosen so the whole set fits Kaggle disk; every path is an explicit
# file, not `-m auto`, so a resolution surprise cannot be mistaken for a result.
BACKENDS = [
    dict(name="chatterbox", repo="cstr/chatterbox-GGUF",
         files=["chatterbox-v3-t3-q8_0.gguf", "chatterbox-v3-s3gen-q8_0.gguf"],
         model="chatterbox-v3-t3-q8_0.gguf",
         # discover_s3gen()'s sibling list holds the NON-v3 names
         # ("chatterbox-s3gen-q8_0.gguf"), so a v3 s3gen beside the v3 t3 is
         # never found -- name it explicitly instead of relying on discovery.
         extra=["--codec-model", "@chatterbox-v3-s3gen-q8_0.gguf"], env={},
         ref="chatterbox", var_floor=0.0, timeout=3600),
    dict(name="confucius4-tts", repo="cstr/confucius4-tts-GGUF",
         files=["confucius4-tts-t2s-q4_k.gguf", "confucius4-tts-s2a-q4_k.gguf",
                "confucius4-tts-bigvgan-22k-f16.gguf"],
         model="confucius4-tts-t2s-q4_k.gguf",
         extra=["--codec-model", "@confucius4-tts-s2a-q4_k.gguf", "--tts-steps", "16", "-l", "en"],
         env={}, ref="funasr", var_floor=0.0, timeout=5400),
    dict(name="cosyvoice3-tts", repo="cstr/cosyvoice3-0.5b-2512-GGUF",
         files=["cosyvoice3-llm-q4_k.gguf", "cosyvoice3-flow-q8_0.gguf",
                "cosyvoice3-hift-f16.gguf", "cosyvoice3-s3tok-f16.gguf",
                "cosyvoice3-campplus-f16.gguf", "cosyvoice3-voices.gguf"],
         model="cosyvoice3-llm-q4_k.gguf", extra=["-l", "en"], env={},
         ref="funasr", var_floor=0.0, timeout=3600),
    dict(name="dots-tts", repo="cstr/dots-tts-soar-GGUF",
         files=["dots-tts-soar-q4_k.gguf", "dots-tts-soar-vocoder-q4_k.gguf",
                "dots-tts-soar-spk-f16.gguf"],
         model="dots-tts-soar-q4_k.gguf", extra=[], env={},
         ref="dots", var_floor=1e-2, timeout=5400),
    # POSITIVE CONTROL. Known answer: ~0.999452 fixed, ~0.268 legacy. If this
    # backend does not reproduce it, the instrument is wrong and nothing else
    # measured here stands.
    dict(name="fireredtts3", repo="cstr/fireredtts3-GGUF",
         files=["fireredtts3-base-q4_k.gguf", "fireredtts3-redae-f16.gguf"],
         model="fireredtts3-base-q4_k.gguf", extra=["--ref-text", JFK_TEXT],
         env={},  # the CLI discovers fireredtts3-redae-f16.gguf as a sibling

         ref="voxceleb", var_floor=0.0, timeout=3600, control=True),
]

ONLY = os.environ.get("CAMPP_ONLY", "").strip()
if ONLY:
    BACKENDS = [b for b in BACKENDS if b["name"] in ONLY.split(",")]
    print(f"  CAMPP_ONLY -> {[b['name'] for b in BACKENDS]}")

results = {}

for cfg in BACKENDS:
    name = cfg["name"]
    kh.step(f"backend {name}")
    d = MD / name
    d.mkdir(parents=True, exist_ok=True)
    try:
        for f in cfg["files"]:
            hf_hub_download(cfg["repo"], f, local_dir=str(d), token=hf_token)
    except Exception as e:
        print(f"  [{name}] MODEL DOWNLOAD FAILED: {e!r}")
        results[name] = {"status": "MODELS_UNAVAILABLE", "detail": repr(e)}
        continue

    def subst(v):
        return str(d / v[1:]) if isinstance(v, str) and v.startswith("@") else v

    extra = [subst(v) for v in cfg["extra"]]
    base_env = dict(os.environ)
    base_env.update({k: subst(v) for k, v in cfg["env"].items()})
    base_env["OMP_NUM_THREADS"] = "4"

    arms = {}
    for arm in ("fixed", "legacy"):
        dump = d / f"emb_{arm}.bin"
        wav = d / f"out_{arm}.wav"
        for p in (dump, wav):
            if p.exists():
                p.unlink()
        env = dict(base_env)
        env["CRISPASR_CAMPP_DUMP_EMB"] = str(dump)
        if arm == "legacy":
            env["CRISPASR_CAMPP_LEGACY_SEGPOOL"] = "1"
        else:
            env.pop("CRISPASR_CAMPP_LEGACY_SEGPOOL", None)
        cmd = [str(BIN), "--backend", name, "-m", str(d / cfg["model"]),
               "--voice", str(PROMPT_WAV), "--tts", TEST_TEXT,
               "--tts-output", str(wav), "--seed", "42",
               "--i-have-rights", "--no-spoken-disclaimer"] + extra
        t0 = time.monotonic()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg["timeout"], env=env)
            rc, err = r.returncode, r.stderr
        except subprocess.TimeoutExpired as e:
            rc, err = "TIMEOUT", (e.stderr or b"").decode("utf-8", "replace") if e.stderr else ""
        wall = time.monotonic() - t0
        recs = read_records(dump)
        ok_wav = wav.exists() and wav.stat().st_size > 100
        print(f"  [{name}/{arm}] rc={rc} wall={wall:.0f}s records={len(recs)} "
              f"wav={'yes' if ok_wav else 'NO'}")
        for line in err.split("\n"):
            if any(k in line for k in ("campplus:", "voice", "WARNING", "error", "failed",
                                       "GGML_ASSERT", "not found")):
                print(f"    [{name}/{arm}] {line.strip()[:200]}")
        arms[arm] = {"rc": rc, "wall": wall, "recs": recs, "wav": wav if ok_wav else None,
                     "stderr_tail": "\n".join([l for l in err.split("\n") if l.strip()][-20:])}

    res = {"status": "OK", "wall_fixed": arms["fixed"]["wall"]}

    rf, rl = arms["fixed"]["recs"], arms["legacy"]["recs"]
    if not rf or not rl:
        # The speaker encoder never ran. That is NOT a seg_pool result, and it
        # must not be allowed to read as one.
        res["status"] = "NO_EMBEDDING_RECORD"
        res["detail"] = (f"fixed={len(rf)} legacy={len(rl)} records; the CAM++ encoder did not run "
                         f"(voice ignored, or the model lacks the campplus bake)")
        res["stderr_fixed"] = arms["fixed"]["stderr_tail"]
        results[name] = res
        print(f"  [{name}] {res['status']}: {res['detail']}")
        continue

    n_s, t_fb, t_cam, emb_f, fb_f = rf[0]
    _, t_fb_l, t_cam_l, emb_l, fb_l = rl[0]
    res.update(n_samples=n_s, T_fbank=t_fb, T_cam=t_cam, tail=t_cam % 100, dim=int(emb_f.size))

    # C1 — the gate must not touch the front end.
    if fb_f is None or fb_l is None:
        res["C1_fbank_identical"] = "NO_FBANK_IN_RECORD"
    else:
        same_fb = fb_f.shape == fb_l.shape and np.array_equal(fb_f, fb_l)
        res["C1_fbank_identical"] = bool(same_fb)
        if not same_fb:
            res["status"] = "CONTROL_C1_FAILED"

    # C2 — with a partial tail the arms MUST differ.
    arms_differ = not np.array_equal(emb_f, emb_l)
    res["C2_arms_differ"] = bool(arms_differ)
    res["arms_cos"] = cosine(emb_f, emb_l)
    res["norm_fixed"] = float(np.linalg.norm(emb_f))
    res["norm_legacy"] = float(np.linalg.norm(emb_l))
    if res["tail"] != 0 and not arms_differ:
        res["status"] = "GATE_NOT_REACHED"
        res["detail"] = (f"T_cam={t_cam} has a {res['tail']}-frame partial tail, so the two arms "
                         f"must differ -- they are identical, so CRISPASR_CAMPP_LEGACY_SEGPOOL is "
                         f"not reaching this backend and every reading through it is vacuous")
    if res["tail"] == 0:
        # No partial tail: the arms SHOULD be identical, and the A/B carries no
        # information for this clip. Say so instead of reporting a direction.
        res["detail"] = (f"T_cam={t_cam} is a whole number of 100-frame windows, so this clip "
                         f"cannot distinguish the two divisors")
        if arms_differ:
            res["status"] = "CONTROL_C3_FAILED"

    # reference
    if fb_f is not None and res["status"] in ("OK", "GATE_NOT_REACHED"):
        try:
            ref = ref_embed(cfg["ref"], fb_f, cfg["var_floor"])
            cf, cl = cosine(emb_f, ref), cosine(emb_l, ref)
            res.update(cos_fixed=cf, cos_legacy=cl, norm_ref=float(np.linalg.norm(ref)),
                       ref_kind=cfg["ref"], var_floor=cfg["var_floor"])
            # C5 — a reference that nothing matches is a broken reference.
            if max(cf, cl) < 0.5:
                res["reference_verdict"] = "REFERENCE_UNUSABLE"
            elif abs(cf - cl) < 1e-6:
                res["reference_verdict"] = "NO_CHANGE"
            else:
                res["reference_verdict"] = "TOWARD_REFERENCE" if cf > cl else "AWAY_FROM_REFERENCE"
        except Exception as e:
            res["reference_verdict"] = "NO_REFERENCE"
            res["reference_error"] = repr(e)
            print(f"  [{name}] reference unavailable: {e!r}")
    else:
        res.setdefault("reference_verdict", "NO_REFERENCE")

    # end-to-end: whisper roundtrip on the DEFAULT (fixed) arm
    res["e2e_wav"] = arms["fixed"]["wav"] is not None
    results[name] = res
    print(f"  [{name}] {json.dumps({k: v for k, v in res.items() if not k.startswith('stderr')}, default=str)}")

# ── end-to-end ASR roundtrip ────────────────────────────────────────────────
kh.step("ASR roundtrip (default arm)")
whisper_model = TEMP / "ggml-base.en.bin"
if not whisper_model.exists():
    import urllib.request
    urllib.request.urlretrieve(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        str(whisper_model))
ORIG = set(w.strip(".,!?").lower() for w in TEST_TEXT.split())
import soundfile as sf  # noqa: E402

for cfg in BACKENDS:
    name = cfg["name"]
    res = results.get(name, {})
    wav = MD / name / "out_fixed.wav"
    if not wav.exists() or wav.stat().st_size <= 100:
        res["e2e"] = "NO_WAV"
        print(f"  [{name}] e2e NO_WAV")
        continue
    a = subprocess.run([str(BIN), "-m", str(whisper_model), "-f", str(wav),
                        "--no-gpu", "--no-prints"], capture_output=True, text=True, timeout=1200)
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", a.stdout)
    hit = {w for w in ORIG if w in clean.lower()}
    dat, sr = sf.read(str(wav))
    rms = float(np.sqrt(np.mean(np.square(dat)))) if len(dat) else 0.0
    res.update(e2e_overlap=len(hit) / len(ORIG), e2e_rms=rms, e2e_sec=len(dat) / sr,
               e2e_transcript=clean.strip()[:300])
    # A silent file transcribes to nothing and would otherwise score 0 overlap
    # the same way garbled speech does; separate the two.
    res["e2e"] = ("SILENT" if rms < 1e-4 else
                  "PASS" if len(hit) / len(ORIG) >= 0.6 else "WEAK")
    print(f"  [{name}] e2e {res['e2e']} overlap={len(hit)}/{len(ORIG)} "
          f"rms={rms:.4f} {len(dat)/sr:.2f}s :: {clean.strip()[:160]}")

# ── verdict ─────────────────────────────────────────────────────────────────
kh.step("verdict")
ctrl = results.get("fireredtts3", {})
ctrl_ok = (ctrl.get("reference_verdict") == "TOWARD_REFERENCE"
           and ctrl.get("cos_fixed", 0) > 0.99 and ctrl.get("cos_legacy", 1) < 0.9)
print()
print("CONTROL (fireredtts3, known answer ~0.999 fixed / ~0.27 legacy): "
      f"{'REPRODUCED' if ctrl_ok else 'NOT REPRODUCED'}  "
      f"cos_fixed={ctrl.get('cos_fixed')} cos_legacy={ctrl.get('cos_legacy')}")
if not ctrl_ok:
    print("  !! the instrument did not reproduce its known answer -- treat every")
    print("  !! per-backend number below as unproven, not as a result.")
print()
hdr = (f"{'backend':16s} {'status':20s} {'T_cam':>6s} {'tail':>5s} {'cos_fixed':>10s} "
       f"{'cos_legacy':>11s} {'|fixed|':>9s} {'|legacy|':>9s} {'|ref|':>9s} {'verdict':20s} {'e2e':8s}")
print(hdr)
print("-" * len(hdr))
for cfg in BACKENDS:
    n = cfg["name"]
    r = results.get(n, {})
    def g(k, f="{:.6f}"):
        v = r.get(k)
        return f.format(v) if isinstance(v, float) else "-"
    print(f"{n:16s} {r.get('status','?'):20s} {str(r.get('T_cam','-')):>6s} {str(r.get('tail','-')):>5s} "
          f"{g('cos_fixed'):>10s} {g('cos_legacy'):>11s} {g('norm_fixed','{:.4f}'):>9s} "
          f"{g('norm_legacy','{:.4f}'):>9s} {g('norm_ref','{:.4f}'):>9s} "
          f"{r.get('reference_verdict','-'):20s} {str(r.get('e2e','-')):8s}")
print()
for cfg in BACKENDS:
    n = cfg["name"]
    r = results.get(n, {})
    if r.get("detail"):
        print(f"  {n}: {r['detail']}")
    if r.get("reference_error"):
        print(f"  {n}: reference_error {r['reference_error']}")

(WORK / "campp_segpool_results.json").write_text(json.dumps(results, indent=2, default=str))
print()
print("===JSON-BEGIN===")
print(json.dumps(results, indent=2, default=str))
print("===JSON-END===")
print("CAMPP_SEGPOOL_DONE", flush=True)
