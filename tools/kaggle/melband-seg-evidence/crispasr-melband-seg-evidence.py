"""
CrispASR — mel-band-roformer SEGMENTATION evidence (review finding on PR #422).

THE GAP THIS CLOSES. PR #422 added Demucs-style segmentation (25% overlap,
triangular weight, normalised by accumulated weight) for audio longer than the
model's trained chunk. It is the largest behavioural change in that PR and it
has NO reference-anchored evidence, for a structural reason:

  * The PR's headline 30 s arm compares per-layer graphs vs the fused graph.
    Segmentation lives ABOVE the path gates, so BOTH arms are segmented
    identically and any overlap-add error cancels exactly between them. That
    arm is real evidence about per-layer-vs-fused and is incapable of being
    evidence about segmentation.
  * The golden10s fixture sits exactly at the threshold (the guard is
    `n_samples <= seg_len`), so it takes the whole-buffer path and never
    segments at all.

Already established on the VPS, so this kernel does NOT re-test it: the
overlap-add ARITHMETIC is sound — the schedule was replicated with the model as
identity over ~200 (length, seg_len) combinations including every seg_len+1
boundary, giving zero coverage holes, no sum_weight==0 division, and exact
reconstruction to 2.4e-07. What is untested is whether the SEGMENTED PIPELINE
matches the reference model on long audio: receptive-field effects at chunk
boundaries and the zero-padded tail of the final segment.

DESIGN. The reference (lucidrains MelBandRoformer) runs the WHOLE clip in one
pass — it has no chunking of its own — so it is the unsegmented ground truth.
The clip must therefore be long enough to force C++ to segment but short enough
that the reference's O(T^2) attention fits in GPU memory. samples/jfk.wav is
~11 s; at 44.1 kHz that is T ~ 1100 frames, roughly (1100/3000)^2 ~ 13% of the
~19 GB a 30 s clip needs. Comfortable.

ARMS, all against the same reference output:
  ref        reference, unsegmented                      <- ground truth
  A no-seg   C++ CRISPASR_MELBAND_NO_SEGMENT=1           <- validates the port
             on long audio independently of segmentation
  B default  C++ default = the checkpoint's TRAINED chunk (8 s for Kim, read
             from GGUF `mel-band-roformer.chunk_size`)   <- THE QUESTION
  C seg=3    C++ CRISPASR_MELBAND_SEG_S=3                <- graded control

THE CONTROL IS THE POINT. Arm C has ~3x more chunk boundaries per unit time
than arm B. If B and C score the SAME against the reference, the metric cannot
see boundary effects and the whole run is uninformative no matter how good the
numbers look. A verdict is only reported if C is measurably worse than B, or
if B is close enough to A that the segmentation cost is bounded.

Metrics are SDR + magnitude ratio, NOT cosine alone: cosine saturates near 1 on
audio and is scale-blind. Comparison is on the int16 PCM PAYLOAD, never the WAV
file — the container's trailing C2PA chunk carries a timestamp and differs
every run.

Both sides use weights converted IN THIS KERNEL from the same
KimberleyJSN/melbandroformer checkpoint (MIT), so "do the two sides even have
the same weights" is not a variable.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
OUT = WORK / "out"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
SCRIPT_VERSION = "seg-evidence-v1.2-realimport"


def run(cmd, check=True, env=None, timeout=None, cwd=None):
    print(f"  $ {' '.join(str(c) for c in cmd) if isinstance(cmd, list) else cmd}", flush=True)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), env=env, cwd=cwd,
                       timeout=timeout, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        print(f"  !! rc={r.returncode}", flush=True)
        if r.stderr:
            print(r.stderr[-6000:], flush=True)
        if check:
            raise SystemExit(f"command failed: {cmd}")
    elif r.stderr:
        print(r.stderr[-2000:], flush=True)
    return r


# ── 1. clone ───────────────────────────────────────────────────────
print("[1/8] Clone", flush=True)
OUT.mkdir(parents=True, exist_ok=True)
run(["git", "clone", "--recursive", "--depth", "1", "-b", CRISPASR_REF, CRISPASR_REPO, str(REPO)])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
prov = kh.provenance(SCRIPT_VERSION, REPO)
print(f"  provenance: {prov}", flush=True)
kh.resolve_hf_token()
run(["nvidia-smi", "-L"], check=False)

# ── 2. build ───────────────────────────────────────────────────────
print("\n[2/8] Build CUDA", flush=True)
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (["cmake", "-S", str(REPO), "-B", str(BUILD),
               "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_BUILD_TESTS=OFF"]
              + kh.crispasr_cmake_flags() + kh.cache_and_link_flags() + kh.cuda_build_flags(arch))
run(cmake_args)
kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr-cli -j {kh.safe_build_jobs(True)}")
CLI = BUILD / "bin" / "crispasr"
assert CLI.exists(), "crispasr-cli missing"

# ── 3. deps + checkpoint ───────────────────────────────────────────
print("\n[3/8] Reference deps + Kim checkpoint", flush=True)
# `gguf` is NOT in the Kaggle image and the converter imports it (v1 run
# 2026-09-03 died here after the full build + 913 MB checkpoint download).
run("pip install -q 'bs-roformer==0.3.10' soundfile librosa gguf 2>&1 | tail -5", check=False)
# Verify the EXACT symbols the converter imports (line 106), not a proxy.
# v1.1 asserted gguf.__version__, which the package does not define — so the
# guard added to catch a silent pip failure became the failure itself, after
# the full build and the 913 MB download. Test the thing that must work.
run([sys.executable, "-c",
     "from gguf import GGUFWriter, GGMLQuantizationType; print('gguf import OK', GGUFWriter)"], check=True)
from huggingface_hub import snapshot_download  # noqa: E402

CKPT_DIR = Path(snapshot_download("KimberleyJSN/melbandroformer",
                                  cache_dir=str(WORK / "hf"),
                                  token=os.environ.get("HF_TOKEN")))
print(f"  checkpoint dir: {CKPT_DIR}", flush=True)
for p in sorted(CKPT_DIR.rglob("*")):
    if p.is_file() and p.suffix in (".ckpt", ".yaml", ".yml", ".pt", ".bin"):
        print(f"    {p.name}  {p.stat().st_size/1e6:.1f} MB", flush=True)

# ── 4. convert to GGUF from the SAME checkpoint ────────────────────
print("\n[4/8] Convert -> GGUF (same weights both sides)", flush=True)
GGUF = WORK / "melband-vocals-f32.gguf"
run([sys.executable, str(REPO / "models" / "convert-mel-band-roformer-to-gguf.py"),
     "--model", str(CKPT_DIR), "--output", str(GGUF), "--dtype", "f32"])
print(f"  GGUF: {GGUF.stat().st_size/1e6:.1f} MB", flush=True)

# ── 5. input: 11 s, 44.1 kHz stereo ────────────────────────────────
print("\n[5/8] Prepare input", flush=True)
import soundfile as sf  # noqa: E402
import librosa  # noqa: E402

SRC = REPO / "samples" / "jfk.wav"
y, sr0 = librosa.load(str(SRC), sr=44100, mono=True)
stereo = np.stack([y, y], axis=-1).astype(np.float32)
IN_WAV = WORK / "in_44k_stereo.wav"
sf.write(str(IN_WAV), stereo, 44100, subtype="PCM_16")
dur = len(y) / 44100.0
T_frames = len(y) // 441
print(f"  {IN_WAV.name}: {dur:.2f} s, T~{T_frames} frames, 44.1 kHz stereo", flush=True)
if dur <= 8.0:
    raise SystemExit(f"input {dur:.2f}s does not exceed the 8 s trained chunk — it would never segment")

# ── 6. reference, unsegmented ──────────────────────────────────────
print("\n[6/8] Reference (unsegmented ground truth)", flush=True)
sys.path.insert(0, str(REPO / "tools" / "reference_backends"))
import torch  # noqa: E402
import mel_band_roformer as mbr_ref  # noqa: E402

model, ref_cfg = mbr_ref._load(str(CKPT_DIR))  # returns (model, cfg_model_dict)
model.eval()
print(f"  reference cfg: dim={ref_cfg.get('dim')} depth={ref_cfg.get('depth')} "
      f"heads={ref_cfg.get('heads')} num_bands={ref_cfg.get('num_bands')}", flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev)
with torch.no_grad():
    x = torch.from_numpy(stereo.T).unsqueeze(0).to(dev)  # (1, 2, T)
    t0 = time.time()
    ref_out = model(x)
    print(f"  reference forward: {time.time()-t0:.1f}s, out shape {tuple(ref_out.shape)}", flush=True)
ref = ref_out.squeeze(0).detach().cpu().numpy()
if ref.ndim == 3:      # (stems, ch, T) -> take stem 0 (vocals)
    ref = ref[0]
ref_pcm = np.clip(ref.T, -1.0, 1.0)  # (T, ch)
del model, ref_out
torch.cuda.empty_cache()
print(f"  reference vocals: {ref_pcm.shape}, rms={np.sqrt((ref_pcm**2).mean()):.5f}", flush=True)
# Write the reference so the comparison can be REDONE offline against it. v1
# kept it in memory only, which meant a failed or unexpected verdict could not
# be re-analysed without paying for the whole run again.
REF_WAV = OUT / "ref_vocals.wav"
sf.write(str(REF_WAV), ref_pcm.astype(np.float32), 44100, subtype="PCM_16")
print(f"  wrote {REF_WAV}", flush=True)

# ── 7. C++ arms ────────────────────────────────────────────────────
print("\n[7/8] C++ arms", flush=True)
ARMS = [
    ("A_noseg",  {"CRISPASR_MELBAND_NO_SEGMENT": "1"}),
    ("B_default", {}),
    ("C_seg3",   {"CRISPASR_MELBAND_SEG_S": "3"}),
]
cpp = {}
for name, extra in ARMS:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **extra)
    r = run([str(CLI), "--separate", "-m", str(GGUF), "-f", str(IN_WAV),
             "--sep-output-dir", str(d)], check=False, env=env, timeout=5400)
    seg_lines = [l for l in (r.stderr or "").splitlines() if "segment" in l.lower()]
    print(f"  [{name}] segmentation log: {seg_lines[:2] or 'NONE (whole-buffer)'}", flush=True)
    wavs = sorted(d.rglob("*vocal*.wav")) or sorted(d.rglob("*.wav"))
    if not wavs:
        print(f"  [{name}] NO OUTPUT — arm void", flush=True)
        continue
    a, _sr = sf.read(str(wavs[0]), dtype="float32", always_2d=True)
    cpp[name] = a
    print(f"  [{name}] {wavs[0].name}: {a.shape}, rms={np.sqrt((a**2).mean()):.5f}", flush=True)

# ── 8. compare on the PCM payload ──────────────────────────────────
print("\n[8/8] Comparison (SDR + magnitude ratio; payload only)", flush=True)


def metrics(est, refr):
    n = min(len(est), len(refr))
    e, r = est[:n], refr[:n]
    if e.shape[1] != r.shape[1]:
        e = e.mean(axis=1, keepdims=True)
        r = r.mean(axis=1, keepdims=True)
    ef, rf = e.reshape(-1).astype(np.float64), r.reshape(-1).astype(np.float64)
    noise = ef - rf
    sdr = 10 * np.log10((rf @ rf) / max(noise @ noise, 1e-20))
    cos = (ef @ rf) / max(np.sqrt((ef @ ef) * (rf @ rf)), 1e-20)
    mag = np.sqrt(ef @ ef) / max(np.sqrt(rf @ rf), 1e-20)
    return sdr, cos, mag, n


print(f"\n{'arm':<12}{'SDR dB':>10}{'cosine':>12}{'|est|/|ref|':>14}{'samples':>10}")
res = {}
for name, _ in ARMS:
    if name not in cpp:
        print(f"{name:<12}{'VOID':>10}")
        continue
    sdr, cos, mag, n = metrics(cpp[name], ref_pcm)
    res[name] = sdr
    print(f"{name:<12}{sdr:10.2f}{cos:12.6f}{mag:14.4f}{n:10d}", flush=True)

print("\n--- verdict ---", flush=True)
if "B_default" not in res or "C_seg3" not in res:
    print("INCONCLUSIVE: an arm is void; cannot read the control.", flush=True)
else:
    delta = res["B_default"] - res["C_seg3"]
    print(f"control: SDR(B_default) - SDR(C_seg3) = {delta:+.2f} dB", flush=True)
    if abs(delta) < 0.05:
        print("METRIC IS BLIND: 3 s segmenting (3x the boundaries) scores the same as the", flush=True)
        print("8 s default, so this comparison cannot see boundary effects. NO VERDICT on", flush=True)
        print("segmentation is supportable from this run, regardless of the absolute SDRs.", flush=True)
    else:
        print("Metric responds to boundary density, so the arms are readable.", flush=True)
        if "A_noseg" in res:
            print(f"segmentation cost at the trained chunk: "
                  f"{res['A_noseg'] - res['B_default']:+.2f} dB vs unsegmented", flush=True)
print("\nNOTE: the reference has no chunking of its own, so 'ref' is the unsegmented", flush=True)
print("ground truth. A_noseg vs ref measures the PORT; B_default vs ref measures", flush=True)
print("SEGMENTATION on top of it.", flush=True)

# ── 9. shrink the kernel output ────────────────────────────────────
# The clone + build tree live under /kaggle/working and would otherwise BE the
# kernel output — retrieving even the log from the v1 run meant walking the
# whole repo. Keep only what the offline analysis needs.
print("\n[9/9] Pruning output", flush=True)
import shutil  # noqa: E402

for junk in (REPO, BUILD, WORK / "hf"):
    try:
        shutil.rmtree(junk, ignore_errors=True)
    except Exception as e:
        print(f"  prune {junk}: {e}", flush=True)
try:
    GGUF.unlink(missing_ok=True)  # regenerable from the checkpoint
except Exception:
    pass
kept = sorted(p for p in WORK.rglob("*") if p.is_file())
print(f"  kept {len(kept)} files, "
      f"{sum(p.stat().st_size for p in kept)/1e6:.1f} MB", flush=True)
for p in kept[:20]:
    print(f"    {p.relative_to(WORK)}  {p.stat().st_size/1e6:.2f} MB", flush=True)
