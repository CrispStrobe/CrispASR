"""
CrispASR — mel-band-roformer SEGMENTATION evidence, v2 (PR #422 review finding).

v2 of crispasr-melband-seg-evidence. The v1 pilot (chr1s4/crispasr-melband-seg-
evidence) established the method; this run fixes three things that made v1
structurally unable to deliver a trustworthy verdict. Everything through the
GGUF conversion is v1's scaffold unchanged.

WHY v1 COULD NOT ANSWER THE QUESTION
------------------------------------
1. ITS VERDICT GATE WAS AMBIGUOUS — the load-bearing flaw. v1 printed
   "METRIC IS BLIND" when |SDR(B)-SDR(C)| < 0.05 dB. But a null delta is
   equally consistent with "there was no boundary artefact to see", which is
   the RESULT you want, not a void run. Two opposite meanings, one output
   string. v2 never infers from a null: it MEASURES the boundary regions
   directly (see WITHIN-ARM below).

2. THE STIMULUS WAS DEGENERATE. v1 used samples/jfk.wav — solo speech through
   a VOCALS separator, where the model is near pass-through (PLAN.md records
   vocals rms 0.218 vs other rms 0.005, a 41x ratio on speech). Every arm then
   scores high and the very delta v1 was trying to read is compressed toward
   its own blind threshold. Instrumental music is degenerate in the mirror
   direction (vocals stem ~ silence). v2 uses real music WITH VOCALS so both
   stems are non-trivial.

3. ARM B HAD ONE INTERNAL BOUNDARY. At 11.000 s with the 8 s trained chunk
   (352800 samples, stride 264600) the loop yields exactly 2 segments and ONE
   crossfade, with the final segment 38% zero-pad. The arm validating the
   configuration users actually ship rested on a single boundary. v2 uses a
   20 s excerpt: 4 segments, 3 internal boundaries — 3x the evidence — while
   the reference stays unsegmented, which is the property the design rests on.
   Sizing from v1's own numbers: 11 s ~ 2.5 GB of a 16 GB card, and O(T^2)
   scaling puts 20 s at ~3.3x that (~8 GB), leaving ~50% headroom. 30 s does
   not fit; that ceiling is real and is why the clip is not longer.

THE WITHIN-ARM TEST (the reason v2 can conclude something v1 could not)
----------------------------------------------------------------------
Instead of inferring segmentation quality from a cross-arm SDR delta, measure
it where it lives. Arm A (CRISPASR_MELBAND_NO_SEGMENT=1) is the same binary,
same weights, same graph path, differing from arm B ONLY in segmentation — so
A is the ideal local reference for isolating segmentation, better than torch
because it removes the port's own f32/graph differences as a confound. Then
for arm B vs A, compare SDR inside the overlap windows against SDR in the
segment interiors:

    boundary SDR ~= interior SDR   -> POSITIVE evidence of no overlap-add
                                      artefact; a real measurement, not a null
    boundary SDR <  interior SDR   -> the artefact, localised to the crossfades

This is immune to both of v1's ambiguities. It never compares across segment
lengths (so it cannot confuse "boundary density" with "3 s buffers driven
outside an 8 s-trained model"), and a clean result is positive evidence rather
than an unfalsifiable null.

The segment geometry is NOT assumed. The C++ prints
"segmented N samples into K chunks of L (stride S, overlap P%)"; v2 parses that
line and derives the windows from what actually ran, so a change to
resolve_segment_len cannot silently invalidate the analysis. The arm also
records WHICH branch fired (trained chunk_size from GGUF metadata vs the 8 s
fallback) so the middle arm is never assumed to be 352800.

ARMS
----
  ref        reference, unsegmented (torch)              <- absolute ground truth
  A no-seg   C++ CRISPASR_MELBAND_NO_SEGMENT=1           <- validates the PORT
  B default  C++ default = trained chunk from GGUF       <- THE SHIPPED PATH
  C seg=3    C++ CRISPASR_MELBAND_SEG_S=3                <- graded control

METHOD CONSTRAINTS (both from mistakes actually made in this repo)
-----------------------------------------------------------------
* Compare the int16 PCM PAYLOAD, never whole WAV files: the container's
  trailing C2PA/AI-disclosure chunk carries a TIMESTAMP and differs every run.
  A `cmp` over files once produced a confident wrong conclusion here.
* State the printed precision, and never report cosine alone: cosine saturates
  near 1 on audio and is scale-blind, so a real difference can render as "1".
  SDR and |est|/|ref| are printed alongside it at fixed width.
* SDR here is 10*log10(||ref||^2 / ||ref-est||^2) — signal-to-residual against
  the named reference, NOT BSS-Eval SDR with its projection step. Stated so the
  number is not silently compared against BSS-Eval figures elsewhere.

v1 also computed the torch reference in memory and never wrote it out, so it
could not be reused offline. v2 writes ref.wav into the kernel output.

STIMULUS: librosa's `fishin` example — Karissa Hobbs, "Let's Go Fishin'"
(ccMixter, Creative Commons). Vocals over acoustic accompaniment: verified
locally that ASR transcribes the sung lyrics from the chosen 20 s excerpt, so
the vocals stem is genuinely non-trivial. Fetched by librosa itself, so no
asset is vendored and no licence question is introduced.

Weights: KimberleyJSN/melbandroformer (MIT), converted IN THIS KERNEL for both
sides, so "do the two sides have the same weights" is not a variable.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO = SCRATCH / "CrispASR"     # NOT under /kaggle/working: not kernel output
BUILD = SCRATCH / "build"       # same
OUT = WORK / "out"              # the only thing we want back: stems + ref + json

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
SCRIPT_VERSION = "seg-evidence-v2"


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
print("[1/9] Clone", flush=True)
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
# ── 1b. dependencies FIRST — fail in seconds, not after the build ──
# v1 died at `from gguf import ...` inside the converter, ~16 min in, having
# already paid for the CUDA build and the 913 MB checkpoint. The import is
# cheap to verify and the build is not, so verify first.
print("\n[1b/9] Python deps (before the expensive build)", flush=True)
# gguf is NOT optional and is NOT on the Kaggle image: the converter does
# `from gguf import GGUFWriter, GGMLQuantizationType` inside main(), so it dies
# at conversion AFTER the full CUDA build and the 913 MB checkpoint download —
# ~16 minutes of quota for a ModuleNotFoundError. This is exactly how v1 died.
run("pip install -q 'bs-roformer==0.3.10' gguf soundfile librosa 2>&1 | tail -5", check=False)
# Assert the EXACT symbols the converter imports, not a proxy for them.
# crispasr-dc's v1.2 died on its own guard — it asserted gguf.__version__,
# which the package does not define — so the check for a missing dependency
# became a second way to fail. find_spec() has the same weakness in milder
# form: it proves a module is importable, not that the names exist in it.
try:
    from gguf import GGMLQuantizationType, GGUFWriter  # noqa: F401  (exactly what the converter imports)
    import librosa  # noqa: F401
    import soundfile  # noqa: F401
except Exception as _e:
    raise SystemExit(f"FATAL: dependency check failed ({_e!r}) — aborting BEFORE the CUDA build "
                     f"and the 913 MB checkpoint download, which is where v1 discovered this.")
print("  deps OK: gguf.GGUFWriter + gguf.GGMLQuantizationType, soundfile, librosa", flush=True)

print("\n[2/9] Build CUDA", flush=True)
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
print("\n[3/9] Reference deps + Kim checkpoint", flush=True)
from huggingface_hub import snapshot_download  # noqa: E402

CKPT_DIR = Path(snapshot_download("KimberleyJSN/melbandroformer",
                                  cache_dir=str(SCRATCH / "hf"),
                                  token=os.environ.get("HF_TOKEN")))
print(f"  checkpoint dir: {CKPT_DIR}", flush=True)
for p in sorted(CKPT_DIR.rglob("*")):
    if p.is_file() and p.suffix in (".ckpt", ".yaml", ".yml", ".pt", ".bin"):
        print(f"    {p.name}  {p.stat().st_size/1e6:.1f} MB", flush=True)

# ── 3b. the config the checkpoint does not ship ────────────────────
# KimberleyJSN/melbandroformer contains ONLY MelBandRoformer.ckpt. Both the
# converter and the reference loader glob for *.yaml and exit, because the band
# layout is not recoverable from the weights. v1 and v2 both died here, ~5 min
# in, AFTER the CUDA build and the 913 MB download. The values are recorded in
# docs/mel-band-roformer/PLAN.md and synthesised by a shared helper so the next
# person to convert or reference-dump this checkpoint does not rediscover it.
print("\n[3b/9] Synthesise the Kim config (checkpoint ships no YAML)", flush=True)
sys.path.insert(0, str(REPO / "tools"))
from melband_kim_config import verify_against_checkpoint, write_config  # noqa: E402

write_config(CKPT_DIR)

# A synthesised config is exactly the kind of thing that is silently
# almost-right: a wrong dim/depth/num_bands/dim_freqs_in still BUILDS and RUNS
# and emits subtly wrong audio. strict=True turns that into a shape mismatch in
# seconds. Cheap check, run BEFORE the conversion and the reference forward.
verify_against_checkpoint(CKPT_DIR)

# ── 4. convert to GGUF from the SAME checkpoint ────────────────────
print("\n[4/9] Convert -> GGUF (same weights both sides)", flush=True)
GGUF = WORK / "melband-vocals-f32.gguf"
run([sys.executable, str(REPO / "models" / "convert-mel-band-roformer-to-gguf.py"),
     "--model", str(CKPT_DIR), "--output", str(GGUF), "--dtype", "f32"])
print(f"  GGUF: {GGUF.stat().st_size/1e6:.1f} MB", flush=True)

# ── 5. input: 20 s of real music WITH VOCALS, 44.1 kHz stereo ──────
# Not speech (near pass-through through a vocals separator) and not
# instrumental (vocals stem ~ silence) — both are degenerate stimuli for this
# model in opposite directions. 20 s gives arm B 4 segments / 3 internal
# boundaries at the 8 s trained chunk while keeping the reference's O(T^2)
# attention at ~8 GB of a 16 GB card.
print("\n[5/9] Prepare input (20 s music with vocals)", flush=True)
import soundfile as sf  # noqa: E402
import librosa  # noqa: E402

CLIP_S = float(os.environ.get("MBR_CLIP_S", "20"))
CLIP_OFF_S = float(os.environ.get("MBR_CLIP_OFF_S", "10"))
SR = 44100

music_path = librosa.example("fishin")  # Karissa Hobbs, "Let's Go Fishin'" (ccMixter, CC)
y, _ = librosa.load(music_path, sr=SR, mono=False)
if y.ndim == 1:
    y = np.stack([y, y], axis=0)
a = int(CLIP_OFF_S * SR)
b = a + int(CLIP_S * SR)
if b > y.shape[-1]:
    raise SystemExit(f"clip window {CLIP_OFF_S}+{CLIP_S}s exceeds source ({y.shape[-1]/SR:.1f}s)")
stereo = np.ascontiguousarray(y[:, a:b].T).astype(np.float32)  # (T, ch)
IN_WAV = WORK / "in_music_44k_stereo.wav"
sf.write(str(IN_WAV), stereo, SR, subtype="PCM_16")
dur = stereo.shape[0] / SR
print(f"  source: {Path(music_path).name}", flush=True)
print(f"  excerpt: {CLIP_OFF_S:.1f}-{CLIP_OFF_S+CLIP_S:.1f}s -> {dur:.2f}s, "
      f"{stereo.shape[0]} samples, T~{stereo.shape[0]//441} frames, stereo", flush=True)
print(f"  rms={np.sqrt((stereo**2).mean()):.4f} peak={np.abs(stereo).max():.3f} "
      f"silent_frac={np.mean(np.abs(stereo) < 1e-4):.4f}", flush=True)
if dur <= 8.0:
    raise SystemExit(f"input {dur:.2f}s does not exceed the 8 s trained chunk — it would never segment")

# ── 6. reference, unsegmented — AND WRITTEN TO DISK (v1 could not reuse it) ──
print("\n[6/9] Reference (unsegmented ground truth)", flush=True)
sys.path.insert(0, str(REPO / "tools" / "reference_backends"))
import torch  # noqa: E402
import mel_band_roformer as mbr_ref  # noqa: E402

model, ref_cfg = mbr_ref._load(str(CKPT_DIR))
model.eval()
print(f"  reference cfg: dim={ref_cfg.get('dim')} depth={ref_cfg.get('depth')} "
      f"heads={ref_cfg.get('heads')} num_bands={ref_cfg.get('num_bands')}", flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev)
if dev == "cuda":
    torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    x = torch.from_numpy(stereo.T).unsqueeze(0).to(dev)  # (1, ch, T)
    t0 = time.time()
    ref_out = model(x)
    fwd_s = time.time() - t0
if dev == "cuda":
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  reference forward: {fwd_s:.1f}s, peak VRAM {peak_gb:.2f} GB "
          f"(headroom check for the 20 s sizing)", flush=True)
else:
    print(f"  reference forward: {fwd_s:.1f}s (CPU)", flush=True)
ref = ref_out.squeeze(0).detach().cpu().numpy()
if ref.ndim == 3:
    ref = ref[0]
ref_pcm = np.clip(ref.T, -1.0, 1.0)  # (T, ch)
del model, ref_out
torch.cuda.empty_cache()
REF_WAV = OUT / "ref_vocals.wav"
sf.write(str(REF_WAV), ref_pcm, SR, subtype="PCM_16")   # v1 hole: ref never hit disk
print(f"  reference vocals: {ref_pcm.shape}, rms={np.sqrt((ref_pcm**2).mean()):.5f} "
      f"-> {REF_WAV.name}", flush=True)

# ── 7. C++ arms, capturing the ACTUAL segment geometry ─────────────
# The geometry is parsed from the runtime's own log line rather than assumed,
# so a change to resolve_segment_len cannot silently invalidate the analysis
# below. Also records which branch fired (trained chunk vs 8 s fallback).
print("\n[7/9] C++ arms", flush=True)
import re  # noqa: E402

SEG_RE = re.compile(r"segmented\s+(\d+)\s+samples into\s+(\d+)\s+chunks of\s+(\d+)\s+\(stride\s+(\d+)")
ARMS = [
    ("A_noseg",   {"CRISPASR_MELBAND_NO_SEGMENT": "1"}),
    ("B_default", {}),
    ("C_seg3",    {"CRISPASR_MELBAND_SEG_S": "3"}),
]
cpp, geom = {}, {}
for name, extra in ARMS:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **extra)
    r = run([str(CLI), "--separate", "-m", str(GGUF), "-f", str(IN_WAV),
             "--sep-output-dir", str(d)], check=False, env=env, timeout=5400)
    err = r.stderr or ""
    m = SEG_RE.search(err)
    if m:
        n_s, k, L, S = (int(g) for g in m.groups())
        geom[name] = {"n": n_s, "chunks": k, "seg_len": L, "stride": S}
        print(f"  [{name}] segmented: {k} chunks of {L} ({L/SR:.2f}s), stride {S} ({S/SR:.2f}s)", flush=True)
    else:
        geom[name] = None
        print(f"  [{name}] whole-buffer (no segmentation)", flush=True)
    wavs = sorted(d.rglob("*vocal*.wav")) or sorted(d.rglob("*.wav"))
    if not wavs:
        print(f"  [{name}] NO OUTPUT — arm void", flush=True)
        continue
    arr, _sr = sf.read(str(wavs[0]), dtype="float32", always_2d=True)
    cpp[name] = arr

gB = geom.get("B_default")
if gB:
    branch = "trained chunk_size from GGUF" if gB["seg_len"] != 8 * SR else "8 s (either trained==8s or the fallback)"
    print(f"  default seg_len = {gB['seg_len']} samples = {gB['seg_len']/SR:.3f}s  [{branch}]", flush=True)
if geom.get("A_noseg") is not None:
    print("  !! WARNING: arm A segmented; NO_SEGMENT did not take effect — A is not a clean reference", flush=True)

# ── 8. whole-clip metrics vs the torch reference ───────────────────
# SDR := 10*log10(||ref||^2 / ||ref-est||^2) — signal-to-residual against the
# named reference, NOT BSS-Eval SDR (no projection step). Printed to 2 dp;
# cosine to 6 dp (it saturates near 1 on audio, so it is never read alone).
print("\n[8/9] Whole-clip vs reference  [SDR 2dp = 10log10(|ref|^2/|ref-est|^2); cos 6dp]", flush=True)


def to_i16(a):
    return np.clip(a, -1.0, 1.0).astype(np.float64)


def metrics(est, ref_a, sl=None):
    n = min(len(est), len(ref_a))
    e, rr = to_i16(est[:n]), to_i16(ref_a[:n])
    if sl is not None:
        e, rr = e[sl], rr[sl]
    ef, rf = e.reshape(-1), rr.reshape(-1)
    noise = rf - ef
    denom = float(noise @ noise)
    sdr = 10 * np.log10(float(rf @ rf) / max(denom, 1e-20))
    cos = float(ef @ rf) / max(np.linalg.norm(ef) * np.linalg.norm(rf), 1e-20)
    mag = float(np.linalg.norm(ef) / max(np.linalg.norm(rf), 1e-20))
    return sdr, cos, mag, len(ef) // max(rr.shape[1], 1)


def sdr_and_resid(est, ref_a, sl):
    """SDR plus the RAW residual energy, so an exactly-zero residual can be
    reported as a STATE rather than as a number. crispasr-dc's improvement on
    the FLOOR_DB heuristic below: `residual == 0` is unambiguous, whereas a dB
    threshold would also swallow an arm that is legitimately very close but not
    identical — which is a real result, not a floor artefact."""
    n = min(len(est), len(ref_a))
    e = to_i16(est[:n])[sl].reshape(-1)
    r = to_i16(ref_a[:n])[sl].reshape(-1)
    d = r - e
    resid = float(d @ d)
    sdr = 10 * np.log10(float(r @ r) / max(resid, 1e-20))
    return sdr, resid


print(f"\n{'arm':<12}{'SDR dB':>10}{'cosine':>12}{'|est|/|ref|':>14}{'samples':>10}", flush=True)
whole = {}
for name, _ in ARMS:
    if name not in cpp:
        print(f"{name:<12}{'VOID':>10}", flush=True)
        continue
    sdr, cos, mag, n = metrics(cpp[name], ref_pcm)
    whole[name] = sdr
    print(f"{name:<12}{sdr:10.2f}{cos:12.6f}{mag:14.4f}{n:10d}", flush=True)

# ── 9. WITHIN-ARM boundary localisation — the measurement v1 could not make ──
# Arm A is the same binary, weights and graph path, differing ONLY in
# segmentation, so it isolates segmentation better than torch (no f32/graph
# confound). Comparing boundary windows against interiors WITHIN one arm is
# immune to v1's two ambiguities: it never compares across segment lengths, and
# a clean result is positive evidence rather than an unfalsifiable null.
# Thresholds, calibrated offline against synthetic arms before this ever ran on
# GPU (a tolerance wider than the defect is not a test). On a realistic signal
# where B differs from A by a small broadband amount: no crossfade artefact
# gives delta = +0.01 dB; 2x noise at the crossfades gives -3.01 dB; 10x gives
# -19.14 dB. So -1.0 dB sits well clear of the no-artefact case and well below
# the smallest injected defect. FLOOR_DB catches the bit-identical degenerate
# case described at the verdict below.
ARTEFACT_DB = float(os.environ.get("MBR_ARTEFACT_DB", "-1.0"))
FLOOR_DB = float(os.environ.get("MBR_FLOOR_DB", "120"))

print("\n[9/9] Within-arm boundary localisation (reference = arm A, no-seg)", flush=True)


def boundary_mask(n, seg_len, stride):
    """True inside overlap windows: segment k covers [k*stride, k*stride+seg_len),
    so consecutive segments overlap on [(k+1)*stride, k*stride+seg_len)."""
    m = np.zeros(n, dtype=bool)
    offs, off = [], 0
    while off < n:
        offs.append(off)
        off += stride
    for k in range(len(offs) - 1):
        lo, hi = offs[k + 1], min(offs[k] + seg_len, n)
        if hi > lo:
            m[lo:hi] = True
    return m, offs


if "A_noseg" not in cpp:
    print("  SKIPPED: arm A void, no local reference", flush=True)
else:
    A = cpp["A_noseg"]
    # Sample counts per region are printed so the reader can see how much data
    # each half of the comparison rests on — a boundary estimate drawn from one
    # crossfade is far noisier than one drawn from three.
    print(f"\n{'arm':<12}{'boundary SDR':>14}{'interior SDR':>14}{'delta dB':>10}"
          f"{'#bnd':>6}{'bnd smp':>10}{'int smp':>10}", flush=True)
    local = {}
    for name in ("B_default", "C_seg3"):
        if name not in cpp or not geom.get(name):
            print(f"{name:<12}{'n/a (unsegmented or void)':>14}", flush=True)
            continue
        g = geom[name]
        n = min(len(cpp[name]), len(A))
        mask, offs = boundary_mask(n, g["seg_len"], g["stride"])
        n_bnd = max(len(offs) - 1, 0)
        if mask.sum() == 0 or (~mask).sum() == 0:
            print(f"{name:<12}  degenerate mask — cannot localise", flush=True)
            continue
        b_sdr, b_res = sdr_and_resid(cpp[name], A, mask)
        i_sdr, i_res = sdr_and_resid(cpp[name], A, ~mask)
        local[name] = (b_sdr, i_sdr, n_bnd, b_res, i_res)
        print(f"{name:<12}{b_sdr:14.2f}{i_sdr:14.2f}{b_sdr - i_sdr:10.2f}"
              f"{n_bnd:6d}{int(mask.sum()):10d}{int((~mask).sum()):10d}", flush=True)

    print("\n--- VERDICT ---", flush=True)
    # Power is a property of the STIMULUS and is reported separately from the
    # measurement, so an underpowered run can never be read as "no artefact"
    # (v1's gate conflated exactly these two).
    nb = local.get("B_default", (None, None, 0, 0.0, 0.0))[2]
    if nb < 2:
        print(f"UNDERPOWERED: arm B has only {nb} internal boundary/boundaries on this clip.", flush=True)
        print("Any null below is weak evidence — lengthen the clip rather than trusting it.", flush=True)
    else:
        print(f"POWER: arm B has {nb} internal boundaries on this clip.", flush=True)

    if "B_default" in local:
        b_sdr, i_sdr, _, b_res, i_res = local["B_default"]
        d = b_sdr - i_sdr
        print(f"Arm B boundary-vs-interior SDR delta = {d:+.2f} dB "
              f"(boundary {b_sdr:.2f}, interior {i_sdr:.2f}).", flush=True)
        # Degenerate guard, found by testing this logic offline before spending
        # GPU time: if an arm is bit-identical to A in a region the residual is
        # exactly zero, SDR is pinned by the 1e-20 epsilon, and the two regions'
        # differing signal energy manufactures a spurious delta (measured -3.68
        # dB on identical arrays — enough to trip the artefact threshold below as
        # a FALSE POSITIVE). Anything past ~120 dB is the floor, not a result.
        if b_res == 0.0 or i_res == 0.0:
            where = "boundary" if b_res == 0.0 else "interior"
            if b_res == 0.0 and i_res == 0.0:
                where = "both"
            print(f"BIT-IDENTICAL to arm A in the {where} region(s) — reporting the STATE, not a", flush=True)
            print("number. The residual is exactly zero there, so SDR is pinned by the epsilon and", flush=True)
            print("any delta would be fabricated from the regions' differing signal energy.", flush=True)
            print("Segmentation either did not engage or is an exact no-op — read the", flush=True)
            print("segmentation log line before concluding anything.", flush=True)
        elif max(b_sdr, i_sdr) > FLOOR_DB:
            print(f"NEAR THE MEASUREMENT FLOOR (>{FLOOR_DB:.0f} dB): the arms differ, but by so little that", flush=True)
            print("the delta is dominated by numerical noise. Reported for completeness, not as a", flush=True)
            print("verdict.", flush=True)
        elif d < ARTEFACT_DB:
            print("ARTEFACT LOCALISED AT THE CROSSFADES: the overlap windows reconstruct", flush=True)
            print("measurably worse than segment interiors. This is a real finding — the", flush=True)
            print("overlap-add arithmetic is known sound, so suspect receptive-field/context", flush=True)
            print("effects at chunk edges, not the adder.", flush=True)
        else:
            print("NO BOUNDARY ARTEFACT DETECTED: the overlap windows reconstruct as well as", flush=True)
            print("the segment interiors. This is a positive, localised measurement rather", flush=True)
            print("than an inference from a null delta.", flush=True)

    if whole.get("A_noseg") is not None and whole.get("B_default") is not None:
        print(f"Segmentation cost vs the port itself: SDR(A)={whole['A_noseg']:.2f} dB, "
              f"SDR(B)={whole['B_default']:.2f} dB, cost={whole['A_noseg'] - whole['B_default']:+.2f} dB.", flush=True)

# Small machine-readable summary alongside the wavs, so a second implementation
# can cross-check these numbers rather than re-deriving them from the log.
import json  # noqa: E402

summary = {
    "script_version": SCRIPT_VERSION,
    "clip": {"source": Path(music_path).name, "offset_s": CLIP_OFF_S,
             "dur_s": round(dur, 3), "sr": SR, "samples": int(stereo.shape[0])},
    "geometry": geom,
    "whole_clip_sdr_vs_torch_ref": {k: round(v, 4) for k, v in whole.items()},
    "within_arm_vs_A": {
        k: {"boundary_sdr": round(v[0], 4), "interior_sdr": round(v[1], 4),
            "n_internal_boundaries": v[2], "boundary_resid": v[3], "interior_resid": v[4]}
        for k, v in local.items()
    } if "A_noseg" in cpp else {},
    "thresholds": {"artefact_db": ARTEFACT_DB, "floor_db": FLOOR_DB},
    "sdr_definition": "10*log10(|ref|^2 / |ref-est|^2) vs the named reference; NOT BSS-Eval",
}
(OUT / "results.json").write_text(json.dumps(summary, indent=2))
print(f"\nWrote {OUT/'results.json'}", flush=True)
print("Outputs (kernel-output mount only; clone/build/hf kept off it):", flush=True)
print("  out/ref_vocals.wav, out/<arm>/*.wav, out/results.json", flush=True)
print("DONE", flush=True)
