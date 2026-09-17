#!/usr/bin/env python3
"""Breeze TTS 2 (#412): is q4_k good enough to be the registry default?

THE QUESTION. The published q4_k preserves none of frame 0's codes beyond
codebook 0 while still producing intelligible speech, so code-exactness and
perceptual quality have come apart. Every code-level gate we own is therefore
BLIND to whatever q4_k is doing to the audio, and q4_k is what `-m auto` hands
people today. So the audio has to be measured directly.

THE DESIGN.
  * three arms — f16 (the ceiling), q8_0 (the middle), q4_k (the default);
  * the SAME four sentences and the SAME voice in every arm, because a single
    short clip cannot separate three codecs;
  * UNCONDITIONED synthesis, not voice cloning. Cloning would pin the speaker
    and be the better control, but it requires --i-have-rights, an attestation
    that the operator has the speaker's consent. That is not a box an automated
    benchmark may tick, so the weaker control is the correct one here and the
    limitation is stated rather than worked around;
  * one fixed seed and one max-new-tokens cap across all arms, so no arm can
    win or lose on utterance length;
  * WER and word-overlap against the known target text, plus the raw
    transcript, because an aggregate hides the difference between a mangled
    word and a dropped clause;
  * file size reported alongside, because the decision is quality-per-GB.

THE CONTROLS. Two, and neither is decoration:
  * the ASR's own floor — every arm is scored by the same whisper, so a bad
    score that appears in ALL THREE arms is the ASR or the text, not the
    quantizer;
  * code-exactness per arm from the same diff dump, so the code-level metric
    and the perceptual one can be seen diverging in a single table rather than
    argued about.

WHAT WOULD CHANGE THE DECISION. If q8_0 is materially better than q4_k, the
answer is a one-line registry default change and the quant carve-out is not
touched at all. Widening the carve-out is a code change affecting every future
conversion and is NOT the first move.

Push (chr1str):
  export KAGGLE_API_TOKEN=<chr1str token>
  python -m kaggle kernels push -p tools/kaggle/breeze-quant-ab
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_VERSION = "2026-09-17.3"
WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else WORK
REPO = WORK / "CrispASR"
BRANCH = os.environ.get("CRISPASR_REF", "feat/412-breeze-tts-2")

HF_MODEL = "cstr/breeze-tts-2-GGUF"
HF_CODEC = "cstr/qwen3-tts-tokenizer-12hz-GGUF"
HF_FIX = "cstr/crispasr-regression-fixtures"
FIX_PREFIX = "breeze-tts-2"

# ORDER MATTERS: q4_k is the arm under scrutiny and q8_0 is the candidate
# replacement, so the decisive comparison lands before the ceiling. Results
# are written after every arm, so a timeout during f16 still leaves the
# actionable q4_k-vs-q8_0 answer rather than nothing.
QUANTS = ["q4_k", "q8_0", "f16"]
SEED = "42"

# Four sentences, deliberately varied: a pangram, digits and a proper noun, a
# longer clause, and ordinary prose. One short clip cannot separate three
# codecs, and different failure modes (mangled word vs dropped clause) show up
# on different material.
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sold seventeen blue umbrellas in Manchester last Tuesday.",
    "Although the weather had turned, the travellers pressed on toward the harbour before nightfall.",
    "Please remember to close the window before you leave the office.",
]

verdict = {"script_version": SCRIPT_VERSION, "conclusive": False, "arms": {}}


def save():
    (WORK / "quant_ab.json").write_text(json.dumps(verdict, indent=1))


# Orthographic normalisation BEFORE scoring. Run 2 showed why this is not
# fussiness: whisper writes "seventeen" as "17" and Americanises "travellers"
# and "harbour". Those are the ASR's spelling conventions, not the TTS's
# pronunciation, and unnormalised they dominated the ranking — q8_0 scored
# WORSE than q4_k purely by collecting one more spelling artifact. A metric
# that ranks quantizers by the ASR's orthography is measuring the wrong thing.
_NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
            "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
            "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
            "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
            "19": "nineteen", "20": "twenty"}
_BRIT = {"traveller": "traveler", "travellers": "travelers", "harbour": "harbor",
         "colour": "color", "favourite": "favorite", "neighbour": "neighbor",
         "theatre": "theater", "centre": "center", "realise": "realize"}


def norm_words(s):
    out = []
    for w in re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split():
        w = _NUMBERS.get(w, w)
        out.append(_BRIT.get(w, w))
    return out


def wer(ref, hyp):
    """Word error rate by Levenshtein on word lists. 0.0 == perfect."""
    r, h = norm_words(ref), norm_words(hyp)
    if not r:
        return 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / len(r)


def overlap(ref, hyp):
    """Multiset word overlap — insensitive to order, unlike WER."""
    from collections import Counter
    r, h = Counter(norm_words(ref)), Counter(norm_words(hyp))
    if not r:
        return 0.0
    return sum((r & h).values()) / sum(r.values())


print(f"=== breeze-quant-ab {SCRIPT_VERSION} (branch {BRANCH}) ===", flush=True)

if not REPO.exists():
    for attempt in range(4):
        if REPO.exists():
            shutil.rmtree(REPO)
        rc = subprocess.run(["git", "clone", "--depth", "1", "--recursive",
                             "--shallow-submodules", "-b", BRANCH,
                             "https://github.com/CrispStrobe/CrispASR", str(REPO)]).returncode
        if rc == 0:
            break
    else:
        raise SystemExit("clone failed")
subprocess.check_call(["git", "log", "--oneline", "-1"], cwd=str(REPO))
subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
               cwd=str(REPO), check=False)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
if hasattr(kh, "provenance"):
    kh.provenance(SCRIPT_VERSION, clone_dir=REPO)
kh.sh_with_progress("pip install -q huggingface_hub numpy soundfile")
hf_token = kh.resolve_hf_token()
os.environ["HF_TOKEN"] = hf_token or ""
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

kh.step("build")
kh.install_build_toolchain()
BUILD = REPO / "build"
subprocess.check_call(
    ["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
     "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF"] + kh.cache_and_link_flags())
with kh.build_heartbeat("build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} --target crispasr crispasr-diff -j {kh.safe_build_jobs(gpu=False)}")
crispasr = BUILD / "bin" / "crispasr"
diffbin = BUILD / "bin" / "crispasr-diff"
assert crispasr.is_file() and diffbin.is_file()

kh.step("download codec + fixture")
codec = hf_hub_download(HF_CODEC, "qwen3-tts-tokenizer-12hz.gguf", token=hf_token,
                        local_dir=str(TEMP / "m"))
fixdir = str(Path(snapshot_download(HF_FIX, repo_type="dataset", token=hf_token,
                                    allow_patterns=[f"{FIX_PREFIX}/*"],
                                    local_dir=str(TEMP / "fix"))) / FIX_PREFIX)
# No voice reference is loaded: see the synthesis call for why cloning is not
# used here. Leaving a VOICE path defined but unused would imply the speaker is
# pinned when it is not.

env = dict(os.environ, CRISPASR_ACCEPT_LICENSE="other", BREEZE_CODEC=codec)

import numpy as np  # noqa: E402

ref_frame0 = np.load(Path(fixdir) / "dd_codes_frame0_stepwise.npy")

for q in QUANTS:
    kh.step(f"arm {q}")
    arm = {"quant": q}
    try:
        gguf = hf_hub_download(HF_MODEL, f"breeze-tts-2-{q}.gguf", token=hf_token,
                               local_dir=str(TEMP / "m"))
    except Exception as e:
        arm["error"] = f"download failed: {e}"
        verdict["arms"][q] = arm
        save()
        continue
    arm["size_gib"] = round(os.path.getsize(gguf) / 2**30, 2)
    print(f"[{q}] {arm['size_gib']} GiB", flush=True)

    # --- code exactness, from the same binary that measures parity ---------
    dump = WORK / f"dump_{q}"
    dump.mkdir(exist_ok=True)
    d = subprocess.run([str(diffbin), "bt2-tts", gguf, fixdir, str(dump)],
                       capture_output=True, text=True, timeout=7200, env=env)
    f0 = dump / "dd_codes_frame0_stepwise.npy"
    if f0.exists():
        got = np.load(f0)
        n = min(len(got), len(ref_frame0))
        arm["frame0_codes_match"] = f"{int((got[:n] == ref_frame0[:n]).sum())}/{n}"
    else:
        arm["frame0_codes_match"] = "n/a"
        print(f"[{q}] dump stderr: {d.stderr[-1500:]}", flush=True)

    # --- perceptual: same sentences, same voice, same seed -----------------
    arm["sentences"] = []
    for i, text in enumerate(SENTENCES):
        wav = WORK / f"{q}_{i}.wav"
        # NO --voice. Cloning requires --i-have-rights, whose attestation is
        # "I have the consent of the speaker whose voice this clones, or it is
        # my own voice." samples/jfk.wav is a real person; that is a claim only
        # a human with standing can make, and auto-passing the flag to make a
        # benchmark run would be exactly the box-ticking the gate exists to
        # prevent. Run 1 of this A/B discovered that the hard way: all twelve
        # syntheses were refused with rc=17 and the gate was right to do it.
        #
        # Unconditioned synthesis instead, with a fixed seed and length cap.
        # The speaker is then whatever the model produces rather than a pinned
        # reference — a weaker control, and stated as such — but the metric is
        # INTELLIGIBILITY of the same four sentences under three quantizations,
        # which this still measures honestly.
        s = subprocess.run([str(crispasr), "--backend", "bt2-tts", "-m", gguf,
                            "--codec-model", codec, "--accept-license", "other",
                            "--seed", SEED, "--max-new-tokens", "220",
                            "--tts", text, "--tts-output", str(wav)],
                           capture_output=True, text=True, timeout=7200, env=env)
        row = {"target": text, "synth_rc": s.returncode, "wav": wav.is_file()}
        if not wav.is_file():
            row["stderr"] = s.stderr[-1500:]
            arm["sentences"].append(row)
            # PRINT the reason. Run 1 captured stderr into a dict field that the
            # summary dropped, so twelve identical failures reported a bare
            # "rc=17" and the cause had to be recovered from the C++ source
            # afterwards. A readout that cannot say why it failed wastes the
            # whole run it was measuring.
            print(f"[{q}][{i}] SYNTH FAILED rc={s.returncode}\n"
                  f"    stdout: {s.stdout[-400:].strip()}\n"
                  f"    stderr: {s.stderr[-1200:].strip()}", flush=True)
            continue
        import soundfile as sf
        a, sr = sf.read(str(wav))
        row.update(seconds=round(len(a) / sr, 2), peak=round(float(abs(a).max()), 4),
                   rms=round(float((a ** 2).mean() ** 0.5), 5))
        r = subprocess.run([str(crispasr), "--backend", "whisper", "-m", "auto",
                            "--auto-download", "-f", str(wav), "-l", "en", "--no-prints"],
                           capture_output=True, text=True, timeout=2400, env=env)
        hyp = " ".join(r.stdout.split())
        row.update(asr=hyp, wer=round(wer(text, hyp), 4), overlap=round(overlap(text, hyp), 4))
        arm["sentences"].append(row)
        print(f"[{q}][{i}] {row['seconds']}s wer={row['wer']} ovl={row['overlap']} :: {hyp[:110]}",
              flush=True)

    ok = [r for r in arm["sentences"] if "wer" in r]
    if ok:
        arm["mean_wer"] = round(sum(r["wer"] for r in ok) / len(ok), 4)
        arm["mean_overlap"] = round(sum(r["overlap"] for r in ok) / len(ok), 4)
        arm["n_scored"] = len(ok)
    verdict["arms"][q] = arm
    save()
    print(f"[{q}] MEAN wer={arm.get('mean_wer')} overlap={arm.get('mean_overlap')} "
          f"size={arm['size_gib']} GiB codes={arm['frame0_codes_match']}", flush=True)
    os.remove(gguf)  # ~10.5 GiB across three arms; the scratch will not hold them

kh.step("summary")
print("\n=== QUANT A/B ===", flush=True)
print(f"{'quant':6} {'size GiB':>9} {'mean WER':>9} {'mean ovl':>9} {'frame0 codes':>13}", flush=True)
for q in QUANTS:
    a = verdict["arms"].get(q, {})
    print(f"{q:6} {a.get('size_gib','?'):>9} {a.get('mean_wer','?'):>9} "
          f"{a.get('mean_overlap','?'):>9} {a.get('frame0_codes_match','?'):>13}", flush=True)
verdict["conclusive"] = True
save()
print("\nVERDICT " + json.dumps({q: {k: v for k, v in verdict["arms"].get(q, {}).items()
                                    if k != "sentences"} for q in QUANTS}), flush=True)
print("[DONE]", flush=True)
