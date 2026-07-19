"""CREPE accuracy eval on real monophonic music.

No hand-labelled ground truth, so we use three load-independent proxies:

 1. OCTAVE DISAGREEMENT tiny-vs-full — the specific failure the handoff forbids.
    An octave error shows as |log2(f0_a/f0_b)| within 0.08 of exactly 1.0.
 2. CENTS DEVIATION from the nearest equal-tempered semitone. Real instrument
    pitches cluster near semitones; octave/garbage errors do not. Reported as
    the median |deviation| and the fraction within +/-50 cents.
 3. RANGE PLAUSIBILITY vs the instrument's real tessitura.

Only frames with voiced_prob >= THRESH are scored — that is how a note
segmenter would consume this, and unvoiced frames have no true pitch.
"""
import subprocess, sys, os, math, json

CLI = "./build/bin/crispasr"
SONGS = "/private/tmp/claude-501/-Users-christianstrobele-code-mus/c4eefe59-aea5-4022-98d3-840f0bb73f22/scratchpad/songs"
SCRATCH = "/private/tmp/claude-501/-Volumes-backups-ai-CrispASR/60fa4f2b-c095-4ac2-ba89-5799246836e7/scratchpad"
THRESH = 0.5

# rough tessitura (Hz) for a sanity check, generous bounds
RANGE = {
    "01_violin_scale": (196, 2637), "02_violin_pizz": (196, 2637),
    "03_fur_elise": (65, 2093), "04_glock_entchen": (523, 4186),
    "05_carillon_ode": (262, 2093), "06_cello_bach": (65, 880),
    "07_flute_bach": (262, 2093), "08_row_boat": (196, 1047),
    "09_old_macdonald": (196, 1047), "10_amazing_brass": (117, 932),
}


def run(model, wav):
    out = subprocess.run([CLI, "--pitch", "-m", model, "-f", wav],
                         capture_output=True, text=True, timeout=1800)
    frames = []
    for line in out.stdout.splitlines():
        p = line.split("\t")
        if len(p) == 3:
            try:
                frames.append((float(p[0]), float(p[1]), float(p[2])))
            except ValueError:
                pass
    return frames


def cents_dev(f0):
    """Signed cents from the nearest equal-tempered semitone (A4=440)."""
    if f0 <= 0:
        return None
    midi = 69 + 12 * math.log2(f0 / 440.0)
    return (midi - round(midi)) * 100.0


def main():
    models = {"tiny": f"{SCRATCH}/crepe-tiny-f16.gguf",
              "full": f"{SCRATCH}/crepe-full-f16.gguf"}
    rows = []
    for wav in sorted(f for f in os.listdir(SONGS) if f.endswith(".wav")):
        stem = wav[:-4]
        res = {}
        for name, path in models.items():
            fr = run(path, os.path.join(SONGS, wav))
            v = [(t, f, c) for (t, f, c) in fr if c >= THRESH]
            res[name] = v
            print(f"  {stem:22s} {name:5s} frames={len(fr):5d} voiced={len(v):5d}", flush=True)

        r = {"song": stem}
        for name in models:
            v = res[name]
            if not v:
                r[name] = None
                continue
            devs = [abs(d) for d in (cents_dev(f) for _, f, _ in v) if d is not None]
            lo, hi = RANGE.get(stem, (0, 99999))
            inrange = sum(1 for _, f, _ in v if lo * 0.97 <= f <= hi * 1.03) / len(v)
            f0s = sorted(f for _, f, _ in v)
            r[name] = {
                "voiced": len(v),
                "median_f0": f0s[len(f0s) // 2],
                "median_cents_dev": sorted(devs)[len(devs) // 2] if devs else None,
                "within_50c": sum(1 for d in devs if d <= 50) / len(devs) if devs else 0,
                "in_range": inrange,
            }

        # octave disagreement on the frames both call voiced
        a = {round(t): f for t, f, _ in res["tiny"]}
        b = {round(t): f for t, f, _ in res["full"]}
        common = sorted(set(a) & set(b))
        oct_err = sum(1 for t in common
                      if a[t] > 0 and b[t] > 0 and abs(abs(math.log2(a[t] / b[t])) - 1.0) < 0.08)
        big = sum(1 for t in common
                  if a[t] > 0 and b[t] > 0 and abs(math.log2(a[t] / b[t])) > 0.08)
        r["common"] = len(common)
        r["octave_disagree"] = oct_err / len(common) if common else None
        r["any_disagree"] = big / len(common) if common else None
        rows.append(r)
        print(f"    -> octave-disagree {r['octave_disagree']:.1%}  "
              f"any-disagree {r['any_disagree']:.1%}  (n={len(common)})", flush=True)

    with open(f"{SCRATCH}/pitch_eval.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    print("\n" + "=" * 100)
    print(f"{'song':22s} {'tiny medF0':>10s} {'full medF0':>10s} {'t cents':>8s} {'f cents':>8s} "
          f"{'t<50c':>7s} {'f<50c':>7s} {'oct-dis':>8s}")
    print("=" * 100)
    for r in rows:
        t, f = r.get("tiny"), r.get("full")
        if not t or not f:
            print(f"{r['song']:22s}  (no voiced frames)")
            continue
        print(f"{r['song']:22s} {t['median_f0']:10.1f} {f['median_f0']:10.1f} "
              f"{t['median_cents_dev']:8.1f} {f['median_cents_dev']:8.1f} "
              f"{t['within_50c']:7.1%} {f['within_50c']:7.1%} {r['octave_disagree']:8.1%}")


if __name__ == "__main__":
    main()
