#!/usr/bin/env python3
"""#609: bt2-tts A/B on the reporter's defect payloads, CPU, through the server.

The reporter synthesises each Frankenstein chunk with one POST to the CrispASR
server (bt2-tts q4_k, voice "Piedmont" from --voice-dir, the 15-line
instruction, seed 87, cfg_scale 1) and checks it with Whisper. The server
splits a request into <= 600-byte pieces and synthesises each piece as its own
generation with the seed reset, so a piece sent alone is the same generation
the server made for it (bit-exact on one machine; another CPU can round
differently). This harness therefore sends only the pieces that contain a
reported defect, which is what makes the matrix affordable on CPU.

  mode=boundary   000522's over-long sentence (the only one in the book) and
                  the sentence after it, under the old whitespace split
                  (CRISPASR_TTS_CLAUSE_SPLIT=0) and the clause split, one
                  server per arm, interleaved per seed.  voices= seeds=
  mode=defects    every reported defect piece as-is ("orig") and, for the
                  pieces with the French honorific "M.", the same text with
                  "Monsieur" ("monsieur").  voices= seeds=
  mode=control    parakeet on the reporter's own defect audio: which reported
                  defects this readout can see at all (positive control).

Options: quant=q4_k|q8_0|f16 (default q4_k, the reporter's) or q4_k_te8 (the
f16 re-quantized to q4_k with the text encoder's Q4_0-fallback tensors at
Q8_0, built in the job), only=<id|idpN>,... or only=names restricts the
pieces, monsieur=0 drops
the "Monsieur" arm. Downloads are read as data only (json/txt/wav/flac). Every
output goes through the same parakeet roundtrip; per-run word diffs are in
result.json. On a 4-vCPU runner bt2-tts q4_k synthesises at ~7 s per second of
audio, so one defects pass is ~2 h: split seeds across dispatches (heavy-cpu
queues dispatches of one script path, so parallel runs need copies of this
file under other names).

  gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/bt2_609_ab.py \\
     -f pip=huggingface_hub -f timeout_minutes=330 -f args="mode=boundary voices=Piedmont seeds=87,0,1"
"""
import difflib
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import wave
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_controls_common as C  # noqa: E402

ARG = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
MODE = ARG.get("mode", "boundary")
SEEDS = [int(s) for s in ARG.get("seeds", "87").split(",") if s]
VOICES = [v for v in ARG.get("voices", "Piedmont").split(",") if v]
TAG = ARG.get("tag", MODE)
QUANT = ARG.get("quant", "q4_k")  # breeze-tts-2-<quant>.gguf: q4_k (the reporter's), q8_0, f16
MONSIEUR = ARG.get("monsieur", "1") != "0"
# only=<id or idpN>,... restricts mode=defects to some pieces; only=names is the
# 13 pieces with a name / rare-word defect.
NAMES = "000078p0,000120p1,000167p0,000179p0,000186p0,000197p0,000214p0,000239p0,000305p0,000371p0,000700p0," \
        "000733p1,000791p1"
ONLY = {x for x in ARG.get("only", "").replace("names", NAMES).split(",") if x}
NT = str(os.cpu_count() or 4)

ZIPS = {
    "repro": "https://github.com/user-attachments/files/35273668/repro.zip",
    "defect1": "https://github.com/user-attachments/files/35445266/defect1.zip",
    "defect2": "https://github.com/user-attachments/files/35445267/defect2.zip",
}

# Where each reported defect is (an anchor inside the piece that holds it).
# id -> list of anchors; "" = the whole (single-piece) request.
DEFECTS = {
    "000078": ["Beaufort had taken"], "000116": ["continued M. Krempe"], "000117": ["M. Waldman, a fellow"],
    "000118": ["M. Krempe was a little"], "000120": ["which M. Waldman entered", "panegyric"],
    "000125": ["said M. Waldman"], "000130": ["whilst M. Waldman"], "000167": ["pertinacity"],
    "000179": ["to enter into foreign"], "000186": ["M. Duvillard"], "000189": [""],
    "000193": ["M. Krempe was not", "M. Frankenstein is modest"], "000197": ["Clerval had always"],
    "000214": [""], "000239": [""], "000241": ["confusion of manner"], "000305": ["soul-inspiriting"],
    "000325": ["[The moon]"], "000371": ["Safie was always"], "000473": [""],
    "000487": ["triumph; remember that"], "000522": ["where the priest and", "his mistress were"],
    "000525": [""], "000595": ["put in for port. It was"], "000700": [""], "000733": ["adversary's gibe"],
    "000791": ["my sufferings shall"],
}

PROBE_SRC = r'''
#include "crispasr_tts_chunking.h"
#include <iostream>
#include <iterator>
#include <string>
int main() {
    std::string text((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    for (const auto& c : crispasr_tts_plan_chunks_for_backend(text, "bt2-tts"))
        std::cout << c << '\x1e';
    return 0;
}
'''


def fetch_zips():
    out = {}
    for name, url in ZIPS.items():
        d = C.SCRATCH / f"dl-{name}"
        if not d.exists():
            with urllib.request.urlopen(url, timeout=300) as r:
                z = zipfile.ZipFile(io.BytesIO(r.read()))
            for m in z.namelist():
                p = Path(m)
                if p.is_absolute() or ".." in p.parts:
                    raise SystemExit(f"refusing unsafe zip member {m!r} in {name}")
            z.extractall(d)
        out[name] = d
    return out


def build_probe():
    src = C.SCRATCH / "chunkprobe.cpp"
    src.write_text(PROBE_SRC)
    exe = C.SCRATCH / "chunkprobe"
    R = C.ROOT
    C.run(["g++", "-std=c++17", "-O1", f"-I{R}/examples/cli", f"-I{R}/src", f"-I{R}/src/core", f"-I{R}/include",
           src, R / "examples/cli/crispasr_tts_chunking.cpp", "-o", exe], C.OUT / "probe-build.log")
    return exe


def plan(probe, text, old_split):
    env = dict(os.environ)
    if old_split:
        env["CRISPASR_TTS_CLAUSE_SPLIT"] = "0"
    else:
        env.pop("CRISPASR_TTS_CLAUSE_SPLIT", None)
    r = subprocess.run([str(probe)], input=text.encode(), capture_output=True, env=env, check=True)
    return [c for c in r.stdout.decode().split("\x1e") if c]


class Server:
    def __init__(self, b, breeze, codec, voices, port, env_extra, name):
        self.port, self.name = port, name
        env = dict(os.environ)
        env.update(env_extra)
        self.log = open(C.OUT / f"server-{name}.log", "w")
        self.p = subprocess.Popen(
            [str(b), "--server", "--backend", "bt2-tts", "-m", str(breeze), "--codec-model", str(codec), "--host",
             "127.0.0.1", "--port", str(port), "--voice-dir", str(voices), "--accept-license", "other",
             "--i-have-rights", "--speaker-identity", "synthetic", "--no-watermark", "--accept-marking-responsibility",
             "--no-spoken-disclaimer", "-t", NT, "--no-gpu"],
            stdout=self.log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise SystemExit(f"server {name} exited rc={self.p.returncode}; see server-{name}.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200 and b"ok" in r.read().lower():
                        return
            except Exception:
                pass
            time.sleep(3)
        raise SystemExit(f"server {name} not healthy after 900 s")

    def speech(self, payload, out_wav):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/audio/speech",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                out_wav.write_bytes(r.read())
        except urllib.error.HTTPError as e:
            return {"err": f"HTTP {e.code}: {e.read()[:300]!r}", "s": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001
            return {"err": repr(e)[:300], "s": round(time.time() - t0, 1)}
        with wave.open(str(out_wav)) as w:
            dur = w.getnframes() / w.getframerate()
        return {"s": round(time.time() - t0, 1), "dur": round(dur, 2)}

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(60)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()


def asr(b, model, audio):
    r = C.run([b, "-m", model, "-f", audio, "-np", "-nt", "-t", NT], check=False, timeout=1800)
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return " ".join(lines) if r.returncode == 0 else None


# --- scoring --------------------------------------------------------------
# Spelling the ASR is free to choose; none of these is a defect.
SAME = {"labour": "labor", "labourers": "laborers", "favour": "favor", "favourite": "favorite", "honour": "honor",
        "colours": "colors", "fervour": "fervor", "ardour": "ardor", "harbour": "harbor", "grey": "gray",
        "travelled": "traveled", "equalled": "equaled", "endeavoured": "endeavored", "tranquillity": "tranquility",
        "catalogue": "catalog", "solemnising": "solemnizing", "fifty": "50", "twelfth": "12th",
        "eighteenth": "18th", "esq": "esquire", "unborrow'd": "unborrowed", "outstript": "outstripped"}
JOIN = [("for", "ever", "forever"), ("every", "one", "everyone"), ("fairy", "land", "fairyland"),
        ("french", "woman", "frenchwoman"), ("water", "spout", "waterspout"), ("ice", "rifts", "icerifts"),
        ("arch", "enemy", "archenemy"), ("brother", "in"), ("soul", "inspiriting", "soulinspiriting")]


def words(t):
    t = (t or "").lower().replace("’", "'").replace("‘", "'")
    w = [x.strip("'") for x in re.findall(r"[a-z0-9']+", t)]
    w = [SAME.get(x, x) for x in w if x]
    out, i = [], 0
    while i < len(w):
        for j in JOIN:
            if len(j) == 3 and w[i:i + 2] == [j[0], j[1]]:
                out.append(j[2])
                i += 2
                break
        else:
            out.append(w[i])
            i += 1
    return out


def diff_ops(ref, hyp):
    rw, hw = words(ref), words(hyp)
    sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
    ops = []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op != "equal":
            ops.append({"op": op, "at": a0, "ref": " ".join(rw[a0:a1]), "hyp": " ".join(hw[b0:b1]),
                        "ctx": " ".join(rw[max(0, a0 - 3):a0])})
    edits = sum(max(a1 - a0, b1 - b0) for op, a0, a1, b0, b1 in sm.get_opcodes() if op != "equal")
    return {"n_ref": len(rw), "edits": edits, "ops": ops}


def anchor_hit(ref, hyp, anchor):
    """Edits that touch the anchor's words (the reported defect site)."""
    if not anchor:
        return None
    rw, hw = words(ref), words(hyp)
    aw = words(anchor)
    pos = next((i for i in range(len(rw) - len(aw) + 1) if rw[i:i + len(aw)] == aw), None)
    if pos is None:
        return None
    lo, hi = pos, pos + len(aw)
    sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
    bad = []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            continue
        if (a0 < hi and a1 > lo) or (op == "insert" and lo <= a0 <= hi):
            bad.append(f"{' '.join(rw[a0:a1])} -> {' '.join(hw[b0:b1])}")
    return bad


def payloads(d):
    p = {}
    for name in ("defect1", "defect2"):
        for j in sorted((d[name]).glob("[0-9]*.json")):
            p[j.stem] = json.loads(j.read_text(encoding="utf-8"))
    return p


def voices_dir(d):
    root = d["repro"] / "repro"
    v = C.SCRATCH / "voices"
    v.mkdir(exist_ok=True)
    for src, name in ((root / "repro1 - baseline" / "voicedir", "Piedmont"),
                      (root / "repro3 - changed voice" / "voicedir", "mixed")):
        for ext in (".wav", ".txt"):
            (v / f"{name}{ext}").write_bytes((src / f"{name}{ext}").read_bytes())
    return v


def request(base, text, voice, seed):
    q = {k: base[k] for k in ("instructions", "cfg_scale", "consent_attestation", "speaker_identity",
                              "spoken_disclaimer", "response_format") if k in base}
    q.update({"model": "bt2-tts", "input": text, "voice": voice, "seed": seed, "stream": False,
              "response_format": "wav"})
    return q


def main():
    head = subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"head": head, "mode": MODE, "seeds": SEEDS, "voices": VOICES, "runs": []}
    d = fetch_zips()
    b = C.build_crispasr()
    asr_model = C.hf("cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf", None)
    pays = payloads(d)
    wav_dir = C.OUT / "wav"
    wav_dir.mkdir(exist_ok=True)

    if MODE == "control":
        for pid in sorted(pays):
            src = next(iter((d["defect1"]).glob(f"{pid}.flac")), None) or next(iter((d["defect2"]).glob(f"{pid}.flac")))
            heard = asr(b, asr_model, src)
            ref = pays[pid]["input"]
            res["runs"].append({"id": pid, "asr": heard, **diff_ops(ref, heard or ""),
                                "defect_sites": {a: anchor_hit(ref, heard or "", a) for a in DEFECTS[pid] if a}})
            C.save(res)
        write_summary(res)
        return

    if QUANT == "q4_k_te8":
        # The published q4_k keeps the text encoder's d=1152 projections at the
        # Q4_0 row-fit fallback (1152 % 256 != 0 rules out k-quants). This arm
        # re-quantizes the f16 with those tensors at Q8_0 instead.
        f16 = C.hf("cstr/breeze-tts-2-GGUF", "breeze-tts-2-f16.gguf", None)
        C.run(["cmake", "--build", C.SCRATCH / "build", "-j", NT, "--target", "crispasr-quantize"],
              C.OUT / "build-quantize.log")
        breeze = C.SCRATCH / "breeze-tts-2-q4_k_te8.gguf"
        C.run([C.SCRATCH / "build" / "bin" / "crispasr-quantize", f16, breeze, "q4_k", "--tensor-type",
               r"^te\.blk\.[0-9]+\.(attn_(q|k|v)|ffn_(gate|up))\.weight$=q8_0"], C.OUT / "quantize.log")
        os.remove(os.path.realpath(f16))  # the cache blob, 5.3 GiB of runner disk
    else:
        breeze = C.hf("cstr/breeze-tts-2-GGUF", f"breeze-tts-2-{QUANT}.gguf", None)
    res["quant"] = QUANT
    codec = C.hf("cstr/qwen3-tts-tokenizer-12hz-GGUF", "qwen3-tts-tokenizer-12hz.gguf", None)
    vdir = voices_dir(d)
    probe = build_probe()

    if MODE == "boundary":
        full = pays["000522"]["input"]
        start = full.index('"I have seen," he said')
        unit = full[start:]
        arms = {"old_split": {"CRISPASR_TTS_CLAUSE_SPLIT": "0"}, "clause_split": {}}
        res["unit"] = unit
        res["plans"] = {a: plan(probe, unit, a == "old_split") for a in arms}
        servers = {a: Server(b, breeze, codec, vdir, 8091 + i, env, a) for i, (a, env) in enumerate(arms.items())}
        try:
            for voice in VOICES:
                for seed in SEEDS:
                    for arm, srv in servers.items():
                        wav = wav_dir / f"522_{arm}_{voice}_s{seed}.wav"
                        r = srv.speech(request(pays["000522"], unit, voice, seed), wav)
                        heard = asr(b, asr_model, wav) if "err" not in r else None
                        run = {"arm": arm, "id": "000522", "voice": voice, "seed": seed, **r, "asr": heard}
                        if heard is not None:
                            run.update(diff_ops(unit, heard))
                            run["his"] = anchor_hit(unit, heard, "the priest and his mistress were")
                        res["runs"].append(run)
                        C.save(res)
                        print(f"[{arm}] {voice} s{seed}: {r} edits={run.get('edits')} his={run.get('his')}", flush=True)
        finally:
            for s in servers.values():
                s.stop()
        write_summary(res)
        return

    # mode=defects: the pieces the reporter's server generated, as-is, and the
    # "M." pieces again with "Monsieur".
    units = []
    for pid, anchors in DEFECTS.items():
        chunks = plan(probe, pays[pid]["input"], old_split=True)
        for a in anchors:
            idx = next(i for i, c in enumerate(chunks) if a in c) if a else 0
            text = chunks[idx]
            if any(u["id"] == pid and u["text"] == text for u in units):
                continue
            units.append({"id": pid, "piece": idx, "n_pieces": len(chunks), "text": text,
                          "anchors": [x for x in anchors if x and x in text]})
    if ONLY:
        units = [u for u in units if u["id"] in ONLY or f"{u['id']}p{u['piece']}" in ONLY]
    res["units"] = units
    # Pieces are <= 600 bytes, so the split rule cannot touch them; pin the
    # old one anyway so this arm is main's behaviour by construction.
    srv = Server(b, breeze, codec, vdir, 8091, {"CRISPASR_TTS_CLAUSE_SPLIT": "0"}, "main")
    try:
        for voice in VOICES:
            for seed in SEEDS:
                for u in units:
                    arms = {"orig": u["text"]}
                    if MONSIEUR and re.search(r"\bM\. [A-Z]", u["text"]):
                        arms["monsieur"] = re.sub(r"\bM\. (?=[A-Z])", "Monsieur ", u["text"])
                    for arm, text in arms.items():
                        wav = wav_dir / f"{u['id']}p{u['piece']}_{arm}_{voice}_s{seed}.wav"
                        r = srv.speech(request(pays[u["id"]], text, voice, seed), wav)
                        heard = asr(b, asr_model, wav) if "err" not in r else None
                        run = {"arm": arm, "id": u["id"], "piece": u["piece"], "voice": voice, "seed": seed, **r,
                               "asr": heard}
                        if heard is not None:
                            run.update(diff_ops(text, heard))
                            anchors = u["anchors"] if arm == "orig" else [
                                re.sub(r"\bM\. (?=[A-Z])", "Monsieur ", a) for a in u["anchors"]]
                            run["defect_sites"] = {a: anchor_hit(text, heard, a) for a in anchors}
                        res["runs"].append(run)
                        C.save(res)
                        print(f"[{arm}] {u['id']}p{u['piece']} {voice} s{seed}: {r} edits={run.get('edits')} "
                              f"sites={run.get('defect_sites')}", flush=True)
    finally:
        srv.stop()
    write_summary(res)


def write_summary(res):
    runs = res["runs"]
    lines = [f"## #609 bt2-tts A/B `{res['mode']}` on `{res['head'][:9]}` ({res.get('quant', 'q4_k')}, CPU, server, "
             f"CFG 1, seeds {res['seeds']}, voices {res['voices']})", ""]
    bad = [r for r in runs if r.get("err") or r.get("asr") is None]
    if res["mode"] == "boundary":
        lines += ["| arm | runs | his kept | mean edits / ref words | edits per run |", "|---|---|---|---|---|"]
        for arm in ("old_split", "clause_split"):
            rs = [r for r in runs if r.get("arm") == arm and r.get("asr") is not None]
            if not rs:
                continue
            kept = sum(1 for r in rs if r.get("his") == [])
            e = [r["edits"] for r in rs]
            lines.append(f"| {arm} | {len(rs)} | {kept} | {sum(e) / len(rs):.2f} / {rs[0]['n_ref']} | {e} |")
        lines += ["", "plans: " + json.dumps({a: [len(c) for c in p] for a, p in res.get("plans", {}).items()})]
    else:
        lines += ["| arm | runs | defect sites hit | total edits |", "|---|---|---|---|"]
        for arm in sorted({r.get("arm", "control") for r in runs}):
            rs = [r for r in runs if r.get("arm", "control") == arm and r.get("asr") is not None]
            sites = [v for r in rs for v in (r.get("defect_sites") or {}).values() if v is not None]
            lines.append(f"| {arm} | {len(rs)} | {sum(1 for v in sites if v)}/{len(sites)} | "
                         f"{sum(r['edits'] for r in rs)} |")
    lines.append(f"\nfailed runs: {len(bad)}")
    C.summary(lines)
    if bad:
        raise SystemExit(f"{len(bad)} run(s) failed; see result.json")


if __name__ == "__main__":
    main()
