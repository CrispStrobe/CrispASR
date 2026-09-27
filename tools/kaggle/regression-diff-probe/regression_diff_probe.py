#!/usr/bin/env python3
"""Measure the diff harness on skip_diff entries whose ref.gguf is ALREADY on the
pinned fixtures revision, before flipping them to gated diffs.

Their notes say "Q4K diverges (all encoder layers cos<0.9)" / "cohere encoder
cos=-0.253" - numbers taken before the harness snapshot-layout fix (3bc9f1ac).
For each entry: the pinned GGUF and, where the same repo has one, an F16/F32
GGUF, run through tests/regression/run_one.py with skip_diff off and every stage
ungated (INFO). CPU build, exactly like the GH regression job; the kernel keeps
its GPU only for internet access.
"""
import json, os, re, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); G = Path("/tmp/g"); G.mkdir(exist_ok=True)
REF = os.environ.get("CRISPASR_REF", "main")
ENTRIES = {  # name -> fixture_ref_path on the pinned fixtures revision
    "parakeet-tdt-0.6b-en": "parakeet-tdt-0.6b-en/ref.gguf",
    "parakeet-rnnt-0.6b": "parakeet-rnnt-0.6b/ref.gguf",
    "parakeet-tdt-1.1b": "parakeet-tdt-1.1b/ref.gguf",
    "parakeet-tdt_ctc-1.1b": "parakeet-tdt_ctc-1.1b/ref.gguf",
    "parakeet-rnnt-1.1b": "parakeet-rnnt-1.1b/ref.gguf",
    "cohere-transcribe": "cohere-transcribe/jfk_11s/ref.gguf",
}
res = {"errors": [], "entries": {}}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
LINE = re.compile(r"(PASS|FAIL|INFO)\s+(\S+)\s+cos_min=(-?[0-9.eE+-]+)")
try:
    subprocess.check_call(["git", "clone", "-b", REF, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle")); import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok: os.environ["HF_TOKEN"] = tok
    from huggingface_hub import HfApi
    kh.install_build_toolchain()
    kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_BUILD_TESTS=OFF "
          "-DCRISPASR_BUILD_EXAMPLES=ON -DCRISPASR_BUILD_SERVER=OFF " + " ".join(kh.cache_and_link_flags()))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {REPO}/build -j$(nproc) --target crispasr-cli crispasr-diff")
    B = REPO / "build" / "bin"
    man = json.loads((REPO / "tests/regression/manifest.json").read_text())
    api = HfApi()
    for name, ref in ENTRIES.items():
        e = next(b for b in man["backends"] if b["name"] == name)
        arms = [("pinned", e["gguf"])]
        try:
            files = api.list_repo_files(e["gguf"]["repo"])
            full = [f for f in files if f.endswith(".gguf") and not re.search(r"q\d|iq\d|_k|tq\d", f.lower())]
            if full:
                arms.append(("full", {"repo": e["gguf"]["repo"], "revision": "main", "file": sorted(full, key=len)[0]}))
        except Exception as ex:
            res["errors"].append(f"{name} list: {ex}")
        res["entries"][name] = {}
        for arm, gg in arms:
            e2 = dict(e, skip_diff=False, fixture_ref_path=ref, diff_thresholds={}, gguf=gg)
            for k in ("stage_threshold_default", "advisory_stages", "transcript_tolerance"):
                e2.pop(k, None)
            e2["transcript_tolerance"] = {"wer_max": 1.0, "cer_max": 1.0}  # measuring, not gating
            m2 = dict(man, backends=[e2]); mp = G / f"m_{name}_{arm}.json"; mp.write_text(json.dumps(m2))
            env = dict(os.environ, REGRESSION_MANIFEST=str(mp), CRISPASR_BIN=str(B / "crispasr"),
                       DIFF_BIN=str(B / "crispasr-diff"), WORK_DIR=str(G / "w"), HF_HOME=str(G / "hf"))
            r = subprocess.run([sys.executable, str(REPO / "tests/regression/run_one.py"), name],
                               capture_output=True, text=True, env=env, cwd=str(REPO), timeout=3000)
            txt = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)
            stages = {m.group(2): float(m.group(3)) for m in LINE.finditer(txt)}
            act = re.search(r"actual:\s+'(.*)'", txt)
            res["entries"][name][arm] = {"gguf": gg["file"], "rc": r.returncode, "stages": stages,
                                         "transcript": act.group(1) if act else ("byte-equal" if "byte-equal" in txt else None),
                                         "tail": txt[-1200:] if not stages else ""}
            print(name, arm, gg["file"], {k: round(v, 5) for k, v in stages.items() if not k.startswith("encoder_layer_")}, flush=True)
            save()
            subprocess.run(f"rm -rf {G}/hf/hub/models--cstr--*GGUF* {G}/w", shell=True)
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
