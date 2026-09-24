#!/usr/bin/env python3
"""docs/contributing.md gate on Kaggle: build the CLI + shared libcrispasr from
a branch, regenerate the generated artifacts, run the wiring audit and the unit
tier. Outputs: regenerated docs/feature-matrix.{md,html} +
src/core/backend_caps_table.h, and every log, under out/.

Branch: BRANCH below (a Kaggle script kernel takes no arguments)."""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path
BRANCH = "chore/454-455-contrib"
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); BUILD = REPO / "build"
res = {"errors": [], "steps": {}}
def save(): (OUT / "wiring.json").write_text(json.dumps(res, indent=1, default=str))
def run(name, cmd, cwd=REPO, timeout=7200):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    (OUT / f"{name}.log").write_text(cmd + "\n" + r.stdout[-60000:] + "\n--- stderr ---\n" + r.stderr[-30000:])
    res["steps"][name] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]}; save()
    return r.returncode
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.install_build_toolchain()
    kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON "
          f"-DCRISPASR_BUILD_TESTS=ON " + " ".join(kh.cache_and_link_flags()))
    with kh.build_heartbeat("build"):
        rc = run("build", f"cmake --build {BUILD} -j{os.cpu_count()} --target crispasr-cli crispasr-lib", timeout=10800)
    assert rc == 0, "build failed"
    cli = BUILD / "bin" / "crispasr"
    run("caps_check", f"python3 tools/gen-backend-caps-table.py --check --crispasr {cli}")
    run("caps_regen", f"python3 tools/gen-backend-caps-table.py --crispasr {cli}")
    run("matrix_regen", f"python3 tools/gen-feature-matrix.py --crispasr {cli}")
    run("wiring", f"python3 tools/check-backend-wiring.py --crispasr {cli} --require-lib")
    run("go_ldflags", "python3 tools/sync_go_cgo_ldflags.py --check")
    for f in ("docs/feature-matrix.md", "docs/feature-matrix.html", "src/core/backend_caps_table.h"):
        shutil.copy(REPO / f, OUT / Path(f).name)
    run("git_diff", "git diff --stat")
    with kh.build_heartbeat("unit"):
        rc = run("build_tests", f"cmake --build {BUILD} -j{os.cpu_count()}", timeout=14400)
    run("ctest_unit", f"ctest --test-dir {BUILD} -L unit -j{os.cpu_count()} --timeout 300 --output-on-failure", timeout=7200)
except BaseException:
    res["errors"].append(traceback.format_exc()[-3000:])
finally:
    save()
