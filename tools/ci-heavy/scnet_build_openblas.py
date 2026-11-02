#!/usr/bin/env python3
"""Build the pinned LP64/pthread OpenBLAS provider used by SCNet CPU CI.

Ubuntu 22.04's 0.3.20 release predates current EPYC CPU identifiers and can
select Barcelona kernels on a Zen 5 runner. Keep the upstream dynamic CPU
dispatch and ordinary F32 arithmetic; do not override the detected core.
This helper changes the native dependency only, not the original Torch model.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REPOSITORY = "https://github.com/OpenMathLib/OpenBLAS.git"
REVISION = "e0166008be8e466242aa76b2ff75ce3f0fbf574a"
TREE = "48a954159e77896d31a458369a8ebe198ad56ca7"
VERSION = "0.3.34"
SOURCE_FILES = ("LICENSE", "README.md", "CMakeLists.txt", "cmake/system.cmake",
                "driver/others/dynamic.c", "cpuid_x86.c")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, help="Reuse an existing clean checkout of the exact pinned commit")
    parser.add_argument("--build-dir", type=Path, help="Reuse a compatible build directory for that checkout")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 16 or sys.platform != "linux" or platform.machine() != "x86_64":
        parser.error("this hosted Linux x86_64 helper requires 1 through 16 build jobs")
    if os.environ.get("OPENBLAS_CORETYPE"):
        parser.error("use automatic CPU dispatch without OPENBLAS_CORETYPE")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "2":
        parser.error("OPENBLAS_NUM_THREADS=2 must be explicit")
    if any(os.environ.get(key) for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT")):
        parser.error("provider controls require an environment without loader overrides")
    work, output = args.work_dir.resolve(), args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    source = args.source_dir.resolve() if args.source_dir else work / "source"
    build = args.build_dir.resolve() if args.build_dir else work / "build"
    prefix = work / "install"
    receipt = {"schema": 1, "complete": False, "acceptance": False, "qualification": False,
               "scope": "Pinned native BLAS dependency build and ABI control, not model qualification",
               "repository": REPOSITORY, "revision": REVISION, "tree": TREE,
               "version": VERSION, "license": "BSD-3-Clause", "commands": [], "errors": []}

    def portable(value):
        return str(value).replace(str(work), "{provider}").replace(str(output), "{evidence}")

    def save():
        (output / "provider.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def run(label, argv):
        log = output / (label + ".log")
        receipt["commands"].append({"label": label, "argv": [portable(x) for x in argv]})
        save()
        with log.open("w") as stream:
            result = subprocess.run([str(x) for x in argv], stdout=stream, stderr=subprocess.STDOUT)
        receipt["commands"][-1].update(exit_code=result.returncode, log=log.name, sha256=sha256(log))
        save()
        if result.returncode:
            raise RuntimeError(f"{label} failed: see {log.name}")
        return log.read_text()

    def git(*argv):
        return subprocess.check_output(["git", "-C", str(source), *argv], text=True).strip()

    try:
        if not (source / ".git").is_dir():
            if source.exists() and any(source.iterdir()):
                raise ValueError("refusing a nonempty source directory without Git identity")
            run("source-init", ["git", "init", source])
            run("source-fetch", ["git", "-C", source, "fetch", "--depth=1", REPOSITORY, REVISION])
            run("source-checkout", ["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"])
        if git("rev-parse", "HEAD") != REVISION or git("rev-parse", "HEAD^{tree}") != TREE:
            raise ValueError("OpenBLAS checkout identity does not match the pinned upstream source")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("OpenBLAS tracked source is modified")
        before = {name: sha256(source / name) for name in SOURCE_FILES}
        receipt["source_hashes"] = before
        configuration = [
            "cmake", "-S", source, "-B", build, "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_STATIC_LIBS=OFF", "-DBUILD_WITHOUT_LAPACK=ON",
            "-DBUILD_WITHOUT_LAPACKE=ON", "-DBUILD_WITHOUT_CBLAS=OFF",
            "-DBUILD_TESTING=OFF", "-DBUILD_BENCHMARKS=OFF", "-DDYNAMIC_ARCH=ON",
            "-DDYNAMIC_OLDER=OFF", "-DTARGET=CORE2", "-DUSE_THREAD=ON",
            "-DUSE_OPENMP=OFF", "-DINTERFACE64=OFF", "-DNUM_THREADS=64",
            "-DNO_AFFINITY=ON", "-DNO_WARMUP=ON",
        ]
        run("configure", configuration)
        run("build", ["cmake", "--build", build, "--parallel", args.jobs, "--target", "openblas_shared"])
        run("install", ["cmake", "--install", build])
        library = prefix / "lib/libopenblas.so"
        header = prefix / "include/openblas/cblas.h"
        if not library.is_file() or not header.is_file():
            raise ValueError("expected installed shared provider/header is missing")
        control_source = output / "abi-control.c"
        control_source.write_text('''#include <cblas.h>
#include <stdio.h>
#include <string.h>
_Static_assert(sizeof(blasint) == 4, "SCNet requires LP64 BLAS integers");
int main(void) {
    if (strncmp(openblas_get_config(), "OpenBLAS 0.3.34 ", 16) != 0 ||
        openblas_get_parallel() != 1 || openblas_get_num_threads() != 2) return 2;
    const float a[4] = {1, 2, 3, 4}, b[4] = {5, 6, 7, 8};
    const float expected[4] = {19, 22, 43, 50};
    float result[4] = {0, 0, 0, 0};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                2, 2, 2, 1, a, 2, b, 2, 0, result, 2);
    for (int i = 0; i < 4; ++i) if (result[i] != expected[i]) return 1;
    puts("LP64 CBLAS SGEMM control passed");
    return 0;
}
''')
        control = work / "abi-control"
        run("abi-build", ["cc", "-std=c11", control_source, f"-I{header.parent}",
                          library, f"-Wl,-rpath,{prefix / 'lib'}", "-o", control])
        run("abi-run", [control])
        provider = ctypes.CDLL(str(library))
        runtime = {}
        for name in ("openblas_get_config", "openblas_get_corename", "openblas_get_parallel",
                     "openblas_get_num_threads"):
            query = getattr(provider, name)
            query.argtypes = []
            query.restype = ctypes.c_char_p if name in ("openblas_get_config", "openblas_get_corename") else ctypes.c_int
            value = query()
            runtime[name] = value.decode() if isinstance(value, bytes) else value
        if (not runtime["openblas_get_config"].startswith("OpenBLAS " + VERSION + " ") or
                "USE64BITINT" in runtime["openblas_get_config"] or
                "DYNAMIC_ARCH" not in runtime["openblas_get_config"] or
                runtime["openblas_get_parallel"] != 1 or runtime["openblas_get_num_threads"] != 2):
            raise ValueError("loaded provider does not meet the version/dynamic-dispatch/LP64/pthread/two-thread contract")
        receipt["runtime"] = runtime
        receipt["source_unchanged"] = (before == {name: sha256(source / name) for name in SOURCE_FILES}
                                       and not git("status", "--porcelain", "--untracked-files=no"))
        if not receipt["source_unchanged"]:
            raise ValueError("upstream source changed during the dependency build")
        shutil.copyfile(source / "LICENSE", output / "OpenBLAS-LICENSE.txt")
        shutil.copyfile(build / "CMakeCache.txt", output / "CMakeCache.txt")
        receipt["installed_files"] = {str(path.relative_to(prefix)): {"bytes": path.stat().st_size,
                                       "sha256": sha256(path)} for path in sorted(prefix.rglob("*")) if path.is_file()}
        receipt["complete"] = True
        save()
        print(json.dumps({"prefix": str(prefix), "runtime": runtime, "receipt": str(output / "provider.json")}))
    except Exception as error:
        receipt["errors"].append(str(error))
        save()
        raise


if __name__ == "__main__":
    main()
