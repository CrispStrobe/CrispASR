#!/usr/bin/env python3
"""Build the Index-Echo CLI, shared ABI and diff on CI; execute on a real GPU."""
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
out = repo / "index-echo-cuda-out"
out.mkdir(exist_ok=True)
subprocess.run(["uptime"], check=True)
subprocess.run(["free", "-h"], check=True)
build = repo / "index-echo-cuda-build"
cuda_root = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))
driver_stubs = list(cuda_root.rglob("stubs/libcuda.so"))
if not driver_stubs:
    raise RuntimeError("CUDA toolkit driver stub is required for the GPU-less CI linker")
link_stubs = build / "link-stubs"
link_stubs.mkdir(parents=True, exist_ok=True)
(link_stubs / "libcuda.so.1").symlink_to(driver_stubs[0])
with (out / "build.log").open("w") as log:
    for command in (
        ["cmake", "-G", "Ninja", "-S", str(repo), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON", "-DGGML_CUDA=ON",
         "-DCMAKE_CUDA_ARCHITECTURES=75", "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF",
         f"-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{link_stubs}",
         "-DCRISPASR_BUILD_TESTS=OFF", "-DCRISPASR_BUILD_SERVER=OFF",
         "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
         "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"],
        ["cmake", "--build", str(build), "--target", "crispasr-cli", "crispasr-lib", "crispasr-diff", "crispasr-quantize", "-j2"],
    ):
        print(command, flush=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if process.wait() != 0:
            raise RuntimeError(f"build command failed ({process.returncode})")

bundle = out / "bundle"
bundle.mkdir(exist_ok=True)
for executable in ["crispasr", "crispasr-diff", "crispasr-quantize"]:
    shutil.copy2(build / "bin" / executable, bundle)
for source in build.rglob("*.so*"):
    if source.is_file() and not source.name.startswith("libcuda.so"):
        soname = subprocess.check_output(["patchelf", "--print-soname", str(source)], text=True).strip()
        shutil.copy2(source, bundle / (soname or source.name))
# Bundle user-space CUDA runtime dependencies, never a driver/compatibility shim.
for library in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"):
    candidates = list(cuda_root.rglob(library))
    if not candidates:
        raise RuntimeError(f"missing CUDA runtime {library}")
    shutil.copy2(candidates[0], bundle / library)
for binary in bundle.iterdir():
    subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(binary)], check=True)
sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
(bundle / "provenance.json").write_text(json.dumps({"sha": sha, "cuda": "12.4.1", "architectures": [75], "ggml_sha": subprocess.check_output(["git", "-C", str(repo / "ggml"), "rev-parse", "HEAD"], text=True).strip()}, indent=2))
archive = out / "index-echo-cuda-validation.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    tar.add(bundle, arcname="bundle")
(out / "sha256.txt").write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "\n")
shutil.rmtree(bundle)
print(f"Packaged {archive.stat().st_size} bytes at {sha}", flush=True)
