#!/usr/bin/env bash
# Compile experiment on an authorized MUSA host. No model loads or publishing.
set -euo pipefail
mkdir -p /out/licenses
exec > >(tee /out/probe.log) 2>&1
trap 'rc=$?; printf "exit_code=%s\n" "$rc" > /out/status.txt' EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y build-essential git cmake ninja-build ccache python3 libssl-dev libsdl2-dev \
    musa-mualg-5-2 musa-muthrust-5-2 libmthreads-compute
dpkg-query -W -f='${binary:Package}\t${Version}\t${Architecture}\n' > /out/packages.txt
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import shutil

records = []
for line in Path('/out/packages.txt').read_text().splitlines():
    package, version, architecture = line.split('\t')
    if not any(term in package.lower() for term in ('musa', 'muthrust', 'mualg', 'mthreads')):
        continue
    source = Path('/usr/share/doc') / package.split(':')[0] / 'copyright'
    row = dict(package=package, version=version, architecture=architecture,
               copyright_present=source.is_file() and bool(source.stat().st_size))
    records.append(row)
    if not row['copyright_present']:
        # This is an unpublished compiler experiment. Missing notices remain
        # explicit evidence for a later packaging/license review.
        continue
    dest = Path('/out/licenses') / (package.replace(':', '_') + '-copyright')
    shutil.copyfile(source, dest)
    row['copyright_sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
assert records
Path('/out/sdk-packages.json').write_text(json.dumps(records, indent=2) + '\n')
PY

git config --global --add safe.directory /src
git config --global --add safe.directory /src/ggml
git config --global --add safe.directory /src/third_party/c2pa-audio
test "$(git -C /src rev-parse HEAD)" = e58bb48b52cf1fe957c19586297c83a2ed0d9fe4
export PATH=/usr/local/musa/bin:/opt/musa/bin:$PATH
export CCACHE_DIR=/out/ccache
export CCACHE_MAXSIZE=2G
if [ -x /usr/local/musa/bin/clang++ ]; then
    /usr/local/musa/bin/clang++ --version | tee /out/compiler.txt
else
    /opt/musa/bin/clang++ --version | tee /out/compiler.txt
fi

resources() {
    python3 - <<'PY'
import os
from pathlib import Path
available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                 if line.startswith('MemAvailable:'))
print('Before compile: load=', os.getloadavg(), 'available_KiB=', available, flush=True)
assert os.getloadavg()[0] < 2 * (os.cpu_count() or 1), 'Runner overloaded'
assert available >= 4 * 1024**2, 'Less than 4 GiB available'
PY
}

resources
cmake -S /src -B /out/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCRISPASR_BUILD_TESTS=OFF -DGGML_MUSA=ON -DMUSA_ARCHITECTURES=31 \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    -DGGML_BMI2=ON -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX512=OFF \
    2>&1 | tee /out/configure.log
python3 - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path('/out/build/compile_commands.json').read_text())
rows = [row for row in rows if row['file'].endswith('/wkv.cu')]
assert len(rows) == 1, rows
Path('/out/wkv-command.json').write_text(json.dumps(rows[0], indent=2) + '\n')
PY
resources
ninja -C /out/build -j2 -v \
    ggml/src/ggml-musa/CMakeFiles/ggml-musa.dir/__/ggml-cuda/wkv.cu.o \
    2>&1 | tee /out/wkv.log
test -s /out/build/ggml/src/ggml-musa/CMakeFiles/ggml-musa.dir/__/ggml-cuda/wkv.cu.o
printf 'passed\n' > /out/wkv-result.txt
resources
cmake --build /out/build --parallel 2 --target crispasr-cli 2>&1 | tee /out/full-build.log
test -x /out/build/bin/crispasr
# CMake regenerates source-tree package version metadata; compiled sources must
# still match the qualified tag exactly.
git -C /src diff --exit-code -- src include examples > /out/source-diff.txt
git -C /src/ggml diff --exit-code -- src include >> /out/source-diff.txt
find /out/build -name 'libggml-musa.so*' -type f -exec sha256sum {} + > /out/musa-libraries.txt
test -s /out/musa-libraries.txt
sha256sum /out/build/bin/crispasr > /out/cli-sha256.txt
printf 'compile-only; architecture=31; hardware_inference_qualified=false; published=false\n' > /out/acceptance.txt
