# S5000 / architecture 31 only. SDK 4.2/4.3 failed WKV code generation;
# this exact SDK 5.2 + target configuration compiled the unchanged release.
# Build on a Moore Threads platform in accordance with the vendor SDK terms.
# Automatic registry publication is disabled pending runtime/license review.
ARG MUSA_SDK_IMAGE=registry.mthreads.com/mcconline/musa_sdk:5.2.0-devel-ubuntu22.04-s5000@sha256:ffb9f46ba25a14d5978b21fe59a7cb804c534a01e5b4a134112d4962a3e3e657
FROM ${MUSA_SDK_IMAGE} AS sdk

FROM sdk AS build
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential libsdl2-dev libssl-dev wget cmake git ninja-build ccache \
      musa-mualg-5-2=1.17.2 musa-muthrust-5-2=1.17.2 libmthreads-compute=5.2.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

COPY . .
# Use the SDK's official CUB/Thrust packages; do not replace SDK headers with
# an unrelated checkout. Retain installed notice metadata and exact versions.
RUN mkdir -p /app/musa-package-metadata && \
    dpkg-query -W > /app/musa-package-metadata/packages.txt && \
    cp /usr/share/doc/musa-mualg-5-2/copyright /app/musa-package-metadata/muAlg-copyright && \
    cp /usr/share/doc/musa-muthrust-5-2/copyright /app/musa-package-metadata/muThrust-copyright
ARG CRISPASR_BUILD_JOBS=2
ENV CCACHE_MAXSIZE=2G
# Enable muBLAS
# NOTE (#261): GGML_NATIVE=OFF does NOT restrict the ISA to the flags listed —
# ggml sets INS_ENB=ON in that case, so BMI2/SSE42/AVX default ON anyway. The
# real baseline is Haswell-class (AVX2+FMA+F16C+BMI2); spelled out explicitly
# so it's visible. See .devops/main-cuda.Dockerfile for the full explanation.
RUN cat /proc/loadavg && grep MemAvailable /proc/meminfo && \
    awk -v limit="$(nproc)" '$1 > limit {exit 1}' /proc/loadavg && \
    test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 4194304 && \
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_BUILD_TESTS=OFF \
    -DGGML_MUSA=ON -DMUSA_ARCHITECTURES=31 \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    -DGGML_BMI2=ON -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX512=OFF && \
    cmake --build build -j"${CRISPASR_BUILD_JOBS}" --target crispasr-cli

RUN find /app/build -name "*.o" -delete && \
    find /app/build -name "*.a" -delete && \
    rm -rf /app/build/CMakeFiles && \
    rm -rf /app/build/cmake_install.cmake && \
    rm -rf /app/build/_deps

# Keep the same SDK base and its libraries/notices until a separately verified
# runtime-only packaging profile is available. No cross-version runtime mix.
FROM sdk AS runtime
LABEL org.crispasr.musa.target="S5000 / architecture 31" \
      org.crispasr.musa.validation="compile-candidate; hardware-validation-pending"
WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl passwd ffmpeg wget cmake git tini libgomp1 libsdl2-2.0-0 libmthreads-compute=5.2.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

# Keep the shared-library tree and its relative loader paths as well as bin/.
COPY --from=build /app/build /app/build
COPY --from=build /app/samples /app/samples
COPY --from=build /app/.devops/run-server.sh /app/.devops/run-server.sh
COPY --from=build /app/LICENSE /app/LICENSE
COPY --from=build /app/THIRD_PARTY_NOTICES.txt /app/THIRD_PARTY_NOTICES.txt
COPY --from=build /app/musa-package-metadata /app/musa-package-metadata
RUN test -x /app/build/bin/crispasr && \
    bash -n /app/.devops/run-server.sh && \
    test -s /app/LICENSE && test -s /app/THIRD_PARTY_NOTICES.txt && \
    test -s /app/musa-package-metadata/muAlg-copyright && \
    test -s /app/musa-package-metadata/muThrust-copyright
# /models is a volume mount point — actual models (and the convert-*.py
# tools) are excluded by .dockerignore. Don't COPY them from the build
# stage; the user mounts their model directory at /models at runtime.
RUN (id -u crispasr 2>/dev/null || \
     useradd -m -u 1000 crispasr 2>/dev/null || \
     useradd -m crispasr) && \
    mkdir -p /cache /models && \
    chown -R crispasr:crispasr /app /cache /models

ENV PATH=/app/build/bin:$PATH
ENV CRISPASR_CACHE_DIR=/cache
USER crispasr
ENTRYPOINT [ "tini", "--", "bash", "/app/.devops/run-server.sh" ]
