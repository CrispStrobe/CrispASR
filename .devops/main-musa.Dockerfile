ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
# Both public SDK 4.2 and 4.3 currently hit an upstream WKV compiler error.
# The release recovery records that blocker; do not silently alter kernels.
ARG MUSA_VERSION=rc4.2.0
# Target the MUSA build image
ARG BASE_MUSA_DEV_CONTAINER=mthreads/musa:${MUSA_VERSION}-devel-ubuntu${UBUNTU_VERSION}-amd64
# Target the MUSA runtime image
ARG BASE_MUSA_RUN_CONTAINER=mthreads/musa:${MUSA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}-amd64

FROM ${BASE_MUSA_DEV_CONTAINER} AS build
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential libsdl2-dev wget cmake git ninja-build ccache && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

COPY . .
# The SDK image omits CUB, but the current ggml MUSA backend uses it. Supply
# Moore Threads' own header-only port, pinned to muThrust's dependency revision.
# Preserve both its MIT notice and the upstream CUB BSD notices in the image.
RUN git clone --no-checkout https://github.com/MooreThreads/muAlg.git /opt/mualg && \
    git -C /opt/mualg checkout --detach e189e81f545694cfc79db7d216a049637235d176 && \
    test "$(git -C /opt/mualg rev-parse HEAD)" = e189e81f545694cfc79db7d216a049637235d176 && \
    printf '%s  %s\n' 2ac06c9d4c058c9159e16ee8a83b41e3a403cf690fde725e664c2595e5f425a8 /opt/mualg/LICENSE.TXT | sha256sum -c - && \
    rm -rf /usr/local/musa/include/cub && \
    cp -a /opt/mualg/cub /usr/local/musa/include/cub && \
    test -f /usr/local/musa/include/cub/cub.cuh
ARG CRISPASR_BUILD_JOBS=2
# Enable muBLAS
# NOTE (#261): GGML_NATIVE=OFF does NOT restrict the ISA to the flags listed —
# ggml sets INS_ENB=ON in that case, so BMI2/SSE42/AVX default ON anyway. The
# real baseline is Haswell-class (AVX2+FMA+F16C+BMI2); spelled out explicitly
# so it's visible. See .devops/main-cuda.Dockerfile for the full explanation.
RUN cat /proc/loadavg && grep MemAvailable /proc/meminfo && \
    cmake -B build -G Ninja -DCRISPASR_BUILD_TESTS=OFF -DGGML_MUSA=1 \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    -DGGML_BMI2=ON -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX512=OFF && \
    cmake --build build -j"${CRISPASR_BUILD_JOBS}" --target crispasr-cli

RUN find /app/build -name "*.o" -delete && \
    find /app/build -name "*.a" -delete && \
    rm -rf /app/build/CMakeFiles && \
    rm -rf /app/build/cmake_install.cmake && \
    rm -rf /app/build/_deps

FROM ${BASE_MUSA_RUN_CONTAINER} AS runtime
WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl passwd ffmpeg wget cmake git tini libgomp1 libsdl2-2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

# Keep the shared-library tree and its relative loader paths as well as bin/.
COPY --from=build /app/build /app/build
COPY --from=build /app/samples /app/samples
COPY --from=build /app/.devops/run-server.sh /app/.devops/run-server.sh
COPY --from=build /app/LICENSE /app/LICENSE
COPY --from=build /app/THIRD_PARTY_NOTICES.txt /app/THIRD_PARTY_NOTICES.txt
COPY --from=build /opt/mualg/LICENSE.TXT /app/THIRD_PARTY_LICENSES/muAlg-LICENSE.txt
RUN test -x /app/build/bin/crispasr && \
    bash -n /app/.devops/run-server.sh && \
    test -s /app/LICENSE && test -s /app/THIRD_PARTY_NOTICES.txt && \
    printf '%s  %s\n' 2ac06c9d4c058c9159e16ee8a83b41e3a403cf690fde725e664c2595e5f425a8 /app/THIRD_PARTY_LICENSES/muAlg-LICENSE.txt | sha256sum -c -
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
