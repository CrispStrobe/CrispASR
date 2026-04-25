ARG UBUNTU_VERSION=22.04
# CUDA 12.4: lower NVIDIA driver floor (R510+) than the main-cuda image
# (which is on CUDA 13.0 → R535+). Use this tag if your host driver is
# older than R535 — common on RHEL 7/8, older Ubuntu LTS, and most
# enterprise/cluster setups that don't update drivers frequently.
#
# Trade-off: caps at sm_90 (Hopper / RTX 40-series). RTX 50xx (sm_120,
# Blackwell consumer) is NOT supported by CUDA 12.4 — those users
# need the main-cuda tag and a recent driver.
ARG CUDA_VERSION=12.4.0
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build
WORKDIR /app

ARG CUDA_DOCKER_ARCH=all
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}

RUN apt-get update && \
    apt-get install -y build-essential libsdl2-dev wget cmake git ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Note: do NOT preload /usr/local/cuda-12.4/compat onto LD_LIBRARY_PATH —
# see main-cuda.Dockerfile for the explanation. Hosts whose driver is
# NEWER than the compat libs hit 'unsupported driver/cuda combo' (#31).
# The compat libs remain available in the image for callers who need
# them; we just don't force them on top of LD_LIBRARY_PATH.

COPY . .
ARG CRISPASR_BUILD_JOBS
RUN jobs="${CRISPASR_BUILD_JOBS:-$(nproc)}" && \
    cmake -S . -B build -G Ninja -DWHISPER_BUILD_TESTS=OFF -DGGML_CUDA=1 \
        -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;90-virtual" && \
    cmake --build build -j"${jobs}" --target crispasr

RUN find /app/build -name "*.o" -delete && \
    find /app/build -name "*.a" -delete && \
    rm -rf /app/build/CMakeFiles && \
    rm -rf /app/build/cmake_install.cmake && \
    rm -rf /app/build/_deps

FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime
# See note in the build stage about not prepending compat to LD_LIBRARY_PATH.
WORKDIR /app

RUN apt-get update && \
  apt-get install -y curl ffmpeg wget cmake git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /app /app
RUN (id -u crispasr 2>/dev/null || \
     useradd -m -u 1000 crispasr 2>/dev/null || \
     useradd -m crispasr) && \
    mkdir -p /cache /models && \
    chown -R crispasr:crispasr /app /cache /models
ENV PATH=/app/build/bin:$PATH
ENV CRISPASR_CACHE_DIR=/cache
USER crispasr
ENTRYPOINT [ "bash", "/app/.devops/run-server.sh" ]
