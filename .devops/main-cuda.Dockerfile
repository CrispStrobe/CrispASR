ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=13.0.0
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Target the CUDA runtime image
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build
WORKDIR /app

# Unless otherwise specified, we make a fat build.
ARG CUDA_DOCKER_ARCH=all
# Set nvcc architecture
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}

RUN apt-get update && \
    apt-get install -y build-essential libsdl2-dev wget cmake git ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Note: do NOT preload /usr/local/cuda-13.0/compat onto LD_LIBRARY_PATH.
# The forward-compat libcuda is for OLDER host drivers loading newer
# runtimes; force-prepending it shadows the host libcuda.so.1 that the
# nvidia container runtime mounts, which breaks initialization on hosts
# whose driver is NEWER than the compat libs were built against — see
# issue #31 (driver R595 + this image → 'unsupported driver/cuda combo').

COPY . .
ARG CRISPASR_BUILD_JOBS
RUN jobs="${CRISPASR_BUILD_JOBS:-$(nproc)}" && \
    cmake -S . -B build -G Ninja -DWHISPER_BUILD_TESTS=OFF -DGGML_CUDA=1 \
        -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;120-real;120-virtual" && \
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
RUN useradd -m -u 1000 crispasr && \
  mkdir -p /cache /models && \
  chown -R crispasr:crispasr /app /cache /models
RUN du -sh /app/*
RUN find /app -type f -size +100M
ENV PATH=/app/build/bin:$PATH
ENV CRISPASR_CACHE_DIR=/cache
USER crispasr
ENTRYPOINT [ "bash", "/app/.devops/run-server.sh" ]
