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

# LIBRARY_PATH is read by ld at LINK time only (not at runtime). Without
# this, the build fails 'libggml-cuda.so undefined reference to
# cuGetErrorString' because CMake's CUDAToolkit module doesn't always
# locate the libcuda stub in CUDA 13's lib64/stubs layout.
#
# Critically NOT LD_LIBRARY_PATH: that's the runtime path, and force-
# prepending the compat libs there shadows the host libcuda.so.1 that
# the nvidia container runtime mounts → 'unsupported driver/cuda combo'
# on hosts whose driver is newer than the compat libs (#31).
ENV LIBRARY_PATH=/usr/local/cuda-13.0/lib64/stubs:/usr/local/cuda-13.0/compat:$LIBRARY_PATH

COPY . .
ARG CRISPASR_BUILD_JOBS
RUN jobs="${CRISPASR_BUILD_JOBS:-$(nproc)}" && \
    cmake -S . -B build -G Ninja -DCRISPASR_BUILD_TESTS=OFF -DGGML_CUDA=1 \
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
  apt-get install -y curl ffmpeg wget cmake git pciutils \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# The real second half of the #31 fix.
#
# nvidia/cuda:*-runtime-* installs `cuda-compat-<ver>` which drops
# /etc/ld.so.conf.d/000_cuda_compat.conf pointing at /usr/local/cuda/compat.
# That dir is processed first by ldconfig (the `000_` prefix is deliberate),
# so the cache resolves libcuda.so.1 → the compat copy that ships with the
# image. When the nvidia container runtime later mounts the host's libcuda
# at --gpus startup, the cache entry is *still* the compat one if we don't
# rebuild ldconfig. The compat libcuda is older than the host kernel
# driver on R580+ hosts → 'unsupported display driver / cuda driver
# combination' (#31, even after the LD_LIBRARY_PATH-only fix in 7587ad2).
#
# Removing the .conf entry and rerunning ldconfig means the cache is
# *empty* of libcuda at image build time. nvidia-container-runtime then
# adds the host libcuda when --gpus is passed and runs ldconfig again,
# which is exactly the flow that works on the bare nvidia/cuda image.
#
# Compat libs stay on disk at /usr/local/cuda/compat/. Users with old host
# drivers can still opt back in at runtime via:
#   docker run -e CRISPASR_USE_CUDA_COMPAT=1 ...
# which run-server.sh honours by prepending the dir to LD_LIBRARY_PATH.
RUN rm -f /etc/ld.so.conf.d/000_cuda_compat.conf /etc/ld.so.conf.d/cuda-compat.conf && ldconfig

# Image labels + /app/build-info.txt so users (and run-server.sh) can
# answer "which tag is this?" without needing the source tree (#31 user
# explicitly asked for this).
ARG GIT_SHA=unknown
ARG GIT_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="crispasr"
LABEL org.opencontainers.image.source="https://github.com/CrispStrobe/CrispASR"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.ref.name="${GIT_REF}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.description="crispasr unified ASR — CUDA 13.0 build (driver R580+ recommended)"
COPY --from=build /app /app
# build-info.txt comes AFTER the COPY so it isn't clobbered. run-server.sh
# cats it on every startup so the first 5 lines of any user log identify
# the build (#31).
RUN printf 'image=main-cuda\ncuda_version=13.0\ngit_sha=%s\ngit_ref=%s\nbuild_date=%s\n' \
        "${GIT_SHA}" "${GIT_REF}" "${BUILD_DATE}" > /app/build-info.txt
RUN useradd -m -u 1000 crispasr && \
  mkdir -p /cache /models && \
  chown -R crispasr:crispasr /app /cache /models
RUN du -sh /app/*
RUN find /app -type f -size +100M
ENV PATH=/app/build/bin:$PATH
ENV CRISPASR_CACHE_DIR=/cache
USER crispasr
ENTRYPOINT [ "bash", "/app/.devops/run-server.sh" ]
