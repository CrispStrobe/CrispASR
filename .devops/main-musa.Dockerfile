ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG MUSA_VERSION=rc4.2.0
# Target the MUSA build image
ARG BASE_MUSA_DEV_CONTAINER=mthreads/musa:${MUSA_VERSION}-devel-ubuntu${UBUNTU_VERSION}-amd64
# Target the MUSA runtime image
ARG BASE_MUSA_RUN_CONTAINER=mthreads/musa:${MUSA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}-amd64

FROM ${BASE_MUSA_DEV_CONTAINER} AS build
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential libsdl2-dev wget cmake git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

COPY . .
# Enable muBLAS
RUN cmake -B build -DWHISPER_BUILD_TESTS=OFF -DGGML_MUSA=1 && \
    cmake --build build -j"$(nproc)" --target crispasr

RUN find /app/build -name "*.o" -delete && \
    find /app/build -name "*.a" -delete && \
    rm -rf /app/build/CMakeFiles && \
    rm -rf /app/build/cmake_install.cmake && \
    rm -rf /app/build/_deps

FROM ${BASE_MUSA_RUN_CONTAINER} AS runtime
WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl ffmpeg wget cmake git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

COPY --from=build /app/build/bin /app/build/bin
COPY --from=build /app/samples /app/samples
COPY --from=build /app/models /app/models
RUN useradd -m -u 1000 crispasr && \
    mkdir -p /cache /models && \
    chown -R crispasr:crispasr /app /cache /models

ENV PATH=/app/build/bin:$PATH
USER crispasr
ENTRYPOINT [ "bash", "-c" ]
