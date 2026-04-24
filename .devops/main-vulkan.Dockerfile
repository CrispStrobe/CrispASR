FROM ubuntu:24.04 AS build
WORKDIR /app

RUN apt-get update && \
  apt-get install -y build-essential wget cmake git libvulkan-dev glslc \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY . .
RUN cmake -B build -DWHISPER_BUILD_TESTS=OFF -DGGML_VULKAN=1 && \
  cmake --build build -j"$(nproc)" --target crispasr

FROM ubuntu:24.04 AS runtime
WORKDIR /app

RUN apt-get update && \
  apt-get install -y curl ffmpeg libsdl2-dev wget cmake git libvulkan1 mesa-vulkan-drivers \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /app /app
RUN useradd -m -u 1000 crispasr && \
  mkdir -p /cache /models && \
  chown -R crispasr:crispasr /app /cache /models
ENV PATH=/app/build/bin:$PATH
USER crispasr
ENTRYPOINT [ "bash", "-c" ]
