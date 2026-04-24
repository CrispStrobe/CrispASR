FROM ubuntu:22.04 AS build
WORKDIR /app

RUN apt-get update && \
  apt-get install -y build-essential wget cmake git ninja-build \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY . .
ARG CRISPASR_BUILD_JOBS
RUN jobs="${CRISPASR_BUILD_JOBS:-$(nproc)}" && \
  cmake -S . -B build -G Ninja -DWHISPER_BUILD_TESTS=OFF && \
  cmake --build build -j"${jobs}" --target crispasr

FROM ubuntu:22.04 AS runtime
WORKDIR /app

RUN apt-get update && \
  apt-get install -y curl ffmpeg libsdl2-dev wget cmake git \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /app /app
RUN useradd -m -u 1000 crispasr && \
  mkdir -p /cache /models && \
  chown -R crispasr:crispasr /app /cache /models
ENV PATH=/app/build/bin:$PATH
ENV CRISPASR_CACHE_DIR=/cache
USER crispasr
ENTRYPOINT [ "bash", "/app/.devops/run-server.sh" ]
