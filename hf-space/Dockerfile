FROM ubuntu:22.04 AS build
WORKDIR /src

# archive.ubuntu.com flakes intermittently from CI runners; retries +
# longer timeouts let transient connection failures resolve themselves
# rather than failing the whole build.
RUN printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
      > /etc/apt/apt.conf.d/80-retries && \
  apt-get update && \
  apt-get install -y --fix-missing build-essential cmake git curl ninja-build && \
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ARG CRISPASR_REPO=https://github.com/CrispStrobe/CrispASR
ARG CRISPASR_REF=main

RUN git clone --depth 1 "${CRISPASR_REPO}" /src/CrispASR && \
  git -C /src/CrispASR fetch --depth 1 origin "${CRISPASR_REF}" && \
  git -C /src/CrispASR checkout --detach FETCH_HEAD
WORKDIR /src/CrispASR
ARG CRISPASR_BUILD_JOBS
RUN jobs="${CRISPASR_BUILD_JOBS:-$(nproc)}" && \
  cmake -S . -B build -G Ninja -DCRISPASR_BUILD_TESTS=OFF && \
  cmake --build build -j"${jobs}" --target crispasr

FROM ubuntu:22.04 AS runtime
WORKDIR /space

RUN printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
      > /etc/apt/apt.conf.d/80-retries && \
  apt-get update && \
  apt-get install -y --fix-missing bash curl ffmpeg python3 python3-pip && \
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /src/CrispASR/build /opt/crispasr/build
COPY hf-space/requirements.txt /space/requirements.txt
RUN pip3 install --no-cache-dir -r /space/requirements.txt
RUN find /opt/crispasr/build -type f -name 'lib*.so*' -exec cp -a {} /usr/local/lib/ \; && ldconfig

COPY hf-space/app.py hf-space/start.sh /space/
RUN chmod +x /space/start.sh && \
  useradd -m -u 1000 app && \
  mkdir -p /models /cache && \
  chown -R app:app /space /opt/crispasr /models /cache

ENV PATH=/opt/crispasr/build/bin:$PATH
ENV CRISPASR_SERVER_URL=http://127.0.0.1:8080
ENV CRISPASR_CACHE_DIR=/cache
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

USER app
EXPOSE 7860
EXPOSE 8080
ENTRYPOINT ["/space/start.sh"]
