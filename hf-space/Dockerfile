FROM ubuntu:22.04 AS build
WORKDIR /src/CrispASR

RUN apt-get update && \
  apt-get install -y build-essential cmake git curl && \
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ARG CRISPASR_REPO=https://github.com/CrispStrobe/CrispASR
ARG CRISPASR_REF=main

RUN git clone --depth 1 --branch "${CRISPASR_REF}" "${CRISPASR_REPO}" /src/CrispASR
RUN cmake -B build -DWHISPER_BUILD_TESTS=OFF && \
  cmake --build build -j"$(nproc)" --target whisper-cli

FROM ubuntu:22.04 AS runtime
WORKDIR /space

RUN apt-get update && \
  apt-get install -y bash curl ffmpeg python3 python3-pip && \
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /src/CrispASR/build /opt/crispasr/build
COPY hf-space/requirements.txt /space/requirements.txt
RUN pip3 install --no-cache-dir -r /space/requirements.txt

COPY hf-space/app.py hf-space/start.sh /space/
RUN chmod +x /space/start.sh && \
  useradd -m -u 1000 app && \
  mkdir -p /models /cache && \
  chown -R app:app /space /opt/crispasr /models /cache

ENV PATH=/opt/crispasr/build/bin:$PATH
ENV CRISPASR_SERVER_URL=http://127.0.0.1:8080
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

USER app
EXPOSE 7860
EXPOSE 8080
ENTRYPOINT ["/space/start.sh"]
