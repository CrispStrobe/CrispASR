ARG ONEAPI_VERSION=2025.1.1-0-devel-ubuntu24.04

FROM intel/oneapi-basekit:$ONEAPI_VERSION AS build
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential libsdl2-dev wget cmake git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY . .
# Enable SYCL
ARG GGML_SYCL_F16=OFF
RUN if [ "${GGML_SYCL_F16}" = "ON" ]; then \
        echo "GGML_SYCL_F16 is set" \
        && export OPT_SYCL_F16="-DGGML_SYCL_F16=ON"; \
    fi && \
    cmake -B build -DWHISPER_BUILD_TESTS=OFF -DGGML_SYCL=1 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ${OPT_SYCL_F16} && \
    cmake --build build -j"$(nproc)" --target crispasr

FROM intel/oneapi-basekit:$ONEAPI_VERSION AS runtime
WORKDIR /app

RUN apt-get update && \
  apt-get install -y curl ffmpeg libsdl2-dev wget cmake git \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /app /app
RUN useradd -m -u 1000 crispasr && \
  mkdir -p /cache /models && \
  chown -R crispasr:crispasr /app /cache /models
ENV PATH=/app/build/bin:$PATH
USER crispasr
ENTRYPOINT [ "bash", "-c" ]
