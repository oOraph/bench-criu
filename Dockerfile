# base
FROM ubuntu:24.04 AS base

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential git \
    libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler \
    python3-protobuf pkg-config libnftables-dev libcap-dev libbsd-dev \
    libnet-dev libnl-3-dev libgnutls28-dev \
    python3-yaml libaio-dev uuid-dev ca-certificates wget \
    tini net-tools \
    python3 python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint \
      -O /usr/local/bin/cuda-checkpoint && \
      chmod +x /usr/local/bin/cuda-checkpoint

RUN python3 -m venv /venv && . /venv/bin/activate && pip install --no-cache-dir torch

COPY test_app.py /test_app.py

ENV PATH="/venv/bin:$PATH"

ENTRYPOINT ["tini", "--", "sleep", "infinity"]

# criu-dev — upstream baseline (commit just before PR #3021 and #3022 were merged)
FROM base AS criu-dev

RUN git clone https://github.com/ooraph/criu.git /criu && \
    cd /criu && \
    git checkout 4d76d1acd && \
    make -j$(nproc) && make install-criu && \
    mkdir -p /usr/lib/criu && \
    cp plugins/cuda/cuda_plugin.so /usr/lib/criu/

# criu-optimized — baseline + PR #3021 (parallel memfd restore) + PR #3022 (native AIO page reads)
FROM base AS criu-optimized

RUN git clone https://github.com/ooraph/criu.git /criu && \
    cd /criu && \
    git checkout optim1 && \
    make -j$(nproc) && make install-criu && \
    mkdir -p /usr/lib/criu && \
    cp plugins/cuda/cuda_plugin.so /usr/lib/criu/

# criu-fast-cuda-1 (or equivalently cuda-plugin-optim) — baseline + custom CUDA plugin (https://github.com/oOraph/criu/tree/fast_cuda_plugin_final): GPU pages offloaded via O_DIRECT to gpu-pages-*.img
FROM base AS criu-fast-cuda-1

RUN git clone https://github.com/ooraph/criu.git /criu && \
    cd /criu && \
    git checkout fast-cuda-1 && \
    make -j$(nproc) && make install-criu && \
    mkdir -p /usr/lib/criu && \
    cp plugins/cuda/cuda_plugin.so /usr/lib/criu/
