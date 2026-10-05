# criu-base
FROM ubuntu:24.04 AS base

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential git \
    libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler \
    python3-protobuf pkg-config libnftables-dev libcap-dev libbsd-dev \
    libnet-dev libnl-3-dev libgnutls28-dev \
    python3-yaml libaio-dev liblz4-dev uuid-dev ca-certificates wget \
    tini net-tools \
    python3 python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint \
      -O /usr/local/bin/cuda-checkpoint && \
      chmod +x /usr/local/bin/cuda-checkpoint

ENTRYPOINT ["tini", "--", "sleep", "infinity"]

# bench-base
FROM base AS bench-base

RUN python3 -m venv /venv && . /venv/bin/activate && pip install --no-cache-dir torch

COPY test_app.py /test_app.py

ENV PATH="/venv/bin:$PATH"

# criu-dev — upstream baseline (commit just before PR #3021 and #3022 were merged)
FROM bench-base AS criu-dev

RUN git clone https://github.com/ooraph/criu.git /criu && \
    cd /criu && \
    git checkout 4d76d1acd && \
    make -j$(nproc) && make install-criu && \
    mkdir -p /usr/lib/criu && \
    cp plugins/cuda/cuda_plugin.so /usr/lib/criu/

# criu-optimized — baseline + PR #3021 (parallel memfd restore) + PR #3022 (native AIO page reads)
FROM bench-base AS criu-optimized

RUN git clone https://github.com/ooraph/criu.git /criu && \
    cd /criu && \
    git checkout optim1 && \
    make -j$(nproc) && make install-criu && \
    mkdir -p /usr/lib/criu && \
    cp plugins/cuda/cuda_plugin.so /usr/lib/criu/

# criu-fast-cuda-1 (or equivalently cuda-plugin-optim) — baseline + custom CUDA plugin (https://github.com/oOraph/criu/tree/fast_cuda_plugin_final): GPU pages offloaded via O_DIRECT to gpu-pages-*.img
FROM bench-base AS criu-fast-cuda-1

RUN git clone https://github.com/ooraph/criu.git /criu && \
    cd /criu && \
    git checkout fast-cuda-1 && \
    make -j$(nproc) && make install-criu && \
    mkdir -p /usr/lib/criu && \
    cp plugins/cuda/cuda_plugin.so /usr/lib/criu/

# criu-v42-ours — CRIU v4.2 + custom CUDA plugin (pre-October state of the fork),
# tag v4.2-cuda-plugin-optim == branch fast_cuda_plugin_final
FROM bench-base AS criu-v42-ours

RUN git clone https://github.com/ooraph/criu.git /criu && \
    cd /criu && \
    git checkout v4.2-cuda-plugin-optim && \
    make -j$(nproc) && make install-criu && \
    mkdir -p /usr/lib/criu && \
    cp plugins/cuda/cuda_plugin.so /usr/lib/criu/

# criu-upstream-head — upstream criu-dev pinned at 2026-10-01 head: libcuda driver-api
# backend + device-map, PR #3021/#3022 (asyncd memfd, AIO), --image-io-mode (#3066), LZ4.
ARG UPSTREAM_HEAD=4485a86da237
FROM bench-base AS criu-upstream-head
ARG UPSTREAM_HEAD

RUN git clone https://github.com/checkpoint-restore/criu.git /criu && \
    cd /criu && \
    git checkout ${UPSTREAM_HEAD} && \
    make -j$(nproc) && make install-criu && \
    mkdir -p /usr/lib/criu && \
    cp plugins/cuda/cuda_plugin.so /usr/lib/criu/

# criu-local — build from a local source tree (rsync your criu checkout to ./criu-src,
# e.g. `rsync -a --exclude .git ~/workspace_idea/hf/criu/ criu-src/`) to validate a
# branch without pushing it. ./criu-src is git-ignored.
FROM bench-base AS criu-local

COPY criu-src /criu
RUN cd /criu && make clean >/dev/null 2>&1; cd /criu && \
    make -j$(nproc) && make install-criu && \
    mkdir -p /usr/lib/criu && \
    cp plugins/cuda/cuda_plugin.so /usr/lib/criu/ && \
    { [ -f plugins/cuda/cuda-offload ] && cp plugins/cuda/cuda-offload /usr/local/bin/ || true; }

# criu-ref — build from any pushed ref of a CRIU repo, e.g.
#   docker build --target criu-ref -t criu-head-parallel \
#       --build-arg CRIU_REPO=https://github.com/oOraph/criu.git --build-arg CRIU_REF=fast_cuda_plugin_on_head_parallel .
FROM bench-base AS criu-ref
ARG CRIU_REPO=https://github.com/oOraph/criu.git
ARG CRIU_REF=fast_cuda_plugin_on_head

RUN git clone ${CRIU_REPO} /criu && cd /criu && git checkout ${CRIU_REF} && \
    make -j$(nproc) && make install-criu && \
    mkdir -p /usr/lib/criu && cp plugins/cuda/cuda_plugin.so /usr/lib/criu/ && \
    { [ -f plugins/cuda/cuda-offload ] && cp plugins/cuda/cuda-offload /usr/local/bin/ || true; }

# criu-base (crit helper)
FROM base AS criu-base

RUN apt-get update -y && apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ooraph/criu.git /criu && \
    cd /criu && \
    git checkout optim1 && \
    make -j$(nproc) && make install-criu && \
    mkdir -p /usr/lib/criu && \
    cp plugins/cuda/cuda_plugin.so /usr/lib/criu/ && \
    pip3 install --break-system-packages /criu/lib /criu/crit

ENTRYPOINT ["bash"]

