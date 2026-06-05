# bench-criu

Benchmark comparing CRIU checkpoint/restore performance for GPU (CUDA/PyTorch) workloads across different CRIU builds.

## What it measures

The benchmark runs a PyTorch test application that allocates GPU tensors of configurable size (`TENSOR_SIZE`, default 60 000), checkpoints it with CRIU, then restores it and verifies tensor integrity. It reports:

- **Dump time** — time (ms) for `criu dump` (run inside the container's namespaces via `nsenter`) to checkpoint the process and flush GPU memory to disk
- **Restore time** — time (ms) from `criu restore` (also run inside the container's namespaces) until the process is live and writing output again
- **GPU pages size** — size of the dumped GPU memory image
- **CPU pages size** — size of the dumped host memory pages

Three CRIU builds are compared head-to-head:

| Label | Branch | Description |
|---|---|---|
| `orig` | `criu-dev` | Baseline / upstream-tracking build |
| `new` | `criu-optimized` | Optimized CRIU build (contains https://github.com/checkpoint-restore/criu/pull/3021 + 3022 on top of criu-dev) |
| `home-made` | `fast-cuda-1` | Experimental fast CUDA checkpoint plugin (optimization scoped to the cuda plugin only where we offload gpu memory pages to drive with O_DIRECT) |

## Setup

The benchmark targets a bare-metal machine with:
- 4× NVMe drives striped into a RAID-0 at `/mnt/nvme` (for maximum I/O throughput)
- NVIDIA GPU with driver persistence mode enabled (`nvidia-smi -pm 1`)
- Docker + nvidia-container-toolkit

Run `setup.sh` to install everything (NVIDIA driver, Docker, nvidia-container-toolkit, RAID array). BEWARE, use in a disposable vm
The script is mean to run on a vm with 4 nvme drives (like g6.12xlarge for example), pay attention the the nvme names before running.

## Build

```bash
docker build --target criu-dev -t criu-dev .
docker build --target criu-optimized -t criu-optimized .
docker build --target criu-fast-cuda-1 -t criu-fast-cuda-1 .
```

## Run

```bash
# Run all three scenarios, 2 rounds each
./bench_compare.sh

# Tune parameters
TENSOR_SIZE=100000 RUNS=5 ./bench_compare.sh
```

Results are printed as `RESULT label=... run=... dump_ms=... restore_ms=...` lines for easy grepping.

## Results

- [results/mini_benchmark/summary.md](results/mini_benchmark/summary.md) — synthetic benchmark on AWS `g6.12xlarge` (NVIDIA L4, `TENSOR_SIZE=60000`)
- [results/real_inference/summary.md](results/real_inference/summary.md) — real inference workloads (SDXL, Llama-3.1-8B, Qwen3-8B) via runc checkpoint / restore + cuda plugin + cuda-checkpoint
