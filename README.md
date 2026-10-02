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

### 2026-10 variants

| Label | Image target | Description |
|---|---|---|
| `v42-ours` | `criu-v42-ours` | CRIU v4.2 + custom CUDA plugin (tag `v4.2-cuda-plugin-optim`, what production runs) |
| `upstream` | `criu-upstream-head` | upstream `criu-dev` pinned at `4485a86da` (2026-10-01): libcuda driver-api backend, PR #3021/#3022, LZ4 |
| `upstream-direct` | `criu-upstream-head` | same + `--image-io-mode=direct` (O_DIRECT/AIO page reads are OFF by default since PR #3066) |
| `upstream-cli-direct` | `criu-upstream-head` | same + `--plugin-option=cuda_plugin.backend=cuda-checkpoint` (CLI backend instead of libcuda) |

Scenarios are defined in `bench_compare.sh` as `label|image|extra criu options` (options are passed to both dump and restore). Override with `SCENARIOS="a|img|opts;b|img2|opts2"`.

## Setup

The benchmark targets a bare-metal machine with:
- 4× NVMe drives striped into a RAID-0 at `/mnt/nvme` (for maximum I/O throughput)
- NVIDIA GPU with driver persistence mode enabled (`nvidia-smi -pm 1`)
- Docker + nvidia-container-toolkit

Run `setup.sh` to install everything (NVIDIA driver, Docker, nvidia-container-toolkit, RAID array). BEWARE, use in a disposable vm.
The script auto-detects the free instance-store NVMe disks and stripes them (a single disk is used as is). Driver version defaults to `NVIDIA_DRIVER=610` (Ubuntu 26.04 ships 610.57.04 and 595.91.07 as of 2026-10).

## Build

```bash
docker build --target criu-dev -t criu-dev .
docker build --target criu-optimized -t criu-optimized .
docker build --target criu-fast-cuda-1 -t criu-fast-cuda-1 .
# 2026-10 variants
docker build --target criu-v42-ours -t criu-v42-ours .
docker build --target criu-upstream-head -t criu-upstream-head .
```

## Run

```bash
# Run all three scenarios, 2 rounds each
./bench_compare.sh

# Tune parameters
TENSOR_SIZE=100000 RUNS=5 ./bench_compare.sh
```

Results are printed as `RESULT label=... run=... dump_ms=... restore_ms=...` lines for easy grepping.

## Real-inference benchmark (vLLM)

`bench_vllm.sh` checkpoints and restores a running `vllm serve` (OpenAI API server) inside a Docker
container that carries CRIU, reproducing the June 2026 Kubernetes measurements
(`results/real_inference`) on a bench box. Images come from `Dockerfile.vllm` (vLLM image + CRIU
built from a chosen ref). Restore time is measured until `/health` answers; a completion request
then validates the engine. CRIU options follow the shim's vLLM settings
(`--shell-job --skip-in-flight --file-locks --ghost-limit 10485760`).

```bash
docker build -f Dockerfile.vllm -t vllm-criu-upstream --build-arg CRIU_REPO=https://github.com/checkpoint-restore/criu.git --build-arg CRIU_REF=4485a86da237 .
docker build -f Dockerfile.vllm -t vllm-criu-ours     --build-arg CRIU_REF=fast_cuda_plugin_on_head .
docker build -f Dockerfile.vllm -t vllm-criu-parallel --build-arg CRIU_REF=fast_cuda_plugin_on_head_parallel .
# weights go to $HF_CACHE (default /mnt/nvme/hf), e.g. `hf download Qwen/Qwen3-8B`
MODEL=Qwen/Qwen3-8B RUNS=2 sudo -E ./bench_vllm.sh
```

## Results

- [results/2026-10_g5.12xlarge_a10g/summary.md](results/2026-10_g5.12xlarge_a10g/summary.md) — production plugin (v4.2) vs upstream `criu-dev` head `4485a86da` on AWS `g5.12xlarge` (A10G, driver 610), `TENSOR_SIZE=60000`
- [results/mini_benchmark/summary.md](results/mini_benchmark/summary.md) — synthetic benchmark on AWS `g6.12xlarge` (NVIDIA L4, `TENSOR_SIZE=60000`)
- [results/real_inference/summary.md](results/real_inference/summary.md) — real inference workloads (SDXL, Llama-3.1-8B, Qwen3-8B) via runc checkpoint / restore + cuda plugin + cuda-checkpoint
- [results/vllm_sleep_awake/summary.md](results/vllm_sleep_awake/summary.md) — vLLM cooperative sleep/wake-up benchmark (Llama-3.1-8B, Qwen3-8B)
