# bench-criu

Checkpoint/restore benchmarks for GPU (CUDA / PyTorch / vLLM) workloads with CRIU and the NVIDIA
`cuda-checkpoint` driver API, comparing upstream CRIU with the changes we intend to propose upstream
([oOraph/criu `upstream-cuda-custom-storage`](https://github.com/oOraph/criu/tree/upstream-cuda-custom-storage)).

## Headline result (2026-10-06)

vLLM serving **gpt-oss-120b** on an **A100-80GB** (AWS `p4de.24xlarge`, 8× NVMe RAID-0 at 16 GB/s,
driver 615.71.09), all variants on the same box. vLLM fills the card with KV cache, so the checkpoint is
**76 GB of GPU memory** whatever the model. Cold start of the server is 3 minutes.

![gpt-oss-120b on A100: dump/restore seconds per CRIU variant](results/2026-10_p4de_parallel_dump/gptoss-120b-a100.png)

| variant | dump | restore | what it does |
|---|---|---|---|
| upstream `criu-dev` `4485a86da`, `--image-io-mode=direct` | 58 s | 43 s | driver copies VRAM to host RAM, CRIU dumps those pages like any other memory |
| our branch, custom storage off | **52 s** | **17 s** | driver still copies VRAM↔host RAM; the plugin moves the staging pages itself, 8 threads on dump (9 GB/s) and 16 on restore (21 GB/s), with O_DIRECT and 2 MB pages |
| our branch, **custom storage on** (driver ≥ 615) | **11.4 s** | **10.4 s** | the plugin reads/writes VRAM directly through driver-exposed device mappings; no host staging copy at all |

With custom storage, restore is 7 s of VRAM copy (11 GB/s into the mapping) plus 3 s of CRIU work on the
3.7 GB of CPU pages and the process tree. On a model whose KV cache is mostly untouched (Qwen3-8B, same
76 GB), skipping all-zero chunks brings it to **9.0 s dump / 5.1 s restore**. Details and raw logs:
[parallel dump](results/2026-10_p4de_parallel_dump/summary.md),
[compression and zero skip](results/2026-10_p4de_compression_zero_skip/summary.md).

## What is compared

| Image target | Source | Description |
|---|---|---|
| `criu-upstream-head` | [checkpoint-restore/criu](https://github.com/checkpoint-restore/criu) `criu-dev` @ `4485a86da` (2026-09-24) | upstream: libcuda Driver API backend, PR #3021/#3022 (parallel memfd restore, AIO/O_DIRECT image reads), `--image-io-mode=direct` (PR #3066, off by default), LZ4 |
| `criu-ref` with `CRIU_REF=upstream-cuda-custom-storage` | [oOraph/criu](https://github.com/oOraph/criu/tree/upstream-cuda-custom-storage) | upstream head + the series proposed upstream: staging-page offload (O_DIRECT, parallel `process_vm_readv` dump and `process_vm_writev` restore) and CUDA custom storage (`--plugin-option=cuda_plugin.custom-storage=auto\|on\|off`, `auto` enables it on driver ≥ 615) |

`Dockerfile.vllm` builds either CRIU into a vLLM image (`APP_IMAGE`, `CRIU_REPO`, `CRIU_REF`). Older
targets in the `Dockerfile` (`criu-dev`, `criu-optimized`, `criu-fast-cuda-1`, `criu-v42-ours`) are the
builds behind the earlier result sets and are described in those summaries.

### Why the plugin is faster

With `cuda-checkpoint` the driver copies all of VRAM into the target process's host memory, and CRIU
then dumps that memory as ordinary pages. Upstream pays the driver copy (A100: ~38 s for 76 GB at
~2 GB/s on dump, ~10 s on restore) plus a page-cache-bound write/read of the same volume. Our plugin
detects the staging VMAs, writes them with parallel workers and O_DIRECT and drops them from the target
(dump), and on restore fills them in parallel straight from disk before the driver copies them back.
Upstream's restore is bound by one thread faulting in 4 KB pages. With the
CUDA 13.4 custom-storage API (driver ≥ 615) the driver copy disappears entirely: the plugin streams VRAM
to and from disk itself, and the time is the disk or PCIe time, whichever is slower.

## Benchmarks

| Script | What it measures |
|---|---|
| `bench_compare.sh` | synthetic: a torch app holding `TENSOR_SIZE`² random floats (60 000 ≈ 14.9 GB), plus `ZERO_SIZE`² floats that are three-quarters zero (like an untouched KV cache), dump and restore with integrity check |
| `bench_vllm.sh` | real inference: `vllm serve` checkpointed while serving, restore timed until `/health` answers, then a completion validates the engine; `SLEEP_MODE=1` adds vLLM sleep level 1 before the dump |
| `bench_sdxl.sh` | huggingface-inference-toolkit + SDXL |
| `fio.sh` | storage ceiling of the box (sync qd1, libaio qd32, raw single drive) |
| `run_p4de_cs.sh` | the one-shot session behind the headline result (driver, images, weights, tensor + vLLM matrices) |
| `run_p4de_compress.sh` | upstream LZ4 memory compression (`--compress`, `--compress-block`, `--decompress-threads`) vs custom storage with zero-chunk skipping |
| `run_p4de_dumppar.sh` | parallel staging-page dump (`CUDA_DUMP_THREADS`) and custom storage, on a box already set up by `run_p4de_compress.sh` |
| `zdtm_cuda_matrix.sh` | CRIU's ZDTM CUDA tests with custom storage `auto` and `off`, on a box with the CRIU tree built (setup in its header) |
| `proto/` | standalone custom-storage prototype (`cuda_cs.c`), driver `.run` installer with Fabric Manager handling, probes |

Scenarios are `label|image|opts[|dump_opts[|restore_opts[|env]]]`, semicolon separated: `opts` go to both
dump and restore, `dump_opts` / `restore_opts` to one side only, `env` (`VAR=value ...`) is set for both CRIU runs.
Result lines carry `img_gb`, the allocated size of the image directory.

CRIU runs as root on the host and enters the app container's namespaces with `nsenter`, as runc does.
Dump directories live on the NVMe array; caches are dropped between dump and restore (`DROP_CACHE=yes`).

### vLLM dumpability recipe

The server must be started with `UV_USE_IO_URING=0` (uvloop would use io_uring, undumpable),
`HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1` (no open HTTPS sessions to the Hub),
`GLOO_SOCKET_IFNAME=lo` and `TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_DUMP_ON_TIMEOUT=0`.
CRIU options: `--shell-job --skip-in-flight --file-locks --ghost-limit 10485760 --tcp-established --link-remap`.
Sleep mode needs `VLLM_SERVER_DEV_MODE=1`. `bench_vllm.sh` sets all of this.

## Setup

Disposable VM with NVMe instance store, an NVIDIA GPU, Ubuntu 24.04/26.04.

```bash
./setup.sh                      # NVIDIA driver (NVIDIA_DRIVER=610 from the Ubuntu archive), Docker, nvidia-container-toolkit,
                                # RAID-0 of the free NVMe drives at /mnt/nvme (single drive used as is)
SKIP_DRIVER=1 ./setup.sh        # keep a preinstalled driver
sudo proto/install-driver-run.sh 615.71.09   # proprietary driver from NVIDIA's .run (needed for custom storage, driver >= 615);
                                             # on NVSwitch boards (p4d/p4de/p5) also installs the matching nvidia-fabricmanager
                                             # from NVIDIA's CUDA apt repo (the Ubuntu archive stops at 595)
```

Fast storage matters: on a single NVMe (g5: ~2.5 GB/s read) every variant is disk-bound and the gains
shrink; the 16 GB/s array of the p4de is what exposes the plugin ceilings.

## Build and run

```bash
docker build --target criu-upstream-head -t criu-upstream-head .
docker build --target criu-ref -t criu-head-cs --build-arg CRIU_REF=upstream-cuda-custom-storage .

# synthetic
SCENARIOS="cs-on|criu-head-cs|--plugin-option=cuda_plugin.custom-storage=on;upstream-direct|criu-upstream-head|--image-io-mode=direct" \
  TENSOR_SIZE=60000 RUNS=2 sudo -E ./bench_compare.sh

# vLLM
docker build -f Dockerfile.vllm -t vllm-criu-upstream --build-arg CRIU_REPO=https://github.com/checkpoint-restore/criu.git --build-arg CRIU_REF=4485a86da237 .
docker build -f Dockerfile.vllm -t vllm-criu-cs --build-arg CRIU_REF=upstream-cuda-custom-storage .
hf download openai/gpt-oss-120b --cache-dir /mnt/nvme/hf   # weights go to $HF_CACHE
MODEL=openai/gpt-oss-120b RUNS=2 \
  SCENARIOS="cs-on|vllm-criu-cs|--plugin-option=cuda_plugin.custom-storage=on;upstream-direct|vllm-criu-upstream|--image-io-mode=direct" \
  sudo -E ./bench_vllm.sh
```

Results are printed as `RESULT label=... run=... dump_ms=... restore_ms=...` lines.

## All results

| Date | Box | What | Link |
|---|---|---|---|
| 2026-10-06 | p4de.24xlarge, 8× A100-80GB, driver 615 + FM, CUDA 13.4 | **ZDTM CUDA tests on real GPUs**, custom storage auto/off, 14/14 PASS incl. 8-GPU `cuda_multigpu00` | [summary](results/2026-10_p4de_zdtm_cuda/summary.md) |
| 2026-10-06 | p4de.24xlarge, A100-80GB, 8× NVMe 16 GB/s, driver 615 + FM | **parallel staging-page dump** (gpt-oss dump 66 s → 52 s), `cuStreamGetCtx_v2` validation | [summary](results/2026-10_p4de_parallel_dump/summary.md) |
| 2026-10-06 | same box | **upstream LZ4 compression** (`--compress`, `--compress-block`, `--decompress-threads`) vs custom storage with zero-chunk skipping, gpt-oss-120b + Qwen3-8B | [summary](results/2026-10_p4de_compression_zero_skip/summary.md) |
| 2026-10-05 | p4de.24xlarge, A100-80GB, 8× NVMe 16 GB/s, driver 615 + FM | **custom storage vs parallel plugin vs upstream**, tensor + gpt-oss-120b + Qwen3-8B | [summary](results/2026-10_p4de_custom_storage_615/summary.md) |
| 2026-10-02 | p4de.24xlarge, A100-80GB, driver 595 | plugin ceilings with the disk out of the way: serial vs parallel restore, upstream; SDXL, Qwen3-8B (plain and sleep level 1) and gpt-oss-120b through vLLM; the "driver copy" measurements that motivated custom storage | [summary](results/2026-10_p4de.24xlarge_a100/summary.md) |
| 2026-10-02 | g5.12xlarge, A10G, driver 615 | first custom-storage prototype measurements (standalone tool, single NVMe, tmpfs ceiling) | [summary](results/2026-10_g5_custom_storage_615/summary.md) |
| 2026-10-02 | g6.48xlarge, 8× L4, 8× NVMe capped at 5 GB/s | is the plugin or the disk the bottleneck (tensor test, mlock control) | [summary](results/2026-10_g6.48xlarge_l4/summary.md) |
| 2026-10-01 | g5.12xlarge, A10G, single NVMe | our v4.2 plugin vs upstream head | [summary](results/2026-10_g5.12xlarge_a10g/summary.md) |
| 2026-06 | g6.12xlarge, L4 | synthetic mini benchmark, June builds | [summary](results/mini_benchmark/summary.md) |
| 2026-06 | g6.12xlarge, L4, Kubernetes | real inference (SDXL, Llama-3.1-8B, Qwen3-8B) via runc checkpoint/restore | [summary](results/real_inference/summary.md) |
| 2026-06 | g6.12xlarge, L4 | vLLM cooperative sleep/wake-up (Llama-3.1-8B, Qwen3-8B) | [summary](results/vllm_sleep_awake/summary.md) |
