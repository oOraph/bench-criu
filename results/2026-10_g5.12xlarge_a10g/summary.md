# Benchmark Results — production plugin vs upstream criu-dev head (2026-10-01)

## Hardware

| Component | Detail |
|-----------|--------|
| Instance | AWS `g5.12xlarge` |
| CPU | AMD EPYC 7R32, 48 vCPUs |
| RAM | 186 GiB |
| GPU | 4× NVIDIA A10G (23 GiB), driver 610.57.04, persistence mode on (test uses GPU 0 only) |
| Storage | 1× 3.5 TiB NVMe (Amazon EC2 Instance Storage), XFS, no RAID |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-1006-aws |

## Storage throughput (fio, same job as June)

| Metric | Value |
|--------|-------|
| Bandwidth | 2,411 MiB/s (2,528 MB/s) — identical to the June 4× NVMe RAID-0 (2,401 MiB/s) |
| IOPS | 2,411 |
| Avg latency | 13.3 ms |

## Configuration

- `TENSOR_SIZE=60000` (~14 GB of GPU pages), `RUNS=2`, `DROP_CACHE=yes`
- Dump dir on `/mnt/nvme`
- Images (see Dockerfile): `criu-v42-ours` (tag `v4.2-cuda-plugin-optim`, production) and
  `criu-upstream-head` (upstream `criu-dev` @ `4485a86da`, 2026-09-24).
- Scenario options are passed to both `criu dump` and `criu restore`.

## Results

| label | options | run | dump (ms) | restore (ms) |
|-------|---------|-----|-----------|--------------|
| v42-ours | — | 1 | 19,892 | 7,745 |
| v42-ours | — | 2 | 19,876 | 7,867 |
| **v42-ours avg** | | | **19,884** | **7,806** |
| upstream | defaults (buffered I/O, Driver API backend) | 1 | 12,370 | 12,914 |
| upstream | | 2 | 12,325 | 12,893 |
| **upstream avg** | | | **12,348** | **12,904** |
| upstream-direct | `--image-io-mode=direct` | 1 | 20,288 | 10,641 |
| upstream-direct | | 2 | 20,330 | 10,755 |
| **upstream-direct avg** | | | **20,309** | **10,698** |
| upstream-cli-direct | `--image-io-mode=direct --plugin-option=cuda_plugin.backend=cuda-checkpoint` | 1 | 20,427 | 10,963 |
| upstream-cli-direct | | 2 | 20,402 | 11,071 |
| **upstream-cli-direct avg** | | | **20,415** | **11,017** |

Restore vs upstream defaults: upstream-direct −17%, v42-ours **−40%**. Restore vs best upstream (direct): v42-ours **−27%**.

## Image breakdown

| label | pages-*.img | gpu-pages-*.img |
|-------|-------------|-----------------|
| v42-ours | 319 M | 14 G |
| upstream (all variants) | 15 G | none |

## Timing breakdown, v42-ours restore (from plugin `[timing]` lines)

```
O_DIRECT pread (14 GB GPU pages):  ~5.1 s  @ 2.9 GB/s   (mlock 0 ms)
cuda-checkpoint restore+unlock:    ~1.6 s
CRIU overhead + CPU pages (319M):  ~1.0 s
Total:                             ~7.8 s
```

## Analysis

- **The custom plugin still restores fastest** (7.8 s vs 10.7 s for upstream with direct I/O and 12.9 s with upstream defaults). Same shape as June on L4 (8.0 s vs 9.6 s vs 11.5 s).
- **Upstream `--image-io-mode=direct` is a trade, not a free win**: since PR #3066 the flag also makes the dump O_DIRECT (splice), so dump goes from 12.3 s to 20.3 s — the same dump penalty as our plugin — for a 2.2 s restore gain. Our plugin pays the same dump cost and gains 5.1 s on restore.
- **Upstream defaults regressed vs the June "optimized" image**: O_DIRECT/AIO page reads became opt-in after regression #3053, so a default upstream build restores at buffered speed (12.9 s here vs 9.6 s in June with the PRs always on).
- **libcuda Driver API backend vs `cuda-checkpoint` CLI**: ~0.3 s faster restore (10.7 s vs 11.0 s), consistent with the ~450 ms/call process-spawn overhead measured in upstream PR #3076.
- Storage is not the explanation for any difference: single NVMe here = June RAID-0 bandwidth.

## Next

Port the plugin onto upstream head (`cuda_plugin.c` was split into CLI/Driver-API backends, so the dump/restore hooks need re-placing) and add a `upstream-head-ours` scenario; try the plugin together with `--image-io-mode=direct` (expected marginal: only 319 MB left for CRIU core).
