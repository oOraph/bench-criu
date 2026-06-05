# Benchmark Results — CRIU PR #2021 / #2022

**Date:** 2026-06-03

## Hardware

| Component | Detail |
|-----------|--------|
| CPU | AMD EPYC 7R13, 24 cores / 48 threads, 1 NUMA node |
| RAM | 181 GiB |
| GPU | 4× NVIDIA L4 (23 GiB each), driver 595.71.05 (the test is using only one though) |
| Storage | 4× 875 GiB NVMe (Amazon EC2 Instance Storage) in RAID-0 → 3.4 TiB, XFS |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-1004-aws |

## Storage throughput (fio)

Sequential read, 1M block, 32 parallel jobs, `ioengine=sync`, `iodepth=1`:

| Metric | Value |
|--------|-------|
| Bandwidth | 2,401 MiB/s (2,518 MB/s) |
| IOPS | 2,401 |
| Avg latency | 13.3 ms |

## Benchmark configuration

- `TENSOR_SIZE=60000` (~27 GB GPU tensor)
- `RUNS=2`, `DROP_CACHE=yes` (page cache dropped between dump and restore)
- Dump dir on `/mnt/nvme` (RAID-0 NVMe)
- Images compared:
  - **orig** (`criu-dev`): upstream CRIU, no CUDA plugin
  - **new** (`criu-optimized`): upstream CRIU + PR #2022 (native AIO page reads + O_DIRECT), no CUDA plugin
  - **home-made** (`criu-fast-cuda-1`): upstream CRIU + [custom CUDA plugin](https://github.com/oOraph/criu/tree/fast_cuda_plugin_final) (GPU pages offloaded to `gpu-pages-*.img`)

## Results

| label | run | dump (ms) | restore (ms) |
|-------|-----|-----------|--------------|
| orig | 1 | 9,960 | 11,537 |
| orig | 2 | 9,912 | 11,460 |
| **orig avg** | | **9,936** | **11,499** |
| new | 1 | 9,882 | 9,623 |
| new | 2 | 9,916 | 9,614 |
| **new avg** | | **9,899** | **9,619** |
| home-made | 1 | 18,085 | 8,253 |
| home-made | 2 | 17,665 | 7,713 |
| **home-made avg** | | **17,875** | **7,983** |

Restore improvement vs orig:
- new (PR #2022): **−16%** (11,499 → 9,619 ms)
- home-made: **−31%** (11,499 → 7,983 ms)

## Image breakdown

| label | pages-*.img | gpu-pages-*.img |
|-------|-------------|-----------------|
| orig | 14 G | none |
| new | 14 G | none |
| home-made | 313 M | 14 G |

## Analysis

### PR #2022 — native AIO page reads

The optimization activates (O_DIRECT is supported on the RAID-0 NVMe). It gives a real **~16% restore improvement** but not the ~57% announced in the PR. The gap still has to be explained

### Home-made plugin optim

Home-made has a **faster restore** (7.98s vs 9.62s) despite doing more work (dump is 2× slower: 17.9s vs 9.9s). The reason for the slower dump / faster restore is architectural:

- **PR #2022** applies AIO to CRIU core's restore of `pages-*.img`. All 14 GB of GPU tensor pages flow through CRIU's per-VMA machinery at ~1.46 GB/s.
- **Home-made optim scoped on plugin** offloads GPU pages out of CRIU core entirely. On restore, the plugin reads `gpu-pages-14G` as a single contiguous O_DIRECT stream at **3.1–3.4 GB/s** (confirmed by timing lines in the output), then hands off to `cuda-checkpoint` for GPU restore (2.1 s). CRIU core only handles 313 MB of CPU pages.

Results are still unclear. May be the home made optim achieves 2× the throughput on the same data because it reads one large contiguous file rather than the fragmented per-VMA extents in `pages-*.img`.

### Timing breakdown for home-made restore

```
O_DIRECT pread (14 GB GPU pages):  ~4.5 s  @ 3.1–3.4 GB/s
cuda-checkpoint restore+unlock:    ~2.1 s
CRIU overhead + CPU pages (313M):  ~1.4 s
Total:                             ~7.9–8.3 s
```

### PR #2021 — parallel memfd restore

PR #2021 is included in the `criu-optimized` image alongside PR #2022, so its code runs but its contribution cannot be isolated from PR #2022's effect. The number of memfd-backed VMAs actually present in the test app's checkpoint was not measured, so whether this workload meaningfully exercises the parallel memfd path is unknown.

### Complementarity

PR #2022 and the custom plugin are not redundant — they address different data paths. Combined (plugin offloads GPU pages + AIO for CPU pages), the benefit of AIO on 313 MB of CPU pages is marginal (~0.2 s) (in this bench, expected). The bottleneck in the home-made scenario is GPU restore (pread + cuda-checkpoint unlock), which neither PR addresses.
