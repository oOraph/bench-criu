# Benchmark Results — p4de.24xlarge, 16 GB/s array: plugin ceilings with the disk out of the way (2026-10-02)

## Hardware

| Component | Detail |
|-----------|--------|
| Instance | AWS `p4de.24xlarge` (open capacity reservation, us-east-1c) |
| CPU | 96 vCPUs |
| RAM | 1,121 GiB |
| GPU | 8× NVIDIA A100-SXM4-80GB (test uses GPU 0); driver **595.91.07-server** + Fabric Manager (see note) |
| Storage | 8× 875 GiB NVMe instance store, mdadm RAID-0, XFS, 6.8 TiB |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-1006-aws, THP `madvise` |

Driver note: p4de is an NVSwitch (HGX) system; CUDA returns error 802 "system not yet initialized" until
NVIDIA Fabric Manager runs, and FM must match the driver to the patch level. Ubuntu 26.04 ships FM only
for 595.91.07 (`nvidia-driver-595-server` + `nvidia-fabricmanager-595`); no FM exists for 610.57.04 or
615.71.09, and `NVreg_NvLinkDisable=1` does not lift the requirement. Hence 595 on this box (610 elsewhere).

## Storage throughput (fio)

| job | bandwidth |
|---|---|
| 32 sync jobs qd1, 1M | 15.0 GiB/s (16.1 GB/s) |
| libaio 8 × qd32, 1M | 15.0 GiB/s (16.1 GB/s) |
| raw single drive | 1,918 MiB/s |

Matches AWS's documented 16 GB/s aggregate; 3.2× the g6.48xlarge array.

## Tensor test (`bench_compare.sh`, `TENSOR_SIZE=60000` ≈ 14.7 GB staging, `RUNS=2`, `DROP_CACHE=yes`)

| label | run | dump (ms) | restore (ms) | GPU page fill | driver restore+unlock |
|---|---|---|---|---|---|
| upstream-direct (`--image-io-mode=direct`) | 1 | 11,143 | 8,910 | — | — |
| upstream-direct | 2 | 11,060 | 8,722 | — | — |
| head-ours (serial, `9b67fbc91`) | 1 | 13,004 | 6,226 | 3,159 ms (4.7 GB/s) | 2,158 ms |
| head-ours | 2 | 13,028 | 6,096 | 3,108 ms (4.8 GB/s) | 2,158 ms |
| parallel-t16 (`415e9795a`) | 1 | 13,045 | 3,960 | 710 ms (21.0 GB/s) | 2,360 ms |
| parallel-t16 | 2 | 12,997 | 3,855 | 692 ms (21.6 GB/s) | 2,299 ms |
| parallel-t32 | 1 | 12,832 | 3,871 | 723 ms (20.7 GB/s) | 2,268 ms |
| parallel-t32 | 2 | 12,942 | 3,974 | 706 ms (21.1 GB/s) | 2,370 ms |

### Analysis

- **Serial path ceiling is its own, ~4.8 GB/s**, on a 16 GB/s array (one 64 MB request in flight).
- **Parallel path fills 14.7 GB in 0.7 s (21 GB/s)**, i.e. at its memory-bound ceiling measured from tmpfs
  (24 GB/s); the array is no longer visible. 16 threads = 32 threads.
- **Restore is now driver-bound**: 2.3 s of the 3.9 s is the driver's host→VRAM copy on the A100
  (~6.4 GB/s), the rest is CRIU core. Restore −56% vs upstream-direct (3.9 s vs 8.8 s).
- Dump: 13.0 s for ours vs 11.1 s upstream; the driver's VRAM→host copy dominates both (see g6.48xlarge
  breakdown); our extra readv+write costs ~2 s at this write bandwidth.
