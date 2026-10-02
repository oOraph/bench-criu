# Benchmark Results — 8-drive array: is the plugin or the disk the bottleneck? (2026-10-02)

## Hardware

| Component | Detail |
|-----------|--------|
| Instance | AWS `g6.48xlarge` (p4d had no capacity) |
| CPU | 192 vCPUs |
| RAM | 728 GiB |
| GPU | 8× NVIDIA L4 (23 GiB), driver 610.57.04, persistence mode on (test uses GPU 0) |
| Storage | 8× 875 GiB NVMe instance store, mdadm RAID-0, XFS, 6.8 TiB |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-1006-aws, THP `madvise` |

## Storage throughput (fio)

| job | bandwidth |
|---|---|
| 32 sync jobs qd1, 1M (historical) | 4,802 MiB/s |
| libaio 8 jobs × qd32, 1M | 4,800 MiB/s |
| raw single drive, libaio 4 × qd32 | 604 MiB/s |

G6 instance-store drives are capped at ~600 MiB/s each (same figure on g6.12xlarge and g6.24xlarge):
the array ceiling is **5.0 GB/s**.

## Configuration

`TENSOR_SIZE=60000` (~14.7 GB staging), `RUNS=2`, `DROP_CACHE=yes`. Images: `criu-head-ours` =
branch `fast_cuda_plugin_on_head` incl. `3c9682e21` (no mlock pre-fault), Driver API backend;
`criu-upstream-head` = criu-dev `4485a86da`.

## Results

| label | options | run | dump (ms) | restore (ms) | plugin pread |
|-------|---------|-----|-----------|--------------|--------------|
| head-ours | — | 1 | 11,780 | 6,842 | 4,083 ms (3.6 GB/s) |
| head-ours | — | 2 | 11,869 | 6,428 | 4,074 ms (3.6 GB/s) |
| upstream-direct | `--image-io-mode=direct` | 1 | 12,633 | 8,708 | — |
| upstream-direct | | 2 | 12,381 | 8,598 | — |
| head-ours-memlock | `MEMLOCK_UNLIMITED=yes` (control: no mlock in code) | 1 | 12,281 | 6,329 | 4,067 ms (3.6 GB/s) |
| head-ours-memlock | | 2 | 11,770 | 6,437 | 4,106 ms (3.6 GB/s) |

## Analysis

- **The single injected-pread thread is the bottleneck**: 3.6 GB/s on a 5.0 GB/s array (it was
  disk-bound at 2.9 GB/s on the 2.5 GB/s A10G box). This is the plugin's serial ceiling: THP
  fault+zero of the destination inside each 64 MB `pread64` plus per-chunk DMA setup, with no
  overlap between chunks. Matches the April p4de observation ("didn't saturate the disks").
- Restore breakdown: pread 4.1 s + Driver API restore+unlock 1.5–1.65 s + CRIU ~1 s ≈ 6.4–6.8 s.
- Upstream direct mode: 8.6 s, −27% for us again.
- Dump dropped from 19.8 s (1 drive) to 11.8 s thanks to write bandwidth; dump now ≈ upstream's.
- The mlock-free build (`3c9682e21`) compiled and passed on both backends; raising the memlock
  limit no longer changes anything (control).

## Next

Parallel restore (June WIP, now commit `415e9795a` on `fast_cuda_plugin_on_head_parallel`):
N plugin threads, O_DIRECT reads into bounce buffers + `process_vm_writev`. Results below when run.
