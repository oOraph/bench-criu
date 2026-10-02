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

## Parallel restore (branch `fast_cuda_plugin_on_head_parallel`, commit `415e9795a`)

June 2026 WIP applied on the port: `CUDA_RESTORE_THREADS` plugin threads read 64 MB chunks of
`gpu-pages-*.img` (O_DIRECT) into bounce buffers and fill the target VMAs with `process_vm_writev`,
so the page faults run in parallel across cores. Image `criu-head-parallel`. Same 14.7 GB tensor,
`DROP_CACHE=yes`, Driver API backend.

| threads | run | dump (ms) | GPU page restore | rate | restore total (ms) |
|---|---|---|---|---|---|
| 8 | 1 | 11,851 | 2,479 ms | 5.9 GB/s | 4,814 |
| 8 | 2 | 11,815 | 2,215 ms | 6.6 GB/s | 4,584 |
| 16 | 1 | 11,912 | 2,139 ms | 6.8 GB/s | 4,521 |
| 16 | 2 | 11,841 | 2,105 ms | 7.0 GB/s | 4,508 |
| 32 | 1 | 11,840 | 1,995 ms | 7.3 GB/s | 4,382 |
| 32 | 2 | 11,845 | 1,976 ms | 7.4 GB/s | 4,385 |

Restore total: **4.4 s** vs 6.4 s serial (−31%) vs 8.6 s upstream-direct (−49%). Remaining: Driver API
restore+unlock ~1.5 s + CRIU ~0.9 s.

### Why above the 5.0 GB/s array ceiling: instance-store burst allowance

fio per-second log (libaio 8×qd32, after 20 s idle): **9,459 MiB/s in the first second, then 4,805 MiB/s
sustained**; a 3 s burst averages 6,331 MiB/s. A 2 s restore rides the burst bucket, which is the
realistic situation for a restore after idle. The sustained 4.8 GiB/s applies to long transfers (dumps).

### No-disk ceilings (tmpfs, `BENCH_DIR=/mnt/tmpfs`, kernel 7.0 accepts O_DIRECT on tmpfs)

| scenario | GPU page restore | rate | restore total |
|---|---|---|---|
| serial (head-ours) | 1,950 ms | 7.5 GB/s | 3,707 ms |
| parallel 16 threads | 609 ms | 24.0 GB/s | 2,401 ms |
| parallel 32 threads | 611 ms | 24.0 GB/s | 2,425 ms |
| upstream-direct | — | — | 9,333 ms |

The serial path tops out at 7.5 GB/s even from RAM (one 64 MB request in flight, no overlap), so on
the array its 3.6 GB/s was an I/O-concurrency limit, not CPU. The parallel path reaches 24 GB/s
(memory-bandwidth / `process_vm_writev` bound, flat from 16 to 32 threads); on any real array it is
disk-bound. Upstream gets no benefit from tmpfs (9.3 s).

## Dump breakdown (parallel-t32 run 2, `dump.log`, 14.0 GB staging)

| step | time | rate |
|---|---|---|
| Driver API checkpoint (driver copies VRAM → host staging pages) | 6,176 ms | 2.3 GB/s |
| plugin `process_vm_readv` of staging pages | 1,967 ms | 7.1 GB/s |
| plugin O_DIRECT write of `gpu-pages-*.img` | 2,483 ms | 5.6 GB/s (burst) |
| injected `madvise(MADV_DONTNEED)` | 938 ms | |
| CRIU core (319 MB) + rest | ~0.3 s | |
| **total** | **11.8 s** | |

More than half of the dump is the driver's own VRAM → host copy at 2.3 GB/s (cf. the GCR paper's
3.0 GB/s measurement for cuda-checkpoint). Neither the plugin nor CRIU can touch that step today;
only the custom-storage mode (driver ≥ 615), where the checkpointer drives the copies itself from
the zero-copy mapped device pointer, can. Parallelising our readv+write would save at most ~2 s.

## Real inference: vLLM 0.30 + Qwen/Qwen3-8B (`bench_vllm.sh`, Docker, same box)

Server: `python3 -m vllm.entrypoints.openai.api_server --max-model-len 4096 --gpu-memory-utilization 0.9`
on GPU 0 (L4 23 GiB), process tree = API server + EngineCore. CRIU options
`--shell-job --skip-in-flight --file-locks --ghost-limit 10485760 --tcp-established --link-remap`,
Driver API backend, `DROP_CACHE=yes`, restore measured until `/health` answers, then a completion request.
Checkpoint: 19 GB of GPU pages (weights + KV cache), 2.9 GB of CPU pages. Cold start ~140 s.

| variant | run | dump (ms) | restore (ms) | inference |
|---|---|---|---|---|
| upstream head, `--image-io-mode=direct` | 1 | 19,951 | 12,967 | OK |
| upstream head, `--image-io-mode=direct` | 2 | 19,963 | 12,919 | OK |
| parallel plugin (16 threads) | 1 | 19,151 | 7,119 | OK |
| parallel plugin (16 threads) | 2 | 19,055 | 7,436 | OK |

Parallel restore breakdown: GPU page fill 2.8–3.2 s (6.4–7.1 GB/s for 19 GB) + Driver API restore+unlock
1.9 s (+0.2 s for the API server task) + CRIU core/CPU pages ~2 s. **Restore −44% vs upstream** (7.3 s vs
12.9 s); dump equal (the driver's VRAM→host copy is 8.3 s of it, see breakdown above).

Compared with the June k8s run on the same model (baseline 14.5 s, upstream PRs 15 s, our serial plugin
14–15 s restore; our dump 33 s): the serial plugin brought nothing on vLLM then, the parallel one halves
the restore now, and the 8-drive array removes the dump penalty.

From tmpfs (`BENCH_DIR=/mnt/tmpfs`, one round each), i.e. the no-disk floor on a real vLLM tree:

| variant | dump (ms) | restore (ms) | GPU page fill |
|---|---|---|---|
| parallel plugin (16 threads) | 23,022 | 3,885 | 706 ms (28.8 GB/s) |
| upstream head, `--image-io-mode=direct` | 22,686 | 14,946 | — |

Parallel floor ≈ 1.9 s driver restore+unlock + ~1.3 s CRIU core + 0.7 s fill. Upstream is slower from RAM
than from disk (14.9 s vs 12.9 s), as in the tensor test. Dumps are slower on tmpfs for both (writing 22 GB
into RAM-backed files competes with the driver's VRAM→host copy for memory bandwidth).

Serial port (`fast_cuda_plugin_on_head` before `9b67fbc91`) failed to restore vLLM on the Driver API
backend: `setns(CLONE_NEWNS)` → EINVAL because CRIU is multithreaded once the backend's worker thread
exists (multi-process tree only). Fixed by forking a helper for the bind mount (`9b67fbc91`); validated:

| variant | run | dump (ms) | restore (ms) | plugin pread | inference |
|---|---|---|---|---|---|
| serial plugin (`9b67fbc91`) | 1 | 19,112 | 9,891 | 5,617 ms (3.6 GB/s) | OK |
| serial plugin (`9b67fbc91`) | 2 | 19,045 | 7,374 | 2,907 ms (7.0 GB/s) | OK |

The serial path's rate swings with the array's burst state (3.6 vs 7.0 GB/s on consecutive runs, each
right after a 22 GB dump + drop_caches); the parallel path was 6.4–7.1 GB/s on both runs.

## Conclusions

1. **Adopt the parallel restore** as the plugin's restore path; 16 threads is the knee (`min(ncpu, 16)`).
2. **Dump is now the dominant cost** (11.8 s vs 4.4 s restore), and 6.2 s of it is the driver's VRAM → host
   copy at 2.3 GB/s. Parallelising our own readv+write saves ~2 s at best; the real lever is the
   custom-storage mode (driver ≥ 615), which replaces the driver copy on both dump and restore
   (also the 1.5 s restore+unlock) with copies we drive at PCIe speed, overlapped with disk I/O.
3. With `MADV_HUGEPAGE` and no `mlock`, THP mode `madvise` is a node requirement worth asserting.
