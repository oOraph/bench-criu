# Custom-storage checkpoint/restore prototype (CUDA 13.4 API, driver 615.71.09) — first measurements (2026-10-02)

## Setup

| Component | Detail |
|---|---|
| Instance | AWS `g5.12xlarge` (sandbox), test uses GPU 0 |
| GPU | NVIDIA A10G 24 GB, PCIe gen4 |
| Driver | **615.71.09 proprietary kernel module** via NVIDIA's `.run` installer (`proto/install-driver-run.sh`); CUDA driver API 13040 |
| Storage | 1× 3.5 TB NVMe instance store, XFS (read ~2.4 GiB/s, write ~1.4 GB/s measured here) |
| Target | `test_app.py` (torch) in the `criu-head-ours` image, container started with `--cap-add SYS_PTRACE` |
| Tool | `proto/cuda_cs.c` — lock, checkpoint with `customStorageInfo_out`, pinned 4×64 MB ring, O_DIRECT image, `cuCheckpointOperationComplete`; mirror for restore |

## Result, 14.7 GB of VRAM (`TENSOR_SIZE=60000`), caches dropped between dump and restore

| | driver staging path (`cuda-checkpoint`) | custom storage (`cuda_cs`) |
|---|---|---|
| checkpoint | 7,591 ms (VRAM → target host RAM; **no disk I/O included**) | **11,295 ms** total: map 55 ms + copy 10,800 ms (disk write 10,537 ms @1.4 GB/s, PCIe wait 8 ms) + complete 59 ms |
| host RAM held by target afterwards | 14.9 GB | 0.6 GB |
| restore | 3,366 ms (host RAM → VRAM; after a separate disk read) | **5,807 ms** total: map 117 ms + copy 4,928 ms (disk read 4,672 ms @3.0 GB/s, PCIe wait 4 ms) + complete 2 ms + unlock 1 ms |
| correctness | tensors verified | tensors verified |

## Reading

- With custom storage the driver's own copies are gone: the operation time **is the disk time**, PCIe waits are
  milliseconds (the transfers overlap the I/O), and the target holds no VRAM-sized host staging.
- End to end on this box vs the current plugin path (dump = 7.6 s driver copy + ~12 s readv/write; restore =
  ~5 s fill + 3.4 s driver copy): dump 11.3 s and restore 5.8 s, both disk-bound.
- Extrapolation to the p4de (16 GB/s array, A100): the 37 s dump copy and 10 s restore copy measured there
  would become ~PCIe-bound (~5 s each for 71 GB), i.e. dump ~65 s → ~10 s and restore 16 s → ~7 s.

## Same test from tmpfs (no disk): where the prototype's own ceiling is

| | driver staging path | custom storage (`cuda_cs`) |
|---|---|---|
| checkpoint | 7,606 ms (VRAM → host RAM) | 7,522 ms total: copy 7,024 ms (file write 6,761 ms @2.1 GB/s, PCIe wait 12 ms) |
| restore | 3,029 ms (host RAM → VRAM) | 2,861 ms total: copy 2,410 ms (file read 2,150 ms @6.1 GB/s, PCIe wait 11 ms) |

Even from RAM the time is spent in the prototype's **single-threaded 64 MB `pwrite`/`pread` loop**, not on
PCIe (waits stay at ~10 ms): the copies the driver does at 1.9–2.3 GB/s are now done at whatever rate the
I/O side delivers, and that side is trivially parallelisable (the same lesson as the parallel restore plugin).
Next step for the engine: N I/O threads over the ring buffers, which should push both directions toward the
PCIe gen4 ceiling (~20+ GB/s on this card) when the storage allows.

## Parallel engine (N worker threads, own CUDA stream + 2 pinned 64 MB buffers each), 14.7 GB

| storage | threads | checkpoint (copy wall) | restore (copy wall) | notes |
|---|---|---|---|---|
| NVMe (g5, write ~1.4 GB/s, read ~2.7 GB/s) | 16 | 11.3 s total (10.8 s copy, per-thread I/O 9.4 s) | 6.3 s total (5.4 s copy, per-thread I/O 3.7 s) | both purely disk-bound; PCIe waits 0.2–0.4 s |
| tmpfs (no disk) | 16 | 7.6 s (2.1 GB/s excl. 0.75 s pinned alloc; threads spend 6.3 s in `pwrite`) | 2.8 s (**7.1 GB/s** excl. alloc; PCIe wait 1.3 s/thread) | tmpfs serialises shmem writes (artefact); restore shows the mapping's H2D rate |
| tmpfs | 32 | 7.6 s | 4.4 s (worse) | more threads do not help |

Reading: on real storage the engine is storage-bound as intended. The host→device copy **into the custom-storage
mapping runs at ~7 GB/s on the A10G**, far below the card's ~20 GB/s for ordinary pinned copies and in the same
range as the driver's own host→VRAM copy (14.7 GB in 3.0 s). So on the restore side custom storage buys the
overlap of disk and PCIe time and the removal of host staging, not a faster copy, unless the mapping's throughput
can be raised (open question: intrinsic to the zero-copy mapping, or our copy pattern?). The device→host side
showed no such limit (PCIe waits ~0.2 s while writes took 9–10 s), so the dump-side gain stands.
Revised projection for the p4de/gpt-oss case: dump 66 s → ~10–15 s, restore 17 s → ~10–12 s (was "~7 s").

### Copy pattern sweep (tmpfs restore, 14.7 GB, rates excluding pinned-buffer allocation)

| threads × chunk | rate | note |
|---|---|---|
| 1 × 64 MB | 6.8 GB/s | bound by the single thread's file read; PCIe wait 11 ms |
| **4 × 64 MB** | **10.9 GB/s** | PCIe wait 0.6 s/thread: the mapping is now the limiter |
| 8 × 64 MB | 9.3 GB/s | |
| 16 × 64 MB | 7.1 GB/s | |
| 32 × 64 MB | 5.0 GB/s | |
| 2 × 64 MB | 12.0 GB/s | PCIe wait 49 ms: read-bound again |
| 3 × 64 MB | 11.4 GB/s | |
| 4 × 32 MB | 12.0 GB/s | |
| 6 × 64 MB | 9.9 GB/s | |
| 8 × 256 MB / 16 × 256 MB / 4 × 512 MB | 4.7 / 2.9 / 4.5 GB/s | large transfers into the mapping are slow |

H2D into the mapping plateaus around **12 GB/s on the A10G with 2–4 streams and 32–64 MB chunks** and degrades with
more concurrent streams or larger transfers (vs ~20 GB/s for ordinary pinned copies on this card). Engine
defaults changed to 4 copy threads; next design step: decouple I/O parallelism (many readers feeding a
queue) from the copy side (2–4 streams).

## Through CRIU (branch `custom_storage`, image `criu-head-cs`, `--plugin-option=cuda_plugin.custom-storage=on`)

Smoke (0.41 GB): dump 1.58 s, restore 1.52 s, `gpu-cs-66.img` 409 MB, `pages-*.img` 334 MB (no staging pages),
tensors verified.

14.7 GB tensor, `bench_compare.sh`, `RUNS=2`, `DROP_CACHE=yes`, same single NVMe:

| path | run | dump (ms) | restore (ms) | GPU copy | driver step |
|---|---|---|---|---|---|
| custom storage on | 1 | 11,261 | 6,773 | 5,550 ms (2.7 GB/s read) | restore+unlock incl. copy 5,936 ms |
| custom storage on | 2 | 11,527 | 6,882 | 5,647 ms (2.6 GB/s) | 6,028 ms |
| parallel plugin (cs off) | 1 | 19,736 | 7,021 | 4,593 ms fill (3.2 GB/s) | restore+unlock 1,555 ms |
| parallel plugin (cs off) | 2 | 19,682 | 7,007 | 4,595 ms fill (3.2 GB/s) | 1,545 ms |

Dump: **−42%** (11.4 s vs 19.7 s): the driver's VRAM→host copy (7.6 s) and our readv/write are replaced by one
disk-write-bound pass (checkpoint copy 10.5 s at the drive's 1.4 GB/s). Restore: tie (6.8 vs 7.0 s) — both are
bound by the drive's read speed; custom storage hides the 1.5 s driver copy by overlapping it with the read but
reads at 2.6 GB/s vs 3.2 GB/s for the parallel fill. The restore gain appears when the array is faster than
the mapping's H2D rate (see the tmpfs numbers) and in host RAM: no VRAM-sized staging.

## Requirements / gotchas found

- Driver ≥ 615 (API 13040) with the **proprietary** kernel module (open module untested with a correct pid).
- Caller needs **CAP_SYS_PTRACE** (Yama scope 1) to map another process's VRAM: in a plain container
  `cuCheckpointProcessCheckpoint` fails with `CUDA_ERROR_OPERATING_SYSTEM` (304). The shim runs CRIU on the host, fine.
- Load driver symbols via `cuGetProcAddress` (dlsym gives legacy ABIs → `CUDA_ERROR_INVALID_CONTEXT` on memcpy).
- `cuPointerGetAttribute(CONTEXT)` on the mapped pointer returns NULL; use `cuStreamGetCtx` on the per-device stream.
- `CUDA_ERROR_NOT_INITIALIZED` from the checkpoint API = "pid has no CUDA state" (don't pass the shell wrapper's pid).
- No Fabric Manager exists for 615 → not usable on NVSwitch boxes (p4d/p4de/p5) yet.
