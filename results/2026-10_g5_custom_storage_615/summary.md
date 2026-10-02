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

## Requirements / gotchas found

- Driver ≥ 615 (API 13040) with the **proprietary** kernel module (open module untested with a correct pid).
- Caller needs **CAP_SYS_PTRACE** (Yama scope 1) to map another process's VRAM: in a plain container
  `cuCheckpointProcessCheckpoint` fails with `CUDA_ERROR_OPERATING_SYSTEM` (304). The shim runs CRIU on the host, fine.
- Load driver symbols via `cuGetProcAddress` (dlsym gives legacy ABIs → `CUDA_ERROR_INVALID_CONTEXT` on memcpy).
- `cuPointerGetAttribute(CONTEXT)` on the mapped pointer returns NULL; use `cuStreamGetCtx` on the per-device stream.
- `CUDA_ERROR_NOT_INITIALIZED` from the checkpoint API = "pid has no CUDA state" (don't pass the shell wrapper's pid).
- No Fabric Manager exists for 615 → not usable on NVSwitch boxes (p4d/p4de/p5) yet.
