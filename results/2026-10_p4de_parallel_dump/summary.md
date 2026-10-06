# Parallel staging-page dump and `cuStreamGetCtx_v2`: p4de.24xlarge, driver 615.71.09 (2026-10-06)

Validates two plugin changes on vLLM + gpt-oss-120b (76 GB of GPU memory), same box as
[the compression session](../2026-10_p4de_compression_zero_skip/summary.md):

- **parallel staging-page dump** (custom storage off, the path of
  [#3191](https://github.com/checkpoint-restore/criu/issues/3191)): the staging pages used to be copied one 64 MB
  chunk at a time (`process_vm_readv`, then a blocking O_DIRECT write). They are now copied by
  `CUDA_DUMP_THREADS` workers (default 8) that read chunks out of the target and `pwrite` them at their offset;
- **custom storage** with `cuStreamGetCtx` resolved through `cuGetProcAddress` (the 3-argument `_v2` ABI)
  instead of a `dlsym` of the legacy symbol.

Setup: A100-SXM4-80GB (GPU 0), 8× NVMe RAID-0 (16 GB/s), driver 615.71.09 + Fabric Manager; CRIU from the
`upstream-cuda-custom-storage` series + these commits (`criu-src/`); `./run_p4de_dumppar.sh`, `RUNS=2`, caches
dropped before restore, inference validated after every restore.

## Results

| variant | dump | restore | staging-page copy on dump | parallel fill on restore |
|---|---|---|---|---|
| upstream `criu-dev`, no compression (same box) | 57.1 / 58.3 s | 44.2 / 41.6 s | — | — |
| custom storage off, `CUDA_DUMP_THREADS=1` (= previous serial dump) | 67.8 / 65.1 s | 17.1 / 16.5 s | 23.3 s (3.3 GB/s) | 20.4 GB/s |
| custom storage off, **8 threads** (default) | **51.8 / 51.5 s** | **17.4 / 16.5 s** | 8.4 s (9.0 GB/s) | 20–21 GB/s |
| custom storage off, 16 threads | 51.2 / 51.6 s | 17.1 / 16.8 s | 8.4 s (9.0 GB/s) | 21.3 GB/s |
| custom storage on (`cuStreamGetCtx_v2`, zero skip) | 11.4 / 11.3 s | 10.4 / 10.3 s | — | — |

## Reading

- The staging-page offload now wins on both sides against upstream: **dump 58 s → 52 s**, restore 42 s → 17 s.
  It used to cost 7 s more than upstream on dump.
- What remains of that dump: the driver's own VRAM→host copy (~38 s), the parallel copy (8.4 s), the injected
  `madvise(MADV_DONTNEED)` that frees the pages (3 s).
- 16 threads give nothing over 8: the copy levels off at ~9 GB/s, the same rate as the custom-storage
  checkpoint copy on this box. The array's write ceiling was not measured (fio measured reads only), so
  whether this is the disk or the copy is open.
- Custom storage with the `_v2` symbol path behaves exactly like before (11.4 s / 10.4 s).

Raw logs in `raw/` (`run_dp.log` = session driver, `dp_smoke.log`, `dp_vllm_gptoss.log`, status file).
