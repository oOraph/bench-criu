# Upstream LZ4 compression vs custom storage with zero-chunk skipping: p4de.24xlarge, driver 615.71.09 (2026-10-06)

Question from the CRIU review ([#3189](https://github.com/checkpoint-restore/criu/issues/3189)): does upstream's
memory compression with parallel decompression (`--compress` / `--compress-block` + `--decompress-threads`)
already give the restore gain? And does skipping all-zero chunks help the custom-storage path?

## Setup

| Component | Detail |
|---|---|
| Instance | AWS `p4de.24xlarge`, A100-SXM4-80GB (GPU 0 only), 8× NVMe RAID-0 (16 GB/s) |
| Driver | 615.71.09 proprietary + `nvidia-fabricmanager` 615.71.09 |
| Upstream | `criu-dev` @ `4485a86da`, `--image-io-mode=direct` on dump and restore |
| Ours | `upstream-cuda-custom-storage` + zero-chunk skipping (local branch `cs-zero-skip`, built from `criu-src/`), `--plugin-option=cuda_plugin.custom-storage=on` |
| Session | `PHASES="gptoss qwen" ./run_p4de_compress.sh`, `RUNS=2`, caches dropped before restore, inference validated after every restore |

Image size is the allocated size of the image directory (holes not counted).

## vLLM + gpt-oss-120b (76 GB of GPU memory; weights take 66 GiB of it)

| variant | dump | restore | image |
|---|---|---|---|
| upstream, no compression | 57.1 / 58.3 s | 44.2 / 41.6 s | 80.1 GB |
| upstream, `--compress-block 256K` + `--decompress-threads 0` | 101.1 / 100.1 s | 70.5 / 69.8 s | 66.9 GB |
| upstream, `--compress-block 256K`, default (serial) decompression | 101.2 / 101.4 s | 71.8 / 72.2 s | 66.9 GB |
| upstream, `--compress` (per page) + `--decompress-threads 0` | 141.6 / 140.4 s | 87.0 / 88.3 s | 65.8 GB |
| ours, custom storage | 11.5 / 11.4 s | 10.6 / 10.6 s | 80.1 GB |
| ours, custom storage + zero skip | **11.5 / 11.3 s** | **10.4 / 10.4 s** | 77.6 GB (38 of 1,136 chunks zero) |

## vLLM + Qwen3-8B (76 GB of GPU memory; weights ~16 GB, the rest mostly untouched KV cache)

| variant | dump | restore | image |
|---|---|---|---|
| upstream, no compression | 58.2 / 57.8 s | 45.6 / 42.5 s | 79.0 GB |
| upstream, `--compress-block 256K` + `--decompress-threads 0` | 66.7 / 67.5 s | 30.4 / 31.5 s | 17.5 GB |
| upstream, `--compress-block 256K`, default (serial) decompression | 68.1 / 67.7 s | 60.4 / 60.5 s | 17.5 GB |
| upstream, `--compress` (per page) + `--decompress-threads 0` | 85.1 / 85.0 s | 38.6 / 37.9 s | 17.7 GB |
| ours, custom storage | 11.0 / 11.0 s | 10.1 / 10.1 s | 79.0 GB |
| ours, custom storage + zero skip | **9.1 / 9.0 s** | **5.1 / 5.1 s** | 23.5 GB (828 of 1,131 chunks zero) |

## Reading

- **Upstream compression makes gpt-oss slower on both sides.** Dense weights barely compress (−16% image),
  the dump adds a single-threaded LZ4 pass over 76 GB, and restore goes from 42–44 s to 70–88 s.
- **The restore is bound by page faults and allocation, not by decompression.** During a compressed gpt-oss
  restore the worker pool exists (64 threads) but criu's main thread does almost all the work. `perf` on it:
  25.6% `native_queued_spin_lock_slowpath`, 12.6% `kernel_init_pages`, 9.2% `rep_movs_alternative`, ~15% in the
  fault and memcg charge paths, and only 6.0% `LZ4_decompress_safe`. Parallel decompression saves ~2 s of 70.
- **On sparse memory compression does pay off**: Qwen3-8B's image shrinks to 17.5 GB and parallel
  decompression halves the restore (60 s serial → 30 s), since zero blocks are filled by the workers.
  It still stays 6× slower than custom storage.
- **Zero skip in custom storage follows the KV-cache share**: no gain on gpt-oss (3% zero chunks), restore
  10.1 → 5.1 s and dump 11.0 → 9.0 s on Qwen3-8B, where the zero chunks are cleared on the GPU instead of
  being read from disk.

Raw logs in `raw/` (`run.log` = session driver, `cz_vllm_gptoss.log`, `cz_vllm_qwen.log`, status file).
