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
NVIDIA Fabric Manager runs, and FM must match the driver to the patch level. Ubuntu's archive ships FM only
for 595.91.07 (`nvidia-driver-595-server` + `nvidia-fabricmanager-595`), and `NVreg_NvLinkDisable=1` does not
lift the requirement. Hence 595 on this box (610 elsewhere). Correction (2026-10-05): NVIDIA's CUDA apt repo
does ship `nvidia-fabricmanager` 610.57.04 and 615.71.09 (ubuntu2204/2404/2604), so 610 or 615 + FM is
possible on HGX boxes; see `proto/install-driver-run.sh`.

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

## Real inference: vLLM 0.30 + Qwen/Qwen3-8B on the A100-80GB (`bench_vllm.sh`, `RUNS=2`, `DROP_CACHE=yes`)

Same server settings as on the L4 (`--gpu-memory-utilization 0.9`, `--max-model-len 4096`), but on an 80 GB
GPU vLLM reserves 72 GB, so the checkpoint is **~71 GB of GPU pages + 3 GB of CPU pages (74 GB total)**
instead of 22 GB on the L4. The KV cache, not the model, is the checkpoint on big GPUs.

| variant | run | dump (ms) | restore (ms) | GPU page fill | driver restore+unlock | inference |
|---|---|---|---|---|---|---|
| upstream-direct | 1 | 58,889 | 40,657 | — | — | OK |
| upstream-direct | 2 | 57,908 | 40,641 | — | — | OK |
| serial plugin (`9b67fbc91`) | 1 | 64,972 | 27,966 | 15,906 ms (4.8 GB/s) | 9,511 ms | OK |
| serial plugin | 2 | 65,244 | 27,727 | 15,731 ms (4.8 GB/s) | 9,492 ms | OK |
| parallel plugin, 16 threads | 1 | 65,187 | 16,387 | 3,657 ms (20.7 GB/s) | 10,220 ms | OK |
| parallel plugin, 16 threads | 2 | 65,583 | 15,749 | 3,511 ms (21.6 GB/s) | 9,713 ms | OK |

- **Restore: parallel 16.1 s vs upstream 40.6 s (−61%)**; serial 27.8 s (−32%). The parallel fill of 71 GB
  takes 3.6 s; the driver's host→VRAM copy (~10 s, ~7 GB/s) is now 2/3 of our restore.
- **Dump: 65 s vs 58 s**, of which the driver's VRAM→host checkpoint copy is **37 s** (~1.9 GB/s) for every
  variant; our readv+write of 71 GB adds ~7 s at this array's write bandwidth.
- Upstream restores at 1.8 GB/s regardless of hardware (same per-byte rate as on the L4).
- The driver-side copies (37 s dump + 10 s restore) dominate everything at this checkpoint size: that is
  the custom-storage case. Workload-side, sleep level 1 / KV-cache unmap would shrink the checkpoint to the
  16 GB of weights (see sleep-mode results below).

## Real inference: huggingface-inference-toolkit 0.5.6 + stabilityai/stable-diffusion-xl-base-1.0 (`bench_sdxl.sh`, `RUNS=2`)

Image `raphael31415/huggingface-inference-toolkit:gpu-2` (torch 2.5.1, diffusers 0.33.1) + CRIU; gunicorn/uvicorn
server, text-to-image; restore measured until `/health`, then a real generation (~7.5 s on the A100).
Checkpoint: 7.4 GB GPU pages + 3.1–3.5 GB CPU pages. Cold start ~22 s from the NVMe model dir.

| variant | run | dump (ms) | restore (ms) | GPU page fill | driver restore+unlock | inference |
|---|---|---|---|---|---|---|
| upstream-direct | 1 | 6,680 | FAILED (see raw) | — | — | — |
| upstream-direct | 2 | 7,764 | 7,713 | — | — | OK |
| serial plugin | 1 | 8,594 | 6,906 | 1,784 ms (4.4 GB/s) | 1,942 ms | OK |
| serial plugin | 2 | 8,362 | 6,395 | 1,622 ms (4.8 GB/s) | 1,911 ms | OK |
| parallel plugin, 16 threads | 1 | 8,745 | 5,934 | 397 ms (19.9 GB/s) | 2,002 ms | OK |
| parallel plugin, 16 threads | 2 | 8,811 | 5,679 | 408 ms (19.4 GB/s) | 1,982 ms | OK |

- Restore: parallel 5.8 s vs upstream 7.7 s (−25%). With only 7.4 GB of GPU pages the fixed costs dominate:
  driver restore+unlock ~2 s, CRIU core on 3+ GB of CPU pages and the gunicorn tree ~3 s.
- Dump: ~8.5 s ours vs 7.8 s upstream; driver checkpoint copy 4.1 s, our readv+write 2.4 s.
- June k8s reference (L4): baseline 6.3 s restore, ours 5.6 s, upstream PRs 7.8 s; dump 9.6 / 15.2 / 9.1 s.

## vLLM sleep level 1 → checkpoint → restore → wake_up (`bench_vllm.sh SLEEP_MODE=1`, Qwen3-8B, `RUNS=2`)

`--enable-sleep-mode` + `VLLM_SERVER_DEV_MODE=1`; `POST /sleep?level=1` before the dump (weights → host RAM,
KV cache freed), `POST /wake_up` after the restore. Checkpoint: **24 GB of CPU pages + 2.2 GB of GPU pages**
(vs 71 + 3 GB without sleep). Our plugin has almost nothing to offload here; the CPU pages go through CRIU core.

| variant | run | sleep (ms) | dump (ms) | restore (ms) | wake_up (ms) | **dump side** (sleep+dump) | **restore side** (restore+wake) |
|---|---|---|---|---|---|---|---|
| upstream, buffered | 1 | 10,774 | 15,174 | 9,693 | 3,611 | 26.0 s | 13.3 s |
| upstream, buffered | 2 | 10,765 | 15,153 | 9,829 | 3,606 | 25.9 s | 13.4 s |
| upstream, `--image-io-mode=direct` | 1 | 10,795 | 11,596 | 8,301 | 3,609 | 22.4 s | 11.9 s |
| upstream, `--image-io-mode=direct` | 2 | 10,745 | 11,806 | 7,678 | 3,608 | 22.6 s | 11.3 s |
| parallel plugin + direct | 1 | 10,751 | 11,912 | 6,927 (GPU fill 184 ms) | 3,626 | 22.7 s | 10.6 s |
| parallel plugin + direct | 2 | 10,697 | 11,616 | 6,973 (GPU fill 178 ms) | 3,618 | 22.3 s | 10.6 s |

Compared with the full (no-sleep) checkpoint on this box:

| | dump side | restore side |
|---|---|---|
| upstream-direct, no sleep | 58 s | 40.6 s |
| parallel plugin, no sleep | 65 s | 16.1 s |
| upstream-direct + sleep 1 | 22.5 s | 11.6 s |
| parallel + direct + sleep 1 | 22.5 s | 10.6 s |

- Sleep level 1 cuts the dump side ~3× for everyone (no 37 s driver VRAM→host copy: the model offload is
  done by vLLM itself in ~10.8 s, and the KV cache is dropped) and makes the restore side 10.6–11.6 s.
- On the restore side with sleep, CRIU core moves the 24 GB of CPU pages at ~4–5 GB/s with `--image-io-mode=direct`
  (buffered: ~3 GB/s); our plugin's share is 0.18 s. The 3.6 s wake_up (vLLM copies 16 GB host→VRAM and
  re-allocates the KV cache) is the floor on the restore side.
- Net for the shim: sleep 1 + direct restore ≈ 10.6 s vs 16.1 s with our full-checkpoint parallel path, at
  the price of app cooperation and ~11 s of sleep latency on the dump side.

### SDXL upstream run-1 dump failure (for the record)

`criu/util.c: sh exited, status=127` → `Iptables configuration failed` → `Failed to lock TCP connection`:
the toolkit image lacks `iptables`, which CRIU needs to lock established TCP connections (`--tcp-established`).
The first start still held an HTTPS session from the model download; later starts had no connection.
Fixed by adding `iptables` to `Dockerfile.vllm`.

## vLLM 0.30 + openai/gpt-oss-120b on the A100-80GB (`RUNS=1`, `--max-model-len 4096`, `--gpu-memory-utilization 0.9`)

MXFP4 weights (~63 GB) on a single GPU; vLLM 0.30 runs it on Ampere. Cold start ~170 s. Checkpoint again
fills the GPU: **71 GB of GPU pages + 3.5 GB of CPU pages** (75 GB total). Reference: NVIDIA's Dynamo Snapshot
blog reports 31.1 s restore for this model with upstream CRIU on a B200 (NFS).

| variant | dump (ms) | restore (ms) | GPU page fill | driver restore+unlock | inference |
|---|---|---|---|---|---|
| upstream-direct | 58,852 | 46,864 | — | — | OK (3.3 s) |
| parallel plugin, 16 threads | 65,902 | 17,227 | 3,714 ms (20.5 GB/s) | 10,565 ms | OK (3.2 s) |

Restore **−63%** (17.2 s vs 46.9 s). Same shape as Qwen3-8B: dump dominated by the driver's 37.3 s VRAM→host
copy; restore = 3.7 s fill + 10.6 s driver copy + ~3 s CRIU core.

### gpt-oss-120b + sleep level 1: not feasible on the A100-80GB with vLLM 0.30

`--enable-sleep-mode` forces the `cumem` allocator ("Enabling cumem allocator because sleep mode requires it"),
and weight loading then OOMs on the 80 GB GPU (`CUDA Error: out of memory at cumem_allocator.cpp:163`, free 4 MB of
85 GB) while the same model loads fine with the default allocator. Likely a transient second copy during the
MXFP4→Marlin repack that the caching allocator reuses in place. Reproduced twice with a verified-free GPU
(raw log: `raw/vllm_gptoss_sleep_oom.txt`). Would need a larger GPU (H200/B200).
