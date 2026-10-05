# Custom storage on HGX: p4de.24xlarge, driver 615.71.09 + Fabric Manager 615 (2026-10-05)

## Setup

| Component | Detail |
|---|---|
| Instance | AWS `p4de.24xlarge` (open CR, us-east-1c), 8× A100-SXM4-80GB (GPU 0 used), 8× NVMe RAID-0 = 16.1 GB/s |
| Driver | **615.71.09 proprietary** via NVIDIA's `.run` + **`nvidia-fabricmanager` 615.71.09-2ubuntu1 from NVIDIA's CUDA apt repo** (`proto/install-driver-run.sh`); FM log: "Successfully configured all the available GPUs and NVSwitches" |
| Images | `criu-head-cs` / `vllm-criu-cs` built from branch `upstream-cuda-custom-storage` (the clean PR branch); `criu-upstream-head` / `vllm-criu-upstream` from criu-dev 4485a86da |
| Driver | `run_p4de_cs.sh` (one-shot session), `RUNS=2`, `DROP_CACHE=yes` |

## Tensor 14.9 GB (`TENSOR_SIZE=60000`)

| path | dump | restore | GPU copy |
|---|---|---|---|
| custom storage on | **2,484 / 2,489 ms** | **2,988 / 2,874 ms** | restore copy 1,636 / 1,617 ms (9.1–9.2 GB/s) |
| parallel plugin (cs off) | 12,959 / 13,009 ms | 3,854 / 3,966 ms | fill 895 / 851 ms (16.7–17.5 GB/s) + driver restore 2,126 / 2,275 ms |
| upstream head, direct | 11,225 / 11,236 ms | 9,010 / 8,870 ms | — |

## vLLM 0.30 + gpt-oss-120b (76.2 GB of GPU memory mapped + 3.7 GB CPU pages, inference validated)

| path | run | dump (ms) | restore (ms) | GPU copy |
|---|---|---|---|---|
| custom storage on | 1 | **11,481** | **10,556** | ckpt copy 8,534 ms (8.9 GB/s); restore copy 7,088 ms (10.8 GB/s) |
| custom storage on | 2 | **11,497** | **10,553** | 8,633 ms (8.8 GB/s); 7,081 ms (10.8 GB/s) |
| parallel plugin (cs off) | 1 | 66,699 | 16,914 | driver checkpoint 37,999 ms; fill 3,547 ms (21.5 GB/s) + driver restore 10,352 ms |

(remaining rounds appended below when the session completes)

## Reading

- The driver's own copies are gone: no 38 s VRAM→host copy on dump, no 10.4 s host→VRAM copy on restore.
- Restore 16.9 s → **10.6 s** (−38%): 7.1 s of copy into the mapping at 10.8 GB/s + ~3.1 s of CRIU (process
  tree, 3.7 GB of CPU pages). Dump 66.7 s → **11.5 s** (−83%): one disk-write-bound pass.
- The copy rate into the mapping on the A100 (10.8 GB/s H2D, 8.9 GB/s D2H with 4 threads) matches the A10G
  plateau (~12 GB/s); the array (16 GB/s) is not the limit. The remaining restore lever is CRIU core (~3 s);
  the remaining copy lever is the mapping's rate (copy-engine/NUMA placement experiments pending).
- Against upstream criu-dev on this workload: restore 46.9 s (2026-10-02) → 10.6 s, dump 58.9 s → 11.5 s.
