# ZDTM CUDA tests on real GPUs, custom storage on/auto/off: p4de.24xlarge, driver 615.71.09 (2026-10-06)

CRIU's own regression suite (ZDTM) has seven CUDA tests in `test/zdtm/static`. Each starts a small CUDA program,
checkpoints and restores it with the cuda plugin (`zdtm.py run --cuda-checkpoint`), and checks after the restore
that its GPU state is intact. Run here on real A100s for both custom-storage modes, to cover the TODO of
[#3189](https://github.com/checkpoint-restore/criu/issues/3189).

| Component | Detail |
|---|---|
| Box | AWS `p4de.24xlarge`, 8× A100-SXM4-80GB, driver 615.71.09 + Fabric Manager, CUDA toolkit 13.4 |
| CRIU | `cuda-plugin-and-co` (custom-storage series, staging-page offload with parallel dump, zero skip) + `cuStreamGetCtx_v2` (local branch `zdtm-cs`, `6386f9aac`) |
| Modes | `plugin-option cuda_plugin.custom-storage=on`, `=auto` (= on with this driver) and `=off`, via `CRIU_CONFIG_FILE` |
| Script | `zdtm_cuda_matrix.sh` (setup notes in its header) |

## Results: 21 / 21 PASS

| test | what it checks | on | auto | off |
|---|---|---|---|---|
| `cuda00` | device memory written by a kernel keeps its values | PASS (`gpu-cs`) | PASS (`gpu-cs`) | PASS (`gpu-pages`) |
| `cuda_streams00` | work on several CUDA streams from several threads | PASS (`gpu-cs`) | PASS (`gpu-cs`) | PASS (`gpu-pages`) |
| `cuda_zerocopy00` | mapped host memory used directly by kernels | PASS (`gpu-cs`) | PASS (`gpu-cs`) | PASS (`gpu-pages`) |
| `cuda_mempool00` | stream-ordered allocations (`cudaMallocAsync` pools) | PASS (`gpu-cs`) | PASS (`gpu-cs`) | PASS (`gpu-pages`) |
| `cuda_graph00` | an instantiated CUDA graph still replays | PASS (`gpu-cs`) | PASS (`gpu-cs`) | PASS (`gpu-pages`) |
| `cuda_cublas00` | cuBLAS handles and their device memory | PASS (`gpu-cs`) | PASS (`gpu-cs`) | PASS (`gpu-pages`) |
| `cuda_multigpu00` | state on every visible GPU (all 8 here) | PASS (`gpu-cs`, 8 devices) | PASS (`gpu-cs`, 8 devices) | PASS (`gpu-pages`) |

The image type in each cell shows which path ran: `gpu-cs-<pid>.img` is the custom-storage image,
`gpu-pages-<pid>.img` the staging-page offload. With `auto`, every dump and restore logged its custom-storage
copies with `on` and `auto` (about 0.44 GB per GPU, 5–6 of 7 chunks zero); `cuda_multigpu00` logged 8 checkpoint and 8 restore
copies, one per GPU, the first multi-GPU run of custom storage.

Raw output per run in `raw/` (`zdtm-<mode>-<test>.log`), one-line summary per run in `zdtm-matrix.txt` (auto, off) and `zdtm-matrix-on.txt` (on).
