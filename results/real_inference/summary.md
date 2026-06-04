# Real Inference Benchmark Results

Checkpoint/restore timings measured on real inference workloads via `runc checkpoint` with the CUDA plugin. Times extracted from Kubernetes events (`Scaled down` = dump, `Restored` = restore).

**Hardware:** AWS g6.12xlarge — AMD EPYC 7R13 (24c/48t), 181 GiB RAM, 4× NVIDIA L4 (23 GiB), 4× NVMe RAID-0 (`/dev/md0`, XFS), Amazon Linux 2023 kernel 6.12, NVIDIA driver 580.159.03. Checkpoint images written to `/var/lib/zeropod` on `/dev/md0`.

## Variants tested

| Variant | Description |
|---------|-------------|
| **baseline** | `criu-dev` at commit `4d76d1acd`, before PR #3021 and #3022 |
| **upstream-optimized** | baseline + [PR #3021](https://github.com/checkpoint-restore/criu/pull/3021) (parallel memfd restore) + [PR #3022](https://github.com/checkpoint-restore/criu/pull/3022) (native AIO page reads) |
| **home-made** | baseline + [custom CUDA plugin](https://github.com/oOraph/criu/tree/fast_cuda_plugin_final) (GPU pages offloaded via O_DIRECT to `gpu-pages-*.img`) |
| **all** | baseline + PR #3021 + PR #3022 + custom CUDA plugin |

## Results

### huggingface-inference-toolkit + stabilityai/stable-diffusion-xl-base-1.0

| variant | dump (s) | restore (s) |
|---------|----------|-------------|
| baseline | 9.57 | 6.32 |
| upstream-optimized | 9.05 | 7.76 |
| home-made | 15.22 | 5.59 |
| all | 12.80 | 6.95 |

### vllm + meta-llama/Llama-3.1-8B-Instruct

| variant | dump (s) | restore 1st (s) | restore 2nd (s) | restore 3rd (s) |
|---------|----------|-----------------|-----------------|-----------------|
| baseline | 19.66 | 14.16 | — | — |
| upstream-optimized run 1 | 19.69 | 21.57 | — | — |
| upstream-optimized run 2 | 19.57 | 20.83 | 14.84 | — |
| upstream-optimized run 3 | 19.79 | 21.08 | 15.03 | 14.82 |
| home-made | 31.85 | 12.55 | — | — |
| all | 32.87 | 14.26 | — | — |

The first restore for upstream-optimized is consistently ~21s across all runs. Subsequent restores within the same run drop to ~14.8–15s, very close to the baseline (possible page cache warm-up effect).

### vllm + Qwen/Qwen3-8B

| variant | dump (s) | restore 1st (s) | restore 2nd (s) | restore 3rd (s) | restore 4th (s) |
|---------|----------|-----------------|-----------------|-----------------|-----------------|
| baseline | 20.86 | 14.49 | 15.31 | — | — |
| upstream-optimized | 19.87 | 14.97 | 15.07 | — | — |
| home-made run 1 | 33.19 | 15.23 | 14.51 | — | — |
| home-made run 2 | 33.59 | 13.99 | 15.05 | 15.08 | 14.21 |
| all | 33.57 | 13.53 | 14.64 | — | — |

## Analysis

### PR #3021 + #3022 (upstream-optimized): unexpected restore times on large models

The upstream-optimized variant shows notably higher **first** restore times for Llama-3.1-8B (~21s vs baseline 14.16s) and SDXL (+23%: 6.32s → 7.76s). Qwen3-8B is unaffected.

For Llama-3.1-8B, additional runs revealed that the elevated time only affects the **first restore** in a session — subsequent restores drop to ~14.8–15s, very close to baseline. This pattern is consistent across 3 independent runs, suggesting a cold-start effect (e.g. page cache warming) rather than a constant overhead. Whether this is specific to the AIO/O_DIRECT path introduced by PR #3022 or something else is unclear.

Since #3021 and #3022 were only tested together, the source of this difference cannot be attributed to either PR individually.

The `all` variant (PRs + custom plugin) shows Llama first-restore back at baseline level (14.26s ≈ 14.16s), which is an interesting data point but does not by itself explain what is happening in the upstream-optimized case.

### Home-made plugin: faster restore, significantly slower dump

The custom plugin consistently speeds up restore for SDXL and Llama (~10–12%) by offloading GPU pages out of CRIU core, but at the cost of doubling dump time (+59–62%) (this is the point of the optim: we move some memory regions twice at dump time, hopping for a faster restore after offloading the gpu pages on drive). For Qwen3-8B, the restore benefit is negligible though.

### Summary table (vs baseline)

| variant | workload | dump Δ | restore Δ |
|---------|----------|--------|-----------|
| upstream-optimized | SDXL | −5% | +23% ❓ |
| upstream-optimized | Llama-3.1-8B | +0% | +52% ❓ (1st restore only; subsequent ~+5%) |
| upstream-optimized | Qwen3-8B | −5% | +1% |
| home-made | SDXL | +59% | −12% |
| home-made | Llama-3.1-8B | +62% | −11% |
| home-made | Qwen3-8B | +59% | 0% |
| all | SDXL | +34% | +10% |
| all | Llama-3.1-8B | +67% | +1% |
| all | Qwen3-8B | +61% | −9% |

### Benchmark relevance and context

The [NVIDIA Dynamo Snapshot blog post](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes) — co-authored by the PR contributor — shows the following restore time progression for PRs #3021 and #3022 applied on top of upstream CRIU:

| Model | Upstream CRIU | + AIO (PR #3022) | + Parallel memfd (PR #3021) | Speedup |
|-------|--------------|-----------------|------------------------------|---------|
| Qwen3-0.6B | 6.8 s | 2.9 s | 2.4 s | 2.8× |
| Qwen3-8B | 24 s | 11 s | 4.7 s | 5.1× |
| gpt-oss-120b | 119 s | 54 s | 15 s | 7.9× |

The blog also describes a separate optimization (KV cache unmap via `cuMemUnmap` + `cuMemRelease` before checkpointing) that reduces the checkpoint from ~190 GiB to ~6 GiB for Qwen3-0.6B. This **requires application-side code changes** and does not satisfy a "bring your own code" / zero-modification constraint (unless I misunderstood the section).

### Conclusion

TODO: it is still unclear why we don't observe the announced performance boost from [PR #3021](https://github.com/checkpoint-restore/criu/pull/3021) and [PR #3022](https://github.com/checkpoint-restore/criu/pull/3022). Were our test or benchmark conditions somehow irrelevant? The restore regression observed for Llama-3.1-8B (+52%) warrants further investigation, including isolating which PR is responsible, if any, or if there is a bias/flaw in the benchmark.
