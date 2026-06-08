# vLLM Sleep/Wake-up Benchmark

Benchmark using sleep/wake-up cooperative checkpointing against a live vLLM process in a pod. Two scenarios are compared:

- **Standalone restore**: CRIU checkpoint taken while vLLM is running (no sleep call), direct restore
- **Sleep → checkpoint → restore → wake_up**: vLLM calls `/sleep?level=1` before snapshot, then `/wake_up` after restore

Times extracted from Kubernetes events + `curl` wall-clock measurements.

## Variants

| Variant | CRIU base | CUDA plugin | Source |
|---------|-----------|-------------|--------|
| **baseline** | `criu-dev` @ `4d76d1acd` | Upstream | https://github.com/oOraph/criu/tree/criu-dev |
| **baseline + plugin** | `criu-dev` @ `4d76d1acd` | Custom | https://github.com/oOraph/criu/tree/cuda-plugin-optim |
| **v4.2 + plugin** | `v4.2` | Custom | https://github.com/oOraph/criu/tree/v4.2-cuda-plugin-optim |
| **upstream-optimized** | baseline + [PR #3021](https://github.com/checkpoint-restore/criu/pull/3021) + [PR #3022](https://github.com/checkpoint-restore/criu/pull/3022) | Upstream | https://github.com/oOraph/criu/tree/optim1 |
| **all** | baseline + PR #3021 + PR #3022 + Plugin | Custom | https://github.com/oOraph/criu/tree/fast-cuda-2 |

## Results — meta-llama/Llama-3.1-8B-Instruct

### Standalone restore (no sleep)

| variant | dump (s) | restore (s) |
|---------|----------|-------------|
| baseline | 19.2 | 14.3 |
| baseline + plugin | 30.9 | 13.1 |
| v4.2 + plugin | 32.1 | 12.6 |
| upstream-optimized | 19.2 | 14.5 |
| all | 32.4 | 13.7 |

### Sleep → checkpoint → restore → wake_up

| variant | sleep (s) | dump (s) | restore (s) | wake_up (s) | restore + wake_up (s) |
|---------|-----------|----------|-------------|-------------|----------------------|
| baseline | 10.8 | 14.3 | 18.8 | ~2.0 | **~20.7** |
| baseline + plugin | 10.8 | 14.6 | 18.8 | ~2.0 | **~20.7** |
| v4.2 + plugin | ~11 | 14.8 | 18.9 | ~1.7 | **~20.6** |
| upstream-optimized | 10.7 | 14.1 | 13.4 | ~2.0 | **~15.3** |
| all | 10.8 | 14.6 | 13.1 | ~2.0 | **~15.1** |

## Results — Qwen/Qwen3-8B

### Standalone restore (no sleep)

| variant | dump (s) | restore (s) |
|---------|----------|-------------|
| baseline | 19.3 | 14.4 |
| baseline + plugin | 35.4 | 13.0 |
| v4.2 + plugin | 32.6 | 13.7 |
| upstream-optimized | 19.4 | 14.5 |
| all | 33.0 | 13.2 |

### Sleep → checkpoint → restore → wake_up

| variant | sleep (s) | dump (s) | restore (s) | wake_up (s) | restore + wake_up (s) |
|---------|-----------|----------|-------------|-------------|----------------------|
| baseline | 12.8 | 16.1 | 22.6 | ~2.0 | **~24.5** |
| baseline + plugin | 12.8 | 16.7 | 22.5 | ~2.0 | **~24.4** |
| v4.2 + plugin | ~13 | 16.0 | 22.6 | ~2.0 | **~24.5** |
| upstream-optimized | 12.8 | 16.0 | 15.8 | ~1.7 | **~17.8** |
| all | 12.9 | 16.2 | ~17–18.7 | ~2.0 | **~19–20.6** |

## Analysis

### Upstream PRs only benefit the sleep case

In the standalone restore scenario, all variants converge to ~13–14.5s restore — PR #3021 and #3022 make no meaningful difference.

In the sleep scenario, the upstream PRs deliver a significant improvement:
- **Llama**: 20.7s → 15.3s (−26%) with upstream-optimized
- **Qwen3-8B**: 24.5s → 17.8s (−27%) with upstream-optimized

This aligns with the PRs' design: the sleep procedure reduces GPU memory pressure (KV cache freed/flushed), leaving fewer dirty pages in `pages-*.img`. The AIO + parallel memfd optimizations have more leverage when the checkpoint is leaner.

### Home-made plugin brings no benefit in the sleep case

The plugin offloads GPU pages to `gpu-pages-*.img` at dump time, which helps standalone restore. But after a sleep call, the GPU memory footprint is already reduced — there is less to offload, and the restore + wake-up overhead (~18.9s + ~1.7s) is similar to the baseline sleep case. The plugin's advantage is neutralized.

### Sleep + restore is always slower than standalone restore

For all variants, the full sleep → checkpoint → restore → wake_up cycle is slower than a direct CRIU checkpoint + restore of a running vLLM:

| | Llama standalone restore | Llama sleep cycle (restore + wake_up) |
|-|--------------------------|--------------------------------------|
| baseline | 14.3s | 20.7s |
| upstream-optimized | 14.5s | 15.3s |
| all | 13.7s | 15.1s |

The cooperative sleep approach only makes sense if it unlocks other benefits (e.g. smaller checkpoint size reducing storage cost, faster dump via frozen snapshot on repeat cycles). As a pure latency optimization for restore, it is not competitive with direct CRIU checkpoint.

