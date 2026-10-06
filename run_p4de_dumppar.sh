#!/bin/bash
# Session (box already set up by run_p4de_compress.sh): validate two plugin changes on vLLM + gpt-oss-120b.
#  - staging-page offload (custom storage off) with the parallel dump: CUDA_DUMP_THREADS 1 / 8 / 16
#  - custom storage on, with cuStreamGetCtx resolved through cuGetProcAddress (the _v2 ABI)
# Rebuilds the images from ./criu-src first. Status in ~/p4de-dp-status.txt, logs in ~/.
set -u
cd "$(dirname "$(readlink -f "$0")")"
S=~/p4de-dp-status.txt
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo "$1: $2" >> $S; }
git -C criu-src log --oneline -1 2>/dev/null || head -c 0 /dev/null
docker build -q --target criu-local -t criu-head-cz . > ~/build-cz2.log 2>&1; step criu-head-cz $?
docker build -q -f Dockerfile.vllm --target vllm-criu-local -t vllm-criu-cz . > ~/build-vllm-cz2.log 2>&1; step vllm-criu-cz $?

CS_ON="--plugin-option=cuda_plugin.custom-storage=on"; CS_OFF="--plugin-option=cuda_plugin.custom-storage=off"
log "=== smoke"
sudo rm -rf /mnt/nvme/dump_*
sudo env TENSOR_SIZE=5000 ZERO_SIZE=8000 RUNS=1 DROP_CACHE=no SCENARIOS="smoke-cs-on|criu-head-cz|$CS_ON;smoke-off-t8|criu-head-cz|$CS_OFF|||CUDA_DUMP_THREADS=8" ./bench_compare.sh > ~/dp_smoke.log 2>&1
grep -E "^RESULT|parallel dump|FAILED|Error" ~/dp_smoke.log | cut -c1-170; step smoke "$(grep -c 'success=True' ~/dp_smoke.log)/2"

log "=== vLLM gpt-oss-120b"
sudo rm -rf /mnt/nvme/dumpvllm_*
SC="gptoss-off-t8|vllm-criu-cz|$CS_OFF|||CUDA_DUMP_THREADS=8"
SC+=";gptoss-off-t16|vllm-criu-cz|$CS_OFF|||CUDA_DUMP_THREADS=16"
SC+=";gptoss-off-t1|vllm-criu-cz|$CS_OFF|||CUDA_DUMP_THREADS=1"
SC+=";gptoss-cs-on|vllm-criu-cz|$CS_ON"
sudo env RUNS=2 MODEL=openai/gpt-oss-120b MAX_MODEL_LEN=4096 READY_TIMEOUT=1500 SCENARIOS="$SC" ./bench_vllm.sh > ~/dp_vllm_gptoss.log 2>&1; step vllm-gptoss $?
grep -E "^RESULT|parallel dump|custom-storage .* copy|madvise|Driver API checkpoint: [0-9]{4,}" ~/dp_vllm_gptoss.log | cut -c1-200
echo DONE >> $S
log "=== all done"
