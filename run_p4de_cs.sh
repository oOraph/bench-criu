#!/bin/bash
# End-to-end session on a p4de (single A100-80GB + 8× NVMe RAID-0 at 16 GB/s; the NVSwitch board only matters because it forces driver 615 + matching Fabric Manager), images, weights, then the
# custom-storage matrix. Run from ~/bench-criu on the box after `setup.sh` (SKIP_DRIVER=1) has installed
# Docker/toolkit and the NVMe array. Each phase appends to ~/p4de-cs-status.txt; logs in ~/.
#   CRIU_REF=upstream-cuda-custom-storage ./run_p4de_cs.sh
set -u
cd "$(dirname "$(readlink -f "$0")")"
S=~/p4de-cs-status.txt
CRIU_REF=${CRIU_REF:-upstream-cuda-custom-storage}
UPSTREAM_REF=${UPSTREAM_REF:-4485a86da237}
DRIVER=${DRIVER:-615.71.09}
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo "$1: $2" >> $S; }

log "=== driver $DRIVER + Fabric Manager"
./proto/install-driver-run.sh "$DRIVER" > ~/drv.log 2>&1; step driver $?
nvidia-smi --query-gpu=name,driver_version,persistence_mode --format=csv,noheader | head -1
nvidia-smi -q | grep -A1 '^ *Fabric' | head -2
docker run --rm --gpus '"device=0"' --entrypoint python criu-head-ours -c 'import torch; print("cuda ok", torch.cuda.get_device_name(0))' 2>/dev/null || log "(bench image not built yet, CUDA check deferred)"

log "=== fio"
sudo mkdir -p /mnt/nvme/fio && sudo chown -R ubuntu /mnt/nvme; RAW_DEV=/dev/nvme1n1 ./fio.sh > ~/fio.log 2>&1; step fio $?
grep -E "^===|READ: bw" ~/fio.log

log "=== images (CRIU bench + vLLM) and weights"
( docker pull -q vllm/vllm-openai:latest > ~/pull.log 2>&1 || docker pull -q vllm/vllm-openai:latest >> ~/pull.log 2>&1; step pull $?
  sudo mkdir -p /mnt/nvme/hf && sudo chown -R ubuntu /mnt/nvme/hf
  for m in Qwen/Qwen3-8B openai/gpt-oss-120b; do docker run --rm --entrypoint bash -v /mnt/nvme/hf:/root/.cache/huggingface vllm/vllm-openai:latest -c "hf download $m" > ~/dl-$(basename $m).log 2>&1; step "weights $m" $?; done ) &
docker build -q --target criu-upstream-head -t criu-upstream-head . > ~/build-upstream.log 2>&1; step criu-upstream-head $?
docker build -q --target criu-ref -t criu-head-cs --build-arg CRIU_REF=$CRIU_REF . > ~/build-cs.log 2>&1; step criu-head-cs $?
wait
docker build -q -f Dockerfile.vllm -t vllm-criu-upstream --build-arg CRIU_REPO=https://github.com/checkpoint-restore/criu.git --build-arg CRIU_REF=$UPSTREAM_REF . > ~/build-vllm-upstream.log 2>&1; step vllm-criu-upstream $?
docker build -q -f Dockerfile.vllm -t vllm-criu-cs --build-arg CRIU_REF=$CRIU_REF . > ~/build-vllm-cs.log 2>&1; step vllm-criu-cs $?
docker run --rm --gpus '"device=0"' --entrypoint python criu-head-cs -c 'import torch; print("cuda ok", torch.cuda.get_device_name(0))'

CS_ON="--plugin-option=cuda_plugin.custom-storage=on"; CS_OFF="--plugin-option=cuda_plugin.custom-storage=off"
log "=== smoke"
sudo rm -rf /mnt/nvme/dump_*
sudo env TENSOR_SIZE=5000 RUNS=1 DROP_CACHE=no SCENARIOS="smoke-cs-on|criu-head-cs|$CS_ON;smoke-cs-off|criu-head-cs|$CS_OFF;smoke-upstream|criu-upstream-head|--image-io-mode=direct" ./bench_compare.sh > ~/smoke.log 2>&1
grep -E "^RESULT|FAILED" ~/smoke.log; step smoke $(grep -c "SUCCESS=True" ~/smoke.log)

log "=== tensor 14.7 GB"
sudo rm -rf /mnt/nvme/dump_*
sudo env TENSOR_SIZE=60000 RUNS=2 DROP_CACHE=yes SCENARIOS="cs-on|criu-head-cs|$CS_ON;cs-off|criu-head-cs|$CS_OFF;upstream-direct|criu-upstream-head|--image-io-mode=direct" ./bench_compare.sh > ~/bench_tensor.log 2>&1; step tensor $?
grep -hE "^RESULT|custom-storage .* copy|parallel restore|restore\+unlock|Driver API checkpoint: [0-9]{4,}" ~/bench_tensor.log | cut -c1-160

log "=== vLLM gpt-oss-120b"
sudo rm -rf /mnt/nvme/dumpvllm_*
sudo env RUNS=2 MODEL=openai/gpt-oss-120b MAX_MODEL_LEN=4096 READY_TIMEOUT=1500 SCENARIOS="gptoss-cs-on|vllm-criu-cs|$CS_ON;gptoss-cs-off|vllm-criu-cs|$CS_OFF;gptoss-upstream-direct|vllm-criu-upstream|--image-io-mode=direct" ./bench_vllm.sh > ~/vllm_gptoss.log 2>&1; step vllm-gptoss $?
grep -E "^RESULT|custom-storage .* copy|parallel restore|Driver API (checkpoint|restore)" ~/vllm_gptoss.log | grep -vE ": 11 ms$|: [0-9]{3} ms$" | cut -c1-170

log "=== vLLM Qwen3-8B"
sudo rm -rf /mnt/nvme/dumpvllm_*
sudo env RUNS=2 SCENARIOS="qwen-cs-on|vllm-criu-cs|$CS_ON;qwen-cs-off|vllm-criu-cs|$CS_OFF" ./bench_vllm.sh > ~/vllm_qwen.log 2>&1; step vllm-qwen $?
grep -E "^RESULT" ~/vllm_qwen.log
echo DONE >> $S
log "=== all done"
