#!/bin/bash
# Session on a p4de (single A100-80GB, 8× NVMe RAID-0 at 16 GB/s, driver 615 + matching Fabric Manager):
# upstream CRIU memory compression (--compress / --compress-block + --decompress-threads) vs the
# custom-storage branch with zero-chunk skipping, on a zero-heavy tensor and on vLLM (gpt-oss-120b, Qwen3-8B).
# Prereqs: `SKIP_DRIVER=1 ./setup.sh` done, and the CRIU tree to test rsynced to ./criu-src, e.g.
#   rsync -a --delete --exclude .git ~/workspace_idea/hf/criu-zero/ <box>:bench-criu/criu-src/
# Each phase appends to ~/p4de-cz-status.txt; logs in ~/.
set -u
cd "$(dirname "$(readlink -f "$0")")"
S=~/p4de-cz-status.txt
UPSTREAM_REF=${UPSTREAM_REF:-4485a86da237}
DRIVER=${DRIVER:-615.71.09}
PHASES=${PHASES:-"fio synth gptoss qwen"}   # e.g. PHASES="gptoss" to go straight to vLLM + gpt-oss-120b
has() { [[ " $PHASES " == *" $1 "* ]]; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo "$1: $2" >> $S; }
[ -f criu-src/Makefile ] || { echo "criu-src/ missing: rsync the CRIU tree first"; exit 1; }

if ! nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | grep -q "^$DRIVER"; then
    log "=== driver $DRIVER + Fabric Manager"
    ./proto/install-driver-run.sh "$DRIVER" > ~/drv.log 2>&1; step driver $?
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
systemctl is-active nvidia-fabricmanager

if has fio; then
log "=== fio"
sudo mkdir -p /mnt/nvme/fio && sudo chown -R ubuntu /mnt/nvme; RAW_DEV=/dev/$(ls /sys/block/md0/slaves 2>/dev/null | head -1) ./fio.sh > ~/fio.log 2>&1; step fio $?
grep -E "^===|READ: bw" ~/fio.log
fi

log "=== images and weights"
( docker pull -q vllm/vllm-openai:latest > ~/pull.log 2>&1 || docker pull -q vllm/vllm-openai:latest >> ~/pull.log 2>&1; step pull $?
  sudo mkdir -p /mnt/nvme/hf && sudo chown -R ubuntu /mnt/nvme/hf
  for m in Qwen/Qwen3-8B openai/gpt-oss-120b; do docker run --rm --entrypoint bash -v /mnt/nvme/hf:/root/.cache/huggingface vllm/vllm-openai:latest -c "hf download $m" > ~/dl-$(basename $m).log 2>&1; step "weights $m" $?; done ) &
docker build -q --target criu-upstream-head -t criu-upstream-head . > ~/build-upstream.log 2>&1; step criu-upstream-head $?
docker build -q --target criu-local -t criu-head-cz . > ~/build-cz.log 2>&1; step criu-head-cz $?
wait
docker build -q -f Dockerfile.vllm --target vllm-criu-ref -t vllm-criu-upstream --build-arg CRIU_REPO=https://github.com/checkpoint-restore/criu.git --build-arg CRIU_REF=$UPSTREAM_REF . > ~/build-vllm-upstream.log 2>&1; step vllm-criu-upstream $?
docker build -q -f Dockerfile.vllm --target vllm-criu-local -t vllm-criu-cz . > ~/build-vllm-cz.log 2>&1; step vllm-criu-cz $?
docker run --rm --gpus '"device=0"' --entrypoint python criu-head-cz -c 'import torch; print("cuda ok", torch.cuda.get_device_name(0))'

# label|image|opts|dump_opts|restore_opts|env
D="--image-io-mode=direct"
mk() {  # mk <prefix> <upstream image> <ours image>
    local p=$1 up=$2 cz=$3
    echo "$p-up-lz4-256k-par|$up|$D|--compress-block 256K|--decompress-threads 0;$p-up-lz4-4k-par|$up|$D|--compress|--decompress-threads 0;$p-up-direct|$up|$D||;$p-up-lz4-256k-serial|$up|$D|--compress-block 256K|;$p-cs-zero|$cz|--plugin-option=cuda_plugin.custom-storage=on||;$p-cs-nozero|$cz|--plugin-option=cuda_plugin.custom-storage=on|||CUDA_CS_ZERO_SKIP=0"
}

if has synth; then
log "=== smoke (small tensor + zero tensor)"
sudo rm -rf /mnt/nvme/dump_*
sudo env TENSOR_SIZE=5000 ZERO_SIZE=16000 RUNS=1 DROP_CACHE=no SCENARIOS="$(mk smoke criu-upstream-head criu-head-cz)" ./bench_compare.sh > ~/cz_smoke.log 2>&1
grep -E "^RESULT|FAILED" ~/cz_smoke.log; step smoke "$(grep -c 'success=True' ~/cz_smoke.log)/6"

log "=== tensor 14.4 GB random + 14.4 GB three-quarters zero"
sudo rm -rf /mnt/nvme/dump_*
sudo env TENSOR_SIZE=60000 ZERO_SIZE=60000 RUNS=2 DROP_CACHE=yes SCENARIOS="$(mk tensor criu-upstream-head criu-head-cz)" ./bench_compare.sh > ~/cz_tensor.log 2>&1; step tensor $?
grep -hE "^RESULT|custom-storage .* copy" ~/cz_tensor.log | cut -c1-200
fi

if has gptoss; then
log "=== vLLM gpt-oss-120b"
sudo rm -rf /mnt/nvme/dumpvllm_*
sudo env RUNS=2 MODEL=openai/gpt-oss-120b MAX_MODEL_LEN=4096 READY_TIMEOUT=1500 RESTORE_TIMEOUT=900 SCENARIOS="$(mk gptoss vllm-criu-upstream vllm-criu-cz)" ./bench_vllm.sh > ~/cz_vllm_gptoss.log 2>&1; step vllm-gptoss $?
grep -E "^RESULT|custom-storage .* copy" ~/cz_vllm_gptoss.log | cut -c1-200
fi

if has qwen; then
log "=== vLLM Qwen3-8B"
sudo rm -rf /mnt/nvme/dumpvllm_*
sudo env RUNS=2 RESTORE_TIMEOUT=900 SCENARIOS="$(mk qwen vllm-criu-upstream vllm-criu-cz)" ./bench_vllm.sh > ~/cz_vllm_qwen.log 2>&1; step vllm-qwen $?
grep -E "^RESULT|custom-storage .* copy" ~/cz_vllm_qwen.log | cut -c1-200
fi
echo DONE >> $S
log "=== all done"
