#!/bin/bash
# Real-inference benchmark: checkpoint/restore a running `vllm serve` with CRIU, Docker-based.
# Mirrors the June 2026 Kubernetes measurements (results/real_inference) on a bench box.
#
# Scenarios: "label|image|extra criu options" (semicolon separated), images from Dockerfile.vllm.
# Restore time = from `criu restore` start until /health answers; then a completion request
# validates the restored engine.
set -eu -o pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
NVME_BASE=${BENCH_DIR:-/mnt/nvme}
HF_CACHE=${HF_CACHE:-/mnt/nvme/hf}
CONTAINER=benchvllm
PORT=8000
RUNS=${RUNS:-2}
DROP_CACHE=${DROP_CACHE:-"yes"}
READY_TIMEOUT=${READY_TIMEOUT:-900}
RESTORE_TIMEOUT=${RESTORE_TIMEOUT:-300}
# CRIU options the shim uses for vLLM workloads
# (+ --tcp-established --link-remap per the vLLM recipe that worked on k8s, see README)
CRIU_BASE_OPTS=${CRIU_BASE_OPTS:-"--shell-job --skip-in-flight --file-locks --ghost-limit 10485760 --tcp-established --link-remap"}
DEFAULT_SCENARIOS="upstream-direct|vllm-criu-upstream|--image-io-mode=direct"
DEFAULT_SCENARIOS+=";ours|vllm-criu-ours|"
DEFAULT_SCENARIOS+=";parallel|vllm-criu-parallel|"
SCENARIOS=${SCENARIOS:-$DEFAULT_SCENARIOS}
export CUDA_RESTORE_THREADS=${CUDA_RESTORE_THREADS:-16}   # inherited by criu via nsenter

log() { echo "[$(date '+%H:%M:%S')] $*"; }
drop_caches() {
    if [[ "${DROP_CACHE,,}" =~ ^(yes|true|1)$ ]]; then
        if findmnt -n -o FSTYPE "$NVME_BASE" 2>/dev/null | grep -q tmpfs; then log "skip drop_caches (tmpfs)";
        else log "drop caches"; sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null; fi
    else log "skip drop caches (deactivated)"; fi
}
cleanup_container() { docker rm -f $CONTAINER 2>/dev/null || true; }
curl_health() { docker exec $CONTAINER curl -sf http://localhost:${PORT}/health -o /dev/null 2>/dev/null; }

run_one() {
    local label=$1 image=$2 run=$3 criu_opts=${4:-}
    local dump_dir=$NVME_BASE/dumpvllm_${label}_$run
    cleanup_container
    sudo rm -rf "$dump_dir" && sudo mkdir -p "$dump_dir"

    log "[$label run=$run] starting container image=$image model=$MODEL"
    # Env that makes a vLLM server dumpable (recipe from the k8s runs):
    #  UV_USE_IO_URING=0        uvloop in the API server would use io_uring, which CRIU can't dump
    #  HF_HUB_OFFLINE / VLLM_NO_USAGE_STATS / DO_NOT_TRACK  no lingering HTTPS sessions to the Hub
    #  GLOO_SOCKET_IFNAME=lo    Gloo groups (created even single-GPU) must not bind the container IP
    #  TORCH_NCCL_*=0           no NCCL monitoring threads / dump-on-timeout
    docker run -d --rm --name $CONTAINER --gpus '"device=0"' --shm-size=16g \
        -e UV_USE_IO_URING=0 -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
        -e GLOO_SOCKET_IFNAME=lo -e TORCH_NCCL_ENABLE_MONITORING=0 -e TORCH_NCCL_DUMP_ON_TIMEOUT=0 \
        -e VLLM_LOGGING_LEVEL=INFO \
        -v "$HF_CACHE:/root/.cache/huggingface" -v "$dump_dir:$dump_dir" "$image" >/dev/null
    sleep 1
    # all stdio fds on a file inside the container's own fs (CRIU needs resolvable mounts)
    docker exec $CONTAINER bash -c "touch /tmp/server.log && python3 -m vllm.entrypoints.openai.api_server --model $MODEL --port $PORT --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL </tmp/server.log >>/tmp/server.log 2>&1 &"

    log "[$label run=$run] waiting for /health (timeout ${READY_TIMEOUT}s)"
    local t_start=$(date +%s) i
    for i in $(seq 1 $READY_TIMEOUT); do curl_health && break; sleep 1; done
    if ! curl_health; then
        log "[$label run=$run] TIMEOUT: server not up; last logs:"; docker exec $CONTAINER tail -20 /tmp/server.log || true
        echo "RESULT label=$label run=$run coldstart_s=FAILED dump_ms=FAILED restore_ms=FAILED"; cleanup_container; return 1
    fi
    local coldstart_s=$(( $(date +%s) - t_start ))
    log "[$label run=$run] server READY after ${coldstart_s}s"

    local app_pid init_pid
    init_pid=$(docker inspect -f '{{.State.Pid}}' $CONTAINER)
    app_pid=$(docker exec $CONTAINER pgrep -o -f "vllm.entrypoints.openai.api_server")
    log "[$label run=$run] app_pid=$app_pid (container ns) init_pid=$init_pid; tree:"
    docker exec $CONTAINER ps -o pid,ppid,rss,comm --forest 2>/dev/null | grep -vE "ps|bash|sleep|tini" | head -8 || true

    log "[$label run=$run] dump start"
    local t0=$(( $(date +%s%N) / 1000000 )) dump_rc=0
    nsenter -n -m -u -p -i -t "$init_pid" -- \
        criu dump $CRIU_BASE_OPTS $criu_opts -D "$dump_dir" -t "$app_pid" -v3 -o dump.log > "$dump_dir/criu-dump.out" 2>&1 || dump_rc=$?
    local dump_ms=$(( $(( $(date +%s%N) / 1000000 )) - t0 ))
    sudo grep -hE 'timing|Error' "$dump_dir/dump.log" | grep -v "tun.c" | head -15 || true
    local gpu_sz pages_sz
    gpu_sz=$(sudo du -shc "$dump_dir"/gpu-pages-*.img 2>/dev/null | tail -1 | awk '{print $1}' || echo none)
    pages_sz=$(sudo du -shc "$dump_dir"/pages-*.img 2>/dev/null | tail -1 | awk '{print $1}' || echo none)
    log "[$label run=$run] dump=${dump_ms}ms rc=$dump_rc gpu-pages=${gpu_sz:-none} pages-*.img=${pages_sz:-none}"
    if [[ $dump_rc -ne 0 || ! -f "$dump_dir/inventory.img" ]]; then
        sudo cat "$dump_dir/criu-dump.out" 2>/dev/null | head -10; sudo tail -30 "$dump_dir/dump.log" 2>/dev/null || true
        echo "RESULT label=$label run=$run coldstart_s=$coldstart_s dump_ms=$dump_ms restore_ms=FAILED"; cleanup_container; return 1
    fi

    drop_caches
    log "[$label run=$run] restore start"
    t0=$(( $(date +%s%N) / 1000000 ))
    nsenter -n -m -u -p -i -t "$init_pid" -- \
        bash -c "criu restore $CRIU_BASE_OPTS $criu_opts -D $dump_dir --manage-cgroups -v3 -o restore.log" > "$dump_dir/criu-restore.out" 2>&1 &
    local restore_pid=$! restore_ms=TIMEOUT
    for i in $(seq 1 $((RESTORE_TIMEOUT*10))); do
        if curl_health; then restore_ms=$(( $(( $(date +%s%N) / 1000000 )) - t0 )); break; fi
        sleep 0.1
    done
    disown $restore_pid 2>/dev/null || true
    sudo grep -hE 'timing|Error' "$dump_dir/restore.log" | grep -v "tun.c" | head -15 || true
    log "[$label run=$run] restore=${restore_ms}ms"

    local infer=FAILED infer_ms=n/a
    if [[ "$restore_ms" != "TIMEOUT" ]]; then
        local ti=$(( $(date +%s%N) / 1000000 )) out
        out=$(docker exec $CONTAINER curl -sf http://localhost:${PORT}/v1/completions -H 'content-type: application/json' \
              -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" 2>/dev/null || true)
        infer_ms=$(( $(( $(date +%s%N) / 1000000 )) - ti ))
        echo "$out" | grep -q '"text"' && infer=OK
        log "[$label run=$run] inference=$infer infer_ms=${infer_ms} text=$(echo "$out" | grep -o '"text":"[^"]*"' | head -1)"
    else
        sudo cat "$dump_dir/criu-restore.out" 2>/dev/null | head -10; sudo tail -30 "$dump_dir/restore.log" 2>/dev/null || true
    fi
    echo "RESULT label=$label run=$run coldstart_s=$coldstart_s dump_ms=$dump_ms restore_ms=$restore_ms inference=$infer infer_ms=$infer_ms"
    cleanup_container
}

log "=== vLLM benchmark: MODEL=$MODEL RUNS=$RUNS DROP_CACHE=$DROP_CACHE CUDA_RESTORE_THREADS=$CUDA_RESTORE_THREADS CRIU_BASE_OPTS='$CRIU_BASE_OPTS'"
IFS=';' read -ra SCENARIO_LIST <<< "$SCENARIOS"
for sc in "${SCENARIO_LIST[@]}"; do
    IFS='|' read -r label image opts <<< "$sc"
    echo; echo "=== SCENARIO $label: image=$image opts='$opts' ==="
    for r in $(seq 1 $RUNS); do run_one "$label" "$image" $r "$opts" || true; echo; done
done
log "=== All vLLM benchmarks complete ==="
