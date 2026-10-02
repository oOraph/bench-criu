#!/bin/bash
# Real-inference benchmark: checkpoint/restore a running huggingface-inference-toolkit server
# (default: stabilityai/stable-diffusion-xl-base-1.0, text-to-image), Docker-based.
# Images: Dockerfile.vllm with APP_IMAGE=<toolkit image> (adds CRIU), e.g.
#   docker build -f Dockerfile.vllm -t sdxl-criu-ours --build-arg APP_IMAGE=raphael31415/huggingface-inference-toolkit:gpu-2 --build-arg CRIU_REF=fast_cuda_plugin_on_head .
# Scenarios: "label|image|extra criu options" (semicolon separated).
set -eu -o pipefail

HF_MODEL_ID=${HF_MODEL_ID:-stabilityai/stable-diffusion-xl-base-1.0}
HF_TASK=${HF_TASK:-text-to-image}
NVME_BASE=${BENCH_DIR:-/mnt/nvme}
HF_CACHE=${HF_CACHE:-/mnt/nvme/hf}
CONTAINER=benchsdxl
PORT=5000
RUNS=${RUNS:-2}
DROP_CACHE=${DROP_CACHE:-"yes"}
READY_TIMEOUT=${READY_TIMEOUT:-900}
RESTORE_TIMEOUT=${RESTORE_TIMEOUT:-300}
CRIU_BASE_OPTS=${CRIU_BASE_OPTS:-"--shell-job --skip-in-flight --file-locks --ghost-limit 10485760 --tcp-established --link-remap"}
DEFAULT_SCENARIOS="upstream-direct|sdxl-criu-upstream|--image-io-mode=direct"
DEFAULT_SCENARIOS+=";ours|sdxl-criu-ours|"
DEFAULT_SCENARIOS+=";parallel|sdxl-criu-parallel|"
SCENARIOS=${SCENARIOS:-$DEFAULT_SCENARIOS}
export CUDA_RESTORE_THREADS=${CUDA_RESTORE_THREADS:-16}

log() { echo "[$(date '+%H:%M:%S')] $*"; }
drop_caches() {
    if [[ "${DROP_CACHE,,}" =~ ^(yes|true|1)$ ]]; then
        if findmnt -n -o FSTYPE "$NVME_BASE" 2>/dev/null | grep -q tmpfs; then log "skip drop_caches (tmpfs)";
        else log "drop caches"; sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null; fi
    else log "skip drop caches (deactivated)"; fi
}
cleanup_container() { docker rm -f $CONTAINER 2>/dev/null || true; }
# A CRIU-restored tree is re-parented into its recorded cgroup, so `docker rm -f` may not kill it and it
# can hold the GPU for a while after the container is gone (seen: next round OOM'd at startup). Kill any
# leftover server processes on the host and wait for GPU 0 to be released before starting a round.
wait_gpu_free() {
    sudo pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sudo pkill -f "VLLM::EngineCore" 2>/dev/null
    sudo pkill -f "gunicorn webservice_starlette" 2>/dev/null; sudo pkill -f "uvicorn.workers.UvicornWorker" 2>/dev/null
    local i used
    for i in $(seq 1 120); do
        used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [[ -n "$used" && "$used" -lt 1024 ]] && return 0
        sleep 1
    done
    log "WARNING: GPU 0 still has ${used} MiB in use after 120s"
}
curl_health() { docker exec $CONTAINER curl -sf http://localhost:${PORT}/health -o /dev/null 2>/dev/null; }

run_one() {
    local label=$1 image=$2 run=$3 criu_opts=${4:-}
    local dump_dir=$NVME_BASE/dumpsdxl_${label}_$run
    cleanup_container
    wait_gpu_free
    sudo rm -rf "$dump_dir" && sudo mkdir -p "$dump_dir"

    log "[$label run=$run] starting container image=$image model=$HF_MODEL_ID"
    # The toolkit downloads HF_MODEL_ID into HF_MODEL_DIR (/opt/huggingface) at startup; mount a
    # persistent per-model dir there so only the first start downloads (same as the June k8s setup).
    local model_dir="$HF_CACHE/toolkit/$HF_MODEL_ID"; sudo mkdir -p "$model_dir"
    docker run -d --rm --name $CONTAINER --gpus '"device=0"' --shm-size=8g \
        -e HF_MODEL_ID="$HF_MODEL_ID" -e HF_TASK="$HF_TASK" -e LOG_LEVEL=INFO \
        -e UV_USE_IO_URING=0 -e DO_NOT_TRACK=1 -e GLOO_SOCKET_IFNAME=lo \
        -v "$model_dir:/opt/huggingface" -v "$dump_dir:$dump_dir" "$image" >/dev/null
    sleep 1
    # all stdio fds on a file inside the container's own fs (CRIU needs resolvable mounts)
    docker exec $CONTAINER bash -c "touch /tmp/server.log && /app/entrypoint.sh </tmp/server.log >>/tmp/server.log 2>&1 &"

    log "[$label run=$run] waiting for /health (timeout ${READY_TIMEOUT}s)"
    local t_start=$(date +%s) i
    for i in $(seq 1 $READY_TIMEOUT); do curl_health && break; sleep 1; done
    if ! curl_health; then
        log "[$label run=$run] TIMEOUT: server not up; last logs:"; docker exec $CONTAINER tail -20 /tmp/server.log || true
        echo "RESULT label=$label run=$run coldstart_s=FAILED dump_ms=FAILED restore_ms=FAILED"; cleanup_container; return 1
    fi
    local coldstart_s=$(( $(date +%s) - t_start ))
    log "[$label run=$run] server READY after ${coldstart_s}s"

    # dump the process owning the listening socket (gunicorn/uvicorn master)
    local app_pid init_pid
    init_pid=$(docker inspect -f '{{.State.Pid}}' $CONTAINER)
    app_pid=$(docker exec $CONTAINER bash -c "netstat -nltp 2>/dev/null | awk -v p=':${PORT} ' '\$0 ~ p {print \$NF}' | cut -d/ -f1 | head -1")
    [[ -z "$app_pid" ]] && app_pid=$(docker exec $CONTAINER pgrep -o -f "entrypoint|uvicorn|gunicorn|python")
    log "[$label run=$run] app_pid=$app_pid (container ns) init_pid=$init_pid; tree:"
    docker exec $CONTAINER ps -o pid,ppid,rss,comm --forest 2>/dev/null | grep -vE "ps$|bash|sleep|tini" | head -8 || true

    log "[$label run=$run] dump start"
    local t0=$(( $(date +%s%N) / 1000000 )) dump_rc=0
    nsenter -n -m -u -p -i -t "$init_pid" -- \
        criu dump $CRIU_BASE_OPTS $criu_opts -D "$dump_dir" -t "$app_pid" -v3 -o dump.log > "$dump_dir/criu-dump.out" 2>&1 || dump_rc=$?
    local dump_ms=$(( $(date +%s%N) / 1000000 - t0 ))
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
        if curl_health; then restore_ms=$(( $(date +%s%N) / 1000000 - t0 )); break; fi
        sleep 0.1
    done
    disown $restore_pid 2>/dev/null || true
    sudo grep -hE 'timing|Error' "$dump_dir/restore.log" | grep -v "tun.c" | head -15 || true
    log "[$label run=$run] restore=${restore_ms}ms"

    local infer=FAILED infer_ms=n/a
    if [[ "$restore_ms" != "TIMEOUT" ]]; then
        local ti=$(( $(date +%s%N) / 1000000 ))
        if docker exec $CONTAINER curl -sf http://localhost:${PORT} -H 'content-type: application/json' -H 'accept: image/jpeg' \
              -d '{"inputs": "a tiger in the snow"}' --output /tmp/infer_out.jpg 2>/dev/null; then
            infer_ms=$(( $(date +%s%N) / 1000000 - ti ))
            docker exec $CONTAINER bash -c 'head -c 3 /tmp/infer_out.jpg | od -An -tx1' | grep -q 'ff d8 ff' && infer=OK || infer=WRONG_FORMAT
        fi
        log "[$label run=$run] inference=$infer infer_ms=${infer_ms}"
    else
        sudo cat "$dump_dir/criu-restore.out" 2>/dev/null | head -10; sudo tail -30 "$dump_dir/restore.log" 2>/dev/null || true
    fi
    echo "RESULT label=$label run=$run coldstart_s=$coldstart_s dump_ms=$dump_ms restore_ms=$restore_ms inference=$infer infer_ms=$infer_ms"
    cleanup_container
}

log "=== SDXL/inference-toolkit benchmark: MODEL=$HF_MODEL_ID RUNS=$RUNS DROP_CACHE=$DROP_CACHE CUDA_RESTORE_THREADS=$CUDA_RESTORE_THREADS"
IFS=';' read -ra SCENARIO_LIST <<< "$SCENARIOS"
for sc in "${SCENARIO_LIST[@]}"; do
    IFS='|' read -r label image opts <<< "$sc"
    echo; echo "=== SCENARIO $label: image=$image opts='$opts' ==="
    for r in $(seq 1 $RUNS); do run_one "$label" "$image" $r "$opts" || true; echo; done
done
log "=== All SDXL benchmarks complete ==="
