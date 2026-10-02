#!/bin/bash
# Compare the driver's own staging path (cuda-checkpoint) with the custom-storage prototype (cuda_cs)
# on the tensor test app, same GPU, same box. Runs the app inside the bench image (torch), drives the
# checkpoint from the host (libcuda from the host driver; pids are host pids).
#   TENSOR_SIZE=60000 ./test_cs.sh            # ~14.7 GB of VRAM
set -eu -o pipefail
TENSOR_SIZE=${TENSOR_SIZE:-60000}
IMAGE=${IMAGE:-criu-head-ours}
NVME=${BENCH_DIR:-/mnt/nvme}
C=cstest
DROP_CACHE=${DROP_CACHE:-yes}
CS=$(dirname "$(readlink -f "$0")")/cuda_cs
log() { echo "[$(date '+%H:%M:%S')] $*"; }
drop_caches() { [[ "$DROP_CACHE" == yes ]] && { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; } || true; }
# The tools run INSIDE the container (docker exec) with the container-ns pid printed by the app itself.
# Do not derive the pid from `docker top`/pgrep: the `bash -c` wrapper also matches "test_app.py" and
# has no CUDA state, and the checkpoint API answers CUDA_ERROR_NOT_INITIALIZED for such a pid.
start_app() {
    docker rm -f $C >/dev/null 2>&1 || true
    docker run -d --rm --name $C --gpus '"device=0"' -v "$NVME:$NVME" "$IMAGE" >/dev/null
    docker cp "$CS" $C:/usr/local/bin/cuda_cs
    docker exec -e TENSOR_SIZE=$TENSOR_SIZE $C bash -c "rm -f /tmp/go; touch /tmp/app.log && nohup python /test_app.py >> /tmp/app.log 2>&1 &"
    for i in $(seq 1 120); do docker exec $C grep -q READY /tmp/app.log 2>/dev/null && break; sleep 1; done
    APP_PID=$(docker exec $C grep -o 'pid=[0-9]*' /tmp/app.log | head -1 | cut -d= -f2)   # container-ns pid
    log "app ready, container pid=$APP_PID, VRAM: $(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader)"
}
rss_gb() { docker exec $C ps -o rss= -p $APP_PID | awk '{printf "%.1f GB", $1/1e6}'; }
finish_app() {
    docker exec $C touch /tmp/go; for i in $(seq 1 30); do docker exec $C grep -q "SUCCESS" /tmp/app.log 2>/dev/null && break; sleep 1; done
    docker exec $C grep -E "x_match|SUCCESS" /tmp/app.log | tail -2
}

echo "=== A. driver staging path: cuda-checkpoint lock+checkpoint / restore+unlock (VRAM <-> target host memory)"
start_app
t0=$(date +%s%N); docker exec $C cuda-checkpoint --action lock --pid $APP_PID; docker exec $C cuda-checkpoint --action checkpoint --pid $APP_PID; t1=$(date +%s%N)
log "cuda-checkpoint checkpoint: $(( (t1-t0)/1000000 )) ms (VRAM now in host RAM, RSS: $(rss_gb))"
drop_caches
t0=$(date +%s%N); docker exec $C cuda-checkpoint --action restore --pid $APP_PID; docker exec $C cuda-checkpoint --action unlock --pid $APP_PID; t1=$(date +%s%N)
log "cuda-checkpoint restore+unlock: $(( (t1-t0)/1000000 )) ms"
finish_app; docker rm -f $C >/dev/null

echo; echo "=== B. custom storage: cuda_cs checkpoint -> $NVME/cs.img -> cuda_cs restore"
start_app
sudo rm -f $NVME/cs.img
t0=$(date +%s%N); docker exec $C cuda_cs checkpoint $APP_PID $NVME/cs.img; t1=$(date +%s%N)
log "cuda_cs checkpoint total: $(( (t1-t0)/1000000 )) ms, image $(sudo du -sh $NVME/cs.img | cut -f1), app RSS: $(rss_gb), VRAM: $(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader)"
drop_caches
t0=$(date +%s%N); docker exec $C cuda_cs restore $APP_PID $NVME/cs.img; t1=$(date +%s%N)
log "cuda_cs restore total: $(( (t1-t0)/1000000 )) ms"
finish_app; docker rm -f $C >/dev/null
