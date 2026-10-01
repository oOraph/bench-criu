#!/bin/bash
# Comparative profiling: test_app+new_criu vs test_app+orig_criu vs test_app+cuda_optimized_plugin
# Storage: NVMe (/mnt/nvme)
set -eu -o pipefail

TENSOR_SIZE=${TENSOR_SIZE:-60000}
NVME_BASE=${BENCH_DIR:-/mnt/nvme}
CONTAINER=bench1
# Scenarios: "label|image|extra criu options (passed to both dump and restore)".
# Override with SCENARIOS="a|img|opts;b|img2|opts2" (semicolon separated).
# Note: since upstream PR #3066, O_DIRECT/AIO page reads are OFF unless --image-io-mode=direct.
DEFAULT_SCENARIOS="v42-ours|criu-v42-ours|"
DEFAULT_SCENARIOS+=";upstream|criu-upstream-head|"
DEFAULT_SCENARIOS+=";upstream-direct|criu-upstream-head|--image-io-mode=direct"
DEFAULT_SCENARIOS+=";upstream-cli-direct|criu-upstream-head|--image-io-mode=direct --plugin-option=cuda_plugin.backend=cuda-checkpoint"
SCENARIOS=${SCENARIOS:-$DEFAULT_SCENARIOS}
RUNS=${RUNS:-2}
DROP_CACHE=${DROP_CACHE:-"yes"}

log() { echo "[$(date '+%H:%M:%S')] $*"; }
# Only drop caches on real block devices — tmpfs IS the page cache, dropping it destroys data
drop_caches() {
    if [[ "${DROP_CACHE,,}" =~ ^(yes|true|1)$ ]];then
        if findmnt -n -o FSTYPE "$NVME_BASE" 2>/dev/null | grep -q tmpfs; then
            log "skip drop_caches (tmpfs — data lives in RAM)"
        else
            log "drop caches"
            sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
        fi
    else
      log "skip drop caches (deactivated)"
    fi
}
cleanup_container() { docker rm -f $CONTAINER 2>/dev/null || true; }

run_plugin() {
    local label=$1
    local image=$2
    local run=$3
    local criu_opts=${4:-}
    local dump_dir=$NVME_BASE/dump_${label}_$run

    cleanup_container
    sudo rm -rf "$dump_dir" && sudo mkdir -p "$dump_dir"
    local outfile=$dump_dir/app_out.txt

    log "[$label run=$run] starting container image=$image (TENSOR_SIZE=$TENSOR_SIZE)"
    docker run -d --rm --name $CONTAINER --gpus '"device=0"' \
        -v "$dump_dir:$dump_dir" \
        "$image"

    docker exec -e TENSOR_SIZE=$TENSOR_SIZE $CONTAINER bash -c \
        "touch $outfile && nohup python /test_app.py >> $outfile 2>&1 &"

    for i in $(seq 1 60); do grep -q "READY" "$outfile" 2>/dev/null && break; sleep 1; done
    cat "$outfile"

    local app_pid container_init_pid
    app_pid=$(grep -o 'pid=[0-9]*' "$outfile" | head -1 | cut -d= -f2)
    container_init_pid=$(docker inspect -f '{{.State.Pid}}' $CONTAINER)

    log "[$label run=$run] dump start"
    local t0; t0=$(( $(date +%s%N) / 1000000 ))
    local dump_log dump_rc
    dump_log=$(nsenter -n -m -u -p -i -t "$container_init_pid" -- \
        criu dump --shell-job --skip-in-flight $criu_opts -D "$dump_dir" -t "$app_pid"  2>&1) && dump_rc=0 || dump_rc=$?
    local dump_ms=$(( $(( $(date +%s%N) / 1000000 )) - t0 ))
    echo "$dump_log" | grep -E 'timing|Error|Warn|Err' || true
    if [ $dump_rc -ne 0 ]; then
        log "[$label run=$run] DUMP FAILED (rc=$dump_rc) — full output:"
        echo "$dump_log" | tail -40
    fi

    local gpu_sz pages_sz inv_ok
    gpu_sz=$(ls -lh "$dump_dir/gpu-pages-${app_pid}.img" 2>/dev/null | awk '{print $5}' || echo none)
    pages_sz=$(ls "$dump_dir"/pages-*.img 2>/dev/null | xargs du -shc 2>/dev/null | tail -1 | awk '{print $1}' || echo none)
    inv_ok=$([ -f "$dump_dir/inventory.img" ] && echo yes || echo NO)
    log "[$label run=$run] dump=${dump_ms}ms  gpu-pages=$gpu_sz  pages-*.img=$pages_sz  inventory=$inv_ok"

    if [ "$inv_ok" != "yes" ]; then
        log "[$label run=$run] SKIP restore — dump incomplete (no inventory.img)"
        echo "RESULT label=$label run=$run dump_ms=$dump_ms restore_ms=FAILED dump_rc=$dump_rc"
        cleanup_container
        return 1
    fi

    # Reuse the same container (CUDA already initialized) — same as manual nsenter test
    # Add memlock unlimited for mlock in restore path
    docker update --ulimit memlock=-1 $CONTAINER 2>/dev/null || true
    drop_caches

    log "[$label run=$run] restore start"
    local pre_size; pre_size=$(wc -c < "$outfile" 2>/dev/null || echo 0)
    t0=$(( $(date +%s%N) / 1000000 ))
    nsenter -n -m -u -p -i -t "$container_init_pid" -- \
        bash -c "touch /tmp/go && criu restore --shell-job -D $dump_dir --manage-cgroups --skip-in-flight $criu_opts -v3" \
        > "$dump_dir/restore.log" 2>&1 &
    local restore_pid=$!

    # Measure time until test_app writes new output (= process is restored and running)
    local restore_ms=TIMEOUT
    for i in $(seq 1 900); do
        local cur_size; cur_size=$(wc -c < "$outfile" 2>/dev/null || echo 0)
        if (( cur_size > pre_size )); then
            restore_ms=$(( $(( $(date +%s%N) / 1000000 )) - t0 ))
            break
        fi
        sleep 0.1
    done
    wait $restore_pid 2>/dev/null || true
    grep -E 'timing|Error|Warn' "$dump_dir/restore.log" || true

    log "[$label run=$run] restore=${restore_ms}ms"
    for i in $(seq 1 15); do grep -q "SUCCESS" "$outfile" 2>/dev/null && break; sleep 1; done
    cat "$outfile"
    echo "RESULT label=$label run=$run dump_ms=$dump_ms restore_ms=$restore_ms"

    cleanup_container
}

log "=== Benchmark start: TENSOR_SIZE=$TENSOR_SIZE RUNS=$RUNS ==="
IFS=';' read -ra SCENARIO_LIST <<< "$SCENARIOS"
for sc in "${SCENARIO_LIST[@]}"; do log "scenario: $sc"; done

for sc in "${SCENARIO_LIST[@]}"; do
    IFS='|' read -r label image opts <<< "$sc"
    echo ""
    echo "=== SCENARIO $label: image=$image opts='$opts' ==="
    for r in $(seq 1 $RUNS); do run_plugin "$label" "$image" $r "$opts" || true; echo; done
done

log "=== All benchmarks complete ==="
