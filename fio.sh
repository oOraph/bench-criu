#!/bin/bash
# Storage baseline for the bench array. Two jobs:
#  1. the historical one (32 sync jobs, qd1, 1M) — comparable with the June/Oct results, but it
#     saturates around 2.4 GiB/s by itself (32 MB in flight, ~13 ms latency), so it cannot tell
#     a fast array from a slow one;
#  2. a deep-queue libaio job that shows the real array ceiling.
# Optionally a raw single-drive read (RAW_DEV=/dev/nvme1n1) to get the per-drive cap.
set -euo pipefail

DIR=${BENCH_DIR:-/mnt/nvme}/fio
mkdir -p "$DIR"

echo "=== seqread 1M, 32 sync jobs, qd1 (historical)"
fio --name=seqread-1M --filename="$DIR/test1" --size=32G --bs=1M --rw=read --iodepth=1 --numjobs=32 --time_based --runtime=45 --ramp_time=10 --group_reporting --direct=1 --ioengine=sync --invalidate=1

echo "=== seqread 1M, libaio, 8 jobs x qd32 (array ceiling)"
fio --name=seqread-1M-aio --filename="$DIR/test1" --size=32G --bs=1M --rw=read --iodepth=32 --numjobs=8 --time_based --runtime=30 --ramp_time=5 --group_reporting --direct=1 --ioengine=libaio --invalidate=1

if [ -n "${RAW_DEV:-}" ]; then
    echo "=== raw single drive $RAW_DEV, libaio, 4 jobs x qd32 (read-only)"
    sudo fio --name=raw --filename="$RAW_DEV" --bs=1M --rw=read --ioengine=libaio --iodepth=32 --numjobs=4 --time_based --runtime=20 --ramp_time=5 --group_reporting --direct=1 --readonly
fi
