#!/bin/bash

set -euo pipefail

mkdir -p /mnt/nvme/fio

fio --name=seqread-1M --filename=/mnt/nvme/fio/test1 --size=32G --bs=1M --rw=read --iodepth=1 --numjobs=32 --time_based --runtime=45 --ramp_time=10 --group_reporting --direct=1 --ioengine=sync --invalidate=1
