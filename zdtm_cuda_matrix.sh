#!/bin/bash
# Setup used on the p4de (driver 615.71.09): CRIU source tree at ~/criu-zdtm built with `make` and `make cuda_plugin`;
#   apt: cuda-nvcc-13-4 cuda-cudart-dev-13-4 libcublas-dev-13-4 (NVIDIA CUDA repo), CRIU build deps, libnl-route-3-dev;
#   the CUDA tests are built explicitly: make -C test/zdtm/static cuda00 cuda_streams00 ... (PATH with /usr/local/cuda/bin);
#   /tmp/cs-auto.conf and /tmp/cs-off.conf hold "plugin-option cuda_plugin.custom-storage=auto|off" (CRIU_CONFIG_FILE is
#   read even with --no-default-config, which zdtm.py always passes).
# ZDTM CUDA matrix: 7 tests x custom-storage auto/off. Summary in ~/zdtm-matrix.txt, full output in ~/zdtm-<mode>-<test>.log
cd ~/criu-zdtm/test
OUT=~/zdtm-matrix.txt; : > $OUT
for mode in auto off; do
  for t in cuda00 cuda_streams00 cuda_zerocopy00 cuda_mempool00 cuda_graph00 cuda_cublas00 cuda_multigpu00; do
    sudo rm -rf dump/zdtm/static/$t
    sudo env PATH=/usr/local/cuda/bin:$PATH CRIU_CONFIG_FILE=/tmp/cs-$mode.conf timeout 900 ./zdtm.py run -t zdtm/static/$t --cuda-checkpoint --keep-img always > ~/zdtm-$mode-$t.log 2>&1
    rc=$?
    res=$(grep -oE "Test zdtm/static/$t (PASS|FAIL|SKIP)[^=]*" ~/zdtm-$mode-$t.log | tail -1 | awk '{print $3}')
    d=$(ls -dt dump/zdtm/static/$t/*/1 2>/dev/null | head -1)
    imgs=$(ls $d 2>/dev/null | grep -E "^gpu-(cs|pages)-" | sed 's/-[0-9]*\.img//' | sort -u | tr '\n' ' ')
    cs=$(sudo grep -hoE "custom-storage (checkpoint|restore) copy: [0-9.]+ GB, [0-9]+ threads, [0-9]+ ms[^,]*, O_DIRECT\), zero chunks [0-9/]+" $d/dump.log $d/restore.log 2>/dev/null | tr '\n' ';')
    echo "mode=$mode test=$t result=${res:-?} rc=$rc images=[${imgs}] ${cs}" | tee -a $OUT
  done
done
echo DONE >> $OUT
