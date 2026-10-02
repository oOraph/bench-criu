#!/bin/bash
# Build cuda_cs against the CUDA 13.4 driver header (fetched from the cudart redistributable; no toolkit needed).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f include/cuda.h ]; then
  url=$(curl -fsSL https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.1.json | python3 -c 'import sys,json; print(json.load(sys.stdin)["cuda_cudart"]["linux-x86_64"]["relative_path"])')
  curl -fsSL -o cudart.tar.xz "https://developer.download.nvidia.com/compute/cuda/redist/$url"
  mkdir -p include && tar -xJf cudart.tar.xz --wildcards --strip-components=2 -C include "*/include/cuda.h"
  rm -f cudart.tar.xz
fi
grep -q cuCheckpointOperationComplete include/cuda.h || { echo "cuda.h lacks the custom-storage API"; exit 1; }
gcc -O2 -Wall -Iinclude cuda_cs.c -o cuda_cs -ldl -lpthread && echo built: $(pwd)/cuda_cs
