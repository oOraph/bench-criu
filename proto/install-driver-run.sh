#!/bin/bash
# Replace the Ubuntu-packaged NVIDIA driver with NVIDIA's .run installer (e.g. 615.71.09, which has no apt
# package yet). Non-NVSwitch boxes only (no Fabric Manager exists for 615). Use on a disposable VM.
set -euo pipefail
VER=${1:-615.71.09}
RUN=/mnt/nvme/drv/NVIDIA-Linux-x86_64-$VER.run
sudo mkdir -p /mnt/nvme/drv
[ -f "$RUN" ] || sudo curl -fsSL -o "$RUN" "https://download.nvidia.com/XFree86/Linux-x86_64/$VER/NVIDIA-Linux-x86_64-$VER.run"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get install -y -qq dkms build-essential "linux-headers-$(uname -r)" >/dev/null
docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true
sudo apt-get purge -y -qq '*nvidia*' 'libnvidia*' >/dev/null 2>&1 || true
sudo apt-get autoremove -y -qq >/dev/null 2>&1 || true
sudo sh "$RUN" --silent --dkms --no-questions --disable-nouveau --no-cc-version-check
sudo modprobe nvidia && sudo modprobe nvidia_uvm
sudo nvidia-smi -pm 1
nvidia-smi --query-gpu=name,driver_version,persistence_mode --format=csv,noheader
nm -D --defined-only /usr/lib/x86_64-linux-gnu/libcuda.so.1 | grep -c cuCheckpointOperationComplete && echo "custom-storage API present"
# nvidia-container-toolkit keeps working with .run drivers (libnvidia-container discovers the libs)
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L | head -1
