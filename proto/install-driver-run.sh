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
# purge the packaged driver but keep the container toolkit (its packages also match *nvidia*)
sudo apt-mark hold nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container1 libnvidia-container-tools >/dev/null 2>&1 || true
sudo apt-get purge -y -qq 'nvidia-driver-*' 'nvidia-dkms-*' 'nvidia-kernel-*' 'nvidia-utils-*' 'nvidia-compute-utils-*' 'nvidia-firmware-*' 'libnvidia-*-[0-9]*' 'xserver-xorg-video-nvidia-*' 'nvidia-persistenced' >/dev/null 2>&1 || true
sudo apt-get autoremove -y -qq >/dev/null 2>&1 || true
sudo apt-mark unhold nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container1 libnvidia-container-tools >/dev/null 2>&1 || true
sudo sh "$RUN" --silent --dkms --no-questions --disable-nouveau --no-cc-version-check
sudo modprobe nvidia && sudo modprobe nvidia_uvm
sudo nvidia-smi -pm 1
# the .run installer does not ship the persistenced unit the Ubuntu package had; a stale CDI spec may
# still reference /run/nvidia-persistenced/socket -> regenerate it for the new driver
sudo nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 || true
sudo rm -f /etc/cdi/nvidia.yaml /var/run/cdi/nvidia.yaml
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml >/dev/null 2>&1 || true
sudo systemctl restart docker
nvidia-smi --query-gpu=name,driver_version,persistence_mode --format=csv,noheader
nm -D --defined-only /usr/lib/x86_64-linux-gnu/libcuda.so.1 | grep -c cuCheckpointOperationComplete && echo "custom-storage API present"
# nvidia-container-toolkit keeps working with .run drivers (libnvidia-container discovers the libs)
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L | head -1
