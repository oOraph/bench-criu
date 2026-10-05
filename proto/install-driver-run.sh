#!/bin/bash
# Replace the Ubuntu-packaged NVIDIA driver with NVIDIA's .run installer (e.g. 615.71.09, which has no
# Ubuntu-archive package yet). On NVSwitch (HGX) boxes the matching Fabric Manager is installed from
# NVIDIA's CUDA apt repo, which ships `nvidia-fabricmanager` for every driver version (610.57.04,
# 615.71.09, ...; Ubuntu's archive only has 595). Use on a disposable VM.
set -euo pipefail
VER=${1:-615.71.09}
RUN=/mnt/nvme/drv/NVIDIA-Linux-x86_64-$VER.run
sudo mkdir -p /mnt/nvme/drv
[ -f "$RUN" ] || sudo curl -fsSL -o "$RUN" "https://download.nvidia.com/XFree86/Linux-x86_64/$VER/NVIDIA-Linux-x86_64-$VER.run"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get install -y -qq dkms build-essential "linux-headers-$(uname -r)" >/dev/null
docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sudo pkill nvidia-persistenced 2>/dev/null || true
sudo nvidia-smi -pm 0 >/dev/null 2>&1 || true   # legacy persistence keeps the module busy
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true
# purge the packaged driver but keep the container toolkit (its packages also match *nvidia*)
sudo apt-mark hold nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container1 libnvidia-container-tools >/dev/null 2>&1 || true
sudo apt-get purge -y -qq 'nvidia-driver-*' 'nvidia-dkms-*' 'nvidia-kernel-*' 'nvidia-utils-*' 'nvidia-compute-utils-*' 'nvidia-firmware-*' 'libnvidia-*-[0-9]*' 'xserver-xorg-video-nvidia-*' 'nvidia-persistenced' >/dev/null 2>&1 || true
sudo apt-get autoremove -y -qq >/dev/null 2>&1 || true
sudo apt-mark unhold nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container1 libnvidia-container-tools >/dev/null 2>&1 || true
# The installer defaults to the OPEN kernel module on Turing+; the checkpoint API returns
# CUDA_ERROR_NOT_INITIALIZED (3) for every target under it (2026-10-02, 615.71.09, A10G). Use proprietary.
sudo sh "$RUN" --silent --dkms --no-questions --disable-nouveau --no-cc-version-check --kernel-module-type=proprietary
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
# NVSwitch systems: CUDA returns error 802 until Fabric Manager (same version as the driver) runs
if lspci -d 10de: | grep -qi bridge; then
    . /etc/os-release; REPO="ubuntu${VERSION_ID//./}"
    if [ ! -f /etc/apt/sources.list.d/cuda-${REPO}-x86_64.list ]; then
        curl -fsSL -o /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-keyring_1.1-1_all.deb" && sudo dpkg -i /tmp/cuda-keyring.deb
        sudo apt-get update -qq
    fi
    FMVER=$(apt-cache madison nvidia-fabricmanager | awk -v v="$VER" '$3 ~ "^"v {print $3}' | sort -V | tail -1)
    [ -n "$FMVER" ] || { echo "no nvidia-fabricmanager $VER in the CUDA repo"; exit 1; }
    sudo apt-get install -y -qq nvidia-fabricmanager="$FMVER"
    sudo systemctl enable --now nvidia-fabricmanager
    for i in $(seq 1 90); do nvidia-smi -q 2>/dev/null | grep -A1 '^ *Fabric' | grep -q 'State *: Completed' && break; sleep 2; done
    nvidia-smi -q | grep -A2 '^ *Fabric' | head -3
fi
# nvidia-container-toolkit keeps working with .run drivers (libnvidia-container discovers the libs)
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L | head -1
