#!/bin/bash
# Setup script for criu + cuda-checkpoint benchmark
# Use at your own risk

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# Instance-store NVMe = NVMe disks that are not the root/EBS volume and carry no
# partition or filesystem. Stripe them all (RAID-0); with a single disk, use it directly.
mapfile -t NVME_DEVS < <(lsblk -dnpo NAME,TYPE,MOUNTPOINTS,FSTYPE | awk '$2=="disk" && $1 ~ /nvme/ && $3=="" && $4==""' | awk '{print $1}' | while read -r d; do [ -z "$(lsblk -nro NAME "$d" | tail -n +2)" ] && echo "$d"; done)
echo "instance-store NVMe devices: ${NVME_DEVS[*]:-none}"
if [ "${#NVME_DEVS[@]}" -eq 0 ]; then echo "no free NVMe device found, aborting"; exit 1; fi
if [ "${#NVME_DEVS[@]}" -eq 1 ]; then
    DATA_DEV=${NVME_DEVS[0]}
else
    sudo apt-get update && sudo apt-get install -y mdadm
    sudo mdadm --create /dev/md0 --level=0 --raid-devices="${#NVME_DEVS[@]}" "${NVME_DEVS[@]}"
    DATA_DEV=/dev/md0
fi
sudo mkfs.xfs "$DATA_DEV"

sudo mkdir -p /mnt/nvme
sudo mount "$DATA_DEV" /mnt/nvme

echo "=== [1/2] NVIDIA driver ==="
# SKIP_DRIVER=1 keeps a preinstalled driver (e.g. Lambda Labs / HGX nodes whose driver must match the
# vendor's Fabric Manager); only persistence mode is enabled.
if [ "${SKIP_DRIVER:-0}" = 1 ]; then
    echo "SKIP_DRIVER=1: keeping the installed driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    sudo nvidia-smi -pm 1 || true
    nvidia-smi -q | grep -A2 '^ *Fabric' | head -3 || true
fi
sudo apt-get update
[ "${SKIP_DRIVER:-0}" = 1 ] || sudo apt-get install -y ubuntu-drivers-common
# Ubuntu 24.04 updates pocket ships 590.48.01 and 610.57.04 (checked 2026-10-01).
# R610 adds legacy CUDA IPC support to cuda-checkpoint; override with NVIDIA_DRIVER=590.
NVIDIA_DRIVER=${NVIDIA_DRIVER:-610}
# NVSwitch systems (p4d/p4de/p5, HGX boards) need NVIDIA Fabric Manager or CUDA fails with
# error 802 "system not yet initialized". FM must match the driver to the patch level. Ubuntu's
# archive ships it only for the -server flavour (nvidia-driver-595-server + nvidia-fabricmanager-595
# = 595.91.07 on 26.04); NVIDIA's CUDA apt repo ships `nvidia-fabricmanager` for every driver
# version (610.57.04, 615.71.09, ...) — see proto/install-driver-run.sh for that path.
# Neither NVreg_NvLinkDisable=1 nor unbinding the NVSwitch devices (pci-stub) avoids the requirement,
# even after a reboot (tested 2026-10-02 on p4de): FM is mandatory on HGX boards.
if [ "${SKIP_DRIVER:-0}" = 1 ]; then
    NVSWITCH=0
elif lspci -d 10de: | grep -qi bridge; then
    echo "NVSwitch detected: installing nvidia-driver-${NVIDIA_DRIVER}-server + nvidia-fabricmanager-${NVIDIA_DRIVER}"
    sudo apt-get install -y "nvidia-driver-${NVIDIA_DRIVER}-server" "nvidia-fabricmanager-${NVIDIA_DRIVER}"
    NVSWITCH=1
else
    sudo apt-get install -y "nvidia-driver-${NVIDIA_DRIVER}"
    NVSWITCH=0
fi
# The driver package blacklists nouveau but that only applies after a reboot;
# on a fresh VM nouveau already holds the GPUs, so unload it and load nvidia now.
if [ "${SKIP_DRIVER:-0}" != 1 ]; then
if lsmod | grep -q '^nouveau'; then sudo rmmod nouveau; fi
sudo modprobe nvidia && sudo modprobe nvidia_uvm
fi
if [ "$NVSWITCH" = 1 ]; then
    sudo systemctl enable --now nvidia-fabricmanager
    for i in $(seq 1 90); do
        nvidia-smi -q 2>/dev/null | grep -A1 '^ *Fabric' | grep -q 'State *: Completed' && break
        sleep 2
    done
    nvidia-smi -q | grep -A2 '^ *Fabric' | head -3
fi
# Enable persistence mode: keeps driver loaded between processes.
# Critical for restore performance: ~10s without, ~2.5s with.
sudo nvidia-smi -pm 1
nvidia-smi

#echo "=== [2/6] cuda-checkpoint (from NVIDIA/cuda-checkpoint GitHub) ==="
#git clone --depth=1 https://github.com/NVIDIA/cuda-checkpoint.git ~/cuda-checkpoint
#sudo cp ~/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/
#cuda-checkpoint --help

echo "=== [2/2] Docker + nvidia-container-toolkit ==="
# Add Docker's official GPG key:
sudo apt update
sudo apt install -y ca-certificates curl fio
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl start docker
sudo systemctl status docker

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -qq && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo systemctl status docker

echo ""
echo "=== Setup complete ==="

