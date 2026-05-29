#!/bin/bash
# Setup script for criu + cuda-checkpoint benchmark
# Use at your own risk

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

sudo mdadm --create /dev/md0 --level=0 --raid-devices=4  /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1
sudo mkfs.xfs /dev/md0

sudo mkdir /mnt/nvme
sudo mount /dev/md0 /mnt/nvme

echo "=== [1/2] NVIDIA driver ==="
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo apt-get install -y nvidia-driver-590
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

