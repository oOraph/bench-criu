import torch, os, time

gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
device = torch.device("cuda:0")

size = int(os.environ.get("TENSOR_SIZE", 1000))
torch.manual_seed(42)
x = torch.randn(size, size, device=device)
y = torch.randn(size, device=device)
checksum_x = x.sum().item()
checksum_y = y.sum().item()
gpu_name = torch.cuda.get_device_name(0)

print(f"[PRE]  device: {gpu_name} (CUDA_VISIBLE_DEVICES={gpu_id})", flush=True)
print(f"[PRE]  x={checksum_x:.6f} y={checksum_y:.6f} pid={os.getpid()}", flush=True)
print(f"[PRE]  READY — waiting for /tmp/go to proceed", flush=True)

# Block here until /tmp/go is created. This is where CRIU will dump us.
# After restore, the loop resumes and finds /tmp/go (created by the restore script).
while not os.path.exists("/tmp/go"):
    time.sleep(0.1)
print(f"[POST] RESTORED pid={os.getpid()}", flush=True)

# Post-restore: verify tensors and GPU are intact
t_wake = time.perf_counter()
restored_gpu = torch.cuda.get_device_name(0)
t_name = time.perf_counter()
rx = x.sum().item()
t_sum = time.perf_counter()
ry = y.sum().item()
print(f"[POST] device: {restored_gpu} (+{t_name-t_wake:.3f}s)", flush=True)
print(f"[POST] first sum: +{t_sum-t_wake:.3f}s from wake", flush=True)
print(f"[POST] x={rx:.6f} y={ry:.6f}", flush=True)
print(f"[POST] x_match={abs(rx-checksum_x)<1e-3} y_match={abs(ry-checksum_y)<1e-3}", flush=True)
print(f"[POST] same_gpu_model={restored_gpu==gpu_name}", flush=True)
print(f"[POST] SUCCESS={abs(rx-checksum_x)<1e-3 and abs(ry-checksum_y)<1e-3}", flush=True)
