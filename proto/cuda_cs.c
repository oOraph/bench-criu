/*
 * cuda_cs — prototype of CUDA checkpoint/restore with the custom-storage mode
 * (CUDA 13.4 driver API, driver >= R615).
 *
 * Instead of letting the driver stage VRAM into the target's host memory
 * (cuda-checkpoint / CRIU cuda plugin), the driver maps the target's GPU memory
 * into THIS process and we move the bytes ourselves:
 *   checkpoint: devPtr --(cuMemcpyDtoHAsync, pinned ring buffer)--> O_DIRECT file
 *   restore:    O_DIRECT file --(pinned ring buffer, cuMemcpyHtoDAsync)--> devPtr
 * Disk I/O and PCIe transfers are double-buffered so they overlap.
 *
 *   cuda_cs checkpoint <pid> <file>   # lock + checkpoint(custom) + copy + complete  -> CHECKPOINTED
 *   cuda_cs restore    <pid> <file>   # restore(custom) + copy + complete + unlock   -> RUNNING
 *   cuda_cs state      <pid>
 *
 * Build (needs cuda.h >= 13.4 in the include path; libcuda is dlopen'ed):
 *   gcc -O2 -Wall -I<cuda-13.4>/include cuda_cs.c -o cuda_cs -ldl -lpthread
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define CHUNK (64UL << 20)      /* 64 MiB per transfer */
#define NBUF  4                 /* pinned ring buffer depth */
#define HDR_SIZE 4096

/* ---- libcuda via dlopen (so the binary builds/runs without the toolkit) ---- */
static CUresult (*p_cuInit)(unsigned);
static CUresult (*p_cuGetErrorString)(CUresult, const char **);
static CUresult (*p_cuDeviceGetCount)(int *);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
static CUresult (*p_cuDevicePrimaryCtxRelease)(CUdevice);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuStreamGetCtx)(CUstream, CUcontext *);
static CUresult (*p_cuMemHostAlloc)(void **, size_t, unsigned);
static CUresult (*p_cuMemFreeHost)(void *);
static CUresult (*p_cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
static CUresult (*p_cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
static CUresult (*p_cuEventCreate)(CUevent *, unsigned);
static CUresult (*p_cuEventRecord)(CUevent, CUstream);
static CUresult (*p_cuEventSynchronize)(CUevent);
static CUresult (*p_cuStreamSynchronize)(CUstream);
static CUresult (*p_cuCheckpointProcessGetState)(int, CUprocessState *);
static CUresult (*p_cuCheckpointProcessLock)(int, CUcheckpointLockArgs *);
static CUresult (*p_cuCheckpointProcessCheckpoint)(int, CUcheckpointCheckpointArgs *);
static CUresult (*p_cuCheckpointProcessRestore)(int, CUcheckpointRestoreArgs *);
static CUresult (*p_cuCheckpointProcessUnlock)(int, CUcheckpointUnlockArgs *);
static CUresult (*p_cuCheckpointOperationComplete)(CUcheckpointOperationHandle);

#define LOAD(sym) do { p_##sym = dlsym(h, #sym); if (!p_##sym) { fprintf(stderr, "libcuda: missing %s\n", #sym); return -1; } } while (0)
static int load_cuda(void)
{
	void *h = dlopen("libcuda.so.1", RTLD_NOW);
	if (!h) { fprintf(stderr, "dlopen libcuda.so.1: %s\n", dlerror()); return -1; }
	LOAD(cuInit); LOAD(cuGetErrorString); LOAD(cuDeviceGetCount); LOAD(cuDeviceGet);
	LOAD(cuDevicePrimaryCtxRetain); LOAD(cuDevicePrimaryCtxRelease); LOAD(cuCtxSetCurrent);
	LOAD(cuStreamGetCtx); LOAD(cuMemHostAlloc); LOAD(cuMemFreeHost);
	LOAD(cuMemcpyDtoHAsync); LOAD(cuMemcpyHtoDAsync);
	LOAD(cuEventCreate); LOAD(cuEventRecord); LOAD(cuEventSynchronize); LOAD(cuStreamSynchronize);
	LOAD(cuCheckpointProcessGetState); LOAD(cuCheckpointProcessLock);
	LOAD(cuCheckpointProcessCheckpoint); LOAD(cuCheckpointProcessRestore);
	LOAD(cuCheckpointProcessUnlock); LOAD(cuCheckpointOperationComplete);
	return 0;
}

static const char *cuerr(CUresult r) { const char *s = "?"; p_cuGetErrorString(r, &s); return s; }
#define CU(call) do { CUresult _r = (call); if (_r != CUDA_SUCCESS) { fprintf(stderr, "%s failed: %s (%d)\n", #call, cuerr(_r), _r); return -1; } } while (0)
static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }

static const char *state_name(CUprocessState s)
{
	switch (s) {
	case CU_PROCESS_STATE_RUNNING: return "RUNNING";
	case CU_PROCESS_STATE_LOCKED: return "LOCKED";
	case CU_PROCESS_STATE_CHECKPOINTED: return "CHECKPOINTED";
	case CU_PROCESS_STATE_FAILED: return "FAILED";
	case CU_PROCESS_STATE_CHECKPOINTING: return "CHECKPOINTING";
	case CU_PROCESS_STATE_RESTORING: return "RESTORING";
	default: return "unknown";
	}
}

/* Retain the primary context of every device: required by the custom-storage mode. */
static int retain_all_primary_ctx(int *ndev)
{
	CU(p_cuDeviceGetCount(ndev));
	for (int i = 0; i < *ndev; i++) {
		CUdevice d; CUcontext c;
		CU(p_cuDeviceGet(&d, i));
		CU(p_cuDevicePrimaryCtxRetain(&c, d));
	}
	return 0;
}

/* image header: magic, device count, then per device {size} */
struct hdr { uint32_t magic; uint32_t ndev; uint64_t size[32]; };
#define MAGIC 0x43554353 /* "CUCS" */

/*
 * Move one device's mapped region between the device pointer and the file at
 * file_off, chunked through NBUF pinned buffers so that disk I/O of chunk k
 * overlaps the PCIe transfer of chunk k+1.  dir: 0 = checkpoint (D2H + write),
 * 1 = restore (read + H2D).
 */
static int xfer_region(int fd, off_t file_off, CUdeviceptr dptr, size_t size, CUstream st, int dir,
		       int direct, double *io_ms, double *pcie_ms)
{
	void *buf[NBUF]; CUevent ev[NBUF];
	CUcontext ctx;
	CU(p_cuStreamGetCtx(st, &ctx));
	CU(p_cuCtxSetCurrent(ctx));
	for (int i = 0; i < NBUF; i++) {
		CU(p_cuMemHostAlloc(&buf[i], CHUNK, CU_MEMHOSTALLOC_PORTABLE));
		CU(p_cuEventCreate(&ev[i], CU_EVENT_DISABLE_TIMING));
	}
	size_t nchunks = (size + CHUNK - 1) / CHUNK;
	int ret = -1;

	if (dir == 0) {
		/* checkpoint: issue D2H for chunk k, then write chunk k-? as events complete */
		size_t issued = 0, written = 0;
		while (written < nchunks) {
			while (issued < nchunks && issued - written < NBUF) {
				size_t off = issued * CHUNK, len = size - off < CHUNK ? size - off : CHUNK;
				double t = now_ms();
				CU(p_cuMemcpyDtoHAsync(buf[issued % NBUF], dptr + off, len, st));
				CU(p_cuEventRecord(ev[issued % NBUF], st));
				*pcie_ms += now_ms() - t; /* issue cost only; completion overlaps the write below */
				issued++;
			}
			size_t k = written, off = k * CHUNK, len = size - off < CHUNK ? size - off : CHUNK;
			double t = now_ms();
			CU(p_cuEventSynchronize(ev[k % NBUF]));
			*pcie_ms += now_ms() - t;
			t = now_ms();
			size_t wlen = direct ? ((len + 4095) & ~4095UL) : len; /* O_DIRECT needs 4K multiples; buffer is page aligned */
			if (pwrite(fd, buf[k % NBUF], wlen, file_off + off) != (ssize_t)wlen) { perror("pwrite"); goto out; }
			*io_ms += now_ms() - t;
			written++;
		}
	} else {
		/* restore: read chunk k into a free buffer, issue H2D; wait on the event before reusing the buffer */
		size_t issued = 0;
		int inflight[NBUF] = {0};
		while (issued < nchunks) {
			int b = issued % NBUF;
			if (inflight[b]) { double t = now_ms(); CU(p_cuEventSynchronize(ev[b])); *pcie_ms += now_ms() - t; inflight[b] = 0; }
			size_t off = issued * CHUNK, len = size - off < CHUNK ? size - off : CHUNK;
			size_t rlen = direct ? ((len + 4095) & ~4095UL) : len;
			double t = now_ms();
			ssize_t n = pread(fd, buf[b], rlen, file_off + off);
			if (n < (ssize_t)len) { perror("pread"); goto out; }
			*io_ms += now_ms() - t;
			t = now_ms();
			CU(p_cuMemcpyHtoDAsync(dptr + off, buf[b], len, st));
			CU(p_cuEventRecord(ev[b], st));
			*pcie_ms += now_ms() - t;
			inflight[b] = 1;
			issued++;
		}
		double t = now_ms();
		CU(p_cuStreamSynchronize(st));
		*pcie_ms += now_ms() - t;
	}
	ret = 0;
out:
	for (int i = 0; i < NBUF; i++) p_cuMemFreeHost(buf[i]);
	return ret;
}

static int open_image(const char *path, int write, int *direct)
{
	int flags = write ? (O_WRONLY | O_CREAT | O_TRUNC) : O_RDONLY;
	int fd = open(path, flags | O_DIRECT, 0600);
	*direct = 1;
	if (fd < 0 && errno == EINVAL) { fd = open(path, flags, 0600); *direct = 0; }
	if (fd < 0) perror(path);
	return fd;
}

static int do_checkpoint(int pid, const char *path)
{
	int ndev;
	if (retain_all_primary_ctx(&ndev)) return -1;
	double t0 = now_ms();
	CU(p_cuCheckpointProcessLock(pid, NULL));
	double t_lock = now_ms() - t0;

	CUcheckpointCustomStorageInfo *info = NULL;
	CUcheckpointCheckpointArgs args; memset(&args, 0, sizeof(args));
	args.customStorageInfo_out = &info;
	t0 = now_ms();
	CU(p_cuCheckpointProcessCheckpoint(pid, &args));
	double t_ckpt = now_ms() - t0;
	if (!info) { fprintf(stderr, "driver did not return custom storage info (driver too old?)\n"); return -1; }

	struct hdr h; memset(&h, 0, sizeof(h)); h.magic = MAGIC; h.ndev = info->deviceCount;
	uint64_t total = 0;
	for (unsigned i = 0; i < info->deviceCount; i++) { h.size[i] = info->perDeviceData[i].size; total += h.size[i]; }
	printf("custom storage: %u device(s), %.2f GB mapped; lock %.0f ms, checkpoint(map) %.0f ms\n",
	       info->deviceCount, total / 1e9, t_lock, t_ckpt);

	int direct, fd = open_image(path, 1, &direct);
	if (fd < 0) return -1;
	void *hb; if (posix_memalign(&hb, 4096, HDR_SIZE)) return -1;
	memset(hb, 0, HDR_SIZE); memcpy(hb, &h, sizeof(h));
	if (pwrite(fd, hb, HDR_SIZE, 0) != HDR_SIZE) { perror("pwrite hdr"); return -1; }

	double io_ms = 0, pcie_ms = 0; off_t off = HDR_SIZE;
	t0 = now_ms();
	for (unsigned i = 0; i < info->deviceCount; i++) {
		CUcheckpointCustomStoragePerDeviceData *d = &info->perDeviceData[i];
		if (xfer_region(fd, off, d->devPtr, d->size, d->stream, 0, direct, &io_ms, &pcie_ms)) return -1;
		off += (d->size + 4095) & ~4095UL;
	}
	double t_copy = now_ms() - t0;
	if (!direct) fdatasync(fd);
	close(fd);

	t0 = now_ms();
	CU(p_cuCheckpointOperationComplete(info->handle));
	double t_done = now_ms() - t0;
	CUprocessState st; CU(p_cuCheckpointProcessGetState(pid, &st));
	printf("[timing] checkpoint: copy %.0f ms (%.1f GB/s, %s) [disk %.0f ms, pcie-wait %.0f ms], complete %.0f ms -> %s\n",
	       t_copy, total / t_copy / 1e6, direct ? "O_DIRECT" : "buffered", io_ms, pcie_ms, t_done, state_name(st));
	return 0;
}

static int do_restore(int pid, const char *path)
{
	int ndev;
	if (retain_all_primary_ctx(&ndev)) return -1;
	int direct, fd = open_image(path, 0, &direct);
	if (fd < 0) return -1;
	void *hb; if (posix_memalign(&hb, 4096, HDR_SIZE)) return -1;
	if (pread(fd, hb, HDR_SIZE, 0) != HDR_SIZE) { perror("pread hdr"); return -1; }
	struct hdr h; memcpy(&h, hb, sizeof(h));
	if (h.magic != MAGIC) { fprintf(stderr, "bad image magic\n"); return -1; }

	CUcheckpointCustomStorageInfo *info = NULL;
	CUcheckpointRestoreArgs args; memset(&args, 0, sizeof(args));
	args.customStorageInfo_out = &info;
	double t0 = now_ms();
	CU(p_cuCheckpointProcessRestore(pid, &args));
	double t_rst = now_ms() - t0;
	if (!info) { fprintf(stderr, "driver did not return custom storage info\n"); return -1; }
	if (info->deviceCount != h.ndev) { fprintf(stderr, "device count mismatch: image %u, driver %u\n", h.ndev, info->deviceCount); return -1; }

	uint64_t total = 0; double io_ms = 0, pcie_ms = 0; off_t off = HDR_SIZE;
	t0 = now_ms();
	for (unsigned i = 0; i < info->deviceCount; i++) {
		CUcheckpointCustomStoragePerDeviceData *d = &info->perDeviceData[i];
		if (d->size != h.size[i]) { fprintf(stderr, "size mismatch dev %u: image %lu, driver %lu\n", i, (unsigned long)h.size[i], (unsigned long)d->size); return -1; }
		if (xfer_region(fd, off, d->devPtr, d->size, d->stream, 1, direct, &io_ms, &pcie_ms)) return -1;
		off += (d->size + 4095) & ~4095UL; total += d->size;
	}
	double t_copy = now_ms() - t0;
	close(fd);

	t0 = now_ms();
	CU(p_cuCheckpointOperationComplete(info->handle));
	double t_done = now_ms() - t0;
	t0 = now_ms();
	CU(p_cuCheckpointProcessUnlock(pid, NULL));
	double t_unlock = now_ms() - t0;
	CUprocessState st; CU(p_cuCheckpointProcessGetState(pid, &st));
	printf("[timing] restore: restore(map) %.0f ms, copy %.0f ms (%.1f GB/s, %s) [disk %.0f ms, pcie-wait %.0f ms], complete %.0f ms, unlock %.0f ms -> %s\n",
	       t_rst, t_copy, total / t_copy / 1e6, direct ? "O_DIRECT" : "buffered", io_ms, pcie_ms, t_done, t_unlock, state_name(st));
	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: %s checkpoint|restore|state <pid> [file]\n", argv[0]); return 2; }
	if (load_cuda()) return 1;
	CU(p_cuInit(0));
	int pid = atoi(argv[2]);
	if (!strcmp(argv[1], "state")) {
		CUprocessState st; CU(p_cuCheckpointProcessGetState(pid, &st));
		printf("%s\n", state_name(st)); return 0;
	}
	if (argc < 4) { fprintf(stderr, "missing file\n"); return 2; }
	if (!strcmp(argv[1], "checkpoint")) return do_checkpoint(pid, argv[3]) ? 1 : 0;
	if (!strcmp(argv[1], "restore")) return do_restore(pid, argv[3]) ? 1 : 0;
	fprintf(stderr, "unknown action %s\n", argv[1]); return 2;
}
