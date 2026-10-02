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
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <unistd.h>

static size_t CHUNK = 64UL << 20;  /* per transfer; CUDA_CS_CHUNK_MB overrides */
#define MAXTHR 32               /* I/O worker threads (CUDA_CS_THREADS, default min(ncpu,16)) */
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
static CUresult (*p_cuPointerGetAttribute)(void *, CUpointer_attribute, CUdeviceptr);
static CUresult (*p_cuCtxGetDevice)(CUdevice *);
static CUresult (*p_cuMemHostAlloc)(void **, size_t, unsigned);
static CUresult (*p_cuMemFreeHost)(void *);
static CUresult (*p_cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
static CUresult (*p_cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
static CUresult (*p_cuEventCreate)(CUevent *, unsigned);
static CUresult (*p_cuEventRecord)(CUevent, CUstream);
static CUresult (*p_cuEventSynchronize)(CUevent);
static CUresult (*p_cuStreamSynchronize)(CUstream);
static CUresult (*p_cuStreamCreate)(CUstream *, unsigned);
static CUresult (*p_cuStreamDestroy)(CUstream);
static CUresult (*p_cuCheckpointProcessGetState)(int, CUprocessState *);
static CUresult (*p_cuCheckpointProcessLock)(int, CUcheckpointLockArgs *);
static CUresult (*p_cuCheckpointProcessCheckpoint)(int, CUcheckpointCheckpointArgs *);
static CUresult (*p_cuCheckpointProcessRestore)(int, CUcheckpointRestoreArgs *);
static CUresult (*p_cuCheckpointProcessUnlock)(int, CUcheckpointUnlockArgs *);
static CUresult (*p_cuCheckpointOperationComplete)(CUcheckpointOperationHandle);

/*
 * Driver symbols are ABI-versioned (cuMemcpyDtoHAsync -> cuMemcpyDtoHAsync_v2, ...): dlsym() of the
 * bare name returns the oldest ABI. Resolve through cuGetProcAddress with the header's CUDA version,
 * which returns the ABI the header declares; fall back to dlsym for symbols it does not know.
 */
static CUresult (*p_cuGetProcAddress)(const char *, void **, int, cuuint64_t, CUdriverProcAddressQueryResult *);
static void *resolve(void *h, const char *name)
{
	void *fn = NULL;
	CUdriverProcAddressQueryResult q;
	if (p_cuGetProcAddress && p_cuGetProcAddress(name, &fn, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &q) == CUDA_SUCCESS && fn)
		return fn;
	return dlsym(h, name);
}
#define LOAD(sym) do { p_##sym = resolve(h, #sym); if (!p_##sym) { fprintf(stderr, "libcuda: missing %s\n", #sym); return -1; } } while (0)
static int load_cuda(void)
{
	void *h = dlopen("libcuda.so.1", RTLD_NOW);
	if (!h) { fprintf(stderr, "dlopen libcuda.so.1: %s\n", dlerror()); return -1; }
	p_cuGetProcAddress = dlsym(h, "cuGetProcAddress_v2");
	if (!p_cuGetProcAddress) p_cuGetProcAddress = dlsym(h, "cuGetProcAddress");
	LOAD(cuInit); LOAD(cuGetErrorString); LOAD(cuDeviceGetCount); LOAD(cuDeviceGet);
	LOAD(cuDevicePrimaryCtxRetain); LOAD(cuDevicePrimaryCtxRelease); LOAD(cuCtxSetCurrent);
	/* cuStreamGetCtx: cuGetProcAddress(13040) returns the _v2 (3-arg, green-context) variant; keep the 2-arg one */
	p_cuStreamGetCtx = dlsym(h, "cuStreamGetCtx"); if (!p_cuStreamGetCtx) { fprintf(stderr, "libcuda: missing cuStreamGetCtx\n"); return -1; }
	LOAD(cuPointerGetAttribute); LOAD(cuCtxGetDevice); LOAD(cuMemHostAlloc); LOAD(cuMemFreeHost);
	LOAD(cuMemcpyDtoHAsync); LOAD(cuMemcpyHtoDAsync);
	LOAD(cuEventCreate); LOAD(cuEventRecord); LOAD(cuEventSynchronize); LOAD(cuStreamSynchronize);
	LOAD(cuStreamCreate); LOAD(cuStreamDestroy);
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
 * Move one device's mapped region between the device pointer and the file at file_off with N worker
 * threads.  Each worker owns a CUDA stream in the mapping's context and two pinned 64 MB buffers and
 * pulls chunk indices from a shared counter:
 *   checkpoint: DtoH(chunk) on its stream -> wait -> pwrite        (next chunk's DtoH overlaps the write)
 *   restore:    pread(chunk) -> HtoD on its stream, alternating buffers so the next pread overlaps the HtoD
 * All worker streams are synchronised before returning (required before cuCheckpointOperationComplete).
 */
struct xfer {
	int fd; off_t file_off; CUdeviceptr dptr; size_t size; CUcontext ctx; int dir, direct;
	size_t nchunks; atomic_size_t next; atomic_int err;
	double io_ms[MAXTHR], pcie_ms[MAXTHR], alloc_ms[MAXTHR];
};

static int nthreads(void)
{
	const char *e = getenv("CUDA_CS_THREADS");
	int n = e ? atoi(e) : 0;
	if (n <= 0) n = 4; /* 4 x 64 MB measured best on A10G; more streams degrade the mapping's H2D rate */
	if (n > MAXTHR) n = MAXTHR;
	return n < 1 ? 1 : n;
}

struct warg { struct xfer *x; int id; };
static void *xfer_worker(void *p)
{
	struct warg *w = p; struct xfer *x = w->x; int id = w->id;
	void *buf[2] = {NULL, NULL}; CUevent ev[2]; CUstream st = NULL; int inflight[2] = {0, 0};
	double io = 0, pc = 0, talloc = now_ms();
	CUresult r;

	if ((r = p_cuCtxSetCurrent(x->ctx)) != CUDA_SUCCESS) { fprintf(stderr, "[w%d] cuCtxSetCurrent: %s\n", id, cuerr(r)); goto fail; }
	if ((r = p_cuStreamCreate(&st, CU_STREAM_NON_BLOCKING)) != CUDA_SUCCESS) { fprintf(stderr, "[w%d] cuStreamCreate: %s\n", id, cuerr(r)); goto fail; }
	for (int i = 0; i < 2; i++) {
		if ((r = p_cuMemHostAlloc(&buf[i], CHUNK, CU_MEMHOSTALLOC_PORTABLE)) != CUDA_SUCCESS) { fprintf(stderr, "[w%d] cuMemHostAlloc: %s\n", id, cuerr(r)); goto fail; }
		if ((r = p_cuEventCreate(&ev[i], CU_EVENT_DISABLE_TIMING)) != CUDA_SUCCESS) { fprintf(stderr, "[w%d] cuEventCreate: %s\n", id, cuerr(r)); goto fail; }
	}
	x->alloc_ms[id] = now_ms() - talloc;

	for (int b = 0;; b ^= 1) {
		size_t k = atomic_fetch_add(&x->next, 1);
		if (k >= x->nchunks || atomic_load(&x->err)) break;
		size_t off = k * CHUNK, len = x->size - off < CHUNK ? x->size - off : CHUNK;
		size_t iolen = x->direct ? ((len + 4095) & ~4095UL) : len;
		double t;
		if (inflight[b]) { t = now_ms(); p_cuEventSynchronize(ev[b]); pc += now_ms() - t; inflight[b] = 0; }
		if (x->dir == 0) {
			t = now_ms();
			if ((r = p_cuMemcpyDtoHAsync(buf[b], x->dptr + off, len, st)) != CUDA_SUCCESS) { fprintf(stderr, "[w%d] DtoH: %s\n", id, cuerr(r)); goto fail; }
			p_cuEventRecord(ev[b], st);
			p_cuEventSynchronize(ev[b]);      /* the write needs the data; the other buffer's write overlapped this */
			pc += now_ms() - t;
			t = now_ms();
			if (pwrite(x->fd, buf[b], iolen, x->file_off + off) != (ssize_t)iolen) { perror("pwrite"); goto fail; }
			io += now_ms() - t;
		} else {
			t = now_ms();
			if (pread(x->fd, buf[b], iolen, x->file_off + off) < (ssize_t)len) { perror("pread"); goto fail; }
			io += now_ms() - t;
			t = now_ms();
			if ((r = p_cuMemcpyHtoDAsync(x->dptr + off, buf[b], len, st)) != CUDA_SUCCESS) { fprintf(stderr, "[w%d] HtoD: %s\n", id, cuerr(r)); goto fail; }
			p_cuEventRecord(ev[b], st);
			pc += now_ms() - t;
			inflight[b] = 1;
		}
	}
	{ double t = now_ms(); p_cuStreamSynchronize(st); pc += now_ms() - t; }
	goto out;
fail:
	atomic_store(&x->err, 1);
out:
	x->io_ms[id] = io; x->pcie_ms[id] = pc;
	if (st) { p_cuStreamSynchronize(st); p_cuStreamDestroy(st); }
	for (int i = 0; i < 2; i++) if (buf[i]) p_cuMemFreeHost(buf[i]);
	return NULL;
}

static int xfer_region(int fd, off_t file_off, CUdeviceptr dptr, size_t size, CUstream st, int dir,
		       int direct, double *io_ms, double *pcie_ms)
{
	struct xfer x; memset(&x, 0, sizeof(x));
	CUcontext ctx = NULL;
	CUresult r = p_cuPointerGetAttribute(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, dptr);
	if (r != CUDA_SUCCESS || !ctx)
		CU(p_cuStreamGetCtx(st, &ctx));
	CU(p_cuCtxSetCurrent(ctx));
	x.fd = fd; x.file_off = file_off; x.dptr = dptr; x.size = size; x.ctx = ctx; x.dir = dir; x.direct = direct;
	x.nchunks = (size + CHUNK - 1) / CHUNK;
	atomic_init(&x.next, 0); atomic_init(&x.err, 0);

	int n = nthreads();
	if ((size_t)n > x.nchunks) n = (int)x.nchunks;
	pthread_t th[MAXTHR]; struct warg wa[MAXTHR];
	double t0 = now_ms();
	for (int i = 0; i < n; i++) { wa[i].x = &x; wa[i].id = i; pthread_create(&th[i], NULL, xfer_worker, &wa[i]); }
	for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
	double wall = now_ms() - t0;
	/* the driver synchronises its own stream in OperationComplete; make sure ours are done too */
	CU(p_cuStreamSynchronize(st));
	double io = 0, pc = 0, al = 0;
	for (int i = 0; i < n; i++) { io += x.io_ms[i]; pc += x.pcie_ms[i]; al += x.alloc_ms[i]; }
	fprintf(stderr, "region: %.2f GB in %zu chunks, %d threads, wall %.0f ms (%.1f GB/s; %.1f GB/s excluding %.0f ms pinned alloc); per-thread avg io %.0f ms, pcie %.0f ms\n",
		size / 1e9, x.nchunks, n, wall, size / wall / 1e6, size / (wall - al / n) / 1e6, al / n, io / n, pc / n);
	*io_ms += io / n; *pcie_ms += pc / n;
	return atomic_load(&x.err) ? -1 : 0;
}

static int open_image(const char *path, int write, int *direct)
{
	int flags = write ? (O_WRONLY | O_CREAT | O_TRUNC) : O_RDONLY;
	int fd;
	if (getenv("CUDA_CS_BUFFERED")) { *direct = 0; fd = open(path, flags, 0600); if (fd < 0) perror(path); return fd; }
	fd = open(path, flags | O_DIRECT, 0600);
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
		if (xfer_region(fd, off, d->devPtr, d->size, d->stream, 0, direct, &io_ms, &pcie_ms)) {
			fprintf(stderr, "copy failed; completing the operation anyway to leave the target in a defined state\n");
			p_cuCheckpointOperationComplete(info->handle);
			return -1;
		}
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
		if (xfer_region(fd, off, d->devPtr, d->size, d->stream, 1, direct, &io_ms, &pcie_ms)) {
			fprintf(stderr, "copy failed; completing the operation anyway\n");
			p_cuCheckpointOperationComplete(info->handle);
			return -1;
		}
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
	setvbuf(stdout, NULL, _IONBF, 0); setvbuf(stderr, NULL, _IONBF, 0);
	if (getenv("CUDA_CS_CHUNK_MB")) CHUNK = (size_t)atoi(getenv("CUDA_CS_CHUNK_MB")) << 20;
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
